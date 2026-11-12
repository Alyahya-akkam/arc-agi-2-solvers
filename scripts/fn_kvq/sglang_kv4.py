"""The KV4 pool of the SGLang fork and the two gather shims its guarded edits call.

Imported only inside the patched engine (``configs/P5.5-01.yaml``, ``patches.items``): the
edits of ``kv_cache_configurator.py`` swap the QSA layers' ``MHATokenToKVPool`` for
``KV4MHATokenToKVPool`` when ``ARC26_KV4`` is ``1`` (or ``kv``) or ``v``; the edits of
``qwen_sparse_attn_backend.py`` route the decode gather and the chunked-prefill gather through
``kv_extract_compact`` and ``kv_gather_rows`` whenever a buffer is ``uint8`` (the packed rows);
the edit of ``pool_configurator.py`` prices a token through ``kv4_cell_size`` so the pool grows
by itself. With ``ARC26_KV4`` unset nothing here is imported and the FP8 path runs as before.

Pool layout: ``k_buffer[l]`` and ``v_buffer[l]`` are ``[size + page_size, heads, row_bytes]``
``uint8``, one packed row (nibbles then FP16 scales, ``fn_kvq.layout``) per token slot and KV
head, so every base-class mechanism that moves or measures rows (``move_kv_cache``,
``get_kv_size_bytes``, ``get_cpu_copy``, the PD descriptors) works unchanged. In mode ``v`` the
K buffer keeps the engine's KV dtype (FP8 at ``--kv-cache-dtype fp8_e4m3``) as ``uint8`` bytes
viewed on read, which is what the fork's own FP8 pool stores (``store_dtype`` ``uint8``).
The fork's ``fp4_mx_block16`` recipe keeps scales in separate buffers; one row per token was
chosen instead so the gather reads one contiguous 144-byte row per token and the pool needs no
second buffer list, no scale pointer table and no override of the slot-move or size code.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from fn_kvq import layout, reference
from fn_kvq.triton_kernels import kv4_compact_gather, kv4_gather_rows, kv4_write

try:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc
except ImportError as exc:  # pragma: no cover - the CPU test box stubs the module
    raise ImportError("fn_kvq.sglang_kv4 runs inside the patched SGLang engine") from exc

# Read once per process: the engine's environment at boot.
MODE = layout.mode_from_env()
GROUP = layout.group_from_env()
HADAMARD = layout.hadamard_from_env()


def kv4_cell_size(n: int, head_dim: int, v_head_dim: int, layers: int, kv_size: int) -> int:
    """Bytes per token of the full-attention layers, the fork's ``_compute_cell_size`` term.

    Args:
        n: KV heads per GPU.
        head_dim: K head dimension.
        v_head_dim: V head dimension.
        layers: Effective full-attention layers.
        kv_size: Bytes per element of the engine's KV dtype (1 for FP8).

    Returns:
        ``n * (K row + V row) * layers`` with the packed rows of the current mode.
    """
    return layout.bytes_per_token(
        layers=layers, kv_heads=n, head_dim=head_dim, v_head_dim=v_head_dim, group=GROUP,
        mode=MODE, kv_itemsize=kv_size,
    )


class KV4MHATokenToKVPool(MHATokenToKVPool):
    """``MHATokenToKVPool`` with packed INT4 rows (mode ``kv``) or packed V only (mode ``v``).

    The base class allocates through ``_create_buffers_normal`` from ``_kv_buffer_shapes`` and
    ``store_dtype``; both are overridden here, the dtype to ``uint8`` and the shapes to the
    packed rows, so the buffers, their descriptors and the pointer tables come from the base.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kv4_mode = MODE if MODE != "off" else "kv"
        self.kv4_group = GROUP
        self.kv4_hadamard = HADAMARD
        super().__init__(*args, **kwargs)

    def _create_buffers(self) -> None:
        if self.use_hnd or getattr(self, "kv_cache_layout", "nhd") != "nhd":
            raise ValueError("KV4 supports the NHD layout only")
        self.store_dtype = torch.uint8
        super()._create_buffers()

    def _kv_buffer_shapes(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        rows = self.size + self.page_size
        if self.kv4_mode == "kv":
            k_row = layout.row_bytes(self.head_dim, self.kv4_group)
        else:
            k_row = self.head_dim * self.dtype.itemsize
        v_row = layout.row_bytes(self.v_head_dim, self.kv4_group)
        return (rows, self.head_num, k_row), (rows, self.head_num, v_row)

    def _get_key_buffer(self, layer_id: int) -> Tensor:
        buffer = self.k_buffer[layer_id - self.start_layer]
        if self.kv4_mode == "v":
            return buffer.view(self.dtype)
        return buffer

    def _get_value_buffer(self, layer_id: int) -> Tensor:
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: Any,
        loc_info: Any,
        cache_k: Tensor,
        cache_v: Tensor,
        k_scale: float | None = None,
        v_scale: float | None = None,
        layer_id_override: int | None = None,
        dcp_kv_mask: Tensor | None = None,
    ) -> None:
        """Quantize and scatter one layer's new K and V rows (the fork's signature).

        Args:
            layer: The ``RadixAttention`` layer (its ``layer_id`` when no override).
            loc_info: The physical slots, a tensor or a ``KVWriteLoc``.
            cache_k: Tensor  # [tokens, heads, head_dim].
            cache_v: Tensor  # [tokens, heads, v_head_dim].
            k_scale: The FP8 scale of K (applied in mode ``v`` as the base class does).
            v_scale: Unused: the packed rows carry their own scales.
            layer_id_override: The pool-local layer id the hybrid pool passes.
            dcp_kv_mask: Unsupported (decode context parallelism).

        Raises:
            RuntimeError: On ``dcp_kv_mask``.
        """
        if dcp_kv_mask is not None:
            raise RuntimeError("dcp_kv_mask is not supported by the KV4 pool")
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        local = layer_id - self.start_layer
        limit = self.size + self.page_size
        if self.kv4_mode == "kv":
            kv4_write(self.k_buffer[local], loc, cache_k, self.kv4_group, limit, self.kv4_hadamard)
        else:
            k = cache_k
            if k.dtype != self.dtype:
                if k_scale is not None:
                    k = k / k_scale
                k = k.to(self.dtype)
            # The fork's own naive write: the bytes of the KV dtype into the uint8 buffer.
            self.k_buffer[local][loc] = k.view(torch.uint8)
        kv4_write(self.v_buffer[local], loc, cache_v, self.kv4_group, limit, self.kv4_hadamard)


def kv_extract_compact(
    k: Tensor,
    v: Tensor,
    req_to_token: Tensor,
    req_indices: Tensor,
    indices: Tensor,
    seq_lens: Tensor,
    cu_k: Tensor,
    out_k: Tensor,
    out_v: Tensor,
    batch: int,
    topk: int,
) -> None:
    """``qwen_sparse_kv_extraction_compact_triton`` for the KV4 pool (the decode gather).

    Args:
        k: Tensor  # the pool's K buffer (packed uint8, or float8 in mode ``v``).
        v: Tensor  # the pool's V buffer (packed uint8).
        req_to_token: Tensor  # [requests, max_context] int32.
        req_indices: Tensor  # [batch] int32.
        indices: Tensor  # [batch, topk] int32 logical positions.
        seq_lens: Tensor  # [batch] int32.
        cu_k: Tensor  # [batch + 1] int32.
        out_k: Tensor  # [capacity, heads, head_dim] the BF16 scratch.
        out_v: Tensor  # [capacity, heads, head_dim].
        batch: Rows.
        topk: Columns.
    """
    kv4_compact_gather(
        k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
        out_k.shape[2], GROUP, HADAMARD,
    )


def kv_gather_rows(
    k: Tensor, v: Tensor, index: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """The chunked-prefill gather of a request's full context from the KV4 pool.

    Replaces ``k_buffer.index_select(0, gather_index), v_buffer.index_select(0, gather_index)``
    in the backend's ``forward_extend``: packed buffers are dequantized to ``dtype`` (the
    query's), an FP8 K (mode ``v``) is gathered as the fork does and cast by the kernel.

    Args:
        k: Tensor  # the pool's K buffer.
        v: Tensor  # the pool's V buffer (packed).
        index: Tensor  # [n] int64 physical slots.
        dtype: The output dtype.

    Returns:
        ``(K, V)``  # each [n, heads, head_dim].
    """
    head_dim = layout.head_dim_from_row(v.shape[2], GROUP)
    if reference.is_packed(k, head_dim):
        k_rows = kv4_gather_rows(k, index, head_dim, GROUP, dtype, HADAMARD)
    else:
        k_rows = k.index_select(0, index)
    return k_rows, kv4_gather_rows(v, index, head_dim, GROUP, dtype, HADAMARD)


__all__ = [
    "GROUP",
    "HADAMARD",
    "MODE",
    "KV4MHATokenToKVPool",
    "kv4_cell_size",
    "kv_extract_compact",
    "kv_gather_rows",
]

if os.environ.get("ARC26_KV4_LOG", "1") != "0":  # pragma: no cover - the engine's log line
    import logging

    logging.getLogger(__name__).info(
        "arc26 KV4 cache: mode=%s group=%d hadamard=%s", MODE, GROUP, HADAMARD
    )
