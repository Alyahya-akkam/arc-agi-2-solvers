r"""Pure-PyTorch reference of the KV4 cache: rotate, quantize, dequantize, paged write, gathers.

Every function runs on CPU or GPU with plain tensor ops; the Triton kernels of
``fn_kvq.triton_kernels`` are tested against these. The math is the one of ``fn_kvq.layout``:
per group of $G$ channels, $s = \\mathrm{fp16}(\\max |x| / 7)$, $q = \\mathrm{clip}(\\mathrm{round}
(x / s), -8, 7)$, nibbles $u = q + 8$ packed low then high, FP16 scales after the nibbles.

Rotation (``hadamard=True``, the default): before quantization each block of 64 channels is
multiplied by the normalized Sylvester Hadamard matrix $H_{64} / 8$ ($H^2 = 64 I$, so the
normalized matrix is symmetric and its own inverse), and the gather multiplies the dequantized
block by it again. The attention products are unchanged, $q^\\top k = (Hq)^\\top (Hk)$, and the
rotation spreads a per-channel outlier over 64 channels so a group's absmax no longer wastes the
other channels' levels (llama.cpp PR 21038; arXiv 2504.04823). The byte layout is the same with
or without it; a pool written with one setting must be read with the same one.

The paged pool is a tensor ``[slots, heads, row_bytes]`` of ``uint8`` per layer, indexed by the
fork's physical token slot (``req_to_token[req, position]``); page 64 only means slots of one
page are contiguous. Slot 0's page is the fork's padding target, written and never read.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fn_kvq.layout import (
    DEFAULT_GROUP,
    HADAMARD_BLOCK,
    NIBBLE_OFFSET,
    QLEVELS,
    QMAX,
    QMIN,
    row_bytes,
)

_HADAMARD_CACHE: dict[tuple[torch.device, torch.dtype], Tensor] = {}


def hadamard_matrix(
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> Tensor:
    r"""The normalized 64-point Sylvester Hadamard matrix.

    $H_{ij} = (-1)^{\\mathrm{popcount}(i \\wedge j)} / 8$, built once per device and dtype.

    Args:
        device: Where to build it.
        dtype: Its dtype (float32 for the exact rotation).

    Returns:
        Tensor  # [64, 64], symmetric, orthogonal.
    """
    key = (torch.device(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        i = torch.arange(HADAMARD_BLOCK, device=device)
        parity = torch.zeros((HADAMARD_BLOCK, HADAMARD_BLOCK), dtype=torch.int64, device=device)
        both = i[:, None] & i[None, :]
        for bit in range(HADAMARD_BLOCK.bit_length() - 1):
            parity += (both >> bit) & 1
        sign = 1.0 - 2.0 * (parity % 2).to(torch.float32)
        cached = (sign / (HADAMARD_BLOCK**0.5)).to(dtype)
        _HADAMARD_CACHE[key] = cached
    return cached


def rotate(x: Tensor) -> Tensor:
    """Apply the block Hadamard rotation to the last dimension (its own inverse).

    Args:
        x: Tensor  # [..., head_dim] with head_dim a multiple of 64; computed in float32.

    Returns:
        Tensor  # [..., head_dim] float32.
    """
    head_dim = x.shape[-1]
    if head_dim % HADAMARD_BLOCK:
        raise ValueError(f"head_dim {head_dim} is not a multiple of {HADAMARD_BLOCK}")
    blocks = x.to(torch.float32).reshape(*x.shape[:-1], head_dim // HADAMARD_BLOCK, HADAMARD_BLOCK)
    return (blocks @ hadamard_matrix(x.device)).reshape(*x.shape[:-1], head_dim)


def quantize_rows(x: Tensor, group: int = DEFAULT_GROUP, hadamard: bool = True) -> Tensor:
    """Quantize rows of ``head_dim`` values into packed KV4 rows.

    Args:
        x: Tensor  # [..., head_dim], any float dtype; computed in float32.
        group: The group size $G$.
        hadamard: Rotate each 64-channel block before quantizing.

    Returns:
        Tensor  # [..., row_bytes(head_dim, group)] uint8, nibbles then FP16 scales.
    """
    head_dim = x.shape[-1]
    half = group // 2
    groups = head_dim // group
    x32 = rotate(x) if hadamard else x.to(torch.float32)
    x32 = x32.reshape(*x.shape[:-1], groups, group)
    amax = x32.abs().amax(dim=-1)  # [..., groups]
    scale16 = (amax / QLEVELS).to(torch.float16)
    scale32 = scale16.to(torch.float32)
    inv = torch.where(scale32 > 0, 1.0 / scale32, torch.zeros_like(scale32))
    q = torch.round(x32 * inv.unsqueeze(-1)).clamp_(QMIN, QMAX)  # half to even, as the kernel
    u = (q + NIBBLE_OFFSET).to(torch.uint8).reshape(*x.shape[:-1], groups, half, 2)
    packed = u[..., 0] | (u[..., 1] << 4)  # [..., groups, half]
    packed = packed.reshape(*x.shape[:-1], head_dim // 2)
    scale_bytes = scale16.contiguous().view(torch.uint8).reshape(*x.shape[:-1], 2 * groups)
    return torch.cat([packed, scale_bytes], dim=-1)


def dequantize_rows(
    rows: Tensor,
    head_dim: int,
    group: int = DEFAULT_GROUP,
    dtype: torch.dtype = torch.float32,
    hadamard: bool = True,
) -> Tensor:
    """Dequantize packed KV4 rows (and undo the rotation).

    Args:
        rows: Tensor  # [..., row_bytes(head_dim, group)] uint8.
        head_dim: $D$.
        group: $G$.
        dtype: The output dtype (the attention scratch's, BF16 in the engine).
        hadamard: The setting the rows were written with.

    Returns:
        Tensor  # [..., head_dim] of ``dtype``, $H (s_g q)$ or $s_g q$.
    """
    if rows.shape[-1] != row_bytes(head_dim, group):
        raise ValueError(f"row has {rows.shape[-1]} bytes, expected {row_bytes(head_dim, group)}")
    half = group // 2
    groups = head_dim // group
    packed = rows[..., : head_dim // 2].to(torch.int32)
    lo = (packed & 0xF) - NIBBLE_OFFSET
    hi = (packed >> 4) - NIBBLE_OFFSET
    q = torch.stack([lo, hi], dim=-1).reshape(*rows.shape[:-1], groups, half * 2)
    scales = rows[..., head_dim // 2 :].contiguous().view(torch.float16).to(torch.float32)
    out = (q.to(torch.float32) * scales.unsqueeze(-1)).reshape(*rows.shape[:-1], head_dim)
    if hadamard:
        out = rotate(out)
    return out.to(dtype)


def write_rows(
    buffer: Tensor, loc: Tensor, x: Tensor, group: int = DEFAULT_GROUP, hadamard: bool = True
) -> None:
    """Quantize ``x`` and scatter it into the paged buffer at slots ``loc`` (the write path).

    Args:
        buffer: Tensor  # [slots, heads, row_bytes] uint8, one layer's K or V pool.
        loc: Tensor  # [tokens] int; a slot outside [0, slots) is skipped (the fork's padding).
        x: Tensor  # [tokens, heads, head_dim].
        group: $G$.
        hadamard: Rotate before quantizing.
    """
    loc = loc.long()
    keep = (loc >= 0) & (loc < buffer.shape[0])
    rows = quantize_rows(x, group, hadamard)
    buffer[loc[keep]] = rows[keep].to(buffer.dtype)


def gather_rows(
    buffer: Tensor,
    slots: Tensor,
    head_dim: int,
    group: int = DEFAULT_GROUP,
    dtype: torch.dtype = torch.bfloat16,
    hadamard: bool = True,
) -> Tensor:
    """Dequantize the rows at ``slots`` (chunked prefill's full-context gather).

    Args:
        buffer: Tensor  # [slots, heads, row_bytes] uint8.
        slots: Tensor  # [n] int physical slots.
        head_dim: $D$.
        group: $G$.
        dtype: The output dtype.
        hadamard: The setting the rows were written with.

    Returns:
        Tensor  # [n, heads, head_dim].
    """
    return dequantize_rows(buffer.index_select(0, slots.long()), head_dim, group, dtype, hadamard)


def is_packed(buffer: Tensor, head_dim: int) -> bool:
    """Whether a pool buffer is a KV4 one (uint8 rows narrower than the head dimension).

    The FP8 pool's rows are ``head_dim`` bytes viewed as ``float8``; the BF16 fallback's are
    ``bfloat16``; only the KV4 pool hands the backend ``uint8`` rows, and they are always shorter
    than ``head_dim`` (``row_bytes`` is $D/2 + 2D/G < D$ for $G > 4$).
    """
    return buffer.dtype == torch.uint8 and buffer.shape[-1] < head_dim


def compact_gather(
    k_buffer: Tensor,
    v_buffer: Tensor,
    req_to_token: Tensor,
    req_indices: Tensor,
    indices: Tensor,
    seq_lens: Tensor,
    cu_k: Tensor,
    out_k: Tensor,
    out_v: Tensor,
    head_dim: int,
    group: int = DEFAULT_GROUP,
    hadamard: bool = True,
) -> None:
    """The decode gather into the packed FA2 scratch, the fork's ``_compact_kv`` with dequant.

    Row ``b`` of ``indices`` holds the logical positions the indexer selected for request
    ``req_indices[b]``; column ``c`` is valid when ``c < cu_k[b + 1] - cu_k[b]`` and
    ``0 <= indices[b, c] < seq_lens[b]`` (the fork's expand kernel packs the valid entries first).
    Valid entries are written to scratch rows ``cu_k[b] + c``; the other scratch rows are left as
    they are, as the fork's kernel leaves them (FlashAttention reads only ``cu_k`` ranges).
    A buffer that is not packed (the FP8 K of mode ``v``) is copied with a plain cast.

    Args:
        k_buffer: Tensor  # [slots, heads, row_bytes] uint8, or [slots, heads, head_dim] float8.
        v_buffer: Tensor  # [slots, heads, row_bytes] uint8.
        req_to_token: Tensor  # [requests, max_context] int, position to physical slot.
        req_indices: Tensor  # [batch] int request rows.
        indices: Tensor  # [batch, topk] int32 logical positions, -1 for none.
        seq_lens: Tensor  # [batch] int32.
        cu_k: Tensor  # [batch + 1] int32 cumulative valid counts.
        out_k: Tensor  # [capacity, heads, head_dim], the BF16 scratch.
        out_v: Tensor  # [capacity, heads, head_dim].
        head_dim: $D$.
        group: $G$.
        hadamard: The setting the rows were written with.
    """
    batch, topk = indices.shape
    for b in range(batch):
        start = int(cu_k[b])
        valid_count = int(cu_k[b + 1]) - start
        positions = indices[b].long()
        cols = torch.arange(topk, device=indices.device)
        valid = (cols < valid_count) & (positions >= 0) & (positions < int(seq_lens[b]))
        if not bool(valid.any()):
            continue
        slots = req_to_token[int(req_indices[b]), positions[valid]].long()
        rows = start + cols[valid]
        for buffer, out in ((k_buffer, out_k), (v_buffer, out_v)):
            if is_packed(buffer, head_dim):
                out[rows] = gather_rows(buffer, slots, head_dim, group, out.dtype, hadamard)
            else:
                out[rows] = buffer.index_select(0, slots).to(out.dtype)


def nibble_stats(
    rows: Tensor, ref_rows: Tensor, head_dim: int, group: int = DEFAULT_GROUP
) -> dict[str, float]:
    r"""Compare two packings of the same data nibble by nibble (the kernel's against the reference).

    A kernel that rounds a quotient or sums a rotation in another order than torch lands a rare
    element one quantization step away; that is benign. A layout bug moves many nibbles or moves
    them by more than one step. Both are told apart here: the share of nibbles that differ, the
    largest nibble difference, the share of scales that differ, and the largest dequantized
    difference in steps of the reference's scale $s_g$, $\max |x - x_{\mathrm{ref}}| / s_g$.

    Args:
        rows: Tensor  # [..., row_bytes] uint8, the kernel's rows.
        ref_rows: Tensor  # [..., row_bytes] uint8, the reference's rows of the same data.
        head_dim: $D$.
        group: $G$.

    Returns:
        ``{"nibble_diff_share", "nibble_diff_max", "scale_diff_share", "dequant_diff_max_steps"}``.
    """
    if rows.shape != ref_rows.shape or rows.shape[-1] != row_bytes(head_dim, group):
        raise ValueError(f"rows {tuple(rows.shape)} and {tuple(ref_rows.shape)} must be equal rows")
    half = head_dim // 2

    def nibbles(packed: Tensor) -> Tensor:
        p = packed[..., :half].to(torch.int32)
        return torch.stack([p & 0xF, p >> 4], dim=-1).reshape(*packed.shape[:-1], head_dim)

    diff = (nibbles(rows) - nibbles(ref_rows)).abs()
    ref_scales = ref_rows[..., half:].contiguous().view(torch.float16).to(torch.float32)
    scales = rows[..., half:].contiguous().view(torch.float16).to(torch.float32)
    x = dequantize_rows(rows, head_dim, group, torch.float32, hadamard=False)
    x_ref = dequantize_rows(ref_rows, head_dim, group, torch.float32, hadamard=False)
    step = ref_scales.unsqueeze(-1).expand(*ref_scales.shape, group)
    step = step.reshape(*rows.shape[:-1], head_dim)
    steps = torch.where(step > 0, (x - x_ref).abs() / step, (x - x_ref).abs())
    return {
        "nibble_diff_share": (diff > 0).float().mean().item(),
        "nibble_diff_max": int(diff.max().item()),
        "scale_diff_share": (scales != ref_scales).float().mean().item(),
        "dequant_diff_max_steps": steps.max().item(),
    }


def fp8_roundtrip(x: Tensor) -> Tensor:
    """The FP8 E4M3 path's own round trip, for the error comparison of the benchmark.

    Args:
        x: Tensor  # any shape, float.

    Returns:
        Tensor  # ``x`` cast to ``float8_e4m3fn`` and back to float32.
    """
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def error_stats(x: Tensor, y: Tensor) -> dict[str, float]:
    r"""Max and mean absolute error and the relative RMS error of ``y`` against ``x``.

    Args:
        x: Tensor  # the reference values.
        y: Tensor  # the reconstruction, same shape.

    Returns:
        ``{"max_abs", "mean_abs", "rel_rms"}`` as floats; ``rel_rms`` is
        $\\|y - x\\|_2 / \\|x\\|_2$.
    """
    diff = (y.to(torch.float32) - x.to(torch.float32)).abs()
    norm = x.to(torch.float32).pow(2).sum().sqrt()
    rel = (diff.pow(2).sum().sqrt() / norm).item() if norm > 0 else 0.0
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item(), "rel_rms": rel}
