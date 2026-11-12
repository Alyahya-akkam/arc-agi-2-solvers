"""Triton kernels of the KV4 cache for sm89 (Ada: L4, L40S) and their dispatchers.

Three kernels, all plain loads and stores plus one 64 x 64 FP32 ``tl.dot`` per block for the
Hadamard rotation (no tensor-core tiling tricks: the traffic per decode step is a few
megabytes, so the kernels are launch-bound, and the simplest code that matches
``fn_kvq.reference`` wins):

- ``kv4_write``: rotate, quantize and pack ``[tokens, heads, D]`` rows and scatter them to the
  slots ``loc`` of one layer's packed pool, ``BLOCK_T`` tokens of one head per program.
- ``kv4_gather_rows``: dequantize (and un-rotate) the rows at given slots into a
  ``[n, heads, D]`` tensor (chunked prefill's full-context gather).
- ``kv4_compact_gather``: the fork's ``_compact_kv`` decode gather with dequantization into the
  FA2 scratch, same grid and same validity rule, so the CUDA graph sees one kernel in place of
  the other. In mode ``v`` the K buffer is FP8 and is decoded in-kernel from its bytes.

Every kernel works in blocks of 64 channels: a block is one Hadamard rotation, ``64 / G``
quantization groups and 32 packed bytes. Without the rotation the write is bit-identical to the
reference (IEEE division, round half to even); with it the 64-term ``tl.dot`` sums in another
order than torch's matmul, so a rare element at a rounding boundary lands one quantization step
away, which ``fn_kvq.reference.nibble_stats`` measures and the benchmark bounds.

Every dispatcher takes the same arguments as its
reference and calls the reference when the tensors are not on CUDA (CPU tests, the engine's CPU
path), unless ``TRITON_INTERPRET=1`` is set, in which case the kernels run on CPU tensors
through Triton's interpreter, which is how they were checked before any GPU time.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from fn_kvq import reference
from fn_kvq.layout import DEFAULT_GROUP, HADAMARD_BLOCK, row_bytes

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the CPU test box without triton
    triton = None
    tl = None


def _use_triton(*tensors: Tensor) -> bool:
    """Run the Triton kernel: on CUDA tensors, or under the interpreter on any tensor."""
    if triton is None:
        return False
    if all(t.is_cuda for t in tensors):
        return True
    return os.environ.get("TRITON_INTERPRET") == "1"


if triton is not None:

    @triton.jit
    def _round_half_even(x):
        """``torch.round``: round half to even, so the kernel matches the reference bit for bit."""
        r = tl.floor(x + 0.5)
        tie = (x + 0.5) == r
        odd = (r - 2.0 * tl.floor(r * 0.5)) != 0.0
        return tl.where(tie & odd, r - 1.0, r)

    @triton.jit
    def _hadamard_tile():
        r"""$H_{ij} = (-1)^{\\mathrm{popcount}(i \\wedge j)} / 8$ for the 64-point block, FP32."""
        i = tl.arange(0, 64)[:, None]
        j = tl.arange(0, 64)[None, :]
        a = i & j
        parity = (
            (a & 1) + ((a >> 1) & 1) + ((a >> 2) & 1) + ((a >> 3) & 1) + ((a >> 4) & 1)
            + ((a >> 5) & 1)
        )
        return tl.where((parity & 1) == 1, -0.125, 0.125)

    @triton.jit
    def _kv4_write_kernel(
        x_ptr,
        out_ptr,
        loc_ptr,
        n_tokens,
        size_limit,
        stride_xt,
        stride_xh,
        stride_ot,
        stride_oh,
        D: tl.constexpr,
        G: tl.constexpr,
        HADAMARD: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Quantize BLOCK_T (token, head) rows of D values and store their packed bytes."""
        h = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        in_range = rows < n_tokens
        loc = tl.load(loc_ptr + rows, mask=in_range, other=-1).to(tl.int64)
        # Slots at or past the limit are the fork's padding target: skipped, as its writer does.
        valid = in_range & (loc >= 0) & (loc < size_limit)
        GB: tl.constexpr = 64 // G
        src_row = x_ptr + rows.to(tl.int64)[:, None] * stride_xt + h * stride_xh
        dst_row = out_ptr + loc[:, None] * stride_ot + h * stride_oh
        c = tl.arange(0, 64)[None, :]
        j = tl.arange(0, 32)[None, :]
        gidx = tl.arange(0, GB)[None, :]
        cmask = valid[:, None] & (c >= 0)
        jmask = valid[:, None] & (j >= 0)
        gmask = valid[:, None] & (gidx >= 0)
        if HADAMARD:
            had = _hadamard_tile()
        for b in tl.static_range(D // 64):
            x = tl.load(src_row + b * 64 + c, mask=cmask, other=0.0).to(tl.float32)
            if HADAMARD:
                x = tl.dot(x, had, input_precision="ieee")
            xg = tl.reshape(x, [BLOCK_T, GB, G])
            amax = tl.max(tl.abs(xg), axis=2)  # [BLOCK_T, GB]
            # IEEE division (div.rn), not Triton's default div.full (2 ulp): the reference divides
            # in torch, and a 1-ulp quotient difference lands a value at a rounding boundary one
            # nibble away (P5.5-01 phase K on the L40S: write_rows_equal false, the rest exact).
            s16 = tl.fdiv(amax, 7.0, ieee_rounding=True).to(tl.float16)
            s32 = s16.to(tl.float32)
            safe = tl.where(s32 > 0, s32, 1.0)
            inv = tl.where(s32 > 0, tl.fdiv(safe * 0.0 + 1.0, safe, ieee_rounding=True), 0.0)
            q = _round_half_even(xg * inv[:, :, None])
            q = tl.minimum(tl.maximum(q, -8.0), 7.0)
            qlo, qhi = tl.split(tl.reshape(q, [BLOCK_T, 32, 2]))
            packed = ((qlo + 8.0).to(tl.int32) | ((qhi + 8.0).to(tl.int32) << 4)).to(tl.uint8)
            tl.store(dst_row + b * 32 + j, packed, mask=jmask)
            bits = s16.to(tl.uint16, bitcast=True).to(tl.int32)
            sptr = dst_row + D // 2 + 2 * (b * GB + gidx)
            tl.store(sptr, (bits & 0xFF).to(tl.uint8), mask=gmask)
            tl.store(sptr + 1, ((bits >> 8) & 0xFF).to(tl.uint8), mask=gmask)

    @triton.jit
    def _kv4_dequant_block(
        buf_ptr, row_base, valid, had, b, D: tl.constexpr, G: tl.constexpr, HADAMARD: tl.constexpr
    ):
        """Dequantize block ``b`` (64 channels) of [T] packed rows into [T, 64] FP32."""
        HALF: tl.constexpr = G // 2
        GB: tl.constexpr = 64 // G
        j = tl.arange(0, 32)[None, :]
        g = b * GB + j // HALF
        m = valid[:, None] & (j >= 0)
        rows = row_base[:, None]
        packed = tl.load(buf_ptr + rows + b * 32 + j, mask=m, other=0).to(tl.int32)
        b0 = tl.load(buf_ptr + rows + D // 2 + 2 * g, mask=m, other=0).to(tl.int32)
        b1 = tl.load(buf_ptr + rows + D // 2 + 2 * g + 1, mask=m, other=0).to(tl.int32)
        s = (b0 | (b1 << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        lo = ((packed & 0xF) - 8).to(tl.float32) * s
        hi = ((packed >> 4) - 8).to(tl.float32) * s
        x = tl.interleave(lo, hi)
        if HADAMARD:
            x = tl.dot(x, had, input_precision="ieee")
        return x

    @triton.jit
    def _fp8_block(buf_ptr, row_base, valid, b):
        """Decode block ``b`` of [T] FP8 E4M3 rows (bytes) into [T, 64] FP32 (mode ``v``'s K)."""
        c = tl.arange(0, 64)[None, :]
        m = valid[:, None] & (c >= 0)
        byte = tl.load(buf_ptr + row_base[:, None] + b * 64 + c, mask=m, other=0).to(tl.int32)
        sign = byte >> 7
        exp = (byte >> 3) & 0xF
        man = (byte & 0x7).to(tl.float32) * 0.125
        normal = (1.0 + man) * tl.exp2((exp - 7).to(tl.float32))
        subnormal = man * tl.exp2(tl.full([1, 1], -6.0, tl.float32))
        mag = tl.where(exp > 0, normal, subnormal)
        return tl.where(sign != 0, -mag, mag)

    @triton.jit
    def _kv4_gather_rows_kernel(
        buf_ptr,
        slots_ptr,
        out_ptr,
        n,
        stride_bt,
        stride_bh,
        stride_ot,
        stride_oh,
        D: tl.constexpr,
        G: tl.constexpr,
        HADAMARD: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Dequantize BLOCK_T rows of one head at the given slots into out[row, head, :]."""
        h = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        valid = rows < n
        slots = tl.load(slots_ptr + rows, mask=valid, other=0).to(tl.int64)
        row_base = slots * stride_bt + h * stride_bh
        c = tl.arange(0, 64)[None, :]
        dst = out_ptr + rows.to(tl.int64)[:, None] * stride_ot + h * stride_oh
        mask = valid[:, None] & (c >= 0)
        had = _hadamard_tile()
        for b in tl.static_range(D // 64):
            x = _kv4_dequant_block(buf_ptr, row_base, valid, had, b, D, G, HADAMARD)
            tl.store(dst + b * 64 + c, x.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _kv4_compact_gather_kernel(
        k_ptr,
        v_ptr,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        req_stride,
        idx_stride,
        stride_kt,
        stride_kh,
        stride_vt,
        stride_vh,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        G: tl.constexpr,
        HADAMARD: tl.constexpr,
        K_PACKED: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
    ):
        """The fork's ``_compact_kv`` (qsa/sparse_attn.py) reading packed rows."""
        batch = tl.program_id(0)
        head = tl.program_id(1)
        block = tl.program_id(2)
        cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
        length = tl.load(seq_lens + batch)
        req = tl.load(req_indices + batch).to(tl.int64)
        pack_start = tl.load(cu_k + batch).to(tl.int64)
        valid_count = tl.load(cu_k + batch + 1).to(tl.int64) - pack_start
        positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
        valid = (cols < valid_count) & (positions >= 0) & (positions < length)
        slots = tl.load(
            req_to_token + req * req_stride + tl.where(valid, positions, 0),
            mask=valid,
            other=0,
        ).to(tl.int64)
        k_base = slots * stride_kt + head * stride_kh
        v_base = slots * stride_vt + head * stride_vh
        c = tl.arange(0, 64)[None, :]
        dst = (pack_start + cols.to(tl.int64))[:, None] * HEADS * D + head * D
        mask = valid[:, None] & (c >= 0)
        had = _hadamard_tile()
        for b in tl.static_range(D // 64):
            if K_PACKED:
                k_values = _kv4_dequant_block(k_ptr, k_base, valid, had, b, D, G, HADAMARD)
            else:
                k_values = _fp8_block(k_ptr, k_base, valid, b)
            tl.store(out_k + dst + b * 64 + c, k_values.to(out_k.dtype.element_ty), mask=mask)
            v_values = _kv4_dequant_block(v_ptr, v_base, valid, had, b, D, G, HADAMARD)
            tl.store(out_v + dst + b * 64 + c, v_values.to(out_v.dtype.element_ty), mask=mask)

    @triton.jit
    def _compact_kv_fp8(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk: tl.constexpr,
        heads: tl.constexpr,
        dim: tl.constexpr,
        req_stride: tl.constexpr,
        idx_stride: tl.constexpr,
        BLOCK_TOPK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """The fork's ``_compact_kv`` (qsa/sparse_attn.py:369-406 at d00d88e, Apache 2.0).

        The FP8 path's gather, kept here as the benchmark's baseline.
        """
        batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
        dims = tl.arange(0, BLOCK_D)
        length = tl.load(seq_lens + batch)
        req = tl.load(req_indices + batch)
        pack_start = tl.load(cu_k + batch)
        valid_count = tl.load(cu_k + batch + 1) - pack_start
        positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
        valid = (cols < valid_count) & (positions >= 0) & (positions < length)
        slots = tl.load(
            req_to_token + req * req_stride + tl.where(valid, positions, 0),
            mask=valid,
            other=0,
        )
        src = slots[:, None] * heads * dim + head * dim + dims[None, :]
        dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
        mask = valid[:, None] & (dims[None, :] < dim)
        tl.store(out_k + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
        tl.store(out_v + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)


def _check_pool(buffer: Tensor, head_dim: int, group: int) -> None:
    if buffer.dtype != torch.uint8 or buffer.ndim != 3:
        raise ValueError(f"KV4 pool must be [slots, heads, row_bytes] uint8, got {buffer.shape}")
    if buffer.shape[2] != row_bytes(head_dim, group):
        raise ValueError(f"row has {buffer.shape[2]} bytes, expected {row_bytes(head_dim, group)}")
    if buffer.stride(2) != 1:
        raise ValueError("KV4 pool rows must be contiguous")
    if head_dim % HADAMARD_BLOCK or HADAMARD_BLOCK % group:
        raise ValueError(f"head_dim {head_dim} and group {group} must fit the 64-channel block")


def kv4_write(
    buffer: Tensor,
    loc: Tensor,
    x: Tensor,
    group: int = DEFAULT_GROUP,
    size_limit: int | None = None,
    hadamard: bool = True,
    block_t: int = 16,
) -> None:
    """Quantize ``x`` and scatter it into ``buffer`` at ``loc`` (the pool's ``set_kv_buffer``).

    Args:
        buffer: Tensor  # [slots, heads, row_bytes] uint8.
        loc: Tensor  # [tokens] int32 or int64 physical slots.
        x: Tensor  # [tokens, heads, head_dim] (strided per token is fine; last dim contiguous).
        group: $G$.
        size_limit: Slots at or above it are skipped (the fork's ``size + page_size``); the
            buffer's slot count when None.
        hadamard: Rotate each 64-channel block before quantizing.
        block_t: Tokens per program (16 is the smallest ``tl.dot`` M).
    """
    tokens, heads, head_dim = x.shape
    _check_pool(buffer, head_dim, group)
    limit = buffer.shape[0] if size_limit is None else min(size_limit, buffer.shape[0])
    if tokens == 0:
        return
    if not _use_triton(buffer, loc, x):
        reference.write_rows(buffer[:limit], loc, x, group, hadamard)
        return
    if x.stride(2) != 1:
        x = x.contiguous()
    loc = loc.contiguous()
    _kv4_write_kernel[(triton.cdiv(tokens, block_t), heads)](
        x,
        buffer,
        loc,
        tokens,
        limit,
        x.stride(0),
        x.stride(1),
        buffer.stride(0),
        buffer.stride(1),
        D=head_dim,
        G=group,
        HADAMARD=hadamard,
        BLOCK_T=block_t,
        num_warps=4,
    )


def kv4_gather_rows(
    buffer: Tensor,
    slots: Tensor,
    head_dim: int,
    group: int = DEFAULT_GROUP,
    dtype: torch.dtype = torch.bfloat16,
    hadamard: bool = True,
    block_t: int = 32,
) -> Tensor:
    """Dequantize the rows at ``slots`` into a new ``[n, heads, head_dim]`` tensor.

    Args:
        buffer: Tensor  # [slots, heads, row_bytes] uint8.
        slots: Tensor  # [n] int physical slots, all inside the pool.
        head_dim: $D$.
        group: $G$.
        dtype: The output dtype.
        hadamard: The setting the rows were written with.
        block_t: Rows per program.

    Returns:
        Tensor  # [n, heads, head_dim] of ``dtype``.
    """
    _check_pool(buffer, head_dim, group)
    n = slots.shape[0]
    heads = buffer.shape[1]
    if not _use_triton(buffer, slots):
        return reference.gather_rows(buffer, slots, head_dim, group, dtype, hadamard)
    out = torch.empty((n, heads, head_dim), dtype=dtype, device=buffer.device)
    if n == 0:
        return out
    slots = slots.contiguous()
    _kv4_gather_rows_kernel[(triton.cdiv(n, block_t), heads)](
        buffer,
        slots,
        out,
        n,
        buffer.stride(0),
        buffer.stride(1),
        out.stride(0),
        out.stride(1),
        D=head_dim,
        G=group,
        HADAMARD=hadamard,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return out


def kv4_compact_gather(
    k_buffer: Tensor,
    v_buffer: Tensor,
    req_to_token: Tensor,
    req_indices: Tensor,
    indices: Tensor,
    seq_lens: Tensor,
    cu_k: Tensor,
    out_k: Tensor,
    out_v: Tensor,
    batch: int,
    topk: int,
    head_dim: int,
    group: int = DEFAULT_GROUP,
    hadamard: bool = True,
    block_topk: int = 16,
) -> None:
    """The decode gather into the FA2 scratch (the fork's ``_compact_kv`` with dequantization).

    Args:
        k_buffer: Tensor  # [slots, heads, row_bytes] uint8, or [slots, heads, head_dim] float8
            (mode ``v``: K stays FP8 and is decoded from its bytes).
        v_buffer: Tensor  # [slots, heads, row_bytes] uint8.
        req_to_token: Tensor  # [requests, max_context] int32.
        req_indices: Tensor  # [batch] int32.
        indices: Tensor  # [batch, topk] int32.
        seq_lens: Tensor  # [batch] int32.
        cu_k: Tensor  # [batch + 1] int32.
        out_k: Tensor  # [capacity, heads, head_dim] contiguous scratch.
        out_v: Tensor  # [capacity, heads, head_dim] contiguous scratch.
        batch: Rows of ``indices`` to process.
        topk: Columns of ``indices``.
        head_dim: $D$.
        group: $G$.
        hadamard: The setting the rows were written with.
        block_topk: Columns per program (the fork uses 16; it is also ``tl.dot``'s M).
    """
    k_packed = reference.is_packed(k_buffer, head_dim)
    if k_packed:
        _check_pool(k_buffer, head_dim, group)
    elif k_buffer.dtype != torch.float8_e4m3fn or k_buffer.shape[2] != head_dim:
        raise ValueError("an unpacked K buffer must be float8_e4m3fn [slots, heads, head_dim]")
    _check_pool(v_buffer, head_dim, group)
    heads = v_buffer.shape[1]
    if out_k.shape[1:] != (heads, head_dim) or not out_k.is_contiguous():
        raise ValueError(f"scratch must be contiguous [capacity, {heads}, {head_dim}]")
    if not _use_triton(k_buffer, v_buffer, out_k):
        reference.compact_gather(
            k_buffer, v_buffer, req_to_token, req_indices, indices[:batch, :topk], seq_lens,
            cu_k, out_k, out_v, head_dim, group, hadamard,
        )
        return
    if batch == 0:
        return
    if indices.stride(1) != 1 or req_to_token.stride(1) != 1:
        raise ValueError("indices and req_to_token must have unit stride along their rows")
    k_bytes = k_buffer if k_packed else k_buffer.view(torch.uint8)
    _kv4_compact_gather_kernel[(batch, heads, triton.cdiv(topk, block_topk))](
        k_bytes,
        v_buffer,
        req_to_token,
        req_indices.contiguous(),
        indices,
        seq_lens.contiguous(),
        cu_k.contiguous(),
        out_k,
        out_v,
        topk,
        req_to_token.stride(0),
        indices.stride(0),
        k_bytes.stride(0),
        k_bytes.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        HEADS=heads,
        D=head_dim,
        G=group,
        HADAMARD=hadamard,
        K_PACKED=k_packed,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )


def fp8_compact_gather(
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
    """The FP8 path's decode gather, the benchmark's baseline: the fork's kernel on CUDA.

    Args:
        k: Tensor  # [slots, heads, head_dim] float8 (or any dtype the scratch can take).
        v: Tensor  # [slots, heads, head_dim].
        req_to_token: Tensor  # [requests, max_context] int32.
        req_indices: Tensor  # [batch] int32.
        indices: Tensor  # [batch, topk] int32.
        seq_lens: Tensor  # [batch] int32.
        cu_k: Tensor  # [batch + 1] int32.
        out_k: Tensor  # [capacity, heads, head_dim] scratch.
        out_v: Tensor  # [capacity, heads, head_dim] scratch.
        batch: Rows.
        topk: Columns.
    """
    if k.is_cuda and triton is not None:
        _, heads, dim = k.shape
        _compact_kv_fp8[(batch, heads, triton.cdiv(topk, 16))](
            k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, topk, heads,
            dim, req_to_token.stride(0), indices.stride(0), BLOCK_TOPK=16,
            BLOCK_D=triton.next_power_of_2(dim), num_warps=8,
        )
        return
    for b in range(batch):
        start, count = int(cu_k[b]), int(cu_k[b + 1]) - int(cu_k[b])
        cols = torch.arange(topk, device=indices.device)
        positions = indices[b].long()
        valid = (cols < count) & (positions >= 0) & (positions < int(seq_lens[b]))
        slots = req_to_token[int(req_indices[b]), positions[valid]].long()
        out_k[start + cols[valid]] = k.index_select(0, slots).to(out_k.dtype)
        out_v[start + cols[valid]] = v.index_select(0, slots).to(out_v.dtype)
