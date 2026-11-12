"""Tests of the KV4 cache (task P5.5, scripts/fn_kvq/ and configs/P5.5-01.yaml).

CPU, no GPU: the layout and byte accounting (no torch), the PyTorch reference (round-trip error
bounds, packing and scale bytes, the Hadamard rotation, the padding slot, page and slot
indexing of the decode gather), the dispatchers' CPU fall-back, the pool subclass against a stub
of the fork's base class, the bench script, and every hash-guarded edit of configs/P5.5-01.yaml
applied to the fork's own files at /tmp/fnsg/penny (skipped without the clone).

Under ``TRITON_INTERPRET=1`` (set before the first import of triton) the Triton kernels run on
CPU tensors through Triton's interpreter and are compared with the reference; the ``gpu`` tests
do the same on a CUDA device with BF16 outputs.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "configs" / "P5.5-01.yaml"
BASE_CONFIG = ROOT / "configs" / "P5.2-01.yaml"
FORK = Path("/tmp/fnsg/penny/python")
# The wheel's own files (pennyroyal-v253's sglang-0.5.19+gd00d88efc8d6), read out on Kaggle:
# kv_cache_configurator.py differs from the clone's d00d88e there, so the edits are checked
# against the wheel first and the clone only without the peek.
PEEK = ROOT / "runs" / "wheel-peek" / "run_logs" / "peek"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fn_kvq import layout  # noqa: E402

torch = pytest.importorskip("torch")
from fn_kvq import reference, triton_kernels  # noqa: E402

D, G, ROW = 256, 32, 144
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1" and triton_kernels.triton is not None


def load_probe() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("fn_probe", SCRIPTS / "fn_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fn_probe"] = module
    spec.loader.exec_module(module)
    return module


def rows(n: int, seed: int = 0, heads: int = 1, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    channel = torch.exp(0.5 * torch.randn(D, generator=gen))
    return (torch.randn(n, heads, D, generator=gen) * channel * scale).to(torch.bfloat16)


# ---------------------------------------------------------------------------- layout, no torch


def test_row_bytes_and_cell_sizes_of_flash_next():
    assert layout.row_bytes(256, 32) == 144
    assert layout.row_bytes(256, 64) == 136
    assert layout.row_bytes(256, 16) == 160
    assert layout.head_dim_from_row(144, 32) == 256
    with pytest.raises(ValueError):
        layout.row_bytes(256, 24)
    with pytest.raises(ValueError):
        layout.head_dim_from_row(145, 32)
    fp8 = layout.bytes_per_token(layers=12, kv_heads=1, head_dim=256, indexer_bytes=768,
                                 mode="off")
    both = layout.bytes_per_token(layers=12, kv_heads=1, head_dim=256, indexer_bytes=768)
    v_only = layout.bytes_per_token(layers=12, kv_heads=1, head_dim=256, indexer_bytes=768,
                                    mode="v")
    assert (fp8, both, v_only) == (6912, 4224, 5568)
    # Page-aligned pools (page 64), the research notes' figures.
    assert int(169_728 * fp8 / both) // 64 * 64 == 277_696
    assert int(169_728 * fp8 / v_only) // 64 * 64 == 210_688
    # The BF16 fallback's K in mode v is 512 bytes.
    assert layout.bytes_per_token(layers=12, kv_heads=1, head_dim=256, mode="v",
                                  kv_itemsize=2) == 12 * (512 + 144)


def test_layout_byte_positions():
    lay = layout.KV4Layout(256, 32)
    assert (lay.groups, lay.packed_bytes, lay.scale_offset, lay.row_bytes) == (8, 128, 128, 144)
    assert lay.scale_byte(0) == 128 and lay.scale_byte(7) == 142
    assert lay.element_byte(0) == (0, 0) and lay.element_byte(1) == (0, 4)
    assert lay.element_byte(255) == (127, 4)


def test_env_switches():
    assert layout.mode_from_env({}) == "off"
    assert layout.mode_from_env({"ARC26_KV4": "1"}) == "kv"
    assert layout.mode_from_env({"ARC26_KV4": "kv"}) == "kv"
    assert layout.mode_from_env({"ARC26_KV4": "v"}) == "v"
    assert layout.mode_from_env({"ARC26_KV4": "0"}) == "off"
    with pytest.raises(ValueError):
        layout.mode_from_env({"ARC26_KV4": "maybe"})
    assert layout.hadamard_from_env({}) is True
    assert layout.hadamard_from_env({"ARC26_KV4_HADAMARD": "0"}) is False
    assert layout.group_from_env({}) == 32
    assert layout.group_from_env({"ARC26_KV4_GROUP": "64"}) == 64


# ---------------------------------------------------------------------------- the reference


@pytest.mark.parametrize("hadamard", [False, True])
def test_round_trip_error_is_within_half_a_step_per_group(hadamard):
    x = rows(64, seed=1)
    packed = reference.quantize_rows(x, G, hadamard)
    assert packed.shape == (64, 1, ROW) and packed.dtype == torch.uint8
    y = reference.dequantize_rows(packed, D, G, torch.float32, hadamard)
    scales = packed[..., 128:].contiguous().view(torch.float16).float()  # [64, 1, 8]
    # The bound holds in the rotated domain: |x' - s q| <= s / 2 (plus the FP16 rounding of s).
    x_dom = reference.rotate(x) if hadamard else x.float()
    y_dom = reference.rotate(y) if hadamard else y
    err = (x_dom - y_dom).abs().reshape(64, 1, 8, 32).amax(-1)
    assert torch.all(err <= 0.5 * scales * (1 + 2**-9) + 1e-6)
    # The relative RMS error in the original domain (log-normal channel scales, so above the
    # Gaussian case's 0.08); FP8 E4M3's is about 0.035 on the same data.
    rel = reference.error_stats(x, y)["rel_rms"]
    assert rel < 0.15
    assert reference.error_stats(x, reference.fp8_roundtrip(x))["rel_rms"] < 0.05


def test_zero_rows_and_exact_grid_values_round_trip_exactly():
    x = torch.zeros(3, 1, D)
    for hadamard in (False, True):
        packed = reference.quantize_rows(x, G, hadamard)
        assert torch.all(reference.dequantize_rows(packed, D, G, hadamard=hadamard) == 0)
    # Values on the grid s * q with s = 0.25: exact without rotation.
    q = torch.randint(-7, 8, (5, 1, D)).float()
    q[:, :, 0::32] = 7  # every group's absmax is 7 * 0.25
    x = q * 0.25
    y = reference.dequantize_rows(reference.quantize_rows(x, G, False), D, G, hadamard=False)
    assert torch.equal(x, y)


def test_packing_layout_nibbles_and_little_endian_fp16_scales():
    x = torch.zeros(1, 1, D)
    x[0, 0, :32] = torch.arange(32).float() - 16.0  # group 0: absmax 16, scale 16/7
    packed = reference.quantize_rows(x, G, False)[0, 0]
    scale = torch.tensor([16.0 / 7.0]).to(torch.float16)
    lo, hi = scale.view(torch.uint8).tolist()
    assert packed[128].item() == lo and packed[129].item() == hi  # little-endian FP16
    assert torch.all(packed[130:144] == 0)  # the other groups are zero
    inv = 1.0 / scale.float().item()
    for i in range(32):
        q = int(torch.round(torch.tensor(x[0, 0, i].item() * inv)).clamp(-8, 7).item())
        byte, shift = layout.KV4Layout(D, G).element_byte(i)
        assert (packed[byte].item() >> shift) & 0xF == q + 8, i
    assert torch.all(packed[16:128] == 0x88)  # zero elements pack as nibble 8 twice


def test_hadamard_matrix_is_orthogonal_symmetric_and_its_own_inverse():
    h = reference.hadamard_matrix()
    assert torch.allclose(h @ h, torch.eye(64), atol=1e-6)
    assert torch.equal(h, h.T)
    assert h[0, 0].item() == 0.125 and h[1, 1].item() == -0.125
    x = torch.randn(4, 1, D)
    assert torch.allclose(reference.rotate(reference.rotate(x)), x, atol=1e-5)
    # An outlier channel spreads over its 64-block: the rotated group absmax drops.
    x = torch.zeros(1, 1, D)
    x[0, 0, 5] = 64.0
    rotated = reference.rotate(x)
    assert rotated.abs().max().item() == pytest.approx(8.0)
    assert torch.count_nonzero(rotated).item() == 64


def test_write_skips_the_padding_slot_and_gather_reads_slots():
    buffer = torch.zeros(10, 1, ROW, dtype=torch.uint8)
    x = rows(4, seed=2)
    loc = torch.tensor([7, 10, 2, -1], dtype=torch.int32)  # 10 and -1 are outside the pool
    reference.write_rows(buffer, loc, x, G, False)
    assert torch.all(buffer[[0, 1, 3, 4, 5, 6, 8, 9]] == 0)
    got = reference.gather_rows(buffer, torch.tensor([2, 7]), D, G, torch.float32, False)
    want = reference.dequantize_rows(reference.quantize_rows(x[[2, 0]], G, False), D, G,
                                     hadamard=False)
    assert torch.equal(got, want)


def decode_batch(slots: int, seed: int = 3):
    """Page-64 slot tables for 3 requests and packed top-k indices with invalid entries."""
    gen = torch.Generator().manual_seed(seed)
    page = 64
    assert slots >= 13 * page, "3 requests of 4 pages plus the padding page"
    req_to_token = torch.zeros((6, 4 * page), dtype=torch.int32)
    pages = torch.randperm(slots // page - 1, generator=gen)[:12] + 1
    for r in range(3):
        req_to_token[r + 1] = (pages[4 * r : 4 * r + 4, None] * page
                               + torch.arange(page)[None, :]).reshape(-1)
    req_indices = torch.tensor([1, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([200, 70, 256], dtype=torch.int32)
    topk = 48
    indices = torch.full((3, topk), -1, dtype=torch.int32)
    counts = [40, 30, 48]
    for b in range(3):
        indices[b, : counts[b]] = torch.randperm(int(seq_lens[b]), generator=gen)[
            : counts[b]].to(torch.int32)
    indices[1, 25] = 500  # beyond the length: skipped though counted
    indices[0, 44] = 3  # past the valid count: skipped
    cu_k = torch.tensor([0, 40, 70, 118], dtype=torch.int32)
    return dict(req_to_token=req_to_token, req_indices=req_indices, indices=indices,
                seq_lens=seq_lens, cu_k=cu_k, topk=topk)


@pytest.mark.parametrize("mode", ["kv", "v"])
def test_compact_gather_follows_the_forks_page_and_slot_rule(mode):
    slots = 64 * 20
    full = rows(slots, seed=4)
    vb = torch.zeros(slots, 1, ROW, dtype=torch.uint8)
    reference.write_rows(vb, torch.arange(slots), full * 2, G, True)
    if mode == "kv":
        kb = torch.zeros(slots, 1, ROW, dtype=torch.uint8)
        reference.write_rows(kb, torch.arange(slots), full, G, True)
    else:
        kb = full.to(torch.float8_e4m3fn)
    b = decode_batch(slots)
    out_k = torch.full((3 * b["topk"], 1, D), 7.0, dtype=torch.bfloat16)
    out_v = out_k.clone()
    reference.compact_gather(kb, vb, b["req_to_token"], b["req_indices"], b["indices"],
                             b["seq_lens"], b["cu_k"], out_k, out_v, D, G, True)
    for r in range(3):
        start, count = int(b["cu_k"][r]), int(b["cu_k"][r + 1] - b["cu_k"][r])
        for c in range(b["topk"]):
            pos = int(b["indices"][r, c])
            valid = c < count and 0 <= pos < int(b["seq_lens"][r])
            row = start + c
            if not valid:
                if row < start + count:
                    assert torch.all(out_k[row] == 7.0) and torch.all(out_v[row] == 7.0)
                continue
            slot = int(b["req_to_token"][int(b["req_indices"][r]), pos])
            want_v = reference.dequantize_rows(vb[slot], D, G, torch.bfloat16, True)
            assert torch.equal(out_v[row], want_v)
            if mode == "kv":
                want_k = reference.dequantize_rows(kb[slot], D, G, torch.bfloat16, True)
            else:
                want_k = kb[slot].to(torch.bfloat16)
            assert torch.equal(out_k[row], want_k)
    # Rows past the last valid count are untouched, as the fork's kernel leaves them.
    assert torch.all(out_k[118:] == 7.0)


def test_nibble_stats_tell_a_rounding_flip_from_a_layout_bug():
    x = rows(8, seed=9)
    ref = reference.quantize_rows(x, G, False)
    same = reference.nibble_stats(ref, ref, D, G)
    assert same == {"nibble_diff_share": 0.0, "nibble_diff_max": 0, "scale_diff_share": 0.0,
                    "dequant_diff_max_steps": 0.0}
    flipped = ref.clone()
    byte = int(flipped[3, 0, 10])
    lo = byte & 0xF
    flipped[3, 0, 10] = (byte & 0xF0) | (lo + 1 if lo < 15 else lo - 1)  # one nibble, one step
    one = reference.nibble_stats(flipped, ref, D, G)
    assert one["nibble_diff_share"] == pytest.approx(1 / (8 * D))
    assert one["nibble_diff_max"] == 1 and one["scale_diff_share"] == 0.0
    assert one["dequant_diff_max_steps"] == pytest.approx(1.0)
    swapped = ref.clone()
    swapped[..., :128] = (ref[..., :128] << 4) | (ref[..., :128] >> 4)  # nibble order bug
    bug = reference.nibble_stats(swapped, ref, D, G)
    assert bug["nibble_diff_share"] > 0.5 and bug["nibble_diff_max"] > 1
    assert bug["dequant_diff_max_steps"] > 1.0
    scaled = ref.clone()
    scaled[0, 0, 129] ^= 0x01  # one scale's high byte
    assert reference.nibble_stats(scaled, ref, D, G)["scale_diff_share"] == pytest.approx(1 / 64)


def test_is_packed_tells_the_pools_apart():
    assert reference.is_packed(torch.zeros(2, 1, ROW, dtype=torch.uint8), D)
    assert not reference.is_packed(torch.zeros(2, 1, D, dtype=torch.float8_e4m3fn), D)
    assert not reference.is_packed(torch.zeros(2, 1, D, dtype=torch.bfloat16), D)
    assert not reference.is_packed(torch.zeros(2, 1, D, dtype=torch.uint8), D)


# ---------------------------------------------------------------------------- the dispatchers


def test_dispatchers_fall_back_to_the_reference_on_cpu(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    slots = 64 * 4
    kb = torch.zeros(slots, 1, ROW, dtype=torch.uint8)
    kb_ref = kb.clone()
    x = rows(20, seed=5)
    loc = torch.randperm(slots)[:20].to(torch.int32)
    loc[0] = slots  # the padding slot
    triton_kernels.kv4_write(kb, loc, x, G, size_limit=slots, hadamard=True)
    reference.write_rows(kb_ref, loc, x, G, True)
    assert torch.equal(kb, kb_ref)
    got = triton_kernels.kv4_gather_rows(kb, loc[1:], D, G, torch.bfloat16, True)
    assert torch.equal(got, reference.gather_rows(kb, loc[1:], D, G, torch.bfloat16, True))
    with pytest.raises(ValueError):
        triton_kernels.kv4_write(torch.zeros(4, 1, 100, dtype=torch.uint8), loc, x, G)
    with pytest.raises(ValueError):
        triton_kernels.kv4_write(kb, loc, x, 24)


def kernel_cases(device: torch.device, dtype: torch.dtype):
    """Write, gather and compact-gather through the kernels against the reference."""
    slots = 64 * 14
    for hadamard in (False, True):
        full = rows(slots, seed=6).to(device)
        locs = torch.arange(slots, dtype=torch.int32, device=device)
        kb = torch.zeros(slots, 1, ROW, dtype=torch.uint8, device=device)
        vb = torch.zeros_like(kb)
        triton_kernels.kv4_write(kb, locs, full, G, hadamard=hadamard)
        triton_kernels.kv4_write(vb, locs, full * 0.5, G, hadamard=hadamard)
        assert torch.equal(kb.cpu(), reference.quantize_rows(full.cpu(), G, hadamard)), hadamard
        # A strided input (the QKV split's token stride) and the padding slot.
        wide = torch.randn(17, 3 * D, device=device).to(torch.bfloat16)[:, :D].view(17, 1, D)
        loc = torch.randperm(slots, device=device)[:17].to(torch.int32)
        loc[3] = slots
        kb2, kb3 = torch.zeros_like(kb), torch.zeros_like(kb)
        triton_kernels.kv4_write(kb2, loc, wide, G, hadamard=hadamard)
        reference.write_rows(kb3, loc, wide, G, hadamard)
        assert torch.equal(kb2, kb3)
        sel = locs[::7]
        got = triton_kernels.kv4_gather_rows(kb, sel, D, G, dtype, hadamard)
        want = reference.gather_rows(kb, sel, D, G, dtype, hadamard)
        assert torch.equal(got, want), hadamard
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in decode_batch(slots).items()}
        for k_buffer in (kb, full.to(torch.float8_e4m3fn)):
            out_k = torch.full((3 * b["topk"], 1, D), 7.0, dtype=dtype, device=device)
            out_v, ref_k, ref_v = out_k.clone(), out_k.clone(), out_k.clone()
            triton_kernels.kv4_compact_gather(
                k_buffer, vb, b["req_to_token"], b["req_indices"], b["indices"], b["seq_lens"],
                b["cu_k"], out_k, out_v, 3, b["topk"], D, G, hadamard)
            reference.compact_gather(
                k_buffer, vb, b["req_to_token"], b["req_indices"], b["indices"], b["seq_lens"],
                b["cu_k"], ref_k, ref_v, D, G, hadamard)
            assert torch.equal(out_k, ref_k), (hadamard, k_buffer.dtype)
            assert torch.equal(out_v, ref_v), (hadamard, k_buffer.dtype)


@pytest.mark.skipif(not INTERPRET, reason="TRITON_INTERPRET=1 runs the kernels on CPU")
def test_kernels_under_the_triton_interpreter_match_the_reference_in_fp32():
    # The interpreter truncates FP32 to BF16 where the GPU rounds, so compare in FP32 here.
    kernel_cases(torch.device("cpu"), torch.float32)


@pytest.mark.gpu
def test_kernels_on_the_gpu_match_the_reference_in_bf16():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    kernel_cases(torch.device("cuda"), torch.bfloat16)


@pytest.mark.gpu
def test_bench_runs_on_the_gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from fn_kvq import bench

    out = tmp_path / "bench.json"
    assert bench.main(["--out", str(out), "--slots", str(64 * 40), "--seq-len", "1000",
                       "--topk", "256", "--iters", "3", "--warmup", "1"]) == 0


# ---------------------------------------------------------------------------- the pool


class StubPool:
    """The slice of the fork's ``MHATokenToKVPool`` the subclass relies on."""

    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device,
                 enable_memory_saver=False, start_layer=None, **_):
        self.size, self.page_size, self.dtype, self.device = size, page_size, dtype, device
        self.head_num, self.head_dim, self.v_head_dim = head_num, head_dim, head_dim
        self.layer_num, self.start_layer = layer_num, start_layer or 0
        self.store_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
        self.use_hnd, self.kv_cache_layout = False, "nhd"
        self.k_buffer = self.v_buffer = None
        self._create_buffers()

    def _create_buffers(self):
        k_shape, v_shape = self._kv_buffer_shapes()
        self.k_buffer = [torch.zeros(k_shape, dtype=self.store_dtype)
                         for _ in range(self.layer_num)]
        self.v_buffer = [torch.zeros(v_shape, dtype=self.store_dtype)
                         for _ in range(self.layer_num)]

    def _kv_buffer_shapes(self):
        rows = self.size + self.page_size
        return (rows, self.head_num, self.head_dim), (rows, self.head_num, self.v_head_dim)

    def get_key_buffer(self, layer_id):
        return self._get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id):
        return self._get_value_buffer(layer_id)

    def get_kv_size_bytes(self):
        return (sum(t.numel() * t.element_size() for t in self.k_buffer),
                sum(t.numel() * t.element_size() for t in self.v_buffer))


def load_sglang_kv4(monkeypatch, env: dict[str, str]):
    stub = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    stub.MHATokenToKVPool = StubPool
    stub.unwrap_write_loc = lambda loc: (loc, None, None)
    for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.memory_pool", stub)
    for name in ("ARC26_KV4", "ARC26_KV4_HADAMARD", "ARC26_KV4_GROUP"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ARC26_KV4_LOG", "0")
    monkeypatch.delitem(sys.modules, "fn_kvq.sglang_kv4", raising=False)
    import fn_kvq.sglang_kv4 as module

    return importlib.reload(module)


class Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


@pytest.mark.parametrize("mode", ["1", "v"])
def test_pool_subclass_allocates_packed_rows_and_round_trips(monkeypatch, mode):
    module = load_sglang_kv4(monkeypatch, {"ARC26_KV4": mode, "ARC26_KV4_HADAMARD": "1"})
    pool = module.KV4MHATokenToKVPool(size=64 * 3, page_size=64, dtype=torch.float8_e4m3fn,
                                      head_num=1, head_dim=D, layer_num=2, device="cpu",
                                      start_layer=5)
    k_row = ROW if mode == "1" else D
    assert tuple(pool.k_buffer[0].shape) == (256, 1, k_row)
    assert tuple(pool.v_buffer[0].shape) == (256, 1, ROW)
    assert pool.store_dtype == torch.uint8
    k_bytes, v_bytes = pool.get_kv_size_bytes()
    assert (k_bytes, v_bytes) == (2 * 256 * k_row, 2 * 256 * ROW)
    assert module.kv4_cell_size(1, D, D, 12, 1) == 12 * (k_row + ROW)
    k, v = rows(5, seed=7), rows(5, seed=8)
    # Slot 0 is in the fork's padding page (written, never read); 256 = size + page_size would
    # be out of bounds, which the fork's own writer rejects.
    loc = torch.tensor([70, 71, 130, 3, 0], dtype=torch.int32)
    pool.set_kv_buffer(Layer(6), loc, k, v, layer_id_override=6)
    assert torch.all(pool.v_buffer[0] == 0)
    if mode == "1":
        assert torch.equal(pool.get_key_buffer(6)[[70, 71, 130, 3, 0]],
                           reference.quantize_rows(k, G, True))
    else:
        assert pool.get_key_buffer(6).dtype == torch.float8_e4m3fn
        # Layer 6 is local layer 1 (start_layer 5); the bytes of the FP8 cast, as the fork stores.
        assert torch.equal(pool.k_buffer[1][[70, 71, 130, 3, 0]],
                           k.to(torch.float8_e4m3fn).view(torch.uint8))
    assert torch.equal(pool.get_value_buffer(6)[[70, 71, 130, 3, 0]],
                       reference.quantize_rows(v, G, True))
    # The shims the backend edits call.
    out = torch.empty(4, 1, D, dtype=torch.bfloat16)
    got_k, got_v = module.kv_gather_rows(pool.get_key_buffer(6), pool.get_value_buffer(6),
                                         torch.tensor([70, 71, 130, 3]), torch.bfloat16)
    assert torch.equal(got_v, reference.dequantize_rows(
        reference.quantize_rows(v[:4], G, True), D, G, torch.bfloat16, True))
    assert got_k.shape == out.shape
    with pytest.raises(RuntimeError):
        pool.set_kv_buffer(Layer(6), loc, k, v, dcp_kv_mask=torch.ones(5))


def test_pool_subclass_reads_the_group_and_rotation_from_the_environment(monkeypatch):
    module = load_sglang_kv4(monkeypatch, {"ARC26_KV4": "kv", "ARC26_KV4_HADAMARD": "0",
                                           "ARC26_KV4_GROUP": "64"})
    assert (module.MODE, module.GROUP, module.HADAMARD) == ("kv", 64, False)
    pool = module.KV4MHATokenToKVPool(size=64, page_size=64, dtype=torch.bfloat16, head_num=1,
                                      head_dim=D, layer_num=1, device="cpu")
    assert tuple(pool.k_buffer[0].shape) == (128, 1, 136)
    assert module.kv4_cell_size(1, D, D, 12, 2) == 12 * 2 * 136


# ---------------------------------------------------------------------------- the bench script


def test_bench_smoke_runs_the_reference_path_on_cpu(tmp_path):
    out = tmp_path / "bench.json"
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS)}
    env.pop("TRITON_INTERPRET", None)
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "fn_kvq" / "bench.py"), "--smoke", "--out", str(out)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    doc = json.loads(out.read_text())
    assert doc["ok"] is True
    assert doc["bytes_per_token"] == {"off": 6912, "kv": 4224, "v": 5568}
    assert doc["pool_tokens_at_fp8_pool"]["kv"] == 277_696
    assert set(doc["timings"]) == {"fp8", "hadamard0", "hadamard1"}
    for check in doc["checks"].values():
        assert check["write_k_rows_equal"] and check["write_v_rows_equal"]
        assert check["write_k_nibble_diff_share"] == 0.0 and check["write_v_nibble_diff_max"] == 0
    assert doc["errors"]["hadamard1"]["k"]["rel_rms"] < doc["errors"]["hadamard0"]["k"]["rel_rms"]


# ---------------------------------------------------------------------------- the edits


def patch_items() -> list[dict]:
    return yaml.safe_load(CONFIG.read_text())["patches"]["items"]


def test_the_config_carries_the_switch_and_the_fp8_edits_unchanged():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["engine"]["env"] == {"ARC26_KV4": "1", "ARC26_KV4_HADAMARD": "1",
                                       "ARC26_KV4_GROUP": "32"}
    base = yaml.safe_load(BASE_CONFIG.read_text())["patches"]["items"]
    items = patch_items()
    names = [item["name"] for item in items]
    assert len(names) == len(set(names))
    for base_item in base:
        mine = next(item for item in items if item["name"] == base_item["name"])
        for key in ("file", "sha256_before", "pattern", "replacement", "count", "fallback_flags"):
            assert mine[key] == base_item[key], (base_item["name"], key)
    for item in items:
        assert set(item) >= {"name", "file", "sha256_before", "sha256_after", "pattern",
                             "replacement", "count", "fallback_flags"}
        re.compile(item["pattern"])
        assert len(item["sha256_before"]) == 64 and len(item["sha256_after"]) == 64
    # The chain: in one file each edit starts from the previous one's result.
    last: dict[str, str] = {}
    for item in items:
        if item["file"] in last:
            assert item["sha256_before"] == last[item["file"]], item["name"]
        last[item["file"]] = item["sha256_after"]
    kv4 = [item for item in items if item["name"].startswith("kv4_")]
    assert len(kv4) == 9
    assert all(item["fallback_flags"] == {} for item in kv4)
    # Call sites are renamed before the helper that names the fork's gather is inserted.
    assert names.index("kv4_backend_extract") < names.index("kv4_backend_helpers")


def wheel_file(file: str) -> Path | None:
    """The wheel's copy of a fork file (the peek), else the clone's, else None."""
    peek = PEEK / (file.replace("/", "__") + ".txt")
    if peek.is_file():
        return peek
    clone = FORK / file
    return clone if clone.is_file() else None


def test_the_peek_hashes_are_the_edits_starting_hashes():
    if not (PEEK / "peek.json").is_file():
        pytest.skip("no wheel peek")
    hashes = json.loads((PEEK / "peek.json").read_text())["files"]
    first = {}
    for item in patch_items():
        first.setdefault(item["file"], item["sha256_before"])
    for file, digest in first.items():
        assert hashes[file] == digest, file
        assert hashlib.sha256(wheel_file(file).read_bytes()).hexdigest() == digest, file


def test_every_edit_applies_to_the_forks_files_in_order(tmp_path):
    items = patch_items()
    sources = {item["file"]: wheel_file(item["file"]) for item in items}
    if any(source is None for source in sources.values()):
        pytest.skip("neither the wheel peek nor the clone of the fork at d00d88e")
    probe = load_probe()
    for file, source in sources.items():
        target = tmp_path / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    for item in items:
        result = probe.apply_patch(tmp_path, item)
        assert result["status"] == "applied", (item["name"], result)
        assert result["matches"] == item["count"], item["name"]
        assert result["sha256_after"] == item["sha256_after"], item["name"]
        # Applied twice: already applied, by the hash.
        assert probe.apply_patch(tmp_path, item)["status"] == "already applied"
    for file in {item["file"] for item in items}:
        text = (tmp_path / file).read_text()
        ast.parse(text)
        assert hashlib.sha256(text.encode()).hexdigest() == max(
            (i for i in items if i["file"] == file), key=items.index)["sha256_after"]
    backend = (tmp_path / "sglang/srt/layers/attention/qwen_sparse_attn_backend.py").read_text()
    assert backend.count("_arc26_kv_extract(") == 3  # two call sites and the definition
    assert "layer.head_dim,\n            q.dtype," in backend
    assert "*_arc26_kv_gather(k_buffer, v_buffer, gather_index, q.dtype)," in backend
    assert "qwen_sparse_kv_extraction_compact_triton(k, v, *args)" in backend
    configurator = (tmp_path / "sglang/srt/mem_cache/kv_cache_configurator.py").read_text()
    assert configurator.count("_arc26_kv4_pool_class(full_pool_class)") == 1
    sizing = (tmp_path / "sglang/srt/model_executor/pool_configurator.py").read_text()
    assert "cell_size = _arc26_kv4_cell_size(n, model_config.head_dim," in sizing


def exec_helper(name: str, tail: str, namespace: dict) -> dict:
    """Execute the inserted module-level helpers of one edit (up to the anchor it precedes)."""
    item = next(i for i in patch_items() if i["name"] == name)
    exec(item["replacement"].split(tail)[0], namespace)
    return namespace


@pytest.mark.parametrize("switch", [None, "0", "off"])
def test_the_shims_are_the_forks_calls_with_the_switch_off(monkeypatch, switch):
    """Inertness: with ARC26_KV4 unset or 0, an FP8 store (uint8 bytes, or their float8 view)
    and a BF16 pool all take the fork's own gather and index_select; fn_kvq is never imported."""
    monkeypatch.delenv("ARC26_KV4", raising=False)
    if switch is not None:
        monkeypatch.setenv("ARC26_KV4", switch)
    calls = []
    namespace = exec_helper("kv4_backend_helpers", "\n\n\nclass ", {
        "torch": torch,
        "qwen_sparse_kv_extraction_compact_triton": lambda *a: calls.append(("fork", a))})
    assert namespace["_ARC26_KV4"] is False
    monkeypatch.setitem(sys.modules, "fn_kvq.sglang_kv4", None)  # an import would raise
    store = torch.randint(0, 255, (4, 1, D), dtype=torch.uint8)  # the FP8 pool's store_dtype
    for k, v in ((store, store), (store.view(torch.float8_e4m3fn), store.view(torch.float8_e4m3fn)),
                 (store.view(torch.bfloat16)[..., :128], store.view(torch.bfloat16)[..., :128])):
        namespace["_arc26_kv_extract"](k, v, "rest")
        assert calls[-1] == ("fork", (k, v, "rest"))
        got_k, got_v = namespace["_arc26_kv_gather"](k, v, torch.tensor([2, 0]), torch.bfloat16)
        assert torch.equal(got_k.view(torch.uint8), k.view(torch.uint8)[[2, 0]])
        assert got_v.dtype == v.dtype
    assert len(calls) == 3


def test_the_shims_route_packed_buffers_with_the_switch_on(monkeypatch):
    monkeypatch.setenv("ARC26_KV4", "1")
    calls = []
    namespace = exec_helper("kv4_backend_helpers", "\n\n\nclass ", {
        "torch": torch,
        "qwen_sparse_kv_extraction_compact_triton": lambda *a: calls.append(("fork", a))})
    assert namespace["_ARC26_KV4"] is True
    fp8 = torch.zeros(2, 1, D, dtype=torch.float8_e4m3fn)
    namespace["_arc26_kv_extract"](fp8, fp8, "rest")  # a float8 view still takes the fork's path
    assert calls == [("fork", (fp8, fp8, "rest"))]
    stub = types.ModuleType("fn_kvq.sglang_kv4")
    stub.kv_extract_compact = lambda *a: calls.append(("kv4_extract", a))
    stub.kv_gather_rows = lambda *a: calls.append(("kv4_gather", a))
    monkeypatch.setitem(sys.modules, "fn_kvq.sglang_kv4", stub)
    packed = torch.zeros(2, 1, ROW, dtype=torch.uint8)
    namespace["_arc26_kv_extract"](fp8, packed, "rest")  # mode v: K float8, V packed
    namespace["_arc26_kv_gather"](packed, packed, torch.tensor([1]), torch.bfloat16)
    assert [c[0] for c in calls[1:]] == ["kv4_extract", "kv4_gather"]


@pytest.mark.parametrize("switch", [None, "0"])
def test_the_pool_class_and_cell_size_helpers_are_inert_with_the_switch_off(monkeypatch, switch):
    monkeypatch.delenv("ARC26_KV4", raising=False)
    if switch is not None:
        monkeypatch.setenv("ARC26_KV4", switch)
    monkeypatch.setitem(sys.modules, "fn_kvq.sglang_kv4", None)
    # Revision 20: the helper sits above the class's @dataclass decorator.
    tail = "\n\n\n@dataclass(slots=True, kw_only=True)\nclass KVCacheConfigurator"
    pool = exec_helper("kv4_pool_class_helper", tail, {})
    assert pool["_arc26_kv4_pool_class"](StubPool) is StubPool
    cell = exec_helper("kv4_cell_size_helper", "\n\n\ndef _dflash", {})
    # The fork's own term: n * (head_dim + v_head_dim) * layers * kv_size.
    assert cell["_arc26_kv4_cell_size"](1, 256, 256, 12, 1) == 6144
    assert cell["_arc26_kv4_cell_size"](2, 128, 96, 7, 2) == 2 * (128 + 96) * 7 * 2


def test_the_v_head_dim_edit_returns_the_buffers_last_dim_for_the_forks_pools():
    """The fork's MHA pool sets v_head_dim at construction and its FP8 (uint8 store, float8
    view) and BF16 value buffers end in exactly that dim, so the edit's answer is the original's."""
    item = next(i for i in patch_items() if i["name"] == "kv4_hybrid_v_head_dim")
    original = ("        return self.full_kv_pool.get_value_buffer(self.full_kv_pool.start_layer)"
                ".shape[\n            -1\n        ]")
    body = re.sub(item["pattern"], item["replacement"], original)
    assert body != original
    namespace = {}
    exec("def get_v_head_dim(self):\n" + body, namespace)

    def make_hybrid(dtype):
        class FullPool:
            start_layer, v_head_dim = 3, D

            def get_value_buffer(self, layer_id):
                store = torch.zeros(5, 1, D * dtype.itemsize, dtype=torch.uint8)
                return store.view(dtype)

        class Hybrid:
            full_kv_pool = FullPool()

        return Hybrid()

    for dtype in (torch.float8_e4m3fn, torch.bfloat16):
        hybrid = make_hybrid(dtype)
        original_answer = hybrid.full_kv_pool.get_value_buffer(3).shape[-1]
        assert namespace["get_v_head_dim"](hybrid) == original_answer == D
