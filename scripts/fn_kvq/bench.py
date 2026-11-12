"""Standalone one-GPU benchmark and correctness check of the KV4 kernels (no model, no sglang).

Random K and V at the real shapes (1 KV head of 256 per GPU, 12 QSA layers, page 64, top-k 2,051,
batch 3), the Triton kernels against the PyTorch reference, their timings against the FP8 path's
own write and gather (the fork's ``_compact_kv``, kept in ``fn_kvq.triton_kernels``), the
reconstruction error of INT4 against FP8 E4M3, and one JSON file. Runs on Modal (L40S or L4)
through the ``P5.5-01`` job; on a box without a GPU ``--cpu`` runs the reference path at small
shapes so the script itself is tested.

Usage::

    python scripts/fn_kvq/bench.py --out runs/P5.5/bench.json
    python scripts/fn_kvq/bench.py --smoke --out runs/P5.5/tmp/bench_smoke.json

Every mode and both Hadamard settings are measured in one run: ``kv`` (both INT4) and ``v``
(K FP8) gathers, ``hadamard`` 0 and 1; the error section compares each with the FP8 round trip
of the same data.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fn_kvq import layout, reference  # noqa: E402
from fn_kvq.triton_kernels import (  # noqa: E402
    fp8_compact_gather,
    kv4_compact_gather,
    kv4_gather_rows,
    kv4_write,
    triton,
)


def make_data(
    tokens: int, heads: int, head_dim: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Random K and V with per-channel scale variation and a few outlier channels.

    K: unit Gaussian times a log-normal channel scale ($\sigma = 0.5$) with 4 channels at 8x
    (the rotary dims' outliers the methods note names); V: the same without outliers, half the
    scale. Neither is real data; the Modal phase M run measures the model.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k_scale = torch.exp(0.5 * torch.randn(head_dim, generator=gen))
    k_scale[torch.randperm(head_dim, generator=gen)[:4]] *= 8.0
    v_scale = 0.5 * torch.exp(0.5 * torch.randn(head_dim, generator=gen))
    k = (torch.randn(tokens, heads, head_dim, generator=gen) * k_scale).to(torch.bfloat16)
    v = (torch.randn(tokens, heads, head_dim, generator=gen) * v_scale).to(torch.bfloat16)
    return k.to(device), v.to(device)


def time_it(
    fn: Callable[[], Any], iters: int, warmup: int, device: torch.device
) -> dict[str, float]:
    """Median and mean microseconds of ``fn()`` (CUDA events on the GPU, wall clock on CPU)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return {"median_us": samples[len(samples) // 2], "mean_us": sum(samples) / len(samples),
            "iters": iters}


def build_decode_batch(
    slots: int, batch: int, topk: int, seq_len: int, page: int, device: torch.device, seed: int
) -> dict[str, torch.Tensor]:
    """A decode batch like the engine's: page-aligned slot tables, packed top-k indices."""
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    pages_per_req = math.ceil(seq_len / page)
    total_pages = slots // page
    if batch * pages_per_req + 1 > total_pages:
        raise ValueError(f"{slots} slots hold fewer than {batch} requests of {seq_len} tokens")
    order = torch.randperm(total_pages - 1, generator=gen)[: batch * pages_per_req] + 1
    req_to_token = torch.zeros((batch + 2, pages_per_req * page), dtype=torch.int32)
    req_indices = torch.arange(1, batch + 1, dtype=torch.int32)
    for b in range(batch):
        pages = order[b * pages_per_req : (b + 1) * pages_per_req]
        req_to_token[b + 1] = (pages[:, None] * page + torch.arange(page)[None, :]).reshape(-1)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    count = min(topk, seq_len)
    indices = torch.full((batch, topk), -1, dtype=torch.int32)
    for b in range(batch):
        indices[b, :count] = torch.randperm(seq_len, generator=gen)[:count].to(torch.int32)
    cu_k = torch.arange(batch + 1, dtype=torch.int32) * count
    tensors = dict(req_to_token=req_to_token, req_indices=req_indices, seq_lens=seq_lens,
                   indices=indices, cu_k=cu_k)
    return {name: t.to(device) for name, t in tensors.items()}


class Bench:
    """The data, the batch and the scratch of one run; one method per measured setting."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device
        self.head_dim, self.heads, self.group, self.page = (
            args.head_dim, args.heads, args.group, args.page)
        self.row = layout.row_bytes(self.head_dim, self.group)
        self.k_all, self.v_all = make_data(args.slots, self.heads, self.head_dim, device, args.seed)
        self.locs = torch.arange(args.slots, dtype=torch.int32, device=device)
        self.fp8_k = self.k_all.to(torch.float8_e4m3fn)
        self.fp8_v = self.v_all.to(torch.float8_e4m3fn)
        self.batch = build_decode_batch(args.slots, args.batch, args.topk, args.seq_len, self.page,
                                        device, args.seed)
        capacity = args.batch * args.topk
        self.scratch_k = torch.zeros((capacity, self.heads, self.head_dim), dtype=torch.bfloat16,
                                     device=device)
        self.scratch_v = torch.zeros_like(self.scratch_k)
        self.prefill_loc = self.locs[self.page : self.page + args.prefill_tokens]
        self.prefill_k, self.prefill_v = self.k_all[self.prefill_loc], self.v_all[self.prefill_loc]
        self.decode_loc = self.locs[self.page : self.page + args.batch]
        self.gather_index = self.batch["req_to_token"][1, : args.seq_len].long()

    def timed(self, fn: Callable[[], Any]) -> dict[str, float]:
        """``time_it`` with the run's iteration counts."""
        return time_it(fn, self.args.iters, self.args.warmup, self.device)

    def compact(self, k: torch.Tensor, v: torch.Tensor, out_k: torch.Tensor, out_v: torch.Tensor,
                hadamard: bool) -> None:
        """The decode gather of ``k`` (packed or FP8) and ``v`` (packed) into the scratch."""
        b = self.batch
        kv4_compact_gather(k, v, b["req_to_token"], b["req_indices"], b["indices"], b["seq_lens"],
                           b["cu_k"], out_k, out_v, self.args.batch, self.args.topk,
                           self.head_dim, self.group, hadamard)

    def compact_reference(self, k: torch.Tensor, v: torch.Tensor, out_k: torch.Tensor,
                          out_v: torch.Tensor, hadamard: bool) -> None:
        """The reference of ``compact``."""
        b = self.batch
        reference.compact_gather(k, v, b["req_to_token"], b["req_indices"], b["indices"],
                                 b["seq_lens"], b["cu_k"], out_k, out_v, self.head_dim,
                                 self.group, hadamard)

    def fp8(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The FP8 path's timings and round-trip errors."""
        k_buf, v_buf = self.fp8_k.view(torch.uint8).clone(), self.fp8_v.view(torch.uint8).clone()
        b = self.batch

        def write(k: torch.Tensor, v: torch.Tensor, loc: torch.Tensor) -> None:
            k_buf[loc] = k.to(torch.float8_e4m3fn).view(torch.uint8)
            v_buf[loc] = v.to(torch.float8_e4m3fn).view(torch.uint8)

        timings = {
            "write_prefill": self.timed(
                lambda: write(self.prefill_k, self.prefill_v, self.prefill_loc)),
            "write_decode": self.timed(
                lambda: write(self.k_all[self.decode_loc], self.v_all[self.decode_loc],
                              self.decode_loc)),
            "gather_decode": self.timed(lambda: fp8_compact_gather(
                self.fp8_k, self.fp8_v, b["req_to_token"], b["req_indices"], b["indices"],
                b["seq_lens"], b["cu_k"], self.scratch_k, self.scratch_v, self.args.batch,
                self.args.topk)),
            "gather_extend": self.timed(lambda: (self.fp8_k.index_select(0, self.gather_index),
                                                 self.fp8_v.index_select(0, self.gather_index))),
        }
        errors = {"k": reference.error_stats(self.k_all, reference.fp8_roundtrip(self.k_all)),
                  "v": reference.error_stats(self.v_all, reference.fp8_roundtrip(self.v_all))}
        return timings, errors

    def kv4(self, hadamard: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Checks, errors and timings of the packed cache at one Hadamard setting."""
        args, head_dim, group = self.args, self.head_dim, self.group
        kb = torch.zeros((args.slots, self.heads, self.row), dtype=torch.uint8, device=self.device)
        vb = torch.zeros_like(kb)
        kv4_write(kb, self.locs, self.k_all, group, hadamard=hadamard)
        kv4_write(vb, self.locs, self.v_all, group, hadamard=hadamard)
        checks: dict[str, Any] = {}
        # Correctness. The write against the reference nibble by nibble: exact without the
        # rotation (IEEE division in both); with it the dot's summation order differs from
        # torch's, so a rare element may land one step away (benign); a layout bug would move
        # many nibbles or move them by more than one step. The gathers are exact in FP32.
        n = min(args.slots, args.check_rows)
        for name, buffer, data in (("k", kb, self.k_all), ("v", vb, self.v_all)):
            ref_rows = reference.quantize_rows(data[:n], group, hadamard)
            checks[f"write_{name}_rows_equal"] = bool(torch.equal(buffer[:n], ref_rows))
            for key, value in reference.nibble_stats(buffer[:n], ref_rows, head_dim, group).items():
                checks[f"write_{name}_{key}"] = value
        got = kv4_gather_rows(kb, self.locs[:n], head_dim, group, torch.float32, hadamard)
        want = reference.gather_rows(kb, self.locs[:n], head_dim, group, torch.float32, hadamard)
        checks["gather_rows_max_diff"] = (got - want).abs().max().item()
        ref_k, ref_v = torch.zeros_like(self.scratch_k), torch.zeros_like(self.scratch_v)
        self.compact(kb, vb, self.scratch_k, self.scratch_v, hadamard)
        self.compact_reference(kb, vb, ref_k, ref_v, hadamard)
        checks["compact_k_max_diff"] = (self.scratch_k.float() - ref_k.float()).abs().max().item()
        checks["compact_v_max_diff"] = (self.scratch_v.float() - ref_v.float()).abs().max().item()
        self.compact(self.fp8_k, vb, self.scratch_k, self.scratch_v, hadamard)
        self.compact_reference(self.fp8_k, vb, ref_k, ref_v, hadamard)
        checks["mode_v_k_max_diff"] = (self.scratch_k.float() - ref_k.float()).abs().max().item()
        deq_k = reference.dequantize_rows(kb, head_dim, group, torch.float32, hadamard)
        deq_v = reference.dequantize_rows(vb, head_dim, group, torch.float32, hadamard)
        errors = {"k": reference.error_stats(self.k_all, deq_k),
                  "v": reference.error_stats(self.v_all, deq_v)}
        timings = {
            "write_prefill": self.timed(lambda: (
                kv4_write(kb, self.prefill_loc, self.prefill_k, group, hadamard=hadamard),
                kv4_write(vb, self.prefill_loc, self.prefill_v, group, hadamard=hadamard))),
            "write_decode": self.timed(lambda: (
                kv4_write(kb, self.decode_loc, self.k_all[self.decode_loc], group,
                          hadamard=hadamard),
                kv4_write(vb, self.decode_loc, self.v_all[self.decode_loc], group,
                          hadamard=hadamard))),
            "gather_decode": self.timed(
                lambda: self.compact(kb, vb, self.scratch_k, self.scratch_v, hadamard)),
            "gather_decode_mode_v": self.timed(
                lambda: self.compact(self.fp8_k, vb, self.scratch_k, self.scratch_v, hadamard)),
            "gather_extend": self.timed(lambda: (
                kv4_gather_rows(kb, self.gather_index, head_dim, group, torch.bfloat16, hadamard),
                kv4_gather_rows(vb, self.gather_index, head_dim, group, torch.bfloat16, hadamard))),
        }
        return checks, errors, timings


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run every check and timing; return the JSON document."""
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    doc: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "capability": list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None,
        "torch": torch.__version__, "triton": getattr(triton, "__version__", None),
        "shapes": {"slots": args.slots, "heads": args.heads, "head_dim": args.head_dim,
                   "group": args.group, "layers": args.layers, "batch": args.batch,
                   "topk": args.topk, "seq_len": args.seq_len,
                   "prefill_tokens": args.prefill_tokens, "page": args.page},
        "bytes_per_token": {
            mode: layout.bytes_per_token(layers=args.layers, kv_heads=args.heads,
                                         head_dim=args.head_dim, group=args.group,
                                         indexer_bytes=args.indexer_bytes, mode=mode)
            for mode in layout.MODES
        },
        "checks": {}, "timings": {}, "errors": {},
    }
    # Page-aligned, as the fork rounds the pool: 277,696 (kv) and 210,688 (v) from 169,728.
    doc["pool_tokens_at_fp8_pool"] = {
        mode: int(args.fp8_pool_tokens * doc["bytes_per_token"]["off"]
                  / doc["bytes_per_token"][mode]) // args.page * args.page
        for mode in layout.MODES
    }
    bench = Bench(args, device)
    doc["timings"]["fp8"], doc["errors"]["fp8"] = bench.fp8()
    for hadamard in (False, True):
        tag = f"hadamard{int(hadamard)}"
        doc["checks"][tag], doc["errors"][tag], doc["timings"][tag] = bench.kv4(hadamard)
    doc["ok"] = all(
        c[f"write_{name}_nibble_diff_max"] <= 1
        and c[f"write_{name}_dequant_diff_max_steps"] <= 1.0 + 1e-3
        and c[f"write_{name}_nibble_diff_share"] <= args.max_nibble_diff_share
        for c in doc["checks"].values() for name in ("k", "v")
    ) and all(
        c["gather_rows_max_diff"] == 0.0
        and c["compact_k_max_diff"] <= args.bf16_tolerance
        and c["compact_v_max_diff"] <= args.bf16_tolerance
        and c["mode_v_k_max_diff"] == 0.0
        for c in doc["checks"].values()
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, run, write the JSON, print a summary; 0 when every check passed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the JSON report")
    parser.add_argument("--cpu", action="store_true", help="the reference path on CPU")
    parser.add_argument("--smoke", action="store_true",
                        help="BRIEF A8's CPU smoke: --cpu at small shapes (1,024 slots, 2 streams "
                             "of 200 tokens, top-k 64, 2 iterations), about 20 s")
    parser.add_argument("--slots", type=int, default=169_728 + 64,
                        help="pool slots (revision 15's estimated pool plus the padding page)")
    parser.add_argument("--heads", type=int, default=1, help="KV heads per GPU (TP 4)")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--group", type=int, default=layout.DEFAULT_GROUP)
    parser.add_argument("--layers", type=int, default=12,
                        help="QSA layers, for the bytes per token")
    parser.add_argument("--indexer-bytes", type=int, default=768,
                        help="the indexer cache per token (BF16, dim 128, compressed 4x)")
    parser.add_argument("--fp8-pool-tokens", type=int, default=169_728,
                        help="the FP8 pool the token estimates are scaled from")
    parser.add_argument("--batch", type=int, default=3, help="decode streams")
    parser.add_argument("--topk", type=int, default=2051, help="selected tokens per row")
    parser.add_argument("--seq-len", type=int, default=56_576, help="context of each stream")
    parser.add_argument("--prefill-tokens", type=int, default=2048, help="one prefill chunk")
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--check-rows", type=int, default=4096)
    parser.add_argument("--bf16-tolerance", type=float, default=0.0,
                        help="allowed max difference of the BF16 scratch against the reference")
    parser.add_argument("--max-nibble-diff-share", type=float, default=0.005,
                        help="allowed share of written nibbles one step from the reference's "
                             "(the rotation's summation order; 0 without it)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.smoke:
        args.cpu = True
        args.slots, args.seq_len, args.topk, args.batch = 1024, 200, 64, 2
        args.prefill_tokens, args.iters, args.warmup, args.check_rows = 32, 2, 1, 100
    doc = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps({"ok": doc["ok"], "device": doc["device"],
                      "bytes_per_token": doc["bytes_per_token"],
                      "errors": doc["errors"], "checks": doc["checks"],
                      "gather_decode_us": {k: v["gather_decode"]["median_us"]
                                           for k, v in doc["timings"].items()}}, indent=1))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
