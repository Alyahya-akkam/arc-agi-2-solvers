"""Run the reference rank-256 solver on a per-puzzle H-merged base, restoring it exactly.

H emits rank-16 attention factors from demonstrations only. Its factors never replace the
live rank-256 adapter: the original fit resets and trains that adapter as usual. Every base
write is covered by a CPU snapshot and a finally restore, including failed fit/decode calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
REFERENCE_TARGETS = (*ATTENTION_TARGETS, "gate_proj", "up_proj", "down_proj",
                     "embed_tokens", "lm_head")
MODEL_ID = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"


def checkpoint_digest(path: str | os.PathLike[str], expected: str) -> str:
    """Check the checkpoint file's SHA-256 before any checkpoint deserialization."""
    if not isinstance(expected, str) or len(expected) != 64 or any(
        c not in "0123456789abcdef" for c in expected
    ):
        raise ValueError("hyper_sha256 must be 64 lowercase hexadecimal characters")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"H checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def check_checkpoint(path: str | os.PathLike[str], expected: str) -> dict[str, Any]:
    """Validate the pinned H on CPU, without loading a base model or initializing CUDA."""
    digest = checkpoint_digest(path, expected)
    from arc26.hyper.hypernet import BaseShapes, Hypernet, HypernetConfig
    from arc26.hyper.train import EncoderSettings, load_checkpoint

    payload = load_checkpoint(Path(path))
    cfg = payload["config"]
    if cfg.get("model_id") != MODEL_ID:
        raise ValueError("checkpoint model_id differs from the reference base")
    if type(payload.get("step")) is not int or payload["step"] <= 0:
        raise ValueError("checkpoint step must be a positive integer")
    scope = HypernetConfig(**cfg["hypernet"])
    if (scope.rank != 16 or scope.alpha != 32 or scope.embed_dim != 2560
            or set(scope.target_modules) != set(ATTENTION_TARGETS)):
        raise ValueError("checkpoint requires rank 16, alpha 32, width 2560, q/k/v/o scope")
    encoder = EncoderSettings(**cfg["encoder"])
    if (encoder.layer != 24 or encoder.pool != "output" or encoder.n_perm != 1
            or encoder.view_seed != 2):
        raise ValueError("checkpoint encoder must use layer 24, output pooling, eight views")
    # Fixed reference architecture; meta construction checks all parameter names/shapes
    # without allocating H a second time or loading any base-model weights.
    base = BaseShapes(36, 2560, 9728, 32, 8, 128)
    with torch.device("meta"):
        shapes = Hypernet(scope, base).state_dict()
    state = payload["hypernet"]
    if set(state) != set(shapes):
        raise ValueError("checkpoint H parameter names do not match its architecture")
    parameters = 0
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or not tensor.is_floating_point() or tensor.shape != shapes[name].shape):
            raise ValueError(f"checkpoint H parameter shape/device/dtype mismatch: {name}")
        for chunk in tensor.detach().reshape(-1).split(1 << 20):
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError(f"nonfinite checkpoint H parameter: {name}")
        parameters += tensor.numel()
    return {"ok": True, "checkpoint_sha256": digest, "step": payload["step"],
            "kind": payload.get("kind"), "model_id": cfg["model_id"], "layers": 36,
            "rank": scope.rank, "alpha": scope.alpha,
            "target_modules": list(scope.target_modules), "parameter_count": parameters,
            "parameter_tensors": len(state), "device": "cpu"}


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Refuse changes to the original fit's rank, nine targets, or refinement behavior."""
    targets = cfg.get("lora_targets", ())
    if (type(cfg.get("lora_r")) is not int or cfg["lora_r"] != 256
            or not isinstance(targets, (list, tuple))
            or len(targets) != 9 or set(targets) != set(REFERENCE_TARGETS)):
        raise ValueError("H submission requires the original rank-256, nine-target adapter")
    if cfg.get("holdout") is not None or cfg.get("adapters") is not None:
        raise ValueError("H submission uses the original fit/decode without refinement arms")
    if type(cfg.get("workers")) is not int or cfg["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    for key in ("hyper_checkpoint", "hyper_sha256", "hyper_log_dir"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty string")


def _digest(state: Mapping[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class HBaseMerge:
    """Validate emitted factors and hold CPU copies of only their attention base weights."""

    def __init__(self, model: Any, adapter: Any) -> None:
        if (type(adapter.rank) is not int or adapter.rank != 16
                or isinstance(adapter.alpha, bool) or not math.isfinite(adapter.alpha)
                or adapter.alpha != 32
                or len(adapter.target_modules) != 4
                or set(adapter.target_modules) != set(ATTENTION_TARGETS)):
            raise ValueError("H must emit rank 16, alpha 32, exactly q/k/v/o attention targets")
        layers = getattr(model.config, "num_hidden_layers", None)
        if type(layers) is not int or layers < 1:
            raise ValueError("model.config.num_hidden_layers must be a positive integer")
        names = [f"base_model.model.model.layers.{i}.self_attn.{target}"
                 for i in range(layers) for target in ATTENTION_TARGETS]
        expected = {f"{name}.lora_{side}.weight" for name in names for side in ("A", "B")}
        if set(adapter.state) != expected:
            raise ValueError("H factors do not cover exactly every attention projection and layer")
        modules = dict(model.named_modules())
        self.entries: list[tuple[str, Any, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.scale = float(adapter.alpha) / math.sqrt(adapter.rank)
        # Validate everything before allocating snapshots or changing any base weight.
        for name in names:
            module = modules.get(name)
            base = getattr(module, "base_layer", None)
            if (not isinstance(base, torch.nn.Linear)
                    or getattr(module, "merged", False)
                    or getattr(module, "fan_in_fan_out", False)
                    or any(getattr(module, "use_dora", {}).values())):
                raise ValueError(f"{name}: expected an unmerged plain Linear PEFT layer")
            weight = base.weight
            if (weight.dtype != torch.bfloat16 or weight.requires_grad
                    or getattr(weight, "quant_state", None) is not None
                    or not bool(torch.isfinite(weight).all())):
                raise ValueError(f"{name}: base must be frozen finite unquantized bfloat16")
            a, b = (adapter.state[f"{name}.lora_{side}.weight"] for side in ("A", "B"))
            if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                    or tuple(a.shape) != (16, weight.shape[1])
                    or tuple(b.shape) != (weight.shape[0], 16)
                    or not a.is_floating_point() or not b.is_floating_point()
                    or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())):
                raise ValueError(f"{name}: incomplete, nonfinite, or incorrectly shaped factors")
            # Match the validated merge path: round factors to bf16 BEFORE the fp32 product.
            a = a.detach().to(device=weight.device, dtype=torch.bfloat16)
            b = b.detach().to(device=weight.device, dtype=torch.bfloat16)
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                raise ValueError(f"{name}: factors overflow bfloat16")
            self.entries.append((name, module, weight, a, b))
        self.digest = _digest(adapter.state)
        self.copies = [weight.detach().to(device="cpu", copy=True)
                       for _, _, weight, _, _ in self.entries]
        self.active = False
        self.restored = False

    def add(self) -> dict[str, Any]:
        """Add bf16-rounded H factors using fp32 multiplication and addition, once."""
        if self.active or self.restored:
            raise RuntimeError("an H merge can only be applied once")
        self.active = True  # Set before the first write, including a partially failed merge.
        delta_sq = 0.0
        effective_delta_sq = 0.0
        changed_elements = 0
        with torch.no_grad():
            for (_, _, weight, a, b), original in zip(self.entries, self.copies, strict=True):
                delta = (b.float() @ a.float()) * self.scale
                original_fp32 = original.to(weight.device).float()
                merged = original_fp32 + delta
                rounded = merged.to(weight.dtype)
                if (not bool(torch.isfinite(merged).all())
                        or not bool(torch.isfinite(rounded).all())):
                    raise ValueError("nonfinite merged H weight")
                norm = float(torch.linalg.vector_norm(delta.double()))
                delta_sq += norm * norm
                effective_delta = rounded.float() - original_fp32
                changed_elements += int(torch.count_nonzero(effective_delta))
                effective_norm = float(torch.linalg.vector_norm(effective_delta.double()))
                effective_delta_sq += effective_norm * effective_norm
                weight.copy_(rounded)
        if not math.isfinite(delta_sq):
            raise ValueError("nonfinite H merge norm")
        if delta_sq == 0:
            raise ValueError("zero H merge norm")
        if not math.isfinite(effective_delta_sq):
            raise ValueError("nonfinite effective H merge norm")
        if changed_elements == 0:
            raise ValueError("H merge changes no bfloat16 base elements")
        return {"layers": len(self.entries), "delta_norm": math.sqrt(delta_sq),
                "changed_elements": changed_elements,
                "effective_delta_norm": math.sqrt(effective_delta_sq),
                "scale": self.scale, "factor_dtype": "bfloat16", "compute_dtype": "float32",
                "snapshot_bytes": sum(t.numel() * t.element_size() for t in self.copies)}

    def restore(self, model: Any) -> None:
        """Restore every saved base exactly, never subtract; fail if fit replaced a weight."""
        modules = dict(model.named_modules())
        replaced = []
        with torch.no_grad():
            for (name, module, weight, _, _), original in zip(
                self.entries, self.copies, strict=True
            ):
                current = modules.get(name)
                if (current is not module or getattr(getattr(current, "base_layer", None),
                                                    "weight", None) is not weight):
                    replaced.append(name)
                weight.copy_(original)
        if any(not torch.equal(weight.detach().cpu(), original)
               for (_, _, weight, _, _), original in zip(self.entries, self.copies, strict=True)):
            raise RuntimeError("H base restoration failed exact equality")
        if replaced:
            raise RuntimeError(f"fit replaced H target weights: {replaced}")
        self.active = False
        self.restored = True


def solve_one(
    loaded: Any, provider: Any, key: str, rank: int, end_time: float,
    fit_fn: Callable[..., Any], decode_fn: Callable[..., Any], record: dict[str, Any],
) -> Any:
    """Emit H from demonstrations, merge, run original fit/decode, and restore in finally."""
    from arc26.loop import DemoSet, Pair

    start = time.time()
    task = loaded.arc_test_set.queries[key]
    # Deliberately omit even test inputs from the state handed to H.
    demos = DemoSet(key, tuple(Pair(p["input"], p["output"]) for p in task["train"]), None, ())
    if not callable(getattr(loaded.model, "disable_adapter", None)):
        raise ValueError("H encoding requires a PEFT model with disable_adapter")
    if any(getattr(m, "merged", False) for m in loaded.model.modules()):
        raise ValueError("cannot encode H with a merged live adapter")
    began = time.perf_counter()
    # Provider.emit does not use _live_state or install rank-16 factors into rank-256 slots.
    with loaded.model.disable_adapter():
        adapter = provider.emit(demos)
    record["emit_s"] = time.perf_counter() - began
    began = time.perf_counter()
    merge = HBaseMerge(loaded.model, adapter)
    record["snapshot_s"] = time.perf_counter() - began
    if loaded.timing_logger is not None:
        loaded.timing_logger.record(key, "h_snapshot", record["snapshot_s"])
    record["h_digest"] = merge.digest
    record["restored"] = False
    try:
        began = time.perf_counter()
        record["merge"] = merge.add()
        record["merge_s"] = time.perf_counter() - began
        if loaded.timing_logger is not None:
            loaded.timing_logger.record(key, "h_merge", record["merge_s"])
        record["h_used"] = True
        model, puzzle_ds = fit_fn(
            model=loaded.model, default_weights=loaded.default_weights,
            arc_test_set=loaded.arc_test_set, key=key, formatter=loaded.formatter,
            max_seq_length=loaded.max_seq_length, tokenizer=loaded.tokenizer,
            collator=loaded.collator, train_args=loaded.train_args,
            timing_logger=loaded.timing_logger, rank=rank,
        )
        loaded = loaded._replace(model=model)
        decode_fn(
            puzzle_ds=puzzle_ds, formatter=loaded.formatter,
            max_seq_length=loaded.max_seq_length, max_new_tokens=loaded.max_new_tokens,
            start_time=start, end_time=end_time, rank=rank, key=key,
            tokenizer=loaded.tokenizer, model=loaded.model, max_score=loaded.max_score,
            timing_logger=loaded.timing_logger, dir_outputs=loaded.dir_outputs,
        )
        record["decoded"] = True
        return loaded
    finally:
        began = time.perf_counter()
        merge.restore(loaded.model)
        record["restore_s"] = time.perf_counter() - began
        record["restored"] = merge.restored
        if loaded.timing_logger is not None:
            loaded.timing_logger.record(key, "h_restore", record["restore_s"])


def _log(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def worker(rank: int, queue: Any, end_time: float, cfg: dict[str, Any]) -> None:
    """Load the original solver and H once, then solve queued puzzles without an H fallback."""
    from arc26 import solver
    from arc26.decode import decode_puzzle
    from arc26.hyper.providers import HyperProvider
    from arc26.ttt import fit

    path = Path(cfg["hyper_log_dir"]) / f"h-rank{rank}.jsonl"
    used = completed = 0
    try:
        digest = checkpoint_digest(cfg["hyper_checkpoint"], cfg["hyper_sha256"])
        loaded = solver.load(cfg)
        provider = HyperProvider.from_checkpoint(
            cfg["hyper_checkpoint"], model=loaded.model, tokenizer=loaded.tokenizer,
            cache_root=Path(cfg.get("hyper_cache_dir", "/tmp/arc26-h-cache")) / f"rank{rank}",
            timing_logger=loaded.timing_logger,
        )
        _log(path, {"event": "loaded", "rank": rank, "checkpoint_sha256": digest})
        while time.time() < end_time:
            key = queue.get()
            if key is None:
                break
            record: dict[str, Any] = {"event": "puzzle", "puzzle_id": key, "rank": rank,
                                      "checkpoint_sha256": digest, "h_used": False,
                                      "decoded": False, "restored": False}
            began = time.perf_counter()
            try:
                loaded = solve_one(
                    loaded, provider, key, rank, end_time, fit, decode_puzzle, record
                )
                completed += 1
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                used += int(record["h_used"])
                record.update(h_used_count=used, seconds=time.perf_counter() - began)
                _log(path, record)
                if loaded.timing_logger is not None:
                    loaded.timing_logger.record(key, "puzzle", record["seconds"])
    except Exception as error:
        _log(path, {"event": "fatal", "rank": rank,
                    "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        _log(path, {"event": "summary", "rank": rank, "h_used_count": used,
                    "completed": completed})


def local_worker(rank: int, queue: Any, end_time: float, cfg: dict[str, Any]) -> None:
    """Pin a worker to its GPU and preserve the reference's serialized Unsloth imports."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")
    markers = Path(cfg["marker_dir"])
    if rank > 0:
        while not (markers / f"h-worker{rank - 1}").is_file():
            if time.time() >= end_time:
                raise TimeoutError("deadline while waiting for previous worker's imports")
            time.sleep(1)
    from arc26 import decode, timing, ttt  # noqa: F401

    (markers / f"h-worker{rank}").write_text("Ok", encoding="utf-8")
    if cfg.get("timing_dir") is not None:
        timing.configure(cfg["timing_dir"], f"rank{rank}")
    worker(rank, queue, end_time, cfg)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one worker per configured GPU over sorted challenges, never evaluation solutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--config", help="original solver JSON plus H paths/hash")
    parser.add_argument("--check-only", action="store_true", help="CPU checkpoint validation only")
    parser.add_argument("--checkpoint")
    parser.add_argument("--sha256")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.check_only:
        if not args.checkpoint or not args.sha256 or args.config or args.end_time is not None:
            parser.error("--check-only requires --checkpoint/--sha256 and excludes solver options")
        try:
            report = check_checkpoint(args.checkpoint, args.sha256)
        except Exception as error:
            report = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        text = json.dumps(report, allow_nan=False, sort_keys=True) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0 if report["ok"] else 1
    if args.config is None or args.end_time is None:
        parser.error("solver mode requires --config and --end-time")
    if args.checkpoint or args.sha256 or args.report:
        parser.error("--checkpoint/--sha256/--report require --check-only")
    cfg = json.loads(args.config)
    validate_config(cfg)
    if not math.isfinite(args.end_time) or args.end_time <= time.time():
        raise ValueError("end-time must be a future finite absolute timestamp")
    checkpoint_digest(cfg["hyper_checkpoint"], cfg["hyper_sha256"])
    # Match solver.load's rerun condition exactly; no development subset enters a rerun.
    rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    name = "arc-agi_test_challenges.json" if rerun else "arc-agi_evaluation_challenges.json"
    data = json.loads((Path(cfg["data_dir"]) / name).read_text(encoding="utf-8"))
    keys = [key for key in sorted(data)
            if rerun or cfg.get("subset") is None or key in cfg["subset"]]
    Path(cfg["marker_dir"]).mkdir(parents=True, exist_ok=True)
    for rank in range(cfg["workers"]):
        (Path(cfg["marker_dir"]) / f"h-worker{rank}").unlink(missing_ok=True)
    import torch.multiprocessing as mp

    with mp.Manager() as manager:
        queue = manager.Queue()
        for key in keys:
            queue.put(key)
        for _ in range(cfg["workers"]):
            queue.put(None)
        mp.spawn(local_worker, args=(queue, args.end_time, cfg), nprocs=cfg["workers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
