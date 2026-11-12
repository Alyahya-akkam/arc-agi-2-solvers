"""P3.6: scripts/train_throughput.py on the CPU, its configs, its build, and its notebook.

The base is the tiny random Qwen3 and the stand-in tokenizer of arc26.hyper.train; nothing is
downloaded. The smoke runs the script end to end in spawned gloo processes; the out-of-memory,
stall, and kill paths run sessions in this process (``run_session`` with one process), where a
patched ``train_step`` can raise. The build and the simulated script notebook reuse the helpers
of tests/test_build_submission.py and tests/test_notebook_local.py. Without torch, transformers,
or safetensors the module is skipped (CI's main job; the hyper-cpu job runs it). Run with
CUDA_VISIBLE_DEVICES="" (BRIEF A0).
"""

import ast
import dataclasses
import importlib.util
import inspect
import json
import math
import os
import random
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import test_build_submission as tbs  # noqa: E402
import test_notebook_local as nl  # noqa: E402

from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.episodes import PAIR_OVERHEAD_TOKENS, PermutationTable, grid_tokens  # noqa: E402
from arc26.hyper.hypernet import HypernetConfig  # noqa: E402
from arc26.nvarc.arc_loader import QwenFormatter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "train_throughput.py"
CONFIG = REPO / "configs" / "train_throughput.yaml"
BUILD_CONFIG = REPO / "configs" / "script_P3.6-01.yaml"
BRIEF = REPO / "docs" / "prerun" / "P3.6-01.md"
LONG_RUN = REPO / "configs" / "train_hypernet.yaml"
TOY = REPO / "configs" / "train_hypernet_toy.yaml"
HYPERNET = REPO / "configs" / "hypernet.yaml"
P3502_BUILD = REPO / "configs" / "script_P3.5-02.yaml"


def load_script(name):
    """Load a script as a module of its own name: spawned sessions import it by that name."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tt = load_script("train_throughput")
train_hypernet = load_script("train_hypernet")
kaggle_ops = load_script("kaggle_ops")


def smoke_cfg(tmp_path, **changes):
    cfg = tt.load_config(CONFIG, True, {"out_dir": str(tmp_path / "run_logs")})
    return dataclasses.replace(cfg, **changes)


def point(name, tokens=None, kind="fixed", gpus=1, accum=1, warm=1, timed=2):
    return tt.Point(name, kind, gpus, accum, tokens, warm, timed)


def plan_for(cfg, points, index=0, world=1, deadline=math.inf):
    return tt.SessionPlan(
        index=index,
        world=world,
        points=tuple(points),
        train=tt.train_config(cfg, None, world),
        out_dir=cfg.out_dir,
        started=time.time(),
        point_deadline=deadline,
        max_step_s=cfg.limits.max_step_s,
        projection=cfg.projection,
        seed=cfg.seed,
    )


class InlineSession:
    """A one-process session run to its end in this process, so patches reach it."""

    def __init__(self, plan):
        assert plan.world == 1
        self.code = tt.run_session(0, 1, plan, 0)

    def wait(self, timeout):
        return "ok" if self.code == 0 else f"failed: process 0 exited with code {self.code}"

    def kill(self):
        raise AssertionError("an inline session has ended before it is watched")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --- the configs ---------------------------------------------------------------------------------


def test_config_validates_in_both_modes_and_measures_the_briefs_points():
    cfg = tt.load_config(CONFIG, False, {})
    fixed = {(p.gpus, p.tokens) for p in cfg.points if p.kind == "fixed"}
    # B5 P3.6's lengths (2k, 5k, 9.5k), the 8192 cap, the short end, on 4 GPUs as P3.5-01 ran;
    # the cap and 512 in one process too (the task brief; DDP's cost).
    assert {(4, t) for t in (512, 1024, 2048, 5120, 8192, 9728)} <= fixed
    assert {(1, 8192), (1, 512)} <= fixed
    assert all(p.accum == 1 for p in cfg.points if p.kind == "fixed")
    (mix,) = [p for p in cfg.points if p.kind == "mix"]
    (hyper,) = [p for p in cfg.points if p.kind == "hypernet"]
    assert (mix.gpus, mix.accum) == (cfg.projection.gpus, cfg.projection.accum) == (4, 8)
    assert hyper.gpus == 1 and hyper.timed_steps >= 50
    assert (cfg.decision.cap_tokens, cfg.decision.longer_tokens) == (8192, 9728)
    assert cfg.decision.hypernet_budget_ms == 50 and cfg.decision.headroom_fraction == 0.10
    assert cfg.model_dir is None and cfg.input_root == "/kaggle/input"
    assert cfg.out_dir == "/kaggle/working/run_logs" and cfg.train["device"] == "cuda"
    # Consecutive points of one size share a session: two planned, 4 GPUs then 1.
    sizes = [p.gpus for p in cfg.points]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == 2
    smoke = tt.load_config(CONFIG, True, {})
    assert smoke.train["device"] == "cpu" and smoke.model_dir is None
    assert max(p.gpus for p in smoke.points) == 2 and Path(smoke.out_dir).is_absolute()
    assert {p.kind for p in smoke.points} == set(tt.KINDS)


def test_the_projection_is_p3_4s_training_split_and_both_quota_readings():
    cfg = tt.load_config(CONFIG, False, {})
    projection = cfg.projection
    # P3.4 build 2's episodes_stats.md, train, all sources (docs/experiments/P3.4.md).
    assert projection.counts == (51762, 26389, 12586, 5775, 2181, 690, 188, 103)
    assert sum(projection.counts) == projection.episodes == 99674
    assert projection.bin_tokens == 1024 and projection.mean_tokens == 1348.4
    assert projection.rates == (2.0, 1.0)
    full = yaml.safe_load(LONG_RUN.read_text())
    assert (projection.gpus, projection.accum) == (full["gpus"], full["accum"])
    # The bin-uniform draw runs 1.5% long against the measured mean: the error is conservative.
    assert tt.histogram_mean(projection) == pytest.approx(1368.46, abs=0.01)


def test_the_train_section_is_the_long_runs_step():
    """The timed step is configs/train_hypernet.yaml's: every key the step reads is copied, and
    only max_tokens differs (the 9.5k point is above the long run's 8192 cap)."""
    train = yaml.safe_load(CONFIG.read_text())["train"]
    full = yaml.safe_load(LONG_RUN.read_text())
    assert set(train) == set(tt.TRAIN_KEYS)
    differs = {key for key in tt.TRAIN_KEYS if train[key] != full[key]}
    assert differs == {"max_tokens"}
    assert (full["max_tokens"], train["max_tokens"]) == (8192, 9728)
    assert HypernetConfig(**train["hypernet"]) == HypernetConfig.from_yaml(HYPERNET)
    cfg = tt.train_config(tt.load_config(CONFIG, False, {}), "/models/base", 4, 8)
    assert (cfg.gpus, cfg.accum, cfg.model_dir, cfg.run_id) == (4, 8, "/models/base", "P3.6-01")
    assert cfg.gradient_checkpointing and cfg.dtype == "bfloat16" and cfg.loss_includes_eos


def test_every_config_value_has_a_comment():
    previous = ""
    for line in CONFIG.read_text().splitlines():
        if re.match(r"\s*[a-z_]+:", line):
            assert "#" in line or previous.lstrip().startswith("#"), line
        if line.strip():
            previous = line


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw["points"].append({"name": "x", "kind": "warp", "gpus": 1}), "kind"),
        (lambda raw: raw["points"].append(dict(raw["points"][0])), "names must be distinct"),
        (lambda raw: raw["points"][0].pop("tokens"), "tokens must be an integer"),
        (lambda raw: raw["points"][0].update(tokens=9729), "train.max_tokens (9728)"),
        (lambda raw: raw["points"][0].update(tokens=21), "no episode in the notebook format"),
        (lambda raw: raw["points"][-1].update(gpus=2), "runs on 1 GPU"),
        (lambda raw: raw["points"][4].update(accum=4), "a mix point measures the projected"),
        (lambda raw: raw["points"][0].update(seed=1), "unknown keys ['seed']"),
        (lambda raw: raw["train"].pop("clip"), "missing keys ['clip']"),
        (lambda raw: raw["train"].update(kl_weight=0.5), "kl_weight must be 0"),
        (lambda raw: raw["projection"].update(counts=[]), "counts must be a non-empty list"),
        (lambda raw: raw["projection"].update(bin_tokens=2048), "the histogram reaches"),
        (lambda raw: raw["limits"].update(reserve_s=2400), "reserve_s must be below"),
        (lambda raw: raw["decision"].update(cap_tokens=10000), "decision.cap_tokens"),
        (lambda raw: raw["smoke"].update(nonsense=1), "smoke.nonsense is not a config key"),
    ],
)
def test_config_errors_are_refused(tmp_path, change, message):
    raw = yaml.safe_load(CONFIG.read_text())
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(tt.ConfigError, match=re.escape(message)):
        tt.load_config(path, False, {})


def test_command_line_errors_exit_2(tmp_path, capsys):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["limits"]["max_sessions"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert tt.main(["--config", str(path), "--smoke"]) == tr.EXIT_CONFIG
    assert "max_sessions must be an integer >= 1" in capsys.readouterr().err
    assert tt.main(["--config", str(CONFIG), "--smoke", "--model-dir", "x"]) == tr.EXIT_CONFIG
    assert tt.main(["--config", str(CONFIG), "--smoke", "--out-dir", ""]) == tr.EXIT_CONFIG


# --- synthetic episodes --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def formatter():
    return QwenFormatter(tr.standin_tokenizer())


@pytest.mark.parametrize("tokens", [20, 64, 511, 512, 1024, 2048, 3331, 5120, 8192, 9330, 9728])
def test_an_episode_has_exactly_the_requested_tokens_in_the_notebook_format(formatter, tokens):
    table = PermutationTable.seeded(tt.TABLE_SIZE, 0)
    record = tt.synthetic_record(tokens, "synthetic/t/0", np.random.default_rng(0), table)
    sequence = tr.build_sequence(record, table, formatter, 9728)
    assert sequence.input_ids.numel() == record.tokens == tokens
    assert record.n_demos >= 1 and len(record.queries) == 1
    for grid in (g for pair in record.pairs for g in pair):
        rows, cols = grid.shape
        assert 1 <= cols <= rows <= 30  # the read orientation is the longer one
    # The query's output, with its closing token, is the loss's target.
    assert len(sequence.target) == grid_tokens(*record.queries[0][1].shape) + 1


def test_the_long_points_are_large_grids_as_b3s_worst_case():
    """8192 and 9.5k come out as B3's worst case does: large grids near 30 by 30 (the last ones
    take the remainder); 9330 tokens is exactly four 30 by 30 demonstrations and a 30 by 30
    query."""
    assert tt.episode_shapes(9330) == ((30, 30),) * 10
    assert len(tt.episode_shapes(8192)) == 10 and len(tt.episode_shapes(9728)) == 12
    for tokens in (8192, 9728):
        shapes = tt.episode_shapes(tokens)
        assert all(rows >= 20 and cols >= 20 for rows, cols in shapes)
        assert sum(rows >= 28 for rows, _ in shapes) >= 8


def test_every_length_but_three_short_ones_can_be_built():
    infeasible = [t for t in range(tt.MIN_EPISODE_TOKENS, 9729) if not tt.feasible(t)]
    assert infeasible == [21, 23, 25]
    for tokens in range(tt.MIN_EPISODE_TOKENS, 9729, 97):
        if tt.feasible(tokens):
            shapes = tt.episode_shapes(tokens)
            grids = sum(grid_tokens(r, c) for r, c in shapes)
            assert grids + PAIR_OVERHEAD_TOKENS * len(shapes) // 2 == tokens
    with pytest.raises(ValueError, match="exactly 21 tokens"):
        tt.episode_shapes(21)


def test_lengths_are_drawn_from_the_histogram():
    projection = tt.load_config(CONFIG, False, {}).projection
    draws = tt.draw_lengths(projection, np.random.default_rng(1), 200_000)
    assert draws.min() >= tt.MIN_EPISODE_TOKENS and draws.max() < 8192
    shares = np.bincount(draws // 1024, minlength=8) / len(draws)
    expected = np.asarray(projection.counts) / sum(projection.counts)
    assert np.abs(shares - expected).max() < 0.005
    assert draws.mean() == pytest.approx(tt.histogram_mean(projection), rel=0.01)
    lengths = tt.mix_lengths(projection, 0, "mix", 500)
    assert lengths == tt.mix_lengths(projection, 0, "mix", 500)
    assert all(tt.feasible(t) for t in lengths)


# --- the timed step is the training step ---------------------------------------------------------


def test_the_throughput_worker_runs_the_training_workers_step(tmp_path):
    """ThroughputWorker replaces only setup: its model, H, optimizer, schedule, and random state
    equal the training worker's after setup, and the same episodes give the same H, bitwise."""
    assert tt.ThroughputWorker.train_step is tr._Worker.train_step
    assert tt.ThroughputWorker.micro_step is tr._Worker.micro_step
    cfg = train_hypernet.load_config(
        TOY, True, {"out_dir": str(tmp_path / "logs"), "ckpt_dir": str(tmp_path / "ckpt")}
    )
    cfg = dataclasses.replace(cfg, gpus=1, accum=1)
    real = tr._Worker(0, 1, cfg, None, time.time())
    real.setup()
    states = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    throughput = smoke_cfg(tmp_path)
    plan = dataclasses.replace(plan_for(throughput, []), train=cfg)
    mine = tt.ThroughputWorker(0, 1, cfg, plan)
    mine.setup()
    assert torch.equal(torch.get_rng_state(), states[0]) and random.getstate() == states[1]
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), states[2], strict=True))
    for a, b in ((real.model, mine.model), (real.hypernet, mine.hypernet)):
        sa, sb = a.state_dict(), b.state_dict()
        assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert real.keys == mine.keys and mine.model.is_gradient_checkpointing
    assert real.attn == mine.attn and type(real.h_train) is type(mine.h_train)
    for ga, gb in zip(real.optimizer.param_groups, mine.optimizer.param_groups, strict=True):
        assert {k: v for k, v in ga.items() if k != "params"} == {
            k: v for k, v in gb.items() if k != "params"
        }
        assert [p.shape for p in ga["params"]] == [p.shape for p in gb["params"]]
    assert vars(real.scheduler.lr_lambdas[0]) == vars(mine.scheduler.lr_lambdas[0])
    # The same synthetic episodes and z_bar through both workers' (one) train_step.
    records, offset = mine._records(point("p", 64), 3)
    for worker in (real, mine):
        worker.train_records, worker.order = records, tt.SlotOrder(offset)
        worker.embeddings, worker.table = mine.embeddings, mine.table
        torch.set_rng_state(states[0])
        for _ in range(3):
            assert worker.train_step() == (0, "")
    sa, sb = real.hypernet.state_dict(), mine.hypernet.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert real.step == mine.step == 3


# Calls of _Worker.setup that ThroughputWorker.setup copies: seeds, the base, injection,
# checkpointing, the formatter, H, DDP, the optimizer, and the schedule.
COPIED_CALLS = {
    "torch.manual_seed", "random.seed", "np.random.seed", "load_base", "wrap_targets",
    "model.gradient_checkpointing_enable", "QwenFormatter", "Hypernet", "BaseShapes.from_config",
    "DistributedDataParallel", "torch.optim.AdamW", "param_groups", "LambdaLR", "WarmupCosine",
}  # fmt: skip


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def setup_tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def copied_calls(tree):
    """Every watched call with its arguments, in source order."""
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and dotted(node.func) in COPIED_CALLS
    ]  # fmt: skip
    return [ast.dump(n) for n in sorted(calls, key=lambda n: (n.lineno, n.col_offset))]


def assigned(tree, target):
    """The expression assigned to ``target`` (``self.h_train``), annotated or not."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(dotted(t) == target for t in targets):
            found.append(node.value)
    assert len(found) == 1, (target, found)
    return ast.dump(found[0])


def guard_of(tree, call):
    """The test of the ``if`` around a call (checkpointing is enabled only when configured)."""
    guards = [
        node.test for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(n, ast.Call) and dotted(n.func) == call for n in ast.walk(node))
    ]  # fmt: skip
    assert len(guards) == 1, (call, guards)
    return ast.dump(guards[0])


def test_the_subclass_replaces_only_setup_and_copies_its_calls_verbatim():
    """A later change to _Worker.setup's DDP, checkpointing, seeds, H, or optimizer arguments
    fails here unless ThroughputWorker.setup follows it; the one-process equivalence test above
    cannot see DDP's arguments."""
    assert tt.ThroughputWorker.__bases__ == (tr._Worker,)
    own = {n for n in vars(tt.ThroughputWorker) if not n.startswith("__") or n == "__init__"}
    assert {n for n in own if hasattr(tr._Worker, n)} == {"__init__", "setup"}
    real, mine = setup_tree(tr._Worker.setup), setup_tree(tt.ThroughputWorker.setup)
    assert copied_calls(mine) == copied_calls(real)
    assert len(copied_calls(real)) == len(COPIED_CALLS)  # each watched call appears once
    assert assigned(mine, "self.h_train") == assigned(real, "self.h_train")
    assert guard_of(mine, "model.gradient_checkpointing_enable") == guard_of(
        real, "model.gradient_checkpointing_enable"
    )
    # The values that matter, spelled out: DDP on this process's GPU, non-reentrant checkpointing.
    ddp = next(n for n in ast.walk(mine) if isinstance(n, ast.Call)
               and dotted(n.func) == "DistributedDataParallel")  # fmt: skip
    assert [k.arg for k in ddp.keywords] == ["device_ids"]
    enable = next(n for n in ast.walk(mine) if isinstance(n, ast.Call)
                  and dotted(n.func) == "model.gradient_checkpointing_enable")  # fmt: skip
    assert ast.literal_eval(enable.keywords[0].value) == {"use_reentrant": False}


def snapshot(total=22, used=None, allocated=8.0, reserved=None, peak_allocated=None,
             peak_reserved=None, retries=0):
    gib = tt.GIB
    reserved = allocated if reserved is None else reserved
    return {
        "device_total": total * gib, "device_used": (used or reserved + 0.5) * gib,
        "allocated": allocated * gib, "reserved": reserved * gib,
        "peak_allocated": (peak_allocated or allocated) * gib,
        "peak_reserved": (peak_reserved or reserved) * gib, "alloc_retries": retries, "ooms": 0,
        "cpu_rss_peak": None,
    }  # fmt: skip


def test_the_cache_is_never_emptied_and_its_inherited_part_is_recorded():
    """A training run never empties the allocator's cache, so no point does: the peak reserved
    memory includes what the earlier points left, recorded as reserved_before, and the headroom
    is taken from it; the retries are the point's own."""
    assert "empty_cache" not in inspect.getsource(tt)
    gib = tt.GIB
    results = [
        {"memory_before": snapshot(allocated=10.0, reserved=12.0, retries=1),
         "memory": snapshot(allocated=10.0, reserved=15.0, used=15.75, peak_allocated=13.0,
                            peak_reserved=15.0, retries=1)},
        {"memory_before": snapshot(allocated=10.0, reserved=11.0),
         "memory": snapshot(allocated=10.0, reserved=14.0, used=14.5, peak_allocated=13.5,
                            peak_reserved=16.0, retries=2)},
    ]  # fmt: skip
    memory = tt.memory_summary(results)
    assert memory["reserved_before"] == 12.0 * gib and memory["resident_before"] == 10.0 * gib
    assert memory["peak_allocated"] == 13.5 * gib and memory["peak_reserved"] == 16.0 * gib
    # Rank 1: 22 - (16.0 + (14.5 - 14.0)) = 5.5 GiB, the tighter one; rank 0 has 22 - 15.75.
    assert memory["headroom"] == 5.5 * gib
    assert memory["headroom_fraction"] == pytest.approx(5.5 / 22)
    assert memory["alloc_retries"] == 2  # rank 0's earlier retry is not this point's
    assert [r["reserved_before"] for r in memory["per_rank"]] == [12.0 * gib, 11.0 * gib]
    assert tt.memory_summary([{"memory_before": {}, "memory": {"device_total": None}}]) is None


# --- out of memory, stalls, kills ----------------------------------------------------------------


def oom_at(tokens, monkeypatch):
    """Make train_step raise CUDA's out-of-memory error on the episodes of one length."""
    original = tt.ThroughputWorker.train_step

    def train_step(self):
        first = next(iter(self.train_records.records.values()))
        if first.tokens == tokens:
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of "
                "22.05 GiB of which 1.10 GiB is free."
            )
        return original(self)

    monkeypatch.setattr(tt.ThroughputWorker, "train_step", train_step)


def test_an_out_of_memory_point_is_a_recorded_result_that_ends_its_session(tmp_path, monkeypatch):
    oom_at(128, monkeypatch)
    cfg = smoke_cfg(tmp_path)
    points = [point("a", 64), point("b", 128), point("c", 256)]
    assert tt.run_session(0, 1, plan_for(cfg, points), 0) == 1
    folder = Path(cfg.out_dir) / "points"
    first = json.loads((folder / "a" / "rank0.json").read_text())
    failed = json.loads((folder / "b" / "rank0.json").read_text())
    assert first["outcome"] == "complete" and len(first["steps"]) == 3
    assert failed["outcome"] == "oom" and failed["steps"] == []
    assert failed["reason"].startswith("OutOfMemoryError: CUDA out of memory. Tried to allocate")
    assert not (folder / "c").exists()
    beat = json.loads((Path(cfg.out_dir) / "sessions" / "s0" / "rank0.json").read_text())
    assert (beat["state"], beat["point"]) == ("failed", "b") and beat["ready"]


@pytest.mark.parametrize(
    ("error", "out_of_memory"),
    [
        (torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB"), True),
        (RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)"),
         True),
        (RuntimeError("NCCL error in: ProcessGroupNCCL.cpp\nCuda failure 'out of memory'"), True),
        (RuntimeError("CUDA error: an illegal memory access was encountered"), False),
        (ValueError("synthetic/p/0: 9800 tokens, above max_tokens 9728"), False),
    ],
)  # fmt: skip
def test_allocation_failures_of_every_kind_count_as_out_of_memory(error, out_of_memory):
    assert tt.is_out_of_memory(error) is out_of_memory


def test_another_error_ends_its_point_and_session_with_its_traceback(tmp_path, monkeypatch):
    def broken(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(tt.ThroughputWorker, "train_step", broken)
    cfg = smoke_cfg(tmp_path)
    assert tt.run_session(0, 1, plan_for(cfg, [point("a", 64), point("b", 128)]), 0) == 1
    result = json.loads((Path(cfg.out_dir) / "points" / "a" / "rank0.json").read_text())
    assert result["outcome"] == "error" and "illegal memory access" in result["reason"]
    assert "Traceback" in result["traceback"]
    assert not (Path(cfg.out_dir) / "points" / "b").exists()


def test_the_run_continues_after_an_out_of_memory_point_in_a_new_session(tmp_path, monkeypatch):
    oom_at(128, monkeypatch)
    cfg = smoke_cfg(tmp_path, points=(point("a", 64), point("b", 128), point("c", 256)))
    cfg = dataclasses.replace(cfg, decision=dataclasses.replace(cfg.decision, cap_tokens=128))
    run = tt.run_points(cfg, None, time.time(), {}, launch=InlineSession)
    assert {name: r["outcome"] for name, r in run.records.items()} == {
        "a": "complete", "b": "oom", "c": "complete",
    }  # fmt: skip
    assert run.status == "complete" and run.reason == ""
    assert [s["points"] for s in run.sessions] == [["a", "b", "c"], ["c"]]
    assert run.sessions[0]["state"].startswith("failed")
    assert run.records["b"]["reason"].startswith("OutOfMemoryError")
    # The decision rule reads an out-of-memory cap as Fail (with a 1-GPU projection here).
    cfg = dataclasses.replace(cfg, projection=dataclasses.replace(cfg.projection, gpus=1))
    assert tt.read_decision(run.records, cfg)["row"] == "Fail"


def test_a_failing_setup_takes_its_points_and_the_run_goes_on(tmp_path, monkeypatch):
    def broken(self):
        raise RuntimeError("no model\nthe folder is empty")

    monkeypatch.setattr(tt.ThroughputWorker, "setup", broken)
    cfg = smoke_cfg(tmp_path, points=(point("a", 64), point("b", 128)))
    run = tt.run_points(cfg, None, time.time(), {}, launch=InlineSession)
    assert run.status == "failed" and len(run.sessions) == 1
    for name in ("a", "b"):
        assert run.records[name]["outcome"] == "error"
        assert run.records[name]["reason"] == (
            "session setup failed: rank 0: RuntimeError: the folder is empty"
        )


def test_a_silent_session_is_ended_and_its_point_marked_stalled(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "POLL_S", 0.01)
    cfg = smoke_cfg(tmp_path)
    limits = dataclasses.replace(cfg.limits, stall_s=0.2)
    plan = plan_for(cfg, [point("x", 64)])
    beat = tt.Heartbeat(plan, 0)
    beat.write(state="ready", ready=True)
    beat.write(state="running", point="x", step=2, of=3)
    past = time.time() - 60
    os.utime(beat.path, (past, past))

    class Silent:
        killed = False

        def wait(self, timeout):
            time.sleep(timeout)
            return None

        def kill(self):
            Silent.killed = True

    began = time.monotonic()
    state = tt.watch(Silent(), Path(cfg.out_dir), 0, time.time() + 60, limits)
    assert state.startswith("stalled: no process reported progress") and Silent.killed
    assert time.monotonic() - began < 5
    records, setup = tt.collect(plan, state, Path(cfg.out_dir))
    assert setup is None and records["x"]["outcome"] == "stalled"
    assert records["x"]["reason"].endswith("(at step 2 of the point)")


def test_the_hard_deadline_stops_a_session(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "POLL_S", 0.01)

    class Endless:
        def wait(self, timeout):
            return None

        def kill(self):
            pass

    cfg = smoke_cfg(tmp_path)
    state = tt.watch(Endless(), Path(cfg.out_dir), 0, time.time() - 1, cfg.limits)
    assert state.startswith("stopped: the wall budget")


def test_a_step_over_the_limit_kills_the_run(tmp_path):
    cfg = smoke_cfg(tmp_path, points=(point("a", 64), point("b", 128)))
    cfg = dataclasses.replace(cfg, limits=dataclasses.replace(cfg.limits, max_step_s=1e-9))
    run = tt.run_points(cfg, None, time.time(), {}, launch=InlineSession)
    assert run.status == "killed" and run.records["a"]["outcome"] == "killed"
    assert "over max_step_s 1e-09 s times accum 1" in run.reason
    assert run.records["b"]["outcome"] == "not run"
    assert len(run.records["a"]["step_s"]["warm"]) == 1  # killed after its first step


def test_no_point_starts_after_the_budget(tmp_path):
    cfg = smoke_cfg(tmp_path, points=(point("a", 64),))
    run = tt.run_points(cfg, None, time.time() - 10_000, {}, launch=InlineSession)
    assert run.status == "stopped" and run.sessions == []
    assert run.records["a"]["outcome"] == "not run"
    # Inside a session, rank 0's clock decides and the point is left for a later run.
    assert tt.run_session(0, 1, plan_for(cfg, cfg.points, deadline=0.0), 0) == 0
    assert not (Path(cfg.out_dir) / "points" / "a").exists()


# --- the projection and the decision rule --------------------------------------------------------


def test_micro_cost_interpolates_is_flat_below_and_extends_the_last_slope():
    table = [(512, 1.0), (1024, 2.0), (2048, 3.0)]
    tokens = np.array([100, 512, 768, 1024, 1536, 2048, 3072])
    assert tt.micro_cost(table, tokens).tolist() == [1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    assert tt.micro_cost([(512, 1.5)], [10, 9000]).tolist() == [1.5, 1.5]


def test_a_predicted_step_waits_for_the_slowest_process_and_all_reduces_once():
    table = [(512, 1.0), (1024, 2.0), (2048, 3.0)]
    lengths = [[[512, 1024], [2048, 512]], [[512, 512], [512, 512]]]  # [steps, W, A]
    # Step 1: ranks 3.0 and 4.0 s; step 2: 2.0 and 2.0 s; one all-reduce of A = 2 kept.
    predicted = tt.predict_steps(lengths, table, t_ddp=0.1, remainder=0.05)
    assert predicted.tolist() == pytest.approx([4.0 - 0.1 + 0.05, 2.0 - 0.1 + 0.05])


def fixed_record(name, gpus, tokens, step, micro, outcome="complete", memory=None):
    return {
        "point": name, "kind": "fixed", "gpus": gpus, "accum": 1, "tokens": tokens,
        "outcome": outcome, "reason": "", "step_s": {"median": step, "timed": [step]},
        "micro_s": {"median": micro}, "remainder_s": step - micro, "memory": memory,
    }  # fmt: skip


def test_the_projection_arithmetic(tmp_path):
    """Constant micro steps make every drawn step the same: 8 * 2.0 - 7 * 0.2 + 0.5 = 15.1 s;
    the mix measured 10% slower, so 16.61 s, and a week follows by division."""
    projection = tt.load_config(CONFIG, False, {}).projection
    records = {
        "ddp-512": fixed_record("ddp-512", 4, 512, 2.5, 2.0),
        "ddp-8192": fixed_record("ddp-8192", 4, 8192, 2.5, 2.0),
        "single-512": fixed_record("single-512", 1, 512, 2.3, 2.1),
        "mix": {
            "point": "mix", "kind": "mix", "gpus": 4, "accum": 8, "outcome": "complete",
            "lengths": [[[600] * 8] * 4] * 3, "step_s": {"timed": [16.61, 16.61, 16.61]},
        },
    }  # fmt: skip
    result = tt.project(records, projection)
    assert result["t_ddp_s"] == pytest.approx(0.2) and result["t_ddp_tokens"] == 512
    assert result["remainder_s"] == pytest.approx(0.5)
    assert result["model_step_s"]["mean"] == pytest.approx(15.1)
    assert result["calibration"]["factor"] == pytest.approx(1.1)
    assert result["step_s"] == pytest.approx(16.61) and result["basis"] == "calibrated"
    one = result["one_pass"]
    assert one["steps"] == math.ceil(99674 / 32) == 3115
    assert one["wall_h"] == pytest.approx(3115 * 16.61 / 3600)
    assert one["quota_h"] == {
        "2": pytest.approx(2 * one["wall_h"]), "1": pytest.approx(one["wall_h"]),
    }  # fmt: skip
    assert one["gpu_s_per_episode"] == pytest.approx(16.61 / 8)
    rows = {(row["rate"], row["quota_h"]): row for row in result["weekly"]}
    assert len(rows) == 8
    row = rows[(2.0, 30.0)]
    assert row["wall_h"] == 15.0 and row["steps"] == pytest.approx(3600 * 15 / 16.61)
    assert row["episodes"] == pytest.approx(32 * row["steps"])
    assert row["passes"] == pytest.approx(row["episodes"] / 99674)
    # Without a mix point the model stands; without a 4-GPU point there is no projection.
    del records["mix"]
    assert tt.project(records, projection)["basis"] == "model"
    assert not tt.project({"single-512": records["single-512"]}, projection)["available"]


def memory(headroom_gib, retries=0, total_gib=22.0):
    gib = tt.GIB
    return {
        "device_total": total_gib * gib, "headroom": headroom_gib * gib,
        "headroom_fraction": headroom_gib / total_gib, "alloc_retries": retries,
    }  # fmt: skip


@pytest.mark.parametrize(
    ("cap", "row"),
    [
        (fixed_record("c", 4, 8192, 9.0, 8.9, memory=memory(8.0)), "Pass"),
        (fixed_record("c", 4, 8192, 9.0, 8.9, memory=memory(1.5)), "Borderline"),
        (fixed_record("c", 4, 8192, 9.0, 8.9, memory=memory(8.0, retries=2)), "Borderline"),
        (fixed_record("c", 4, 8192, 9.0, 8.9, outcome="oom"), "Fail"),
        (fixed_record("c", 4, 8192, 9.0, 8.9, outcome="stalled"), "No result"),
        (fixed_record("c", 4, 8192, 9.0, 8.9, memory=None), "No result"),
    ],
)
def test_the_decision_rule_reading(cap, row):
    cfg = tt.load_config(CONFIG, False, {})
    records = {
        "c": cap,
        "l": fixed_record("l", 4, 9728, 11.0, 10.9, memory=memory(1.0)),
        "h": {"point": "h", "kind": "hypernet", "gpus": 1, "outcome": "complete",
              "hypernet_ms": {"eval": {"median": 4.0}}},
    }  # fmt: skip
    reading = tt.read_decision(records, cfg)
    assert reading["row"] == row
    assert reading["longer"] == "tight" and reading["hypernet"] is True
    records["l"]["outcome"] = "oom"
    assert tt.read_decision(records, cfg)["longer"] == "does not fit"


# --- the smoke, end to end -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke") / "run_logs"
    began = time.monotonic()
    code = tt.main(["--config", str(CONFIG), "--smoke", "--out-dir", str(out)])
    return out, code, time.monotonic() - began


def test_smoke_exits_0_in_under_two_minutes(smoke_run):
    out, code, seconds = smoke_run
    assert code == tr.EXIT_OK, (out / "summary.md").read_text()
    assert seconds < 120
    print(f"P3.6 smoke: {seconds:.1f} s")


def test_smoke_measures_every_point_and_writes_the_a8_outputs(smoke_run):
    out, _, _ = smoke_run
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    top = {"config.yaml", "run.json", "status.json", "measurements.jsonl", "projection.json",
           "summary.md"}  # fmt: skip
    assert top <= files and "error.log" not in files
    for name in files:
        assert kaggle_ops.pull_allowed(f"run_logs/{name}"), name
    records = read_jsonl(out / "measurements.jsonl")
    cfg = tt.load_config(CONFIG, True, {})
    assert [r["point"] for r in records] == [p.name for p in cfg.points]
    assert all(r["outcome"] == "complete" for r in records)
    for r in records:
        expected = {f"points/{r['point']}/rank{k}.json" for k in range(r["gpus"])}
        assert expected <= files
        if r["kind"] == "fixed":
            assert r["tokens_actual"] == [r["tokens"]] and r["step_s"]["n"] == cfg.timed_steps
            assert len(r["step_s"]["warm"]) == cfg.warm_steps and r["micro_s"]["median"] > 0
        if r["kind"] == "mix":
            assert np.asarray(r["lengths"]).shape == (cfg.timed_steps, 2, 2)
    timing = [line for p in out.glob("points/*/timing-rank*.jsonl") for line in read_jsonl(p)]
    assert {t["op"] for t in timing} == {"h_forward", "train_step", "step",
                                          "h_forward_eval", "h_forward_train"}  # fmt: skip
    assert all(set(t) == {"puzzle_id", "op", "seconds", "tokens"} for t in timing)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "complete" and status["exit_code"] == 0
    info = json.loads((out / "run.json").read_text())
    assert info["smoke"] and info["model_dir"] is None
    assert [s["world"] for s in info["sessions"]] == [2, 1]
    assert all(s["state"] == "ok" for s in info["sessions"])


def test_smoke_projection_is_calibrated_and_the_summary_reads(smoke_run):
    out, _, _ = smoke_run
    projection = json.loads((out / "projection.json").read_text())
    assert projection["available"] and projection["basis"] == "calibrated"
    assert [row["tokens"] for row in projection["table"]] == [64, 128, 256]
    assert projection["t_ddp_tokens"] == 64 and projection["calibration"]["steps"] == 2
    assert len(projection["weekly"]) == 8
    summary = (out / "summary.md").read_text()
    for text in ("**complete**", "## Steps and memory", "## Projection for P0.4",
                 "**Expected step:", "H's forward for all layers and modules",
                 "a smoke run has no brief"):  # fmt: skip
        assert text in summary, text
    assert chr(0x2014) not in summary  # built at runtime: this file passes the dash check


def test_smoke_resume_keeps_every_finished_point(smoke_run):
    out, _, _ = smoke_run
    before = read_jsonl(out / "measurements.jsonl")
    args = ["--config", str(CONFIG), "--smoke", "--out-dir", str(out), "--resume", str(out)]
    assert tt.main(args) == tr.EXIT_OK
    after = read_jsonl(out / "measurements.jsonl")
    assert [{k: v for k, v in r.items() if k != "resumed_from"} for r in after] == before
    assert all(r["resumed_from"] == str(out) for r in after)
    assert json.loads((out / "run.json").read_text())["sessions"] == []


def test_resume_reruns_a_point_whose_settings_changed(smoke_run, tmp_path):
    out, _, _ = smoke_run
    cfg = tt.load_config(CONFIG, True, {"out_dir": str(tmp_path / "next")})
    changed = dataclasses.replace(
        cfg, points=tuple(dataclasses.replace(p, timed_steps=3) if p.name == "ddp-64" else p
                          for p in cfg.points)
    )  # fmt: skip
    kept = tt.reusable(out, changed)
    assert set(kept) == {p.name for p in cfg.points} - {"ddp-64"}
    tt.prepare_out_dir(tmp_path / "next", changed, out, kept)
    copied = {p.name for p in (tmp_path / "next" / "points").iterdir()}
    assert copied == set(kept)
    stepped = dataclasses.replace(changed, train={**changed.train, "lr": 1e-3})
    assert tt.reusable(out, stepped) == {}


# --- the model search ----------------------------------------------------------------------------


def test_the_model_is_found_by_the_end_of_its_handle(tmp_path):
    handle = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
    model = tmp_path.joinpath("models", "sorokin", "qwen3_4b_grids15_sft139", "transformers",
                              "bfloat16", "1")  # fmt: skip
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    other = tmp_path / "datasets" / "someone" / "qwen3_4b_grids15_sft139"
    other.mkdir(parents=True)
    (other / "config.json").write_text("{}")
    assert tt.find_model_dir(tmp_path, handle) == model
    copy = tmp_path / "qwen3_4b_grids15_sft139" / "transformers" / "bfloat16" / "1"
    copy.mkdir(parents=True)
    (copy / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="found 2 model folders"):
        tt.find_model_dir(tmp_path, handle)
    with pytest.raises(ValueError, match="model_id must be"):
        tt.find_model_dir(tmp_path, "sorokin/qwen")


# --- the build and the simulated script notebook -------------------------------------------------


def test_the_build_config_matches_its_brief():
    config = tbs.load_config("configs/script_P3.6-01.yaml")
    base = tbs.load_config("configs/script_P3.5-02.yaml")
    assert config.kind == "script" and config.kaggle_notebook == "alyahya737/arc26-p3-6-throughput"
    assert config.accelerator == "NvidiaL4" and config.notebook_inputs == {}
    assert config.script == "scripts/train_throughput.py"
    assert config.script_config == "configs/train_throughput.yaml"
    assert config.args == ("--commit", "{commit}")
    assert {f: getattr(config, f) for f in tbs.STACK} == {f: getattr(base, f) for f in tbs.STACK}
    assert config.model_slug == base.model_slug
    limits = yaml.safe_load(CONFIG.read_text())["limits"]
    assert limits["max_wall_s"] + 600 == config.script_kill_s
    assert config.script_kill_s <= tbs.nbb.SESSION_LIMIT_S - tbs.nbb.WRITE_RESERVE_S
    text = BRIEF.read_text(encoding="utf-8")
    header = kaggle_ops.brief_header(text)
    assert (header["Run-id"], header["Revision"], header["Kind"]) == ("P3.6-01", "1", "throughput")
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Accelerator"] == config.accelerator
    assert header["Config"] == config.script_config
    assert int(header["Timeout"]) == config.script_kill_s + tbs.nbb.WRITE_RESERVE_S
    assert "TODO" not in text and config.kaggle_notebook in kaggle_ops.briefed_notebooks()


def test_the_build_config_has_a_comment_above_every_key():
    lines = BUILD_CONFIG.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"no comment above {line!r}"


@tbs.needs_builder
def test_the_build_carries_the_script_and_its_config_and_passes_the_gateway(tmp_path):
    repo = tbs.snapshot(tmp_path / "repo")
    folder = tbs.bs.build("P3.6-01", repo / "configs" / "script_P3.6-01.yaml", repo)
    files = tbs.read_files(folder)
    assert set(files) == {
        "MANIFEST.md", "kaggle_script.ipynb", "kernel-metadata.json", f"arc26-code/{tbs.WHEEL}",
        "arc26-code/dataset-metadata.json", "arc26-code/train_throughput.py",
        "arc26-code/train_throughput.yaml",
    }  # fmt: skip
    assert files["arc26-code/train_throughput.py"] == SCRIPT.read_bytes()
    assert files["arc26-code/train_throughput.yaml"] == CONFIG.read_bytes()
    folder = folder.resolve()
    meta = kaggle_ops.load_kernel_metadata(folder)
    assert meta["id"] == "alyahya737/arc26-p3-6-throughput" and meta["machine_shape"] == "NvidiaL4"
    assert meta["kernel_sources"] == ["sorokin/pip-install-unsloth-flash-patch/2"]
    kaggle_ops.check_kernel_metadata(meta, True)
    kaggle_ops.check_flags(meta, "NvidiaL4")
    kaggle_ops.check_code_file(folder, meta)
    kaggle_ops.check_dataset_folder(folder / "arc26-code")
    namespace = {}
    exec(tbs.script_code(folder)["arc26-02-config"], namespace)
    assert namespace["SCRIPT_KILL_S"] == 3000 and namespace["ARGS"] == ["--commit", "{commit}"]
    assert namespace["BUILD"]["script_config"] == "configs/train_throughput.yaml"
    assert "notebook_inputs" not in namespace["BUILD"]
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "with the brief's `Timeout` at least 3300 s" in manifest
    # The wheel of the same commit as P3.5-02's build: arc26/ is untouched, one wheel for both.
    other = tbs.bs.build("P3.5-02", repo / "configs" / "script_P3.5-02.yaml", repo)
    assert files[f"arc26-code/{tbs.WHEEL}"] == (other / "arc26-code" / tbs.WHEEL).read_bytes()


def test_the_simulated_script_notebook_runs_the_smoke(tmp_path):
    """The environment check finds train_throughput.py and its config beside the wheel with the
    build's sha256; the bf16 stop passes on four L4s; the script runs from the code dataset's
    folder with the commit, its two sessions import the installed wheel, and its logs land beside
    the notebook's. The arguments added suit this machine only: --smoke and the run-log folder."""
    root = tmp_path / "kaggle"
    working = root / "working"
    wheel = nl.make_wheel(tmp_path)
    config = tbs.load_config("configs/script_P3.6-01.yaml")
    extra = ("--smoke", "--out-dir", str(working / "run_logs"))
    config = config._replace(args=(*config.args, *extra))
    info = tbs.nbb.build_info(
        config, tag="local-test", commit=nl.COMMIT, config_file="configs/script_P3.6-01.yaml",
        config_sha256=nl.sha256(BUILD_CONFIG), wheel=wheel.name, wheel_sha256=nl.sha256(wheel),
        script_sha256=nl.sha256(SCRIPT), script_config_sha256=nl.sha256(CONFIG),
    )  # fmt: skip
    notebook = tmp_path / "script_P3.6-01.ipynb"
    notebook.write_bytes(tbs.nbb.notebook_bytes(tbs.nbb.build_script_notebook(config, info)))
    extras = {path.name: path.read_bytes() for path in (SCRIPT, CONFIG)}
    assert nl.make_tree(tmp_path, wheel, extras=extras) == root
    run = nl.simulate(tmp_path, notebook, root, gpus=4, real_transformers=True)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [], run.output
    stages = nl.stages_json(run)
    assert stages["ok"] is True and stages["script_exit"] == 0, run.output
    assert stages["settings"]["script_kill_s"] == 3000
    dataset = root.joinpath("input", *nl.DATASET_DIRS["2026"])
    (call,) = nl.script_calls(run)
    assert call["argv"] == [
        str(dataset / SCRIPT.name), "--config", str(dataset / CONFIG.name), "--commit",
        nl.COMMIT, *extra,
    ]  # fmt: skip
    logs = nl.run_logs(run)
    ours = {"config.yaml", "run.json", "status.json", "measurements.jsonl", "projection.json",
            "summary.md", "points", "sessions"}  # fmt: skip
    assert set(logs) == ours | set(nl.NOTEBOOK_LOGS)
    status = json.loads(logs["status.json"].read_text())
    assert status["status"] == "complete", logs["summary.md"].read_text()
    info = json.loads(logs["run.json"].read_text())
    assert info["commit"] == nl.COMMIT and info["smoke"] is True
    assert sorted(p.name for p in working.iterdir()) == ["run_logs"]
    # The kernel imported no model code: only the script's processes did.
    assert run.record["imports"] == []
