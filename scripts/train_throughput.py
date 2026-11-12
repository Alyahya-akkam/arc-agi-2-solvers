r"""Throughput and memory of one meta-training step of H (task P3.6; BRIEF B5 P3.6, B3, B8, A8).

Purpose: P0.4, the human's choice of compute for meta-training, needs the cost of one training
step at the lengths the episodes reach and proof that the worst case fits an L4's 24 GB (B5 P3.6;
B8, "sequence length"). This script times P3.5's training step, unchanged, on synthetic episodes
of chosen lengths: step seconds, micro-step seconds, and peak device memory per GPU, with
DistributedDataParallel over 4 processes and in one process; H's forward alone (B5 P3.3's bound);
and the long run's step (accumulation 8) on lengths drawn from P3.4's training histogram. From
these it projects the steps per week of meta-training under both readings of the 4xL4 quota rate.
Pre-run brief ``docs/prerun/P3.6-01.md``, runbook ``docs/runbooks/train_throughput.md``, config
``configs/train_throughput.yaml``.

Inputs: the YAML config and the base model, found under ``input_root`` by the end of its Kaggle
Models handle (or given with ``--model-dir``). No episode build is read: every episode is
synthetic. Outputs in ``out_dir`` (on Kaggle ``/kaggle/working/run_logs``, pulled):
``config.yaml`` (resolved), ``run.json`` (commit, versions, model folder, sessions),
``status.json``, ``measurements.jsonl`` (one line per point), ``projection.json``,
``summary.md``; per point ``points/<name>/rank<k>.json`` (each process's raw measurements) and
``points/<name>/timing-rank<k>.jsonl`` (``{puzzle_id, op, seconds, tokens}`` per GPU op); per
session ``sessions/s<i>/rank<k>.json`` (each process's last state, the stall watchdog's input);
``error.log`` on a failure of the script itself.

Method: a point is one measurement: episodes of $T$ tokens on $W$ processes with accumulation $A$
(``fixed``), lengths drawn from P3.4's histogram (``mix``), or H's forward alone (``hypernet``).
Consecutive points with the same $W$ share a session: $W$ processes started with
``torch.multiprocessing`` (spawn; NCCL on GPUs, gloo on the CPU), each loading the frozen bf16 base
and H as ``arc26.hyper.train`` does for a training run. ``ThroughputWorker`` replaces only that
worker's setup (synthetic episodes instead of P3.4's build, one fixed $\bar z$ instead of the
encoder); its ``train_step`` is the training worker's own, so a timed step is the training step:
H's forward, the injection, the base's forward and backward with non-reentrant checkpointing per
layer, DDP's all-reduce, clipping, and AdamW. A point runs ``warm_steps`` untimed then
``timed_steps`` timed steps after resetting the CUDA allocator's peak statistics. An
out-of-memory error ends its point as a result (``oom``) and ends its session; the session's
remaining points run in a new one. The projection interpolates the micro-step seconds measured
at each length, charges the measured per-step costs, draws accumulation-8 steps from P3.4's
histogram, calibrates on the measured accumulation-8 step, and divides a week's hours by it.

Exit codes: 0 every point measured (an out-of-memory point is a result), 1 failed (a point's
error, a stalled session, or the script's own failure), 2 invalid config or command line,
3 stopped by the wall budget (``--resume`` continues), 4 killed by the step-time criterion.

Usage::

    python scripts/train_throughput.py --config configs/train_throughput.yaml --commit SHA
    python scripts/train_throughput.py --config configs/train_throughput.yaml --smoke
    python scripts/train_throughput.py --config configs/train_throughput.yaml \
        --resume /kaggle/input/<previous output>/run_logs --commit SHA
"""

import argparse
import bisect
import functools
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from arc26.hyper.episodes import (
    PAIR_OVERHEAD_TOKENS,
    EpisodeRecord,
    PermutationTable,
    derive_seed,
    grid_tokens,
)
from arc26.hyper.hypernet import BaseShapes, Hypernet
from arc26.hyper.inject import wrap_targets
from arc26.hyper.train import (
    EXIT_CONFIG,
    EXIT_FAILED,
    EXIT_KILLED,
    EXIT_OK,
    EXIT_STOPPED,
    TrainConfig,
    WarmupCosine,
    _Worker,
    atomic_write_bytes,
    free_port,
    load_base,
    param_groups,
    train_slot,
)
from arc26.nvarc.arc_loader import QwenFormatter
from arc26.timing import TimingLogger

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nothing is downloaded; Kaggle runs offline

REPO = Path(__file__).resolve().parent.parent
SCHEMA = 1
KINDS = ("fixed", "mix", "hypernet")
COMPLETE, OOM, ERROR, KILLED, STALLED, NOT_RUN = (
    "complete", "oom", "error", "killed", "stalled", "not run",
)  # fmt: skip
# Results a resumed run keeps: a measurement, or an out-of-memory error, which is a result too.
FINAL = (COMPLETE, OOM)
# Which outcome names a point when its processes disagree: the worst wins.
SEVERITY = {COMPLETE: 0, ERROR: 1, KILLED: 2, OOM: 3}
SMOKE_PATH_KEYS = ("out_dir", "input_root")
# The step's settings the config states; TrainConfig's other keys are filled by train_config.
TRAIN_KEYS = (
    "model_id",
    "device",
    "dtype",
    "attn_implementation",
    "gradient_checkpointing",
    "max_tokens",
    "loss_includes_eos",
    "hypernet",
    "steps",
    "lr",
    "min_lr",
    "warmup_steps",
    "weight_decay",
    "betas",
    "eps",
    "clip",
    "penalty",
    "penalty_weight",
    "kl_weight",
    "seed",
)
# TrainConfig requires these, and a throughput session reads none of them: there is no episode
# build, encoder, validation, checkpoint, or training loop (only train_step runs).
UNREAD_TRAIN_VALUES: dict[str, Any] = {
    "episodes_root": "none: synthetic episodes",
    "manifest_sha256": "0" * 64,
    "check_root": "none: synthetic episodes",
    "train_split": "train",
    "val_split": "val",
    "encoder": {
        "layer": 24,
        "pool": "output",
        "n_perm": 2,
        "view_seed": 2,
        "batch_tokens": 16384,
        "chunk_pairs": 256,
    },
    "embedding_cache": None,
    "eval_every": 1,
    "ckpt_every": 1,
    "keep_checkpoints": 1,
}
# Grids as the notebook renders them (arc26.hyper.episodes): a 30 by 30 grid is 929 tokens, a pair
# adds 8 for the chat markers, and the smallest episode is one demonstration and a query of 1 by 1.
MAX_SIDE = 30
MAX_PAIR_TOKENS = 2 * grid_tokens(MAX_SIDE, MAX_SIDE) + PAIR_OVERHEAD_TOKENS
MIN_EPISODE_TOKENS = 2 * (2 * grid_tokens(1, 1) + PAIR_OVERHEAD_TOKENS)
# Color permutations of the synthetic episodes' augmentations; their number changes no length.
TABLE_SIZE = 16
# How often the parent checks its sessions, seconds.
POLL_S = 2.0
# The model search (find_model_dir) walks input_root as the script notebook's environment check
# does: at most this deep (the model's config.json is at level 7 in the 2026 layout, K0), package
# trees skipped, each real folder once.
SEARCH_DEPTH = 8
SKIPPED_DIRS = frozenset({"site-packages", "dist-packages", "__pycache__", "node_modules", ".git"})
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
GIB = 1024**3
# Files the script writes in out_dir; a fresh run removes earlier ones, nothing else.
OWN_OUTPUTS = (
    "config.yaml",
    "run.json",
    "status.json",
    "measurements.jsonl",
    "projection.json",
    "summary.md",
    "error.log",
)


class ConfigError(ValueError):
    """The config file or a command-line override is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _keys(
    section: Mapping[str, Any], required: Sequence[str], optional: Sequence[str] = ()
) -> None:
    missing = sorted(set(required) - section.keys())
    unknown = sorted(section.keys() - set(required) - set(optional))
    _require(not missing and not unknown, f"missing keys {missing}, unknown keys {unknown}")


# --- configuration -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One measurement of a run.

    Attributes:
        name: Unique id, the folder of its outputs.
        kind: ``fixed`` (every episode ``tokens`` long), ``mix`` (lengths drawn from the
            projection's histogram), or ``hypernet`` (H's forward alone).
        gpus: Processes of its session, one GPU each (gloo processes on the CPU).
        accum: Micro steps (episodes) per process per optimizer step.
        tokens: Episode length of a ``fixed`` point; None otherwise.
        warm_steps: Untimed steps first (forward calls of a ``hypernet`` point).
        timed_steps: Timed steps (forward calls of a ``hypernet`` point).
    """

    name: str
    kind: str
    gpus: int
    accum: int
    tokens: int | None
    warm_steps: int
    timed_steps: int


@dataclass(frozen=True)
class Projection:
    """The meta-training run the projection prices and the week it is priced against.

    Attributes:
        bin_tokens: Width of the length histogram's bins.
        counts: Training episodes per bin; bin $i$ holds lengths $[i b, (i + 1) b)$.
        mean_tokens: The measured mean length, compared with the draw's mean.
        episodes: Training episodes in one pass.
        gpus: Processes of the long run ($W$).
        accum: Its accumulation ($A$); $W A$ episodes per optimizer step.
        samples: Steps drawn for the expected step time.
        seed: Seed of the draw.
        quota_hours: Weekly quota hours tabulated.
        rates: Quota hours charged per hour of a session on the accelerator, one per reading.
    """

    bin_tokens: int
    counts: tuple[int, ...]
    mean_tokens: float
    episodes: int
    gpus: int
    accum: int
    samples: int
    seed: int
    quota_hours: tuple[float, ...]
    rates: tuple[float, ...]


@dataclass(frozen=True)
class Decision:
    """Settings of the pre-registered decision rule that ``summary.md`` reads.

    Attributes:
        cap_tokens: The training cap, whose point decides Pass, Borderline, or Fail.
        longer_tokens: The longer length whose point says whether a longer cap fits.
        headroom_fraction: Free share of a GPU's memory at the peak needed to pass.
        hypernet_budget_ms: Bound on H's forward (B5 P3.3).
    """

    cap_tokens: int
    longer_tokens: int
    headroom_fraction: float
    hypernet_budget_ms: float


@dataclass(frozen=True)
class Limits:
    """Budgets and kill criteria the script enforces itself.

    Attributes:
        max_wall_s: Seconds from the script's start after which nothing runs.
        reserve_s: No point starts with less than this left of ``max_wall_s``.
        stall_s: A session that reports no progress for this long is ended.
        max_step_s: Kill criterion: a step slower than this many seconds per micro step.
        max_sessions: Sessions at most, replacements of failed ones included.
    """

    max_wall_s: float
    reserve_s: float
    stall_s: float
    max_step_s: float
    max_sessions: int


@dataclass(frozen=True)
class ThroughputConfig:
    r"""One throughput run; ``configs/train_throughput.yaml`` gives every value with its reason.

    Attributes:
        run_id: Pre-run brief id.
        input_root: Folder searched for the base model when ``model_dir`` is None.
        model_dir: The base model's folder, or None to search (GPU) or build the tiny one (smoke).
        out_dir: Run-log folder.
        train: The training step's settings, the keys ``TRAIN_KEYS``.
        points: Measurements in run order.
        warm_steps: Default untimed steps of a point.
        timed_steps: Default timed steps of a point.
        projection: The priced meta-training run.
        decision: The decision rule's settings.
        limits: Budgets and kill criteria.
        seed: Seed of the synthetic episodes and of $\bar z$.
    """

    run_id: str
    input_root: str
    model_dir: str | None
    out_dir: str
    train: dict[str, Any]
    points: tuple[Point, ...]
    warm_steps: int
    timed_steps: int
    projection: Projection
    decision: Decision
    limits: Limits
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain YAML and JSON values."""
        return _plain(asdict(self))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _positive_ints(section: Mapping[str, Any], names: Sequence[str], where: str) -> None:
    for name in names:
        value = section[name]
        _require(_is_int(value) and value >= 1, f"{where}{name} must be an integer >= 1")


def _parse_point(raw: Any, warm: int, timed: int, max_tokens: int) -> Point:
    _require(isinstance(raw, Mapping), "every point must be a mapping")
    _keys(raw, ("name", "kind", "gpus"), ("accum", "tokens", "warm_steps", "timed_steps"))
    name, kind = raw["name"], raw["kind"]
    _require(isinstance(name, str) and bool(NAME.fullmatch(name)), f"bad point name {name!r}")
    where = f"point {name}: "
    _require(kind in KINDS, f"{where}kind must be one of {KINDS}")
    values = {"gpus": raw["gpus"], "accum": raw.get("accum", 1)}
    values["warm_steps"] = raw.get("warm_steps", warm)
    values["timed_steps"] = raw.get("timed_steps", timed)
    _positive_ints(values, ("gpus", "accum", "timed_steps"), where)
    _require(
        _is_int(values["warm_steps"]) and values["warm_steps"] >= 0,
        f"{where}warm_steps must be an integer >= 0",
    )
    tokens = raw.get("tokens")
    if kind == "fixed":
        _require(
            _is_int(tokens) and MIN_EPISODE_TOKENS <= tokens <= max_tokens,
            f"{where}tokens must be an integer from {MIN_EPISODE_TOKENS} to train.max_tokens "
            f"({max_tokens})",
        )
        _require(feasible(tokens), f"{where}no episode in the notebook format has {tokens} tokens")
    else:
        _require(tokens is None, f"{where}only a fixed point has tokens")
    if kind == "hypernet":
        _require(values["gpus"] == 1 and values["accum"] == 1, f"{where}runs on 1 GPU, accum 1")
    return Point(
        name=name,
        kind=kind,
        gpus=values["gpus"],
        accum=values["accum"],
        tokens=tokens,
        warm_steps=values["warm_steps"],
        timed_steps=values["timed_steps"],
    )


def _parse_projection(raw: Any, max_tokens: int) -> Projection:
    _require(isinstance(raw, Mapping), "projection must be a mapping")
    _keys(raw, [f.name for f in fields(Projection)])
    where = "projection."
    _positive_ints(raw, ("bin_tokens", "episodes", "gpus", "accum", "samples"), where)
    _require(_is_int(raw["seed"]), f"{where}seed must be an integer")
    counts = raw["counts"]
    _require(
        isinstance(counts, list)
        and counts
        and all(_is_int(c) and c >= 0 for c in counts)
        and sum(counts) > 0,
        f"{where}counts must be a non-empty list of integers >= 0 with a positive sum",
    )
    _require(
        raw["bin_tokens"] * len(counts) <= max_tokens + 1,
        f"{where}the histogram reaches {raw['bin_tokens'] * len(counts) - 1} tokens, above "
        f"train.max_tokens ({max_tokens})",
    )
    _require(
        _is_number(raw["mean_tokens"]) and raw["mean_tokens"] > 0,
        f"{where}mean_tokens must be a positive number",
    )
    for name in ("quota_hours", "rates"):
        values = raw[name]
        _require(
            isinstance(values, list) and values and all(_is_number(v) and v > 0 for v in values),
            f"{where}{name} must be a non-empty list of positive numbers",
        )
    return Projection(
        raw["bin_tokens"], tuple(counts), float(raw["mean_tokens"]), raw["episodes"], raw["gpus"],
        raw["accum"], raw["samples"], raw["seed"], tuple(float(v) for v in raw["quota_hours"]),
        tuple(float(v) for v in raw["rates"]),
    )  # fmt: skip


def _parse_decision(raw: Any, max_tokens: int) -> Decision:
    _require(isinstance(raw, Mapping), "decision must be a mapping")
    _keys(raw, ("cap_tokens", "longer_tokens", "headroom_fraction", "hypernet_budget_ms"))
    for name in ("cap_tokens", "longer_tokens"):
        value = raw[name]
        _require(
            _is_int(value) and MIN_EPISODE_TOKENS <= value <= max_tokens,
            f"decision.{name} must be an integer up to train.max_tokens ({max_tokens})",
        )
    fraction = raw["headroom_fraction"]
    _require(_is_number(fraction) and 0 <= fraction < 1, "decision.headroom_fraction in [0, 1)")
    budget = raw["hypernet_budget_ms"]
    _require(_is_number(budget) and budget > 0, "decision.hypernet_budget_ms must be positive")
    return Decision(raw["cap_tokens"], raw["longer_tokens"], float(fraction), float(budget))


def _parse_limits(raw: Any) -> Limits:
    _require(isinstance(raw, Mapping), "limits must be a mapping")
    _keys(raw, ("max_wall_s", "reserve_s", "stall_s", "max_step_s", "max_sessions"))
    for name in ("max_wall_s", "reserve_s", "stall_s", "max_step_s"):
        _require(_is_number(raw[name]) and raw[name] > 0, f"limits.{name} must be positive")
    _require(raw["reserve_s"] < raw["max_wall_s"], "limits.reserve_s must be below max_wall_s")
    _positive_ints(raw, ("max_sessions",), "limits.")
    return Limits(
        float(raw["max_wall_s"]), float(raw["reserve_s"]), float(raw["stall_s"]),
        float(raw["max_step_s"]), raw["max_sessions"],
    )  # fmt: skip


TOP_KEYS = (
    "run_id",
    "input_root",
    "model_dir",
    "out_dir",
    "train",
    "points",
    "warm_steps",
    "timed_steps",
    "projection",
    "decision",
    "limits",
    "seed",
)


def parse_config(raw: Mapping[str, Any]) -> ThroughputConfig:
    """Validate a config mapping (without its smoke section).

    Args:
        raw: Every key of ``TOP_KEYS``.

    Returns:
        The config.

    Raises:
        ConfigError: If a key is missing or unknown, a value is invalid, or the train section
            does not make a valid ``TrainConfig``.
    """
    _keys(raw, TOP_KEYS)
    for name in ("run_id", "input_root", "out_dir"):
        value = raw[name]
        _require(isinstance(value, str) and value != "", f"{name} must be a non-empty string")
    model_dir = raw["model_dir"]
    _require(model_dir is None or (isinstance(model_dir, str) and model_dir != ""), "model_dir")
    train = raw["train"]
    _require(isinstance(train, Mapping), "train must be a mapping")
    _keys(train, TRAIN_KEYS)
    max_tokens = train["max_tokens"]
    _require(_is_int(max_tokens) and max_tokens >= MIN_EPISODE_TOKENS, "train.max_tokens")
    _require(_is_int(raw["seed"]), "seed must be an integer")
    _require(_is_int(raw["timed_steps"]) and raw["timed_steps"] >= 1, "timed_steps must be >= 1")
    _require(_is_int(raw["warm_steps"]) and raw["warm_steps"] >= 0, "warm_steps must be >= 0")
    points = raw["points"]
    _require(isinstance(points, list) and points, "points must be a non-empty list")
    parsed = tuple(
        _parse_point(p, raw["warm_steps"], raw["timed_steps"], max_tokens) for p in points
    )
    names = [p.name for p in parsed]
    _require(len(set(names)) == len(names), f"point names must be distinct: {names}")
    projection = _parse_projection(raw["projection"], max_tokens)
    for point in parsed:
        if point.kind == "mix":
            _require(
                (point.gpus, point.accum) == (projection.gpus, projection.accum),
                f"point {point.name}: a mix point measures the projected step, so its gpus and "
                f"accum must be the projection's ({projection.gpus}, {projection.accum})",
            )
    cfg = ThroughputConfig(
        run_id=raw["run_id"],
        input_root=raw["input_root"],
        model_dir=model_dir,
        out_dir=raw["out_dir"],
        train=_plain(dict(train)),
        points=parsed,
        warm_steps=raw["warm_steps"],
        timed_steps=raw["timed_steps"],
        projection=projection,
        decision=_parse_decision(raw["decision"], max_tokens),
        limits=_parse_limits(raw["limits"]),
        seed=raw["seed"],
    )
    try:
        train_config(cfg, None, 1)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"train: {err}") from err
    return cfg


def train_config(
    cfg: ThroughputConfig, model_dir: str | None, gpus: int, accum: int = 1
) -> TrainConfig:
    """Build the ``TrainConfig`` a session's processes run the training step with.

    Args:
        cfg: The run config.
        model_dir: The base model's folder; None selects the tiny random model (smoke).
        gpus: The session's processes.
        accum: Accumulation of the first point (each point sets its own).

    Returns:
        The config: the train section, the run's id and folders, and ``UNREAD_TRAIN_VALUES``.

    Raises:
        ValueError: If ``TrainConfig`` refuses a value.
    """
    values = {
        **UNREAD_TRAIN_VALUES,
        **cfg.train,
        "run_id": cfg.run_id,
        "model_dir": model_dir,
        "out_dir": cfg.out_dir,
        # Never written: a throughput session saves no checkpoint.
        "ckpt_dir": cfg.out_dir,
        "gpus": gpus,
        "accum": accum,
        "max_wall_s": cfg.limits.max_wall_s,
        "kill": {
            "max_step_s": cfg.limits.max_step_s,
            "max_projected_s": cfg.limits.max_wall_s,
            "projection_after_steps": 1,
        },
    }
    return TrainConfig.from_dict(values)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Apply the smoke section: mappings merge key by key, other values replace."""
    merged = dict(base)
    for key, value in override.items():
        if key not in base:
            raise ConfigError(f"{where}{key} is not a config key")
        if isinstance(base[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge(base[key], value, f"{where}{key}.")
        else:
            merged[key] = value
    return merged


def load_config(path: Path, smoke: bool, overrides: Mapping[str, str | None]) -> ThroughputConfig:
    """Read the YAML config, apply the smoke section and command-line overrides, and validate.

    Both the file's own values and its smoke values are validated in either mode, so a smoke run
    also checks the GPU run's settings. In smoke mode relative paths are taken from the
    repository root.

    Args:
        path: The YAML file.
        smoke: Whether to apply the ``smoke`` section.
        overrides: Top-level keys set on the command line; None means not given.

    Returns:
        The resolved config.

    Raises:
        ConfigError: If the file is unreadable or a key or value is invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"cannot read {path}: {err}") from err
    if not isinstance(raw, dict) or not isinstance(raw.get("smoke"), dict):
        raise ConfigError(f"{path} must hold a mapping with a smoke section")
    section = raw.pop("smoke")
    smoked = _merge(raw, section, "smoke.")
    for key in SMOKE_PATH_KEYS:
        value = smoked.get(key)
        if smoke and isinstance(value, str) and not Path(value).is_absolute():
            smoked[key] = str(REPO / value)
    try:
        base = parse_config(raw)
    except ConfigError as err:
        raise ConfigError(f"{path}: {err}") from err
    try:
        smoke_cfg = parse_config(smoked)
    except ConfigError as err:
        raise ConfigError(f"{path}, smoke section: {err}") from err
    chosen = smoke_cfg if smoke else base
    changes = {key: value for key, value in overrides.items() if value is not None}
    for key, value in changes.items():
        _require(isinstance(value, str) and value != "", f"command line: {key} must be a path")
    try:
        return replace(chosen, **changes)
    except TypeError as err:
        raise ConfigError(f"command line: {err}") from err


# --- synthetic episodes of exact lengths ---------------------------------------------------------


@functools.cache
def _grid_shapes() -> dict[int, tuple[int, int]]:
    """Grid tokens to the most nearly square shape (rows, cols) with rows >= cols.

    Rows at least the columns keep the untransposed orientation the longer one, so an episode's
    ``EpisodeRecord.tokens`` (its longer orientation, as P3.4 counts) is the length it is read at.
    """
    shapes: dict[int, tuple[int, int]] = {}
    for rows in range(1, MAX_SIDE + 1):
        for cols in range(1, rows + 1):
            count = grid_tokens(rows, cols)
            best = shapes.get(count)
            if best is None or rows - cols < best[0] - best[1]:
                shapes[count] = (rows, cols)
    return shapes


@functools.cache
def _grid_values() -> tuple[int, ...]:
    return tuple(sorted(_grid_shapes()))


@functools.cache
def _reachable(grids: int) -> int:
    """Bitset of the token totals that exactly ``grids`` grids can make (bit $s$ for total $s$)."""
    if grids == 0:
        return 1
    previous, total = _reachable(grids - 1), 0
    for value in _grid_values():
        total |= previous << value
    return total


def _pair_counts(tokens: int) -> range:
    """Pairs tried for an episode: the fewest that can hold it, and two more."""
    first = max(2, math.ceil(tokens / MAX_PAIR_TOKENS))
    return range(first, first + 3)


def feasible(tokens: int) -> bool:
    """Whether an episode in the notebook format can have exactly ``tokens`` tokens.

    Args:
        tokens: The length.

    Returns:
        True when some count of pairs (``_pair_counts``) has grid shapes summing to it.
    """
    for pairs in _pair_counts(tokens):
        total = tokens - PAIR_OVERHEAD_TOKENS * pairs
        if total >= 0 and (_reachable(2 * pairs) >> total) & 1:
            return True
    return False


def _nearest_first(values: Sequence[int], share: float) -> Iterator[int]:
    """Yield ``values`` (ascending) in order of distance from ``share``, the lower one on a tie."""
    high = bisect.bisect_left(values, share)
    low = high - 1
    while low >= 0 or high < len(values):
        if high >= len(values) or (low >= 0 and share - values[low] <= values[high] - share):
            yield values[low]
            low -= 1
        else:
            yield values[high]
            high += 1


@functools.cache
def episode_shapes(tokens: int) -> tuple[tuple[int, int], ...]:
    """Grid shapes of an episode of exactly ``tokens`` tokens, in the notebook's format.

    The fewest pairs that can hold the length (one demonstration at least, the query last), each
    grid as close to an even share of the grid tokens as keeps the rest reachable: large, nearly
    square grids, as the worst case of B3 has (30 by 30 grids, four demonstrations and a query).

    Args:
        tokens: The length; a pair of grids $(x, y)$ takes
            $h_x w_x + h_x - 1 + h_y w_y + h_y - 1 + 8$ tokens (``episodes.pair_tokens``).

    Returns:
        Shapes ``(rows, cols)``, input then output of each pair: demonstrations, then the query.

    Raises:
        ValueError: If no episode has exactly that length.
    """
    values = _grid_values()
    for pairs in _pair_counts(tokens):
        total = tokens - PAIR_OVERHEAD_TOKENS * pairs
        if total < 0 or not (_reachable(2 * pairs) >> total) & 1:
            continue
        shapes = []
        for left in range(2 * pairs, 0, -1):
            for value in _nearest_first(values, total / left):
                if value <= total and (_reachable(left - 1) >> (total - value)) & 1:
                    shapes.append(_grid_shapes()[value])
                    total -= value
                    break
        return tuple(shapes)
    raise ValueError(f"no episode in the notebook format has exactly {tokens} tokens")


def synthetic_record(
    tokens: int, episode_id: str, rng: np.random.Generator, table: PermutationTable
) -> EpisodeRecord:
    """An episode of exactly ``tokens`` tokens with random colors and a shape-keeping augmentation.

    Args:
        tokens: The length.
        episode_id: Its id, the timing records' ``puzzle_id``.
        rng: Source of the colors and the augmentation.
        table: The permutation table that decodes the augmentation id.

    Returns:
        The record; ``record.tokens == tokens``.

    Raises:
        ValueError: If no episode has that length.
    """
    shapes = episode_shapes(tokens)
    grids = [rng.integers(0, 10, size=shape, dtype=np.uint8) for shape in shapes]
    pairs = list(zip(grids[0::2], grids[1::2], strict=True))
    # Augmentations that keep every grid's shape (the identity, a half turn, and their
    # compositions with a transpose and a quarter turn), so the length stays exactly tokens.
    keeping = [d for d in range(8) if not table.augmentation(d).swaps_axes]
    augmentation = 8 * int(rng.integers(len(table.permutations))) + int(rng.choice(keeping))
    record = EpisodeRecord.create(
        episode_id=episode_id,
        source="synthetic",
        task_id="synthetic",
        family=episode_id,
        split="train",
        augmentation_id=augmentation,
        demos=pairs[:-1],
        queries=pairs[-1:],
    )
    if record.tokens != tokens:
        raise ValueError(f"{episode_id}: built {record.tokens} tokens, wanted {tokens}")
    return record


def draw_lengths(projection: Projection, rng: np.random.Generator, shape: Any) -> np.ndarray:
    r"""Draw episode lengths from the projection's histogram: a bin by its count, then uniformly.

    Within bin $i$ the length is uniform over $[\max(i b, T_\min), (i + 1) b)$, $T_\min$ the
    shortest episode (``MIN_EPISODE_TOKENS``).

    Args:
        projection: The histogram.
        rng: The generator.
        shape: Shape of the result.

    Returns:
        Integer lengths of that shape.
    """
    counts = np.asarray(projection.counts, dtype=np.float64)
    bins = rng.choice(len(counts), size=shape, p=counts / counts.sum())
    low = np.maximum(bins * projection.bin_tokens, MIN_EPISODE_TOKENS)
    return rng.integers(low, (bins + 1) * projection.bin_tokens)


def histogram_mean(projection: Projection) -> float:
    """Mean length of ``draw_lengths``'s distribution, to compare with the measured mean."""
    counts = np.asarray(projection.counts, dtype=np.float64)
    width = projection.bin_tokens
    low = np.maximum(np.arange(len(counts)) * width, MIN_EPISODE_TOKENS)
    high = (np.arange(len(counts)) + 1) * width - 1
    return float((counts * (low + high) / 2).sum() / counts.sum())


def mix_lengths(projection: Projection, seed: int, point: str, count: int) -> list[int]:
    """The lengths of a mix point's draws in slot order, each the next feasible length up.

    Every process computes the same list. A drawn length no episode can have (only a few short
    ones) moves up to the next length that one can.

    Args:
        projection: The histogram.
        seed: The run's seed.
        point: The point's name, part of the draw's seed.
        count: Draws.

    Returns:
        The lengths.
    """
    rng = np.random.default_rng(derive_seed(seed, "mix", point))
    lengths = []
    for value in draw_lengths(projection, rng, count).tolist():
        while not feasible(value):
            value += 1
        lengths.append(int(value))
    return lengths


# --- the worker ----------------------------------------------------------------------------------


class FixedEmbedding:
    r"""Stands in for ``PairEmbeddings``: one seeded $\bar z$ for every episode.

    H's cost does not depend on the values of $\bar z$, and the encoder runs before a training
    run's first step (P3.5-01 measured it apart: 543 s for 2018 pairs), so it is no part of a
    step. H's task code normalizes its input (a LayerNorm), so the vector's scale is immaterial.

    Args:
        width: The base's hidden size, $\bar z$'s width.
        seed: The run's seed.
    """

    def __init__(self, width: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(derive_seed(seed, "z_bar") % 2**63)
        self.value = torch.randn(width, generator=generator)  # [hidden]

    def z_bar(self, hashes: Sequence[str]) -> Tensor:
        r"""Return $\bar z$ for any demonstrations, ``[hidden]`` fp32 on the CPU."""
        return self.value


class SlotOrder:
    """The data order of a point: training slot $g$ draws the point's record $g - g_0$.

    Args:
        offset: $g_0$, the point's first slot.
    """

    def __init__(self, offset: int) -> None:
        self.offset = offset

    def __call__(self, slot: int) -> int:
        """Return the record index of ``slot``."""
        return slot - self.offset


class PointRecords:
    """A point's records by index, built before its first step (only this process's slots).

    Args:
        records: Index to record.
    """

    def __init__(self, records: Mapping[int, EpisodeRecord]) -> None:
        self.records = dict(records)

    def __getitem__(self, index: int) -> EpisodeRecord:
        """Return record ``index``."""
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)


class KeptTiming(TimingLogger):
    """A timing logger that also keeps its records, so a point reads its micro steps back."""

    def __init__(self, log_dir: str | os.PathLike[str], worker: str) -> None:
        super().__init__(log_dir, worker)
        self.kept: list[dict[str, Any]] = []

    def record(
        self, puzzle_id: str | None, op: str, seconds: float, tokens: Any = None
    ) -> dict[str, object]:
        """Append one record to the file, as ``TimingLogger.record`` does, and keep it."""
        entry = super().record(puzzle_id, op, seconds, tokens)
        self.kept.append(dict(entry))
        return entry


@dataclass(frozen=True)
class SessionPlan:
    """What one session's processes run.

    Attributes:
        index: The session's number in the run.
        world: Its processes.
        points: Its points, in order.
        train: The training step's config (``gpus`` is ``world``).
        out_dir: The run-log folder.
        started: ``time.time()`` at the script's start.
        point_deadline: No point starts after this ``time.time()``.
        max_step_s: The step-time kill criterion per micro step.
        projection: The histogram a mix point draws from.
        seed: The run's seed.
    """

    index: int
    world: int
    points: tuple[Point, ...]
    train: TrainConfig
    out_dir: str
    started: float
    point_deadline: float
    max_step_s: float
    projection: Projection
    seed: int


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    """Read the CUDA caching allocator's counters and the device's free memory.

    Args:
        device: The process's device.

    Returns:
        Bytes: ``device_total``, ``device_used`` (every process and the CUDA context),
        ``allocated``, ``reserved``, ``peak_allocated``, ``peak_reserved`` (since the last reset),
        and the counts ``alloc_retries`` and ``ooms``; on the CPU every value is None but
        ``cpu_rss_peak``, the process's peak resident memory.
    """
    if device.type != "cuda":
        rss = None
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KiB on Linux
        except (ImportError, OSError):
            pass
        names = ("device_total", "device_used", "allocated", "reserved", "peak_allocated")
        return dict.fromkeys((*names, "peak_reserved", "alloc_retries", "ooms")) | {
            "cpu_rss_peak": rss
        }
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    return {
        "device_total": int(total),
        "device_used": int(total - free),
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        "alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "ooms": int(stats.get("num_ooms", 0)),
        "cpu_rss_peak": None,
    }


def _failure(err: BaseException) -> str:
    lines = [line.strip() for line in str(err).splitlines() if line.strip()]
    return f"{type(err).__name__}: {lines[-1]}" if lines else type(err).__name__


def is_out_of_memory(err: BaseException) -> bool:
    """Whether an error is a failed device allocation.

    Why: most reach Python as CUDA's ``OutOfMemoryError``, but an allocation inside cuBLAS (its
    handle or workspace) or NCCL (its buffers) fails as another error whose message names the
    cause; the decision rule reads all of them as out of memory.

    Args:
        err: The error a point ended with.

    Returns:
        True for ``torch.cuda.OutOfMemoryError`` and for errors whose message says "out of
        memory" or ``CUBLAS_STATUS_ALLOC_FAILED``.
    """
    text = str(err)
    return (
        isinstance(err, torch.cuda.OutOfMemoryError)
        or "out of memory" in text.lower()
        or "CUBLAS_STATUS_ALLOC_FAILED" in text
    )


class ThroughputWorker(_Worker):
    r"""One process of a throughput session: the training worker on synthetic episodes.

    Only ``setup`` is replaced, with ``_Worker.setup``'s model, H, DDP, optimizer, and schedule
    steps in its order and its seeds, and synthetic episodes and a fixed $\bar z$ instead of
    P3.4's build and the encoder. ``train_step`` and ``micro_step`` are ``_Worker``'s own, so a
    timed step is the training step (``tests/test_train_throughput.py`` checks both, bitwise on
    the CPU).

    Args:
        rank: This process's rank.
        world: Processes in the session.
        cfg: The training step's config.
        plan: The session's plan.
    """

    def __init__(self, rank: int, world: int, cfg: TrainConfig, plan: SessionPlan) -> None:
        super().__init__(rank, world, cfg, None, plan.started)
        self.plan = plan
        self.metrics = None
        self.load_seconds = 0.0
        self.attn = ""

    def setup(self) -> None:
        """Load the base and H and build the optimizer as a training run does; no data yet."""
        cfg = self.cfg
        torch.manual_seed(derive_seed(cfg.seed, "torch", self.rank) % 2**63)
        random.seed(derive_seed(cfg.seed, "python", self.rank))
        np.random.seed(derive_seed(cfg.seed, "numpy", self.rank) % 2**32)
        self.table = PermutationTable.seeded(TABLE_SIZE, cfg.seed)
        load_started = time.perf_counter()
        model, tokenizer, self.attn = load_base(cfg, self.device)
        self.load_seconds = time.perf_counter() - load_started
        self.keys = wrap_targets(model, cfg.hypernet.target_modules)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model = model
        self.formatter = QwenFormatter(tokenizer)
        if cfg.hypernet.embed_dim != model.config.hidden_size:
            raise ValueError(
                f"hypernet.embed_dim {cfg.hypernet.embed_dim} differs from the base's hidden "
                f"size {model.config.hidden_size}, the width of z_bar"
            )
        self.hypernet = Hypernet(cfg.hypernet, BaseShapes.from_config(model.config)).to(
            self.device
        )
        self.h_train = (
            DistributedDataParallel(
                self.hypernet,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
            if self.world > 1
            else self.hypernet
        )
        self.optimizer = torch.optim.AdamW(
            param_groups(self.hypernet, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps
        )
        self.scheduler = LambdaLR(
            self.optimizer, WarmupCosine(cfg.warmup_steps, cfg.steps, cfg.min_lr / cfg.lr)
        )
        self.embeddings = FixedEmbedding(model.config.hidden_size, self.plan.seed)
        self.model.eval()

    # -- facts and files ---------------------------------------------------------------------

    def facts(self) -> dict[str, Any]:
        """The session facts every result carries: devices, kernel, load time, H's size."""
        rank = self.cfg.hypernet.rank
        return {
            "session": self.plan.index,
            "world": self.world,
            "device_name": (
                torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
            ),
            "attn_implementation": self.attn,
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "load_seconds": self.load_seconds,
            "hypernet_parameters": sum(p.numel() for p in self.hypernet.parameters()),
            "emitted_per_adapter": self.hypernet.base.n_layers * sum(
                rank * (d_in + d_out) for d_in, d_out in self.hypernet.shapes.values()
            ),
        }  # fmt: skip

    def point_dir(self, point: Point) -> Path:
        """The folder of a point's outputs."""
        return Path(self.plan.out_dir) / "points" / point.name

    # -- points ------------------------------------------------------------------------------

    def run_point(self, point: Point, beat: Callable[..., None]) -> dict[str, Any]:
        """Measure one point and write this process's result, also when the point fails.

        Args:
            point: The point.
            beat: Records progress (``Heartbeat.write``).

        Returns:
            The result as written to ``points/<name>/rank<k>.json``; its ``outcome`` is
            ``complete``, ``oom``, ``killed``, or ``error``.
        """
        began = time.perf_counter()
        folder = self.point_dir(point)
        # Made now: a point can fail before its first timing record would create it.
        folder.mkdir(parents=True, exist_ok=True)
        self.timing = KeptTiming(folder, worker=f"rank{self.rank}")
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "point": point.name,
            "kind": point.kind,
            "rank": self.rank,
            "accum": point.accum,
            "tokens_target": point.tokens,
            "outcome": COMPLETE,
            "reason": "",
            **self.facts(),
        }
        # The cache is not emptied: a training run never empties it (P3.5's loop), so each point
        # meets the allocator as the points before it left it, and the long points come after the
        # mixed lengths. The peak counters start over, so the peak allocated memory is the point's
        # own; the reserved memory at the start separates the inherited cache from its growth.
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        result["memory_before"] = memory_snapshot(self.device)
        try:
            if point.kind == "hypernet":
                self._time_hypernet(point, result, beat)
            else:
                self._run_steps(point, result, beat)
        except Exception as err:
            result["outcome"] = OOM if is_out_of_memory(err) else ERROR
            result["reason"] = _failure(err)
            result["traceback"] = traceback.format_exc()
        result["memory"] = memory_snapshot(self.device)
        result["seconds"] = time.perf_counter() - began
        atomic_write_bytes(
            folder / f"rank{self.rank}.json",
            (json.dumps(result, indent=1, allow_nan=True) + "\n").encode("utf-8"),
        )
        return result

    def _records(self, point: Point, steps: int) -> tuple[PointRecords, int]:
        """This process's records of a point, by index from the point's first slot."""
        accum, world, start = point.accum, self.world, self.step
        offset = train_slot(start, 0, 0, accum, world)
        count = steps * accum * world
        if point.kind == "mix":
            lengths = mix_lengths(self.plan.projection, self.plan.seed, point.name, count)
        else:
            lengths = [int(point.tokens or 0)] * count
        records = {}
        for step in range(start, start + steps):
            for micro in range(accum):
                index = train_slot(step, micro, self.rank, accum, world) - offset
                rng = np.random.default_rng(derive_seed(self.plan.seed, point.name, index))
                episode_id = f"synthetic/{point.name}/{index}"
                records[index] = synthetic_record(lengths[index], episode_id, rng, self.table)
        return PointRecords(records), offset

    def _run_steps(self, point: Point, result: dict[str, Any], beat: Callable[..., None]) -> None:
        """Run a fixed or mix point's steps with ``train_step``; fill ``result``."""
        self.cfg = replace(self.cfg, accum=point.accum)
        total = point.warm_steps + point.timed_steps
        self.train_records, offset = self._records(point, total)
        self.order = SlotOrder(offset)
        steps: list[dict[str, Any]] = []
        micro: list[dict[str, Any]] = []
        result["steps"], result["micro"] = steps, micro
        limit = self.plan.max_step_s * point.accum
        for number in range(1, total + 1):
            beat(state="running", point=point.name, step=number, of=total)
            kept, counted = len(self.timing.kept), len(self.clock.steps)
            began = time.perf_counter()
            code, _ = self.train_step()
            outside = time.perf_counter() - began
            # train_step appends its own step time, P3.5-01's step_s, unless it found a non-finite
            # loss or gradient norm and returned before the update.
            seconds = self.clock.steps[-1] if len(self.clock.steps) > counted else outside
            records = self.timing.kept[kept:]
            forward = [r for r in records if r["op"] == "h_forward"]
            passes = [r for r in records if r["op"] == "train_step"]
            for index, (h, pass_) in enumerate(zip(forward, passes, strict=True)):
                micro.append({
                    "step": number, "micro": index, "tokens": pass_["tokens"],
                    "h_forward_s": h["seconds"], "train_step_s": pass_["seconds"],
                })  # fmt: skip
            tokens = sum(int(r["tokens"]) for r in passes)
            self.timing.record(f"synthetic/{point.name}", "step", seconds, tokens)
            steps.append({
                "step": number, "seconds": seconds, "timed": number > point.warm_steps,
                "tokens": tokens, "nonfinite": code != EXIT_OK,
            })  # fmt: skip
            slow = self.comm.broadcast_int(int(self.main and seconds > limit))
            if slow:
                result["outcome"] = KILLED
                result["reason"] = (
                    f"step {number} took {seconds:.1f} s on rank 0, over max_step_s "
                    f"{self.plan.max_step_s:g} s times accum {point.accum}"
                )
                return

    def _time_hypernet(
        self, point: Point, result: dict[str, Any], beat: Callable[..., None]
    ) -> None:
        """Time H's forward for all layers and modules: evaluation (no autograd) and training."""
        z_bar = self.embeddings.z_bar(()).to(self.device)[None]  # [1, hidden]
        calls: dict[str, list[float]] = {}
        for mode in ("eval", "train"):
            beat(state="running", point=point.name, step=mode)
            self.hypernet.train(mode == "train")
            context = torch.enable_grad() if mode == "train" else torch.no_grad()
            times = []
            for call in range(point.warm_steps + point.timed_steps):
                _synchronize(self.device)
                began = time.perf_counter()
                with context:
                    factors = self.hypernet(z_bar)[0]
                _synchronize(self.device)
                seconds = time.perf_counter() - began
                del factors
                if call >= point.warm_steps:
                    times.append(seconds)
                    self.timing.record(None, f"h_forward_{mode}", seconds)
            calls[mode] = times
        self.hypernet.train()
        result["calls"] = calls


class Heartbeat:
    """A process's last state, ``sessions/s<i>/rank<k>.json``, rewritten at every change.

    The parent ends a session whose processes all stay silent for ``limits.stall_s``.

    Args:
        plan: The session's plan.
        rank: The process's rank.
    """

    def __init__(self, plan: SessionPlan, rank: int) -> None:
        self.path = Path(plan.out_dir) / "sessions" / f"s{plan.index}" / f"rank{rank}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {"session": plan.index, "rank": rank}
        self.started = plan.started

    def write(self, **changes: Any) -> None:
        """Merge ``changes`` into the state and write it."""
        self.state.update(changes)
        self.state["elapsed_s"] = time.time() - self.started
        text = json.dumps(self.state, indent=1, default=str) + "\n"
        atomic_write_bytes(self.path, text.encode("utf-8"))


def run_session(rank: int, world: int, plan: SessionPlan, port: int) -> int:
    """Run one process of a session: set up, then measure the session's points in order.

    Callable in this process when ``world`` is 1 (the tests do).

    Args:
        rank: This process's rank.
        world: The session's processes.
        plan: The plan.
        port: TCP port of the process group's rendezvous on 127.0.0.1 (unused when ``world``
            is 1).

    Returns:
        0 when every point ran or the budget stopped the rest, 1 when the setup or a point failed
        or a kill criterion fired (the session then ends).
    """
    beat = Heartbeat(plan, rank)
    beat.write(state="starting")
    cuda = plan.train.device == "cuda"
    if world > 1:
        if cuda:
            torch.cuda.set_device(rank)
        else:
            # One process per core group: oversubscribed threads slow every process down.
            torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))
        dist.init_process_group(
            backend="nccl" if cuda else "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
        )
    code = 0
    try:
        worker = ThroughputWorker(rank, world, plan.train, plan)
        try:
            worker.setup()
        except Exception as err:
            beat.write(state="failed", error=_failure(err), traceback=traceback.format_exc())
            code = 1
            return code
        facts = worker.facts()
        beat.write(state="ready", ready=True, **facts, memory=memory_snapshot(worker.device))
        for point in plan.points:
            go = worker.comm.broadcast_int(int(rank != 0 or time.time() < plan.point_deadline))
            if not go:
                beat.write(state="stopped", reason="the wall budget left no time for the point")
                return code
            result = worker.run_point(point, beat.write)
            if result["outcome"] != COMPLETE:
                beat.write(state="failed", point=point.name, error=result["reason"])
                code = 1
                return code
        beat.write(state="done")
        return code
    finally:
        if world > 1 and code == 0 and dist.is_initialized():
            dist.destroy_process_group()


def session_entry(rank: int, world: int, plan: SessionPlan, port: int) -> None:
    """Entry point of one spawned process of a session (``torch.multiprocessing``).

    A process whose session failed leaves at once, without collectives or interpreter teardown:
    a peer may wait in an all-reduce that never completes, which would hold
    ``destroy_process_group``; the parent then sees the exit code and ends the peers.

    Args:
        rank: This process's rank.
        world: The session's processes.
        plan: The plan.
        port: The rendezvous port.
    """
    code = run_session(rank, world, plan, port)
    if code:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


# --- the parent: sessions ------------------------------------------------------------------------


class Session(Protocol):
    """A running session as the parent sees it."""

    def wait(self, timeout: float) -> str | None:
        """Return None while it runs, ``ok`` when every process ended with 0, else why not."""
        ...

    def kill(self) -> None:
        """End every process of the session."""
        ...


class SpawnedSession:
    """A session's processes, started with ``torch.multiprocessing`` (spawn).

    The processes import this file by its module name: a script run as a program is found by its
    path (``__main__``), and when it was loaded as a module (the tests) its folder goes on
    ``sys.path`` for the start, which the processes inherit.

    Args:
        plan: The plan.
    """

    def __init__(self, plan: SessionPlan) -> None:
        folder = str(Path(__file__).resolve().parent)
        added = folder not in sys.path
        if added:
            sys.path.insert(0, folder)
        try:
            self.context = mp.start_processes(
                session_entry,
                args=(plan.world, plan, free_port()),
                nprocs=plan.world,
                join=False,
                start_method="spawn",
            )
        finally:
            if added:
                sys.path.remove(folder)

    def wait(self, timeout: float) -> str | None:
        """Wait up to ``timeout`` seconds; see ``Session.wait``."""
        try:
            done = self.context.join(timeout=timeout)
        except mp.ProcessRaisedException as err:
            return f"failed: process {err.error_index} raised {_failure(err)}"
        except mp.ProcessExitedException as err:
            return f"failed: process {err.error_index} exited with code {err.exit_code}"
        return "ok" if done else None

    def kill(self) -> None:
        """Terminate every process, then kill those still alive after 10 s."""
        processes = self.context.processes
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()


def heartbeats(out_dir: Path, index: int) -> list[dict[str, Any]]:
    """The last states of a session's processes that wrote one, with each file's time."""
    states = []
    for path in sorted((out_dir / "sessions" / f"s{index}").glob("rank*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        state["mtime"] = path.stat().st_mtime
        states.append(state)
    return states


def watch(
    session: Session, out_dir: Path, index: int, hard_deadline: float, limits: Limits
) -> str:
    """Wait for a session; end it at the hard deadline or when it stalls.

    Args:
        session: The running session.
        out_dir: The run-log folder.
        index: The session's number.
        hard_deadline: ``time.time()`` at which the run's budget ends.
        limits: ``stall_s``: the longest silence of every process.

    Returns:
        ``ok``, the session's failure (``failed: ...``), ``stopped: ...``, or ``stalled: ...``.
    """
    began = time.time()
    while True:
        state = session.wait(POLL_S)
        if state is not None:
            return state
        now = time.time()
        if now >= hard_deadline:
            session.kill()
            return "stopped: the wall budget max_wall_s ran out during the session"
        last = max([began, *(s["mtime"] for s in heartbeats(out_dir, index))])
        if now - last > limits.stall_s:
            session.kill()
            return f"stalled: no process reported progress for {limits.stall_s:g} s"


# --- the parent: results -------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def memory_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    r"""Peak memory of a point over its processes, and the headroom of the tightest one.

    For process $k$: $u_k = d_k - r_k$, the device memory outside PyTorch's allocator (the CUDA
    context, NCCL's buffers) at the point's end, $d_k$ the device's used memory and $r_k$ the
    allocator's reserved memory then; the headroom is $M_k - (\hat r_k + u_k)$, $M_k$ the device's
    total and $\hat r_k$ the point's peak reserved memory. The cache is never emptied between
    points, so $\hat r_k$ includes the cache the earlier points left (``reserved_before``), as a
    training run's allocator would hold it; the peak allocated memory is the point's own.

    Args:
        results: The processes' results.

    Returns:
        Bytes and counts, or None on the CPU (no device counters).
    """
    rows = [r["memory"] for r in results if r.get("memory", {}).get("device_total")]
    if not rows:
        return None
    before = [r["memory_before"] for r in results if r.get("memory", {}).get("device_total")]
    per_rank = []
    for row, start in zip(rows, before, strict=True):
        outside = max(0, row["device_used"] - row["reserved"])
        headroom = row["device_total"] - (row["peak_reserved"] + outside)
        per_rank.append({
            "device_total": row["device_total"], "peak_allocated": row["peak_allocated"],
            "peak_reserved": row["peak_reserved"], "outside_allocator": outside,
            "headroom": headroom, "alloc_retries": row["alloc_retries"] - start["alloc_retries"],
            "resident_before": start["allocated"], "reserved_before": start["reserved"],
        })  # fmt: skip
    tightest = min(per_rank, key=lambda r: r["headroom"])
    return {
        "device_total": tightest["device_total"],
        "peak_allocated": max(r["peak_allocated"] for r in per_rank),
        "peak_reserved": max(r["peak_reserved"] for r in per_rank),
        "outside_allocator": max(r["outside_allocator"] for r in per_rank),
        "headroom": tightest["headroom"],
        "headroom_fraction": tightest["headroom"] / tightest["device_total"],
        "alloc_retries": sum(r["alloc_retries"] for r in per_rank),
        "resident_before": max(r["resident_before"] for r in per_rank),
        "reserved_before": max(r["reserved_before"] for r in per_rank),
        "per_rank": per_rank,
    }


def aggregate(point: Point, results: Sequence[dict[str, Any] | None], world: int) -> dict[str, Any]:
    """Combine the processes' results of a point into its line of ``measurements.jsonl``.

    Args:
        point: The point.
        results: Each rank's result, None where a process wrote none.
        world: The session's processes.

    Returns:
        The record: the outcome (the worst over the processes), the step seconds (the slowest
        process's per step), the micro-step seconds, memory, and for a mix point the lengths of
        each timed step by rank and micro step.
    """
    present = [r for r in results if r is not None]
    worst = max(present, key=lambda r: SEVERITY.get(r["outcome"], 1))
    missing = [k for k, r in enumerate(results) if r is None]
    reason = worst["reason"]
    if missing and worst["outcome"] == COMPLETE:
        worst = {**worst, "outcome": ERROR}
        reason = f"no result from ranks {missing}"
    elif missing:
        reason += f" (no result from ranks {missing})"
    first = present[0]
    record: dict[str, Any] = {
        "point": point.name, "kind": point.kind, "gpus": point.gpus, "accum": point.accum,
        "tokens": point.tokens, "outcome": worst["outcome"], "reason": reason,
        "ranks": sorted(r["rank"] for r in present), "session": first["session"],
        "device_name": first["device_name"], "attn_implementation": first["attn_implementation"],
        "dtype": first["dtype"], "load_seconds": first["load_seconds"],
        "hypernet_parameters": first["hypernet_parameters"],
        "emitted_per_adapter": first["emitted_per_adapter"],
        "memory": memory_summary(present),
        "cpu_rss_peak": max((r["memory"].get("cpu_rss_peak") or 0) for r in present) or None,
        "seconds": max(r["seconds"] for r in present),
    }  # fmt: skip
    if point.kind == "hypernet":
        calls = first.get("calls", {})
        record["hypernet_ms"] = {
            mode: {**_stats([1000 * s for s in times]),
                   "p90": float(np.percentile([1000 * s for s in times], 90)) if times else None}
            for mode, times in calls.items()
        }  # fmt: skip
        return record
    steps = {}
    for result in present:
        for step in result.get("steps", []):
            row = steps.setdefault(step["step"], {"seconds": [], "timed": step["timed"]})
            row["seconds"].append(step["seconds"])
            row["nonfinite"] = row.get("nonfinite", False) or step["nonfinite"]
    complete_steps = {n: row for n, row in steps.items() if len(row["seconds"]) == len(present)}
    timed = [max(row["seconds"]) for n, row in sorted(complete_steps.items()) if row["timed"]]
    warm = [max(row["seconds"]) for n, row in sorted(complete_steps.items()) if not row["timed"]]
    timed_numbers = {n for n, row in complete_steps.items() if row["timed"]}
    micro = [
        m["h_forward_s"] + m["train_step_s"]
        for r in present
        for m in r.get("micro", [])
        if m["step"] in timed_numbers
    ]
    record["step_s"] = {**_stats(timed), "timed": timed, "warm": warm}
    record["micro_s"] = _stats(micro)
    record["nonfinite_steps"] = sum(1 for row in steps.values() if row.get("nonfinite"))
    record["remainder_s"] = (
        record["step_s"]["median"] - record["micro_s"]["median"]
        if timed and micro and point.accum == 1
        else None
    )
    tokens = {m["tokens"] for r in present for m in r.get("micro", [])}
    if point.kind == "fixed":
        record["tokens_actual"] = sorted(tokens)
    else:
        lengths = []
        for number in sorted(timed_numbers):
            by_rank = [[0] * point.accum for _ in range(world)]
            for result in present:
                for m in result.get("micro", []):
                    if m["step"] == number:
                        by_rank[result["rank"]][m["micro"]] = m["tokens"]
            lengths.append(by_rank)
        record["lengths"] = lengths
        drawn = [t for step in lengths for rank in step for t in rank]
        record["mean_tokens"] = float(np.mean(drawn)) if drawn else None
    return record


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect(
    plan: SessionPlan, state: str, out_dir: Path
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Read a finished session's results.

    Args:
        plan: The session's plan.
        state: What ``watch`` returned.
        out_dir: The run-log folder.

    Returns:
        The records of the points that have one (measured, failed, or cut short), and, when the
        setup failed before any point, why (the session's points then have no record).
    """
    records: dict[str, dict[str, Any]] = {}
    beats = heartbeats(out_dir, plan.index)
    for point in plan.points:
        folder = out_dir / "points" / point.name
        results = [_read_json(folder / f"rank{k}.json") for k in range(plan.world)]
        if any(r is not None for r in results):
            records[point.name] = aggregate(point, results, plan.world)
            continue
        running = [b for b in beats if b.get("state") == "running" and b.get("point") == point.name]
        if running and state != "ok":
            outcome = STALLED if state.startswith("stalled") else ERROR
            if state.startswith("stopped"):
                outcome = NOT_RUN
            step = running[0].get("step")
            records[point.name] = {
                "point": point.name, "kind": point.kind, "gpus": point.gpus,
                "accum": point.accum, "tokens": point.tokens, "outcome": outcome,
                "reason": f"{state} (at step {step} of the point)", "session": plan.index,
            }  # fmt: skip
    setup = None
    failed = [b for b in beats if b.get("state") == "failed" and not b.get("ready")]
    if failed:
        setup = f"rank {failed[0]['rank']}: {failed[0].get('error')}"
    elif state != "ok" and not any(b.get("ready") for b in beats):
        setup = state
    return records, setup


# --- the projection ------------------------------------------------------------------------------


def micro_table(records: Mapping[str, dict[str, Any]], gpus: int) -> list[tuple[int, float]]:
    """The measured micro-step seconds by length: complete fixed points on ``gpus`` processes."""
    rows = {
        r["tokens"]: r["micro_s"]["median"]
        for r in records.values()
        if r.get("kind") == "fixed"
        and r.get("gpus") == gpus
        and r.get("accum") == 1
        and r.get("outcome") == COMPLETE
        and r.get("micro_s", {}).get("median") is not None
    }
    return sorted(rows.items())


def micro_cost(table: Sequence[tuple[int, float]], tokens: Any) -> np.ndarray:
    r"""$m(T)$: the micro-step seconds at length $T$ from the measured table.

    Linear between measured lengths; constant below the shortest (the micro step's fixed costs
    dominate there, so this errs high); linear beyond the longest, on the last two points.

    Args:
        table: ``(tokens, seconds)`` rows, ascending, at least one.
        tokens: Lengths, any shape.

    Returns:
        Seconds, the shape of ``tokens``.
    """
    lengths = np.asarray(tokens, dtype=np.float64)
    xs = np.array([t for t, _ in table], dtype=np.float64)
    ys = np.array([s for _, s in table], dtype=np.float64)
    cost = np.interp(lengths, xs, ys)
    if len(xs) >= 2:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        cost = np.where(lengths > xs[-1], ys[-1] + (lengths - xs[-1]) * slope, cost)
    return cost


def predict_steps(
    lengths: Any, table: Sequence[tuple[int, float]], t_ddp: float, remainder: float
) -> np.ndarray:
    r"""Predicted optimizer-step seconds for episode lengths ``[steps, W, A]``.

    $\hat s = \max_k \sum_{j=1}^{A} m(T_{kj}) - (A - 1)\, t_\text{ddp} + \bar r$: each process runs
    its $A$ micro steps without waiting, DDP's all-reduce, which every measured $m$ includes,
    happens once per step, and the step waits for the slowest process.

    Args:
        lengths: Tokens, ``[steps, W, A]``.
        table: ``micro_table``'s rows.
        t_ddp: DDP's cost per synchronized micro step.
        remainder: The step's work outside its micro steps.

    Returns:
        Seconds per step, ``[steps]``.
    """
    lengths = np.asarray(lengths)
    accum = lengths.shape[2]
    per_rank = micro_cost(table, lengths).sum(axis=2)  # [steps, W]
    return per_rank.max(axis=1) - (accum - 1) * t_ddp + remainder


def _find(
    records: Mapping[str, dict[str, Any]], kind: str, gpus: int, tokens: int | None = None
) -> dict[str, Any] | None:
    for record in records.values():
        if record.get("kind") == kind and record.get("gpus") == gpus and (
            tokens is None or record.get("tokens") == tokens
        ):
            return record
    return None


def project(records: Mapping[str, dict[str, Any]], projection: Projection) -> dict[str, Any]:
    r"""The projection for P0.4: expected step seconds of the long run, and a week's steps.

    The model (``predict_steps``) takes $m(T)$ from the complete fixed points on the long run's
    $W$ processes; $t_\text{ddp} = s_W(T_0) - s_1(T_0)$, at the shortest length $T_0$ measured
    on $W$ processes and on one (0 when there is none; at least 0); $\bar r$ the median over
    those points of the step's seconds minus the micro step's. The expected step
    $\bar s_\text{model}$ is the mean of $\hat s$ over ``samples`` steps of lengths drawn from the
    histogram. A complete mix point calibrates it:
    $c = \bar s_\text{mix} / \bar{\hat s}_\text{mix}$, measured over predicted on the mix's own
    lengths, and $\bar s = c\, \bar s_\text{model}$. A
    week of $Q$ quota hours at $\rho$ quota hours per session hour gives
    $n = 3600\, Q / (\rho\, \bar s)$ steps of $W A$ episodes.

    Args:
        records: The run's records by point name.
        projection: The priced run.

    Returns:
        ``available`` and the model, the draw, the calibration, one pass, and the weekly table;
        ``reason`` when there is no measured length to build the model from.
    """
    gpus, accum = projection.gpus, projection.accum
    table = micro_table(records, gpus)
    if not table:
        return {"available": False, "reason": f"no complete fixed point on {gpus} processes"}
    remainders = [
        r["remainder_s"]
        for r in records.values()
        if r.get("kind") == "fixed" and r.get("gpus") == gpus and r.get("outcome") == COMPLETE
        and r.get("remainder_s") is not None
    ]  # fmt: skip
    remainder = float(np.median(remainders)) if remainders else 0.0
    t_ddp, t_ddp_at = 0.0, None
    for tokens, _ in table:
        single, multi = _find(records, "fixed", 1, tokens), _find(records, "fixed", gpus, tokens)
        if gpus > 1 and single and single["outcome"] == COMPLETE and multi:
            t_ddp = max(0.0, multi["step_s"]["median"] - single["step_s"]["median"])
            t_ddp_at = tokens
            break
    rng = np.random.default_rng(projection.seed)
    drawn = draw_lengths(projection, rng, (projection.samples, gpus, accum))
    predicted = predict_steps(drawn, table, t_ddp, remainder)
    model = float(predicted.mean())
    result: dict[str, Any] = {
        "available": True,
        "gpus": gpus,
        "accum": accum,
        "table": [{"tokens": t, "micro_s": s} for t, s in table],
        "t_ddp_s": t_ddp,
        "t_ddp_tokens": t_ddp_at,
        "remainder_s": remainder,
        "draw": {
            "samples": projection.samples,
            "mean_tokens": float(drawn.mean()),
            "histogram_mean_tokens": histogram_mean(projection),
            "measured_mean_tokens": projection.mean_tokens,
        },
        "model_step_s": {
            "mean": model,
            "p10": float(np.percentile(predicted, 10)),
            "p50": float(np.percentile(predicted, 50)),
            "p90": float(np.percentile(predicted, 90)),
        },
        "calibration": None,
    }
    step_s, basis = model, "model"
    mix = _find(records, "mix", gpus)
    if mix and mix["outcome"] == COMPLETE and mix.get("lengths") and mix["step_s"]["timed"]:
        on_mix = predict_steps(mix["lengths"], table, t_ddp, remainder)
        factor = float(np.mean(mix["step_s"]["timed"]) / on_mix.mean())
        result["calibration"] = {
            "point": mix["point"],
            "steps": len(mix["step_s"]["timed"]),
            "measured_mean_s": float(np.mean(mix["step_s"]["timed"])),
            "predicted_mean_s": float(on_mix.mean()),
            "factor": factor,
        }
        step_s, basis = factor * model, "calibrated"
    per_step = gpus * accum
    result["step_s"], result["basis"] = step_s, basis
    pass_steps = math.ceil(projection.episodes / per_step)
    pass_wall_h = pass_steps * step_s / 3600
    result["one_pass"] = {
        "episodes": projection.episodes,
        "steps": pass_steps,
        "wall_h": pass_wall_h,
        "quota_h": {f"{rate:g}": rate * pass_wall_h for rate in projection.rates},
        "gpu_s_per_episode": gpus * step_s / per_step,
    }
    weekly = []
    for rate in projection.rates:
        for quota in projection.quota_hours:
            wall_h = quota / rate
            steps = 3600 * wall_h / step_s
            weekly.append({
                "rate": rate, "quota_h": quota, "wall_h": wall_h, "steps": steps,
                "episodes": steps * per_step, "passes": steps * per_step / projection.episodes,
            })  # fmt: skip
    result["weekly"] = weekly
    return result


# --- the decision rule's reading -----------------------------------------------------------------


def _memory_fits(record: dict[str, Any], fraction: float) -> bool | None:
    memory = record.get("memory")
    if record.get("outcome") != COMPLETE or memory is None:
        return None
    return memory["headroom_fraction"] >= fraction and memory["alloc_retries"] == 0


def read_decision(records: Mapping[str, dict[str, Any]], cfg: ThroughputConfig) -> dict[str, Any]:
    """Read the pre-registered rule (``docs/prerun/<run-id>.md``, section 1) off the records.

    Args:
        records: The run's records by point name.
        cfg: The run config (``decision`` and the projection's process count).

    Returns:
        ``row`` (Pass, Borderline, Fail, or No result) with its reason; ``longer`` (fits, tight,
        does not fit, or not measured); ``hypernet`` (within the bound or not, None unmeasured).
    """
    decision, gpus = cfg.decision, cfg.projection.gpus
    cap = _find(records, "fixed", gpus, decision.cap_tokens)
    reading: dict[str, Any] = {}
    if cap is None or cap["outcome"] in (ERROR, STALLED, NOT_RUN, KILLED):
        why = "no point" if cap is None else f"{cap['outcome']}: {cap.get('reason')}"
        reading["row"], reading["reason"] = "No result", f"{decision.cap_tokens} tokens: {why}"
    elif cap["outcome"] == OOM:
        reading["row"], reading["reason"] = "Fail", f"out of memory: {cap['reason']}"
    else:
        fits = _memory_fits(cap, decision.headroom_fraction)
        memory = cap.get("memory")
        if fits is None:
            reading["row"] = "No result"
            reading["reason"] = "no device memory counters (a CPU run)"
        else:
            reading["row"] = "Pass" if fits else "Borderline"
            reading["reason"] = (
                f"headroom {memory['headroom'] / GIB:.2f} GiB of {memory['device_total'] / GIB:.2f}"
                f" ({100 * memory['headroom_fraction']:.0f}%, bound "
                f"{100 * decision.headroom_fraction:.0f}%), allocator retries "
                f"{memory['alloc_retries']}"
            )
    longer = _find(records, "fixed", gpus, decision.longer_tokens)
    if longer is None or longer["outcome"] not in (COMPLETE, OOM):
        reading["longer"] = "not measured"
    elif longer["outcome"] == OOM:
        reading["longer"] = "does not fit"
    else:
        fits = _memory_fits(longer, decision.headroom_fraction)
        reading["longer"] = {
            True: "fits", False: "tight", None: "no device memory counters (a CPU run)",
        }[fits]  # fmt: skip
    hyper = _find(records, "hypernet", 1)
    if hyper and hyper["outcome"] == COMPLETE and hyper.get("hypernet_ms", {}).get("eval"):
        median = hyper["hypernet_ms"]["eval"]["median"]
        reading["hypernet"] = median < decision.hypernet_budget_ms
    else:
        reading["hypernet"] = None
    return reading


# --- the run -------------------------------------------------------------------------------------


@dataclass
class RunState:
    """What the run did: records by point, sessions, and how it ended.

    Attributes:
        records: Point name to its record (also ``not run`` ones at the end).
        sessions: One entry per session: plan summary, state, setup failure, seconds.
        status: ``complete``, ``stopped``, ``killed``, or ``failed``.
        reason: Why, when not complete.
    """

    records: dict[str, dict[str, Any]]
    sessions: list[dict[str, Any]]
    status: str = "complete"
    reason: str = ""


def _not_run(point: Point, reason: str) -> dict[str, Any]:
    return {
        "point": point.name, "kind": point.kind, "gpus": point.gpus, "accum": point.accum,
        "tokens": point.tokens, "outcome": NOT_RUN, "reason": reason,
    }  # fmt: skip


def run_points(
    cfg: ThroughputConfig,
    model_dir: str | None,
    started: float,
    reused: Mapping[str, dict[str, Any]],
    launch: Callable[[SessionPlan], Session] = SpawnedSession,
) -> RunState:
    """Run the points not yet measured, in sessions of consecutive points with equal ``gpus``.

    A session that ends early (a point's failure, a kill, a stall, the budget) leaves its later
    points for a new session, up to ``limits.max_sessions``; a session whose setup failed takes
    its points with it. A kill criterion stops the run.

    Args:
        cfg: The run config.
        model_dir: The base model's folder (None: the tiny model).
        started: ``time.time()`` at the script's start.
        reused: Records kept from ``--resume``.
        launch: Starts a session (the tests run one in this process).

    Returns:
        The run's state.
    """
    out_dir, limits = Path(cfg.out_dir), cfg.limits
    run = RunState(dict(reused), [])
    pending = [p for p in cfg.points if p.name not in run.records]
    while pending:
        now = time.time()
        if now - started > limits.max_wall_s - limits.reserve_s:
            run.status, run.reason = "stopped", "the wall budget max_wall_s ran out"
            break
        if len(run.sessions) >= limits.max_sessions:
            run.status, run.reason = "failed", f"max_sessions ({limits.max_sessions}) reached"
            break
        world = pending[0].gpus
        group = []
        for point in pending:
            if point.gpus != world:
                break
            group.append(point)
        plan = SessionPlan(
            index=len(run.sessions),
            world=world,
            points=tuple(group),
            train=train_config(cfg, model_dir, world, group[0].accum),
            out_dir=str(out_dir),
            started=started,
            point_deadline=started + limits.max_wall_s - limits.reserve_s,
            max_step_s=limits.max_step_s,
            projection=cfg.projection,
            seed=cfg.seed,
        )
        processes = f"{world} process" + ("es" if world > 1 else "")
        print(f"session {plan.index}: {processes}, points {[p.name for p in group]}", flush=True)
        began = time.time()
        state = watch(launch(plan), out_dir, plan.index, started + limits.max_wall_s, limits)
        records, setup = collect(plan, state, out_dir)
        run.sessions.append({
            "index": plan.index, "world": world, "points": [p.name for p in group],
            "state": state, "setup_failure": setup, "seconds": time.time() - began,
        })  # fmt: skip
        print(f"session {plan.index}: {state}", flush=True)
        if setup is not None:
            for point in group:
                run.records[point.name] = {**_not_run(point, ""), "outcome": ERROR,
                                           "reason": f"session setup failed: {setup}"}  # fmt: skip
            pending = pending[len(group) :]
            continue
        run.records.update(records)
        killed = [r for r in records.values() if r["outcome"] == KILLED]
        if killed:
            run.status, run.reason = "killed", f"{killed[0]['point']}: {killed[0]['reason']}"
            break
        if state.startswith("stopped"):
            run.status, run.reason = "stopped", state
            break
        pending = [p for p in pending if p.name not in run.records]
    for point in cfg.points:
        if point.name not in run.records:
            run.records[point.name] = _not_run(point, run.reason or "not reached")
    if run.status == "complete":
        bad = [r for r in run.records.values() if r["outcome"] not in FINAL]
        if bad:
            run.status = "failed"
            run.reason = "; ".join(f"{r['point']}: {r['outcome']}" for r in bad)
    return run


def reusable(previous: Path, cfg: ThroughputConfig) -> dict[str, dict[str, Any]]:
    """Records of a previous run that a resumed run keeps.

    A point is kept when it ended ``complete`` or ``oom`` under the same point settings and the
    same training step (the ``train`` section and the seed).

    Args:
        previous: The previous run's ``out_dir``.
        cfg: This run's config.

    Returns:
        Point name to record.

    Raises:
        FileNotFoundError: If ``previous`` holds no ``measurements.jsonl`` or ``config.yaml``.
    """
    old = yaml.safe_load((previous / "config.yaml").read_text(encoding="utf-8"))
    same_step = old.get("train") == cfg.to_dict()["train"] and old.get("seed") == cfg.seed
    points = {p.name: _plain(asdict(p)) for p in cfg.points}
    old_points = {p["name"]: p for p in old.get("points", [])}
    kept: dict[str, dict[str, Any]] = {}
    for line in (previous / "measurements.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        name = record.get("point")
        if (
            same_step
            and name in points
            and old_points.get(name) == points[name]
            and record.get("outcome") in FINAL
        ):
            kept[name] = {**record, "resumed_from": str(previous)}
    return kept


def prepare_out_dir(
    out_dir: Path, cfg: ThroughputConfig, previous: Path | None, kept: Mapping[str, Any]
) -> None:
    """Remove this script's earlier outputs from ``out_dir``, keeping the kept points' folders.

    The notebook's own files there are left alone. A kept point's folder is copied in when the
    previous run's folder is another one.

    Args:
        out_dir: The run-log folder.
        cfg: This run's config.
        previous: The previous run's ``out_dir`` (``--resume``), or None.
        kept: The kept records by point name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OWN_OUTPUTS:
        (out_dir / name).unlink(missing_ok=True)
    shutil.rmtree(out_dir / "sessions", ignore_errors=True)
    same = previous is not None and previous.resolve() == out_dir.resolve()
    for folder in sorted((out_dir / "points").glob("*")) if (out_dir / "points").is_dir() else []:
        if not (same and folder.name in kept):
            shutil.rmtree(folder, ignore_errors=True)
    if previous is not None and not same:
        for name in kept:
            source = previous / "points" / name
            if source.is_dir():
                shutil.copytree(source, out_dir / "points" / name, dirs_exist_ok=True)


# --- inputs and provenance -----------------------------------------------------------------------


def model_folder_parts(model_id: str) -> tuple[str, ...]:
    """The path ending under which Kaggle mounts a Kaggle Models handle.

    Why: the handle ``<owner>/<model>/<framework>/<variation>/<version>`` is mounted in a folder
    whose path ends with ``<model>/<framework>/<variation>/<version>``, the framework in lower
    case: under ``models/<owner>/`` in 2026 (K0), directly under the input root in 2025. As
    ``scripts/cache_embeddings.py`` (which a script notebook cannot import here).

    Args:
        model_id: The handle.

    Returns:
        Its four parts after the owner, lower case.

    Raises:
        ValueError: If ``model_id`` is not five non-empty parts.
    """
    parts = model_id.split("/")
    if len(parts) != 5 or not all(parts):
        raise ValueError(f"model_id must be <owner>/<model>/<framework>/<variation>/<version>, "
                         f"got {model_id!r}")  # fmt: skip
    return tuple(part.lower() for part in parts[1:])


def find_model_dir(root: Path, model_id: str) -> Path:
    """Find the base model's folder under ``root`` by the end of its path.

    Args:
        root: Folder searched, such as ``/kaggle/input``.
        model_id: The Kaggle Models handle.

    Returns:
        The one folder holding ``config.json`` whose path ends with the handle's parts.

    Raises:
        FileNotFoundError: If no folder or more than one matches.
    """
    tail = (*model_folder_parts(model_id), "config.json")
    top = Path(os.path.abspath(root))
    found, seen = [], set()
    for folder, dirs, files in os.walk(top, followlinks=True):
        real = os.path.realpath(folder)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        level = len(Path(folder).relative_to(top).parts)
        dirs[:] = sorted(
            d for d in dirs
            if d not in SKIPPED_DIRS and not d.endswith((".dist-info", ".egg-info"))
        ) if level + 1 < SEARCH_DEPTH else []  # fmt: skip
        if "config.json" in files:
            parts = tuple(part.lower() for part in (Path(folder) / "config.json").parts)
            if parts[-len(tail) :] == tail:
                found.append(Path(folder))
    if len(found) != 1:
        raise FileNotFoundError(
            f"found {len(found)} model folders ending in {'/'.join(tail[:-1])} with config.json "
            f"under {root}, expected 1: {[str(f) for f in found[:5]]}"
        )
    return found[0]


def git_commit(repo: Path) -> tuple[str | None, bool | None]:
    """Return the checkout's HEAD and whether its tree has changes; ``(None, None)`` outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return sha, bool(status.strip())


def versions() -> dict[str, Any]:
    """Library versions of this process; flash_attn's from its package metadata when present."""
    try:
        flash = metadata.version("flash_attn")
    except metadata.PackageNotFoundError:
        flash = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "flash_attn": flash,
    }


# --- the report ----------------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _gib(value: Any) -> str:
    return "n/a" if value is None else f"{value / GIB:.2f}"


def _point_rows(cfg: ThroughputConfig, records: Mapping[str, dict[str, Any]]) -> list[str]:
    rows = [
        "| Point | GPUs | Accum | Tokens | Outcome | Step s (median) | Step s (min to max) | "
        "Micro step s | Peak allocated GiB | Reserved before GiB | Peak reserved GiB | "
        "Headroom GiB |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in cfg.points:
        r = records[point.name]
        steps, memory = r.get("step_s") or {}, r.get("memory") or {}
        tokens = point.tokens if point.kind == "fixed" else (
            f"mix, mean {r['mean_tokens']:.0f}" if r.get("mean_tokens") else point.kind
        )  # fmt: skip
        spread = (
            f"{steps['min']:.2f} to {steps['max']:.2f}" if steps.get("min") is not None else "n/a"
        )
        rows.append(
            f"| {point.name} | {point.gpus} | {point.accum} | {tokens} | {r['outcome']} | "
            f"{_fmt(steps.get('median'))} | {spread} | "
            f"{_fmt((r.get('micro_s') or {}).get('median'))} | "
            f"{_gib(memory.get('peak_allocated'))} | {_gib(memory.get('reserved_before'))} | "
            f"{_gib(memory.get('peak_reserved'))} | {_gib(memory.get('headroom'))} |"
        )
    return rows


def render_summary(
    cfg: ThroughputConfig,
    run: RunState,
    projection: dict[str, Any],
    reading: dict[str, Any],
    info: Mapping[str, Any],
) -> str:
    """Write the short human-readable report of a run.

    Args:
        cfg: The run config.
        run: The run's state.
        projection: ``project``'s result.
        reading: ``read_decision``'s result.
        info: ``run.json``.

    Returns:
        Markdown text.
    """
    records = run.records
    measured = [r for r in records.values() if r.get("device_name")]
    first = measured[0] if measured else {}
    done = sum(1 for r in records.values() if r["outcome"] in FINAL)
    lines = [
        f"# {cfg.run_id}: throughput and memory of H's meta-training step",
        "",
        f"- Status: **{run.status}**" + (f": {run.reason}" if run.reason else "")
        + f"; {done} of {len(cfg.points)} points measured; wall "
        f"{_fmt(info.get('elapsed_s'), 0)} s.",
        f"- Commit: {info.get('commit')} (dirty: {info.get('dirty')}); smoke: {info.get('smoke')};"
        f" resumed from: {info.get('resume')}.",
        f"- Device: {first.get('device_name', 'n/a')}; attention "
        f"{first.get('attn_implementation', 'n/a')}; base {first.get('dtype', 'n/a')}; "
        f"model loaded in {_fmt(first.get('load_seconds'), 1)} s; versions "
        + ", ".join(f"{k} {v}" for k, v in info.get("versions", {}).items()) + ".",
        "- Step: P3.5's train_step (arc26.hyper.train) with "
        f"{', '.join(f'{k} {cfg.train[k]}' for k in ('lr', 'clip', 'seed'))}; H "
        f"{first.get('hypernet_parameters', 'n/a')} parameters, "
        f"{first.get('emitted_per_adapter', 'n/a')} values per adapter; synthetic episodes in the "
        "notebook format at exact lengths; one fixed z_bar.",
        "",
        "## Steps and memory",
        "",
        "Step seconds are the slowest process's per optimizer step over the timed steps; the micro "
        "step is H's forward plus the episode's forward and backward; headroom is the device "
        "total minus the peak reserved memory and the memory outside the allocator, on the "
        "tightest GPU. The allocator's cache is never emptied between points, as in training: "
        "the peak reserved memory includes what the earlier points left (reserved before), the "
        "peak allocated memory is the point's own.",
        "",
        *_point_rows(cfg, records),
        "",
    ]
    problems = [r for r in records.values() if r["outcome"] != COMPLETE]
    for r in problems:
        lines.append(f"- {r['point']}: {r['outcome']}: {r.get('reason') or 'no reason given'}")
    for r in records.values():
        if r.get("tokens_actual") and r["tokens_actual"] != [r["tokens"]]:
            lines.append(
                f"- {r['point']}: the base read {r['tokens_actual']} tokens per episode, not "
                f"{r['tokens']}: the tokenizer splits the format otherwise"
            )
    nonfinite = [r for r in records.values() if r.get("nonfinite_steps")]
    for r in nonfinite:
        lines.append(f"- {r['point']}: {r['nonfinite_steps']} steps with a non-finite loss or "
                     "gradient norm (their update skipped, their time kept)")  # fmt: skip
    hyper = _find(records, "hypernet", 1)
    if hyper and hyper.get("hypernet_ms"):
        ms = hyper["hypernet_ms"]
        lines += [
            "",
            f"H's forward for all layers and modules on one GPU: median "
            f"{_fmt(ms.get('eval', {}).get('median'))} ms in evaluation (no autograd, as at test "
            f"time), {_fmt(ms.get('train', {}).get('median'))} ms in training (dropout, autograd)"
            f" over {ms.get('eval', {}).get('n')} calls each; bound "
            f"{cfg.decision.hypernet_budget_ms:g} ms (B5 P3.3).",
        ]
    if info.get("smoke"):
        lines += ["", "## Decision rule (a smoke run has no brief)", ""]
        lines.append("Smoke run: a tiny random base on the CPU checks the code path; the numbers "
                     "say nothing about an L4.")  # fmt: skip
    else:
        lines += ["", f"## Decision rule (docs/prerun/{cfg.run_id}.md, section 1)", ""]
    hyper_text = {True: "within the bound", False: "over the bound", None: "not measured"}
    lines += [
        f"- {cfg.decision.cap_tokens} tokens on {cfg.projection.gpus} GPUs (the cap): "
        f"**{reading['row']}**: {reading['reason']}.",
        f"- {cfg.decision.longer_tokens} tokens (a longer cap): {reading['longer']}.",
        f"- H's forward: {hyper_text[reading['hypernet']]}.",
        "",
        "## Projection for P0.4",
        "",
    ]
    if not projection.get("available"):
        lines.append(f"Not available: {projection.get('reason')}.")
    else:
        lines += _projection_lines(cfg, projection)
    lines += [
        "",
        "Logs: measurements.jsonl, projection.json, points/<name>/rank<k>.json, "
        "points/<name>/timing-rank<k>.jsonl, sessions/s<i>/rank<k>.json, run.json, status.json.",
        "",
    ]
    return "\n".join(lines)


def _projection_lines(cfg: ThroughputConfig, p: dict[str, Any]) -> list[str]:
    pr = cfg.projection
    table = ", ".join(f"{row['tokens']}: {row['micro_s']:.3f}" for row in p["table"])
    draw, model = p["draw"], p["model_step_s"]
    lines = [
        f"- Micro step $m(T)$ in seconds by tokens ({p['gpus']} GPUs, interpolated linearly, "
        f"constant below the shortest): {table}. DDP's cost per synchronized micro step "
        f"$t_\\text{{ddp}}$ = {p['t_ddp_s']:.3f} s"
        + (f" (at {p['t_ddp_tokens']} tokens)" if p["t_ddp_tokens"] else " (not measured: 0)")
        + f"; the step's remainder $\\bar r$ = {p['remainder_s']:.3f} s.",
        f"- Draw: {draw['samples']} steps of {p['gpus']} x {p['accum']} episodes from P3.4's "
        f"training histogram; mean {draw['mean_tokens']:.0f} tokens (the bins' "
        f"{draw['histogram_mean_tokens']:.0f}, measured {draw['measured_mean_tokens']:.1f}).",
        f"- Model step: mean {model['mean']:.2f} s (p10 {model['p10']:.2f}, p50 "
        f"{model['p50']:.2f}, p90 {model['p90']:.2f}).",
    ]
    cal = p.get("calibration")
    if cal:
        lines.append(
            f"- Measured {cal['point']}: {cal['measured_mean_s']:.2f} s over {cal['steps']} steps "
            f"against {cal['predicted_mean_s']:.2f} s predicted on its lengths: factor "
            f"{cal['factor']:.3f}" + (" (more than 15% from 1: the model misses a cost)"
                                      if abs(cal["factor"] - 1) > 0.15 else "") + "."
        )  # fmt: skip
    else:
        lines.append("- No complete mix point: the step is the model's, uncalibrated.")
    one = p["one_pass"]
    quota = "; ".join(f"{q:.1f} quota h at {rate}x" for rate, q in one["quota_h"].items())
    lines += [
        f"- **Expected step: {p['step_s']:.2f} s** ({p['basis']}) for {p['gpus'] * p['accum']} "
        f"episodes: {one['gpu_s_per_episode']:.2f} GPU-seconds per episode.",
        f"- One pass over {one['episodes']} episodes: {one['steps']} steps, "
        f"{one['wall_h']:.1f} h of session; {quota}.",
        "",
        "| Quota h charged per session h | Quota h per week | Session h | Steps | Episodes | "
        "Passes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p["weekly"]:
        lines.append(
            f"| {row['rate']:g} | {row['quota_h']:g} | {row['wall_h']:.1f} | {row['steps']:.0f} | "
            f"{row['episodes']:.0f} | {row['passes']:.2f} |"
        )
    lines += [
        "",
        f"Readings of the quota rate: {', '.join(f'{r:g}' for r in pr.rates)} quota hours per hour "
        "of a 4xL4 session. Not priced here: the encoder's pass over the demonstrations and the "
        "validations (docs/prerun/P3.6-01.md, section 6).",
    ]
    return lines


# --- command line --------------------------------------------------------------------------------


def _write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, allow_nan=True) + "\n").encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; no option other than ``--config`` names a config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, type=Path, help="configs/train_throughput.yaml")
    parser.add_argument(
        "--smoke", action="store_true", help="CPU run: tiny random Qwen3, small points, gloo"
    )
    parser.add_argument("--commit", help="code commit to record when not in a git checkout")
    parser.add_argument("--out-dir", help="run-log folder (config: out_dir)")
    parser.add_argument("--input-root", help="folder searched for the model (config: input_root)")
    parser.add_argument("--model-dir", help="Hugging Face model folder; no search (model_dir)")
    parser.add_argument("--resume", help="a previous run's out_dir: keep its finished points")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        The exit code: 0 complete, 1 failed, 2 invalid config or command line, 3 stopped by the
        wall budget, 4 killed by the step-time criterion.
    """
    started = time.time()
    args = build_parser().parse_args(argv)
    if args.smoke and args.model_dir:
        print("--smoke builds its own tiny model: drop --model-dir", file=sys.stderr)
        return EXIT_CONFIG
    overrides = {
        "out_dir": args.out_dir, "input_root": args.input_root, "model_dir": args.model_dir,
    }  # fmt: skip
    try:
        cfg = load_config(args.config, args.smoke, overrides)
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return EXIT_CONFIG
    out_dir = Path(cfg.out_dir)
    resume = Path(args.resume) if args.resume else None
    try:
        reused = reusable(resume, cfg) if resume is not None else {}
    except (OSError, ValueError, yaml.YAMLError) as err:
        print(f"--resume {resume}: {err}", file=sys.stderr)
        return EXIT_CONFIG
    prepare_out_dir(out_dir, cfg, resume, reused)
    commit, dirty = (args.commit, None) if args.commit else git_commit(REPO)
    info: dict[str, Any] = {
        "run_id": cfg.run_id,
        "smoke": args.smoke,
        "argv": list(sys.argv[1:] if argv is None else argv),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "resume": args.resume,
        "versions": versions(),
        "input_root": cfg.input_root,
        "model_dir": None,
    }
    code, run = EXIT_FAILED, RunState({}, [], "failed", "")
    try:
        atomic_write_bytes(
            out_dir / "config.yaml", yaml.safe_dump(cfg.to_dict(), sort_keys=False).encode()
        )
        _write_json(out_dir / "run.json", info)
        model_dir = cfg.model_dir
        if not args.smoke and model_dir is None:
            model_dir = str(find_model_dir(Path(cfg.input_root), cfg.train["model_id"]))
        info["model_dir"] = model_dir
        print(f"model folder: {model_dir or 'the tiny random model (smoke)'}", flush=True)
        run = run_points(cfg, model_dir, started, reused)
        code = {"complete": EXIT_OK, "stopped": EXIT_STOPPED, "killed": EXIT_KILLED}.get(
            run.status, EXIT_FAILED
        )
    except KeyboardInterrupt:
        run.reason = "interrupted"
    except Exception as err:
        run.reason = _failure(err)
        (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    for point in cfg.points:
        run.records.setdefault(point.name, _not_run(point, run.reason or "not reached"))
    records = [run.records[p.name] for p in cfg.points]
    text = "".join(json.dumps(r, allow_nan=True) + "\n" for r in records)
    atomic_write_bytes(out_dir / "measurements.jsonl", text.encode("utf-8"))
    projection = project(run.records, cfg.projection)
    _write_json(out_dir / "projection.json", projection)
    reading = read_decision(run.records, cfg)
    info.update({"sessions": run.sessions, "elapsed_s": time.time() - started})
    _write_json(out_dir / "run.json", info)
    _write_json(
        out_dir / "status.json",
        {
            "status": run.status, "exit_code": code, "reason": run.reason,
            "points": {r["point"]: r["outcome"] for r in records}, "decision": reading,
            "elapsed_s": info["elapsed_s"],
        },
    )  # fmt: skip
    summary = render_summary(cfg, run, projection, reading, info)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
