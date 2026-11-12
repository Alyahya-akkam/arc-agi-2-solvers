r"""Injection parity on the GPU: P3.1's injected LoRA against PEFT, bf16, DEV4 (task P3.1; B5 P3.1).

Purpose: B5 P3.1's acceptance on the hardware the seam runs on. Meta-training (P3.5) trains H
through P3.1's differentiable injection (``arc26.hyper.inject``); test time (P4.1) loads H's
adapter through PEFT. The CPU tier (``tests/test_inject.py``) found both paths bitwise equal, which
says nothing about GPU kernels: cuBLAS picks kernels by shape, stride, and alignment. This script
measures the seam on one L4 in bf16 and computes the decision row of the pre-run brief
``docs/prerun/P3.1-01.md`` (section 1). In bf16 the bound $10^{-3}$ means bitwise-equal logits
for $|z| \ge 0.25$ (``docs/STATE.md``, conflict 3). Runbook ``docs/runbooks/inject_parity.md``;
module ``docs/modules/inject.md``.

Inputs: the YAML config (``configs/inject_parity.yaml``); the base model folder and the
competition's challenges file, found by search under ``input_root`` as the script notebook's
environment check finds them, or given with ``--model-dir`` and ``--challenges``. Only DEV4's
train pairs and test inputs are read; no solutions file is opened (BRIEF A0). Outputs in
``out_dir`` (``run_logs/`` in the notebook's working folder, pulled to ``runs/P3.1-01/``):
``config.yaml`` (resolved), ``parity.json`` (provenance, every measurement, the decision; written
after every stage), ``timing-parity.jsonl`` (one ``{puzzle_id, op, seconds, tokens}`` record per
GPU op), ``summary.md``, and ``error.log`` on a failure.

Method: one fixed random adapter, seeded, on every target projection $m$ (H's scope, rank $r$,
rsLoRA scale $s = \alpha / \sqrt{r}$): $A_m$ drawn as PEFT draws a new ``lora_A``, $B_m$ a random
direction scaled so that $\|s B_m A_m\|_F = \rho \|W_m\|_F$. The frozen bf16 base reads each DEV4
sequence alone (the notebook's decode prompt per test input and its training text, identity view,
cut by the notebook's ``cut_to_len``) through two paths: P3.1's ``inject``, and PEFT's
``lora.Linear`` loaded from the exported adapter with ``PeftModel.from_pretrained`` and its
defaults. Per sequence the bf16 logits $z^\text{inj}$ and $z^\text{peft}$ are compared: the
largest $|z^\text{inj} - z^\text{peft}|$, the count of differing elements, the largest distance in
bf16 ULPs over the elements the bound $10^{-3}$ does not cover (and over all, a diagnostic), and
the positions whose argmax differs. An element is within conflict 3's tolerance when it is below
the bound or at most 2 ULPs off. Controls: with $B = 0$ the injected path must equal the base
bitwise; the adapter must move the logits (mean $|z^\text{peft} - z^\text{base}|$ at least
``min_mean_effect``, so the comparison is not vacuous); the base and PEFT passes, repeated, show
whether the stack is deterministic; two adapter swaps (``load_adapter`` into the existing name,
``set_peft_model_state_dict``) are P4.1's diagnostics. Gradient check: on each
task's training sequence, the query-output cross-entropy (P3.5's loss) backward through the
injected path with non-reentrant checkpointing must give every $\nabla A_m$ and $\nabla B_m$
finite and non-zero (a backward that raises is recorded on its text and counts as no gradient);
the largest relative difference to PEFT's ``lora_A``/``lora_B`` gradients, to a repeat of the
injected backward, and the calls per projection (2 when checkpointing recomputed the layers) are
diagnostics. ``decide`` turns the measurements into Pass, Human decides, Fail, or Error.

Exit codes: 0 measured and decided (the outcome is in ``parity.json`` and ``summary.md``),
1 failed (outcome Error), 2 invalid config or command line, 3 stopped at the wall budget (Error).

Usage::

    python scripts/inject_parity.py --config configs/inject_parity.yaml --commit SHA
    python scripts/inject_parity.py --config configs/inject_parity.yaml --model-dir DIR \
        --challenges FILE --out-dir DIR --device cuda:0
    python scripts/inject_parity.py --config configs/inject_parity.yaml --smoke
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# huggingface_hub reads this when it is first imported; nothing is downloaded, Kaggle is offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from peft import PeftModel, set_peft_model_state_dict  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Qwen3ForCausalLM,
)

import arc26  # noqa: E402
from arc26.data import subset_ids  # noqa: E402
from arc26.hyper.encoder import render_pair  # noqa: E402
from arc26.hyper.hypernet import PROJECTIONS  # noqa: E402
from arc26.hyper.inject import (  # noqa: E402
    ADAPTER_WEIGHTS,
    InjectableLinear,
    LoRAFactors,
    inject,
    rslora_scaling,
    save_adapter,
    to_peft_state_dict,
    wrap_targets,
)
from arc26.hyper.train import (  # noqa: E402
    atomic_write_bytes,
    query_ce,
    resolve_attention,
    standin_tokenizer,
)
from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter  # noqa: E402
from arc26.timing import TimingLogger  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_STOPPED = 0, 1, 2, 3
STATUS = {EXIT_OK: "complete", EXIT_FAILED: "failed", EXIT_STOPPED: "stopped"}
# The decision rows of the brief (section 1) and the class of one comparison.
PASS, HUMAN, FAIL, ERROR = "Pass", "Human decides", "Fail", "Error"
EXACT, WITHIN, BEYOND = "exact", "within", "beyond"
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
ATTENTION = ("auto", "flash_attention_2", "sdpa", "eager")
SECTIONS = ("scope", "adapter", "decision")
# Paths a smoke run reads relative to the repository root, so it runs from any folder.
PATH_KEYS = ("input_root", "challenges", "model_dir", "out_dir")
# The script notebook's file search (find_inputs in scripts/build_kaggle_notebook.py, which this
# script cannot import on Kaggle): the same depth and skipped folders find the same files.
SEARCH_DEPTH = 8
SKIPPED_DIRS = frozenset({"site-packages", "dist-packages", "__pycache__", "node_modules", ".git"})
SOLUTION_MARKER = "solution"
RESULTS_FILE = "parity.json"
SUMMARY_FILE = "summary.md"
ERROR_FILE = "error.log"
TIMING_WORKER = "parity"
ADAPTER_NAME = "default"
# inject.py replays this PEFT version's op sequence (docs/modules/inject.md); another is flagged.
PEFT_REPLAYED = "0.17.1"
PEFT_PREFIX = "base_model.model.model."
DECODER = "model."  # factor keys are relative to the causal LM's decoder, as in inject
# Every training text's gradient record has these keys (None until measured), so a failed
# backward leaves the same schema.
GRADIENT_KEYS = (
    "loss_injected", "loss_injected_repeat", "tensors", "finite", "zero_tensors", "missing",
    "min_max_abs", "flows", "error", "rel_diff_repeat", "rel_diff_repeat_at", "repeat_error",
    "loss_peft", "loss_equal", "rel_diff_peft", "rel_diff_peft_at", "peft_error",
    "calls_per_projection",
)  # fmt: skip
# --smoke: synthetic tasks under DEV4's ids with DEV4's test-input counts (1, 2, 1, 1; K0) and
# random grids: (train pairs, input shape, output shape, test inputs). With the smoke's
# max_tokens of 2048 the first task's training text (5 pairs of 426 tokens) and decode prompt
# (cap 2048 - 931) are cut by the notebook's cut_to_len, and the others are not.
SMOKE_TASKS: dict[str, tuple[int, tuple[int, int], tuple[int, int], int]] = {
    "0934a4d8": (5, (14, 14), (14, 14), 1),
    "36a08778": (3, (6, 8), (6, 8), 2),
    "981571dc": (4, (10, 10), (10, 10), 1),
    "aa4ec2a5": (2, (5, 7), (3, 3), 1),
}
SMOKE_SEED = 0


class ConfigError(ValueError):
    """The config file or a command-line override is invalid."""


class WallTimeExceeded(RuntimeError):
    """The script's wall budget ran out between two passes."""


# --- configuration -------------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Scope:
    """The adapter scope: H's, from ``configs/hypernet.yaml`` (ADR-002).

    Attributes:
        rank: LoRA rank r of every pair.
        alpha: LoRA alpha; the scale is ``rslora_scaling(rank, alpha)``.
        target_modules: Projection names adapted in every decoder layer.

    Raises:
        ValueError: If a value has the wrong type or range, or a name is not a projection.
    """

    rank: int
    alpha: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        _require(_is_int(self.rank) and self.rank >= 1, "scope.rank must be an integer >= 1")
        _require(_is_number(self.alpha) and self.alpha > 0, "scope.alpha must be positive")
        names = self.target_modules
        _require(
            bool(names) and len(set(names)) == len(names) and set(names) <= PROJECTIONS.keys(),
            f"scope.target_modules must be distinct names from {list(PROJECTIONS)}",
        )


@dataclass(frozen=True)
class AdapterSettings:
    r"""The fixed random adapter.

    Attributes:
        seed: Seed of $A$ and of $B$'s direction.
        relative_norm: $\rho$, the ratio $\|s B A\|_F / \|W\|_F$ for every projection.

    Raises:
        ValueError: If a value has the wrong type or range.
    """

    seed: int
    relative_norm: float

    def __post_init__(self) -> None:
        _require(_is_int(self.seed) and self.seed >= 0, "adapter.seed must be an integer >= 0")
        _require(
            _is_number(self.relative_norm) and self.relative_norm > 0,
            "adapter.relative_norm must be positive",
        )


@dataclass(frozen=True)
class Thresholds:
    """The decision rule's numbers (the brief, section 1).

    Attributes:
        max_abs_diff: B5 P3.1's bound on the largest logit difference, injected against PEFT.
        max_ulp: The largest distance in bf16 ULPs the human may accept for an element at or
            above ``max_abs_diff`` (conflict 3).
        min_mean_effect: The smallest mean change of the logits by the adapter, per sequence,
            for the comparison to count.

    Raises:
        ValueError: If a value has the wrong type or range.
    """

    max_abs_diff: float
    max_ulp: int
    min_mean_effect: float

    def __post_init__(self) -> None:
        _require(
            _is_number(self.max_abs_diff) and self.max_abs_diff > 0,
            "decision.max_abs_diff must be positive",
        )
        _require(_is_int(self.max_ulp) and self.max_ulp >= 0, "decision.max_ulp must be >= 0")
        _require(
            _is_number(self.min_mean_effect) and self.min_mean_effect > 0,
            "decision.min_mean_effect must be positive",
        )


@dataclass(frozen=True)
class ParityConfig:
    """One parity run; ``configs/inject_parity.yaml`` gives every value with its reason.

    Attributes:
        run_id: Pre-run brief id.
        input_root: Folder searched for the challenges file and the model.
        competition: Folder name that marks the competition's copy of the challenges file.
        challenges_file: File name of the challenges.
        challenges: The challenges file; None finds it by search.
        model_slug: Folder name the base model's ``config.json`` sits under.
        model_dir: The model folder; None finds it by search.
        model_id: Weights and precision of the base, recorded.
        subset: Task subset of ``arc26.data`` (DEV4).
        device: ``cpu`` or ``cuda:<index>``.
        dtype: Base precision, a key of ``DTYPES``.
        attn_implementation: One of ``ATTENTION``; ``auto`` is meta-training's rule.
        max_tokens: The notebook's max_seq_length.
        scope: The adapter scope.
        adapter: The fixed random adapter.
        decision: The decision rule's numbers.
        max_wall_s: Wall budget in seconds from the script's start.
        out_dir: Run-log folder.

    Raises:
        ValueError: If a value has the wrong type or range.
    """

    run_id: str
    input_root: str
    competition: str
    challenges_file: str
    challenges: str | None
    model_slug: str
    model_dir: str | None
    model_id: str
    subset: str
    device: str
    dtype: str
    attn_implementation: str
    max_tokens: int
    scope: Scope
    adapter: AdapterSettings
    decision: Thresholds
    max_wall_s: float
    out_dir: str

    def __post_init__(self) -> None:
        for name in ("run_id", "input_root", "competition", "challenges_file", "model_slug"):
            value = getattr(self, name)
            _require(isinstance(value, str) and value != "", f"{name} must be a non-empty string")
        for name in ("model_id", "out_dir"):
            value = getattr(self, name)
            _require(isinstance(value, str) and value != "", f"{name} must be a non-empty string")
        for name in ("challenges", "model_dir"):
            value = getattr(self, name)
            _require(value is None or (isinstance(value, str) and value != ""), f"{name}: null")
        _require(
            SOLUTION_MARKER not in self.challenges_file.lower(),
            f"challenges_file names a {SOLUTION_MARKER}s file; only challenges are read (A0)",
        )
        _require("/" not in self.model_slug, "model_slug must be one folder name")
        try:
            subset_ids(self.subset)
        except ValueError as err:
            raise ValueError(f"subset: {err}") from None
        device = self.device
        _require(
            device == "cpu" or (device.startswith("cuda:") and device[5:].isdigit()),
            "device must be cpu or cuda:<index>",
        )
        _require(self.dtype in DTYPES, f"dtype must be one of {sorted(DTYPES)}")
        _require(self.attn_implementation in ATTENTION, f"attn_implementation in {ATTENTION}")
        _require(_is_int(self.max_tokens) and self.max_tokens >= 1, "max_tokens must be >= 1")
        _require(_is_number(self.max_wall_s) and self.max_wall_s > 0, "max_wall_s must be positive")
        for name, kind in (
            ("scope", Scope),
            ("adapter", AdapterSettings),
            ("decision", Thresholds),
        ):
            _require(isinstance(getattr(self, name), kind), f"{name} must be a {kind.__name__}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParityConfig":
        """Build a config from plain values, such as a parsed YAML file.

        Args:
            data: Every field; ``scope``, ``adapter``, and ``decision`` as mappings.

        Returns:
            The validated config.

        Raises:
            ValueError: If a key is missing or unknown, or a value is invalid.
        """
        names = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING}
        missing, unknown = sorted(required - data.keys()), sorted(data.keys() - names)
        _require(not missing and not unknown, f"missing keys {missing}, unknown keys {unknown}")
        nested = {"scope": Scope, "adapter": AdapterSettings, "decision": Thresholds}
        values = dict(data)
        for key, kind in nested.items():
            section = values[key]
            _require(isinstance(section, Mapping), f"{key} must be a mapping")
            expected = {f.name for f in fields(kind)}
            missing, unknown = sorted(expected - section.keys()), sorted(section.keys() - expected)
            _require(not missing and not unknown, f"{key}: missing {missing}, unknown {unknown}")
            values[key] = kind(**section)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain YAML and JSON values (tuples become lists)."""
        return _plain(asdict(self))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Apply the smoke section: top-level keys replace, the nested sections merge by key."""
    merged = dict(base)
    for key, value in override.items():
        if key not in base:
            raise ConfigError(f"smoke.{key} is not a config key")
        if key in SECTIONS:
            if not isinstance(value, dict) or not set(value) <= set(base[key]):
                raise ConfigError(f"smoke.{key} must map keys of {key}")
            merged[key] = {**base[key], **value}
        else:
            merged[key] = value
    return merged


def load_config(path: Path, smoke: bool, overrides: Mapping[str, str | None]) -> ParityConfig:
    """Read the YAML config, apply the smoke section and command-line overrides, and validate.

    The file's own values and its smoke values are both validated in either mode, so a smoke run
    also checks the GPU run's settings. In smoke mode relative paths are taken from the
    repository root.

    Args:
        path: The YAML file.
        smoke: Whether to apply the ``smoke`` section.
        overrides: Config keys set on the command line; None means not given.

    Returns:
        The resolved config.

    Raises:
        ConfigError: If the file is unreadable, a key is missing or unknown, or a value is
            invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"cannot read {path}: {err}") from err
    if not isinstance(raw, dict) or not isinstance(raw.get("smoke"), dict):
        raise ConfigError(f"{path} must hold a mapping with a smoke section")
    section = raw.pop("smoke")
    smoked = _merge(raw, section)
    if smoke:
        for key in PATH_KEYS:
            value = smoked.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                smoked[key] = str(REPO / value)
    try:
        base = ParityConfig.from_dict(raw)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"{path}: {err}") from err
    try:
        smoke_cfg = ParityConfig.from_dict(smoked)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"{path}, smoke section: {err}") from err
    values = (smoke_cfg if smoke else base).to_dict()
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    try:
        return ParityConfig.from_dict(values)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"command line: {err}") from err


# --- inputs ---------------------------------------------------------------------------------------


def find_files(
    root: Path, names: Sequence[str], max_depth: int = SEARCH_DEPTH
) -> dict[str, list[Path]]:
    """Find files by exact name under ``root``, at most ``max_depth`` levels down.

    The script notebook's ``find_inputs``, so both find the same files: Kaggle's 2026 mount layout
    differs from 2025's and moved within 2026 (``docs/STATE.md``), so inputs are found by name.
    The walk is bounded and skips package trees.

    Args:
        root: Folder to search, ``/kaggle/input`` on Kaggle.
        names: File names to look for.
        max_depth: Deepest level searched; a file directly in ``root`` is at level 1.

    Returns:
        For every name, the ``Path`` of each file found, sorted; empty when there is none.
    """
    found: dict[str, list[Path]] = {name: [] for name in names}
    top = os.path.abspath(root)
    for directory, subdirectories, files in os.walk(top, followlinks=True):
        level = 0 if directory == top else os.path.relpath(directory, top).count(os.sep) + 1
        if level + 1 >= max_depth:
            subdirectories[:] = []
        else:
            subdirectories[:] = sorted(
                d
                for d in subdirectories
                if d not in SKIPPED_DIRS and not d.endswith((".dist-info", ".egg-info"))
            )
        for name in files:
            if name in found:
                found[name].append(Path(directory, name))
    return {name: sorted(paths) for name, paths in found.items()}


def find_model_dir(root: Path, model_slug: str) -> Path:
    """Return the one folder under ``root`` that holds a ``config.json`` below ``model_slug``.

    Args:
        root: Folder to search.
        model_slug: Folder name of the base model, as the notebook's ``model_slug``.

    Returns:
        The model folder.

    Raises:
        FileNotFoundError: If no folder or more than one qualifies.
    """
    paths = find_files(root, ["config.json"])["config.json"]
    folders = [p.parent for p in paths if model_slug in p.parts]
    if len(folders) != 1:
        raise FileNotFoundError(
            f"{len(folders)} model folders named {model_slug} under {root}, expected 1: "
            f"{[str(f) for f in folders]}"
        )
    return folders[0]


def find_challenges(root: Path, name: str, competition: str) -> Path:
    """Return the one challenges file under ``root``, preferring copies in the competition's folder.

    Args:
        root: Folder to search.
        name: The challenges file name.
        competition: The competition slug, the folder its data is mounted under.

    Returns:
        The file.

    Raises:
        FileNotFoundError: If no file or more than one qualifies.
    """
    paths = find_files(root, [name])[name]
    preferred = [p for p in paths if competition in p.parts] or paths
    if len(preferred) != 1:
        raise FileNotFoundError(
            f"{len(preferred)} files {name} under {root}, expected 1: {[str(p) for p in preferred]}"
        )
    return preferred[0]


def read_tasks(path: Path, task_ids: Sequence[str]) -> ArcDataset:
    """Read the challenges of ``task_ids`` into the notebook's ``ArcDataset``.

    Args:
        path: The challenges JSON; a name containing ``solution`` is refused (BRIEF A0).
        task_ids: Tasks to keep, in order.

    Returns:
        The dataset of those tasks: train pairs and test inputs, no replies.

    Raises:
        ValueError: If the file is a solutions file or lacks a task.
    """
    if SOLUTION_MARKER in path.name.lower():
        raise ValueError(f"{path.name} is a {SOLUTION_MARKER}s file; only challenges are read")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [task_id for task_id in task_ids if task_id not in data]
    if missing:
        raise ValueError(f"{path} lacks the tasks {missing}")
    queries = {task_id: data[task_id] for task_id in task_ids}
    return ArcDataset(queries=queries, replies={}, keys=list(task_ids))


@dataclass(frozen=True)
class ParitySequence:
    """One sequence of the comparison, as the notebook builds it.

    Attributes:
        name: The notebook's key before any cut: ``<task>_<i>`` for the decode prompt of test
            input i, ``<task>`` for the training text.
        task_id: The task.
        kind: ``decode`` or ``train``.
        key: The notebook's key after ``cut_to_len`` (``.ex...`` when pairs were dropped).
        input_ids: Tokens, int64 on the CPU, ``[seq]``.
        pairs: Demonstration pairs in the sequence.
        dropped: Pairs ``cut_to_len`` dropped to fit the cap.
        target: Training text only: the positions of its last pair's output-grid tokens and of
            the ``<|im_end|>`` closing them, P3.5's query-output loss positions.
    """

    name: str
    task_id: str
    kind: str
    key: str
    input_ids: Tensor  # [seq]
    pairs: int
    dropped: int
    target: range | None


def build_sequences(
    dataset: ArcDataset, task_ids: Sequence[str], formatter: QwenFormatter, max_tokens: int
) -> list[ParitySequence]:
    """Build each task's decode prompts and training text under the identity view.

    The notebook (``arc_solver.worker``) trains on ``fmt_train`` of the train pairs cut to
    ``max_tokens`` and decodes from ``fmt_train`` of the train pairs plus ``fmt_query`` of one
    test input, cut to ``max_tokens - max_new_tokens``; ``cut_to_len`` drops the first pairs until
    the text fits. Here the same calls run without the augmentation, so each text is the one the
    notebook builds for its identity view, tokenized with ``tokenizer.encode`` as the notebook
    does. The training text's loss positions come from ``render_pair``, as P3.5's
    ``build_sequence`` locates them.

    Args:
        dataset: The tasks, from ``read_tasks``.
        task_ids: Tasks in order.
        formatter: The notebook's formatter around the base's tokenizer.
        max_tokens: The notebook's max_seq_length.

    Returns:
        Per task: its decode prompts in test-input order, then its training text.

    Raises:
        ValueError: If a test input or a last pair alone exceeds its cap (``cut_to_len`` would
            not end), or the training text's tokens differ from its pairs' tokens.
    """
    tokenizer = formatter.tokenizer
    decode_cap = max_tokens - formatter.max_new_tokens()
    sequences = []
    for task_id in task_ids:
        puzzle = dataset.change_keys([task_id])
        n_train = len(puzzle.queries[task_id]["train"])
        prompts = puzzle.split_multi_replies()
        for key in prompts.keys:
            query = formatter.fmt_query(prompts.queries[key]["test"])
            if len(tokenizer.encode(query)) > decode_cap:
                raise ValueError(f"{key}: the test input alone exceeds the decode cap {decode_cap}")
            cut = prompts.change_keys([key]).cut_to_len(
                formatter=formatter, name="input", max_len=decode_cap
            )
            (cut_key,) = cut.keys
            pairs = len(cut.queries[cut_key]["train"])
            sequences.append(
                ParitySequence(
                    name=key,
                    task_id=task_id,
                    kind="decode",
                    key=cut_key,
                    input_ids=torch.tensor(
                        tokenizer.encode(cut.get(cut_key, formatter)["input"]), dtype=torch.long
                    ),
                    pairs=pairs,
                    dropped=n_train - pairs,
                    target=None,
                )
            )
        last = puzzle.queries[task_id]["train"][-1]
        if len(tokenizer.encode(formatter.fmt_train([last]))) > max_tokens:
            raise ValueError(
                f"{task_id}: the last train pair alone exceeds max_tokens {max_tokens}"
            )
        cut = puzzle.cut_to_len(formatter=formatter, name="text", max_len=max_tokens)
        (cut_key,) = cut.keys
        pairs = cut.queries[cut_key]["train"]
        rendered = [render_pair(pair, formatter) for pair in pairs]
        ids = [token for pair in rendered for token in pair.token_ids]
        if list(tokenizer.encode(cut.get(cut_key, formatter)["text"])) != ids:
            raise ValueError(f"{cut_key}: the tokens of the training text differ from its pairs'")
        span = rendered[-1].output_positions
        if span.stop != len(rendered[-1].token_ids) - 1:
            raise ValueError(f"{cut_key}: the last pair's grid is not closed by one token")
        offset = len(ids) - len(rendered[-1].token_ids)
        sequences.append(
            ParitySequence(
                name=task_id,
                task_id=task_id,
                kind="train",
                key=cut_key,
                input_ids=torch.tensor(ids, dtype=torch.long),
                pairs=len(pairs),
                dropped=n_train - len(pairs),
                target=range(offset + span.start, offset + span.stop + 1),
            )
        )
    return sequences


# --- the base and the adapter --------------------------------------------------------------------


def load_base(
    model_dir: Path, device: torch.device, dtype: torch.dtype, requested: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load the frozen base and its tokenizer as P3.5's ``load_base`` does.

    Args:
        model_dir: Hugging Face model folder.
        device: Target device.
        dtype: Base precision.
        requested: Attention kernel; ``auto`` is FlashAttention 2 when ``flash_attn`` imports on a
            bf16 GPU, else SDPA (``resolve_attention``).

    Returns:
        The model (eval mode, gradients off, ``use_cache`` off), the tokenizer, and the attention
        kernel the model runs.

    Raises:
        ValueError: If the base has attention dropout (training mode, which checkpointing needs,
            would compute something other than evaluation mode).
    """
    attn = resolve_attention(requested, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": True}
    if device.type == "cuda":
        kwargs["device_map"] = {"": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, attn_implementation=attn, **kwargs)
    except (ImportError, ValueError):
        if requested != "auto" or attn == "sdpa":
            raise
        attn = "sdpa"  # flash_attn present but unusable with this build: fall back, recorded
        model = AutoModelForCausalLM.from_pretrained(model_dir, attn_implementation=attn, **kwargs)
    # Device only: a dtype cast would round the rotary inv_freq buffer, kept in fp32 (P3.5).
    model = model.to(device).eval().requires_grad_(False)
    model.config.use_cache = False
    if getattr(model.config, "attention_dropout", 0.0):
        raise ValueError("the base has attention dropout; training mode would differ from eval")
    return model, tokenizer, getattr(model.config, "_attn_implementation", attn)


def unwrap_targets(model: nn.Module) -> int:
    """Put each wrapped projection's original ``nn.Linear`` back: the inverse of ``wrap_targets``.

    PEFT cannot adapt a wrapped model (``docs/modules/inject.md``, Limits), so the PEFT path runs
    on the same base after this. The layers are the same objects, so the base is unchanged.

    Args:
        model: A model prepared by ``wrap_targets``, outside any ``inject`` block.

    Returns:
        The number of projections unwrapped.
    """
    wrapped = [(n, m) for n, m in model.named_modules() if isinstance(m, InjectableLinear)]
    for name, module in wrapped:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, module.base_layer)
    return len(wrapped)


def make_adapter(
    model: nn.Module, keys: Sequence[str], scope: Scope, settings: AdapterSettings
) -> LoRAFactors:
    r"""Draw the fixed random adapter, float32 on the CPU, in PEFT layout.

    $A$ $[r, d_\text{in}]$ is drawn as PEFT draws a new ``lora_A`` (``kaiming_uniform_`` with
    $a = \sqrt 5$, H's $A_0$ too); $\tilde B$ $[d_\text{out}, r]$ is standard normal; then
    $B = \tilde B \, \rho \|W\|_F / (s \|\tilde B A\|_F)$, so the delta has Frobenius norm
    $\rho \|W\|_F$, with $\|\tilde B A\|_F^2 = \sum_{ij} (A A^\top)_{ij} (\tilde B^\top
    \tilde B)_{ij}$ from two $r \times r$ Gram matrices. One generator, seeded, draws every factor
    in ``keys`` order.

    Args:
        model: The base, wrapped or not; only the target weights' norms are read.
        keys: Factor keys ``layers.{i}.self_attn.{m}``, as ``wrap_targets`` returns them.
        scope: Rank, alpha, and targets.
        settings: Seed and $\rho$.

    Returns:
        ``(A [r, d_in], B [d_out, r])`` per key.
    """
    scaling = rslora_scaling(scope.rank, scope.alpha)
    generator = torch.Generator().manual_seed(settings.seed)
    factors: LoRAFactors = {}
    for key in keys:
        module = model.get_submodule(f"model.{key}")
        weight = getattr(module, "base_layer", module).weight  # [d_out, d_in]
        d_out, d_in = weight.shape
        lora_a = torch.empty(scope.rank, d_in)
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5), generator=generator)
        direction = torch.randn(d_out, scope.rank, generator=generator)
        weight_norm = float(weight.detach().double().norm())
        delta_norm = float(((lora_a @ lora_a.T) * (direction.T @ direction)).sum().sqrt())
        lora_b = direction * (settings.relative_norm * weight_norm / (scaling * delta_norm))
        factors[key] = (lora_a, lora_b)
    return factors


# --- comparisons and the decision ----------------------------------------------------------------


def bf16_ordinal(values: Tensor) -> Tensor:
    """Number bf16 values in order, so that two values' distance in ULPs is their difference.

    The bits read as a signed 16-bit integer grow with the value for positive numbers; for
    negative numbers the magnitude bits are negated. +0 and -0 both map to 0.

    Args:
        values: bf16 values, any shape.

    Returns:
        int32 ordinals of the same shape.

    Raises:
        ValueError: If ``values`` is not bf16.
    """
    if values.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 values, got {values.dtype}")
    bits = values.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def compare(candidate: Tensor, reference: Tensor, bound: float) -> dict[str, Any]:
    """Compare two bf16 logit tensors of one sequence element by element and position by position.

    The ULP distance is counted twice: over every element (a diagnostic, dominated by logits near
    zero, whose ULPs are tiny) and over the elements at or above ``bound`` only, the ones B5's
    bound does not cover and conflict 3's ULP tolerance must.

    Args:
        candidate: Logits under test, ``[positions, vocab]``, bf16.
        reference: Reference logits, same shape, bf16.
        bound: B5's bound on ``|candidate - reference|`` (the decision's ``max_abs_diff``).

    Returns:
        ``bound``; ``max_abs_diff``; ``differing`` (elements not equal); ``elements``;
        ``above_bound`` (elements with ``|candidate - reference| >= bound``); ``max_ulp`` (over
        all elements); ``max_ulp_above_bound`` (over the elements at or above the bound, 0 when
        there are none); ``argmax_mismatch`` (positions whose argmax differs); ``positions``;
        ``finite`` (candidate); ``reference_finite``. The distances and ``above_bound`` are None
        unless both tensors are finite.

    Raises:
        ValueError: If the shapes differ or a tensor is not bf16 ``[positions, vocab]``.
    """
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError(f"shapes {list(candidate.shape)} and {list(reference.shape)}")
    if candidate.dtype != torch.bfloat16 or reference.dtype != torch.bfloat16:
        raise ValueError(f"dtypes {candidate.dtype} and {reference.dtype}, expected bf16")
    finite = bool(torch.isfinite(candidate).all())
    reference_finite = bool(torch.isfinite(reference).all())
    stats: dict[str, Any] = {
        "bound": bound,
        "max_abs_diff": None,
        "differing": int((candidate != reference).sum()),
        "elements": candidate.numel(),
        "above_bound": None,
        "max_ulp": None,
        "max_ulp_above_bound": None,
        "argmax_mismatch": int((candidate.argmax(-1) != reference.argmax(-1)).sum()),
        "positions": candidate.shape[0],
        "finite": finite,
        "reference_finite": reference_finite,
    }
    if finite and reference_finite:
        diff = (candidate.float() - reference.float()).abs()
        ulp = (bf16_ordinal(candidate) - bf16_ordinal(reference)).abs()
        above = diff >= bound
        stats["max_abs_diff"] = float(diff.max()) if diff.numel() else 0.0
        stats["above_bound"] = int(above.sum())
        stats["max_ulp"] = int(ulp.max()) if ulp.numel() else 0
        stats["max_ulp_above_bound"] = int(ulp[above].max()) if bool(above.any()) else 0
    return stats


def classify(stats: Mapping[str, Any], bitwise: bool, max_ulp: int) -> str:
    """Class of one comparison under the brief's rule, element by element.

    An element is within B5's bound when ``|candidate - reference|`` is below ``stats["bound"]``,
    and within conflict 3's tolerance when it is below the bound or at most ``max_ulp`` bf16 ULPs
    off. Logits near zero have tiny ULPs, so the tolerance counts ULPs only where the bound does
    not already hold.

    Args:
        stats: ``compare``'s result.
        bitwise: Exact means no differing element (the zero-B control) instead of every element
            within the bound.
        max_ulp: The ULP distance the human may accept above the bound.

    Returns:
        ``exact``; ``within`` (not exact, but every element is within the tolerance and every
        argmax agrees); or ``beyond`` (anything else, a non-finite value included).
    """
    if not (stats["finite"] and stats["reference_finite"]):
        return BEYOND
    exact = stats["differing"] == 0 if bitwise else stats["above_bound"] == 0
    if exact:
        return EXACT
    if stats["argmax_mismatch"] == 0 and stats["max_ulp_above_bound"] <= max_ulp:
        return WITHIN
    return BEYOND


def effect(adapted: Tensor, base: Tensor) -> dict[str, Any]:
    """How far the adapter moved one sequence's logits from the base's.

    Args:
        adapted: Logits with the adapter, ``[positions, vocab]``.
        base: Logits of the base, same shape.

    Returns:
        ``mean_abs`` and ``max_abs`` of the difference (None unless both are finite),
        ``changed_fraction`` of elements, and ``argmax_changed_fraction`` of positions.
    """
    finite = bool(torch.isfinite(adapted).all() and torch.isfinite(base).all())
    diff = (adapted.float() - base.float()).abs()
    return {
        "mean_abs": float(diff.mean()) if finite and diff.numel() else None,
        "max_abs": float(diff.max()) if finite and diff.numel() else None,
        "changed_fraction": float((adapted != base).float().mean()) if diff.numel() else 0.0,
        "argmax_changed_fraction": (
            float((adapted.argmax(-1) != base.argmax(-1)).float().mean()) if diff.numel() else 0.0
        ),
    }


def gradient_report(grads: Mapping[str, tuple[Tensor | None, Tensor | None]]) -> dict[str, Any]:
    """Whether the gradient reached every factor: finite and non-zero, A and B of every key.

    Args:
        grads: ``(dL/dA, dL/dB)`` per key; None where no gradient arrived.

    Returns:
        ``tensors``, ``finite`` (every tensor present and finite), ``zero_tensors`` (present but
        all zero), ``missing``, ``min_max_abs`` (the smallest per-tensor max ``|g|``), and
        ``flows`` (finite and none zero).
    """
    finite, zero, missing, smallest = True, 0, 0, math.inf
    for pair in grads.values():
        for grad in pair:
            if grad is None:
                missing += 1
                continue
            if not bool(torch.isfinite(grad).all()):
                finite = False
                continue
            top = float(grad.abs().max())
            if top == 0.0:
                zero += 1
            smallest = min(smallest, top)
    finite = finite and missing == 0
    return {
        "tensors": 2 * len(grads),
        "finite": finite,
        "zero_tensors": zero,
        "missing": missing,
        "min_max_abs": smallest if math.isfinite(smallest) else None,
        "flows": finite and zero == 0,
    }


def relative_difference(
    grads: Mapping[str, tuple[Tensor | None, Tensor | None]],
    reference: Mapping[str, tuple[Tensor | None, Tensor | None]],
) -> tuple[float | None, str | None]:
    r"""The largest per-tensor relative difference of two gradient sets.

    For each factor tensor $g$ against its reference $h$: $\max|g - h| / \max|h|$.

    Args:
        grads: ``(dL/dA, dL/dB)`` per key.
        reference: The same keys from the other path.

    Returns:
        The largest value (None when a tensor is missing or non-finite) and the tensor it was
        found at, ``<key>.lora_A`` or ``<key>.lora_B``.
    """
    worst, at = 0.0, None
    for key, pair in grads.items():
        for label, mine, theirs in zip(("lora_A", "lora_B"), pair, reference[key], strict=True):
            if mine is None or theirs is None:
                return None, f"{key}.{label}"
            diff = float((mine.float() - theirs.float()).abs().max())
            scale = float(theirs.float().abs().max())
            if not (math.isfinite(diff) and math.isfinite(scale)):
                return None, f"{key}.{label}"
            value = diff / scale if scale > 0 else (0.0 if diff == 0 else math.inf)
            if value > worst or at is None:
                worst, at = value, f"{key}.{label}"
    return (worst if math.isfinite(worst) else None), at


def gradient_record(tensors: int, **values: Any) -> dict[str, Any]:
    """One training text's gradient record, every key of ``GRADIENT_KEYS`` present.

    Args:
        tensors: Factor tensors the gradient should reach, A and B of every projection.
        **values: Measured values; ``flows`` stays False unless given.

    Returns:
        The record.
    """
    record: dict[str, Any] = dict.fromkeys(GRADIENT_KEYS)
    record.update(tensors=tensors, flows=False, calls_per_projection={})
    record.update(values)
    return record


def checkpointing_info(model: nn.Module) -> dict[str, Any]:
    """Read back what transformers set up for activation checkpointing.

    ``gradient_checkpointing_enable`` gives every module with a ``gradient_checkpointing``
    attribute (the decoder and its layers) the flag and one checkpoint function, a
    ``functools.partial`` of ``torch.utils.checkpoint.checkpoint``.

    Args:
        model: The base.

    Returns:
        ``modules`` (how many have the flag set) and ``use_reentrant`` (the keyword of the first
        one's checkpoint function, None when it cannot be read).
    """
    modules = [m for m in model.modules() if getattr(m, "gradient_checkpointing", False) is True]
    func = getattr(modules[0], "_gradient_checkpointing_func", None) if modules else None
    keywords = getattr(func, "keywords", None) or {}
    return {"modules": len(modules), "use_reentrant": keywords.get("use_reentrant")}


def decide(
    records: Sequence[Mapping[str, Any]],
    thresholds: Thresholds,
    expected: int,
    failure: str | None = None,
) -> dict[str, Any]:
    """The brief's decision row (section 1) from the per-sequence measurements.

    Order: Error (the run did not finish, a sequence lacks a measurement, a reference logit is
    not finite, or the adapter moved some sequence's logits by less than ``min_mean_effect``),
    then Fail (a comparison is ``beyond``, or some factor gradient is missing, zero, or not
    finite, or its backward failed), then Human decides (a comparison is ``within``), else Pass.
    The comparisons are the injected logits against PEFT's (exact: every element below
    ``max_abs_diff``) and, for the zero-B control, the injected logits with $B = 0$ against the
    base's (exact: bitwise); ``classify`` gives their class, element by element.

    Args:
        records: One mapping per sequence with ``name``, ``kind``, ``parity`` and ``zero_b``
            (``compare`` results with the bound ``max_abs_diff``), ``effect`` (``effect`` of
            PEFT's logits), and for training texts ``gradient`` (with ``flows``, and ``error``
            when its backward failed).
        thresholds: The rule's numbers.
        expected: How many sequences the run built.
        failure: Why the run did not finish, or None.

    Returns:
        ``outcome`` and ``reasons`` (one line each).
    """
    if failure is not None:
        return {"outcome": ERROR, "reasons": [f"The run did not finish: {failure}"]}
    incomplete = [
        r["name"]
        for r in records
        if any(r.get(key) is None for key in ("parity", "zero_b", "effect"))
        or (r["kind"] == "train" and r.get("gradient") is None)
    ]
    if len(records) != expected or not records or incomplete:
        return {
            "outcome": ERROR,
            "reasons": [
                f"Measured {len(records)} of {expected} sequences; incomplete: {incomplete}"
            ],
        }
    reasons = []
    invalid = [
        r["name"]
        for r in records
        if not (r["parity"]["reference_finite"] and r["zero_b"]["reference_finite"])
    ]
    if invalid:
        reasons.append(f"Non-finite reference logits (PEFT or the base) on {invalid}")
    floor = thresholds.min_mean_effect
    vacuous = [
        r["name"]
        for r in records
        if r["effect"]["mean_abs"] is None or r["effect"]["mean_abs"] < floor
    ]
    if vacuous:
        reasons.append(
            f"The adapter moved the logits by less than {floor:g} on average on {vacuous}"
        )
    if reasons:
        return {"outcome": ERROR, "reasons": reasons}
    bound, ulps = thresholds.max_abs_diff, thresholds.max_ulp
    classes = {
        (r["name"], check): classify(r[check], check == "zero_b", ulps)
        for r in records
        for check in ("parity", "zero_b")
    }
    beyond = [f"{name} {check}" for (name, check), c in classes.items() if c == BEYOND]
    stalled = [
        r["name"] + (f" ({r['gradient']['error']})" if r["gradient"].get("error") else "")
        for r in records
        if r["kind"] == "train" and not r["gradient"]["flows"]
    ]
    if beyond:
        reasons.append(
            f"An argmax differs, or a logit {bound:g} or more off is more than {ulps} ULPs off, "
            f"on {beyond}"
        )
    if stalled:
        reasons.append(
            f"A factor gradient is missing, zero, or not finite, or its backward failed, on "
            f"{stalled}"
        )
    if reasons:
        return {"outcome": FAIL, "reasons": reasons}
    within = [f"{name} {check}" for (name, check), c in classes.items() if c == WITHIN]
    if within:
        return {
            "outcome": HUMAN,
            "reasons": [
                f"Not every logit below {bound:g} (or, for the zero-B control, bitwise) on "
                f"{within}, but every argmax agrees and every logit {bound:g} or more off is at "
                f"most {ulps} ULPs off (conflict 3)"
            ],
        }
    worst = max(r["parity"]["max_abs_diff"] for r in records)
    return {
        "outcome": PASS,
        "reasons": [
            f"Every sequence below {thresholds.max_abs_diff:g} (largest {worst:.3g}), the zero-B "
            "control bitwise, and gradients reach every A and B"
        ],
    }


# --- the smoke inputs ----------------------------------------------------------------------------


def write_tiny_model(folder: Path, max_positions: int, seed: int = SMOKE_SEED) -> None:
    """Save the tiny random Qwen3 of ``tests/test_inject.py`` and the stand-in tokenizer.

    Hidden size 64, 2 layers, 4 query heads, 2 key-value heads of width 16 (Qwen3-4B's shape
    relations), vocabulary 16, ``initializer_range`` 0.2 (logits of order 1); its positions run
    to ``max_positions`` instead of 64, for the synthetic sequences. The tokenizer is P3.5's
    stand-in with the notebook's 16 ids. Saved as a Hugging Face folder, so the smoke loads it
    the way the GPU run loads the base.

    Args:
        folder: The model folder, created if missing.
        max_positions: ``max_position_embeddings``.
        seed: Seed of the random weights; the global generator is left as it was.
    """
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=max_positions,
        initializer_range=0.2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    standin_tokenizer().save_pretrained(folder)


def write_synthetic_challenges(path: Path, seed: int = SMOKE_SEED) -> None:
    """Write ``SMOKE_TASKS`` as a challenges file of random grids: no competition data.

    Args:
        path: The file, its folder created if missing.
        seed: Seed of the grids.
    """
    rng = np.random.RandomState(seed)

    def grid(shape: tuple[int, int]) -> list[list[int]]:
        return rng.randint(0, 10, size=shape).tolist()

    tasks = {
        task_id: {
            "train": [{"input": grid(shape), "output": grid(out)} for _ in range(n_pairs)],
            "test": [{"input": grid(shape)} for _ in range(n_tests)],
        }
        for task_id, (n_pairs, shape, out, n_tests) in SMOKE_TASKS.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks), encoding="utf-8")


def prepare_smoke_inputs(cfg: ParityConfig) -> None:
    """Lay out the smoke's inputs under ``input_root`` as Kaggle mounts the real ones.

    The tiny model in ``models/<model_slug>/`` and the synthetic tasks in
    ``competitions/<competition>/<challenges_file>``, so the smoke finds them by the same search.

    Args:
        cfg: The smoke config.
    """
    root = Path(cfg.input_root)
    write_tiny_model(root / "models" / cfg.model_slug, cfg.max_tokens)
    write_synthetic_challenges(root / "competitions" / cfg.competition / cfg.challenges_file)


# --- the run ------------------------------------------------------------------------------------


def git_commit(repo: Path) -> tuple[str | None, bool | None]:
    """Return the checkout's HEAD and whether its tree has changes; ``(None, None)`` outside git.

    Args:
        repo: A folder inside the checkout.

    Returns:
        ``(sha, dirty)``.
    """
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


def failure_reason(err: BaseException) -> str:
    """Return one line naming a failure: the exception's type and its last non-empty line.

    Args:
        err: The exception the run ended with.

    Returns:
        ``<type>: <last non-empty line of its message>``, or the type alone.
    """
    lines = [line.strip() for line in str(err).splitlines() if line.strip()]
    return f"{type(err).__name__}: {lines[-1]}" if lines else type(err).__name__


def versions() -> dict[str, str | None]:
    """Return the versions of Python and of the packages the comparison depends on."""
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in (
        "torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "flash_attn",
        "numpy", "pyyaml",
    ):  # fmt: skip
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def precision_flags() -> dict[str, Any]:
    """Return the global settings that choose kernels or roundings; both paths share them."""
    probes = {
        "float32_matmul_precision": lambda: torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": lambda: torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": lambda: torch.backends.cudnn.allow_tf32,
        "bf16_reduced_precision_reduction": (
            lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "deterministic_algorithms": lambda: torch.are_deterministic_algorithms_enabled(),
        "sdpa_flash": lambda: torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient": lambda: torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdpa_math": lambda: torch.backends.cuda.math_sdp_enabled(),
        "sdpa_cudnn": lambda: torch.backends.cuda.cudnn_sdp_enabled(),
    }
    flags: dict[str, Any] = {}
    for name, probe in probes.items():
        try:
            flags[name] = probe()
        except (AttributeError, RuntimeError):
            flags[name] = None
    flags["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return flags


def device_info(device: torch.device) -> dict[str, Any]:
    """Return what the run computed on: GPU name, compute capability, memory, CUDA and cuDNN.

    Args:
        device: The run's device.

    Returns:
        A mapping; the CPU's is short.
    """
    if device.type != "cuda":
        return {"type": "cpu", "name": platform.processor() or platform.machine()}
    props = torch.cuda.get_device_properties(device)
    return {
        "type": "cuda",
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "memory_bytes": props.total_memory,
        "visible_gpus": torch.cuda.device_count(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "bf16": torch.cuda.is_bf16_supported(),
    }


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _to_cpu(grad: Tensor | None) -> Tensor | None:
    return None if grad is None else grad.detach().float().to("cpu")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParityRun:
    """One parity run: the stages in order, the records, ``parity.json``, and ``summary.md``.

    Args:
        cfg: The resolved config.
        smoke: Whether the smoke section is in effect.
        provenance: Run id, command line, commit, versions.
        started: The script's start, ``time.time()``.
    """

    def __init__(
        self, cfg: ParityConfig, smoke: bool, provenance: dict[str, Any], started: float
    ) -> None:
        self.cfg, self.smoke, self.started = cfg, smoke, started
        self.out_dir = Path(cfg.out_dir)
        self.timing = TimingLogger(self.out_dir, worker=TIMING_WORKER)
        self.device = torch.device(cfg.device)
        self.scaling = rslora_scaling(cfg.scope.rank, cfg.scope.alpha)
        self.records: dict[str, dict[str, Any]] = {}
        self.sequences: list[ParitySequence] = []
        self.logits: dict[str, dict[str, Tensor]] = {}
        self.keys: list[str] = []
        self.expected = 0
        self.report: dict[str, Any] = {
            "run_id": cfg.run_id,
            "smoke": smoke,
            "status": "running",
            "exit_code": None,
            "decision": None,
            "thresholds": asdict(cfg.decision),
            "provenance": provenance,
            "device": {},
            "precision": {},
            "inputs": {},
            "model": {},
            "adapter": {},
            "sequences": [],
            "peak_memory_bytes": None,
            "elapsed_s": None,
            "error_log": False,
        }

    # -- bookkeeping ---------------------------------------------------------------------------

    def elapsed(self) -> float:
        """Seconds since the script started."""
        return time.time() - self.started

    def log(self, message: str) -> None:
        """Print one progress line with the elapsed time (the Kaggle session log keeps it)."""
        print(f"[{self.elapsed():7.1f} s] {message}", flush=True)

    def check_budget(self, stage: str) -> None:
        """Stop between passes once the wall budget is spent.

        Args:
            stage: What would run next, for the message.

        Raises:
            WallTimeExceeded: If ``max_wall_s`` has passed.
        """
        if self.elapsed() > self.cfg.max_wall_s:
            raise WallTimeExceeded(
                f"the wall budget of {self.cfg.max_wall_s:g} s ran out before {stage}"
            )

    def sync(self) -> None:
        """Wait for the device, so a timing record covers the GPU work it names."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def save(self) -> None:
        """Write ``parity.json`` atomically, records in sequence order."""
        self.report["sequences"] = list(self.records.values())
        self.report["elapsed_s"] = round(self.elapsed(), 1)
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.report["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
        data = json.dumps(self.report, indent=2, allow_nan=False) + "\n"
        atomic_write_bytes(self.out_dir / RESULTS_FILE, data.encode("utf-8"))

    # -- stages --------------------------------------------------------------------------------

    def setup(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Check the device, find and read the inputs, load the base, and build the sequences.

        Returns:
            The base model and its tokenizer.

        Raises:
            RuntimeError: If CUDA is requested but absent, or the GPU has no bf16.
            FileNotFoundError: If an input is not found exactly once.
        """
        cfg, device = self.cfg, self.device
        if device.type == "cuda":
            if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"{cfg.device} requested, {torch.cuda.device_count()} CUDA devices visible"
                )
            if DTYPES[cfg.dtype] is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"{torch.cuda.get_device_name(device)} has no bf16 (a T4?)")
            torch.cuda.reset_peak_memory_stats(device)
        self.report["device"] = device_info(device)
        self.report["precision"] = precision_flags()
        root = Path(cfg.input_root)
        model_dir = Path(cfg.model_dir) if cfg.model_dir else find_model_dir(root, cfg.model_slug)
        challenges = (
            Path(cfg.challenges)
            if cfg.challenges
            else find_challenges(root, cfg.challenges_file, cfg.competition)
        )
        task_ids = subset_ids(cfg.subset)
        self.report["inputs"] = {
            "model_dir": str(model_dir),
            "model_id": cfg.model_id,
            "challenges": str(challenges),
            "challenges_sha256": _sha256(challenges),
            "subset": cfg.subset,
            "task_ids": list(task_ids),
        }
        self.log(f"model {model_dir}; challenges {challenges}")
        dataset = read_tasks(challenges, task_ids)
        with self.timing.timed(None, "load"):
            began = time.perf_counter()
            model, tokenizer, attn = load_base(
                model_dir, device, DTYPES[cfg.dtype], cfg.attn_implementation
            )
            self.sync()
        config = model.config
        self.report["model"] = {
            "load_seconds": round(time.perf_counter() - began, 2),
            "attn_implementation": attn,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
        }
        self.log(f"base loaded: {config.num_hidden_layers} layers, attention {attn}")
        # Checkpointing acts in training mode only (transformers), so the logit passes, in
        # evaluation mode, run without it; the gradient passes run with it, as P3.5 trains.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.report["model"]["checkpointing"] = checkpointing_info(model)
        self.sequences = build_sequences(
            dataset, task_ids, QwenFormatter(tokenizer), cfg.max_tokens
        )
        self.expected = len(self.sequences)
        for seq in self.sequences:
            self.records[seq.name] = {
                "name": seq.name,
                "task_id": seq.task_id,
                "kind": seq.kind,
                "key": seq.key,
                "tokens": int(seq.input_ids.numel()),
                "pairs": seq.pairs,
                "dropped_pairs": seq.dropped,
                "target_tokens": None if seq.target is None else len(seq.target),
                "zero_b": None,
                "parity": None,
                "effect": None,
                "effect_injected": None,
                "repeat": {},
                "swaps": {},
                "gradient": None,
            }
        lengths = [int(s.input_ids.numel()) for s in self.sequences]
        self.log(f"{len(lengths)} sequences, {min(lengths)} to {max(lengths)} tokens")
        return model, tokenizer

    def forward_pass(self, model: nn.Module, label: str) -> dict[str, Tensor]:
        """Run every sequence alone through ``model`` in evaluation mode; keep the logits.

        Args:
            model: The base, wrapped or adapted as the stage set it up.
            label: Pass name; the timing op is ``forward_<label>``.

        Returns:
            bf16 logits ``[positions, vocab]`` on the CPU per sequence name.
        """
        model.eval()
        out = {}
        for seq in self.sequences:
            self.check_budget(f"the {label} pass on {seq.name}")
            ids = seq.input_ids.to(self.device)
            with self.timing.timed(seq.task_id, f"forward_{label}", int(ids.numel())):
                with torch.no_grad():
                    logits = model(input_ids=ids[None], use_cache=False).logits[0]
                out[seq.name] = logits.detach().to("cpu")
        self.logits[label] = out
        self.log(f"{label} pass done")
        return out

    def gradient_pass(
        self, model: nn.Module, label: str, factors: LoRAFactors | None, seq: ParitySequence
    ) -> tuple[float, dict[str, tuple[Tensor | None, Tensor | None]], float]:
        """Backward of the query-output CE of one training text, with checkpointing.

        Forward, loss, and backward run inside one ``inject`` block, so the recompute sees the
        same factors (research note, risk 2), as P3.5's ``micro_step`` runs them. A forward
        pre-hook on every target projection's base layer counts its calls: 2 per projection
        when checkpointing recomputed each layer during the backward, 1 without it.

        Args:
            model: The wrapped base (``factors`` given) or the PEFT model (``factors`` None,
                its ``lora_A``/``lora_B`` weights trainable).
            label: Timing op suffix.
            factors: Factors on the device for the injected path; fresh leaves get the grads.
            seq: A training text.

        Returns:
            The loss, ``(dL/dA, dL/dB)`` per key on the CPU in float32, and the forward calls per
            target projection.

        Raises:
            ValueError: If ``seq`` has no loss positions (a decode prompt).
        """
        if seq.target is None:
            raise ValueError(f"{seq.name} is a decode prompt: it has no query-output positions")
        ids = seq.input_ids.to(self.device)
        leaves = None
        if factors is not None:
            leaves = {
                k: (
                    a.detach().clone().requires_grad_(True),
                    b.detach().clone().requires_grad_(True),
                )
                for k, (a, b) in factors.items()
            }
        block = nullcontext() if leaves is None else inject(model, leaves, self.scaling)
        prefix = DECODER if leaves is not None else PEFT_PREFIX
        layers = [model.get_submodule(f"{prefix}{key}") for key in self.keys]
        calls = [0]

        def count(module: nn.Module, args: tuple[Any, ...]) -> None:
            calls[0] += 1

        hooks = [layer.base_layer.register_forward_pre_hook(count) for layer in layers]
        model.train()  # transformers checkpoints in training mode only; Qwen3's dropout is 0
        try:
            with self.timing.timed(seq.task_id, f"grad_{label}", int(ids.numel())):
                with block:
                    logits = model(input_ids=ids[None], use_cache=False).logits[0]
                    loss = query_ce(logits, ids, seq.target)
                    loss.backward()
                self.sync()
            if leaves is not None:
                pairs = [leaves[key] for key in self.keys]
            else:
                pairs = [
                    (x.lora_A[ADAPTER_NAME].weight, x.lora_B[ADAPTER_NAME].weight) for x in layers
                ]
            grads = {
                key: (_to_cpu(a.grad), _to_cpu(b.grad))
                for key, (a, b) in zip(self.keys, pairs, strict=True)
            }
            value = float(loss.detach())
        finally:
            for hook in hooks:
                hook.remove()
            model.eval()
            if leaves is None:
                model.zero_grad(set_to_none=True)  # also after a failure, partial grads included
        return value, grads, calls[0] / len(layers)

    def try_gradient(
        self, model: nn.Module, label: str, factors: LoRAFactors | None, seq: ParitySequence
    ) -> tuple[
        tuple[float, dict[str, tuple[Tensor | None, Tensor | None]], float] | None, str | None
    ]:
        """``gradient_pass``, with a failure returned instead of raised.

        A backward that raises (an OOM in a long text, a loss that reaches no factor) is a
        gradient that does not flow: it is recorded on its text with the reason, its traceback
        goes to ``error.log``, and the run goes on, so the logit comparisons are not lost.

        Args:
            model: As ``gradient_pass``.
            label: As ``gradient_pass``.
            factors: As ``gradient_pass``.
            seq: As ``gradient_pass``.

        Returns:
            ``gradient_pass``'s result and None, or None and the failure's one-line reason.
        """
        try:
            return self.gradient_pass(model, label, factors, seq), None
        except Exception as err:
            reason = failure_reason(err)
            with (self.out_dir / ERROR_FILE).open("a", encoding="utf-8") as handle:
                handle.write(f"{label} gradient on {seq.name}:\n{traceback.format_exc()}\n")
            self.report["error_log"] = True
        # The traceback, and the tensors its frames hold, are gone once the handler has ended.
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.log(f"{label} gradient on {seq.name} failed: {reason}")
        return None, reason

    def compare_all(
        self, field: str, candidate: str, reference: str, bitwise: bool, nested: str = ""
    ) -> None:
        """Compare two passes on every sequence and store the result in each record.

        Args:
            field: Record field to fill.
            candidate: Pass under test.
            reference: Reference pass.
            bitwise: ``classify``'s criterion for exact: bitwise, or every element below the
                decision's bound.
            nested: Sub-field of ``field`` (for ``repeat`` and ``swaps``), or ``""``.
        """
        decision = self.cfg.decision
        for seq in self.sequences:
            stats = compare(
                self.logits[candidate][seq.name],
                self.logits[reference][seq.name],
                decision.max_abs_diff,
            )
            stats["class"] = classify(stats, bitwise, decision.max_ulp)
            record = self.records[seq.name]
            if nested:
                record[field][nested] = stats
            else:
                record[field] = stats

    def measure(self) -> None:
        """Run every stage; the records fill as the passes complete."""
        cfg = self.cfg
        if self.smoke:
            prepare_smoke_inputs(cfg)
            self.log(f"smoke inputs written under {cfg.input_root}")
        model, _ = self.setup()
        self.save()
        self.check_budget("the base pass")
        base = self.forward_pass(model, "base")
        self.keys = wrap_targets(model, cfg.scope.target_modules)
        expected_keys = model.config.num_hidden_layers * len(cfg.scope.target_modules)
        if len(self.keys) != expected_keys:
            raise RuntimeError(f"{len(self.keys)} projections wrapped, expected {expected_keys}")
        factors = make_adapter(model, self.keys, cfg.scope, cfg.adapter)
        on_device = {k: (a.to(self.device), b.to(self.device)) for k, (a, b) in factors.items()}
        zero_b = {k: (a, torch.zeros_like(b)) for k, (a, b) in on_device.items()}
        emitted = sum(a.numel() + b.numel() for a, b in factors.values())
        b_std = [float(b.std()) for _, b in factors.values()]
        self.report["adapter"] = {
            "rank": cfg.scope.rank,
            "alpha": cfg.scope.alpha,
            "scaling": self.scaling,
            "target_modules": list(cfg.scope.target_modules),
            "projections": len(self.keys),
            "values": emitted,
            "dtype": "float32",
            "seed": cfg.adapter.seed,
            "relative_norm": cfg.adapter.relative_norm,
            "a_init": "kaiming_uniform_(a=sqrt(5)), as PEFT draws lora_A",
            "b_std_min": min(b_std),
            "b_std_max": max(b_std),
        }
        self.log(f"adapter: {len(self.keys)} projections, {emitted} values, s = {self.scaling:g}")
        with inject(model, zero_b, self.scaling):
            self.forward_pass(model, "zero_b")
        self.compare_all("zero_b", "zero_b", "base", True)
        with inject(model, on_device, self.scaling):
            injected = self.forward_pass(model, "inject")
        for seq in self.sequences:
            self.records[seq.name]["effect_injected"] = effect(injected[seq.name], base[seq.name])
        self.save()

        inject_grads = {}
        tensors = 2 * len(self.keys)
        for seq in (s for s in self.sequences if s.kind == "train"):
            self.check_budget(f"the injected gradient on {seq.name}")
            first, error = self.try_gradient(model, "inject", on_device, seq)
            if first is None:
                self.records[seq.name]["gradient"] = gradient_record(tensors, error=error)
                continue
            loss, grads, calls = first
            inject_grads[seq.name] = grads
            record = gradient_record(
                tensors,
                loss_injected=_finite(loss),
                **{k: v for k, v in gradient_report(grads).items() if k != "tensors"},
            )
            record["calls_per_projection"]["inject"] = calls
            repeat, error = self.try_gradient(model, "inject_repeat", on_device, seq)
            if repeat is None:
                record["repeat_error"] = error
            else:
                loss_repeat, repeat_grads, calls = repeat
                rel, rel_at = relative_difference(repeat_grads, grads)
                record.update(
                    loss_injected_repeat=_finite(loss_repeat),
                    rel_diff_repeat=_finite(rel),
                    rel_diff_repeat_at=rel_at,
                )
                record["calls_per_projection"]["inject_repeat"] = calls
            self.records[seq.name]["gradient"] = record
        self.log("injected gradients done")
        self.save()

        unwrapped = unwrap_targets(model)
        if unwrapped != len(self.keys):
            raise RuntimeError(f"{unwrapped} projections unwrapped, expected {len(self.keys)}")
        self.forward_pass(model, "base_repeat")
        self.compare_all("repeat", "base_repeat", "base", True, nested="base")

        with tempfile.TemporaryDirectory(prefix="arc26-parity-") as tmp:
            adapter_dir, previous_dir = Path(tmp, "adapter"), Path(tmp, "previous")
            targets = list(cfg.scope.target_modules)
            save_adapter(factors, adapter_dir, cfg.scope.rank, cfg.scope.alpha, targets)
            previous = {k: (a / 2, b / 2) for k, (a, b) in factors.items()}
            save_adapter(previous, previous_dir, cfg.scope.rank, cfg.scope.alpha, targets)
            self.report["adapter"]["sha256"] = _sha256(adapter_dir / ADAPTER_WEIGHTS)
            self.check_budget("the PEFT load")
            with self.timing.timed(None, "peft_load"):
                peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
                self.sync()
            layer = peft_model.get_submodule(f"{PEFT_PREFIX}{self.keys[0]}")
            loaded = layer.lora_A[ADAPTER_NAME].weight
            self.report["adapter"]["peft_dtype"] = str(loaded.dtype).removeprefix("torch.")
            self.report["adapter"]["peft_scaling"] = layer.scaling[ADAPTER_NAME]
            peft = self.forward_pass(peft_model, "peft")
            self.compare_all("parity", "inject", "peft", False)
            for seq in self.sequences:
                self.records[seq.name]["effect"] = effect(peft[seq.name], base[seq.name])
            self.forward_pass(peft_model, "peft_repeat")
            self.compare_all("repeat", "peft_repeat", "peft", True, nested="peft")
            self.save()

            lora = [p for n, p in peft_model.named_parameters() if ".lora_" in n]
            for param in lora:
                param.requires_grad_(True)
            for seq in (s for s in self.sequences if s.kind == "train"):
                self.check_budget(f"the PEFT gradient on {seq.name}")
                result, error = self.try_gradient(peft_model, "peft", None, seq)
                record = self.records[seq.name]["gradient"]
                if result is None:
                    record["peft_error"] = error
                    continue
                loss, grads, calls = result
                rel, rel_at = None, None
                if seq.name in inject_grads:
                    rel, rel_at = relative_difference(inject_grads[seq.name], grads)
                injected = record["loss_injected"]
                record.update(
                    loss_peft=_finite(loss),
                    loss_equal=None if injected is None else loss == injected,
                    rel_diff_peft=_finite(rel),
                    rel_diff_peft_at=rel_at,
                )
                record["calls_per_projection"]["peft"] = calls
            for param in lora:
                param.requires_grad_(False)
            self.log("PEFT gradients done")
            self.save()

            # P4.1's swaps: the previous puzzle's adapter, then this one, into the same name.
            self.check_budget("the swaps")
            peft_model.load_adapter(str(previous_dir), ADAPTER_NAME)
            peft_model.load_adapter(str(adapter_dir), ADAPTER_NAME)
            self.forward_pass(peft_model, "swap_load_adapter")
            set_peft_model_state_dict(peft_model, to_peft_state_dict(previous))
            loaded_back = set_peft_model_state_dict(peft_model, to_peft_state_dict(factors))
            missing = [k for k in loaded_back.missing_keys if ".lora_" in k]
            if loaded_back.unexpected_keys or missing:
                raise RuntimeError(
                    f"adapter keys: unexpected {loaded_back.unexpected_keys[:3]}, missing "
                    f"{missing[:3]}"
                )
            self.forward_pass(peft_model, "swap_state_dict")
            for label, pass_name in (
                ("load_adapter", "swap_load_adapter"),
                ("set_peft_model_state_dict", "swap_state_dict"),
            ):
                self.compare_all("swaps", pass_name, "inject", False, label)
        self.save()

    def execute(self) -> int:
        """Measure, decide, and write the outputs, however the run ends.

        Returns:
            The exit code.
        """
        code, failure = EXIT_OK, None
        self.save()
        try:
            self.measure()
        except WallTimeExceeded as err:
            code, failure = EXIT_STOPPED, str(err)
        except KeyboardInterrupt:
            code, failure = EXIT_FAILED, "interrupted"
        except Exception as err:
            code, failure = EXIT_FAILED, failure_reason(err)
            with (self.out_dir / ERROR_FILE).open("a", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
            self.report["error_log"] = True
        self.report["status"] = STATUS[code]
        self.report["exit_code"] = code
        self.report["decision"] = decide(
            list(self.records.values()), self.cfg.decision, self.expected, failure
        )
        self.save()
        summary = render_summary(self.report)
        (self.out_dir / SUMMARY_FILE).write_text(summary, encoding="utf-8")
        print(summary, flush=True)
        return code


# --- the summary --------------------------------------------------------------------------------


def _num(value: Any, spec: str = ".3g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _cmp(stats: Mapping[str, Any] | None) -> str:
    if stats is None:
        return "n/a"
    if stats["differing"] == 0:
        return "bitwise"
    return f"{_num(stats['max_abs_diff'])}, {stats['differing']} differ, {stats['class']}"


READING = {
    PASS: (
        "P3.1 is accepted on the GPU tier: training through inject and decoding through PEFT see "
        "the same model. docs/modules/inject.md records the result; P3.5 and P4.1 rely on it."
    ),
    HUMAN: (
        "Conflict 3: not every logit is below the bound, but every argmax agrees and every logit "
        "at or above it is at most the allowed ULPs off. The human accepts or rejects at review; "
        "rejected, it is a Fail."
    ),
    FAIL: (
        "The seam does not hold. Find the differing op (the per-sequence table, the repeat and "
        "swap diagnostics: a differing repeat means the stack itself is not deterministic), fix "
        "inject.py, and write a new brief."
    ),
    ERROR: (
        "No conclusion: pull the logs (error.log, the session log), fix the cause, and rerun "
        "under an amended, re-approved brief."
    ),
}


def render_summary(report: Mapping[str, Any]) -> str:
    """Write the short human-readable report of a run.

    Args:
        report: ``parity.json``'s content.

    Returns:
        Markdown text.
    """
    decision = report.get("decision") or {"outcome": ERROR, "reasons": ["no decision"]}
    prov = report.get("provenance") or {}
    vers = prov.get("versions") or {}
    dev = report.get("device") or {}
    model = report.get("model") or {}
    inputs = report.get("inputs") or {}
    adapter = report.get("adapter") or {}
    prec = report.get("precision") or {}
    thresholds = report.get("thresholds") or {}
    records = report.get("sequences") or []
    peak = report.get("peak_memory_bytes")
    ckpt = model.get("checkpointing") or {}
    lines = [
        f"# {report.get('run_id')}: injected LoRA against PEFT, bf16, DEV4",
        "",
        f"- Outcome: **{decision['outcome']}**. " + "; ".join(decision["reasons"]) + ".",
        f"- Rule: docs/prerun/P3.1-01.md, section 1: bound {thresholds.get('max_abs_diff')}, "
        f"at most {thresholds.get('max_ulp')} ULPs for the human's row, adapter effect at least "
        f"{thresholds.get('min_mean_effect')}.",
        f"- Run: status {report.get('status')}, exit {report.get('exit_code')}, wall "
        f"{report.get('elapsed_s')} s, peak GPU memory "
        f"{'n/a' if peak is None else f'{peak / 2**30:.1f} GiB'}; commit {prov.get('commit')} "
        f"(dirty: {prov.get('dirty')}); smoke: {report.get('smoke')}; torch "
        f"{vers.get('torch')}, transformers {vers.get('transformers')}, peft {vers.get('peft')}, "
        f"flash_attn {vers.get('flash_attn')}.",
    ]
    if dev:
        lines.append(
            f"- Device: {dev.get('name')} (compute capability {dev.get('capability', 'n/a')}, "
            f"CUDA {dev.get('cuda', 'n/a')}); TF32 matmul {prec.get('cuda_matmul_allow_tf32')}, "
            f"bf16 reduced-precision reduction {prec.get('bf16_reduced_precision_reduction')}, "
            f"deterministic algorithms {prec.get('deterministic_algorithms')}."
        )
    if inputs:
        lines.append(
            f"- Inputs: {inputs.get('subset')} {inputs.get('task_ids')} from "
            f"{inputs.get('challenges')} (sha256 {inputs.get('challenges_sha256')}); base "
            f"{inputs.get('model_dir')} ({inputs.get('model_id')})."
        )
    if model:
        lines.append(
            f"- Base: {model.get('dtype')}, {model.get('num_hidden_layers')} layers, attention "
            f"{model.get('attn_implementation')}, loaded in {model.get('load_seconds')} s; "
            f"checkpointing on {ckpt.get('modules')} modules, use_reentrant "
            f"{ckpt.get('use_reentrant')}."
        )
    if adapter:
        lines.append(
            f"- Adapter: rank {adapter.get('rank')}, alpha {adapter.get('alpha')}, s = "
            f"{adapter.get('scaling')}, {adapter.get('target_modules')} on "
            f"{adapter.get('projections')} projections, {adapter.get('values')} values in "
            f"{adapter.get('dtype')} (PEFT holds them in {adapter.get('peft_dtype', 'n/a')}); A "
            f"as PEFT draws lora_A, B scaled to ||sBA||_F = {adapter.get('relative_norm')} "
            f"||W||_F; seed {adapter.get('seed')}; sha256 {adapter.get('sha256', 'n/a')}."
        )
    if vers.get("peft") not in (None, PEFT_REPLAYED):
        lines.append(
            f"- Note: peft {vers.get('peft')} is not the {PEFT_REPLAYED} that inject.py replays."
        )
    if records:
        lines += [
            "",
            "## Logits, injected against PEFT `from_pretrained`",
            "",
            "| Sequence | Kind | Tokens | Pairs (dropped) | Max abs diff | Differing | "
            "At or above bound | Max ULP there (all) | Argmax differs | Zero-B vs base | "
            "Effect mean (max) | Class |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in records:
            par, eff = r.get("parity") or {}, r.get("effect") or {}
            lines.append(
                f"| {r['name']} | {r['kind']} | {r['tokens']} | {r['pairs']} "
                f"({r['dropped_pairs']}) | {_num(par.get('max_abs_diff'))} | "
                f"{_num(par.get('differing'))} | {_num(par.get('above_bound'))} | "
                f"{_num(par.get('max_ulp_above_bound'))} ({_num(par.get('max_ulp'))}) | "
                f"{_num(par.get('argmax_mismatch'))} | {_cmp(r.get('zero_b'))} | "
                f"{_num(eff.get('mean_abs'))} ({_num(eff.get('max_abs'))}) | "
                f"{_num(par.get('class'))} |"
            )
        lines += [
            "",
            "## Gradients: query-output CE through the injected path, non-reentrant checkpointing",
            "",
            "| Sequence | Target tokens | CE injected | CE PEFT | Flows | Zero tensors | "
            "Smallest max abs grad | Rel. diff to PEFT | Rel. diff, repeat | Calls per "
            "projection (inject, repeat, PEFT) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        failures = []
        for r in (r for r in records if r["kind"] == "train"):
            g = r.get("gradient") or {}
            calls = g.get("calls_per_projection") or {}
            lines.append(
                f"| {r['name']} | {r['target_tokens']} | {_num(g.get('loss_injected'), '.6f')} | "
                f"{_num(g.get('loss_peft'), '.6f')} | {_num(g.get('flows'))} | "
                f"{_num(g.get('zero_tensors'))} | {_num(g.get('min_max_abs'))} | "
                f"{_num(g.get('rel_diff_peft'))} | {_num(g.get('rel_diff_repeat'))} | "
                + ", ".join(_num(calls.get(k), "g") for k in ("inject", "inject_repeat", "peft"))
                + " |"
            )
            for key, what in (
                ("error", "injected backward"),
                ("repeat_error", "repeated injected backward"),
                ("peft_error", "PEFT backward"),
            ):
                if g.get(key):
                    failures.append(f"- {r['name']}: the {what} failed: {g[key]}")
        lines += ["", *failures] if failures else []
        lines += [
            "",
            "Calls per projection: 2 means checkpointing recomputed every layer during the "
            "backward (the forward, then the recompute); 1 would mean no checkpointing.",
        ]
        lines += [
            "",
            "## Diagnostics: determinism and P4.1's swaps",
            "",
            "| Sequence | Base repeat | PEFT repeat | Swap `load_adapter` | Swap "
            "`set_peft_model_state_dict` |",
            "|---|---|---|---|---|",
        ]
        for r in records:
            repeat, swaps = r.get("repeat") or {}, r.get("swaps") or {}
            lines.append(
                f"| {r['name']} | {_cmp(repeat.get('base'))} | {_cmp(repeat.get('peft'))} | "
                f"{_cmp(swaps.get('load_adapter'))} | "
                f"{_cmp(swaps.get('set_peft_model_state_dict'))} |"
            )
    else:
        lines += ["", "No sequence was measured."]
    reading = READING[decision["outcome"]]
    if report.get("smoke"):
        reading = (
            "Smoke run: the tiny random Qwen3 and synthetic DEV4-shaped tasks on the CPU check the "
            "code path and the decision logic; the numbers say nothing about GPU kernels, and "
            f"only the GPU run decides P3.1. On a GPU run this row would read: {reading}"
        )
    lines += [
        "",
        "## Reading",
        "",
        reading,
        "Swaps are compared with the injected logits; repeats with the first pass. Logs: "
        f"{RESULTS_FILE}, timing-{TIMING_WORKER}.jsonl, config.yaml"
        + (f", {ERROR_FILE}" if report.get("error_log") else "")
        + ".",
        "",
    ]
    return "\n".join(lines)


# --- the command line ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; no option other than ``--config`` names a config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, type=Path, help="configs/inject_parity.yaml")
    parser.add_argument(
        "--smoke", action="store_true", help="CPU run: tiny random Qwen3, synthetic DEV4 tasks"
    )
    parser.add_argument("--commit", help="code commit to record when not in a git checkout")
    parser.add_argument("--out-dir", help="run-log folder (config: out_dir)")
    parser.add_argument("--input-root", help="folder searched for the inputs (config: input_root)")
    parser.add_argument("--model-dir", help="model folder, skips the search (config: model_dir)")
    parser.add_argument("--challenges", help="challenges file, skips the search (config)")
    parser.add_argument("--device", help="cpu or cuda:<index> (config: device)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        The exit code: 0 measured and decided, 1 failed, 2 invalid config or command line,
        3 stopped at the wall budget.
    """
    started = time.time()
    args = build_parser().parse_args(argv)
    if args.smoke and (args.model_dir or args.challenges):
        print(
            "--smoke builds its own model and tasks: drop --model-dir and --challenges",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    overrides = {
        "out_dir": args.out_dir,
        "input_root": args.input_root,
        "model_dir": args.model_dir,
        "challenges": args.challenges,
        "device": args.device,
    }
    try:
        cfg = load_config(args.config, args.smoke, overrides)
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return EXIT_CONFIG
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_file = f"timing-{TIMING_WORKER}.jsonl"
    for stale in (ERROR_FILE, RESULTS_FILE, SUMMARY_FILE, timing_file):
        (out_dir / stale).unlink(missing_ok=True)
    resolved = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    atomic_write_bytes(out_dir / "config.yaml", resolved.encode("utf-8"))
    commit, dirty = (args.commit, None) if args.commit else git_commit(REPO)
    provenance = {
        "argv": list(sys.argv[1:] if argv is None else argv),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "versions": versions(),
        "platform": platform.platform(),
        # Which copy of the package ran: on Kaggle the wheel the notebook installed.
        "arc26": arc26.__file__,
    }
    return ParityRun(cfg, args.smoke, provenance, started).execute()


if __name__ == "__main__":
    sys.exit(main())
