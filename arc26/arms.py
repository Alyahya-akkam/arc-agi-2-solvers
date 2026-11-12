r"""Paired adapter arms: one solver, several adapter sources per puzzle (task P3.7, ADR-019).

Gate G4 (BRIEF B7, B5 P3.7) reads how much of the gap between the zero adapter and the SGD adapter
H closes. Every arm runs P2.6-01's build path (the split solver, ADR-011; rank 16 on q, k, v, o,
ADR-002; no hold-out, no loop, ADR-018; ``arc26.decode.decode_puzzle``'s views, rescoring, and the
reference's ``score_kgmon`` selection); only the adapter differs. An arm's source is one of:

- ``base``: the adapter's initial weights (``lora_B`` zero), so the model is the base (the zero
  adapter equals it, P3.5);
- ``hyper``: H's adapter, $H(\bar z(D))$, through ``arc26.hyper.providers.HyperProvider``;
- ``sgd``: ``arc26.ttt.fit``, one epoch from the initial weights, P2.6-01's training;
- ``hyper_polish``: H's adapter polished for $s$ steps (``HyperPolishProvider``, ADR-017);
- ``polish``: the initial weights polished for $s$ steps (``SGDProvider``), which separates H's
  share of ``hyper_polish`` from the short polish's;
- ``hyper_fit``: ``sgd``'s training with H's adapter as its start (task brief amendment 8, run
  ``P3.7-03``): H's adapter is swapped in, copied, and passed to ``arc26.ttt.fit`` as its
  ``default_weights``; every other argument is ``sgd``'s, so the data (the 128 views), the steps
  (one epoch), the learning rate, the schedule, and a fresh AdamW are the same and only the
  starting adapter differs.
- ``hyper_merge_fit``: H's adapter merged into the base, then ``sgd``'s training of a fresh LoRA
  (task P3.9b amendment 1, run ``P3.9-02``; HyperDreamBooth's recipe): after an exact copy of
  every targeted base weight, each LoRA layer's $W \leftarrow \mathrm{bf16}(W + s\,B_H A_H)$,
  the delta computed in float32 (``BaseMerge``); then ``arc26.ttt.fit`` from the initial weights
  ($B = 0$) exactly as ``sgd``, so the fit starts from H's function with the standard factors.
  The copy is put back bit for bit after the arm's decode, also when the arm raises, before the
  next arm; a copy that could not be put back stops the run (exit 7).

With ``train_log`` set, every arm that trains (``sgd``, ``hyper_fit``, the polishes) records its
per-step training loss, learning rate, and gradient norm in its record (``train_log``): the
trainer logs every step (``logging_strategy`` ``steps``, ``logging_steps`` 1, which only reads
the loss), and the log calls are captured around the fit (``train_log_capture``).

**Paired by puzzle.** Each worker takes one puzzle at a time from a queue in a fixed shuffled order
(``queue_order``: sha256 of the seed and the task id) and runs every arm on it before the next, in
the configured order, the adapter of the live PEFT model replaced per arm. No puzzle starts after
the soft deadline; the arms of a puzzle already started may run ``finish_s`` seconds longer, and
an arm still decoding then is cut, which leaves its puzzle incomplete in every arm's statistics.
So a cut leaves the same shuffled puzzles in every arm. Each arm writes its candidates to its own
folder, its timing records (``decode``, ``selection``, ``train``, ``encode``, ``h_forward``, ...,
and the arm's span as ``puzzle``) to its own ``timing/`` folder, and one record per puzzle with
its fingerprint (``adapter_digest``), the teacher-forced CE of every answer (one forward,
``answer_ce``), and its status. After the workers, the parent selects every arm's two attempts
with the reference's selection and scores them against the solutions (development mode only).

Before the workers, the parent builds GEN when asked (``arc26.hyper.validation.gen_from_episodes``)
and runs H's load check in a child process on GPU 0 (``python -m arc26.hyper.validation check``);
a failed check stops the run (exit 5) before any decode. The load check runs transformers' base,
so the H arms' path on the solver's unsloth-patched model (the live encode under
``disable_adapter``, the swap) is first run by the workers: each worker probes it once after its
model load (``ArmRunner.probe``: H's adapter for the smallest DEV4 puzzle, then the rollback, a
few seconds), and a probe that raises, or an H arm that raises on the first puzzle a worker
finishes, stops the run (exit 7) with its records and summary written, instead of marking every
H arm ``raised`` for hours.

``python -m arc26.arms --end-time T --config JSON`` takes ``arc26.runner.solver_command``'s
arguments; ``arc26.runner`` starts it in place of ``arc26.solver`` when ``RunConfig.adapters`` is
set. Importing this module imports no torch: the notebook kernel imports it through the runner.
See ``docs/modules/arms.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

from arc26 import timing
from arc26.data import seeded_key
from arc26.loop import EPOCH_VIEWS

if TYPE_CHECKING:
    from arc26.solver import Loaded

BASE, HYPER, SGD, HYPER_POLISH, POLISH = "base", "hyper", "sgd", "hyper_polish", "polish"
HYPER_FIT, HYPER_MERGE_FIT = "hyper_fit", "hyper_merge_fit"
SOURCES = (BASE, HYPER, SGD, HYPER_POLISH, POLISH, HYPER_FIT, HYPER_MERGE_FIT)
H_SOURCES = frozenset({HYPER, HYPER_POLISH, HYPER_FIT, HYPER_MERGE_FIT})
# The sources whose adapter is trained on the puzzle, so train_log applies.
TRAINED_SOURCES = frozenset({SGD, HYPER_POLISH, POLISH, HYPER_FIT, HYPER_MERGE_FIT})
# What train_log adds to the trainer's arguments: a log entry every optimizer step. Logging reads
# the step's loss (one device sync); it changes no gradient, no update, and no random draw.
TRAIN_LOG_ARGS = {"logging_strategy": "steps", "logging_steps": 1}
POLISH_SOURCES = frozenset({HYPER_POLISH, POLISH})
# An arm's name is a folder name under the candidate root and run_logs/arms/ that the pull
# allowlist accepts: lowercase letters, digits, "-".
ARM_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
# decode_puzzle's views of one test input: its 16 view keys in sorted order are the 8 dihedral maps
# in this order (forward_mod's ops: transpose, then quarter turns), 2 color permutations each, so
# view 2 d + c is map d under permutation c.
DIHEDRAL_ORDER = (
    "identity", "rot90", "rot180", "rot270",
    "transpose", "transpose.rot90", "transpose.rot180", "transpose.rot270",
)  # fmt: skip
VIEW_COUNT = 2 * len(DIHEDRAL_ORDER)
# A batch's prompts must have one length (inference_turbo_dfs stacks them). A map that keeps every
# grid's rows as rows (identity, rot180, transpose.rot90, transpose.rot270) gives one length, a map
# that swaps rows and columns the other, whatever the grids.
LENGTH_GROUPS = (
    frozenset({0, 1, 4, 5, 10, 11, 14, 15}),
    frozenset({2, 3, 6, 7, 8, 9, 12, 13}),
)
# The reference's batches of one input (decode_puzzle with views None, which orders them over all
# test inputs: the first two batches of every input, then the last two).
REFERENCE_VIEWS = ((0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15))
# GEN's 8 views (ADR-019): every dihedral map once, its first color permutation, as two full
# batches of four, one per length group.
DIHEDRAL_ONCE = ((0, 4, 10, 14), (2, 6, 8, 12))
# The reference's batch width; a wider batch has no memory measurement.
MAX_BATCH = 4
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_LOAD_CHECK = 0, 1, 2, 5
# H's path failed on the solver's model: the workers' probe raised, or an H arm raised on the first
# puzzle a worker finished (5 is the load check's stop, 6 its failed verdict).
EXIT_H_ARM = 7
# Timing ops of the arms beside the moved code's and the providers': the reset to the initial
# weights, the copy of cached embeddings into the worker's cache, the adapter digest, and the
# answer's CE.
RESET, CACHE_SEED, FINGERPRINT, CE = "reset", "cache_seed", "fingerprint", "ce"
# hyper_merge_fit's ops: the copy of the base weights and the merge of H's delta, then the copy
# put back after the decode.
MERGE, UNMERGE = "merge", "unmerge"
# The workers' probe of H's path (ArmRunner.probe): its timing op and its records' arm name.
H_PROBE, PROBE = "h_probe", "probe"
RECORDS_PREFIX = "records-"
# A worker that stops the run writes stop-rank<k>.json into the log folder before it raises.
STOP_PREFIX = "stop-"
SUMMARY_FILE = "arms.json"
SCORES_FILE = "scores.json"
SUBMISSION_FILE = "submission.json"
SELECTION_LOG = "selection.log"
LOAD_CHECK_FILE = "load_check.json"
GEN_FILE = "gen.json"
SOLUTIONS_FILE = "arc-agi_evaluation_solutions.json"
CHALLENGES_FILE = "arc-agi_evaluation_challenges.json"
# Arm statuses in the records: decoded to the end; cut by the finish deadline; never started (the
# deadline passed first); raised (the error is recorded, the worker goes on).
DONE, CUT, NOT_STARTED, RAISED = "done", "cut", "not_started", "raised"
# Longest the load check may take before the run gives up on it (P3.5-01: about 76 s per
# validation of two references on four L4s; this is three references on one, plus the load).
LOAD_CHECK_TIMEOUT_S = 1800


class ArmStop(RuntimeError):
    """H's path failed on the solver's model; the worker stops the run (exit 7)."""


def _text(name: str, value: object, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")


def _integer(value: object) -> bool:
    return type(value) is int


def check_views(views: object) -> tuple[tuple[int, ...], ...] | None:
    """Check a view selection for ``decode_puzzle`` and return it as tuples.

    Args:
        views: None (the reference's 16 views and batches), or a non-empty sequence of batches,
            each a non-empty sequence of at most ``MAX_BATCH`` view indices from 0 to 15, all
            distinct over the batches, each batch inside one of ``LENGTH_GROUPS``.

    Returns:
        The selection as a tuple of tuples, or None.

    Raises:
        ValueError: If a rule is broken.
    """
    if views is None:
        return None
    if isinstance(views, str | bytes) or not isinstance(views, Sequence) or not views:
        raise ValueError(f"views must be null or a non-empty list of batches; got {views!r}")
    batches = []
    for batch in views:
        if isinstance(batch, str | bytes) or not isinstance(batch, Sequence) or not batch:
            raise ValueError(f"each batch of views must be a non-empty list; got {batch!r}")
        if len(batch) > MAX_BATCH:
            raise ValueError(f"a batch holds at most {MAX_BATCH} views; got {list(batch)}")
        if not all(_integer(i) and 0 <= i < VIEW_COUNT for i in batch):
            raise ValueError(f"view indices are integers 0 to {VIEW_COUNT - 1}; got {list(batch)}")
        if not any(set(batch) <= group for group in LENGTH_GROUPS):
            groups = [sorted(group) for group in LENGTH_GROUPS]
            raise ValueError(
                f"a batch's views must give one prompt length (one of {groups}); got {list(batch)}"
            )
        batches.append(tuple(batch))
    flat = [i for batch in batches for i in batch]
    if len(set(flat)) != len(flat):
        raise ValueError(f"a view appears twice: {flat}")
    return tuple(batches)


def queue_order(keys: Iterable[str], seed: int) -> list[str]:
    """Return the puzzles in the run's fixed shuffled order.

    Why: a soft deadline cuts the end of the queue; a shuffled order makes the puzzles it leaves a
    random subset, the same in every arm (paired). ``arc26.data.seeded_key`` is a sha256 of the
    seed and the id, so the order does not depend on the platform or Python's hash salt.

    Args:
        keys: Task ids.
        seed: The order's seed.

    Returns:
        The ids sorted by ``seeded_key(id, seed)``.
    """
    return sorted(dict.fromkeys(keys), key=lambda key: seeded_key(key, seed))


@dataclass(frozen=True)
class ArmSpec:
    """One arm: a name and the adapter source it decodes with.

    Attributes:
        name: Folder name of the arm's candidates and logs (``ARM_NAME``).
        source: One of ``SOURCES``.
        steps: The polish's optimizer steps $s$ for ``hyper_polish`` and ``polish`` (1 to 128,
            ``POLISH_STEPS`` = 13 in P3.7's configs); None for the other sources.

    Raises:
        ValueError: If the name, the source, or the steps break their rule.
    """

    name: str
    source: str
    steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not ARM_NAME.fullmatch(self.name):
            raise ValueError(f"arm name must match {ARM_NAME.pattern}; got {self.name!r}")
        if self.source not in SOURCES:
            raise ValueError(
                f"arm {self.name}: source must be one of {SOURCES}; got {self.source!r}"
            )
        if self.source in POLISH_SOURCES:
            if not (_integer(self.steps) and 1 <= self.steps <= EPOCH_VIEWS):
                raise ValueError(
                    f"arm {self.name}: a polish needs steps from 1 to {EPOCH_VIEWS}; got "
                    f"{self.steps!r}"
                )
        elif self.steps is not None:
            raise ValueError(f"arm {self.name}: steps apply to a polish only; got {self.steps!r}")


@dataclass(frozen=True)
class GenSettings:
    """GEN's construction (``arc26.hyper.validation.gen_puzzles``).

    Attributes:
        out_dir: GEN's folder, the workers' ``data_dir``.
        barc: BARC puzzles, the first by sorted id.
        rearc: ReARC puzzles, dealt round-robin over the held-out families.
    """

    out_dir: str
    barc: int
    rearc: int

    def __post_init__(self) -> None:
        _text("gen out_dir", self.out_dir)
        for name in ("barc", "rearc"):
            value = getattr(self, name)
            if not (_integer(value) and value >= 0):
                raise ValueError(f"gen {name} must be an integer >= 0; got {value!r}")
        if self.barc + self.rearc < 1:
            raise ValueError("gen needs at least one puzzle")


@dataclass(frozen=True)
class LoadCheckSettings:
    """H's load check (``arc26.hyper.validation.load_check``).

    Attributes:
        run_logs: The training run's ``run_logs`` folder (``status.json``, ``metrics.jsonl``).
        max_abs: Largest per-episode difference allowed, current H and zero adapter.
        max_mean: Largest mean difference allowed.
        dev4_challenges: The challenges file holding DEV4 for the fingerprint stage, or None.
    """

    run_logs: str
    max_abs: float
    max_mean: float
    dev4_challenges: str | None = None

    def __post_init__(self) -> None:
        _text("load_check run_logs", self.run_logs)
        _text("load_check dev4_challenges", self.dev4_challenges, optional=True)
        for name in ("max_abs", "max_mean"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
                raise ValueError(f"load_check {name} must be a positive number; got {value!r}")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class Arms:
    """The arms of one solver run: the solver JSON's ``adapters`` (``RunConfig.adapters``).

    Attributes:
        arms: The arms in the order every puzzle runs them.
        outputs_root: Candidate folders, one subfolder per arm; never the run's ``outputs_dir``.
        log_dir: Run logs of the arms: per arm its timing records, selection log, submission, and
            scores; the workers' records; the load check's and GEN's reports; the summary.
        cache_dir: Writable embedding caches, one per worker (``rank<k>``; one writer each).
        checkpoint: H's checkpoint (``best.pt``); needed by an H arm.
        cache: A read-only embedding cache root whose entries are copied into the workers'
            caches before H runs (``MT-01``'s), or None.
        hypernet: The expected H config (``configs/hypernet.yaml`` as built); needed by an H arm.
        views: ``decode_puzzle``'s view selection for every arm (``check_views``); None decodes the
            reference's 16 views.
        order_seed: Seed of the queue's shuffled order.
        ce: Whether each arm records the teacher-forced CE of every answer (reads the solutions,
            development mode only).
        finish_s: Seconds after the soft deadline that a puzzle already started may take to run
            its arms; decoding stops there.
        train_log: Whether every training arm records its per-step loss (``TRAIN_LOG_ARGS``).
        episodes_root: Folder of P3.4's build, for GEN and the load check.
        check_root: Folder of its leakage check report.
        manifest_sha256: The build's manifest sha256.
        gen: GEN's construction, or None to solve the run's ``data_dir``.
        load_check: H's load check, or None.

    Raises:
        TypeError: If a nested setting has the wrong type.
        ValueError: If a value breaks its rule.
    """

    arms: tuple[ArmSpec, ...]
    outputs_root: str
    log_dir: str
    cache_dir: str
    checkpoint: str | None = None
    cache: str | None = None
    hypernet: Mapping[str, Any] | None = None
    views: tuple[tuple[int, ...], ...] | None = None
    order_seed: int = 0
    ce: bool = True
    finish_s: float = 0.0
    train_log: bool = False
    episodes_root: str | None = None
    check_root: str | None = None
    manifest_sha256: str | None = None
    gen: GenSettings | None = None
    load_check: LoadCheckSettings | None = None

    NESTED: ClassVar[dict[str, type]] = {"gen": GenSettings, "load_check": LoadCheckSettings}

    def __post_init__(self) -> None:
        arms = tuple(
            ArmSpec(**arm) if isinstance(arm, Mapping) else arm for arm in (self.arms or ())
        )
        if not arms or not all(isinstance(arm, ArmSpec) for arm in arms):
            raise ValueError("adapters needs at least one arm")
        names = [arm.name for arm in arms]
        if len(set(names)) != len(names):
            raise ValueError(f"arm names must be distinct; got {names}")
        object.__setattr__(self, "arms", arms)
        for name in ("outputs_root", "log_dir", "cache_dir"):
            _text(name, getattr(self, name))
        for name in ("checkpoint", "cache", "episodes_root", "check_root", "manifest_sha256"):
            _text(name, getattr(self, name), optional=True)
        for name, kind in self.NESTED.items():
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, kind(**value))
            elif value is not None and not isinstance(value, kind):
                raise TypeError(f"{name} must be null or {kind.__name__} settings")
        if self.hypernet is not None:
            if not isinstance(self.hypernet, Mapping):
                raise TypeError(f"hypernet must be a mapping; got {self.hypernet!r}")
            object.__setattr__(self, "hypernet", dict(self.hypernet))
        object.__setattr__(self, "views", check_views(self.views))
        if not _integer(self.order_seed):
            raise ValueError(f"order_seed must be an integer; got {self.order_seed!r}")
        if not isinstance(self.ce, bool):
            raise ValueError(f"ce must be true or false; got {self.ce!r}")
        if not isinstance(self.train_log, bool):
            raise ValueError(f"train_log must be true or false; got {self.train_log!r}")
        finish = self.finish_s
        if isinstance(finish, bool) or not isinstance(finish, int | float) or finish < 0:
            raise ValueError(f"finish_s must be a number >= 0; got {finish!r}")
        object.__setattr__(self, "finish_s", float(finish))
        if any(arm.source in H_SOURCES for arm in arms) and (
            self.checkpoint is None or self.hypernet is None
        ):
            raise ValueError(
                "an H arm needs checkpoint (best.pt) and hypernet (the expected config)"
            )
        if self.load_check is not None and self.checkpoint is None:
            raise ValueError("the load check needs checkpoint")
        if (self.gen is not None or self.load_check is not None) and None in (
            self.episodes_root, self.check_root, self.manifest_sha256
        ):  # fmt: skip
            raise ValueError("gen and load_check need episodes_root, check_root, manifest_sha256")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Arms:
        """Build the settings from a mapping, such as the solver JSON's ``adapters``.

        Args:
            mapping: The fields by name; ``arms`` as a list of mappings, ``gen`` and
                ``load_check`` as mappings or null.

        Returns:
            The settings.

        Raises:
            TypeError: If ``mapping`` is not a mapping.
            ValueError: If a key is unknown or a value is invalid.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"adapters must be a mapping; got {mapping!r}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"adapters: unknown keys {unknown}; expected {sorted(known)}")
        return cls(**mapping)

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> Arms | None:
        """Read the settings from a solver config.

        Args:
            cfg: The solver config; its optional ``adapters`` is read.

        Returns:
            The settings, or None when the key is absent or null.
        """
        raw = cfg.get("adapters")
        return None if raw is None else cls.from_mapping(raw)

    def to_json(self) -> dict[str, Any]:
        """Return the settings as the solver JSON's ``adapters``: plain JSON values.

        Returns:
            Every field by name; tuples as lists.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "arms":
                value = [asdict(arm) for arm in value]
            elif f.name in self.NESTED and value is not None:
                value = asdict(value)
            elif f.name == "views" and value is not None:
                value = [list(batch) for batch in value]
            elif f.name == "hypernet" and value is not None:
                value = json.loads(json.dumps(value))
            out[f.name] = value
        return out

    def folders(self) -> list[tuple[str, str]]:
        """Return the folders the arms write, by name, for ``RunConfig``'s overlap check."""
        found = [
            ("adapters outputs_root", self.outputs_root),
            ("adapters log_dir", self.log_dir),
            ("adapters cache_dir", self.cache_dir),
        ]
        if self.gen is not None:
            found.append(("adapters gen out_dir", self.gen.out_dir))
        return found


# --- one worker's arms ---------------------------------------------------------------------------


def seed_cache(target: Any, source: Any, hashes: Iterable[str]) -> int:
    """Copy the embeddings ``source`` holds and ``target`` lacks into ``target``, in one put.

    Why: ``MT-01``'s cache is a read-only mount and ``EmbeddingCache`` has one writer, so each
    worker encodes into a cache of its own; the pairs ``MT-01`` cached (GEN's demonstrations)
    are copied, bit for bit (both store float32), instead of encoded again.

    Args:
        target: The worker's writable ``EmbeddingCache``.
        source: The read-only ``EmbeddingCache`` of the same namespace.
        hashes: Pair hashes wanted.

    Returns:
        The entries copied.
    """
    found = [key for key in dict.fromkeys(hashes) if key not in target and key in source]
    if found:
        target.put(dict(zip(found, source.get(found), strict=True)))
    return len(found)


def answer_ce(
    task: Mapping[str, Any],
    solutions: Sequence[Any],
    key: str,
    *,
    formatter: Any,
    tokenizer: Any,
    model: Any,
    max_seq_length: int,
    max_new_tokens: int,
) -> list[dict[str, float]]:
    r"""Return the teacher-forced CE of each test output under the live adapter.

    Why: the continuous metric with the smallest noise (task brief): one forward per output, under
    ``torch.inference_mode`` as ``decode_puzzle`` scores,
    $\mathrm{CE} = -\frac{1}{|y|} \sum_t \log p(y_t \mid x, y_{<t})$ over the reply's tokens (the
    grid, its newlines, and the closing ``<|im_end|>``), the prompt the decode's identity view
    (``cut_to_len`` to the decode's limit), scored with the reference's ``calc_scores``.

    Args:
        task: The puzzle's challenges entry.
        solutions: Its output grids, one per test input.
        key: The puzzle's task id.
        formatter: The worker's formatter.
        tokenizer: The model's tokenizer.
        model: The live model in inference mode.
        max_seq_length: The token limit of a prompt and its reply.
        max_new_tokens: The longest reply.

    Returns:
        Per test output ``{"nll", "tokens", "ce"}``.
    """
    import torch

    from arc26.decode import calc_scores
    from arc26.nvarc.arc_loader import ArcDataset

    out = []
    for index, (test, solution) in enumerate(zip(task["test"], solutions, strict=True)):
        sub = f"{key}_{index}"
        dataset = ArcDataset(
            queries={sub: {"train": task["train"], "test": [test]}}, replies={sub: [solution]}
        )
        dataset = dataset.cut_to_len(
            formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens
        )
        sample = dataset.as_list(formatter)[0]
        # decode_puzzle's mode (decode.py runs calc_scores under inference_mode): no autograd
        # bookkeeping for the forward of any arm or puzzle.
        with torch.inference_mode():
            nll = calc_scores([sample["input"]], [sample["reply"]], tokenizer, model)[0]
        tokens = len(tokenizer.encode(sample["reply"]))
        out.append({"nll": nll, "tokens": tokens, "ce": nll / tokens})
    return out


def _digest(model: Any) -> str:
    from peft import get_peft_model_state_dict

    from arc26.hyper.providers import ADAPTER_NAME, adapter_digest

    state = get_peft_model_state_dict(model, adapter_name=ADAPTER_NAME, save_embedding_layers=False)
    return adapter_digest(state)


def _hypernet_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(value)
    out["target_modules"] = list(out.get("target_modules") or [])
    return out


class BaseMerge:
    r"""The live model's targeted base weights with a LoRA delta merged in, and their exact copy.

    Why: arm ``hyper_merge_fit`` trains a fresh LoRA on the base plus H's adapter (task P3.9b
    amendment 1). Every forward the solver runs reads a LoRA layer's frozen weight as
    ``base_layer.weight``: PEFT's ``lora.Linear.forward`` through ``self.base_layer(x)``, and
    unsloth 2025.9.7's fast paths (``get_lora_parameters`` and ``get_lora_parameters_bias`` in
    ``unsloth/kernels/utils.py``, read by ``apply_lora_qkv``, ``apply_lora_o``, the patched PEFT
    forward, and the decode's ``fast_linear_forward``); none keeps a fused copy. So the merge
    writes into those tensors in place. The copy is taken on the weights' own device before any
    write; ``restore`` copies it back, bit for bit, never by subtracting the delta (bf16 rounds
    $W + \Delta$, so $\mathrm{bf16}(W + \Delta) - \Delta \ne W$ in general).

    The delta of layer $\ell$ is computed in float32 from the factors the live adapter holds,
    $\Delta_\ell = s\,B_\ell A_\ell$ with $s$ the layer's own ``scaling`` (rsLoRA,
    $\alpha/\sqrt{r}$), and the sum is rounded once, $W_\ell \leftarrow \mathrm{dtype}(W_\ell +
    \Delta_\ell)$. ``add`` measures that rounding:
    $\rho = \lVert \mathrm{dtype}(W + \Delta) - W - \Delta \rVert_F / \lVert \Delta \rVert_F$ over
    all layers.

    Attributes:
        layers: ``(name, module)`` of every PEFT LoRA layer holding the adapter.
        params: Each layer's ``base_layer.weight`` as merged (the objects, for ``restore``).
        copies: Each weight's exact copy, same dtype and device.

    Raises:
        ValueError: If the model holds no LoRA layer of the adapter, or a layer is not a plain
            unquantized ``Linear`` LoRA (fan-in-fan-out, DoRA, quantized, or already merged).
    """

    def __init__(self, model: Any, adapter_name: str = "default") -> None:
        """Find the adapter's LoRA layers and copy their base weights.

        Args:
            model: The live PEFT model.
            adapter_name: The adapter whose layers are merged (the solver's ``default``).
        """
        from peft.tuners.lora import LoraLayer

        self.adapter_name = adapter_name
        self.layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, LoraLayer) and adapter_name in module.lora_A
        ]
        if not self.layers:
            raise ValueError(f"the model holds no LoRA layer of adapter {adapter_name!r}")
        for name, module in self.layers:
            weight = module.get_base_layer().weight
            plain = (
                not getattr(module, "fan_in_fan_out", False)
                and not module.use_dora.get(adapter_name, False)
                and not module.merged
                and weight.dtype.is_floating_point
                and getattr(weight, "quant_state", None) is None
            )
            if not plain:
                raise ValueError(f"{name}: not a plain unquantized Linear LoRA layer to merge into")
        self.params = [module.get_base_layer().weight for _, module in self.layers]
        self.copies = [param.detach().clone() for param in self.params]

    @property
    def nbytes(self) -> int:
        """Bytes the copy holds."""
        return sum(copy.numel() * copy.element_size() for copy in self.copies)

    def add(self) -> dict[str, Any]:
        r"""Merge each layer's $s\,B A$ into its base weight in place; return the merge's numbers.

        Returns:
            ``layers``, ``copy_bytes``, ``device``, ``dtype``, ``delta_norm``
            ($\lVert\Delta\rVert_F$ over all layers), ``weight_norm`` (the copy's),
            ``rounding_norm`` and ``rounding_rel`` ($\rho$).
        """
        import torch

        delta_sq = rounding_sq = weight_sq = 0.0
        with torch.no_grad():
            for (_, module), param, copy in zip(self.layers, self.params, self.copies, strict=True):
                lora_a = module.lora_A[self.adapter_name].weight.float()  # [r, in]
                lora_b = module.lora_B[self.adapter_name].weight.float()  # [out, r]
                delta = module.scaling[self.adapter_name] * (lora_b @ lora_a)  # [out, in]
                merged = copy.float() + delta
                param.copy_(merged)
                delta_sq += float(delta.square().sum())
                rounding_sq += float((param.float() - merged).square().sum())
                weight_sq += float(copy.float().square().sum())
        delta_norm = math.sqrt(delta_sq)
        return {
            "layers": len(self.layers),
            "copy_bytes": self.nbytes,
            "device": str(self.params[0].device),
            "dtype": str(self.params[0].dtype).removeprefix("torch."),
            "delta_norm": delta_norm,
            "weight_norm": math.sqrt(weight_sq),
            "rounding_norm": math.sqrt(rounding_sq),
            "rounding_rel": math.sqrt(rounding_sq) / delta_norm if delta_norm > 0 else None,
        }

    def restore(self) -> None:
        """Copy every base weight back from its copy, bit for bit, and check it.

        Raises:
            RuntimeError: If a layer's base weight is no longer the tensor merged into (the
                forward then read other weights), or a weight differs from its copy afterwards.
        """
        import torch

        held = list(zip(self.layers, self.params, self.copies, strict=True))
        with torch.no_grad():
            for (name, module), param, copy in held:
                if module.get_base_layer().weight is not param:
                    raise RuntimeError(f"{name}: the base weight was replaced after the merge")
                param.copy_(copy)
        unequal = [name for (name, _), param, copy in held if not torch.equal(param, copy)]
        if unequal:
            raise RuntimeError(f"base weights differ from their copy after the restore: {unequal}")


class ArmRunner:
    """One worker's arms: the providers, caches, loggers, and the per-puzzle loop.

    Attributes:
        settings: The run's arms.
        loaded: What ``arc26.solver.load`` returned (the live model at its first state).
        rank: Worker rank.
        hyper: The H provider, or None without an H arm.
        source_cache: The read-only cache, or None.
        solutions: Output grids by task id for the CE, or None.
        loggers: The timing logger of each arm, by name.
        records: The worker's records file.
    """

    def __init__(
        self,
        settings: Arms,
        loaded: Loaded,
        rank: int,
        *,
        hyper: Any = None,
        source_cache: Any = None,
        solutions: Mapping[str, Any] | None = None,
    ) -> None:
        """Set up the loggers and folders of one worker.

        Args:
            settings: The run's arms.
            loaded: What ``arc26.solver.load`` returned.
            rank: Worker rank.
            hyper: The H provider (``HyperProvider``), needed by an H arm.
            source_cache: The read-only ``EmbeddingCache``, or None.
            solutions: Output grids by task id, needed when ``settings.ce``.

        Raises:
            ValueError: If an H arm has no provider, or the CE has no solutions.
        """
        if any(arm.source in H_SOURCES for arm in settings.arms) and hyper is None:
            raise ValueError("an H arm needs the H provider")
        if settings.ce and solutions is None:
            raise ValueError("the CE needs the solutions")
        self.settings = settings
        self.loaded = loaded
        self.rank = rank
        self.hyper = hyper
        self.source_cache = source_cache
        self.solutions = solutions
        root = Path(settings.log_dir)
        self.loggers = {
            arm.name: timing.TimingLogger(root / arm.name / "timing", f"rank{rank}")
            for arm in settings.arms
        }
        for arm in settings.arms:
            Path(settings.outputs_root, arm.name).mkdir(parents=True, exist_ok=True)
        self.records = root / f"{RECORDS_PREFIX}rank{rank}.jsonl"
        # The records of the last solve, for the first-puzzle guard (run_worker).
        self.last: list[dict[str, Any]] = []
        # hyper_merge_fit's merged base weights and their copy, from the merge until solve puts
        # the copy back after the arm's decode.
        self.merged: BaseMerge | None = None

    @classmethod
    def for_worker(
        cls, settings: Arms, loaded: Loaded, rank: int, data_dir: str | os.PathLike[str]
    ) -> ArmRunner:
        """Build a worker's runner: H from the checkpoint, the caches, the solutions.

        Args:
            settings: The run's arms.
            loaded: What ``arc26.solver.load`` returned.
            rank: Worker rank; its cache is ``<cache_dir>/rank<rank>``.
            data_dir: The run's data folder; its solutions file is read when ``settings.ce``.

        Returns:
            The runner.

        Raises:
            ValueError: If H's config is not the expected one.
        """
        hyper = source = None
        if any(arm.source in H_SOURCES for arm in settings.arms):
            from arc26.hyper.encoder import EmbeddingCache
            from arc26.hyper.providers import HyperProvider

            hyper = HyperProvider.from_checkpoint(
                settings.checkpoint,
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                cache_root=Path(settings.cache_dir) / f"rank{rank}",
            )
            # The checkpoint's form (task P3.3b): a default aggregator is left out, as the build
            # leaves it out of the expected config.
            found = _hypernet_dict(hyper.hypernet.config.to_dict())
            if found != _hypernet_dict(settings.hypernet or {}):
                raise ValueError(f"H's config {found} is not the build's {settings.hypernet}")
            if settings.cache is not None:
                namespace = hyper.cache.namespace
                source = EmbeddingCache(
                    settings.cache,
                    model_id=namespace["model_id"],
                    layer=namespace["layer"],
                    pool=namespace["pool"],
                    view_set_id=namespace["view_set_id"],
                )
        solutions = None
        if settings.ce:
            path = Path(data_dir) / SOLUTIONS_FILE
            solutions = json.loads(path.read_text(encoding="utf-8"))
        return cls(settings, loaded, rank, hyper=hyper, source_cache=source, solutions=solutions)

    def _record(self, entry: Mapping[str, Any]) -> None:
        self.records.parent.mkdir(parents=True, exist_ok=True)
        with self.records.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, allow_nan=True) + "\n")

    def solve(self, model: Any, key: str, end_time: float) -> Any:
        """Run every arm on one puzzle, in order, and record each.

        Why: paired arms (module docstring). An arm that raises is recorded and the next runs,
        since ``torch.multiprocessing.spawn`` would stop every worker; the puzzle is then
        incomplete. No arm starts after ``end_time + finish_s``.

        Args:
            model: The live model as the previous puzzle left it.
            key: The task id.
            end_time: The soft deadline, ``time.time()`` scale.

        Returns:
            The live model, for the next puzzle.
        """
        import gc

        import torch

        from arc26.decode import decode_puzzle
        from arc26.loop import DemoSet, Pair

        loaded = self.loaded
        task = loaded.arc_test_set.queries[key]
        demos = DemoSet(
            key,
            tuple(Pair(item["input"], item["output"]) for item in task["train"]),
            None,
            tuple(item["input"] for item in task["test"]),
        )
        finish = end_time + self.settings.finish_s
        seeded = False
        self.last = []
        for position, arm in enumerate(self.settings.arms):
            logger = self.loggers[arm.name]
            entry: dict[str, Any] = {
                "puzzle_id": key,
                "arm": arm.name,
                "source": arm.source,
                "rank": self.rank,
                "position": position,
                "steps": arm.steps,
            }
            if time.time() > finish:
                self._record({**entry, "status": NOT_STARTED})
                print(f"[Rank {self.rank}] {key}: arm {arm.name} not started, past the deadline")
                continue
            start_time = time.time()
            start = time.perf_counter()
            try:
                try:
                    if arm.source in H_SOURCES and not seeded and self.source_cache is not None:
                        hashes = [demos_pair_hash(pair) for pair in demos.pairs]
                        seed_start = time.perf_counter()
                        entry["seeded"] = seed_cache(self.hyper.cache, self.source_cache, hashes)
                        logger.record(key, CACHE_SEED, time.perf_counter() - seed_start)
                        seeded = True
                    logged = self.settings.train_log and arm.source in TRAINED_SOURCES
                    train_steps: list[dict[str, Any]] = []
                    with train_log_capture(train_steps) if logged else contextlib.nullcontext():
                        model = self._adapter(arm, model, demos, key, finish, logger, entry)
                    if logged:
                        entry["train_log"] = train_steps
                    entry["adapter_s"] = time.perf_counter() - start
                    fingerprint_start = time.perf_counter()
                    entry["fingerprint"] = _digest(model)
                    logger.record(key, FINGERPRINT, time.perf_counter() - fingerprint_start)
                    if self.solutions is not None:
                        ce_start = time.perf_counter()
                        entry["ce"] = answer_ce(
                            task,
                            self.solutions[key],
                            key,
                            formatter=loaded.formatter,
                            tokenizer=loaded.tokenizer,
                            model=model,
                            max_seq_length=loaded.max_seq_length,
                            max_new_tokens=loaded.max_new_tokens,
                        )
                        logger.record(
                            key,
                            CE,
                            time.perf_counter() - ce_start,
                            sum(item["tokens"] for item in entry["ce"]),
                        )
                    decode_puzzle(
                        puzzle_ds=loaded.arc_test_set.change_keys([key]),
                        formatter=loaded.formatter,
                        max_seq_length=loaded.max_seq_length,
                        max_new_tokens=loaded.max_new_tokens,
                        start_time=start_time,
                        end_time=finish,
                        rank=self.rank,
                        key=key,
                        tokenizer=loaded.tokenizer,
                        model=model,
                        max_score=loaded.max_score,
                        timing_logger=logger,
                        dir_outputs=str(Path(self.settings.outputs_root) / arm.name),
                        views=self.settings.views,
                    )
                finally:
                    # hyper_merge_fit: the base weights back after the decode, or after the arm
                    # raised, before anything else reads the model.
                    self._unmerge(key, logger, entry)
            except Exception as err:
                traceback.print_exc()
                entry.update(status=RAISED, error=f"{type(err).__name__}: {err}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                entry["status"] = CUT if time.time() >= finish else DONE
                logger.record(key, timing.PUZZLE, time.perf_counter() - start)
            entry["seconds"] = time.perf_counter() - start
            self._record(entry)
            self.last.append(entry)
            print(
                f"[Rank {self.rank}] {key}: arm {arm.name} ({arm.source}) {entry['status']} in "
                f"{entry['seconds']:.1f}s"
            )
            if self.merged is not None:
                # Every later arm and puzzle would run on H's merged base: stop the run.
                self.stop(f"arm {arm.name} on {key}: the base weights were not restored", entry)
        return model

    def _unmerge(self, key: str, logger: timing.TimingLogger, entry: dict[str, Any]) -> None:
        # Put the base weights back from hyper_merge_fit's copy, bit for bit; nothing to do for
        # the other arms. self.merged stays set when the restore raises (solve then stops).
        if self.merged is None:
            return
        start = time.perf_counter()
        self.merged.restore()
        self.merged = None
        logger.record(key, UNMERGE, time.perf_counter() - start)
        entry["restored"] = True

    def probe(self, model: Any) -> Any:
        """Run H's adapter path once on the solver's model, before the worker's first puzzle.

        Why: the load check runs transformers' base; the H arms' live encode (under
        ``disable_adapter``) and swap on the unsloth-patched model first run here, so a failure
        costs a few seconds, not a puzzle. The puzzle is the smallest DEV4 task (fewest
        demonstration cells) of the load check's challenges, or of the run's own when there are
        none; the live adapter is rolled back to its weights before the probe, bit for bit.

        Args:
            model: The live model, as ``arc26.solver.load`` returned it.

        Returns:
            The live model with its weights before the probe.

        Raises:
            ArmStop: If the probe raised; its record is written first.
        """
        from arc26.data import DEV4
        from arc26.loop import DemoSet, Pair

        tasks: dict[str, Any] = {}
        check = self.settings.load_check
        if check is not None and check.dev4_challenges is not None:
            data = json.loads(Path(check.dev4_challenges).read_text(encoding="utf-8"))
            tasks = {key: data[key] for key in DEV4 if key in data}
        if not tasks:
            tasks = dict(self.loaded.arc_test_set.queries)

        def cells(task: Mapping[str, Any]) -> int:
            return sum(
                len(row) for pair in task["train"] for grid in pair.values() for row in grid
            )

        key = min(sorted(tasks), key=lambda name: cells(tasks[name]))
        task = tasks[key]
        demos = DemoSet(
            key,
            tuple(Pair(item["input"], item["output"]) for item in task["train"]),
            None,
            tuple(item["input"] for item in task["test"]),
        )
        entry: dict[str, Any] = {
            "puzzle_id": key, "arm": PROBE, "source": HYPER, "rank": self.rank, "position": -1
        }  # fmt: skip
        start = time.perf_counter()
        try:
            self.hyper.timing_logger = self.loaded.timing_logger
            entry["cache_misses"] = len(
                self.hyper.cache.missing([demos_pair_hash(p) for p in demos.pairs])
            )
            model = self.hyper.adapt(demos, model, 0.0)
            entry["fingerprint"] = _digest(model)
            model = self.hyper.rollback(model)
        except Exception as err:
            traceback.print_exc()
            entry.update(
                status=RAISED, error=f"{type(err).__name__}: {err}",
                seconds=time.perf_counter() - start,
            )  # fmt: skip
            self._record(entry)
            self.stop(f"H's probe on {key} raised", entry)
        entry.update(status=DONE, seconds=time.perf_counter() - start)
        self._record(entry)
        if self.loaded.timing_logger is not None:
            self.loaded.timing_logger.record(key, H_PROBE, entry["seconds"])
        print(f"[Rank {self.rank}] H's probe on {key}: done in {entry['seconds']:.1f}s")
        return model

    def stop(self, reason: str, entry: Mapping[str, Any]) -> None:
        """Write this worker's stop marker and raise ``ArmStop`` (the parent exits 7).

        Args:
            reason: Why the run stops.
            entry: The record of the arm or probe that failed.

        Raises:
            ArmStop: Always.
        """
        marker = Path(self.settings.log_dir) / f"{STOP_PREFIX}rank{self.rank}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"rank": self.rank, "reason": reason, "record": dict(entry)}, indent=1)
            + "\n",
            encoding="utf-8",
        )
        print(f"[Rank {self.rank}] STOP: {reason}")
        raise ArmStop(reason)

    def _adapter(
        self,
        arm: ArmSpec,
        model: Any,
        demos: Any,
        key: str,
        finish: float,
        logger: timing.TimingLogger,
        entry: dict[str, Any],
    ) -> Any:
        # Put the arm's adapter into the live model and return the model in inference mode.
        from peft import set_peft_model_state_dict
        from unsloth import FastLanguageModel

        from arc26.ttt import SGDProvider, fit

        loaded = self.loaded
        budget = max(0.0, finish - time.time())
        # Every training arm gets the same arguments; train_log only adds the per-step logging.
        train_args = dict(loaded.train_args)
        if self.settings.train_log:
            train_args.update(TRAIN_LOG_ARGS)

        def reset() -> None:
            # fit's own reset: the adapter's weights right after wrapping (lora_B zero).
            start = time.perf_counter()
            set_peft_model_state_dict(model, loaded.default_weights.copy(), adapter_name="default")
            logger.record(key, RESET, time.perf_counter() - start)

        def polish(steps: int) -> SGDProvider:
            return SGDProvider(
                key=key,
                formatter=loaded.formatter,
                max_seq_length=loaded.max_seq_length,
                tokenizer=loaded.tokenizer,
                collator=loaded.collator,
                train_args=train_args,
                timing_logger=logger,
                rank=self.rank,
                steps=steps,
            )

        def full_fit(model: Any, start: Mapping[str, Any]) -> Any:
            # P2.6-01's training, statement for statement: arc26.ttt.fit, one epoch over the
            # puzzle's 128 views, reset first to ``start``; the sgd and hyper_fit arms differ
            # only there (the initial weights, or H's adapter).
            model, _ = fit(
                model=model,
                default_weights=start,
                arc_test_set=loaded.arc_test_set,
                key=key,
                formatter=loaded.formatter,
                max_seq_length=loaded.max_seq_length,
                tokenizer=loaded.tokenizer,
                collator=loaded.collator,
                train_args=train_args,
                timing_logger=logger,
                rank=self.rank,
            )
            return model

        if arm.source == SGD:
            return full_fit(model, loaded.default_weights)
        if arm.source == BASE:
            reset()
            return FastLanguageModel.for_inference(model)
        if arm.source == POLISH:
            reset()
            return polish(arm.steps).adapt(demos, model, budget)
        hyper = self.hyper
        hyper.timing_logger = logger
        entry["cache_misses"] = len(hyper.cache.missing([demos_pair_hash(p) for p in demos.pairs]))
        if arm.source == HYPER:
            model = hyper.adapt(demos, model, budget)
            return FastLanguageModel.for_inference(model)
        if arm.source == HYPER_FIT:
            from peft import get_peft_model_state_dict

            from arc26.loop import SNAPSHOT

            model = hyper.adapt(demos, model, budget)
            # H's adapter as the live model holds it (bf16, the solver's cast), copied to the
            # CPU as SGDProvider copies its snapshot: fit's reset loads it back bit for bit.
            snap = time.perf_counter()
            weights = get_peft_model_state_dict(model, adapter_name="default")
            start = {name: t.detach().to("cpu", copy=True) for name, t in weights.items()}
            logger.record(key, SNAPSHOT, time.perf_counter() - snap)
            return full_fit(model, start)
        if arm.source == HYPER_MERGE_FIT:
            model = hyper.adapt(demos, model, budget)
            entry["h_fingerprint"] = _digest(model)
            merge_start = time.perf_counter()
            merge = BaseMerge(model)
            # Set before the first write, so solve's restore covers a merge that raises midway.
            self.merged = merge
            entry["merge"] = merge.add()
            logger.record(key, MERGE, time.perf_counter() - merge_start)
            # sgd's call: fit resets the live LoRA to the initial weights (B = 0) before any
            # forward, so H's factors are not applied twice.
            return full_fit(model, loaded.default_weights)
        from arc26.hyper.providers import HyperPolishProvider

        return HyperPolishProvider(hyper, polish(arm.steps)).adapt(demos, model, budget)


@contextlib.contextmanager
def train_log_capture(sink: list[dict[str, Any]]) -> Iterator[None]:
    """Record every per-step log entry of ``arc26.ttt.UnslothFixedTrainer`` into ``sink``.

    Why: ``fit`` keeps the trainer to itself and captures its printout (moved code, ADR-011), so
    the per-step losses are read where the trainer emits them, its ``log`` call, without a
    change to ``fit``. With ``TRAIN_LOG_ARGS`` the trainer logs after every optimizer step.

    Args:
        sink: Receives ``{"step", "loss", "learning_rate", "grad_norm"}`` per logged step (the
            last two None when the trainer gives none).

    Yields:
        None; the trainer's ``log`` is restored on exit.
    """
    from arc26.ttt import UnslothFixedTrainer

    own = "log" in vars(UnslothFixedTrainer)
    original = UnslothFixedTrainer.log

    def log(self: Any, logs: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        if "loss" in logs:
            sink.append(
                {
                    "step": int(self.state.global_step),
                    "loss": float(logs["loss"]),
                    "learning_rate": _float_or_none(logs.get("learning_rate")),
                    "grad_norm": _float_or_none(logs.get("grad_norm")),
                }
            )
        return original(self, logs, *args, **kwargs)

    UnslothFixedTrainer.log = log
    try:
        yield
    finally:
        if own:
            UnslothFixedTrainer.log = original
        else:
            del UnslothFixedTrainer.log


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def demos_pair_hash(pair: Any) -> str:
    """Return the cache key of a ``DemoSet`` pair (P3.2's ``pair_hash`` of its task pair).

    Args:
        pair: An ``arc26.loop.Pair``.

    Returns:
        64 lowercase hex characters.
    """
    from arc26.hyper.encoder import pair_hash

    return pair_hash(pair.task_pair())


def run_worker(
    rank: int,
    queue: Any,
    end_time: float,
    cfg: Mapping[str, Any],
    loaded: Loaded,
    runner: ArmRunner | None = None,
) -> None:
    """Solve puzzles from the queue, every arm on each, as the reference worker loops.

    Why: ``arc26.solver.worker``'s loop (deadline, queue, ``None``, the printed lines, and the
    ``puzzle`` record of the whole puzzle) around ``ArmRunner.solve``. ``loaded`` is passed in,
    so the CPU smoke path runs the loop on a tiny model. With an H arm, H's path is probed first
    (``ArmRunner.probe``), and an H arm that raises on the worker's first finished puzzle stops
    the run: the GPU path of H on the solver's model is new, and a broken one would otherwise
    leave every H arm ``raised`` until the deadline.

    Args:
        rank: Worker rank.
        queue: The shared task queue: task ids, then one ``None`` per worker.
        end_time: The soft deadline, ``time.time()`` scale: no puzzle starts after it.
        cfg: The solver config with ``adapters``.
        loaded: The model and what every puzzle shares (``arc26.solver.load``).
        runner: The worker's runner; None builds it (``ArmRunner.for_worker``).

    Raises:
        ArmStop: If the probe raised, or an H arm raised on the first puzzle this worker
            finished.
    """
    import torch

    settings = Arms.from_cfg(cfg)
    if runner is None:
        runner = ArmRunner.for_worker(settings, loaded, rank, cfg["data_dir"])
    model = loaded.model
    if any(arm.source in H_SOURCES for arm in settings.arms):
        model = runner.probe(model)
    first = True
    while not queue.empty():
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break
        key = queue.get()
        if key is None:
            break
        start_time = time.time()
        puzzle_start = time.perf_counter()
        model = runner.solve(model, key, end_time)
        if first:
            first = False
            failed = [e for e in runner.last if e["source"] in H_SOURCES and e["status"] == RAISED]
            if failed:
                reason = f"H arm {failed[0]['arm']} raised on {key}, the first puzzle"
                runner.stop(reason, failed[0])
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB")
        print(f"[Rank {rank}] finished {key} in {time.time() - start_time:.1f}s")
        if loaded.timing_logger is not None:
            loaded.timing_logger.record(key, timing.PUZZLE, time.perf_counter() - puzzle_start)


def worker(rank: int, queue: Any, end_time: float, cfg: Mapping[str, Any]) -> None:
    """Load the model (``arc26.solver.load``), then run the arms on the queue's puzzles.

    Args:
        rank: Worker rank.
        queue: The shared task queue.
        end_time: The soft deadline, ``time.time()`` scale.
        cfg: The solver config with ``adapters``.
    """
    from arc26.solver import load

    run_worker(rank, queue, end_time, cfg, load(dict(cfg)))


def local_worker(rank: int, queue: Any, end_time: float, cfg: Mapping[str, Any]) -> None:
    """Run one spawned worker process, as ``arc26.solver.local_worker`` does, with the arms.

    Why: the reference's process body: GPU ``rank`` only, the GPU stack imported one rank after
    another (marker files, unsloth first), the timing logger for ``timing_dir``.

    Args:
        rank: Worker rank, also the GPU index.
        queue: The shared task queue.
        end_time: The soft deadline, ``time.time()`` scale.
        cfg: The solver config; this reads ``marker_dir`` and ``timing_dir``.
    """
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")
    if rank > 0:
        while not os.path.exists(os.path.join(cfg["marker_dir"], f"worker{rank - 1}")):
            time.sleep(5)
    from arc26 import decode, ttt  # noqa: F401

    with open(os.path.join(cfg["marker_dir"], f"worker{rank}"), "w") as handle:
        handle.write("Ok")
    if cfg["timing_dir"] is not None:
        timing.configure(cfg["timing_dir"], f"rank{rank}")
    print(f"[Rank {rank}] start!")
    worker(rank, queue, end_time, cfg)
    print(f"[Rank {rank}] done!")


# --- the parent: GEN, the load check, the queue, selection and scores -----------------------------


class _Tee:
    """A text stream that copies every write to a file as well."""

    def __init__(self, stream: TextIO, handle: TextIO) -> None:
        self.stream, self.handle = stream, handle

    def write(self, text: str) -> int:
        """Write to both; return what the stream returned."""
        self.handle.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        """Flush both."""
        self.handle.flush()
        self.stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


@contextlib.contextmanager
def _tee(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(_Tee(sys.stdout, handle)):
            with contextlib.redirect_stderr(_Tee(sys.stderr, handle)):
                yield


def score_arm(
    predictions: Mapping[str, Any], solutions: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, Any]:
    """Score one arm's attempts on the run's puzzles: the competition metric and attempt 1.

    Args:
        predictions: The arm's submission (task id to one ``{attempt_1, attempt_2}`` per output).
        solutions: Output grids by task id.
        keys: The run's puzzles.

    Returns:
        ``tasks``, ``score`` (the sum of task scores, two attempts), ``outputs``,
        ``solved_outputs``, ``attempt_1_outputs`` (exact matches of attempt 1), and per puzzle
        its ``correct`` and ``attempt_1`` flags per output and its ``score``.
    """
    from arc26.evaluate import is_grid, score_task

    puzzles = {}
    for key in keys:
        expected = solutions[key]
        task = score_task(key, predictions, expected)
        entries = predictions.get(key)
        entries = entries if isinstance(entries, list) else []
        first = []
        for index, solution in enumerate(expected):
            item = entries[index] if index < len(entries) else None
            attempt = item.get("attempt_1") if isinstance(item, Mapping) else None
            first.append(bool(is_grid(attempt) and attempt == solution))
        puzzles[key] = {"correct": list(task.correct), "attempt_1": first, "score": task.score}
    return {
        "tasks": len(keys),
        "score": sum(p["score"] for p in puzzles.values()),
        "outputs": sum(len(p["correct"]) for p in puzzles.values()),
        "solved_outputs": sum(sum(p["correct"]) for p in puzzles.values()),
        "attempt_1_outputs": sum(sum(p["attempt_1"]) for p in puzzles.values()),
        "puzzles": puzzles,
    }


def select_and_score(
    settings: Arms, data_dir: str | os.PathLike[str], keys: Sequence[str]
) -> dict[str, Any]:
    """Select each arm's two attempts with the reference's selection, then score them.

    Why: every arm is scored as the competition scores the submission: the reference's
    ``write_submission`` over the arm's candidate folder (its ``score_kgmon`` selection), then
    ``score_arm`` on the run's puzzles. What the selection prints goes to the arm's
    ``selection.log``; its development scoring raises when no candidate is correct, after the
    file is written, which is recorded, not raised.

    Args:
        settings: The run's arms.
        data_dir: The run's data folder (challenges and solutions).
        keys: The run's puzzles.

    Returns:
        Per arm: ``seconds``, ``error`` (None or the selection's), and the score summary.
    """
    from arc26.evaluate import load_predictions, load_solutions
    from arc26.nvarc.select import write_submission

    solutions = load_solutions(Path(data_dir) / SOLUTIONS_FILE)
    results = {}
    for arm in settings.arms:
        folder = Path(settings.outputs_root) / arm.name
        folder.mkdir(parents=True, exist_ok=True)
        logs = Path(settings.log_dir) / arm.name
        path = logs / SUBMISSION_FILE
        path.unlink(missing_ok=True)
        entry: dict[str, Any] = {"error": None}
        start = time.perf_counter()
        print(f"== selection of arm {arm.name}", flush=True)
        with _tee(logs / SELECTION_LOG):
            try:
                write_submission(str(data_dir), str(folder), str(path))
            except Exception as err:
                traceback.print_exc()
                entry["error"] = f"{type(err).__name__}: {err}"
        entry["seconds"] = time.perf_counter() - start
        if path.is_file():
            scores = score_arm(load_predictions(path), solutions, keys)
            (logs / SCORES_FILE).write_text(json.dumps(scores, indent=1) + "\n", encoding="utf-8")
            entry.update({k: v for k, v in scores.items() if k != "puzzles"})
        results[arm.name] = entry
    return results


def _clear(folder: str | os.PathLike[str]) -> None:
    # A folder of this run only: what an earlier launch left would mix into its files.
    path = Path(folder)
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_load_check(settings: Arms, cfg: Mapping[str, Any], end_time: float) -> int:
    """Run H's load check in a child process on GPU 0; return its exit code.

    Args:
        settings: The run's arms, with ``load_check``.
        cfg: The solver config (``model_dir``, ``timing_dir``).
        end_time: The soft deadline; the check gets at most ``LOAD_CHECK_TIMEOUT_S`` before it.

    Returns:
        0 when it passed, 6 when it failed, another code on an error or a timeout.
    """
    check = settings.load_check
    assert check is not None
    config = {
        "checkpoint": settings.checkpoint,
        "episodes_root": settings.episodes_root,
        "check_root": settings.check_root,
        "manifest_sha256": settings.manifest_sha256,
        "run_logs": check.run_logs,
        "hypernet": settings.hypernet,
        "model_dir": cfg["model_dir"],
        "cache": settings.cache,
        "device": "cuda",
        "max_abs": check.max_abs,
        "max_mean": check.max_mean,
        "dev4_challenges": check.dev4_challenges,
        "timing_dir": cfg.get("timing_dir"),
        "out": str(Path(settings.log_dir) / LOAD_CHECK_FILE),
    }
    command = [sys.executable, "-m", "arc26.hyper.validation", "check", "--config"]
    timeout = max(60.0, min(LOAD_CHECK_TIMEOUT_S, end_time - time.time()))
    print(f"load check: {' '.join(command)} <config>, at most {timeout:.0f} s", flush=True)
    try:
        done = subprocess.run(
            [*command, json.dumps(config)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("load check: timed out")
        return EXIT_FAILED
    return done.returncode


def spawn_workers(order: Sequence[str], end_time: float, cfg: Mapping[str, Any]) -> None:
    """Start one worker process per GPU on a shared queue of ``order``, as the reference does.

    Args:
        order: The puzzles in queue order.
        end_time: The soft deadline, ``time.time()`` scale.
        cfg: The workers' solver config (``workers``, ``data_dir`` the run's data).
    """
    import torch.multiprocessing as mp

    queue = mp.Manager().Queue()
    for key in order:
        queue.put(key)
    for _ in range(cfg["workers"]):
        queue.put(None)
    mp.spawn(local_worker, args=(queue, end_time, cfg), nprocs=cfg["workers"])


def run(
    cfg: Mapping[str, Any],
    end_time: float,
    *,
    launch: Callable[[Sequence[str], float, Mapping[str, Any]], None] = spawn_workers,
    check: Callable[[Arms, Mapping[str, Any], float], int] = run_load_check,
) -> int:
    """Run the arms: GEN, the load check, the workers, the selection and scores, the summary.

    Args:
        cfg: The solver config with ``adapters``.
        end_time: The soft deadline, ``time.time()`` scale.
        launch: Runs the workers on the ordered puzzles (``spawn_workers``; the CPU smoke path
            runs one worker in this process).
        check: Runs the load check and returns its exit code (``run_load_check``; the smoke
            path runs it in this process on the CPU).

    Returns:
        0, 1 (a worker failed), 2 (a competition rerun, or no adapters), 5 (the load check did
        not pass), or 7 (H's path failed on the solver's model: the probe, or an H arm on a
        worker's first puzzle; no selection).
    """
    started = time.time()
    settings = Arms.from_cfg(cfg)
    if settings is None or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        print("arc26.arms runs development arms only: no adapters, or a competition rerun")
        return EXIT_CONFIG
    for folder in (settings.outputs_root, settings.log_dir, settings.cache_dir):
        _clear(folder)
    summary: dict[str, Any] = {"settings": settings.to_json(), "end_time": end_time}
    data_dir = cfg["data_dir"]
    if settings.gen is not None:
        from arc26.hyper.validation import gen_from_episodes

        _clear(settings.gen.out_dir)
        summary["gen"] = gen_from_episodes(
            settings.episodes_root,
            settings.check_root,
            settings.manifest_sha256,
            settings.gen.out_dir,
            barc=settings.gen.barc,
            rearc=settings.gen.rearc,
        )
        Path(settings.log_dir, GEN_FILE).write_text(
            json.dumps(summary["gen"], indent=1) + "\n", encoding="utf-8"
        )
        data_dir = settings.gen.out_dir
    code = EXIT_OK
    if settings.load_check is not None:
        result = check(settings, cfg, end_time)
        summary["load_check_exit"] = result
        if result != 0:
            print(f"load check exit {result}: no puzzle is decoded (see {LOAD_CHECK_FILE})")
            code = EXIT_LOAD_CHECK
    if code == EXIT_OK:
        challenges = json.loads(Path(data_dir, CHALLENGES_FILE).read_text(encoding="utf-8"))
        subset = cfg.get("subset")
        order = queue_order(
            [key for key in challenges if subset is None or key in subset], settings.order_seed
        )
        summary["order"] = order
        try:
            launch(order, end_time, {**cfg, "data_dir": str(data_dir)})
        except Exception as err:
            traceback.print_exc()
            summary["workers_error"] = f"{type(err).__name__}: {err}"
            stops = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(Path(settings.log_dir).glob(f"{STOP_PREFIX}rank*.json"))
            ]
            code = EXIT_FAILED
            if stops:
                summary["stopped"] = stops
                code = EXIT_H_ARM
                print(f"stopped: {stops[0]['reason']} (exit {EXIT_H_ARM})")
        if code != EXIT_H_ARM:
            summary["selection"] = select_and_score(settings, data_dir, order)
    summary.update(exit=code, seconds=time.time() - started)
    Path(settings.log_dir, SUMMARY_FILE).write_text(
        json.dumps(summary, indent=1, allow_nan=True) + "\n", encoding="utf-8"
    )
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``solver_command``'s arguments and run the arms.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        The exit code of ``run``.
    """
    parser = argparse.ArgumentParser(prog="python -m arc26.arms")
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args(argv)
    end_time = args.end_time
    if not math.isfinite(end_time):
        raise SystemExit("--end-time must be finite")
    return run(json.loads(args.config), end_time)


if __name__ == "__main__":
    sys.exit(main())
