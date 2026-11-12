r"""Pair-embedding cache and encoder throughput on one GPU (task P3.2; BRIEF B2 Q1, B5 P3.2, A8).

Purpose: the GPU half of P3.2. It embeds every demonstration pair of the configured ARC tasks with
``arc26.hyper.encoder.PairEncoder`` into an ``EmbeddingCache``, and measures what the B5 P3.2
acceptance asks for: the seconds to encode k = 4 pairs under 16 views on one L4. The pre-run
brief is ``docs/prerun/P3.2-01.md``; the runbook ``docs/runbooks/cache_embeddings.md``.

Inputs: the YAML config (``configs/cache_embeddings.yaml``), an ARC challenges JSON (only the
``train`` pairs are read; a file named like a solutions file is refused), and a Hugging Face model
folder. Both are found by search under ``input_root`` (``/kaggle/input`` on Kaggle, whose mounts
moved within 2026), or named on the command line. The tasks are an evaluation subset of
``arc26.data`` (DEV40 for ``P3.2-01``). Outputs in ``out_dir`` (on Kaggle
``/kaggle/working/run_logs``, pulled to ``runs/<run_id>/``): ``config.yaml`` (resolved),
``run.json`` (commit, versions, device, the inputs found, model, encoder), ``timing.jsonl`` (one
record per encode stage), ``throughput.jsonl`` (one record per measured task), ``metrics.json``,
``summary.md``, ``view_diagnostic.jsonl`` when the view diagnostic runs, and ``error.log`` on a
failure. The cache shards go to ``cache_dir`` and stay on Kaggle.

Method (ADR-001): each pair is rendered in the reference notebook's token format under the decode
path's 16 views (8 dihedral transforms times 2 color permutations), prefilled through the first L
of the frozen base model's layers, and mean-pooled in fp32 over its output-grid tokens; a pair's
embedding is the mean over its views and a task's the mean over its pairs. Before the model loads,
every view of every pair is rendered once on the CPU, which checks that its grid tokens decode
back to its grids under the real tokenizer and gives the token-length distribution. The
throughput phase then times, per task, one fresh encode of its first k pairs (repeated when it has
fewer) after a warm-up, bypassing the cache. The cache phase encodes only pairs the cache lacks,
writing every ``flush_pairs`` pairs, so a rerun resumes where a killed one stopped; the throughput
ledger resumes the same way. With ``reference_n_perm`` set (run ``P3.2-02``), a last phase, a
diagnostic that decides nothing, measures how far each pair's embedding moves from the one under
a larger reference view set: the run's views are the first rounds of that set, so only the other
views are encoded. The script checks its kill criteria itself: wall time (exit 3), non-finite
embeddings, span checks, and inputs not found exactly once (exit 1).

Episode mode (task MT, run ``MT-01``; a config with an ``episodes`` section): the pairs are the
distinct demonstration pairs of P3.4's episode build (its training and validation splits,
deduplicated by ``pair_hash``), which meta-training reads $\bar z$ from. The build is verified
as training verifies it (manifest sha256, passing leakage report, shard hashes) and the exact
count is logged. The sorted missing pairs are cut into chunks of ``chunk_pairs``; ``gpus``
processes encode chunk $j$ on rank $j \bmod W$, one GPU each, and each writes every chunk to a
part file of its own as soon as it is done, so a cut session keeps all finished chunks. This
process, the only cache writer, merges the part files into the ``EmbeddingCache`` every few
minutes and at the end; a later session started with ``--seed-cache`` (the earlier session's
cache, attached as an input) copies it, merges leftover parts, and encodes only the rest. At the
end a seeded sample of cached pairs is re-encoded in other batches and compared with the cache
(the check). No throughput phase and no view diagnostic run in this mode.

Usage::

    python scripts/cache_embeddings.py --config configs/cache_embeddings.yaml --commit SHA
    python scripts/cache_embeddings.py --config configs/cache_embeddings.yaml --model-dir DIR \
        --challenges FILE --cache-dir DIR --out-dir DIR --device cuda:0
    python scripts/cache_embeddings.py --config configs/cache_embeddings.yaml --smoke
    python scripts/cache_embeddings.py --config configs/cache_embeddings_MT-01.yaml --commit SHA \
        --episodes-root DIR --check-root DIR [--seed-cache DIR]
"""

import argparse
import json
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import safetensors
import torch
import transformers
import yaml
from safetensors import safe_open
from safetensors.torch import save as safetensors_bytes
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from arc26.data import SUBSETS, subset_ids
from arc26.hyper.encoder import (
    EmbeddingCache,
    PairEncoder,
    apply_view,
    pair_hash,
    render_pair,
    view_descriptors,
    view_set_id,
)
from arc26.hyper.episodes import SPLITS
from arc26.hyper.train import load_split, open_episode_set
from arc26.nvarc.arc_loader import QwenFormatter

REPO = Path(__file__).resolve().parent.parent
EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_STOPPED = 0, 1, 2, 3
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
ATTENTION = ("sdpa", "eager", "flash_attention_2")
# Keys --smoke replaces, and the two it adds. subset is among them so that the GPU run's tasks
# (DEV40) never select from the smoke run's synthetic tasks.
SMOKE_OVERRIDES = (
    "run_id",
    "model_id",
    "subset",
    "device",
    "dtype",
    "attn_implementation",
    "layer",
    "batch_tokens",
    "max_wall_s",
    "cache_dir",
    "out_dir",
)
SMOKE_ONLY = ("tasks", "seed")
# The notebook's vocabulary (ARC_VOCAB and the token ids in arc26/nvarc/arc_solver.py): digits
# 0 to 9, "Ċ" (a newline under byte-level encoding) 10, user 11, assistant 12, padding 13, and
# <|im_end|> 15; <|im_start|> takes the one id left, 14 (inferred, unverified).
SMOKE_VOCAB = {
    **{str(d): d for d in range(10)},
    "Ċ": 10,
    "user": 11,
    "assistant": 12,
    "<|endoftext|>": 13,
    "<|im_start|>": 14,
    "<|im_end|>": 15,
}
SOLUTION_MARKER = "solution"
# The input search (find_files) walks input_root as the script notebook's environment check does:
# files at most SEARCH_DEPTH levels down (the model's config.json is at level 7 in the 2026 layout,
# K0), package trees skipped, each real folder once.
SEARCH_DEPTH = 8
SKIPPED_DIRS = frozenset({"site-packages", "dist-packages", "__pycache__", "node_modules", ".git"})
MODEL_CONFIG = "config.json"
# How many found paths an error message lists.
SHOWN_PATHS = 5
# The view diagnostic's per-pair records in out_dir.
DIAGNOSTIC_FILE = "view_diagnostic.jsonl"
# The diagnostic's between-pairs scale is taken over at most this many pairs: the matrix is
# quadratic in them, and 2,000 pairs (32 MB in float64) give a stable median.
BETWEEN_PAIRS_CAP = 2000
# Episode mode (task MT). The keys both modes share; an episode-mode config has these, the
# episode keys, and smoke, and no challenge-mode key.
COMMON_KEYS = (
    "run_id", "model_id", "input_root", "model_dir", "device", "dtype", "attn_implementation",
    "layer", "pool", "n_perm", "view_seed", "batch_tokens", "store_views", "flush_pairs",
    "max_wall_s", "cache_dir", "out_dir",
)  # fmt: skip
EPISODE_FIELDS = (
    "episodes", "gpus", "chunk_pairs", "seed_cache", "verify_pairs", "verify_min_cosine",
)  # fmt: skip
EPISODE_KEYS = (*COMMON_KEYS, *EPISODE_FIELDS)
EPISODE_SUBKEYS = ("root", "manifest_sha256", "check_root", "splits", "max_tokens")
EPISODE_SMOKE_OVERRIDES = (
    "run_id", "model_id", "device", "dtype", "attn_implementation", "layer", "batch_tokens",
    "max_wall_s", "cache_dir", "out_dir", "episodes", "gpus", "chunk_pairs", "verify_pairs",
)  # fmt: skip
EPISODE_SMOKE_ONLY = ("seed",)
# Rank k of an episode run uses cuda:k, so the device names the kind only.
EPISODE_DEVICES = ("cuda", "cpu")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# Part files of the encoding processes, inside the cache's namespace folder: the cache's index
# never names them, and a seeded session copies them with the namespace.
PARTS_DIR = "parts"
PART_GLOB = "part-*.safetensors"
# Seconds between the parent's merges of finished part files into the cache, and between its
# looks at the encoding processes.
MERGE_EVERY_S = 300.0
POLL_S = 2.0
# The check's sample is drawn with this seed (one fixed seed, CLAUDE.md).
VERIFY_SEED = 0
# The worst pairs the check reports by name.
VERIFY_WORST = 5
# Where the host record reads the machine's memory and the container's cgroup limit (v2, then v1);
# cgroup v1 reports "no limit" as a page-rounded 2**63 - 1, so anything this large is none.
MEMINFO = Path("/proc/meminfo")
CGROUP_MEMORY_FILES = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)
CGROUP_UNLIMITED = 2**60
SCRIPT = Path(__file__).resolve()


class ConfigError(ValueError):
    """The config file or a command-line override is invalid."""


class WallTimeExceeded(RuntimeError):
    """The wall-time kill criterion fired."""


class InputSearchError(RuntimeError):
    """The model folder or the challenges file was found zero times or more than once."""


class CheckFailed(RuntimeError):
    """The check's re-encoded sample disagrees with the cache (episode mode)."""


@dataclass(frozen=True)
class EpisodeSource:
    """The P3.4 build whose demonstration pairs an episode-mode run caches (task MT).

    Attributes:
        root: Folder searched for the build's ``MANIFEST.json``.
        manifest_sha256: sha256 of that manifest; any other build is refused.
        check_root: Folder searched for the build's passing leakage report.
        splits: Splits whose demonstration pairs are cached, in order.
        max_tokens: The episode cap that ``load_split`` asserts.
    """

    root: str
    manifest_sha256: str
    check_root: str
    splits: tuple[str, ...]
    max_tokens: int


@dataclass(frozen=True)
class Config:
    """The resolved run settings; ``configs/cache_embeddings.yaml`` gives each value's reason.

    Attributes:
        run_id: Pre-run brief id.
        model_id: Weights and precision: part of the cache key, and the Kaggle Models handle whose
            mount folder is searched for when ``model_dir`` is None.
        input_root: Folder searched for the model folder and the challenges file.
        model_dir: Hugging Face model folder; None finds it under ``input_root`` (``find_inputs``).
        challenges: ARC challenges JSON: an absolute path is read as is; a relative one is the end
            of the path of the one file under ``input_root`` that is read. None in episode mode.
        subset: Name of the ``arc26.data`` evaluation subset to process; None means every task.
        task_ids: The subset's task ids in order, derived from ``subset`` (not a config key).
        device: Torch device of the prefills.
        dtype: Model precision, a key of ``DTYPES``.
        attn_implementation: One of ``ATTENTION``.
        layer: Decoder layers run (L).
        pool: ``"output"`` or ``"input"``.
        n_perm: Color permutations per dihedral transform.
        view_seed: Seed of the permutation draws.
        reference_n_perm: The view diagnostic's reference set, ``view_descriptors(
            reference_n_perm, view_seed)``, whose first views are the run's; None skips it.
        batch_tokens: Padded tokens per prefill batch.
        store_views: Whether the cache keeps the per-view embeddings.
        flush_pairs: Encoded pairs held between cache writes.
        throughput_k: Pairs per throughput measurement; 0 skips it.
        throughput_tasks: Tasks measured from the start of the list; None means all.
        acceptance_s: Seconds bound that summary.md counts tasks against.
        max_wall_s: Wall-time kill criterion in seconds.
        cache_dir: Cache root.
        out_dir: Run-log directory.
        smoke_tasks: Synthetic tasks of the smoke run (0 in episode mode).
        smoke_seed: Seed of the smoke run's grids and weights.
        episodes: Episode mode's input (task MT); None in challenge mode.
        gpus: Encoding processes in episode mode, one GPU each.
        chunk_pairs: Pairs per encode call and per part file in episode mode.
        seed_cache: An earlier session's cache root to start from (episode mode), or None.
        verify_pairs: Cached pairs the check re-encodes at the end (episode mode).
        verify_min_cosine: The check's bound: every re-encoded pair's cosine to its cache entry.
    """

    run_id: str
    model_id: str
    input_root: str
    model_dir: str | None
    challenges: str | None
    subset: str | None
    task_ids: tuple[str, ...] | None
    device: str
    dtype: str
    attn_implementation: str
    layer: int
    pool: str
    n_perm: int
    view_seed: int
    reference_n_perm: int | None
    batch_tokens: int
    store_views: bool
    flush_pairs: int
    throughput_k: int
    throughput_tasks: int | None
    acceptance_s: float
    max_wall_s: float
    cache_dir: str
    out_dir: str
    smoke_tasks: int
    smoke_seed: int
    episodes: EpisodeSource | None = None
    gpus: int = 1
    chunk_pairs: int = 256
    seed_cache: str | None = None
    verify_pairs: int = 0
    verify_min_cosine: float = 0.999


# Config fields derived from other keys rather than read from the file.
DERIVED = ("task_ids",)
# Challenge mode's keys (P3.2's configs): every field but the derived, smoke, and episode ones.
TOP_KEYS = tuple(
    f.name
    for f in fields(Config)
    if not f.name.startswith("smoke_") and f.name not in DERIVED and f.name not in EPISODE_FIELDS
) + ("smoke",)


def _check(condition: bool, message: str) -> None:
    """Raise ``ConfigError(message)`` unless ``condition`` holds."""
    if not condition:
        raise ConfigError(message)


def _is_int(value: Any) -> bool:
    """True for an int that is not a bool (YAML reads ``true`` as a bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def model_folder_parts(model_id: str) -> tuple[str, ...]:
    """Return the path ending under which Kaggle mounts a Kaggle Models handle.

    Why: the handle ``<owner>/<model>/<framework>/<variation>/<version>`` is mounted in a folder
    whose path ends with ``<model>/<framework>/<variation>/<version>``, the framework in lower
    case: under ``models/<owner>/`` in 2026 (K0), directly under the input root in 2025 (the
    reference notebook's path). Matching that ending finds the folder in either layout, and only
    the version the handle names.

    Args:
        model_id: The handle, for example
            ``sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1``.

    Returns:
        The four parts after the owner, lower case.

    Raises:
        ConfigError: If ``model_id`` is not five non-empty parts separated by ``/``.
    """
    parts = model_id.split("/")
    _check(
        len(parts) == 5 and all(parts),
        f"model_id must be a Kaggle Models handle <owner>/<model>/<framework>/<variation>/"
        f"<version> when model_dir is null; got {model_id!r}",
    )
    return tuple(part.lower() for part in parts[1:])


def _validate(values: dict[str, Any]) -> None:
    """Check every resolved value's type and range; raises ``ConfigError``."""
    for key in ("run_id", "model_id", "input_root", "challenges", "device", "cache_dir", "out_dir"):
        _check(isinstance(values[key], str) and values[key] != "", f"{key} must be a string")
    model_dir = values["model_dir"]
    _check(
        model_dir is None or (isinstance(model_dir, str) and model_dir != ""),
        "model_dir must be null or a path",
    )
    _check(".." not in Path(values["challenges"]).parts, "challenges must not contain '..'")
    subset = values["subset"]
    _check(subset is None or subset in SUBSETS, f"subset must be null or one of {list(SUBSETS)}")
    _check(values["dtype"] in DTYPES, f"dtype must be one of {sorted(DTYPES)}")
    _check(values["attn_implementation"] in ATTENTION, f"attn_implementation in {ATTENTION}")
    _check(values["pool"] in ("output", "input"), "pool must be 'output' or 'input'")
    _check(isinstance(values["store_views"], bool), "store_views must be true or false")
    for key in ("layer", "n_perm", "view_seed", "throughput_k", "smoke_seed"):
        _check(_is_int(values[key]) and values[key] >= 0, f"{key} must be an integer >= 0")
    for key in ("batch_tokens", "flush_pairs", "smoke_tasks"):
        _check(_is_int(values[key]) and values[key] >= 1, f"{key} must be an integer >= 1")
    tasks = values["throughput_tasks"]
    _check(tasks is None or (_is_int(tasks) and tasks >= 1), "throughput_tasks: null or >= 1")
    # The run's views are the first n_perm rounds of the reference set only when both use the same
    # seed and the run has at least one round of permutations (n_perm 0 views carry none).
    reference = values["reference_n_perm"]
    _check(
        reference is None or (_is_int(reference) and reference > values["n_perm"] >= 1),
        "reference_n_perm must be null, or an integer above n_perm with n_perm >= 1",
    )
    for key in ("acceptance_s", "max_wall_s"):
        number = values[key]
        _check(_is_int(number) or isinstance(number, float), f"{key} must be a number")
        _check(number > 0, f"{key} must be positive")


def load_config(path: Path, smoke: bool, overrides: dict[str, str | None]) -> Config:
    """Read the YAML config, apply the smoke section and command-line overrides, and validate.

    The file's own values and its smoke values are both validated in either mode, so a smoke run
    also checks the GPU run's settings, its model handle included. ``task_ids`` is resolved from
    ``subset`` through ``arc26.data``. A config with an ``episodes`` section is in episode mode
    (``load_episode_config``); every other config keeps P3.2's exact keys.

    Args:
        path: The YAML file.
        smoke: Whether to apply the ``smoke`` section.
        overrides: Values from the command line (``None`` means not given); keys are config keys,
            plus ``episodes_root`` and ``check_root`` (episode mode only).

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
    _check(isinstance(raw, dict), f"{path} must hold a mapping")
    if "episodes" in raw:
        return load_episode_config(raw, smoke, overrides)
    episode_only = [
        key for key in ("episodes_root", "check_root", "seed_cache") if overrides.get(key)
    ]
    _check(not episode_only, f"{episode_only} apply to episode mode only (an episodes section)")
    overrides = {key: value for key, value in overrides.items() if key not in episode_only}
    _check(set(raw) == set(TOP_KEYS), f"keys differ from {sorted(TOP_KEYS)}: {sorted(raw)}")
    smoke_section = raw.pop("smoke")
    _check(isinstance(smoke_section, dict), "smoke must be a mapping")
    allowed = set(SMOKE_OVERRIDES) | set(SMOKE_ONLY)
    _check(set(smoke_section) == allowed, f"smoke keys must be exactly {sorted(allowed)}")
    base = {**raw, "smoke_tasks": smoke_section["tasks"], "smoke_seed": smoke_section["seed"]}
    smoked = {**base, **{key: smoke_section[key] for key in SMOKE_OVERRIDES}}
    _validate(base)
    _validate(smoked)
    if base["model_dir"] is None:
        model_folder_parts(base["model_id"])
    values = dict(smoked if smoke else base)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    _validate(values)
    if not smoke and values["model_dir"] is None:
        model_folder_parts(values["model_id"])
    values["task_ids"] = None if values["subset"] is None else subset_ids(values["subset"])
    values["acceptance_s"] = float(values["acceptance_s"])
    values["max_wall_s"] = float(values["max_wall_s"])
    return Config(**values)


def smoke_tokenizer() -> PreTrainedTokenizerFast:
    """Build the stand-in tokenizer for the smoke path and the tests; nothing is downloaded.

    A word-level vocabulary with the notebook's 16 ids (``SMOKE_VOCAB``): the chat markers are
    special tokens, the rest of the text splits into single digits, newlines, and the role words,
    and newlines are byte-level encoded as ``Ċ``, as in the real vocabulary. Text outside the
    format (spaces, letters other than the role words) raises.

    Returns:
        A fast tokenizer with ``<|endoftext|>`` as padding and ``<|im_end|>`` as end of sequence.
    """
    tok = Tokenizer(models.WordLevel(vocab=SMOKE_VOCAB, unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\d|\n|[a-z]+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )


def smoke_model(
    seed: int = 0, attn_implementation: str = "eager", layers: int = 4
) -> PreTrainedModel:
    """Build the tiny random Qwen3 of the smoke path and the tests, in eval mode, on the CPU.

    Hidden size 64, 4 layers, 4 attention heads, 2 key-value heads, head size 16, vocabulary 16
    (the task brief's stand-in). The global torch generator is left as it was. The tests save a
    36-layer copy as a stand-in base model folder, so the GPU run's settings (``layer`` 24) run
    on the CPU.

    Args:
        seed: Seed of the random weights.
        attn_implementation: Attention kernel name.
        layers: Decoder layers.

    Returns:
        The model, fp32, gradients off.
    """
    config = Qwen3Config(
        vocab_size=len(SMOKE_VOCAB),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        attn_implementation=attn_implementation,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config)
    return model.eval().requires_grad_(False)


def smoke_challenges(n_tasks: int, seed: int) -> dict[str, Any]:
    """Build synthetic ARC challenges for the smoke path.

    Task ``t`` has ``2 + t % 3`` pairs of random grids with sides 1 to 8; with three or more
    tasks, task 1 repeats task 0's first pair (the cache stores it once), and the last task's
    first pair is 30 by 30, the longest pair the format allows.

    Args:
        n_tasks: Number of tasks.
        seed: Seed of the grids.

    Returns:
        ``{task_id: {"train": [pair, ...], "test": [{"input": grid}]}}``, ids 8 hex digits.
    """
    rng = np.random.default_rng(seed)

    def grid(rows: int, cols: int) -> list[list[int]]:
        return rng.integers(0, 10, size=(rows, cols)).tolist()

    def sides() -> tuple[int, int]:
        return int(rng.integers(1, 9)), int(rng.integers(1, 9))

    tasks: dict[str, Any] = {}
    for t in range(n_tasks):
        train = [{"input": grid(*sides()), "output": grid(*sides())} for _ in range(2 + t % 3)]
        if t == n_tasks - 1:
            train[0] = {"input": grid(30, 30), "output": grid(30, 30)}
        tasks[f"{t:08x}"] = {"train": train, "test": [{"input": grid(*sides())}]}
    if n_tasks >= 3:
        first = tasks[f"{0:08x}"]["train"][0]
        tasks[f"{1:08x}"]["train"][0] = {"input": first["input"], "output": first["output"]}
    return tasks


def _searched(name: str) -> bool:
    """Whether the input search enters a folder of this name: package trees are skipped."""
    return name not in SKIPPED_DIRS and not name.endswith((".dist-info", ".egg-info"))


def find_files(
    root: Path, endings: dict[str, Sequence[str]], max_depth: int = SEARCH_DEPTH
) -> dict[str, list[Path]]:
    """Find the files under a root whose paths end with given parts.

    Why: Kaggle mounts every attached input under one root, and where exactly changed from 2025
    to 2026 and within 2026 (``docs/STATE.md``, rules for scripts), so inputs are found by the end
    of their path instead of a hardcoded one. The walk is the script notebook's (``find_inputs``
    in ``scripts/build_kaggle_notebook.py``, which cannot be imported here: only this file and its
    config are shipped): bounded, package trees skipped, links followed into each real folder once.

    Args:
        root: Folder searched; a missing one finds nothing.
        endings: For each key, the last parts a path must have, compared case-insensitively.
        max_depth: Deepest file level searched; a file directly in ``root`` is at level 1.

    Returns:
        For each key, the matching paths in walk order (folders and files sorted by name).
    """
    wanted = {key: tuple(part.lower() for part in parts) for key, parts in endings.items()}
    found: dict[str, list[Path]] = {key: [] for key in endings}
    top = Path(os.path.abspath(root))
    seen: set[str] = set()
    for folder, dirs, files in os.walk(top, followlinks=True):
        real = os.path.realpath(folder)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        level = len(Path(folder).relative_to(top).parts)
        dirs[:] = sorted(filter(_searched, dirs)) if level + 1 < max_depth else []
        for name in sorted(files):
            path = Path(folder) / name
            parts = tuple(part.lower() for part in path.parts)
            for key, tail in wanted.items():
                if parts[-len(tail) :] == tail:
                    found[key].append(path)
    return found


def _listed(paths: Sequence[Path]) -> str:
    """At most ``SHOWN_PATHS`` paths for an error message, with the count of the rest."""
    shown = ", ".join(str(p) for p in paths[:SHOWN_PATHS])
    more = len(paths) - SHOWN_PATHS
    return f"[{shown}{f', and {more} more' if more > 0 else ''}]"


def find_inputs(cfg: Config) -> tuple[Path, Path | None]:
    """Return the base model folder and the challenges file of a GPU run.

    Explicit paths are used as given: ``model_dir`` when set, ``challenges`` when absolute.
    Otherwise each is searched under ``input_root`` (``find_files``) and must be found exactly
    once: the model folder holds ``config.json`` and its path ends with ``model_id``'s model,
    framework, variation, and version (``model_folder_parts``); the challenges file's path ends
    with the relative ``challenges``. A second match (two mounts of different inputs) is refused
    rather than guessed between. Episode mode has no challenges file (None).

    Args:
        cfg: The run config.

    Returns:
        ``(model_dir, challenges)``.

    Raises:
        InputSearchError: If a searched input is found zero times or more than once; the
            message lists what was found.
    """
    endings: dict[str, Sequence[str]] = {}
    if cfg.model_dir is None:
        endings["model"] = (*model_folder_parts(cfg.model_id), MODEL_CONFIG)
        endings["any model"] = (MODEL_CONFIG,)
    challenges = None if cfg.challenges is None else Path(cfg.challenges)
    if challenges is not None and not challenges.is_absolute():
        endings["challenges"] = challenges.parts
        endings["any challenges"] = (challenges.name,)
    found = find_files(Path(cfg.input_root), endings) if endings else {}

    def one(key: str, what: str) -> Path:
        paths = found[key]
        if len(paths) == 1:
            return paths[0]
        others = [p for p in found[f"any {key}"] if p not in paths]
        raise InputSearchError(
            f"found {len(paths)} {what} under {cfg.input_root}, expected 1: {_listed(paths)}; "
            f"other files of that name: {_listed(others)}"
        )

    if cfg.model_dir is not None:
        model_dir = Path(cfg.model_dir)
    else:
        ending = "/".join(endings["model"][:-1])
        model_dir = one("model", f"model folders ending in {ending} with {MODEL_CONFIG}").parent
    if challenges is not None and not challenges.is_absolute():
        challenges = one("challenges", f"files ending in {cfg.challenges}")
    return model_dir, challenges


class Task(NamedTuple):
    """One task's demonstration pairs.

    Attributes:
        task_id: The challenges file's key.
        pairs: ``{"input": grid, "output": grid}`` per demonstration, in file order.
    """

    task_id: str
    pairs: list[dict[str, Any]]


def load_tasks(path: Path, task_ids: Sequence[str] | None) -> list[Task]:
    """Read the demonstration pairs of an ARC challenges file; nothing else is kept.

    Args:
        path: The challenges JSON.
        task_ids: Tasks in the order wanted; None takes every task in sorted id order.

    Returns:
        The tasks.

    Raises:
        ValueError: If the file name contains ``solution`` (BRIEF A0: solutions are never read
            here), a task id is absent, or a task has no demonstration pairs.
    """
    if SOLUTION_MARKER in path.name.lower():
        raise ValueError(f"refusing to read {path.name}: only challenges files are read")
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = sorted(data) if task_ids is None else list(task_ids)
    absent = [t for t in ids if t not in data]
    if absent:
        raise ValueError(f"{len(absent)} task ids are not in {path.name}: {absent[:5]}")
    tasks = []
    for task_id in ids:
        pairs = [{"input": p["input"], "output": p["output"]} for p in data[task_id]["train"]]
        if not pairs:
            raise ValueError(f"task {task_id} has no demonstration pairs")
        tasks.append(Task(task_id, pairs))
    return tasks


def preflight(tasks: Sequence[Task], formatter: QwenFormatter, views: Sequence[str]) -> list[int]:
    """Render every view of every pair on the CPU before any GPU work.

    Why: the real tokenizer exists only on Kaggle, so this is where its handling of the format is
    first checked (``render_pair``'s prefix and decode checks). A failure stops the run before
    the model loads.

    Args:
        tasks: The run's tasks.
        formatter: The notebook formatter around the run's tokenizer.
        views: The encoder's view descriptors.

    Returns:
        The token count of every rendered view, in task, pair, view order.

    Raises:
        ValueError: If a view fails ``render_pair``; the message names the task and the view.
    """
    lengths = []
    for task in tasks:
        for pair in task.pairs:
            for view in views:
                try:
                    lengths.append(len(render_pair(apply_view(pair, view), formatter).token_ids))
                except ValueError as err:
                    raise ValueError(f"task {task.task_id}, view {view!r}: {err}") from err
    return lengths


class TimingLog:
    """The encoder's timing callback: appends each record to ``timing.jsonl``.

    Every line is the B4 record ``{puzzle_id, op, seconds, tokens}`` plus ``phase`` (``warmup``,
    ``throughput``, or ``cache``). ``take`` returns the records since its last call.

    Args:
        path: The JSONL file; appended to, so a resumed run keeps earlier records.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.phase = "setup"
        self._recent: list[dict[str, Any]] = []

    def __call__(self, record: dict[str, Any]) -> None:
        """Append one record with the current phase, and keep it for ``take``."""
        append_jsonl(self.path, {**record, "phase": self.phase})
        self._recent.append(record)

    def take(self) -> list[dict[str, Any]]:
        """Return and forget the records received since the previous call."""
        records, self._recent = self._recent, []
        return records


@dataclass
class RunState:
    """What a run has learned so far; ``build_metrics`` reads it even after a stop or failure.

    Attributes:
        tasks: The loaded tasks.
        lengths: Token count of every rendered view.
        model: Model facts for ``metrics.json``.
        cache: The open cache, once created.
        device_name: GPU name, or ``cpu``.
        encoded_pairs: Pairs encoded by the cache phase in this session.
        encode_seconds: Seconds of those encode calls.
        encode_tokens: Real tokens of those encode calls.
        measured_tasks: Tasks the throughput phase measured in this session.
        peak_memory: CUDA peak memory statistics in bytes.
        diagnostic: The view diagnostic's summary, once it has run.
    """

    tasks: list[Task] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    cache: EmbeddingCache | None = None
    device_name: str = "cpu"
    encoded_pairs: int = 0
    encode_seconds: float = 0.0
    encode_tokens: int = 0
    measured_tasks: int = 0
    peak_memory: dict[str, int] = field(default_factory=dict)
    diagnostic: dict[str, Any] | None = None


def _check_deadline(deadline: float) -> None:
    """Raise ``WallTimeExceeded`` once the monotonic clock passes ``deadline``."""
    if time.monotonic() > deadline:
        raise WallTimeExceeded("wall time over max_wall_s")


def _require_log(encoder: PairEncoder, log: TimingLog) -> None:
    """Refuse a ``log`` that is not the encoder's callback: the phases read their times from it."""
    if encoder.timing is not log:
        raise ValueError("log must be the encoder's timing callback")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line, first ending a last line that a killed session cut short.

    Args:
        path: The JSONL file; created if missing.
        record: The record to append.
    """
    with path.open("a+b") as f:
        if f.seek(0, os.SEEK_END):
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write(json.dumps(record).encode("utf-8") + b"\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file written by this script; a missing file is empty.

    A line cut short by a killed session is skipped, so a rerun measures that task again.

    Args:
        path: The file.

    Returns:
        One dict per complete line.
    """
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def measure_throughput(
    encoder: PairEncoder,
    tasks: Sequence[Task],
    cfg: Config,
    ledger: Path,
    log: TimingLog,
    state: RunState,
    deadline: float,
) -> None:
    """Time one fresh encode of each task's first k pairs; one ``throughput.jsonl`` line each.

    Tasks with a record in ``ledger`` are skipped, so a rerun measures only the rest. Each
    session first runs one untimed warm-up encode (CUDA context, kernel selection). Pairs are
    repeated in order when a task has fewer than k. The cache is neither read nor written.

    Args:
        encoder: The pair encoder.
        tasks: The run's tasks; the first ``cfg.throughput_tasks`` are measured.
        cfg: The run config.
        ledger: ``throughput.jsonl``.
        log: The encoder's timing callback (``encoder.timing``), which supplies the stage times.
        state: Run state; ``measured_tasks`` is updated.
        deadline: Monotonic time of the wall-time kill criterion.

    Raises:
        ValueError: If ``log`` is not ``encoder.timing``.
        WallTimeExceeded: Between tasks, once ``deadline`` passes.
    """
    _require_log(encoder, log)
    k = cfg.throughput_k
    selected = list(tasks if cfg.throughput_tasks is None else tasks[: cfg.throughput_tasks])
    done = {record["task_id"] for record in read_jsonl(ledger)}
    todo = [task for task in selected if task.task_id not in done]
    if not todo:
        return
    log.phase = "warmup"
    first = todo[0]
    encoder.encode_views([first.pairs[i % len(first.pairs)] for i in range(k)], first.task_id)
    log.take()
    log.phase = "throughput"
    for task in todo:
        _check_deadline(deadline)
        sample = [task.pairs[i % len(task.pairs)] for i in range(k)]
        start = time.perf_counter()
        encoder.encode_views(sample, puzzle_id=task.task_id)
        seconds = time.perf_counter() - start
        stages = {record["op"]: record for record in log.take()}
        record = {
            "task_id": task.task_id,
            "k": k,
            "n_pairs": len(task.pairs),
            "cycled": len(task.pairs) < k,
            "sequences": k * len(encoder.views),
            "tokens": stages["encode_prefill"]["tokens"],
            "seconds": seconds,
            "render_seconds": stages["encode_render"]["seconds"],
            "prefill_seconds": stages["encode_prefill"]["seconds"],
            "device": state.device_name,
        }
        append_jsonl(ledger, record)
        state.measured_tasks += 1


def fill_cache(
    encoder: PairEncoder,
    cache: EmbeddingCache,
    tasks: Sequence[Task],
    cfg: Config,
    log: TimingLog,
    state: RunState,
    deadline: float,
) -> None:
    """Encode every pair the cache lacks, task by task, writing every ``flush_pairs`` pairs.

    Args:
        encoder: The pair encoder.
        cache: The open cache of the encoder's namespace.
        tasks: The run's tasks.
        cfg: The run config.
        log: The encoder's timing callback (``encoder.timing``), which supplies the token counts.
        state: Run state; the session counters are updated.
        deadline: Monotonic time of the wall-time kill criterion.

    Raises:
        ValueError: If ``log`` is not ``encoder.timing``.
        WallTimeExceeded: Between tasks, once ``deadline`` passes, after the pending pairs are
            written.
    """
    _require_log(encoder, log)
    log.phase = "cache"
    means: dict[str, torch.Tensor] = {}
    views: dict[str, torch.Tensor] = {}

    def flush() -> None:
        if means:
            cache.put(means, views if cfg.store_views else None)
            means.clear()
            views.clear()

    try:
        for task in tasks:
            _check_deadline(deadline)
            todo: dict[str, dict[str, Any]] = {}
            for pair in task.pairs:
                key = pair_hash(pair)
                if key not in cache and key not in means:
                    todo.setdefault(key, pair)
            if todo:
                start = time.perf_counter()
                per_view = encoder.encode_views(list(todo.values()), puzzle_id=task.task_id)
                state.encode_seconds += time.perf_counter() - start
                prefill = [r for r in log.take() if r["op"] == "encode_prefill"]
                state.encode_tokens += sum(r["tokens"] for r in prefill)
                state.encoded_pairs += len(todo)
                for key, embedding in zip(todo, per_view, strict=True):
                    means[key] = embedding.mean(dim=0)
                    if cfg.store_views:
                        views[key] = embedding
            if len(means) >= cfg.flush_pairs:
                flush()
    except WallTimeExceeded:
        flush()
        raise
    flush()


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of two ``[n, hidden]`` arrays; 0 where a row has no length."""
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    dots = np.einsum("ij,ij->i", a, b)
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def _spread(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Median, 10th percentile, and minimum of a non-empty sequence."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "min": float(arr.min()),
    }


def diagnose_views(
    complement: PairEncoder,
    cache: EmbeddingCache,
    tasks: Sequence[Task],
    reference_views: int,
    log: TimingLog,
    state: RunState,
    deadline: float,
    out_path: Path,
) -> None:
    r"""Measure how far each pair's embedding moves from its embedding under the reference views.

    Why: fewer views make the encoder cheaper but change $\bar e$ (ADR-001), and H must be trained
    and used with one view set, so the human weighs the run's time against this diagnostic; the
    script never acts on it. The run's views $R$ are the first of the reference set $V$ (the same
    seed, drawn round by round), so with $C = V \setminus R$,
    $\bar e_V(i) = (|R|\,\bar e_R(i) + \sum_{c \in C} e_c(i)) / |V|$: $\bar e_R(i)$ is the run's
    cached embedding, and only $C$ is encoded here.

    Reported: the cosine of $\bar e_R(i)$ with $\bar e_V(i)$ over the run's distinct pairs; the
    same after subtracting the mean of $\bar e_V$ over those pairs, because mean-pooled hidden
    states share a large common direction that lifts every raw cosine; the cosine of the task
    embeddings $\bar z$ (each task's pairs as listed); and, as the scale for the rest, the median
    cosine between different pairs' $\bar e_V$ (over the first ``BETWEEN_PAIRS_CAP`` pairs). The
    per-pair values go to ``out_path``, the summary to ``state.diagnostic``.

    Args:
        complement: An encoder whose views are $C$, on the run's model, with the run's timing
            callback.
        cache: The run's cache, holding every pair of ``tasks`` (the cache phase ran).
        tasks: The run's tasks.
        reference_views: $|V|$.
        log: The encoder's timing callback (``complement.timing``).
        state: Run state; ``diagnostic`` is set at the end.
        deadline: Monotonic time of the wall-time kill criterion.
        out_path: The per-pair JSONL file, rewritten.

    Raises:
        ValueError: If ``log`` is not ``complement.timing``.
        WallTimeExceeded: Between tasks, once ``deadline`` passes; nothing is reported then.
    """
    _require_log(complement, log)
    log.phase = "diagnostic"
    log.take()
    run_views = reference_views - len(complement.views)
    first_task: dict[str, str] = {}
    reference: dict[str, np.ndarray] = {}
    tokens = 0
    start = time.perf_counter()
    for task in tasks:
        _check_deadline(deadline)
        todo: dict[str, dict[str, Any]] = {}
        for pair in task.pairs:
            key = pair_hash(pair)
            first_task.setdefault(key, task.task_id)
            if key not in reference:
                todo.setdefault(key, pair)
        if todo:
            rows = complement.encode_views(list(todo.values()), puzzle_id=task.task_id)
            tokens += sum(r["tokens"] for r in log.take() if r["op"] == "encode_prefill")
            means = cache.get(list(todo))
            for key, mean, extra in zip(todo, means, rows, strict=True):
                total = run_views * mean.double() + extra.double().sum(dim=0)
                reference[key] = (total / reference_views).numpy()
    seconds = time.perf_counter() - start
    keys = list(reference)
    index = {key: i for i, key in enumerate(keys)}
    run = cache.get(keys).double().numpy()
    ref = np.stack([reference[key] for key in keys])
    pair_cos = _cosines(run, ref)
    center = ref.mean(axis=0, keepdims=True)
    centered = _cosines(run - center, ref - center) if len(keys) > 1 else None
    task_cos = []
    for task in tasks:
        rows_of = [index[pair_hash(pair)] for pair in task.pairs]
        z_run = run[rows_of].mean(axis=0, keepdims=True)
        z_ref = ref[rows_of].mean(axis=0, keepdims=True)
        task_cos.append(float(_cosines(z_run, z_ref)[0]))
    sample = ref[:BETWEEN_PAIRS_CAP]
    lengths = np.linalg.norm(sample, axis=1, keepdims=True)
    unit = np.divide(sample, lengths, out=np.zeros_like(sample), where=lengths > 0)
    between = (unit @ unit.T)[np.triu_indices(len(sample), k=1)]
    with out_path.open("w", encoding="utf-8") as f:
        for i, key in enumerate(keys):
            record = {
                "pair": key,
                "task_id": first_task[key],
                "cosine": float(pair_cos[i]),
                "centered_cosine": None if centered is None else float(centered[i]),
            }
            f.write(json.dumps(record) + "\n")
    state.diagnostic = {
        "views": run_views,
        "reference_views": reference_views,
        "pairs": len(keys),
        "tasks": len(tasks),
        "pair_cosine": _spread(pair_cos),
        "pair_cosine_centered": None if centered is None else _spread(centered),
        "task_cosine": _spread(task_cos),
        "between_pairs_cosine_median": float(np.median(between)) if between.size else None,
        "between_pairs_sample": len(sample),
        "seconds": seconds,
        "tokens": tokens,
    }


def git_commit(repo: Path) -> tuple[str | None, bool | None]:
    """Return the checkout's HEAD commit and whether its working tree has changes.

    Args:
        repo: A directory inside the checkout.

    Returns:
        ``(sha, dirty)``, or ``(None, None)`` outside a git checkout (as on Kaggle, where the
        notebook passes ``--commit``).
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


def _stats(values: Sequence[float]) -> dict[str, float]:
    """Median, p90, max, and mean of a non-empty sequence."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def summarize_throughput(records: Sequence[dict[str, Any]], acceptance_s: float) -> dict[str, Any]:
    """Summarize the throughput ledger.

    Args:
        records: ``throughput.jsonl`` lines.
        acceptance_s: The acceptance bound in seconds.

    Returns:
        Task counts, seconds and tokens statistics, the rate per 1,000 tokens, the slowest task,
        the render share of the time, and the count of tasks above ``acceptance_s``; only the
        counts when there are no records.
    """
    summary: dict[str, Any] = {"tasks": len(records), "acceptance_s": acceptance_s}
    if not records:
        return summary
    seconds = [r["seconds"] for r in records]
    tokens = [r["tokens"] for r in records]
    slowest = max(records, key=lambda r: r["seconds"])
    summary.update(
        {
            "k": records[0]["k"],
            "views": records[0]["sequences"] // records[0]["k"],
            "cycled": sum(r["cycled"] for r in records),
            "seconds": _stats(seconds),
            "tokens": _stats(tokens),
            "seconds_per_1k_tokens": 1000 * sum(seconds) / sum(tokens),
            "render_share": sum(r["render_seconds"] for r in records) / sum(seconds),
            "slowest": {k: slowest[k] for k in ("task_id", "seconds", "tokens", "n_pairs")},
            "over_acceptance": sum(s > acceptance_s for s in seconds),
            "devices": sorted({r["device"] for r in records}),
        }
    )
    return summary


def build_metrics(
    cfg: Config, state: RunState, status: str, elapsed: float, ledger: Path
) -> dict[str, Any]:
    """Collect ``metrics.json`` from the run state and the throughput ledger.

    Args:
        cfg: The run config.
        state: What the run learned, possibly partial.
        status: ``complete``, ``stopped: ...``, or ``failed: ...``.
        elapsed: Wall seconds since the start.
        ledger: ``throughput.jsonl``.

    Returns:
        The metrics; sections the run did not reach are absent.
    """
    metrics: dict[str, Any] = {"run_id": cfg.run_id, "status": status, "elapsed_s": elapsed}
    if state.tasks:
        hashes = [[pair_hash(p) for p in task.pairs] for task in state.tasks]
        metrics["tasks"] = {
            "count": len(state.tasks),
            "pairs": sum(len(h) for h in hashes),
            "unique_pairs": len({h for task in hashes for h in task}),
        }
    if state.lengths:
        lengths = np.asarray(state.lengths)
        metrics["lengths"] = {
            "sequences": int(lengths.size),
            **{f"p{q}": float(np.percentile(lengths, q)) for q in (50, 90, 99)},
            "max": int(lengths.max()),
        }
    if state.model:
        metrics["model"] = state.model
    metrics["throughput"] = {
        **summarize_throughput(read_jsonl(ledger), cfg.acceptance_s),
        "measured_this_session": state.measured_tasks,
    }
    cache = state.cache
    if cache is not None:
        entries = len(cache)
        per_entry = cache.size_bytes / entries if entries else None
        task_pairs = [len(set(map(pair_hash, task.pairs))) for task in state.tasks]
        run_pairs = {pair_hash(p) for task in state.tasks for p in task.pairs}
        metrics["cache"] = {
            "dir": str(cache.path),
            "entries": entries,
            "run_pairs_cached": len(run_pairs) - len(cache.missing(run_pairs)),
            "encoded_this_session": state.encoded_pairs,
            "encode_seconds": state.encode_seconds,
            "pairs_per_second": (
                state.encoded_pairs / state.encode_seconds if state.encode_seconds else None
            ),
            "tokens_per_second": (
                state.encode_tokens / state.encode_seconds if state.encode_seconds else None
            ),
            "bytes": cache.size_bytes,
            "bytes_per_pair": per_entry,
            "payload_bytes_per_pair": (
                4 * cache.hidden_size * (1 + (cache.n_views or 0)) if cache.hidden_size else None
            ),
            "bytes_per_task": (
                _stats([n * per_entry for n in task_pairs]) if per_entry and task_pairs else None
            ),
        }
    if state.peak_memory:
        metrics["memory"] = state.peak_memory
    if state.diagnostic is not None:
        metrics["view_diagnostic"] = state.diagnostic
    return metrics


def _fmt(value: Any, digits: int = 2) -> str:
    """A number for summary.md: fixed decimals for floats, ``n/a`` for None, as is otherwise."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_summary(cfg: Config, metrics: dict[str, Any], provenance: dict[str, Any]) -> str:
    """Write the short human-readable report of a run.

    Args:
        cfg: The run config.
        metrics: ``metrics.json``.
        provenance: ``run.json`` (may be partial).

    Returns:
        Markdown text.
    """
    lines = [
        f"# {cfg.run_id}: pair-encoder throughput and embedding cache",
        "",
        f"- Status: {metrics['status']}; wall {_fmt(metrics['elapsed_s'], 1)} s.",
        f"- Commit: {provenance.get('commit')} (dirty: {provenance.get('dirty')}).",
        f"- Device: {provenance.get('device_name')}; torch "
        f"{provenance.get('versions', {}).get('torch')}, transformers "
        f"{provenance.get('versions', {}).get('transformers')}; {cfg.dtype}, "
        f"{cfg.attn_implementation}.",
        f"- Encoder: {cfg.model_id}, layer {cfg.layer}, pool {cfg.pool}, {8 * max(cfg.n_perm, 1)} "
        f"views (seed {cfg.view_seed}), batch_tokens {cfg.batch_tokens}.",
    ]
    inputs = provenance.get("inputs")
    if inputs:
        lines.append(
            f"- Inputs: model `{inputs['model_dir']}`; challenges `{inputs['challenges']}`."
        )
    if "tasks" in metrics:
        t = metrics["tasks"]
        subset = cfg.subset or "every task in the file"
        lines.append(
            f"- Tasks: {t['count']} ({subset}), pairs {t['pairs']} ({t['unique_pairs']} distinct)."
        )
    if "lengths" in metrics:
        n = metrics["lengths"]
        lines.append(
            f"- Tokens per view: p50 {_fmt(n['p50'], 0)}, p90 {_fmt(n['p90'], 0)}, "
            f"p99 {_fmt(n['p99'], 0)}, max {n['max']} ({n['sequences']} views)."
        )
    if "model" in metrics:
        lines.append(f"- Model load: {_fmt(metrics['model'].get('load_seconds'), 1)} s.")
    tp = metrics["throughput"]
    lines += ["", f"## Encode time for k = {cfg.throughput_k} pairs, fresh (no cache)", ""]
    if tp.get("seconds"):
        s = tp["seconds"]
        lines += [
            "| Tasks | Repeated pairs | Median s | p90 s | Max s | s per 1k tokens | Over "
            f"{_fmt(tp['acceptance_s'], 0)} s |",
            "|---|---|---|---|---|---|---|",
            f"| {tp['tasks']} | {tp['cycled']} | {_fmt(s['median'])} | {_fmt(s['p90'])} | "
            f"{_fmt(s['max'])} | {_fmt(tp['seconds_per_1k_tokens'], 3)} | "
            f"{tp['over_acceptance']} |",
            "",
            f"Slowest task: {tp['slowest']['task_id']} ({tp['slowest']['n_pairs']} pairs, "
            f"{tp['slowest']['tokens']} tokens, {_fmt(tp['slowest']['seconds'])} s). "
            f"Rendering is {_fmt(100 * tp['render_share'], 1)}% of the time.",
        ]
    else:
        lines.append("Not measured.")
    lines += ["", "## Cache", ""]
    c = metrics.get("cache")
    if c:
        per_task = c["bytes_per_task"]
        lines += [
            f"- Directory: `{c['dir']}`; {c['entries']} entries, {c['bytes']} bytes.",
            f"- This session: {c['encoded_this_session']} pairs in "
            f"{_fmt(c['encode_seconds'], 1)} s ({_fmt(c['pairs_per_second'])} pairs per second).",
            f"- Per pair: {_fmt(c['bytes_per_pair'], 0)} bytes on disk "
            f"({c['payload_bytes_per_pair']} of tensor data).",
        ]
        if per_task:
            lines.append(
                f"- Per task: mean {_fmt(per_task['mean'], 0)} bytes, max "
                f"{_fmt(per_task['max'], 0)}."
            )
    else:
        lines.append("Not reached.")
    if "memory" in metrics:
        lines += ["", f"Peak GPU memory allocated: {metrics['memory']['max_allocated']} bytes."]
    d = metrics.get("view_diagnostic")
    if d:

        def spread(name: str) -> str:
            v = d[name]
            return f"median {v['median']:.4f}, p10 {v['p10']:.4f}, min {v['min']:.4f}"

        centered = (
            f"centered on the mean pair: {spread('pair_cosine_centered')}"
            if d["pair_cosine_centered"]
            else "centered: n/a (one pair)"
        )
        lines += [
            "",
            f"## View diagnostic: {d['views']} views against {d['reference_views']} "
            "(reported, decides nothing)",
            "",
            f"- Pairs {d['pairs']}, cosine of the {d['views']}-view and the "
            f"{d['reference_views']}-view embedding: {spread('pair_cosine')}; {centered}.",
            f"- Tasks {d['tasks']}, cosine of z_bar: {spread('task_cosine')}.",
            f"- Scale: different pairs' {d['reference_views']}-view embeddings have a median "
            f"cosine of {_fmt(d['between_pairs_cosine_median'], 4)} "
            f"({d['between_pairs_sample']} pairs).",
            f"- The other {d['reference_views'] - d['views']} views of {d['pairs']} pairs encoded "
            f"in {_fmt(d['seconds'], 1)} s ({d['tokens']} tokens); per pair: {DIAGNOSTIC_FILE}.",
        ]
    if provenance.get("smoke"):
        reading = "Smoke run: a random-weight stand-in on the CPU checks the code path; its times "
        reading += "say nothing about an L4."
    else:
        reading = f"Decision rule: docs/prerun/{cfg.run_id}.md, section 1."
    lines += ["", f"{reading} Logs: timing.jsonl, throughput.jsonl, metrics.json.", ""]
    return "\n".join(lines)


def _device_name(device: torch.device) -> str:
    """The GPU's name, or ``cpu``."""
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def run(
    cfg: Config,
    smoke: bool,
    out_dir: Path,
    state: RunState,
    provenance: dict[str, Any],
    deadline: float,
) -> None:
    """Find the inputs, then run the preflight, the throughput phase, and the cache phase.

    Args:
        cfg: The run config.
        smoke: Whether to use the stand-in model, tokenizer, and synthetic tasks.
        out_dir: The run-log directory.
        state: Filled as the run proceeds.
        provenance: ``run.json``, completed and written once the model is loaded.
        deadline: Monotonic time of the wall-time kill criterion.

    Raises:
        RuntimeError: If CUDA is requested but absent.
        InputSearchError: If the model folder or the challenges file is not found exactly once.
        WallTimeExceeded: When the wall-time kill criterion fires.
    """
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {cfg.device} requested but CUDA is not available")
    state.device_name = provenance["device_name"] = _device_name(device)
    if smoke:
        model_dir, challenges = str(cfg.model_dir), Path(cfg.challenges)
    else:
        found_dir, challenges = find_inputs(cfg)
        model_dir = str(found_dir)
        print(f"model folder: {model_dir}\nchallenges: {challenges}", flush=True)
    provenance["inputs"] = {"model_dir": model_dir, "challenges": str(challenges)}
    if smoke:
        tokenizer: PreTrainedTokenizerBase = smoke_tokenizer()
        challenges.write_text(json.dumps(smoke_challenges(cfg.smoke_tasks, cfg.smoke_seed)))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    state.tasks = load_tasks(challenges, cfg.task_ids)
    views = view_descriptors(cfg.n_perm, cfg.view_seed)
    state.lengths = preflight(state.tasks, QwenFormatter(tokenizer), views)
    _check_deadline(deadline)

    start = time.perf_counter()
    if smoke:
        model = smoke_model(cfg.smoke_seed, cfg.attn_implementation).to(DTYPES[cfg.dtype])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=DTYPES[cfg.dtype],
            attn_implementation=cfg.attn_implementation,
            local_files_only=True,
        )
    model = model.to(device).eval().requires_grad_(False)
    state.model = {
        "load_seconds": time.perf_counter() - start,
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
    }
    log = TimingLog(out_dir / "timing.jsonl")
    encoder = PairEncoder(
        model,
        tokenizer,
        layer=cfg.layer,
        pool=cfg.pool,
        views=views,
        batch_tokens=cfg.batch_tokens,
        timing=log,
    )
    state.cache = EmbeddingCache(
        cfg.cache_dir,
        model_id=cfg.model_id,
        layer=cfg.layer,
        pool=cfg.pool,
        view_set_id=encoder.view_set_id,
    )
    provenance.update(
        {
            "challenges": str(challenges),
            "model": {**state.model, "model_id": cfg.model_id, "model_dir": model_dir},
            "encoder": {
                "layer": cfg.layer,
                "pool": cfg.pool,
                "views": encoder.views,
                "view_set_id": encoder.view_set_id,
                "batch_tokens": cfg.batch_tokens,
            },
            "cache": {"dir": str(state.cache.path), "entries_at_start": len(state.cache)},
        }
    )
    complement = None
    if cfg.reference_n_perm is not None:
        reference = view_descriptors(cfg.reference_n_perm, cfg.view_seed)
        if reference[: len(views)] != views:
            raise ValueError("the run's views are not the first views of the reference set")
        complement = PairEncoder(
            model,
            tokenizer,
            layer=cfg.layer,
            pool=cfg.pool,
            views=reference[len(views) :],
            batch_tokens=cfg.batch_tokens,
            timing=log,
        )
        provenance["view_diagnostic"] = {
            "reference_n_perm": cfg.reference_n_perm,
            "reference_view_set_id": view_set_id(reference),
            "views_encoded": complement.views,
        }
    (out_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if cfg.throughput_k:
        measure_throughput(
            encoder, state.tasks, cfg, out_dir / "throughput.jsonl", log, state, deadline
        )
    fill_cache(encoder, state.cache, state.tasks, cfg, log, state, deadline)
    if complement is not None:
        diagnose_views(
            complement,
            state.cache,
            state.tasks,
            len(views) + len(complement.views),
            log,
            state,
            deadline,
            out_dir / DIAGNOSTIC_FILE,
        )
    if device.type == "cuda":
        state.peak_memory = {
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device),
        }


# --- episode mode (task MT, run MT-01) -----------------------------------------------------------


def _validate_episode_values(values: dict[str, Any]) -> None:
    """Check an episode-mode config's resolved values; raises ``ConfigError``."""
    for key in ("run_id", "model_id", "input_root", "cache_dir", "out_dir"):
        _check(isinstance(values[key], str) and values[key] != "", f"{key} must be a string")
    _check(
        values["device"] in EPISODE_DEVICES,
        f"device must be one of {EPISODE_DEVICES} in episode mode (rank k uses cuda:k)",
    )
    for key in ("model_dir", "seed_cache"):
        value = values[key]
        _check(value is None or (isinstance(value, str) and value != ""), f"{key}: null or a path")
    _check(values["dtype"] in DTYPES, f"dtype must be one of {sorted(DTYPES)}")
    _check(values["attn_implementation"] in ATTENTION, f"attn_implementation in {ATTENTION}")
    _check(values["pool"] in ("output", "input"), "pool must be 'output' or 'input'")
    _check(isinstance(values["store_views"], bool), "store_views must be true or false")
    for key in ("layer", "n_perm", "view_seed", "verify_pairs", "smoke_seed"):
        _check(_is_int(values[key]) and values[key] >= 0, f"{key} must be an integer >= 0")
    for key in ("batch_tokens", "flush_pairs", "gpus", "chunk_pairs"):
        _check(_is_int(values[key]) and values[key] >= 1, f"{key} must be an integer >= 1")
    wall, bound = values["max_wall_s"], values["verify_min_cosine"]
    _check((_is_int(wall) or isinstance(wall, float)) and wall > 0, "max_wall_s must be positive")
    _check(
        (_is_int(bound) or isinstance(bound, float)) and 0 < bound <= 1,
        "verify_min_cosine must be in (0, 1]",
    )
    episodes = values["episodes"]
    _check(
        isinstance(episodes, dict) and set(episodes) == set(EPISODE_SUBKEYS),
        f"episodes must map exactly {list(EPISODE_SUBKEYS)}",
    )
    for key in ("root", "check_root"):
        _check(isinstance(episodes[key], str) and episodes[key] != "", f"episodes.{key}: a path")
    sha = episodes["manifest_sha256"]
    _check(isinstance(sha, str) and bool(SHA256_HEX.fullmatch(sha)), "episodes.manifest_sha256")
    splits = episodes["splits"]
    _check(
        isinstance(splits, list)
        and len(splits) > 0
        and len(set(splits)) == len(splits)
        and all(split in SPLITS for split in splits),
        f"episodes.splits must be distinct names from {SPLITS}",
    )
    cap = episodes["max_tokens"]
    _check(_is_int(cap) and cap >= 1, "episodes.max_tokens must be an integer >= 1")


def load_episode_config(
    raw: dict[str, Any], smoke: bool, overrides: dict[str, str | None]
) -> Config:
    """Resolve an episode-mode config (task MT): exact keys, smoke section, command line.

    The smoke section replaces ``EPISODE_SMOKE_OVERRIDES`` and adds the stand-in model's seed;
    the file's values and its smoke values are both validated. In a smoke run, relative
    ``episodes`` paths are taken from the repository root, where the training fixture lives.
    ``episodes_root`` and ``check_root`` on the command line replace ``episodes.root`` and
    ``episodes.check_root``.

    Args:
        raw: The parsed YAML mapping, with an ``episodes`` key.
        smoke: Whether to apply the ``smoke`` section.
        overrides: Values from the command line; ``challenges`` is refused here.

    Returns:
        The config; the challenge-mode fields hold their inert values.

    Raises:
        ConfigError: If a key is missing or unknown, or a value is invalid.
    """
    expected = {*EPISODE_KEYS, "smoke"}
    _check(set(raw) == expected, f"episode-mode keys differ from {sorted(expected)}: {sorted(raw)}")
    _check(overrides.get("challenges") is None, "--challenges does not apply to episode mode")
    section = raw.pop("smoke")
    _check(isinstance(section, dict), "smoke must be a mapping")
    allowed = set(EPISODE_SMOKE_OVERRIDES) | set(EPISODE_SMOKE_ONLY)
    _check(set(section) == allowed, f"smoke keys must be exactly {sorted(allowed)}")
    base = {**raw, "smoke_seed": section["seed"]}
    smoked = {**base, **{key: section[key] for key in EPISODE_SMOKE_OVERRIDES}}
    _validate_episode_values(base)
    _validate_episode_values(smoked)
    if base["model_dir"] is None:
        model_folder_parts(base["model_id"])
    values = dict(smoked if smoke else base)
    episodes = dict(values["episodes"])
    for key, value in overrides.items():
        if value is None or key == "challenges":
            continue
        if key == "episodes_root":
            episodes["root"] = value
        elif key == "check_root":
            episodes["check_root"] = value
        else:
            values[key] = value
    if smoke:
        for key in ("root", "check_root"):
            if not Path(episodes[key]).is_absolute():
                episodes[key] = str(REPO / episodes[key])
    values["episodes"] = episodes
    _validate_episode_values(values)
    if not smoke and values["model_dir"] is None:
        model_folder_parts(values["model_id"])
    return Config(
        **{key: values[key] for key in COMMON_KEYS if key != "max_wall_s"},
        max_wall_s=float(values["max_wall_s"]),
        challenges=None,
        subset=None,
        task_ids=None,
        reference_n_perm=None,
        throughput_k=0,
        throughput_tasks=None,
        acceptance_s=0.0,
        smoke_tasks=0,
        smoke_seed=values["smoke_seed"],
        episodes=EpisodeSource(
            root=episodes["root"],
            manifest_sha256=episodes["manifest_sha256"],
            check_root=episodes["check_root"],
            splits=tuple(episodes["splits"]),
            max_tokens=episodes["max_tokens"],
        ),
        gpus=values["gpus"],
        chunk_pairs=values["chunk_pairs"],
        seed_cache=values["seed_cache"],
        verify_pairs=values["verify_pairs"],
        verify_min_cosine=float(values["verify_min_cosine"]),
    )


def config_record(cfg: Config) -> dict[str, Any]:
    """The resolved config as plain YAML and JSON values (tuples become lists).

    A challenge-mode record leaves the episode fields out, so P3.2's ``config.yaml`` keeps its
    keys.
    """
    record = asdict(cfg)
    record["task_ids"] = None if cfg.task_ids is None else list(cfg.task_ids)
    if cfg.episodes is None:
        for key in EPISODE_FIELDS:
            del record[key]
    else:
        record["episodes"]["splits"] = list(cfg.episodes.splits)
    return record


def config_from_record(record: dict[str, Any]) -> Config:
    """Rebuild a config from ``config_record``'s values (the rank processes read it)."""
    values = dict(record)
    if values.get("task_ids") is not None:
        values["task_ids"] = tuple(values["task_ids"])
    if values.get("episodes") is not None:
        source = dict(values["episodes"])
        source["splits"] = tuple(source["splits"])
        values["episodes"] = EpisodeSource(**source)
    return Config(**values)


def _now() -> float:
    """Wall-clock seconds, the clock every process of an episode run shares for its deadline."""
    return time.time()


def _meminfo_total(path: Path) -> int | None:
    """``MemTotal`` of a ``/proc/meminfo`` file in bytes, or None where it cannot be read."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_limit(paths: Sequence[Path]) -> tuple[int | None, str | None]:
    """The container's memory limit from the first readable cgroup file, and that file.

    cgroup v2 writes ``max`` for no limit; cgroup v1 writes a page-rounded :math:`2^{63} - 1`.
    Both read as None (no limit), as does a missing or unreadable file.

    Args:
        paths: Candidate files in order (v2 ``memory.max``, then v1 ``memory.limit_in_bytes``).

    Returns:
        ``(limit in bytes or None, the file read or None)``.
    """
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text == "max":
            return None, str(path)
        try:
            value = int(text)
        except ValueError:
            continue
        return (None if value >= CGROUP_UNLIMITED else value), str(path)
    return None, None


def host_facts(
    meminfo: Path = MEMINFO, cgroup_files: Sequence[Path] = CGROUP_MEMORY_FILES
) -> dict[str, Any]:
    """The machine's CPUs and memory, which the training sessions' memory plan reads.

    Why: no earlier run recorded the 4xL4 machine's memory (K0 measured a CPU session), and each
    of ``MT-02``'s four processes holds the whole embedding table and the training records. A
    container's cgroup limit can be below the machine's ``MemTotal``; what the processes may use
    is the smaller of the two.

    Args:
        meminfo: The ``/proc/meminfo`` file.
        cgroup_files: The cgroup memory-limit files to try, in order.

    Returns:
        ``cpus``; ``memory_bytes`` (``MemTotal``); ``cgroup_memory_limit_bytes`` and
        ``cgroup_file`` (None for no limit); ``memory_limit_bytes``, the smaller known value
        (None when neither is known).
    """
    total = _meminfo_total(meminfo)
    limit, source = _cgroup_limit(cgroup_files)
    known = [value for value in (total, limit) if value is not None]
    return {
        "cpus": os.cpu_count(),
        "memory_bytes": total,
        "cgroup_memory_limit_bytes": limit,
        "cgroup_file": source,
        "memory_limit_bytes": min(known) if known else None,
    }


def namespace_text(cache: EmbeddingCache) -> str:
    """The cache namespace as the compact JSON its shards carry, for the part files' metadata."""
    return json.dumps(cache.namespace, sort_keys=True, separators=(",", ":"))


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` under a temporary name, flush it to disk, and rename it over ``path``."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_part(
    path: Path,
    keys: Sequence[str],
    means: torch.Tensor,
    views: torch.Tensor | None,
    namespace: str,
) -> None:
    """Write one encoded chunk as a part file, atomically (an encoding process's only output).

    Args:
        path: The part file, ``part-<chunk>.safetensors`` in the rank's folder.
        keys: The chunk's pair hashes, in row order.
        means: ``e_bar``, fp32, ``[n, hidden]``.
        views: ``e`` per view, fp32, ``[n, n_views, hidden]``, or None when views are not kept.
        namespace: ``namespace_text`` of the cache it belongs to.
    """
    tensors = {key: means[i].contiguous() for i, key in enumerate(keys)}
    if views is not None:
        tensors |= {f"{key}/views": views[i].contiguous() for i, key in enumerate(keys)}
    _atomic_bytes(path, safetensors_bytes(tensors, metadata={"namespace": namespace}))


def merge_parts(cache: EmbeddingCache, flush_pairs: int) -> dict[str, int]:
    """Move every finished part file into the cache, as its only writer, then delete it.

    Part files appear by rename, so a file named ``part-*`` is complete. Entries go to the cache
    in puts of at least ``flush_pairs`` pairs (one shard and one index rewrite each), and a part
    is deleted only after the put that holds it, so a crash in between leaves a part whose pairs
    the next merge finds cached and skips. A part of another namespace stops the merge.

    Args:
        cache: The open cache, in this process only.
        flush_pairs: Pairs per put.

    Returns:
        ``files`` merged and ``entries`` new in the cache.

    Raises:
        ValueError: If a part holds another namespace.
    """
    root = cache.path / PARTS_DIR
    files = sorted(root.rglob(PART_GLOB)) if root.is_dir() else []
    expected = namespace_text(cache)
    means: dict[str, torch.Tensor] = {}
    views: dict[str, torch.Tensor] = {}
    pending: list[Path] = []
    totals = {"files": 0, "entries": 0}

    def flush() -> None:
        if means:
            totals["entries"] += cache.put(means, views or None)
        for done in pending:
            done.unlink(missing_ok=True)
        means.clear()
        views.clear()
        pending.clear()

    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            if (handle.metadata() or {}).get("namespace") != expected:
                raise ValueError(f"{path} holds another namespace than {cache.path}")
            for key in handle.keys():
                if key.endswith("/views"):
                    views[key.removesuffix("/views")] = handle.get_tensor(key)
                else:
                    means[key] = handle.get_tensor(key)
        pending.append(path)
        totals["files"] += 1
        if len(means) >= flush_pairs:
            flush()
    flush()
    return totals


def seed_from(cache: EmbeddingCache, seed_root: Path) -> dict[str, Any]:
    """Copy an earlier session's cache namespace, part files included, into an empty one.

    Why: Kaggle mounts an earlier notebook's output read-only and refuses a notebook's own
    output, so a second session runs in another notebook and writes to its own working folder;
    copying keeps one writer and one cache, which the next notebook attaches whole.

    Args:
        cache: The open, empty cache of this session.
        seed_root: The earlier session's ``cache_dir``.

    Returns:
        ``path``, ``files`` copied, and ``entries`` of the copied index.

    Raises:
        ValueError: If this cache already holds entries or parts, or the seed lacks the namespace.
    """
    parts = cache.path / PARTS_DIR
    if len(cache) or (parts.is_dir() and any(parts.rglob(PART_GLOB))):
        raise ValueError(f"{cache.path} already holds embeddings: drop --seed-cache to resume it")
    source = Path(seed_root) / cache.path.name
    if not (source / "namespace.json").is_file():
        found = sorted(p.parent.name for p in Path(seed_root).glob("*/namespace.json"))
        raise ValueError(f"the seed {seed_root} has no namespace {cache.path.name}; found {found}")
    # Contents only: a read-only mount's permission bits must not reach the working copy, which
    # this session writes (shutil.copytree would copy them onto every folder).
    files = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts):
            continue  # a temporary file a cut session left
        target = cache.path / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            files += 1
    seeded = EmbeddingCache(cache.path.parent, **_namespace_args(cache))
    return {"path": str(seed_root), "files": files, "entries": len(seeded)}


def _namespace_args(cache: EmbeddingCache) -> dict[str, Any]:
    """The keyword arguments that open ``cache``'s namespace again."""
    ns = cache.namespace
    return {
        "model_id": ns["model_id"],
        "layer": ns["layer"],
        "pool": ns["pool"],
        "view_set_id": ns["view_set_id"],
    }


def load_encoder_model(
    cfg: Config, smoke: bool, model_dir: str, device: torch.device
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the frozen base and its tokenizer as challenge mode does (or the smoke stand-ins).

    Args:
        cfg: The run config.
        smoke: Whether to build the tiny random Qwen3 and the stand-in tokenizer.
        model_dir: The model folder (ignored in a smoke run).
        device: Where the model runs.

    On CUDA the weights load straight onto the GPU (``device_map``), as training's ``load_base``
    does: four processes each staging about 8 GB of bf16 weights in host memory at once could
    exhaust the machine's memory before any encode, and no run has measured it.

    Returns:
        The model (eval mode, no gradients) on ``device`` and the tokenizer.
    """
    if smoke:
        tokenizer: PreTrainedTokenizerBase = smoke_tokenizer()
        model = smoke_model(cfg.smoke_seed, cfg.attn_implementation).to(DTYPES[cfg.dtype])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        kwargs: dict[str, Any] = {"torch_dtype": DTYPES[cfg.dtype], "local_files_only": True}
        if device.type == "cuda":
            kwargs["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, attn_implementation=cfg.attn_implementation, **kwargs
        )
    return model.to(device).eval().requires_grad_(False), tokenizer


def _rank_device(cfg: Config, rank: int) -> torch.device:
    """``cuda:rank``, checked (CUDA present, enough GPUs, bf16 when asked), or the CPU."""
    if cfg.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= rank:
        raise RuntimeError(f"rank {rank} needs cuda:{rank}; {torch.cuda.device_count()} visible")
    if DTYPES[cfg.dtype] is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(rank)} has no bf16 (T4 is not usable)")
    torch.cuda.set_device(rank)
    return torch.device("cuda", rank)


def rank_main(job_path: Path) -> int:
    """One encoding process of an episode run: encode its chunks, one part file each.

    The job (written by ``run_episodes``) names the config, the model folder, this rank's chunks
    (a pickle of ``(chunk index, [(pair hash, pair), ...])``), the parts folder, the log folder,
    and the deadline. Before each chunk the deadline is checked, so a stop loses no finished
    chunk. ``timing-rank<k>.jsonl`` gets one ``encode`` record per chunk and
    ``encode-rank<k>.json`` the process's summary, written however it ends.

    Args:
        job_path: The rank's job file (JSON).

    Returns:
        0 when every chunk is written or the deadline stopped it, 1 on a failure
        (``error-rank<k>.log`` holds the traceback).
    """
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    cfg = config_from_record(job["config"])
    rank, world, out_dir = job["rank"], job["world"], Path(job["out_dir"])
    summary: dict[str, Any] = {
        "rank": rank,
        "status": "started",
        "chunks_assigned": 0,
        "chunks_written": 0,
        "pairs": 0,
        "tokens": 0,
        "encode_seconds": 0.0,
        "device_name": None,
    }
    began = time.perf_counter()
    try:
        device = _rank_device(cfg, rank)
        if device.type == "cpu" and world > 1:
            torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))
        summary["device_name"] = _device_name(device)
        model, tokenizer = load_encoder_model(cfg, job["smoke"], job["model_dir"], device)
        summary["load_seconds"] = time.perf_counter() - began
        encoder = PairEncoder(
            model,
            tokenizer,
            layer=cfg.layer,
            pool=cfg.pool,
            views=view_descriptors(cfg.n_perm, cfg.view_seed),
            batch_tokens=cfg.batch_tokens,
        )
        with open(job["pairs_file"], "rb") as handle:
            work = pickle.load(handle)
        summary["chunks_assigned"] = len(work)
        parts = Path(job["parts_dir"]) / f"rank-{rank}"
        parts.mkdir(parents=True, exist_ok=True)
        timing = out_dir / f"timing-rank{rank}.jsonl"
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for chunk, items in work:
            if _now() > job["deadline"]:
                summary["status"] = "stopped: wall time over max_wall_s"
                break
            prefill: list[int] = []
            encoder.timing = lambda r, t=prefill: (
                t.append(int(r["tokens"])) if r["op"] == "encode_prefill" else None
            )
            start = time.perf_counter()
            pairs = [pair for _, pair in items]
            per_view = encoder.encode_views(pairs) if cfg.store_views else None
            means = per_view.mean(dim=1) if per_view is not None else encoder.encode_pairs(pairs)
            seconds = time.perf_counter() - start
            keys = [key for key, _ in items]
            write_part(parts / f"part-{chunk:06d}.safetensors", keys, means, per_view, job["ns"])
            record = {"puzzle_id": None, "op": "encode", "seconds": seconds}
            record |= {"tokens": sum(prefill), "pairs": len(items), "chunk": chunk}
            append_jsonl(timing, record)
            summary["chunks_written"] += 1
            summary["pairs"] += len(items)
            summary["tokens"] += sum(prefill)
            summary["encode_seconds"] += seconds
        else:
            summary["status"] = "complete"
        if device.type == "cuda":
            summary["max_allocated"] = torch.cuda.max_memory_allocated(device)
        return EXIT_OK
    except Exception as err:
        summary["status"] = f"failed: {type(err).__name__}: {err}"
        (out_dir / f"error-rank{rank}.log").write_text(traceback.format_exc(), encoding="utf-8")
        return EXIT_FAILED
    finally:
        summary["seconds"] = time.perf_counter() - began
        text = json.dumps(summary, indent=2) + "\n"
        (out_dir / f"encode-rank{rank}.json").write_text(text, encoding="utf-8")


def _rank_command(job: Path) -> list[str]:
    """This interpreter, with its no-site flag (the notebook kernel runs -S), on this file."""
    flags = ["-S"] if sys.flags.no_site else []
    return [sys.executable, *flags, str(SCRIPT), "--rank-job", str(job)]


def run_ranks(jobs: Sequence[Path], cache: EmbeddingCache, cfg: Config) -> list[int]:
    """Run the encoding processes and merge their part files while they run and at the end.

    One process runs in this one (``rank_main``); several run as child processes of this
    script, each on its own GPU, and this process merges every ``MERGE_EVERY_S`` seconds. A
    failed process stops the others; whatever they wrote is merged either way.

    Args:
        jobs: One job file per rank.
        cache: The open cache; only this process writes it.
        cfg: The run config.

    Returns:
        Each rank's exit code.
    """
    if len(jobs) == 1:
        try:
            return [rank_main(jobs[0])]
        finally:
            merge_parts(cache, cfg.flush_pairs)
    procs = [subprocess.Popen(_rank_command(job)) for job in jobs]
    try:
        merged_at = time.monotonic()
        while True:
            codes = [proc.poll() for proc in procs]
            done = all(code is not None for code in codes)
            if done or any(code not in (None, 0) for code in codes):
                break
            time.sleep(POLL_S)
            if time.monotonic() - merged_at > MERGE_EVERY_S:
                merge_parts(cache, cfg.flush_pairs)
                merged_at = time.monotonic()
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        merge_parts(cache, cfg.flush_pairs)
    return [proc.returncode for proc in procs]


@dataclass
class EpisodeState:
    """What an episode run has learned so far; the report reads it after any ending.

    Attributes:
        inputs: The model folder, the build folder, its manifest, and the check report.
        splits: Per split: episodes, demonstration occurrences, distinct pairs.
        by_source: Distinct demonstration pairs per source.
        distinct: The distinct demonstration pairs of all splits (the cache's target).
        cached_at_start: Target pairs in the cache when encoding started (after seeding).
        seed: What ``seed_from`` copied, or None.
        leftover: Part files merged before encoding (a cut session's).
        to_encode: Target pairs missing at the start.
        chunks: Chunks of the missing pairs.
        ranks: Each encoding process's summary.
        missing: Target pairs missing at the end.
        cache: The open cache.
        check: The check's result.
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    distinct: int | None = None
    cached_at_start: int | None = None
    seed: dict[str, Any] | None = None
    leftover: dict[str, int] | None = None
    to_encode: int | None = None
    chunks: int | None = None
    ranks: list[dict[str, Any]] = field(default_factory=list)
    missing: int | None = None
    cache: EmbeddingCache | None = None
    check: dict[str, Any] | None = None


def collect_pairs(
    records_by_split: dict[str, Sequence[Any]], state: EpisodeState
) -> dict[str, dict[str, Any]]:
    """Every distinct demonstration pair of the splits, by ``pair_hash``, with the counts.

    Args:
        records_by_split: ``EpisodeRecord``s per split.
        state: Gets ``splits``, ``by_source``, and ``distinct``.

    Returns:
        Hash to ``{"input", "output"}``, the first occurrence kept, in sorted hash order.
    """
    pairs: dict[str, dict[str, Any]] = {}
    by_source: dict[str, set[str]] = defaultdict(set)
    for split, records in records_by_split.items():
        hashes: set[str] = set()
        occurrences = 0
        for record in records:
            for x, y in record.demos:
                pair = {"input": x, "output": y}
                key = pair_hash(pair)
                pairs.setdefault(key, pair)
                hashes.add(key)
                by_source[record.source].add(key)
                occurrences += 1
        state.splits[split] = {
            "episodes": len(records),
            "demo_occurrences": occurrences,
            "distinct": len(hashes),
        }
    state.by_source = {source: len(keys) for source, keys in sorted(by_source.items())}
    state.distinct = len(pairs)
    return {key: pairs[key] for key in sorted(pairs)}


def check_sample(
    cfg: Config,
    smoke: bool,
    model_dir: str,
    cache: EmbeddingCache,
    pairs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    r"""The check: re-encode a seeded sample of cached pairs in other batches; compare.

    Why: a merge or index error (an embedding under another pair's hash) or a corrupt shard
    would pass every count; a fresh encode of a random sample, batched with other pairs than
    when it was cached, catches both, while batch composition moves a bf16 embedding by far less
    than the bound (the 8-view embedding is within cosine 0.9983 of the 16-view one, P3.2-02;
    different pairs sit at a median cosine of 0.55).

    Args:
        cfg: The run config (``verify_pairs``, ``verify_min_cosine``, ``chunk_pairs``).
        smoke: Whether the model is the stand-in.
        model_dir: The model folder.
        cache: The merged cache.
        pairs: The target pairs by hash.

    Returns:
        ``pairs`` checked, the cosines' minimum and median, the largest absolute difference,
        the bound, ``pass``, the worst pairs, and the seconds.
    """
    began = time.perf_counter()
    cached = [key for key in pairs if key in cache]
    sample = random.Random(VERIFY_SEED).sample(cached, min(cfg.verify_pairs, len(cached)))
    device = torch.device("cuda", 0) if cfg.device == "cuda" else torch.device("cpu")
    model, tokenizer = load_encoder_model(cfg, smoke, model_dir, device)
    encoder = PairEncoder(
        model,
        tokenizer,
        layer=cfg.layer,
        pool=cfg.pool,
        views=view_descriptors(cfg.n_perm, cfg.view_seed),
        batch_tokens=cfg.batch_tokens,
    )
    step = cfg.chunk_pairs
    fresh = torch.cat(
        [
            encoder.encode_pairs([pairs[key] for key in sample[i : i + step]])
            for i in range(0, len(sample), step)
        ]
    )  # [n, hidden]
    stored = cache.get(sample)  # [n, hidden]
    cosine = torch.nn.functional.cosine_similarity(fresh.double(), stored.double(), dim=1)
    worst = torch.argsort(cosine)[:VERIFY_WORST].tolist()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "pairs": len(sample),
        "min_cosine": float(cosine.min()),
        "median_cosine": float(cosine.median()),
        "max_abs_diff": float((fresh - stored).abs().max()),
        "bound": cfg.verify_min_cosine,
        "pass": bool(cosine.min() >= cfg.verify_min_cosine),
        "worst": [{"pair": sample[i], "cosine": float(cosine[i])} for i in worst],
        "seconds": time.perf_counter() - began,
    }


def run_episodes(
    cfg: Config,
    smoke: bool,
    out_dir: Path,
    state: EpisodeState,
    provenance: dict[str, Any],
    started: float,
) -> None:
    """An episode-mode run: inputs, the exact count, seeding, the encoding processes, the check.

    Args:
        cfg: The run config (``episodes`` set).
        smoke: Whether to use the stand-in model and tokenizer.
        out_dir: The run-log folder.
        state: Filled as the run proceeds.
        provenance: ``run.json``, completed and written before encoding starts.
        started: ``_now()`` at the script's start; no chunk starts after it plus ``max_wall_s``.

    Raises:
        RuntimeError: If CUDA is asked for but fewer GPUs than ``gpus`` are visible, or an
            encoding process failed.
        CheckFailed: If the check's sample disagrees with the cache.
    """
    if cfg.episodes is None:
        raise ValueError("run_episodes needs an episode-mode config")
    if cfg.device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() < cfg.gpus
    ):
        raise RuntimeError(f"{cfg.gpus} CUDA devices requested, {torch.cuda.device_count()} seen")
    if smoke:
        model_dir = "smoke_model() in scripts/cache_embeddings.py"
    else:
        found, _ = find_inputs(cfg)
        model_dir = str(found)
        print(f"model folder: {model_dir}", flush=True)
    source = cfg.episodes
    episode_set = open_episode_set(
        Path(source.root), source.manifest_sha256, Path(source.check_root), source.max_tokens
    )
    state.inputs = {
        "model_dir": model_dir,
        "episodes": str(episode_set.folder),
        "manifest_sha256": episode_set.manifest_sha256,
        "check_report": str(episode_set.check_report),
    }
    records = {split: load_split(episode_set, split, source.max_tokens) for split in source.splits}
    pairs = collect_pairs(records, state)
    del records
    per_split = ", ".join(f"{split} {row['distinct']}" for split, row in state.splits.items())
    print(
        f"distinct demonstration pairs: {state.distinct} ({per_split}; by source "
        f"{state.by_source})",
        flush=True,
    )
    # The real tokenizer meets the format on a sample and the largest pair before 4 model loads;
    # every other pair is checked as it is rendered for its encode.
    if smoke:
        tokenizer: PreTrainedTokenizerBase = smoke_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    views = view_descriptors(cfg.n_perm, cfg.view_seed)
    sample = random.Random(VERIFY_SEED).sample(sorted(pairs), min(64, len(pairs)))
    longest = max(pairs, key=lambda key: sum(np.asarray(g).size for g in pairs[key].values()))
    probe = [pairs[key] for key in [*sample, longest]]
    preflight([Task("preflight", probe)], QwenFormatter(tokenizer), views)
    cache = EmbeddingCache(
        cfg.cache_dir,
        model_id=cfg.model_id,
        layer=cfg.layer,
        pool=cfg.pool,
        view_set_id=view_set_id(views),
    )
    state.cache = cache
    if cfg.seed_cache is not None:
        state.seed = seed_from(cache, Path(cfg.seed_cache))
        cache = state.cache = EmbeddingCache(cfg.cache_dir, **_namespace_args(cache))
    state.leftover = merge_parts(cache, cfg.flush_pairs)
    todo = [key for key in pairs if key not in cache]
    state.cached_at_start = len(pairs) - len(todo)
    state.to_encode = len(todo)
    chunks = [todo[i : i + cfg.chunk_pairs] for i in range(0, len(todo), cfg.chunk_pairs)]
    state.chunks = len(chunks)
    provenance.update(
        {
            "inputs": state.inputs,
            "encoder": {
                "layer": cfg.layer,
                "pool": cfg.pool,
                "views": views,
                "view_set_id": view_set_id(views),
                "batch_tokens": cfg.batch_tokens,
                "gpus": cfg.gpus,
                "chunk_pairs": cfg.chunk_pairs,
            },
            "cache": {"dir": str(cache.path), "entries_at_start": len(cache), "seed": state.seed},
            "host": host_facts(),
        }
    )
    (out_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if chunks:
        world = min(cfg.gpus, len(chunks))
        deadline = started + cfg.max_wall_s
        with tempfile.TemporaryDirectory(prefix="arc26-mt-") as scratch:
            jobs = []
            for rank in range(world):
                mine = [
                    (j, [(key, pairs[key]) for key in chunks[j]])
                    for j in range(rank, len(chunks), world)
                ]
                pairs_file = Path(scratch) / f"pairs-rank{rank}.pkl"
                with pairs_file.open("wb") as handle:
                    pickle.dump(mine, handle, protocol=pickle.HIGHEST_PROTOCOL)
                job = {
                    "config": config_record(cfg),
                    "smoke": smoke,
                    "rank": rank,
                    "world": world,
                    "model_dir": model_dir,
                    "pairs_file": str(pairs_file),
                    "parts_dir": str(cache.path / PARTS_DIR / f"session-{int(started)}"),
                    "out_dir": str(out_dir),
                    "deadline": deadline,
                    "ns": namespace_text(cache),
                }
                path = Path(scratch) / f"job-rank{rank}.json"
                path.write_text(json.dumps(job), encoding="utf-8")
                jobs.append(path)
            codes = run_ranks(jobs, cache, cfg)
        for rank in range(world):
            summary = out_dir / f"encode-rank{rank}.json"
            if summary.is_file():
                state.ranks.append(json.loads(summary.read_text(encoding="utf-8")))
        # Every process has ended, so emptied part folders can go (never while one may write).
        parts_root = cache.path / PARTS_DIR
        for folder in sorted(parts_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        failed = {rank: code for rank, code in enumerate(codes) if code != 0}
        if failed:
            raise RuntimeError(
                f"encoding processes exited non-zero {failed}; see error-rank<k>.log in {out_dir}"
            )
    state.missing = sum(1 for key in pairs if key not in cache)
    if cfg.verify_pairs and len(cache):
        state.check = check_sample(cfg, smoke, model_dir, cache, pairs)
        if not state.check["pass"]:
            raise CheckFailed(
                f"the check's minimum cosine {state.check['min_cosine']:.6f} is under "
                f"{cfg.verify_min_cosine}"
            )


def build_episode_metrics(
    cfg: Config, state: EpisodeState, status: str, elapsed: float
) -> dict[str, Any]:
    """Collect ``metrics.json`` of an episode run; sections the run did not reach are absent.

    Args:
        cfg: The run config.
        state: What the run learned, possibly partial.
        status: ``complete``, ``stopped: ...``, or ``failed: ...``.
        elapsed: Wall seconds since the start.

    Returns:
        The metrics.
    """
    metrics: dict[str, Any] = {
        "run_id": cfg.run_id,
        "mode": "episodes",
        "status": status,
        "elapsed_s": elapsed,
    }
    if state.inputs:
        metrics["episodes"] = {**state.inputs, "splits": state.splits}
    if state.distinct is not None:
        metrics["pairs"] = {
            "distinct": state.distinct,
            "by_source": state.by_source,
            "cached_at_start": state.cached_at_start,
            "seeded": None if state.seed is None else state.seed["entries"],
            "leftover_parts": None if state.leftover is None else state.leftover["files"],
            "to_encode": state.to_encode,
            "missing": state.missing,
        }
    if state.ranks:
        seconds = sum(r["encode_seconds"] for r in state.ranks)
        tokens = sum(r["tokens"] for r in state.ranks)
        pairs = sum(r["pairs"] for r in state.ranks)
        metrics["encode"] = {
            "gpus": len(state.ranks),
            "chunk_pairs": cfg.chunk_pairs,
            "chunks": state.chunks,
            "pairs": pairs,
            "tokens": tokens,
            "gpu_seconds": seconds,
            "seconds_per_1k_tokens": 1000 * seconds / tokens if tokens else None,
            "pairs_per_gpu_second": pairs / seconds if seconds else None,
            "ranks": state.ranks,
        }
    cache = state.cache
    if cache is not None:
        entries = len(cache)
        metrics["cache"] = {
            "dir": str(cache.path),
            "entries": entries,
            "bytes": cache.size_bytes,
            "bytes_per_pair": cache.size_bytes / entries if entries else None,
            "payload_bytes_per_pair": (
                4 * cache.hidden_size * (1 + (cache.n_views or 0)) if cache.hidden_size else None
            ),
        }
    if state.check is not None:
        metrics["check"] = state.check
    return metrics


def render_episode_summary(cfg: Config, metrics: dict[str, Any], provenance: dict[str, Any]) -> str:
    """Write the short human-readable report of an episode run.

    Args:
        cfg: The run config.
        metrics: ``metrics.json``.
        provenance: ``run.json`` (may be partial).

    Returns:
        Markdown text.
    """
    lines = [
        f"# {cfg.run_id}: embedding cache of the episode build's demonstration pairs",
        "",
        f"- Status: {metrics['status']}; wall {_fmt(metrics['elapsed_s'], 1)} s.",
        f"- Commit: {provenance.get('commit')} (dirty: {provenance.get('dirty')}).",
        f"- Encoder: {cfg.model_id}, layer {cfg.layer}, pool {cfg.pool}, "
        f"{8 * max(cfg.n_perm, 1)} views (seed {cfg.view_seed}), {cfg.dtype}, "
        f"{cfg.attn_implementation}, batch_tokens {cfg.batch_tokens}; {cfg.gpus} processes, "
        f"chunks of {cfg.chunk_pairs} pairs.",
    ]
    episodes = metrics.get("episodes")
    if episodes:
        lines.append(
            f"- Episodes: `{episodes['episodes']}` (manifest `{episodes['manifest_sha256']}`), "
            f"check report `{episodes['check_report']}`."
        )
        for split, row in episodes["splits"].items():
            lines.append(
                f"  - {split}: {row['episodes']} episodes, {row['demo_occurrences']} "
                f"demonstrations, {row['distinct']} distinct pairs."
            )
    pairs = metrics.get("pairs")
    if pairs:
        lines += [
            f"- **Distinct demonstration pairs (the cache's target): {pairs['distinct']}**; by "
            f"source {pairs['by_source']}.",
            f"- Cached at the start: {pairs['cached_at_start']} (seeded: {pairs['seeded']}; "
            f"leftover part files merged: {pairs['leftover_parts']}); to encode: "
            f"{pairs['to_encode']}; missing at the end: {pairs['missing']}.",
        ]
    encode = metrics.get("encode")
    if encode:
        lines.append(
            f"- Encoded this session: {encode['pairs']} pairs, {encode['tokens']} tokens, "
            f"{_fmt(encode['gpu_seconds'], 1)} GPU-seconds over {encode['gpus']} processes "
            f"({_fmt(encode['seconds_per_1k_tokens'], 3)} s per 1,000 tokens per GPU)."
        )
    cache = metrics.get("cache")
    if cache:
        lines.append(
            f"- Cache: `{cache['dir']}`, {cache['entries']} entries, {cache['bytes']} bytes "
            f"({_fmt(cache['bytes_per_pair'], 0)} per pair)."
        )
    check = metrics.get("check")
    if check:
        verdict = "pass" if check["pass"] else "**fail**"
        lines.append(
            f"- Check: {check['pairs']} cached pairs re-encoded in other batches: cosine minimum "
            f"{check['min_cosine']:.6f}, median {check['median_cosine']:.6f}, largest absolute "
            f"difference {check['max_abs_diff']:.3g}; bound {check['bound']}: {verdict}."
        )
    if provenance.get("smoke"):
        reading = "Smoke run: a random-weight stand-in on the CPU checks the code path."
    else:
        reading = f"Decision rule: docs/prerun/{cfg.run_id}.md, section 1."
    host = provenance.get("host")
    if host:

        def gib(value: int | None) -> str:
            return "unknown" if value is None else f"{value / 2**30:.1f} GiB"

        cgroup = host["cgroup_memory_limit_bytes"]
        limit = "no cgroup limit" if cgroup is None else f"cgroup limit {gib(cgroup)}"
        source = host["cgroup_file"] or "no cgroup memory file"
        lines.append(
            f"- Host: {host['cpus']} CPUs, MemTotal {gib(host['memory_bytes'])}, {limit} "
            f"({source}): the processes may use {gib(host['memory_limit_bytes'])}."
        )
    lines += ["", f"{reading} Logs: metrics.json, timing-rank<k>.jsonl, encode-rank<k>.json.", ""]
    return "\n".join(lines)


def finish_episode_run(
    cfg: Config, smoke: bool, out_dir: Path, provenance: dict[str, Any], started: float
) -> int:
    """Run an episode-mode config and write its logs however it ends.

    Args:
        cfg: The run config (``episodes`` set).
        smoke: Whether to use the stand-in model and tokenizer.
        out_dir: The run-log folder.
        provenance: ``run.json`` so far.
        started: ``_now()`` at the script's start.

    Returns:
        0 when every target pair is cached and the check passed (or is off), 3 when the wall time
        stopped the run with pairs missing (resumable with ``--seed-cache`` or in place), 1 on a
        failure, a failed check included.
    """
    state = EpisodeState()
    began = time.monotonic()
    code, status = EXIT_OK, "complete"
    try:
        run_episodes(cfg, smoke, out_dir, state, provenance, started)
        if state.missing:
            code = EXIT_STOPPED
            status = f"stopped: wall time over max_wall_s, {state.missing} pairs missing"
    except CheckFailed as err:
        code, status = EXIT_FAILED, f"failed: CheckFailed: {err}"
    except KeyboardInterrupt:
        code, status = EXIT_FAILED, "failed: interrupted"
    except Exception as err:
        code, status = EXIT_FAILED, f"failed: {type(err).__name__}: {err}"
        (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    (out_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    metrics = build_episode_metrics(cfg, state, status, time.monotonic() - began)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary = render_episode_summary(cfg, metrics, provenance)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return code


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, help="configs/cache_embeddings.yaml (required)")
    parser.add_argument(
        "--smoke", action="store_true", help="CPU run: tiny random Qwen3, synthetic tasks"
    )
    parser.add_argument("--out-dir", help="run-log directory (config: out_dir)")
    parser.add_argument("--cache-dir", help="cache root (config: cache_dir)")
    parser.add_argument(
        "--input-root", help="folder searched for the model and the challenges (config: input_root)"
    )
    parser.add_argument(
        "--model-dir", help="Hugging Face model folder; no search (config: model_dir)"
    )
    parser.add_argument("--challenges", help="ARC challenges JSON; no search (config: challenges)")
    parser.add_argument("--device", help="torch device, for example cuda:0 (config: device)")
    parser.add_argument("--commit", help="code commit to record when not in a git checkout")
    parser.add_argument(
        "--episodes-root", help="episode mode: folder holding the P3.4 build (episodes.root)"
    )
    parser.add_argument(
        "--check-root", help="episode mode: folder holding its leakage report (episodes.check_root)"
    )
    parser.add_argument(
        "--seed-cache", help="episode mode: an earlier session's cache root to start from"
    )
    # An encoding process of an episode run: run_episodes starts this script with its job file.
    parser.add_argument("--rank-job", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        0 when complete, 1 on a failure (``error.log`` holds the traceback), 2 for an invalid
        config or command line, 3 when the wall-time kill criterion stopped the run. An encoding
        process (``--rank-job``) returns ``rank_main``'s code.
    """
    script_started = _now()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rank_job is not None:
        return rank_main(args.rank_job)
    if args.config is None:
        parser.error("--config is required")
    if args.smoke and (args.model_dir or args.challenges or args.input_root):
        print(
            "--smoke builds its own model and tasks: drop --model-dir, --challenges, and "
            "--input-root"
        )
        return EXIT_CONFIG
    # A path on the command line names the folder or file itself, so it is made absolute: a
    # relative challenges path would otherwise be read as the end of a searched path.
    explicit = {
        "model_dir": args.model_dir,
        "challenges": args.challenges,
        "episodes_root": args.episodes_root,
        "check_root": args.check_root,
        "seed_cache": args.seed_cache,
    }
    overrides = {
        "out_dir": args.out_dir,
        "cache_dir": args.cache_dir,
        "input_root": args.input_root,
        **{key: None if path is None else os.path.abspath(path) for key, path in explicit.items()},
        "device": args.device,
    }
    try:
        cfg = load_config(args.config, args.smoke, overrides)
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return EXIT_CONFIG
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke and cfg.episodes is None:
        cfg = replace(
            cfg,
            challenges=os.path.abspath(out_dir / "smoke_challenges.json"),
            model_dir="smoke_model() in scripts/cache_embeddings.py",
        )
    resolved = config_record(cfg)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), "utf-8")
    commit, dirty = (args.commit, None) if args.commit else git_commit(REPO)
    provenance: dict[str, Any] = {
        "run_id": cfg.run_id,
        "smoke": args.smoke,
        "argv": list(sys.argv[1:] if argv is None else argv),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": np.__version__,
        },
        "device": cfg.device,
        "device_name": None,
    }
    if cfg.episodes is not None:
        return finish_episode_run(cfg, args.smoke, out_dir, provenance, script_started)
    state = RunState()
    ledger = out_dir / "throughput.jsonl"
    started = time.monotonic()
    code, status = EXIT_OK, "complete"
    try:
        run(cfg, args.smoke, out_dir, state, provenance, started + cfg.max_wall_s)
    except WallTimeExceeded as err:
        code, status = EXIT_STOPPED, f"stopped: {err}"
    except KeyboardInterrupt:
        code, status = EXIT_FAILED, "failed: interrupted"
    except Exception as err:
        code, status = EXIT_FAILED, f"failed: {type(err).__name__}: {err}"
        (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    provenance["device_name"] = state.device_name
    # Again in every case: a run that failed before the model loaded still records its inputs.
    (out_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    metrics = build_metrics(cfg, state, status, time.monotonic() - started, ledger)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary = render_summary(cfg, metrics, provenance)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
