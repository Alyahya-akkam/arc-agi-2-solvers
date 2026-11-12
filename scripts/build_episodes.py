r"""Build the meta-training episodes and check them for leakage (task P3.4; BRIEF B5 P3.4, B8).

Purpose: P3.5 trains the hypernetwork H on episodes, each a task's demonstrations $D$ and one
left-out query $Q$ under one augmentation (Q6). This script builds them from ReARC, BARC, and
ARC-AGI-2 training, and proves that no ARC-AGI-2 evaluation pair entered them. Plan:
``docs/research/meta_training_sources.md`` section 5.2; runbook ``docs/runbooks/build_episodes.md``.

Modes, each a Kaggle CPU notebook (``scripts/build_episodes_kernel.py`` writes the push folders):

- ``build`` (internet on, no competition data, gate G0): fetch the sources pinned in
  ``configs/episodes.yaml`` (ReARC ``re_arc.zip`` and the BARC JSONL by URL with size and sha256
  checks, ARC-AGI-2 ``data/training`` by a blobless sparse git checkout that never fetches an
  evaluation file), apply the drop list, split by family, sample the train, validation, and toy
  episodes, drop over-length episodes and every training task with a pair canonically equal to a
  validation pair, then write ``episodes/*.jsonl``, ``val/val_{challenges,solutions}.json``,
  ``sources/arc2train_pair_sets.json``, ``MANIFEST.json``, and ``run_logs/`` statistics.
- ``check`` (internet off, competition attached): find the build and the competition files under
  the given roots, verify every file against the manifest, validate every record, run gates G1 to
  G5 (``arc26.hyper.episodes.leakage_report``), the positive control, and report R3; write
  ``run_logs/leakage.{json,md}`` with counts and ids, never grids.
- ``--smoke``: the whole loop on synthetic data, offline: synthetic sources served by ``file://``
  URLs and a local git repository, a synthetic evaluation set, planted leaks of every gate; build,
  check (the planted tasks must be flagged, nothing else), drop them, rebuild, check (must pass).

Method: every pair is validated on load (``validate_grid``); each random choice has its own
seeded generator, so a task's episodes do not change when other tasks are dropped; tokens follow
the notebook format exactly (``episode_tokens``); the split is by family (a ReARC generator with
the same-id ARC task, or one BARC problem). The leakage gates compare exact and
$D_4 \times S_{10}$-canonical hashes of every pair, and of every large input and output grid,
with the evaluation set.

Usage::

    python scripts/build_episodes.py build --config configs/episodes.yaml --out DIR --work DIR
    python scripts/build_episodes.py check --config configs/episodes.yaml --episodes ROOT \
        --competition ROOT --out DIR
    python scripts/build_episodes.py --smoke [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from arc26.hyper.episodes import (
    SEED_PREFIX,
    SOURCES,
    SPLITS,
    Augmentation,
    EpisodeRecord,
    ExampleStream,
    PermutationTable,
    Puzzle,
    SourceTask,
    allocate,
    barc_task_id,
    canonical_matches,
    compare_pair_sets,
    control_outcome,
    derive_seed,
    episode_tokens,
    iter_barc_jsonl,
    keyed_order,
    leakage_report,
    load_arc_dir,
    load_rearc_zip,
    pair_set_digests,
    positive_control,
    read_shards,
    rng_for,
    sample_episodes,
    split_families,
    validate_grid,
    write_shards,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "episodes.yaml"
EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2
TASK = "P3.4"
SCHEMA = 1
MANIFEST = "MANIFEST.json"
LOG_DIR = "run_logs"
USER_AGENT = "arc26-build-episodes/1"
# Competition file names (K0 listing, docs/experiments/K0.md) and the slug that wins on ties.
COMPETITION = "arc-prize-2026-arc-agi-2"
EVAL_CHALLENGES = "arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS = "arc-agi_evaluation_solutions.json"
TRAIN_CHALLENGES = "arc-agi_training_challenges.json"
TRAIN_SOLUTIONS = "arc-agi_training_solutions.json"
# Only these paths of the ARC-AGI-2 repository are checked out: gate G0.
ARC_SPARSE = ("/data/training/", "/data/training.txt")
# Token-length histogram bin width for run_logs/episodes_stats.
TOKEN_BIN = 1024
# Hit rows written to leakage.md; leakage.json keeps them all.
MD_HIT_ROWS = 200
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARC_ID = re.compile(r"[0-9a-f]{8}")
_NAME = re.compile(r"[a-z0-9]+")


class ConfigError(Exception):
    """The config breaks a rule of ``CONFIG_SPEC``; nothing was run."""


class PinError(Exception):
    """A fetched source differs from its pin, or a download failed; nothing was built."""


class BuildError(Exception):
    """The build or check found an inconsistency it must not paper over."""


# Configuration.

Check = Callable[[Any, str], None]


def _fail(where: str, rule: str) -> None:
    raise ConfigError(f"{where} {rule}")


def _int(low: int, high: int = 2**62) -> Check:
    def check(value: Any, where: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            _fail(where, f"must be an integer from {low} to {high}")

    return check


def _matches(pattern: re.Pattern[str], what: str) -> Check:
    def check(value: Any, where: str) -> None:
        if not isinstance(value, str) or not pattern.fullmatch(value):
            _fail(where, f"must be {what}")

    return check


def _url(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.startswith(("https://", "file://")):
        _fail(where, "must be an https:// (or, for --smoke, file://) URL")


def _probs(value: Any, where: str) -> None:
    ok = isinstance(value, list) and 1 <= len(value) <= 10
    ok = ok and all(isinstance(p, int | float) and not isinstance(p, bool) and p > 0 for p in value)
    if not ok or abs(sum(value) - 1) > 1e-9:
        _fail(where, "must list 1 to 10 positive probabilities that sum to 1")


def _ids(value: Any, where: str) -> None:
    ok = isinstance(value, list) and all(isinstance(v, str) and _ARC_ID.fullmatch(v) for v in value)
    if not ok or len(set(value)) != len(value):
        _fail(where, "must be a list of distinct 8-hex task ids")


def _sizes(value: Any, where: str) -> None:
    ok = isinstance(value, list) and bool(value)
    for row in value if ok else ():
        ok = ok and isinstance(row, list) and len(row) == 3
        ok = ok and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in row)
    if not ok:
        _fail(where, "must be a non-empty list of [train pairs, test pairs, weight], each >= 1")


def _drop(value: Any, where: str) -> None:
    if not isinstance(value, list):
        _fail(where, "must be a list")
    for i, item in enumerate(value):
        keys = {"source", "task_id", "reason"}
        if not isinstance(item, dict) or set(item) != keys:
            _fail(f"{where}[{i}]", f"must have exactly the keys {sorted(keys)}")
        if item["source"] not in SOURCES or not isinstance(item["task_id"], str):
            _fail(f"{where}[{i}]", f"needs a source in {SOURCES} and a task_id string")
        seed = item["task_id"].removeprefix(SEED_PREFIX)
        if seed != item["task_id"] and not _ARC_ID.fullmatch(seed):
            _fail(f"{where}[{i}]", f"a {SEED_PREFIX}<id> task_id needs an 8-hex id")
        if not isinstance(item["reason"], str) or not item["reason"]:
            _fail(f"{where}[{i}]", "needs a reason")


_sha1 = _matches(_SHA1, "a 40-hex git object id")
_sha256 = _matches(_SHA256, "a 64-hex sha256")
CONFIG_SPEC: dict[str, Any] = {
    "seed": _int(0),
    "max_tokens": _int(16, 10**6),
    "demo_probs": _probs,
    "color_permutations": _int(1, 100000),
    "sources": {
        "arc2train": {
            "repo": _url, "commit": _sha1, "tree": _sha1, "index_blob": _sha1, "tasks": _int(1),
        },
        "rearc": {
            "url": _url, "commit": _sha1, "blob": _sha1, "sha256": _sha256, "bytes": _int(1),
            "tasks": _int(1),
        },
        "barc": {
            "url": _url, "revision": _sha1, "sha256": _sha256, "bytes": _int(1), "rows": _int(1),
            "name": _matches(_NAME, "a short lowercase name"), "max_pairs": _int(2),
        },
    },
    "train": {"barc_tasks": _int(0), "rearc_episodes": _int(0), "arc2train_episodes": _int(0)},
    "val": {
        "rearc_held_out": _ids,
        "rearc_puzzles_per_family": _int(0),
        "barc_puzzles": _int(0),
        "sizes": _sizes,
        "max_attempts": _int(1),
    },
    "toy": {
        "barc": _int(0), "rearc": _int(0), "arc2train": _int(0), "episodes_per_puzzle": _int(1),
        "pairs_per_puzzle": _int(2),
    },
    "gates": {"min_cells": _int(1), "min_colors": _int(1), "positive_controls": _int(1)},
    "shards": {split: _int(1, 1000) for split in SPLITS},
    "drop": _drop,
}  # fmt: skip


def _check_tree(value: Any, spec: Any, where: str) -> None:
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            _fail(where, "must be a mapping")
        if set(value) != set(spec):
            _fail(where, f"keys must be exactly {sorted(spec)}, got {sorted(value)}")
        for key, sub in spec.items():
            _check_tree(value[key], sub, f"{where}.{key}" if where else key)
    else:
        spec(value, where)


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Check a resolved config against ``CONFIG_SPEC`` and its cross-field rules.

    Args:
        cfg: The config without its ``smoke`` section.

    Raises:
        ConfigError: If a key is missing or unknown, a value has the wrong type or range, a URL
            does not contain its pinned commit or revision, or ``toy.pairs_per_puzzle`` exceeds
            ``sources.barc.max_pairs`` (the pairs a BARC problem keeps).
    """
    _check_tree(cfg, CONFIG_SPEC, "")
    rearc, barc = cfg["sources"]["rearc"], cfg["sources"]["barc"]
    if rearc["commit"] not in rearc["url"]:
        _fail("sources.rearc.url", "must name sources.rearc.commit")
    if barc["revision"] not in barc["url"]:
        _fail("sources.barc.url", "must name sources.barc.revision")
    if cfg["toy"]["pairs_per_puzzle"] > barc["max_pairs"]:
        _fail("toy.pairs_per_puzzle", "must not exceed sources.barc.max_pairs")


def _merge(base: dict[str, Any], override: Mapping[str, Any], where: str = "smoke") -> None:
    for key, value in override.items():
        if key not in base:
            _fail(f"{where}.{key}", "is not a key of the main config")
        if isinstance(value, dict) and isinstance(base[key], dict):
            _merge(base[key], value, f"{where}.{key}")
        else:
            base[key] = value


def read_config(path: Path, smoke: bool = False) -> dict[str, Any]:
    """Read a YAML or JSON config, apply its ``smoke`` section in smoke mode, and validate it.

    Args:
        path: ``configs/episodes.yaml``, or the JSON copy a push folder carries.
        smoke: Apply the ``smoke`` section; its ``sources`` come from ``smoke_sources`` later,
            so they are not validated here.

    Returns:
        The resolved config without the ``smoke`` key.

    Raises:
        ConfigError: If the file cannot be read or parsed, or breaks ``validate_config``.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix in (".yaml", ".yml"):
            import yaml  # only the YAML path needs it; push folders carry JSON

            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
    except (OSError, ValueError) as err:
        raise ConfigError(f"cannot read {path}: {err}") from err
    if not isinstance(raw, dict) or not isinstance(raw.get("smoke"), dict):
        raise ConfigError(f"{path} must be a mapping with a smoke section")
    smoke_section = raw.pop("smoke")
    if smoke:
        _merge(raw, smoke_section)
    else:
        validate_config(raw)
    return raw


def config_sha256(cfg: Mapping[str, Any]) -> str:
    """Return the sha256 of a config's compact, key-sorted JSON (YAML and JSON copies agree).

    Args:
        cfg: A resolved config.

    Returns:
        64 hex characters.
    """
    data = json.dumps(cfg, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# Fetching the pinned sources.


def _git(*args: str, cwd: Path | None = None) -> str:
    # The user's and system git config never apply (no hooks, signing, or URL rewrites), and git
    # never prompts for credentials.
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise PinError(f"git {args[0]} failed: {result.stderr.strip()[-500:]}")
    return result.stdout.strip()


def download(
    url: str,
    dest: Path,
    *,
    sha256: str,
    size: int,
    git_blob: str | None = None,
    attempts: int = 3,
    timeout: float = 60.0,
) -> dict[str, Any]:
    r"""Stream a URL to a file and verify its size, sha256, and optionally its git blob id.

    Args:
        url: ``https://`` or ``file://`` URL.
        dest: Target file; a ``.part`` file is written first and renamed after verification.
        sha256: Expected sha256 of the bytes.
        size: Expected byte count.
        git_blob: Expected git blob id, sha1 of ``blob <size>\0`` and the bytes; None skips it.
        attempts: Tries on network errors; a pin mismatch is never retried.
        timeout: Socket timeout per read, seconds.

    Returns:
        ``url``, ``bytes``, ``sha256``, ``git_blob`` (when checked).

    Raises:
        PinError: If every attempt fails or the bytes differ from a pin.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for _ in range(attempts):
        digest, blob, count = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % size), 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                open(part, "wb") as out,
            ):
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
                    blob.update(chunk)
                    count += len(chunk)
        except (OSError, http.client.HTTPException) as err:
            last = err
            continue
        if count != size or digest.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise PinError(
                f"{url}: got {count} bytes, sha256 {digest.hexdigest()}; pinned {size}, {sha256}"
            )
        if git_blob is not None and blob.hexdigest() != git_blob:
            part.unlink(missing_ok=True)
            raise PinError(f"{url}: git blob {blob.hexdigest()}, pinned {git_blob}")
        part.replace(dest)
        result = {"url": url, "bytes": count, "sha256": sha256}
        return result | ({"git_blob": git_blob} if git_blob else {})
    raise PinError(f"{url}: download failed after {attempts} attempts: {last}")


def fetch_arc_training(spec: Mapping[str, Any], dest: Path) -> tuple[Path, dict[str, Any]]:
    """Check out ARC-AGI-2 ``data/training`` at the pinned commit and nothing else (gate G0).

    A blobless clone holds commits and trees only; the sparse checkout then fetches the blobs of
    ``data/training/`` and ``data/training.txt``, so no blob under ``data/evaluation`` is ever
    downloaded. The pins are checked with git's own object ids, and the task files must equal the
    ids in ``data/training.txt``.

    Args:
        spec: ``sources.arc2train`` of the config: ``repo``, ``commit``, ``tree`` (of
            ``data/training``), ``index_blob`` (of ``data/training.txt``), ``tasks``.
        dest: Folder for the clone; replaced if it exists.

    Returns:
        The ``data/training`` folder and the input record for the manifest.

    Raises:
        PinError: If git fails, an object id differs from its pin, an evaluation path was checked
            out, or the files do not match the index.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", spec["repo"], str(dest))
    _git("config", "core.sparseCheckout", "true", cwd=dest)
    _git("config", "core.sparseCheckoutCone", "false", cwd=dest)
    info = dest / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "sparse-checkout").write_text("\n".join(ARC_SPARSE) + "\n", encoding="utf-8")
    _git("checkout", "--quiet", "--detach", spec["commit"], cwd=dest)
    found = {
        "commit": _git("rev-parse", "HEAD", cwd=dest),
        "tree": _git("rev-parse", "HEAD:data/training", cwd=dest),
        "index_blob": _git("rev-parse", "HEAD:data/training.txt", cwd=dest),
    }
    for key, value in found.items():
        if value != spec[key]:
            raise PinError(f"ARC-AGI-2 {key} is {value}, pinned {spec[key]}")
    if (dest / "data" / "evaluation").exists():
        raise PinError("the sparse checkout holds data/evaluation (gate G0)")
    training = dest / "data" / "training"
    index = (dest / "data" / "training.txt").read_text(encoding="utf-8").split()
    files = sorted(p.stem for p in training.glob("*.json"))
    if sorted(index) != files or len(files) != spec["tasks"]:
        raise PinError(
            f"ARC-AGI-2 training holds {len(files)} files, the index {len(index)} ids, "
            f"pinned {spec['tasks']}"
        )
    git_version = _git("--version")
    return training, {"repo": spec["repo"], **found, "files": len(files), "git": git_version}


@dataclass
class SourcePaths:
    """Local copies of the three sources.

    Attributes:
        arc_dir: ARC-AGI-2 ``data/training``.
        rearc_zip: ``re_arc.zip``.
        barc_jsonl: The BARC JSONL file.
    """

    arc_dir: Path
    rearc_zip: Path
    barc_jsonl: Path


def fetch_sources(cfg: Mapping[str, Any], work: Path) -> tuple[SourcePaths, dict[str, Any]]:
    """Fetch and verify the three pinned sources into a scratch folder.

    Args:
        cfg: Resolved config.
        work: Scratch folder, on Kaggle under ``/tmp``, never ``/kaggle/working``.

    Returns:
        The local paths and, per source, the verified pins for the manifest.

    Raises:
        PinError: If a source cannot be fetched or differs from its pin.
    """
    src = cfg["sources"]
    rearc = download(
        src["rearc"]["url"], work / "re_arc.zip", sha256=src["rearc"]["sha256"],
        size=src["rearc"]["bytes"], git_blob=src["rearc"]["blob"],
    )  # fmt: skip
    barc = download(
        src["barc"]["url"], work / "barc.jsonl", sha256=src["barc"]["sha256"],
        size=src["barc"]["bytes"],
    )  # fmt: skip
    arc_dir, arc = fetch_arc_training(src["arc2train"], work / "ARC-AGI-2")
    paths = SourcePaths(arc_dir, work / "re_arc.zip", work / "barc.jsonl")
    inputs = {
        "arc2train": arc,
        "rearc": {**rearc, "commit": src["rearc"]["commit"]},
        "barc": {**barc, "revision": src["barc"]["revision"]},
    }
    return paths, inputs


# Loading and building.


@dataclass
class Sources:
    """The loaded sources.

    Attributes:
        arc2train: ARC-AGI-2 training tasks by id.
        rearc: ReARC generators by id.
        barc: BARC problems by id, first occurrence of each program, in file order.
        stats: Per source: tasks, examples, invalid pairs; for BARC also rows, duplicate
            programs, and malformed lines.
    """

    arc2train: dict[str, SourceTask]
    rearc: dict[str, SourceTask]
    barc: dict[str, SourceTask]
    stats: dict[str, Any] = field(default_factory=dict)


def _source_stats(tasks: Iterable[SourceTask]) -> dict[str, int]:
    tasks = list(tasks)
    return {
        "tasks": len(tasks),
        "examples": sum(t.examples for t in tasks),
        "valid_pairs": sum(len(t.pairs) for t in tasks),
        "invalid_pairs": sum(t.invalid_pairs for t in tasks),
    }


def load_sources(paths: SourcePaths, cfg: Mapping[str, Any]) -> Sources:
    """Load the three sources and check their sizes against the config.

    Args:
        paths: Local copies (``fetch_sources``).
        cfg: Resolved config.

    Returns:
        The sources; BARC problems keep at most ``max_pairs`` valid pairs each.

    Raises:
        BuildError: If a source holds a different number of tasks or rows than pinned.
    """
    src = cfg["sources"]
    arc = load_arc_dir(paths.arc_dir)
    rearc = load_rearc_zip(paths.rearc_zip)
    malformed: Counter = Counter()
    barc: dict[str, SourceTask] = {}
    rows = duplicates = 0
    for task in iter_barc_jsonl(
        paths.barc_jsonl,
        src["barc"]["name"],
        max_pairs=src["barc"]["max_pairs"],
        malformed=malformed,
    ):
        rows += 1
        if task.task_id in barc:
            duplicates += 1
        else:
            barc[task.task_id] = task
    rows += malformed["lines"]
    for name, found, pinned in (
        ("arc2train tasks", len(arc), src["arc2train"]["tasks"]),
        ("rearc generators", len(rearc), src["rearc"]["tasks"]),
        ("barc rows", rows, src["barc"]["rows"]),
    ):
        if found != pinned:
            raise BuildError(f"{name}: found {found}, pinned {pinned}")
    stats = {
        "arc2train": _source_stats(arc.values()),
        "rearc": _source_stats(rearc.values()),
        "barc": {
            **_source_stats(barc.values()),
            "rows": rows,
            "duplicate_programs": duplicates,
            "malformed_lines": malformed["lines"],
        },
    }
    return Sources(arc, rearc, barc, stats)


@dataclass
class BuildResult:
    """Everything a build writes.

    Attributes:
        train: Training episodes in pool order (a run of N steps uses the first N).
        val: Validation episodes, one per test pair of each puzzle.
        toy: Toy-split episodes for P3.5.
        puzzles: Validation puzzles.
        table: The augmentation permutation table.
        arc2train_digests: Pair-set digests of every loaded ARC-AGI-2 training task (report R3).
        stats: Counts of the build.
        provenance: Held-out families and BARC seed ids.
    """

    train: list[EpisodeRecord]
    val: list[EpisodeRecord]
    toy: list[EpisodeRecord]
    puzzles: list[Puzzle]
    table: PermutationTable
    arc2train_digests: dict[str, dict[str, str]]
    stats: dict[str, Any]
    provenance: dict[str, Any]


class _Tally:
    """Per-split, per-source counters of a build."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, Counter]] = {s: {} for s in SPLITS}

    def add(self, split: str, source: str, **values: int) -> None:
        self.counts[split].setdefault(source, Counter()).update(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            split: {source: dict(sorted(c.items())) for source, c in sorted(by_source.items())}
            for split, by_source in self.counts.items()
        }


def _draw_size(rng: random.Random, sizes: Sequence[Sequence[int]]) -> tuple[int, int]:
    row = rng.choices(list(sizes), weights=[w for _, _, w in sizes])[0]
    return row[0], row[1]


def _fits(puzzle: Puzzle, max_tokens: int) -> bool:
    return all(episode_tokens([*puzzle.train, pair]) <= max_tokens for pair in puzzle.test)


def build_episode_set(cfg: Mapping[str, Any], sources: Sources) -> BuildResult:
    """Split the sources by family and sample every split's episodes (note, sections 4.4, 4.5).

    - Drop list: tasks named in ``drop`` are removed from every split first; an entry
      ``seed:<id>`` removes every task of its source that names seed ``<id>`` (BARC problems).
    - Validation: ``rearc_puzzles_per_family`` puzzles from each held-out ReARC generator, drawn
      from its example stream, and ``barc_puzzles`` puzzles from BARC problems in keyed order;
      each puzzle's train and test counts are drawn from ``val.sizes`` (DEV120's joint counts),
      and a puzzle whose validation episode exceeds ``max_tokens`` is redrawn. The ARC-AGI-2
      tasks of held-out families leave training.
    - Training: one fresh episode from each of the first ``barc_tasks`` BARC problems,
      ``rearc_episodes`` from disjoint examples of the training generators, and
      ``arc2train_episodes`` leave-one-out episodes of the ARC-AGI-2 training tasks.
    - Toy: puzzles of training tasks of each source, ``episodes_per_puzzle`` episodes each.
    - Then every training or toy task with a pair canonically equal to a validation pair is
      dropped, and the build fails unless no family and no canonical pair crosses the split.

    Args:
        cfg: Resolved config.
        sources: Loaded sources.

    Returns:
        The episodes, puzzles, table, digests, statistics, and provenance.

    Raises:
        ConfigError: If a held-out generator is not in ReARC.
        BuildError: If a family or a canonical pair still crosses the split after the drops.
    """
    started = time.perf_counter()
    seed, max_tokens, probs = cfg["seed"], cfg["max_tokens"], cfg["demo_probs"]
    table = PermutationTable.seeded(cfg["color_permutations"], derive_seed(seed, "permutations"))
    n_aug = table.n_augmentations
    common = {"demo_probs": probs, "n_augmentations": n_aug, "max_tokens": max_tokens}
    tally = _Tally()
    dropped = {(d["source"], d["task_id"]) for d in cfg["drop"]}
    seeds = {
        s: {
            t.removeprefix(SEED_PREFIX)
            for src, t in dropped
            if src == s and t.startswith(SEED_PREFIX)
        }
        for s in SOURCES
    }
    pools = {"arc2train": sources.arc2train, "rearc": sources.rearc, "barc": sources.barc}
    kept = {
        s: {
            t: task
            for t, task in pool.items()
            if (s, t) not in dropped and not seeds[s].intersection(task.seeds)
        }
        for s, pool in pools.items()
    }
    drop_stats = {s: len(pools[s]) - len(kept[s]) for s in SOURCES}
    arc, rearc, barc = kept["arc2train"], kept["rearc"], kept["barc"]

    unknown = sorted(set(cfg["val"]["rearc_held_out"]) - set(sources.rearc))
    if unknown:
        raise ConfigError(f"val.rearc_held_out names generators missing from ReARC: {unknown}")
    held = sorted(set(cfg["val"]["rearc_held_out"]) & set(rearc))
    rearc_split = split_families(rearc, held)
    arc_split = split_families(arc, set(held) & set(arc))
    streams: dict[str, ExampleStream] = {}

    def stream(gen: str) -> ExampleStream:
        if gen not in streams:
            streams[gen] = ExampleStream(
                len(rearc[gen].pairs), rng_for(seed, "stream", "rearc", gen)
            )
        return streams[gen]

    puzzles: list[Puzzle] = []
    sizes, attempts = cfg["val"]["sizes"], cfg["val"]["max_attempts"]
    for gen in held:
        rng = rng_for(seed, "val", "rearc", gen)
        pairs = rearc[gen].pairs
        for i in range(cfg["val"]["rearc_puzzles_per_family"]):
            for _ in range(attempts):
                n_train, n_test = _draw_size(rng, sizes)
                if stream(gen).remaining < n_train + n_test:
                    tally.add("val", "rearc", short=1)
                    break
                idx = stream(gen).take(n_train + n_test)
                puzzle = Puzzle(
                    f"rearc-{gen}-{i}", "rearc", gen, gen,
                    [pairs[j] for j in idx[:n_train]], [pairs[j] for j in idx[n_train:]],
                )  # fmt: skip
                if _fits(puzzle, max_tokens):
                    puzzles.append(puzzle)
                    break
                tally.add("val", "rearc", over_length=1)
            else:
                tally.add("val", "rearc", short=1)
    barc_val = []
    for task_id in keyed_order(barc, seed, "val", "barc"):
        if len(barc_val) == cfg["val"]["barc_puzzles"]:
            break
        task, rng = barc[task_id], rng_for(seed, "val", "barc", task_id)
        n_train, n_test = _draw_size(rng, sizes)
        if len(task.pairs) < n_train + n_test:
            tally.add("val", "barc", too_few_pairs=1)
            continue
        idx = rng.sample(range(len(task.pairs)), n_train + n_test)
        puzzle = Puzzle(
            f"barc-{task_id.rsplit(':', 1)[1]}", "barc", task_id, task.family,
            [task.pairs[j] for j in idx[:n_train]], [task.pairs[j] for j in idx[n_train:]],
        )  # fmt: skip
        if _fits(puzzle, max_tokens):
            barc_val.append(puzzle)
        else:
            tally.add("val", "barc", over_length=1)
    puzzles += barc_val
    barc_split = split_families(barc, {p.family for p in barc_val})
    val = []
    for puzzle in puzzles:
        episodes = puzzle.episodes(rng_for(seed, "val-augmentation", puzzle.puzzle_id), n_aug)
        val += episodes
        tally.add("val", puzzle.source, puzzles=1, episodes=len(episodes))

    def run(
        task: SourceTask, n: int, split: str, rng: random.Random, pool: ExampleStream | None
    ) -> list[EpisodeRecord]:
        sampled = sample_episodes(task, n, split=split, rng=rng, stream=pool, **common)
        tally.add(
            split,
            task.source,
            requested=n,
            episodes=len(sampled.records),
            over_length=sampled.over_length,
            short=sampled.short,
        )
        return sampled.records

    train: list[EpisodeRecord] = []
    barc_train = keyed_order(
        [t for t, s in barc_split.items() if s == "train"], seed, "train", "barc"
    )
    barc_train = barc_train[: cfg["train"]["barc_tasks"]]
    for task_id in barc_train:
        task = barc[task_id]
        pool = ExampleStream(len(task.pairs), rng_for(seed, "stream", "barc", task_id))
        train += run(task, 1, "train", rng_for(seed, "train", "barc", task_id), pool)
    rearc_alloc = allocate(
        [g for g, s in rearc_split.items() if s == "train"],
        cfg["train"]["rearc_episodes"],
        seed,
        "allocate",
        "rearc",
    )
    for gen in sorted(rearc_alloc):
        train += run(
            rearc[gen], rearc_alloc[gen], "train", rng_for(seed, "train", "rearc", gen), stream(gen)
        )
    arc_alloc = allocate(
        [t for t, s in arc_split.items() if s == "train"],
        cfg["train"]["arc2train_episodes"],
        seed,
        "allocate",
        "arc2train",
    )
    for task_id in sorted(arc_alloc):
        train += run(
            arc[task_id],
            arc_alloc[task_id],
            "train",
            rng_for(seed, "train", "arc2train", task_id),
            None,
        )

    toy: list[EpisodeRecord] = []
    toy_cfg = cfg["toy"]
    per_puzzle, n_pairs = toy_cfg["episodes_per_puzzle"], toy_cfg["pairs_per_puzzle"]
    for task_id in keyed_order(barc_train, seed, "toy", "barc")[: toy_cfg["barc"]]:
        task, rng = barc[task_id], rng_for(seed, "toy", "barc", task_id)
        pairs = rng.sample(task.pairs, min(n_pairs, len(task.pairs)))
        toy += run(SourceTask("barc", task_id, task.family, pairs), per_puzzle, "toy", rng, None)
    training_gens = [g for g, s in rearc_split.items() if s == "train"]
    for gen in keyed_order(training_gens, seed, "toy", "rearc")[: toy_cfg["rearc"]]:
        idx = stream(gen).take(min(n_pairs, stream(gen).remaining))
        sub = SourceTask("rearc", gen, gen, [rearc[gen].pairs[j] for j in idx])
        toy += run(sub, per_puzzle, "toy", rng_for(seed, "toy", "rearc", gen), None)
    training_arc = [t for t, s in arc_split.items() if s == "train"]
    for task_id in keyed_order(training_arc, seed, "toy", "arc2train")[: toy_cfg["arc2train"]]:
        toy += run(
            arc[task_id], per_puzzle, "toy", rng_for(seed, "toy", "arc2train", task_id), None
        )

    val_pairs = [pair for puzzle in puzzles for pair in (*puzzle.train, *puzzle.test)]
    crossing = canonical_matches(val_pairs, train + toy)
    for source, _ in sorted(crossing):
        tally.add("train", source, cross_split_tasks=1)
    train = [r for r in train if (r.source, r.task_id) not in crossing]
    toy = [r for r in toy if (r.source, r.task_id) not in crossing]
    if canonical_matches(val_pairs, train + toy):
        raise BuildError("a training pair still equals a validation pair after the drops")
    val_families = {r.family for r in val}
    shared = sorted(val_families & {r.family for r in train + toy})
    if shared:
        raise BuildError(f"families in both validation and training: {shared[:10]}")
    rng_for(seed, "order", "train").shuffle(train)
    rng_for(seed, "order", "toy").shuffle(toy)

    stats = {
        "sources": sources.stats,
        "dropped_by_list": drop_stats,
        "splits": tally.as_dict(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    provenance = {
        "held_out_families": {
            "rearc": held,
            "arc2train": sorted(t for t, s in arc_split.items() if s == "val"),
            "barc": sorted(p.family for p in barc_val),
        },
        "barc_seed_ids": sorted({s for t in barc.values() for s in t.seeds}),
    }
    digests = pair_set_digests({t: task.pairs for t, task in sources.arc2train.items()})
    return BuildResult(train, val, toy, puzzles, table, digests, stats, provenance)


# Statistics.


def _percentile(values: Sequence[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0


def split_stats(records: Sequence[EpisodeRecord], max_tokens: int) -> dict[str, Any]:
    """Summarize a split per source: episodes, tasks, token lengths, demo counts, large grids.

    Args:
        records: The split's records.
        max_tokens: Upper edge of the token histogram.

    Returns:
        Per source and ``all``: ``episodes``, ``tasks``, ``pairs``, token ``mean``, ``p50``,
        ``p90``, ``p99``, ``max``, ``token_histogram`` (bins of ``TOKEN_BIN`` by lower edge),
        ``demos`` (episodes per demo count), and ``max_side_over_20`` (episodes with a side
        above 20, the DEV120 regime).
    """
    groups: dict[str, list[EpisodeRecord]] = {"all": list(records)}
    for record in records:
        groups.setdefault(record.source, []).append(record)
    summary = {}
    for name, group in sorted(groups.items()):
        tokens = [r.tokens for r in group]
        bins = Counter((t // TOKEN_BIN) * TOKEN_BIN for t in tokens)
        summary[name] = {
            "episodes": len(group),
            "tasks": len({(r.source, r.task_id) for r in group}),
            "pairs": sum(len(r.pairs) for r in group),
            "mean": round(statistics.fmean(tokens), 1) if tokens else 0,
            "p50": _percentile(tokens, 0.5),
            "p90": _percentile(tokens, 0.9),
            "p99": _percentile(tokens, 0.99),
            "max": max(tokens, default=0),
            "token_histogram": {
                str(k): bins.get(k, 0) for k in range(0, max_tokens + 1, TOKEN_BIN)
            },
            "demos": dict(sorted(Counter(r.n_demos for r in group).items())),
            "max_side_over_20": sum(
                any(max(g.shape) > 20 for pair in r.pairs for g in pair) for r in group
            ),
        }
    return summary


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def stats_markdown(stats: Mapping[str, Any]) -> str:
    """Render ``run_logs/episodes_stats.json`` as Markdown: inputs, splits, drops, token lengths.

    Args:
        stats: The statistics dict ``write_build`` saves.

    Returns:
        The Markdown text.
    """
    lines = [
        "# P3.4 episodes: build statistics",
        "",
        f"Manifest sha256: `{stats['manifest_sha256']}`",
        "",
    ]
    lines += ["## Sources", ""]
    lines += _table(
        ("source", "tasks", "examples", "valid pairs", "invalid pairs", "dropped by list"),
        (
            (
                s,
                v["tasks"],
                v["examples"],
                v["valid_pairs"],
                v["invalid_pairs"],
                stats["dropped_by_list"][s],
            )
            for s, v in stats["sources"].items()
        ),
    )
    barc = stats["sources"]["barc"]
    lines += [
        "",
        f"BARC rows {barc['rows']}, duplicate programs {barc['duplicate_programs']}, "
        f"malformed lines {barc['malformed_lines']}.",
        "",
    ]
    lines += ["## Episodes per split and source", ""]
    keys = (
        "puzzles",
        "requested",
        "episodes",
        "over_length",
        "short",
        "too_few_pairs",
        "cross_split_tasks",
    )
    rows = []
    for split, by_source in stats["splits"].items():
        for source, counts in by_source.items():
            rows.append((split, source, *(counts.get(k, 0) for k in keys)))
    lines += _table(("split", "source", *keys), rows)
    lines += ["", "## Token lengths (episode tokens in the longer orientation)", ""]
    rows = []
    for split, by_source in stats["lengths"].items():
        for source, s in by_source.items():
            rows.append(
                (
                    split,
                    source,
                    s["episodes"],
                    s["tasks"],
                    s["mean"],
                    s["p50"],
                    s["p90"],
                    s["p99"],
                    s["max"],
                    s["max_side_over_20"],
                )
            )
    lines += _table(
        ("split", "source", "episodes", "tasks", "mean", "p50", "p90", "p99", "max", "side > 20"),
        rows,
    )
    lines += ["", f"## Token histogram, all sources (bins of {TOKEN_BIN})", ""]
    for split, by_source in stats["lengths"].items():
        hist = by_source["all"]["token_histogram"]
        lines += [f"- {split}: " + ", ".join(f"{k}: {v}" for k, v in hist.items())]
    lines += ["", "## Demo counts", ""]
    for split, by_source in stats["lengths"].items():
        lines += [
            f"- {split}: " + ", ".join(f"{k}: {v}" for k, v in by_source["all"]["demos"].items())
        ]
    timings = ", ".join(f"{k} {v} s" for k, v in stats.get("timings", {}).items())
    lines += ["", f"Seconds: build {stats['seconds']}; {timings or 'no fetch timings'}.", ""]
    return "\n".join(lines)


# Writing a build.


def _write_json(path: Path, obj: Any, indent: int | None = None) -> dict[str, Any]:
    if indent is None:
        data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    else:
        data = (json.dumps(obj, indent=indent, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def environment() -> dict[str, Any]:
    """Describe the interpreter and the Kaggle image for the manifest (note, 3.1).

    Returns:
        ``python``, ``numpy``, ``image`` (``KAGGLE_DOCKER_IMAGE``), and the image's
        ``/etc/git_commit`` and ``/etc/build_date`` (None where absent).
    """
    files = {}
    for name in ("/etc/git_commit", "/etc/build_date"):
        try:
            files[Path(name).name] = Path(name).read_text(encoding="utf-8").strip()[:200]
        except OSError:
            files[Path(name).name] = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "image": os.environ.get("KAGGLE_DOCKER_IMAGE"),
        **files,
    }


def write_build(
    result: BuildResult,
    out: Path,
    cfg: Mapping[str, Any],
    inputs: Mapping[str, Any],
    code: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a build's files, its manifest, and its run logs.

    ``MANIFEST.json`` holds no timing or path, so the same code, config, and inputs give the same
    bytes; downstream notebooks bind to its sha256. Run logs (statistics, a copy of the manifest)
    go to ``run_logs/``, which ``kaggle_ops.py pull`` brings back; the episodes never match the
    pull allowlist.

    Args:
        result: The build.
        out: Output root (``/kaggle/working`` on Kaggle).
        cfg: Resolved config.
        inputs: Verified source pins (``fetch_sources``).
        code: ``commit`` and ``dirty`` of the code that built it.

    Returns:
        The manifest.
    """
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for split, records in (("train", result.train), ("val", result.val), ("toy", result.toy)):
        files += write_shards(records, out, split, cfg["shards"][split])
    puzzles = sorted(result.puzzles, key=lambda p: p.puzzle_id)
    extra = {
        "val/val_challenges.json": {p.puzzle_id: p.challenge() for p in puzzles},
        "val/val_solutions.json": {p.puzzle_id: p.solution() for p in puzzles},
        "sources/arc2train_pair_sets.json": result.arc2train_digests,
    }
    for rel, obj in extra.items():
        files.append({"path": rel, **_write_json(out / rel, obj)})
    manifest = {
        "schema": SCHEMA,
        "task": TASK,
        "code": dict(code),
        "config_sha256": config_sha256(cfg),
        "config": cfg,
        "inputs": inputs,
        "environment": environment(),
        "augmentation": {
            "id": "8 * j + 4 * t + k",
            "ops": "transpose if t, then np.rot90 k times, then color c -> permutations[j][c]",
            "permutations": result.table.to_strings(),
        },
        "max_tokens": cfg["max_tokens"],
        "counts": {
            "train": len(result.train),
            "val": len(result.val),
            "toy": len(result.toy),
            "val_puzzles": len(result.puzzles),
        },
        "provenance": result.provenance,
        "files": files,
    }
    written = _write_json(out / MANIFEST, manifest, indent=1)
    (out / LOG_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / MANIFEST, out / LOG_DIR / MANIFEST)
    stats = {
        **result.stats,
        "manifest_sha256": written["sha256"],
        "counts": manifest["counts"],
        "lengths": {
            split: split_stats(records, cfg["max_tokens"])
            for split, records in (
                ("train", result.train),
                ("val", result.val),
                ("toy", result.toy),
            )
        },
    }
    _write_json(out / LOG_DIR / "episodes_stats.json", stats, indent=1)
    (out / LOG_DIR / "episodes_stats.md").write_text(stats_markdown(stats), encoding="utf-8")
    return manifest


def code_version(commit: str | None) -> dict[str, Any]:
    """Return the code commit and whether the tree differs from it.

    Args:
        commit: The commit a push folder embedded (``<sha>`` or ``<sha>+dirty``), or None to ask
            git in this repository.

    Returns:
        ``commit`` (``unknown`` without git) and ``dirty`` (None when unknown).
    """
    if commit is not None:
        # build_episodes_kernel.py embeds "<sha>+dirty" when it bundled uncommitted files.
        return {"commit": commit.removesuffix("+dirty"), "dirty": commit.endswith("+dirty")}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": head, "dirty": bool(status.strip())}


def run_build(
    cfg: Mapping[str, Any], out: Path, work: Path, code: Mapping[str, Any]
) -> dict[str, Any]:
    """Fetch, load, build, and write: the ``build`` mode.

    Args:
        cfg: Resolved config.
        out: Output root.
        work: Scratch folder for the downloads.
        code: ``code_version`` of the running code.

    Returns:
        The manifest.
    """
    timings = {}
    started = time.perf_counter()
    paths, inputs = fetch_sources(cfg, work)
    timings["fetch"] = time.perf_counter() - started
    sources = load_sources(paths, cfg)
    timings["load"] = time.perf_counter() - started - timings["fetch"]
    result = build_episode_set(cfg, sources)
    result.stats["timings"] = {k: round(v, 1) for k, v in timings.items()}
    return write_build(result, out, cfg, inputs, code)


# Checking a build.


def _walk_files(root: Path) -> Iterator[Path]:
    # Follows links, as Kaggle mounts may be links, but enters each real folder once.
    seen: set[str] = set()
    for folder, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(folder)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        for name in sorted(files):
            yield Path(folder) / name


def locate_build(root: Path) -> Path:
    """Find the one P3.4 build under a root, by its ``MANIFEST.json``.

    Args:
        root: The build folder itself, or a folder above it such as ``/kaggle/input``.

    Returns:
        The folder holding the manifest.

    Raises:
        BuildError: If there is no P3.4 manifest or more than one.
    """
    if (root / MANIFEST).is_file():
        return root
    found = []
    for path in _walk_files(root):
        if path.name == MANIFEST and path.parent.name != LOG_DIR:
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("task") == TASK:
                    found.append(path.parent)
            except (OSError, ValueError):
                continue
    if len(found) != 1:
        raise BuildError(f"expected one {TASK} {MANIFEST} under {root}, found {len(found)}")
    return found[0]


def locate_competition(root: Path) -> dict[str, Path | None]:
    """Find the competition's evaluation and training files under a root.

    Args:
        root: Folder to search, such as ``/kaggle/input``.

    Returns:
        ``eval_challenges``, ``eval_solutions``, ``train_challenges``, ``train_solutions``: the
        copy in the competition's folder if several, else the shortest path; the training files
        may be None.

    Raises:
        BuildError: If an evaluation file is missing.
    """
    names = {
        "eval_challenges": EVAL_CHALLENGES,
        "eval_solutions": EVAL_SOLUTIONS,
        "train_challenges": TRAIN_CHALLENGES,
        "train_solutions": TRAIN_SOLUTIONS,
    }
    found: dict[str, list[Path]] = {key: [] for key in names}
    by_name = {name: key for key, name in names.items()}
    for path in _walk_files(root):
        if path.name in by_name:
            found[by_name[path.name]].append(path)
    chosen: dict[str, Path | None] = {}
    for key, paths in found.items():
        ranked = sorted(paths, key=lambda p: (COMPETITION not in str(p), len(p.parts), str(p)))
        chosen[key] = ranked[0] if ranked else None
    for key in ("eval_challenges", "eval_solutions"):
        if chosen[key] is None:
            raise BuildError(f"no {names[key]} under {root}")
    return chosen


def verify_build(folder: Path) -> tuple[dict[str, Any], str]:
    """Check every file the manifest lists against its size and sha256.

    Args:
        folder: The build folder.

    Returns:
        The manifest and the sha256 of its bytes.

    Raises:
        BuildError: If a file is missing or differs.
    """
    data = (folder / MANIFEST).read_bytes()
    manifest = json.loads(data)
    for entry in manifest["files"]:
        path = folder / entry["path"]
        if not path.is_file():
            raise BuildError(f"{entry['path']} is listed in the manifest but missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise BuildError(f"{entry['path']} differs from the manifest")
    return manifest, hashlib.sha256(data).hexdigest()


def _task_pairs(challenges: Mapping[str, Any], solutions: Mapping[str, Any]) -> dict[str, list]:
    tasks = {}
    for task_id, task in challenges.items():
        pairs = [(validate_grid(p["input"]), validate_grid(p["output"])) for p in task["train"]]
        for pair, output in zip(task["test"], solutions[task_id], strict=True):
            pairs.append((validate_grid(pair["input"]), validate_grid(output)))
        tasks[task_id] = pairs
    return tasks


def _report_r3(folder: Path, paths: Mapping[str, Path | None]) -> dict[str, Any]:
    # Report R3 is informational, so a problem with the training files is recorded, not raised.
    if paths["train_challenges"] is None or paths["train_solutions"] is None:
        return {"available": False, "error": "competition training files not found"}
    try:
        theirs = pair_set_digests(
            _task_pairs(
                json.loads(paths["train_challenges"].read_text(encoding="utf-8")),
                json.loads(paths["train_solutions"].read_text(encoding="utf-8")),
            )
        )
        ours = json.loads((folder / "sources" / "arc2train_pair_sets.json").read_text("utf-8"))
    except (OSError, KeyError, TypeError, ValueError) as err:
        return {"available": False, "error": type(err).__name__}
    return {"available": True, **compare_pair_sets(ours, theirs)}


def run_check(
    cfg: Mapping[str, Any], episodes: Path, competition: Path, out: Path, code: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the leakage check of a build against the evaluation set: the ``check`` mode.

    Reads the evaluation solutions, as BRIEF A0 allows for this check only; writes counts and
    ids, never a grid.

    Args:
        cfg: Resolved config (gate thresholds, number of positive controls, seed).
        episodes: The build folder or a root above it.
        competition: Folder holding the competition files, or a root above it.
        out: Output root; the report goes to ``run_logs/leakage.{json,md}``.
        code: ``code_version`` of the running code.

    Returns:
        The report: ``leakage_report`` plus ``manifest_sha256``, ``build``, ``positive_control``,
        ``R3``, and ``pass`` (gates clean and the control caught).

    Raises:
        BuildError: If the build or the evaluation files cannot be found or verified.
    """
    started = time.perf_counter()
    folder = locate_build(episodes)
    manifest, manifest_sha = verify_build(folder)
    table = PermutationTable.from_strings(manifest["augmentation"]["permutations"])
    paths = locate_competition(competition)
    challenges = json.loads(paths["eval_challenges"].read_text(encoding="utf-8"))
    solutions = json.loads(paths["eval_solutions"].read_text(encoding="utf-8"))
    limits = {"n_augmentations": table.n_augmentations, "max_tokens": manifest["max_tokens"]}
    shards = [e for e in manifest["files"] if e["path"].startswith("episodes/")]
    by_shard = [list(read_shards([folder / e["path"]], **limits)) for e in shards]
    records = [record for shard in by_shard for record in shard]
    if len(records) != sum(manifest["counts"][s] for s in SPLITS):
        raise BuildError("the shards hold a different number of records than the manifest")
    gates = cfg["gates"]
    thresholds = {"min_cells": gates["min_cells"], "min_colors": gates["min_colors"]}
    seed_ids = {"barc": manifest["provenance"]["barc_seed_ids"]}
    report = leakage_report(records, challenges, solutions, seed_ids=seed_ids, **thresholds)

    # The control goes into an in-memory copy of the first training shard (note, 2.3, row P).
    controls = positive_control(
        challenges, solutions, n=gates["positive_controls"], seed=cfg["seed"]
    )
    first = next(shard for e, shard in zip(shards, by_shard, strict=True) if e["split"] == "train")
    injected = [*first, *(record for record, _ in controls)]
    control_report = leakage_report(injected, challenges, solutions, **thresholds)
    report["positive_control"] = control_outcome(control_report, controls)

    report["reports"]["R3"] = _report_r3(folder, paths)
    report["gates_pass"] = report["pass"]
    report["pass"] = report["gates_pass"] and report["positive_control"]["pass"]
    report["manifest_sha256"] = manifest_sha
    report["build"] = {
        "code": manifest["code"],
        "config_sha256": manifest["config_sha256"],
        "counts": manifest["counts"],
    }
    report["check"] = {
        "code": dict(code),
        "config_sha256": config_sha256(cfg),
        "environment": environment(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    log_dir = out / LOG_DIR
    _write_json(log_dir / "leakage.json", report, indent=1)
    (log_dir / "leakage.md").write_text(leakage_markdown(report), encoding="utf-8")
    return report


GATE_RULES = {
    "G1": "source task id or family is an evaluation task id",
    "G2": "exact pair hash equals an evaluation pair's",
    "G3": "canonical pair hash equals an evaluation pair's",
    "G4": "large input grid equals an evaluation test input (exact or canonical)",
    "G5": "large output grid equals an evaluation test output (exact or canonical)",
}


def leakage_markdown(report: Mapping[str, Any]) -> str:
    """Render a check report as Markdown: verdict, gates, control, reports, drop list, hits.

    Args:
        report: ``run_check``'s report.

    Returns:
        The Markdown text; ids and counts only.
    """
    verdict = "PASS" if report["pass"] else "FAIL"
    lines = [
        "# P3.4 leakage check",
        "",
        f"**{verdict}** for manifest `{report['manifest_sha256']}`.",
        "",
    ]
    ev = report["eval"]
    lines += [
        f"Evaluation set: {ev.get('tasks', 0)} tasks, {ev.get('train_pairs', 0)} demonstration "
        f"pairs, {ev.get('test_pairs', 0)} test pairs.",
        "",
    ]
    lines += ["## Gates", ""]
    lines += _table(
        ("gate", "rule", "hits", "source tasks", "by source"),
        (
            (g, GATE_RULES[g], v["hits"], v["tasks"], v["by_source"] or "")
            for g, v in report["gates"].items()
        ),
    )
    pc = report["positive_control"]
    lines += [
        "",
        "## Positive control",
        "",
        f"{pc['injected']} evaluation test pairs injected under a random dihedral map and color "
        f"permutation: G3 caught {pc['caught_g3']}, G2 caught {pc['caught_g2']}; "
        f"{'pass' if pc['pass'] else 'FAIL'}.",
        "",
    ]
    lines += ["## R1: grid matches with evaluation demonstration grids (count only)", ""]
    r1 = report["reports"]["R1"]
    buckets = sorted({b for kind in r1.values() for b in kind}, key=lambda b: int(b.split("-")[0]))
    lines += _table(
        ("cells", "exact", "canonical"),
        ((b, r1["exact"].get(b, 0), r1["canonical"].get(b, 0)) for b in buckets),
    )
    so = report["reports"]["split_overlap"]
    lines += [
        "",
        f"Split overlap: {so['val_families_in_training']} validation families in training; "
        f"{so['training_tasks_sharing_a_val_pair']} training tasks share a canonical pair with "
        "validation.",
        "",
    ]
    r3 = report["reports"]["R3"]
    lines += ["## R3: ARC-AGI-2 training, GitHub copy vs competition copy", ""]
    if r3.get("available"):
        lines += [
            f"{r3['common']} common tasks ({r3['only_ours']} only in ours, {r3['only_theirs']} "
            f"only in theirs); equal exact pair sets {r3['equal_exact']}, equal canonical "
            f"{r3['equal_canonical']}; differing: {', '.join(r3['canonical_differs']) or 'none'}.",
            "",
        ]
    else:
        lines += ["Competition training files not found; skipped.", ""]
    if report["drop"]:
        lines += ["## Drop list to add to configs/episodes.yaml", "", "```yaml"]
        lines += [
            f'- {{source: {d["source"]}, task_id: "{d["task_id"]}", reason: "leakage check"}}'
            for d in report["drop"]
        ]
        lines += ["```", ""]
    if report["hits"]:
        lines += [f"## Hits (first {MD_HIT_ROWS} of {len(report['hits'])}; ids only)", ""]
        keys = ("gate", "source", "task_id", "split", "eval_task_id", "match")
        hit_rows = ([h[k] for k in keys] for h in report["hits"][:MD_HIT_ROWS])
        lines += _table(keys, hit_rows)
        lines += [""]
    return "\n".join(lines)


# The smoke run: synthetic sources, a synthetic evaluation set, planted leaks.

SMOKE_EVAL_IDS = [f"e1{i:06x}" for i in range(6)]
SMOKE_ARC_IDS = [f"a1{i:06x}" for i in range(11)]
SMOKE_REARC_IDS = [f"a1{i:06x}" for i in range(6)] + [f"b1{i:06x}" for i in range(4)]
# The smoke must flag exactly these tasks with exactly these gates in its first check.
SMOKE_PLANTED = {
    ("arc2train", "e1000005"): {"G1"},
    ("arc2train", "a1000003"): {"G2", "G3"},
    ("arc2train", "a1000006"): {"G4"},
    ("arc2train", "a1000007"): {"G5"},
    ("barc", "seed:e1000004"): {"G1"},
}
_SMOKE_GIT_ENV = {
    "GIT_AUTHOR_NAME": "arc26 smoke",
    "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
    "GIT_AUTHOR_DATE": "2025-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "arc26 smoke",
    "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
    "GIT_COMMITTER_DATE": "2025-01-01T00:00:00Z",
}


def _grid(rng: random.Random, rows: int, cols: int, colors: int = 10) -> list[list[int]]:
    return [[rng.randrange(colors) for _ in range(cols)] for _ in range(rows)]


def _side(rng: random.Random, low: int = 4, high: int = 10) -> int:
    return rng.randint(low, high)


def _pair(rng: random.Random, low: int = 4, high: int = 10) -> dict[str, Any]:
    return {
        "input": _grid(rng, _side(rng, low, high), _side(rng, low, high)),
        "output": _grid(rng, _side(rng, low, high), _side(rng, low, high)),
    }


def _moved(
    grid: list[list[int]], transpose: bool, turns: int, perm: Sequence[int]
) -> list[list[int]]:
    return Augmentation(transpose, turns, tuple(perm)).apply(grid).tolist()


def smoke_evaluation(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the synthetic evaluation set of the smoke run (never real data).

    Six tasks; every grid has sides 4 to 10, except the test input of the fifth task, which is
    4 by 4 (16 cells, under the G4 floor) for a negative control.

    Args:
        rng: Generator of the grids.

    Returns:
        Challenges and solutions in the competition's format.
    """
    challenges, solutions = {}, {}
    for i, task_id in enumerate(SMOKE_EVAL_IDS):
        train = [_pair(rng) for _ in range(2 + i % 2)]
        tests = [_pair(rng, 5, 10) for _ in range(1 + (i % 3 == 0))]
        if i == 4:
            tests[0]["input"] = _grid(rng, 4, 4)
        challenges[task_id] = {"train": train, "test": [{"input": t["input"]} for t in tests]}
        solutions[task_id] = [t["output"] for t in tests]
    return challenges, solutions


def _git_repo(folder: Path, files: Mapping[str, str]) -> str:
    folder.mkdir(parents=True)
    for rel, text in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_text(text, encoding="utf-8")
    env = {
        **os.environ,
        **_SMOKE_GIT_ENV,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for args in (
        ("init", "--quiet", "--initial-branch=main"),
        ("config", "uploadpack.allowFilter", "true"),
        ("config", "uploadpack.allowAnySHA1InWant", "true"),
        ("add", "--all"),
        ("commit", "--quiet", "--message", "smoke"),
    ):
        subprocess.run(["git", *args], cwd=folder, env=env, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=folder,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _zip_bytes(members: Mapping[str, str]) -> bytes:
    path = Path(tempfile.mkstemp(suffix=".zip")[1])
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, text in members.items():
                archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), text)
        return path.read_bytes()
    finally:
        path.unlink()


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def smoke_sources(root: Path, seed: int = 0) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Write synthetic sources in the real formats, with planted leaks, and their pins.

    - ARC-AGI-2: a local git repository with ``data/training`` (11 tasks plus ``e1000005``, an
      evaluation id: G1), ``data/training.txt``, and ``data/evaluation`` files that the fetch
      must never download (G0). ``a1000003`` holds an evaluation demonstration pair (G2, G3),
      ``a1000006`` a moved evaluation test input (G4), ``a1000007`` a moved test output (G5), and
      ``a1000008`` a moved test input under the G4 floor and a one-cell edit of an evaluation pair
      (negative controls).
    - ReARC: ``re_arc.zip`` in the real layout (``re_arc/tasks``, ``metadata.json``, a
      ``__MACOSX`` member), 10 generators of 40 examples, one example 31 wide.
    - BARC: a JSONL of 30 problems plus a duplicate program, a malformed line, float cells, a
      non-integral cell, a problem seeded by an evaluation id (G1 on ``seed:e1000004``), and a
      two-pair problem holding a moved evaluation test pair (G3, G4, G5).
    - The competition folder: the evaluation set and ARC-AGI-2 training with one task changed (R3).

    Args:
        root: Empty folder to write into.
        seed: Seed of the synthetic grids.

    Returns:
        The config's ``sources`` section (``file://`` URLs and the pins of these files), the
        competition folder, and the planted tasks with their expected gates.
    """
    rng = random.Random(seed)
    challenges, solutions = smoke_evaluation(rng)
    perm = (3, 8, 1, 0, 2, 4, 5, 6, 7, 9)
    arc: dict[str, dict[str, Any]] = {}
    for task_id in [*SMOKE_ARC_IDS, "e1000005"]:
        n_train, n_test = rng.randint(2, 4), rng.randint(1, 2)
        arc[task_id] = {
            "train": [_pair(rng) for _ in range(n_train)],
            "test": [_pair(rng) for _ in range(n_test)],
        }
    e0, e1, e2, e3, e4 = (challenges[t] for t in SMOKE_EVAL_IDS[:5])
    arc["a1000003"]["train"].append(copy.deepcopy(e0["train"][0]))
    arc["a1000006"]["train"].append(
        {"input": _moved(e2["test"][0]["input"], True, 2, perm), "output": _grid(rng, 5, 5)}
    )
    arc["a1000007"]["train"].append(
        {"input": _grid(rng, 5, 5), "output": _moved(solutions["e1000003"][0], False, 3, perm)}
    )
    edited = copy.deepcopy(e0["train"][1])
    edited["output"][0][0] = (edited["output"][0][0] + 1) % 10
    arc["a1000008"]["train"] += [
        {"input": _moved(e4["test"][0]["input"], False, 1, perm), "output": _grid(rng, 5, 5)},
        edited,
    ]
    files = {f"data/training/{t}.json": json.dumps(task) for t, task in arc.items()}
    files["data/training.txt"] = "\n".join(sorted(arc))
    files["data/evaluation.txt"] = "\n".join(SMOKE_EVAL_IDS)
    files |= {f"data/evaluation/{t}.json": json.dumps(challenges[t]) for t in SMOKE_EVAL_IDS}
    files["README.md"] = "synthetic ARC-AGI-2 for the P3.4 smoke run\n"
    repo = root / "ARC-AGI-2-src"
    commit = _git_repo(repo, files)

    def object_id(spec: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", spec], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    members = {"re_arc/metadata.json": "{}", "__MACOSX/re_arc/tasks/._a1000000.json": "junk"}
    for gen in SMOKE_REARC_IDS:
        examples = [_pair(rng, 3, 9) for _ in range(40)]
        if gen == "b1000003":
            examples[0]["input"] = _grid(rng, 4, 31)
        members[f"re_arc/tasks/{gen}.json"] = json.dumps(examples)
    rearc_bytes = _zip_bytes(members)
    rearc_commit = hashlib.sha1(b"smoke-rearc", usedforsecurity=False).hexdigest()
    (root / rearc_commit).mkdir()
    (root / rearc_commit / "re_arc.zip").write_bytes(rearc_bytes)

    lines = []
    for i in range(30):
        examples = [
            [p["input"], p["output"]] for p in (_pair(rng, 3, 8) for _ in range(rng.randint(4, 12)))
        ]
        lines.append(
            {
                "source": f"def main(grid):  # smoke program {i}\n    return grid\n",
                "examples": examples,
                "seeds": ["a1000002.py", "a free-text seed"],
            }
        )
    lines[3]["examples"] = [
        [[[float(c) for c in row] for row in x], y] for x, y in lines[3]["examples"]
    ]
    lines[4]["examples"][0][0][0][0] = 0.5
    lines[5]["seeds"] = ["e1000004.py"]  # an evaluation id as a seed: G1 on provenance
    leak = (
        _moved(e1["test"][0]["input"], True, 1, perm),
        _moved(solutions["e1000001"][0], True, 1, perm),
    )
    planted = {
        "source": "def main(grid):  # smoke program with a leak\n",
        "seeds": [],
        "examples": [list(leak), [_grid(rng, 5, 5), _grid(rng, 5, 5)]],
    }
    text = "\n".join(
        json.dumps(row) for row in [*lines, planted, {**lines[0], "examples": lines[1]["examples"]}]
    )
    text += "\nnot json\n"
    barc_bytes = text.encode("utf-8")
    revision = hashlib.sha1(b"smoke-barc", usedforsecurity=False).hexdigest()
    (root / revision).mkdir()
    (root / revision / "barc.jsonl").write_bytes(barc_bytes)

    competition = root / "competition" / COMPETITION
    competition.mkdir(parents=True)
    train_tasks = {t: task for t, task in arc.items() if t != "e1000005"}
    changed = copy.deepcopy(train_tasks)
    changed["a1000001"]["test"][0]["output"][0][0] = (
        changed["a1000001"]["test"][0]["output"][0][0] + 1
    ) % 10
    for name, obj in (
        (EVAL_CHALLENGES, challenges),
        (EVAL_SOLUTIONS, solutions),
        (
            TRAIN_CHALLENGES,
            {
                t: {"train": v["train"], "test": [{"input": p["input"]} for p in v["test"]]}
                for t, v in changed.items()
            },
        ),
        (TRAIN_SOLUTIONS, {t: [p["output"] for p in v["test"]] for t, v in changed.items()}),
    ):
        (competition / name).write_text(json.dumps(obj), encoding="utf-8")

    sources = {
        "arc2train": {
            "repo": repo.as_uri(), "commit": commit, "tree": object_id("HEAD:data/training"),
            "index_blob": object_id("HEAD:data/training.txt"), "tasks": len(arc),
        },
        "rearc": {
            "url": (root / rearc_commit / "re_arc.zip").as_uri(), "commit": rearc_commit,
            "blob": _git_blob(rearc_bytes), "sha256": hashlib.sha256(rearc_bytes).hexdigest(),
            "bytes": len(rearc_bytes), "tasks": len(SMOKE_REARC_IDS),
        },
        "barc": {
            "url": (root / revision / "barc.jsonl").as_uri(), "revision": revision,
            "sha256": hashlib.sha256(barc_bytes).hexdigest(), "bytes": len(barc_bytes),
            "rows": len(lines) + 3, "name": "smoke", "max_pairs": 9,
        },
    }  # fmt: skip
    expected = {
        **SMOKE_PLANTED,
        ("barc", barc_task_id("smoke", planted["source"])): {"G3", "G4", "G5"},
    }
    return sources, competition, expected


def smoke_config(config: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return the config with its ``smoke`` section applied (sources still the real pins).

    Args:
        config: The YAML config.

    Returns:
        The resolved smoke config; ``smoke_sources`` supplies its ``sources``.
    """
    return read_config(config, smoke=True)


def smoke(out: Path, config: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Run build, check, drop, rebuild, and check on synthetic data, and test the outcomes.

    Args:
        out: Empty or missing folder for all smoke files.
        config: The YAML config whose ``smoke`` section sizes the run.

    Returns:
        ``ok`` (every expectation held), ``failures`` (what did not), the two reports' verdicts,
        flagged tasks, and the two manifests' sha256.
    """
    cfg = smoke_config(config)
    sources, competition, expected = smoke_sources(out / "sources", cfg["seed"])
    cfg["sources"] = sources
    validate_config(cfg)
    code = {"commit": "smoke", "dirty": None}
    failures: list[str] = []

    def expect(condition: bool, what: str) -> None:
        if not condition:
            failures.append(what)

    run_build(cfg, out / "build-1", out / "work-1", code)
    first = run_check(cfg, out / "build-1", out / "sources" / "competition", out / "check-1", code)
    flagged: dict[tuple[str, str], set[str]] = {}
    for hit in first["hits"]:
        flagged.setdefault((hit["source"], hit["task_id"]), set()).add(hit["gate"])
    expect(not first["gates_pass"], "first check must fail on the planted leaks")
    expect(
        flagged == expected,
        f"flagged {sorted(flagged.items())}, expected {sorted(expected.items())}",
    )
    expect(first["positive_control"]["pass"], "positive control must pass")
    expect(first["positive_control"]["caught_g2"] == 0, "the moved control pairs must escape G2")
    expect(
        first["reports"]["R3"].get("canonical_differs") == ["a1000001"],
        "R3 must find the changed task",
    )
    expect(
        not (out / "work-1" / "ARC-AGI-2" / "data" / "evaluation").exists(),
        "G0: no evaluation file",
    )

    cfg2 = copy.deepcopy(cfg)
    cfg2["drop"] = [{**d, "reason": "smoke leakage check"} for d in first["drop"]]
    validate_config(cfg2)
    second_manifest = run_build(cfg2, out / "build-2", out / "work-2", code)
    second = run_check(
        cfg2, out / "build-2", out / "sources" / "competition", out / "check-2", code
    )
    expect(second["pass"], f"second check must pass, gates {second['gates']}")
    expect(
        second["reports"]["split_overlap"]["val_families_in_training"] == 0,
        "families cross the split",
    )
    expect(all(second_manifest["counts"][s] > 0 for s in SPLITS), "every split must hold episodes")
    return {
        "ok": not failures,
        "failures": failures,
        "first": {"pass": first["pass"], "flagged": sorted(f"{s}/{t}" for s, t in flagged)},
        "second": {"pass": second["pass"], "counts": second_manifest["counts"]},
        "manifests": [first["manifest_sha256"], second["manifest_sha256"]],
    }


# Command line.


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser: ``build``, ``check``, or ``--smoke``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML or JSON config")
    parser.add_argument("--smoke", action="store_true", help="synthetic end-to-end run, offline")
    parser.add_argument("--out", type=Path, help="smoke output folder (default: a temporary one)")
    sub = parser.add_subparsers(dest="mode")
    build = sub.add_parser("build", help="fetch the sources and write the episodes")
    check = sub.add_parser("check", help="run the leakage gates against the evaluation set")
    for mode in (build, check):
        # SUPPRESS keeps a --config given before the mode name.
        mode.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        mode.add_argument("--out", type=Path, required=True, help="output root")
        mode.add_argument("--code-commit", help="commit of the code, set by the push folder")
    build.add_argument("--work", type=Path, required=True, help="scratch folder for the sources")
    check.add_argument("--episodes", type=Path, required=True, help="build folder or a root above")
    check.add_argument("--competition", type=Path, required=True, help="competition folder or root")
    return parser


def _log_error(out: Path | None) -> None:
    text = traceback.format_exc()
    print(text, file=sys.stderr, flush=True)
    if out is not None:
        (out / LOG_DIR).mkdir(parents=True, exist_ok=True)
        (out / LOG_DIR / "error.log").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run one mode.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success (for ``check``, whenever the report is written: its ``pass`` field carries
        the verdict), 1 on a failure (``run_logs/error.log`` written) or a failed smoke
        expectation, 2 when the config or arguments are refused.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke == (args.mode is not None):
        parser.print_usage(sys.stderr)
        print("give exactly one of --smoke, build, check", file=sys.stderr)
        return EXIT_CONFIG
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="arc26-p3-4-smoke-") as tmp:
            out = args.out or Path(tmp)
            try:
                summary = smoke(out, args.config)
            except (
                ConfigError,
                PinError,
                BuildError,
                OSError,
                ValueError,
                subprocess.SubprocessError,
            ):
                _log_error(out)
                return EXIT_FAILED
        print(json.dumps(summary, indent=1))
        return EXIT_OK if summary["ok"] else EXIT_FAILED
    try:
        cfg = read_config(args.config)
    except ConfigError as err:
        print(f"refused: {err}", file=sys.stderr)
        return EXIT_CONFIG
    code = code_version(args.code_commit)
    try:
        if args.mode == "build":
            manifest = run_build(cfg, args.out, args.work, code)
            print(f"build written: {manifest['counts']}", flush=True)
        else:
            report = run_check(cfg, args.episodes, args.competition, args.out, code)
            print(
                f"check {'PASS' if report['pass'] else 'FAIL'}: source tasks per gate "
                f"{ {g: v['tasks'] for g, v in report['gates'].items()} }, positive control "
                f"{report['positive_control']}",
                flush=True,
            )
    except (ConfigError, PinError, BuildError, OSError, ValueError):
        _log_error(args.out)
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
