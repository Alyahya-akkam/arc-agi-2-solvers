r"""H's validation inputs: GEN, the held-out generated set, and H's load check (P3.7, ADR-019).

Gate G4 (BRIEF B7, B5 P3.7) compares four adapter sources on two sets. This module builds the
generated one and checks, before any decode, that the H the run loads is the H that meta-training
validated.

**GEN.** P3.4's validation split (build 2: 286 episodes of 200 puzzles, BARC 100 problems, ReARC
20 held-out families of 5 puzzles) holds one episode per test pair of a puzzle, each with all the
puzzle's demonstrations, un-augmented. ``gen_puzzles`` takes one episode per puzzle, the lowest
episode index, and makes it a competition-format puzzle: the episode's demonstrations as ``train``
(the solver applies its own views) and its query as the one test pair, whose output is the answer,
the held-out pair. It keeps a fixed stratified subset: the first ``barc`` BARC puzzles by sorted
id, and ``rearc`` ReARC puzzles dealt round-robin over the held-out families in sorted order (each
family's puzzles by sorted id), so 50 of 20 families give 3 to the first 10 families and 2 to the
rest. ``write_gen`` writes them under the competition's file names, so the solver and the
reference's selection read them unchanged.

**Load check.** H from ``best.pt`` must reproduce the validation CE that ``MT-02`` logged at its
best point, per episode, through ``arc26.hyper.train``'s own validation path: P3.4's verified
build (``open_episode_set``, ``load_split``), ``build_sequence`` with each episode's augmentation,
the frozen base from ``load_base`` with ``wrap_targets``, $\bar z$ from the embedding cache
(``PairEmbeddings`` in record order, as training takes it), H from the checkpoint, and the query CE
of ``_Worker.episode_ce`` (one forward under ``inject``). With $c_i$ the recomputed and $\ell_i$ the
logged CE of episode $i$ of $n$, the check passes when

$$\max_i |c_i - \ell_i| \le \epsilon_{\max} \quad\text{and}\quad
\Big|\frac{1}{n}\sum_i (c_i - \ell_i)\Big| \le \epsilon_{\text{mean}},$$

for the current H and for the zero adapter (the base), and fails otherwise (wrong checkpoint,
config, injection, or base). It also measures the bf16 cast the solver applies to H's adapter:
$d_i = \mathrm{CE}^{\text{bf16}}_i - \mathrm{CE}^{\text{fp32}}_i$ with the factors cast to bf16
and the delta computed in bf16 (PEFT's path on bf16 weights, ``inject(dtype=torch.bfloat16)``);
the report flags it when $\bar d > 2\,\mathrm{SE}(d)$ (the H arms are then a lower bound). On four
DEV4 puzzles it emits H's adapter three times through ``HyperProvider`` (a fresh cache, another
fresh cache, and a fresh cache with the pairs shuffled) and compares the sha256 digests (B5 P4.1's
determinism criterion); a mismatch is recorded, not a stop, since every arm of a run reads one
cache. The hypernet section of the checkpoint's config must equal the build's
``configs/hypernet.yaml``, and its state dict must load strictly (every key and shape).

CLI (run by ``arc26.arms`` before its workers start)::

    python -m arc26.hyper.validation check --config <json>   # exit 0 pass, 6 fail, 1 error
    python -m arc26.hyper.validation gen --config <json>     # exit 0, or 1 on an error

See ``docs/modules/validation.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc26.data import DEV4
from arc26.hyper.episodes import EpisodeRecord

# The competition's file names, so the solver (arc26.solver.load) and the reference's selection
# (arc26.nvarc.select.write_submission) read GEN as they read the evaluation set.
CHALLENGES_FILE = "arc-agi_evaluation_challenges.json"
SOLUTIONS_FILE = "arc-agi_evaluation_solutions.json"
INDEX_FILE = "gen_puzzles.json"
# P3.4's validation episode ids: <source>/<puzzle id>/val/<index of the test pair>.
EPISODE_ID = re.compile(r"(?P<source>[a-z0-9]+)/(?P<puzzle>.+)/val/(?P<index>\d+)")
BARC, REARC = "barc", "rearc"
# A puzzle key holds letters, digits, and "-" only: the solver splits view keys on "_" and ".".
_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9-]")
# Exit codes of the CLI.
EXIT_PASS, EXIT_ERROR, EXIT_FAIL = 0, 1, 6
# The load check's references, and its timing ops.
ZERO, FP32, BF16 = "zero", "fp32", "bf16"
LOAD_OP, EMBED_OP, EMIT_OP = "load", "embed", "dev4_emit"
# The flag of the bf16 cast (task brief amendment 3): the bf16 CE above the float32 CE by more
# than this many paired standard errors.
BF16_FLAG_Z = 2.0


@dataclass(frozen=True)
class GenPuzzle:
    """One GEN puzzle: a P3.4 validation episode as a competition-format puzzle.

    Attributes:
        key: The puzzle's key in GEN's files: the P3.4 puzzle id, any character other than
            letters, digits, and ``-`` replaced by ``-``.
        puzzle_id: The P3.4 puzzle id (``barc-<problem>``, ``rearc-<generator>-<i>``).
        source: ``barc`` or ``rearc``.
        family: The held-out family (a ReARC generator, a BARC problem).
        task_id: The source task.
        episode_id: The episode taken, the lowest index of the puzzle.
        train: The demonstrations, un-augmented, ``{"input", "output"}`` grids as lists.
        test_input: The query's input.
        test_output: The query's output, the answer (the held-out pair's).
        tokens: The episode's tokens in the notebook format (P3.4's count).
    """

    key: str
    puzzle_id: str
    source: str
    family: str
    task_id: str
    episode_id: str
    train: tuple[dict[str, list[list[int]]], ...]
    test_input: list[list[int]]
    test_output: list[list[int]]
    tokens: int

    def challenge(self) -> dict[str, Any]:
        """Return the challenges-file entry: the demonstrations and the test input."""
        return {"train": [dict(pair) for pair in self.train], "test": [{"input": self.test_input}]}

    def solution(self) -> list[list[list[int]]]:
        """Return the solutions-file entry: the answer grid, one per test input."""
        return [self.test_output]

    def index(self) -> dict[str, Any]:
        """Return the puzzle's line of ``gen_puzzles.json``: provenance and size, no grids."""
        return {
            "key": self.key,
            "puzzle_id": self.puzzle_id,
            "source": self.source,
            "family": self.family,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "n_demos": len(self.train),
            "tokens": self.tokens,
        }


def _grid(array: Any) -> list[list[int]]:
    return [[int(v) for v in row] for row in array.tolist()]


def _puzzle_of(record: EpisodeRecord) -> tuple[str, int]:
    match = EPISODE_ID.fullmatch(record.episode_id)
    if match is None or match.group("source") != record.source:
        raise ValueError(f"{record.episode_id}: not a validation episode id <source>/<id>/val/<i>")
    return match.group("puzzle"), int(match.group("index"))


def gen_puzzles(records: Iterable[EpisodeRecord], *, barc: int, rearc: int) -> list[GenPuzzle]:
    """Make GEN from P3.4's validation episodes: one per puzzle, a fixed stratified subset.

    Why: the brief's GEN (amendment 1): the lowest episode of each puzzle, its query the test
    pair; ``barc`` BARC puzzles by sorted id and ``rearc`` ReARC puzzles balanced over the
    held-out families (round-robin over the sorted families, each family's puzzles by sorted id),
    so the subset is a function of the build alone.

    Args:
        records: The validation split's records (``arc26.hyper.train.load_split``).
        barc: BARC puzzles to keep, at least 0.
        rearc: ReARC puzzles to keep, at least 0.

    Returns:
        The puzzles in key order.

    Raises:
        ValueError: If a record is not a validation episode with one query, its source is neither
            ``barc`` nor ``rearc``, a source has fewer puzzles than asked, or two puzzles share a
            key.
    """
    if not (isinstance(barc, int) and isinstance(rearc, int) and barc >= 0 and rearc >= 0):
        raise ValueError(f"barc and rearc must be integers >= 0; got {barc!r}, {rearc!r}")
    lowest: dict[tuple[str, str], tuple[int, EpisodeRecord]] = {}
    for record in records:
        if record.source not in (BARC, REARC):
            raise ValueError(f"{record.episode_id}: GEN takes barc and rearc only")
        if len(record.queries) != 1:
            raise ValueError(f"{record.episode_id}: a validation episode has one query")
        puzzle, index = _puzzle_of(record)
        held = lowest.get((record.source, puzzle))
        if held is None or index < held[0]:
            lowest[(record.source, puzzle)] = (index, record)
    by_source: dict[str, dict[str, EpisodeRecord]] = defaultdict(dict)
    for (source, puzzle), (_, record) in lowest.items():
        by_source[source][puzzle] = record
    barc_ids = sorted(by_source[BARC])
    if len(barc_ids) < barc:
        raise ValueError(f"GEN asks {barc} BARC puzzles; the split has {len(barc_ids)}")
    chosen = [(BARC, puzzle) for puzzle in barc_ids[:barc]]
    families: dict[str, list[str]] = defaultdict(list)
    for puzzle, record in by_source[REARC].items():
        families[record.family].append(puzzle)
    queues = [sorted(families[family]) for family in sorted(families)]
    dealt: list[str] = []
    while len(dealt) < rearc and any(queues):
        for queue in queues:
            if queue and len(dealt) < rearc:
                dealt.append(queue.pop(0))
    if len(dealt) < rearc:
        raise ValueError(f"GEN asks {rearc} ReARC puzzles; the split has {len(dealt)}")
    chosen += [(REARC, puzzle) for puzzle in dealt]
    puzzles = []
    for source, puzzle in chosen:
        record = by_source[source][puzzle]
        (x, y) = record.queries[0]
        puzzles.append(
            GenPuzzle(
                key=_KEY_UNSAFE.sub("-", puzzle),
                puzzle_id=puzzle,
                source=source,
                family=record.family,
                task_id=record.task_id,
                episode_id=record.episode_id,
                train=tuple({"input": _grid(a), "output": _grid(b)} for a, b in record.demos),
                test_input=_grid(x),
                test_output=_grid(y),
                tokens=record.tokens,
            )
        )
    keys = [p.key for p in puzzles]
    if len(set(keys)) != len(keys):
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"two GEN puzzles share a key: {repeated}")
    return sorted(puzzles, key=lambda p: p.key)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def write_gen(puzzles: Sequence[GenPuzzle], out_dir: str | Path) -> dict[str, Any]:
    """Write GEN's challenges, solutions, and index under the competition's file names.

    Args:
        puzzles: The puzzles (``gen_puzzles``).
        out_dir: The folder, created if missing; the solver's ``data_dir`` for a GEN run.

    Returns:
        The summary: puzzle count, per source and per family counts, each file's sha256, and
        the index (each puzzle's provenance, no grids).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        CHALLENGES_FILE: _json_bytes({p.key: p.challenge() for p in puzzles}),
        SOLUTIONS_FILE: _json_bytes({p.key: p.solution() for p in puzzles}),
        INDEX_FILE: (json.dumps([p.index() for p in puzzles], indent=1) + "\n").encode("utf-8"),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)
    sources: dict[str, int] = defaultdict(int)
    families: dict[str, int] = defaultdict(int)
    for p in puzzles:
        sources[p.source] += 1
        if p.source == REARC:
            families[p.family] += 1
    return {
        "puzzles": len(puzzles),
        "by_source": dict(sorted(sources.items())),
        "rearc_families": dict(sorted(families.items())),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "out_dir": str(out),
        "index": [p.index() for p in puzzles],
    }


def gen_from_episodes(
    episodes_root: str | Path,
    check_root: str | Path,
    manifest_sha256: str,
    out_dir: str | Path,
    *,
    barc: int,
    rearc: int,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """Build GEN from a verified P3.4 build: find it, check it, convert, and write it.

    Args:
        episodes_root: Folder searched for the build's ``MANIFEST.json``.
        check_root: Folder searched for its passing leakage check report.
        manifest_sha256: The build's manifest sha256; any other build is refused.
        out_dir: GEN's folder.
        barc: BARC puzzles.
        rearc: ReARC puzzles.
        max_tokens: The notebook's cap; a longer episode raises (P3.4 dropped them).

    Returns:
        ``write_gen``'s summary with the manifest sha256 and the split's episode count.

    Raises:
        arc26.hyper.train.EpisodeSetError: If the build or its check is refused.
        ValueError: As ``gen_puzzles`` raises it.
    """
    from arc26.hyper.train import load_split, open_episode_set

    episodes = open_episode_set(Path(episodes_root), manifest_sha256, Path(check_root), max_tokens)
    records = load_split(episodes, "val", max_tokens)
    summary = write_gen(gen_puzzles(records, barc=barc, rearc=rearc), out_dir)
    summary.update(manifest_sha256=manifest_sha256, val_episodes=len(records))
    return summary


# --- the load check ------------------------------------------------------------------------------


def _mean_se(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n if n else math.nan
    if n < 2:
        return {"mean": mean, "se": 0.0, "n": n}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"mean": mean, "se": math.sqrt(var / n), "n": n}


def bf16_flag(differences: Sequence[float]) -> dict[str, Any]:
    r"""Return the bf16 cast's cost and its flag (task brief amendment 3).

    Why: pre-registered in the briefs: the H arms are a lower bound when the bf16 CE exceeds the
    float32 CE by more than 2 paired standard errors, $\bar d > 2\,\mathrm{SE}(d)$ with
    $\mathrm{SE}(d) = s_d / \sqrt{n}$. A bf16 CE below the float32 CE is never flagged.

    Args:
        differences: $d_i = \mathrm{CE}^{\text{bf16}}_i - \mathrm{CE}^{\text{fp32}}_i$ per episode.

    Returns:
        ``mean``, ``se``, ``n``, ``z`` (``mean / se``, None when the SE is 0), and ``flagged``.
    """
    stats: dict[str, Any] = _mean_se(differences)
    stats["z"] = stats["mean"] / stats["se"] if stats["se"] > 0 else None
    stats["flagged"] = bool(stats["mean"] > BF16_FLAG_Z * stats["se"])
    return stats


def logged_best(run_logs: str | Path) -> dict[str, Any]:
    """Return the validation record at the best point a training run logged.

    Why: ``status.json`` names the best step over all sessions (``best.step``); ``metrics.jsonl``
    holds that validation's per-episode CEs for every reference (``validation_record``).

    Args:
        run_logs: The run's ``run_logs`` folder (``status.json``, ``metrics.jsonl``, optionally
            ``run.json``).

    Returns:
        The ``metrics.jsonl`` record of kind ``val`` at the best step, with ``val_episodes``, the
        episode ids in order from ``run.json`` when it is there (else None).

    Raises:
        FileNotFoundError: If ``status.json`` or ``metrics.jsonl`` is missing.
        ValueError: If no best point is named or no validation record has its step.
    """
    folder = Path(run_logs)
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    best = status.get("best")
    if not isinstance(best, Mapping) or "step" not in best:
        raise ValueError(f"{folder / 'status.json'} names no best point")
    found = None
    with open(folder / "metrics.jsonl", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "val" and record.get("step") == best["step"]:
                found = record
    if found is None:
        raise ValueError(f"no validation record at the best step {best['step']}")
    ids = None
    run = folder / "run.json"
    if run.is_file():
        episodes = json.loads(run.read_text(encoding="utf-8")).get("episodes") or {}
        listed = episodes.get("val_episodes")
        if isinstance(listed, list):
            ids = [entry["episode_id"] for entry in listed]
    return {**found, "val_episodes": ids}


def compare(recomputed: Sequence[float], logged: Sequence[float]) -> dict[str, Any]:
    r"""Compare recomputed and logged per-episode CEs.

    Args:
        recomputed: $c_i$, one per episode.
        logged: $\ell_i$, in the same order.

    Returns:
        ``n``, the means, ``max_abs`` $\max_i |c_i - \ell_i|$, and ``mean_diff``.

    Raises:
        ValueError: If the lengths differ.
    """
    if len(recomputed) != len(logged):
        raise ValueError(f"{len(recomputed)} recomputed and {len(logged)} logged episode CEs")
    diffs = [c - ell for c, ell in zip(recomputed, logged, strict=True)]
    n = len(diffs)
    return {
        "n": n,
        "recomputed_mean": sum(recomputed) / n if n else math.nan,
        "logged_mean": sum(logged) / n if n else math.nan,
        "max_abs": max((abs(d) for d in diffs), default=0.0),
        "mean_diff": sum(diffs) / n if n else math.nan,
    }


def _hypernet_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(value)
    out["target_modules"] = list(out.get("target_modules") or [])
    return out


def dev4_fingerprints(
    hypernet: Any,
    encoder: Any,
    model_id: str,
    tasks: Mapping[str, Mapping[str, Any]],
    seed: int = 0,
    timing_logger: Any = None,
) -> dict[str, Any]:
    """Emit H's adapter three times per puzzle and compare the digests (B5 P4.1's criterion).

    Why: evidence for ADR-017's determinism: a fresh cache, another fresh cache (the pairs encoded
    again), and a fresh cache with the pairs in a shuffled order must give one digest.

    Args:
        hypernet: H, on its device.
        encoder: P3.2's encoder on the frozen base.
        model_id: The cache namespace's model id (the checkpoint's).
        tasks: Challenges entries by task id; only ``train`` and ``test`` inputs are read.
        seed: Seed of the shuffles.
        timing_logger: Receives one ``dev4_emit`` record per emission, or None.

    Returns:
        Per task the three digests and whether they are equal, and ``equal`` over all tasks.
    """
    from arc26.hyper.encoder import EmbeddingCache
    from arc26.hyper.providers import HyperProvider
    from arc26.loop import DemoSet, Pair

    results: dict[str, Any] = {}
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="arc26-dev4-") as root:
        for task_id in sorted(tasks):
            task = tasks[task_id]
            pairs = [Pair(item["input"], item["output"]) for item in task["train"]]
            tests = tuple(item["input"] for item in task["test"])
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            digests = []
            for run, order in enumerate((pairs, pairs, shuffled)):
                cache = EmbeddingCache(
                    Path(root) / f"{task_id}-{run}",
                    model_id=model_id,
                    layer=encoder.layer,
                    pool=encoder.pool,
                    view_set_id=encoder.view_set_id,
                )
                provider = HyperProvider(hypernet, encoder, cache)
                start = time.perf_counter()
                digests.append(provider.emit(DemoSet(task_id, tuple(order), None, tests)).digest())
                if timing_logger is not None:
                    timing_logger.record(task_id, EMIT_OP, time.perf_counter() - start)
            results[task_id] = {"digests": digests, "equal": len(set(digests)) == 1}
    return {"tasks": results, "equal": all(entry["equal"] for entry in results.values())}


def load_check(settings: Mapping[str, Any]) -> dict[str, Any]:
    r"""Run the load check (see the module docstring) and return its report.

    Args:
        settings: ``checkpoint`` (``best.pt``), ``episodes_root``, ``check_root``,
            ``manifest_sha256``, ``run_logs`` (the training run's log folder), ``hypernet`` (the
            expected H config), ``model_dir`` (None keeps the checkpoint's, the tiny stand-in in a
            smoke run), ``cache`` (an embedding cache root or None), ``device`` (``cuda`` or
            ``cpu``), ``max_abs`` and ``max_mean`` ($\epsilon_{\max}$, $\epsilon_{\text{mean}}$),
            ``dev4_challenges`` (a challenges file or None), ``dev4_ids`` (default DEV4), and
            ``timing_dir`` (None records nothing).

    Returns:
        The report: ``verdict`` (``pass`` or ``fail``), ``reasons``, the comparisons for the
        current H and the zero adapter, the bf16 measurement and its flag, H's gain over the zero
        adapter, the DEV4 fingerprints, and the setup.
    """
    import torch

    from arc26.hyper.encoder import EmbeddingCache, PairEncoder, view_descriptors
    from arc26.hyper.hypernet import BaseShapes, Hypernet
    from arc26.hyper.inject import inject, wrap_targets
    from arc26.hyper.train import (
        PairEmbeddings,
        TrainConfig,
        _Comm,
        build_sequence,
        demo_pairs,
        load_base,
        load_checkpoint,
        load_split,
        open_episode_set,
        query_ce,
        resolve_checkpoint,
    )
    from arc26.nvarc.arc_loader import QwenFormatter
    from arc26.timing import TimingLogger

    began = time.perf_counter()
    logger = None
    if settings.get("timing_dir") is not None:
        logger = TimingLogger(settings["timing_dir"], "load-check")
    reasons: list[str] = []
    path = resolve_checkpoint(Path(settings["checkpoint"]))
    payload = load_checkpoint(path)
    config = dict(payload["config"])
    expected = _hypernet_dict(settings["hypernet"])
    if _hypernet_dict(config["hypernet"]) != expected:
        reasons.append(
            f"the checkpoint's hypernet config {config['hypernet']} differs from the build's "
            f"{expected}"
        )
    if config.get("manifest_sha256") != settings["manifest_sha256"]:
        reasons.append(
            f"the checkpoint was validated on manifest {config.get('manifest_sha256')}, the run "
            f"attaches {settings['manifest_sha256']}"
        )
    overrides = {
        "episodes_root": str(settings["episodes_root"]),
        "check_root": str(settings["check_root"]),
        "embedding_cache": None if settings.get("cache") is None else str(settings["cache"]),
        "device": settings["device"],
        "gpus": 1,
    }
    if settings.get("model_dir") is not None:
        overrides["model_dir"] = str(settings["model_dir"])
    with tempfile.TemporaryDirectory(prefix="arc26-load-check-") as scratch:
        overrides.update(out_dir=scratch, ckpt_dir=scratch)
        cfg = TrainConfig.from_dict({**config, **overrides})
        device = torch.device("cuda", 0) if cfg.device == "cuda" else torch.device("cpu")
        episodes = open_episode_set(
            Path(cfg.episodes_root), cfg.manifest_sha256, Path(cfg.check_root), cfg.max_tokens
        )
        records = load_split(episodes, cfg.val_split, cfg.max_tokens)
        start = time.perf_counter()
        model, tokenizer, attn = load_base(cfg, device)
        wrap_targets(model, cfg.hypernet.target_modules)
        if logger is not None:
            logger.record(None, LOAD_OP, time.perf_counter() - start)
        formatter = QwenFormatter(tokenizer)
        sequences = [
            build_sequence(r, episodes.table, formatter, cfg.max_tokens, cfg.loss_includes_eos)
            for r in records
        ]
        hypernet = Hypernet(cfg.hypernet, BaseShapes.from_config(model.config))
        # Strict: every key and shape of the checkpoint's H must be the config's.
        hypernet.load_state_dict(payload["hypernet"])
        hypernet = hypernet.to(device).eval()
        encoder = PairEncoder(
            model,
            tokenizer,
            layer=cfg.encoder.layer,
            pool=cfg.encoder.pool,
            views=view_descriptors(cfg.encoder.n_perm, cfg.encoder.view_seed),
            batch_tokens=cfg.encoder.batch_tokens,
        )
        cache = None
        if cfg.embedding_cache is not None:
            cache = EmbeddingCache(
                cfg.embedding_cache,
                model_id=cfg.model_id,
                layer=cfg.encoder.layer,
                pool=cfg.encoder.pool,
                view_set_id=encoder.view_set_id,
            )
        embeddings = PairEmbeddings()
        start = time.perf_counter()
        # One process: _Comm(1, ...) makes the sharded fill a plain loop over the misses.
        comm = _Comm(1, device)
        chunk = cfg.encoder.chunk_pairs
        embeddings.fill(demo_pairs(records), encoder, cache, comm, 0, chunk, logger)
        if logger is not None:
            logger.record(None, EMBED_OP, time.perf_counter() - start)
        ces: dict[str, list[float]] = {ZERO: [], FP32: [], BF16: []}
        for seq in sequences:
            ids = seq.input_ids.to(device)
            z_bar = embeddings.z_bar(seq.demo_hashes)
            with torch.no_grad():
                factors = hypernet(z_bar.to(device)[None])[0]
                for ref in (ZERO, FP32, BF16):
                    start = time.perf_counter()
                    if ref == ZERO:
                        logits = model(input_ids=ids[None], use_cache=False).logits[0]
                    else:
                        # float32: train.py's path (PEFT's float32 adapter on the bf16 base);
                        # bf16: the factors cast as arc26.solver.load casts the live adapter.
                        dtype = torch.float32 if ref == FP32 else torch.bfloat16
                        with inject(model, factors, cfg.hypernet.scaling, dtype=dtype):
                            logits = model(input_ids=ids[None], use_cache=False).logits[0]
                    ces[ref].append(float(query_ce(logits, ids, seq.target).item()))
                    if logger is not None:
                        logger.record(
                            seq.episode_id,
                            f"validate_{ref}",
                            time.perf_counter() - start,
                            int(seq.input_ids.numel()),
                        )
        best = logged_best(settings["run_logs"])
        order = [seq.episode_id for seq in sequences]
        if best["val_episodes"] is not None and best["val_episodes"] != order:
            reasons.append("the logged validation episodes are not this build's, in this order")
        if payload.get("step") is not None and payload["step"] != best["step"]:
            reasons.append(
                f"the checkpoint holds H at step {payload['step']}, the log's best point is "
                f"step {best['step']}"
            )
        current = compare(ces[FP32], best["per_episode"]["current"])
        zero = compare(ces[ZERO], best["per_episode"]["zero"])
        for name, result in (("current H", current), ("zero adapter", zero)):
            if result["max_abs"] > settings["max_abs"]:
                reasons.append(
                    f"{name}: max per-episode |diff| {result['max_abs']:.3g} above "
                    f"{settings['max_abs']}"
                )
            if abs(result["mean_diff"]) > settings["max_mean"]:
                reasons.append(
                    f"{name}: mean diff {result['mean_diff']:.3g} beyond {settings['max_mean']}"
                )
        bf16 = bf16_flag([b - f for b, f in zip(ces[BF16], ces[FP32], strict=True)])
        gain = _mean_se([f - z for f, z in zip(ces[FP32], ces[ZERO], strict=True)])
        dev4: dict[str, Any] | None = None
        if settings.get("dev4_challenges") is not None:
            data = json.loads(Path(settings["dev4_challenges"]).read_text(encoding="utf-8"))
            ids4 = list(settings.get("dev4_ids") or DEV4)
            dev4 = dev4_fingerprints(
                hypernet,
                encoder,
                cfg.model_id,
                {key: data[key] for key in ids4 if key in data},
                timing_logger=logger,
            )
            dev4["missing"] = [key for key in ids4 if key not in data]
    return {
        "verdict": "fail" if reasons else "pass",
        "reasons": reasons,
        "checkpoint": str(path),
        "checkpoint_step": payload.get("step"),
        "logged_best_step": best["step"],
        "episodes": len(sequences),
        "manifest_sha256": cfg.manifest_sha256,
        "attn_implementation": attn,
        "device": str(device),
        "tolerance": {"max_abs": settings["max_abs"], "max_mean": settings["max_mean"]},
        "current": current,
        "zero": zero,
        "bf16_minus_fp32": bf16,
        "fp32_minus_zero": gain,
        "per_episode": {
            "episode_id": order,
            "zero": [round(v, 6) for v in ces[ZERO]],
            "fp32": [round(v, 6) for v in ces[FP32]],
            "bf16": [round(v, 6) for v in ces[BF16]],
        },
        "embeddings": {
            "pairs": len(embeddings),
            "cache_hits": embeddings.cache_hits,
            "encoded": embeddings.encoded,
        },
        "dev4": dev4,
        "seconds": time.perf_counter() - began,
    }


def _write(path: str | Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1, allow_nan=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``check`` or ``gen`` from a JSON config; write the report to its ``out`` path.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        The exit code: 0 pass (or GEN written), 6 a failed check, 1 an error.
    """
    parser = argparse.ArgumentParser(prog="python -m arc26.hyper.validation")
    parser.add_argument("command", choices=("check", "gen"))
    parser.add_argument("--config", required=True, help="the settings as one JSON object")
    args = parser.parse_args(argv)
    settings = json.loads(args.config)
    out = settings.get("out")
    try:
        if args.command == "gen":
            report = gen_from_episodes(
                settings["episodes_root"],
                settings["check_root"],
                settings["manifest_sha256"],
                settings["out_dir"],
                barc=settings["barc"],
                rearc=settings["rearc"],
                max_tokens=settings.get("max_tokens", 8192),
            )
            _write(out, report)
            print(f"GEN: {report['puzzles']} puzzles {report['by_source']} in {report['out_dir']}")
            return EXIT_PASS
        report = load_check(settings)
    except Exception as err:
        traceback.print_exc()
        _write(out, {"verdict": "error", "error": f"{type(err).__name__}: {err}"})
        return EXIT_ERROR
    _write(out, report)
    current, zero, bf16 = report["current"], report["zero"], report["bf16_minus_fp32"]
    print(
        f"load check: {report['verdict']}; current H max |diff| {current['max_abs']:.3g}, mean "
        f"diff {current['mean_diff']:.3g}; zero max |diff| {zero['max_abs']:.3g}; bf16 minus "
        f"fp32 {bf16['mean']:.3g} (SE {bf16['se']:.3g}, flagged {bf16['flagged']}); DEV4 "
        f"digests equal {None if report['dev4'] is None else report['dev4']['equal']}"
    )
    for reason in report["reasons"]:
        print(f"  {reason}")
    return EXIT_PASS if report["verdict"] == "pass" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
