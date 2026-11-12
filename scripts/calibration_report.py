r"""Calibration of the free confidence signal, the oracle gap, and fixed re-ranking rules (P2.8).

Why: BRIEF B5 P2.8 asks how well the augmentation agreement predicts correctness on DEV120,
separately for the promotion decision (ADR-014) and for attempt selection, and how many tasks the
selection loses although a correct grid was decoded (the oracle gap). The gap gates P4.6, the
learned candidate scorer: it is worth asking for only when the gap is at least 3 tasks on DEV120.

Each run is read from its pulled logs only, through ``scripts/compare_runs.py``'s parsers:
``candidates.json`` (every candidate of every decode view, with its ``beam_score`` and its 8
``score_aug`` rescoring NLLs), the selection output (``selection.log``, or the session log of a
run without one: the ``ALL_CORRECT`` lines, the ``score_kgmon`` correct set, the ``Reload score``
line), and ``submission.json``. No solution file is read and no grid is written: an output's
correct grid is the grid of a candidate the notebook marked ``ALL_CORRECT``.

Each run is first reproduced: from ``candidates.json`` alone, every test output's distinct grids,
their ``score_kgmon`` values (the moved selection, run through ``arc26.loop.tally``), and the two
attempts are rebuilt and must equal ``submission.json``; the score of the rebuilt attempts must
equal the notebook's ``score_kgmon`` set and its ``Reload score``. A run that does not reproduce
stops the script.

Per distinct grid :math:`g` of output :math:`o`, with :math:`C_g` the view candidates equal to
:math:`g` (one per parsed beam, as ``decode_puzzle`` writes them):

- agreement :math:`a(g) = |C_g| / 16`, what ``arc26.loop.promote`` compares with :math:`\tau`
  (``Tally.agreement`` with ``DECODE_VIEWS``);
- selection score :math:`s(g) = |C_g| - \frac{1}{|C_g|} \sum_{c \in C_g} \bar s(c)`, the
  ``score_kgmon`` value, and its softmax over the output's grids at temperature 1, :math:`\hat
  p(g)`;
- beam score, the mean ``beam_score`` (the NLL under the candidate's own view) over :math:`C_g`.

Usage::

    python scripts/calibration_report.py --out reports/P2.8 \
        --run P2.6-01=~/wt-P2.6/runs/P2.6-01 --run P1.1-01=~/wt-P1.1-timing/runs/P1.1-01 \
        --run P2.1-03=~/wt-P2.1/runs/P2.1-03/run_logs/r256-proj-emb

A run folder holds ``candidates.json`` (or ``run_logs/candidates.json``), ``submission.json``,
and ``selection.log`` or one session log (``*.log``); a session log also gives the views decoded
per output. A run's subset is the first of DEV4, DEV40, DEV120 that holds every task with a
candidate. Writes ``calibration.md``, one CSV per run (one row per distinct grid, no grids), and,
when matplotlib is installed, the reliability diagrams and the coverage-accuracy curve as PNG.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import re
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from types import ModuleType
from typing import Any

from arc26.data import DEV240, SUBSETS, Grid
from arc26.evaluate import SubsetScore, TaskScore
from arc26.loop import DECODE_VIEWS, TAU, Candidate, as_grid, tally

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str) -> ModuleType:
    # Sibling scripts are loaded by path, as compare_runs.py loads its parser: scripts/ is not a
    # package.
    spec = importlib.util.spec_from_file_location(f"_arc26_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The run parsers and the scoring of P2.1-03's per-output comparison.
cr = _load("compare_runs")

BINS = 10
# Fixed, not fit: the softmax maps the selection score to [0, 1] without touching the labels.
TEMPERATURE = 1.0
REPS = 2000
SEED = 0
LEVEL = 0.95
# The rebuilt selection score must equal the moved getter's to float precision.
TOLERANCE = 1e-9
# A rule becomes a proposal when it recovers at least this many tasks in every held-out direction.
PROPOSAL_TASKS = 2.0
# Ranks of the correct grid for the oracle gap's split; 1 and 2 are submitted.
NEAR_RANKS = (3, 5)
PRIMARY = "DEV120"
SUBSET_ORDER = ("DEV4", "DEV40", "DEV120", "DEV240")
# The subsets a run may cover: arc26.data's, and DEV240 (task P6.5b), DEV120 with a copy of each
# task under a derived id, which the timing check's notebook writes.
RUN_SUBSETS = {**SUBSETS, "DEV240": DEV240}
RULES = ("kgmon", "agreement", "kgmon-agreement", "product")
RULE_TEXT = {
    "kgmon": "`score_kgmon`, the notebook's selection (the baseline)",
    "agreement": "agreement first: $|C_g|$ descending, ties in `score_kgmon` order",
    "kgmon-agreement": "selection score first, agreement as tie break: $(s(g), |C_g|)$ "
    "descending",
    "product": r"the product $a(g) \hat p(g)$ of the two decisions' confidences",
}
# Chart tokens: the dataviz reference palette as scripts/train_hypernet.py uses it; the first
# three categorical slots validate all-pairs on the light surface.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)  # fmt: skip
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


class ReproductionError(ValueError):
    """A run's attempts or score do not follow from its ``candidates.json``."""


@dataclass(frozen=True)
class GridRow:
    r"""One distinct candidate grid of one test output.

    Attributes:
        rank: Place under ``score_kgmon``, from 1 (attempt 1).
        count: View candidates equal to the grid, :math:`|C_g|`.
        views: Distinct views among them.
        agreement: :math:`a(g) = |C_g| / 16`.
        score: The ``score_kgmon`` value :math:`s(g)`.
        prob: :math:`\hat p(g)`, the softmax of :math:`s` over the output's grids.
        beam: Mean ``beam_score`` over :math:`C_g`.
        correct: The grid equals the output's ``ALL_CORRECT`` grid.
        grid: The grid, held in memory only.
    """

    rank: int
    count: int
    views: int
    agreement: float
    score: float
    prob: float
    beam: float
    correct: bool
    grid: Grid


@dataclass(frozen=True)
class OutputRow:
    """One test output of a run, rebuilt from its candidates.

    Attributes:
        task: Task id.
        index: Test output index.
        weight: :math:`1/n`, the output's share of its task's credit.
        grids: The distinct grids in ``score_kgmon`` order; empty without candidates.
        decoded_views: Views decoded for the output (the session log's ``decoding`` lines), or
            None without a session log.
    """

    task: str
    index: int
    weight: float
    grids: tuple[GridRow, ...]
    decoded_views: int | None = None

    @property
    def key(self) -> str:
        """The output key ``<task>_<i>``."""
        return f"{self.task}_{self.index}"

    @property
    def solved(self) -> bool:
        """Whether an attempt (``score_kgmon``'s first two grids) is correct."""
        return any(g.correct for g in self.grids[:2])

    @property
    def correct(self) -> GridRow | None:
        """The correct grid among the candidates, or None."""
        return next((g for g in self.grids if g.correct), None)

    @property
    def top_agreement(self) -> GridRow | None:
        """The grid with the most view candidates, ties in ``score_kgmon`` order; None if empty."""
        return min(self.grids, key=lambda g: (-g.count, g.rank), default=None)


@dataclass(frozen=True)
class RunData:
    """One run, reproduced.

    Attributes:
        name: Label in the tables.
        path: The run folder.
        subset: The subset its tasks belong to.
        outputs: Every test output of the subset's tasks, in task and output order.
        score: The rebuilt attempts' score (``arc26.evaluate``).
        reload: The notebook's ``Reload score``.
        candidates: View candidates in the subset's outputs.
        duplicates: View candidates whose view already holds a candidate of the same grid.
    """

    name: str
    path: Path
    subset: str
    outputs: tuple[OutputRow, ...]
    score: SubsetScore
    reload: float
    candidates: int
    duplicates: int

    @property
    def decoded(self) -> bool:
        """Whether the views decoded per output were read from a session log."""
        return any(o.decoded_views is not None for o in self.outputs)

    @property
    def with_candidates(self) -> tuple[OutputRow, ...]:
        """The outputs with at least one candidate."""
        return tuple(o for o in self.outputs if o.grids)


def run_files(path: Path) -> dict[str, Path | None]:
    """Locate a run folder's candidate file, submission, selection output, and session log.

    Args:
        path: An arm folder (``candidates.json`` and ``selection.log`` inside) or a run folder
            (``run_logs/candidates.json`` and the session log at its root).

    Returns:
        ``candidates``, ``submission``, ``selection`` paths, and ``session`` (None without a
        session log).

    Raises:
        FileNotFoundError: If the candidate file, the submission, or a selection output is
            missing.
        ValueError: If the folder holds more than one session log.
    """
    candidates = next(
        (p for p in (path / "candidates.json", path / "run_logs" / "candidates.json")
         if p.is_file()),
        None,
    )
    logs = sorted(p for p in path.glob("*.log") if p.name != "selection.log")
    if len(logs) > 1:
        raise ValueError(f"{path}: more than one session log: {[p.name for p in logs]}")
    session = logs[0] if logs else None
    selection = path / "selection.log" if (path / "selection.log").is_file() else session
    submission = path / "submission.json"
    for label, found in (("candidates.json", candidates), ("submission.json", submission),
                         ("a selection output", selection)):
        if found is None or not found.is_file():
            raise FileNotFoundError(f"{path}: no {label}")
    return {"candidates": candidates, "submission": submission, "selection": selection,
            "session": session}


def infer_subset(tasks: Sequence[str]) -> str:
    """Return the first of DEV4, DEV40, DEV120, DEV240 that holds every given task.

    Args:
        tasks: Task ids with a candidate.

    Returns:
        The subset name.

    Raises:
        ValueError: If no subset holds them all, or none is given.
    """
    if not tasks:
        raise ValueError("a run without candidates")
    for name in SUBSET_ORDER:
        if set(tasks) <= set(RUN_SUBSETS[name]):
            return name
    raise ValueError(f"tasks outside DEV240: {sorted(set(tasks) - set(DEV240))}")


def decoded_views(path: Path, tasks: Sequence[str]) -> dict[str, int]:
    """Count the views decoded per test output from a session log's ``decoding`` lines.

    Args:
        path: A session log of one arm.
        tasks: Task ids to read.

    Returns:
        Output key to its distinct decoded views.
    """
    views: dict[str, set[str]] = {}
    for puzzle in cr.parse_puzzles(cr.read_lines(str(path)), tasks).values():
        for batch in puzzle.batches:
            for view in batch:
                views.setdefault(view.split(".", 1)[0], set()).add(view)
    return {key: len(found) for key, found in views.items()}


def rebuild(
    samples: Mapping[str, Mapping[str, Any]],
    correct: Grid | None,
    views: int = DECODE_VIEWS,
) -> tuple[tuple[GridRow, ...], list[object], int]:
    """Rebuild one output's distinct grids, their signals, and its two attempts.

    Why: the reference's own vote decides the ranking, so ``arc26.loop.tally`` (the moved
    ``score_sum`` with ``getter_kgmon``) ranks the grids and gives their scores; the score is
    then recomputed from its definition as a check of the formula the softmax uses.

    Args:
        samples: The output's candidates as ``compare_runs.load_run`` holds them: candidate key
            ``<view>.out<j>`` to ``{"beam_score", "score_aug", "solution"}``, in file order.
        correct: The output's correct grid, or None.
        views: :math:`n`, the views the agreement divides by.

    Returns:
        The grids in ``score_kgmon`` order, the two attempts padded with the notebook's
        ``[[0]]``, and the view candidates whose view already holds that grid.

    Raises:
        ReproductionError: If the recomputed selection score differs from the moved getter's.
    """
    found = [
        Candidate(s["solution"], s["beam_score"], s["score_aug"], key.rsplit(".out", 1)[0])
        for key, s in samples.items()
    ]
    votes = tally(found, views)
    groups: dict[Grid, list[Candidate]] = {}
    for candidate in found:
        groups.setdefault(candidate.grid, []).append(candidate)
    top = max(votes.scores, default=0.0)
    weights = [math.exp((s - top) / TEMPERATURE) for s in votes.scores]
    rows = []
    for rank, (grid, count, score, weight) in enumerate(
        zip(votes.grids, votes.counts, votes.scores, weights, strict=True), start=1
    ):
        members = groups[grid]
        check = len(members) - statistics.fmean(statistics.fmean(c.score_aug) for c in members)
        if count != len(members) or not math.isclose(
            check, score, rel_tol=TOLERANCE, abs_tol=TOLERANCE
        ):
            raise ReproductionError(f"selection score {score} rebuilt as {check}")
        rows.append(GridRow(
            rank=rank,
            count=count,
            views=len({c.view for c in members}),
            agreement=count / views,
            score=score,
            prob=weight / sum(weights),
            beam=statistics.fmean(c.beam_score for c in members),
            correct=grid == correct,
            grid=grid,
        ))
    attempts: list[object] = [[list(row) for row in g] for g in votes.grids[:2]]
    attempts += [cr.slt.PLACEHOLDER] * (2 - len(attempts))
    duplicates = len(found) - len({(c.view, c.grid) for c in found})
    return tuple(rows), attempts, duplicates


def reload_score(texts: Sequence[str]) -> float:
    """Return the notebook's ``*** Reload score:`` value.

    Args:
        texts: The selection output's lines.

    Returns:
        The last value printed.

    Raises:
        ValueError: If no such line exists.
    """
    values = [float(m.group(1)) for t in texts if (m := cr.slt.NOTEBOOK_RELOAD.match(t.strip()))]
    if not values:
        raise ValueError("no Reload score line in the selection output")
    return values[-1]


def reproduce(name: str, path: Path) -> RunData:
    """Read a run and check that its attempts and score follow from ``candidates.json``.

    Args:
        name: Label in the tables.
        path: The run folder (:func:`run_files`).

    Returns:
        The reproduced run.

    Raises:
        ReproductionError: If an attempt, the ``score_kgmon`` set, or the score differs.
        FileNotFoundError: If a file is missing.
        ValueError: If a log is malformed.
    """
    files = run_files(path)
    spec = cr.parse_spec(
        [f"name={name}", *(f"{k}={files[k]}" for k in ("candidates", "selection", "submission"))],
        False,
    )
    run = cr.load_run(spec, (), False)
    subset = infer_subset(sorted({base.rsplit("_", 1)[0] for base in run.candidates}))
    tasks = RUN_SUBSETS[subset]
    correct = {base: as_grid(grid) for base, grid in cr.correct_grids([run]).items()}
    decoded = decoded_views(files["session"], tasks) if files["session"] else {}
    outputs: list[OutputRow] = []
    problems: list[str] = []
    candidates = duplicates = 0
    for task in tasks:
        entries = run.submission.get(task)
        if not isinstance(entries, list) or not entries:
            problems.append(f"{task}: no test outputs in submission.json")
            continue
        for index, entry in enumerate(entries):
            key = f"{task}_{index}"
            samples = run.candidates.get(key, {})
            try:
                grids, attempts, dups = rebuild(samples, correct.get(key))
            except ReproductionError as err:
                problems.append(f"{key}: {err}")
                continue
            candidates += len(samples)
            duplicates += dups
            if not isinstance(entry, Mapping) or attempts != [
                entry.get("attempt_1"), entry.get("attempt_2")
            ]:
                problems.append(f"{key}: the rebuilt attempts differ from submission.json")
            outputs.append(OutputRow(task, index, 1 / len(entries), grids, decoded.get(key)))
    score = SubsetScore(
        subset,
        tuple(TaskScore(t, tuple(o.solved for o in outputs if o.task == t)) for t in tasks),
    )
    solved = {o.key for o in outputs if o.solved}
    logged = {k for k in run.solved if k.rsplit("_", 1)[0] in tasks}
    if solved != logged:
        problems.append(
            f"rebuilt solved set differs from score_kgmon's: {sorted(solved ^ logged)}"
        )
    reload = reload_score(cr.read_lines(spec["selection"]))
    credit = cr.credit(run, tasks, {t: len(run.submission[t]) for t in tasks}, subset).score
    for label, value in (("score_kgmon set", credit), ("Reload score", reload)):
        if not math.isclose(score.score, value, abs_tol=TOLERANCE):
            problems.append(f"rebuilt score {score.score} differs from the {label} {value}")
    if problems:
        raise ReproductionError(f"{name} does not reproduce: " + "; ".join(problems[:10]))
    return RunData(name, path, subset, tuple(outputs), score, reload, candidates, duplicates)


def equal_mass_bins(p: Sequence[float], bins: int = BINS) -> list[list[int]]:
    r"""Split item indices into at most ``bins`` bins of nearly equal size by predicted value.

    Why: the agreement takes 16 values, so fixed-width bins would leave most of them empty; a
    bin boundary never splits equal values, so the binning is a function of :math:`\hat p`
    and not of the items' order. Item of rank :math:`r` (0-based, ascending) goes to bin
    :math:`\lfloor r B / N \rfloor`, and a run of equal values to its first member's bin;
    bins left empty by such runs are dropped.

    Args:
        p: Predicted values.
        bins: :math:`B`, at least 1.

    Returns:
        The non-empty bins, ascending in :math:`\hat p`, each a list of indices into ``p``.

    Raises:
        ValueError: If ``bins`` is under 1.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1; got {bins}")
    order = sorted(range(len(p)), key=lambda i: p[i])
    found: dict[int, list[int]] = {}
    current = 0
    for rank, i in enumerate(order):
        if rank == 0 or p[i] != p[order[rank - 1]]:
            current = rank * bins // len(p)
        found.setdefault(current, []).append(i)
    return [found[b] for b in sorted(found)]


@dataclass(frozen=True)
class Bin:
    r"""One bin of a reliability diagram.

    Attributes:
        n: Items in the bin.
        p_mean: Mean predicted value, :math:`\bar p_b`.
        c_mean: Fraction correct, :math:`\bar c_b`.
        p_min: Smallest predicted value.
        p_max: Largest predicted value.
    """

    n: int
    p_mean: float
    c_mean: float
    p_min: float
    p_max: float


def _check(p: Sequence[float], c: Sequence[bool]) -> None:
    if len(p) != len(c):
        raise ValueError(f"{len(p)} predictions against {len(c)} labels")
    if not p:
        raise ValueError("no items")


def reliability(p: Sequence[float], c: Sequence[bool], bins: int = BINS) -> list[Bin]:
    """Return the reliability diagram's bins (:func:`equal_mass_bins`).

    Args:
        p: Predicted probabilities of correctness.
        c: Correctness.
        bins: Bins at most.

    Returns:
        The non-empty bins, ascending.

    Raises:
        ValueError: If the lengths differ or there are no items.
    """
    _check(p, c)
    return [
        Bin(
            len(idx),
            statistics.fmean(p[i] for i in idx),
            statistics.fmean(float(c[i]) for i in idx),
            min(p[i] for i in idx),
            max(p[i] for i in idx),
        )
        for idx in equal_mass_bins(p, bins)
    ]


def ece(p: Sequence[float], c: Sequence[bool], bins: int = BINS) -> float:
    r"""Expected calibration error, :math:`\sum_b \frac{n_b}{N} |\bar c_b - \bar p_b|`.

    Args:
        p: Predicted probabilities of correctness.
        c: Correctness.
        bins: Equal-mass bins at most.

    Returns:
        The ECE, in [0, 1].

    Raises:
        ValueError: If the lengths differ or there are no items.
    """
    return sum(b.n * abs(b.c_mean - b.p_mean) for b in reliability(p, c, bins)) / len(p)


def brier(p: Sequence[float], c: Sequence[bool]) -> float:
    r"""Brier score, :math:`\frac{1}{N} \sum_i (\hat p_i - c_i)^2`.

    Args:
        p: Predicted probabilities of correctness.
        c: Correctness.

    Returns:
        The score, in [0, 1].

    Raises:
        ValueError: If the lengths differ or there are no items.
    """
    _check(p, c)
    return statistics.fmean((x - float(y)) ** 2 for x, y in zip(p, c, strict=True))


def auroc(p: Sequence[float], c: Sequence[bool]) -> float | None:
    """Area under the ROC curve: the chance a correct item outranks a wrong one, ties half.

    Args:
        p: Scores, higher meaning more likely correct.
        c: Correctness.

    Returns:
        The AUROC, or None when every item is correct or every item is wrong.

    Raises:
        ValueError: If the lengths differ or there are no items.
    """
    _check(p, c)
    pos = sum(1 for y in c if y)
    neg = len(c) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and p[order[end + 1]] == p[order[start]]:
            end += 1
        for k in range(start, end + 1):
            ranks[order[k]] = (start + end) / 2 + 1
        start = end + 1
    rank_sum = sum(r for r, y in zip(ranks, c, strict=True) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval.

    Attributes:
        low: Lower end.
        high: Upper end.
        skipped: Resamples where the statistic was undefined (one class only).
    """

    low: float
    high: float
    skipped: int = 0


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of sorted values."""
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


Item = tuple[str, float, bool]
Stat = Callable[[Sequence[float], Sequence[bool]], float | None]


def bootstrap(
    items: Sequence[Item],
    stats: Mapping[str, Stat],
    reps: int = REPS,
    seed: int = SEED,
    level: float = LEVEL,
) -> dict[str, Interval | None]:
    """Percentile intervals of statistics over resampled tasks.

    Why: the outputs of one task share its adapter and its candidates, so tasks, not items, are
    the independent units; each resample draws as many tasks as there are, with replacement, and
    keeps all their items. Every statistic is computed on the same resamples.

    Args:
        items: ``(task, p, c)`` triples.
        stats: Name to a statistic of ``(p, c)``; None marks an undefined value.
        reps: Resamples.
        seed: Seed of the resampling (``random.Random``).
        level: Coverage of the interval.

    Returns:
        Name to its interval, or None when no resample defined the statistic.

    Raises:
        ValueError: If ``items`` is empty or ``reps`` is under 1.
    """
    if not items:
        raise ValueError("no items")
    if reps < 1:
        raise ValueError(f"reps must be at least 1; got {reps}")
    groups: dict[str, list[tuple[float, bool]]] = {}
    for task, p, c in items:
        groups.setdefault(task, []).append((p, c))
    keys = sorted(groups)
    rng = random.Random(seed)
    values: dict[str, list[float]] = {name: [] for name in stats}
    for _ in range(reps):
        drawn = [x for k in rng.choices(keys, k=len(keys)) for x in groups[k]]
        p = [x for x, _ in drawn]
        c = [y for _, y in drawn]
        for name, stat in stats.items():
            value = stat(p, c)
            if value is not None:
                values[name].append(value)
    tail = (1 - level) / 2
    return {
        name: Interval(
            _quantile(sorted(v), tail), _quantile(sorted(v), 1 - tail), reps - len(v)
        ) if v else None
        for name, v in values.items()
    }


@dataclass(frozen=True)
class Reference:
    """The ECE a perfectly calibrated predictor would show with the same predicted values.

    Attributes:
        mean: Mean simulated ECE.
        p95: Its 95th percentile.
        exceed: Share of the simulated ECEs at or above the observed one.
    """

    mean: float
    p95: float
    exceed: float


def reference_ece(
    p: Sequence[float],
    observed: float | None = None,
    bins: int = BINS,
    reps: int = REPS,
    seed: int = SEED,
) -> Reference:
    r"""Simulate the ECE of a perfectly calibrated predictor with the same :math:`\hat p`.

    Why: binned ECE is positive even for a perfect predictor, by an amount set by the items'
    count and their :math:`\hat p` values; an observed ECE means something only against that
    floor. Each draw labels item :math:`i` correct with probability :math:`\hat p_i`
    (independently) and measures the ECE with the same bins, which depend on :math:`\hat p`
    alone.

    Args:
        p: Predicted probabilities, in [0, 1].
        observed: The observed ECE, for ``exceed``; None gives ``exceed`` 0.
        bins: Equal-mass bins at most.
        reps: Draws.
        seed: Seed of the draws (``random.Random``).

    Returns:
        The reference.

    Raises:
        ValueError: If ``p`` is empty or ``reps`` is under 1.
    """
    if not p:
        raise ValueError("no items")
    if reps < 1:
        raise ValueError(f"reps must be at least 1; got {reps}")
    rng = random.Random(seed)
    draws = sorted(ece(p, [rng.random() < x for x in p], bins) for _ in range(reps))
    exceed = 0.0 if observed is None else sum(d >= observed for d in draws) / reps
    return Reference(statistics.fmean(draws), _quantile(draws, 0.95), exceed)


@dataclass(frozen=True)
class Calibration:
    r"""Calibration of one decision on one run.

    Attributes:
        n: Items.
        tasks: Tasks with items.
        mean_p: Mean predicted value.
        accuracy: Fraction correct.
        ece: Expected calibration error.
        brier: Brier score.
        auroc: AUROC, or None with one class.
        intervals: ``ece``, ``brier``, ``auroc`` bootstrap intervals over tasks; ECE's is a
            resampled range, not a confidence interval (binned ECE is biased upward).
        reference: The ECE of a perfectly calibrated predictor with the same :math:`\hat p`.
        bins: The reliability diagram.
    """

    n: int
    tasks: int
    mean_p: float
    accuracy: float
    ece: float
    brier: float
    auroc: float | None
    intervals: Mapping[str, Interval | None]
    reference: Reference
    bins: tuple[Bin, ...]


def calibrate(
    items: Sequence[Item], bins: int = BINS, reps: int = REPS, seed: int = SEED
) -> Calibration:
    """Measure ECE, Brier, and AUROC of ``(task, p, c)`` items, with bootstrap intervals.

    Args:
        items: One per decision: its task, its predicted probability, its correctness.
        bins: Equal-mass bins at most.
        reps: Bootstrap resamples, and draws of the perfect-calibration reference.
        seed: Seed of both.

    Returns:
        The calibration.

    Raises:
        ValueError: If there are no items.
    """
    if not items:
        raise ValueError("no items")
    p = [x for _, x, _ in items]
    c = [y for _, _, y in items]
    stats: dict[str, Stat] = {
        "ece": lambda u, v: ece(u, v, bins),
        "brier": brier,
        "auroc": auroc,
    }
    observed = ece(p, c, bins)
    return Calibration(
        n=len(items),
        tasks=len({t for t, _, _ in items}),
        mean_p=statistics.fmean(p),
        accuracy=statistics.fmean(float(y) for y in c),
        ece=observed,
        brier=brier(p, c),
        auroc=auroc(p, c),
        intervals=bootstrap(items, stats, reps, seed),
        reference=reference_ece(p, observed, bins, reps, seed),
        bins=tuple(reliability(p, c, bins)),
    )


def promotion_items(run: RunData, top_agreement: bool = False, decoded: bool = False) -> list[Item]:
    r"""Items of the promotion decision: one grid per output with candidates, :math:`\hat p = a`.

    Args:
        run: A reproduced run.
        top_agreement: Take the grid with the most view candidates instead of the voted answer
            (``score_kgmon``'s first, the grid ``promote`` checks).
        decoded: Divide by the views decoded for the output instead of 16.

    Returns:
        ``(task, p, c)`` per output with candidates (and, with ``decoded``, a decoded count).
    """
    items = []
    for o in run.with_candidates:
        g = o.top_agreement if top_agreement else o.grids[0]
        if decoded and o.decoded_views is None:
            continue
        p = min(1.0, g.count / o.decoded_views) if decoded else g.agreement
        items.append((o.task, p, g.correct))
    return items


def selection_items(run: RunData) -> list[Item]:
    r"""Items of attempt selection: the two attempts per output, :math:`\hat p` the softmax.

    Args:
        run: A reproduced run.

    Returns:
        ``(task, p, c)`` per attempt that holds a candidate grid (one for an output with one).
    """
    return [(o.task, g.prob, g.correct) for o in run.with_candidates for g in o.grids[:2]]


def signal_auroc(run: RunData) -> dict[str, float | None]:
    """AUROC of each per-grid signal against correctness, over every distinct grid.

    Args:
        run: A reproduced run.

    Returns:
        ``agreement``, ``selection score``, ``beam score`` (lower NLL ranks higher).
    """
    grids = [g for o in run.with_candidates for g in o.grids]
    c = [g.correct for g in grids]
    return {
        "agreement": auroc([g.agreement for g in grids], c),
        "selection score": auroc([g.score for g in grids], c),
        "beam score": auroc([-g.beam for g in grids], c),
    }


@dataclass(frozen=True)
class CoverageRow:
    r"""Promotion at one threshold :math:`\tau = k/16`.

    Attributes:
        k: Views needed.
        passed: Outputs whose voted answer has :math:`a \ge \tau`.
        right: Of them, correct.
        outputs: Outputs with candidates.
    """

    k: int
    passed: int
    right: int
    outputs: int

    @property
    def coverage(self) -> float:
        """Share of the outputs with candidates that pass."""
        return self.passed / self.outputs if self.outputs else 0.0

    @property
    def accuracy(self) -> float | None:
        """Share of the passing outputs that are correct; None when none passes."""
        return self.right / self.passed if self.passed else None


def coverage_curve(run: RunData, views: int = DECODE_VIEWS) -> list[CoverageRow]:
    r"""What the loop's agreement screen would have passed at each :math:`\tau = k / 16`.

    Args:
        run: A reproduced run.
        views: The views the agreement divides by.

    Returns:
        One row per :math:`k` from 1 to ``views``.
    """
    voted = [o.grids[0] for o in run.with_candidates]
    return [
        CoverageRow(
            k,
            sum(1 for g in voted if g.count >= k),
            sum(1 for g in voted if g.count >= k and g.correct),
            len(voted),
        )
        for k in range(1, views + 1)
    ]


@dataclass(frozen=True)
class Gap:
    r"""The oracle gap: outputs with a correct candidate and no correct attempt.

    Attributes:
        outputs: Those outputs.
        tasks: Their credit, :math:`\sum 1/n`.
        near: Credit of those whose correct grid ranks 3rd to 5th.
        lower: Credit of those whose correct grid ranks 6th or lower.
        one_view: Credit of those whose correct grid came from one view only.
        points: ``tasks`` in leaderboard points (100 over the subset's tasks).
    """

    outputs: tuple[OutputRow, ...]
    tasks: float
    near: float
    lower: float
    one_view: float
    points: float


def oracle_gap(run: RunData) -> Gap:
    """Measure the oracle gap, in tasks with the :math:`1/n` weights, split by the correct rank.

    Args:
        run: A reproduced run.

    Returns:
        The gap.
    """
    missed = tuple(o for o in run.outputs if o.correct is not None and not o.solved)
    lo, hi = NEAR_RANKS
    tasks = sum(o.weight for o in missed)
    return Gap(
        outputs=missed,
        tasks=tasks,
        near=sum(o.weight for o in missed if lo <= o.correct.rank <= hi),
        lower=sum(o.weight for o in missed if o.correct.rank > hi),
        one_view=sum(o.weight for o in missed if o.correct.views == 1),
        points=100 * tasks / run.score.tasks,
    )


def rank_by(rule: str, grids: Sequence[GridRow]) -> list[GridRow]:
    """Order one output's grids by a fixed rule; ties keep ``score_kgmon``'s order.

    Args:
        rule: A name of ``RULES``.
        grids: The grids in ``score_kgmon`` order.

    Returns:
        The grids, best first.

    Raises:
        ValueError: If the rule is unknown.
    """
    keys: dict[str, Callable[[GridRow], Any]] = {
        "kgmon": lambda g: 0,
        "agreement": lambda g: -g.count,
        "kgmon-agreement": lambda g: (-g.score, -g.count),
        "product": lambda g: -(g.agreement * g.prob),
    }
    if rule not in keys:
        raise ValueError(f"unknown rule {rule!r}")
    return sorted(grids, key=keys[rule])


def rule_score(run: RunData, rule: str) -> float:
    """Score a run's outputs with a rule's first two grids as the attempts.

    Args:
        run: A reproduced run.
        rule: A name of ``RULES``.

    Returns:
        The score in tasks (``arc26.evaluate``).
    """
    flags: dict[str, list[bool]] = {}
    for o in run.outputs:
        flags.setdefault(o.task, []).append(any(g.correct for g in rank_by(rule, o.grids)[:2]))
    return SubsetScore(run.subset, tuple(TaskScore(t, tuple(f)) for t, f in flags.items())).score


@dataclass(frozen=True)
class RuleChange:
    """What a rule changes against ``score_kgmon`` on one run, in tasks.

    Attributes:
        recovered: Credit of the outputs the rule solves and ``score_kgmon`` does not.
        lost: Credit of the outputs ``score_kgmon`` solves and the rule does not.
        identical: The rule submits the same two grids as ``score_kgmon`` for every output.
    """

    recovered: float
    lost: float
    identical: bool

    @property
    def net(self) -> float:
        """Recovered minus lost: the score difference."""
        return self.recovered - self.lost


def rule_change(run: RunData, rule: str) -> RuleChange:
    """Compare a rule's attempts with ``score_kgmon``'s, output by output.

    Args:
        run: A reproduced run.
        rule: A name of ``RULES``.

    Returns:
        The gross credit recovered and lost (each output weighted :math:`1/n`), and whether the
        two grids submitted per output are the same.
    """
    recovered = lost = 0.0
    identical = True
    for o in run.outputs:
        top = rank_by(rule, o.grids)[:2]
        mine, base = any(g.correct for g in top), o.solved
        recovered += o.weight * (mine and not base)
        lost += o.weight * (base and not mine)
        identical &= {g.rank for g in top} == {g.rank for g in o.grids[:2]}
    return RuleChange(recovered, lost, identical)


def choose(gains: Mapping[str, float], identical: frozenset[str] = frozenset()) -> str | None:
    """Return the alternative rule with the largest gain, the earliest of ``RULES`` on a tie.

    Why: the held-out check asks what the best fixed rule found on one run is worth on
    another, so the choice is among the alternatives to ``score_kgmon`` even when none gains.
    A rule that submits ``score_kgmon``'s grids on that run is ``score_kgmon`` there, and
    choosing it would report the baseline as a rule, so it is left out.

    Args:
        gains: Rule to its net gain in tasks over ``kgmon`` on the run it is chosen on.
        identical: Rules identical to ``kgmon`` on that run.

    Returns:
        The chosen rule, never ``kgmon``; None when every alternative is identical to it.
    """
    left = [r for r in RULES[1:] if r not in identical]
    return max(left, key=lambda r: (gains[r], -RULES.index(r))) if left else None


def score_ties(run: RunData) -> int:
    """Count a run's outputs where two distinct grids have equal selection scores.

    Args:
        run: A reproduced run.

    Returns:
        Outputs with a tie in :math:`s(g)`, where only ``kgmon-agreement`` can differ from
        ``score_kgmon``.
    """
    return sum(len({g.score for g in o.grids}) < len(o.grids) for o in run.outputs)


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(value: float | None, interval: Interval | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if interval is None:
        return _f(value, digits)
    note = f"; {interval.skipped} undefined" if interval.skipped else ""
    return f"{value:.{digits}f} [{interval.low:.{digits}f}, {interval.high:.{digits}f}{note}]"


def _signed(value: float) -> str:
    return f"{value:+.2f}" if abs(value) > TOLERANCE else "0"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(str(c) for c in row) + " |" for row in rows),
    ]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)


def write_csv(run: RunData, path: Path) -> None:
    """Write one row per distinct grid of a run: its signals and how selection treated it.

    Args:
        run: A reproduced run.
        path: The CSV file.
    """
    rules = RULES[1:]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "task_id", "output", "weight", "decoded_views", "rank_kgmon", "candidates", "views",
            "agreement", "selection_score", "softmax_p", "beam_score_mean", "correct", "attempt",
            "voted", "top_agreement", *(f"rank_{r.replace('-', '_')}" for r in rules),
        ])
        for o in run.outputs:
            top = o.top_agreement
            ranks = {r: {id(g): i for i, g in enumerate(rank_by(r, o.grids), 1)} for r in rules}
            for g in o.grids:
                writer.writerow([
                    o.task, o.index, f"{o.weight:.4f}",
                    "" if o.decoded_views is None else o.decoded_views,
                    g.rank, g.count, g.views, f"{g.agreement:.4f}", f"{g.score:.6f}",
                    f"{g.prob:.6f}", f"{g.beam:.6f}", int(g.correct),
                    g.rank if g.rank <= 2 else "", int(g.rank == 1), int(g is top),
                    *(ranks[r][id(g)] for r in rules),
                ])


def write_plots(
    primary: Sequence[RunData],
    results: Mapping[tuple[str, str], Calibration],
    curves: Mapping[str, Sequence[CoverageRow]],
    out: Path,
) -> tuple[list[Path], str]:
    """Draw the reliability diagrams and the coverage-accuracy curve of the primary runs.

    Args:
        primary: The DEV120 runs (at most three, one categorical slot each).
        results: ``(run, decision)`` to its calibration, decisions ``promotion`` and
            ``selection``.
        curves: Run name to its coverage-accuracy rows.
        out: The report folder.

    Returns:
        The PNG files written, and why none was written (empty when drawn).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return [], "matplotlib is not installed here; the CSV files and the tables hold the data"
    if not primary:
        return [], "no DEV120 run"
    runs = list(primary)[: len(SERIES)]

    def style(ax: Any) -> None:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelcolor=INK_2)

    written = []
    titles = {
        "promotion": ("Promotion: agreement of the voted answer", r"agreement $a(\hat y)$"),
        "selection": ("Attempt selection: softmax of the selection score",
                      r"$\hat p$, softmax of $s(g)$ at $T = 1$"),
    }
    for decision, (title, xlabel) in titles.items():
        fig, axes = plt.subplots(
            2, len(runs), figsize=(5.2 * len(runs), 6.2), squeeze=False, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE,
        )
        for col, run in enumerate(runs):
            cal = results[(run.name, decision)]
            top, bottom = axes[0][col], axes[1][col]
            style(top)
            style(bottom)
            top.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--")
            px = [b.p_mean for b in cal.bins]
            top.plot(
                px, [b.c_mean for b in cal.bins], color=SERIES[0], linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
            )
            top.set_xlim(0, 1)
            top.set_ylim(0, 1)
            top.set_title(
                f"{run.name} ({run.subset}): ECE {cal.ece:.3f}, {cal.n} items",
                color=INK, loc="left", fontsize=11,
            )
            if col == 0:
                top.set_ylabel("fraction correct in the bin", color=INK_2)
                bottom.set_ylabel("items", color=INK_2)
            bottom.vlines(px, 0, [b.n for b in cal.bins], color=SERIES[0], linewidth=2)
            bottom.plot(px, [b.n for b in cal.bins], "o", color=SERIES[0], markersize=5)
            bottom.set_ylim(0, max(b.n for b in cal.bins) * 1.15)
            bottom.set_xlabel(f"{xlabel}, bin mean", color=INK_2)
        fig.suptitle(f"{title} (10 equal-mass bins)", color=INK, x=0.01, ha="left")
        fig.tight_layout()
        path = out / f"reliability_{decision}.png"
        fig.savefig(path, dpi=120, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)
    fig, (acc, cov) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, facecolor=SURFACE)
    for ax in (acc, cov):
        style(ax)
        ax.axvline(TAU, color=MUTED, linewidth=1.2, linestyle="--")
    for color, run in zip(SERIES, runs, strict=False):
        rows = curves[run.name]
        x = [r.k / DECODE_VIEWS for r in rows]
        line = {"color": color, "linewidth": 2, "marker": "o", "markersize": 5,
                "markeredgecolor": SURFACE, "markeredgewidth": 1.2, "label": run.name}
        acc.plot(x, [r.accuracy for r in rows], **line)
        cov.plot(x, [r.coverage for r in rows], **line)
    acc.text(TAU, 0.02, r" $\tau = 9/16$ (ADR-014)", color=INK_2, fontsize=9)
    acc.set_ylim(0, 1.02)
    cov.set_ylim(0, 1.02)
    acc.set_ylabel("accuracy of the passing outputs", color=INK_2)
    cov.set_ylabel("outputs passing", color=INK_2)
    cov.set_xlabel(r"$\tau$ (agreement of the voted answer at least $\tau$)", color=INK_2)
    acc.set_title("Promotion screen against $\\tau$ (outputs with candidates)", color=INK,
                  loc="left")
    acc.legend(frameon=False, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    path = out / "coverage_accuracy.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)
    return written, ""


def render(
    runs: Sequence[RunData],
    results: Mapping[tuple[str, str], Calibration],
    curves: Mapping[str, Sequence[CoverageRow]],
    plots: Sequence[Path],
    note: str,
    command: str,
    reps: int,
    seed: int,
) -> list[str]:
    """Build ``calibration.md``.

    Args:
        runs: Reproduced runs, DEV120 first.
        results: ``(run, decision)`` to its calibration; decisions ``promotion``,
            ``promotion-top``, ``promotion-decoded`` (runs with a session log), ``selection``.
        curves: Run name to its coverage-accuracy rows.
        plots: PNG files written.
        note: Why no PNG was written, or empty.
        command: The command line, for the header.
        reps: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        The report's lines.
    """
    primary = [r for r in runs if r.subset == PRIMARY]
    short = [
        f"{r.name} {sum(1 for o in r.outputs if (o.decoded_views or DECODE_VIEWS) < DECODE_VIEWS)}"
        for r in primary if r.decoded
    ]
    decoded_clause = (
        ", and $\\hat y$ over the views actually decoded instead of 16 (the 1200 s cap cut "
        + ", ".join(short) + " outputs short)"
        if short else ""
    )
    lines = [
        "# P2.8: calibration of the free confidence signal, the oracle gap, re-ranking",
        "",
        "Generated by `scripts/calibration_report.py` from the pulled logs; no solution file read "
        "(an output's correct grid is a candidate its run marked `ALL_CORRECT`), no grid written. "
        "Command, from the repository root:",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Reproduction",
        "",
        "From `candidates.json` alone: the distinct grids per test output, their `score_kgmon` "
        "values (the moved `score_sum` and `getter_kgmon` through `arc26.loop.tally`, each value "
        "recomputed from its definition to $10^{-9}$), the two attempts (padded with `[[0]]` as "
        "`get_submission` does), and their score (`arc26.evaluate`). Every run below reproduces "
        "exactly: every output's attempts equal `submission.json`, the solved outputs equal the "
        "logged `score_kgmon` set, and the score equals the `Reload score` line.",
        "",
        *_table(
            ["Run", "Subset", "Outputs (with candidates)", "View candidates", "Distinct grids",
             "Attempts equal", "Score (tasks)", "`Reload score`"],
            [[r.name, f"{r.subset}, {r.score.tasks} tasks", f"{len(r.outputs)} "
              f"({len(r.with_candidates)})", r.candidates,
              sum(len(o.grids) for o in r.outputs), f"{len(r.outputs)} of {len(r.outputs)}",
              f"{r.score.score:.2f}", f"{r.reload:.4f}"] for r in runs],
        ),
        "",
        "## Definitions",
        "",
        "- **Agreement** $a(g) = |C_g| / 16$: the view candidates equal to $g$ over the 16 "
        "decoding views, the quantity `arc26.loop.promote` compares with $\\tau$ "
        "(`Tally.agreement`, `DECODE_VIEWS`; ADR-014). A view's beams count once each; no view "
        "of any run here parsed two beams into one grid ("
        + ", ".join(f"{r.name} {r.duplicates}" for r in runs)
        + " such candidates), so beams and views give the same count.",
        "- **Selection score** $s(g) = |C_g| - \\frac{1}{|C_g|} \\sum_{c \\in C_g} \\bar s(c)$, "
        "`score_kgmon`'s value ($\\bar s(c)$: the mean of $c$'s 8 rescoring NLLs). "
        "$\\hat p(g) = e^{s(g)/T} / \\sum_{g'} e^{s(g')/T}$ over the output's distinct grids, "
        "$T = 1$ fixed, nothing fit.",
        "- **Beam score**: the mean `beam_score` (NLL under the candidate's own view) over $C_g$.",
        "- **Promotion decision**: per output with candidates, the voted answer $\\hat y$ "
        "(`score_kgmon`'s first grid, the one `promote` checks) with $\\hat p = a(\\hat y)$. "
        "Secondary rows: the grid with the most view candidates instead (it differs from "
        "$\\hat y$ on "
        + ", ".join(
            f"{r.name} {sum(o.top_agreement is not o.grids[0] for o in r.with_candidates)}"
            for r in primary
        )
        + f" outputs){decoded_clause}.",
        "- **Attempt selection**: the two attempts per output with candidates (one when the "
        "output has one grid), $\\hat p(g)$ each.",
        f"- **Metrics**: {BINS} equal-mass bins (a bin boundary never splits equal $\\hat p$, so "
        "fewer bins can result), ECE $= \\sum_b \\frac{n_b}{N} |\\bar c_b - \\bar p_b|$, Brier "
        "$= \\frac{1}{N} \\sum_i (\\hat p_i - c_i)^2$, AUROC (ties count half). Brier and "
        f"AUROC intervals: 95% percentile bootstrap over tasks, {reps} resamples, "
        f"`random.Random({seed})`; \"undefined\": resamples with one class only, where AUROC "
        "has no value.",
        "- **ECE against perfect calibration**: binned ECE is biased upward, so the resampled "
        "ECEs sit above the point estimate and their 2.5th to 97.5th percentiles (\"resampled "
        "range\") are not a confidence interval. The reference is the ECE a perfectly "
        "calibrated predictor would show with the same $\\hat p$ values and bins: each item "
        f"correct with probability $\\hat p_i$, independently, {reps} draws, "
        f"`random.Random({seed})`; its mean, its 95th percentile, and the share of draws at or "
        "above the observed ECE. An observed ECE under the 95th percentile cannot be told apart "
        "from perfect calibration.",
        "",
        "## Calibration",
        "",
    ]
    names = {
        "promotion": "promotion, $\\hat y$",
        "promotion-top": "promotion, most view candidates",
        "promotion-decoded": "promotion, $\\hat y$ over decoded views",
        "selection": "attempt selection",
    }
    rows = []
    for run in runs:
        for decision, label in names.items():
            cal = results.get((run.name, decision))
            if cal is None:
                continue
            iv, ref = cal.intervals, cal.reference
            rows.append([
                run.name, run.subset, label, f"{cal.n} ({cal.tasks} tasks)", _f(cal.mean_p),
                _f(cal.accuracy), _f(cal.ece),
                f"{iv['ece'].low:.3f} to {iv['ece'].high:.3f}" if iv["ece"] else "n/a",
                f"{ref.mean:.3f} ({ref.p95:.3f})", f"{ref.exceed:.2f}",
                _ci(cal.brier, iv["brier"]), _ci(cal.auroc, iv["auroc"]),
            ])
    lines += _table(
        ["Run", "Subset", "Decision", "Items", "Mean $\\hat p$", "Correct", "ECE",
         "ECE resampled range", "ECE if calibrated: mean (95th pct)",
         "Calibrated draws at or above", "Brier [95%]", "AUROC [95%]"],
        rows,
    )
    lines += [
        "",
        "AUROC of each signal over every distinct grid (a ranking measure; the beam score ranks "
        "lower NLL higher):",
        "",
        *_table(
            ["Run", "Distinct grids (correct)", "Agreement", "Selection score", "Beam score"],
            [[r.name, f"{sum(len(o.grids) for o in r.outputs)} "
              f"({sum(o.correct is not None for o in r.outputs)})",
              *(_f(v) for v in signal_auroc(r).values())] for r in runs],
        ),
        "",
        "### Reliability diagrams",
        "",
    ]
    if plots:
        lines += [f"![{p.stem}]({p.name})" for p in plots if p.stem.startswith("reliability")]
        lines.append("")
    else:
        lines += [f"No PNG: {note}.", ""]
    for run in primary:
        for decision in ("promotion", "selection"):
            cal = results[(run.name, decision)]
            lines += [
                f"{run.name}, {names[decision]}:",
                "",
                *_table(
                    ["Bin", "Items", "$\\hat p$ range", "Mean $\\hat p$", "Fraction correct"],
                    [[i, b.n, f"{b.p_min:.3f} to {b.p_max:.3f}", f"{b.p_mean:.3f}",
                      f"{b.c_mean:.3f}"] for i, b in enumerate(cal.bins, 1)],
                ),
                "",
            ]
    lines += [
        "### Coverage and accuracy of the promotion screen against $\\tau$",
        "",
        "Outputs with candidates whose voted answer reaches $a(\\hat y) \\ge \\tau = k/16$ "
        "(what the loop's agreement screen would have passed before the re-decode and the "
        "gate), and how many of them are correct.",
        "",
    ]
    if plots:
        lines += [f"![{p.stem}]({p.name})" for p in plots if p.stem == "coverage_accuracy"]
        lines.append("")
    header = ["$\\tau$"]
    for run in primary:
        header += [f"{run.name} passing", f"{run.name} correct"]
    table = []
    for k in range(1, DECODE_VIEWS + 1):
        cells = [f"{k}/16"]
        for run in primary:
            row = curves[run.name][k - 1]
            cells += [f"{row.passed} of {row.outputs} ({row.coverage:.3f})",
                      f"{row.right} ({_f(row.accuracy)})"]
        table.append(cells)
    lines += [*_table(header, table), ""]
    lines += ["## Oracle gap", ""]
    gaps = {r.name: oracle_gap(r) for r in runs}
    lines += [
        "Outputs with a correct candidate and no correct attempt, each weighted $1/n$ for a task "
        f"with $n$ test outputs; points are 100 over the subset's tasks. Split by the correct "
        f"grid's place under `score_kgmon` ({NEAR_RANKS[0]}rd to {NEAR_RANKS[1]}th, lower); "
        "\"one view\": the correct grid came from a single view (a cut across the two).",
        "",
        *_table(
            ["Run", "Subset", "Score", "Oracle", "Gap (tasks)", "Gap (points)", "Outputs",
             "Ranked 3rd to 5th", "Lower", "One view"],
            [[r.name, f"{r.subset}, {r.score.tasks} tasks", f"{r.score.score:.2f}",
              f"{r.score.score + g.tasks:.2f}", f"{g.tasks:.2f}", f"{g.points:.2f}",
              len(g.outputs), f"{g.near:.2f}", f"{g.lower:.2f}", f"{g.one_view:.2f}"]
             for r, g in ((r, gaps[r.name]) for r in runs)],
        ),
        "",
    ]
    for run in primary:
        gap = gaps[run.name]
        lines += [
            f"{run.name}: the gap's outputs (rank and signals of the correct grid; the voted "
            "answer's agreement for comparison).",
            "",
            *_table(
                ["Output", "Weight", "Rank", "Distinct grids", "Views", "$a$", "$s$",
                 "$\\hat p$", "$a(\\hat y)$"],
                [[f"`{o.key}`", f"{o.weight:.2f}", o.correct.rank, len(o.grids), o.correct.views,
                  f"{o.correct.agreement:.4f}", f"{o.correct.score:.3f}", f"{o.correct.prob:.3f}",
                  f"{o.grids[0].agreement:.4f}"] for o in gap.outputs],
            ),
            "",
        ]
    lines += [
        "## Re-ranking (report only)",
        "",
        "Fixed rules over each output's distinct grids, the first two as the attempts; ties "
        "keep `score_kgmon`'s order. Nothing in `arc26/decode.py` or any protected path "
        "changes.",
        "",
        *(f"- `{r}`: {RULE_TEXT[r]}." for r in RULES),
        "",
        "`kgmon-agreement` can differ from `score_kgmon` only where two grids tie in $s(g)$: "
        + ", ".join(f"{r.name} {score_ties(r)}" for r in runs)
        + " such outputs.",
        "",
    ]
    changes = {r.name: {rule: rule_change(r, rule) for rule in RULES[1:]} for r in runs}
    gains = {n: {rule: ch.net for rule, ch in c.items()} for n, c in changes.items()}

    def change(run: RunData, rule: str) -> str:
        ch = changes[run.name][rule]
        if ch.identical:
            return "identical to `kgmon`"
        return f"{_signed(ch.net)} (recovered {ch.recovered:.2f}, lost {ch.lost:.2f})"

    lines += [
        "Net gain in tasks over `score_kgmon` on each run (in sample), with the gross credit "
        "recovered (outputs the rule solves and `score_kgmon` does not, $1/n$ each) and lost; "
        "\"identical\": the rule submits `score_kgmon`'s two grids for every output:",
        "",
        *_table(
            ["Run", "Subset", "`score_kgmon`", "Gap", *(f"`{r}`" for r in RULES[1:])],
            [[r.name, r.subset, f"{r.score.score:.2f}", f"{gaps[r.name].tasks:.2f}",
              *(change(r, rule) for rule in RULES[1:])] for r in runs],
        ),
        "",
    ]
    chosen = {
        a.name: choose(
            gains[a.name],
            frozenset(rule for rule, ch in changes[a.name].items() if ch.identical),
        )
        for a in primary
    }
    held = [(a, b, chosen[a.name]) for a, b in permutations(primary, 2) if chosen[a.name]]
    if held:

        def cell(source: RunData, target: RunData) -> str:
            rule = chosen[source.name]
            if rule is None:
                return "n/a"
            return change(target, rule) + (" in sample" if source is target else "")

        lines += [
            "Held out: on each DEV120 run the alternative rule with the largest net gain is "
            "chosen (the earlier one above on a tie, even when none gains), leaving out a rule "
            "identical to `score_kgmon` on that run, and scored on the other run; the DEV40 "
            "runs are extra rows.",
            "",
            *_table(
                ["Scored on", "Subset", "Gap",
                 *(f"chosen on {a.name}: `{chosen[a.name]}`" if chosen[a.name]
                   else f"chosen on {a.name}: none" for a in primary)],
                [[r.name, r.subset, f"{gaps[r.name].tasks:.2f}", *(cell(a, r) for a in primary)]
                 for r in runs],
            ),
            "",
        ]
        best = [
            rule for rule in RULES[1:]
            if all(c == rule and gains[b.name][c] >= PROPOSAL_TASKS for _, b, c in held)
        ]
        if best:
            lines.append(
                f"**Proposal:** `{best[0]}` recovers at least {PROPOSAL_TASKS:g} tasks net in "
                "both held-out directions; a later task may make it the selection."
            )
        else:
            found = "; ".join(
                f"`{c}` chosen on {a.name}: {_signed(gains[b.name][c])} on {b.name}"
                for a, b, c in held
            )
            lines.append(
                f"No rule recovers {PROPOSAL_TASKS:g} tasks net in both held-out directions "
                f"(held out, net: {found}): no proposal."
            )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, reproduce the runs, and write the report, CSVs, and diagrams.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when None.

    Returns:
        0 on success.

    Raises:
        ReproductionError: If a run does not reproduce.
        ValueError: If a run spec is malformed or two runs share a name.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run", action="append", required=True, metavar="KEY=PATH",
                        help="a run's name and folder; repeat per run")
    parser.add_argument("--out", type=Path, required=True, help="report folder")
    parser.add_argument("--reps", type=int, default=REPS, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=SEED, help="bootstrap seed")
    args = parser.parse_args(argv)
    specs = []
    for word in args.run:
        name, sep, path = word.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"bad run spec {word!r}: expected KEY=PATH")
        specs.append((name, Path(path).expanduser()))
    if len({n for n, _ in specs}) != len(specs):
        raise ValueError("two runs share a name")
    runs = [reproduce(name, path) for name, path in specs]
    runs.sort(key=lambda r: r.subset != PRIMARY)
    results: dict[tuple[str, str], Calibration] = {}
    for run in runs:
        results[(run.name, "promotion")] = calibrate(promotion_items(run), reps=args.reps,
                                                     seed=args.seed)
        results[(run.name, "promotion-top")] = calibrate(
            promotion_items(run, top_agreement=True), reps=args.reps, seed=args.seed
        )
        if run.decoded:
            results[(run.name, "promotion-decoded")] = calibrate(
                promotion_items(run, decoded=True), reps=args.reps, seed=args.seed
            )
        results[(run.name, "selection")] = calibrate(selection_items(run), reps=args.reps,
                                                     seed=args.seed)
    curves = {run.name: coverage_curve(run) for run in runs}
    args.out.mkdir(parents=True, exist_ok=True)
    for run in runs:
        path = args.out / f"{_safe(run.name)}.csv"
        write_csv(run, path)
        print(f"wrote {path}")
    plots, note = write_plots(
        [r for r in runs if r.subset == PRIMARY], results, curves, args.out
    )
    for path in plots:
        print(f"wrote {path}")
    words = sys.argv[1:] if argv is None else list(argv)
    command = "python scripts/calibration_report.py " + " ".join(words).replace(
        str(Path.home()), "~"
    )
    lines = render(runs, results, curves, plots, note, command, args.reps, args.seed)
    report = args.out / "calibration.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
