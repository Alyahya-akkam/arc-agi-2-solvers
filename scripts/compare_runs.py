r"""Per-puzzle and per-output comparison of runs of one configuration (run P2.1-03).

Why: run ``P2.1-03``'s Below row (``docs/prerun/P2.1-03.md``) asks whether a DEV40 gap between
the split solver and the moved reference is a behavior difference (a token, view, or op that
differs) or the reference's run-to-run nondeterminism (identical inputs, different DFS
outcomes). Each run is read from its pulled logs only:

- the session log, cut to the arm's time window: per puzzle the training stats (loss,
  ``total_flos``), the peak memory lines, each ``decoding`` line's views, each ``scoring`` line,
  and the ``timeout after`` line of the 1200 s decode cap;
- the timing records, optional: ``train`` tokens and seconds, ``decode`` prompt tokens and
  seconds per DFS batch, ``selection`` tokens and seconds per rescored grid;
- ``candidates.json``: every candidate grid per view with its ``beam_score`` and ``score_aug``;
- the selection output (``selection.log``, or the session log of a run without one): the
  ``ALL_CORRECT`` lines, which name the candidates equal to the solution, and the
  ``score_kgmon`` correct set;
- ``submission.json``: the attempts, for ``arc26.evaluate.agreement``.

No solution file is read and no grid is written: an output's correct grid is a candidate that
some run's notebook marked ``ALL_CORRECT``, held in memory only. Credit is
``arc26.evaluate.TaskScore`` over membership in each run's ``score_kgmon`` set. Context runs
(``--context``) add only their solved outputs and correct candidates.

Writes ``tasks.md`` (per puzzle), ``outputs.md`` (per test output), and ``pairs.md`` (per pair
of runs) to ``--out``. Usage::

    python scripts/compare_runs.py --subset DEV40 --out reports/P2.1-03 --base arm1 \
        --run name=P2.1-03 log=runs/P2.1-03/arc26-split-dev40.log \
            arm=runs/P2.1-03/run_logs/r256-proj-emb \
        --run name=arm1 log=<P1.3-01 log> before=6630.5 arm=<P1.3-01 arm 1 folder> \
        --context name=r16-attn arm=<P1.3-01 arm 2 folder>

A run spec is ``key=value`` words: ``name`` and ``log`` (required for ``--run``), ``after`` and
``before`` (the arm's window in the log, seconds), ``arm`` (the default folder of the others),
``candidates``, ``selection``, ``submission``, and ``timing`` (a folder, or ``none``).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from types import ModuleType

from arc26.data import subset_ids
from arc26.evaluate import SubsetScore, TaskScore, agreement
from arc26.timing import DECODE, PUZZLE, SELECTION, TRAIN, read_records

SCRIPTS = Path(__file__).resolve().parent
KGMON = "('score_kgmon')"
SPEC_KEYS = frozenset(
    ("name", "log", "after", "before", "arm", "candidates", "selection", "submission", "timing")
)
DEFAULT_FILES = (
    ("candidates", "candidates.json"),
    ("selection", "selection.log"),
    ("submission", "submission.json"),
    ("timing", "timing"),
)


def _load(name: str) -> ModuleType:
    # Sibling scripts are loaded by path, as the tests load them: scripts/ is not a package.
    spec = importlib.util.spec_from_file_location(f"_arc26_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The session log parser and the notebook's line patterns (P1.1-01's timing report).
slt = _load("session_log_timing")


@dataclass
class Puzzle:
    """What one run logged for one puzzle.

    Attributes:
        loss: Mean training loss (the ``training stats`` line).
        flos: The trainer's ``total_flos``, a function of the training tokens and steps.
        train_mb: Peak MB allocated in training.
        inference_mb: Peak MB allocated in inference.
        timeout_after: Seconds of the ``timeout after`` line (the 1200 s decode cap), or None.
        batches: The views of each ``decoding`` line, in order.
        scored: ``(view, candidate index)`` of each ``scoring`` line, in order.
        train: ``(seconds, tokens)`` of the ``train`` record, or None.
        decode: ``(seconds, prompt tokens)`` per ``decode`` record, in order.
        selection: ``(seconds, tokens)`` per ``selection`` record, in order.
        puzzle_s: Seconds of the ``puzzle`` record, or None.
    """

    loss: float | None = None
    flos: float | None = None
    train_mb: int | None = None
    inference_mb: int | None = None
    timeout_after: float | None = None
    batches: list[tuple[str, ...]] = field(default_factory=list)
    scored: list[tuple[str, int]] = field(default_factory=list)
    train: tuple[float, int | None] | None = None
    decode: list[tuple[float, int | None]] = field(default_factory=list)
    selection: list[tuple[float, int | None]] = field(default_factory=list)
    puzzle_s: float | None = None


@dataclass
class Run:
    """One run's logs, parsed.

    Attributes:
        name: Label in the tables.
        candidates: Per test output (``<task>_<i>``), candidate key (``<view>.out<j>``) to
            ``{"beam_score", "score_aug", "solution"}``.
        correct_keys: Candidate keys of the ``ALL_CORRECT`` lines.
        solved: The ``score_kgmon`` correct set.
        submission: ``submission.json``.
        puzzles: Per task, what the session log and the timing records hold; empty for a
            context run.
        timed: Whether timing records were read.
    """

    name: str
    candidates: dict[str, dict[str, dict]]
    correct_keys: list[str]
    solved: set[str]
    submission: dict[str, object]
    puzzles: dict[str, Puzzle] = field(default_factory=dict)
    timed: bool = False

    def tokens(self, op: str) -> bool:
        """Return whether every ``decode`` or ``selection`` record of the run carries tokens.

        Args:
            op: ``DECODE`` or ``SELECTION``.

        Returns:
            False without timing records or when a record has no token count.
        """
        records = [r for p in self.puzzles.values() for r in getattr(p, op)]
        return self.timed and bool(records) and all(t is not None for _, t in records)


def parse_spec(words: Sequence[str], full: bool) -> dict[str, str]:
    """Read a run spec of ``key=value`` words and fill in the arm folder's default files.

    Args:
        words: The words after ``--run`` or ``--context``.
        full: A ``--run`` spec, which needs a log.

    Returns:
        The spec with ``candidates``, ``selection``, ``submission``, and ``timing`` set.

    Raises:
        ValueError: On a word without ``=``, an unknown key, or a missing required key.
    """
    spec: dict[str, str] = {}
    for word in words:
        key, sep, value = word.partition("=")
        if not sep or key not in SPEC_KEYS:
            raise ValueError(f"bad run spec word {word!r}")
        spec[key] = value
    for key in ("name", "log") if full else ("name",):
        if key not in spec:
            raise ValueError(f"run spec without {key}: {' '.join(words)}")
    arm = Path(spec.get("arm", "."))
    for key, default in DEFAULT_FILES:
        spec.setdefault(key, str(arm / default))
    return spec


def read_lines(path: str, after: float | None = None, before: float | None = None) -> list[str]:
    """Return the text lines of a Kaggle session log inside a time window, or of a text file.

    Args:
        path: A session log (a JSON array of entries) or a plain text file.
        after: Keep session log lines at or after this many seconds.
        before: Keep session log lines before this many seconds.

    Returns:
        The lines, in time order for a session log.

    Raises:
        ValueError: If a session log has skipped entries or is truncated.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if not text.startswith("["):
        return text.splitlines()
    lines, skipped, truncated = slt.parse_session_log(text)
    if skipped or truncated:
        raise ValueError(f"{path}: {skipped} entries skipped, truncated {truncated}")
    return [
        line.text
        for line in lines
        if (after is None or line.time >= after) and (before is None or line.time < before)
    ]


def kgmon_set(texts: Sequence[str]) -> set[str]:
    """Return the ``score_kgmon`` correct set: the line printed before its ``acc:`` line.

    Args:
        texts: The selection output's lines.

    Returns:
        The output keys (``<task>_<i>``) whose top two ``score_kgmon`` grids hold the solution.

    Raises:
        ValueError: If the selection output has no ``score_kgmon`` line.
    """
    for previous, text in zip(texts, texts[1:], strict=False):
        if text.rstrip().endswith(KGMON):
            return set() if previous.strip() == "set()" else set(ast.literal_eval(previous))
    raise ValueError("no score_kgmon line in the selection output")


def parse_puzzles(texts: Sequence[str], tasks: Sequence[str]) -> dict[str, Puzzle]:
    """Read each task's training, memory, cap, ``decoding``, and ``scoring`` lines.

    Args:
        texts: Session log lines of one arm, in time order.
        tasks: Task ids to keep.

    Returns:
        Per task found, its lines.

    Raises:
        ValueError: If a task was logged twice.
    """
    found: dict[str, Puzzle] = {}
    lines = [slt.LogLine(0.0, "stdout", text) for text in texts]
    for rank in slt.parse_ranks(lines).values():
        for parsed in rank.puzzles:
            if parsed.puzzle_id not in tasks:
                continue
            if parsed.puzzle_id in found:
                raise ValueError(f"puzzle {parsed.puzzle_id} logged twice")
            found[parsed.puzzle_id] = Puzzle(
                loss=parsed.training_loss,
                flos=parsed.total_flos,
                train_mb=parsed.train_mb,
                inference_mb=parsed.inference_mb,
                timeout_after=parsed.timeout_after,
            )
    for text in texts:
        match = slt.RANK_LINE.match(text)
        if match is None:
            continue
        message = match.group(2).strip()
        if decoding := slt.DECODING.match(message):
            views = tuple(slt.SUBKEY.findall(decoding.group(1)))
            if views and slt.task_of(views[0]) in found:
                found[slt.task_of(views[0])].batches.append(views)
        elif scoring := slt.SCORING.match(message):
            task = slt.task_of(scoring.group(1))
            if task in found:
                found[task].scored.append((scoring.group(1), int(scoring.group(2))))
    return found


def load_run(spec: Mapping[str, str], tasks: Sequence[str], full: bool) -> Run:
    """Read one run's files as its spec names them.

    Args:
        spec: From :func:`parse_spec`.
        tasks: The compared task ids.
        full: Also read the session log and the timing records.

    Returns:
        The parsed run.

    Raises:
        OSError: If a file cannot be read.
        ValueError: If a log is malformed or has invalid timing lines.
    """
    raw = json.loads(Path(spec["candidates"]).read_text(encoding="utf-8"))
    candidates: dict[str, dict[str, dict]] = defaultdict(dict)
    for view in sorted(raw):
        for index, sample in enumerate(raw[view]):
            candidates[view.split(".", 1)[0]][f"{view}.out{index}"] = sample
    after = float(spec["after"]) if "after" in spec else None
    before = float(spec["before"]) if "before" in spec else None
    selection = read_lines(spec["selection"], after, before)
    run = Run(
        spec["name"],
        dict(candidates),
        [m.group(1) for text in selection if (m := slt.ALL_CORRECT.match(text))],
        kgmon_set(selection),
        json.loads(Path(spec["submission"]).read_text(encoding="utf-8")),
    )
    if not full:
        return run
    run.puzzles = parse_puzzles(read_lines(spec["log"], after, before), tasks)
    timing = Path(spec["timing"])
    if spec["timing"] == "none" or not timing.is_dir():
        return run
    records, skipped = read_records(sorted(timing.glob("timing-*.jsonl")))
    if skipped:
        raise ValueError(f"{timing}: {skipped} invalid timing lines")
    run.timed = True
    for record in records:
        puzzle = run.puzzles.get(record.puzzle_id)
        if puzzle is None:
            continue
        if record.op == TRAIN:
            puzzle.train = (record.seconds, record.tokens)
        elif record.op in (DECODE, SELECTION):
            getattr(puzzle, record.op).append((record.seconds, record.tokens))
        elif record.op == PUZZLE:
            puzzle.puzzle_s = record.seconds
    return run


def correct_grids(runs: Sequence[Run]) -> dict[str, object]:
    """Return each test output's correct grid from every run's ``ALL_CORRECT`` candidates.

    Args:
        runs: All runs, context runs included.

    Returns:
        Output key to its correct grid, for the outputs some run marked.

    Raises:
        ValueError: If a marked key is missing from its candidates or two marked grids differ.
    """
    grids: dict[str, object] = {}
    for run in runs:
        for key in run.correct_keys:
            base = key.split(".", 1)[0]
            sample = run.candidates.get(base, {}).get(key)
            if sample is None:
                raise ValueError(f"{run.name}: ALL_CORRECT {key} is not in its candidates")
            if grids.setdefault(base, sample["solution"]) != sample["solution"]:
                raise ValueError(f"{run.name}: {base} has two different correct grids")
    return grids


@dataclass(frozen=True)
class OutputStats:
    """One run's candidates for one test output.

    Attributes:
        candidates: Candidates over all views.
        distinct: Distinct grids.
        correct: Candidates equal to the correct grid (0 when none is known).
        rank: The correct grid's place under ``score_kgmon`` (1 and 2 are submitted), or None.
        solved: The output is in the run's ``score_kgmon`` set.
        submitted: An attempt of ``submission.json`` equals the correct grid.
    """

    candidates: int
    distinct: int
    correct: int
    rank: int | None
    solved: bool
    submitted: bool


def output_stats(run: Run, base: str, grid: object | None) -> OutputStats:
    """Count one run's candidates for an output and rank its correct grid.

    Args:
        run: A parsed run.
        base: Output key ``<task>_<i>``.
        grid: The output's correct grid, or None when no run marked one.

    Returns:
        The output's counts.
    """
    from arc26.nvarc.arc_decoder import score_kgmon  # the notebook's selection; needs numpy

    guesses = run.candidates.get(base, {})
    ordered = [[list(row) for row in g] for g in score_kgmon(guesses)] if guesses else []
    task, index = base.rsplit("_", 1)
    outputs = run.submission.get(task)
    entry = outputs[int(index)] if isinstance(outputs, list) else {}
    known = grid is not None
    return OutputStats(
        len(guesses),
        len(ordered),
        sum(sample["solution"] == grid for sample in guesses.values()) if known else 0,
        next((i + 1 for i, g in enumerate(ordered) if g == grid), None) if known else None,
        base in run.solved,
        known and grid in (entry.get("attempt_1"), entry.get("attempt_2")),
    )


def credit(run: Run, tasks: Sequence[str], outputs: Mapping[str, int], label: str) -> SubsetScore:
    """Score a run with ``arc26.evaluate`` from its ``score_kgmon`` set.

    Args:
        run: A parsed run.
        tasks: Task ids.
        outputs: Test outputs per task.
        label: Subset name.

    Returns:
        The per-task score.
    """
    return SubsetScore(
        label,
        tuple(
            TaskScore(t, tuple(f"{t}_{i}" in run.solved for i in range(outputs[t])))
            for t in tasks
        ),
    )


def _corr(u: Sequence[float], v: Sequence[float]) -> float | None:
    """Pearson correlation; None for a constant vector."""
    mu, mv = statistics.fmean(u), statistics.fmean(v)
    su = math.sqrt(sum((x - mu) ** 2 for x in u))
    sv = math.sqrt(sum((y - mv) ** 2 for y in v))
    if su == 0 or sv == 0:
        return None
    return sum((x - mu) * (y - mv) for x, y in zip(u, v, strict=True)) / (su * sv)


def _views(run: Run, tasks: Sequence[str]) -> dict[str, list[dict]]:
    """Candidates per view (the candidate file's name), compared tasks only."""
    views: dict[str, list[dict]] = defaultdict(list)
    for base, samples in run.candidates.items():
        if base.rsplit("_", 1)[0] in tasks:
            for key, sample in samples.items():
                views[key.rsplit(".out", 1)[0]].append(sample)
    return views


def _grids(samples: Sequence[dict]) -> list[str]:
    return [json.dumps(sample["solution"]) for sample in samples]


def _rescored(run: Run) -> dict[tuple[str, str], tuple[list[float], int | None]]:
    """Per (output, grid) rescored in the run: its ``score_aug`` and its ``selection`` tokens."""
    found: dict[tuple[str, str], tuple[list[float], int | None]] = {}
    for task, puzzle in run.puzzles.items():
        tokens = [t for _, t in puzzle.selection] if run.timed else []
        if run.timed and len(tokens) != len(puzzle.scored):
            raise ValueError(f"{run.name} {task}: selection records differ from scoring lines")
        for n, (view, index) in enumerate(puzzle.scored):
            base = view.split(".", 1)[0]
            sample = run.candidates[base][f"{view}.out{index}"]
            key = (base, json.dumps(sample["solution"]))
            found[key] = (sample["score_aug"], tokens[n] if tokens else None)
    return found


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _of(hits: int, total: int) -> str:
    return f"{hits} of {total}"


def _prefix_equal(first: Sequence[object], second: Sequence[object]) -> bool:
    """Equal over the common length (a run cut by the decode cap has fewer batches)."""
    n = min(len(first), len(second))
    return list(first[:n]) == list(second[:n])


def pair_rows(x: Run, y: Run, tasks: Sequence[str], label: str) -> list[tuple[str, str]]:
    """Compare two runs on every measure of ``pairs.md``.

    Args:
        x: First run.
        y: Second run.
        tasks: Compared task ids.
        label: Subset name.

    Returns:
        ``(measure, value)`` rows.
    """
    agree = agreement(x.submission, y.submission, tasks, label)
    n_out = agree.test_outputs
    both = sum(o.both for o in agree.outputs)
    first = sum(o.attempt_1 for o in agree.outputs)
    keep = {f"{t}_" for t in tasks}
    sx = {k for k in x.solved if k[: k.rindex("_") + 1] in keep}
    sy = {k for k in y.solved if k[: k.rindex("_") + 1] in keep}
    rows = [
        ("attempt pairs identical in order, $g$", f"{agree.g:.3f} ({_of(both, n_out)})"),
        ("first attempts identical", f"{agree.attempt_1:.3f} ({_of(first, n_out)})"),
        ("outputs solved by the first only", ", ".join(f"`{k}`" for k in sorted(sx - sy))),
        ("outputs solved by the second only", ", ".join(f"`{k}`" for k in sorted(sy - sx))),
    ]
    px, py = x.puzzles, y.puzzles
    common = [t for t in tasks if t in px and t in py]
    n = len(common)
    flos = sum(px[t].flos is not None and px[t].flos == py[t].flos for t in common)
    views = sum(_prefix_equal(px[t].batches, py[t].batches) for t in common)
    batches = sum(min(len(px[t].batches), len(py[t].batches)) for t in common)
    rows += [
        ("puzzles with equal `total_flos` (training tokens and steps)", _of(flos, n)),
        ("puzzles whose common decode batches hold equal views, in order",
         f"{_of(views, n)} ({batches} batches)"),
    ]
    prompts = sum(
        _prefix_equal([b for _, b in px[t].decode], [b for _, b in py[t].decode])
        for t in common
    )
    rows.append((
        "puzzles whose common decode batches have equal prompt tokens",
        _of(prompts, n) if x.tokens(DECODE) and y.tokens(DECODE) else "n/a (no token counts)",
    ))
    rows += [
        ("puzzles with equal peak training MB",
         _of(sum(px[t].train_mb == py[t].train_mb for t in common), n)),
        ("puzzles with equal peak inference MB",
         _of(sum(px[t].inference_mb == py[t].inference_mb for t in common), n)),
    ]
    losses = [(px[t].loss, py[t].loss) for t in common if px[t].loss and py[t].loss]
    rel = [abs(a - b) / b for a, b in losses]
    rows += [
        ("training loss: median and max relative difference",
         f"{statistics.median(rel):.4f}, {max(rel):.4f}" if rel else "n/a"),
        ("training loss lower in the first run; equal",
         f"{_of(sum(a < b for a, b in losses), len(losses))}; "
         f"{sum(a == b for a, b in losses)}"),
    ]
    vx, vy = _views(x, tasks), _views(y, tasks)
    shared_views = sorted(set(vx) & set(vy))
    same = sum(_grids(vx[v]) == _grids(vy[v]) for v in shared_views)
    gaps = [
        abs(a["beam_score"] - b["beam_score"])
        for v in shared_views
        for a in vx[v]
        for b in vy[v]
        if a["solution"] == b["solution"]
    ]
    rows += [
        ("views with candidates: first, second, both",
         f"{len(vx)}, {len(vy)}, {len(shared_views)}"),
        ("views in both with identical candidate lists", _of(same, len(shared_views))),
        ("same grid in the same view: median and max beam-score gap",
         f"{_num(statistics.median(gaps))}, {_num(max(gaps))} ({len(gaps)} pairs)"
         if gaps else "n/a"),
    ]
    rx, ry = _rescored(x), _rescored(y)
    shared = sorted(set(rx) & set(ry))
    corr = [c for k in shared if (c := _corr(rx[k][0], ry[k][0])) is not None]
    rolled = [
        c for k in shared if (c := _corr(rx[k][0], ry[k][0][1:] + ry[k][0][:1])) is not None
    ]
    rows.append((
        "grids rescored in both: median correlation of their 8 view scores "
        "(the second run's rolled by one view)",
        f"{_num(statistics.median(corr))} ({_num(statistics.median(rolled))}), "
        f"{len(shared)} grids" if corr else "n/a",
    ))
    rows.append((
        "grids rescored in both with equal rescoring tokens",
        _of(sum(rx[k][1] == ry[k][1] for k in shared), len(shared))
        if x.tokens(SELECTION) and y.tokens(SELECTION)
        else "n/a (no token counts)",
    ))
    timed = x.timed and y.timed
    train = sorted(px[t].train[0] / py[t].train[0] for t in common if px[t].train and py[t].train)
    batch = sorted(
        a / b for t in common for (a, _), (b, _) in zip(px[t].decode, py[t].decode, strict=False)
    )
    rows += [
        ("train seconds per puzzle, first over second: median (min, max)",
         f"{_num(statistics.median(train))} ({_num(train[0])}, {_num(train[-1])})"
         if timed and train else "n/a"),
        ("decode seconds per common batch, first over second: median (p10, p90); in all",
         f"{_num(statistics.median(batch))} ({_num(_rank(batch, 0.1))}, "
         f"{_num(_rank(batch, 0.9))}); {_num(_common_decode(x, y, common))} "
         f"({len(batch)} batches)" if timed and batch else "n/a"),
    ]
    return rows


def _rank(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of sorted values."""
    return values[max(0, math.ceil(q * len(values)) - 1)]


def _common_decode(x: Run, y: Run, tasks: Sequence[str]) -> float:
    """Total decode seconds of x over y on the batches both decoded."""
    pairs = [
        (a, b)
        for t in tasks
        for (a, _), (b, _) in zip(x.puzzles[t].decode, y.puzzles[t].decode, strict=False)
    ]
    return sum(a for a, _ in pairs) / sum(b for _, b in pairs)


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(str(c) for c in row) + " |" for row in rows),
    ]


def _joined(values: Sequence[object]) -> str:
    return " / ".join("-" if v is None else str(v) for v in values)


def render_tasks(
    runs: Sequence[Run], base: Run, tasks: Sequence[str], scores: Sequence[SubsetScore]
) -> list[str]:
    """Build the per-puzzle tables of ``tasks.md``.

    Args:
        runs: The compared runs, in column order.
        base: The run whose training tokens are shown.
        tasks: Task ids.
        scores: Each run's score, in run order.

    Returns:
        The Markdown lines after the header.
    """
    lines = [
        "Scores (`arc26.evaluate` over each run's `score_kgmon` set): "
        + "; ".join(
            f"{r.name} {s.score:g} of {s.tasks} ({_of(s.solved_outputs, s.test_outputs)} outputs)"
            for r, s in zip(runs, scores, strict=True)
        )
        + ".",
        "",
        f"Columns with ` / ` give the runs in the order above. $c$: the task's credit. Train "
        f"tokens: {base.name}'s `train` record. Equal: every run has the same `total_flos`, the "
        "same views in each common `decoding` line, and (runs with `decode` token counts) the "
        "same prompt tokens per common batch. MB: peak allocated in training and in inference. "
        "Cap: the `timeout after` seconds of the 1200 s decode cap.",
        "",
    ]
    timed = [r for r in runs if r.tokens(DECODE)]
    rows = []
    for i, t in enumerate(tasks):
        ps = [r.puzzles[t] for r in runs]
        equal = (
            all(p.flos == ps[0].flos for p in ps)
            and all(_prefix_equal(p.batches, ps[0].batches) for p in ps)
            and all(
                _prefix_equal([b for _, b in r.puzzles[t].decode],
                              [b for _, b in timed[0].puzzles[t].decode])
                for r in timed
            )
        )
        rows.append([
            f"`{t}`",
            _joined([f"{s.per_task[i].score:g}" for s in scores]),
            base.puzzles[t].train[1] if base.puzzles[t].train else "n/a",
            "yes" if equal else "no",
            _joined([len(p.batches) for p in ps]),
            _joined([p.train_mb for p in ps]),
            _joined([p.inference_mb for p in ps]),
            _joined([_num(p.timeout_after, 1) if p.timeout_after else None for p in ps]),
            _joined([_num(p.loss, 5) for p in ps]),
        ])
    lines += _table(
        ["Task", "$c$", "Train tokens", "Equal", "Decode batches", "Train MB", "Inference MB",
         "Cap s", "Training loss"],
        rows,
    )
    seconds = [r for r in runs if r.timed]
    if not seconds:
        return lines
    summary = []
    for r in seconds:
        ps = [r.puzzles[t] for t in tasks]
        train = sum(p.train[0] for p in ps if p.train)
        tokens = sum(p.train[1] or 0 for p in ps if p.train)
        dec = [s for p in ps for s, _ in p.decode]
        sel = [s for p in ps for s, _ in p.selection]
        summary.append([
            r.name,
            _num(train / len(ps), 1),
            _num(sum(dec) / len(ps), 1),
            len(dec),
            _num(sum(dec) / len(dec), 2) if dec else "n/a",
            len(sel),
            _num(sum(sel) / len(sel), 3) if sel else "n/a",
            _num(sum(sel) / len(ps), 1),
            _num(sum(p.puzzle_s or 0 for p in ps) / len(ps), 1),
            _num(tokens / train, 0) if train else "n/a",
        ])
    lines += [
        "",
        f"Means over the {len(tasks)} puzzles, from the timing records:",
        "",
        *_table(
            ["Run", "Train s", "Decode s", "Decode batches", "s per batch", "Rescored grids",
             "s per grid", "Selection s", "Puzzle s", "Train tokens per s"],
            summary,
        ),
        "",
        "Seconds per puzzle (" + ", ".join(r.name for r in seconds) + "): train; decode, in "
        "all and per batch; rescored grids and their seconds; the puzzle.",
        "",
    ]
    rows = []
    for t in tasks:
        cells = [f"`{t}`"]
        for r in seconds:
            p = r.puzzles[t]
            dec = sum(s for s, _ in p.decode)
            cells += [
                _num(p.train[0], 1) if p.train else "n/a",
                f"{dec:.1f}; {dec / len(p.decode):.1f}" if p.decode else "0",
                f"{len(p.selection)}; {sum(s for s, _ in p.selection):.1f}",
                _num(p.puzzle_s, 1),
            ]
        rows.append(cells)
    header = ["Task"]
    for r in seconds:
        header += [f"{r.name} train", f"{r.name} decode; per batch", f"{r.name} rescored; s",
                   f"{r.name} puzzle"]
    return lines + _table(header, rows)


def render_outputs(
    runs: Sequence[Run], context: Sequence[Run], tasks: Sequence[str], outputs: Mapping[str, int]
) -> list[str]:
    """Build the per-output table of ``outputs.md``.

    Args:
        runs: The compared runs, in column order.
        context: Runs that add solved outputs and correct grids only.
        tasks: Task ids.
        outputs: Test outputs per task.

    Returns:
        The Markdown lines after the header.

    Raises:
        ValueError: If a run's ``score_kgmon`` set and its submission disagree on an output.
    """
    grids = correct_grids([*runs, *context])
    rows = []
    totals = [0] * len(runs)
    for t in tasks:
        for i in range(outputs[t]):
            key = f"{t}_{i}"
            stats = [output_stats(r, key, grids.get(key)) for r in runs]
            for j, (r, s) in enumerate(zip(runs, stats, strict=True)):
                totals[j] += s.correct
                if key in grids and s.solved != s.submitted:
                    raise ValueError(f"{r.name} {key}: score_kgmon set and submission differ")
            if key not in grids:
                continue
            solved = [r.name for r, s in zip(runs, stats, strict=True) if s.solved]
            rows.append([
                f"`{key}`",
                ", ".join(solved) or "none",
                _of(sum(key in c.solved for c in context), len(context)) if context else "n/a",
                _joined([s.candidates for s in stats]),
                _joined([s.distinct for s in stats]),
                _joined([s.correct for s in stats]),
                _joined([s.rank for s in stats]),
            ])
    names = ", ".join(c.name for c in context) or "none"
    counts = [
        sum(len(v) for b, v in r.candidates.items() if b.rsplit("_", 1)[0] in tasks) for r in runs
    ]
    return [
        "Rows: the outputs with a correct candidate in some run, compared or context; no run "
        "has one for the others. Columns with ` / ` give the compared runs in the order above. "
        f"Context runs: {names}. Rank: the correct grid's place under `score_kgmon` (ranks 1 "
        "and 2 are submitted), `-` when no candidate equals it.",
        "",
        *_table(
            ["Output", "Solved by", "Solved in context runs", "Candidates", "Distinct grids",
             "Correct candidates", "Rank"],
            rows,
        ),
        "",
        "Correct candidates in all: "
        + "; ".join(
            f"{r.name} {_of(n, c)}" for r, n, c in zip(runs, totals, counts, strict=True)
        )
        + ".",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, read the runs, and write ``tasks.md``, ``outputs.md``, ``pairs.md``.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when None.

    Returns:
        0 on success.

    Raises:
        ValueError: If a compared run has no log lines for a task of the subset.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--subset", required=True, help="DEV4, DEV40, or DEV120")
    parser.add_argument("--out", type=Path, required=True, help="folder of the three reports")
    parser.add_argument("--base", help="run whose training tokens tasks.md shows (the first)")
    parser.add_argument("--run", nargs="+", action="append", required=True, metavar="KEY=VALUE")
    parser.add_argument("--context", nargs="+", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    tasks = subset_ids(args.subset)
    runs = [load_run(parse_spec(words, True), tasks, True) for words in args.run]
    context = [load_run(parse_spec(words, False), tasks, False) for words in args.context]
    for run in runs:
        missing = [t for t in tasks if t not in run.puzzles]
        if missing:
            raise ValueError(f"{run.name}: no log lines for {missing}")
    base = next((r for r in runs if r.name == args.base), runs[0])
    outputs = {t: len(base.submission[t]) for t in tasks}
    words = sys.argv[1:] if argv is None else list(argv)
    command = "python scripts/compare_runs.py " + " ".join(words).replace(str(Path.home()), "~")
    head = [
        f"Generated by `scripts/compare_runs.py` from the pulled logs; {args.subset}, "
        f"{len(tasks)} tasks, {sum(outputs.values())} test outputs; no solution file read. "
        "Command, from the repository root:",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    names = ", ".join(r.name for r in runs)
    scores = [credit(r, tasks, outputs, args.subset) for r in runs]
    measures: dict[str, list[str]] = {}
    pairs = list(combinations(runs, 2))
    for x, y in pairs:
        for measure, value in pair_rows(x, y, tasks, args.subset):
            measures.setdefault(measure, []).append(value or "none")
    reports = {
        "tasks.md": [f"# Per puzzle: {names}", "", *head,
                     *render_tasks(runs, base, tasks, scores)],
        "outputs.md": [f"# Per test output: {names}", "", *head,
                       *render_outputs(runs, context, tasks, outputs)],
        "pairs.md": [
            f"# Per pair of runs: {names}",
            "",
            *head,
            *_table(["Measure", *(f"{x.name} against {y.name}" for x, y in pairs)],
                    [[m, *v] for m, v in measures.items()]),
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, lines in reports.items():
        (args.out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
