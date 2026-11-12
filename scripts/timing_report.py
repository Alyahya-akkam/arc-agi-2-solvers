r"""Per-puzzle timing table and linear cost model from a run's timing logs (P0.3a, BRIEF B3).

Reads the ``timing-<worker>.jsonl`` files that ``arc26.timing`` writes, one per worker process,
and writes Markdown with three sections:

1. Per puzzle: train, decode, selection, overhead, and total seconds, and the token counts of
   train, decode, and selection (summed over the puzzle's records of each op; ``n/a`` when none
   carries a count). Overhead is the ``puzzle`` record's seconds minus train, decode, and
   selection. Without a ``puzzle`` record, overhead is the sum of the puzzle's other ops and the
   total is the sum of all its ops.
2. Ops outside puzzles: records with a null ``puzzle_id``, such as each worker's model loading
   and the main-process selection.
3. Cost model per op: the least-squares fit of :math:`\text{seconds} = a + b \cdot \text{tokens}`
   over the op's records that carry a token count, with :math:`a`, :math:`b`, :math:`R^2`, and
   the sample count :math:`n`. With fewer than two distinct token counts only :math:`a` (the
   mean) is reported; :math:`R^2` is n/a when every sample takes the same time.

Lines that are not valid records (a truncated last line after a hard kill) are skipped and
counted in the report.

Usage::

    python scripts/timing_report.py <dir-or-file>... [--out <report.md>] [--title <title>]

A directory is searched recursively for ``timing-*.jsonl``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from arc26.timing import (
    DECODE,
    FILE_PREFIX,
    FILE_SUFFIX,
    PUZZLE,
    SELECTION,
    TRAIN,
    TimingRecord,
    read_records,
)

PHASES = (TRAIN, DECODE, SELECTION)


@dataclass(frozen=True)
class Fit:
    """Least-squares fit of seconds against tokens for one op.

    Attributes:
        op: Operation name.
        n: Records with a token count (the samples).
        a: Intercept in seconds; the mean seconds when ``b`` is None; None when ``n`` is 0.
        b: Slope in seconds per token; None with fewer than two distinct token counts.
        r2: Coefficient of determination; None without a slope or when all seconds are equal.
    """

    op: str
    n: int
    a: float | None
    b: float | None
    r2: float | None


@dataclass(frozen=True)
class PuzzleRow:
    """Timing totals of one puzzle.

    Attributes:
        puzzle_id: Task id.
        workers: Worker labels that logged the puzzle, comma-separated.
        seconds: Seconds per phase (train, decode, selection).
        tokens: Token count per phase, None when no record of the phase carries one.
        overhead: Total minus the three phases.
        total: The puzzle's wall time.
    """

    puzzle_id: str
    workers: str
    seconds: dict[str, float]
    tokens: dict[str, int | None]
    overhead: float
    total: float


def fit_linear(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[float, float | None, float | None]:
    r"""Fit :math:`y = a + b x` by ordinary least squares.

    Why: pure Python (no numpy in the package). With :math:`\bar x, \bar y` the means,
    :math:`b = \sum (x - \bar x)(y - \bar y) / \sum (x - \bar x)^2`,
    :math:`a = \bar y - b \bar x`, and :math:`R^2 = 1 - SS_{res} / SS_{tot}`.

    Args:
        xs: Predictor values (token counts).
        ys: Responses (seconds), as many as ``xs``.

    Returns:
        ``(a, b, r2)``. ``b`` and ``r2`` are None when the ``xs`` do not vary (then ``a`` is the
        mean of ``ys``); ``r2`` is None when the ``ys`` do not vary.

    Raises:
        ValueError: If the inputs are empty or differ in length.
    """
    n = len(xs)
    if n == 0 or n != len(ys):
        raise ValueError(f"need equally many xs and ys, at least one; got {n} and {len(ys)}")
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(ys) / n
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return mean_y, None, None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    a = mean_y - b * mean_x
    ss_res = math.fsum((y - a - b * x) ** 2 for x, y in zip(xs, ys, strict=True))
    ss_tot = math.fsum((y - mean_y) ** 2 for y in ys)
    return a, b, (1.0 - ss_res / ss_tot if ss_tot > 0 else None)


def fit_ops(records: Iterable[TimingRecord]) -> list[Fit]:
    """Fit the linear cost model for every op, over its records that carry a token count.

    Args:
        records: Timing records.

    Returns:
        One fit per op name, sorted by name; ``n`` is 0 for an op without token counts.
    """
    samples: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        points = samples.setdefault(record.op, [])
        if record.tokens is not None:
            points.append((record.tokens, record.seconds))
    fits = []
    for op in sorted(samples):
        points = samples[op]
        if not points:
            fits.append(Fit(op, 0, None, None, None))
            continue
        a, b, r2 = fit_linear([x for x, _ in points], [y for _, y in points])
        fits.append(Fit(op, len(points), a, b, r2))
    return fits


def per_puzzle(records: Iterable[TimingRecord]) -> list[PuzzleRow]:
    """Aggregate records into one row per puzzle.

    Args:
        records: Timing records; those with a null ``puzzle_id`` are ignored here.

    Returns:
        One row per puzzle, sorted by puzzle id.
    """
    by_puzzle: dict[str, list[TimingRecord]] = defaultdict(list)
    for record in records:
        if record.puzzle_id is not None:
            by_puzzle[record.puzzle_id].append(record)
    rows = []
    for puzzle_id in sorted(by_puzzle):
        group = by_puzzle[puzzle_id]
        seconds = {op: math.fsum(r.seconds for r in group if r.op == op) for op in PHASES}
        tokens: dict[str, int | None] = {}
        for op in PHASES:
            counts = [r.tokens for r in group if r.op == op and r.tokens is not None]
            tokens[op] = sum(counts) if counts else None
        phases = math.fsum(seconds.values())
        spans = [r.seconds for r in group if r.op == PUZZLE]
        if spans:
            total = math.fsum(spans)
            overhead = total - phases
        else:
            overhead = math.fsum(r.seconds for r in group if r.op not in PHASES)
            total = phases + overhead
        workers = ",".join(sorted({r.worker for r in group}))
        rows.append(PuzzleRow(puzzle_id, workers, seconds, tokens, overhead, total))
    return rows


def ops_outside_puzzles(records: Iterable[TimingRecord]) -> list[tuple[str, int, float]]:
    """Summarize records with a null ``puzzle_id`` (model loading, main-process selection) by op.

    Args:
        records: Timing records.

    Returns:
        ``(op, count, total seconds)`` per op, sorted by op.
    """
    totals: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.puzzle_id is None:
            totals[record.op].append(record.seconds)
    return [(op, len(values), math.fsum(values)) for op, values in sorted(totals.items())]


def _seconds(value: float) -> str:
    return f"{value:.2f}"


def _optional(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _token_cells(rows: Sequence[PuzzleRow], reduce: str) -> list[str]:
    cells = []
    for op in PHASES:
        known = [row.tokens[op] for row in rows if row.tokens[op] is not None]
        if not known:
            cells.append("n/a")
        elif reduce == "sum":
            cells.append(str(sum(known)))
        else:
            cells.append(f"{sum(known) / len(known):.0f}")
    return cells


def render(
    records: Sequence[TimingRecord], skipped: int, sources: Sequence[Path], title: str
) -> str:
    """Render the timing report as Markdown.

    Args:
        records: Timing records of the run.
        skipped: Lines that were not valid records.
        sources: Files the records came from.
        title: Report heading.

    Returns:
        The Markdown text, ending in a newline.
    """
    rows = per_puzzle(records)
    lines = [
        f"# {title}",
        "",
        f"Records: {len(records)}; skipped lines: {skipped}; files: "
        + ", ".join(f"`{path}`" for path in sources)
        + ".",
        "",
        "## Per puzzle",
        "",
    ]
    if rows:
        lines += [
            "| Puzzle | Worker | Train s | Decode s | Selection s | Overhead s | Total s "
            "| Train tokens | Decode tokens | Selection tokens |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            cells = [row.puzzle_id, row.workers]
            cells += [_seconds(row.seconds[op]) for op in PHASES]
            cells += [_seconds(row.overhead), _seconds(row.total)]
            cells += ["n/a" if row.tokens[op] is None else str(row.tokens[op]) for op in PHASES]
            lines.append("| " + " | ".join(cells) + " |")
        count = len(rows)
        for label, reduce in ((f"mean of {count}", "mean"), ("sum", "sum")):
            scale = 1.0 / count if reduce == "mean" else 1.0
            cells = [label, ""]
            cells += [_seconds(scale * math.fsum(r.seconds[op] for r in rows)) for op in PHASES]
            cells += [
                _seconds(scale * math.fsum(r.overhead for r in rows)),
                _seconds(scale * math.fsum(r.total for r in rows)),
            ]
            cells += _token_cells(rows, reduce)
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            f"Overhead: the `{PUZZLE}` record minus train, decode, and selection; for a puzzle "
            f"without a `{PUZZLE}` record, the sum of its other ops. Token means run over the "
            "puzzles that report a count.",
        ]
    else:
        lines.append("No puzzle records.")

    ops = ops_outside_puzzles(records)
    if ops:
        lines += ["", "## Ops outside puzzles", "", "| Op | Records | Total s | Mean s |"]
        lines.append("|---|---:|---:|---:|")
        for op, count, total in ops:
            lines.append(f"| {op} | {count} | {_seconds(total)} | {_seconds(total / count)} |")

    lines += [
        "",
        "## Cost model: seconds = a + b * tokens",
        "",
        "| Op | n | a (s) | b (s per token) | R^2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for fit in fit_ops(records):
        lines.append(
            f"| {fit.op} | {fit.n} | {_optional(fit.a, '.4f')} | {_optional(fit.b, '.4e')} "
            f"| {_optional(fit.r2, '.4f')} |"
        )
    lines += [
        "",
        "Least squares over the op's records that carry a token count (n); a alone is the mean "
        "when the token counts do not vary.",
    ]
    return "\n".join(lines) + "\n"


def timing_files(paths: Iterable[Path]) -> list[Path]:
    """Expand directories into their timing files.

    Args:
        paths: Timing files or run directories.

    Returns:
        Files in argument order, each directory's ``timing-*.jsonl`` sorted, without repeats.

    Raises:
        FileNotFoundError: If a path does not exist.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found += sorted(path.rglob(f"{FILE_PREFIX}*{FILE_SUFFIX}"))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return list(dict.fromkeys(found))


def main(argv: Sequence[str] | None = None) -> int:
    """Write the timing report of a run.

    Args:
        argv: Command-line arguments without the program name; ``sys.argv[1:]`` when None.

    Returns:
        0 on success, 1 when no valid record is found or a path does not exist.
    """
    parser = argparse.ArgumentParser(description="Per-puzzle timing table and cost model.")
    parser.add_argument("paths", nargs="+", type=Path, help="timing files or run directories")
    parser.add_argument("--out", type=Path, help="write the Markdown here instead of stdout")
    parser.add_argument("--title", default="Timing report", help="report heading")
    args = parser.parse_args(argv)
    try:
        sources = timing_files(args.paths)
        records, skipped = read_records(sources)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not records:
        print(
            f"error: no valid timing records (files: {len(sources)}, skipped lines: {skipped})",
            file=sys.stderr,
        )
        return 1
    text = render(records, skipped, sources, args.title)
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
