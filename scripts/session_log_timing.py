r"""Timing records and run summary from a Kaggle session log of the reference solver (P1.1, B3).

Run ``P1.1-01`` predates the timing wiring (P0.3b), so its per-op times come from the timestamps
of the lines the reference's workers print (``arc26/nvarc/arc_solver.py``, ``worker``). Two
commands:

- ``records`` reads the session log and writes ``arc26.timing`` records, one file per rank
  (``timing-rank<r>.jsonl``, through ``arc26.timing.TimingLogger``), plus ``puzzles.csv`` with
  the per-puzzle extras. Pure Python: the standard library and ``arc26``.
- ``summary`` turns those files, the run's ``candidates.json`` and ``submission.json``, and K0's
  per-task statistics into ``summary.md`` (per-op statistics, cost model, workers, the 1200 s
  cap, the B3 comparison, solved tasks, candidate statistics) and ``outputs.csv`` (one row per
  test output). It imports ``arc26.evaluate`` for the metric and the moved ``score_kgmon`` for
  candidate ranks (numpy, a dev dependency); it never reads a solutions file.

Parsing assumptions of ``records`` (each is checked by ``summary`` where the log allows):

1. The log is a JSON array of ``{"stream_name", "time", "data"}`` objects, ``time`` in seconds
   since the kernel started. A truncated array is read up to its last complete object. An entry
   that is not an object with a finite numeric ``time`` and a string ``data`` is skipped and
   counted. ``data`` may hold several lines; each gets the entry's time. Within a line, a
   carriage return overwrites what precedes it (progress bars), so only the text after the last
   ``\r`` is parsed. Lines are processed in time order (a stable sort; the run's log is already
   ordered).
2. Emission time: the solver runs with ``PYTHONUNBUFFERED=1``, so a line's ``time`` is when its
   worker printed it, up to the streaming latency, which is assumed equal for all lines.
3. A worker line starts with ``[Rank <r>] ``; any other line (the notebook, the main process,
   library banners, ``ALL_CORRECT``) never opens or closes an interval. A rank line that matches
   no known message (for example two prints interleaved) closes the rank's open interval, opens
   none, and is counted.
4. Puzzle span: from the rank's previous ``finished`` line to the puzzle's ``finished`` line. The
   worker loop takes the next key right after printing ``finished``, so nothing else lies in
   between. No line marks the start of a rank's first puzzle: ``start!`` is printed before the
   worker loads the model. Its start is therefore the ``finished`` time minus the printed
   seconds (the worker's own clock), and the time from ``start!`` to that start is one
   ``model_load`` record with a null ``puzzle_id``. ``summary`` checks every other span against
   ``finished <id> in <s>s`` (within 2 s).
5. Train: one record per puzzle with ``seconds`` = ``train_runtime`` from the printed
   ``TrainOutput`` and ``tokens`` = :math:`T = \text{total\_flos} / (6P)` (see
   :data:`REFERENCE_PARAMS`). Training prints nothing (its output is redirected), so the first
   rank line of a puzzle comes after training; ``summary`` checks that this line is at least
   ``train_runtime`` after the puzzle start.
6. Decode and selection: one record per interval between consecutive lines of the same rank. A
   ``decoding`` line opens a decode interval (one ``inference_turbo_dfs`` call on a batch of four
   augmented views); a ``scoring`` line opens a selection interval (two ``calc_scores`` calls on
   one new candidate). The next line of that rank closes the interval, whatever it is (in
   practice ``scoring``, ``decoding``, ``timeout``, or ``allocated ... for inference``). Their
   token counts are not in the log, so ``tokens`` is null.
7. A puzzle without a ``finished`` line (a hard stop) keeps its train, decode, and selection
   records but gets no ``puzzle`` record, and its open interval is dropped; ``puzzles.csv`` marks
   it incomplete.

Usage::

    python scripts/session_log_timing.py records <session.log> --out <dir> [--params P]
    python scripts/session_log_timing.py summary --timing <dir> --log <session.log> \
        --candidates <candidates.json> --submission <submission.json> \
        --task-stats <env_probe.json> --stages <stages.json> --out <summary.md>
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from arc26.evaluate import SubsetScore, TaskScore, score_task
from arc26.timing import (
    DECODE,
    FILE_PREFIX,
    FILE_SUFFIX,
    PUZZLE,
    SELECTION,
    TRAIN,
    TimingLogger,
    TimingRecord,
    read_records,
)

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str) -> ModuleType:
    # Sibling scripts are loaded by path, as the tests load them: scripts/ is not a package.
    spec = importlib.util.spec_from_file_location(f"_arc26_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


# fit_linear and timing_files, so summary.md and timing_table.md share one least-squares fit.
timing_report = _load("timing_report")

MODEL_LOAD = "model_load"

# Qwen3-4B shapes (docs/research/d2l_t2l_reading.md, section 3) and the reference's LoRA.
HIDDEN = 2560
LAYERS = 36
INTERMEDIATE = 9728
Q_OUT = 32 * 128
KV_OUT = 8 * 128
HEAD_DIM = 128
LORA_RANK = 256
# Rows of nn.Linear heads that num_parameters(exclude_embeddings=True) counts: the 16-row lm_head
# and the trainable copy Unsloth makes of it (PEFT modules_to_save). The run's flos fix the
# product with HIDDEN, not the split (one tied 32-row head would count the same).
HEAD_ROWS = 2 * 16


def reference_params() -> int:
    r"""Return :math:`P`, the parameter count in the reference's ``total_flos``.

    Why: HF ``Trainer`` adds ``model.floating_point_ops(inputs)`` per step, which is
    :math:`6 \cdot \text{input\_ids.numel()} \cdot` ``num_parameters(exclude_embeddings=True)``
    (transformers 4.55.4, ``modeling_utils.py``), on the model the PEFT wrapper forwards to: the
    Qwen3 model with the LoRA layers injected. That count excludes the weights of ``nn.Embedding``
    modules only. Per decoder layer: q :math:`2560 \cdot 4096`, k and v :math:`2560 \cdot 1024`,
    o :math:`4096 \cdot 2560`, gate and up :math:`2560 \cdot 9728`, down :math:`9728 \cdot 2560`,
    two RMSNorms of 2560 and q_norm, k_norm of 128: 100,930,816; 36 layers and the final norm:
    3,633,511,936. LoRA :math:`r = 256` on the seven projections adds
    :math:`36 \cdot 256 \cdot (6656 + 3584 + 3584 + 6656 + 3 \cdot 12288) = 528{,}482{,}304`.
    Unsloth trains ``embed_tokens`` and ``lm_head`` as full copies, not LoRA, and the heads add
    :math:`32 \cdot 2560 = 81{,}920`. Total 4,162,076,160. The run's 120 ``total_flos`` values
    are integer multiples of :math:`6P` for this :math:`P` and for none of its neighbors (no head,
    one 16-row head, or a 151,936-row vocabulary), which fixes the head term.

    Returns:
        :math:`P`.
    """
    attention = HIDDEN * Q_OUT + 2 * HIDDEN * KV_OUT + Q_OUT * HIDDEN
    mlp = 3 * HIDDEN * INTERMEDIATE
    norms = 2 * HIDDEN + 2 * HEAD_DIM
    base = LAYERS * (attention + mlp + norms) + HIDDEN
    lora = (
        LAYERS
        * LORA_RANK
        * (
            (HIDDEN + Q_OUT)
            + 2 * (HIDDEN + KV_OUT)
            + (Q_OUT + HIDDEN)
            + 3 * (HIDDEN + INTERMEDIATE)
        )
    )
    return base + lora + HEAD_ROWS * HIDDEN


REFERENCE_PARAMS = reference_params()


def flos_to_tokens(total_flos: float, params: int = REFERENCE_PARAMS) -> tuple[int, float]:
    r"""Convert the trainer's ``total_flos`` into input tokens, :math:`T = \text{flos} / (6P)`.

    Args:
        total_flos: ``total_flos`` from the printed ``TrainOutput``.
        params: :math:`P`, the parameter count the trainer used.

    Returns:
        ``(T, residual)``: :math:`T` rounded to an integer, and the distance of the exact quotient
        from it. A residual well above 0 means :math:`P` is not the count the trainer used.

    Raises:
        ValueError: If ``total_flos`` is negative or not finite, or ``params`` is not positive.
    """
    if not math.isfinite(total_flos) or total_flos < 0:
        raise ValueError(f"total_flos must be finite and non-negative; got {total_flos!r}")
    if params <= 0:
        raise ValueError(f"params must be positive; got {params}")
    quotient = total_flos / (6 * params)
    tokens = round(quotient)
    return tokens, abs(quotient - tokens)


# Session log


@dataclass(frozen=True)
class LogLine:
    """One line of output.

    Attributes:
        time: Seconds since the kernel started.
        stream: ``stdout`` or ``stderr``.
        text: The line without its newline, after the last carriage return.
    """

    time: float
    stream: str
    text: str


def _log_entries(text: str) -> tuple[list[object], bool]:
    """Decode the log's JSON array; a truncated array yields its complete leading entries."""
    try:
        data = json.loads(text)
    except ValueError:
        pass
    else:
        return (data, False) if isinstance(data, list) else ([data], False)
    decoder = json.JSONDecoder()
    entries: list[object] = []
    position = text.find("[") + 1
    if position == 0:
        return entries, True
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text) or text[position] == "]":
            return entries, position >= len(text)
        try:
            entry, position = decoder.raw_decode(text, position)
        except ValueError:
            return entries, True
        entries.append(entry)


def parse_session_log(text: str) -> tuple[list[LogLine], int, bool]:
    """Split a Kaggle session log into time-ordered lines.

    Args:
        text: The log file's content.

    Returns:
        The lines in time order, the number of skipped entries (not an object with a finite
        numeric ``time`` and a string ``data``), and whether the array was truncated.
    """
    entries, truncated = _log_entries(text)
    lines: list[LogLine] = []
    skipped = 0
    for entry in entries:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        stamp, data = entry.get("time"), entry.get("data")
        stream = entry.get("stream_name")
        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, int | float)
            or not math.isfinite(stamp)
            or not isinstance(data, str)
        ):
            skipped += 1
            continue
        parts = data.split("\n")
        if parts and parts[-1] == "":
            parts.pop()
        for part in parts:
            visible = part.rstrip("\r").rsplit("\r", 1)[-1]
            lines.append(LogLine(float(stamp), stream if isinstance(stream, str) else "", visible))
    lines.sort(key=lambda line: line.time)
    return lines, skipped, truncated


def read_session_log(path: str | Path) -> tuple[list[LogLine], int, bool]:
    """Read a session log file; see :func:`parse_session_log`.

    Args:
        path: The pulled log, for example ``runs/<task-id>/<kernel>.log``.

    Returns:
        As :func:`parse_session_log`.

    Raises:
        OSError: If the file cannot be read.
    """
    return parse_session_log(Path(path).read_text(encoding="utf-8", errors="replace"))


# Worker lines

RANK_LINE = re.compile(r"^\[Rank (\d+)\] (.*)$")
ALLOCATED = re.compile(r"^allocated (\d+)MB for (training|inference)$")
TRAIN_STATS = re.compile(
    r"^training stats for puzzle (\S+): TrainOutput\(global_step=(\d+), "
    r"training_loss=([^,]+), metrics=(\{.*\})\)$"
)
METRIC = re.compile(r"'(train_runtime|total_flos)': ([-+0-9.eE]+|nan|inf)")
DECODING = re.compile(r"^decoding \[(.*)\]$")
SUBKEY = re.compile(r"'([^']+)'")
SCORING = re.compile(r"^scoring (\S+) #(\d+)$")
TIMEOUT = re.compile(r"^timeout after (\d+(?:\.\d+)?)s for puzzle (\S+)$")
FINISHED = re.compile(r"^finished (\S+) in (\d+(?:\.\d+)?)s$")
START, DONE, STOP = "start!", "done!", "stop!"

START_FROM_PREVIOUS = "previous finished"
START_FROM_PRINTED = "printed seconds"


def task_of(subkey: str) -> str:
    """Return the task id of an augmented subkey such as ``136b0064_0.rot90.permute...ex012``.

    Args:
        subkey: A subkey from a ``decoding`` or ``scoring`` line, or a candidate file name.

    Returns:
        The part before the first ``_`` of the part before the first ``.``.
    """
    return subkey.split(".", 1)[0].split("_", 1)[0]


@dataclass
class PuzzleLog:
    """What one rank printed for one puzzle.

    Attributes:
        rank: Worker rank.
        index: Position of the puzzle in the rank's sequence, from 0.
        puzzle_id: Task id, from the first line that names it; None until then.
        start: Span start (see the module docstring); None until known.
        start_source: :data:`START_FROM_PREVIOUS` or :data:`START_FROM_PRINTED`.
        end: Time of the ``finished`` line; None for an incomplete puzzle.
        printed_seconds: Seconds printed in the ``finished`` line.
        global_step: Training steps.
        training_loss: Mean training loss.
        train_runtime: Trainer wall seconds.
        total_flos: Trainer floating-point operations.
        train_mb: Peak MB allocated during training.
        inference_mb: Peak MB allocated during inference.
        first_line: Time of the puzzle's first rank line (it follows training).
        intervals: ``(op, seconds)`` per decode or selection interval, in order.
        decode_batches: ``decoding`` lines.
        scored: ``scoring`` lines (new candidates rescored).
        timeout_after: Seconds printed by a ``timeout`` line; None without one.
        id_conflicts: Lines naming another task than the puzzle's.
    """

    rank: int
    index: int
    puzzle_id: str | None = None
    start: float | None = None
    start_source: str = START_FROM_PREVIOUS
    end: float | None = None
    printed_seconds: float | None = None
    global_step: int | None = None
    training_loss: float | None = None
    train_runtime: float | None = None
    total_flos: float | None = None
    train_mb: int | None = None
    inference_mb: int | None = None
    first_line: float | None = None
    intervals: list[tuple[str, float]] = field(default_factory=list)
    decode_batches: int = 0
    scored: int = 0
    timeout_after: float | None = None
    id_conflicts: int = 0

    @property
    def complete(self) -> bool:
        """Whether the puzzle has a ``finished`` line and a known start."""
        return self.end is not None and self.start is not None

    @property
    def span(self) -> float | None:
        """Puzzle wall seconds from its start to its ``finished`` line."""
        if self.end is None or self.start is None:
            return None
        return self.end - self.start

    def name(self, puzzle_id: str) -> None:
        """Record a task id named by a line; a different id than the first counts as a conflict.

        Args:
            puzzle_id: Task id the line names.
        """
        if self.puzzle_id is None:
            self.puzzle_id = puzzle_id
        elif puzzle_id != self.puzzle_id:
            self.id_conflicts += 1


@dataclass
class RankLog:
    """One worker's lines, parsed.

    Attributes:
        rank: Worker rank.
        start_line: Time of ``start!``.
        end_line: Time of ``done!`` or ``stop!``.
        stopped: The worker printed ``stop!`` (its end time passed).
        puzzles: Puzzles in the order the rank took them.
        unknown_lines: Rank lines that match no known message.
    """

    rank: int
    start_line: float | None = None
    end_line: float | None = None
    stopped: bool = False
    puzzles: list[PuzzleLog] = field(default_factory=list)
    unknown_lines: int = 0

    @property
    def load_seconds(self) -> float | None:
        """``start!`` to the first puzzle's start: model loading, outside every puzzle."""
        if self.start_line is None or not self.puzzles or self.puzzles[0].start is None:
            return None
        return self.puzzles[0].start - self.start_line


def _metrics(text: str) -> dict[str, float]:
    """Read finite ``train_runtime`` and ``total_flos`` values from a printed metrics dict."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        value = None
    if isinstance(value, dict):
        pairs = [(k, value[k]) for k in ("train_runtime", "total_flos") if k in value]
    else:  # truncated, or a value literal_eval refuses, such as nan
        pairs = METRIC.findall(text)
    found = {}
    for name, number in pairs:
        if isinstance(number, bool):
            continue
        try:
            number = float(number)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            found[name] = number
    return found


def _subkeys(text: str) -> list[str]:
    try:
        value = ast.literal_eval(f"[{text}]")
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return SUBKEY.findall(text)
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def parse_ranks(lines: Iterable[LogLine]) -> dict[int, RankLog]:
    """Assign worker lines to ranks and puzzles, timing the intervals between them.

    Args:
        lines: Session log lines in time order.

    Returns:
        One :class:`RankLog` per rank seen, by rank.
    """
    ranks: dict[int, RankLog] = {}
    current: dict[int, PuzzleLog | None] = {}
    last_finished: dict[int, float] = {}
    open_interval: dict[int, tuple[str, float] | None] = {}
    for line in lines:
        match = RANK_LINE.match(line.text)
        if match is None:
            continue
        rank, message, now = int(match.group(1)), match.group(2).strip(), line.time
        log = ranks.setdefault(rank, RankLog(rank))
        puzzle = current.get(rank)
        opened = open_interval.get(rank)
        if opened is not None and puzzle is not None:
            puzzle.intervals.append((opened[0], now - opened[1]))
        open_interval[rank] = None
        if message in (START, DONE, STOP):
            if message == START:
                log.start_line = now
            else:
                log.end_line, log.stopped = now, log.stopped or message == STOP
                current[rank] = None
            continue
        if puzzle is None:
            puzzle = PuzzleLog(
                rank, len(log.puzzles), start=last_finished.get(rank), first_line=now
            )
            log.puzzles.append(puzzle)
            current[rank] = puzzle
        if allocated := ALLOCATED.match(message):
            if allocated.group(2) == "training":
                puzzle.train_mb = int(allocated.group(1))
            else:
                puzzle.inference_mb = int(allocated.group(1))
        elif stats := TRAIN_STATS.match(message):
            puzzle.name(stats.group(1))
            puzzle.global_step = int(stats.group(2))
            try:
                puzzle.training_loss = float(stats.group(3))
            except ValueError:
                puzzle.training_loss = None
            found = _metrics(stats.group(4))
            puzzle.train_runtime = found.get("train_runtime")
            puzzle.total_flos = found.get("total_flos")
        elif decoding := DECODING.match(message):
            for subkey in _subkeys(decoding.group(1)):
                puzzle.name(task_of(subkey))
            puzzle.decode_batches += 1
            open_interval[rank] = (DECODE, now)
        elif scoring := SCORING.match(message):
            puzzle.name(task_of(scoring.group(1)))
            puzzle.scored += 1
            open_interval[rank] = (SELECTION, now)
        elif timeout := TIMEOUT.match(message):
            puzzle.name(timeout.group(2))
            puzzle.timeout_after = float(timeout.group(1))
        elif finished := FINISHED.match(message):
            puzzle.name(finished.group(1))
            puzzle.end, puzzle.printed_seconds = now, float(finished.group(2))
            if puzzle.start is None:
                puzzle.start = now - puzzle.printed_seconds
                puzzle.start_source = START_FROM_PRINTED
            last_finished[rank] = now
            current[rank] = None
        else:
            log.unknown_lines += 1
    return dict(sorted(ranks.items()))


# Records and CSV

CSV_FIELDS = (
    "puzzle_id",
    "rank",
    "index",
    "complete",
    "start_s",
    "end_s",
    "span_s",
    "printed_s",
    "start_source",
    "first_line_s",
    "global_step",
    "training_loss",
    "train_runtime_s",
    "total_flos",
    "train_tokens",
    "train_mb",
    "inference_mb",
    "decode_batches",
    "scored_candidates",
    "timed_out",
    "timeout_after_s",
    "id_conflicts",
)


def puzzle_records(
    puzzle: PuzzleLog, params: int = REFERENCE_PARAMS
) -> list[tuple[str, float, int | None]]:
    """Return a puzzle's timing records as ``(op, seconds, tokens)``.

    Args:
        puzzle: A parsed puzzle with a task id.
        params: :math:`P` for the ``total_flos`` conversion.

    Returns:
        Train (when its runtime was printed), the decode and selection intervals in order, and
        the puzzle span when the puzzle is complete.
    """
    records: list[tuple[str, float, int | None]] = []
    if puzzle.train_runtime is not None:
        tokens = None
        if puzzle.total_flos is not None:
            tokens = flos_to_tokens(puzzle.total_flos, params)[0]
        records.append((TRAIN, puzzle.train_runtime, tokens))
    records += [(op, seconds, None) for op, seconds in puzzle.intervals]
    if puzzle.span is not None:
        records.append((PUZZLE, puzzle.span, None))
    return records


def _num(value: float | None, digits: int | None = None) -> str:
    """A number as CSV text: empty for None, else ``repr``, rounded first when ``digits`` is set."""
    if value is None:
        return ""
    return repr(value if digits is None else round(value, digits))


def csv_row(puzzle: PuzzleLog, params: int = REFERENCE_PARAMS) -> dict[str, str]:
    """Return a puzzle's row of ``puzzles.csv``.

    Args:
        puzzle: A parsed puzzle.
        params: :math:`P` for the ``total_flos`` conversion.

    Returns:
        The row by :data:`CSV_FIELDS`: times in seconds since the kernel started, rounded to
        microseconds; printed values as printed; flags as 1 or 0; unknown values empty.
    """
    tokens = None
    if puzzle.total_flos is not None:
        tokens = flos_to_tokens(puzzle.total_flos, params)[0]
    first = None
    if puzzle.first_line is not None and puzzle.start is not None:
        first = puzzle.first_line - puzzle.start
    optional_ints = {
        "global_step": puzzle.global_step,
        "train_tokens": tokens,
        "train_mb": puzzle.train_mb,
        "inference_mb": puzzle.inference_mb,
    }
    row = {
        "puzzle_id": puzzle.puzzle_id or "",
        "rank": str(puzzle.rank),
        "index": str(puzzle.index),
        "complete": str(int(puzzle.complete)),
        "start_s": _num(puzzle.start, 6),
        "end_s": _num(puzzle.end, 6),
        "span_s": _num(puzzle.span, 6),
        "printed_s": _num(puzzle.printed_seconds),
        "start_source": puzzle.start_source,
        "first_line_s": _num(first, 6),
        "training_loss": _num(puzzle.training_loss),
        "train_runtime_s": _num(puzzle.train_runtime),
        "total_flos": _num(puzzle.total_flos),
        "decode_batches": str(puzzle.decode_batches),
        "scored_candidates": str(puzzle.scored),
        "timed_out": str(int(puzzle.timeout_after is not None)),
        "timeout_after_s": _num(puzzle.timeout_after),
        "id_conflicts": str(puzzle.id_conflicts),
    }
    row.update({name: "" if value is None else str(value) for name, value in optional_ints.items()})
    return {name: row[name] for name in CSV_FIELDS}


@dataclass(frozen=True)
class RecordsResult:
    """What :func:`write_records` wrote.

    Attributes:
        files: Timing files written, one per rank.
        csv_path: The per-puzzle CSV.
        records: Timing records written.
        puzzles: Puzzles with a ``puzzle`` record.
        incomplete: Named puzzles without one (no ``finished`` line).
        unnamed: Puzzles whose task id no line named; they get no records.
        skipped_entries: Log entries skipped by :func:`parse_session_log`.
        truncated: The log's JSON array was truncated.
        unknown_lines: Rank lines that match no known message.
    """

    files: tuple[Path, ...]
    csv_path: Path
    records: int
    puzzles: int
    incomplete: int
    unnamed: int
    skipped_entries: int
    truncated: bool
    unknown_lines: int


def write_records(
    log_path: str | Path, out_dir: str | Path, params: int = REFERENCE_PARAMS
) -> RecordsResult:
    """Parse a session log and write the timing files and ``puzzles.csv``.

    Existing ``timing-rank*.jsonl`` files in ``out_dir`` are deleted first, since the logger
    appends and a stale rank's file would otherwise mix into the run.

    Args:
        log_path: The session log.
        out_dir: Output directory, created if missing.
        params: :math:`P` for the ``total_flos`` conversion.

    Returns:
        Counts of what was written.

    Raises:
        OSError: If the log cannot be read or an output cannot be written.
    """
    lines, skipped, truncated = read_session_log(log_path)
    ranks = parse_ranks(lines)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(f"{FILE_PREFIX}rank*{FILE_SUFFIX}"):
        stale.unlink()
    files, written, complete, incomplete, unnamed = [], 0, 0, 0, 0
    rows = []
    for rank, log in ranks.items():
        logger = TimingLogger(out, worker=f"rank{rank}")
        if log.load_seconds is not None and log.load_seconds >= 0:
            logger.record(None, MODEL_LOAD, log.load_seconds)
            written += 1
        for puzzle in log.puzzles:
            if puzzle.puzzle_id is None:
                unnamed += 1
                continue
            for op, seconds, tokens in puzzle_records(puzzle, params):
                logger.record(puzzle.puzzle_id, op, max(seconds, 0.0), tokens)
                written += 1
            complete += puzzle.complete
            incomplete += not puzzle.complete
            rows.append(csv_row(puzzle, params))
        if logger.path.exists():
            files.append(logger.path)
    csv_path = out / "puzzles.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return RecordsResult(
        tuple(files),
        csv_path,
        written,
        complete,
        incomplete,
        unnamed,
        skipped,
        truncated,
        sum(log.unknown_lines for log in ranks.values()),
    )


# Summary

SPAN_TOLERANCE_S = 2.0
# The reference's caps (arc26/nvarc/arc_solver.py): a puzzle decodes no further batch once
# 1200 s have passed, and one DFS call stops after 540 s.
PUZZLE_CAP_S = 1200.0
DFS_CAP_S = 540.0
# Four batches of four augmented views per test input (worker, the batch construction).
BATCHES_PER_TEST_INPUT = 4
# Production soft deadline after the notebook start (configs/submission_P1.1-01.yaml): workers
# take no new puzzle after it. G1's rule and the B3 projection measure against it.
WINDOW_S = 41400.0
B3_BUDGET_S = 720.0
B3_PLANNING_S = 600.0
B3_STEP_TOKENS = 2000
G1_LIMIT = 0.9
# Hidden-set sizes for the B3 projection: DEV120's, and the 240 tasks of the 2025 hidden set
# (docs/research/kaggle_first_runs.md; unverified for 2026).
HIDDEN_TASKS = (120, 240)
# Notebook lines (scripts/build_kaggle_notebook.py, cell 7 in development mode).
SOLVER_LAUNCH = "solver: end time at "
SOLVER_EXIT = "solver exit code: "
ALL_CORRECT = re.compile(r"^ALL_CORRECT:\s*\S+\s+-\s+\S+\s+\d+x\d+\s+\[(\S+)\]$")
NOTEBOOK_ACC = re.compile(r"^ acc:\s*([0-9.]+)/\s*(\d+) \('(\w+)'\)$")
NOTEBOOK_RELOAD = re.compile(r"^\*\*\* Reload score: ([0-9.eE+-]+)$")
NOTEBOOK_SUBKEYS = re.compile(r"^ subkeys: (\d+)/(\d+)$")
CANDIDATE_KEY = re.compile(r"^(.+)\.out(\d+)$")
PLACEHOLDER = [[0]]
# Training text (arc26/nvarc/arc_loader.py, QwenFormatter.fmt_train): per demonstration pair,
# <|im_start|>, the role, and a newline before each of the two grids and <|im_end|> after it.
MARKERS_PER_PAIR = 8
# The worker's max_seq_length: cut_to_len drops leading pairs while a sequence is longer.
TRAIN_MAX_TOKENS = 8192


@dataclass(frozen=True)
class Stats:
    r"""Distribution of a sample.

    Attributes:
        n: Sample size.
        mean: Mean.
        median: Median (the mean of the two middle values for even ``n``).
        p90: Nearest-rank 90th percentile, the :math:`\lceil 0.9 n \rceil`-th smallest value.
        min: Minimum.
        max: Maximum.
        total: Sum.
    """

    n: int
    mean: float
    median: float
    p90: float
    min: float
    max: float
    total: float


def describe(values: Iterable[float]) -> Stats:
    """Return the distribution of a non-empty sample.

    Args:
        values: The sample.

    Returns:
        Its :class:`Stats`.

    Raises:
        ValueError: If the sample is empty.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    n = len(ordered)
    total = math.fsum(ordered)
    p90 = ordered[math.ceil(0.9 * n) - 1]
    return Stats(n, total / n, statistics.median(ordered), p90, ordered[0], ordered[-1], total)


@dataclass(frozen=True)
class Fit:
    """One row of the cost model.

    Attributes:
        op: What is timed, with its unit (per puzzle, per call).
        axis: The size axis and whether it is exact or a proxy.
        subset: Which puzzles the fit uses.
        n: Samples.
        a: Intercept in seconds.
        b: Slope in seconds per axis unit; None when the axis does not vary.
        r2: Coefficient of determination; None when undefined.
    """

    op: str
    axis: str
    subset: str
    n: int
    a: float
    b: float | None
    r2: float | None


def fit(op: str, axis: str, subset: str, points: Sequence[tuple[float, float]]) -> Fit:
    """Fit seconds against a size axis with ``timing_report.fit_linear``.

    Args:
        op: What is timed.
        axis: The size axis.
        subset: Which puzzles the points come from.
        points: ``(size, seconds)`` pairs.

    Returns:
        The fit.
    """
    a, b, r2 = timing_report.fit_linear([x for x, _ in points], [y for _, y in points])
    return Fit(op, axis, subset, len(points), a, b, r2)


def fit_two(
    x1: Sequence[float], x2: Sequence[float], ys: Sequence[float]
) -> tuple[float, float, float, float | None]:
    r"""Fit :math:`y = a + b x_1 + c x_2` by ordinary least squares.

    Why: pure Python, like ``timing_report.fit_linear``. With centered sums
    :math:`S_{jk} = \sum (x_j - \bar x_j)(x_k - \bar x_k)` and
    :math:`S_{jy} = \sum (x_j - \bar x_j)(y - \bar y)`, the normal equations give
    :math:`b = (S_{1y} S_{22} - S_{2y} S_{12}) / D`, :math:`c = (S_{2y} S_{11} - S_{1y} S_{12}) / D`
    with :math:`D = S_{11} S_{22} - S_{12}^2`, and :math:`a = \bar y - b \bar x_1 - c \bar x_2`.

    Args:
        x1: First regressor.
        x2: Second regressor, as many values as ``x1``.
        ys: Responses, as many values as ``x1``.

    Returns:
        ``(a, b, c, r2)``; ``r2`` is None when the ``ys`` do not vary.

    Raises:
        ValueError: If the lengths differ, there are fewer than three samples, or the
            regressors are collinear.
    """
    n = len(ys)
    if n < 3 or len(x1) != n or len(x2) != n:
        raise ValueError(f"need three or more equal-length samples; got {len(x1)}, {len(x2)}, {n}")
    m1, m2, my = math.fsum(x1) / n, math.fsum(x2) / n, math.fsum(ys) / n
    d1 = [x - m1 for x in x1]
    d2 = [x - m2 for x in x2]
    dy = [y - my for y in ys]
    s11 = math.fsum(u * u for u in d1)
    s22 = math.fsum(v * v for v in d2)
    s12 = math.fsum(u * v for u, v in zip(d1, d2, strict=True))
    s1y = math.fsum(u * w for u, w in zip(d1, dy, strict=True))
    s2y = math.fsum(v * w for v, w in zip(d2, dy, strict=True))
    det = s11 * s22 - s12 * s12
    if det <= 0 or det <= 1e-12 * s11 * s22:
        raise ValueError("the regressors are collinear")
    b = (s1y * s22 - s2y * s12) / det
    c = (s2y * s11 - s1y * s12) / det
    a = my - b * m1 - c * m2
    ss_res = math.fsum((y - a - b * u - c * v) ** 2 for u, v, y in zip(x1, x2, ys, strict=True))
    ss_tot = math.fsum(w * w for w in dy)
    return a, b, c, (1.0 - ss_res / ss_tot if ss_tot > 0 else None)


def square_sequence_tokens(side: int, pairs: int) -> tuple[int, int]:
    r"""Tokens of one training sequence of a puzzle whose grids are all ``side`` by ``side``.

    Why: an exact check of :math:`T`. ``fmt_train`` writes a grid as its digits with a newline
    between rows, :math:`s^2 + s - 1` tokens, and each pair adds :data:`MARKERS_PER_PAIR`
    markers. Dihedral views keep a square grid's shape and color permutations keep lengths, so
    all of the puzzle's sequences have this length.

    Args:
        side: Grid side :math:`s`.
        pairs: Demonstration pairs :math:`n`.

    Returns:
        ``(tokens, kept)``: :math:`n' (2 (s^2 + s - 1) + 8)` and :math:`n'`, the pairs left after
        ``cut_to_len`` drops leading pairs beyond :data:`TRAIN_MAX_TOKENS`.
    """
    per_pair = 2 * (side * side + side - 1) + MARKERS_PER_PAIR
    kept = pairs
    while kept > 1 and kept * per_pair > TRAIN_MAX_TOKENS:
        kept -= 1
    return kept * per_pair, kept


@dataclass(frozen=True)
class OutputRow:
    """Candidates of one test output and how selection treated them.

    Attributes:
        task_id: Task id.
        index: Test output index.
        candidates: Decoded candidates (every beam of every augmented view).
        distinct: Distinct candidate grids.
        correct_candidates: Candidates the notebook marked ``ALL_CORRECT``.
        correct_rank: Rank of the correct grid under ``score_kgmon`` (1 is ``attempt_1``); None
            without a correct candidate.
        submitted_correct: An attempt in ``submission.json`` equals the correct grid
            (``arc26.evaluate``).
        top2_matches: ``score_kgmon``'s first two grids, padded with ``[[0]]``, are the submitted
            attempts; None without candidates.
        probmul_correct: The correct grid is among ``score_full_probmul_3``'s first two.
    """

    task_id: str
    index: int
    candidates: int
    distinct: int
    correct_candidates: int
    correct_rank: int | None
    submitted_correct: bool
    top2_matches: bool | None
    probmul_correct: bool


OUTPUT_FIELDS = (
    "task_id",
    "output",
    "candidates",
    "distinct_grids",
    "correct_candidates",
    "correct_exists",
    "correct_rank_kgmon",
    "submitted_correct",
    "kgmon_top2_is_submission",
)


def correct_candidate_keys(lines: Iterable[LogLine]) -> list[str]:
    """Return the candidate keys of the notebook's ``ALL_CORRECT`` lines.

    Args:
        lines: Session log lines.

    Returns:
        Keys such as ``<augmented subkey>.out<i>``: candidate ``i`` of that file in
        ``candidates.json``, in log order.
    """
    return [m.group(1) for line in lines if (m := ALL_CORRECT.match(line.text))]


def candidate_outputs(
    candidates: Mapping[str, object],
    submission: Mapping[str, object],
    correct_keys: Sequence[str],
) -> tuple[list[OutputRow], list[str]]:
    """Rank every test output's candidates and check what was submitted.

    Why: ``ALL_CORRECT`` lines say which candidates equal the solution (the notebook checked them
    in development mode), so the correct grid of an output is a candidate's grid and no solution
    file is read. The grids are used in memory only; nothing here writes a grid.

    Args:
        candidates: ``candidates.json``: augmented subkey to its list of
            ``{"beam_score", "score_aug", "solution"}``.
        submission: ``submission.json``.
        correct_keys: Keys from :func:`correct_candidate_keys`.

    Returns:
        One row per test output of ``submission`` in task and output order, and problems found
        (keys missing from the candidates, outputs whose correct candidates differ).
    """
    # Imported here: the moved scorer needs numpy, which only this command uses.
    from arc26.nvarc.arc_decoder import score_full_probmul_3, score_kgmon

    decoded: dict[str, dict[str, Mapping[str, object]]] = {}
    for key in sorted(candidates):
        samples = candidates[key]
        if not isinstance(samples, list):
            continue
        base = key.split(".", 1)[0]
        for i, sample in enumerate(samples):
            if isinstance(sample, Mapping):
                decoded.setdefault(base, {})[f"{key}.out{i}"] = sample
    problems: list[str] = []
    correct: dict[str, object] = {}
    marked: dict[str, int] = defaultdict(int)
    for key in correct_keys:
        base = key.split(".", 1)[0]
        sample = decoded.get(base, {}).get(key)
        if sample is None:
            problems.append(f"ALL_CORRECT candidate {key} is not in the candidates")
            continue
        grid = sample.get("solution")
        if base in correct and correct[base] != grid:
            problems.append(f"{base}: ALL_CORRECT candidates hold different grids")
        correct.setdefault(base, grid)
        marked[base] += 1
    rows = []
    for task_id in sorted(submission):
        outputs = submission[task_id]
        for index in range(len(outputs) if isinstance(outputs, list) else 0):
            base = f"{task_id}_{index}"
            guesses = decoded.get(base, {})
            grid = correct.get(base)
            ordered = [list(g) for g in score_kgmon(guesses)] if guesses else []
            top2 = None
            if guesses:
                padded = ordered[:2] + [PLACEHOLDER] * (2 - len(ordered[:2]))
                attempts = outputs[index] if isinstance(outputs[index], Mapping) else {}
                top2 = padded == [attempts.get("attempt_1"), attempts.get("attempt_2")]
            rank = None
            probmul = False
            submitted = False
            if grid is not None:
                rank = next((i + 1 for i, g in enumerate(ordered) if g == grid), None)
                if rank is None:
                    problems.append(f"{base}: the correct grid is not among the ranked grids")
                probmul = grid in [list(g) for g in score_full_probmul_3(guesses)[:2]]
                submitted = score_task(task_id, {task_id: [outputs[index]]}, [grid]).correct[0]
            rows.append(
                OutputRow(
                    task_id,
                    index,
                    len(guesses),
                    len(ordered),
                    marked.get(base, 0),
                    rank,
                    submitted,
                    top2,
                    probmul,
                )
            )
    return rows, problems


def subset_score(
    rows: Sequence[OutputRow], solved: Callable[[OutputRow], bool], label: str = "DEV120"
) -> SubsetScore:
    """Score test outputs with ``arc26.evaluate``'s per-task metric.

    Args:
        rows: Output rows, in output order within each task.
        solved: Whether an output counts as solved.
        label: Subset name.

    Returns:
        The per-task score with fractional credit.
    """
    flags: dict[str, list[bool]] = {}
    for row in rows:
        flags.setdefault(row.task_id, []).append(solved(row))
    return SubsetScore(label, tuple(TaskScore(t, tuple(f)) for t, f in sorted(flags.items())))


def notebook_scores(lines: Iterable[LogLine]) -> dict[str, str]:
    """Return the scores the notebook printed in development mode, as printed.

    Args:
        lines: Session log lines.

    Returns:
        ``reload`` (the per-task credit of ``submission.json``), ``subkeys`` (correct and all
        candidates), and one entry per selection algorithm (``<credit>/<tasks with candidates>``).
    """
    found: dict[str, str] = {}
    for line in lines:
        if m := NOTEBOOK_RELOAD.match(line.text):
            found["reload"] = m.group(1)
        elif m := NOTEBOOK_SUBKEYS.match(line.text):
            found["subkeys"] = f"{m.group(1)}/{m.group(2)}"
        elif m := NOTEBOOK_ACC.match(line.text):
            found[m.group(3)] = f"{m.group(1)}/{m.group(2)}"
    return found


def solver_window(lines: Sequence[LogLine]) -> tuple[float, float]:
    """Return the solver's launch and exit times.

    Args:
        lines: Session log lines.

    Returns:
        The times of the notebook's ``solver: end time at`` and ``solver exit code:`` lines; the
        first and last rank line when either is missing.

    Raises:
        ValueError: If the log has no rank line and no solver lines.
    """
    launch = next((line.time for line in lines if line.text.startswith(SOLVER_LAUNCH)), None)
    end = next((line.time for line in lines if line.text.startswith(SOLVER_EXIT)), None)
    ranked = [line.time for line in lines if RANK_LINE.match(line.text)]
    if launch is None or end is None:
        if not ranked:
            raise ValueError("no solver lines and no rank lines in the log")
        launch = ranked[0] if launch is None else launch
        end = ranked[-1] if end is None else end
    return launch, end


def sha256_prefix(path: Path) -> str:
    """Return the first 12 hex digits of a file's sha256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@dataclass
class RunData:
    """Everything the summary reads.

    Attributes:
        lines: Session log lines.
        skipped_entries: Log entries skipped while parsing.
        truncated: The log's array was truncated.
        ranks: Worker lines by rank (for ``start!``, ``done!``).
        records: Timing records.
        skipped_records: Invalid lines in the timing files.
        timing_paths: Timing files read.
        puzzles: Rows of ``puzzles.csv`` for complete puzzles, by puzzle id.
        incomplete: Rows of ``puzzles.csv`` without a ``finished`` line, by puzzle id.
        tasks: K0's per-task statistics by task id.
        seconds: Seconds per op per puzzle, summed over its records.
        calls: Records per op per puzzle.
        launch: Solver launch time.
        exit: Solver exit time.
        deadline: The production soft deadline, seconds after the notebook start.
        offset: Seconds from the notebook start to the solver's launch (0 when unknown).
    """

    lines: list[LogLine]
    skipped_entries: int
    truncated: bool
    ranks: dict[int, RankLog]
    records: list[TimingRecord]
    skipped_records: int
    timing_paths: list[Path]
    puzzles: dict[str, dict[str, str]]
    incomplete: dict[str, dict[str, str]]
    tasks: dict[str, dict[str, int]]
    seconds: dict[str, dict[str, float]]
    calls: dict[str, dict[str, int]]
    launch: float
    exit: float
    deadline: float
    offset: float

    @property
    def window(self) -> float:
        """Seconds from the solver's launch to the soft deadline: the denominator of :math:`u`."""
        return self.deadline - self.offset

    def op(self, puzzle_id: str, op: str) -> float:
        """Seconds of an op in a puzzle (0 without records)."""
        return self.seconds.get(puzzle_id, {}).get(op, 0.0)

    def overhead(self, puzzle_id: str) -> float:
        """Puzzle span minus train, decode, and selection."""
        phases = sum(self.op(puzzle_id, name) for name in (TRAIN, DECODE, SELECTION))
        return self.op(puzzle_id, PUZZLE) - phases

    def size(self, puzzle_id: str, key: str) -> int:
        """K0's statistic ``key`` (``cells``, ``train_pairs``, ``test_inputs``, ``max_side``)."""
        return int(self.tasks[puzzle_id][key])

    def tokens(self, puzzle_id: str) -> int:
        """Training input tokens :math:`T` of a puzzle."""
        return int(self.puzzles[puzzle_id]["train_tokens"])

    def timed_out(self, puzzle_id: str) -> bool:
        """Whether the puzzle hit the 1200 s decode cap."""
        return self.puzzles[puzzle_id]["timed_out"] == "1"


def load_run(
    timing_dir: Path,
    log_path: Path,
    task_stats: Path,
    deadline: float = WINDOW_S,
    offset: float = 0.0,
) -> RunData:
    """Read the timing files, ``puzzles.csv``, the session log, and K0's per-task statistics.

    Args:
        timing_dir: Output directory of ``records``.
        log_path: The session log.
        task_stats: K0's ``env_probe.json`` (``evaluation_stats.per_task``).
        deadline: The production soft deadline, seconds after the notebook start.
        offset: Seconds from the notebook start to the solver's launch.

    Returns:
        The run's data.

    Raises:
        OSError: If a file cannot be read.
        ValueError: If a file is malformed or holds no complete puzzle.
    """
    paths = timing_report.timing_files([timing_dir])
    records, skipped = read_records(paths)
    with open(timing_dir / "puzzles.csv", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["puzzle_id"]]
    puzzles = {row["puzzle_id"]: row for row in rows if row["complete"] == "1"}
    incomplete = {row["puzzle_id"]: row for row in rows if row["complete"] != "1"}
    probe = json.loads(task_stats.read_text(encoding="utf-8"))
    tasks = probe["evaluation_stats"]["per_task"]
    seconds: dict[str, dict[str, float]] = {}
    calls: dict[str, dict[str, int]] = {}
    for record in records:
        if record.puzzle_id is None:
            continue
        by_op = seconds.setdefault(record.puzzle_id, {})
        by_op[record.op] = by_op.get(record.op, 0.0) + record.seconds
        counts = calls.setdefault(record.puzzle_id, {})
        counts[record.op] = counts.get(record.op, 0) + 1
    if not seconds or not puzzles:
        raise ValueError(f"no complete puzzle under {timing_dir}")
    lines, skipped_entries, truncated = read_session_log(log_path)
    launch, end = solver_window(lines)
    return RunData(
        lines,
        skipped_entries,
        truncated,
        parse_ranks(lines),
        records,
        skipped,
        paths,
        puzzles,
        incomplete,
        tasks,
        seconds,
        calls,
        launch,
        end,
        deadline,
        offset,
    )


def _s(value: float) -> str:
    return f"{value:,.1f}"


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def _r2(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _slope(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _table(header: Sequence[str], rows: Iterable[Sequence[object]], right: int = 1) -> list[str]:
    """A Markdown table; columns from ``right`` on are right-aligned."""
    align = ["---" if i < right else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _stats_cells(stats: Stats, whole: float | None) -> list[str]:
    cells = [str(stats.n), _s(stats.mean), _s(stats.median), _s(stats.p90), _s(stats.max)]
    cells.append(_s(stats.total))
    if whole is not None:
        cells.append(_pct(stats.total, whole))
    return cells


def section_checks(run: RunData) -> list[str]:
    """Parsing checks: spans, training, phase sums, the DFS cap, the token conversion."""
    ids = sorted(run.puzzles)
    previous = [p for p in ids if run.puzzles[p]["start_source"] == START_FROM_PREVIOUS]
    deviation = [
        float(run.puzzles[p]["span_s"]) - float(run.puzzles[p]["printed_s"]) for p in previous
    ]
    outside = [p for p, d in zip(previous, deviation, strict=True) if abs(d) > SPAN_TOLERANCE_S]
    first = [p for p in ids if run.puzzles[p]["start_source"] == START_FROM_PRINTED]
    trained = [p for p in ids if run.puzzles[p]["train_runtime_s"]]
    lag = [
        float(run.puzzles[p]["first_line_s"]) - float(run.puzzles[p]["train_runtime_s"])
        for p in trained
    ]
    early = sum(value < 0 for value in lag)
    over = [
        p
        for p in ids
        if run.op(p, TRAIN) + run.op(p, DECODE) + run.op(p, SELECTION) > run.op(p, PUZZLE) + 1e-6
    ]
    decode_calls = [r.seconds for r in run.records if r.op == DECODE and r.puzzle_id in run.puzzles]
    longest = max(decode_calls, default=0.0)
    residuals, mismatched = [], 0
    with_flos = [p for p in ids if run.puzzles[p]["total_flos"]]
    for p in with_flos:
        tokens, residual = flos_to_tokens(float(run.puzzles[p]["total_flos"]))
        residuals.append(residual)
        mismatched += tokens != run.tokens(p)
    unknown = sum(log.unknown_lines for log in run.ranks.values())
    conflicts = sum(int(run.puzzles[p]["id_conflicts"]) for p in ids)
    unfinished = ", ".join(sorted(run.incomplete)) or "none"
    lines = [
        "## 1. Parsing checks",
        "",
        f"- Session log: {len(run.lines)} lines, {run.skipped_entries} entries skipped, "
        f"truncated: {'yes' if run.truncated else 'no'}; rank lines matching no known message: "
        f"{unknown}; lines naming another task than their puzzle: {conflicts}.",
        f"- Timing files: {len(run.records)} records, {run.skipped_records} invalid lines. "
        f"Complete puzzles (a `finished` line and a `puzzle` record): {len(ids)}; incomplete "
        f"(no `finished` line; left out below): {unfinished}.",
    ]
    if previous:
        lines.append(
            f"- Span against the printed `finished ... in <s>s`: {len(previous)} puzzles start "
            f"at their rank's previous `finished` line; span minus printed seconds ranges from "
            f"{min(deviation):+.3f} to {max(deviation):+.3f} s; {len(outside)} differ by more "
            f"than {SPAN_TOLERANCE_S:.0f} s{': ' + ', '.join(outside) if outside else ''}."
        )
    lines.append(
        f"- The {len(first)} first puzzles of their ranks ({', '.join(first) or 'none'}) start "
        "at the printed seconds by construction: no line marks their start."
    )
    if lag:
        lines.append(
            f"- Training against timestamps: the first rank line of a puzzle comes "
            f"{min(lag):.1f} to {max(lag):.1f} s (median {statistics.median(lag):.1f} s) after "
            f"start plus `train_runtime`; earlier for {early} of {len(lag)} puzzles."
        )
    lines += [
        f"- Train plus decode plus selection is at most the span for {len(ids) - len(over)} of "
        f"{len(ids)} puzzles{': not ' + ', '.join(over) if over else ''}; the smallest "
        f"overhead is {min(run.overhead(p) for p in ids):.1f} s.",
        f"- DFS cap: the longest decode call took {longest:.1f} s; "
        f"{sum(s >= DFS_CAP_S for s in decode_calls)} of {len(decode_calls)} calls reached the "
        f"{DFS_CAP_S:.0f} s cap.",
    ]
    if residuals:
        tex_params = f"{REFERENCE_PARAMS:,}".replace(",", "{,}")
        lines.append(
            f"- Token conversion: `train_tokens` in `puzzles.csv` equal "
            f"$T = \\text{{total\\_flos}} / (6P)$ with $P = {tex_params}$ for "
            f"{len(with_flos) - mismatched} of {len(with_flos)} puzzles; the largest distance of "
            f"the quotient from an integer is {max(residuals):.2e}."
        )
    return lines


def section_ops(run: RunData) -> list[str]:
    """Per-op statistics per puzzle and per call, with shares of worker time."""
    ids = sorted(run.puzzles)
    workers = len(worker_rows(run))
    worker_seconds = workers * (run.exit - run.launch)
    rows = []
    for op in (TRAIN, DECODE, SELECTION):
        rows.append([op, *_stats_cells(describe(run.op(p, op) for p in ids), worker_seconds)])
    rows.append(["overhead", *_stats_cells(describe(run.overhead(p) for p in ids), worker_seconds)])
    rows.append(["puzzle", *_stats_cells(describe(run.op(p, PUZZLE) for p in ids), worker_seconds)])
    per_call = []
    for op, unit in ((DECODE, "decode, per batch"), (SELECTION, "selection, per candidate")):
        seconds = [r.seconds for r in run.records if r.op == op and r.puzzle_id in run.puzzles]
        if seconds:
            per_call.append([unit, *_stats_cells(describe(seconds), None)])
    header = ["Op, per puzzle", "n", "mean s", "median s", "p90 s", "max s", "total s"]
    lines = ["## 2. Seconds per op", ""]
    lines += _table([*header, "share of worker time"], rows)
    lines += [""]
    lines += _table(["Op, per call", *header[1:]], per_call)
    lines += [
        "",
        f"Worker time is the {workers} workers times the solver's wall time "
        f"({_s(run.exit - run.launch)} s from launch to exit): {_s(worker_seconds)} s. Overhead "
        "is the span minus train, decode, and selection. p90 is the nearest-rank percentile.",
    ]
    return lines


@dataclass(frozen=True)
class WorkerRow:
    """One worker's time, in seconds; times are relative to the solver's launch.

    Attributes:
        rank: Worker rank.
        puzzles: Puzzles it finished.
        startup: Launch to its ``start!`` line (imports; rank :math:`r` waits for rank
            :math:`r - 1`'s marker file).
        load: Its ``model_load`` record.
        busy: Sum of its puzzle spans.
        train: Its train seconds.
        decode: Its decode seconds.
        selection: Its selection seconds.
        overhead: Its overhead seconds.
        finish: Its last ``finished`` line.
        unfinished: From the start of a puzzle it never finished (a hard stop) to the solver's
            exit.
        idle: Its last ``finished`` line to the solver's exit, less ``unfinished``.
        last: Its last puzzle.
        last_start: Start of its last puzzle.
    """

    rank: int
    puzzles: int
    startup: float
    load: float
    busy: float
    train: float
    decode: float
    selection: float
    overhead: float
    finish: float
    unfinished: float
    idle: float
    last: str
    last_start: float


def worker_rows(run: RunData) -> list[WorkerRow]:
    """Return each worker's time from the puzzle rows, the records, and its ``start!`` line.

    Args:
        run: The run's data.

    Returns:
        One row per rank, by rank.
    """
    by_rank: dict[int, list[str]] = defaultdict(list)
    for puzzle_id, row in run.puzzles.items():
        by_rank[int(row["rank"])].append(puzzle_id)
    loads = {r.worker: r.seconds for r in run.records if r.op == MODEL_LOAD}
    workers = []
    for rank in sorted(by_rank):
        ids = sorted(by_rank[rank], key=lambda p: int(run.puzzles[p]["index"]))
        first_start = float(run.puzzles[ids[0]]["start_s"])
        last = ids[-1]
        end = float(run.puzzles[last]["end_s"])
        log = run.ranks.get(rank)
        start_line = first_start if log is None or log.start_line is None else log.start_line
        unfinished = math.fsum(
            run.exit - float(row["start_s"])
            for row in run.incomplete.values()
            if int(row["rank"]) == rank and row["start_s"]
        )
        workers.append(
            WorkerRow(
                rank=rank,
                puzzles=len(ids),
                startup=start_line - run.launch,
                load=loads.get(f"rank{rank}", first_start - start_line),
                busy=math.fsum(run.op(p, PUZZLE) for p in ids),
                train=math.fsum(run.op(p, TRAIN) for p in ids),
                decode=math.fsum(run.op(p, DECODE) for p in ids),
                selection=math.fsum(run.op(p, SELECTION) for p in ids),
                overhead=math.fsum(run.overhead(p) for p in ids),
                finish=end - run.launch,
                unfinished=unfinished,
                idle=run.exit - end - unfinished,
                last=last,
                last_start=float(run.puzzles[last]["start_s"]) - run.launch,
            )
        )
    return workers


def section_worker_time(run: RunData) -> list[str]:
    """Where the workers' time went, from launch to exit."""
    workers = worker_rows(run)
    total = len(workers) * (run.exit - run.launch)
    parts = [
        ("startup: launch to `start!` (imports; ranks start one after another)", "startup"),
        ("model load: `start!` to the first puzzle", "load"),
        ("train", "train"),
        ("decode", "decode"),
        ("selection", "selection"),
        ("overhead", "overhead"),
        ("unfinished puzzles: no `finished` line before the exit", "unfinished"),
        ("idle tail: last `finished` to the solver's exit", "idle"),
    ]
    rows = []
    accounted = 0.0
    for label, key in parts:
        value = math.fsum(getattr(w, key) for w in workers)
        accounted += value
        rows.append([label, _s(value), _pct(value, total)])
    rows.append(["sum", _s(accounted), _pct(accounted, total)])
    lines = ["## 3. Worker time", ""]
    lines += _table(["Part", "seconds", f"share of {_s(total)} worker seconds"], rows)
    return lines


def cost_model(run: RunData) -> list[Fit]:
    """Fit the cost model per op on the exact token axis and on K0's size proxies.

    Args:
        run: The run's data.

    Returns:
        The fits, train first.
    """
    ids = sorted(run.puzzles)
    uncapped = [p for p in ids if not run.timed_out(p)]
    every = f"all {len(ids)}"
    free = f"{len(uncapped)} without the {PUZZLE_CAP_S:.0f} s cap"
    cells = "cells (K0, proxy)"

    def cells_of(puzzle_id: str) -> float:
        return run.size(puzzle_id, "cells")

    def seconds_of(op: str) -> Callable[[str], float]:
        return lambda puzzle_id: run.op(puzzle_id, op)

    per_puzzle: list[
        tuple[str, str, str, list[str], Callable[[str], float], Callable[[str], float]]
    ]
    per_puzzle = [
        ("train", "T, input tokens (exact)", every, ids, run.tokens, seconds_of(TRAIN)),
        ("train", cells, every, ids, cells_of, seconds_of(TRAIN)),
        ("decode", cells, every, ids, cells_of, seconds_of(DECODE)),
        ("decode", cells, free, uncapped, cells_of, seconds_of(DECODE)),
        (
            "decode",
            "cells times test inputs (K0, proxy)",
            free,
            uncapped,
            lambda p: cells_of(p) * run.size(p, "test_inputs"),
            seconds_of(DECODE),
        ),
        ("selection", cells, every, ids, cells_of, seconds_of(SELECTION)),
        ("overhead", cells, every, ids, cells_of, run.overhead),
        ("puzzle", cells, every, ids, cells_of, seconds_of(PUZZLE)),
        ("puzzle", cells, free, uncapped, cells_of, seconds_of(PUZZLE)),
    ]
    fits = [
        fit(f"{op}, per puzzle", axis, subset, [(size(p), seconds(p)) for p in chosen])
        for op, axis, subset, chosen, size, seconds in per_puzzle
    ]
    for op, label in ((DECODE, "decode, per batch"), (SELECTION, "selection, per candidate")):
        points = [
            (cells_of(r.puzzle_id), r.seconds)
            for r in run.records
            if r.op == op and r.puzzle_id in run.puzzles
        ]
        fits.append(fit(label, cells, f"all {len(points)} calls", points))
    return fits


@dataclass(frozen=True)
class TrainModel:
    """A model of train seconds per puzzle against its input tokens :math:`T`.

    Attributes:
        formula: The model.
        n: Puzzles.
        a: Intercept in seconds.
        b: Seconds per token; None when :math:`T` does not vary.
        c: Seconds per squared token per step, the attention term; None for the line.
        r2: Coefficient of determination; None when undefined.
        worst: Largest absolute residual in seconds.
        worst_puzzle: The puzzle with that residual.
    """

    formula: str
    n: int
    a: float
    b: float | None
    c: float | None
    r2: float | None
    worst: float
    worst_puzzle: str


def train_models(run: RunData) -> list[TrainModel]:
    r"""Fit train seconds per puzzle on :math:`T`, without and with :math:`T^2 / \text{steps}`.

    Why: a step's attention cost grows with the square of its length :math:`T / \text{steps}`,
    so over a puzzle's steps it grows with :math:`T^2 / \text{steps}`; a line alone tilts. The
    second model is left out when it cannot be fitted (fewer than three puzzles, or collinear).

    Args:
        run: The run's data.

    Returns:
        The line, then the model with the attention term when it can be fitted.
    """
    ids = sorted(run.puzzles)
    tokens = [float(run.tokens(p)) for p in ids]
    steps = [int(run.puzzles[p]["global_step"]) for p in ids]
    squares = [x * x / k for x, k in zip(tokens, steps, strict=True)]
    seconds = [run.op(p, TRAIN) for p in ids]
    a, b, r2 = timing_report.fit_linear(tokens, seconds)
    candidates = [("a + b T", a, b, None, r2, [a + (b or 0.0) * x for x in tokens])]
    try:
        a2, b2, c2, r22 = fit_two(tokens, squares, seconds)
    except ValueError:
        pass
    else:
        curve = [a2 + b2 * x + c2 * q for x, q in zip(tokens, squares, strict=True)]
        candidates.append(("a + b T + c T^2 / steps", a2, b2, c2, r22, curve))
    models = []
    for formula, a_, b_, c_, r2_, predicted in candidates:
        residuals = [abs(y - f) for y, f in zip(seconds, predicted, strict=True)]
        worst = max(range(len(ids)), key=residuals.__getitem__)
        models.append(TrainModel(formula, len(ids), a_, b_, c_, r2_, residuals[worst], ids[worst]))
    return models


def section_cost_model(fits: Sequence[Fit], models: Sequence[TrainModel], steps: str) -> list[str]:
    """The cost model table and the train models with and without the attention term."""
    rows = [[f.op, f.axis, f.subset, f.n, f"{f.a:.2f}", _slope(f.b), _r2(f.r2)] for f in fits]
    poor = [f"{f.op} on {f.axis.split(' (')[0]} ({f.subset})" for f in fits if (f.r2 or 0) < 0.5]
    lines = ["## 4. Cost model: seconds = a + b * size", ""]
    lines += _table(["Op", "size axis", "puzzles", "n", "a (s)", "b (s per unit)", "R^2"], rows, 3)
    lines += [
        "",
        "Least squares (`timing_report.fit_linear`). $T$ is exact (section 5). Decode and "
        "selection token counts are not in the log, so their axis is K0's per-task `cells` "
        "(cells of the demonstration pairs and the test inputs), a proxy until P0.3b logs "
        "tokens; a decode batch and a selection candidate take their task's `cells`. Straight "
        f"lines with $R^2$ below 0.5: {'; '.join(poor) or 'none'}.",
        "",
        "Train per puzzle with an attention term: a step's attention cost grows with the square "
        "of its length $T / \\text{steps}$, so over a puzzle it grows with "
        f"$T^2 / \\text{{steps}}$ (steps per puzzle: {steps}):",
        "",
    ]
    lines += _table(
        ["Model", "n", "a (s)", "b (s per token)", "c (s per token^2)", "R^2", "worst residual s"],
        [
            [
                f"`{m.formula}`",
                m.n,
                f"{m.a:.2f}",
                _slope(m.b),
                _slope(m.c),
                _r2(m.r2),
                f"{m.worst:.1f} ({m.worst_puzzle})",
            ]
            for m in models
        ],
    )
    return lines


def section_tokens(run: RunData) -> list[str]:
    """The ``total_flos`` conversion and its cross-check against K0's cells."""
    ids = sorted(run.puzzles)
    flos = {p: float(run.puzzles[p]["total_flos"]) for p in ids}
    without_heads = REFERENCE_PARAMS - HEAD_ROWS * HIDDEN
    options = [
        ("no head", without_heads),
        ("one 16-row head", without_heads + 16 * HIDDEN),
        ("two 16-row heads (used)", REFERENCE_PARAMS),
        ("one 151,936-row head (Qwen3-4B's vocabulary)", without_heads + 151936 * HIDDEN),
    ]
    rows = [
        [label, f"{params:,}", f"{max(flos_to_tokens(flos[p], params)[1] for p in ids):.3f}"]
        for label, params in options
    ]
    steps = {p: int(run.puzzles[p]["global_step"]) for p in ids}
    multiples = sum(run.tokens(p) % 64 == 0 for p in ids)
    ratio = {p: run.tokens(p) / steps[p] / run.size(p, "cells") for p in ids}
    groups: dict[int, list[float]] = defaultdict(list)
    for p in ids:
        groups[run.size(p, "test_inputs")].append(ratio[p])
    by_inputs = "; ".join(
        f"{k} test input{'s' if k > 1 else ''}: {len(v)} puzzles, median {statistics.median(v):.2f}"
        for k, v in sorted(groups.items())
    )
    per_step = describe(run.tokens(p) / steps[p] for p in ids)
    square = [
        p
        for p in ids
        if run.size(p, "cells")
        == (2 * run.size(p, "train_pairs") + run.size(p, "test_inputs"))
        * run.size(p, "max_side") ** 2
    ]
    counted = {
        p: square_sequence_tokens(run.size(p, "max_side"), run.size(p, "train_pairs"))
        for p in square
    }
    exact = sum(run.tokens(p) == steps[p] * counted[p][0] for p in square)
    cut = sum(counted[p][1] < run.size(p, "train_pairs") for p in square)
    largest = sorted(ids, key=lambda p: (-run.tokens(p), p))[:3]
    lines = [
        "## 5. Training tokens",
        "",
        "$T = \\text{total\\_flos} / (6P)$. In transformers 4.55.4 the trainer adds "
        "`floating_point_ops(inputs)` per step (`trainer.py:2598`), which calls the model's "
        "(`trainer.py:4707`): `6 * estimate_tokens(inputs) * "
        "num_parameters(exclude_embeddings=True)` (`modeling_utils.py:1941`), where "
        "`estimate_tokens` is `input_ids.numel()` (`modeling_utils.py:1909`) and the count drops "
        "only the weights of `nn.Embedding` modules (`modeling_utils.py:1857` to 1863). peft "
        "0.17.1 forwards the attribute from `PeftModel` (`peft_model.py:853`) through `LoraModel` "
        "(`tuners/lora/model.py:379`) to the Qwen3 model with the LoRA layers injected. With "
        "$P$ = 3,633,511,936 (36 layers and the final norm) + 528,482,304 (LoRA $r = 256$ on the "
        f"seven projections) + 81,920 (heads, $32 \\cdot 2560$) = {REFERENCE_PARAMS:,}, every "
        "quotient is an integer; with each neighboring count it is not:",
        "",
    ]
    lines += _table(["Parameter count", "P", "max distance from an integer"], rows)
    lines += [
        "",
        f"- Exact count: {len(square)} puzzles have every grid $s \\times s$ (K0's `cells` "
        "equal $(2n + m) s^2$ for $n$ pairs, $m$ test inputs, and largest side $s$). Each of "
        "their training sequences has $n (2 (s^2 + s - 1) + 8)$ tokens: per grid $s^2$ digits "
        "and $s - 1$ newlines, per pair 8 markers (`<|im_start|>`, the role, and a newline "
        f"before each grid, `<|im_end|>` after it); {cut} of them reach the "
        f"{TRAIN_MAX_TOKENS:,}-token cut. $T$ equals the steps times this count for {exact} of "
        f"{len(square)}. Integrality alone admits $P / 2$ (every $T$ doubled); this count fixes "
        "$T$, and with it $P$.",
        f"- Training steps per puzzle: {', '.join(map(str, sorted(set(steps.values()))))} "
        "(the reference trains one epoch on 16 color permutations of the 8 dihedral views); "
        f"$T$ is a multiple of 64 for {multiples} of {len(ids)} puzzles (each orientation class "
        "of 64 sequences has one length).",
        f"- Tokens per step, $T$ over the steps: mean {per_step.mean:.0f}, median "
        f"{per_step.median:.0f}, max {per_step.max:.0f}. Largest $T$: "
        + ", ".join(f"{p} {run.tokens(p):,} (train {run.op(p, TRAIN):.1f} s)" for p in largest)
        + ".",
        f"- Cross-check against K0's `cells`: tokens per step over cells has median "
        f"{statistics.median(ratio.values()):.2f}, range {min(ratio.values()):.2f} to "
        f"{max(ratio.values()):.2f}, above 1 for {sum(v > 1 for v in ratio.values())} of "
        f"{len(ids)} puzzles; {by_inputs}.",
    ]
    return lines


def section_workers(run: RunData) -> list[str]:
    """Per-worker table and the critical path."""
    workers = worker_rows(run)
    rows = [
        [
            w.rank,
            w.puzzles,
            _s(w.startup),
            _s(w.load),
            _s(w.busy),
            _s(w.finish),
            _s(w.idle),
            f"{w.finish / run.window:.3f}",
        ]
        for w in workers
    ]
    last = max(workers, key=lambda w: w.finish)
    drained = max(w.last_start for w in workers)
    wall = run.exit - run.launch
    balanced = math.fsum(w.startup + w.load + w.busy for w in workers) / len(workers)
    lines = ["## 6. Workers and the critical path", ""]
    lines += _table(
        [
            "Rank",
            "puzzles",
            "startup s",
            "model load s",
            "busy s",
            "finish s",
            "idle tail s",
            "u_w",
        ],
        rows,
    )
    lines += [
        "",
        f"Times from the solver's launch; busy is the sum of puzzle spans; $u_w$ = finish / "
        f"{_s(run.window)} s, the production window from the launch ({run.offset:g} s after "
        f"the notebook start) to the soft deadline ({run.deadline:,.0f} s after it). Solver "
        f"wall {_s(wall)} s.",
        "",
        f"- Critical path: launch, rank {last.rank}'s startup ({_s(last.startup)} s, behind "
        f"the earlier ranks' imports) and model load ({_s(last.load)} s), then its "
        f"{last.puzzles} puzzles ({_s(last.busy)} s); its last puzzle {last.last} started at "
        f"{_s(last.last_start)} s and finished at {_s(last.finish)} s; the solver exited "
        f"{_s(last.idle)} s later.",
        f"- The queue drained at {_s(drained)} s (the last puzzle taken); the other workers "
        f"then idled {', '.join(_s(w.idle) for w in workers if w is not last)} s.",
        f"- Perfect balance (startup, load, and busy time spread evenly) would end at "
        f"{_s(balanced)} s, {_s(wall - balanced)} s before the measured exit.",
    ]
    return lines


def section_cap(run: RunData) -> list[str]:
    """The puzzles that hit the 1200 s decode cap."""
    capped = sorted((p for p in run.puzzles if run.timed_out(p)), key=lambda p: -run.op(p, PUZZLE))
    rows = []
    for p in capped:
        ops = {op: run.op(p, op) for op in (TRAIN, DECODE, SELECTION)}
        planned = BATCHES_PER_TEST_INPUT * run.size(p, "test_inputs")
        rows.append(
            [
                p,
                run.puzzles[p]["rank"],
                _s(run.op(p, PUZZLE)),
                _s(ops[TRAIN]),
                _s(ops[DECODE]),
                _s(ops[SELECTION]),
                _s(run.overhead(p)),
                max(ops, key=lambda op: ops[op]),
                f"{run.calls[p].get(DECODE, 0)} of {planned}",
                run.puzzles[p]["timeout_after_s"],
                run.size(p, "cells"),
                run.size(p, "train_pairs"),
                run.size(p, "test_inputs"),
                run.size(p, "max_side"),
                f"{run.tokens(p):,}",
            ]
        )
    if not capped:
        return [f"## 7. Puzzles at the {PUZZLE_CAP_S:.0f} s cap", "", "None."]
    total = math.fsum(run.op(p, PUZZLE) for p in capped)
    shares = ", ".join(
        f"{op} {_pct(math.fsum(run.op(p, op) for p in capped), total)}"
        for op in (TRAIN, DECODE, SELECTION)
    )
    stops = [float(run.puzzles[p]["timeout_after_s"]) for p in capped]
    others = [p for p in run.puzzles if not run.timed_out(p)]
    lines = [f"## 7. Puzzles at the {PUZZLE_CAP_S:.0f} s cap", ""]
    lines += _table(
        [
            "Puzzle",
            "rank",
            "span s",
            "train s",
            "decode s",
            "selection s",
            "overhead s",
            "largest op",
            "decode batches",
            "timeout after s",
            "cells",
            "train pairs",
            "test inputs",
            "max side",
            "T",
        ],
        rows,
        2,
    )
    lines += [
        "",
        f"{len(capped)} of {len(run.puzzles)} puzzles; of their {_s(total)} s: {shares}. "
        f"Decode batches: done of planned ({BATCHES_PER_TEST_INPUT} per test input). The cap "
        f"is checked before each batch, so these puzzles stopped at {_s(min(stops))} to "
        f"{_s(max(stops))} s, {_s(min(stops) - PUZZLE_CAP_S)} to "
        f"{_s(max(stops) - PUZZLE_CAP_S)} s past it (the batch in flight and its selection).",
    ]
    if others:
        lines[-1] += (
            f" The other {len(others)} puzzles have median cells "
            f"{statistics.median(run.size(p, 'cells') for p in others):.0f} and median $T$ "
            f"{statistics.median(run.tokens(p) for p in others):,.0f}."
        )
    return lines


def section_b3(run: RunData, train_fit: Fit, models: Sequence[TrainModel]) -> list[str]:
    """Measured seconds per puzzle against B3's budget and G1's 90% rule."""
    ids = sorted(run.puzzles)
    puzzle = describe(run.op(p, PUZZLE) for p in ids)
    workers = worker_rows(run)
    fixed = [w.startup + w.load for w in workers]
    mean_fixed = math.fsum(fixed) / len(fixed)
    count = len(workers)
    u = max(w.finish for w in workers) / run.window
    step = None if train_fit.b is None else train_fit.b * B3_STEP_TOKENS
    curved = [m for m in models if m.c is not None and m.b is not None]
    if step is not None and curved:
        step_text = (
            f"{step:.2f} s (line: b times {B3_STEP_TOKENS:,}); "
            f"{curved[0].b * B3_STEP_TOKENS + curved[0].c * B3_STEP_TOKENS**2:.2f} s with the "
            f"attention term (b times {B3_STEP_TOKENS:,} plus c times {B3_STEP_TOKENS:,} squared)"
        )
    else:
        step_text = "n/a" if step is None else f"{step:.2f} s (line: b times {B3_STEP_TOKENS:,})"
    rows = [
        [
            "seconds per puzzle per worker",
            f"{B3_BUDGET_S:.0f} budget, {B3_PLANNING_S:.0f} planning",
            f"mean {_s(puzzle.mean)}, median {_s(puzzle.median)}, p90 {_s(puzzle.p90)}, "
            f"max {_s(puzzle.max)} (n = {puzzle.n})",
        ],
        [
            "headroom of the mean",
            "",
            f"{_s(B3_BUDGET_S - puzzle.mean)} against {B3_BUDGET_S:.0f}; "
            f"{_s(B3_PLANNING_S - puzzle.mean)} against {B3_PLANNING_S:.0f}",
        ],
        [
            "puzzles over the planning number",
            "",
            f"{sum(run.op(p, PUZZLE) > B3_PLANNING_S for p in ids)} of {len(ids)} over "
            f"{B3_PLANNING_S:.0f} s; {sum(run.op(p, PUZZLE) > B3_BUDGET_S for p in ids)} over "
            f"{B3_BUDGET_S:.0f} s",
        ],
        [
            "per-worker reserve before the first puzzle",
            "model load 2 to 3 min",
            f"startup plus model load {_s(min(fixed))} to {_s(max(fixed))} s "
            f"(mean {_s(mean_fixed)})",
        ],
        [
            f"train step on {B3_STEP_TOKENS:,} tokens",
            "about 1 s",
            step_text,
        ],
    ]
    projection = []
    for tasks in HIDDEN_TASKS:
        per_worker = tasks / count
        budget = (run.window - mean_fixed) / per_worker
        g1_budget = (G1_LIMIT * run.window - mean_fixed) / per_worker
        projected = (mean_fixed + per_worker * puzzle.mean) / run.window
        projection.append(
            [
                tasks,
                f"{per_worker:g}",
                _s(budget),
                _s(budget - puzzle.mean),
                _s(g1_budget - puzzle.mean),
                f"{projected:.3f}",
                "above" if projected > G1_LIMIT else "at or below",
            ]
        )
    lines = ["## 8. B3 comparison and G1's 90% rule", ""]
    lines += _table(["Quantity", "B3", "measured"], rows, 3)
    lines += [
        "",
        f"Projection to a hidden set of $N$ tasks with this run's mean puzzle time, the mean "
        f"startup and model load, {count} workers, and the {_s(run.window)} s window "
        "(estimate: hidden tasks sized like these, no tail term for the last puzzle):",
        "",
    ]
    lines += _table(
        [
            "N tasks",
            "puzzles per worker",
            "budget per puzzle s",
            "headroom per puzzle s",
            "headroom at G1's 90% s",
            "projected u",
            f"against G1's {G1_LIMIT}",
        ],
        projection,
    )
    lines += [
        "",
        f"Measured $u = \\max_w$ finish$_w$ / {_s(run.window)} s = {u:.3f} on this run "
        f"({len(ids)} tasks), against G1's limit {G1_LIMIT}.",
    ]
    return lines


def section_solved(
    rows: Sequence[OutputRow], problems: Sequence[str], notebook: Mapping[str, str]
) -> list[str]:
    """Per-task credit from the ``ALL_CORRECT`` lines and the submission."""
    score = subset_score(rows, lambda r: r.submitted_correct)
    probmul = subset_score(rows, lambda r: r.probmul_correct)
    with_candidates = [r for r in rows if r.top2_matches is not None]
    matches = sum(bool(r.top2_matches) for r in with_candidates)
    marked = sum(r.correct_candidates for r in rows)
    total = sum(r.candidates for r in rows)
    reload = notebook.get("reload")
    agree = reload is not None and abs(float(reload) - score.score) < 1e-9
    known = [r for r in rows if r.correct_rank is not None]
    consistent = sum(r.submitted_correct == (r.correct_rank in (1, 2)) for r in known)
    lines = [
        "## 9. Solved tasks",
        "",
        f"- Credit (`arc26.evaluate`, an output counts when an attempt equals the grid of a "
        f"candidate marked `ALL_CORRECT`): **{score.score:.2f} of {score.tasks} tasks "
        f"({score.percent:.2f}%)**; {score.solved_outputs} of {score.test_outputs} test outputs "
        f"({score.output_percent:.2f}%). The notebook printed `Reload score: {reload}`: "
        f"{'equal' if agree else 'DIFFERENT'}.",
        f"- `score_kgmon`'s first two grids are the submitted attempts for {matches} of "
        f"{len(with_candidates)} test outputs with candidates; an output with a correct "
        f"candidate is solved exactly when its rank is 1 or 2 for {consistent} of {len(known)}.",
        f"- Cross-check: `score_full_probmul_3` credit {probmul.score:.2f} (notebook "
        f"`{notebook.get('score_full_probmul_3', 'n/a')}`), `score_kgmon` "
        f"{score.score:.2f} (notebook `{notebook.get('score_kgmon', 'n/a')}`); candidates "
        f"marked correct {marked} of {total} (notebook `subkeys: "
        f"{notebook.get('subkeys', 'n/a')}`).",
        f"- Problems: {'; '.join(problems) if problems else 'none'}.",
        "",
    ]
    table = [
        [task.task_id, f"{task.solved} of {task.outputs}", f"{task.score:.2f}"]
        for task in score.per_task
        if task.score > 0
    ]
    full = sum(task.score == 1 for task in score.per_task)
    lines += [f"Tasks with credit ({len(table)} of {score.tasks}; {full} fully solved):", ""]
    lines += _table(["Task", "outputs solved", "credit"], table)
    return lines


def section_candidates(rows: Sequence[OutputRow]) -> list[str]:
    """Candidate statistics for P2.2 and P2.8, and the oracle gap."""
    score = subset_score(rows, lambda r: r.submitted_correct)
    oracle = subset_score(rows, lambda r: r.correct_rank is not None)
    with_candidates = [r for r in rows if r.candidates]
    tasks_with = {r.task_id for r in with_candidates}
    tasks_all = {r.task_id for r in rows}
    correct = [r for r in rows if r.correct_rank is not None]
    ranks: dict[int, int] = defaultdict(int)
    for row in correct:
        ranks[row.correct_rank or 0] += 1
    gap = [r for r in correct if not r.submitted_correct]
    count = describe(r.candidates for r in with_candidates)
    distinct = describe(r.distinct for r in with_candidates)
    lines = [
        "## 10. Candidates (P2.2, P2.8)",
        "",
        f"- Test outputs: {len(rows)}; with candidates {len(with_candidates)} (in "
        f"{len(tasks_with)} of {len(tasks_all)} tasks); tasks without any candidate: "
        f"{', '.join(sorted(tasks_all - tasks_with)) or 'none'}.",
        f"- Candidates per output with candidates: mean {count.mean:.1f}, median "
        f"{count.median:.0f}, p90 {count.p90:.0f}, max {count.max:.0f}, total {count.total:.0f}; "
        f"distinct grids: mean {distinct.mean:.1f}, median {distinct.median:.0f}, p90 "
        f"{distinct.p90:.0f}, max {distinct.max:.0f}.",
        f"- A correct candidate exists for {len(correct)} of {len(rows)} test outputs (in "
        f"{len({r.task_id for r in correct})} of {len(tasks_all)} tasks).",
        f"- Oracle (an output counts when any candidate is correct): {oracle.score:.2f} of "
        f"{oracle.tasks} tasks ({oracle.percent:.2f}%) against {score.score:.2f} submitted; "
        f"gap {oracle.score - score.score:.2f} tasks: {len(gap)} test outputs in "
        f"{len({r.task_id for r in gap})} tasks have a correct candidate that was not "
        "submitted.",
        "",
        "Rank of the correct grid under `score_kgmon` (ranks 1 and 2 are submitted):",
        "",
    ]
    lines += _table(["Rank", "test outputs"], [[k, v] for k, v in sorted(ranks.items())])
    lines += ["", "Oracle gap:", ""]
    lines += _table(
        ["Test output", "candidates", "distinct grids", "correct candidates", "rank"],
        [
            [
                f"{r.task_id}_{r.index}",
                r.candidates,
                r.distinct,
                r.correct_candidates,
                r.correct_rank,
            ]
            for r in gap
        ],
    )
    lines += ["", "Every test output: `outputs.csv` (no grids)."]
    return lines


def write_outputs_csv(rows: Sequence[OutputRow], path: Path) -> None:
    """Write one row per test output, without grids.

    Args:
        rows: Output rows.
        path: Destination.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_FIELDS)
        for r in rows:
            writer.writerow(
                [
                    r.task_id,
                    r.index,
                    r.candidates,
                    r.distinct,
                    r.correct_candidates,
                    int(r.correct_rank is not None),
                    "" if r.correct_rank is None else r.correct_rank,
                    int(r.submitted_correct),
                    "" if r.top2_matches is None else int(r.top2_matches),
                ]
            )


def render_summary(
    run: RunData,
    rows: Sequence[OutputRow],
    problems: Sequence[str],
    inputs: Sequence[Path],
    command: str,
    provenance: Mapping[str, object],
) -> str:
    """Render ``summary.md``.

    Args:
        run: The run's data.
        rows: Output rows from :func:`candidate_outputs`.
        problems: Problems from :func:`candidate_outputs`.
        inputs: Files read, listed with their sha256.
        command: The command that wrote the summary.
        provenance: ``tag``, ``commit``, ``hash_seed``, ``subset``, ``workers`` when known.

    Returns:
        The Markdown text, ending in a newline.
    """
    fits = cost_model(run)
    models = train_models(run)
    steps = ", ".join(sorted({run.puzzles[p]["global_step"] for p in run.puzzles}, key=int))
    notebook = notebook_scores(run.lines)
    title = provenance.get("tag") or "Run"
    commit = provenance.get("commit") or "unknown"
    seed = provenance.get("hash_seed", "unknown")
    subset = provenance.get("subset", "unknown") or "none (every task)"
    lines = [
        f"# {title}: timing, cost model, and candidates from the session log",
        "",
        "Generated by `scripts/session_log_timing.py summary`; do not edit by hand. Code commit "
        f"`{commit}`, `PYTHONHASHSEED={seed}`, subset filter {subset}, "
        f"{len(run.puzzles)} puzzles on {len(run.ranks)} workers.",
        "",
        "Commands, from the repository root:",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    lines += _table(["Input", "sha256 (first 12)"], [[f"`{p}`", sha256_prefix(p)] for p in inputs])
    lines += [""]
    for section in (
        section_checks(run),
        section_ops(run),
        section_worker_time(run),
        section_cost_model(fits, models, steps),
        section_tokens(run),
        section_workers(run),
        section_cap(run),
        section_b3(run, fits[0], models),
        section_solved(rows, problems, notebook),
        section_candidates(rows),
    ):
        lines += [*section, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_summary(args: argparse.Namespace, command: str) -> tuple[Path, Path]:
    """Read every input, then write ``summary.md`` and ``outputs.csv``.

    Args:
        args: Parsed ``summary`` arguments.
        command: The command line, for the report.

    Returns:
        The summary and CSV paths.

    Raises:
        OSError: If a file cannot be read or written.
        ValueError: If an input is malformed.
    """
    provenance: dict[str, object] = {}
    offset = 0.0
    if args.stages is not None:
        stages = json.loads(args.stages.read_text(encoding="utf-8"))
        build, settings = stages.get("build", {}), stages.get("settings", {})
        provenance = {
            "tag": build.get("tag"),
            "commit": build.get("commit"),
            "hash_seed": settings.get("hash_seed"),
            "subset": settings.get("subset"),
        }
        offset = float(stages.get("stages", {}).get("run", {}).get("start_s", 0.0))
    run = load_run(args.timing, args.log, args.task_stats, args.window, offset)
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    rows, problems = candidate_outputs(candidates, submission, correct_candidate_keys(run.lines))
    inputs = [args.log, *run.timing_paths, args.timing / "puzzles.csv"]
    inputs += [args.candidates, args.submission, args.task_stats]
    if args.stages is not None:
        inputs.append(args.stages)
    text = render_summary(run, rows, problems, inputs, command, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    outputs_csv = args.outputs_csv or args.out.with_name("outputs.csv")
    write_outputs_csv(rows, outputs_csv)
    return args.out, outputs_csv


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``records`` or ``summary`` command.

    Args:
        argv: Command-line arguments without the program name; ``sys.argv[1:]`` when None.

    Returns:
        0 on success, 1 when an input cannot be read or holds no worker lines.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    records_cmd = commands.add_parser("records", help="timing JSONL and puzzles.csv from a log")
    records_cmd.add_argument("log", type=Path, help="the session log (a JSON array)")
    records_cmd.add_argument("--out", type=Path, required=True, help="output directory")
    records_cmd.add_argument(
        "--params", type=int, default=REFERENCE_PARAMS, help="P in T = total_flos / (6 P)"
    )
    summary_cmd = commands.add_parser("summary", help="summary.md and outputs.csv")
    summary_cmd.add_argument("--timing", type=Path, required=True, help="records output dir")
    summary_cmd.add_argument("--log", type=Path, required=True, help="the session log")
    summary_cmd.add_argument("--candidates", type=Path, required=True, help="candidates.json")
    summary_cmd.add_argument("--submission", type=Path, required=True, help="submission.json")
    summary_cmd.add_argument(
        "--task-stats", type=Path, required=True, help="K0's env_probe.json (per-task sizes)"
    )
    summary_cmd.add_argument("--stages", type=Path, help="the run's stages.json (provenance)")
    summary_cmd.add_argument("--out", type=Path, required=True, help="summary Markdown")
    summary_cmd.add_argument("--outputs-csv", type=Path, help="default: outputs.csv beside --out")
    summary_cmd.add_argument(
        "--window",
        type=float,
        default=WINDOW_S,
        help="production soft deadline, seconds after the notebook start",
    )
    args = parser.parse_args(argv)
    if args.command == "summary":
        given = list(sys.argv[1:] if argv is None else argv)
        command = "\n".join(
            [
                f"python scripts/session_log_timing.py records {args.log} --out {args.timing}",
                " ".join(["python scripts/session_log_timing.py", *given]),
            ]
        )
        try:
            out, outputs_csv = write_summary(args, command)
        except (OSError, ValueError, KeyError) as error:
            print(f"error: {error!r}", file=sys.stderr)
            return 1
        print(f"wrote {out} and {outputs_csv}")
        return 0
    try:
        result = write_records(args.log, args.out, args.params)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"{result.records} records in {len(result.files)} files, {result.puzzles} puzzles "
        f"({result.incomplete} incomplete, {result.unnamed} unnamed), skipped log entries "
        f"{result.skipped_entries}, truncated {result.truncated}, unknown rank lines "
        f"{result.unknown_lines}; {result.csv_path}"
    )
    return 0 if result.files else 1


if __name__ == "__main__":
    sys.exit(main())
