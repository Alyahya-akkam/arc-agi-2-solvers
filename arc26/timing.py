"""JSONL timing records for GPU operations (BRIEF B4 conventions, task P0.3a).

Every GPU operation appends one JSON line ``{"puzzle_id", "op", "seconds", "tokens"}``:
``puzzle_id`` is the task id (null for an op outside any puzzle, such as model loading), ``op``
the operation name, ``seconds`` its wall time from ``time.perf_counter``, and ``tokens`` the token
count it processed (null when unknown). ``scripts/timing_report.py`` turns a run's files into the
per-puzzle table and the cost model of B3 and P1.1.

Each worker process writes its own file, ``<log_dir>/timing-<worker>.jsonl``, so the four queue
workers never interleave lines. Each record is one append of one complete line, so a hard kill
loses at most the line being written, which the reader skips.

Standard op names. :data:`PUZZLE` is a puzzle's span on its worker; inside it run :data:`TRAIN`
(the TTT fit), :data:`DECODE` (DFS decoding), and :data:`SELECTION` (rescoring each decoded
candidate under augmented views, the reference's ``calc_scores``). Overhead is :data:`PUZZLE`
minus those three; it is never negative, since all three run inside the span.
:data:`MAIN_SELECTION` is the selection algorithm that the main process runs over all puzzles
after the workers finish (the reference's ``run_selection_algo``): it lies outside every puzzle
span and is recorded once, with a null ``puzzle_id``.

Wiring into the extracted notebook code is task P0.3b: each process (every worker, and the main
process for :data:`MAIN_SELECTION`) calls :func:`configure` once, then wraps ops in :func:`timed`.
This module holds no GPU code.
"""

from __future__ import annotations

import json
import math
import operator
import os
import re
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsIndex

TRAIN = "train"
DECODE = "decode"
SELECTION = "selection"
PUZZLE = "puzzle"
MAIN_SELECTION = "main_selection"
FILE_PREFIX = "timing-"
FILE_SUFFIX = ".jsonl"
FIELDS = ("puzzle_id", "op", "seconds", "tokens")
_SAFE_WORKER = re.compile(r"[A-Za-z0-9._-]+")


def _token_count(tokens: SupportsIndex | None) -> int | None:
    """A token count as a plain int; numpy and torch integer scalars convert via ``__index__``."""
    if tokens is None:
        return None
    if isinstance(tokens, bool):
        raise TypeError(f"tokens must be an integer or None; got {tokens!r}")
    try:
        count = operator.index(tokens)
    except TypeError:
        raise TypeError(f"tokens must be an integer or None; got {tokens!r}") from None
    if count < 0:
        raise ValueError(f"tokens must be non-negative; got {count}")
    return count


def make_record(
    puzzle_id: str | None, op: str, seconds: float, tokens: SupportsIndex | None = None
) -> dict[str, object]:
    """Validate the fields of one timing record and return it in schema order.

    Args:
        puzzle_id: Task id, or None for an op outside any puzzle.
        op: Operation name, for example :data:`TRAIN`.
        seconds: Wall time; finite and non-negative.
        tokens: Tokens processed by the op, or None when unknown. Any integer type converts to
            ``int`` through ``operator.index`` (a numpy or torch integer scalar, which JSON cannot
            serialize as is); booleans and non-integers are refused.

    Returns:
        ``{"puzzle_id", "op", "seconds", "tokens"}`` with ``seconds`` rounded to microseconds
        and ``tokens`` a plain ``int`` or None.

    Raises:
        TypeError: If a field has the wrong type (booleans are not numbers here).
        ValueError: If ``op`` is empty, ``seconds`` is negative or not finite, or ``tokens`` is
            negative.
    """
    if puzzle_id is not None and not isinstance(puzzle_id, str):
        raise TypeError(f"puzzle_id must be a str or None; got {puzzle_id!r}")
    if not isinstance(op, str):
        raise TypeError(f"op must be a str; got {op!r}")
    if not op:
        raise ValueError("op must be non-empty")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be a number; got {seconds!r}")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"seconds must be finite and non-negative; got {seconds!r}")
    count = _token_count(tokens)
    return {"puzzle_id": puzzle_id, "op": op, "seconds": round(float(seconds), 6), "tokens": count}


@dataclass
class OpTimer:
    """Handle yielded by :meth:`TimingLogger.timed` and :func:`timed`.

    Attributes:
        tokens: Token count to record; set it inside the block when it is known only there.
        seconds: Measured wall time, set when the block exits normally.
    """

    tokens: SupportsIndex | None = None
    seconds: float | None = None


class TimingLogger:
    """Appends timing records to this worker's JSONL file.

    Why: one file per worker process, because concurrent appends from four processes to one file
    are not guaranteed to keep lines whole; a lock keeps threads within a process apart. The file
    is opened, appended, and closed per record, so everything written survives a hard kill.

    Attributes:
        log_dir: Directory of the run's timing files; created on the first record.
        worker: Worker label in the file name, for example the queue rank.
        path: ``<log_dir>/timing-<worker>.jsonl``.
    """

    def __init__(self, log_dir: str | os.PathLike[str], worker: int | str | None = None) -> None:
        """Create a logger; it writes nothing until the first record.

        Args:
            log_dir: Directory of the run's timing files.
            worker: Worker label, for example the rank; the process id when None, so separate
                processes never share a file by default.

        Raises:
            ValueError: If the label is not letters, digits, ``.``, ``_``, ``-``, or is ``.``
                or ``..``.
        """
        label = f"pid{os.getpid()}" if worker is None else str(worker)
        if not _SAFE_WORKER.fullmatch(label) or label in {".", ".."}:
            raise ValueError(f"worker label must be a plain file-name part; got {label!r}")
        self.log_dir = Path(log_dir)
        self.worker = label
        self.path = self.log_dir / f"{FILE_PREFIX}{label}{FILE_SUFFIX}"
        self._lock = threading.Lock()

    def record(
        self,
        puzzle_id: str | None,
        op: str,
        seconds: float,
        tokens: SupportsIndex | None = None,
    ) -> dict[str, object]:
        """Append one record as one line.

        Args:
            puzzle_id: Task id, or None for an op outside any puzzle.
            op: Operation name.
            seconds: Wall time, for example a duration reported by the trainer.
            tokens: Tokens processed, or None when unknown.

        Returns:
            The record as written.

        Raises:
            TypeError: As :func:`make_record` raises it.
            ValueError: As :func:`make_record` raises it.
            OSError: If the file cannot be written.
        """
        entry = make_record(puzzle_id, op, seconds, tokens)
        line = (json.dumps(entry, allow_nan=False) + "\n").encode("utf-8")
        with self._lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab") as handle:
                handle.write(line)
        return entry

    @contextmanager
    def timed(
        self, puzzle_id: str | None, op: str, tokens: SupportsIndex | None = None
    ) -> Iterator[OpTimer]:
        """Time the enclosed block and record it when the block exits normally.

        A block that raises writes no record, so a failed op never becomes a cost-model sample;
        its time still falls into the enclosing :data:`PUZZLE` record as overhead.

        Args:
            puzzle_id: Task id, or None for an op outside any puzzle.
            op: Operation name.
            tokens: Token count if known up front; the yielded timer's ``tokens`` can set it
                inside the block.

        Yields:
            The timer; ``seconds`` is set once the block has exited normally.
        """
        timer = OpTimer(tokens=tokens)
        start = time.perf_counter()
        yield timer
        timer.seconds = time.perf_counter() - start
        self.record(puzzle_id, op, timer.seconds, timer.tokens)


_process_logger: TimingLogger | None = None


def configure(log_dir: str | os.PathLike[str], worker: int | str | None = None) -> TimingLogger:
    """Set this process's logger, which :func:`timed` writes to.

    Why: the extracted notebook functions record ops without a logger argument threaded through
    their signatures (move, do not rewrite). Each worker process configures its own.

    Args:
        log_dir: Directory of the run's timing files.
        worker: Worker label, for example the rank; the process id when None.

    Returns:
        The new logger.
    """
    global _process_logger
    _process_logger = TimingLogger(log_dir, worker)
    return _process_logger


def get_logger() -> TimingLogger | None:
    """Return this process's logger, or None before :func:`configure`."""
    return _process_logger


@contextmanager
def timed(
    puzzle_id: str | None, op: str, tokens: SupportsIndex | None = None
) -> Iterator[OpTimer]:
    """Time the enclosed block with this process's logger, as :meth:`TimingLogger.timed`.

    Before :func:`configure` the block is still timed (``seconds`` is set) but nothing is
    written, so CPU tests of instrumented code need no log directory.

    Args:
        puzzle_id: Task id, or None for an op outside any puzzle.
        op: Operation name.
        tokens: Token count if known up front.

    Yields:
        The timer.
    """
    logger = _process_logger
    if logger is not None:
        with logger.timed(puzzle_id, op, tokens) as timer:
            yield timer
        return
    timer = OpTimer(tokens=tokens)
    start = time.perf_counter()
    yield timer
    timer.seconds = time.perf_counter() - start


@dataclass(frozen=True)
class TimingRecord:
    """One timing record as read back, with the worker taken from its file name.

    Attributes:
        puzzle_id: Task id, or None for an op outside any puzzle.
        op: Operation name.
        seconds: Wall time.
        tokens: Tokens processed, or None.
        worker: Worker label from ``timing-<worker>.jsonl`` (the file stem otherwise).
    """

    puzzle_id: str | None
    op: str
    seconds: float
    tokens: int | None
    worker: str


def worker_label(path: Path) -> str:
    """Return the worker label encoded in a timing file's name.

    Args:
        path: A timing file.

    Returns:
        ``<worker>`` for ``timing-<worker>.jsonl``, else the file stem.
    """
    name = path.name
    if name.startswith(FILE_PREFIX) and name.endswith(FILE_SUFFIX):
        return name[len(FILE_PREFIX) : -len(FILE_SUFFIX)]
    return path.stem


def _parse(line: str, worker: str) -> TimingRecord | None:
    try:
        data = json.loads(line)
    except ValueError:
        return None
    if not isinstance(data, Mapping) or any(field not in data for field in FIELDS):
        return None
    try:
        make_record(data["puzzle_id"], data["op"], data["seconds"], data["tokens"])
    except (TypeError, ValueError):
        return None
    return TimingRecord(
        data["puzzle_id"], data["op"], float(data["seconds"]), data["tokens"], worker
    )


def read_records(paths: Iterable[str | os.PathLike[str]]) -> tuple[list[TimingRecord], int]:
    """Read timing records from JSONL files, skipping lines that are not valid records.

    Why: a hard kill can leave a truncated last line, and the report must still run on
    everything else. Blank lines are ignored; unknown extra keys are allowed.

    Args:
        paths: Timing files, read in the given order.

    Returns:
        The records in file and line order, and the number of skipped lines.

    Raises:
        OSError: If a file cannot be read.
    """
    records: list[TimingRecord] = []
    skipped = 0
    for raw_path in paths:
        path = Path(raw_path)
        worker = worker_label(path)
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parsed = _parse(line, worker)
                if parsed is None:
                    skipped += 1
                else:
                    records.append(parsed)
    return records, skipped
