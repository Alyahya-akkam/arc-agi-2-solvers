"""Choose the two attempts per test input from the agent's records and NVARC (P6.2).

The rules (``docs/research/agent_library_design.md``, section 4), per test input:

1. Verified programs are grouped by their output on that test input. Groups rank by votes (the
   independent attempts behind them), then by their smallest program (AST nodes; Hodel's
   "shorter is better"), then non-demoted first (a group is demoted when every program behind it
   has a non-fatal guard finding), then first seen.
2. When NVARC's first attempt equals a verified group's output, that group goes first.
3. With a verified group, attempt 1 is the top group and attempt 2 the next group, else NVARC's
   first attempt if it differs, else the model's direct answer, else the best unverified
   program's output, else NVARC's second attempt.
4. Without one, NVARC's two attempts come first (BARC's induction-then-transduction ensemble),
   then direct answers, then the best unverified programs.

Direct answers rank by votes (attempts that gave the same grid), then first seen. Unverified
programs are those that are not verified and have no fatal guard finding; they rank by mean cell
accuracy over the demonstrations, then demonstrations passed, then size. An NVARC grid equal to
``[[0]]`` is NVARC's placeholder for a slot it did not fill (``ArcDataset.get_submission``
pre-fills both attempts with it), so it counts as missing. Grids are never repeated within a
test input; a slot nothing fills gets ``[[0]]``, so every task of the challenges file has a valid
entry.
"""

from __future__ import annotations

import ast
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc26.agent.loop import plain, read_records
from arc26.agent.types import AttemptRecord, Candidate, Verification
from arc26.data import Grid
from arc26.submit import (
    ATTEMPTS,
    entry_counts,
    fallback_grid,
    grid_problem,
    validate,
    write_json_atomic,
)

# NVARC's two attempts for one test input; None where its grid is missing or invalid.
NvarcPair = tuple[Grid | None, Grid | None]
# Size given to a source that does not parse, so it ranks after every parsable program.
UNPARSABLE_SIZE = 10**9


def program_size(source: str) -> int:
    """Return a program's size as its number of AST nodes.

    Args:
        source: Python source.

    Returns:
        The node count of ``ast.parse(source)``, or :data:`UNPARSABLE_SIZE` on a syntax error.
    """
    try:
        return sum(1 for _ in ast.walk(ast.parse(source)))
    except (SyntaxError, ValueError):
        return UNPARSABLE_SIZE


def _valid(grid: Any) -> Grid | None:
    """A valid grid as plain lists (numpy values converted), else None."""
    try:
        grid = plain(grid)
    except (TypeError, ValueError):
        return None
    return grid if grid_problem(grid) is None else None


def _nvarc_grid(grid: Any) -> Grid | None:
    """One of NVARC's attempts, or None when it is missing, invalid, or the ``[[0]]`` filler."""
    grid = None if grid is None else _valid(grid)
    return None if grid == fallback_grid() else grid


def _key(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))


def nvarc_from_submission(data: Any) -> dict[str, list[NvarcPair]]:
    """Read NVARC's attempts from a parsed ``submission.json``.

    Args:
        data: Task id to a list with one ``{"attempt_1": grid, "attempt_2": grid}`` per test
            input (Kaggle's format).

    Returns:
        Task id to one ``(attempt_1, attempt_2)`` per test input, with None for a grid that is
        missing, invalid, or NVARC's ``[[0]]`` placeholder; entries that are not objects give
        ``(None, None)``.

    Raises:
        ValueError: If ``data`` is not a JSON object.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"NVARC submission must be a JSON object; got {type(data).__name__}")
    out: dict[str, list[NvarcPair]] = {}
    for task_id, entries in data.items():
        if not isinstance(entries, list):
            continue
        pairs = []
        for entry in entries:
            if isinstance(entry, Mapping):
                first, second = (_nvarc_grid(entry.get(name)) for name in ATTEMPTS)
                pairs.append((first, second))
            else:
                pairs.append((None, None))
        out[str(task_id)] = pairs
    return out


def load_nvarc(path: str | os.PathLike[str]) -> dict[str, list[NvarcPair]]:
    """Read NVARC's attempts from a ``submission.json`` file.

    Args:
        path: The file.

    Returns:
        As :func:`nvarc_from_submission`.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not JSON or not a JSON object.
    """
    return nvarc_from_submission(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class _Group:
    grid: Grid
    first: int
    attempts: set[str] = field(default_factory=set)
    size: int = UNPARSABLE_SIZE
    demoted: bool = True
    accuracy: float = 0.0
    passed: int = 0


def _mean_accuracy(verification: Verification) -> float:
    pairs = verification.pairs
    return sum(p.cell_accuracy for p in pairs) / len(pairs) if pairs else 0.0


def _program_groups(
    records: Sequence[AttemptRecord], test_index: int, verified: bool
) -> list[_Group]:
    """Group programs by their output on one test input: verified ones, or unverified ones."""
    groups: dict[str, _Group] = {}
    sizes: dict[str, int] = {}
    seen = 0
    for record in records:
        for v in record.verifications:
            seen += 1
            if v.verified != verified or any(g.fatal for g in v.guards):
                continue
            if test_index >= len(v.test_outputs) or v.test_outputs[test_index] is None:
                continue
            grid = _valid(v.test_outputs[test_index])
            if grid is None:
                continue
            group = groups.setdefault(_key(grid), _Group(grid, seen))
            group.attempts.add(record.attempt_id)
            group.size = min(group.size, sizes.setdefault(v.source, program_size(v.source)))
            group.demoted = group.demoted and any(not g.fatal for g in v.guards)
            group.accuracy = max(group.accuracy, _mean_accuracy(v))
            group.passed = max(group.passed, v.passed)
    if verified:
        return sorted(groups.values(), key=lambda g: (-len(g.attempts), g.size, g.demoted, g.first))
    return sorted(groups.values(), key=lambda g: (-g.accuracy, -g.passed, g.size, g.first))


def _direct_groups(records: Sequence[AttemptRecord], test_index: int) -> list[_Group]:
    """Group the direct answers for one test input."""
    groups: dict[str, _Group] = {}
    seen = 0
    for record in records:
        for index, answer in record.direct_answers:
            seen += 1
            grid = _valid(answer) if index == test_index else None
            if grid is not None:
                groups.setdefault(_key(grid), _Group(grid, seen)).attempts.add(record.attempt_id)
    return sorted(groups.values(), key=lambda g: (-len(g.attempts), g.first))


def rank_candidates(
    task_id: str,
    test_index: int,
    records: Sequence[AttemptRecord],
    nvarc: Sequence[Grid | None] = (),
) -> list[Candidate]:
    """Return every distinct candidate for one test input, best first.

    Args:
        task_id: The task id; records of other tasks are ignored.
        test_index: The test input's index.
        records: Attempt records (any phases).
        nvarc: NVARC's ``(attempt_1, attempt_2)`` for this test input, None where missing; empty
            when NVARC did not run. A ``[[0]]`` grid counts as missing (NVARC's placeholder).

    Returns:
        The candidates in the order of the module's rules, one per distinct grid. A verified
        group is a ``"program"`` candidate with its votes; an unverified program's output is a
        ``"program"`` candidate with 0 votes; NVARC's carry 1 vote.
    """
    own = [r for r in records if r.task_id == task_id]
    groups = _program_groups(own, test_index, verified=True)
    nvarc_grids = [_nvarc_grid(g) for g in list(nvarc)[:2]]
    nvarc_grids += [None] * (2 - len(nvarc_grids))
    top = nvarc_grids[0]
    if top is not None:
        match = next((g for g in groups if g.grid == top), None)
        if match is not None:
            groups.remove(match)
            groups.insert(0, match)

    def nvarc_candidates(grids: Sequence[Grid | None]) -> list[Candidate]:
        return [
            Candidate(task_id, test_index, g, "nvarc", 1, 0, False) for g in grids if g is not None
        ]

    verified = [
        Candidate(task_id, test_index, g.grid, "program", len(g.attempts), g.size, g.demoted)
        for g in groups
    ]
    directs = [
        Candidate(task_id, test_index, g.grid, "direct", len(g.attempts), 0, False)
        for g in _direct_groups(own, test_index)
    ]
    unverified = [
        Candidate(task_id, test_index, g.grid, "program", 0, g.size, g.demoted)
        for g in _program_groups(own, test_index, verified=False)
    ]
    if verified:
        ranked = (
            verified
            + nvarc_candidates(nvarc_grids[:1])
            + directs
            + unverified
            + nvarc_candidates(nvarc_grids[1:])
        )
    else:
        ranked = nvarc_candidates(nvarc_grids) + directs + unverified
    out: list[Candidate] = []
    seen: set[str] = set()
    for candidate in ranked:
        key = _key(candidate.grid)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def build_selection(
    test_counts: Mapping[str, int],
    records: Iterable[AttemptRecord],
    nvarc: Mapping[str, Sequence[NvarcPair]] | None = None,
) -> dict[str, list[list[Candidate]]]:
    """Choose up to two candidates for every test input of every task.

    Args:
        test_counts: Task id to its number of test inputs (the challenges file's tasks).
        records: Attempt records; records of other tasks are ignored.
        nvarc: NVARC's attempts per task (:func:`load_nvarc`), or None.

    Returns:
        Task id to one list per test input of at most two candidates, attempt 1 first.
    """
    by_task: dict[str, list[AttemptRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)
    selection = {}
    for task_id, count in test_counts.items():
        pairs = (nvarc or {}).get(task_id, ())
        own = by_task.get(task_id, [])
        selection[task_id] = [
            rank_candidates(task_id, j, own, pairs[j] if j < len(pairs) else ())[:2]
            for j in range(count)
        ]
    return selection


def submission_from_selection(
    selection: Mapping[str, Sequence[Sequence[Candidate]]],
) -> dict[str, list[dict[str, Grid]]]:
    """Turn chosen candidates into Kaggle's submission format.

    Args:
        selection: As :func:`build_selection` returns it.

    Returns:
        Task id to one ``{"attempt_1", "attempt_2"}`` per test input; ``[[0]]`` fills a slot
        without a candidate.
    """
    submission = {}
    for task_id, entries in selection.items():
        rows = []
        for chosen in entries:
            grids = [c.grid for c in chosen[:2]]
            grids += [fallback_grid() for _ in range(2 - len(grids))]
            rows.append(dict(zip(ATTEMPTS, grids, strict=True)))
        submission[task_id] = rows
    return submission


def build_submission(
    test_counts: Mapping[str, int],
    records: Iterable[AttemptRecord],
    nvarc: Mapping[str, Sequence[NvarcPair]] | None = None,
) -> dict[str, list[dict[str, Grid]]]:
    """Choose the attempts of every task and return the submission.

    Args:
        test_counts: Task id to its number of test inputs.
        records: Attempt records.
        nvarc: NVARC's attempts per task, or None.

    Returns:
        A submission with an entry for every test input of every task.
    """
    return submission_from_selection(build_selection(test_counts, records, nvarc))


def write_submission(
    path: str | os.PathLike[str],
    submission: Mapping[str, Any],
    test_counts: Mapping[str, int],
) -> None:
    """Validate a submission against the task list and write it atomically.

    Args:
        path: The file, ``/kaggle/working/submission.json`` on Kaggle.
        submission: The submission.
        test_counts: Task id to its number of test inputs.

    Raises:
        ValueError: If the submission is invalid for these tasks (``arc26.submit.validate``).
        OSError: If the file cannot be written.
    """
    challenges = {task_id: {"test": [{}] * count} for task_id, count in test_counts.items()}
    problems = validate(dict(submission), challenges)
    if problems:
        raise ValueError(f"invalid submission: {problems[:5]}")
    write_json_atomic(path, submission)


def select_files(
    records_path: str | os.PathLike[str] | None,
    challenges_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    nvarc_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Write ``submission.json`` from an attempts JSONL, the challenges, and NVARC's submission.

    Args:
        records_path: The JSONL of attempt records, or None (no agent records).
        challenges_path: The challenges file (task ids and test counts only are read).
        out_path: The submission to write.
        nvarc_path: NVARC's ``submission.json``, or None.

    Returns:
        A report: ``tasks``, ``test_inputs``, ``records``, ``skipped_lines``, and ``slots``,
        the number of attempts filled from each source (``program``, ``direct``, ``nvarc``,
        ``empty``).

    Raises:
        OSError: If a file cannot be read or written.
        ValueError: If the challenges or NVARC's file is malformed.
    """
    counts = entry_counts(json.loads(Path(challenges_path).read_text(encoding="utf-8")))
    records: list[AttemptRecord] = []
    skipped = 0
    if records_path is not None:
        records, skipped = read_records(records_path)
    nvarc = None if nvarc_path is None else load_nvarc(nvarc_path)
    selection = build_selection(counts, records, nvarc)
    write_submission(out_path, submission_from_selection(selection), counts)
    slots = {"program": 0, "direct": 0, "nvarc": 0, "empty": 0}
    for entries in selection.values():
        for chosen in entries:
            for candidate in chosen:
                slots[candidate.source] += 1
            slots["empty"] += 2 - len(chosen)
    return {
        "tasks": len(counts),
        "test_inputs": sum(counts.values()),
        "records": len(records),
        "skipped_lines": skipped,
        "slots": slots,
    }
