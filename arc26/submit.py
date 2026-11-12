"""Validate, repair, and safely write the competition's ``submission.json`` (BRIEF A5, B5 P0.5).

Why: a scored rerun that ends without a valid ``/kaggle/working/submission.json`` scores nothing,
and the reference writes that file once, at the very end, with a plain ``open`` and ``json.dump``.
``validate`` lists every way a submission breaks the format. ``repair`` keeps every valid attempt
and replaces anything invalid or missing with ``[[0]]``, the reference's own placeholder
(``ArcDataset.get_submission``). ``finalize_submission`` is the crash-safe writer the notebook
calls in ``finally``: it writes the repaired submission atomically, so a reader sees the old file
or the new one, never a partial one.

Format (the competition's data tab, as copied in ``docs/research/kaggle_first_runs.md``, section
2): a JSON object with one key per task id of the challenges file. Each value is a list with one
entry per test input of that task, in order, and each entry maps ``attempt_1`` and ``attempt_2``
to a grid: a rectangular list of rows of integers 0 to 9, with both sides from 1 to 30.

This module imports only the standard library and nothing from ``arc26``, and has no
``__future__`` import: ``scripts/build_kaggle_notebook.py`` copies its source verbatim into the
notebook, so the validator runs before, and without, the package install.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ATTEMPTS = ("attempt_1", "attempt_2")
MAX_SIDE = 30
MAX_COLOR = 9
# A finalize report lists at most this many problems; its problem count is always complete.
MAX_LISTED = 20


def fallback_grid() -> list[list[int]]:
    """Return a new ``[[0]]``, the reference's placeholder for an attempt without a prediction.

    Returns:
        A fresh one-cell grid, so no two attempts share a list.
    """
    return [[0]]


def grid_problem(grid: Any) -> str | None:
    """Return why ``grid`` is not a valid ARC grid, or None when it is valid.

    Why: the scorer compares attempts to the solution cell by cell, so an attempt must be a
    rectangular list of rows of plain integers 0 to 9 with sides 1 to 30. Booleans are refused
    although Python counts them as integers: JSON writes them as ``true`` and ``false``.

    Args:
        grid: One attempt, as parsed from JSON.

    Returns:
        A short reason, for example ``"ragged: row 2 has 3 cells, row 0 has 4"``, or None.
    """
    if not isinstance(grid, list):
        return f"not a list of rows ({type(grid).__name__})"
    if not 1 <= len(grid) <= MAX_SIDE:
        return f"{len(grid)} rows, expected 1 to {MAX_SIDE}"
    width = None
    for r, row in enumerate(grid):
        if not isinstance(row, list):
            return f"row {r} is not a list ({type(row).__name__})"
        if width is None:
            width = len(row)
            if not 1 <= width <= MAX_SIDE:
                return f"{width} columns, expected 1 to {MAX_SIDE}"
        elif len(row) != width:
            return f"ragged: row {r} has {len(row)} cells, row 0 has {width}"
        for c, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                return f"value {value!r} at row {r}, column {c} is not an integer"
            if not 0 <= value <= MAX_COLOR:
                return f"value {value} at row {r}, column {c} is outside 0 to {MAX_COLOR}"
    return None


def entry_counts(challenges: Mapping[str, Any]) -> dict[str, int]:
    """Return the number of test inputs of every task, in the challenges file's order.

    Args:
        challenges: The parsed challenges file: task id to ``{"train": [...], "test": [...]}``.

    Returns:
        Task id to the number of entries its submission list must hold.

    Raises:
        ValueError: If ``challenges`` is not a mapping of task ids to tasks with a non-empty
            ``test`` list.
    """
    if not isinstance(challenges, Mapping):
        raise ValueError(f"challenges must be a JSON object; got {type(challenges).__name__}")
    counts = {}
    for task, content in challenges.items():
        tests = content.get("test") if isinstance(content, Mapping) else None
        if not isinstance(task, str) or not isinstance(tests, list) or not tests:
            raise ValueError(f"challenges task {task!r} has no list of test inputs")
        counts[task] = len(tests)
    return counts


def validate(submission: Any, challenges: Mapping[str, Any]) -> list[str]:
    """List every way ``submission`` breaks the competition format for ``challenges``.

    Why: the notebook validates what it writes, and the tests use the same check, so "valid" means
    one thing everywhere: every task id present and no other; one entry per test input; each entry
    holding exactly ``attempt_1`` and ``attempt_2``; every attempt a valid grid (``grid_problem``).

    Args:
        submission: The parsed submission.
        challenges: The parsed challenges file the submission answers.

    Returns:
        One message per problem, in the challenges file's task order; empty when valid.

    Raises:
        ValueError: If ``challenges`` is malformed (``entry_counts``).
    """
    counts = entry_counts(challenges)
    if not isinstance(submission, dict):
        return [f"submission is not a JSON object ({type(submission).__name__})"]
    problems = []
    for task, count in counts.items():
        if task not in submission:
            problems.append(f"{task}: missing")
            continue
        entries = submission[task]
        if not isinstance(entries, list):
            problems.append(f"{task}: not a list of test entries ({type(entries).__name__})")
            continue
        if len(entries) != count:
            problems.append(f"{task}: {len(entries)} test entries, the challenges have {count}")
        for i, entry in enumerate(entries[:count]):
            where = f"{task}[{i}]"
            if not isinstance(entry, dict):
                problems.append(f"{where}: not an object ({type(entry).__name__})")
                continue
            for attempt in ATTEMPTS:
                if attempt not in entry:
                    problems.append(f"{where}: missing {attempt}")
                    continue
                reason = grid_problem(entry[attempt])
                if reason is not None:
                    problems.append(f"{where}.{attempt}: {reason}")
            extra = sorted(map(str, set(entry) - set(ATTEMPTS)))
            if extra:
                problems.append(f"{where}: unexpected keys {extra}")
    for task in submission:
        if task not in counts:
            problems.append(f"{task!r}: not a task of the challenges file")
    return problems


def _repaired(submission: Any, challenges: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    counts = entry_counts(challenges)
    source = submission if isinstance(submission, dict) else {}
    fixed = {}
    replaced = 0
    for task, count in counts.items():
        entries = source.get(task)
        entries = entries if isinstance(entries, list) else []
        rows = []
        for i in range(count):
            entry = entries[i] if i < len(entries) and isinstance(entries[i], dict) else {}
            attempts = {}
            for attempt in ATTEMPTS:
                grid = entry.get(attempt)
                if grid_problem(grid) is None:
                    attempts[attempt] = [list(row) for row in grid]
                else:
                    attempts[attempt] = fallback_grid()
                    replaced += 1
            rows.append(attempts)
        fixed[task] = rows
    return fixed, replaced


def repair(submission: Any, challenges: Mapping[str, Any]) -> dict[str, Any]:
    """Return a valid submission that keeps every valid attempt of ``submission``.

    Why: a failure in one task, or a truncated write, must not cost the attempts that are fine.
    Each attempt is judged on its own: a valid one is copied, and a missing or invalid one becomes
    ``[[0]]``. Missing tasks and entries are added with ``[[0]]`` attempts; unknown tasks, surplus
    entries, and keys other than the two attempts are dropped. Anything that is not a JSON object,
    including None, repairs to all ``[[0]]``.

    Args:
        submission: The parsed submission, or None when there is none.
        challenges: The parsed challenges file, which fixes the task ids and entry counts.

    Returns:
        A new submission, in the challenges file's task order, that ``validate`` accepts.

    Raises:
        ValueError: If ``challenges`` is malformed (``entry_counts``).
    """
    return _repaired(submission, challenges)[0]


def write_json_atomic(path: str | os.PathLike[str], data: Any) -> None:
    """Write ``data`` as JSON so a reader sees the old file or the new one, never a partial one.

    Why: Kaggle scores whatever ``submission.json`` holds when the session ends, and a session
    killed during a plain write leaves a truncated file. The data goes to a temporary file in the
    same directory, is flushed to disk, and replaces ``path`` with one ``os.replace``. On failure
    the temporary file is removed and ``path`` is left as it was.

    Args:
        path: The file to write; its parent directory is created if needed.
        data: A JSON-serializable value.

    Raises:
        OSError: If the file cannot be written or replaced.
        TypeError: If ``data`` is not JSON-serializable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_submission(
    out_path: str | os.PathLike[str],
    challenges_path: str | os.PathLike[str],
    raw_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Write a valid submission to ``out_path`` from whatever ``raw_path`` holds.

    Why: this is the crash-safe writer. The notebook calls it once before the solver starts, with
    no raw file, so a valid all-``[[0]]`` file exists even if the session is killed, and again in
    ``finally`` after ``make_submission``, whatever happened to it. Every valid attempt of the raw
    file is kept (``repair``). The file is written with ``write_json_atomic``, then read back and
    validated before the function returns.

    Args:
        out_path: The submission to write, ``/kaggle/working/submission.json`` on Kaggle.
        challenges_path: The challenges file the submission answers.
        raw_path: The submission as ``make_submission`` wrote it, or None when there is none yet.
            A missing, unreadable, or invalid raw file is reported, not raised.

    Returns:
        A report with ``out``; ``raw`` and ``raw_status`` (``none``, ``missing``,
        ``unreadable``, or ``read``); ``tasks`` and ``test_inputs``; ``problem_count`` and
        ``problems`` (the raw file's problems, at most ``MAX_LISTED``); ``replaced_attempts``
        (attempts that became ``[[0]]`` because they were missing or invalid); ``zero_attempts``
        (attempts equal to ``[[0]]`` in the written file); and ``valid`` (the written file passed
        ``validate``).

    Raises:
        OSError: If the challenges file cannot be read or ``out_path`` cannot be written.
        ValueError: If the challenges file is not JSON in the challenges format.
    """
    challenges = json.loads(Path(challenges_path).read_text(encoding="utf-8"))
    counts = entry_counts(challenges)
    raw = None
    problems = []
    if raw_path is None:
        status = "none"
    else:
        try:
            raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            status = "missing"
            problems.append(f"raw submission {raw_path} does not exist")
        except (OSError, ValueError) as err:
            status = "unreadable"
            problems.append(f"raw submission {raw_path} is unreadable: {err}")
        else:
            status = "read"
            problems.extend(validate(raw, challenges))
    fixed, replaced = _repaired(raw, challenges)
    write_json_atomic(out_path, fixed)
    written = json.loads(Path(out_path).read_text(encoding="utf-8"))
    zeros = sum(
        entry[attempt] == [[0]]
        for entries in written.values()
        for entry in entries
        for attempt in ATTEMPTS
    )
    return {
        "out": str(out_path),
        "raw": None if raw_path is None else str(raw_path),
        "raw_status": status,
        "tasks": len(counts),
        "test_inputs": sum(counts.values()),
        "problem_count": len(problems),
        "problems": problems[:MAX_LISTED],
        "replaced_attempts": replaced,
        "zero_attempts": zeros,
        "valid": not validate(written, challenges),
    }
