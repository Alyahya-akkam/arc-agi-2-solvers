"""Load agent tasks from an ARC challenges file (P6.2).

Why: the agent solver sees demonstrations and test inputs only (:class:`AgentTask`), so this
loader is its one entry point for task data. It reads the Kaggle challenges format, one JSON
object ``{task_id: {"train": [{"input", "output"}, ...], "test": [{"input"}, ...]}}``, validates
every grid, and never reads a solutions file or a test output: a ``"output"`` key inside a test
entry is ignored, and a file whose name contains ``solution`` is refused.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from arc26.agent.types import AgentTask, Pair
from arc26.data import Grid
from arc26.submit import grid_problem


def load_tasks(
    path: str | os.PathLike[str], ids: Iterable[str] | None = None
) -> tuple[AgentTask, ...]:
    """Read the tasks of a challenges file.

    Args:
        path: A challenges JSON in the Kaggle format, for example
            ``arc-agi_evaluation_challenges.json``.
        ids: Task ids to keep, or None for every task.

    Returns:
        The tasks in file order (filtered by ``ids`` when given).

    Raises:
        ValueError: If the file name names a solutions file, the JSON is not in the challenges
            format, a grid is invalid, or an id in ``ids`` is not in the file.
        OSError: If the file cannot be read.
    """
    file = Path(path)
    if "solution" in file.name.lower():
        raise ValueError(f"refusing to read {file.name}: the agent never reads solutions")
    challenges = json.loads(file.read_text(encoding="utf-8"))
    return tasks_from_challenges(challenges, ids)


def tasks_from_challenges(
    challenges: Mapping[str, Any], ids: Iterable[str] | None = None
) -> tuple[AgentTask, ...]:
    """Convert a parsed challenges object into tasks.

    Args:
        challenges: Task id to ``{"train": [...], "test": [...]}``.
        ids: Task ids to keep, or None for every task.

    Returns:
        The tasks in the object's order (filtered by ``ids`` when given).

    Raises:
        ValueError: If ``challenges`` is not a JSON object of tasks, a task is malformed, or an
            id in ``ids`` is missing.
    """
    if not isinstance(challenges, Mapping):
        raise ValueError(f"challenges must be a JSON object; got {type(challenges).__name__}")
    wanted = None if ids is None else set(ids)
    if wanted is not None:
        missing = sorted(wanted - set(challenges))
        if missing:
            raise ValueError(f"task ids not in the challenges: {missing}")
    return tuple(
        task_from_json(task_id, content)
        for task_id, content in challenges.items()
        if wanted is None or task_id in wanted
    )


def task_from_json(task_id: str, content: Any) -> AgentTask:
    """Build one task from its challenges entry.

    Args:
        task_id: The task id.
        content: ``{"train": [{"input", "output"}, ...], "test": [{"input"}, ...]}``; any test
            output present is ignored.

    Returns:
        The task, with grids copied into plain lists.

    Raises:
        ValueError: If the entry has no demonstration or no test input, or a grid is invalid.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"task id must be a non-empty string; got {task_id!r}")
    if not isinstance(content, Mapping):
        raise ValueError(f"{task_id}: entry is not an object")
    train = content.get("train")
    test = content.get("test")
    if not isinstance(train, list) or not train:
        raise ValueError(f"{task_id}: no list of demonstration pairs")
    if not isinstance(test, list) or not test:
        raise ValueError(f"{task_id}: no list of test inputs")
    pairs = []
    for i, pair in enumerate(train):
        if not isinstance(pair, Mapping):
            raise ValueError(f"{task_id}: train[{i}] is not an object")
        pairs.append(
            Pair(
                _grid(pair.get("input"), f"{task_id}: train[{i}].input"),
                _grid(pair.get("output"), f"{task_id}: train[{i}].output"),
            )
        )
    inputs = []
    for i, entry in enumerate(test):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{task_id}: test[{i}] is not an object")
        inputs.append(_grid(entry.get("input"), f"{task_id}: test[{i}].input"))
    return AgentTask(task_id, tuple(pairs), tuple(inputs))


def count_test_inputs(tasks: Iterable[AgentTask]) -> dict[str, int]:
    """Return the number of test inputs of each task, the entries its submission needs.

    Args:
        tasks: The tasks.

    Returns:
        Task id to its number of test inputs, in the given order.
    """
    return {task.task_id: len(task.test_inputs) for task in tasks}


def _grid(value: Any, where: str) -> Grid:
    """A validated copy of one grid; ``where`` names it in the error."""
    problem = grid_problem(value)
    if problem is not None:
        raise ValueError(f"{where}: {problem}")
    return [list(row) for row in value]
