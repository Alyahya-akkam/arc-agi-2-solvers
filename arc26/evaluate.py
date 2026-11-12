"""Competition metric and run agreement on prediction files (BRIEF B4, task P0.2).

Score: test output :math:`i` of task :math:`t` gets two attempts, and :math:`c_{t,i} = 1` when
either equals the solution grid exactly. A task scores the mean of :math:`c_{t,i}` over its
:math:`n_t` test outputs; a subset scores the sum over its tasks, reported with the task count
(the organizers' per-task metric, as the reference notebook's ``validate_submission`` computes
it). The per-output variant, the mean of :math:`c_{t,i}` over all test outputs of the subset, is
reported next to it (``docs/STATE.md``, Notes).

Agreement (``docs/prerun/P0.1-01.md``, section 1): :math:`g(X, A)` is the share of the
reference's test outputs whose two attempt grids are identical in X and A, in order; the
``attempt_1`` agreement compares the first attempts only.

Missing tasks and malformed attempts score 0 and never raise. An attempt is malformed unless
:func:`is_grid` holds; a malformed attempt matches nothing, not even an identical one.

CLI::

    python -m arc26.evaluate score <submission> <solutions> [--subset DEV4|DEV40|DEV120]
    python -m arc26.evaluate agree <submission> <reference> [--subset DEV4|DEV40|DEV120]

A file whose top-level object has a ``predictions`` object (the P0.1 parity fixture) is read as
that object. Exit status 2 on an unreadable or malformed input file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from arc26.data import SUBSETS, Grid, subset_ids

ATTEMPTS = ("attempt_1", "attempt_2")
MAX_SIDE = 30
COLORS = 10


def is_grid(value: object) -> bool:
    """Return whether a value is a well-formed ARC grid.

    Why: attempts come from decoded model output and from files, so anything else must score 0
    instead of raising. Cells must be ``int`` exactly: ``True == 1`` in Python, but a boolean
    cell is not a color.

    Args:
        value: Any decoded JSON value.

    Returns:
        True for a list of 1 to 30 rows, all lists of the same length 1 to 30, whose cells are
        ints in 0..9.
    """
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SIDE:
        return False
    width = len(value[0]) if isinstance(value[0], list) else 0
    if not 1 <= width <= MAX_SIDE:
        return False
    return all(
        isinstance(row, list)
        and len(row) == width
        and all(type(cell) is int and 0 <= cell < COLORS for cell in row)
        for row in value
    )


def _same(first: object, second: object) -> bool:
    return is_grid(first) and is_grid(second) and first == second


def _attempts(predictions: Mapping[str, object], task_id: str, index: int) -> tuple[object, ...]:
    """The two attempts of one test output; None for each one that is absent."""
    outputs = predictions.get(task_id)
    if isinstance(outputs, list) and index < len(outputs) and isinstance(outputs[index], Mapping):
        return tuple(outputs[index].get(name) for name in ATTEMPTS)
    return (None,) * len(ATTEMPTS)


@dataclass(frozen=True)
class TaskScore:
    """Score of one task.

    Attributes:
        task_id: ARC task id.
        correct: Per test output, in order, whether either attempt equals the solution.
        missing: The task is absent from the predictions.
        malformed: Attempts of the task's test outputs that are not well-formed grids, absent
            ones included; 0 for a missing task.
    """

    task_id: str
    correct: tuple[bool, ...]
    missing: bool = False
    malformed: int = 0

    @property
    def outputs(self) -> int:
        """Number of test outputs."""
        return len(self.correct)

    @property
    def solved(self) -> int:
        """Number of test outputs that either attempt solved."""
        return sum(self.correct)

    @property
    def score(self) -> float:
        """Mean of :math:`c_{t,i}` over the task's test outputs, in [0, 1]."""
        return self.solved / self.outputs if self.outputs else 0.0


@dataclass(frozen=True)
class SubsetScore:
    """Score of a set of tasks in both variants, always with its counts.

    Attributes:
        label: Subset name (``DEV4``, ``DEV40``, ``DEV120``) or ``all``.
        per_task: One score per task, in task-id order.
    """

    label: str
    per_task: tuple[TaskScore, ...]

    @property
    def tasks(self) -> int:
        """Number of tasks scored."""
        return len(self.per_task)

    @property
    def test_outputs(self) -> int:
        """Number of test outputs over the scored tasks."""
        return sum(task.outputs for task in self.per_task)

    @property
    def score(self) -> float:
        """Per-task metric in tasks: the sum of the task scores (for example 2.5 of 4)."""
        return sum(task.score for task in self.per_task)

    @property
    def percent(self) -> float:
        """Per-task metric as a percent of the task count."""
        return 100.0 * self.score / self.tasks if self.tasks else 0.0

    @property
    def solved_outputs(self) -> int:
        """Number of test outputs solved over the scored tasks."""
        return sum(task.solved for task in self.per_task)

    @property
    def output_percent(self) -> float:
        """Per-output variant: percent of all test outputs solved."""
        return 100.0 * self.solved_outputs / self.test_outputs if self.test_outputs else 0.0

    @property
    def missing_tasks(self) -> tuple[str, ...]:
        """Scored tasks absent from the predictions."""
        return tuple(task.task_id for task in self.per_task if task.missing)

    @property
    def malformed(self) -> int:
        """Malformed or absent attempts over the tasks present in the predictions."""
        return sum(task.malformed for task in self.per_task)

    def report(self, per_task: bool = False) -> list[str]:
        """Return the score as text lines, each naming its counts.

        Args:
            per_task: Also list every task's score and solved outputs.

        Returns:
            The report lines.
        """
        lines = [
            f"score {self.label}: {self.tasks} tasks, {self.test_outputs} test outputs",
            f"task score: {_num(self.score)} of {self.tasks} tasks ({_num(self.percent, 2)}%)",
            f"output score: {self.solved_outputs} of {self.test_outputs} test outputs "
            f"({_num(self.output_percent, 2)}%)",
            f"missing tasks: {_listing(self.missing_tasks)}; malformed attempts: {self.malformed}",
        ]
        if per_task:
            for task in self.per_task:
                notes = " missing" if task.missing else ""
                notes += f" malformed={task.malformed}" if task.malformed else ""
                lines.append(
                    f"  {task.task_id} {_num(task.score)} ({task.solved}/{task.outputs}){notes}"
                )
        return lines


def score_task(
    task_id: str, predictions: Mapping[str, object], solutions: Sequence[Grid]
) -> TaskScore:
    """Score one task's predictions against its solution grids.

    Args:
        task_id: ARC task id.
        predictions: Submission-format predictions by task id.
        solutions: The task's output grids, one per test input, in order.

    Returns:
        The task's score; a task absent from ``predictions`` scores 0 and is marked missing.
    """
    if task_id not in predictions:
        return TaskScore(task_id, (False,) * len(solutions), missing=True)
    correct: list[bool] = []
    malformed = 0
    for index, solution in enumerate(solutions):
        attempts = _attempts(predictions, task_id, index)
        malformed += sum(not is_grid(attempt) for attempt in attempts)
        correct.append(any(_same(attempt, solution) for attempt in attempts))
    return TaskScore(task_id, tuple(correct), malformed=malformed)


def check_solutions(solutions: Mapping[str, object], task_ids: Sequence[str]) -> None:
    """Check that every task to score has a non-empty list of well-formed solution grids.

    Args:
        solutions: Solution grids by task id.
        task_ids: Tasks that will be scored.

    Raises:
        ValueError: If a task is absent from ``solutions`` or its entry is not a non-empty list
            of grids.
    """
    absent = [task_id for task_id in task_ids if task_id not in solutions]
    if absent:
        raise ValueError(f"missing from the solutions: {', '.join(absent)}")
    for task_id in task_ids:
        outputs = solutions[task_id]
        if not (isinstance(outputs, list) and outputs and all(map(is_grid, outputs))):
            raise ValueError(f"{task_id}: solutions must be a non-empty list of grids")


def score_submission(
    predictions: Mapping[str, object],
    solutions: Mapping[str, object],
    task_ids: Iterable[str] | None = None,
    label: str = "all",
) -> SubsetScore:
    """Score predictions against solutions over a set of tasks.

    Args:
        predictions: Submission-format predictions by task id.
        solutions: Output grids by task id (``arc-agi_evaluation_solutions.json``).
        task_ids: Tasks to score; every task of ``solutions`` when None.
        label: Name reported with the score.

    Returns:
        The per-task and per-output scores with their counts.

    Raises:
        ValueError: As :func:`check_solutions` raises it.
    """
    ids = sorted(solutions if task_ids is None else set(task_ids))
    check_solutions(solutions, ids)
    return SubsetScore(label, tuple(score_task(t, predictions, solutions[t]) for t in ids))


class OutputAgreement(NamedTuple):
    """Agreement on one test output of the reference.

    Attributes:
        task_id: ARC task id.
        index: Test output index within the task.
        attempt_1: The first attempts are identical grids.
        both: Both attempts are identical grids, in order.
    """

    task_id: str
    index: int
    attempt_1: bool
    both: bool


@dataclass(frozen=True)
class Agreement:
    """Agreement of predictions X with reference predictions A over A's test outputs.

    Attributes:
        label: Subset name (``DEV4``, ``DEV40``, ``DEV120``) or ``all``.
        outputs: One entry per test output of the compared tasks, in task-id order.
        missing_in_predictions: Compared tasks absent from X; their outputs count as disagreeing.
        missing_in_reference: Requested tasks absent from A (or with no outputs); not compared.
    """

    label: str
    outputs: tuple[OutputAgreement, ...]
    missing_in_predictions: tuple[str, ...] = ()
    missing_in_reference: tuple[str, ...] = ()

    @property
    def tasks(self) -> int:
        """Number of tasks compared."""
        return len({output.task_id for output in self.outputs})

    @property
    def test_outputs(self) -> int:
        """:math:`N`, the number of the reference's test outputs compared."""
        return len(self.outputs)

    @property
    def g(self) -> float:
        """Share of test outputs whose two attempts are identical, in order."""
        return sum(output.both for output in self.outputs) / self.test_outputs

    @property
    def attempt_1(self) -> float:
        """Share of test outputs whose first attempts are identical (diagnostic)."""
        return sum(output.attempt_1 for output in self.outputs) / self.test_outputs

    def report(self, per_task: bool = False) -> list[str]:
        """Return the agreement as text lines, each naming its counts.

        Args:
            per_task: Also list every test output's agreement.

        Returns:
            The report lines.
        """
        both = sum(output.both for output in self.outputs)
        first = sum(output.attempt_1 for output in self.outputs)
        n = self.test_outputs
        lines = [
            f"agree {self.label}: {self.tasks} tasks, {n} test outputs",
            f"g = {_num(self.g)} ({both} of {n} test outputs, both attempts identical in order)",
            f"attempt_1 agreement = {_num(self.attempt_1)} ({first} of {n} test outputs)",
            f"missing in submission: {_listing(self.missing_in_predictions)}; "
            f"missing in reference: {_listing(self.missing_in_reference)}",
        ]
        if per_task:
            for output in self.outputs:
                lines.append(
                    f"  {output.task_id}[{output.index}] both={'yes' if output.both else 'no'}"
                    f" attempt_1={'yes' if output.attempt_1 else 'no'}"
                )
        return lines


def agreement(
    predictions: Mapping[str, object],
    reference: Mapping[str, object],
    task_ids: Iterable[str] | None = None,
    label: str = "all",
) -> Agreement:
    """Compare predictions X with reference predictions A, output by output.

    Args:
        predictions: Submission-format predictions X by task id.
        reference: Submission-format predictions A by task id; they define the compared test
            outputs.
        task_ids: Tasks to compare; every task of ``reference`` when None.
        label: Name reported with the result.

    Returns:
        The agreement, with :math:`g` and the ``attempt_1`` agreement over :math:`N` outputs.

    Raises:
        ValueError: If no test output of the reference is left to compare.
    """
    ids = sorted(reference if task_ids is None else set(task_ids))
    compared = [t for t in ids if isinstance(reference.get(t), list) and reference[t]]
    outputs: list[OutputAgreement] = []
    for task_id in compared:
        for index in range(len(reference[task_id])):
            x1, x2 = _attempts(predictions, task_id, index)
            a1, a2 = _attempts(reference, task_id, index)
            first = _same(x1, a1)
            outputs.append(OutputAgreement(task_id, index, first, first and _same(x2, a2)))
    if not outputs:
        raise ValueError(f"no reference test outputs to compare for {label}")
    return Agreement(
        label,
        tuple(outputs),
        missing_in_predictions=tuple(t for t in compared if t not in predictions),
        missing_in_reference=tuple(t for t in ids if t not in compared),
    )


def _read_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object keyed by task id")
    return data


def load_predictions(path: str | Path) -> dict[str, object]:
    """Read submission-format predictions from a JSON file.

    Why: the P0.1 parity fixture keeps run A's predictions under ``predictions`` next to its
    provenance, and ``agree`` takes it as the reference directly. Task ids are 8 hex digits, so
    a task can never be named ``predictions``.

    Args:
        path: A ``submission.json``, or a file with a top-level ``predictions`` object.

    Returns:
        Predictions by task id, unvalidated: scoring tolerates malformed entries.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not JSON or its predictions are not a JSON object.
    """
    data = _read_object(Path(path))
    predictions = data.get("predictions", data)
    if not isinstance(predictions, dict):
        raise ValueError(f"{path}: 'predictions' must be a JSON object keyed by task id")
    return predictions


def load_solutions(path: str | Path) -> dict[str, object]:
    """Read a solutions file (task id to its list of output grids).

    Args:
        path: For example ``arc-agi_evaluation_solutions.json`` (dev mode only, never in
            training or in competition mode; BRIEF A0).

    Returns:
        Solutions by task id; :func:`score_submission` checks the entries it scores.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not JSON or not a JSON object.
    """
    return _read_object(Path(path))


def _num(value: float, digits: int = 4) -> str:
    return str(round(value, digits))


def _listing(task_ids: Sequence[str]) -> str:
    return f"{len(task_ids)} ({', '.join(task_ids)})" if task_ids else "0"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``score`` or ``agree`` command and print its report.

    Args:
        argv: Command-line arguments without the program name; ``sys.argv[1:]`` when None.

    Returns:
        0 on success, 2 when an input file cannot be read or is malformed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m arc26.evaluate", description="Score or compare ARC prediction files."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    score_cmd = commands.add_parser("score", help="competition metric against a solutions file")
    score_cmd.add_argument("submission", type=Path)
    score_cmd.add_argument("solutions", type=Path)
    agree_cmd = commands.add_parser("agree", help="agreement g and attempt_1 agreement")
    agree_cmd.add_argument("submission", type=Path)
    agree_cmd.add_argument("reference", type=Path, help="a submission file or the P0.1 fixture")
    for command in (score_cmd, agree_cmd):
        command.add_argument("--subset", choices=tuple(SUBSETS), help="default: every task")
        command.add_argument("--per-task", action="store_true", help="list each task or output")
    args = parser.parse_args(argv)

    task_ids = subset_ids(args.subset) if args.subset else None
    label = args.subset or "all"
    try:
        predictions = load_predictions(args.submission)
        if args.command == "score":
            solutions = load_solutions(args.solutions)
            result: SubsetScore | Agreement = score_submission(
                predictions, solutions, task_ids, label
            )
        else:
            result = agreement(predictions, load_predictions(args.reference), task_ids, label)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print("\n".join(result.report(per_task=args.per_task)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
