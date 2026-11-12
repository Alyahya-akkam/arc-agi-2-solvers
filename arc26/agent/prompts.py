"""Prompts of the agent loop: the shared system prompt and the task as text (P6.2).

Why: the system prompt is one fixed prefix for every task, so the engine's prefix cache holds it
once (``docs/research/agent_library_design.md``, section 5.2). It states the rules, the REPL's
API, the ``transform`` contract (a self-contained cell that may import numpy and
``arc26.agent.lib``), the direct-answer format, and the helper library's index when
``arc26.agent.lib`` is importable. :data:`DIRECT_SYSTEM_PROMPT` is the direct-answer arm's
(run ``P6.3-04``): no REPL, the test outputs as answer blocks. The task follows in the first
user message: grids as rows of digits, sizes as rows x columns, indices from 0 as in the REPL.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

from arc26.agent.feedback import grid_text, shape_text, verification_report
from arc26.agent.types import AgentTask, Message, Verification

LIB_MODULE = "arc26.agent.lib"
# Earlier programs shown in a later phase's prompt (design note 5.2: "at most 2").
MAX_PRIOR = 2

SYSTEM_PROMPT = """\
You solve ARC-AGI tasks by writing Python programs. A task shows training examples, each an \
input grid and its output grid, and one or more test inputs. Find the rule that turns every \
training input into its output and write it as a function transform(grid).

Grids have 1 to 30 rows and 1 to 30 columns; each cell is a color from 0 to 9. Here a grid is \
written as rows of digits, one row per line, and its size as rows x columns. Indices count from 0.

You work in a persistent Python REPL. Put code in ```python blocks: every block in your reply \
runs in order, and the next message shows what it printed (shortened when long). The REPL \
already holds:
- train: a list of (input, output) pairs of numpy integer arrays
- test_inputs: a list of numpy integer arrays
- np: numpy
- lib: the helper library arc26.agent.lib
Use it to inspect the grids and to test ideas.

The program:
- A block that defines transform is also checked in a fresh, isolated process on every training \
input and every test input. Only its imports, function and class definitions, and assignments \
that do not use train, test_inputs or transform are checked; the rest of the block (prints, \
loops, tests) runs only in the REPL.
- It must be self-contained: it may import numpy, scipy, math, itertools, collections, \
functools, copy, typing and arc26.agent.lib, nothing else, and it cannot see train, \
test_inputs, files or the network.
- transform(grid) gets one input grid as a 2-D numpy array of ints (int64) and returns the \
output grid as a 2-D numpy array or a list of lists of ints 0 to 9, within 2 seconds per call.
- It must implement a general rule. Programs that store training outputs, compare the input \
with stored grids, or branch on the sizes of particular examples are rejected or ranked last.
- The check reports how many training examples the program reproduces exactly, the shape, cell \
accuracy and first mismatching rows of each failing example, and its outputs on the test inputs. \
A program that reproduces every training example and returns a grid for every test input is \
accepted, and the task ends.

If no program works, you may give your best guess for a test output directly, as a block like:
```answer
{"test_index": 0, "grid": [[0, 1], [1, 0]]}
```
A direct answer is a last resort; an accepted program is worth more.

You have a limited number of replies and tokens: keep your reasoning focused and send code \
early."""


DIRECT_SYSTEM_PROMPT = """\
You solve ARC-AGI tasks by giving the output grids directly. A task shows training examples, \
each an input grid and its output grid, and one or more test inputs. Find the rule that turns \
every training input into its output, apply it to each test input, and write the resulting \
output grid.

Grids have 1 to 30 rows and 1 to 30 columns; each cell is a color from 0 to 9. Here a grid is \
written as rows of digits, one row per line, and its size as rows x columns. Indices count from 0.

There is no code interpreter: reason, then answer. Give exactly one answer block per test \
input, as JSON with the test input's index and the output grid as a list of rows:
```answer
{"test_index": 0, "grid": [[0, 1], [1, 0]]}
```
Write nothing after the answer blocks. Check the output's size and every row before you \
answer."""


# Run P6.3-05: up to two answers per test input in one reply, the two attempts the competition
# scores (pass@2), best first.
DIRECT_TWO_SYSTEM_PROMPT = DIRECT_SYSTEM_PROMPT.replace(
    "Give exactly one answer block per test input,",
    "Give one or two answer blocks per test input, your best answer first (two different grids "
    "are two attempts; either may be right),",
)


def helper_index() -> str | None:
    """Return the helper library's index for the prompt, or None when the library is absent.

    Why: the library (P6.1) is optional for this module; without it the prompt simply omits the
    index. A library that exists but fails to import is a bug and is not hidden.

    Returns:
        ``arc26.agent.lib.index_text()``, or None if ``arc26.agent.lib`` does not exist or has
        no ``index_text``.

    Raises:
        ImportError: If the library exists but one of its own imports fails.
    """
    try:
        lib = importlib.import_module(LIB_MODULE)
    except ModuleNotFoundError as err:
        if err.name in {LIB_MODULE, "arc26.agent"}:
            return None
        raise
    index_text = getattr(lib, "index_text", None)
    return index_text() if callable(index_text) else None


def system_prompt(index: str | None = None, *, with_index: bool = True) -> str:
    """Return the shared system prompt.

    Args:
        index: The helper index to show; None reads it with :func:`helper_index`.
        with_index: False leaves the helper index out (the no-library arm of an ablation).

    Returns:
        :data:`SYSTEM_PROMPT`, followed by the helper library's index when there is one.
    """
    if not with_index:
        return SYSTEM_PROMPT
    text = helper_index() if index is None else index
    if not text:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n\nHelper library (in programs: import arc26.agent.lib as lib; in the "
        f"REPL: lib):\n{text.strip()}"
    )


def task_text(task: AgentTask) -> str:
    """Render a task's demonstrations and test inputs as text.

    Args:
        task: The task.

    Returns:
        A header with the counts, then each example's input and output and each test input,
        every grid as rows of digits under a label with its size.
    """
    tests = len(task.test_inputs)
    parts = [
        f"Task {task.task_id}: {len(task.train)} training examples, {tests} test "
        f"input{'s' if tests != 1 else ''}."
    ]
    for i, pair in enumerate(task.train):
        parts.append(
            f"Example {i} input ({shape_text(pair.input)}):\n{grid_text(pair.input)}\n"
            f"Example {i} output ({shape_text(pair.output)}):\n{grid_text(pair.output)}"
        )
    for j, grid in enumerate(task.test_inputs):
        parts.append(f"Test input {j} ({shape_text(grid)}):\n{grid_text(grid)}")
    return "\n\n".join(parts)


def task_message(task: AgentTask, prior: Sequence[Verification] = ()) -> str:
    """Return the first user message of an attempt.

    Args:
        task: The task.
        prior: Earlier programs for this task that were not accepted, newest last; at most
            :data:`MAX_PRIOR` of them (the last ones) are shown with their reports, so a later
            phase starts from what failed.

    Returns:
        The task text, then the earlier programs if any.
    """
    text = task_text(task)
    shown = list(prior)[-MAX_PRIOR:]
    if not shown:
        return text
    parts = [text, "Earlier programs for this task (not accepted) and their checks:"]
    for verification in shown:
        parts.append(
            f"```python\n{verification.source.rstrip()}\n```\n"
            f"{verification_report(task, verification)}"
        )
    return "\n\n".join(parts)


def initial_messages(
    task: AgentTask, *, system: str | None = None, prior: Sequence[Verification] = ()
) -> list[Message]:
    """Return the messages that open an attempt.

    Args:
        task: The task.
        system: The system prompt; None builds it with :func:`system_prompt`.
        prior: Earlier programs to show, as in :func:`task_message`.

    Returns:
        The system message and the task's user message.
    """
    return [
        Message("system", system_prompt() if system is None else system),
        Message("user", task_message(task, prior)),
    ]
