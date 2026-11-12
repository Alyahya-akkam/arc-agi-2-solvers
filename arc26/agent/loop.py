"""The agent loop: one attempt on one task, and an asyncio runner over many tasks (P6.2).

An attempt is one conversation (``docs/research/agent_library_design.md``, section 5.2). Each
turn the model replies; the loop runs the reply's ```python blocks in the task's REPL, in order,
and feeds back what they printed; a block that defines ``transform`` is also verified in a fresh
sandbox (:class:`~arc26.agent.types.Verifier`), and its NVIDIA-style report goes back to the
model; an ```answer block records a direct grid. The attempt stops on the first verified
program, when its generated tokens are spent, after its reply limit, on an engine error, or at
the run's deadline. Every reply's prompt and generated tokens are counted.

Forcing turns (``Budget.forcing_tokens``; run ``P6.3-02``, revision 5): when a thinking reply
ends at its cap, the next request repeats the conversation, the cut reasoning kept when
``Budget.preserve_thinking`` is on, plus a short message asking for the program now, and turns
thinking off for that one call, so the model writes code at once. Every thinking reply leaves
room for its forcing turn in the attempt's tokens and in the context.

In-request forcing (``Budget.forcing_in_request``; task P6.9, option 2b (iii)): a thinking
reply's request carries the turn cap as the server's thinking budget and room for the forced
answer, so the server ends the reasoning at the cap (with ``Budget.thinking_end_note`` before
``</think>``) and the model answers in the same request, without the second request that
prefills the whole conversation again. Three fault paths keep the attempt answered and bounded:
a request the server refuses for its budget field (vLLM's HTTP 400 when its reasoning config is
off) is sent once more without it, and the attempt goes on with the separate forcing turn; a reply
that still ends at its cap without a readable answer (a server that ignored the budget) gets a
forcing turn with its cut reasoning left out of the context, so nothing long is prefilled again;
and a thinking request not answered within ``Budget.turn_cut_s`` (the client-side cut) is
dropped and followed by a forcing turn on the opening alone.

Direct answers are fenced JSON, not a REPL function, so they need nothing from the sandbox and
survive a REPL restart: a block with the info string ``answer`` holding
``{"test_index": j, "grid": [[...], ...]}``, or a bare grid meaning test input 0. Under
``Budget.lenient_answers`` (task P6.9, option 4b) a block the strict parser refuses is read once
more by :func:`parse_answer`'s lenient rules (rows written as digit strings, cells as one-digit
strings, digit-string rows whose brackets do not balance); a block the strict parser reads is
never read differently.

What is verified: the block's imports (also inside a ``try`` whose body only imports), function
and class definitions, and assignments whose value does not read ``train``, ``test_inputs`` or
``transform``. Everything else in the block (prints, loops, calls that test the program on
``train``) is exploration and runs only in the REPL, so a model that defines and tests
``transform`` in one block is not failed for it. What the verifier returns is converted to plain
Python values (:func:`plain`) before it is reported or recorded.

The runner :func:`run_tasks` keeps up to ``concurrency`` attempts in flight against one engine,
runs the blocking REPL and verifier calls in a thread pool, skips attempts not started by the
deadline, and appends each finished :class:`~arc26.agent.types.AttemptRecord` as one JSON line,
so a killed run keeps every finished attempt.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args

import numpy as np

from arc26.agent import feedback, prompts
from arc26.agent.types import (
    AgentTask,
    AttemptRecord,
    Completion,
    Engine,
    GuardFinding,
    Message,
    PairResult,
    Repl,
    StopReason,
    Verification,
    Verifier,
)
from arc26.data import Grid
from arc26.submit import grid_problem

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

CODE_TAGS = frozenset({"", "python", "py", "python3"})
ANSWER_TAG = "answer"
# Names that only the REPL defines, plus the program itself: an assignment whose value reads one
# of them is a test, not part of the program.
REPL_NAMES = frozenset({"train", "test_inputs", "transform"})
# Template tokens per message (role markers, empty think block) and per prompt (the effort
# instruction the chat template adds), for the upper bound on a prompt's tokens.
MESSAGE_SLACK = 16
PROMPT_SLACK = 64
# Template tokens around a reply resent with its reasoning (``<think>\n`` and ``\n</think>\n\n``):
# the preserve check measured a growth of the reasoning's count plus 4 (run P6.3-02, 12 Oct,
# preserve_check.json); 16 leaves room for merges at the texts' edges.
THINK_SLACK = 16
# Tokens added to the server's count of the opening prompt (Engine.count_prompt_tokens): the
# count renders the same messages with the served template, so it is the prompt's; 64 covers a
# template variable the count and the chat request could treat apart (Qwen3.8's effort
# instruction is about 25 to 35 tokens) and keeps the bound an upper bound.
COUNT_SLACK = 64
# A 30 by 30 grid is under 3,000 characters of JSON; a far longer answer block is not a grid.
ANSWER_CHARS = 100_000

_FENCE = re.compile(r"^[ \t]*```([^\n`]*)\n(.*?)^[ \t]*```[ \t]*$", re.MULTILINE | re.DOTALL)
_STOP_REASONS = frozenset(get_args(StopReason))
# The lenient parser's row and salvage patterns: a row of digits, optionally separated by spaces
# or commas (colors are single digits, so each digit is one cell); the test index; and the
# quoted digit-string rows of a block whose brackets do not balance (run P6.5-02, 36a08778: a
# closing bracket missing). A row such as "01 02" is four cells, 0, 1, 0, 2.
_DIGIT_ROW = re.compile(r"[0-9](?:[ ,]*[0-9])*")
_TEST_INDEX = re.compile(r'"test_index"\s*:\s*(-?\d+)')
_GRID_KEY = re.compile(r'"grid"\s*:')
_QUOTED_ROW = re.compile(r'"([^"\\]*)"')
_PUNCTUATION = re.compile(r"[\[\]{},\s]*")
# AttemptRecord fields left out of a record's JSON when they hold their default, so a run with
# P6.9's options off writes Submission 3's rows byte for byte.
_QUIET_FIELDS = {"budget_forced": 0, "answer_lenient": [], "budget_refused": 0, "turn_cuts": 0}


@dataclass(frozen=True)
class AttemptContext:
    """Which attempt an engine call serves.

    Why: :meth:`Engine.complete` takes only messages, yet the engine's timing log needs the
    task id and the scripted engine picks its reply by attempt. The loop sets this context for
    the duration of an attempt (a ``contextvars`` variable, so each asyncio task sees its own).

    Attributes:
        task_id: The task id.
        attempt_id: The attempt id, for example ``"0934a4d8/p1/0"``.
        phase: The scheduler phase.
    """

    task_id: str
    attempt_id: str
    phase: int


_ATTEMPT: contextvars.ContextVar[AttemptContext | None] = contextvars.ContextVar(
    "arc26_agent_attempt", default=None
)


def current_attempt() -> AttemptContext | None:
    """Return the attempt the current code runs for, or None outside an attempt.

    Returns:
        The context set by :func:`run_attempt` or :func:`attempt_context`.
    """
    return _ATTEMPT.get()


@contextlib.contextmanager
def attempt_context(context: AttemptContext) -> Iterator[AttemptContext]:
    """Set the attempt context for the enclosed block, as :func:`run_attempt` does.

    Why: a caller that talks to the engine outside :func:`run_attempt` (a throughput probe, a
    test) can still attribute its calls to a task in the timing log.

    Args:
        context: The attempt.

    Yields:
        The context.
    """
    token = _ATTEMPT.set(context)
    try:
        yield context
    finally:
        _ATTEMPT.reset(token)


@dataclass(frozen=True)
class Budget:
    """Limits and sampling settings of one attempt.

    Why: the scheduler's phases differ only in these numbers (design note, sections 5.4 and 5.5):
    phase 1 is one short attempt with up to 2 feedback rounds, phase 2 a long one.

    Attributes:
        max_turns: Thinking replies per attempt (forcing turns do not count).
        attempt_tokens: Generated tokens per attempt over all replies, reasoning included.
        turn_tokens: The cap on one thinking reply (the request's ``max_tokens``), lowered to
            what is left of ``attempt_tokens`` after the forcing reserve.
        min_turn_tokens: No reply is requested with less room than this; the attempt stops with
            ``"budget"`` instead.
        context_tokens: The engine's context window (vLLM's ``max_model_len``), or None. A
            reply's ``max_tokens`` is lowered so that an upper bound of the prompt plus the reply
            (plus, for a thinking reply, the forcing turn's reserve) fits. The first prompt is
            bounded at one token per character; a later one from the engine's counts of the
            previous call (its prompt and generated tokens) plus one token per character of the
            new message.
        temperature: Sampling temperature (Qwen3.8's thinking mode: 1.0).
        reasoning_effort: The engine's effort level (Qwen3.8: ``"low"``, ``"medium"``,
            ``"xhigh"``), or None for the model's default.
        cell_timeout_s: Wall-time limit per REPL cell.
        max_cells: Code blocks run per reply; later ones are skipped with a note.
        feedback_chars: The cap on one feedback message's body, in characters.
        preserve_thinking: Keep each reply's reasoning in the conversation
            (``Message.reasoning``): the engine sends it back with the earlier turns and turns on
            Qwen3.8's ``preserve_thinking``. Off by default, since it lengthens every later
            prompt; the spike measures both.
        forcing_tokens: The cap on a forcing turn's reply, or None for no forcing turns. When
            a thinking reply ends at its cap (finish ``"length"``) and verifies nothing, the next
            request is a forcing turn: :func:`feedback.forcing_message` with thinking off for
            that call only, so the model writes its program at once from the reasoning kept so
            far. At most one per cut reply; a cut forcing reply goes on under the usual rules.
            Every thinking reply leaves this many tokens of ``attempt_tokens``, and the forcing
            turn's whole request (:meth:`forcing_reserve`) of the context, so a cut reply always
            gets its forcing turn.
        forcing_kind: The forcing note: ``"program"`` (one python block defining ``transform``),
            ``"answer"`` (the test outputs as answer blocks; run ``P6.3-04``'s direct-answer
            arm), or ``"answer2"`` (one or two answers per test input; run ``P6.3-05``).
        forcing_in_request: Force the answer inside the thinking reply's own request (task
            P6.9): the request's thinking budget is its turn cap and its ``max_tokens`` the cap
            plus ``forcing_tokens``, so the server ends the reasoning at the cap and the answer
            follows at once. A separate forcing turn follows only a reply that still ends at its
            cap without a readable answer; ``attempt_tokens`` must hold both
            (:meth:`in_request_tokens`). Needs ``forcing_tokens`` and ``thinking_end_note``.
        thinking_end_note: The transition phrase the server forces before ``</think>`` when
            the budget runs out (its ``--reasoning-config`` ``reasoning_end_str`` without the
            tag): a reply whose reasoning ends with it counts in ``budget_forced``.
        lenient_answers: Read an answer block the strict parser refuses with
            :func:`parse_answer`'s lenient rules (task P6.9, option 4b).
        turn_cut_s: The client-side cut of an in-request thinking request in seconds, or None:
            a request not answered by then is dropped (nothing of it arrives) and a forcing turn
            on the opening alone follows, so a server that ignores the budget cannot hold the
            attempt past what its budget plus the answer would take.
    """

    max_turns: int = 3
    attempt_tokens: int = 12_000
    turn_tokens: int = 8_000
    min_turn_tokens: int = 256
    context_tokens: int | None = None
    temperature: float = 1.0
    reasoning_effort: str | None = None
    cell_timeout_s: float = 10.0
    max_cells: int = 4
    feedback_chars: int = 6_000
    preserve_thinking: bool = False
    forcing_tokens: int | None = None
    forcing_kind: str = "program"
    forcing_in_request: bool = False
    thinking_end_note: str = ""
    lenient_answers: bool = False
    turn_cut_s: float | None = None

    def __post_init__(self) -> None:
        """Reject limits that cannot work.

        Raises:
            ValueError: If a count or a token limit is below 1, the cell timeout is not
                positive, the temperature is negative, the forcing cap is below
                ``min_turn_tokens``, the forcing kind is unknown, or in-request forcing lacks
                a forcing cap or an end note.
        """
        names = (
            "max_turns",
            "attempt_tokens",
            "turn_tokens",
            "min_turn_tokens",
            "max_cells",
            "feedback_chars",
        )
        for name in names:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1; got {getattr(self, name)}")
        if self.context_tokens is not None and self.context_tokens < 1:
            raise ValueError(f"context_tokens must be at least 1; got {self.context_tokens}")
        if self.cell_timeout_s <= 0:
            raise ValueError(f"cell_timeout_s must be positive; got {self.cell_timeout_s}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative; got {self.temperature}")
        if self.forcing_kind not in feedback.FORCING_NOTES:
            raise ValueError(
                f"forcing_kind must be program, answer, or answer2; got {self.forcing_kind!r}"
            )
        if self.forcing_tokens is not None and self.forcing_tokens < self.min_turn_tokens:
            raise ValueError(
                f"forcing_tokens must be at least min_turn_tokens ({self.min_turn_tokens}); "
                f"got {self.forcing_tokens}"
            )
        if self.forcing_in_request and (
            self.forcing_tokens is None or not self.thinking_end_note.strip()
        ):
            raise ValueError("forcing_in_request needs forcing_tokens and a thinking_end_note")
        if self.turn_cut_s is not None and (not self.forcing_in_request or self.turn_cut_s <= 0):
            raise ValueError("turn_cut_s must be positive and needs forcing_in_request")

    def in_request_tokens(self) -> int:
        """Return the answer room a thinking reply's own request adds under in-request forcing.

        Returns:
            ``forcing_tokens`` when ``forcing_in_request`` is on, else 0.
        """
        return (self.forcing_tokens or 0) if self.forcing_in_request else 0

    def forcing_reserve(self) -> int:
        """Return the context a thinking reply leaves for its forcing turn, in tokens.

        Returns:
            0 without forcing turns; else the forcing reply's cap, the forcing message at one
            token per character (:func:`feedback.forcing_chars`), and the template's tokens
            around the kept reasoning, the two messages, and the prompt.
        """
        if self.forcing_tokens is None:
            return 0
        return (
            self.forcing_tokens
            + feedback.forcing_chars(self.forcing_kind)
            + THINK_SLACK
            + 2 * MESSAGE_SLACK
            + PROMPT_SLACK
        )


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return the complete fenced blocks of a reply.

    Args:
        text: The reply's visible text.

    Returns:
        ``(tag, body)`` per block in order: ``tag`` is the info string's first word, lower
        case (empty for a bare fence). An unterminated block, as in a reply cut off at the
        token limit, is not returned.
    """
    blocks = []
    for match in _FENCE.finditer(text):
        info = match.group(1).strip().lower()
        blocks.append((info.split()[0] if info else "", match.group(2)))
    return blocks


def budget_refused(err: BaseException) -> bool:
    """Whether an engine error is a server's refusal of the ``thinking_token_budget`` field.

    vLLM 0.31 answers HTTP 400 ("thinking_token_budget is set but reasoning_config is not
    configured", ``vllm/v1/engine/input_processor.py``) when it runs without a reasoning
    config; the client raises that as ``EngineError("HTTP 400: ...")``.

    Args:
        err: The exception of an engine call.

    Returns:
        True when its text names the field and an HTTP 400.
    """
    text = str(err)
    return "thinking_token_budget" in text and "400" in text


def _lenient_cell(cell: Any) -> Any:
    """A one-digit string cell as its integer; any other cell unchanged."""
    if isinstance(cell, str) and len(cell.strip()) == 1 and cell.strip().isdigit():
        return int(cell.strip())
    return cell


def _lenient_grid(grid: Any) -> Any:
    """Rows written as digit strings as lists of cells, string cells as integers."""
    if not isinstance(grid, list):
        return grid
    rows = []
    for row in grid:
        if isinstance(row, str) and _DIGIT_ROW.fullmatch(row.strip()):
            rows.append([int(c) for c in row if c.isdigit()])
        elif isinstance(row, list):
            rows.append([_lenient_cell(cell) for cell in row])
        else:
            rows.append(row)
    return rows


def _salvage(body: str) -> tuple[Any, Any] | None:
    """The test index and the digit-string rows of a block that is not JSON, or None.

    Only a block whose text after ``"grid":`` is quoted digit rows and brackets, commas, braces,
    and white space qualifies: the form of run ``P6.5-02``'s ``36a08778`` (rows as strings, a
    closing bracket missing).
    """
    key = _GRID_KEY.search(body)
    if key is None:
        return None
    tail = body[key.end() :]
    rows = _QUOTED_ROW.findall(tail)
    if not rows or not all(_DIGIT_ROW.fullmatch(r.strip()) for r in rows):
        return None
    if not _PUNCTUATION.fullmatch(_QUOTED_ROW.sub("", tail)):
        return None
    index = _TEST_INDEX.search(body[: key.start()])
    return (int(index.group(1)) if index else 0), rows


def parse_answer(body: str, test_count: int, *, lenient: bool = False) -> tuple[int, Grid]:
    """Parse one direct answer block.

    The lenient rules (task P6.9, option 4b; runs ``P6.3-06``, ``P6.5-02``, ``P6.5-03`` lost
    answers to them) add three readings of a block the strict rules refuse: a row written as a
    string of digits (``"0123"``, also with spaces or commas between the digits) is that row's
    cells; a cell written as a one-digit string is that digit; and a block that is not JSON
    but whose quoted strings after ``"grid":`` are all digit-string rows is those rows. Every
    grid then meets the strict grid rules (rectangular, 1 to 30 per side, colors 0 to 9): a
    ragged or oversized grid is refused either way. On a block the strict rules accept the
    lenient rules return the same answer, since they only rewrite strings and a strict grid
    holds none.

    Args:
        body: The block's text: ``{"test_index": j, "grid": [[...]]}`` or a bare grid.
        test_count: The task's number of test inputs.
        lenient: Apply the lenient rules.

    Returns:
        The test input's index and the grid.

    Raises:
        ValueError: If the text is longer than :data:`ANSWER_CHARS`, is not JSON (and, when
            lenient, cannot be salvaged) or nests too deeply, the index is out of range, or the
            grid is invalid.
    """
    if len(body) > ANSWER_CHARS:
        raise ValueError(f"longer than {ANSWER_CHARS} characters")
    try:
        data = json.loads(body)
    except RecursionError:
        raise ValueError("nested too deeply") from None
    except ValueError as err:
        salvaged = _salvage(body) if lenient else None
        if salvaged is None:
            raise ValueError(f"not JSON ({err})") from None
        data = {"test_index": salvaged[0], "grid": salvaged[1]}
    if isinstance(data, dict):
        index, grid = data.get("test_index", 0), data.get("grid")
    else:
        index, grid = 0, data
    if lenient:
        grid = _lenient_grid(grid)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < test_count:
        raise ValueError(f"test_index must be an integer from 0 to {test_count - 1}; got {index!r}")
    problem = grid_problem(grid)
    if problem is not None:
        raise ValueError(f"invalid grid: {problem}")
    return index, [list(row) for row in grid]


def _is_program_statement(node: ast.stmt) -> bool:
    if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return True
    if isinstance(node, ast.Try):
        return all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in node.body)
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
        names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        return not names & REPL_NAMES
    return False


def program_source(code: str) -> str:
    """Return the part of a code block that the verifier judges.

    Why: models often define ``transform`` and test it on ``train`` in the same block; the test
    lines need the REPL's variables and would fail the program in the verifier's fresh process.

    Args:
        code: A code block's source.

    Returns:
        Its imports, ``try`` blocks whose body only imports (with their handlers), function and
        class definitions (with decorators), and assignments whose value reads none of
        :data:`REPL_NAMES`, as the original lines in order; ``code`` itself when it does not
        parse (the verifier reports the syntax error).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines(keepends=True)
    kept = []
    last = 0
    for node in tree.body:
        if not _is_program_statement(node):
            continue
        decorators = getattr(node, "decorator_list", ())
        start = max(min([node.lineno, *(d.lineno for d in decorators)]), last + 1)
        end = node.end_lineno or node.lineno
        if start <= end:
            kept.append("".join(lines[start - 1 : end]))
            last = end
    source = "".join(kept)
    return source if not source or source.endswith("\n") else source + "\n"


def attempt_seed(seed: int, attempt_id: str) -> int:
    """Return the sampling seed of one attempt, derived from the run's seed.

    Why: the project compares runs at one fixed seed; a hash keeps attempts of the same run
    distinct and identical across runs, platforms and Python versions.

    Args:
        seed: The run's seed.
        attempt_id: The attempt id.

    Returns:
        A 32-bit non-negative integer.
    """
    digest = hashlib.sha256(f"{seed}:{attempt_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def plain(value: Any) -> Any:
    """Return a copy of a value made of plain JSON types.

    Why: a verifier or a program can hand back numpy scalars and arrays (``np.bool_``,
    ``np.int64``, ``ndarray``), which ``json`` cannot write and which compare elementwise; the
    reports, the records and the selection need plain Python values.

    Args:
        value: None, a bool, int, float or str, a numpy scalar or array, a list, tuple or mapping
            of such values, or a dataclass instance holding them.

    Returns:
        The same data with numpy scalars as Python scalars, arrays and tuples as lists, mappings
        as dicts with string keys, and dataclasses as dicts of their fields.

    Raises:
        ValueError: If a float is NaN or infinite (JSON has no such number).
        TypeError: If a value has any other type.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"not a finite number: {value!r}")
        return float(value)
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    raise TypeError(f"not plain data: {type(value).__name__}")


def _plain_grid(value: Any) -> Grid | None:
    """A valid grid as plain lists, else None (the contract's value for a missing grid)."""
    if value is None:
        return None
    try:
        grid = plain(value)
    except (TypeError, ValueError):
        return None
    return grid if grid_problem(grid) is None else None


def _plain_verification(v: Verification) -> Verification:
    """The verifier's result with plain Python values, so reports and records can use it."""
    return Verification(
        source=str(v.source),
        source_sha256=str(v.source_sha256),
        pairs=tuple(
            PairResult(
                index=int(p.index),
                exact=bool(p.exact),
                output=_plain_grid(p.output),
                error=None if p.error is None else str(p.error),
                shape_match=bool(p.shape_match),
                cell_accuracy=float(p.cell_accuracy),
            )
            for p in v.pairs
        ),
        test_outputs=tuple(_plain_grid(t) for t in v.test_outputs),
        guards=tuple(GuardFinding(str(g.rule), str(g.detail), bool(g.fatal)) for g in v.guards),
        seconds=float(v.seconds),
    )


def _prompt_bound(messages: Sequence[Message]) -> int:
    """An upper bound of a prompt's tokens: one per character, plus template slack."""
    return sum(len(m.content) + MESSAGE_SLACK for m in messages) + PROMPT_SLACK


class _Attempt:
    """The state of one running attempt; :func:`run_attempt` is its public face."""

    def __init__(
        self,
        task: AgentTask,
        engine: Engine,
        repl: Repl,
        verifier: Verifier,
        budget: Budget,
        *,
        seed: int | None,
        deadline: float | None,
        executor: Executor | None,
        transcript: list[dict[str, Any]] | None,
    ) -> None:
        self.task = task
        self.engine = engine
        self.repl = repl
        self.verifier = verifier
        self.budget = budget
        self.seed = seed
        self.deadline = deadline
        self.executor = executor
        self.transcript = transcript
        self.verifications: list[Verification] = []
        self.checked: dict[str, Verification] = {}
        self.answers: list[tuple[int, Grid]] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.turns = 0
        self.rounds = 0
        self.forcing_turns = 0
        self.forced_programs = 0
        self.budget_forced = 0
        self.budget_refused = 0
        self.turn_cuts = 0
        # In-request forcing for this attempt: off after the server refused the budget field.
        self.in_request = budget.forcing_in_request
        self.lenient: list[bool] = []
        self.verified = False

    def _log(self, message: Message, reasoning: str | None = None, forcing: bool = False) -> None:
        if self.transcript is not None:
            entry: dict[str, Any] = {"role": message.role, "content": message.content}
            if reasoning is not None:
                entry["reasoning"] = reasoning
            if forcing:
                entry["forcing"] = True
            self.transcript.append(entry)

    def _past_deadline(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    async def _call(self, fn: Callable[..., _T], *args: Any) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, functools.partial(fn, *args))

    async def _complete(
        self, messages: list[Message], max_tokens: int, forcing: bool, think: int | None = None
    ) -> Completion:
        # Only a forcing call passes the switch, and only an in-request forced call its thinking
        # budget, so an Engine without either argument serves every other call.
        extra: dict[str, Any] = {"thinking": False} if forcing else {}
        if think is not None:
            extra["thinking_token_budget"] = think
        call = self.engine.complete(
            tuple(messages),
            max_tokens=max_tokens,
            temperature=self.budget.temperature,
            seed=self.seed,
            reasoning_effort=self.budget.reasoning_effort,
            **extra,
        )
        timeout = None if self.deadline is None else max(0.0, self.deadline - time.monotonic())
        if think is not None and self.budget.turn_cut_s is not None:
            cut = self.budget.turn_cut_s
            timeout = cut if timeout is None else min(timeout, cut)
        if timeout is None:
            return await call
        return await asyncio.wait_for(call, timeout=timeout)

    async def _opening_bound(self, messages: list[Message]) -> int:
        """The first prompt's bound: the server's count plus slack, else one token per character.

        An engine with ``count_prompt_tokens`` (optional; P6.3-02 revision 5) counts the prompt
        as it renders it; a failed or odd count falls back to the character bound, and the
        smaller of the two is used, so the bound is never looser than before.
        """
        chars = _prompt_bound(messages)
        counter = getattr(self.engine, "count_prompt_tokens", None)
        if counter is None or self._past_deadline():
            return chars
        call = counter(tuple(messages), reasoning_effort=self.budget.reasoning_effort)
        try:
            if self.deadline is None:
                count = await call
            else:
                count = await asyncio.wait_for(call, max(0.0, self.deadline - time.monotonic()))
        except Exception:
            logger.info("prompt count failed in %s; the character bound is used",
                        current_attempt(), exc_info=True)  # fmt: skip
            return chars
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return chars
        return min(chars, count + COUNT_SLACK)

    def _room(self, bound: int, forcing: bool) -> tuple[int, int | None]:
        """The next reply's ``max_tokens`` and, under in-request forcing, its thinking budget.

        A thinking reply leaves the forcing turn's reserve. Under ``forcing_in_request`` it also
        holds the answer's ``forcing_tokens`` after its reasoning: the budget is the room less
        those, and the room is too small when the budget is.
        """
        budget = self.budget
        left = budget.attempt_tokens - self.completion_tokens
        answer = 0 if forcing or not self.in_request else budget.in_request_tokens()
        if forcing:
            room = min(budget.forcing_tokens or 0, left)
            reserve = 0
        else:
            room = min(budget.turn_tokens + answer, left - (budget.forcing_tokens or 0))
            reserve = budget.forcing_reserve()
        if budget.context_tokens is not None:
            room = min(room, budget.context_tokens - bound - reserve)
        if not answer:
            return room, None
        return (room, room - answer) if room - answer >= budget.min_turn_tokens else (0, None)

    def _readable(self, text: str) -> bool:
        """Whether a reply holds an answer block the attempt's parser reads."""
        count = len(self.task.test_inputs)
        for tag, body in fenced_blocks(text):
            if tag != ANSWER_TAG:
                continue
            for lenient in (False, True) if self.budget.lenient_answers else (False,):
                try:
                    parse_answer(body, count, lenient=lenient)
                except ValueError:
                    continue
                return True
        return False

    async def run(self, messages: list[Message]) -> StopReason:
        """Run turns until a stop rule fires; return the rule."""
        budget = self.budget
        for message in messages:
            self._log(message)
        bound = await self._opening_bound(messages)
        forcing = False
        while True:
            if self._past_deadline():
                return "deadline"
            room, think = self._room(bound, forcing)
            if room < budget.min_turn_tokens:
                return "budget"
            try:
                completion = await self._complete(messages, room, forcing, think)
            except TimeoutError:
                if self._past_deadline():
                    return "deadline"
                if think is not None and budget.turn_cut_s is not None:
                    # The client-side cut: nothing of the dropped request arrived; the forcing
                    # turn answers from the opening alone.
                    logger.warning("thinking request cut at %.0f s in %s", budget.turn_cut_s,
                                   current_attempt())  # fmt: skip
                    self.turn_cuts += 1
                    self.rounds += 1
                    text = feedback.forcing_message([], limit=room, kind=budget.forcing_kind)
                    messages.append(Message("user", text))
                    self._log(messages[-1], forcing=True)
                    bound += len(text) + MESSAGE_SLACK
                    forcing = True
                    continue
                logger.warning("engine timeout in %s", current_attempt(), exc_info=True)
                return "engine_error"
            except Exception as err:
                if think is not None and budget_refused(err):
                    # vLLM without a reasoning config refuses the field (HTTP 400): once more
                    # without it, then the separate forcing turn as in Submission 3.
                    logger.warning("the server refused thinking_token_budget in %s; sent again "
                                   "without it", current_attempt())  # fmt: skip
                    self.budget_refused += 1
                    self.in_request = False
                    continue
                logger.warning("engine error in %s", current_attempt(), exc_info=True)
                return "engine_error"
            self.turns += 1
            if forcing:
                self.forcing_turns += 1
            else:
                self.rounds += 1
            self.prompt_tokens += completion.prompt_tokens
            self.completion_tokens += completion.completion_tokens
            kept = completion.reasoning if budget.preserve_thinking else None
            # A reply that ran to its in-request cap without a readable answer: the server
            # ignored the budget. Its forcing turn leaves the cut reasoning out, so the answer
            # does not wait for about 100k tokens of prefill.
            ignored = (
                think is not None
                and completion.finish_reason == "length"
                and not self._readable(completion.text)
            )
            if ignored:
                kept = None
            reply = Message("assistant", completion.text, kept)
            messages.append(reply)
            self._log(reply, completion.reasoning, forcing)
            if think is not None and (completion.reasoning or "").rstrip().endswith(
                budget.thinking_end_note.strip()
            ):
                self.budget_forced += 1
            # Under in-request forcing the answer came in this reply; a separate forcing turn
            # follows only a reply cut at its cap without a readable answer.
            force_next = (
                completion.finish_reason == "length"
                and not forcing
                and budget.forcing_tokens is not None
                and (think is None or ignored)
            )
            sections = await self._process(completion, room, forced=forcing, force_next=force_next)
            if self.verified:
                return "verified"
            # The engine's own counts bound the resent conversation: the prompt as rendered, and
            # every generated token (reasoning included) when the reasoning is kept; without it,
            # the visible text at one token per character, never more than was generated.
            generated = completion.completion_tokens + THINK_SLACK
            if kept is None:
                generated = min(generated, len(completion.text))
            base = completion.prompt_tokens + generated + 2 * MESSAGE_SLACK + PROMPT_SLACK
            if force_next:
                text = feedback.forcing_message(sections, limit=room, kind=budget.forcing_kind)
            else:
                if self.rounds >= budget.max_turns:
                    return "rounds"
                left = budget.attempt_tokens - self.completion_tokens - (budget.forcing_tokens or 0)
                if left < budget.min_turn_tokens:
                    return "budget"
                if budget.context_tokens is not None:
                    left = min(left, budget.context_tokens - base - budget.forcing_reserve())
                text = feedback.compose_feedback(
                    sections,
                    next_round=self.rounds + 1,
                    max_rounds=budget.max_turns,
                    tokens_left=max(0, left),
                    max_chars=budget.feedback_chars,
                )
            report = Message("user", text)
            messages.append(report)
            self._log(report, forcing=force_next)
            bound = base + len(text)
            forcing = force_next

    async def _process(
        self, completion: Completion, max_tokens: int, *, forced: bool, force_next: bool
    ) -> list[str]:
        """Run the reply's code blocks and record its answers; return the feedback sections.

        A reply followed by a forcing turn gets no note on its truncation or on a missing code
        block: the forcing message says both.
        """
        blocks = fenced_blocks(completion.text)
        cells = [body for tag, body in blocks if tag in CODE_TAGS]
        answers = [body for tag, body in blocks if tag == ANSWER_TAG]
        notes = []
        for body in answers:
            count, lenient = len(self.task.test_inputs), False
            try:
                index, grid = parse_answer(body, count)
            except ValueError as err:
                if not self.budget.lenient_answers:
                    notes.append(f"[answer] not recorded: {err}")
                    continue
                try:
                    index, grid = parse_answer(body, count, lenient=True)
                except ValueError:
                    notes.append(f"[answer] not recorded: {err}")
                    continue
                lenient = True
            self.answers.append((index, grid))
            self.lenient.append(lenient)
            notes.append(f"[answer] recorded for test input {index}.")
        sections = []
        if completion.finish_reason == "length" and not force_next:
            sections.append(feedback.truncated_note(max_tokens))
        timeout_s = self.budget.cell_timeout_s
        to_run = cells[: self.budget.max_cells]
        program = False
        for k, code in enumerate(to_run):
            if k and self._past_deadline():
                sections.append(f"Deadline reached: {len(to_run) - k} code block(s) not run.")
                break
            result = await self._call(self.repl.run_cell, code, timeout_s)
            sections.append(feedback.cell_report(k, result, timeout_s=timeout_s))
            if result.defines_transform:
                program = True
                sections.append(await self._verify(k, code))
                if self.verified:
                    break
        if forced and program:
            self.forced_programs += 1
        skipped = len(cells) - self.budget.max_cells
        if skipped > 0 and not self.verified:
            sections.append(
                f"{skipped} more code block(s) not run: at most {self.budget.max_cells} per reply."
            )
        if not cells and not answers and not force_next:
            sections.append(feedback.NO_CODE_NOTE)
        return sections + notes

    async def _verify(self, k: int, code: str) -> str:
        """Verify the program in one code block; return its report."""
        source = program_source(code)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        known = self.checked.get(digest)
        if known is not None:
            return (
                f"[cell {k}] this program was checked before: {known.passed}/{len(known.pairs)} "
                "training examples passed; not accepted."
            )
        verification = _plain_verification(
            await self._call(self.verifier.verify, self.task, source)
        )
        self.checked[digest] = verification
        self.verifications.append(verification)
        self.verified = verification.verified
        return f"[cell {k}] {feedback.verification_report(self.task, verification)}"


async def run_attempt(
    task: AgentTask,
    engine: Engine,
    repl: Repl,
    verifier: Verifier,
    budget: Budget,
    *,
    attempt_id: str | None = None,
    phase: int = 1,
    seed: int | None = None,
    deadline: float | None = None,
    system: str | None = None,
    prior: Sequence[Verification] = (),
    executor: Executor | None = None,
    transcript: list[dict[str, Any]] | None = None,
) -> AttemptRecord:
    """Run one attempt of the agent on one task.

    Args:
        task: The task.
        engine: The chat engine.
        repl: The task's REPL, already started; the caller closes it.
        verifier: The program verifier.
        budget: The attempt's limits and sampling settings.
        attempt_id: The attempt id; ``"<task_id>/p<phase>/0"`` when None.
        phase: The scheduler phase, recorded with the attempt.
        seed: The sampling seed sent with every reply, or None.
        deadline: A ``time.monotonic()`` value after which no reply is awaited, or None.
        system: The system prompt; None builds it with :func:`prompts.system_prompt`.
        prior: Earlier failed programs to show in the task message (a later phase).
        executor: The executor for the blocking REPL and verifier calls; None uses the event
            loop's default.
        transcript: A list to which every message is appended as a dict (``role``,
            ``content``, and ``reasoning`` for replies), or None.

    Returns:
        The attempt's record. ``turns`` counts the replies received, forcing turns included; a
        call that failed or was cut by the deadline adds no tokens.

    Raises:
        Exception: Whatever the REPL or the verifier raise; their results report program
            failures, so an exception there is an infrastructure fault and ends the attempt.
    """
    attempt_id = f"{task.task_id}/p{phase}/0" if attempt_id is None else attempt_id
    start = time.monotonic()
    state = _Attempt(
        task,
        engine,
        repl,
        verifier,
        budget,
        seed=seed,
        deadline=deadline,
        executor=executor,
        transcript=transcript,
    )
    with attempt_context(AttemptContext(task.task_id, attempt_id, phase)):
        stop = await state.run(prompts.initial_messages(task, system=system, prior=prior))
    return AttemptRecord(
        task_id=task.task_id,
        attempt_id=attempt_id,
        phase=phase,
        verifications=tuple(state.verifications),
        direct_answers=tuple(state.answers),
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        turns=state.turns,
        seconds=time.monotonic() - start,
        stop_reason=stop,
        forcing_turns=state.forcing_turns,
        forced_programs=state.forced_programs,
        budget_forced=state.budget_forced,
        budget_refused=state.budget_refused,
        turn_cuts=state.turn_cuts,
        answer_lenient=tuple(state.lenient) if any(state.lenient) else (),
    )


def record_to_json(record: AttemptRecord) -> dict[str, Any]:
    """Return an attempt record as a JSON-ready dict of plain values.

    Args:
        record: The record.

    Returns:
        The record's fields as :func:`plain` makes them: nested dataclasses as dicts, tuples
        and numpy arrays as lists, numpy scalars as Python scalars. ``budget_forced`` and
        ``answer_lenient`` are left out at their defaults (0 and empty), so a run without
        P6.9's options writes the rows it wrote before them.

    Raises:
        ValueError: If a number in it is NaN or infinite.
        TypeError: If a value is not plain data.
    """
    data = plain(record)
    for key, default in _QUIET_FIELDS.items():
        if data.get(key) == default:
            del data[key]
    return data


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an integer; got {value!r}")
    return value


def _grid_or_none(value: Any) -> Grid | None:
    if value is None:
        return None
    problem = grid_problem(value)
    if problem is not None:
        raise ValueError(f"invalid grid: {problem}")
    return [list(row) for row in value]


def _verification(data: Mapping[str, Any]) -> Verification:
    pairs = tuple(
        PairResult(
            index=_int(p["index"]),
            exact=bool(p["exact"]),
            output=_grid_or_none(p["output"]),
            error=None if p["error"] is None else str(p["error"]),
            shape_match=bool(p["shape_match"]),
            cell_accuracy=float(p["cell_accuracy"]),
        )
        for p in data["pairs"]
    )
    guards = tuple(
        GuardFinding(rule=str(g["rule"]), detail=str(g["detail"]), fatal=bool(g["fatal"]))
        for g in data["guards"]
    )
    return Verification(
        source=str(data["source"]),
        source_sha256=str(data["source_sha256"]),
        pairs=pairs,
        test_outputs=tuple(_grid_or_none(t) for t in data["test_outputs"]),
        guards=guards,
        seconds=float(data["seconds"]),
    )


def record_from_json(data: Mapping[str, Any]) -> AttemptRecord:
    """Rebuild an attempt record from :func:`record_to_json`'s dict.

    Args:
        data: The parsed dict.

    Returns:
        The record.

    Raises:
        KeyError: If a field is missing (``forcing_turns``, ``forced_programs``, and
            ``budget_forced`` default to 0 and ``answer_lenient`` to empty, as in records
            written before them).
        TypeError: If a field has the wrong type.
        ValueError: If a grid is invalid, the stop reason unknown, or ``answer_lenient`` does
            not hold one flag per answer.
    """
    stop = data["stop_reason"]
    if stop not in _STOP_REASONS:
        raise ValueError(f"unknown stop reason {stop!r}")
    answers = []
    for index, grid in data["direct_answers"]:
        checked = _grid_or_none(grid)
        if checked is None:
            raise ValueError("a direct answer has no grid")
        answers.append((_int(index), checked))
    lenient = tuple(bool(flag) for flag in data.get("answer_lenient", ()))
    if lenient and len(lenient) != len(answers):
        raise ValueError("answer_lenient must hold one flag per direct answer")
    return AttemptRecord(
        task_id=str(data["task_id"]),
        attempt_id=str(data["attempt_id"]),
        phase=_int(data["phase"]),
        verifications=tuple(_verification(v) for v in data["verifications"]),
        direct_answers=tuple(answers),
        prompt_tokens=_int(data["prompt_tokens"]),
        completion_tokens=_int(data["completion_tokens"]),
        turns=_int(data["turns"]),
        seconds=float(data["seconds"]),
        stop_reason=stop,
        forcing_turns=_int(data.get("forcing_turns", 0)),
        forced_programs=_int(data.get("forced_programs", 0)),
        budget_forced=_int(data.get("budget_forced", 0)),
        budget_refused=_int(data.get("budget_refused", 0)),
        turn_cuts=_int(data.get("turn_cuts", 0)),
        answer_lenient=lenient,
    )


def _ensure_line_end(path: Path) -> None:
    """Terminate a partial last line (a write cut by a kill) so the next record starts clean."""
    try:
        with open(path, "rb+") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    except FileNotFoundError:
        return


def _append_line(path: Path, data: Mapping[str, Any]) -> None:
    line = json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def append_record(path: str | os.PathLike[str], record: AttemptRecord) -> None:
    """Append one attempt record to a JSONL file as one line.

    Why: one complete line per finished attempt, written and closed at once, so a run killed at
    any moment keeps every finished attempt; :func:`read_records` skips a cut last line.

    Args:
        path: The JSONL file; it and its directory are created if needed.
        record: The record.

    Raises:
        OSError: If the file cannot be written.
        ValueError: If the record holds a NaN or infinite number.
        TypeError: If the record holds a value that is not plain data.
    """
    _append_line(Path(path), record_to_json(record))


def read_records(path: str | os.PathLike[str]) -> tuple[list[AttemptRecord], int]:
    """Read attempt records from a JSONL file, skipping lines that are not valid records.

    Args:
        path: The JSONL file written by :func:`append_record`.

    Returns:
        The records in file order, and the number of skipped lines (blank lines are ignored).

    Raises:
        OSError: If the file cannot be read.
    """
    records = []
    skipped = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(record_from_json(json.loads(line)))
            except (ValueError, TypeError, KeyError):
                skipped += 1
    return records, skipped


async def _attempt_with_repl(
    task: AgentTask,
    engine: Engine,
    repl_factory: Callable[[AgentTask], Repl],
    verifier: Verifier,
    budget: Budget,
    executor: Executor,
    **kwargs: Any,
) -> AttemptRecord:
    """Start a REPL for the task, run one attempt in it, and close it."""
    loop = asyncio.get_running_loop()
    repl = await loop.run_in_executor(executor, repl_factory, task)
    try:
        return await run_attempt(task, engine, repl, verifier, budget, executor=executor, **kwargs)
    finally:
        try:
            await loop.run_in_executor(executor, repl.close)
        except Exception:
            logger.exception("closing the REPL of %s failed", kwargs.get("attempt_id"))


async def run_tasks(
    tasks: Sequence[AgentTask],
    engine: Engine,
    repl_factory: Callable[[AgentTask], Repl],
    verifier: Verifier,
    budget: Budget,
    *,
    concurrency: int = 16,
    attempts_per_task: int = 1,
    phase: int = 1,
    seed: int = 0,
    deadline: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
    transcript_path: str | os.PathLike[str] | None = None,
    system: str | None = None,
    prior: Mapping[str, Sequence[Verification]] | None = None,
    on_record: Callable[[AttemptRecord], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[AttemptRecord]:
    """Run attempts on many tasks concurrently against one engine.

    Each attempt gets a fresh REPL from ``repl_factory`` and the same ``budget``; attempts start
    in task order, up to ``concurrency`` at once. An attempt that raises (a REPL or verifier
    fault) is logged with its traceback, passed to ``on_error``, and leaves no record; the others
    go on.

    Args:
        tasks: The tasks, in the order to start them (the scheduler orders by ease).
        engine: The chat engine, shared by all attempts.
        repl_factory: Starts a REPL for a task; called in a worker thread.
        verifier: The program verifier, shared by all attempts and called from worker threads
            (up to ``concurrency`` at once).
        budget: The limits of each attempt.
        concurrency: Attempts in flight at once.
        attempts_per_task: Independent attempts per task (ids ``<task_id>/p<phase>/<k>``).
        phase: The scheduler phase, recorded with each attempt.
        seed: The run's seed; each attempt's seed is :func:`attempt_seed` of it.
        deadline: A ``time.monotonic()`` value: attempts not started by then are skipped, and
            running ones stop with ``"deadline"`` at their next engine call.
        log_path: The crash-safe JSONL of records (one line appended per finished attempt), or
            None.
        transcript_path: A JSONL of conversations (one line per finished attempt: its id and
            messages, replies with their reasoning), or None.
        system: The system prompt; None builds it once with :func:`prompts.system_prompt`.
        prior: Earlier failed programs per task id, shown in the task message.
        on_record: Called with each record after it is logged (for example to rewrite the
            submission); an exception there, or in writing the logs, is logged and the run goes
            on.
        on_error: Called with the attempt id and the exception of each attempt that raised, after
            it is logged (for example to count faults by kind); an exception there is logged and
            the run goes on.

    Returns:
        The records of the finished attempts, in task order, then attempt order.

    Raises:
        ValueError: If ``concurrency`` or ``attempts_per_task`` is below 1.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1; got {concurrency}")
    if attempts_per_task < 1:
        raise ValueError(f"attempts_per_task must be at least 1; got {attempts_per_task}")
    system_text = prompts.system_prompt() if system is None else system
    jobs = deque(
        (order, task, k) for order, task in enumerate(tasks) for k in range(attempts_per_task)
    )
    log = None if log_path is None else Path(log_path)
    transcripts = None if transcript_path is None else Path(transcript_path)
    for path in (log, transcripts):
        if path is not None:
            _ensure_line_end(path)
    done: list[tuple[int, int, AttemptRecord]] = []
    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="arc26-agent")

    async def worker() -> None:
        while jobs:
            order, task, k = jobs.popleft()
            if deadline is not None and time.monotonic() >= deadline:
                continue
            attempt_id = f"{task.task_id}/p{phase}/{k}"
            transcript: list[dict[str, Any]] | None = None if transcripts is None else []
            try:
                record = await _attempt_with_repl(
                    task,
                    engine,
                    repl_factory,
                    verifier,
                    budget,
                    executor,
                    attempt_id=attempt_id,
                    phase=phase,
                    seed=attempt_seed(seed, attempt_id),
                    deadline=deadline,
                    system=system_text,
                    prior=() if prior is None else prior.get(task.task_id, ()),
                    transcript=transcript,
                )
            except Exception as exc:
                logger.exception("attempt %s failed", attempt_id)
                if on_error is not None:
                    try:
                        on_error(attempt_id, exc)
                    except Exception:
                        logger.exception("on_error failed for %s", attempt_id)
                continue
            try:
                if log is not None:
                    append_record(log, record)
                if transcripts is not None and transcript is not None:
                    _append_line(
                        transcripts,
                        {"task_id": task.task_id, "attempt_id": attempt_id, "messages": transcript},
                    )
            except Exception:
                logger.exception("writing the logs of %s failed", attempt_id)
            done.append((order, k, record))
            if on_record is not None:
                try:
                    on_record(record)
                except Exception:
                    logger.exception("on_record failed for %s", attempt_id)

    try:
        await asyncio.gather(*(worker() for _ in range(min(concurrency, len(jobs)))))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    done.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in done]
