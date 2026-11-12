"""Shared types and interfaces of the agent solver (P6, ADR-020).

Why: three tasks build the solver at once: P6.1 (REPL sandbox, guards, verifier, helper library),
P6.2 (engine client, mock engine, agent loop, attempt selection), and P6.3 (the GPU spike). This
module is their contract, fixed before they start, so none waits on another. Change it only
through the orchestrator; a task that needs a new field returns the request instead.

Grids are ``arc26.data.Grid``: rows of cells, each cell a color in 0..9, 1 to 30 cells per side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from arc26.data import Grid


@dataclass(frozen=True)
class Pair:
    """One demonstration pair of a task.

    Attributes:
        input: The input grid.
        output: The output grid.
    """

    input: Grid
    output: Grid


@dataclass(frozen=True)
class AgentTask:
    """A task as the agent sees it: demonstrations and test inputs, never test outputs.

    Why: in competition mode no test output exists, so no object the solver handles may carry
    one; development-mode scoring reads solutions elsewhere (``arc26.evaluate``).

    Attributes:
        task_id: The ARC task id, for example ``"0934a4d8"``.
        train: The demonstration pairs, in file order.
        test_inputs: The test input grids, in file order.
    """

    task_id: str
    train: tuple[Pair, ...]
    test_inputs: tuple[Grid, ...]


@dataclass(frozen=True)
class CellResult:
    """What one REPL cell did.

    Attributes:
        stdout: Captured standard output and standard error, truncated by the sandbox to its
            configured limit (the truncation is marked in the text).
        error: The exception's last traceback line, ``"timeout"``, ``"memory"``,
            ``"PolicyError: ..."`` (the cell broke the sandbox's policy), ``"crashed: ..."`` (the
            process died), or None.
        defines_transform: Whether the cell's source defines a top-level ``transform``.
        seconds: Wall time of the cell.
        restarted: Whether the sandbox was killed and restarted during or after this cell (the
            REPL state is then lost; the task's grids are loaded again).
    """

    stdout: str
    error: str | None
    defines_transform: bool
    seconds: float
    restarted: bool


class Repl(Protocol):
    """A persistent per-task Python REPL (P6.1).

    It starts with the task's grids loaded as ``train`` (a list of ``(input, output)`` pairs of
    numpy arrays) and ``test_inputs`` (a list of numpy arrays), numpy as ``np``, and the helper
    library as ``lib``. It is for exploration; programs are judged only by :class:`Verifier`.
    """

    def run_cell(self, code: str, timeout_s: float) -> CellResult:
        """Run one cell in the persistent namespace.

        Args:
            code: Python source of the cell.
            timeout_s: Wall-time limit; on breach the process is killed and restarted.

        Returns:
            The cell's result.
        """
        ...

    def close(self) -> None:
        """Stop the REPL process and release its resources."""
        ...


@dataclass(frozen=True)
class GuardFinding:
    """One finding of the hardcoding and safety guards on a program.

    Attributes:
        rule: A stable rule name, for example ``"forbidden_import"``, ``"literal_output"``,
            ``"input_literal_compare"``, ``"shape_dispatch"``, ``"output_equals_input"``.
        detail: A short human-readable explanation (line number where known).
        fatal: True rejects the program; False only demotes it when attempts are ranked.
    """

    rule: str
    detail: str
    fatal: bool


@dataclass(frozen=True)
class PairResult:
    """A program's result on one demonstration pair.

    Attributes:
        index: The pair's index in ``AgentTask.train``.
        exact: Whether the output equals the expected grid exactly.
        output: The produced grid, or None if the call raised, timed out, or returned something
            that is not a valid grid.
        error: The error text, or None.
        shape_match: Whether the produced grid has the expected shape.
        cell_accuracy: The fraction of equal cells when shapes match, else 0.0.
    """

    index: int
    exact: bool
    output: Grid | None
    error: str | None
    shape_match: bool
    cell_accuracy: float


@dataclass(frozen=True)
class Verification:
    """The verifier's judgement of one program on one task.

    Attributes:
        source: The program source as verified (it must define ``transform``).
        source_sha256: The sha256 of ``source`` encoded as UTF-8, hex.
        pairs: One result per demonstration pair, in order.
        test_outputs: The program's output on each test input, or None where it failed.
        guards: Every guard finding, fatal or not.
        seconds: Wall time of the whole verification.
    """

    source: str
    source_sha256: str
    pairs: tuple[PairResult, ...]
    test_outputs: tuple[Grid | None, ...]
    guards: tuple[GuardFinding, ...]
    seconds: float

    @property
    def passed(self) -> int:
        """Return the number of demonstration pairs reproduced exactly."""
        return sum(1 for p in self.pairs if p.exact)

    @property
    def verified(self) -> bool:
        """Return whether the program counts as an attempt source.

        True when every demonstration is reproduced, no guard finding is fatal, and every test
        input produced a valid grid.
        """
        return (
            bool(self.pairs)
            and self.passed == len(self.pairs)
            and not any(g.fatal for g in self.guards)
            and all(t is not None for t in self.test_outputs)
        )


class Verifier(Protocol):
    """Judges programs in a fresh, least-privilege sandbox (P6.1).

    The program runs in a new subprocess whose namespace holds only numpy and the helper library
    snapshot: no task variables, files, or network. It is called on every demonstration input and
    every test input with a per-call time limit.
    """

    def verify(self, task: AgentTask, source: str) -> Verification:
        """Verify one self-contained program on one task.

        Args:
            task: The task.
            source: Python source that defines ``transform(grid)``; it may import numpy and
                ``arc26.agent.lib`` and define helpers, and must not depend on REPL state.

        Returns:
            The verification.
        """
        ...


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """One chat message sent to the engine.

    Attributes:
        role: The speaker.
        content: The text.
        reasoning: An assistant turn's reasoning text, sent back only when the run keeps earlier
            reasoning in context (Qwen3.8's ``preserve_thinking``); None otherwise. Keeping it
            lengthens every later prompt, which lowers throughput, so the default is to drop it
            and the spike measures both.
    """

    role: Role
    content: str
    reasoning: str | None = None


@dataclass(frozen=True)
class Completion:
    """One engine reply with its token accounting.

    Attributes:
        text: The visible answer text (code cells are fenced in it).
        reasoning: The model's reasoning text when the engine returns it separately, else None.
        prompt_tokens: Tokens in the prompt as the engine counted them.
        completion_tokens: Tokens generated, reasoning included.
        finish_reason: The engine's finish reason (``"stop"``, ``"length"``, ...).
        seconds: Wall time from request to last token.
    """

    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    seconds: float


class Engine(Protocol):
    """An OpenAI-compatible chat engine: vLLM or SGLang on Kaggle, a mock on the CPU (P6.2).

    Optional (P6.3-02, revision 5): ``async count_prompt_tokens(messages, *, reasoning_effort,
    thinking) -> int``, the server's count of the prompt a call would send. The loop uses it,
    when present, to bound an attempt's opening prompt, and falls back to one token per
    character when it is absent or fails.

    Optional (P6.9): a ``thinking_token_budget`` keyword of :meth:`complete`, the reasoning
    tokens after which the server ends the thinking inside the request (vLLM's field of that
    name). The loop passes it only under ``Budget.forcing_in_request``.
    """

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
    ) -> Completion:
        """Generate one reply.

        Args:
            messages: The conversation so far.
            max_tokens: Cap on generated tokens, reasoning included.
            temperature: Sampling temperature.
            seed: Sampling seed, if the engine supports one.
            reasoning_effort: An engine-specific effort level (for example ``"low"``), or None
                for the model's default.
            thinking: This call's thinking switch (Qwen3.8's ``enable_thinking``), or None for
                the engine's default. The loop's forcing turn sends False (P6.3-02, revision 5);
                it passes the argument only then, so an engine without it serves every other
                call.

        Returns:
            The reply.
        """
        ...


CandidateSource = Literal["program", "direct", "nvarc"]


@dataclass(frozen=True)
class Candidate:
    """One candidate output for one test input, with what supports it.

    Attributes:
        task_id: The task id.
        test_index: The test input's index.
        grid: The candidate output.
        source: ``"program"`` (a verified program's output), ``"direct"`` (a grid the model
            answered without a program), or ``"nvarc"`` (the transducer's candidate).
        votes: Independent attempts that produced this grid (verified programs for
            ``"program"``).
        size: The smallest supporting program's size in AST nodes (0 for other sources).
        demoted: Whether any non-fatal guard finding applies to every supporting program.
    """

    task_id: str
    test_index: int
    grid: Grid
    source: CandidateSource
    votes: int
    size: int
    demoted: bool


StopReason = Literal["verified", "budget", "rounds", "engine_error", "deadline"]


@dataclass(frozen=True)
class AttemptRecord:
    """The log of one agent attempt on one task (P6.2), for selection and for the reports.

    Attributes:
        task_id: The task id.
        attempt_id: A unique id within the run, for example ``"0934a4d8/p1/0"``.
        phase: The scheduler phase that ran it (1 cheap, 2 deep, 3 frontier).
        verifications: Every program the attempt submitted, in order.
        direct_answers: Grids the model answered directly, as ``(test_index, grid)``.
        prompt_tokens: Prompt tokens over all turns.
        completion_tokens: Generated tokens over all turns.
        turns: Engine calls made, forcing turns included.
        seconds: Wall time.
        stop_reason: Why the attempt ended.
        forcing_turns: Forcing turns among ``turns``: calls with thinking off that followed a
            reply cut at its cap (P6.3-02, revision 5).
        forced_programs: Forcing turns whose reply held a code block defining ``transform``.
        budget_forced: Thinking replies whose reasoning the server ended at the request's
            thinking budget (``Budget.forcing_in_request``; P6.9): their reasoning ends with
            ``Budget.thinking_end_note``.
        budget_refused: Thinking requests sent again without the budget field after the
            server refused it (HTTP 400; P6.9).
        turn_cuts: Thinking requests dropped at ``Budget.turn_cut_s`` (P6.9).
        answer_lenient: One flag per entry of ``direct_answers``: True when only the lenient
            parser (``Budget.lenient_answers``; P6.9) read its block. Empty when every answer
            was read by the strict parser, as in records written before the flag.
    """

    task_id: str
    attempt_id: str
    phase: int
    verifications: tuple[Verification, ...]
    direct_answers: tuple[tuple[int, Grid], ...]
    prompt_tokens: int
    completion_tokens: int
    turns: int
    seconds: float
    stop_reason: StopReason
    forcing_turns: int = 0
    forced_programs: int = 0
    budget_forced: int = 0
    answer_lenient: tuple[bool, ...] = ()
    budget_refused: int = 0
    turn_cuts: int = 0
