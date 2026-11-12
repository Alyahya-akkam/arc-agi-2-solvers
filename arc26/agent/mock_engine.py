"""A scripted chat engine for CPU tests and the CPU end-to-end run (P6.2).

Why: the agent loop, the runner and the selection must be tested without a GPU. A
:class:`ScriptedEngine` replays, per attempt or per task, a fixed list of replies: one per turn,
chosen by the number of assistant messages in the conversation. Replies cover the cases the loop
must handle: a correct program, a wrong program then a fix after feedback, hostile programs, a
direct answer, a reply without code, an overlong reply cut at ``max_tokens`` (followed by the
forcing turn's reply when the budget has one), and an engine error. Token counts are simulated
from text length (``chars_per_token``), and replies can be delayed to test concurrency and
deadlines.

The builders below make such replies; :func:`hostile_programs` returns programs that the guards
of P6.1 must reject. This module never runs any program.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from arc26.agent.loop import current_attempt
from arc26.agent.types import AgentTask, Completion, Message
from arc26.data import Grid
from arc26.timing import TimingLogger

ENGINE_OP = "engine"


class ScriptedEngineError(RuntimeError):
    """The error a scripted reply with ``error`` set raises, as a failed engine call would."""


@dataclass(frozen=True)
class ScriptedReply:
    """One scripted engine reply.

    Attributes:
        text: The visible answer text.
        reasoning: The reasoning text, or None; it counts toward the generated tokens.
        delay_s: Seconds to wait before replying (on top of the engine's own delay).
        error: When set, the call raises :class:`ScriptedEngineError` with this message.
        finish_reason: The finish reason when the reply fits ``max_tokens``; ``"length"``
            replaces it when it does not.
    """

    text: str = ""
    reasoning: str | None = None
    delay_s: float = 0.0
    error: str | None = None
    finish_reason: str = "stop"


@dataclass(frozen=True)
class ScriptedCall:
    """One call the scripted engine received, for the tests to inspect.

    Attributes:
        task_id: The attempt context's task id, or None outside an attempt.
        attempt_id: The attempt context's attempt id, or None.
        turn: The number of assistant messages in the conversation (0 for the first reply).
        messages: The conversation as sent.
        max_tokens: The call's token cap.
        temperature: The call's temperature.
        seed: The call's seed.
        reasoning_effort: The call's effort level.
        thinking: The call's thinking switch (False on a forcing turn), or None.
        thinking_token_budget: The call's thinking budget (in-request forcing, P6.9), or None.
    """

    task_id: str | None
    attempt_id: str | None
    turn: int
    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float
    seed: int | None
    reasoning_effort: str | None
    thinking: bool | None = None
    thinking_token_budget: int | None = None


def code_reply(source: str, note: str = "Here is a program.") -> ScriptedReply:
    """Return a reply with one ```python block.

    Args:
        source: The code.
        note: The prose before the block.

    Returns:
        The reply.
    """
    return ScriptedReply(text=f"{note}\n\n```python\n{source.rstrip()}\n```\n")


def answer_reply(test_index: int, grid: Grid, note: str = "My direct answer.") -> ScriptedReply:
    """Return a reply with one ```answer block.

    Args:
        test_index: The test input the grid answers.
        grid: The answered grid.
        note: The prose before the block.

    Returns:
        The reply.
    """
    body = json.dumps({"test_index": test_index, "grid": grid})
    return ScriptedReply(text=f"{note}\n\n```answer\n{body}\n```\n")


def no_code_reply(text: str = "I need to think about the rule more.") -> ScriptedReply:
    """Return a reply without any fenced block.

    Args:
        text: The reply's text.

    Returns:
        The reply.
    """
    return ScriptedReply(text=text)


def overlong_reply(
    reasoning_chars: int, source: str = "def transform(grid):\n    return grid\n"
) -> ScriptedReply:
    """Return a reply whose reasoning alone exceeds a small ``max_tokens``.

    Args:
        reasoning_chars: The reasoning's length in characters.
        source: A program after the reasoning, cut off with it when the cap is hit.

    Returns:
        The reply.
    """
    sentence = "Let me look at the grids once more. "
    reasoning = (sentence * (reasoning_chars // len(sentence) + 1))[:reasoning_chars]
    return ScriptedReply(text=f"```python\n{source.rstrip()}\n```\n", reasoning=reasoning)


def error_reply(message: str = "engine unavailable") -> ScriptedReply:
    """Return a reply that makes the call fail.

    Args:
        message: The error message.

    Returns:
        The reply.
    """
    return ScriptedReply(error=message)


def hostile_programs(task: AgentTask) -> dict[str, str]:
    """Return programs that pass demonstrations by cheating or attack the sandbox.

    Why: the guards and the sandbox (P6.1) must reject each of these; scripted into an attempt,
    they test the loop's handling of rejections end to end. Never run them outside the sandbox.

    Args:
        task: The task the literal and dispatch programs are written for.

    Returns:
        Rule name to source: ``literal_lookup`` (demonstration outputs keyed by the input's
        string form), ``shape_dispatch`` (branches on a demonstration's exact size and returns
        its output), ``input_hash`` (keys on the input's hash), ``imports_os``, ``opens_file``,
        ``dunder_import``, ``infinite_loop``, ``huge_allocation``.
    """
    table = {str(pair.input): pair.output for pair in task.train}
    first = task.train[0]
    height, width = len(first.input), len(first.input[0])
    literal = json.dumps(first.output)
    return {
        "literal_lookup": (
            f"TABLE = {table!r}\n\ndef transform(grid):\n    return TABLE.get(str(grid), grid)\n"
        ),
        "shape_dispatch": (
            "def transform(grid):\n"
            f"    if len(grid) == {height} and len(grid[0]) == {width}:\n"
            f"        return {literal}\n"
            "    return grid\n"
        ),
        "input_hash": (
            "def transform(grid):\n"
            f"    if hash(str(grid)) == {hash(str(first.input))}:\n"
            f"        return {literal}\n"
            "    return grid\n"
        ),
        "imports_os": "import os\n\ndef transform(grid):\n    os.listdir('/')\n    return grid\n",
        "opens_file": "def transform(grid):\n    open('/etc/hostname').read()\n    return grid\n",
        "dunder_import": (
            "def transform(grid):\n    __import__('subprocess').run(['true'])\n    return grid\n"
        ),
        "infinite_loop": "def transform(grid):\n    while True:\n        pass\n",
        "huge_allocation": (
            "def transform(grid):\n    block = bytearray(64 * 1024 ** 3)\n    return grid\n"
        ),
    }


class ScriptedEngine:
    """An :class:`~arc26.agent.types.Engine` that replays scripted replies.

    The script of a call is ``scripts[attempt_id]``, else ``scripts[task_id]`` (from the
    attempt context the loop sets), else none; reply ``k`` answers the call whose conversation
    holds ``k`` assistant messages, and calls past the script's end get ``fallback``.

    Attributes:
        calls: Every call received, in order.
        counts: Every conversation passed to :meth:`count_prompt_tokens`.
        count_error: When set, :meth:`count_prompt_tokens` raises with this message.
    """

    def __init__(
        self,
        scripts: Mapping[str, Sequence[ScriptedReply | str]] | None = None,
        *,
        fallback: ScriptedReply | None = None,
        chars_per_token: float = 4.0,
        delay_s: float = 0.0,
        timing: TimingLogger | None = None,
        budget_note: str = "",
    ) -> None:
        """Create the engine.

        Args:
            scripts: Replies per attempt id or task id; a plain string is a reply's text.
            fallback: The reply when a script is missing or exhausted; a reply without code by
                default.
            chars_per_token: Characters per simulated token.
            delay_s: Seconds every call waits before replying.
            timing: Where each call's ``{puzzle_id, op: "engine", seconds, tokens}`` record
                goes, or None.
            budget_note: What a server configured with ``--reasoning-config`` forces before
                ``</think>`` when a call's thinking budget runs out (P6.9): appended to the
                reasoning it cuts.

        Raises:
            ValueError: If ``chars_per_token`` is not positive or ``delay_s`` is negative.
        """
        if chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be positive; got {chars_per_token}")
        if delay_s < 0:
            raise ValueError(f"delay_s must be non-negative; got {delay_s}")
        self._scripts = {
            key: tuple(ScriptedReply(text=r) if isinstance(r, str) else r for r in replies)
            for key, replies in (scripts or {}).items()
        }
        self._fallback = fallback if fallback is not None else no_code_reply()
        self._chars_per_token = chars_per_token
        self._delay_s = delay_s
        self._timing = timing
        self._budget_note = budget_note
        self.calls: list[ScriptedCall] = []
        # Conversations counted by count_prompt_tokens; a message here makes each count fail.
        self.counts: list[tuple[Message, ...]] = []
        self.count_error: str | None = None

    def tokens(self, text: str) -> int:
        """Return the simulated token count of a text.

        Args:
            text: The text.

        Returns:
            ``ceil(len(text) / chars_per_token)``.
        """
        return math.ceil(len(text) / self._chars_per_token)

    def prompt_tokens(self, messages: Sequence[Message]) -> int:
        """Return the simulated prompt count of a conversation, as :meth:`complete` reports it.

        Args:
            messages: The conversation.

        Returns:
            Each message's content and reasoning at ``chars_per_token``, plus 4 per message.
        """
        return sum(self.tokens(m.content) + self.tokens(m.reasoning or "") + 4 for m in messages)

    async def count_prompt_tokens(
        self,
        messages: Sequence[Message],
        *,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
    ) -> int:
        """Return :meth:`prompt_tokens`, as a server's tokenize endpoint would count the prompt.

        Args:
            messages: The conversation.
            reasoning_effort: Ignored.
            thinking: Ignored.

        Returns:
            The simulated count.

        Raises:
            ScriptedEngineError: If ``count_error`` is set (a failed count).
        """
        self.counts.append(tuple(messages))
        if self.count_error is not None:
            raise ScriptedEngineError(self.count_error)
        return self.prompt_tokens(messages)

    def _reply(self, task_id: str | None, attempt_id: str | None, turn: int) -> ScriptedReply:
        for key in (attempt_id, task_id):
            if key is not None and key in self._scripts:
                script = self._scripts[key]
                return script[turn] if turn < len(script) else self._fallback
        return self._fallback

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
        thinking_token_budget: int | None = None,
    ) -> Completion:
        """Return the scripted reply for this call.

        Args:
            messages: The conversation so far.
            max_tokens: Cap on generated tokens; a longer reply is cut (reasoning first, as a
                thinking model spends it) and finishes with ``"length"``.
            temperature: Recorded only.
            seed: Recorded only.
            reasoning_effort: Recorded only.
            thinking: Recorded; False drops the scripted reasoning, as a reply with thinking off
                has none.
            thinking_token_budget: As vLLM's: reasoning longer than this many tokens is cut
                there and ends with ``budget_note``; the reply's text follows it, so the
                scripted answer arrives in the same reply (then ``max_tokens`` applies).

        Returns:
            The reply with simulated token counts (the prompt's count includes reasoning that
            earlier assistant messages carry, as an engine that sends it back would see).

        Raises:
            ScriptedEngineError: If the scripted reply has ``error`` set.
        """
        start = time.perf_counter()
        context = current_attempt()
        task_id = None if context is None else context.task_id
        attempt_id = None if context is None else context.attempt_id
        turn = sum(1 for m in messages if m.role == "assistant")
        self.calls.append(
            ScriptedCall(
                task_id,
                attempt_id,
                turn,
                tuple(messages),
                max_tokens,
                temperature,
                seed,
                reasoning_effort,
                thinking,
                thinking_token_budget,
            )
        )
        reply = self._reply(task_id, attempt_id, turn)
        delay = self._delay_s + reply.delay_s
        if delay > 0:
            await asyncio.sleep(delay)
        if reply.error is not None:
            raise ScriptedEngineError(reply.error)
        reasoning = "" if thinking is False else reply.reasoning or ""
        text = reply.text
        if thinking_token_budget is not None and self.tokens(reasoning) > thinking_token_budget:
            reasoning = reasoning[: int(thinking_token_budget * self._chars_per_token)]
            reasoning += self._budget_note
        used = self.tokens(reasoning) + self.tokens(text)
        finish = reply.finish_reason
        if used > max_tokens:
            room = int(max_tokens * self._chars_per_token)
            text = text[: max(0, room - len(reasoning))]
            reasoning = reasoning[:room]
            used = max_tokens
            finish = "length"
        prompt = self.prompt_tokens(messages)
        seconds = time.perf_counter() - start
        if self._timing is not None:
            self._timing.record(task_id, ENGINE_OP, seconds, used)
        return Completion(
            text=text,
            reasoning=reasoning or None,
            prompt_tokens=prompt,
            completion_tokens=used,
            finish_reason=finish,
            seconds=seconds,
        )
