"""Tests for ``arc26.agent.toolshot``: the "think with Python" tool loop (task P5.3).

Written from ``docs/tasks/P5.3.md`` and the design's "Recommendation" section, before the
implementation. The model is a scripted fake ``generate``; the tokenizer is character-level with
Qwen's markers as single ids, so continuation ids can be compared exactly. The REPL and the
verifier are the real CPU sandboxes.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from arc26.agent import toolshot
from arc26.agent.prompts import DIRECT_TWO_SYSTEM_PROMPT
from arc26.agent.types import AgentTask, Pair

BUDGET_MESSAGE = "Tool budget used: write your answer blocks now."
TOOL_TURN_HEAD = "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
TOOL_TURN_TAIL = "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
CAP = 65_536
# Alternative ids: ``ALT + ord(c)`` decodes to ``c`` but ``encode`` never yields it, so a loop
# that re-tokenizes the engine's text instead of keeping its ids breaks the continuation.
ALT = 2_000_000
FORCING_TOKENS = 6_144


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokenizer: one id per character, Qwen's markers as single ids."""

    SPECIAL = {
        "<think>": 1_000_001, "</think>": 1_000_002, "<|im_start|>": 1_000_003,
        "<|im_end|>": 1_000_004, "<|endoftext|>": 1_000_005, "<tool_call>": 1_000_006,
        "</tool_call>": 1_000_007, "<tool_response>": 1_000_008, "</tool_response>": 1_000_009,
    }
    _SPLIT = re.compile("(" + "|".join(re.escape(k) for k in SPECIAL) + ")")

    def __init__(self) -> None:
        self.render_calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in self._SPLIT.split(text):
            if part in self.SPECIAL:
                ids.append(self.SPECIAL[part])
            elif part:
                ids.extend(ord(c) for c in part)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        inverse = {v: k for k, v in self.SPECIAL.items()}
        return "".join(
            inverse[t] if t in inverse else chr(t - ALT if t >= ALT else t) for t in ids)

    def render(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        self.render_calls.append(([dict(m) for m in messages], dict(kwargs)))
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        return "".join(parts) + "<|im_start|>assistant\n<think>\n"


@dataclass
class FakeGen:
    """What the fake engine returns, in the shape of ``fn_probe.GenResult``."""

    text: str
    ids: list[int]
    finish: str
    completion_tokens: int
    cached_tokens: int | None = None

    def row(self) -> dict[str, Any]:
        return {"finish": self.finish, "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens}


@dataclass
class Step:
    """One scripted reply: its full text, finish reason, and the wall seconds it takes."""

    text: str
    finish: str = "stop"
    seconds: float = 1.0


@dataclass
class Request:
    input_ids: list[int]
    max_new_tokens: int
    cut_at: float
    kind: str
    extra: dict[str, Any]


@dataclass
class Clock:
    t: float = 0.0

    def __call__(self) -> float:
        return self.t


@dataclass
class FakeEngine:
    """A scripted ``generate``: each call pops one :class:`Step` and records the request.

    It applies the request's stop strings (kept in the output, as ``no_stop_trim``), the
    ``max_new_tokens`` limit (finish ``length``), and the wall cut: a step whose end passes
    ``cut_at`` returns its first half with finish ``cancelled``.
    """

    tok: CharTokenizer
    script: list[Step]
    clock: Clock = field(default_factory=Clock)
    requests: list[Request] = field(default_factory=list)
    replies: list[FakeGen] = field(default_factory=list)
    overflow: int = 0
    alt_ids: bool = False

    def __call__(self, input_ids: list[int], max_new_tokens: int, cut_at_wall_time: float,
                 kind: str, extra_sampling: Mapping[str, Any]) -> FakeGen:
        self.requests.append(Request(list(input_ids), int(max_new_tokens),
                                     float(cut_at_wall_time), kind, dict(extra_sampling or {})))
        if self.script:
            step = self.script.pop(0)
        else:
            self.overflow += 1
            step = Step("</think>\n\n")
        text, finish = step.text, step.finish
        for stop in (extra_sampling or {}).get("stop", []) or []:
            at = text.find(stop)
            if at >= 0 and at + len(stop) < len(text):
                text, finish = text[: at + len(stop)], "stop"
        ids = self.tok.encode(text)
        if self.alt_ids:
            ids = [t + ALT if t < 1_000_000 else t for t in ids]
        if self.clock.t + step.seconds >= cut_at_wall_time:
            self.clock.t = max(self.clock.t, cut_at_wall_time)
            ids, finish = ids[: len(ids) // 2], "cancelled"
        else:
            self.clock.t += step.seconds
        if len(ids) > max_new_tokens:
            ids, finish = ids[:max_new_tokens], "length"
        gen = FakeGen(self.tok.decode(ids), ids, finish, len(ids), 0)
        self.replies.append(gen)
        return gen

    def tool_reply(self, k: int) -> str:
        """Return the tool reply that request ``k + 1`` appends after request ``k``'s output."""
        before = self.requests[k].input_ids + self.replies[k].ids
        after = self.requests[k + 1].input_ids
        assert after[: len(before)] == before, "the next request does not continue the last"
        suffix = self.tok.decode(after[len(before):])
        assert suffix.startswith(TOOL_TURN_HEAD), suffix[:200]
        assert suffix.endswith(TOOL_TURN_TAIL), suffix[-200:]
        return suffix[len(TOOL_TURN_HEAD): len(suffix) - len(TOOL_TURN_TAIL)]


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def make_task(train: Sequence[tuple[list[list[int]], list[list[int]]]],
              tests: Sequence[list[list[int]]], task_id: str = "synth001") -> AgentTask:
    """Build a synthetic task from (input, output) pairs and test inputs."""
    return AgentTask(task_id=task_id,
                     train=tuple(Pair(input=i, output=o) for i, o in train),
                     test_inputs=tuple(tests))


def inc_task(tests: Sequence[list[list[int]]] | None = None) -> AgentTask:
    """Two demonstrations of "add 1 to every cell, mod 10" (k = 2)."""
    return make_task(
        [([[0, 1], [2, 3]], [[1, 2], [3, 4]]), ([[5, 9, 7]], [[6, 0, 8]])],
        tests if tests is not None else [[[7, 8]]],
    )


INC = "def transform(grid):\n    return (np.asarray(grid) + 1) % 10"
IDENTITY = "def transform(grid):\n    return np.asarray(grid)"


def call(code: str, think: str = "Let me test a rule.", closed: bool = True) -> str:
    """A reply that ends with one ``python`` call in the template's qwen3_coder form."""
    body = ("<tool_call>\n<function=python>\n<parameter=code>\n"
            f"{code}\n</parameter>\n</function>\n</tool_call>")
    return f"{think}\n</think>\n\n{body}" if closed else f"{think}\n{body}"


def answer(*blocks: tuple[int, list[list[int]]], think: str = "Done.") -> str:
    """A final reply with ``direct_two`` answer blocks."""
    text = "\n".join(
        f"```answer\n{json.dumps({'test_index': i, 'grid': g})}\n```" for i, g in blocks)
    return f"{think}\n</think>\n\n{text}"


def norm(value: Any) -> Any:
    """Grids and lists of grids as plain nested lists of ints."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    return value


def cfg(**changes: Any) -> Any:
    """The default ``tool:`` config, optionally with fields replaced."""
    base = toolshot.tool_config({})
    return dataclasses.replace(base, **changes) if changes else base


def run(task: AgentTask, script: list[Step], *, force_at: float = 1e9, end: float = 2e9,
        config: Any = None, alt_ids: bool = False,
        **kwargs: Any) -> tuple[Any, FakeEngine, CharTokenizer]:
    tok = CharTokenizer()
    engine = FakeEngine(tok, list(script), alt_ids=alt_ids)
    out = toolshot.solve_with_tools(task, tok, engine, config if config is not None else cfg(),
                                    force_at=force_at, end=end, now=engine.clock, **kwargs)
    return out, engine, tok


def statuses(out: Any) -> list[str]:
    return [row["status"] for row in out.calls]


def kinds(out: Any) -> list[str]:
    return [row["kind"] for row in out.turns]


# --------------------------------------------------------------------------------------------
# Config and prompt
# --------------------------------------------------------------------------------------------


def test_tool_config_defaults_carry_the_budgets() -> None:
    values = set(dataclasses.asdict(cfg()).values())
    for budget in (CAP, 86_016, 40, 16_384, FORCING_TOKENS, 2_000):
        assert budget in values, budget


def test_prompt_is_direct_two_with_the_python_tool() -> None:
    out, engine, tok = run(inc_task(), [Step(answer((0, [[8, 9]])))])
    messages, kwargs = tok.render_calls[0]
    assert "python" in repr(kwargs.get("tools"))
    system = next(m["content"] for m in messages if m["role"] == "system")
    sentence = "There is no code interpreter: reason, then answer."
    assert DIRECT_TWO_SYSTEM_PROMPT.count(sentence) == 1
    head, tail = DIRECT_TWO_SYSTEM_PROMPT.split(sentence)
    assert system.startswith(head)
    assert system.endswith(tail.lstrip())
    assert "no code interpreter" not in system
    c = cfg()
    paragraph = toolshot.TOOL_PARAGRAPH.format(timeout=c.call_timeout_s,
                                               chars=f"{c.reply_chars:,}")
    assert paragraph.strip() in system
    first = engine.requests[0]
    assert tok.decode(first.input_ids).endswith("<|im_start|>assistant\n<think>\n")
    assert first.kind == "think"
    assert first.max_new_tokens <= CAP
    assert "</tool_call>" in first.extra["stop"]
    assert first.extra.get("no_stop_trim") is True


def test_no_call_final_answer_only() -> None:
    out, engine, _ = run(inc_task(), [Step(answer((0, [[8, 9]]), (0, [[1, 1]])))])
    assert len(engine.requests) == 1
    assert list(out.calls) == []
    assert norm(out.attempts) == [[[[8, 9]], [[1, 1]]]]
    assert out.program is None
    assert out.forced is False


# --------------------------------------------------------------------------------------------
# Calls
# --------------------------------------------------------------------------------------------


def test_one_call() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("print(len(train), len(test_inputs), np.__name__)")),
        Step(answer((0, [[8, 9]]))),
    ])
    assert len(engine.requests) == 2
    assert engine.requests[1].kind == "tool"
    assert engine.tool_reply(0).strip() == "2 1 numpy"
    assert statuses(out) == ["ok"]
    assert kinds(out) == ["think", "tool"]
    assert "2 1 numpy" in out.text
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_many_calls_repl_state_kept() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("a = 41")),
        Step(call("b = a + 1")),
        Step(call("print(b)")),
        Step(answer((0, [[8, 9]]))),
    ])
    assert statuses(out) == ["ok", "ok", "ok"]
    assert engine.tool_reply(2).strip() == "42"
    assert len(out.turns) == 4


def test_continuation_only_the_reply_is_new() -> None:
    out, engine, tok = run(inc_task(), [
        Step(call("print('one')")),
        Step(call("print('two')")),
        Step(answer((0, [[8, 9]]))),
    ])
    used = 0
    for k in range(len(engine.requests) - 1):
        # tool_reply asserts: next ids = previous ids + generated ids + the tool-turn tokens.
        reply = engine.tool_reply(k)
        assert reply.strip() in {"one", "two"}
        used += engine.replies[k].completion_tokens
        nxt = engine.requests[k + 1]
        assert nxt.max_new_tokens <= CAP - used
        assert nxt.max_new_tokens <= 86_016 - len(nxt.input_ids)
        assert "</tool_call>" in nxt.extra["stop"]
    assert out.completion_tokens == sum(r.completion_tokens for r in engine.replies)


def test_call_leaked_inside_reasoning_is_run() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("print(6 * 7)", closed=False)),
        Step(answer((0, [[8, 9]]))),
    ])
    assert "</think>" not in engine.replies[0].text
    assert engine.tool_reply(0).strip() == "42"
    assert statuses(out) == ["ok"]


def test_call_without_closing_tag_is_run() -> None:
    open_call = "Try.\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(6 * 7)\n"
    out, engine, tok = run(inc_task(), [Step(open_call), Step(answer((0, [[8, 9]])))])
    assert len(engine.requests) == 2
    assert statuses(out) == ["ok"]
    later = tok.decode(engine.requests[1].input_ids)
    assert "<tool_response>\n42" in later


def test_code_timeout() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("while True:\n    pass")),
        Step(call("print('alive')")),
        Step(answer((0, [[8, 9]]))),
    ])
    assert statuses(out)[0] == "timeout"
    assert "timeout" in engine.tool_reply(0).lower()
    assert out.calls[0]["seconds"] < 30
    # the REPL answers again after the timeout
    assert "alive" in engine.tool_reply(1)


def test_exception_reply_ends_with_it() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("print('before')\nraise ValueError('boom')")),
        Step(answer((0, [[8, 9]]))),
    ])
    assert statuses(out) == ["error"]
    reply = engine.tool_reply(0).rstrip()
    assert "before" in reply
    assert reply.splitlines()[-1].endswith("ValueError: boom")


def test_reply_clipped_to_2000_chars() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("print('x' * 5000)")),
        Step(answer((0, [[8, 9]]))),
    ])
    reply = engine.tool_reply(0)
    assert len(reply) <= 2_000
    assert "characters cut" in reply


def test_call_budget_40() -> None:
    script = [Step(call(f"print({i})")) for i in range(43)] + [Step(answer((0, [[8, 9]])))]
    out, engine, _ = run(inc_task(), script)
    st = statuses(out)
    assert st[:40] == ["ok"] * 40
    assert st[40:] and all(s == "budget" for s in st[40:])
    assert engine.tool_reply(39).strip() == "39"
    assert engine.tool_reply(40).strip() == BUDGET_MESSAGE
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_reply_token_budget_16384() -> None:
    script = [Step(call(f"print('{i}' * 5000)")) for i in range(12)]
    script.append(Step(answer((0, [[8, 9]]))))
    out, engine, _ = run(inc_task(), script)
    st = statuses(out)
    assert "budget" in st
    first = st.index("budget")
    # replies are clipped to 2,000 characters (one token each here): 16,384 lasts 7 to 9 calls
    assert 7 <= first <= 9
    assert all(s == "budget" for s in st[first:])
    assert engine.tool_reply(first).strip() == BUDGET_MESSAGE
    assert out.tool_reply_tokens <= 16_384 + 2_100
    assert len(st) == 12


def test_cap_reached_inside_a_call_forces_and_forced_call_not_run() -> None:
    long_think = "z" * (CAP - 40)
    cut_call = long_think + "\n<tool_call>\n<function=python>\n<parameter=code>\n" + "y" * 200
    forced_reply = (answer((0, [[8, 9]]), think="")
                    + "\n" + call("print('RAN_IN_FORCE')", think="", closed=False))
    out, engine, tok = run(inc_task(), [Step(cut_call), Step(forced_reply)])
    assert engine.replies[0].finish == "length"
    assert engine.replies[0].completion_tokens <= CAP
    assert out.forced is True
    assert kinds(out)[-1] == "force"
    assert engine.requests[-1].kind == "force"
    assert engine.requests[-1].max_new_tokens <= FORCING_TOKENS
    assert len(engine.requests) == 2
    assert engine.overflow == 0
    assert all(s != "ok" for s in statuses(out))
    assert out.text.count("RAN_IN_FORCE") == 1  # only the call's source, never its output
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_forced_reply_call_not_run_after_tool_use() -> None:
    forced_reply = (answer((0, [[8, 9]]), think="")
                    + "\n" + call("print('X' + 'Y')", think="", closed=False))
    script = [Step(call("print('fine')"), seconds=1.0),
              Step("thinking " * 50, seconds=200.0),
              Step(forced_reply)]
    out, engine, tok = run(inc_task(), script, force_at=100.0, end=300.0)
    assert statuses(out) == ["ok"]
    assert out.forced is True
    assert kinds(out) == ["think", "tool", "force"]
    assert "XY" not in out.text
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_force_time() -> None:
    out, engine, tok = run(inc_task(), [
        Step("I keep thinking " * 100, seconds=150.0),
        Step(answer((0, [[8, 9]]), think="")),
    ], force_at=100.0, end=300.0)
    assert engine.requests[0].cut_at == pytest.approx(100.0)
    assert engine.replies[0].finish == "cancelled"
    force = engine.requests[1]
    assert force.kind == "force"
    assert force.cut_at == pytest.approx(300.0)
    assert force.max_new_tokens <= FORCING_TOKENS
    before = engine.requests[0].input_ids + engine.replies[0].ids
    assert force.input_ids[: len(before)] == before
    assert "</think>" in tok.decode(force.input_ids[len(before):])
    assert out.forced is True
    assert kinds(out) == ["think", "force"]
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_tool_turn_cut_at_force_time() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("print(1)"), seconds=120.0),
        Step(answer((0, [[8, 9]]))),
    ], force_at=150.0, end=300.0)
    assert statuses(out) == ["ok"]
    assert [r.kind for r in engine.requests] == ["think", "tool"]
    assert engine.requests[1].cut_at == pytest.approx(150.0)
    assert out.forced is False
    assert norm(out.attempts) == [[[[8, 9]]]]


# --------------------------------------------------------------------------------------------
# check(fn)
# --------------------------------------------------------------------------------------------


def check_task() -> AgentTask:
    return make_task(
        [([[3]], [[3]]), ([[1, 1], [1, 1]], [[1, 2], [1, 1]]), ([[4, 4]], [[4], [4]])],
        [[[5]]])


def test_check_pass_fail_and_same_shape_miss() -> None:
    out, engine, _ = run(check_task(), [
        Step(call(IDENTITY + "\ncheck(transform)")),
        Step(answer((0, [[5]]))),
    ])
    reply = engine.tool_reply(0).lower()
    assert "pass" in reply
    assert "fail" in reply
    assert "1/2" in reply  # yours/expected for the one wrong cell of the same-shape miss
    assert statuses(out) == ["ok"]


def test_check_all_pass() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call(INC + "\ncheck(transform)")),
        Step(answer((0, [[8, 9]]))),
    ])
    reply = engine.tool_reply(0).lower()
    assert "pass" in reply
    assert "fail" not in reply


def test_check_function_that_raises() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call("def transform(grid):\n    raise RuntimeError('nope')\ncheck(transform)")),
        Step(call("print('still here')")),
        Step(answer((0, [[8, 9]]))),
    ])
    assert "fail" in engine.tool_reply(0).lower() or "nope" in engine.tool_reply(0)
    assert "still here" in engine.tool_reply(1)


# --------------------------------------------------------------------------------------------
# Program slot and answer slots (A1, P, A2)
# --------------------------------------------------------------------------------------------


def test_verified_transform_fills_a_slot() -> None:
    out, _, _ = run(inc_task(), [
        Step(call(INC)),
        Step(answer((0, [[1, 1]]), (0, [[2, 2]]))),
    ])
    assert out.program is not None
    assert norm(out.direct) == [[[[1, 1]], [[2, 2]]]]
    # (A1, P, A2): A1 first, the verified program's output second
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_unverified_transform_does_not_fill_a_slot() -> None:
    out, _, _ = run(inc_task(), [
        Step(call(IDENTITY)),
        Step(answer((0, [[1, 1]]), (0, [[2, 2]]))),
    ])
    assert out.program is None
    assert norm(out.attempts) == [[[[1, 1]], [[2, 2]]]]


def test_failing_transform_with_exception_does_not_fill_a_slot() -> None:
    out, _, _ = run(inc_task(), [
        Step(call("def transform(grid):\n    raise ValueError('x')")),
        Step(answer((0, [[1, 1]]))),
    ])
    assert out.program is None
    assert norm(out.attempts) == [[[[1, 1]]]]


def test_program_joins_definitions_across_calls() -> None:
    out, _, _ = run(inc_task(), [
        Step(call("def inc(g):\n    return (np.asarray(g) + 1) % 10")),
        Step(call("def transform(grid):\n    return inc(grid)")),
        Step(answer((0, [[1, 1]]))),
    ])
    assert out.program is not None
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_program_last_definition_of_a_name_wins() -> None:
    out, _, _ = run(inc_task(), [
        Step(call("def step(g):\n    return np.asarray(g)\n"
                  "def transform(grid):\n    return step(grid)")),
        Step(call("def step(g):\n    return (np.asarray(g) + 1) % 10")),
        Step(call("def transform(grid):\n    return step(grid)")),
        Step(answer((0, [[1, 1]]))),
    ])
    assert out.program is not None
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_program_is_the_last_verified_transform() -> None:
    dec = "def transform(grid):\n    return (np.asarray(grid) + 1) % 10\n# v2"
    out, _, _ = run(inc_task(), [
        Step(call(INC)),
        Step(call(IDENTITY)),
        Step(call(dec)),
        Step(answer((0, [[1, 1]]))),
    ])
    assert out.program is not None
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_slot_dedupe_p_equals_a1() -> None:
    out, _, _ = run(inc_task(), [
        Step(call(INC)),
        Step(answer((0, [[8, 9]]), (0, [[2, 2]]))),
    ])
    assert norm(out.attempts) == [[[[8, 9]], [[2, 2]]]]


def test_slot_dedupe_p_equals_a2() -> None:
    out, _, _ = run(inc_task(), [
        Step(call(INC)),
        Step(answer((0, [[1, 1]]), (0, [[8, 9]]))),
    ])
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_slot_dedupe_a1_equals_a2() -> None:
    out, _, _ = run(inc_task(), [Step(answer((0, [[1, 1]]), (0, [[1, 1]])))])
    assert norm(out.attempts) == [[[[1, 1]]]]


def test_slot_program_only_when_no_direct_answer() -> None:
    out, _, _ = run(inc_task(), [Step(call(INC)), Step("No answer.\n</think>\n\nSorry.")])
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_slots_two_test_inputs() -> None:
    task = inc_task(tests=[[[7, 8]], [[0, 0, 9]]])
    out, _, _ = run(task, [
        Step(call(INC)),
        Step(answer((0, [[1, 1]]), (1, [[1, 1, 0]]), (0, [[2, 2]]), (1, [[5, 5, 5]]))),
    ])
    # input 0: (A1, P, A2) -> A1, P; input 1: P equals A1, so A1, A2
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]], [[[1, 1, 0]], [[5, 5, 5]]]]


def test_slots_two_test_inputs_dedupe_per_input() -> None:
    task = inc_task(tests=[[[7, 8]], [[0, 0, 9]]])
    out, _, _ = run(task, [
        Step(call(INC)),
        Step(answer((0, [[8, 9]]), (1, [[3, 3, 3]]))),
    ])
    # input 0: A1 == P, one slot; input 1: A1 then P
    assert norm(out.attempts) == [[[[8, 9]]], [[[3, 3, 3]], [[1, 1, 0]]]]


def test_model_answer_never_replaced_by_placeholder() -> None:
    out, _, _ = run(inc_task(), [Step(call(IDENTITY)), Step(answer((0, [[4, 4]])))])
    flat = norm(out.attempts)
    assert flat == [[[[4, 4]]]]
    assert [[0]] not in flat[0]


def test_no_answer_gives_no_attempts_not_placeholder() -> None:
    out, _, _ = run(inc_task(), [Step("Nothing.\n</think>\n\nNo idea.")])
    assert norm(out.attempts) in ([[]], [])


# --------------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------------


def test_30x30_grids_verified_and_clipped() -> None:
    grid_in = [[(r + c) % 10 for c in range(30)] for r in range(30)]
    grid_out = [[(v + 1) % 10 for v in row] for row in grid_in]
    test_in = [[(r * c) % 10 for c in range(30)] for r in range(30)]
    test_out = [[(v + 1) % 10 for v in row] for row in test_in]
    task = make_task([(grid_in, grid_out), (grid_out, [[(v + 1) % 10 for v in row]
                                                       for row in grid_out])], [test_in])
    out, engine, _ = run(task, [
        Step(call("print(train[0][0].tolist())\n" + INC + "\ncheck(transform)")),
        Step(answer((0, grid_in))),
    ])
    assert len(engine.tool_reply(0)) <= 2_000
    assert out.program is not None
    assert norm(out.attempts) == [[grid_in, test_out]]


def test_turn_rows_and_call_rows_shape() -> None:
    out, _, _ = run(inc_task(), [Step(call("print(1)")), Step(answer((0, [[8, 9]])))])
    assert len(out.turns) == 2
    assert {"kind"} <= set(out.turns[0])
    row = out.calls[0]
    assert {"status", "seconds"} <= set(row)
    assert out.tool_reply_tokens > 0
    assert out.completion_tokens > 0
    assert isinstance(out.errors, list)
    assert out.ended_by


# --------------------------------------------------------------------------------------------
# Review round 2: behaviors a broken variant must not pass
# --------------------------------------------------------------------------------------------


def force_suffix(engine: FakeEngine, tok: CharTokenizer) -> str:
    """The text the forcing request appends after the previous request's ids and output."""
    before = engine.requests[-2].input_ids + engine.replies[-2].ids
    force = engine.requests[-1]
    assert force.kind == "force"
    assert force.input_ids[: len(before)] == before, "the forcing turn does not continue"
    return tok.decode(force.input_ids[len(before):])


def test_continuation_keeps_engine_ids_not_retokenized_text() -> None:
    # (a) the engine's ids differ from encode(text): only a loop that keeps result.ids passes.
    out, engine, tok = run(inc_task(), [
        Step(call("print('one')")),
        Step(call("print('two')")),
        Step("still thinking " * 20, seconds=500.0),
        Step(answer((0, [[8, 9]]), think="")),
    ], force_at=100.0, end=1_000.0, alt_ids=True)
    assert tok.encode(engine.replies[0].text) != engine.replies[0].ids
    assert engine.tool_reply(0).strip() == "one"
    assert engine.tool_reply(1).strip() == "two"
    assert force_suffix(engine, tok).endswith("</think>\n\n")
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_guards_reject_hardcoded_transform() -> None:
    # (b) passes both demonstrations by lookup: the guards (policy on) must reject it.
    hardcoded = (
        "def transform(grid):\n"
        "    g = np.asarray(grid).tolist()\n"
        "    if g == [[0, 1], [2, 3]]:\n"
        "        return [[1, 2], [3, 4]]\n"
        "    if g == [[5, 9, 7]]:\n"
        "        return [[6, 0, 8]]\n"
        "    return [[8, 9]]"
    )
    out, engine, _ = run(inc_task(), [
        Step(call(hardcoded + "\ncheck(transform)")),
        Step(answer((0, [[1, 1]]), (0, [[2, 2]]))),
    ])
    assert "fail" not in engine.tool_reply(0).lower()  # it does reproduce the demonstrations
    assert out.program is None
    assert norm(out.attempts) == [[[[1, 1]], [[2, 2]]]]


def test_latest_of_two_passing_transforms_wins() -> None:
    # (c) both pass the demonstrations (no 4 in them) but differ on the test input [[4, 6]].
    first = INC
    second = ("def transform(grid):\n    g = np.asarray(grid)\n"
              "    return np.where(g == 4, 0, (g + 1) % 10)")
    task = inc_task(tests=[[[4, 6]]])
    out, _, _ = run(task, [
        Step(call(first)),
        Step(call(second)),
        Step(answer((0, [[1, 1]]))),
    ])
    assert out.program is not None
    assert norm(out.attempts) == [[[[1, 1]], [[0, 7]]]]

    out, _, _ = run(task, [
        Step(call(second)),
        Step(call(first)),
        Step(answer((0, [[1, 1]]))),
    ])
    assert norm(out.attempts) == [[[[1, 1]], [[5, 7]]]]


def test_forcing_note_only_inside_unclosed_thinking() -> None:
    # (d) cut after </think>, inside the answer: nothing is appended before the forced reply.
    c = cfg()
    partial = "Rule found.\n</think>\n\n```answer\n" + "{" * 400
    out, engine, tok = run(inc_task(), [
        Step(partial, seconds=150.0),
        Step(answer((0, [[8, 9]]), think="")),
    ], force_at=100.0, end=300.0)
    assert "</think>" in engine.replies[0].text
    suffix = force_suffix(engine, tok)
    assert c.thinking_end_note.strip() not in suffix
    assert "</think>" not in suffix
    assert out.forced is True


def test_forcing_note_in_a_later_turns_unclosed_thinking() -> None:
    # (d) the first turn closed its thinking before its call; the cut turn did not: note and tag.
    c = cfg()
    out, engine, tok = run(inc_task(), [
        Step(call("print(1)")),
        Step("more thinking " * 40, seconds=150.0),
        Step(answer((0, [[8, 9]]), think="")),
    ], force_at=100.0, end=300.0)
    assert force_suffix(engine, tok) == c.thinking_end_note + c.forcing_suffix
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_non_def_statements_never_raise() -> None:
    # (e) assignments, expressions, imports and classes among the cells; the program still works.
    out, _, _ = run(inc_task(), [
        Step(call("k = 3; print(k)")),
        Step(call("k * 2")),
        Step(call("import math\nprint(math.pi > 3)")),
        Step(call("class Box:\n    size = 2")),
        Step(call(INC)),
        Step(answer((0, [[1, 1]]), (0, [[2, 2]]))),
    ])
    assert norm(out.direct) == [[[[1, 1]], [[2, 2]]]]
    assert norm(out.attempts)[0][0] == [[1, 1]]


class RaisingVerifier:
    """A verifier that fails in every call."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def verify(self, task: Any, source: str) -> Any:
        raise self.error

    def close(self) -> None:
        pass


@pytest.mark.parametrize("error", [RuntimeError("worker died"), ValueError("bad"),
                                   OSError("pipe"), TimeoutError("slow")])
def test_verification_failure_keeps_direct_answers(error: BaseException) -> None:
    # (e) whatever the verifier raises, the outcome is returned with the model's answers.
    out, _, _ = run(inc_task(), [
        Step(call("k = 3\n" + INC)),
        Step(answer((0, [[1, 1]]), (0, [[2, 2]]))),
    ], verifier=RaisingVerifier(error))
    assert out.program is None
    assert norm(out.attempts) == [[[[1, 1]], [[2, 2]]]]


def test_cap_inside_call_after_think_close_forces_both_answers() -> None:
    # (f) the cap cuts a call written after </think>: not run; closed call, budget reply, note.
    c = cfg()
    # the call starts below the cap minus the forcing room and runs past the cap
    cut_call = ("z" * (CAP - FORCING_TOKENS - 3_000) + "\n</think>\n\n<tool_call>\n"
                "<function=python>\n<parameter=code>\nprint('RAN' + 'X')\n" + "y" * 20_000)
    out, engine, tok = run(inc_task(), [
        Step(cut_call),
        Step(answer((0, [[1, 1]]), (0, [[2, 2]]), think="")),
    ])
    assert engine.replies[0].finish == "length"
    assert len(engine.requests) == 2
    assert all(s != "ok" for s in statuses(out))
    assert "RANX" not in out.text
    suffix = force_suffix(engine, tok)
    assert "</tool_call>" in suffix
    assert "<tool_response>" in suffix and c.budget_message in suffix
    assert suffix.endswith(c.thinking_end_note + c.forcing_suffix)
    assert suffix.index("</tool_call>") < suffix.index(c.budget_message)
    assert norm(out.direct) == [[[[1, 1]], [[2, 2]]]]
    assert norm(out.attempts) == [[[[1, 1]], [[2, 2]]]]
    force = engine.requests[-1]
    assert out.completion_tokens <= CAP
    assert len(force.input_ids) + force.max_new_tokens <= c.context_tokens
    assert force.max_new_tokens > 0


def test_context_cut_inside_call_after_think_close_forces_both_answers() -> None:
    # (f) the same with the context bound reached first.
    c = cfg(context_tokens=30_000)
    cut_call = ("z" * 10_000 + "\n</think>\n\n<tool_call>\n<function=python>\n"
                "<parameter=code>\nprint('RAN' + 'X')\n" + "y" * 30_000)
    out, engine, tok = run(inc_task(), [
        Step(cut_call),
        Step(answer((0, [[1, 1]]), (0, [[2, 2]]), think="")),
    ], config=c)
    assert engine.replies[0].finish == "length"
    assert len(engine.requests) == 2
    assert "RANX" not in out.text
    suffix = force_suffix(engine, tok)
    assert c.budget_message in suffix
    assert suffix.endswith(c.thinking_end_note + c.forcing_suffix)
    force = engine.requests[-1]
    assert len(force.input_ids) + force.max_new_tokens <= c.context_tokens
    assert force.max_new_tokens > 0
    assert norm(out.attempts) == [[[[1, 1]], [[2, 2]]]]


def test_answer_turn_with_function_mention_is_final() -> None:
    # (g) the answer blocks end the task; a later "<function=" mention is not a call.
    text = (answer((0, [[1, 1]]), (0, [[2, 2]]))
            + "\nI could have used <function=python> to check, but the answer stands.")
    out, engine, _ = run(inc_task(), [Step(text), Step(call("print('late')"))])
    assert len(engine.requests) == 1
    assert list(out.calls) == []
    assert norm(out.attempts) == [[[[1, 1]], [[2, 2]]]]


def test_repl_starts_with_numpy_under_1_gib() -> None:
    # (h) the REPL starts and imports numpy with a 1 GiB address-space cap.
    out, engine, _ = run(inc_task(), [
        Step(call("import numpy\nprint(int(numpy.zeros(3).sum()) + len(train))")),
        Step(answer((0, [[8, 9]]))),
    ], config=cfg(repl_memory_bytes=1 << 30))
    assert statuses(out) == ["ok"]
    assert engine.tool_reply(0).strip() == "2"
    assert not [e for e in out.errors if "sandbox" in e.lower()]


# --------------------------------------------------------------------------------------------
# Review round 3
# --------------------------------------------------------------------------------------------


@pytest.fixture
def repl_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count the cells the task's REPL session is asked to run."""
    seen: list[str] = []
    original = toolshot._Session.run

    def counting(self: Any, code: str, timeout_s: float) -> Any:
        seen.append(code)
        return original(self, code, timeout_s)

    monkeypatch.setattr(toolshot._Session, "run", counting)
    return seen


CALL_ONLY = ("<tool_call>\n<function=python>\n<parameter=code>\nprint('FORCED' + 'RUN')\n"
             "</parameter>\n</function>\n</tool_call>")


def test_call_only_forced_reply_not_run_no_answer(repl_runs: list[str]) -> None:
    out, engine, _ = run(inc_task(), [
        Step("thinking " * 100, seconds=150.0),
        Step(CALL_ONLY),
        Step(blocks((0, [[8, 9]]))),
    ], force_at=100.0, end=300.0)
    # P5.8: the call is stripped, never run, and a second forcing request answers.
    assert [r.kind for r in engine.requests] == ["think", "force", "force2"]
    assert repl_runs == []
    assert all(s != "ok" for s in statuses(out))
    assert "FORCEDRUN" not in out.text
    assert norm(out.attempts) == [[[[8, 9]]]]
    assert out.forced is True


def test_call_only_forced_reply_not_run_keeps_earlier_answers(repl_runs: list[str]) -> None:
    # one call runs; the next turn writes both answers and is cut after them at the force time
    answered = answer((0, [[1, 1]]), (0, [[2, 2]])) + "\nChecking once more. " + "w" * 400
    out, engine, _ = run(inc_task(), [
        Step(call("print('early')")),
        Step(answered, seconds=150.0),
        Step(CALL_ONLY),
        Step(answer((0, [[8, 9]]))),
    ], force_at=100.0, end=300.0)
    assert [r.kind for r in engine.requests] == ["think", "tool", "force"]
    assert len(repl_runs) == 1 and "early" in repl_runs[0]
    assert statuses(out) == ["ok"]
    assert "FORCEDRUN" not in out.text
    assert norm(out.direct) == [[[[1, 1]], [[2, 2]]]]
    assert norm(out.attempts) == [[[[1, 1]], [[2, 2]]]]


BIG = "np.empty(200_000_000)"  # 1.6 GB of address space: over 1 GiB, under the 2 GiB default


def test_repl_memory_cap_1_gib_enforced() -> None:
    assert cfg().repl_memory_bytes == 1 << 30
    out, engine, _ = run(inc_task(), [
        Step(call(f"x = {BIG}\nprint('allocated')")),
        Step(call("print('after')")),
        Step(answer((0, [[8, 9]]))),
    ])
    reply = engine.tool_reply(0)
    assert "allocated" not in reply
    assert "memory" in reply.lower()
    assert statuses(out)[0] != "ok"
    assert "after" in engine.tool_reply(1)


def test_verifier_memory_cap_1_gib_enforced() -> None:
    big_inc = ("def transform(grid):\n"
               f"    scratch = {BIG}\n"
               "    scratch[0] = 1.0\n"
               "    return (np.asarray(grid) + 1) % 10")
    out, _, _ = run(inc_task(), [
        Step(call(big_inc)),
        Step(answer((0, [[1, 1]]))),
    ])
    assert out.program is None
    assert norm(out.attempts) == [[[[1, 1]]]]


# --------------------------------------------------------------------------------------------
# P5.8: a forced reply that writes a call; the program slot whatever the rule's name; refusals
# --------------------------------------------------------------------------------------------


def blocks(*items: tuple[int, list[list[int]]]) -> str:
    """Answer blocks alone, as a forced reply after ``</think>`` writes them."""
    return "\n".join(
        f"```answer\n{json.dumps({'test_index': i, 'grid': g})}\n```" for i, g in items)


def forced_call(code: str, summary: str = "Summary of the rule.\n\n") -> str:
    """A forced reply that writes a summary and then a call instead of answer blocks."""
    return (summary + "<tool_call>\n<function=python>\n<parameter=code>\n"
            f"{code}\n</parameter>\n</function>\n</tool_call><|im_end|>")


def test_forced_reply_with_a_call_is_stripped_and_forced_again() -> None:
    summary = "Summary of the rule.\n\n"
    out, engine, tok = run(inc_task(), [
        Step("I keep thinking " * 100, seconds=150.0),
        Step(forced_call("print('RAN_' + 'IN_FORCE')", summary)),
        Step(blocks((0, [[8, 9]]))),
    ], force_at=100.0, end=300.0, alt_ids=True)
    assert kinds(out) == ["think", "force", "force2"]
    assert out.forced is True
    assert out.calls == []
    assert "RAN_IN_FORCE" not in out.text
    assert engine.overflow == 0
    force, again = engine.requests[1], engine.requests[2]
    assert again.kind == "force2"
    assert again.max_new_tokens <= FORCING_TOKENS
    assert len(again.input_ids) + again.max_new_tokens <= toolshot.ToolConfig().context_tokens
    # token for token: the forcing request, the engine's own ids up to the call, then the note
    kept = len(summary)
    assert again.input_ids[: len(force.input_ids)] == force.input_ids
    assert again.input_ids[len(force.input_ids): len(force.input_ids) + kept] == (
        engine.replies[1].ids[:kept])
    assert tok.decode(again.input_ids[len(force.input_ids) + kept:]) == (
        toolshot.FORCE_AGAIN_NOTE)
    assert out.turns[1]["stripped_call_chars"] > 0
    assert "<tool_call>" not in out.text.split("</think>")[-1]
    assert norm(out.attempts) == [[[[8, 9]]]]
    assert out.completion_tokens == sum(r.completion_tokens for r in engine.replies)


def test_both_forcing_requests_fail_and_the_program_fills_the_slot() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call(INC + "\ncheck(transform)"), seconds=1.0),
        Step("thinking " * 50, seconds=200.0),
        Step(forced_call("print(1)")),
        Step(forced_call("print(2)", summary="")),
    ], force_at=100.0, end=300.0)
    assert kinds(out) == ["think", "tool", "force", "force2"]
    assert statuses(out) == ["ok"]
    assert engine.overflow == 0
    assert out.direct == [[]]
    assert out.program is not None and out.program["name"] == "transform"
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_forced_reply_with_answers_and_a_call_is_not_forced_again() -> None:
    out, engine, _ = run(inc_task(), [
        Step("I keep thinking " * 100, seconds=150.0),
        Step(blocks((0, [[8, 9]])) + "\n" + forced_call("print(1)", summary="")),
    ], force_at=100.0, end=300.0)
    assert kinds(out) == ["think", "force"]
    assert engine.overflow == 0
    assert norm(out.attempts) == [[[[8, 9]]]]


PLUS = "def plus_transform(grid):\n    return (np.asarray(grid) + 1) % 10"


def test_passing_checked_function_of_another_name_fills_the_program_slot() -> None:
    out, engine, _ = run(inc_task(), [
        Step(call(PLUS + "\ncheck(plus_transform)")),
        Step(answer((0, [[1, 1]]))),
    ])
    assert "for plus_transform" in engine.tool_reply(0)
    assert out.calls[0]["checked"] == [["plus_transform", 2, 2]]
    assert out.calls[0]["defines_transform"] is False
    assert out.program is not None
    assert out.program["name"] == "plus_transform"
    assert out.program["source"].endswith(
        "def transform(grid):\n    return plus_transform(grid)\n")
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_checked_function_that_fails_a_demo_does_not_fill_the_slot() -> None:
    out, _, _ = run(inc_task(), [
        Step(call("def ident(grid):\n    return np.asarray(grid)\ncheck(ident)")),
        Step(answer((0, [[1, 1]]))),
    ])
    assert out.program is None
    assert [(v["name"], v["verified"]) for v in out.verifications] == [("ident", False)]
    assert norm(out.attempts) == [[[[1, 1]]]]


def test_failing_transform_loses_to_a_passing_foo() -> None:
    foo = "def foo(grid):\n    return (np.asarray(grid) + 1) % 10"
    out, _, _ = run(inc_task(), [
        Step(call(IDENTITY + "\ncheck(transform)")),
        Step(call(foo + "\ncheck(foo)")),
        Step(answer((0, [[1, 1]]))),
    ])
    assert [row["checked"] for row in out.calls] == [[["transform", 0, 2]], [["foo", 2, 2]]]
    assert out.program is not None and out.program["name"] == "foo"
    assert [v["name"] for v in out.verifications] == ["transform", "foo"]
    assert [v["verified"] for v in out.verifications] == [False, True]
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_refused_answer_grid_is_logged_with_its_reason() -> None:
    ragged = "```answer\n" + json.dumps({"test_index": 0, "grid": [[1, 2], [3]]}) + "\n```"
    out, _, _ = run(inc_task(), [Step("Done.\n</think>\n\n" + ragged)])
    assert out.attempts == [[]]
    refused = [e for e in out.errors if e.startswith("answer: ")]
    assert len(refused) == 1
    assert "block 0" in refused[0] and "ragged" in refused[0]


def test_strip_keeps_whole_engine_tokens_before_the_call() -> None:
    pieces = ["Sum", "mary.", "\n\n<", "tool_call>", "\nprint(1)"]

    class PieceTok:
        def decode(self, ids: Sequence[int]) -> str:
            return "".join(pieces[i] for i in ids)

    reply = "".join(pieces)
    kept = toolshot._strip_forced_call("x</think>\n\n", reply, [0, 1, 2, 3, 4], PieceTok())
    # the token "\n\n<" crosses the call's start, so it is dropped with the call
    assert kept == ([0, 1], "Summary.")
    assert toolshot._strip_forced_call("x</think>\n\n", "No call here.", [0], PieceTok()) is None


# The three python calls of task 1818057f in P5.2-01 revision 20, verbatim from
# ~/wt-P5.5/runs/P5.2-01-r20/run_logs/fn/transcripts.jsonl (model output, no task data). The model
# tested plus_transform with its own np.array_equal loop and never called check.
R20_1818057F_CALLS = [
    '''\
import numpy as np

def plus_transform(g):
    g = np.array(g)
    out = g.copy()
    r,c = g.shape
    fg = 4
    for i in range(r):
        for j in range(c):
            if 1<=i<r-1 and 1<=j<c-1:
                if g[i,j]==fg and g[i-1,j]==fg and g[i+1,j]==fg and g[i,j-1]==fg and g[i,j+1]==fg:
                    out[i,j]=8
                    out[i-1,j]=8; out[i+1,j]=8; out[i,j-1]=8; out[i,j+1]=8
    return out

for k,(inp,out) in enumerate(train):
    mine = plus_transform(inp)
    print(k, np.array_equal(mine,out))
    if not np.array_equal(mine,out):
        d = np.argwhere(mine!=out)
        print(d[:20], [(mine[a,b],out[a,b]) for a,b in d[:20]])''',
    '''\
res = plus_transform(test_inputs[0])
print(res.shape)
for row in res: print(''.join(map(str,row)))
# how many pluses, and where
g = test_inputs[0]
r,c = g.shape
cnt=0
for i in range(1,r-1):
    for j in range(1,c-1):
        if all(g[i,j]==4 and g[a]==4 for a in (0,)) : pass
for i in range(1,r-1):
    for j in range(1,c-1):
        if g[i,j]==4 and g[i-1,j]==4 and g[i+1,j]==4 and g[i,j-1]==4 and g[i,j+1]==4:
            cnt+=1; print('plus center',(i,j))
print(cnt)''',
    '''\
g = test_inputs[0]
r,c = g.shape
centers=[]
for i in range(1,r-1):
    for j in range(1,c-1):
        if g[i,j]==4 and g[i-1,j]==4 and g[i+1,j]==4 and g[i,j-1]==4 and g[i,j+1]==4:
            centers.append((i,j))
print(centers)
res = plus_transform(g)
assert res.shape==(22,22)
assert all(len(row)==22 for row in res)
# changed cells must be 4->8 only
print(sorted(set(zip(g[res==8].tolist(), res[res==8].tolist()))))
print(set(np.unique(res)) )
grid=res.tolist()
print(grid[:2])''',
]
R20_TRANSCRIPTS = Path.home() / "wt-P5.5/runs/P5.2-01-r20/run_logs/fn/transcripts.jsonl"


def plus_rule(grid: list[list[int]]) -> list[list[int]]:
    """Every 4-plus (a 4 whose four neighbours are 4) becomes 8, centre and arms."""
    out = [row[:] for row in grid]
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[0]) - 1):
            cells = [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            if all(grid[a][b] == 4 for a, b in cells):
                for a, b in cells:
                    out[a][b] = 8
    return out


def plus_grid(size: int, centres: Sequence[tuple[int, int]],
              scattered: Sequence[tuple[int, int]]) -> list[list[int]]:
    grid = [[2] * size for _ in range(size)]
    for i, j in centres:
        for a, b in [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            grid[a][b] = 4
    for a, b in scattered:
        grid[a][b] = 4
    return grid


def test_replayed_1818057f_calls_fill_the_program_slot() -> None:
    if R20_TRANSCRIPTS.exists():
        for line in R20_TRANSCRIPTS.read_text().splitlines():
            row = json.loads(line)
            if row["task_id"] == "1818057f":
                parts = row["text"].split("<|im_start|>user\n<tool_response>")[:-1]
                seen = [toolshot.parse_call(p.split("</tool_response>")[-1]) for p in parts]
                assert [c.arguments["code"] for c in seen if c] == R20_1818057F_CALLS
    inputs = [plus_grid(8, [(2, 2), (5, 5)], [(0, 7), (7, 0)]),
              plus_grid(9, [(4, 4)], [(0, 0), (1, 7), (8, 8)]),
              plus_grid(10, [(1, 7), (6, 2), (7, 7)], [(0, 0), (9, 4)])]
    test_input = plus_grid(22, [(3, 3), (8, 15), (12, 6), (18, 18)], [(0, 21), (10, 10)])
    task = make_task([(g, plus_rule(g)) for g in inputs], [test_input], task_id="1818057f")
    out, _, _ = run(task, [Step(call(code)) for code in R20_1818057F_CALLS]
                    + [Step(answer((0, [[1]])))])
    assert all(row["ran"] for row in out.calls)
    assert not any(row["defines_transform"] for row in out.calls)
    assert all(row["checked"] is None for row in out.calls)
    assert out.program is not None and out.program["name"] == "plus_transform"
    assert out.program["demos_passed"] == 3
    assert norm(out.attempts) == [[[[1]], plus_rule(test_input)]]


def test_hash_dispatch_in_a_named_candidate_is_refused() -> None:
    cheat = ("def foo(grid):\n"
             "    if hash(np.asarray(grid).tobytes()) == hash(np.array([[5, 9, 7]]).tobytes()):\n"
             "        return np.array([[6, 0, 8]])\n"
             "    return (np.asarray(grid) + 1) % 10\n"
             "check(foo)")
    out, _, _ = run(inc_task(), [Step(call(cheat)), Step(answer((0, [[1, 1]])))])
    assert out.calls[0]["checked"] == [["foo", 2, 2]]
    assert out.program is None
    assert [v["name"] for v in out.verifications] == ["foo"]
    assert "input_hash_key" in out.verifications[0]["fatal"]
    assert norm(out.attempts) == [[[[1, 1]]]]


def test_program_is_verified_before_force2_and_force2_leaves_its_time() -> None:
    task = inc_task()
    end = 300.0
    out, engine, tok = run(task, [
        Step(call(INC + "\ncheck(transform)"), seconds=1.0),
        Step("thinking " * 50, seconds=200.0),
        Step(forced_call("print(1)", summary="")),
        Step(blocks((0, [[3, 3]])), seconds=10_000.0),
    ], force_at=100.0, end=end)
    assert kinds(out) == ["think", "tool", "force", "force2"]
    again = engine.requests[-1]
    assert again.cut_at == pytest.approx(end - toolshot.verify_worst_s(task, cfg()))
    assert engine.replies[-1].finish == "cancelled"
    # the forced reply was only the call: the note follows the forcing suffix's two newlines
    note = tok.decode(again.input_ids[len(engine.requests[2].input_ids):])
    assert note == toolshot.FORCE_AGAIN_NOTE and not note.startswith("\n")
    assert out.program is not None and out.program["name"] == "transform"
    assert len(out.verifications) == 1
    assert norm(out.attempts) == [[[[8, 9]]]]


def test_force2_skipped_when_only_the_verify_time_is_left() -> None:
    task = inc_task()
    worst = toolshot.verify_worst_s(task, cfg())
    out, engine, _ = run(task, [
        Step("thinking " * 50, seconds=200.0),
        Step(forced_call("print(1)"), seconds=300.0 - worst - 100.0 + 1.0),
    ], force_at=100.0, end=300.0)
    assert kinds(out) == ["think", "force"]
    assert engine.overflow == 0
    assert out.attempts == [[]]


# --------------------------------------------------------------------------------------------
# P5.8b: a kept top-level assignment that reads a name only a dropped REPL statement bound
# --------------------------------------------------------------------------------------------

# Version 17's four failing statements (P5.2-01 revision 21's record), each after the dropped
# statement that bound its names in the REPL.
LOAD_FAULTS = [
    "for i, (a, b) in enumerate(train):\n    o = transform(a)\n"
    "d=np.argwhere(o!=np.array(b))",
    "for t in test_inputs:\n    pass\nbg = np.bincount(t.ravel()).argmax()",
    "def sim(x, y, mode):\n    return 0\n"
    "inp, boxes = train[0][0], []\npos=sim(inp,boxes,'same')",
    "for inp, out in train:\n    pass\nd=np.argwhere(inp!=out)",
]


@pytest.mark.parametrize("fault", LOAD_FAULTS)
def test_load_name_error_retried_without_assignments(fault: str) -> None:
    out, engine, _ = run(inc_task(), [
        Step(call(INC)),
        Step(call(fault)),
        Step(call(INC + "\ncheck(transform)")),
        Step(answer((0, [[1, 1]]))),
    ])
    assert "Error" not in engine.tool_reply(1)  # the statement runs in the REPL
    assert out.verifications[0]["call"] == 2
    rows = out.verifications
    assert rows[0]["verified"] is False and rows[0]["passed"] == 0
    assert rows[1]["verified"] is True and rows[1]["without_assignments"] is True
    assert out.program is not None and out.program["without_assignments"] is True
    assert "np.argwhere" not in out.program["source"] and "bincount" not in out.program["source"]
    assert norm(out.attempts) == [[[[1, 1]], [[8, 9]]]]


def test_retry_without_assignments_still_fails_a_wrong_program() -> None:
    out, _, _ = run(inc_task(), [
        Step(call(IDENTITY)),
        Step(call(LOAD_FAULTS[3])),
        Step(call(IDENTITY)),
        Step(answer((0, [[1, 1]]))),
    ])
    assert [r["without_assignments"] for r in out.verifications][:2] == [False, True]
    assert not any(r["verified"] for r in out.verifications)
    assert out.program is None


def test_retry_counts_against_max_verify() -> None:
    out, _, _ = run(inc_task(), [
        Step(call(INC)),
        Step(call(LOAD_FAULTS[3])),
        Step(call(INC)),
        Step(answer((0, [[1, 1]]))),
    ], config=cfg(max_verify=1))
    assert len(out.verifications) == 1
    assert out.program is None


def test_without_assignments_keeps_imports_defs_and_classes() -> None:
    source = ("import numpy as np\nk = 3\n\n\ndef f(g):\n    return g\n\n\nclass C:\n    x = 1\n"
              "d = f(inp)\n")
    assert toolshot.without_assignments(source) == (
        "import numpy as np\ndef f(g):\n    return g\nclass C:\n    x = 1\n")
