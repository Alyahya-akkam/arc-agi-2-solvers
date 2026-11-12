"""Implementer's unit tests for ``arc26.agent.toolshot`` and ``arc26.agent.toolcheck`` (P5.3).

The scenario tests from the brief are in ``tests/test_toolshot.py``; these cover the pieces:
the call parser, the joined program, the config, the ``check`` prelude, and loop edges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from arc26.agent import guards, toolshot
from arc26.agent.toolcheck import CHECK_SOURCE, check_summaries
from arc26.agent.types import AgentTask, Pair, Verification


class Tok:
    """Character tokenizer with ``<|im_end|>`` as one id."""

    IM_END = 1_000_000

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for k, part in enumerate(text.split("<|im_end|>")):
            if k:
                out.append(self.IM_END)
            out.extend(ord(c) for c in part)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return "".join("<|im_end|>" if i == self.IM_END else chr(i) for i in ids)

    def render(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + (
            "<|im_start|>assistant\n<think>\n"
        )


@dataclass
class Gen:
    text: str
    ids: list[int]
    finish: str
    completion_tokens: int
    cached_tokens: int | None = None

    def row(self) -> dict[str, Any]:
        return {"finish": self.finish, "completion_tokens": self.completion_tokens}


class Script:
    """A ``generate`` that returns scripted ``(text, finish)`` replies and records requests."""

    def __init__(self, tok: Tok, replies: list[tuple[str, str]]) -> None:
        self.tok = tok
        self.replies = list(replies)
        self.requests: list[tuple[list[int], int, str]] = []

    def __call__(
        self, ids: list[int], max_new: int, cut_at: float, kind: str, extra: Mapping[str, Any]
    ) -> Gen:
        self.requests.append((list(ids), max_new, kind))
        text, finish = self.replies.pop(0) if self.replies else ("</think>\n\n", "stop")
        out = self.tok.encode(text)[:max_new]
        return Gen(self.tok.decode(out), out, finish, len(out))


def task() -> AgentTask:
    return AgentTask(
        "t1",
        (Pair([[0, 1]], [[1, 2]]), Pair([[3]], [[4]])),
        ([[5, 6]],),
    )


CALL = (
    "<tool_call>\n<function=python>\n<parameter=code>\n{code}\n</parameter>\n</function>\n"
    "</tool_call>"
)
ANSWER = '</think>\n\n```answer\n{"test_index": 0, "grid": [[6, 7]]}\n```'


def solve(replies: list[tuple[str, str]], **kwargs: Any) -> tuple[toolshot.ToolOutcome, Script]:
    tok = Tok()
    gen = Script(tok, replies)
    cfg = kwargs.pop("cfg", toolshot.tool_config({}))
    out = toolshot.solve_with_tools(
        task(), tok, gen, cfg, force_at=1e18, end=2e18, now=lambda: 0.0, **kwargs
    )
    return out, gen


# ---- the call parser ----------------------------------------------------------------------------


def test_parse_closed_call() -> None:
    call = toolshot.parse_call("x</think>\n\n" + CALL.format(code="a = 1\n    b"))
    assert call is not None
    assert (call.name, call.closed) == ("python", True)
    assert call.arguments == {"code": "a = 1\n    b"}


def test_parse_keeps_indentation_and_takes_the_last_call() -> None:
    text = CALL.format(code="print(0)") + "\n" + CALL.format(code="for i in x:\n    print(i)")
    call = toolshot.parse_call(text)
    assert call is not None and call.arguments["code"] == "for i in x:\n    print(i)"


@pytest.mark.parametrize(
    "tail",
    ["", "\n</parameter>", "\n</parameter>\n</function>", "\n<|im_end|>"],
)
def test_parse_unclosed_call(tail: str) -> None:
    call = toolshot.parse_call("<tool_call>\n<function=python>\n<parameter=code>\nprint(2)" + tail)
    assert call is not None and not call.closed
    assert call.arguments["code"] == "print(2)"


def test_parse_function_without_tool_call_tag() -> None:
    call = toolshot.parse_call("<function=python>\n<parameter=code>\nprint(3)\n</parameter>")
    assert call is not None and call.arguments == {"code": "print(3)"}


def test_parse_no_call() -> None:
    assert toolshot.parse_call("just thinking </think> ```answer\n[[1]]\n```") is None


def test_parse_markup_without_function() -> None:
    call = toolshot.parse_call("<tool_call>\nprint(1)\n</tool_call>")
    assert call is not None and call.name == "" and call.arguments == {}


# ---- the joined program -------------------------------------------------------------------------


def test_joined_program_last_definition_wins_and_imports_first() -> None:
    source = toolshot.joined_program(
        [
            "import math\ndef f(g):\n    return g\nprint(train)",
            "K = 3\ndef transform(g):\n    return f(g)\ncheck(transform)",
            "import math\ndef f(g):\n    return g + K\n",
        ]
    )
    assert source.startswith("import math\n")
    assert source.count("import math") == 1
    assert source.count("def f") == 1 and "g + K" in source
    assert "print(train)" not in source and "check(" not in source
    namespace: dict[str, Any] = {}
    exec(source, namespace)
    assert namespace["transform"](1) == 4


def test_joined_program_skips_unparsable_calls() -> None:
    assert toolshot.joined_program(["def (:", "def transform(g):\n    return g"]).startswith(
        "def transform"
    )


# ---- config -------------------------------------------------------------------------------------


def test_tool_config_values_and_errors() -> None:
    cfg = toolshot.tool_config({"max_calls": 5, "stop": ["</tool_call>"], "call_timeout_s": 4})
    assert cfg.max_calls == 5 and cfg.stop == ("</tool_call>",) and cfg.call_timeout_s == 4.0
    for bad in (
        {"nope": 1},
        {"max_calls": 0},
        {"max_calls": 1.5},
        {"lenient_answers": 1},
        {"stop": "x"},
        {"reasoning_effort": "high"},
        {"cap": True},
    ):
        with pytest.raises(ValueError):
            toolshot.tool_config(bad)


def test_system_text_keeps_direct_two_and_names_limits() -> None:
    text = toolshot.system_text(toolshot.tool_config({}))
    head, tail = toolshot.DIRECT_TWO_SYSTEM_PROMPT.split(toolshot.NO_INTERPRETER)
    assert text.startswith(head) and text.endswith(tail)
    assert "10 seconds" in text and "2,000 characters" in text and "check(fn)" in text


# ---- check --------------------------------------------------------------------------------------


def test_check_source_passes_the_policy_and_defines_no_transform() -> None:
    assert guards.check_policy(CHECK_SOURCE) == ()
    assert not guards.defines_transform(CHECK_SOURCE)


def test_check_output(capsys: pytest.CaptureFixture[str]) -> None:
    train = [
        (np.array([[1, 2]]), np.array([[1, 2]])),
        (np.array([[1, 2]]), np.array([[1, 3]])),
        (np.array([[1]]), np.array([[1, 1]])),
        (np.array([[0]]), np.array([[0]])),
    ]
    namespace: dict[str, Any] = {"np": np, "train": train}
    exec(CHECK_SOURCE, namespace)

    def fn(grid: np.ndarray) -> np.ndarray:
        if grid[0, 0] == 0:
            raise KeyError("z")
        return grid

    namespace["check"](fn)
    out = capsys.readouterr().out
    assert "demo 0: pass" in out
    assert "demo 1: fail, 1 of 2 cells differ (yours/expected):\n1 2/3" in out
    assert "demo 2: fail, shape 1x1, expected 1x2" in out
    assert "demo 3: fail, raised KeyError" in out
    assert check_summaries(out) == [[1, 4]]


# ---- loop edges ---------------------------------------------------------------------------------


def test_unclosed_call_at_im_end_does_not_double_im_end() -> None:
    reply = "</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(5)\n<|im_end|>"
    out, gen = solve([(reply, "stop"), (ANSWER, "stop")])
    second = gen.tok.decode(gen.requests[1][0])
    assert "<|im_end|>\n<|im_start|>user\n<tool_response>\n5\n</tool_response>" in second
    assert "<|im_end|><|im_end|>" not in second
    assert out.calls[0]["closed"] is False and out.calls[0]["status"] == "ok"


def test_check_survives_a_timeout_restart() -> None:
    cfg = toolshot.tool_config({"call_timeout_s": 1})
    out, gen = solve(
        [
            ("</think>\n\n" + CALL.format(code="while True:\n    pass"), "stop"),
            ("</think>\n\n" + CALL.format(code="check(lambda g: g + 1)"), "stop"),
            (ANSWER, "stop"),
        ],
        cfg=cfg,
    )
    assert [c["status"] for c in out.calls] == ["timeout", "ok"]
    assert out.calls[1]["check"] == [[2, 2]]
    assert out.calls[0]["restarted"] is True


def test_unknown_function_and_missing_code() -> None:
    bad_name = (
        "</think>\n\n<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    no_code = "</think>\n\n<tool_call>\n<function=python>\n</function>\n</tool_call>"
    out, gen = solve([(bad_name, "stop"), (no_code, "stop"), (ANSWER, "stop")])
    assert [c["status"] for c in out.calls] == ["error", "error"]
    assert "only function is python" in gen.tok.decode(gen.requests[1][0])
    assert out.attempts == [[[[6, 7]]]]


def test_generate_exception_is_recorded() -> None:
    def broken(*args: Any) -> Gen:
        raise ConnectionError("down")

    out = toolshot.solve_with_tools(
        task(), Tok(), broken, toolshot.tool_config({}), force_at=1e18, end=2e18, now=lambda: 0.0
    )
    assert out.ended_by == "error" and any("down" in e for e in out.errors)
    assert out.attempts == [[]] and out.forced is False


def test_context_bound_limits_requests_and_forces() -> None:
    cfg = toolshot.tool_config({"context_tokens": 9_000, "forcing_tokens": 100})
    out, gen = solve([("z" * 20_000, "length"), (ANSWER, "stop")], cfg=cfg)
    first_ids, first_max, _ = gen.requests[0]
    assert len(first_ids) + first_max <= 9_000 - 100
    assert out.forced and out.ended_by == "length+force"
    assert gen.requests[1][1] <= 100
    assert out.attempts == [[[[6, 7]]]]


def test_answers_from_an_earlier_turn_are_kept() -> None:
    first = ANSWER + "\n" + CALL.format(code="print(1)")
    out, _ = solve([(first, "stop"), ("</think>\n\nno blocks this time", "stop")])
    assert out.direct == [[[[6, 7]]]]
    assert out.attempts == [[[[6, 7]]]]


def test_injected_verifier_and_outcome_record() -> None:
    seen: list[str] = []

    class Stub:
        def verify(self, task: AgentTask, source: str) -> Verification:
            seen.append(source)
            return Verification(source, "x", (), (None,), (), 0.0)

    code = "def transform(g):\n    return g"
    out, _ = solve(
        [("</think>\n\n" + CALL.format(code=code), "stop"), (ANSWER, "stop")], verifier=Stub()
    )
    assert seen == [code + "\n"]
    assert out.program is None
    assert out.verifications[0]["verified"] is False
    record = out.record()
    assert "text" not in record and record["calls"][0]["defines_transform"] is True


# ---- fix round 1: the prompt and the REPL's limits ----------------------------------------------


def test_tool_prompt_has_no_no_interpreter_wording() -> None:
    tok = Tok()
    gen = Script(tok, [(ANSWER, "stop")])
    toolshot.solve_with_tools(
        task(), tok, gen, toolshot.tool_config({}), force_at=1e18, end=2e18, now=lambda: 0.0
    )
    prompt = tok.decode(gen.requests[0][0]).lower()
    assert "no code interpreter" not in prompt and "there is no" not in prompt
    assert "you have a python tool" in prompt
    assert "give one or two answer blocks per test input" in prompt
    assert toolshot.NO_INTERPRETER in toolshot.DIRECT_TWO_SYSTEM_PROMPT


def test_system_text_refuses_a_prompt_without_the_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(toolshot, "DIRECT_TWO_SYSTEM_PROMPT", "no such sentence")
    with pytest.raises(ValueError):
        toolshot.system_text(toolshot.tool_config({}))


def test_repl_memory_cap_is_a_cell_error_and_the_repl_goes_on() -> None:
    hog = "x = np.ones(8 * 1024**3 // 8)\nprint(x.size)"
    out, gen = solve(
        [
            ("</think>\n\n" + CALL.format(code=hog), "stop"),
            ("</think>\n\n" + CALL.format(code="check(lambda g: g + 1)"), "stop"),
            (ANSWER, "stop"),
        ]
    )
    assert out.calls[0]["status"] == "error" and out.calls[0]["error"] == "memory"
    assert "MemoryError" in gen.tok.decode(gen.requests[1][0])
    assert out.calls[1]["check"] == [[2, 2]]


def test_repl_cpu_cap_restarts_the_repl_not_the_task() -> None:
    cfg = toolshot.tool_config({"repl_cpu_s": 3, "call_timeout_s": 30})
    out, gen = solve(
        [
            ("</think>\n\n" + CALL.format(code="while True:\n    pass"), "stop"),
            ("</think>\n\n" + CALL.format(code="check(lambda g: g + 1)"), "stop"),
            (ANSWER, "stop"),
        ],
        cfg=cfg,
    )
    first, second = out.calls
    assert first["status"] == "error" and first["error"].startswith("crashed")
    assert first["restarted"] is True and first["seconds"] < 25
    assert second["status"] == "ok" and second["check"] == [[2, 2]]
    assert out.attempts == [[[[6, 7]]]]


def test_repl_limits_forks_files_and_workdir() -> None:
    import tempfile

    from arc26.agent.sandbox import SubprocessRepl

    with SubprocessRepl(task(), cpu_seconds=60) as repl:
        workdir = repl._workdir
        assert workdir.startswith(tempfile.gettempdir())
        assert "/kaggle/working" not in workdir
        probe = "import numpy\nprint(numpy.save is None)\n"
        assert repl.run_cell(probe, 10).error is not None  # numpy.save is a forbidden attribute
        fork = repl.run_cell("import os\nos.fork()", 10)
        assert fork.error is not None
        write = repl.run_cell("open('f.txt', 'w').write('x')", 10)
        assert write.error is not None
    assert not __import__("os").path.exists(workdir)


# ---- fix round 2 --------------------------------------------------------------------------------


def test_joined_program_skips_compound_lines_and_odd_targets() -> None:
    source = toolshot.joined_program(
        ["k = 3; print(k)", "a.b = 1\nx, y = 1, 2\nn = 4\ndef transform(g):\n    return g"]
    )
    assert "print" not in source and "a.b" not in source and "x, y" not in source
    assert "n = 4" in source and "def transform" in source


def test_compound_cell_never_costs_the_direct_answers() -> None:
    code = "k = 3; print(k)\ndef transform(g):\n    return g"
    out, _ = solve([("</think>\n\n" + CALL.format(code=code), "stop"), (ANSWER, "stop")])
    assert out.calls[0]["status"] == "ok"
    assert out.attempts == [[[[6, 7]]]]


def test_verifier_exception_is_an_error_not_a_raise() -> None:
    class Broken:
        def verify(self, task: AgentTask, source: str) -> Verification:
            raise RuntimeError("boom")

    code = "def transform(g):\n    return g"
    out, _ = solve(
        [("</think>\n\n" + CALL.format(code=code), "stop"), (ANSWER, "stop")], verifier=Broken()
    )
    assert any("boom" in e for e in out.errors)
    assert out.attempts == [[[[6, 7]]]]


def test_verifier_does_not_start_past_the_end() -> None:
    seen: list[str] = []

    class Stub:
        def verify(self, task: AgentTask, source: str) -> Verification:
            seen.append(source)
            return Verification(source, "x", (), (None,), (), 0.0)

    tok = Tok()
    code = "def transform(g):\n    return g"
    gen = Script(tok, [("</think>\n\n" + CALL.format(code=code), "stop"), (ANSWER, "stop")])
    out = toolshot.solve_with_tools(
        task(),
        tok,
        gen,
        toolshot.tool_config({}),
        force_at=5.0,
        end=6.0,
        now=lambda: 0.0,
        verifier=Stub(),
    )
    assert seen == [] and any("no time left" in e for e in out.errors)
    assert out.attempts == [[[[6, 7]]]]


@pytest.mark.parametrize(
    "cut_text",
    [
        "</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1",
        "</think>\n\n<tool_call>\n<function=pyth",
        "</think>\n\n",
    ],
)
def test_cut_after_think_close_is_answered_and_forced(cut_text: str) -> None:
    out, gen = solve([(cut_text, "length"), (ANSWER[len("</think>") :], "stop")])
    before = len(gen.requests[0][0]) + len(gen.tok.encode(cut_text))
    tail = gen.tok.decode(gen.requests[1][0][before:])
    assert tail.count("</tool_call>") == (1 if "<tool_call>" in cut_text else 0)
    assert "<tool_response>\n" + toolshot.BUDGET_MESSAGE in tail
    assert tail.endswith(toolshot.THINKING_END_NOTE + "</think>\n\n")
    assert gen.requests[1][2] == "force" and out.forced
    assert out.attempts == [[[[6, 7]]]]
    assert all(c["status"] != "ok" for c in out.calls)


def test_cut_inside_answer_blocks_is_continued_as_is() -> None:
    cut_text = '</think>\n\n```answer\n{"test_index": 0, "gr'
    out, gen = solve([(cut_text, "length"), ('id": [[6, 7]]}\n```', "stop")])
    forced = gen.tok.decode(gen.requests[1][0])
    assert forced.endswith(cut_text)
    assert out.attempts == [[[[6, 7]]]]


def test_cut_call_inside_reasoning_is_closed_then_forced() -> None:
    cut_text = "hmm\n<tool_call>\n<function=python>\n<parameter=code>\nx = 1"
    out, gen = solve([(cut_text, "length"), (ANSWER[len("</think>") :], "stop")])
    tail = gen.tok.decode(gen.requests[1][0]).split(cut_text, 1)[1]
    assert tail.startswith("\n</parameter>\n</function>\n</tool_call><|im_end|>")
    assert out.calls[0]["cut"] is True and out.calls[0]["in_reasoning"] is True
    assert out.attempts == [[[[6, 7]]]]


def test_forced_answer_fits_cap_and_context() -> None:
    cfg = toolshot.tool_config({"cap": 3_000, "forcing_tokens": 500, "context_tokens": 20_000})
    out, gen = solve([("z" * 5_000, "length"), (ANSWER, "stop")], cfg=cfg)
    think_ids, think_max, _ = gen.requests[0]
    force_ids, force_max, _ = gen.requests[1]
    assert think_max == 3_000 - 500
    assert think_max + force_max <= 3_000
    assert len(force_ids) + force_max <= 20_000


def test_context_full_call_is_answered_by_the_forcing_turn() -> None:
    cfg = toolshot.tool_config({"context_tokens": 20_000, "forcing_tokens": 100})
    tok = Tok()
    opening = len(tok.encode(toolshot.system_text(cfg))) + 400
    call = "</think>\n\n" + CALL.format(code="print('x' * 3000)")
    filler = "z" * (20_000 - opening - 600 - 1_000 - len(call))
    out, gen = solve([(filler + call, "stop"), (ANSWER[8:], "stop")], cfg=cfg)
    assert out.calls[0].get("context_full") is True
    forced = gen.tok.decode(gen.requests[-1][0])
    assert "<tool_response>\n" + toolshot.BUDGET_MESSAGE in forced
    assert gen.requests[-1][2] == "force" and len(gen.requests) == 2
    assert out.attempts == [[[[6, 7]]]]


def test_parse_call_must_end_the_turn() -> None:
    assert toolshot.parse_call(CALL.format(code="print(1)") + "\nmore text") is None
    assert toolshot.parse_call("<function=python> is how I call it, then I think more") is None
    assert toolshot.parse_call(CALL.format(code="print(1)") + "\n<|im_end|>") is not None


def test_turn_with_answer_blocks_is_final() -> None:
    first = ANSWER + "\n" + CALL.format(code="print(1)")
    assert toolshot.parse_call(first) is None
    out, gen = solve([(first, "stop")])
    assert len(gen.requests) == 1 and out.calls == []
    assert out.attempts == [[[[6, 7]]]]


def test_memory_error_reply_says_the_session_goes_on() -> None:
    hog = "x = 1\ny = np.ones(4 * 1024**3 // 8)"
    out, gen = solve(
        [
            ("</think>\n\n" + CALL.format(code=hog), "stop"),
            ("</think>\n\n" + CALL.format(code="print(x)"), "stop"),
            (ANSWER, "stop"),
        ]
    )
    first = gen.tok.decode(gen.requests[1][0])
    assert "1 GiB memory limit" in first and "restarted" not in first.split("MemoryError")[-1]
    assert out.calls[0]["restarted"] is False
    assert "<tool_response>\n1\n" in gen.tok.decode(gen.requests[2][0])


def test_repl_starts_with_numpy_under_1_gib() -> None:
    from arc26.agent.sandbox import SubprocessRepl

    assert toolshot.tool_config({}).repl_memory_bytes == 1024**3
    with SubprocessRepl(task(), memory_bytes=1024**3) as repl:
        result = repl.run_cell("import numpy\nprint(np.zeros((30, 30)).sum(), train[0][0])", 10)
    assert result.error is None and result.stdout.startswith("0.0")


def test_thinking_end_note_asks_for_both_answer_blocks() -> None:
    assert "both" in toolshot.THINKING_END_NOTE and "answer blocks" in toolshot.THINKING_END_NOTE
