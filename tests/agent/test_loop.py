"""Tests of the agent loop, its prompts and feedback, and the runner (P6.2).

The REPL and the verifier are in-test fakes of the P6.1 protocols. Both run the scripted
(trusted) programs in-process; programs with hostile signatures are never executed: the fake
REPL answers as a sandbox that stopped them, the fake verifier as guards that rejected them.
The end-to-end test runs :func:`run_tasks` with :class:`ScriptedEngine` on 10 synthetic tasks
and writes a submission that must validate.
"""

import ast
import asyncio
import functools
import hashlib
import io
import json
import logging
import sys
import time
import traceback
import types

import numpy as np
import pytest

from arc26.agent import feedback, prompts
from arc26.agent.loop import (
    AttemptContext,
    Budget,
    append_record,
    attempt_context,
    attempt_seed,
    current_attempt,
    fenced_blocks,
    parse_answer,
    plain,
    program_source,
    read_records,
    record_from_json,
    record_to_json,
    run_attempt,
    run_tasks,
)
from arc26.agent.mock_engine import (
    ScriptedEngine,
    ScriptedReply,
    answer_reply,
    code_reply,
    error_reply,
    hostile_programs,
    no_code_reply,
    overlong_reply,
)
from arc26.agent.select import build_submission, load_nvarc, select_files, write_submission
from arc26.agent.tasks import count_test_inputs
from arc26.agent.types import (
    AgentTask,
    CellResult,
    GuardFinding,
    Message,
    Pair,
    PairResult,
    Verification,
)
from arc26.submit import validate
from arc26.timing import TimingLogger
from arc26.timing import read_records as read_timing

HOSTILE_SIGNS = ("import os", "open(", "__import__", "while True", "bytearray(")


def defines_transform(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "transform" for t in node.targets
        ):
            return True
    return False


class FakeRepl:
    """In-process REPL for trusted cells; hostile cells are 'stopped' without running."""

    def __init__(self, task):
        self.task = task
        self.cells = []
        self.closed = False
        self.out = io.StringIO()
        self.namespace = {
            "np": np,
            "train": [(np.array(p.input), np.array(p.output)) for p in task.train],
            "test_inputs": [np.array(g) for g in task.test_inputs],
            "print": functools.partial(print, file=self.out),
        }

    def run_cell(self, code, timeout_s):
        self.cells.append(code)
        defines = defines_transform(code)
        if "while True" in code:
            return CellResult("", "timeout", defines, timeout_s, True)
        if "bytearray(" in code:
            return CellResult("", "memory", defines, 0.1, True)
        if any(sign in code for sign in HOSTILE_SIGNS):
            return CellResult("", "PermissionError: blocked", defines, 0.0, False)
        self.out.seek(0)
        self.out.truncate()
        try:
            exec(compile(code, "<cell>", "exec"), self.namespace)
        except Exception as err:
            error = traceback.format_exception_only(err)[-1].strip()
        else:
            error = None
        return CellResult(self.out.getvalue(), error, defines, 0.0, False)

    def close(self):
        self.closed = True


def fake_guards(task, source):
    findings = []
    if any(sign in source for sign in HOSTILE_SIGNS):
        findings.append(GuardFinding("forbidden_call", "hostile signature", True))
    if "hash(" in source:
        findings.append(GuardFinding("input_hash", "keys on hash", True))
    if any(str(pair.output) in source for pair in task.train):
        findings.append(GuardFinding("literal_output", "embeds a demonstration output", True))
    if "# DEMOTE" in source:
        findings.append(GuardFinding("shape_dispatch", "marker", False))
    return tuple(findings)


def call(fn, grid):
    try:  # as P6.1's verifier: a fresh 2-D int64 array per call
        out = np.asarray(fn(np.array(grid, dtype=np.int64)))
    except Exception as err:
        return None, traceback.format_exception_only(err)[-1].strip()
    if out.ndim != 2 or not (1 <= out.shape[0] <= 30 and 1 <= out.shape[1] <= 30):
        return None, f"invalid grid shape {out.shape}"
    if not np.issubdtype(out.dtype, np.integer) or out.min() < 0 or out.max() > 9:
        return None, "invalid colors"
    return out.tolist(), None


class FakeVerifier:
    """In-process verifier for trusted programs; hostile ones are rejected unrun."""

    def __init__(self):
        self.sources = []

    def verify(self, task, source):
        self.sources.append(source)
        guards = fake_guards(task, source)
        sha = hashlib.sha256(source.encode()).hexdigest()
        if any(g.fatal for g in guards):
            pairs = tuple(
                PairResult(i, False, None, "not run", False, 0.0) for i in range(len(task.train))
            )
            return Verification(source, sha, pairs, (None,) * len(task.test_inputs), guards, 0.0)
        namespace = {}
        try:
            exec(compile(source, "<program>", "exec"), namespace)
            fn = namespace["transform"]
        except Exception as err:
            error = traceback.format_exception_only(err)[-1].strip()
            pairs = tuple(
                PairResult(i, False, None, error, False, 0.0) for i in range(len(task.train))
            )
            return Verification(source, sha, pairs, (None,) * len(task.test_inputs), guards, 0.0)
        pairs = []
        for i, pair in enumerate(task.train):
            out, error = call(fn, pair.input)
            want = np.array(pair.output)
            shape = out is not None and np.array(out).shape == want.shape
            accuracy = float((np.array(out) == want).mean()) if shape else 0.0
            pairs.append(PairResult(i, out == pair.output, out, error, shape, accuracy))
        tests = tuple(call(fn, grid)[0] for grid in task.test_inputs)
        return Verification(source, sha, tuple(pairs), tests, guards, 0.0)


# Ten synthetic tasks: name -> (rule as numpy, program source).
RULES = {
    "flip_lr": (np.fliplr, "np.fliplr(g)"),
    "flip_ud": (np.flipud, "np.flipud(g)"),
    "transpose": (np.transpose, "g.T"),
    "rot90": (np.rot90, "np.rot90(g)"),
    "recolor": (lambda g: np.where(g == 1, 7, g), "np.where(g == 1, 7, g)"),
    "tile": (lambda g: np.tile(g, (2, 2)), "np.tile(g, (2, 2))"),
    "upscale": (lambda g: np.kron(g, np.ones((2, 2), int)), "np.kron(g, np.ones((2, 2), int))"),
    "mirror": (lambda g: np.hstack([g, np.fliplr(g)]), "np.hstack([g, np.fliplr(g)])"),
    "roll": (lambda g: np.roll(g, 1, axis=0), "np.roll(g, 1, axis=0)"),
    "add_border": (
        lambda g: np.pad(g, 1, constant_values=5),
        "np.pad(g, 1, constant_values=5)",
    ),
}


def program(expr, extra=""):
    head = "import numpy as np\n\ndef transform(grid):\n    g = np.asarray(grid)\n"
    return f"{head}    return {expr}\n{extra}"


def make_task(name, seed, tests=1):
    rule = RULES[name][0]
    rng = np.random.default_rng(seed)

    def grid():
        h, w = rng.integers(3, 6, size=2)
        g = rng.integers(0, 4, size=(h, w))
        g[0, 0] = 1  # never all zero, never symmetric by accident in the corner
        g[-1, -1] = 2
        return g

    pairs = []
    for _ in range(3):
        g = grid()
        pairs.append(Pair(g.tolist(), np.asarray(rule(g)).tolist()))
    inputs = tuple(grid().tolist() for _ in range(tests))
    solutions = [np.asarray(rule(np.array(g))).tolist() for g in inputs]
    return AgentTask(f"{name}", tuple(pairs), inputs), solutions


@pytest.fixture
def flip():
    task, solutions = make_task("flip_lr", 1)
    return task, solutions


def attempt(task, script, budget=None, **kwargs):
    engine = kwargs.pop("engine", None) or ScriptedEngine({task.task_id: script})
    repl = FakeRepl(task)
    verifier = FakeVerifier()
    record = asyncio.run(run_attempt(task, engine, repl, verifier, budget or Budget(), **kwargs))
    return record, engine, repl, verifier


def last_user(call):
    return [m for m in call.messages if m.role == "user"][-1].content


# ---------------------------------------------------------------- parsing and prompts


def test_fenced_blocks_tags_and_unterminated():
    text = (
        "intro\n```python\na = 1\n```\n```\nb = 2\n```\n```Python3 extra\nc = 3\n```\n"
        "```answer\n[[1]]\n```\n```json\n{}\n```\n```python\nunterminated = 1\n"
    )
    assert fenced_blocks(text) == [
        ("python", "a = 1\n"),
        ("", "b = 2\n"),
        ("python3", "c = 3\n"),
        ("answer", "[[1]]\n"),
        ("json", "{}\n"),
    ]


def test_parse_answer_forms_and_errors():
    assert parse_answer('{"test_index": 1, "grid": [[1, 2]]}', 2) == (1, [[1, 2]])
    assert parse_answer("[[3]]", 1) == (0, [[3]])
    with pytest.raises(ValueError, match="not JSON"):
        parse_answer("[[1]", 1)
    with pytest.raises(ValueError, match="test_index"):
        parse_answer('{"test_index": 2, "grid": [[1]]}', 2)
    with pytest.raises(ValueError, match="test_index"):
        parse_answer('{"test_index": true, "grid": [[1]]}', 2)
    with pytest.raises(ValueError, match="invalid grid"):
        parse_answer('{"grid": [[1, 2], [3]]}', 1)


def test_program_source_keeps_definitions_and_drops_tests():
    code = (
        "import numpy as np\n"
        "from collections import Counter\n"
        "K = 3\n"
        "LOOKUP: dict = {1: 2}\n"
        "bg = int(np.bincount(train[0][0].ravel()).argmax())\n"
        "@functools.lru_cache\n"
        "def helper(x):\n"
        "    return x\n"
        "class Box:\n"
        "    pass\n"
        "def transform(grid):\n"
        "    return grid\n"
        "for x, y in train:\n"
        "    print(transform(x))\n"
        "out = transform(test_inputs[0])\n"
        "print('done')\n"
        "transform2 = transform\n"
        "count = 0\n"
        "count += 1\n"
    )
    kept = program_source(code)
    assert kept == (
        "import numpy as np\n"
        "from collections import Counter\n"
        "K = 3\n"
        "LOOKUP: dict = {1: 2}\n"
        "@functools.lru_cache\n"
        "def helper(x):\n"
        "    return x\n"
        "class Box:\n"
        "    pass\n"
        "def transform(grid):\n"
        "    return grid\n"
        "count = 0\n"
    )


def test_program_source_of_unparsable_code_is_the_code():
    assert program_source("def transform(:\n") == "def transform(:\n"
    assert program_source("transform = lambda g: g") == "transform = lambda g: g\n"


def test_task_text_renders_digit_rows_sizes_and_indices():
    task = AgentTask(
        "abcd1234",
        (Pair([[1, 0, 2]], [[2], [0], [1]]), Pair([[3]], [[3]])),
        ([[0, 1], [1, 0]],),
    )
    text = prompts.task_text(task)
    assert text == (
        "Task abcd1234: 2 training examples, 1 test input.\n\n"
        "Example 0 input (1x3):\n102\nExample 0 output (3x1):\n2\n0\n1\n\n"
        "Example 1 input (1x1):\n3\nExample 1 output (1x1):\n3\n\n"
        "Test input 0 (2x2):\n01\n10"
    )


def test_system_prompt_states_the_contract():
    text = prompts.system_prompt(with_index=False)
    for phrase in (
        "transform(grid)",
        "```python",
        "```answer",
        "test_index",
        "train",
        "test_inputs",
        "arc26.agent.lib",
        "self-contained",
        "2-D numpy array of ints (int64)",
        "a list of lists of ints 0 to 9",
    ):
        assert phrase in text
    assert chr(0x2014) not in text  # no em dash (repository rule)


def test_helper_index_included_when_the_library_is_importable(monkeypatch):
    lib = types.ModuleType("arc26.agent.lib")
    lib.index_text = lambda: "crop(grid) -> grid: crop to the bounding box"
    monkeypatch.setitem(sys.modules, "arc26.agent.lib", lib)
    text = prompts.system_prompt()
    assert text.startswith(prompts.SYSTEM_PROMPT)
    assert "Helper library" in text
    assert text.endswith("crop(grid) -> grid: crop to the bounding box")
    assert prompts.system_prompt(with_index=False) == prompts.SYSTEM_PROMPT


def test_helper_index_omitted_when_the_library_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "arc26.agent.lib", None)
    assert prompts.helper_index() is None
    assert prompts.system_prompt() == prompts.SYSTEM_PROMPT


def test_helper_index_omitted_without_index_text(monkeypatch):
    monkeypatch.setitem(sys.modules, "arc26.agent.lib", types.ModuleType("arc26.agent.lib"))
    assert prompts.helper_index() is None


def test_task_message_shows_at_most_two_prior_programs(flip):
    task, _ = flip
    verifier = FakeVerifier()
    prior = [verifier.verify(task, program(expr)) for expr in ("g", "g.T", "np.flipud(g)")]
    text = prompts.task_message(task, prior)
    assert text.startswith(prompts.task_text(task))
    assert "return g\n" not in text  # the oldest of three is dropped
    assert "return g.T" in text and "return np.flipud(g)" in text
    assert text.count("Program check:") == 2


# ---------------------------------------------------------------- feedback


def test_clip_keeps_head_and_tail():
    text = "".join(f"line {i}\n" for i in range(200))
    clipped = feedback.clip(text, 300)
    assert len(clipped) <= 300
    assert clipped.startswith("line 0\n")
    assert clipped.endswith("line 199\n")
    assert "characters cut" in clipped
    assert feedback.clip("short", 300) == "short"
    with pytest.raises(ValueError):
        feedback.clip("x", 0)


def test_verification_report_failing_example(flip):
    task, _ = flip
    verification = FakeVerifier().verify(task, program("np.where(g == 1, 0, np.fliplr(g))"))
    report = feedback.verification_report(task, verification)
    assert report.startswith(f"Program check: {verification.passed}/3 training examples passed.")
    assert "exact_match=False" in report
    assert "first mismatching rows (yours | expected):" in report
    assert "Test input 0: output" in report
    assert report.endswith("Verdict: not accepted.")


def test_verification_report_shape_mismatch_errors_and_guards(flip):
    task, _ = flip
    verifier = FakeVerifier()
    shape = verifier.verify(task, program("g[:1]"))
    report = feedback.verification_report(task, shape)
    assert "(expected" in report and "your output starts:" in report
    crash = verifier.verify(task, program("undefined_helper(g)"))
    report = feedback.verification_report(task, crash)
    assert "All examples: error: NameError: name 'undefined_helper' is not defined" in report
    varied = verifier.verify(task, program("g[99, 99]"))  # the message names each size
    report = feedback.verification_report(task, varied)
    assert "Example 0: exact_match=False, error: IndexError" in report
    assert "Test input 0: no valid output" in report
    hostile = verifier.verify(task, hostile_programs(task)["literal_lookup"])
    report = feedback.verification_report(task, hostile)
    assert "literal_output (rejects the program)" in report
    assert "Verdict: rejected by a guard" in report


def test_verification_report_accepted_and_capped(flip):
    task, _ = flip
    good = FakeVerifier().verify(task, program("np.fliplr(g)  # DEMOTE"))
    report = feedback.verification_report(task, good)
    assert "3/3 training examples passed" in report
    assert "shape_dispatch (ranks it lower): marker" in report
    assert report.endswith(
        "Verdict: accepted: it reproduces every training example and runs on every test input."
    )
    assert len(feedback.verification_report(task, good, max_chars=120)) <= 120


def test_cell_report_cases():
    assert feedback.cell_report(0, CellResult("hi\n", None, False, 0.0, False), timeout_s=10) == (
        "[cell 0] output:\nhi"
    )
    assert feedback.cell_report(1, CellResult("", None, False, 0.0, False), timeout_s=10) == (
        "[cell 1] ran without output."
    )
    text = feedback.cell_report(2, CellResult("", "timeout", False, 10.0, True), timeout_s=10)
    assert "timeout (stopped after 10 s)" in text and "REPL was restarted" in text
    text = feedback.cell_report(3, CellResult("", "memory", False, 1.0, True), timeout_s=10)
    assert "memory limit exceeded" in text
    long = feedback.cell_report(
        4, CellResult("x" * 5000, None, False, 0.0, False), timeout_s=10, max_chars=100
    )
    assert len(long) < 140


def test_compose_feedback_header_and_cap():
    text = feedback.compose_feedback(
        ["a" * 50, "b" * 50], next_round=2, max_rounds=3, tokens_left=900, max_chars=60
    )
    header, body = text.split("\n\n", 1)
    assert header == "Round 2/3. About 900 tokens left for this task."
    assert len(body) <= 60
    assert (
        feedback.compose_feedback([], next_round=2, max_rounds=3, tokens_left=1, max_chars=60)
        == "Round 2/3. About 1 tokens left for this task."
    )


# ---------------------------------------------------------------- one attempt


def test_correct_program_verifies_in_one_turn(flip):
    task, solutions = flip
    record, engine, repl, verifier = attempt(task, [code_reply(program("np.fliplr(g)"))])
    assert record.stop_reason == "verified"
    assert record.turns == 1
    assert len(record.verifications) == 1 and record.verifications[0].verified
    assert record.verifications[0].test_outputs == tuple(solutions)
    assert record.attempt_id == "flip_lr/p1/0" and record.phase == 1
    assert record.completion_tokens == engine.tokens(code_reply(program("np.fliplr(g)")).text)
    assert record.prompt_tokens > 0 and record.seconds >= 0
    assert len(repl.cells) == 1 and len(verifier.sources) == 1


def test_wrong_program_then_fix_after_feedback(flip):
    task, _ = flip
    script = [code_reply(program("np.flipud(g)")), code_reply(program("np.fliplr(g)"))]
    record, engine, _, _ = attempt(task, script)
    assert record.stop_reason == "verified"
    assert record.turns == 2
    assert [v.verified for v in record.verifications] == [False, True]
    second = engine.calls[1]
    assert second.turn == 1
    assert [m.role for m in second.messages] == ["system", "user", "assistant", "user"]
    report = last_user(second)
    assert report.startswith("Round 2/3. About ")
    assert "[cell 0] Program check: " in report
    assert "first mismatching rows (yours | expected):" in report
    assert "Test input 0: output" in report


def test_round_limit(flip):
    task, _ = flip
    record, engine, _, _ = attempt(task, [code_reply(program("g"))] * 5, Budget(max_turns=2))
    assert record.stop_reason == "rounds"
    assert record.turns == 2
    # The same program twice is verified once.
    assert len(record.verifications) == 1
    assert "this program was checked before" not in last_user(engine.calls[1])  # 1st report
    assert len(engine.calls) == 2


def test_resubmitted_program_is_not_verified_again(flip):
    task, _ = flip
    record, engine, _, verifier = attempt(task, [code_reply(program("g"))] * 3)
    assert len(verifier.sources) == 1
    assert "this program was checked before: 0/3" in last_user(engine.calls[2])
    assert record.stop_reason == "rounds"


def test_token_budget_lowers_max_tokens_and_stops(flip):
    task, _ = flip
    budget = Budget(max_turns=10, attempt_tokens=1000, turn_tokens=600, min_turn_tokens=100)
    reply = ScriptedReply(text="no code here", reasoning="r" * 2000)  # 503 tokens at 4 chars
    record, engine, _, _ = attempt(task, [reply] * 10, budget)
    assert [c.max_tokens for c in engine.calls] == [600, 497]
    assert record.stop_reason == "budget"
    assert record.completion_tokens == 503 + 497
    assert record.turns == 2


def test_context_window_caps_the_reply(flip):
    task, _ = flip
    messages = prompts.initial_messages(task)
    bound = sum(len(m.content) + 16 for m in messages) + 64
    budget = Budget(context_tokens=bound + 300, turn_tokens=8000, min_turn_tokens=100)
    # Two simulated tokens per character: the engine's prompt count after turn 1 exceeds the
    # window, so the second prompt's bound leaves no room for a reply.
    engine = ScriptedEngine({task.task_id: [code_reply(program("g"))] * 3}, chars_per_token=0.5)
    record, engine, _, _ = attempt(task, None, budget, engine=engine)
    assert engine.calls[0].max_tokens == 300
    assert record.turns == 1
    assert record.stop_reason == "budget"
    # Without the engine's count (it fails), the opening is bounded at one token per character.
    tight = Budget(context_tokens=bound + 50, min_turn_tokens=100)
    engine = ScriptedEngine({task.task_id: [code_reply(program("g"))]})
    engine.count_error = "no tokenize endpoint"
    record, engine, _, _ = attempt(task, None, tight, engine=engine)
    assert record.stop_reason == "budget" and record.turns == 0 and not engine.calls


def test_overlong_reply_is_cut_and_reported(flip):
    task, _ = flip
    script = [overlong_reply(4000), code_reply(program("np.fliplr(g)"))]
    record, engine, repl, _ = attempt(task, script, Budget(turn_tokens=500))
    assert record.stop_reason == "verified"
    assert record.turns == 2
    assert record.completion_tokens == 500 + engine.tokens(script[1].text)
    assert len(repl.cells) == 1  # the cut reply's code block never ran
    note = last_user(engine.calls[1])
    assert "hit the limit of 500 tokens" in note
    assert feedback.NO_CODE_NOTE in note


def test_reply_without_code_is_nudged(flip):
    task, _ = flip
    record, engine, _, _ = attempt(task, [no_code_reply(), code_reply(program("np.fliplr(g)"))])
    assert record.stop_reason == "verified"
    assert feedback.NO_CODE_NOTE in last_user(engine.calls[1])


def test_direct_answers_are_recorded(flip):
    task, solutions = flip
    bad = ScriptedReply(
        text='```answer\n{"test_index": 3, "grid": [[1]]}\n```\n```answer\n[[1,\n```'
    )
    script = [answer_reply(0, solutions[0]), bad]
    record, engine, _, _ = attempt(task, script, Budget(max_turns=2))
    assert record.direct_answers == ((0, solutions[0]),)
    assert record.stop_reason == "rounds"
    assert "[answer] recorded for test input 0." in last_user(engine.calls[1])
    assert record.verifications == ()


def test_invalid_answers_are_reported(flip):
    task, _ = flip
    bad = ScriptedReply(
        text='```answer\n{"test_index": 3, "grid": [[1]]}\n```\n```answer\n[[1, 2], [3]]\n```'
    )
    record, engine, _, _ = attempt(task, [bad], Budget(max_turns=2))
    report = last_user(engine.calls[1])
    assert "[answer] not recorded: test_index must be an integer from 0 to 0" in report
    assert "[answer] not recorded: invalid grid: ragged" in report
    assert record.direct_answers == ()


def test_engine_error_on_first_call(flip):
    task, _ = flip
    record, _, repl, _ = attempt(task, [error_reply()])
    assert record.stop_reason == "engine_error"
    assert (record.turns, record.completion_tokens, record.verifications) == (0, 0, ())
    assert repl.cells == []


def test_engine_error_after_a_turn_keeps_its_work(flip):
    task, _ = flip
    record, _, _, _ = attempt(task, [code_reply(program("g")), error_reply()])
    assert record.stop_reason == "engine_error"
    assert record.turns == 1
    assert len(record.verifications) == 1


def test_deadline_cuts_a_waiting_call(flip):
    task, _ = flip
    engine = ScriptedEngine({task.task_id: [ScriptedReply(text="slow", delay_s=5.0)]})
    start = time.monotonic()
    record, _, _, _ = attempt(task, None, engine=engine, deadline=time.monotonic() + 0.2)
    assert time.monotonic() - start < 2.0
    assert record.stop_reason == "deadline"
    assert record.turns == 0


def test_deadline_already_passed(flip):
    task, _ = flip
    record, engine, _, _ = attempt(task, [code_reply(program("g"))], deadline=time.monotonic() - 1)
    assert record.stop_reason == "deadline"
    assert not engine.calls


def test_hostile_programs_are_rejected(flip):
    task, _ = flip
    hostile = hostile_programs(task)
    script = [
        code_reply(hostile[name]) for name in ("literal_lookup", "shape_dispatch", "input_hash")
    ]
    record, engine, _, _ = attempt(task, script)
    assert record.stop_reason == "rounds"
    assert all(not v.verified for v in record.verifications)
    assert all(any(g.fatal for g in v.guards) for v in record.verifications)
    assert "Verdict: rejected by a guard" in last_user(engine.calls[1])


def test_sandbox_attacks_end_as_cell_errors(flip):
    task, _ = flip
    hostile = hostile_programs(task)
    names = ("infinite_loop", "huge_allocation", "imports_os", "opens_file", "dunder_import")
    script = [ScriptedReply(text="".join(code_reply(hostile[n]).text for n in names))]
    record, engine, repl, verifier = attempt(task, script, Budget(max_turns=2, max_cells=5))
    report = last_user(engine.calls[1])
    assert "[cell 0] error: timeout" in report and "REPL was restarted" in report
    assert "[cell 1] error: memory limit exceeded" in report
    assert "[cell 2] error: PermissionError: blocked" in report
    assert len(verifier.sources) == 5
    assert not any(v.verified for v in record.verifications)


def test_cells_per_reply_are_capped(flip):
    task, _ = flip
    text = "".join(f"```python\nprint({i})\n```\n" for i in range(6))
    record, engine, repl, _ = attempt(
        task, [ScriptedReply(text=text)], Budget(max_turns=2, max_cells=4)
    )
    assert len(repl.cells) == 4
    report = last_user(engine.calls[1])
    assert "2 more code block(s) not run: at most 4 per reply." in report
    assert "[cell 3] output:\n3" in report


def test_define_and_test_in_one_block_verifies(flip):
    task, _ = flip
    code = program(
        "np.fliplr(g)", "\nfor x, y in train:\n    print(np.array_equal(transform(x), y))\n"
    )
    record, _, repl, verifier = attempt(task, [code_reply(code)])
    assert record.stop_reason == "verified"
    assert "for x, y in train" not in verifier.sources[0]
    assert repl.out.getvalue() == "True\nTrue\nTrue\n"


def test_verified_program_stops_the_reply_and_keeps_answers(flip):
    task, solutions = flip
    text = (
        code_reply(program("np.fliplr(g)")).text
        + "```python\nprint('never')\n```\n"
        + answer_reply(0, solutions[0]).text
    )
    record, _, repl, _ = attempt(task, [ScriptedReply(text=text)])
    assert record.stop_reason == "verified"
    assert len(repl.cells) == 1
    assert record.direct_answers == ((0, solutions[0]),)


def test_sampling_settings_and_context_reach_the_engine(flip):
    task, _ = flip
    budget = Budget(temperature=0.6, reasoning_effort="low", max_turns=2)
    _, engine, _, _ = attempt(
        task, [code_reply(program("g"))] * 2, budget, seed=42, attempt_id="flip_lr/p2/3", phase=2
    )
    assert [(c.temperature, c.reasoning_effort, c.seed) for c in engine.calls] == [
        (0.6, "low", 42)
    ] * 2
    assert {(c.task_id, c.attempt_id) for c in engine.calls} == {("flip_lr", "flip_lr/p2/3")}
    assert current_attempt() is None


def test_transcript_records_messages_and_reasoning(flip):
    task, _ = flip
    transcript = []
    script = [
        ScriptedReply(text=code_reply(program("g")).text, reasoning="hmm"),
        code_reply(program("np.fliplr(g)")),
    ]
    attempt(task, script, transcript=transcript)
    assert [m["role"] for m in transcript] == ["system", "user", "assistant", "user", "assistant"]
    assert transcript[2]["reasoning"] == "hmm"
    assert "reasoning" not in transcript[3]


def test_repl_failure_propagates(flip):
    task, _ = flip

    class BrokenRepl(FakeRepl):
        def run_cell(self, code, timeout_s):
            raise OSError("sandbox died")

    engine = ScriptedEngine({task.task_id: [code_reply(program("g"))]})
    with pytest.raises(OSError, match="sandbox died"):
        asyncio.run(run_attempt(task, engine, BrokenRepl(task), FakeVerifier(), Budget()))


def test_attempt_context_manager():
    assert current_attempt() is None
    context = AttemptContext("t", "t/p1/0", 1)
    with attempt_context(context) as inner:
        assert inner is context and current_attempt() is context
    assert current_attempt() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_turns": 0},
        {"attempt_tokens": 0},
        {"context_tokens": 0},
        {"cell_timeout_s": 0},
        {"temperature": -1.0},
        {"max_cells": 0},
    ],
)
def test_budget_validation(kwargs):
    with pytest.raises(ValueError):
        Budget(**kwargs)


def test_attempt_seed_is_stable_and_distinct():
    assert attempt_seed(0, "a/p1/0") == attempt_seed(0, "a/p1/0")
    assert attempt_seed(0, "a/p1/0") != attempt_seed(0, "a/p1/1")
    assert attempt_seed(0, "a/p1/0") != attempt_seed(1, "a/p1/0")
    assert 0 <= attempt_seed(0, "a") < 2**32


# ---------------------------------------------------------------- records


def test_record_json_round_trip_and_crash_safe_log(tmp_path, flip):
    task, solutions = flip
    script = [
        code_reply(program("g  # DEMOTE")),
        answer_reply(0, solutions[0]),
        code_reply(program("np.fliplr(g)")),
    ]
    record, _, _, _ = attempt(task, script)
    assert record_from_json(json.loads(json.dumps(record_to_json(record)))) == record
    log = tmp_path / "attempts.jsonl"
    append_record(log, record)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write('{"task_id": "cut by a ki')  # a write cut by a kill
    records, skipped = read_records(log)
    assert records == [record] and skipped == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(stop_reason="crashed"),
        lambda d: d.pop("turns"),
        lambda d: d.update(turns="2"),
        lambda d: d["direct_answers"].append([0, [[1, 2], [3]]]),
    ],
)
def test_invalid_records_are_skipped(tmp_path, flip, mutate):
    task, solutions = flip
    record, _, _, _ = attempt(task, [answer_reply(0, solutions[0])], Budget(max_turns=1))
    data = json.loads(json.dumps(record_to_json(record)))
    mutate(data)
    log = tmp_path / "attempts.jsonl"
    log.write_text(json.dumps(data) + "\n\n", encoding="utf-8")
    assert read_records(log) == ([], 1)


# ---------------------------------------------------------------- runner


class CountingEngine:
    def __init__(self, inner):
        self.inner = inner
        self.active = 0
        self.peak = 0

    async def complete(self, messages, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            return await self.inner.complete(messages, **kwargs)
        finally:
            self.active -= 1


def test_run_tasks_concurrency_order_and_log(tmp_path):
    names = list(RULES)[:8]
    tasks = [make_task(name, seed)[0] for seed, name in enumerate(names)]
    scripts = {t.task_id: [code_reply(program(RULES[t.task_id][1]))] for t in tasks}
    engine = CountingEngine(ScriptedEngine(scripts, delay_s=0.05))
    repls = []

    def factory(task):
        repls.append(FakeRepl(task))
        return repls[-1]

    log = tmp_path / "run" / "attempts.jsonl"
    seen = []
    records = asyncio.run(
        run_tasks(
            tasks,
            engine,
            factory,
            FakeVerifier(),
            Budget(),
            concurrency=3,
            log_path=log,
            on_record=seen.append,
            seed=5,
        )
    )
    assert engine.peak == 3
    assert [r.task_id for r in records] == names
    assert all(r.stop_reason == "verified" for r in records)
    assert all(repl.closed for repl in repls) and len(repls) == 8
    logged, skipped = read_records(log)
    assert skipped == 0 and sorted(r.attempt_id for r in logged) == sorted(
        r.attempt_id for r in records
    )
    assert sorted(r.attempt_id for r in seen) == sorted(r.attempt_id for r in records)
    seeds = {c.attempt_id: c.seed for c in engine.inner.calls}
    assert seeds == {f"{n}/p1/0": attempt_seed(5, f"{n}/p1/0") for n in names}


def test_run_tasks_attempts_per_task_and_phase():
    task, _ = make_task("flip_lr", 3)
    engine = ScriptedEngine(
        {
            "flip_lr/p2/0": [code_reply(program("g"))],
            "flip_lr/p2/1": [code_reply(program("np.fliplr(g)"))],
        }
    )
    records = asyncio.run(
        run_tasks(
            [task],
            engine,
            FakeRepl,
            FakeVerifier(),
            Budget(max_turns=1),
            attempts_per_task=2,
            phase=2,
        )
    )
    assert [(r.attempt_id, r.phase, r.stop_reason) for r in records] == [
        ("flip_lr/p2/0", 2, "rounds"),
        ("flip_lr/p2/1", 2, "verified"),
    ]


def test_run_tasks_deadline_skips_unstarted_attempts():
    tasks = [make_task(name, seed)[0] for seed, name in enumerate(list(RULES)[:4])]
    engine = ScriptedEngine({t.task_id: [ScriptedReply(text="slow", delay_s=5.0)] for t in tasks})
    start = time.monotonic()
    records = asyncio.run(
        run_tasks(
            tasks,
            engine,
            FakeRepl,
            FakeVerifier(),
            Budget(),
            concurrency=2,
            deadline=time.monotonic() + 0.2,
        )
    )
    assert time.monotonic() - start < 3.0
    assert [r.stop_reason for r in records] == ["deadline", "deadline"]
    assert len(engine.calls) == 2


def test_run_tasks_survives_failing_attempts_and_callbacks(tmp_path, caplog):
    tasks = [make_task(name, seed)[0] for seed, name in enumerate(list(RULES)[:3])]
    scripts = {t.task_id: [code_reply(program(RULES[t.task_id][1]))] for t in tasks}

    def factory(task):
        if task.task_id == "flip_ud":
            raise RuntimeError("cannot start a sandbox")
        return FakeRepl(task)

    def on_record(record):
        raise ValueError("submission writer broke")

    with caplog.at_level(logging.ERROR, logger="arc26.agent.loop"):
        records = asyncio.run(
            run_tasks(
                tasks,
                ScriptedEngine(scripts),
                factory,
                FakeVerifier(),
                Budget(),
                log_path=tmp_path / "a.jsonl",
                on_record=on_record,
            )
        )
    assert [r.task_id for r in records] == ["flip_lr", "transpose"]
    assert "attempt flip_ud/p1/0 failed" in caplog.text
    assert "on_record failed" in caplog.text
    assert len(read_records(tmp_path / "a.jsonl")[0]) == 2


def test_run_tasks_passes_each_raising_attempt_to_on_error(caplog):
    tasks = [make_task(name, seed)[0] for seed, name in enumerate(list(RULES)[:3])]
    scripts = {t.task_id: [code_reply(program(RULES[t.task_id][1]))] for t in tasks}
    seen = []

    def factory(task):
        if task.task_id != "flip_lr":
            raise RuntimeError(f"cannot start a sandbox for {task.task_id}")
        return FakeRepl(task)

    def on_error(attempt_id, exc):
        seen.append((attempt_id, str(exc)))
        raise ValueError("the error counter broke")

    with caplog.at_level(logging.ERROR, logger="arc26.agent.loop"):
        records = asyncio.run(
            run_tasks(
                tasks, ScriptedEngine(scripts), factory, FakeVerifier(), Budget(),
                on_error=on_error,
            )
        )  # fmt: skip
    assert [r.task_id for r in records] == ["flip_lr"]
    assert sorted(seen) == [
        ("flip_ud/p1/0", "cannot start a sandbox for flip_ud"),
        ("transpose/p1/0", "cannot start a sandbox for transpose"),
    ]
    # Each fault is logged with its traceback, and a raising on_error is logged too.
    failed = [r for r in caplog.records if r.getMessage().endswith("failed")]
    assert [type(r.exc_info[1]).__name__ for r in failed] == ["RuntimeError", "RuntimeError"]
    assert caplog.text.count("on_error failed for") == 2


def test_run_tasks_appends_after_a_cut_line_and_writes_transcripts(tmp_path):
    task, _ = make_task("flip_lr", 1)
    log = tmp_path / "attempts.jsonl"
    log.write_text('{"task_id": "cut', encoding="utf-8")
    transcripts = tmp_path / "transcripts.jsonl"
    engine = ScriptedEngine({task.task_id: [code_reply(program("np.fliplr(g)"))]})
    asyncio.run(
        run_tasks(
            [task],
            engine,
            FakeRepl,
            FakeVerifier(),
            Budget(),
            log_path=log,
            transcript_path=transcripts,
        )
    )
    records, skipped = read_records(log)
    assert len(records) == 1 and skipped == 1
    lines = transcripts.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["attempt_id"] == "flip_lr/p1/0"
    assert [m["role"] for m in entry["messages"]] == ["system", "user", "assistant"]


def test_run_tasks_rejects_bad_arguments(flip):
    task, _ = flip
    for kwargs in ({"concurrency": 0}, {"attempts_per_task": 0}):
        with pytest.raises(ValueError):
            asyncio.run(
                run_tasks([task], ScriptedEngine(), FakeRepl, FakeVerifier(), Budget(), **kwargs)
            )


def test_run_tasks_with_no_tasks():
    assert asyncio.run(run_tasks([], ScriptedEngine(), FakeRepl, FakeVerifier(), Budget())) == []


# ---------------------------------------------------------------- end to end


def end_to_end_case(tmp_path):
    """Ten synthetic tasks, one scripted case each; returns everything the checks need."""
    names = list(RULES)
    made = {
        name: make_task(name, 100 + i, tests=2 if name == "tile" else 1)
        for i, name in enumerate(names)
    }
    tasks = [made[name][0] for name in names]
    solutions = {name: made[name][1] for name in names}

    def right(name, extra=""):
        return code_reply(program(RULES[name][1], extra))

    scripts = {
        # correct program at once
        "flip_lr": [right("flip_lr")],
        # wrong program, then the fix after feedback
        "flip_ud": [code_reply(program("np.fliplr(g)")), right("flip_ud")],
        # a hostile program first, then a correct one
        "transpose": [
            code_reply(hostile_programs(made["transpose"][0])["literal_lookup"]),
            right("transpose"),
        ],
        # no code, then correct
        "rot90": [no_code_reply(), right("rot90")],
        # an overlong reply cut at max_tokens, then correct
        "recolor": [overlong_reply(40_000), right("recolor")],
        # direct answers for both test inputs and no program
        "tile": [
            answer_reply(0, solutions["tile"][0]),
            answer_reply(1, solutions["tile"][1]),
            no_code_reply(),
        ],
        # the engine fails: NVARC must fill the slots
        "upscale": [error_reply()],
        # define and test in one block
        "mirror": [
            right("mirror", "\nfor x, y in train:\n    print(transform(x).shape, y.shape)\n")
        ],
        # wrong every time: the best unverified program fills attempt 1
        "roll": [code_reply(program("np.roll(g, 2, axis=0)"))] * 3,
        # correct but demoted by a non-fatal guard finding: still chosen
        "add_border": [right("add_border", "# DEMOTE\n")],
    }
    timing = TimingLogger(tmp_path, "engine")
    engine = ScriptedEngine(scripts, delay_s=0.01, timing=timing)
    log = tmp_path / "attempts.jsonl"
    records = asyncio.run(
        run_tasks(
            tasks,
            engine,
            FakeRepl,
            FakeVerifier(),
            Budget(max_turns=3, attempt_tokens=12_000, turn_tokens=8_000),
            concurrency=4,
            log_path=log,
        )
    )
    # NVARC ran on two tasks: it alone has upscale's answer, and it agrees with flip_lr's program.
    nvarc = {
        "upscale": [{"attempt_1": solutions["upscale"][0], "attempt_2": [[0]]}],
        "flip_lr": [{"attempt_1": solutions["flip_lr"][0], "attempt_2": [[1]]}],
    }
    nvarc_path = tmp_path / "nvarc_submission.json"
    nvarc_path.write_text(json.dumps(nvarc), encoding="utf-8")
    challenges = {
        t.task_id: {
            "train": [{"input": p.input, "output": p.output} for p in t.train],
            "test": [{"input": g} for g in t.test_inputs],
        }
        for t in tasks
    }
    challenges_path = tmp_path / "challenges.json"
    challenges_path.write_text(json.dumps(challenges), encoding="utf-8")
    return tasks, solutions, records, log, nvarc_path, challenges, challenges_path, timing


def score(submission, solutions):
    total = 0.0
    for task_id, outputs in solutions.items():
        hits = sum(
            any(entry[a] == want for a in ("attempt_1", "attempt_2"))
            for entry, want in zip(submission[task_id], outputs, strict=True)
        )
        total += hits / len(outputs)
    return total


def test_end_to_end_ten_tasks_valid_submission(tmp_path):
    tasks, solutions, records, log, nvarc_path, challenges, challenges_path, timing = (
        end_to_end_case(tmp_path)
    )
    stops = {r.task_id: r.stop_reason for r in records}
    assert stops == {
        "flip_lr": "verified",
        "flip_ud": "verified",
        "transpose": "verified",
        "rot90": "verified",
        "recolor": "verified",
        "tile": "rounds",
        "upscale": "engine_error",
        "mirror": "verified",
        "roll": "rounds",
        "add_border": "verified",
    }
    logged, skipped = read_records(log)
    assert skipped == 0 and len(logged) == 10

    nvarc = load_nvarc(nvarc_path)
    submission = build_submission(count_test_inputs(tasks), logged, nvarc)
    assert validate(submission, challenges) == []
    out = tmp_path / "submission.json"
    write_submission(out, submission, count_test_inputs(tasks))
    assert validate(json.loads(out.read_text(encoding="utf-8")), challenges) == []
    assert score(submission, solutions) == 9.0  # every task but "roll"
    roll = next(t for t in tasks if t.task_id == "roll")
    unverified = np.roll(np.array(roll.test_inputs[0]), 2, axis=0).tolist()
    assert submission["roll"] == [{"attempt_1": unverified, "attempt_2": [[0]]}]
    assert submission["flip_lr"] == [{"attempt_1": solutions["flip_lr"][0], "attempt_2": [[1]]}]
    assert submission["tile"][1]["attempt_1"] == solutions["tile"][1]

    report = select_files(log, challenges_path, tmp_path / "submission2.json", nvarc_path)
    assert json.loads((tmp_path / "submission2.json").read_text(encoding="utf-8")) == submission
    assert report["tasks"] == 10 and report["test_inputs"] == 11 and report["records"] == 10
    assert sum(report["slots"].values()) == 22

    engine_records, _ = read_timing([timing.path])
    assert {r.puzzle_id for r in engine_records} == set(solutions) - {"upscale"}  # it failed
    assert all(r.op == "engine" and r.tokens > 0 for r in engine_records)
    assert sum(r.tokens for r in engine_records) == sum(r.completion_tokens for r in records)


def test_end_to_end_without_records_or_nvarc_is_still_valid(tmp_path):
    tasks, _, _, _, _, challenges, challenges_path, _ = end_to_end_case(tmp_path)
    report = select_files(None, challenges_path, tmp_path / "empty.json")
    written = json.loads((tmp_path / "empty.json").read_text(encoding="utf-8"))
    assert validate(written, challenges) == []
    assert report["slots"]["empty"] == 22
    assert all(e == {"attempt_1": [[0]], "attempt_2": [[0]]} for v in written.values() for e in v)


def test_write_submission_rejects_invalid(tmp_path):
    with pytest.raises(ValueError, match="invalid submission"):
        write_submission(tmp_path / "s.json", {"a": [{"attempt_1": [[1]]}]}, {"a": 1})
    assert not (tmp_path / "s.json").exists()


def test_messages_are_plain_protocol_messages(flip):
    task, _ = flip
    _, engine, _, _ = attempt(task, [code_reply(program("g"))] * 2, Budget(max_turns=2))
    assert all(isinstance(m, Message) for c in engine.calls for m in c.messages)
    assert engine.calls[0].messages[0].content == prompts.system_prompt()


# ---------------------------------------------------------------- plain values, numpy verifiers


def test_plain_converts_numpy_and_rejects_nan():
    data = {
        "b": np.bool_(True),
        "i": np.int64(3),
        "f": np.float64(0.5),
        "a": np.array([[1, 2]], dtype=np.int64),
        "t": (1, (2, 3)),
        5: "keys become strings",
    }
    out = plain(data)
    assert out == {
        "b": True,
        "i": 3,
        "f": 0.5,
        "a": [[1, 2]],
        "t": [1, [2, 3]],
        "5": "keys become strings",
    }
    assert (type(out["b"]), type(out["i"]), type(out["f"])) == (bool, int, float)
    assert type(out["a"][0][0]) is int
    context = AttemptContext("t", "t/p1/0", 1)
    assert plain(context) == {"task_id": "t", "attempt_id": "t/p1/0", "phase": 1}
    for bad in (float("nan"), np.float64("inf"), [1.0, float("-inf")]):
        with pytest.raises(ValueError, match="finite"):
            plain(bad)
    with pytest.raises(TypeError, match="set"):
        plain({1, 2})


class NumpyVerifier(FakeVerifier):
    """The fake's results with numpy scalars and arrays, as a careless verifier might return."""

    def __init__(self, nan=False):
        super().__init__()
        self.nan = nan

    def verify(self, task, source):
        v = super().verify(task, source)
        pairs = tuple(
            PairResult(
                np.int64(p.index),
                np.bool_(p.exact),
                None if p.output is None else np.array(p.output, dtype=np.int64),
                p.error,
                np.bool_(p.shape_match),
                np.float64("nan") if self.nan and not p.exact else np.float64(p.cell_accuracy),
            )
            for p in v.pairs
        )
        tests = tuple(None if t is None else np.array(t, dtype=np.int64) for t in v.test_outputs)
        seconds = np.float64(v.seconds)
        return Verification(v.source, v.source_sha256, pairs, tests, v.guards, seconds)


def test_numpy_results_are_reported_recorded_and_logged(tmp_path, flip):
    task, solutions = flip
    script = [code_reply(program("np.flipud(g)")), code_reply(program("np.fliplr(g)"))]
    engine = ScriptedEngine({task.task_id: script})
    log = tmp_path / "attempts.jsonl"
    (record,) = asyncio.run(
        run_tasks([task], engine, FakeRepl, NumpyVerifier(), Budget(), log_path=log)
    )
    assert record.stop_reason == "verified"
    first, second = record.verifications
    pair = first.pairs[1]
    assert (type(pair.exact), type(pair.cell_accuracy), type(pair.index)) == (bool, float, int)
    assert type(pair.output) is list and type(pair.output[0][0]) is int
    assert type(first.seconds) is float
    assert second.test_outputs == tuple(solutions)
    assert "first mismatching rows (yours | expected):" in last_user(engine.calls[1])
    assert read_records(log) == ([record], 0)


def test_a_record_that_cannot_be_logged_does_not_stop_the_run(tmp_path, caplog):
    tasks = [make_task("flip_lr", 1)[0], make_task("flip_ud", 2)[0]]
    scripts = {
        "flip_lr": [code_reply(program("g"))],  # fails every pair: NaN accuracies
        "flip_ud": [code_reply(program("np.flipud(g)"))],  # verified: no NaN
    }
    log = tmp_path / "attempts.jsonl"
    seen = []
    with caplog.at_level(logging.ERROR, logger="arc26.agent.loop"):
        records = asyncio.run(
            run_tasks(
                tasks,
                ScriptedEngine(scripts),
                FakeRepl,
                NumpyVerifier(nan=True),
                Budget(max_turns=1),
                log_path=log,
                on_record=seen.append,
            )
        )
    assert [r.task_id for r in records] == ["flip_lr", "flip_ud"]
    assert [r.task_id for r in seen] == ["flip_lr", "flip_ud"]
    assert "writing the logs of flip_lr/p1/0 failed" in caplog.text
    assert [r.task_id for r in read_records(log)[0]] == ["flip_ud"]
    with pytest.raises(ValueError, match="finite"):
        record_to_json(records[0])


# ---------------------------------------------------------------- event loop, tokens, deadline


class SlowRepl(FakeRepl):
    def run_cell(self, code, timeout_s):
        time.sleep(0.3)
        return super().run_cell(code, timeout_s)


class SlowVerifier(FakeVerifier):
    def verify(self, task, source):
        time.sleep(0.3)
        return super().verify(task, source)


def test_repl_and_verifier_calls_run_off_the_event_loop(flip):
    task, _ = flip

    async def main():
        gaps = []
        running = True

        async def ticker():
            last = time.monotonic()
            while running:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        tick = asyncio.create_task(ticker())
        engine = ScriptedEngine({task.task_id: [code_reply(program("np.fliplr(g)"))]})
        record = await run_attempt(task, engine, SlowRepl(task), SlowVerifier(), Budget())
        running = False
        await tick
        return record, gaps

    record, gaps = asyncio.run(main())
    assert record.stop_reason == "verified"
    assert record.seconds >= 0.6  # one blocking cell and one blocking verification
    assert len(gaps) > 20 and max(gaps) < 0.2  # the loop kept ticking meanwhile


def test_prompt_and_completion_tokens_are_summed_over_turns(flip):
    task, _ = flip
    script = [code_reply(program(e)) for e in ("g", "np.flipud(g)", "np.fliplr(g)")]
    record, engine, _, _ = attempt(task, script)
    assert (record.turns, record.stop_reason) == (3, "verified")
    sent = [sum(engine.tokens(m.content) + 4 for m in c.messages) for c in engine.calls]
    assert sent[0] < sent[1] < sent[2]
    assert record.prompt_tokens == sum(sent)
    assert record.completion_tokens == sum(engine.tokens(r.text) for r in script)


def test_deadline_is_checked_between_cells(flip):
    task, _ = flip
    text = "".join(f"```python\nprint({i})\n```\n" for i in range(3))
    engine = ScriptedEngine({task.task_id: [ScriptedReply(text=text)]})
    repl = SlowRepl(task)
    transcript = []
    record = asyncio.run(
        run_attempt(
            task,
            engine,
            repl,
            FakeVerifier(),
            Budget(),
            deadline=time.monotonic() + 0.45,
            transcript=transcript,
        )
    )
    assert record.stop_reason == "deadline"
    assert len(repl.cells) == 2  # the deadline passed while the second cell ran
    assert "Deadline reached: 1 code block(s) not run." in transcript[-1]["content"]


# ------------------------------------------------------- parsing guards, preserved thinking


def test_parse_answer_rejects_deep_nesting_and_huge_blocks():
    with pytest.raises(ValueError, match="nested too deeply"):
        parse_answer("[" * 5000 + "]" * 5000, 1)
    with pytest.raises(ValueError, match="longer than 100000 characters"):
        parse_answer(" " * 100_000 + "[[1]]", 1)


def test_program_source_keeps_import_only_try_blocks():
    guarded = "try:\n    from scipy import ndimage\nexcept ImportError:\n    ndimage = None\n"
    probe = "try:\n    first = train[0]\nexcept IndexError:\n    first = None\n"
    definition = "def transform(grid):\n    return grid\n"
    assert program_source(guarded + probe + definition) == guarded + definition


def test_preserve_thinking_keeps_reasoning_in_the_conversation(flip):
    task, _ = flip
    script = [
        ScriptedReply(text=code_reply(program("g")).text, reasoning="r" * 2000),
        code_reply(program("np.fliplr(g)")),
    ]

    def second_call(preserve):
        budget = Budget(
            context_tokens=30_000,  # room for the helper index once P6.1 provides it
            attempt_tokens=100_000,
            turn_tokens=50_000,
            preserve_thinking=preserve,
        )
        record, engine, _, _ = attempt(task, script, budget)
        assert record.stop_reason == "verified"
        return engine.calls[1]

    off, on = second_call(False), second_call(True)
    assert off.messages[2].reasoning is None
    assert on.messages[2].reasoning == "r" * 2000
    # The next prompt's bound comes from the engine's counts: with the reasoning kept, every
    # generated token (500 of reasoning and the text's) plus the think tags' slack; without it,
    # the visible text at one token per character.
    text = script[0].text
    generated = 500 + -(-len(text) // 4)
    assert off.max_tokens - on.max_tokens == generated + 16 - len(text)


# ------------------------------------------------------- forcing turns (P6.3-02, revision 5)


def cut_reply(reasoning_tokens=2000):
    """A reply that is all reasoning, cut at any cap below its length (4 characters a token)."""
    return ScriptedReply(text="", reasoning="r" * (4 * reasoning_tokens))


def forcing_budget(**overrides):
    settings = dict(
        max_turns=2, attempt_tokens=20_000, turn_tokens=1000, forcing_tokens=300,
        min_turn_tokens=100, context_tokens=60_000, preserve_thinking=True,
    )  # fmt: skip
    settings.update(overrides)
    return Budget(**settings)


def test_a_cut_reply_gets_a_forcing_turn_with_thinking_off_whose_program_verifies(flip):
    task, _ = flip
    script = [cut_reply(), code_reply(program("np.fliplr(g)"))]
    transcript = []
    record, engine, repl, _ = attempt(task, script, forcing_budget(), transcript=transcript)
    assert record.stop_reason == "verified"
    assert (record.turns, record.forcing_turns, record.forced_programs) == (2, 1, 1)
    first, forced = engine.calls
    assert first.thinking is None and first.max_tokens == 1000
    # Thinking off, the reserved room, and the cut reasoning carried back.
    assert forced.thinking is False
    assert forced.max_tokens == 300
    assert forced.messages[2].role == "assistant" and forced.messages[2].reasoning == "r" * 4000
    note = forced.messages[3].content
    assert note == feedback.forcing_message([], limit=1000)
    assert "Stop reasoning now" in note
    assert "Think more briefly" not in note and feedback.NO_CODE_NOTE not in note
    assert [e.get("forcing", False) for e in transcript[-2:]] == [True, True]
    assert len(repl.cells) == 1


def test_only_the_forcing_call_turns_thinking_off_and_every_call_names_the_effort(flip):
    task, _ = flip
    script = [cut_reply(), no_code_reply(), code_reply(program("np.fliplr(g)"))]
    record, engine, _, _ = attempt(task, script, forcing_budget(reasoning_effort="low"))
    assert record.stop_reason == "verified"
    # The engine turns the forcing call's effort into its system note (test_engine.py).
    assert [(c.thinking, c.reasoning_effort) for c in engine.calls] == [
        (None, "low"), (False, "low"), (None, "low"),
    ]  # fmt: skip


def test_at_most_one_forcing_turn_per_cut_reply_and_a_cut_forced_reply_is_not_forced(flip):
    task, _ = flip
    # The forced reply is cut too: the next request is a normal thinking turn (round 2), which
    # is cut and gets its own forcing turn; that forced reply has no program, and the rounds end.
    long_code = "x = 1\n" * 2000
    script = [
        cut_reply(),
        code_reply(long_code),
        cut_reply(),
        no_code_reply("still unsure"),
    ]
    record, engine, _, _ = attempt(task, script, forcing_budget())
    assert [c.thinking for c in engine.calls] == [None, False, None, False]
    assert record.stop_reason == "rounds"
    assert (record.turns, record.forcing_turns, record.forced_programs) == (4, 2, 0)
    # The cut forced reply is reported as cut; the round counter counts thinking turns only.
    report = last_user(engine.calls[2])
    assert report.startswith("Round 2/2.") and "hit the limit of 300 tokens" in report


def test_a_forced_reply_without_a_program_continues_under_the_usual_rules(flip):
    task, _ = flip
    script = [cut_reply(), no_code_reply(), code_reply(program("np.fliplr(g)"))]
    record, engine, _, _ = attempt(task, script, forcing_budget())
    assert record.stop_reason == "verified"
    assert [c.thinking for c in engine.calls] == [None, False, None]
    assert feedback.NO_CODE_NOTE in last_user(engine.calls[2])


def test_without_forcing_tokens_a_cut_reply_is_reported_as_before(flip):
    task, _ = flip
    script = [cut_reply(), code_reply(program("np.fliplr(g)"))]
    record, engine, _, _ = attempt(task, script, forcing_budget(forcing_tokens=None))
    assert [c.thinking for c in engine.calls] == [None, None]
    assert "hit the limit of 1000 tokens" in last_user(engine.calls[1])
    assert record.forcing_turns == 0


def test_thinking_turns_reserve_the_forcing_turn_in_the_attempt_budget(flip):
    task, _ = flip
    # 1,000 + 300 of 1,500: turn 2 may take only 1,500 - 1,000 - 300 = 200 thinking tokens.
    budget = forcing_budget(attempt_tokens=1500, max_turns=3)
    script = [cut_reply(), no_code_reply(), cut_reply(), code_reply(program("np.fliplr(g)"))]
    record, engine, _, _ = attempt(task, script, budget)
    forced_tokens = engine.tokens(no_code_reply().text)
    assert [c.max_tokens for c in engine.calls] == [1000, 300, 200 - forced_tokens, 300]
    assert record.stop_reason == "verified"
    assert record.completion_tokens <= 1500


def test_thinking_turns_reserve_the_forcing_turn_in_the_context(flip):
    task, _ = flip
    bound = sum(len(m.content) + 16 for m in prompts.initial_messages(task)) + 64
    budget = forcing_budget(turn_tokens=50_000)
    reserve = budget.forcing_reserve()
    assert reserve == 300 + feedback.forcing_chars() + 16 + 2 * 16 + 64
    window = bound + 1000 + reserve
    script = [cut_reply(5000), code_reply(program("np.fliplr(g)"))]
    engine = ScriptedEngine({task.task_id: script})
    engine.count_error = "no tokenize endpoint"  # the character bound: the context binds
    budget = forcing_budget(context_tokens=window, turn_tokens=50_000)
    record, engine, _, _ = attempt(task, None, budget, engine=engine)
    first, forced = engine.calls
    assert first.max_tokens == 1000
    # The cut reply's prompt and its 1,000 tokens leave the forcing turn its whole cap.
    assert forced.max_tokens == 300 and record.stop_reason == "verified"
    # Without room for the reserve, no thinking turn starts.
    tight = forcing_budget(context_tokens=bound + reserve + 50)
    engine = ScriptedEngine({task.task_id: script})
    engine.count_error = "no tokenize endpoint"
    record, engine, _, _ = attempt(task, None, tight, engine=engine)
    assert record.stop_reason == "budget" and not engine.calls


def test_a_60k_character_kept_reasoning_does_not_stop_the_attempt_when_its_tokens_fit(flip):
    task, _ = flip
    # 60,000 characters of reasoning that the engine counts as 15,000 tokens (4 to a token): the
    # old bound (one token per character) left no room in a 32k window after turn 1.
    reasoning = "r" * 60_000
    script = [
        ScriptedReply(text=code_reply(program("g")).text, reasoning=reasoning),
        code_reply(program("np.fliplr(g)")),
    ]
    budget = Budget(
        max_turns=3, attempt_tokens=40_000, turn_tokens=16_000, min_turn_tokens=256,
        context_tokens=32_768, preserve_thinking=True,
    )  # fmt: skip
    record, engine, _, _ = attempt(task, script, budget)
    assert record.stop_reason == "verified" and record.turns == 2
    first, second = engine.calls
    assert second.messages[2].reasoning == reasoning
    completion_tokens = 15_000 + engine.tokens(script[0].text)
    prompt_tokens = sum(engine.tokens(m.content) + 4 for m in first.messages)
    report = second.messages[3].content
    bound = prompt_tokens + completion_tokens + 16 + 2 * 16 + 64 + len(report)
    assert second.max_tokens == min(16_000, 40_000 - completion_tokens, 32_768 - bound) > 256


def test_forcing_counts_survive_the_record_round_trip(flip):
    task, _ = flip
    script = [cut_reply(), code_reply(program("np.fliplr(g)"))]
    record, _, _, _ = attempt(task, script, forcing_budget())
    data = record_to_json(record)
    assert (data["forcing_turns"], data["forced_programs"]) == (1, 1)
    assert record_from_json(data) == record
    old = {k: v for k, v in data.items() if k not in ("forcing_turns", "forced_programs")}
    assert record_from_json(old).forcing_turns == 0


def test_forcing_tokens_below_the_minimum_turn_are_refused():
    with pytest.raises(ValueError, match="forcing_tokens must be at least"):
        Budget(forcing_tokens=10, min_turn_tokens=100)


def test_forcing_message_bound_holds_for_a_long_report():
    text = feedback.forcing_message(["x" * 10_000, "y" * 10_000], limit=10**11 - 1)
    assert len(text) <= feedback.forcing_chars()
    assert text.endswith(feedback.FORCING_NOTE.format(limit=10**11 - 1))


# ------------------------------------------- the opening prompt counted by the server (revision 5)


def prior_programs(task, chars):
    """Two failed programs of about ``chars`` characters each, with their verifier reports."""
    body = "\n".join(f"    # step {k}: look at the grid once more" for k in range(chars // 40))
    sources = [f"def transform(grid):\n{body}\n    return grid[::{k}]\n" for k in (1, -1)]
    return [FakeVerifier().verify(task, source) for source in sources]


def test_a_phase_2_opening_with_long_prior_programs_gets_its_stated_first_turn(flip):
    # Revision 5's phase 2: a 32k context, the forcing reserve, a cap of 24,576. The opening
    # carries two failed programs and their reports, over 8,500 characters, which the character
    # bound alone counts at one token per character.
    task, _ = flip
    prior = prior_programs(task, 4000)
    messages = prompts.initial_messages(task, prior=prior)
    assert sum(len(m.content) for m in messages[1:]) > 8500
    budget = Budget(
        max_turns=4, attempt_tokens=49_152, turn_tokens=24_576, context_tokens=32_768,
        preserve_thinking=True, forcing_tokens=4096,
    )  # fmt: skip
    engine = ScriptedEngine({task.task_id: [code_reply(program("np.fliplr(g)"))]})
    record = asyncio.run(
        run_attempt(task, engine, FakeRepl(task), FakeVerifier(), budget, prior=prior)
    )
    assert record.stop_reason == "verified" and len(engine.counts) == 1
    counted = engine.prompt_tokens(messages)
    stated = min(24_576, 32_768 - (counted + 64) - budget.forcing_reserve())
    assert engine.calls[0].max_tokens == stated
    # The character bound alone would have given about 5,000 fewer tokens.
    chars = sum(len(m.content) + 16 for m in messages) + 64
    assert stated - (32_768 - chars - budget.forcing_reserve()) > 5_000


def test_a_failed_count_falls_back_to_the_character_bound(flip):
    task, _ = flip
    messages = prompts.initial_messages(task)
    chars = sum(len(m.content) + 16 for m in messages) + 64
    budget = Budget(turn_tokens=50_000, attempt_tokens=60_000, context_tokens=40_000)
    engine = ScriptedEngine({task.task_id: [code_reply(program("np.fliplr(g)"))]})
    engine.count_error = "tokenize: HTTP 404"
    record, engine, _, _ = attempt(task, None, budget, engine=engine)
    assert record.stop_reason == "verified" and len(engine.counts) == 1
    assert engine.calls[0].max_tokens == 40_000 - chars


class OddCountEngine(ScriptedEngine):
    def __init__(self, scripts, count):
        super().__init__(scripts)
        self.count = count

    async def count_prompt_tokens(self, messages, **kwargs):
        return self.count


@pytest.mark.parametrize("count", [None, 0, True, "12", 10**9])
def test_an_odd_or_larger_count_never_loosens_the_character_bound(flip, count):
    task, _ = flip
    chars = sum(len(m.content) + 16 for m in prompts.initial_messages(task)) + 64
    engine = OddCountEngine({task.task_id: [code_reply(program("np.fliplr(g)"))]}, count)
    budget = Budget(turn_tokens=50_000, attempt_tokens=60_000, context_tokens=40_000)
    record, engine, _, _ = attempt(task, None, budget, engine=engine)
    assert engine.calls[0].max_tokens == 40_000 - chars


def test_the_count_carries_the_slack_for_the_templates_tags(flip):
    task, _ = flip
    engine = OddCountEngine({task.task_id: [code_reply(program("np.fliplr(g)"))]}, 1000)
    budget = Budget(turn_tokens=50_000, attempt_tokens=60_000, context_tokens=40_000)
    record, engine, _, _ = attempt(task, None, budget, engine=engine)
    assert engine.calls[0].max_tokens == 40_000 - (1000 + 64)


def test_an_engine_without_a_counter_keeps_the_character_bound(flip):
    task, _ = flip
    chars = sum(len(m.content) + 16 for m in prompts.initial_messages(task)) + 64

    class Plain:
        def __init__(self):
            self.inner = ScriptedEngine({task.task_id: [code_reply(program("np.fliplr(g)"))]})

        async def complete(self, messages, **kwargs):
            self.max_tokens = kwargs["max_tokens"]
            return await self.inner.complete(messages, **kwargs)

    plain = Plain()
    budget = Budget(turn_tokens=50_000, attempt_tokens=60_000, context_tokens=40_000)
    record, _, _, _ = attempt(task, None, budget, engine=plain)
    assert record.stop_reason == "verified" and plain.max_tokens == 40_000 - chars
