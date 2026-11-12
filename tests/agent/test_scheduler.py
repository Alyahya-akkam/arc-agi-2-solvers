"""Tests of the scheduler, the governor, and the crash-safe submission writer (P6.4).

The REPL and the verifier are in-process fakes that run the scripted (trusted) programs; the
engine is P6.2's ``ScriptedEngine``, wrapped where a test needs a fake clock that advances with
the tokens generated. ``tests/agent/test_run.py`` runs the real REPL and verifier end to end.
"""

import ast
import asyncio
import hashlib
import io
import json
import time
import traceback

import numpy as np
import pytest

from arc26.agent import prompts
from arc26.agent.loop import (
    MESSAGE_SLACK,
    PROMPT_SLACK,
    AttemptContext,
    attempt_context,
    read_records,
)
from arc26.agent.mock_engine import (
    ScriptedEngine,
    ScriptedReply,
    code_reply,
    error_reply,
    no_code_reply,
)
from arc26.agent.scheduler import (
    INCOMPLETE,
    EaseFeatures,
    Governor,
    GovernorSettings,
    MeteredEngine,
    PhaseSettings,
    RateMeter,
    Scheduler,
    SchedulerSettings,
    SubmissionWriter,
    clip_program,
    default_phases,
    ease_features,
    ease_order,
    failed_programs,
    nvarc_confidence,
    task_solved,
    verified_groups,
)
from arc26.agent.select import nvarc_from_submission
from arc26.agent.types import (
    AgentTask,
    AttemptRecord,
    CellResult,
    Message,
    Pair,
    PairResult,
    Verification,
)
from arc26.submit import validate

MIRROR = "import numpy as np\n\ndef transform(grid):\n    return np.fliplr(np.asarray(grid))\n"
ROT180 = "import numpy as np\n\ndef transform(grid):\n    return np.rot90(np.asarray(grid), 2)\n"
IDENTITY = "def transform(grid):\n    return grid\n"


# ---------------------------------------------------------------- fakes


def defines_transform(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == "transform" for n in tree.body)


class FakeRepl:
    """Runs trusted cells in-process."""

    def __init__(self, task):
        self.namespace = {"np": np}
        self.closed = False

    def run_cell(self, code, timeout_s):
        out = io.StringIO()
        self.namespace["print"] = lambda *a, **k: print(*a, file=out, **k)
        try:
            exec(compile(code, "<cell>", "exec"), self.namespace)
            error = None
        except Exception as err:
            error = traceback.format_exception_only(err)[-1].strip()
        return CellResult(out.getvalue(), error, defines_transform(code), 0.0, False)

    def close(self):
        self.closed = True


class FakeVerifier:
    """Runs trusted programs in-process on every pair and test input."""

    def verify(self, task, source):
        namespace = {}
        exec(compile(source, "<program>", "exec"), namespace)
        fn = namespace["transform"]
        pairs = []
        for i, pair in enumerate(task.train):
            out = np.asarray(fn(np.array(pair.input))).tolist()
            want = np.array(pair.output)
            shape = np.array(out).shape == want.shape
            accuracy = float((np.array(out) == want).mean()) if shape else 0.0
            pairs.append(PairResult(i, out == pair.output, out, None, shape, accuracy))
        tests = tuple(np.asarray(fn(np.array(g))).tolist() for g in task.test_inputs)
        sha = hashlib.sha256(source.encode()).hexdigest()
        return Verification(source, sha, tuple(pairs), tests, (), 0.0)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class ClockEngine:
    """A scripted engine whose calls take ``tokens / rate`` seconds of a fake clock."""

    def __init__(self, inner, clock, rate):
        self.inner, self.clock, self.rate = inner, clock, rate
        self.starts = {}

    async def complete(self, messages, **kwargs):
        from arc26.agent.loop import current_attempt

        self.starts.setdefault(current_attempt().attempt_id, self.clock.now)
        completion = await self.inner.complete(messages, **kwargs)
        self.clock.now += completion.completion_tokens / self.rate
        return completion


def mirror_task(task_id, width, rows=1, tall_test=False):
    """A left-right mirror task whose demonstration 0 is a palindrome (identity passes it)."""

    def grid(row):
        return [list(row) for _ in range(rows)]

    pal = [1 + min(i, width - 1 - i) % 3 for i in range(width)]
    one = [1 + i % 3 for i in range(width)]
    two = [3 - i % 3 for i in range(width)]
    train = tuple(Pair(grid(r), grid(r[::-1])) for r in (pal, one, two))
    test = [1 + (i + 1) % 3 for i in range(width)]
    tests = ([test, test[::-1][1:] + test[:1]],) if tall_test else (grid(test),)
    return AgentTask(task_id, train, tests)


def solution(task):
    return [np.fliplr(np.array(g)).tolist() for g in task.test_inputs]


def phase(**kwargs):
    base = {
        "attempts": 1, "min_attempt_tokens": 100, "max_attempt_tokens": 10**9,
        "turn_tokens": 10**9, "max_turns": 1, "reasoning_effort": "low", "share": 1.0,
        "prior": False,
    }  # fmt: skip
    base.update(kwargs)
    return PhaseSettings(**base)


def settings(phases=None, **kwargs):
    phases = phases or (
        phase(share=0.5),
        phase(reasoning_effort="xhigh", prior=True),
        phase(attempts=2, reasoning_effort="xhigh", prior=True),
        phase(attempts=0),  # the cycle off: these scenarios end when their phases are done
    )
    governor = kwargs.pop(
        "governor", GovernorSettings(prior_rate=400, window_s=1e6, min_calls=1, rate_safety=1.0)
    )
    return SchedulerSettings(phases=phases, governor=governor, **kwargs)


def record(task_id, attempt_id, phase_no, verifications=(), stop="rounds", answers=()):
    return AttemptRecord(
        task_id, attempt_id, phase_no, tuple(verifications), tuple(answers), 10, 20, 1, 0.1, stop
    )


def verification(task, source, test_outputs, passed):
    pairs = tuple(
        PairResult(i, i < passed, p.output if i < passed else [[0]], None, True,
                   1.0 if i < passed else 0.5)
        for i, p in enumerate(task.train)
    )
    sha = hashlib.sha256(source.encode()).hexdigest()
    return Verification(source, sha, pairs, tuple(test_outputs), (), 0.0)


def first_user(call):
    return [m for m in call.messages if m.role == "user"][0].content


def run(scheduler):
    return asyncio.run(scheduler.run())


# ---------------------------------------------------------------- settings


@pytest.mark.parametrize(
    "kwargs",
    [{"attempts": -1}, {"min_attempt_tokens": 0},
     {"min_attempt_tokens": 10, "max_attempt_tokens": 5}, {"share": 0}, {"share": 1.5},
     {"temperature": -1}, {"max_turns": 0}],
)  # fmt: skip
def test_phase_settings_reject_bad_values(kwargs):
    with pytest.raises(ValueError):
        phase(**kwargs)


def test_settings_reject_bad_values():
    with pytest.raises(ValueError):
        SchedulerSettings(phases=default_phases()[:3])
    unlimited_early = (phase(attempts=None), *default_phases()[1:])
    with pytest.raises(ValueError, match="only the last phase"):
        SchedulerSettings(phases=unlimited_early)
    with pytest.raises(ValueError):
        SchedulerSettings(prior_programs=3)
    with pytest.raises(ValueError):
        SchedulerSettings(prior_chars=1000)
    with pytest.raises(ValueError):
        SchedulerSettings(cycle_error_rounds=0)
    with pytest.raises(ValueError):
        SchedulerSettings(concurrency=0)
    with pytest.raises(ValueError):
        GovernorSettings(rate_safety=0)
    with pytest.raises(ValueError):
        GovernorSettings(min_calls=0)


# ---------------------------------------------------------------- ease order


def test_ease_features():
    task = mirror_task("t", 3, rows=2)
    assert ease_features(task) == EaseFeatures(cells=6 * 7, same_shape=True, colors=3)
    shaped = AgentTask("s", (Pair([[1, 2]], [[1], [2]]),), ([[3, 4]],))
    assert ease_features(shaped) == EaseFeatures(cells=6, same_shape=False, colors=4)


def test_ease_order_counts_size_shape_and_colors_equally():
    small = AgentTask("small", (Pair([[1, 2]], [[2, 1]]),), ([[1, 1]],))  # 6 cells, 2 colors
    big = AgentTask("big", (Pair([[1, 2] * 5] * 5, [[2, 1] * 5] * 5),), ([[1] * 10] * 5,))
    reshape = AgentTask("reshape", (Pair([[1, 2]], [[1], [2]]),), ([[2, 1]],))  # 6 cells
    colorful = AgentTask("colorful", (Pair([[1, 2, 3]], [[4, 5, 6]]),), ([[7, 8, 9]],))
    order = [t.task_id for t in ease_order([big, colorful, reshape, small])]
    # Rank sums: small 1.5 + 2 + 2, reshape 1.5 + 4 + 2, big 4 + 2 + 2, colorful 3 + 2 + 4.
    assert order == ["small", "reshape", "big", "colorful"]
    tied = [AgentTask(n, (Pair([[1, 2]], [[2, 1]]),), ([[1, 2]],)) for n in ("b", "a")]
    assert [t.task_id for t in ease_order(tied)] == ["a", "b"]


# ---------------------------------------------------------------- NVARC's confidence


def candidate(grid, beam=0.5, aug=(0.5,) * 8):
    return {"solution": grid, "beam_score": beam, "score_aug": list(aug)}


def test_nvarc_confidence_is_the_top_grid_count_over_16_views():
    good, other = [[1, 2]], [[2, 1]]
    candidates = {
        # Task a, test 0: 12 of 16 views give NVARC's attempt 1.
        **{f"a_0.view{i}": [candidate(good)] for i in range(12)},
        **{f"a_0.view{i}": [candidate(other)] for i in range(12, 16)},
        # Task a, test 1: 4 views agree on its attempt 1; no candidate counts twice beyond 16.
        **{f"a_1.view{i}": [candidate(other)] for i in range(4)},
        # Task b: no submission entry, so the reference's vote picks the top grid (6 views).
        **{f"b_0.view{i}": [candidate(good)] for i in range(6)},
        **{f"b_0.view{i}": [candidate(other, beam=0.1)] for i in range(6, 9)},
        "junk": [candidate(good)],
        "c_x.view0": [candidate(good)],
        "a_0.bad": [{"no": "solution"}, "not a mapping"],
    }
    nvarc = nvarc_from_submission(
        {
            "a": [
                {"attempt_1": good, "attempt_2": other},
                {"attempt_1": other, "attempt_2": [[0]]},
            ],
            "c": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
        }
    )
    confidence = nvarc_confidence({"a": 2, "b": 1, "c": 1}, candidates, nvarc)
    assert confidence["a"] == pytest.approx((12 / 16 + 4 / 16) / 2)
    assert confidence["b"] == pytest.approx(6 / 16)
    assert confidence["c"] == 0.0  # no candidates and NVARC's placeholder


def test_nvarc_confidence_without_candidates_counts_filled_attempts():
    nvarc = nvarc_from_submission(
        {"a": [{"attempt_1": [[1]], "attempt_2": [[0]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}]}
    )
    assert nvarc_confidence({"a": 2, "b": 1}, None, nvarc) == {"a": 0.5, "b": 0.0}
    assert nvarc_confidence({"a": 2}, None, None) == {}


# ---------------------------------------------------------------- rate and budgets


def test_rate_meter_counts_tokens_by_overlap_with_the_window():
    clock = FakeClock(0.0)
    meter = RateMeter(window_s=100.0, clock=clock)
    assert meter.rate() is None
    a = meter.begin()  # 0 to 50 s, 500 tokens
    b = meter.begin()  # 0 to 150 s, 3000 tokens
    clock.now = 50.0
    meter.end(a, 500)
    # 10 tokens per call-second, and b in flight for those 50 s: 2 calls on average.
    assert meter.rate() == pytest.approx(10 * (50 + 50) / 50)
    assert meter.rate(min_calls=2) is None
    clock.now = 150.0
    meter.end(b, 3000)
    # Window (50, 150]: a is out; b counts 100 of its 150 s, 2000 tokens.
    assert meter.rate() == pytest.approx(2000 / 100)
    c = meter.begin()
    clock.now = 200.0
    # Window (100, 200]: b counts 50 s (1000 tokens); c, in flight for 50 s, at b's 20 per s.
    assert meter.rate() == pytest.approx(1000 / 50 * (50 + 50) / 100)
    meter.discard(c)  # a failed call counts neither tokens nor time
    assert meter.in_flight() == 0 and meter.rate() == pytest.approx(1000 / 100)
    assert meter.total_calls == 2 and meter.total_tokens == 3500


def test_rate_estimate_follows_a_constant_rate_engine():
    # A batched engine at a constant 400 tokens per second shared by the 32 calls in flight,
    # calls of 1,000 to 60,000 tokens (80 s to 80 minutes): after warm-up the estimate stays
    # within 15% of 400 (counting the calls completed in a 15-minute window swung from 84 to 533).
    import random

    true_rate, slots, step = 400.0, 32, 5.0
    rng = random.Random(0)
    clock = FakeClock(0.0)
    meter = RateMeter(3600.0, clock)
    governor = Governor(10**9, GovernorSettings(prior_rate=100, window_s=3600, min_calls=8),
                        meter, clock)  # fmt: skip
    calls = {}

    def start():
        calls[meter.begin()] = [rng.randint(1_000, 60_000), 0.0]

    for _ in range(slots):
        start()
    seen = []
    while clock.now < 8 * 3600:
        clock.now += step
        share = true_rate * step / len(calls)
        for handle in list(calls):
            calls[handle][1] += share
            if calls[handle][1] >= calls[handle][0]:
                meter.end(handle, calls.pop(handle)[0])
                start()
        estimate, source = governor.rate()
        if clock.now >= 1800:  # warm-up: enough completions for a measurement
            assert source in ("measured", "last")
            seen.append(estimate)
    assert 0.85 * true_rate < min(seen) and max(seen) < 1.15 * true_rate


def test_metered_engine_reports_each_completed_call():
    clock = FakeClock(0.0)
    meter = RateMeter(60.0, clock)
    engine = MeteredEngine(ScriptedEngine({"t": ["x" * 40], "u": [error_reply()]}), meter)

    async def call(task_id):
        with attempt_context(AttemptContext(task_id, f"{task_id}/p1/0", 1)):
            return await engine.complete([Message("user", "hi")], max_tokens=100, temperature=1.0)

    completion = asyncio.run(call("t"))
    assert completion.completion_tokens == 10 and meter.total_tokens == 10
    with pytest.raises(Exception, match="engine unavailable"):
        asyncio.run(call("u"))
    assert meter.in_flight() == 0 and meter.total_calls == 1


def test_governor_budget_arithmetic_and_the_last_measured_rate():
    clock = FakeClock(0.0)
    meter = RateMeter(1000.0, clock)
    governor = Governor(3600.0, GovernorSettings(prior_rate=400, min_calls=2, rate_safety=0.5),
                        meter, clock)  # fmt: skip
    one = phase(min_attempt_tokens=1000, max_attempt_tokens=500_000, share=0.5)
    # The prior until two calls are measured: 0.5 * 0.5 * 400 * 3600 / 10.
    assert governor.rate() == (400, "prior")
    assert governor.attempt_tokens(one, 10) == 36_000
    first, second = meter.begin(), meter.begin()
    clock.now = 100.0
    meter.end(first, 10_000)
    meter.end(second, 10_000)  # two calls of 100 s each: 200 tokens per second together
    assert governor.rate() == (pytest.approx(200.0), "measured")
    assert governor.time_left() == 3500.0
    assert governor.attempt_tokens(one, 10) == int(0.5 * 0.5 * 200 * 3500 / 10)
    assert governor.attempt_tokens(one, 0) == governor.attempt_tokens(one, 1)
    assert governor.attempt_tokens(one, 10**6) == 1000  # the floor
    assert governor.attempt_tokens(phase(max_attempt_tokens=5_000), 1) == 5_000  # the cap
    clock.now = 2000.0  # both calls left the window: too thin, so the last measurement stays
    assert governor.rate() == (pytest.approx(200.0), "last")
    clock.now = 4000.0  # past the deadline: the floor
    assert governor.attempt_tokens(one, 1) == 1000


# ---------------------------------------------------------------- what the records say


def test_record_helpers():
    task = mirror_task("t", 3)
    out = solution(task)
    partial = verification(task, IDENTITY, [[[0]]], passed=1)
    worse = verification(task, "def transform(grid):\n    return grid * 0\n", [[[0]]], passed=0)
    right = verification(task, MIRROR, out, passed=3)
    other = verification(task, ROT180, [[[5]]], passed=3)
    records = [record("t", "t/p1/0", 1, [partial, worse]), record("t", "t/p1/1", 1, [partial])]
    assert not task_solved(records)
    assert [v.source for v in failed_programs(records)] == [worse.source, partial.source]
    assert verified_groups(records, 1) == 0
    records.append(record("t", "t/p1/2", 1, [right], stop="verified"))
    assert task_solved(records) and verified_groups(records, 1) == 1
    records.append(record("t", "t/p3/0", 3, [other], stop="verified"))
    assert verified_groups(records, 1) == 2
    assert INCOMPLETE == {"engine_error", "deadline"}


# ---------------------------------------------------------------- phases


def test_phases_order_budgets_effort_and_prior_with_a_fake_clock(tmp_path):
    widths = zip("ABCDE", (2, 3, 4, 5, 6), strict=True)
    tasks = {name: mirror_task(name, width) for name, width in widths}
    scripts = {
        "A/p1/0": [code_reply(MIRROR)],
        "B/p1/0": [code_reply(IDENTITY)],
        "B/p2/0": [code_reply(MIRROR)],
        "C/p1/0": [no_code_reply()],
        "C/p2/0": [no_code_reply()],
        "D/p1/0": [code_reply(IDENTITY)],
        "D/p2/0": [code_reply(IDENTITY)],
        "D/p3/0": [code_reply(IDENTITY)],
        "D/p3/1": [code_reply(MIRROR)],
        "E/p1/0": [code_reply(MIRROR)],
    }
    clock = FakeClock()
    engine = ClockEngine(ScriptedEngine(scripts), clock, rate=10.0)
    deadline = clock.now + 100_000
    confidence = {"A": 0.0, "B": 0.9, "C": 0.1, "D": 0.5, "E": 0.0}
    seen = []
    scheduler = Scheduler(
        list(tasks.values())[::-1], engine, FakeRepl, FakeVerifier(), settings(concurrency=1),
        deadline=deadline, clock=clock, confidence=confidence, log_path=tmp_path / "a.jsonl",
        on_record=seen.append,
    )  # fmt: skip
    report = run(scheduler)
    order = [
        "A/p1/0", "B/p1/0", "C/p1/0", "D/p1/0", "E/p1/0",  # every task, easiest first
        "C/p2/0", "D/p2/0", "B/p2/0",  # the unsolved, NVARC's least confident first
        "D/p3/0", "D/p3/1",  # partial progress, while time remains
    ]  # fmt: skip
    calls = engine.inner.calls
    assert [c.attempt_id for c in calls] == order
    assert report.stopped == "done" and report.failed == 0
    assert [r.attempt_id for r in seen] == order
    assert [r.attempt_id for r in read_records(tmp_path / "a.jsonl")[0]] == order
    # Effort per phase, and each call's cap is the attempt's budget.
    assert [c.reasoning_effort for c in calls] == ["low"] * 5 + ["xhigh"] * 5
    assert all(c.max_tokens == report.budgets[c.attempt_id] for c in calls)
    # Budgets: share * rate * time left / attempts left; the prior (400) before any call, then
    # the measured 10 tokens per second.
    # Phase 3 allows D two attempts, so its first counts 2 left.
    left = {"A/p1/0": 5, "B/p1/0": 4, "C/p1/0": 3, "D/p1/0": 2, "E/p1/0": 1, "C/p2/0": 3,
            "D/p2/0": 2, "B/p2/0": 1, "D/p3/0": 2, "D/p3/1": 1}  # fmt: skip
    assert {i: plan["remaining"] for i, plan in report.plans.items()} == left
    for attempt_id, count in left.items():
        share = 0.5 if "/p1/" in attempt_id else 1.0
        rate = 400.0 if attempt_id == "A/p1/0" else 10.0
        start = engine.starts[attempt_id]
        expected = int(share * rate * (deadline - start) / count)
        assert abs(report.budgets[attempt_id] - expected) <= 1, attempt_id
    # Phase 2 and 3 attempts see the failed programs; C has none; phase 1 never.
    by_id = {c.attempt_id: first_user(c) for c in calls}
    assert "Earlier programs" in by_id["B/p2/0"] and IDENTITY.strip() in by_id["B/p2/0"]
    assert "Earlier programs" in by_id["D/p3/1"]
    assert "Earlier programs" not in by_id["C/p2/0"] and "Earlier programs" not in by_id["B/p1/0"]
    assert dict(report.verified) == {1: 2, 2: 1, 3: 1}
    assert report.as_dict()["started"] == {"1": 5, "2": 3, "3": 2, "4": 0}


def test_phase_two_follows_ease_without_nvarc():
    tasks = [mirror_task(name, width) for name, width in zip("ABC", (2, 3, 4), strict=True)]
    engine = ScriptedEngine({})  # never any code: all unsolved
    scheduler = Scheduler(tasks, engine, FakeRepl, FakeVerifier(), settings(concurrency=1),
                          deadline=time.monotonic() + 600)  # fmt: skip
    run(scheduler)
    p2 = [c.attempt_id for c in engine.calls if "/p2/" in c.attempt_id]
    assert p2 == ["A/p2/0", "B/p2/0", "C/p2/0"]
    assert not [c for c in engine.calls if "/p3/" in c.attempt_id]  # no progress, no phase 3


def test_phase_one_votes_and_disagreeing_programs_reach_phase_three():
    # 1-row demonstrations: mirror and rot180 agree on them and differ on the 2-row test.
    task = mirror_task("X", 4, tall_test=True)
    scripts = {
        "X/p1/0": [code_reply(MIRROR)],
        "X/p1/1": [code_reply(ROT180)],
        "X/p3/0": [code_reply(MIRROR)],
        "X/p3/1": [code_reply(MIRROR)],
    }
    engine = ScriptedEngine(scripts)
    phases = (phase(attempts=2), phase(prior=True), phase(attempts=2, prior=True),
              phase(attempts=0))  # fmt: skip
    seen = []
    scheduler = Scheduler([task], engine, FakeRepl, FakeVerifier(),
                          settings(phases=phases, concurrency=2),
                          deadline=time.monotonic() + 600, on_record=seen.append)  # fmt: skip
    report = run(scheduler)
    ids = [c.attempt_id for c in engine.calls]
    assert ids == ["X/p1/0", "X/p1/1", "X/p3/0", "X/p3/1"]  # no phase 2: solved
    assert all("Earlier programs" not in first_user(c) for c in engine.calls)  # independent votes
    assert verified_groups(seen, 1) == 2 and report.verified[3] == 2


def test_budgets_count_attempts_in_flight_and_unsolved_tasks_still_in_phase_one():
    # Concurrency 2: A and B start together; C's phase-1 attempt is slow, so phase 2 starts on
    # A and B while C is still in phase 1, and N_2 must count C.
    tasks = [mirror_task(name, width) for name, width in (("A", 2), ("B", 3), ("C", 4))]
    scripts = {
        "A/p1/0": [ScriptedReply(text="No idea yet.")],
        "B/p1/0": [ScriptedReply(text="No idea yet.", delay_s=0.05)],
        "C/p1/0": [ScriptedReply(text="No idea yet.", delay_s=0.5)],
    }
    report = run(Scheduler(tasks, ScriptedEngine(scripts), FakeRepl, FakeVerifier(),
                           settings(concurrency=2), deadline=time.monotonic() + 600))  # fmt: skip
    remaining = {i: plan["remaining"] for i, plan in report.plans.items()}
    assert remaining == {
        "A/p1/0": 3, "B/p1/0": 3,  # B's count includes A, in flight
        "C/p1/0": 2,  # B in flight, and C
        "A/p2/0": 3,  # A, B, and C, which is still in its phase-1 attempt
        "B/p2/0": 2, "C/p2/0": 1,
    }  # fmt: skip
    sources = {plan["rate_source"] for plan in report.plans.values()}
    assert sources <= {"prior", "measured", "last"}


def test_shown_programs_are_capped_in_number_and_size():
    task = mirror_task("A", 3)
    # Statements, not comments: the verified source keeps only statements (P6.2).
    huge_identity = f'PAD = "{"x" * 30_000}"\n' + IDENTITY
    huge_other = f'PAD = "{"y" * 30_000}"\n' + "def transform(grid):\n    return [[0]]\n"
    scripts = {
        "A/p1/0": [code_reply(huge_other), code_reply(huge_identity)],
        "A/p2/0": [code_reply(MIRROR)],
    }
    engine = ScriptedEngine(scripts)
    phases = (phase(max_turns=2), phase(prior=True), phase(attempts=0), phase(attempts=0))
    scheduler = Scheduler([task], engine, FakeRepl, FakeVerifier(),
                          settings(phases=phases, prior_programs=1, prior_chars=4000),
                          deadline=time.monotonic() + 600)  # fmt: skip
    run(scheduler)
    message = first_user(next(c for c in engine.calls if c.attempt_id == "A/p2/0"))
    assert message.count("```python") == 1  # one program shown: the best, identity
    assert "characters cut" in message and "y" * 100 not in message
    assert len(message) < len(prompts.task_text(task)) + 4000 + 200
    clipped = clip_program(FakeVerifier().verify(task, huge_identity), 4000)
    assert len(clipped.source) <= 1000 and clipped.source.startswith('PAD = "xxx')
    assert clipped.source.rstrip().endswith("return grid")
    small = FakeVerifier().verify(task, IDENTITY)
    assert clip_program(small, 4000) is small


def prompt_bound(task, system, prior=()):
    messages = prompts.initial_messages(task, system=system, prior=prior)
    return sum(len(m.content) + MESSAGE_SLACK for m in messages) + PROMPT_SLACK


def test_an_attempt_without_room_for_a_reply_is_run_again_without_the_programs():
    task = mirror_task("A", 3)
    system = "Rules."
    shown = FakeVerifier().verify(task, IDENTITY)
    bare, crowded = prompt_bound(task, system), prompt_bound(task, system, (shown,))
    context = bare + 300  # room for a reply without the earlier program, none with it
    assert crowded > context - 256
    scripts = {"A/p1/0": [code_reply(IDENTITY)], "A/p2/1": [code_reply(MIRROR)]}
    engine = ScriptedEngine(scripts)
    log = []
    scheduler = Scheduler([task], engine, FakeRepl, FakeVerifier(),
                          settings(context_tokens=context), deadline=time.monotonic() + 600,
                          system=system, on_record=log.append)  # fmt: skip
    report = run(scheduler)
    stops = [(r.attempt_id, r.stop_reason, r.turns) for r in log]
    assert stops == [
        ("A/p1/0", "rounds", 1),
        ("A/p2/0", "budget", 0),  # the earlier program left no room: not a completed attempt
        ("A/p2/1", "verified", 1),  # run again without it
    ]
    retry = next(c for c in engine.calls if c.attempt_id == "A/p2/1")
    assert "Earlier programs" not in first_user(retry)
    assert report.stopped == "done"


def test_two_attempts_without_room_end_the_phase():
    task = mirror_task("A", 3)
    context = prompt_bound(task, "Rules.") - 1  # not even the task fits
    engine = ScriptedEngine({})
    log = []
    report = run(Scheduler([task], engine, FakeRepl, FakeVerifier(),
                           settings(context_tokens=context), deadline=time.monotonic() + 600,
                           system="Rules.", on_record=log.append))  # fmt: skip
    assert [(r.attempt_id, r.stop_reason, r.turns) for r in log] == [
        ("A/p1/0", "budget", 0), ("A/p1/1", "budget", 0),
        ("A/p2/0", "budget", 0), ("A/p2/1", "budget", 0),
    ]  # fmt: skip
    assert not engine.calls and report.stopped == "done"


def test_the_cycle_keeps_working_until_the_stop_and_the_submission_stays_valid(tmp_path):
    tasks = [mirror_task("U1", 3), mirror_task("U2", 4), mirror_task("S", 5), mirror_task("D", 4,
             tall_test=True)]  # fmt: skip
    d_task = tasks[3]
    # D's two verified programs disagree on its 2-row test input (seeded as a resumed log).
    mirror_out = [np.fliplr(np.array(d_task.test_inputs[0])).tolist()]
    rot_out = [np.rot90(np.array(d_task.test_inputs[0]), 2).tolist()]
    seeded = [
        record("D", "D/p1/0", 1, [verification(d_task, MIRROR, mirror_out, 3)], stop="verified"),
        record("D", "D/p1/1", 1, [verification(d_task, ROT180, rot_out, 3)], stop="verified"),
    ]
    # U1 and U2 are through phases 1 and 2 (a partial program each), so all three cycle tasks
    # are in the cycle from its start.
    for task in tasks[:2]:
        partial = verification(task, IDENTITY, [task.test_inputs[0]], passed=1)
        seeded += [
            record(task.task_id, f"{task.task_id}/p1/0", 1, [partial]),
            record(task.task_id, f"{task.task_id}/p2/0", 2),
        ]
    slow = ScriptedReply(text=f"```python\n{IDENTITY}```", delay_s=0.04)
    scripts = {"U1": [slow], "U2": [slow], "S": [code_reply(MIRROR)],
               "D": [ScriptedReply(text=f"```python\n{MIRROR}```", delay_s=0.04)]}  # fmt: skip
    engine = ScriptedEngine(scripts)
    out = tmp_path / "submission.json"
    counts = {t.task_id: 1 for t in tasks}
    writer = SubmissionWriter(out, counts, records=seeded)
    writer.write()
    valid = []

    def on_record(record):
        writer.add(record)
        valid.append(validate(json.loads(out.read_text()), {t: {"test": [{}]} for t in counts}))

    phases = (phase(), phase(prior=True), phase(attempts=0),
              phase(attempts=None, prior=True, reasoning_effort="xhigh"))  # fmt: skip
    stop = time.monotonic() + 1.5
    report = run(Scheduler(tasks, engine, FakeRepl, FakeVerifier(),
                           settings(phases=phases, concurrency=2, grace_s=2.0), deadline=stop,
                           records=seeded, on_record=on_record))  # fmt: skip
    assert report.stopped == "deadline"
    assert valid and all(problems == [] for problems in valid)
    cycle = [c for c in engine.calls if "/p4/" in c.attempt_id]
    assert {c.task_id for c in cycle} == {"U1", "U2", "D"}  # never the solved, agreeing S
    # An attempt started in the last instant may end at the deadline before its first call.
    assert len(cycle) >= 9 and report.started[4] >= len(cycle)
    per_task = {t: sum(1 for c in cycle if c.task_id == t) for t in ("U1", "U2", "D")}
    assert max(per_task.values()) - min(per_task.values()) <= 2  # round robin
    seeds = [c.seed for c in cycle if c.task_id == "U1"]
    assert len(set(seeds)) == len(seeds)  # fresh seeds
    assert all("Earlier programs" in first_user(c) for c in cycle if c.task_id != "D")
    assert all("Earlier programs" not in first_user(c) for c in cycle if c.task_id == "D")
    assert all(c.reasoning_effort == "xhigh" for c in cycle)
    # N_4, the tasks in the cycle: U1, U2, and D, busy or not (never S), one attempt each.
    assert {plan["remaining"] for i, plan in report.plans.items() if "/p4/" in i} == {3}
    # The cycle started early and kept starting attempts until the last moments before the stop.
    left_at_start = [plan["time_left_s"] for i, plan in report.plans.items() if "/p4/" in i]
    assert max(left_at_start) > 0.5 and min(left_at_start) < 0.4


def test_engine_errors_are_retried_once_with_a_new_id():
    tasks = [mirror_task("T", 3), mirror_task("U", 4)]
    scripts = {
        "T/p1/0": [error_reply()],
        "T/p1/1": [code_reply(MIRROR)],
        "U/p1/0": [error_reply()],
        "U/p1/1": [error_reply()],
        "U/p2/0": [error_reply()],
        "U/p2/1": [error_reply()],
    }
    engine = ScriptedEngine(scripts)
    report = run(Scheduler(tasks, engine, FakeRepl, FakeVerifier(), settings(concurrency=1),
                           deadline=time.monotonic() + 600))  # fmt: skip
    ids = [c.attempt_id for c in engine.calls]
    assert ids == ["T/p1/0", "U/p1/0", "T/p1/1", "U/p1/1", "U/p2/0", "U/p2/1"]
    assert report.stops[1]["engine_error"] == 3 and report.stops[2]["engine_error"] == 2
    assert report.stopped == "done"


def test_a_failing_sandbox_does_not_loop_forever(caplog):
    def broken(task):
        raise RuntimeError("cannot start a sandbox")

    tasks = [mirror_task("A", 3)]
    report = run(Scheduler(tasks, ScriptedEngine({}), broken, FakeVerifier(),
                           settings(concurrency=2), deadline=time.monotonic() + 600))  # fmt: skip
    assert report.stopped == "done"
    assert report.failed == 4  # phases 1 and 2, each tried twice (one retry)
    assert "cannot start a sandbox" in caplog.text


# ---------------------------------------------------------------- deadline, engine, kill


def nvarc_for(tasks):
    return nvarc_from_submission(
        {t.task_id: [{"attempt_1": solution(t)[0], "attempt_2": [[0]]}] for t in tasks}
    )


def test_the_deadline_stops_attempts_and_leaves_a_valid_submission(tmp_path):
    tasks = [mirror_task(name, 3 + i) for i, name in enumerate("ABCD")]
    slow = [ScriptedReply(text="Still looking.", delay_s=0.3)] * 5
    engine = ScriptedEngine({t.task_id: slow for t in tasks})
    counts = {t.task_id: 1 for t in tasks}
    out = tmp_path / "submission.json"
    writer = SubmissionWriter(out, counts, nvarc_for(tasks[:2]))
    writer.write()
    long = phase(max_turns=5, min_attempt_tokens=10_000)
    scheduler = Scheduler(
        tasks, engine, FakeRepl, FakeVerifier(),
        settings(phases=(long, long, long, long), concurrency=2, grace_s=5.0),
        deadline=time.monotonic() + 0.5, log_path=tmp_path / "a.jsonl", on_record=writer.add,
    )  # fmt: skip
    began = time.monotonic()
    report = run(scheduler)
    assert time.monotonic() - began < 3.0
    assert report.stopped == "deadline" and report.cancelled == 0
    logged, _ = read_records(tmp_path / "a.jsonl")
    assert {r.stop_reason for r in logged} == {"deadline"} and len(logged) == 2
    written = json.loads(out.read_text())
    assert validate(written, {t.task_id: {"test": [{}]} for t in tasks}) == []
    assert written["A"][0]["attempt_1"] == solution(tasks[0])[0]  # NVARC's fill stays
    assert written["D"][0] == {"attempt_1": [[0]], "attempt_2": [[0]]}


def test_attempts_stuck_past_the_grace_are_cancelled():
    class StuckRepl(FakeRepl):
        def run_cell(self, code, timeout_s):
            time.sleep(1.5)
            return super().run_cell(code, timeout_s)

    tasks = [mirror_task("A", 3)]
    engine = ScriptedEngine({"A": [code_reply(MIRROR)]})
    scheduler = Scheduler(tasks, engine, StuckRepl, FakeVerifier(),
                          settings(concurrency=1, grace_s=0.2),
                          deadline=time.monotonic() + 0.2)  # fmt: skip
    began = time.monotonic()
    report = run(scheduler)
    assert time.monotonic() - began < 1.4
    assert report.cancelled == 1 and report.stopped == "deadline"


class Gate:
    def __init__(self, answers):
        self.answers = list(answers)

    async def wait_ready(self):
        return self.answers.pop(0) if self.answers else False


def test_an_engine_gone_for_good_stops_the_scheduler():
    tasks = [mirror_task(name, 3 + i) for i, name in enumerate("ABC")]
    engine = ScriptedEngine({t.task_id: [code_reply(MIRROR)] for t in tasks})
    report = run(Scheduler(tasks, engine, FakeRepl, FakeVerifier(), settings(concurrency=1),
                           deadline=time.monotonic() + 600, gate=Gate([True])))  # fmt: skip
    assert report.stopped == "engine"
    assert [c.attempt_id for c in engine.calls] == ["A/p1/0"]


def test_a_gate_that_never_answers_stops_at_the_deadline():
    class Silent:
        async def wait_ready(self):
            await asyncio.sleep(60)

    tasks = [mirror_task("A", 3)]
    began = time.monotonic()
    report = run(Scheduler(tasks, ScriptedEngine({}), FakeRepl, FakeVerifier(), settings(),
                           deadline=time.monotonic() + 0.3, gate=Silent()))  # fmt: skip
    assert report.stopped == "deadline" and time.monotonic() - began < 2


def test_nothing_left_to_start_does_not_wait_on_the_engine():
    class Silent:  # an engine that is restarting
        async def wait_ready(self):
            await asyncio.sleep(60)

    task = mirror_task("A", 3)
    right = verification(task, MIRROR, solution(task), passed=3)
    done = [record("A", "A/p1/0", 1, [right], stop="verified")]
    began = time.monotonic()
    scheduler = Scheduler([task], ScriptedEngine({}), FakeRepl, FakeVerifier(), settings(),
                          deadline=time.monotonic() + 30, records=done, gate=Silent())  # fmt: skip
    report = run(scheduler)
    assert report.stopped == "done" and report.resumed == 1
    assert time.monotonic() - began < 1


def test_kill_and_resume_from_the_jsonl(tmp_path):
    tasks = [mirror_task(name, 2 + i) for i, name in enumerate("ABCDEF")]
    scripts = {}
    for t in tasks:
        first = [ScriptedReply(text=f"```python\n{IDENTITY}```", delay_s=0.05)]
        scripts[f"{t.task_id}/p1/0"] = first
        scripts[f"{t.task_id}/p2/0"] = [code_reply(MIRROR)]
    log = tmp_path / "attempts.jsonl"
    out = tmp_path / "submission.json"
    counts = {t.task_id: 1 for t in tasks}

    async def killed_run():
        engine = ScriptedEngine(scripts)
        writer = SubmissionWriter(out, counts)
        scheduler = Scheduler(tasks, engine, FakeRepl, FakeVerifier(), settings(concurrency=2),
                              deadline=time.monotonic() + 600, log_path=log,
                              on_record=writer.add)  # fmt: skip
        job = asyncio.ensure_future(scheduler.run())
        while len(read_records(log)[0]) < 3 if log.exists() else True:
            await asyncio.sleep(0.01)
        job.cancel()  # the kill: no cleanup beyond what is already on disk
        with pytest.raises(asyncio.CancelledError):
            await job
        return engine

    first_engine = asyncio.run(killed_run())
    done_before = {r.attempt_id for r in read_records(log)[0]}
    assert len(done_before) >= 3
    written = json.loads(out.read_text())
    assert validate(written, {t: {"test": [{}]} for t in counts}) == []
    with open(log, "a", encoding="utf-8") as handle:
        handle.write('{"task_id": "cut mid-wri')  # a write cut by the kill

    records, skipped = read_records(log)
    assert skipped == 1
    engine = ScriptedEngine(scripts)
    writer = SubmissionWriter(out, counts, records=records)
    scheduler = Scheduler(tasks, engine, FakeRepl, FakeVerifier(), settings(concurrency=2),
                          deadline=time.monotonic() + 600, records=records, log_path=log,
                          on_record=writer.add)  # fmt: skip
    report = run(scheduler)
    assert report.resumed == len(records) and report.stopped == "done"
    rerun = {c.attempt_id for c in engine.calls}
    assert not rerun & done_before  # finished attempts are not run again
    final, skipped = read_records(log)
    ids = [r.attempt_id for r in final]
    assert len(ids) == len(set(ids)) and skipped == 1
    assert {i for i in ids if "/p1/" in i} == {f"{t.task_id}/p1/0" for t in tasks}
    assert {i for i in ids if "/p2/" in i} == {f"{t.task_id}/p2/0" for t in tasks}
    written = json.loads(out.read_text())
    assert validate(written, {t: {"test": [{}]} for t in counts}) == []
    assert all(written[t.task_id][0]["attempt_1"] == solution(t)[0] for t in tasks)
    assert first_engine.calls  # the first run did work


# ---------------------------------------------------------------- the submission writer


def test_submission_writer_fill_updates_and_failed_writes(tmp_path):
    tasks = [mirror_task("A", 3), mirror_task("B", 4), mirror_task("C", 5)]
    counts = {"A": 1, "B": 1, "C": 1}
    nvarc = nvarc_from_submission(
        {
            "A": [{"attempt_1": [[7]], "attempt_2": [[8]]}],
            "Z": [{"attempt_1": [[1]], "attempt_2": [[1]]}],
        }
    )
    out = tmp_path / "sub" / "submission.json"
    writer = SubmissionWriter(out, counts, nvarc)
    assert writer.write() and writer.writes == 1
    written = json.loads(out.read_text())
    assert list(written) == ["A", "B", "C"]
    assert written["A"] == [{"attempt_1": [[7]], "attempt_2": [[8]]}]
    assert written["B"] == [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    assert writer.slots() == {"program": 0, "direct": 0, "nvarc": 2, "empty": 4}
    right = verification(tasks[0], MIRROR, solution(tasks[0]), passed=3)
    writer.add(record("A", "A/p1/0", 1, [right], stop="verified"))
    writer.add(record("B", "B/p1/0", 1, answers=[(0, [[3, 3]])]))
    writer.add(record("Q", "Q/p1/0", 1))  # not a task of the challenges: ignored
    written = json.loads(out.read_text())
    assert written["A"] == [{"attempt_1": solution(tasks[0])[0], "attempt_2": [[7]]}]
    assert written["B"][0]["attempt_1"] == [[3, 3]]
    assert writer.writes == 3
    assert not list(out.parent.glob(".*.tmp"))
    # A write that fails leaves the previous file and is counted.
    before = out.read_text()
    writer.path = tmp_path / "missing-dir" / "x" / "submission.json"
    writer.path.parent.parent.mkdir()
    writer.path.parent.write_text("a file where a folder should be")
    assert writer.write() is False and writer.failures == 1 and writer.last_error
    assert out.read_text() == before


def test_submission_writer_resumes_from_records(tmp_path):
    task = mirror_task("A", 3)
    right = verification(task, MIRROR, solution(task), passed=3)
    done = [record("A", "A/p1/0", 1, [right], stop="verified")]
    writer = SubmissionWriter(tmp_path / "s.json", {"A": 1}, records=done)
    assert writer.submission()["A"][0]["attempt_1"] == solution(task)[0]


def cycle_phases():
    return (phase(), phase(prior=True), phase(attempts=0), phase(attempts=None, prior=True))


def test_the_cycle_drops_a_task_only_after_consecutive_errored_rounds():
    # U's cycle attempts fail every other round, V's every round: U keeps cycling until the stop
    # (errors hours apart never add up), V leaves after 2 consecutive errored rounds.
    tasks = [mirror_task("U", 3), mirror_task("V", 4)]
    quiet = ScriptedReply(text="No idea yet.", delay_s=0.02)
    failing = ScriptedReply(error="engine unavailable", delay_s=0.02)
    scripts = {f"{t}/p{n}/0": [quiet] for t in ("U", "V") for n in (1, 2)}
    for k in range(400):
        scripts[f"U/p4/{k}"] = [failing if k % 2 == 0 else quiet]
        scripts[f"V/p4/{k}"] = [failing]
    engine = ScriptedEngine(scripts)
    report = run(Scheduler(tasks, engine, FakeRepl, FakeVerifier(),
                           settings(phases=cycle_phases(), concurrency=2, grace_s=2.0),
                           deadline=time.monotonic() + 1.0))  # fmt: skip
    assert report.stopped == "deadline"
    v_cycle = [c.attempt_id for c in engine.calls if c.attempt_id.startswith("V/p4/")]
    assert v_cycle == ["V/p4/0", "V/p4/1"]
    u_cycle = [c.attempt_id for c in engine.calls if c.attempt_id.startswith("U/p4/")]
    assert len(u_cycle) >= 6  # far more than 2 errors, never 2 in a row
    left = [plan["time_left_s"] for i, plan in report.plans.items() if i.startswith("U/p4/")]
    assert min(left) < 0.4  # still cycling at the stop


def test_a_resume_rebuilds_the_cycles_error_streak():
    tasks = [mirror_task("U", 3), mirror_task("W", 4)]
    done = [record(t, f"{t}/p{n}/0", n) for t in ("U", "W") for n in (1, 2)]
    done += [
        record("U", "U/p4/0", 4, stop="engine_error"),
        record("U", "U/p4/1", 4, stop="engine_error"),  # two errored rounds in a row: dropped
        record("W", "W/p4/0", 4, stop="engine_error"),
        record("W", "W/p4/1", 4),
        record("W", "W/p4/2", 4, stop="engine_error"),  # one errored round since the last good
    ]
    engine = ScriptedEngine({})
    report = run(Scheduler(tasks, engine, FakeRepl, FakeVerifier(),
                           settings(phases=cycle_phases(), concurrency=1),
                           deadline=time.monotonic() + 0.3, records=done))  # fmt: skip
    ids = [c.attempt_id for c in engine.calls]
    assert ids and ids[0] == "W/p4/3" and all(i.startswith("W/p4/") for i in ids)
    assert report.resumed == len(done)
