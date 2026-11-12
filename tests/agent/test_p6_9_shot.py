"""Tests of task P6.9's shot options (arc26.agent.loop, engine, mock engine, fake engine, shot).

What they prove: with every new key absent the shot, the loop, and the engine behave and log as
Submission 3's did; the in-request forcing sends the cap as vLLM's thinking budget with the
answer's room after it, counts the turns the server ended, and falls back to the separate forcing
turn only when a cut reply holds no readable answer; the lenient parser reads digit-string rows,
string cells, and unbalanced digit-string rows, refuses what the strict rules refuse otherwise,
and never reads a strict-valid block differently; the bucket order, the implied-shape choice of
S1, the refill queue's plans and stop rules; and, end to end with P6.4's fake engine (which
honors the budget as vLLM does), a refill run with in-request forcing that merges and scores. The
Submission 4 candidate config differs from Submission 3's only in the listed keys. No GPU, no
network beyond localhost, no competition data.
"""

import asyncio
import json
import random
from collections import deque
from pathlib import Path

import pytest
import yaml

from arc26.agent import shot
from arc26.agent.engine import EngineConfig, OpenAIEngine
from arc26.agent.loop import (
    Budget,
    parse_answer,
    record_from_json,
    record_to_json,
    run_attempt,
)
from arc26.agent.mock_engine import ScriptedEngine, ScriptedReply, answer_reply
from arc26.agent.run import ConfigError
from arc26.agent.server import fake_command, fake_completion, reasoning_end_str
from arc26.agent.types import AgentTask, AttemptRecord, Pair

REPO = Path(__file__).resolve().parents[2]
S3 = REPO / "configs" / "agent_shot.yaml"
S4 = REPO / "configs" / "agent_shot_P6.9.yaml"
NOTE = "\n\nI have used my whole thinking budget, so I stop reasoning here and write my final answer blocks now, best first.\n"  # noqa: E501
GRID = [[1, 2, 3], [4, 5, 6]]


def task(test=((0, 1, 2), (3, 4, 5)), train=None):
    grid = [list(r) for r in test]
    pairs = train or (Pair([[1, 1], [2, 2]], [[2, 2], [1, 1]]),)
    return AgentTask("aaaa0001", tuple(pairs), (grid,))


class NoRepl:
    def run_cell(self, code, timeout_s):
        from arc26.agent.types import CellResult

        return CellResult("", "no repl", False, 0.0, False)

    def close(self):
        pass


class NoVerifier:
    def verify(self, task, source):
        raise RuntimeError("no programs")


def attempt(engine, budget, the_task=None):
    return asyncio.run(
        run_attempt(the_task or task(), engine, NoRepl(), NoVerifier(), budget, phase=0, seed=0)
    )


def in_request_budget(**changes):
    values = dict(max_turns=1, attempt_tokens=1024 + 2 * 256, turn_tokens=1024,
                  context_tokens=131072, forcing_tokens=256, forcing_kind="answer2",
                  preserve_thinking=True, forcing_in_request=True,
                  thinking_end_note=NOTE)  # fmt: skip
    values.update(changes)
    return Budget(**values)


# --- the engine and the fake engine -------------------------------------------------------------


def test_the_request_carries_the_budget_only_when_given():
    engine = OpenAIEngine(EngineConfig())
    messages = (__import__("arc26.agent.types", fromlist=["Message"]).Message("user", "hi"),)
    plain = engine.request_body(messages, max_tokens=10, temperature=1.0, seed=0)
    assert "thinking_token_budget" not in plain
    body = engine.request_body(
        messages, max_tokens=10, temperature=1.0, seed=0, thinking_token_budget=7
    )
    assert body["thinking_token_budget"] == 7
    assert {k: v for k, v in body.items() if k != "thinking_token_budget"} == plain


def test_the_fake_engine_ends_the_reasoning_at_the_budget_and_answers():
    replies = {"aaaa0001": [{"reasoning": "x" * 4000, "text": "the answer"}]}
    body = {"messages": [{"role": "user", "content": "Task aaaa0001: ..."}], "max_tokens": 2000}
    cut, _ = fake_completion(replies, {**body, "thinking_token_budget": 100}, budget_note=NOTE)
    message = cut["choices"][0]["message"]
    assert message["reasoning"] == "x" * 400 + NOTE and message["content"] == "the answer"
    assert cut["choices"][0]["finish_reason"] == "stop"
    whole, _ = fake_completion(replies, body, budget_note=NOTE)
    assert whole["choices"][0]["message"]["reasoning"] == "x" * 4000
    assert "--budget-note" not in fake_command(8000)
    assert fake_command(8000, budget_note="n")[-2:] == ["--budget-note", "n"]


def test_the_reasoning_config_is_read_from_the_extra_flags():
    flag = json.dumps({"reasoning_end_str": "done.</think>"})
    assert reasoning_end_str(["--reasoning-config", flag]) == "done.</think>"
    assert reasoning_end_str([]) is None
    for bad in (["--reasoning-config"], ["--reasoning-config", "{"],
                ["--reasoning-config", "[]"], ["--reasoning-config", flag] * 2):  # fmt: skip
        with pytest.raises(ValueError):
            reasoning_end_str(bad)


# --- the loop's in-request forcing --------------------------------------------------------------


def test_in_request_forcing_needs_a_forcing_cap_and_an_end_note():
    with pytest.raises(ValueError, match="thinking_end_note"):
        Budget(forcing_in_request=True, forcing_tokens=256)
    with pytest.raises(ValueError, match="thinking_end_note"):
        Budget(forcing_in_request=True, thinking_end_note=NOTE)
    assert in_request_budget().in_request_tokens() == 256
    assert Budget(forcing_tokens=256).in_request_tokens() == 0


def test_the_answer_comes_in_the_thinking_request_and_the_forced_end_is_counted():
    long = "the rows flip " * 2000
    engine = ScriptedEngine(
        {"aaaa0001": [ScriptedReply(text=answer_reply(0, GRID).text, reasoning=long)]},
        budget_note=NOTE,
    )
    record = attempt(engine, in_request_budget())
    (call,) = engine.calls
    assert (call.thinking_token_budget, call.max_tokens, call.thinking) == (1024, 1280, None)
    assert record.direct_answers == ((0, GRID),) and record.forcing_turns == 0
    assert record.budget_forced == 1 and record.turns == 1


def test_a_reply_the_server_did_not_end_gets_the_separate_forcing_turn():
    # A server that ignores the budget (no --reasoning-config): the reasoning runs to the cap.
    class Ignoring(ScriptedEngine):
        async def complete(self, messages, *, thinking_token_budget=None, **kwargs):
            return await super().complete(messages, **kwargs)

    long = "the rows flip " * 2000
    engine = Ignoring({"aaaa0001": [ScriptedReply(text="", reasoning=long),
                                    answer_reply(0, GRID)]})  # fmt: skip
    record = attempt(engine, in_request_budget())
    first, second = engine.calls
    assert first.max_tokens == 1280 and second.thinking is False and second.max_tokens == 256
    assert record.forcing_turns == 1 and record.budget_forced == 0
    assert record.direct_answers == ((0, GRID),)


def test_a_cut_reply_with_a_readable_answer_gets_no_second_request():
    # The answer outran its room but one block is complete: no re-prefill for a second one.
    text = answer_reply(0, GRID).text + "```answer\n{\"test_index\": 0, \"grid\": [[1, 2"
    engine = ScriptedEngine({"aaaa0001": [ScriptedReply(text=text * 30, reasoning="r")]})
    record = attempt(engine, in_request_budget(turn_tokens=256, attempt_tokens=256 + 512))
    assert len(engine.calls) == 1 and record.forcing_turns == 0
    assert record.direct_answers[0] == (0, GRID)


def test_too_little_room_for_the_budget_and_the_answer_stops_with_budget():
    engine = ScriptedEngine()
    record = attempt(engine, in_request_budget(context_tokens=600 + 256 + 2000))
    assert engine.calls == [] and record.stop_reason == "budget"


def test_without_the_option_the_requests_are_submission_3s():
    long = "the rows flip " * 2000
    replies = [ScriptedReply(text="", reasoning=long), answer_reply(0, GRID)]
    old = Budget(max_turns=1, attempt_tokens=1024 + 256, turn_tokens=1024, context_tokens=131072,
                 forcing_tokens=256, forcing_kind="answer2", preserve_thinking=True)  # fmt: skip
    engine = ScriptedEngine({"aaaa0001": list(replies)})
    record = attempt(engine, old)
    assert [c.thinking_token_budget for c in engine.calls] == [None, None]
    assert [c.max_tokens for c in engine.calls] == [1024, 256]
    data = record_to_json(record)
    assert "budget_forced" not in data and "answer_lenient" not in data
    assert record_from_json(data) == record


class Refusing(ScriptedEngine):
    """vLLM without a reasoning config: HTTP 400 for any request with a thinking budget."""

    async def complete(self, messages, *, thinking_token_budget=None, **kwargs):
        if thinking_token_budget is not None:
            self.refused = getattr(self, "refused", 0) + 1
            raise RuntimeError(
                "HTTP 400: {\"error\": {\"message\": \"thinking_token_budget is set but "
                "reasoning_config is not configured.\"}}"
            )
        return await super().complete(messages, **kwargs)


def test_a_refused_budget_field_is_sent_once_more_without_it():
    long = "the rows flip " * 2000
    engine = Refusing({"aaaa0001": [ScriptedReply(text="", reasoning=long),
                                    answer_reply(0, GRID)]})  # fmt: skip
    record = attempt(engine, in_request_budget())
    assert engine.refused == 1 and record.budget_refused == 1
    # Then Submission 3's request: the cap alone, and the separate forcing turn.
    first, second = engine.calls
    assert (first.thinking_token_budget, first.max_tokens) == (None, 1024)
    assert second.thinking is False and record.forcing_turns == 1
    assert record.direct_answers == ((0, GRID),) and record.stop_reason == "rounds"
    assert record_to_json(record)["budget_refused"] == 1


def test_another_engine_error_still_ends_the_attempt():
    class Failing(ScriptedEngine):
        async def complete(self, messages, **kwargs):
            raise RuntimeError("HTTP 500: boom")

    record = attempt(Failing(), in_request_budget())
    assert record.stop_reason == "engine_error" and record.budget_refused == 0


def test_the_client_side_cut_drops_the_request_and_answers_from_the_opening():
    # A thinking request that hangs past turn_cut_s: dropped, then a forcing turn on the opening.
    class Hanging(ScriptedEngine):
        async def complete(self, messages, *, thinking=None, **kwargs):
            if thinking is None:
                await asyncio.sleep(5)
            return await super().complete(messages, thinking=thinking, **kwargs)

    engine = Hanging({"aaaa0001": [answer_reply(0, GRID)]})
    record = attempt(engine, in_request_budget(turn_cut_s=0.2))
    assert record.turn_cuts == 1 and record.forcing_turns == 1
    forced = engine.calls[-1]
    assert forced.thinking is False and forced.max_tokens == 256
    assert [m.role for m in forced.messages] == ["system", "user", "user"]
    assert len(engine.calls) == 1  # the dropped request never answered; the forcing turn did
    assert record.direct_answers == ((0, GRID),)
    with pytest.raises(ValueError, match="turn_cut_s"):
        Budget(turn_cut_s=10.0)


def test_a_reply_past_an_ignored_budget_is_forced_without_its_reasoning():
    class Ignoring(ScriptedEngine):
        async def complete(self, messages, *, thinking_token_budget=None, **kwargs):
            return await super().complete(messages, **kwargs)

    long = "the rows flip " * 2000
    engine = Ignoring({"aaaa0001": [ScriptedReply(text="", reasoning=long),
                                    answer_reply(0, GRID)]})  # fmt: skip
    attempt(engine, in_request_budget())
    forced = engine.calls[-1]
    assert forced.thinking is False
    assert all(m.reasoning is None for m in forced.messages)


def test_the_probe_turns_the_in_request_forcing_off_on_a_refusal_or_no_forced_end():
    ok = asyncio.run(shot.budget_probe(
        ScriptedEngine({"budget-check": [ScriptedReply(text="42", reasoning="x" * 400)]},
                       budget_note=NOTE), NOTE, 0, 10.0))  # fmt: skip
    assert ok["ok"] and ok["forced"]
    refused = asyncio.run(shot.budget_probe(Refusing(), NOTE, 0, 10.0))
    assert not refused["ok"] and "400" in refused["error"]
    ignored = asyncio.run(shot.budget_probe(
        ScriptedEngine({"budget-check": [ScriptedReply(text="42", reasoning="x" * 400)]}),
        NOTE, 0, 10.0))  # fmt: skip
    assert not ignored["ok"] and not ignored["forced"]
    config = shot.without_in_request(shot.config_from_mapping(in_request_mapping()))
    assert not config.request.forcing_in_request and not config.waves.forcing_in_request
    assert config.request.budget(94208).attempt_tokens == 94208 + 6144


# --- the lenient parser -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, grid",
    [
        ('{"test_index": 0, "grid": ["012", "345"]}', [[0, 1, 2], [3, 4, 5]]),
        ('{"test_index": 0, "grid": ["0 1 2", "3,4,5"]}', [[0, 1, 2], [3, 4, 5]]),
        # Space-separated digits are cells, never two-digit numbers.
        ('{"test_index": 0, "grid": ["01 02", "34 50"]}', [[0, 1, 0, 2], [3, 4, 5, 0]]),
        ('{"test_index": 0, "grid": [["0","1","2"], ["3", 4, "5"]]}', [[0, 1, 2], [3, 4, 5]]),
        ('["012", "345"]', [[0, 1, 2], [3, 4, 5]]),
        # P6.5-02's 36a08778: rows as strings and a closing bracket missing.
        ('{"test_index": 0, "grid": [\n["012",\n "345"]}', [[0, 1, 2], [3, 4, 5]]),
    ],
)  # fmt: skip
def test_the_lenient_rules_read_the_observed_forms(body, grid):
    with pytest.raises(ValueError):
        parse_answer(body, 1)
    assert parse_answer(body, 1, lenient=True) == (0, grid)


@pytest.mark.parametrize(
    "body",
    [
        '{"test_index": 0, "grid": ["012", "34"]}',  # ragged
        '{"test_index": 0, "grid": ["' + "1" * 31 + '"]}',  # too wide
        '{"test_index": 0, "grid": ["01a"]}',  # not digits
        '{"test_index": 0, "grid": [["12"]]}',  # a two-digit cell
        '{"test_index": 0, "grid": [[0, 0, 0, 3][0:2]]}',  # P6.5-03's slicing
        '{"test_index": 5, "grid": ["01"]}',  # index out of range
        '{"test_index": 0, "grid": [\n["01", 2,\n "34"]}',  # mixed rows, not salvaged
    ],
)
def test_the_lenient_rules_refuse_the_rest(body):
    with pytest.raises(ValueError):
        parse_answer(body, 1, lenient=True)


def test_a_strict_valid_block_is_read_the_same_way_leniently():
    rng = random.Random(0)
    for _ in range(300):
        h, w = rng.randint(1, 30), rng.randint(1, 30)
        grid = [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]
        for body in (json.dumps({"test_index": 0, "grid": grid}), json.dumps(grid),
                     json.dumps({"test_index": 0, "grid": grid}, indent=1)):  # fmt: skip
            assert parse_answer(body, 1, lenient=True) == parse_answer(body, 1)


def test_the_loop_flags_the_answers_only_the_lenient_parser_read():
    text = answer_reply(0, GRID).text + '```answer\n{"test_index": 0, "grid": ["789"]}\n```\n'
    budget = Budget(max_turns=1, attempt_tokens=4096, turn_tokens=2048, lenient_answers=True)
    record = attempt(ScriptedEngine({"aaaa0001": [text]}), budget)
    assert record.direct_answers == ((0, GRID), (0, [[7, 8, 9]]))
    assert record.answer_lenient == (False, True)
    data = record_to_json(record)
    assert data["answer_lenient"] == [False, True] and record_from_json(data) == record
    strict = attempt(ScriptedEngine({"aaaa0001": [text]}), Budget(max_turns=1,
                     attempt_tokens=4096, turn_tokens=2048))  # fmt: skip
    assert strict.direct_answers == ((0, GRID),) and strict.answer_lenient == ()
    with pytest.raises(ValueError, match="one flag per"):
        record_from_json({**data, "answer_lenient": [True]})


# --- the shot: selection, shapes, plans -----------------------------------------------------------


CONF = {"a": 0.0, "b": 0.45, "c": 0.38, "d": 0.0, "e": 0.55, "f": 0.69, "g": 0.75, "h": None,
        "i": 0.38}  # fmt: skip


def test_submission_3s_order_is_unchanged_by_default():
    assert shot.select_tasks(CONF, 0.5) == [("a", 0.0), ("d", 0.0), ("c", 0.38), ("i", 0.38),
                                            ("b", 0.45)]  # fmt: skip


def test_the_bucket_order_puts_the_middle_bucket_first_then_zero():
    assert shot.select_tasks(CONF, 0.5, order="buckets") == [
        ("c", 0.38), ("i", 0.38), ("b", 0.45), ("a", 0.0), ("d", 0.0)
    ]  # fmt: skip
    # Option 2c's order with the switch on: (0, 0.5), then [0.5, 0.7), then 0.
    mid = shot.select_tasks(CONF, 0.7, exclude={"c"}, order="buckets", mid_bucket=True)
    assert mid == [("i", 0.38), ("b", 0.45), ("e", 0.55), ("f", 0.69), ("a", 0.0), ("d", 0.0)]
    assert [shot.confidence_bucket(c, True) for c in (0.3, 0.6, 0.0, 0.8)] == [0, 1, 2, None]
    # max_confidence still caps the queue; 0.75 is in no bucket.
    assert shot.select_tasks(CONF, 1.0, order="buckets", mid_bucket=True)[-3:] == [
        ("f", 0.69), ("a", 0.0), ("d", 0.0)
    ]  # fmt: skip
    with pytest.raises(ValueError, match="order"):
        shot.select_tasks(CONF, 0.5, order="random")


def test_the_implied_shape_rules():
    same = (Pair([[1, 2]], [[2, 1]]), Pair([[1], [2]], [[2], [1]]))
    fixed = (Pair([[1, 2]], [[0, 0, 0]]), Pair([[1], [2]], [[1, 1, 1]]))
    open_ = (Pair([[1, 2]], [[0]]), Pair([[1], [2]], [[1, 1]]))
    test = (((1, 2, 3),),)
    assert shot.implied_shapes(task(test[0], same)) == ((1, 3),)
    assert shot.implied_shapes(task(test[0], fixed)) == ((1, 3),)
    assert shot.implied_shapes(task(((1, 2),), fixed)) == ((1, 3),)
    assert shot.implied_shapes(task(test[0], open_)) == (None,)


def record_of(*answers):
    return AttemptRecord("t", "t/p0/0", 0, (), tuple((0, g) for g in answers), 0, 0, 1, 0.0,
                         "rounds")  # fmt: skip


def test_s1_is_the_first_answer_of_the_implied_shape_in_the_models_order():
    draft, best, other = [[1, 1]], [[2, 2, 2]], [[3, 3, 3]]
    records = [record_of(draft, best, other)]
    assert shot.shot_answers(records, 0) == [draft, best]
    # The draft of another size is skipped; the model's best-first order is kept after it.
    assert shot.shot_answers(records, 0, (1, 3)) == [best, other]
    # abc82100's form: two answers of the implied shape keep their order.
    assert shot.shot_answers([record_of(best, other)], 0, (1, 3)) == [best, other]
    # Nothing distinct after S1: the first distinct answer before it.
    assert shot.shot_answers([record_of(draft, best)], 0, (1, 3)) == [best, draft]
    # No answer of the shape: Submission 3's choice.
    assert shot.shot_answers(records, 0, (5, 5)) == [draft, best]
    assert shot.shot_answers([record_of(best, best)], 0, (1, 3)) == [best]
    assert shot.shot_answers([], 0, (1, 3)) == []


def test_the_writer_uses_the_shapes_only_when_given(tmp_path):
    draft, final = [[1, 1]], [[2, 2, 2]]
    records = [AttemptRecord("x", "x/p0/0", 0, (), ((0, draft), (0, final)), 0, 0, 1, 0.0,
                             "rounds")]  # fmt: skip
    nvarc = {"x": [([[9]], [[8]])]}
    plain = shot.ShotWriter(tmp_path / "a.json", {"x": 1}, nvarc, records)
    shaped = shot.ShotWriter(tmp_path / "b.json", {"x": 1}, nvarc, records, {"x": [(1, 3)]})
    assert plain.submission()["x"][0]["attempt_2"] == draft
    assert shaped.submission()["x"][0]["attempt_2"] == final


def model(**changes):
    values = dict(wave_tasks=15, max_cap=94208, min_cap=65536, plan_factor=0.85,
                  rate_tokens_per_s=19.0, prefill_tokens_per_s=958.8, opening_tokens=6771,
                  margin_s=300, kv_pool_tokens=1699566, setup_s=450,
                  forcing_tokens=6144)  # fmt: skip
    values.update(changes)
    return shot.WaveModel(**values)


def test_in_request_forcing_drops_the_prefill_term():
    separate, inside = model(), model(forcing_in_request=True)
    assert inside.forcing_seconds(94208, 15, 6771) == pytest.approx(6144 / 19.0)
    assert separate.forcing_seconds(94208, 15, 6771) == pytest.approx(
        15 * (6771 + 94208) / 958.8 + 6144 / 19.0
    )
    full = inside.wave_seconds(94208, 15, 6771)
    assert full == pytest.approx(94208 / (0.85 * 19.0) + 6144 / 19.0 + 300)
    assert inside.largest_cap(full, 15, 6771) == 94208
    # In-request, the cap does not depend on the streams.
    assert inside.largest_cap(5000, 1, 6771) == inside.largest_cap(5000, 15, 6771)


def test_a_refill_start_plans_one_task_against_the_streams():
    inside = model(forcing_in_request=True)
    start = shot.plan_start(("t", 5000), 10000, inside, 1699566, 7)
    assert start.tasks == ("t",) and start.cap == 94208 and start.reason is None
    assert start.kv_need == 5000 + 94208 + 6144
    short = shot.plan_start(("t", 5000), 3000, inside, 1699566, 7)
    assert short.tasks == () and short.reason.startswith("time")
    full = shot.plan_start(("t", 5000), 10000, inside, 50000, 7)
    assert full.tasks == () and full.reason.startswith("kv") and full.dropped == ("t",)
    # Without in-request forcing the forced replies of all running streams are prefilled.
    separate = model()
    assert shot.plan_start(("t", 5000), 5500, separate, 1699566, 15).tasks == ()
    assert shot.plan_start(("t", 5000), 5500, separate, 1699566, 1).tasks == ("t",)


def test_the_refill_stop_rules_read_the_last_wave_of_finished_tasks():
    m = model(wave_tasks=4)
    recent = deque([(True, False), (True, False), (False, False)], maxlen=4)
    assert shot._refill_stop(recent, 5, m, 0.5) is None
    recent.append((True, False))
    assert "degenerate" in shot._refill_stop(recent, 5, m, 0.5)
    errors = deque([(False, True)] * 2, maxlen=4)
    assert shot._refill_stop(errors, 4, m, 0.5) is None
    assert "engine error" in shot._refill_stop(errors, 2, m, 0.5)


# --- the configs ----------------------------------------------------------------------------------


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


def test_submission_3s_config_keeps_every_default():
    config = shot.load_config(S3)
    assert (config.order, config.mid_bucket) == ("confidence", False)
    assert not config.waves.refill and not config.waves.forcing_in_request
    request = config.request
    assert not request.forcing_in_request and not request.lenient_answers
    assert request.budget(94208) == Budget(
        max_turns=1, attempt_tokens=94208 + 6144, turn_tokens=94208, min_turn_tokens=256,
        context_tokens=131072, temperature=1.0, reasoning_effort="xhigh", cell_timeout_s=1.0,
        max_cells=4, feedback_chars=6000, preserve_thinking=True, forcing_tokens=6144,
        forcing_kind="answer2",
    )  # fmt: skip


def test_the_submission_4_candidate_differs_only_in_the_listed_keys():
    """Run P6.9-01's rows (docs/experiments/P6.9-01.md): the bucket order without the middle
    bucket, min_cap 65,536, the refill queue, the separate forcing turn (row 5' Revert, so no end
    note and Submission 3's server flags), and the lenient parser."""
    new = flatten(yaml.safe_load(S4.read_text()))
    old = flatten(yaml.safe_load(S3.read_text()))
    added = {"selection.order", "selection.mid_bucket", "waves.refill",
             "request.forcing_in_request", "request.lenient_answers"}  # fmt: skip
    assert set(new) - set(old) == added and set(old) <= set(new)
    assert {k for k in old if new[k] != old[k]} == {"waves.min_cap"}
    config = shot.load_config(S4)
    assert (config.order, config.mid_bucket, config.waves.min_cap) == ("buckets", False, 65536)
    assert config.waves.refill and not config.waves.forcing_in_request
    assert not config.request.forcing_in_request and config.request.lenient_answers
    assert config.request.thinking_end_note == "" and config.server.extra_args == ()
    assert config.request.budget(94208).attempt_tokens == 94208 + 6144
    assert shot.without_in_request(config) == config


def in_request_mapping():
    """The Submission 4 candidate with run P6.9-01's in-request forcing on (its note and flag)."""
    raw = yaml.safe_load(S4.read_text())
    wave = yaml.safe_load((REPO / "configs" / "P6.9-01.yaml").read_text())
    raw["request"].update(forcing_in_request=True, thinking_end_note=NOTE)
    raw["server"]["extra_args"] = wave["server"]["extra_args"]
    return raw


def test_the_in_request_mapping_is_run_p6_9_01s():
    wave = yaml.safe_load((REPO / "configs" / "P6.9-01.yaml").read_text())
    assert wave["direct"]["thinking_end_note"] == NOTE
    config = shot.config_from_mapping(in_request_mapping())
    assert config.waves.forcing_in_request and config.request.thinking_end_note == NOTE
    assert config.request.budget(94208).attempt_tokens == 94208 + 2 * 6144


def test_every_submission_4_config_value_has_a_comment():
    # test_shot's rule for configs/agent_shot.yaml: a comment above every value, sections aside.
    lines = S4.read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        assert lines[k - 1].strip().startswith("#") or stripped.endswith(":"), line


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["selection"].update(order="random"), "selection.order"),
        (lambda c: c["selection"].update(mid_bucket=True), "mid_bucket"),
        (lambda c: c["selection"].update(order="confidence", mid_bucket=True), "mid_bucket"),
        (lambda c: c["waves"].update(refill="yes"), "waves.refill"),
        (lambda c: c["request"].update(thinking_end_note="x</think>"), "think tags"),
        (lambda c: c["request"].update(thinking_end_note=""), "thinking_end_note"),
        (lambda c: c["server"].update(extra_args=[]), "reasoning-config"),
        (lambda c: c["server"].update(extra_args=["--reasoning-config", "{"]), "not JSON"),
        (lambda c: c["selection"].update(colour=1), "unknown keys"),
    ],
)  # fmt: skip
def test_the_new_keys_are_checked(edit, message):
    raw = in_request_mapping()
    edit(raw)
    with pytest.raises(ConfigError, match=message):
        shot.config_from_mapping(raw)


def test_the_middle_bucket_is_accepted_with_its_cap():
    raw = yaml.safe_load(S4.read_text())
    raw["selection"].update(mid_bucket=True, max_confidence=0.7)
    assert shot.config_from_mapping(raw).mid_bucket


# --- end to end: the refill queue with in-request forcing ----------------------------------------

# pytest's prepend import mode puts this folder on sys.path, so test_shot's helpers import.
from test_shot import TASKS, arguments, make_case, solved, stop_engine, stub, write_config  # noqa: E402,I001

from arc26.submit import validate  # noqa: E402


def p69_config(tmp_path):
    """The end-to-end config of test_shot with task P6.9's options on (the in-request forcing
    too, run P6.9-01's) and two streams."""
    path = write_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    s4 = in_request_mapping()
    config["selection"].update(order="buckets", mid_bucket=False)
    config["waves"].update(refill=True, wave_tasks=2, min_cap=256)
    config["request"].update(forcing_in_request=True, thinking_end_note=NOTE,
                             lenient_answers=True)  # fmt: skip
    config["server"]["extra_args"] = s4["server"]["extra_args"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run_p69(tmp_path, config):
    challenges, solutions, paths = make_case(tmp_path)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    try:
        code = shot.main(arguments(paths, out, config, logs, "--fake-engine",
                                   str(paths["replies"]), "--solutions",
                                   str(paths["solutions"])))  # fmt: skip
    finally:
        stop_engine(logs)
    plan = json.loads((logs / "shot_plan.json").read_text())
    return code, json.loads((logs / "run.json").read_text()), plan, logs


def test_a_failed_probe_turns_in_request_off_for_the_whole_run(tmp_path, monkeypatch):
    async def failing(*args, **kwargs):
        return {"ok": False, "forced": False, "error": "HTTP 400: thinking_token_budget is set"}

    monkeypatch.setattr(shot, "budget_probe", failing)
    code, summary, plan, logs = run_p69(tmp_path, p69_config(tmp_path))
    assert code == 0 and summary["shot"]["in_request"] is False
    assert any("in-request forcing off" in p for p in summary["problems"])
    assert plan["model"]["forcing_in_request"] is False
    calls = [json.loads(line) for line in (logs / "engine_calls.jsonl").open()]
    assert not any("thinking_token_budget" in c for c in calls)
    rows = [json.loads(line) for line in (logs / "shot_attempts.jsonl").open()]
    cut = next(r for r in rows if r["task_id"] == "aaaa0005")
    assert cut["forcing_turns"] == 1 and cut["finish"] == ["length", "stop"]


def test_a_refill_run_with_in_request_forcing_merges_and_scores(tmp_path):
    challenges, solutions, paths = make_case(tmp_path)
    # aaaa0005 thinks past the cap: in-request, its answer follows the forced end in one reply,
    # and its rows are digit strings, which only the lenient parser reads.
    replies = json.loads(paths["replies"].read_text())
    right = solutions["aaaa0005"][0]
    rows = ", ".join(json.dumps("".join(map(str, r))) for r in right)
    text = f'```answer\n{{"test_index": 0, "grid": [{rows}]}}\n```\n'
    replies["aaaa0005"] = [{"text": text, "reasoning": "the rows flip over and over " * 2000}]
    paths["replies"].write_text(json.dumps(replies), encoding="utf-8")
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    try:
        code = shot.main(arguments(paths, out, p69_config(tmp_path), logs, "--fake-engine",
                                   str(paths["replies"]), "--solutions",
                                   str(paths["solutions"])))  # fmt: skip
    finally:
        stop_engine(logs)
    written = json.loads(out.read_text())
    summary = json.loads((logs / "run.json").read_text())
    plan = json.loads((logs / "shot_plan.json").read_text())
    assert code == 0 and summary["problems"] == [], summary["problems"]
    assert validate(written, stub(challenges)) == []
    # Every queued task is one start, at most two streams at a time.
    assert [w["tasks"] for w in plan["waves"]] == [[t] for t, _ in plan["queue"]]
    assert plan["order"] == "buckets" and plan["stopped"] == "no tasks left"
    assert solved(written, solutions) == sorted(TASKS)
    rows = [json.loads(line) for line in (logs / "shot_attempts.jsonl").open()]
    forced = next(r for r in rows if r["task_id"] == "aaaa0005")
    assert forced["finish"] == ["stop"] and forced["budget_forced"] == 1
    assert forced["forcing_turns"] == 0 and forced["answer_lenient"] == [True]
    calls = [json.loads(line) for line in (logs / "engine_calls.jsonl").open()]
    budgets = {c.get("thinking_token_budget") for c in calls
               if c["attempt_id"] not in ("sanity/0", "budget-check/0")}  # fmt: skip
    assert budgets == {4096}
    assert plan["budget_probe"]["ok"] and summary["shot"]["in_request"] is True
    metrics = json.loads((logs / "shot_metrics.json").read_text())
    assert metrics["hybrid"]["score"] == 5 and metrics["lost"] == []


def test_a_refill_run_with_submission_4s_request_forces_in_a_separate_turn(tmp_path):
    """Submission 4's request (configs/agent_shot_P6.9.yaml: run P6.9-01's row 5' Revert): no
    budget probe and no thinking budget in any request; the turn cut at the cap gets the separate
    forcing turn, whose digit-string rows only the lenient parser reads; every task one refill
    start in the bucket order; the merge scores every task."""
    challenges, solutions, paths = make_case(tmp_path)
    replies = json.loads(paths["replies"].read_text())
    right = solutions["aaaa0005"][0]
    rows = ", ".join(json.dumps("".join(map(str, r))) for r in right)
    forced_text = f'```answer\n{{"test_index": 0, "grid": [{rows}]}}\n```\n'
    replies["aaaa0005"] = [
        {"text": "", "reasoning": "the rows flip over and over " * 2000},
        {"text": forced_text},
    ]
    paths["replies"].write_text(json.dumps(replies), encoding="utf-8")
    path = p69_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    s4 = yaml.safe_load(S4.read_text())
    config["request"]["forcing_in_request"] = s4["request"]["forcing_in_request"]
    config["request"].pop("thinking_end_note")
    config["server"]["extra_args"] = s4["server"]["extra_args"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    try:
        code = shot.main(arguments(paths, out, path, logs, "--fake-engine",
                                   str(paths["replies"]), "--solutions",
                                   str(paths["solutions"])))  # fmt: skip
    finally:
        stop_engine(logs)
    written = json.loads(out.read_text())
    summary = json.loads((logs / "run.json").read_text())
    plan = json.loads((logs / "shot_plan.json").read_text())
    assert code == 0 and summary["problems"] == [], summary["problems"]
    assert validate(written, stub(challenges)) == []
    assert [w["tasks"] for w in plan["waves"]] == [[t] for t, _ in plan["queue"]]
    assert plan["order"] == "buckets" and plan["stopped"] == "no tasks left"
    assert "budget_probe" not in plan and not plan["model"]["forcing_in_request"]
    assert solved(written, solutions) == sorted(TASKS)
    attempts = [json.loads(line) for line in (logs / "shot_attempts.jsonl").open()]
    cut = next(r for r in attempts if r["task_id"] == "aaaa0005")
    assert cut["finish"] == ["length", "stop"] and cut["forcing_turns"] == 1
    assert "budget_forced" not in cut and cut["answer_lenient"] == [True]
    calls = [json.loads(line) for line in (logs / "engine_calls.jsonl").open()]
    assert not any("thinking_token_budget" in c for c in calls)
    metrics = json.loads((logs / "shot_metrics.json").read_text())
    assert metrics["hybrid"]["score"] == 5 and metrics["lost"] == []
