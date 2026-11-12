"""Structured forcing validates complete answers and preserves the ordinary slot rules."""

import json

import pytest
from test_toolshot import INC, Step, answer, call, cfg, inc_task, norm, run

from arc26.agent import toolshot


def test_schema_request_and_two_attempts():
    payload = '{"answers":{"0":[["12","34"],["56","78"]],"1":[["90"]]}}'
    task = inc_task([[[1]], [[2]]])
    out, engine, tok = run(task, [Step("thinking", "length"), Step(payload + "<|im_end|>")],
                           config=cfg(structured_forcing=True))
    request = engine.requests[-1]
    schema = json.loads(request.extra["json_schema"])
    assert schema["properties"]["answers"]["required"] == ["0", "1"]
    assert schema["properties"]["answers"]["additionalProperties"] is False
    assert request.extra["stop"] == [] and request.extra["no_stop_trim"] is False
    prefix = tok.decode(request.input_ids)
    assert prefix.endswith("</think>\n\n")
    assert "row strings" in prefix and not prefix.endswith("```answer\n")
    assert norm(out.attempts) == [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 0]]]]
    assert payload in out.text and "```answer" not in out.text
    assert out.turns[-1]["schema_translation"] == "accepted"
    assert engine.requests[0].extra == {"stop": ["</tool_call>"], "no_stop_trim": True}


def test_schema_has_one_rectangular_branch_per_width():
    choices = toolshot._answer_schema(1)["properties"]["answers"]["properties"]["0"]
    branches = choices["items"]["anyOf"]
    assert len(branches) == 30
    for width, branch in enumerate(branches, 1):
        assert branch["items"]["pattern"] == f"^[0-9]{{{width}}}$"
        assert (branch["minItems"], branch["maxItems"]) == (1, 30)


@pytest.mark.parametrize("reply", [
    '{"answers":{"0":[["12"]]',
    '{"answers":{"0":[["1"]],"0":[["2"]]}}',
    '{"answers":{"0":[["12","3"]]}}',
    '{"answers":{"0":[["1a"]]}}',
    '{"answers":{"0":[["1234567890123456789012345678901"]]}}',
    '{"answers":{"0":[["1"]],"1":[["2"]]}}',
    '{"answers":{}}',
    '{"answers":{"0":[["1"]]}} trailing',
    '<tool_call>bad</tool_call>',
])
def test_invalid_object_preserves_natural_answer(reply):
    natural = answer((0, [[7]]))
    out, engine, _ = run(inc_task(), [Step(natural, "length"), Step(reply)],
                         config=cfg(structured_forcing=True))
    assert norm(out.attempts) == [[[[7]]]]
    assert out.turns[-1]["schema_translation"] == "rejected"
    assert len(engine.requests) == 2 and not out.calls


def test_complete_object_at_cut_and_four_input_limit():
    for finish in ("length", "cancelled"):
        assert "```answer" in toolshot._schema_answers(
            '{"answers":{"0":[["1"]]}}', 1, finish, ()
        )
    for finish in ("stop", "length", "cancelled", "error", "abort"):
        with pytest.raises(ValueError):
            toolshot._schema_answers('{"answers":{"0":[["1"]]', 1, finish, ())
    for finish in ("error", "abort"):
        with pytest.raises(ValueError):
            toolshot._schema_answers('{"answers":{"0":[["1"]]}}', 1, finish, ())
    schema = toolshot._answer_schema(4)
    assert schema["properties"]["answers"]["properties"]["0"]["maxItems"] == 1
    with pytest.raises(ValueError):
        toolshot._schema_answers(json.dumps({"answers": {str(i): [["1"], ["2"]]
                                                               for i in range(4)}}), 4, "stop", ())


@pytest.mark.parametrize("finish", ["length", "cancelled"])
def test_complete_cut_reply_is_used(finish):
    out, _, _ = run(inc_task(), [Step("thinking", "length"),
                                  Step('{"answers":{"0":[["1"]]}}', finish)],
                    config=cfg(structured_forcing=True))
    assert out.turns[-1]["schema_translation"] == "accepted"
    assert norm(out.attempts) == [[[[1]]]]


def test_forcing_respects_deadline_and_cap():
    out, engine, _ = run(inc_task(), [], force_at=0, end=0,
                         config=cfg(structured_forcing=True))
    assert not engine.requests and not out.forced
    out, engine, _ = run(inc_task(), [Step('{"answers":{"0":[["1"]]}}')], force_at=0,
                         config=cfg(structured_forcing=True, forcing_tokens=10))
    assert engine.requests[0].max_new_tokens == 10
    assert out.turns[0]["schema_translation"] == "rejected"


def test_invalid_force_keeps_verified_program_and_natural_priority(monkeypatch):
    monkeypatch.setattr(toolshot, "_verify", lambda *args, **kwargs: {"test_outputs": [[[8]]]})
    out, _, _ = run(inc_task(), [Step(answer((0, [[7]]), (0, [[9]])), "length"),
                                  Step('{"answers":')],
                    config=cfg(structured_forcing=True))
    assert norm(out.attempts) == [[[[7]], [[8]]]]


def test_valid_force_keeps_natural_and_verified_program_priority(monkeypatch):
    monkeypatch.setattr(toolshot, "_verify", lambda *args, **kwargs: {"test_outputs": [[[8]]]})
    out, _, _ = run(inc_task(), [Step(answer((0, [[7]])), "length"),
                                  Step('{"answers":{"0":[["6"]]}}')],
                    config=cfg(structured_forcing=True))
    assert norm(out.direct) == [[[[7]], [[6]]]]
    assert norm(out.attempts) == [[[[7]], [[8]]]]


def test_valid_force_fills_missing_inputs_deduplicates_and_keeps_two_natural():
    task = inc_task([[[0]], [[1]], [[2]]])
    natural = answer((0, [[1]]), (0, [[2]]), (1, [[3]]))
    forced = '{"answers":{"0":[["9"]],"1":[["3"],["4"]],"2":[["5"]]}}'
    out, _, _ = run(task, [Step(natural, "length"), Step(forced)],
                    config=cfg(structured_forcing=True))
    assert norm(out.direct) == [[[[1]], [[2]]], [[[3]], [[4]]], [[[5]]]]
    assert norm(out.attempts) == norm(out.direct)


def test_program_verified_before_forcing_can_consume_remaining_time(monkeypatch):
    events = []

    def verify(task, codes, defining, checked, config, verifier, errors, records, end, now):
        events.append((list(codes), now()))
        now.t += 1
        return {"test_outputs": [[[8]]]}

    monkeypatch.setattr(toolshot, "_verify", verify)
    out, engine, _ = run(
        inc_task(), [Step(call(INC)), Step("thinking", "length"),
                     Step('{"answers":{"0":[["1"]]}}', seconds=50)],
        end=20, config=cfg(structured_forcing=True),
    )
    assert len(events) == 1 and INC in events[0][0]
    assert engine.requests[-1].kind == "force" and engine.clock.t == 20
    assert out.turns[-1]["schema_translation"] == "rejected"
    assert norm(out.attempts) == [[[[8]]]]
