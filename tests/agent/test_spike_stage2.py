"""Checks stage 2 of the agent spike (run P6.3-02): the merged harness end to end, and its config.

The smoke path is the integration test of P6.1 and P6.2: P6.2's ``ScriptedEngine``, agent loop,
runner, and selection drive P6.1's real ``SubprocessRepl`` and ``PoolVerifier`` on the fixture
tasks (and DEV4 when its files are given); the forcing check (revision 5) and the preserve check
run P6.2's loop and ``OpenAIEngine`` against the script's fake server. No GPU, no network beyond
localhost.
"""

import asyncio
import csv
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from arc26.agent.types import (
    AgentTask,
    AttemptRecord,
    GuardFinding,
    Pair,
    PairResult,
    Verification,
)
from arc26.data import DEV4, DEV40

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.3-02.yaml"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike_stage2")


def smoke(out, *extra, timeout=600, config=CONFIG):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--smoke", "--out-dir", str(out),
         *extra],
        capture_output=True, text=True, timeout=timeout, cwd=REPO,
    )  # fmt: skip


# Revision 4's preserve check, which revision 5's config no longer runs: the script keeps it.
PRESERVE = {"n_tasks": 3, "turn1_max_tokens": 4096, "reasoning_effort": "low",
            "tolerance_tokens": 16, "budget_s": 600}  # fmt: skip


def preserve_config(folder):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["phases"] = ["preserve", "agent"]
    raw.pop("forcing")
    raw["preserve"] = dict(PRESERVE)
    path = folder / "preserve.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def stage2(tmp_path_factory):
    out = tmp_path_factory.mktemp("stage2")
    result = smoke(out)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out


# --- the smoke path: P6.1 and P6.2 together ------------------------------------------------------


def test_cascade_runs_the_real_repl_and_verifier_and_scores_the_selection(stage2):
    metrics = json.loads((stage2 / "agent_metrics.json").read_text())
    assert (metrics["subset"], metrics["tasks"], metrics["score"]) == ("fixtures", 3, 3.0)
    assert metrics["verified"] == 2 and metrics["precision_tasks"] == 1.0
    assert (metrics["verified_programs"], metrics["right_programs"]) == (2, 2)
    assert [p["attempts"] for p in metrics["phases"]] == [3, 1]
    assert metrics["phases"][0]["stop_reasons"] == {"rounds": 1, "verified": 2}
    rows = metrics["per_task"]
    # smoke-mirror's first reply is cut at its cap; its forcing turn writes the program.
    assert rows["smoke-mirror"]["turns"] == 2 and rows["smoke-transpose"]["turns"] == 2
    mirror = rows["smoke-mirror"]
    assert (mirror["forcing_turns"], mirror["forced_programs"]) == (1, 1)
    assert rows["smoke-transpose"]["forcing_turns"] == 0
    assert (metrics["forcing_turns"], metrics["forced_programs"]) == (1, 1)
    assert [p["forcing_turns"] for p in metrics["phases"]] == [1, 0]
    # The copied-outputs program: P6.1's guards reject it in both phases; the direct answer scores.
    recolor = rows["smoke-recolor"]
    assert recolor["guard_rejections"] == 2 and recolor["direct_answers"] == 2
    assert recolor["verified"] is False and recolor["score"] == 1.0
    assert metrics["verifier_workers_replaced"] == 0 and metrics["sandbox_failures"] == 0


def test_m_counts_only_misses_solved_by_verified_programs(stage2):
    # The smoke counts every fixture as an NVARC miss: recolor's direct answer scores, but not
    # toward M.
    metrics = json.loads((stage2 / "agent_metrics.json").read_text())
    assert metrics["nvarc_misses_solved_by_verified_programs"] == [
        "smoke-mirror",
        "smoke-transpose",
    ]
    assert metrics["nvarc_misses_scored_otherwise"] == ["smoke-recolor"]
    assert metrics["nvarc_misses_partial"] == []
    assert metrics["per_task"]["smoke-recolor"]["solved_by"] == "other answer"
    assert metrics["gate"] == {
        "M": 2, "of": 3, "precision": 1.0, "healthy": True, "harness": True, "row": "Bad",
    }  # fmt: skip


def test_cascade_writes_p62_records_transcripts_and_a_valid_submission(stage2):
    records = jsonl(stage2 / "agent_attempts.jsonl")
    assert sorted(r["attempt_id"] for r in records) == [
        "smoke-mirror/p1/0", "smoke-recolor/p1/0", "smoke-recolor/p2/0", "smoke-transpose/p1/0",
    ]  # fmt: skip
    rejected = next(r for r in records if r["attempt_id"] == "smoke-recolor/p1/0")
    rules = {g["rule"] for v in rejected["verifications"] for g in v["guards"] if g["fatal"]}
    assert "literal_output" in rules
    transcripts = jsonl(stage2 / "agent_transcripts.jsonl")
    assert len(transcripts) == 4 and transcripts[0]["messages"][0]["role"] == "system"
    submission = json.loads((stage2 / "agent_submission.json").read_text())
    assert set(submission) == {"smoke-mirror", "smoke-recolor", "smoke-transpose"}
    for rows in submission.values():
        assert all(set(row) == {"attempt_1", "attempt_2"} for row in rows)


def test_forcing_turns_reach_the_engine_with_thinking_off_and_the_reasoning_kept(stage2):
    lines = jsonl(stage2 / "agent_transcripts.jsonl")
    transcripts = {t["attempt_id"]: t["messages"] for t in lines}
    mirror = transcripts["smoke-mirror/p1/0"]
    assert [m["role"] for m in mirror] == ["system", "user", "assistant", "user", "assistant"]
    assert mirror[2]["content"] == "" and len(mirror[2]["reasoning"]) == 4 * 1024
    assert mirror[3].get("forcing") and "Stop reasoning now" in mirror[3]["content"]
    assert mirror[4].get("forcing") and "```python" in mirror[4]["content"]


def test_forcing_check_passes_through_p62s_loop_and_client(stage2):
    check = json.loads((stage2 / "forcing_check.json").read_text())
    assert check["verdict"] == "pass" and check["reasons"] == []
    assert (check["programs"], check["min_programs"], len(check["tasks"])) == (3, 2, 3)
    for row in check["tasks"]:
        assert row["verdict"] == "pass" and row["turn1"]["finish_reason"] == "length"
        forcing = row["forcing"]
        assert forcing["thinking"] is False and forcing["program"] and forcing["max_tokens"] == 256
        growth = row["growth"]
        assert growth["kept"] and growth["reasoning_tokens"] > 2 * 16
        # The fake's tags: 4 around the reasoning, 4 per message.
        assert growth["prompt_growth"] == growth["reasoning_tokens"] + growth["other_tokens"] + 12
        assert row["render"]["status"] == "ok" and row["render"]["prefix_kept"] is True
        # Revision 6: the prompt the call fed the model, from its own response's token ids.
        assert row["render"]["source"] == "prompt_token_ids"
    calls = jsonl(stage2 / "engine_calls.jsonl")
    assert len(calls) == 2 * 3 and all(c["attempt_id"].endswith("/forcing/0") for c in calls)
    forced = [c for c in calls if c["thinking"] is False]
    assert len(forced) == 3
    assert all(c["preserve_thinking"] is True and c["reasoning_sent"] == 1 for c in forced)
    assert all(c["thinking"] is None and c["finish_reason"] == "length"
               for c in calls if c not in forced)  # fmt: skip
    metrics = json.loads((stage2 / "metrics.json").read_text())
    assert metrics["engine_calls"] == {
        "calls": 6, "by_location": {"reasoning": 3, "none": 3}, "cancelled": 0, "errors": 0,
        "error_share": 0.0,
        "degenerate": 0, "empty": 0, "finish": {"length": 3, "stop": 3},
        "forcing_finish": {"stop": 3}, "healthy": True,
    }  # fmt: skip


@pytest.mark.parametrize(
    "mode, reason",
    [("drop", "a forcing prompt lacked the kept reasoning"),
     ("prose", "0 forced programs, fewer than 2")],
)  # fmt: skip
def test_a_failed_forcing_check_is_a_fault_and_no_agent_phase_runs(tmp_path, mode, reason):
    result = smoke(tmp_path, "--fake-mode", mode)
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-2000:]
    check = json.loads((tmp_path / "forcing_check.json").read_text())
    assert check["verdict"] == "fail" and reason in check["reasons"]
    metrics = json.loads((tmp_path / "agent_metrics.json").read_text())
    assert metrics["gate"]["row"] == "Fault" and metrics["stopped_by"] == "forcing check"
    assert not (tmp_path / "agent_attempts.jsonl").exists()
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["exit_code"] == 1 and "forcing check failed" in run["problems"][0]
    summary = (tmp_path / "summary.md").read_text()
    assert "forcing check: fail" in summary and "gate: Fault; the forcing check" in summary
    if mode == "drop":
        rows = check["tasks"]
        assert all(row["growth"]["kept"] is False for row in rows)
        assert all(row["render"]["status"].startswith("failed: no the reasoning's start")
                   for row in rows)  # fmt: skip


def test_preserve_check_passes_through_p62s_client_and_every_call_is_recorded(tmp_path):
    result = smoke(tmp_path / "out", config=preserve_config(tmp_path))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    out = tmp_path / "out"
    check = json.loads((out / "preserve_check.json").read_text())
    assert check["verdict"] == "pass" and len(check["tasks"]) == 2
    for row in check["tasks"]:
        assert row["on_keeps_reasoning"] and row["field_keeps_reasoning"]
        assert row["off_drops_reasoning"] and row["growth"]["sent_not_kept"] == 0
        assert row["growth"]["on"] == row["reasoning_tokens"] + 4  # the fake's think tags
    calls = jsonl(out / "engine_calls.jsonl")
    assert len(calls) == 2 * 5 and {c["location"] for c in calls} == {"reasoning"}
    on = [c for c in calls if c["attempt_id"].endswith("/preserve/on")]
    assert all(c["preserve_thinking"] is True and c["reasoning_sent"] == 1 for c in on)
    assert all(c["error"] is None and not c["degenerate"] and not c["empty"] for c in calls)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["engine_calls"] == {
        "calls": 10, "by_location": {"reasoning": 10}, "cancelled": 0, "errors": 0,
        "error_share": 0.0,
        "degenerate": 0, "empty": 0, "finish": {"stop": 1, "length": 9}, "forcing_finish": {},
        "healthy": True,
    }  # fmt: skip


def test_reasoning_inline_before_think_close_is_recorded_and_still_parsed(tmp_path):
    result = smoke(tmp_path / "out", "--fake-mode", "inline", config=preserve_config(tmp_path))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    calls = jsonl(tmp_path / "out" / "engine_calls.jsonl")
    assert {c["location"] for c in calls} == {"inline"}
    assert all(c["think_close_in_content"] for c in calls)
    # P6.2's client splits the reasoning off the content, so the check still measures it.
    assert json.loads((tmp_path / "out" / "preserve_check.json").read_text())["verdict"] == "pass"
    # The forcing check through the same server: the cut reasoning still reaches the prompt.
    result = smoke(tmp_path / "forcing", "--fake-mode", "inline")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    check = json.loads((tmp_path / "forcing" / "forcing_check.json").read_text())
    assert check["verdict"] == "pass"


def test_preserve_check_fails_when_the_server_drops_the_reasoning(tmp_path):
    result = smoke(tmp_path / "out", "--fake-mode", "drop", config=preserve_config(tmp_path))
    assert result.returncode == 0  # recorded, not a kill
    check = json.loads((tmp_path / "out" / "preserve_check.json").read_text())
    assert check["verdict"] == "fail"
    rows = check["tasks"]
    assert all(row["growth"]["on"] == 0 and not row["on_keeps_reasoning"] for row in rows)


def write_competition(root, ids):
    folder = root / "competitions" / "arc-prize-2026-arc-agi-2"
    folder.mkdir(parents=True)
    challenges, solutions = {}, {}
    for k, task_id in enumerate(ids):
        grid = [[(k + i + j) % 10 for j in range(3)] for i in range(2)]
        challenges[task_id] = {"train": [{"input": grid, "output": grid[::-1]}] * 2,
                               "test": [{"input": grid}]}  # fmt: skip
        solutions[task_id] = [grid[::-1]]
    (folder / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    (folder / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions))


def test_smoke_adds_dev4_when_the_competition_files_are_given(tmp_path):
    # Stand-in files with DEV4's ids (the repository holds no evaluation data).
    write_competition(tmp_path / "input", DEV4)
    result = smoke(tmp_path / "out", "--input-root", str(tmp_path / "input"))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    metrics = json.loads((tmp_path / "out" / "agent_metrics.json").read_text())
    assert metrics["subset"] == "fixtures+DEV4" and metrics["tasks"] == 7
    for task_id in DEV4:
        row = metrics["per_task"][task_id]
        assert row["attempts"] == 2 and row["guard_rejections"] == 2 and not row["verified"]


# --- pieces --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message, location, fields",
    [
        ({"content": "x", "reasoning": "r"}, "reasoning", ["reasoning"]),
        ({"content": "x", "reasoning_content": "r"}, "reasoning_content", ["reasoning_content"]),
        ({"content": "x", "reasoning": "r", "reasoning_content": "r"}, "reasoning",
         ["reasoning", "reasoning_content"]),
        ({"content": "<think>\nr\n</think>\n\nx"}, "inline", []),
        ({"content": "x", "reasoning": ""}, "none", []),
        ({"content": None}, "none", []),
    ],
)  # fmt: skip
def test_reasoning_location_classifies_each_case(message, location, fields):
    found = spike.reasoning_location(message)
    assert found["location"] == location and found["fields"] == fields


def pair(exact, error=None):
    return PairResult(0, exact, [[1]] if exact else None, error, exact, 1.0 if exact else 0.0)


def verification(test_outputs, verified=True, fatal=False, error=None):
    guards = (GuardFinding("literal_output", "x", True),) if fatal else ()
    return Verification(
        "def transform(g):\n    return g\n", "0" * 64, (pair(verified, error),),
        tuple(test_outputs), guards, 0.1,
    )  # fmt: skip


def attempt(task_id, phase, verifications, tokens=100, turns=2, direct=()):
    return AttemptRecord(
        task_id, f"{task_id}/p{phase}/0", phase, tuple(verifications), tuple(direct), 50, tokens,
        turns, 1.0, "rounds",
    )  # fmt: skip


def task(task_id, tests=1):
    return AgentTask(task_id, (Pair([[1]], [[1]]),), tuple([[0]] for _ in range(tests)))


def test_agent_metrics_score_with_arc26_evaluate_and_count_the_gate():
    from arc26.agent.select import build_selection, submission_from_selection

    tasks = [task("aaaaaaaa", tests=2), task("bbbbbbbb"), task("cccccccc"), task("dddddddd")]
    records = [
        # aaaaaaaa: a verified program right on test 0, wrong on test 1: partly solved.
        attempt("aaaaaaaa", 1, [verification([[[1]], [[7]]])], tokens=300),
        # bbbbbbbb: verified but wrong (a false positive), a guard rejection, and a timeout.
        attempt("bbbbbbbb", 1, [
            verification([[[4]]]),
            verification([[[0]]], verified=False, fatal=True, error="timeout"),
        ]),
        # cccccccc: no program, a right direct answer: scored, but not by a verified program.
        attempt("cccccccc", 2, [], tokens=500, turns=5, direct=[(0, [[3]])]),
        # dddddddd: a verified program right on its test output.
        attempt("dddddddd", 1, [verification([[[6]]])]),
    ]  # fmt: skip
    solutions = {
        "aaaaaaaa": [[[1]], [[2]]], "bbbbbbbb": [[[5]]], "cccccccc": [[[3]]], "dddddddd": [[[6]]],
    }  # fmt: skip
    counts = {"aaaaaaaa": 2, "bbbbbbbb": 1, "cccccccc": 1, "dddddddd": 1}
    selection = build_selection(counts, records)
    submission = submission_from_selection(selection)
    misses = ["aaaaaaaa", "cccccccc", "dddddddd"]
    metrics = spike.agent_metrics(
        tasks, records, selection, submission, solutions, misses, "T", {"healthy": True},
        {"healthy": True, "faults": []},
    )  # fmt: skip
    assert metrics["score"] == 2.5 and metrics["test_outputs"] == 5
    assert metrics["nvarc_misses_solved_by_verified_programs"] == ["dddddddd"]
    assert metrics["nvarc_misses_scored_otherwise"] == ["cccccccc"]
    assert metrics["nvarc_misses_partial"] == ["aaaaaaaa"]
    # Precision over verified programs: aaaaaaaa's and bbbbbbbb's are wrong, dddddddd's right.
    assert (metrics["verified_programs"], metrics["right_programs"]) == (3, 1)
    assert metrics["gate"] == {
        "M": 1, "of": 3, "precision": 0.333, "healthy": True, "harness": True, "row": "Bad",
    }  # fmt: skip
    # The same numbers with a harness fault: never Bad.
    broken = spike.agent_metrics(
        tasks, records, selection, submission, solutions, misses, "T", {"healthy": True},
        {"healthy": False, "faults": ["phase 1: none of 4 attempts finished"]},
    )  # fmt: skip
    assert (broken["gate"]["harness"], broken["gate"]["row"]) == (False, "Fault")
    assert metrics["guard_rejections"] == 1 and metrics["sandbox_failures"] == 1
    assert metrics["per_task"]["cccccccc"]["direct_answers"] == 1


@pytest.mark.parametrize(
    "solved, precision, healthy, row",
    [
        (7, 0.8, True, "Good"),
        (9, 0.95, False, "Good"),
        (7, 0.79, True, "Borderline"),
        (8, 0.6, True, "Borderline"),
        (6, 0.1, True, "Borderline"),
        (5, None, False, "Borderline"),
        (4, 0.9, True, "Bad"),
        (0, None, True, "Bad"),
        (3, 0.7, True, "Bad"),
        (4, 0.9, False, "Fault"),
        (0, None, False, "Fault"),
        (7, 0.59, True, "Fault"),
    ],
)  # fmt: skip
def test_gate_rows_follow_adr_020s_amendment(solved, precision, healthy, row):
    assert spike.gate_row(solved, precision, healthy) == row


DEGENERATE = {"max_run": 64, "special_min_id": 248044, "eos_ids": [248046, 248044]}


def test_call_health_flags_runs_and_empty_replies_but_not_one_token_probes():
    ok = spike.call_health([1, 2, 3], "answer", "thinking", 4096, DEGENERATE)
    assert ok["degenerate"] is False and ok["empty"] is False
    nan_like = spike.call_health([0] * 80, "!" * 80, "", 4096, DEGENERATE)
    assert nan_like["degenerate"] and nan_like["token0_count"] == 80
    after_eos = spike.call_health([5, 248046] + [7] * 100, "x", "", 4096, DEGENERATE)
    assert after_eos["degenerate"] is False
    # An empty reply is reported, not degenerate (run P6.3-05: P6.3-04's Fault came from one
    # empty forced reply, a model behavior, not a broken kernel).
    empty = spike.call_health([5], "  ", "", 4096, DEGENERATE)
    assert empty["empty"] and not empty["degenerate"]
    probe = spike.call_health([], "", "", 1, DEGENERATE)
    assert not probe["empty"] and not probe["degenerate"]


def test_engine_health_needs_few_errors_and_no_degenerate_output(tmp_path):
    path = tmp_path / "calls.jsonl"
    good = '{"error": null, "location": "reasoning", "degenerate": false, "empty": false}\n'
    path.write_text(good * 19 + '{"error": "EngineError: HTTP 500"}\n')
    summary = spike.engine_call_summary(path)
    assert (summary["errors"], summary["error_share"], summary["healthy"]) == (1, 0.05, True)
    path.write_text(good * 9 + '{"error": "EngineError: HTTP 500"}\n')
    assert spike.engine_call_summary(path)["healthy"] is False  # 10% of calls failed
    # Calls the run's deadline cancelled are not engine failures (revision 4: 5 at phase 2's end).
    cut = '{"error": "CancelledError: "}\n'
    path.write_text(good * 9 + cut * 5)
    summary = spike.engine_call_summary(path)
    assert (summary["cancelled"], summary["errors"], summary["healthy"]) == (5, 0, True)
    path.write_text(good * 9 + cut * 5 + '{"error": "EngineError: HTTP 500"}\n')
    assert spike.engine_call_summary(path)["error_share"] == 0.1  # 1 of 10 answered or failed
    bad = '{"error": null, "location": "reasoning", "degenerate": true, "empty": true}\n'
    path.write_text(good * 50 + bad)
    summary = spike.engine_call_summary(path)
    assert summary["healthy"] is False and (summary["degenerate"], summary["empty"]) == (1, 1)


class HangingEngine:
    async def complete(self, *args, **kwargs):
        await asyncio.sleep(3600)

    async def aclose(self):
        pass


def test_preserve_check_stops_at_its_deadline(monkeypatch, tmp_path):
    monkeypatch.setattr(spike, "make_engine", lambda *args, **kwargs: HangingEngine())
    settings = {"turn1_max_tokens": 64, "reasoning_effort": "low", "tolerance_tokens": 16}
    run = type("R", (), {"config": {"seed": 0}})()
    began = time.monotonic()
    row = asyncio.run(spike.preserve_task(run, task("aaaaaaaa"), settings, time.monotonic() + 0.3))
    assert time.monotonic() - began < 5
    assert row["verdict"] == "inconclusive: the check's deadline passed"


class StubRepl:
    def close(self):
        pass


def test_forcing_check_stops_at_its_deadline(monkeypatch):
    monkeypatch.setattr(spike, "make_engine", lambda *args, **kwargs: HangingEngine())
    monkeypatch.setattr(spike, "run_sandbox", lambda run, settings: (lambda t: StubRepl(), None))
    agent = {"forcing_tokens": 256, "context_tokens": 32768, "cell_timeout_s": 1.0}
    run = type("R", (), {"config": {"seed": 0, "agent": agent}})()
    settings = {"turn1_max_tokens": 64, "reasoning_effort": "low", "tolerance_tokens": 16}
    began = time.monotonic()
    row = asyncio.run(spike.forcing_task(run, task("aaaaaaaa"), settings, time.monotonic() + 0.3))
    assert time.monotonic() - began < 5
    assert row["stop_reason"] == "deadline"
    assert row["verdict"] == "inconclusive: no engine call answered"


class Kit:
    def __init__(self):
        self.engine, self.verifier, self.calls = object(), object(), []

    def repl_factory(self, task):
        self.calls.append(task.task_id)
        raise AssertionError("no attempt may start past the deadline")


def test_cascade_starts_nothing_past_the_run_deadline(tmp_path):
    run = type("R", (), {})()
    run.deadline = time.monotonic() - 1
    run.config = {"seed": 0, "server": {"max_model_len": 1000}}
    run.out = tmp_path
    settings = {"concurrency": 2, "cell_timeout_s": 1.0, "context_tokens": 1000, "phases": [
        {"max_tokens": 10, "token_budget": 20, "rounds": 1, "reasoning_effort": None,
         "temperature": None, "budget_s": 30, "preserve_thinking": False}]}  # fmt: skip
    kit = Kit()
    result = asyncio.run(spike.run_cascade([task("aaaaaaaa")], kit, settings, run))
    assert result["records"] == [] and result["phases"][0]["skipped"] == "deadline"
    assert kit.calls == []


def test_engine_call_summary_counts_locations(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text(
        '{"location": "reasoning"}\n{"location": "inline"}\n{"location": "reasoning"}\n'
    )
    summary = spike.engine_call_summary(path)
    assert summary["calls"] == 3 and summary["by_location"] == {"reasoning": 2, "inline": 1}
    assert spike.engine_call_summary(tmp_path / "absent.jsonl") == {}


# --- the config ----------------------------------------------------------------------------------


def test_every_config_value_has_a_comment():
    for path in (CONFIG, REPO / "configs" / "P6.3-02-script.yaml"):
        previous = ""
        for line in path.read_text().splitlines():
            if re.match(r"\s*[A-Za-z_0-9]+:", line):
                assert "#" in line or previous.lstrip().startswith("#"), f"{path.name}: {line}"
            if line.strip():
                previous = line


def nvarc_dev40_misses():
    # Recomputed from P2.6-01's per-candidate table, as the config's comment says.
    solved = defaultdict(float)
    for row in csv.DictReader((REPO / "reports" / "P2.8" / "P2.6-01.csv").open()):
        if row["attempt"] in ("1", "2") and row["correct"] == "1":
            solved[row["task_id"]] += float(row["weight"])
    return sorted(t for t in DEV40 if solved.get(t, 0.0) == 0.0)


# P6.3-01's attempt 3 (docs/experiments/P6.3-01.md): the stages stage 2 repeats, in seconds, and
# the rate its Borderline row plans with at 32 requests and contexts capped at 32k.
MEASURED = {"install": 189.6, "engine_prep": 37.4, "boot": 304.1, "notebook_start": 188.9}
PLAN_RATE = 300
KV_POOL = 1_593_344


def test_stage2_config_runs_the_forcing_check_then_dev40_with_stage_1s_engine():
    config = spike.load_config(CONFIG)
    first = spike.load_config(REPO / "configs" / "P6.3-01.yaml")
    agent = config["agent"]
    assert config["phases"] == ["forcing", "agent"]
    assert agent["subset"] == "DEV40" and "evaluation" in agent["challenges"]
    assert agent["nvarc_misses"] == nvarc_dev40_misses() and len(agent["nvarc_misses"]) == 29
    for section in ("model", "wheels", "sanity", "degenerate", "kill"):
        for key in set(config[section]) | set(first[section]):
            if not key.endswith("timeout_s"):
                assert config[section][key] == first[section][key], (section, key)
    server = dict(config["server"])
    # Revision 3: only the context limit and the boot's wait differ from the measured engine.
    for key in ("max_model_len", "boot_timeout_s"):
        server[key] = first["server"][key]
    assert server == first["server"]
    assert [p["reasoning_effort"] for p in agent["phases"]] == ["low", "xhigh"]
    # Revision 5: reasoning kept in both phases, a forcing turn after every cut turn.
    assert all(p["preserve_thinking"] for p in agent["phases"])
    assert agent["forcing_tokens"] == 4096
    forcing = config["forcing"]
    assert (forcing["n_tasks"], forcing["min_programs"]) == (3, 2)
    assert forcing["reasoning_effort"] == "low"
    assert agent["concurrency"] <= config["server"]["max_num_seqs"] and agent["concurrency"] == 32


def test_stage2_caps_contexts_at_32k_where_32_attempts_fit_the_kv_pool():
    config = spike.load_config(CONFIG)
    agent, server = config["agent"], config["server"]
    assert agent["context_tokens"] == server["max_model_len"] == 32768
    assert agent["concurrency"] * agent["context_tokens"] <= KV_POOL
    for phase in agent["phases"]:
        assert phase["max_tokens"] + agent["forcing_tokens"] < agent["context_tokens"]
        assert phase["max_tokens"] + agent["forcing_tokens"] <= phase["token_budget"]


# Revision 4's opening prompts (engine_calls.jsonl, each phase 1 attempt's first call): 2,494 to
# 7,291 tokens; their character bounds (agent_transcripts.jsonl): 7,598 to 12,380.
OPENING_TOKENS, OPENING_CHAR_BOUNDS = (2494, 7291), (7598, 12380)
# Phase 2's earlier programs with their reports: about 1,600 tokens each, two at most shown
# (the config's comment); the largest about 3,300 each.
PRIOR_TOKENS, PRIOR_TOKENS_MAX = 1600, 3300


def test_turn_1_leaves_the_forcing_reserve_on_every_dev40_prompt():
    # The opening is bounded by the server's count plus 64; the character bound is the fallback.
    from arc26.agent.loop import COUNT_SLACK, Budget

    agent = spike.load_config(CONFIG)["agent"]
    first, second = agent["phases"]
    for phase in (first, second):
        budget = Budget(
            turn_tokens=phase["max_tokens"], attempt_tokens=phase["token_budget"],
            context_tokens=agent["context_tokens"], forcing_tokens=agent["forcing_tokens"],
        )  # fmt: skip
        assert budget.forcing_reserve() == 6021 and COUNT_SLACK == 64

    def turn1(phase, opening):
        return min(phase["max_tokens"], phase["token_budget"] - agent["forcing_tokens"],
                   32768 - (opening + 64) - 6021)  # fmt: skip

    # Phase 1: 16,384 on every task (the context would allow 19,392 to 24,189).
    assert [turn1(first, t) for t in OPENING_TOKENS] == [16384, 16384]
    assert [32768 - (t + 64) - 6021 for t in OPENING_TOKENS] == [24189, 19392]
    # Phase 2: 19,392 to 24,189 without earlier programs; with two median ones about 16,200 to
    # 21,100; with two of the largest about 12,800 to 17,600. Never a zero turn.
    assert [turn1(second, t) for t in OPENING_TOKENS] == [24189, 19392]
    assert [turn1(second, t + 2 * PRIOR_TOKENS) for t in OPENING_TOKENS] == [20989, 16192]
    assert [turn1(second, t + 2 * PRIOR_TOKENS_MAX) for t in OPENING_TOKENS] == [17589, 12792]
    # The fallback (a failed count): the character bound, revision 4's arithmetic.
    caps = [min(second["max_tokens"], 32768 - b - 6021) for b in OPENING_CHAR_BOUNDS]
    assert caps == [19149, 14367]


def test_stage2_limits_fit_the_wall_budget_and_the_measured_stages():
    config = spike.load_config(CONFIG)
    agent = config["agent"]
    limits = {
        "install": config["wheels"]["timeout_s"],
        "engine_prep": config["wheels"]["engine_check_timeout_s"],
        "boot": config["server"]["boot_timeout_s"],
    }
    for stage, limit in limits.items():
        assert limit >= 2.5 * MEASURED[stage], stage
    total = sum(limits.values()) + config["forcing"]["budget_s"]
    total += sum(p["budget_s"] for p in agent["phases"])
    # Every stage at its limit, plus the inputs, preflight, sanity, and score (about 15 s): no cut.
    assert total == 9300 and total + 60 <= config["max_wall_s"]
    # The forcing check: turn 1 and a whole forced reply at one request's 44 tokens per second
    # (revision 4's preserve check) fit its budget with room to spare.
    forcing = config["forcing"]
    assert 2 * (forcing["turn1_max_tokens"] + agent["forcing_tokens"]) / 44 <= forcing["budget_s"]


def test_phase_1_reaches_every_dev40_task_at_the_planning_rate():
    agent = spike.load_config(CONFIG)["agent"]
    first, second = agent["phases"]
    assert (first["max_tokens"], first["token_budget"], first["rounds"]) == (16384, 24000, 3)
    assert (second["max_tokens"], second["token_budget"], second["rounds"]) == (24576, 49152, 4)
    # 40 attempts at their whole budget, 960,000 tokens, take 3,200 s at 300 tokens per second.
    assert len(DEV40) * first["token_budget"] / PLAN_RATE <= first["budget_s"] == 3600
    assert second["budget_s"] == 3600


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["agent"].update(context_tokens=65536), "context_tokens exceeds"),
        (lambda c: c["agent"].update(context_tokens=512), "agent.context_tokens must be"),
        (lambda c: c["agent"]["phases"][1].update(max_tokens=30000, token_budget=65536),
         "max_tokens and forcing_tokens reach agent.context_tokens"),
        (lambda c: c["agent"]["phases"][0].update(token_budget=20000),
         "max_tokens and forcing_tokens exceed token_budget"),
        (lambda c: c["agent"].update(forcing_tokens=100), "agent.forcing_tokens must be"),
        (lambda c: c["forcing"].update(turn1_max_tokens=30000), "turn 1 and the forcing reply"),
        (lambda c: c["forcing"].update(min_programs=4), "min_programs exceeds"),
        (lambda c: c["agent"].update(stall_s=10), "agent.stall_s must be"),
        (lambda c: c["agent"].pop("min_tokens_per_s"), "exactly the keys"),
        (lambda c: c.pop("forcing"), "exactly the keys"),
        (lambda c: c["agent"].pop("context_tokens"), "context_tokens"),
    ],
)  # fmt: skip
def test_context_cap_rules_refuse_bad_values(edit, message):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    with pytest.raises(spike.ConfigError, match=message):
        spike.check_config(raw)


@pytest.mark.parametrize("phase", ["preserve", "forcing"])
def test_preserve_and_forcing_need_the_agent_section(phase):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["phases"] = [phase]
    raw.pop("agent")
    with pytest.raises(spike.ConfigError, match="agent section"):
        spike.check_config(raw)


def test_stage2_script_config_has_its_own_notebook_and_dataset():
    builder = load(REPO / "scripts" / "build_kaggle_notebook.py", "_test_nb_builder_stage2")
    loaded = {}
    for run in ("01", "02"):
        path = REPO / "configs" / f"P6.3-{run}-script.yaml"
        loaded[run] = builder.parse_config(path.read_bytes(), path)
    first, second = loaded["01"], loaded["02"]
    assert second.script_config == "configs/P6.3-02.yaml"
    assert second.kaggle_notebook != first.kaggle_notebook
    assert second.code_dataset != first.code_dataset
    assert (second.notebook_inputs, second.args) == (first.notebook_inputs, first.args)
    config = spike.load_config(CONFIG)
    assert config["max_wall_s"] + 120 < second.script_kill_s <= 10800 - 300
    # The script starts at most 188.9 s into the notebook (P6.3-01, attempt 2) and stops itself
    # by max_wall_s; the server's stop (at most 60 s) and the summaries then fit before the kill.
    assert MEASURED["notebook_start"] + config["max_wall_s"] + 300 <= second.script_kill_s


# --- revision 5's fixes: a failed tokenize endpoint, a hung engine -----------------------------


def test_a_tokenize_endpoint_without_its_fields_never_fails_the_forcing_check(tmp_path):
    result = smoke(tmp_path, "--fake-mode", "badtokenize")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    check = json.loads((tmp_path / "forcing_check.json").read_text())
    assert check["verdict"] == "pass" and check["programs"] == 3
    for row in check["tasks"]:
        # The ids cannot be decoded, so the response's own prompt text is judged.
        assert (row["render"]["status"], row["render"]["source"]) == ("ok", "prompt_text")
        assert row["growth"]["counts"].startswith("generated count") and row["growth"]["kept"]
    assert json.loads((tmp_path / "agent_metrics.json").read_text())["gate"]["healthy"] is True


def phase_row(number=2, tokens=900_000, wall=3600.0, started=1000.0):
    return {"phase": number, "attempts": 37, "completion_tokens": tokens, "wall_s": wall,
            "started": started, "ended": started + wall}  # fmt: skip


def call(attempt, ended=None, error=None):
    row = {"attempt_id": attempt, "error": error}
    if ended is not None:
        row["ended"] = ended
    return row


def test_a_healthy_slow_phase_ending_at_its_deadline_stays_healthy():
    # 40 tokens per second (one request's rate), a call answered 100 s before the deadline, and
    # 5 calls cut there, as revision 4's phase 2 ended.
    calls = [call(f"t{k}/p2/0", ended=4500.0) for k in range(3)]
    calls += [call(f"c{k}/p2/0", error="CancelledError: ") for k in range(5)]
    assert spike.phase_stalls([phase_row(tokens=144_000)], calls, 10, 1200) == []


def test_a_hung_engine_is_a_stall_and_reads_fault_not_bad():
    # No read timeout: every call in flight ends cancelled at the deadline, no tokens at all.
    calls = [call(f"c{k}/p1/0", error="CancelledError: ") for k in range(32)]
    stalls = spike.phase_stalls([phase_row(number=1, tokens=0)], calls, 10, 1200)
    assert len(stalls) == 2 and "0.0 tokens per second" in stalls[0]
    assert "no call answered in its last 3600 s" in stalls[1]
    # The engine is then unhealthy, so M of 0 reads Fault, never Bad.
    assert spike.gate_row(0, None, False) == "Fault"


def test_a_hang_after_progress_is_a_stall():
    calls = [call("t/p2/0", ended=2000.0)] + [call(f"c{k}/p2/0", error="CancelledError: ")
                                              for k in range(32)]  # fmt: skip
    stalls = spike.phase_stalls([phase_row(tokens=500_000)], calls, 10, 1200)
    assert stalls == ["phase 2: no call answered in its last 2600 s (over 1200), with 32 calls "
                      "cut at its deadline"]  # fmt: skip


def test_a_phase_without_cancelled_calls_or_attempts_is_not_judged():
    calls = [call("t/p1/0", ended=1100.0)]
    rows = [phase_row(number=1, tokens=0), {"phase": 2, "attempts": 0, "skipped": "none left"}]
    assert spike.phase_stalls(rows, calls, 10, 1200) == []


# --- revision 6: the render check reads the prompt the model saw ------------------------------


def test_a_response_without_its_prompt_leaves_the_render_unavailable_and_the_check_passes(tmp_path):
    result = smoke(tmp_path, "--fake-mode", "noprompt")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    check = json.loads((tmp_path / "forcing_check.json").read_text())
    assert check["verdict"] == "pass" and check["programs"] == 3 and check["reasons"] == []
    for row in check["tasks"]:
        assert row["render"]["status"].startswith("unavailable: the response carried no")
        assert row["growth"]["kept"] is True  # judged from the real calls' prompt_tokens


def test_the_opening_counts_are_logged_and_checked_against_the_real_prompts(stage2):
    counts = jsonl(stage2 / "prompt_counts.jsonl")
    forcing = [r for r in counts if r["attempt_id"].endswith("/forcing/0")]
    assert len(forcing) == 3 and all(isinstance(r["count"], int) for r in forcing)
    check = json.loads((stage2 / "metrics.json").read_text())["prompt_counts"]
    assert check["attempts"] == 3 and check["failed"] == 0 and check["within_slack"] is True


class Run6:
    base_url = "http://127.0.0.1:9"
    config = {"server": {"served_model_name": "m"}}


REASONING = "Let me compare the grids. " * 20


def seen(reasoning=REASONING, closed=True):
    head = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nT<|im_end|>\n"
    turn1 = head + "<|im_start|>assistant\n<think>\n"
    body = reasoning.strip() + ("\n</think>\n\n" if closed else "")
    forced = (head + "<|im_start|>assistant\n<think>\n" + body + "<|im_end|>\n"
              "<|im_start|>user\nF<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return turn1, forced


def ids(text):
    return [ord(c) for c in text]


@pytest.fixture
def detokenize(monkeypatch):
    def fake(base_url, path, body, timeout_s=60):
        assert path == "/detokenize"
        return {"prompt": "".join(map(chr, body["tokens"]))}

    monkeypatch.setattr(spike, "http_post_json", fake)


def test_render_check_passes_on_the_served_prompt(detokenize):
    turn1, forced = seen()
    row = spike.render_check(Run6(), [{"ids": ids(turn1)}, {"ids": ids(forced)}], REASONING)
    assert (row["status"], row["source"], row["prefix_kept"]) == ("ok", "prompt_token_ids", True)


def test_render_check_fails_when_the_served_prompt_lacks_the_reasoning(detokenize):
    turn1, _ = seen()
    _, dropped = seen(reasoning="")
    row = spike.render_check(Run6(), [{"ids": ids(turn1)}, {"ids": ids(dropped)}], REASONING)
    assert row["status"].startswith("failed: no the reasoning's start")


def test_render_check_is_unavailable_without_a_prompt_in_the_responses(detokenize):
    row = spike.render_check(Run6(), [{"ids": None, "text": None}] * 2, REASONING)
    assert row["status"].startswith("unavailable") and row["prefix_kept"] is None


def test_render_check_falls_back_to_the_prompt_text_when_detokenize_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise spike.Stop("POST /detokenize: HTTP 500")

    monkeypatch.setattr(spike, "http_post_json", broken)
    turn1, forced = seen()
    prompts = [{"ids": ids(turn1), "text": turn1}, {"ids": ids(forced), "text": forced}]
    row = spike.render_check(Run6(), prompts, REASONING)
    assert (row["status"], row["source"], row["prefix_kept"]) == ("ok", "prompt_text", True)


def test_count_check_compares_each_opening_count_with_the_first_call():
    calls = [{"attempt_id": "a/p1/0", "prompt_tokens": 2494, "error": None},
             {"attempt_id": "a/p1/0", "prompt_tokens": 9000, "error": None},
             {"attempt_id": "b/p1/0", "prompt_tokens": 2600, "error": None}]  # fmt: skip
    counts = [{"attempt_id": "a/p1/0", "count": 2494, "error": None},
              {"attempt_id": "b/p1/0", "count": 2500, "error": None},
              {"attempt_id": "c/p1/0", "error": "EngineError: tokenize: HTTP 404"}]  # fmt: skip
    check = spike.count_check(counts, calls)
    assert check == {"attempts": 2, "failed": 1, "exact": 1, "max_prompt_minus_count": 100,
                     "min_prompt_minus_count": 0, "within_slack": False}  # fmt: skip
