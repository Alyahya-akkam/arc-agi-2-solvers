"""Checks run P6.3-05, the long shot: its task selection, config, single-shot report, and health.

The smoke runs the shot on the fixtures through P6.2's loop with the two-answer direct prompt
and the scripted engine. No GPU, no network beyond localhost.
"""

import asyncio
import csv
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from arc26.agent import feedback, prompts
from arc26.agent.loop import Budget, run_attempt
from arc26.agent.mock_engine import ScriptedEngine, ScriptedReply, answer_reply, overlong_reply
from arc26.agent.types import AgentTask, AttemptRecord, Pair
from arc26.data import DEV40

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.3-05.yaml"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike_shot")


# --- the task selection ----------------------------------------------------------------------


def nvarc_confidence_and_solves():
    # P2.8's per-candidate rows of P2.6-01: the top selected grid's softmax share per output (0
    # without candidates), the minimum over a task's outputs; a task's solved share.
    top, outputs, solved = {}, {}, defaultdict(float)
    for row in csv.DictReader((REPO / "reports" / "P2.8" / "P2.6-01.csv").open()):
        task = row["task_id"]
        if task not in DEV40:
            continue
        outputs[task] = round(1 / float(row["weight"]))
        if row["rank_kgmon"] == "1":
            top[(task, int(row["output"]))] = float(row["softmax_p"])
        if row["attempt"] in ("1", "2") and row["correct"] == "1":
            solved[task] += float(row["weight"])
    confidence = {
        t: min((top.get((t, o), 0.0) for o in range(outputs[t])), default=0.0) if t in outputs
        else 0.0
        for t in DEV40
    }  # fmt: skip
    return confidence, solved


def test_the_shot_takes_the_15_least_confident_dev40_tasks_with_ties_by_id():
    confidence, solved = nvarc_confidence_and_solves()
    chosen = sorted(DEV40, key=lambda t: (confidence[t], t))[:15]
    config = spike.load_config(CONFIG)
    assert config["direct"]["task_ids"] == chosen
    assert sum(1 for t in chosen if confidence[t] == 0.0) == 10
    # NVARC solved none of the 15; the reading's NVARC list says so.
    assert [t for t in chosen if solved[t] >= 1] == [] == config["reading"]["nvarc_solved"]
    assert set(chosen) <= set(config["agent"]["nvarc_misses"])


# --- the two-answer prompt and forcing note ---------------------------------------------------


def test_the_two_answer_prompt_and_note_ask_for_up_to_two_answers_best_first():
    text = prompts.DIRECT_TWO_SYSTEM_PROMPT
    assert "one or two answer blocks per test input, your best answer first" in text
    assert "transform(" not in text and text != prompts.DIRECT_SYSTEM_PROMPT
    note = feedback.forcing_message([], limit=100, kind="answer2")
    assert "one or two ```answer blocks per test input, your best answer first" in note
    assert Budget(forcing_tokens=300, forcing_kind="answer2").forcing_reserve() == (
        300 + feedback.forcing_chars("answer2") + 112)  # fmt: skip


def test_a_cut_shot_gets_the_two_answer_forcing_turn_and_records_both_answers():
    task = AgentTask("t", (Pair([[1, 2]], [[2, 1]]),), ([[3, 4]],))
    both = answer_reply(0, [[3, 4]]).text + answer_reply(0, [[4, 3]]).text
    engine = ScriptedEngine({"t": [overlong_reply(8000), ScriptedReply(text=both)]})
    budget = Budget(
        max_turns=1, attempt_tokens=2000, turn_tokens=1000, forcing_tokens=500,
        context_tokens=60_000, preserve_thinking=True, forcing_kind="answer2",
    )  # fmt: skip
    record = asyncio.run(run_attempt(task, engine, spike.NoRepl(), spike.NoVerifier(), budget,
                                     system=prompts.DIRECT_TWO_SYSTEM_PROMPT))  # fmt: skip
    assert engine.calls[1].thinking is False
    assert engine.calls[1].messages[-1].content == feedback.forcing_message(
        [], limit=1000, kind="answer2")  # fmt: skip
    assert record.direct_answers == ((0, [[3, 4]]), (0, [[4, 3]]))


# --- the single-shot report and its reading ---------------------------------------------------


def record(task_id, answers, stop="rounds", forcing=0, tokens=1000):
    return AttemptRecord(task_id, f"{task_id}/p0/0", 0, (), tuple(answers), 10, tokens, 1, 1.0,
                         stop, forcing, 0)  # fmt: skip


def task(task_id, tests=1):
    return AgentTask(task_id, (Pair([[1]], [[1]]),), tuple([[0]] for _ in range(tests)))


READING = {"min_solves": 3, "margin_over_nvarc": 2, "nvarc_solved": ["e"], "max_cut_share": 0.10}
A, B, C = [[1, 2]], [[3, 4]], [[5, 6]]


def shot(records, tasks=None, health=None, reading=READING):
    tasks = tasks or [task(t) for t in "abcde"]
    solutions = {t.task_id: [A] * len(t.test_inputs) for t in tasks}
    return spike.shot_metrics(tasks, records, solutions, "T", 3600.0,
                              health or {"healthy": True}, reading)  # fmt: skip


def test_the_shot_scores_two_answers_per_output_and_reads_against_nvarc():
    records = [
        record("a", [(0, A)]),
        record("b", [(0, B), (0, A)]),  # right on its second attempt
        record("c", [(0, B), (0, B), (0, A)]),  # a repeat is no second attempt: A is the second
        record("d", [(0, B), (0, C), (0, A)]),  # A only third: not an attempt
        record("e", []),
    ]  # fmt: skip
    metrics = shot(records, reading={**READING, "nvarc_solved": ["d", "e"]})
    assert metrics["solved"] == ["a", "b", "c"] and metrics["score"] == 3.0
    assert metrics["solved_by_second_attempt_only"] == ["b", "c"]
    # 3 solves, NVARC 2: at least 3, but not 2 more than NVARC.
    assert metrics["nvarc_solves"] == 2 and metrics["reading"] == "does not enter"
    # NVARC 1: 3 solves are 2 more, and at least 3: enters.
    assert shot(records)["reading"] == "enters"


def test_the_shot_reads_fault_on_health_and_fault_rate_on_cut_tasks():
    records = [record(t, [(0, A)]) for t in "abcd"]
    assert shot(records, health={"healthy": False})["reading"] == "Fault"
    # "e" never started and "d" stopped at the deadline before a forcing turn: 2 of 5 cut.
    cut = records[:3] + [record("d", [], stop="deadline")]
    metrics = shot(cut, reading={**READING, "nvarc_solved": []})
    assert metrics["forcing_cut"] == {"attempts": 5, "cut": 2, "share": 0.4}
    assert metrics["reading"] == "Fault (rate)"


# --- health: an empty forced reply is reported, not degenerate ------------------------------


DEGENERATE = {"max_run": 64, "special_min_id": 248044, "eos_ids": [248046, 248044]}


def test_an_empty_reply_is_reported_but_real_degeneracy_stays_a_fault(tmp_path):
    empty = spike.call_health([248046], "", "", 4096, DEGENERATE)
    assert empty["empty"] and not empty["degenerate"]
    assert spike.call_health([0] * 80, "!" * 80, "", 4096, DEGENERATE)["degenerate"]
    assert spike.call_health([7] * 70 + [248046], "x" * 70, "", 4096, DEGENERATE)["degenerate"]
    path = tmp_path / "calls.jsonl"
    good = '{"error": null, "location": "reasoning", "degenerate": false, "empty": false}\n'
    blank = '{"error": null, "location": "none", "degenerate": false, "empty": true}\n'
    path.write_text(good * 10 + blank)
    summary = spike.engine_call_summary(path)
    assert (summary["empty"], summary["degenerate"], summary["healthy"]) == (1, 0, True)
    bad = '{"error": null, "location": "reasoning", "degenerate": true, "empty": false}\n'
    path.write_text(good * 10 + bad)
    assert spike.engine_call_summary(path)["healthy"] is False


# --- the smoke and the config ------------------------------------------------------------------


@pytest.fixture(scope="module")
def shot_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("shot")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--smoke", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=600, cwd=REPO,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out


def test_the_smoke_runs_the_forcing_check_then_the_shot(shot_run):
    run = json.loads((shot_run / "run.json").read_text())
    assert run["phases"] == ["forcing", "direct"] and run["exit_code"] == 0
    assert json.loads((shot_run / "forcing_check.json").read_text())["verdict"] == "pass"
    assert json.loads((shot_run / "harness_preflight.json").read_text())["ok"] is True
    metrics = json.loads((shot_run / "shot_metrics.json").read_text())
    # mirror: cut, then the two-answer forcing turn; transpose: right on its second answer;
    # recolor: prose. One sample per task, no view.
    assert (metrics["tasks"], metrics["score"], metrics["solves"]) == (3, 2.0, 2)
    assert metrics["solved_by_second_attempt_only"] == ["smoke-transpose"]
    assert metrics["forcing_turns"] == 1 and metrics["reading"] == "does not enter"
    attempts = [json.loads(line) for line in (shot_run / "direct_attempts.jsonl").read_text()
                .splitlines()]  # fmt: skip
    assert sorted(a["task_id"] for a in attempts) == ["smoke-mirror", "smoke-recolor",
                                                      "smoke-transpose"]  # fmt: skip
    transcript = (shot_run / "direct_transcripts.jsonl").read_text()
    assert "one or two answer blocks per test input" in transcript
    assert "single shot: score 2.0 of 3" in (shot_run / "summary.md").read_text()


def test_the_shot_fits_the_context_the_kv_pool_and_its_budget():
    config = spike.load_config(CONFIG)
    direct, server = config["direct"], config["server"]
    assert server["max_model_len"] == direct["context_tokens"] == 131072 <= 262144
    reserve = direct["forcing_tokens"] + feedback.forcing_chars("answer2") + 112
    assert 5924 + 64 + direct["max_tokens"] + reserve <= direct["context_tokens"]
    # The 15 openings (P6.3-04's counts) and 15 turns with forced replies of about 2,000 fit the
    # pool measured at 65,536; with every forced reply at its cap they still do.
    assert 44666 + 15 * (direct["max_tokens"] + 2000) <= 0.94 * 1593344
    assert 44666 + 15 * (direct["max_tokens"] + direct["forcing_tokens"]) <= 1593344
    assert direct["concurrency"] == len(direct["task_ids"]) == 15
    assert (direct["reasoning_effort"], direct["with_view"], direct["answers_per_test"]) == (
        "xhigh", False, 2)  # fmt: skip
    # The forcing turns are reached down to 139 tokens per second.
    assert 15 * direct["token_budget"] / direct["budget_s"] <= 140
    stages = 600 + 300 + 900 + config["forcing"]["budget_s"] + direct["budget_s"]
    assert stages == 12900 and stages + 60 <= config["max_wall_s"] == 13200


@pytest.mark.parametrize("run_id", ["P6.3-05", "P6.3-06"])
def test_the_shot_builds_reuse_the_route_dataset_with_their_own_kernels(run_id):
    # P6.3-06 (the confirmation) is the same lineage: the shot's kernel at its version 2.
    builder = load(REPO / "scripts" / "build_kaggle_notebook.py", "_test_nb_builder_shot")
    parsed = {}
    for name in (f"{run_id}-script", f"{run_id}-cpu-script"):
        path = REPO / "configs" / f"{name}.yaml"
        parsed[name] = builder.parse_config(path.read_bytes(), path)
    gpu, cpu = parsed[f"{run_id}-script"], parsed[f"{run_id}-cpu-script"]
    assert gpu.kaggle_notebook == "alyahya737/arc26-p6-3-shot" and gpu.accelerator == "NvidiaL4"
    assert cpu.kaggle_notebook == "alyahya737/arc26-p6-3-shot-cpu" and cpu.accelerator is None
    assert gpu.code_dataset == cpu.code_dataset == "alyahya737/arc26-code-p6-route"
    assert gpu.script_config == cpu.script_config == f"configs/{run_id}.yaml"
    config = spike.load_config(REPO / "configs" / f"{run_id}.yaml")
    assert 188.9 + config["max_wall_s"] + 300 <= gpu.script_kill_s == 13700


@pytest.mark.parametrize("run_id", ["P6.3-05", "P6.3-06"])
def test_every_shot_config_value_has_a_comment(run_id):
    for name in (run_id, f"{run_id}-script", f"{run_id}-cpu-script"):
        path = REPO / "configs" / f"{name}.yaml"
        previous = ""
        for line in path.read_text().splitlines():
            if re.match(r"\s*[A-Za-z_0-9]+:", line):
                assert "#" in line or previous.lstrip().startswith("#"), f"{path.name}: {line}"
            if line.strip():
                previous = line


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["direct"].update(answers_per_test=3), "answers_per_test"),
        (lambda c: c["direct"].update(task_ids=["0934a4d8", "0934a4d8"]), "task_ids"),
        (lambda c: c["reading"].update(nvarc_solved=["0934a4d8"]), "among direct.task_ids"),
        (lambda c: c["reading"].pop("min_solves"), "exactly the keys"),
        (lambda c: c["direct"].update(with_view="no"), "with_view"),
    ],
)  # fmt: skip
def test_shot_config_rules_refuse_bad_values(edit, message):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    with pytest.raises(spike.ConfigError, match=message):
        spike.check_config(raw)
