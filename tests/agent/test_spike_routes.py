"""Checks run P6.3-04: arm D (direct answers, the task and its view), arm L, and their readings.

The smoke path runs both arms on the fixtures: arm D through P6.2's loop with the direct-answer
prompt, ``NoRepl``, and the scripted engine; arm L through the program harness on the tasks
counted as misses. No GPU, no network beyond localhost.
"""

import asyncio
import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from arc26.agent import feedback, prompts
from arc26.agent.loop import Budget, run_attempt
from arc26.agent.mock_engine import ScriptedEngine, answer_reply, overlong_reply
from arc26.agent.types import AgentTask, AttemptRecord, Pair
from arc26.agent.views import View, fixed_view

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.3-04.yaml"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike_routes")


# --- the view and its inverse ------------------------------------------------------------------


@pytest.fixture(scope="module")
def nvarc():
    # NVARC's own ArcDataset.forward_mod and invert_mod (arc26/nvarc/arc_loader.py, imported
    # unchanged). The module imports transformers for its tokenizer class only; the smoke's
    # interpreter has none, so a stand-in module satisfies that import.
    import types

    stand_in = types.ModuleType("transformers")
    stand_in.AutoTokenizer = object
    saved = sys.modules.get("transformers")
    sys.modules.setdefault("transformers", stand_in)
    try:
        from arc26.nvarc.arc_loader import ArcDataset

        yield ArcDataset
    finally:
        if saved is None and sys.modules.get("transformers") is stand_in:
            del sys.modules["transformers"]


@pytest.mark.parametrize(
    "ops", [("transpose", "rot90"), ("rot90",), ("rot90", "rot90", "rot90"), ("transpose",)]
)  # fmt: skip
def test_the_view_equals_nvarcs_forward_and_invert_mod(nvarc, ops):
    rng = random.Random(3)
    for seed in range(3):
        view = fixed_view(ops, seed=seed)
        for _ in range(30):
            h, w = rng.randint(1, 30), rng.randint(1, 30)
            grid = [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]
            seen = view.forward(grid)
            assert seen == np.asarray(nvarc.forward_mod(np.array(grid), view.key)).tolist()
            assert view.inverse(seen) == grid
            assert view.inverse(seen) == np.asarray(
                nvarc.invert_mod(np.array(seen), view.key)).tolist()  # fmt: skip


def test_the_fixed_view_keeps_the_background_and_is_reproducible():
    first, second = fixed_view(["rot90"], 0), fixed_view(["rot90"], 0)
    assert first == second and first.colors[0] == 0 and sorted(first.colors) == list(range(10))
    assert first != fixed_view(["rot90"], 1)
    assert re.fullmatch(r"x\.rot90\.permute0\d{9}", first.key)
    # A quarter turn (counterclockwise) maps a row to a column, the last cell on top.
    colors = first.colors
    assert first.forward([[1, 2, 3]]) == [[colors[3]], [colors[2]], [colors[1]]]


@pytest.mark.parametrize(
    "ops, colors",
    [(("flip",), tuple(range(10))), (("rot90",), (1, 0, 2, 3, 4, 5, 6, 7, 8, 9)),
     (("rot90",), (0, 1, 2, 3, 4, 5, 6, 7, 8, 8))],
)  # fmt: skip
def test_a_view_refuses_unknown_operations_and_bad_permutations(ops, colors):
    with pytest.raises(ValueError):
        View(ops, colors)


def test_a_task_through_the_view_has_a_new_id_and_every_grid_transformed():
    task = AgentTask("t1", (Pair([[1, 2]], [[2, 1]]),), ([[3, 0]],))
    view = fixed_view(["transpose"], 0)
    seen = view.task(task)
    assert seen.task_id == "t1~v1"
    assert seen.train[0].input == view.forward([[1, 2]]) and len(seen.train[0].input) == 2
    assert seen.test_inputs[0] == view.forward([[3, 0]])


# --- the direct-answer prompt and answer forcing -----------------------------------------------


def test_the_direct_prompt_asks_for_answer_blocks_and_offers_no_repl():
    text = prompts.DIRECT_SYSTEM_PROMPT
    assert "```answer" in text and '"test_index"' in text and "no code interpreter" in text
    assert "transform(" not in text and "REPL" not in text


class NoRepl:
    def run_cell(self, code, timeout_s):
        return spike.NoRepl().run_cell(code, timeout_s)

    def close(self):
        pass


def flip_task():
    return AgentTask("flip", (Pair([[1, 2]], [[2, 1]]), Pair([[3, 4, 5]], [[5, 4, 3]])),
                     ([[6, 7]],))  # fmt: skip


def direct_budget(**overrides):
    settings = dict(
        max_turns=1, attempt_tokens=2000, turn_tokens=1000, forcing_tokens=500,
        context_tokens=60_000, preserve_thinking=True, forcing_kind="answer",
    )  # fmt: skip
    settings.update(overrides)
    return Budget(**settings)


def test_a_cut_direct_reply_gets_the_answer_forcing_note_and_its_answer_is_recorded():
    task = flip_task()
    engine = ScriptedEngine({"flip": [overlong_reply(8000), answer_reply(0, [[7, 6]])]})
    record = asyncio.run(run_attempt(task, engine, NoRepl(), spike.NoVerifier(), direct_budget(),
                                     system=prompts.DIRECT_SYSTEM_PROMPT))  # fmt: skip
    first, forced = engine.calls
    assert first.messages[0].content == prompts.DIRECT_SYSTEM_PROMPT
    assert forced.thinking is False and forced.max_tokens == 500
    note = forced.messages[-1].content
    assert note == feedback.forcing_message([], limit=1000, kind="answer")
    assert "exactly one ```answer block per test input" in note and "transform" not in note
    assert record.direct_answers == ((0, [[7, 6]]),) and record.forcing_turns == 1
    assert record.stop_reason == "rounds" and record.verifications == ()


def test_the_answer_reserve_counts_the_answer_note():
    budget = direct_budget()
    assert budget.forcing_reserve() == 500 + feedback.forcing_chars("answer") + 16 + 32 + 64
    with pytest.raises(ValueError, match="forcing_kind"):
        direct_budget(forcing_kind="essay")


def test_a_code_block_in_the_direct_arm_is_answered_with_a_note_never_run():
    result = spike.NoRepl().run_cell("import os\nos.system('true')", 1.0)
    assert result.error.startswith("no code interpreter") and not result.defines_transform
    with pytest.raises(RuntimeError):
        spike.NoVerifier().verify(None, "def transform(g):\n    return g\n")


# --- arm D's report ------------------------------------------------------------------------------


def record(task_id, answers, tokens=100, forcing=0):
    return AttemptRecord(task_id, f"{task_id}/p0/0", 0, (), tuple(answers), 10, tokens, 1, 1.0,
                         "rounds", forcing, 0)  # fmt: skip


def task(task_id, tests=1):
    return AgentTask(task_id, (Pair([[1]], [[1]]),), tuple([[0]] for _ in range(tests)))


READING = {"enter_misses": 2, "affordable_tokens_per_task": 1000, "contamination_gap": 0.10,
           "max_cut_share": 0.10}  # fmt: skip


def test_direct_metrics_score_pass_at_2_map_the_view_back_and_read_the_control():
    view = fixed_view(["rot90"], 0)
    a, b, c, d = [[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]
    tasks = [task("aa", tests=2), task("bb"), task("cc"), task("dd")]
    solutions = {"aa": [a, b], "bb": [c], "cc": [d], "dd": [a]}
    records = [
        # aa: the original right on both outputs; the view right on output 0 only (agree there).
        record("aa", [(0, a), (1, b)]),
        record("aa~v1", [(0, view.forward(a)), (1, view.forward(c))]),
        # bb: only the view right (mapped back by the inverse).
        record("bb", [(0, d)]), record("bb~v1", [(0, view.forward(c))]),
        # cc: both agree on a wrong grid.
        record("cc", [(0, a)]), record("cc~v1", [(0, view.forward(a))]),
        # dd: no answer at all.
        record("dd", []), record("dd~v1", []),
    ]  # fmt: skip
    metrics = spike.direct_metrics(tasks, records, view, solutions, ["aa", "bb", "cc"], "T", 3600.0,
                                   {"healthy": True}, READING)  # fmt: skip
    assert metrics["score"] == 2.0 and metrics["test_outputs"] == 5
    assert metrics["M_direct"] == 2 and metrics["nvarc_misses_solved"] == ["aa", "bb"]
    original, seen = metrics["samples"]["original"], metrics["samples"]["view"]
    assert (original["answered"], original["right"], original["tasks_right"]) == (4, 2, 1)
    assert (seen["answered"], seen["right"], seen["tasks_right"]) == (4, 2, 1)
    assert metrics["samples"]["original_minus_view"] == 0.0
    # Both answered 4 outputs; they agree on aa's output 0 (right) and cc's (wrong).
    assert (metrics["agreed"], metrics["agreement"], metrics["precision_when_agreed"]) == (
        2, 0.5, 0.5)  # fmt: skip
    assert metrics["tokens_per_task"] == 200 and metrics["solves_per_hour"] == 2.0
    assert metrics["route"]["reading"] == "enters"


@pytest.mark.parametrize(
    "solved, fault, tokens, verdict",
    [(5, False, 40 * 54_000, "enters"), (4, False, 1000, "does not enter"),
     (9, False, 40 * 54_001, "does not enter"), (9, True, 1000, "Fault")],
)  # fmt: skip
def test_route_reading_follows_the_pre_registered_rule(solved, fault, tokens, verdict):
    reading = {"enter_misses": 5, "affordable_tokens_per_task": 54_000}
    assert spike.route_reading("D", solved, fault, tokens, 40, reading)["reading"] == verdict
    assert spike.route_reading("L", 9, "Fault (rate)", 1000, 40, reading)["reading"] == (
        "Fault (rate)")  # fmt: skip


# --- the smoke path and the config ---------------------------------------------------------------


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    out = tmp_path_factory.mktemp("routes")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--smoke", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=600, cwd=REPO,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out


def test_the_smoke_runs_the_forcing_check_then_arm_d_then_arm_l(routes):
    run = json.loads((routes / "run.json").read_text())
    assert run["phases"] == ["forcing", "direct", "agent"] and run["exit_code"] == 0
    assert json.loads((routes / "forcing_check.json").read_text())["verdict"] == "pass"
    direct = json.loads((routes / "direct_metrics.json").read_text())
    assert (direct["score"], direct["M_direct"], direct["of"]) == (3.0, 3, 3)
    # mirror: cut, then the answer-forcing turn; transpose: right on the original only; recolor:
    # right on the view only, mapped back.
    assert direct["samples"]["original"]["right"] == 2 and direct["samples"]["view"]["right"] == 2
    assert (direct["agreed"], direct["precision_when_agreed"], direct["forcing_turns"]) == (
        1, 1.0, 1)  # fmt: skip
    attempts = [json.loads(line) for line in (routes / "direct_attempts.jsonl").read_text()
                .splitlines()]  # fmt: skip
    assert sorted(a["task_id"] for a in attempts) == sorted(
        t + s for t in ("smoke-mirror", "smoke-recolor", "smoke-transpose") for s in ("", "~v1"))
    agent = json.loads((routes / "agent_metrics.json").read_text())
    assert agent["subset"] == "fixtures, NVARC misses" and agent["tasks"] == 3
    assert agent["phases"][0]["forcing_turns"] == 1 and agent["route"]["arm"] == "L"
    summary = (routes / "summary.md").read_text()
    assert "arm D (direct answers" in summary and "arm L route" in summary


def test_the_route_config_runs_the_misses_in_one_wave_inside_the_kv_pool():
    config = spike.load_config(CONFIG)
    agent, direct, server = config["agent"], config["direct"], config["server"]
    assert config["phases"] == ["forcing", "direct", "agent"]
    assert server["max_model_len"] == agent["context_tokens"] == 65536
    assert agent["only_nvarc_misses"] and agent["concurrency"] == len(agent["nvarc_misses"]) == 29
    (phase,) = agent["phases"]
    assert (phase["rounds"], phase["reasoning_effort"], phase["budget_s"]) == (1, "xhigh", 6600)
    # 29 sequences at their peak (median prompt 4,250, the cap, a forced reply of 861 tokens) fit
    # the KV pool measured on 12 and 13 Oct.
    assert 29 * (4250 + phase["max_tokens"] + 861) <= 1_416_305
    assert direct["context_tokens"] == 32768 and direct["reasoning_effort"] == "low"
    reserve = direct["forcing_tokens"] + feedback.forcing_chars("answer") + 112
    assert direct["max_tokens"] + reserve + 6000 <= direct["context_tokens"]
    stages = 600 + 300 + 900 + config["forcing"]["budget_s"] + direct["budget_s"]
    stages += phase["budget_s"]
    assert stages == 12300 and stages + 60 <= config["max_wall_s"] == 12600
    # Arm L in lockstep at 220 tokens per second: the thinking turns, then forcing replies at
    # their cap, inside the arm's budget.
    assert 29 * (phase["max_tokens"] + agent["forcing_tokens"]) / 220 <= phase["budget_s"]


def test_the_route_builds_have_their_own_lineage_and_hard_stop():
    builder = load(REPO / "scripts" / "build_kaggle_notebook.py", "_test_nb_builder_routes")
    parsed = {}
    for name in ("P6.3-02-script", "P6.3-04-script", "P6.3-04-cpu-script"):
        path = REPO / "configs" / f"{name}.yaml"
        parsed[name] = builder.parse_config(path.read_bytes(), path)
    old, gpu, cpu = parsed["P6.3-02-script"], parsed["P6.3-04-script"], parsed["P6.3-04-cpu-script"]
    assert gpu.kaggle_notebook == "alyahya737/arc26-p6-3-route" != old.kaggle_notebook
    assert gpu.code_dataset == cpu.code_dataset == "alyahya737/arc26-code-p6-route"
    assert cpu.kaggle_notebook == "alyahya737/arc26-p6-3-route-cpu" and cpu.accelerator is None
    assert gpu.accelerator == "NvidiaL4" and gpu.script_config == cpu.script_config
    config = spike.load_config(CONFIG)
    assert 188.9 + config["max_wall_s"] + 300 <= gpu.script_kill_s == 13100


def test_every_route_config_value_has_a_comment():
    for name in ("P6.3-04", "P6.3-04-script", "P6.3-04-cpu-script"):
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
        (lambda c: c["direct"].update(max_tokens=30000), "reach direct.context_tokens"),
        (lambda c: c["direct"].update(token_budget=1000), "exceed token_budget"),
        (lambda c: c["direct"].update(view_ops=["flip"]), "view_ops"),
        (lambda c: c["direct"].update(context_tokens=131072), "exceeds server.max_model_len"),
        (lambda c: c.pop("reading"), "exactly the keys"),
        (lambda c: c["agent"].update(only_nvarc_misses="yes"), "only_nvarc_misses"),
    ],
)  # fmt: skip
def test_route_config_rules_refuse_bad_values(edit, message):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    with pytest.raises(spike.ConfigError, match=message):
        spike.check_config(raw)


def test_the_stage_2_config_needs_no_route_keys():
    config = spike.load_config(REPO / "configs" / "P6.3-02.yaml")
    assert config["agent"]["only_nvarc_misses"] is False and "direct" not in config


# --- the pre-registered readings (review, 13 Oct) ----------------------------------------------


def contamination_case(view_right):
    view = fixed_view(["rot90"], 0)
    ids = [f"m{k}" for k in range(10)]
    tasks = [task(t) for t in ids]
    answer = [[1, 2]]
    solutions = {t: [answer] for t in ids}
    records = []
    for k, t in enumerate(ids):
        records.append(record(t, [(0, answer)]))
        given = answer if k < view_right else [[3, 4]]
        records.append(record(t + "~v1", [(0, view.forward(given))]))
    reading = {**READING, "enter_misses": 5}
    return spike.direct_metrics(tasks, records, view, solutions, ids, "T", 3600.0,
                                {"healthy": True}, reading)  # fmt: skip


def test_a_gap_of_0_10_or_more_counts_only_the_misses_the_view_solved():
    # The original right on 10 of 10 misses, the view on 2: a gap of 0.8.
    metrics = contamination_case(view_right=2)
    assert metrics["M_direct"] == 10 and metrics["M_view"] == 2 and metrics["contaminated"]
    assert metrics["M_counted"] == 2 and metrics["route"]["solved"] == 2
    assert metrics["route"]["reading"] == "does not enter"
    # Exactly at the threshold: 9 of 10 on the view, a gap of 0.10, still the view's count.
    at = contamination_case(view_right=9)
    assert at["contaminated"] and at["M_counted"] == 9


def test_a_gap_under_0_10_counts_both_samples():
    metrics = contamination_case(view_right=10)
    assert not metrics["contaminated"] and metrics["M_counted"] == 10
    assert metrics["route"]["reading"] == "enters"


@pytest.mark.parametrize(
    "gate, cut, fault",
    [
        ({"healthy": True, "harness": True}, 0.0, False),
        ({"healthy": False, "harness": True}, 0.0, True),
        ({"healthy": True, "harness": False}, 0.0, True),
        ({"healthy": True, "harness": True}, 0.11, "Fault (rate)"),
        ({"healthy": True, "harness": True}, 0.10, False),
        ({"healthy": False, "harness": True}, 0.5, True),
    ],
)  # fmt: skip
def test_arm_l_reads_health_first_then_the_deadline_guard(gate, cut, fault):
    assert spike.arm_l_fault(gate, {"share": cut}, READING) == fault


def test_arm_l_reading_ignores_adr_020s_precision_and_needs_health():
    reading = {"enter_misses": 5, "affordable_tokens_per_task": 54_000}
    # M 8 at precision 0.5 with a healthy engine: ADR-020's rows say Fault; the route enters.
    healthy = spike.arm_l_fault({"healthy": True, "harness": True}, {"share": 0.0}, READING)
    assert spike.route_reading("L", 8, healthy, 29 * 45_000, 29, reading)["reading"] == "enters"
    # M 5 with an unhealthy engine: ADR-020's rows say Borderline; the route reads Fault.
    sick = spike.arm_l_fault({"healthy": False, "harness": True}, {"share": 0.0}, READING)
    assert spike.route_reading("L", 5, sick, 29 * 45_000, 29, reading)["reading"] == "Fault"


def test_forcing_cut_counts_deadline_stops_without_a_forcing_turn_and_unstarted_tasks():
    tasks = [task(t) for t in ("a", "b", "c", "d")]
    records = [
        AttemptRecord("a", "a/p1/0", 1, (), (), 1, 1, 1, 1.0, "deadline", 0, 0),
        AttemptRecord("b", "b/p1/0", 1, (), (), 1, 1, 2, 1.0, "deadline", 1, 1),
        AttemptRecord("c", "c/p1/0", 1, (), (), 1, 1, 2, 1.0, "rounds", 1, 1),
    ]  # fmt: skip
    assert spike.forcing_cut(tasks, records) == {"attempts": 4, "cut": 2, "share": 0.5}
