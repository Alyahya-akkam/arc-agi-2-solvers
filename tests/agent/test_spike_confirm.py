"""Checks run P6.3-06, the long shot's confirmation: P6.3-05's shot unchanged on 15 fresh tasks.

The 15 tasks are recomputed from P2.8's rows of P2.6-01 by P6.3-05's rule on the DEV120 tasks
outside DEV40; the config must equal P6.3-05's but for the listed keys; the opening estimates
must fit the context and the KV pool P6.3-05 measured. No GPU, no network beyond localhost.
"""

import csv
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from arc26.agent import feedback
from arc26.data import DEV40, DEV120

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.3-06.yaml"
PREVIOUS = REPO / "configs" / "P6.3-05.yaml"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike_confirm")
OUTSIDE = [t for t in DEV120 if t not in DEV40]


def confidence_and_solves(tasks):
    # P6.3-05's rule: per output the softmax share of its rank_kgmon 1 row (0 without candidates),
    # the minimum over a task's outputs (0 for a task without rows); a task's solved share on the
    # two submitted attempts.
    top, outputs, solved = {}, {}, defaultdict(float)
    for row in csv.DictReader((REPO / "reports" / "P2.8" / "P2.6-01.csv").open()):
        outputs[row["task_id"]] = round(1 / float(row["weight"]))
        if row["rank_kgmon"] == "1":
            top[(row["task_id"], int(row["output"]))] = float(row["softmax_p"])
        if row["attempt"] in ("1", "2") and row["correct"] == "1":
            solved[row["task_id"]] += float(row["weight"])
    confidence = {
        t: min(top.get((t, o), 0.0) for o in range(outputs[t])) if t in outputs else 0.0
        for t in tasks
    }
    return confidence, solved


def test_the_confirmation_takes_the_15_least_confident_tasks_outside_dev40():
    assert len(OUTSIDE) == 80
    confidence, solved = confidence_and_solves(OUTSIDE)
    order = sorted(OUTSIDE, key=lambda t: (confidence[t], t))
    chosen = order[:15]
    config = spike.load_config(CONFIG)
    assert config["direct"]["task_ids"] == chosen
    assert sum(1 for t in chosen if confidence[t] == 0.0) == 13
    assert [round(confidence[t], 3) for t in chosen[13:]] == [0.381, 0.384]
    # The 16th, reported in the brief.
    assert (order[15], round(confidence[order[15]], 3)) == ("97d7923e", 0.4)
    # NVARC: score 0 and no solve on the 15; the reading's list says so.
    assert sum(solved[t] for t in chosen) == 0.0
    assert [t for t in chosen if solved[t] >= 1] == [] == config["reading"]["nvarc_solved"]
    # Fresh: none of P6.3-05's 15.
    assert not set(chosen) & set(spike.load_config(PREVIOUS)["direct"]["task_ids"])


def test_the_15_have_25_test_outputs():
    stats = json.loads((REPO / "docs" / "research" / "eval_task_stats.json").read_text())
    chosen = spike.load_config(CONFIG)["direct"]["task_ids"]
    assert sum(stats["per_task"][t]["test_inputs"] for t in chosen) == 25


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


@pytest.mark.parametrize(
    "name, changed",
    [
        ("P6.3-06", {"run_id", "direct.task_ids"}),
        ("P6.3-06-script", {"script_config"}),
        ("P6.3-06-cpu-script", {"script_config"}),
    ],
)  # fmt: skip
def test_the_configs_equal_p6_3_05_but_for_the_listed_keys(name, changed):
    new = flatten(yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text()))
    old_name = name.replace("P6.3-06", "P6.3-05")
    old = flatten(yaml.safe_load((REPO / "configs" / f"{old_name}.yaml").read_text()))
    assert new.keys() == old.keys()
    assert {key for key in new if new[key] != old[key]} == changed


# The openings of the 15 are not measured: NVARC's TTT token counts (P1.1-01) scale with a task's
# size, and on DEV40 the ratio of those counts to the shot's measured openings (P6.3-04's
# prompt_counts.jsonl) is at least 95,232 / 2,190 (446ef5d2); dividing by it bounds the opening
# from above, plus the 30 tokens P6.3-05's two-answer prompt added.
MIN_RATIO = 95232 / 2190
POOL = 1699566  # P6.3-05's KV pool at 131,072, its boot log


def opening_bounds():
    rows = {r["puzzle_id"]: r for r in csv.DictReader((REPO / "reports" / "P1.1-01" /
                                                       "puzzles.csv").open())}  # fmt: skip
    chosen = spike.load_config(CONFIG)["direct"]["task_ids"]
    return {t: int(rows[t]["train_tokens"]) / MIN_RATIO + 30 for t in chosen}


def test_the_estimated_openings_fit_the_context_and_the_measured_kv_pool():
    config = spike.load_config(CONFIG)
    direct, server = config["direct"], config["server"]
    bounds = opening_bounds()
    longest, total = max(bounds.values()), sum(bounds.values())
    assert (round(longest), round(total)) == (10035, 84201)
    reserve = direct["forcing_tokens"] + feedback.forcing_chars("answer2") + 112
    assert round(longest) + 64 + direct["max_tokens"] + reserve == 112353
    assert server["max_model_len"] == direct["context_tokens"] == 131072
    # Room for an opening almost 3 times the bound.
    assert direct["context_tokens"] - 64 - direct["max_tokens"] - reserve == 28754
    assert total + 15 * (direct["max_tokens"] + 2000) <= 0.90 * POOL
    assert total + 15 * (direct["max_tokens"] + direct["forcing_tokens"]) <= 0.94 * POOL


@pytest.fixture(scope="module")
def confirm_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("confirm")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--smoke", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=600, cwd=REPO,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out


def test_the_smoke_passes_as_p6_3_05s(confirm_run):
    run = json.loads((confirm_run / "run.json").read_text())
    assert run["run_id"] == "P6.3-06" and run["exit_code"] == 0
    assert json.loads((confirm_run / "forcing_check.json").read_text())["verdict"] == "pass"
    assert json.loads((confirm_run / "harness_preflight.json").read_text())["ok"] is True
    metrics = json.loads((confirm_run / "shot_metrics.json").read_text())
    assert (metrics["tasks"], metrics["score"], metrics["solves"]) == (3, 2.0, 2)
    assert metrics["solved_by_second_attempt_only"] == ["smoke-transpose"]
    assert metrics["forcing_turns"] == 1 and metrics["reading"] == "does not enter"


# --- the shot's tasks from the whole challenges file -------------------------------------------


def write_challenges(path, ids):
    # Stand-in tasks of varied sizes for every id (the repository holds no evaluation data).
    rng = __import__("random").Random(0)

    def grid():
        h, w = rng.randint(1, 30), rng.randint(1, 30)
        return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]

    tasks = {
        t: {"train": [{"input": grid(), "output": grid()} for _ in range(rng.randint(2, 4))],
            "test": [{"input": grid()} for _ in range(rng.randint(1, 2))]}
        for t in ids
    }  # fmt: skip
    path.write_text(json.dumps(tasks))
    return path


def test_the_confirmations_15_load_from_the_whole_file_with_their_label(tmp_path):
    challenges = write_challenges(tmp_path / "arc-agi_evaluation_challenges.json", DEV120)
    config = spike.load_config(CONFIG)
    ids = config["direct"]["task_ids"]
    tasks, label = spike.selected_tasks(challenges, ids, config["agent"]["subset"])
    assert [t.task_id for t in tasks] == ids and label == "DEV120, 15 selected"
    # The forcing check and the harness preflight keep DEV40's tasks.
    assert config["agent"]["subset"] == "DEV40"


def test_p6_3_05s_tasks_and_renderings_are_unchanged_by_the_new_path(tmp_path):
    from arc26.agent.prompts import DIRECT_TWO_SYSTEM_PROMPT, initial_messages
    from arc26.agent.tasks import load_tasks
    from arc26.data import subset_ids

    challenges = write_challenges(tmp_path / "arc-agi_evaluation_challenges.json", DEV120)
    config = spike.load_config(PREVIOUS)
    ids, subset = config["direct"]["task_ids"], config["agent"]["subset"]
    # The path P6.3-05 ran (acac69d): the subset's tasks, then the configured ones by id.
    by_id = {task.task_id: task for task in load_tasks(challenges, subset_ids(subset))}
    old, old_label = [by_id[t] for t in ids], f"{subset}, {len(ids)} selected"
    new, new_label = spike.selected_tasks(challenges, ids, subset)
    assert new == old and new_label == old_label == "DEV40, 15 selected"

    def rendered(tasks):
        opening = [initial_messages(t, system=DIRECT_TWO_SYSTEM_PROMPT) for t in tasks]
        return [json.dumps([(m.role, m.content) for m in ms]).encode() for ms in opening]

    assert rendered(new) == rendered(old)


def test_a_missing_shot_task_is_refused_before_the_install_and_any_engine_call(
    tmp_path, monkeypatch
):
    import argparse

    present = [t for t in DEV120 if t != "62593bfd"]
    folder = tmp_path / "input" / "arc-prize-2026-arc-agi-2"
    folder.mkdir(parents=True)
    challenges = write_challenges(folder / "arc-agi_evaluation_challenges.json", present)
    (folder / "arc-agi_evaluation_solutions.json").write_text("{}")
    ids = spike.load_config(CONFIG)["direct"]["task_ids"]
    with pytest.raises(spike.Stop, match=r"direct.task_ids: .*62593bfd"):
        spike.selected_tasks(challenges, ids, "DEV40")
    # In the run: the inputs stage refuses, before the harness preflight, install, and boot.
    config = spike.load_config(CONFIG)
    config.update(input_root=str(tmp_path / "input"), out_dir=str(tmp_path / "out"),
                  scratch_dir=str(tmp_path / "scratch"))  # fmt: skip
    monkeypatch.setattr(spike, "resolve_model", lambda root, settings: (root, {}))
    monkeypatch.setattr(spike, "resolve_wheelhouse", lambda root, settings: (root, {}))

    def reached(run):
        raise AssertionError("the harness preflight ran")

    monkeypatch.setattr(spike, "harness_preflight", reached)
    args = argparse.Namespace(smoke=False, model_root=str(tmp_path), venv_python=None,
                              wheelhouse_root=str(tmp_path), resume_from=None)  # fmt: skip
    with pytest.raises(spike.Stop, match="62593bfd"):
        spike.execute(spike.Run(config, args))
