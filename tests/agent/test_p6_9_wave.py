"""Checks run P6.9-01, the measurement wave of the re-targeted and tightened shot (task P6.9).

What they prove: the 15 tasks are option 2c's, recomputed from P2.8's rows of P2.6-01 with the
brief's sampling rule; the configs equal P6.3-06's but for the listed keys, every value carries
a comment, and the budgets add up to the brief's Timeout; the estimated openings fit the context
and the KV pool; the CPU smoke exercises the budget check, the in-request forcing, and the lenient
parser; the spike refuses an in-request config whose server would not force the configured note;
the lenient parser on every earlier shot transcript (when the pulled logs are on this machine)
recovers exactly the recorded blocks and reads every strict-valid block as before; and the merge
replay (scripts/merge_replay.py) applies the pre-registered rows, and reproduces P6.3-05's
merge analysis on its logs. No GPU, no network beyond localhost, no competition data.
"""

import csv
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from arc26.agent import feedback
from arc26.agent.loop import fenced_blocks, parse_answer, record_to_json
from arc26.agent.types import AttemptRecord
from arc26.data import DEV120

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.9-01.yaml"
PREVIOUS = REPO / "configs" / "P6.3-06.yaml"
CSV = REPO / "reports" / "P2.8" / "P2.6-01.csv"
BLOCKS = REPO / "tests" / "testdata" / "p6_9_answer_blocks.json"
# The shots that ran before (configs/P6.3-05.yaml, configs/P6.3-06.yaml: direct.task_ids).
SHOT_BEFORE = {
    "136b0064", "195c6913", "446ef5d2", "581f7754", "80a900e0", "9385bd28", "cbebaa4b",
    "e12f9a14", "e3721c99", "f560132c", "16de56c4", "7b5033c1", "3e6067c3", "28a6681f",
    "dd6b8c4b", "13e47133", "21897d95", "269e22fb", "271d71e2", "291dc1e1", "4e34c42c",
    "5dbc8537", "65b59efc", "6e4f6532", "6ffbe589", "a32d8b75", "a47bf94d", "abc82100",
    "35ab12c3", "62593bfd",
}  # fmt: skip
POOL = 1699566  # P6.3-05's KV pool at 131,072, its boot log
MIN_RATIO = 95232 / 2190  # P6.3-06's opening bound (tests/agent/test_spike_confirm.py)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_p6_9_spike")
replay = load(REPO / "scripts" / "merge_replay.py", "_test_p6_9_replay")


def nvarc_facts():
    top, outputs, solved = {}, {}, defaultdict(float)
    for row in csv.DictReader(CSV.open(encoding="utf-8")):
        outputs[row["task_id"]] = round(1 / float(row["weight"]))
        if row["rank_kgmon"] == "1":
            top[(row["task_id"], int(row["output"]))] = float(row["softmax_p"])
        if row["attempt"] in ("1", "2") and row["correct"] == "1":
            solved[row["task_id"]] += float(row["weight"])
    confidence = {
        t: min(top.get((t, o), 0.0) for o in range(outputs[t])) if t in outputs else 0.0
        for t in DEV120
    }
    return confidence, solved


def test_the_wave_takes_option_2cs_15_tasks():
    confidence, solved = nvarc_facts()
    low = sorted((t for t in DEV120 if 0 < confidence[t] < 0.5), key=lambda t: (confidence[t], t))
    assert len(low) == 11
    fresh = [t for t in low if t not in SHOT_BEFORE]
    assert len(fresh) == 4 and all(solved[t] == 0 for t in fresh)
    mid_pool = sorted(t for t in DEV120 if 0.5 <= confidence[t] < 0.7)
    assert len(mid_pool) == 23
    # The task brief's pre-registered draw: 6 of the 23, uniform, seed 0.
    mid = random.Random(0).sample(mid_pool, 6)
    high_pool = sorted(t for t in DEV120 if confidence[t] >= 0.85 and solved[t] >= 1 - 1e-9)
    assert len(high_pool) == 27
    high = random.Random(0).sample(high_pool, 5)
    order = lambda ts: sorted(ts, key=lambda t: (confidence[t], t))  # noqa: E731
    expected = fresh + order(mid) + order(high)
    config = spike.load_config(CONFIG)
    assert config["direct"]["task_ids"] == expected
    assert expected == [
        "97d7923e", "247ef758", "7b3084d4", "898e7135", "142ca369", "88e364bc", "a25697e4",
        "a395ee82", "b5ca7ac4", "b0039139", "e8686506", "7ed72f31", "6e453dd6", "1818057f",
        "7c66cb00",
    ]  # fmt: skip
    # NVARC's solves on the 15, as the reading's list says.
    assert sorted(t for t in expected if solved[t] >= 1) == sorted(
        config["reading"]["nvarc_solved"]
    )
    assert sum(solved[t] for t in expected) == pytest.approx(7.0)
    assert sum(solved[t] for t in mid) == pytest.approx(2.0)


def attempt2_only_loss(path):
    """Per [0.5, 0.7) task of one run, the score an S1 in attempt 2's slot could lose."""
    outputs, top, att = {}, {}, defaultdict(dict)
    for row in csv.DictReader(path.open()):
        outputs[row["task_id"]] = round(1 / float(row["weight"]))
        if row["rank_kgmon"] == "1":
            top[(row["task_id"], int(row["output"]))] = float(row["softmax_p"])
        if row["attempt"] in ("1", "2"):
            att[(row["task_id"], int(row["output"]))][row["attempt"]] = row["correct"] == "1"
    conf = {t: min(top.get((t, o), 0.0) for o in range(n)) for t, n in outputs.items()}
    mid = [t for t in conf if 0.5 <= conf[t] < 0.7]
    loss = defaultdict(float)
    for (t, _), a in att.items():
        if t in mid and a.get("2") and not a.get("1"):
            loss[t] += 1 / outputs[t]
    return len(mid), dict(loss)


def test_the_brief_states_the_buckets_attempt_2_risk_from_both_runs():
    # P2.6-01: 3 of 23 tasks hold an output right by attempt 2 alone, 2.0 task-units in all;
    # P1.1-01: 5 of 23, 3.5 (the brief's bound on what replacing attempt 2 can lose).
    n, loss = attempt2_only_loss(CSV)
    assert (n, len(loss), sum(loss.values())) == (23, 3, 2.0)
    n, loss = attempt2_only_loss(REPO / "reports" / "P2.8" / "P1.1-01.csv")
    assert (n, len(loss), sum(loss.values())) == (23, 5, 3.5)


def test_the_15_have_21_test_outputs():
    stats = json.loads((REPO / "docs" / "research" / "eval_task_stats.json").read_text())
    chosen = spike.load_config(CONFIG)["direct"]["task_ids"]
    assert sum(stats["per_task"][t]["test_inputs"] for t in chosen) == 21


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


@pytest.mark.parametrize(
    "name, added, changed",
    [
        ("P6.9-01", {"direct.forcing_in_request", "direct.thinking_end_note",
                     "direct.lenient_answers", "direct.turn_cut_s"},
         {"run_id", "max_wall_s", "agent.stall_s", "server.extra_args", "direct.token_budget",
          "direct.budget_s", "direct.task_ids", "reading.nvarc_solved"}),
        ("P6.9-01-script", set(), {"script_config", "script_kill_s"}),
        ("P6.9-01-cpu-script", set(), {"script_config"}),
    ],
)  # fmt: skip
def test_the_configs_equal_p6_3_06_but_for_the_listed_keys(name, added, changed):
    new = flatten(yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text()))
    old_name = name.replace("P6.9-01", "P6.3-06")
    old = flatten(yaml.safe_load((REPO / "configs" / f"{old_name}.yaml").read_text()))
    assert set(new) - set(old) == added and set(old) <= set(new)
    assert {key for key in old if new[key] != old[key]} == changed


@pytest.mark.parametrize("name", ["P6.9-01", "P6.9-01-script", "P6.9-01-cpu-script"])
def test_every_wave_config_value_has_a_comment(name):
    path = REPO / "configs" / f"{name}.yaml"
    previous = ""
    for line in path.read_text().splitlines():
        if re.match(r"\s*[A-Za-z_0-9]+:", line):
            assert "#" in line or previous.lstrip().startswith("#"), f"{path.name}: {line}"
        if line.strip():
            previous = line


def test_the_budgets_add_up_to_the_briefs_timeout():
    config = spike.load_config(CONFIG)
    builder = load(REPO / "scripts" / "build_kaggle_notebook.py", "_test_p6_9_builder")
    path = REPO / "configs" / "P6.9-01-script.yaml"
    gpu = builder.parse_config(path.read_bytes(), path)
    cpu_path = REPO / "configs" / "P6.9-01-cpu-script.yaml"
    cpu = builder.parse_config(cpu_path.read_bytes(), cpu_path)
    assert gpu.kaggle_notebook == "alyahya737/arc26-p6-3-shot" and gpu.accelerator == "NvidiaL4"
    assert cpu.kaggle_notebook == "alyahya737/arc26-p6-3-shot-cpu" and cpu.accelerator is None
    assert gpu.code_dataset == cpu.code_dataset == "alyahya737/arc26-code-p6-route"
    assert gpu.script_config == cpu.script_config == "configs/P6.9-01.yaml"
    direct = config["direct"]
    stages = 600 + 300 + 900 + config["forcing"]["budget_s"] + direct["budget_s"]
    assert stages == 9300 and stages + 250 <= config["max_wall_s"] == 9600
    assert config["agent"]["stall_s"] == direct["budget_s"] == 7200
    # The script starts at most 200.2 s in; the hard stop 500 s after max_wall_s, as P6.3-05's.
    assert 200.2 + config["max_wall_s"] + 299 <= gpu.script_kill_s == 10100
    # The brief's Timeout: the hard stop, 300 s of logs, the 45-minute pre-script allowance.
    assert gpu.script_kill_s + 300 + 2700 == 13100
    brief = REPO / "docs" / "prerun" / "P6.9-01.md"
    if brief.exists():  # the brief is committed after its Code-commit
        timeout = int(re.search(r"^Timeout:\s+(\d+)", brief.read_text(), re.M).group(1))
        assert timeout == 13100
    # Every answer reachable down to 13.9 tokens per second per stream.
    assert (direct["max_tokens"] + direct["forcing_tokens"]) / direct["budget_s"] < 13.95


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
    assert (round(longest), round(total)) == (13276, 84200)
    # The thinking request holds the answer; the fallback forcing turn its reserve after it.
    reserve = direct["forcing_tokens"] + feedback.forcing_chars("answer2") + 112
    longest_request = round(longest) + 64 + direct["max_tokens"] + direct["forcing_tokens"]
    assert longest_request + reserve == 121738 <= direct["context_tokens"] == 131072
    assert server["max_model_len"] == direct["context_tokens"]
    assert total + 15 * (direct["max_tokens"] + 2000) <= 0.90 * POOL
    assert total + 15 * (direct["max_tokens"] + direct["forcing_tokens"]) <= 0.94 * POOL
    # The client-side cut: an honored request survives down to 15.9 tokens per second per stream,
    # and a cut one's fallback (the opening and an answer at 16.7) still ends inside budget_s.
    cut = direct["turn_cut_s"]
    assert (direct["max_tokens"] + direct["forcing_tokens"]) / cut == pytest.approx(15.93, 0.01)
    # Its fallback prefills at most the 15 openings at P6.3-05's 958.8 prompt tokens per second
    # and writes an answer at 16.7 per stream: 88 + 368 s.
    fallback = total / 958.8 + direct["forcing_tokens"] / 16.7
    assert cut + fallback == pytest.approx(6756, abs=1) and cut + fallback < direct["budget_s"]


@pytest.fixture(scope="module")
def wave_smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("wave")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--smoke", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=600, cwd=REPO,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out


def test_the_smoke_runs_the_budget_check_and_the_new_request(wave_smoke):
    run = json.loads((wave_smoke / "run.json").read_text())
    assert run["run_id"] == "P6.9-01" and run["exit_code"] == 0
    assert json.loads((wave_smoke / "forcing_check.json").read_text())["verdict"] == "pass"
    check = json.loads((wave_smoke / "budget_check.json").read_text())
    assert check["ok"] and check["forced"] and check["answer"] == "42"
    metrics = json.loads((wave_smoke / "shot_metrics.json").read_text())
    assert (metrics["tasks"], metrics["score"], metrics["solves"]) == (3, 3.0, 3)
    assert metrics["strict"] == {"score": 2.0, "solved": ["smoke-mirror", "smoke-transpose"],
                                 "solved_outputs": 2}  # fmt: skip
    assert metrics["forced_answers"] == {"tasks": 1, "outputs": 1, "answered": 1,
                                         "budget_forced": 1, "forcing_turns": 0, "rate": 1.0}
    per_task = metrics["per_task"]
    assert per_task["smoke-mirror"]["budget_forced"] == 1
    assert per_task["smoke-recolor"]["lenient_answers"] == 1
    assert per_task["smoke-mirror"]["implied"] == [[2, 3]]
    assert per_task["smoke-transpose"]["right"] == [[False, True]]
    attempts = [json.loads(line) for line in (wave_smoke / "direct_attempts.jsonl").open()]
    assert {a["task_id"]: a.get("answer_lenient") for a in attempts}["smoke-recolor"] == [True]


def test_a_failed_budget_check_runs_the_wave_with_the_separate_forcing_turn(tmp_path, monkeypatch):
    async def refused(engine, settings, seed):
        return {"ok": False, "forced": False,
                "error": "EngineError: HTTP 400: thinking_token_budget is set but ..."}

    monkeypatch.setattr(spike, "budget_check", refused)
    out = tmp_path / "out"
    assert spike.main(["--config", str(CONFIG), "--smoke", "--out-dir", str(out)]) == 0
    check = json.loads((out / "budget_check.json").read_text())
    assert check["in_request"] is False
    run = json.loads((out / "run.json").read_text())
    assert any("in-request forcing off" in p for p in run["problems"])
    metrics = json.loads((out / "shot_metrics.json").read_text())
    # smoke-mirror's overlong reply now runs to the cap and gets the separate forcing turn.
    mirror = metrics["per_task"]["smoke-mirror"]
    assert mirror["forcing_turns"] == 1 and mirror["budget_forced"] == 0
    assert metrics["forced_answers"]["budget_forced"] == 0
    calls = [json.loads(line) for line in (out / "engine_calls.jsonl").open()] if (
        out / "engine_calls.jsonl").exists() else []  # fmt: skip
    assert not any("thinking_token_budget" in c for c in calls)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["direct"].update(turn_cut_s=7200), "turn_cut_s"),
        (lambda c: c["direct"].update(forcing_in_request=False, thinking_end_note="",
                                      turn_cut_s=100), "turn_cut_s"),
        (lambda c: c["server"].update(extra_args=[]), "reasoning-config"),
        (lambda c: c["direct"].update(thinking_end_note="other"), "reasoning-config"),
        (lambda c: c["direct"].update(thinking_end_note=""), "thinking_end_note"),
        (lambda c: c["direct"].update(thinking_end_note="a</think>"), "think tags"),
        (lambda c: c["direct"].update(token_budget=100352), "token_budget"),
        (lambda c: c["direct"].update(lenient_answers="yes"), "lenient_answers"),
        (lambda c: c["server"].update(extra_args=["--reasoning-config", "{"]), "not JSON"),
    ],
)  # fmt: skip
def test_the_spike_refuses_an_in_request_config_that_cannot_work(edit, message):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    with pytest.raises(spike.ConfigError, match=message):
        spike.check_config(raw)


def test_p6_3_06s_config_reads_as_before():
    config = spike.load_config(PREVIOUS)["direct"]
    assert not config["forcing_in_request"] and not config["lenient_answers"]
    assert config["thinking_end_note"] == ""


# --- the lenient parser on every earlier shot transcript ----------------------------------------


TRANSCRIPTS = {
    "P6.3-05": Path.home() / "wt-P6.3c/runs/P6.3-05/run_logs/direct_transcripts.jsonl",
    "P6.3-06": Path.home() / "wt-P6.3d/runs/P6.3-06/run_logs/direct_transcripts.jsonl",
    "P6.5-02": Path.home() / "wt-P6.5b/runs/P6.5-02/run_logs/agent/shot_transcripts.jsonl",
    "P6.5-03": Path.home() / "wt-P6.5b/runs/P6.5-03/run_logs/agent/shot_transcripts.jsonl",
}
TEST_LINE = re.compile(r"Test input \d+ \(")


def test_the_manifest_records_p6_3_06s_nine_invalid_blocks():
    manifest = json.loads(BLOCKS.read_text())["runs"]
    blocks = manifest["P6.3-06"]["invalid_blocks"]
    assert len(blocks) == 9 and manifest["P6.3-06"]["valid_blocks"] == 24
    recovered = [b for b in blocks if b["lenient"]]
    assert {b["attempt_id"] for b in recovered} == {"62593bfd/p0/0"} and len(recovered) == 6
    assert sorted(b["attempt_id"] for b in blocks if not b["lenient"]) == [
        "13e47133/p0/0", "13e47133/p0/0", "271d71e2/p0/0"
    ]  # fmt: skip
    assert all("ragged" in b["strict"] for b in blocks if not b["lenient"])
    totals = {run: (v["valid_blocks"], len(v["invalid_blocks"]),
                    sum(1 for b in v["invalid_blocks"] if b["lenient"]))
              for run, v in manifest.items()}  # fmt: skip
    assert totals == {"P6.3-05": (24, 2, 0), "P6.3-06": (24, 9, 6), "P6.5-02": (2, 2, 2),
                      "P6.5-03": (41, 11, 1)}  # fmt: skip


@pytest.mark.parametrize("run", sorted(TRANSCRIPTS))
def test_the_lenient_parser_on_the_pulled_transcripts(run):
    path = TRANSCRIPTS[run]
    if not path.exists():
        pytest.skip(f"{run}'s pulled transcripts are not on this machine")
    expected = json.loads(BLOCKS.read_text())["runs"][run]
    valid, invalid = 0, []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        count = len(TEST_LINE.findall(user))
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            for tag, body in fenced_blocks(message["content"]):
                if tag != "answer":
                    continue
                try:
                    strict = parse_answer(body, count)
                except ValueError as err:
                    try:
                        index, grid = parse_answer(body, count, lenient=True)
                        lenient = {"test_index": index, "shape": [len(grid), len(grid[0])]}
                    except ValueError:
                        lenient = None
                    invalid.append({"attempt_id": row["attempt_id"],
                                    "sha256": hashlib.sha256(body.encode()).hexdigest(),
                                    "strict": str(err)[:80], "lenient": lenient})  # fmt: skip
                    continue
                # No change to a valid parse.
                assert parse_answer(body, count, lenient=True) == strict
                valid += 1
    assert valid == expected["valid_blocks"]
    assert invalid == expected["invalid_blocks"]


# --- the merge replay -----------------------------------------------------------------------------


def record(task_id, *answers):
    return AttemptRecord(task_id, f"{task_id}/p0/0", 0, (), tuple(answers), 0, 0, 1, 0.0, "rounds")


def write_replay_case(tmp_path, cut_rate=1.0):
    """Four tasks of three buckets; the CSV and the submission in P2.8's and NVARC's forms."""
    rows = [  # task, output, rank_kgmon, softmax_p, attempt, correct
        ("t1", 0, 1, 0.40, 1, 0), ("t1", 0, 2, 0.30, 2, 0),
        ("t2", 0, 1, 0.60, 1, 0), ("t2", 0, 2, 0.40, 2, 1),
        ("t3", 0, 1, 0.65, 1, 0), ("t3", 0, 2, 0.35, 2, 0),
        ("t4", 0, 1, 0.99, 1, 1), ("t4", 0, 2, 0.01, 2, 0),
    ]  # fmt: skip
    fields = ["task_id", "output", "weight", "rank_kgmon", "softmax_p", "attempt", "correct"]
    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for t, o, rank, p, a, c in rows:
            writer.writerow([t, o, "1.0", rank, p, a, c])
    A, B, C, D = [[1]], [[2]], [[3]], [[4]]
    nvarc = {t: [{"attempt_1": A, "attempt_2": B}] for t in ("t1", "t2", "t3", "t4")}
    (tmp_path / "submission.json").write_text(json.dumps(nvarc))
    logs = tmp_path / "run_logs"
    logs.mkdir()
    # t1 (0, 0.5): the shot right, gained. t2 [0.5, 0.7): NVARC right by N2 alone, the shot
    # wrong: (N1, S1) loses it. t3 [0.5, 0.7): the shot right, gained. t4: the harness, solved.
    records = [record("t1", (0, C)), record("t2", (0, D)), record("t3", (0, C)),
               record("t4", (0, A))]  # fmt: skip
    (logs / "direct_attempts.jsonl").write_text(
        "".join(json.dumps(record_to_json(r)) + "\n" for r in records)
    )
    right = {"t1": [[True, False]], "t2": [[False, False]], "t3": [[True, False]],
             "t4": [[True, False]]}  # fmt: skip
    metrics = {
        "reading": "does not enter", "score": 3.0, "strict": {"score": 3.0},
        "forced_answers": {"tasks": 3, "outputs": 15, "answered": round(15 * cut_rate),
                           "budget_forced": 3, "forcing_turns": 0, "rate": cut_rate},
        "per_task": {t: {"outputs": 1, "right": right[t], "implied": [[1, 1]]} for t in right},
    }  # fmt: skip
    (logs / "shot_metrics.json").write_text(json.dumps(metrics))
    (logs / "budget_check.json").write_text(json.dumps({"ok": True}))
    return logs, tmp_path / "submission.json", path


def test_the_replay_counts_gains_and_losses_per_bucket(tmp_path):
    logs, submission, rows = write_replay_case(tmp_path)
    out = tmp_path / "report"
    assert replay.main(["--run-logs", str(logs), "--nvarc-submission", str(submission),
                        "--nvarc-csv", str(rows), "--out", str(out)]) == 0  # fmt: skip
    data = json.loads((out / "reading.json").read_text())
    buckets = data["buckets"]
    assert buckets["(0, 0.5)"]["net"] == 1.0 and buckets["(0, 0.5)"]["gained"] == ["t1"]
    mid = buckets["[0.5, 0.7)"]
    assert (mid["nvarc"], mid["merged"], mid["net"]) == (1.0, 1.0, 0.0)
    assert mid["gained"] == ["t3"] and mid["lost"] == ["t2"]
    assert buckets["[0.85, 1]"]["shot_solved"] == ["t4"]
    reading = data["reading"]
    assert reading["fault"] is None and not reading["harness"]["investigate"]
    assert reading["extend"]["decision"].startswith("keep")
    assert reading["forcing"]["decision"] == "keep the in-request forcing"
    pairs = {r["task_id"]: r["merged_pair"] for r in csv.DictReader((out / "merge.csv").open())}
    assert pairs == {"t1": "N1,S1", "t2": "N1,S1", "t3": "N1,S1", "t4": "N1,N2"}
    assert "Queue" in (out / "merge.md").read_text()


def test_the_replays_rows_read_in_order():
    table = {"[0.5, 0.7)": {"net": 1.0, "tasks": 6},
             "[0.85, 1]": {"tasks": 5, "shot_solved": []}}  # fmt: skip
    forced = {"forced_answers": {"rate": 0.8, "budget_forced": 2, "answered": 4, "outputs": 5}}
    reading = replay.read_rows({"reading": "does not enter", **forced}, table, {"ok": True})
    # A net of 1.0 less the CSVs' expected loss, 6 x 5.5 / 46 = 0.717: keep.
    assert reading["harness"]["investigate"] and reading["extend"]["decision"].startswith("keep")
    assert reading["extend"]["adjusted"] == pytest.approx(0.283, abs=1e-3)
    table["[0.5, 0.7)"]["net"] = 2.0
    extended = replay.read_rows({"reading": "does not enter", **forced}, table, {"ok": True})
    assert extended["extend"]["decision"].startswith("extend the queue to [0.5, 0.7) before")
    assert replay.MID_LOSS == pytest.approx(5.5 / 46)
    assert reading["forcing"]["decision"] == "revert to the separate forcing turn"
    unhonored = replay.read_rows({**forced, "forced_answers": {"rate": 1.0, "budget_forced": 0}},
                                 table, {"ok": False})  # fmt: skip
    assert "did not end" in unhonored["forcing"]["decision"]
    none = replay.read_rows({"forced_answers": {"rate": None}}, table, None)
    assert none["forcing"]["decision"].startswith("not measured")
    assert replay.read_rows({"reading": "Fault (rate)"}, table, None) == {"fault": "Fault (rate)"}


P6305 = Path.home() / "wt-P6.3c/runs/P6.3-05/run_logs"
P2601 = Path.home() / "wt-P2.6/runs/P2.6-01/submission.json"


def test_the_replay_reproduces_p6_3_05s_merge_analysis(tmp_path):
    if not (P6305.exists() and P2601.exists()):
        pytest.skip("P6.3-05's and P2.6-01's pulled logs are not on this machine")
    shown = subprocess.run(["git", "show", "task/P6.3c:reports/P6.3-05/merge.csv"], cwd=REPO,
                           capture_output=True, text=True)  # fmt: skip
    if shown.returncode != 0:
        pytest.skip("task/P6.3c's merge table is not in this clone")
    flags = defaultdict(list)
    for row in csv.DictReader(shown.stdout.splitlines()):
        flags[row["task_id"]].append([row["S1_right"] == "1", row["S2_right"] == "1"])
    # 3e6067c3 scored 0.5: one output right by S1; either assignment gives the same numbers.
    flags["3e6067c3"] = [[True, False], [False, False]]
    metrics = json.loads((P6305 / "shot_metrics.json").read_text())
    for task_id, row in metrics["per_task"].items():
        row["right"] = flags[task_id]
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "shot_metrics.json").write_text(json.dumps(metrics))
    (logs / "direct_attempts.jsonl").write_text((P6305 / "direct_attempts.jsonl").read_text())
    out = tmp_path / "report"
    assert replay.main(["--run-logs", str(logs), "--nvarc-submission", str(P2601),
                        "--out", str(out)]) == 0  # fmt: skip
    buckets = json.loads((out / "reading.json").read_text())["buckets"]
    # reports/P6.3-05/merge.md: planned 4.5 of 15 against NVARC's 0; 2.5 in (0, 0.5), 2.0 at 0.
    assert {k: (v["nvarc"], v["merged"]) for k, v in buckets.items()} == {
        "(0, 0.5)": (0.0, 2.5), "0": (0.0, 2.0)
    }  # fmt: skip
