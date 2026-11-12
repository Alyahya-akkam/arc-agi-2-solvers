"""Tests of the long shot after NVARC (task P6.5b, arc26.agent.shot).

What they prove: the confidence equals P2.8's softmax_p and picks P6.3-05's 15 tasks; the
requests are byte-identical to P6.3-05's direct phase (the spike's own code run beside the shot's
on the same stand-in tasks, against one canned server); the engine and its preparation are
P6.3-05's; the wave arithmetic and its constants; every merge case; and, end to end with P6.4's
fake engine (arc26.agent.server, started by the shot as it would start vLLM), that a cut call,
a failed boot, an engine death, and SIGTERM each leave a valid file, that competition mode never
opens a solutions file, and the exit codes. No GPU, no network beyond localhost, no competition
data: the stand-in tasks carry DEV40's ids with synthetic grids.
"""

import argparse
import asyncio
import csv
import gzip
import importlib.util
import inspect
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import yaml

from arc26.agent import engine_prep, recording, shot
from arc26.agent import run as agent_run
from arc26.agent.loop import Budget
from arc26.agent.server import free_port, vllm_command
from arc26.agent.types import AttemptRecord
from arc26.data import DEV40, SUBSETS
from arc26.submit import validate

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "agent_shot.yaml"
SPIKE_CONFIG = REPO / "configs" / "P6.3-05.yaml"
CSV = REPO / "reports" / "P2.8" / "P2.6-01.csv"
# P2.6-01's candidates.json reduced (reduce_candidates below): DEV120's view files with their
# score_aug values; each distinct grid of an output replaced by a one-row grid of its index's
# digits (grouping, not content, decides the tally), beam_score dropped (score_kgmon ignores it).
REDUCED = REPO / "tests" / "testdata" / "p2_6_01_candidates_reduced.json.gz"
ORIGINAL = Path.home() / "wt-P2.6/runs/P2.6-01/run_logs/candidates.json"
P6305_LOGS = Path.home() / "wt-P6.3c/runs/P6.3-05/run_logs"
# P6.3-05's 15 tasks in its order (configs/P6.3-05.yaml, direct.task_ids).
P6305_TASKS = [
    "136b0064", "195c6913", "446ef5d2", "581f7754", "80a900e0", "9385bd28", "cbebaa4b",
    "e12f9a14", "e3721c99", "f560132c", "16de56c4", "7b5033c1", "3e6067c3", "28a6681f",
    "dd6b8c4b",
]  # fmt: skip
# The vLLM command P6.3-05 booted (run_logs/metrics.json, server_command).
P6305_COMMAND = [
    "/tmp/agent-spike/venv/bin/vllm", "serve",
    "/kaggle/input/notebooks/alyahya737/arc26-dl-qwen38-int4/qwen38-27b-int4",
    "--host", "127.0.0.1", "--port", "8000", "--served-model-name", "qwen38-27b-int4",
    "--tensor-parallel-size", "4", "--kv-cache-dtype", "fp8", "--max-model-len", "131072",
    "--max-num-seqs", "64", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization",
    "0.9", "--attention-backend", "TRITON_ATTN", "--reasoning-parser", "qwen3",
    "--generation-config", "auto", "--seed", "0", "--disable-uvicorn-access-log",
    "--enable-prefix-caching", "--language-model-only",
]  # fmt: skip
# The brief's table (docs/tasks/P6.5b.md), measured in P6.3-05's logs.
MEASURED = {
    "rate_tokens_per_s": 19.0,  # engine_calls.jsonl: 94,208 tokens in 4,966.5 to 4,968.2 s
    # engine_calls.jsonl: the 8 forced replies' 783,425 prompt tokens in 817.1 s.
    "prefill_tokens_per_s": 958.8,
    "kv_pool_tokens": 1_699_566,  # boot.json: kv_cache_tokens_first
    "max_cap": 94_208,  # configs/P6.3-05.yaml: direct.max_tokens
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # inspect.getsource finds a class's file through it
    spec.loader.exec_module(module)
    return module


spike = load(REPO / "scripts" / "agent_spike.py", "_test_shot_spike")


def reduce_candidates(raw):
    """The reduction that wrote REDUCED from P2.6-01's candidates.json."""
    codes, out = {}, {}
    for name in sorted(raw):
        base = name.split(".", 1)[0]
        if base.rsplit("_", 1)[0] not in set(SUBSETS["DEV120"]):
            continue
        seen = codes.setdefault(base, {})
        out[name] = [
            {
                "solution": [[int(d) for d in str(seen.setdefault(json.dumps(s["solution"]),
                                                                    len(seen)))]],
                "score_aug": s["score_aug"],
            }
            for s in raw[name]
        ]  # fmt: skip
    return out


def reduced():
    return json.loads(gzip.decompress(REDUCED.read_bytes()))


def csv_rows():
    return list(csv.DictReader(CSV.open(encoding="utf-8")))


def csv_counts(rows):
    """Test outputs per task from the CSV's weights (1/n); 1 for a task without candidates."""
    counts = {r["task_id"]: round(1 / float(r["weight"])) for r in rows}
    return {t: counts.get(t, 1) for t in SUBSETS["DEV120"]}


# --- the confidence ---------------------------------------------------------------------------


def test_the_reduced_fixture_is_p2_6_01s_candidates_reduced():
    if not ORIGINAL.exists():
        pytest.skip("P2.6-01's candidates.json is local only")
    assert reduce_candidates(json.loads(ORIGINAL.read_text(encoding="utf-8"))) == reduced()


def test_the_confidence_equals_p2_8s_softmax_p_on_every_dev120_output():
    from arc26.loop import Candidate

    rows = csv_rows()
    candidates = reduced()
    grouped = defaultdict(list)
    for name in sorted(candidates):
        for sample in candidates[name]:
            grouped[shot.output_key(name)].append(
                Candidate(sample["solution"], 0.0, sample["score_aug"], name)
            )
    top = [r for r in rows if r["rank_kgmon"] == "1"]
    assert len(top) == 142 and {r["task_id"] for r in top} <= set(SUBSETS["DEV120"])
    for row in top:
        p = shot.output_confidence(grouped[(row["task_id"], int(row["output"]))])
        assert abs(p - float(row["softmax_p"])) <= 5e-7, row  # the CSV keeps 6 decimals
    # Per task: the minimum over its outputs, 0 for an output without candidates.
    counts = csv_counts(rows)
    confidence = shot.shot_confidence(counts, candidates)
    for task, count in counts.items():
        want = min(
            float(next((r["softmax_p"] for r in top
                        if r["task_id"] == task and int(r["output"]) == j), 0.0))
            for j in range(count)
        )  # fmt: skip
        assert abs(confidence[task] - want) <= 5e-7, task


def test_its_15_least_sure_dev40_tasks_are_p6_3_05s():
    counts = {t: c for t, c in csv_counts(csv_rows()).items() if t in DEV40}
    confidence = shot.shot_confidence(counts, reduced())
    queue = shot.select_tasks(confidence, shot.load_config(CONFIG).max_confidence)
    assert [t for t, _ in queue] == P6305_TASKS
    assert sum(1 for _, c in queue if c == 0.0) == 10
    # max_confidence 0.5 keeps exactly these: the next task, 4a21e3da, is at 0.511.
    assert round(confidence["4a21e3da"], 3) == 0.511 and round(queue[-1][1], 3) == 0.465
    assert spike.load_config(SPIKE_CONFIG)["direct"]["task_ids"] == P6305_TASKS


def test_the_rule_is_not_the_schedulers_and_ignores_other_tasks():
    from arc26.agent.scheduler import nvarc_confidence

    counts = {t: c for t, c in csv_counts(csv_rows()).items() if t in DEV40}
    mine = shot.shot_confidence(counts, reduced())
    theirs = nvarc_confidence(counts, reduced())
    assert any(abs(mine[t] - theirs[t]) > 0.05 for t in counts)
    assert set(shot.shot_confidence({"136b0064": 1}, reduced())) == {"136b0064"}


def test_without_candidates_only_tasks_nvarc_left_empty_are_taken():
    nvarc = {"a": [([[1]], None)], "b": [(None, None)], "c": [([[1]], [[2]]), (None, None)]}
    counts = {"a": 1, "b": 1, "c": 2, "d": 1}
    confidence = shot.confidence_without_candidates(counts, nvarc)
    assert confidence == {"a": None, "b": 0.0, "c": 0.0, "d": 0.0}
    assert [t for t, _ in shot.select_tasks(confidence, 0.5)] == ["b", "c", "d"]
    assert shot.confidence_without_candidates(counts, None) == dict.fromkeys(counts, 0.0)
    assert confidence == shot.shot_confidence(counts, None, nvarc)
    assert [t for t, _ in shot.select_tasks(confidence, 0.5, exclude={"c"})] == ["b", "d"]


def test_a_partial_candidates_file_leaves_absent_tasks_unknown_but_placeholders_at_zero():
    sure = [{"solution": [[1]], "score_aug": [1.0]}] * 10  # one grid: p = 1
    unsure = [{"solution": [[g]], "score_aug": [1.0]} for g in (1, 2, 3)]  # p = 1/3
    candidates = {"a_0.v": sure, "b_0.v": unsure, "c_1.v": unsure}
    nvarc = {
        "a": [([[1]], None)], "b": [([[1]], None)],
        "c": [([[5]], None), ([[1]], None)],  # output 0 absent from the file, NVARC real
        "d": [([[5]], None)],  # absent, NVARC real: unknown
        "e": [(None, None)],  # absent, NVARC's placeholder: 0
        "f": [(None, None), ([[5]], None)],  # one placeholder output, one unknown: 0
    }  # fmt: skip
    counts = {"a": 1, "b": 1, "c": 2, "d": 1, "e": 1, "f": 2}
    confidence = shot.shot_confidence(counts, candidates, nvarc)
    assert confidence["a"] == 1.0 and confidence["b"] == pytest.approx(1 / 3)
    assert confidence["c"] == pytest.approx(1 / 3)  # the known output bounds the minimum
    assert confidence["d"] is None and confidence["e"] == 0.0 and confidence["f"] == 0.0
    assert [t for t, _ in shot.select_tasks(confidence, 0.5)] == ["e", "f", "b", "c"]
    # Without NVARC's file every output without candidates is its placeholder: 0.
    assert shot.shot_confidence(counts, candidates)["d"] == 0.0


# --- the requests: byte-identical to P6.3-05's -------------------------------------------------


CUT = set(P6305_TASKS[:8])  # cut at the cap, then forced, as 8 of P6.3-05's 15 were


def stand_in_tasks():
    """Synthetic tasks under DEV40's ids (the competition's data stays out of the repository)."""
    rng = np.random.default_rng(5)
    challenges, solutions = {}, {}
    for k, task in enumerate(DEV40):
        shape = (3 + k % 5, 4 + k % 3)
        pairs = [rng.integers(0, 10, size=shape) for _ in range(2 + k % 3)]
        tests = [rng.integers(0, 10, size=shape) for _ in range(1 + (k % 4 == 0))]
        challenges[task] = {
            "train": [{"input": g.tolist(), "output": np.fliplr(g).tolist()} for g in pairs],
            "test": [{"input": g.tolist()} for g in tests],
        }
        solutions[task] = [np.fliplr(g).tolist() for g in tests]
    return challenges, solutions


def canned(url, body):
    """The canned server's reply: a /tokenize count or a chat completion, from the request."""
    if url.endswith("/tokenize"):
        return {"count": 1000 + sum(len(m["content"]) for m in body["messages"]) // 4}
    users = [m["content"] for m in body["messages"] if m["role"] == "user"]
    task = users[0].split(":", 1)[0].removeprefix("Task ")
    forcing = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
    answer = '```answer\n{"test_index": 0, "grid": [[1, 2], [3, 4]]}\n```'
    if task in CUT and not forcing:
        message = {"role": "assistant", "content": "", "reasoning": f"thinking on {task} " * 50}
        finish, tokens = "length", body["max_tokens"]
    else:
        message = {"role": "assistant", "content": answer, "reasoning": "brief"}
        finish, tokens = "stop", 300
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 2000 + len(json.dumps(body)) // 4, "completion_tokens": tokens},
    }


@pytest.fixture
def canned_server(monkeypatch):
    """Every request the engine client posts, recorded with its bytes, answered from canned()."""
    sent = []

    async def post(self, url, *, json=None, **kwargs):
        users = [m["content"] for m in json["messages"] if m["role"] == "user"]
        task = users[0].split(":", 1)[0].removeprefix("Task ")
        turn = sum(1 for m in json["messages"] if m["role"] == "assistant")
        full = str(url) if str(url).startswith("http") else f"{self.base_url}{url}"
        sent.append((full, task, turn, globals()["json"].dumps(json).encode("utf-8")))
        request = httpx.Request("POST", full)
        return httpx.Response(200, json=canned(full, json), request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return sent


def spike_requests(tmp_path, sent):
    """Run the spike's own direct phase (P6.3-05's code and config) against the canned server."""
    challenges, solutions = stand_in_tasks()
    folder = tmp_path / "input" / "competitions" / "arc-prize-2026-arc-agi-2"
    folder.mkdir(parents=True)
    (folder / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    (folder / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions))
    config = spike.load_config(SPIKE_CONFIG)
    config.update(input_root=str(tmp_path / "input"), out_dir=str(tmp_path / "spike"))
    out = tmp_path / "spike"
    out.mkdir()
    run = SimpleNamespace(
        config=config, args=argparse.Namespace(smoke=False), out=out, problems=[], metrics={},
        writer=spike.JsonlWriter(), deadline=time.monotonic() + 600,
        base_url=f"http://{config['server']['host']}:{config['server']['port']}",
    )  # fmt: skip
    spike.direct_phase(run)
    requests = list(sent)
    sent.clear()
    return requests, run, challenges


def shot_requests(tmp_path, sent, challenges):
    """Run one full wave of the shot on the same tasks, its queue from P2.6-01's candidates."""
    config = shot.load_config(CONFIG)
    path = tmp_path / "challenges.json"
    path.write_text(json.dumps(challenges))
    from arc26.agent.tasks import load_tasks

    tasks = {t.task_id: t for t in load_tasks(path, DEV40)}
    counts = {t: c for t, c in csv_counts(csv_rows()).items() if t in DEV40}
    queue = shot.select_tasks(shot.shot_confidence(counts, reduced()), config.max_confidence)
    logs = tmp_path / "shot"
    logs.mkdir()
    options = SimpleNamespace(log_dir=logs, mode="competition")
    writer = shot.ShotWriter(tmp_path / "submission.json", {t: 1 for t in tasks}, None)
    state = shot.ShotState(
        options=options, config=config, writer=writer, tasks=tasks,
        queue=[(t, shot.opening_bound(tasks[t], config.request.system)) for t, _ in queue],
        confidence=dict(queue), kv_pool=MEASURED["kv_pool_tokens"], kv_source="test",
        started=time.monotonic(), deadline=time.monotonic() + 9000, plan={"waves": []},
    )  # fmt: skip
    plan = shot.plan_wave(state.queue, 9000, config.waves, state.kv_pool)
    assert plan.cap == 94_208 and list(plan.tasks) == P6305_TASKS
    engine = recording.recording_engine(
        config.server.root_url, config.server.served_model_name, config.waves.wave_tasks, logs,
        config.degenerate, return_token_ids=config.return_token_ids,
        count_prompts=config.count_prompts, **config.engine,
    )  # fmt: skip

    async def wave():
        try:
            return await shot.run_wave(state, engine, 1, plan, SimpleNamespace(failed=False))
        finally:
            await engine.aclose()

    row = asyncio.run(wave())
    requests = list(sent)
    sent.clear()
    return requests, row, engine, state


def test_the_shots_requests_are_byte_identical_to_p6_3_05s(tmp_path, canned_server):
    theirs, run, challenges = spike_requests(tmp_path, canned_server)
    mine, row, _, state = shot_requests(tmp_path, canned_server, challenges)

    def keyed(requests):
        out = {}
        for url, task, turn, data in requests:
            assert (url, task, turn) not in out
            out[(url, task, turn)] = data
        return out

    theirs, mine = keyed(theirs), keyed(mine)
    chats = [k for k in mine if k[0].endswith("/chat/completions")]
    assert len(chats) == 15 + len(CUT) and len(mine) == 15 + len(chats)  # + 15 counts
    assert set(mine) == set(theirs)
    for key in theirs:
        assert mine[key] == theirs[key], key
    # The full cap reaches the request; the forcing turns run with thinking off.
    first = json.loads(mine[(chats[0][0], P6305_TASKS[0], 0)])
    assert first["max_tokens"] <= 94_208 and first["reasoning_effort"] == "xhigh"
    assert first["return_token_ids"] is True and first["seed"] > 0
    assert row["finished"] == row["used"] == 15 and run.metrics["shot"]["tasks"] == 15
    assert {r.attempt_id for r in state.records} == {f"{t}/p0/0" for t in P6305_TASKS}


def test_the_engine_client_and_budget_are_p6_3_05s(tmp_path):
    config = shot.load_config(CONFIG)
    spike_config = spike.load_config(SPIKE_CONFIG)
    run = SimpleNamespace(
        config=spike_config, out=tmp_path, writer=spike.JsonlWriter(),
        base_url=config.server.root_url,
    )  # fmt: skip
    theirs = spike.make_engine(run, spike_config["direct"]["concurrency"]).config
    mine = recording.recording_engine(
        config.server.root_url, config.server.served_model_name, config.waves.wave_tasks,
        tmp_path, config.degenerate, return_token_ids=config.return_token_ids,
        count_prompts=config.count_prompts, **config.engine,
    ).config  # fmt: skip
    assert mine == theirs
    direct = spike_config["direct"]
    assert config.request.budget(MEASURED["max_cap"]) == Budget(
        max_turns=1, attempt_tokens=direct["token_budget"], turn_tokens=direct["max_tokens"],
        context_tokens=direct["context_tokens"], reasoning_effort=direct["reasoning_effort"],
        cell_timeout_s=1.0, preserve_thinking=True, forcing_tokens=direct["forcing_tokens"],
        forcing_kind="answer2",
    )  # fmt: skip
    assert config.request.system == shot.prompts.DIRECT_TWO_SYSTEM_PROMPT
    assert config.seed == spike_config["seed"] and shot.SHOT_PHASE == 0
    assert config.degenerate["max_run"] == spike_config["degenerate"]["max_run"]
    assert config.degenerate["eos_ids"] == spike_config["degenerate"]["eos_ids"]
    assert config.degenerate["wave_share"] == spike_config["kill"]["degenerate_share"]


def test_the_vllm_command_and_environment_are_p6_3_05s():
    config = shot.load_config(CONFIG)
    command = vllm_command(P6305_COMMAND[0], P6305_COMMAND[2], config.server)
    assert command == P6305_COMMAND
    if (P6305_LOGS / "metrics.json").exists():
        logged = json.loads((P6305_LOGS / "metrics.json").read_text())["server_command"]
        assert logged == P6305_COMMAND
    spike_server = spike.load_config(SPIKE_CONFIG)["server"]
    paths = ("VLLM_CACHE_ROOT", "HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
    mine = dict(config.server.env)
    assert {k: v for k, v in mine.items() if k not in paths} == {
        k: str(v) for k, v in spike_server["env"].items() if k not in paths
    }
    assert set(mine) == set(spike_server["env"])
    assert all(mine[k].startswith("/tmp/arc26-agent/") for k in paths)
    assert list(config.server.unset_env) == spike_server["unset_env"]
    assert config.server.boot_timeout_s == spike_server["boot_timeout_s"]
    wheels = spike.load_config(SPIKE_CONFIG)["wheels"]
    assert list(config.prep.imports) == wheels["engine_imports"]
    assert config.prep.check_timeout_s == wheels["engine_check_timeout_s"]
    assert config.prep.enabled and config.prep.preflight


def test_the_moved_code_is_the_spikes_without_change():
    moved = {
        engine_prep: (
            "Stop", "write_json", "run_command", "preflight", "probe_line", "jit_cache_state",
            "drop_orphan_jit_cache", "quiet_sitecustomize", "check_engine_imports",
            "prepare_engine", "parse_boot_log", "cache_config_info",
        ),
        recording: ("JsonlWriter", "token_stats", "call_health", "reasoning_location"),
        shot: ("NoRepl", "NoVerifier"),
    }  # fmt: skip
    for module, names in moved.items():
        for name in names:
            assert inspect.getsource(getattr(module, name)) == inspect.getsource(
                getattr(spike, name)
            ), name
    for name in ("PREFLIGHT", "JIT_CACHE_PROBE", "ENGINE_IMPORT_CHECK", "QUIET_SITECUSTOMIZE",
                 "BOOT_PATTERNS"):  # fmt: skip
        assert getattr(engine_prep, name) == getattr(spike, name), name


def test_the_shot_prepares_the_venv_as_p6_3_05_did(tmp_path, monkeypatch):
    calls = []

    def prepare(python, env, settings, out, scratch):
        calls.append(("prepare", python, dict(env), settings, out, scratch))
        return {**env, "PYTHONPATH": str(scratch / "engine_site")}, {
            "jit_cache": {"state": "orphan", "removed": ["flashinfer-jit-cache-sm89"]},
            "imports": {"modules": [["m", 0.1]] * 16},
        }

    def preflight(python, tp, out, env):
        calls.append(("preflight", python, tp, env["PYTHONPATH"]))
        return {"ok": True}

    monkeypatch.setattr(engine_prep, "prepare_engine", prepare)
    monkeypatch.setattr(engine_prep, "preflight", preflight)
    monkeypatch.setenv("PYTHONPATH", "/kaggle/lib/torch28")
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    config = shot.load_config(CONFIG)
    options = SimpleNamespace(engine_venv=tmp_path / "venv", log_dir=tmp_path)
    summary = {"engine": {}, "problems": []}
    env = shot.prepare_venv(options, config, summary)
    (_, python, base, settings, out, scratch), (_, py2, tp, path) = calls
    assert python == py2 == tmp_path / "venv" / "bin" / "python" and tp == 4
    assert base["PATH"].startswith(str(tmp_path / "venv" / "bin") + os.pathsep)
    assert "PYTHONPATH" not in base and "TRITON_PTXAS_PATH" not in base
    assert base["NCCL_DEBUG"] == "INFO" and base["PYTHONNOUSERSITE"] == "1"
    wheels = spike.load_config(SPIKE_CONFIG)["wheels"]
    assert settings == {key: wheels[key] for key in ("engine_imports", "engine_check_timeout_s")}
    assert scratch == Path("/tmp/arc26-agent") and path == "/tmp/arc26-agent/engine_site"
    assert env["PYTHONPATH"] == path and summary["engine"]["prep"]["modules"] == 16

    def broken(*args):
        raise engine_prep.Stop("engine import check failed at vllm.v1.worker.gpu_worker")

    monkeypatch.setattr(engine_prep, "prepare_engine", broken)
    summary = {"engine": {}, "problems": []}
    assert shot.prepare_venv(options, config, summary) is None
    assert "gpu_worker" in summary["problems"][0]


# --- the wave arithmetic ------------------------------------------------------------------------


def test_the_constants_trace_to_p6_3_05s_measurements():
    model = shot.load_config(CONFIG).waves
    assert model.rate_tokens_per_s == MEASURED["rate_tokens_per_s"]
    assert model.prefill_tokens_per_s == MEASURED["prefill_tokens_per_s"]
    assert model.kv_pool_tokens == MEASURED["kv_pool_tokens"]
    assert model.max_cap == MEASURED["max_cap"] and model.forcing_tokens == 6144
    assert (model.wave_tasks, model.plan_factor, model.min_cap) == (15, 0.85, 16_384)
    assert (model.margin_s, model.opening_tokens) == (300, 6771)
    if not (P6305_LOGS / "engine_calls.jsonl").exists():
        return
    calls = [json.loads(line) for line in (P6305_LOGS / "engine_calls.jsonl").open()]
    cut = [c for c in calls if c["completion_tokens"] == 94_208]
    forced = [c for c in calls if c["thinking"] is False and "/p0/" in c["attempt_id"]]
    assert len(cut) == len(forced) == 8
    assert round(min(94_208 / c["seconds"] for c in cut), 1) == 19.0
    # p: the forced replies' prompt tokens over the time from their common start to the last end.
    start = min(c["ended"] - c["seconds"] for c in forced)
    wall = max(c["ended"] for c in forced) - start
    assert sum(c["prompt_tokens"] for c in forced) == 783_425 and round(wall, 1) == 817.1
    assert round(783_425 / wall, 1) == MEASURED["prefill_tokens_per_s"]
    boot = json.loads((P6305_LOGS / "boot.json").read_text())
    assert boot["kv_cache_tokens_first"] == MEASURED["kv_pool_tokens"]
    # o: the longest opening bound of P6.3-05's 15 (one token per character plus 96 of slack).
    rows = [json.loads(line) for line in (P6305_LOGS / "direct_transcripts.jsonl").open()]
    bounds = [sum(len(m["content"]) + 16 for m in r["messages"][:2]) + 64 for r in rows]
    assert max(bounds) == 6771 and sum(bounds) == 57_627


def full(model, cap, n, o):
    """T(C, n, o) = C / (0.85 * 19) + n (o + C) / 958.8 + 6,144 / 19 + 300, written out."""
    return cap / (0.85 * 19) + n * (o + cap) / 958.8 + 6144 / 19 + 300


def test_a_full_wave_and_its_planned_length():
    model = shot.load_config(CONFIG).waves
    # 15 tasks at the full cap, the longest opening 6,771: 5,833.3 + 1,579.8 + 323.4 + 300 s.
    assert model.turn_seconds(94_208) == pytest.approx(5833.31, abs=0.01)
    assert model.forcing_seconds(94_208, 15, 6771) == pytest.approx(1903.14, abs=0.01)
    assert model.wave_seconds(94_208, 15, 6771) == pytest.approx(8036.45, abs=0.01)
    assert model.wave_seconds(94_208, 15, 6771) == pytest.approx(full(model, 94_208, 15, 6771))
    queue = [(f"t{k:02d}", 6771 if k == 3 else 4000) for k in range(20)]
    left = model.wave_seconds(94_208, 15, 6771)
    plan = shot.plan_wave(queue, left + 0.1, model, MEASURED["kv_pool_tokens"])
    assert plan.cap == 94_208 and len(plan.tasks) == 15 and plan.reason is None
    assert plan.opening == 6771 and plan.planned_s == pytest.approx(left)
    assert plan.kv_need == 14 * 4000 + 6771 + 15 * (94_208 + 6144) <= plan.kv_pool
    short = shot.plan_wave(queue, left - 1, model, 10**9)
    per_token = 1 / (0.85 * 19) + 15 / 958.8  # 0.077565 s
    fixed = 300 + 6144 / 19 + 15 * 6771 / 958.8
    assert short.cap == math.floor((left - 1 - fixed) / per_token) == 94_195
    arithmetic = shot.stage_arithmetic(shot.load_config(CONFIG))
    assert arithmetic["full_wave_s"] == 8036.5 and arithmetic["full_forcing_s"] == 1903.1


def test_a_reduced_wave_fits_the_time_left():
    model = shot.load_config(CONFIG).waves
    plan = shot.plan_wave([("a", 3000), ("b", 2000)], 4000, model, 10**9)
    per_token = 1 / (0.85 * 19) + 2 / 958.8
    fixed = 300 + 6144 / 19 + 2 * 3000 / 958.8
    assert plan.cap == math.floor((4000 - fixed) / per_token) == 52_657
    assert plan.planned_s <= 4000 < model.wave_seconds(plan.cap + 1, 2, 3000)
    assert plan.tasks == ("a", "b")
    # More tasks, more prefill: fifteen take a smaller cap in the same time.
    many = shot.plan_wave([(f"t{k}", 3000) for k in range(15)], 4000, model, 10**9)
    assert many.cap == 42_928 < plan.cap


def test_below_min_cap_no_wave_starts():
    model = shot.load_config(CONFIG).waves
    smallest = model.wave_seconds(16_384, 15, 6771)
    assert smallest == pytest.approx(2000.11, abs=0.01)
    assert shot.stage_arithmetic(shot.load_config(CONFIG))["min_shot_s"] == 2450.1
    queue = [(f"t{k:02d}", 6771) for k in range(15)]
    assert shot.plan_wave(queue, smallest, model, 10**9).cap == 16_384
    plan = shot.plan_wave(queue, smallest - 1, model, 10**9)
    assert plan.tasks == () and plan.cap < 16_384 and plan.reason.startswith("time")
    assert shot.plan_wave(queue, 200, model, 10**9).cap == 0
    assert shot.plan_wave([], 10_000, model, 10**9).reason == "no tasks left"


def test_how_slow_a_time_bound_wave_of_15_may_run():
    model = shot.load_config(CONFIG).waves
    for cap, least in ((94_208, 15.54), (16_384, 13.75)):
        rate = model.least_rate(cap, 15, 6771)
        assert round(rate, 2) == least
        # At that rate the turn and the forced reply's generation take exactly the planned time.
        planned = model.wave_seconds(cap, 15, 6771)
        assert (cap + 6144) / rate + model.forcing_seconds(cap, 15, 6771) - 6144 / 19 == (
            pytest.approx(planned)
        )


def test_a_kv_bound_wave_takes_fewer_tasks_and_leaves_the_rest_first():
    model = shot.load_config(CONFIG).waves
    queue = [(f"t{k:02d}", 5000) for k in range(15)]
    pool = 10 * (5000 + 94_208 + 6144) + 1000  # room for 10 full sequences
    plan = shot.plan_wave(queue, 10_000, model, pool)
    assert plan.cap == 94_208 and plan.tasks == tuple(f"t{k:02d}" for k in range(10))
    assert plan.dropped == tuple(f"t{k:02d}" for k in range(10, 15))
    assert plan.kv_need <= pool and plan.reason.startswith("kv")
    assert plan.planned_s == pytest.approx(model.wave_seconds(94_208, 10, 5000))
    # A time-bound cap is planned again for the smaller wave and grows.
    left = model.wave_seconds(60_000, 15, 5000)
    tight = shot.plan_wave(queue, left, model, 10 * (5000 + 70_000 + 6144))
    assert tight.cap > 60_000 and len(tight.tasks) <= 10
    assert tight.kv_need <= 10 * (5000 + 70_000 + 6144) and tight.planned_s <= left
    tiny = shot.plan_wave(queue, 10_000, model, 50_000)
    assert tiny.tasks == () and tiny.reason.startswith("kv")


def test_the_config_is_checked_strictly(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    edits = [
        lambda c: c.update(extra=1),
        lambda c: c["waves"].pop("margin_s"),
        lambda c: c["waves"].update(min_cap=200_000),
        lambda c: c["request"].update(context_tokens=65_536),
        lambda c: c["waves"].update(plan_factor=1.5),
        lambda c: c["request"].update(system_prompt="agent"),
        lambda c: c["selection"].update(max_confidence=2),
        lambda c: c["waves"].update(max_cap=130_000),
        lambda c: c["sanity"].update(reasoning_effort="max"),
    ]
    for edit in edits:
        changed = yaml.safe_load(yaml.safe_dump(raw))
        edit(changed)
        with pytest.raises(agent_run.ConfigError):
            shot.config_from_mapping(changed)
    shot.config_from_mapping(raw)


def test_every_config_value_has_a_comment():
    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        assert lines[k - 1].strip().startswith("#") or stripped.endswith(":"), line


# --- the merge ----------------------------------------------------------------------------------


A, B, C, D = [[1]], [[2]], [[3]], [[4]]


@pytest.mark.parametrize(
    ("n1", "n2", "s1", "s2", "want"),
    [
        (A, B, None, None, ((A, "nvarc"), (B, "nvarc"))),  # no S1
        (A, B, C, D, ((A, "nvarc"), (C, "shot"))),  # N1 real, S1 != N1
        (A, B, A, C, ((A, "nvarc"), (B, "nvarc"))),  # N1 real, S1 = N1
        (A, None, A, C, ((A, "nvarc"), (C, "shot"))),  # S1 = N1, N2 a placeholder: S2
        (A, None, A, None, ((A, "nvarc"), (None, "nvarc"))),  # S1 = N1, no S2
        (None, None, C, D, ((C, "shot"), (D, "shot"))),  # N1 a placeholder, S2
        (None, B, C, None, ((C, "shot"), (B, "nvarc"))),  # N1 a placeholder, no S2
        (None, None, None, None, ((None, "nvarc"), (None, "nvarc"))),
    ],
)
def test_every_merge_case(n1, n2, s1, s2, want):
    assert shot.merge_output(n1, n2, s1, s2) == want


def record(task, answers, stop="rounds"):
    return AttemptRecord(task, f"{task}/p0/0", 0, (), tuple(answers), 10, 10, 1, 1.0, stop)


def test_the_writer_merges_per_output_and_keeps_nvarcs_slots(tmp_path):
    out = tmp_path / "submission.json"
    nvarc = {"x": [(A, None), (None, None)], "y": [(None, B)]}
    writer = shot.ShotWriter(out, {"x": 2, "y": 1, "z": 1}, nvarc)
    assert writer.write()
    assert json.loads(out.read_text()) == {
        "x": [{"attempt_1": A, "attempt_2": [[0]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}],
        "y": [{"attempt_1": [[0]], "attempt_2": B}],  # NVARC's slots kept in place
        "z": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    # Answers in order; a repeated grid and an invalid one are skipped; the first two count.
    writer.add(record("x", [(0, A), (1, C), (1, C), (1, [[11]]), (0, D), (1, D), (1, B)]))
    written = json.loads(out.read_text())
    # Output 0: S1 = N1 and NVARC's attempt 2 is empty, so S2 takes it.
    assert written["x"] == [{"attempt_1": A, "attempt_2": D}, {"attempt_1": C, "attempt_2": D}]
    assert shot.shot_answers([record("x", [(0, A), (0, A), (0, B), (0, C)])], 0) == [A, B]
    assert writer.slots() == {"nvarc": 2, "shot": 3, "empty": 3}
    assert writer.nvarc_submission()["x"][0] == {"attempt_1": A, "attempt_2": [[0]]}


# --- end to end with the fake engine ------------------------------------------------------------


TASKS = {  # id: (NVARC's attempt 1 right?, its confidence, the shot's replies)
    "aaaa0001": "nvarc wrong, the shot's first answer right: (N1, S1) solves",
    "aaaa0002": "nvarc empty, the shot's second answer right: (S1, S2) solves",
    "aaaa0003": "nvarc right but unsure, the shot agrees: (N1, N2) stays",
    "aaaa0004": "nvarc sure: never shot",
    "aaaa0005": "cut at the cap, then the forcing turn's answer right",
}


def answer(grid, index=0):
    return f'```answer\n{{"test_index": {index}, "grid": {json.dumps(grid)}}}\n```\n'


def make_case(tmp_path, slow=(), degenerate=()):
    rng = np.random.default_rng(3)
    challenges, solutions, replies, nvarc, candidates = {}, {}, {}, {}, {}
    for task in TASKS:
        pairs = [rng.integers(0, 5, size=(3, 4)) for _ in range(2)]
        test = rng.integers(0, 5, size=(3, 4))
        challenges[task] = {
            "train": [{"input": g.tolist(), "output": np.flipud(g).tolist()} for g in pairs],
            "test": [{"input": test.tolist()}],
        }
        right, wrong = np.flipud(test).tolist(), [[9, 9], [9, 9]]
        solutions[task] = [right]
        # Confidence: three single grids (1/3) for the unsure ones, one grid 10 times for 0004.
        grids = [right, wrong, [[8]]] if task != "aaaa0004" else [right] * 10
        candidates[f"{task}_0.view0"] = [{"solution": g, "score_aug": [1.0]} for g in grids]
        thinking = {"reasoning": "the rows flip over " * 40}
        replies[task] = {
            "aaaa0001": [{"text": answer(right) + answer(wrong), **thinking}],
            "aaaa0002": [{"text": answer(wrong) + answer(right), **thinking}],
            "aaaa0003": [{"text": answer(right), **thinking}],
            "aaaa0004": [{"text": answer(wrong)}],
            "aaaa0005": [{"text": "", "reasoning": "the rows flip over and over " * 2000},
                         {"text": answer(right)}],
        }[task]  # fmt: skip
        if task in slow:
            replies[task][0]["delay_s"] = 60
        if task in degenerate:
            replies[task] = [{"text": answer(right), "reasoning": "0" * 200}]
        nvarc[task] = [
            {
                "aaaa0001": {"attempt_1": wrong, "attempt_2": [[7]]},
                "aaaa0002": {"attempt_1": [[0]], "attempt_2": [[0]]},
                "aaaa0003": {"attempt_1": right, "attempt_2": wrong},
                "aaaa0004": {"attempt_1": right, "attempt_2": wrong},
                "aaaa0005": {"attempt_1": wrong, "attempt_2": [[0]]},
            }[task]
        ]
    paths = {}
    for name, value in (("challenges", challenges), ("solutions", solutions),
                        ("replies", replies), ("nvarc", nvarc), ("candidates", candidates)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value), encoding="utf-8")  # fmt: skip
    return challenges, solutions, paths


def write_config(tmp_path, **sections):
    """The shot's config with waves that take no planned time, and quick engine checks."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["deadline"].update(final_margin_s=1)
    config["waves"].update(max_cap=4096, min_cap=256, prefill_tokens_per_s=1e9,
                           rate_tokens_per_s=1e6, margin_s=0, setup_s=0)  # fmt: skip
    config["server"].update(boot_timeout_s=30, stop_timeout_s=5, health_interval_s=0.05,
                            monitor_interval_s=0.1, hang_s=10)  # fmt: skip
    config["engine"].update(max_retries=0)
    for name, values in sections.items():
        config[name].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def arguments(paths, out, config, logs, *extra, mode="dev", budget_s=60.0):
    return [
        "--mode", mode, "--tasks", str(paths["challenges"]), "--out", str(out),
        "--config", str(config), "--log-dir", str(logs), "--port", str(free_port()),
        "--deadline", str(time.time() + budget_s), "--nvarc-submission", str(paths["nvarc"]),
        "--nvarc-candidates", str(paths["candidates"]), *extra,
    ]  # fmt: skip


def stub(challenges):
    return {task: {"test": entry["test"]} for task, entry in challenges.items()}


def solved(written, solutions):
    return sorted(
        t for t, want in solutions.items()
        if any(written[t][0][a] == want[0] for a in ("attempt_1", "attempt_2"))
    )  # fmt: skip


def stop_engine(logs):
    path = logs / "engine.pid"
    if path.exists():
        try:
            os.killpg(json.loads(path.read_text())["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_main(tmp_path, *extra, mode="dev", config=None, budget_s=60.0, **case):
    challenges, solutions, paths = make_case(tmp_path, **case)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    config = config or write_config(tmp_path)
    try:
        code = shot.main(arguments(paths, out, config, logs, *extra, mode=mode,
                                   budget_s=budget_s))  # fmt: skip
    finally:
        stop_engine(logs)
    written = json.loads(out.read_text()) if out.exists() else None
    summary = json.loads((logs / "run.json").read_text())
    plan = json.loads((logs / "shot_plan.json").read_text())
    return code, written, summary, plan, challenges, solutions, paths


def test_a_dev_run_shoots_the_unsure_tasks_merges_and_scores(tmp_path):
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json"), "--solutions",
        str(tmp_path / "solutions.json"),
    )  # fmt: skip
    assert code == 0 and summary["problems"] == [], summary["problems"]
    assert validate(written, stub(challenges)) == []
    assert [t for t, _ in plan["queue"]] == ["aaaa0001", "aaaa0002", "aaaa0003", "aaaa0005"]
    assert len(plan["waves"]) == 1 and plan["waves"][0]["cap"] == 4096
    assert plan["stopped"] == "no tasks left" and plan["sanity"]["ok"]
    assert solved(written, solutions) == sorted(TASKS)
    assert written["aaaa0003"][0] == json.loads(paths["nvarc"].read_text())["aaaa0003"][0]
    metrics = json.loads((tmp_path / "logs" / "shot_metrics.json").read_text())
    assert metrics["hybrid"]["score"] == 5 and metrics["nvarc"]["score"] == 2
    assert metrics["gained"] == ["aaaa0001", "aaaa0002", "aaaa0005"] and metrics["lost"] == []
    assert metrics["shot_solved"] == ["aaaa0001", "aaaa0002", "aaaa0003", "aaaa0005"]
    assert metrics["nvarc_solved_on_shot_tasks"] == ["aaaa0003"]
    assert summary["score"]["score"] == 5.0
    rows = [json.loads(line) for line in (tmp_path / "logs" / "shot_attempts.jsonl").open()]
    forced = next(r for r in rows if r["task_id"] == "aaaa0005")
    assert forced["finish"] == ["length", "stop"] and forced["forcing_turns"] == 1
    for name in ("engine_calls.jsonl", "nvarc_submission.json", "engine.log", "summary.md",
                 "shot_transcripts.jsonl", "boot.json", "timing-engine.jsonl"):  # fmt: skip
        assert (tmp_path / "logs" / name).exists(), name
    assert json.loads((tmp_path / "logs" / "boot.json").read_text())["kv_source"] == "config"


def test_a_call_cut_at_the_generation_stop_keeps_nvarcs_attempts(tmp_path):
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json"), budget_s=8,
        slow=("aaaa0001", "aaaa0002"),
    )  # fmt: skip
    nvarc = json.loads(paths["nvarc"].read_text())
    assert code == 0 and validate(written, stub(challenges)) == []
    assert written["aaaa0001"] == nvarc["aaaa0001"] and written["aaaa0002"][0] == {
        "attempt_1": [[0]], "attempt_2": [[0]]
    }  # fmt: skip
    wave = plan["waves"][0]
    assert wave["deadline"] == 2 and wave["used"] == 2 and wave["finished"] == 4
    assert solved(written, solutions) == ["aaaa0003", "aaaa0004", "aaaa0005"]


def test_degenerate_replies_keep_nvarc_and_end_the_shot(tmp_path):
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json"),
        degenerate=("aaaa0001", "aaaa0002", "aaaa0003"),
    )  # fmt: skip
    nvarc = json.loads(paths["nvarc"].read_text())
    assert code == 0 and written["aaaa0001"] == nvarc["aaaa0001"]
    assert plan["waves"][0]["degenerate"] == 3 and "degenerate" in plan["stopped"]


def test_a_failed_boot_leaves_nvarcs_file(tmp_path):
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", "unused.json", "--fake-fail-boot"
    )
    assert code == 0 and written == json.loads(paths["nvarc"].read_text())
    assert len([p for p in summary["problems"] if p.startswith("engine start")]) == 1
    assert plan["waves"] == [] and "not up" in plan["stopped"]


def test_an_engine_death_mid_wave_leaves_nvarcs_file(tmp_path):
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json"), "--fake-crash-after", "2"
    )  # the sanity request passes, the wave's first call kills the engine
    assert code == 0 and validate(written, stub(challenges)) == []
    assert written == json.loads(paths["nvarc"].read_text())
    assert plan["waves"][0]["used"] == 0 and "engine" in plan["stopped"]


def test_sigterm_stops_the_engine_and_leaves_a_valid_file(tmp_path):
    challenges, _, paths = make_case(tmp_path, slow=tuple(TASKS))
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]))  # fmt: skip
    process = subprocess.Popen(
        [sys.executable, "-m", "arc26.agent.shot", *argv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_path,
    )  # fmt: skip
    try:
        end = time.monotonic() + 60
        plan = logs / "shot_plan.json"
        while time.monotonic() < end:
            if plan.exists() and json.loads(plan.read_text()).get("waves"):
                break
            time.sleep(0.05)
        engine = json.loads((logs / "engine.pid").read_text())["pid"]
        time.sleep(0.5)  # the wave's calls wait on the slow engine
        process.send_signal(signal.SIGTERM)
        assert process.wait(30) == 0
        with pytest.raises(ProcessLookupError):
            os.killpg(engine, 0)
        summary = json.loads((logs / "run.json").read_text())
        assert "SystemExit: 143" in summary["problems"] and summary["submission"]["valid"]
        assert json.loads(out.read_text()) == json.loads(paths["nvarc"].read_text())
    finally:
        if process.poll() is None:
            process.kill()
        stop_engine(logs)


def test_competition_mode_never_opens_a_solutions_file(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a solutions file was opened")

    monkeypatch.setattr("arc26.evaluate.load_solutions", refuse)
    monkeypatch.setattr(agent_run, "score_dev", refuse)
    monkeypatch.setattr(shot, "shot_dev_metrics", refuse)
    real_open = open

    def guarded(file, *args, **kwargs):
        assert "solution" not in str(file), file
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded)
    code, written, summary, plan, challenges, solutions, _ = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json"), mode="competition"
    )
    assert code == 0 and summary["problems"] == [] and len(plan["waves"]) == 1
    assert solved(written, solutions) == sorted(TASKS)
    assert not (tmp_path / "logs" / "shot_metrics.json").exists()
    assert not (tmp_path / "logs" / "shot_transcripts.jsonl").exists()
    monkeypatch.undo()
    with pytest.raises(PermissionError):
        shot.shot_dev_metrics("competition", tmp_path / "solutions.json", None, "x")


def test_the_arguments_and_exit_codes_are_arc26_agent_runs(tmp_path):
    def options(parser):
        return [(a.dest, a.option_strings, a.required, a.type, a.default, a.choices)
                for a in parser._actions]  # fmt: skip

    assert options(shot.build_parser()) == options(agent_run.build_parser())
    assert shot.build_parser().prog == "python -m arc26.agent.shot"
    _, _, paths = make_case(tmp_path)
    out, logs, config = tmp_path / "submission.json", tmp_path / "logs", write_config(tmp_path)
    base = arguments(paths, out, config, logs, "--fake-engine", str(paths["replies"]))
    assert shot.main([*base, "--mode", "competition", "--solutions", "x.json"]) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text(CONFIG.read_text().replace("wave_tasks: 15", "wave_task: 15"))
    assert shot.main(arguments(paths, out, bad, logs, "--fake-engine", "x")) == 2
    assert shot.main(["--mode", "dev"]) == 2
    assert not out.exists()  # a refusal writes nothing
    paths["challenges"].write_text("[]")
    out.write_text('{"kept": true}')
    assert shot.main(arguments(paths, out, config, logs, "--fake-engine", "x")) == 1
    assert json.loads(out.read_text()) == {"kept": True}


def test_too_little_time_starts_no_engine(tmp_path):
    config = write_config(tmp_path, waves={"setup_s": 3600})
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json"), config=config
    )
    assert code == 0 and written == json.loads(paths["nvarc"].read_text())
    assert "no engine started" in plan["stopped"] and "boot_s" not in summary["engine"]
    assert not (tmp_path / "logs" / "engine.log").exists()


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_an_unreadable_candidates_file_shoots_only_nvarcs_empty_tasks(tmp_path, content):
    challenges, solutions, paths = make_case(tmp_path)
    paths["candidates"].write_text(content)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]))  # fmt: skip
    try:
        assert shot.main(argv) == 0
    finally:
        stop_engine(logs)
    summary = json.loads((logs / "run.json").read_text())
    plan = json.loads((logs / "shot_plan.json").read_text())
    assert summary["selection"]["source"] == plan["confidence_source"] == "submission"
    assert "only tasks with an output NVARC left empty" in summary["selection"]["fallback"]
    assert plan["fallback"] == summary["selection"]["fallback"]
    assert [t for t, _ in plan["queue"]] == ["aaaa0002"]  # NVARC's placeholder
    assert plan["unknown"] == ["aaaa0001", "aaaa0003", "aaaa0004", "aaaa0005"]
    written = json.loads(out.read_text())
    assert validate(written, stub(challenges)) == []
    assert written["aaaa0002"][0]["attempt_2"] == solutions["aaaa0002"][0]


# --- fix round 1 --------------------------------------------------------------------------------


def test_the_sanity_request_never_waits_past_the_stop():
    class Silent:
        calls: dict = {}

        async def complete(self, *args, **kwargs):
            await asyncio.sleep(3600)  # healthy on /health, never finishes a chat call

    config = shot.load_config(CONFIG)
    began = time.monotonic()
    result = asyncio.run(shot.sanity_check(Silent(), config, began + 0.3))
    assert not result["ok"] and result["error"] == "timed out"
    assert time.monotonic() - began < 2
    passed = asyncio.run(shot.sanity_check(Silent(), config, time.monotonic() - 1))
    assert not passed["ok"] and "stop passed" in passed["error"]


def test_an_engine_that_never_answers_ends_the_shot_at_the_stop(tmp_path):
    began = time.monotonic()
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json"), "--fake-delay-s", "600",
        budget_s=5,
    )  # fmt: skip
    assert code == 0 and written == json.loads(paths["nvarc"].read_text())
    assert plan["sanity"]["ok"] is False and plan["sanity"]["error"] == "timed out"
    assert "sanity" in plan["stopped"] and plan["waves"] == []
    assert time.monotonic() - began < 5 + 15  # the stop, then the engine's stop (5 s) at most


def test_a_nan_score_is_least_sure_and_the_minimum_ignores_output_order():
    nan = [
        {"solution": [[1]], "score_aug": [float("nan")]},
        {"solution": [[2]], "score_aug": [1.0]},
    ]
    unsure = [{"solution": [[g]], "score_aug": [1.0]} for g in (1, 2, 3)]
    nvarc = {"t": [([[1]], None), ([[1]], None)]}
    one = shot.shot_confidence({"t": 2}, {"t_0.v": nan, "t_1.v": unsure}, nvarc)
    two = shot.shot_confidence({"t": 2}, {"t_0.v": unsure, "t_1.v": nan}, nvarc)
    assert one == two == {"t": 0.0}
    inf = [{"solution": [[1]], "score_aug": [float("inf")]}]
    assert shot.shot_confidence({"t": 1}, {"t_0.v": inf}, nvarc) == {"t": 0.0}


def test_a_failing_confidence_takes_the_no_candidates_fallback(tmp_path, monkeypatch):
    def broken(*args):
        raise ValueError("an odd candidates file")

    monkeypatch.setattr(shot, "shot_confidence", broken)
    code, written, summary, plan, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", str(tmp_path / "replies.json")
    )
    assert code == 0 and validate(written, stub(challenges)) == []
    assert summary["selection"]["source"] == plan["confidence_source"] == "submission"
    assert "an odd candidates file" in summary["selection"]["fallback"]
    assert [t for t, _ in plan["queue"]] == ["aaaa0002"]
    assert written["aaaa0002"][0]["attempt_2"] == solutions["aaaa0002"][0]
