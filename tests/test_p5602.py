"""Run P5.6-02 (task P5.6): the 8-stream route on revision 21's code, P5.6-01's fault removed.

Checks that configs/P5.6-02.yaml is revision 21 (configs/P5.2-01.yaml) but its lineage, P5.6-01's
engine levers, the fraction 0.95, 8 server slots for 7 streams, Mamba 24 by the fork's lazy rule,
graphs to 7, the sweep, and the ladder; that the ladder's first rung is revision 21's flags and
variables exactly; the plans at the estimated pool; the sweep at 1 2 4 6 7 8 and the solve on the
fake server; the twin; and byte-identical builds. No network, no GPU.
"""

import importlib.util
import json
import socket
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("fn_probe_p5602_under_test", SCRIPTS / "fn_probe.py")
nb = _load("build_fn_notebooks_p5602_under_test", SCRIPTS / "build_fn_notebooks.py")
OPENINGS = [6_896] + [1_782] * 239
FLAGS = {"--mamba-radix-cache-strategy": "extra_buffer_lazy", "--mem-fraction-static": 0.95,
         "--json-model-override-args": {"language_model_only": True},
         "--cuda-graph-backend-prefill": "disabled", "--max-running-requests": 8,
         "--max-mamba-cache-size": 24, "--cuda-graph-max-bs-decode": 7,
         "--cuda-graph-bs-decode": [1, 2, 4, 6, 7], "--chunked-prefill-size": 1024,
         "--prefill-max-requests": 4, "--enable-mixed-chunk": False}
ENV = {"SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1", "SGLANG_VLM_CACHE_SIZE_MB": "0"}


def read(name):
    return (REPO / name).read_bytes()


def config(name="configs/P5.6-02.yaml"):
    return nb.resolve_extends(read, name)


def revision_21():
    return yaml.safe_load(read("configs/P5.2-01.yaml"))


def _diff(a, b, path=""):
    out = set()
    for key in set(a) | set(b):
        if a.get(key) == b.get(key):
            continue
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            out |= _diff(a[key], b[key], f"{path}{key}.")
        else:
            out.add(f"{path}{key}")
    return out


def test_p5602_is_revision_22_but_its_lineage_engine_levers_streams_ladder_and_sweep():
    ours, base = config(), revision_21()
    # Revision 21 already has 0.95; the rest of FLAGS differs.
    same = {"--mem-fraction-static"}
    assert _diff(base, ours) == {
        "run_id", "kernel.id", "kernel.title", "kernel.code_dataset", "inputs.code.path",
        "inputs.code.hint", *(f"engine.flags.{f}" for f in FLAGS if f not in same),
        *(f"engine.env.{k}" for k in ENV), "engine.boot_ladder", "solve.max_streams",
        # Revision 22's stress check falls back to the smaller prefill; P5.6-02 runs it already.
        "stress.fallback_flags", "smoke.stress.fallback_flags", "sweep", "smoke.sweep"}
    # Revision 22's health probe and stress check, inherited.
    assert ours["health"]["probe_timeout_s"] == 30 and ours["health"]["kill_probes"] == 3
    assert ours["stress"]["modes"] == ["development", "competition"]
    assert ours["stress"]["fallback_flags"] is None
    assert {f: ours["engine"]["flags"][f] for f in FLAGS} == FLAGS
    assert {k: ours["engine"]["env"][k] for k in ENV} == ENV
    assert ours["kernel"]["id"] == "alyahya737/arc26-fn-8b"
    assert ours["kernel"]["code_dataset"] == "alyahya737/arc26-code-fn-8b"
    assert ours["kernel"]["accelerator"] == "NvidiaL4" and ours["kernel"]["timeout_s"] == 7500
    assert ours["kernel"]["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
    # Revision 21's KV4, cap, floor, and the development run on the same 15 tasks.
    assert ours["engine"]["env"]["ARC26_KV4"] == "1" and len(ours["patches"]["items"]) == 14
    assert ours["solve"]["cap"] == 49_152 and ours["planner"]["min_thinking_tokens"] == 29_000
    assert ours["solve"]["dev_arms"] == ["tool"]
    assert ours["solve"]["task_ids"] == base["solve"]["task_ids"]
    assert len(base["solve"]["task_ids"]) == 15
    flags = ours["engine"]["flags"]
    # The N+1 rule: 8 server slots for 7 streams.
    assert flags["--max-running-requests"] == ours["solve"]["max_streams"] + 1 == 8
    # The fork's lazy rule with the skip lock: 3 slots per server slot, the cap 24 // 3 = 8.
    assert flags["--max-mamba-cache-size"] == 3 * 8
    assert flags["--max-mamba-cache-size"] // 3 == flags["--max-running-requests"]
    # Graphs to the client's 7 streams.
    assert flags["--cuda-graph-bs-decode"][-1] == flags["--cuda-graph-max-bs-decode"] == 7
    assert ours["sweep"]["batches"] == [1, 2, 4, 6, 7, 8]


def test_the_ladder_is_revision_21_then_revision_16_then_bf16_eager():
    ours, base = config(), revision_21()
    rungs = probe.boot_rungs(ours)
    assert len(rungs) == 3
    flags = {k: v for k, v in {**ours["engine"]["flags"], **rungs[0]["flags"]}.items()
             if v is not None}
    assert flags == {k: v for k, v in base["engine"]["flags"].items() if v is not None}
    env = {k: v for k, v in {**ours["engine"]["env"], **rungs[0]["env"]}.items() if v is not None}
    assert env == base["engine"]["env"]
    assert rungs[1] == {"flags": {}, "env": {"ARC26_KV4": "0"}}
    assert rungs[2]["flags"] == base["engine"]["boot_retry_flags"]


def test_the_planner_gives_7_streams_at_the_estimated_pool():
    ours = config()
    spec, planner = ours["solve"], ours["planner"]

    def plan(pool, limit=8):
        return probe.plan_budget(pool, limit, 36.8, 39_000.0, 240, OPENINGS, 1, spec, planner)

    # P5.6-01's measured 377,856 at 0.96 and 29 slots, less 0.01 of the fraction (0.216 GiB per
    # GPU at 4,224 B per KV4 token), plus 5 Mamba slots of 3,482 (estimate).
    kv4 = 377_856 - int(0.216 * 2**30 / 4_224) + 5 * 3_482
    assert kv4 == 340_359
    assert (plan(kv4)["streams"], plan(kv4)["context"], plan(kv4)["thinking"]) == (7, 48_622,
                                                                                   35_454)
    # The orchestrator's 322k (the fraction only): still 7, at 46,135.
    assert (plan(322_949)["streams"], plan(322_949)["context"]) == (7, 46_135)
    # 7 streams from 7 x 42,168; the server's 8 never gives an 8th client stream.
    assert (plan(295_176)["streams"], plan(295_175)["streams"]) == (7, 6)
    assert plan(10**7)["streams"] == 7
    # A task's cap: the context binds under 49,152 at the estimate.
    assert probe.task_cap(plan(kv4)["context"], 6_896, spec) == 48_622 - 6_896 - 128
    for pool in (kv4, 322_949, 240_256):
        assert plan(pool)["streams"] * plan(pool)["context"] <= pool


def test_rung_1_plans_revision_21s_5_streams():
    """Rung 1 changes the server's flags, not solve.max_streams (7), so the planner's limit is the
    server's 6; revision 21's pool (version 17: 240,256) holds 5 at the floor, since 6 need
    6 x 42,168 = 253,008. The 6th slot stays the latch's spare, as in revision 21."""
    ours = config()
    spec, planner = ours["solve"], ours["planner"]
    plan = probe.plan_budget(240_256, 6, 36.8, 39_000.0, 240, OPENINGS, 1, spec, planner)
    assert plan["streams"] == 5 and 6 * 42_168 > 240_256


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _smoke(tmp_path, edit=None):
    cfg = config()
    cfg["smoke"]["engine"]["port"] = _free_port()
    if edit:
        edit(cfg["smoke"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    root = tmp_path / "root"
    code = probe.main(["--config", str(path), "--smoke", "--root", str(root), "--commit", "x"])
    return code, root / "kaggle/working/run_logs/fn"


def test_the_sweep_runs_before_the_solve_on_the_fake_server(tmp_path):
    def edit(smoke):
        # The GPU run's server limit and streams on the fake.
        smoke["engine"]["flags"]["--max-running-requests"] = 8
        smoke["solve"]["max_streams"] = 7
        smoke["sweep"]["batches"] = [1, 2, 4, 6, 7, 8]

    code, out = _smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["sweep"]["status"] == "ok" and state["solve"]["status"] == "ok"
    assert list(state).index("sweep") < list(state).index("solve")
    sweep = json.loads((out / "sweep.json").read_text())
    assert [r["batch"] for r in sweep["rows"]] == [1, 2, 4, 6, 7, 8]
    for row in sweep["rows"]:
        assert len(row["requests"]) == row["batch"] and not row["errors"]
    assert json.loads((out / "boot.json").read_text())["max_running_requests"] == 8


def test_the_twin_is_p5601s_twin_on_p5602():
    twin = config("configs/P5.6-02-cpu.yaml")
    assert twin["kernel"]["id"] == "alyahya737/arc26-fn-8b-cpu"
    assert twin["kernel"]["accelerator"] is None
    assert twin["kernel"]["code_dataset"] == "alyahya737/arc26-code-fn-8b"
    assert twin["engine"]["script"] == "fn_fake_sglang.py" and twin["engine"]["boot_ladder"] is None
    assert twin["import_check"]["enabled"] and len(twin["patches"]["items"]) == 14
    # The argument check resolves this run's flags (the fake's own dropped).
    assert twin["engine"]["flags"]["--mamba-radix-cache-strategy"] == "extra_buffer_lazy"
    assert twin["engine"]["flags"]["--max-running-requests"] == 8
    assert twin["engine"]["flags"]["--max-mamba-cache-size"] == 24
    assert twin["solve"]["max_streams"] == 7


def test_both_notebooks_build_byte_identically(tmp_path):
    for name in ("configs/P5.6-02.yaml", "configs/P5.6-02-cpu.yaml"):
        first = nb.build(REPO / name, tmp_path / "a", REPO, allow_dirty=True)
        second = nb.build(REPO / name, tmp_path / "b", REPO, allow_dirty=True)
        files_a = {p.relative_to(first): p.read_bytes() for p in first.rglob("*")
                   if p.is_file() and p.name != "MANIFEST.md"}
        files_b = {p.relative_to(second): p.read_bytes() for p in second.rglob("*")
                   if p.is_file() and p.name != "MANIFEST.md"}
        assert files_a == files_b
        meta = json.loads((first / "kernel-metadata.json").read_text())
        assert "alyahya737/arc26-code-fn-8b" in meta["dataset_sources"]
        assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-2"]


# -- P5.6-01's restart 2: the health probe tied to the engine instance ------------------------


class _Proc:
    def __init__(self):
        self.pid = 1

    def poll(self):
        return None


def _inner(tmp_path, monkeypatch, answers):
    """An inner state on a fake process whose /health answers come from ``answers``."""
    cfg = config()
    run = probe.Run(config=cfg, root=tmp_path, out=tmp_path, start=probe._now(), state={})
    inner = probe.Inner(run=run, base="http://127.0.0.1:9", tok=None)
    inner.proc = _Proc()
    inner.generation = 1
    calls = {"probes": [], "kills": 0, "launches": 0}

    def http_json(base, method, path, body=None, timeout=0):
        calls["probes"].append(timeout)
        answer = answers.pop(0) if answers else 200
        if answer is None:
            raise OSError("timed out")
        return answer, {}

    def kill(inner_):
        calls["kills"] += 1

    def launch(inner_, overrides=None):
        calls["launches"] += 1
        inner_.generation += 1
        return 1.0

    monkeypatch.setattr(probe, "http_json", http_json)
    monkeypatch.setattr(probe, "kill_server", kill)
    monkeypatch.setattr(probe, "launch_server", launch)
    monkeypatch.setattr(probe, "read_limits", lambda inner_: {})
    return inner, calls


def test_a_stale_failure_never_probes_or_kills_a_new_engine(tmp_path, monkeypatch):
    inner, calls = _inner(tmp_path, monkeypatch, [None, None, None])
    # A thread saw instance 1 fail; instance 2 is serving now.
    inner.generation = 2
    assert probe.restart_engine(inner, "engine failed during a task", seen=1) is True
    assert calls == {"probes": [], "kills": 0, "launches": 0}
    assert inner.engine_ready.is_set()


def test_a_stale_health_watch_exits_without_probing(tmp_path, monkeypatch):
    inner, calls = _inner(tmp_path, monkeypatch, [None] * 20)
    inner.run.config["health"]["interval_s"] = 0.01
    watch = probe.HealthWatch(inner)
    inner.generation += 1
    watch.start()
    watch.join(timeout=5)
    assert not watch.is_alive() and calls["probes"] == [] and calls["kills"] == 0


def test_a_slow_but_alive_engine_is_not_killed(tmp_path, monkeypatch):
    # Two probes time out (a stall), the third answers.
    inner, calls = _inner(tmp_path, monkeypatch, [None, None, 200])
    assert probe.restart_engine(inner, "engine failed during a task", seen=1) is True
    assert calls["kills"] == 0 and calls["launches"] == 0
    # Each probe waits health.probe_timeout_s (30 s), three at most.
    assert calls["probes"] == [30.0, 30.0, 30.0]


def test_three_failed_probes_restart_and_the_new_instance_gets_its_own_watch(tmp_path,
                                                                             monkeypatch):
    inner, calls = _inner(tmp_path, monkeypatch, [None, None, None])
    inner.run.config["health"]["interval_s"] = 60
    old = probe.HealthWatch(inner)
    inner.watch = old
    assert probe.restart_engine(inner, "engine failed during a task", seen=1) is True
    assert calls["kills"] == 1 and calls["launches"] == 1 and inner.generation == 2
    assert old.stop.is_set() and inner.watch is not old
    assert inner.watch.generation == 2
    inner.watch.stop.set()
    rows = [json.loads(line) for line in (tmp_path / "restarts.jsonl").read_text().splitlines()]
    assert rows[-1]["generation"] == 2


def test_free_device_min_reads_the_engines_lines():
    text = ("x (free device mem: 0.64 GiB). y\n"
            "Triton kernel 'k' device-loaded after serving started (free device mem: 0.22 GiB).")
    pattern = config()["engine"]["free_pattern"]
    assert probe.free_device_min(text, pattern) == {"min_gib": 0.22, "lines": 2}
    assert probe.free_device_min("", pattern) == {"min_gib": None, "lines": 0}


# -- the boot stress check ----------------------------------------------------------------------


def test_the_stress_check_runs_before_the_sweep_and_the_solve(tmp_path):
    code, out = _smoke(tmp_path)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["stress"]["status"] == "ok"
    order = list(state)
    assert order.index("template_check") < order.index("stress") < order.index("sweep") < (
        order.index("solve"))
    stress = json.loads((out / "stress.json").read_text())
    smoke = probe.apply_smoke(config())
    assert stress["streams"] == smoke["solve"]["max_streams"] and not stress["errors"]
    assert all(r["completion_tokens"] == 32 for r in stress["requests"])
    # The largest openings of the plan's tasks, each a distinct prompt.
    assert stress["prompt_tokens"] == sorted(stress["prompt_tokens"], reverse=True)
    assert "free_device" in state["monitor"]


def test_a_stress_failure_moves_the_boot_one_rung(tmp_path):
    def edit(smoke):
        # The first boot dies with more than 3 requests in flight (the stress check's 4 at once,
        # not the template check's one); rung 1 removes the flag.
        smoke["engine"]["flags"]["--die-over-inflight"] = 3
        smoke["engine"]["boot_ladder"] = [{"flags": {"--die-over-inflight": None}}]
        smoke["mode"] = "competition"

    code, out = _smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert state["stress"]["status"] == "failed"
    assert state["boot_stress_retry"]["status"] == "ok"
    assert state["solve"]["status"] == "ok" and state["solve"]["submission_filled"] == 6
    rows = [json.loads(line) for line in (out / "restarts.jsonl").read_text().splitlines()]
    assert [r.get("rung") for r in rows] == [1]
    assert rows[0]["reason"] == "the boot stress check failed"
    assert json.loads((out / "stress.json").read_text())["engine_alive"] is False
    assert code == 0


def test_both_new_flags_are_fields_of_the_forks_server_args():
    """The fork's parser exits on an unknown flag (arg_check reads it as refused, exit 1); this
    reads the local clone's ServerArgs when it is at hand."""
    import pytest

    source = Path("/tmp/fnsg/penny/python/sglang/srt/server_args.py")
    if not source.is_file():
        pytest.skip("no local clone of the fork")
    text = source.read_text()
    assert "    prefill_max_requests: A[" in text and "    enable_mixed_chunk: A[" in text
    assert "    chunked_prefill_size: A[" in text
