"""Run P5.6-01 (task P5.6): configuration 8B on Kaggle's 4 x L4, beside P5.2-01 revision 20.

Checks that configs/P5.6-01.yaml is revision 20 but 8B's flags and variables, its own notebook and
code dataset, the boot ladder, the streams, and the sweep; that the ladder's first rung is
revision 20's flags and variables exactly and its second turns KV4 off; that a rung removes a
variable; that the planner gives 8 streams at the estimated pool; the batch sweep and the ladder
on the fake server; the twin's config; and byte-identical builds. No network, no GPU.
"""

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("fn_probe_p5601_under_test", SCRIPTS / "fn_probe.py")
nb = _load("build_fn_notebooks_p5601_under_test", SCRIPTS / "build_fn_notebooks.py")
OPENINGS = [6_896] + [1_782] * 239
EIGHT_B_FLAGS = {
    "--mamba-radix-cache-strategy": "extra_buffer_lazy", "--mem-fraction-static": 0.96,
    "--json-model-override-args": {"language_model_only": True},
    "--cuda-graph-backend-prefill": "disabled", "--max-running-requests": 9,
    "--max-mamba-cache-size": 29, "--cuda-graph-max-bs-decode": 8,
    "--cuda-graph-bs-decode": [1, 2, 4, 6, 8]}
EIGHT_B_ENV = {"SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1", "SGLANG_VLM_CACHE_SIZE_MB": "0"}


def read(name):
    return (REPO / name).read_bytes()


# P5.6-01's Code-commit: configs/P5.2-01.yaml has moved on to revision 21 (P5.6-02 extends it),
# so the run's own config and its base are read as they were run.
P5601_COMMIT = "c2fd3788248845845ae776289546e7695955ddf8"


def read_at_run(name):
    import subprocess

    done = subprocess.run(["git", "show", f"{P5601_COMMIT}:{name}"], cwd=REPO,
                          capture_output=True, check=False)
    if done.returncode != 0:
        pytest.skip(f"P5.6-01's commit is not in this clone ({name})")
    return done.stdout


def config(name="configs/P5.6-01.yaml"):
    return nb.resolve_extends(read_at_run, name)


def revision_20():
    return yaml.safe_load(read_at_run("configs/P5.2-01.yaml"))


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


def test_8b_is_revision_20_but_its_flags_variables_ids_ladder_streams_and_sweep():
    ours, base = config(), revision_20()
    assert _diff(base, ours) == {
        "run_id", "kernel.id", "kernel.title", "kernel.code_dataset", "inputs.code.path",
        "inputs.code.hint",
        # Revision 20 already has 29 Mamba slots (5 x 5 + 4); 8B's 29 is 3 x 9 + 2.
        *(f"engine.flags.{f}" for f in EIGHT_B_FLAGS if f != "--max-mamba-cache-size"),
        *(f"engine.env.{k}" for k in EIGHT_B_ENV), "engine.boot_ladder", "solve.max_streams",
        "sweep", "smoke.sweep"}
    assert {f: ours["engine"]["flags"][f] for f in EIGHT_B_FLAGS} == EIGHT_B_FLAGS
    assert {k: ours["engine"]["env"][k] for k in EIGHT_B_ENV} == EIGHT_B_ENV
    assert ours["kernel"]["id"] == "alyahya737/arc26-fn-8b"
    assert ours["kernel"]["code_dataset"] == "alyahya737/arc26-code-fn-8b"
    assert ours["kernel"]["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
    assert ours["kernel"]["accelerator"] == "NvidiaL4" and ours["kernel"]["timeout_s"] == 7500
    # KV4 on with revision 20's fourteen edits, the cap, the floor, the development run.
    assert ours["engine"]["env"]["ARC26_KV4"] == "1" and len(ours["patches"]["items"]) == 14
    assert ours["solve"]["cap"] == 49_152 and ours["planner"]["min_thinking_tokens"] == 29_000
    assert ours["solve"]["dev_arms"] == ["tool"] and len(ours["solve"]["task_ids"]) == 15
    # The server's limit is the streams + 1 (the latch); Mamba 3 x 9 + 2.
    assert ours["engine"]["flags"]["--max-running-requests"] == ours["solve"]["max_streams"] + 1
    assert ours["engine"]["flags"]["--max-mamba-cache-size"] == 3 * 9 + 2


def test_the_ladder_is_revision_20_then_revision_16_then_bf16_eager():
    ours, base = config(), revision_20()
    rungs = probe.boot_rungs(ours)
    assert len(rungs) == 3
    flags = {k: v for k, v in {**ours["engine"]["flags"], **rungs[0]["flags"]}.items()
             if v is not None}
    assert flags == {k: v for k, v in base["engine"]["flags"].items() if v is not None}
    env = {k: v for k, v in {**ours["engine"]["env"], **rungs[0]["env"]}.items() if v is not None}
    assert env == base["engine"]["env"]
    assert rungs[1] == {"flags": {}, "env": {"ARC26_KV4": "0"}}
    assert rungs[2]["flags"] == base["engine"]["boot_retry_flags"]


def test_a_rung_removes_a_variable_from_the_boot_environment(monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK", "1")
    run = probe.Run(config=config(), root=Path("/"), out=Path("/"), start=0.0,
                    state={"boot_env": {"SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": None,
                                        "ARC26_KV4": "0"}})
    env = probe.boot_environment(run)
    assert "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK" not in env and env["ARC26_KV4"] == "0"


def test_the_planner_gives_8_streams_at_the_estimated_pool():
    ours = config()
    spec, planner = ours["solve"], ours["planner"]
    plan = probe.plan_budget(377_408, 9, 30.0, 39_000.0, 240, OPENINGS, 1, spec, planner)
    assert (plan["streams"], plan["context"], plan["thinking"]) == (8, 47_176, 34_008)
    assert probe.task_cap(plan["context"], 1_782, spec) == 47_176 - 1_782 - 128
    # The server's 9 never gives a 9th client stream; 8 need 8 x 42,168.
    assert probe.plan_budget(10**6, 9, 30.0, 39_000.0, 240, OPENINGS, 1, spec, planner)[
        "streams"] == 8
    assert probe.plan_budget(337_343, 9, 30.0, 39_000.0, 240, OPENINGS, 1, spec, planner)[
        "streams"] == 7
    # Revision 20's rung: 5 streams at most, whatever the pool.
    assert probe.plan_budget(377_408, 5, 30.0, 39_000.0, 240, OPENINGS, 1, spec, planner)[
        "streams"] == 5


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
    code, out = _smoke(tmp_path)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["sweep"]["status"] == "ok" and state["solve"]["status"] == "ok"
    assert list(state).index("sweep") < list(state).index("solve")
    sweep = json.loads((out / "sweep.json").read_text())
    assert [r["batch"] for r in sweep["rows"]] == [1, 2]
    for row in sweep["rows"]:
        assert len(row["requests"]) == row["batch"] and not row["errors"]
        assert row["requests"][0]["completion_tokens"] == 64
        assert row["aggregate_tok_s"] and row["step_ms"] and row["median_tok_s"]


def test_competition_skips_the_sweep(tmp_path):
    code, out = _smoke(tmp_path, lambda s: s.update(mode="competition"))
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and "sweep" not in state and state["solve"]["submission_filled"] == 6


def test_an_8b_boot_fault_boots_revision_20s_set_without_8bs_variables(tmp_path):
    """The fake refuses 8B's skip lock; rung 1 removes the variable and serves."""
    ladder = config()["engine"]["boot_ladder"]

    def edit(smoke):
        smoke["mode"] = "competition"
        smoke["engine"]["flags"]["--fail-if-env"] = "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"
        smoke["engine"]["env"] = {"SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1", "ARC26_KV4": "1"}
        smoke["engine"]["boot_ladder"] = ladder

    code, out = _smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["boot"]["status"] == "ok"
    rows = [json.loads(line) for line in (out / "restarts.jsonl").read_text().splitlines()]
    assert [r["rung"] for r in rows] == [1] and rows[0]["reason"].startswith("boot:")
    commands = [json.loads(line) for line in (out / "engine_commands.jsonl").read_text()
                .splitlines()]
    assert commands[1]["env"]["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"] is None
    assert "--max-running-requests" in commands[1]["argv"]
    assert state["solve"]["submission_filled"] == 6


def test_the_twin_is_p5201s_twin_on_8b():
    twin = config("configs/P5.6-01-cpu.yaml")
    assert twin["kernel"]["id"] == "alyahya737/arc26-fn-8b-cpu"
    assert twin["kernel"]["accelerator"] is None
    assert twin["kernel"]["code_dataset"] == "alyahya737/arc26-code-fn-8b"
    assert twin["engine"]["script"] == "fn_fake_sglang.py" and twin["engine"]["boot_ladder"] is None
    assert twin["import_check"]["enabled"] and len(twin["patches"]["items"]) == 14
    # The argument check resolves 8B's flags (the fake's own dropped).
    assert twin["engine"]["flags"]["--mamba-radix-cache-strategy"] == "extra_buffer_lazy"
    assert twin["engine"]["flags"]["--max-running-requests"] == 9


def test_extends_resolves_a_chain_and_refuses_a_loop():
    files = {"a": b"extends: b\nx: 1\n", "b": b"extends: c\nx: 2\ny: 2\n", "c": b"y: 3\nz: 3\n"}
    assert nb.resolve_extends(files.__getitem__, "a") == {"x": 1, "y": 2, "z": 3}
    with pytest.raises(nb.BuildError):
        nb.resolve_extends({"a": b"extends: b\n", "b": b"extends: a\n"}.__getitem__, "a")


def test_both_notebooks_build_byte_identically(tmp_path):
    for name in ("configs/P5.6-01.yaml", "configs/P5.6-01-cpu.yaml"):
        first = nb.build(REPO / name, tmp_path / "a", REPO, allow_dirty=True)
        second = nb.build(REPO / name, tmp_path / "b", REPO, allow_dirty=True)
        # MANIFEST.md names the output folder; every pushed file must match.
        files_a = {p.relative_to(first): p.read_bytes() for p in first.rglob("*")
                   if p.is_file() and p.name != "MANIFEST.md"}
        files_b = {p.relative_to(second): p.read_bytes() for p in second.rglob("*")
                   if p.is_file() and p.name != "MANIFEST.md"}
        assert files_a == files_b
        meta = json.loads((first / "kernel-metadata.json").read_text())
        assert "alyahya737/arc26-code-fn-8b" in meta["dataset_sources"]
        assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
