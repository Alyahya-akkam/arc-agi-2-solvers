"""P5.2-01 revision 22 (task P5.10): revision 21's engine with the fixes from P5.6-01's forensics.

Checks that configs/P5.2-01.yaml keeps revision 21's engine flags and variables exactly (5
streams, 6 server slots, 0.95, Mamba 30, extra_buffer, the vision tower on, graphs to 5, the
2,048 chunk) and adds the health probe, the boot stress check with its fallback to the smaller
prefill, and its own code dataset; that the argument check resolves the fallback set; and, on
the fake server, that a failed stress check reboots with the fallback flags and passes, and that
a second failure moves the boot one rung. No network, no GPU.
"""

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
REVISION_21 = "348d958bd861a86b499a6c2381a4b875370cb590"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("fn_probe_r22_under_test", SCRIPTS / "fn_probe.py")


def config():
    return yaml.safe_load((REPO / "configs" / "P5.2-01.yaml").read_text())


def revision_21():
    done = subprocess.run(["git", "show", f"{REVISION_21}:configs/P5.2-01.yaml"], cwd=REPO,
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip("revision 21's commit is not in this clone")
    return yaml.safe_load(done.stdout)


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


def test_revision_22_is_revision_21_with_the_probe_the_stress_check_and_a_new_slug():
    ours, before = config(), revision_21()
    assert _diff(before, ours) == {
        "kernel.code_dataset", "inputs.code.path", "inputs.code.hint",
        "health.probe_timeout_s", "health.kill_probes", "stress", "engine.free_pattern",
        "smoke.health.probe_timeout_s", "smoke.health.kill_probes", "smoke.stress"}
    # Revision 21's engine exactly.
    assert ours["engine"]["flags"] == before["engine"]["flags"]
    assert ours["engine"]["env"] == before["engine"]["env"]
    assert ours["engine"]["boot_ladder"] == before["engine"]["boot_ladder"]
    flags = ours["engine"]["flags"]
    assert (flags["--max-running-requests"], ours["solve"]["max_streams"]) == (6, 5)
    assert (flags["--mem-fraction-static"], flags["--max-mamba-cache-size"]) == (0.95, 30)
    assert flags["--mamba-radix-cache-strategy"] == "extra_buffer"
    assert "--json-model-override-args" not in flags
    assert flags["--cuda-graph-bs-decode"] == [1, 2, 3, 4, 5]
    assert flags["--chunked-prefill-size"] == 2048 and "--prefill-max-requests" not in flags
    assert ours["kernel"]["id"] == "alyahya737/arc26-submission"
    assert ours["kernel"]["code_dataset"] == "alyahya737/arc26-code-fn-kv4c"
    assert ours["inputs"]["code"]["hint"] == ["arc26-code-fn-kv4c", "arc26-code-fn"]
    stress = ours["stress"]
    assert stress["enabled"] and stress["modes"] == ["development", "competition"]
    assert stress["fallback_flags"] == {"--chunked-prefill-size": 1024,
                                        "--prefill-max-requests": 4}
    assert (ours["health"]["probe_timeout_s"], ours["health"]["kill_probes"]) == (30, 3)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _smoke(tmp_path, edit=None):
    cfg = config()
    cfg["smoke"]["engine"]["port"] = _free_port()
    cfg["smoke"]["mode"] = "competition"
    if edit:
        edit(cfg["smoke"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    root = tmp_path / "root"
    code = probe.main(["--config", str(path), "--smoke", "--root", str(root), "--commit", "x"])
    return code, root / "kaggle/working/run_logs/fn"


def _rows(out, name):
    path = out / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_the_stress_check_passes_at_revision_21s_settings_on_the_fake(tmp_path):
    code, out = _smoke(tmp_path)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["stress"]["status"] == "ok"
    assert "boot_stress_fallback" not in state and _rows(out, "restarts.jsonl") == []
    assert state["solve"]["submission_filled"] == 6
    # The argument check resolved the fallback set too.
    assert state["arg_check"]["verdicts"]["stress_fallback"] in ("accepted", "device_stop")


def test_a_failed_stress_check_reboots_with_the_smaller_prefill(tmp_path):
    def edit(smoke):
        # The first boot dies with more than 3 requests in flight (the stress check's 4); the
        # fallback flags (the smaller prefill) also remove the fake's fault.
        smoke["engine"]["flags"]["--die-over-inflight"] = 3
        smoke["stress"]["fallback_flags"] = {"--chunked-prefill-size": 1024,
                                             "--prefill-max-requests": 4,
                                             "--die-over-inflight": None}

    code, out = _smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert state["stress"]["status"] == "failed"
    assert state["boot_stress_fallback"]["status"] == "ok"
    assert state["stress_fallback"]["status"] == "ok" and "boot_stress_retry" not in state
    rows = _rows(out, "restarts.jsonl")
    assert len(rows) == 1 and rows[0]["stress_fallback"]["--prefill-max-requests"] == 4
    commands = _rows(out, "engine_commands.jsonl")
    argv = commands[-1]["argv"]
    assert argv[argv.index("--chunked-prefill-size") + 1] == "1024"
    assert argv[argv.index("--prefill-max-requests") + 1] == "4"
    assert "--die-over-inflight" not in argv
    assert code == 0 and state["solve"]["submission_filled"] == 6


def test_a_second_stress_failure_moves_the_boot_one_rung(tmp_path):
    def edit(smoke):
        smoke["engine"]["flags"]["--die-over-inflight"] = 3
        # The fallback keeps the fault; rung 1 removes it.
        smoke["stress"]["fallback_flags"] = {"--chunked-prefill-size": 1024}
        smoke["engine"]["boot_ladder"] = [{"flags": {"--die-over-inflight": None}}]

    code, out = _smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert state["stress"]["status"] == "failed"
    assert state["stress_fallback"]["status"] == "failed"
    assert state["boot_stress_retry"]["status"] == "ok"
    rows = _rows(out, "restarts.jsonl")
    assert [r.get("rung") for r in rows] == [None, 1]
    assert code == 0 and state["solve"]["submission_filled"] == 6
