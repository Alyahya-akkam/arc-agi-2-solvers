"""Checks the agent spike's harness preflight, its fault reporting, and the gate's harness rule.

Run P6.3-02 (12 Oct) booted the engine, then every one of its 80 attempts failed to start a REPL
(the child's PYTHONPATH put the image's site folder, with a stale pathlib backport, before the
standard library); no traceback reached the log and the gate read Bad. These tests pin the three
fixes in ``scripts/agent_spike.py``: the preflight before the install, the visible and counted
faults, and the Fault row for a phase whose attempts never finished.
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arc26.agent.sandbox import SandboxError, import_roots
from arc26.agent.types import AgentTask, Pair

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.3-02.yaml"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike_harness")

# What a REPL that dies on import raises (the parent's view of run P6.3-02's children).
CHILD_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/usr/local/lib/python3.11/dist-packages/arc26/agent/sandbox.py", line 45, in '
    "<module>\n    from pathlib import Path\n"
    '  File "/usr/local/lib/python3.11/dist-packages/pathlib.py", line 10, in <module>\n'
    "    from collections import Sequence\n"
    "ImportError: cannot import name 'Sequence' from 'collections' "
    "(/usr/lib/python3.11/collections/__init__.py)\n"
)


def dead_repl(task, **kwargs):
    raise SandboxError(
        f"REPL failed to start: SandboxError('child {len(task.task_id) * 1009} exited (exit code "
        f"1)')\n{CHILD_TRACEBACK}"
    )


class IdleVerifier:
    closed = False
    replaced = 0

    def close(self):
        self.closed = True


def smoke_run(tmp_path):
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    challenges, solutions, _ = spike.write_smoke_inputs(scratch / "inputs")
    config = spike.smoke_overrides(spike.load_config(CONFIG), out, scratch, 8000)
    config["input_root"] = str(tmp_path / "no-input")
    args = argparse.Namespace(
        smoke=True, smoke_challenges=str(challenges), smoke_solutions=str(solutions),
        commit="test", resume_from=None, venv_python=None,
    )  # fmt: skip
    run = spike.Run(config, args)
    run.out.mkdir(parents=True)
    run.scratch.mkdir(parents=True, exist_ok=True)
    return run


def task(task_id):
    return AgentTask(task_id, (Pair([[1, 2]], [[2, 1]]),), ([[3, 4]],))


# --- the preflight -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("harness")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--smoke", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=600, cwd=REPO,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out, result.stdout


def test_preflight_runs_before_the_engine_with_the_agents_repl_and_verifier(smoke_out):
    out, stdout = smoke_out
    record = json.loads((out / "harness_preflight.json").read_text())
    assert record["ok"] is True and record["task"]["task_id"].startswith("smoke-")
    parent, child = record["parent"], record["child"]
    numpy_version = parent["modules"]["numpy"]["version"]
    assert record["cell"]["error"] is None and record["cell"]["stdout"].split()[0] == numpy_version
    assert int(record["cell"]["stdout"].split()[1]) > 30  # lib's helpers in the prompt index
    # The child: no site module, the standard library first, then exactly the parent's roots.
    roots = parent["import_roots"]
    assert child["no_site"] == 1 and child["path"][-len(roots) :] == roots
    assert child["path"][: -len(roots)] and "" not in child["path"]
    for name in ("arc26", "numpy"):
        assert child["modules"][name]["file"] == parent["modules"][name]["file"]
    assert parent["child_command"][1:4] == ["-B", "-S", "-c"]
    assert record["verification"]["failures"] == []
    stages = list(json.loads((out / "run.json").read_text())["stages"])
    assert stages[:3] == ["inputs", "harness", "engine_prep"]
    assert "harness preflight: ok" in stdout and (out / "harness.log").is_file()


def test_preflight_failure_stops_the_run_before_the_engine_with_the_traceback(
    tmp_path, monkeypatch, capsys
):
    verifier = IdleVerifier()
    monkeypatch.setattr(spike, "sandbox_parts", lambda settings: (dead_repl, verifier))
    run = smoke_run(tmp_path)
    with pytest.raises(spike.Stop, match="harness preflight failed at repl: SandboxError: "
                       "ImportError: cannot import name 'Sequence'"):  # fmt: skip
        spike.harness_preflight(run)
    record = json.loads((run.out / "harness_preflight.json").read_text())
    assert record["ok"] is False and record["failed_step"] == "repl"
    assert "from collections import Sequence" in record["traceback"]
    assert "from collections import Sequence" in capsys.readouterr().out
    # The whole run: exit 1 within seconds, no engine started, the pool closed.
    run = smoke_run(tmp_path / "again")
    began = time.monotonic()
    code = spike.run_spike(run.config, run.args)
    assert code == spike.EXIT_ERROR and time.monotonic() - began < 30
    written = json.loads((run.out / "run.json").read_text())
    assert list(written["stages"]) == ["inputs"]
    assert "harness preflight failed at repl" in written["problems"][0]
    assert verifier.closed


def test_preflight_takes_the_subsets_smallest_task_when_the_competition_is_there(tmp_path):
    folder = tmp_path / "input" / "arc-prize-2026-arc-agi-2"
    folder.mkdir(parents=True)
    smallest = spike_dev40()[7]
    challenges = {}
    for task_id in spike_dev40():
        size = 2 if task_id == smallest else 5
        challenges[task_id] = {
            "train": [{"input": [[1] * size] * size, "output": [[2] * size] * size}],
            "test": [{"input": [[1] * size] * size}],
        }
    (folder / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    (folder / "arc-agi_evaluation_solutions.json").write_text("{}")
    run = smoke_run(tmp_path)
    run.config["input_root"] = str(tmp_path / "input")
    assert spike.preflight_task(run).task_id == smallest
    run.config["input_root"] = str(tmp_path / "nowhere")
    assert spike.preflight_task(run).task_id.startswith("smoke-")


def spike_dev40():
    from arc26.data import DEV40

    return list(DEV40)


# --- faults in the cascade -----------------------------------------------------------------------


class DeadKit:
    def __init__(self):
        self.engine, self.verifier = object(), object()

    def repl_factory(self, task):
        dead_repl(task)


def test_raising_attempts_are_printed_once_per_kind_counted_and_read_as_fault(tmp_path, capsys):
    run = type("R", (), {})()
    run.deadline = time.monotonic() + 60
    run.config = {"seed": 0}
    run.out = tmp_path
    phase = {"max_tokens": 10, "token_budget": 20, "rounds": 1, "reasoning_effort": None,
             "temperature": None, "budget_s": 30, "preserve_thinking": False}  # fmt: skip
    settings = {"concurrency": 2, "cell_timeout_s": 1.0, "context_tokens": 1000,
                "phases": [phase, dict(phase)]}  # fmt: skip
    tasks = [task("aaaaaaaa"), task("bbbbbbbbbb"), task("cccccccccccc")]
    handlers = spike.configure_logging(tmp_path)
    try:
        result = asyncio.run(spike.run_cascade(tasks, DeadKit(), settings, run))
    finally:
        spike.close_logging(handlers)
    kind = (
        "SandboxError: ImportError: cannot import name 'Sequence' from 'collections' "
        "(/usr/lib/pythonN.N/collections/__init__.py)"
    )
    for number, row in enumerate(result["phases"], start=1):
        assert (row["phase"], row["attempts"], row["finished"]) == (number, 3, 0)
        assert (row["raised"], row["errors"]) == (3, {kind: 3})
    assert result["records"] == [] and list(result["errors"]) == [kind]
    assert "from collections import Sequence" in result["errors"][kind]["traceback"]
    # Standard output: the first traceback of the kind only; the log file: all six.
    out = capsys.readouterr().out
    assert out.count(" failed\nTraceback") == 1
    assert out.count("from collections import Sequence") == 1
    log = (tmp_path / spike.HARNESS_LOG).read_text()
    assert log.count("failed\nTraceback") == 6
    health = spike.harness_health(result["phases"])
    assert health["healthy"] is False and health["faults"] == [
        "phase 1: none of 3 attempts finished",
        "phase 2: none of 3 attempts finished",
    ]
    # M 0 with a healthy engine: Bad only with a healthy harness.
    assert spike.gate_row(0, None, True) == "Bad"
    assert spike.gate_row(0, None, True and health["healthy"]) == "Fault"


@pytest.mark.parametrize(
    "phases, faults",
    [
        ([{"phase": 1, "attempts": 40, "finished": 40, "raised": 0}], []),
        ([{"phase": 1, "attempts": 40, "finished": 37, "raised": 3}], []),
        ([{"phase": 1, "attempts": 40, "finished": 36, "raised": 4}],
         ["phase 1: 4 of 40 attempts raised"]),
        ([{"phase": 1, "attempts": 40, "finished": 0, "raised": 0}],
         ["phase 1: none of 40 attempts finished"]),
        ([{"phase": 1, "attempts": 40, "finished": 40, "raised": 0},
          {"phase": 2, "attempts": 0, "skipped": "deadline"}], []),
    ],
)  # fmt: skip
def test_harness_health_needs_finished_attempts_and_few_faults(phases, faults):
    assert spike.harness_health(phases) == {"healthy": not faults, "faults": faults}


def test_error_kind_names_the_type_and_the_last_line_without_numbers():
    first = spike.error_kind(SandboxError(f"child 12 exited\n{CHILD_TRACEBACK}"))
    second = spike.error_kind(SandboxError(f"child 4567 exited\n{CHILD_TRACEBACK}"))
    assert first == second and first.startswith("SandboxError: ImportError: cannot import name")
    assert spike.error_kind(RuntimeError()) == "RuntimeError"
    assert spike.error_kind(ValueError("port 8000 busy")) == "ValueError: port N busy"


def test_logging_keeps_the_root_loggers_setup_out(tmp_path, capsys):
    # A root handler that swallows records (as in run P6.3-02) no longer hides the arc26 records.
    root = logging.getLogger()
    swallow = logging.NullHandler()
    root.addHandler(swallow)
    handlers = spike.configure_logging(tmp_path)
    try:
        logging.getLogger("arc26.agent.loop").warning("visible %s", 1)
        logging.getLogger("arc26.agent.loop").info("file only")
    finally:
        spike.close_logging(handlers)
        root.removeHandler(swallow)
    assert "visible 1" in capsys.readouterr().out
    text = (tmp_path / spike.HARNESS_LOG).read_text()
    assert "visible 1" in text and "file only" in text
    assert logging.getLogger("arc26").propagate is True


def test_the_preflight_sees_the_roots_this_interpreter_gives_children():
    facts = spike.parent_facts()
    assert facts["import_roots"] == import_roots()
    assert facts["child_command"][-len(facts["import_roots"]) :] == facts["import_roots"]


# --- the Kaggle CPU check's build ----------------------------------------------------------------


def test_cpu_check_runs_the_gpu_builds_files_on_its_own_cpu_notebook():
    builder = load(REPO / "scripts" / "build_kaggle_notebook.py", "_test_nb_builder_harness")
    gateway = load(REPO / "scripts" / "kaggle_ops.py", "_test_gateway_harness")
    parsed = {}
    for name in ("P6.3-02-script", "P6.3-02-cpu-script"):
        path = REPO / "configs" / f"{name}.yaml"
        parsed[name] = builder.parse_config(path.read_bytes(), path)
    gpu, cpu = parsed["P6.3-02-script"], parsed["P6.3-02-cpu-script"]
    assert cpu.accelerator is None and gpu.accelerator == "NvidiaL4"
    # The same dataset files: one dataset version serves the check and then the GPU run.
    same = ("code_dataset", "script", "script_config", "reference_metadata", "competition")
    assert all(getattr(cpu, key) == getattr(gpu, key) for key in same)
    assert cpu.kernel_source_versions == gpu.kernel_source_versions
    assert cpu.kaggle_notebook != gpu.kaggle_notebook and cpu.notebook_inputs == {}
    assert cpu.kaggle_notebook not in gateway.briefed_notebooks()
    # The script reads the arguments as the smoke path on Kaggle's mounts.
    args = spike.build_parser().parse_args(
        ["--config", "x.yaml", *builder.fill_args(list(cpu.args), "0" * 40, {})]
    )
    assert args.smoke and args.out_dir == "/kaggle/working/run_logs"
    assert args.input_root == "/kaggle/input" and args.scratch_dir.startswith("/tmp/")
    config = spike.load_config(CONFIG)
    smoke_wall = spike.smoke_overrides(config, Path("/o"), Path("/s"), 8000)["max_wall_s"]
    assert 113.5 + smoke_wall < cpu.script_kill_s <= 1800 - 300
    text = (REPO / "configs" / "P6.3-02-cpu-script.yaml").read_text().splitlines()
    previous = ""
    for line in text:
        if line and not line.startswith((" ", "#")):
            assert previous.startswith("#"), line
        if line.strip():
            previous = line
