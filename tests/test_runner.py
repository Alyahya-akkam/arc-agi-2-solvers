"""Tests for arc26/runner.py, the interface the P0.5 notebook calls (task P0.1).

run_solver is exercised with a fake interpreter: a shell script named like the interpreter that
runs a small Python program instead of the solver, reporting its arguments and environment, then
exiting, or spawning a child and hanging so that the timeout must kill the whole process group.
"""

import ast
import bz2
import dataclasses
import inspect
import json
import math
import os
import pickle
import re
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from arc26 import runner, timing
from arc26.loop import HoldOut
from arc26.runner import RunConfig, make_submission, run_solver, solver_command, solver_env

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "reference" / "reference_notebook.ipynb"
KEYS = {
    "data_dir", "model_dir", "outputs_dir", "marker_dir", "subset", "workers", "timing_dir",
    "lora_r", "lora_targets",
}  # fmt: skip
# The reference's peft_params (cell 4): rank 256 on these nine modules, in this order.
NINE = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
]  # fmt: skip
LAUNCH = {
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "OMP_NUM_THREADS": "12",
}
END_TIME = 1759240000.25

FAKE_PROGRAM = """\
import json, os, subprocess, sys, time

names = ["PYTHONHASHSEED", "PYTHONUNBUFFERED", "FAKE_EXIT", *json.loads(os.environ["FAKE_NAMES"])]
print("report " + json.dumps({"argv": sys.argv[1:], "env": {n: os.environ.get(n) for n in names}}))
if os.environ.get("FAKE_MODE") == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(os.environ["FAKE_PIDS"], "w") as f:
        f.write(f"{os.getpid()} {child.pid}")
    print("hanging")  # not flushed: PYTHONUNBUFFERED must carry it out before the kill
    time.sleep(60)
sys.stderr.write("to stderr\\n")
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""


def config(**fields):
    return RunConfig(data_dir="/data", model_dir="/model", **fields)


@pytest.fixture
def fake_python(tmp_path, monkeypatch):
    """A fake interpreter at <tmp>/bin/python that prints its own path, then runs FAKE_PROGRAM
    with the real interpreter; FAKE_NAMES lists the variables it reports."""
    program = tmp_path / "fake_solver.py"
    program.write_text(FAKE_PROGRAM, encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "python"
    fake.write_text(
        f'#!/bin/sh\necho "interpreter $0"\nexec "{sys.executable}" "{program}" "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_NAMES", json.dumps(sorted(LAUNCH)))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("FAKE_EXIT", raising=False)
    return fake


def report(output):
    line = next(line for line in output.splitlines() if line.startswith("report "))
    return json.loads(line.removeprefix("report "))


def alive(pid):
    """Whether pid is a live process; a zombie waiting for its reaper counts as dead."""
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (ProcessLookupError, FileNotFoundError):
        # Reaped, possibly between kill() and the read: a killed grandchild is reaped by init.
        return False
    except (OSError, IndexError):
        return True
    return state not in {"Z", "X"}


def wait_dead(pid, seconds=10.0):
    deadline = time.monotonic() + seconds
    while alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


# The interface P0.5 codes against (task brief P0.1, deliverable 5).


def test_run_config_fields_and_defaults_match_the_interface():
    fields = [(f.name, f.default) for f in dataclasses.fields(RunConfig)]
    assert fields == [
        ("data_dir", dataclasses.MISSING),
        ("model_dir", dataclasses.MISSING),
        ("outputs_dir", "/kaggle/inference_outputs"),
        ("marker_dir", "/kaggle"),
        ("subset", None),
        ("workers", 4),
        ("hash_seed", 0),
        ("timing_dir", None),
        ("lora_r", 256),
        ("lora_targets", tuple(NINE)),
        ("solver", "split"),
        ("holdout", None),
        ("adapters", None),  # task P3.7: the adapter arms, arc26.arms
        ("hyper_checkpoint", None),
        ("hyper_sha256", None),
        ("hyper_log_dir", None),
    ]
    assert RunConfig.__dataclass_params__.frozen


def test_function_signatures_match_the_interface():
    def params(function):
        return [(p.name, p.default) for p in inspect.signature(function).parameters.values()]

    empty = inspect.Parameter.empty
    assert params(solver_env) == [("cfg", empty), ("base", empty)]
    # module (task P4.7b): the entry the notebook's rank-256 tail passes, arc26.tail.
    assert params(solver_command) == [
        ("cfg", empty), ("end_time", empty), ("python", empty), ("module", None)
    ]  # fmt: skip
    assert params(run_solver) == [
        ("cfg", empty), ("end_time", empty), ("python", None), ("timeout", None), ("module", None)
    ]  # fmt: skip
    assert params(make_submission) == [("cfg", empty), ("submission_path", empty)]


# Command, JSON config, and environment.


def test_solver_command_runs_the_moved_starter_with_a_json_config():
    cfg = config(subset=("0934a4d8", "aa4ec2a5"), workers=1, solver="nvarc")
    command = solver_command(cfg, END_TIME, "/usr/bin/python3")
    assert command[:6] == [
        "/usr/bin/python3", "-m", "arc26.nvarc.starter", "--end-time", repr(END_TIME), "--config"
    ]  # fmt: skip
    assert len(command) == 7
    assert json.loads(command[6]) == {
        "data_dir": "/data",
        "model_dir": "/model",
        "outputs_dir": "/kaggle/inference_outputs",
        "marker_dir": "/kaggle",
        "subset": ["0934a4d8", "aa4ec2a5"],
        "workers": 1,
        "timing_dir": None,
        "lora_r": 256,
        "lora_targets": NINE,
    }


def test_no_subset_is_json_null():
    assert json.loads(solver_command(config(), END_TIME, "python")[6])["subset"] is None


def test_timing_dir_reaches_the_solver_and_defaults_to_null():
    assert json.loads(solver_command(config(), END_TIME, "python")[6])["timing_dir"] is None
    cfg = config(timing_dir="/kaggle/working/run_logs/timing")
    payload = json.loads(solver_command(cfg, END_TIME, "python")[6])
    assert payload["timing_dir"] == "/kaggle/working/run_logs/timing"


@pytest.mark.parametrize("end_time", [END_TIME, time.time(), 0.0, 1e10 + 1 / 3, 1759240000])
def test_end_time_survives_the_command_line(end_time):
    assert float(solver_command(config(), end_time, "python")[4]) == end_time


@pytest.mark.parametrize("end_time", [math.inf, -math.inf, math.nan])
def test_non_finite_end_time_is_refused(end_time):
    with pytest.raises(ValueError):
        solver_command(config(), end_time, "python")


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"subset": ("0934a4d8", "36a08778", "981571dc", "aa4ec2a5"), "workers": 1},
        {"outputs_dir": "/tmp/out", "marker_dir": "/tmp/markers", "hash_seed": 2**32 - 1},
        {"timing_dir": "/tmp/run_logs/timing", "workers": 1},
        {"lora_r": 16, "lora_targets": ("q_proj", "k_proj", "v_proj", "o_proj")},
        {"solver": "split", "workers": 1},
    ],
    ids=["defaults", "dev4-one-worker", "other-folders", "timing", "lora-scope", "split"],
)
def test_json_config_round_trips_to_the_same_run_config(fields):
    # hash_seed reaches the solver through its environment, solver as the module it starts.
    cfg = config(**fields)
    payload = json.loads(solver_command(cfg, END_TIME, "python")[6])
    assert set(payload) == KEYS
    assert RunConfig(**payload, hash_seed=cfg.hash_seed, solver=cfg.solver) == cfg


def test_json_config_holds_the_keys_the_moved_code_reads():
    moved = REPO / "arc26" / "nvarc"
    read = set()
    for name in ("arc_solver.py", "starter.py"):
        read |= set(re.findall(r'cfg\["(\w+)"\]', (moved / name).read_text(encoding="utf-8")))
    assert read == KEYS


def test_solver_env_adds_the_launch_variables_and_the_hash_seed():
    base = {"PATH": "/usr/bin", "OMP_NUM_THREADS": "4", "PYTHONHASHSEED": "random"}
    env = solver_env(config(hash_seed=7), base)
    assert env == {"PATH": "/usr/bin", **LAUNCH, "PYTHONHASHSEED": "7"}
    assert base == {"PATH": "/usr/bin", "OMP_NUM_THREADS": "4", "PYTHONHASHSEED": "random"}


def test_launch_variables_are_those_of_the_reference_launch_cell():
    source = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][6]["source"]
    line = source if isinstance(source, str) else "".join(source)
    assignments = line.removeprefix("!").split(" python ")[0].split()
    assert dict(item.split("=", 1) for item in assignments) == LAUNCH == runner.LAUNCH_ENV


# Validation.


def test_a_subset_list_is_stored_as_a_tuple():
    cfg = config(subset=["0934a4d8"])
    assert cfg.subset == ("0934a4d8",)
    assert hash(cfg) == hash(config(subset=("0934a4d8",)))


INVALID = {
    "workers-zero": ({"workers": 0}, ValueError),
    "workers-negative": ({"workers": -1}, ValueError),
    "workers-bool": ({"workers": True}, TypeError),
    "workers-float": ({"workers": 2.0}, TypeError),
    "subset-empty": ({"subset": ()}, ValueError),
    "subset-uppercase": ({"subset": ("0934A4D8",)}, ValueError),
    "subset-short": ({"subset": ("0934a4d",)}, ValueError),
    "subset-long": ({"subset": ("0934a4d89",)}, ValueError),
    "subset-not-hex": ({"subset": ("0934a4dg",)}, ValueError),
    "subset-padded": ({"subset": ("0934a4d8", " 36a0877")}, ValueError),
    "subset-string": ({"subset": "0934a4d8"}, TypeError),
    "subset-number": ({"subset": (934,)}, TypeError),
    "hash-seed-negative": ({"hash_seed": -1}, ValueError),
    "hash-seed-too-large": ({"hash_seed": 2**32}, ValueError),
    "hash-seed-string": ({"hash_seed": "0"}, TypeError),
    "outputs-dir-empty": ({"outputs_dir": ""}, ValueError),
    "marker-dir-none": ({"marker_dir": None}, TypeError),
    "timing-dir-empty": ({"timing_dir": ""}, ValueError),
    "timing-dir-path": ({"timing_dir": Path("/timing")}, TypeError),
    "lora-r-zero": ({"lora_r": 0}, ValueError),
    "lora-r-negative": ({"lora_r": -16}, ValueError),
    "lora-r-bool": ({"lora_r": True}, TypeError),
    "lora-r-float": ({"lora_r": 16.0}, TypeError),
    "lora-r-string": ({"lora_r": "16"}, TypeError),
    "lora-targets-string": ({"lora_targets": "q_proj"}, TypeError),
    "lora-targets-number": ({"lora_targets": (1,)}, TypeError),
    "lora-targets-empty": ({"lora_targets": ()}, ValueError),
    "lora-targets-unknown": ({"lora_targets": ("q_proj", "qkv_proj")}, ValueError),
    "lora-targets-case": ({"lora_targets": ("Q_PROJ",)}, ValueError),
    "lora-targets-repeated": ({"lora_targets": ("q_proj", "q_proj")}, ValueError),
    "lora-targets-embeddings-only": ({"lora_targets": ("embed_tokens", "lm_head")}, ValueError),
    "solver-unknown": ({"solver": "reference"}, ValueError),
    "solver-case": ({"solver": "Split"}, ValueError),
    "solver-module": ({"solver": "arc26.solver"}, ValueError),
    "solver-none": ({"solver": None}, TypeError),
    "solver-list": ({"solver": ["split"]}, TypeError),
}


@pytest.mark.parametrize("fields, error", list(INVALID.values()), ids=list(INVALID))
def test_invalid_config_is_refused(fields, error):
    with pytest.raises(error):
        config(**fields)


@pytest.mark.parametrize("fields", [{"data_dir": ""}, {"model_dir": Path("/model")}])
def test_invalid_required_paths_are_refused(fields):
    with pytest.raises((TypeError, ValueError)):
        RunConfig(**{"data_dir": "/data", "model_dir": "/model", **fields})


# run_solver with a fake interpreter.


def test_run_solver_returns_the_exit_code_and_streams_the_output(fake_python, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_EXIT", "3")
    cfg = config(subset=("0934a4d8",), workers=2, hash_seed=11)
    assert run_solver(cfg, END_TIME, python=str(fake_python), timeout=60) == 3
    output = capsys.readouterr().out
    assert "to stderr" in output  # stderr is merged into the stream
    seen = report(output)
    assert seen["argv"] == solver_command(cfg, END_TIME, str(fake_python))[1:]
    assert seen["env"] == {
        **LAUNCH, "PYTHONHASHSEED": "11", "PYTHONUNBUFFERED": "1", "FAKE_EXIT": "3"
    }  # fmt: skip


def test_run_solver_defaults_to_python_on_path(fake_python, monkeypatch, capsys):
    monkeypatch.setenv("PATH", f"{fake_python.parent}{os.pathsep}{os.environ['PATH']}")
    assert run_solver(config(), END_TIME) == 0
    assert f"interpreter {fake_python}" in capsys.readouterr().out


def test_run_solver_without_python_on_path_raises(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(FileNotFoundError):
        run_solver(config(), END_TIME)


def test_run_solver_timeout_kills_the_process_group(fake_python, tmp_path, monkeypatch, capsys):
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MODE", "hang")
    monkeypatch.setenv("FAKE_PIDS", str(pids))
    started = time.monotonic()
    assert run_solver(config(), END_TIME, python=str(fake_python), timeout=2.0) == -signal.SIGKILL
    assert time.monotonic() - started < 30
    assert "hanging" in capsys.readouterr().out  # printed before the kill, not lost
    solver, child = map(int, pids.read_text().split())
    assert wait_dead(solver) and wait_dead(child)


def test_run_solver_kills_the_process_group_on_an_exception(fake_python, tmp_path, monkeypatch):
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MODE", "hang")
    monkeypatch.setenv("FAKE_PIDS", str(pids))
    write = runner._write

    def interrupted(text):
        write(text)
        if "hanging" in text:
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_solver(config(), END_TIME, python=str(fake_python), timeout=30)
    solver, child = map(int, pids.read_text().split())
    assert wait_dead(solver) and wait_dead(child)


# make_submission.


@pytest.fixture
def stub_transformers(monkeypatch):
    moved = ("arc26.nvarc.select", "arc26.nvarc.arc_loader", "arc26.nvarc.arc_decoder")
    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = type("AutoTokenizer", (), {})
    monkeypatch.setitem(sys.modules, "transformers", stub)
    for name in moved:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in moved:
        sys.modules.pop(name, None)


def test_make_submission_creates_the_outputs_folder(stub_transformers, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    challenges = {"0934a4d8": {"train": [], "test": [{"input": [[1]]}]}}
    (data / "arc-agi_test_challenges.json").write_text(json.dumps(challenges), encoding="utf-8")
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    outputs = tmp_path / "never" / "written"
    path = tmp_path / "submission.json"
    make_submission(RunConfig(str(data), str(tmp_path / "model"), str(outputs)), str(path))
    assert outputs.is_dir()
    default = {"attempt_1": [[0]], "attempt_2": [[0]]}
    assert json.loads(path.read_text(encoding="utf-8")) == {"0934a4d8": [default]}


def selection_inputs(tmp_path, candidate=True):
    """A data folder with one task of one test input (test and evaluation challenges, solutions),
    and, when candidate is true, one candidate pickle for it as the solver writes them."""
    np = pytest.importorskip("numpy")
    data = tmp_path / "data"
    data.mkdir()
    task = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1]]}]}
    challenges = {"0934a4d8": task}
    for name in ("arc-agi_test_challenges.json", "arc-agi_evaluation_challenges.json"):
        (data / name).write_text(json.dumps(challenges), encoding="utf-8")
    solutions = {"0934a4d8": [[[2]]]}
    (data / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions), encoding="utf-8")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    if candidate:
        sample = {"beam_score": 0.1, "score_aug": [0.2] * 8, "solution": np.array([[2]])}
        with bz2.BZ2File(outputs / "0934a4d8_0.permute0123456789", "w") as handle:
            pickle.dump([sample], handle)
    return data, outputs


def timing_lines(folder):
    return {
        path.name: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in sorted(folder.rglob("timing-*.jsonl"))
    }


@pytest.mark.parametrize("mode", ["competition", "development"])
def test_make_submission_records_main_selection_once(
    stub_transformers, tmp_path, monkeypatch, mode
):
    if mode == "competition":
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    else:
        monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    data, outputs = selection_inputs(tmp_path)
    timing_dir = tmp_path / "run_logs" / "timing"
    before = timing.get_logger()
    cfg = RunConfig(str(data), str(tmp_path / "model"), str(outputs), timing_dir=str(timing_dir))
    make_submission(cfg, str(tmp_path / "submission.json"))
    written = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert written == {"0934a4d8": [{"attempt_1": [[2]], "attempt_2": [[0]]}]}
    ((name, records),) = timing_lines(tmp_path).items()
    assert name == "timing-main.jsonl" and (timing_dir / name).is_file()
    (record,) = records
    assert record["puzzle_id"] is None and record["op"] == "main_selection"
    assert record["tokens"] is None and record["seconds"] >= 0
    # Its own logger: the calling process (the notebook kernel) keeps no process-wide logger.
    assert timing.get_logger() is before


def test_make_submission_without_timing_dir_records_nothing(
    stub_transformers, tmp_path, monkeypatch
):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    data, outputs = selection_inputs(tmp_path)
    make_submission(RunConfig(str(data), "/model", str(outputs)), str(tmp_path / "s.json"))
    assert (tmp_path / "s.json").is_file()
    assert timing_lines(tmp_path) == {}


@pytest.mark.filterwarnings("ignore:Mean of empty slice", "ignore:invalid value encountered")
def test_a_selection_that_raises_records_nothing(stub_transformers, tmp_path, monkeypatch):
    # Development mode without candidates: the reference's scoring raises after writing the file.
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    data, outputs = selection_inputs(tmp_path, candidate=False)
    timing_dir = tmp_path / "run_logs" / "timing"
    cfg = RunConfig(str(data), "/model", str(outputs), timing_dir=str(timing_dir))
    with pytest.raises(ValueError):
        make_submission(cfg, str(tmp_path / "s.json"))
    assert (tmp_path / "s.json").is_file()
    assert not timing_dir.exists()


def test_importing_the_runner_imports_no_gpu_stack_or_moved_module():
    heavy = ("torch", "unsloth", "transformers", "numpy", "arc26.nvarc.select")
    code = f"import sys, arc26.runner; print([m for m in {heavy!r} if m in sys.modules])"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


# LoRA scope (task P1.3): the reference's settings by default, validated, and in the JSON.


def test_the_lora_defaults_are_the_reference_settings():
    assert runner.LORA_R == 256 and list(runner.LORA_TARGETS) == NINE
    assert runner.LORA_PROJECTIONS == tuple(NINE[:7])
    payload = json.loads(solver_command(config(), END_TIME, "python")[6])
    assert (payload["lora_r"], payload["lora_targets"]) == (256, NINE)


def test_a_lora_scope_reaches_the_solver_as_json():
    cfg = config(lora_r=16, lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"])
    assert cfg.lora_targets == ("q_proj", "k_proj", "v_proj", "o_proj")
    payload = json.loads(solver_command(cfg, END_TIME, "python")[6])
    assert payload["lora_r"] == 16
    assert payload["lora_targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


@pytest.mark.parametrize(
    "targets",
    [("q_proj",), ("down_proj", "lm_head"), ("lm_head", "v_proj", "q_proj"), tuple(NINE[:7])],
)
def test_any_scope_with_a_projection_is_accepted(targets):
    assert config(lora_r=1, lora_targets=targets).lora_targets == targets


# Solver code (task P2.0): RunConfig.solver picks the module; the JSON config is the same.


def test_the_split_solver_is_the_default_since_p2_0_01_passed():
    assert config().solver == "split"
    assert runner.SOLVER_MODULES == {"nvarc": "arc26.nvarc.starter", "split": "arc26.solver"}


@pytest.mark.parametrize(
    "fields",
    [{}, {"subset": ("0934a4d8",), "workers": 1, "timing_dir": "/tmp/timing", "lora_r": 16}],
    ids=["defaults", "settings"],
)
def test_the_split_solver_starts_arc26_solver_with_the_same_arguments(fields):
    nvarc = solver_command(config(solver="nvarc", **fields), END_TIME, "/usr/bin/python3")
    split = solver_command(config(solver="split", **fields), END_TIME, "/usr/bin/python3")
    assert solver_command(config(**fields), END_TIME, "/usr/bin/python3") == split
    assert nvarc[:3] == ["/usr/bin/python3", "-m", "arc26.nvarc.starter"]
    assert split[:3] == ["/usr/bin/python3", "-m", "arc26.solver"]
    assert split[3:] == nvarc[3:]


def test_every_solver_module_is_a_process_entry():
    for module in runner.SOLVER_MODULES.values():
        path = REPO.joinpath(*module.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        entries = [
            node
            for node in tree.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        ]
        assert len(entries) == 1 and entries[0] is tree.body[-1]


def test_the_split_solver_reads_the_json_keys_the_moved_code_reads():
    read = set()
    for name in ("solver.py", "ttt.py", "decode.py"):
        text = (REPO / "arc26" / name).read_text(encoding="utf-8")
        read |= set(re.findall(r'cfg\["(\w+)"\]', text))
    assert read == KEYS


def test_run_solver_starts_the_split_solver(fake_python, capsys):
    cfg = config(solver="split", subset=("0934a4d8",), workers=1)
    assert run_solver(cfg, END_TIME, python=str(fake_python), timeout=60) == 0
    seen = report(capsys.readouterr().out)
    assert seen["argv"][:2] == ["-m", "arc26.solver"]
    assert seen["argv"] == solver_command(cfg, END_TIME, str(fake_python))[1:]
    assert seen["env"]["PYTHONHASHSEED"] == "0"


# Hold-out (task P2.1): RunConfig.holdout reaches the split solver as the JSON's "holdout".

HOLDOUT = {
    "gate_dir": "/kaggle/gate_outputs/arm",
    "log_dir": "/kaggle/working/run_logs/arm/gate",
    "rule": "last",
    "k2": True,
    "in_test_prompts": False,
}


def test_holdout_defaults_to_none_and_leaves_the_json_unchanged():
    for solver in ("nvarc", "split"):
        payload = json.loads(solver_command(config(solver=solver), END_TIME, "python")[6])
        assert set(payload) == KEYS and "holdout" not in payload


@pytest.mark.parametrize("holdout", [HOLDOUT, HoldOut(**HOLDOUT), {"gate_dir": "/g"}])
def test_holdout_reaches_the_split_solver_and_round_trips(holdout):
    cfg = config(solver="split", subset=("0934a4d8",), holdout=holdout)
    assert isinstance(cfg.holdout, HoldOut)
    payload = json.loads(solver_command(cfg, END_TIME, "python")[6])
    assert set(payload) == KEYS | {"holdout"}
    assert payload["holdout"] == cfg.holdout.to_json()
    assert RunConfig(**payload, hash_seed=cfg.hash_seed, solver=cfg.solver) == cfg
    assert HoldOut.from_cfg(payload) == cfg.holdout
    assert hash(cfg) == hash(RunConfig(**payload, solver="split"))


def test_holdout_needs_the_split_solver():
    with pytest.raises(ValueError, match="holdout needs solver 'split'"):
        config(solver="nvarc", holdout=HOLDOUT)
    assert config(holdout=HOLDOUT).solver == "split"


@pytest.mark.parametrize(
    ("holdout", "error"),
    [
        (["gate_dir"], TypeError),
        ("/kaggle/gate", TypeError),
        ({}, ValueError),
        ({**HOLDOUT, "rule": "middle"}, ValueError),
        ({**HOLDOUT, "k2": "yes"}, TypeError),
        ({**HOLDOUT, "views": 8}, ValueError),
        ({**HOLDOUT, "gate_dir": ""}, ValueError),
    ],
    ids=["list", "string", "no-gate-dir", "rule", "k2", "unknown-key", "empty-gate-dir"],
)
def test_invalid_holdout_is_refused(holdout, error):
    with pytest.raises(error):
        config(solver="split", holdout=holdout)


def test_the_split_solver_reads_the_optional_holdout_and_nothing_else_new():
    """The split solver reads the moved code's keys with brackets, and the hold-out settings
    only through HoldOut.from_cfg, which tolerates their absence."""
    texts = {
        name: (REPO / "arc26" / name).read_text(encoding="utf-8")
        for name in ("solver.py", "ttt.py", "decode.py", "loop.py")
    }
    bracketed = set()
    for text in texts.values():
        bracketed |= set(re.findall(r'cfg\["(\w+)"\]', text))
    assert bracketed == KEYS
    assert re.findall(r'cfg\.get\("(\w+)"\)', "".join(texts.values())) == ["holdout"]
    assert "holdout = HoldOut.from_cfg(cfg)" in texts["solver.py"]


@pytest.mark.parametrize(
    "folders",
    [
        {"gate_dir": "/kaggle/inference_outputs"},
        {"gate_dir": "/kaggle/inference_outputs/gate"},
        {"gate_dir": "/kaggle/gate", "log_dir": "/kaggle/inference_outputs/../inference_outputs/g"},
    ],
    ids=["equal", "inside", "log-inside"],
)
def test_the_gate_folders_lie_outside_the_candidate_folder(folders):
    with pytest.raises(ValueError, match="must lie outside outputs_dir"):
        config(solver="split", holdout={**HOLDOUT, **folders})
    # Beside it, as the notebook puts them, is fine.
    assert config(solver="split", holdout={**HOLDOUT, "gate_dir": "/kaggle/inference_gate"})


def test_run_solver_passes_the_holdout_to_the_split_solver(fake_python, capsys):
    cfg = config(solver="split", subset=("0934a4d8",), workers=1, holdout=HOLDOUT)
    assert run_solver(cfg, END_TIME, python=str(fake_python), timeout=60) == 0
    seen = report(capsys.readouterr().out)
    assert seen["argv"][:2] == ["-m", "arc26.solver"]
    assert json.loads(seen["argv"][-1])["holdout"] == HOLDOUT
