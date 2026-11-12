"""Simulated-Kaggle runs of the hybrid submission notebook (task P6.5): NVARC, then the agent.

The notebook is built from ``configs/submission_P6.5-hybrid.yaml`` with the agent's run config of
task P6.4 (``configs/agent_default.yaml``) and the simulation's
times: NVARC's share 600 s, the agent's end 3600 s, a final margin of 1 s, SIGKILL 3 s after
SIGTERM. Its code cells run in order on the simulated Kaggle tree of
``tests/test_notebook_local.py`` (a separate interpreter with ``-S``, a recording ``python``
shim, a fake ``nvidia-smi`` with four L4s, an audit hook on file opens), with the two download
notebooks' outputs mounted (task P6.3: the weights' ``MANIFEST.json`` and files, the wheelhouse's
``MANIFEST.json``, requirements, and ``tests/testdata/fake_install_wheelhouse.py`` as its
installer). The shim runs ``tests/testdata/fake_nvarc_solver.py`` for ``python -m arc26.solver``
(a fake NVARC stage: candidates for two tasks, which the notebook's real selection turns into
NVARC's attempts) and ``tests/testdata/fake_agent_run.py`` for ``python -m arc26.agent.run`` (P6.4's
command line, told how to behave).

Against the task brief's acceptance this file checks, in both modes: the agent succeeds and
improves NVARC's file; the agent fails at the install (a failed install, no engine, a hung
install), at the boot, mid-run (a crash that leaves its engine and a cut temporary file, a
corrupt or invalid file, a refused config), and at the deadline (SIGTERM honored, SIGTERM
ignored), and every case ends with a valid ``submission.json`` that keeps NVARC's attempts; a
failed agent check stops a development run and, in a competition rerun, gives NVARC its full
deadlines; a competition rerun never opens a solutions file and writes nothing but
``submission.json`` to ``/kaggle/working``; NVARC's deadlines follow its share. The last test runs
the real ``arc26.agent.run`` (task P6.4) with its fake engine, once the module is merged.
"""

import contextlib
import copy
import importlib.util
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import NamedTuple

import pytest

import arc26.submit as submit

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


LOCAL = load("_hybrid_notebook_local", REPO / "tests" / "test_notebook_local.py")
nbb = LOCAL.nbb
TESTDATA = REPO / "tests" / "testdata"
FAKE_AGENT = TESTDATA / "fake_agent_run.py"
FAKE_SOLVER = TESTDATA / "fake_nvarc_solver.py"
FAKE_INSTALLER = TESTDATA / "fake_install_wheelhouse.py"
HYBRID_CONFIG = "configs/submission_P6.5-hybrid.yaml"
AGENT_CONFIG = "configs/agent_default.yaml"
GRID = LOCAL.GRID
DEV4 = LOCAL.DEV4
# The fake NVARC solves the first two tasks of the mode's challenges; the fake agent the first
# test output NVARC left at [[0]].
NVARC_TASKS = sorted(DEV4)[:2]
AGENT_OUTPUT = ("981571dc", 0)
# Stand-in hidden test tasks for a competition rerun: ids no evaluation task has.
HIDDEN = {"f0000001": 1, "f0000002": 2, "f0000003": 1}
REVISION = "91bd022d5b49442a868bc35008f6c21e1860edfa"
MODEL_NOTEBOOK = "alyahya737/arc26-dl-qwen38-int4"
WHEELS_NOTEBOOK = "alyahya737/arc26-dl-agent-wheels"
# A driver older than CUDA 13.0's 580.65.06.
OLD_DRIVER_L4 = "NVIDIA L4, 23034 MiB, 8.9, 535.104.05"
HYBRID_STAGES = ["environment", "agent_check", "install", "run", "write", "agent", "run_logs"]

DRIVER = r'''
import json, os, sys, time, traceback

notebook_path, record_path = sys.argv[1], sys.argv[2]
# Changes to AGENT made when the agent stage's cell starts, as a human could set them in the
# Config cell: "end_in" sets end_s that many seconds after that moment.
overrides = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
root = os.environ["ARC26_KAGGLE_ROOT"]
watched = {"torch", "tensorflow", "unsloth", "transformers"}
record = {"cells": [], "imports": [], "opened": [], "error": None}
state = {"cell": None}


class Recorder:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in watched:
            record["imports"].append([state["cell"], name])
        return None


def audit(event, args):
    # Files under the simulated root, and anything named like a solutions file wherever it is.
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.startswith(root) or "solution" in os.path.basename(path).lower():
            record["opened"].append([state["cell"], path])


sys.meta_path.insert(0, Recorder())
sys.addaudithook(audit)
with open(notebook_path, encoding="utf-8") as handle:
    cells = json.load(handle)["cells"]
namespace = {"__name__": "__main__"}
try:
    for cell in cells:
        if cell["cell_type"] == "code":
            state["cell"] = cell["id"]
            if cell["id"].endswith("-agent") and overrides:
                changes = dict(overrides)
                if "end_in" in changes:
                    start = namespace["NOTEBOOK_START"]
                    changes["end_s"] = time.time() - start + changes.pop("end_in")
                namespace["AGENT"].update(changes)
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
            record["cells"].append(cell["id"])
except BaseException:
    record["error"] = traceback.format_exc()
finally:
    names = ("STAGES", "PROBLEMS", "GPUS", "CHALLENGES", "SOLVER_EXIT", "NOTEBOOK_START")
    names += ("AGENT_READY", "AGENT_EXIT", "AGENT_PLAN", "AGENT")
    for name in names:
        record[name] = namespace.get(name)
    record["report"] = namespace.get("SUBMISSION_REPORT")
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, default=str)
'''

# test_notebook_local.py's shim, plus the fakes: with ARC26_FAKE_SOLVER or ARC26_FAKE_AGENT set,
# python -m arc26.solver or python -m arc26.agent.run runs that file instead, in the same process
# (exec keeps the pid, so the notebook's process group reaches it).
SHIM = r'''#!@PYTHON@ -S
import json, os, sys, zipfile

args = sys.argv[1:]
with open(os.environ["ARC26_SHIM_LOG"], "a", encoding="utf-8") as log:
    exists = os.path.exists(os.environ["ARC26_SHIM_SUBMISSION"])
    log.write(json.dumps({"argv": args, "submission_exists": exists}) + "\n")
if args[:3] == ["-m", "pip", "uninstall"]:
    print("WARNING: Skipping tensorflow as it is not installed.")
    sys.exit(0)
if args[:3] == ["-m", "pip", "install"]:
    wheels = [arg for arg in args[3:] if arg.endswith(".whl")]
    if "--no-index" not in args or "--no-deps" not in args or len(wheels) != 1:
        sys.exit(1)
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(os.environ["ARC26_SHIM_TARGET"])
    print("Successfully installed arc26-0.0.1")
    sys.exit(0)
if args[:3] == ["-m", "pip", "freeze"]:
    print("arc26==0.0.1")
    sys.exit(0)
fakes = {"arc26.solver": "ARC26_FAKE_SOLVER", "arc26.agent.run": "ARC26_FAKE_AGENT"}
if args[:1] == ["-m"] and len(args) > 1 and os.environ.get(fakes.get(args[1], "-")):
    os.execv("@PYTHON@", ["@PYTHON@", "-S", os.environ[fakes[args[1]]], *args[2:]])
os.execv("@PYTHON@", ["@PYTHON@", "-S", *args])
'''


def read_repo(name):
    """A repository file's bytes, as resolve_agent reads them."""
    return (REPO / name).read_bytes()


def hybrid_config(subset=None, **agent_changes):
    """The hybrid's config, resolved, with the simulation's times and a missing base python (the
    kernel's interpreter makes the venv)."""
    path = REPO / HYBRID_CONFIG
    config = nbb.resolve_agent(nbb.parse_config(path.read_bytes(), path), read_repo)
    data = copy.deepcopy(config.agent.config_data)
    data["deadline"].update(final_margin_s=1, grace_s=1)
    times = {
        "nvarc_share_s": 600,
        "end_s": 3600,
        "term_s": 3,
        "min_stage_s": 1,
        "install_timeout_s": 60,
        "base_python": "/nonexistent/bin/python3.11",
        "config_data": data,
    }
    agent = config.agent._replace(**{**times, **agent_changes})
    return config._replace(agent=agent, subset=subset)


def cpu_config():
    """The hybrid's CPU check (configs/submission_P6.5-cpu.yaml), resolved, as committed but for
    a missing base python."""
    path = REPO / "configs" / "submission_P6.5-cpu.yaml"
    config = nbb.resolve_agent(nbb.parse_config(path.read_bytes(), path), read_repo)
    return config._replace(agent=config.agent._replace(base_python="/nonexistent/python3.11"))


def build(dest, wheel, name, config):
    info = nbb.build_info(
        config,
        tag=f"local-{name}",
        commit=LOCAL.COMMIT,
        config_file=HYBRID_CONFIG,
        config_sha256=LOCAL.sha256(REPO / HYBRID_CONFIG),
        wheel=wheel.name,
        wheel_sha256=LOCAL.sha256(wheel),
    )
    out = dest / f"{name}.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_notebook(config, info, nbb.wheel_member(wheel))))
    return out


class Built(NamedTuple):
    wheel: Path
    notebooks: dict
    configs: dict


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The hybrid notebook, its agent-only twin, one with the agent stage off, and one for a
    development run on two DEV4 tasks."""
    pytest.importorskip("numpy")
    dest = tmp_path_factory.mktemp("build")
    wheel = LOCAL.make_wheel(dest)
    configs = {
        "hybrid": hybrid_config(),
        "agent-only": hybrid_config(nvarc_share_s=0),
        "disabled": hybrid_config(enabled=False),
        "subset": hybrid_config(subset=("0934a4d8", "981571dc")),
        "cpu": cpu_config(),
        "agent-only-short": hybrid_config(nvarc_share_s=0)._replace(
            solver_end_s=600, solver_kill_s=660
        ),
    }
    notebooks = {name: build(dest, wheel, name, config) for name, config in configs.items()}
    return Built(wheel, notebooks, configs)


def mount_agent_inputs(root, layout="2026", model=None, wheels=None, missing=()):
    """Mount the two download outputs as Kaggle would (a layout of test_notebook_local.py);
    ``model`` and ``wheels`` change their manifests, ``missing`` leaves outputs out."""
    parents, underscores = LOCAL.OUTPUT_DIRS[layout]
    folders = {}
    for name in (MODEL_NOTEBOOK, WHEELS_NOTEBOOK):
        owner, slug = name.split("/")
        folder = root.joinpath(*(part.format(owner=owner) for part in parents))
        folders[name] = folder / (slug.replace("-", "_") if underscores else slug)
    if MODEL_NOTEBOOK not in missing:
        weights = folders[MODEL_NOTEBOOK] / "qwen38-27b-int4"
        files = {"config.json": b"{}", "model.safetensors": b"\0" * 64}
        for name, data in files.items():
            (weights / name).parent.mkdir(parents=True, exist_ok=True)
            (weights / name).write_bytes(data)
        manifest = {
            "kind": "model",
            "ok": True,
            "revision": REVISION,
            "folder": "qwen38-27b-int4",
            "files": [{"name": name, "size": len(data)} for name, data in files.items()],
        }
        LOCAL.write_json(folders[MODEL_NOTEBOOK] / "MANIFEST.json", {**manifest, **(model or {})})
    if WHEELS_NOTEBOOK not in missing:
        house = folders[WHEELS_NOTEBOOK] / "wheelhouse"
        house.mkdir(parents=True)
        shutil.copyfile(FAKE_INSTALLER, house / "install_wheelhouse.py")
        (house / "requirements.txt").write_text("vllm==0.31.0\nhttpx==0.28.1\npip==26.2.1\n")
        manifest = {
            "kind": "wheels",
            "ok": True,
            "files": [
                {"file": "vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl", "name": "vllm",
                 "version": "0.31.0"},
                {"file": "torch-2.13.0-cp311-cp311-manylinux_2_28_x86_64.whl", "name": "torch",
                 "version": "2.13.0"},
                {"file": "install_wheelhouse.py"},
            ],
            "install_check": {"ok": True, "seconds": 41.5},
        }  # fmt: skip
        LOCAL.write_json(house / "MANIFEST.json", {**manifest, **(wheels or {})})
    return folders


def make_tree(tmp, wheel, hidden=False, **mount):
    """The simulated tree with the download outputs mounted; ``hidden`` puts the stand-in hidden
    tasks in the test challenges, so a submission shows which file a run read."""
    root = LOCAL.make_tree(tmp, wheel)
    if hidden:
        data_dir = root / "input" / "competitions" / LOCAL.COMPETITION
        hidden_tasks = LOCAL.standin_challenges(HIDDEN)
        LOCAL.write_json(data_dir / "arc-agi_test_challenges.json", hidden_tasks)
    mount_agent_inputs(root, **mount)
    return root


class Run(NamedTuple):
    record: dict
    calls: list
    root: Path
    scratch: Path
    output: str


def simulate(
    tmp,
    notebook,
    root,
    competition=False,
    gpus=4,
    gpu=LOCAL.L4,
    agent="solve",
    install="ok",
    overrides=None,
    env=None,
    smi=None,
):
    """Run a hybrid notebook's code cells on the simulated tree, as tests/test_notebook_local.py
    does, with the fakes: ``agent`` is the fake agent's mode (None runs the real module),
    ``install`` the fake installer's, ``overrides`` the AGENT changes at the agent stage, ``smi``
    another fake nvidia-smi script."""
    bin_dir, target, stubs = tmp / "bin", tmp / "target", tmp / "stubs"
    for folder in (bin_dir, target, stubs, tmp / "home"):
        folder.mkdir(parents=True, exist_ok=True)
    (stubs / "transformers").mkdir(exist_ok=True)
    (stubs / "transformers" / "__init__.py").write_text("AutoTokenizer = None\n")
    for name, text in (
        ("python", SHIM.replace("@PYTHON@", sys.executable)),
        ("nvidia-smi", smi or LOCAL.nvidia_smi(gpus, gpu)),
    ):
        path = bin_dir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    driver, record_path, log = tmp / "driver.py", tmp / "record.json", tmp / "shim.jsonl"
    driver.write_text(DRIVER)
    scratch = tmp / "agent-scratch"
    purelib = sysconfig.get_paths()["purelib"]
    environment = {
        "PATH": str(bin_dir),
        "PYTHONPATH": os.pathsep.join([str(target), str(stubs), purelib]),
        "ARC26_KAGGLE_ROOT": str(root),
        "ARC26_AGENT_SCRATCH": str(scratch),
        "ARC26_SHIM_LOG": str(log),
        "ARC26_SHIM_TARGET": str(target),
        "ARC26_SHIM_SUBMISSION": str(root / "working" / "submission.json"),
        "ARC26_FAKE_SOLVER": str(FAKE_SOLVER),
        "ARC26_FAKE_SOLVER_LOG": str(tmp / "fake_solver.json"),
        "ARC26_FAKE_INSTALL": install,
        "ARC26_FAKE_PATHS": purelib,
        "HOME": str(tmp / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
    }
    if agent is not None:
        environment.update(ARC26_FAKE_AGENT=str(FAKE_AGENT), ARC26_FAKE_AGENT_MODE=agent)
    if competition:
        environment["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    environment.update(env or {})
    extra = [json.dumps(overrides)] if overrides else []
    result = subprocess.run(
        [sys.executable, "-S", str(driver), str(notebook), str(record_path), *extra],
        cwd=root / "working",
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    record = json.loads(record_path.read_text())
    return Run(record, calls, root, scratch, result.stdout + result.stderr)


def submission(run):
    return json.loads((run.root / "working" / "submission.json").read_text())


def nvarc_file(run):
    """NVARC's submission as the agent stage copied it before the agent ran."""
    return json.loads((run.scratch / "nvarc_submission.json").read_text())


def fake_agent(run):
    """What the fake agent recorded: its arguments and what it read; None if it never ran."""
    path = run.scratch / "logs" / "fake_agent.json"
    return json.loads(path.read_text()) if path.exists() else None


def fake_solver(run):
    path = run.root.parent / "fake_solver.json"
    return json.loads(path.read_text()) if path.exists() else None


def agent_stage(run):
    return run.record["STAGES"]["agent"]


def solved(written, task, index=0):
    return written[task][index]["attempt_1"] == GRID


def engine_pid(run):
    """The pid of the engine the fake agent started, from its line in the session log."""
    for line in run.output.splitlines():
        if line.startswith("fake agent: engine ") and line.endswith(" started"):
            return int(line.split()[3])
    return None


def assert_valid(run, counts=DEV4):
    written = submission(run)
    assert submit.validate(written, LOCAL.standin_challenges(counts)) == [], run.output
    assert run.record["report"]["valid"] is True
    return written


def assert_nvarc_kept(run, counts=DEV4):
    """Every attempt NVARC filled is in the final file, unless the agent changed that slot."""
    written, nvarc = submission(run), nvarc_file(run)
    for task in counts:
        for index, entry in enumerate(nvarc[task]):
            for name, grid in entry.items():
                if grid != [[0]]:
                    assert written[task][index][name] == grid, (task, index, name)


def working_files(run):
    return sorted(p.name for p in (run.root / "working").iterdir())


def assert_kernel_never_imported_torch(run):
    """The kernel imports no torch, tensorflow, or unsloth at all, and transformers (the moved
    selection's annotations) only from the Write cell on, after NVARC's solver has ended: a CUDA
    context in the kernel would take GPU memory from NVARC's workers and then from the engine."""
    names = {name.partition(".")[0] for _, name in run.record["imports"]}
    assert not names & LOCAL.NEVER_IMPORTED, run.record["imports"]
    order = run.record["cells"]
    write = next(i for i, cell in enumerate(order) if cell.endswith("-write"))
    early = [name for cell, name in run.record["imports"] if order.index(cell) < write]
    assert early == [], "the kernel imported transformers before NVARC's solver finished"


@contextlib.contextmanager
def no_stray_engine(run_holder):
    """Kill an engine a failed test left behind, so no process outlives the test."""
    try:
        yield
    finally:
        run = run_holder.get("run")
        pid = engine_pid(run) if run else None
        if pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)


def gone(pid, wait=5.0):
    return LOCAL.gone(pid, wait)


# The agent succeeds: NVARC's file first, then the agent's improvement on top.


def test_a_development_run_writes_nvarc_then_the_agent_improves_and_scores(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["hybrid"], root)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    assert list(run.record["STAGES"]) == HYBRID_STAGES
    start = run.record["NOTEBOOK_START"]
    # NVARC: its soft deadline is the share, its hard stop 600 s later; it ran on DEV4.
    solver = fake_solver(run)
    assert solver["end_time"] - start == pytest.approx(600)
    run_stage = run.record["STAGES"]["run"]
    assert (run_stage["solver_end_s"], run_stage["solver_kill_s"]) == (600, 1200)
    assert solver["tasks"] == NVARC_TASKS
    # The agent check passed and printed the plan.
    check = run.record["STAGES"]["agent_check"]
    assert check["ready"] is True and run.record["AGENT_READY"] is True
    assert check["weights"]["files"] == 2 and check["wheelhouse"]["torch"] == "2.13.0"
    assert check["driver"] is None and check["tensor_parallel_size"] == 4
    assert "plan, seconds after the start: nvarc_end_s 600, nvarc_kill_s 1200" in run.output
    # The install: the wheelhouse's own installer, the kernel's interpreter (the configured one is
    # missing), an environment without PYTHONPATH or PIP_*.
    stage = agent_stage(run)
    assert stage["install"] == {
        "exit": 0, "ended": "exited", "seconds": stage["install"]["seconds"], "ok": True,
        "problems": [], "imports": {"torch": {"ok": True}, "vllm": {"ok": True}}, "engine": True,
    }  # fmt: skip
    report = json.loads((run.scratch / "install.json").read_text())
    assert report["env"] == {"PYTHONPATH": None, "pip": [], "PYTHONNOUSERSITE": "1"}
    # The agent got P6.4's command line: the evaluation challenges, the output, the run config
    # as built, the venv, the weights, NVARC's two files, the deadline, the scratch log folder.
    agent = fake_agent(run)
    assert agent["mode"] == "dev" and agent["tasks"] == run.record["CHALLENGES"]
    out = agent["argv"][agent["argv"].index("--out") + 1]
    assert out == str(root / "working" / "submission.json")
    assert agent["config"] == built.configs["hybrid"].agent.config_data
    assert agent["engine_venv"] == str(run.scratch / "venv")
    assert agent["model_dir"].endswith("arc26-dl-qwen38-int4/qwen38-27b-int4")
    assert agent["deadline"] == pytest.approx(start + 3600)
    assert agent["cwd"] == str(run.scratch / "logs")
    assert agent["task_ids"] == sorted(DEV4) and "--task-ids" not in agent["argv"]
    assert "--solutions" not in agent["argv"]
    assert agent["candidates"] == sorted(f"{task}_{i}.permute0123456789" for task, i in (
        (NVARC_TASKS[0], 0), (NVARC_TASKS[1], 0), (NVARC_TASKS[1], 1)
    ))  # fmt: skip
    # NVARC's attempts reached the agent as written by the Write stage, and the final file keeps
    # them, with the agent's one improvement on top.
    assert agent["nvarc"] == nvarc_file(run)
    assert all(solved(nvarc_file(run), task) for task in NVARC_TASKS)
    written = assert_valid(run)
    assert_nvarc_kept(run)
    assert solved(written, *AGENT_OUTPUT) and written["aa4ec2a5"][0]["attempt_1"] == [[0]]
    assert stage["agent_exit"] == 0 and stage["ended"] == "exited"
    assert stage["final"]["sources"]["changed"] == 1 and stage["final"]["rewritten"] is False
    # The development score: NVARC's 2 of 4, the hybrid's 3 of 4.
    score = stage["score"]
    assert (score["nvarc"]["score"], score["final"]["score"]) == (2.0, 3.0)
    assert score["gained"] == ["981571dc_0"] and score["lost"] == []
    # The logs: the agent's in run_logs/agent/, the stages in stages.json; nothing else in
    # /kaggle/working.
    logs = root / "working" / "run_logs"
    assert {"run.json", "attempts.jsonl", "fake_agent.json", "install.json", "install.log"} <= {
        p.name for p in (logs / "agent").iterdir()
    }
    stages = json.loads((logs / "stages.json").read_text())
    assert stages["stages"]["agent"]["score"]["final"]["score"] == 3.0
    assert stages["submission"]["valid"] is True and stages["submission"]["sources"]
    assert working_files(run) == ["run_logs", "submission.json"]
    assert_kernel_never_imported_torch(run)


def test_a_competition_rerun_runs_both_stages_reads_no_solutions_and_writes_only_the_file(
    built, tmp_path
):
    root = make_tree(tmp_path, built.wheel, hidden=True)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, competition=True)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    # NVARC at its share in the rerun too, on the hidden tasks.
    assert fake_solver(run)["end_time"] - run.record["NOTEBOOK_START"] == pytest.approx(600)
    assert fake_solver(run)["tasks"] == ["f0000001", "f0000002"]
    agent = fake_agent(run)
    assert agent["mode"] == "competition"
    assert agent["tasks"].endswith("arc-agi_test_challenges.json")
    # An explicit deadline, the session's end at AGENT['end_s'], in a rerun as in development.
    assert agent["deadline"] == pytest.approx(run.record["NOTEBOOK_START"] + 3600)
    assert agent_stage(run)["agent_exit"] == 0
    assert not [arg for arg in agent["argv"] if "solution" in arg.lower()]
    written = assert_valid(run, HIDDEN)
    assert sorted(written) == sorted(HIDDEN)
    assert solved(written, "f0000001") and solved(written, "f0000002", 1)
    assert solved(written, "f0000003")
    # No solutions file is opened by the kernel, and nothing but submission.json is written to
    # /kaggle/working: the agent's logs, the venv, and NVARC's copies stay in scratch.
    opened = [path for _, path in run.record["opened"]]
    assert not [path for path in opened if "solution" in Path(path).name.lower()]
    assert working_files(run) == ["submission.json"]
    assert "score" not in agent_stage(run) and "logs" not in agent_stage(run)
    assert_kernel_never_imported_torch(run)


@pytest.mark.parametrize("layout", sorted(LOCAL.OUTPUT_DIRS))
def test_the_agent_check_finds_the_download_outputs_wherever_kaggle_mounts_them(
    built, tmp_path, layout
):
    root = make_tree(tmp_path, built.wheel, hidden=True, layout=layout)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, competition=True, agent="boot-fail")
    assert run.record["error"] is None, run.output
    assert run.record["AGENT_READY"] is True
    assert fake_agent(run)["model_dir"].endswith("qwen38-27b-int4")


def test_an_agent_only_build_skips_nvarc_and_the_agent_fills_from_placeholders(built, tmp_path):
    """NVARC's soft deadline at 600 s leaves the fallback no time, so NVARC never runs."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["agent-only-short"], root)
    assert run.record["error"] is None, run.output
    assert fake_solver(run) is None
    assert "less than 1800 s" in agent_stage(run)["nvarc_fallback"]["skipped"]
    assert run.record["STAGES"]["run"]["skipped"] == "agent only: AGENT['nvarc_share_s'] is 0"
    agent = fake_agent(run)
    assert "--nvarc-candidates" not in agent["argv"] and agent["candidates"] is None
    entries = [entry for task in agent["nvarc"].values() for entry in task]
    assert all(entry["attempt_1"] == [[0]] for entry in entries)
    written = assert_valid(run)
    # The fake agent solves the first open output, here the first task's.
    assert solved(written, sorted(DEV4)[0])
    assert agent_stage(run)["score"]["final"]["score"] == 1.0


# The agent-only build's fallback: NVARC runs in the time the agent stage leaves.


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
@pytest.mark.parametrize(
    ("agent", "install"), [("solve", "fail"), ("boot-fail", "ok"), ("crash", "ok")],
    ids=["failed-install", "failed-boot", "crash"],
)  # fmt: skip
def test_an_agent_only_build_falls_back_to_nvarc_when_the_agent_stage_ends_early(
    built, tmp_path, competition, agent, install
):
    """NVARC did not run; the agent stage ended with hours left, so NVARC's stage runs with the
    deadlines it would have had: the engine stopped, the GPUs logged before and after, and P6.2's
    selection of the agent's attempt log with NVARC's attempts written; valid throughout."""
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    holder = {}
    with no_stray_engine(holder):
        holder["run"] = run = simulate(
            tmp_path, built.notebooks["agent-only"], root, competition, agent=agent,
            install=install,
        )  # fmt: skip
        assert run.record["error"] is None, run.output
        assert run.record["STAGES"]["run"]["skipped"] == "agent only: AGENT['nvarc_share_s'] is 0"
        stage = agent_stage(run)
        fallback = stage["nvarc_fallback"]
        end_s = nbb.COMPETITION_SOLVER_END_S if competition else 41400
        assert (fallback["solver_end_s"], fallback["solver_exit"]) == (end_s, 0)
        assert fake_solver(run)["end_time"] - run.record["NOTEBOOK_START"] == pytest.approx(end_s)
        assert fallback["gpus_before"] == [] and fallback["gpus"]["after"] == []
        assert fallback["gpus"]["used_max_mib"] == nbb.GPU_USED_MAX_MIB
        assert "--query-compute-apps=pid,used_memory --format=csv" in run.output
        assert "--query-gpu=index,memory.used --format=csv,noheader,nounits" in run.output
        records = 1 if agent == "crash" else 0
        assert fallback["selection"]["records"] == records
        assert fallback["selection"]["slots"]["program"] == records
        assert run.record["SOLVER_EXIT"] == 0
        counts = HIDDEN if competition else DEV4
        written = assert_valid(run, counts)
        nvarc_tasks = ["f0000001", "f0000002"] if competition else NVARC_TASKS
        for task in nvarc_tasks:
            assert all(entry["attempt_1"] == GRID for entry in written[task]), task
        if agent == "crash":
            assert gone(engine_pid(run)), "the engine outlived the agent stage"


# NVARC's attempts in the fallback, wrong where the agent verified a program.
WRONG = [[4, 3], [2, 1]]


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_a_verified_answer_keeps_attempt_1_when_a_killed_agent_falls_back_to_nvarc(
    built, tmp_path, competition
):
    """The review's case: the agent verifies one task, logs its record (then a record cut
    mid-line), and dies by SIGKILL with no run.json. The fallback runs NVARC, whose attempt for
    that task is wrong; P6.2's selection of the attempt log puts the verified answer in attempt 1
    and NVARC's in attempt 2, and NVARC fills the tasks the agent left."""
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    holder = {}
    with no_stray_engine(holder):
        holder["run"] = run = simulate(
            tmp_path, built.notebooks["agent-only"], root, competition, agent="crash",
            env={"ARC26_FAKE_SOLVER_GRID": json.dumps(WRONG)},
        )  # fmt: skip
        assert run.record["error"] is None, run.output
        stage = agent_stage(run)
        assert stage["agent_exit"] == -signal.SIGKILL and "run" not in stage
        assert not (run.scratch / "logs" / "run.json").exists()
        fallback = stage["nvarc_fallback"]
        assert fallback["solver_exit"] == 0
        assert fallback["selection"]["records"] == 1
        assert fallback["selection"]["skipped_lines"] == 1
        assert fallback["selection"]["slots"]["program"] == 1
        counts = HIDDEN if competition else DEV4
        written = assert_valid(run, counts)
        verified, other = sorted(counts)[:2]
        assert written[verified] == [{"attempt_1": GRID, "attempt_2": WRONG}]
        assert all(entry["attempt_1"] == WRONG for entry in written[other])
        assert gone(engine_pid(run)), "the engine outlived the agent stage"
        if not competition:
            assert stage["score"]["final"]["solved"] == [f"{verified}_0"]


def test_the_fallback_needs_time_before_nvarcs_soft_deadline(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["agent-only-short"], root, install="fail")
    assert run.record["error"] is None, run.output
    fallback = agent_stage(run)["nvarc_fallback"]
    assert "s are left before NVARC's soft deadline, less than 1800 s" in fallback["skipped"]
    assert fake_solver(run) is None
    assert_valid(run)


def test_a_hybrid_build_runs_no_fallback_since_nvarc_ran(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, install="fail")
    assert run.record["error"] is None, run.output
    assert "nvarc_fallback" not in agent_stage(run)
    assert assert_valid(run) == nvarc_file(run)


HELD_GPU_SMI = """#!/bin/sh
case "$*" in
  *--query-compute-apps*)
    echo "pid, used_gpu_memory [MiB]"
    if [ ! -e "$ARC26_FAKE_GPU_FREED" ]; then
      echo "4242, 15102 MiB"
      : > "$ARC26_FAKE_GPU_FREED"
    fi
    ;;
  *)
    for i in 1 2 3 4; do echo "NVIDIA L4, 23034 MiB, 8.9, 580.159.04"; done
    ;;
esac
"""


# A container's nvidia-smi lists no process while the memory is still held: memory.used is 15,102
# MiB on every GPU at the first query, a few MiB from the second on.
HELD_MEMORY_SMI = """#!/bin/sh
case "$*" in
  *--query-compute-apps*)
    echo "pid, used_gpu_memory [MiB]"
    ;;
  *memory.used*)
    used=3
    if [ ! -e "$ARC26_FAKE_GPU_FREED" ]; then
      used=15102
      : > "$ARC26_FAKE_GPU_FREED"
    fi
    for i in 0 1 2 3; do echo "$i, $used"; done
    ;;
  *)
    for i in 1 2 3 4; do echo "NVIDIA L4, 23034 MiB, 8.9, 580.159.04"; done
    ;;
esac
"""


def test_gpu_memory_still_held_without_a_listed_process_is_waited_for(built, tmp_path):
    """The process list is empty, but memory.used shows the memory NVARC's workers held: the
    stage logs both, queries again 5 s later, finds every GPU under GPU_USED_MAX_MIB, and goes
    on."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(
        tmp_path, built.notebooks["hybrid"], root, smi=HELD_MEMORY_SMI,
        env={"ARC26_FAKE_GPU_FREED": str(tmp_path / "freed")},
    )  # fmt: skip
    assert run.record["error"] is None, run.output
    gpus = agent_stage(run)["gpus_before_engine"]
    assert gpus["before"] == [] and gpus["after"] == []
    assert gpus["memory_before"] == [{"index": i, "used_mib": 15102} for i in range(4)]
    assert gpus["memory_after"] == [{"index": i, "used_mib": 3} for i in range(4)]
    assert gpus["used_max_mib"] == nbb.GPU_USED_MAX_MIB and gpus["seconds"] >= 5
    assert run.output.count("--query-gpu=index,memory.used --format=csv,noheader,nounits") == 2
    assert agent_stage(run)["agent_exit"] == 0


def test_the_gpus_nvarc_held_are_logged_and_waited_for_before_the_engine_boots(built, tmp_path):
    """nvidia-smi still lists a process (an NVARC worker killed a moment before) on its first
    query; the stage logs it, queries again 5 s later, finds the GPUs free, and goes on."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(
        tmp_path, built.notebooks["hybrid"], root, smi=HELD_GPU_SMI,
        env={"ARC26_FAKE_GPU_FREED": str(tmp_path / "freed")},
    )  # fmt: skip
    assert run.record["error"] is None, run.output
    gpus = agent_stage(run)["gpus_before_engine"]
    assert gpus["before"] == [{"pid": "4242", "used_memory": "15102 MiB"}]
    assert gpus["after"] == [] and gpus["seconds"] >= 5
    assert run.output.count("--query-compute-apps=pid,used_memory --format=csv") == 2
    assert agent_stage(run)["agent_exit"] == 0


def test_a_development_run_on_a_subset_passes_the_task_ids_and_keeps_every_task(built, tmp_path):
    """The agent writes the subset's tasks only; the final check takes NVARC's file for the
    others, so the file stays valid for the whole challenges file."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["subset"], root)
    assert run.record["error"] is None, run.output
    agent = fake_agent(run)
    assert agent["argv"][agent["argv"].index("--task-ids") + 1] == "0934a4d8,981571dc"
    written = assert_valid(run)
    final = agent_stage(run)["final"]
    assert final["rewritten"] is True and final["sources"]["fallback"] > 0
    assert solved(written, "981571dc")
    assert agent_stage(run)["score"]["final"]["tasks"] == 2


# The agent fails: at the install, at the boot, mid-run, at the deadline. Each case leaves a
# valid file with NVARC's attempts, and the notebook ends cleanly.


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
@pytest.mark.parametrize("install", ["fail", "no-engine", "hang"])
def test_a_failed_install_leaves_nvarcs_file_and_the_notebook_ends_cleanly(
    built, tmp_path, competition, install
):
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    counts = HIDDEN if competition else DEV4
    overrides = {"install_timeout_s": 2} if install == "hang" else None
    run = simulate(
        tmp_path, built.notebooks["hybrid"], root, competition, install=install, overrides=overrides
    )
    assert run.record["error"] is None, run.output
    stage = agent_stage(run)
    assert stage["failure"] == "the engine install failed (see install), so the agent did not run"
    assert stage["install"]["ok"] is (install == "no-engine")
    if install == "hang":
        assert stage["install"]["ended"] in {"terminated", "killed"}
        assert stage["install"]["seconds"] < 20
    assert fake_agent(run) is None
    written = assert_valid(run, counts)
    assert written == nvarc_file(run)
    assert stage["final"]["sources"]["changed"] == 0 and stage["final"]["rewritten"] is False
    assert run.record["cells"][-1].endswith("-logs")


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_an_engine_that_never_boots_leaves_nvarcs_attempts(built, tmp_path, competition):
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, competition, agent="boot-fail")
    assert run.record["error"] is None, run.output
    stage = agent_stage(run)
    assert stage["agent_exit"] == 0
    assert stage["run"]["problems"] == [
        "engine start 1: the engine exited with code 1 during its boot"
    ]
    assert assert_valid(run, HIDDEN if competition else DEV4) == nvarc_file(run)


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_an_agent_that_crashes_mid_run_keeps_its_last_write_and_its_engine_is_stopped(
    built, tmp_path, competition
):
    """SIGKILL mid-run: the engine the agent started in a session of its own is stopped through
    its pid file, the cut temporary file is removed, and the file holds the agent's last atomic
    write over NVARC's attempts."""
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    holder = {}
    with no_stray_engine(holder):
        holder["run"] = run = simulate(
            tmp_path, built.notebooks["hybrid"], root, competition, agent="crash"
        )
        assert run.record["error"] is None, run.output
        stage = agent_stage(run)
        assert stage["agent_exit"] == -signal.SIGKILL and stage["ended"] == "exited"
        assert stage["engine_stopped"].startswith("stopped the engine's process group")
        assert gone(engine_pid(run)), "the engine outlived the agent stage"
        assert stage["removed"].endswith(".submission.json.tmp")
        assert not (root / "working" / ".submission.json.tmp").exists()
        counts = HIDDEN if competition else DEV4
        written = assert_valid(run, counts)
        assert_nvarc_kept(run, counts)
        assert solved(written, "f0000003" if competition else AGENT_OUTPUT[0])
        expected = ["submission.json"] if competition else ["run_logs", "submission.json"]
        assert working_files(run) == expected


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_a_corrupt_agent_file_is_replaced_by_nvarcs(built, tmp_path, competition):
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, competition, agent="corrupt")
    assert run.record["error"] is None, run.output
    stage = agent_stage(run)
    assert stage["agent_exit"] == 1 and stage["final"]["rewritten"] is True
    assert stage["final"]["sources"]["current"] == 0
    assert assert_valid(run, HIDDEN if competition else DEV4) == nvarc_file(run)


def test_an_invalid_agent_file_keeps_its_valid_attempts_and_takes_nvarcs_for_the_rest(
    built, tmp_path
):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, agent="invalid")
    assert run.record["error"] is None, run.output
    written = assert_valid(run)
    assert written == nvarc_file(run)
    sources = agent_stage(run)["final"]["sources"]
    # The first task's attempt_2 was invalid, the last task missing: 3 attempts from NVARC.
    assert sources["fallback"] == 3 and agent_stage(run)["final"]["rewritten"] is True


def test_a_refused_agent_changes_nothing(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, agent="refused")
    assert run.record["error"] is None, run.output
    stage = agent_stage(run)
    assert stage["agent_exit"] == 2 and stage["final"]["rewritten"] is False
    assert assert_valid(run) == nvarc_file(run)


@pytest.mark.parametrize(
    ("mode", "ended", "code"),
    [("hang-term", "terminated", 143), ("hang-kill", "killed", -signal.SIGKILL)],
)
@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_deadline_stops_the_agent_and_its_engine_and_the_file_stays_valid(
    built, tmp_path, competition, mode, ended, code
):
    """At AGENT['end_s'] the agent's group gets SIGTERM; one that ignores it gets SIGKILL
    AGENT['term_s'] seconds later. Either way the engine is stopped and the file is valid, with
    the writes the agent made before (and, honoring SIGTERM, during) its stop."""
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    holder = {}
    with no_stray_engine(holder):
        began = time.monotonic()
        holder["run"] = run = simulate(
            tmp_path, built.notebooks["hybrid"], root, competition, agent=mode,
            overrides={"end_in": 4, "term_s": 2},
        )  # fmt: skip
        assert run.record["error"] is None, run.output
        stage = agent_stage(run)
        assert (stage["ended"], stage["agent_exit"]) == (ended, code)
        assert gone(engine_pid(run)), "the engine outlived the deadline"
        assert stage["engine_stopped"].startswith("stopped the engine's process group")
        counts = HIDDEN if competition else DEV4
        written = assert_valid(run, counts)
        assert_nvarc_kept(run, counts)
        first = ("f0000003", 0) if competition else AGENT_OUTPUT
        assert solved(written, *first)
        if mode == "hang-term" and not competition:
            # On SIGTERM the agent wrote once more: the second open output's attempt_2.
            assert written["aa4ec2a5"][0]["attempt_2"] == [[4, 3], [2, 1]]
        assert stage["run_s"] < 4 + 2 + 10 and time.monotonic() - began < 120


def test_no_stage_starts_without_the_time_it_needs(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(
        tmp_path, built.notebooks["hybrid"], root, overrides={"end_in": 30, "min_stage_s": 600}
    )
    assert run.record["error"] is None, run.output
    stage = agent_stage(run)
    assert "less than AGENT['min_stage_s'] (600 s)" in stage["skipped"]
    assert "install" not in stage and fake_agent(run) is None
    assert assert_valid(run) == submission(run)
    assert not (run.scratch / "nvarc_submission.json").exists()


def test_the_cpu_check_runs_the_agent_stage_as_a_probe_without_gpus(built, tmp_path):
    """The hybrid's CPU check: no GPU and no driver are required; the inputs are found and
    checked, the install runs, the agent starts with the notebook's command line, and an engine
    that cannot boot leaves NVARC's file."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["cpu"], root, gpus=0, agent="boot-fail")
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and run.record["AGENT_READY"] is True
    check = run.record["STAGES"]["agent_check"]
    assert check["gpus"] == 0 and check["driver"] == "nvidia-smi lists no GPU, so no driver version"
    assert "CPU build: the agent stage runs as a launch probe" in run.output
    stage = agent_stage(run)
    assert stage["install"]["ok"] is True and stage["agent_exit"] == 0
    agent = fake_agent(run)
    assert agent["task_ids"] == ["aa4ec2a5"]
    assert agent["deadline"] == pytest.approx(run.record["NOTEBOOK_START"] + 3600)
    assert assert_valid(run) == nvarc_file(run)


# The agent check: a development run stops before anything is installed or solved; a competition
# rerun skips the agent and gives NVARC its full deadlines.

CHECK_FAILURES = {
    "no-wheelhouse": ({"missing": (WHEELS_NOTEBOOK,)}, {}, "0 distinct folders of the wheelhouse"),
    "no-weights": ({"missing": (MODEL_NOTEBOOK,)}, {}, "0 distinct folders of the weights"),
    "weights-not-ok": ({"model": {"ok": False}}, {}, "does not say ok: the download did not pass"),
    "other-revision": ({"model": {"revision": "0" * 40}}, {}, "names revision '0000"),
    "short-file": (
        {"model": {"files": [{"name": "config.json", "size": 3}]}}, {}, "config.json has 2 bytes"
    ),
    "other-vllm": (
        {"wheels": {"files": [{"name": "vllm", "version": "0.30.0"}]}}, {}, "lists vllm '0.30.0'"
    ),
    "old-driver": ({}, {"gpu": OLD_DRIVER_L4}, "the NVIDIA driver is 535.104.05, older than"),
    "two-gpus": ({}, {"gpus": 2}, "the engine needs 4 GPUs (its tensor parallelism)"),
}  # fmt: skip


@pytest.mark.parametrize("case", sorted(CHECK_FAILURES))
def test_a_failed_agent_check_stops_a_development_run_before_the_install(built, tmp_path, case):
    """Two GPUs stop it one cell earlier: NVARC's four workers need four (the environment
    check); every other case stops in the agent check."""
    mount, hardware, message = CHECK_FAILURES[case]
    root = make_tree(tmp_path, built.wheel, **mount)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, **hardware)
    if case == "two-gpus":
        assert "the solver needs 4 GPUs, one per worker" in run.record["error"]
        assert run.record["cells"][-1].endswith("-validator")
    else:
        assert "RuntimeError: agent check: " in run.record["error"]
        assert message in run.record["error"]
        assert run.record["cells"][-1].endswith("-environment")
    assert LOCAL.install_calls(run) == [] and fake_solver(run) is None
    # The placeholder written by the environment check is valid.
    assert submit.validate(submission(run), LOCAL.standin_challenges()) == []


@pytest.mark.parametrize("case", sorted(CHECK_FAILURES))
def test_a_failed_agent_check_in_a_rerun_gives_nvarc_the_whole_session(built, tmp_path, case):
    mount, hardware, message = CHECK_FAILURES[case]
    root = make_tree(tmp_path, built.wheel, hidden=True, **mount)
    run = simulate(tmp_path, built.notebooks["hybrid"], root, competition=True, **hardware)
    assert run.record["error"] is None, run.output
    assert run.record["AGENT_READY"] is False
    assert [p for p in run.record["PROBLEMS"] if p.startswith("agent check: ") and message in p]
    start = run.record["NOTEBOOK_START"]
    assert fake_solver(run)["end_time"] - start == pytest.approx(nbb.COMPETITION_SOLVER_END_S)
    stage = agent_stage(run)
    assert stage["skipped"] == "the agent check did not pass (stages.agent_check)"
    assert fake_agent(run) is None
    assert_valid(run, HIDDEN)
    assert working_files(run) == ["submission.json"]


def config_edited(notebook, dest, old, new):
    """A copy of a built notebook whose Config cell a human edited, ``old`` replaced by ``new``."""
    data = json.loads(notebook.read_text(encoding="utf-8"))
    (cell,) = [c for c in data["cells"] if c["id"] == "arc26-02-config"]
    text = "".join(cell["source"])
    assert text.count(old) == 1
    cell["source"] = text.replace(old, new).splitlines(keepends=True)
    out = dest / f"edited-{notebook.name}"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_an_agent_edited_badly_stops_a_development_run_and_leaves_a_rerun_to_nvarc(
    built, tmp_path, competition
):
    """AGENT without a key: the agent check's stage fails, before NVARC; a development run stops
    there, and a competition rerun logs it, gives NVARC its full deadlines, and ends valid."""
    notebook = config_edited(
        built.notebooks["hybrid"], tmp_path, "'install_timeout_s': 60}", "}"
    )
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    run = simulate(tmp_path, notebook, root, competition)
    if not competition:
        assert "KeyError: 'install_timeout_s'" in run.record["error"]
        assert run.record["cells"][-1].endswith("-environment")
        return
    assert run.record["error"] is None, run.output
    assert run.record["AGENT_READY"] is False
    assert "KeyError" in run.record["STAGES"]["agent_check"]["error"]
    start = run.record["NOTEBOOK_START"]
    assert fake_solver(run)["end_time"] - start == pytest.approx(nbb.COMPETITION_SOLVER_END_S)
    assert agent_stage(run)["skipped"] == "the agent check did not pass (stages.agent_check)"
    assert_valid(run, HIDDEN)


def test_an_agent_stage_switched_off_checks_nothing_and_nvarc_keeps_its_deadlines(
    built, tmp_path
):
    root = LOCAL.make_tree(tmp_path, built.wheel)  # no download outputs mounted
    run = simulate(tmp_path, built.notebooks["disabled"], root)
    assert run.record["error"] is None, run.output
    assert run.record["STAGES"]["agent_check"]["skipped"] == "AGENT['enabled'] is False"
    assert fake_solver(run)["end_time"] - run.record["NOTEBOOK_START"] == pytest.approx(41400)
    assert agent_stage(run)["skipped"] == "AGENT['enabled'] is False"
    assert assert_valid(run) == submission(run)


# The real agent (task P6.4) with its fake engine, once arc26.agent.run is merged.

# The stand-in tasks map each grid to itself: the identity reproduces every demonstration. The
# fake engine answers the task its first message names; the other tasks get no code.
IDENTITY = "```python\ndef transform(grid):\n    return [list(row) for row in grid]\n```"


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_real_agent_runs_with_the_notebooks_arguments_and_its_fake_engine(
    built, tmp_path, competition
):
    """The notebook's command line through the real arc26.agent.run: the run config as written
    (its port moved to a free one), the venv's vllm started as the engine (here P6.4's fake
    engine), NVARC's files, and the absolute deadline; a verified program solves the output
    NVARC left open, and the file keeps NVARC's attempts."""
    pytest.importorskip("arc26.agent.run")
    server = pytest.importorskip("arc26.agent.server")
    data = json.loads(built.notebooks["hybrid"].read_text())
    port = server.free_port()
    for cell in data["cells"]:
        text = "".join(cell["source"])
        if cell["id"].endswith("-agent-check"):
            # A free port; and a final margin of 10 s, so the run stops itself (phase 4 cycles
            # until the generation stop) well before the notebook's SIGTERM at end_s.
            edits = (
                ("'port': 8000", f"'port': {port}"),
                ("'final_margin_s': 1,", "'final_margin_s': 10,"),
            )
            for old, new in edits:
                assert text.count(old) == 1, old
                text = text.replace(old, new)
            cell["source"] = text.splitlines(True)
    edited = tmp_path / "edited.ipynb"
    edited.write_text(json.dumps(data))
    open_task = "f0000003" if competition else AGENT_OUTPUT[0]
    replies = tmp_path / "replies.json"
    replies.write_text(json.dumps({open_task: [IDENTITY]}))
    root = make_tree(tmp_path, built.wheel, hidden=competition)
    run = simulate(
        tmp_path, edited, root, competition, agent=None,
        env={"ARC26_FAKE_REPLIES": str(replies)}, overrides={"end_in": 25},
    )  # fmt: skip
    assert run.record["error"] is None, run.output
    stage = agent_stage(run)
    assert (stage["agent_exit"], stage["ended"]) == (0, "exited"), run.output
    assert stage["run_s"] < 25
    counts = HIDDEN if competition else DEV4
    written = assert_valid(run, counts)
    assert_nvarc_kept(run, counts)
    assert solved(written, open_task)
    summary = json.loads((run.scratch / "logs" / "run.json").read_text())
    assert summary["mode"] == ("competition" if competition else "dev")
    assert summary["problems"] == [] and summary["submission"]["valid"] is True
    assert summary["submission"]["slots"]["program"] == 1
    assert summary["engine"]["kind"] == "vllm" and summary["engine"]["exit_code"] == 0
    assert not (run.scratch / "logs" / "engine.pid").exists()
    if competition:
        assert working_files(run) == ["submission.json"]
        assert not (run.scratch / "logs" / "transcripts.jsonl").exists()
    else:
        assert stage["score"]["gained"] == [f"{open_task}_0"]
