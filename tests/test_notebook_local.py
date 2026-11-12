"""Simulated-Kaggle run of the generated submission notebook (task P0.5, deliverable 5; BRIEF A5).

The notebook is built by scripts/build_kaggle_notebook.py from configs/submission_cpu_check.yaml
and a wheel of the working tree's ``arc26`` package: with ``RUN_SOLVER`` false for the plumbing
tests, and as committed (the launch probe) for the probe test. Its code cells then run
in order in a separate interpreter, sharing one namespace as in a Jupyter kernel, against a
simulated ``/kaggle`` tree in a temporary folder (``ARC26_KAGGLE_ROOT``). The stand-in challenges
hold the DEV4 ids with 1, 2, 1, and 1 test inputs and dummy grids, never competition data. On
``PATH``, a recording ``python`` shim logs every call and "installs" the wheel by unpacking it into
a temporary target, and a fake ``nvidia-smi`` reports one L4 (four where a test says so, and T4s,
compute capability 7.5, for the bf16 stop of task P0.5d). Both modes run: development, and a
competition rerun (``KAGGLE_IS_COMPETITION_RERUN=1``).

Script notebooks (task P0.5d) run the same way, from ``configs/script_smoke.yaml`` with its script
replaced by ``tests/testdata/tiny_script.py`` and its config, which the tree holds beside the wheel.
Run ``P3.2-01``'s build runs the real ``scripts/cache_embeddings.py`` and its config on a stand-in
base model when torch, transformers, and safetensors are installed (task P3.2).

The kernel runs with ``-S``, so the repository's editable install is invisible and ``arc26`` can
only come from the wheel the notebook installs. An audit hook records every file opened under the
simulated root, and an import hook records every attempt to import torch, tensorflow, unsloth, or
transformers.

Run ``P3.5-02``'s build runs the real ``scripts/train_hypernet.py`` with its config in smoke mode,
on the episode fixture mounted as the two attached P3.4 outputs, when torch, transformers, and
safetensors are installed (task P3.5).
"""

import bz2
import contextlib
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import time
import types
import zipfile
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

import arc26.submit as submit
from arc26.data import DEV40

REPO = Path(__file__).resolve().parent.parent


# Without a GPU stack the launch stops at its first import: torch when it is absent (the main CI
# job), unsloth when torch is installed (the hyper tests' environment, Kaggle's CPU image).
GPU_STACK_MISSING = re.compile(r"No module named '(torch|unsloth)'")

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nbb = load("build_kaggle_notebook", REPO / "scripts" / "build_kaggle_notebook.py")

DEV4 = {"0934a4d8": 1, "36a08778": 2, "981571dc": 1, "aa4ec2a5": 1}
GRID = [[1, 2], [3, 4]]
COMPETITION = "arc-prize-2026-arc-agi-2"
MODEL_DIR = ("models", "sorokin", "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
# The dataset mount layout is unverified for 2026 (research note, section 3.6): both are tried.
DATASET_DIRS = {
    "2026": ("datasets", "alyahya737", "arc26-code"),
    "legacy": ("arc26-code",),
}
WHEEL_NAME = "arc26-0.0.1-py3-none-any.whl"
CODE_CELLS = [
    "arc26-02-config",
    "arc26-04-helpers",
    "arc26-06-validator",
    "arc26-08-environment",
    "arc26-10-install",
    "arc26-12-run",
    "arc26-14-write",
    "arc26-16-logs",
]
STAGES = ["environment", "install", "run", "write", "run_logs"]
# What run_solver adds to the kernel's environment: the reference's launch variables (its cell 6),
# PYTHONHASHSEED from the config, and PYTHONUNBUFFERED.
LAUNCH_ENV = {
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "OMP_NUM_THREADS": "12",
    "PYTHONHASHSEED": "0",
    "PYTHONUNBUFFERED": "1",
}
NEVER_IMPORTED = {"torch", "tensorflow", "unsloth"}
# The reference's LoRA settings (cell 4), which RunConfig passes to the solver by default.
NINE = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
]  # fmt: skip
SERIES_CONFIG = "submission_P1.3-01.yaml"
# The split solver's parity run (task P2.0): P0.5-01's settings with SOLVER = 'split'.
SPLIT_CONFIG = "submission_P2.0-01.yaml"
# The held-out gate's price run (task P2.1): one hold-out arm on DEV40 with the split solver.
HOLDOUT_CONFIG = "submission_P2.1-01.yaml"
HOLDOUT_ARM = "r256-proj-emb-holdout"
HOLDOUT = {"rule": "last", "k2": True, "in_test_prompts": False}
# Two tiny arms for the series plumbing: the notebook configuration first, as the builder
# requires, then rank 1 on one projection.
TINY_ARMS = (
    nbb.Arm("ref", 256, tuple(NINE), "the notebook configuration"),
    nbb.Arm("tiny", 1, ("q_proj",), "rank 1 on q_proj only"),
)
FIXTURE = REPO / "tests" / "fixtures" / "p0_1_reference_dev4.json"
CANDIDATES_A = REPO / "tests" / "fixtures" / "p0_1_reference_candidates_A.json"
# Script notebooks (task P0.5d): the CPU check's config, its script and config replaced by the
# stand-in script, which writes run_logs/result.json.
SCRIPT_CONFIG = "script_smoke.yaml"
TINY_SCRIPT = REPO / "tests" / "testdata" / "tiny_script.py"
TINY_CONFIG = REPO / "tests" / "testdata" / "tiny_script.yaml"
COMMIT = "0" * 40
SCRIPT_CODE_CELLS = [
    "arc26-02-config",
    "arc26-04-helpers",
    "arc26-06-environment",
    "arc26-08-install",
    "arc26-10-script",
    "arc26-12-logs",
]
SCRIPT_STAGES = ["environment", "install", "script", "run_logs"]
# Outputs of other notebooks attached to a script build (task P3.5), as P3.5-01 attaches P3.4's.
EPISODES, CHECK = "alyahya737/arc26-p3-4-episodes", "alyahya737/arc26-p3-4-leakage"
NOTEBOOK_INPUTS = {EPISODES: 2, CHECK: 2}
INPUT_ARGS = ("--note", "{input:" + EPISODES + "}", "--note", "{input:" + CHECK + "}")
# Where Kaggle mounts another notebook's output is unverified for 2026 (datasets moved under
# datasets/<owner>/), and the install notebook, a kernel source, is mounted under
# usr/lib/notebooks/<owner>/ with its slug's hyphens as underscores (P0.5-01's session): each
# layout is tried. A layout is the folders above the output's, then whether underscores replace
# the slug's hyphens.
OUTPUT_DIRS = {
    "2026": (("input", "notebooks", "{owner}"), False),
    "legacy": (("input",), False),
    "usr-lib": (("usr", "lib", "notebooks", "{owner}"), True),
}
NOTEBOOK_LOGS = ["candidates.json", "env.json", "pip_freeze.txt", "stages.json"]
# Run P3.2-01's build (task P3.2): the encoder script and its config, as committed, on NvidiaL4.
P32_CONFIG = "script_P3.2-01.yaml"
CACHE_SCRIPT = REPO / "scripts" / "cache_embeddings.py"
CACHE_CONFIG = REPO / "configs" / "cache_embeddings.yaml"
P32_BRIEF = REPO / "docs" / "prerun" / "P3.2-01.md"
# Run P3.2-02's build (task P3.2): the same script with 8 views and the view diagnostic.
P3202_CONFIG = "script_P3.2-02.yaml"
CACHE_CONFIG_02 = REPO / "configs" / "cache_embeddings_P3.2-02.yaml"
P3202_BRIEF = REPO / "docs" / "prerun" / "P3.2-02.md"
# What cache_embeddings.py writes into run_logs/ beside the notebook's files.
CACHE_LOGS = [
    "config.yaml", "metrics.json", "run.json", "summary.md", "throughput.jsonl", "timing.jsonl",
]  # fmt: skip

DRIVER = r'''
import json, os, sys, time, traceback

notebook_path, record_path = sys.argv[1], sys.argv[2]
# Seconds from the start of a script notebook's Script cell to its hard stop, as a human could set
# SCRIPT_KILL_S in the Config cell; absent, the cell's value stands.
kill_in = float(sys.argv[3]) if len(sys.argv) > 3 else None
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
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.startswith(root):
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
            if kill_in is not None and cell["id"].endswith("-script"):
                namespace["SCRIPT_KILL_S"] = time.time() - namespace["NOTEBOOK_START"] + kill_in
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
            record["cells"].append(cell["id"])
except BaseException:
    record["error"] = traceback.format_exc()
finally:
    record["arc26"] = getattr(sys.modules.get("arc26"), "__file__", None)
    names = ("STAGES", "PROBLEMS", "GPUS", "CHALLENGES", "MODEL_DIR", "WHEEL", "SOLVER_EXIT")
    names += ("SCRIPT_EXIT", "INPUT_DIRS")
    for name in (*names, "NOTEBOOK_START"):
        record[name] = namespace.get(name)
    record["report"] = namespace.get("SUBMISSION_REPORT")
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, default=str)
'''

SHIM = r'''#!@PYTHON@ -S
import json, os, sys, zipfile

args = sys.argv[1:]
with open(os.environ["ARC26_SHIM_LOG"], "a", encoding="utf-8") as log:
    exists = os.path.exists(os.environ["ARC26_SHIM_SUBMISSION"])
    names = ("UNSLOTH_DISABLE_STATISTICS", "TRITON_PTXAS_PATH", "OMP_NUM_THREADS")
    names += ("PYTHONHASHSEED", "PYTHONUNBUFFERED")
    env = {name: os.environ[name] for name in names if name in os.environ}
    log.write(json.dumps({"argv": args, "submission_exists": exists, "env": env}) + "\n")
if args[:3] == ["-m", "pip", "uninstall"]:
    print("WARNING: Skipping tensorflow as it is not installed.")
    sys.exit(0)
if args[:3] == ["-m", "pip", "install"]:
    wheels = [arg for arg in args[3:] if arg.endswith(".whl")]
    if "--no-index" not in args or "--no-deps" not in args or len(wheels) != 1:
        print("shim: refused, not an offline install of one wheel", file=sys.stderr)
        sys.exit(1)
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(os.environ["ARC26_SHIM_TARGET"])
    print("Successfully installed arc26-0.0.1")
    sys.exit(0)
if args[:3] == ["-m", "pip", "freeze"]:
    print("arc26==0.0.1")
    sys.exit(0)
if args[:2] == ["-m", "pip"]:
    print("shim: unexpected pip command", file=sys.stderr)
    sys.exit(1)
os.execv("@PYTHON@", ["@PYTHON@", "-S", *args])
'''

# nvidia-smi's line per GPU for --query-gpu=name,memory.total,compute_cap,driver_version.
L4 = "NVIDIA L4, 23034 MiB, 8.9, 580.159.04"
T4 = "Tesla T4, 15360 MiB, 7.5, 580.159.04"


def nvidia_smi(gpus=1, gpu=L4, stray=None):
    """A fake nvidia-smi that lists ``gpus`` GPUs of one model: L4s, as Kaggle's 1xL4 or 4xL4
    machines would, or T4s (no bf16), as its default 2xT4 would; ``stray`` is a line printed
    first, as a driver warning on stdout."""
    first = "" if stray is None else f'echo "{stray}"\n'
    return "#!/bin/sh\n" + first + f'echo "{gpu}"\n' * gpus


def standin_challenges(counts=DEV4):
    return {
        task: {"train": [{"input": GRID, "output": GRID}], "test": [{"input": GRID}] * count}
        for task, count in counts.items()
    }


def standin_solutions(counts=DEV4):
    return {task: [GRID] * count for task, count in counts.items()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def make_wheel(dest):
    """A wheel of the working tree's arc26 package; the notebook only unpacks and imports it."""
    path = dest / WHEEL_NAME
    info = "arc26-0.0.1.dist-info"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((REPO / "arc26").rglob("*.py")):
            archive.write(file, file.relative_to(REPO).as_posix())
        archive.writestr(f"{info}/METADATA", "Metadata-Version: 2.1\nName: arc26\nVersion: 0.0.1\n")
        archive.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
        )
        archive.writestr(f"{info}/RECORD", "")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Built(NamedTuple):
    wheel: Path
    notebooks: dict


def build_notebook(dest, config_name, wheel, name, **changes):
    path = REPO / "configs" / config_name
    config = nbb.parse_config(path.read_bytes(), path)._replace(**changes)
    info = nbb.build_info(
        config,
        tag="local-test",
        commit=COMMIT,
        config_file=f"configs/{config_name}",
        config_sha256=sha256(path),
        wheel=wheel.name,
        wheel_sha256=sha256(wheel),
    )
    notebook = nbb.build_notebook(config, info, nbb.wheel_member(wheel))
    out = dest / f"{name}.ipynb"
    out.write_bytes(nbb.notebook_bytes(notebook))
    return out


def build_script_notebook(dest, wheel, name, **changes):
    """A script notebook of configs/script_smoke.yaml that runs the stand-in script."""
    path = REPO / "configs" / SCRIPT_CONFIG
    config = nbb.parse_config(path.read_bytes(), path)._replace(
        script="tests/testdata/tiny_script.py",
        script_config="tests/testdata/tiny_script.yaml",
        **changes,
    )
    info = nbb.build_info(
        config,
        tag="local-test",
        commit=COMMIT,
        config_file=f"configs/{SCRIPT_CONFIG}",
        config_sha256=sha256(path),
        wheel=wheel.name,
        wheel_sha256=sha256(wheel),
        script_sha256=sha256(TINY_SCRIPT),
        script_config_sha256=sha256(TINY_CONFIG),
    )
    out = dest / f"{name}.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_script_notebook(config, info)))
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The CPU-check notebook with the solver off (the brief's plumbing test), the same notebook
    as committed (the launch probe), the GPU notebook of P0.5-01, the timing run P0.3-01's
    notebook (four workers on 4xL4) as committed and with the solver off, the same notebook
    built here for one worker on 1xL4 (the builder's one-GPU path), and the scope series P1.3-01's
    notebook as committed, with the solver off and two tiny arms, and with three tiny arms for the
    stop rule (task P1.3), the split solver's P2.0-01 notebook as committed and with the
    solver off (task P2.0), and script notebooks of the stand-in script on CPU and on NvidiaL4,
    failing, and sleeping with a child (task P0.5d), and with two attached notebook outputs passed
    to the script (task P3.5)."""
    dest = tmp_path_factory.mktemp("build")
    wheel = make_wheel(dest)
    cpu, gpu = "submission_cpu_check.yaml", "submission_P0.5-01.yaml"
    timing = "submission_P0.3-01.yaml"
    three = (*TINY_ARMS, nbb.Arm("third", 8, ("v_proj",), "a third arm for the stop rule"))
    return Built(
        wheel,
        {
            "plain": build_notebook(dest, cpu, wheel, "plain", run_solver=False),
            "probe": build_notebook(dest, cpu, wheel, "probe"),
            "gpu": build_notebook(dest, gpu, wheel, "gpu"),
            "timing": build_notebook(dest, timing, wheel, "timing"),
            "timing-plain": build_notebook(dest, timing, wheel, "timing-plain", run_solver=False),
            "one-l4": build_notebook(
                dest, timing, wheel, "one-l4", accelerator="NvidiaL4X1", workers=1
            ),
            "series": build_notebook(dest, SERIES_CONFIG, wheel, "series"),
            "series-plain": build_notebook(
                dest, SERIES_CONFIG, wheel, "series-plain", run_solver=False, arms=TINY_ARMS
            ),
            "series-three": build_notebook(dest, SERIES_CONFIG, wheel, "series-three", arms=three),
            "split": build_notebook(dest, SPLIT_CONFIG, wheel, "split"),
            "split-plain": build_notebook(
                dest, SPLIT_CONFIG, wheel, "split-plain", run_solver=False
            ),
            "holdout": build_notebook(dest, HOLDOUT_CONFIG, wheel, "holdout"),
            "script": build_script_notebook(dest, wheel, "script", args=("--note", "{commit}")),
            "script-gpu": build_script_notebook(dest, wheel, "script-gpu", accelerator="NvidiaL4"),
            "script-fail": build_script_notebook(dest, wheel, "script-fail", args=("--exit", "3")),
            "script-sleep": build_script_notebook(
                dest, wheel, "script-sleep", args=("--spawn", "--sleep", "60")
            ),
            "script-inputs": build_script_notebook(
                dest, wheel, "script-inputs", args=INPUT_ARGS, notebook_inputs=NOTEBOOK_INPUTS
            ),
        },
    )


class Run(NamedTuple):
    record: dict
    calls: list
    root: Path
    target: Path
    output: str


def make_tree(tmp, wheel, layout="2026", decoys=False, wheel_bytes=None, data=True, extras=None):
    root = tmp / "kaggle"
    data_dir = root / "input" / "competitions" / COMPETITION
    if data:
        for name in ("arc-agi_evaluation_challenges.json", "arc-agi_test_challenges.json"):
            write_json(data_dir / name, standin_challenges())
        write_json(data_dir / "arc-agi_evaluation_solutions.json", standin_solutions())
    write_json(root.joinpath("input", *MODEL_DIR) / "config.json", {})
    if wheel is not None:
        dataset = root.joinpath("input", *DATASET_DIRS[layout])
        dataset.mkdir(parents=True)
        target = dataset / WHEEL_NAME
        shutil.copyfile(wheel, target)
        if wheel_bytes is not None:
            target.write_bytes(wheel_bytes)
        # A script build's script and config, beside the wheel (task P0.5d).
        for name, content in (extras or {}).items():
            (dataset / name).write_bytes(content)
    if decoys:
        # Another dataset holding an older challenges file and a model config of its own.
        other = root / "input" / "datasets" / "someone" / "old-arc"
        write_json(other / "arc-agi_evaluation_challenges.json", standin_challenges({"0" * 8: 1}))
        write_json(other / "arc-agi_test_challenges.json", standin_challenges({"0" * 8: 1}))
        write_json(other / "config.json", {})
    (root / "working").mkdir(parents=True)
    return root


def simulate(
    tmp,
    notebook,
    root,
    competition=False,
    outputs=None,
    gpus=1,
    gpu=L4,
    kill_in=None,
    stray=None,
    real_transformers=False,
):
    """Run a notebook's code cells against the simulated tree. ``real_transformers`` leaves out
    the transformers stub, for a script that loads a model with the real library (P3.2-01)."""
    bin_dir, target, stubs = tmp / "bin", tmp / "target", tmp / "stubs"
    for folder in (bin_dir, target, stubs, tmp / "home"):
        folder.mkdir(parents=True, exist_ok=True)
    if not real_transformers:
        # make_submission imports the moved arc_loader, which imports transformers.AutoTokenizer
        # for annotations only.
        (stubs / "transformers").mkdir(exist_ok=True)
        (stubs / "transformers" / "__init__.py").write_text("AutoTokenizer = None\n")
    for name, text in (
        ("python", SHIM.replace("@PYTHON@", sys.executable)),
        ("nvidia-smi", nvidia_smi(gpus, gpu, stray)),
    ):
        path = bin_dir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    if outputs is not None:
        outputs(root / "inference_outputs")
    driver, record_path, log = tmp / "driver.py", tmp / "record.json", tmp / "shim.jsonl"
    driver.write_text(DRIVER)
    env = {
        "PATH": str(bin_dir),
        # The target first, as Kaggle's site-packages; the stubs; then the test interpreter's
        # site-packages as a plain folder, for numpy (with -S its .pth files are not read, so the
        # editable install of the repository stays invisible).
        "PYTHONPATH": os.pathsep.join([str(target), str(stubs), sysconfig.get_paths()["purelib"]]),
        "ARC26_KAGGLE_ROOT": str(root),
        "ARC26_SHIM_LOG": str(log),
        "ARC26_SHIM_TARGET": str(target),
        "ARC26_SHIM_SUBMISSION": str(root / "working" / "submission.json"),
        "HOME": str(tmp / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        # The environment is built from scratch, so the caller's settings are restated: no
        # process of the simulated notebook sees a local GPU or reaches the Hugging Face Hub
        # (BRIEF A0).
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
    }
    if competition:
        env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    extra = [] if kill_in is None else [str(kill_in)]
    result = subprocess.run(
        [sys.executable, "-S", str(driver), str(notebook), str(record_path), *extra],
        cwd=root / "working",
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    record = json.loads(record_path.read_text())
    return Run(record, calls, root, target, result.stdout + result.stderr)


def submission(run):
    return json.loads((run.root / "working" / "submission.json").read_text())


def argvs(run):
    return [call["argv"] for call in run.calls]


def install_calls(run):
    return [argv for argv in argvs(run) if argv[:3] == ["-m", "pip", "install"]]


def solver_starts(run):
    return [call for call in run.calls if call["argv"][:2] == ["-m", "arc26.nvarc.starter"]]


def split_starts(run):
    return [call for call in run.calls if call["argv"][:2] == ["-m", "arc26.solver"]]


def assert_kernel_never_imported_torch(run):
    names = {name.partition(".")[0] for _, name in run.record["imports"]}
    assert not names & NEVER_IMPORTED, run.record["imports"]
    before_write = [
        name for cell, name in run.record["imports"] if CODE_CELLS.index(cell) < 6
    ]
    assert before_write == [], "the kernel imported transformers before the solver finished"


# Both modes, end to end


@pytest.mark.parametrize("layout", sorted(DATASET_DIRS))
def test_development_mode_runs_end_to_end(built, tmp_path, layout):
    root = make_tree(tmp_path, built.wheel, layout=layout)
    run = simulate(tmp_path, built.notebooks["plain"], root)
    assert run.record["error"] is None, run.output
    assert run.record["cells"] == CODE_CELLS
    assert run.record["PROBLEMS"] == []
    assert submit.validate(submission(run), standin_challenges()) == []
    assert run.record["report"]["valid"] and run.record["report"]["zero_attempts"] == 10
    assert list(run.record["STAGES"]) == STAGES
    for stage in run.record["STAGES"].values():
        assert "error" not in stage and stage["seconds"] >= 0
    assert run.record["GPUS"] == [
        {
            "name": "NVIDIA L4",
            "memory.total": "23034 MiB",
            "compute_cap": "8.9",
            "driver_version": "580.159.04",
        }
    ]
    assert run.record["CHALLENGES"].endswith(
        f"competitions/{COMPETITION}/arc-agi_evaluation_challenges.json"
    )
    assert run.record["MODEL_DIR"] == str(root.joinpath("input", *MODEL_DIR))
    assert_kernel_never_imported_torch(run)


def test_development_mode_installs_the_attached_wheel_offline(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["plain"], root)
    assert run.record["error"] is None, run.output
    wheel = str(root.joinpath("input", *DATASET_DIRS["2026"], WHEEL_NAME))
    assert argvs(run)[0] == ["-m", "pip", "uninstall", "-y", "tensorflow"]
    assert install_calls(run) == [
        [
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            "--disable-pip-version-check",
            wheel,
        ]
    ]
    assert ["-c", "import arc26; print(arc26.__file__)"] in argvs(run)
    # The placeholder submission exists before anything is installed.
    assert all(call["submission_exists"] for call in run.calls)
    assert run.record["arc26"].startswith(str(run.target))


def test_the_kernel_sets_the_hash_seed_even_with_the_solver_off(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["plain"], root)
    assert run.record["error"] is None, run.output
    env = json.loads((root / "working" / "run_logs" / "env.json").read_text())
    assert env["environment"]["PYTHONHASHSEED"] == "0"
    # The solver did not run, so there is no launch environment to record.
    assert "launch_env" not in run.record["STAGES"]["run"]


def test_development_mode_writes_run_logs(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["plain"], root)
    assert run.record["error"] is None, run.output
    logs = root / "working" / "run_logs"
    assert sorted(p.name for p in logs.iterdir()) == [
        "candidates.json",
        "env.json",
        "pip_freeze.txt",
        "stages.json",
    ]
    env = json.loads((logs / "env.json").read_text())
    stages = json.loads((logs / "stages.json").read_text())
    assert env["build"] == stages["build"] and env["build"]["wheel"] == WHEEL_NAME
    assert list(stages["stages"]) == STAGES and stages["mode"] == "development"
    assert stages["submission"]["valid"] and stages["problems"] == []
    assert stages["settings"]["run_solver"] is False and stages["solver_exit"] is None
    assert (logs / "pip_freeze.txt").read_text() == "arc26==0.0.1\n"
    assert ["-m", "pip", "freeze"] in argvs(run)


@pytest.mark.parametrize("layout", sorted(DATASET_DIRS))
def test_competition_mode_runs_end_to_end(built, tmp_path, layout):
    root = make_tree(tmp_path, built.wheel, layout=layout)
    run = simulate(tmp_path, built.notebooks["plain"], root, competition=True)
    assert run.record["error"] is None, run.output
    assert run.record["cells"] == CODE_CELLS and run.record["PROBLEMS"] == []
    assert run.record["CHALLENGES"].endswith("arc-agi_test_challenges.json")
    assert submit.validate(submission(run), standin_challenges()) == []
    assert len(install_calls(run)) == 1 and "--no-index" in install_calls(run)[0]
    assert run.record["arc26"].startswith(str(run.target))
    assert_kernel_never_imported_torch(run)


def test_competition_mode_writes_only_the_submission_and_never_reads_solutions(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["plain"], root, competition=True)
    assert run.record["error"] is None, run.output
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]
    opened = [path for _, path in run.record["opened"]]
    assert opened and not [path for path in opened if "solutions" in path]
    assert ["-m", "pip", "freeze"] not in argvs(run)


# Inputs are found by search, and problems stop development runs only


def test_decoy_inputs_elsewhere_are_not_used(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, decoys=True)
    run = simulate(tmp_path, built.notebooks["plain"], root)
    assert run.record["error"] is None, run.output
    assert list(submission(run)) == list(DEV4)
    assert run.record["MODEL_DIR"] == str(root.joinpath("input", *MODEL_DIR))


def test_missing_wheel_stops_a_development_run_after_the_placeholder(built, tmp_path):
    root = make_tree(tmp_path, None)
    run = simulate(tmp_path, built.notebooks["plain"], root)
    assert "RuntimeError: environment check: 0 wheel candidates" in run.record["error"]
    assert run.record["cells"] == CODE_CELLS[:3]
    assert install_calls(run) == []
    assert submission(run) == submit.repair(None, standin_challenges())


def test_missing_wheel_in_a_competition_rerun_still_ends_with_a_valid_file(built, tmp_path):
    root = make_tree(tmp_path, None)
    run = simulate(tmp_path, built.notebooks["plain"], root, competition=True)
    assert run.record["error"] is None, run.output
    assert run.record["cells"] == CODE_CELLS
    assert [p for p in run.record["PROBLEMS"] if "wheel candidates" in p]
    assert install_calls(run) == [] and run.record["arc26"] is None
    assert submit.validate(submission(run), standin_challenges()) == []


def test_wrong_wheel_hash_stops_a_development_run(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, wheel_bytes=built.wheel.read_bytes() + b"\0")
    run = simulate(tmp_path, built.notebooks["plain"], root)
    assert "has sha256" in run.record["error"] and "the build baked" in run.record["error"]
    assert install_calls(run) == []


def test_wrong_wheel_hash_in_a_competition_rerun_is_logged_and_installed(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, wheel_bytes=built.wheel.read_bytes() + b"\0")
    run = simulate(tmp_path, built.notebooks["plain"], root, competition=True)
    assert run.record["error"] is None, run.output
    assert [p for p in run.record["PROBLEMS"] if "has sha256" in p]
    assert len(install_calls(run)) == 1
    assert submit.validate(submission(run), standin_challenges()) == []


def test_missing_data_in_a_competition_rerun_is_logged_not_raised(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, data=False)
    run = simulate(tmp_path, built.notebooks["plain"], root, competition=True)
    assert run.record["error"] is None, run.output
    assert run.record["cells"] == CODE_CELLS
    assert [p for p in run.record["PROBLEMS"] if "challenges" in p]


# The Run cell, in process, with a stand-in arc26.runner


def notebook_code(path):
    notebook = json.loads(path.read_text())
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}


def run_cell_namespace(built, tmp_path, monkeypatch, run_solver, competition=False, notebook="gpu"):
    calls = []

    class RunConfig:
        def __init__(self, **fields):
            self.fields = fields
            self.hash_seed = fields["hash_seed"]

    real = pytest.importorskip("arc26.runner")
    runner = types.ModuleType("arc26.runner")
    runner.RunConfig = RunConfig
    runner.solver_env = real.solver_env
    runner.run_solver = lambda cfg, end_time, python=None, timeout=None: (
        calls.append((cfg.fields, end_time, python, timeout)) or run_solver()
    )
    monkeypatch.setitem(sys.modules, "arc26.runner", runner)
    monkeypatch.setenv("ARC26_KAGGLE_ROOT", str(tmp_path))
    # The Run cell sets PYTHONHASHSEED in its process; monkeypatch restores it afterwards.
    monkeypatch.setenv("PYTHONHASHSEED", "stale")
    if competition:
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    else:
        monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    code = notebook_code(built.notebooks[notebook])
    namespace = {"__name__": "__main__"}
    for cell in CODE_CELLS[:3]:
        exec(compile(code[cell], cell, "exec"), namespace)
    namespace.update(
        INSTALLED=True,
        PYTHON="/usr/local/bin/python",
        DATA_DIR=tmp_path / "data",
        MODEL_DIR=tmp_path / "model",
    )
    return namespace, code["arc26-12-run"], calls


def test_run_cell_passes_the_soft_deadline_and_the_hard_stop(built, tmp_path, monkeypatch):
    namespace, cell, calls = run_cell_namespace(built, tmp_path, monkeypatch, lambda: 0)
    assert namespace["RUN_SOLVER"] is True
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, end_time, python, timeout),) = calls
    start = namespace["NOTEBOOK_START"]
    assert end_time == start + 3600 and python == "/usr/local/bin/python"
    assert 4500 - 60 < timeout <= 4500
    assert fields == {
        "data_dir": str(tmp_path / "data"),
        "model_dir": str(tmp_path / "model"),
        "outputs_dir": str(tmp_path / "inference_outputs"),
        "marker_dir": str(tmp_path),
        "subset": ("0934a4d8", "36a08778", "981571dc", "aa4ec2a5"),
        "workers": 4,
        "hash_seed": 0,
        "timing_dir": str(tmp_path / "working" / "run_logs" / "timing"),
        "solver": "nvarc",
    }
    assert namespace["SOLVER_EXIT"] == 0 and namespace["STAGES"]["run"]["solver_exit"] == 0
    assert os.environ["PYTHONHASHSEED"] == "0"
    assert namespace["STAGES"]["run"]["launch_env"] == LAUNCH_ENV


def test_run_cell_prints_a_solver_failure_and_carries_on(built, tmp_path, monkeypatch):
    def crash():
        raise RuntimeError("worker 2 died")

    namespace, cell, calls = run_cell_namespace(built, tmp_path, monkeypatch, crash)
    exec(compile(cell, "run", "exec"), namespace)
    assert len(calls) == 1 and namespace["SOLVER_EXIT"] is None
    assert "error" not in namespace["STAGES"]["run"]
    assert namespace["STAGES"]["run"]["solver_error"] == "RuntimeError: worker 2 died"


def test_run_cell_skips_the_solver_after_the_hard_stop(built, tmp_path, monkeypatch):
    namespace, cell, calls = run_cell_namespace(built, tmp_path, monkeypatch, lambda: 0)
    namespace["NOTEBOOK_START"] -= 4500
    exec(compile(cell, "run", "exec"), namespace)
    assert calls == [] and namespace["SOLVER_EXIT"] is None


def test_run_cell_refuses_to_start_without_the_install_in_development(
    built, tmp_path, monkeypatch
):
    namespace, cell, calls = run_cell_namespace(built, tmp_path, monkeypatch, lambda: 0)
    namespace["INSTALLED"] = False
    with pytest.raises(RuntimeError, match="arc26 is not installed"):
        exec(compile(cell, "run", "exec"), namespace)
    assert calls == []


@pytest.mark.parametrize("run_solver", [True, False])
def test_a_competition_rerun_always_solves_with_the_production_deadlines(
    built, tmp_path, monkeypatch, run_solver
):
    namespace, cell, calls = run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, competition=True
    )
    namespace["RUN_SOLVER"] = run_solver
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, end_time, _, timeout),) = calls
    # A competition rerun records no timing: the solver gets no timing folder.
    assert fields["timing_dir"] is None
    start = namespace["NOTEBOOK_START"]
    assert end_time == start + nbb.COMPETITION_SOLVER_END_S == start + 12 * 3600 - 30 * 60
    assert nbb.COMPETITION_SOLVER_KILL_S - 60 < timeout <= nbb.COMPETITION_SOLVER_KILL_S
    assert nbb.COMPETITION_SOLVER_KILL_S == 12 * 3600 - 20 * 60
    assert namespace["STAGES"]["run"]["solver_end_s"] == nbb.COMPETITION_SOLVER_END_S
    assert namespace["STAGES"]["run"]["solver_kill_s"] == nbb.COMPETITION_SOLVER_KILL_S


def test_run_cell_removes_candidates_markers_and_timing_of_an_earlier_run(
    built, tmp_path, monkeypatch
):
    namespace, cell, calls = run_cell_namespace(built, tmp_path, monkeypatch, lambda: 0)
    outputs = tmp_path / "inference_outputs"
    timing = tmp_path / "working" / "run_logs" / "timing"
    outputs.mkdir()
    timing.mkdir(parents=True)
    stale = [
        tmp_path / "worker0",
        tmp_path / "worker3",
        outputs / "0934a4d8_0.rot90.ex0",
        timing / "timing-rank0.jsonl",
        timing / "timing-main.jsonl",
    ]
    kept = [tmp_path / "workers.txt", tmp_path / "worker", timing / "notes.txt"]
    for path in stale + kept:
        path.write_text("x")
    exec(compile(cell, "run", "exec"), namespace)
    assert len(calls) == 1
    assert [path.exists() for path in stale] == [False] * 5
    assert [path.exists() for path in kept] == [True] * 3


def test_launch_probe_logs_the_failed_solver_and_still_writes_everything(built, tmp_path):
    """The CPU check as committed: the solver (the split solver, the default since 2 Oct) starts
    through the recording python, fails at its first import of the GPU stack (torch here; unsloth
    on Kaggle's CPU image), and the notebook still writes a valid submission and run_logs, without
    requiring a GPU."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["probe"], root)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    assert solver_starts(run) == []
    starts = split_starts(run)
    assert len(starts) == 1
    argv = starts[0]["argv"]
    assert argv[2] == "--end-time" and argv[4] == "--config"
    assert float(argv[3]) == run.record["NOTEBOOK_START"] + 600
    timing_dir = root / "working" / "run_logs" / "timing"
    assert json.loads(argv[5]) == {
        "data_dir": str(root / "input" / "competitions" / COMPETITION),
        "model_dir": str(root.joinpath("input", *MODEL_DIR)),
        "outputs_dir": str(root / "inference_outputs"),
        "marker_dir": str(root),
        "subset": ["aa4ec2a5"],
        "workers": 1,
        "timing_dir": str(timing_dir),
        "lora_r": 256,
        "lora_targets": NINE,
    }
    assert starts[0]["env"] == LAUNCH_ENV
    assert GPU_STACK_MISSING.search(run.output)
    assert run.record["SOLVER_EXIT"] not in (None, 0)
    stages = json.loads((root / "working" / "run_logs" / "stages.json").read_text())
    assert stages["solver_exit"] == run.record["SOLVER_EXIT"]
    # The probe's worker never starts and selection raises on no candidates: no timing record.
    assert stages["timing_dir"] == str(timing_dir) and not timing_dir.exists()
    assert (stages["stages"]["run"]["solver_end_s"], stages["stages"]["run"]["solver_kill_s"]) == (
        600,
        660,
    )
    # The run record lists exactly what the solver's environment received on top of the kernel's.
    assert stages["stages"]["run"]["launch_env"] == LAUNCH_ENV
    env = json.loads((root / "working" / "run_logs" / "env.json").read_text())
    assert env["environment"]["PYTHONHASHSEED"] == "0"
    assert stages["submission"]["valid"] and stages["submission"]["zero_attempts"] == 10
    assert_kernel_never_imported_torch(run)


def test_competition_rerun_of_a_solver_off_build_still_starts_the_solver(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["plain"], root, competition=True)
    assert run.record["error"] is None, run.output
    starts = split_starts(run)  # the CPU check's default solver
    assert len(starts) == 1 and solver_starts(run) == []
    end_time = float(starts[0]["argv"][3])
    assert end_time == run.record["NOTEBOOK_START"] + nbb.COMPETITION_SOLVER_END_S
    assert run.record["STAGES"]["run"]["solver_kill_s"] == nbb.COMPETITION_SOLVER_KILL_S
    assert submit.validate(submission(run), standin_challenges()) == []


# The runner itself (after P0.1)

RUNNER_PROBE = """
import sys

attempted = set()


class Recorder:
    def find_spec(self, name, path=None, target=None):
        attempted.add(name.partition(".")[0])
        return None


sys.meta_path.insert(0, Recorder())
import arc26.runner

print(sorted(attempted & {"torch", "transformers", "unsloth", "tensorflow"}))
"""


def test_importing_the_runner_loads_no_model_code():
    """The Run cell imports arc26.runner in the kernel before the solver starts, so that import
    must not reach torch or transformers: a CUDA context in the kernel would take GPU 0 memory
    from solver worker 0."""
    pytest.importorskip("arc26.runner")
    env = {"PYTHONPATH": os.pathsep.join([str(REPO), sysconfig.get_paths()["purelib"]])}
    probe = subprocess.run(
        [sys.executable, "-S", "-c", RUNNER_PROBE], env=env, capture_output=True, text=True
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]"


# Replay of run A's candidates through the real make_submission (after P0.1)


def seed_candidates(outputs):
    """Run A's candidates as the workers wrote them: one bz2 pickle per subkey, holding a list of
    {beam_score, score_aug, solution} with the solution as a numpy array."""
    np = pytest.importorskip("numpy")
    outputs.mkdir(parents=True)
    for subkey, candidates in json.loads(CANDIDATES_A.read_text()).items():
        rows = [{**c, "solution": np.array(c["solution"], dtype=int)} for c in candidates]
        with bz2.BZ2File(outputs / subkey, "w") as handle:
            pickle.dump(rows, handle)


def test_a_candidate_file_cut_mid_write_is_moved_aside_and_selection_retried(built, tmp_path):
    """A worker killed at the hard stop can leave a truncated bz2 pickle, which would make the
    reference's load_decoded_results fail for every task."""
    pytest.importorskip("arc26.runner")
    cut = "0934a4d8_0.transpose.permute0123456789.ex0123"

    def outputs(folder):
        seed_candidates(folder)
        whole = bz2.compress(pickle.dumps([{"beam_score": 0.0, "score_aug": [], "solution": 0}]))
        (folder / cut).write_bytes(whole[: len(whole) // 2])

    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["plain"], root, outputs=outputs)
    assert run.record["error"] is None, run.output
    assert run.record["STAGES"]["write"]["quarantined"] == [cut]
    assert (root / "arc26_quarantine" / cut).is_file()
    expected = json.loads(FIXTURE.read_text())["predictions"]
    assert {task: submission(run)[task] for task in DEV4} == expected
    assert run.record["report"]["raw_status"] == "read"


def test_development_replay_of_run_a_reproduces_the_fixture(built, tmp_path):
    pytest.importorskip("arc26.runner")
    if not CANDIDATES_A.is_file():
        pytest.skip(f"{CANDIDATES_A.name} arrives with P0.1")
    root = make_tree(tmp_path, built.wheel)
    run = simulate(
        tmp_path, built.notebooks["plain"], root, outputs=seed_candidates
    )
    assert run.record["error"] is None, run.output
    assert run.record["report"]["raw_status"] == "read" and run.record["report"]["valid"]
    written = submission(run)
    expected = json.loads(FIXTURE.read_text())["predictions"]
    assert {task: written[task] for task in DEV4} == expected
    assert_kernel_never_imported_torch(run)


# Timing records (task P0.3b): development runs write them to run_logs/timing/, competition
# reruns write none.


def seed_correct_candidates(outputs):
    """One candidate per stand-in test output whose grid is the stand-in solution, so selection
    returns normally in development mode (the reference's scoring raises when nothing is
    correct)."""
    np = pytest.importorskip("numpy")
    outputs.mkdir(parents=True)
    for task, count in DEV4.items():
        for index in range(count):
            row = {"beam_score": 0.1, "score_aug": [0.2] * 8, "solution": np.array(GRID)}
            with bz2.BZ2File(outputs / f"{task}_{index}.permute0123456789", "w") as handle:
                pickle.dump([row], handle)


def timing_files(root):
    return sorted(path for path in root.rglob("timing-*.jsonl") if path.is_file())


def test_development_mode_records_the_main_selection_in_run_logs(built, tmp_path):
    pytest.importorskip("arc26.runner")
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["timing-plain"], root, outputs=seed_correct_candidates)
    assert run.record["error"] is None, run.output
    assert "make_submission_error" not in run.record["STAGES"]["write"]
    assert all(entry[0]["attempt_1"] == GRID for entry in submission(run).values() if entry)
    logs = root / "working" / "run_logs"
    timing = logs / "timing"
    assert timing_files(root) == [timing / "timing-main.jsonl"]
    lines = (timing / "timing-main.jsonl").read_text().splitlines()
    (record,) = [json.loads(line) for line in lines]
    assert record["puzzle_id"] is None and record["op"] == "main_selection"
    assert record["tokens"] is None and record["seconds"] >= 0
    # The run_logs dump ran after the record was written and neither removed nor copied it.
    assert sorted(p.name for p in logs.iterdir()) == [
        "candidates.json",
        "env.json",
        "pip_freeze.txt",
        "stages.json",
        "timing",
    ]
    candidates = json.loads((logs / "candidates.json").read_text())
    assert sorted(candidates) == sorted(p.name for p in (root / "inference_outputs").iterdir())
    env = json.loads((logs / "env.json").read_text())
    assert set(env["dump"]["written"]) == {"candidates.json", "pip_freeze.txt"}
    stages = json.loads((logs / "stages.json").read_text())
    assert stages["timing_dir"] == str(timing)


@pytest.mark.parametrize("notebook", ["timing", "timing-plain"])
def test_competition_mode_records_no_timing(built, tmp_path, notebook):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks[notebook], root, competition=True, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    # Selection returned normally, so only the mode explains the missing record.
    assert "make_submission_error" not in run.record["STAGES"]["write"]
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]
    assert timing_files(root) == []
    (start,) = solver_starts(run)
    assert json.loads(start["argv"][5])["timing_dir"] is None


def test_p0_3_01_notebook_runs_four_workers_on_four_l4(built, tmp_path):
    """The P0.3-01 build as committed (revision 2): four workers on the four L4s the fake
    nvidia-smi reports, DEV4, P0.5-01's deadlines, and the timing folder passed to the solver;
    its launch fails here at the GPU stack."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["timing"], root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and len(run.record["GPUS"]) == 4
    (start,) = solver_starts(run)
    config = json.loads(start["argv"][5])
    assert config["workers"] == 4 and config["subset"] == sorted(DEV4)
    assert config["timing_dir"] == str(root / "working" / "run_logs" / "timing")
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + 3600
    assert run.record["STAGES"]["run"]["solver_kill_s"] == 4500
    assert submit.validate(submission(run), standin_challenges()) == []


def test_p0_3_01_notebook_stops_without_a_gpu_per_worker(built, tmp_path):
    """Four workers on a machine that lists one GPU: a development run stops at the environment
    check, before the install, with the placeholder submission written."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["timing"], root)
    assert "the solver needs 4 GPUs, one per worker (NvidiaL4); found 1" in run.record["error"]
    assert install_calls(run) == [] and solver_starts(run) == []
    assert submission(run) == submit.repair(None, standin_challenges())


def test_a_one_l4_notebook_runs_one_worker(built, tmp_path):
    """The builder's one-GPU path, from a config made in the test (NvidiaL4X1, one worker): the
    single L4 the fake nvidia-smi reports is enough, and one worker gets the timing folder."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["one-l4"], root)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and len(run.record["GPUS"]) == 1
    (start,) = solver_starts(run)
    config = json.loads(start["argv"][5])
    assert config["workers"] == 1 and config["subset"] == sorted(DEV4)
    assert config["timing_dir"] == str(root / "working" / "run_logs" / "timing")
    assert submit.validate(submission(run), standin_challenges()) == []


# Series mode (task P1.3): one solver launch per arm with its own folders, the stop rule, and a
# competition rerun that ignores the series. The single-run tests above run unchanged.

SERIES_ARMS = ["r256-proj-emb", "r16-attn", "r32-attn", "r16-proj", "r64-proj"]
ARMS_RUN = ("ref", "tiny")
ARM_FILES = [
    "candidates.json",
    "env.json",
    "pip_freeze.txt",
    "selection.log",
    "stages.json",
    "submission.json",
]


def seed_arm_candidates(outputs):
    """The two tiny arms' candidates, one per stand-in test output, in each arm's folder: arm ref
    decoded the stand-in solution, arm tiny the wrong grid [[5]]."""
    np = pytest.importorskip("numpy")
    for name, grid in (("ref", GRID), ("tiny", [[5]])):
        folder = outputs / name
        folder.mkdir(parents=True)
        for task, count in DEV4.items():
            for index in range(count):
                row = {"beam_score": 0.1, "score_aug": [0.2] * 8, "solution": np.array(grid)}
                with bz2.BZ2File(folder / f"{task}_{index}.permute0123456789", "w") as handle:
                    pickle.dump([row], handle)


def test_series_mode_selects_and_logs_every_arm_in_its_own_folders(built, tmp_path):
    """Two tiny arms with the solver off: each arm is selected from its own candidate folder
    and logged in its own run_logs/<arm>/; submission.json comes from the first arm."""
    pytest.importorskip("arc26.runner")
    root = make_tree(tmp_path, built.wheel)
    run = simulate(
        tmp_path, built.notebooks["series-plain"], root, outputs=seed_arm_candidates, gpus=4
    )
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and solver_starts(run) == []
    logs = root / "working" / "run_logs"
    assert sorted(p.name for p in logs.iterdir()) == ["ref", "stages.json", "tiny"]
    ref, tiny = logs / "ref", logs / "tiny"
    # The tiny arm's development scoring raises (no correct candidate), so it has no
    # main_selection record and no timing folder.
    assert sorted(p.name for p in ref.iterdir()) == sorted([*ARM_FILES, "timing"])
    assert sorted(p.name for p in tiny.iterdir()) == ARM_FILES
    top = submission(run)
    assert top == json.loads((ref / "submission.json").read_text())
    assert all(entry[0]["attempt_1"] == GRID for entry in top.values())
    attempts = json.loads((tiny / "submission.json").read_text())
    assert submit.validate(attempts, standin_challenges()) == []
    assert all(entry[0]["attempt_1"] == [[5]] for entry in attempts.values())
    # What the selection printed, the development score included, per arm.
    assert "*** Reload score:" in (ref / "selection.log").read_text()
    assert "ValueError" in (tiny / "selection.log").read_text()
    for name in ("ref", "tiny"):
        seeded = sorted(p.name for p in (root / "inference_outputs" / name).iterdir())
        assert sorted(json.loads((logs / name / "candidates.json").read_text())) == seeded
    (record,) = [json.loads(line) for line in (ref / "timing" / "timing-main.jsonl").open()]
    assert record["op"] == "main_selection" and record["puzzle_id"] is None
    stages = {name: json.loads((logs / name / "stages.json").read_text()) for name in ARMS_RUN}
    assert stages["ref"]["arm"] == {
        "name": "ref", "lora_r": 256, "lora_targets": NINE,
        "reason": "the notebook configuration", "status": "skipped",
    }  # fmt: skip
    assert stages["ref"]["timing_dir"] == str(ref / "timing")
    assert "make_submission_error" not in stages["ref"]["write"]
    assert stages["tiny"]["write"]["make_submission_error"].startswith("ValueError")
    assert stages["tiny"]["write"]["submission"]["valid"]
    summary = json.loads((logs / "stages.json").read_text())
    # Arms without a held-out pair or rounds keep task P1.3's four keys (tasks P2.1, P2.3).
    assert summary["settings"]["arms"] == [
        {key: value for key, value in arm._asdict().items() if key not in ("holdout", "rounds")}
        | {"lora_targets": list(arm.lora_targets)}
        for arm in TINY_ARMS
    ]
    assert [a["status"] for a in summary["stages"]["run"]["arms"]] == ["skipped", "skipped"]
    assert list(summary["stages"]["write"]["arms"]) == ["ref", "tiny"]
    assert summary["timing_dir"] is None and summary["submission"]["valid"]
    assert_kernel_never_imported_torch(run)


def test_a_failed_first_arm_stops_the_series(built, tmp_path):
    """The P1.3-01 build as committed: the first arm's launch fails here at the GPU stack, so no
    later arm starts, and the notebook still writes a valid submission and the arm's logs."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["series"], root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    (start,) = solver_starts(run)
    first = SERIES_ARMS[0]
    config = json.loads(start["argv"][5])
    assert config["outputs_dir"] == str(root / "inference_outputs" / first)
    assert config["timing_dir"] == str(root / "working" / "run_logs" / first / "timing")
    assert (config["lora_r"], config["lora_targets"], config["workers"]) == (256, NINE, 4)
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + 41400
    assert start["env"] == LAUNCH_ENV
    arms = run.record["STAGES"]["run"]["arms"]
    assert [a["name"] for a in arms] == SERIES_ARMS
    assert arms[0]["status"] == "ran" and arms[0]["solver_exit"] not in (None, 0)
    assert run.record["SOLVER_EXIT"] == arms[0]["solver_exit"]
    for later in arms[1:]:
        assert later["status"] == "not started"
        assert later["why"] == f"arm {first} failed, so no later arm starts"
    logs = root / "working" / "run_logs"
    assert sorted(p.name for p in logs.iterdir()) == [first, "stages.json"]
    assert submit.validate(submission(run), standin_challenges()) == []
    assert_kernel_never_imported_torch(run)


def test_a_competition_rerun_ignores_the_series(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["series"], root, competition=True, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    (start,) = solver_starts(run)
    config = json.loads(start["argv"][5])
    assert (config["lora_r"], config["lora_targets"], config["workers"]) == (256, NINE, 4)
    assert config["outputs_dir"] == str(root / "inference_outputs")
    assert config["timing_dir"] is None
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + nbb.COMPETITION_SOLVER_END_S
    assert "arms" not in run.record["STAGES"]["run"]
    assert list((root / "inference_outputs").iterdir()) == []
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]
    assert submit.validate(submission(run), standin_challenges()) == []


def series_namespace(built, tmp_path, monkeypatch, outcomes, competition=False):
    """The Run cell of the three-arm series in process, with a fake clock: each stand-in solver
    launch checks that no worker marker is left, writes the markers its workers would leave,
    advances the clock by its duration, and returns its exit code (or raises an exception)."""
    clock = [60.0]
    queue = list(outcomes)

    def solve():
        assert not list(tmp_path.glob("worker*")), "a stale marker reached the launch"
        for rank in range(4):
            (tmp_path / f"worker{rank}").write_text("Ok")
        duration, code = queue.pop(0)
        clock[0] += duration
        if isinstance(code, Exception):
            raise code
        return code

    namespace, cell, calls = run_cell_namespace(
        built, tmp_path, monkeypatch, solve, competition=competition, notebook="series-three"
    )
    namespace.update(
        NOTEBOOK_START=0.0,
        SOLVER_END_S=3000,
        SOLVER_KILL_S=3600,
        time=types.SimpleNamespace(time=lambda: clock[0]),
    )
    return namespace, cell, calls


def test_the_series_starts_no_arm_that_the_time_left_cannot_cover(built, tmp_path, monkeypatch):
    namespace, cell, calls = series_namespace(
        built, tmp_path, monkeypatch, [(1000.0, 0), (900.0, 0)]
    )
    exec(compile(cell, "run", "exec"), namespace)
    assert len(calls) == 2
    (ref, ref_end, _, ref_timeout), (tiny, tiny_end, _, tiny_timeout) = calls
    assert (ref["lora_r"], ref["lora_targets"]) == (256, NINE)
    assert (tiny["lora_r"], tiny["lora_targets"]) == (1, ["q_proj"])
    for fields, name in ((ref, "ref"), (tiny, "tiny")):
        assert fields["outputs_dir"] == str(tmp_path / "inference_outputs" / name)
        assert fields["timing_dir"] == str(tmp_path / "working" / "run_logs" / name / "timing")
        assert fields["marker_dir"] == str(tmp_path) and fields["workers"] == 4
    # One soft deadline and one hard stop for the whole series.
    assert ref_end == tiny_end == 3000
    assert (ref_timeout, tiny_timeout) == (3600 - 60, 3600 - 1060)
    records = namespace["ARM_RECORDS"]
    assert [r["status"] for r in records] == ["ran", "ran", "not started"]
    assert [(r["start_s"], r["seconds"], r["solver_exit"]) for r in records[:2]] == [
        (60.0, 1000.0, 0),
        (1060.0, 900.0, 0),
    ]
    # After two arms 1040 s are left, less than 1.25 times the longest arm so far.
    assert records[2]["why"] == (
        "1040 s are left before the soft deadline, less than one more arm needs: 1250 s, "
        "1.25 times the longest arm so far (1000 s)"
    )
    run = namespace["STAGES"]["run"]
    assert run["arms"] is records and run["series_margin"] == nbb.SERIES_MARGIN == 1.25
    assert namespace["SOLVER_EXIT"] == run["solver_exit"] == 0
    assert "launch_env" not in run and records[0]["launch_env"] == LAUNCH_ENV


def test_the_series_runs_every_arm_that_fits(built, tmp_path, monkeypatch):
    namespace, cell, calls = series_namespace(
        built, tmp_path, monkeypatch, [(600.0, 0), (500.0, 0), (400.0, 0)]
    )
    exec(compile(cell, "run", "exec"), namespace)
    assert [fields["outputs_dir"].rpartition("/")[2] for fields, *_ in calls] == [
        "ref", "tiny", "third"
    ]  # fmt: skip
    assert [r["status"] for r in namespace["ARM_RECORDS"]] == ["ran"] * 3


@pytest.mark.parametrize("failure", [3, RuntimeError("worker 1 died")], ids=["exit", "raise"])
def test_no_arm_starts_after_a_failed_one(built, tmp_path, monkeypatch, failure):
    namespace, cell, calls = series_namespace(built, tmp_path, monkeypatch, [(100.0, failure)])
    exec(compile(cell, "run", "exec"), namespace)
    assert len(calls) == 1
    first, *later = namespace["ARM_RECORDS"]
    expected = None if isinstance(failure, Exception) else failure
    assert first["status"] == "ran" and first["solver_exit"] == expected
    assert namespace["SOLVER_EXIT"] == expected
    if expected is None:
        assert first["solver_error"] == "RuntimeError: worker 1 died"
    for entry in later:
        assert (entry["status"], entry["why"]) == (
            "not started",
            "arm ref failed, so no later arm starts",
        )


def test_the_series_skips_every_solver_with_run_solver_off(built, tmp_path, monkeypatch):
    namespace, cell, calls = series_namespace(built, tmp_path, monkeypatch, [])
    namespace["RUN_SOLVER"] = False
    exec(compile(cell, "run", "exec"), namespace)
    assert calls == [] and namespace["SOLVER_EXIT"] is None
    assert [r["status"] for r in namespace["ARM_RECORDS"]] == ["skipped"] * 3


def test_a_competition_rerun_of_a_series_runs_the_reference_settings_once(
    built, tmp_path, monkeypatch
):
    namespace, cell, calls = series_namespace(
        built, tmp_path, monkeypatch, [(1000.0, 0)], competition=True
    )
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, end_time, _, _),) = calls
    # No LoRA fields: RunConfig's defaults, the reference's settings.
    assert not {"lora_r", "lora_targets"} & set(fields)
    assert fields["outputs_dir"] == str(tmp_path / "inference_outputs")
    assert fields["timing_dir"] is None and end_time == nbb.COMPETITION_SOLVER_END_S
    assert namespace["SERIES"] is None and namespace["ARM_RECORDS"] == []
    assert "arms" not in namespace["STAGES"]["run"]


# The split solver (task P2.0): SOLVER = 'split' starts python -m arc26.solver in both modes,
# with the arguments the moved reference would get.


def test_split_development_replay_of_run_a_reproduces_the_fixture(built, tmp_path):
    """RUN_SOLVER False: the split build selects run A's candidates into the fixture's attempts,
    as the moved reference's build does (the replay test above)."""
    pytest.importorskip("arc26.runner")
    root = make_tree(tmp_path, built.wheel)
    run = simulate(
        tmp_path, built.notebooks["split-plain"], root, outputs=seed_candidates, gpus=4
    )
    assert run.record["error"] is None, run.output
    assert solver_starts(run) == [] and split_starts(run) == []
    assert run.record["report"]["raw_status"] == "read" and run.record["report"]["valid"]
    expected = json.loads(FIXTURE.read_text())["predictions"]
    assert {task: submission(run)[task] for task in DEV4} == expected
    stages = json.loads((root / "working" / "run_logs" / "stages.json").read_text())
    assert stages["settings"]["solver"] == "split" and stages["settings"]["run_solver"] is False
    assert "solver code: split" in run.output
    assert_kernel_never_imported_torch(run)


def test_split_competition_rerun_starts_arc26_solver(built, tmp_path):
    """A competition rerun of the solver-off split build still solves, with the split solver
    and the competition deadlines; here it fails at its import of torch, and the notebook still
    ends with a valid submission."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["split-plain"], root, competition=True, gpus=4)
    assert run.record["error"] is None, run.output
    assert solver_starts(run) == []
    (start,) = split_starts(run)
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + nbb.COMPETITION_SOLVER_END_S
    assert json.loads(start["argv"][5])["timing_dir"] is None
    assert start["env"] == LAUNCH_ENV
    assert GPU_STACK_MISSING.search(run.output)
    assert submit.validate(submission(run), standin_challenges()) == []
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]


def test_p2_0_01_notebook_starts_the_split_solver_on_four_l4(built, tmp_path):
    """The P2.0-01 build as committed: DEV4, four workers, the timing folder, and the moved
    reference's JSON config, passed to python -m arc26.solver; its launch fails here at the GPU
    stack, and the notebook still writes the submission and the run logs."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["split"], root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and len(run.record["GPUS"]) == 4
    assert solver_starts(run) == []
    (start,) = split_starts(run)
    assert start["argv"][2] == "--end-time" and start["argv"][4] == "--config"
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + 3600
    assert json.loads(start["argv"][5]) == {
        "data_dir": str(root / "input" / "competitions" / COMPETITION),
        "model_dir": str(root.joinpath("input", *MODEL_DIR)),
        "outputs_dir": str(root / "inference_outputs"),
        "marker_dir": str(root),
        "subset": sorted(DEV4),
        "workers": 4,
        "timing_dir": str(root / "working" / "run_logs" / "timing"),
        "lora_r": 256,
        "lora_targets": NINE,
    }
    assert start["env"] == LAUNCH_ENV
    assert GPU_STACK_MISSING.search(run.output)
    assert run.record["SOLVER_EXIT"] not in (None, 0)
    assert submit.validate(submission(run), standin_challenges()) == []
    assert (root / "working" / "run_logs" / "stages.json").is_file()
    assert_kernel_never_imported_torch(run)


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_run_cell_passes_the_split_solver_in_both_modes(
    built, tmp_path, monkeypatch, competition
):
    namespace, cell, calls = run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, competition=competition, notebook="split"
    )
    assert namespace["SOLVER"] == "split"
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, _, _, _),) = calls
    assert fields["solver"] == "split" and fields["workers"] == 4


# Hold-out arms (task P2.1): the arm's settings and gate folders reach the split solver in a
# development run; a competition rerun ignores the series and with it the hold-out.


def test_the_p2_1_01_notebook_passes_the_hold_out_arm_to_the_split_solver(built, tmp_path):
    """The P2.1-01 build as committed: DEV40, four workers, the arm's folders and its hold-out
    settings with the gate's folders, passed to python -m arc26.solver; its launch fails here at
    the GPU stack, and the notebook still writes the submission and the arm's logs."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["holdout"], root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and solver_starts(run) == []
    (start,) = split_starts(run)
    config = json.loads(start["argv"][5])
    logs = root / "working" / "run_logs" / HOLDOUT_ARM
    assert config["holdout"] == {
        **HOLDOUT,
        "gate_dir": str(root / "gate_outputs" / HOLDOUT_ARM),
        "log_dir": str(logs / "gate"),
    }
    assert config["outputs_dir"] == str(root / "inference_outputs" / HOLDOUT_ARM)
    assert config["timing_dir"] == str(logs / "timing")
    assert (config["lora_r"], config["lora_targets"], config["workers"]) == (256, NINE, 4)
    assert len(config["subset"]) == 40
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + 14400
    assert start["env"] == LAUNCH_ENV and GPU_STACK_MISSING.search(run.output)
    (arm,) = run.record["STAGES"]["run"]["arms"]
    assert arm["name"] == HOLDOUT_ARM and arm["holdout"] == HOLDOUT and arm["status"] == "ran"
    assert "hold-out: the last real pair of each puzzle with 2 or more" in run.output
    assert f"gate verdicts in {logs / 'gate'}" in run.output
    assert submit.validate(submission(run), standin_challenges()) == []
    summary = json.loads((root / "working" / "run_logs" / "stages.json").read_text())
    assert summary["settings"]["arms"][0]["holdout"] == HOLDOUT
    assert_kernel_never_imported_torch(run)


def test_a_competition_rerun_of_the_hold_out_build_runs_the_reference_settings_once(
    built, tmp_path
):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["holdout"], root, competition=True, gpus=4)
    assert run.record["error"] is None, run.output
    (start,) = split_starts(run)
    config = json.loads(start["argv"][5])
    assert "holdout" not in config and config["timing_dir"] is None
    assert config["outputs_dir"] == str(root / "inference_outputs")
    assert (config["lora_r"], config["lora_targets"]) == (256, NINE)
    assert not (root / "gate_outputs").exists()
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]
    assert submit.validate(submission(run), standin_challenges()) == []


def test_the_run_cell_removes_the_gate_files_of_an_earlier_launch(built, tmp_path, monkeypatch):
    namespace, cell, calls = run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, notebook="holdout"
    )
    gate_dir = tmp_path / "gate_outputs" / HOLDOUT_ARM
    gate_log = tmp_path / "working" / "run_logs" / HOLDOUT_ARM / "gate"
    (gate_dir / "136b0064").mkdir(parents=True)
    gate_log.mkdir(parents=True)
    stale = [gate_dir / "136b0064" / "136b0064_0.rot90.ex0", gate_log / "gate-rank0.jsonl"]
    kept = [gate_log / "notes.txt"]
    for path in stale + kept:
        path.write_text("x")
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, _, _, _),) = calls
    assert fields["solver"] == "split" and fields["holdout"] == {
        **HOLDOUT, "gate_dir": str(gate_dir), "log_dir": str(gate_log)
    }  # fmt: skip
    assert [path.exists() for path in stale] == [False, False]
    assert [path.exists() for path in kept] == [True]
    assert namespace["ARM_RECORDS"][0]["holdout"] == HOLDOUT


# Script notebooks (task P0.5d): the stand-in script runs beside the wheel with its config, the
# notebook writes run_logs/ however the run ends, and a competition rerun stops at once.


def script_files(script=None, config=None):
    """The stand-in script and its config as the code dataset holds them, beside the wheel."""
    return {
        TINY_SCRIPT.name: TINY_SCRIPT.read_bytes() if script is None else script,
        TINY_CONFIG.name: TINY_CONFIG.read_bytes() if config is None else config,
    }


def script_calls(run):
    return [call for call in run.calls if call["argv"][:1] and call["argv"][0].endswith(".py")]


def run_logs(run):
    logs = run.root / "working" / "run_logs"
    return {p.name: p for p in logs.iterdir()} if logs.is_dir() else {}


def stages_json(run):
    return json.loads(run_logs(run)["stages.json"].read_text())


def gone(pid, wait=5.0):
    """Whether a process has ended: no /proc entry, or a zombie waiting for its reaper."""
    deadline = time.monotonic() + wait
    while True:
        try:
            stat_line = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return True
        if stat_line.rsplit(")", 1)[1].split()[0] in {"Z", "X"}:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def test_a_script_notebook_runs_the_script_beside_the_wheel_and_writes_the_logs(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    run = simulate(tmp_path, built.notebooks["script"], root)
    assert run.record["error"] is None, run.output
    assert run.record["cells"] == SCRIPT_CODE_CELLS and run.record["PROBLEMS"] == []
    dataset = root.joinpath("input", *DATASET_DIRS["2026"])
    assert install_calls(run) == [
        [
            "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall",
            "--disable-pip-version-check", str(dataset / WHEEL_NAME),
        ]
    ]  # fmt: skip
    # The command line of the config's ARGS, "{commit}" replaced by the build's commit.
    (call,) = script_calls(run)
    script, config = dataset / TINY_SCRIPT.name, dataset / TINY_CONFIG.name
    assert call["argv"] == [str(script), "--config", str(config), "--note", COMMIT]
    assert call["env"]["PYTHONUNBUFFERED"] == "1"
    assert "tiny script: wrote result.json" in run.output
    # The script imported the installed wheel and wrote into run_logs/, from /kaggle/working.
    logs = run_logs(run)
    assert sorted(logs) == sorted([*NOTEBOOK_LOGS, "result.json"])
    result = json.loads(logs["result.json"].read_text())
    assert result["config"] == {"out_dir": "run_logs"} and result["cwd"] == str(root / "working")
    assert result["arc26"].startswith(str(run.target)) and result["unbuffered"] == "1"
    stages = stages_json(run)
    assert stages["mode"] == "script" and stages["ok"] is True and stages["problems"] == []
    assert list(stages["stages"]) == SCRIPT_STAGES and stages["script_exit"] == 0
    script = stages["stages"]["script"]
    assert script["command"][1:] == call["argv"] and script["killed"] is False
    assert script["launch_env"] == {"PYTHONUNBUFFERED": "1"}
    assert stages["settings"] == {"args": ["--note", "{commit}"], "script_kill_s": 600}
    build = stages["build"]
    assert build["kind"] == "script" and build["script"] == "tests/testdata/tiny_script.py"
    assert build["script_sha256"] == sha256(TINY_SCRIPT)
    assert build["script_config_sha256"] == sha256(TINY_CONFIG)
    env = json.loads(logs["env.json"].read_text())
    assert env["build"] == build and env["dump"]["errors"] == []
    assert json.loads(logs["candidates.json"].read_text()) == {}
    assert logs["pip_freeze.txt"].read_text() == "arc26==0.0.1\n"
    # No submission, no solver, no model code in the kernel, no solutions file opened.
    assert sorted(p.name for p in (root / "working").iterdir()) == ["run_logs"]
    assert solver_starts(run) == [] and split_starts(run) == []
    assert run.record["imports"] == []
    assert not [path for _, path in run.record["opened"] if "solutions" in path]


def test_a_script_of_the_same_name_elsewhere_is_never_run(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    decoy = root / "input" / "datasets" / "someone" / "old-code"
    decoy.mkdir(parents=True)
    for name, content in script_files(b"raise SystemExit(9)\n").items():
        (decoy / name).write_bytes(content)
    run = simulate(tmp_path, built.notebooks["script"], root)
    assert run.record["error"] is None and run.record["PROBLEMS"] == [], run.output
    (call,) = script_calls(run)
    assert call["argv"][0] == str(root.joinpath("input", *DATASET_DIRS["2026"], TINY_SCRIPT.name))
    assert stages_json(run)["script_exit"] == 0


@pytest.mark.parametrize("changed", ["script", "config"])
def test_a_changed_script_or_config_stops_the_run_before_the_install(built, tmp_path, changed):
    """The code dataset holds another version of the script or its config than the build's: the
    notebook refuses to run it, and still writes its logs."""
    name = {"script": TINY_SCRIPT, "config": TINY_CONFIG}[changed].name
    files = script_files()
    files[name] += b"\n# changed\n"
    root = make_tree(tmp_path, built.wheel, extras=files)
    run = simulate(tmp_path, built.notebooks["script"], root)
    assert run.record["error"] is None, run.output
    assert run.record["cells"] == SCRIPT_CODE_CELLS
    (problem,) = run.record["PROBLEMS"]
    assert problem.startswith("environment check: ")
    digest = hashlib.sha256(files[name]).hexdigest()
    assert f"{name} has sha256 {digest}, the build baked" in problem
    assert install_calls(run) == [] and script_calls(run) == []
    stages = stages_json(run)
    assert stages["ok"] is False and stages["problems"] == [problem]
    assert stages["script_exit"] is None and stages["stages"]["environment"]["error"] == problem
    for skipped in ("install", "script"):
        assert stages["stages"][skipped]["skipped"] == "an earlier stage stopped the run"
    assert sorted(run_logs(run)) == NOTEBOOK_LOGS
    assert "FAILED, see run_logs/stages.json" in run.output


def test_a_missing_script_stops_the_run_and_the_logs_are_still_written(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras={TINY_CONFIG.name: TINY_CONFIG.read_bytes()})
    run = simulate(tmp_path, built.notebooks["script"], root)
    assert run.record["error"] is None, run.output
    (problem,) = run.record["PROBLEMS"]
    assert "0 script candidates" in problem
    assert install_calls(run) == [] and script_calls(run) == []
    assert stages_json(run)["ok"] is False and sorted(run_logs(run)) == NOTEBOOK_LOGS


def test_a_failing_script_still_gets_the_notebook_logs(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    run = simulate(tmp_path, built.notebooks["script-fail"], root)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and run.record["SCRIPT_EXIT"] == 3
    stages = stages_json(run)
    assert stages["script_exit"] == 3 and stages["ok"] is False
    assert stages["stages"]["script"]["killed"] is False
    # The script's own output before it failed, and the notebook's logs.
    assert sorted(run_logs(run)) == sorted([*NOTEBOOK_LOGS, "result.json"])
    assert "FAILED, see run_logs/stages.json" in run.output


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads process states from /proc")
def test_the_hard_stop_kills_the_script_and_every_process_it_started(built, tmp_path):
    """The script starts a child and sleeps; the hard stop, 3 s into the Script cell, kills both,
    and the notebook still writes its logs."""
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    child = None
    try:
        run = simulate(tmp_path, built.notebooks["script-sleep"], root, kill_in=3.0)
        child = json.loads(run_logs(run)["result.json"].read_text())["child"]
        assert run.record["error"] is None, run.output
        assert run.record["PROBLEMS"] == []
        stages = stages_json(run)
        script = stages["stages"]["script"]
        assert script["killed"] is True and stages["script_exit"] == -signal.SIGKILL
        assert stages["ok"] is False and script["seconds"] < 30
        assert gone(child), "the script's child outlived the hard stop"
        # Lines printed before the kill reached the session log.
        assert "tiny script: sleeping 60.0 s" in run.output
        assert "killed at the timeout" in run.output
        assert sorted(run_logs(run)) == sorted([*NOTEBOOK_LOGS, "result.json"])
    finally:
        if child is not None and not gone(child, wait=0):
            with contextlib.suppress(ProcessLookupError):
                os.kill(child, signal.SIGKILL)


def test_the_smoke_notebook_runs_env_probe_with_its_config(built, tmp_path):
    """configs/script_smoke.yaml as committed, with env_probe.py's --root at the simulated tree's
    parent (on Kaggle the default, /): the probe takes --config without complaint and writes its
    reports into run_logs/ beside the notebook's logs."""
    path = REPO / "configs" / SCRIPT_CONFIG
    probe = REPO / "scripts" / "env_probe.py"
    config = nbb.parse_config(path.read_bytes(), path)._replace(args=("--root", str(tmp_path)))
    info = nbb.build_info(
        config,
        tag="local-test",
        commit=COMMIT,
        config_file=f"configs/{SCRIPT_CONFIG}",
        config_sha256=sha256(path),
        wheel=built.wheel.name,
        wheel_sha256=sha256(built.wheel),
        script_sha256=sha256(probe),
        script_config_sha256=sha256(path),
    )
    notebook = tmp_path / "smoke.ipynb"
    notebook.write_bytes(nbb.notebook_bytes(nbb.build_script_notebook(config, info)))
    extras = {probe.name: probe.read_bytes(), path.name: path.read_bytes()}
    root = make_tree(tmp_path, built.wheel, extras=extras)
    run = simulate(tmp_path, notebook, root)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and stages_json(run)["ok"] is True
    (call,) = script_calls(run)
    dataset = root.joinpath("input", *DATASET_DIRS["2026"])
    assert call["argv"] == [
        str(dataset / probe.name), "--config", str(dataset / path.name), "--root", str(tmp_path),
    ]  # fmt: skip
    assert sorted(run_logs(run)) == sorted([*NOTEBOOK_LOGS, "env_probe.json", "env_probe.md"])
    report = json.loads(run_logs(run)["env_probe.json"].read_text())
    assert report["root"] == str(tmp_path)
    assert report["evaluation_stats"]["summary"]["tasks"] == len(DEV4)


def test_a_competition_rerun_of_a_script_notebook_stops_at_once(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    run = simulate(tmp_path, built.notebooks["script"], root, competition=True)
    assert "competition rerun of a script notebook" in run.record["error"]
    assert run.record["cells"] == SCRIPT_CODE_CELLS[:1]
    assert run.calls == [] and list((root / "working").iterdir()) == []


def script_cell_namespace(built, tmp_path, monkeypatch):
    """The script notebook's Config and Helpers cells in process, after a passing environment
    check and install, with run_streamed recording its calls."""
    monkeypatch.setenv("ARC26_KAGGLE_ROOT", str(tmp_path))
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    code = notebook_code(built.notebooks["script"])
    namespace = {"__name__": "__main__"}
    for cell in SCRIPT_CODE_CELLS[:2]:
        exec(compile(code[cell], cell, "exec"), namespace)
    calls = []

    def run_streamed(command, timeout, cwd, env):
        calls.append((command, timeout))
        return 0, False

    namespace.update(
        INSTALLED=True,
        PYTHON="/usr/local/bin/python",
        INPUT_DIRS={},
        SCRIPT_FILE=tmp_path / "code" / TINY_SCRIPT.name,
        SCRIPT_CONFIG_FILE=tmp_path / "code" / TINY_CONFIG.name,
        run_streamed=run_streamed,
    )
    return namespace, code["arc26-10-script"], calls


def test_the_script_cell_passes_the_time_left_before_the_hard_stop(built, tmp_path, monkeypatch):
    namespace, cell, calls = script_cell_namespace(built, tmp_path, monkeypatch)
    exec(compile(cell, "script", "exec"), namespace)
    ((command, timeout),) = calls
    assert command[0] == "/usr/local/bin/python" and command[-2:] == ["--note", COMMIT]
    assert 600 - 60 < timeout <= 600
    assert namespace["SCRIPT_EXIT"] == 0 and namespace["PROBLEMS"] == []


def test_the_script_cell_starts_nothing_after_the_hard_stop(built, tmp_path, monkeypatch):
    namespace, cell, calls = script_cell_namespace(built, tmp_path, monkeypatch)
    namespace["NOTEBOOK_START"] -= 600
    exec(compile(cell, "script", "exec"), namespace)
    assert calls == [] and namespace["SCRIPT_EXIT"] is None
    assert namespace["PROBLEMS"] == ["the hard stop, 600 s after the start, has passed"]


@pytest.mark.parametrize(
    ("args", "commit", "override"),
    [
        (["--config", "/kaggle/input/other.yaml"], COMMIT, "--config"),
        (["--smoke", "--conf=/kaggle/input/other.yaml"], COMMIT, "--conf=/kaggle/input/other.yaml"),
        (["-c", "/kaggle/input/other.yaml"], COMMIT, "-c"),
        # "{commit}" is replaced first, so a commit starting with c cannot slip -c through.
        (["-{commit}"], "c" * 40, "-" + "c" * 40),
    ],
)
def test_the_script_cell_refuses_args_edited_to_set_config(
    built, tmp_path, monkeypatch, args, commit, override
):
    namespace, cell, calls = script_cell_namespace(built, tmp_path, monkeypatch)
    namespace["ARGS"] = args
    namespace["BUILD"] = {**namespace["BUILD"], "commit": commit}
    exec(compile(cell, "script", "exec"), namespace)
    assert calls == [] and namespace["SCRIPT_EXIT"] is None
    (problem,) = namespace["PROBLEMS"]
    assert problem.startswith(f"ARGS must not set --config ({override!r})")


def test_a_script_that_writes_a_notebook_log_is_reported(built, tmp_path, monkeypatch):
    namespace, cell, calls = script_cell_namespace(built, tmp_path, monkeypatch)
    (tmp_path / "working" / "run_logs").mkdir(parents=True)
    (tmp_path / "working" / "run_logs" / "env.json").write_text("{}")
    exec(compile(cell, "script", "exec"), namespace)
    (problem,) = namespace["PROBLEMS"]
    assert problem.startswith("the script wrote env.json into ") and len(calls) == 1


# The bf16 stop (task P0.5d): a build made for an accelerator stops before the install on GPUs
# under compute capability 8.0, as Kaggle's default 2xT4, or on none; a competition rerun logs it.


def test_a_gpu_build_stops_before_the_install_on_gpus_without_bf16(built, tmp_path):
    """The P0.3-01 build (four workers on NvidiaL4) on four T4s: the GPU count passes, bf16 does
    not, and the placeholder submission is written."""
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["timing"], root, gpus=4, gpu=T4)
    error = run.record["error"]
    assert "RuntimeError: environment check: the build asked for NvidiaL4, but nvidia-smi " in error
    assert "under compute capability 8.0, without bf16: Tesla T4 (7.5), Tesla T4 (7.5)," in error
    assert "the solver needs" not in error
    assert run.record["cells"] == CODE_CELLS[:3]
    assert install_calls(run) == [] and solver_starts(run) == []
    assert submission(run) == submit.repair(None, standin_challenges())


def test_a_warning_line_from_nvidia_smi_does_not_stop_a_4xl4_run(built, tmp_path):
    """A driver warning on nvidia-smi's stdout is no GPU: the P0.3-01 build sees its four L4s and
    runs."""
    root = make_tree(tmp_path, built.wheel)
    stray = "WARNING: infoROM is corrupted at gpu 0000:00:04.0"
    run = simulate(tmp_path, built.notebooks["timing"], root, gpus=4, stray=stray)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and len(run.record["GPUS"]) == 4
    assert len(install_calls(run)) == 1 and len(solver_starts(run)) == 1


def test_a_gpu_build_stops_when_no_gpu_is_visible(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["timing-plain"], root, gpus=0)
    assert "the build asked for NvidiaL4, but nvidia-smi lists no GPU" in run.record["error"]
    assert install_calls(run) == []


def test_a_competition_rerun_on_gpus_without_bf16_logs_it_and_writes_a_valid_submission(
    built, tmp_path
):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["timing"], root, competition=True, gpus=4, gpu=T4)
    assert run.record["error"] is None, run.output
    assert run.record["cells"] == CODE_CELLS
    (problem,) = run.record["PROBLEMS"]
    assert "under compute capability 8.0, without bf16: Tesla T4 (7.5)" in problem
    assert len(install_calls(run)) == 1 and len(solver_starts(run)) == 1
    assert submit.validate(submission(run), standin_challenges()) == []
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]


def test_a_cpu_build_runs_on_any_gpu(built, tmp_path):
    root = make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["plain"], root, gpus=2, gpu=T4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and len(install_calls(run)) == 1


def test_a_script_build_stops_before_the_install_on_gpus_without_bf16(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    run = simulate(tmp_path, built.notebooks["script-gpu"], root, gpus=2, gpu=T4)
    assert run.record["error"] is None, run.output
    (problem,) = run.record["PROBLEMS"]
    assert "the build asked for NvidiaL4, but nvidia-smi lists GPUs under compute capability" in (
        problem
    )
    assert install_calls(run) == [] and script_calls(run) == []
    assert stages_json(run)["ok"] is False and sorted(run_logs(run)) == NOTEBOOK_LOGS


def test_a_script_build_runs_on_l4(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    run = simulate(tmp_path, built.notebooks["script-gpu"], root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and len(run.record["GPUS"]) == 4
    assert stages_json(run)["ok"] is True and len(script_calls(run)) == 1


# Attached notebook outputs (task P3.5): pinned outputs of other notebooks, found by their folder
# names wherever Kaggle mounts them and passed to the script where "{input:<owner>/<slug>}" stands.


def mount_outputs(root, layout="2026", names=(EPISODES, CHECK), marker=None):
    """Mount notebook outputs as folders named by their slugs, each with a marker file."""
    folders = {}
    parents, underscores = OUTPUT_DIRS[layout]
    for name in names:
        owner, slug = name.split("/")
        folder = root.joinpath(*(part.format(owner=owner) for part in parents))
        folder = folder / (slug.replace("-", "_") if underscores else slug)
        write_json(folder / "run_logs" / "marker.json", {"output": marker or name})
        folders[name] = folder
    return folders


@pytest.mark.parametrize("layout", sorted(OUTPUT_DIRS))
def test_a_script_notebook_passes_the_folders_of_its_notebook_inputs(built, tmp_path, layout):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    folders = mount_outputs(root, layout)
    run = simulate(tmp_path, built.notebooks["script-inputs"], root)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and stages_json(run)["ok"] is True
    assert run.record["INPUT_DIRS"] == {name: str(folder) for name, folder in folders.items()}
    (call,) = script_calls(run)
    assert call["argv"][3:] == ["--note", str(folders[EPISODES]), "--note", str(folders[CHECK])]
    result = json.loads(run_logs(run)["result.json"].read_text())
    assert result["argv"][2:] == call["argv"][3:]
    stages = stages_json(run)
    assert stages["build"]["notebook_inputs"] == NOTEBOOK_INPUTS
    assert stages["settings"]["args"] == list(INPUT_ARGS)
    assert f"found notebook output {EPISODES} (version 2): [{str(folders[EPISODES])!r}]" in (
        run.output
    )


def test_a_folder_inside_the_owners_folder_wins_over_one_of_the_same_name(built, tmp_path):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    folders = mount_outputs(root, "2026")
    decoy = root / "input" / "datasets" / "someone" / "arc26-p3-4-episodes"
    write_json(decoy / "marker.json", {"output": "decoy"})
    run = simulate(tmp_path, built.notebooks["script-inputs"], root)
    assert run.record["error"] is None and run.record["PROBLEMS"] == [], run.output
    assert run.record["INPUT_DIRS"][EPISODES] == str(folders[EPISODES])


def test_one_output_mounted_twice_with_the_same_files_is_one_input(built, tmp_path):
    """The same output under /kaggle/input and under /kaggle/usr/lib/notebooks (underscored):
    identical files, so one mount, the first found (the input root)."""
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    folders = mount_outputs(root, "2026")
    mount_outputs(root, "usr-lib")
    run = simulate(tmp_path, built.notebooks["script-inputs"], root)
    assert run.record["error"] is None and run.record["PROBLEMS"] == [], run.output
    assert run.record["INPUT_DIRS"] == {name: str(folder) for name, folder in folders.items()}


@pytest.mark.parametrize("case", ["missing", "ambiguous", "ambiguous-roots"])
def test_a_notebook_input_not_found_once_stops_the_run_before_the_install(built, tmp_path, case):
    root = make_tree(tmp_path, built.wheel, extras=script_files())
    if case == "ambiguous-roots":
        # Inside the owner's folder in both roots, with other files in the second: two outputs.
        mount_outputs(root, "2026")
        mount_outputs(root, "usr-lib", names=(EPISODES,), marker="another build")
    else:
        mount_outputs(root, "legacy", names=(CHECK,) if case == "missing" else (EPISODES, CHECK))
    if case == "ambiguous":
        write_json(root / "input" / "other" / "arc26-p3-4-episodes" / "marker.json", {})
    run = simulate(tmp_path, built.notebooks["script-inputs"], root)
    assert run.record["error"] is None, run.output
    (problem,) = run.record["PROBLEMS"]
    count = {"missing": 0, "ambiguous": 2, "ambiguous-roots": 2}[case]
    assert f"{count} distinct folders of the notebook output {EPISODES} under " in problem
    assert problem.endswith("expected 1: attach version 2")
    assert install_calls(run) == [] and script_calls(run) == []
    assert stages_json(run)["ok"] is False and sorted(run_logs(run)) == NOTEBOOK_LOGS


def test_the_script_cell_refuses_args_naming_an_input_the_build_does_not_attach(
    built, tmp_path, monkeypatch
):
    namespace, cell, calls = script_cell_namespace(built, tmp_path, monkeypatch)
    namespace["ARGS"] = ["--note", "{input:someone/other-output}"]
    exec(compile(cell, "script", "exec"), namespace)
    assert calls == [] and namespace["SCRIPT_EXIT"] is None
    (problem,) = namespace["PROBLEMS"]
    assert problem == (
        "ARGS: {input:someone/other-output} names a notebook output that this build does not "
        "attach"
    )


def test_the_script_cell_fills_the_commit_and_the_input_folders(built, tmp_path, monkeypatch):
    namespace, cell, calls = script_cell_namespace(built, tmp_path, monkeypatch)
    folder = tmp_path / "input" / "arc26-p3-4-episodes"
    namespace["ARGS"] = ["--commit", "{commit}", "--root={input:" + EPISODES + "}"]
    namespace["INPUT_DIRS"] = {EPISODES: folder}
    exec(compile(cell, "script", "exec"), namespace)
    ((command, _),) = calls
    assert command[4:] == ["--commit", COMMIT, f"--root={folder}"]
    assert namespace["PROBLEMS"] == []


# Run P3.2-01 (task P3.2): the build of configs/script_P3.2-01.yaml as committed runs the real
# scripts/cache_embeddings.py with its config in the simulated tree, on a stand-in base model.

HYPER_STACK = ("torch", "transformers", "safetensors")
needs_hyper_stack = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in HYPER_STACK),
    reason="scripts/cache_embeddings.py needs torch, transformers, and safetensors",
)


def p3_2_notebook(dest, wheel, extra_args, build=P32_CONFIG, script_config=CACHE_CONFIG):
    """A P3.2 run's script notebook from its build config as committed (P3.2-01's unless
    ``build`` names another), ``extra_args`` after its args."""
    path = REPO / "configs" / build
    config = nbb.parse_config(path.read_bytes(), path)
    config = config._replace(args=(*config.args, *extra_args))
    info = nbb.build_info(
        config,
        tag="local-test",
        commit=COMMIT,
        config_file=f"configs/{build}",
        config_sha256=sha256(path),
        wheel=wheel.name,
        wheel_sha256=sha256(wheel),
        script_sha256=sha256(CACHE_SCRIPT),
        script_config_sha256=sha256(script_config),
    )
    out = dest / f"{path.stem}.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_script_notebook(config, info)))
    return out


def standin_base(root, task_ids):
    """At the 2026 model mount, a 36-layer copy of the encoder tests' tiny random Qwen3 in bf16
    with the stand-in tokenizer, both built by the script (nothing is downloaded); in the
    competition's folder, stand-in challenges of ``task_ids``, one distinct pair each."""
    import torch

    script = load("cache_embeddings", CACHE_SCRIPT)
    model = root.joinpath("input", *MODEL_DIR)
    script.smoke_model(0, "sdpa", layers=36).to(torch.bfloat16).save_pretrained(model)
    script.smoke_tokenizer().save_pretrained(model)
    challenges = {
        task: {
            "train": [{"input": [[i % 10, i // 10 % 10]], "output": [[i % 10]]}],
            "test": [{"input": GRID}],
        }
        for i, task in enumerate(task_ids)
    }
    data = root / "input" / "competitions" / COMPETITION
    return model, write_json(data / "arc-agi_evaluation_challenges.json", challenges)


def test_the_p3_2_01_build_config_matches_its_brief():
    path = REPO / "configs" / P32_CONFIG
    config = nbb.parse_config(path.read_bytes(), path)
    assert config.kind == "script" and config.kaggle_notebook == "alyahya737/arc26-p3-2-encoder"
    assert config.accelerator == "NvidiaL4" and config.notebook_inputs == {}
    assert config.script == "scripts/cache_embeddings.py"
    assert config.script_config == "configs/cache_embeddings.yaml"
    assert config.args == ("--commit", "{commit}")
    # The script's own stop, max_wall_s after its start, comes first: setup before the script
    # (environment check and install, about a minute) and its last writes fit in 300 s.
    settings = yaml.safe_load(CACHE_CONFIG.read_text(encoding="utf-8"))
    latest = nbb.SESSION_LIMIT_S - nbb.WRITE_RESERVE_S
    assert settings["max_wall_s"] + 300 <= config.script_kill_s <= latest
    header = nbb.gateway.brief_header(P32_BRIEF.read_text(encoding="utf-8"))
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Accelerator"] == config.accelerator
    assert header["Config"] == config.script_config
    assert int(header["Timeout"]) == config.script_kill_s + nbb.WRITE_RESERVE_S
    assert config.kaggle_notebook in nbb.gateway.briefed_notebooks()


def test_the_p3_2_01_build_shares_the_stack_and_finds_the_scripts_inputs():
    """The same image, install notebook, code dataset, competition, and model as every build, and
    the notebook's environment check and the script look for the same model and data: the
    notebook by ``model_slug`` and the competition's folder, the script by the ends of their paths
    (``model_id`` after its owner, ``challenges``)."""
    path = REPO / "configs" / P32_CONFIG
    config = nbb.parse_config(path.read_bytes(), path)
    gpu_path = REPO / "configs" / "submission_P0.5-01.yaml"
    gpu = nbb.parse_config(gpu_path.read_bytes(), gpu_path)
    for name in ("code_dataset", "competition", "reference_metadata", "kernel_source_versions"):
        assert getattr(config, name) == getattr(gpu, name), name
    assert config.model_slug == gpu.model_slug
    settings = yaml.safe_load(CACHE_CONFIG.read_text(encoding="utf-8"))
    assert settings["model_dir"] is None and settings["model_id"].split("/")[1] == config.model_slug
    assert settings["challenges"].split("/")[0] == config.competition
    assert settings["subset"] == "DEV40" and settings["device"] == "cuda:0"


def test_the_p3_2_01_build_config_has_a_comment_above_every_key():
    """BRIEF A8: every config value carries its reason."""
    lines = (REPO / "configs" / P32_CONFIG).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"no comment above {line!r}"


@needs_hyper_stack
def test_the_p3_2_01_build_runs_cache_embeddings_in_the_simulated_tree(built, tmp_path):
    """The environment check finds the script and its config beside the wheel, the bf16 stop
    passes on four L4s, and cache_embeddings.py runs from the code dataset's folder with the
    commit, imports the installed wheel (DEV40 from arc26.data), finds the model and the
    challenges by search, and writes its logs beside the notebook's. The arguments added suit
    this machine only: the simulated input root, folders in the simulated /kaggle/working, and
    the CPU."""
    root = tmp_path / "kaggle"
    working = root / "working"
    extra = (
        "--input-root", str(root / "input"), "--out-dir", str(working / "run_logs"),
        "--cache-dir", str(working / "embedding_cache"), "--device", "cpu",
    )  # fmt: skip
    notebook = p3_2_notebook(tmp_path, built.wheel, extra)
    extras = {path.name: path.read_bytes() for path in (CACHE_SCRIPT, CACHE_CONFIG)}
    assert make_tree(tmp_path, built.wheel, extras=extras) == root
    model, challenges = standin_base(root, DEV40)
    run = simulate(tmp_path, notebook, root, gpus=4, real_transformers=True)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [], run.output
    stages = stages_json(run)
    assert stages["ok"] is True and stages["script_exit"] == 0, run.output
    dataset = root.joinpath("input", *DATASET_DIRS["2026"])
    (call,) = script_calls(run)
    assert call["argv"] == [
        str(dataset / CACHE_SCRIPT.name), "--config", str(dataset / CACHE_CONFIG.name),
        "--commit", COMMIT, *extra,
    ]  # fmt: skip
    logs = run_logs(run)
    assert sorted(logs) == sorted([*NOTEBOOK_LOGS, *CACHE_LOGS])
    record = json.loads(logs["run.json"].read_text())
    assert record["commit"] == COMMIT and record["device_name"] == "cpu"
    assert record["inputs"] == {"model_dir": str(model), "challenges": str(challenges)}
    assert record["encoder"]["layer"] == 24 and record["model"]["dtype"] == "bfloat16"
    metrics = json.loads(logs["metrics.json"].read_text())
    assert metrics["status"] == "complete" and metrics["tasks"]["count"] == len(DEV40)
    assert metrics["throughput"]["tasks"] == len(DEV40)
    assert metrics["cache"]["entries"] == len(DEV40)
    assert sorted(p.name for p in working.iterdir()) == ["embedding_cache", "run_logs"]
    # The kernel imported no model code: only the script's own process did.
    assert run.record["imports"] == []


# Run P3.2-02 (task P3.2): P3.2-01's build with the script's 8-view config, which adds the view
# diagnostic; the same notebook, stack, script, and arguments.


def test_the_p3_2_02_build_config_matches_its_brief():
    path = REPO / "configs" / P3202_CONFIG
    config = nbb.parse_config(path.read_bytes(), path)
    first_path = REPO / "configs" / P32_CONFIG
    first = nbb.parse_config(first_path.read_bytes(), first_path)
    # P3.2-01's build but for the script's config: same notebook, stack, script, args, hard stop.
    assert config == first._replace(script_config="configs/cache_embeddings_P3.2-02.yaml")
    settings = yaml.safe_load(CACHE_CONFIG_02.read_text(encoding="utf-8"))
    assert (settings["n_perm"], settings["view_seed"], settings["reference_n_perm"]) == (1, 2, 2)
    assert settings["max_wall_s"] + 300 <= config.script_kill_s
    header = nbb.gateway.brief_header(P3202_BRIEF.read_text(encoding="utf-8"))
    assert header["Run-id"] == "P3.2-02" and header["Previous run"] == "P3.2-01"
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Accelerator"] == config.accelerator
    assert header["Config"] == config.script_config
    assert int(header["Timeout"]) == config.script_kill_s + nbb.WRITE_RESERVE_S


def test_the_p3_2_02_build_config_has_a_comment_above_every_key():
    """BRIEF A8: every config value carries its reason."""
    lines = (REPO / "configs" / P3202_CONFIG).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"no comment above {line!r}"


@needs_hyper_stack
def test_the_p3_2_02_build_runs_cache_embeddings_in_the_simulated_tree(built, tmp_path):
    """As P3.2-01's simulated run, with 8 views: the throughput phase prefills 32 sequences per
    task, and the view diagnostic encodes the other 8 views of every pair and writes its records
    beside the script's other logs."""
    root = tmp_path / "kaggle"
    working = root / "working"
    extra = (
        "--input-root", str(root / "input"), "--out-dir", str(working / "run_logs"),
        "--cache-dir", str(working / "embedding_cache"), "--device", "cpu",
    )  # fmt: skip
    notebook = p3_2_notebook(
        tmp_path, built.wheel, extra, build=P3202_CONFIG, script_config=CACHE_CONFIG_02
    )
    extras = {path.name: path.read_bytes() for path in (CACHE_SCRIPT, CACHE_CONFIG_02)}
    assert make_tree(tmp_path, built.wheel, extras=extras) == root
    standin_base(root, DEV40)
    run = simulate(tmp_path, notebook, root, gpus=4, real_transformers=True)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [], run.output
    stages = stages_json(run)
    assert stages["ok"] is True and stages["script_exit"] == 0, run.output
    dataset = root.joinpath("input", *DATASET_DIRS["2026"])
    (call,) = script_calls(run)
    assert call["argv"] == [
        str(dataset / CACHE_SCRIPT.name), "--config", str(dataset / CACHE_CONFIG_02.name),
        "--commit", COMMIT, *extra,
    ]  # fmt: skip
    logs = run_logs(run)
    assert sorted(logs) == sorted([*NOTEBOOK_LOGS, *CACHE_LOGS, "view_diagnostic.jsonl"])
    metrics = json.loads(logs["metrics.json"].read_text())
    assert metrics["status"] == "complete" and metrics["throughput"]["views"] == 8
    assert metrics["throughput"]["tasks"] == len(DEV40)
    diagnostic = metrics["view_diagnostic"]
    assert (diagnostic["pairs"], diagnostic["views"], diagnostic["reference_views"]) == (
        len(DEV40), 8, 16,
    )  # fmt: skip
    assert len(logs["view_diagnostic.jsonl"].read_text().splitlines()) == len(DEV40)
    assert run.record["imports"] == []


# Run P3.5-02 (task P3.5): the build of configs/script_P3.5-02.yaml as committed, P3.5-01's with
# half the steps. The simulated notebook accepts it (the real script and its config beside the
# wheel, each P3.4 output found once) and, with --smoke added, the script trains on the episode
# fixture mounted as those outputs.

P3502_CONFIG = "script_P3.5-02.yaml"
TRAIN_SCRIPT = REPO / "scripts" / "train_hypernet.py"
TRAIN_CONFIG_02 = REPO / "configs" / "train_hypernet_toy_P3.5-02.yaml"
TRAIN_FIXTURE = REPO / "tests" / "testdata" / "train_hypernet"
# What train_hypernet.py writes into run_logs/ beside the notebook's files in a smoke on 2
# processes; loss_curve.png only where matplotlib is installed (Kaggle's image, not the dev venv).
TRAIN_LOGS = [
    "config.yaml", "loss_curve.csv", "metrics.jsonl", "run.json", "status.json", "summary.md",
    "timing-rank0.jsonl", "timing-rank1.jsonl",
]  # fmt: skip
TRAIN_STACK = ("torch", "transformers", "safetensors")
needs_train_stack = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in TRAIN_STACK),
    reason="scripts/train_hypernet.py needs torch, transformers, and safetensors",
)


def p3_5_02_notebook(dest, wheel, extra_args):
    """P3.5-02's script notebook from its build config as committed, ``extra_args`` after its
    args."""
    path = REPO / "configs" / P3502_CONFIG
    config = nbb.parse_config(path.read_bytes(), path)
    config = config._replace(args=(*config.args, *extra_args))
    info = nbb.build_info(
        config,
        tag="local-test",
        commit=COMMIT,
        config_file=f"configs/{P3502_CONFIG}",
        config_sha256=sha256(path),
        wheel=wheel.name,
        wheel_sha256=sha256(wheel),
        script_sha256=sha256(TRAIN_SCRIPT),
        script_config_sha256=sha256(TRAIN_CONFIG_02),
    )
    out = dest / f"{path.stem}.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_script_notebook(config, info)))
    return out


def mount_episode_fixture(root):
    """The two P3.4 outputs where P3.5-01's session found them (its stages.json:
    ``/kaggle/input/notebooks/<owner>/<slug>``), holding the episode fixture's build and its
    passing check report in the outputs' layout."""
    episodes = root.joinpath("input", "notebooks", *EPISODES.split("/"))
    check = root.joinpath("input", "notebooks", *CHECK.split("/"))
    shutil.copytree(TRAIN_FIXTURE / "episode_build", episodes)
    shutil.copytree(TRAIN_FIXTURE / "check", check)
    return episodes, check


@needs_train_stack
def test_the_p3_5_02_build_runs_train_hypernet_in_the_simulated_tree(built, tmp_path):
    """The environment check finds train_hypernet.py and its P3.5-02 config beside the wheel with
    the build's sha256 and each P3.4 output once; the bf16 stop passes on four L4s; the script runs
    from the code dataset's folder with the commit and both mount folders, imports the installed
    wheel in both of its processes, and writes its logs beside the notebook's. The arguments added
    suit this machine only: --smoke (the CPU, the tiny random base, the fixture's manifest) and
    folders in the simulated /kaggle/working."""
    root = tmp_path / "kaggle"
    working = root / "working"
    extra = (
        "--smoke", "--out-dir", str(working / "run_logs"),
        "--ckpt-dir", str(working / "checkpoints"),
    )  # fmt: skip
    notebook = p3_5_02_notebook(tmp_path, built.wheel, extra)
    extras = {path.name: path.read_bytes() for path in (TRAIN_SCRIPT, TRAIN_CONFIG_02)}
    assert make_tree(tmp_path, built.wheel, extras=extras) == root
    episodes, check = mount_episode_fixture(root)
    run = simulate(tmp_path, notebook, root, gpus=4, real_transformers=True)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [], run.output
    assert run.record["INPUT_DIRS"] == {EPISODES: str(episodes), CHECK: str(check)}
    stages = stages_json(run)
    assert stages["ok"] is True and stages["script_exit"] == 0, run.output
    assert stages["build"]["script_config"] == "configs/train_hypernet_toy_P3.5-02.yaml"
    assert stages["settings"]["script_kill_s"] == 8400
    dataset = root.joinpath("input", *DATASET_DIRS["2026"])
    (call,) = script_calls(run)
    assert call["argv"] == [
        str(dataset / TRAIN_SCRIPT.name), "--config", str(dataset / TRAIN_CONFIG_02.name),
        "--commit", COMMIT, "--episodes-root", str(episodes), "--check-root", str(check), *extra,
    ]  # fmt: skip
    logs = run_logs(run)
    assert sorted(logs) == sorted([*NOTEBOOK_LOGS, *TRAIN_LOGS])
    status = json.loads(logs["status.json"].read_text())
    assert (status["status"], status["step"], status["steps"]) == ("complete", 12, 12)
    record = json.loads(logs["run.json"].read_text())
    assert (record["run_id"], record["commit"], record["smoke"]) == ("P3.5-02-smoke", COMMIT, True)
    assert record["world_size"] == 2 and record["episodes"]["folder"] == str(episodes)
    assert record["episodes"]["check_report"] == str(check / "run_logs" / "leakage.json")
    assert "Pass 2 minus pass 1, paired over the same episodes" in logs["summary.md"].read_text()
    assert sorted(p.name for p in working.iterdir()) == ["checkpoints", "run_logs"]
    # The kernel imported no model code: only the script's processes did.
    assert run.record["imports"] == []
