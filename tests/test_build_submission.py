"""Acceptance tests for scripts/build_submission.py and the config rules and notebook structure of
scripts/build_kaggle_notebook.py (task P0.5, deliverables 2 to 4).

Builds run in a throwaway git repository that holds a copy of the working tree (tracked and
untracked files) committed with a fixed date, so the clean-tree rule and ``git archive`` see the
code under test. The wheel is built as the builder builds it (uv offline, then online, then pip);
build tests are skipped when neither uv nor pip exists. Nothing reaches Kaggle: the gateway's
metadata checks are called directly.
"""

import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from arc26 import runner

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nbb = load("build_kaggle_notebook", SCRIPTS / "build_kaggle_notebook.py")
bs = load("build_submission", SCRIPTS / "build_submission.py")
ops = load("kaggle_ops", SCRIPTS / "kaggle_ops.py")
reference_run = load("build_reference_run", SCRIPTS / "build_reference_run.py")

GPU_CONFIG = "configs/submission_P0.5-01.yaml"
CPU_CONFIG = "configs/submission_cpu_check.yaml"
FULL_CONFIG = "configs/submission_P1.1-01.yaml"
TIMING_CONFIG = "configs/submission_P0.3-01.yaml"
SERIES_CONFIG = "configs/submission_P1.3-01.yaml"
SPLIT_CONFIG = "configs/submission_P2.0-01.yaml"
# The held-out gate's price run (task P2.1): one hold-out arm on DEV40 with the split solver.
HOLDOUT_CONFIG = "configs/submission_P2.1-01.yaml"
# The script notebook of the Kaggle CPU check of script mode (task P0.5d).
SCRIPT_CONFIG = "configs/script_smoke.yaml"
# P3.5-01's script build (task P3.5): train_hypernet.py on 4xL4 with two P3.4 outputs attached.
P35_CONFIG = "configs/script_P3.5-01.yaml"
# P3.5-02's script build: P3.5-01's with half the steps (its own toy config and hard stop).
P3502_CONFIG = "configs/script_P3.5-02.yaml"
# Config keys that fix the software stack and the inputs; every build shares them.
STACK = (
    "code_dataset",
    "competition",
    "reference_metadata",
    "kernel_source_versions",
    "model_slug",
)
DEV4 = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
WHEEL = "arc26-0.0.1-py3-none-any.whl"
FILES = {
    "MANIFEST.md",
    "kaggle_submission.ipynb",
    "kernel-metadata.json",
    f"arc26-code/{WHEEL}",
    "arc26-code/dataset-metadata.json",
}
DROP = object()
EM_DASH = chr(0x2014)
# The hygiene test's pattern for Kaggle access from code (tests/test_repo_hygiene.py).
KAGGLE_ACCESS = re.compile(
    r"^\s*(import kaggle\b|from kaggle\b)"
    r"|\bkaggle (competitions|kernels|datasets|models) "
    r"|[\"']kaggle[\"']\s*,",
    re.MULTILINE,
)
GIT_ENV = {
    "GIT_AUTHOR_NAME": "tester",
    "GIT_AUTHOR_EMAIL": "tester@example.com",
    "GIT_COMMITTER_NAME": "tester",
    "GIT_COMMITTER_EMAIL": "tester@example.com",
    "GIT_AUTHOR_DATE": "2026-09-30T12:00:00Z",
    "GIT_COMMITTER_DATE": "2026-09-30T12:00:00Z",
    # The human's global git config (signing, hooks) must not reach the throwaway repository.
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
BUILDER = shutil.which("uv") or importlib.util.find_spec("pip")
needs_builder = pytest.mark.skipif(not BUILDER, reason="no wheel builder (uv or pip)")


def git(repo, *args, date=None):
    dates = {} if date is None else {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **GIT_ENV, **dates},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def snapshot(dest):
    """A git repository at ``dest`` holding the working tree's files in one commit."""
    listed = git(REPO, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    for name in filter(None, listed.split("\0")):
        source = REPO / name
        if source.is_file():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    git(dest, "init", "-q")
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "snapshot")
    return dest


def commit_all(repo, message="change", date=None):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message, date=date)


def read_files(folder):
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Builds(NamedTuple):
    repo: Path
    gpu: Path
    cpu: Path
    full: Path
    cpu_first: dict
    series: Path
    split: Path
    script: Path
    p35: Path
    p3502: Path
    holdout: Path


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    if not BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = snapshot(tmp_path_factory.mktemp("repo"))
    cpu = bs.build("P0.5-cpu", repo / CPU_CONFIG, repo)
    first = read_files(cpu)
    gpu = bs.build("P0.5-01", repo / GPU_CONFIG, repo)
    full = bs.build("P1.1-01", repo / FULL_CONFIG, repo)
    cpu = bs.build("P0.5-cpu", repo / CPU_CONFIG, repo)
    series = bs.build("P1.3-01", repo / SERIES_CONFIG, repo)
    split = bs.build("P2.0-01", repo / SPLIT_CONFIG, repo)
    script = bs.build("script-smoke", repo / SCRIPT_CONFIG, repo)
    p35 = bs.build("P3.5-01", repo / P35_CONFIG, repo)
    p3502 = bs.build("P3.5-02", repo / P3502_CONFIG, repo)
    holdout = bs.build("P2.1-01", repo / HOLDOUT_CONFIG, repo)
    return Builds(repo, gpu, cpu, full, first, series, split, script, p35, p3502, holdout)


def cells(folder):
    return json.loads((folder / "kaggle_submission.ipynb").read_text(encoding="utf-8"))["cells"]


def code(folder):
    return {c["id"]: "".join(c["source"]) for c in cells(folder) if c["cell_type"] == "code"}


# The committed configs


def load_config(name):
    path = REPO / name
    return nbb.parse_config(path.read_bytes(), path)


def test_gpu_config_is_the_p0_5_01_smoke():
    config = load_config(GPU_CONFIG)
    assert config.kaggle_notebook == "alyahya737/arc26-submission"
    assert config.code_dataset == "alyahya737/arc26-code"
    assert config.accelerator == "NvidiaL4" and config.run_solver is True
    assert config.subset == tuple(DEV4) and config.workers == 4 and config.hash_seed == 0
    assert config.kernel_source_versions == {"sorokin/pip-install-unsloth-flash-patch": 2}
    assert config.solver_end_s < config.solver_kill_s <= nbb.SESSION_LIMIT_S - nbb.WRITE_RESERVE_S


def test_cpu_config_is_the_launch_probe_on_the_same_stack():
    gpu, cpu = load_config(GPU_CONFIG), load_config(CPU_CONFIG)
    assert cpu.kaggle_notebook == "alyahya737/arc26-submission-cpu-check"
    assert cpu.accelerator is None and cpu.run_solver is True
    assert cpu.workers == 1 and cpu.subset == ("aa4ec2a5",) and cpu.hash_seed == 0
    assert (cpu.solver_end_s, cpu.solver_kill_s) == (600, 660)
    assert {f: getattr(cpu, f) for f in STACK} == {f: getattr(gpu, f) for f in STACK}


def test_p1_1_config_is_the_dev120_run_with_production_deadlines():
    gpu, full = load_config(GPU_CONFIG), load_config(FULL_CONFIG)
    assert full.kaggle_notebook == gpu.kaggle_notebook == "alyahya737/arc26-submission"
    assert full.subset is None and full.workers == 4 and full.hash_seed == 0
    assert full.accelerator == "NvidiaL4" and full.run_solver is True
    assert full.solver_end_s == nbb.COMPETITION_SOLVER_END_S
    assert full.solver_kill_s == nbb.COMPETITION_SOLVER_KILL_S
    assert {f: getattr(full, f) for f in STACK} == {f: getattr(gpu, f) for f in STACK}


def test_p0_3_01_config_repeats_p0_5_01_in_its_own_notebook():
    # Revision 2: Kaggle did not grant NvidiaL4X1 (revision 1 ran on 2xT4, which cannot run the
    # solver), so the timing run repeats P0.5-01: DEV4, one worker per L4 on 4xL4, its deadlines.
    gpu, timing = load_config(GPU_CONFIG), load_config(TIMING_CONFIG)
    assert timing.accelerator == "NvidiaL4" and timing.workers == 4 == nbb.GPU_COUNTS["NvidiaL4"]
    assert timing._replace(kaggle_notebook=gpu.kaggle_notebook) == gpu
    # A development notebook of its own: the submission notebook keeps submission builds only.
    assert timing.kaggle_notebook == "alyahya737/arc26-timing" != gpu.kaggle_notebook
    pulled = json.loads((REPO / timing.reference_metadata).read_text(encoding="utf-8"))
    assert bs.kernel_metadata(timing, pulled)["machine_shape"] == "NvidiaL4"
    # A development build by its notebook alone (task P1.3's guard): only builds for the
    # submission notebook are fit to submit, whatever their workers and accelerator.
    assert nbb.submission_blocker(timing) == (
        "its notebook is alyahya737/arc26-timing, not alyahya737/arc26-submission, the only "
        "notebook submissions come from"
    )


def test_a_one_l4_build_passes_the_gateway():
    """The builder's one-GPU path, with a config made here: NvidiaL4X1 and one worker."""
    config = nbb.parse_config(
        config_text({"accelerator": "NvidiaL4X1", "workers": 1}), Path("config.yaml")
    )
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)  # runs the gateway's metadata checks
    assert meta["machine_shape"] == "NvidiaL4X1" and meta["enable_gpu"] is True
    assert ops.uses_gpu(meta, None) and ops.accelerator_id(meta, None) == "NvidiaL4X1"


def test_the_cpu_notebook_is_not_a_briefed_notebook():
    assert load_config(CPU_CONFIG).kaggle_notebook not in ops.briefed_notebooks()


@pytest.mark.parametrize(
    "name",
    [
        GPU_CONFIG, CPU_CONFIG, FULL_CONFIG, TIMING_CONFIG, SERIES_CONFIG, SPLIT_CONFIG,
        SCRIPT_CONFIG, P35_CONFIG, HOLDOUT_CONFIG,
    ],
)  # fmt: skip
def test_every_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"{name}: no comment above {line!r}"


# Config rules


def config_text(changes):
    raw = yaml.safe_load((REPO / GPU_CONFIG).read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is DROP:
            raw.pop(key)
        else:
            raw[key] = value
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"workers": 5}, "5 workers need 5 GPUs, but NvidiaL4 has 4"),
        ({"accelerator": "NvidiaL4X1"}, "4 workers need 4 GPUs, but NvidiaL4X1 has 1"),
        ({"workers": 0}, "workers must be an integer from 1 to 8"),
        ({"workers": True}, "workers must be an integer"),
        ({"workers": 4.0}, "workers must be an integer"),
        ({"accelerator": "4xL4"}, "accelerator must be null (CPU) or one of"),
        ({"accelerator": "NvidiaTeslaP100"}, "accelerator must be null (CPU) or one of"),
        ({"accelerator": ["NvidiaL4"]}, "accelerator must be null (CPU) or one of"),
        ({"run_solver": "yes"}, "run_solver must be true or false"),
        ({"subset": []}, "subset must be null or a non-empty list"),
        ({"subset": ["0934a4d8", "0934a4d8"]}, "distinct 8-digit hex task ids"),
        ({"subset": ["0934A4D8"]}, "distinct 8-digit hex task ids"),
        ({"subset": [934]}, "distinct 8-digit hex task ids"),
        ({"subset": "0934a4d8"}, "distinct 8-digit hex task ids"),
        ({"hash_seed": -1}, "hash_seed must be an integer from 0 to"),
        ({"hash_seed": 2**32}, "hash_seed must be an integer from 0 to"),
        ({"solver_end_s": 300}, "solver_end_s <= solver_kill_s"),
        ({"solver_kill_s": 3000}, "solver_end_s <= solver_kill_s"),
        ({"solver_kill_s": 43000}, "12 h minus 300 s"),
        ({"solver_end_s": 3600.5}, "need integers"),
        ({"kaggle_notebook": "someone/arc26-submission"}, "KAGGLE_OWNER"),
        ({"code_dataset": "someone/arc26-code"}, "KAGGLE_OWNER"),
        ({"kaggle_notebook": "alyahya737/arc26_submission"}, "kaggle_notebook must be"),
        ({"kaggle_notebook": "alyahya737/Arc26-submission"}, "kaggle_notebook must be"),
        ({"kaggle_notebook": "alyahya737/abcd"}, "kaggle_notebook must be"),
        ({"kaggle_notebook": "arc26-submission"}, "kaggle_notebook must be"),
        ({"code_dataset": "alyahya737/arc26"}, "code_dataset must be"),
        ({"competition": "ARC Prize"}, "competition must be a Kaggle competition slug"),
        ({"reference_metadata": "../kernel-metadata.json"}, "reference_metadata"),
        ({"reference_metadata": "/etc/kernel-metadata.json"}, "reference_metadata"),
        ({"reference_metadata": "reference/kernel-metadata.txt"}, "reference_metadata"),
        ({"kernel_source_versions": {"sorokin/pip-install-unsloth-flash-patch": 0}}, "kernel"),
        ({"kernel_source_versions": {"sorokin/pip-install/2": 2}}, "kernel_source_versions"),
        ({"kernel_source_versions": ["sorokin/pip-install-unsloth-flash-patch"]}, "kernel"),
        ({"model_slug": "models/qwen"}, "model_slug must be one folder name"),
        ({"model_slug": ".."}, "model_slug must be one folder name"),
        ({"extra": 1}, "unknown keys ['extra']"),
        ({"hash_seed": DROP}, "missing keys ['hash_seed']"),
        ({"competition": "TODO"}, "TODO"),
        ({"model_slug": "\x54ODO"}, "TODO"),
        ({"solver": "reference"}, "solver must be one of nvarc, split"),
        ({"solver": "Split"}, "solver must be one of nvarc, split"),
        ({"solver": None}, "solver must be one of nvarc, split"),
        ({"solver": ["split"]}, "solver must be one of nvarc, split"),
        (
            {"accelerator": "NvidiaTeslaT4", "workers": 2},
            "accelerator NvidiaTeslaT4 can never pass the environment check's bf16 stop (T4 has "
            "compute capability 7.5, without bf16); request NvidiaL4",
        ),
        ({"accelerator": "TpuV5E8"}, "TpuV5E8 can never pass the environment check's bf16 stop"),
        ({"accelerator": "TpuV6E8"}, "(a TPU session has no NVIDIA GPU)"),
    ],
)
def test_config_refusals(changes, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(config_text(changes), Path("config.yaml"))
    assert message in str(refused.value)


@pytest.mark.parametrize(
    "data",
    [
        b"workers: 4\n# TODO(orchestrator): fill in\n",
        b"\xff\xfe",
        b"[1, 2]\n",
        b"key: [unclosed\n",
    ],
    ids=["todo-comment", "not-utf8", "not-a-mapping", "invalid-yaml"],
)
def test_malformed_config_text_is_refused(data):
    with pytest.raises(nbb.BuildError):
        nbb.parse_config(data, Path("config.yaml"))


@pytest.mark.parametrize(
    "changes",
    [
        {"subset": None},
        {"accelerator": "NvidiaL4X1", "workers": 1},
        {"accelerator": "NvidiaL4", "workers": 2},
        {"accelerator": "NvidiaH100", "workers": 8},
        {"solver_end_s": 600, "solver_kill_s": 600},
        {"solver_end_s": 42600, "solver_kill_s": 42900},
        {"accelerator": "NvidiaL4", "run_solver": False},
        {"accelerator": None, "run_solver": True},
        {"solver": "split"},
        {"solver": "nvarc"},
    ],
)
def test_config_boundaries_are_accepted(changes):
    config = nbb.parse_config(config_text(changes), Path("config.yaml"))
    for key, value in changes.items():
        assert getattr(config, key) == (value if key != "subset" or value is None else tuple(value))


# The build


@needs_builder
def test_build_writes_exactly_the_folder_of_a5(builds):
    for folder in (builds.gpu, builds.cpu, builds.full, builds.series):
        assert set(read_files(folder)) == FILES


@needs_builder
def test_rebuild_from_the_same_commit_is_byte_identical(builds):
    assert read_files(builds.cpu) == builds.cpu_first


@needs_builder
def test_builds_of_one_commit_share_one_wheel(builds):
    folders = (builds.gpu, builds.cpu, builds.full, builds.series)
    assert len({(folder / "arc26-code" / WHEEL).read_bytes() for folder in folders}) == 1


@needs_builder
def test_manifest_lists_every_file_with_its_hash(builds):
    for folder in (builds.gpu, builds.cpu, builds.full):
        manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
        rows = re.findall(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|$", manifest, re.MULTILINE)
        files = read_files(folder)
        del files["MANIFEST.md"]
        assert {name: (int(size), digest) for name, size, digest in rows} == {
            name: (len(data), sha256(data)) for name, data in files.items()
        }
        commit = git(builds.repo, "rev-parse", "HEAD").strip()
        assert f"| Commit | `{commit}` |" in manifest
        assert "dataset-create" in manifest and "dataset-version" in manifest


@needs_builder
def test_manifest_gives_the_right_push_command(builds):
    gpu = (builds.gpu / "MANIFEST.md").read_text(encoding="utf-8")
    cpu = (builds.cpu / "MANIFEST.md").read_text(encoding="utf-8")
    assert "`python scripts/kaggle_ops.py push dist/P0.5-01 --run-id <run-id>`" in gpu
    assert "pre-run brief is approved" in gpu
    assert "`python scripts/kaggle_ops.py push dist/P0.5-cpu --timeout 1800`" in cpu


@needs_builder
def test_kernel_metadata_pins_run_a_stack(builds):
    reference = json.loads((REPO / "reference" / "kernel-metadata.json").read_text())
    names = {
        builds.gpu: "arc26-submission",
        builds.full: "arc26-submission",
        builds.cpu: "arc26-submission-cpu-check",
    }
    for folder, notebook in names.items():
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta["id"] == f"alyahya737/{notebook}" and "id_no" not in meta
        assert meta["title"] == notebook.replace("-", " ")
        assert meta["code_file"] == "kaggle_submission.ipynb" and meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True and meta["enable_internet"] is False
        assert meta["docker_image_pinning_type"] == "original"
        assert meta["docker_image"] == reference["docker_image"]
        assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
        assert meta["model_sources"] == reference["model_sources"]
        assert meta["kernel_sources"] == ["sorokin/pip-install-unsloth-flash-patch/2"]
        assert meta["dataset_sources"] == ["alyahya737/arc26-code"]
    gpu = json.loads((builds.gpu / "kernel-metadata.json").read_text())
    cpu = json.loads((builds.cpu / "kernel-metadata.json").read_text())
    assert gpu["enable_gpu"] is True and gpu["machine_shape"] == "NvidiaL4"
    assert cpu["enable_gpu"] is False and "machine_shape" not in cpu


@needs_builder
def test_folders_pass_the_gateway_checks(builds):
    for folder, accelerated in ((builds.gpu, True), (builds.full, True), (builds.cpu, False)):
        meta = ops.load_kernel_metadata(folder.resolve())
        assert ops.uses_gpu(meta, None) is accelerated
        assert ops.accelerator_id(meta, None) == ("NvidiaL4" if accelerated else None)
        ops.check_kernel_metadata(meta, accelerated)
        ops.check_flags(meta, ops.accelerator_id(meta, None))
        ops.check_code_file(folder.resolve(), meta)
        ops.check_dataset_folder((folder / "arc26-code").resolve())


@needs_builder
def test_dataset_metadata_meets_kaggle_cli_rules(builds):
    meta = json.loads((builds.cpu / "arc26-code" / "dataset-metadata.json").read_text())
    assert meta["id"] == "alyahya737/arc26-code" and 6 <= len(meta["title"]) <= 50
    assert meta["licenses"] == [{"name": "other"}]
    assert "isPrivate" not in meta and "public" not in json.dumps(meta).lower()


@needs_builder
def test_wheel_holds_only_the_package(builds):
    with zipfile.ZipFile(builds.cpu / "arc26-code" / WHEEL) as archive:
        names = archive.namelist()
        submit = archive.read("arc26/submit.py")
    assert {name.split("/")[0] for name in names} == {"arc26", "arc26-0.0.1.dist-info"}
    assert "arc26/__init__.py" in names
    assert submit == (builds.repo / "arc26" / "submit.py").read_bytes()


def test_wheel_generator_reads_the_wheel_file(tmp_path):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("arc26-0.0.1.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: hatch 9\n")
    assert bs.wheel_generator(wheel) == "hatch 9"


def test_no_wheel_builder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bs.shutil, "which", lambda name: None)
    monkeypatch.setattr(bs.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(bs.BuildError, match="no wheel builder"):
        bs.build_wheel(tmp_path, tmp_path / "out", 0)


# Refusals, and nothing written


@needs_builder
def test_dirty_or_untracked_trees_are_refused_and_nothing_is_written(tmp_path):
    repo = snapshot(tmp_path / "repo")
    readme = repo / "README.md"
    readme.write_text(readme.read_text() + "\nuncommitted\n")
    with pytest.raises(bs.BuildError, match="not clean") as refused:
        bs.build("t", repo / CPU_CONFIG, repo)
    assert "README.md" in str(refused.value)
    git(repo, "checkout", "--", "README.md")
    (repo / "arc26" / "scratch.py").write_text("x = 1\n")
    with pytest.raises(bs.BuildError, match="scratch.py"):
        bs.build("t", repo / CPU_CONFIG, repo)
    assert not (repo / "dist").exists()


@needs_builder
def test_a_docs_only_commit_keeps_the_wheel_byte_identical(tmp_path):
    """The wheel's timestamps come from the last commit that changed its inputs, so a later
    commit that touches only docs, configs, or scripts leaves its sha256, and the code dataset
    version, unchanged; a change to an input does change it."""
    repo = snapshot(tmp_path / "repo")
    first_commit = git(repo, "rev-parse", "HEAD").strip()
    first = (bs.build("t", repo / CPU_CONFIG, repo) / "arc26-code" / WHEEL).read_bytes()
    for name in ("RESULTS.md", CPU_CONFIG, "scripts/build_submission.py"):
        path = repo / name
        path.write_text(path.read_text(encoding="utf-8") + "\n# a later change\n")
    commit_all(repo, "docs, config, and script only", date="2026-10-02T09:00:00Z")
    folder = bs.build("t", repo / CPU_CONFIG, repo)
    assert (folder / "arc26-code" / WHEEL).read_bytes() == first
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert f"the time of `{first_commit[:12]}`, the last commit that changed" in manifest
    assert f"| Commit | `{git(repo, 'rev-parse', 'HEAD').strip()}` |" in manifest
    readme = repo / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nA readme change.\n")
    commit_all(repo, "readme", date="2026-10-03T09:00:00Z")
    third = (bs.build("t", repo / CPU_CONFIG, repo) / "arc26-code" / WHEEL).read_bytes()
    assert third != first


def test_wheel_inputs_cover_everything_the_wheel_holds():
    assert set(bs.WHEEL_INPUTS) == {"arc26", "pyproject.toml", "README.md", ".gitignore"}
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'readme = "README.md"' in pyproject and 'packages = ["arc26"]' in pyproject


@needs_builder
def test_ignored_files_never_reach_the_wheel(tmp_path):
    repo = snapshot(tmp_path / "repo")
    leak = repo / "arc26" / "build" / "leak.py"  # build/ is in .gitignore
    leak.parent.mkdir()
    leak.write_text("SECRET = 1\n")
    folder = bs.build("t", repo / CPU_CONFIG, repo)
    with zipfile.ZipFile(folder / "arc26-code" / WHEEL) as archive:
        assert not [name for name in archive.namelist() if "leak" in name]


@needs_builder
def test_a_rebuild_removes_stale_files(tmp_path):
    repo = snapshot(tmp_path / "repo")
    stale = repo / "dist" / "t" / "arc26-code" / "arc26-0.0.0-py3-none-any.whl"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    folder = bs.build("t", repo / CPU_CONFIG, repo)
    assert set(read_files(folder)) == FILES


def test_config_outside_the_repository_is_refused(tmp_path):
    outside = tmp_path / "config.yaml"
    outside.write_bytes((REPO / CPU_CONFIG).read_bytes())
    with pytest.raises(bs.BuildError, match="not inside the repository"):
        bs.build("t", outside, REPO)


@needs_builder
def test_uncommitted_config_is_refused(tmp_path):
    repo = snapshot(tmp_path / "repo")
    ignored = repo / "dist" / "config.yaml"  # dist/ is ignored, so the tree stays clean
    ignored.parent.mkdir()
    ignored.write_bytes((repo / CPU_CONFIG).read_bytes())
    with pytest.raises(bs.BuildError, match="is not in commit"):
        bs.build("t", ignored, repo)
    assert sorted(p.name for p in (repo / "dist").iterdir()) == ["config.yaml"]


@needs_builder
def test_refused_config_writes_nothing(tmp_path):
    repo = snapshot(tmp_path / "repo")
    config = repo / CPU_CONFIG
    config.write_text(re.sub(r"^workers: \d+$", "workers: 99", config.read_text(), flags=re.M))
    commit_all(repo)
    with pytest.raises(bs.BuildError, match="workers"):
        bs.build("t", config, repo)
    assert not (repo / "dist").exists()


@pytest.mark.parametrize("tag", ["", ".", "..", "a/b", "x y"])
def test_bad_tags_are_refused(tag):
    with pytest.raises(bs.BuildError, match="--tag"):
        bs.build(tag, REPO / CPU_CONFIG, REPO)


@needs_builder
def test_command_line(tmp_path):
    repo = snapshot(tmp_path / "repo")
    command = [sys.executable, "scripts/build_submission.py", "--tag", "t", "--config", CPU_CONFIG]
    done = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "MANIFEST.md lists every file" in done.stdout
    (repo / "untracked.txt").write_text("x")
    refused = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    assert refused.returncode == 2 and "refused: the working tree is not clean" in refused.stderr


# The notebook


@needs_builder
def test_every_code_cell_follows_a_markdown_cell_and_uses_no_magics(builds):
    notebook = json.loads((builds.gpu / "kaggle_submission.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert "kaggle" not in notebook["metadata"]
    kinds = [cell["cell_type"] for cell in notebook["cells"]]
    assert kinds == ["markdown", "code"] * 8
    assert len({cell["id"] for cell in notebook["cells"]}) == 16
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            compile(text, cell["id"], "exec")
            assert not [line for line in text.splitlines() if line.lstrip().startswith(("!", "%"))]
            assert cell["outputs"] == [] and cell["execution_count"] is None


@needs_builder
def test_notebook_has_no_kaggle_cli_calls_or_em_dashes(builds):
    for folder in (builds.gpu, builds.cpu):
        text = (folder / "kaggle_submission.ipynb").read_text(encoding="utf-8")
        assert not KAGGLE_ACCESS.search(text) and EM_DASH not in text
        assert "import torch" not in text


@needs_builder
def test_config_cell_is_first_and_holds_the_build(builds):
    first = code(builds.gpu)["arc26-02-config"]
    namespace = {}
    exec(first, namespace)
    wheel = (builds.gpu / "arc26-code" / WHEEL).read_bytes()
    assert namespace["SUBSET"] == tuple(DEV4) and namespace["WORKERS"] == 4
    assert namespace["HASH_SEED"] == 0 and namespace["RUN_SOLVER"] is True
    assert (namespace["SOLVER_END_S"], namespace["SOLVER_KILL_S"]) == (3600, 4500)
    build = namespace["BUILD"]
    assert build["wheel"] == WHEEL and build["wheel_sha256"] == sha256(wheel)
    assert build["commit"] == git(builds.repo, "rev-parse", "HEAD").strip()
    assert build["config"] == GPU_CONFIG and build["accelerator"] == "NvidiaL4"
    assert build["config_sha256"] == sha256((builds.repo / GPU_CONFIG).read_bytes())
    assert first.index("NOTEBOOK_START = time.time()") < first.index("SUBSET")


@needs_builder
def test_embedded_sources_are_verbatim(builds):
    cells = code(builds.gpu)
    with zipfile.ZipFile(builds.gpu / "arc26-code" / WHEEL) as archive:
        submit = archive.read("arc26/submit.py").decode("utf-8")
    assert submit.rstrip() in cells["arc26-06-validator"]
    for helper in nbb.RUNTIME_HELPERS:
        assert inspect.getsource(helper).rstrip() in cells["arc26-04-helpers"]
    assert inspect.getsource(reference_run.dump_run_logs).rstrip() in cells["arc26-16-logs"]


@needs_builder
def config_cell_changes(first, second):
    one, two = cells(first), cells(second)
    assert [c["id"] for c in one] == [c["id"] for c in two]
    differing = [a["id"] for a, b in zip(one, two, strict=True) if a["source"] != b["source"]]
    assert set(differing) <= {"arc26-01-config-note", "arc26-02-config"}, differing
    lines = [set("".join(c[1]["source"]).splitlines()) for c in (one, two)]
    return lines[0] ^ lines[1]


@needs_builder
@pytest.mark.parametrize("other", ["cpu", "full"])
def test_builds_differ_only_in_their_config_cell(builds, other):
    changed = config_cell_changes(builds.gpu, getattr(builds, other))
    # The CPU check follows the default solver (split since 2 Oct); P0.5-01 states nvarc.
    settings = ("SUBSET", "WORKERS", "RUN_SOLVER", "SOLVER", "BUILD", " '")
    assert changed and all(line.startswith((*settings, "# from configs/")) for line in changed)


@needs_builder
def test_only_builds_with_the_production_settings_are_fit_to_submit(builds):
    for folder in (builds.gpu, builds.full):
        assert "Not for submission" not in "".join(cells(folder)[0]["source"])
        assert "| fit to submit | yes |" in (folder / "MANIFEST.md").read_text(encoding="utf-8")
    for folder, slug, others in (
        (builds.cpu, "arc26-submission-cpu-check", ["it has no accelerator", "it runs 1 worker"]),
        (builds.series, "arc26-lora-scope", ["it is a series of 5 arms"]),
    ):
        # The notebook comes first, as a reason with its slug, then every other deviation.
        reason = f"its notebook is alyahya737/{slug}, not alyahya737/arc26-submission"
        note = "".join(cells(folder)[0]["source"])
        assert f"**Not for submission.** A development build: {reason}" in note
        assert all(other in note for other in others)
        manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
        verdict = r"^\| fit to submit \| no, not for submission: " + re.escape(reason)
        assert re.search(verdict, manifest, re.MULTILINE)
        row = next(line for line in manifest.splitlines() if line.startswith("| fit to submit"))
        assert all(other in row for other in others)


def test_development_deadlines_do_not_block_a_submission(monkeypatch):
    for changes in ({"solver_end_s": 600, "solver_kill_s": 600}, {"run_solver": False}):
        config = nbb.parse_config(config_text(changes), Path("config.yaml"))
        assert nbb.submission_blocker(config) is None
    probe = nbb.parse_config(config_text({"accelerator": None}), Path("config.yaml"))
    assert "no accelerator" in nbb.submission_blocker(probe)
    monkeypatch.setattr(nbb, "COMPETITION_SOLVER_KILL_S", nbb.SESSION_LIMIT_S)
    config = nbb.parse_config(config_text({}), Path("config.yaml"))
    assert "competition deadlines" in nbb.submission_blocker(config)


def test_competition_deadlines_come_from_the_12_hour_limit():
    assert nbb.COMPETITION_SOLVER_END_S == 12 * 3600 - 30 * 60
    assert nbb.COMPETITION_SOLVER_KILL_S == 12 * 3600 - 20 * 60
    source = nbb.competition_deadlines_source()
    assert f"COMPETITION_SOLVER_END_S = {12 * 3600 - 30 * 60}" in source
    assert f"COMPETITION_SOLVER_KILL_S = {12 * 3600 - 20 * 60}" in source


def test_notebook_command_line_writes_a_notebook(tmp_path):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(REPO / "arc26" / "submit.py", "arc26/submit.py")
    out = tmp_path / "nb" / "kaggle_submission.ipynb"
    argv = ["--config", str(REPO / CPU_CONFIG), "--wheel", str(wheel), "--commit", "ab" * 20]
    assert nbb.main([*argv, "--tag", "t", "--out", str(out)]) == 0
    notebook = json.loads(out.read_text(encoding="utf-8"))
    config_cell = "".join(notebook["cells"][1]["source"])
    assert "'config': 'configs/submission_cpu_check.yaml'" in config_cell
    assert nbb.main([*argv, "--tag", "a/b", "--out", str(out)]) == 2


def test_a_wheel_without_the_validator_is_refused(tmp_path):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("arc26/__init__.py", "")
    with pytest.raises(nbb.BuildError, match="arc26/submit.py"):
        nbb.wheel_member(wheel)


# Runtime helpers, called directly


def test_find_inputs_searches_by_name_within_the_depth(tmp_path):
    shallow = tmp_path / "a" / "x.json"
    deep = tmp_path.joinpath(*"bcdefgh") / "x.json"  # level 8
    too_deep = tmp_path.joinpath(*"bcdefghi") / "x.json"  # level 9
    hidden = tmp_path / "lib" / "site-packages" / "x.json"
    for path in (shallow, deep, too_deep, hidden):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    found = nbb.find_inputs(str(tmp_path), ["x.json", "absent.json"], 8)
    assert found == {"x.json": sorted([str(shallow), str(deep)]), "absent.json": []}
    assert nbb.find_inputs(str(tmp_path / "missing"), ["x.json"], 8) == {"x.json": []}


def test_run_logged_reports_failures_without_raising(tmp_path, capsys):
    ok = nbb.run_logged([sys.executable, "-c", "print('hello')"])
    assert ok.returncode == 0 and ok.stdout == "hello\n"
    missing = nbb.run_logged([str(tmp_path / "absent")])
    assert missing.returncode == 127
    slow = nbb.run_logged([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2)
    assert slow.returncode == 124
    assert "exit code 0" in capsys.readouterr().out


def test_gpu_inventory_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert nbb.gpu_inventory() == []


def test_file_sha256(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"x" * (3 << 20))
    assert nbb.file_sha256(str(path)) == sha256(b"x" * (3 << 20))


# Series mode and the production guard (task P1.3)

NINE = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
]  # fmt: skip
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
REFERENCE_ARM = {"name": "ref", "lora_r": 256, "lora_targets": NINE, "reason": "the notebook"}


def arm(name="small", lora_r=16, lora_targets=ATTENTION, reason="H's default scope"):
    return {"name": name, "lora_r": lora_r, "lora_targets": lora_targets, "reason": reason}


def test_p1_3_01_config_is_the_dev40_scope_series_on_4xl4():
    from arc26.data import DEV40

    gpu, series = load_config(GPU_CONFIG), load_config(SERIES_CONFIG)
    assert series.kaggle_notebook == "alyahya737/arc26-lora-scope"
    timing = load_config(TIMING_CONFIG)
    assert series.kaggle_notebook not in {gpu.kaggle_notebook, timing.kaggle_notebook}
    assert series.subset == DEV40 and len(series.subset) == 40
    assert series.accelerator == "NvidiaL4" and series.workers == 4 and series.hash_seed == 0
    assert series.run_solver is True
    assert series.solver_end_s == nbb.COMPETITION_SOLVER_END_S
    assert series.solver_kill_s == nbb.COMPETITION_SOLVER_KILL_S
    assert {f: getattr(series, f) for f in STACK} == {f: getattr(gpu, f) for f in STACK}
    # The brief's priority order; the optional sixth arm (r 256 without the embeddings) is left
    # out.
    assert [(a.name, a.lora_r, list(a.lora_targets)) for a in series.arms] == [
        ("r256-proj-emb", 256, NINE),
        ("r16-attn", 16, ATTENTION),
        ("r32-attn", 32, ATTENTION),
        ("r16-proj", 16, NINE[:7]),
        ("r64-proj", 64, NINE[:7]),
    ]
    assert all(a.reason and len(a.reason) <= nbb.MAX_REASON for a in series.arms)
    assert "embed_tokens" in series.arms[0].reason
    assert all("embeddings frozen" in a.reason for a in series.arms[1:])


def test_p1_3_01_kernel_metadata_passes_the_gateway_on_4xl4():
    config = load_config(SERIES_CONFIG)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)  # runs the gateway's metadata checks
    assert meta["machine_shape"] == "NvidiaL4" and meta["id"] == "alyahya737/arc26-lora-scope"


def test_a_config_without_arms_is_a_single_run():
    for name in (GPU_CONFIG, CPU_CONFIG, FULL_CONFIG, TIMING_CONFIG):
        assert load_config(name).arms is None
    assert nbb.parse_config(config_text({"arms": None}), Path("config.yaml")).arms is None


@pytest.mark.parametrize(
    "arms",
    [
        [REFERENCE_ARM],
        [REFERENCE_ARM, arm()],
        [REFERENCE_ARM, arm("tiny", 1, ["q_proj"]), arm("r256-proj", 256, NINE[:7])],
        [REFERENCE_ARM, arm("mlp-head", 8, ["down_proj", "lm_head"])],
    ],
)
def test_series_configs_are_accepted(arms):
    config = nbb.parse_config(config_text({"arms": arms}), Path("config.yaml"))
    assert [a._asdict() for a in config.arms] == [
        {**a, "lora_targets": tuple(a["lora_targets"]), "holdout": None, "rounds": None}
        for a in arms
    ]


@pytest.mark.parametrize(
    ("arms", "message"),
    [
        ([], "arms must be null or a non-empty list"),
        ({"name": "ref"}, "arms must be null or a non-empty list"),
        ([REFERENCE_ARM, "small"], "arm 2 must have exactly the keys"),
        ([REFERENCE_ARM, {**arm(), "lr": 1e-4}], "arm 2 must have exactly the keys"),
        ([REFERENCE_ARM, {k: v for k, v in arm().items() if k != "reason"}], "exactly the keys"),
        ([REFERENCE_ARM, arm(name="R16-attn")], "name must be 1 to 40 lowercase"),
        ([REFERENCE_ARM, arm(name="-r16")], "name must be 1 to 40 lowercase"),
        ([REFERENCE_ARM, arm(name="r16/attn")], "name must be 1 to 40 lowercase"),
        ([REFERENCE_ARM, arm(name="a" * 41)], "name must be 1 to 40 lowercase"),
        ([REFERENCE_ARM, arm(name="ref")], "arm names must be distinct"),
        ([REFERENCE_ARM, arm(lora_r=0)], "lora_r must be a positive integer"),
        ([REFERENCE_ARM, arm(lora_r=True)], "lora_r must be a positive integer"),
        ([REFERENCE_ARM, arm(lora_r=16.0)], "lora_r must be a positive integer"),
        ([REFERENCE_ARM, arm(lora_targets=[])], "lora_targets must be distinct names"),
        ([REFERENCE_ARM, arm(lora_targets="q_proj")], "lora_targets must be distinct names"),
        ([REFERENCE_ARM, arm(lora_targets=["qkv_proj"])], "lora_targets must be distinct names"),
        ([REFERENCE_ARM, arm(lora_targets=["q_proj", "q_proj"])], "distinct names"),
        ([REFERENCE_ARM, arm(lora_targets=["k_proj", "q_proj"])], "in that order"),
        ([REFERENCE_ARM, arm(lora_targets=["embed_tokens", "lm_head"])], "at least one of"),
        ([REFERENCE_ARM, arm(reason="")], "reason must be one line"),
        ([REFERENCE_ARM, arm(reason="two\nlines")], "reason must be one line"),
        ([REFERENCE_ARM, arm(reason=7)], "reason must be one line"),
        ([REFERENCE_ARM, arm(reason="x" * 201)], "at most 200 characters"),
        ([arm()], "the first arm must be the notebook configuration"),
        ([arm(lora_r=256, lora_targets=NINE[:7])], "the first arm must be the notebook"),
        ([{**REFERENCE_ARM, "lora_r": 128}], "the first arm must be the notebook configuration"),
    ],
)
def test_series_config_refusals(arms, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(config_text({"arms": arms}), Path("config.yaml"))
    assert message in str(refused.value)


def test_the_production_settings_are_fit_to_submit():
    for name in (GPU_CONFIG, FULL_CONFIG):
        config = load_config(name)
        assert (config.accelerator, config.workers, config.arms) == ("NvidiaL4", 4, None)
        assert nbb.submission_blocker(config) is None
        assert "Not for submission" not in nbb.config_note(config, {**BUILD_STUB})


BUILD_STUB = {"tag": "t", "commit": "0" * 40, "config": "configs/x.yaml"}


@pytest.mark.parametrize(
    ("changes", "reasons"),
    [
        ({"accelerator": "NvidiaL4X1", "workers": 1}, ["NvidiaL4X1, not NvidiaL4", "1 worker,"]),
        ({"accelerator": "NvidiaTeslaA100", "workers": 2}, ["NvidiaTeslaA100", "2 workers, not 4"]),
        ({"workers": 3}, ["it runs 3 workers, not 4"]),
        ({"accelerator": None}, ["it has no accelerator"]),
        ({"accelerator": "NvidiaH100", "workers": 4}, ["NvidiaH100, not NvidiaL4"]),
        ({"arms": [REFERENCE_ARM]}, ["it is a series of 1 arm,"]),
        ({"arms": [REFERENCE_ARM, arm()], "workers": 1}, ["series of 2 arms", "1 worker"]),
    ],
)
def test_development_builds_are_not_for_submission(changes, reasons):
    config = nbb.parse_config(config_text(changes), Path("config.yaml"))
    blocker = nbb.submission_blocker(config)
    assert blocker is not None and all(reason in blocker for reason in reasons)
    assert len(blocker.split("; ")) == len(reasons)
    note = nbb.config_note(config, BUILD_STUB)
    assert note.startswith("# arc26 submission notebook\n\n**Not for submission.** A development")
    assert all(reason in note for reason in reasons)


def test_the_committed_development_configs_are_not_for_submission():
    # Every committed config for a notebook other than the submission notebook: the notebook is
    # the first reason, followed by every other deviation from production, and nothing else.
    reasons = {
        TIMING_CONFIG: ["its notebook is alyahya737/arc26-timing"],
        CPU_CONFIG: [
            "its notebook is alyahya737/arc26-submission-cpu-check",
            "it has no accelerator",
            "it runs 1 worker, not 4",
        ],
        SERIES_CONFIG: ["its notebook is alyahya737/arc26-lora-scope", "it is a series of 5 arms"],
        SPLIT_CONFIG: ["its notebook is alyahya737/arc26-split-solver"],
        HOLDOUT_CONFIG: [
            "its notebook is alyahya737/arc26-holdout-gate",
            "it is a series of 1 arm, built to compare LoRA scopes and the held-out gate",
        ],
    }
    for name, expected in reasons.items():
        parts = nbb.submission_blocker(load_config(name)).split("; ")
        assert len(parts) == len(expected)
        assert all(part.startswith(reason) for part, reason in zip(parts, expected, strict=True))


@pytest.mark.parametrize(
    "notebook",
    ["alyahya737/arc26-timing", "alyahya737/arc26-scratch", "alyahya737/arc26-submission-two"],
)
def test_a_production_shaped_build_for_another_notebook_is_not_for_submission(notebook):
    # P0.5-01's settings (four workers on NvidiaL4, no series), pushed to another notebook.
    config = nbb.parse_config(config_text({"kaggle_notebook": notebook}), Path("config.yaml"))
    assert (config.accelerator, config.workers, config.arms) == ("NvidiaL4", 4, None)
    reason = (
        f"its notebook is {notebook}, not {nbb.SUBMISSION_NOTEBOOK}, the only notebook "
        "submissions come from"
    )
    assert nbb.submission_blocker(config) == reason
    note = nbb.config_note(config, BUILD_STUB)
    assert f"**Not for submission.** A development build: {reason}." in note


def test_the_submission_notebook_is_the_production_configs_notebook():
    assert nbb.SUBMISSION_NOTEBOOK == "alyahya737/arc26-submission"
    for name in (GPU_CONFIG, FULL_CONFIG):
        assert load_config(name).kaggle_notebook == nbb.SUBMISSION_NOTEBOOK


@needs_builder
def test_the_config_cell_holds_the_series_arms(builds):
    namespace = {}
    exec(code(builds.series)["arc26-02-config"], namespace)
    config = load_config(SERIES_CONFIG)
    # An arm without a held-out pair keeps task P1.3's four keys (task P2.1).
    assert namespace["ARMS"] == [
        {"name": a.name, "lora_r": a.lora_r, "lora_targets": list(a.lora_targets)}
        | {"reason": a.reason}
        for a in config.arms
    ]
    assert namespace["SUBSET"] == config.subset and namespace["WORKERS"] == 4
    for folder in (builds.gpu, builds.cpu, builds.full):
        namespace = {}
        exec(code(folder)["arc26-02-config"], namespace)
        assert namespace["ARMS"] is None


@needs_builder
def test_a_series_build_differs_from_a_single_run_only_in_its_config_cell(builds):
    changed = config_cell_changes(builds.full, builds.series)
    assert "ARMS = None" in changed
    assert any(line.startswith("ARMS = [{'name': 'r256-proj-emb'") for line in changed)


@needs_builder
def test_the_manifest_lists_the_arms(builds):
    series = (builds.series / "MANIFEST.md").read_text(encoding="utf-8")
    row = next(line for line in series.splitlines() if line.startswith("| arms "))
    assert "`r256-proj-emb` (r 256: q_proj, k_proj" in row and "`r64-proj` (r 64:" in row
    full = (builds.full / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| arms (series, development runs only) | none (one run" in full


# The series stop rule, called directly


def test_the_first_arm_always_starts():
    assert nbb.series_refusal([], None, 0.0, 41400.0, 1.25) is None
    assert nbb.series_refusal([], None, 50000.0, 41400.0, 1.25) is None


def test_an_arm_starts_while_the_time_left_covers_the_margin_times_the_longest_arm():
    # 1.25 * 1000 = 1250 s needed; exactly 1250 s left still starts it.
    assert nbb.series_refusal([1000.0, 800.0], None, 1750.0, 3000.0, 1.25) is None
    refusal = nbb.series_refusal([800.0, 1000.0], None, 1750.1, 3000.0, 1.25)
    assert refusal.startswith("1250 s are left before the soft deadline")
    assert "1250 s, 1.25 times the longest arm so far (1000 s)" in refusal


def test_no_arm_starts_after_a_failed_one():
    assert nbb.series_refusal([60.0], "r16-attn", 100.0, 41400.0, 1.25) == (
        "arm r16-attn failed, so no later arm starts"
    )


# The solver's code (task P2.0): an optional key, both modes; the split solver by default since
# run P2.0-01 passed and the human approved the flip (2 Oct, ADR-011).


def test_the_builders_solvers_are_the_runners():
    assert nbb.SOLVERS == tuple(runner.SOLVER_MODULES)
    assert nbb.DEFAULT_SOLVER == runner.RunConfig.__dataclass_fields__["solver"].default
    assert nbb.DEFAULT_SOLVER == "split"


def test_runs_before_the_flip_state_the_moved_reference_and_a_config_without_solver_splits():
    for name in (GPU_CONFIG, FULL_CONFIG, TIMING_CONFIG, SERIES_CONFIG):
        assert yaml.safe_load((REPO / name).read_text(encoding="utf-8"))["solver"] == "nvarc"
        assert load_config(name).solver == "nvarc"
    assert "solver" not in yaml.safe_load((REPO / CPU_CONFIG).read_text(encoding="utf-8"))
    assert load_config(CPU_CONFIG).solver == "split"


def test_p2_0_01_config_repeats_p0_5_01_with_the_split_solver():
    gpu, split = load_config(GPU_CONFIG), load_config(SPLIT_CONFIG)
    assert split.solver == "split"
    # Everything else is P0.5-01's, which repeats run A: DEV4, four workers on NvidiaL4, hash
    # seed 0, the deadlines, image, install notebook, model, and code dataset.
    assert split._replace(kaggle_notebook=gpu.kaggle_notebook, solver=gpu.solver) == gpu
    others = (GPU_CONFIG, CPU_CONFIG, FULL_CONFIG, TIMING_CONFIG, SERIES_CONFIG)
    assert split.kaggle_notebook == "alyahya737/arc26-split-solver"
    assert split.kaggle_notebook not in {load_config(name).kaggle_notebook for name in others}
    pulled = json.loads((REPO / split.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(split, pulled)  # runs the gateway's metadata checks
    assert meta["machine_shape"] == "NvidiaL4" and meta["id"] == split.kaggle_notebook


def test_the_solver_is_not_part_of_the_submission_rule():
    # submission_blocker does not read the solver: ADR-011 lets either submit (task P2.0).
    production = load_config(GPU_CONFIG)
    assert nbb.submission_blocker(production._replace(solver="split")) is None
    split = load_config(SPLIT_CONFIG)
    assert nbb.submission_blocker(split) == nbb.submission_blocker(
        split._replace(solver="nvarc")
    )


def test_the_config_cell_and_note_name_the_solver():
    for name, solver in ((GPU_CONFIG, "nvarc"), (SPLIT_CONFIG, "split")):
        config = load_config(name)
        namespace = {}
        exec(nbb.config_cell_source(config, BUILD_STUB), namespace)
        assert namespace["SOLVER"] == solver
        note = nbb.config_note(config, BUILD_STUB)
        # Only a build off the default (split since 2 Oct) says which solver it runs.
        assert ("Moved-reference build" in note) is (solver == "nvarc")
        assert "Split-solver build" not in note
    note = nbb.config_note(load_config(GPU_CONFIG), BUILD_STUB)
    assert "`SOLVER = 'nvarc'`" in note and "`python -m arc26.nvarc.starter`" in note


@needs_builder
def test_the_split_build_differs_from_p0_5_01_only_in_its_config_cell(builds):
    assert set(read_files(builds.split)) == FILES
    changed = config_cell_changes(builds.gpu, builds.split)
    assert {"SOLVER = 'nvarc'", "SOLVER = 'split'"} <= changed
    settings = ("SOLVER = ", "BUILD", " '", "# from configs/")
    assert all(line.startswith(settings) for line in changed), changed
    assert (builds.split / "arc26-code" / WHEEL).read_bytes() == (
        builds.gpu / "arc26-code" / WHEEL
    ).read_bytes()
    with zipfile.ZipFile(builds.split / "arc26-code" / WHEEL) as archive:
        assert {"arc26/solver.py", "arc26/ttt.py", "arc26/decode.py"} <= set(archive.namelist())


@needs_builder
def test_the_split_build_passes_the_gateway_and_is_not_for_submission(builds):
    folder = builds.split
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == "alyahya737/arc26-split-solver" and meta["machine_shape"] == "NvidiaL4"
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | no, not for submission: its notebook is alyahya737/arc26-split" in (
        manifest
    )


@needs_builder
def test_the_manifest_names_the_solver(builds):
    assert set(bs.SOLVER_NOTES) == set(nbb.SOLVERS)
    gpu = (builds.gpu / "MANIFEST.md").read_text(encoding="utf-8")
    split = (builds.split / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| solver (both modes) | `nvarc`: the moved reference, " in gpu
    row = next(line for line in split.splitlines() if line.startswith("| solver "))
    assert row.startswith("| solver (both modes) | `split`: ") and "`python -m arc26.solver`" in row


# Hold-out arms (task P2.1): an optional key of an arm, split solver only, development runs only.

HOLDOUT = {"rule": "last", "k2": True, "in_test_prompts": False}


def test_the_builders_hold_out_settings_are_the_loops():
    from arc26 import loop

    assert nbb.HOLDOUT_RULES == tuple(loop.HOLDOUT_RULES)
    assert nbb.HOLDOUT_KEYS == tuple(f for f in loop.HoldOut.FIELDS if not f.endswith("_dir"))
    assert nbb.HOLDOUT_SOLVER == "split" and nbb.HOLDOUT_SOLVER in runner.SOLVER_MODULES


def test_p2_1_01_config_is_the_notebook_configuration_with_a_held_out_pair_on_dev40():
    from arc26.data import DEV40

    gpu, series, held = (load_config(n) for n in (GPU_CONFIG, SERIES_CONFIG, HOLDOUT_CONFIG))
    others = (GPU_CONFIG, CPU_CONFIG, FULL_CONFIG, TIMING_CONFIG, SERIES_CONFIG, SPLIT_CONFIG)
    assert held.kaggle_notebook == "alyahya737/arc26-holdout-gate"
    assert held.kaggle_notebook not in {load_config(name).kaggle_notebook for name in others}
    assert held.subset == DEV40 == series.subset
    assert (held.accelerator, held.workers, held.hash_seed) == ("NvidiaL4", 4, 0)
    assert held.run_solver is True and held.solver == "split"
    assert (held.solver_end_s, held.solver_kill_s) == (14400, 15300)
    assert {f: getattr(held, f) for f in STACK} == {f: getattr(gpu, f) for f in STACK}
    # One arm: P1.3-01's first arm, the hold-out-off arm, with the gate's settings.
    (arm,) = held.arms
    assert arm.name == "r256-proj-emb-holdout" and len(arm.reason) <= nbb.MAX_REASON
    assert (arm.lora_r, arm.lora_targets) == (series.arms[0].lora_r, series.arms[0].lora_targets)
    assert arm.holdout == nbb.ArmHoldOut("last", True, False)
    assert series.arms[0].holdout is None and all(a.holdout is None for a in series.arms)
    pulled = json.loads((REPO / held.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(held, pulled)  # runs the gateway's metadata checks
    assert meta["machine_shape"] == "NvidiaL4" and meta["id"] == held.kaggle_notebook


@pytest.mark.parametrize(
    "holdout",
    [
        HOLDOUT,
        {"rule": "first", "k2": False, "in_test_prompts": True},
        None,
    ],
)
def test_hold_out_arms_are_accepted(holdout):
    arms = [REFERENCE_ARM, {**arm(), "holdout": holdout}]
    config = nbb.parse_config(config_text({"arms": arms, "solver": "split"}), Path("config.yaml"))
    expected = None if holdout is None else nbb.ArmHoldOut(**holdout)
    assert [a.holdout for a in config.arms] == [None, expected]
    # The notebook configuration may hold a pair out too: its LoRA settings are the reference's.
    first = [{**REFERENCE_ARM, "holdout": HOLDOUT}]
    config = nbb.parse_config(config_text({"arms": first, "solver": "split"}), Path("c.yaml"))
    assert config.arms[0].holdout == nbb.ArmHoldOut("last", True, False)


@pytest.mark.parametrize(
    ("holdout", "message"),
    [
        ({"rule": "last", "k2": True}, "holdout must be null or have exactly the keys"),
        ({**HOLDOUT, "views": 16}, "holdout must be null or have exactly the keys"),
        ({**HOLDOUT, "gate_dir": "/kaggle/gate"}, "holdout must be null or have exactly the keys"),
        ("last", "holdout must be null or have exactly the keys"),
        (True, "holdout must be null or have exactly the keys"),
        ({**HOLDOUT, "rule": "middle"}, "holdout rule must be one of last, first"),
        ({**HOLDOUT, "rule": None}, "holdout rule must be one of last, first"),
        ({**HOLDOUT, "k2": "yes"}, "holdout k2 must be true or false"),
        ({**HOLDOUT, "k2": 1}, "holdout k2 must be true or false"),
        ({**HOLDOUT, "in_test_prompts": None}, "holdout in_test_prompts must be true or false"),
    ],
)
def test_hold_out_arm_refusals(holdout, message):
    arms = [REFERENCE_ARM, {**arm(), "holdout": holdout}]
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(config_text({"arms": arms, "solver": "split"}), Path("config.yaml"))
    assert message in str(refused.value) and "arm 2" in str(refused.value)


def test_a_hold_out_arm_needs_the_split_solver():
    arms = [REFERENCE_ARM, {**arm(), "holdout": HOLDOUT}]
    needs = "arm small holds a pair out, which needs solver: split"
    with pytest.raises(nbb.BuildError, match=needs):
        nbb.parse_config(config_text({"arms": arms, "solver": "nvarc"}), Path("config.yaml"))
    # Without the key the split solver runs (the default since 2 Oct), so the arm is accepted.
    held = nbb.parse_config(config_text({"arms": arms, "solver": DROP}), Path("config.yaml"))
    assert held.solver == "split" and held.arms[1].holdout is not None


def test_the_config_cell_and_note_describe_a_hold_out_arm():
    held, series = load_config(HOLDOUT_CONFIG), load_config(SERIES_CONFIG)
    namespace = {}
    source = nbb.config_cell_source(held, BUILD_STUB)
    exec(source, namespace)
    (entry,) = namespace["ARMS"]
    assert entry["holdout"] == HOLDOUT and entry["lora_targets"] == list(NINE)
    assert namespace["SOLVER"] == "split"
    assert all(line in source.splitlines() for line in nbb.HOLDOUT_COMMENT)
    # A series without a hold-out arm keeps the Config cell of task P1.3.
    plain = nbb.config_cell_source(series, BUILD_STUB)
    assert not any(line in plain.splitlines() for line in nbb.HOLDOUT_COMMENT)
    assert "holdout" not in plain
    note = nbb.config_note(held, BUILD_STUB)
    assert "Arm `r256-proj-emb-holdout` runs the held-out gate (task P2.1" in note
    assert f"ADR-013). Its {nbb.holdout_text(HOLDOUT)}.\n" in note
    assert "held-out gate" not in nbb.config_note(series, BUILD_STUB)


@pytest.mark.parametrize(
    ("settings", "text"),
    [
        (
            HOLDOUT,
            "hold-out: the last real pair of each puzzle with 2 or more, kept out of training "
            "and of every prompt",
        ),
        (
            {"rule": "first", "k2": False, "in_test_prompts": True},
            "hold-out: the first real pair of each puzzle with 3 or more, kept out of training "
            "and of the held-out input's prompts (kept in the test inputs' prompts)",
        ),
    ],
)
def test_holdout_text_states_the_settings(settings, text):
    assert nbb.holdout_text(settings) == text


@needs_builder
def test_the_hold_out_build_is_not_for_submission_and_describes_its_arm(builds):
    folder = builds.holdout
    assert set(read_files(folder)) == FILES
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == "alyahya737/arc26-holdout-gate" and meta["machine_shape"] == "NvidiaL4"
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | no, not for submission: its notebook is alyahya737/arc26-holdout" in (
        manifest
    )
    row = next(line for line in manifest.splitlines() if line.startswith("| arms "))
    assert "`r256-proj-emb-holdout` (r 256: q_proj, k_proj" in row
    assert f"lm_head; {nbb.holdout_text(HOLDOUT)})" in row
    series = (builds.series / "MANIFEST.md").read_text(encoding="utf-8")
    assert "hold-out" not in series
    # The same commit, so the same wheel as every other build of the fixture.
    assert (folder / "arc26-code" / WHEEL).read_bytes() == (
        builds.gpu / "arc26-code" / WHEEL
    ).read_bytes()
    with zipfile.ZipFile(folder / "arc26-code" / WHEEL) as archive:
        assert "arc26/loop.py" in archive.namelist()


@needs_builder
def test_the_hold_out_build_differs_from_the_split_build_only_in_its_first_two_cells(builds):
    changed = config_cell_changes(builds.split, builds.holdout)
    assert any(line.startswith("ARMS = [{'name': 'r256-proj-emb-holdout'") for line in changed)
    assert "ARMS = None" in changed


# Script mode (task P0.5d): a kind: script config, its packaging, and its notebook.

SCRIPT_FILES = {
    "MANIFEST.md",
    "kaggle_script.ipynb",
    "kernel-metadata.json",
    f"arc26-code/{WHEEL}",
    "arc26-code/dataset-metadata.json",
    "arc26-code/env_probe.py",
    "arc26-code/script_smoke.yaml",
}
# The guard in front of the submission notebook's Install cell in a script notebook.
INSTALL_GUARD = (
    '    if PROBLEMS:\n        STAGES["install"]["skipped"] = "an earlier stage stopped the run"\n'
    "    elif "
)


def script_code(folder):
    notebook = json.loads((folder / "kaggle_script.ipynb").read_text(encoding="utf-8"))
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}


def script_text(changes):
    raw = yaml.safe_load((REPO / SCRIPT_CONFIG).read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is DROP:
            raw.pop(key)
        else:
            raw[key] = value
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def test_script_smoke_config_runs_the_env_probe_on_cpu_on_the_same_stack():
    gpu, script = load_config(GPU_CONFIG), load_config(SCRIPT_CONFIG)
    assert isinstance(script, nbb.ScriptConfig)
    assert script.kaggle_notebook == "alyahya737/arc26-script-smoke"
    assert script.accelerator is None and script.script == "scripts/env_probe.py"
    assert script.script_config == SCRIPT_CONFIG and script.args == ()
    assert script.script_kill_s == 600
    assert {f: getattr(script, f) for f in STACK} == {f: getattr(gpu, f) for f in STACK}
    # A CPU push needs a notebook that no pre-run brief names, and its own slug.
    assert script.kaggle_notebook not in ops.briefed_notebooks()
    others = (GPU_CONFIG, CPU_CONFIG, FULL_CONFIG, TIMING_CONFIG, SERIES_CONFIG, SPLIT_CONFIG)
    assert script.kaggle_notebook not in {load_config(name).kaggle_notebook for name in others}


def test_the_smoke_script_ignores_its_config_argument():
    # env_probe.py reads no config: it ignores unknown arguments, so --config costs nothing.
    probe = load("env_probe", SCRIPTS / "env_probe.py")
    args, unknown = probe.build_parser().parse_known_args(["--config", "/kaggle/x.yaml"])
    assert args.root == "/" and unknown == ["--config", "/kaggle/x.yaml"]


def test_a_submission_config_may_name_its_kind():
    config = nbb.parse_config(config_text({"kind": "submission"}), Path("config.yaml"))
    assert config == load_config(GPU_CONFIG)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (config_text({"kind": "Script"}), "kind must be one of submission, script; got 'Script'"),
        (config_text({"kind": None}), "kind must be one of submission, script"),
        (
            config_text({"kind": "script"}),
            "missing keys ['args', 'script', 'script_config', 'script_kill_s']",
        ),
        (script_text({"kind": DROP}), "unknown keys ['args', 'script', 'script_config', 'script_"),
        (script_text({"script": "env_probe.py"}), "script must be a relative path to a .py file"),
        (script_text({"script": "/scripts/env_probe.py"}), "script must be a relative path"),
        (script_text({"script": "scripts/../tests/x.py"}), "script must be a relative path"),
        (script_text({"script": "scripts/env_probe.sh"}), "script must be a relative path"),
        (script_text({"script": "tests/testdata/tiny_script.py"}), "script must be a relative"),
        (script_text({"script": ["scripts/env_probe.py"]}), "script must be a relative path"),
        (script_text({"script_config": "x.yaml"}), "script_config must be a relative path to a"),
        (script_text({"script_config": "configs/x.json"}), "script_config must be a relative"),
        (script_text({"script_config": "configs\\x.yaml"}), "script_config must be a relative"),
        (script_text({"args": "--smoke"}), "args must be a list of at most 32 non-empty one-line"),
        (script_text({"args": [""]}), "args must be a list"),
        (script_text({"args": ["a\nb"]}), "args must be a list"),
        (script_text({"args": [1]}), "args must be a list"),
        (script_text({"args": ["x"] * 33}), "args must be a list of at most 32"),
        (script_text({"args": ["--config", "y.yaml"]}), "args must not set --config ('--config')"),
        (script_text({"args": ["--config=y.yaml"]}), "args must not set --config"),
        (script_text({"args": ["--smoke", "--conf", "y.yaml"]}), "set --config ('--conf')"),
        (script_text({"args": ["-c", "y.yaml"]}), "args must not set --config ('-c')"),
        (script_text({"args": ["-cy.yaml"]}), "args must not set --config ('-cy.yaml')"),
        (script_text({"args": ["-c=y.yaml"]}), "args must not set --config ('-c=y.yaml')"),
        (script_text({"accelerator": "NvidiaTeslaT4"}), "NvidiaTeslaT4 can never pass"),
        (script_text({"script_kill_s": 299}), "script_kill_s must be an integer from 300 to 42900"),
        (script_text({"script_kill_s": 42901}), "script_kill_s must be an integer"),
        (script_text({"script_kill_s": 600.0}), "script_kill_s must be an integer"),
        (script_text({"script_kill_s": True}), "script_kill_s must be an integer"),
        (
            script_text({"kaggle_notebook": "alyahya737/arc26-submission"}),
            "a script notebook is never submitted, so it cannot be alyahya737/arc26-submission",
        ),
        (script_text({"kaggle_notebook": "someone/arc26-script-smoke"}), "KAGGLE_OWNER"),
        (script_text({"accelerator": "NvidiaTeslaP100"}), "accelerator must be null (CPU) or"),
        (script_text({"workers": 4}), "unknown keys ['workers']"),
        (script_text({"run_solver": True, "arms": None}), "unknown keys ['arms', 'run_solver']"),
        (script_text({"script_kill_s": DROP}), "missing keys ['script_kill_s']"),
        (script_text({"model_slug": "\x54ODO"}), "TODO"),
    ],
)
def test_script_config_refusals(text, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(text, Path("config.yaml"))
    assert message in str(refused.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"script_kill_s": 300},
        {"script_kill_s": 42900},
        {"accelerator": "NvidiaL4"},
        {"args": ["--smoke", "--commit", "{commit}"]},
        {"args": ["--configure"]},
        {"args": ["-v", "--cache-dir", "x", "--"]},
        {"script": "scripts/tools/probe.py", "script_config": "configs/probe.yml"},
    ],
)
def test_script_config_boundaries_are_accepted(changes):
    config = nbb.parse_config(script_text(changes), Path("config.yaml"))
    for key, value in changes.items():
        assert getattr(config, key) == (tuple(value) if key == "args" else value)


def test_check_script_files_refuses_what_would_fail_on_kaggle():
    config = load_config(SCRIPT_CONFIG)
    script, script_config = b"print('hi')\n", b"out_dir: run_logs\n"
    assert nbb.check_script_files(config, script, script_config) == {
        "env_probe.py": script,
        "script_smoke.yaml": script_config,
    }
    for bad_script, bad_config, message in [
        (b"def broken(:\n", script_config, "scripts/env_probe.py does not compile"),
        (b"\xff\xfe", script_config, "scripts/env_probe.py does not compile"),
        (b"x = 1\0", script_config, "scripts/env_probe.py does not compile"),
        (script, b"\xff", "configs/script_smoke.yaml is not UTF-8"),
        (script, b"steps: 3  # TODO\n", "configs/script_smoke.yaml still has TODO"),
        (script, b"key: [\n", "configs/script_smoke.yaml is not valid YAML"),
        (script, b"[1, 2]\n", "configs/script_smoke.yaml must hold a mapping"),
    ]:
        with pytest.raises(nbb.BuildError, match=re.escape(message)):
            nbb.check_script_files(config, bad_script, bad_config)


def test_build_info_of_a_script_build_names_the_script_and_its_config():
    common = {
        "tag": "t", "commit": "0" * 40, "config_file": SCRIPT_CONFIG, "config_sha256": "1" * 64,
        "wheel": WHEEL, "wheel_sha256": "2" * 64,
    }  # fmt: skip
    script = load_config(SCRIPT_CONFIG)
    info = nbb.build_info(script, **common, script_sha256="3" * 64, script_config_sha256="4" * 64)
    assert list(info)[-5:] == [
        "kind", "script", "script_sha256", "script_config", "script_config_sha256",
    ]  # fmt: skip
    assert (info["script"], info["script_config"]) == ("scripts/env_probe.py", SCRIPT_CONFIG)
    with pytest.raises(nbb.BuildError, match="needs the sha256 of its script and its config"):
        nbb.build_info(script, **common, script_sha256="3" * 64)
    # A submission build's provenance is as before task P0.5d.
    assert list(nbb.build_info(load_config(GPU_CONFIG), **common)) == [
        "tag", "commit", "config", "config_sha256", "wheel", "wheel_sha256", "kaggle_notebook",
        "code_dataset", "competition", "model_slug", "accelerator",
    ]  # fmt: skip


@needs_builder
def test_a_script_build_holds_the_script_and_its_config_byte_for_byte(builds):
    files = read_files(builds.script)
    assert set(files) == SCRIPT_FILES
    assert files["arc26-code/env_probe.py"] == (builds.repo / "scripts/env_probe.py").read_bytes()
    assert files["arc26-code/script_smoke.yaml"] == (builds.repo / SCRIPT_CONFIG).read_bytes()
    # The wheel of every build of the commit: one code dataset version serves them all.
    assert files[f"arc26-code/{WHEEL}"] == (builds.gpu / "arc26-code" / WHEEL).read_bytes()


@needs_builder
def test_the_script_manifest_lists_every_file_and_says_it_is_never_submitted(builds):
    manifest = (builds.script / "MANIFEST.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|$", manifest, re.MULTILINE)
    files = read_files(builds.script)
    del files["MANIFEST.md"]
    assert {name: (int(size), digest) for name, size, digest in rows} == {
        name: (len(data), sha256(data)) for name, data in files.items()
    }
    for row in (
        "| fit to submit | no, a script notebook is never submitted |",
        "| kind | `script` |",
        "| accelerator | CPU |",
        "| script | `scripts/env_probe.py`, as `arc26-code/env_probe.py` |",
        "| script_config | `configs/script_smoke.yaml`, as `arc26-code/script_smoke.yaml` |",
        "| args (after `--config`; `{commit}` is the commit's SHA) | none |",
        "| script_kill_s (hard stop) | 600 |",
        "| Notebook | `kaggle_script.ipynb`, generated by `scripts/build_kaggle_notebook.py` |",
    ):
        assert row in manifest, row
    assert "`python scripts/kaggle_ops.py push dist/script-smoke --timeout 1800`" in manifest
    assert "| outputs | `run_logs/` into `runs/<task-id>/` |" in manifest
    assert "the wheel, the script, and its config above; the notebook checks every sha256" in (
        manifest
    )


def test_a_gpu_script_manifest_asks_for_a_timeout_past_the_hard_stop():
    config = nbb.parse_config(
        script_text({"accelerator": "NvidiaL4", "script_kill_s": 7200}), Path("c.yaml")
    )
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)
    assert meta["machine_shape"] == "NvidiaL4" and meta["code_file"] == "kaggle_script.ipynb"
    build = {"commit": "0" * 40, "config": SCRIPT_CONFIG, "config_sha256": "1" * 64,
             "wheel": WHEEL}  # fmt: skip
    manifest = bs.render_manifest(
        tag="t", build=build, commit_time=0, wheel_commit="0" * 40, wheel_epoch=0, config=config,
        builder="uv", generator="hatchling", metadata=meta, files={},
    ).decode("utf-8")  # fmt: skip
    assert "--run-id <run-id>`, with the brief's `Timeout` at least 7500 s" in manifest


@needs_builder
def test_the_script_build_passes_the_gateway_checks(builds):
    folder = builds.script.resolve()
    meta = ops.load_kernel_metadata(folder)
    assert meta["id"] == "alyahya737/arc26-script-smoke"
    assert meta["code_file"] == "kaggle_script.ipynb" and meta["kernel_type"] == "notebook"
    assert meta["enable_gpu"] is False and "machine_shape" not in meta
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert not ops.uses_gpu(meta, None) and ops.accelerator_id(meta, None) is None
    ops.check_kernel_metadata(meta, False)
    ops.check_flags(meta, None)
    ops.check_code_file(folder, meta)
    ops.check_dataset_folder(folder / "arc26-code")
    gpu = json.loads((builds.gpu / "kernel-metadata.json").read_text())
    for key in ("docker_image", "docker_image_pinning_type", "competition_sources",
                "model_sources", "kernel_sources", "dataset_sources"):  # fmt: skip
        assert meta[key] == gpu[key], key
    dataset = json.loads((folder / "arc26-code" / "dataset-metadata.json").read_text())
    assert dataset["id"] == "alyahya737/arc26-code" and dataset["licenses"] == [{"name": "other"}]
    assert "scripts/env_probe.py" in dataset["description"]
    # A submission build's dataset metadata is as before task P0.5d.
    assert "for the arc26 submission notebooks" in json.loads(
        (builds.gpu / "arc26-code" / "dataset-metadata.json").read_text()
    )["description"]


@needs_builder
def test_the_script_notebook_alternates_markdown_and_plain_code(builds):
    path = builds.script / "kaggle_script.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert "kaggle" not in notebook["metadata"]
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code"] * 6
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            compile(text, cell["id"], "exec")
            assert not [line for line in text.splitlines() if line.lstrip().startswith(("!", "%"))]
            assert cell["outputs"] == [] and cell["execution_count"] is None
    text = path.read_text(encoding="utf-8")
    assert not KAGGLE_ACCESS.search(text) and EM_DASH not in text
    assert "import torch" not in text and "make_submission" not in text
    note = "".join(notebook["cells"][0]["source"])
    assert note.startswith("# arc26 script notebook\n\n**Never submitted.**")
    assert "`python scripts/env_probe.py --config configs/script_smoke.yaml`" in note


@needs_builder
def test_the_script_config_cell_holds_the_hashes_of_the_script_and_its_config(builds):
    cells = script_code(builds.script)
    namespace = {}
    exec(cells["arc26-02-config"], namespace)
    assert namespace["ARGS"] == [] and namespace["SCRIPT_KILL_S"] == 600
    build = namespace["BUILD"]
    assert build["kind"] == "script" and build["accelerator"] is None
    assert build["script_sha256"] == sha256((builds.repo / "scripts/env_probe.py").read_bytes())
    assert build["script_config_sha256"] == build["config_sha256"] == sha256(
        (builds.repo / SCRIPT_CONFIG).read_bytes()
    )
    assert build["commit"] == git(builds.repo, "rev-parse", "HEAD").strip()


@needs_builder
def test_embedded_script_sources_are_verbatim(builds):
    cells = script_code(builds.script)
    for helper in nbb.SCRIPT_RUNTIME_HELPERS:
        assert inspect.getsource(helper).rstrip() in cells["arc26-04-helpers"]
    assert inspect.getsource(reference_run.dump_run_logs).rstrip() in cells["arc26-12-logs"]
    # The Install cell is the submission notebook's, behind the guard.
    install = cells["arc26-08-install"]
    assert install.count(INSTALL_GUARD) == 1
    assert install.replace(INSTALL_GUARD, "    if ") == code(builds.gpu)["arc26-10-install"]


def test_a_script_missing_from_the_commit_is_refused_and_nothing_is_written(tmp_path):
    repo = snapshot(tmp_path / "repo")
    config = repo / SCRIPT_CONFIG
    text = config.read_text(encoding="utf-8")
    config.write_text(text.replace("script: scripts/env_probe.py", "script: scripts/absent.py"))
    commit_all(repo)
    with pytest.raises(bs.BuildError, match="scripts/absent.py is not in commit"):
        bs.build("t", config, repo)
    assert not (repo / "dist").exists()


def test_notebook_command_line_writes_a_script_notebook(tmp_path):
    wheel = tmp_path / WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(REPO / "arc26" / "submit.py", "arc26/submit.py")
    out = tmp_path / "nb" / "kaggle_script.ipynb"
    argv = ["--config", str(REPO / SCRIPT_CONFIG), "--wheel", str(wheel), "--commit", "ab" * 20]
    assert nbb.main([*argv, "--tag", "t", "--out", str(out)]) == 0
    notebook = json.loads(out.read_text(encoding="utf-8"))
    assert notebook["cells"][0]["source"][0] == "# arc26 script notebook\n"
    config_cell = "".join(notebook["cells"][1]["source"])
    digest = sha256((REPO / "scripts" / "env_probe.py").read_bytes())
    assert f"'script_sha256': '{digest}'" in config_cell


# The bf16 check and the script runner, called directly (task P0.5d)

L4 = {"name": "NVIDIA L4", "memory.total": "23034 MiB", "compute_cap": "8.9", "driver_version": "5"}
T4 = {"name": "Tesla T4", "memory.total": "15360 MiB", "compute_cap": "7.5", "driver_version": "5"}
A100 = {"name": "NVIDIA A100", "compute_cap": "8.0"}


@pytest.mark.parametrize(
    ("gpus", "accelerator", "reason"),
    [
        ([], None, None),
        ([T4, T4], None, None),
        ([L4] * 4, "NvidiaL4", None),
        ([L4], "NvidiaL4X1", None),
        ([A100], "NvidiaTeslaA100", None),
        ([T4, T4], "NvidiaL4X1", "without bf16: Tesla T4 (7.5), Tesla T4 (7.5)."),
        ([L4, T4], "NvidiaL4", "without bf16: Tesla T4 (7.5)."),
        ([], "NvidiaL4", "the build asked for NvidiaL4, but nvidia-smi lists no GPU"),
        ([{"name": "NVIDIA L4"}], "NvidiaL4", "NVIDIA L4 (capability unknown)"),
        ([{"name": "X", "compute_cap": "[N/A]"}], "NvidiaL4", "X ([N/A])"),
    ],
)
def test_bf16_refusal(gpus, accelerator, reason):
    refusal = nbb.bf16_refusal(gpus, accelerator)
    if reason is None:
        assert refusal is None
    else:
        assert reason in refusal and refusal.startswith(f"the build asked for {accelerator}")


def test_run_streamed_copies_the_output_and_returns_the_exit_code(tmp_path, capsys):
    program = "import os, sys; print(os.getcwd(), os.environ['ARC26_X'], flush=True); sys.exit(3)"
    env = {**os.environ, "ARC26_X": "y"}
    assert nbb.run_streamed([sys.executable, "-c", program], 30, str(tmp_path), env) == (3, False)
    out = capsys.readouterr().out
    assert f"{tmp_path} y\n" in out and "exit code 3\n" in out


def test_run_streamed_reports_a_command_that_cannot_start(tmp_path):
    assert nbb.run_streamed([str(tmp_path / "absent")], 30) == (127, False)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="reads process states from /proc")
def test_run_streamed_kills_the_process_group_at_the_timeout(tmp_path, capsys):
    pid_file = tmp_path / "child.pid"
    program = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    assert nbb.run_streamed([sys.executable, "-c", program], 2.0) == (-signal.SIGKILL, True)
    assert time.monotonic() - start < 20
    assert "started\n" in capsys.readouterr().out
    child = int(pid_file.read_text())
    for _ in range(100):
        try:
            state = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state in {"Z", "X"}:
            break
        time.sleep(0.05)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("the child outlived the timeout")


def test_save_json_replaces_the_file_whole(tmp_path):
    path = tmp_path / "logs" / "stages.json"
    nbb.save_json(path, {"a": 1, "path": tmp_path})
    assert json.loads(path.read_text()) == {"a": 1, "path": str(tmp_path)}
    nbb.save_json(path, [1])
    assert json.loads(path.read_text()) == [1]
    assert [p.name for p in path.parent.iterdir()] == ["stages.json"]


@pytest.mark.parametrize(
    ("args", "override"),
    [
        ([], None),
        (["--smoke", "--commit", "0" * 40, "--configure", "-v", "--", "-"], None),
        (["--config", "x.yaml"], "--config"),
        (["--smoke", "--config=x.yaml"], "--config=x.yaml"),
        (["--conf", "x.yaml"], "--conf"),
        (["--c"], "--c"),
        (["-c", "x.yaml"], "-c"),
        (["-cx.yaml"], "-cx.yaml"),
        (["-c=x.yaml"], "-c=x.yaml"),
        (["-vc", "x.yaml"], "-vc"),
        (["-v", "-xc=x.yaml"], "-xc=x.yaml"),
        (["--commit", "c" * 40, "-v"], None),
    ],
)
def test_config_override_finds_every_way_to_set_config(args, override):
    assert nbb.config_override(args) == override


def fake_nvidia_smi(tmp_path, monkeypatch, stdout, fields=4):
    """An nvidia-smi on PATH that prints stdout for a query of ``fields`` fields and fails any
    other query, as a driver without compute_cap would for the four-field one."""
    path = tmp_path / "nvidia-smi"
    path.write_text(
        "#!/bin/sh\n"
        f'[ "$(echo "$1" | tr -cd , | wc -c)" -eq {fields - 1} ] || exit 2\n'
        f"cat <<'EOF'\n{stdout}EOF\n"
    )
    path.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(path) if name == "nvidia-smi" else None)


L4_LINE = "NVIDIA L4, 23034 MiB, 8.9, 580.159.04\n"


def test_gpu_inventory_counts_only_lines_with_every_queried_field(tmp_path, monkeypatch):
    """A warning that nvidia-smi prints on stdout is no GPU: four L4s stay four, and pass the bf16
    check of a 4xL4 build."""
    stray = "WARNING: infoROM is corrupted at gpu 0000:00:04.0\n"
    fake_nvidia_smi(tmp_path, monkeypatch, stray + L4_LINE * 4 + "\n")
    gpus = nbb.gpu_inventory()
    assert gpus == [
        {
            "name": "NVIDIA L4",
            "memory.total": "23034 MiB",
            "compute_cap": "8.9",
            "driver_version": "580.159.04",
        }
    ] * 4
    assert nbb.bf16_refusal(gpus, "NvidiaL4") is None


def test_gpu_inventory_asks_a_driver_without_compute_cap_again(tmp_path, monkeypatch):
    stray = "WARNING: a message, with commas, of four, parts\n"
    fake_nvidia_smi(tmp_path, monkeypatch, stray + "NVIDIA L4, 23034 MiB, 580.159.04\n", 3)
    gpus = nbb.gpu_inventory()
    assert gpus == [
        {"name": "NVIDIA L4", "memory.total": "23034 MiB", "driver_version": "580.159.04"}
    ]
    assert "capability unknown" in nbb.bf16_refusal(gpus, "NvidiaL4")


# Attached notebook outputs (task P3.5): a script config may pin outputs of other notebooks in
# notebook_inputs; they join kernel_sources, and "{input:<owner>/<slug>}" in args is each folder.

EPISODES, CHECK = "alyahya737/arc26-p3-4-episodes", "alyahya737/arc26-p3-4-leakage"
INPUTS = {EPISODES: 2, CHECK: 2}
INPUT_ARGS = ["--episodes-root", "{input:" + EPISODES + "}", "--check-root={input:" + CHECK + "}"]


def test_p3_5_01_config_trains_h_on_4xl4_with_the_p3_4_outputs_attached():
    gpu, config = load_config(GPU_CONFIG), load_config(P35_CONFIG)
    assert isinstance(config, nbb.ScriptConfig)
    assert config.kaggle_notebook == "alyahya737/arc26-p3-5-train"
    assert config.accelerator == "NvidiaL4" and config.script == "scripts/train_hypernet.py"
    assert config.script_config == "configs/train_hypernet_toy.yaml"
    assert config.notebook_inputs == INPUTS
    assert config.args == (
        "--commit", "{commit}", "--episodes-root", "{input:" + EPISODES + "}",
        "--check-root", "{input:" + CHECK + "}",
    )  # fmt: skip
    assert {f: getattr(config, f) for f in STACK} == {f: getattr(gpu, f) for f in STACK}
    # The hard stop leaves the brief's Timeout (script_kill_s + 300 s) inside the 12 h session,
    # after the script's own budget.
    toy = yaml.safe_load((REPO / config.script_config).read_text(encoding="utf-8"))
    assert toy["max_wall_s"] < config.script_kill_s <= 43200 - 300
    assert config.kaggle_notebook in ops.briefed_notebooks()


@pytest.mark.parametrize(
    "inputs",
    [None, {}],
    ids=["null", "empty"],
)
def test_a_script_config_without_notebook_inputs_attaches_none(inputs):
    config = nbb.parse_config(script_text({"notebook_inputs": inputs}), Path("c.yaml"))
    assert config.notebook_inputs == {} and config == load_config(SCRIPT_CONFIG)


def test_notebook_inputs_are_accepted_with_their_placeholders():
    config = nbb.parse_config(
        script_text({"notebook_inputs": INPUTS, "args": ["--commit", "{commit}", *INPUT_ARGS]}),
        Path("c.yaml"),
    )
    assert config.notebook_inputs == INPUTS and config.args[-1] == INPUT_ARGS[-1]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"notebook_inputs": [EPISODES]}, "notebook_inputs must map at most 8 notebooks"),
        ({"notebook_inputs": {EPISODES: 0}}, "to a version number >= 1"),
        ({"notebook_inputs": {EPISODES: True}}, "to a version number >= 1"),
        ({"notebook_inputs": {EPISODES: "2"}}, "to a version number >= 1"),
        ({"notebook_inputs": {"arc26-p3-4-episodes": 2}}, "notebooks <owner>/<slug>"),
        ({"notebook_inputs": {"a/B-C": 2}}, "notebooks <owner>/<slug>"),
        ({"notebook_inputs": {"a/abc": 2}}, "the slug 5 to 50"),
        ({"notebook_inputs": {f"a/out{i:02d}": 1 for i in range(9)}}, "at most 8 notebooks"),
        (
            {
                "notebook_inputs": {"sorokin/pip-install-unsloth-flash-patch": 2},
                "args": ["{input:sorokin/pip-install-unsloth-flash-patch}"],
            },
            "in both kernel_source_versions, the reference's own kernel sources, and notebook_",
        ),
        (
            {"notebook_inputs": {EPISODES: 2}, "args": ["{input:" + CHECK + "}"]},
            f"args pass {{input:...}} for ['{CHECK}'], which notebook_inputs does not attach",
        ),
        ({"args": ["{input:}"]}, "args pass {input:...} for [''], which notebook_inputs does not"),
        (
            {"notebook_inputs": INPUTS, "args": INPUT_ARGS[:2]},
            f"notebook_inputs ['{CHECK}'] never reach the script",
        ),
        ({"notebook_inputs": INPUTS, "args": INPUT_ARGS, "workers": 4}, "unknown keys"),
    ],
)
def test_notebook_input_refusals(changes, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(script_text(changes), Path("c.yaml"))
    assert message in str(refused.value)


def test_a_submission_config_cannot_attach_notebook_inputs():
    # Task P3.7: a submission config may attach them for its adapter arms only (tested in
    # tests/test_arms_build.py); without adapters it is still refused.
    with pytest.raises(nbb.BuildError, match="notebook_inputs attach other notebooks' outputs"):
        nbb.parse_config(config_text({"notebook_inputs": INPUTS}), Path("c.yaml"))


def test_notebook_inputs_join_the_kernel_sources_and_the_manifest():
    config = nbb.parse_config(
        script_text({"notebook_inputs": INPUTS, "args": INPUT_ARGS, "accelerator": "NvidiaL4"}),
        Path("c.yaml"),
    )
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)
    assert meta["kernel_sources"] == [
        "sorokin/pip-install-unsloth-flash-patch/2", f"{EPISODES}/2", f"{CHECK}/2",
    ]  # fmt: skip
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    build = {"commit": "0" * 40, "config": SCRIPT_CONFIG, "config_sha256": "1" * 64,
             "wheel": WHEEL}  # fmt: skip
    manifest = bs.render_manifest(
        tag="t", build=build, commit_time=0, wheel_commit="0" * 40, wheel_epoch=0, config=config,
        builder="uv", generator="hatchling", metadata=meta, files={},
    ).decode("utf-8")  # fmt: skip
    for line in (
        "- install notebook `sorokin/pip-install-unsloth-flash-patch/2` (the package stack, "
        "ADR-010);",
        f"- notebook output `{EPISODES}/2`, passed to the script as `{{input:{EPISODES}}}`;",
        f"- notebook output `{CHECK}/2`, passed to the script as `{{input:{CHECK}}}`;",
        "| notebook_inputs (`{input:<owner>/<slug>}` in args is its folder) | "
        f"`{EPISODES}` version 2, `{CHECK}` version 2 |",
    ):
        assert line in manifest, line
    # A submission build attaches what it attached before task P3.5.
    gpu = load_config(GPU_CONFIG)
    assert bs.kernel_metadata(gpu, pulled)["kernel_sources"] == [
        "sorokin/pip-install-unsloth-flash-patch/2"
    ]


def test_an_input_that_is_a_reference_kernel_source_is_refused():
    config = load_config(SCRIPT_CONFIG)._replace(
        kernel_source_versions={},
        notebook_inputs={"sorokin/pip-install-unsloth-flash-patch": 2},
    )
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    pulled["kernel_sources"] = ["sorokin/pip-install-unsloth-flash-patch/2"]
    with pytest.raises(bs.BuildError, match="are kernel sources of reference/kernel-metadata"):
        bs.kernel_metadata(config, pulled)


def test_build_info_names_the_notebook_inputs_only_when_there_are_any():
    common = {
        "tag": "t", "commit": "0" * 40, "config_file": P35_CONFIG, "config_sha256": "1" * 64,
        "wheel": WHEEL, "wheel_sha256": "2" * 64, "script_sha256": "3" * 64,
        "script_config_sha256": "4" * 64,
    }  # fmt: skip
    info = nbb.build_info(load_config(P35_CONFIG), **common)
    assert list(info)[-1] == "notebook_inputs" and info["notebook_inputs"] == INPUTS
    assert "notebook_inputs" not in nbb.build_info(load_config(SCRIPT_CONFIG), **common)


@pytest.mark.parametrize(
    ("args", "inputs", "filled"),
    [
        ([], {}, []),
        (["--note", "{commit}"], {}, ["--note", "c" * 40]),
        (
            ["--a", "{input:o/out-1}", "--b={input:o/out-2}/x", "{commit}{input:o/out-1}"],
            {"o/out-1": "/k/in/out-1", "o/out-2": "/k/in/{commit}"},
            ["--a", "/k/in/out-1", "--b=/k/in/{commit}/x", "c" * 40 + "/k/in/out-1"],
        ),
    ],
)
def test_fill_args_puts_the_commit_and_then_the_input_folders(args, inputs, filled):
    # The commit goes in first, so a folder holding "{commit}" stays as it is.
    assert nbb.fill_args(args, "c" * 40, inputs) == filled


def test_fill_args_refuses_an_input_it_does_not_have():
    with pytest.raises(ValueError, match=re.escape("{input:o/other} names a notebook output")):
        nbb.fill_args(["{input:o/other}"], "c" * 40, {"o/out-1": "/x"})


def test_find_input_dirs_finds_folders_by_slug_and_prefers_the_owners(tmp_path):
    for parts in (
        ("notebooks", "alyahya737", "arc26-p3-4-episodes"),
        ("datasets", "someone", "arc26-p3-4-episodes"),
        ("arc26-p3-4-leakage",),
        ("a", "b", "c", "d", "e", "f", "g", "deep-output"),
        ("x", "site-packages", "hidden-output"),
    ):
        tmp_path.joinpath(*parts).mkdir(parents=True)
    names = [EPISODES, CHECK, "o/deep-output", "o/hidden-output", "o/absent-output"]
    found = nbb.find_input_dirs([str(tmp_path)], names, 7)
    assert found == {
        EPISODES: [str(tmp_path / "notebooks" / "alyahya737" / "arc26-p3-4-episodes")],
        CHECK: [str(tmp_path / "arc26-p3-4-leakage")],
        "o/deep-output": [],
        "o/hidden-output": [],
        "o/absent-output": [],
    }
    assert nbb.find_input_dirs([str(tmp_path)], ["o/deep-output"], 8) == {
        "o/deep-output": [str(tmp_path.joinpath("a", "b", "c", "d", "e", "f", "g", "deep-output"))]
    }
    # Without a folder named by the owner, every folder of the slug is a candidate; these two are
    # both empty, so their contents are equal and they count once.
    first = tmp_path / "datasets" / "someone" / "arc26-p3-4-episodes"
    assert nbb.find_input_dirs([str(tmp_path)], ["other/arc26-p3-4-episodes"], 7) == {
        "other/arc26-p3-4-episodes": [str(first)]
    }


def test_find_input_dirs_searches_every_root_in_both_spellings_and_merges_equal_mounts(tmp_path):
    first, second = tmp_path / "input", tmp_path / "usr" / "lib" / "notebooks"
    hyphens = first / "notebooks" / "alyahya737" / "arc26-p3-4-episodes"
    underscores = second / "alyahya737" / "arc26_p3_4_episodes"
    for folder in (hyphens, underscores):
        (folder / "episodes").mkdir(parents=True)
        (folder / "episodes" / "toy.jsonl").write_text("{}\n")
        (folder / "MANIFEST.json").write_text("{}")
    roots = [str(first), str(second)]
    # The same files in both: one output mounted twice, found first under the first root.
    assert nbb.find_input_dirs(roots, [EPISODES], 8) == {EPISODES: [str(hyphens)]}
    assert nbb.find_input_dirs(roots[::-1], [EPISODES], 8) == {EPISODES: [str(underscores)]}
    # Only the underscored mount: found by that spelling.
    assert nbb.find_input_dirs(roots[1:], [EPISODES], 8) == {EPISODES: [str(underscores)]}
    # Other contents: two distinct folders, which the notebook refuses.
    (underscores / "MANIFEST.json").write_text('{"task": "another build"}')
    assert nbb.find_input_dirs(roots, [EPISODES], 8) == {EPISODES: [str(hyphens), str(underscores)]}
    # A link to a found folder is the same mount.
    link = first / "arc26-p3-4-episodes"
    link.symlink_to(hyphens, target_is_directory=True)
    assert nbb.find_input_dirs([str(first)], ["x/arc26-p3-4-episodes"], 8) == {
        "x/arc26-p3-4-episodes": [str(link)]
    }
    assert nbb.find_input_dirs([str(tmp_path / "absent")], [EPISODES], 8) == {EPISODES: []}


def test_a_script_config_cannot_attach_its_own_notebook():
    own = "alyahya737/arc26-script-smoke"
    with pytest.raises(nbb.BuildError, match="invalidKernelSources"):
        nbb.parse_config(
            script_text({"notebook_inputs": {own: 1}, "args": ["{input:" + own + "}"]}),
            Path("c.yaml"),
        )


@needs_builder
def test_the_p3_5_01_build_carries_the_script_its_config_and_the_inputs(builds):
    files = read_files(builds.p35)
    assert set(files) == {
        "MANIFEST.md", "kaggle_script.ipynb", "kernel-metadata.json", f"arc26-code/{WHEEL}",
        "arc26-code/dataset-metadata.json", "arc26-code/train_hypernet.py",
        "arc26-code/train_hypernet_toy.yaml",
    }  # fmt: skip
    for name, source in (
        ("train_hypernet.py", "scripts/train_hypernet.py"),
        ("train_hypernet_toy.yaml", "configs/train_hypernet_toy.yaml"),
    ):
        assert files[f"arc26-code/{name}"] == (builds.repo / source).read_bytes()
    folder = builds.p35.resolve()
    meta = ops.load_kernel_metadata(folder)
    assert meta["id"] == "alyahya737/arc26-p3-5-train" and meta["machine_shape"] == "NvidiaL4"
    assert meta["kernel_sources"][1:] == [f"{EPISODES}/2", f"{CHECK}/2"]
    assert ops.uses_gpu(meta, None) and ops.accelerator_id(meta, None) == "NvidiaL4"
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, "NvidiaL4")
    ops.check_code_file(folder, meta)
    ops.check_dataset_folder(folder / "arc26-code")
    cells = script_code(builds.p35)
    namespace = {}
    exec(cells["arc26-02-config"], namespace)
    assert namespace["BUILD"]["notebook_inputs"] == INPUTS
    assert namespace["SCRIPT_KILL_S"] == 15600
    manifest = (builds.p35 / "MANIFEST.md").read_text(encoding="utf-8")
    assert "with the brief's `Timeout` at least 15900 s" in manifest


# Run P3.5-02 (task P3.5): P3.5-01's build with half the steps; one code dataset version holds its
# files beside main's wheel together with P3.2-02's.

P3502_BRIEF = REPO / "docs" / "prerun" / "P3.5-02.md"


def test_p3_5_02_config_is_p3_5_01s_with_its_own_toy_config_and_hard_stop():
    first, second = load_config(P35_CONFIG), load_config(P3502_CONFIG)
    assert second == first._replace(
        script_config="configs/train_hypernet_toy_P3.5-02.yaml", script_kill_s=8400
    )
    # The script's own stops come first: the projected-wall kill, the session budget, then the
    # hard stop 600 s later, as in P3.5-01; the brief's Timeout adds the logs' 300 s.
    toy = yaml.safe_load((REPO / second.script_config).read_text(encoding="utf-8"))
    assert toy["kill"]["max_projected_s"] < toy["max_wall_s"] == second.script_kill_s - 600
    assert second.script_kill_s <= nbb.SESSION_LIMIT_S - nbb.WRITE_RESERVE_S
    header = ops.brief_header(P3502_BRIEF.read_text(encoding="utf-8"))
    assert (header["Run-id"], header["Previous run"]) == ("P3.5-02", "P3.5-01")
    assert header["Kaggle notebook"] == second.kaggle_notebook
    assert header["Accelerator"] == second.accelerator
    assert header["Config"] == second.script_config
    assert int(header["Timeout"]) == second.script_kill_s + nbb.WRITE_RESERVE_S
    assert second.kaggle_notebook in ops.briefed_notebooks()


@needs_builder
def test_the_p3_5_02_build_differs_from_p3_5_01s_only_in_the_toy_config_and_hard_stop(builds):
    """One commit, one wheel: the P3.5-02 build carries P3.5-01's script, wheel, dataset metadata,
    and kernel metadata (notebook, accelerator, attached outputs), its own toy config under its
    own name, and the 8400 s hard stop in the Config cell and the manifest."""
    files, first = read_files(builds.p3502), read_files(builds.p35)
    config = "arc26-code/train_hypernet_toy_P3.5-02.yaml"
    assert set(files) == set(first) - {"arc26-code/train_hypernet_toy.yaml"} | {config}
    assert files[config] == (builds.repo / "configs/train_hypernet_toy_P3.5-02.yaml").read_bytes()
    for name in (
        f"arc26-code/{WHEEL}", "arc26-code/dataset-metadata.json", "arc26-code/train_hypernet.py",
        "kernel-metadata.json",
    ):  # fmt: skip
        assert files[name] == first[name], name
    folder = builds.p3502.resolve()
    meta = ops.load_kernel_metadata(folder)
    assert meta["id"] == "alyahya737/arc26-p3-5-train" and meta["machine_shape"] == "NvidiaL4"
    assert meta["kernel_sources"][1:] == [f"{EPISODES}/2", f"{CHECK}/2"]
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, "NvidiaL4")
    ops.check_code_file(folder, meta)
    ops.check_dataset_folder(folder / "arc26-code")
    namespace = {}
    exec(script_code(builds.p3502)["arc26-02-config"], namespace)
    assert namespace["SCRIPT_KILL_S"] == 8400
    assert namespace["BUILD"]["script_config"] == "configs/train_hypernet_toy_P3.5-02.yaml"
    assert namespace["BUILD"]["notebook_inputs"] == INPUTS
    manifest = (builds.p3502 / "MANIFEST.md").read_text(encoding="utf-8")
    assert "with the brief's `Timeout` at least 8700 s" in manifest
