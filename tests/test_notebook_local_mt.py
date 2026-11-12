"""Task MT: the script notebooks of runs MT-01 to MT-04 in the simulated Kaggle tree.

Each build is made from its config as committed (``configs/script_MT-0*.yaml``) by
scripts/build_kaggle_notebook.py and run by tests/test_notebook_local.py's driver against a
simulated ``/kaggle`` tree: the wheel, the script, and its config in the code dataset
``alyahya737/arc26-code-mt``, P3.4's two outputs mounted as the episode fixture of
tests/testdata/train_hypernet/, four L4s reported. The environment check must find every input
once and pass the build's arguments with the mounts filled in; with ``--smoke`` added (the CPU,
the stand-in base), the scripts run for real: MT-01's cache script in episode mode with two
encoding processes; MT-02's training reading every $\\bar z$ from that cache (none encoded), its
session cap ``--max-wall-s`` reaching the script; MT-03's training resumed from MT-02's capped
checkpoints; and MT-04's resumed from MT-03's, after a first session stopped by its cap. The
configs' values are checked against their briefs' headers as the gateway reads them. Needs torch,
transformers, and safetensors.
"""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
import test_notebook_local as tnl
import yaml

REPO = Path(__file__).resolve().parent.parent
nbb = tnl.nbb
OWNER = "alyahya737"
EPISODES, CHECK = tnl.EPISODES, tnl.CHECK
CACHE_OUTPUT, TRAIN_OUTPUT = f"{OWNER}/arc26-mt-cache", f"{OWNER}/arc26-mt-train-1"
SECOND_OUTPUT = f"{OWNER}/arc26-mt-train-2"
CODE_DATASET = ("datasets", OWNER, "arc26-code-mt")
CACHE_SCRIPT = REPO / "scripts" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO / "scripts" / "train_hypernet.py"
CACHE_CONFIG = REPO / "configs" / "cache_embeddings_MT-01.yaml"
TRAIN_CONFIG = REPO / "configs" / "train_hypernet_MT.yaml"
RUNS = {
    "MT-01": ("script_MT-01.yaml", CACHE_SCRIPT, CACHE_CONFIG, "P3.6-01"),
    "MT-02": ("script_MT-02.yaml", TRAIN_SCRIPT, TRAIN_CONFIG, "MT-01"),
    "MT-03": ("script_MT-03.yaml", TRAIN_SCRIPT, TRAIN_CONFIG, "MT-02"),
    "MT-04": ("script_MT-04.yaml", TRAIN_SCRIPT, TRAIN_CONFIG, "MT-03"),
}
# The script starts this many seconds after the notebook (MT-01's environment check and install,
# measured), and needs at most this many after its budget for the stop: one step past the mean
# (150 s at worst, the step kill), the last checkpoint, and its logs, with room to spare.
SCRIPT_START_S, STOP_S = 213, 450
# MT-03 revision 3 (amendment of 17 Oct): 45 minutes for the time Kaggle may spend between RUNNING
# and the notebook's first cell, taken off its budget and its hard stop, and kept in its Timeout,
# which Kaggle may count from RUNNING.
PRE_SCRIPT_ALLOWANCE_S = {"MT-03": 2700}
STACK = ("torch", "transformers", "safetensors")
needs_stack = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in STACK),
    reason="the MT scripts need torch, transformers, and safetensors",
)


def parsed(build):
    path = REPO / "configs" / build
    return nbb.parse_config(path.read_bytes(), path)


def header(run):
    return nbb.gateway.brief_header((REPO / "docs" / "prerun" / f"{run}.md").read_text())


def cap(config):
    """A build's session budget on the script's command line (``--max-wall-s``), or None."""
    args = list(config.args)
    return float(args[args.index("--max-wall-s") + 1]) if "--max-wall-s" in args else None


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the configs against their briefs ------------------------------------------------------------


@pytest.mark.parametrize("run", sorted(RUNS))
def test_each_mt_build_matches_its_brief(run):
    build, script, script_config, previous = RUNS[run]
    config = parsed(build)
    assert config.kind == "script" and config.accelerator == "NvidiaL4"
    assert config.code_dataset == f"{OWNER}/arc26-code-mt"
    assert config.script == f"scripts/{script.name}"
    assert config.script_config == f"configs/{script_config.name}"
    head = header(run)
    assert head["Run-id"] == run and head["Previous run"] == previous
    assert head["Kaggle notebook"] == config.kaggle_notebook
    assert head["Accelerator"] == config.accelerator
    assert head["Config"] == config.script_config
    allowance = PRE_SCRIPT_ALLOWANCE_S.get(run, 0)
    assert int(head["Timeout"]) == allowance + config.script_kill_s + nbb.WRITE_RESERVE_S
    assert int(head["Timeout"]) <= nbb.gateway.MAX_TIMEOUT_S
    assert config.kaggle_notebook in nbb.gateway.briefed_notebooks()
    # The script's own stop comes first: it starts 213 s into the session and needs time after
    # its budget (the build's --max-wall-s, else the config's) for the last checkpoint or check
    # and its logs.
    settings = yaml.safe_load(script_config.read_text(encoding="utf-8"))
    budget = cap(config) or settings["max_wall_s"]
    assert budget + SCRIPT_START_S + STOP_S <= config.script_kill_s


def test_mt_02s_worst_case_fits_the_weeks_quota_at_twice_the_t4_rate():
    """Revision 2: the quota page showed 7.87 h left on 4 Oct, lost at the reset; the session's
    worst case, its Timeout, costs at most 7.6 quota h even if 4xL4 costs twice the T4 rate, and
    the cap is below the config's full budget, which MT-03 and MT-04 keep."""
    config = parsed(RUNS["MT-02"][0])
    head = header("MT-02")
    assert head["Revision"] == "2"
    assert 2 * int(head["Timeout"]) / 3600 <= 7.6
    settings = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert cap(config) == 12600 < settings["max_wall_s"]
    assert cap(parsed(RUNS["MT-04"][0])) is None


def test_mt_03_leaves_the_pre_script_allowance_and_keeps_revision_2s_margins():
    """Revision 3: the budget and the hard stop are revision 2's (the config's 41,400 s and
    42,300 s) less the 45-minute allowance, so the margin between them is revision 2's, and the
    session ends under Kaggle's 12 h even if the 45 minutes before the first cell count."""
    config, head = parsed(RUNS["MT-03"][0]), header("MT-03")
    assert head["Revision"] == "3"
    allowance = PRE_SCRIPT_ALLOWANCE_S["MT-03"]
    settings = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert cap(config) == settings["max_wall_s"] - allowance == 38700
    assert config.script_kill_s == 42300 - allowance
    assert allowance + config.script_kill_s + nbb.WRITE_RESERVE_S == int(head["Timeout"]) <= 43200


def test_the_mt_builds_chain_their_outputs():
    cache, first, second, third = (
        parsed(RUNS[run][0]) for run in ("MT-01", "MT-02", "MT-03", "MT-04")
    )
    assert cache.notebook_inputs == {EPISODES: 2, CHECK: 2}
    assert first.notebook_inputs == {EPISODES: 2, CHECK: 2, CACHE_OUTPUT: 1}
    assert cache.kaggle_notebook == CACHE_OUTPUT and first.kaggle_notebook == TRAIN_OUTPUT
    assert "--embedding-cache" in first.args
    assert first.args[first.args.index("--embedding-cache") + 1] == (
        "{input:" + CACHE_OUTPUT + "}/embedding_cache"
    )
    # MT-03 is MT-02's build with its session cap at MT-03's budget (revision 3), MT-02's output
    # attached, and --resume, nothing else (the hard stop aside); MT-04 (revision 1) is MT-02's
    # without the cap, with MT-03's output attached and resumed.
    assert first.args[-2:] == ("--max-wall-s", "12600")
    resume = ("--resume", "{input:" + TRAIN_OUTPUT + "}/checkpoints")
    assert second == first._replace(
        kaggle_notebook=SECOND_OUTPUT,
        notebook_inputs={**first.notebook_inputs, TRAIN_OUTPUT: 1},
        args=(*first.args[:-1], "38700", *resume),
        script_kill_s=second.script_kill_s,
    )
    assert first.script_kill_s < second.script_kill_s == 39600
    inputs = {k: v for k, v in second.notebook_inputs.items() if k != TRAIN_OUTPUT}
    assert third == second._replace(
        kaggle_notebook=f"{OWNER}/arc26-mt-train-3",
        notebook_inputs={**inputs, SECOND_OUTPUT: 1},
        args=(*first.args[:-2], "--resume", "{input:" + SECOND_OUTPUT + "}/checkpoints"),
        script_kill_s=42300,
    )
    for config in (cache, first, second, third):
        assert config.args[:2] == ("--commit", "{commit}")
        for name in ("competition", "reference_metadata", "kernel_source_versions", "model_slug"):
            assert getattr(config, name) == getattr(parsed("script_P3.5-02.yaml"), name), name


@pytest.mark.parametrize("run", sorted(RUNS))
def test_each_mt_build_config_has_a_comment_above_every_key(run):
    lines = (REPO / "configs" / RUNS[run][0]).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"no comment above {line!r}"


# --- the builds in the simulated tree ------------------------------------------------------------


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    return tnl.make_wheel(tmp_path_factory.mktemp("wheel"))


def notebook(dest, wheel, run, extra_args):
    """The run's script notebook from its build config as committed, ``extra_args`` after its
    args."""
    build, script, script_config, _ = RUNS[run]
    path = REPO / "configs" / build
    config = parsed(build)._replace(args=(*parsed(build).args, *extra_args))
    info = nbb.build_info(
        config,
        tag="local-test",
        commit=tnl.COMMIT,
        config_file=f"configs/{build}",
        config_sha256=tnl.sha256(path),
        wheel=wheel.name,
        wheel_sha256=tnl.sha256(wheel),
        script_sha256=tnl.sha256(script),
        script_config_sha256=tnl.sha256(script_config),
    )
    out = dest / f"{run}.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_script_notebook(config, info)))
    return out


# The one code dataset version all three runs share holds both builds' files beside the wheel
# (MT-01's brief, launch condition): each notebook must pick its own script and config by name.
MERGED = (CACHE_SCRIPT, CACHE_CONFIG, TRAIN_SCRIPT, TRAIN_CONFIG)


def tree(tmp_path, wheel, run):
    """The simulated tree, the merged code dataset mounted as alyahya737/arc26-code-mt, P3.4's
    outputs mounted where P3.5-01's session found them."""
    assert RUNS[run][1] in MERGED and RUNS[run][2] in MERGED
    extras = {path.name: path.read_bytes() for path in MERGED}
    root = tnl.make_tree(tmp_path, wheel, extras=extras)
    dataset = root.joinpath("input", *CODE_DATASET)
    shutil.move(root.joinpath("input", *tnl.DATASET_DIRS["2026"]), dataset)
    episodes, check = tnl.mount_episode_fixture(root)
    return root, episodes, check


def output_mount(root, name):
    return root.joinpath("input", "notebooks", *name.split("/"))


def assert_ran(run, argv, dataset, script, script_config):
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [], run.output
    stages = tnl.stages_json(run)
    assert stages["ok"] is True and stages["script_exit"] == 0, run.output
    (call,) = tnl.script_calls(run)
    assert call["argv"] == [
        str(dataset / script.name), "--config", str(dataset / script_config.name), *argv,
    ]  # fmt: skip
    assert run.record["imports"] == []  # only the script's processes imported model code


def saved_budget(path):
    """The session budget a full checkpoint was written under (its saved config)."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)["config"]["max_wall_s"]


def mt01_cache(folder):
    """MT-01's smoke cache, written where a later notebook finds MT-01's output."""
    script = load_script("cache_embeddings_mt", CACHE_SCRIPT)
    args = ["--config", str(CACHE_CONFIG), "--smoke", "--out-dir", str(folder / "run_logs")]
    assert script.main([*args, "--cache-dir", str(folder / "embedding_cache")]) == 0
    return folder


@needs_stack
def test_the_mt_01_build_caches_the_fixtures_pairs_in_the_simulated_tree(wheel, tmp_path):
    root, episodes, check = tree(tmp_path, wheel, "MT-01")
    working = root / "working"
    extra = (
        "--smoke", "--out-dir", str(working / "run_logs"),
        "--cache-dir", str(working / "embedding_cache"),
    )  # fmt: skip
    run = tnl.simulate(tmp_path, notebook(tmp_path, wheel, "MT-01", extra), root, gpus=4,
                       real_transformers=True)  # fmt: skip
    dataset = root.joinpath("input", *CODE_DATASET)
    argv = [
        "--commit", tnl.COMMIT, "--episodes-root", str(episodes), "--check-root", str(check),
        *extra,
    ]  # fmt: skip
    assert_ran(run, argv, dataset, CACHE_SCRIPT, CACHE_CONFIG)
    assert run.record["INPUT_DIRS"] == {EPISODES: str(episodes), CHECK: str(check)}
    logs = tnl.run_logs(run)
    expected = ["config.yaml", "metrics.json", "run.json", "summary.md"]
    expected += [
        "encode-rank0.json",
        "encode-rank1.json",
        "timing-rank0.jsonl",
        "timing-rank1.jsonl",
    ]
    assert sorted(logs) == sorted([*tnl.NOTEBOOK_LOGS, *expected])
    metrics = json.loads(logs["metrics.json"].read_text())
    assert metrics["status"] == "complete" and metrics["pairs"]["distinct"] == 21
    assert metrics["pairs"]["missing"] == 0 and metrics["check"]["pass"] is True
    assert metrics["episodes"]["episodes"] == str(episodes)
    assert sorted(p.name for p in working.iterdir()) == ["embedding_cache", "run_logs"]


@needs_stack
def test_the_mt_02_build_trains_on_mt_01s_cache_in_the_simulated_tree(wheel, tmp_path):
    root, episodes, check = tree(tmp_path, wheel, "MT-02")
    cache = mt01_cache(output_mount(root, CACHE_OUTPUT))
    working = root / "working"
    extra = (
        "--smoke", "--out-dir", str(working / "run_logs"),
        "--ckpt-dir", str(working / "checkpoints"),
    )  # fmt: skip
    run = tnl.simulate(tmp_path, notebook(tmp_path, wheel, "MT-02", extra), root, gpus=4,
                       real_transformers=True)  # fmt: skip
    dataset = root.joinpath("input", *CODE_DATASET)
    argv = [
        "--commit", tnl.COMMIT, "--episodes-root", str(episodes), "--check-root", str(check),
        "--embedding-cache", str(cache / "embedding_cache"), "--max-wall-s", "12600", *extra,
    ]  # fmt: skip
    assert_ran(run, argv, dataset, TRAIN_SCRIPT, TRAIN_CONFIG)
    logs = tnl.run_logs(run)
    # The session cap reaches the script, over the smoke section's 600 s, and nothing else moves.
    resolved = yaml.safe_load(logs["config.yaml"].read_text())
    assert resolved["max_wall_s"] == 12600.0
    assert tnl.stages_json(run)["settings"]["script_kill_s"] == 13300
    info = json.loads(logs["run.json"].read_text())
    assert info["argv"][-len(extra) - 2 : -len(extra)] == ["--max-wall-s", "12600"]
    # Every z_bar comes from MT-01's cache: the namespaces agree and nothing is encoded.
    assert info["embeddings"]["cache_hits"] == 21 and info["embeddings"]["encoded"] == 0
    assert info["order"]["per_block"] == {"arc2train": 200, "barc": 650, "rearc": 150}
    status = json.loads(logs["status.json"].read_text())
    assert (status["status"], status["step"]) == ("complete", 12)
    assert status["best"]["file"] == str(working / "checkpoints" / "best.pt")
    assert (working / "checkpoints" / "best.pt").is_file()


@needs_stack
def test_the_mt_03_build_resumes_mt_02s_checkpoints_in_the_simulated_tree(wheel, tmp_path):
    root, episodes, check = tree(tmp_path, wheel, "MT-03")
    cache = mt01_cache(output_mount(root, CACHE_OUTPUT))
    trained = output_mount(root, TRAIN_OUTPUT)
    train = load_script("train_hypernet_mt", TRAIN_SCRIPT)
    # MT-02's session under its build's own cap, so its checkpoints carry max_wall_s 12,600.
    first = [
        "--config", str(TRAIN_CONFIG), "--smoke", "--out-dir", str(trained / "run_logs"),
        "--ckpt-dir", str(trained / "checkpoints"), "--embedding-cache",
        str(cache / "embedding_cache"), "--max-wall-s", str(int(cap(parsed(RUNS["MT-02"][0])))),
    ]  # fmt: skip
    assert train.main(first) == 0
    assert saved_budget(trained / "checkpoints" / "ckpt-000008.pt") == 12600.0
    (trained / "checkpoints" / "ckpt-000012.pt").unlink()  # MT-02 stopped at its budget
    (trained / "checkpoints" / "latest.json").write_text(
        json.dumps({"file": "ckpt-000008.pt", "step": 8}) + "\n"
    )
    working = root / "working"
    extra = (
        "--smoke", "--out-dir", str(working / "run_logs"),
        "--ckpt-dir", str(working / "checkpoints"),
    )  # fmt: skip
    run = tnl.simulate(tmp_path, notebook(tmp_path, wheel, "MT-03", extra), root, gpus=4,
                       real_transformers=True)  # fmt: skip
    dataset = root.joinpath("input", *CODE_DATASET)
    argv = [
        "--commit", tnl.COMMIT, "--episodes-root", str(episodes), "--check-root", str(check),
        "--embedding-cache", str(cache / "embedding_cache"), "--max-wall-s", "38700",
        "--resume", str(trained / "checkpoints"), *extra,
    ]  # fmt: skip
    assert_ran(run, argv, dataset, TRAIN_SCRIPT, TRAIN_CONFIG)
    logs = tnl.run_logs(run)
    info = json.loads(logs["run.json"].read_text())
    assert info["start_step"] == 8 and info["resumed_from"].endswith("ckpt-000008.pt")
    # Resumed under the build's budget (revision 3's 38,700 s), not MT-02's cap: the budget is not
    # a resume setting, so nothing is refused.
    assert yaml.safe_load(logs["config.yaml"].read_text())["max_wall_s"] == 38700
    # The environment check logged the clocks that may see the time before the first cell.
    clocks = tnl.stages_json(run)["stages"]["environment"]["clocks"]
    assert {"time", "since_start_s", "uptime_s", "pid1_age_s", "kernel_age_s"} <= set(clocks)
    assert "errors" not in clocks and 0 <= clocks["since_start_s"] < 60
    status = json.loads(logs["status.json"].read_text())
    assert (status["status"], status["step"]) == ("complete", 12)
    val = [json.loads(x) for x in logs["metrics.jsonl"].read_text().splitlines()]
    assert [r["step"] for r in val if r["kind"] == "val"] == [0, 4, 8, 12]
    assert [r["session_start"] for r in val if r["kind"] == "val"] == [0, 0, 0, 8]


@needs_stack
def test_the_mt_04_build_finishes_a_run_whose_first_session_stopped_at_its_cap(wheel, tmp_path):
    """Three sessions: MT-02 stopped by its command-line cap (here below its setup, so at step 0,
    exit 3, with a checkpoint), MT-03 resumed (here without a cap) and stopped at step 8 (its later
    checkpoint removed), and MT-04's build resuming MT-03's output to the end."""
    root, episodes, check = tree(tmp_path, wheel, "MT-04")
    cache = mt01_cache(output_mount(root, CACHE_OUTPUT))
    first, second = output_mount(root, TRAIN_OUTPUT), output_mount(root, SECOND_OUTPUT)
    train = load_script("train_hypernet_mt", TRAIN_SCRIPT)

    def session(folder, *extra):
        args = [
            "--config", str(TRAIN_CONFIG), "--smoke", "--out-dir", str(folder / "run_logs"),
            "--ckpt-dir", str(folder / "checkpoints"), "--embedding-cache",
            str(cache / "embedding_cache"),
        ]  # fmt: skip
        return train.main([*args, *extra])

    assert session(first, "--max-wall-s", "1e-6") == 3
    status = json.loads((first / "run_logs" / "status.json").read_text())
    assert (status["status"], status["step"]) == ("stopped", 0)
    assert saved_budget(first / "checkpoints" / "ckpt-000000.pt") == 1e-6
    assert session(second, "--resume", str(first / "checkpoints")) == 0
    (second / "checkpoints" / "ckpt-000012.pt").unlink()  # MT-03 stopped at its budget
    (second / "checkpoints" / "latest.json").write_text(
        json.dumps({"file": "ckpt-000008.pt", "step": 8}) + "\n"
    )
    shutil.rmtree(first)  # MT-04 attaches MT-03's output only
    working = root / "working"
    extra = (
        "--smoke", "--out-dir", str(working / "run_logs"),
        "--ckpt-dir", str(working / "checkpoints"),
    )  # fmt: skip
    run = tnl.simulate(tmp_path, notebook(tmp_path, wheel, "MT-04", extra), root, gpus=4,
                       real_transformers=True)  # fmt: skip
    dataset = root.joinpath("input", *CODE_DATASET)
    argv = [
        "--commit", tnl.COMMIT, "--episodes-root", str(episodes), "--check-root", str(check),
        "--embedding-cache", str(cache / "embedding_cache"),
        "--resume", str(second / "checkpoints"), *extra,
    ]  # fmt: skip
    assert_ran(run, argv, dataset, TRAIN_SCRIPT, TRAIN_CONFIG)
    assert run.record["INPUT_DIRS"] == {
        EPISODES: str(episodes), CHECK: str(check), CACHE_OUTPUT: str(cache),
        SECOND_OUTPUT: str(second),
    }  # fmt: skip
    logs = tnl.run_logs(run)
    info = json.loads(logs["run.json"].read_text())
    assert info["start_step"] == 8 and info["resumed_from"].endswith("ckpt-000008.pt")
    status = json.loads(logs["status.json"].read_text())
    assert (status["status"], status["step"]) == ("complete", 12)
    val = [json.loads(x) for x in logs["metrics.jsonl"].read_text().splitlines()]
    assert [r["step"] for r in val if r["kind"] == "val"] == [0, 4, 8, 12]
    assert [r["session_start"] for r in val if r["kind"] == "val"] == [0, 0, 0, 8]
