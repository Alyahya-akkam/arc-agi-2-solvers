"""Simulated-Kaggle runs of a shot build with the weights' warm-up (task P6.9c).

The notebook is ``tests/test_shot_notebook.py``'s real-shot build (``python -m
arc26.agent.shot`` with P6.4's fake engine as the venv's ``vllm``, at the simulation's times) with
``agent.warmup`` set. Its code cells run on the simulated Kaggle tree of
``tests/test_hybrid_notebook.py``, the weights output mounted as Kaggle mounts it. Against the
task brief (``docs/tasks/P6.9c.md``) this file checks, in both modes: the agent check starts the
copy before NVARC; the engine's command line (``run.json``'s ``engine.command``) names the local
copy when it is done and verified, and the input mount with ``--safetensors-load-strategy
prefetch`` when the copy fails its sha256 check; the shot runs and merges either way, NVARC's
attempts kept; a development run keeps ``warmup.json`` in ``run_logs/agent/``; a competition rerun
writes only ``submission.json``.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


SHOTN = load("_warmup_shot_notebook", REPO / "tests" / "test_shot_notebook.py")
HYBN, nbb = SHOTN.HYBN, SHOTN.nbb
DEV4, HIDDEN, GRID = SHOTN.DEV4, SHOTN.HIDDEN, SHOTN.GRID
# The simulation's warm-up: at once, small pieces, quick status writes and waits.
SIM_WARMUP = {
    "delay_s": 0, "workers": 2, "piece_mib": 1, "nice": 0, "progress_s": 1, "reserve_gb": 0,
    "wait_s": 60, "stall_s": 30, "fallback_load_strategy": "prefetch",
}  # fmt: skip
FOLDER = "qwen38-27b-int4"


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    pytest.importorskip("numpy")
    return HYBN.LOCAL.make_wheel(tmp_path_factory.mktemp("build"))


@pytest.fixture(scope="module")
def notebook(tmp_path_factory, wheel):
    """The real-shot build of tests/test_shot_notebook.py (real_shot_config's run config at the
    simulation's times) with the simulation's warm-up."""
    pytest.importorskip("arc26.agent.shot")
    import yaml

    server = pytest.importorskip("arc26.agent.server")
    run_config = yaml.safe_load((REPO / "configs/agent_shot.yaml").read_text(encoding="utf-8"))
    run_config["deadline"].update(final_margin_s=1)
    run_config["waves"].update(
        max_cap=4096, min_cap=256, rate_tokens_per_s=1e6, prefill_tokens_per_s=1e9,
        margin_s=0, setup_s=0,
    )  # fmt: skip
    run_config["server"].update(
        port=server.free_port(), boot_timeout_s=30, stop_timeout_s=5, health_interval_s=0.05,
        monitor_interval_s=0.1, hang_s=10,
    )  # fmt: skip
    run_config["engine"].update(max_retries=0)
    run_config["engine_prep"].update(enabled=False, preflight=False)
    config = SHOTN.shot_config(
        config="configs/agent_shot.yaml", config_data=run_config, warmup=SIM_WARMUP
    )
    return HYBN.build(tmp_path_factory.mktemp("warm"), wheel, "warm", config)


def replies_file(tmp_path, competition):
    replies = tmp_path / "replies.json"
    thinking = {"reasoning": "the grid stays as it is " * 20}
    replies.write_text(json.dumps({
        task: [{"text": "".join(SHOTN.answer(GRID, i) for i in range(n)), **thinking}]
        for task, n in (HIDDEN if competition else DEV4).items()
    }))  # fmt: skip
    return replies


def run_notebook(tmp_path, notebook, wheel, competition, **mount):
    root = HYBN.make_tree(tmp_path, wheel, hidden=competition, **mount)
    run = SHOTN.simulate(
        tmp_path, notebook, root, competition, agent=None,
        env={"ARC26_FAKE_REPLIES": str(replies_file(tmp_path, competition))},
        overrides={"end_in": 40},
    )  # fmt: skip
    assert run.record["error"] is None, run.output
    return run


def engine_command(run):
    return json.loads((run.scratch / "logs" / "run.json").read_text())["engine"]["command"]


def assert_shot_merged(run, competition):
    stage = HYBN.agent_stage(run)
    assert (stage["agent_exit"], stage["ended"]) == (0, "exited"), run.output
    counts = HIDDEN if competition else DEV4
    written = HYBN.assert_valid(run, counts)
    HYBN.assert_nvarc_kept(run, counts)
    open_tasks = ["f0000003"] if competition else ["981571dc", "aa4ec2a5"]
    assert all(HYBN.solved(written, task) for task in open_tasks)
    return stage


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_engine_loads_the_verified_local_copy(tmp_path, notebook, wheel, competition):
    run = run_notebook(tmp_path, notebook, wheel, competition)
    check = run.record["STAGES"]["agent_check"]
    assert check["ready"] and check["warmup"]["started"], check
    local = run.scratch / "weights" / FOLDER
    assert check["warmup"]["target"] == str(local)
    stage = assert_shot_merged(run, competition)
    weights = stage["weights"]
    assert (weights["source"], weights["model_dir"]) == ("local", str(local)), weights
    assert weights["load_strategy"] is None and weights["warmup"]["exit"] == 0
    command = engine_command(run)
    assert command[1:3] == ["serve", str(local)]
    assert "--safetensors-load-strategy" not in command
    assert (local / "model.safetensors").read_bytes() == b"\0" * 64
    assert stage["warmup"]["status"]["ok"] is True
    if competition:
        assert HYBN.working_files(run) == ["submission.json"]
    else:
        pulled = run.root / "working" / "run_logs" / "agent" / "warmup.json"
        status = json.loads(pulled.read_text())
        assert (status["state"], status["ok"], status["files"]) == ("done", True, 2)
        assert sorted(status["size_only"]) == ["config.json", "model.safetensors"]


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_a_copy_that_fails_its_check_falls_back_to_the_input_with_prefetch(
    tmp_path, notebook, wheel, competition
):
    rows = [
        {"name": "config.json", "size": 2, "sha256": "0" * 64},
        {"name": "model.safetensors", "size": 64},
    ]
    run = run_notebook(tmp_path, notebook, wheel, competition, model={"files": rows})
    stage = assert_shot_merged(run, competition)
    weights = stage["weights"]
    assert weights["source"] == "input" and "sha256" in weights["reason"], weights
    assert weights["model_dir"].endswith(f"arc26-dl-qwen38-int4/{FOLDER}")
    command = engine_command(run)
    assert command[2] == weights["model_dir"]
    at = command.index("--safetensors-load-strategy")
    assert command[at + 1] == "prefetch"
    served = json.loads((run.scratch / "agent_config.json").read_text())
    assert served["server"]["extra_args"][-2:] == ["--safetensors-load-strategy", "prefetch"]
    if competition:
        assert HYBN.working_files(run) == ["submission.json"]
