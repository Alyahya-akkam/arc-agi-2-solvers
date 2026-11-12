"""Simulated-Kaggle runs of the shot build (task P6.5b): NVARC unchanged, then the long shot.

The notebook is the hybrid's (``configs/submission_P6.5-hybrid.yaml``) with ``entry:
arc26.agent.shot`` and NVARC's share at the competition's soft deadline, at the simulation's
times (``tests/test_hybrid_notebook.py``'s: the shot's end 3600 s, a final margin of 1 s, SIGKILL
3 s after SIGTERM). Its code cells run on the simulated Kaggle tree of that file, whose
``python`` shim here also runs ``tests/testdata/fake_agent_run.py`` for ``python -m
arc26.agent.shot``: the shot takes exactly ``arc26.agent.run``'s command line, so P6.4's stand-in
serves until the shot module is merged; the real module runs in the last test once it is.

Against the task brief's acceptance this file checks, in both modes: NVARC keeps its deadlines
and writes first; the shot runs as ``arc26.agent.shot``, with the evaluation solutions in a
development run only; the shot fails at the install, at the boot, mid-run, and at the deadline,
each leaving a valid file with NVARC's attempts; NVARC ending too late for ``min_stage_s`` runs no
shot and leaves NVARC's file; a competition rerun opens no solutions file and writes only
``submission.json``; DEV240 runs every stage on the 240-task files in development mode only.
"""

import contextlib
import importlib.util
import json
import os
import signal
import sys
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


HYBN = load("_shot_hybrid_notebook", REPO / "tests" / "test_hybrid_notebook.py")
LOCAL, nbb = HYBN.LOCAL, HYBN.nbb
SHOT = "arc26.agent.shot"
# The shim of test_hybrid_notebook.py, with python -m arc26.agent.shot run by the fake agent too.
OLD_FAKES = '{"arc26.solver": "ARC26_FAKE_SOLVER", "arc26.agent.run": "ARC26_FAKE_AGENT"}'
NEW_FAKES = (
    '{"arc26.solver": "ARC26_FAKE_SOLVER", "arc26.agent.run": "ARC26_FAKE_AGENT", '
    '"arc26.agent.shot": "ARC26_FAKE_AGENT"}'
)
assert HYBN.SHIM.count(OLD_FAKES) == 1
HYBN.SHIM = HYBN.SHIM.replace(OLD_FAKES, NEW_FAKES)
DEV4, HIDDEN, GRID = HYBN.DEV4, HYBN.HIDDEN, HYBN.GRID
AGENT_OUTPUT = HYBN.AGENT_OUTPUT
SHOT_STAGES = HYBN.HYBRID_STAGES
# The DEV240 copies of the stand-in DEV4 tasks.
COPIES = {nbb.hashlib.sha256(task.encode()).hexdigest()[:8]: task for task in DEV4}
DEV8 = {**DEV4, **{copy: DEV4[task] for copy, task in COPIES.items()}}


def shot_config(dev240=False, **agent_changes):
    """The hybrid's simulation config as a shot build: the shot's entry, NVARC's share at the
    competition's soft deadline (its own deadlines stand), the simulation's times."""
    config = HYBN.hybrid_config(
        **{"entry": SHOT, "nvarc_share_s": nbb.COMPETITION_SOLVER_END_S, **agent_changes}
    )
    return config._replace(dev240=dev240)


class Built(NamedTuple):
    wheel: Path
    notebooks: dict
    configs: dict


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The shot notebook and its DEV240 twin."""
    pytest.importorskip("numpy")
    dest = tmp_path_factory.mktemp("build")
    wheel = LOCAL.make_wheel(dest)
    configs = {
        "shot": shot_config(),
        "dev240": shot_config(dev240=True),
    }
    notebooks = {name: HYBN.build(dest, wheel, name, config) for name, config in configs.items()}
    return Built(wheel, notebooks, configs)


def simulate(tmp_path, notebook, root, competition=False, **kwargs):
    return HYBN.simulate(tmp_path, notebook, root, competition, **kwargs)


def shim_modules(run):
    """The modules the kernel ran with python -m, in order."""
    return [call["argv"][1] for call in run.calls if call["argv"][:1] == ["-m"]]


def assert_shot_ran_as_the_entry(run):
    modules = shim_modules(run)
    assert SHOT in modules and "arc26.agent.run" not in modules, modules


# The shot succeeds: NVARC's file first, with NVARC's own deadlines, then the shot on top.


def test_a_development_run_keeps_nvarcs_deadlines_then_the_shot_improves_and_scores(
    built, tmp_path
):
    root = HYBN.make_tree(tmp_path, built.wheel)
    run = simulate(tmp_path, built.notebooks["shot"], root)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    assert list(run.record["STAGES"]) == SHOT_STAGES
    start = run.record["NOTEBOOK_START"]
    # NVARC: the development deadlines of the build, never cut by the shot.
    config = built.configs["shot"]
    run_stage = run.record["STAGES"]["run"]
    assert (run_stage["solver_end_s"], run_stage["solver_kill_s"]) == (
        config.solver_end_s, config.solver_kill_s,
    )  # fmt: skip
    assert HYBN.fake_solver(run)["end_time"] - start == pytest.approx(config.solver_end_s)
    # The agent check printed the shot's plan: the latest start that still runs the shot.
    assert run.record["AGENT_READY"] is True
    plan = run.record["AGENT_PLAN"]
    assert plan["latest_start_s"] == 3600 - 1 - 1 and "agent_start_s" not in plan
    assert "plan, seconds after the start: nvarc_end_s 41400, nvarc_kill_s 42000, " in run.output
    # The shot: python -m arc26.agent.shot with arc26.agent.run's command line, and the
    # development solutions for its own score.
    assert_shot_ran_as_the_entry(run)
    agent = HYBN.fake_agent(run)
    assert agent["mode"] == "dev" and agent["deadline"] == pytest.approx(start + 3600)
    solutions = agent["argv"][agent["argv"].index("--solutions") + 1]
    data_dir = Path(run.record["CHALLENGES"]).parent
    assert solutions == str(data_dir / "arc-agi_evaluation_solutions.json")
    assert agent["nvarc"] == HYBN.nvarc_file(run)
    assert "agent stage: arc26.agent.shot until 3600 s after the start" in run.output
    written = HYBN.assert_valid(run)
    HYBN.assert_nvarc_kept(run)
    assert HYBN.solved(written, *AGENT_OUTPUT)
    stage = HYBN.agent_stage(run)
    assert stage["agent_exit"] == 0 and stage["ended"] == "exited"
    assert stage["score"]["gained"] == ["981571dc_0"] and stage["score"]["lost"] == []
    assert HYBN.working_files(run) == ["run_logs", "submission.json"]
    stages = json.loads((root / "working" / "run_logs" / "stages.json").read_text())
    assert stages["build"]["agent"]["entry"] == SHOT
    # The clocks that may see the time before the first cell (task P6.5b), logged only.
    clocks = stages["stages"]["environment"]["clocks"]
    assert 0 <= clocks["since_start_s"] < 60 and clocks["time"] >= start
    assert {"uptime_s", "pid1_age_s", "kernel_age_s", "working_ctime_age_s"} <= set(clocks)
    assert clocks["kernel_age_s"] <= clocks["pid1_age_s"] <= clocks["uptime_s"] + 1
    HYBN.assert_kernel_never_imported_torch(run)


def test_a_competition_rerun_runs_nvarc_in_full_then_the_shot_and_reads_no_solutions(
    built, tmp_path
):
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=True)
    run = simulate(tmp_path, built.notebooks["shot"], root, competition=True)
    assert run.record["error"] is None, run.output
    start = run.record["NOTEBOOK_START"]
    # NVARC keeps the competition's deadlines: the shot never takes NVARC's time.
    assert HYBN.fake_solver(run)["end_time"] - start == pytest.approx(41400)
    run_stage = run.record["STAGES"]["run"]
    assert (run_stage["solver_end_s"], run_stage["solver_kill_s"]) == (41400, 42000)
    assert_shot_ran_as_the_entry(run)
    agent = HYBN.fake_agent(run)
    assert agent["mode"] == "competition"
    assert agent["tasks"].endswith("arc-agi_test_challenges.json")
    assert not [arg for arg in agent["argv"] if "solution" in arg.lower()]
    written = HYBN.assert_valid(run, HIDDEN)
    assert sorted(written) == sorted(HIDDEN) and HYBN.solved(written, "f0000003")
    HYBN.assert_nvarc_kept(run, HIDDEN)
    opened = [path for _, path in run.record["opened"]]
    assert not [path for path in opened if "solution" in Path(path).name.lower()]
    assert HYBN.working_files(run) == ["submission.json"]
    HYBN.assert_kernel_never_imported_torch(run)


# The shot fails: every case leaves a valid file with NVARC's attempts.


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
@pytest.mark.parametrize("install", ["fail", "no-engine", "hang"])
def test_a_failed_install_runs_no_shot_and_leaves_nvarcs_file(
    built, tmp_path, competition, install
):
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=competition)
    overrides = {"install_timeout_s": 2} if install == "hang" else None
    run = simulate(
        tmp_path, built.notebooks["shot"], root, competition, install=install, overrides=overrides
    )
    assert run.record["error"] is None, run.output
    stage = HYBN.agent_stage(run)
    assert stage["failure"] == "the engine install failed (see install), so the agent did not run"
    assert HYBN.fake_agent(run) is None and SHOT not in shim_modules(run)
    assert HYBN.assert_valid(run, HIDDEN if competition else DEV4) == HYBN.nvarc_file(run)
    assert run.record["cells"][-1].endswith("-logs")


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
@pytest.mark.parametrize("mode", ["boot-fail", "crash", "corrupt", "invalid", "refused"])
def test_a_shot_that_fails_to_boot_or_mid_run_leaves_a_valid_file_with_nvarcs_attempts(
    built, tmp_path, competition, mode
):
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=competition)
    holder = {}
    with HYBN.no_stray_engine(holder):
        holder["run"] = run = simulate(
            tmp_path, built.notebooks["shot"], root, competition, agent=mode
        )
        assert run.record["error"] is None, run.output
        assert_shot_ran_as_the_entry(run)
        counts = HIDDEN if competition else DEV4
        written = HYBN.assert_valid(run, counts)
        HYBN.assert_nvarc_kept(run, counts)
        stage = HYBN.agent_stage(run)
        if mode == "crash":
            assert stage["agent_exit"] == -signal.SIGKILL
            assert HYBN.gone(HYBN.engine_pid(run)), "the engine outlived the shot"
            assert not (root / "working" / ".submission.json.tmp").exists()
        elif mode in ("boot-fail", "corrupt", "refused", "invalid"):
            assert written == HYBN.nvarc_file(run)
        if not competition:
            assert stage["score"]["lost"] == []


@pytest.mark.parametrize(
    ("mode", "ended", "code"),
    [("hang-term", "terminated", 143), ("hang-kill", "killed", -signal.SIGKILL)],
)
@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_shots_end_stops_it_and_its_engine_and_the_file_stays_valid(
    built, tmp_path, competition, mode, ended, code
):
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=competition)
    holder = {}
    with HYBN.no_stray_engine(holder):
        began = time.monotonic()
        holder["run"] = run = simulate(
            tmp_path, built.notebooks["shot"], root, competition, agent=mode,
            overrides={"end_in": 4, "term_s": 2},
        )  # fmt: skip
        assert run.record["error"] is None, run.output
        stage = HYBN.agent_stage(run)
        assert (stage["ended"], stage["agent_exit"]) == (ended, code)
        assert HYBN.gone(HYBN.engine_pid(run)), "the engine outlived the shot's end"
        counts = HIDDEN if competition else DEV4
        HYBN.assert_valid(run, counts)
        HYBN.assert_nvarc_kept(run, counts)
        assert stage["run_s"] < 4 + 2 + 10 and time.monotonic() - began < 120


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_nvarc_ending_too_late_for_min_stage_s_runs_no_shot_and_keeps_nvarcs_file(
    built, tmp_path, competition
):
    """NVARC's stage left 4 s before the generation stop, less than min_stage_s: the stage is
    skipped before anything is copied or installed, and NVARC's file is the submission."""
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=competition)
    run = simulate(
        tmp_path, built.notebooks["shot"], root, competition,
        overrides={"end_in": 5, "min_stage_s": 60},
    )  # fmt: skip
    assert run.record["error"] is None, run.output
    stage = HYBN.agent_stage(run)
    assert "less than AGENT['min_stage_s'] (60 s)" in stage["skipped"]
    assert "install" not in stage and HYBN.fake_agent(run) is None
    written = HYBN.assert_valid(run, HIDDEN if competition else DEV4)
    tasks = ["f0000001", "f0000002"] if competition else HYBN.NVARC_TASKS
    assert all(HYBN.solved(written, task) for task in tasks)
    assert not (run.scratch / "nvarc_submission.json").exists()


# DEV240: every stage reads the doubled files in a development run; a rerun is unchanged.


def test_a_dev240_run_solves_every_task_twice_and_scores_each_copy_apart(built, tmp_path):
    root = HYBN.make_tree(tmp_path, built.wheel)
    run = simulate(
        tmp_path, built.notebooks["dev240"], root, env={"ARC26_FAKE_SOLVER_TASKS": "8"}
    )
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == []
    folder = root / nbb.DEV240_DIR
    assert run.record["CHALLENGES"] == str(folder / "arc-agi_evaluation_challenges.json")
    challenges = json.loads((folder / "arc-agi_evaluation_challenges.json").read_text())
    solutions = json.loads((folder / "arc-agi_evaluation_solutions.json").read_text())
    assert sorted(challenges) == sorted(solutions) == sorted(DEV8)
    for copy, task in COPIES.items():
        assert challenges[copy] == challenges[task] and solutions[copy] == solutions[task]
    # NVARC read the doubled folder; every stage after it saw 8 tasks.
    assert HYBN.fake_solver(run)["config"]["data_dir"] == str(folder)
    assert HYBN.fake_solver(run)["tasks"] == sorted(DEV8)
    agent = HYBN.fake_agent(run)
    assert agent["tasks"] == run.record["CHALLENGES"] and sorted(agent["task_ids"]) == sorted(DEV8)
    assert agent["argv"][agent["argv"].index("--solutions") + 1] == str(
        folder / "arc-agi_evaluation_solutions.json"
    )
    written = HYBN.assert_valid(run, DEV8)
    assert sorted(written) == sorted(DEV8)
    score = HYBN.agent_stage(run)["score"]
    assert score["nvarc"]["tasks"] == 8 and score["nvarc"]["score"] == 8.0
    assert {f"{copy}_0" for copy in COPIES} <= set(score["nvarc"]["solved"])
    stages = json.loads((root / "working" / "run_logs" / "stages.json").read_text())
    dev240 = stages["settings"]["dev240"]
    assert dev240["copies"] == COPIES and dev240["tasks"] == 8
    assert dev240["checked"] == [
        "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json",
        "arc-agi_test_challenges.json",
    ]  # fmt: skip
    assert "DEV240: 8 tasks, 10 test inputs in " in run.output


def test_a_dev240_build_in_a_competition_rerun_solves_the_test_tasks_as_always(built, tmp_path):
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=True)
    run = simulate(tmp_path, built.notebooks["dev240"], root, competition=True)
    assert run.record["error"] is None, run.output
    assert not (root / nbb.DEV240_DIR).exists()
    assert run.record["CHALLENGES"].endswith("arc-agi_test_challenges.json")
    assert sorted(HYBN.assert_valid(run, HIDDEN)) == sorted(HIDDEN)
    opened = [path for _, path in run.record["opened"]]
    assert not [path for path in opened if "solution" in Path(path).name.lower()]
    assert HYBN.working_files(run) == ["submission.json"]


def test_a_dev240_copy_id_that_is_taken_stops_a_development_run(built, tmp_path):
    root = HYBN.make_tree(tmp_path, built.wheel)
    data_dir = root / "input" / "competitions" / LOCAL.COMPETITION
    copy = next(iter(COPIES))
    LOCAL.write_json(data_dir / "arc-agi_training_challenges.json", {copy: {}})
    run = simulate(tmp_path, built.notebooks["dev240"], root)
    assert "DEV240 could not be written: the copy id" in run.record["error"]
    assert not (root / nbb.DEV240_DIR).exists()
    # The placeholder file is valid for the original tasks: the run stopped, nothing was solved.
    assert submit.validate(HYBN.submission(run), LOCAL.standin_challenges(DEV4)) == []


# The real module (task P6.5s merged): the notebook's command line through arc26.agent.shot.


def answer(grid, index=0):
    return f'```answer\n{{"test_index": {index}, "grid": {json.dumps(grid)}}}\n```\n'


def real_shot_config(tmp_path_factory, wheel, run_config="configs/agent_shot.yaml"):
    """The shot build with ``run_config`` (configs/agent_shot.yaml unless named) as built, at the
    simulation's times: waves that take no planned time, quick engine checks, a free port, and no
    engine preparation (the simulated venv has no vLLM; P6.3-05's preparation is
    tests/agent/test_shot.py's)."""
    import yaml

    server = pytest.importorskip("arc26.agent.server")
    data = yaml.safe_load((REPO / run_config).read_text(encoding="utf-8"))
    data["deadline"].update(final_margin_s=1)
    data["waves"].update(
        max_cap=4096, min_cap=256, rate_tokens_per_s=1e6, prefill_tokens_per_s=1e9,
        margin_s=0, setup_s=0,
    )  # fmt: skip
    data["server"].update(
        port=server.free_port(), boot_timeout_s=30, stop_timeout_s=5, health_interval_s=0.05,
        monitor_interval_s=0.1, hang_s=10,
    )  # fmt: skip
    data["engine"].update(max_retries=0)
    data["engine_prep"].update(enabled=False, preflight=False)
    config = shot_config(config=run_config, config_data=data)
    dest = tmp_path_factory.mktemp("real")
    return HYBN.build(dest, wheel, "real", config)


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_real_shot_runs_with_the_notebooks_arguments_and_its_fake_engine(
    built, tmp_path, tmp_path_factory, competition
):
    """python -m arc26.agent.shot itself after the fake NVARC: it reads NVARC's candidates (the
    tasks without one are the least sure), boots the venv's vllm (here P6.4's fake engine),
    passes the sanity request, shoots one wave, merges its answers over NVARC's placeholders,
    and exits 0 before the notebook's SIGTERM; NVARC's attempts stay."""
    pytest.importorskip("arc26.agent.shot")
    notebook = real_shot_config(tmp_path_factory, built.wheel)
    open_tasks = ["f0000003"] if competition else ["981571dc", "aa4ec2a5"]
    replies = tmp_path / "replies.json"
    thinking = {"reasoning": "the grid stays as it is " * 20}
    replies.write_text(json.dumps({
        task: [{"text": "".join(answer(GRID, i) for i in range(n)), **thinking}]
        for task, n in (HIDDEN if competition else DEV4).items()
    }))  # fmt: skip
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=competition)
    run = simulate(
        tmp_path, notebook, root, competition, agent=None,
        env={"ARC26_FAKE_REPLIES": str(replies)}, overrides={"end_in": 40},
    )  # fmt: skip
    assert run.record["error"] is None, run.output
    stage = HYBN.agent_stage(run)
    assert (stage["agent_exit"], stage["ended"]) == (0, "exited"), run.output
    assert stage["run_s"] < 40
    assert_shot_ran_as_the_entry(run)
    counts = HIDDEN if competition else DEV4
    written = HYBN.assert_valid(run, counts)
    HYBN.assert_nvarc_kept(run, counts)
    assert all(HYBN.solved(written, task) for task in open_tasks)
    logs = run.scratch / "logs"
    summary = json.loads((logs / "run.json").read_text())
    assert summary["problems"] == [], summary["problems"]
    assert summary["engine"]["exit_code"] == 0 and not (logs / "engine.pid").exists()
    plan = json.loads((logs / "shot_plan.json").read_text())
    assert [task for task, _ in plan["queue"]] == open_tasks and len(plan["waves"]) == 1
    assert stage["run"]["shot"]["waves"] == 1 and stage["run"]["selection"]
    if competition:
        assert HYBN.working_files(run) == ["submission.json"]
        assert not (logs / "shot_metrics.json").exists()
        assert not (logs / "shot_transcripts.jsonl").exists()
        opened = [path for _, path in run.record["opened"]]
        assert not [path for path in opened if "solution" in Path(path).name.lower()]
    else:
        metrics = json.loads((logs / "shot_metrics.json").read_text())
        assert sorted(metrics["gained"]) == open_tasks and metrics["lost"] == []
        assert stage["score"]["gained"] == [f"{task}_0" for task in open_tasks]
        assert (root / "working" / "run_logs" / "agent" / "shot_plan.json").exists()


@pytest.mark.parametrize(
    ("term_s", "ended"), [(30, "terminated"), (0.05, "killed")], ids=["sigterm", "sigkill"]
)
def test_the_notebooks_stop_reaches_the_real_shots_engine(
    built, tmp_path, tmp_path_factory, term_s, ended
):
    """The real shot with calls still running at AGENT['end_s']: on SIGTERM it cancels them,
    stops its engine, and writes once more within term_s; SIGKILL at once (term_s 0.05) leaves
    its engine running in a session of its own, which the notebook then stops through engine.pid,
    its process group. Either way no engine outlives the stage and NVARC's attempts stay."""
    pytest.importorskip("arc26.agent.shot")
    # A shot that overruns its end (a hang, a stop that never comes): its --deadline 600 s past
    # AGENT['end_s'], so only the notebook's SIGTERM and SIGKILL stop it.
    data = json.loads(real_shot_config(tmp_path_factory, built.wheel).read_text())
    old = '"--deadline", repr(NOTEBOOK_START + AGENT["end_s"]),'
    for cell in data["cells"]:
        text = "".join(cell["source"])
        if cell["id"].endswith("-agent"):
            assert text.count(old) == 1
            text = text.replace(old, '"--deadline", repr(NOTEBOOK_START + AGENT["end_s"] + 600),')
            cell["source"] = text.splitlines(True)
    notebook = tmp_path / "overrun.ipynb"
    notebook.write_text(json.dumps(data))
    replies = tmp_path / "replies.json"
    slow = {"text": answer(GRID), "reasoning": "thinking " * 20, "delay_s": 120}
    replies.write_text(json.dumps({task: [slow] for task in DEV4}))
    root = HYBN.make_tree(tmp_path, built.wheel)
    logs = tmp_path / "agent-scratch" / "logs"
    pid = {}
    try:
        run = simulate(
            tmp_path, notebook, root, agent=None, env={"ARC26_FAKE_REPLIES": str(replies)},
            overrides={"end_in": 12, "term_s": term_s},
        )  # fmt: skip
        assert run.record["error"] is None, run.output
        stage = HYBN.agent_stage(run)
        assert stage["ended"] == ended, run.output
        # The notebook's stop_engine removes engine.pid; its record names the group it stopped.
        assert not (logs / "engine.pid").exists()
        if ended == "killed":
            stopped = stage["engine_stopped"]
            assert stopped.startswith("stopped the engine's process group "), stopped
            pid["engine"] = int(stopped.split("process group ")[1].split(",")[0])
            assert HYBN.gone(pid["engine"]), "the engine outlived the stage"
        else:
            assert "engine_stopped" not in stage
            summary = json.loads((logs / "run.json").read_text())
            assert summary["engine"]["exit_code"] is not None
            assert stage["run_s"] < 12 + term_s
        assert HYBN.assert_valid(run) == HYBN.nvarc_file(run)
    finally:
        if "engine" in pid:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid["engine"], signal.SIGKILL)


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_shot_v2_runs_in_the_notebook_as_a_refill_queue_in_the_bucket_order(
    built, tmp_path, tmp_path_factory, competition
):
    """Submission 4's run config (configs/agent_shot_P6.9.yaml, task P6.9) through the notebook:
    the queue in the bucket order, one refill start per task, no thinking budget in any request
    (the separate forcing turn), the answers merged over NVARC's placeholders, NVARC's attempts
    kept, and no solutions file opened in a competition rerun."""
    pytest.importorskip("arc26.agent.shot")
    notebook = real_shot_config(tmp_path_factory, built.wheel, "configs/agent_shot_P6.9.yaml")
    open_tasks = ["f0000003"] if competition else ["981571dc", "aa4ec2a5"]
    replies = tmp_path / "replies.json"
    thinking = {"reasoning": "the grid stays as it is " * 20}
    replies.write_text(json.dumps({
        task: [{"text": "".join(answer(GRID, i) for i in range(n)), **thinking}]
        for task, n in (HIDDEN if competition else DEV4).items()
    }))  # fmt: skip
    root = HYBN.make_tree(tmp_path, built.wheel, hidden=competition)
    run = simulate(
        tmp_path, notebook, root, competition, agent=None,
        env={"ARC26_FAKE_REPLIES": str(replies)}, overrides={"end_in": 40},
    )  # fmt: skip
    assert run.record["error"] is None, run.output
    stage = HYBN.agent_stage(run)
    assert (stage["agent_exit"], stage["ended"]) == (0, "exited"), run.output
    assert_shot_ran_as_the_entry(run)
    counts = HIDDEN if competition else DEV4
    written = HYBN.assert_valid(run, counts)
    HYBN.assert_nvarc_kept(run, counts)
    assert all(HYBN.solved(written, task) for task in open_tasks)
    logs = run.scratch / "logs"
    summary = json.loads((logs / "run.json").read_text())
    assert summary["problems"] == [], summary["problems"]
    plan = json.loads((logs / "shot_plan.json").read_text())
    assert plan["order"] == "buckets" and plan["mid_bucket"] is False
    assert [task for task, _ in plan["queue"]] == open_tasks
    assert [w["tasks"] for w in plan["waves"]] == [[task] for task in open_tasks]
    assert plan["model"]["refill"] and not plan["model"]["forcing_in_request"]
    assert "budget_probe" not in plan
    calls = [json.loads(line) for line in (logs / "engine_calls.jsonl").open()]
    assert calls and not any("thinking_token_budget" in c for c in calls)
    if competition:
        assert HYBN.working_files(run) == ["submission.json"]
        opened = [path for _, path in run.record["opened"]]
        assert not [path for path in opened if "solution" in Path(path).name.lower()]
    else:
        metrics = json.loads((logs / "shot_metrics.json").read_text())
        assert sorted(metrics["gained"]) == open_tasks and metrics["lost"] == []
