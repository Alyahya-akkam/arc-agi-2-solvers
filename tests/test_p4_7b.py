"""Task P4.7b: version 8 plus a rank-256 NVARC tail on rank 16's confidence-0 tasks (run P4.7-02).

``docs/tasks/P4.7b.md``, ``docs/prerun/P4.7-02.md``. This file checks, on CPU:

- the shot's run config ``configs/agent_shot_P4.7b.yaml`` is version 8's
  (``configs/agent_shot_P6.9e.yaml``) in every key but ``selection.zero_bucket`` (false), and
  the shot's queue then takes no task at confidence 0, in either order;
- the shot returns, and stops its engine, as soon as no queued task can start and no call runs
  (``arc26.agent.shot.run_refill``), far before its deadline;
- ``arc26.tail`` queues its challenges file's tasks in the file's order;
- the builder's ``agent.tail``: the keys, bounds, and deadline rule; the tail's helpers: the task
  choice and order, the deadlines and the skip rule (version 8's arithmetic: the notebook still
  ends inside 12 h), and the merge (placeholders only);
- the build configs: ``P4.7-02`` is version 8's (``P6.9-05``) with only ``agent.config``,
  ``agent.tail``, and the subset changed, fit to submit; its twin is ``P6.9-05-cpu``'s with the
  same two agent changes; two builds byte-identical, every other cell version 8's but the edited
  ones;
- end to end on the simulated Kaggle tree (fake NVARC, fake tail solver, the real shot with
  P6.4's fake engine), both modes: an empty shot queue installs no engine and the tail runs at
  once; the tail takes the confidence-0 tasks cheapest first through its own challenges file,
  fills only their placeholders, and records itself; the real shot ends early, then the tail
  runs; a tail that crashes, hangs, or leaves a cut candidate file, a tail with too little time,
  and GPUs still held each leave a valid file with the shot's answers; a competition rerun reads
  no solutions file and writes only ``submission.json``.
"""

import ast
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


SB = load("_p4_7b_shot_build", REPO / "tests" / "test_shot_build.py")
SHOTN = load("_p4_7b_shot_notebook", REPO / "tests" / "test_shot_notebook.py")
WN = load("_p4_7b_warmup_notebook", REPO / "tests" / "test_warmup_notebook.py")
HYB, bs = SB.HYB, SB.bs
HYBN, nbb = SHOTN.HYBN, SHOTN.nbb
# The shim of test_shot_notebook.py, with python -m arc26.tail run by the fake tail solver too.
OLD_FAKES = '"arc26.agent.shot": "ARC26_FAKE_AGENT"}'
NEW_FAKES = '"arc26.agent.shot": "ARC26_FAKE_AGENT", "arc26.tail": "ARC26_FAKE_TAIL"}'
assert HYBN.SHIM.count(OLD_FAKES) == 1
HYBN.SHIM = HYBN.SHIM.replace(OLD_FAKES, NEW_FAKES)
FAKE_TAIL = REPO / "tests" / "testdata" / "fake_tail_solver.py"
DEV4, HIDDEN, GRID = SHOTN.DEV4, SHOTN.HIDDEN, SHOTN.GRID
SHOT_B, SHOT_E = "configs/agent_shot_P4.7b.yaml", "configs/agent_shot_P6.9e.yaml"
GPU, CPU = "configs/submission_P4.7-02.yaml", "configs/submission_P4.7-02-cpu.yaml"
GPU_8, CPU_8 = "configs/submission_P6.9-05.yaml", "configs/submission_P6.9-05-cpu.yaml"
TAIL_SETTINGS = {"write_s": 300, "kill_gap_s": 300, "min_s": 600}
# The simulation's tail: seconds, not minutes.
SIM_TAIL = {"write_s": 2, "kill_gap_s": 2, "min_s": 1}
NINE = list(nbb.REFERENCE_LORA_TARGETS)
# NVARC's fake puzzle seconds: the two tasks it leaves empty, cheapest last by id.
SECONDS = {"0934a4d8": 5.0, "36a08778": 7.0, "981571dc": 50.0, "aa4ec2a5": 10.0}
HIDDEN_SECONDS = {"f0000001": 4.0, "f0000002": 6.0, "f0000003": 30.0}


def raw(name):
    return yaml.safe_load((REPO / name).read_text(encoding="utf-8"))


def leaves(tree, path=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from leaves(value, (*path, key))
    else:
        yield path, tree


def shot():
    return pytest.importorskip("arc26.agent.shot")


# The shot's run config and its queue.


def test_the_run_config_is_version_8s_but_selection_zero_bucket():
    new, old = dict(leaves(raw(SHOT_B))), dict(leaves(raw(SHOT_E)))
    assert set(new) - set(old) == {("selection", "zero_bucket")} and set(old) <= set(new)
    assert {key for key in old if new[key] != old[key]} == set()
    assert new[("selection", "zero_bucket")] is False


def test_every_run_config_value_has_a_comment():
    lines = (REPO / SHOT_B).read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        assert lines[k - 1].strip().startswith("#") or stripped.endswith(":"), line
    assert "TODO" not in "\n".join(lines)


def test_zero_bucket_false_queues_no_task_at_confidence_0_in_either_order():
    module = shot()
    confidence = {"a": 0.0, "b": 0.3, "c": 0.0, "d": 0.45, "e": 0.9, "f": None}
    assert module.select_tasks(confidence, 0.5, order="buckets") == [
        ("b", 0.3), ("d", 0.45), ("a", 0.0), ("c", 0.0)
    ]  # fmt: skip
    assert module.select_tasks(confidence, 0.5, order="buckets", zero_bucket=False) == [
        ("b", 0.3), ("d", 0.45)
    ]  # fmt: skip
    assert module.select_tasks(confidence, 0.5, zero_bucket=False) == [("b", 0.3), ("d", 0.45)]


def test_the_key_defaults_to_true_and_is_checked():
    module = shot()
    assert module.load_config(REPO / SHOT_E).zero_bucket is True
    config = module.load_config(REPO / SHOT_B)
    assert config.zero_bucket is False and config.order == "buckets"
    data = raw(SHOT_B)
    data["selection"]["zero_bucket"] = "no"
    with pytest.raises(module.ConfigError, match="selection.zero_bucket"):
        module.config_from_mapping(data)


def test_the_time_model_is_version_8s():
    module = shot()
    a = module.stage_arithmetic(module.load_config(REPO / SHOT_B))
    b = module.stage_arithmetic(module.load_config(REPO / SHOT_E))
    assert a == b and a["min_shot_s"] == 6262.5


# The shot's early return: the refill queue ends when nothing can start and nothing runs.

ST = load("_p4_7b_agent_test_shot", REPO / "tests" / "agent" / "test_shot.py")


def test_the_shot_returns_and_stops_its_engine_once_nothing_can_start(tmp_path):
    """The refill queue (arc26/agent/shot.py, run_refill: no running task and no start left)
    returns long before the generation stop; run() then stops the engine and writes once more.
    With zero_bucket false the confidence-0 task (aaaa0002 without candidates) is not queued."""
    module = shot()
    challenges, solutions, paths = ST.make_case(tmp_path)
    candidates = json.loads(paths["candidates"].read_text())
    del candidates["aaaa0002_0.view0"]
    paths["candidates"].write_text(json.dumps(candidates))
    config = ST.write_config(
        tmp_path,
        selection={"order": "buckets", "zero_bucket": False},
        waves={"refill": True, "wave_tasks": 2},
    )
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    began = time.monotonic()
    try:
        code = module.main(ST.arguments(paths, out, config, logs, "--fake-engine",
                                        str(paths["replies"]), budget_s=600.0))  # fmt: skip
    finally:
        ST.stop_engine(logs)
    took = time.monotonic() - began
    plan = json.loads((logs / "shot_plan.json").read_text())
    summary = json.loads((logs / "run.json").read_text())
    assert code == 0 and plan["stopped"] == "no tasks left" and plan["zero_bucket"] is False
    assert [task for task, _ in plan["queue"]] == ["aaaa0001", "aaaa0003", "aaaa0005"]
    assert took < 120 < summary["deadline_in_s"]
    assert summary["engine"]["exit_code"] == 0 and not (logs / "engine.pid").exists()


# arc26.tail: the challenges file's order.


def test_the_tail_entry_queues_the_files_order_and_imports_no_gpu_stack():
    module = importlib.import_module("arc26.tail")
    data = {"ffff0001": {}, "aaaa0002": {}, "cccc0003": {}}
    assert module.queue_order(data, {"subset": None}, False) == list(data)
    assert module.queue_order(data, {"subset": ["cccc0003", "ffff0001"]}, False) == [
        "ffff0001", "cccc0003"
    ]  # fmt: skip
    assert module.queue_order(data, {"subset": ["cccc0003"]}, True) == list(data)
    assert module.challenges_path("/d", True) == "/d/arc-agi_test_challenges.json"
    assert module.challenges_path("/d", False) == "/d/arc-agi_evaluation_challenges.json"
    tree = ast.parse((REPO / "arc26" / "tail.py").read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    names = {getattr(n, "module", None) or n.names[0].name for n in top}
    assert not names & {"torch", "torch.multiprocessing", "arc26.solver"}


def test_the_runner_starts_the_given_module_with_the_solvers_arguments():
    from arc26.runner import RunConfig, solver_command

    cfg = RunConfig(data_dir="/d", model_dir="/m")
    split = solver_command(cfg, 5.0, "python")
    tail = solver_command(cfg, 5.0, "python", "arc26.tail")
    assert split[2] == "arc26.solver" and tail[2] == "arc26.tail"
    assert tail[:2] + tail[3:] == split[:2] + split[3:]


# The builder's agent.tail.


def agent_raw(**tail):
    data = raw(GPU)
    data["agent"]["tail"] = tail or dict(TAIL_SETTINGS)
    return data


def parse(data):
    path = REPO / GPU
    return nbb.parse_config(yaml.safe_dump(data).encode(), path)


def test_the_tail_settings_are_parsed_and_built():
    config = parse(agent_raw())
    assert config.agent.tail == TAIL_SETTINGS
    resolved = SB.resolved(GPU)
    assert resolved.agent.build_settings()["tail"] == TAIL_SETTINGS
    assert "tail" not in SB.resolved(GPU_8).agent.build_settings()


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        ({"write_s": 300, "kill_gap_s": 300}, "exactly the keys"),
        ({"write_s": 300, "kill_gap_s": 300, "min_s": 600, "more": 1}, "exactly the keys"),
        ({"write_s": 10, "kill_gap_s": 300, "min_s": 600}, "write_s must be"),
        ({"write_s": 300, "kill_gap_s": 2000, "min_s": 600}, "kill_gap_s must be"),
        ({"write_s": 300, "kill_gap_s": 300, "min_s": 1.5}, "min_s must be"),
    ],
)
def test_a_malformed_tail_is_refused(tail, message):
    with pytest.raises(nbb.BuildError, match=message):
        parse(agent_raw(**tail))


def test_a_tail_needs_a_shot_build_and_time_before_the_agents_end():
    data = agent_raw()
    data["agent"]["entry"] = nbb.DEFAULT_AGENT_ENTRY
    data["agent"]["nvarc_share_s"] = 30000
    del data["agent"]["warmup"]
    with pytest.raises(nbb.BuildError, match="tail needs entry"):
        parse(data)
    data = agent_raw(write_s=3600, kill_gap_s=1800, min_s=600)
    data["agent"]["end_s"] = 5000
    with pytest.raises(nbb.BuildError, match="soft deadline"):
        parse(data)


# The tail's helpers.


def test_the_tail_takes_confidence_0_tasks_the_shot_left_cheapest_first():
    confidence = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.2, "e": 1.0, "f": None, "g": 0.0}
    order, source = nbb.tail_tasks(confidence, ["g"], {"a": 30.0, "b": 10.0, "d": 1.0})
    assert order == ["b", "a", "c"] and source.startswith("rank-16 puzzle seconds, then 1")
    assert nbb.tail_tasks(confidence, [], {}) == (
        ["a", "b", "c", "g"], "task id (no puzzle records)"
    )  # fmt: skip
    assert nbb.tail_tasks({"a": 0.4}, [], {}) == ([], "task id (no puzzle records)")


def test_version_8s_deadlines_end_the_tail_before_the_agents_end_inside_12_h():
    agent = SB.resolved(GPU).agent
    plan = nbb.tail_deadlines(36915.0, agent.end_s, agent.tail)
    assert (plan["solver_end_s"], plan["solver_kill_s"]) == (39450, 39750)
    assert plan["skipped"] is None and plan["left_s"] == 2535.0
    assert plan["solver_kill_s"] + agent.tail["write_s"] == agent.end_s == 40050
    # The final check and the logs keep the shot build's reserve; Kaggle's 12 h with the
    # pre-script allowance hold.
    last = agent.end_s + agent.term_s + nbb.AGENT_KILL_WAIT_S + nbb.WRITE_RESERVE_S
    assert last == 40500 <= nbb.SESSION_LIMIT_S - nbb.PRE_SCRIPT_ALLOWANCE_S
    late = nbb.tail_deadlines(38851.0, agent.end_s, agent.tail)
    assert late["skipped"].startswith("599 s are left") and "min_s (600 s)" in late["skipped"]
    assert nbb.tail_deadlines(38850.0, agent.end_s, agent.tail)["skipped"] is None


def grid_problem(grid):
    from arc26.submit import grid_problem as problem

    return problem(grid)


def test_the_merge_fills_placeholders_only():
    zero, a, b, c = [[0]], [[1]], [[2]], [[3]]
    current = {
        "t1": [{"attempt_1": zero, "attempt_2": zero}, {"attempt_1": a, "attempt_2": zero}],
        "t2": [{"attempt_1": zero, "attempt_2": zero}],
        "t3": [{"attempt_1": zero, "attempt_2": b}],
        "t4": [{"attempt_1": zero, "attempt_2": zero}],
    }
    nvarc = {
        "t1": [{"attempt_1": zero, "attempt_2": zero}, {"attempt_1": a, "attempt_2": zero}],
        "t2": [{"attempt_1": c, "attempt_2": zero}],
        "t3": [{"attempt_1": zero, "attempt_2": zero}],
        "t4": [{"attempt_1": zero, "attempt_2": zero}],
    }
    tail = {
        "t1": [{"attempt_1": b, "attempt_2": c}, {"attempt_1": c, "attempt_2": b}],
        "t2": [{"attempt_1": b, "attempt_2": c}],
        "t3": [{"attempt_1": a, "attempt_2": c}],
        "t4": [{"attempt_1": zero, "attempt_2": zero}],
    }
    counts = {"t1": 2, "t2": 1, "t3": 1, "t4": 1}
    before = copy.deepcopy(current)
    merged, filled = nbb.tail_merge(current, nvarc, tail, counts, grid_problem)
    assert current == before
    assert filled == ["t1_0", "t3_0"]
    assert merged["t1"] == [{"attempt_1": b, "attempt_2": c}, {"attempt_1": a, "attempt_2": zero}]
    assert merged["t2"] == current["t2"] and merged["t4"] == current["t4"]
    assert merged["t3"] == [{"attempt_1": a, "attempt_2": b}]
    bad = {"t1": [{"attempt_1": [[10]], "attempt_2": c}], "t2": "x"}
    assert nbb.tail_merge(current, nvarc, bad, counts, grid_problem)[1] == []


# The build configs and the builds.


def comments_above_every_key(name, count):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if HYB.re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == count, len(keys)
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


@pytest.mark.parametrize("name", [GPU, CPU])
def test_every_p4_7_02_build_config_value_has_a_comment(name):
    comments_above_every_key(name, 18 + 13 + 8 + 10 + 4)


def test_p4_7_02_is_version_8_with_the_shot_config_the_tail_and_the_subset():
    new, old = HYB.parsed(GPU), HYB.parsed(GPU_8)
    assert new.agent.config == SHOT_B and new.agent.tail == TAIL_SETTINGS
    assert new.subset == ("65b59efc", "7666fa5d", "80a900e0")
    agent = new.agent._replace(config=old.agent.config, tail=None)
    assert new._replace(agent=agent, subset=old.subset) == old
    assert nbb.submission_blocker(new) is None


def test_its_twin_is_version_8s_twin_with_the_same_two_agent_changes():
    new, old = HYB.parsed(CPU), HYB.parsed(CPU_8)
    assert new.agent.config == SHOT_B and new.agent.tail == TAIL_SETTINGS
    assert new._replace(agent=new.agent._replace(config=old.agent.config, tail=None)) == old
    assert "no accelerator" in nbb.submission_blocker(new)
    assert new.kaggle_notebook not in HYB.ops.briefed_notebooks()


@pytest.mark.parametrize(("name", "before"), [(GPU, GPU_8), (CPU, CPU_8)], ids=["gpu", "cpu"])
def test_two_builds_are_identical_and_only_the_tail_cells_differ_from_version_8s(name, before):
    first = SB.hybrid_golden_files(bs, name)
    assert first == SB.hybrid_golden_files(bs, name)
    old = SB.hybrid_golden_files(bs, before)
    for key in (bs.METADATA_FILE, f"{bs.CODE_DIR}/{bs.DATASET_METADATA_FILE}"):
        assert first[key] == old[key], key
    new_cells = HYB.cell_sources(json.loads(first[nbb.NOTEBOOK_FILE]))
    old_cells = HYB.cell_sources(json.loads(old[nbb.NOTEBOOK_FILE]))
    assert set(new_cells) - set(old_cells) == {"tail-note", "tail"}
    changed = {cell for cell in old_cells if new_cells[cell] != old_cells[cell]}
    assert changed == {"config-note", "config", "helpers", "agent-check", "agent-note", "agent"}
    assert 'TIMING_DIR = ROOT / "arc26_timing" if COMPETITION' in new_cells["helpers"]
    assert "shot_queue(confidence, AGENT_CONFIG)" in new_cells["agent"]
    assert 'TAIL_MODULE = "arc26.tail"' in new_cells["tail"]
    assert "stop_warmup(" in new_cells["tail"]
    manifest = first[bs.MANIFEST_FILE].decode("utf-8")
    assert "| agent tail | `python -m arc26.tail`" in manifest


# End to end on the simulated Kaggle tree.


def sim_run_config(port):
    """agent_shot_P4.7b.yaml at the simulation's times, as test_shot_notebook.py's real shot."""
    data = raw(SHOT_B)
    data["deadline"].update(final_margin_s=1)
    data["waves"].update(
        max_cap=4096, min_cap=256, rate_tokens_per_s=1e6, prefill_tokens_per_s=1e9,
        margin_s=0, setup_s=0,
    )  # fmt: skip
    data["server"].update(
        port=port, boot_timeout_s=30, stop_timeout_s=5, health_interval_s=0.05,
        monitor_interval_s=0.1, hang_s=10,
    )  # fmt: skip
    data["engine"].update(max_retries=0)
    data["engine_prep"].update(enabled=False, preflight=False)
    return data


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    pytest.importorskip("numpy")
    return HYBN.LOCAL.make_wheel(tmp_path_factory.mktemp("build"))


def tail_notebook(dest, wheel, name, **tail):
    shot()
    server = pytest.importorskip("arc26.agent.server")
    config = SHOTN.shot_config(
        config=SHOT_B, config_data=sim_run_config(server.free_port()), warmup=WN.SIM_WARMUP,
        tail={**SIM_TAIL, **tail},
    )  # fmt: skip
    return HYBN.build(dest, wheel, name, config)


@pytest.fixture(scope="module")
def notebook(tmp_path_factory, wheel):
    return tail_notebook(tmp_path_factory.mktemp("tail"), wheel, "tail")


def replies_file(tmp_path, tasks):
    replies = tmp_path / "replies.json"
    thinking = {"reasoning": "the grid stays as it is " * 20}
    replies.write_text(json.dumps({
        task: [{"text": "".join(SHOTN.answer(GRID, i) for i in range(n)), **thinking}]
        for task, n in tasks.items()
    }))  # fmt: skip
    return replies


def run_tail(tmp_path, notebook, wheel, competition=False, mode="ok", env=None, **kwargs):
    root = HYBN.make_tree(tmp_path, wheel, hidden=competition)
    seconds = HIDDEN_SECONDS if competition else SECONDS
    environment = {
        "ARC26_FAKE_TAIL": str(FAKE_TAIL),
        "ARC26_FAKE_TAIL_LOG": str(tmp_path / "fake_tail.json"),
        "ARC26_FAKE_TAIL_MODE": mode,
        "ARC26_FAKE_SOLVER_SECONDS": json.dumps(seconds),
        "ARC26_FAKE_REPLIES": str(replies_file(tmp_path, HIDDEN if competition else DEV4)),
        **(env or {}),
    }
    kwargs.setdefault("overrides", {"end_in": 60})
    run = SHOTN.simulate(tmp_path, notebook, root, competition, env=environment, **kwargs)
    assert run.record["error"] is None, run.output
    return run


def fake_tail(run):
    path = run.root.parent / "fake_tail.json"
    return json.loads(path.read_text()) if path.exists() else None


def tail_stage(run):
    return run.record["STAGES"]["tail"]


def shot_file(run):
    return json.loads((run.scratch / "shot_submission.json").read_text())


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_an_empty_shot_queue_installs_nothing_and_the_tail_fills_the_empty_tasks(
    tmp_path, notebook, wheel, competition
):
    run = run_tail(tmp_path, notebook, wheel, competition, agent="solve")
    agent = HYBN.agent_stage(run)
    assert agent["shot_queue"] == [] and "queue is empty" in agent["skipped"]
    assert "install" not in agent and HYBN.fake_agent(run) is None
    assert not [a for a in HYBN.LOCAL.argvs(run) if any("install_wheelhouse" in x for x in a)]
    stage = tail_stage(run)
    counts = HIDDEN if competition else DEV4
    empty = ["f0000003"] if competition else ["aa4ec2a5", "981571dc"]
    assert stage["queued"] == empty and stage["order_source"] == "rank-16 puzzle seconds"
    assert stage["started"] == empty and stage["finished"] == empty
    filled = ["f0000003_0"] if competition else ["981571dc_0", "aa4ec2a5_0"]
    assert stage["solver_exit"] == 0 and stage["filled"] == filled
    assert stage["merge"] == f"{len(empty)} outputs filled"
    assert stage["workers"] == len(empty)
    assert stage["lora_r"] == 256 and stage["lora_targets"] == NINE
    assert stage["solver_end_s"] == run.record["AGENT"]["end_s"] - 4
    assert stage["warmup"]["exit"] is not None
    tail = fake_tail(run)
    assert tail["order"] == empty
    config = tail["config"]
    root = run.root
    assert (config["outputs_dir"], config["marker_dir"]) == (
        str(root / "tail_outputs"), str(root / "tail_markers")
    )  # fmt: skip
    assert config["lora_r"] == 256 and config["lora_targets"] == NINE and config["subset"] is None
    assert tail["argv"][1] == repr(run.record["NOTEBOOK_START"] + stage["solver_end_s"])
    written = HYBN.assert_valid(run, counts)
    HYBN.assert_nvarc_kept(run, counts)
    assert all(HYBN.solved(written, task) for task in empty)
    nvarc = HYBN.fake_solver(run)
    if competition:
        assert tail["data_files"] == ["arc-agi_test_challenges.json"]
        assert config["timing_dir"] == str(run.scratch / "tail" / "timing")
        assert nvarc["config"]["timing_dir"] == str(root / "arc26_timing")
        assert HYBN.working_files(run) == ["submission.json"]
        opened = [path for _, path in run.record["opened"]]
        assert not [path for path in opened if "solution" in Path(path).name.lower()]
        assert (run.scratch / "tail" / "tail.json").exists()
    else:
        assert tail["data_files"] == [
            "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"
        ]  # fmt: skip
        logs = root / "working" / "run_logs"
        assert config["timing_dir"] == str(logs / "tail" / "timing")
        assert stage["score"]["gained"] == [f"{t}_0" for t in sorted(empty)]
        assert stage["score"]["lost"] == []
        saved = json.loads((logs / "tail" / "tail.json").read_text())
        assert saved["filled"] == stage["filled"] and saved["seconds"] >= 0
        assert (logs / "tail" / "candidates.json").exists()
        assert json.loads((logs / "stages.json").read_text())["stages"]["tail"]["queued"] == empty
    HYBN.assert_kernel_never_imported_torch(run)


def test_the_real_shot_ends_early_and_the_tail_follows_once_its_engine_is_gone(
    tmp_path, notebook, wheel
):
    """0934a4d8 at NVARC's confidence 1/3 is the shot's one task: the real shot boots P6.4's
    fake engine, answers it, returns long before the agent stage's end, and stops its engine; the
    tail then runs on the two empty tasks, the shot's answer kept."""
    run = run_tail(
        tmp_path, notebook, wheel, agent=None,
        env={"ARC26_FAKE_SOLVER_SPLIT": "0934a4d8"}, overrides={"end_in": 90},
    )  # fmt: skip
    agent = HYBN.agent_stage(run)
    assert agent["shot_queue"] == ["0934a4d8"]
    assert (agent["agent_exit"], agent["ended"]) == (0, "exited"), run.output
    logs = run.scratch / "logs"
    plan = json.loads((logs / "shot_plan.json").read_text())
    summary = json.loads((logs / "run.json").read_text())
    assert plan["stopped"] == "no tasks left" and [t for t, _ in plan["queue"]] == ["0934a4d8"]
    assert summary["engine"]["exit_code"] == 0 and not (logs / "engine.pid").exists()
    stage = tail_stage(run)
    assert agent["run_s"] < 60 and stage["start_s"] >= agent["start_s"] + agent["run_s"]
    assert "engine_stopped" not in stage and stage["gpus"]["after"] == []
    assert stage["shot_answered"] == ["0934a4d8"] and stage["queued"] == ["aa4ec2a5", "981571dc"]
    written = HYBN.assert_valid(run)
    # The shot's merge may move NVARC's attempt 2 of its own task (tests/agent/test_shot.py).
    HYBN.assert_nvarc_kept(run, {task: n for task, n in DEV4.items() if task != "0934a4d8"})
    assert GRID in written["0934a4d8"][0].values()
    assert all(HYBN.solved(written, t) for t in ("981571dc", "aa4ec2a5"))
    assert shot_file(run)["aa4ec2a5"][0]["attempt_1"] == [[0]]
    assert stage["score"]["gained"] == ["981571dc_0", "aa4ec2a5_0"]


@pytest.mark.parametrize("mode", ["crash", "hang", "corrupt"])
def test_a_failing_tail_leaves_a_valid_file_with_the_shots_answers(tmp_path, notebook, wheel, mode):
    overrides = {"end_in": 20 if mode == "hang" else 60}
    run = run_tail(tmp_path, notebook, wheel, mode=mode, agent="solve", overrides=overrides)
    stage = tail_stage(run)
    written = HYBN.assert_valid(run)
    HYBN.assert_nvarc_kept(run)
    if mode == "corrupt":
        assert stage["selection"]["quarantined"] == ["aa4ec2a5_0.rot90"]
        assert stage["filled"] == ["981571dc_0", "aa4ec2a5_0"]
        return
    assert stage["filled"] == [] and written == shot_file(run)
    if mode == "crash":
        assert stage["solver_exit"] == 1 and stage["started"] == []
    else:
        assert stage["solver_exit"] == -9 and stage["solver_s"] < 30


def test_a_tail_without_the_time_it_needs_is_skipped(tmp_path, wheel):
    (tmp_path / "nb").mkdir()
    notebook = tail_notebook(tmp_path / "nb", wheel, "late", min_s=1000)
    run = run_tail(tmp_path, notebook, wheel, agent="solve")
    stage = tail_stage(run)
    assert "less than min_s (1000 s)" in stage["skipped"] and fake_tail(run) is None
    written = HYBN.assert_valid(run)
    HYBN.assert_nvarc_kept(run)
    assert written["aa4ec2a5"][0]["attempt_1"] == written["981571dc"][0]["attempt_1"] == [[0]]


ALWAYS_HELD_SMI = """#!/bin/sh
case "$*" in
  *--query-compute-apps*)
    echo "pid, used_gpu_memory [MiB]"
    ;;
  *memory.used*)
    for i in 0 1 2 3; do echo "$i, 15102"; done
    ;;
  *)
    for i in 1 2 3 4; do echo "NVIDIA L4, 23034 MiB, 8.9, 580.159.04"; done
    ;;
esac
"""


def agent_check_edited(notebook, dest, old, new):
    data = json.loads(notebook.read_text(encoding="utf-8"))
    (cell,) = [c for c in data["cells"] if c["id"].endswith("-agent-check")]
    text = "".join(cell["source"])
    assert text.count(old) == 1
    cell["source"] = text.replace(old, new).splitlines(keepends=True)
    out = dest / f"edited-{notebook.name}"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out


def test_gpus_still_held_skip_the_tail(tmp_path, notebook, wheel):
    edited = agent_check_edited(
        notebook, tmp_path, "GPU_FREE_WAIT_S = 60\n", "GPU_FREE_WAIT_S = 1\n"
    )
    run = run_tail(tmp_path, edited, wheel, agent="solve", smi=ALWAYS_HELD_SMI)
    stage = tail_stage(run)
    assert stage["skipped"].startswith("the GPUs are still held") and fake_tail(run) is None
    assert stage["gpus"]["memory_after"][0]["used_mib"] == 15102
    assert HYBN.assert_valid(run) == shot_file(run)


def test_the_cpu_twins_cells_run_the_tail_as_a_probe(tmp_path, wheel):
    """The twin's build at the simulation's times on a CPU session: no GPU, the shot's queue
    empty, the tail's solver started (here the fake), the file valid."""
    path = REPO / CPU
    config = nbb.resolve_agent(nbb.parse_config(path.read_bytes(), path), HYBN.read_repo)
    data = sim_run_config(pytest.importorskip("arc26.agent.server").free_port())
    agent = config.agent._replace(
        base_python="/nonexistent/python3.11", config_data=data, tail=dict(SIM_TAIL),
        warmup=WN.SIM_WARMUP,
    )  # fmt: skip
    (tmp_path / "nb").mkdir()
    built = HYBN.build(tmp_path / "nb", wheel, "cpu", config._replace(agent=agent))
    run = run_tail(tmp_path, built, wheel, agent="solve", gpus=0, overrides={"end_in": 120})
    assert "queue is empty" in HYBN.agent_stage(run)["skipped"]
    assert tail_stage(run)["solver_exit"] == 0 and fake_tail(run)["config"]["workers"] == 1
    HYBN.assert_valid(run)
