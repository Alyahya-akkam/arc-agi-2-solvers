"""Task P5.1: LLM first, Qwen3.8-27B on every task for the whole session (run P5.1-01).

``docs/tasks/P5.1.md``, ``docs/prerun/P5.1-01.md``. This file checks, on CPU:

- the shot's run config ``configs/agent_shot_P5.1.yaml`` is version 8's
  (``configs/agent_shot_P6.9e.yaml``) in every key but the listed ones, every value commented;
  its stream count is the KV arithmetic's and its time model the stated one;
- ``selection.order: opening`` queues every task by ascending opening bound, then id, and needs
  ``zero_bucket``; ``waves.dev_cap`` lowers the caps in development mode only;
- the real shot (P6.4's fake engine) in that order: every task queued with no NVARC confidence
  and no problem, the development cap in its plan, a competition run at the full cap;
- the builder's ``agent.skip_nvarc``: its rules (a shot build, no tail, the opening order both
  ways), its build settings, and version 9's builds byte for byte as the base commit's builder
  wrote them;
- the build configs: ``P5.1-01`` is version 9's (``P4.7-02``) with only the listed changes, fit to
  submit, and its twin is version 9's twin with the same changes; two builds byte-identical;
  only the listed cells differ from version 9's;
- end to end on the simulated Kaggle tree (the real shot with P6.4's fake engine), both modes: no
  NVARC launch and no selection once the agent check passes, every task shot by opening, the
  shot's answers in both slots, an unanswered task left at its placeholders, the file valid; a
  failed agent check runs NVARC alone; a competition rerun reads no solutions file.
"""

import importlib.util
import json
import sys
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


SB = load("_p5_1_shot_build", REPO / "tests" / "test_shot_build.py")
SHOTN = load("_p5_1_shot_notebook", REPO / "tests" / "test_shot_notebook.py")
ST = load("_p5_1_agent_test_shot", REPO / "tests" / "agent" / "test_shot.py")
HYB, bs = SB.HYB, SB.bs
HYBN, nbb = SHOTN.HYBN, SHOTN.nbb
DEV4, HIDDEN, GRID = SHOTN.DEV4, SHOTN.HIDDEN, SHOTN.GRID
SHOT_P51, SHOT_E = "configs/agent_shot_P5.1.yaml", "configs/agent_shot_P6.9e.yaml"
GPU, CPU = "configs/submission_P5.1-01.yaml", "configs/submission_P5.1-01-cpu.yaml"
GPU_9, CPU_9 = "configs/submission_P4.7-02.yaml", "configs/submission_P4.7-02-cpu.yaml"
SUBSET = ("1818057f", "6e453dd6", "e8686506")
POOL = 1_699_566
# The linear-attention layers' state per request beyond its tokens in the KV use
# (docs/research/shot_tail_screen.md on task/ideas 6e54ba4: the value that best replays the
# logged waves).
STATE = 5000
# Version 9's builds through tests/test_shot_build.py's hybrid_golden_files (fixed provenance),
# written by the builder of the base commit a8b3e5c (task/P4.7, version 9's lineage): recorded
# with runs/P5.1/v9_digests.py on a git archive of that commit, 28 Oct.
V9_DIGESTS = {
    GPU_9: {
        "MANIFEST.md": "b7bdc0e348e6542265c38da47870d5cc057e861c3999e310cd1c974a46e21ba2",
        "arc26-code/dataset-metadata.json":
            "42f2f6cfc4692b609086f31ac7cd251d652dd1a673f1537ec62a54b269f37869",
        "kaggle_submission.ipynb":
            "3d9daf958db83cfac53298d673b34d33f973e66d48e988060116b4090ce704f2",
        "kernel-metadata.json":
            "c8efa0430d63be2227a245fae7c0f3fcaa737d51273e5d57e1f9df6194c1319d",
    },
    CPU_9: {
        "MANIFEST.md": "eac2dc5b2af9e8a7e7e9cc822972fc94dbd693f4562f8e2527e58f29d17362e4",
        "arc26-code/dataset-metadata.json":
            "42f2f6cfc4692b609086f31ac7cd251d652dd1a673f1537ec62a54b269f37869",
        "kaggle_submission.ipynb":
            "57e8075429e31e8c434bbccc044a6eb197f31629ee139f587bc9008790960095",
        "kernel-metadata.json":
            "cea00fceb3c64ab790aa93038921a22d0bd53141c9ab2af798053477a08f063d",
    },
}  # fmt: skip


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


# The shot's run config.


def test_the_run_config_is_version_8s_but_the_listed_keys():
    new, old = dict(leaves(raw(SHOT_P51))), dict(leaves(raw(SHOT_E)))
    assert set(new) - set(old) == {("waves", "dev_cap")} and set(old) <= set(new)
    changed = {key: new[key] for key in old if new[key] != old[key]}
    assert changed == {
        ("selection", "order"): "opening",
        ("waves", "wave_tasks"): 20,
        ("waves", "max_cap"): 65536,
        ("waves", "rate_tokens_per_s"): 30.0,
    }
    assert new[("waves", "dev_cap")] == 16384 and new[("waves", "min_cap")] == 65536
    assert new[("waves", "refill")] is True and new[("waves", "kv_pool_tokens")] == POOL
    assert new[("request", "reasoning_effort")] == "xhigh"
    assert new[("request", "system_prompt")] == "direct_two"
    assert new[("request", "lenient_answers")] is True


def test_every_run_config_value_has_a_comment():
    lines = (REPO / SHOT_P51).read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        assert lines[k - 1].strip().startswith("#") or stripped.endswith(":"), line
    assert "TODO" not in "\n".join(lines)


def test_the_streams_are_what_the_kv_pool_holds_at_the_cap():
    config = shot().load_config(REPO / SHOT_P51)
    model = config.waves
    per_stream = model.opening_tokens + model.max_cap + model.forcing_tokens
    assert per_stream == 6771 + 65536 + 6144 == 78451
    assert POOL // (per_stream + STATE) == model.wave_tasks == 20
    assert POOL // per_stream == 21
    assert model.wave_tasks * (per_stream + STATE) <= POOL < 21 * (per_stream + STATE)
    assert model.wave_tasks <= config.server.max_num_seqs


def test_the_time_model_and_min_stage_s():
    module = shot()
    arithmetic = module.stage_arithmetic(module.load_config(REPO / SHOT_P51))
    assert arithmetic["full_turn_s"] == 2570.0 and arithmetic["full_forcing_s"] == 1713.1
    assert arithmetic["full_wave_s"] == arithmetic["smallest_wave_s"] == 4583.1
    assert arithmetic["min_shot_s"] == 5033.1 and arithmetic["least_rate_full"] == 23.31
    agent = HYB.parsed(GPU).agent
    assert 138 + 60 + arithmetic["min_shot_s"] <= agent.min_stage_s == 5300


# The opening order and the development cap.


def test_the_opening_order_queues_every_task_smallest_opening_first():
    module = shot()
    confidence = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    openings = {"a": 900, "b": 300, "c": 900, "d": 120}
    assert module.select_tasks(confidence, 0.5, order="opening", openings=openings) == [
        ("d", 0.0), ("b", 0.0), ("a", 0.0), ("c", 0.0)
    ]  # fmt: skip
    assert [t for t, _ in module.select_tasks(
        confidence, 0.5, {"b"}, order="opening", openings=openings)] == ["d", "a", "c"]  # fmt: skip
    with pytest.raises(ValueError, match="opening bounds"):
        module.select_tasks(confidence, 0.5, order="opening")


def test_the_opening_order_needs_zero_bucket_and_refuses_mid_bucket():
    module = shot()
    data = raw(SHOT_P51)
    data["selection"]["zero_bucket"] = False
    with pytest.raises(module.ConfigError, match="zero_bucket must be true"):
        module.config_from_mapping(data)
    data = raw(SHOT_P51)
    data["selection"].update(mid_bucket=True, max_confidence=0.7)
    with pytest.raises(module.ConfigError, match="mid_bucket needs order buckets"):
        module.config_from_mapping(data)


@pytest.mark.parametrize(
    ("value", "message"),
    [(70000, "must not exceed waves.max_cap"), (0, "dev_cap must be an integer"),
     (1.5, "dev_cap must be an integer"), (True, "dev_cap must be an integer")],
)  # fmt: skip
def test_a_bad_dev_cap_is_refused(value, message):
    module = shot()
    data = raw(SHOT_P51)
    data["waves"]["dev_cap"] = value
    with pytest.raises(module.ConfigError, match=message):
        module.config_from_mapping(data)


def test_the_dev_cap_applies_in_development_mode_only():
    module = shot()
    config = module.load_config(REPO / SHOT_P51)
    dev = module.for_mode(config, "dev")
    assert (dev.waves.max_cap, dev.waves.min_cap) == (16384, 16384)
    assert dev.waves.wave_tasks == 20 and dev.dev_cap == 16384
    assert module.for_mode(config, "competition") is config
    assert (config.waves.max_cap, config.waves.min_cap) == (65536, 65536)
    v8 = module.load_config(REPO / SHOT_E)
    assert v8.dev_cap is None and v8.order == "buckets" and module.for_mode(v8, "dev") is v8


@pytest.mark.parametrize("mode", ["dev", "competition"])
def test_the_real_shot_queues_every_task_without_nvarc_and_caps_a_dev_run(tmp_path, mode):
    """The shot in the opening order: aaaa0004, at NVARC's confidence 1 in the candidates, is
    queued like every other task; no confidence is computed and no problem logged; the
    development cap is in the plan in a development run only."""
    module = shot()
    challenges, solutions, paths = ST.make_case(tmp_path)
    config = ST.write_config(
        tmp_path,
        selection={"order": "opening", "zero_bucket": True},
        waves={"refill": True, "wave_tasks": 2, "dev_cap": 2048},
    )
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    extra = ["--fake-engine", str(paths["replies"])]
    if mode == "dev":
        extra += ["--solutions", str(tmp_path / "solutions.json")]
    try:
        code = module.main(ST.arguments(paths, out, config, logs, *extra, mode=mode,
                                        budget_s=120.0))  # fmt: skip
    finally:
        ST.stop_engine(logs)
    plan = json.loads((logs / "shot_plan.json").read_text())
    summary = json.loads((logs / "run.json").read_text())
    assert code == 0 and summary["problems"] == [], summary["problems"]
    assert plan["order"] == "opening" and plan["confidence_source"] == "none (order opening)"
    # Equal openings (every task's grids have one size): the id breaks the tie.
    assert [task for task, _ in plan["queue"]] == sorted(challenges)
    assert {conf for _, conf in plan["queue"]} == {0.0}
    if mode == "dev":
        assert plan["dev_cap"] == 2048 and plan["model"]["max_cap"] == 2048
        assert {w["cap"] for w in plan["waves"]} == {2048}
    else:
        assert "dev_cap" not in plan and plan["model"]["max_cap"] == 4096
    written = json.loads(out.read_text())
    # aaaa0004's right answer is the fixture's NVARC attempt 1, which the merge keeps.
    assert ST.solved(written, solutions) == sorted(challenges)
    assert len(plan["waves"]) == len(challenges)


# The builder's agent.skip_nvarc.


def parse(data):
    return nbb.parse_config(yaml.safe_dump(data).encode(), REPO / GPU)


def test_skip_nvarc_is_parsed_and_built_only_when_set():
    config = SB.resolved(GPU)
    assert config.agent.skip_nvarc is True and config.agent.tail is None
    assert config.agent.build_settings()["skip_nvarc"] is True
    v9 = SB.resolved(GPU_9).agent.build_settings()
    assert "skip_nvarc" not in v9 and "tail" in v9
    data = raw(GPU)
    data["agent"]["skip_nvarc"] = False
    assert parse(data).agent.skip_nvarc is False
    assert "skip_nvarc" not in parse(data).agent._replace(
        config_data={}, config_sha256="0"
    ).build_settings()  # fmt: skip


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"skip_nvarc": "yes"}, "skip_nvarc must be true or false"),
        ({"tail": {"write_s": 300, "kill_gap_s": 300, "min_s": 600}}, "exclude each other"),
    ],
)
def test_a_malformed_skip_nvarc_is_refused(changes, message):
    data = raw(GPU)
    data["agent"].update(changes)
    with pytest.raises(nbb.BuildError, match=message):
        parse(data)


def test_skip_nvarc_needs_a_shot_build():
    data = raw(GPU)
    data["agent"].update(entry=nbb.DEFAULT_AGENT_ENTRY, nvarc_share_s=30000)
    del data["agent"]["warmup"]
    with pytest.raises(nbb.BuildError, match="skip_nvarc needs entry"):
        parse(data)


def test_the_opening_order_and_skip_nvarc_go_together():
    def resolve(name, run_config):
        config = HYB.parsed(name)
        config = config._replace(agent=config.agent._replace(config=run_config))
        return nbb.resolve_agent(config, HYBN.read_repo)

    with pytest.raises(nbb.BuildError, match="needs selection.order opening"):
        resolve(GPU, "configs/agent_shot_P4.7b.yaml")
    with pytest.raises(nbb.BuildError, match="needs agent.skip_nvarc"):
        resolve(GPU_9, SHOT_P51)


@pytest.mark.parametrize("name", [GPU_9, CPU_9])
def test_version_9s_builds_are_byte_for_byte_the_base_builders(name):
    files = SB.hybrid_golden_files(bs, name)
    assert {key: SB.sha256(value) for key, value in files.items()} == V9_DIGESTS[name]


# The build configs and the builds.


@pytest.mark.parametrize("name", [GPU, CPU])
def test_every_p5_1_01_build_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if HYB.re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == 18 + 13 + 8 + 10 + 1, len(keys)
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


def test_p5_1_01_is_version_9_with_the_listed_changes():
    new, old = HYB.parsed(GPU), HYB.parsed(GPU_9)
    assert new.code_dataset == "alyahya737/arc26-code-p51" != old.code_dataset
    assert new.subset == SUBSET and new.agent.config == SHOT_P51
    assert new.agent.min_stage_s == 5300 and new.agent.warmup["delay_s"] == 0
    assert new.agent.skip_nvarc and new.agent.tail is None
    agent = new.agent._replace(
        config=old.agent.config, min_stage_s=old.agent.min_stage_s, skip_nvarc=False,
        tail=old.agent.tail, warmup={**new.agent.warmup, "delay_s": 180},
    )  # fmt: skip
    assert new._replace(agent=agent, subset=old.subset, code_dataset=old.code_dataset) == old
    assert nbb.submission_blocker(new) is None
    assert (new.agent.end_s, new.agent.term_s, new.agent.nvarc_share_s) == (40050, 120, 41400)


def test_its_twin_is_version_9s_twin_with_the_same_changes():
    new, old = HYB.parsed(CPU), HYB.parsed(CPU_9)
    gpu = HYB.parsed(GPU)
    assert new.code_dataset == gpu.code_dataset and new.agent.config == SHOT_P51
    assert new.agent.skip_nvarc and new.agent.tail is None
    agent = new.agent._replace(config=old.agent.config, skip_nvarc=False, tail=old.agent.tail)
    assert new._replace(agent=agent, code_dataset=old.code_dataset) == old
    assert "no accelerator" in nbb.submission_blocker(new)
    assert new.kaggle_notebook not in HYB.ops.briefed_notebooks()


@pytest.mark.parametrize(("name", "before"), [(GPU, GPU_9), (CPU, CPU_9)], ids=["gpu", "cpu"])
def test_two_builds_are_identical_and_only_the_listed_cells_differ_from_version_9s(name, before):
    first = SB.hybrid_golden_files(bs, name)
    assert first == SB.hybrid_golden_files(bs, name)
    old = SB.hybrid_golden_files(bs, before)
    assert first[bs.METADATA_FILE] != old[bs.METADATA_FILE]  # the code dataset
    meta = json.loads(first[bs.METADATA_FILE])
    assert "alyahya737/arc26-code-p51" in meta["dataset_sources"]
    assert "alyahya737/arc26-code-p69" not in meta["dataset_sources"]
    new_cells = HYB.cell_sources(json.loads(first[nbb.NOTEBOOK_FILE]))
    old_cells = HYB.cell_sources(json.loads(old[nbb.NOTEBOOK_FILE]))
    assert set(old_cells) - set(new_cells) == {"tail-note", "tail"}
    assert set(new_cells) <= set(old_cells)
    changed = {cell for cell in new_cells if new_cells[cell] != old_cells[cell]}
    assert changed == {
        "config-note", "config", "helpers", "run-note", "run", "write-note", "write",
        "agent-check-note", "agent-check", "agent-note", "agent",
    }  # fmt: skip
    assert nbb.SKIP_NVARC_REASON in new_cells["run"] and nbb.SKIP_NVARC_REASON in new_cells["write"]
    assert "shot_queue(" not in new_cells["agent"]
    assert nbb.SKIP_NVARC_TITLE in new_cells["config-note"]
    manifest = first[bs.MANIFEST_FILE].decode("utf-8")
    assert "LLM first (task P5.1)" in manifest and "agent tail" not in manifest


# End to end on the simulated Kaggle tree.


def sim_run_config(port, dev_cap=2048):
    """agent_shot_P5.1.yaml at the simulation's times, as test_shot_notebook.py's real shot."""
    data = raw(SHOT_P51)
    data["deadline"].update(final_margin_s=1)
    data["waves"].update(
        max_cap=4096, min_cap=256, rate_tokens_per_s=1e6, prefill_tokens_per_s=1e9,
        margin_s=0, setup_s=0, dev_cap=dev_cap,
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


@pytest.fixture(scope="module")
def notebook(tmp_path_factory, wheel):
    shot()
    server = pytest.importorskip("arc26.agent.server")
    load_warmup = load("_p5_1_warmup_notebook", REPO / "tests" / "test_warmup_notebook.py")
    config = SHOTN.shot_config(
        config=SHOT_P51, config_data=sim_run_config(server.free_port()),
        warmup=load_warmup.SIM_WARMUP, skip_nvarc=True,
    )  # fmt: skip
    return HYBN.build(tmp_path_factory.mktemp("p51"), wheel, "p51", config)


def replies_file(tmp_path, tasks, silent=()):
    """Every task's reply: GRID for each test input, or no answer block for ``silent``."""
    replies = tmp_path / "replies.json"
    thinking = {"reasoning": "the grid stays as it is " * 20}
    replies.write_text(json.dumps({
        task: [{"text": "no grid" if task in silent else "".join(
            SHOTN.answer(GRID, i) for i in range(n)), **thinking}]
        for task, n in tasks.items()
    }))  # fmt: skip
    return replies


def run_p51(tmp_path, notebook, wheel, competition=False, silent=(), **kwargs):
    root = HYBN.make_tree(tmp_path, wheel, hidden=competition)
    tasks = HIDDEN if competition else DEV4
    env = {"ARC26_FAKE_REPLIES": str(replies_file(tmp_path, tasks, silent))}
    kwargs.setdefault("overrides", {"end_in": 60})
    run = SHOTN.simulate(tmp_path, notebook, root, competition, agent=None, env=env, **kwargs)
    assert run.record["error"] is None, run.output
    return run


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_no_nvarc_runs_and_the_shot_answers_every_task_by_opening(
    tmp_path, notebook, wheel, competition
):
    silent = ("f0000002",) if competition else ("36a08778",)
    run = run_p51(tmp_path, notebook, wheel, competition, silent=silent)
    stages = run.record["STAGES"]
    assert stages["run"]["skipped"] == nbb.SKIP_NVARC_REASON
    assert stages["run"]["solver_exit"] is None and HYBN.fake_solver(run) is None
    assert stages["write"]["skipped"].startswith("LLM first")
    assert "make_submission_error" not in stages["write"]
    assert stages["write"]["submission"]["raw_status"] == "none"
    assert "nvarc_fallback" not in stages["agent"]
    agent = HYBN.agent_stage(run)
    assert (agent["agent_exit"], agent["ended"]) == (0, "exited"), run.output
    assert not agent["candidates"]["written"]
    assert not any("--nvarc-candidates" in str(x) for x in agent["command"])
    counts = HIDDEN if competition else DEV4
    written = HYBN.assert_valid(run, counts)
    logs = run.scratch / "logs"
    summary = json.loads((logs / "run.json").read_text())
    assert summary["problems"] == [], summary["problems"]
    plan = json.loads((logs / "shot_plan.json").read_text())
    assert plan["order"] == "opening" and sorted(t for t, _ in plan["queue"]) == sorted(counts)
    openings = [w["opening"] for w in plan["waves"]]
    assert openings == sorted(openings) and len(openings) == len(counts)
    assert plan["model"]["max_cap"] == (4096 if competition else 2048)
    for task, n in counts.items():
        for index in range(n):
            entry = written[task][index]
            if task in silent:
                assert entry == {"attempt_1": [[0]], "attempt_2": [[0]]}, task
            else:
                assert entry["attempt_1"] == GRID, task
    if competition:
        assert HYBN.working_files(run) == ["submission.json"]
        opened = [path for _, path in run.record["opened"]]
        assert not [path for path in opened if "solution" in Path(path).name.lower()]
    else:
        assert agent["score"]["final"]["solved"]
        assert json.loads((run.root / "working" / "run_logs" / "stages.json").read_text())
    HYBN.assert_kernel_never_imported_torch(run)


def test_a_failed_agent_check_runs_nvarc_alone(tmp_path, notebook, wheel):
    """Without the weights' output the agent check fails: in a competition rerun NVARC then runs
    with its full deadlines and the agent stage is skipped, as in version 9."""
    root = HYBN.make_tree(tmp_path, wheel, hidden=True, missing=(HYBN.MODEL_NOTEBOOK,))
    env = {"ARC26_FAKE_REPLIES": str(replies_file(tmp_path, HIDDEN))}
    run = SHOTN.simulate(tmp_path, notebook, root, True, agent=None, env=env,
                         overrides={"end_in": 60})  # fmt: skip
    assert run.record["error"] is None, run.output
    stages = run.record["STAGES"]
    assert run.record["AGENT_READY"] is False and stages["agent_check"]["ready"] is False
    assert "skipped" not in stages["run"] and "skipped" not in stages["write"]
    start = run.record["NOTEBOOK_START"]
    solver = HYBN.fake_solver(run)
    assert solver["end_time"] - start == pytest.approx(nbb.COMPETITION_SOLVER_END_S)
    assert HYBN.agent_stage(run)["skipped"] == "the agent check did not pass (stages.agent_check)"
    HYBN.assert_valid(run, HIDDEN)
