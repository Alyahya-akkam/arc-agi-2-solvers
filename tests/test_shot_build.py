"""The shot build in the notebook builder (task P6.5b): NVARC unchanged, then the long shot.

A hybrid config may set ``agent.entry`` to ``arc26.agent.shot``: the agent stage then runs
P6.3-05's long shot (``python -m arc26.agent.shot``, ``arc26.agent.run``'s arguments and exit
codes) in the time NVARC leaves, NVARC keeping its deadlines. A submission config may also set
``subset: DEV240``, every evaluation task and a copy of each under a derived id, for the 12 h
timing check. Against the task brief's acceptance this file checks:

- every hybrid build config of task P6.5 (no ``entry``) writes, byte for byte, what the builders
  wrote before task P6.5b (``tests/testdata/p6_5b_golden_builds.json``, recorded at
  ``dbbc782``); the pre-P6.5 goldens are ``tests/test_hybrid_build.py``'s;
- the ``entry`` key: its default changes nothing, a shot build keeps NVARC's deadlines, and the
  shot's cells differ from the hybrid's only where ``SHOT_EDITS`` and the first two cells say;
- the shot's time arithmetic (``shot_plan``): every plan the builder accepts ends before 12 h,
  with the latest start that still runs the shot after NVARC's write;
- DEV240: the copies' ids, ``dev240_files``' checks, the build's cells, and every id parser.

The notebook's runs on the simulated Kaggle tree are ``tests/test_shot_notebook.py``'s.
"""

import hashlib
import importlib.util
import inspect
import itertools
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


HYB = load("_shot_hybrid_build", REPO / "tests" / "test_hybrid_build.py")
SCOPE_TESTS = HYB.SCOPE_TESTS
nbb, bs = HYB.nbb, HYB.bs
DROP = HYB.DROP
GOLDEN = json.loads((REPO / "tests" / "testdata" / "p6_5b_golden_builds.json").read_text())
HYBRID_CONFIG = HYB.HYBRID_CONFIG
AGENT_CONFIG = HYB.AGENT_CONFIG
SHOT = "arc26.agent.shot"
# A shot build's agent changes on the hybrid's config, until the shot's own configs are merged:
# NVARC's deadlines kept, the shot's setup and smallest wave in min_stage_s, P6.3-05's install
# limit.
SHOT_CHANGES = {
    "entry": SHOT, "nvarc_share_s": 41400, "end_s": 40050, "min_stage_s": 2700,
    "install_timeout_s": 600,
}  # fmt: skip


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_repo(name):
    return (REPO / name).read_bytes()


def hybrid_golden_files(builder, name, read=read_repo):
    """What ``builder`` writes for a committed hybrid config, from test_scope_build.py's fixed
    provenance, the agent's run config resolved: the notebook, both metadata files, MANIFEST.md."""
    notebooks = builder.notebook_builder
    data = read(name)
    config = notebooks.resolve_agent(
        notebooks.resolve_rounds(notebooks.parse_config(data, Path(name)), read), read
    )
    info = notebooks.build_info(
        config,
        tag="golden",
        commit=SCOPE_TESTS.GOLDEN_COMMIT,
        config_file=name,
        config_sha256=sha256(data),
        wheel=SCOPE_TESTS.GOLDEN_WHEEL,
        wheel_sha256=SCOPE_TESTS.GOLDEN_WHEEL_SHA256,
    )
    notebook = notebooks.build_notebook(config, info, SCOPE_TESTS.SUBMIT_STAND_IN)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = builder.kernel_metadata(config, pulled)
    dataset = builder.dataset_metadata(config.code_dataset, None)
    files = {
        notebooks.notebook_file(config): notebooks.notebook_bytes(notebook),
        builder.METADATA_FILE: builder._json_bytes(meta),
        f"{builder.CODE_DIR}/{builder.DATASET_METADATA_FILE}": builder._json_bytes(dataset),
    }
    files[builder.MANIFEST_FILE] = builder.render_manifest(
        tag="golden",
        build=info,
        commit_time=0,
        wheel_commit=SCOPE_TESTS.GOLDEN_COMMIT,
        wheel_epoch=0,
        config=config,
        builder="uv build --wheel --offline",
        generator="hatchling 1.27.0",
        metadata=meta,
        files=dict(files),
    )
    return files


def shot_text(agent_changes=None, changes=None, name=HYBRID_CONFIG):
    """The hybrid's config as a shot build (SHOT_CHANGES), with more changes, as text."""
    return HYB.config_text(changes or {}, {**SHOT_CHANGES, **(agent_changes or {})}, name)


def shot_config(agent_changes=None, changes=None, read=read_repo):
    config = nbb.parse_config(shot_text(agent_changes, changes), Path("c.yaml"))
    return nbb.resolve_agent(config, read)


def build_info(config):
    stub = dict(HYB.BUILD_STUB)
    return nbb.build_info(config, config_file=stub.pop("config"), **stub)


def cells(config):
    return HYB.cell_sources(nbb.build_notebook(config, build_info(config), "# submit\n"))


def plan(agent, data):
    return nbb.shot_plan(
        agent, data["deadline"]["final_margin_s"], data["server"]["boot_timeout_s"],
        nbb.NVARC_KILL_GAP_S, nbb.WRITE_RESERVE_S, nbb.AGENT_KILL_WAIT_S, nbb.SHOT_SESSION_S,
    )  # fmt: skip


# The hybrid builds of task P6.5: byte for byte as before task P6.5b.


def test_the_hybrid_goldens_cover_every_hybrid_config_of_p6_5():
    assert GOLDEN["commit"].startswith("dbbc782")
    assert set(GOLDEN["configs"]) == set(GOLDEN["inputs"]) == set(HYB.HYBRID_CONFIGS)


@pytest.mark.parametrize("name", sorted(GOLDEN["configs"]))
def test_every_hybrid_build_without_entry_writes_its_recorded_bytes(name):
    """The notebook, both metadata files, and MANIFEST.md of each P6.5 hybrid config equal what
    the builders wrote before task P6.5b, while its inputs hash as recorded then."""
    changed = [
        file
        for file, digest in GOLDEN["inputs"][name].items()
        if not (REPO / file).is_file() or sha256(read_repo(file)) != digest
    ]
    if changed:
        pytest.skip(f"{name}: changed since the goldens were recorded: {', '.join(changed)}")
    assert HYB.parsed(name).agent.entry == nbb.DEFAULT_AGENT_ENTRY
    digests = {file: sha256(data) for file, data in hybrid_golden_files(bs, name).items()}
    assert digests == GOLDEN["configs"][name]


def test_an_explicit_default_entry_builds_what_no_entry_builds():
    """entry: arc26.agent.run is the default spelled out: the same cells, and BUILD without it."""
    text = HYB.config_text({}, {"entry": nbb.DEFAULT_AGENT_ENTRY})
    explicit = nbb.resolve_agent(nbb.parse_config(text, Path("c.yaml")), read_repo)
    implicit = HYB.resolved()
    assert explicit.agent == implicit.agent
    assert "entry" not in explicit.agent.build_settings()
    assert cells(explicit) == cells(implicit)


# The entry key and the shot's rules.


@pytest.mark.parametrize(
    ("agent_changes", "message"),
    [
        ({"entry": "x"}, "agent: entry must be one of arc26.agent.run, arc26.agent.shot"),
        ({"entry": 1}, "entry must be one of"),
        ({"nvarc_share_s": 21600}, "a shot build (arc26.agent.shot) keeps NVARC's deadlines"),
        ({"nvarc_share_s": 0}, "so nvarc_share_s must be 41400, the competition's soft deadline"),
        ({"other": 1}, "agent must be null or have exactly the keys enabled"),
    ],
)  # fmt: skip
def test_a_malformed_entry_or_shot_is_refused(agent_changes, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(shot_text(agent_changes), Path("c.yaml"))
    assert message in str(refused.value)
    assert str(refused.value).startswith("c.yaml: agent")


def test_a_shot_build_records_its_entry_and_keeps_nvarcs_deadlines():
    agent = shot_config().agent
    assert agent.shot and agent.entry == SHOT
    assert agent.build_settings()["entry"] == SHOT
    settings = agent.stage_settings()
    assert "entry" not in settings
    gap = nbb.NVARC_KILL_GAP_S
    competition = (nbb.COMPETITION_SOLVER_END_S, nbb.COMPETITION_SOLVER_KILL_S)
    assert nbb.nvarc_deadlines(*competition, settings, gap) == competition
    # A development run's own deadlines stand too: the share never shortens them.
    assert nbb.nvarc_deadlines(2400, 3000, settings, gap) == (2400, 3000)


def test_the_shot_plan():
    agent = shot_config().agent
    final = agent.config_data["deadline"]["final_margin_s"]
    assert plan(agent.stage_settings(), agent.config_data) == {
        "nvarc_end_s": 41400,
        "nvarc_kill_s": 42000,
        "latest_start_s": 40050 - final - 2700,
        "generation_stop_s": 40050 - final,
        "agent_end_s": 40050,
        "agent_kill_s": 40170,
        "notebook_end_s": 40500,
        "problems": [],
    }
    # The allowance for the time before the first cell (task P6.5b): 45 minutes off 12 h.
    assert nbb.SHOT_SESSION_S == nbb.SESSION_LIMIT_S - nbb.PRE_SCRIPT_ALLOWANCE_S == 40500


def test_a_shot_plan_problem_names_what_breaks():
    agent = {
        "nvarc_share_s": 41400, "end_s": 42000, "term_s": 120, "min_stage_s": 2700,
        "install_timeout_s": 600,
    }  # fmt: skip
    args = (600, 300, 30, 43200)
    assert nbb.shot_plan(agent, 300, 900, *args)["problems"] == []
    never = nbb.shot_plan({**agent, "min_stage_s": 41500}, 300, 900, *args)["problems"]
    assert never == [
        "the shot could never run: it needs min_stage_s (41500 s) before the generation stop at "
        "41700 s (end_s minus final_margin_s), so the stage must start by 200 s, but it follows "
        "NVARC's write (300 s)"
    ]
    alone = nbb.shot_plan({**agent, "nvarc_share_s": 0}, 300, 900, *args)["problems"]
    assert alone == [
        "the shot picks its tasks from NVARC's candidates and keeps NVARC's attempts, so it needs "
        "NVARC's stage: nvarc_share_s is 0"
    ]
    late = nbb.shot_plan({**agent, "end_s": 42800}, 300, 900, *args)["problems"]
    assert late == [
        "the notebook could end at 43250 s, after the session's usable 43200 s (12 h less the "
        "time before the first cell): SIGKILL at 42920 s, 30 s for the process group, 300 s for "
        "the final check and the logs"
    ]
    # The builder plans a shot against SHOT_SESSION_S: an end of 42,000 s no longer fits.
    with pytest.raises(nbb.BuildError, match="after the session's usable 40500 s"):
        shot_config({"end_s": 42000})


def test_every_shot_plan_the_builder_accepts_ends_before_kaggles_limit_in_order():
    """Over a grid of ends, graces, stage minimums, and final margins: each shot the builder
    accepts has a plan whose times follow in order, whose latest start comes after NVARC's write,
    and whose notebook ends by 12 h, NVARC's deadlines the competition's; the rest break a rule."""
    base = HYB.parsed(HYBRID_CONFIG)
    raw = {**yaml.safe_load((REPO / HYBRID_CONFIG).read_text(encoding="utf-8"))["agent"]}
    raw.update(SHOT_CHANGES)
    data = yaml.safe_load(HYB.agent_config_bytes())
    data["server"]["stop_timeout_s"] = 1
    texts = {}
    for final in (0, 300, 1800, 9000):
        data["deadline"]["final_margin_s"] = final
        texts[final] = yaml.safe_dump(data).encode("utf-8")
    order = ["latest_start_s", "generation_stop_s", "agent_end_s", "agent_kill_s", "notebook_end_s"]
    accepted = refused = 0
    grid = itertools.product(
        (600, 3600, 5400, 40050, 42000, 42750),  # end_s
        (30, 120),  # term_s
        (60, 2700, 12000),  # min_stage_s
        (0, 300, 1800, 9000),  # the run config's final_margin_s
    )
    for end, term, stage, final in grid:
        changes = {"end_s": end, "term_s": term, "min_stage_s": stage}

        def read(name, text=texts[final]):
            return text if name == AGENT_CONFIG else read_repo(name)

        try:
            agent = nbb.parse_agent({**raw, **changes}, base._asdict(), Path("c.yaml"))
            config = nbb.resolve_agent(base._replace(agent=agent), read)
        except nbb.BuildError:
            refused += 1
            continue
        accepted += 1
        times = plan(config.agent.stage_settings(), config.agent.config_data)
        values = [times[key] for key in order]
        assert times["problems"] == [] and values == sorted(values), times
        assert times["latest_start_s"] >= nbb.WRITE_RESERVE_S
        assert times["generation_stop_s"] - times["latest_start_s"] == stage
        assert times["notebook_end_s"] <= nbb.SHOT_SESSION_S
        assert (times["nvarc_end_s"], times["nvarc_kill_s"]) == (
            nbb.COMPETITION_SOLVER_END_S, nbb.COMPETITION_SOLVER_KILL_S,
        )  # fmt: skip
    assert accepted > 20 and refused > 20


# The shot build's cells.


def test_a_shot_build_is_the_hybrid_with_the_shot_edits():
    """Against its twin with the default entry (built without the plan check, which the twin's
    share of 41400 fails), a shot build differs in the first two cells, the agent check (the
    entry's constant, shot_plan, and its call), and the cells SHOT_EDITS name, nothing else."""
    shot = shot_config()
    twin = shot._replace(agent=shot.agent._replace(entry=nbb.DEFAULT_AGENT_ENTRY))
    mine, theirs = cells(shot), cells(twin)
    assert list(mine) == list(theirs)
    differ = {name for name in mine if mine[name] != theirs[name]}
    assert differ == {"config-note", "config", "agent-check", *nbb.SHOT_EDITS}
    for name, edits in nbb.SHOT_EDITS.items():
        if name != "agent-check":
            assert mine[name] == nbb._edited(theirs[name], edits, name), name
    assert 'AGENT_ENTRY = "arc26.agent.shot"' in mine["agent-check"]
    assert "def shot_plan(" in mine["agent-check"] and "def shot_plan(" not in theirs["agent-check"]
    assert 'AGENT_PLAN = record["plan"] = shot_plan(\n' in mine["agent-check"]
    assert 'PYTHON, "-m", AGENT_ENTRY,' in mine["agent"]
    assert '"arc26.agent.run"' not in mine["agent"]
    assert '"--solutions"' in mine["agent"] and "if not COMPETITION and DATA_DIR" in mine["agent"]
    assert "'entry': 'arc26.agent.shot'" in mine["config"]
    assert mine["config"].count("arc26.agent.shot") == 2
    for name in ("config", "agent-check", "run", "agent"):
        compile(mine[name], name, "exec")


def test_the_shot_builds_first_cell_and_manifest_state_the_plan():
    shot = shot_config()
    first = cells(shot)["config-note"]
    assert nbb.SHOT_TITLE in first and nbb.agent_text(shot.agent) in first
    assert "Not for submission" not in first
    assert "it runs only when its stage starts by" in nbb.agent_text(shot.agent)
    assert "Hybrid build (task P6.5)" not in first
    rows = "\n".join(bs._submission_summary(shot, "4xL4"))
    assert "| agent entry | `python -m arc26.agent.shot`, the long shot (task P6.5b;" in rows
    assert "NVARC keeps its deadlines, the shot takes the time it leaves |" in rows


def test_the_shot_check_cell_compiles_and_holds_the_helpers_verbatim():
    text = cells(shot_config())["agent-check"]
    compile(text, "agent-check", "exec")
    for helper in (*nbb.AGENT_RUNTIME_HELPERS, nbb.shot_plan):
        assert inspect.getsource(helper).rstrip() in text, helper.__name__


# DEV240.


def test_dev240s_copy_ids_are_the_sha256_prefix_and_collide_with_nothing():
    from arc26.data import DEV120, DEV240, DEV240_COPIES, SUBSETS, dev240_copy_id

    assert dev240_copy_id("0934a4d8") == hashlib.sha256(b"0934a4d8").hexdigest()[:8]
    assert len(DEV240_COPIES) == 120 and not set(DEV240_COPIES) & set(DEV120)
    assert sorted(DEV240_COPIES.values()) == list(DEV120)
    assert DEV240 == tuple(sorted({*DEV120, *DEV240_COPIES})) and len(DEV240) == 240
    assert all(nbb.TASK_ID.fullmatch(task) for task in DEV240)
    assert "DEV240" not in SUBSETS


def challenges(ids, outputs=1):
    grid = [[1]]
    return {t: {"train": [{"input": grid, "output": grid}], "test": [{"input": grid}] * outputs}
            for t in ids}  # fmt: skip


def write_data(folder, ids, extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "arc-agi_evaluation_challenges.json": challenges(ids),
        "arc-agi_evaluation_solutions.json": {t: [[[2]]] for t in ids},
        "arc-agi_test_challenges.json": challenges(["f0000001"]),
        "sample_submission.json": {"a": 1},
        **(extra or {}),
    }
    for name, value in files.items():
        (folder / name).write_text(json.dumps(value))


def test_dev240_files_writes_every_task_twice_with_the_originals_solutions(tmp_path):
    from arc26.data import DEV120, DEV240, DEV240_COPIES

    write_data(tmp_path / "data", DEV120)
    report = nbb.dev240_files(str(tmp_path / "data"), str(tmp_path / "out"))
    written = json.loads((tmp_path / "out" / "arc-agi_evaluation_challenges.json").read_text())
    answers = json.loads((tmp_path / "out" / "arc-agi_evaluation_solutions.json").read_text())
    assert sorted(written) == sorted(answers) == list(DEV240)
    assert list(written)[:120] == list(DEV120)
    for copy, original in DEV240_COPIES.items():
        assert written[copy] == written[original] and answers[copy] == answers[original]
    assert report["copies"] == dict(DEV240_COPIES)
    assert (report["tasks"], report["test_inputs"]) == (240, 240)
    assert report["checked"] == [
        "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json",
        "arc-agi_test_challenges.json",
    ]  # fmt: skip
    assert report["challenges"] == str(tmp_path / "out" / "arc-agi_evaluation_challenges.json")
    assert not list((tmp_path / "out").glob(".*.tmp"))


def test_dev240_files_refuses_a_copy_id_that_is_taken(tmp_path):
    from arc26.data import dev240_copy_id

    copy = dev240_copy_id("0934a4d8")
    write_data(tmp_path / "a", ["0934a4d8"], {"arc-agi_training_challenges.json": {copy: {}}})
    with pytest.raises(ValueError, match=f"the copy id {copy} of 0934a4d8 is taken by "
                       "arc-agi_training_challenges.json"):  # fmt: skip
        nbb.dev240_files(str(tmp_path / "a"), str(tmp_path / "out"))
    # A copy's id equal to another task's of the same file.
    write_data(tmp_path / "b", ["0934a4d8", copy])
    with pytest.raises(ValueError, match="is taken by arc-agi_evaluation_challenges.json"):
        nbb.dev240_files(str(tmp_path / "b"), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_dev240_files_refuses_solutions_that_do_not_match(tmp_path):
    write_data(tmp_path / "data", ["0934a4d8"])
    (tmp_path / "data" / "arc-agi_evaluation_solutions.json").write_text("{}")
    with pytest.raises(ValueError, match="do not cover exactly the evaluation challenges"):
        nbb.dev240_files(str(tmp_path / "data"), str(tmp_path / "out"))
    (tmp_path / "data" / "arc-agi_evaluation_solutions.json").write_text("[1")
    with pytest.raises(ValueError, match="arc-agi_evaluation_solutions.json does not load"):
        nbb.dev240_files(str(tmp_path / "data"), str(tmp_path / "out"))


def test_a_dev240_build_writes_the_copies_in_development_mode_only():
    config = shot_config(changes={"subset": "DEV240"})
    assert config.subset is None and config.dev240
    mine = cells(config)
    theirs = cells(config._replace(dev240=False))
    differ = {name for name in mine if mine[name] != theirs[name]}
    assert differ == {"config-note", "config", "environment", "logs"}
    assert "\nDEV240 = True\n" in mine["config"] and "SUBSET = None\n" in mine["config"]
    assert inspect.getsource(nbb.dev240_files).rstrip() in mine["environment"]
    assert "if DEV240 and not COMPETITION and DATA_DIR is not None:" in mine["environment"]
    assert '"dev240": DEV240_REPORT,' in mine["logs"]
    assert "DEV240 (task P6.5b), development runs only" in mine["config-note"]
    rows = "\n".join(bs._submission_summary(config, "4xL4"))
    assert "| subset (outside a competition rerun) | DEV240: every evaluation task" in rows
    # A competition rerun ignores it: no reason against submitting.
    assert nbb.submission_blocker(config) is None


@pytest.mark.parametrize("subset", ["DEV120", "dev240", ["DEV240"]])
def test_only_dev240_is_a_named_subset(subset):
    with pytest.raises(nbb.BuildError, match=r"subset must be null or a non-empty list .*DEV240"):
        nbb.parse_config(shot_text(changes={"subset": subset}), Path("c.yaml"))


def test_every_id_parser_takes_a_dev240_copy():
    """The runner's subset check, the agent's task loader and NVARC-candidate keys, NVARC's
    output keys, and the calibration report's subset all take a copy's id."""
    from arc26.agent.scheduler import _output_key
    from arc26.agent.tasks import tasks_from_challenges
    from arc26.data import DEV240, DEV240_COPIES
    from arc26.runner import TASK_ID, RunConfig

    copy = next(iter(DEV240_COPIES))
    assert TASK_ID.fullmatch(copy)
    cfg = RunConfig(
        data_dir="/d", model_dir="/m", outputs_dir="/o", marker_dir="/k", subset=(copy,),
        workers=1, hash_seed=0, timing_dir=None,
    )  # fmt: skip
    assert cfg.subset == (copy,)
    assert [task.task_id for task in tasks_from_challenges(challenges([copy]))] == [copy]
    assert _output_key(f"{copy}_1.tp.rot90") == (copy, 1)
    # NVARC's selection splits its output keys on "_" (arc26/nvarc/arc_loader.py).
    assert f"{copy}_0".split("_") == [copy, "0"]
    cal = load("_shot_calibration_report", REPO / "scripts" / "calibration_report.py")
    assert cal.infer_subset([copy]) == "DEV240"
    assert cal.infer_subset(list(DEV240)) == "DEV240"
    assert cal.infer_subset(list(DEV240_COPIES.values())) == "DEV120"


# The P6.5b configs: the smoke's CPU twin, the smoke, and the 12 h check.

P3_CONFIG = "configs/submission_P6.5-03.yaml"
P2_CONFIG = "configs/submission_P6.5-02.yaml"
CPU_CONFIG = "configs/submission_P6.5-02-cpu.yaml"
SHOT_CONFIGS = (CPU_CONFIG, P2_CONFIG, P3_CONFIG)
SHOT_RUN_CONFIG = "configs/agent_shot.yaml"
SMOKE_RUN_CONFIG = "configs/agent_shot_P6.5-02.yaml"
needs_shot_run_config = pytest.mark.skipif(
    not (REPO / SHOT_RUN_CONFIG).is_file() or not (REPO / SMOKE_RUN_CONFIG).is_file(),
    reason="the shot's run configs (task/P6.5s) are not merged yet",
)


def resolved(name):
    return nbb.resolve_agent(HYB.parsed(name), read_repo)


@pytest.mark.parametrize("name", SHOT_CONFIGS)
def test_every_p6_5b_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if HYB.re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == 18 + 13 + 8, len(keys)  # top level, agent (entry included), model, wheels
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


def test_p6_5_03_is_submission_2s_nvarc_then_the_shot_on_dev240():
    """Submission 3's candidate: P2.6-01's NVARC (its deadlines, scope, solver, workers, seed) on
    the P6.5 lineage's code dataset, DEV240, and the shot at the competition's times; fit to
    submit, on the submission notebook."""
    check, p26 = HYB.parsed(P3_CONFIG), HYB.parsed(HYB.P26_CONFIG)
    hybrid = HYB.parsed(HYBRID_CONFIG)
    assert check._replace(code_dataset=p26.code_dataset, agent=None, dev240=False) == p26
    assert check.dev240 and check.subset is None
    assert check.code_dataset == hybrid.code_dataset == "alyahya737/arc26-code-p65"
    assert nbb.submission_blocker(check) is None
    agent = check.agent
    assert agent.entry == SHOT and agent.config == SHOT_RUN_CONFIG
    assert agent.stage_settings() == {
        "enabled": True, "nvarc_share_s": 41400, "end_s": 40050, "term_s": 120,
        "min_stage_s": 2700, "install_timeout_s": 600,
    }  # fmt: skip
    same = ("model", "wheels", "base_python", "scratch_dir", "min_driver")
    assert all(getattr(agent, key) == getattr(hybrid.agent, key) for key in same)
    # A production shot ends by the competition soft deadline less the pre-script allowance.
    early = check._replace(agent=agent._replace(end_s=41400 - nbb.PRE_SCRIPT_ALLOWANCE_S - 1))
    assert "a production agent ends at 38700 s or later" in nbb.submission_blocker(early)


def test_p6_5_02_is_the_check_on_dev4_at_development_scale_on_its_own_notebook():
    smoke, check = HYB.parsed(P2_CONFIG), HYB.parsed(P3_CONFIG)
    from arc26.data import DEV4

    assert tuple(sorted(smoke.subset)) == DEV4 and not smoke.dev240
    assert smoke.kaggle_notebook == "alyahya737/arc26-hybrid-dev"
    assert (smoke.solver_end_s, smoke.solver_kill_s, smoke.agent.end_s) == (2100, 2400, 5300)
    assert smoke.agent.config == SMOKE_RUN_CONFIG
    other = ("kaggle_notebook", "subset", "dev240", "solver_end_s", "solver_kill_s")
    agent = smoke.agent._replace(end_s=check.agent.end_s, config=check.agent.config)
    assert smoke._replace(agent=agent, **{key: getattr(check, key) for key in other}) == check
    blocker = nbb.submission_blocker(smoke)
    assert blocker.startswith("its notebook is alyahya737/arc26-hybrid-dev")
    assert "its agent stage ends 5300 s after the start, a development scale" in blocker


def test_p6_5_02_cpu_is_the_smokes_notebook_on_cpu_with_the_cpu_checks_probe():
    cpu, smoke = HYB.parsed(CPU_CONFIG), HYB.parsed(P2_CONFIG)
    probe = HYB.parsed(HYB.CPU_CONFIG)
    assert cpu.accelerator is None and cpu.kaggle_notebook == probe.kaggle_notebook
    assert (cpu.subset, cpu.workers, cpu.solver_end_s, cpu.solver_kill_s) == (
        probe.subset, probe.workers, probe.solver_end_s, probe.solver_kill_s
    )  # fmt: skip
    assert (cpu.code_dataset, cpu.scope, cpu.solver) == (
        smoke.code_dataset, smoke.scope, smoke.solver
    )  # fmt: skip
    same = ("enabled", "entry", "nvarc_share_s", "term_s", "config", "model", "wheels")
    same += ("base_python", "scratch_dir", "min_driver")
    assert all(getattr(cpu.agent, key) == getattr(smoke.agent, key) for key in same)
    assert (cpu.agent.end_s, cpu.agent.min_stage_s, cpu.agent.install_timeout_s) == (
        probe.agent.end_s, probe.agent.min_stage_s, probe.agent.install_timeout_s
    )  # fmt: skip
    assert "no accelerator" in nbb.submission_blocker(cpu)
    assert cpu.kaggle_notebook not in HYB.ops.briefed_notebooks()


@needs_shot_run_config
def test_the_smokes_run_config_is_the_shots_but_its_max_confidence():
    shot = yaml.safe_load(read_repo(SHOT_RUN_CONFIG))
    smoke = yaml.safe_load(read_repo(SMOKE_RUN_CONFIG))
    assert shot["selection"]["max_confidence"] == 0.5
    assert smoke["selection"]["max_confidence"] == 1.0
    smoke["selection"]["max_confidence"] = 0.5
    assert smoke == shot


@needs_shot_run_config
def test_the_p6_5b_plans_and_min_stage_s_from_the_shots_arithmetic():
    """The plans the runbook and the briefs state, and min_stage_s covering the install (P6.3-05's
    106 s plus 30%), the GPU wait's cap, and the shot's setup plus smallest wave
    (arc26.agent.shot.stage_arithmetic), with under a minute to spare."""
    shot = pytest.importorskip("arc26.agent.shot")
    arithmetic = shot.stage_arithmetic(shot.load_config(REPO / SHOT_RUN_CONFIG))
    need = 1.3 * 106 + nbb.GPU_FREE_WAIT_S + arithmetic["min_shot_s"]
    assert arithmetic["final_margin_s"] == 120
    for name in (P2_CONFIG, P3_CONFIG):
        agent = resolved(name).agent
        assert 0 <= agent.min_stage_s - need < 60, (agent.min_stage_s, need)
    check = resolved(P3_CONFIG).agent
    assert plan(check.stage_settings(), check.config_data) == {
        "nvarc_end_s": 41400, "nvarc_kill_s": 42000, "latest_start_s": 37230,
        "generation_stop_s": 39930, "agent_end_s": 40050, "agent_kill_s": 40170,
        "notebook_end_s": 40500, "problems": [],
    }  # fmt: skip
    smoke = resolved(P2_CONFIG)
    times = plan(smoke.agent.stage_settings(), smoke.agent.config_data)
    assert (times["latest_start_s"], times["generation_stop_s"], times["notebook_end_s"]) == (
        2480, 5180, 5750
    )  # fmt: skip
    # NVARC's worst case, its hard stop and the Write cell's about 30 s on DEV4, still runs a wave.
    assert smoke.solver_kill_s + 30 < times["latest_start_s"]
    cpu = resolved(CPU_CONFIG).agent
    assert plan(cpu.stage_settings(), cpu.config_data)["problems"] == []


@needs_shot_run_config
def test_the_shots_window_on_240_tasks_at_u_0_74():
    """The runbook's arithmetic: NVARC's launch about 86 s after the start (the agent check
    first), its end 0.74 of the 41,350 s window later, the Write cell twice P2.6-01's 44.2 s, the
    install, the GPU wait, and the setup P6.3-05's; the shot then plans a full wave of 15 with
    P6.3-05's longest opening bound, and a second one of 15 only if the first ends at P6.3-05's
    measured 5,784 s, none at its planned 8,036.5 s (the shot's end 40,050 s, task P6.5b's
    allowance for the time before the first cell)."""
    shot = pytest.importorskip("arc26.agent.shot")
    config = shot.load_config(REPO / SHOT_RUN_CONFIG)
    model = config.waves
    stop = 40050 - config.final_margin_s
    first_wave = 86 + 0.74 * 41350 + 2 * 44.2 + 106 + 5 + 345.5
    assert round(stop - first_wave) == 8700
    queue = [(f"{i:08x}", model.opening_tokens) for i in range(30)]
    wave = shot.plan_wave(queue, stop - first_wave, model, 1699566)
    assert wave.cap == 94208 and len(wave.tasks) == 15 and round(wave.planned_s, 1) == 8036.5
    planned = shot.plan_wave(queue[15:], stop - (first_wave + wave.planned_s), model, 1699566)
    measured = shot.plan_wave(queue[15:], stop - (first_wave + 5784), model, 1699566)
    assert planned.tasks == () and planned.cap < model.min_cap
    assert len(measured.tasks) == 15 and 28000 < measured.cap < 28500
    # The latest stage start that still runs a wave: the smallest one, at min_cap.
    agent = resolved(P3_CONFIG).agent
    latest = 40050 - config.final_margin_s - agent.min_stage_s
    assert latest == 37230
    left = stop - (latest + 1.3 * 106 + nbb.GPU_FREE_WAIT_S + model.setup_s)
    assert shot.plan_wave(queue, left, model, 1699566).cap >= model.min_cap
    # The latest NVARC end, the Write cell before the stage: u about 0.896 on 240 tasks.
    assert round((latest - 2 * 44.2 - 86) / 41350, 3) == 0.896


@needs_shot_run_config
@pytest.mark.parametrize("name", SHOT_CONFIGS)
def test_two_builds_of_each_p6_5b_config_are_byte_identical(name):
    first = hybrid_golden_files(bs, name)
    assert first == hybrid_golden_files(bs, name)
    notebook = json.loads(first[nbb.NOTEBOOK_FILE])
    sources = HYB.cell_sources(notebook)
    assert 'AGENT_ENTRY = "arc26.agent.shot"' in sources["agent-check"]
    assert nbb.SHOT_TITLE in sources["config-note"]
    meta = json.loads(first[bs.METADATA_FILE])
    assert {"alyahya737/arc26-dl-qwen38-int4/1", "alyahya737/arc26-dl-agent-wheels/1"} <= set(
        meta["kernel_sources"]
    )


# Task P6.9: Submission 4's build (NVARC as Submissions 2 and 3, then shot v2) and its CPU twin.

P4_CONFIG = "configs/submission_P6.9-02.yaml"
P4_CPU_CONFIG = "configs/submission_P6.9-02-cpu.yaml"
P4_CONFIGS = (P4_CPU_CONFIG, P4_CONFIG)
P4_RUN_CONFIG = "configs/agent_shot_P6.9.yaml"
# Task P6.9c: Submission 4 with the weights' warm-up, the same lineage (tests/test_warmup.py);
# task P6.9d: the same build with MTP and the 69,632 cap (tests/test_p6_9d.py), and its variant 2
# with the cap alone (tests/test_p6_9e.py).
P4_NEXT_CONFIGS = (
    "configs/submission_P6.9-03-cpu.yaml", "configs/submission_P6.9-03.yaml",
    "configs/submission_P6.9-04-cpu.yaml", "configs/submission_P6.9-04.yaml",
    "configs/submission_P6.9-05-cpu.yaml", "configs/submission_P6.9-05.yaml",
    # Task P4.7's rank-256 test and its twin: NVARC alone, built from this lineage's commit with
    # the same wheel, so they attach the same code dataset version.
    "configs/submission_P4.7-01-cpu.yaml", "configs/submission_P4.7-01.yaml",
    # Task P4.7b's version 8 with the rank-256 tail and its twin (tests/test_p4_7b.py).
    "configs/submission_P4.7-02-cpu.yaml", "configs/submission_P4.7-02.yaml",
)  # fmt: skip


@pytest.mark.parametrize("name", P4_CONFIGS)
def test_every_p6_9_build_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if HYB.re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == 18 + 13 + 8, len(keys)
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


def test_p6_9_02_is_submission_3s_build_with_shot_v2_on_its_own_code_dataset():
    """Submission 4's candidate: P6.5-03's build (P2.6-01's NVARC with its deadlines, scope,
    split solver, workers, seed; DEV240; the shot's end at 40,050 s; the submission notebook)
    with only the code dataset, min_stage_s, and the shot's run config changed."""
    check, s3 = HYB.parsed(P4_CONFIG), HYB.parsed(P3_CONFIG)
    assert check.code_dataset == "alyahya737/arc26-code-p69" != s3.code_dataset
    assert check.agent.config == P4_RUN_CONFIG and check.agent.min_stage_s == 6500
    agent = check.agent._replace(config=s3.agent.config, min_stage_s=s3.agent.min_stage_s)
    assert check._replace(code_dataset=s3.code_dataset, agent=agent) == s3
    assert check.kaggle_notebook == "alyahya737/arc26-submission" and check.dev240
    assert (check.solver_end_s, check.solver_kill_s, check.solver) == (41400, 42000, "split")
    assert nbb.submission_blocker(check) is None
    # Every other build lineage's dataset: never this one (one code dataset per lineage; P6.9-03,
    # P6.9-04, P6.9-05, and their twins, tasks P6.9c and P6.9d, are this lineage's next builds of
    # the same wheel).
    lineage = {Path(n).name for n in (*P4_CONFIGS, *P4_NEXT_CONFIGS)}
    others = {HYB.parsed(f"configs/{p.name}").code_dataset
              for p in (REPO / "configs").glob("submission_*.yaml")
              if p.name not in lineage}  # fmt: skip
    assert check.code_dataset not in others


def test_p6_9_02_cpu_is_submission_3s_twin_with_the_check_s_wheel_and_shot():
    cpu, s3_cpu = HYB.parsed(P4_CPU_CONFIG), HYB.parsed(CPU_CONFIG)
    check = HYB.parsed(P4_CONFIG)
    assert cpu.accelerator is None and cpu.kaggle_notebook == s3_cpu.kaggle_notebook
    assert cpu.code_dataset == check.code_dataset and cpu.agent.config == P4_RUN_CONFIG
    assert cpu.agent.end_s == 7200
    agent = cpu.agent._replace(end_s=s3_cpu.agent.end_s, config=s3_cpu.agent.config)
    assert cpu._replace(code_dataset=s3_cpu.code_dataset, agent=agent) == s3_cpu
    assert "no accelerator" in nbb.submission_blocker(cpu)
    assert cpu.kaggle_notebook not in HYB.ops.briefed_notebooks()


def test_the_p6_9_plans_and_min_stage_s_from_shot_v2s_arithmetic():
    """The brief's (docs/prerun/P6.9-02.md, section 6) times: min_stage_s covers the install
    (106 s plus 30%), the GPU wait's cap, and shot v2's setup plus smallest wave of 15 at min_cap
    65,536, with under a minute to spare; the same generation stop and notebook end as P6.5-03,
    the latest start 33,430 s."""
    shot = pytest.importorskip("arc26.agent.shot")
    config = shot.load_config(REPO / P4_RUN_CONFIG)
    arithmetic = shot.stage_arithmetic(config)
    assert (arithmetic["smallest_wave_s"], arithmetic["min_shot_s"]) == (5812.5, 6262.5)
    need = 1.3 * 106 + nbb.GPU_FREE_WAIT_S + arithmetic["min_shot_s"]
    check = resolved(P4_CONFIG).agent
    assert 0 <= check.min_stage_s - need < 60, (check.min_stage_s, need)
    assert plan(check.stage_settings(), check.config_data) == {
        "nvarc_end_s": 41400, "nvarc_kill_s": 42000, "latest_start_s": 33430,
        "generation_stop_s": 39930, "agent_end_s": 40050, "agent_kill_s": 40170,
        "notebook_end_s": 40500, "problems": [],
    }  # fmt: skip
    cpu = resolved(P4_CPU_CONFIG).agent
    times = plan(cpu.stage_settings(), cpu.config_data)
    assert times["problems"] == [] and times["notebook_end_s"] == 7650
    # The twin's probe reaches the engine's preparation: from a shot start after the slowest
    # install it allows (the stage at P6.5-02-cpu's 173.1 s plus 1,500 s), one task's smallest
    # wave still fits before its generation stop.
    one = config.waves.setup_s + config.waves.wave_seconds(config.waves.min_cap, 1, 10_000)
    assert times["generation_stop_s"] - (173.1 + cpu.install_timeout_s) > one


def test_the_latest_nvarc_end_and_the_refill_window_on_240_tasks():
    """The brief's arithmetic: the latest NVARC end that still runs the shot (u about 0.804 on
    240 tasks), NVARC's Timing-fail bound before it, and at P6.5-03's measured first start the
    refill queue's full-cap and last starts (plan_start at 15 streams)."""
    shot = pytest.importorskip("arc26.agent.shot")
    config = shot.load_config(REPO / P4_RUN_CONFIG)
    model = config.waves
    latest = resolved(P4_CONFIG).agent
    latest_start = 40050 - config.final_margin_s - latest.min_stage_s
    assert round((latest_start - 2 * 44.2 - 86) / 41350, 3) == 0.804
    # NVARC's Timing-fail bound (1.1 times P6.5-03's stage time) leaves the shot its start.
    assert 46.9 + 1.1 * 29882.5 + 2 * 44.2 < latest_start
    stop, first, o = 39930, 30445, model.opening_tokens
    assert shot.plan_start(("t", o), stop - first, model, 1699566, 15).cap == 94208
    full = model.wave_seconds(model.max_cap, 15, o)
    smallest = model.wave_seconds(model.min_cap, 15, o)
    assert (round(stop - full), round(stop - smallest)) == (31894, 34117)
    assert shot.plan_start(("t", o), smallest + 1, model, 1699566, 15).cap >= model.min_cap
    assert not shot.plan_start(("t", o), smallest - 1, model, 1699566, 15).tasks
    # Fifteen streams at the full cap fit the KV pool P6.5-03's boot reported.
    assert 15 * (o + model.max_cap + model.forcing_tokens) <= 1699566


@pytest.mark.parametrize("name", P4_CONFIGS)
def test_two_builds_of_each_p6_9_config_are_byte_identical(name):
    first = hybrid_golden_files(bs, name)
    assert first == hybrid_golden_files(bs, name)
    sources = HYB.cell_sources(json.loads(first[nbb.NOTEBOOK_FILE]))
    assert 'AGENT_ENTRY = "arc26.agent.shot"' in sources["agent-check"]
    assert nbb.SHOT_TITLE in sources["config-note"]
    meta = json.loads(first[bs.METADATA_FILE])
    assert "alyahya737/arc26-code-p69" in meta["dataset_sources"]
