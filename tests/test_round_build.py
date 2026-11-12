"""Round arms in the notebook builder and run P2.3-01's build (task P2.3; ADR-015).

A hold-out arm of a series may run the refinement rounds after its gate (``rounds``): the builder
checks the settings with ``arc26.loop.Rounds``' rules, embeds the commit's governor config (the
wheel carries ``arc26/`` only), and the notebook passes the settings inside the arm's hold-out
settings, with the arm's round folder. This file checks the config rules, the embedding, the
notebook's cells, ``MANIFEST.md``, run ``P2.3-01``'s config and build, its notebook on the
simulated Kaggle tree of ``tests/test_notebook_local.py`` (whose helpers it imports, as the build
helpers of ``tests/test_build_submission.py``), and the brief's header. Nothing reaches Kaggle.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from arc26 import governor, loop
from arc26.loop import Rounds

REPO = Path(__file__).resolve().parent.parent
CONFIG = "configs/submission_P2.3-01.yaml"
P21_CONFIG = "configs/submission_P2.1-01.yaml"
BRIEF = REPO / "docs" / "prerun" / "P2.3-01.md"
ARM = "r256-proj-emb-rounds"
DEV4 = ("0934a4d8", "36a08778", "981571dc", "aa4ec2a5")
GOVERNOR = yaml.safe_load((REPO / "configs" / "governor.yaml").read_text(encoding="utf-8"))
ROUNDS = {
    "max_rounds": 1, "tau": 0.5625, "continue_steps": 32, "final_in_prompts": True,
    "governor": "configs/governor.yaml",
}  # fmt: skip
HOLDOUT = {"rule": "last", "k2": True, "in_test_prompts": False}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


BUILD_TESTS = load("_round_build_submission", REPO / "tests" / "test_build_submission.py")
LOCAL = load("_round_notebook_local", REPO / "tests" / "test_notebook_local.py")
nbb, bs, ops = BUILD_TESTS.nbb, BUILD_TESTS.bs, BUILD_TESTS.ops


def parsed(name=CONFIG):
    path = REPO / name
    return nbb.parse_config(path.read_bytes(), path)


def resolved(config):
    return nbb.resolve_rounds(config, lambda name: (REPO / name).read_bytes())


def series(arms):
    """P2.3-01's config with other arms, as text, for parse_config."""
    raw = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    raw["arms"] = arms
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def arm(**changes):
    entry = {
        "name": ARM, "lora_r": 256, "lora_targets": list(LOCAL.NINE), "reason": "a round arm",
        "holdout": dict(HOLDOUT), "rounds": dict(ROUNDS),
    }  # fmt: skip
    entry.update(changes)
    return entry


# The builder's rules are the loop's.


def test_the_builders_round_settings_are_the_loops():
    assert (nbb.MAX_ROUNDS, nbb.EPOCH_VIEWS) == (loop.MAX_ROUNDS, loop.EPOCH_VIEWS)
    assert set(nbb.ROUNDS_KEYS) == set(Rounds.FIELDS) - {"round_dir"}
    assert nbb.GOVERNOR_POLICY == governor.REFERENCE
    assert set(nbb.GOVERNOR_KEYS) <= set(GOVERNOR)
    assert "rounds" in nbb.ARM_OPTIONAL_KEYS and nbb.rounds_text in nbb.RUNTIME_HELPERS


# Run P2.3-01's config.


def test_p2_3_01_config_is_p2_1_01s_arm_with_one_round_on_dev4():
    config, before = parsed(), parsed(P21_CONFIG)
    assert config.kaggle_notebook == "alyahya737/arc26-refine-rounds"
    assert (config.accelerator, config.workers, config.hash_seed) == ("NvidiaL4", 4, 0)
    assert config.subset == DEV4 and config.run_solver is True and config.solver == "split"
    assert (config.solver_end_s, config.solver_kill_s) == (18000, 18900)
    for key in BUILD_TESTS.STACK:
        assert getattr(config, key) == getattr(before, key)
    (round_arm,) = config.arms
    (gate_arm,) = before.arms
    assert round_arm.name == ARM and len(round_arm.reason) <= nbb.MAX_REASON
    assert (round_arm.lora_r, round_arm.lora_targets, round_arm.holdout) == (
        gate_arm.lora_r, gate_arm.lora_targets, gate_arm.holdout
    )  # fmt: skip
    # The production path: the final decode with the held-out pair (ADR-013 as amended).
    assert round_arm.rounds == nbb.ArmRounds(1, 0.5625, 32, True, "configs/governor.yaml")
    assert round_arm.rounds.final_in_prompts is Rounds.final_in_prompts
    assert gate_arm.rounds is None
    blocker = nbb.submission_blocker(config)
    assert blocker.startswith("its notebook is alyahya737/arc26-refine-rounds")
    assert "the held-out gate, and the refinement rounds" in blocker


def test_every_p2_3_01_config_value_has_a_comment():
    lines = (REPO / CONFIG).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"no comment above {line!r}"


def test_the_round_arm_reaches_the_loop_as_rounds_settings():
    """What the builder embeds in the Config cell, with the notebook's folder added, is accepted
    by arc26.loop.Rounds and carries the commit's governor config."""
    config = resolved(parsed())
    (round_arm,) = config.arms
    data = (REPO / "configs" / "governor.yaml").read_bytes()
    assert round_arm.rounds.governor_config == GOVERNOR
    assert round_arm.rounds.governor_sha256 == BUILD_TESTS.sha256(data)
    build = {"tag": "t", "commit": "0" * 40, "config": CONFIG}
    namespace = {}
    exec(nbb.config_cell_source(config, build), namespace)
    (entry,) = namespace["ARMS"]
    assert entry["holdout"] == HOLDOUT
    assert entry["rounds"] == {**{k: v for k, v in ROUNDS.items() if k != "governor"},
                               "governor": GOVERNOR}  # fmt: skip
    settings = Rounds.from_mapping({**entry["rounds"], "round_dir": "/kaggle/round_outputs/a"})
    assert (settings.max_rounds, settings.tau, settings.continue_steps) == (1, 0.5625, 32)
    assert settings.governor == GOVERNOR and settings.final_in_prompts is True


@pytest.mark.parametrize(
    "changes",
    [
        {"max_rounds": 2},
        {"tau": 0},
        {"tau": 1},
        {"continue_steps": 1},
        {"continue_steps": 128},
        {"final_in_prompts": False},
    ],
)
def test_round_arms_are_accepted(changes):
    config = nbb.parse_config(series([arm(rounds={**ROUNDS, **changes})]), Path("c.yaml"))
    (round_arm,) = config.arms
    for key, value in changes.items():
        assert getattr(round_arm.rounds, key) == value


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rounds": {k: v for k, v in ROUNDS.items() if k != "tau"}},
         "rounds must be null or have exactly the keys"),
        ({"rounds": {**ROUNDS, "round_dir": "/kaggle/r"}},
         "rounds must be null or have exactly the keys"),
        ({"rounds": "one"}, "rounds must be null or have exactly the keys"),
        ({"holdout": None}, "rounds need the arm's holdout"),
        ({"holdout": {**HOLDOUT, "in_test_prompts": True}}, "rounds need holdout in_test_prompts"),
        ({"rounds": {**ROUNDS, "max_rounds": 3}}, "rounds max_rounds must be 1 to 2"),
        ({"rounds": {**ROUNDS, "max_rounds": 0}}, "rounds max_rounds must be 1 to 2"),
        ({"rounds": {**ROUNDS, "max_rounds": 1.0}}, "rounds max_rounds must be 1 to 2"),
        ({"rounds": {**ROUNDS, "tau": 1.5}}, "rounds tau must be a number from 0 to 1"),
        ({"rounds": {**ROUNDS, "tau": True}}, "rounds tau must be a number from 0 to 1"),
        ({"rounds": {**ROUNDS, "tau": "9/16"}}, "rounds tau must be a number from 0 to 1"),
        ({"rounds": {**ROUNDS, "continue_steps": 0}}, "rounds continue_steps must be 1 to 128"),
        ({"rounds": {**ROUNDS, "continue_steps": 129}}, "rounds continue_steps must be 1 to 128"),
        ({"rounds": {**ROUNDS, "final_in_prompts": "no"}}, "final_in_prompts must be true or"),
        ({"rounds": {**ROUNDS, "governor": "governor.yaml"}}, "governor must be a .yaml file"),
        ({"rounds": {**ROUNDS, "governor": "configs/../x.yaml"}}, "governor must be a .yaml file"),
        ({"rounds": {**ROUNDS, "governor": "configs/governor.json"}},
         "governor must be a .yaml file"),
    ],
)  # fmt: skip
def test_round_arm_refusals(changes, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(series([arm(**changes)]), Path("c.yaml"))
    assert message in str(refused.value)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"- 1\n- 2\n", "must hold a mapping"),
        (b"ops: {}\n", "a governor config needs margin, candidates_per_batch, deadline"),
        (yaml.safe_dump({**GOVERNOR, "decode_policy": {"mode": "trade"}}).encode(),
         "decode_policy mode must be 'reference'"),
        (yaml.safe_dump({**GOVERNOR, "decode_policy": "trade"}).encode(),
         "decode_policy mode must be 'reference'"),
        (yaml.safe_dump({**GOVERNOR, "margin": float("nan")}).encode(), "the solver JSON cannot"),
        (b"margin: [unclosed\n", "is not valid YAML"),
        (b"# TODO: fill in\nops: {}\n", "still has TODO"),
        (b"\xff\xfe", "is not UTF-8"),
    ],
)  # fmt: skip
def test_governor_config_refusals(data, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_governor(data, "configs/governor.yaml")
    assert message in str(refused.value)


def test_resolve_rounds_reads_each_round_arms_governor_from_the_given_reader():
    reads = []
    config = parsed()

    def read(name):
        reads.append(name)
        return (REPO / name).read_bytes()

    assert nbb.resolve_rounds(config, read).arms[0].rounds.governor_config == GOVERNOR
    assert reads == ["configs/governor.yaml"]
    plain = parsed(P21_CONFIG)
    assert nbb.resolve_rounds(plain, read) == plain and reads == ["configs/governor.yaml"]

    def missing(name):
        raise FileNotFoundError(name)

    with pytest.raises(nbb.BuildError, match="cannot read configs/governor.yaml"):
        nbb.resolve_rounds(config, missing)
    with pytest.raises(nbb.BuildError, match="was not read"):
        nbb.config_cell_source(config, {"tag": "t", "commit": "0" * 40, "config": CONFIG})


def test_the_first_cell_and_the_config_cell_describe_a_round_arm():
    config = resolved(parsed())
    build = {"tag": "t", "commit": "0" * 40, "config": CONFIG}
    note = nbb.config_note(config, build)
    assert f"Arm `{ARM}` runs the refinement rounds (task P2.3, ADR-015)" in note
    assert nbb.rounds_text(ROUNDS) in note and "`configs/governor.yaml` of this commit" in note
    cell = nbb.config_cell_source(config, build)
    assert all(line in cell for line in nbb.ROUNDS_COMMENT)
    plain = nbb.config_cell_source(parsed(P21_CONFIG), build)
    assert not any(line in plain for line in nbb.ROUNDS_COMMENT) and "'rounds'" not in plain


@pytest.mark.parametrize(
    ("changes", "text"),
    [
        ({}, "rounds: at most 1 per puzzle after the gate, promotion at agreement 0.5625, the "
             "adapter continued 32 steps and restored when the gate fails, the final decode "
             "with the held-out pair back in its prompts"),
        ({"max_rounds": 2, "tau": 0.75, "continue_steps": 64, "final_in_prompts": False},
         "rounds: at most 2 per puzzle after the gate, promotion at agreement 0.75, the adapter "
         "continued 64 steps and restored when the gate fails, the final decode without the "
         "held-out pair"),
    ],
)  # fmt: skip
def test_rounds_text_states_the_settings(changes, text):
    assert nbb.rounds_text({**ROUNDS, **changes}) == text


# The build, from a throwaway repository holding the working tree, as test_build_submission's.


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    if not BUILD_TESTS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = BUILD_TESTS.snapshot(tmp_path_factory.mktemp("repo"))
    first = bs.build("P2.3-01", repo / CONFIG, repo)
    files = BUILD_TESTS.read_files(first)
    again = bs.build("P2.3-01", repo / CONFIG, repo)
    return repo, again, files


def test_the_p2_3_01_build_is_a5s_folder_rebuilt_byte_for_byte(build):
    _, folder, first = build
    files = BUILD_TESTS.read_files(folder)
    assert set(files) == BUILD_TESTS.FILES and files == first


def test_the_p2_3_01_build_passes_the_gateway_and_is_not_for_submission(build):
    _, folder, _ = build
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == "alyahya737/arc26-refine-rounds" and meta["machine_shape"] == "NvidiaL4"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder((folder / "arc26-code").resolve())
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | no, not for submission: its notebook is alyahya737/arc26-refine" in (
        manifest
    )
    row = next(line for line in manifest.splitlines() if line.startswith("| arms "))
    assert f"`{ARM}` (r 256: q_proj" in row and nbb.rounds_text(ROUNDS) in row
    digest = BUILD_TESTS.sha256((REPO / "configs" / "governor.yaml").read_bytes())
    assert f"governor `configs/governor.yaml` (sha256 `{digest}`)" in row


def test_the_p2_3_01_notebook_embeds_the_commits_governor_config(build):
    _, folder, _ = build
    code = BUILD_TESTS.code(folder)
    namespace = {}
    exec(code["arc26-02-config"], namespace)
    (entry,) = namespace["ARMS"]
    assert entry["rounds"]["governor"] == GOVERNOR and entry["name"] == ARM
    assert "def round_dir(arm):" in code["arc26-04-helpers"]
    assert "scope[\"holdout\"][\"rounds\"]" in code["arc26-12-run"]


# The notebook on the simulated Kaggle tree.


def round_notebook(dest, wheel, **changes):
    path = REPO / CONFIG
    config = resolved(nbb.parse_config(path.read_bytes(), path))._replace(**changes)
    info = nbb.build_info(
        config,
        tag="local-test",
        commit=LOCAL.COMMIT,
        config_file=CONFIG,
        config_sha256=LOCAL.sha256(path),
        wheel=wheel.name,
        wheel_sha256=LOCAL.sha256(wheel),
    )
    out = dest / "rounds.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_notebook(config, info, nbb.wheel_member(wheel))))
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    dest = tmp_path_factory.mktemp("build")
    wheel = LOCAL.make_wheel(dest)
    return LOCAL.Built(wheel, {"rounds": round_notebook(dest, wheel)})


def test_the_p2_3_01_notebook_passes_the_round_arm_to_the_split_solver(built, tmp_path):
    """In development the split solver starts with DEV4, four workers, the arm's folders, and its
    hold-out settings holding the rounds with the round folder and the governor config; RunConfig
    accepted them (arc26.loop.Rounds), and the launch stops only at the missing GPU stack."""
    root = LOCAL.make_tree(tmp_path, built.wheel)
    run = LOCAL.simulate(tmp_path, built.notebooks["rounds"], root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and LOCAL.solver_starts(run) == []
    (start,) = LOCAL.split_starts(run)
    config = json.loads(start["argv"][5])
    logs = root / "working" / "run_logs" / ARM
    assert config["holdout"] == {
        **HOLDOUT,
        "gate_dir": str(root / "gate_outputs" / ARM),
        "log_dir": str(logs / "gate"),
        "rounds": {
            "round_dir": str(root / "round_outputs" / ARM), "governor": GOVERNOR,
            "max_rounds": 1, "tau": 0.5625, "continue_steps": 32, "final_in_prompts": True,
        },
    }  # fmt: skip
    assert config["subset"] == list(DEV4) and config["workers"] == 4
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + 18000
    assert start["env"] == LOCAL.LAUNCH_ENV and LOCAL.GPU_STACK_MISSING.search(run.output)
    (record,) = run.record["STAGES"]["run"]["arms"]
    assert record["rounds"] == {k: v for k, v in ROUNDS.items() if k != "governor"}
    assert record["status"] == "ran" and record["holdout"] == HOLDOUT
    assert nbb.rounds_text(ROUNDS) in run.output
    assert f"round files in {root / 'round_outputs' / ARM}" in run.output
    assert LOCAL.submit.validate(LOCAL.submission(run), LOCAL.standin_challenges()) == []
    LOCAL.assert_kernel_never_imported_torch(run)


def test_a_competition_rerun_of_the_round_build_runs_the_reference_settings_once(built, tmp_path):
    root = LOCAL.make_tree(tmp_path, built.wheel)
    run = LOCAL.simulate(tmp_path, built.notebooks["rounds"], root, competition=True, gpus=4)
    assert run.record["error"] is None, run.output
    (start,) = LOCAL.split_starts(run)
    config = json.loads(start["argv"][5])
    assert "holdout" not in config and config["outputs_dir"] == str(root / "inference_outputs")
    assert not (root / "round_outputs").exists() and not (root / "gate_outputs").exists()


def test_the_run_cell_removes_the_round_files_and_records_of_an_earlier_launch(
    built, tmp_path, monkeypatch
):
    namespace, cell, calls = LOCAL.run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, notebook="rounds"
    )
    rounds = tmp_path / "round_outputs" / ARM / "36a08778" / "round1"
    gate_log = tmp_path / "working" / "run_logs" / ARM / "gate"
    rounds.mkdir(parents=True)
    gate_log.mkdir(parents=True)
    prefixes = ("gate", "plan", "promote", "round", "decode")
    stale = [rounds / "36a08778_1.permute0123456789"]
    stale += [gate_log / f"{prefix}-rank0.jsonl" for prefix in prefixes]
    kept = [gate_log / "notes.txt"]
    for path in stale + kept:
        path.write_text("x")
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, _, _, _),) = calls
    assert fields["holdout"]["rounds"]["round_dir"] == str(tmp_path / "round_outputs" / ARM)
    assert [path.exists() for path in stale] == [False] * len(stale)
    assert [path.exists() for path in kept] == [True]


# The pre-run brief.


def test_the_p2_3_01_brief_header_parses_with_the_gateways_reader():
    text = BRIEF.read_text(encoding="utf-8")
    assert "TODO" not in text
    header = ops.brief_header(text)
    assert header["Run-id"] == "P2.3-01" and header["Accelerator"] == "NvidiaL4"
    assert header["Kaggle notebook"] == parsed().kaggle_notebook
    assert header["Config"] == CONFIG
    assert int(header["Timeout"]) == parsed().solver_kill_s + 900
    assert parsed().kaggle_notebook in ops.briefed_notebooks()
