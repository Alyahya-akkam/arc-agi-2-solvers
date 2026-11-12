"""Round arms with variants in the notebook builder, and runs P2.5-01 and P2.5-02 (task P2.5).

A round arm may name the round variants it offers (``variants``, task P2.5, ADR-016): the builder
checks them with ``arc26.loop.Rounds``' rule, and the notebook passes them inside the arm's round
settings. Run ``P2.5-01`` forces the in-context round and ``P2.5-02`` the gradient round, each in
a notebook of its own built from one commit, so one code dataset version serves both, on
``P1.3-01`` arm 2's scope (rank 16 on q, k, v, o, the orchestrator's change of 3 Oct: P2.6's
scope), which the builder takes as the first arm because it holds a pair out. This file checks
the rule, the text and cells, both configs against ``P1.3-01`` arm 2's scope, ``P2.1-01``'s
hold-out, and ``P2.3-01``'s rounds, both builds (rebuilt byte for byte, the same wheel), the
notebooks on the simulated Kaggle tree of ``tests/test_notebook_local.py``, and both briefs'
headers. Nothing reaches Kaggle.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from arc26 import governor
from arc26.loop import Rounds

REPO = Path(__file__).resolve().parent.parent
CONFIGS = {
    "P2.5-01": "configs/submission_P2.5-01.yaml", "P2.5-02": "configs/submission_P2.5-02.yaml"
}  # fmt: skip
ARMS = {"P2.5-01": "r16-attn-in-context", "P2.5-02": "r16-attn-gradient"}
NOTEBOOKS = {"P2.5-01": "alyahya737/arc26-round-in-context",
             "P2.5-02": "alyahya737/arc26-round-gradient"}  # fmt: skip
VARIANTS = {"P2.5-01": ["in_context"], "P2.5-02": ["gradient"]}
P21_CONFIG = "configs/submission_P2.1-01.yaml"
P13_CONFIG = "configs/submission_P1.3-01.yaml"
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
P23_CONFIG = "configs/submission_P2.3-01.yaml"
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


BUILD_TESTS = load("_in_context_build_submission", REPO / "tests" / "test_build_submission.py")
LOCAL = load("_in_context_notebook_local", REPO / "tests" / "test_notebook_local.py")
nbb, bs, ops = BUILD_TESTS.nbb, BUILD_TESTS.bs, BUILD_TESTS.ops


def parsed(name):
    path = REPO / name
    return nbb.parse_config(path.read_bytes(), path)


def resolved(config):
    return nbb.resolve_rounds(config, lambda name: (REPO / name).read_bytes())


def series(arms):
    """P2.5-01's config with other arms, as text, for parse_config."""
    raw = yaml.safe_load((REPO / CONFIGS["P2.5-01"]).read_text(encoding="utf-8"))
    raw["arms"] = arms
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def arm(rounds):
    return {
        "name": "a", "lora_r": 256, "lora_targets": list(LOCAL.NINE), "reason": "a round arm",
        "holdout": dict(HOLDOUT), "rounds": rounds,
    }  # fmt: skip


# The builder's rule is the loop's.


def test_the_builders_variants_are_the_loops():
    assert nbb.VARIANTS == governor.LADDER
    assert nbb.ROUNDS_OPTIONAL_KEYS == Rounds.OPTIONAL
    assert set(nbb.ROUNDS_KEYS) == set(Rounds.FIELDS) - {"round_dir"}


@pytest.mark.parametrize("variants", [["in_context"], ["gradient"], ["gradient", "in_context"]])
def test_round_arms_with_variants_are_accepted_and_reach_the_loop(variants):
    config = resolved(nbb.parse_config(series([arm({**ROUNDS, "variants": variants})]), Path("c")))
    (round_arm,) = config.arms
    assert round_arm.rounds.variants == tuple(variants)
    settings = round_arm.rounds.settings()
    assert settings["variants"] == variants
    loop_rounds = Rounds.from_mapping({**settings, "round_dir": "/kaggle/round_outputs/a"})
    assert loop_rounds.variants == tuple(variants)
    # A round arm without variants keeps P2.3's settings: no key, Rounds' default.
    plain = resolved(nbb.parse_config(series([arm(dict(ROUNDS))]), Path("c"))).arms[0]
    assert plain.rounds.variants is None and "variants" not in plain.rounds.settings()


@pytest.mark.parametrize(
    "variants",
    [["in_context", "gradient"], ["gradient", "gradient"], [], ["sgd"], "gradient", None, [1]],
)
def test_bad_variants_are_refused(variants):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(series([arm({**ROUNDS, "variants": variants})]), Path("c.yaml"))
    assert "rounds variants must be a list of gradient, in_context" in str(refused.value)


def test_an_unknown_round_key_names_the_optional_one():
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(series([arm({**ROUNDS, "variant": ["gradient"]})]), Path("c.yaml"))
    message = str(refused.value)
    assert "rounds must be null or have exactly the keys" in message
    assert "and optionally variants" in message


def test_a_first_arm_of_another_scope_is_taken_when_it_holds_a_pair_out():
    """submission.json comes from the first arm; a hold-out arm's attempts are a loop
    measurement's whatever its scope, and a series is never fit to submit, so the builder takes
    it first (task P2.5); a plain arm of another scope first is still refused."""
    small = {**arm({**ROUNDS, "variants": ["in_context"]}), "lora_r": 16, "lora_targets": ATTENTION}
    config = resolved(nbb.parse_config(series([small]), Path("c.yaml")))
    (first,) = config.arms
    assert (first.lora_r, first.lora_targets) == (16, tuple(ATTENTION))
    assert nbb.submission_blocker(config) is not None
    plain = {k: v for k, v in small.items() if k not in ("holdout", "rounds")}
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(series([plain]), Path("c.yaml"))
    assert "the first arm must be the notebook configuration" in str(refused.value)
    assert "or hold a pair out" in str(refused.value)
    build = {"tag": "t", "commit": "0" * 40, "config": CONFIGS["P2.5-01"]}
    assert "comes from the first, `a`, which holds a pair out (task P2.5)" in nbb.config_note(
        config, build
    )
    cell = nbb.config_cell_source(config, build)
    assert all(line in cell for line in nbb.SERIES_HOLDOUT_FIRST_COMMENT)
    assert not any(line in cell for line in nbb.SERIES_FIRST_COMMENT)
    # A series led by the notebook configuration keeps its lines.
    old = nbb.config_cell_source(resolved(parsed(P23_CONFIG)), build)
    assert all(line in old for line in nbb.SERIES_FIRST_COMMENT)
    assert "comes from the first, the notebook configuration" in nbb.config_note(
        resolved(parsed(P23_CONFIG)), build
    )


@pytest.mark.parametrize(
    ("variants", "how"),
    [
        (None, "the adapter continued 32 steps and restored when the gate fails"),
        (["gradient"], "the adapter continued 32 steps and restored when the gate fails"),
        (["in_context"], "in context: the promoted pairs join the prompts, no fit, and only the "
                         "pairs are rolled back when the gate fails"),
        (["gradient", "in_context"], "the adapter continued 32 steps and restored when the gate "
                                     "fails, or, when that does not fit the time left, in "
                                     "context: the promoted pairs join the prompts, no fit, and "
                                     "only the pairs are rolled back when the gate fails"),
    ],
)  # fmt: skip
def test_rounds_text_states_the_variants(variants, how):
    settings = {**ROUNDS, "variants": variants}
    assert nbb.rounds_text(settings) == (
        f"rounds: at most 1 per puzzle after the gate, promotion at agreement 0.5625, {how}, the "
        "final decode with the held-out pair back in its prompts"
    )


def test_the_config_cell_and_the_first_cell_describe_the_variants():
    build = {"tag": "t", "commit": "0" * 40, "config": CONFIGS["P2.5-01"]}
    config = resolved(parsed(CONFIGS["P2.5-01"]))
    cell = nbb.config_cell_source(config, build)
    assert all(line in cell for line in nbb.ROUNDS_COMMENT + nbb.VARIANTS_COMMENT)
    namespace = {}
    exec(cell, namespace)
    (entry,) = namespace["ARMS"]
    assert entry["rounds"]["variants"] == ["in_context"] and entry["rounds"]["governor"] == GOVERNOR
    note = nbb.config_note(config, build)
    assert "runs the refinement rounds (tasks P2.3 and P2.5, ADR-015 and ADR-016)" in note
    # P2.3-01's build names no variants: its cells keep P2.3's lines.
    old = resolved(parsed(P23_CONFIG))
    assert not any(line in nbb.config_cell_source(old, build) for line in nbb.VARIANTS_COMMENT)
    assert "(task P2.3, ADR-015)" in nbb.config_note(old, build)


# Runs P2.5-01 and P2.5-02: P2.1-01's arm, P2.3-01's round, one variant each.


@pytest.mark.parametrize("run", ["P2.5-01", "P2.5-02"])
def test_each_p2_5_config_is_r16_attn_with_p2_1_01s_hold_out_and_one_round_of_its_variant(run):
    """P1.3-01 arm 2's scope (the orchestrator's change, 3 Oct: the scope P2.6 runs), P2.1-01's
    hold-out, P2.3-01's rounds with the variant, DEV40, the stack of every build."""
    config, before, rounds = parsed(CONFIGS[run]), parsed(P21_CONFIG), parsed(P23_CONFIG)
    scope = next(a for a in parsed(P13_CONFIG).arms if a.name == "r16-attn")
    assert config.kaggle_notebook == NOTEBOOKS[run]
    assert (config.accelerator, config.workers, config.hash_seed) == ("NvidiaL4", 4, 0)
    assert config.subset == before.subset and len(config.subset) == 40
    assert config.run_solver is True and config.solver == "split"
    assert (config.solver_end_s, config.solver_kill_s) == (41400, 42300)
    assert config.solver_end_s == GOVERNOR["deadline"]["end_s"]
    for key in BUILD_TESTS.STACK:
        assert getattr(config, key) == getattr(before, key)
    (round_arm,) = config.arms
    (gate_arm,) = before.arms
    (p23_arm,) = rounds.arms
    assert round_arm.name == ARMS[run] and len(round_arm.reason) <= nbb.MAX_REASON
    assert (round_arm.lora_r, round_arm.lora_targets) == (scope.lora_r, scope.lora_targets)
    assert (scope.lora_r, scope.lora_targets) == (16, tuple(ATTENTION))
    assert round_arm.holdout == gate_arm.holdout == nbb.ArmHoldOut("last", True, False)
    assert round_arm.rounds == p23_arm.rounds._replace(variants=tuple(VARIANTS[run]))
    blocker = nbb.submission_blocker(config)
    assert blocker.startswith(f"its notebook is {NOTEBOOKS[run]}")


def test_the_two_configs_differ_in_the_notebook_the_arm_and_its_variant_alone():
    first, second = parsed(CONFIGS["P2.5-01"]), parsed(CONFIGS["P2.5-02"])
    assert first._replace(kaggle_notebook="n", arms=None) == second._replace(
        kaggle_notebook="n", arms=None
    )  # fmt: skip
    (one,), (two,) = first.arms, second.arms
    assert one._replace(name="a", reason="r", rounds=None) == two._replace(
        name="a", reason="r", rounds=None
    )  # fmt: skip
    assert one.rounds._replace(variants=None) == two.rounds._replace(variants=None)


@pytest.mark.parametrize("run", ["P2.5-01", "P2.5-02"])
def test_every_p2_5_config_value_has_a_comment(run):
    lines = (REPO / CONFIGS[run]).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"no comment above {line!r}"


# The builds, from a throwaway repository holding the working tree, as test_build_submission's.


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    if not BUILD_TESTS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = BUILD_TESTS.snapshot(tmp_path_factory.mktemp("repo"))
    found = {}
    for run, config in CONFIGS.items():
        first = BUILD_TESTS.read_files(bs.build(run, repo / config, repo))
        again = bs.build(run, repo / config, repo)
        found[run] = (again, first)
    return found


@pytest.mark.parametrize("run", ["P2.5-01", "P2.5-02"])
def test_each_p2_5_build_is_a5s_folder_rebuilt_byte_for_byte(builds, run):
    folder, first = builds[run]
    files = BUILD_TESTS.read_files(folder)
    assert set(files) == BUILD_TESTS.FILES and files == first


def test_both_builds_carry_one_wheel_for_one_code_dataset_version(builds):
    """One commit, one wheel: the code dataset is versioned once, from either build's folder, and
    both notebooks check the same sha256 before installing it."""
    one, two = (BUILD_TESTS.read_files(builds[run][0]) for run in ("P2.5-01", "P2.5-02"))
    wheel = f"arc26-code/{BUILD_TESTS.WHEEL}"
    assert one[wheel] == two[wheel]
    assert one["arc26-code/dataset-metadata.json"] == two["arc26-code/dataset-metadata.json"]
    assert one["kaggle_submission.ipynb"] != two["kaggle_submission.ipynb"]
    digest = BUILD_TESTS.sha256(one[wheel])
    for folder, _ in builds.values():
        assert digest in BUILD_TESTS.code(folder)["arc26-02-config"]


@pytest.mark.parametrize("run", ["P2.5-01", "P2.5-02"])
def test_each_p2_5_build_passes_the_gateway_and_is_not_for_submission(builds, run):
    folder, _ = builds[run]
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == NOTEBOOKS[run] and meta["machine_shape"] == "NvidiaL4"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder((folder / "arc26-code").resolve())
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert f"| fit to submit | no, not for submission: its notebook is {NOTEBOOKS[run]}" in manifest
    row = next(line for line in manifest.splitlines() if line.startswith("| arms "))
    assert f"`{ARMS[run]}` (r 16: q_proj, k_proj, v_proj, o_proj;" in row
    assert nbb.rounds_text({**ROUNDS, "variants": VARIANTS[run]}) in row


# The notebooks on the simulated Kaggle tree.


def p25_notebook(dest, wheel, run):
    path = REPO / CONFIGS[run]
    config = resolved(nbb.parse_config(path.read_bytes(), path))
    info = nbb.build_info(
        config,
        tag=f"local-{run}",
        commit=LOCAL.COMMIT,
        config_file=CONFIGS[run],
        config_sha256=LOCAL.sha256(path),
        wheel=wheel.name,
        wheel_sha256=LOCAL.sha256(wheel),
    )
    out = dest / f"{run}.ipynb"
    out.write_bytes(nbb.notebook_bytes(nbb.build_notebook(config, info, nbb.wheel_member(wheel))))
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    dest = tmp_path_factory.mktemp("build")
    wheel = LOCAL.make_wheel(dest)
    return LOCAL.Built(wheel, {run: p25_notebook(dest, wheel, run) for run in CONFIGS})


@pytest.mark.parametrize("run", ["P2.5-01", "P2.5-02"])
def test_each_p2_5_notebook_passes_its_round_arm_to_the_split_solver(built, tmp_path, run):
    """The split solver starts with DEV40, four workers, the arm's folders, and its hold-out
    settings holding the rounds: P2.5-01's with variants in_context, P2.5-02's without the key
    (the gradient round is Rounds' default, which the JSON leaves out, as P2.3-01's)."""
    root = LOCAL.make_tree(tmp_path, built.wheel)
    notebook = built.notebooks[run]
    simulated = LOCAL.simulate(tmp_path, notebook, root, gpus=4)
    assert simulated.record["error"] is None, simulated.output
    assert simulated.record["PROBLEMS"] == [] and LOCAL.solver_starts(simulated) == []
    (start,) = LOCAL.split_starts(simulated)
    config = json.loads(start["argv"][5])
    rounds = {
        "round_dir": str(root / "round_outputs" / ARMS[run]), "governor": GOVERNOR,
        "max_rounds": 1, "tau": 0.5625, "continue_steps": 32, "final_in_prompts": True,
    }  # fmt: skip
    if run == "P2.5-01":
        rounds["variants"] = ["in_context"]
    logs = root / "working" / "run_logs" / ARMS[run]
    assert config["holdout"] == {
        **HOLDOUT, "gate_dir": str(root / "gate_outputs" / ARMS[run]),
        "log_dir": str(logs / "gate"), "rounds": rounds,
    }  # fmt: skip
    assert len(config["subset"]) == 40 and config["workers"] == 4
    assert (config["lora_r"], config["lora_targets"]) == (16, ATTENTION)
    assert float(start["argv"][3]) == simulated.record["NOTEBOOK_START"] + 41400
    (record,) = simulated.record["STAGES"]["run"]["arms"]
    assert record["rounds"]["variants"] == VARIANTS[run] and record["status"] == "ran"
    assert nbb.rounds_text({**ROUNDS, "variants": VARIANTS[run]}) in simulated.output
    LOCAL.assert_kernel_never_imported_torch(simulated)


def test_the_run_cell_removes_the_context_records_of_an_earlier_launch(
    built, tmp_path, monkeypatch
):
    namespace, cell, calls = LOCAL.run_cell_namespace(
        built, tmp_path, monkeypatch, lambda: 0, notebook="P2.5-01"
    )
    gate_log = tmp_path / "working" / "run_logs" / ARMS["P2.5-01"] / "gate"
    gate_log.mkdir(parents=True)
    stale = [gate_log / f"{prefix}-rank1.jsonl" for prefix in ("context", "plan", "round")]
    kept = gate_log / "notes.txt"
    for path in [*stale, kept]:
        path.write_text("x")
    exec(compile(cell, "run", "exec"), namespace)
    ((fields, _, _, _),) = calls
    assert fields["holdout"]["rounds"]["variants"] == ["in_context"]
    assert [path.exists() for path in stale] == [False, False, False] and kept.exists()


# The pre-run briefs.


@pytest.mark.parametrize("run", ["P2.5-01", "P2.5-02"])
def test_each_p2_5_brief_header_parses_with_the_gateways_reader(run):
    text = (REPO / "docs" / "prerun" / f"{run}.md").read_text(encoding="utf-8")
    assert "TODO" not in text
    header = ops.brief_header(text)
    config = parsed(CONFIGS[run])
    assert header["Run-id"] == run and header["Accelerator"] == "NvidiaL4"
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Config"] == CONFIGS[run]
    assert int(header["Timeout"]) == config.solver_kill_s + 900 == 43200
    assert config.kaggle_notebook in ops.briefed_notebooks()
