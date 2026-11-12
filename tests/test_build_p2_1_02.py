"""Run P2.1-02's build (task P2.1-02): ADR-013's overturn arm, P2.1-01's settings with the held-out
pair kept in the test inputs' prompts.

Four layers, each against what this arm is compared with. The config, field by field against
P2.1-01's. The arm's hold-out settings, run through the split solver's worker on the GPU stand-ins
of tests/test_gate.py: the fit and the gate must be P2.1-01's and the test prompts those of a run
without a hold-out (P1.3-01's arm 1). The build (scripts/build_submission.py in a throwaway
repository, as tests/test_build_submission.py builds), against P2.1-01's build of the same
commit. The notebook on the simulated /kaggle tree of tests/test_notebook_local.py, up to the split
solver's launch, whose JSON must carry ``in_test_prompts: true``, also from a code dataset version
shared with a script build. The helpers are imported from those three files; the tests live here,
since parallel branches appending to those files made PR #27 conflict.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

import arc26.submit as submit
from arc26 import loop
from arc26.data import DEV40

REPO = Path(__file__).resolve().parent.parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "tests" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDS = load("_p2_1_02_build_submission", "test_build_submission.py")
LOCAL = load("_p2_1_02_notebook_local", "test_notebook_local.py")
GATE = load("_p2_1_02_gate", "test_gate.py")
nbb, bs, ops = BUILDS.nbb, BUILDS.bs, BUILDS.ops

CONFIG = "configs/submission_P2.1-02.yaml"
PREVIOUS = BUILDS.HOLDOUT_CONFIG
ARM = "r256-proj-emb-holdout-intest"
HOLDOUT = {"rule": "last", "k2": True, "in_test_prompts": True}
HOLDOUT_TEXT = (
    "hold-out: the last real pair of each puzzle with 2 or more, kept out of training and of the "
    "held-out input's prompts (kept in the test inputs' prompts)"
)


def configs():
    return BUILDS.load_config(CONFIG), BUILDS.load_config(PREVIOUS)


# The config


def test_every_config_value_has_a_comment():
    lines = (REPO / CONFIG).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"{CONFIG}: no comment above {line!r}"


def test_the_config_is_p2_1_01s_with_the_held_out_pair_in_the_test_prompts():
    """Every setting but the arm is P2.1-01's: notebook, code dataset, stack, accelerator, DEV40,
    workers, seed, solver, deadlines. The arm differs in its name, its reason, and
    in_test_prompts alone."""
    new, old = configs()
    assert new._replace(arms=None) == old._replace(arms=None)
    (arm,), (before,) = new.arms, old.arms
    assert arm.name == ARM and before.name == "r256-proj-emb-holdout"
    assert arm._replace(name=before.name, reason=before.reason, holdout=before.holdout) == before
    assert before.holdout == nbb.ArmHoldOut("last", True, False)
    assert arm.holdout == before.holdout._replace(in_test_prompts=True) == nbb.ArmHoldOut(**HOLDOUT)
    assert 0 < len(arm.reason) <= nbb.MAX_REASON
    assert new.kaggle_notebook == "alyahya737/arc26-holdout-gate" and new.subset == DEV40
    assert (new.accelerator, new.workers, new.hash_seed, new.solver) == ("NvidiaL4", 4, 0, "split")
    assert (new.solver_end_s, new.solver_kill_s) == (14400, 15300)
    pulled = json.loads((REPO / new.reference_metadata).read_text(encoding="utf-8"))
    # The same notebook and inputs: the gateway's metadata checks run inside kernel_metadata.
    assert bs.kernel_metadata(new, pulled) == bs.kernel_metadata(old, pulled)


def test_the_build_is_not_for_submission_for_p2_1_01s_reasons():
    new, old = configs()
    assert nbb.submission_blocker(new) == nbb.submission_blocker(old)
    assert nbb.submission_blocker(new).startswith(
        "its notebook is alyahya737/arc26-holdout-gate, not alyahya737/arc26-submission"
    )


def test_the_note_states_that_the_pair_stays_in_the_test_prompts():
    new, _ = configs()
    assert nbb.holdout_text(dict(new.arms[0].holdout._asdict())) == HOLDOUT_TEXT
    note = nbb.config_note(new, BUILDS.BUILD_STUB)
    assert f"Arm `{ARM}` runs the held-out gate (task P2.1" in note
    assert f"Its {HOLDOUT_TEXT}.\n" in note


# The arm's settings on the split solver's worker (GPU stand-ins, identity DFS)


@pytest.fixture(scope="module")
def worker_runs(tmp_path_factory):
    """The worker on tests/test_gate.py's three synthetic tasks with this arm's settings, with
    P2.1-01's, and without a hold-out (the setting of P1.3-01's arm 1)."""
    root = tmp_path_factory.mktemp("p2_1_02_worker")
    new, old = configs()
    settings = {
        "intest": dict(new.arms[0].holdout._asdict()),
        "out": dict(old.arms[0].holdout._asdict()),
        "off": None,
    }
    return {name: GATE.run_holdout_worker(root / name, h) for name, h in settings.items()}


def phase(summary, name):
    """The DFS prompts and the rescoring queries of one phase (``test`` or ``gate``), in order."""
    return [text for p, text in summary["prompts"] + summary["score_prompts"] if p == name]


def test_the_fit_is_p2_1_01s_and_never_holds_the_held_out_pair(worker_runs):
    intest, out = (worker_runs[name].summary for name in ("intest", "out"))
    assert intest["train_texts"] == out["train_texts"]
    for key, texts in GATE.by_task(intest["train_texts"]).items():
        assert len(texts) == 128
        for text in texts:
            pairs, query = GATE.turns(text)
            assert len(pairs) == GATE.REAL_PAIRS[key] - 1 and query is None
            assert GATE.HELD_CELLS[key] not in {GATE.cells(g) for pair in pairs for g in pair}


def test_the_gate_is_p2_1_01s(worker_runs):
    """The gate decodes and rescores the held-out input with the other pairs only, as in
    P2.1-01, so its files and verdicts are P2.1-01's (apart from the measured seconds)."""
    intest, out = (worker_runs[name].summary for name in ("intest", "out"))
    gate = phase(intest, "gate")
    assert gate == phase(out, "gate") and len(gate) >= 16 * len(GATE.TASKS)
    for text in gate:
        key = GATE.task_of(text)
        pairs, query = GATE.turns(text)
        assert len(pairs) == GATE.REAL_PAIRS[key] - 1 and GATE.cells(query) == GATE.HELD_CELLS[key]
    assert intest["gate_files"] == out["gate_files"] and intest["gate_files"]

    def timeless(verdicts):
        return [{k: v for k, v in verdict.items() if k != "seconds"} for verdict in verdicts]

    assert timeless(intest["verdicts"]) == timeless(out["verdicts"])
    assert [v["passed"] for v in intest["verdicts"]] == [True, False, True]


def test_the_test_prompts_and_candidates_are_those_of_a_run_without_hold_out(worker_runs):
    """Every test prompt, DFS and rescoring alike, holds all k pairs, as P1.3-01's arm 1 does;
    the stand-in fit changes no weight, so the candidate files match byte for byte."""
    intest, out, off = (worker_runs[name].summary for name in ("intest", "out", "off"))
    tests = phase(intest, "test")
    assert tests == phase(off, "test") and tests != phase(out, "test")
    for text in tests:
        key = GATE.task_of(text)
        assert len(GATE.turns(text)[0]) == GATE.REAL_PAIRS[key]
    assert intest["candidates"] == off["candidates"] and intest["candidates"]
    assert intest["candidates"] != out["candidates"]


def test_the_token_counts_are_the_runs_diagnostics(worker_runs):
    """The run's section 1 diagnostics in miniature: per puzzle, train and gate_decode tokens
    equal P2.1-01's, decode tokens equal the run without hold-out's."""

    def tokens(run, op):
        found = {}
        for record in run.records:
            if record["op"] == op:
                found.setdefault(record["puzzle_id"], []).append(record["tokens"])
        return found

    intest, out, off = (worker_runs[name] for name in ("intest", "out", "off"))
    assert tokens(intest, "train") == tokens(out, "train") != tokens(off, "train")
    assert tokens(intest, loop.GATE_DECODE) == tokens(out, loop.GATE_DECODE)
    assert tokens(intest, "decode") == tokens(off, "decode") != tokens(out, "decode")
    assert set(tokens(intest, loop.GATE_DECODE)) == set(GATE.TASKS)


# The build, against P2.1-01's build of the same commit


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not BUILDS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = BUILDS.snapshot(tmp_path_factory.mktemp("repo"))
    return (
        bs.build("P2.1-02", repo / CONFIG, repo),
        bs.build("P2.1-01", repo / PREVIOUS, repo),
    )


@BUILDS.needs_builder
def test_the_build_passes_the_gateway_and_its_manifest_names_the_arm(built):
    folder, _ = built
    assert set(BUILDS.read_files(folder)) == BUILDS.FILES
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
    assert f"`{ARM}` (r 256: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj" in row
    assert row.endswith(f"embed_tokens, lm_head; {HOLDOUT_TEXT}) |")
    assert "--config configs/submission_P2.1-02.yaml" in manifest


@BUILDS.needs_builder
def test_the_build_shares_p2_1_01s_wheel_metadata_and_cells_but_the_first_two(built):
    new, old = built
    code = (f"arc26-code/{BUILDS.WHEEL}", "arc26-code/dataset-metadata.json")
    for name in (*code, "kernel-metadata.json"):
        assert (new / name).read_bytes() == (old / name).read_bytes()
    BUILDS.config_cell_changes(old, new)  # asserts that only the note and the Config cell differ

    def settings(folder):
        namespace = {}
        exec(BUILDS.code(folder)["arc26-02-config"], namespace)
        return {k: v for k, v in namespace.items() if k.isupper() and k != "NOTEBOOK_START"}

    one, two = settings(old), settings(new)
    assert {k: v for k, v in one.items() if k not in ("ARMS", "BUILD")} == {
        k: v for k, v in two.items() if k not in ("ARMS", "BUILD")
    }
    (arm,), (before,) = two["ARMS"], one["ARMS"]
    assert {**arm, "name": before["name"], "reason": before["reason"]} == {
        **before,
        "holdout": {**before["holdout"], "in_test_prompts": True},
    }
    assert arm["name"] == ARM and arm["holdout"] == HOLDOUT
    changed = {k for k in one["BUILD"] if one["BUILD"][k] != two["BUILD"][k]}
    assert changed == {"tag", "config", "config_sha256"}


# The notebook on the simulated /kaggle tree


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    dest = tmp_path_factory.mktemp("p2_1_02_notebook")
    wheel = LOCAL.make_wheel(dest)
    return wheel, LOCAL.build_notebook(dest, Path(CONFIG).name, wheel, "intest")


def test_the_notebook_passes_in_test_prompts_to_the_split_solver(notebook, tmp_path):
    """The P2.1-02 build as committed: DEV40, four workers, the arm's folders, and its hold-out
    with in_test_prompts true and the gate's folders, in the JSON of python -m arc26.solver,
    which arc26.loop.HoldOut reads back as the worker does; the launch then fails here at the
    GPU stack, and the notebook still writes the submission and the arm's logs."""
    wheel, path = notebook
    root = LOCAL.make_tree(tmp_path, wheel)
    run = LOCAL.simulate(tmp_path, path, root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and LOCAL.solver_starts(run) == []
    (start,) = LOCAL.split_starts(run)
    config = json.loads(start["argv"][5])
    logs = root / "working" / "run_logs" / ARM
    folders = {"gate_dir": str(root / "gate_outputs" / ARM), "log_dir": str(logs / "gate")}
    assert config["holdout"] == {**HOLDOUT, **folders}
    assert loop.HoldOut.from_cfg(config) == loop.HoldOut(**folders, **HOLDOUT)
    assert config["outputs_dir"] == str(root / "inference_outputs" / ARM)
    assert config["timing_dir"] == str(logs / "timing")
    assert (config["lora_r"], config["lora_targets"], config["workers"]) == (256, LOCAL.NINE, 4)
    assert config["subset"] == list(DEV40)
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + 14400
    assert start["env"] == LOCAL.LAUNCH_ENV and LOCAL.GPU_STACK_MISSING.search(run.output)
    (arm,) = run.record["STAGES"]["run"]["arms"]
    assert arm["name"] == ARM and arm["holdout"] == HOLDOUT and arm["status"] == "ran"
    assert HOLDOUT_TEXT in run.output and f"gate verdicts in {logs / 'gate'}" in run.output
    assert submit.validate(LOCAL.submission(run), LOCAL.standin_challenges()) == []
    stages = json.loads((root / "working" / "run_logs" / "stages.json").read_text())
    assert stages["settings"]["arms"][0]["holdout"] == HOLDOUT
    LOCAL.assert_kernel_never_imported_torch(run)


def test_the_notebook_runs_from_a_code_dataset_version_shared_with_a_script_build(
    notebook, tmp_path
):
    """One version of the code dataset can serve this run and a script build of the same wheel
    (P3.6-01): the script and its config lie beside the wheel, and this notebook installs the
    wheel, found by name, and never reads or runs them."""
    wheel, path = notebook
    extras = {"train_hypernet.py": b"raise SystemExit(3)\n", "script_P3.6-01.yaml": b"steps: 1\n"}
    root = LOCAL.make_tree(tmp_path, wheel, extras=extras)
    run = LOCAL.simulate(tmp_path, path, root, gpus=4)
    assert run.record["error"] is None and run.record["PROBLEMS"] == [], run.output
    (install,) = LOCAL.install_calls(run)
    assert install[-1].endswith("/" + LOCAL.WHEEL_NAME)
    (start,) = LOCAL.split_starts(run)
    assert json.loads(start["argv"][5])["holdout"]["in_test_prompts"] is True
    assert not any(name in arg for call in run.calls for arg in call["argv"] for name in extras)


def test_a_competition_rerun_of_the_build_runs_the_reference_settings_once(notebook, tmp_path):
    wheel, path = notebook
    root = LOCAL.make_tree(tmp_path, wheel)
    run = LOCAL.simulate(tmp_path, path, root, competition=True, gpus=4)
    assert run.record["error"] is None, run.output
    (start,) = LOCAL.split_starts(run)
    config = json.loads(start["argv"][5])
    assert "holdout" not in config and config["timing_dir"] is None
    assert config["outputs_dir"] == str(root / "inference_outputs")
    assert (config["lora_r"], config["lora_targets"]) == (256, LOCAL.NINE)
    assert not (root / "gate_outputs").exists()
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]
    assert submit.validate(LOCAL.submission(run), LOCAL.standin_challenges()) == []
