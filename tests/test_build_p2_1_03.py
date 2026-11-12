"""Run P2.1-03's build (task P2.1-03): the notebook configuration on DEV40 without a held-out pair,
on the split solver: P2.1's hold-out-off arm in P2.1-01's code and the split solver's DEV40 parity.

As tests/test_build_p2_1_02.py does for the overturn arm: the config, field by field against
P2.1-01's (and its arm against P1.3-01's arm 1); the arm's settings on the split solver's worker
with the GPU stand-ins of tests/test_gate.py (the reference's per-puzzle body, no gate, the test
prompts of P2.1-02's arm); the build against P2.1-01's build of the same commit; and the notebook
on the simulated /kaggle tree of tests/test_notebook_local.py, whose solver JSON must carry no
``holdout``, also from the code dataset version that holds a script build's files. The helpers
are imported from those files; the tests live here, since parallel branches appending to them
made PR #27 conflict.
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


BUILDS = load("_p2_1_03_build_submission", "test_build_submission.py")
LOCAL = load("_p2_1_03_notebook_local", "test_notebook_local.py")
GATE = load("_p2_1_03_gate", "test_gate.py")
nbb, bs, ops = BUILDS.nbb, BUILDS.bs, BUILDS.ops

CONFIG = "configs/submission_P2.1-03.yaml"
HOLDOUT_CONFIG = BUILDS.HOLDOUT_CONFIG
OVERTURN_CONFIG = "configs/submission_P2.1-02.yaml"
NOTEBOOK = "alyahya737/arc26-split-dev40"
ARM = "r256-proj-emb"
GATE_OPS = {loop.GATE, loop.GATE_DECODE, loop.GATE_SELECTION}


def load_config(name):
    return BUILDS.load_config(name)


# The config


def test_every_config_value_has_a_comment():
    lines = (REPO / CONFIG).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.match(r"[a-z_]+:", line):
            assert lines[index - 1].startswith("#"), f"{CONFIG}: no comment above {line!r}"


def test_the_config_is_p2_1_01s_without_the_hold_out_in_its_own_notebook():
    """Every setting but the notebook and the arm is P2.1-01's: code dataset, stack, NvidiaL4,
    DEV40, workers, seed, the split solver, the deadlines. The arm is P1.3-01's arm 1 (name,
    rank, targets, no hold-out), with its own reason."""
    new, old = load_config(CONFIG), load_config(HOLDOUT_CONFIG)
    series = load_config(BUILDS.SERIES_CONFIG)
    assert new._replace(kaggle_notebook=None, arms=None) == old._replace(
        kaggle_notebook=None, arms=None
    )
    assert new.kaggle_notebook == NOTEBOOK
    others = (BUILDS.GPU_CONFIG, BUILDS.FULL_CONFIG, BUILDS.SERIES_CONFIG, BUILDS.SPLIT_CONFIG)
    assert NOTEBOOK not in {load_config(name).kaggle_notebook for name in (*others, HOLDOUT_CONFIG)}
    (arm,), (before,), first = new.arms, old.arms, series.arms[0]
    assert arm.holdout is None and before.holdout is not None
    assert arm._replace(reason=first.reason) == first and arm.name == ARM
    assert arm._replace(name=before.name, reason=before.reason, holdout=before.holdout) == before
    assert 0 < len(arm.reason) <= nbb.MAX_REASON
    assert new.subset == DEV40 and (new.accelerator, new.workers, new.hash_seed) == (
        "NvidiaL4", 4, 0
    )  # fmt: skip
    assert (new.solver, new.solver_end_s, new.solver_kill_s) == ("split", 14400, 15300)
    pulled = json.loads((REPO / new.reference_metadata).read_text(encoding="utf-8"))
    meta, before_meta = bs.kernel_metadata(new, pulled), bs.kernel_metadata(old, pulled)
    assert meta["id"] == NOTEBOOK and meta["machine_shape"] == "NvidiaL4"
    assert {k: v for k, v in meta.items() if k not in ("id", "title")} == {
        k: v for k, v in before_meta.items() if k not in ("id", "title")
    }


def test_the_build_is_not_for_submission():
    reason = nbb.submission_blocker(load_config(CONFIG))
    assert reason.startswith(
        f"its notebook is {NOTEBOOK}, not alyahya737/arc26-submission, the only notebook"
    )
    assert "it is a series of 1 arm, built to compare LoRA scopes" in reason
    assert "held-out gate" not in reason


def test_the_note_names_no_hold_out():
    note = nbb.config_note(load_config(CONFIG), BUILDS.BUILD_STUB)
    assert f"in this order: `{ARM}`." in note and "held-out gate" not in note
    assert "hold-out: " not in note


# The arm's settings on the split solver's worker (GPU stand-ins, identity DFS)


@pytest.fixture(scope="module")
def worker_runs(tmp_path_factory):
    """The worker on tests/test_gate.py's three synthetic tasks with this arm's settings (no
    hold-out) and with P2.1-02's arm's (the pair held out of the fit, kept in the test prompts)."""
    root = tmp_path_factory.mktemp("p2_1_03_worker")
    (arm,), (overturn,) = load_config(CONFIG).arms, load_config(OVERTURN_CONFIG).arms
    settings = {"off": arm.holdout, "intest": dict(overturn.holdout._asdict())}
    return {name: GATE.run_holdout_worker(root / name, h) for name, h in settings.items()}


def phase(summary, name):
    """The DFS prompts and the rescoring queries of one phase (``test`` or ``gate``), in order."""
    return [text for p, text in summary["prompts"] + summary["score_prompts"] if p == name]


def test_the_worker_runs_the_reference_body_without_a_gate(worker_runs):
    """Every training sample and every test prompt holds all k pairs; no gate decodes, writes
    files or verdicts, or records timing."""
    off = worker_runs["off"]
    for key, texts in GATE.by_task(off.summary["train_texts"]).items():
        assert len(texts) == 128
        for text in texts:
            pairs, query = GATE.turns(text)
            assert len(pairs) == GATE.REAL_PAIRS[key] and query is None
            assert GATE.HELD_CELLS[key] in {GATE.cells(g) for g, _ in pairs}
    tests = phase(off.summary, "test")
    assert tests and all(
        len(GATE.turns(text)[0]) == GATE.REAL_PAIRS[GATE.task_of(text)] for text in tests
    )
    assert phase(off.summary, "gate") == [] and off.summary["gate_files"] == {}
    assert off.summary["verdicts"] == [] and "gate for" not in off.output
    assert not {record["op"] for record in off.records} & GATE_OPS


def test_its_test_prompts_and_candidates_are_p2_1_02s(worker_runs):
    """P2.1-02 keeps the held-out pair in the test prompts, so its test decode is this arm's; the
    stand-in fit changes no weight, so the candidate files match byte for byte. Only the fit
    differs: every pair here, the other pairs there."""
    off, intest = worker_runs["off"], worker_runs["intest"]
    assert phase(off.summary, "test") == phase(intest.summary, "test")
    assert off.summary["candidates"] == intest.summary["candidates"] and off.summary["candidates"]

    def tokens(run, op):
        found = {}
        for record in run.records:
            if record["op"] == op:
                found.setdefault(record["puzzle_id"], []).append(record["tokens"])
        return found

    assert tokens(off, "decode") == tokens(intest, "decode")
    assert tokens(off, "train") != tokens(intest, "train")


# The build, against P2.1-01's build of the same commit


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not BUILDS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = BUILDS.snapshot(tmp_path_factory.mktemp("repo"))
    return (
        bs.build("P2.1-03", repo / CONFIG, repo),
        bs.build("P2.1-01", repo / HOLDOUT_CONFIG, repo),
    )


@BUILDS.needs_builder
def test_the_build_passes_the_gateway_and_its_manifest_names_the_arm(built):
    folder, _ = built
    assert set(BUILDS.read_files(folder)) == BUILDS.FILES
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == NOTEBOOK and meta["machine_shape"] == "NvidiaL4"
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert f"| fit to submit | no, not for submission: its notebook is {NOTEBOOK}" in manifest
    row = next(line for line in manifest.splitlines() if line.startswith("| arms "))
    assert row.endswith(
        f"| `{ARM}` (r 256: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, "
        "embed_tokens, lm_head) |"
    )
    assert "hold-out: " not in manifest and "--config configs/submission_P2.1-03.yaml" in manifest
    assert "| solver (both modes) | `split`: " in manifest


@BUILDS.needs_builder
def test_the_build_shares_p2_1_01s_wheel_and_cells_but_the_first_two(built):
    new, old = built
    for name in (f"arc26-code/{BUILDS.WHEEL}", "arc26-code/dataset-metadata.json"):
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
    assert "holdout" not in arm and arm["name"] == ARM
    assert {**before, "name": ARM, "reason": arm["reason"]} == {**arm, "holdout": before["holdout"]}
    changed = {k for k in one["BUILD"] if one["BUILD"][k] != two["BUILD"][k]}
    assert changed == {"tag", "config", "config_sha256", "kaggle_notebook"}
    assert not any(line in BUILDS.code(new)["arc26-02-config"] for line in nbb.HOLDOUT_COMMENT)


# The notebook on the simulated /kaggle tree


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    dest = tmp_path_factory.mktemp("p2_1_03_notebook")
    wheel = LOCAL.make_wheel(dest)
    return wheel, LOCAL.build_notebook(dest, Path(CONFIG).name, wheel, "split-dev40")


@pytest.mark.parametrize("shared", [False, True], ids=["own", "shared-with-a-script-build"])
def test_the_notebook_starts_the_split_solver_without_a_hold_out(notebook, tmp_path, shared):
    """The P2.1-03 build as committed: DEV40, four workers, the arm's folders, and no holdout in
    the JSON of python -m arc26.solver, so arc26.loop.HoldOut reads none and the worker runs the
    reference's body; the launch then fails here at the GPU stack, and the notebook still writes
    the submission and the arm's logs. Code dataset version 10 also holds P3.6-01's script and
    config beside the wheel; the notebook installs the wheel by name and never reads them."""
    wheel, path = notebook
    extras = {"train_hypernet.py": b"raise SystemExit(3)\n", "script_P3.6-01.yaml": b"steps: 1\n"}
    root = LOCAL.make_tree(tmp_path, wheel, extras=extras if shared else None)
    run = LOCAL.simulate(tmp_path, path, root, gpus=4)
    assert run.record["error"] is None, run.output
    assert run.record["PROBLEMS"] == [] and LOCAL.solver_starts(run) == []
    (install,) = LOCAL.install_calls(run)
    assert install[-1].endswith("/" + LOCAL.WHEEL_NAME)
    (start,) = LOCAL.split_starts(run)
    config = json.loads(start["argv"][5])
    logs = root / "working" / "run_logs" / ARM
    assert "holdout" not in config and loop.HoldOut.from_cfg(config) is None
    assert config["outputs_dir"] == str(root / "inference_outputs" / ARM)
    assert config["timing_dir"] == str(logs / "timing")
    assert (config["lora_r"], config["lora_targets"], config["workers"]) == (256, LOCAL.NINE, 4)
    assert config["subset"] == list(DEV40)
    assert float(start["argv"][3]) == run.record["NOTEBOOK_START"] + 14400
    assert start["env"] == LOCAL.LAUNCH_ENV and LOCAL.GPU_STACK_MISSING.search(run.output)
    (arm,) = run.record["STAGES"]["run"]["arms"]
    assert arm["name"] == ARM and "holdout" not in arm and arm["status"] == "ran"
    assert "hold-out: " not in run.output and not (root / "gate_outputs").exists()
    assert submit.validate(LOCAL.submission(run), LOCAL.standin_challenges()) == []
    stages = json.loads((root / "working" / "run_logs" / "stages.json").read_text())
    assert "holdout" not in stages["settings"]["arms"][0]
    assert not any(name in arg for call in run.calls for arg in call["argv"] for name in extras)
    LOCAL.assert_kernel_never_imported_torch(run)


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
    assert sorted(p.name for p in (root / "working").iterdir()) == ["submission.json"]
    assert submit.validate(LOCAL.submission(run), LOCAL.standin_challenges()) == []
