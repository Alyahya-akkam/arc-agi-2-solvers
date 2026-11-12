"""CPU parity check of the moved selection code against run A of P0.1-01 (task P0.1).

Run A's candidates (tests/fixtures/p0_1_reference_candidates_A.json, a byte copy of its
run_logs/candidates.json) are written back as the bz2 pickles its workers wrote, with numpy
solution grids, into a temporary candidate folder. The moved cell 7 (arc26.nvarc.select) then
selects the attempts in competition-rerun mode, so it reads only a stand-in test challenges file
with the DEV4 ids and dummy grids: no competition data and no solutions. transformers is stubbed
(arc_loader imports it for an annotation only). The attempts must equal the P0.1 parity fixture,
which holds run A's DEV4 attempts, exactly.
"""

import bz2
import hashlib
import importlib
import json
import os
import pickle
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from arc26.runner import RunConfig, make_submission

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
PARITY = FIXTURES / "p0_1_reference_dev4.json"
CANDIDATES = FIXTURES / "p0_1_reference_candidates_A.json"
CANDIDATES_SHA256 = "db98f0ca87ddf5298357f52819351e4810087a14a82d9b2c918749d4077e813d"
# Test inputs per DEV4 task: 5 test outputs, as in the parity fixture.
DEV4 = {"0934a4d8": 1, "36a08778": 2, "981571dc": 1, "aa4ec2a5": 1}
MOVED = ("arc26.nvarc.select", "arc26.nvarc.arc_loader", "arc26.nvarc.arc_decoder")


def predictions():
    return json.loads(PARITY.read_text(encoding="utf-8"))["predictions"]


def candidates():
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))


def run_a_file(relative):
    """A file of run A's pulled outputs (runs/P0.1-A, gitignored), looked up in this checkout
    and in the main checkout of this worktree; None when neither has it (CI, fresh clones)."""
    roots = [REPO]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        roots.append(Path(common).parent)
    except (OSError, subprocess.CalledProcessError):
        pass
    for root in roots:
        path = root / "runs" / "P0.1-A" / relative
        if path.is_file():
            return path
    return None


def write_candidates(folder):
    """Write each candidates.json entry as the worker did: a bz2 pickle of a list of
    {beam_score, score_aug, solution} dicts, the solution an integer numpy grid."""
    folder.mkdir(parents=True)
    for name, samples in candidates().items():
        decoded = [
            {
                "beam_score": sample["beam_score"],
                "score_aug": sample["score_aug"],
                "solution": np.array(sample["solution"], dtype=int),
            }
            for sample in samples
        ]
        with bz2.BZ2File(folder / name, "w") as f:
            pickle.dump(decoded, f)
    return folder


def write_stand_in_data(folder):
    """arc-agi_test_challenges.json with the DEV4 ids and their number of test inputs, but
    dummy grids: selection needs only the ids and the test counts."""
    folder.mkdir(parents=True)
    grid = [[0]]
    challenges = {
        task: {"train": [{"input": grid, "output": grid}], "test": [{"input": grid}] * count}
        for task, count in DEV4.items()
    }
    (folder / "arc-agi_test_challenges.json").write_text(json.dumps(challenges), encoding="utf-8")
    return folder


@pytest.fixture
def select(monkeypatch):
    """arc26.nvarc.select imported with a stub transformers; removed again afterwards so no
    other test sees modules bound to the stub."""
    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = type("AutoTokenizer", (), {})
    monkeypatch.setitem(sys.modules, "transformers", stub)
    for name in MOVED:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield importlib.import_module("arc26.nvarc.select")
    for name in MOVED:
        sys.modules.pop(name, None)


@pytest.fixture
def replay(select, tmp_path, monkeypatch):
    """Run the moved cell 7 on run A's candidates; returns the submission it wrote."""
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    data = write_stand_in_data(tmp_path / "data")
    outputs = write_candidates(tmp_path / "inference_outputs")

    def run():
        path = tmp_path / "submission.json"
        select.write_submission(str(data), str(outputs), str(path))
        # Rerun mode read the stand-in challenges only: no other data file exists.
        assert [p.name for p in data.iterdir()] == ["arc-agi_test_challenges.json"]
        return json.loads(path.read_text(encoding="utf-8"))

    run.outputs = outputs
    run.data = data
    return run


def test_replay_reproduces_run_a_dev4_attempts(replay):
    assert replay() == predictions()


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled-0", "shuffled-1", "shuffled-2"])
def test_replay_does_not_depend_on_the_listing_order(replay, monkeypatch, order):
    # load_decoded_results iterates os.listdir, whose order depends on the filesystem; ties in
    # selection would follow it. Run A's candidates have no ties, so every order agrees.
    listdir = os.listdir

    def ordered(path="."):
        names = listdir(path)
        if Path(path) != replay.outputs:
            return names
        names = sorted(names, reverse=order == "reversed")
        if order.startswith("shuffled"):
            random.Random(int(order[-1])).shuffle(names)
        return names

    monkeypatch.setattr(os, "listdir", ordered)
    assert replay() == predictions()


def test_make_submission_reproduces_run_a(replay, tmp_path):
    cfg = RunConfig(
        data_dir=str(replay.data),
        model_dir=str(tmp_path / "model"),
        outputs_dir=str(replay.outputs),
    )
    path = tmp_path / "made.json"
    make_submission(cfg, str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == predictions()


def test_replay_equals_run_a_submission_file(replay):
    path = run_a_file("submission.json")
    if path is None:
        pytest.skip("run A's submission.json is not on this machine (runs/ is gitignored)")
    run_a = json.loads(path.read_text(encoding="utf-8"))
    assert {task: run_a[task] for task in DEV4} == replay()


def test_candidates_fixture_is_the_pinned_byte_copy():
    assert hashlib.sha256(CANDIDATES.read_bytes()).hexdigest() == CANDIDATES_SHA256


def test_candidates_fixture_equals_run_a_file():
    path = run_a_file("run_logs/candidates.json")
    if path is None:
        pytest.skip("run A's candidates.json is not on this machine (runs/ is gitignored)")
    assert path.read_bytes() == CANDIDATES.read_bytes()


def test_candidates_fixture_covers_the_dev4_test_outputs():
    found = candidates()
    outputs = {name.split(".")[0] for name in found}
    assert outputs == {f"{task}_{i}" for task, count in DEV4.items() for i in range(count)}
    for samples in found.values():
        for sample in samples:
            assert set(sample) == {"beam_score", "score_aug", "solution"}
            assert isinstance(sample["beam_score"], float)
            assert all(isinstance(score, float) for score in sample["score_aug"])
            assert np.array(sample["solution"]).ndim == 2
