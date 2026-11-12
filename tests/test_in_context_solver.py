"""The in-context round and the length guard on the split solver's worker (task P2.5).

The worker runs on the GPU stand-ins of ``tests/test_rounds_solver.py`` (synthetic tasks under
DEV4's ids, an identity DFS, adapter weights that the stand-in trainer changes in place), whose
driver and helpers this file imports. Three things are checked end to end:

- The defaults leave the split solver's stand-in runs as they were: every run of
  ``tests/test_rounds_solver.py`` reproduces the digests recorded at task P2.3's commit
  (``tests/testdata/p2_3_rounds_stand_in.json``): its candidates, round files, records, timing
  ops, prompts, fits, and weight events.
- An arm with the in-context round forced (``variants: [in_context]``): no fit and no snapshot,
  the adapter's fingerprints unchanged, the gate's prompts holding the pseudo-pair, a kept round
  (version + 1), a rolled-back one (the version alone), and the final decode with the pair.
- The length guard in ``decode_puzzle``'s own prompts at the 30 by 30 worst case: a pseudo-pair
  that the final decode's prompts cannot hold is left out of them, never a real pair, and recorded.

The synthetic tasks are never competition data.
"""

import bz2
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest

from arc26.loop import (
    CANDIDATE_TRAIN,
    FINAL_DECODE,
    FINGERPRINT,
    GATE,
    GATE_DECODE,
    KEPT,
    NOTHING_TO_PROMOTE,
    REDECODE,
    RESTORE,
    ROLLED_BACK,
    ROUND,
    SNAPSHOT,
)

REPO = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((REPO / "tests" / "testdata" / "p2_3_rounds_stand_in.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


ROUNDS = load("_in_context_rounds_solver", REPO / "tests" / "test_rounds_solver.py")
run_rounds_worker, TASKS = ROUNDS.run_rounds_worker, ROUNDS.TASKS
log, by_key, demo_cells, fits = ROUNDS.log, ROUNDS.by_key, ROUNDS.demo_cells, ROUNDS.fits
TIME_KEYS = {"seconds", "at", "deadline", "remaining", "budget", "observed_batch"}


def strip(value):
    """A record without its wall-time fields, which no two runs share."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIME_KEYS}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def decoded(folder):
    """Every candidate file under a folder, decoded, by its relative path."""
    found = {}
    if not folder.is_dir():
        return found
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        with bz2.BZ2File(path) as handle:
            rows = pickle.load(handle)
        found[path.relative_to(folder).as_posix()] = [
            {
                "beam_score": float(row["beam_score"]),
                "score_aug": [float(s) for s in row["score_aug"]],
                "solution": [[int(c) for c in r] for r in row["solution"]],
            }
            for row in rows
        ]
    return found


def normalized(run):
    """What a stand-in run computed, without its wall times: P2.3's digests were taken of this."""
    summary = run.summary
    return {
        "candidates": decoded(run.work / "outputs"),
        "rounds": decoded(run.work / "round_outputs"),
        "logs": strip(summary["logs"]),
        "dfs": [
            [key, phase, weights, hashlib.sha256(text.encode()).hexdigest()[:16]]
            for key, phase, weights, text in summary["dfs"]
        ],
        "trainings": summary["trainings"],
        "events": strip(summary["events"]),
        "timing": [[r["puzzle_id"], r["op"], r["tokens"]] for r in run.records],
    }


def digests(run):
    return {
        name: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for name, value in normalized(run).items()
    }


@pytest.fixture(scope="module")
def default_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("defaults")
    found = {
        name: run_rounds_worker(root / name, rounds, mode, end_in)
        for name, (rounds, mode, end_in) in ROUNDS.RUNS.items()
    }
    found["k2-off"] = run_rounds_worker(root / "k2-off", {}, tasks=ROUNDS.K2_TASKS, k2=False)
    return found


def test_defaults_leave_the_split_solvers_stand_in_runs_byte_identical(default_runs):
    """Every run of P2.3's stand-in suite, with the round settings it states (no variants, the
    length guard on), computes what task P2.3's code computed: the same candidate and round
    files, the same records (no context record), timing ops, DFS prompts, fits, and weights."""
    assert set(default_runs) == set(GOLDEN["runs"])
    for name, run in default_runs.items():
        assert digests(run) == GOLDEN["runs"][name], name
        assert "context-rank0.jsonl" not in run.summary["logs"]


# The in-context round forced, on P2.3's four synthetic tasks.


@pytest.fixture(scope="module")
def in_context_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("in-context")
    return run_rounds_worker(root / "in-context", {"variants": ["in_context"]})


def test_an_in_context_arm_runs_its_rounds_without_a_fit(in_context_run):
    run = in_context_run
    rounds = {key: lines for key, lines in by_key(log(run, "round")).items()}
    outcomes = {key: [line["outcome"] for line in lines] for key, lines in rounds.items()}
    # The identity task keeps its round; the color swap's wrong pair fails its gate; input 1 of
    # 36a08778 and aa4ec2a5's input disagree across views (nothing to promote there).
    assert outcomes == {
        "0934a4d8": [KEPT], "36a08778": [KEPT], "981571dc": [ROLLED_BACK],
        "aa4ec2a5": [NOTHING_TO_PROMOTE],
    }  # fmt: skip
    for key in ("0934a4d8", "36a08778", "981571dc"):
        (record,) = rounds[key]
        assert record["decision"] == "in_context"
        # The adapter is never changed: equal fingerprints, before and after.
        assert record["weights_before"] == record["weights_after"] is not None
    kept = rounds["36a08778"][0]
    assert (kept["version"], kept["next_version"], kept["promoted"]) == (0, 1, [0])
    rolled = rounds["981571dc"][0]
    assert (rolled["version"], rolled["next_version"], rolled["restored"]) == (0, 0, True)
    plans = log(run, "plan")
    assert {plan["decision"] for plan in plans} == {"in_context"}
    assert all(set(plan["estimates"]) == {"in_context"} for plan in plans)
    assert all(plan["shape"]["in_context"]["too_long"] == [] for plan in plans)
    # No candidate fit: the stand-in trainer ran round 0's fits only, and the gradient
    # provider was never asked to adapt or roll back.
    assert [t["steps"] for t in run.summary["trainings"]] == [-1] * len(TASKS)
    assert run.summary["events"] == []


def test_the_in_context_gate_and_final_decode_hold_the_pseudo_pair(in_context_run):
    """The round's gate decodes with the promoted pair in its prompts, the adapter of round 0;
    the final decode adds the held-out pair back, after the real ones, the pseudo-pair last."""
    run = in_context_run
    gate = [c for c in run.summary["dfs"] if c[0] == "36a08778" and c[1] == "round1-gate"]
    tests = [c for c in run.summary["dfs"] if c[0] == "36a08778" and c[1] == "test"]
    assert len(gate) == 16 and {c[2] for c in gate} == {c[2] for c in tests}
    # Round 0's two training pairs (3 and 4 cells) and the pseudo-pair (input 0, 4 cells).
    assert {tuple(demo_cells(c[3])) for c in gate} == {(3, 4, 4)}
    final = [c for c in run.summary["dfs"] if c[0] == "36a08778" and c[1] == "final"]
    assert {tuple(demo_cells(c[3])) for c in final} == {(3, 4, 4, 6)}
    ((decode),) = by_key(log(run, "decode"))["36a08778"]
    assert (decode["name"], decode["prompts"], decode["installed"]) == ("final", "held_out", [1])


def test_the_in_context_rounds_ops_have_no_fit_snapshot_or_restore(in_context_run):
    groups = ROUNDS.STAND_INS.by_puzzle(in_context_run.records)
    kept = [r["op"] for r in groups["36a08778"]]
    assert kept.count(REDECODE) == 1 and kept.count(GATE) == 2 and kept.count(ROUND) == 1
    assert kept.count(GATE_DECODE) == 8 and kept.count(FINAL_DECODE) == 4
    assert kept.count(FINGERPRINT) == 2  # before and after, as the gradient provider digests
    for op in (CANDIDATE_TRAIN, SNAPSHOT, RESTORE):
        assert op not in kept
    rolled = [r["op"] for r in groups["981571dc"]]
    assert RESTORE not in rolled and SNAPSHOT not in rolled and rolled.count(FINGERPRINT) == 2


# The length guard in decode_puzzle's prompts at the 30 by 30 worst case.


def pattern(rows, cols, shift):
    """A grid of every color, so its views differ (the view check)."""
    return [[(r + 2 * c + shift) % 9 + 1 for c in range(cols)] for r in range(rows)]


def identity(rows):
    return {"input": rows, "output": rows}


# Three real 30 by 30 identity pairs (the last held out) and two test inputs: a 29 by 30 grid,
# which the round promotes (identity, every view agrees), and a 1 by 5 grid, which stays open
# (5 cells: the stand-in DFS changes its first cell, so its views disagree). One stand-in token
# per character: a 30 by 30 pair is 1,917 tokens, the 29 by 30 pseudo-pair 1,855, the limit 7,252.
BIG = {
    "0934a4d8": {
        "train": [identity(pattern(30, 30, shift)) for shift in (0, 3, 5)],
        "test": [{"input": pattern(29, 30, 7)}, {"input": [[1, 2, 3, 4, 6]]}],
    },
}


@pytest.fixture(scope="module")
def guard_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("guard")
    return {
        name: run_rounds_worker(root / name, rounds, tasks=BIG)
        for name, rounds in (("gradient", {}), ("in-context", {"variants": ["in_context"]}))
    }


@pytest.mark.parametrize("name", ["gradient", "in-context"])
def test_the_final_decode_leaves_out_the_pseudo_pair_it_cannot_hold(guard_runs, name):
    """The round's gate (two real pairs, the pseudo-pair, the held-out input) fits, so it holds
    the pseudo-pair; the final decode's prompts with the held-out pair back would not, so they
    hold the three real pairs only, and a context record says so. Without the guard the
    reference's cut would drop each view's first demonstration, a real pair in some views."""
    run = guard_runs[name]
    ((record),) = log(run, "round")
    assert record["outcome"] == KEPT and record["promoted"] == [0]
    gate = [c for c in run.summary["dfs"] if c[1] == "round1-gate"]
    assert gate and {tuple(demo_cells(c[3])) for c in gate} == {(870, 900, 900)}
    final = [c for c in run.summary["dfs"] if c[1] == "final"]
    assert len(final) == 16 and {tuple(demo_cells(c[3])) for c in final} == {(900, 900, 900)}
    ((context),) = log(run, "context")
    assert (context["name"], context["prompts"], context["queries"]) == ("final", "held_out", [1])
    assert (context["kept"], context["dropped"], context["limit"]) == ([], [0], 7252)
    assert context["lengths"][0] <= 7252 < context["lengths"][1]
    ((decode),) = log(run, "decode")
    assert (decode["name"], decode["installed"], decode["cut"]) == ("final", [1], False)
    assert "[Rank 0] round 1 for 0934a4d8: the prompts of final left out" in run.output


def test_the_gradient_round_still_trains_on_the_pair_the_final_decode_left_out(guard_runs):
    _, candidate = fits(guard_runs["gradient"], "0934a4d8")
    assert candidate["steps"] == 32 and candidate["after"] != candidate["before"]
    assert [t["steps"] for t in guard_runs["in-context"].summary["trainings"]] == [-1]
