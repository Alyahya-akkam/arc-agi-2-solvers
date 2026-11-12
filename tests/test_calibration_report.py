"""Calibration metrics, the reproduction, the oracle gap, and the rules of
scripts/calibration_report.py on hand-computed cases and a tiny synthetic run (P2.8)."""

import csv
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

from arc26.data import DEV4, DEV40

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "calibration_report", REPO / "scripts" / "calibration_report.py"
)
cal = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cal  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(cal)

T0, T1, T2, T3 = DEV4
G1, G2 = [[1, 2], [3, 4]], [[2, 2], [3, 4]]
H1, H2, H3 = [[5]], [[6]], [[7, 7, 7]]
J1, J2 = [[1, 1], [1, 1]], [[1, 1], [1, 2]]
K1, K2, K3 = [[3]], [[4]], [[8, 8], [8, 8], [8, 8]]


def sample(grid, aug, beam=0.1):
    return {"beam_score": beam, "score_aug": [aug] * 8, "solution": grid}


# Per view, its candidates; s(g) = |C_g| - mean of the candidates' mean rescoring NLL.
CANDIDATES = {
    # T0_0: G1 in two views (s = 2 - 0.6 = 1.4), G2 in one (s = 0.8); G1 correct, solved.
    f"{T0}_0.rot90.ex0": [sample(G1, 0.5)],
    f"{T0}_0.transpose.ex0": [sample(G1, 0.7), sample(G2, 0.2)],
    # T0_1: three grids in one view each (s = 0.9, 0.8, 0.7); H3 correct, ranked 3rd.
    f"{T0}_1.rot90.ex0": [sample(H1, 0.1)],
    f"{T0}_1.flip.ex0": [sample(H2, 0.2)],
    f"{T0}_1.transpose.ex0": [sample(H3, 0.3)],
    # T1_0: J2 correct, ranked 2nd (s = 1.0 against 0.5), solved.
    f"{T1}_0.rot90.ex0": [sample(J1, 0.0), sample(J2, 0.5)],
    # T2_0: no candidate. T3_0: K3 in three views with high NLL (s = 3 - 3 = 0), ranked 3rd
    # behind K1 (s = 1.0) and K2 (s = 0.5): agreement first or the product would submit it.
    f"{T3}_0.rot90.ex0": [sample(K1, 0.0), sample(K3, 3.0)],
    f"{T3}_0.flip.ex0": [sample(K2, 0.5), sample(K3, 3.0)],
    f"{T3}_0.transpose.ex0": [sample(K3, 3.0)],
}
CORRECT = {f"{T0}_0": G1, f"{T0}_1": H3, f"{T1}_0": J2, f"{T3}_0": K3}
SUBMISSION = {
    T0: [{"attempt_1": G1, "attempt_2": G2}, {"attempt_1": H1, "attempt_2": H2}],
    T1: [{"attempt_1": J1, "attempt_2": J2}],
    T2: [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    T3: [{"attempt_1": K1, "attempt_2": K2}],
}


def selection_log(reload="1.5", solved=None):
    lines = []
    for view in sorted(CANDIDATES):
        for i, s in enumerate(CANDIDATES[view]):
            grid = s["solution"]
            if grid == CORRECT.get(view.split(".", 1)[0]):
                lines.append(
                    f"ALL_CORRECT: 0.10000 - 1.00000 {len(grid)}x{len(grid[0])}   "
                    f"[{view}.out{i}]"
                )
    lines += [
        str(solved if solved is not None else {f"{T0}_0", f"{T1}_0"}),
        " acc:  1.5/3 ('score_kgmon')",
        f"*** Reload score: {reload}",
    ]
    return "\n".join(lines) + "\n"


def write_arm(folder, submission=SUBMISSION, reload="1.5"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "candidates.json").write_text(json.dumps(CANDIDATES), encoding="utf-8")
    (folder / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
    (folder / "selection.log").write_text(selection_log(reload), encoding="utf-8")
    return folder


@pytest.fixture
def run(tmp_path):
    return cal.reproduce("synthetic", write_arm(tmp_path / "arm"))


def by_key(run, key):
    return next(o for o in run.outputs if o.key == key)


# Metrics on hand-computed cases.


def test_ece_brier_auroc_hand_computed():
    p, c = [0.2, 0.2, 0.8, 0.8], [False, True, True, True]
    # Bins {0.2, 0.2} and {0.8, 0.8}: 2/4 * |0.5 - 0.2| + 2/4 * |1 - 0.8| = 0.15 + 0.10.
    assert cal.ece(p, c, bins=2) == pytest.approx(0.25)
    assert cal.brier(p, c) == pytest.approx((0.04 + 0.64 + 0.04 + 0.04) / 4)
    # Positives 0.2, 0.8, 0.8 against the negative 0.2: a tie (1/2), then two wins.
    assert cal.auroc(p, c) == pytest.approx(2.5 / 3)


def test_ece_weights_bins_by_their_counts():
    # Bins {0.1, 0.1, 0.1} and {0.9}, all wrong: 3/4 * 0.1 + 1/4 * 0.9 = 0.3; the unweighted
    # mean of the two bin errors would be 0.5.
    assert cal.ece([0.1, 0.1, 0.1, 0.9], [False] * 4, bins=2) == pytest.approx(0.3)
    # Bins {0.1 x 3} with one right (error 7/30) and {0.9} right (error 0.1): weighted 0.2,
    # unweighted 1/6.
    assert cal.ece([0.1, 0.1, 0.1, 0.9], [True, False, False, True], bins=2) == pytest.approx(0.2)


def test_reference_ece_of_a_perfect_predictor():
    # Certain predictions are their own labels: the reference is exactly 0.
    ref = cal.reference_ece([0.0, 1.0, 1.0, 0.0], observed=0.0, reps=50)
    assert (ref.mean, ref.p95, ref.exceed) == (0.0, 0.0, 1.0)
    # p = 1/2 on four items, one bin: ECE = |k/4 - 1/2| with k ~ Binomial(4, 1/2), whose mean
    # is 3/16 and which is 1/2 with probability 2/16, above the 95th percentile's 5%.
    ref = cal.reference_ece([0.5] * 4, observed=0.5, reps=4000, seed=1)
    assert ref.mean == pytest.approx(3 / 16, abs=0.015)
    assert ref.p95 == 0.5
    assert ref.exceed == pytest.approx(2 / 16, abs=0.02)
    assert ref == cal.reference_ece([0.5] * 4, observed=0.5, reps=4000, seed=1)
    with pytest.raises(ValueError):
        cal.reference_ece([])


def test_calibrate_carries_the_reference():
    items = [("a", 0.2, False), ("b", 0.8, True), ("c", 0.6, True), ("d", 0.1, False)]
    out = cal.calibrate(items, bins=2, reps=100, seed=3)
    assert out.reference == cal.reference_ece([0.2, 0.8, 0.6, 0.1], out.ece, 2, 100, 3)


def test_metrics_on_extreme_cases():
    assert cal.ece([0.0, 0.0, 1.0, 1.0], [False, False, True, True], bins=2) == 0.0
    assert cal.brier([1.0, 0.0], [False, True]) == 1.0
    assert cal.auroc([0.1, 0.9], [False, True]) == 1.0
    assert cal.auroc([0.9, 0.1], [False, True]) == 0.0
    assert cal.auroc([0.3, 0.3], [False, True]) == 0.5
    assert cal.auroc([0.1, 0.9], [True, True]) is None
    # One bin holding everything: ECE is |accuracy - mean p|.
    assert cal.ece([0.5] * 4, [True, False, False, False]) == pytest.approx(0.25)


def test_metrics_reject_bad_inputs():
    for metric in (cal.ece, cal.brier, cal.auroc):
        with pytest.raises(ValueError):
            metric([], [])
        with pytest.raises(ValueError):
            metric([0.5], [True, False])


# Equal-mass binning edge cases.


def sizes(p, bins):
    return [len(b) for b in cal.equal_mass_bins(p, bins)]


def test_equal_mass_bins_sizes():
    assert sizes([i / 10 for i in range(10)], 10) == [1] * 10
    assert sizes([i / 25 for i in range(25)], 10) == [3, 2] * 5
    assert sizes([0.1, 0.2, 0.3], 10) == [1, 1, 1]  # fewer items than bins
    assert sizes([0.4] * 7, 10) == [7]  # all equal: one bin
    assert cal.equal_mass_bins([], 10) == []


def test_equal_mass_bins_keep_ties_together_whatever_the_order():
    # Rank 1 would open bin 0 and rank 2 bin 1; equal values stay with their first member.
    p = [0.1, 0.5, 0.5, 0.9]
    assert cal.equal_mass_bins(p, 2) == [[0, 1, 2], [3]]
    assert cal.equal_mass_bins([0.1, 0.1, 0.1, 0.9], 2) == [[0, 1, 2], [3]]
    shuffled = [0.5, 0.9, 0.1, 0.5]
    assert [sorted(shuffled[i] for i in b) for b in cal.equal_mass_bins(shuffled, 2)] == [
        [0.1, 0.5, 0.5],
        [0.9],
    ]
    # Every index lands in exactly one bin, ascending in p.
    rng = random.Random(3)
    values = [rng.choice([0.0625 * k for k in range(1, 17)]) for _ in range(161)]
    bins = cal.equal_mass_bins(values, 10)
    assert sorted(i for b in bins for i in b) == list(range(161))
    assert all(max(values[i] for i in a) < min(values[i] for i in b)
               for a, b in zip(bins, bins[1:], strict=False))


def test_equal_mass_bins_reject_zero_bins():
    with pytest.raises(ValueError):
        cal.equal_mass_bins([0.5], 0)


def test_reliability_bins():
    bins = cal.reliability([0.2, 0.2, 0.8, 0.8], [False, True, True, True], bins=2)
    assert [(b.n, b.p_mean, b.c_mean) for b in bins] == [(2, 0.2, 0.5), (2, 0.8, 1.0)]


# Bootstrap over tasks.


def test_bootstrap_one_task_gives_the_point_value():
    items = [("t", 0.2, False), ("t", 0.8, True), ("t", 0.6, True)]
    out = cal.bootstrap(items, {"brier": cal.brier, "auroc": cal.auroc}, reps=50, seed=0)
    assert out["brier"].low == out["brier"].high == pytest.approx(cal.brier(
        [0.2, 0.8, 0.6], [False, True, True]))
    assert out["auroc"] == cal.Interval(1.0, 1.0, 0)


def test_bootstrap_is_seeded_and_counts_undefined_resamples():
    items = [("a", 0.9, True), ("b", 0.1, False), ("c", 0.4, False), ("d", 0.7, True)]
    stats = {"auroc": cal.auroc, "brier": cal.brier}
    first = cal.bootstrap(items, stats, reps=200, seed=7)
    assert first == cal.bootstrap(items, stats, reps=200, seed=7)
    # A resample drawing only "a" and "d" (or only "b" and "c") has one class.
    assert first["auroc"].skipped > 0
    assert first["brier"].skipped == 0
    assert first["brier"].low <= cal.brier([0.9, 0.1, 0.4, 0.7], [True, False, False, True])
    with pytest.raises(ValueError):
        cal.bootstrap([], stats)


# Reproduction of a synthetic run.


def test_reproduction_rebuilds_attempts_score_and_signals(run):
    assert run.subset == "DEV4"
    assert run.score.score == pytest.approx(1.5)
    assert run.reload == 1.5
    assert run.candidates == 13
    assert run.duplicates == 0
    out = by_key(run, f"{T0}_0")
    assert [g.count for g in out.grids] == [2, 1]
    assert [g.score for g in out.grids] == pytest.approx([1.4, 0.8])
    assert out.grids[0].agreement == 2 / 16
    assert out.grids[0].prob == pytest.approx(math.exp(1.4) / (math.exp(1.4) + math.exp(0.8)))
    assert out.solved and out.correct.rank == 1
    assert by_key(run, f"{T1}_0").correct.rank == 2
    empty = by_key(run, f"{T2}_0")
    assert empty.grids == () and not empty.solved and empty.correct is None
    t3 = by_key(run, f"{T3}_0")
    assert [g.count for g in t3.grids] == [1, 1, 3]
    assert t3.top_agreement.correct and t3.top_agreement.views == 3


def test_weights_are_one_over_the_task_outputs(run):
    assert [o.weight for o in run.outputs if o.task == T0] == [0.5, 0.5]
    assert all(o.weight == 1.0 for o in run.outputs if o.task != T0)
    # T0 solves one of its two outputs: half a task.
    assert run.score.per_task[0].score == 0.5


def test_reproduction_fails_on_a_changed_attempt(tmp_path):
    changed = json.loads(json.dumps(SUBMISSION))
    changed[T1][0]["attempt_2"] = J1
    with pytest.raises(cal.ReproductionError, match=f"{T1}_0"):
        cal.reproduce("bad", write_arm(tmp_path / "arm", submission=changed))


def test_reproduction_fails_on_another_reload_score(tmp_path):
    with pytest.raises(cal.ReproductionError, match="Reload score"):
        cal.reproduce("bad", write_arm(tmp_path / "arm", reload="2.5"))


def test_run_files_reads_a_run_folder_with_its_session_log(tmp_path):
    root = tmp_path / "run"
    (root / "run_logs").mkdir(parents=True)
    (root / "run_logs" / "candidates.json").write_text(json.dumps(CANDIDATES), encoding="utf-8")
    (root / "submission.json").write_text(json.dumps(SUBMISSION), encoding="utf-8")
    (root / "session.log").write_text(selection_log(), encoding="utf-8")
    files = cal.run_files(root)
    assert files["selection"] == files["session"] == root / "session.log"
    assert cal.reproduce("run", root).score.score == pytest.approx(1.5)
    (root / "other.log").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="more than one session log"):
        cal.run_files(root)
    with pytest.raises(FileNotFoundError):
        cal.run_files(tmp_path / "missing")


def test_infer_subset():
    assert cal.infer_subset(list(DEV4)) == "DEV4"
    assert cal.infer_subset([DEV40[0]]) == ("DEV4" if DEV40[0] in DEV4 else "DEV40")
    with pytest.raises(ValueError):
        cal.infer_subset(["00000000"])
    with pytest.raises(ValueError):
        cal.infer_subset([])


# Decisions, the oracle gap, the coverage curve, the rules.


def test_decision_items(run):
    assert cal.promotion_items(run) == [
        (T0, 2 / 16, True), (T0, 1 / 16, False), (T1, 1 / 16, False), (T3, 1 / 16, False)
    ]
    assert cal.promotion_items(run, top_agreement=True)[-1] == (T3, 3 / 16, True)
    assert cal.promotion_items(run, decoded=True) == []  # no session log
    selection = cal.selection_items(run)
    assert len(selection) == 8
    assert [c for _, _, c in selection] == [True, False, False, False, False, True, False, False]


def test_oracle_gap_weights_and_split(run):
    gap = cal.oracle_gap(run)
    assert [o.key for o in gap.outputs] == [f"{T0}_1", f"{T3}_0"]
    assert gap.tasks == pytest.approx(1.5)
    assert gap.near == pytest.approx(1.5)
    assert gap.lower == 0
    assert gap.one_view == pytest.approx(0.5)  # H3 came from one view, K3 from three
    assert gap.points == pytest.approx(100 * 1.5 / 4)


def test_coverage_curve(run):
    rows = cal.coverage_curve(run)
    assert len(rows) == 16
    assert (rows[0].passed, rows[0].right, rows[0].outputs) == (4, 1, 4)
    assert (rows[1].passed, rows[1].right, rows[1].accuracy) == (1, 1, 1.0)
    assert rows[2].passed == 0 and rows[2].accuracy is None and rows[2].coverage == 0


def test_rules(run):
    scores = {rule: cal.rule_score(run, rule) for rule in cal.RULES}
    assert scores == pytest.approx(
        {"kgmon": 1.5, "agreement": 2.5, "kgmon-agreement": 1.5, "product": 2.5}
    )
    changes = {rule: cal.rule_change(run, rule) for rule in cal.RULES[1:]}
    for rule, ch in changes.items():
        assert ch.net == pytest.approx(scores[rule] - scores["kgmon"])
    assert (changes["agreement"].recovered, changes["agreement"].lost) == (1.0, 0.0)
    assert changes["kgmon-agreement"].identical  # no tie in s on this run
    assert not changes["agreement"].identical and not changes["product"].identical
    assert cal.score_ties(run) == 0
    with pytest.raises(ValueError):
        cal.rank_by("nope", by_key(run, f"{T0}_0").grids)


def test_choose_leaves_out_rules_identical_to_kgmon():
    gains = {"agreement": -0.67, "kgmon-agreement": 0.0, "product": -1.0}
    # Without the exclusion the identical rule wins on its zero gain.
    assert cal.choose(gains) == "kgmon-agreement"
    assert cal.choose(gains, frozenset({"kgmon-agreement"})) == "agreement"
    # The earlier rule on a tie; never kgmon; None when every alternative is identical.
    assert cal.choose(dict.fromkeys(cal.RULES, 0.0)) == "agreement"
    assert cal.choose({"agreement": -5.83, "kgmon-agreement": 0.0, "product": 0.0},
                      frozenset({"kgmon-agreement"})) == "product"
    assert cal.choose(gains, frozenset(cal.RULES[1:])) is None


def grid_row(rank, count, score, correct):
    return cal.GridRow(rank, count, count, count / 16, score, 0.0, 0.1, correct, ((rank,),))


def test_rule_change_counts_gross_recovered_and_lost():
    # Output 1 (half a task): agreement first submits C and D and loses the correct A.
    lost = cal.OutputRow(T0, 0, 0.5, (
        grid_row(1, 1, 0.9, True), grid_row(2, 1, 0.5, False),
        grid_row(3, 3, 0.0, False), grid_row(4, 2, -1.0, False),
    ))
    # Output 2 (a task): agreement first submits the correct Z, ranked 3rd by score_kgmon.
    won = cal.OutputRow(T1, 0, 1.0, (
        grid_row(1, 1, 1.0, False), grid_row(2, 1, 0.5, False), grid_row(3, 4, 0.0, True),
    ))
    run = cal.RunData("hand", Path("."), "DEV4", (lost, won), cal.SubsetScore("DEV4", ()), 0.0,
                      0, 0)
    ch = cal.rule_change(run, "agreement")
    assert (ch.recovered, ch.lost, ch.identical) == (1.0, 0.5, False)
    assert ch.net == 0.5
    # With every softmax value 0 the product ties everywhere and keeps score_kgmon's order.
    assert cal.rule_change(run, "product") == cal.RuleChange(0.0, 0.0, True)


def test_main_writes_report_and_csv_without_grids(tmp_path):
    arm = write_arm(tmp_path / "arm")
    out = tmp_path / "out"
    assert cal.main(["--run", f"syn/one={arm}", "--out", str(out), "--reps", "20"]) == 0
    report = (out / "calibration.md").read_text(encoding="utf-8")
    assert "syn/one" in report and "Oracle gap" in report
    text = (out / "syn-one.csv").read_text(encoding="utf-8")
    assert "[[" not in text
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 10  # distinct grids: 2 + 3 + 2 + 3
    assert sum(int(r["correct"]) for r in rows) == 4
    with pytest.raises(ValueError, match="KEY=PATH"):
        cal.main(["--run", "nopath", "--out", str(out)])
