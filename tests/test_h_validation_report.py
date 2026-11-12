"""P3.7: the report's arithmetic on hand cases, its reading of a run folder, and the CPU smoke path.

``scripts/h_validation_report.py``: $f_x = (m_x - m_a)/(m_c - m_a)$ per metric, the paired
bootstrap interval, DEV40's signed noise floor on task accuracy, G4's rows (No gap first, then
Good, Borderline, Bad) with their consequences before and after H's training ended, the
decision over both sets, its fallback to CE,
the paired set (puzzles every arm finished and scored), the seconds per op, the per-source tables
on GEN, and the smoke path (BRIEF A8): a tiny H from train.py's smoke mode, GEN from the episode
fixture, the load check, every arm on the CPU, the selection, and the report. The smoke test is
skipped without torch, transformers, peft, or safetensors; the rest needs none of them.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "_t_h_validation_report", REPO / "scripts" / "h_validation_report.py"
)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

ARMS = {"base": "base", "h": "hyper", "sgd": "sgd", "h-polish": "hyper_polish"}


def values(acc, a1=None, outputs=1, ce=1.0):
    return {"accuracy": acc, "a1": acc if a1 is None else a1, "outputs": outputs, "ce": ce}


def test_gap_closed_is_the_share_of_the_zero_to_sgd_gap():
    assert rp.gap_closed(6.0, 4.0, 8.0) == 0.5
    assert rp.gap_closed(8.0, 4.0, 8.0) == 1.0 and rp.gap_closed(4.0, 4.0, 8.0) == 0.0
    assert rp.gap_closed(2.0, 4.0, 8.0) == -0.5  # worse than the zero adapter
    # CE's gap is negative (lower is better); the ratio reads the same way.
    assert rp.gap_closed(0.75, 1.0, 0.5) == 0.5
    assert rp.gap_closed(1.0, 2.0, 2.0) is None and rp.gap_closed(math.nan, 1.0, 2.0) is None


def test_aggregate_sums_tasks_pools_outputs_and_averages_ce():
    rows = [
        {"accuracy": 1.0, "a1": 2.0, "outputs": 2.0, "ce": 0.2},
        {"accuracy": 0.5, "a1": 0.0, "outputs": 2.0, "ce": 0.4},
        {"accuracy": 0.0, "a1": 0.0, "outputs": 1.0, "ce": 0.9},
    ]
    m = rp.aggregate(rows)
    assert m["accuracy"] == 1.5 and m["attempt_1"] == 2.0 / 5.0
    assert m["ce"] == pytest.approx(0.5)


def test_the_bootstrap_is_paired_and_deterministic():
    a = [values(0.0, ce=2.0) for _ in range(10)]
    c = [values(1.0, ce=1.0) for _ in range(10)]
    same_as_c = {"a": a, "c": c, "b": c}
    out = rp.bootstrap(same_as_c, "b", "accuracy", resamples=200)
    assert (out["low"], out["high"], out["undefined"]) == (1.0, 1.0, 0)
    out = rp.bootstrap({"a": a, "c": c, "b": a}, "b", "ce", resamples=200)
    assert (out["low"], out["high"]) == (0.0, 0.0)
    # Half the puzzles closed: the interval holds 0.5 and repeats exactly with the seed.
    b = [values(1.0 if i % 2 else 0.0) for i in range(10)]
    first = rp.bootstrap({"a": a, "c": c, "b": b}, "b", "accuracy", resamples=500)
    again = rp.bootstrap({"a": a, "c": c, "b": b}, "b", "accuracy", resamples=500)
    assert first == again and first["low"] < 0.5 < first["high"]
    # No gap in any resample: every resample undefined.
    flat = rp.bootstrap({"a": a, "c": a, "b": b}, "b", "accuracy", resamples=50)
    assert flat == {"low": None, "high": None, "undefined": 50}


@pytest.mark.parametrize(
    ("f", "row"),
    [
        (0.5, "Good"),
        (0.9, "Good"),
        (0.25, "Borderline"),
        (0.4999, "Borderline"),
        (0.2499, "Bad"),
        (-0.3, "Bad"),
        (None, "Undefined"),
    ],
)
def test_g4_rows(f, row):
    assert rp.g4_row(f) == row
    assert rp.g4_branch(f).startswith(f"{row}: ")


def test_a_bad_row_stops_phase_4_only_after_hs_training_ended():
    # Not ended (MT-02's step 1,000, also while MT-03 runs): training continues, P3.7 repeats.
    early = rp.g4_branch(0.1)
    assert "training continues" in early and "repeats" in early and "does not stop" in early
    assert "Phase 4 stops" not in early
    final = rp.g4_branch(0.1, training_ended=True)
    assert "Phase 4 stops for the deadline" in final and "repeats" not in final
    # Good and Borderline mean the same in both cases.
    for f in (0.3, 0.7):
        assert rp.g4_branch(f) == rp.g4_branch(f, training_ended=True)
    # Every row has a consequence in both cases.
    rows = (rp.NO_GAP, rp.GOOD, rp.BORDERLINE, rp.BAD, rp.UNDEFINED)
    assert set(rp.CONSEQUENCES) == {(r, e) for r in rows for e in (False, True)}


def synthetic_run(acc, ce, order=None, statuses=None, timing=None, sources=None):
    """A run as read_run returns it: per arm its task scores and CEs per puzzle."""
    keys = order or [f"p{i}" for i in range(len(acc["base"]))]
    records, scores = {}, {name: {} for name in ARMS}
    for name in ARMS:
        for i, key in enumerate(keys):
            status = (statuses or {}).get((key, name), "done")
            records.setdefault(key, {})[name] = {"status": status, "ce": [{"ce": ce[name][i]}]}
            scores[name][key] = {
                "score": acc[name][i], "attempt_1": [acc[name][i] == 1.0], "correct": [True]
            }  # fmt: skip
    return {
        "arms": dict(ARMS), "order": keys, "records": records, "scores": scores,
        "timing": timing or {}, "load_check": None, "sources": sources or {}, "summary": {},
    }  # fmt: skip


def test_analyse_on_a_hand_case_with_the_noise_floor():
    acc = {
        "base": [0, 0, 0, 0, 0, 0], "h": [1, 1, 0, 0, 0, 0],
        "sgd": [1, 1, 1, 1, 0, 0], "h-polish": [1, 1, 1, 0, 0, 0],
    }  # fmt: skip
    ce = {"base": [2.0] * 6, "h": [1.5] * 6, "sgd": [1.0] * 6, "h-polish": [1.25] * 6}
    table = rp.analyse(synthetic_run(acc, ce), "DEV40")
    assert table["puzzles"] == 6
    assert table["arms"]["sgd"]["metrics"]["accuracy"] == 4
    b, d = table["fractions"]["b"], table["fractions"]["d"]
    assert b["accuracy"]["f"] == 0.5 and d["accuracy"]["f"] == 0.75
    assert b["ce"]["f"] == pytest.approx(0.5) and d["ce"]["f"] == pytest.approx(0.75)
    assert b["attempt_1"]["f"] == 0.5
    assert rp.g4_line(table)["metric"] == "accuracy" and "Phase 4 as planned" in (
        rp.g4_line(table)["branch"]
    )
    # A gap under 2 tasks: accuracy's f is undefined (the floor) and G4 reads CE.
    small = {**acc, "sgd": [1, 0, 0, 0, 0, 0], "h": [0] * 6, "h-polish": [0] * 6}
    table = rp.analyse(synthetic_run(small, ce), "DEV40")
    entry = table["fractions"]["b"]["accuracy"]
    assert entry["floor"] and entry["f"] is None and entry["raw"] == 0.0
    line = rp.g4_line(table)
    assert line["metric"] == "ce" and line["f"] == pytest.approx(0.5)


def test_the_paired_set_leaves_out_a_puzzle_any_arm_did_not_finish():
    acc = {name: [1, 0, 1] for name in ARMS}
    ce = {name: [1.0, 1.0, 1.0] for name in ARMS}
    run = synthetic_run(acc, ce, statuses={("p1", "sgd"): "cut", ("p2", "h"): "raised"})
    assert rp.paired(run) == ["p0"]
    del run["scores"]["h-polish"]["p0"]
    assert rp.paired(run) == []


def test_seconds_per_op_average_over_the_paired_puzzles():
    timing = {
        "sgd": {
            "p0": {"puzzle": 100.0, "train": 60.0, "decode": 30.0, "selection": 5.0},
            "p1": {"puzzle": 200.0, "train": 80.0, "decode": 100.0, "selection": 10.0},
        },
        "h": {"p0": {"puzzle": 50.0, "encode": 3.0, "h_forward": 0.01, "decode": 40.0}},
    }
    run = {"timing": timing}
    s = rp.op_seconds(run, "sgd", ["p0", "p1"])
    assert (s["span"], s["train"], s["decode"], s["rescoring"]) == (150.0, 70.0, 65.0, 7.5)
    h = rp.op_seconds(run, "h", ["p0"])
    assert (h["encode"], h["h_forward"], h["train"]) == (3.0, 0.01, 0.0)


def write_run(root, acc, ce, sources=None):
    """A run folder as the arms entry writes it (the pulled run_logs/arms/)."""
    arms = root / "arms"
    keys = [f"p{i}" for i in range(len(acc["base"]))]
    summary = {
        "settings": {"arms": [{"name": n, "source": s} for n, s in ARMS.items()]},
        "order": keys,
    }
    if sources:
        summary["gen"] = {"index": [{"key": k, "source": sources[k]} for k in keys]}
    arms.mkdir(parents=True)
    (arms / "arms.json").write_text(json.dumps(summary), encoding="utf-8")
    lines = []
    for name in ARMS:
        puzzles = {}
        (arms / name / "timing").mkdir(parents=True)
        timing = []
        for i, key in enumerate(keys):
            lines.append(json.dumps({"puzzle_id": key, "arm": name, "status": "done",
                                     "ce": [{"ce": ce[name][i]}]}))  # fmt: skip
            puzzles[key] = {"score": acc[name][i], "attempt_1": [bool(acc[name][i])],
                            "correct": [bool(acc[name][i])]}  # fmt: skip
            timing.append(json.dumps({"puzzle_id": key, "op": "puzzle", "seconds": 10.0,
                                      "tokens": None}))  # fmt: skip
        (arms / name / "scores.json").write_text(json.dumps({"puzzles": puzzles}), "utf-8")
        (arms / name / "timing" / "timing-rank0.jsonl").write_text("\n".join(timing) + "\n")
    (arms / "records-rank0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_report_reads_two_runs_writes_per_source_tables_and_flags_split_branches(tmp_path):
    acc_dev = {"base": [0] * 4, "h": [1, 1, 1, 0], "sgd": [1, 1, 1, 1], "h-polish": [1] * 4}
    ce = {"base": [2.0] * 4, "h": [1.2] * 4, "sgd": [1.0] * 4, "h-polish": [1.0] * 4}
    acc_gen = {"base": [0] * 4, "h": [0, 0, 0, 0], "sgd": [1, 1, 1, 1], "h-polish": [1, 0, 0, 0]}
    sources = {"p0": "barc", "p1": "barc", "p2": "rearc", "p3": "rearc"}
    dev = write_run(tmp_path / "dev", acc_dev, ce)
    gen = write_run(tmp_path / "gen", acc_gen, ce, sources)
    result = rp.report(
        {"DEV40": dev, "GEN": gen}, tmp_path / "report.md", tmp_path / "report.json"
    )
    labels = [t["label"] for t in result["tables"]]
    assert labels == ["DEV40", "GEN", "GEN barc", "GEN rearc"]
    assert result["g4"]["DEV40"]["f"] == 0.75 and result["g4"]["GEN"]["f"] == 0.0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "different G4 branches" in text and "the human decides at G4" in text
    assert "reference: P2.6-01" in text and "10.5 of 40" in text
    assert json.loads((tmp_path / "report.json").read_text())["g4"]["GEN"]["metric"] == "accuracy"


def test_the_cli_needs_a_run_or_the_smoke(capsys):
    assert rp.main([]) == 2
    assert rp.main(["--run", "nolabel"]) == 2


def test_the_smoke_path_runs_every_arm_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    result = rp.smoke(tmp_path / "smoke")
    table = result["tables"][0]
    assert table["label"] == "GEN" and table["puzzles"] == 3
    roles = [arm["role"] for arm in table["arms"].values()]
    assert roles == ["a", "b", "c", "d", "e", "f", "g"]
    # Arm (f) and train_log (amendment 8): the start comparison runs on every logged puzzle.
    start = table["start"]
    assert start is not None and start["logged"] == 3 and start["steps"] > 0
    # Arm (g) (task P3.9b amendment 1): read against (c) and against (f), on the same puzzles.
    merge, merge_fit = table["merge"], table["merge_fit"]
    assert merge["roles"] == ["c", "g"] and merge_fit["roles"] == ["f", "g"]
    assert merge["row"] in rp.MERGE_ROWS and merge_fit["row"] in rp.MERGE_FIT_ROWS
    assert merge["logged"] == merge_fit["logged"] == 3 and start["roles"] == ["c", "f"]
    text = (tmp_path / "smoke" / "report.md").read_text(encoding="utf-8")
    assert "**(g) against (c) (GEN):**" in text and "**(g) against (f) (GEN):**" in text
    assert "**(f) against (c) (GEN):**" in text
    run = rp.read_run(tmp_path / "smoke" / "run_logs")
    for rows in run["records"].values():
        if "h-merge-fit" in rows:  # every puzzle but the probe's
            assert rows["h-merge-fit"]["restored"] is True and rows["h-merge-fit"]["merge"]
        for name in ("sgd", "h-fit", "polish", "h-polish", "h-merge-fit"):
            if name in rows:
                assert rows[name]["train_log"], name
        assert "train_log" not in rows.get("base", {}) and "train_log" not in rows.get("h", {})
    assert run["load_check"]["verdict"] == "pass" and run["load_check"]["dev4"]["equal"]
    assert run["summary"]["exit"] == 0 and len(run["summary"]["gen"]["index"]) == 3
    assert {r["status"] for rows in run["records"].values() for r in rows.values()} == {"done"}
    assert (tmp_path / "smoke" / "report.md").is_file()


# --- fix round 1: the bootstrap is paired, its interval the 5th and 95th percentiles ------------


def test_the_bootstrap_resamples_one_set_of_puzzles_for_every_arm():
    """b equals c puzzle by puzzle while the puzzles differ: a paired resample gives f = 1 every
    time; arms resampled apart would give a spread."""
    c_scores = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    a = [values(0.0) for _ in c_scores]
    c = [values(float(x)) for x in c_scores]
    out = rp.bootstrap({"a": a, "c": c, "b": list(c)}, "b", "accuracy", resamples=2000)
    assert (out["low"], out["high"]) == (1.0, 1.0)
    # b equals a: f = 0 every time, though a and c differ per puzzle.
    out = rp.bootstrap({"a": a, "c": c, "b": list(a)}, "b", "accuracy", resamples=2000)
    assert (out["low"], out["high"]) == (0.0, 0.0)


def test_the_interval_is_the_5th_and_95th_percentile_of_the_resampled_fractions():
    import random

    n, resamples = 24, 4000
    rng = random.Random(7)
    a = [values(rng.choice([0.0, 0.5])) for _ in range(n)]
    c = [values(rng.choice([0.5, 1.0])) for _ in range(n)]
    b = [values(rng.choice([0.0, 0.5, 1.0])) for _ in range(n)]
    out = rp.bootstrap({"a": a, "c": c, "b": b}, "b", "accuracy", resamples=resamples, seed=3)
    # The same resamples, computed here: every defined fraction.
    draw = random.Random(3)
    found = []
    for _ in range(resamples):
        index = [draw.randrange(n) for _ in range(n)]
        m = {k: rp.aggregate([v[i] for i in index])["accuracy"] for k, v in
             (("a", a), ("c", c), ("b", b))}  # fmt: skip
        f = rp.gap_closed(m["b"], m["a"], m["c"])
        if f is not None:
            found.append(f)
    found.sort()
    assert len(found) + out["undefined"] == resamples
    below = sum(f < out["low"] for f in found) / len(found)
    at_or_below = sum(f <= out["low"] for f in found) / len(found)
    above = sum(f > out["high"] for f in found) / len(found)
    at_or_above = sum(f >= out["high"] for f in found) / len(found)
    assert below <= 0.05 <= at_or_below and above <= 0.05 <= at_or_above
    inside = sum(out["low"] <= f <= out["high"] for f in found) / len(found)
    assert inside >= 0.90 and out["low"] < out["high"]


def gap_table(acc_a, acc_c, ce_a, ce_c, acc_b=None, ce_b=None, n=4):
    """A set of n puzzles with the zero adapter's and SGD arm's totals (spread over puzzles)."""

    def spread(total):
        return [total / n] * n

    acc = {"base": spread(acc_a), "sgd": spread(acc_c)}
    acc["h"] = spread(acc_a if acc_b is None else acc_b)
    acc["h-polish"] = acc["h"]
    ce = {"base": [ce_a] * n, "sgd": [ce_c] * n}
    ce["h"] = [ce_a if ce_b is None else ce_b] * n
    ce["h-polish"] = ce["h"]
    return rp.analyse(synthetic_run(acc, ce), "DEV40")


@pytest.mark.parametrize(
    ("acc_c", "ce_c", "gap"),
    [
        (0.0, 2.5, True),  # SGD beats the zero adapter on neither metric
        (1.0, 2.0, True),  # one task better (under the floor), CE equal
        (-3.0, 2.5, True),  # signed: three tasks worse is no gap, not a defined accuracy f
        (1.0, 1.5, False),  # under the floor on accuracy, but better CE: G4 reads CE
        (3.0, 2.5, False),  # beats it by 3 tasks: G4 reads accuracy
    ],
)
def test_no_gap_is_signed_on_both_metrics(acc_c, ce_c, gap):
    table = gap_table(acc_a=4.0, acc_c=4.0 + acc_c, ce_a=2.0, ce_c=ce_c)
    assert rp.no_gap(table) is gap
    assert (rp.g4_line(table)["row"] == rp.NO_GAP) is gap


def test_no_gap_is_read_first_and_overrides_a_defined_ce_fraction():
    # SGD's CE is above the zero adapter's: f_b on CE is defined (0.5, Good by number), but the
    # SGD arm beats the zero adapter on neither metric, so the row is No gap.
    table = gap_table(acc_a=4.0, acc_c=4.0, ce_a=2.0, ce_c=2.5, ce_b=2.25)
    line = rp.g4_line(table)
    assert line["metric"] == "ce" and line["f"] == pytest.approx(0.5)
    assert line["row"] == rp.NO_GAP and line["no_gap"] is True and line["straddles"] is None
    assert "G4 reads the other set alone" in line["branch"]
    # A signed accuracy floor: SGD three tasks worse leaves accuracy's f undefined.
    worse = gap_table(acc_a=4.0, acc_c=1.0, ce_a=2.0, ce_c=2.5)
    assert worse["fractions"]["b"]["accuracy"]["f"] is None


def line_of(row, straddles=False):
    return {"row": row, "straddles": straddles}


@pytest.mark.parametrize("ended", [False, True])
def test_the_decision_combines_the_sets(ended):
    good, bad, gap = line_of(rp.GOOD), line_of(rp.BAD), line_of(rp.NO_GAP)
    # No gap on one set: the other alone decides.
    alone = rp.g4_decision({"DEV40": gap, "GEN": bad}, ended)
    assert alone.startswith("No gap on DEV40: G4 reads GEN alone.")
    assert rp.CONSEQUENCES[(rp.BAD, ended)] in alone
    # No gap on both: G4 cannot measure closure.
    both = rp.g4_decision({"DEV40": gap, "GEN": gap}, ended)
    assert "cannot measure closure" in both
    if ended:
        assert "does not enter Phase 4 for the deadline" in both
    else:
        assert "training continues" in both and "repeats" in both
    # Agreement gives the row's consequence; disagreement or a straddle goes to the human.
    assert rp.g4_decision({"DEV40": good, "GEN": good}, ended) == "Good: Phase 4 as planned."
    assert "the human decides at G4" in rp.g4_decision({"DEV40": good, "GEN": bad}, ended)
    straddle = {"DEV40": line_of(rp.GOOD, True), "GEN": good}
    assert "the human decides at G4" in rp.g4_decision(straddle, ended)


def test_the_report_states_the_training_case(tmp_path):
    acc = {"base": [0] * 4, "h": [0] * 4, "sgd": [1, 1, 1, 1], "h-polish": [0] * 4}
    ce = {"base": [2.0] * 4, "h": [2.0] * 4, "sgd": [1.0] * 4, "h-polish": [2.0] * 4}
    run = write_run(tmp_path / "dev", acc, ce)
    early = rp.report({"DEV40": run}, tmp_path / "early.md", None)
    assert early["training_ended"] is False and early["g4"]["DEV40"]["row"] == rp.BAD
    assert "H's training continues" in early["decision"]
    assert "not ended" in (tmp_path / "early.md").read_text(encoding="utf-8")
    final = rp.report({"DEV40": run}, tmp_path / "final.md", None, training_ended=True)
    assert "Phase 4 stops for the deadline" in final["decision"]
    assert rp.main(["--run", f"DEV40={run}", "--training-ended"]) == 0


FIT_ARMS = {"sgd": "sgd", "h-fit": "hyper_fit"}


def fit_run(acc_c, acc_f, loss_c=None, loss_f=None):
    """Run P3.7-03's shape: arms (c) and (f) only, per puzzle a task score and a loss curve."""
    keys = [f"p{i}" for i in range(len(acc_c))]
    records, scores = {}, {"sgd": {}, "h-fit": {}}
    for i, key in enumerate(keys):
        for name, acc, losses in (("sgd", acc_c, loss_c), ("h-fit", acc_f, loss_f)):
            row = {"status": "done", "ce": [{"ce": 1.0}]}
            if losses is not None:
                row["train_log"] = [
                    {"step": s, "loss": v, "learning_rate": 1e-5, "grad_norm": None}
                    for s, v in enumerate(losses[i], start=1)
                ]
            records.setdefault(key, {})[name] = row
            scores[name][key] = {"score": acc[i], "attempt_1": [acc[i] == 1.0], "correct": [True]}
    return {
        "arms": dict(FIT_ARMS), "order": keys, "records": records, "scores": scores,
        "timing": {}, "load_check": None, "sources": {}, "summary": {},
    }  # fmt: skip


def test_trailing_means_and_the_reach_step():
    assert rp.trailing([4.0, 2.0, 0.0], window=2) == [4.0, 3.0, 1.0]
    c = [3.0] * 8 + [1.0] * 8  # (c)'s final trailing mean over 8 steps: 1.0
    f = [1.0] * 16  # (f) is there from step 1
    assert rp.reach_step(f, c) == 1
    assert rp.reach_step([2.0] * 16, c) is None
    assert rp.reach_step([], c) is None
    slow = [3.0] * 8 + [1.0] * 8
    assert rp.reach_step(slow, c) == 16


@pytest.mark.parametrize(
    ("acc_f", "row"),
    [
        ([1, 1, 1, 0, 0, 0], "Start helps"),  # +2 tasks: at the floor
        ([1, 1, 0, 0, 0, 0], "Same score"),  # +1
        ([1, 0, 0, 0, 0, 0], "Same score"),  # 0
        ([0, 0, 0, 0, 0, 0], "Same score"),  # -1
    ],
)
def test_the_start_rows_on_hand_cases(acc_f, row):
    acc_c = [1, 0, 0, 0, 0, 0]
    table = rp.analyse(fit_run(acc_c, acc_f), "DEV40")
    start = table["start"]
    assert start["delta"] == sum(acc_f) - 1 and start["row"] == row
    assert table["fractions"] == {} and not rp.has_gap_arms(table)


def test_a_worse_start_and_the_interval():
    acc_c = [1, 1, 1, 0, 0, 0]
    table = rp.analyse(fit_run(acc_c, [1, 0, 0, 0, 0, 0]), "DEV40")
    start = table["start"]
    assert start["delta"] == -2 and start["row"] == "Start hurts"
    assert start["low"] <= -2 <= start["high"] and start["high"] <= 0


def test_the_loss_curves_and_the_sooner_reading():
    acc = [0, 0, 0, 0]
    c = [[3.0] * 64 + [1.0] * 64 for _ in acc]  # (c) ends at 1.0
    fast = [[1.0] * 128 for _ in acc]  # (f) starts there
    late = [[3.0] * 120 + [1.0] * 8 for _ in acc]  # (f) gets there at step 128
    start = rp.analyse(fit_run(acc, acc, c, fast), "DEV40")["start"]
    assert start["reach_median"] == 1 and start["sooner"] is True and start["steps"] == 128
    assert start["loss"]["c"][1] == 3.0 and start["loss"]["f"][128] == 1.0
    assert start["reached"] == start["logged"] == 4
    start = rp.analyse(fit_run(acc, acc, c, late), "DEV40")["start"]
    assert start["reach_median"] == 128 and start["sooner"] is False
    # Without logs: no curves, no reading.
    start = rp.analyse(fit_run(acc, acc), "DEV40")["start"]
    assert start["logged"] == 0 and start["sooner"] is None and start["loss"]["c"][1] is None


def test_the_report_of_a_fit_run_has_the_start_and_no_g4_line(tmp_path):
    run = fit_run([1, 0, 0, 0], [1, 1, 1, 0], [[2.0] * 16] * 4, [[1.0] * 16] * 4)
    table = rp.analyse(run, "DEV40")
    text = rp.render([table], {"DEV40": run})
    assert "**(f) against (c) (DEV40):**" in text and "Start helps" in text
    assert "G4 line" not in text and "**G4 (" not in text
    assert "median step 1 of 16" in text
