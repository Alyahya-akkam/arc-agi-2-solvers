"""Tests of arc26.governor (task P2.4): cost model, budgets, the ladder, the decode trade.

Three groups: the config against its sources (the P1.1-01 report rows and the raw records), unit
tests with an injected clock, and a simulator of four workers draining one shared queue of
synthetic puzzles under the governor (``simulate``). The simulator follows the reference's worker
loop (``arc26/nvarc/arc_solver.py``, ``worker``) and adds one governed round after each puzzle's
baseline. Its true op costs are the cost model's central line times heavy-tailed noise, calibrated
to the P1.1-01 records; the governor only sees its conservative config.
"""

import functools
import heapq
import importlib.util
import json
import math
import random
import re
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import pytest
import yaml

from arc26 import governor as gv
from arc26.timing import DECODE, PUZZLE, SELECTION, TRAIN, read_records

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "governor.yaml"
REPORT = REPO / "reports" / "P1.1-01"
SUMMARY = REPORT / "summary.md"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@functools.cache
def shipped():
    return gv.CostModel.from_config(CONFIG)


def model(margin=None):
    """The shipped cost model, with another margin when given."""
    base = shipped()
    if margin is None:
        return base
    return gv.CostModel(base.ops, margin, base.candidates_per_batch, base.overhead)


class Clock:
    """Injectable clock for the governor."""

    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class Index:
    """An integer scalar that is not an int, as numpy's: converts through __index__."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


# -------------------------------------------------------------------------------------------
# The config against its sources


def summary_row(label, cells):
    """Cells of the summary.md table row with this label and this many cells."""
    for line in SUMMARY.read_text(encoding="utf-8").splitlines():
        row = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and row[0] == label and len(row) == cells:
            return row
    raise AssertionError(f"no {cells}-cell row {label!r} in {SUMMARY}")


def number(cell):
    return float(cell.split()[0].replace(",", ""))


def nearest_rank(values, p):
    ordered = sorted(values)
    return ordered[math.ceil(p * len(ordered)) - 1]


@functools.cache
def p11_records():
    records, skipped = read_records(sorted(REPORT.glob("timing-rank*.jsonl")))
    assert skipped == 0
    return records


@functools.cache
def p11_puzzles():
    """Per P1.1-01 puzzle: T, the decode batches, the candidates after each batch, the span."""
    puzzles = {}
    for record in p11_records():
        if record.puzzle_id is None:
            continue
        p = puzzles.setdefault(
            record.puzzle_id,
            {"tokens": None, "batches": [], "candidates": [], "rescoring": [], "ops": 0.0},
        )
        if record.op == TRAIN:
            p["tokens"] = record.tokens
            p["ops"] += record.seconds
        elif record.op == DECODE:
            p["batches"].append(record.seconds)
            p["candidates"].append(0)
            p["rescoring"].append(0.0)
            p["ops"] += record.seconds
        elif record.op == SELECTION:
            p["candidates"][-1] += 1
            p["rescoring"][-1] += record.seconds
            p["ops"] += record.seconds
        elif record.op == PUZZLE:
            p["span"] = record.seconds
    return puzzles


def test_config_loads_into_the_model_and_the_governor():
    cost = shipped()
    assert set(cost.ops) == {TRAIN, DECODE, SELECTION}
    assert cost.provenance["run_id"] == "P1.1-01"
    assert cost.provenance["train_fit"] == {"r2": 0.998, "n": 120}
    governor = gv.Governor.from_config(CONFIG, end_time=1000.0, workers=4, clock=Clock())
    assert governor.policy == gv.REFERENCE
    assert governor.reserve == 0.0
    assert governor.min_batches_per_input == 2
    assert governor.cost_model.margin == cost.margin == 0.35
    # configs/ is not in the wheel: the loop can pass the parsed config instead of the file.
    mapped = gv.Governor.from_mapping(config(), end_time=1000.0, workers=4, clock=Clock())
    assert mapped.cost_model.ops == cost.ops
    assert (mapped.policy, mapped.reserve, mapped.cost_model.overhead) == (
        governor.policy,
        governor.reserve,
        cost.overhead,
    )
    assert gv.CostModel.from_mapping(config()).ops == cost.ops


def test_config_coefficients_equal_their_report_rows():
    cost = shipped()
    train = cost.ops[TRAIN]
    fit = summary_row("`a + b T + c T^2 / steps`", 7)  # summary.md line 90
    assert int(fit[1]) == config()["provenance"]["train_fit"]["n"]
    assert (train.a, train.b, train.c) == (number(fit[2]), number(fit[3]), number(fit[4]))
    assert number(fit[5]) == config()["provenance"]["train_fit"]["r2"]
    assert train.allowance == number(fit[6])
    assert train.steps == 128
    decode = summary_row("decode, per batch", 7)  # line 48: n, mean, median, p90, max, total
    assert int(decode[1]) == config()["provenance"]["decode_batches"]["n"]
    assert cost.ops[DECODE].a == number(decode[4])
    selection = summary_row("selection, per candidate", 7)  # line 49
    assert int(selection[1]) == config()["provenance"]["selection_candidates"]["n"]
    assert cost.ops[SELECTION].a == number(selection[4])
    overhead = summary_row("overhead", 8)  # line 43: the max per-puzzle overhead
    assert cost.overhead == number(overhead[5])
    distinct = re.search(r"distinct grids: mean [\d.]+, median \d+, p90 (\d+)", SUMMARY.read_text())
    per_output = gv.REFERENCE_BATCHES_PER_INPUT  # batches per test output in the reference
    assert cost.candidates_per_batch == math.ceil(int(distinct.group(1)) / per_output)


def test_config_coefficients_recompute_from_the_p11_records():
    cost = shipped()
    decode = [r.seconds for r in p11_records() if r.op == DECODE]
    selection = [r.seconds for r in p11_records() if r.op == SELECTION]
    assert len(decode) == 669 and len(selection) == 1181
    assert round(nearest_rank(decode, 0.9), 1) == cost.ops[DECODE].a
    assert round(nearest_rank(selection, 0.9), 1) == cost.ops[SELECTION].a
    candidates = [c for p in p11_puzzles().values() for c in p["candidates"]]
    assert len(candidates) == 669
    assert nearest_rank(candidates, 0.9) == cost.candidates_per_batch
    # The config's comment on selection: 4 candidates at the p90 price 31.2 s of rescoring per
    # batch, against the measured p90 of rescoring seconds per batch.
    rescoring = [r for p in p11_puzzles().values() for r in p["rescoring"]]
    assert round(nearest_rank(rescoring, 0.9), 1) == 20.6
    assert cost.candidates_per_batch * cost.ops[SELECTION].a == pytest.approx(31.2)
    overheads = [p["span"] - p["ops"] for p in p11_puzzles().values()]
    assert round(max(overheads), 1) == cost.overhead
    # Least squares of train seconds on (1, T, T^2 / 128), solved by Cramer's rule.
    rows = [(1.0, r.tokens, r.tokens**2 / 128, r.seconds) for r in p11_records() if r.op == TRAIN]
    gram = [[math.fsum(x[i] * x[j] for x in rows) for j in range(3)] for i in range(3)]
    moment = [math.fsum(x[i] * x[3] for x in rows) for i in range(3)]

    def det(m):
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    full = det(gram)
    a, b, c = (
        det([[moment[r] if col == i else gram[r][col] for col in range(3)] for r in range(3)])
        / full
        for i in range(3)
    )
    train = cost.ops[TRAIN]
    assert round(a, 2) == train.a
    assert float(f"{b:.4g}") == train.b and float(f"{c:.4g}") == train.c
    residuals = [y - (a + b * t + c * q) for _, t, q, y in rows]
    mean = statistics.fmean(y for *_, y in rows)
    r2 = 1 - math.fsum(e * e for e in residuals) / math.fsum((y - mean) ** 2 for *_, y in rows)
    assert round(r2, 3) == config()["provenance"]["train_fit"]["r2"]
    assert round(max(abs(e) for e in residuals), 1) == train.allowance


def test_margin_covers_every_p11_puzzle_and_is_needed():
    # Each P1.1-01 puzzle is a measured round: one fit of T tokens in 128 steps, its decode
    # batches, their rescoring, overhead. Priced as a gradient round of that shape, every
    # measured span must fit the estimate; without the margin, some must not.
    margin = shipped().margin
    ratios = {}
    for pid, p in p11_puzzles().items():
        shape = {"train_tokens": p["tokens"], "test_batches": len(p["batches"])}
        bare = model(0.0).round_costs(**shape)[gv.GRADIENT]
        assert p["span"] <= shipped().round_costs(**shape)[gv.GRADIENT], pid
        ratios[pid] = p["span"] / bare
    worst = max(ratios, key=ratios.get)
    assert 1.0 < ratios[worst] <= 1.0 + margin
    assert (worst, round(ratios[worst], 2)) == ("7b0280bc", 1.30)
    # 8 of 120 take longer than the bare estimate (8b7bacbf sits 0.02 s under its own).
    assert sum(r > 1.001 for r in ratios.values()) == 8


def test_deadline_is_the_rerun_solver_end_time():
    notebook = load_script("build_kaggle_notebook")
    run = yaml.safe_load((REPO / "configs" / "submission_P1.1-01.yaml").read_text())
    end_s = config()["deadline"]["end_s"]
    assert end_s == notebook.COMPETITION_SOLVER_END_S == run["solver_end_s"] == 12 * 3600 - 1800


# -------------------------------------------------------------------------------------------
# Cost model


def test_train_estimate_is_the_quadratic_fit_plus_allowance_times_the_margin():
    cost = shipped()
    t = 211_264
    fit = 24.35 + 0.0005812 * t + 3.487e-08 * t * t / 128
    assert cost.estimate(TRAIN, t) == pytest.approx(1.35 * (fit + 22.1))
    # A round's fit passes its own steps: fewer, longer steps cost more attention.
    short = 24.35 + 0.0005812 * t + 3.487e-08 * t * t / 32
    assert cost.estimate(TRAIN, t, steps=32) == pytest.approx(1.35 * (short + 22.1))
    assert cost.estimate(TRAIN, t, count=2) == pytest.approx(2 * cost.estimate(TRAIN, t))
    assert cost.estimate(TRAIN, Index(t)) == cost.estimate(TRAIN, t)


def test_zero_tokens_and_zero_counts_cost_nothing():
    cost = shipped()
    assert cost.estimate(TRAIN, 0) == 0.0
    assert cost.estimate(TRAIN, 5_000, count=0) == 0.0
    assert cost.estimate(DECODE, count=0) == 0.0
    assert cost.decode_cost(0) == 0.0
    costs = cost.round_costs(train_tokens=0, test_batches=4)
    assert costs[gv.GRADIENT] == pytest.approx(costs[gv.IN_CONTEXT])


def test_a_negative_fit_is_clamped_at_zero():
    linear = gv.OpCost("fit", -7.06, 0.000798, token_axis=True)  # the linear fit, line 89
    assert linear.seconds(1_000) == 0.0
    assert linear.seconds(100_000) == pytest.approx(-7.06 + 79.8)


def test_decode_and_selection_are_per_unit_quantiles_without_a_token_axis():
    cost = shipped()
    assert cost.estimate(DECODE) == pytest.approx(1.35 * 143.9)
    assert cost.estimate(DECODE, 50_000, count=3) == pytest.approx(3 * 1.35 * 143.9)
    assert cost.estimate(SELECTION, count=4) == pytest.approx(4 * 1.35 * 7.8)
    per_batch = 1.35 * (143.9 + 4 * 7.8)
    assert cost.decode_cost(6) == pytest.approx(6 * per_batch)
    assert cost.decode_cost(6, length_ratio=1.3) == pytest.approx(1.3 * 6 * per_batch)


def test_observed_batch_raises_but_never_lowers_the_decode_price():
    cost = shipped()
    assert cost.decode_cost(4, observed_batch=50.0) == cost.decode_cost(4)
    slow = cost.decode_cost(4, observed_batch=260.0)
    assert slow == pytest.approx(4 * 1.35 * (260.0 + 4 * 7.8))
    costs = cost.round_costs(train_tokens=100_000, test_batches=4, observed_batch=260.0)
    plain = cost.round_costs(train_tokens=100_000, test_batches=4)
    assert costs[gv.GRADIENT] - plain[gv.GRADIENT] == pytest.approx(slow - cost.decode_cost(4))
    assert costs[gv.IN_CONTEXT] - plain[gv.IN_CONTEXT] == pytest.approx(slow - cost.decode_cost(4))


def test_round_costs_compose_the_ops_of_each_variant():
    cost = shipped()
    costs = cost.round_costs(
        train_tokens=300_000, test_batches=8, gate_batches=2, train_steps=64, length_ratio=1.25
    )
    overhead = 1.35 * 9.6
    train = cost.estimate(TRAIN, 300_000, steps=64)
    assert costs[gv.GRADIENT] == pytest.approx(train + cost.decode_cost(10) + overhead)
    assert costs[gv.IN_CONTEXT] == pytest.approx(1.25 * cost.decode_cost(10) + overhead)
    assert set(costs) == set(gv.LADDER)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda c: c.estimate("prefill", 10), ValueError),
        (lambda c: c.estimate(TRAIN), ValueError),
        (lambda c: c.estimate(TRAIN, -1), ValueError),
        (lambda c: c.estimate(TRAIN, True), TypeError),
        (lambda c: c.estimate(TRAIN, 2.5), TypeError),
        (lambda c: c.estimate(TRAIN, 10, steps=0), ValueError),
        (lambda c: c.estimate(DECODE, count=-1), ValueError),
        (lambda c: c.estimate(DECODE, count=1.0), TypeError),
        (lambda c: c.decode_cost(2, length_ratio=0.9), ValueError),
        (lambda c: c.decode_cost(2, length_ratio=math.inf), ValueError),
        (lambda c: c.decode_cost(2, observed_batch=-1.0), ValueError),
        (lambda c: c.round_costs(train_tokens=10, test_batches=-4), ValueError),
    ],
)
def test_cost_model_refuses_bad_arguments(call, error):
    with pytest.raises(error):
        call(shipped())


def test_cost_model_refuses_bad_ops_and_config_sections():
    ops = dict(shipped().ops)
    with pytest.raises(ValueError, match="missing"):
        gv.CostModel({k: v for k, v in ops.items() if k != SELECTION}, 0.1, 4)
    with pytest.raises(ValueError, match="unknown"):
        gv.CostModel({**ops, "prefill": ops[DECODE]}, 0.1, 4)
    with pytest.raises(ValueError, match="token axis"):
        gv.CostModel({**ops, DECODE: ops[TRAIN]}, 0.1, 4)
    with pytest.raises(TypeError):
        gv.CostModel({**ops, DECODE: 143.9}, 0.1, 4)
    with pytest.raises(ValueError):
        gv.CostModel(ops, -0.1, 4)
    with pytest.raises(ValueError):
        gv.OpCost("batch", 143.9, b=0.01)
    with pytest.raises(ValueError):
        gv.OpCost("batch", -1.0)
    with pytest.raises(ValueError):
        gv.OpCost("fit", 1.0, 0.1, steps=0, token_axis=True)
    bad = json.loads(json.dumps(config()))
    bad["ops"][DECODE]["p90"] = 143.9
    with pytest.raises(ValueError, match="unknown keys"):
        gv.CostModel.from_mapping(bad)
    del bad["ops"][DECODE]
    with pytest.raises(KeyError):
        gv.CostModel.from_mapping(bad)


# -------------------------------------------------------------------------------------------
# Budgets


def test_puzzle_budget_is_the_remaining_time_over_this_workers_share():
    clock = Clock(1_000.0)
    governor = gv.Governor(shipped(), end_time=41_000.0, workers=4, clock=clock)
    left = 40_000.0
    sentinels = 4  # the reference queues one None per worker after the task ids
    assert governor.puzzle_budget(0 + sentinels) == left  # the last puzzle: all the time left
    for queued in range(1, 5):
        assert governor.puzzle_budget(queued + sentinels) == left / 2  # ceil(q / 4) = 1
    assert governor.puzzle_budget(5 + sentinels) == left / 3
    assert governor.puzzle_budget(116 + sentinels) == left / 30
    assert governor.puzzle_budget(Index(8 + sentinels)) == left / 3


def test_puzzle_budget_subtracts_the_reference_queues_sentinels():
    # arc26/nvarc/starter.py queues the task ids, then one None per worker; a worker stops when
    # it takes a None, so the loop's queue.qsize() counts the sentinels still queued.
    workers, ids = 4, [f"{i:08x}" for i in range(9)]
    reference = Queue()
    for item in [*ids, *[None] * workers]:
        reference.put(item)
    governor = gv.Governor(shipped(), end_time=900.0, workers=workers, clock=Clock())
    taken = 0
    while (item := reference.get()) is not None:
        taken += 1
        share = 1 + math.ceil((len(ids) - taken) / workers)
        assert governor.puzzle_budget(reference.qsize()) == 900.0 / share
    assert taken == len(ids)
    # A worker that took its sentinel has stopped; the others' last puzzles get all the time.
    assert reference.qsize() == workers - 1
    assert governor.puzzle_budget(reference.qsize()) == 900.0


def test_an_empty_queue_leaves_all_the_time_to_the_puzzle_in_hand():
    governor = gv.Governor(shipped(), end_time=900.0, workers=4, clock=Clock(300.0))
    for size in (0, 1, 4):  # nothing queued, or only sentinels
        assert governor.puzzle_budget(size) == 600.0
    assert governor.puzzle_budget(5) == 300.0  # one task id still queued


def test_one_worker_owns_the_whole_queue():
    governor = gv.Governor(shipped(), end_time=600.0, workers=1, clock=Clock())
    assert governor.puzzle_budget(0 + 1) == 600.0
    assert governor.puzzle_budget(5 + 1) == 100.0


def test_budgets_shrink_to_zero_as_the_deadline_nears():
    clock = Clock()
    governor = gv.Governor(shipped(), end_time=41_400.0, workers=4, clock=clock, reserve=600.0)
    budgets = []
    for now in range(0, 42_001, 600):
        clock.now = float(now)
        budgets.append(governor.puzzle_budget(10 + 4))
    positive = [b for b in budgets if b > 0]
    assert all(later < earlier for earlier, later in zip(positive, positive[1:], strict=False))
    assert budgets[: len(positive)] == positive
    assert set(budgets[len(positive) :]) == {0.0}  # from end - reserve on
    clock.now = 40_200.0
    assert governor.puzzle_budget(4) == 600.0  # the reserve is held back


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"workers": 0}, ValueError),
        ({"workers": 2.0}, TypeError),
        ({"end_time": math.nan}, ValueError),
        ({"end_time": "41400"}, TypeError),
        ({"reserve": -1.0}, ValueError),
        ({"policy": "cheapest"}, ValueError),
        ({"min_batches_per_input": 0}, ValueError),
        ({"min_batches_per_input": 5}, ValueError),
        ({"clock": 0.0}, TypeError),
        ({"cost_model": {"train": 1}}, TypeError),
    ],
)
def test_governor_refuses_bad_settings(kwargs, error):
    settings = {"cost_model": shipped(), "end_time": 41_400.0, "workers": 4, "clock": Clock()}
    with pytest.raises(error):
        gv.Governor(**{**settings, **kwargs})


def test_puzzle_budget_refuses_a_bad_queue_length():
    governor = gv.Governor(shipped(), end_time=100.0, workers=4, clock=Clock())
    with pytest.raises(ValueError):
        governor.puzzle_budget(-1)
    with pytest.raises(TypeError):
        governor.puzzle_budget(None)


# -------------------------------------------------------------------------------------------
# The ladder


@pytest.fixture
def governor():
    return gv.Governor(shipped(), end_time=10_000.0, workers=4, clock=Clock())


def test_ladder_prefers_gradient_even_when_in_context_is_cheaper(governor):
    costs = {gv.GRADIENT: 900.0, gv.IN_CONTEXT: 400.0}
    assert governor.choose(costs, 900.0) == gv.GRADIENT  # B2 Q4 order, not the cheapest
    assert governor.choose(costs, 899.9) == gv.IN_CONTEXT
    assert governor.choose(costs, 400.0) == gv.IN_CONTEXT
    assert governor.choose(costs, 399.9) == gv.NONE


def test_a_puzzle_with_no_room_gets_none(governor):
    costs = {gv.GRADIENT: 10.0, gv.IN_CONTEXT: 5.0}
    assert governor.choose(costs, 0.0) == gv.NONE
    assert governor.choose(costs, -50.0) == gv.NONE
    assert governor.choose({}, 1e9) == gv.NONE
    assert governor.choose({gv.IN_CONTEXT: 0.0}, 0.0) == gv.IN_CONTEXT


def test_an_expensive_in_context_round_never_replaces_gradient(governor):
    # With a quarter-epoch fit the gradient round can cost less than the in-context round's
    # longer prompts; when gradient does not fit, in-context does not either.
    costs = {gv.GRADIENT: 1_500.0, gv.IN_CONTEXT: 1_800.0}
    assert governor.choose(costs, 1_600.0) == gv.GRADIENT
    assert governor.choose(costs, 1_499.0) == gv.NONE


@pytest.mark.parametrize(
    ("costs", "remaining", "error"),
    [
        ({"none": 0.0}, 10.0, ValueError),
        ({gv.GRADIENT: -1.0}, 10.0, ValueError),
        ({gv.GRADIENT: math.nan}, 10.0, ValueError),
        ({gv.GRADIENT: "1"}, 10.0, TypeError),
        ({gv.GRADIENT: 1.0}, math.inf, ValueError),
        ([(gv.GRADIENT, 1.0)], 10.0, TypeError),
    ],
)
def test_choose_refuses_bad_costs(governor, costs, remaining, error):
    with pytest.raises(error):
        governor.choose(costs, remaining)


# -------------------------------------------------------------------------------------------
# Plans and the decode trade


def test_plan_after_the_baseline_is_choose_on_the_time_left():
    clock = Clock(500.0)
    governor = gv.Governor(shipped(), end_time=10_000.0, workers=4, clock=clock)
    costs = {gv.GRADIENT: 900.0, gv.IN_CONTEXT: 400.0}
    for deadline in (1_400.0, 1_399.0, 900.0, 899.0, 100.0):
        plan = governor.plan(costs, deadline, budget=1_000.0, puzzle_id="0934a4d8")
        assert plan.decision == governor.choose(costs, deadline - 500.0)
        assert plan.remaining == deadline - 500.0
        assert plan.cut_batches == plan.decode_batches == plan.pending_batches == 0
    record = governor.plan(costs, 1_400.0, budget=1_000.0, puzzle_id="0934a4d8").to_record()
    assert json.loads(json.dumps(record)) == record
    assert record["decision"] == gv.GRADIENT
    assert record["estimates"] == {gv.GRADIENT: 900.0, gv.IN_CONTEXT: 400.0}
    assert record["margin"] == 0.35 and record["policy"] == gv.REFERENCE


PER_BATCH = 1.35 * (143.9 + 4 * 7.8)  # conservative decode and rescoring of one batch
COSTS = {gv.GRADIENT: 1_000.0, gv.IN_CONTEXT: 500.0}


def baseline_plan(policy, room, costs=COSTS, **kwargs):
    """Plan at clock 0 with ``room`` batches' worth of time beyond the gradient round's cost."""
    governor = gv.Governor(shipped(), end_time=10_000.0, workers=4, clock=Clock(), policy=policy)
    deadline = room * PER_BATCH + 1_000.0
    return governor.plan(costs, deadline, budget=deadline, **kwargs)


def test_reference_policy_never_cuts_the_baseline_decode():
    assert shipped().decode_cost(1) == pytest.approx(PER_BATCH)
    plan = baseline_plan(gv.REFERENCE, 7.5, pending_batches=8, test_inputs=2)
    assert (plan.decision, plan.decode_batches, plan.cut_batches) == (gv.IN_CONTEXT, 8, 0)
    plan = baseline_plan(gv.REFERENCE, 5.5, pending_batches=8, test_inputs=2)
    assert (plan.decision, plan.decode_batches, plan.cut_batches) == (gv.NONE, 8, 0)
    assert plan.decode_seconds == pytest.approx(8 * PER_BATCH)


def test_trade_cuts_the_fewest_batches_that_make_room_for_the_first_fitting_variant():
    # Two test inputs: 8 batches pending; the floor keeps 2 per input, 4 in all.
    plan = baseline_plan(gv.TRADE, 6.5, pending_batches=8, test_inputs=2)
    assert (plan.decision, plan.decode_batches, plan.cut_batches) == (gv.GRADIENT, 6, 2)
    # Gradient would need a cut below the floor; in-context fits in 3.5 + 2.1 batches' room.
    plan = baseline_plan(gv.TRADE, 3.5, pending_batches=8, test_inputs=2)
    assert (plan.decision, plan.decode_batches, plan.cut_batches) == (gv.IN_CONTEXT, 5, 3)
    # Nothing fits even at the floor: no batch is given up.
    plan = baseline_plan(gv.TRADE, 3.5, {gv.GRADIENT: 1_000.0}, pending_batches=8, test_inputs=2)
    assert (plan.decision, plan.decode_batches, plan.cut_batches) == (gv.NONE, 8, 0)
    # Everything fits: no cut either.
    plan = baseline_plan(gv.TRADE, 9.0, pending_batches=8, test_inputs=2)
    assert (plan.decision, plan.decode_batches, plan.cut_batches) == (gv.GRADIENT, 8, 0)


def test_trade_floor_counts_the_batches_already_run():
    gradient = {gv.GRADIENT: 1_000.0}
    # One test input, 2 of its 4 batches done: the floor is met, so the trade may stop now.
    plan = baseline_plan(gv.TRADE, 0.5, gradient, pending_batches=2, done_batches=2)
    assert (plan.decision, plan.decode_batches) == (gv.GRADIENT, 0)
    # 1 of 4 done: one more batch must run first.
    plan = baseline_plan(gv.TRADE, 0.5, gradient, pending_batches=3, done_batches=1)
    assert (plan.decision, plan.decode_batches) == (gv.NONE, 3)
    plan = baseline_plan(gv.TRADE, 1.5, gradient, pending_batches=3, done_batches=1)
    assert (plan.decision, plan.decode_batches) == (gv.GRADIENT, 1)


def test_trade_prices_pending_batches_at_the_slowest_one_seen():
    gradient = {gv.GRADIENT: 1_000.0}
    plan = baseline_plan(gv.TRADE, 2.5, gradient, pending_batches=2, done_batches=2)
    assert plan.decode_batches == 2
    slow = baseline_plan(
        gv.TRADE, 2.5, gradient, pending_batches=2, done_batches=2, observed_batch=500.0
    )
    assert (slow.decision, slow.decode_batches, slow.decode_seconds) == (gv.GRADIENT, 0, 0.0)


def test_a_plan_before_a_baseline_batch_holds_the_baseline_overhead():
    # The trade may stop the decode only if the variant still fits after the baseline's own
    # overhead, (1 + m) o = 12.96 s; after the baseline nothing is held.
    held = 1.35 * 9.6
    gradient = {gv.GRADIENT: 1_000.0}
    governor = gv.Governor(shipped(), end_time=10_000.0, workers=4, clock=Clock(), policy=gv.TRADE)
    shape = {"budget": 0.0, "pending_batches": 2, "test_inputs": 1, "done_batches": 2}
    plan = governor.plan(gradient, 1_000.0 + held, **shape)
    assert (plan.decision, plan.decode_batches, plan.overhead_seconds) == (gv.GRADIENT, 0, held)
    plan = governor.plan(gradient, 1_000.0 + held - 0.01, **shape)
    assert (plan.decision, plan.decode_batches, plan.cut_batches) == (gv.NONE, 2, 0)
    after = governor.plan(gradient, 1_000.0, budget=0.0)
    assert (after.decision, after.overhead_seconds) == (gv.GRADIENT, 0.0)
    assert after.to_record()["overhead_seconds"] == 0.0


# -------------------------------------------------------------------------------------------
# Simulator

END = 41_400.0  # the solver's end time, seconds after the notebook start
# Worker w takes its first puzzle at the solver's launch (25.6 s after the notebook start) plus
# its startup and model load in P1.1-01 (summary.md lines 112 to 115 and 117).
STARTS = (25.6 + 167.0 + 111.2, 25.6 + 217.2 + 60.9, 25.6 + 265.2 + 13.1, 25.6 + 308.8 + 11.3)
DFS_CAP = 540.0  # the reference stops every DFS call after 540 s
DECODE_CAP = 1_200.0  # and checks 1200 s from the puzzle start before each decode batch
GATE_BATCHES = 2  # the held-out gate decodes 8 views (an assumed P2.1 setting)
# Calibration from the P1.1-01 records (reports/P1.1-01/timing-rank*.jsonl, puzzles.csv):
LOG_TOKENS = (12.304, 0.544)  # mean and sd of log T over the 120 puzzles
LOG_BATCH = 3.852  # mean log seconds per decode batch (669 batches)
PUZZLE_SD, BATCH_SD = 0.829, 0.352  # between- and within-puzzle sd of log seconds per batch
TOKENS_RHO = 0.553  # correlation of log train seconds and a puzzle's mean log batch seconds
TRAIN_NOISE = 0.03  # scale of the log noise around the train fit (measured log sd 0.031)
LOG_CANDIDATE = (1.393, 0.576)  # mean and sd of log seconds per rescored candidate (1181)
CANDIDATES = ((0, 181), (1, 136), (2, 148), (3, 110), (4, 63), (5, 21), (6, 8), (7, 2))
INPUTS = ((1, 71), (2, 46), (3, 3))  # test inputs per puzzle (summary.md line 106)
PAIRS = ((2, 3), (3, 4), (4, 2), (5, 1))  # demonstration pairs per puzzle: an assumption


def student_t(rng, dof):
    """Heavy-tailed noise: a Student t draw with ``dof`` degrees of freedom."""
    return rng.gauss(0.0, 1.0) / math.sqrt(rng.gammavariate(dof / 2, 2.0) / dof)


def draw(rng, table):
    return rng.choices([value for value, _ in table], weights=[w for _, w in table])[0]


@dataclass(frozen=True)
class Puzzle:
    pid: str
    tokens: int  # T of the baseline fit
    inputs: int
    pairs: int
    speed: float  # log factor shared by the puzzle's decode batches


def make_puzzles(rng, count):
    puzzles = []
    for index in range(count):
        z = rng.gauss(0.0, 1.0)
        tokens = int(min(128 * 8192, max(50_000, math.exp(LOG_TOKENS[0] + LOG_TOKENS[1] * z))))
        own = student_t(rng, 5) / math.sqrt(5 / 3)
        speed = PUZZLE_SD * (TOKENS_RHO * z + math.sqrt(1 - TOKENS_RHO**2) * own)
        puzzles.append(Puzzle(f"{index:08x}", tokens, draw(rng, INPUTS), draw(rng, PAIRS), speed))
    return puzzles


class TrueCosts:
    """True op seconds: the model's central line times heavy-tailed noise, all scaled by k."""

    def __init__(self, rng, k):
        self.rng, self.k = rng, k

    def train(self, tokens, steps):
        noise = min(math.log(1.6), TRAIN_NOISE * student_t(self.rng, 3))
        fit = 24.35 + 0.0005812 * tokens + 3.487e-08 * tokens * tokens / steps
        return self.k * fit * math.exp(noise)

    def batch(self, puzzle, ratio=1.0):
        noise = BATCH_SD * student_t(self.rng, 4) / math.sqrt(2)
        return min(DFS_CAP, self.k * ratio * math.exp(LOG_BATCH + puzzle.speed + noise))

    def selection(self, ratio=1.0):
        seconds = 0.0
        for _ in range(draw(self.rng, CANDIDATES)):
            log = LOG_CANDIDATE[0] + LOG_CANDIDATE[1] * self.rng.gauss(0.0, 1.0)
            seconds += self.k * ratio * min(20.0, math.exp(log))
        return seconds

    def overhead(self):
        return self.k * self.rng.uniform(5.3, 9.6)  # smallest and largest, lines 32 and 43


@dataclass
class Outcome:
    finish: dict = field(default_factory=dict)  # worker: when its last op ended
    plans: list = field(default_factory=list)  # each puzzle's final plan
    rounds: list = field(default_factory=list)  # (start, end, deadline, estimate) per round
    budgets: list = field(default_factory=list)  # (start, queued, budget) of each puzzle
    # (test inputs, batches run, batches planned, decision cut for, final decision) per cut
    cuts: list = field(default_factory=list)


@functools.cache
def simulate(seed, count, workers=4, policy=gv.REFERENCE, k=1.0, margin=None, reserve=0.0):
    """Four workers (or ``workers``) drain one queue of ``count`` synthetic puzzles.

    The queue is the reference's: the task ids, then one ``None`` per worker, which stops the
    worker that takes it; the governor gets the raw queue size. Each worker runs the reference's
    loop: it takes an item while the queue is not empty and the clock is before the deadline,
    fits (not interruptible), decodes 4 batches per test input
    with the reference's checks (1200 s from the puzzle start and the deadline before each batch;
    a DFS call stops at 540 s and at the deadline), and rescores each batch's candidates. The
    governor then plans one round; a round's ops run to completion, so only the governor keeps
    them inside the deadline. Under ``trade`` the loop asks the governor before each baseline
    batch. True costs are ``k`` times the calibrated ones; the governor keeps its config, with
    ``margin`` and ``reserve`` replaced when given.
    """
    rng = random.Random(seed)
    queue = deque([*make_puzzles(rng, count), *[None] * workers])
    truth = TrueCosts(random.Random(seed + 1_000_003), k)
    cost = model(margin)
    clock = Clock()
    governor = gv.Governor(cost, END, workers, clock=clock, policy=policy, reserve=reserve)
    out = Outcome()
    ready = [(STARTS[w % len(STARTS)], w) for w in range(workers)]
    heapq.heapify(ready)
    while ready:
        now, worker = heapq.heappop(ready)
        if not queue or now > END:
            out.finish[worker] = now
            continue
        puzzle = queue.popleft()
        if puzzle is None:  # this worker's sentinel: it stops
            out.finish[worker] = now
            continue
        clock.now = start = now
        budget = governor.puzzle_budget(len(queue))
        deadline = start + budget
        ids_left = count - len(out.budgets) - 1
        out.budgets.append((start, ids_left, budget))
        clock.now += truth.train(puzzle.tokens, 128)
        # The round's shape, as the loop would count it from the DemoSet: one epoch on the
        # enlarged set (one pair held out, one pseudo-pair per test input), and an in-context
        # prompt that grows by two grids per pseudo-pair.
        shape = {
            "train_tokens": puzzle.tokens * (puzzle.pairs - 1 + puzzle.inputs) // puzzle.pairs,
            "test_batches": 4 * puzzle.inputs,
            "gate_batches": GATE_BATCHES,
            "train_steps": 128,
            "length_ratio": 1 + 2 * puzzle.inputs / (2 * puzzle.pairs + 1),
        }
        planned = 4 * puzzle.inputs
        slowest = cut = None
        for done in range(planned):
            if clock.now - start > DECODE_CAP or clock.now > END:
                break
            if policy == gv.TRADE:
                costs = cost.round_costs(**shape, observed_batch=slowest)
                plan = governor.plan(
                    costs,
                    deadline,
                    budget=budget,
                    pending_batches=planned - done,
                    test_inputs=puzzle.inputs,
                    done_batches=done,
                    observed_batch=slowest,
                )
                if plan.decode_batches == 0:
                    cut = (puzzle.inputs, done, planned, plan.decision)
                    break
            seconds = min(truth.batch(puzzle), END - clock.now)
            slowest = seconds if slowest is None else max(slowest, seconds)
            clock.now += seconds + truth.selection()
        clock.now += truth.overhead()
        costs = cost.round_costs(**shape, observed_batch=slowest)
        plan = governor.plan(costs, deadline, budget=budget, puzzle_id=puzzle.pid)
        out.plans.append(plan)
        if cut is not None:
            out.cuts.append((*cut, plan.decision))
        if plan.decision != gv.NONE:
            begun, ratio = clock.now, 1.0
            if plan.decision == gv.GRADIENT:
                clock.now += truth.train(shape["train_tokens"], shape["train_steps"])
            else:
                ratio = math.sqrt(shape["length_ratio"])  # a DFS step grows less than its prompt
            for _ in range(shape["test_batches"] + shape["gate_batches"]):
                clock.now += truth.batch(puzzle, ratio) + truth.selection(ratio)
            clock.now += truth.overhead()
            out.rounds.append((begun, clock.now, deadline, plan.estimates[plan.decision]))
        heapq.heappush(ready, (clock.now, worker))
    return out


SEEDS = range(30)
SCENARIOS = [(120, gv.REFERENCE), (120, gv.TRADE), (240, gv.REFERENCE), (240, gv.TRADE)]


def all_rounds(count, **kwargs):
    return [r for seed in SEEDS for r in simulate(seed, count, **kwargs).rounds]


@pytest.mark.parametrize(("count", "policy"), SCENARIOS)
def test_simulated_workers_never_pass_the_deadline(count, policy):
    for seed in SEEDS:
        out = simulate(seed, count, policy=policy)
        assert len(out.plans) == count, seed  # every puzzle was taken
        assert max(out.finish.values()) <= END, seed
        assert all(deadline <= END for _, _, deadline, _ in out.rounds), seed
    # A round may end after its own puzzle's deadline (the next budgets absorb it), but rarely.
    rounds = all_rounds(count, policy=policy)
    assert sum(end > deadline for _, end, deadline, _ in rounds) <= 0.01 * len(rounds)


def test_simulated_ladder_is_exercised():
    decisions = Counter(
        plan.decision
        for count, policy in SCENARIOS
        for seed in SEEDS
        for plan in simulate(seed, count, policy=policy).plans
    )
    assert set(decisions) == set(gv.VARIANTS), decisions
    assert decisions[gv.IN_CONTEXT] < decisions[gv.GRADIENT] < decisions[gv.NONE]


def test_simulated_decisions_follow_the_ladder_and_no_room_means_none():
    governor = gv.Governor(shipped(), END, 4, clock=Clock())
    for count, policy in SCENARIOS:
        for seed in SEEDS[:10]:
            for plan in simulate(seed, count, policy=policy).plans:
                assert plan.decision == governor.choose(plan.estimates, plan.remaining)
                if plan.remaining < min(plan.estimates.values()):
                    assert plan.decision == gv.NONE


def test_simulated_last_puzzles_get_all_the_time_left():
    for count, policy in SCENARIOS:
        out = simulate(0, count, policy=policy)
        last = [(start, budget) for start, queued, budget in out.budgets if queued == 0]
        assert len(last) == 1
        start, budget = last[0]
        assert budget == pytest.approx(END - start)


def test_simulated_budgets_shrink_as_the_deadline_nears_when_work_outgrows_the_window():
    # 360 puzzles need more than the 41,400 s window, so time runs short: the last quarter of
    # the puzzles get smaller budgets than the first, and no round runs near the deadline. (With
    # spare time, budgets grow instead, as unused time passes to the later puzzles.)
    for seed in SEEDS[:10]:
        out = simulate(seed, 360)
        budgets = [budget for _, _, budget in out.budgets]
        quarter = len(budgets) // 4
        assert statistics.median(budgets[-quarter:]) < statistics.median(budgets[:quarter])
        assert all(budget <= END - start for start, _, budget in out.budgets)
        late = [p.decision for p in out.plans if p.at > END - 1_800.0]
        assert set(late) <= {gv.NONE}, seed
        assert all(end <= END for _, end, _, _ in out.rounds), seed


def test_simulated_trade_cuts_are_logged_and_keep_the_floor():
    trade = [simulate(seed, 120, policy=gv.TRADE) for seed in SEEDS]
    reference = [simulate(seed, 120) for seed in SEEDS]
    cuts = [cut for out in trade for cut in out.cuts]
    assert cuts and not any(out.cuts for out in reference)
    assert all(done >= 2 * inputs and done < planned for inputs, done, planned, *_ in cuts)

    def rounds(outs):
        return sum(len(out.rounds) for out in outs)

    assert rounds(trade) > rounds(reference)


@pytest.mark.parametrize("k", [1.0, 1.3])
def test_simulated_cuts_are_never_followed_by_none(k):
    # The cut plan holds (1 + m) o for the baseline's overhead after the decode; the simulated
    # overhead is at most 9.6 k s, within (1 + m) o for k <= 1.35, so the variant the views were
    # given up for (or gradient) still fits at the final decision.
    cuts = [cut for seed in SEEDS for cut in simulate(seed, 120, policy=gv.TRADE, k=k).cuts]
    assert cuts
    assert all(final in (cut_for, gv.GRADIENT) for *_, cut_for, final in cuts)


@pytest.mark.parametrize("policy", [gv.REFERENCE, gv.TRADE])
def test_margin_catches_an_optimistic_model(policy):
    # True costs 25% above the model. With the shipped margin no round costs more than its
    # estimate; without a margin some do, and each of those can end past its puzzle deadline.
    over = {}
    for margin in (None, 0.0):
        rounds = all_rounds(120, policy=policy, k=1.25, margin=margin)
        over[margin] = sum(end - begun > estimate for begun, end, _, estimate in rounds)
    assert over[None] == 0
    assert over[0.0] >= 5


@pytest.mark.parametrize(("count", "policy"), [(120, gv.REFERENCE), (200, gv.TRADE)])
def test_shipped_margin_keeps_the_deadline_with_an_optimistic_model(count, policy):
    # True costs 30% above the model; 200 tasks is near the window's capacity at that cost.
    for seed in SEEDS:
        out = simulate(seed, count, policy=policy, k=1.3)
        assert max(out.finish.values()) <= END, seed


# Every seed of 400 at 200 tasks where an optimistic model without a margin passes the deadline;
# with the shipped margin each run stays inside.
NO_MARGIN_PASSES = [
    (gv.TRADE, 1.25, 259),
    (gv.REFERENCE, 1.3, 173),
    (gv.TRADE, 1.3, 87),
    (gv.TRADE, 1.3, 259),
    (gv.TRADE, 1.3, 289),
]


@pytest.mark.parametrize(("policy", "k", "seed"), NO_MARGIN_PASSES)
def test_margin_is_what_keeps_these_runs_inside_the_deadline(policy, k, seed):
    bare = simulate(seed, 200, policy=policy, k=k, margin=0.0)
    assert max(bare.finish.values()) > END
    assert max(simulate(seed, 200, policy=policy, k=k).finish.values()) <= END


def test_one_simulated_worker_keeps_the_deadline():
    for seed in SEEDS[:10]:
        out = simulate(seed, 40, workers=1)
        assert len(out.plans) == 40
        assert max(out.finish.values()) <= END
        assert all(end <= deadline for _, end, deadline, _ in out.rounds)
        # With one worker the share is the whole queue: q queued puzzles plus this one.
        start, queued, budget = out.budgets[0]
        assert budget == pytest.approx((END - start) / (queued + 1))


def test_simulated_rounds_stay_out_of_the_reserve():
    reserve = 1_380.7  # the longest P1.1-01 puzzle (summary.md line 44)
    for count in (120, 240):
        rounds = {}
        for held in (0.0, reserve):
            rounds[held] = 0
            for seed in SEEDS[:10]:
                out = simulate(seed, count, reserve=held)
                assert max(out.finish.values()) <= END
                assert all(end <= END - held for _, end, _, _ in out.rounds), seed
                rounds[held] += len(out.rounds)
        assert 0 < rounds[reserve] < rounds[0.0]
