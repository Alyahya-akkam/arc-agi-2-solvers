"""Acceptance tests of the refinement round and its rollback (task P2.3; BRIEF B5 P2.3).

B5's acceptance: "integration test on DEV4 runs one round end to end; a test injects a wrong
pseudo-pair and asserts rollback restores the previous adapter and version". This file holds the
round's logic with CPU stand-ins for the model: ``arc26.loop.refine`` (one round: the free checks,
the governor's plan, ``promote`` with the provider's fit, keep or roll back), ``run_rounds`` (the
cap, round 2 only after a kept round 1, the test decode after a kept round and the final decode's
two prompt policies), the round's settings on their way to the solver (``Rounds``, ``HoldOut``,
``RunConfig``), the governor's re-decode term, and the records. The provider stand-in trains a
numpy weight vector in place, as the GPU fit trains the adapter, so a rollback must restore it
bit for bit. The end-to-end run on the split solver's worker, with ``arc26.ttt.SGDProvider`` on the
GPU stand-ins, is ``tests/test_rounds_solver.py``. The synthetic tasks are never competition data.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from arc26 import loop, timing
from arc26.governor import GRADIENT, IN_CONTEXT, NONE, CostModel, Governor
from arc26.loop import (
    ALL_PROMOTED,
    ALREADY_PROMOTED,
    BELOW_TAU,
    CONTINUE_STEPS,
    DECLINED,
    DECODE_PREFIX,
    DEFAULT_ROUNDS,
    EPOCH_VIEWS,
    GATE,
    GATE_FAILED,
    GATE_RAISED,
    HELD_OUT_PROMPTS,
    KEPT,
    MAX_ROUNDS,
    NO_HELD_OUT,
    NO_TIME,
    NOT_PROMOTED,
    NOT_RESTORED,
    NOTHING_TO_PROMOTE,
    OUTCOMES,
    PAIRS_PROMPTS,
    PLAN_PREFIX,
    PROMOTED,
    PROMOTION_PREFIX,
    RAISED,
    REDECODE,
    REDECODE_CUT,
    REDECODE_RAISED,
    ROLLED_BACK,
    ROUND,
    ROUND_PREFIX,
    TAU,
    VIEWS_NOT_DISJOINT,
    Candidate,
    DecodeCut,
    Decoded,
    DemoSet,
    HeldOutGate,
    HoldOut,
    OpWatch,
    Pair,
    RoundLog,
    Rounds,
    fresh_views,
    install_candidates,
    read_puzzle,
    refine,
    renumber_candidates,
    run_rounds,
    screen,
)
from arc26.runner import RunConfig, solver_command

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
GOVERNOR = yaml.safe_load((REPO / "configs" / "governor.yaml").read_text(encoding="utf-8"))
MOVED = ("arc26.nvarc.arc_loader",)


@pytest.fixture(autouse=True, scope="module")
def moved_loader():
    """The moved arc_loader imported with a stub transformers (it imports AutoTokenizer for an
    annotation), removed again afterwards, as in tests/test_promote.py."""
    with pytest.MonkeyPatch.context() as patch:
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = type("AutoTokenizer", (), {})
        patch.setitem(sys.modules, "transformers", stub)
        for name in MOVED:
            patch.delitem(sys.modules, name, raising=False)
        yield
        for name in MOVED:
            sys.modules.pop(name, None)


def grid(*rows):
    return tuple(tuple(row) for row in rows)


# A three-pair puzzle with one test input, T0, or two, T0 and T1; the last pair, C, is held out by
# the default rule. Many colors, so fresh views are disjoint from the first decode's at once.
A = Pair(grid([1, 2], [3, 4]), grid([4, 3], [2, 1]))
B = Pair(grid([5, 6, 7]), grid([7, 6, 5]))
C = Pair(
    grid([1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]), grid([2, 2, 1, 1], [4, 4, 3, 3], [6, 6, 5, 5])
)
T0, T1 = grid([8, 9], [9, 8]), grid([3, 8, 1])
Y0, Y1 = grid([9, 8], [8, 9]), grid([1, 8, 3])  # the voted answers
W = grid([0])  # a wrong answer
PUZZLE = "0a0b0c0d"


def task(*tests, pairs=(A, B, C)):
    return {
        "train": [pair.task_pair() for pair in pairs],
        "test": [{"input": [list(row) for row in test]} for test in tests],
    }


def demos(*tests, pairs=(A, B, C), **changes):
    return DemoSet.from_task(PUZZLE, task(*(tests or (T0,)), pairs=pairs), **changes)


def votes(*counts):
    """Synthetic view candidates: ``count`` candidates of each grid, in order."""
    found = []
    for answer, count in counts:
        for _ in range(count):
            index = len(found)
            found.append(Candidate(answer, 0.01 * index, (0.5,) * 8, f"v{index}"))
    return found


AGREE = {T0: votes((Y0, 14), (W, 2)), T1: votes((Y1, 13), (W, 3))}


class Model:
    """The stand-in model: its adapter is a weight vector that the fit trains in place."""

    def __init__(self):
        self.weights = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        self.trained_on = []  # the pairs of every fit, in order


class InPlaceProvider:
    """An adapter provider that trains the model's weights in place, as SGDProvider does: adapt
    keeps a snapshot first, rollback copies it back. ``raises`` fails the fit after it has
    already changed the weights, as an out-of-memory error mid-training would."""

    def __init__(self, model, raises=None, fingerprint_raises=None, steps=CONTINUE_STEPS,
                 copy_raises=None, first_fingerprint_raises=None):  # fmt: skip
        self.model = model
        self.raises = raises
        self.fingerprint_raises = fingerprint_raises
        self.first_fingerprint_raises = first_fingerprint_raises
        self.copy_raises = copy_raises  # {adapt call number: error} for the snapshot's copy
        self.steps = steps
        self.snapshot = None
        self.adapts = []
        self.rollbacks = []
        self.fingerprints = 0

    def adapt(self, state, init, budget_s):
        self.adapts.append((state, init, budget_s))
        # As SGDProvider: the previous round's copy is dropped before the new one is taken.
        self.snapshot = None
        error = (self.copy_raises or {}).get(len(self.adapts))
        if error is not None:
            raise error
        self.snapshot = init.weights.copy()
        # A deterministic step that depends on what the fit saw: the pairs' cells.
        init.weights += np.float32(0.001 * sum(len(p.output) for p in state.pairs) + 0.37)
        init.trained_on.append(state.pairs)
        if self.raises is not None:
            raise self.raises
        return init

    def rollback(self, previous):
        self.rollbacks.append(previous)
        if self.snapshot is not None:
            np.copyto(self.model.weights, self.snapshot)
        return self.model

    def fingerprint(self, adapter):
        self.fingerprints += 1
        if self.first_fingerprint_raises is not None and self.fingerprints == 1:
            raise self.first_fingerprint_raises
        if self.fingerprint_raises is not None and self.fingerprints >= 2:
            raise self.fingerprint_raises
        return hashlib.blake2b(adapter.weights.tobytes(), digest_size=16).hexdigest()

    def fit_shape(self, state):
        return 1000 * len(state.pairs), self.steps


class GateDecode:
    """The gate's decode of the held-out input: the held-out output ``right`` when the adapter's
    fit saw only right pseudo-pairs (``wrong`` names the grids a wrong pseudo-pair would teach),
    a wrong grid otherwise; ``raises`` fails the decode."""

    def __init__(self, wrong=(), raises=None, right=C.output):
        self.wrong = set(wrong)
        self.raises = raises
        self.right = right
        self.calls = []

    def __call__(self, adapter, state, query, views):
        self.calls.append((adapter.weights.copy(), state, query, views))
        if self.raises is not None:
            raise self.raises
        taught = {pair.output for pair in state.pairs if pair.kind == "pseudo"}
        return votes((W if taught & self.wrong else self.right, views))


class Redecode:
    """The re-decode under fresh views: fixed candidates per query; ``cut`` stops it after one of
    its four DFS batches, ``raises`` fails it."""

    def __init__(self, answers=None, cut=False, raises=None):
        self.answers = dict(AGREE if answers is None else answers)
        self.cut = cut
        self.raises = raises
        self.calls = []

    def __call__(self, adapter, state, query, seed):
        self.calls.append((adapter.weights.copy(), state, query, seed))
        if self.raises is not None:
            raise self.raises
        found = list(self.answers[query])
        if self.cut:
            raise DecodeCut(found[:4], 1, 4)
        return found


class Clock:
    """The governor's clock (time.time scale), advanced by hand."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def governor(end=1e7, clock=None, workers=1):
    return Governor.from_mapping(GOVERNOR, end_time=end, workers=workers, clock=clock or Clock())


def run(state=None, *, candidates=None, model=None, provider=None, gate_decode=None,
        redecode=None, gov=None, budget=None, **kwargs):  # fmt: skip
    """One refine call with the stand-ins; returns the result and the stand-ins."""
    state = state if state is not None else demos()
    model = model or Model()
    provider = provider or InPlaceProvider(model)
    gate_decode = gate_decode or GateDecode()
    redecode = redecode or Redecode()
    gov = gov or governor()
    if budget is None:
        budget = gov.puzzle_budget(0)
    if candidates is None:
        candidates = [AGREE[query] for query in state.test_inputs]
    log = kwargs.pop("log", None) or RoundLog(None, "rank0")
    result = refine(
        state,
        model,
        provider,
        HeldOutGate(lambda s: None, gate_decode),
        gov,
        candidates=candidates,
        redecode=redecode,
        deadline=gov.clock() + budget,
        budget=budget,
        log=log,
        **kwargs,
    )
    return result, types.SimpleNamespace(
        model=model, provider=provider, gate=gate_decode, redecode=redecode, log=log, gov=gov
    )


def digest(model):
    return hashlib.blake2b(model.weights.tobytes(), digest_size=16).hexdigest()


# B5 acceptance: one round, and the rollback after a wrong pseudo-pair.


def test_one_round_end_to_end_keeps_the_promoted_pair_and_the_candidate_adapter():
    """Promote, enlarge (version + 1), adapt from the previous adapter, gate, keep."""
    state = demos()
    (kept, adapter), stand = run(state)
    assert kept.version == state.version + 1 and kept is not state
    (pair,) = [p for p in kept.pairs if p.kind == "pseudo"]
    assert (pair.input, pair.output, pair.round_added, pair.test_index) == (T0, Y0, 1, 0)
    assert kept.held_out == state.held_out and state.version == 0  # the old version intact
    # The candidate adapter is the provider's, started from the previous one, and the gate
    # checked it on the enlarged state.
    ((fitted, init, budget_s),) = stand.provider.adapts
    assert adapter is init is stand.model and fitted.pairs[:2] == (A, B)
    assert (fitted.pairs[2].input, fitted.pairs[2].output) == (T0, Y0)
    assert fitted.held_out is None and budget_s > 0
    ((checked, gate_state, query, _),) = stand.gate.calls
    assert np.array_equal(checked, stand.model.weights) and query == C.input
    assert gate_state.pairs[-1].output == Y0
    (record,) = stand.log.rounds
    assert (record.outcome, record.version, record.next_version) == (KEPT, 0, 1)
    assert record.promoted == (0,) and record.screened == (0,) and record.decision == GRADIENT
    assert record.weights_before != record.weights_after == digest(stand.model)
    assert record.restored is None and stand.provider.rollbacks == []
    (plan,) = stand.log.plans
    assert plan["decision"] == GRADIENT and plan["round"] == 1 and plan["puzzle_id"] == PUZZLE
    assert [p.reason for p in stand.log.promotions] == [PROMOTED]


def test_a_wrong_pseudo_pair_is_rolled_back_to_the_previous_adapter_and_version():
    """B5's rollback test: the voted answer W is wrong but confident (14 of 16 views, and the
    re-decode agrees); the candidate adapter trained with it fails the gate, so the state stays
    at its version and the adapter's weights equal the snapshot bit for bit."""
    state = demos()
    model = Model()
    before = model.weights.copy()
    wrong = {T0: votes((W, 14), (Y0, 2))}
    (rolled, adapter), stand = run(
        state,
        model=model,
        candidates=[wrong[T0]],
        redecode=Redecode(wrong),
        gate_decode=GateDecode(wrong={W}),
    )
    assert rolled is state and rolled.version == 0 and rolled.pairs == (A, B)
    assert adapter is model and model.weights.tobytes() == before.tobytes()
    # The fit did train the wrong pair in place; the rollback undid it.
    ((fitted, _, _),) = stand.provider.adapts
    assert fitted.pairs[-1].output == W and stand.provider.rollbacks == [model]
    (record,) = stand.log.rounds
    assert (record.outcome, record.version, record.next_version) == (ROLLED_BACK, 0, 0)
    assert record.restored is True and record.weights_before == record.weights_after
    assert record.weights_after == digest(model) and record.promoted == ()
    (promotion,) = stand.log.promotions
    assert promotion.reason == GATE_FAILED and promotion.gate.passed is False


def test_rollback_restores_the_weights_after_a_fit_that_raised_mid_training():
    model = Model()
    before = model.weights.copy()
    provider = InPlaceProvider(model, raises=MemoryError("CUDA out of memory (stand-in)"))
    (state, adapter), stand = run(model=model, provider=provider)
    assert state.version == 0 and adapter is model
    assert np.array_equal(model.weights, before) and len(provider.rollbacks) == 1
    (record,) = stand.log.rounds
    assert record.outcome == ROLLED_BACK and record.restored is True
    assert stand.log.promotions[0].reason == "fit_raised"
    assert stand.gate.calls == []


# The governor decides first; free checks bound the round.


def test_the_governor_declines_no_round_runs_and_the_plan_is_recorded(tmp_path):
    clock = Clock(1000.0)
    gov = governor(end=1000.0 + 600.0, clock=clock)  # 600 s left: less than one round
    log = RoundLog(tmp_path, "rank3")
    model = Model()
    before = model.weights.copy()
    (state, adapter), stand = run(model=model, gov=gov, log=log)
    assert state.version == 0 and adapter is model and np.array_equal(model.weights, before)
    assert stand.provider.adapts == [] and stand.redecode.calls == [] and stand.gate.calls == []
    (record,) = log.rounds
    assert record.outcome == DECLINED and record.decision == NONE and record.estimate > 600
    lines = (tmp_path / "plan-rank3.jsonl").read_text().splitlines()
    (plan,) = [json.loads(line) for line in lines]
    assert plan["decision"] == NONE and plan["remaining"] == 600.0
    assert plan["shape"]["redecode_batches"] == 4 and plan["shape"]["screened"] == [0]
    assert set(plan["estimates"]) == {GRADIENT}  # the in-context variant is P2.5's
    # The free checks' record, with the voted answer's agreement, is written all the same
    # (ADR-014: P2.8 gets every voted answer's agreement), with nothing past them run.
    lines = (tmp_path / "promote-rank3.jsonl").read_text().splitlines()
    (promotion,) = [json.loads(line) for line in lines]
    assert (promotion["reason"], promotion["agreement"], promotion["votes"]) == (
        DECLINED, 14 / 16, [14, 2]
    )  # fmt: skip
    assert promotion["views"]["common"] == 0 and promotion["redecode"] is None
    assert promotion["gate"] is None and promotion["promoted"] is False


def test_nothing_to_promote_makes_no_plan_and_runs_no_gpu_op():
    (state, adapter), stand = run(candidates=[votes((Y0, 8), (W, 8))])
    assert state.version == 0 and stand.log.plans == [] and stand.provider.adapts == []
    (record,) = stand.log.rounds
    assert record.outcome == NOTHING_TO_PROMOTE and record.screened == ()
    assert [p.reason for p in stand.log.promotions] == ["tied"]
    _, again = run(candidates=[votes((Y0, 8), (W, 3), (Y1, 5))])
    assert [p.reason for p in again.log.promotions] == [BELOW_TAU]


def test_a_puzzle_without_a_held_out_pair_runs_no_round_but_records_its_free_checks():
    state = demos(k2=False, pairs=(A, B))
    assert state.held_out is None
    (same, adapter), stand = run(state)
    assert same is state and stand.redecode.calls == [] and stand.log.plans == []
    assert stand.log.rounds[0].outcome == NO_HELD_OUT
    (promotion,) = stand.log.promotions
    assert promotion.reason == NO_HELD_OUT and promotion.agreement == 14 / 16


def test_the_plan_prices_every_op_of_the_round():
    """Re-decodes (4 batches per screened input), the fit on the state with every screened
    pseudo-pair (the provider's tokens and steps), the gate (4 batches), and the test decode
    after a kept round (4 per test input), at the observed batch."""
    state = demos(T0, T1)
    model = Model()
    gov = governor()
    (_, _), stand = run(state, model=model, gov=gov, observed_batch=200.0)
    (plan,) = stand.log.plans
    shape = plan["shape"]
    # The decode after a kept round decodes the inputs still open: at most one of the two.
    assert shape == {
        "screened": [0, 1], "train_tokens": 4000, "train_steps": CONTINUE_STEPS,
        "test_batches": 4, "gate_batches": 4, "redecode_batches": 8, "observed_batch": 200.0,
    }  # fmt: skip
    expected = gov.cost_model.round_costs(
        train_tokens=4000, train_steps=CONTINUE_STEPS, test_batches=4, gate_batches=4,
        redecode_batches=8, observed_batch=200.0,
    )[GRADIENT]  # fmt: skip
    assert plan["estimates"] == {GRADIENT: round(expected, 3)}
    assert stand.log.rounds[0].estimate == pytest.approx(expected)
    # The caller may price more decodes after the round (run_rounds: the final decode).
    (_, _), more = run(state, model=Model(), gov=gov, observed_batch=200.0, decode_batches=12)
    assert more.log.plans[0]["shape"]["test_batches"] == 12
    with pytest.raises(ValueError, match="decode_batches must be at least 0"):
        run(state, decode_batches=-4)
    for bad in (4.0, True):
        with pytest.raises(TypeError, match="decode_batches must be None or an integer"):
            run(state, decode_batches=bad)


def test_the_provider_gets_the_time_left_to_the_deadline_as_its_budget():
    clock = Clock(5000.0)
    gov = governor(end=5000.0 + 1e6, clock=clock)
    (_, _), stand = run(gov=gov, budget=10_000.0)
    ((_, _, budget_s),) = stand.provider.adapts
    assert budget_s == 10_000.0


# Checks that fail: a cut re-decode, a cut gate, raising ops, and errors outside promote.


def test_a_redecode_cut_by_the_time_cap_fails_with_its_own_reason():
    """Before P2.3 a re-decode that the 1200 s cap stopped after one batch still agreed on its
    top grid alone; now it fails as redecode_cut, and nothing is fit."""
    (state, _), stand = run(redecode=Redecode(cut=True))
    (promotion,) = stand.log.promotions
    assert promotion.reason == REDECODE_CUT and promotion.redecode.cut is True
    assert promotion.redecode.agreed is False and promotion.redecode.candidates == 4
    assert promotion.redecode.expected_rank == 1  # its top grid was the voted answer
    assert stand.provider.adapts == [] and stand.log.rounds[0].outcome == NOT_PROMOTED
    assert state.version == 0


def test_a_gate_decode_cut_by_the_time_cap_fails_the_gate_and_rolls_back():
    model = Model()
    before = model.weights.copy()
    cut = GateDecode(raises=DecodeCut(votes((C.output, 4)), 1, 4))
    (state, _), stand = run(model=model, gate_decode=cut)
    (promotion,) = stand.log.promotions
    assert promotion.reason == GATE_RAISED and "DecodeCut: 1 of 4 DFS batches" in (
        promotion.gate.error
    )
    assert state.version == 0 and np.array_equal(model.weights, before)
    assert stand.log.rounds[0].outcome == ROLLED_BACK and stand.log.rounds[0].restored


def test_a_raising_redecode_fails_its_check_and_the_round_ends_unpromoted():
    (state, adapter), stand = run(redecode=Redecode(raises=MemoryError("stand-in")))
    assert stand.log.promotions[0].reason == REDECODE_RAISED
    assert state.version == 0 and stand.log.rounds[0].outcome == NOT_PROMOTED
    assert stand.provider.rollbacks == []


def test_an_error_outside_the_checks_is_rolled_back_and_recorded():
    """The fingerprint of the candidate adapter raises after the fit has trained in place: the
    round is recorded as raised, the weights are restored, and the caller goes on."""
    model = Model()
    before = model.weights.copy()
    provider = InPlaceProvider(model, fingerprint_raises=RuntimeError("stand-in"))
    (state, adapter), stand = run(model=model, provider=provider)
    assert state.version == 0 and adapter is model and np.array_equal(model.weights, before)
    (record,) = stand.log.rounds
    assert record.outcome == RAISED and record.error == "RuntimeError: stand-in"
    # The restore ran, but its fingerprint raised too, so the record cannot show it restored.
    assert record.weights_before is not None and record.weights_after is None
    assert record.restored is False and len(provider.rollbacks) == 1


def test_a_fingerprint_that_raises_before_the_fit_rolls_nothing_back():
    """The adapter is fingerprinted before the fit can change it: when that raises, no fit ran,
    so no rollback runs either (it could only load an older copy)."""
    model = Model()
    before = model.weights.copy()
    provider = InPlaceProvider(model, first_fingerprint_raises=RuntimeError("stand-in"))
    (state, adapter), stand = run(model=model, provider=provider)
    assert state.version == 0 and adapter is model and np.array_equal(model.weights, before)
    assert provider.adapts == [] and provider.rollbacks == []
    (promotion,) = stand.log.promotions
    assert promotion.reason == "fit_raised" and "stand-in" in promotion.error
    (record,) = stand.log.rounds
    assert record.outcome == NOT_PROMOTED and record.weights_before is None
    assert record.restored is None


def test_refine_refuses_bad_arguments():
    with pytest.raises(TypeError, match="AdapterProvider"):
        run(provider=object())
    with pytest.raises(ValueError, match="at most 2"):
        run(round_added=3)
    with pytest.raises(TypeError, match="Governor"):
        refine(
            demos(), Model(), InPlaceProvider(Model()), HeldOutGate(lambda s: None, GateDecode()),
            object(), candidates=[AGREE[T0]], redecode=Redecode(), deadline=1.0, budget=1.0,
        )  # fmt: skip
    with pytest.raises(ValueError, match="finite"):
        run(budget=float("inf"))


# Edge cases: k = 2, two test inputs, 30 by 30 grids, the round's span.


def test_k2_holds_one_pair_out_and_fits_the_other_with_the_pseudo_pair():
    state = demos(pairs=(A, B))
    assert state.held_out is not None and state.pairs == (A,)
    gate = GateDecode(right=B.output)
    (kept, _), stand = run(state, gate_decode=gate)
    ((fitted, _, _),) = stand.provider.adapts
    assert [p.kind for p in fitted.pairs] == ["real", "pseudo"] and kept.version == 1
    assert all(B.input != p.input for p in fitted.pairs)


def test_two_test_inputs_share_one_candidate_fit():
    (kept, _), stand = run(demos(T0, T1))
    ((fitted, _, _),) = stand.provider.adapts
    assert [p.test_index for p in fitted.pairs if p.kind == "pseudo"] == [0, 1]
    assert stand.log.rounds[0].promoted == (0, 1) and kept.version == 1


def test_30_by_30_grids_go_through_a_round():
    big = grid(*[[(r + c) % 9 + 1 for c in range(30)] for r in range(30)])
    query = grid(*[[(r + 2 * c) % 9 + 1 for c in range(30)] for r in range(30)])
    answer = grid(*[[(r * c) % 9 + 1 for c in range(30)] for r in range(30)])
    state = DemoSet.from_task(PUZZLE, task(query, pairs=(A, Pair(answer, answer), Pair(big, big))))
    (kept, _), stand = run(
        state, candidates=[votes((answer, 16))], redecode=Redecode({query: votes((answer, 16))}),
        gate_decode=GateDecode(right=big),
    )  # fmt: skip
    assert stand.log.promotions[0].reason == PROMOTED and stand.log.rounds[0].outcome == KEPT
    assert kept.version == 1 and kept.pairs[-1].output == answer


def test_the_round_records_its_span_only_when_its_promotion_ran(tmp_path):
    logger = timing.TimingLogger(tmp_path, "rank0")
    run(timing_logger=logger)
    run(candidates=[votes((Y0, 8), (W, 8))], timing_logger=logger)  # nothing to promote
    records, skipped = timing.read_records([logger.path])
    assert skipped == 0
    assert [r.op for r in records] == [REDECODE, GATE, ROUND]
    assert records[-1].seconds >= sum(r.seconds for r in records[:-1]) - 1e-6


# run_rounds: the cap, round 2 after a kept round 1, the test decodes after them.


class Decoder:
    """decode_tests and install: records each decode, answers with fixed candidates per test
    input, and may be cut or raise."""

    def __init__(self, answers=None, cut=False, raises=None):
        self.answers = answers or {T0: votes((Y0, 14), (W, 2)), T1: votes((Y1, 14), (W, 2))}
        self.cut = cut
        self.raises = raises
        self.decodes = []
        self.installs = []

    def decode_tests(self, adapter, state, name, held_out, inputs):
        self.decodes.append((adapter.weights.copy(), state, name, held_out, inputs))
        if self.raises is not None:
            raise self.raises
        found = tuple(
            tuple(self.answers[q]) if i in inputs else () for i, q in enumerate(state.test_inputs)
        )
        expected = 4 * len(inputs)
        return Decoded(found, 1 if self.cut else expected, expected, slowest=150.0)

    def install(self, name, indices):
        self.installs.append((name, tuple(indices)))
        return tuple(indices)


def rounds(**changes):
    return Rounds(round_dir="/kaggle/round_outputs/arm", governor=GOVERNOR, **changes)


def drive(state, settings, *, model=None, gate_decode=None, redecode=None, decoder=None,
          candidates=None, gov=None, budget=None, provider=None):  # fmt: skip
    model = model or Model()
    provider = provider or InPlaceProvider(model)
    gate_decode = gate_decode or GateDecode()
    redecode = redecode or Redecode()
    decoder = decoder or Decoder()
    gov = gov or governor()
    budget = gov.puzzle_budget(0) if budget is None else budget
    log = RoundLog(None, "rank0")
    gates, redecodes = [], []

    def make_gate(round_added):
        gates.append(round_added)
        return HeldOutGate(lambda s: None, gate_decode)

    def make_redecode(round_added):
        redecodes.append(round_added)
        return redecode

    result = run_rounds(
        state,
        model,
        settings,
        provider=provider,
        governor=gov,
        candidates=candidates or [AGREE[q] for q in state.test_inputs],
        decode_tests=decoder.decode_tests,
        install=decoder.install,
        make_gate=make_gate,
        make_redecode=make_redecode,
        deadline=gov.clock() + budget,
        budget=budget,
        log=log,
    )
    return result, types.SimpleNamespace(
        model=model, provider=provider, decoder=decoder, log=log, gates=gates,
        redecodes=redecodes, redecode=redecode,
    )  # fmt: skip


def test_max_rounds_defaults_to_one_and_is_capped_at_two():
    assert DEFAULT_ROUNDS == rounds().max_rounds == 1 and MAX_ROUNDS == 2
    with pytest.raises(ValueError, match="max_rounds must be 1 to 2"):
        rounds(max_rounds=3)
    with pytest.raises(ValueError, match="max_rounds must be 1 to 2"):
        rounds(max_rounds=0)
    (state, _), stand = drive(demos(T0, T1), rounds(), candidates=[AGREE[T0], votes((Y1, 4))])
    assert [r.round_added for r in stand.log.rounds] == [1] and stand.gates == [1]


def test_after_a_kept_round_the_kept_adapter_decodes_the_test_inputs():
    """Input 0 is promoted, input 1 is below tau: the kept adapter decodes input 1 alone with the
    enlarged set in the prompts, and its candidates are installed; input 0 keeps the candidates
    that promoted it and is not decoded again."""
    state = demos(T0, T1)
    (kept, adapter), stand = drive(
        state, rounds(final_in_prompts=False), candidates=[AGREE[T0], votes((Y1, 4))]
    )
    assert kept.version == 1 and stand.log.rounds[0].promoted == (0,)
    ((weights, decoded_state, name, held_out, inputs),) = stand.decoder.decodes
    assert name == "round1" and held_out is False and decoded_state is kept and inputs == (1,)
    assert np.array_equal(weights, adapter.weights)  # the candidate adapter, not round 0's
    assert stand.decoder.installs == [("round1", (1,))]
    (record,) = stand.log.decodes
    assert (record.inputs, record.installed, record.prompts) == ((1,), (1,), PAIRS_PROMPTS)
    assert (record.batches, record.expected, record.cut) == (4, 4, False)


@pytest.mark.parametrize(("final", "name"), [(True, "final"), (False, "round1")])
def test_every_input_promoted_skips_the_decode_after_the_round(final, name):
    (kept, _), stand = drive(demos(), rounds(final_in_prompts=final))
    assert kept.version == 1 and stand.decoder.decodes == []
    (record,) = stand.log.decodes
    assert (record.name, record.skipped, record.inputs) == (name, ALL_PROMOTED, ())


def test_round_two_runs_only_after_a_kept_round_one_on_its_decodes_candidates():
    state = demos(T0, T1)
    decoder = Decoder(answers={T0: votes((Y0, 16)), T1: votes((Y1, 15), (W, 1))})
    (final, _), stand = drive(
        state, rounds(max_rounds=2, final_in_prompts=False), decoder=decoder,
        candidates=[AGREE[T0], votes((Y1, 4))],
    )  # fmt: skip
    assert [r.outcome for r in stand.log.rounds] == [KEPT, KEPT]
    assert final.version == 2 and stand.gates == [1, 2]
    # Round 2 promoted input 1 from the round-1 decode's candidates (15 of 16 views).
    assert [p.test_index for p in final.pairs if p.kind == "pseudo"] == [0, 1]
    round2 = [p for p in stand.log.promotions if p.round_added == 2]
    assert [p.reason for p in round2] == ["already_promoted", PROMOTED]
    assert round2[1].votes == (15, 1)
    # After round 2 every input is a pseudo-pair: no decode follows it.
    assert [d.name for d in stand.log.decodes] == ["round1", "round2"]
    assert stand.log.decodes[-1].skipped == ALL_PROMOTED


def test_a_failed_copy_in_round_two_keeps_round_ones_kept_adapter():
    """Round 1 is kept; round 2's copy of the adapter raises (host memory, say) after round 1's
    copy was dropped, so its rollback restores nothing: the weights stay round 1's kept ones,
    never round 0's."""
    model = Model()
    provider = InPlaceProvider(model, copy_raises={2: MemoryError("stand-in host memory")})
    decoder = Decoder(answers={T0: votes((Y0, 16)), T1: votes((Y1, 15), (W, 1))})
    (state, adapter), stand = drive(
        demos(T0, T1), rounds(max_rounds=2), model=model, provider=provider, decoder=decoder,
        candidates=[AGREE[T0], votes((Y1, 4))],
    )  # fmt: skip
    first, second = stand.log.rounds
    assert (first.outcome, second.outcome) == (KEPT, ROLLED_BACK)
    assert state.version == 1 and adapter is model and len(provider.rollbacks) == 1
    # The weights after round 2 are round 1's kept ones, not round 0's.
    assert second.weights_before == first.weights_after == second.weights_after == digest(model)
    assert digest(model) != first.weights_before and second.restored is True
    round2 = [p for p in stand.log.promotions if p.round_added == 2]
    assert [p.reason for p in round2] == [ALREADY_PROMOTED, "fit_raised"]


def test_no_promotion_ends_the_rounds():
    gate = GateDecode(wrong={Y0})
    model = Model()
    before = model.weights.copy()
    (state, _), stand = drive(
        demos(T0, T1), rounds(max_rounds=2, final_in_prompts=False), model=model,
        gate_decode=gate,
    )  # fmt: skip
    assert [r.outcome for r in stand.log.rounds] == [ROLLED_BACK]
    assert state.version == 0 and stand.decoder.decodes == [] and stand.gates == [1]
    assert np.array_equal(model.weights, before)


def test_a_declined_round_ends_the_rounds():
    gov = governor(end=1000.0 + 600.0, clock=Clock(1000.0))
    (state, _), stand = drive(demos(T0, T1), rounds(max_rounds=2), gov=gov)
    assert [r.outcome for r in stand.log.rounds] == [DECLINED] and stand.decoder.decodes == []


@pytest.mark.parametrize(
    ("decoder", "field"),
    [(Decoder(cut=True), "cut"), (Decoder(raises=MemoryError("stand-in")), "error")],
    ids=["cut", "raises"],
)
def test_a_cut_or_raising_decode_after_the_round_replaces_nothing(decoder, field):
    (kept, _), stand = drive(
        demos(T0, T1), rounds(max_rounds=2, final_in_prompts=False), decoder=decoder,
        candidates=[AGREE[T0], votes((Y1, 4))],
    )  # fmt: skip
    assert kept.version == 1 and decoder.installs == []
    (record,) = stand.log.decodes
    assert record.installed == () and getattr(record, field)
    # The rounds end: no round 2 on a decode that did not finish.
    assert [r.round_added for r in stand.log.rounds] == [1]


# The final decode's prompt policy: the held-out pair back by default (ADR-013 as amended after
# P2.1-02's Recovers row); False keeps P2.1-01's prompts.


def test_the_pairs_policy_submits_the_kept_rounds_decode_without_the_held_out_pair():
    (kept, _), stand = drive(
        demos(T0, T1), rounds(final_in_prompts=False), candidates=[AGREE[T0], votes((Y1, 4))]
    )
    assert [(d[2], d[3], d[4]) for d in stand.decoder.decodes] == [("round1", False, (1,))]
    assert kept.task()["train"] == [A.task_pair(), B.task_pair(), kept.pairs[-1].task_pair()]


def test_final_in_prompts_is_the_default_and_decodes_with_the_held_out_pair_in_the_prompts():
    assert rounds().final_in_prompts is True
    (kept, adapter), stand = drive(
        demos(T0, T1), rounds(), candidates=[AGREE[T0], votes((Y1, 4))]
    )
    # The last round decodes nothing itself; the final decode of the open input follows it.
    ((weights, state, name, held_out, inputs),) = stand.decoder.decodes
    assert (name, held_out, state, inputs) == ("final", True, kept, (1,))
    assert np.array_equal(weights, adapter.weights)
    (record,) = stand.log.decodes
    assert record.prompts == HELD_OUT_PROMPTS and record.installed == (1,)
    assert record.estimate > 0 and record.remaining > record.estimate
    # Its prompts: the real pairs in the file's order, the held-out one back at its place, then
    # the pseudo-pair.
    assert [p["input"] for p in state.task(held_out=True)["train"]] == [
        [list(r) for r in g] for g in (A.input, B.input, C.input, T0)
    ]


def test_final_in_prompts_after_a_rolled_back_round_uses_round_0s_restored_adapter():
    model = Model()
    before = model.weights.copy()
    (state, _), stand = drive(
        demos(T0, T1), rounds(final_in_prompts=True), model=model,
        gate_decode=GateDecode(wrong={Y0}),
    )  # fmt: skip
    assert state.version == 0 and stand.log.rounds[0].outcome == ROLLED_BACK
    ((weights, decoded, name, held_out, inputs),) = stand.decoder.decodes
    assert (name, held_out, inputs) == ("final", True, (0, 1)) and np.array_equal(weights, before)
    assert stand.log.decodes[0].installed == (0, 1)


def test_final_in_prompts_without_the_time_for_it_is_skipped():
    clock = Clock(1000.0)
    gov = governor(end=1000.0 + 600.0, clock=clock)
    (_, _), stand = drive(demos(T0, T1), rounds(final_in_prompts=True), gov=gov)
    (record,) = stand.log.decodes
    assert record.skipped == NO_TIME and stand.decoder.decodes == []
    assert record.estimate > record.remaining == 600.0


@pytest.mark.parametrize(
    ("decoder", "field"),
    [(Decoder(cut=True), "cut"), (Decoder(raises=MemoryError("stand-in")), "error")],
    ids=["cut", "raises"],
)
def test_a_cut_or_raising_final_decode_replaces_nothing(decoder, field):
    (kept, _), stand = drive(
        demos(T0, T1), rounds(), decoder=decoder, candidates=[AGREE[T0], votes((Y1, 4))]
    )
    assert kept.version == 1 and decoder.installs == []
    (record,) = stand.log.decodes
    assert record.name == "final" and record.installed == () and getattr(record, field)


class LossyProvider(InPlaceProvider):
    """A provider whose rollback leaves the candidate's weights in place, as a restore that
    missed a tensor would."""

    def rollback(self, previous):
        self.rollbacks.append(previous)
        return self.model


def test_no_final_decode_after_a_rollback_that_did_not_restore_the_weights():
    """The weights are then neither the kept adapter's nor round 0's: the open inputs keep the
    candidates they have, and the record says why."""
    model = Model()
    (state, _), stand = drive(
        demos(T0, T1), rounds(), model=model, provider=LossyProvider(model),
        gate_decode=GateDecode(wrong={Y0}),
    )  # fmt: skip
    (record,) = stand.log.rounds
    assert (record.outcome, record.restored) == (ROLLED_BACK, False)
    (decode,) = stand.log.decodes
    assert (decode.name, decode.skipped, decode.inputs) == ("final", NOT_RESTORED, (0, 1))
    assert stand.decoder.decodes == [] and stand.decoder.installs == []


def test_with_two_rounds_the_final_decode_follows_the_kept_rounds_own_decode():
    """Round 1 is kept and decodes the open input without the pair, which feeds round 2's
    promotion; round 2 has nothing to promote, and the final decode follows with the pair."""
    decoder = Decoder(answers={T1: votes((Y1, 4))})
    (state, adapter), stand = drive(
        demos(T0, T1), rounds(max_rounds=2), decoder=decoder,
        candidates=[AGREE[T0], votes((Y1, 4))],
    )  # fmt: skip
    assert [r.outcome for r in stand.log.rounds] == [KEPT, NOTHING_TO_PROMOTE]
    assert [(d[2], d[3], d[4]) for d in decoder.decodes] == [
        ("round1", False, (1,)), ("final", True, (1,))
    ]  # fmt: skip
    assert [d.prompts for d in stand.log.decodes] == [PAIRS_PROMPTS, HELD_OUT_PROMPTS]
    assert decoder.decodes[-1][1] is state and state.version == 1


@pytest.mark.parametrize(
    ("changes", "batches"),
    [({}, [8]), ({"final_in_prompts": False}, [4]), ({"max_rounds": 2}, [8, 4]),
     ({"max_rounds": 2, "final_in_prompts": False}, [4, 0])],
    ids=["final", "pairs", "two-final", "two-pairs"],
)  # fmt: skip
def test_each_rounds_plan_reserves_the_test_decodes_that_may_follow_it(changes, batches):
    """With the final decode, the last round's plan prices it for every open input (a round that
    keeps nothing leaves them all open); without it, the decode after a kept round, of the inputs
    still open after it. Round 2 here sees one open input."""
    (_, _), stand = drive(
        demos(T0, T1), rounds(**changes), candidates=[AGREE[T0], votes((Y1, 4))]
    )
    assert [plan["shape"]["test_batches"] for plan in stand.log.plans] == batches


def test_before_the_last_round_the_plan_also_reserves_the_rounds_own_decode():
    """Three open inputs, two rounds: a kept round 1 decodes the two still open, and the final
    decode may decode them again (16 batches); a round 1 that keeps nothing leaves the final
    decode of all three (12)."""
    t2, y2 = grid([4, 4], [8, 1]), grid([1, 8], [4, 4])
    decoder = Decoder(answers={T1: votes((Y1, 4)), t2: votes((y2, 4))})
    (_, _), stand = drive(
        demos(T0, T1, t2), rounds(max_rounds=2), decoder=decoder,
        candidates=[AGREE[T0], votes((Y1, 4)), votes((y2, 4))],
    )  # fmt: skip
    assert [plan["shape"]["test_batches"] for plan in stand.log.plans] == [16]
    assert [d.name for d in stand.log.decodes] == ["round1", "final"]


# The settings' path to the solver.


def test_rounds_round_trip_through_the_hold_out_settings_and_the_solver_json():
    settings = HoldOut(
        gate_dir="/kaggle/gate_outputs/a",
        log_dir="/kaggle/run_logs/a/gate",
        rounds={"round_dir": "/kaggle/round_outputs/a", "governor": GOVERNOR},
    )
    assert isinstance(settings.rounds, Rounds)
    assert (settings.rounds.max_rounds, settings.rounds.tau) == (1, TAU)
    assert (settings.rounds.continue_steps, settings.rounds.final_in_prompts) == (32, True)
    payload = json.loads(json.dumps(settings.to_json()))
    assert list(payload) == [*HoldOut.FIELDS, "rounds"]
    assert list(payload["rounds"]) == list(Rounds.FIELDS)
    assert payload["rounds"]["governor"] == GOVERNOR
    assert HoldOut.from_mapping(payload) == settings and hash(HoldOut.from_mapping(payload))
    cfg = RunConfig(data_dir="/d", model_dir="/m", holdout=payload)
    assert json.loads(solver_command(cfg, 1.0, "python")[6])["holdout"] == payload


def test_without_rounds_the_solver_json_is_task_p2_1s():
    plain = HoldOut(gate_dir="/g", log_dir="/l")
    assert plain.rounds is None and list(plain.to_json()) == list(HoldOut.FIELDS)
    cfg = RunConfig(data_dir="/d", model_dir="/m", holdout=plain)
    assert "rounds" not in json.loads(solver_command(cfg, 1.0, "python")[6])["holdout"]


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"max_rounds": 3}, ValueError, "max_rounds must be 1 to 2"),
        ({"max_rounds": 1.0}, TypeError, "max_rounds must be an integer"),
        ({"continue_steps": 0}, ValueError, "continue_steps must be 1 to 128"),
        ({"continue_steps": EPOCH_VIEWS + 1}, ValueError, "continue_steps must be 1 to 128"),
        ({"tau": 1.5}, ValueError, "tau must be in"),
        ({"tau": True}, TypeError, "tau must be a number"),
        ({"final_in_prompts": "no"}, TypeError, "final_in_prompts must be true or false"),
        ({"round_dir": ""}, ValueError, "round_dir must not be empty"),
        ({"governor": {**GOVERNOR, "decode_policy": {"mode": "trade"}}}, ValueError,
         "the mode must be 'reference'"),
        ({"governor": {k: v for k, v in GOVERNOR.items() if k != "margin"}}, ValueError,
         "missing key"),
        ({"governor": "configs/governor.yaml"}, TypeError, "governor must be a mapping"),
        ({"governor": {**GOVERNOR, "margin": float("nan")}}, TypeError, "JSON values"),
    ],
)  # fmt: skip
def test_bad_round_settings_are_refused(changes, error, message):
    settings = {"round_dir": "/r", "governor": GOVERNOR, **changes}
    with pytest.raises(error, match=message):
        Rounds.from_mapping(settings)


def test_round_settings_refuse_unknown_and_missing_keys_and_the_pair_in_test_prompts():
    with pytest.raises(ValueError, match="unknown keys"):
        Rounds.from_mapping({"round_dir": "/r", "governor": GOVERNOR, "views": 8})
    with pytest.raises(ValueError, match="rounds needs governor"):
        Rounds.from_mapping({"round_dir": "/r"})
    with pytest.raises(ValueError, match="rounds need in_test_prompts False"):
        HoldOut(gate_dir="/g", in_test_prompts=True, rounds=rounds())
    with pytest.raises(TypeError, match="rounds must be None or Rounds"):
        HoldOut(gate_dir="/g", rounds=["round_dir"])


def test_the_round_folder_lies_outside_the_candidate_folder():
    inside = {"round_dir": "/kaggle/inference_outputs/rounds", "governor": GOVERNOR}
    with pytest.raises(ValueError, match="must lie outside outputs_dir"):
        RunConfig(data_dir="/d", model_dir="/m", holdout={"gate_dir": "/g", "rounds": inside})
    beside = {"round_dir": "/kaggle/round_outputs", "governor": GOVERNOR}
    assert RunConfig(data_dir="/d", model_dir="/m", holdout={"gate_dir": "/g", "rounds": beside})


# The governor's re-decode term.


def test_round_costs_price_the_redecode_batches_at_the_round_0_prompt_length():
    model = CostModel.from_mapping(GOVERNOR)
    base = model.round_costs(train_tokens=50_000, test_batches=8, gate_batches=4)
    assert model.round_costs(
        train_tokens=50_000, test_batches=8, gate_batches=4, redecode_batches=0
    ) == base  # fmt: skip
    more = model.round_costs(
        train_tokens=50_000, test_batches=8, gate_batches=4, redecode_batches=8,
        length_ratio=1.5, observed_batch=300.0,
    )  # fmt: skip
    again = model.decode_cost(8, 1.0, 300.0)
    shorter = model.round_costs(
        train_tokens=50_000, test_batches=8, gate_batches=4, length_ratio=1.5, observed_batch=300.0
    )  # fmt: skip
    assert more[GRADIENT] == pytest.approx(shorter[GRADIENT] + again)
    assert more[IN_CONTEXT] == pytest.approx(shorter[IN_CONTEXT] + again)
    with pytest.raises(ValueError, match="redecode_batches"):
        model.round_costs(train_tokens=0, test_batches=4, redecode_batches=-4)


# The state, the records, and the helpers the solver uses.


def test_the_held_out_pair_goes_back_only_where_asked():
    state = demos(T0)
    assert state.task()["train"] == [A.task_pair(), B.task_pair()]
    assert state.task(held_out=True)["train"] == [A.task_pair(), B.task_pair(), C.task_pair()]
    first = demos(T0, rule="first")
    assert first.task(held_out=True)["train"] == [A.task_pair(), B.task_pair(), C.task_pair()]
    assert first.task()["train"] == [B.task_pair(), C.task_pair()]
    with pytest.raises(ValueError, match="no held-out pair"):
        demos(T0, k2=False, pairs=(A, B)).task(held_out=True)


def test_fresh_views_follow_the_layout_of_the_decode_that_produced_the_candidates():
    """After a kept round only the open inputs are decoded, in one call, and an input's views
    depend on the inputs decoded before it. Decoded alone at the reference's seed, T1's 16
    decoding views are exactly those of a re-decode of T1 alone at that seed (their 8 rescoring
    views differ, seeded from view_seed); decoded second, after T0, they are others. The check
    therefore takes the first decode's layout."""
    state = demos(T0, T1)
    alone = fresh_views(state, 1, first_inputs=(1,), fresh_seed=2, tries=1)
    assert alone.common == 16 and not alone.disjoint
    together = fresh_views(state, 1, fresh_seed=2, tries=1)
    assert together.common < alone.common
    assert fresh_views(state, 1, first_inputs=(1,)).disjoint
    with pytest.raises(ValueError, match="was not decoded by the first decode"):
        fresh_views(state, 0, first_inputs=(1,))
    for bad, error in (
        ((1, 0), ValueError), ((), ValueError), ((2,), ValueError), (("1",), TypeError),
        ("1", TypeError),
    ):  # fmt: skip
        with pytest.raises(error):
            fresh_views(state, 1, first_inputs=bad)
    # screen takes the layout too: input 1, decoded alone at seed 2, shares every view with a
    # re-decode at seed 2; input 0, below tau, never reaches the check. An input that passes the
    # free checks but was not decoded by the first decode is refused.
    first, second = screen(
        [votes((Y0, 4)), AGREE[T1]], state, first_inputs=(1,), fresh_seed=2, tries=1
    )
    assert (first.record.reason, second.record.reason) == (BELOW_TAU, VIEWS_NOT_DISJOINT)
    with pytest.raises(ValueError, match="was not decoded by the first decode"):
        screen([AGREE[T0], AGREE[T1]], state, first_inputs=(1,))


def test_renumbering_names_a_decodes_files_by_the_puzzles_test_indices(tmp_path):
    """A decode of the open inputs (1, 2) writes files for positions 0 and 1; they are renamed to
    the puzzle's indices, the last position first, so no name is taken twice."""
    write_candidates(tmp_path, {
        "0a0b0c0d_0.permute0123456789": [[[1]]],
        "0a0b0c0d_0.rot90.permute0123456789": [[[1]]],
        "0a0b0c0d_1.permute0123456789": [[[2]]],
    })  # fmt: skip
    renumber_candidates(tmp_path, "0a0b0c0d", (1, 2))
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "0a0b0c0d_1.permute0123456789", "0a0b0c0d_1.rot90.permute0123456789",
        "0a0b0c0d_2.permute0123456789",
    ]  # fmt: skip
    none, first, second = read_puzzle(tmp_path, "0a0b0c0d", 3)
    assert none == () and [c.grid for c in first] == [grid([1]), grid([1])]
    assert [c.grid for c in second] == [grid([2])]
    renumber_candidates(tmp_path, "0a0b0c0d", (0, 1, 2))  # every input: the names stay
    assert len(list(tmp_path.iterdir())) == 3


def test_screen_runs_the_free_checks_of_promote():
    state = demos(T0, T1)
    screened = screen([AGREE[T0], votes((Y1, 4))], state)
    assert [item.passed for item in screened] == [True, False]
    assert screened[0].check.disjoint and screened[1].record.reason == BELOW_TAU


def test_the_round_records_go_to_one_file_per_kind_beside_the_verdicts(tmp_path):
    log = RoundLog(tmp_path, "rank1")
    run(demos(T0, T1), log=log)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["plan-rank1.jsonl", "promote-rank1.jsonl", "round-rank1.jsonl"]
    assert log.path(PLAN_PREFIX) == tmp_path / "plan-rank1.jsonl"
    assert log.path(PROMOTION_PREFIX).name == "promote-rank1.jsonl"
    assert log.path(ROUND_PREFIX).name == "round-rank1.jsonl"
    assert log.path(DECODE_PREFIX).name == "decode-rank1.jsonl"
    lines = (tmp_path / "round-rank1.jsonl").read_text().splitlines()
    (round_line,) = [json.loads(line) for line in lines]
    assert set(round_line) == {
        "puzzle_id", "round", "version", "next_version", "outcome", "screened", "promoted",
        "decision", "estimate", "remaining", "weights_before", "weights_after", "restored",
        "seconds", "error",
    }  # fmt: skip
    assert round_line["outcome"] in OUTCOMES
    promotions = (tmp_path / "promote-rank1.jsonl").read_text().splitlines()
    assert [json.loads(x)["test_index"] for x in promotions] == [0, 1]
    assert RoundLog(None, "rank0").path(PLAN_PREFIX) is None


def test_the_op_watch_counts_and_renames_what_it_forwards(tmp_path):
    logger = timing.TimingLogger(tmp_path, "rank0")
    watch = OpWatch(logger, loop.ROUND_OPS)
    watch.record("0a0b0c0d", timing.DECODE, 2.5, 40)
    watch.record("0a0b0c0d", timing.DECODE, 7.25, 40)
    watch.record("0a0b0c0d", timing.SELECTION, 1.0, 80)
    assert watch.counts == {timing.DECODE: 2, timing.SELECTION: 1}
    assert watch.slowest[timing.DECODE] == 7.25
    records, _ = timing.read_records([logger.path])
    assert [r.op for r in records] == ["round_decode", "round_decode", "round_selection"]
    quiet = OpWatch(None)
    quiet.record(None, timing.DECODE, 1.0)
    assert quiet.counts == {timing.DECODE: 1}
    with pytest.raises(ValueError):
        quiet.record(None, timing.DECODE, -1.0)
    assert quiet.counts == {timing.DECODE: 1}


def write_candidates(folder, rows):
    import bz2
    import pickle

    folder.mkdir(parents=True, exist_ok=True)
    for name, answers in rows.items():
        payload = [
            {"beam_score": 0.1, "score_aug": [0.5] * 8, "solution": np.array(answer)}
            for answer in answers
        ]
        with bz2.BZ2File(folder / name, "w") as handle:
            pickle.dump(payload, handle)


def test_read_puzzle_groups_one_puzzles_files_by_test_input(tmp_path):
    big = [[(r + c) % 10 for c in range(30)] for r in range(30)]
    write_candidates(tmp_path, {
        "0a0b0c0d_1.transpose.permute0123456789": [big],
        "0a0b0c0d_0.permute0123456789": [[[1]], [[2]]],
        "0a0b0c0e_0.permute0123456789": [[[3]]],
        "0a0b0c0d_7.permute0123456789": [[[4]]],
    })  # fmt: skip
    first, second = read_puzzle(tmp_path, "0a0b0c0d", 2)
    assert [c.grid for c in first] == [grid([1]), grid([2])]
    assert [c.grid for c in second] == [grid(*big)]
    assert read_puzzle(tmp_path / "missing", "0a0b0c0d", 1) == ((),)


def test_installing_replaces_only_the_given_inputs_files_and_keeps_a_backup(tmp_path):
    outputs, source, backup = tmp_path / "outputs", tmp_path / "round1", tmp_path / "round0"
    write_candidates(outputs, {
        "0a0b0c0d_0.permute0123456789": [[[1]]],
        "0a0b0c0d_1.permute0123456789": [[[2]]],
        "0a0b0c0d_1.transpose.permute0123456789": [[[2]]],
        "0a0b0c0e_1.permute0123456789": [[[9]]],
    })  # fmt: skip
    write_candidates(source, {
        "0a0b0c0d_0.permute0123456789": [[[5]]],
        "0a0b0c0d_1.rot90.permute0123456789": [[[6]]],
    })  # fmt: skip
    old = {p.name: p.read_bytes() for p in outputs.iterdir()}
    replaced = install_candidates(source, outputs, "0a0b0c0d", [1, 2], backup=backup)
    assert replaced == (1,)  # input 2 has no new candidate, input 0 was not asked for
    names = sorted(p.name for p in outputs.iterdir())
    assert names == [
        "0a0b0c0d_0.permute0123456789", "0a0b0c0d_1.rot90.permute0123456789",
        "0a0b0c0e_1.permute0123456789",
    ]  # fmt: skip
    assert (outputs / names[0]).read_bytes() == old[names[0]]
    assert (outputs / names[1]).read_bytes() == (source / names[1]).read_bytes()
    assert {p.name: p.read_bytes() for p in backup.iterdir()} == {
        k: v for k, v in old.items() if k.startswith("0a0b0c0d_")
    }
    # A second install keeps the first backup: round 0's candidates.
    install_candidates(source, outputs, "0a0b0c0d", [0], backup=backup)
    assert (backup / "0a0b0c0d_0.permute0123456789").read_bytes() == old[names[0]]
