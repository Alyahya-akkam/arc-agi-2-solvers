"""Acceptance tests of pseudo-pair promotion (task P2.2; BRIEF B5 P2.2, docs/tasks/P2.2.md).

B5's acceptance: "unit tests on synthetic vote tallies for each of the three conditions
independently". ``arc26.loop.promote`` runs on CPU with mocks for the candidate fit, the
re-decode, and the gate's decode: one test per condition that fails only that condition
(agreement below tau, the gate failing with the candidate adapter, the re-decode disagreeing),
all three passing, and the disjointness check of the re-decode's views; then the edge cases the
brief lists (k = 2 with the hold-out flag on, several test inputs, no candidate, a tie in the
vote) and those of the test-engineer role (30 by 30 grids, rollback, raising checks), the
exclusion of the held-out pair from every fit and prompt, the records, and the timing spans.
The view checks run the moved ``ArcDataset`` (numpy, and transformers for an annotation, stubbed
here as in ``tests/test_replay.py``). The last part runs ``arc26.decode.decode_puzzle`` against
the GPU stand-ins of ``tests/test_solver_timing.py``: with ``view_seed`` at its default it decodes
and rescores under the reference's views, with a seed under the views ``view_keys`` predicts, and
the re-decode's records go under their own op names. The synthetic tasks are never competition
data.
"""

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from arc26 import loop, timing
from arc26.loop import (
    ALREADY_PROMOTED,
    BELOW_TAU,
    CANDIDATE_OPS,
    CANDIDATE_TRAIN,
    DECODE_VIEWS,
    FIT_RAISED,
    FRESH_SEED,
    GATE,
    GATE_FAILED,
    GATE_RAISED,
    NO_CANDIDATES,
    NO_HELD_OUT,
    PROMOTED,
    PROMOTION_PREFIX,
    REASONS,
    REDECODE,
    REDECODE_DECODE,
    REDECODE_DISAGREES,
    REDECODE_OPS,
    REDECODE_RAISED,
    REDECODE_SELECTION,
    TAU,
    TIED,
    VIEWS_NOT_DISJOINT,
    Candidate,
    DemoSet,
    HeldOutGate,
    Pair,
    Promotion,
    PromotionRecord,
    fresh_views,
    promote,
    tally,
    view_keys,
    view_signatures,
    vote,
)

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
MOVED = ("arc26.nvarc.arc_loader",)


@pytest.fixture(autouse=True, scope="module")
def moved_loader():
    """The moved arc_loader imported with a stub transformers (it imports AutoTokenizer for an
    annotation), removed again afterwards so no other test sees modules bound to the stub."""
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


# A three-pair puzzle with one test input, T0, or two, T0 and T1; the last pair, C, is the held-out
# one by the default rule. Many colors, so fresh views are disjoint from the first decode's at once.
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


def votes(*counts, nll=None):
    """Synthetic view candidates: ``count`` candidates of each grid, in order; every candidate of
    a grid has the rescoring NLL ``nll[grid]`` (0.5 when not given), so ``score_kgmon`` scores a
    grid ``count - nll``."""
    nll = nll or {}
    found = []
    for answer, count in counts:
        for _ in range(count):
            index = len(found)
            found.append(Candidate(answer, 0.01 * index, (nll.get(answer, 0.5),) * 8, f"v{index}"))
    return found


class MockFit:
    """P2.1's from-scratch fit; promote never calls it."""

    def __init__(self):
        self.calls = []

    def __call__(self, state):
        self.calls.append(state)
        return "refit"


class MockAdapt:
    """The candidate adapter's fit (sgd_continue): records the state and the init it got."""

    def __init__(self, raises=None):
        self.calls = []
        self.raises = raises

    def __call__(self, state, init):
        self.calls.append((state, init))
        if self.raises is not None:
            raise self.raises
        return f"candidate-{len(self.calls)}"


class MockRedecode:
    """The re-decode: fixed view candidates per query; records what it was handed."""

    def __init__(self, answers, raises=None):
        self.answers = dict(answers)
        self.raises = raises
        self.calls = []

    def __call__(self, adapter, state, query, seed):
        self.calls.append((adapter, state, query, seed))
        if self.raises is not None:
            raise self.raises
        return list(self.answers[query])


class GateDecode:
    """The gate's decode of the held-out input: the held-out output ``right`` for a candidate
    adapter when ``passing``, a wrong grid otherwise (and always for the current adapter, so a
    passing gate shows that the candidate adapter was checked)."""

    def __init__(self, passing=True, raises=None, right=C.output):
        self.passing = passing
        self.raises = raises
        self.right = right
        self.calls = []

    def __call__(self, adapter, state, query, views):
        self.calls.append((adapter, state, query, views))
        if self.raises is not None:
            raise self.raises
        right = self.passing and str(adapter).startswith("candidate-")
        return votes((self.right if right else W, views))


AGREE = {T0: votes((Y0, 14), (W, 2)), T1: votes((Y1, 13), (W, 3))}


def run(candidates, state=None, *, passing=True, redecode=None, adapt=None, gate_decode=None,
        tau=TAU, **kwargs):  # fmt: skip
    state = state if state is not None else demos()
    adapt = adapt or MockAdapt()
    redecode = redecode or MockRedecode(AGREE)
    gate_decode = gate_decode or GateDecode(passing)
    gate = HeldOutGate(MockFit(), gate_decode)
    result = promote(
        candidates, gate, "adapter-0", state, tau, adapt=adapt, redecode=redecode, **kwargs
    )
    return result, adapt, redecode, gate_decode


# B5 acceptance: all three conditions passing promotes the voted answer.


def test_all_three_conditions_pass_and_the_voted_answer_is_promoted():
    state = demos()
    result, adapt, redecode, gate_decode = run([votes((Y0, 12), (W, 4))], state)
    (record,) = result.records
    assert (record.promoted, record.reason) == (True, PROMOTED)
    assert record.agreement == 12 / 16 >= TAU and record.votes == (12, 4) and not record.tied
    assert record.views.disjoint and record.redecode.agreed and record.gate.passed
    (pair,) = result.pairs
    assert (pair.input, pair.output, pair.kind) == (T0, Y0, "pseudo")
    assert result.adapter == "candidate-1" and result.gate is record.gate
    # One re-decode with the current adapter, one fit from it, one gate check of the candidate.
    ((adapter, _, query, seed),) = redecode.calls
    assert (adapter, query, seed) == ("adapter-0", T0, record.views.seed)
    ((fitted, init),) = adapt.calls
    draft = Pair(T0, Y0, kind="pseudo", round_added=1, agreement=0.75, test_index=0)
    assert init == "adapter-0" and fitted.pairs == (A, B, draft) and fitted.held_out is None
    ((checked, _, held_input, views),) = gate_decode.calls
    assert (checked, held_input, views) == ("candidate-1", C.input, DECODE_VIEWS)
    assert result.demos.version == state.version + 1 and result.demos.pairs == (A, B, pair)


def test_agreement_below_tau_fails_only_that_condition():
    """Agreement 11/16 is below tau = 0.75; the re-decode and the gate would pass, but neither
    runs (cheapest check first)."""
    result, adapt, redecode, gate_decode = run([votes((Y0, 11), (W, 5))], tau=0.75)
    (record,) = result.records
    assert (record.promoted, record.reason) == (False, BELOW_TAU)
    assert record.agreement == 11 / 16 < record.tau == 0.75 and not record.tied
    assert record.views is record.redecode is record.gate is None
    assert redecode.calls == adapt.calls == gate_decode.calls == []
    assert result.pairs == () and result.adapter is None and result.demos == demos()
    # The same tally at the default tau, 9/16, passes all three.
    assert run([votes((Y0, 11), (W, 5))])[0].records[0].reason == PROMOTED


def test_gate_failing_with_the_candidate_adapter_fails_only_that_condition():
    result, adapt, redecode, gate_decode = run([votes((Y0, 12), (W, 4))], passing=False)
    (record,) = result.records
    assert (record.promoted, record.reason) == (False, GATE_FAILED)
    assert record.agreement >= TAU and record.redecode.agreed and record.views.disjoint
    assert record.gate.applied and record.gate.passed is False and record.gate.expected_rank is None
    assert record.fitted == (0,)
    # The gate checked the candidate adapter, fit on the state with the pseudo-pair.
    ((checked, context, held_input, _),) = gate_decode.calls
    assert checked == "candidate-1" and held_input == C.input
    assert [(p.input, p.output) for p in context.pairs] == [
        (A.input, A.output), (B.input, B.output), (T0, Y0)
    ]  # fmt: skip
    # Nothing is promoted: the state and the adapter roll back to the previous ones.
    assert result.pairs == () and result.adapter is None and result.demos == demos()
    assert result.gate is record.gate and len(adapt.calls) == 1 and len(redecode.calls) == 1


def test_redecode_disagreeing_fails_only_that_condition():
    """Agreement 12/16 passes and the gate would pass, but the fresh views vote another grid."""
    redecode = MockRedecode({T0: votes((W, 9), (Y0, 7))})
    result, adapt, _, gate_decode = run([votes((Y0, 12), (W, 4))], redecode=redecode)
    (record,) = result.records
    assert (record.promoted, record.reason) == (False, REDECODE_DISAGREES)
    assert record.agreement >= TAU and record.views.disjoint
    assert record.redecode.agreed is False and record.redecode.expected_rank == 2
    assert record.redecode.votes == (9, 7) and record.gate is None
    assert adapt.calls == gate_decode.calls == [] and result.pairs == ()


def test_redecode_without_the_answer_or_without_candidates_disagrees():
    for answers in (votes((W, 16)), []):
        result, adapt, _, _ = run([votes((Y0, 12), (W, 4))], redecode=MockRedecode({T0: answers}))
        (record,) = result.records
        assert record.reason == REDECODE_DISAGREES and record.redecode.expected_rank is None
        assert record.redecode.candidates == len(answers) and adapt.calls == []


# B5 acceptance: the disjointness check of the re-decode's views.


def literal_views(keys):
    """(transforms, permutation) of each key: the brief's definition of a view."""
    found = set()
    for name in keys:
        parts = name.split(".")[1:]
        transforms = tuple(p for p in parts if p in ("transpose", "rot90"))
        (permutation,) = [p for p in parts if p.startswith("permute")]
        found.add((transforms, permutation))
    return found


def test_the_disjointness_check_compares_every_view_of_both_decodes():
    """The first decode (two test inputs, the reference's views) decodes test input 1 under 16
    views, 2 permutations of each of the 8 transforms, and rescores under 8 more; the re-decode
    (the input alone, a fresh seed) shares none of them, by key and by what they show."""
    state = demos(T0, T1)
    context = state.without_held_out()
    first = view_keys(PUZZLE, context.task(), 1, None)
    check = fresh_views(state, 1)
    fresh = view_keys(PUZZLE, context.task([T1]), 0, check.seed)
    assert len(first) == len(fresh) == 24 and len(set(first)) == 24
    transforms = [name.split(".permute")[0].split(".", 1)[-1] for name in first]
    assert len(set(transforms[:16])) == 8 and len(set(transforms[16:])) == 8
    assert all(name.startswith(f"{PUZZLE}_1.") for name in first)
    assert all(name.startswith(f"{PUZZLE}_0.") for name in fresh)
    assert not literal_views(first) & literal_views(fresh)
    grids = [g for pair in context.pairs for g in (pair.input, pair.output)] + [T1]
    shown_first, shown_fresh = view_signatures(first, grids), view_signatures(fresh, grids)
    assert not shown_first & shown_fresh and check.disjoint and check.common == 0
    assert (check.first, check.fresh) == (len(shown_first), len(shown_fresh))
    assert check.seed >= FRESH_SEED and check.tries == check.seed - FRESH_SEED + 1


def test_views_that_coincide_are_found_and_the_next_seed_is_tried():
    """A re-decode under the first decode's own seed shares every view (one test input: the same
    key, the same draws); the search moves on, and with one try the promotion stops there."""
    state = demos()
    same = fresh_views(state, 0, first_seed=7, fresh_seed=7, tries=1)
    assert not same.disjoint and same.common == same.first == same.fresh and same.seed == 7
    moved = fresh_views(state, 0, first_seed=7, fresh_seed=7)
    assert moved.disjoint and moved.seed > 7 and moved.tries == moved.seed - 6
    result, adapt, redecode, _ = run([votes((Y0, 12), (W, 4))], first_seed=7, fresh_seed=7,
                                     tries=1)  # fmt: skip
    (record,) = result.records
    assert record.reason == VIEWS_NOT_DISJOINT and record.views == same
    assert redecode.calls == adapt.calls == [] and result.pairs == ()
    # With more tries, the re-decode runs under the seed the search found, not the first tried.
    result, _, redecode, _ = run([votes((Y0, 12), (W, 4))], first_seed=7, fresh_seed=7)
    (record,) = result.records
    assert record.reason == PROMOTED and record.views == moved
    assert [call[3] for call in redecode.calls] == [moved.seed]
    # With the reference's views as the first decode, a fresh seed of 2 repeats its 16 decoding
    # views (one test input: the same draws) but not its 8 rescoring views, whose seeds differ.
    reference = fresh_views(state, 0, fresh_seed=2, tries=1)
    assert reference.common == 16 and reference.first == 24


def test_permutations_that_differ_but_show_the_same_prompt_are_one_view():
    """Every grid of this puzzle is one cell of color 0, so a view shows only where its
    permutation sends 0: permutations that differ (the brief's views are disjoint) still show
    the model the same prompt, and no seed avoids them."""
    zero = Pair(grid([0]), grid([0]))
    state = DemoSet.from_task(PUZZLE, task(grid([0]), pairs=(zero, zero, zero)))
    check = fresh_views(state, 0, tries=4)
    context = state.without_held_out()
    first = view_keys(PUZZLE, context.task(), 0, None)
    fresh = view_keys(PUZZLE, context.task([grid([0])]), 0, check.seed)
    assert not literal_views(first) & literal_views(fresh)
    assert not check.disjoint and check.tries == 4 and check.first <= 10
    result, *_ = run(
        [votes((grid([5]), 16))],
        state,
        redecode=MockRedecode({grid([0]): votes((grid([5]), 16))}),
        tries=4,
    )
    assert result.records[0].reason == VIEWS_NOT_DISJOINT and result.pairs == ()


def test_the_first_decodes_views_follow_its_test_inputs():
    """decode_puzzle draws the permutations for every test input of the call in turn, so test
    input 0's views differ between a decode of the puzzle and a decode of the input alone."""
    context = demos(T0, T1).without_held_out()
    together = view_keys(PUZZLE, context.task(), 0, None)
    alone = view_keys(PUZZLE, context.task([T0]), 0, None)
    assert literal_views(together[:16]) != literal_views(alone[:16])
    # The rescoring views depend on the test output's key and the seed only.
    assert literal_views(together[16:]) == literal_views(alone[16:])


def test_the_check_uses_the_first_decodes_layout_of_test_inputs():
    """augment draws one permutation per key, in key order, transforms t = 0..7 outer and test
    inputs j inner, twice: in a decode of m inputs, (j, t) takes draw 2t + j in the first round
    and 16 + 2t + j in the second; decoded alone, t takes draw t, then 8 + t. At the reference's
    seed 2, input 0 alone meets the puzzle's decode at t = 0 (draw 0, first rounds) and input 1
    at t = 7 (draw 15: its second round, the puzzle's first): one view each. Computing the first
    decode's views as if the input were alone would find all 16 in common."""
    state = demos(T0, T1)
    for index in (0, 1):
        check = fresh_views(state, index, fresh_seed=2, tries=1)
        assert check.common == 1 and check.first == check.fresh == 24
    assert fresh_views(state, 1, fresh_seed=2).seed > 2


def test_view_checks_leave_numpys_random_state_alone():
    np.random.seed(123)
    before = np.random.get_state()
    view_keys(PUZZLE, task(T0), 0, 9)
    fresh_views(demos(), 0)
    run([votes((Y0, 12), (W, 4))])
    after = np.random.get_state()
    assert before[0] == after[0] and (before[1] == after[1]).all() and before[2:] == after[2:]


# The agreement: the count term of the reference's score_kgmon over the n views.


def test_agreement_is_the_reference_votes_count_over_the_views():
    from arc26.nvarc.arc_decoder import getter_kgmon, score_kgmon

    found = votes((W, 5), (Y0, 6), (Y1, 5), nll={W: 0.1, Y0: 2.5, Y1: 0.2})
    counted = tally(found)
    # score_kgmon counts each view 1 and subtracts the mean NLL: W 4.9, Y1 4.8, then Y0 3.5,
    # which more views produced.
    guesses = {
        str(i): {"solution": c.grid, "beam_score": c.beam_score, "score_aug": list(c.score_aug)}
        for i, c in enumerate(found)
    }
    assert list(counted.grids) == [loop.as_grid(g) for g in score_kgmon(guesses)] == vote(found)
    assert counted.grids == (W, Y1, Y0) and counted.counts == (5, 5, 6)
    groups = {g: [x for x in guesses.values() if x["solution"] == g] for g in counted.grids}
    assert counted.scores == tuple(float(getter_kgmon(groups[g])) for g in counted.grids)
    assert counted.voted == W and counted.agreement == 5 / 16 and not counted.tied
    assert tally([]).voted is None and tally([]).agreement is None and not tally([]).tied


def test_tau_starts_at_a_strict_majority_of_the_views_and_is_inclusive():
    assert TAU == 9 / 16 == 0.5625
    assert run([votes((Y0, 9), (W, 7))])[0].records[0].reason == PROMOTED
    assert run([votes((Y0, 8), (W, 8), nll={W: 0.9})])[0].records[0].reason == BELOW_TAU
    # tau is a parameter: 0 admits any agreement, 1 only all 16 views.
    assert run([votes((Y0, 3), (W, 2), (Y1, 1))], tau=0.0)[0].records[0].reason == PROMOTED
    assert run([votes((Y0, 15), (W, 1))], tau=1.0)[0].records[0].reason == BELOW_TAU
    assert run([votes((Y0, 16))], tau=1.0)[0].records[0].reason == PROMOTED


# Edge cases: k = 2 with the hold-out flag on, several test inputs, no candidate, a tie.


def test_k2_with_the_hold_out_flag_on_promotes_with_one_real_pair_left():
    state = demos(pairs=(A, B))
    assert state.held_out.input == B.input and state.pairs == (A,) and state.real_pairs == 2
    gate_decode = GateDecode(right=B.output)
    result, adapt, *_ = run([votes((Y0, 12), (W, 4))], state, gate_decode=gate_decode)
    assert result.records[0].reason == PROMOTED
    ((fitted, _),) = adapt.calls
    # The candidate adapter trains on the one real pair and the pseudo-pair, never on B.
    assert [p.kind for p in fitted.pairs] == ["real", "pseudo"] and fitted.held_out is None
    assert B.input not in {p.input for p in fitted.pairs}
    assert gate_decode.calls[0][2] == B.input and result.demos.real_pairs == 2


def test_k2_with_the_hold_out_flag_off_promotes_nothing():
    """No held-out pair, no gate: three checks or none (ADR-014). Nothing GPU-side runs."""
    state = demos(pairs=(A, B), k2=False)
    result, adapt, redecode, gate_decode = run([votes((Y0, 16))], state)
    assert result.records[0].reason == NO_HELD_OUT and result.records[0].agreement == 1.0
    assert redecode.calls == adapt.calls == gate_decode.calls == [] and result.pairs == ()


def test_several_test_inputs_are_checked_one_by_one():
    """Input 0 passes; input 1 fails agreement: one fit with input 0's pseudo-pair alone."""
    state = demos(T0, T1)
    result, adapt, redecode, _ = run([votes((Y0, 12), (W, 4)), votes((Y1, 6), (W, 5))], state)
    first, second = result.records
    assert (first.reason, second.reason) == (PROMOTED, BELOW_TAU)
    assert (first.test_index, second.test_index) == (0, 1) and second.fitted == ()
    assert [call[2] for call in redecode.calls] == [T0]
    ((fitted, _),) = adapt.calls
    assert [(p.input, p.test_index) for p in fitted.pairs if p.kind == "pseudo"] == [(T0, 0)]
    assert [p.test_index for p in result.pairs] == [0]


def test_several_test_inputs_share_one_candidate_adapter_and_one_gate():
    state = demos(T0, T1)
    candidates = [votes((Y0, 12), (W, 4)), votes((Y1, 10), (W, 6))]
    result, adapt, redecode, gate_decode = run(candidates, state)
    assert [r.reason for r in result.records] == [PROMOTED, PROMOTED]
    assert [r.fitted for r in result.records] == [(0, 1), (0, 1)]
    assert len(adapt.calls) == len(gate_decode.calls) == 1 and len(redecode.calls) == 2
    assert result.records[0].gate is result.records[1].gate is result.gate
    assert [(p.input, p.output, p.test_index) for p in result.pairs] == [(T0, Y0, 0), (T1, Y1, 1)]
    assert result.demos.version == 1 and len(result.demos.pairs) == 4
    # Each input's re-decode has its own view check and seed.
    assert all(r.views.disjoint for r in result.records)
    # A failing gate promotes neither: one gate cannot tell a right pseudo-pair from a wrong one.
    failed, *_ = run(candidates, state, passing=False)
    assert [r.reason for r in failed.records] == [GATE_FAILED, GATE_FAILED]
    assert failed.pairs == () and failed.demos == state


def test_an_input_without_candidates_is_not_promoted():
    result, adapt, redecode, _ = run([[]])
    (record,) = result.records
    assert (record.reason, record.candidates, record.votes, record.agreement) == (
        NO_CANDIDATES, 0, (), None
    )  # fmt: skip
    assert redecode.calls == adapt.calls == [] and result.pairs == ()
    # Several inputs, none decoded (a puzzle cut at the deadline).
    none, *_ = run([[], []], demos(T0, T1))
    assert [r.reason for r in none.records] == [NO_CANDIDATES, NO_CANDIDATES]


def test_a_tie_in_the_vote_is_not_promoted():
    """Two grids with equal score_kgmon values: the candidates' order picked the voted answer, so
    the views did not; agreement 8/16 would pass at tau = 0.5."""
    tie = votes((Y0, 8), (W, 8))
    result, adapt, redecode, _ = run([tie], tau=0.5)
    (record,) = result.records
    assert record.tied and record.reason == TIED and record.agreement == 0.5
    assert tally(tie).scores[0] == tally(tie).scores[1] and redecode.calls == adapt.calls == []
    # Equal counts are no tie when the rescoring NLL separates them.
    apart = run([votes((Y0, 8), (W, 8), nll={W: 0.9})], tau=0.5)[0].records[0]
    assert not apart.tied and apart.reason == PROMOTED
    # A tie in the re-decode's vote fails the re-decode.
    tied_again = MockRedecode({T0: votes((Y0, 8), (W, 8))})
    record = run([votes((Y0, 12), (W, 4))], redecode=tied_again)[0].records[0]
    assert record.reason == REDECODE_DISAGREES and record.redecode.tied
    assert record.redecode.expected_rank == 1 and not record.redecode.agreed


def test_30x30_grids_are_promoted():
    big_in = grid(*[[(r + c) % 10 for c in range(30)] for r in range(30)])
    big_out = grid(*[[(r * c) % 10 for c in range(30)] for r in range(30)])
    state = demos(big_in)
    redecode = MockRedecode({big_in: votes((big_out, 16))})
    result, *_ = run([votes((big_out, 15), (W, 1))], state, redecode=redecode)
    assert result.records[0].reason == PROMOTED and result.pairs[0].output == big_out
    assert len(result.pairs[0].input) == 30 and result.records[0].views.disjoint


def test_an_input_promoted_in_an_earlier_round_is_not_promoted_again():
    first = run([votes((Y0, 12), (W, 4))])[0]
    second, adapt, redecode, _ = run([votes((Y0, 16))], first.demos, round_added=2)
    assert second.records[0].reason == ALREADY_PROMOTED and second.records[0].round_added == 2
    assert redecode.calls == adapt.calls == [] and second.demos is first.demos


# A check that raises fails, with its error logged; the worker goes on.


@pytest.mark.parametrize(
    ("where", "reason"),
    [("redecode", REDECODE_RAISED), ("fit", FIT_RAISED), ("gate", GATE_RAISED)],
)
def test_a_check_that_raises_fails_and_logs_its_error(where, reason):
    error = MemoryError("CUDA out of memory")
    result, adapt, _, gate_decode = run(
        [votes((Y0, 12), (W, 4))],
        redecode=MockRedecode(AGREE, raises=error if where == "redecode" else None),
        adapt=MockAdapt(raises=error if where == "fit" else None),
        gate_decode=GateDecode(raises=error if where == "gate" else None),
    )
    (record,) = result.records
    assert record.reason == reason and not record.promoted and result.pairs == ()
    message = "MemoryError: CUDA out of memory"
    if where == "redecode":
        assert record.redecode.error == message and adapt.calls == []
    elif where == "fit":
        assert record.error == message and record.gate is None and gate_decode.calls == []
    else:
        assert record.gate.error == message and record.gate.passed is None


# The held-out pair reaches no fit and no prompt.


def held_out_free(state):
    return state.held_out is None and all(
        pair.input != C.input and pair.output != C.output for pair in state.pairs
    )


def test_the_held_out_pair_reaches_no_fit_and_no_prompt():
    result, adapt, redecode, gate_decode = run([votes((Y0, 12), (W, 4))])
    ((_, context, query, _),) = redecode.calls
    assert held_out_free(context) and context.pairs == (A, B) and query == T0
    ((fitted, _),) = adapt.calls
    pseudo = {"input": [list(r) for r in T0], "output": [list(r) for r in Y0]}
    assert held_out_free(fitted)
    assert fitted.task()["train"] == [A.task_pair(), B.task_pair(), pseudo]
    ((_, gate_context, held_input, _),) = gate_decode.calls
    assert held_out_free(gate_context) and held_input == C.input
    # The promoted state still holds C apart, never among its pairs.
    assert result.demos.held_out == demos().held_out and held_out_free(
        result.demos.without_held_out()
    )


# Provenance, versions, and rollback.


def test_a_promoted_pair_carries_its_provenance():
    result, *_ = run([votes((Y0, 12), (W, 4))], round_added=2)
    (pair,) = result.pairs
    (record,) = result.records
    assert (pair.kind, pair.round_added, pair.agreement, pair.test_index) == ("pseudo", 2, 0.75, 0)
    assert pair.verdicts.views == record.views and pair.verdicts.redecode == record.redecode
    assert pair.verdicts.gate == record.gate and pair.verdicts.gate.passed
    assert result.demos.pairs[-1] is pair and hash(pair) == hash(pair)


def test_rollback_keeps_the_previous_version_untouched():
    state = demos()
    kept = run([votes((Y0, 12), (W, 4))], state)[0]
    rolled_back = run([votes((Y0, 12), (W, 4))], state, passing=False)[0]
    assert kept.demos.version == 1 and kept.demos is not state
    assert state.version == 0 and state.pairs == (A, B)  # immutable: the old version is intact
    assert rolled_back.demos is state and rolled_back.adapter is None


def test_enlarging_a_demo_set_takes_pseudo_pairs_only():
    state = demos()
    pseudo = Pair(T0, Y0, kind="pseudo", round_added=1, agreement=0.75, test_index=0)
    assert state.enlarged([pseudo]).version == 1
    with pytest.raises(ValueError, match="at least one"):
        state.enlarged([])
    with pytest.raises(ValueError, match="only pseudo-pairs"):
        state.enlarged([A])


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"kind": "real", "test_index": 0}, ValueError),
        ({"kind": "held_out", "test_index": 1}, ValueError),
        ({"kind": "pseudo", "round_added": 1, "test_index": -1}, ValueError),
        ({"kind": "pseudo", "round_added": 1, "test_index": 1.0}, TypeError),
        ({"kind": "pseudo", "round_added": 1, "verdicts": "passed"}, TypeError),
    ],
)
def test_pair_provenance_fields_are_checked(fields, error):
    with pytest.raises(error):
        Pair(A.input, A.output, **fields)


# Records, logs, and timing.


def test_every_test_input_gets_one_json_record(tmp_path):
    result, *_ = run([votes((Y0, 12), (W, 4)), votes((Y1, 4), (W, 2))], demos(T0, T1))
    log = loop.VerdictLog(tmp_path, "rank1", prefix=PROMOTION_PREFIX)
    written = [log.write(record) for record in result.records]
    assert log.path == tmp_path / "promote-rank1.jsonl"
    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert lines == written and [line["test_index"] for line in lines] == [0, 1]
    first, second = lines
    assert set(first) == {
        "puzzle_id", "round", "test_index", "candidates", "votes", "agreement", "tau", "tied",
        "views", "redecode", "gate", "fitted", "promoted", "reason", "error",
    }  # fmt: skip
    assert (first["promoted"], first["reason"], first["fitted"]) == (True, PROMOTED, [0])
    assert set(first["views"]) == {"seed", "tries", "first", "fresh", "common"}
    assert first["redecode"]["agreed"] and first["gate"]["passed"] and first["votes"] == [12, 4]
    assert (second["reason"], second["views"], second["redecode"], second["gate"]) == (
        BELOW_TAU, None, None, None
    )  # fmt: skip
    assert second["agreement"] == 0.25 and second["tau"] == TAU
    # Twelve reasons of P2.2, and P2.3's redecode_cut (a re-decode stopped by the time cap) and
    # declined (the governor declined the round after the free checks).
    assert set(REASONS) >= {line["reason"] for line in lines} and len(set(REASONS)) == 15


def test_the_promotions_gpu_spans_are_timed(tmp_path):
    logger = timing.TimingLogger(tmp_path, "rank0")
    ticks = iter(range(100))
    run([votes((Y0, 12), (W, 4))], timing_logger=logger, clock=lambda: float(next(ticks)))
    records, skipped = timing.read_records([logger.path])
    assert skipped == 0 and [(r.puzzle_id, r.op, r.tokens) for r in records] == [
        (PUZZLE, REDECODE, None), (PUZZLE, GATE, None)
    ]  # fmt: skip
    assert all(r.seconds > 0 for r in records)
    # A raising re-decode writes no record; without a logger nothing is written anywhere.
    quiet = timing.TimingLogger(tmp_path / "raising", "rank0")
    run([votes((Y0, 12), (W, 4))], redecode=MockRedecode(AGREE, raises=MemoryError()),
        timing_logger=quiet)  # fmt: skip
    assert not quiet.path.exists()


def test_the_new_op_names_rename_the_reference_ops(tmp_path):
    logger = timing.TimingLogger(tmp_path, "rank0")
    loop.RenamedOps(logger, REDECODE_OPS).record(PUZZLE, timing.DECODE, 2.0, 40)
    loop.RenamedOps(logger, REDECODE_OPS).record(PUZZLE, timing.SELECTION, 1.0, 80)
    loop.RenamedOps(logger, CANDIDATE_OPS).record(PUZZLE, timing.TRAIN, 30.0, 5000)
    records, _ = timing.read_records([logger.path])
    assert [(r.op, r.tokens) for r in records] == [
        (REDECODE_DECODE, 40), (REDECODE_SELECTION, 80), (CANDIDATE_TRAIN, 5000)
    ]  # fmt: skip
    assert (REDECODE, REDECODE_DECODE, REDECODE_SELECTION, CANDIDATE_TRAIN) == (
        "redecode", "redecode_decode", "redecode_selection", "candidate_train"
    )  # fmt: skip


# Arguments.


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"tau": 1.5}, ValueError),
        ({"tau": -0.1}, ValueError),
        ({"tau": True}, TypeError),
        ({"round_added": 0}, ValueError),
        ({"round_added": 1.0}, TypeError),
        ({"views": 0}, ValueError),
        ({"fresh_seed": -1}, ValueError),
        ({"fresh_seed": 2**32}, ValueError),
        ({"first_seed": "2"}, TypeError),
        ({"tries": 0}, ValueError),
        ({"candidates": [[], []]}, ValueError),
        ({"candidates": [["not a candidate"]]}, TypeError),
        ({"adapt": None}, TypeError),
    ],
)
def test_bad_arguments_are_refused(change, error):
    arguments = {"candidates": [votes((Y0, 12), (W, 4))], "tau": TAU}
    arguments.update(change)
    candidates, tau = arguments.pop("candidates"), arguments.pop("tau")
    adapt = arguments.pop("adapt", MockAdapt())
    with pytest.raises(error):
        promote(
            candidates, HeldOutGate(MockFit(), GateDecode()), "adapter-0", demos(), tau,
            adapt=adapt, redecode=MockRedecode(AGREE), **arguments,
        )  # fmt: skip


def test_a_redecode_must_yield_candidates():
    redecode = MockRedecode({T0: [Y0]})
    with pytest.raises(TypeError, match="Candidate"):
        run([votes((Y0, 12), (W, 4))], redecode=redecode)


def test_promote_returns_b4s_pairs_and_its_records():
    result, *_ = run([votes((Y0, 12), (W, 4))])
    assert isinstance(result, Promotion) and result.promoted
    assert all(isinstance(record, PromotionRecord) for record in result.records)
    nothing = run([[]])[0]
    assert not nothing.promoted and nothing.gate is None


# decode_puzzle's view seed, against the GPU stand-ins of tests/test_solver_timing.py.


def load_timing_tests():
    path = REPO / "tests" / "test_solver_timing.py"
    spec = importlib.util.spec_from_file_location("_promote_stand_ins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAND_INS = load_timing_tests()
STAND_IN_HEAD, _IMPORT, _ = STAND_INS.DRIVER.partition("import numpy as np\n")
assert _IMPORT and "def stand_in(" in STAND_IN_HEAD

DRIVER = STAND_IN_HEAD + r'''
import ast
import io
import numpy as np
from arc26 import decode, loop, timing
from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter

key, tasks = sys.argv[4], json.loads(sys.argv[5])
augmented, decoded = [], []
original_augment = ArcDataset.augment


def recording_augment(self, n=1, shfl_keys=False, seed=42):
    result = original_augment(self, n=n, shfl_keys=shfl_keys, seed=seed)
    augmented.append({"n": n, "keys": list(result.keys)})
    return result


def fake_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    grid = Tokenizer().encode("7\n") + [EOS]
    return [(i, [(0.25, grid)]) for i in range(len(prefix_tokens))]


def fake_scores(queries, answers, tokenizer, model):
    return [1.0 for _ in queries]


decode.inference_turbo_dfs = fake_dfs
decode.calc_scores = fake_scores
formatter = QwenFormatter(Tokenizer())
results = {}
for name, (entry, seed) in tasks.items():
    augmented.clear()
    folder = os.path.join(work, name)
    os.makedirs(folder)
    logger = timing.TimingLogger(os.path.join(work, "timing"), name)
    ArcDataset.augment = recording_augment
    old = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        decode.decode_puzzle(
            puzzle_ds=ArcDataset(queries={key: entry}),
            formatter=formatter,
            max_seq_length=8192,
            max_new_tokens=formatter.max_new_tokens(),
            start_time=time.time(),
            end_time=time.time() + 1e6,
            rank=0,
            key=key,
            tokenizer=Tokenizer(),
            model=None,
            max_score=1.6,
            timing_logger=loop.RenamedOps(logger, loop.REDECODE_OPS),
            dir_outputs=folder,
            **({} if seed is None else {"view_seed": seed}),
        )
    finally:
        sys.stdout = old
        ArcDataset.augment = original_augment
    batches = [
        ast.literal_eval(line.split("decoding ", 1)[1])
        for line in captured.getvalue().splitlines()
        if " decoding " in line
    ]
    rescoring = sorted({k for call in augmented if call["n"] == 1 for k in call["keys"]})
    predicted = {
        index: list(loop.view_keys(key, entry, index, seed))
        for index in range(len(entry["test"]))
    }
    records = []
    if os.path.exists(logger.path):
        records = [json.loads(line) for line in open(logger.path)]
    results[name] = {
        "batches": batches,
        "rescoring": rescoring,
        "predicted": predicted,
        "files": sorted(os.listdir(folder)),
        "ops": sorted({r["op"] for r in records}),
    }
with open(os.path.join(work, "summary.json"), "w") as handle:
    json.dump(results, handle)
'''


@pytest.fixture(scope="module")
def stand_in_decodes(tmp_path_factory):
    """decode_puzzle on a two-input puzzle with the reference's views and with a seed, and on
    one input alone with a seed, as the re-decode calls it."""
    work = tmp_path_factory.mktemp("view-seed")
    context = demos(T0, T1).without_held_out()
    runs = {
        "reference": (context.task(), None),
        "seeded": (context.task(), 11),
        "alone": (context.task([T1]), 11),
    }
    driver = work.parent / f"{work.name}-driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "KAGGLE_IS_COMPETITION_RERUN"}
    # decode_puzzle seeds its rescoring views with hash(), so the driver fixes the hash seed.
    env.update(PYTHONHASHSEED="0", PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(driver), str(work), "", "plain", PUZZLE, json.dumps(runs)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((work / "summary.json").read_text(encoding="utf-8"))


def test_decode_puzzle_decodes_the_views_view_keys_predicts(stand_in_decodes):
    for name, run_ in stand_in_decodes.items():
        decoded = [subkey for batch in run_["batches"] for subkey in batch]
        for index, keys in run_["predicted"].items():
            base = f"{PUZZLE}_{index}"
            assert sorted(k for k in decoded if k.startswith(base + ".")) == sorted(keys[:16])
            mine = sorted(k for k in run_["rescoring"] if k.startswith(base + "."))
            assert mine == sorted(keys[16:]), name
        # Four DFS batches of four views per test input, each batch one input's views.
        assert all(len(batch) == 4 for batch in run_["batches"])
        assert len(run_["batches"]) == 4 * len(run_["predicted"])


def test_view_seed_changes_every_view_and_its_default_is_the_reference(stand_in_decodes):
    reference, seeded = stand_in_decodes["reference"], stand_in_decodes["seeded"]
    for index in ("0", "1"):
        first, other = reference["predicted"][index], seeded["predicted"][index]
        assert not literal_views(first) & literal_views(other)
    # The reference's views: augment(n=2, seed=2) on the puzzle, as the moved worker draws them.
    assert reference["predicted"]["0"][0].startswith(f"{PUZZLE}_0.")
    # The candidate files are named by the views decoded: none in common.
    assert not set(reference["files"]) & set(seeded["files"])


def test_the_redecode_layout_is_one_input_alone(stand_in_decodes):
    alone = stand_in_decodes["alone"]
    assert list(alone["predicted"]) == ["0"] and len(alone["files"]) == 16
    assert all(name.startswith(f"{PUZZLE}_0.") for name in alone["files"])


def test_the_redecodes_records_go_under_their_own_op_names(stand_in_decodes):
    for run_ in stand_in_decodes.values():
        assert run_["ops"] == [REDECODE_DECODE, REDECODE_SELECTION]
