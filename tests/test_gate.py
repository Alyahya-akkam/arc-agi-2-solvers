"""Acceptance tests of the held-out gate (task P2.1; BRIEF B5 P2.1, docs/tasks/P2.1.md).

B5's acceptance: "unit tests with a mocked decoder cover pass, fail, and the exclusion of the
held-out pair from every training path". The first part tests ``arc26.loop`` with mock fit and
decode callables: the gate passes only when the voted answer (the reference's selection,
``score_kgmon``) equals the held-out output exactly; the fit callable never receives the held-out
pair; the decode callable gets it only as the query, never as a demonstration; the hold-out rule,
its ``k = 2`` flag, and the edge cases (30 by 30 grids, two pairs, no candidate). The second part
runs the split solver's worker (``arc26.solver.local_worker``) with a held-out pair against the
stand-ins for the GPU stack of ``tests/test_solver_timing.py``, with an identity DFS that answers
each query with its own input grid: the reference's fit sees every training sample without the
held-out pair, the gate decodes the held-out input into its own folder, the test inputs keep their
candidate files, and the gate's timing records and verdicts are written. The synthetic tasks are
never competition data.
"""

import bz2
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

from arc26 import loop, timing
from arc26.loop import (
    DECODE_VIEWS,
    GATE,
    GATE_DECODE,
    GATE_SELECTION,
    Candidate,
    DemoSet,
    GateVerdict,
    HeldOutGate,
    HoldOut,
    Pair,
    as_grid,
    held_out_index,
    read_candidates,
    vote,
)

REPO = Path(__file__).resolve().parent.parent


def grid(*rows):
    return tuple(tuple(row) for row in rows)


# A three-pair puzzle: the last pair, the held-out one by the default rule, has 12 cells, a count
# no other grid of the puzzle has.
A = Pair(grid([1, 2], [3, 4]), grid([4, 3], [2, 1]))
B = Pair(grid([5, 6, 7]), grid([7, 6, 5]))
C = Pair(
    grid([1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]), grid([2, 2, 1, 1], [4, 4, 3, 3], [6, 6, 5, 5])
)
TEST = grid([8, 9], [9, 8])
TASK = {
    "train": [pair.task_pair() for pair in (A, B, C)],
    "test": [{"input": [list(row) for row in TEST]}],
}
WRONG = grid([0])
HELD_B = Pair(B.input, B.output, kind="held_out")


def demos(**changes):
    return DemoSet.from_task("0a0b0c0d", TASK, **changes)


class MockFit:
    """A fit callable that records the state it was handed and returns a fresh adapter."""

    def __init__(self):
        self.calls = []

    def __call__(self, state):
        self.calls.append(state)
        return f"adapter-{len(self.calls)}"


class MockDecode:
    """A decode callable that returns fixed view candidates and records what it was handed.

    ``answers`` lists one grid per candidate; candidate ``i`` gets beam score ``0.1 * i`` and a
    flat rescoring NLL of ``nll[i]`` (1.0 when not given).
    """

    def __init__(self, answers, nll=None):
        self.answers = list(answers)
        self.nll = list(nll or [1.0] * len(self.answers))
        self.calls = []

    def __call__(self, adapter, state, query, views):
        self.calls.append((adapter, state, query, views))
        return [
            Candidate(answer, 0.1 * index, (self.nll[index],) * 8, f"view{index:02d}")
            for index, answer in enumerate(self.answers)
        ]


def gate_with(answers, nll=None, **kwargs):
    fit, decode = MockFit(), MockDecode(answers, nll)
    return HeldOutGate(fit, decode, **kwargs), fit, decode


# B5 acceptance: pass and fail, with a mocked decoder.


def test_pass_when_the_voted_answer_is_the_held_out_output():
    state = demos()
    gate, _, decode = gate_with([C.output] * 10 + [WRONG] * 6)
    adapter = gate.fit(state)
    verdict = gate.verdict(adapter, state)
    assert verdict.passed is True and gate.passes(adapter, state) is True
    assert verdict.expected_rank == 1 and verdict.votes == (10, 6)
    assert (verdict.candidates, verdict.views, verdict.real_pairs) == (16, DECODE_VIEWS, 3)
    assert verdict.applied and verdict.held_out_index == 2
    assert decode.calls[0][0] == adapter


def test_fail_when_the_voted_answer_is_another_grid():
    state = demos()
    gate, _, _ = gate_with([WRONG] * 9 + [C.output] * 7)
    verdict = gate.verdict("adapter", state)
    # The held-out output is among the candidates, but the vote ranks it second: the gate needs
    # the voted answer, attempt 1, not any candidate.
    assert verdict.passed is False and gate.passes("adapter", state) is False
    assert verdict.expected_rank == 2 and verdict.votes == (9, 7)


def test_fail_when_no_candidate_is_decoded():
    # A decode cut at the deadline, or no parsable beam: nothing to vote for.
    gate, _, _ = gate_with([])
    verdict = gate.verdict("adapter", demos())
    assert verdict.passed is False and verdict.candidates == 0
    assert verdict.votes == () and verdict.expected_rank is None


@pytest.mark.parametrize(
    "answer",
    [
        grid([2, 2, 1, 1], [4, 4, 3, 3], [6, 6, 5, 4]),  # one cell differs
        grid([2, 2, 1, 1], [4, 4, 3, 3]),  # one row short
        grid([2, 2, 1], [4, 4, 3], [6, 6, 5]),  # one column short
        grid([2, 4, 6], [2, 4, 6], [1, 3, 5], [1, 3, 5]),  # the transpose
    ],
    ids=["cell", "row", "column", "transpose"],
)
def test_fail_unless_the_match_is_exact(answer):
    gate, _, _ = gate_with([answer] * 16)
    verdict = gate.verdict("adapter", demos())
    assert verdict.passed is False and verdict.expected_rank is None


def test_the_vote_is_the_reference_selection():
    """score_kgmon ranks by view count minus the mean rescoring NLL: with equal counts, the grid
    the adapter finds likelier under the rescoring views wins."""
    from arc26.nvarc.arc_decoder import score_kgmon

    answers = [WRONG] * 8 + [C.output] * 8
    nll = [0.9] * 8 + [0.4] * 8
    gate, _, _ = gate_with(answers, nll)
    assert gate.passes("adapter", demos())
    gate, _, _ = gate_with(answers, [0.4] * 8 + [0.9] * 8)
    assert not gate.passes("adapter", demos())
    candidates = MockDecode(answers, nll)("adapter", demos(), C.input, 16)
    guesses = {
        str(i): {"solution": c.grid, "beam_score": c.beam_score, "score_aug": list(c.score_aug)}
        for i, c in enumerate(candidates)
    }
    assert vote(candidates) == [as_grid(g) for g in score_kgmon(guesses)] == [C.output, WRONG]


def test_a_custom_selection_replaces_the_vote():
    # The selection is a parameter: the reference's other algorithm, or any callable.
    gate, _, _ = gate_with([WRONG] * 12 + [C.output] * 4, select=lambda guesses: [C.output, WRONG])
    assert gate.passes("adapter", demos())


def test_the_held_out_input_is_decoded_under_n_views():
    gate, _, decode = gate_with([C.output], views=8)
    verdict = gate.verdict("adapter", demos())
    ((_, _, query, views),) = decode.calls
    assert query == C.input and views == 8 == verdict.views
    assert HeldOutGate(MockFit(), MockDecode([])).views == DECODE_VIEWS == 16


# B5 acceptance: the held-out pair is excluded from every training path.


def held_out_free(state):
    """Whether a state handed to a callable carries no trace of the held-out pair C."""
    task = state.task()
    return (
        state.held_out is None
        and state.held_out_index is None
        and all(pair.input != C.input and pair.output != C.output for pair in state.pairs)
        and C.task_pair() not in task["train"]
        and all(item["input"] != C.task_pair()["input"] for item in task["test"])
    )


def test_exclusion_of_the_held_out_pair_from_the_fit():
    state = demos()
    gate, fit, _ = gate_with([C.output])
    assert gate.fit(state) == "adapter-1"
    (handed,) = fit.calls
    assert held_out_free(handed)
    assert handed.pairs == (A, B) and handed.test_inputs == (TEST,)
    # The reference's training set is built from DemoSet.task(): the two remaining pairs, in
    # the challenges file's order, and the puzzle's own test input. No leave-one-out task is
    # generated (the reference has none), so this is the only training set.
    assert handed.task() == {"train": [A.task_pair(), B.task_pair()], "test": TASK["test"]}


def test_exclusion_of_the_held_out_pair_from_the_gates_prompts():
    state = demos()
    gate, _, decode = gate_with([C.output])
    gate.verdict("adapter", state)
    ((_, context, query, _),) = decode.calls
    # The held-out pair reaches the decoder as the query's input only: its output is what the
    # gate checks, and it is never a demonstration.
    assert held_out_free(context) and query == C.input
    prompt = context.task([query])
    assert prompt["train"] == [A.task_pair(), B.task_pair()]
    assert prompt["test"] == [{"input": [list(row) for row in C.input]}]


def test_every_state_handed_out_drops_the_held_out_pair():
    state = demos()
    assert state.held_out == Pair(C.input, C.output, kind="held_out")
    stripped = state.without_held_out()
    assert held_out_free(stripped) and stripped.pairs == state.pairs
    assert (stripped.version, stripped.test_inputs) == (state.version, state.test_inputs)
    # DemoSet.task() builds every dataset from pairs only, whatever the state holds.
    full = DemoSet("0a0b0c0d", (A, B), state.held_out, (TEST,), 0, 2)
    assert held_out_free(full.without_held_out())
    assert state.task() == stripped.task()


def test_a_held_out_pair_cannot_enter_the_training_pairs():
    held = Pair(C.input, C.output, kind="held_out")
    with pytest.raises(ValueError, match="never trained on"):
        DemoSet("0a0b0c0d", (A, held), None, (TEST,))
    with pytest.raises(ValueError, match="held-out Pair"):
        DemoSet("0a0b0c0d", (A, B), C, (TEST,), held_out_index=2)


# The hold-out rule: last real pair by index, a parameter; k >= 3, and k = 2 behind a flag.


def test_the_rule_holds_out_the_last_real_pair_by_default():
    state = demos()
    assert loop.DEFAULT_RULE == "last" and state.held_out_index == 2
    assert state.pairs == (A, B) and state.real_pairs == 3
    assert held_out_index([A, B, C, A]) == 3


def test_the_rule_is_a_parameter():
    first = demos(rule="first")
    assert first.held_out_index == 0 and first.held_out.input == A.input
    assert first.pairs == (B, C)
    middle = demos(rule=lambda pairs: len(pairs) // 2)
    assert middle.held_out_index == 1 and middle.pairs == (A, C)
    assert set(loop.HOLDOUT_RULES) == {"last", "first"}


@pytest.mark.parametrize(
    ("rule", "error"),
    [
        ("middle", ValueError),
        (3, TypeError),
        (lambda pairs: 5, ValueError),
        (lambda pairs: 1.0, TypeError),
    ],
    ids=["unknown-name", "not-callable", "out-of-range", "not-an-index"],
)
def test_a_bad_rule_is_refused(rule, error):
    with pytest.raises(error):
        held_out_index([A, B, C], rule)


def test_hold_out_applies_for_three_or_more_pairs():
    for count in (3, 4, 7):
        state = DemoSet.from_task("0a0b0c0d", {**TASK, "train": TASK["train"][:1] * count})
        assert state.held_out is not None and len(state.pairs) == count - 1
        assert DemoSet.from_task(
            "0a0b0c0d", {**TASK, "train": TASK["train"][:1] * count}, k2=False
        ).held_out is not None


def test_k2_flag_is_on_by_default_and_holds_out_one_of_two_pairs():
    two = {**TASK, "train": TASK["train"][:2]}
    state = DemoSet.from_task("0a0b0c0d", two)
    assert state.held_out.input == B.input and state.pairs == (A,) and state.real_pairs == 2


def test_k2_flag_off_keeps_both_pairs():
    two = {**TASK, "train": TASK["train"][:2]}
    state = DemoSet.from_task("0a0b0c0d", two, k2=False)
    assert state.held_out is None and state.held_out_index is None and state.pairs == (A, B)
    gate, _, _ = gate_with([B.output])
    with pytest.raises(ValueError, match="no held-out pair"):
        gate.passes("adapter", state)
    verdict = GateVerdict.not_applied(state)
    assert (verdict.applied, verdict.passed, verdict.real_pairs) == (False, None, 2)


def test_a_single_pair_is_never_held_out():
    one = {**TASK, "train": TASK["train"][:1]}
    assert DemoSet.from_task("0a0b0c0d", one).held_out is None
    assert held_out_index([A], "last", True) is None and held_out_index([], "last", True) is None


def test_k2_must_be_a_bool():
    with pytest.raises(TypeError):
        held_out_index([A, B], "last", 1)


# Edge cases: 30 by 30 grids, grid validation, the state's invariants.


def test_30x30_grids_pass_through_the_gate():
    big = Pair(
        [[(r + c) % 10 for c in range(30)] for r in range(30)], [[r % 10] * 30 for r in range(30)]
    )
    task = {
        "train": [A.task_pair(), B.task_pair(), big.task_pair()],
        "test": [{"input": big.task_pair()["input"]}],
    }
    state = DemoSet.from_task("0a0b0c0d", task)
    assert state.held_out.output == big.output and len(state.held_out.output) == 30
    gate, _, _ = gate_with([big.output] * 16)
    assert gate.passes("adapter", state)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ([[1, 2], [3]], ValueError),  # ragged
        ([], ValueError),  # no row
        ([[]], ValueError),  # no column
        ([[0] * 31], ValueError),  # 31 columns
        ([[0]] * 31, ValueError),  # 31 rows
        ([[10]], ValueError),  # color out of range
        ([[-1]], ValueError),
        ([[True]], TypeError),  # a bool is not a color
        ([[1.0]], TypeError),
        (["12"], TypeError),  # a string row
        ("12", TypeError),
        (7, TypeError),
    ],
)
def test_malformed_grids_are_refused(value, error):
    with pytest.raises(error):
        as_grid(value)


def test_grids_from_numpy_and_lists_are_equal():
    np = pytest.importorskip("numpy")
    assert as_grid(np.array([[1, 2], [3, 4]])) == as_grid([[1, 2], [3, 4]]) == ((1, 2), (3, 4))
    assert type(as_grid(np.array([[1]]))[0][0]) is int


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "pseudo"},  # round 0
        {"kind": "real", "round_added": 1},
        {"kind": "real", "agreement": 0.5},
        {"kind": "held_out", "round_added": 1},
        {"kind": "other"},
        {"kind": "pseudo", "round_added": 1, "agreement": float("nan")},
    ],
)
def test_pair_provenance_is_checked(fields):
    with pytest.raises(ValueError):
        Pair(A.input, A.output, **fields)


def test_a_pseudo_pair_carries_its_round_and_agreement():
    pair = Pair(A.input, A.output, kind="pseudo", round_added=1, agreement=0.75)
    assert (pair.kind, pair.round_added, pair.agreement) == ("pseudo", 1, 0.75)
    state = DemoSet("0a0b0c0d", (A, pair), None, (TEST,), version=1)
    assert state.real_pairs == 1 and state.version == 1


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("", (A,), None, (TEST,)), {}),
        (("0a0b0c0d", (A,), None, (TEST,)), {"version": -1}),
        (("0a0b0c0d", (A,), None, (TEST,)), {"held_out_index": 0}),
        (("0a0b0c0d", (A,), HELD_B, (TEST,)), {}),
        (("0a0b0c0d", (A,), HELD_B, (TEST,)), {"held_out_index": 2}),
    ],
    ids=[
        "no-id", "negative-version", "index-without-pair", "pair-without-index",
        "index-out-of-range",
    ],  # fmt: skip
)
def test_the_state_invariants_are_checked(args, kwargs):
    with pytest.raises(ValueError):
        DemoSet(*args, **kwargs)


def test_a_task_needs_a_pair_and_a_query():
    with pytest.raises(ValueError, match="no pair"):
        DemoSet("0a0b0c0d", (), None, (TEST,)).task()
    with pytest.raises(ValueError, match="no test input"):
        DemoSet("0a0b0c0d", (A,), None, ()).task()


def test_a_decoder_must_yield_candidates():
    gate = HeldOutGate(MockFit(), lambda adapter, state, query, views: [C.output])
    with pytest.raises(TypeError, match="Candidate"):
        gate.verdict("adapter", demos())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"views": 0}, ValueError), ({"views": 16.0}, TypeError), ({"select": 3}, TypeError)],
)
def test_bad_gate_settings_are_refused(kwargs, error):
    with pytest.raises(error):
        HeldOutGate(MockFit(), MockDecode([]), **kwargs)


def test_the_verdict_record_holds_every_field():
    clock = iter([10.0, 12.5])
    gate = HeldOutGate(MockFit(), MockDecode([C.output] * 3), clock=lambda: next(clock))
    record = gate.verdict("adapter", demos()).to_record()
    assert record == {
        "puzzle_id": "0a0b0c0d",
        "applied": True,
        "passed": True,
        "real_pairs": 3,
        "held_out_index": 2,
        "views": 16,
        "candidates": 3,
        "votes": [3],
        "expected_rank": 1,
        "seconds": 2.5,
        "error": None,
    }
    assert json.loads(json.dumps(record)) == record


def test_a_check_that_raises_is_neither_passed_nor_failed():
    verdict = GateVerdict.raised(demos(), MemoryError("CUDA out of memory"), 3.25)
    assert (verdict.applied, verdict.passed, verdict.held_out_index) == (True, None, 2)
    assert verdict.error == "MemoryError: CUDA out of memory" and verdict.seconds == 3.25
    assert verdict.to_record()["error"] == "MemoryError: CUDA out of memory"


# The run settings and the solver-side helpers.


def test_hold_out_settings_round_trip_through_the_solver_json():
    settings = HoldOut(gate_dir="/kaggle/gate_outputs/a", log_dir="/kaggle/working/run_logs/a/gate")
    assert (settings.rule, settings.k2, settings.in_test_prompts) == ("last", True, False)
    payload = json.loads(json.dumps(settings.to_json()))
    assert list(payload) == list(HoldOut.FIELDS)
    assert HoldOut.from_mapping(payload) == settings
    assert HoldOut.from_cfg({"holdout": payload}) == settings
    assert HoldOut.from_cfg({}) is None and HoldOut.from_cfg({"holdout": None}) is None


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        ({}, ValueError),
        ({"gate_dir": "/g", "views": 8}, ValueError),
        ({"gate_dir": ""}, ValueError),
        ({"gate_dir": "/g", "log_dir": ""}, ValueError),
        ({"gate_dir": Path("/g")}, TypeError),
        ({"gate_dir": "/g", "rule": "middle"}, ValueError),
        ({"gate_dir": "/g", "k2": 1}, TypeError),
        ({"gate_dir": "/g", "in_test_prompts": "no"}, TypeError),
        (["gate_dir"], TypeError),
    ],
)
def test_bad_hold_out_settings_are_refused(mapping, error):
    with pytest.raises(error):
        HoldOut.from_mapping(mapping)


def test_the_gates_decode_records_go_under_its_own_op_names(tmp_path):
    logger = timing.TimingLogger(tmp_path, "rank0")
    renamed = loop.RenamedOps(logger, loop.GATE_OPS)
    renamed.record("0a0b0c0d", timing.DECODE, 1.5, 40)
    renamed.record("0a0b0c0d", timing.SELECTION, 0.5, 80)
    renamed.record("0a0b0c0d", "other", 0.25)
    records, skipped = timing.read_records([logger.path])
    assert skipped == 0
    assert [(r.op, r.tokens) for r in records] == [
        (GATE_DECODE, 40), (GATE_SELECTION, 80), ("other", None)
    ]  # fmt: skip
    assert (GATE, GATE_DECODE, GATE_SELECTION) == ("gate", "gate_decode", "gate_selection")


def test_verdicts_are_appended_one_line_each(tmp_path):
    log = loop.VerdictLog(tmp_path / "gate", "rank2")
    gate, _, _ = gate_with([C.output])
    first = log.write(gate.verdict("adapter", demos()))
    second = log.write(GateVerdict.not_applied(DemoSet("0a0b0c0e", (A, B), None, (TEST,))))
    assert log.path == tmp_path / "gate" / "gate-rank2.jsonl"
    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert lines == [first, second] and lines[1]["applied"] is False


def test_candidate_files_are_read_back_in_name_order(tmp_path):
    np = pytest.importorskip("numpy")
    rows = {
        "0a0b0c0d_0.transpose.permute0123456789": [
            {"beam_score": 0.2, "score_aug": [1.0] * 8, "solution": np.array(C.output)},
        ],
        "0a0b0c0d_0.permute0123456789": [
            {"beam_score": 0.1, "score_aug": [0.5] * 8, "solution": np.array(C.output)},
            {"beam_score": 0.3, "score_aug": [2.0] * 8, "solution": np.array(WRONG)},
        ],
    }
    for name, value in rows.items():
        with bz2.BZ2File(tmp_path / name, "w") as handle:
            pickle.dump(value, handle)
    found = read_candidates(tmp_path)
    assert [(c.view, c.grid, c.beam_score) for c in found] == [
        ("0a0b0c0d_0.permute0123456789", C.output, 0.1),
        ("0a0b0c0d_0.permute0123456789", WRONG, 0.3),
        ("0a0b0c0d_0.transpose.permute0123456789", C.output, 0.2),
    ]
    assert found[1].score_aug == (2.0,) * 8
    assert read_candidates(tmp_path / "missing") == []


def test_importing_the_loop_needs_only_the_standard_library():
    heavy = (
        "numpy", "torch", "unsloth", "transformers", "arc26.nvarc.arc_decoder", "arc26.decode",
        "arc26.ttt",
    )  # fmt: skip
    code = f"import sys, arc26.loop; print(sorted(m for m in {heavy!r} if m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


# The split solver's worker with a held-out pair, against the stand-ins for the GPU stack.


def load_timing_tests():
    path = REPO / "tests" / "test_solver_timing.py"
    spec = importlib.util.spec_from_file_location("_gate_stand_ins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAND_INS = load_timing_tests()
# The stand-in modules of test_solver_timing's driver, up to its imports of the solver; the rest
# of this driver runs the split solver's local_worker with an identity DFS.
STAND_IN_HEAD, _IMPORT, _ = STAND_INS.DRIVER.partition("import numpy as np\n")
assert _IMPORT and "def stand_in(" in STAND_IN_HEAD

DRIVER = STAND_IN_HEAD + r'''
import numpy as np
from arc26 import decode, timing

holdout = json.loads(sys.argv[4])
# Cell count of a held-out input whose gate decode raises, as an out-of-memory error would.
raise_cells = int(sys.argv[5])
# Every training sample the reference's fit builds, per call, as the trainer receives them.
train_texts = []
datasets_stand_in = sys.modules["datasets"].Dataset
datasets_stand_in.from_list = classmethod(
    lambda cls, rows: train_texts.append([row["text"] for row in rows]) or cls(rows)
)
# Which decode is running: the test inputs' (into outputs_dir) or the gate's.
phase = ["test"]
prompts, score_prompts = [], []
moved_decode_puzzle = decode.decode_puzzle


def tagged_decode_puzzle(**kwargs):
    phase[0] = "test" if kwargs["dir_outputs"] == cfg["outputs_dir"] else "gate"
    return moved_decode_puzzle(**kwargs)


def identity_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    """Each view's answer is its query's input grid, so the gate passes exactly when the
    held-out pair maps its input to itself."""
    found = []
    for row, tokens in enumerate(prefix_tokens):
        text = "".join(map(chr, tokens))
        prompts.append([phase[0], text])
        query = text.rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        if phase[0] == "gate" and sum(c.isdigit() for c in query) == raise_cells:
            raise MemoryError("CUDA out of memory (stand-in)")
        found.append((row, [(0.25, [ord(c) for c in query] + [EOS])]))
    return found


def fake_scores(queries, answers, tokenizer, model):
    score_prompts.extend([phase[0], query] for query in queries)
    return [float(len(q) % 97 + len(a)) for q, a in zip(queries, answers)]


decode.decode_puzzle = tagged_decode_puzzle
decode.inference_turbo_dfs = identity_dfs
decode.calc_scores = fake_scores

from arc26 import solver

cfg = {
    "data_dir": os.path.join(work, "data"),
    "model_dir": os.path.join(work, "model"),
    "outputs_dir": os.path.join(work, "outputs"),
    "marker_dir": os.path.join(work, "markers"),
    "subset": None,
    "workers": 1,
    "timing_dir": timing_dir,
    "lora_r": 256,
    "lora_targets": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ],
}
if holdout is not None:
    cfg["holdout"] = holdout
os.makedirs(cfg["marker_dir"])
with open(os.path.join(cfg["data_dir"], "arc-agi_evaluation_challenges.json")) as handle:
    keys = sorted(json.load(handle))
tasks = queue.Queue()
for key in keys:
    tasks.put(key)
tasks.put(None)
solver.local_worker(0, tasks, time.time() + 1e6, cfg)


def digests(folder):
    found = {}
    for directory, _, files in sorted(os.walk(folder)):
        for name in sorted(files):
            path = os.path.join(directory, name)
            with open(path, "rb") as handle:
                found[os.path.relpath(path, folder)] = hashlib.sha256(handle.read()).hexdigest()
    return found


verdicts = []
if holdout is not None and holdout.get("log_dir"):
    with open(os.path.join(holdout["log_dir"], "gate-rank0.jsonl")) as handle:
        verdicts = [json.loads(line) for line in handle]
summary = {
    "candidates": digests(cfg["outputs_dir"]),
    "gate_files": digests(holdout["gate_dir"]) if holdout is not None else {},
    "train_texts": train_texts,
    "prompts": prompts,
    "score_prompts": score_prompts,
    "verdicts": verdicts,
}
with open(os.path.join(work, "summary.json"), "w") as handle:
    json.dump(summary, handle)
'''


def identity(rows):
    return {"input": rows, "output": rows}


# Synthetic tasks, never competition data. The last pair of each, the one the default rule
# holds out, has a cell count that no other grid of its task has.
TASKS = {
    # k = 3, the identity: the gate passes.
    "00000001": {
        "train": [
            identity([[1, 2], [3, 4]]),
            identity([[5, 6, 7]]),
            identity([[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]]),
        ],
        "test": [{"input": [[8, 9], [9, 8]]}],
    },
    # k = 3, colors 1 and 2 swapped: the identity answer is wrong, so the gate fails.
    "00000002": {
        "train": [
            {"input": [[1, 2]], "output": [[2, 1]]},
            {"input": [[1, 1, 2, 2, 1]], "output": [[2, 2, 1, 1, 2]]},
            {
                "input": [[1, 2, 1, 2, 1], [2, 2, 1, 1, 2]],
                "output": [[2, 1, 2, 1, 2], [1, 1, 2, 2, 1]],
            },
        ],
        "test": [{"input": [[1, 2, 1]]}, {"input": [[2, 2], [1, 1]]}],
    },
    # k = 2, the identity: held out with the k2 flag on, kept whole with it off.
    "00000003": {
        "train": [identity([[3, 4], [4, 3], [3, 3]]), identity([[5, 6, 5], [6, 5, 6], [5, 5, 5]])],
        "test": [{"input": [[7, 7], [6, 6]]}],
    },
}
# No two tasks share their demonstrations' cell counts, so a prompt tells its task (task_of).
HELD_CELLS = {"00000001": 12, "00000002": 10, "00000003": 9}
REAL_PAIRS = {"00000001": 3, "00000002": 3, "00000003": 2}
# Decode batches per task: 16 augmented views per test input, 4 views per batch.
BATCHES = {"00000001": 4, "00000002": 8, "00000003": 4}
USER = "<|im_start|>user\n"
ASSISTANT = "<|im_start|>assistant\n"
END = "<|im_end|>"


class Run(NamedTuple):
    work: Path
    summary: dict
    output: str
    records: list


def run_holdout_worker(work, holdout, raise_cells=0):
    """Run the split solver's local_worker on TASKS with the given hold-out settings; the gate's
    decode raises for the held-out input of ``raise_cells`` cells."""
    pytest.importorskip("numpy")
    data = work / "data"
    data.mkdir(parents=True)
    (data / "arc-agi_evaluation_challenges.json").write_text(json.dumps(TASKS), encoding="utf-8")
    driver = work.parent / f"{work.name}-driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    timing_dir = work / "run_logs" / "timing"
    if holdout is not None:
        holdout = {
            **holdout,
            "gate_dir": str(work / "gate_outputs"),
            "log_dir": str(work / "run_logs" / "gate"),
        }
    env = {key: value for key, value in os.environ.items() if key != "KAGGLE_IS_COMPETITION_RERUN"}
    env.update(PYTHONHASHSEED="0", PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [
            sys.executable, str(driver), str(work), str(timing_dir), "plain", json.dumps(holdout),
            str(raise_cells),
        ],  # fmt: skip
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    lines = (timing_dir / "timing-rank0.jsonl").read_text(encoding="utf-8").splitlines()
    return Run(work, summary, result.stdout, [json.loads(line) for line in lines])


SETTINGS = {
    "off": None,
    "on": {"rule": "last", "k2": True, "in_test_prompts": False},
    "prompts": {"rule": "last", "k2": True, "in_test_prompts": True},
    "k2-off": {"rule": "last", "k2": False, "in_test_prompts": False},
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate")
    found = {name: run_holdout_worker(root / name, holdout) for name, holdout in SETTINGS.items()}
    # The second task's gate raises; the worker goes on to the third.
    found["raises"] = run_holdout_worker(root / "raises", SETTINGS["on"], HELD_CELLS["00000002"])
    return found


def turns(text):
    """The demonstration pairs and the query of a formatted text: (input, output) grids as
    strings, then the trailing query's input or None."""
    pairs, query = [], None
    for part in text.split(USER)[1:]:
        given, _, rest = part.partition(END)
        if rest.startswith(ASSISTANT):
            answer = rest[len(ASSISTANT) :].partition(END)[0]
            if answer:
                pairs.append((given, answer))
                continue
        query = given
    return pairs, query


def cells(text):
    return len(re.findall(r"\d", text))


def by_task(texts_per_call):
    """Training texts per fit call: one call per puzzle, in queue order."""
    assert len(texts_per_call) == len(TASKS)
    return dict(zip(sorted(TASKS), texts_per_call, strict=True))


def task_of(text):
    """The task whose test or held-out query a prompt holds, from its demonstrations' sizes."""
    pairs, _ = turns(text)
    sizes = {cells(given) for given, _ in pairs}
    for key, task in TASKS.items():
        if sizes <= {cells(str(item["input"])) for item in task["train"]}:
            return key
    raise AssertionError(f"no task has demonstrations of {sizes} cells")


def test_exclusion_of_the_held_out_pair_from_every_training_path(runs):
    """The reference's fit trains on 128 augmented samples per puzzle, each the puzzle's pairs as
    one conversation; with a held-out pair none of them holds it. The off run is the control:
    every sample holds every pair there, so the check would see a leak."""
    for name in ("on", "prompts"):
        for key, texts in by_task(runs[name].summary["train_texts"]).items():
            assert len(texts) == 128
            for text in texts:
                pairs, query = turns(text)
                assert len(pairs) == REAL_PAIRS[key] - 1 and query is None
                assert HELD_CELLS[key] not in {cells(g) for pair in pairs for g in pair}
    for key, texts in by_task(runs["off"].summary["train_texts"]).items():
        assert {len(turns(text)[0]) for text in texts} == {REAL_PAIRS[key]}
        assert all(HELD_CELLS[key] in {cells(g) for g, _ in turns(text)[0]} for text in texts)


def test_a_puzzle_that_keeps_every_pair_trains_on_all_of_them(runs):
    texts = by_task(runs["k2-off"].summary["train_texts"])
    assert {len(turns(text)[0]) for text in texts["00000003"]} == {2}
    assert {len(turns(text)[0]) for text in texts["00000001"]} == {2}


def test_the_gate_decodes_the_held_out_input_with_the_other_pairs_only(runs):
    """In every gate prompt, DFS and rescoring alike, the held-out input is the query and the
    demonstrations are the other pairs: its output appears in none."""
    on = runs["on"].summary
    gate_prompts = [text for phase, text in on["prompts"] if phase == "gate"]
    assert len(gate_prompts) == 16 * len(TASKS)
    for text in gate_prompts + [q for phase, q in on["score_prompts"] if phase == "gate"]:
        key = task_of(text)
        pairs, query = turns(text)
        assert len(pairs) == REAL_PAIRS[key] - 1
        assert cells(query) == HELD_CELLS[key]
        assert HELD_CELLS[key] not in {cells(g) for pair in pairs for g in pair}


def test_the_test_prompts_leave_the_held_out_pair_out_unless_asked(runs):
    """ADR-013: the test inputs are decoded and rescored with the training pairs only; with
    in_test_prompts the held-out pair stays in their prompts, as without a hold-out."""
    expected = {"on": -1, "prompts": 0, "off": 0}
    for name, offset in expected.items():
        summary = runs[name].summary
        tests = [t for p, t in summary["prompts"] + summary["score_prompts"] if p == "test"]
        assert len([t for p, t in summary["prompts"] if p == "test"]) == 16 * 4
        for text in tests:
            key = task_of(text)
            pairs, query = turns(text)
            assert len(pairs) == REAL_PAIRS[key] + offset
            assert cells(query) != HELD_CELLS[key]


def test_the_gate_writes_nothing_where_the_real_outputs_go(runs):
    """With the held-out pair in the test prompts, the test decode is the reference's, so the
    candidate folder is byte for byte the run without a hold-out (the stand-in fit changes no
    weight); the gate's files are in its own folder, one subfolder per puzzle."""
    off, on, prompts = (runs[name].summary for name in ("off", "on", "prompts"))
    assert off["candidates"] and prompts["candidates"] == off["candidates"]

    def views(files):
        # Candidate files per decoded input; a file's name is its view's key, which also spells
        # the demonstrations' shuffled order (".ex120" with three pairs, ".ex10" with two).
        counts = {}
        for name in files:
            subkey = name.split(".")[0]
            counts[subkey] = counts.get(subkey, 0) + 1
        return counts

    # One file per view of each real test input, and no other: the gate's 16 files per puzzle,
    # named as test input 0's would be, are elsewhere.
    tests = {"00000001_0": 16, "00000002_0": 16, "00000002_1": 16, "00000003_0": 16}
    assert views(on["candidates"]) == views(off["candidates"]) == tests
    for summary, gated in ((on, set(TASKS)), (prompts, set(TASKS))):
        folders = {name.split(os.sep)[0] for name in summary["gate_files"]}
        assert folders == gated
        assert views(name.split(os.sep)[1] for name in summary["gate_files"]) == {
            f"{key}_0": 16 for key in gated
        }
    k2_off = runs["k2-off"].summary["gate_files"]
    assert {name.split(os.sep)[0] for name in k2_off} == {"00000001", "00000002"}


def test_the_gate_verdicts_pass_and_fail(runs):
    verdicts = {v["puzzle_id"]: v for v in runs["on"].summary["verdicts"]}
    assert list(verdicts) == sorted(TASKS)
    assert [verdicts[key]["passed"] for key in sorted(TASKS)] == [True, False, True]
    passed, failed, two = (verdicts[key] for key in sorted(TASKS))
    assert passed["expected_rank"] == 1 and passed["votes"] == [16] and passed["candidates"] == 16
    # Every view answers the held-out input itself, which is not its output.
    assert failed["expected_rank"] is None and failed["votes"] == [16]
    assert (two["real_pairs"], two["held_out_index"], two["applied"]) == (2, 1, True)
    assert all(v["views"] == 16 for v in verdicts.values())
    output = runs["on"].output
    assert "[Rank 0] gate for 00000001: decoding the held-out pair, index 2 of 3 real pairs" in (
        output
    )
    assert "[Rank 0] gate for 00000001: passed, 16 candidates, the held-out output ranked 1" in (
        output
    )
    assert "[Rank 0] gate for 00000002: failed, 16 candidates, the held-out output not among" in (
        output
    )


def test_a_puzzle_that_keeps_every_pair_runs_no_gate(runs):
    verdicts = {v["puzzle_id"]: v for v in runs["k2-off"].summary["verdicts"]}
    assert verdicts["00000003"] == {
        "puzzle_id": "00000003", "applied": False, "passed": None, "real_pairs": 2,
        "held_out_index": None, "views": 0, "candidates": 0, "votes": [], "expected_rank": None,
        "seconds": 0.0, "error": None,
    }  # fmt: skip
    assert [verdicts[key]["passed"] for key in ("00000001", "00000002")] == [True, False]
    assert "[Rank 0] gate for 00000003: not applied, 2 real pairs" in runs["k2-off"].output
    # Its test decode is the reference's: the same files as without a hold-out.
    off, k2_off = runs["off"].summary["candidates"], runs["k2-off"].summary["candidates"]
    assert {n: d for n, d in k2_off.items() if n.startswith("00000003")} == {
        n: d for n, d in off.items() if n.startswith("00000003")
    }


def test_the_gate_records_its_own_timing_ops(runs):
    """Per puzzle: train, the test inputs' decode and selection as without a hold-out, then the
    gate's gate_decode and gate_selection, its gate span, all inside the puzzle span."""
    for name in ("on", "k2-off"):
        groups = STAND_INS.by_puzzle(runs[name].records)
        off = STAND_INS.by_puzzle(runs["off"].records)
        for key, group in groups.items():
            ops = [r["op"] for r in group]
            gated = not (name == "k2-off" and key == "00000003")
            assert ops[0] == "train" and ops[-1] == "puzzle"
            reference = [r["op"] for r in off[key]]
            assert ops.count("decode") == BATCHES[key] == reference.count("decode")
            if not gated:
                assert not {GATE, GATE_DECODE, GATE_SELECTION} & set(ops)
                continue
            assert ops.count(GATE_DECODE) == 4 and ops.count(GATE) == 1 and ops[-2] == GATE
            assert ops.index(GATE_DECODE) > max(i for i, op in enumerate(ops) if op == "decode")
            assert set(ops[ops.index(GATE_DECODE) : -2]) == {GATE_DECODE, GATE_SELECTION}
            gate = next(r for r in group if r["op"] == GATE)
            inner = sum(r["seconds"] for r in group if r["op"] in (GATE_DECODE, GATE_SELECTION))
            assert gate["seconds"] >= inner - 1e-6 * len(group)
            # The gate span holds its own decode and rescoring records.
            outer = [r for r in group[:-1] if r["op"] not in (GATE_DECODE, GATE_SELECTION)]
            assert group[-1]["seconds"] >= sum(r["seconds"] for r in outer) - 1e-3
            assert all(r["tokens"] for r in group if r["op"] == GATE_DECODE)


def test_a_gate_that_raises_is_logged_and_the_worker_goes_on(runs):
    """torch.multiprocessing.spawn stops every worker when one raises: the gate's failure is
    logged with its error, the puzzle keeps its test candidates, and the next puzzle runs."""
    run = runs["raises"]
    verdicts = {v["puzzle_id"]: v for v in run.summary["verdicts"]}
    assert list(verdicts) == sorted(TASKS)
    failed = verdicts["00000002"]
    assert failed["applied"] is True and failed["passed"] is None
    assert failed["error"] == "MemoryError: CUDA out of memory (stand-in)"
    assert [verdicts[key]["passed"] for key in ("00000001", "00000003")] == [True, True]
    assert "[Rank 0] gate for 00000002: raised MemoryError" in run.output
    assert run.summary["candidates"] == runs["on"].summary["candidates"]
    groups = STAND_INS.by_puzzle(run.records)
    assert list(groups) == sorted(TASKS)
    ops = [r["op"] for r in groups["00000002"]]
    assert GATE not in ops and ops[-1] == "puzzle" and GATE_DECODE not in ops
