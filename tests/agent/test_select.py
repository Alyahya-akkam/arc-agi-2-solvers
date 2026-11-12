"""Tests of attempt selection (arc26/agent/select.py, P6.2) on hand-built records."""

import hashlib
import json

import numpy as np
import pytest

from arc26.agent.select import (
    UNPARSABLE_SIZE,
    build_selection,
    build_submission,
    load_nvarc,
    nvarc_from_submission,
    program_size,
    rank_candidates,
    submission_from_selection,
)
from arc26.agent.types import AttemptRecord, Candidate, GuardFinding, PairResult, Verification
from arc26.submit import validate

A, B, C, D, E = [[1]], [[2]], [[3]], [[4]], [[5]]
SMALL = "def transform(grid):\n    return grid\n"
LARGE = "def transform(grid):\n    x = [row[:] for row in grid]\n    return x\n"
DEMOTE = GuardFinding("shape_dispatch", "branches on a demonstration size", False)
FATAL = GuardFinding("literal_output", "embeds output 0", True)


def program(tests, source=SMALL, passed=2, n=2, accuracy=0.5, guards=()):
    """A verification with ``passed`` of ``n`` pairs exact and the given test outputs."""
    pairs = tuple(
        PairResult(i, i < passed, [[0]], None, True, 1.0 if i < passed else accuracy)
        for i in range(n)
    )
    sha = hashlib.sha256(source.encode()).hexdigest()
    return Verification(source, sha, pairs, tuple(tests), tuple(guards), 0.0)


def record(attempt_id, *verifications, answers=(), task_id="t"):
    return AttemptRecord(
        task_id, attempt_id, 1, tuple(verifications), tuple(answers), 10, 20, 1, 0.1, "rounds"
    )


def grids(candidates):
    return [c.grid for c in candidates]


def test_program_size_counts_ast_nodes():
    assert program_size("x = 1") == 5  # Module, Assign, Name, Store, Constant
    assert program_size(SMALL) < program_size(LARGE)
    assert program_size("def (:") == UNPARSABLE_SIZE


def test_votes_rank_first_and_count_attempts_not_programs():
    records = [
        record("t/p1/0", program([A]), program([A], LARGE)),  # one attempt, two programs
        record("t/p1/1", program([B], LARGE)),
        record("t/p1/2", program([B], LARGE)),
    ]
    ranked = rank_candidates("t", 0, records)
    assert grids(ranked) == [B, A]
    assert [(c.source, c.votes) for c in ranked] == [("program", 2), ("program", 1)]


def test_ties_break_by_size_then_demotion_then_first_seen():
    by_size = [record("a", program([A], LARGE)), record("b", program([B], SMALL))]
    assert grids(rank_candidates("t", 0, by_size)) == [B, A]
    by_flag = [record("a", program([A], guards=[DEMOTE])), record("b", program([B]))]
    ranked = rank_candidates("t", 0, by_flag)
    assert grids(ranked) == [B, A]
    assert [c.demoted for c in ranked] == [False, True]
    by_order = [record("a", program([C])), record("b", program([A]))]
    assert grids(rank_candidates("t", 0, by_order)) == [C, A]


def test_group_is_demoted_only_if_every_program_is():
    records = [record("a", program([A], guards=[DEMOTE])), record("b", program([A]))]
    (only,) = rank_candidates("t", 0, records)
    assert not only.demoted and only.votes == 2
    assert only.size == program_size(SMALL)


def test_nvarc_fills_attempt_two_when_it_differs():
    ranked = rank_candidates("t", 0, [record("a", program([A]))], nvarc=(C, D))
    assert grids(ranked) == [A, C, D]
    assert [c.source for c in ranked] == ["program", "nvarc", "nvarc"]
    assert ranked[1].votes == 1 and ranked[1].size == 0


def test_nvarc_top_equal_to_a_verified_output_goes_first():
    records = [record("a", program([A])), record("b", program([A])), record("c", program([B]))]
    ranked = rank_candidates("t", 0, records, nvarc=(B, C))
    assert grids(ranked) == [B, A, C]
    assert ranked[0].source == "program" and ranked[0].votes == 1


def test_nvarc_second_attempt_is_the_last_resort_after_a_verified_group():
    records = [record("a", program([A]), answers=[(0, D)]), record("b", program([E], passed=1))]
    ranked = rank_candidates("t", 0, records, nvarc=(A, C))
    assert grids(ranked) == [A, D, E, C]


def test_direct_answer_then_best_unverified_fill_attempt_two():
    records = [
        record("a", program([A]), program([C], passed=1, accuracy=0.9)),
        record("b", answers=[(0, D)]),
    ]
    assert grids(rank_candidates("t", 0, records)) == [A, D, C]
    no_direct = [record("a", program([A]), program([C], passed=1, accuracy=0.9))]
    ranked = rank_candidates("t", 0, no_direct)
    assert grids(ranked) == [A, C]
    assert (ranked[1].source, ranked[1].votes) == ("program", 0)


def test_without_verified_programs_nvarc_takes_both_attempts():
    records = [record("a", program([C], passed=1), answers=[(0, D)])]
    assert grids(rank_candidates("t", 0, records, nvarc=(A, B)))[:2] == [A, B]
    # NVARC's attempts that repeat each other leave room for the next source.
    assert grids(rank_candidates("t", 0, records, nvarc=(A, A)))[:2] == [A, D]
    # An invalid NVARC grid is skipped.
    assert grids(rank_candidates("t", 0, records, nvarc=(None, [[10]])))[:2] == [D, C]


def test_direct_answers_rank_by_votes_then_first_seen():
    records = [
        record("a", answers=[(0, C), (1, E)]),
        record("b", answers=[(0, D)]),
        record("c", answers=[(0, D), (0, D)]),
    ]
    ranked = rank_candidates("t", 0, records)
    assert grids(ranked) == [D, C]
    assert [(c.source, c.votes) for c in ranked] == [("direct", 2), ("direct", 1)]
    assert grids(rank_candidates("t", 1, records)) == [E]


def test_unverified_programs_rank_by_cell_accuracy_and_skip_fatal_ones():
    records = [
        record("a", program([A], passed=0, accuracy=0.4)),
        record("b", program([B], passed=1, accuracy=0.95)),
        record("c", program([C], passed=1, accuracy=1.0, guards=[FATAL])),
        record("d", program([None], passed=1, accuracy=1.0)),
    ]
    assert grids(rank_candidates("t", 0, records)) == [B, A]


def test_verified_needs_every_test_output_but_unverified_uses_what_exists():
    partial = program([A, None])  # passes every demonstration, fails on test input 1
    assert not partial.verified
    records = [record("a", partial)]
    assert grids(rank_candidates("t", 0, records)) == [A]
    assert rank_candidates("t", 0, records)[0].votes == 0
    assert rank_candidates("t", 1, records) == []


def test_grids_are_never_repeated_across_sources():
    records = [record("a", program([A]), answers=[(0, A)])]
    ranked = rank_candidates("t", 0, records, nvarc=(A, A))
    assert grids(ranked) == [A]


def test_records_of_other_tasks_are_ignored():
    records = [record("x", program([A]), task_id="other")]
    assert rank_candidates("t", 0, records) == []


def test_candidates_carry_their_slot():
    (candidate,) = rank_candidates("t", 0, [record("a", program([A]))])
    assert candidate == Candidate("t", 0, A, "program", 1, program_size(SMALL), False)


def test_build_submission_covers_every_task_and_test_input():
    records = [
        record("t1/p1/0", program([A, B]), task_id="t1"),
        record("x/p1/0", program([C]), task_id="not in the challenges"),
    ]
    nvarc = {"t2": [(D, None)], "t3": [(E, A), (B, C)]}
    counts = {"t1": 2, "t2": 1, "t3": 2, "t4": 1}
    submission = build_submission(counts, records, nvarc)
    assert submission == {
        "t1": [{"attempt_1": A, "attempt_2": [[0]]}, {"attempt_1": B, "attempt_2": [[0]]}],
        "t2": [{"attempt_1": D, "attempt_2": [[0]]}],
        "t3": [{"attempt_1": E, "attempt_2": A}, {"attempt_1": B, "attempt_2": C}],
        "t4": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    challenges = {task: {"test": [{}] * n} for task, n in counts.items()}
    assert validate(submission, challenges) == []


def test_selection_keeps_candidates_and_fills_empty_slots():
    selection = build_selection({"t": 1}, [record("a", program([A]))])
    assert [[c.source for c in entry] for entry in selection["t"]] == [["program"]]
    assert submission_from_selection(selection) == {"t": [{"attempt_1": A, "attempt_2": [[0]]}]}
    empty = submission_from_selection(build_selection({"t": 1}, []))
    assert empty == {"t": [{"attempt_1": [[0]], "attempt_2": [[0]]}]}
    assert empty["t"][0]["attempt_1"] is not empty["t"][0]["attempt_2"]


def test_nvarc_submission_parsing(tmp_path):
    data = {
        "t1": [{"attempt_1": A, "attempt_2": [[1, 2], [3]]}, "not an entry", {"attempt_2": B}],
        "t2": "not a list",
        "t3": [{"attempt_1": C, "attempt_2": [[0]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    assert nvarc_from_submission(data) == {
        "t1": [(A, None), (None, None), (None, B)],
        "t3": [(C, None), (None, None)],  # [[0]] is NVARC's placeholder, not an attempt
    }
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_nvarc(path) == nvarc_from_submission(data)
    with pytest.raises(ValueError, match="JSON object"):
        nvarc_from_submission([data])


def test_nvarc_placeholders_count_as_missing():
    records = [record("a", program([A]), answers=[(0, D)])]
    assert grids(rank_candidates("t", 0, records, nvarc=([[0]], [[0]]))) == [A, D]


def test_one_real_nvarc_grid_and_its_placeholder_take_one_slot():
    records = [record("a", answers=[(0, D)])]
    ranked = rank_candidates("t", 0, records, nvarc=(C, [[0]]))
    assert grids(ranked) == [C, D]
    assert [c.source for c in ranked] == ["nvarc", "direct"]
    submission = build_submission({"t": 1}, records, {"t": [(C, [[0]])]})
    assert submission == {"t": [{"attempt_1": C, "attempt_2": D}]}


def test_direct_answer_ranks_after_nvarc_first_attempt():
    records = [record("a", program([A]), answers=[(0, D)])]
    ranked = rank_candidates("t", 0, records, nvarc=(C, E))
    assert grids(ranked) == [A, C, D, E]
    assert [c.source for c in ranked] == ["program", "nvarc", "direct", "nvarc"]


def test_size_ranks_before_demotion():
    small_demoted = record("a", program([A], SMALL, guards=[DEMOTE]))
    large_clean = record("b", program([B], LARGE))
    ranked = rank_candidates("t", 0, [large_clean, small_demoted])
    assert grids(ranked) == [A, B]
    assert [(c.size < program_size(LARGE), c.demoted) for c in ranked] == [
        (True, True),
        (False, False),
    ]


def test_numpy_grids_in_records_are_used_as_plain_lists():
    tests = [np.array([[7, 8]], dtype=np.int64)]
    records = [record("a", program(tests), answers=[(0, np.array([[9]]))])]
    ranked = rank_candidates("t", 0, records, nvarc=(np.array([[6]]), None))
    assert grids(ranked) == [[[7, 8]], [[6]], [[9]]]
    assert all(type(c.grid[0][0]) is int for c in ranked)
