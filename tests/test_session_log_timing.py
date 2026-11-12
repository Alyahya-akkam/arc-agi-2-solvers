"""Session log parsing, timing records, and summary of scripts/session_log_timing.py (P1.1).

Synthetic logs only: made-up task ids, grids, and numbers, no competition data. The log has two
ranks with interleaved lines, a puzzle that hits the decode cap, a puzzle with no candidates, a
puzzle cut by a hard stop, the ``total_flos`` conversion, and truncated or noisy entries.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from arc26.timing import read_records

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "session_log_timing", REPO / "scripts" / "session_log_timing.py"
)
slt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = slt  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(slt)

P = slt.REFERENCE_PARAMS
A, B, C, D = "aaaa0001", "bbbb0002", "cccc0003", "dddd0004"
# B's grids are all 2 by 2 with 2 pairs (its K0 statistics below): 128 steps of
# 2 * (2 * (4 + 2 - 1) + 8) = 36 tokens.
B_TOKENS = 128 * 36


def entry(time, text, stream="stdout"):
    return {"stream_name": stream, "time": time, "data": text + "\n"}


def stats(rank, task, runtime, tokens, steps=128, loss=0.25):
    metrics = {
        "train_runtime": runtime,
        "train_samples_per_second": 1.0,
        "train_steps_per_second": 1.0,
        "total_flos": float(6 * P * tokens),
        "train_loss": loss,
        "epoch": 1.0,
    }
    return (
        f"[Rank {rank}] training stats for puzzle {task}: TrainOutput(global_step={steps}, "
        f"training_loss={loss}, metrics={metrics})"
    )


def decoding(rank, *subkeys):
    return f"[Rank {rank}] decoding {list(subkeys)}"


def worker_entries():
    """Rank 0: A (first, two candidates scored), D (cut by the hard stop). Rank 1: B (first, no
    candidates), C (stops at the decode cap). Starts: A at 130 - 100 = 30, B at 110.5 - 69.5 = 41.
    """
    return [
        entry(5.0, "== run, 1.0 s after the start"),
        entry(5.5, "solver: end time at 41400 s, hard stop in 41974 s"),
        entry(20.0, "[Rank 0] start!"),
        {"stream_name": "stdout", "time": 21.0, "data": "\rLoading 50%\rLoading 100%\n"},
        entry(25.0, "[Rank 1] start!"),
        {"stream_name": "stdout", "time": "abc", "data": "not a time\n"},
        {"stream_name": "stdout", "time": True, "data": "a boolean time\n"},
        {"stream_name": "stdout", "time": 50.0},
        42,
        entry(75.0, "[Rank 0] allocated 13000MB for training"),
        entry(75.0, stats(0, A, 40.0, 1920)),
        entry(76.0, decoding(0, f"{A}_0.permute0123456789.ex01", f"{A}_0.rot90.permute0123.ex01")),
        {
            "stream_name": "stdout",
            "time": 77.0,
            "data": "\rLoading 100%\r[Rank 1] allocated 13000MB for training\n",
        },
        entry(77.0, stats(1, B, 30.0, B_TOKENS)),
        entry(78.0, decoding(1, f"{B}_0.permute0123456789.ex01")),
        entry(96.0, f"[Rank 0] scoring {A}_0.permute0123456789.ex01 #0"),
        entry(98.0, decoding(1, f"{B}_0.rot90.permute0123456789.ex01")),
        entry(99.0, f"[Rank 0] scoring {A}_0.rot90.permute0123.ex01 #0"),
        entry(100.0, "UserWarning: a library warning between worker lines", stream="stderr"),
        entry(102.0, decoding(0, f"{A}_0.transpose.permute0123456789.ex01")),
        {
            "stream_name": "stdout",
            "time": 110.5,
            "data": f"[Rank 1] allocated 11000MB for inference\n[Rank 1] finished {B} in 69.5s\n",
        },
        entry(130.0, "[Rank 0] allocated 11000MB for inference"),
        entry(130.0, f"[Rank 0] finished {A} in 100.0s"),
        entry(180.0, "[Rank 0] allocated 13000MB for training"),
        entry(180.0, stats(0, D, 45.0, 640)),
        entry(181.0, decoding(0, f"{D}_0.permute0123456789.ex01")),
        entry(190.0, "[Rank 0] scor"),
        entry(600.0, "[Rank 1] allocated 13000MB for training"),
        entry(600.0, stats(1, C, 480.0, 6400)),
        entry(601.0, decoding(1, f"{C}_0.permute0123456789.ex012")),
        entry(1101.0, f"[Rank 1] scoring {C}_0.permute0123456789.ex012 #0"),
        entry(1105.0, decoding(1, f"{C}_1.permute0123456789.ex012")),
        entry(1320.0, f"[Rank 1] timeout after 1209.6s for puzzle {C}"),
        entry(1320.1, "[Rank 1] allocated 11000MB for inference"),
        entry(1320.1, f"[Rank 1] finished {C} in 1209.7s"),
        entry(1321.0, "[Rank 1] done!"),
    ]


def notebook_entries():
    """The notebook's development-mode lines after the solver (cell 7)."""
    return [
        entry(1322.0, "solver exit code: 0"),
        entry(1330.0, "*** Benchmark selection algorithms..."),
        entry(1330.1, f"ALL_CORRECT: 0.50000 -  2.00000 1x1   [{A}_0.permute0123456789.ex01.out1]"),
        entry(
            1330.1, f"ALL_CORRECT: 0.30000 -  0.30000 1x1   [{C}_0.permute0123456789.ex012.out2]"
        ),
        entry(
            1330.1, f"ALL_CORRECT: 0.05000 -  0.05000 2x2   [{C}_1.permute0123456789.ex012.out0]"
        ),
        entry(1330.2, " subkeys: 3/7"),
        entry(1330.3, " acc:   1.5/  2 ('score_full_probmul_3')"),
        entry(1330.4, " acc:   1.5/  2 ('score_kgmon')"),
        entry(1330.5, "*** Reload score: 1.5"),
    ]


def truncated_log():
    """The worker lines, cut in the middle of one more entry as a hard kill leaves the file."""
    text = json.dumps(worker_entries())
    return text[:-1] + ', {"stream_name": "stdout", "time": 1400.0, "data": "[Rank 0] dec'


@pytest.fixture
def records_dir(tmp_path):
    log = tmp_path / "session.log"
    log.write_text(truncated_log(), encoding="utf-8")
    out = tmp_path / "timing"
    result = slt.write_records(log, out)
    return result, out


# Token conversion


def test_reference_params_is_the_derived_count():
    assert P == 3_633_511_936 + 528_482_304 + 32 * 2560 == 4_162_076_160


def test_flos_to_tokens_inverts_six_p_t():
    assert slt.flos_to_tokens(float(6 * P * 1920)) == (1920, 0.0)
    tokens, residual = slt.flos_to_tokens(float(6 * P * 1920), params=P - 40_960)
    assert tokens == 1920 and residual > 0.01  # a neighboring P leaves a fraction


@pytest.mark.parametrize("flos", [-1.0, float("inf"), float("nan")])
def test_flos_to_tokens_refuses_impossible_counts(flos):
    with pytest.raises(ValueError):
        slt.flos_to_tokens(flos)


def test_flos_to_tokens_refuses_a_non_positive_parameter_count():
    with pytest.raises(ValueError):
        slt.flos_to_tokens(6.0, params=0)


def test_square_sequence_tokens_follows_the_training_format():
    # Per pair: two 3x3 grids of 9 digits and 2 newlines, and 8 markers.
    assert slt.square_sequence_tokens(3, 2) == (2 * (2 * 11 + 8), 2)
    # Five 30x30 pairs exceed 8192 tokens; cut_to_len keeps four (7464 tokens).
    assert slt.square_sequence_tokens(30, 5) == (4 * 1866, 4)


def test_fit_two_recovers_an_exact_plane():
    x1 = [1.0, 2.0, 3.0, 5.0, 8.0]
    x2 = [x * x / 4 for x in x1]
    ys = [2.0 + 0.5 * u + 0.25 * v for u, v in zip(x1, x2, strict=True)]
    a, b, c, r2 = slt.fit_two(x1, x2, ys)
    assert (a, b, c, r2) == pytest.approx((2.0, 0.5, 0.25, 1.0))


@pytest.mark.parametrize("x1, x2", [([1.0, 2.0], [1.0, 4.0]), ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])])
def test_fit_two_refuses_too_few_or_collinear_samples(x1, x2):
    with pytest.raises(ValueError):
        slt.fit_two(x1, x2, [1.0] * len(x1))


# Session log


def test_truncated_and_noisy_entries_are_skipped():
    lines, skipped, truncated = slt.parse_session_log(truncated_log())
    assert truncated is True
    assert skipped == 4  # a string time, a boolean time, no data, a bare number
    assert len(lines) == 33  # 32 valid entries, one of them two lines
    assert [line.time for line in lines] == sorted(line.time for line in lines)
    assert lines[3].text == "Loading 100%"  # only the text after the last carriage return
    assert not any(line.text.startswith('{"stream') for line in lines)


def test_lines_are_sorted_by_time_stably():
    text = json.dumps([entry(2.0, "b"), entry(1.0, "a"), entry(2.0, "c")])
    lines, skipped, truncated = slt.parse_session_log(text)
    assert [line.text for line in lines] == ["a", "b", "c"]
    assert (skipped, truncated) == (0, False)


def test_a_log_that_is_not_an_array_yields_no_lines():
    assert slt.parse_session_log("no log here") == ([], 0, True)
    assert slt.parse_session_log('{"time": 1.0, "data": "x\\n"}')[0][0].text == "x"


def test_task_of_reads_the_task_from_a_subkey():
    assert slt.task_of(f"{A}_1.transpose.rot90.permute0123456789.ex012") == A
    assert slt.task_of(f"{A}_0.permute0123456789.ex01.out3") == A


# Ranks and puzzles


def parsed():
    lines, _, _ = slt.parse_session_log(truncated_log())
    return slt.parse_ranks(lines)


def test_interleaved_ranks_get_their_own_intervals():
    ranks = parsed()
    assert sorted(ranks) == [0, 1]
    first, cut = ranks[0].puzzles
    assert first.puzzle_id == A and first.intervals == [
        ("decode", 20.0),
        ("selection", 3.0),
        ("selection", 3.0),
        ("decode", 28.0),
    ]
    assert (first.decode_batches, first.scored, first.train_mb, first.inference_mb) == (
        2,
        2,
        13000,
        11000,
    )
    assert first.span == pytest.approx(100.0)
    assert ranks[0].start_line == 20.0 and ranks[1].end_line == 1321.0


def test_first_puzzle_starts_at_the_printed_seconds_after_the_model_load():
    ranks = parsed()
    first = ranks[0].puzzles[0]
    assert first.start == pytest.approx(30.0)
    assert first.start_source == slt.START_FROM_PRINTED
    assert ranks[0].load_seconds == pytest.approx(10.0)
    assert ranks[1].load_seconds == pytest.approx(16.0)  # B starts at 110.5 - 69.5 = 41


def test_later_puzzles_start_at_the_previous_finished_line():
    capped = parsed()[1].puzzles[1]
    assert capped.puzzle_id == C
    assert capped.start == 110.5 and capped.start_source == slt.START_FROM_PREVIOUS
    assert capped.span == pytest.approx(1209.6)
    assert capped.printed_seconds == 1209.7


def test_timeout_line_closes_the_decode_interval():
    capped = parsed()[1].puzzles[1]
    assert capped.timeout_after == 1209.6
    assert capped.intervals == [("decode", 500.0), ("selection", 4.0), ("decode", 215.0)]


def test_a_puzzle_without_candidates_has_only_decode_intervals():
    empty = parsed()[1].puzzles[0]
    assert empty.puzzle_id == B and empty.scored == 0
    assert empty.intervals == [("decode", 20.0), ("decode", 12.5)]
    assert empty.span == pytest.approx(69.5)


def test_untagged_lines_do_not_close_intervals():
    first = parsed()[0].puzzles[0]
    assert ("selection", 1.0) not in first.intervals  # the stderr warning at 100 s


def test_train_stats_are_parsed():
    first = parsed()[0].puzzles[0]
    assert (first.global_step, first.training_loss, first.train_runtime) == (128, 0.25, 40.0)
    assert slt.flos_to_tokens(first.total_flos) == (1920, 0.0)


def test_an_unknown_rank_line_closes_the_interval_and_is_counted():
    ranks = parsed()
    cut = ranks[0].puzzles[1]
    assert ranks[0].unknown_lines == 1 and ranks[1].unknown_lines == 0
    assert cut.puzzle_id == D and cut.intervals == [("decode", 9.0)]
    assert not cut.complete and cut.span is None


def test_a_line_naming_another_task_is_a_conflict():
    lines, _, _ = slt.parse_session_log(
        json.dumps(
            [
                entry(1.0, stats(0, A, 1.0, 64)),
                entry(2.0, decoding(0, f"{B}_0.permute0123456789.ex0")),
                entry(3.0, f"[Rank 0] finished {A} in 3.0s"),
            ]
        )
    )
    puzzle = slt.parse_ranks(lines)[0].puzzles[0]
    assert puzzle.puzzle_id == A and puzzle.id_conflicts == 1


def test_nan_metrics_are_left_unknown():
    line = f"[Rank 0] training stats for puzzle {A}: TrainOutput(global_step=4, training_loss=nan, "
    line += "metrics={'train_runtime': nan, 'total_flos': 7.0e12, 'train_loss': nan})"
    lines, _, _ = slt.parse_session_log(json.dumps([entry(1.0, line)]))
    puzzle = slt.parse_ranks(lines)[0].puzzles[0]
    assert puzzle.train_runtime is None and puzzle.total_flos == 7.0e12
    assert slt.puzzle_records(puzzle) == []  # no train record without a runtime


# Records and CSV


def test_write_records_writes_one_file_per_rank(records_dir):
    result, out = records_dir
    assert [path.name for path in result.files] == ["timing-rank0.jsonl", "timing-rank1.jsonl"]
    assert (result.records, result.puzzles, result.incomplete, result.unnamed) == (19, 3, 1, 0)
    assert (result.skipped_entries, result.truncated, result.unknown_lines) == (4, True, 1)
    records, skipped = read_records(result.files)
    assert skipped == 0
    by_worker = {}
    for record in records:
        by_worker.setdefault(record.worker, []).append(
            (record.puzzle_id, record.op, record.seconds, record.tokens)
        )
    assert by_worker["rank0"] == [
        (None, "model_load", 10.0, None),
        (A, "train", 40.0, 1920),
        (A, "decode", 20.0, None),
        (A, "selection", 3.0, None),
        (A, "selection", 3.0, None),
        (A, "decode", 28.0, None),
        (A, "puzzle", 100.0, None),
        (D, "train", 45.0, 640),
        (D, "decode", 9.0, None),
    ]
    assert by_worker["rank1"] == [
        (None, "model_load", 16.0, None),
        (B, "train", 30.0, B_TOKENS),
        (B, "decode", 20.0, None),
        (B, "decode", 12.5, None),
        (B, "puzzle", 69.5, None),
        (C, "train", 480.0, 6400),
        (C, "decode", 500.0, None),
        (C, "selection", 4.0, None),
        (C, "decode", 215.0, None),
        (C, "puzzle", 1209.6, None),
    ]


def test_train_decode_selection_fit_inside_each_span(records_dir):
    result, _ = records_dir
    records, _ = read_records(result.files)
    for task in (A, B, C):
        ops = {}
        for record in records:
            if record.puzzle_id == task:
                ops[record.op] = ops.get(record.op, 0.0) + record.seconds
        phases = ops["train"] + ops.get("decode", 0.0) + ops.get("selection", 0.0)
        assert 0 < phases <= ops["puzzle"]


def test_puzzles_csv_holds_the_extras(records_dir):
    _, out = records_dir
    with open(out / "puzzles.csv", encoding="utf-8", newline="") as handle:
        rows = {row["puzzle_id"]: row for row in csv.DictReader(handle)}
    assert list(rows) == [A, D, B, C]
    assert rows[A]["start_source"] == slt.START_FROM_PRINTED
    assert rows[A]["first_line_s"] == "45.0" and rows[A]["train_tokens"] == "1920"
    assert rows[B]["scored_candidates"] == "0" and rows[B]["decode_batches"] == "2"
    assert rows[C]["timed_out"] == "1" and rows[C]["timeout_after_s"] == "1209.6"
    assert float(rows[C]["span_s"]) == pytest.approx(1209.6) and rows[C]["printed_s"] == "1209.7"
    assert rows[D]["complete"] == "0" and rows[D]["end_s"] == "" and rows[D]["start_s"] == "130.0"
    assert float(rows[D]["total_flos"]) == 6 * P * 640


def test_writing_again_replaces_the_files(records_dir):
    result, out = records_dir
    stale = out / "timing-rank9.jsonl"
    stale.write_text('{"puzzle_id": "x", "op": "train", "seconds": 1.0, "tokens": null}\n')
    again = slt.write_records(out.parent / "session.log", out)
    records, _ = read_records(again.files)
    assert len(records) == result.records == 19
    assert not stale.exists()


def test_records_command(tmp_path, capsys):
    log = tmp_path / "session.log"
    log.write_text(truncated_log(), encoding="utf-8")
    assert slt.main(["records", str(log), "--out", str(tmp_path / "timing")]) == 0
    assert "19 records in 2 files, 3 puzzles (1 incomplete" in capsys.readouterr().out


def test_records_command_without_worker_lines_fails(tmp_path):
    log = tmp_path / "session.log"
    log.write_text(json.dumps([entry(1.0, "no worker here")]), encoding="utf-8")
    assert slt.main(["records", str(log), "--out", str(tmp_path / "timing")]) == 1


# Summary

GRIDS = {"1": [[1, 1]], "2": [[2]], "3": [[3]], "4": [[4]], "5": [[5]], "6": [[6, 6]]}


def candidate(beam, aug, grid):
    return {"beam_score": beam, "score_aug": aug, "solution": GRIDS[grid]}


CANDIDATES = {
    f"{A}_0.permute0123456789.ex01": [
        candidate(0.1, [1.0, 1.0], "1"),
        candidate(0.5, [2.0, 2.0], "2"),
    ],
    f"{A}_0.rot90.permute0123.ex01": [candidate(0.2, [1.0, 1.0], "1")],
    f"{C}_0.permute0123456789.ex012": [
        candidate(0.1, [0.1], "3"),
        candidate(0.2, [0.2], "4"),
        candidate(0.3, [0.3], "5"),
    ],
    f"{C}_1.permute0123456789.ex012": [candidate(0.05, [0.05], "6")],
}
ZERO = [[0]]
SUBMISSION = {
    A: [{"attempt_1": GRIDS["1"], "attempt_2": GRIDS["2"]}],
    B: [{"attempt_1": ZERO, "attempt_2": ZERO}],
    C: [
        {"attempt_1": GRIDS["3"], "attempt_2": GRIDS["4"]},
        {"attempt_1": GRIDS["6"], "attempt_2": ZERO},
    ],
    D: [{"attempt_1": ZERO, "attempt_2": ZERO}],
}
CORRECT = [
    f"{A}_0.permute0123456789.ex01.out1",
    f"{C}_0.permute0123456789.ex012.out2",
    f"{C}_1.permute0123456789.ex012.out0",
]


def test_candidate_outputs_rank_and_credit():
    rows, problems = slt.candidate_outputs(CANDIDATES, SUBMISSION, CORRECT)
    assert problems == []
    by_output = {(r.task_id, r.index): r for r in rows}
    assert len(rows) == 5
    first = by_output[(A, 0)]
    assert (first.candidates, first.distinct, first.correct_rank) == (3, 2, 2)
    assert first.submitted_correct and first.top2_matches and first.probmul_correct
    assert by_output[(B, 0)].top2_matches is None and by_output[(B, 0)].correct_rank is None
    missed = by_output[(C, 0)]
    assert missed.correct_rank == 3 and not missed.submitted_correct and missed.top2_matches
    assert by_output[(C, 1)].correct_rank == 1 and by_output[(C, 1)].submitted_correct
    assert slt.subset_score(rows, lambda r: r.submitted_correct).score == pytest.approx(1.5)
    oracle = slt.subset_score(rows, lambda r: r.correct_rank is not None)
    assert oracle.score == pytest.approx(2.0) and oracle.tasks == 4


def test_candidate_outputs_report_unknown_and_conflicting_marks():
    marks = [*CORRECT, f"{A}_0.permute0123456789.ex01.out0", f"{A}_9.missing.ex0.out0"]
    _, problems = slt.candidate_outputs(CANDIDATES, SUBMISSION, marks)
    assert any("not in the candidates" in problem for problem in problems)
    assert any("different grids" in problem for problem in problems)


def test_describe_uses_the_nearest_rank_p90():
    stats = slt.describe(range(1, 11))
    assert (stats.n, stats.mean, stats.median, stats.p90, stats.max) == (10, 5.5, 5.5, 9, 10)
    with pytest.raises(ValueError):
        slt.describe([])


def write_summary_inputs(tmp_path):
    log = tmp_path / "session.log"
    log.write_text(json.dumps(worker_entries() + notebook_entries()), encoding="utf-8")
    timing = tmp_path / "timing"
    slt.write_records(log, timing)
    files = {
        "candidates.json": CANDIDATES,
        "submission.json": SUBMISSION,
        "env_probe.json": {
            "evaluation_stats": {
                "per_task": {
                    A: {"train_pairs": 2, "test_inputs": 1, "max_side": 3, "cells": 30},
                    B: {"train_pairs": 2, "test_inputs": 1, "max_side": 2, "cells": 20},
                    C: {"train_pairs": 3, "test_inputs": 2, "max_side": 5, "cells": 100},
                    D: {"train_pairs": 2, "test_inputs": 1, "max_side": 2, "cells": 10},
                }
            }
        },
        "stages.json": {
            "build": {"tag": "SYN-01", "commit": "0123abc"},
            "settings": {"hash_seed": 0, "subset": None},
            "stages": {"run": {"start_s": 1.0}},
        },
    }
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")
    return [
        "summary",
        "--timing",
        str(timing),
        "--log",
        str(log),
        "--candidates",
        str(tmp_path / "candidates.json"),
        "--submission",
        str(tmp_path / "submission.json"),
        "--task-stats",
        str(tmp_path / "env_probe.json"),
        "--stages",
        str(tmp_path / "stages.json"),
        "--out",
        str(tmp_path / "report" / "summary.md"),
    ]


def test_summary_command_reports_credit_timing_and_candidates(tmp_path):
    assert slt.main(write_summary_inputs(tmp_path)) == 0
    text = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert text.startswith("# SYN-01: ")
    assert "Code commit `0123abc`, `PYTHONHASHSEED=0`" in text
    assert "**1.50 of 4 tasks (37.50%)**" in text
    assert "`Reload score: 1.5`: equal" in text
    assert "attempts for 3 of 3 test outputs with candidates" in text
    assert "rank is 1 or 2 for 3 of 3" in text
    assert "Problems: none." in text
    assert "incomplete (no `finished` line; left out below): dddd0004" in text
    assert "0 differ by more than 2 s" in text
    assert "| cccc0003 | 1 |" in text  # the capped puzzle's row
    assert "$T$ equals the steps times this count for 1 of 1." in text  # B is square
    assert "| `a + b T + c T^2 / steps` | 3 |" in text and "steps per puzzle: 128" in text
    # Mean puzzle 459.7 s, mean startup and load 30 s, 2 workers, 41,399 s window.
    assert "| 0.667 | at or below |" in text and "| 240 | 120 |" in text
    assert "| 1.333 | above |" in text
    assert "Oracle (an output counts when any candidate is correct): 2.00 of 4 tasks" in text
    assert "solver exit code" not in text  # lines are summarized, not copied
    for grid in GRIDS.values():
        assert json.dumps(grid) not in text
    with open(tmp_path / "report" / "outputs.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["task_id"], r["output"], r["correct_rank_kgmon"]) for r in rows] == [
        (A, "0", "2"),
        (B, "0", ""),
        (C, "0", "3"),
        (C, "1", "1"),
        (D, "0", ""),
    ]
    assert [r["submitted_correct"] for r in rows] == ["1", "0", "0", "1", "0"]


def test_summary_worker_time_adds_up(tmp_path):
    write_summary_inputs(tmp_path)
    run = slt.load_run(
        tmp_path / "timing",
        tmp_path / "session.log",
        tmp_path / "env_probe.json",
        offset=1.0,
    )
    workers = slt.worker_rows(run)
    wall = run.exit - run.launch
    assert (run.launch, run.exit) == (5.5, 1322.0)
    for worker in workers:
        parts = worker.startup + worker.load + worker.busy + worker.unfinished + worker.idle
        assert parts == pytest.approx(wall)
    assert workers[0].unfinished == pytest.approx(1322.0 - 130.0) and workers[0].idle == 0
    assert run.window == 41399.0


def test_summary_command_fails_cleanly_on_a_missing_input(tmp_path, capsys):
    args = write_summary_inputs(tmp_path)
    (tmp_path / "submission.json").unlink()
    assert slt.main(args) == 1
    assert "error" in capsys.readouterr().err


def test_train_models_add_the_attention_term(tmp_path):
    write_summary_inputs(tmp_path)
    run = slt.load_run(tmp_path / "timing", tmp_path / "session.log", tmp_path / "env_probe.json")
    line, curve = slt.train_models(run)
    assert line.formula == "a + b T" and line.c is None and line.worst > 1.0
    # Three puzzles, three coefficients: the curve passes through every point.
    assert curve.n == 3 and curve.r2 == pytest.approx(1.0) and curve.worst < 1e-6
