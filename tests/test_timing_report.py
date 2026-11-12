"""Per-puzzle table and linear cost model of scripts/timing_report.py on synthetic logs (P0.3a)."""

import importlib.util
import random
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from arc26.timing import (
    DECODE,
    MAIN_SELECTION,
    PUZZLE,
    SELECTION,
    TRAIN,
    TimingLogger,
    TimingRecord,
)

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "timing_report", REPO / "scripts" / "timing_report.py"
)
report = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = report  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(report)

DEV4 = ("0934a4d8", "36a08778", "981571dc", "aa4ec2a5")


def rec(puzzle_id, op, seconds, tokens=None, worker="0"):
    return TimingRecord(puzzle_id, op, seconds, tokens, worker)


def write_dev4_run(log_dir):
    """Four workers, one DEV4 puzzle each, with exact linear costs per op, and the main process."""
    TimingLogger(log_dir, worker="main").record(None, MAIN_SELECTION, 0.5)
    for rank, puzzle_id in enumerate(DEV4):
        logger = TimingLogger(log_dir, worker=rank)
        logger.record(None, "model_load", 150.0)
        train_tokens = 1000 * (rank + 1)
        logger.record(puzzle_id, TRAIN, 5.0 + 0.02 * train_tokens, train_tokens)
        for batch in range(2):
            decode_tokens = 500 * (rank + 1) + 100 * batch
            logger.record(puzzle_id, DECODE, 1.0 + 0.004 * decode_tokens, decode_tokens)
        logger.record(puzzle_id, SELECTION, 3.0, None)
        phases = 5.0 + 0.02 * train_tokens + 2.0 + 0.004 * (1000 * (rank + 1) + 100) + 3.0
        logger.record(puzzle_id, PUZZLE, phases + 4.0, None)


# Least squares


def test_fit_recovers_an_exact_line():
    xs = [100, 400, 900, 1600]
    a, b, r2 = report.fit_linear(xs, [2.0 + 0.01 * x for x in xs])
    assert a == pytest.approx(2.0) and b == pytest.approx(0.01) and r2 == pytest.approx(1.0)


def test_fit_matches_statistics_linear_regression_on_noisy_data():
    rng = random.Random(0)
    xs = [rng.randint(100, 9000) for _ in range(50)]
    ys = [3.0 + 0.002 * x + rng.gauss(0, 1) for x in xs]
    a, b, r2 = report.fit_linear(xs, ys)
    slope, intercept = statistics.linear_regression(xs, ys)
    assert b == pytest.approx(slope) and a == pytest.approx(intercept)
    assert r2 == pytest.approx(statistics.correlation(xs, ys) ** 2)
    assert 0 < r2 < 1


def test_fit_without_varying_tokens_reports_the_mean_only():
    assert report.fit_linear([7], [2.5]) == (2.5, None, None)
    assert report.fit_linear([5, 5, 5], [1.0, 2.0, 3.0]) == (2.0, None, None)


def test_fit_with_constant_seconds_has_zero_slope_and_no_r2():
    a, b, r2 = report.fit_linear([1, 2, 3], [4.0, 4.0, 4.0])
    assert (a, b, r2) == (4.0, 0.0, None)


@pytest.mark.parametrize(("xs", "ys"), [([], []), ([1, 2], [1.0])])
def test_fit_refuses_empty_or_unequal_inputs(xs, ys):
    with pytest.raises(ValueError):
        report.fit_linear(xs, ys)


def test_fit_ops_fits_each_op_over_records_with_tokens():
    records = [
        rec("a", TRAIN, 3.0, 100),
        rec("b", TRAIN, 5.0, 200),
        rec("c", TRAIN, 9.0, None),  # no token count: not a sample
        rec("a", SELECTION, 1.0),
        rec(None, "model_load", 150.0),
    ]
    fits = {fit.op: fit for fit in report.fit_ops(records)}
    assert list(fits) == ["model_load", "selection", "train"]
    train = fits[TRAIN]
    assert train.n == 2 and train.a == pytest.approx(1.0) and train.b == pytest.approx(0.02)
    assert fits[SELECTION] == report.Fit(SELECTION, 0, None, None, None)


# Per-puzzle table


def test_overhead_is_the_puzzle_span_minus_the_three_phases():
    records = [
        rec("p", TRAIN, 100.0, 5000),
        rec("p", DECODE, 20.0, 800),
        rec("p", DECODE, 30.0, 900),
        rec("p", SELECTION, 10.0),
        rec("p", "augment", 1.0),  # inside the span: part of the overhead
        rec("p", PUZZLE, 175.0),
    ]
    (row,) = report.per_puzzle(records)
    assert row.seconds == {TRAIN: 100.0, DECODE: 50.0, SELECTION: 10.0}
    assert row.tokens == {TRAIN: 5000, DECODE: 1700, SELECTION: None}
    assert row.overhead == pytest.approx(15.0) and row.total == 175.0


def test_main_selection_never_enters_a_puzzles_overhead():
    records = [
        rec("p", TRAIN, 10.0),
        rec("p", DECODE, 5.0),
        rec("p", SELECTION, 2.0),
        rec("p", PUZZLE, 20.0),
        rec(None, MAIN_SELECTION, 1.0, worker="main"),
        rec("p", MAIN_SELECTION, 1.0, worker="main"),  # even if logged with the puzzle's id
    ]
    (row,) = report.per_puzzle(records)
    assert row.overhead == pytest.approx(3.0) and row.total == 20.0


def test_without_a_puzzle_span_overhead_is_the_other_ops():
    records = [rec("p", TRAIN, 10.0), rec("p", "augment", 2.0), rec("p", "reset", 0.5)]
    (row,) = report.per_puzzle(records)
    assert row.overhead == pytest.approx(2.5) and row.total == pytest.approx(12.5)


def test_rows_are_sorted_and_name_their_workers():
    records = [
        rec("b", TRAIN, 1.0, worker="1"),
        rec("a", TRAIN, 1.0, worker="0"),
        rec("b", DECODE, 1.0, worker="2"),  # moved to another worker
        rec(None, "model_load", 9.0, worker="0"),
    ]
    rows = report.per_puzzle(records)
    assert [(row.puzzle_id, row.workers) for row in rows] == [("a", "0"), ("b", "1,2")]


def test_ops_outside_puzzles_summarize_records_without_a_puzzle():
    records = [
        rec(None, "model_load", 150.0),
        rec(None, "model_load", 160.0),
        rec(None, MAIN_SELECTION, 0.5, worker="main"),
        rec("p", TRAIN, 1.0),
    ]
    assert report.ops_outside_puzzles(records) == [
        ("main_selection", 1, 0.5),
        ("model_load", 2, 310.0),
    ]


# Rendering and CLI


def test_dev4_run_yields_the_table_and_the_cost_model(tmp_path):
    write_dev4_run(tmp_path / "run_logs")
    out = tmp_path / "reports" / "timing.md"
    assert report.main([str(tmp_path / "run_logs"), "--out", str(out), "--title", "DEV4"]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# DEV4\n\nRecords: 25; skipped lines: 0; files: ")
    table = [line for line in text.splitlines() if line.startswith("| 0934a4d8 ")]
    # Train 5 + 0.02 * 1000; decode 2 + 0.004 * 1100; selection 3; overhead 4.
    assert table == ["| 0934a4d8 | 0 | 25.00 | 6.40 | 3.00 | 4.00 | 38.40 | 1000 | 1100 | n/a |"]
    # Decode tokens per puzzle 1000 (rank + 1) + 100: 1100, 2100, 3100, 4100.
    assert "| mean of 4 |  | 55.00 | 12.40 | 3.00 | 4.00 | 74.40 | 2500 | 2600 | n/a |" in text
    assert "| sum |  | 220.00 | 49.60 | 12.00 | 16.00 | 297.60 | 10000 | 10400 | n/a |" in text
    assert "## Ops outside puzzles" in text
    assert "| main_selection | 1 | 0.50 | 0.50 |" in text
    assert "| model_load | 4 | 600.00 | 150.00 |" in text
    assert "| train | 4 | 5.0000 | 2.0000e-02 | 1.0000 |" in text
    assert "| decode | 8 | 1.0000 | 4.0000e-03 | 1.0000 |" in text
    assert "| selection | 0 | n/a | n/a | n/a |" in text


def test_truncated_lines_are_skipped_and_counted(tmp_path, capsys):
    write_dev4_run(tmp_path)
    with open(tmp_path / "timing-3.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"puzzle_id": "aa4ec2a5", "op": "tra')
    assert report.main([str(tmp_path)]) == 0
    assert "Records: 25; skipped lines: 1; files: " in capsys.readouterr().out


def test_directories_are_searched_recursively_and_files_taken_once(tmp_path):
    nested = tmp_path / "run" / "run_logs" / "timing"
    TimingLogger(nested, worker=0).record("p", TRAIN, 1.0, 10)
    (nested / "notes.jsonl").write_text("{}\n", encoding="utf-8")
    files = report.timing_files([tmp_path / "run", nested / "timing-0.jsonl"])
    assert files == [nested / "timing-0.jsonl"]


def test_errors_without_records_or_with_a_missing_path(tmp_path, capsys):
    (tmp_path / "timing-0.jsonl").write_text("garbage\n", encoding="utf-8")
    assert report.main([str(tmp_path)]) == 1
    assert "no valid timing records (files: 1, skipped lines: 1)" in capsys.readouterr().err
    assert report.main([str(tmp_path / "absent")]) == 1
    assert "no such file or directory" in capsys.readouterr().err


def test_puzzle_free_logs_still_render(tmp_path):
    records = [rec(None, "model_load", 150.0, 0)]
    text = report.render(records, 0, [tmp_path / "timing-0.jsonl"], "Load only")
    assert "No puzzle records." in text and "| model_load | 1 | 150.0000 | n/a | n/a |" in text


def test_script_runs_as_a_program(tmp_path):
    write_dev4_run(tmp_path)
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "timing_report.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "## Cost model: seconds = a + b * tokens" in out
    assert "| 0934a4d8 | 0 | 25.00 |" in out
