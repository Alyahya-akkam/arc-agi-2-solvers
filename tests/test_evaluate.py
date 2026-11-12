"""Competition metric, agreement, and CLI of arc26.evaluate on synthetic predictions (P0.2)."""

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from arc26 import evaluate
from arc26.data import DEV4, DEV120
from arc26.evaluate import agreement, is_grid, score_submission, score_task

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "p0_1_reference_dev4.json"

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[0]]


def entry(first, second):
    return {"attempt_1": first, "attempt_2": second}


# is_grid


@pytest.mark.parametrize(
    "grid",
    [[[0]], [[9] * 30] * 30, [[1, 2, 3]], [[1], [2], [3]]],
)
def test_is_grid_accepts_rectangular_color_grids(grid):
    assert is_grid(grid)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [[]],
        [[1, 2], [3]],  # ragged
        [[1], 2],
        [[True]],  # bool is not a color
        [[1.0]],
        [["1"]],
        [[10]],
        [[-1]],
        [[0] * 31],
        [[0]] * 31,
        {"0": [0]},
        "[[0]]",
        [[None]],
    ],
)
def test_is_grid_refuses_malformed_values(value):
    assert not is_grid(value)


# Scoring


def test_either_attempt_scores_and_neither_does_not():
    solutions = {"t1": [A], "t2": [A], "t3": [A], "t4": [A]}
    predictions = {
        "t1": [entry(A, B)],
        "t2": [entry(B, A)],
        "t3": [entry(B, C)],
        "t4": [entry(A, A)],
    }
    result = score_submission(predictions, solutions)
    assert [t.correct for t in result.per_task] == [(True,), (True,), (False,), (True,)]
    assert result.score == 3 and result.tasks == 4 and result.percent == 75.0


def test_partial_credit_is_the_mean_over_a_tasks_test_outputs():
    solutions = {"two": [A, B], "three": [A, B, C], "one": [C]}
    predictions = {
        "two": [entry(A, C), entry(C, C)],  # 1 of 2
        "three": [entry(C, A), entry(B, B), entry(A, B)],  # 2 of 3
        "one": [entry(B, A)],  # 0 of 1
    }
    result = score_submission(predictions, solutions, label="toy")
    by_task = {t.task_id: t for t in result.per_task}
    assert by_task["two"].score == 0.5
    assert by_task["three"].score == pytest.approx(2 / 3)
    assert by_task["one"].score == 0.0
    assert result.score == pytest.approx(0.5 + 2 / 3)
    assert result.test_outputs == 6 and result.solved_outputs == 3
    assert result.percent == pytest.approx(100 * (0.5 + 2 / 3) / 3)
    assert result.output_percent == 50.0


def test_per_task_and_per_output_variants_differ_as_defined():
    # Task "multi" solves both of its outputs, task "single" misses its only one.
    solutions = {"multi": [A, B], "single": [C]}
    predictions = {"multi": [entry(A, C), entry(C, B)], "single": [entry(A, B)]}
    result = score_submission(predictions, solutions)
    assert (result.score, result.tasks, result.percent) == (1.0, 2, 50.0)
    assert (result.solved_outputs, result.test_outputs) == (2, 3)
    assert result.output_percent == pytest.approx(200 / 3)


def test_missing_tasks_score_zero_and_are_listed():
    solutions = {"here": [A], "gone": [A, B]}
    result = score_submission({"here": [entry(A, A)]}, solutions)
    gone = next(t for t in result.per_task if t.task_id == "gone")
    assert gone.missing and gone.correct == (False, False) and gone.malformed == 0
    assert result.missing_tasks == ("gone",)
    assert result.score == 1.0 and result.tasks == 2 and result.test_outputs == 3


def test_empty_submission_scores_zero_on_every_task():
    result = score_submission({}, {"t1": [A], "t2": [B]})
    assert result.score == 0 and result.missing_tasks == ("t1", "t2")


@pytest.mark.parametrize(
    ("task_value", "malformed", "solved"),
    [
        ("not a list", 2, False),
        (None, 2, False),
        ([], 2, False),
        (["not a dict"], 2, False),
        ([{"attempt_1": A}], 1, True),  # attempt_2 absent
        ([{"attempt_2": A}], 1, True),
        ([{"attempt_1": B}], 1, False),
        ([entry([[1, 2], [3]], A)], 1, True),  # ragged first attempt, right second one
        ([entry([[True, 2], [3, 4]], B)], 1, False),
        ([entry([[1.0, 2.0], [3.0, 4.0]], A)], 1, True),
        ([entry({"rows": A}, [[1, 2]])], 1, False),
    ],
)
def test_malformed_predictions_never_raise_and_count(task_value, malformed, solved):
    result = score_task("t", {"t": task_value}, [A])
    assert result.malformed == malformed and not result.missing
    assert result.correct == (solved,)


def test_malformed_attempt_equal_to_solution_values_does_not_score():
    # [[1.0, 2.0], [3.0, 4.0]] == A in Python, but float cells are not a well-formed grid.
    floats = [[1.0, 2.0], [3.0, 4.0]]
    assert floats == A
    assert score_task("t", {"t": [entry(floats, floats)]}, [A]).correct == (False,)


def test_fewer_entries_than_test_outputs_score_the_present_ones():
    result = score_task("t", {"t": [entry(A, A)]}, [A, B])
    assert result.correct == (True, False) and result.malformed == 2


def test_extra_entries_and_attempts_are_ignored():
    predictions = {"t": [{"attempt_1": C, "attempt_2": C, "attempt_3": A}, entry(A, A)]}
    assert score_task("t", predictions, [A]).correct == (False,)


def test_subset_selects_tasks_and_deduplicates():
    solutions = {"t1": [A], "t2": [B], "t3": [C]}
    predictions = {"t1": [entry(A, A)], "t2": [entry(A, A)], "t3": [entry(C, C)]}
    result = score_submission(predictions, solutions, ["t3", "t1", "t3"], label="pick")
    assert [t.task_id for t in result.per_task] == ["t1", "t3"]
    assert result.score == 2 and result.label == "pick"


def test_scoring_requires_solutions_for_every_selected_task():
    with pytest.raises(ValueError, match="missing from the solutions: t9"):
        score_submission({}, {"t1": [A]}, ["t1", "t9"])


@pytest.mark.parametrize("bad", [[], [[[1], [2, 3]]], "grid", [A, None]])
def test_scoring_refuses_malformed_solutions(bad):
    with pytest.raises(ValueError, match="solutions must be a non-empty list of grids"):
        score_submission({}, {"t1": bad})


def reference_validate_submission(solutions, submission):
    """The reference notebook's ArcDataset.validate_submission, with list equality."""
    score = 0
    for k, v in solutions.items():
        for i, r in enumerate(v):
            for attempt in ["attempt_1", "attempt_2"]:
                if r == submission[k][i][attempt]:
                    score += 1 / len(v)
                    break
    return score


def random_grid(rng):
    height, width = rng.randint(1, 2), rng.randint(1, 2)
    return [[rng.randrange(2) for _ in range(width)] for _ in range(height)]


def test_metric_matches_the_reference_notebook_on_random_submissions():
    rng = random.Random(0)
    for _ in range(200):
        solutions, submission = {}, {}
        for t in range(rng.randint(1, 6)):
            outputs = [random_grid(rng) for _ in range(rng.randint(1, 3))]
            solutions[f"t{t}"] = outputs
            submission[f"t{t}"] = [
                entry(*(r if rng.random() < 0.3 else random_grid(rng) for _ in range(2)))
                for r in outputs
            ]
        expected = reference_validate_submission(solutions, submission)
        assert score_submission(submission, solutions).score == pytest.approx(expected)


def test_report_names_subset_and_counts():
    solutions = {"t1": [A, B], "t2": [C]}
    result = score_submission({"t1": [entry(A, C), entry(C, C)]}, solutions, label="DEV4")
    lines = result.report(per_task=True)
    assert lines[0] == "score DEV4: 2 tasks, 3 test outputs"
    assert lines[1] == "task score: 0.5 of 2 tasks (25.0%)"
    assert lines[2] == "output score: 1 of 3 test outputs (33.33%)"
    assert lines[3] == "missing tasks: 1 (t2); malformed attempts: 0"
    assert lines[4:] == ["  t1 0.5 (1/2)", "  t2 0.0 (0/1) missing"]


# Agreement


def fixture_predictions():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["predictions"]


def run_b_shape(reference):
    """Run B against run A in P0.1-01: attempt_1 agrees on all 5 outputs, attempt_2 on 2."""
    run_b = copy.deepcopy(reference)
    for task_id, index in (("0934a4d8", 0), ("36a08778", 0), ("36a08778", 1)):
        run_b[task_id][index]["attempt_2"] = [[0]]
    # The real submission also holds every other evaluation task, as [[0]] placeholders.
    for task_id in DEV120:
        run_b.setdefault(task_id, [entry([[0]], [[0]])])
    return run_b


def test_agreement_with_itself_is_one():
    reference = fixture_predictions()
    result = agreement(reference, reference)
    assert (result.g, result.attempt_1) == (1.0, 1.0)
    assert (result.tasks, result.test_outputs) == (4, 5)


def test_agreement_on_the_run_b_versus_run_a_shape():
    reference = fixture_predictions()
    result = agreement(run_b_shape(reference), reference, DEV4, label="DEV4")
    assert result.g == 0.4 and result.attempt_1 == 1.0
    assert (result.tasks, result.test_outputs) == (4, 5)
    assert [(o.task_id, o.index, o.both) for o in result.outputs] == [
        ("0934a4d8", 0, False),
        ("36a08778", 0, False),
        ("36a08778", 1, False),
        ("981571dc", 0, True),
        ("aa4ec2a5", 0, True),
    ]
    # Without a subset the reference's own tasks are compared: the same 5 outputs.
    assert agreement(run_b_shape(reference), reference).g == 0.4


def test_agreement_requires_both_attempts_in_order():
    reference = {"t": [entry(A, B)]}
    swapped = agreement({"t": [entry(B, A)]}, reference)
    assert (swapped.g, swapped.attempt_1) == (0.0, 0.0)
    second_only = agreement({"t": [entry(C, B)]}, reference)
    assert (second_only.g, second_only.attempt_1) == (0.0, 0.0)
    first_only = agreement({"t": [entry(A, C)]}, reference)
    assert (first_only.g, first_only.attempt_1) == (0.0, 1.0)


def test_agreement_counts_missing_predictions_as_disagreeing():
    reference = {"t1": [entry(A, B)], "t2": [entry(A, B), entry(B, A)]}
    result = agreement({"t1": [entry(A, B)]}, reference)
    assert result.test_outputs == 3 and result.g == pytest.approx(1 / 3)
    assert result.missing_in_predictions == ("t2",)


def test_agreement_skips_tasks_absent_from_the_reference():
    reference = {"t1": [entry(A, B)], "t2": "broken"}
    result = agreement({"t1": [entry(A, B)]}, reference, ["t1", "t2", "t3"], label="pick")
    assert (result.tasks, result.test_outputs, result.g) == (1, 1, 1.0)
    assert result.missing_in_reference == ("t2", "t3")


def test_malformed_attempts_never_agree_even_when_identical():
    ragged = [[1, 2], [3]]
    reference = {"t": [entry(ragged, A)], "u": [{"attempt_1": A}]}
    result = agreement(copy.deepcopy(reference), reference)
    assert result.g == 0.0 and result.attempt_1 == 0.5


def test_agreement_without_outputs_to_compare_raises():
    with pytest.raises(ValueError, match="no reference test outputs"):
        agreement({"t": [entry(A, B)]}, {"t": [entry(A, B)]}, ["x"], label="x")


def test_agreement_report_prints_g_and_attempt_1():
    reference = fixture_predictions()
    lines = agreement(run_b_shape(reference), reference, DEV4, label="DEV4").report()
    assert lines == [
        "agree DEV4: 4 tasks, 5 test outputs",
        "g = 0.4 (2 of 5 test outputs, both attempts identical in order)",
        "attempt_1 agreement = 1.0 (5 of 5 test outputs)",
        "missing in submission: 0; missing in reference: 0",
    ]


# Files and CLI


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_predictions_reads_the_fixture_wrapper_and_plain_files(tmp_path):
    assert evaluate.load_predictions(FIXTURE) == fixture_predictions()
    plain = write(tmp_path / "submission.json", {"t": [entry(A, B)]})
    assert evaluate.load_predictions(plain) == {"t": [entry(A, B)]}


@pytest.mark.parametrize("data", [[1, 2], "x", {"predictions": [1]}])
def test_load_predictions_refuses_non_objects(tmp_path, data):
    with pytest.raises(ValueError):
        evaluate.load_predictions(write(tmp_path / "bad.json", data))


def test_cli_agree_prints_the_p0_1_numbers(tmp_path, capsys):
    submission = write(tmp_path / "b.json", run_b_shape(fixture_predictions()))
    code = evaluate.main(["agree", str(submission), str(FIXTURE), "--subset", "DEV4"])
    out = capsys.readouterr().out
    assert code == 0
    assert "g = 0.4 " in out and "attempt_1 agreement = 1.0 " in out
    assert out.startswith("agree DEV4: 4 tasks, 5 test outputs\n")


def test_cli_score_with_subset_and_per_task(tmp_path, capsys):
    solutions = {task_id: [A] * (2 if task_id == "36a08778" else 1) for task_id in DEV120}
    predictions = {"0934a4d8": [entry(A, B)], "36a08778": [entry(B, A), entry(C, C)]}
    sub = write(tmp_path / "submission.json", predictions)
    sol = write(tmp_path / "solutions.json", solutions)
    code = evaluate.main(["score", str(sub), str(sol), "--subset", "DEV4", "--per-task"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert out[:4] == [
        "score DEV4: 4 tasks, 5 test outputs",
        "task score: 1.5 of 4 tasks (37.5%)",
        "output score: 2 of 5 test outputs (40.0%)",
        "missing tasks: 2 (981571dc, aa4ec2a5); malformed attempts: 0",
    ]
    assert out[4:] == [
        "  0934a4d8 1.0 (1/1)",
        "  36a08778 0.5 (1/2)",
        "  981571dc 0.0 (0/1) missing",
        "  aa4ec2a5 0.0 (0/1) missing",
    ]


def test_cli_score_without_subset_scores_every_solution_task(tmp_path, capsys):
    sub = write(tmp_path / "submission.json", {"t1": [entry(A, A)]})
    sol = write(tmp_path / "solutions.json", {"t1": [A], "t2": [B]})
    assert evaluate.main(["score", str(sub), str(sol)]) == 0
    assert capsys.readouterr().out.startswith("score all: 2 tasks, 2 test outputs\n")


def test_cli_reports_input_errors_with_status_2(tmp_path, capsys):
    sub = write(tmp_path / "submission.json", {})
    assert evaluate.main(["score", str(sub), str(tmp_path / "missing.json")]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert evaluate.main(["agree", str(sub), str(tmp_path / "broken.json")]) == 2
    sol = write(tmp_path / "solutions.json", {"t1": [A]})
    assert evaluate.main(["score", str(sub), str(sol), "--subset", "DEV4"]) == 2
    assert "missing from the solutions: 0934a4d8, 36a08778" in capsys.readouterr().err


def test_cli_refuses_unknown_subsets(tmp_path):
    sub = write(tmp_path / "submission.json", {})
    with pytest.raises(SystemExit):
        evaluate.main(["agree", str(sub), str(sub), "--subset", "DEV5"])


def test_module_entry_point_runs(tmp_path):
    submission = write(tmp_path / "b.json", run_b_shape(fixture_predictions()))
    out = subprocess.run(
        [sys.executable, "-m", "arc26.evaluate", "agree", str(submission), str(FIXTURE)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    ).stdout
    assert "g = 0.4 " in out and "attempt_1 agreement = 1.0 " in out
