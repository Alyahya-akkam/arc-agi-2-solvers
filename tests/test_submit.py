"""Acceptance tests for arc26/submit.py (task P0.5, deliverable 1; BRIEF A5).

The A5 cases (missing task, missing attempt, ragged grid, out-of-range value, a crash mid-run
still yields a valid file) plus the repair and atomic-write rules the notebook relies on. All
data is synthetic: stand-in challenges with the DEV4 ids and dummy grids, never competition data.
"""

import ast
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import arc26.submit as submit

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "arc26" / "submit.py"
# DEV4 with its test-input counts (K0 statistics): 5 test inputs, 10 attempts.
DEV4 = {"0934a4d8": 1, "36a08778": 2, "981571dc": 1, "aa4ec2a5": 1}
GRID = [[1, 2], [3, 4]]


def challenges(counts=DEV4):
    return {
        task: {"train": [{"input": GRID, "output": GRID}], "test": [{"input": GRID}] * count}
        for task, count in counts.items()
    }


def full(counts=DEV4, grid=GRID):
    return {
        task: [{"attempt_1": grid, "attempt_2": grid} for _ in range(count)]
        for task, count in counts.items()
    }


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


# validate


def test_a_complete_submission_is_valid():
    assert submit.validate(full(), challenges()) == []


def test_missing_task_is_reported():
    submission = full()
    del submission["36a08778"]
    assert submit.validate(submission, challenges()) == ["36a08778: missing"]


def test_missing_attempt_is_reported():
    submission = full()
    del submission["36a08778"][1]["attempt_2"]
    assert submit.validate(submission, challenges()) == ["36a08778[1]: missing attempt_2"]


def test_ragged_grid_is_reported():
    submission = full()
    submission["981571dc"][0]["attempt_1"] = [[1, 2], [3]]
    problems = submit.validate(submission, challenges())
    assert problems == ["981571dc[0].attempt_1: ragged: row 1 has 1 cells, row 0 has 2"]


@pytest.mark.parametrize("value", [10, -1])
def test_out_of_range_value_is_reported(value):
    submission = full()
    submission["aa4ec2a5"][0]["attempt_2"] = [[0, value]]
    problems = submit.validate(submission, challenges())
    assert len(problems) == 1
    assert problems[0].startswith("aa4ec2a5[0].attempt_2: value")
    assert problems[0].endswith("is outside 0 to 9")


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [[]],
        [[0] * 31],
        [[0]] * 31,
        [[True]],
        [[1.0]],
        [["1"]],
        [[None]],
        "[[0]]",
        None,
        [0, 1],
        [(0, 1)],
    ],
    ids=[
        "no-rows",
        "empty-row",
        "31-columns",
        "31-rows",
        "bool",
        "float",
        "string-cell",
        "null-cell",
        "string",
        "null",
        "flat",
        "tuple-row",
    ],
)
def test_invalid_grids_are_reported(grid):
    assert submit.grid_problem(grid) is not None
    submission = full()
    submission["0934a4d8"][0]["attempt_1"] = grid
    assert len(submit.validate(submission, challenges())) == 1


@pytest.mark.parametrize("cell", [True, False])
def test_bool_cells_are_rejected_although_bool_is_an_int(cell, tmp_path):
    # isinstance(True, int) holds in Python, and JSON would write the cell as true or false.
    assert isinstance(cell, int)
    assert submit.grid_problem([[1, cell]]) == (
        f"value {cell!r} at row 0, column 1 is not an integer"
    )
    submission = full()
    submission["aa4ec2a5"][0]["attempt_1"] = [[1, cell]]
    assert submit.validate(submission, challenges()) == [
        f"aa4ec2a5[0].attempt_1: value {cell!r} at row 0, column 1 is not an integer"
    ]
    assert submit.repair(submission, challenges())["aa4ec2a5"][0]["attempt_1"] == [[0]]
    raw = write(tmp_path / "raw.json", submission)
    out = tmp_path / "submission.json"
    report = submit.finalize_submission(out, write(tmp_path / "c.json", challenges()), raw)
    assert report["valid"] and report["replaced_attempts"] == 1
    assert "true" not in out.read_text() and "false" not in out.read_text()


def test_largest_and_smallest_grids_are_valid():
    for grid in ([[9] * 30] * 30, [[0]]):
        assert submit.grid_problem(grid) is None


def test_wrong_entry_counts_and_extra_keys_and_tasks_are_reported():
    submission = full()
    submission["36a08778"] = submission["36a08778"][:1]
    submission["981571dc"].append({"attempt_1": GRID, "attempt_2": GRID})
    submission["aa4ec2a5"][0]["attempt_3"] = GRID
    submission["deadbeef"] = []
    assert submit.validate(submission, challenges()) == [
        "36a08778: 1 test entries, the challenges have 2",
        "981571dc: 2 test entries, the challenges have 1",
        "aa4ec2a5[0]: unexpected keys ['attempt_3']",
        "'deadbeef': not a task of the challenges file",
    ]


@pytest.mark.parametrize("submission", [None, [], "x", 3])
def test_a_submission_that_is_not_an_object_is_one_problem(submission):
    assert submit.validate(submission, challenges()) == [
        f"submission is not a JSON object ({type(submission).__name__})"
    ]


@pytest.mark.parametrize(
    "bad", [[], {"t": {"train": []}}, {"t": {"test": []}}, {"t": []}, {1: {"test": [{}]}}]
)
def test_malformed_challenges_raise(bad):
    with pytest.raises(ValueError):
        submit.validate({}, bad)


# repair


def test_repair_keeps_every_valid_attempt_and_replaces_the_rest():
    submission = full()
    submission["36a08778"][1]["attempt_1"] = [[1, 2], [3]]
    del submission["981571dc"]
    submission["aa4ec2a5"][0]["attempt_2"] = [[10]]
    submission["deadbeef"] = full({"x": 1})["x"]
    repaired = submit.repair(submission, challenges())
    assert submit.validate(repaired, challenges()) == []
    assert list(repaired) == list(DEV4)
    assert repaired["0934a4d8"] == full()["0934a4d8"]
    assert repaired["36a08778"][0] == {"attempt_1": GRID, "attempt_2": GRID}
    assert repaired["36a08778"][1] == {"attempt_1": [[0]], "attempt_2": GRID}
    assert repaired["981571dc"] == [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    assert repaired["aa4ec2a5"] == [{"attempt_1": GRID, "attempt_2": [[0]]}]


@pytest.mark.parametrize("submission", [None, [], "x", {}])
def test_repair_of_nothing_is_all_zero(submission):
    repaired = submit.repair(submission, challenges())
    assert repaired == full(grid=[[0]])


def test_repair_copies_grids():
    submission = full()
    repaired = submit.repair(submission, challenges())
    repaired["0934a4d8"][0]["attempt_1"][0][0] = 7
    assert submission["0934a4d8"][0]["attempt_1"][0][0] == 1
    assert repaired["36a08778"][0]["attempt_1"] is not repaired["36a08778"][1]["attempt_1"]


def test_fallback_grids_are_fresh_lists():
    repaired = submit.repair({}, challenges())
    repaired["0934a4d8"][0]["attempt_1"][0][0] = 5
    assert repaired["0934a4d8"][0]["attempt_2"] == [[0]]
    assert submit.fallback_grid() == [[0]]


# finalize_submission and the atomic write


def test_finalize_without_raw_writes_all_zero(tmp_path):
    out = tmp_path / "working" / "submission.json"
    report = submit.finalize_submission(out, write(tmp_path / "c.json", challenges()))
    assert json.loads(out.read_text()) == full(grid=[[0]])
    assert report["valid"] and report["raw_status"] == "none" and report["raw"] is None
    assert (report["tasks"], report["test_inputs"]) == (4, 5)
    assert report["replaced_attempts"] == report["zero_attempts"] == 10
    assert report["problem_count"] == 0


def test_finalize_keeps_a_valid_raw_submission_unchanged(tmp_path):
    raw = write(tmp_path / "raw.json", full())
    out = tmp_path / "submission.json"
    report = submit.finalize_submission(out, write(tmp_path / "c.json", challenges()), raw)
    assert json.loads(out.read_text()) == full()
    assert report["raw_status"] == "read" and report["problems"] == []
    assert report["replaced_attempts"] == report["zero_attempts"] == 0


def test_finalize_repairs_and_reports_a_partial_raw_submission(tmp_path):
    submission = full()
    del submission["981571dc"]
    submission["36a08778"][0]["attempt_2"] = [[11]]
    raw = write(tmp_path / "raw.json", submission)
    out = tmp_path / "submission.json"
    report = submit.finalize_submission(out, write(tmp_path / "c.json", challenges()), raw)
    assert report["valid"] and report["problem_count"] == 2
    assert report["replaced_attempts"] == 3 and report["zero_attempts"] == 3
    assert json.loads(out.read_text())["36a08778"][1] == {"attempt_1": GRID, "attempt_2": GRID}


@pytest.mark.parametrize("content", [None, b"", b'{"0934a4d8": [{"attempt_1": [[1', b"\xff\xfe"])
def test_finalize_survives_a_missing_truncated_or_binary_raw_file(tmp_path, content):
    raw = tmp_path / "raw.json"
    if content is not None:
        raw.write_bytes(content)
    out = tmp_path / "submission.json"
    report = submit.finalize_submission(out, write(tmp_path / "c.json", challenges()), raw)
    assert report["raw_status"] == ("missing" if content is None else "unreadable")
    assert report["valid"] and report["zero_attempts"] == 10
    assert submit.validate(json.loads(out.read_text()), challenges()) == []


def test_finalize_caps_the_listed_problems_but_counts_them_all(tmp_path):
    counts = {f"{i:08x}": 1 for i in range(30)}
    out = tmp_path / "submission.json"
    raw = write(tmp_path / "raw.json", {})
    report = submit.finalize_submission(out, write(tmp_path / "c.json", challenges(counts)), raw)
    assert report["problem_count"] == 30 and len(report["problems"]) == submit.MAX_LISTED


def test_finalize_refuses_unreadable_challenges(tmp_path):
    with pytest.raises(OSError):
        submit.finalize_submission(tmp_path / "s.json", tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(ValueError):
        submit.finalize_submission(tmp_path / "s.json", bad)
    assert not (tmp_path / "s.json").exists()


def test_atomic_write_leaves_the_old_file_when_replace_fails(tmp_path, monkeypatch):
    out = write(tmp_path / "submission.json", {"old": True})

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(submit.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        submit.write_json_atomic(out, {"new": True})
    assert json.loads(out.read_text()) == {"old": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["submission.json"]


def test_atomic_write_leaves_the_old_file_when_serializing_fails(tmp_path):
    out = write(tmp_path / "submission.json", {"old": True})
    with pytest.raises(TypeError):
        submit.write_json_atomic(out, {"new": object()})
    assert json.loads(out.read_text()) == {"old": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["submission.json"]


# A crash mid-run still yields a valid file


def run_child(code, cwd, **kwargs):
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs,
    )


def test_exception_mid_run_still_yields_a_valid_file(tmp_path):
    """The notebook's pattern: placeholder first, the solver's writer in try, finalize in finally.
    The writer dies halfway through a plain json.dump, as the reference's writer would."""
    write(tmp_path / "c.json", challenges())
    code = """
        import json
        from arc26.submit import finalize_submission

        finalize_submission("working/submission.json", "c.json")
        try:
            with open("raw.json", "w") as handle:
                handle.write('{"0934a4d8": [{"attempt_1": [[1, 2], [3, 4]], "attempt_2": [[1')
                raise RuntimeError("solver crashed mid-write")
        finally:
            report = finalize_submission("working/submission.json", "c.json", "raw.json")
            print(json.dumps(report))
    """
    child = run_child(code, tmp_path)
    stdout, stderr = child.communicate(timeout=60)
    assert child.returncode != 0 and "solver crashed mid-write" in stderr
    report = json.loads(stdout)
    assert report["valid"] and report["raw_status"] == "unreadable"
    written = json.loads((tmp_path / "working" / "submission.json").read_text())
    assert submit.validate(written, challenges()) == []


def test_session_killed_mid_run_leaves_the_placeholder(tmp_path):
    """A killed process runs no finally: the placeholder written before the solver started is
    what the scorer finds, and it is valid."""
    write(tmp_path / "c.json", challenges())
    code = """
        import sys, time
        from arc26.submit import finalize_submission

        finalize_submission("working/submission.json", "c.json")
        print("ready", flush=True)
        while True:
            time.sleep(1)
    """
    child = run_child(code, tmp_path)
    try:
        assert child.stdout.readline().strip() == "ready"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
    assert child.returncode == -signal.SIGKILL
    written = json.loads((tmp_path / "working" / "submission.json").read_text())
    assert written == full(grid=[[0]])


# The notebook embeds this module verbatim


def test_module_imports_only_the_standard_library():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module != "__future__", ast.dump(node)
            imported.add(node.module.split(".")[0])
    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


def test_module_source_runs_standalone(tmp_path):
    namespace = {"__name__": "embedded"}
    exec(compile(MODULE.read_text(encoding="utf-8"), "arc26/submit.py", "exec"), namespace)
    out = tmp_path / "submission.json"
    report = namespace["finalize_submission"](out, write(tmp_path / "c.json", challenges()))
    assert report["valid"] and json.loads(out.read_text()) == full(grid=[[0]])
