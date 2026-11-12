"""Tests of the agent's task loader (arc26/agent/tasks.py, P6.2)."""

import json

import pytest

from arc26.agent.tasks import count_test_inputs, load_tasks, task_from_json, tasks_from_challenges
from arc26.agent.types import AgentTask, Pair

CHALLENGES = {
    "aaaa0001": {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
            {"input": [[2, 2, 0]], "output": [[0, 2, 2]]},
        ],
        "test": [{"input": [[3, 0, 0]]}, {"input": [[0], [4]]}],
    },
    "bbbb0002": {
        "train": [{"input": [[5]], "output": [[5, 5]]}],
        # A test output in a challenges file must never reach the agent.
        "test": [{"input": [[6]], "output": [[6, 6]]}],
    },
}


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_tasks_reads_kaggle_format_in_file_order(tmp_path):
    tasks = load_tasks(write(tmp_path, "arc-agi_evaluation_challenges.json", CHALLENGES))
    assert [t.task_id for t in tasks] == ["aaaa0001", "bbbb0002"]
    first = tasks[0]
    assert isinstance(first, AgentTask)
    assert first.train == (
        Pair([[1, 0], [0, 0]], [[0, 1], [0, 0]]),
        Pair([[2, 2, 0]], [[0, 2, 2]]),
    )
    assert first.test_inputs == ([[3, 0, 0]], [[0], [4]])


def test_test_outputs_are_never_loaded(tmp_path):
    tasks = load_tasks(write(tmp_path, "challenges.json", CHALLENGES))
    second = tasks[1]
    assert second.test_inputs == ([[6]],)
    assert not hasattr(second, "test_outputs")
    assert "[[6, 6]]" not in repr(second)


def test_ids_filter_keeps_file_order_and_rejects_unknown_ids(tmp_path):
    path = write(tmp_path, "challenges.json", CHALLENGES)
    assert [t.task_id for t in load_tasks(path, ids=["bbbb0002"])] == ["bbbb0002"]
    assert [t.task_id for t in load_tasks(path, ids={"bbbb0002", "aaaa0001"})] == [
        "aaaa0001",
        "bbbb0002",
    ]
    with pytest.raises(ValueError, match="cccc0003"):
        load_tasks(path, ids=["cccc0003"])


@pytest.mark.parametrize(
    "name", ["arc-agi_evaluation_solutions.json", "Solutions.json", "my_solution_file.json"]
)
def test_refuses_solutions_files(tmp_path, name):
    with pytest.raises(ValueError, match="never reads solutions"):
        load_tasks(write(tmp_path, name, CHALLENGES))


def test_grids_are_copied():
    source = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]}
    task = task_from_json("t", source)
    source["train"][0]["input"][0][0] = 9
    assert task.train[0].input == [[1]]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not a dict", "not an object"),
        ({"train": [], "test": [{"input": [[1]]}]}, "no list of demonstration pairs"),
        ({"train": [{"input": [[1]], "output": [[1]]}], "test": []}, "no list of test inputs"),
        ({"train": [{"input": [[1]]}], "test": [{"input": [[1]]}]}, r"train\[0\]\.output"),
        (
            {"train": [{"input": [[1, 2], [3]], "output": [[1]]}], "test": [{"input": [[1]]}]},
            "ragged",
        ),
        (
            {"train": [{"input": [[10]], "output": [[1]]}], "test": [{"input": [[1]]}]},
            "outside 0 to 9",
        ),
        (
            {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[True]]}]},
            "not an integer",
        ),
        (
            {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[0] * 31]}]},
            "31 columns",
        ),
        ({"train": [{"input": [[1]], "output": [[1]]}], "test": ["x"]}, r"test\[0\] is not"),
    ],
)
def test_malformed_tasks_raise_with_the_task_id(content, message):
    with pytest.raises(ValueError, match=message):
        task_from_json("t0", content)


def test_challenges_must_be_an_object():
    with pytest.raises(ValueError, match="JSON object"):
        tasks_from_challenges([CHALLENGES])  # type: ignore[arg-type]


def test_count_test_inputs():
    tasks = tasks_from_challenges(CHALLENGES)
    assert count_test_inputs(tasks) == {"aaaa0001": 2, "bbbb0002": 1}
