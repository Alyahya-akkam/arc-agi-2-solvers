"""Structure and provenance of the P0.1 parity fixture (run A of pre-run brief P0.1-01)."""

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "p0_1_reference_dev4.json"
DEV4 = {"0934a4d8": 1, "36a08778": 2, "981571dc": 1, "aa4ec2a5": 1}


def load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def is_grid(grid):
    return (
        isinstance(grid, list)
        and 1 <= len(grid) <= 30
        and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
        and len({len(row) for row in grid}) == 1
        and all(isinstance(v, int) and 0 <= v <= 9 for row in grid for v in row)
    )


def test_fixture_holds_dev4_with_two_attempts_per_test_output():
    predictions = load()["predictions"]
    assert {task: len(outputs) for task, outputs in predictions.items()} == DEV4
    for outputs in predictions.values():
        for attempts in outputs:
            assert set(attempts) == {"attempt_1", "attempt_2"}
            assert is_grid(attempts["attempt_1"]) and is_grid(attempts["attempt_2"])


def test_fixture_records_its_provenance_and_noise_floor():
    fixture = load()
    source = fixture["source"]
    assert source["brief"].startswith("docs/prerun/P0.1-01.md")
    assert source["run"] == "A" and source["pythonhashseed"] == 0
    assert len(source["code_commit"]) == 40 and len(source["built_notebook_sha256"]) == 64
    assert fixture["dev4_test_outputs"] == sum(DEV4.values())
    assert fixture["noise_floor_g_BA"] == 0.4 and fixture["dev4_score"] == 2.5
