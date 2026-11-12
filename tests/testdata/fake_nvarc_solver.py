"""A stand-in for NVARC's solver in the hybrid notebook's CPU simulation (task P6.5).

``tests/test_hybrid_notebook.py`` runs the generated notebook on a simulated Kaggle tree; its
``python`` shim runs this file instead of ``python -m arc26.solver`` when ``ARC26_FAKE_SOLVER``
names it. It takes the solver's arguments (``arc26.runner.solver_command``: ``--end-time`` and
the ``--config`` JSON), writes one candidate pickle per test output of the first
``ARC26_FAKE_SOLVER_TASKS`` tasks (default 2) of the mode's challenges file, as the workers
write them (bz2, a list of ``{beam_score, score_aug, solution}`` with the stand-in solution as a
numpy array, or the JSON grid ``ARC26_FAKE_SOLVER_GRID`` when set: a wrong answer), so the
notebook's real selection turns them into NVARC's attempts, records its arguments in
``ARC26_FAKE_SOLVER_LOG``, and exits 0. Task P4.7b adds two optional settings:
``ARC26_FAKE_SOLVER_SECONDS`` (a JSON object, task id to seconds) writes one ``puzzle`` timing
record per listed task into the config's ``timing_dir`` when it is set, as the workers do; and
``ARC26_FAKE_SOLVER_SPLIT`` (comma-separated task ids) gives each listed task three candidate
files with three different grids, so NVARC's confidence there is 1/3, inside the shot's bucket
(0, 0.5).
"""

import bz2
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

SOLUTION = [[1, 2], [3, 4]]


def main():
    args = sys.argv[1:]
    end_time = float(args[args.index("--end-time") + 1])
    config = json.loads(args[args.index("--config") + 1])
    competition = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
    name = "arc-agi_test_challenges.json" if competition else "arc-agi_evaluation_challenges.json"
    tasks = json.loads((Path(config["data_dir"]) / name).read_text(encoding="utf-8"))
    ids = sorted(tasks)[: int(os.environ.get("ARC26_FAKE_SOLVER_TASKS", "2"))]
    grid = json.loads(os.environ.get("ARC26_FAKE_SOLVER_GRID") or json.dumps(SOLUTION))
    outputs = Path(config["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    split = set(filter(None, os.environ.get("ARC26_FAKE_SOLVER_SPLIT", "").split(",")))
    others = ([[4, 3], [2, 1]], [[0, 1], [1, 0]])
    for task in ids:
        for index in range(len(tasks[task]["test"])):
            grids = [grid, *others] if task in split else [grid]
            for view, answer in enumerate(grids):
                row = {"beam_score": 0.1, "score_aug": [0.2] * 8, "solution": np.array(answer)}
                name = f"{task}_{index}.permute0123456789" + ("" if view == 0 else f".v{view}")
                with bz2.BZ2File(outputs / name, "w") as handle:
                    pickle.dump([row], handle)
    seconds = json.loads(os.environ.get("ARC26_FAKE_SOLVER_SECONDS") or "{}")
    if seconds and config.get("timing_dir"):
        timing = Path(config["timing_dir"])
        timing.mkdir(parents=True, exist_ok=True)
        with open(timing / "timing-rank0.jsonl", "a", encoding="utf-8") as handle:
            for task, value in seconds.items():
                line = {"puzzle_id": task, "op": "puzzle", "seconds": value, "tokens": None}
                handle.write(json.dumps(line) + "\n")
    record = {"end_time": end_time, "config": config, "tasks": ids}
    Path(os.environ["ARC26_FAKE_SOLVER_LOG"]).write_text(json.dumps(record), encoding="utf-8")
    print(f"fake NVARC: candidates for {ids}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
