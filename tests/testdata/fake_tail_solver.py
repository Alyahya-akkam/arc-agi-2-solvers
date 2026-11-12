"""A stand-in for the rank-256 tail's solver in the notebook's CPU simulation (task P4.7b).

``tests/test_p4_7b.py`` runs the generated tail build on the simulated Kaggle tree of
``tests/test_hybrid_notebook.py``; its ``python`` shim runs this file instead of ``python -m
arc26.tail`` when ``ARC26_FAKE_TAIL`` names it. It takes the solver's arguments (``--end-time``
and the ``--config`` JSON), reads the mode's challenges file in the config's ``data_dir`` in
the file's order (as ``arc26.tail`` queues it), and records in ``ARC26_FAKE_TAIL_LOG`` its
arguments, that order, the files of the data folder, and whether the pid in
``ARC26_FAKE_TAIL_WATCH`` (a file holding one) is still alive. ``ARC26_FAKE_TAIL_MODE`` sets what
it does then: ``ok`` (the default) writes one candidate pickle per test output of every task,
the stand-in solution, and one ``puzzle`` timing record per task into ``timing_dir``, then exits
0; ``crash`` exits 1 having written nothing; ``hang`` sleeps until it is killed; ``corrupt``
writes the candidates and a cut pickle, then exits 0.
"""

import bz2
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SOLUTION = [[1, 2], [3, 4]]


def alive(pid_file):
    try:
        pid = int(Path(pid_file).read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def main():
    args = sys.argv[1:]
    end_time = float(args[args.index("--end-time") + 1])
    config = json.loads(args[args.index("--config") + 1])
    competition = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
    name = "arc-agi_test_challenges.json" if competition else "arc-agi_evaluation_challenges.json"
    data_dir = Path(config["data_dir"])
    tasks = json.loads((data_dir / name).read_text(encoding="utf-8"))
    watch = os.environ.get("ARC26_FAKE_TAIL_WATCH")
    record = {
        "argv": args,
        "end_time": end_time,
        "config": config,
        "order": list(tasks),
        "data_files": sorted(p.name for p in data_dir.iterdir()),
        "engine_alive": alive(watch) if watch else None,
        "started": time.time(),
    }
    Path(os.environ["ARC26_FAKE_TAIL_LOG"]).write_text(json.dumps(record), encoding="utf-8")
    mode = os.environ.get("ARC26_FAKE_TAIL_MODE", "ok")
    if mode == "crash":
        print("fake tail: crashed", flush=True)
        return 1
    if mode == "hang":
        print("fake tail: hanging", flush=True)
        while True:
            time.sleep(1)
    outputs = Path(config["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    for task, value in tasks.items():
        for index in range(len(value["test"])):
            row = {"beam_score": 0.1, "score_aug": [0.2] * 8, "solution": np.array(SOLUTION)}
            with bz2.BZ2File(outputs / f"{task}_{index}.permute0123456789", "w") as handle:
                pickle.dump([row], handle)
    if mode == "corrupt":
        (outputs / f"{next(iter(tasks))}_0.rot90").write_bytes(b"BZh91AY&SY cut")
    timing = Path(config["timing_dir"])
    timing.mkdir(parents=True, exist_ok=True)
    with open(timing / "timing-rank0.jsonl", "a", encoding="utf-8") as handle:
        for task in tasks:
            line = {"puzzle_id": task, "op": "puzzle", "seconds": 1.5, "tokens": None}
            handle.write(json.dumps(line) + "\n")
    print(f"fake tail: candidates for {list(tasks)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
