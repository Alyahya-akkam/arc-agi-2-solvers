"""The rank-256 tail's solver entry: the split solver in its challenges file's order (P4.7b).

``python -m arc26.tail --end-time T --config JSON`` takes ``arc26.solver``'s arguments and runs
its workers (``arc26.solver.local_worker``: one process per GPU, the same worker markers, model
load, TTT, decode, candidate files, and timing records) with one difference: the queue holds the
tasks of the mode's challenges file in the order the file lists them, where ``arc26.solver``
sorts them by id. The notebook's tail (task P4.7b, ``docs/tasks/P4.7b.md``) writes that file in
its own data folder with rank 16's confidence-0 tasks ordered by NVARC's ``puzzle`` seconds,
cheapest first, so the tasks the tail's deadline leaves unstarted are the most expensive ones.
The tasks are read as the solver reads them: ``arc-agi_test_challenges.json`` under
``cfg["data_dir"]`` in a competition rerun (``KAGGLE_IS_COMPETITION_RERUN`` set), else
``arc-agi_evaluation_challenges.json`` (``arc26/solver.py``'s entry block). Importing this module
imports neither torch nor the GPU stack; :func:`main` imports ``arc26.solver``, whose import is
torch's alone, as the solver's own entry does. ``arc26.runner.run_solver`` starts it through its
``module`` argument. See ``docs/runbooks/build_submission.md``, "The rank-256 tail".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

PROG = "python -m arc26.tail"
RERUN_VARIABLE = "KAGGLE_IS_COMPETITION_RERUN"
TEST_CHALLENGES = "arc-agi_test_challenges.json"
EVALUATION_CHALLENGES = "arc-agi_evaluation_challenges.json"


def challenges_path(data_dir: str, rerun: bool) -> str:
    """Return the challenges file the solver reads in this mode.

    Args:
        data_dir: The solver config's ``data_dir``.
        rerun: Whether this is a competition rerun.

    Returns:
        ``<data_dir>/arc-agi_test_challenges.json`` in a rerun, else the evaluation challenges.
    """
    return os.path.join(data_dir, TEST_CHALLENGES if rerun else EVALUATION_CHALLENGES)


def queue_order(data: Mapping[str, Any], cfg: Mapping[str, Any], rerun: bool) -> list[str]:
    """Return the task ids the workers take, in the challenges file's order.

    Why: the tail's file lists its tasks cheapest first; the solver's own entry sorts them by id.
    Outside a rerun a config ``subset`` filters them as the solver's entry does.

    Args:
        data: The parsed challenges file, in its order (``json.load`` keeps it).
        cfg: The solver config; ``subset`` is read.
        rerun: Whether this is a competition rerun (the subset is then ignored).

    Returns:
        The ids in queue order.
    """
    subset = cfg.get("subset")
    keys = list(data)
    if not rerun and subset is not None:
        keys = [key for key in keys if key in subset]
    return keys


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser: ``arc26.solver``'s two arguments.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n", 1)[0])
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--config", type=str, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Queue the challenges file's tasks in its order and spawn the split solver's workers.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        0 once every worker has returned (a worker's exception propagates, as in the solver).
    """
    args = build_parser().parse_args(argv)
    cfg = json.loads(args.config)
    rerun = bool(os.getenv(RERUN_VARIABLE))
    with open(challenges_path(cfg["data_dir"], rerun), encoding="utf-8") as handle:
        data = json.load(handle)
    # The GPU side, as the solver's entry: torch's spawn, and the worker it starts per GPU.
    import torch.multiprocessing as mp

    from arc26.solver import local_worker

    queue = mp.Manager().Queue()
    for key in queue_order(data, cfg, rerun):
        queue.put(key)
    for _ in range(cfg["workers"]):
        queue.put(None)
    mp.spawn(local_worker, args=(queue, args.end_time, cfg), nprocs=cfg["workers"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
