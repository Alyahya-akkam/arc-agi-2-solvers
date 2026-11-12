# Module: `arc26/tail.py`

Task P4.7b (`docs/tasks/P4.7b.md`); ADR-023 Amendment 4. Tests: `tests/test_p4_7b.py`. Importing
it imports neither torch nor the GPU stack; `main` imports `arc26.solver` (torch) and its workers
import the rest, as `arc26.solver`'s own entry does.

## Purpose

The rank-256 tail's solver entry. A tail build's notebook (`agent.tail`, the builder's
`TAIL_CODE`) runs NVARC's solver a second time after the shot stage, with the reference's LoRA
settings, on rank 16's confidence-0 tasks, cheapest first. `arc26.solver` queues the tasks of its
challenges file sorted by id, so the tail could not choose its order through it; this entry takes
the same arguments and runs the same workers (`arc26.solver.local_worker`) with the queue in the
challenges file's order. The notebook writes that file, in its own data folder, in the order it
chose.

## Interfaces

| Name | What |
|---|---|
| `python -m arc26.tail --end-time T --config JSON` | `arc26.solver`'s command line (`arc26.runner.solver_command(cfg, T, python, "arc26.tail")`): reads the mode's challenges file under `cfg["data_dir"]`, queues its ids in file order (outside a rerun only those in `subset`, when set), one `None` per worker, and spawns `cfg["workers"]` processes of `arc26.solver.local_worker`. |
| `challenges_path(data_dir, rerun)` | `<data_dir>/arc-agi_test_challenges.json` in a competition rerun (`KAGGLE_IS_COMPETITION_RERUN` set), else `arc-agi_evaluation_challenges.json`: the file the solver reads. |
| `queue_order(data, cfg, rerun)` | The ids in queue order. |
| `main(argv=None)` | The entry; returns 0 once the workers return. |

`arc26.runner.run_solver(cfg, end_time, python, timeout, module="arc26.tail")` starts it; the
`--config` JSON is `RunConfig`'s, so the LoRA settings are `RunConfig`'s defaults unless set
(rank 256 on the nine target modules, the reference's).

## Data flow

The notebook's tail cell: `nvarc_confidence` (the shot's `shot_confidence` over NVARC's
candidates) gives the tasks at exactly 0, `tail_tasks` orders them by NVARC's `puzzle` seconds,
the cell writes their challenges in that order to `<root>/tail_data/`, runs this entry with
candidate, marker, and timing folders of its own, selects with `make_submission`, and merges with
`tail_merge` (placeholders only). See `docs/runbooks/build_submission.md`, "The rank-256 tail".

## Known limits

The workers' deadline is the soft deadline of `--end-time`: a puzzle in progress then stops its
DFS, and puzzles not started never start, so the order decides which tasks the deadline drops.
