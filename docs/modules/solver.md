# Module: `arc26/solver.py`

Task P2.0 (BRIEF B4, B5 P2.1 to P2.3), ADR-011; the hold-out branch, task P2.1 (ADR-013,
`docs/modules/loop.md`), and its rounds, task P2.3 (ADR-015). Tests: `tests/test_split_solver.py`, `tests/test_runner.py`,
`tests/test_notebook_local.py`, `tests/test_gate.py`. Importing it imports torch only; its
workers import the GPU stack.

## Purpose

The split solver's model load, per-puzzle worker, and process entry: the reference's worker with
its loop body split into `arc26.ttt.fit` and `arc26.decode.decode_puzzle`, run as
`python -m arc26.solver --end-time T --config JSON`. It is equivalent to
`python -m arc26.nvarc.starter`: the same arguments, task queue, worker markers, and
`torch.multiprocessing.spawn` with one worker per GPU. `arc26.runner` starts it when
`RunConfig.solver` is `"split"`, the default since run `P2.0-01` passed and the human approved
the flip (2 Oct, ADR-011). With `RunConfig.holdout` set (task P2.1), each puzzle holds one
demonstration pair out of training and prompts and runs the held-out gate
(`arc26.loop.solve_puzzle`); without it the worker runs the reference's statements unchanged.

## Interfaces

| Name | What |
|---|---|
| `load(cfg) -> Loaded` | Loads the model, wraps it with the LoRA adapter (`lora_r`, `lora_targets`), casts float32 parameters to bf16, copies the adapter's default weights, records `load`, and builds the collator, formatter, constants, challenges, and candidate folder. |
| `Loaded` | NamedTuple of the 12 locals the per-puzzle loop reads: `train_args`, `max_seq_length`, `timing_logger`, `model`, `tokenizer`, `default_weights`, `collator`, `formatter`, `max_new_tokens`, `max_score`, `arc_test_set`, `dir_outputs`. |
| `worker(rank, queue, end_time, cfg)` | Takes task ids until the queue is empty, a `None` arrives, or `end_time` has passed; per puzzle calls `fit`, then `decode_puzzle`, prints the peak inference memory and the puzzle's time, and records `puzzle`. With `cfg["holdout"]` (task P2.1, read once by `arc26.loop.HoldOut.from_cfg`), each puzzle calls `arc26.loop.solve_puzzle` instead, with the same locals, and (task P2.3) the queue's size right after the puzzle was taken and `cfg["workers"]`, the governor's inputs for the rounds. |
| `local_worker(rank, queue, end_time, cfg)` | The spawned process: pins it to GPU `rank`, waits for the marker of rank `rank - 1`, imports the GPU stack, writes its own marker, configures the timing logger when `timing_dir` is set, and runs `worker`. |
| `python -m arc26.solver` | Reads the challenges (the test file in a competition rerun, the evaluation file otherwise), queues the task ids in sorted order (outside a rerun only those in `subset`, when set) and one `None` per worker, and spawns `workers` processes. |

`cfg` is the JSON of `arc26.runner.solver_command`, the same for both solvers: `data_dir`,
`model_dir`, `outputs_dir`, `marker_dir`, `subset`, `workers`, `timing_dir`, `lora_r`,
`lora_targets`; and, only when a run holds a pair out, `holdout` (`arc26.loop.HoldOut.to_json()`,
which only the split solver accepts).

## Data flow

1. `arc26.runner.run_solver` starts `python -m arc26.solver` with the reference's launch
   environment and `PYTHONHASHSEED`.
2. The entry block spawns one `local_worker` per GPU. Each sets `CUDA_VISIBLE_DEVICES=rank`, and
   rank $r > 0$ waits for `marker_dir/worker{r-1}`; it then imports `arc26.decode` and
   `arc26.ttt`, which import unsloth before anything else of the GPU stack, and only then writes
   `marker_dir/worker{r}`. The imports therefore run one rank after another and after the GPU is
   pinned, as in the reference.
3. `worker` calls `load(cfg)` and unpacks it into the reference's names, then for each task id
   calls `fit` and `decode_puzzle` with keyword arguments of the same names; the candidate files
   go to `outputs_dir` and the timing records to `timing_dir/timing-rank{rank}.jsonl`. With
   `holdout`, it calls `arc26.loop.solve_puzzle` instead: `fit` on the puzzle's pairs less the
   held-out one, the test decode into `outputs_dir`, then the held-out gate into the gate's own
   folder, with its verdict in `holdout.log_dir`, and with `holdout.rounds` the refinement rounds
   (`docs/modules/loop.md`).
4. `arc26.runner.make_submission` selects the attempts from the candidate files, as for the
   moved reference.

## Origin

`reference/reference_notebook.ipynb` through the generated modules: `load` is cell 4,
`arc26/nvarc/arc_solver.py` lines 233-316 (the start of `worker`); `worker` is lines 318-329 and
497-503 around the two calls; `local_worker` and the entry block are cell 5,
`arc26/nvarc/starter.py` lines 20-42 and 45-73. Changes: type hints, docstrings, `load`'s
return, the function-local imports (the GPU names `load` reads, `fit` and `decode_puzzle` in
`worker`, and the two modules in `local_worker` in place of `arc_solver`'s `worker`), and the two
calls. Task P2.1 adds to `worker` the import of `HoldOut` and `solve_puzzle`, the settings
(`holdout = HoldOut.from_cfg(cfg)`), and an `if holdout is None:` around the two calls, whose
`else` calls `solve_puzzle` with the settings and the 16 locals the two blocks read. Task P2.3
adds to that call `queue_size=queue.qsize()` and `workers=cfg["workers"]`.
`tests/test_split_solver.py` lists them and checks every statement on the AST.

## Tests

- `tests/test_split_solver.py`: the moved statements and the wiring (above), the hold-out
  branch included: the two calls unchanged in its `if`, `solve_puzzle` with the settings and the
  locals both blocks read, in order of first use, in its `else`; the stand-in runs of both
  solvers (no hold-out) stay byte for byte equal; importing
  `arc26.solver` attempts no import of the GPU stack; `python -m arc26.solver` with two workers
  under a real spawn, on stand-ins: only the workers import unsloth, each after it is pinned to
  its GPU, and both write their markers and `load` records.
- `tests/test_runner.py`: `RunConfig.solver` picks `arc26.solver` with the same arguments.
- `tests/test_notebook_local.py`: the P2.0-01 notebook starts `python -m arc26.solver` in both
  modes on the simulated Kaggle tree; the P2.1-01 notebook passes its hold-out arm.
- `tests/test_gate.py`: the worker with a hold-out on GPU stand-ins (`docs/modules/loop.md`).

## Known limits

- `load` and `worker` import the GPU names at their start, because importing this module, as the
  parent process and every spawned worker do before `local_worker` runs, must not import them.
- Rank $r > 0$ polls for the previous marker every 5 s, as the reference does; markers left by
  an earlier launch let a worker start early, so the notebook removes them first.
- The puzzle's locals are freed when `fit` and `decode_puzzle` return, where the reference kept
  them until the next puzzle: CPU objects only, with no effect on what is computed.
