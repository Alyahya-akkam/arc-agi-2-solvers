# ADR-011: The solver's worker split into `arc26.ttt`, `arc26.decode`, and `arc26.solver`

- Status: Accepted, 2026-10-02: run `P2.0-01` passed its parity rule ($g(E', A) = 0.4$,
  environment equal to run A's, complete timing records; `docs/experiments/P2.0-01.md`). The
  default flipped to `"split"` on 2026-10-02 in a separate commit (`task/P2.0-flip`), after the
  human read the result and approved the flip; the configs of earlier runs state `solver: nvarc`.
  Proposed 2026-10-01. Source: task brief `docs/tasks/P2.0.md`.
- Tasks: P2.0; the refinement loop P2.1 to P2.5, which calls the split functions.
- Related: ADR-009 (the moved layout; its overturn criterion is this ADR's acceptance test).

## Context

The refinement loop trains on a changed pair set (the held-out pair removed, pseudo-pairs added,
BRIEF B5 P2.1 to P2.3) and decodes inputs that are not test inputs (the held-out input). The
moved `arc_solver.worker` does both inside one per-puzzle loop body, so neither can be called on
its own. ADR-009 kept the reference's cells whole because a split is a refactor that a GPU run
alone cannot verify: two runs of the reference agree on only $g(B, A) = 0.4$ of DEV4's test
outputs. It deferred the split to Phase 2, under the GPU parity check.

## Decision

1. **Move the statements into functions, change nothing they compute.** Line numbers refer to
   the generated `arc26/nvarc/arc_solver.py` (with P0.3b's timing records and P1.3's LoRA
   settings) and `starter.py`:

   | New | Holds | From |
   |---|---|---|
   | `arc26.ttt.fit` | adapter reset, `for_training`, 128 training views, trainer, unwrap, `for_inference`, cache release, memory and stats prints, the `train` record | `worker`, lines 331-382 |
   | `arc26.ttt` module level | `UnslothFixedTrainer`, `QwenDataCollatorForCompletionOnlyLM`, `logging.disable(logging.WARNING)` | lines 60-105, 36 |
   | `arc26.decode.decode_puzzle` | 16 decoding views per test input, the batch order, DFS per batch, rescoring under 8 views, candidate files, the `decode` and `selection` records | `worker`, lines 384-495 |
   | `arc26.decode` module level | `turbo_dfs`, `inference_turbo_dfs`, `calc_scores`, `ARC_VOCAB` to `EOS_ID` | lines 108-228, 38-57 |
   | `arc26.solver.load` | settings, model load and PEFT wrapping, bf16 cast, default weights, the `load` record, collator, formatter, constants, challenges, candidate folder | `worker`, lines 233-316 |
   | `arc26.solver.worker` | the per-puzzle loop: deadline and queue, `fit`, `decode_puzzle`, inference memory, the `puzzle` record | `worker`, lines 318-329 and 497-503 |
   | `arc26.solver.local_worker`, entry block | the process body and `python -m arc26.solver` | `starter.py`, lines 20-42 and 45-73 |

   A block's locals become the new function's parameters with the same names, in order of first
   use, and the function returns exactly the locals that the rest of the reference reads after
   the block: `fit` returns `(model, puzzle_ds)`, `decode_puzzle` nothing, `load` the 12 names
   of `Loaded`. The worker unpacks `load(cfg)` into those names and calls `fit` and
   `decode_puzzle` with keyword arguments of the same names, in the reference's order. New code
   is limited to signatures, returns, docstrings, type hints, the function-local imports of
   point 2, and the two calls; the layout changes only where long lines are wrapped. The
   imports are the reference's, less `tqdm`, which `arc_solver.py` imports but never uses.
2. **Import order as in the reference.** The reference's parent process imports torch only, and
   each worker imports `arc_solver` (unsloth first) in `local_worker`, after
   `CUDA_VISIBLE_DEVICES` is set and before it writes the marker that lets the next rank start.
   The split keeps this: importing `arc26.solver` imports torch only, `local_worker` imports
   `arc26.decode` and `arc26.ttt` at the reference's import point, and both modules import
   unsloth before anything that imports transformers. Since `arc26.solver` must stay light,
   `load` and `worker` import the GPU names they read at their start.
3. **Proof, twice.** On CPU, `tests/test_split_solver.py`: every moved block is `ast.dump`-equal
   to its slice of the reference, and the imports are compared by module and alias; parameters
   and returns follow from the reference's name flow, with no decorator, default, `*args`, or
   `**kwargs` on the new functions; both solvers, run against stand-ins for the GPU stack, write
   byte-identical candidates, leave numpy's random state equal, and write the same timing
   records; and `python -m arc26.solver` spawns its workers as the starter does. On GPU, run
   `P2.0-01`: DEV4 on 4xL4 with the split
   solver, which passes with $g(E', A) \ge 0.4$ against `tests/fixtures/p0_1_reference_dev4.json`
   and complete timing records for every DEV4 puzzle.
4. **Selection.** `RunConfig.solver` (`"nvarc"` or `"split"`) picks the process entry
   (`arc26.nvarc.starter` or `arc26.solver`); the `--config` JSON is the same. The notebook
   builder reads an optional config key `solver` (absent: `split` since the flip, `nvarc`
   before) into the Config cell's `SOLVER`, which applies in both modes. The default flips to
   `split` in a separate commit, after the human has read `P2.0-01`'s result (done 2 Oct).
5. **The anchor stays.** `arc26/nvarc/` and `configs/extraction.yaml` are unchanged: the moved
   reference remains the parity anchor that the AST test reads, and was the default solver until
   the flip.

## Production builds

`submission_blocker` does not read `solver`. Until `P2.0-01` passed, a build meant for
submission left `solver` out, so it ran the moved reference; since the flip (2 Oct) a production
config leaves it out and runs the split solver. Since this ADR is Accepted, a split-solver
production build needs what a production build needs today (the submission notebook, four
workers on `NvidiaL4`, no series) and nothing more, because both solvers run the same
statements; `P2.0-01`'s per-puzzle times, against `P0.5-01`'s and `P1.1-01`'s on the same
hardware, confirm that the split keeps the scored rerun's time budget.

## Alternatives considered

- **Import the classes and constants from `arc26.nvarc.arc_solver` and move only the functions.**
  Lost: the B4 modules would depend on the anchor that ADR-009's overturn retires from the spec.
- **Generate the split from `configs/extraction.yaml`.** Lost: the spec makes literal
  replacements; turning a block into a function with parameters and returns is not one, and the
  AST test checks the move as strictly as `--check` checks the spec.
- **Edit the generated `arc_solver.py` in place.** Lost: it would change the parity anchor
  while the split is unproven, and drop the fallback to the moved reference.
- **Pass one state object instead of named locals** (`state.model`). Lost: every moved statement
  would change, so no statement would be verbatim.
- **Split without a GPU run, on the CPU checks alone.** Lost: the CPU checks run stand-ins, not
  unsloth, transformers, and CUDA; ADR-009's criterion asks for the GPU parity run.

## Consequences

- P2.1 to P2.5 call `fit` and `decode_puzzle` directly: training on a changed pair set means
  passing another `ArcDataset` as `arc_test_set`, and decoding the held-out input means passing
  it as `puzzle_ds`.
- The solver's statements exist twice while the moved reference stays the parity anchor (after
  the flip too), generated in `arc26/nvarc/arc_solver.py` and moved in the split modules;
  `tests/test_split_solver.py` fails as soon as they differ.
- `arc26/decode.py` is a protected path (BRIEF A0): every change must keep the P0.1 parity
  fixture green, and its PRs carry `needs-human`.
- Importing `arc26.ttt` disables logging at `WARNING` and below for the process, and importing
  `arc26.decode` imports unsloth, as importing `arc_solver` did.
- The blocks' locals are freed when `fit` and `decode_puzzle` return, where the reference kept
  them until the next puzzle rebound them: CPU objects only, with no effect on what is computed.
- `MANIFEST.md`'s config summary, the notebook's first cell, and its Config cell show the solver,
  so a human about to submit sees it beside "fit to submit".

## Overturn criterion

Reject or amend this ADR if `P2.0-01` fails parity ($g(E', A) < 0.4$) while the CPU checks pass:
the moved statements are then equal, and the difference lies outside them (import order, the
process entry, the environment), which the per-view comparison of that run's Parity fail row
locates. A later change to `fit` or `decode_puzzle` that changes a statement needs its own ADR
and parity check, as ADR-009's Amendment 1 did for the timing records.
