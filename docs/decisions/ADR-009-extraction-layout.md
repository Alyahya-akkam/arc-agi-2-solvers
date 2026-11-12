# ADR-009: Moved notebook code lives in `arc26/nvarc/`, one module per cell, generated from a spec

- Status: Accepted, 2026-09-30; amended 2026-10-01 (Amendment 1, timing insertions). Source:
  task brief `docs/tasks/P0.1.md`; Amendment 1, `docs/tasks/P0.3b.md`.
- Tasks: P0.1 (extraction), P0.5 (the notebook calls `arc26.runner`), P0.3b (timing records),
  later refactors (P2).

## Context

BRIEF B4 lists target modules "from the notebook" (`augment`, `tokenize`, `ttt`, `decode`,
`runner`). The reference notebook keeps its code in four `%%writefile` cells (`arc_loader.py`,
`arc_decoder.py`, `arc_solver.py`, `starter.py`) and one cell it runs inline (cell 7, selection).
Every B4 module would take pieces of several cells: `arc_solver.worker` alone holds test-time
training, DFS decoding, and augmented rescoring. Splitting now is a rewrite, which P0.1 forbids
("move, do not rewrite"). The GPU check of the moved code (run `P0.5-01`) cannot catch a small
refactor mistake either: two runs of the reference itself agree on only $g(B, A) = 0.4$ of DEV4's
test outputs (`docs/experiments/P0.1.md`).

## Decision

1. Each code cell moves whole into `arc26/nvarc/`, under its `%%writefile` name: `arc_loader.py`
   (cell 2), `arc_decoder.py` (cell 3), `arc_solver.py` (cell 4), `starter.py` (cell 5). Cell 7
   becomes `select.py`: its imports stay at module top and the rest becomes the body of
   `write_submission(data_dir, outputs_dir, submission_path)`.
2. `scripts/extract_reference.py` generates the modules from `configs/extraction.yaml`: the
   notebook's sha256 and, per module, the source cell and literal `{old, new, count}`
   replacements, each with its reason. The spec is the complete list of deviations from the
   reference. `tests/test_extraction.py` fails whenever a committed module differs from the
   spec's output (`--check`).
3. The replacements only turn hard-coded values into run settings: data, model, candidate, and
   marker paths from a `cfg` dict; the DEV4 filter into `subset`; the worker count into
   `workers`; file imports into package imports. With the reference's own values the behavior is
   unchanged. Amendment 1 (below) adds a second kind: timing insertions.
4. `arc26/runner.py` is our code with the B4 name: `RunConfig`, the launch command and
   environment, `run_solver`, and `make_submission`. B4 names (`runner`, `evaluate`, `submit`,
   `timing`, later `ttt`, `decode`, `loop`) stay reserved for our code.
5. `arc26/nvarc/` and `configs/extraction.yaml` are protected paths (the human merges). Ruff
   skips `arc26/nvarc/`, since a lint fix would change moved code.

## Alternatives considered

- **Split into the B4 modules now.** Lost: a rewrite whose mistakes no CPU check can see, judged
  only by a GPU run with a noise floor of 0.4.
- **Hand-copied modules, reviewed once in the PR.** Lost: nothing would keep the copies equal to
  the notebook; reformatting or a lint fix would drift unnoticed.
- **Run the notebook's own `%%writefile` cells at run time.** Lost: nothing importable, testable,
  or configurable, and P0.5 needs a package.

## Consequences

- Provenance is obvious: every module names its cell, and every deviation is one reviewable YAML
  list. The CPU replay (`tests/test_replay.py`) reproduces run A's DEV4 attempts exactly from run
  A's candidates.
- Moved code keeps the notebook's style (no docstrings or type hints, trailing whitespace). Ruff
  excludes it; `tests/test_extraction.py` guards it instead.
- B4's `augment.py`, `tokenize.py`, `decode.py`, and `ttt.py` do not exist yet. P2 splits
  `arc_solver.worker` under the GPU parity fixture.
- P2.0 (ADR-011, Accepted 2 Oct 2026) moves the statements of `arc_solver.py` and
  `starter.local_worker` into `arc26/ttt.py`, `arc26/decode.py`, and `arc26/solver.py`, and leaves
  this layout and the spec unchanged as the parity anchor. Run `P2.0-01` passed the overturn
  criterion below for `arc_solver.worker` (2 Oct), and the default solver flipped to the split
  (`RunConfig.solver = "split"`); the moved modules remain the parity anchor. Retiring
  `arc_solver.py` from the spec, now that it has passed, is a separate change, since
  `tests/test_split_solver.py` compares the split with it.
- `select.py` shadows the standard library's `select` whenever `arc26/nvarc/` itself is on
  `sys.path`, so the solver is started only as `python -m arc26.nvarc.starter`, never by file
  path.

## Overturn criterion

Replace this layout for a module once a refactor that splits it into B4 modules passes the GPU
parity check ($g(E, A) \ge 0.4$ on DEV4, `tests/fixtures/p0_1_reference_dev4.json`); the spec
then keeps only the cells still moved verbatim.

## Amendment 1 (2026-10-01, task P0.3b): timing insertions

BRIEF B4 requires a JSONL timing record for every GPU op, and the GPU ops run inside the moved
`arc_solver.worker`, so the timing calls have to be in moved code. Decision point 3 therefore gains
a second kind of replacement, the timing insertion: 11 entries of `configs/extraction.yaml`, 9 in
`arc_solver.py` and 2 in `starter.py` (`docs/modules/timing.md`, Wiring).

1. **Insertions only.** A timing replacement keeps the one reference line it is anchored on
   unchanged and adds lines before or after it; no reference line is changed or removed. A
   module's diff against its cell is therefore the point-3 replacements plus inserted lines. The
   inserted lines read clocks and objects that already exist (for the token counts) and append
   records; they draw no random number and change neither the data the solver sees nor the
   order of its steps.
2. **No effect without a timing folder.** Every inserted record call is guarded by the worker's
   logger, which only `starter.local_worker` creates, and only when `cfg["timing_dir"]` is not
   null; with null nothing is counted or written. `RunConfig.timing_dir` defaults to None, and the
   notebook passes null in every competition rerun (development runs pass `run_logs/timing/`),
   so the scored path is the one validated before timing existed.
3. **Enforced by tests.** `tests/test_extraction.py` rebuilds each module with only the point-3
   replacements and fails unless the committed module differs from it by inserted timing lines
   alone. `tests/test_solver_timing.py` fails unless, on the AST, every record call is guarded and
   only `local_worker` configures a logger, for a non-null `timing_dir`; it also runs the wired
   worker against stand-ins for the GPU stack and fails if a null `timing_dir` writes anything or
   if timing changes the candidates or numpy's random state. `tests/test_notebook_local.py` fails
   unless a competition rerun passes a null `timing_dir` and writes no timing file.

A later change to moved code that is neither a point-3 replacement nor a timing insertion with
these three properties needs its own amendment or ADR.
