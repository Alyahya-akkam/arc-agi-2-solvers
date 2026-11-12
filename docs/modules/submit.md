# Module `arc26/submit.py`

Validator, repair, and crash-safe writer of the competition's `submission.json` (BRIEF A5, task
P0.5). Protected path (`needs-human`).

## Purpose

A competition rerun that ends without a valid `/kaggle/working/submission.json` scores nothing,
and the reference writes that file once, at the very end, with a plain `open` and `json.dump`: a
crash before that line leaves no file, and a kill during it leaves a truncated one. This module
makes "valid" one checkable definition and guarantees a valid file on disk from the moment the
notebook knows the task ids.

## Format and rules

For a challenges file with tasks $T$, where task $t$ has $n_t \ge 1$ test inputs, a submission is
valid when:

- its keys are exactly $T$;
- the value of task $t$ is a list of $n_t$ entries, in the challenges file's order;
- each entry has exactly the keys `attempt_1` and `attempt_2`;
- each attempt is a grid $g \in \{0, \dots, 9\}^{h \times w}$ with $1 \le h, w \le 30$, written
  as a list of $h$ rows, each a list of $w$ integers (rectangular; booleans refused, since JSON
  writes them as `true` and `false`).

The format follows the competition's data tab as copied in `docs/research/kaggle_first_runs.md`
(section 2) and NVARC's `get_submission`.

## Interfaces

| Function | What it does |
|---|---|
| `validate(submission, challenges) -> list[str]` | Every problem, one message each (for example `36a08778[1]: missing attempt_2`); empty when valid. |
| `repair(submission, challenges) -> dict` | A new, valid submission: each valid attempt copied, each missing or invalid one replaced by `[[0]]` (the reference's placeholder), missing tasks and entries added, unknown tasks, surplus entries, and extra keys dropped. |
| `finalize_submission(out_path, challenges_path, raw_path=None) -> dict` | The crash-safe writer: reads the raw submission if any (missing or unreadable is reported, not raised), repairs it, writes `out_path` atomically, reads it back, validates it, and returns a report. |
| `write_json_atomic(path, data)` | Temporary file in the same directory, `fsync`, `os.replace`: a reader sees the old file or the new one, never a part. |
| `grid_problem(grid)`, `entry_counts(challenges)`, `fallback_grid()` | The building blocks. |

The report of `finalize_submission` holds `raw_status` (`none`, `missing`, `unreadable`, `read`),
`tasks`, `test_inputs`, `problem_count` with up to 20 `problems`, `replaced_attempts` (attempts that
fell back to `[[0]]`), `zero_attempts` (attempts equal to `[[0]]` in the written file, which also
counts the reference's own placeholders for tasks it did not solve), and `valid`.

## Data flow in the notebook

1. Environment check: `finalize_submission(SUBMISSION, CHALLENGES)` with no raw file writes an
   all-`[[0]]` placeholder before anything is installed, so a session killed later still leaves a
   valid file.
2. Write cell: `make_submission` (the reference's selection, P0.1) writes a staging file outside
   `/kaggle/working`; `finalize_submission(SUBMISSION, CHALLENGES, STAGING)` runs in `finally`
   and is the only writer of `submission.json`.
3. The challenges file is `arc-agi_test_challenges.json` in a competition rerun and
   `arc-agi_evaluation_challenges.json` otherwise, as in the reference.

The notebook does not import this module from the installed package:
`scripts/build_kaggle_notebook.py` copies its source verbatim from the wheel into the
"Submission validator" cell, so it works before, and without, the install. That is why the module
imports only the standard library, nothing from `arc26`, and has no `__future__` import.

## Origin

New code. The `[[0]]` placeholder is the one the reference's `ArcDataset.get_submission` fills in
for attempts without a prediction.

## Tests

`tests/test_submit.py`: the A5 cases (missing task, missing attempt, ragged grid, out-of-range
value), boolean cells (refused although `bool` is an `int` subclass), invalid grid shapes and
types, wrong entry counts and extra keys, per-attempt repair,
truncated and binary raw files, the atomic write under a failing `os.replace` or serializer, an
exception mid-write, a process killed with `SIGKILL` (the placeholder survives), and that the
source runs standalone with standard-library imports only. `tests/test_notebook_local.py` runs the
embedded copy inside the simulated notebook in both modes.

## Known limits

- A valid file is not a good file: when `make_submission` fails outright, every attempt is
  `[[0]]`. The report's `raw_status` and `replaced_attempts` show it.
- Validation needs the challenges file. Without it (the competition data not attached) nothing can
  be written, and the notebook logs that.
- A process killed between the temporary write and `os.replace` leaves a `.submission.json.tmp`
  file next to a still-valid `submission.json`.
