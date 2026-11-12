# Module: `arc26/evaluate.py`

Task P0.2 (BRIEF B4, B5). Tests: `tests/test_evaluate.py`. Pure Python, no GPU. Subsets come from
[`arc26/data.py`](data.md).

## Purpose

The competition metric on a predictions file, in its per-task form and the per-output variant,
always with the task and test-output counts; the agreement $g$ between two runs, which is the
P0.1 parity criterion; and a small CLI for both.

## Metric

A submission maps each task id to a list, one entry per test input in order, of
`{"attempt_1": grid, "attempt_2": grid}`. A solutions file maps each task id to its list of
output grids. For test output $i$ of task $t$ with solution $y_{t,i}$ and attempts
$a^{(1)}_{t,i}, a^{(2)}_{t,i}$:

$$c_{t,i} = \mathbb{1}\!\left[a^{(1)}_{t,i} = y_{t,i} \ \lor\ a^{(2)}_{t,i} = y_{t,i}\right].$$

With $n_t$ test outputs per task and $T$ the scored tasks, the per-task metric and the per-output
variant are

$$S = \sum_{t \in T} \frac{1}{n_t} \sum_{i=1}^{n_t} c_{t,i}, \qquad
\text{score} = \frac{100\, S}{|T|}\,\%, \qquad
\text{output score} = \frac{100}{N} \sum_{t \in T} \sum_{i=1}^{n_t} c_{t,i}\,\%, \quad
N = \sum_{t \in T} n_t .$$

$S$ is what the organizers' scorer and every 2025 leaderboard score compute, and what the
reference notebook's `validate_submission` prints; the per-output variant follows the wording of
the competition pages (`docs/STATE.md`, Notes). Reports always carry $|T|$ and $N$, for example
`task score: 2.5 of 4 tasks (62.5%)`.

**Equality.** Exact: both grids are well formed and equal as lists. A grid is well formed when it
is a list of 1 to 30 rows, all lists of the same length 1 to 30, whose cells are `int` in 0..9
(`is_grid`). A boolean or float cell is malformed, although `True == 1` and `1.0 == 1` in Python.

**Tolerance.** A task absent from the submission scores 0 and is listed as missing. A malformed
or absent attempt (a task value that is not a list, an entry that is not an object, a missing
`attempt_2`, a ragged grid, fewer entries than test outputs) scores 0, is counted as malformed,
and never raises; the other attempt still counts. Extra entries and extra keys are ignored. A task
to score with no solutions, or with malformed solutions, is a usage error (`ValueError`, CLI exit
status 2), because it would change the denominator silently.

## Agreement

For a run $X$ and a reference run $A$, over the $N$ test outputs that $A$ holds for the compared
tasks (`docs/prerun/P0.1-01.md`, section 1):

$$g(X, A) = \frac{1}{N} \sum_{j=1}^{N}
\mathbb{1}\!\left[\left(x^{(1)}_j, x^{(2)}_j\right) = \left(a^{(1)}_j, a^{(2)}_j\right)\right],
\qquad
g_1(X, A) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}\!\left[x^{(1)}_j = a^{(1)}_j\right].$$

Order matters: swapped attempts do not agree. $g_1$, the `attempt_1` agreement, is a diagnostic
next to $g$: it shows whether a shortfall comes from the first choice or only from the second.
A task missing from $X$ counts as disagreeing on all its outputs; a requested task missing from
$A$ is not compared and is listed. Malformed attempts never agree, not even with an identical
malformed attempt. P0.1 passes when the extracted code $E$ reaches $g(E, A) \ge 0.4$, the noise
floor $g(B, A)$ measured by `P0.1-01`.

## Interfaces

| Name | What |
|---|---|
| `is_grid(value)` | well-formed grid check |
| `score_task(task_id, predictions, solutions)` | `TaskScore`: `correct` per output, `missing`, `malformed`, `score` |
| `score_submission(predictions, solutions, task_ids=None, label="all")` | `SubsetScore`: `score` ($S$), `percent`, `solved_outputs`, `output_percent`, `tasks`, `test_outputs`, `missing_tasks`, `malformed`, `report()` |
| `check_solutions(solutions, task_ids)` | raises `ValueError` on absent or malformed solutions |
| `agreement(predictions, reference, task_ids=None, label="all")` | `Agreement`: `g`, `attempt_1`, `tasks`, `test_outputs`, per-output `outputs`, `missing_in_predictions`, `missing_in_reference`, `report()`; `ValueError` when nothing is left to compare |
| `load_predictions(path)` | a submission file, or the `predictions` object of a file that has one (the P0.1 fixture) |
| `load_solutions(path)` | a solutions file |
| `main(argv)` | the CLI |

## CLI

```bash
python -m arc26.evaluate score <submission> <solutions> [--subset DEV4|DEV40|DEV120] [--per-task]
python -m arc26.evaluate agree <submission> <reference> [--subset DEV4|DEV40|DEV120] [--per-task]
```

Without `--subset`, `score` covers every task of the solutions file and `agree` every task of the
reference. `--per-task` lists each task's score, or each output's agreement. Exit status 2 on an
unreadable or malformed input file.

The P0.1 check, run B against the parity fixture (run A):

```console
$ python -m arc26.evaluate agree runs/P0.1-B/submission.json tests/fixtures/p0_1_reference_dev4.json --subset DEV4
agree DEV4: 4 tasks, 5 test outputs
g = 0.4 (2 of 5 test outputs, both attempts identical in order)
attempt_1 agreement = 1.0 (5 of 5 test outputs)
missing in submission: 0; missing in reference: 0
```

`score` cannot be checked on run A locally: the dev box holds no solutions (A0). Run A's DEV4
score, 2.5 of 4 tasks, is the notebook's own printed number (`*** Reload score: 2.5`,
`docs/experiments/P0.1.md`).

## Data flow

A run's `submission.json` (pulled into `runs/<task-id>/`) and, in dev mode on Kaggle,
`arc-agi_evaluation_solutions.json` -> `score`. Two runs' `submission.json` files, or one and the
fixture -> `agree`. Results go into `docs/experiments/<task-id>.md` and `RESULTS.md` with the
subset and counts.

## Origin

The metric reproduces the reference notebook's `ArcDataset.validate_submission`
(`reference/reference_notebook.ipynb`, cell 2, `arc_loader.py`) in pure Python; it is not moved
code, since the notebook version needs numpy and its dataset class. A test compares both on 200
random submissions. One deliberate difference: the cell check is strict (`int` only). Both
`np.array_equal` in the notebook and the list `==` of the official scorer also accept `1.0` and
`True` for 1; `is_grid` does not. Our decoder always writes ints: every attempt is
`np.array(..., dtype=int).tolist()` (`convert_tokens_to_array`, `fill_submission`) or the `[[0]]`
placeholder (`get_submission`), so no score or $g$ of ours can differ from the official scorer's.

## Tests

`tests/test_evaluate.py`: well-formed and malformed grids; either attempt scores, neither does not;
partial credit over 2 and 3 test outputs; the per-task and per-output variants on a case where
they differ; missing tasks; ten malformed-prediction shapes, none raising; float grids equal in
value do not score; the notebook metric on random submissions; agreement with itself, on the
run B versus run A shape built from the fixture ($g = 0.4$, $g_1 = 1.0$), with swapped attempts,
missing tasks on either side, malformed attempts, and nothing to compare; file loading including
the fixture wrapper; the CLI (`score`, `agree`, `--subset`, `--per-task`, error status 2, unknown
subset, `python -m`).

## Known limits

- The format checks here are only what scoring needs; the submission validator and the
  crash-safe fallback are `arc26/submit.py` (P0.5b).
- Solutions files are dev mode only: never read in competition mode, never used in training
  (BRIEF A0).
