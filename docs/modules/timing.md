# Module: `arc26/timing.py` and `scripts/timing_report.py`

Task P0.3a (BRIEF B3, B4 conventions, B5 P0.3); wiring into the moved solver and the runner,
task P0.3b. Tests: `tests/test_timing.py`, `tests/test_timing_report.py`,
`tests/test_solver_timing.py` (the wiring). Pure Python, no GPU code.

## Purpose

B4 makes timing logs mandatory: every GPU operation appends a JSONL record
`{puzzle_id, op, seconds, tokens}`. P1.1 turns them into the per-puzzle timing table and the cost
model that replace B3's estimates, and the Phase 2 and Phase 4 budgets (the governor, P2.4) derive
from them.

## Record and files

| Field | Type | Meaning |
|---|---|---|
| `puzzle_id` | string or null | task id; null for an op outside any puzzle, such as model loading |
| `op` | non-empty string | operation name |
| `seconds` | number $\ge 0$, finite | wall time from `time.perf_counter`, rounded to microseconds |
| `tokens` | integer $\ge 0$ or null | tokens the op processed; null when unknown. Any integer type (a numpy or torch integer scalar) is stored as a plain `int` through `operator.index`; booleans and non-integers are refused |

Each process writes `<log_dir>/timing-<worker>.jsonl`, where `<worker>` is the label passed to
the logger (the queue rank, or `main` for the main process) or `pid<process id>` by default. One
file per process, because concurrent appends from four processes to one file are not guaranteed
to keep lines whole; a lock separates threads within a process. Each record is one append of one
complete line to a file opened and closed for it, so a hard kill loses at most the line being
written, and the reader skips a truncated line.

Standard op names, placed by where the reference notebook runs each step:

| Constant | `op` | Covers | Where |
|---|---|---|---|
| `PUZZLE` | `puzzle` | the whole puzzle on its worker | worker; encloses the next three |
| `TRAIN` | `train` | the per-puzzle TTT fit | worker, inside the puzzle span |
| `DECODE` | `decode` | DFS decoding (`inference_turbo_dfs`), per batch | worker, inside the puzzle span |
| `SELECTION` | `selection` | rescoring each decoded candidate under augmented views (`calc_scores`, 8 views per new candidate) | worker, inside the puzzle span |
| `MAIN_SELECTION` | `main_selection` | the selection algorithm over all puzzles (`run_selection_algo`, cell 7) | main process, after the workers; outside every puzzle span, one record with a null `puzzle_id` |
| `arc26.loop.GATE` | `gate` | the held-out gate's whole span on a puzzle (task P2.1); also the candidate adapter's gate check in `promote` (P2.2) | worker, inside the puzzle span, after the test decode; with hold-out on, the puzzle span includes it |
| `arc26.loop.GATE_DECODE` | `gate_decode` | the gate's DFS batches on the held-out input | worker, inside the gate span |
| `arc26.loop.GATE_SELECTION` | `gate_selection` | the gate's rescoring of each candidate | worker, inside the gate span |
| `arc26.loop.REDECODE` | `redecode` | the promotion's re-decode of one test input under fresh views (task P2.2): decode, read back, vote | `promote`, inside the puzzle span, after the test decode |
| `arc26.loop.REDECODE_DECODE` | `redecode_decode` | the re-decode's DFS batches (`decode_puzzle` with `view_seed`, through `RenamedOps(logger, REDECODE_OPS)`), prompt tokens | worker, inside the redecode span |
| `arc26.loop.REDECODE_SELECTION` | `redecode_selection` | the re-decode's rescoring of each candidate | worker, inside the redecode span |
| `arc26.loop.CANDIDATE_TRAIN` | `candidate_train` | the candidate adapter's fit on the enlarged pair set (P2.2; Phase 2 `sgd_continue`), its trainer's `train` record through `RenamedOps(logger, CANDIDATE_OPS)` | worker, inside the puzzle span, between the re-decodes and the candidate's gate |

Other names are free; the solver also records `load`, the model load, with a null `puzzle_id`
(Wiring, below). The overhead of a
puzzle is its `puzzle` time minus train, decode, and selection. All three run inside the span, so
overhead is never negative; the main-process selection is not part of any puzzle's time and
appears under the report's ops outside puzzles.

## Interfaces

| Name | What |
|---|---|
| `TimingLogger(log_dir, worker=None)` | `record(puzzle_id, op, seconds, tokens=None)` appends a line; `timed(puzzle_id, op, tokens=None)` is a context manager that yields an `OpTimer` and records the block when it exits normally; attributes `log_dir`, `worker`, `path` |
| `OpTimer` | `tokens`, settable inside the block; `seconds`, set on exit |
| `configure(log_dir, worker=None)` | sets this process's logger |
| `timed(puzzle_id, op, tokens=None)` | as `TimingLogger.timed` on this process's logger; before `configure`, a stopwatch that writes nothing |
| `get_logger()` | this process's logger or None |
| `make_record(...)` | validates the fields; `TypeError` or `ValueError` on a bad one |
| `read_records(paths)` | `(records, skipped)`: `TimingRecord`s in file and line order, and the count of lines that are not valid records |
| `TimingRecord`, `worker_label(path)` | a record read back, with the worker from its file name |

A block that raises writes no record, so a failed op never becomes a cost-model sample; its time
still falls into the enclosing `puzzle` record as overhead.

## Wiring (P0.3b)

The solver's workers and the selection record their ops when `arc26.runner.RunConfig.timing_dir`
is set; with it null (the default, and every competition rerun) nothing is counted or written.
The moved modules get the calls through `configs/extraction.yaml` (ADR-009): every timing
replacement keeps the reference line it is anchored on and only adds lines, reads only clocks and
objects that already exist, and draws no random number, so the solver's data and the order of its
steps do not change (`tests/test_extraction.py`, `tests/test_solver_timing.py`).

| Where | Call | Effect |
|---|---|---|
| `runner.RunConfig.timing_dir` | passed to the solver as `cfg["timing_dir"]` | null: no timing anywhere |
| `nvarc/starter.py`, `local_worker` | `timing.configure(cfg["timing_dir"], f"rank{rank}")` when not null, before the worker runs | `<timing_dir>/timing-rank<rank>.jsonl`, one file per worker process |
| `nvarc/arc_solver.py`, `worker` | `timing_logger = timing.get_logger()` once, before the model load | None unless configured; every record below is guarded by `if timing_logger is not None:` |
| same | `<op>_start = time.perf_counter()` before the timed statements, `timing_logger.record(...)` right after them | one record per op as below |
| `runner.make_submission` | `TimingLogger(timing_dir, "main").timed(None, MAIN_SELECTION)` around `write_submission` | `<timing_dir>/timing-main.jsonl`; a logger of its own, so the calling kernel keeps no process-wide logger |

The worker records spans measured across several statements (the whole puzzle, two rescoring
calls), which a context manager could only wrap by re-indenting the moved code, so it uses
`TimingLogger.record` with `time.perf_counter` differences, the clock `timed` uses. The seconds
are taken before any token count is computed, so counting never inflates an op's time.

| `op` | `puzzle_id` | Span | `tokens` |
|---|---|---|---|
| `load` | null | `FastLanguageModel.from_pretrained` through the snapshot of the initial adapter (`get_peft_model`, the bf16 cast, `default_weights`) | null |
| `train` | task id | `trainer.train()` | $\sum_i \lvert x_i \rvert$ over the trainer's prepared training samples $x_i$ (`trainer.train_dataset["input_ids"]`), the input ids the collator receives: one epoch at batch size 1, so each sample once. Null if that column is missing; never an error |
| `decode` | task id | one `inference_turbo_dfs` call (one batch of 4 augmented views) | prompt tokens of the batch, $\sum_{v=1}^{4} \lvert p_v \rvert$ over the token lists the call receives; generated tokens are not counted |
| `selection` | task id | the pair of `calc_scores` calls that rescore one new candidate grid under 8 augmented views | $\sum_{v=1}^{8} (\lvert q_v \rvert + \lvert a_v \rvert)$, query plus answer tokens without padding, tokenized again after the time is taken (`calc_scores` keeps its token lists) |
| `puzzle` | task id | from the reference's `start_time = time.time()` to its `finished` print | null |
| `main_selection` | null | `write_submission` in the main process (cell 7: load candidates, select, write, and in development score) | null |

Every train, decode, and selection span lies inside its puzzle span, so overhead is never
negative. A puzzle cut by the 1200 s cap or the end time breaks out of its decode loop only and
still reaches its `puzzle` record; an op that raises writes no record (a crashed worker leaves its
puzzle without one). A DEV4 run decodes 20 batches and rescores 29 (`P0.5-01`) to 36 (`P1.1-01`)
new candidates (measured from the session logs' `decoding` and `scoring` lines), so with four
workers it yields 4 `load` records (one per `timing-rank<rank>.jsonl`), 4 `train`, 20 `decode`,
about 29 to 36 `selection`, and 4 `puzzle` records spread over those files by the puzzle each
worker took, and 1 `main_selection` record in `timing-main.jsonl`.

The notebook (`scripts/build_kaggle_notebook.py`) sets `timing_dir` to
`/kaggle/working/run_logs/timing/` in development mode and to null in a competition rerun; the
pull (`scripts/kaggle_ops.py pull`, allowlist `run_logs/**/*.jsonl`) brings the files back to
`runs/<task-id>/run_logs/timing/`. Then:

```bash
python scripts/timing_report.py runs/<task-id>/run_logs/timing \
    --out reports/<task-id>-timing.md --title "<task-id> timing"
```

CUDA kernels run asynchronously, so a timed span must end on a host synchronization. Each one
does: `trainer.train()` ends on the loss's `.item()`, every DFS step and every `calc_scores` call
copies logits to the host (`.cpu()`), and a puzzle ends after its last decode or rescoring.

## `scripts/timing_report.py`

```bash
python scripts/timing_report.py runs/<task-id>/run_logs [more dirs or files] \
    [--out reports/<task-id>-timing.md] [--title "<task-id> timing"]
```

A directory is searched recursively for `timing-*.jsonl`. The Markdown has three sections:

1. **Per puzzle:** worker, train, decode, selection, overhead, and total seconds, and the train,
   decode, and selection token counts (summed over the puzzle's records of each op; `n/a` when
   none carries a count), then a mean row and a sum row. Overhead is the `puzzle` time minus the
   three phases; for a puzzle without a `puzzle` record, overhead is the sum of its other ops and
   the total the sum of all its ops.
2. **Ops outside puzzles:** records with a null `puzzle_id` (each worker's model loading, the
   main-process selection), by op: count, total, and mean seconds.
3. **Cost model:** per op, the ordinary least-squares fit over its $n$ records that carry a token
   count $x_j$, with seconds $y_j$:

$$\text{seconds} = a + b \cdot \text{tokens}, \qquad
b = \frac{\sum_j (x_j - \bar x)(y_j - \bar y)}{\sum_j (x_j - \bar x)^2}, \qquad
a = \bar y - b \bar x, \qquad
R^2 = 1 - \frac{\sum_j (y_j - a - b x_j)^2}{\sum_j (y_j - \bar y)^2}.$$

With fewer than two distinct token counts, only $a = \bar y$ is reported; $R^2$ is n/a when every
sample takes the same time. Sums use `math.fsum`. Skipped lines are counted in the header. Exit
status 1 when a path does not exist or no valid record is found.

## Tests

`tests/test_timing.py`: the four fields in order, null tokens; file per worker and the default
process-id label; unsafe labels refused; nothing written before the first record; the timed block
with a late token count and a fake clock; a raising block writes nothing; twelve invalid-field
cases; integer-scalar token counts (a stand-in for numpy's) stored as plain ints; the
module-level stopwatch before `configure` and the configured logger after; 8 threads on one logger
and 4 forked worker processes, all lines whole; the reader skipping blank, truncated, and invalid
lines while allowing extra keys.

`tests/test_timing_report.py`: the fit on an exact line, against `statistics.linear_regression`
and `statistics.correlation` on noisy data, and in its degenerate cases; per-op fits; overhead with
and without a `puzzle` record; the main-process selection never entering a puzzle's overhead;
worker labels; ops outside puzzles; a synthetic DEV4 run of four workers and the main process
through `TimingLogger` and the CLI, checked cell by cell; a truncated line; recursive search;
error exits; a log without puzzles; the script run as a program.

`tests/test_solver_timing.py` (P0.3b): on the AST of the generated `arc_solver.py` and
`starter.py`, each op's span holds exactly the statements it times, the train, decode, and
selection spans lie inside the puzzle span, nothing in a puzzle (the 1200 s cap included) skips
its record, every record call is guarded by the worker's logger, the seconds are taken before the
token count, and only `local_worker` configures a logger, only for a non-null `timing_dir`. Then
`starter.local_worker` runs in a subprocess against stand-ins for torch, unsloth, datasets,
transformers, peft, and tqdm, with a one-token-per-character tokenizer and fakes for the DFS and
the rescoring, on two synthetic tasks: a null `timing_dir` writes nothing but the candidates and
the marker; with one, every op is recorded in order with the token counts above; a trainer
dataset without `input_ids` gives null train tokens; a puzzle cut by the cap keeps its records;
and the candidates and numpy's random state are identical with and without timing.
`tests/test_runner.py` covers `main_selection` (one record, none without `timing_dir`, none when
the selection raises); `tests/test_notebook_local.py` the notebook (development writes
`run_logs/timing/timing-main.jsonl` and the run_logs dump keeps it as is; a competition rerun
writes no timing file).

## Known limits

- Wall time, not GPU time: concurrent work on the same GPU inflates every op that overlaps it.
- A linear model in tokens ignores the quadratic attention term, which matters at the 8192-token
  cap; $R^2$ and the residuals in P1.1 show whether a second regressor is needed.
- Decode tokens count the prompts only. The DFS's time depends mostly on how far it branches,
  which no prompt count captures; the decode fit's $R^2$ shows how much prompt length explains.
- The token counts cost a little time outside the timed spans, inside the puzzle's overhead
  (**estimate**): the train count reads the prepared `input_ids` column, at most about $10^6$
  integers (128 samples of up to 8192 tokens), under 0.1 s per puzzle; the selection count
  tokenizes 16 strings again, a few milliseconds per new candidate. Each record is one append of
  under 100 bytes.
- A worker that crashes, or a solver killed at the hard stop, leaves its last puzzle without a
  `puzzle` record; the report then shows that puzzle with zero overhead and a total equal to the
  sum of its recorded train, decode, and selection time.
- `arc26/timing.py`'s module docstring still describes the generic pattern (every process calls
  `configure`, then `timed`); the wiring above is what P0.3b does: the main process uses a logger
  of its own, and the workers record spans with `TimingLogger.record`.
