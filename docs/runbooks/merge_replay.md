# Runbook: `scripts/merge_replay.py`

Task P6.9 (`docs/tasks/P6.9.md`; BRIEF A8). CPU only, no config file: its inputs are the pulled
logs of one shot run and NVARC's stored attempts, its thresholds the brief's pre-registered
values (constants in the script, each with its source). Brief: `docs/prerun/P6.9-01.md`; tests:
`tests/agent/test_p6_9_wave.py`.

## 1. Why this script exists

Run `P6.9-01` shoots 15 DEV120 tasks with the new request but without NVARC in the same session:
the 27B's answers are worth what Submission 4's merge (`arc26.agent.shot.merge_output`) would make
of them next to NVARC's attempts. This script computes that from `P2.6-01`'s stored attempts, as
`reports/P6.3-05/merge.md` (task P6.3, on `task/P6.3c`) did by hand for `P6.3-05`, and reads the
brief's decision rows: whether Submission 4's queue extends to $[0.5, 0.7)$, whether the
in-request forcing stays, and whether the harness needs an investigation first.

## 2. What it does, step by step

1. Read the run's `shot_metrics.json` (per task the outputs, `right` per output and attempt,
   `implied` shapes), `direct_attempts.jsonl` (the attempt records), and `budget_check.json`;
   NVARC's `submission.json` and its P2.8 rows (`reports/P2.8/P2.6-01.csv`).
2. Per test output $j$ of task $t$: $N_1, N_2$ from the submission, real where the rows hold an
   `attempt` 1 or 2 candidate (else NVARC's placeholder, None), right where that row's `correct`
   is 1; $S_1, S_2$ from the records by `arc26.agent.shot.shot_answers` with the logged implied
   shape (the run's own choice), right by `right`. The merged pair is
   $M = \operatorname{merge}(N_1, N_2, S_1, S_2)$, and the output is right under a rule when one of
   its two slots holds a right grid. No solutions file is read: every grid's correctness was
   scored by the run that made it.
3. Per task the competition's score, $s_t = \frac{1}{n_t}\sum_j \mathbb{1}[\text{output } j
   \text{ right}]$, for NVARC alone, the shot alone, and the merge; per bucket of NVARC's
   confidence $c$ ($0$, $(0, 0.5)$, $[0.5, 0.7)$, $[0.7, 0.85)$, $[0.85, 1]$) the sums, the net
   $\Delta = \sum_t (s_t^{M} - s_t^{N})$, and the tasks gained and lost.
4. The rows, in order: a Fault of the run (its `reading` starts with Fault) stops here; the
   harness check investigates at 0 shot solves of the $[0.85, 1]$ tasks; the queue extends
   (with $[0.5, 0.7)$ after $(0, 0.5)$ and before confidence 0) when the middle bucket's
   $\Delta - n\ell \ge 1$, $n$ its tasks and $\ell = 5.5/46 = 0.12$ the expected loss per task
   of a replaced attempt 2 from P2.8's rows (2.0 of 23 task units in `P2.6-01`, 3.5 of 23 in
   `P1.1-01`; an upper bound, as if the shot were wrong on every such output); the in-request forcing stays when the cut tasks'
   answered share is at least $13/15$, at least one turn ended at the budget, and the budget
   check passed; without a cut turn the row is not measured and the separate turn stays.
5. Write `merge.csv`, `merge.md`, and `reading.json`.

## 3. Why it is built this way

- **Replay on stored attempts, not NVARC beside the shot.** Considered: run NVARC on the 15 in
  the same session. Lost: about 30 minutes more of 4x L4, and batched DFS is nondeterministic, so
  NVARC's confidences would move and the sample, drawn from `P2.6-01`'s, would no longer be its
  buckets.
- **Correctness from the logs, no solutions file.** Considered: score the merged file with
  `arc26.evaluate` against the solutions. Lost: the solutions stay on Kaggle (dev scoring only,
  BRIEF A0); every grid here was already scored where it was made, so the result is the same.
- **The run's own choice of $S_1, S_2$.** Considered: recompute them from the transcripts. Lost:
  the records hold the parsed answers and the metrics their shapes; the same function then
  makes the same choice.
- **The attempt-2 loss from the CSVs, not the wave.** Considered: the replay's own losses. Lost:
  a 6-task sample may hold no output right by NVARC's attempt 2 alone (the uniform draw holds
  none), so the wave cannot see the loss the extension risks; two runs' rows can.
- **Thresholds as constants with flags.** Considered: a YAML config. Lost: the thresholds are the
  brief's pre-registration, fixed before the run; a flag can only make a sensitivity reading,
  which the experiment record would have to call one.
- **Validated on `P6.3-05`.** The test replays `P6.3-05`'s logs (when pulled) with `MERGE`'s
  right flags and gets its planned 4.5 of 15 against NVARC's 0, 2.5 in $(0, 0.5)$ and 2.0 at 0.

## 4. How to run it

```bash
.venv/bin/python scripts/merge_replay.py --run-logs runs/P6.9-01/run_logs \
    --nvarc-submission ~/wt-P2.6/runs/P2.6-01/submission.json \
    --nvarc-csv reports/P2.8/P2.6-01.csv --out reports/P6.9-01
```

Seconds on a laptop, under 100 MB of memory; nothing to attach.

## 5. Outputs

| File | Contents |
|---|---|
| `merge.csv` | one row per test output: bucket, confidence, $N_1, N_2$ real and right, the shot's answers and their rights, $S_1 = N_1$, the merged pair, merged right |
| `merge.md` | the bucket table (NVARC, shot alone, merged, net, gained, lost) and the reading |
| `reading.json` | the same as data: `buckets`, `tasks`, `reading` (`fault`, `harness`, `extend`, `forcing`, `lenient`) |

## 6. How to read the results

- **Good:** no Fault; the harness passes; `extend.decision` and `forcing.decision` are the
  configuration of Submission 4 (`selection.mid_bucket`, `request.forcing_in_request` in
  `configs/agent_shot_P6.9.yaml`), checked by `P6.9-02`.
- **Borderline:** a decision row near its threshold (a middle-bucket net of 0.5 or 1.0; a forced
  rate within one output of $13/15$): the row decides as written; the experiment record says how
  little separates it.
- **Bad:** the harness reads 0 of 5 (investigate the prompt and the quantization before any
  decision), or a Fault: the decision rows are not read.

## 7. Failure modes

| Symptom | First thing to change |
|---|---|
| `refused: ... has no per-output flags` | The run's config lacked `lenient_answers` and no turn was budget-forced, so `shot_metrics.json` has no `right`: rerun the metrics from the pulled records, never guess the flags. |
| `refused: ... outputs, NVARC's files disagree` | The wrong NVARC run or CSV: use `P2.6-01`'s pair. |
| `refused: ... not in NVARC's submission` | A shot task outside DEV120: the replay needs NVARC's attempts for every shot task. |
