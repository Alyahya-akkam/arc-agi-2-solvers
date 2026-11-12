# Pre-run brief: <run-id>

Copy to `docs/prerun/<run-id>.md` (for example `P3.5-01.md`). Fill every section, or write
"n/a:" plus the reason. Every number is **measured** (give the command or file it came from) or
**estimate** (give the derivation). Sections 1 and 9 are written before the run and never edited
afterwards. Rules: BRIEF A9, ADR-008.

```
Run-id:           <task-id>-<nn>
Revision:         1
Task:             P?.? <title>
Title:            <one line: what this run does>
Kind:             training | evaluation | ablation | smoke | throughput | cache
Decides:          <gate or decision this run feeds>
Code-commit:      <full SHA of the code and config being run>
Config:           configs/<name>.yaml
Provider:         kaggle
Kaggle notebook:  <owner>/<slug>
Accelerator:      <Kaggle ID: NvidiaL4 = 4xL4, NvidiaL4X1 = 1xL4, NvidiaTeslaT4 = 2xT4>
GPU:              <Modal only: a type of modal_job.GPU_RATES_USD_S, such as H100!>
GPU count:        <Modal only: GPUs in the training container, the script's world size>
Max attempts:     <Modal only: GPU attempts allowed, preemption restarts included, 1 to 3>
Timeout:          <hard stop in seconds, from the wall-time kill criterion (see below)>
Internet:         off
GPU hours:        <expected, a number (Modal: GPUs x hours)>
Quota left after: <hours left this week after this run, a number (Modal: n/a)>
Cost expected USD: <Modal only: expected dollars at modal.com/pricing, a number>
Cost cap USD:     <Modal only: at least the job's worst case at every hard stop, a number>
Output dataset:   <Modal only: alyahya737/<slug>, the private dataset the job writes; none for an evaluation job>
Previous run:     <run-id or none>
```

`Revision` goes up by one every time the brief changes after it was sent; a decision applies only
to the revision and code commit it was made on. `scripts/kaggle_ops.py push` reads only this
block, refuses a repeated key, and enforces `Code-commit`, `Kaggle notebook` (the notebook's
`id`), `Accelerator` (exactly one Kaggle ID), and `Timeout` (an integer); see
[README.md](README.md).

A Kaggle brief deletes the five "Modal only" lines. A Modal brief (ADR-022) sets
`Provider: modal`, deletes `Kaggle notebook` and `Accelerator`, and sets `Internet: on (Kaggle
reads and writes inside the job)`; its `Timeout` (60 to 86400) is the GPU stage's hard stop per
attempt and its total over all attempts. Only `scripts/modal_ops.py run` launches it (`kaggle_ops.py
push` refuses any `Provider` but `kaggle`), enforcing `Code-commit`, `GPU`, `GPU count`,
`Timeout`, `Max attempts`, and `Output dataset` against the dist folder's `job.json`, and refusing
a job whose worst case exceeds `Cost cap USD`. Section 6 of a Modal brief gives the cost in
dollars at the listed rate instead of quota hours. A Modal evaluation job (`kind: eval`, task X2)
sets `Output dataset: none`; its `Timeout` and `Max attempts` apply to each of its steps, and
section 2 says how the evaluation solutions stay inside it (ADR-022, decision 9).

## 1. Question and decision rule

- The question this run answers, in one sentence.
- The gate or decision it feeds (G0 to G5, an ADR overturn criterion, or a default in a config).
- Pre-registered outcomes, with the metric, subset, and task count:

| Result | Threshold | What happens next |
|---|---|---|
| Good | | |
| Borderline | | |
| Bad | | |

- Noise floor for this comparison (for example, 2 tasks on DEV40, one fixed seed).

## 2. Data

- **Sources and versions:** each dataset or generator, its version, commit, or Kaggle dataset
  version, and how it reaches the notebook (attached input, generated in-notebook, internet).
- **Counts:** puzzles, pairs, augmentations, episodes (or evaluation tasks), per source.
- **Splits:** how train, validation, and evaluation are separated (by task ID), sizes of each.
- **Leakage check:** result of the ARC-AGI-2 evaluation hash check (P3.4), with the count of
  matches found and removed. Evaluation solutions read? Only for dev scoring, yes or no.
- **Filtering:** length cap in tokens, how many items it drops and from which source, any other
  filter and its effect.
- **Length distribution:** token-length histogram or percentiles (p50, p90, p99, max).
- **Augmentations:** which transforms, how many per item, applied consistently across demos and
  query (yes or no).
- **Examples:** two or three rendered items (demos and query grids) so the data can be eyeballed.
- **Known quirks:** duplicates, near-duplicates, class or source imbalance, anything odd.

## 3. Objective

- **Loss** in LaTeX, with every symbol defined.
- Which tokens contribute (output-grid tokens only, teacher forcing) and how the loss is
  normalized (per token, per episode, per puzzle).
- Regularizers and auxiliary terms (L1 or L2 on generated factors, KL term) with weights, and
  whether each is on in this run.
- What is frozen and what receives gradients.
- For evaluation runs: the metric (competition metric via `arc26.evaluate`), attempts, subset.

## 4. Architecture

- Base model, its precision, and what is frozen.
- Adapter: rank $r$, $\alpha$, scaling rule, target modules, layers, emitted values per adapter.
- Trained network (H or other): inputs, trunk widths and depths, heads, initialization (zero-init
  $B$ head?), what the output is at step 0.
- Parameter table:

| Component | Parameters | Trainable | Precision |
|---|---|---|---|

- Encoder settings (layer $L$, pooling mask), cache used or recomputed.
- Diff against the ADRs: any departure from ADR-001 to ADR-007 and why.

## 5. Optimization

| Setting | Value | Why this value (ADR, paper, measurement) |
|---|---|---|
| Optimizer | | |
| Learning rate, schedule, warmup, minimum lr | | |
| Batch size and gradient accumulation (effective batch) | | |
| Steps or epochs | | |
| Gradient clipping | | |
| Weight decay and exclusions | | |
| Precision (bf16, TF32, fp32 parts) | | |
| Activation checkpointing (reentrant or not) | | |
| Attention kernel | | |
| Seed(s) | | |

## 6. Compute and budget

- Accelerator and why (and whether timings transfer to the scored 4xL4 rerun).
- **Measured** step time and peak memory from the smoke test or P3.6 at the relevant sequence
  lengths; expected wall time and its derivation.
- Memory headroom at the worst-case sequence length.
- Checkpoint cadence, where checkpoints live on Kaggle, and the resume command.
- Kaggle session limit versus expected wall time: how many sessions, how they chain.
- Quota: GPU hours this run, hours used this week so far, hours left for scheduled runs (A7).

## 7. Monitoring and kill criteria

- What is logged and how often (loss, validation episode CE, grad norm, lr, step time, memory).
- What healthy curves look like, and what memorization or divergence would look like.
- Kill criteria, applied without asking: NaN or inf; loss above <x> after <n> steps; validation
  CE not below the zero-adapter baseline by step <n>; step time over <s>; projected wall time over
  <h>. On a kill: stop, pull logs, report, and wait for an amended, re-approved brief.
- Enforcement: nothing can cancel a running kernel (the Kaggle CLI has no cancel command), so the
  script checks every kill criterion itself and exits when one fires. `Timeout` is the outer
  bound: Kaggle ends the session after that many seconds. Set it a little above the wall-time
  criterion, so the script's own stop, which writes the logs, comes first.

## 8. Baselines and comparisons

- What the result is compared against (zero adapter, SGD adapter at the same scope, previous
  run), with their numbers, subsets, and task counts, or "to be measured in this run".

## 9. Pre-registered expectations

- Expected result with a range, and the reasoning. Written before the run, never edited after.

## 10. Changes since the previous run

- Config diff and code diff (commit range), each with its reason. "First run" if none.

## 11. Smoke-test evidence

- The CPU `--smoke` command, its commit, and its result (passed, steps, loss moved).
- Tests that cover this script and their status (`pytest -m "not gpu"`).

## 12. Outputs

| File | Size (estimate) | Stays on Kaggle or pulled to `runs/<task-id>/` |
|---|---|---|

Only allowlisted small files are pulled (`kaggle_ops.py pull`); weights and checkpoints stay on
Kaggle and are attached to later notebooks as inputs.

## 13. Risks and unknowns

- What could make this run wasted, how likely, and what guards it.
- Unverified assumptions this run depends on.

## 14. Approval

Written by the human, either directly or through the approval page, whose decision
`scripts/prerun_sync.py record` copies here verbatim with its time, revision, and commit. The
agent never approves a brief itself. This block holds the brief's only approval line: a brief
with one anywhere else is refused.

```
Approved-by:
Notes:
```
