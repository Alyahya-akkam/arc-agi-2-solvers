# Runbook: original version 1 solver with H initialization

## 1. Why this script exists

The owner requests a competition submission using H, followed by the original rank-256 training.
This is distinct from development-only H validation arms. Existing public anchors stay intact;
the new candidate is experimental and does not imply an accuracy improvement.

## 2. What it does, step by step

Build the existing reference notebook and wheel from a clean commit, add the pinned H dataset,
then inject a mandatory post-install checkpoint preflight and explicit H runner fields. The CPU
preflight checks SHA256 before deserializing, base identity, H and encoder architecture, all
parameter names and shapes, and finite values. In either mode, each worker loads the original
rank-256, nine-target solver plus H, emits H from demonstration pairs on the frozen unmerged
base, rounds H factors to bf16, merges their FP32 product into attention weights, runs the
original training and decode, and restores exact CPU snapshots in `finally`. The cache is
scratch under /tmp, not an output artifact.

## 3. Why it is built this way

H emits rank-16 attention adapters, incompatible with direct replacement of v1's rank-256
adapter. Merging changes the base function while preserving v1's original training scope.
Every intended projection must be present. Reject zero effective bf16 change and nonfinite
values. Exact copy restoration prevents one task's inferred adapter from leaking into another.
Both competition and development share the H route; a preflight failure cannot silently run v1.

## 4. How to run it

Build from the isolated clone after a clean commit:

```bash
python scripts/build_h_submission.py --tag H-submit-01 --config configs/h_submission_base.yaml
python scripts/build_h_submission.py --tag H-submit-01-cpu --config configs/h_submission_cpu.yaml
python scripts/kaggle_ops.py dataset-create dist/H-submit-01/arc26-code
python scripts/kaggle_ops.py push dist/H-submit-01-cpu --timeout 4500
python scripts/kaggle_ops.py status alyahya737/arc26-h-submit-cpu
python scripts/kaggle_ops.py pull alyahya737/arc26-h-submit-cpu --task-id H-submit-01-cpu
```

After the actual CPU checks pass, the owner's recorded authorization in H-submit-01 applies:

```bash
python scripts/kaggle_ops.py push dist/H-submit-01 --run-id H-submit-01
python scripts/kaggle_ops.py status alyahya737/arc26-submission
python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id H-submit-01
```

Do not duplicate a queued push. Capture the exact successful notebook version from the push.
After GPU readiness passes, submit that version with the gateway's competition-submit command,
with explicit competition, kernel, version, file and message. Query competition-submissions to
confirm an actual receipt. The CLI can return 0 after some HTTP errors; exit 0 alone is not
acceptance. Never retry an uncertain submission blindly. Do not change counted final selections.

## 5. Outputs

Pinned private code dataset arc26-code-h-submit; the existing private checkpoint dataset
arc26-p3-9-01 version 2 stays on Kaggle. The final submission.json uses the ordinary ARC format.
Small reports: run_logs/h_checkpoint.json, stages.json, h-start/h-rankN.jsonl and standard
candidate and timing logs. Large base weights and H checkpoints are never pulled locally.

## 6. How to read the results

CPU: successful offline install, h_checkpoint.ok=true and exact SHA256, step 4260, rank 16,
alpha 32, valid final file, no H preflight exception. CPU has run_solver=false and does not
establish GPU memory or accuracy. GPU: both requested development tasks must have
decoded=true, h_used=true, merge.layers 144, positive effective_delta_norm and
changed_elements, restored=true; no fatal worker or OOM, and solver exit 0. Inspect actual
predictions, not just valid placeholders. Report scores on those two tasks without inferring a
leaderboard gain. The competition rerun uses all hidden tasks and original production
deadlines; its subset is never the development subset.

## 7. Failure modes

A wrong or missing input or checkpoint pin stops before inference. H and model
incompatibility, an unsafe merge, zero effective change, nonfinite weights or failed
restoration stop the solver; there is no H-disabled fallback. Any failure blocks submission
until fixed and revalidated. GPU memory of rank-256 plus H is measured only by the actual
short run. API submission quota, queue or receipt failures require inspection; never report
success without a confirmed receipt.
