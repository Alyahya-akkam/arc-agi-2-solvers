# H-submit-01: version 1 training with H initialization, submitted as version 19

## Outcome

CPU and GPU readiness passed. The owner authorized the build, its validation and the
competition submission. Kaggle receipt 57886280 confirms `alyahya737/arc26-submission`
version 19 submitted 1 Nov 2026 at 23:12:40 UTC; status PENDING, public and private scores
empty at confirmation. This proves acceptance, not scoring success. No counted-final
selections were changed.

Code 99e1d61e09b105cc4b9740779b04d922b00a63c6; GPU notebook SHA256
f1e378a9c5de7c06203406314f0ba55766c05b56d6706a883f656a19d15e5972; wheel
e5c1cf5ac8f40a228bfd6687f1915ff8688b76e3acce9c43ad4ca03521647700. The private code dataset
arc26-code-h-submit is separate from version 18. The H checkpoint remains on Kaggle, private
dataset arc26-p3-9-01 version 2, best.pt, SHA256
a8809762dc852f76a44b67353cb8f3b35182ac1a9231766d848bf68b39d58ca6.

## What actually ran

Frozen FOMAML-trained H at step 4260 emits rank-16 attention factors from demonstrations only.
Its bf16-rounded factors form a float32 delta, merged into 144 original base projections.
The ordinary version 1 rank-256, nine-target LoRA then trains and decodes; exact CPU snapshots
restore the base in `finally`. Both development and competition route through the H entry.
The scored rerun uses all hidden tasks and original production deadlines, not the two-task
development subset. No reasoning model, judge, refinement loop or H-disabled fallback.

Independent review APPROVE; new H, builder and runner tests 54 passed, integration 554 passed
2 skipped, packaging and parity 322 passed 2 skipped; Ruff clean. GPU and CPU builds each
rebuilt byte-identically. The watcher received an independent APPROVE and 9 network-free
simulated API tests, records intent before submission, never retries an uncertain request,
and verified the actual receipt.

## CPU twin

arc26-h-submit-cpu version 1: complete, 253.2 s notebook. Offline install and all stages passed.
Actual checkpoint hash matched; step 4260, 134958208 finite parameters in 40 tensors,
36 layers, rank 16, alpha 32, q/k/v/o scope, reference base identity. Final placeholder file
valid on all 120 evaluation tasks / 172 outputs. The CPU twin deliberately runs no model solver.

## GPU check

Version 19 on 4xL4: COMPLETE; notebook 940.1 s, solver 815.6 s, install 22.7 s, checkpoint
preflight 79.8 s. All four workers loaded the pinned H and ended normally. Solver exit 0,
notebook problems [], no fatal H errors or OOM found in pulled logs. The training allocator
reported 13,594 MiB on both active workers; no continuous device-memory peak was measured by
this build. Final file valid, 120 tasks / 172 outputs, no repairs. The other 118 tasks are
deliberate placeholders in this short development run; the competition rerun loads every
hidden task.

| Task | H emission seconds | Merge seconds | Restore seconds | Changed base elements | Effective delta norm | Task seconds | Used/decoded/restored |
|---|---:|---:|---:|---:|---:|---:|---|
| 0934a4d8 | 7.9813 | 0.9248 | 0.8948 | 906263315 | 31.8163 | 529.5436 | all true |
| aa4ec2a5 | 7.6544 | 1.0063 | 0.8925 | 912242097 | 37.2598 | 694.6939 | all true |

Each task merged 144 projections using scale 8, bf16 factors and float32 multiplication and
addition. Snapshots held 1,887,436,800 bytes on CPU per active worker, then restored bit for
bit. Nonzero effective weight changes rule out a nominal adapter whose delta entirely rounded
away.

## Accuracy interpretation

Development score 1.0 / 2 task credit and 1 / 2 outputs: 0934a4d8 wrong, aa4ec2a5 correct.
This is a two-task mechanical validation, not evidence of a gain over version 1 or a
leaderboard estimate. There was no paired baseline rerun. Prior rank-16 H merge validation
tied SGD at 10.5 / 40; this build instead retains the original rank-256 training. The owner
requested the experimental entry despite the absence of an established lift. Competition
score remains pending.

## Durable evidence

Pulled logs runs/H-submit-01 and runs/H-submit-01-cpu; development score
runs/H-submit-01/dev_score.json. The original workspace runs/codex-handoff holds
H-submit-v1.bundle, check_h_submit.py, watch_h_submit.py, CPU and GPU readiness JSONs,
h-submit-command.log, h-submit-attempt.json and h-submit-watch-status.json. The watcher exited
0 after the confirmed receipt; never duplicate the request.
