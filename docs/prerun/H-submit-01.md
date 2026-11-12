# Pre-run brief: H-submit-01

```
Run-id:           H-submit-01
Revision:         1
Task:             Original version 1 solver with demonstration-conditioned H merge
Title:            H adapter merged before reference rank-256 test-time training
Kind:             evaluation
Decides:          readiness of the owner's requested experimental H competition submission
Code-commit:      99e1d61e09b105cc4b9740779b04d922b00a63c6
Config:           configs/h_submission_base.yaml
Provider:         kaggle
Kaggle notebook:  alyahya737/arc26-submission
Accelerator:      NvidiaL4
Timeout:          4500
Internet:         off
GPU hours:        at most 2.5 quota h at 2x for the development check
Quota left after: about 23h46m estimated; last dashboard report 28h28m preceded P5.8-01
Previous run:     P3.9-02 H merge validation; v1 reference solver
```

## 1. Question and decision rule

Does the concrete build actually use H on demonstrations, merge all intended attention weights,
run the original rank-256, nine-target training and decoding, restore base weights exactly,
and produce valid outputs? Require the CPU twin's offline install, pinned checkpoint check and
meaningful CPU tests to pass. Require both development tasks to finish with nonzero H delta,
all expected layers merged, exact restoration, finite outputs, no H fallback or worker error,
and a valid final submission. Record development accuracy; no accuracy gain is required by
the owner's express experimental submission request. Do not submit a placeholder-only or
H-disabled build. If a readiness check fails, fix and rebuild rather than silently submit NVARC.

## 2. Data

Development only: evaluation tasks 0934a4d8 and aa4ec2a5. Competition: every hidden test task,
sorted as in v1, never evaluation solutions. H reads demonstration input/output pairs only.
No training data, solutions, model weights or checkpoint download to the local machine.
The checkpoint stays on Kaggle as private dataset alyahya737/arc26-p3-9-01 version 2, best.pt,
SHA256 a8809762dc852f76a44b67353cb8f3b35182ac1a9231766d848bf68b39d58ca6.

## 3. Objective

Competition inference, with v1's ordinary per-puzzle one-epoch LoRA training. H is frozen.
No H retraining, selection on hidden solutions, refinement loop, judge or reasoning model.

## 4. Architecture

Original ARC-tuned Qwen3-4B, bf16, one worker per L4. Use the parity-validated split solver
implementation of v1's training and decode. H emits rank-16 attention factors at alpha 32.
Merge their delta into the original base attention weights, then train a fresh rank-256 LoRA
on v1's nine targets. Do not replace rank 256 with rank 16 or load incompatible factors into
it. Copy only H-targeted base weights to CPU before writing and restore bit for bit in
`finally`. H emits on the unmerged frozen base; trained per-puzzle LoRAs are disabled during
embedding.

## 5. Optimization

Preserve v1's learning rate 5e-5, AdamW, cosine schedule, one epoch over augmented examples,
rank 256, alpha 32, original targets, train seed 42, hash seed 0, decoding views and rescoring.
The H checkpoint is the already validated FOMAML step 4260; it is not claimed better than SGD.

## 6. Compute and budget

Kaggle 4xL4 only; no local GPU. Development timeout 4500 s, at most 2.5 quota h at 2x,
expected about 20 to 45 minutes plus queue. No fallback accelerator. Notify after 30 minutes
queued; no duplicate push. The CPU twin uses no GPU quota. The scored rerun can consume about
22 to 24 quota h; the estimated remaining 26h16m before this development run is historical
and not a refreshed quota. The owner explicitly requested submission despite possible late
scoring. Verified deadline 2 Nov 23:59 UTC, 3 Nov 02:59 Riyadh. Submit only before that
cutoff; do not promise completion in time.

## 7. Monitoring and kill criteria

Keep standard bounded solver process groups, original competition soft and hard deadlines,
development soft 2400 s and hard 3300 s, and the final write reserve. Inspect all worker H
logs, layer counts, delta norms, finite factor checks, checkpoint hash, restore records,
timing and GPU memory. Any missing or mismatched checkpoint, H failure, un-restored weight,
OOM, no completed H task, or invalid final output blocks submission until fixed. No silent
H-disabled fallback.

## 8. Baselines and comparisons

Public reference 30.97 and v6 30.28 remain scored anchors. This is an explicit H experiment,
not evidence to displace either final pick. Prior DEV40: H 5/40, SGD 10.5/40, H-fit 9.5/40,
H-merge-fit 10.5/40; that merge used rank-16 training, so it does not validate rank-256
memory or quality.

## 9. Pre-registered expectations

No established accuracy lift. H inference and CPU copies add cost; base perturbation can
help or hurt. Checkpoint scoping and actual H usage must be proven. Two development puzzles
establish plumbing only and cannot estimate hidden-score improvement.

## 10. Changes since previous run

New explicit H submission entry, opt-in runner fields, deterministic dedicated notebook
builder, checkpoint attachment, new immutable private code dataset, and gateway submission
commands for this owner-authorized action. Existing scored versions and datasets unchanged.
The builder must inject H in both development and competition, unlike old validation arms.

## 11. Smoke-test evidence

Independent review APPROVE at 99e1d61. New H, builder and runner tests 54 passed, integration
554 passed 2 skipped, earlier runner 107 passed; Ruff and whitespace clean. GPU and CPU each
rebuilt byte-identically, including manifests. GPU notebook SHA256
f1e378a9c5de7c06203406314f0ba55766c05b56d6706a883f656a19d15e5972; CPU notebook
abf41a6607d7a1a705897cc5e3e1fbf8a369444a5da1e966975457a971c3ec98; wheel
e5c1cf5ac8f40a228bfd6687f1915ff8688b76e3acce9c43ad4ca03521647700. Private code dataset
created; CPU twin arc26-h-submit-cpu version 1 COMPLETE, ready=true: offline install and all
stages passed; actual H checkpoint SHA256 matches, step 4260, 134958208 finite parameters,
40 tensors, rank 16, alpha 32, 36 layers, qkvo, 253.2 s notebook, valid 120-task,
172-output placeholder file. This CPU no-solver run does not establish GPU solving; logs
runs/H-submit-01-cpu. Additional packaging and parity checks 322 passed 2 skipped. GPU
validation still pending. No submission until the actual CPU and GPU readiness checks pass.

## 12. Outputs

GPU: next private version of alyahya737/arc26-submission. CPU: own private slug
alyahya737/arc26-h-submit-cpu. Code dataset alyahya737/arc26-code-h-submit, separate from v18.
Pull only allowlisted small logs. Competition output submission.json, the exact pinned
notebook version submitted by the project gateway under the owner's explicit instruction.

## 13. Risks and unknowns

The original rank-256 training plus H and merge buffers has not run on 4xL4. CPU snapshots
reduce extra GPU memory, but do not prove peak safety. The H checkpoint is trained for the
original base encoder; base restoration before subsequent embeddings is mandatory. Kaggle
submission quota and queue, and successful submit acceptance, remain unverified until the
gateway read and response.

## 14. Approval

Owner explicitly instructed 1 Nov 2026: "build v1 with H and let's submit it" and then
"no submit it yourself no need for my approval with this one...". This authorizes the H
development validation and exact checked competition submission without another permission
question, overriding the repository's manual-submit rule for this task. Fill Code-commit and
record this existing authorization after the build is concrete; do not broaden it to unrelated
GPU experiments or change the owner's final selections.

```
Approved-by: Owner, explicit chat authorization 1 Nov 2026, revision 1: submit it yourself, no need for my approval with this one.
Notes: Applies to this requested v1 plus H build and its validation/submission; readiness checks remain mandatory.
```
