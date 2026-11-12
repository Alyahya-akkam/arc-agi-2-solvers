# ADR-022: Training jobs on Modal through one gateway, data moved between Kaggle and Modal only

- Status: Accepted, 2026-10-18 (task X1): run `X1-01` read Pass (`docs/experiments/X1-01.md`):
  start-step validation CE bitwise equal to `MT-02`'s, paired step ratio 0.984, the private output
  dataset at version 1, about 1.55 USD (estimate). Proposed 2026-10-17. Amends ADR-008.
- Tasks: X1; the meta-training sessions after `MT-02` (`MT-04` or its warm-started replacement),
  `P3.3-01`, and new H objectives. Amended 18 Oct by task X2 (decisions 8 to 12, development
  evaluations): Proposed until run `X2-01` reads Pass for a kind; the human accepted the option
  (9b of `docs/research/beyond_h_options.md` on `task/ideas`) on 18 Oct. Amended 26 Oct by task
  P4.8 (decision 13, the `judge` step kind): Proposed until run `P4.8-01` runs its step.

## Context

Kaggle's weekly GPU quota cannot be raised, and H's meta-training needs about 15.5 h of 4xL4 for
the 4,572 steps after `MT-02` (`docs/experiments/MT-02.md`), more than the quota leaves beside the
scheduled evaluation runs. On 17 Oct the human offered external GPUs and chose Modal on the
orchestrator's recommendation: the agent drives it from the shell through the Python SDK, with no
SSH, and Modal holds the credentials as secrets. The human ran `modal token new` (workspace
`alyahyakkam`), creates the Modal secret `kaggle` (`KAGGLE_USERNAME`, `KAGGLE_KEY`), and sets a
workspace spending limit. BRIEF A0 still holds: no GPU work and no large artifact on the local
machine.

## Decision

1. **Scope.** Modal runs training jobs only: meta-training sessions, pilots, and new H
   objectives. Timed and scored runs (P3.7's comparisons, the 12 h checks, the final run) stay on
   Kaggle's 4xL4, the scored rerun's hardware, because their timings must transfer.
2. **One gateway.** The agent reaches Modal only through `scripts/modal_ops.py`, as it reaches
   Kaggle only through `scripts/kaggle_ops.py`: `run <dist> --run-id <id>` (with `--dry-run`),
   `status`, `logs` (allowlisted small files only), and `stop`. There is no command that deploys a
   persistent app, writes to a volume from the local machine, or reads, prints, or creates a
   secret. `scripts/modal_app.py` defines the app and is imported only by the gateway and by
   Modal's containers; `tests/test_repo_hygiene.py` holds both rules.
3. **Approval.** `run` needs an approved pre-run brief (BRIEF A9) with the approval rules of
   ADR-008 (one `Approved-by:` line in the Approval block, its `revision N` stamp equal to
   `Revision`, a clean tree, nothing outside `docs/` changed since `Code-commit`), plus
   `Provider: modal`, `GPU`, `GPU count`, `Timeout`, `Max attempts`, `Cost cap USD`, and
   `Output dataset` in the header, each equal to the dist folder's `job.json`. `kaggle_ops.py push`
   refuses any `Provider` but `kaggle`.
4. **Cost bound in code.** Every function is single-use, with Modal's smallest
   `scaledown_window` ($W = 2$ s) and an explicit `startup_timeout` ($S_g = 600$ s for the GPU
   image of about 7 GB, $S_c = 120$ s for the CPU image), and passes its cores and memory as
   `(request, limit)` with the limit equal to the request (Modal bills the higher of request and
   use; a single value sets no limit), so each container bills its requested cores and GiB for at
   most $S + T + W$. The job's worst case,
   $$C_\max = A\,(S_g + T + W)\,\rho_\text{gpu} + (S_c + T_f + W)\,\rho_f
     + (S_c + T_p + W)\,\rho_p + (S_c + T_d + W)\,\rho_d,$$
   must not exceed `Cost cap USD`, where $A$ is `Max attempts`, $T$ the GPU stage's `Timeout`
   (Modal's per-attempt execution timeout), $T_f$, $T_p$, $T_d$ the fetch, publish, and driver
   timeouts, and $\rho$ each container's price per second at Modal's list prices
   ($\rho_\text{gpu} = n\,g + c\,\kappa + m\,\mu$ for $n$ GPUs at $g$, $c$ cores at $\kappa$, $m$ GiB
   at $\mu$; fetch, publish, and the driver are nonpreemptible, so never restarted, at 3 times for
   CPU and memory). Inside the job, a ledger on the volume also caps the GPU stage's seconds
   summed over attempts at $T$, so a preemption restart continues within the same budget. Outside
   the bound: GPU restarts that open no attempt (a preemption during the container's start, during
   the stale wait of at most 90 s, or after the last attempt), at most
   $(S_g + 90 + W)\,\rho_\text{gpu}$ each and not limited in number by the code; the workspace
   spending limit backstops them. The bound holds only at the listed prices, so each brief names
   the rate per GPU-second it assumes and leaves at least 15% headroom under its cap (`X1-01`,
   revision 2 on 4 x L4: 6.00 USD under 7.1, 15.4%).
5. **Data path.** Inputs come from Kaggle inside the job: a CPU `fetch` stage with the `kaggle`
   secret downloads each input pinned to its version with kagglehub 0.3.12 into the volume
   (`fetch.json` records every file's sha256). Outputs go back the same way: a CPU `publish` stage
   uploads the run folder (checkpoints, logs, `MANIFEST.json` with every sha256) as a new version
   of the brief's private `Output dataset`, so Kaggle notebooks attach it as an input. The GPU
   stage holds no secret. Only the small allowlisted files (`modal_ops.py logs`) reach
   `runs/<task-id>/`.
6. **Same training code and stack.** The GPU stage runs `scripts/train_hypernet.py` with the
   arguments a Kaggle session passes, in an image pinned to `MT-02`'s Kaggle stack (Python 3.11,
   torch 2.8.0+cu128, transformers 4.55.4, peft 0.17.1, flash-attn 2.8.2, the same wheel file),
   with the job's arc26 wheel, at the world size of the checkpoint it continues (`RESUME_KEYS`
   fixes `gpus`; `--init-from` checks it too).
7. **Reproducible folder.** `scripts/build_modal_job.py` builds `dist/<run-id>/` from a clean
   `HEAD`, the job config at `configs/modal_<run-id>.yaml`. Before launching, the gateway checks
   that the folder holds exactly the files `job.json` lists, each with its sha256; `job.json` is
   regenerated from `configs/modal_<run-id>.yaml`, the training script, and its config as
   `git show <Code-commit>` gives them (`build_modal_job.job_from_commit`, the builder's own code)
   and must equal the folder's byte for byte, so every argument, input, volume, resource, and
   timeout is the approved commit's; and the wheel holds exactly the commit's `arc26/` files, byte
   for byte, plus `METADATA`, `WHEEL`, and `RECORD` of `arc26-<version>.dist-info` and nothing else,
   with `METADATA` naming the commit's project and version, `WHEEL` declaring a pure-Python wheel,
   and `RECORD` listing every other member with its size and sha256. This closes ADR-008's "dist is untracked" gap for
   Modal runs.

## Amendment 1 (task X2, 18 Oct): paired development evaluations

Context: Kaggle's weekly quota is the binding constraint on the evaluation runs (P3.7's arms,
the shot's waves), and those runs compare arms with each other on one platform, so their timings
need not transfer to the scored rerun; the 12 h checks of a submission candidate do. The human
accepted moving the paired development evaluations to Modal on 18 Oct.

8. **Scope widened.** Modal may also run paired development evaluations: the arms comparisons on
   DEV4, DEV40, or DEV120 (`arc26.arms`, task P3.7) and the shot's measurement waves
   (`arc26.agent.shot`, task P6.5b). Still Kaggle only: the 12 h timing checks of a submission
   candidate, anything submitted, and any timing claim about the scored rerun. A step time
   measured on Modal compares arms within its run; it is never quoted as the rerun's.
9. **The evaluation solutions.** They reach Modal only inside an evaluation job (`kind: eval`),
   in its fetch container and its GPU steps, and never leave Modal except as scores (BRIEF A0's
   dev-mode-only rule):
   - an evaluation job's volume starts with `arc26-modal-eval`, a training job's may not, and a
     `competition` input is refused in a training job (`modal_job.parse_job`), so no training
     container ever mounts them;
   - an evaluation job publishes nothing: no publish stage, `output_dataset` null, the brief's
     `Output dataset: none`, checked equal by the gateway;
   - the GPU step runs with Modal's `block_network`, `restrict_modal_access` (no other Modal
     resource), and no secret (the Kaggle twins ran with the internet off too); its final volume
     commit raises when refused (Modal's exit commit is the same in-container call, so there is
     no independent fallback), every step returns its allowlisted numbers in its function
     result for the driver to write and commit, and a step whose commit was refused or whose
     numbers did not come back stops the run;
   - `modal_ops.py logs` fails closed: it pulls only the files `EVAL_PULL_PATTERN` names, each
     written by known code and holding numbers, ids, and settings (scores, timings, the
     records, the shot's run, plan, boot, and engine-call records), never a file that can hold
     an answer grid, since a correct attempt is a solution: no submission, candidates, attempts,
     replies, transcripts, competition file, or free-text log of third-party code (the
     commands' output, `engine.log`, `install.log`);
   - the shot step's entry is pinned to `arc26.agent.shot`: `arc26.agent.run`, for one, writes
     every reply to `transcripts.jsonl`, so no other module may be launched as a step.
   P3.4's hash check is unaffected: it guards H's training data, whose path does not change, and
   an evaluation job trains no H.
10. **Two step kinds, one gateway.** An evaluation job runs one or more GPU steps through the same
    `run`, `status`, `logs`, `stop`, approval rules, dist-folder checks, ledger, and cost bound:
    `arms`, the arms entry through `arc26.runner.run_solver` with a `RunConfig` from the job
    config, as the Kaggle notebook's Run cell starts it; `shot`, the entry `arc26.agent.shot`
    named by the step (the only one allowed, decision 9) with a run config read by name from the
    commit and changed only by dotted `overrides`
    that must exist in it, after the vLLM wheelhouse is installed into a venv by the installer
    it ships, as the notebook's agent stage does. Neither forks code: the builder checks each
    step against the commit's own `RunConfig` and the entry's `config_from_mapping`, imported
    from the commit's exported tree in a fresh interpreter (asserted). An attempt stopped by a
    preemption or a cancel closes as `interrupted`, in the phase it was in; one killed hard is
    marked `lost` by the restart that refuses it.
11. **The image is the Kaggle runs'.** The GPU steps run on Kaggle's GPU image by the digest that
    `P3.7-03`'s, `P6.5-02`'s, and `P6.5-03`'s `env.json` name (their `pip_freeze.txt` files are
    equal but for the arc26 line), with tensorflow removed and the wheel installed as the
    notebook's install cell does. The unsloth stack cannot be reproduced from pins alone: it is
    the output of the install notebook `sorokin/pip-install-unsloth-flash-patch` (version 2), a
    folder of installed packages in which that notebook patched unsloth after pip installed it.
    The closest faithful route is that output itself: fetched at version 2 by kagglehub, checked
    on the CPU (the 13 distributions of `P3.7-03`'s `env.json` at their versions, an unsloth
    file that differs from unsloth's own `RECORD`, the patch, and every file under a `bin`
    folder executable, its bits restored first since the archive kept none), linked at
    Kaggle's mount path,
    and put on `PYTHONPATH` in Kaggle's order. If kagglehub's download of it lacks the packages,
    the fallback is a CPU Kaggle notebook that copies its mount into its own output, fetched the
    same way.
12. **Cost bound.** With $n$ steps, $A$ attempts per step, and the Kaggle image's start bound
    $S_e = 1200$ s:
    $$C_\max = n\,A\,(S_e + T + W)\,\rho_\text{gpu} + (S_c + T_f + W)\,\rho_f
      + (S_c + T_d + W)\,\rho_d,$$
    $T$ the per-step `Timeout` (each step's ledger caps its attempts' sum at it), no publish
    term. `X2-01`: $2 \times 4{,}802 \times 0.00118728 + 1.46 + 0.26 = 13.12$ USD under a cap of
    15.5 (15.4%).

Overturn criterion of the amendment: `X2-01` reads Fault for a kind, and that kind stays on
Kaggle until fixed and re-approved; any evaluation solution found outside an evaluation volume,
in a dataset, or in pulled logs stops evaluation jobs on Modal until the gateway is fixed and the
human re-approves.

## Amendment 2 (task X2, 19 Oct): the numbers travel through the volume

Measured in `X2-01` attempt 6 (app `ap-s3JU85hOLHwzcHoIimPRuH`): the evaluate container's
volume commit works under `restrict_modal_access` (the step's final `step.json`, written just
before the commit, is on the volume), while its function result cannot leave the container when
it exceeds Modal's inline limit for a spawned call ($\min(2\ \text{MiB}, 8\ \text{KiB})$ in
Modal 1.6.2): Modal then uploads it as a blob, which the restricted container may not do
(`AuthError: Received :status = '401'` from `BlobCreate`, for a result of about 30 KB holding the
returned files' texts). The driver spawns every stage since `P3.9`'s `await_stage`. Decision:
the isolation stays (no other Modal resource for a container that reads the evaluation
solutions); a step's result names its allowlisted files by size and sha256 only
(`returned_digests`) and stays under 6 KiB (`compact_result`; the whole result is in
`step.json`); the driver reloads the volume and checks each file against its digest
(`verify_returned`); a step whose commit was refused or whose record is not on the volume with
its digest stops the run, as before. A refused commit now loses that step's numbers instead of
carrying them in the result; it reads Fault (platform: commit refused), as pre-registered.

## Amendment 3 (task P4.8, 26 Oct): the judge step kind

Context: run `P4.8-01` measures Qwen3.8-27B as a judge of NVARC's candidates (attempt 2 among
ranks 2 to 8) on the engine of version 8, and the no-thinking ranker after it
(`docs/tasks/P4.8.md`). It needs the shot step's install, venv, deadlines, and sampling, but a
module that is not the shot; decision 9 pins a shot step's entry to `arc26.agent.shot`.

13. **A third step kind, `judge`.** Its keys, checks, plan (`plan_shot`), install, and stage are a
    shot step's; its entry is pinned to `arc26.agent.judge` (`modal_job.STEP_ENTRIES`; a stand-in
    only in the smoke), as decision 9 pins the shot's, and a shot step still may not run the
    judge. `arc26.agent.judge` writes its replies only to `judge_transcripts.jsonl`, never pulled;
    `EVAL_PULL_PATTERN` adds `judgments.jsonl`, `ranker.jsonl`, `judge_plan.json`,
    `boot_ranker.json`, and `timing-ranker.jsonl` under `logs/agent/`, each written by that
    module with ranks, numbers, and ids only (the smoke checks that no pulled file holds a
    candidate grid in any form). The judge never opens the solutions: an evaluation job of
    judge steps alone takes no `--solutions`, and its correctness is scored locally after the
    pull from the NVARC run's own `ALL_CORRECT` marks (`scripts/judge_report.py`). The cost bound
    (decision 12) and every other rule are unchanged.

Alternatives: a judge mode inside `arc26.agent.shot` lost (version 8's submitted module would
carry new branches and config keys); allowing any entry for a shot step lost (decision 9's
guarantee). Overturn criterion: a pulled judge file that holds a grid or a reply stops judge
steps until the pattern is fixed and the human re-approves.

## Amendment 4 (task P5.2m, 29 Oct): probe jobs

Context: version 11 (`P5.2-01`) first meets a GPU on Sat 31 Oct, when the Kaggle quota resets; a
fault found then costs Saturday. The human asked for checks on Modal ("couldn't we do checks in
Modal?", 29 Oct; `docs/tasks/P5.2m.md`).

14. **A third job kind, `probe`** (`modal_job.parse_probe_job`, stages `scripts/modal_probe.py`):
    a Kaggle notebook's own code cells run on Modal GPUs on a fake `/kaggle` whose input folders
    link to the job's volume, the run config the notebook's with overrides by dotted key only
    (dev mode stated), its code from `Code-commit` in the dist folder. Its inputs are pinned
    Kaggle outputs and datasets (a dataset the notebook attaches unpinned is fetched at its
    latest, recorded), single challenge files of the competition (never a solutions file, so no
    probe volume holds solutions and none may be an evaluation volume), and Hugging Face models
    at a 40-hex commit with `expect` digests. Fetched files are verified once and kept in a
    marker, so a relaunch hashes nothing. The GPU container has the network blocked and no
    secret; nothing is published. The cost bound is decision 12's without the publish term,
    plus one GPU restart after a preemption (which refuses at once with one attempt). A probe's
    numbers are Modal's: the Kaggle check stays the version's reference, and nothing timed or
    scored moves to Modal.

Alternatives: waiting for the Kaggle check lost (no time to fix before the submission); an
evaluation step kind lost (evaluation volumes hold the solutions and need the install notebook's
`kaggle_lib`, which the notebook does not use); a Modal fork of `fn_probe.py` lost (it would test
other code than Saturday's). Overturn criterion: a probe whose readings disagree with its Kaggle
check on a pass or fail fact (boot, install, memory row) stops probe jobs until the cause is known.

## Alternatives considered

- **Keep Kaggle only.** Lost: the quota cannot be raised, and the rest of MT alone needs about
  15.5 h of 4xL4 (`MT-02` measured 11.42 s per step), beside the evaluation runs the schedule
  needs.
- **Rented machines with SSH (Lambda, RunPod, vast.ai).** Lost: the agent's shell has no working
  SSH key (`CLAUDE.md`), a machine bills while idle, and data would cross the local machine or need
  a separate transfer path; Modal bills per second of a container and keeps credentials as
  secrets.
- **Modal's CLI (`modal run --detach`) instead of the SDK in the gateway.** Lost: status, logs,
  and cancellation would mean parsing CLI output, and the launch would rely on the CLI's import of
  the app in another Python. The gateway uses the SDK (pinned `modal==1.6.2`) and the CLI only for
  `modal app stop`, which ends every container of the app.
- **Mount the inputs through the local machine (`modal volume put`).** Lost: about 15 GB over the
  human's slow connection and disk, against BRIEF A0.
- **kaggle-cli inside the job instead of kagglehub.** Lost: kaggle-cli 2.2.4's `kernels output`
  always downloads the latest session (it drops the version of `owner/slug/N`) and holds each file
  in memory; kagglehub 0.3.12 downloads a notebook output at a given version, or one file of it,
  and creates private datasets.
- **Checkpoints copied from local container disk to the volume by the stage.** Lost: Modal Volumes
  commit in the background every few seconds and on shutdown (modal.com/docs/guide/volumes), and
  their own training example writes checkpoints straight to a volume; the script's atomic writes
  then suffice, and a copy loop would be new code to get wrong.
- **One function for the whole job.** Lost: downloads and uploads would run on GPU time (about
  1 USD per hour on CPU against 4.0 on 4 x L4 or 17.6 on 4 x H100), and the Kaggle secret would sit in the
  training container.
- **Retries for the GPU stage.** Lost: Modal restarts a preempted function on the same input
  anyway (modal.com/docs/guide/preemption); a retry of a crash would only spend the budget again.
  The ledger's `Max attempts` bounds the restarts.

Of the amendment (task X2):

- **Keep every evaluation on Kaggle.** Lost: the quota is spent on comparisons whose timings
  need not transfer, while the 12 h checks need it (the human's choice, 18 Oct).
- **The unsloth stack rebuilt from PyPI pins (X1's image route).** Lost: pins give unsloth
  2025.9.7 without the install notebook's patch of its batched decode
  (`docs/research/kaggle_first_runs.md`, section 1.4); re-applying the patch would fork a third
  party's notebook into the repository.
- **A slim image with the pins and the install notebook's folder.** Lost: the solver's launch
  line names `/usr/local/cuda/bin/ptxas` (`arc26.runner.LAUNCH_ENV`), Triton compiles its
  launchers with the image's C compiler, and the shot's venv is made with the image's
  `/usr/bin/python3.11`; Kaggle's image has all three as the twins had them.
- **A CPU Kaggle notebook that re-exports the install notebook's mount.** Kept as the fallback
  (decision 11): it costs a Kaggle session and a brief, and kagglehub may serve the output as is.
- **A copy of the shot's config with the cap changed.** Lost: task P6.9 is changing the shot;
  a copy would drift from its schema, while an override of a missing key fails at build.
- **Two briefs and two jobs for the pilot.** Lost: one job shares one fetch, one brief, and
  one cap; its steps fail independently.
- **Evaluation outputs published as a dataset, and every log pulled as Kaggle pulls do.** Lost:
  the attempts and candidates hold solutions whenever they are right; they stay on the
  evaluation volume, and only scores and numbers leave.
- **The network open in the GPU step.** Lost: the container holds the solutions, and the
  Kaggle twins ran offline.
- **Two attempts per evaluation step.** Lost for the pilot: an evaluation cannot continue after
  a preemption (the arms entry clears its folders, the shot boots again), so a restart would
  measure something else at twice the bound.

## Consequences

- MT's remaining sessions can run on Modal without Kaggle quota, at a cost in dollars the brief
  states and the gateway bounds; `X1-01` measures the step time and the price per step first.
- Pre-run briefs gain provider fields (`docs/prerun/TEMPLATE.md`); `prerun_sync.py export` and the
  email carry the cost in dollars. The approval page shows the new fields only in the brief text
  until the page itself is updated.
- A Kaggle notebook continues a Modal run by attaching its output dataset
  (`/kaggle/input/datasets/alyahya737/<slug>/checkpoints`, `--resume` there) at the same world
  size.
- Runbook `docs/runbooks/modal_ops.md`; tests `tests/test_modal_ops.py`.
- Amendment 1: evaluation runs on Modal need only dollars; their briefs say which Kaggle run each
  step pairs with. The first is `X2-01` (`docs/prerun/X2-01.md`), whose `Code-commit` must hold
  task P3.7's arms entry and task P6.5b's shot; tests `tests/test_modal_eval.py`.

## Overturn criterion

`X1-01` reads Fault (numerics differ by more than 2% at the start step, a stage fails, or the
dataset does not appear), or any Modal spend without an approved brief, or any secret in a log:
Modal use stops until the gateway is fixed and the human re-approves this ADR.
