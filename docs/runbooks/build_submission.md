# Runbook: `scripts/build_submission.py` and the submission notebook

Task P0.5 (BRIEF A5, B5 P0.5). Scripts: `scripts/build_submission.py` (the build folder) and
`scripts/build_kaggle_notebook.py` (the notebook, called by the first). Configs:
`configs/submission_P0.5-01.yaml` (DEV4 smoke on 4xL4, run `P0.5-01`),
`configs/submission_P1.1-01.yaml` (DEV120 run on 4xL4, run `P1.1-01`, the Submission 1 candidate),
`configs/submission_P0.3-01.yaml` (DEV4 on 4xL4 with one worker per GPU, the timing run
`P0.3-01`, task P0.3b), `configs/submission_P1.3-01.yaml` (the DEV40 LoRA scope series on 4xL4,
run `P1.3-01`, task P1.3, series mode below), `configs/submission_P2.0-01.yaml` (DEV4 on 4xL4
with the split solver, run `P2.0-01`, task P2.0, the solver choice below),
`configs/submission_P2.1-01.yaml` (the held-out gate's price on DEV40 on 4xL4, one hold-out arm,
run `P2.1-01`, task P2.1, hold-out arms below), `configs/submission_P2.3-01.yaml` (one refinement
round on DEV4 on 4xL4, one round arm, run `P2.3-01`, task P2.3, round arms below),
`configs/submission_P2.5-01.yaml` and `configs/submission_P2.5-02.yaml` (one round on DEV40 on
4xL4 at rank 16 on q, k, v, o, the in-context and the gradient round forced, one notebook each,
runs `P2.5-01` and `P2.5-02`, task P2.5, variants below), `configs/submission_P2.6-01.yaml`
(Submission 2: DEV120 on 4xL4 at rank 16 on q, k, v, o in both modes, run `P2.6-01`, task P2.6,
scope below), `configs/P3.7-01.yaml` and `configs/P3.7-02.yaml` (H's validation on DEV40 and GEN,
four adapter arms per puzzle, runs `P3.7-01` and `P3.7-02`, task P3.7, adapter arms below),
`configs/submission_P6.5-hybrid.yaml` (the hybrid: Submission 2's NVARC for its
share of the session, then the agent solver, on 4xL4, a submission build, task P6.5, hybrid build
below), `configs/submission_P6.5-01.yaml` (the same at a sixth of its times on DEV40, the
measurement run `P6.5-01` on its own notebook, never submitted),
`configs/submission_P6.5-agent.yaml` (the agent alone; NVARC runs only as its fallback, in the
time an agent stage that ends early leaves, so an agent that runs to its end but verifies little
leaves no NVARC attempt: risks in the hybrid build below), `configs/submission_P6.5-cpu.yaml`
(the hybrid's Kaggle CPU check, task `P6.5-cpu`), `configs/submission_P6.5-03.yaml` (Submission
3's candidate: Submission 2's NVARC with its deadlines, then the long shot in the time it leaves,
on DEV240 at the competition's times, the 12 h timing check `P6.5-03`, task P6.5b, shot build
below), `configs/submission_P6.5-02.yaml` (the same on DEV4 with the shot's end at development
scale, the GPU smoke `P6.5-02` on its own notebook), `configs/submission_P6.5-02-cpu.yaml` (the
smoke's Kaggle CPU twin, task `P6.5-02-cpu`),
`configs/submission_cpu_check.yaml` (Kaggle CPU check with a launch probe, task `P0.5-cpu`),
`configs/script_smoke.yaml` (Kaggle CPU check of script mode, task P0.5d, script mode below), and
`configs/script_P3.5-01.yaml` (H's toy meta-training on 4xL4 with two P3.4 notebook outputs
attached, run `P3.5-01`, task P3.5, script mode below), and `configs/script_P3.5-02.yaml` (the
same with half the steps, run `P3.5-02`).
Decisions: ADR-008 (Kaggle
only through `scripts/kaggle_ops.py`), ADR-010 (only the `arc26` wheel is uploaded), ADR-011 (the
split solver, Accepted 2 Oct: the default since `P2.0-01` passed), ADR-013 (what a held-out pair
may reach), ADR-015 (the round mechanics), ADR-016 (the in-context round, the length guard),
ADR-018 (Submission 2 without the loop: the build's scope in both modes), and ADR-020 (the agent
solver: the hybrid build's agent stage).
Validator:
`arc26/submit.py` (`docs/modules/submit.md`). Steps for the human: `docs/SUBMITTING.md`. "Run A" is
run A of `P0.1-01` (`docs/experiments/P0.1.md`).

## Why this script exists

A0 and A5 ask for a submittable Kaggle notebook built locally at every milestone, reproducible from
a commit, that runs offline and always writes a valid `/kaggle/working/submission.json`, even when
the solver crashes or times out. This script builds that notebook and everything Kaggle needs next
to it. Its first GPU use, run `P0.5-01`, is also P0.1's GPU parity check: the moved code $E$ passes
when $g(E, A) \ge 0.4$ against `tests/fixtures/p0_1_reference_dev4.json`, on run A's image, install
notebook, accelerator, and `PYTHONHASHSEED`. Everything that can fail without a GPU is meant to
fail first, locally (`tests/test_notebook_local.py`) or in the Kaggle CPU check. The results feed
gate G0 ("DEV4 smoke of the built notebook") and every submission. Task P1.3 adds series mode:
one development session trains several LoRA scopes on the same tasks, one solver launch per arm,
for gate G2 (BRIEF B2 Q2: H's scope is the smallest within 2 points of the notebook
configuration on DEV40); and a production guard that marks every build whose competition rerun
would differ from production as not for submission. Task P2.0 adds the solver choice:
`solver: split` runs the reference's statements split into `arc26.ttt` and `arc26.decode`
(`python -m arc26.solver`, ADR-011), which run `P2.0-01` checks against the same parity rule.
Task P2.1 adds hold-out arms: an arm may hold one demonstration pair per puzzle out of training
and prompts and run the held-out gate (`docs/modules/loop.md`), so run `P2.1-01` can measure the
gate's price on DEV40 before the refinement loop (P2.3) is built on it. Task P2.3 adds round
arms: a hold-out arm may run the refinement rounds after each puzzle's gate, with the governor's
config embedded in the build, so run `P2.3-01` can run one round end to end on DEV4.
Task P2.6 adds the build's scope: one LoRA scope for the single run that a competition rerun
trains too, so Submission 2 can run rank 16 on q, k, v, o (ADR-018) and its development run
`P2.6-01` measures exactly what the rerun will run.
Task P6.5 adds the hybrid build (ADR-020): one notebook, `alyahya737/arc26-submission`, that runs
NVARC for a configured share of the 12 h and then the agent solver (Qwen3.8-27B writing programs
that must reproduce every demonstration) on what NVARC leaves, NVARC's attempts the fallback at
every moment; the same builder still builds the pure NVARC build (Submission 2) and builds an
agent-only one, so the final selections come from one tested path. Run `P6.5-01` measures the
hybrid on DEV40 at the rerun's time per task (the session scaled by 40/240) against `P2.6-01`'s
DEV40 subset score and the agent spike's numbers.
Task P6.5b adds the shot build (Submission 3): run `P6.3-05` solved 4 of the 15 DEV40 tasks NVARC
is least sure of, against NVARC's 0, and Submission 2's NVARC leaves about 3 of the 12 h idle on
240 tasks ($u = 0.74$, `P2.6-01`). The key `agent.entry: arc26.agent.shot` runs that shot after
NVARC, with NVARC's deadlines unchanged, in the time NVARC leaves; `subset: DEV240` (DEV120 and a
copy of each task) lets one development run measure the 12 h timing of a rerun of that size.
Task P0.5d adds script mode: a config with `kind: script` builds a notebook that runs one script
of `scripts/` (BRIEF A8) on the same stack and plumbing, never submitted, for the Phase 3 GPU runs
(`P3.1-01`, `P3.2-01`, `P3.6`); and the bf16 stop: a build made for an accelerator stops before
the install when Kaggle gives it GPUs without bf16, as it did in `P0.3-01`.

## What it does, step by step

### Build (local, CPU, about a second)

1. **Clean tree.** `git status --porcelain --untracked-files=all` must print nothing, so the build
   is exactly `HEAD`.
2. **Export.** `git archive HEAD` is extracted into a temporary folder. The config and
   `reference/kernel-metadata.json` are read from that export, so both are the committed versions.
3. **Config.** `parse_config` in `scripts/build_kaggle_notebook.py` refuses a `TODO` anywhere, a
   missing or unknown key, an owner other than the gateway's `KAGGLE_OWNER`, an accelerator ID the
   gateway does not accept, an accelerator that can never pass the bf16 stop (`NvidiaTeslaT4`,
   compute capability 7.5, and the TPU IDs, task P0.5d), more workers than the accelerator has
   GPUs (4xL4: 4, 1xL4: 1),
   and development deadlines outside $600 \le s_\text{end} \le s_\text{kill} \le 43200 - 300$
   seconds. A config without an accelerator may still run the solver: that is the launch probe.
   The optional `arms` (series mode) must be a non-empty list of mappings with exactly `name`
   (1 to 40 lowercase letters, digits, `-`; distinct), `lora_r` (a positive integer),
   `lora_targets` (distinct names of the reference's nine, in its order, at least one of the
   seven projections), `reason` (one line, at most 200 characters), and optionally `holdout`
   (task P2.1: null, or exactly `rule` (`last` or `first`), `k2`, and `in_test_prompts`, both
   true or false; an arm with one needs `solver: split`), the first arm the
   notebook configuration ($r = 256$ on all nine) or, from task P2.5, an arm that holds a pair
   out. The optional `solver` (task P2.0) must be
   `nvarc` or `split`; a config without it runs `split`, the default since 2 Oct (ADR-011), and
   the configs of the runs before the flip state `nvarc`, the moved reference. The optional
   `scope` (task P2.6) must be null or exactly `lora_r` and `lora_targets`, under an arm's rules
   and with an arm's messages; not the reference's own scope ($r = 256$ on all nine: leave the
   key out), and never beside `arms`. The optional `agent` (task P6.5) must be null or exactly
   the hybrid's keys, each checked by `parse_agent` (hybrid build below), never beside `arms`;
   `resolve_agent` then reads the agent's run config from the export and refuses a plan that
   could not start the agent stage or end before Kaggle's limit. The optional `kind`
   (task P0.5d) is `submission`, the default, or `script`; a script config has exactly `kind`, the
   seven keys every config shares
   (`kaggle_notebook`, `code_dataset`, `competition`, `accelerator`, `reference_metadata`,
   `kernel_source_versions`, `model_slug`), `script` (a `.py` under `scripts/`), `script_config`
   (a `.yaml` under `configs/`), `args` (at most 32 one-line strings, none of which sets
   `--config`: no `--config`, no prefix of it such as `--conf`, no `-c` in any form, alone or
   clustered with other short flags such as `-vc`), and `script_kill_s`
   ($300 \le k \le 43200 - 300$ seconds), and its notebook is never
   `alyahya737/arc26-submission`. A script config may add `notebook_inputs` (task P3.5): at most
   8 outputs of other notebooks, `<owner>/<slug>: <version>` with a version of at least 1, none
   of them a key of `kernel_source_versions` or the config's own `kaggle_notebook` (Kaggle
   refuses a notebook's own output, `invalidKernelSources`), each passed in `args` as
   `{input:<owner>/<slug>}`, and no placeholder in `args` naming another.
4. **Metadata.** `kernel-metadata.json` gets the reference's pinned image, model source, and
   install notebook at version 2 through `build_reference_run.kernel_metadata` (the function that
   built run A's), plus the config's identity, accelerator, and code dataset. A script config's
   `notebook_inputs` follow the install notebook in `kernel_sources`, each as
   `<owner>/<slug>/<version>`; one that is a kernel source of the reference is refused. A hybrid
   build's two download outputs (the weights and the engine wheelhouse, task P6.3) follow the
   same way. The gateway's own checks run on the result before anything is written.
5. **Wheel.** Built from the export with `SOURCE_DATE_EPOCH` set to the time of the last commit
   that changed one of its inputs (`arc26/`, `pyproject.toml`, `README.md`, which the wheel's
   `METADATA` embeds, and `.gitignore`, which hatchling applies): `uv build --wheel --offline`,
   then online, then `pip wheel --no-deps`.
6. **Notebook.** Generated from the config and the wheel (next section), then every code cell is
   compiled. A script build reads its script and the script's config from the same export,
   refuses a script that does not compile or a config that is not a YAML mapping or holds
   `TODO`, and adds both, byte for byte, to `arc26-code/` beside the wheel, under their file
   names (kaggle-cli drops subfolders of a dataset upload).
7. **Manifest.** `MANIFEST.md` lists the commit, the wheel builder and backend, every file with its
   size and sha256, the commands that put each file on Kaggle, the config with its arms and its
   solver, and
   whether the build is fit to submit: only for the submission notebook
   `alyahya737/arc26-submission` with the production settings (four workers on `NvidiaL4`, no
   series, valid competition deadlines); otherwise "no, not for submission" with every reason,
   which the notebook's first cell repeats. A script build says "no, a script notebook is never
   submitted" and lists the script, its config, `args`, `notebook_inputs`, and `script_kill_s`,
   and names each attached notebook output with its placeholder.
8. **Write.** All files go to a new folder beside `dist/<tag>/`, which then replaces it. A refused
   build writes nothing, and no stale file survives a rebuild.

### On Kaggle (the notebook, top to bottom)

Let $t_0$ be the time the Config cell runs (`NOTEBOOK_START`).

1. **Config**, the only cell a human edits: `SUBSET`, `WORKERS`, `HASH_SEED`, `SOLVER`
   (`'nvarc'` or `'split'`), in a scoped build `SCOPE` (task P2.6, below), `RUN_SOLVER`,
   `SOLVER_END_S` ($s_\text{end}$), `SOLVER_KILL_S` ($s_\text{kill}$), `ARMS` (None, or the
   series' arms as dicts), and `BUILD` (commit, config and its sha256, wheel name and sha256).
2. **Helpers.** Mode from `KAGGLE_IS_COMPETITION_RERUN`, paths under `ARC26_KAGGLE_ROOT` (default
   `/kaggle`), `SERIES` (the arms in a development run, None in a competition rerun),
   `SERIES_MARGIN`, `arm_dirs` (each arm's folders), stage timing, and six helpers copied from
   the builder: file search, command runner, GPU query, the bf16 check `bf16_refusal`, file hash,
   and the series stop rule `series_refusal`.
3. **Submission validator.** `arc26/submit.py`, copied verbatim from the wheel.
4. **Environment check.** Prints the mode, the build, both interpreters, the GPUs from `nvidia-smi`
   (with compute capability; 8.0 and up means bf16 and FlashAttention 2), and package versions from
   package metadata. Searches the input root, at most 8 levels deep and skipping package trees, for
   the challenges file (preferring the copy under the competition's folder), a `config.json` under
   a folder named after the model, and the wheel. In development it also needs the evaluation
   solutions next to the challenges (the reference scores DEV4 with them). A build made for an
   accelerator needs one GPU per worker whenever the solver runs; a build without one needs none.
   A build made for an accelerator also needs bf16 on every visible GPU, compute capability 8.0
   or higher from `nvidia-smi --query-gpu=compute_cap` (a missing or non-numeric value counts as
   below), and at least one visible GPU (`bf16_refusal`, task P0.5d); a CPU build is not checked.
   Only lines of `nvidia-smi`'s output with one value per queried field count as GPUs, so a
   warning it prints on stdout does not.
   It prints the deadlines in effect, the solver's code (`solver code: <SOLVER>`), the series'
   arms, and writes an all-`[[0]]` placeholder `submission.json`.
5. **Install.** `python -m pip uninstall -y tensorflow`; the wheel's sha256 against `BUILD`;
   `python -m pip install --no-index --no-deps --force-reinstall --disable-pip-version-check
   <wheel>`; then `python -c "import arc26; print(arc26.__file__)"` and `import arc26` in the
   kernel must give the same file. `python` is the one on PATH throughout.
6. **Run.** `arc26.runner.run_solver(cfg, end_time, python, timeout)` with
   $\text{end\_time} = t_0 + s_\text{end}$ and $\text{timeout} = t_0 + s_\text{kill} - t_\text{now}$.
   A development run follows the Config cell (`RUN_SOLVER`, `SUBSET`, the config's $s_\text{end}$
   and $s_\text{kill}$). A competition rerun always runs the solver, on every test task, with
   $s_\text{end} = 43200 - 1800$ and $s_\text{kill} = 43200 - 1200$. The solver is the moved
   reference code (`SOLVER = 'nvarc'`, `python -m arc26.nvarc.starter`) or, with
   `SOLVER = 'split'`, the same statements split into `arc26.ttt` and `arc26.decode`
   (`python -m arc26.solver`), in both modes, one worker per GPU; both take the same arguments.
   The cell first sets `PYTHONHASHSEED = HASH_SEED` in the
   kernel, as run A's first cell did, so `env.json` records it as run A's does; `run_solver` also
   passes it to the solver explicitly, and the run record in `stages.json` (`launch_env`) lists
   every variable the solver's environment adds: the reference's `UNSLOTH_DISABLE_STATISTICS`,
   `TRITON_PTXAS_PATH`, and `OMP_NUM_THREADS`, `PYTHONHASHSEED`, and `PYTHONUNBUFFERED`.
   Candidate, marker, and timing files of an earlier run in the same session are removed
   first. In development mode the solver gets `timing_dir = /kaggle/working/run_logs/timing`
   and each worker appends its timing records there (`timing-rank<rank>.jsonl`: the model
   load, then per puzzle the training, each decode batch, each rescored candidate, and the
   puzzle; `docs/modules/timing.md`); a competition rerun passes null and records nothing.
   After `end_time` the workers start no new task and stop decoding (the reference's own rule);
   at the hard stop `run_solver` kills the process group. A failure or a non-zero exit is
   printed and recorded, not raised. In series mode this step repeats per arm (below).
7. **Write.** `make_submission` (the reference's selection) writes a staging file,
   `/kaggle/arc26_staging/submission.json`, inside `try`. If it fails before writing anything
   (a candidate pickle cut mid-write by the hard stop makes the reference's
   `load_decoded_results` fail for every task), the candidate files that do not load are moved to
   `/kaggle/arc26_quarantine/` and selection runs once more. In `finally`, `finalize_submission`
   keeps every valid attempt, replaces the rest with `[[0]]`, writes `submission.json`
   atomically, and validates it. In development mode a selection that returns normally is one
   `main_selection` record in `run_logs/timing/timing-main.jsonl`.
8. **Run logs**, development only: `dump_run_logs` from `scripts/build_reference_run.py`, verbatim
   (`candidates.json`, `pip_freeze.txt`, `env.json`, as run A recorded them), and `stages.json`.
   The timing records are already in `run_logs/timing/`; `dump_run_logs` reads only
   `inference_outputs/`, so it neither copies nor removes them.

Development runs stop at the first failed check of steps 4 and 5; a competition rerun logs it and
carries on, so it always ends with a valid file.

A hybrid build (task P6.5) adds two stages: the agent check after step 4 (4b), and the agent stage
after step 7 (7b); step 6 then ends NVARC at its share of the session when the agent stage will
follow (hybrid build below).

### Series mode (task P1.3)

A config with `arms` runs a series in a development run. Arm $i$ is a LoRA scope: rank
$r_i$ = `lora_r` on the modules `lora_targets`, scaled by rsLoRA's $\alpha / \sqrt{r_i}$ with the
notebook's $\alpha = 32$, so a targeted projection computes
$W x + \frac{32}{\sqrt{r_i}} B A x$ with $A \in \mathbb{R}^{r_i \times d_\text{in}}$ and
$B \in \mathbb{R}^{d_\text{out} \times r_i}$; `embed_tokens` and `lm_head`, when listed, are
trained as full copies. Learning rate, schedule, and steps are the notebook's in every arm.

1. **Run** (step 6), once per arm in the listed order. Each launch gets
   `RunConfig(lora_r=r_i, lora_targets=...)`, its own candidate folder
   `/kaggle/inference_outputs/<name>/`, and its own timing folder
   `/kaggle/working/run_logs/<name>/timing/`; the worker markers are shared and removed before
   every launch. Every arm has the same $s_\text{end}$ and $s_\text{kill}$, counted from $t_0$.
   Before arm $i > 1$ the stop rule (`series_refusal`) checks, with $t$ the seconds since $t_0$,
   $d_j$ the wall time of arm $j$ from its launch to the solver's exit, and
   $m$ = `SERIES_MARGIN` $= 1.25$:
   $$t + m \max_{j < i} d_j \le s_\text{end}.$$
   If it fails, or if an earlier arm's solver failed (an exception or a non-zero exit), arm $i$
   and every later arm are recorded as `not started` with the reason. The first arm always
   starts. With `RUN_SOLVER = False` every arm is `skipped`, and the next steps still run.
2. **Write** (step 7), after the last solver: every arm that ran or was skipped is selected from
   its own folder by `make_submission`, with what it prints (the per-arm `ALL_CORRECT` lines,
   the `acc:` lines of both scorers, and `*** Reload score:`) also written to
   `run_logs/<name>/selection.log`, its `main_selection` record in `run_logs/<name>/timing/`,
   and its own repaired, validated submission in `run_logs/<name>/submission.json`. The first
   arm, which the builder requires to be the notebook configuration, writes the staging file
   behind `/kaggle/working/submission.json`.
3. **Run logs** (step 8): `dump_run_logs` per arm into `run_logs/<name>/` (`candidates.json`,
   `env.json`, `pip_freeze.txt`) and the arm's `stages.json`; the top-level `stages.json` holds
   every arm's record under `stages.run.arms`, its selection under `stages.write.arms`, and the
   arms under `settings.arms`.

A competition rerun ignores `ARMS`: one launch with `RunConfig`'s defaults (the reference's LoRA
settings) in the single run's folders, no timing, as for every other build. A config without
`arms` runs as before series mode: one launch with the reference's LoRA settings in the single
run's folders. A series build and a single-run build differ only in their Config cells
(`tests/test_build_submission.py`), so the series code that a single run skips is the same code
the local test runs for both.

### Hold-out arms (task P2.1)

An arm with `holdout` runs the split solver with the held-out gate (`docs/modules/loop.md`,
ADR-013). For a puzzle with $k$ real pairs, the rule picks one, the last by default, when
$k \ge 3$, or $k = 2$ with `k2`; the arm then trains on the other $k - 1$ pairs, decodes the test
inputs with them (or with all $k$ when `in_test_prompts`), and after each puzzle's test decode
decodes the held-out input under the same 16 views, votes with the reference's selection, and
logs whether the voted answer equals the held-out output. A puzzle the rule keeps whole runs as
the reference.

1. **Run** (step 6): the Run cell passes the arm's settings to `RunConfig.holdout`, with two
   folders of the arm, `/kaggle/gate_outputs/<name>/` (the gate's candidate files, one subfolder
   per puzzle, beside `inference_outputs/` and never in the arm's candidate folder, so no gate
   file reaches a submission) and `/kaggle/working/run_logs/<name>/gate/` (the verdicts,
   `gate-rank<rank>.jsonl`, one JSON line per puzzle). Files left in either by an earlier launch
   are removed first. The session log shows `[Rank r] gate for <task>: decoding the held-out
   pair, index i of k real pairs`, the gate's `decoding` and `scoring` lines, then `passed` or
   `failed` with the candidate count and the held-out output's place in the vote.
2. **Timing**: each puzzle's records gain `gate_decode` (per DFS batch of the held-out input),
   `gate_selection` (per rescored candidate), and `gate` (the whole check), inside the `puzzle`
   span; `train`, `decode`, and `selection` stay the test inputs'.
3. **Write and Run logs** (steps 7 and 8): as for any arm; the gate's verdicts are already in
   `run_logs/<name>/gate/`, and `stages.json` holds the arm's settings under `holdout`.

A competition rerun ignores the series and with it every hold-out arm. The environment check
prints each arm's hold-out, and the first cell and `MANIFEST.md` state it beside the arm's
scope.

### Round arms (task P2.3)

A hold-out arm with `rounds` runs the refinement rounds after each puzzle's gate
(`docs/modules/loop.md`, ADR-015): at most `max_rounds` (1 or 2), each priced by the governor
before it runs; a round promotes test inputs whose voted answer has agreement at least `tau` and
survives a re-decode under fresh views, continues the adapter `continue_steps` steps from a CPU
snapshot with them, and keeps the candidate adapter when it passes the gate, or restores the
snapshot. After the rounds the test inputs that are not pseudo-pairs are decoded again and take
those candidates. With `final_in_prompts: true`, the production setting and `arc26.loop.Rounds`'
default, that is one final decode with the current adapter and the held-out pair back in its
prompts, whatever the rounds' outcome: run `P2.1-02` recovered 2.0 DEV40 tasks with the pair in
the test prompts (7.0 against 5.0 of 40), so ADR-013 as amended keeps it in the decode that is
submitted and out of every decode that feeds a promotion, and each round's plan prices that
decode. With `false`, the decode after a kept round, without the pair, is the one submitted
(`P2.1-01`'s prompts). All five settings are spelled out in the config;
`governor` names a config file of the repository, `configs/governor.yaml`, which the builder reads
from the built commit and embeds (the wheel carries `arc26/` only).

1. **Build**: `parse_rounds` checks the settings with `arc26.loop.Rounds`' rules and needs the
   arm's `holdout` with `in_test_prompts: false`; `resolve_rounds` reads the governor config
   (`scripts/build_submission.py` from the export, the command line from the checkout), checks it
   (`parse_governor`: a mapping with `ops`, `margin`, `candidates_per_batch`, `deadline`, the
   `reference` decode policy, JSON values only), and embeds it in the Config cell's arm;
   `MANIFEST.md` names the file and its sha256.
2. **Run** (step 6): the Run cell passes the round settings inside the arm's hold-out settings,
   with the arm's round folder `/kaggle/round_outputs/<name>/` (one subfolder per puzzle:
   `round<r>-redecode<i>/`, `round<r>-gate/`, `round<r>/`, `final/`, and `round0/`, round 0's
   candidates once replaced). Round files and the round records of an earlier launch are removed
   first. The session log shows per round `[Rank r] round 1 for <task>: plan gradient, estimated
   ... s, ... s left`, `continuing the adapter of <task> for 32 steps`, a line per test input with
   its promotion reason, the round's outcome (`kept`, `rolled_back, weights restored`,
   `not_promoted`, `declined`, `nothing_to_promote`), and the decode after it (`test decode final
   for <task> (held_out prompts): ...`, or `skipped, all_promoted`, `not_restored`, or
   `no_time`).
3. **Records**, beside the gate's verdicts in `run_logs/<name>/gate/`: `plan-rank<rank>.jsonl`,
   `promote-rank<rank>.jsonl` (every voted answer's free checks, a declined round's too),
   `round-rank<rank>.jsonl` (with the weight fingerprints), `decode-rank<rank>.jsonl`. **Timing**: `redecode_decode`, `redecode_selection`, `redecode`,
   `snapshot`, `fingerprint`, `candidate_train`, `gate_decode`, `gate_selection`, `gate`,
   `restore`, `round`, `round_decode`, `round_selection`, `final_decode`, `final_selection`, all
   inside the `puzzle` span.

A competition rerun ignores the series and with it every round arm.

**Variants** (task P2.5, ADR-016). A round arm may add `variants`, the round variants it offers:
`[gradient]` (`arc26.loop.Rounds`' default, the round above), `[in_context]` (the in-context
round: the promoted pairs join the prompts, the adapter is not fit, and a failed gate rolls back
the pairs alone), or `[gradient, in_context]` (the governor's ladder: the gradient round when it
fits the time left, else the in-context one), in that order. Absent, the arm offers the gradient
round and builds as in P2.3; `[gradient]` spelled out changes nothing the solver sees (the solver
JSON leaves the default out). `parse_rounds` refuses another value; `rounds_text` states the
variant in the environment check, the first cell, and `MANIFEST.md`, and the Config cell explains
the key. In every round arm the loop keeps each prompt that carries pseudo-pairs within the
model's limit (`max_seq_length - max_new_tokens`): a decode leaves out the pseudo-pairs its
prompts cannot hold, the latest first, never a real pair, and writes a line to
`run_logs/<name>/gate/context-rank<rank>.jsonl` (the Run cell removes those of an earlier
launch too). The session log shows `[Rank r] round 1 for <task>: plan in_context, ...` for an
in-context round, no `continuing the adapter` line, and `the prompts of <decode> left out the
pseudo-pairs of test inputs [...]` for a decode the guard trimmed.

### Scope (task P2.6)

A config without `arms` may set `scope`, the LoRA scope of its single run (ADR-018): rank
$r$ = `lora_r` on the modules `lora_targets`, scaled by rsLoRA's $\alpha / \sqrt{r}$ with the
notebook's $\alpha = 32$, as an arm's; a listed `embed_tokens` or `lm_head` trains as a full
copy, an unlisted one stays frozen. Unlike an arm's, it applies in **both modes**:

1. **Config cell**: `SCOPE = {'lora_r': ..., 'lora_targets': [...]}` after `SOLVER`, with its
   comment; `ARMS`' comment says a rerun runs `SCOPE` once.
2. **Environment check**: prints `LoRA scope, both modes: {...}` after `solver code: ...`.
3. **Run** (step 6): `solver_config(None)` passes `SCOPE` to `RunConfig`, which writes `lora_r`
   and `lora_targets` into the solver's `--config` JSON; both solvers read them from there
   (`cfg["lora_r"]`, `cfg["lora_targets"]`), so a competition rerun trains what a development run
   trains. The run stage records `scope` in `stages.json`. A `SCOPE` edited to None or `{}` in the
   Config cell runs the reference's settings.
4. **First cell and `MANIFEST.md`**: "Scoped build (task P2.6): `SCOPE` sets the LoRA scope of
   every solver launch, ..." and a `scope (both modes)` row beside the solver's; the arms row
   says "none (one run, the scope above)". `submission_blocker` does not read the scope, so a
   scoped build with the production settings is fit to submit.

A scoped build differs from the same config without `scope` in its first two cells and, by
`SCOPE_EDITS`, in four fixed cells: the Helpers comment and the environment check's series line
(a rerun runs `SCOPE`), the environment check's scope line, the Run note's scope paragraph and
series sentence, and the Run cell's `solver_config` line, docstring, and run record. Every build
without `scope` is byte for byte as before (`tests/test_scope_build.py` rebuilds every committed
build config against digests recorded before the change).

### Adapter arms (task P3.7)

A config with a `scope` may set `adapters` (ADR-019; runbook `docs/runbooks/h_validation.md`):
the adapter arms of its single run, development runs only, with `notebook_inputs`, the other
notebooks' outputs they read (`{input:<owner>/<slug>}` in a path stands for that mount). The
builder checks them with `arc26.arms`' rules (`parse_adapters`: every key spelled out, arms with a
source of `base`, `hyper`, `sgd`, `hyper_polish`, `polish`, `steps` for a polish, `views` of one
prompt length per batch, the load check's tolerances, GEN's counts), reads the expected H config
(`hypernet`, `configs/hypernet.yaml`) from the built commit and embeds it (`resolve_adapters`),
and refuses: the submission notebook, no accelerator, the moved solver, a series, no scope or a
scope that is not H's (rank, targets, and `arc26.solver.load`'s alpha 32), a subset with GEN, a
hard stop before `solver_end_s` + `finish_s` + 900 s, inputs the paths do not name,
`notebook_inputs` without `adapters`, an `agent` stage (task P6.5), and `subset: DEV240` (task
P6.5b; both refused since task int-W43: the arms and the hybrid build were never combined or
tested together). The H config may hold task P3.3b's `aggregator` block (`kind` `mean` or `set`,
positive integer `width`, `blocks`, `heads`, `heads` dividing `width`: `AggregatorConfig`'s rules,
checked by `parse_hypernet`); it is embedded in the form a checkpoint's config stores
(`HypernetConfig.to_dict`): left out when equal to the defaults (`mean`, 512, 2, 8), else with all
four keys, so the load check and every worker compare equal mappings. An adapter build may also
set `dataset_inputs` (task P3.9b): private datasets the arms read at pinned versions,
`<owner>/<slug>: <version>` (H's checkpoint published by a Modal run, `P3.9-02`). They follow the
code dataset in `dataset_sources` as `<owner>/<slug>/<version>`, `{input:<owner>/<slug>}` names
their mount as it names a notebook output's (the environment check finds
`datasets/<owner>/<slug>` by the same search), `BUILD` and `MANIFEST.md` list them, and the
builder refuses them without `adapters`, malformed, unused by the arms' paths, named in
`notebook_inputs` too, or naming the code dataset; `notebook_inputs` and `dataset_inputs`
together attach at most 8. A build without `dataset_inputs` keeps its metadata. In the notebook:

1. **Config cell**: `ADAPTERS = {...}` after `SCOPE`, with its comment; `BUILD` lists the
   `notebook_inputs`.
2. **Helpers**: the arms in effect (`ADAPTER_ARMS`, None in a competition rerun), their folders
   (`/kaggle/working/arm_outputs`, `run_logs/arms`, `/kaggle/h_cache`, `/kaggle/working/gen_data`),
   and two more helpers, `find_input_dirs` and `fill_args`.
3. **Environment check**: lists the arms, the views, GEN and the load check, and finds each
   attached output as the one folder named by its slug (`INPUT_DIRS`).
4. **Run**: `adapter_settings()` fills the paths and adds the folders, the load check's DEV4
   challenges (the competition's file), and GEN's folder; `solver_config(None)` passes them to
   `RunConfig(adapters=...)`, so `run_solver` starts `python -m arc26.arms`; the run stage records
   `adapters`. A competition rerun runs `SCOPE` once, without them.
5. **Write**: unchanged; the arms write nothing into `inference_outputs/`, so `submission.json`
   holds placeholders; each arm's selection and scores are in `run_logs/arms/<arm>/`.
6. **First cell and `MANIFEST.md`**: "Adapter arms build (task P3.7, ...)", "Not for submission"
   (`submission_blocker` names the arms), the adapters rows, the hypernet file's sha256; the
   notebook outputs join `kernel_sources` at their versions.

An adapter build differs from its twin without `adapters` in its first two cells, the Helpers
cell, and by `ADAPTER_EDITS` in the Helpers note, the environment check, the Run note and cell,
and the logs' summary (`tests/test_arms_build.py`); every other build is byte for byte as before.

### Hybrid build (task P6.5)

A config without `arms` may set `agent` (ADR-020): after NVARC's stage and its write, the agent
solver runs in the same session, in both modes. The config spells out every key:

| Key | Meaning | Checked |
|---|---|---|
| `enabled` | the Config cell's `AGENT['enabled']`; false runs NVARC alone, its deadlines unchanged | true or false |
| `nvarc_share_s` | $S$: NVARC's soft deadline when the agent stage follows; 0 for an agent-only build | 0, or $600 \le S \le 41{,}400$ (the competition soft deadline) |
| `end_s` | $E$: the agent's session end, its `--deadline`; SIGTERM then | $600 \le E \le T - w - k - \tau$ |
| `term_s` | $\tau$: from SIGTERM to SIGKILL | 30 to 600, and more than the engine's `stop_timeout_s` plus $k$ |
| `min_stage_s` | $m$: the least time from the stage's start to the generation stop worth an install and a boot | at least 60 |
| `install_timeout_s` | $I$: the offline install's limit | 60 to 7,200 |
| `config` | the agent's run config (task P6.4, `configs/agent_default.yaml`), embedded | a `.yaml` under `configs/`; `resolve_agent` (below) |
| `model` | the weights' download output: `notebook`, `version`, `folder`, `revision` | `<owner>/<slug>`, version at least 1, a folder name, a full commit SHA |
| `wheels` | the engine wheelhouse's: `notebook`, `version`, `folder`, `vllm` | as `model`, a dotted vLLM version; not the model's notebook, not the build's own, no reference kernel source |
| `base_python` | the interpreter that makes the engine's venv | an absolute path (the kernel's own when it is missing on the image) |
| `scratch_dir` | the venv, the engine's caches, NVARC's copies, the agent's logs | an absolute path outside `/kaggle/working` and `/kaggle/input` |
| `min_driver` | the oldest NVIDIA driver for the engine's CUDA 13.0 runtime | a dotted version |

`resolve_agent` reads the run config from the built commit (as `resolve_rounds` reads a governor
config) and checks what the notebook reads from it: `deadline.final_margin_s` ($f$) and
`grace_s`, `server.tensor_parallel_size` (at most the accelerator's GPUs), `boot_timeout_s` ($B$),
and `stop_timeout_s`; and that its JSON copy reads back the same through `yaml.safe_load`, since
the notebook writes it to scratch as JSON for `--config`. Then `agent_plan` must find no problem.

**The time arithmetic.** With $T = 43{,}200$ s, $g = 600$ s (`NVARC_KILL_GAP_S`, the competition
deadlines' gap), $w = 300$ s (`WRITE_RESERVE_S`), and $k = 30$ s (`AGENT_KILL_WAIT_S`), in the
worst case:

$$
\begin{aligned}
\text{NVARC: soft deadline } &\min(s_\text{end}, S),\ \text{hard stop } \min(s_\text{kill}, S + g)\\
\text{the stage starts by } a &= S + g + w \quad (a = w \text{ when } S = 0)\\
\text{the install ends by } &\min(a + I,\ E - f)\\
\text{the engine is up by } &\min(a + I + B,\ E - f)\\
\text{generation stops at } &E - f\\
\text{SIGTERM at } E,\ &\text{SIGKILL at } E + \tau\\
\text{the notebook ends by } &E + \tau + k + w \le T
\end{aligned}
$$

and the builder requires $E - f - a \ge m$. For the hybrid
(`configs/submission_P6.5-hybrid.yaml`: $S = 21{,}600$, $E = 42{,}000$, $f = 1{,}800$,
$\tau = 120$, $I = 1{,}500$, $B = 1{,}800$, $m = 1{,}800$): NVARC's hard stop at 22,200 s, the
stage by 22,500 s, the install by 24,000 s, the engine by 25,800 s, generation until 40,200 s
(11 h 10 min), SIGTERM at 42,000 s, SIGKILL at 42,120 s, and the notebook's end by 42,450 s, 12.5
minutes before the limit. On DEV120, its development subset, NVARC ends near 15,700 s, before its
share, so the agent starts near 15,800 s there. For `P6.5-01` (the hybrid at a sixth of its
times on DEV40: $S = 3{,}600$, $E = 9{,}800$, the rest unchanged): NVARC's hard stop at 4,200 s,
the stage by 4,500 s, the install by 6,000 s, the engine by 7,800 s (near 5,160 s at the spike's
install and boot estimates), generation until 8,000 s, SIGTERM at 9,800 s, SIGKILL at 9,920 s, and
the notebook's end by 10,250 s. $S$ and the generation window $E - f - a$ scale by 40/240; $g$,
$w$, $I$, $B$, $f$, and $\tau$ do not, the distortion `docs/prerun/P6.5-01.md` states.

**The notebook** has 20 cells, the 16 of a single run plus four:

1. **First cell and Config cell.** A "Hybrid build" paragraph states the plan and the two inputs;
   the Config cell adds `AGENT` (the first six keys, editable), and `BUILD` adds `agent` (the run
   config with its sha256, the inputs with their versions, the interpreter, the scratch folder,
   the driver). `MANIFEST.md` adds the rows `agent (both modes)`, `agent run config`,
   `agent inputs`, and `agent engine`, and asks for a brief `Timeout` of at least
   $E + \tau + k + w$ (42,450 s).
2. **4b. Agent check**, after the environment check. It holds the constants, the run config as
   built (`AGENT_CONFIG`), and fourteen helpers copied verbatim from the builder. With
   `AGENT['enabled']` it finds each download output as the one folder named by its slug (hyphens
   or underscores) under `/kaggle/input` or `/kaggle/usr/lib/notebooks` (`find_input_dirs`, a
   script notebook's search); reads the weights' `MANIFEST.json` (`ok`, the pinned revision,
   every listed file at its size; `check_model_output`) and the wheelhouse's (`ok`, the pinned
   vLLM, `install_wheelhouse.py` and `requirements.txt` beside it; `check_wheelhouse`); for a
   build with an accelerator, requires the engine's tensor parallelism in GPUs and a driver of at
   least `min_driver` on every GPU (`driver_refusal`, from `nvidia-smi`); prints the plan, whose
   problems (an `AGENT` edited badly) count as a failed check; and makes the scratch folder. All
   passed: `AGENT_READY` is True. Otherwise a development run stops here (`problem`), and a
   competition rerun logs it and goes on with `AGENT_READY` False.
3. **6. Run** (`HYBRID_EDITS`): when `AGENT_READY`, `nvarc_deadlines` cuts NVARC's soft deadline
   to $S$ and its hard stop to $S + g$, in both modes, each only ever shortened; a share of 0
   skips NVARC (recorded as `skipped`, `agent only: ...`), so the Write cell writes `[[0]]` for
   every task. Without `AGENT_READY`, NVARC keeps its deadlines: a rerun whose agent inputs are
   missing gives NVARC the whole session, and an agent-only build then runs as Submission 2.
4. **7b. Agent stage**, after the Write cell, in both modes. It runs when `AGENT['enabled']`,
   `AGENT_READY`, `arc26` installed, and at least $m$ seconds left before $E - f$; otherwise it
   records `skipped` with the reason. Then, in scratch: NVARC's `submission.json` copied
   (the fallback); NVARC's candidates from `inference_outputs/` as one JSON file, as
   `dump_run_logs` writes them (`candidates_json`; phase 2's order by NVARC's confidence); the
   run config as JSON. The install: `<base_python> <wheelhouse>/install_wheelhouse.py
   --wheelhouse ... --venv <scratch>/venv --report ... --log ... --imports torch,vllm`, without
   `PYTHONPATH` or `PIP_*` (`clean_env`), in a process group of its own stopped at
   $\min(I, E - f - t)$ (`run_supervised`: SIGTERM, SIGKILL 10 s later); it counts only with exit
   0, a report saying `ok`, and `bin/vllm` in the venv. The agent:

   ```text
   python -m arc26.agent.run --mode {dev|competition} --tasks <challenges of the mode>
       --out /kaggle/working/submission.json --config <scratch>/agent_config.json
       --engine-venv <scratch>/venv --model-dir <weights folder>
       --nvarc-submission <scratch>/nvarc_submission.json
       [--nvarc-candidates <scratch>/nvarc_candidates.json]
       --deadline <NOTEBOOK_START + end_s> --log-dir <scratch>/logs
       [--task-ids <SUBSET>, development runs only]
   ```

   with the kernel's environment and `PYTHONUNBUFFERED=1`, in `<scratch>/logs`, its output
   copied to the cell as it arrives. Before it starts, `wait_for_free_gpus` logs `nvidia-smi
   --query-compute-apps=pid,used_memory --format=csv` and `nvidia-smi
   --query-gpu=index,memory.used --format=csv,noheader,nounits` (`gpus_before_engine`: `before`
   and `after`, the processes; `memory_before` and `memory_after`, the MiB in use per GPU;
   `used_max_mib`; `seconds`) and repeats both every 5 s for up to 60 s (`GPU_FREE_WAIT_S`) until
   no process is listed and no GPU uses more than 1,024 MiB (`GPU_USED_MAX_MIB`): NVARC's
   workers, killed at its hard stop or just exited, can hold their memory a moment longer, inside
   a container the process list can be empty while they do, and vLLM asks for 0.90 of each GPU at
   its boot. GPUs still held after 60 s are logged, not waited for further: the boot then
   decides. It starts vLLM from the venv
   itself (`arc26.agent.server`), restarts it once, keeps `submission.json` valid at every moment
   (each finished task rewritten atomically, NVARC's attempts as the fill), and exits 0 whenever
   a valid file is in place. It never gets a solutions path. At $E$ its process group gets
   SIGTERM, at $E + \tau$ SIGKILL
   (`run_supervised`); then an engine it left running is stopped through `<scratch>/logs/engine.pid`
   (`stop_engine`: the pid's argv must still match, or its process gone with members of its group
   left). **The final check** runs whatever happened before it: it removes a cut
   `.submission.json.tmp`, keeps every real grid of the file on disk, and fills every slot that
   is missing, invalid, or a `[[0]]` placeholder with NVARC's copy's grids in their order (never
   one the task's row already holds), the file's own `[[0]]` or NVARC's where neither has a real
   grid (`merge_submission`; P6.2 counts a `[[0]]` grid as missing, so a placeholder never stands
   over an NVARC grid); it writes atomically only when that changes the file, and validates it;
   its report becomes the notebook's `SUBMISSION_REPORT`. A development run then scores NVARC's copy
   and the final file on the subset with `arc26.evaluate` (`score`: both scores, the outputs
   `gained` and `lost`) and copies the agent's small logs to `run_logs/agent/`. Every failure in
   the stage is recorded (`failure`), never raised.
5. **The agent-only fallback** (`nvarc_fallback`, in 7b between the agent and the final check).
   In an agent-only build NVARC did not run, so an agent stage that ends with at least
   `NVARC_FALLBACK_MIN_S` (1,800 s) left before NVARC's soft deadline is followed by NVARC's
   stage, with the deadlines it would have had (41,400 s and 42,000 s in a rerun, `SOLVER_END_S`
   and `SOLVER_KILL_S` in a development run), whatever the agent verified before: the install or
   the boot failed, the agent crashed, was killed, or refused its arguments, its engine was gone
   for good after its restart, every task was dropped for engine errors, or it ran out of work.
   With less time left it records `skipped`; an agent that runs to its generation stop leaves
   less. It logs the GPUs (`gpus_before`), stops an engine still running through its pid file
   (`engine_stopped`), and waits for free GPUs (`gpus`, as above); then `launch` runs NVARC and
   `select_attempts` writes NVARC's selection to a staging file. P6.2's `select_files` then
   chooses every slot from the agent's attempt log (`attempts.jsonl` through `read_records`, a
   line cut by a kill skipped; never `run.json`, which the agent writes only at its exit) with
   that staging file as its NVARC input (`nvarc_fallback.selection`: `records`,
   `skipped_lines`, the `slots` by source), so a verified program keeps attempt 1 and NVARC only
   fills; `merge_submission` then adds the agent's grids on disk in any slot both left empty
   (`sources`), written atomically. The agent logs each record before it rewrites the file
   (P6.4's scheduler), so a kill loses no logged answer. The agent's file stays on disk, valid,
   until that write; when NVARC's selection wrote no file, it stays (`merge`); a fallback that
   raises is recorded (`fallback_error`), and the final check runs either way. What it does not
   cover, the agent-only build's risk: an agent that runs to its generation stop keeps the whole
   session with no NVARC attempt, however few programs it verifies; an engine that answers but
   fails part of its calls, since nothing watches the error rate live (P6.4's run ends early only
   when its engine is gone or every task is dropped);
   and any failure with less than 1,800 s left before NVARC's soft deadline.

A competition rerun runs the same stages with the same times: the hybrid's settings are run
settings, like `SOLVER` and `SCOPE`. So a build whose agent ends before 41,400 s (`P6.5-01`'s
sixth of the session) is a development scale and `submission_blocker` marks it not for
submission. A build without `agent` is byte for byte as before
(`tests/test_hybrid_build.py` rebuilds every committed build config against digests recorded at
`8cac42f`); `HYBRID_EDITS` holds the two fixed cells' hybrid lines, each old text matched exactly
once after `SCOPE_EDITS`.

### Shot build and DEV240 (task P6.5b)

A hybrid config may set `agent.entry` (ADR-020; brief `docs/tasks/P6.5b.md`): `arc26.agent.run`,
the default (a config without the key builds as before, byte for byte:
`tests/test_shot_build.py` rebuilds the four P6.5 hybrid configs against digests recorded at
`dbbc782`), or `arc26.agent.shot`, the long shot of run `P6.3-05` (`docs/modules/agent_shot.md`,
`configs/agent_shot.yaml`), which takes exactly `arc26.agent.run`'s arguments and exit codes. In a
shot build:

| Key | Meaning | Checked |
|---|---|---|
| `entry` | `arc26.agent.shot`; `BUILD['agent']['entry']` and the agent check's `AGENT_ENTRY` | one of `AGENT_ENTRIES` |
| `nvarc_share_s` | NVARC's deadlines unchanged (brief, design 1): `nvarc_deadlines` never shortens them | exactly 41,400, the competition's soft deadline |
| `end_s` | $E$: the shot's `--deadline`; SIGTERM then | as in a hybrid build |
| `min_stage_s` | $m$: the least time from the stage's start to the generation stop worth an install, the shot's setup, and its smallest wave | `shot_plan` below |
| `config` | the shot's run config, `configs/agent_shot.yaml` (the smoke: `configs/agent_shot_P6.5-02.yaml`) | `resolve_agent`, as a hybrid's |

The other keys are the hybrid's. **The plan** (`shot_plan`, which the builder runs instead of
`agent_plan` and the agent check prints): NVARC may use its whole session, so there is no
worst-case start; the plan names the latest start of the agent stage that still runs the shot,

$$
L = E - f - m,
$$

and the builder refuses a plan with $L < w$ (no NVARC end could leave the shot $m$ seconds), one
with a share of 0 (the shot picks its tasks from NVARC's candidates), or one whose notebook could
end after $E + \tau + k + w > T$. At run time the agent stage starts after the Write cell, at
$a = n + W$ with $n$ NVARC's end and $W$ the Write cell's time, and skips the shot when $a > L$
(the hybrid's own check, `min_stage_s` before $E - f$). Then the notebook installs the wheelhouse
(at most `install_timeout_s`), waits for free GPUs (at most 60 s), and runs `python -m
arc26.agent.shot` with the hybrid's command line plus, in a development run only, `--solutions`
(the evaluation solutions of `DATA_DIR`, for the shot's own score in `shot_metrics.json`, read
after its answers). The shot prepares the venv as `P6.3-05` did (`engine_prep`: the orphan
FlashInfer provider removed, the empty `sitecustomize`, the workers' imports, the GPU preflight),
boots vLLM once with `P6.3-05`'s flags at `max_model_len` 131,072, sends a sanity request, ranks
the tasks by NVARC's confidence from its candidates (per test output the softmax at temperature 1
of `score_kgmon` over the output's distinct grids, the top grid's share; 0 without a candidate;
per task the minimum; ascending, ties by id), keeps those under `selection.max_confidence`, and
runs waves of at most 15 tasks, each wave's thinking cap $C$ the largest whose planned length
(Part A's model, `configs/agent_shot.yaml`, `waves`; `docs/modules/agent_shot.md`), with $n$ the
wave's tasks and $o$ its longest opening bound,

$$
T(C, n, o) = \frac{C}{\phi r} + \frac{n (o + C)}{p} + \frac{F}{r} + \mu, \qquad \phi = 0.85,\
r = 19.0,\ p = 958.8,\ F = 6{,}144,\ \mu = 300,
$$

(the thinking turn, then the forced replies with every turn of the wave assumed cut, prefilled
together at `P6.3-05`'s 958.8 prompt tokens per second, then the longest forced reply) ends before
$E - f$ and whose KV need fits the pool the boot reports; below $C = 16{,}384$ no wave
starts. Per test output the merge keeps NVARC's first attempt and puts the shot's first answer in
the second slot where it differs, its second answer where the first equals NVARC's and NVARC's
second slot is empty (both of the shot's answers where NVARC had no candidate). At
$E - f$ every call still running is cancelled and its task keeps NVARC's attempts; SIGTERM at
$E$, SIGKILL at $E + \tau$, the final check after, as in the hybrid.

**The stop path.** A shot that still runs at $E$ (a hang, a stop that never came) gets SIGTERM in
its process group: it cancels its calls, stops its engine (`stop_timeout_s` 60 s, then SIGKILL and
`arc26.agent.server`'s 30 s wait for the engine's group), and writes once more, about 95 s with
the drain; $\tau = 120$ s leaves room. vLLM runs in a session of its own, so the shot's SIGKILL at
$E + \tau$ misses it: the notebook then stops it through `<scratch>/logs/engine.pid`
(`stop_engine`: SIGTERM to the engine's process group, SIGKILL 30 s later, its argv checked
first). `tests/test_shot_notebook.py` runs both with the real `arc26.agent.shot` and its fake
engine, its `--deadline` moved 600 s past $E$: SIGTERM honored (`ended` `terminated`, the shot
stopped its engine) and SIGKILL at once (`ended` `killed`, `engine_stopped` names the group the
notebook stopped); neither leaves an engine running.

**$m$, from `P6.3-05`'s measurements** (`docs/tasks/P6.5b.md`, its table; `configs/agent_shot.yaml`):
the install 106 s plus 30% (138 s), the GPU wait's cap 60 s, and the shot's `min_shot_s`
(`arc26.agent.shot.stage_arithmetic`, 2,450.1 s): its `setup_s` 450 s (engine preparation 40.3 s,
preflight 3.4 s, boot 300.1 s, sanity 1.7 s, plus 30%) and the smallest wave of 15 at
$C = 16{,}384$ and $o = 6{,}771$, $T = 1{,}014.5 + 362.2 + 323.4 + 300 = 2{,}000.1$ s: 2,647.9 s,
rounded up to $m = 2{,}700$ s (45 minutes) for the stage's copies (NVARC's submission and
candidates, about 4 MB on 240 tasks) and the shot's start. A full wave of 15 at $C = 94{,}208$
is planned at $T = 8{,}036.5$ s (the turn 5,833.3 s, the forced replies 1,903.1 s).

**The time before the first cell** (task P6.5b, the coordinator's decision of 14 Oct). Kaggle
showed `P6.5-02` RUNNING at 16:15:49 and its first cell started at 16:52:33, 36.7 minutes later
(`P6.3-02` revision 5: 41 minutes; `P6.3-04`: about 10; it grows with the attached inputs). If
Kaggle's 12 h limit counts from RUNNING, a plan against 43,200 s of notebook time would be killed
mid-wave, in `P6.5-03` and in a scored rerun alike. So a shot build plans against
`SHOT_SESSION_S` $= T - A = 43{,}200 - 2{,}700 = 40{,}500$ s of notebook time, with the allowance
$A$ (`PRE_SCRIPT_ALLOWANCE_S`, 45 minutes) above every measured case; the builder refuses a shot
whose notebook could end later (`shot_plan`), and a production shot ends at 38,700 s or later
(41,400 s less $A$; `submission_blocker`). The allowance applies to the shot only: NVARC keeps
Submission 2's deadlines (41,400 s and 42,000 s), which neither scored run came near (`P2.6-01`
ended its solver at 15,703 s on DEV120; Submission 1 and Submission 2 both scored). A shot
build's environment check logs, from inside, the clocks that could see those minutes
(`session_clocks`, `stages.environment.clocks` in `stages.json`): the wall time and the time
since `NOTEBOOK_START`, `/proc/uptime`, the ages of PID 1 and of the kernel (their start ticks in
`/proc/<pid>/stat`), and the ages of `/kaggle/working`'s times; nothing depends on them, and
`P6.5-03` shows which of them sees the gap.

**The time arithmetic of the 12 h check `P6.5-03`** ($E = 40{,}050$, $f = 120$, $m = 2{,}700$,
$\tau = 120$; `configs/submission_P6.5-03.yaml`): $E = 43{,}200 - 2{,}700 - \tau - k - w$. NVARC's
soft deadline 41,400 s and hard stop 42,000 s (its own); $L = 37{,}230$ s, the latest start of the
agent stage (NVARC's end plus the Write cell) that still runs the shot; generation stop 39,930 s;
SIGTERM 40,050 s, SIGKILL 40,170 s; the notebook's last line by 40,500 s of notebook time (about
40,100 s expected: the shot stops itself at the generation stop and exits about 90 s later), so
by 43,200 s from RUNNING with up to 45 minutes before the first cell. That end is nominal: $k = 30$
s counts one kill wait, while after SIGKILL the notebook drains the output for up to 5 s and
`stop_engine` waits up to 30 s after SIGTERM and 30 s after SIGKILL; the 300 s write reserve holds
that (worst case about 40,300 s before the final check, which takes seconds). On 240 tasks at
$u = 0.74$ (`P2.6-01`: the window from the solver's launch to the soft deadline is 41,350 s; the
launch comes about 86 s after the start, after the agent check's 36.5 s measured in the CPU twin),
NVARC ends near $n = 86 + 0.74 \cdot 41{,}350 = 30{,}685$ s; the Write cell takes about 88 s on 240
tasks (44.2 s on 120, `P2.6-01`), so the stage starts near 30,773 s and the first wave near
31,230 s (install 106 s, GPU wait about 5 s, setup 345.5 s): 8,700 s (2.42 h) of waves before the
generation stop at 39,930 s. A full wave of 15 at $C = 94{,}208$ is planned at $T = 8{,}036.5$ s;
a second wave of 15 runs only if the first ends early: at `P6.3-05`'s measured 5,784 s it gets a
cap of about 28,200, at the planned time none (664 s left). So 15 to 30 tasks are shot (an
estimate: the openings of DEV240's first 30 are not `P6.3-05`'s). The latest NVARC end that still
runs a wave is $L - W \approx 37{,}142$ s ($u \approx 0.896$), a wave of 15 at a cap of about
17,000 then. Past that, or if NVARC runs to its own deadlines, the notebook ends near 42,200 s of
notebook time without a shot: Submission 2's exposure, unchanged.

**The smoke `P6.5-02`** (DEV4 on `alyahya737/arc26-hybrid-dev`; $E = 5{,}300$, NVARC's
development deadlines 2,100 s and 2,400 s): $L = 2{,}480$ s, after NVARC's hard stop plus the Write
cell (about 2,430 s); generation stop 5,180 s, SIGKILL 5,420 s, the notebook's end by about
5,750 s. NVARC ends near 1,300 to 1,600 s (`P0.5-01`: the solver 1,443 s at rank 256; rank 16
trains in about 0.8 of that), so one reduced wave of 3 runs with $C$ about 37,600 to 42,300. Its run config
`configs/agent_shot_P6.5-02.yaml` is `configs/agent_shot.yaml` with `selection.max_confidence`
1.0 instead of 0.5: on `P2.6-01`'s candidates every DEV4 task is at 0.687 or above (`0934a4d8`
0.687, `36a08778` 0.700, `aa4ec2a5` 0.9995, `981571dc` 1.0), so at 0.5 the smoke would shoot
nothing; at 1.0 it takes the three tasks with two or more distinct grids.

**Submission 4: the 12 h check `P6.9-02`** (task P6.9; `configs/submission_P6.9-02.yaml`, which
is `P6.5-03`'s config with three changes a test checks: the code dataset
`alyahya737/arc26-code-p69`, $m = 6{,}500$, and shot v2's run config `configs/agent_shot_P6.9.yaml`,
the bucket order, `min_cap` 65,536, the refill queue, the separate forcing turn, the lenient
parser). $E$, the generation stop 39,930 s, SIGKILL 40,170 s, and the notebook's end by 40,500 s are
`P6.5-03`'s; $m$ grows because the smallest start is 15 streams at $C = 65{,}536$
(`stage_arithmetic`: $T = 5{,}812.5$ s, `min_shot_s` 6,262.5 s, plus the install's 137.8 s and
the GPU wait's 60 s: 6,460.3 s), so $L = 40{,}050 - 120 - 6{,}500 = 33{,}430$ s and NVARC must end
by about 33,340 s ($u \approx 0.804$; `P6.5-03` measured 0.723). Under the refill queue
(`plan_start`) a task starts when a stream is free with the cap the time left allows at the
streams then running: at 15 streams the full cap needs 8,036.5 s before the stop (starts after
about 31,894 s get less) and the smallest 5,812.5 s, so no task starts after about 34,117 s. At
`P6.5-03`'s measured times (first starts near 30,445 s) that gives 16 to 21 starts, the last task
ending near 36,800 to 39,000 s (a Monte Carlo over `P6.9-01`'s turn lengths, brief section 6).
Its CPU twin `configs/submission_P6.9-02-cpu.yaml` is `P6.5-02-cpu`'s with the code dataset, the
run config, and $E = 7{,}200$ s: the shot's boot rule starts the engine's preparation only with the
setup and one task's smallest wave left (about 5,210 s at `min_cap` 65,536), which 3,600 s would
never leave; its notebook ends by 7,650 s, the push's `--timeout`.

**DEV240** (`subset: DEV240`, development runs only): the environment check writes every
evaluation task and a copy of each, under the first 8 hex digits of the sha256 of its id
(`arc26.data.dev240_copy_id`), to `/kaggle/arc26_dev240/` with the original's solutions
(`dev240_files`), and points `CHALLENGES` and `DATA_DIR` there: NVARC, its selection and score,
the agent stage, and the final scores read the 240 tasks, each copy scored apart under its own
id. A copy's id must be no id of any `*_challenges.json` or `*_solutions.json` in the competition's
folder and no other copy's (else the run stops in the environment check); the 120 copies collide
with no DEV120 id (`arc26.data.DEV240_COPIES`). `SUBSET` stays None, `stages.json` records the
report under `settings.dev240` (paths, `tasks`, `test_inputs`, `copies`, `checked`), and
`MANIFEST.md`'s subset row says DEV240. A competition rerun is unchanged and writes no copy. Every
id parser takes a copy (8 hex digits, no underscore): the runner's subset check, NVARC's
`<id>_<n>` keys, the agent's task loader and candidate keys, and `scripts/calibration_report.py`
(`infer_subset` names DEV240).

The shot build's notebook has the hybrid's 20 cells. It differs from the hybrid's in the first
cell ("Hybrid build: NVARC, then the long shot"), the Config cell (the shot's `AGENT` comment and
`BUILD['agent']['entry']`), the agent check (`AGENT_ENTRY`, `shot_plan`'s source, and its call),
and `SHOT_EDITS` in the Run note, the agent check's note, the agent stage's note, and the agent
stage (`AGENT_ENTRY` as the module, `--solutions` in a development run, the log line). A DEV240
build adds `DEV240 = True` to the Config cell, a paragraph to the first cell, `dev240_files` and
its call to the environment check, and `settings.dev240` to `stages.json`
(`dev240_edits`). `MANIFEST.md` adds the row `agent entry` and the DEV240 subset row.

### Weights warm-up (task P6.9c)

Configs: `configs/submission_P6.9-03.yaml` (Submission 4 with the warm-up: `P6.9-02`'s build plus
`agent.warmup`, on 4 DEV120 tasks in its own development run `P6.9-03`) and
`configs/submission_P6.9-03-cpu.yaml` (its CPU twin `P6.9-03-cpu`, the warm-up at no delay).
Brief: `docs/prerun/P6.9-03.md`. Run `P6.9-02` read Fault at the boot: Kaggle now serves
notebook-output inputs through FUSE.GCSFUSE, vLLM 0.31.0 prefetches only on NFS and Lustre
(`vllm/model_executor/model_loader/weight_utils.py`, `safetensors_weights_iterator`, at
`db9527a4`), and its memory-mapped load of the 19.5 GB took 802.41 s, past the shot's 900 s boot
limit. A shot build with the optional key `agent.warmup` copies the weights to local disk while
NVARC solves, in both modes:

1. **The start (agent check, before NVARC).** When the check passes, `start_warmup` reads the
   weights output's `MANIFEST.json` (each file's name, size, sha256), requires the copy plus
   `reserve_gb` free in scratch, writes the spec and the copier (`WARMUP_SCRIPT`: the builder's
   `copy_weights`, verbatim, with a `__main__` line) to scratch, and starts it with the kernel's
   interpreter (`-S`, standard library only) in a session of its own, its output in
   `warmup.log`. It never fails the check; a warm-up that cannot start is recorded in
   `stages.agent_check.warmup`.
2. **The copy (`copy_weights`).** After `delay_s`, at `nice` and the lowest best-effort IO level
   (`ionice -c 2 -n 7` when on `PATH`), every file is created at its size and cut into pieces of
   `piece_mib`; `workers` threads copy the pieces, each read in order through its own handle in
   8 MiB blocks and written at its offset. Then every file is checked at its size and, where the
   manifest has one, its sha256 (the Hub's, which the download notebook verified). The status
   `<scratch>/warmup.json` is rewritten atomically at each step and every `progress_s`; it ends
   `done` with `ok` true only when every file passed.
3. **The choice (agent stage, after the install).** `warmup_weights` reads the status every 5 s:
   `done` and `ok` with every file at its manifest size on disk gives the local folder
   `<scratch>/weights/<folder>`. A copy still making progress is waited for, at most `wait_s` and
   never `stall_s` without a byte copied or hashed. Otherwise (no start, a failure, a stall, the
   wait over, a file changed since its check) the copier's process group is killed and the engine
   loads from the input mount with `--safetensors-load-strategy <fallback_load_strategy>` appended
   to `server.extra_args` in the run config the stage writes to scratch. The choice is
   `stages.agent.weights`; the shot gets its folder as `--model-dir`. The boot limit is the run
   config's.
4. **The end.** A copier still running (a stage that did not reach the engine) is killed
   (`stages.agent.warmup`); a development run copies `warmup.json` and `warmup.log` to
   `run_logs/agent/`; a competition rerun writes nothing but `submission.json` to
   `/kaggle/working`.

The time arithmetic. With $B = 19{,}472{,}990{,}100$ bytes, a copy rate $r$, the hash time $h$,
the copy's start $t_0 = 15 + \texttt{delay\_s}$ s (the agent check ends near 15 s), and the
folder's choice at $t_c$ (NVARC's end $n$, the Write cell, and the install: $t_c \approx n + 150$
s), the engine reads the local copy without a wait when

$$
t_0 + \frac{B}{r} + h \le t_c ,
$$

and with the wait when the left side is at most $t_c + \texttt{wait\_s}$. In a rerun ($n \approx
30{,}086$ s, `P6.9-02`) that holds down to $r \approx 0.65$ MB/s. In `P6.9-03` ($n \approx 1{,}180$
s, $t_c \approx 1{,}330$ s, $h \approx 125$ s) it needs $r \ge B / (1{,}330 - 195 - 125) \approx
19.3$ MB/s without the wait and $r \ge 10.2$ MB/s with it; vLLM's own memory-mapped load of this
mount read $B / 802.41 \approx 24.3$ MB/s, and the CPU twin `P6.9-03-cpu` measured the copy at
219.8 MB/s (88.6 s) and the hash at 123.5 s on 4 CPUs (`runs/P6.9-03-cpu/`). The wait comes before the stage computes the shot's timeout
from `NOTEBOOK_START + AGENT['end_s']`, so the shot's SIGTERM, SIGKILL, and the notebook's end stay
where `shot_plan` puts them.

The cells: a warm-up build is the shot build with `WARMUP_EDITS` in the agent check's note, the
agent check (the manifest's path, the start), the agent stage's note, and the agent stage (the
choice, the folder in the command, the end), plus `start_warmup`, `stop_warmup`,
`warmup_weights`, and `warmup_constants_source`'s constants in the agent check. A shot build
without the key keeps every cell byte for byte (`tests/test_warmup.py`; `P6.9-02`'s notebook
rebuilt at `354f5b5` differs from `dist/P6.9-02` only in the provenance cells). No file of
`arc26/` changes, so the wheel and the code dataset stay `P6.9-02`'s.

### MTP and the 69,632 cap (task P6.9d)

Configs: `configs/submission_P6.9-04.yaml` (`P6.9-03`'s build, `arc26-submission` version 6's,
with `agent.config` naming `configs/agent_shot_P6.9d.yaml`; `P6.9-03`'s 4 DEV120 tasks in its own
development run `P6.9-04`) and `configs/submission_P6.9-04-cpu.yaml` (its CPU twin
`P6.9-04-cpu`, `P6.9-03-cpu`'s with the same one change). Brief: `docs/prerun/P6.9-04.md`;
[ADR-023](../decisions/ADR-023-submission-4-shot-v2.md) Amendment 2. The shot's run config differs
from Submission 4's in two values:

1. **`server.extra_args`**: `["--speculative-config", '{"method": "mtp",
   "num_speculative_tokens": 1}']`, task/P6.10's flag verbatim. vLLM 0.31.0 drafts one token per
   step with the checkpoint's own MTP head (`model_mtp.safetensors`, one of the 12 files the
   warm-up copies and verifies) and verifies it in the next forward pass. The warm-up's fallback
   appends `--safetensors-load-strategy prefetch` after it.
2. **`waves.max_cap`**: 69,632 instead of 94,208. With $o = 6{,}771$ (the opening bound), $F =
   6{,}144$ (the forcing cap), and $n = 15$ streams, the plan's KV need at cap $C$ is
   $n\,(o + C + F)$: 1,238,205 at 69,632 and 1,606,845 at 94,208, against MTP's Kaggle pool
   estimate $\hat P = 1{,}507{,}328 \cdot 1{,}699{,}566 / 1{,}746{,}170 \approx 1{,}467{,}099$
   (`P6.10-01`'s two Modal boots scaled by `P6.5-03`'s Kaggle boot). With MTP's measured
   per-request blocks ($m = 6.53$ blocks of 1,588.3 tokens, `P6.10-02`'s brief) the need is
   $n\,(o + C + F + m B) \approx 1{,}393{,}779$, still inside. The plan checks the pool the boot
   reports (`plan_wave`, `plan_start`), so a smaller pool runs fewer streams instead of failing.

Nothing else moves. `min_cap` stays 65,536, so the shot's planned times keep P6.9's
$T(C, n, o) = C / (\phi r) + n (o + C) / p + F / r + m_s$ with $\phi = 0.85$, $r = 19.0$, $p =
958.8$, $m_s = 300$: the smallest start $T(65{,}536, 15, 6{,}771) = 5{,}812.5$ s and the minimum
shot $6{,}262.5$ s are P6.9's, so the notebook's `min_stage_s` (6,500 s), the latest stage start
(33,430 s), the last shot start (about 34,118 s at `P6.5-03`'s times), the generation stop (39,930
s), SIGTERM (40,050 s), and the notebook's end (by 40,500 s) are `P6.9-03`'s. A start at the full
cap plans $T(69{,}632, 15, 6{,}771) = 6{,}130.2$ s instead of 8,036.5 s. The rate $r$ stays the
base's 19.0: MTP's rate at long context on Kaggle is unmeasured, so the plan keeps its margin and
MTP's speed shows as turns that end sooner and more refills in the same window. Every start's cap
is $\min(C_\text{P6.9}, 69{,}632)$, P6.9's at the same time left (`tests/test_p6_9d.py`).

The boot: from the local copy the base booted in 278.1 s (`P6.9-03`); MTP added 60.0 s on Modal
(`P6.10-01`: 416.2 s against 356.2 s), so about 340 s (**estimate**) against the 900 s limit. The
notebook's cells differ from `P6.9-03`'s only in the provenance (`config-note`, `config`) and the
embedded run config (`agent-check`); the wheel and the code dataset are `P6.9-02`'s, byte for
byte, as no file of `arc26/` changed.

### The 69,632 cap without MTP (run P6.9-05, variant 2 of task P6.9d)

Configs: `configs/submission_P6.9-05.yaml` (`P6.9-04`'s build with `agent.config` naming
`configs/agent_shot_P6.9e.yaml`) and `configs/submission_P6.9-05-cpu.yaml` (its CPU twin
`P6.9-05-cpu`, `P6.9-04-cpu`'s with the same one change). Brief: `docs/prerun/P6.9-05.md`;
[ADR-023](../decisions/ADR-023-submission-4-shot-v2.md) Amendment 3. The shot's run config is
Submission 4's (`configs/agent_shot_P6.9.yaml`) with one value changed, `waves.max_cap` 94,208 to
69,632, and `server.extra_args` left empty: it is `configs/agent_shot_P6.9d.yaml` without the
speculative flag (`tests/test_p6_9e.py`). The engine, its flags, and its KV pool are version 6's,
which `P6.9-03` ran (`boot.json` `kv_pool` 1,699,566); the 69,632 cap ran on GPU in `P6.9-04`
(`shot_plan.json` `model.max_cap` 69,632, three starts at it). With $o = 6{,}771$, $F = 6{,}144$,
and $n = 15$, the plan's need at the cap is $n\,(o + C + F) = 1{,}238{,}205$, inside the base's
pool. A turn that would run to 94,208 ends after
$$\frac{69{,}632}{94{,}208} = 0.739$$
of its tokens, 26% sooner, and its stream refills; a turn between the caps gains less. The time
model, `min_cap`, `min_stage_s`, and every deadline are `P6.9-04`'s, so P6.9d's arithmetic above
holds unchanged: the full-cap start plans 6,130.2 s, the smallest 5,812.5 s.

### A rank-256 pass on rank 16's confidence-0 tasks (run P4.7-01, task P4.7)

Configs: `configs/submission_P4.7-01.yaml` (NVARC alone with the reference's LoRA settings on 8
DEV120 tasks, 4 workers on `NvidiaL4`, a development build for its own notebook
`alyahya737/arc26-p4-7-r256`) and `configs/submission_P4.7-01-cpu.yaml` (its CPU twin
`P4.7-01-cpu` on the CPU check's notebook, the build with only the CPU checks' four values
changed). Brief: `docs/prerun/P4.7-01.md`; task brief `docs/tasks/P4.7.md`; the screen behind it
`docs/research/second_pass_screen.md` on `task/ideas` (`e79a1a0`). Tests: `tests/test_p4_7.py`.

The config sets no `scope`, no `agent`, and no `arms`, so the notebook's Run cell builds
`RunConfig` with `scope = {}` and the solver trains `RunConfig`'s defaults: rank 256 on the nine
target modules (`arc26/runner.py` lines 63 to 67, 137 and 138), the reference's `peft_params`
and `train_args` (`arc26/solver.py` lines 120 to 156), the run of `P1.1-01` and Submission 1.
The notebook's cells are `P1.1-01`'s and `P2.0-01`'s from the same builder but the first cell and
the Config cell; the Config cell differs from `P2.0-01`'s (the split solver at the same LoRA
settings, Pass) only in `SUBSET`, `SOLVER_END_S`, and `SOLVER_KILL_S`, and from `P1.1-01`'s also
in `SOLVER` (`nvarc` there).

The deadlines come from `P1.1-01`'s puzzle seconds $s_i$ on the 8 tasks (its session log). The
solver's queue hands out the ids in sorted order to the next free worker, so with 4 workers
started together the last puzzle ends at the list-scheduling makespan
$$M = \max_{w} \sum_{i \in Q_w} s_i = 1{,}289.2 \text{ s}, \qquad
M \le \frac{1}{4} \sum_i s_i + \frac{3}{4} \max_i s_i = 1{,}011.0 + 706.5 = 1{,}717.5 \text{ s},$$
where $Q_w$ is the set of tasks worker $w$ takes; `e12f9a14`, the last id, starts at 464.2 s. With
the start-up $u$ (launch and four model loads, 120 s in `P6.9-02`, about 400 s in `P2.0-01`, whose
solver stage of 1,632 s held a longest puzzle of 1,231 s), the stage ends near $u + M$ = 1,410 to
1,690 s. The soft deadline $e = 2{,}700$ s also stops a DFS in progress (`arc26/decode.py` line
129), so it covers the per-puzzle cap's worst case for the last task:
$u + 464.2 + 1{,}200 + 540 \approx 2{,}604$ s plus its rescoring (the decode stop, line 369, and
one DFS batch's budget). The hard stop $k = e + 600 = 3{,}300$ s keeps the competition's gap. The
brief's `Timeout` is
$$T = k + R + A + S = 3{,}300 + 300 + 2{,}700 + 180 = 6{,}480 \text{ s} \to 6{,}600 \text{ s},$$
with $R$ the write reserve (`WRITE_RESERVE_S`), $A$ the pre-script allowance
(`PRE_SCRIPT_ALLOWANCE_S`), and $S$ the save. The wheel is `a5bb333f`, byte for byte (no wheel
input changed since `task/P6.9` `5666867`), so `alyahya737/arc26-code-p69` is attached as it is.

### The rank-256 tail (task P4.7b, run P4.7-02)

Configs: `configs/submission_P4.7-02.yaml` (version 8's build, `configs/submission_P6.9-05.yaml`,
with `agent.config` naming `configs/agent_shot_P4.7b.yaml`, the key `agent.tail`, and a three-task
development subset) and `configs/submission_P4.7-02-cpu.yaml` (its CPU twin `P4.7-02-cpu`,
version 8's twin with the same two agent changes). Brief: `docs/prerun/P4.7-02.md`; task brief
`docs/tasks/P4.7b.md`; [ADR-023](../decisions/ADR-023-submission-4-shot-v2.md) Amendment 4. Tests:
`tests/test_p4_7b.py`. In a scored rerun:

1. **NVARC** runs as in version 8 (rank 16 on q, k, v, o, every test task, the competition
   deadlines). The one difference: its workers record their timing in the rerun too, in
   `/kaggle/arc26_timing/` (scratch; `TIMING_DIR`), so the tail can read NVARC's seconds.
2. **The shot** runs as in version 8 with `selection.zero_bucket: false`: its queue holds only
   tasks in $(0, 0.5)$. Before the install, the agent stage computes that queue as the shot does
   (`nvarc_confidence`, `shot_queue`); an empty queue installs nothing and boots no engine. The
   shot returns, and stops its engine, as soon as no queued task can start and no call is running:
   `run_refill` returns once `running` is empty with no start left (`arc26/agent/shot.py` lines
   1949 to 1953), `run_waves` closes its supervisor, and `run` stops the server in its `finally`
   (line 2385); the notebook's `run_supervised` returns with the process, and `stop_engine` checks
   the pid file. So the shot holds the GPUs only while it works.
3. **The tail** (the tail cell, after the agent cell) takes NVARC's tasks at confidence exactly 0
   that the shot did not answer, ordered by NVARC's `puzzle` seconds $s_i$, cheapest first
   (`tail_tasks`; by id without records), and runs `python -m arc26.tail` on them: the split
   solver's workers with `RunConfig`'s defaults (rank 256 on the nine target modules, the
   reference's, as `P1.1-01`), its tasks in a challenges file of their own
   (`<root>/tail_data/`, named as the mode's file), its own candidate, marker, and timing folders.
   With the agent stage's end $E$ (`AGENT['end_s']`, 40,050 s), the write time $w$ and the kill
   gap $g$ (`write_s`, `kill_gap_s`, both 300 s), its soft deadline and hard stop are
   $$e = E - w - g = 39{,}450 \text{ s}, \qquad k = e + g = 39{,}750 \text{ s},$$
   and it is skipped with less than `min_s` (600 s) before $e$. Before its workers load, the
   engine is stopped through its pid file, a warm-up copy still running is stopped, and the GPUs
   are waited for as before the engine's boot; GPUs still held skip it.
4. **The merge** (`tail_merge`): NVARC's selection over the tail's candidates gives two attempts
   per output; an output takes them only when its attempt 1 is `[[0]]` in the file and in NVARC's
   copy (no NVARC candidate, no shot answer). The merged file is validated before its atomic
   write; any failure leaves the shot's file. The final check runs again, and the notebook's last
   line still comes by $E + 120 + 30 + 300 = 40{,}500$ s.

How much fits: at version 8's median shot end at 240 tasks (about 6,360 s after its first start
near 30,555 s, `docs/research/shot_tail_screen.md` on `task/ideas`) the tail starts near
36,915 s with $e - 36{,}915 = 2{,}535$ s; with a start-up of about 400 s and rank 256's 554 GPU s
per confidence-0 task (`docs/research/second_pass_screen.md`), 4 workers finish about
$4 \cdot (2{,}535 - 400) / 554 \approx 15$ tasks, the cheapest first (**estimate**).

### LLM first (task P5.1, run P5.1-01)

Configs: `configs/submission_P5.1-01.yaml` (version 9's build, `configs/submission_P4.7-02.yaml`,
without its tail, with the key `agent.skip_nvarc: true`, `agent.config` naming
`configs/agent_shot_P5.1.yaml`, the warm-up's `delay_s` 0, `min_stage_s` 5,300, its own code
dataset `alyahya737/arc26-code-p51`, and a three-task development subset) and
`configs/submission_P5.1-01-cpu.yaml` (its CPU twin `P5.1-01-cpu`, version 9's twin with the
same agent changes). Brief: `docs/prerun/P5.1-01.md`; task brief `docs/tasks/P5.1.md`;
[ADR-023](../decisions/ADR-023-submission-4-shot-v2.md) Amendment 5. Tests: `tests/test_p5_1.py`.
In a scored rerun:

1. **The checks and the warm-up as version 9.** The environment check, the agent check, and the
   `arc26` install run unchanged; the weights' warm-up starts at the agent check at once
   (`delay_s` 0: no NVARC model load to leave the mounts to).
2. **No NVARC.** When the agent check passes (`AGENT_READY`), the Run cell records
   `stages.run.skipped` = `SKIP_NVARC_REASON` and starts no solver, in both modes; the Write cell
   runs no selection and `finalize_submission` writes `[[0]]` in every slot
   (`stages.write.skipped`, `raw_status` `none`). The reason does not start with "agent only", so
   the agent-only fallback never runs. A failed agent check runs NVARC alone with its full
   deadlines, as in every hybrid build. The lines come from `SKIP_NVARC_EDITS`
   (`scripts/build_kaggle_notebook.py`); a build without the key keeps every cell, byte for byte
   (version 9's four files hash as the base commit's builder wrote them, a test).
3. **The shot on every task** (`arc26/agent/shot.py`). `selection.order: opening` queues every
   task by ascending opening bound $o_i$ (`opening_bound`: one token per character plus slack),
   then id; NVARC's confidence is not computed. The refill queue runs $n = 20$ streams at
   $C = 65{,}536$ (`max_cap` = `min_cap`), effort xhigh, `direct_two`, the forcing turn, the
   lenient parser, as version 9. A start is planned as one task among the streams running once it
   starts,
   $$T(C, n, o) = \frac{C}{\phi r} + \frac{n (o + C)}{p} + \frac{F}{r} + m,$$
   with $\phi = 0.85$, the planner's rate $r = 30$, $p = 958.8$, $F = 6{,}144$, $m = 300$; at
   $n = 20$ and $o = 6{,}771$ that is $2{,}570.0 + 1{,}508.3 + 204.8 + 300 = 4{,}583.1$ s, so no
   task starts after about $39{,}930 - 4{,}583 = 35{,}347$ s.
4. **The merge.** NVARC's file holds only placeholders, so each answered output takes the
   shot's $S_1$ and $S_2$ (`merge_output`: both of the shot's answers where NVARC has no
   candidate); a task the shot does not reach, or answers with no readable block, keeps `[[0]]`.
   The file is valid at every moment; the final check validates it again.
5. **Development mode** (the version's own run): `waves.dev_cap` 16,384 lowers `max_cap` and
   `min_cap` (`for_mode`), recorded as `shot_plan.json`'s `dev_cap`; a competition rerun never
   reads it.

**Streams from the KV pool.** Each stream holds at most its opening, the cap, the forcing
reserve, and the linear-attention layers' state $s \approx 5{,}000$ tokens per request (the KV use
the logs show beyond the tokens, `docs/research/shot_tail_screen.md`):
$$n = \left\lfloor \frac{1{,}699{,}566}{6{,}771 + 65{,}536 + 6{,}144 + 5{,}000} \right\rfloor
= \left\lfloor \frac{1{,}699{,}566}{83{,}451} \right\rfloor = 20,$$
21 without $s$ (which could preempt). `plan_start` also checks each start's
$o_i + C + F$ against the free pool the boot reports.

**Coverage at 240 tasks** (**estimate**). The shot-tail screen's decode model (step
$\tau = a + b n + c\,C_{\text{KV}}$, $a = 21.55$ ms, $b = 0.913$ ms, $c = 3.78 \times 10^{-8}$ s per
context token, fitted to 2,599 engine lines of seven runs), its Kaplan-Meier thinking lengths
(85 logged turns, $P(L < 65{,}536) \approx 0.30$), and its refill simulator, rerun with 240 tasks
in opening order, 20 streams, and a window of $39{,}930 - 560 = 39{,}370$ s (the first start
near 560 s: the checks and install about 40 s, the Write cell, the wheelhouse install 114 to
145 s, the preparation 29 to 37 s, the preflight 3.4 s, the boot from the warmed copy 274 to
278 s, the sanity request: `P6.9-05`, `P4.7-02`), 1,000 draws (`runs/P5.1/coverage/`):

| Streams, planner's rate | Finished of 240 | Cancelled at the stop | Last start (s) | Idle stream-h |
|---|---:|---:|---:|---:|
| 15, 19.0 (version 9's) | 129 (123 to 135) | 0 | 33,420 | 16.1 |
| 20, 19.0 | 141 (134 to 149) | 0 | 33,110 | 21.8 |
| 20, 25 | 146 (141 to 153) | 0 | 34,120 | 14.0 |
| **20, 30 (this build)** | **147 (142 to 154)** | 0 (0 to 1) | 34,623 | 11.4 |
| 20, 35 | 148 (141 to 155) | 0 (0 to 4) | 35,035 | 9.6 |
| 20, 40 | 148 (141 to 155) | 2 (0 to 5) | 35,335 | 8.4 |
| 20, start whenever free | 145 (137 to 153) | 20 | 39,230 | 0.0 |
| 21, 30 (no linear-attention state in the pool; preemption not modelled) | 149 (141 to 157) | 0 (0 to 2) | 34,648 | 12.6 |

Median (10th to 90th percentile). A first start 300 s or 900 s later finishes 146 (140 to 153) or
144 (137 to 151). A closed form agrees: a finished task decodes about
$60{,}600$ thinking tokens (the simulation's mean), at about 293 tokens per second over 20 streams
(14.7 each at about 32k context) that is 207 s of the engine, plus about 40 s of prefill (70% of
tasks re-prefill $o + C \approx 69$k tokens for the forced reply at 1,303 tokens per second, shared
with the decodes); $39{,}370 / 247 \approx 159$, less the last start's tail (about 11 stream-hours
idle, $11.4 \cdot 3{,}600 / 20 / 247 \approx 8$ tasks): about 151. So about 147 of 240 tasks are
answered and about 93 keep placeholders. Reaching all 240 at this length would need about
$240 \cdot 60{,}600 / 39{,}370 \approx 370$ decode tokens per second, the task brief's 365.

### Script mode (task P0.5d)

A config with `kind: script` builds `kaggle_script.ipynb`, 6 markdown and 6 code cells, on the
submission notebook's stack and plumbing but with no solver, no validator, and no
`submission.json`. Let $t_0$ be the time the Config cell runs and $k$ = `SCRIPT_KILL_S`.

1. **Config**, the only cell a human edits: `ARGS` (the arguments after `--config <config>`;
   `{commit}` in one stands for the built commit's full SHA, `{input:<owner>/<slug>}` for the
   mount folder of an attached notebook output) and `SCRIPT_KILL_S`; `BUILD` adds `kind`, the
   script and its config with their sha256, and `notebook_inputs` when the build attaches any.
2. **Helpers.** A competition rerun stops here, before anything runs or is written. Then paths,
   stage bookkeeping, and eleven helpers from the builder: file search, the folder search for
   attached notebook outputs (`find_input_dirs`), the argument filler (`fill_args`), command
   runner, GPU query, `bf16_refusal`, `config_override`, file hash, `run_streamed`,
   `save_json`, and the clock reader `session_clocks` (task MT, `MT-03` revision 3). A problem
   (`problem`, or an
   exception in a stage) is recorded instead of raised: every later stage but the logs is
   skipped.
3. **Environment check.** First it logs the clocks that may see the time Kaggle spends between
   RUNNING and the notebook's first cell (up to about 41 minutes measured, `P6.5-02` and
   `P6.3-02` revision 5): `stages.environment.clocks` holds the wall time, `since_start_s`,
   `/proc/uptime`, the ages of PID 1 and of the kernel, and the ages of `/kaggle/working`'s times,
   with `errors` for any clock it cannot read; logging only, nothing depends on the values (as
   task P6.5b's submission builds). Then, as in the submission notebook (mode, build,
   interpreters, GPUs and the bf16 stop, package versions), inputs found by name: the
   evaluation challenges file under the competition's folder, the model's `config.json`, the wheel, and, in the wheel's folder
   only, the script and its config, whose sha256 must equal `BUILD`'s. Each attached notebook
   output must be exactly one folder (`INPUT_DIRS`) named by its slug, with hyphens or with
   underscores, under `/kaggle/input` or `/kaggle/usr/lib/notebooks`, counting only folders
   inside one named by its owner when there are any and one folder for each set of identical
   files; none, or folders with different files, is a problem that names the version to attach.
4. **Install**: the submission notebook's Install cell, run only while no problem stopped the run.
5. **Script**: `python <script> --config <config> <ARGS>` with the `python` on PATH, in
   `/kaggle/working`, with `PYTHONUNBUFFERED=1`, in a session of its own (`run_streamed`). `ARGS`
   are filled first (`fill_args`: `{commit}`, then each `{input:<owner>/<slug>}`; a placeholder of
   an output the build does not attach is a problem) and checked again, since a human may have
   edited them: one that sets `--config` (`config_override`) is a problem. The script's output is
   copied to the cell as it arrives; at $t_0 + k$ its process group is killed with `SIGKILL`.
   The exit code and whether the hard stop killed the script are recorded, not raised. A script
   that wrote one of the notebook's file names into `run_logs/` is a problem. Only the script file
   and its config are shipped: the script may import `arc26` (the wheel), the image's packages,
   and the standard library, but no sibling module of `scripts/`, which is not in the dataset
   folder it runs from.
6. **Run logs**, after any problem: `dump_run_logs` as in a submission build (`env.json`,
   `pip_freeze.txt`, and `candidates.json`, `{}` unless the script left candidates in
   `/kaggle/inference_outputs/`), then `stages.json` with `ok`. The notebook then ends without an
   error, so Kaggle keeps `run_logs/`; `ok` says whether the run passed.

## Why it is built this way

- **Only the wheel is uploaded** (ADR-010): the stack is run A's pinned image plus the install
  notebook at version 2. A wheelhouse is several GB, over ADR-008's cap, and could differ from run
  A's stack. `--no-deps` keeps `pyproject.toml` from touching the stack.
- **A generated notebook with plain Python cells** (A5, the task brief): no `!` or `%` magics, every
  command through `subprocess` with its exit code checked, so a failed install cannot pass
  silently, as `!pip` would.
- **Inputs found by search.** The 2026 mounts differ from 2025, and they moved within 2026. The
  CPU check of 30 Sep found the code dataset at `/kaggle/input/datasets/alyahya737/arc26-code/`,
  the competition data and the model where K0 saw them. The search is by exact file name, bounded,
  and prefers the competition's folder for the challenges file, so an attached dataset with an old
  copy cannot replace the real one. Alternative: fixed paths from K0. Lost: one mount change
  breaks a scored rerun.
- **The kernel never imports torch.** Touching CUDA in the kernel creates a context on GPU 0 and
  takes memory from solver worker 0; the reference's kernel never did before its solver finished.
  GPUs are read from `nvidia-smi`, versions from package metadata. `tests/test_notebook_local.py`
  checks that no cell before the Write cell imports torch, tensorflow, unsloth, or transformers.
- **The `python` on PATH installs and runs.** In run A the kernel ran `/usr/bin/python3` while
  `!python starter.py` ran `/usr/local/bin/python`. Installing with the kernel's `sys.executable`
  could put the package where the solver does not look. The import check proves that both see the
  same copy.
- **The validator is embedded, not imported.** It must work when the install fails (a missing or
  wrong dataset), and it must be the code the tests run, so the builder copies `arc26/submit.py`
  from the wheel itself. Alternative: import it after the install. Lost: no valid file when the
  install fails. Alternative: a smaller fallback writer in the notebook. Lost: two
  implementations of "valid".
- **A placeholder before the install, a staging file, an atomic write.** A kill runs no `finally`,
  so a valid file must exist before the first thing that can hang. The reference's writer is a plain
  `open` and `json.dump`, so its output goes to a staging file, and only `write_json_atomic`
  (temporary file, `fsync`, `os.replace`) writes `submission.json`.
- **The watchdog is `run_solver`'s timeout.** A5 asks for a watchdog that writes whatever
  predictions exist. The solver is a separate process group, so killing it is the only brake that
  frees the GPUs and hands control back; the Write cell then selects from the candidates on disk.
  A thread in the kernel could not stop CUDA work in another process. The builder refuses a hard
  stop later than $43200 - 300$ s, leaving the write at least 5 minutes (run A printed its score
  40 s after its last worker finished).
- **Competition deadlines are fixed, so development settings never cut a rerun short.** Kaggle
  reruns the submitted version with its Config cell, and if `P0.5-01` passes, the human may submit
  that very version as Submission 1. The rerun therefore ignores `RUN_SOLVER`, `SUBSET` (as the
  reference does), `ARMS`, and the development deadlines: the workers' soft deadline is 12 h minus
  30 min (A5's watchdog margin, 20 min before the reference's own 12 h minus 600 s; Kaggle reports
  up to 10 min of runtime variance), and the hard stop 12 h minus 20 min. The 10 min between them
  cover what a worker does after its DFS stops (rescoring and pickle writes); the last 20 min
  cover the write.
- **Production guard: only production builds are fit to submit** (P0.3b review, added in P1.3).
  Submissions come only from `alyahya737/arc26-submission` (`SUBMISSION_NOTEBOOK`), so a build for
  any other notebook is a development build by construction, however production-like its
  settings: the timing run's `alyahya737/arc26-timing` repeats `P0.5-01`'s settings and is still
  not for submission, and so are the CPU check's, a series', and future script notebooks. What
  the rerun takes from a build must also be production: an accelerator (on CPU the solver cannot
  start), `NvidiaL4` with four workers, one per L4, as the reference ran (one worker would reach
  about 66 hidden tasks by the deadline at `P1.1-01`'s 622 s per puzzle), valid competition
  deadlines, and no series (a series build would rerun the reference settings, but it was built
  to compare arms). `submission_blocker` lists every deviation, the notebook first;
  `MANIFEST.md` then says "no, not for submission" and the notebook's first cell
  "**Not for submission.** A development build", each with the reasons. `P0.5-01` and `P1.1-01`
  are production builds. A competition rerun records no timing in every build (the Helpers
  cell), so that needs no check per build. Builds made before the guard were labeled only when
  they had no accelerator: the `P0.3-01` build of commit `c273699` (revision 1, one worker on
  `NvidiaL4X1`) says `fit to submit | yes` and its first cell has no warning. Alternative: refuse
  to build development configs at all. Lost: development builds are pushed to their own
  notebooks and must build; the label is what keeps them from the Submit button.
- **Series mode: arms in sequence in one session** (task P1.3, BRIEF B2 Q2). G2 compares scopes
  on the same 40 tasks, the same hardware, and the same seed. One push, one queue wait, and one
  install serve every arm, and each arm is complete (all 40 tasks) before the next starts, so a
  session cut short still leaves comparable arms. Alternative: one session per arm. Lost: five
  pushes, five queue waits (`P0.1-01` run A waited over 30 minutes for 4xL4), five briefs or one
  brief over five runs. Alternative: arms in parallel, one L4 per arm. Lost: the moved starter
  gives worker `rank` GPU `rank` and one cfg to all workers, so it would need more moved-code
  changes (ADR-009 allows only run settings); no arm would finish before about 6.8 h; and a short
  session would cut every arm instead of the last in priority.
- **The stop rule** (`series_refusal`): an arm still running at the soft deadline stops taking
  tasks, so its score would cover part of DEV40 and could not be compared; an arm after a failed
  one would most likely fail the same way. The estimate is the longest arm so far times
  $m = 1.25$, a heuristic rather than a bound: the longest arm is already the estimate, the first
  arm (the notebook configuration) is the most expensive scope per step, puzzle times of equal
  runs differed by at most 9% between `P0.5-01` and `P1.1-01` (DEV4, measured), and the rest
  leaves room for a longer queue tail (one puzzle is at most 1381 s in `P1.1-01`, about 20% of a
  DEV40 arm). The spread and a worst-case tail together could exceed 25%; a larger $m$ would
  refuse arms that would likely finish. An arm that still overruns is cut at the soft deadline,
  exits 0 with puzzles missing, and does not count; a decode that branches far more is also
  bounded by the reference's 1200 s cap per puzzle. Alternative: start every arm and let the soft
  deadline cut. Lost: quota spent on an incomparable arm. Alternative: a fixed per-arm budget in
  the config. Lost: it would need the very estimate the first arm measures.
- **Each arm's own folders, the single run's layout.** The reference's selection reads every file
  in its candidate folder, and the workers append to timing files, so arms cannot share either.
  Per-arm `inference_outputs/<name>/` and `run_logs/<name>/` keep each arm's files exactly as a
  single run writes them, so `scripts/timing_report.py`, the P0.1 parity tools, and the pull's
  allowlist (nested `run_logs/` folders) work per arm unchanged.
- **Selection after the last solver.** The kernel imports `transformers` (the moved
  `arc_loader`) only in the Write cell, after every solver has finished, as before; a selection
  between arms would load model code in the kernel while later arms run. What each arm's selection
  prints also goes to its own `selection.log`, so each arm's development score is a file of its
  own, not a stretch of the session log. Alternative: select each arm in a subprocess right after
  it. Lost: a `transformers` import (about a minute on Kaggle's CPU check) per arm, and the same
  result.
- **The first arm is the notebook configuration.** `submission.json` comes from the first arm, so
  in a development run it means what it means in every other build: the reference settings'
  attempts. The builder refuses a series whose first arm differs, unless the first arm holds a
  pair out (task P2.5): a hold-out arm's attempts are a loop measurement's whatever its scope
  (`P2.1-01`'s arm had the notebook's scope and still not its attempts), and a series is never fit
  to submit, so `P2.5-01` and `P2.5-02` run their one arm at rank 16 on q, k, v, o, the scope
  P2.6 runs.
- **Hold-out as an arm setting** (task P2.1). An arm is one solver launch with its own folders
  and records, so a hold-out arm and the notebook configuration can be compared within one
  series, and a competition rerun, which ignores the series, never holds a pair out until P2.6
  decides how the loop enters production. The gate's candidates get folders of their own, beside
  `inference_outputs/`: the selection reads every file of the arm's candidate folder, and the
  gate's files carry the names of test input 0's views. Alternative: a top-level `holdout` key
  for the whole build. Lost: no on and off arms in one session, and a single run would hold a
  pair out in a rerun unless the rerun dropped the key, a second rule beside the series'.
- **Rounds as an arm setting, the governor's config embedded** (task P2.3, ADR-015). Rounds need
  the gate, so they ride inside the arm's hold-out settings, and a competition rerun, which
  ignores the series, never runs one. The governor's config cannot be read from `configs/` on
  Kaggle; the builder embeds the built commit's file, whose sha256 `MANIFEST.md` records.
  Alternative: the YAML as package data in the wheel. Lost: a second copy of the config, and a
  `pyproject.toml` change for one file.
- **Variants per arm, one notebook per run** (task P2.5, ADR-016). The variant is an arm's
  setting, so one build can force the in-context round in one arm and the gradient round in
  another. `P2.5-01` and `P2.5-02` are two configs of one arm each, built from one commit: one
  notebook per arm lets the account's two batch GPU sessions run them at once, and the two
  builds' wheels are the same bytes, so one code dataset version serves both. Alternative: one
  series of two arms. Lost: the arms would run one after the other, about 7.3 h of 4xL4 in one
  session against about 3.7 h each in parallel.
- **The solver choice** (task P2.0, ADR-011). The refinement loop needs training and decoding as
  separate calls, so `arc26.solver` runs the reference's statements moved into `arc26.ttt.fit`
  and `arc26.decode.decode_puzzle`, and `SOLVER` picks it or the moved reference. It applies in
  both modes, a competition rerun included: unlike the arms, it does not change what is computed,
  only which copy of the statements runs. The key is optional; `nvarc` was the default until run
  `P2.0-01` passed, and a separate commit flipped it to `split` (2 Oct), adding `solver: nvarc`
  to the configs of the runs before it so each still builds what ran. A production config leaves
  it out. `submission_blocker` does not read it (ADR-011 lets either solver submit), and
  `MANIFEST.md` and the first cell show it (the first cell only for a build off the default).
- **The build's scope, in both modes** (task P2.6, ADR-018). Submission 2 runs rank 16 on q, k,
  v, o, and Kaggle reruns the submitted version, so the scope must reach the rerun's solver:
  like `SOLVER`, it is a setting of what runs, not of what a development run measures, and the
  development run `P2.6-01` then measures exactly what the rerun trains (`tests/test_scope_build.py`
  checks the solver's `--config` JSON in both modes on the simulated tree). It is checked as an
  arm's scope is, with the same messages, so a scope has one spelling. Alternative: a one-arm
  series. Lost: a rerun ignores the series and trains the reference's settings, so the submitted
  run would not be the measured one. Alternative: run the first arm of a series in a rerun too.
  Lost: every series build's rerun and bytes would change, and a series is built to compare
  arms. Alternative: a new default in `arc26.runner.RunConfig`. Lost: every build's solver would
  change, the parity runs' included. A scope beside `arms` is refused: a development run would
  train the arms' scopes and the rerun the build's, which no development run trained; the
  reference's own scope is refused because leaving the key out builds exactly that. The scoped
  build's lines in fixed cells come from `SCOPE_EDITS`, each matched exactly once, so a builder
  change that moves one stops the build instead of shipping a cell without its edit; every
  other build keeps its bytes. Alternative: `SCOPE = None` in every Config cell and the Run
  cell reading it in every build. Lost: every committed build's notebook would change.
- **The hybrid build: one notebook, two stages in sequence** (task P6.5, ADR-020). The final
  picks need one tested path that runs NVARC, the agent, or both, and Kaggle reruns the submitted
  version: so the agent stage is a setting of the build, applied in both modes like `SCOPE`, and
  the pure NVARC build is the same builder without it. NVARC's four workers hold all four L4s and
  the engine needs all four (tensor parallelism 4), so the stages cannot overlap; NVARC goes first
  because its file is the fallback the agent fills in. Alternative: the agent first. Lost: NVARC's
  file would come last, and a slow agent would cost NVARC's sure tasks. Alternative: a separate
  agent notebook. Lost: a competition submission is one notebook.
- **The agent check before NVARC.** A missing wheelhouse, another revision, too few GPUs, or an
  old driver would otherwise show after NVARC's hours: a development run stops in minutes, and a
  competition rerun learns in time to give NVARC the whole session (`nvarc_deadlines` cuts only
  when `AGENT_READY`). The checks read manifests and sizes only (the weights were hashed by their
  download notebook; hashing 19.5 GB again would cost 1 to 2 minutes of 4xL4). The install itself
  stays after NVARC, as the brief's flow has it; a failed install then costs the share's rest,
  which the wheels notebook's install check on the same image makes unlikely. Alternative: install
  before NVARC. Lost: 3 to 5 minutes of NVARC's GPUs idle, or CPU contention during its decode.
- **NVARC's file first, the agent's on top, a final check last.** The agent rewrites
  `submission.json` atomically after every task with NVARC's attempts as the fill (P6.4), so the
  file is valid at every moment after NVARC's write. The notebook keeps NVARC's own copy in
  scratch and, after the agent, merges slot by slot (`merge_submission`): a corrupt or partial
  file, a development subset, or a cut temporary file never costs an NVARC attempt. Alternative:
  validate and fall back to NVARC's file as a whole. Lost: one bad slot would discard every agent
  solve.
- **A `[[0]]` counts as missing on both sides.** A run that could not read NVARC's file writes
  `[[0]]` into every slot it did not solve (the probe case), and P6.2's selection already counts
  an NVARC `[[0]]` as missing; so the merge keeps only real grids from the agent's file and fills
  the rest with NVARC's real grids in order, skipping one the entry already holds. Alternative:
  keep every valid grid of the agent's file. Lost: a failed read would replace NVARC's attempts
  with placeholders, valid and worth nothing.
- **The agent-only build falls back to NVARC after the agent, not during it.** When the agent
  stage ends with 30 minutes or more left before NVARC's soft deadline, the notebook stops the
  engine, waits for the GPUs, and runs NVARC's stage with its full deadlines (`nvarc_fallback`).
  That covers the failures that end the stage early: the install, the boot, a crash or a kill,
  an engine gone for good (P6.4's run stops when its supervisor gives up), every task dropped
  for engine errors. The trigger is the time left, not what the agent verified: the agent's
  `run.json` is written only at its exit, so a killed agent leaves none, and the GPUs would idle
  for the rest of the session either way. Alternative: run NVARC only when the agent verified
  nothing. Lost: an agent killed after its first verified program would leave hours unused.
  Alternative: a live watch that stops an agent whose engine errs above a threshold within its
  first minutes. Lost, for now: more code in the notebook's hot path and a threshold with no
  measurement behind it (the spike measures the error rate); without it, an agent that runs to
  its end but verifies little keeps the session, which the agent-only pick must weigh
  (`docs/SUBMITTING.md`, "The hybrid").
- **The fallback's file is P6.2's selection, from the attempt log.** The agent appends each
  record to `attempts.jsonl` before it rewrites `submission.json`, and `read_records` skips a
  line a kill cut, so the log holds every answer the file held; P6.2's selection over it, with
  NVARC's new attempts as its NVARC input, ranks a verified program above NVARC's attempt 1, as
  the agent's own writes do. Alternative: NVARC's selection first, the agent's grids in the
  slots it left empty. Lost: two NVARC attempts would push out a verified answer (the review's
  case: a development score of 1.0 fell to 0.0 on that task).
- **The GPUs logged and waited for before each boot.** NVARC's workers, killed at its hard stop,
  can hold GPU memory a moment after their process group is gone, and vLLM asks for 0.90 of each
  GPU's memory at its boot: a boot right after the kill could fail. So the notebook logs
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and `nvidia-smi
  --query-gpu=index,memory.used --format=csv,noheader,nounits` and repeats both for up to 60 s
  until no process is listed and no GPU uses more than 1,024 MiB, before the engine and before
  the fallback's NVARC. Both, since inside a container the process list can be empty while the
  memory is still held. 1,024 MiB is under the 2,300 MiB that vLLM's 0.90 of an L4 leaves to
  others; what an idle Kaggle L4 reports is unmeasured, and the log records it. Alternative: a
  fixed sleep. Lost: no record of what held the GPUs, and a wait either too short or always
  paid.
- **The shot after NVARC, with NVARC's deadlines** (task P6.5b). NVARC's next task is worth far
  more than the shot's (Submission 2 solves about a third of DEV120; the shot solved 4 of the 15
  tasks NVARC is least sure of), so the shot takes only the time NVARC leaves, and the plan names
  the latest start that still runs it instead of cutting NVARC. Alternative: a share cut for the
  shot. Lost: NVARC's last tasks on a slow hidden set. Alternative: run the shot on whatever time
  is left, however little. Lost: an install, a boot, and a cut wave that change nothing; below
  `min_stage_s` the stage is skipped before the install.
- **The shot gets the development solutions, the agent does not** (task P6.5b). The shot's record
  needs its own solves, the merge's gains and losses, and its waves' reading in one file
  (`shot_metrics.json`); `arc26.agent.shot` reads the solutions only after its answers and only in
  development mode, and a competition rerun's command line never names them (the audit test).
  Alternative: the notebook's `dev_scores` alone. Lost: the per-wave reading the 12 h check's
  Pass row needs.
- **DEV240 by copies under derived ids** (task P6.5b). The 12 h check must run NVARC on a rerun's
  size with development solutions, and the public evaluation set has 120 tasks; a copy under an
  8-hex id keeps every parser and scorer as they are, and its time per task equals its
  original's, so NVARC's end measures $u$ on 240. Alternative: the training set's tasks. Lost:
  another difficulty, so another time per task than the hidden set's sample DEV120 stands for.
- **SIGTERM at the end, SIGKILL `term_s` later, then the engine by its pid file.** The agent turns
  SIGTERM into its cleanup: it stops vLLM and writes once more. vLLM runs in a session of its
  own, so killing the agent's group misses it; the pid file P6.4's server writes lets the
  notebook stop it, signalling only a process that still runs the recorded argv (or the
  leftovers of its group), never a pid another program now holds. `run_supervised` also stops
  waiting 5 seconds after the agent exits if a process it left still holds its output.
  Alternative: `run_streamed` (SIGKILL at the timeout). Lost: no last write, and the engine left
  holding the GPUs.
- **Absolute `--deadline`, generation stopping `final_margin_s` before it.** The notebook passes
  $E$ as Unix time in both modes (P6.4's fix round requires it in competition mode); the run's
  boot and generation stop at $E - f$, its own stop and write take about 90 s, and the notebook
  keeps $\tau + k + w$ after $E$. $E = 42{,}000$ s is the competition hard stop's 12 h minus 20
  min, so 18 minutes stay for the kill wait, the final check, the logs, and Kaggle's runtime
  variance of up to 10 minutes. With $f = 1{,}800$ s from P6.4's config, generation stops at 11 h
  10 min; the notebook's own margins make part of $f$ redundant (an open question for the split).
- **Scratch for everything but `submission.json`.** The venv (about 10 GB), vLLM's caches, NVARC's
  copies, and the agent's logs go to `/tmp/arc26-agent` (about 1.1 TB free in every probe, K0):
  a rerun's `/kaggle/working` holds only `submission.json`, and development logs are copied to
  `run_logs/agent/` for the pull, each file cut to its last 20 MB. `ARC26_AGENT_SCRATCH` points
  the folder elsewhere, as `ARC26_KAGGLE_ROOT` points the tree.
- **The run config embedded, the inputs attached.** The wheel carries `arc26/` only, so the
  notebook carries P6.4's run config as built (its sha256 in `BUILD`), as round arms carry the
  governor's; the weights (19.5 GB) and the wheelhouse (about 4 GB) stay Kaggle notebook outputs
  attached at pinned versions, as `P6.3-02` attaches them, never uploaded from the human's machine.
- **Development scores in the notebook, never in the agent.** The notebook scores NVARC's copy
  and the final file after the agent has ended, so the agent never gets a solutions path in
  either mode, and a development run reports the agent's gain over NVARC per output.
- **The bf16 stop** (task P0.5d, `P0.3-01` finding 2). Kaggle ran a push that asked for
  `NvidiaL4X1` on its default 2xT4 without an error or a warning; T4 has compute capability 7.5
  and no bf16, so the solver installed, loaded the model, and failed at its first puzzle about 5
  minutes in. bf16 needs compute capability 8.0 (Ampere), which `nvidia-smi` reports without
  CUDA, so the kernel still imports no torch. A build made for an accelerator stops in
  development and script mode before the install; a competition rerun only logs it, because it
  must still write `submission.json` (and the scored rerun's hardware is Kaggle's choice). A GPU
  whose capability is missing or not a number counts as below 8.0: a false stop costs a push, a
  false pass the queue wait and the install. CPU builds are not checked. Alternative: trust the
  accelerator request. Lost: Kaggle ignores it silently. Alternative: let the solver fail. Lost:
  5 minutes of quota and a run that says nothing about the code. Hence every solver or bf16
  build requests `NvidiaL4`, and the builder refuses `NvidiaTeslaT4` and the TPU IDs, whose
  sessions could never pass the stop: a build for one would cost a push and a queue wait to learn
  it. Only lines of `nvidia-smi`'s output with every queried field are GPUs, so a driver warning
  on stdout cannot stop a healthy run.
- **Script mode: one builder for both notebooks** (task P0.5d). Every Phase 3 GPU run executes a
  script, not the solver, and needs exactly the submission notebook's environment: the pinned
  image, the install notebook at version 2, the offline wheel install, the model and data mounts
  found by search, internet off, a hard stop, and `run_logs/` for the pull. A `kind` key keeps the
  config rules, the gateway checks, and the cells shared; the Install cell is the submission
  notebook's own. Alternative: a builder per script. Lost: the stack and the plumbing proven by
  `P0.5-01` would be copied, and drift. Alternative: `!python` cells. Lost: A5's plain-Python
  rule, exit codes, and the hard stop.
- **The script and its config beside the wheel.** They are packaged byte for byte from the
  commit, found in the wheel's folder only (a file of the same name in another dataset is never
  run), and refused when their sha256 differs from the build's, so a run executes exactly the
  reviewed code and settings. `args` may not set `--config`, not even through an abbreviation
  argparse accepts or `-c`, since that would swap the checked config for another; the builder
  checks the config's `args`, and the Script cell checks `ARGS` again, because a human may edit
  the Config cell. An A8 script names its config `--config` and defines no other option that
  could replace it. `{commit}` in `args` passes the commit to a script that records it
  (`cache_embeddings.py --commit`): its copy in the dataset has no git checkout.
- **Only the script file is shipped.** The code dataset holds the wheel, the script, and its
  config, and the script runs from there, so it can import `arc26`, the image's packages, and the
  standard library, but no sibling module of `scripts/`. Library code a script needs belongs in
  the wheel (`arc26/`), where it is tested, versioned with the wheel's sha256, and installed
  offline. Alternative: ship `scripts/` whole. Lost: every script build would carry unreviewed
  neighbors, and their hashes would join every script's check.
- **A process group and `SIGKILL` at the hard stop.** A script can run for hours and start
  processes of its own; `subprocess.run`'s timeout kills the direct child only. `run_streamed`
  starts the script in a session of its own, copies its output as it arrives (so a killed session
  still shows it), and kills the whole group at $t_0 + k$, as `run_solver` does for the solver.
  A script keeps to its own budget below $k$ (P3.2's `max_wall_s`, exit 3); the hard stop is the
  backstop, and the session's `Timeout` must exceed $k$ by the time the logs take ($k + 300$ s,
  which `MANIFEST.md` prints).
- **Logs however the run ends.** A script notebook has no `submission.json` to protect, so a
  problem skips the remaining stages instead of raising, the Run logs cell always writes
  `run_logs/`, and the notebook ends without an error: whether Kaggle keeps the outputs of a
  failed session is unverified, and a failed run is the one whose logs matter. `ok` in
  `stages.json` says whether the run passed (no problem, exit code 0). Alternative: raise at the
  end for an `ERROR` status. Lost: possibly the logs.
- **Never submitted.** The competition is attached for its data (the DEV subsets). A script
  notebook writes no `submission.json`, so a submitted version could only fail: the builder
  refuses a script config for `alyahya737/arc26-submission`, the first cell says "Never
  submitted", `MANIFEST.md` says it, and a competition rerun stops in the Helpers cell, before
  anything runs.
- **Attached notebook outputs, found by search, passed as arguments** (task P3.5). A script that
  reads another notebook's output (`P3.5-01` reads P3.4's episodes and its leakage check) needs
  it attached at a pinned version, so a rerun reads the same files: `notebook_inputs` pins each,
  and the builder adds it to `kernel_sources` after the reference's own. Where Kaggle mounts a
  notebook output is unverified for 2026 (datasets moved under `datasets/<owner>/`): P3.4's
  checker found the builder's output by a search of `/kaggle/input`, while the install notebook,
  also a kernel source, is mounted at
  `/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`, its slug with underscores
  (P0.5-01's session). So the notebook searches both roots for both spellings, as it finds the
  model by its folder name; an output mounted in both places holds the same files and counts
  once, and two folders with different files are refused rather than guessed between. The script
  gets the folder itself through `{input:<owner>/<slug>}` in its arguments, which `stages.json`
  records with the command. Alternative: pins in `kernel_source_versions`, the same
  mapping as the reference's. Lost: a misspelled install-notebook pin would turn into an extra
  input instead of an error. Alternative: let the script search `/kaggle/input` itself. Lost: two
  outputs could match one search, and the folder used would show only in the script's own logs.
  The placeholder is filled before `config_override` runs, as `{commit}` is, so no input folder
  can set `--config`; the version of a mounted output is not visible in its path, so the script
  checks the content (P3.5-01 checks the manifest's sha256 and the report's `pass`).
- **The CPU check runs `scripts/env_probe.py`** (`configs/script_smoke.yaml`). The brief's first
  choice, `scripts/cache_embeddings.py --smoke`, is on `task/P3.2` and not in this branch.
  `env_probe.py` is equally small: standard library only, about a minute on Kaggle (K0), and its
  reports land in `run_logs/`. It reads no config and ignores unknown arguments by design, so the
  smoke config passes itself as the script's config, and the check still exercises the config's
  packaging, search, sha256 check, and `--config` argument. Once P3.2 is merged,
  `cache_embeddings.py` with `configs/cache_embeddings.yaml` and `args: ["--smoke"]` is the
  stronger check, since it imports `arc26` and the GPU stack.
- **Development deadlines** (`configs/submission_P0.5-01.yaml`): $s_\text{end} = 3600$ s is over
  twice run A's 1568 s session, so the soft deadline never binds, exactly as run A's own (12 h
  minus 600 s) never bound; parity is kept. $s_\text{kill} = 4500$ s leaves 900 s after the soft
  deadline, more than run A's longest TTT training (831 s), so workers stop on their own first.
  `configs/submission_P1.1-01.yaml` uses the competition values, so its timing transfers to the
  scored rerun.
- **Salvage after a kill.** One unreadable candidate pickle makes the reference's selection fail
  for every task; moving only the files that fail as truncated or corrupt bz2 pickles (never on
  another error, such as a missing numpy) and selecting again keeps every other task's attempts.
  The moved code stays untouched.
- **Stale files are removed before the solver starts.** Candidates or worker markers left by an
  earlier run in an interactive session would mix into the new run, and a stale marker lets a
  worker start before the previous one has imported unsloth (the reference's workaround). A fresh
  Kaggle session has none, so batch runs are unaffected.
- **The launch probe** (`configs/submission_cpu_check.yaml`): the solver runs on the CPU image with
  one worker and one task. Its worker fails at `from unsloth import` for lack of a GPU, and that
  failure proves the rest on Kaggle's image before any GPU time: `python -m arc26.solver` (the
  default solver; `arc26.nvarc.starter` for a config that states `nvarc`) with the installed wheel,
  the environment `run_solver` sets, the spawned worker, and the notebook's handling of a failed
  solver.
- **Development fails loudly, competition carries on** (the task brief). A development run is for
  finding problems; a rerun is scored only if it writes a valid file.
- **Timing records in development only** (task P0.3b, BRIEF B4). Every development run writes
  the records the P1.1 table, the cost model, and the P2.4 governor need, directly under
  `run_logs/`, so the pull's allowlist (`run_logs/**/*.jsonl`) brings them back without log
  scraping and without a copy step. A competition rerun passes no timing folder: its path is
  the one validated before timing existed, and its output stays `submission.json` alone. There is
  no config switch: timing costs well under 1% of a puzzle (`docs/modules/timing.md`, Known
  limits), and every development run should yield the table.
- **Run logs as run A recorded them.** The same `dump_run_logs` gives files that compare directly
  with run A's (`env.json` packages and image, `candidates.json`), which the P0.1 parity analysis
  needs. `stages.json` adds each stage's wall time, errors, the solver's exit code, and the
  submission report. In a rerun the notebook writes nothing but `submission.json` to
  `/kaggle/working`: extra outputs can be picked by mistake in the Submit dialog (research note,
  P0.5 section). The solver's own caches (unsloth's compiled cache in its working directory, as
  in the reference) may still appear there.
- **Reproducible builds.** A clean tree plus `git archive` means nothing uncommitted or ignored
  reaches the wheel; `SOURCE_DATE_EPOCH` fixes the wheel's timestamps; nothing in the notebook or
  the manifest depends on the build time. The epoch comes from the last commit that changed the
  wheel's inputs, not from `HEAD`: a later commit that touches only docs (a pre-run brief, say),
  configs, or scripts leaves the wheel's sha256 unchanged, so the code dataset needs no new
  version. uv builds offline first because the human's connection is slow; hatchling honors
  `SOURCE_DATE_EPOCH`.
- **The local test runs cells in a plain interpreter.** A5 names `nbclient`; it needs ipykernel and
  pyzmq, which the task brief excludes (disk and connection). The test runs each code cell in one
  shared namespace, like a kernel, with `-S` so only the installed wheel can provide `arc26`, a
  recording `python` shim that installs into a temporary target, and an audit hook on file opens.
  It proves the plumbing, not the score.

- **The weights' warm-up (task P6.9c) copies to local disk during NVARC.** Alternatives
  considered, and why each lost:
  - *Only `--safetensors-load-strategy prefetch` on the input mount.* vLLM 0.31.0 prefetches each
    rank's share of `sorted_files[rank::world_size]` in a background thread
    (`_prefetch_all_checkpoints`, `weight_utils.py` lines 959 to 1031): with 2 safetensors files
    and 4 ranks, the 18.6 GB `model.safetensors` is one sequential reader, inside the 900 s boot.
    Its time on GCSFuse is unmeasured. It stays the fallback, where nothing better is left.
  - *`--safetensors-load-strategy eager`.* Each worker reads whole files into CPU memory
    (`load(f.read())`, lines 1126 to 1128): 4 ranks times 18.6 GB of RAM, from the same mount, no
    faster.
  - *A longer boot limit.* `P6.9-02`'s engine would have been healthy near 1,077 s
    (**estimate**, its record); a limit near 1,200 s costs every rerun minutes of the shot's
    window and still rides on the mount's speed, which changed once already between 19 and 23
    Oct.
  - *Copying in the agent stage.* The copy would then take its minutes from the shot's window
    after NVARC; during NVARC's 8.3 h it costs nothing.
  - *One `shutil.copyfile` per file.* One sequential reader for the 18.6 GB file; the piece copy
    has `workers` readers on one file and needs no tools outside the standard library.
  - *Hashing while copying.* Pieces end out of order, so a streaming sha256 needs ordered reads;
    the hash after the copy reads the files just written (page cache: 135.82 GiB of RAM free in
    `P6.9-02`'s engine log) and checks the bytes on disk, not the bytes read.
  - *A module in `arc26` (`python -m arc26.agent.warmup`).* It would change the wheel and so the
    code dataset; in the builder the wheel stays `a5bb333f`, byte for byte (ADR-010: only the
    wheel is uploaded; the reproducible-builds bullet above).
  - *The idle IO class (`ionice -c 3`).* It can starve the copy while another process keeps the
    disk busy; best-effort at level 7 yields to NVARC and still progresses.
  - *Falling back at once when the copy is unfinished.* The fallback reads the same bytes from the
    same mount through fewer readers, so a copy still moving ends first: wait while it progresses,
    up to the boot limit.
  - *A development-only shot end for the short run `P6.9-03`* (to cap its quota): the scored
    config would then carry a key `P6.9-02`'s does not, against the task's rule that it equal
    `P6.9-02`'s apart from the warm-up; the short run uses the competition's times and a short
    subset instead, and its length is set by the shot's turns (`docs/prerun/P6.9-03.md`, section
    6).

- **MTP with one speculative token and the 69,632 cap (task P6.9d).** Alternatives considered,
  and why each lost:
  - *MTP at the 94,208 cap.* 15 streams need 1,606,845 tokens of KV at it, over MTP's pool
    estimate of 1,467,099 (`docs/experiments/P6.10-01.md`, "KV pool"), so the plan would run 13
    streams; the cap's top 24k tokens bought 1 of 7 solves (`docs/tasks/P6.9d.md`).
  - *The largest cap the plan fits at 15 streams, 84,891* (`P6.10-01`'s record). With MTP's
    measured per-request blocks (6.53 of 1,588.3 tokens) the need is
    $15\,(6{,}771 + 84{,}891 + 6{,}144 + 10{,}372) \approx 1{,}622{,}664$, over the pool: vLLM
    would preempt late in the turns. 69,632 leaves about 73,000 tokens (5%) and is the cap
    `P6.10-02`'s brief derived its time from.
  - *MTP with $k \ge 2$.* vLLM runs the one MTP layer $k$ times (its warning, `vllm/config/
    speculative.py` lines 1370 to 1379) and TRITON_ATTN reads the KV 2 to 3 times per step; only
    $k = 1$ was measured (`P6.10-01`).
  - *The plan's rate raised to MTP's (for example $19.0 \cdot 1.21$).* Starts would come later
    and at longer caps, but MTP's rate at contexts near 70k on Kaggle is unmeasured (`P6.10-02`
    never booted) and a plan above the real rate cuts turns at the generation stop; the base's
    rate keeps every planned start inside its time.
  - *A lower `min_cap`.* It changes the shot's minimum stage and the notebook's `min_stage_s`, a
    third change, and short caps never solved (0 of 15 at 26,469 in `P6.5-03`; ADR-023's
    decision 3).
  - *Prefix caching off for vLLM issue #55766* (NaN outputs with speculation and prefix
    caching). A third change; `P6.10-01`'s MTP step counted 0 NaN lines and 0 degenerate replies
    in 30 turns, and the shot's degenerate check keeps NVARC's attempts for any such reply.
  - *A development subset of 16 or more shot tasks*, to run the refill path on a GPU. NVARC's
    stage and the shot's turns would pass the task's 1.5 h and 3 quota h; `P6.9-03`'s 4 tasks
    keep this run comparable with `P6.9-03`'s on the same tasks and seed.

- **The 69,632 cap without MTP (run P6.9-05).** Alternatives considered, and why each lost:
  - *Keep MTP (`P6.9-04`'s version 7).* `P6.9-04`'s engine log shows MTP slower than the base at
    long context on Kaggle's 4 x L4: one stream (`97d7923e` alone, its context near 65k to 70k)
    decoded 21.1 to 22.7 tokens per second (`runs/P6.9-04/run_logs/agent/engine.log`, the
    `Running: 1 reqs` lines from 20:50:15 to 20:51:55), against 38.9 to 41.3 for one stream from
    about 48k to 82k without MTP (`runs/P6.9-03/run_logs/agent/engine.log`, 12:53:56 to
    13:05:36). Acceptance was not the cause: accepted over drafted 95,133 / 115,444 = 0.824 over
    the 267 SpecDecoding lines, so a step emitted about 1.82 tokens and took about
    $1.82 / 21.8 \approx 83$ ms against the base's $1 / 39.5 \approx 25$ ms. MTP gained only at
    short context (3 streams at 163.8 tokens per second near the start, 20:08:55), the range
    `P6.10-01` measured (contexts up to about 18k). The whole turns: 26.4 to 29.8 tokens per
    second per stream at 3 streams (`engine_calls.jsonl`) against `P6.9-03`'s 38.9 and 39.4 at 2.
  - *Version 6 unchanged (94,208).* The cap's top 24,576 tokens bought 1 of 7 long-attempt solves
    (`docs/tasks/P6.9d.md`); the cut frees streams for refills in the same window.
  - *A GPU check of this build before the submission.* Its engine and flags are version 6's
    (`P6.9-03` Pass) and its cap ran on GPU in `P6.9-04`; the only new element, the config, is
    loaded on Kaggle's image by the CPU twin, so a check run would cost about 2.5 quota h for no
    new evidence. The version's own run (the 4-task development subset) still runs at the push.
  - *MTP with prefix caching off, or with another attention backend.* Untested changes in the
    last week before the deadline; the cap alone needs no new engine measurement.

- **A rank-256 pass on rank 16's confidence-0 tasks (run P4.7-01).** Alternatives considered,
  and why each lost:
  - *The screen's smallest form, all 23 DEV120 tasks at confidence 0* (about
    $23 \times 554$ GPU s, 0.9 h of the machine). The core question is whether `P1.1-01`'s three
    confidence-0 solves (`65b59efc_0`, `80a900e0_0`, `9385bd28_1`) come back; the 8 tasks hold
    them and the 5 others where rank 256 gave candidates, at about a third of the cost. The other
    15 stay the Borderline row's option (about 1.5 quota h).
  - *Both DEV240 halves (about 38 tasks, 1.5 h of the machine).* Two replicates of the rescoring
    seed for about 3 quota h, before the core is known (`docs/research/second_pass_screen.md`,
    "What follows").
  - *`solver: nvarc`, `P1.1-01`'s exact code.* A variant (task P4.7b) would run in Submission 4's
    build, whose NVARC is the split solver; it runs the same statements
    (`tests/test_split_solver.py`, ADR-011), and `P2.0-01` ran it at these settings on 4xL4
    (Pass, per-puzzle totals within 13% of `P1.1-01`'s). So this run measures the code that would
    ship.
  - *`r16-proj` instead of rank 256.* It gave candidates on the same empty outputs in `P1.3-01`
    but has no solve measured there, and its 567 s per puzzle against 606 s saves little (the
    screen, "What follows").
  - *A second rank-256 pass inside Submission 4's build (the variant itself).* It needs a second
    pass and a pooled selection in the builder before the core is known; that is task P4.7b.
  - *Another `PYTHONHASHSEED`, for a draw with other rescoring views.* One fixed seed for all
    comparisons (CLAUDE.md), and a final build would run seed 0; the run-to-run nondeterminism at
    seed 0 is what a final build would see on these tasks.
  - *`arc26-submission` as the notebook.* Its versions are the submissions (the task's rule); a
    build for any other notebook is a development build by construction (`submission_blocker`).
  - *`NvidiaL4X1` or `NvidiaTeslaT4`.* T4 has no bf16, and `NvidiaL4X1` sessions run on a T4
    (measured 30 Sep); four workers need four GPUs.
  - *The competition deadlines (41,400 s and 42,000 s).* The `Timeout` would be about 12 h, an
    upper bound of 24 quota h for a run of about half an hour; 2,700 s and 3,300 s cover the
    queue's worst case above and bound the run at 6,600 s.
  - *A twin with the CPU checks' one task (`aa4ec2a5`).* The probe's worker fails before it reads
    a task, so the run's own 8 ids cost nothing and the twin queues exactly the run's tasks.

- **The rank-256 tail (task P4.7b).** Alternatives considered, and why each lost:
  - *A standalone rank-256 run first (`P4.7-01`).* The human's call of 26 Oct: quota is short, so
    each idea becomes a submission whose own run is a short check; `P4.7-01` is built and not run,
    and its question moves into `P4.7-02`'s own run.
  - *The tail before the shot.* The shot's $(0, 0.5)$ bucket yields 0.19 tasks per task against
    rank 256's 0.09 per confidence-0 task (+1.80 of about 20), and version 8 finishes all 17 of
    that bucket's tasks with about 3,000 s of the machine idle at the end
    (`shot_tail_screen.md`), so the tail goes into the idle time, not in the shot's way.
  - *The shot keeps confidence 0 in its queue.* It reaches about 1 of 38 such tasks at 240 tasks
    (`shot_tail_screen.md`) at 0.066 each, 0.53 tasks per wall hour against rank 256's 2.4
    (`second_pass_screen.md`); leaving them to the tail costs the shot about 0.07 tasks.
  - *`arc26.solver` itself with the tail's tasks as its subset.* It sorts the queue by id, so the
    deadline would drop tasks by id, not by cost; its entry block is the reference starter's, AST
    for AST (`tests/test_split_solver.py`), so it stays as is, and `arc26.tail` reruns its workers
    with the file's order.
  - *Renamed task ids that sort by cost.* The rescoring views' seed is `hash(bk) % 1024**2` with
    the task id in `bk`, so renaming changes the views against `P1.1-01`'s; the file order keeps
    them.
  - *Ordering by a size proxy instead of NVARC's seconds.* NVARC's `puzzle` records measure the
    cost per task directly; a competition rerun wrote none before, so a tail build writes them to
    scratch (`/kaggle/arc26_timing/`), a file append per op that changes no result.
  - *Pooling the tail's candidates with NVARC's.* The screen's pooled score equals the switch
    score on these tasks (`second_pass_screen.md`, rank 16 then rank 256 at confidence 0: both
    $+1.80$), and the placeholder-only merge can never replace an answer NVARC or the shot gave.
  - *The shot stage holding the engine until its generation stop.* Measured not to happen: the
    refill queue returns once nothing runs and nothing can start (above), so no early-return code
    was needed in `arc26/agent/shot.py`; the notebook adds only the check of an empty queue before
    the install, which saves the install and the warm-up wait.
  - *A development-only tail deadline for the own run.* The tail's deadline is the rerun's
    formula in both modes; the own run's length is set by its two tasks (as `P6.9-03`'s by its
    shot), and the brief's `Timeout` bounds it.

- **LLM first (task P5.1).** Alternatives considered, and why each lost:
  - *Keep NVARC first (version 9).* NVARC takes about 8.35 h of the 12 (`P6.9-02`: its end at
    30,085.5 s), so the shot gets about 2.6 h at about 285 tokens per second and starts about 18
    tasks (`shot_tail_screen.md`); the human's call of 28 Oct, after the competition's discussion
    (leading teams run Qwen3.8 on every task), is to give the 27B the whole session. Version 9
    stays submittable on its own code dataset.
  - *An agent-only build through `nvarc_share_s: 0`.* Its skip is recorded as "agent only", which
    starts the agent-only fallback (`nvarc_fallback`) when the stage ends 1,800 s or more before
    41,400 s: that merge reads `arc26.agent.run`'s `attempts.jsonl`, which the shot does not
    write, so NVARC's attempts would replace the shot's answers, and NVARC's hard stop at
    42,000 s would end past the shot build's 40,500 s plan with the 2,700 s pre-script allowance.
    `shot_plan` also refuses a share of 0, and changing it would change version 9's embedded
    helpers. A new key with its own edits keeps every earlier cell.
  - *NVARC after the shot on what it left.* The shot runs to its generation stop by design
    (39,930 s), so no NVARC time is left; a fallback for an early shot failure would need its own
    merge and deadlines inside 40,500 s, untested and not in the task brief.
  - *Order by the shot's own estimate of difficulty, or by id.* There is no confidence without
    NVARC; the opening bound is known before any request, and the smallest prompts start first
    with the least prefill and KV. Whether small prompts are also easier is unmeasured.
  - *15 streams (version 9's) or 21.* 15 finishes 129 (123 to 135) of 240 at version 9's rate
    against 141 at 20 (the screen's decode model: each step's fixed 21.55 ms is shared by more
    streams); 21 streams fit the pool only without the linear-attention state, and vLLM would
    preempt near the cap.
  - *Version 9's planner's rate 19.0, or a higher one.* 19.0 stops starts about 1,500 s earlier
    and finishes 141; 30 finishes 147 with 0 (0 to 1) cancelled; 35 and 40 add at most 1 finish
    at more cancelled starts, whose compute slows the streams that would finish; starting
    whenever a stream is free cancels 20 and finishes 145. The screen's own answer for version 8
    was 30 too.
  - *A cap below 65,536 so more tasks start.* Short caps never solved: `P6.5-03`'s 26,469 solved
    0 of 15, third parties' 32k budgets 1.0 of 36 against 3.5 to 6.0 at 64k
    (`beyond_h_options.md`, 1.6); `min_cap` stays 65,536.
  - *A separate development build for the check.* The version's own run is the check of the
    exact notebook the human submits; the development cap (`waves.dev_cap`) is the one setting
    that differs, read only in development mode, so the check takes about 30 minutes instead of
    hours.

## How to run it

From the repository root, with the tree committed:

```bash
.venv/bin/ruff check . && .venv/bin/pytest -m "not gpu"
.venv/bin/python scripts/build_submission.py --tag P0.5-cpu --config configs/submission_cpu_check.yaml
.venv/bin/python scripts/build_submission.py --tag P0.5-01 --config configs/submission_P0.5-01.yaml
.venv/bin/python scripts/build_submission.py --tag P1.1-01 --config configs/submission_P1.1-01.yaml
```

Builds from one commit share one wheel, so one dataset version serves all three notebooks, and
the `P0.5-01` and `P1.1-01` briefs can share one `Code-commit`.

**Kaggle CPU check** (no quota, no pre-run brief; the notebook `alyahya737/arc26-submission-cpu-check`
is named in no brief):

```bash
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/P0.5-cpu/arc26-code   # first time
.venv/bin/python scripts/kaggle_ops.py push dist/P0.5-cpu --timeout 1800
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check  # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P0.5-cpu
```

Later builds version the dataset instead:
`scripts/kaggle_ops.py dataset-version dist/<tag>/arc26-code -m "<tag> <commit>"`. Wait a few
minutes after a dataset action before pushing, until Kaggle has processed the new version. Never
version the dataset over a version a pending GPU run expects without telling the orchestrator: the
notebook refuses a wheel whose sha256 differs from its build.

**GPU smoke `P0.5-01`**, only after the human approves `docs/prerun/P0.5-01.md`, whose `Timeout`
must be at least $s_\text{kill} + 900 = 5400$ s:

```bash
.venv/bin/python scripts/kaggle_ops.py push dist/P0.5-01 --run-id P0.5-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P0.5-01
```

**DEV120 run `P1.1-01`**, the same way under its own approved brief, whose `Timeout` must be 43200
s (the hard stop is at 42000 s). Pull `P0.5-01` before pushing `P1.1-01`: both use the notebook
`alyahya737/arc26-submission`, and kaggle-cli 2.2.4 pulls only the latest run.

**Timing run `P0.3-01`** (DEV4 with `P0.5-01`'s settings: four workers on `NvidiaL4`, deadlines
3600 s and 4500 s; its own notebook `alyahya737/arc26-timing`), only after the human approves
`docs/prerun/P0.3-01.md`, whose `Timeout` is $s_\text{kill} + 900 = 5400$ s. A development build,
never to be submitted (submissions come from `alyahya737/arc26-submission` only); from task P1.3
on its `MANIFEST.md` says "no, not for submission" for that reason (the production guard), while
builds made before the guard say "fit to submit: yes". Revision 1 asked for `NvidiaL4X1`; Kaggle
silently ran it on 2xT4, which has no bf16, and the solver failed at its first puzzle
(`docs/experiments/P0.3-01.md`), so no solver run uses a smaller accelerator than `NvidiaL4`. Its
wheel differs from the `P0.5-01` one, so the code dataset needs a version from a P0.3b build first
(version 3):

```bash
.venv/bin/python scripts/build_submission.py --tag P0.3-01 --config configs/submission_P0.3-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P0.3-01/arc26-code -m "P0.3-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P0.3-01 --run-id P0.3-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-timing
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-timing --task-id P0.3-01
.venv/bin/python scripts/timing_report.py runs/P0.3-01/run_logs/timing \
    --out reports/P0.3-01-timing.md --title "P0.3-01 timing"
```

**Scope series `P1.3-01`** (DEV40, five arms on `NvidiaL4`, its own notebook
`alyahya737/arc26-lora-scope`), only after the human approves `docs/prerun/P1.3-01.md`, whose
`Timeout` is 43200 s (the hard stop is at 42000 s). A development build, never to be submitted
(a series). Its wheel differs from `P0.3-01`'s (the LoRA settings), so the code dataset needs a
new version from this build first; the per-arm timing tables come from each arm's folder:

```bash
.venv/bin/python scripts/build_submission.py --tag P1.3-01 --config configs/submission_P1.3-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P1.3-01/arc26-code -m "P1.3-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P1.3-01 --run-id P1.3-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-lora-scope
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-lora-scope --task-id P1.3-01
for arm in r256-proj-emb r16-attn r32-attn r16-proj r64-proj; do
    .venv/bin/python scripts/timing_report.py runs/P1.3-01/run_logs/$arm/timing \
        --out reports/P1.3-01/$arm-timing.md --title "P1.3-01 $arm timing"
done
```

To add or drop an arm, edit `arms` in the config (the first arm stays the notebook
configuration, or a hold-out arm, task P2.5), commit, rebuild, and amend the brief: the arms are
part of `Code-commit`.

**Split-solver parity run `P2.0-01`** (DEV4 with `P0.5-01`'s settings and `solver: split`; its
own notebook `alyahya737/arc26-split-solver`), only after the human approves
`docs/prerun/P2.0-01.md`, whose `Timeout` is $s_\text{kill} + 900 = 5400$ s. A development
build, never to be submitted (its notebook). Its wheel adds `arc26/solver.py`, `ttt.py`, and
`decode.py`, so the code dataset needs a new version from this build first; version it only
after `P1.3-01` has finished, or with the human's go-ahead, since a running or queued notebook
may expect the previous version:

```bash
.venv/bin/python scripts/build_submission.py --tag P2.0-01 --config configs/submission_P2.0-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P2.0-01/arc26-code -m "P2.0-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P2.0-01 --run-id P2.0-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-split-solver
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-split-solver --task-id P2.0-01
.venv/bin/python -m arc26.evaluate agree runs/P2.0-01/submission.json \
    tests/fixtures/p0_1_reference_dev4.json --subset DEV4
.venv/bin/python scripts/timing_report.py runs/P2.0-01/run_logs/timing \
    --out reports/P2.0-01-timing.md --title "P2.0-01 timing"
```

**Held-out gate price `P2.1-01`** (DEV40, one hold-out arm on `NvidiaL4` with `solver: split`;
its own notebook `alyahya737/arc26-holdout-gate`), only after `P2.0-01` passes and the human
approves `docs/prerun/P2.1-01.md`, whose `Timeout` is $s_\text{kill} + 900 = 16200$ s. A
development build, never to be submitted (its notebook, a series). Its wheel adds
`arc26/loop.py`, so the code dataset needs a new version from this build; version it only when no
queued or running notebook expects the current version (`P2.0-01`, `P3.5-01`):

```bash
.venv/bin/python scripts/build_submission.py --tag P2.1-01 --config configs/submission_P2.1-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P2.1-01/arc26-code -m "P2.1-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P2.1-01 --run-id P2.1-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-holdout-gate
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-holdout-gate --task-id P2.1-01
.venv/bin/python scripts/timing_report.py runs/P2.1-01/run_logs/r256-proj-emb-holdout/timing \
    --out reports/P2.1-01/r256-proj-emb-holdout-timing.md --title "P2.1-01 hold-out timing"
```

**One refinement round `P2.3-01`** (DEV4, one round arm on `NvidiaL4` with `solver: split`; its
own notebook `alyahya737/arc26-refine-rounds`), only after the human approves
`docs/prerun/P2.3-01.md`, whose `Timeout` is $s_\text{kill} + 900 = 19800$ s. A development
build, never to be submitted (its notebook, a series). Its wheel adds the rounds to
`arc26/loop.py`, `SGDProvider` to `arc26/ttt.py`, and the re-decode term to
`arc26/governor.py`, so the code dataset needs a new version from this build; version it only
when no queued or running notebook expects the current version:

```bash
.venv/bin/python scripts/build_submission.py --tag P2.3-01 --config configs/submission_P2.3-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P2.3-01/arc26-code -m "P2.3-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P2.3-01 --run-id P2.3-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-refine-rounds
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-refine-rounds --task-id P2.3-01
.venv/bin/python scripts/timing_report.py runs/P2.3-01/run_logs/r256-proj-emb-rounds/timing \
    --out reports/P2.3-01/r256-proj-emb-rounds-timing.md --title "P2.3-01 round timing"
```

**The in-context and the gradient round on DEV40, `P2.5-01` and `P2.5-02`** (one round arm each,
rank 16 on q, k, v, o, on `NvidiaL4` with `solver: split`; their own notebooks
`alyahya737/arc26-round-in-context` and
`alyahya737/arc26-round-gradient`), only after the human approves `docs/prerun/P2.5-01.md` and
`docs/prerun/P2.5-02.md`, whose `Timeout` is $s_\text{kill} + 900 = 43200$ s (the soft deadline is
the competition's, 41,400 s, for the governor's budgets). Development builds, never to be
submitted. Both builds come from one commit and carry the same wheel: the code dataset gets one
version, from either folder, when no queued or running notebook expects the current version;
then both pushes, which take the account's two batch GPU sessions:

```bash
.venv/bin/python scripts/build_submission.py --tag P2.5-01 --config configs/submission_P2.5-01.yaml
.venv/bin/python scripts/build_submission.py --tag P2.5-02 --config configs/submission_P2.5-02.yaml
cmp dist/P2.5-01/arc26-code/arc26-0.0.1-py3-none-any.whl dist/P2.5-02/arc26-code/arc26-0.0.1-py3-none-any.whl
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P2.5-01/arc26-code -m "P2.5-01 P2.5-02 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P2.5-01 --run-id P2.5-01
.venv/bin/python scripts/kaggle_ops.py push dist/P2.5-02 --run-id P2.5-02
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-round-in-context
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-round-gradient
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-round-in-context --task-id P2.5-01
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-round-gradient --task-id P2.5-02
.venv/bin/python scripts/timing_report.py runs/P2.5-01/run_logs/r16-attn-in-context/timing \
    --out reports/P2.5-01/r16-attn-in-context-timing.md --title "P2.5-01 in-context round timing"
.venv/bin/python scripts/timing_report.py runs/P2.5-02/run_logs/r16-attn-gradient/timing \
    --out reports/P2.5-02/r16-attn-gradient-timing.md --title "P2.5-02 gradient round timing"
```

**Submission 2, `P2.6-01`** (DEV120 at rank 16 on q, k, v, o in both modes, `NvidiaL4`, four
workers, the competition deadlines; the submission notebook `alyahya737/arc26-submission`), only
after the human approves `docs/prerun/P2.6-01.md`, whose `Timeout` is 43200 s (the hard stop is
at 42000 s). Fit to submit: its `MANIFEST.md` says `fit to submit | yes`, and the version this
run makes is the one the human submits (`docs/SUBMITTING.md`). Its code dataset is its own,
`alyahya737/arc26-code-p26`, so it never waits for a version that runs of other branches expect;
its first version comes from `dataset-create`, and the push waits until Kaggle has processed it:

```bash
.venv/bin/python scripts/build_submission.py --tag P2.6-01 --config configs/submission_P2.6-01.yaml
sha256sum dist/P2.6-01/kaggle_submission.ipynb   # must print the brief's hash, else rebuild at its Code-commit
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/P2.6-01/arc26-code
.venv/bin/python scripts/kaggle_ops.py push dist/P2.6-01 --run-id P2.6-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P2.6-01
.venv/bin/python scripts/timing_report.py runs/P2.6-01/run_logs/timing \
    --out reports/P2.6-01/timing_table.md --title "P2.6-01 timing"
```

The push makes a new version of the submission notebook; `P1.1-01`'s version (Submission 1's
candidate) stays on Kaggle and stays submittable.

**The hybrid's measurement, `P6.5-01`** (task P6.5: the hybrid at a sixth of its times on DEV40,
Submission 2's NVARC until 3,600 s, then the agent solver; `NvidiaL4`, its own development
notebook `alyahya737/arc26-hybrid-dev`), only after the orchestrator has confirmed the split
(`nvarc_share_s`: the hybrid's placeholder of 6 h, scaled to 3,600 s, until the spike `P6.3-01`,
`P6.3-02` reports; this run's share is the hybrid's times 40/240), the spike's gate has passed
(ADR-020), and the human has approved `docs/prerun/P6.5-01.md`, whose `Timeout` is 10,800 s
(`MANIFEST.md` asks for at least 10,250 s). Never submitted: its `MANIFEST.md` says `fit to
submit | no, not for submission: ...` (its notebook, and an agent that ends at 9,800 s). Its code
dataset is the lineage's own, `alyahya737/arc26-code-p65`, which the hybrid, `P6.5-agent`, and
`P6.5-cpu` share (one commit, one wheel). The two download outputs are attached by
`kernel-metadata.json` at version 1, as `P6.3-02` attaches them. First the free CPU check of the
same commit, then the GPU run:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.5-cpu --config configs/submission_P6.5-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P6.5-01 --config configs/submission_P6.5-01.yaml
cmp dist/P6.5-cpu/arc26-code/arc26-0.0.1-py3-none-any.whl dist/P6.5-01/arc26-code/arc26-0.0.1-py3-none-any.whl
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/P6.5-01/arc26-code   # arc26-code-p65, first version
# wait a few minutes until Kaggle has processed the dataset
.venv/bin/python scripts/kaggle_ops.py push dist/P6.5-cpu --timeout 4050
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P6.5-cpu
# after the CPU check's Good reading and the brief's approval:
sha256sum dist/P6.5-01/kaggle_submission.ipynb   # must print the brief's hash, else rebuild at its Code-commit
.venv/bin/python scripts/kaggle_ops.py push dist/P6.5-01 --run-id P6.5-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-hybrid-dev
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-hybrid-dev --task-id P6.5-01
.venv/bin/python scripts/timing_report.py runs/P6.5-01/run_logs/timing \
    --out reports/P6.5-01/timing_table.md --title "P6.5-01 NVARC timing"
```

**The hybrid's submission build** (`configs/submission_P6.5-hybrid.yaml`, the submission notebook
`alyahya737/arc26-submission`, fit to submit), on `P6.5-01`'s Lift row, with the spike's share,
after the human approves its own brief (a run id after `P6.5-01`, not drafted yet), whose
`Timeout` is 43,200 s (`MANIFEST.md` asks for at least 42,450 s). Its development run (DEV120)
lasts until the agent's generation stop, since P6.4's last phase cycles until then: about 11.2 h
of 4xL4. Its push makes the version to submit; `P2.6-01`'s version (Submission 2) stays
submittable:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.5-hybrid --config configs/submission_P6.5-hybrid.yaml
sha256sum dist/P6.5-hybrid/kaggle_submission.ipynb   # must print its brief's hash
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P6.5-hybrid/arc26-code -m "P6.5-hybrid <commit>"   # when its commit is not P6.5-01's
.venv/bin/python scripts/kaggle_ops.py push dist/P6.5-hybrid --run-id <run-id>
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id <run-id>
```

The agent-only build builds the same way (`--tag P6.5-agent --config
configs/submission_P6.5-agent.yaml`); it has no run of its own yet, and its risks (the fallback
runs NVARC after a failed install or boot, a crash or a kill, or an engine gone, verified
programs kept first; an agent that runs to its end but verifies little leaves no NVARC attempt)
are in the hybrid build above. On a plain 4-GPU machine the
hybrid does not apply as a notebook (it needs Kaggle's mounts); its two stages are the solver's
(`python -m arc26.solver`) and P6.4's run entry (`docs/modules/agent_run.md`), run one after the
other.

**Submission 3's runs, `P6.5-02-cpu`, `P6.5-02`, `P6.5-03`** (task P6.5b, the shot build: NVARC
with its deadlines, then the long shot; one commit, one wheel, the code dataset
`alyahya737/arc26-code-p65`, which the orchestrator creates or versions from this commit's build).
First the free CPU twin of the smoke (`alyahya737/arc26-submission-cpu-check`, named in no brief),
then the DEV4 smoke under `docs/prerun/P6.5-02.md` (`alyahya737/arc26-hybrid-dev`, `Timeout`
6,600 s; `MANIFEST.md` asks for at least 5,750 s), then, only after the smoke's Pass, the 12 h
check under `docs/prerun/P6.5-03.md` (`alyahya737/arc26-submission`, `Timeout` 43,200 s; at least
40,500 s), whose version is Submission 3 if it passes:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.5-02-cpu --config configs/submission_P6.5-02-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P6.5-02 --config configs/submission_P6.5-02.yaml
.venv/bin/python scripts/build_submission.py --tag P6.5-03 --config configs/submission_P6.5-03.yaml
cmp dist/P6.5-02/arc26-code/arc26-0.0.1-py3-none-any.whl dist/P6.5-03/arc26-code/arc26-0.0.1-py3-none-any.whl
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P6.5-03/arc26-code -m "P6.5b <commit>"   # dataset-create if arc26-code-p65 does not exist yet
# wait until Kaggle has processed the version (kaggle datasets status, the human's CLI)
.venv/bin/python scripts/kaggle_ops.py push dist/P6.5-02-cpu --timeout 4050
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P6.5-02-cpu
# after the CPU twin's Good reading and P6.5-02's approval:
sha256sum dist/P6.5-02/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P6.5-02 --run-id P6.5-02
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-hybrid-dev
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-hybrid-dev --task-id P6.5-02
# after P6.5-02's Pass and P6.5-03's approval:
sha256sum dist/P6.5-03/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P6.5-03 --run-id P6.5-03
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P6.5-03
.venv/bin/python scripts/timing_report.py runs/P6.5-03/run_logs/timing \
    --out reports/P6.5-03/timing_table.md --title "P6.5-03 NVARC timing on DEV240"
```

**Submission 4's runs, `P6.9-02-cpu` and `P6.9-02`** (task P6.9: Submission 3's NVARC and times,
then shot v2; one commit, one wheel, the new code dataset `alyahya737/arc26-code-p69`, which the
orchestrator creates from this commit's build; Submission 3's `arc26-code-p65` stays as it is,
since a rerun of its version reads it). First the free CPU twin, then, at or after the Sat 24 Oct
03:00 quota reset and on approval, the 12 h check under `docs/prerun/P6.9-02.md`
(`alyahya737/arc26-submission`, `Timeout` 43,200 s; at least 40,500 s), whose version is
Submission 4 if it passes:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.9-02-cpu --config configs/submission_P6.9-02-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P6.9-02 --config configs/submission_P6.9-02.yaml
cmp dist/P6.9-02-cpu/arc26-code/arc26-0.0.1-py3-none-any.whl dist/P6.9-02/arc26-code/arc26-0.0.1-py3-none-any.whl
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/P6.9-02/arc26-code   # dataset-version -m "P6.9-02 <commit>" once it exists
# wait until Kaggle has processed the dataset (kaggle datasets status, the human's CLI)
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-02-cpu --timeout 7650
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P6.9-02-cpu
# after the CPU twin's Good reading, P6.9-02's approval, and the Sat 24 Oct 03:00 reset:
sha256sum dist/P6.9-02/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-02 --run-id P6.9-02
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P6.9-02
.venv/bin/python scripts/timing_report.py runs/P6.9-02/run_logs/timing \
    --out reports/P6.9-02/timing_table.md --title "P6.9-02 NVARC timing on DEV240"
```

**Submission 4 with the warm-up, `P6.9-03-cpu` and `P6.9-03`** (task P6.9c; one commit, one
wheel, byte for byte `P6.9-02`'s, so `alyahya737/arc26-code-p69` needs no new version). First the
free CPU twin, then, on approval of `docs/prerun/P6.9-03.md` (`Timeout` 9,200 s), the short run
whose version is Submission 4 on Pass:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.9-03-cpu --config configs/submission_P6.9-03-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P6.9-03 --config configs/submission_P6.9-03.yaml
cmp dist/P6.9-03/arc26-code/arc26-0.0.1-py3-none-any.whl dist/P6.9-02/arc26-code/arc26-0.0.1-py3-none-any.whl
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-03-cpu --timeout 7650
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P6.9-03-cpu
# after the twin's Good reading and P6.9-03's approval:
sha256sum dist/P6.9-03/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-03 --run-id P6.9-03
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P6.9-03
```

**MTP and the 69,632 cap, `P6.9-04-cpu` and `P6.9-04`** (task P6.9d; one commit, one wheel, byte
for byte `P6.9-02`'s, so `alyahya737/arc26-code-p69` needs no new version). First the free CPU
twin, then, on approval of `docs/prerun/P6.9-04.md`, the short run whose version is a leaderboard
test on Pass:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.9-04-cpu --config configs/submission_P6.9-04-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P6.9-04 --config configs/submission_P6.9-04.yaml
cmp dist/P6.9-04/arc26-code/arc26-0.0.1-py3-none-any.whl dist/P6.9-03/arc26-code/arc26-0.0.1-py3-none-any.whl
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-04-cpu --timeout 7650
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P6.9-04-cpu
# after the twin's Good reading and P6.9-04's approval:
sha256sum dist/P6.9-04/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-04 --run-id P6.9-04
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P6.9-04
```

**The 69,632 cap without MTP, `P6.9-05-cpu` and `P6.9-05`** (variant 2 of task P6.9d; the
wheel `a5bb333f`, so `alyahya737/arc26-code-p69` needs no new version). First the free CPU twin,
then, on approval of `docs/prerun/P6.9-05.md`, the push whose version (version 8 if no other push
comes first) the human submits as the day's leaderboard test:

```bash
.venv/bin/python scripts/build_submission.py --tag P6.9-05-cpu --config configs/submission_P6.9-05-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P6.9-05 --config configs/submission_P6.9-05.yaml
cmp dist/P6.9-05/arc26-code/arc26-0.0.1-py3-none-any.whl dist/P6.9-03/arc26-code/arc26-0.0.1-py3-none-any.whl
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-05-cpu --timeout 7650
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P6.9-05-cpu
# after the twin's Good reading and P6.9-05's approval:
sha256sum dist/P6.9-05/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P6.9-05 --run-id P6.9-05
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P6.9-05
```

**A rank-256 pass on rank 16's confidence-0 tasks, `P4.7-01-cpu` and `P4.7-01`** (task P4.7;
the wheel `a5bb333f`, so `alyahya737/arc26-code-p69` needs no new version). First the free CPU
twin, when no other version of `arc26-submission-cpu-check` is queued or running; then, on
approval of `docs/prerun/P4.7-01.md`, the GPU run on its own new notebook (its first push creates
it):

```bash
.venv/bin/python scripts/build_submission.py --tag P4.7-01-cpu --config configs/submission_P4.7-01-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P4.7-01 --config configs/submission_P4.7-01.yaml
sha256sum dist/P4.7-01/arc26-code/arc26-0.0.1-py3-none-any.whl   # a5bb333f..., the brief's
.venv/bin/python scripts/kaggle_ops.py push dist/P4.7-01-cpu --timeout 1800
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P4.7-01-cpu
# after the twin's Pass and P4.7-01's approval:
sha256sum dist/P4.7-01/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P4.7-01 --run-id P4.7-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p4-7-r256
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p4-7-r256 --task-id P4.7-01
```

**The rank-256 tail, `P4.7-02-cpu` and `P4.7-02`** (task P4.7b; a new wheel, so
`alyahya737/arc26-code-p69` needs a new version from this build, made when no queued or running
notebook expects the current one). First the free CPU twin, then, on approval of
`docs/prerun/P4.7-02.md`, the push whose version the human submits as a leaderboard test:

```bash
.venv/bin/python scripts/build_submission.py --tag P4.7-02-cpu --config configs/submission_P4.7-02-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P4.7-02 --config configs/submission_P4.7-02.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P4.7-02/arc26-code -m "P4.7-02 <commit>"
# wait until Kaggle has processed the version (kaggle datasets status, the human's CLI)
.venv/bin/python scripts/kaggle_ops.py push dist/P4.7-02-cpu --timeout 7650
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P4.7-02-cpu
# after the twin's Pass and P4.7-02's approval:
sha256sum dist/P4.7-02/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P4.7-02 --run-id P4.7-02
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P4.7-02
```

**LLM first, `P5.1-01-cpu` and `P5.1-01`** (task P5.1; a new wheel on its own new code dataset
`alyahya737/arc26-code-p51`, so versions 6 to 9 keep `arc26-code-p69` as built). First the free
CPU twin, then, on approval of `docs/prerun/P5.1-01.md`, the push whose version the human submits
as a leaderboard test:

```bash
.venv/bin/python scripts/build_submission.py --tag P5.1-01-cpu --config configs/submission_P5.1-01-cpu.yaml
.venv/bin/python scripts/build_submission.py --tag P5.1-01 --config configs/submission_P5.1-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/P5.1-01/arc26-code
# wait until Kaggle has processed the dataset (kaggle datasets status, the human's CLI)
.venv/bin/python scripts/kaggle_ops.py push dist/P5.1-01-cpu --timeout 7650
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission-cpu-check   # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission-cpu-check --task-id P5.1-01-cpu
# after the twin's Pass and P5.1-01's approval:
sha256sum dist/P5.1-01/kaggle_submission.ipynb   # must print the brief's hash
.venv/bin/python scripts/kaggle_ops.py push dist/P5.1-01 --run-id P5.1-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-submission
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-submission --task-id P5.1-01
```

**Script smoke** (CPU, no quota, no pre-run brief; the notebook `alyahya737/arc26-script-smoke` is
named in no brief). The code dataset version must come from this build, since it holds
`env_probe.py` and `script_smoke.yaml` beside the wheel; the wheel equals every other build's of
the same code, so a submission notebook still finds its own. Version the dataset only when no
queued or running notebook expects the current version:

```bash
.venv/bin/python scripts/build_submission.py --tag script-smoke --config configs/script_smoke.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/script-smoke/arc26-code -m "script-smoke <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/script-smoke --timeout 1800
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-script-smoke  # until complete
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-script-smoke --task-id script-smoke
```

**A GPU script notebook** (a script config with `accelerator: NvidiaL4`, for example for
`P3.2-01`) runs the same way under its approved brief: `push dist/<tag> --run-id <run-id>`, with
the brief's `Timeout` at least $k + 300$ s (`MANIFEST.md` prints the value). `P3.5-01`
(`configs/script_P3.5-01.yaml`, `Timeout` 15900 s) attaches the P3.4 builder and checker outputs
at version 2; its wheel and script are new, so the code dataset needs a version from this build:

```bash
.venv/bin/python scripts/build_submission.py --tag P3.5-01 --config configs/script_P3.5-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P3.5-01/arc26-code -m "P3.5-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P3.5-01 --run-id P3.5-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p3-5-train
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p3-5-train --task-id P3.5-01
```

`P3.5-02` (`configs/script_P3.5-02.yaml`, `Timeout` 8700 s) is `P3.5-01`'s build with its own toy
config (`train_hypernet_toy_P3.5-02.yaml`, half the steps) and hard stop (8400 s), pushed as the
next version of the same notebook. Its wheel is `main`'s, the one `P3.2-02`'s build carries, so one
dataset version holds both builds' `arc26-code/` files (the same wheel, scripts and configs of
distinct names, and either build's `dataset-metadata.json`, which differ only in the description
naming the script) and serves both notebooks; the commands are in
`docs/runbooks/train_hypernet.md`, section 4.

Ask for `NvidiaL4`
whenever the script needs bf16, and pass the GPU count the script uses in its own config or
`args`: Kaggle does not grant `NvidiaL4X1`, and its default 2xT4 has no bf16.

What to attach: nothing by hand; `kernel-metadata.json` attaches the competition, the model, the
install notebook, the code dataset, and a script build's `notebook_inputs`. A plain GPU machine
does not apply: the notebook needs
Kaggle's mounts and the attached install notebook. The CPU path is `tests/test_notebook_local.py`.

Expected cost:

| Step | Wall time | Quota | Source |
|---|---|---|---|
| Local build | about 1 s | none | measured |
| CPU check with the launch probe | 3 min (177 s session, 150 s of notebook) | none | measured 30 Sep, `runs/P0.5-cpu/`: install 50 s (mostly the tensorflow uninstall), probe 34 s, write 61 s (the kernel's transformers import), logs 5 s |
| GPU smoke on `NvidiaL4` | about 27 min, plus the queue | about 1 h (4xL4 counts 2x) | run A 1568 s, run B 1689 s |
| DEV120 run on `NvidiaL4` | 6 to 8 h (**estimate**), capped at 11 h 40 min | 12 to 16 h | `docs/STATE.md` week-1 plan |
| DEV4 timing run on `NvidiaL4` | about 0.45 h (**estimate**: `P0.5-01`'s session, 1516 s, same settings) | about 0.9 h (4xL4 counts 2x) | `docs/prerun/P0.3-01.md`, section 6 |
| DEV40 scope series on `NvidiaL4`, five arms | 8.9 to 9.6 h (**estimate**: about 6800 s for the first arm, `P1.1-01`'s DEV40 puzzle spans in a four-worker queue, and 6200 to 6900 s for each smaller scope), at most 11 h 40 min | 18 to 19 h (2x) | `docs/prerun/P1.3-01.md`, section 6 |
| DEV4 split-solver parity on `NvidiaL4` | about 0.45 h (**estimate**: `P0.5-01`'s session, 1516 s, same settings and statements) | about 1 h (4xL4 counts 2x) | `docs/prerun/P2.0-01.md`, section 6 |
| DEV40 held-out gate price on `NvidiaL4`, one arm | about 2.7 h, 3.4 h if decode and the gate run 30% slower (**estimate**: `P1.3-01` arm 1's puzzles with a shorter fit and one more decoded input each, in a four-worker queue), at most 4.5 h | about 5.5 h, at most 9 h (2x) | `docs/prerun/P2.1-01.md`, section 6 |
| DEV120 Submission 2 `P2.6-01` on `NvidiaL4`, rank 16 on q, k, v, o | about 4.35 h, 5.2 h if every puzzle runs 20% slower (**estimate**: `P1.1-01`'s puzzles with `P1.3-01`'s arm 2 to arm 1 ratios per op, replayed in the four-worker queue), at most 11 h 40 min | about 4.4 h at 1x, 8.7 h at 2x (10.4 h in the slower case) | `docs/prerun/P2.6-01.md`, section 6 |
| Hybrid CPU check `P6.5-cpu` | about 15 min (**estimate**: the CPU check's 3 min, the wheelhouse install on 4 CPUs, 3 to 10 min as the wheels notebook's install check, two failed engine boots, 1 to 2 min) | none | `runs/P0.5-cpu/`, `docs/runbooks/agent_spike.md` |
| DEV40 hybrid `P6.5-01` on `NvidiaL4`, a sixth of the hybrid's times | about 2.26 h (**estimate**: NVARC until its cut puzzles end, about 3,800 s; the write, the install, and the boot, about 4,750 s; generation until 8,000 s, since P6.4's last phase cycles until its stop; the stop, the final check, the scores, and the logs, about 8,150 s); at most 10,250 s | about 2.3 h at 1x, 4.5 h at 2x; at most 2.9 h and 5.7 h | `docs/prerun/P6.5-01.md`, section 6 |
| DEV120 hybrid submission build (`configs/submission_P6.5-hybrid.yaml`) on `NvidiaL4` | about 11.2 h (**estimate**: NVARC as `P2.6-01`, its session 4.4 h, then the agent from about 15,800 s until its generation stop at 40,200 s, then its stop, the final check, and the logs); at most 42,450 s | about 11.2 h at 1x, 22.4 h at 2x; at most 11.8 h and 23.6 h | its own brief, after `P6.5-01` |
| Script smoke on CPU (`env_probe.py`) | about 3 min (**estimate**: the install as in the CPU check of 30 Sep, 50 s, and the probe as in K0, about a minute) | none | `runs/P0.5-cpu/`, `docs/experiments/K0.md` |
| Rank-256 pass on 8 DEV120 tasks `P4.7-01` on `NvidiaL4` | about 0.56 h, 0.45 to 0.8 h (**estimate**: `P2.0-01`'s session, 1,771 s for a longest puzzle of 1,231 s, with this queue's makespan of 1,289 s, the save, and a pre-script gap of 13 to 861 s), at most 6,600 s | about 1.1 h at 2x, 0.9 to 1.6; at most 3.7 h | `docs/prerun/P4.7-01.md`, section 6 |
| Version 8 plus the rank-256 tail on 3 DEV120 tasks `P4.7-02` on `NvidiaL4` | about 0.6 h, 0.5 to 0.9 h (**estimate**: NVARC about 450 to 730 s, the tail's two tasks at rank 256 about 1,060 to 1,340 s, the rest about 200 s, a pre-script gap of 13 to 861 s, the save), at most 6,600 s | about 1.2 h at 2x, 1.0 to 1.8; at most 3.7 h | `docs/prerun/P4.7-02.md`, section 6 |
| H's toy meta-training `P3.5-01` on `NvidiaL4` | about 2.2 h (**estimate**), the script's own stops at 4 h | about 4.4 h (4xL4 counts 2x) | `docs/prerun/P3.5-01.md`, section 6 |
| The same with half the steps, `P3.5-02`, on `NvidiaL4` | about 1.0 h (**estimate** from `P3.5-01`'s measured 6446 s session), the script's own stops at 2 h | about 2.1 h (4xL4 counts 2x) | `docs/prerun/P3.5-02.md`, section 6 |

GPU memory is the reference's: a peak of 13.0 GB in training and 17.1 GB in inference per L4 of
22 GB (third-party logs, unverified; `docs/runbooks/build_reference_run.md`). Disk: the build
folder is about 60 KB.

## Outputs

Build folder `dist/<tag>/` (gitignored):

| File | Contents |
|---|---|
| `kaggle_submission.ipynb` | The notebook above, nbformat 4.5, 8 markdown and 8 code cells, no Kaggle editor settings. |
| `kernel-metadata.json` | `id`, `title` (the slug with spaces), `code_file`, private, internet off, `enable_gpu` and, with an accelerator, `machine_shape`; the pinned image (`original`); competition, model, code dataset, install notebook `/2`. |
| `arc26-code/arc26-0.0.1-py3-none-any.whl` | The package, built from `git archive`. |
| `arc26-code/dataset-metadata.json` | `id`, title `arc26 code`, license `other` (Apache-2.0 moved code plus project code whose license G5 decides), a description. |
| `MANIFEST.md` | Provenance, file hashes, Kaggle commands, config summary with the arms, the solver, and a scoped build's `scope (both modes)` row, fit to submit ("yes", or "no, not for submission" with every reason). |

An LLM-first build (task P5.1) writes the same files as a shot build with the warm-up and no tail; its `MANIFEST.md`'s `agent entry` row names the shot on every task, and a development run's `run_logs/agent/shot_plan.json` holds `order` `opening`, `dev_cap`, and one start per task.

A hybrid build (task P6.5) writes the same files: its notebook has 10 markdown and 10 code
cells (the agent check and the agent stage added), its `kernel-metadata.json` attaches the two
download outputs after the install notebook (`<owner>/<slug>/<version>`), and its `MANIFEST.md`
adds the agent rows and the `Timeout` the session needs. The code dataset holds the wheel alone:
the agent's run config is embedded in the notebook.

A script build writes `kaggle_script.ipynb` (6 markdown and 6 code cells) instead of the
submission notebook, and `arc26-code/` also holds the script and its config under their file
names (`env_probe.py` and `script_smoke.yaml` for the smoke). Its `MANIFEST.md` lists them with
their sha256, says "no, a script notebook is never submitted", and gives the config summary
`kind`, `accelerator`, `script`, `script_config`, `args`, `script_kill_s`, `model_slug`; its
`dataset-metadata.json` description names the script.

On Kaggle, in `/kaggle/working/`, pulled into `runs/<task-id>/`:

| File | Contents |
|---|---|
| `submission.json` | Always valid for the challenges file of the mode: in development all 120 evaluation tasks (unsolved ones `[[0]]`), in a rerun every test task. |
| `run_logs/candidates.json` | Development only: every candidate the solver wrote to `/kaggle/inference_outputs/`, keyed by file name: grid, `beam_score`, `score_aug`. |
| `run_logs/env.json` | Development only: Python, image build (`/etc/git_commit`, `/etc/build_date`), `KAGGLE_*` variables, GPUs, package versions, `sys.path`, `BUILD`. |
| `run_logs/pip_freeze.txt` | Development only: `pip freeze` of the `python` on PATH. |
| `run_logs/stages.json` | Development only: mode, `BUILD`, the Config values (`settings`, with `solver`), per stage `start_s`, `seconds`, and any `error`; the run stage's `solver_end_s`, `solver_kill_s`, `launch_env`, `solver_exit`, `solver_error`, and in a scoped build `scope` (task P2.6); the write stage's `make_submission_error` and `quarantined` files; `problems`; the submission report; `timing_dir`. |
| `run_logs/timing/timing-rank<rank>.jsonl` | Development only: one per solver worker, one JSON line per op, `{puzzle_id, op, seconds, tokens}`: `load` (null `puzzle_id`), then per puzzle `train`, `decode` (per batch), `selection` (per rescored candidate), and `puzzle` (`docs/modules/timing.md`). About 60 lines for DEV4; about 2100 for DEV120 across four workers (`P1.1-01`'s 669 decode batches and 1181 rescored candidates). |
| `run_logs/timing/timing-main.jsonl` | Development only: one `main_selection` line from `make_submission`, unless the selection raised (as it does with no correct candidate, for example with `RUN_SOLVER = False`). |

A hybrid build adds, in a development run (a competition rerun writes `submission.json` alone;
everything else stays in `/tmp/arc26-agent`):

| File | Contents |
|---|---|
| `submission.json` | The final check's file: the agent's real grids, NVARC's in every slot the agent left missing, invalid, or `[[0]]`, `[[0]]` where neither has a grid; after an agent-only fallback, NVARC's selection first and the agent's grids in its empty slots; valid for the challenges file of the mode. |
| `run_logs/stages.json` | As above, plus the stages `agent_check` (`settings`, the `plan`, the `weights` and `wheelhouse` checked, `gpus`, `tensor_parallel_size`, `driver`, `scratch_free_bytes`, `ready`, or `skipped`) and `agent` (`settings`, `start_s`, `generation_stop_s`, `candidates` (the JSON written for the agent), `install` (`exit`, `ended`, `seconds`, `ok`, `problems`, `imports`, `engine`), `command`, `gpus_before_engine` (`before` and `after`, the `nvidia-smi --query-compute-apps` rows `pid` and `used_memory`; `memory_before` and `memory_after`, the `--query-gpu=index,memory.used` rows `index` and `used_mib`; each null without `nvidia-smi`; `used_max_mib`; `seconds` waited), `agent_exit`, `ended` (`exited`, `terminated`, `killed`), `run_s`, `engine_stopped`, `run` (from the agent's `run.json`: `problems`, `submission` with its slots by source, `engine`, `scheduler`, `deadline_in_s`, `wall_s`), in an agent-only build `nvarc_fallback` (`start_s`, `solver_end_s`, `solver_kill_s`, then `skipped` with the reason, or `gpus_before`, `engine_stopped`, `gpus` (as `gpus_before_engine`), `solver_exit`, NVARC's selection's `make_submission_error` and `quarantined`, then `merge` when that selection wrote no file, or `selection` (P6.2's `select_files` report: `tasks`, `test_inputs`, `records` and `skipped_lines` of the attempt log, `slots` by source `program`, `direct`, `nvarc`, `empty`) and `sources` (the agent's grids added in empty slots)) or `fallback_error`, `removed`, `final` (the final check: `rewritten`, `sources` with `changed`, `valid`), `score` (`nvarc` and `final`, each `score`, `tasks`, `solved_outputs`, `test_outputs`, `solved`; `gained`, `lost`), `logs`, or `skipped` or `failure` with the reason); the run stage's `solver_end_s` and `solver_kill_s` are NVARC's cut deadlines; top-level `submission` is the final check's report. |
| `run_logs/agent/attempts.jsonl`, `run.json`, `timing-engine.jsonl`, `transcripts.jsonl`, `engine.log` | The agent's log folder (`docs/modules/agent_run.md`), each file cut to its last 20 MB. |
| `run_logs/agent/install.json`, `install.log` | The wheelhouse installer's report (`ok`, `problems`, steps, imports with torch's CUDA) and log. |

A shot build (task P6.5b) writes the hybrid's files; in `stages.json` its `agent_check.plan` is
`shot_plan`'s (`nvarc_end_s`, `nvarc_kill_s`, `latest_start_s`, `generation_stop_s`,
`agent_end_s`, `agent_kill_s`, `notebook_end_s`, `problems`), its `agent.command` runs
`arc26.agent.shot`, and `run_logs/agent/` holds the shot's log folder
(`docs/modules/agent_shot.md`): `run.json`, `shot_plan.json` (per wave: tasks, $C$, planned and
actual end), `shot_attempts.jsonl` (per task: answers, tokens, finish, forcing),
`engine_calls.jsonl`, `prompt_counts.jsonl`, `boot.json`, `engine_prep.json`, the engine log,
`shot_metrics.json` (development only: the hybrid's score, NVARC's alone, the shot's solves, the
merge's gains and losses), and `shot_transcripts.jsonl` (development only), each cut to its last
20 MB. A DEV240 build adds `settings.dev240`, and its `submission.json` covers the 240 tasks.

In series mode the single run's `run_logs/` files above are per arm instead:

| File | Contents |
|---|---|
| `submission.json` | The first arm's attempts (the notebook configuration, or a hold-out arm's, task P2.5), valid as above. |
| `run_logs/stages.json` | As above, plus `settings.arms` (the Config cell's arms), `stages.run.arms` (per arm, in order: `name`, `lora_r`, `lora_targets`, `reason`, a hold-out arm's `holdout`, `status` `ran`, `skipped`, or `not started`, and `why`, `start_s`, `seconds`, `solver_exit`, `launch_env`, `solver_error`), `stages.run.series_margin`, and `stages.write.arms` (per selected arm: `seconds`, `make_submission_error`, `quarantined`, its submission report); `timing_dir` is null. |
| `run_logs/<arm>/candidates.json`, `env.json`, `pip_freeze.txt` | `dump_run_logs` on `/kaggle/inference_outputs/<arm>/`, as for a single run. |
| `run_logs/<arm>/selection.log` | What the arm's selection printed: the `ALL_CORRECT` candidates, both scorers' `acc:` lines, and `*** Reload score:`, the arm's DEV40 score in tasks (or the traceback if scoring raised). |
| `run_logs/<arm>/submission.json` | The arm's own attempts, repaired and validated like `submission.json`. |
| `run_logs/<arm>/stages.json` | The arm's run record, its write record, and its `timing_dir`. |
| `run_logs/<arm>/timing/timing-rank<rank>.jsonl`, `timing-main.jsonl` | The arm's timing records, as for a single run; a hold-out arm adds per puzzle `gate_decode`, `gate_selection`, and `gate`. |
| `run_logs/<arm>/gate/gate-rank<rank>.jsonl` | Hold-out arms only (task P2.1): one verdict per puzzle, `{puzzle_id, applied, passed, real_pairs, held_out_index, views, candidates, votes, expected_rank, seconds, error}` (`docs/modules/loop.md`). The gate's candidate pickles stay on Kaggle in `/kaggle/gate_outputs/<arm>/`. |
| `run_logs/<arm>/gate/plan-`, `promote-`, `round-`, `decode-rank<rank>.jsonl` | Round arms only (task P2.3): the governor's plan per round (`RoundPlan` with its `round` and the `shape` it priced), one promotion record per test input and round, one record per round (`outcome`, `screened`, `promoted`, `decision`, `estimate`, `remaining`, `weights_before`, `weights_after`, `restored`, `seconds`, `error`), and one per test decode after the rounds (`name`, `prompts`, `inputs`, `installed`, `batches`, `expected`, `cut`, `skipped`, `error`) (`docs/modules/loop.md`). The round files stay on Kaggle in `/kaggle/round_outputs/<arm>/`. Task P2.5 adds `context-rank<rank>.jsonl`, one line per decode whose prompts left out pseudo-pairs (`name`, `prompts`, `queries`, `kept`, `dropped`, `lengths`, `limit`), none when nothing was left out; a round's `decision` may be `in_context`, and its plan's `shape` then holds `in_context` (`screened`, `too_long`, `redecode_batches`, `length_ratio`). |

A script notebook writes no `submission.json`. Its `/kaggle/working/run_logs/` holds:

| File | Contents |
|---|---|
| the script's own files | What the script writes into `run_logs/` (BRIEF A8: resolved config, commit, timing, metrics, `summary.md`); `env_probe.json` and `env_probe.md` for the smoke. |
| `candidates.json` | `{}`, unless the script left candidate pickles in `/kaggle/inference_outputs/`. |
| `env.json`, `pip_freeze.txt` | As for a submission build (`dump_run_logs`), `BUILD` with the script's and its config's sha256. |
| `stages.json` | `mode` `script`, `BUILD`, `settings` (`args`, `script_kill_s`), per stage `start_s`, `seconds`, `error`, and `skipped`; the script stage's `command`, `launch_env`, `script_kill_s`, `script_exit`, and `killed`; `problems`; `script_exit`; `ok` (no problem and exit code 0); `total_s`. |

**The weights' warm-up (task P6.9c).** `run_logs/agent/warmup.json` (development runs; in
scratch in a rerun): `state` (`waiting`, `copying`, `verifying`, `done`), `ok`, `errors`, `pid`,
`source`, `target`, `files`, `bytes`, `manifest` (name and size per file), `workers`,
`piece_bytes`, `delay_s`, `priority` (`nice`, `ionice`), the epoch seconds `created`, `start`,
`end`, the spans `copy_s`, `verify_s`, `seconds` (from `start`), `rate_mb_s`, the counters
`copied_bytes` and `hashed_bytes`, and the lists `verified` (sha256 checked) and `size_only` (no
sha256 in the manifest). `run_logs/agent/warmup.log`: the copier's own output (empty when all is
well). In `stages.json`: `stages.agent_check.warmup` (`started`, `pid`, `target`, `status`,
`log`, `files`, `bytes`, `free_bytes`, or `reason`), `stages.agent.weights` (`model_dir`,
`source` `local` or `input`, `load_strategy`, `waited_s`, `reason`, `warmup`), and
`stages.agent.warmup` (`stop_warmup` at the stage's end: `exit`, `stopped`, `status`). The copy
itself, 19.5 GB, stays in `/tmp/arc26-agent/weights/` on Kaggle.

**MTP (task P6.9d).** No new file. `run_logs/agent/engine.log` holds vLLM's periodic
`SpecDecoding metrics:` lines (mean acceptance length, accepted and drafted tokens, the per-position
acceptance rate); `run.json` `engine.command` holds `--speculative-config` and its JSON; `boot.json`
`kv_pool` is MTP's pool; `shot_plan.json` `model.max_cap` is 69,632 and each `waves` row's `cap`
at most that.

## How to read the results

**CPU check (`runs/P0.5-cpu/`).**

- Good: `stages.json` lists `environment`, `install`, `run`, `write`, `run_logs` with no `error`,
  and `problems` is empty. The launch probe started the solver (the session log shows
  `python -m arc26.solver`; `python -m arc26.nvarc.starter` in checks built before 2 Oct), its
  worker failed at `from unsloth import` for lack of a GPU
  (or the hard stop killed it), and `solver_exit` is non-zero. The submission report says `valid`,
  120 tasks, 172 test inputs, `raw_status` `read`, 344 zero attempts, and
  `write.make_submission_error` is the `ValueError` from the reference's
  `benchmark_selection_algos` on no candidates, which is expected. `env.json` shows run A's image
  and packages, and `pip_freeze.txt` equals run A's plus one `arc26 @ file://...` line (pip names
  a package installed from a file by its path). Next: the P0.5-01 pre-run brief.
- Borderline: all of that, but the session log shows the kernel and the `python` on PATH as
  different installations, or package versions differ from run A's. Fix the build or record the
  difference in the P0.5-01 brief before any GPU run.
- Bad: the notebook stopped in the environment check or the install (a missing input, a wrong
  sha256, a pip error, an import mismatch), or the probe failed before its worker: for example
  `No module named arc26` or a JSON or argument error from `starter.py`. Read the printed search
  results, pip output, and solver output, fix, rebuild, run again. Never go to the GPU run with a
  failing CPU check.

**GPU smoke (`runs/P0.5-01/`).** The decision rule is the approved brief's. The inputs: DEV4
attempts in `submission.json` against the fixture, $g(E, A)$ and the `attempt_1` agreement,
`solver_exit` 0, the DEV4 score the reference prints, `env.json` against run A's (image, packages,
`PYTHONHASHSEED` "0", 4 L4s), `launch_env` in `stages.json` (the solver's `PYTHONHASHSEED` "0" and
the reference's launch variables), and stage times against run A's 1568 s. The analysis goes to
`docs/experiments/P0.5.md`.

**Timing records (every development GPU run from P0.3b on, first `runs/P0.3-01/`).** Run
`scripts/timing_report.py runs/<task-id>/run_logs/timing`. Good: every puzzle of the subset has a
`train`, `puzzle`, at least one `decode`, and (when its DFS found a grid) `selection` records;
each worker has one `load`; `timing-main.jsonl` has one `main_selection`; no skipped lines; each
`puzzle` record is within about a second of the session log's `finished <task> in <s>s`; train
tokens are not null. Borderline: null train tokens (the trainer's prepared dataset had no
`input_ids` column: the table holds, the train cost model does not) or a puzzle without its
`puzzle` record (a crashed worker or the hard stop). Bad: no timing files although the run solved
puzzles (check `timing_dir` in `stages.json`).

**Scope series (`runs/P1.3-01/`).** The decision rule is the approved brief's (G2). Read
`stages.run.arms` in `run_logs/stages.json` first: which arms ran, their wall time and exit, and
why any arm did not start. Per arm that ran: the DEV40 score is the `*** Reload score:` line of
`run_logs/<arm>/selection.log` (tasks of 40, per-task credit; the `score_kgmon` `acc:` line is the
submitted scorer), the per-output credit follows from its `ALL_CORRECT` lines, `candidates.json`,
and `submission.json` as for `P1.1-01` (no solutions file is read locally), and train and decode
seconds per puzzle come from `scripts/timing_report.py runs/P1.3-01/run_logs/<arm>/timing`.

An arm counts when it ran with `solver_exit` 0 and has a `puzzle` record for every DEV40 task.
An arm cut at the soft deadline exits 0 with puzzles missing: it did not fail, it does not count.

- Good: every arm counts.
- Incomplete: arm 1 counts and a later arm does not (not started by the stop rule or after a
  failed arm, failed, or cut at the soft deadline): G2 decides on the arms that count if they
  settle the rule, otherwise the rest run in a new series under an amended brief (the brief's
  Incomplete row).
- Bad: arm 1 does not count (failed, or cut at the soft deadline or the hard stop), or the session
  fails as a whole (the brief's Bad row).

**Split-solver parity (`runs/P2.0-01/`).** The decision rule is the approved brief's. First
check that the split solver ran: the environment check printed `solver code: split`, and
`settings.solver` in `stages.json` is `split` (a traceback would name `arc26/solver.py`,
`ttt.py`, or `decode.py`). Then the inputs, as for the GPU smoke: $g(E', A)$ and the `attempt_1`
agreement from `arc26.evaluate agree` against the fixture, `solver_exit` 0, `env.json` against
run A's, and the timing records, complete for every DEV4 puzzle by the brief's definition
(`scripts/timing_report.py`, then the `decoding` and `scoring` lines of the session log per
puzzle). The analysis goes to `docs/experiments/P2.0-01.md`.

**Held-out gate price (`runs/P2.1-01/`).** The decision rule is the approved brief's. Read
`stages.run.arms` first: the arm counts when it ran with `solver_exit` 0 and has a `puzzle` record
for every DEV40 task. Then the arm's DEV40 score, the `*** Reload score:` line of
`run_logs/r256-proj-emb-holdout/selection.log`, against `P1.3-01` arm 1's 8.5 of 40, and per
output the `score_kgmon` correct set of the same log. The gate: `run_logs/<arm>/gate/` holds one
verdict per puzzle (40 lines over the four files); its pass rate is `passed` over `applied`, and
each verdict joins the selection log's correct outputs by `puzzle_id` for the brief's table of
gate verdict against test outcome. A verdict with `error` set means the gate raised for that
puzzle and the worker went on: count it apart. The gate's seconds per puzzle come from the
`gate`, `gate_decode`, and `gate_selection` records (`scripts/timing_report.py` lists them in its
cost-model section and counts them as overhead per puzzle).

**One refinement round (`runs/P2.3-01/`).** The decision rule is the approved brief's. Read
`stages.run.arms` first: the arm ran with `solver_exit` 0 and has a `puzzle` record for every
DEV4 task. Then `run_logs/r256-proj-emb-rounds/gate/round-rank*.jsonl`: one round record per
puzzle (`max_rounds` 1); for each with `decision` set, its plan in `plan-rank*.jsonl`; for each
`rolled_back`, `restored` true and `weights_before` equal to `weights_after`. The promotion records
say why each test input was or was not promoted, and `decode-rank*.jsonl` what the final decode,
with the held-out pair (`name` `final`, `prompts` `held_out`), replaced or why it was skipped. The
round's seconds come from the timing records (`round` and the ops in it), the final decode's from
`final_decode` and `final_selection`.

**The in-context and the gradient round on DEV40 (`runs/P2.5-01/`, `runs/P2.5-02/`).** The
decision rule is each approved brief's. For each run: `stages.run.arms` (the arm ran with
`solver_exit` 0 and a `puzzle` record for every DEV40 task), then the score, the `*** Reload
score:` line of `run_logs/<arm>/selection.log`, against `P1.3-01` arm 2's 10.5 of 40 (the same
scope with every pair and no round; it ran the moved reference solver), with `P2.1-02`'s 7.0
and `P2.1-03`'s (when it has reported) as reference rows. Then the rounds as for `P2.3-01`: one
round record per puzzle, `decision` `in_context` (`P2.5-01`) or `gradient` (`P2.5-02`) for every
round that ran, the plans' `shape.in_context` (pairs held and `too_long`) in `P2.5-01`, the
promotion reasons, and the final decodes. `context-rank*.jsonl` lists every decode whose prompts
left out a pseudo-pair: for each,
$\text{lengths}[|\text{kept}|] \le \text{limit} < \text{lengths}[|\text{kept}| + 1]$ (the prompt
with the pseudo-pairs it kept fits, one more would not), or, when the prompt without any already
exceeded the limit, $\text{limit} < \text{lengths}[0]$ with none kept (the reference's own cut, as
without rounds). The two runs share round 0 (the same fit, test decode, and gate, at rank 16 on
q, k, v, o, up to the GPU's nondeterminism), so their difference per puzzle is the round's.

**Submission 2 (`runs/P2.6-01/`).** The decision rule is the approved brief's: its rows read the
DEV120 score against `P1.1-01`'s 37.17 of 120. First check that the scope ran: the session log's
environment check prints `LoRA scope, both modes: {'lora_r': 16, 'lora_targets': ['q_proj',
'k_proj', 'v_proj', 'o_proj']}` and `solver code: split`, and `stages.run.scope` in
`run_logs/stages.json` holds the same. Then the counting (`solver_exit` 0, a `puzzle` record for each of the 120 tasks, no
`stop!` line, a valid `submission.json`), the score (`*** Reload score:` in the session log,
checked with `arc26.evaluate`), and, per test output, the outputs solved in both runs and in one
only, each with its correct candidates in both runs (`score_kgmon` over `candidates.json`; the
correct grids from the runs' `ALL_CORRECT` lines; no solutions file is read locally). The timing
table comes from `scripts/timing_report.py runs/P2.6-01/run_logs/timing`; $u$ is the last
worker's finish over the 41,374 s window, projected to 240 tasks as `P1.1-01` projected.

**The hybrid's CPU check (`runs/P6.5-cpu/`).**

- Good: `stages.json` has the hybrid's seven stages without `error` and `problems` empty;
  `stages.agent_check` has `ready` true, the `weights` (12 files, 19,472,990,100 bytes) and the
  `wheelhouse` (vLLM 0.31.0, torch 2.13.0) checked, and the session log names the folders the
  search found (record that mount layout); `stages.agent.install` has `exit` 0, `ok` true, and
  `engine` true, its `imports` torch with CUDA 13.0 and vllm; `agent_exit` 0, `ended` `exited`,
  and `run.problems` naming the engine's failed boots (no GPU), the log tail in
  `run_logs/agent/engine.log`; `final.valid` true, `final.sources.changed` 0, and the final
  file equal to NVARC's copy (`run_logs/agent/nvarc_submission.json`). `final.rewritten` may be
  true: on a subset the agent writes only its `--task-ids`' tasks and the final check restores the
  rest from NVARC's copy (`sources.fallback`). Next: the `P6.5-01` brief's launch.
- Borderline: all of that, but the install took more than half of `install_timeout_s`, or the
  agent stopped for another reason than the boot (a config refused: exit 2; tasks unreadable:
  exit 1). Fix or record it in the brief before the GPU run.
- Bad: the agent check failed (a missing or ambiguous mount, a manifest not `ok`, another
  revision or vLLM), the install failed, or `final.valid` is false. Read `problems` and the
  session log; fix, rebuild, run the check again. Never go to the GPU run with a failing check.

**The hybrid's measurement (`runs/P6.5-01/`).** The decision rule is the approved brief's: its
rows (Lift, Flat, Loss, Bad) read the final DEV40 score, `stages.agent.score.final.score`, against
`P2.6-01`'s 10.5 of 40 on the same tasks, with NVARC's own part, `score.nvarc.score`, beside it
(expected near 6.5: NVARC cut at a sixth of the hybrid's share reaches about 24 of the 40 tasks),
and the verified solves among `P2.6-01`'s 29 misses against the spike's. First check that both
stages ran as planned: the run stage's `solver_end_s` 3,600 and `solver_kill_s` 4,200 (NVARC's cut
deadlines), NVARC's last `puzzle` record in `run_logs/timing/`, `stages.agent_check.ready` true,
`stages.agent.gpus_before_engine` with `after` empty and every `memory_after` at most
`used_max_mib` (else the GPUs NVARC's workers held were still held 60 s later: read the boot;
record `memory_before`, the first reading of an L4 after NVARC), `stages.agent.install.ok`, the
agent's `run.engine.boot_s`,
`generation_stop_s` 8,000, `agent_exit` 0 and `ended` `exited`, and `final.valid`. The distortions
of the brief's section 6 (the unscaled install, boot, and margins) are read with every row. Then
the gain: `score.gained` and `score.lost` (`lost` lists outputs NVARC solved whose two attempts
verified but wrong programs took over, P6.2's selection putting verified programs first),
`final.sources.changed`, the agent's slots by source (`run.submission.slots`: `program`, `direct`, `nvarc`, `empty`), and its scheduler report
(attempts started and verified per phase, tokens, the measured rate). The timing table for NVARC
comes from `scripts/timing_report.py runs/P6.5-01/run_logs/timing`; the agent's engine calls are
`run_logs/agent/timing-engine.jsonl`.

**The shot's CPU twin (`runs/P6.5-02-cpu/`).**

- Good: the hybrid CPU check's Good reading (above), with `stages.agent.command` running
  `arc26.agent.shot` and naming `--solutions`; the shot's `run.json` names its stop before any
  wave (the engine's preparation or preflight without a GPU, or its boot; on one probe task
  without candidates, after its no-candidates fallback); `agent_exit` 0; `final.valid` true,
  `final.sources.changed` 0, and the final file equal to `run_logs/agent/nvarc_submission.json`
  (`final.rewritten` true is expected: the shot writes only its task ids, the final check restores
  the rest). The twin of `f797d34` read so (install `ok` in 161.6 s, the shot stopped at the GPU
  preflight). Next: the `P6.5-02` brief's launch.
- Bad: the agent check, the install, or the shot's start failed for another reason than the
  missing GPU (a refused config: exit 2), or `final.valid` is false. Fix before any GPU run.

**The shot's smoke (`runs/P6.5-02/`)** and **the 12 h check (`runs/P6.5-03/`).** The decision
rules are the approved briefs'. First the order of stages: the run stage's `solver_end_s` and
`solver_kill_s` NVARC's own (2,100 and 2,400 s in the smoke; 41,400 and 42,000 s in the check),
its `seconds` (NVARC's end, against the projection: 30,640 s from the launch on DEV240), the
agent check's `plan` and `ready`, `stages.agent.start_s` against the plan's `latest_start_s`,
`install.ok`, `gpus_before_engine`, then the shot's `run_logs/agent/run.json` (its stages: engine
preparation, preflight, boot, sanity), `shot_plan.json` (each wave's tasks, $C$, planned and
actual end; the tasks queued under `max_confidence`), and `shot_metrics.json` (the shot's solves,
the merge's `gained` and `lost`, the hybrid's score against NVARC's alone on the same run).
`stages.agent.score` (`nvarc`, `final`, `gained`, `lost`) is the notebook's own score of the same
two files. The notebook's last log line is `done after <total_s> s`: the check's Timing row reads
it against 40,500 s, and Kaggle's RUNNING to COMPLETE (the orchestrator's watcher) against
43,200 s. `stages.environment.clocks` says which clock saw the minutes before the first cell:
compare `uptime_s`, `pid1_age_s`, `kernel_age_s`, and `working_ctime_age_s` with the watcher's
RUNNING (each a time before the first cell's `time`). In DEV240 a copy and its original are scored apart (`solved` lists both
ids when both are right; `settings.dev240.copies` maps them).

**Script smoke (`runs/script-smoke/`).**

- Good: `run_logs/stages.json` has `ok` true, `problems` empty, `script_exit` 0, `killed` false,
  and the stages `environment`, `install`, `script`, `run_logs` without `error`; the session log
  shows the script found in the code dataset's folder with both sha256 lines and the command
  `python .../env_probe.py --config .../script_smoke.yaml`; `run_logs/` holds `env_probe.json` and
  `env_probe.md` beside the notebook's four files, and `env.json` shows run A's image and
  packages. Next: the first GPU script notebook's pre-run brief (`P3.2-01` or `P3.1-01`).
- Borderline: `ok` true, but `env_probe.json` lists sections in `errors` or a mount where the
  research did not expect it. Record it in the next brief; the plumbing itself passed.
- Bad: `ok` false. Read `problems` (a missing file or a sha256 mismatch means the dataset version
  is not this build's), then the session log; fix, rebuild, run again before any GPU script
  notebook.

**The weights' warm-up (`runs/P6.9-03-cpu/`, `runs/P6.9-03/`).** The decision rule is the
approved brief's. Read `stages.agent.weights` first: `source` `local` and `model_dir`
`/tmp/arc26-agent/weights/qwen38-27b-int4` mean the shot got the verified copy; then
`warmup.json`: `ok`, 12 names in `verified`, `copy_s` and `rate_mb_s` (the mount's sequential
read rate with 8 readers), `verify_s`, and `end` against `stages.agent.start_s` plus the install
(how early the copy was done). In the GPU run, `run.json` `engine.command[2]` is the folder vLLM
served, `engine.boot_s` the boot, and `engine.log`'s "Filesystem type for checkpoints" and
"Loading weights took" the load. NVARC's 4 `puzzle` records against `P6.9-02`'s (107, 170, 261,
1,034 s) show whether the copy slowed NVARC. A `source` of `input` names the reason in `reason`;
the boot then shows what the prefetch fallback gives on the mount.

**MTP and the 69,632 cap (`runs/P6.9-04-cpu/`, `runs/P6.9-04/`).** The decision rule is the
approved brief's. The twin: everything `P6.9-03-cpu` read Good on, `stages.json`
`build.agent.config` `configs/agent_shot_P6.9d.yaml`, and `shot_plan.json` `model.max_cap` 69,632
(the shot loaded the config on Kaggle's image). The GPU run: `run.json` `engine.command` with
`--speculative-config` and the local folder, `engine.boot_s` under 900; `engine.log`: the
`SpecDecoding metrics:` lines with `Accepted:` and `Drafted:` above 0 (their acceptance against
`P6.10-01`'s $\alpha = 0.87$), no line with the word `nan` outside them; `boot.json` `kv_pool`
against the 1,467,099 estimate; every `waves` row's `cap` from 65,536 to 69,632; and each turn's
tokens per second (`engine_calls.jsonl`, tokens over seconds) against `P6.9-03`'s 38.9 and 39.4 at
2 streams on the same tasks, the decode speed-up at 2 streams (**reported**: one session, 2 or 3
turns).

**The 69,632 cap without MTP (`runs/P6.9-05-cpu/`, `runs/P6.9-05/`).** The decision rule is the
approved brief's. The twin: everything `P6.9-04-cpu` read Good on, with `stages.json`
`build.agent.config` `configs/agent_shot_P6.9e.yaml` and its `config_sha256` the brief's,
`shot_plan.json` `model.max_cap` 69,632, and no `--speculative-config` in the built config. The
version's own run: `run.json` `engine.command` without `--speculative-config`, every `waves` row's
`cap` from 65,536 to 69,632, and each turn's tokens per second against `P6.9-03`'s 38.9 and 39.4
(**reported**). The scored rerun's public score against version 6's 30.28 decides the shot pick.

## Failure modes

A GPU run that stops, in its environment check or on any other kill criterion of its brief, is
reported and never pushed again without an amended, re-approved brief (BRIEF A9; kill criterion 1
of the briefs: "Bad (infrastructure: report, no rerun before an amended brief)"). For a stop that
says nothing about the code, such as the bf16 stop on substituted T4s, the amendment can be the
next `Revision` with the same `Code-commit` and a note on the stop; the human approves it again
before the push. A CPU push needs no brief and may simply be repeated.

| Symptom | Cause | First thing to change |
|---|---|---|
| `environment check: 0 wheel candidates` | Code dataset not attached, not processed yet, or a different wheel name | Wait for the dataset version, check `dataset_sources`, rebuild |
| `... has sha256 ..., the build baked ...` | The attached dataset version is from another build | `dataset-version` from this build's `arc26-code/`, or rebuild the notebook from the dataset's commit |
| `0 model candidates` or `0 challenges candidates` | Model or competition not attached, or mounted under an unexpected name | Read the printed search results; fix `model_slug` or the metadata |
| `pip install exited ...` | Broken wheel or pip error | Read the pip output in the session log |
| `the kernel does not import the arc26 that the python on PATH installed` | Kernel and PATH interpreters use different site-packages | Compare the two interpreter lines; the fix goes into the Install cell before any GPU run |
| `the solver needs 4 GPUs, one per worker` | Accelerator with fewer GPUs than workers | `accelerator` or `workers` in the config |
| `solver_exit` not 0, `solver_error` set | Solver crash: OOM, CUDA error, a NaN in training (reference code) | Read the session log for the rank that failed; the submission still holds every candidate written |
| Solver killed at the hard stop | A hang or a slow GPU | Raise `solver_kill_s` or investigate. A candidate pickle cut mid-write is moved aside (`quarantined` in `stages.json`) and selection runs again, so only that augmented view's candidates are lost |
| Session killed by Kaggle's `Timeout`; only the placeholder in the output | The brief's `Timeout` is below $s_\text{kill}$ plus the write time | `Timeout` at least $s_\text{kill} + 900$ |
| No `run_logs/timing/` after a development run that solved puzzles | `timing_dir` did not reach the workers | Check `timing_dir` in `stages.json`; locally, `tests/test_notebook_local.py` asserts the solver's `--config` |
| `train` records with null tokens | The trainer's prepared dataset has no `input_ids` column (a trl or unsloth change) | The run is still valid; fix the count in `configs/extraction.yaml` before the next timing run |
| A competition rerun scores 0 | A build that is not for submission was submitted (no accelerator, say), or the rerun logged problems | Submit only a build whose `MANIFEST.md` says `fit to submit \| yes`; read `PROBLEM` lines in the rerun's log |
| `MANIFEST.md` says "no, not for submission" | The build is for a notebook other than `alyahya737/arc26-submission`, or its settings are not production: not four workers on `NvidiaL4`, a series, no accelerator, or an agent stage that ends before 41,400 s (a development scale) | Expected for `P0.3-01`, `P1.3-01`, `P2.0-01`, `P2.1-01`, `P2.3-01`, `P2.5-01`, `P2.5-02`, `P6.5-01` (a sixth of the hybrid's times on its own notebook), and the CPU checks (`P6.5-cpu` included); for a submission build (`P1.1-01`, `P2.6-01`, `P6.5-hybrid`, `P6.5-agent`), fix the config. A `scope` is never a reason, nor an `agent` at the rerun's times |
| Build refused: `solver must be one of nvarc, split` | The config's `solver` is another value: null, a module name, a capital letter | Write `nvarc` for the moved reference, or leave the key out for `split`, the default |
| Build refused: `arm <name> holds a pair out, which needs solver: split` | A hold-out arm in a build of the moved reference, which trains and decodes in one loop body | Remove `solver: nvarc` (the default is `split`) or write `solver: split` |
| Build refused: `holdout must be null or have exactly the keys rule, k2, in_test_prompts` (or a rule or flag message) | A missing, extra, or misspelled hold-out setting; the folders are the notebook's, not the config's | Spell out all three settings; `rule` is `last` or `first`, the flags true or false |
| `[Rank r] gate for <task>: raised <error>` in the session log, a verdict with `error` set | The gate's decode of the held-out input failed (out of memory on a large input, say); the worker went on, and the puzzle kept its test candidates | Count the puzzle apart in the gate's statistics; if it repeats, read the traceback above the line before the next run |
| A hold-out arm without `run_logs/<arm>/gate/` | `log_dir` did not reach the workers, or no puzzle finished | Check `stages.run.arms` and the solver's `--config` in the session log; locally, `tests/test_notebook_local.py` asserts both gate folders |
| Build refused: `rounds must be null or have exactly the keys max_rounds, tau, continue_steps, final_in_prompts, governor, and optionally variants` (or a setting's message) | A missing, extra, or misspelled round setting; the round folder is the notebook's, not the config's | Spell out all five settings, within `arc26.loop.Rounds`' ranges, and `variants` only to force a variant |
| Build refused: `rounds variants must be a list of gradient, in_context, one or both in that order` | Another variant name, a repeat, the reverse order, or a bare string | `[gradient]`, `[in_context]`, or `[gradient, in_context]` (the governor's ladder) |
| An in-context round `too_long`, with no plan | Only the in-context round was offered, and no input's pseudo-pair fits the gate's prompts (large grids: three $30 \times 30$ pairs and a query fill a prompt) | Expected for such puzzles: no GPU op ran, the final decode follows; offering both variants lets the gradient round, which trains on the pair, run there |
| `context-rank*.jsonl` lines for many decodes | The prompts with every pseudo-pair would pass the model's limit, so the guard left the latest out (the final decode holds the held-out pair as well) | Expected at large grid sizes; the reference's cut would otherwise drop a random pair, real ones included. If a line's `lengths[0]` exceeds `limit`, the prompt was over the limit without any pseudo-pair: the reference's own cut applied, as without rounds |
| Build refused: `rounds need the arm's holdout` or `rounds need holdout in_test_prompts false` | A round arm without a held-out pair, or with the pair in the test prompts (round 0's decode feeds the promotion, ADR-013) | Add `holdout` with `in_test_prompts: false`; `final_in_prompts: true` puts the pair back in the final decode, the one submitted |
| Build refused: `decode_policy mode must be 'reference'` or `a governor config needs ...` | The governor config named by `governor` is in the trade policy, which the rounds cannot run (it needs a plan before each baseline batch), or incomplete | Use `configs/governor.yaml` with `decode_policy.mode: reference` |
| Every round `declined`, each with a plan in `plan-rank*.jsonl` | The time left to each puzzle's deadline cannot cover its round's estimate: the governor shares the soft deadline among the queued puzzles, and prices batches at its p90 with a margin | Expected when the soft deadline is tight; for a run meant to exercise rounds, raise `solver_end_s` (`docs/prerun/P2.3-01.md`, section 6, derives it) |
| A round `rolled_back` with `restored` false | The restore or its fingerprint failed (`error` set), or the restore was not bit for bit | ADR-015's overturn criterion: stop using rounds until the restore is fixed; the run's candidates are round 0's, as without rounds (the final decode is skipped, `not_restored`) |
| A final decode `skipped` `no_time` | The time left to the puzzle's deadline was below the final decode's own estimate: the round before it overran its price, or the round was declined and the deadline is near | The open inputs keep their candidates (round 0's, without the held-out pair); compare the round's measured seconds with its plan's estimate (P2.4's calibration) |
| A round `raised`, `[Rank r] round 1 for <task>: raised ...` | An error outside the promotion's own checks (the provider, the governor, the records); the round was rolled back and the worker went on | Read the traceback above the line; the puzzle's candidates are as before the round |
| `P2.0-01` (or a rerun of the parity check): $g(E', A) < 0.4$ while the CPU tests pass | GPU nondeterminism (one test output moves $g$ by 0.2), or a difference outside the moved statements: the worker's imports, the process entry, the environment | The brief's Parity fail row: compare the per-view candidates with A's, B's, and `P0.5-01`'s before any rerun; until it is explained, builds state `solver: nvarc` |
| Build refused: `scope must be null or have exactly the keys lora_r, lora_targets`, or `scope: lora_r ...` / `scope: lora_targets ...` (an arm's message) | A malformed scope: another key, a rank that is not a positive integer, targets out of the reference's order or without a linear projection | Spell out `lora_r` and `lora_targets` as an arm does |
| Build refused: `scope is the reference's, lora_r 256 on all nine targets` | The scope repeats the notebook configuration | Leave `scope` out: the build is then the notebook configuration's, byte for byte |
| Build refused: `scope and arms exclude each other` | A config with both: the arms set each launch's scope in development runs, while a rerun ignores them and would train the build's scope | Measure scopes in a series (development), then build the submission from a config with `scope` and no `arms` |
| Build refused: `the first arm must be the notebook configuration` | A series config whose first arm is not $r = 256$ on all nine targets and holds no pair out | Put the notebook configuration first, or give the first arm a `holdout` (a loop measurement, task P2.5); `submission.json` comes from it |
| `arm <name> not started: ... s are left before the soft deadline` | The arms so far took longer than the time left allows, times 1.25 | The arms that ran stand; run the missing ones in a new series (notebook configuration first) under an amended brief, or decide G2 on fewer arms |
| `arm <name> failed, so no later arm starts` | That arm's solver crashed or exited non-zero (`solver_exit`, `solver_error` in its record) | Read the session log after its `== arm` line; fix, then rerun the remaining arms under an amended brief |
| An arm exits 0 but some DEV40 tasks have no `puzzle` record, and its workers printed `stop!` | It was still running at the soft deadline (the margin was too small for it) | It did not fail, it does not count; the arms before it stand, and it reruns in a new series under an amended brief |
| An arm's `selection.log` ends in a `ValueError` | The reference's scoring found no correct candidate in that arm (`np.max` of an empty list), after writing its submission | Expected for an arm that solves nothing; its score is 0 of 40 and its `submission.json` is valid |
| `environment check: the build asked for NvidiaL4, but nvidia-smi lists GPUs under compute capability 8.0, without bf16: Tesla T4 (7.5), ...` | Kaggle ran the push on GPUs it substituted, such as its default 2xT4 for an accelerator it does not grant | Nothing in the code. Report the stop; push the build again on `NvidiaL4` only under an amended, re-approved brief (the rule above). Never ask for `NvidiaL4X1` for a solver or bf16 build. In a competition rerun it is only logged |
| `... but nvidia-smi lists no GPU` | A build made for an accelerator got none, or the driver did not answer | Check the push's accelerator and `status`, report the stop, and push again only under an amended, re-approved brief |
| Script mode: `<script or config> has sha256 ..., the build baked ...`, or `0 script candidates` | The code dataset version is not this build's: a submission build's version holds no script, another script build's holds other files | `dataset-version` from this build's `arc26-code/`; then repeat a CPU push, or report a GPU run and push it again only under an amended, re-approved brief |
| Script mode: `script_exit` -9 and `killed` true in `stages.json` | The script ran past the hard stop $t_0 + k$ | Lower the script's own budget below $k$ (it should stop itself and save), or raise `script_kill_s` and the brief's `Timeout` |
| Script mode: `script_exit` not 0, `killed` false | The script failed | Read its output in the session log and its own `error.log` in `run_logs/` |
| Script mode: `the script wrote env.json into ...` (or another of the notebook's file names) | The script uses a name the notebook writes in `run_logs/` | Rename the script's file; the notebook's replaced it |
| `competition rerun of a script notebook` | A script notebook was submitted | Never submit a script notebook (its first cell and `MANIFEST.md` say so); submit a build of `alyahya737/arc26-submission` |
| Build refused: `a script notebook is never submitted, so it cannot be alyahya737/arc26-submission` | A script config names the submission notebook | Give the script notebook its own slug (for a CPU push, one that no pre-run brief names) |
| Build refused: `args must not set --config` | `args` holds `--config`, a prefix of it, or `-c` | Put the setting into the script's config |
| Script mode: `ARGS must not set --config` in `problems` | The Config cell's `ARGS` were edited to set `--config` | Restore `ARGS`; settings go into the script's config, which a new build checks |
| Build refused: `accelerator NvidiaTeslaT4 can never pass the environment check's bf16 stop` (or a TPU ID) | The config asks for an accelerator without bf16 or without an NVIDIA GPU | Request `NvidiaL4` |
| Script mode: `ModuleNotFoundError` for a module of `scripts/` | The script imports a sibling module, which is not shipped | Move the shared code into `arc26/`, which the wheel ships |
| Build refused: `scripts/<name>.py does not compile`, or the script config `still has TODO` | The script or its config is not ready | Fix and commit; the build reads both from the commit |
| Build refused: `cannot read configs/agent_default.yaml` | A hybrid config before task P6.4 is merged: the agent's run config is not in the commit | Merge P6.4 (its run config and `arc26.agent.run`), then build |
| Build refused: `agent: the agent stage could not start: ...` | The share, the agent's end, and the run config's `final_margin_s` leave less than `min_stage_s` after NVARC's hard stop and its write | Lower `nvarc_share_s`, or raise `end_s` within its bound; the message gives each time |
| Build refused: `agent end_s must be an integer from 600 to ...` | The agent's end leaves less than SIGKILL's grace, the kill wait, and the final write before 12 h | `end_s` at most $43{,}200 - 300 - 30 - \tau$ (42,750 s with $\tau = 120$) |
| Build refused: `agent term_s ... must exceed the engine's stop_timeout_s` | The agent, given SIGTERM, could not stop its engine before SIGKILL | Raise `term_s` above `stop_timeout_s` plus 30 s |
| Build refused: `agent and arms exclude each other` | A series config with an agent stage | Measure scopes in a series; build the hybrid from a single-run config |
| `agent check: N distinct folders of the weights output ...` (or the wheelhouse's) | The download output is not attached at its pinned version, not processed yet, or mounted twice with different files | Check `kernel_sources` in `kernel-metadata.json` and the output's version on Kaggle; a development run stops here, a rerun gives NVARC the whole session |
| `agent check: weights: ... does not say ok` or `names revision ...`; `wheelhouse: ... lists vllm ...` | The attached output is another run's (a failed download, another revision or vLLM) | Attach the version the config pins, or rebuild with the new pin (a config change: new `Code-commit`) |
| `agent check: the NVIDIA driver is ..., older than 580.65.06` | The session's driver cannot run CUDA 13.0 (torch 2.13 in the wheelhouse) | Report it; the engine cannot run on that image. A rerun gives NVARC the whole session |
| `stages.agent.failure`: `the engine install failed (see install), so the agent did not run` | The installer exited non-zero, timed out (`ended` `terminated` or `killed`), or left no `bin/vllm` | Read `run_logs/agent/install.log` and `install.json`; NVARC's file stands |
| `stages.agent.agent_exit` 0 with `run.problems` naming the engine's boot | vLLM did not come up (out of memory, a kernel error, the deadline); the run kept NVARC's attempts | Read `run_logs/agent/engine.log`; the spike's settings (`configs/agent_default.yaml`, `server`) are where to change it |
| `stages.agent.ended` `terminated` or `killed` | The agent still ran at `end_s`: a phase overran its deadline, or its stop hung | The file holds its last atomic write over NVARC's; check `run.scheduler` and the engine log; `engine_stopped` says whether the notebook stopped a leftover engine |
| `stages.agent.agent_exit` 2 | `arc26.agent.run` refused its arguments or its config | Read the session log's `refused:` line; the run config is the one embedded at build time |
| `stages.agent.skipped`: `... less than AGENT['min_stage_s']` | NVARC and its write left too little time before the generation stop | Expected after an overrun; on DEV120 NVARC ends near 15,700 s, and in `P6.5-01` its hard stop at 4,200 s leaves 3,500 s before the generation stop at 8,000 s, so a skip there points to a slow NVARC stage |
| `stages.agent.gpus_before_engine.after` lists processes, or `memory_after` a GPU above `used_max_mib` | GPUs still held 60 s after NVARC's stage: its killed workers, or another process, had not released their memory (inside a container only `memory_after` may show it) | The engine booted on what was free: read `run.engine.boot_s` and `run_logs/agent/engine.log` for an out-of-memory boot; the pids, when listed, name the holders. If it repeats, raise `GPU_FREE_WAIT_S` (a builder change, new `Code-commit`) |
| `stages.agent.gpus_before_engine.memory_before` above 1,024 MiB on an idle GPU, every run | What an idle Kaggle L4 reports is above `GPU_USED_MAX_MIB` (unmeasured when it was set) | Each wait then lasts the full 60 s and the boot decides: raise `GPU_USED_MAX_MIB` to the idle reading plus a margin, under the 2,300 MiB vLLM's 0.90 leaves (a builder change) |
| `stages.agent.nvarc_fallback` with `solver_exit` and `selection` | An agent-only build's agent stage ended with 30 minutes or more left (the install or the boot failed, the agent crashed or was killed, its engine was gone), so NVARC's stage ran after it | Expected after such a failure: the file is P6.2's selection, the agent's verified programs in attempt 1 (`selection.slots.program`), NVARC's attempts next, the agent's other grids after. Read `failure`, `install`, `agent_exit`, and `run.problems` (absent after a kill) for why the agent stopped; NVARC started late, so it reached fewer tasks than Submission 2 |
| `stages.agent.nvarc_fallback.selection.skipped_lines` above 0 | A record line cut by a kill (the agent's last write), or a line that is not a record | Expected after a kill: the cut record had not reached the file either. Many skipped lines mean a log another program wrote: read `run_logs/agent/attempts.jsonl` |
| `stages.agent.nvarc_fallback.merge`: `NVARC's selection wrote no file, ...` | NVARC's fallback run selected nothing (`make_submission_error`) | The agent's file stands; read the fallback's `solver_exit` and the session log |
| `stages.agent.nvarc_fallback.skipped`: `... less than 1800 s` | The agent-only stage ended within 30 minutes of NVARC's soft deadline: at its own end, or by a failure late in the session | The file holds the agent's grids alone (verified programs, direct answers, unverified programs); after a failure, the agent-only build's risk (`docs/SUBMITTING.md`, "The hybrid") |
| `stages.agent.fallback_error` | The fallback raised (its launch, its selection, or its write) | The final check ran after it on the last file written atomically; read the traceback in the session log |
| An agent-only version with few verified programs and no NVARC attempt | The agent ran to its generation stop, so no fallback: an engine that fails part of its calls, or a slow one, keeps the session (nothing watches the error rate live) | The agent-only build's risk: prefer the hybrid unless the spike's rate and verified solves carry the agent alone |
| `stages.agent.final.rewritten` true with `sources.fallback` above 0 | The agent's file lacked tasks (a development subset) or held invalid attempts | Expected for a subset; otherwise read the agent's `run.json` for failed writes |
| Build refused: `agent: a shot build (arc26.agent.shot) keeps NVARC's deadlines, so nvarc_share_s must be 41400` | A shot config with a share cut (task P6.5b, design 1) | Set `nvarc_share_s: 41400`; the shot runs in the time NVARC leaves |
| Build refused: `agent: the shot could never run: ...` | `end_s` minus the run config's `final_margin_s` minus `min_stage_s` is before NVARC's write | Raise `end_s` within its bound, or lower `min_stage_s` only with a measured setup |
| `stages.agent.skipped`: `... less than AGENT['min_stage_s']` in a shot build | The agent stage (NVARC's end plus the Write cell) started after the plan's `latest_start_s` (37,230 s in `P6.5-03`: NVARC ended after about 37,140 s, $u$ above about 0.896 on 240 tasks) | Expected on a slow hidden set: NVARC's file is the submission, as Submission 2's. In `P6.5-03` it means NVARC ran past its projection: the brief's Timing row |
| The shot's `run.json` stops before any wave: the engine preparation, the preflight, the boot, or the sanity request | An engine that cannot start or answers degenerately (`docs/modules/agent_shot.md`) | NVARC's file stands; read `run_logs/agent/engine_prep.json`, `boot.json`, and the engine log. A Fault row in `P6.5-02` and `P6.5-03` |
| `shot_plan.json`: `no task under max_confidence` | NVARC's confidence is at or above `selection.max_confidence` on every task (DEV4 at 0.5) | Expected on DEV4 at 0.5, hence the smoke's 1.0; elsewhere NVARC is sure of everything, and its file stands |
| `shot_plan.json`: no wave, $C$ below 16,384 | The time left after the setup did not hold the smallest wave: the stage started near `latest_start_s`, or the install or boot was slow | NVARC's file stands; compare `install.seconds` and the boot with `P6.3-05`'s 106 s and 300 s |
| `DEV240 could not be written: the copy id ... is taken by ...` | A copy's derived id is a task id of a competition file (a new file on Kaggle's mount) | The development run stops in the environment check; report it: the copy rule needs a salt, a builder change |
| `stages.agent_check.warmup.started` false, `reason` naming free bytes | Scratch had less than the copy plus `reserve_gb` free | The engine loads from the input with the fallback strategy; free scratch, or lower `reserve_gb` with the venv's measured size (a config change, new `Code-commit`) |
| `stages.agent.weights.source` `input`, `reason` `the warm-up was not done after 900 s` | The mount read slower than about 10.2 MB/s in `P6.9-03` (0.65 MB/s in a rerun) | Read `warmup.json`'s `copied_bytes` and `rate_mb_s`; the boot from the input decides the run. A Fault in `P6.9-03`: report the measured rate before any repeat |
| `stages.agent.weights.reason`: `no progress for 120 s` | The copier hung on a read (the mount stalled) or was starved | The copier was killed; read `warmup.log` and `priority`; a stall in `copying` with `copied_bytes` flat points at the mount |
| `stages.agent.weights.reason` naming `sha256 ... is not the manifest's` | The bytes read differ from the Hub's: a mount error or a changed output | The engine loaded from the input; compare the output's version with the build's pin (version 1) |
| `stages.agent.weights.reason`: `the warm-up exited ...` | The copier died before `done` (out of memory, a signal) | Read `warmup.log`; the status's last `state` says where |
| `warmup.json` `priority.ionice` `not set: ...` | No `ionice` on `PATH`, or it refused | The copy ran at `nice` alone; harmless unless NVARC's puzzle times moved |
| NVARC's puzzle records slower than `P6.9-02`'s by more than 10% with the copy running | The copy competed with NVARC for the mount, the CPUs, or the disk | Raise `delay_s` or lower `workers` (a config change, new `Code-commit`); in a rerun the copy ends hours before NVARC |
| `run.json` problems: `the engine was not healthy after 900 s` with `--speculative-config` in `engine.command` | MTP's boot on Kaggle (draft load, compile, graph capture) far slower than the base's 278.1 s, or a hang | Read `engine.log`'s last lines and their times; the brief's Fault row: version 6 stays the shot pick |
| No `SpecDecoding metrics:` line, or `Accepted:` 0 | vLLM did not take the draft (a flag spelling, a checkpoint without `model_mtp.safetensors` in the folder served) | Read `engine.log`'s speculative config lines and `warmup.json` `verified` (12 names, `model_mtp.safetensors` among them) |
| A line with `nan` outside the `SpecDecoding` lines, or a degenerate reply | vLLM issue #55766 (speculation with prefix caching) | The reply's task keeps NVARC's attempts; the brief's Fault row; a repeat would turn prefix caching off (its own brief) |
| `shot_plan.json` reasons `kv: ...` with fewer than 15 streams | MTP's pool on Kaggle under the plan's need, 1,238,205 tokens for 15 openings of 6,771 | Expected only below that; read `boot.json` `kv_pool`; a lower cap is a config change |
| LLM first: `stages.run` not skipped, NVARC ran | The agent check failed (`stages.agent_check.ready` false: weights or wheelhouse missing, too few GPUs, an old driver), so NVARC took the session as designed | Read `stages.agent_check` and `PROBLEMS`; the agent stage is skipped and NVARC's file is the submission |
| LLM first: `shot_plan.json` reasons `kv: ...` with fewer than 20 streams | The boot's pool under $20 \cdot 78{,}451 = 1{,}569{,}020$ tokens for openings at the 6,771 bound | Expected only with a smaller pool; read `boot.json` `kv_pool`; the starts wait for free KV, nothing fails |
| LLM first: vLLM `Preempted` lines or a KV use near 100% with 20 running | The linear-attention state beyond the plan's $o + C + F$ per stream | Lower `waves.wave_tasks` to 19 in a new build; the run's answers stand |
| LLM first: many tasks still `[[0]]` after the run | The window ran out (about 93 of 240 expected at about 293 tokens per second) or tasks were cancelled at the stop | Expected; compare `shot_plan.json`'s starts and `stopped` with the runbook's coverage table |
