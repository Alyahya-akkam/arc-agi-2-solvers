# Runbook: `scripts/train_throughput.py` (P3.6, throughput and memory of H's meta-training step)

Task P3.6 (BRIEF B5 P3.6, B3, B8, A8, A9). Config `configs/train_throughput.yaml` (run `P3.6-01`),
build config `configs/script_P3.6-01.yaml` (script mode, `docs/runbooks/build_submission.md`);
pre-run brief [`docs/prerun/P3.6-01.md`](../prerun/P3.6-01.md). The step it times is P3.5's
(`arc26/hyper/train.py`, [module doc](../modules/train.md), runbook
[`train_hypernet.md`](train_hypernet.md)). The deliverable B5 names, `docs/training_throughput.md`,
is written from this run's results. "P3.5-01" as a source means that run's pulled logs
(`runs/P3.5-01/run_logs/`, `docs/experiments/P3.5-01.md`); "P3.4 stats" is P3.4 build 2's
`episodes_stats.md` (`docs/experiments/P3.4.md`).

## 1. Why this script exists

P0.4, the human's decision between Kaggle's weekly quota and rented GPUs for meta-training, needs
what one training step costs at the lengths P3.4's episodes reach, and proof that the longest
episode the cap allows fits an L4 (B5 P3.6; B8: "9.5k-token episodes on 24 GB need checkpointing
and flash attention; P3.6 tests the worst case before any long run is launched"). Run `P3.6-01`
answers three questions on the code that trained `P3.5-01` and `P3.5-02`:

- Does one step at the 8192-token cap (`max_tokens` of every P3.5 config; P3.4 dropped the 17
  longer episodes) run on each L4 of 4xL4 without running out of memory, and with what headroom?
  And at 9.5k tokens (B5's worst case), which decides whether a longer cap could fit later?
- What does a step cost at 512, 1024, 2048, 5120, 8192, and 9728 tokens, and H's forward alone
  (B5 P3.3's bound, under 50 ms on an L4)?
- How many steps of the long run's configuration (`configs/train_hypernet.yaml`: 4 GPUs,
  accumulation 8, P3.4's training lengths) fit in a week of quota, under both readings of the
  4xL4 rate?

Its results feed the brief's decision rule (B5 P3.6's acceptance), P0.4, and the meta-training
brief (A7 week 3: step time, kill criteria, session plan).

## 2. What it does, step by step

1. **Config.** The YAML config is resolved (the smoke section, command-line paths) and validated,
   the GPU values also in a smoke run; `config.yaml` holds the result. The `train` section is the
   long run's step (`configs/train_hypernet.yaml`'s values; only `max_tokens` differs, since the
   9.5k point is above the cap), completed into P3.5's `TrainConfig` with the keys no part of a
   step reads (the episode build, the encoder, validation, checkpoints).
2. **Model.** The base's folder is found under `input_root` (`/kaggle/input`) as the one folder
   holding `config.json` whose path ends with the Kaggle Models handle's model, framework,
   variation, and version, as `scripts/cache_embeddings.py` finds it; `--model-dir` skips the
   search.
3. **Sessions.** Points run in config order. Consecutive points with the same number of processes
   $W$ form a session: $W$ processes started with `torch.multiprocessing` (spawn), NCCL on GPUs
   and gloo on the CPU, rendezvous on 127.0.0.1, each loading the frozen bf16 base (FlashAttention
   2 when `flash_attn` imports, else SDPA), wrapping q, k, v, o of every layer for injection,
   enabling non-reentrant checkpointing per decoder layer, building H, DDP on H when $W > 1$,
   AdamW, and the warmup-cosine schedule, exactly as `_Worker.setup` of `arc26.hyper.train` does,
   in its order and with its seeds. `P3.6-01` has two sessions: 4 processes, then 1.
4. **Episodes of exact lengths.** An episode of $T$ tokens in the notebook's format
   (`QwenFormatter.fmt_train`) has $p$ pairs, the fewest that can hold it
   ($p = \max(2, \lceil T / 1866 \rceil)$, a pair of two 30 by 30 grids being
   $2 \cdot 929 + 8 = 1866$ tokens), and grids whose tokens
   $g(h, w) = h w + h - 1$ ($h \ge w$, so the read orientation is the longer one, as P3.4
   counts) satisfy
   $$\sum_{i=1}^{2p} g(h_i, w_i) + 8p = T,$$
   each grid as near an even share as keeps the rest reachable (a dynamic program over grid
   counts). Every length from 20 to 9728 tokens has such an episode except 21, 23, and 25. Colors
   are random (seeded), the augmentation keeps every grid's shape (ids $8j + \{0, 2, 5, 7\}$), and
   P3.5's `build_sequence` renders and checks it; every result records the length the base read.
   A `fixed` point's episodes all have $T$ tokens; a `mix` point draws each length from P3.4's
   training histogram (below).
5. **A point** (`run_point`): the CUDA caching allocator's peak counters are reset and its
   reserved memory recorded; its cache is not emptied, since a training run never empties it, so
   each point meets the allocator as the points before it left it. Then `warm_steps` untimed and
   `timed_steps` timed calls of `_Worker.train_step`, the
   training worker's own method: per micro step, $\bar z$ to the device, H's forward (DDP), the
   factors injected, the base's forward and backward under them (the gradient reaching H, DDP's
   all-reduce on the last micro step), then clipping, AdamW, the schedule. The step's seconds are
   P3.5-01's `step_s`; the timing records give each micro step's H forward and forward-backward.
   $\bar z$ is one seeded vector for every episode (no encoder: it runs before a training run's
   first step). After the point: the allocator's peaks, the device's used memory, and the
   allocator's retry and out-of-memory counts. A `hypernet` point instead times H's forward for
   all layers and modules, in evaluation mode without autograd (test time) and in training mode
   with autograd, each call between device synchronizations.
6. **Failures.** An out-of-memory error ends its point as a result (`oom`, with the error's text
   and the memory counters at that moment; an allocation that fails inside cuBLAS or NCCL, which
   reaches Python as another error, counts too) and ends its session: that process leaves at once,
   the parent sees its exit code and ends the others, and the session's later points run in a
   new session. Other errors end the point as `error` the same way. A failed setup takes its
   session's points with it.
7. **Watching.** Each process rewrites `sessions/s<i>/rank<k>.json` at every step. The parent
   ends a session whose processes all stay silent for `limits.stall_s` (`stalled`), and any
   session at `limits.max_wall_s`; no point starts later than `max_wall_s - reserve_s`.
8. **Results.** Per point, over the timed steps: the step's seconds per step as the slowest
   process's, their median, mean, minimum, and maximum; the micro step's median
   $m(T)$ = H's forward plus forward and backward; and, on the tightest GPU, the headroom
   $$H = M - (\hat r + u), \qquad u = d - r,$$
   $M$ the device's total, $\hat r$ the point's peak reserved memory, $d$ and $r$ the device's used
   and the allocator's reserved memory at the point's end ($u$: the CUDA context, NCCL's
   buffers). $\hat r$ includes the cache the earlier points left, recorded as the reserved memory
   at the point's start; the peak allocated memory is the point's own.
9. **Projection** (`project`). From the complete fixed points on $W = 4$ processes:
   $m(T)$ interpolated linearly between measured lengths, constant below the shortest, extended
   on the last slope above the longest; $t_\text{ddp} = s_4(512) - s_1(512)$, DDP's cost per
   synchronized micro step; $\bar r$, the median over those points of the step's seconds minus the
   micro step's. With $A = 8$, each process runs its micro steps without waiting (DDP's
   `no_sync`) and all wait at the last one's all-reduce, so a step of lengths $T_{kj}$ is
   $$\hat s = \max_{k} \sum_{j=1}^{A} m(T_{kj}) - (A - 1)\, t_\text{ddp} + \bar r.$$
   The model step $\bar s_\text{model}$ is the mean of $\hat s$ over 20,000 steps whose
   $W A = 32$ lengths are drawn from P3.4's training histogram: a bin with probability
   proportional to its count, then uniformly within it. The mix point measured real
   accumulation-8 steps; on its own lengths, $c = \bar s_\text{mix} / \bar{\hat s}_\text{mix}$, and
   the projected step is $\bar s = c\, \bar s_\text{model}$. For $Q$ quota hours a week charged
   at $\rho$ quota hours per session hour (2 or 1),
   $$n = \frac{3600\, Q}{\rho\, \bar s}$$
   steps of 32 episodes; one pass over the 99,674 training episodes is
   $\lceil 99{,}674 / 32 \rceil = 3115$ steps.
10. **Report.** `measurements.jsonl`, `projection.json`, `status.json` (with the decision rule's
    reading), `run.json`, and `summary.md`.

## 3. Why it is built this way

- **A script of its own, not a mode of `train_hypernet.py`.** A mode would change
  `train_hypernet.py`, the file `P3.5-01` and `P3.5-02` ran byte-identical (sha256
  `1e7ed3df...`) and the long run will reuse, and its setup requires P3.4's build and leakage
  report, which a cost measurement does not need. A script notebook ships one script and one
  config either way (P0.5d script mode), so a second script costs nothing in packaging.
- **The training worker's own step, not a copy.** `ThroughputWorker` subclasses
  `arc26.hyper.train._Worker` and replaces only `setup`; `train_step` and `micro_step` are
  inherited (`tests/test_train_throughput.py` checks that they are the same functions, that the
  setup leaves the model, H, the optimizer, the schedule, and every random generator as the
  training worker's does, and that three steps give the same H, bitwise on the CPU).
  Alternatives: a reimplemented step, which would time a copy that drifts from the training code;
  a public hook in `arc26/hyper/train.py`, which would change the wheel, so this build could not
  share a code dataset version with `P2.1-02` (task brief) and the library would need a review for
  a measurement that does not need it. The cost is a dependency on a private class, which the
  tests and the module doc's consumer note guard.
- **Synthetic episodes of exact lengths, not P3.4's.** P3.4 has no episodes of exactly 2048, 5120,
  8192, or 9728 tokens (the longest training episode is 8150, the cap drops longer ones; P3.4
  stats), and a step's cost depends on the length, not on the colors. Exact lengths hit the cap
  exactly and make the points comparable; the format is the notebook's, which `build_sequence`
  checks token by token. Large, nearly square grids reproduce B3's worst case (9330 tokens is
  exactly four 30 by 30 demonstrations and a 30 by 30 query). Attaching P3.4's build would add
  its mounts and guards for nothing measured.
- **One fixed $\bar z$, not the encoder.** The encoder runs once before a training run's first
  step (P3.5-01: 543 s for 2018 pairs, `run.json`), not inside a step, and H's cost does not
  depend on $\bar z$'s values (its task code is a linear map then a LayerNorm).
- **Sessions per run of points with equal $W$, not a process set per point.** A new set of
  processes per point would isolate each point's memory completely, but each costs start-up and a
  model load: 206 s from the script's start to a loaded model in `P3.5-01` (session log), so about
  30 minutes for ten points against about 6 minutes of steps. Two sessions load the model twice;
  each failed point costs one more session. Within a session the allocator's peak counters
  restart at each point, so its peak allocated memory is its own (the cache is never emptied: next
  item).
- **A failed process leaves at once.** After an out-of-memory error inside DDP's backward, DDP's
  reducer is left mid-iteration and a peer may wait in the gradient all-reduce for an answer that
  never comes; recovering in place would need a new DDP wrapper and could still deadlock. The
  standard pattern is to end the process group: the failing process exits without collectives
  (`os._exit`), torch's process context ends its peers, and a new session continues. A stall
  watchdog (600 s) is the backstop for a hang in which no process exits.
- **The cache is never emptied, and the mix point comes before the long points.** P0.4 asks
  whether an 8192-token step fits in a meta-training run, where lengths are mixed and P3.5's loop
  never empties the cache: when a new segment cannot be reserved, the allocator frees its unused
  cache, retries (a counted retry), and only then fails. So no point empties the cache, and the
  long points come after the shorter ones and the 224 mixed episodes, meeting the allocator as a
  long run would. Alternative: emptying the cache before each point, which isolates each point's
  reserved memory. Lost: it measures a compact allocator that no training run has, so it
  under-reports the reserved footprint and hides the retries that signal pressure. The points stay
  separable: the peak allocated memory is each point's own, and the reserved memory at its start
  is recorded beside its peak reserved memory. The single-process 8192 point follows only the 512
  one, so it reads the step's need with little inherited cache. The 10% bound covers fragmentation
  beyond these minutes of mixed lengths.
- **Headroom from reserved memory, not allocated.** An allocation fails when the allocator cannot
  reserve: reserved memory includes the cache's fragmentation (here the cache the earlier points
  left, as in training), and the CUDA context and NCCL's buffers occupy the device outside the
  allocator. Peak allocated memory is reported beside it.
- **A median of 6 timed steps after 2 untimed.** The first steps carry one-time costs (cuBLAS and
  kernel setup, AdamW's states at the first update, DDP's bucket rebuild, the allocator's growth):
  `P3.5-01`'s first four steps took 3.1 to 4.3 s against a mean of 2.3 s. The median resists a
  stray slow step; the minimum and maximum are reported.
- **Per-length points as `P3.5-01` ran (4 GPUs, one episode per GPU per step).** B5 asks for step
  seconds at given lengths on this code, and per-length costs are the projection's building
  blocks. The long run's accumulation 8 is measured too (the mix point), on its own lengths.
- **The projection: interpolation, not a fitted curve.** `P3.5-01`'s step times rise faster than
  linearly with the step's longest episode (a quadratic fit gives
  $0.19 + 7.1 \times 10^{-4} T + 5.6 \times 10^{-8} T^2$ seconds), so a straight line would err at
  the ends; linear interpolation between six measured lengths needs no model form, and they span
  the histogram (its longest episode is 8150). Constant below 512 errs high for the shortest
  episodes.
- **The slowest process, not the mean.** With accumulation the processes run their micro steps
  independently and wait for each other only at the last micro step's all-reduce, so a step lasts
  as long as the slowest process's sum; with P3.5-01's costs, the mean sum is about 11.0 s and the
  slowest of four about 14.5 s (estimate, the brief's section 6).
- **Calibrated on the measured accumulation-8 step.** The mix point's six steps alone would
  estimate the expected step with sampling noise (each draws only 32 lengths); the model alone
  would carry its own bias (for example the 7 sequence builds per step it does not count). The
  ratio of measured to predicted on the mix's own lengths removes the model's bias without the
  mix's sampling noise, and the model then integrates over the whole histogram.
- **Lengths uniform within P3.4's 1024-token bins.** Only the histogram came back from Kaggle (A0:
  episode builds stay there); the draw's mean is 1368 tokens against the measured 1348.4, 1.5%
  long, which errs toward a slower step.
- **Kill criteria for a measurement.** There is no learning question, so no loss criterion. A step
  slower than 60 s per micro step (about 5 times the 9.5k step's estimate) means a broken
  environment (no GPU, a throttled node) and stops the run; the wall budget and the stall watchdog
  bound the cost of a hang.
- **4xL4, no fallback.** The long run, P3.5-01, and P3.5-02 run on `NvidiaL4`; `NvidiaL4X1` ran on
  2xT4 when requested (`P0.3-01`), and T4 has no bf16. The single-process points stand for a
  one-GPU L4 machine.
- **Everything else is the training step's:** precision (base bf16, H, the delta, and the loss
  fp32), non-reentrant checkpointing, FlashAttention 2, AdamW with D2L's settings, clipping at 1.0
  (`train_hypernet.md`, section 3, gives each reason).

## 4. How to run it

**CPU smoke (local, required before any GPU run; BRIEF A0: CPU only).** From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python scripts/train_throughput.py \
    --config configs/train_throughput.yaml --smoke
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python scripts/train_throughput.py \
    --config configs/train_throughput.yaml --smoke --resume runs/P3.6-smoke/run_logs
```

The first runs every kind of point on the tiny random Qwen3 (hidden 64, 4 layers) and the
stand-in tokenizer: a session of 2 gloo processes (64, 128, a mix with accumulation 2, 256 tokens),
then one of 1 process (64, 256, H's forward), and writes the outputs under
`runs/P3.6-smoke/run_logs/` (measured: the brief's section 11). The second keeps every finished
point and starts no session. Tests: `CUDA_VISIBLE_DEVICES="" .venv/bin/pytest
tests/test_train_throughput.py`.

**Kaggle (`P3.6-01`), after the human approves the brief.** The script runs in a script notebook
(P0.5d, `docs/runbooks/build_submission.md`, script mode) built from
`configs/script_P3.6-01.yaml`: notebook `alyahya737/arc26-p3-6-throughput` on `NvidiaL4`, the
model, the code dataset (wheel, this script, `train_throughput.yaml`), and the install notebook
at version 2; no notebook outputs are attached. The notebook passes `--commit` (`{commit}`); the
hard stop is `script_kill_s` 3000 s. The wheel is the base's (`arc26/` untouched), the same as
`P2.1-02`'s build, so one code dataset version can hold both builds' `arc26-code/` files (the
`P3.2-02` and `P3.5-02` precedent); check the two wheels' sha256 first:

```bash
.venv/bin/python scripts/build_submission.py --tag P3.6-01 --config configs/script_P3.6-01.yaml
sha256sum dist/P3.6-01/arc26-code/arc26-0.0.1-py3-none-any.whl   # the brief's wheel hash
# <shared>/arc26-code: dist/P3.6-01/arc26-code/ plus dist/P2.1-02/arc26-code/ (task/P2.1-02)
.venv/bin/python scripts/kaggle_ops.py dataset-version <shared>/arc26-code \
    -m "P2.1-02 <commit>, P3.6-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P3.6-01 --run-id P3.6-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p3-6-throughput
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p3-6-throughput --task-id P3.6-01
```

The notebook runs `python train_throughput.py --config train_throughput.yaml --commit <sha>` in
`/kaggle/working`. Make a dataset version only when no run waits in the queue, and push about 5
minutes after it; at most two batch GPU sessions run or wait per account. If `NvidiaL4` waits in
the queue for 30 minutes, the orchestrator tells the human; there is no fallback.

**Plain GPU machine** (four bf16 GPUs with CUDA for the 4-process points):

```bash
uv venv --python 3.11 .venv && uv pip install -e ".[dev,hyper]"
.venv/bin/python scripts/train_throughput.py --config configs/train_throughput.yaml \
    --model-dir /data/qwen3_4b_grids15_sft139 --out-dir runs/P3.6-01/run_logs
```

With fewer GPUs, copy the config and change the points' `gpus` and the projection's `gpus`.

**Expected cost of `P3.6-01`** (estimates, brief section 6, from `P3.5-01`'s measured times):
about 710 s of script (about 180 s to the first session's loaded model, 345 s of 4-process
points, 90 s to the second session's model, 85 s of 1-process points), about 13 minutes of
session with the notebook's check, install, and logs: 0.22 h of 4xL4, about 0.44 quota hours at
twice the T4 rate. Peak memory allocated per GPU about 12.8 to 13.1 GiB at 8192 tokens and 13.4
to 13.8 GiB at 9728, reserved 5% to 15% above (estimate, brief section 6), of an L4's 22.5 GiB;
disk: a few hundred KB of logs, no checkpoints.

**Resume.** Each point's results are written when it ends, so a lost session loses at most the
point in progress. `--resume <previous out_dir>` keeps every point that ended `complete` or
`oom` under the same point settings, training step, and seed, and runs the rest. On Kaggle, as
for P3.5 (`train_hypernet.md`, "Resuming across sessions"), a continuation notebook of another
slug pins this notebook's output in `notebook_inputs` and passes
`--resume {input:alyahya737/arc26-p3-6-throughput}/run_logs`, under its own brief.

## 5. Outputs

| File (in `out_dir`, pulled) | Contents |
|---|---|
| `config.yaml` | The resolved config. |
| `run.json` | Run id, smoke flag, argv, start, commit (from `--commit` on Kaggle), library versions (torch, CUDA, transformers, numpy, flash_attn), the model folder found, and each session: processes, points, how it ended, setup failure, seconds. |
| `status.json` | `complete`, `stopped`, `killed`, or `failed`, the exit code, the reason, each point's outcome, the decision rule's reading, the wall time. |
| `measurements.jsonl` | One line per point: kind, processes, accumulation, tokens (and the tokens the base read), outcome and reason, the timed steps' seconds (slowest process per step) with median, mean, minimum, maximum, and the untimed ones, the micro step's statistics, the step's remainder, non-finite steps, memory (peak allocated and reserved, the reserved memory at the point's start, memory outside the allocator, headroom and its share, allocator retries, the resident memory before the point, per process), the device, the attention kernel, the model's load seconds, H's size; for the mix point each timed step's lengths by process and micro step; for H's forward the milliseconds per call in each mode. |
| `projection.json` | The micro-step table, $t_\text{ddp}$, $\bar r$, the draw (its mean against the histogram's and the measured mean), the model step (mean and percentiles), the calibration, the projected step, one pass, and the weekly table. |
| `summary.md` | Status, device, the points' table, H's forward, the decision rule's reading, the projection. |
| `points/<name>/rank<k>.json` | Each process's raw result: every step's seconds and tokens, every micro step's H forward and forward-backward seconds, the memory counters before and after, the error and traceback of a failure. |
| `points/<name>/timing-rank<k>.jsonl` | One `{puzzle_id, op, seconds, tokens}` per op: `h_forward` and `train_step` per micro step (P3.5's ops), `step` per optimizer step, `h_forward_eval` and `h_forward_train` per call of H alone. |
| `sessions/s<i>/rank<k>.json` | Each process's last state: setup, running (point, step), done, stopped, or failed with the error; the stall watchdog reads its time. |
| `error.log` | The traceback of a failure of the script itself (a point's failure is in its results). |

The notebook adds its own `env.json`, `pip_freeze.txt`, `candidates.json`, and `stages.json`.

## 6. How to read the results

The pre-registered rule is the brief's section 1; `summary.md` and `status.json` read it off the
results:

| Result | Rule (8192 tokens on 4 GPUs) | Next |
|---|---|---|
| Pass | complete, headroom at least 10% of the device on the tightest GPU, no allocator retries | B5 P3.6 accepted: the meta-training brief keeps the 8192 cap and the step as P3.5 runs it, with this run's step time and kill criteria; P0.4 gets the weekly table |
| Borderline | complete, but headroom under 10% or allocator retries | accepted with a memory guard in the meta-training brief (a peak-memory log and `expandable_segments`, or a lower cap) |
| Fail | out of memory | the cap (a P3.4 rebuild) or the checkpointing changes before meta-training; then another P3.6 run |
| No result | an error, a stall, a kill, or the budget | pull the logs, fix, rerun under an amended brief |

- **9728 tokens:** `fits` (complete, the same memory bound) means a cap above 8192 could fit
  later; `tight` or `does not fit` keeps 8192.
- **H's forward:** the evaluation-mode median under 50 ms meets B5 P3.3 on an L4.
- **The projection:** the expected step $\bar s$ (calibrated) and the weekly table are P0.4's
  inputs; a calibration factor more than 15% from 1 means the model misses a cost, and the
  meta-training brief then quotes the mix point's measured step beside it. The step's seconds
  per length should rise smoothly; a point far off the others' trend (say a 9.5k step under the
  8192 one) points at a measurement problem, not a cost.

## 7. Failure modes

- **Out of memory at a point.** A recorded result. First: `attn_implementation` in the results
  (SDPA instead of FlashAttention 2 costs memory at long sequences); then the memory counters in
  the point's `rank<k>.json` (peak allocated against the device). At 8192 it is the Fail row.
- **Every point of a session errs with "session setup failed".** The model search found no
  folder or two (the message lists them), `flash_attn` failed in an unexpected way, or the
  process group did not form; read `sessions/s<i>/rank<k>.json` for the error. Fix and rerun
  under an amended brief; `--model-dir` names the folder directly.
- **A stalled session** (`stalled: no process reported progress`): a process hung, most likely in
  NCCL after a peer failed without exiting. The session's later points run in the next session;
  look at the last states in `sessions/s<i>/`.
- **Step-time kill (exit 4):** a step over 60 s per micro step. Check `device_name` and the
  attention kernel; a run on the wrong device or a throttled node shows there.
- **Wall budget (exit 3), the notebook's hard stop, or Kaggle's `Timeout`:** the finished points
  are in `measurements.jsonl`; resume (section 4) for the rest.
- **Non-finite steps** (`nonfinite_steps` above 0): the step found a non-finite loss or gradient
  norm and skipped its update; its time is kept. Random colors at a warmup learning rate should
  give none; report it, since P3.5's kill criterion would stop a training run there.
- **The base read another length than the point's** (`tokens_actual`): the real tokenizer splits
  the format otherwise than the notebook vocabulary; the measurement stands at the length read.
  Compare with P3.2's tokenizer preflight before trusting the lengths of other episodes.
- **Divergence:** not applicable; H's updates during the measurement change nothing measured.
- **Queue wait over 30 minutes:** the orchestrator tells the human; no fallback accelerator.
- **Config error (exit 2):** the message on stderr names the key; nothing is written.
