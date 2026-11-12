# Runbook: the KV-cache A/B probe (`scripts/fn_kvab.py`)

Task P5.5 (`docs/tasks/P5.5.md`). Run `P5.5-01` on Modal's 2 x L40S: job
config `configs/modal_P5.5-01.yaml`, run config `configs/P5.5-01-ab.yaml` (version 11's
`configs/P5.2-01.yaml` revision 16 and the kernel writer's fragment `configs/P5.5-01.yaml` by
`extends`, plus `entry`, `bundle`, P5.2-07's boot flags, and `kvab`), brief
[`docs/prerun/P5.5-01.md`](../prerun/P5.5-01.md). Tests `tests/test_fn_kvab.py`. The kernels and
their edits are the kernel writer's (`scripts/fn_kvq/`, `tests/test_fn_kvq.py`); the research
notes are `docs/research/fn_kv_compress_map.md` (M), `fn_kv_compress_methods.md` (Q) and
`fn_l4_memory.md` (L) on `task/ideas`. The gateway and the probe job are
`modal_ops.md` ("Probe jobs"); the engine bisect it borrows from is
[`fn_bisect.md`](fn_bisect.md); version 11 is [`fn_probe.md`](fn_probe.md).

## 1. Why this script exists

Flash-Next's 12 QSA layers keep a paged FP8 K and V cache of 6,912 B per token per GPU at TP 4
(L: K 3,072, V 3,072, the BF16 indexer 768). The kernel writer packs K and V into INT4 with
FP16 scales per group of 32 channels, $12 \times 2 \times 144 + 768 = 4{,}224$ B per token, so
the pool holds $6912 / 4224 = 1.636$ times the tokens (`scripts/fn_kvq/layout.py`). On Kaggle's
4 x L4 that is about 277,700 tokens per GPU instead of revision 15's 169,728: more streams or
longer thinking for the planner. Two things must hold before it ships in a `P5.2-01` revision
for the Sunday or Monday submission: the kernels and the hash-guarded edits run on sm89 (the L4's
architecture), and the model's outputs do not move: the same next tokens, the same
log-probabilities within a small margin, no flatter distribution, no longer thinking (Q: MXFP4 KV
on Qwen3.8-27B kept accuracy but made generations 43% to 61% longer, SGLang PR #38724), and no
slower decode.

The script answers both in one Modal job, before the submission: a kernel check on one GPU, then
boots of the full model with the FP8 cache (A), the 4-bit cache (B), and a third boot chosen by
B's reading (C), each measured the same way. Its decision row (the brief, section 1) decides
whether the KV4 edits, the V-only fallback, or nothing goes into the next Kaggle revision.

## 2. What it does, step by step

1. **The notebook** (`build_fn_notebooks.notebook(config, script="fn_kvab.py")`): cell 1 writes
   the placeholder `submission.json` and checks the code dataset's files against the build's
   sha256 (the wheel, `fn_probe.py`, `fn_fake_sglang.py`, `fn_bisect.py`, `fn_kvab.py`,
   `fn_fake_kvab.py`, and the bundle `fn_kvq.tar`); cell 2 runs `fn_kvab.py`; cell 3 checks the
   placeholders.
2. **The outer process** runs `fn_probe.py`'s own stages unchanged: inputs (the transcripts input
   among them), local copy, Python 3.12, the venv from the wheelhouse with every edit of
   `patches.items` (revision 16's five and the fragment's KV4 ones, which change nothing while
   `ARC26_KV4` is unset), and the CUDA links. Then it starts the inner process in the venv.
3. **Tasks:** `fn_probe.stage_tasks` loads the 3 DEV tasks (`solve.task_ids`) and renders their
   prompts with the model's tokenizer.
4. **Phase K** (`stage_kernel_check`): the bundle is extracted to the scratch folder (its
   `scripts/fn_kvq/` and `tests/test_fn_kvq.py`, the repository's layout), and its commands run in
   order on GPU 0 alone (`CUDA_VISIBLE_DEVICES=0`), with the tree and the probe image's pytest on
   `PYTHONPATH`: `bench.py --out kvq_bench.json` (Triton against the reference, timings against
   the FP8 gather, the INT4 error), then `pytest -m gpu tests/test_fn_kvq.py`. Each command's
   timeout is cut to the phase's 140 s. A failure or a timeout ends the run here: no boot starts.
5. **The teacher-forced text** (`tf_text_ids`): record `7c66cb00/p0/0` of P6.9-01's
   `direct_transcripts.jsonl` (its file sha256 checked again), its system and user messages
   rendered with Flash-Next's chat template, then the assistant's reasoning, encoded once and cut
   to $L = 26{,}000$ ids. Only the ids' sha256 and count are written.
6. **Phase M** (`stage_ab`): for each variant in order, its `run_if` and its window decide whether
   it boots; then `boot_variant`:
   1. the KV4 edits it requires must read `applied` in `patches.json`, else `patch_missing`;
   2. the server starts with version 11's flags, `base_flags` (`--random-seed 0`), and its own
      flags; with the inner environment, `base_env` (the prompt log-probability chunk of 256 rows,
      and the fragment's KV4 variables removed), and its own variables (B and C: `PYTHONPATH`
      the bundle's `scripts/` folder, since the edits import `fn_kvq` and the arc26 wheel does not
      carry it; A never imports it); health is polled until
      the variant's deadline less `measure_floor_s`;
   3. the pool and the request limit from `/get_server_info`, each GPU's memory from
      `nvidia-smi`;
   4. the teacher-forced pass (`measure_tf`): one request of the $L$ ids, one new token, greedy,
      `return_logprob`, `logprob_start_len` 0 (SGLang caps the radix prefix match at it, so
      nothing comes from the cache), `top_logprobs_num` 20. Per position $t \ge 1$ it keeps the
      sent token's $\ell_t = \log p(x_t \mid x_{<t})$, the top-1 token $a_t$, and the truncated
      entropy over the top $k$
      $$\hat H_t = -\sum_{i=1}^{k} e^{\lambda_{t,i}} \lambda_{t,i},$$
      checks that the server read exactly the ids sent, and compares with A (`compare_tf`):
      $$\text{agree} = \frac{1}{n}\sum_{t}[a^V_t = a^A_t], \qquad
      \bar\Delta = \frac{1}{n}\sum_t \lvert \ell^V_t - \ell^A_t \rvert, \qquad
      \rho_H = \frac{\sum_t \hat H^V_t}{\sum_t \hat H^A_t},$$
      over the $n = L - 1$ positions, with the 99th percentile of
      $\lvert \ell^V_t - \ell^A_t \rvert$ by nearest rank;
   5. decode (`measure_decode`) at $s = 1, 2, 3$ concurrent requests of the config's short
      prompts, 512 new tokens each, greedy, EOS ignored: the per-stream rate is the median of the
      server's `gen throughput` lines logged with $s$ running requests, divided by $s$ (the mean
      wall rate per request is kept beside it, and used when a boot has no such line);
   6. the DEV tasks (`measure_dev`): the 3 tasks at once through `fn_probe.attempt_task` on the
      plain arm with the model's sampling (temperature 1.0, top-p 0.95, top-k 20): the thinking
      turn to $16{,}896 - 512 = 16{,}384$ tokens, then the forcing turn writes the answers. The
      thinking turn's end is the length reading: its tokens if it ended by itself, the cap if it
      hit it, censored (`cut`) if the deadline cut it;
   7. 3 s later the liveness check; then the stop (`stop_boot`: the process group, every process
      tagged with the variant's variable, the port, and the GPUs back within 2 GiB of their
      reading before the boot), the log's first fatal line and marks, the tail and a bounded copy.
7. **The reading** of each boot (`judge`, the brief's rules in `kvab.readings`), written into
   `kvab.json` after every boot, and the decision (`decision`): B's reading.
8. **The variants** of `P5.5-01`: `a_fp8` (FP8, P5.2-07's command line); `b_kv4`
   (A's flags, `--kv-cache-dtype fp8_e4m3` included: `ARC26_KV4=1`, `ARC26_KV4_HADAMARD=1`,
   `ARC26_KV4_GROUP=32` make the edits select the packed pool subclass, `fn_kvq.md` section 3),
   only after A served;
   `c_vision` (B plus `--json-model-override-args '{"language_model_only": true}'` and
   `--cuda-graph-backend-prefill disabled`, L risk 5) when B passes on quality; `c_vonly`
   (`ARC26_KV4=v`, K FP8, V INT4) when B fails on quality. C is the first boot dropped: it starts
   only with 270 s before the run's stop.

Time. With $t_0$ the notebook's first cell, $E$ the development `end_s` (1,850) and $K$ the kill
reserve (90 s), a variant starting at $t$ runs only if $t_0 + E - K - t \ge w$ (its
`min_window_s`: 600, C 270), and its deadline is $d = \min(t + B, t_0 + E - K)$ with $B$ = 900 s.
Each measurement starts only with its floor left before $d$ (pass 60 s, each decode 45 s, the DEV
tasks 120 s), so the run ends by $t_0 + E$ whatever the boots do.

## 3. Why it is built this way

| Choice | Alternative considered | Why it lost |
|---|---|---|
| Phase K inside the phase M job, on GPU 0 of the 2 x L40S | A separate one-GPU job (L4 at 0.000222 USD/s, or one L40S), before the A/B | A second job pays its own container start, fetch, Python unpack (80.3 s in P5.2-07) and install (23.8 s) again, and its worst case (about 0.4 to 0.5 USD with its restart, fetch and driver terms) would come out of the same 4.5 USD cap the A/B needs. Inside, phase K costs at most 140 s x 0.00142812 = 0.20 USD and shares the venv phase M uses; its failure still stops everything before a boot |
| A new entry beside `fn_probe.py`, importing it and `fn_bisect.py` | A mode of `fn_probe.py`, or new variants of `fn_bisect.py` | `fn_probe.py` and `fn_fake_sglang.py` are revision 16's Kaggle code (their bytes are in the code dataset); `fn_bisect.py` is P5.2-06's record. Importing keeps the install, the edits and a boot's mechanics exactly theirs (the tests compare the bytes) |
| The KV4 edits applied at the install, switched by the variable | The edits applied per variant and restored (`fn_bisect.apply_edits`) | The kernel writer's fragment ships them as `patches.items` for the Kaggle revision, where `ARC26_KV4` switches them; testing that exact form is the point. A's pool, read against P5.2-07's 448,192, shows the edits inert while unset (brief, section 9) |
| B and C read `fn_kvq` from the extracted bundle (`PYTHONPATH={kvq}`) | `fn_kvq` in the arc26 wheel, or a `fn_kvq/` folder in the code dataset | The wheel and the code dataset are revision 16's competition path, byte for byte (the fragment's own note); the bundle is already among the code files and extracted for phase K, so the server reads the same bytes the kernel tests ran |
| `kvab.base_env` removes the fragment's KV4 variables from every boot | Override `engine.env` in the run config | `fn_probe.stage_cuda` stringifies every `engine.env` value, so a null would become `"None"`; removing them per boot keeps the fragment untouched and makes each boot's KV4 state its own (`tests/test_fn_kvab.py`) |
| `extends` as a list (`resolve_extends`) | Ask the kernel writer to add `extends` to the fragment | The fragment is the Kaggle revision's shape (a patch over revision 16); a list merges both without editing their file and keeps the single-name form's meaning for every earlier config |
| The bundle: one tar of `scripts/fn_kvq/` and its test file | Each file as a flat code file | Phase K must run the tests as the repository lays them out (`scripts/fn_kvq/` as a package, `tests/`); flat names collide (`__init__.py`) and lose the import path. The tar's members are sorted with fixed times, owners and modes, so two builds are byte-identical and the code check binds it |
| pytest in its own folder of the probe image (`/opt/arc26-pytest`) | pytest in the venv, or in the image's Python | The wheelhouse has no pytest and the GPU stage no network; the image's Python is 3.13 and the venv's 3.12, so its site-packages cannot be shared; a `--target` folder of the pure-Python wheels (pins of the dev venv, `--no-deps`: packaging and pygments are the venv's) is read only by phase K's `PYTHONPATH` |
| A teacher-forced pass with prompt log-probabilities | Compare greedy generations of A and B token by token | A greedy continuation diverges at the first flipped token and then compares unrelated texts; teacher forcing scores every position of one fixed text in one request, so agreement and $\bar\Delta$ cover 25,999 positions with no sampling noise |
| P6.9-01's record fetched inside Modal (a notebook-output input with its digest) | A committed text, or the text built from the challenges | Its prompt holds an evaluation task's grids (competition data is never committed), and a synthetic text would not be the long reasoning the cache must hold. `7c66cb00` is unsolved in P6.9-01, so the text holds no correct answer; the cut at 26,000 ids (inside the task's 20k to 30k band) keeps its 5,636-token prompt and about 20k of reasoning |
| Top-20 truncated entropy | The full-vocabulary entropy | SGLang returns at most the top $k$; the top 20 hold almost all of the mass of a thinking model's next token (`mean_top_mass` is recorded, so a flatter tail shows as a lower mass) and it is computed identically in every boot (Q's suggestion) |
| `SGLANG_LOGPROB_CHUNK_SIZE=256` in every boot | The fork's default 2,048 rows | 2,048 rows of the vocabulary in FP32 is about 2 GB next to the 3.35 GB P5.2-07 had free after its capture (its `boot.json`); 256 rows bound it at about 0.25 GB. Decode never computes prompt log-probabilities, so the rates are untouched |
| `--random-seed 0` as the DEV tasks' seed | A per-request `sampling_seed` | The fork honours it only under `--enable-deterministic-inference`, and the flashinfer sampler asserts it unset (`sampling_batch_info.py`, `sampler.py` at `d00d88e`); deterministic inference changes the kernels being timed. The fixed server seed makes A's and B's first tokens identical until their first numeric difference |
| The DEV tasks on the plain arm | The tool arm (revision 16's `dev_arms`) | The tool loop's Python calls add CPU time and turns that differ between boots, which would blur the thinking-length reading; the plain arm's thinking turn is one request to the cap |
| A thinking cap of 16,384 (`solve.cap` 16,896, `forcing_tokens` 512) | Revision 16's 65,536 cap and 6,144 forcing tokens | 3 streams at about 60 tok/s reach 16,384 in about 280 s per boot; 65,536 would take 18 minutes per boot. 6,144 forcing tokens would cut the thinking at 10,752 |
| The decode rate from the server's `gen throughput` lines at exactly $s$ running | The wall rate per request | The wall time includes the prefill and the queueing of the last request; the server's lines measure decode alone (P5.2-07: 247.7 to 252.3 tok/s at 4 running, a 1.9% spread) |
| P5.2-07's boot flags (fraction 0.92, 8 running, chunk 4,096, Mamba 44, graphs 1 to 8) for every boot | Revision 16's L4 values | P5.2-07 served on this hardware with them (pool 448,192, boot 391 s), so A is a measured reference; revision 16's 0.95 was sized for the L4's 22.5 GiB and is untested on 48 GB. The ratios B/A do not depend on these values |
| C chosen at run time by B's quality (`run_if`) | A fixed third boot | The methods note's fallback (V only, 1.24x) is only worth a boot when B fails on quality; when B passes, the vision skip is the next memory step (L, risk 5). The rule is in the config and pre-registered |
| `Timeout` 2,300 s, fetch 600 s, `term_s` 30, the container of P5.2-07 | A smaller container (8 cores, 80 GiB) for about 140 s more | P5.2-07's host peaked at about 64 GiB (`host_memory.jsonl`, min 31.6 GiB free of 96); 80 GiB would leave 16 GiB for three boots in sequence, a risk for 140 s. C is dropped first instead (the orchestrator's rule) |

## 4. How to run it

**Modal** (the orchestrator, every call through `scripts/modal_ops.py`, after the brief's
approval): the commands of `modal_ops.md` section 4 with `P5.5-01`:

```bash
.venv/bin/python scripts/build_modal_job.py --config configs/modal_P5.5-01.yaml   # at Code-commit
.venv/bin/python scripts/modal_ops.py run dist/P5.5-01 --run-id P5.5-01 --dry-run
.venv/bin/python scripts/modal_ops.py run dist/P5.5-01 --run-id P5.5-01
.venv/bin/python scripts/modal_ops.py status --run-id P5.5-01                     # every 5 min
.venv/bin/python scripts/modal_ops.py logs --run-id P5.5-01 --task-id P5.5-01
```

Inputs: P5.2-07's on the volume `arc26-modal-fn` (the model at `c6af9fee`, the wheelhouse, the
`arc26-fn-env` output, the challenges) and P6.9-01's notebook output
(`alyahya737/arc26-p6-3-shot/versions/3`, fetched once, a few MB; its digest is checked). Expected
wall time about 1,700 s of notebook (the brief, section 6); GPU memory as P5.2-07 (about 42 GB
per GPU at the pool); local disk about 20 GB (the venv and caches), host memory about 64 GiB.

**The CPU smoke** (no GPU, no network; the fake server `fn_fake_kvab.py`, stand-ins for the
kernel bundle in `tests/testdata/fn_kvab/`):

```bash
.venv/bin/python scripts/modal_ops.py --smoke --config configs/modal_P5.5-01.yaml
.venv/bin/pytest -m "not gpu" tests/test_fn_kvab.py
```

**A plain GPU machine** with the wheelhouse venv: `python scripts/fn_kvab.py --config
fn_config.json --root <a folder laid out as /kaggle>`; `fn_config.json` is the resolved run
config the build embeds (`dist/P5.5-01/job.json`, `probe.config_resolved`).

**After the run** (local, CPU): the readings again with the config's rules, and the DEV answers
scored (development scoring only):

```bash
.venv/bin/python scripts/fn_kvab.py --config runs/P5.5-01/probe/attempts/1/kaggle_working/fn_config.json \
  --reread runs/P5.5-01/probe/attempts/1/kaggle_working/run_logs/fn --solutions <solutions>
```

## 5. Outputs

Under `run_logs/fn/` (pulled by `modal_ops.py logs`, allowlisted by `PROBE_PULL_PATTERN`):

| File | Contents |
|---|---|
| `kvab.json` | the text's record (source, file sha256, tokens, ids sha256), one row per variant (outcome, flags, environment, command line, pool, memory, the pass's summary and comparison, decode per stream count, DEV lengths, stop, marks, first error, reading), the decision |
| `kvq_bench.log`, `kvq_pytest_gpu.log`, `kvq_bench.json` | phase K: each command's output (head and tail), the benchmark's report |
| `kvab_<n>_<name>_tf.json` | per position: `lp`, `top1`, `entropy`, `mass` (numbers only, never the ids or the text) |
| `kvab_<n>_<name>_dev.jsonl` | each DEV task's record (turns, tokens, forcing, answers as grids) |
| `kvab_<n>_<name>_transcripts.jsonl` | each DEV task's generated text |
| `kvab_<n>_<name>_tail.log`, `_engine.log` | the boot's last 300 lines and its whole log, bounded at 2 MB |
| `engine_commands.jsonl`, `timing-inner.jsonl` | each boot's command line; a timing record per boot, pass, and decode request |
| `run.json`, `state.json`, `inputs.json`, `install.json`, `patches.json`, `versions.txt`, `summary.md` | `fn_probe.py`'s outer records and the run's table |

## 6. How to read the results

`summary.md` has one line per boot; `kvab.json` the numbers. The decision is B's reading:

- **Good:** B's pool at least 1.6 times A's, top-1 agreement at least 99%, $\bar\Delta \le 0.02$,
  $\rho_H \le 1.05$, the median thinking length at most 1.20 times A's (when judged), the
  per-stream decode rate at least 0.95 of A's at each of 1, 2 and 3 streams, no request error, no
  fatal log line. The KV4 edits go into the next `P5.2-01` revision with B's flags and variables.
- **Bad:** any of these misses; `judged.misses` names each. A quality miss (agreement,
  $\bar\Delta$, $\rho_H$, length) chooses `c_vonly`, whose reading decides the V-only fallback; a
  pool, speed or error miss alone chooses `c_vision` and the kernel writer's next fix.
- **Fault:** phase K failed (no boot), A did not serve, or B did not boot, met a fatal line, died,
  or its pass failed. Revision 16 stays as it is; the logs name the kernel.

Also read: A's pool against P5.2-07's 448,192 (the edits inert while unset); B's `kv_dtype` and
K and V sizes (the marks); `mean_top_mass` (a flatter distribution lowers it); the DEV tasks'
`ended` (natural, cap, cut). Noise: the pass is deterministic per boot (no sampling), so
agreement, $\bar\Delta$ and $\rho_H$ carry no sampling noise but cover one text; decode varies by
about 2% between lines (P5.2-07); the length reading is weak (3 tasks at temperature 1, the
brief's section 1).

## 7. Failure modes

| Symptom | Where it shows | First thing to change |
|---|---|---|
| Phase K fails or times out | `state.json` `kernel_check`, `kvq_*.log` | The kernel writer's fix; nothing booted, about 0.4 USD spent |
| `patch_missing` for B | `kvab.json` row, `patches.json` | A KV4 edit's `sha256_before` does not match the wheel's file; the fragment's chain of hashes |
| B `exited_at_boot` with `: error: argument` | the row's `first_error` | A flag the fork refuses: B's `flags` in the run config |
| B `boot_timeout` | `health` (last status), the tail log | A slow first capture of the new kernels: a larger `budget_s` needs a larger `Timeout` and cap |
| `CUDA error` or `illegal memory access` in B | `first_error`, `kernel_frame` | The kernel `kernel_frame` names; the pin run of `fn_bisect.py` (`CUDA_LAUNCH_BLOCKING=1`) on that variant |
| `CUDA out of memory` in the pass | the row's `tf.error`, the engine log | A smaller `SGLANG_LOGPROB_CHUNK_SIZE` in `base_env`, or a shorter `tf.max_tokens` |
| The pass's `ids_match` false | the row's `tf` | The server re-tokenized or truncated: the context length or the request's `input_ids` |
| No decode lines at a stream count | `decode.<s>.log.lines` 0 | The wall rate is used; a longer `max_tokens` gives more lines (one per 40 steps) |
| DEV tasks `cut` | `dev_lengths.cut` | The boot came late; the length is not judged. A shorter `solve.cap` or a larger `Timeout` |
| A's pool far from 448,192 | `kvab.json` A's `pool` | The edits are not inert: compare A with P5.2-07's flags and the fragment's pool edits |
| The fetch refuses the transcripts | `fetch.json` | Another version of `arc26-p6-3-shot`: the job's handle (the digest is the local pull's) |
