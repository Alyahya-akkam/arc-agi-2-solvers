# Runbook: the engine bisect (`scripts/fn_bisect.py`)

Task P5.2m (`docs/tasks/P5.2m.md`). Run `P5.2-06` on Modal's 2 x L40S: job
config `configs/modal_P5.2-06.yaml`, run config `configs/P5.2-06.yaml` (version 11's
`configs/P5.2-01.yaml` by `extends`, plus `entry` and `bisect`), brief
[`docs/prerun/P5.2-06.md`](../prerun/P5.2-06.md). Tests `tests/test_fn_bisect.py` and
`tests/test_modal_probe_p5206.py`. The variants come from the research note
`docs/research/fn_sm89_illegal_access.md` on `task/ideas` (`b2a5a9e`, cited as N). The gateway
and the probe job are `modal_ops.md` ("Probe jobs"); version 11 is
[`fn_probe.md`](fn_probe.md).

## 1. Why this script exists

`P5.2-05` (30 Oct, about 4.4 USD) booted version 11 on 2 x L40S (sm89, TP 2 + EP 2) and met
`an illegal memory access` in both boots: boot 1 in the decode CUDA graph capture (reported at
`fused_marlin_moe.py:423`, our `torch.sum`), boot 2 (BF16 KV, no graphs) at the first prefill
after `Uvicorn running` (reported while Triton loaded `_fused_gate_sigmoid_mul_add_kernel`). CUDA
reports a fault at a later checked call, so neither traceback names the kernel, and Kaggle's 4 x L4
(TP 4 + EP 4, the same chip) would fail the same way. N reads the cause from the fork's code: at
EP above 1 the dispatcher marks the other rank's experts $-1$, the GPTQ Marlin MoE call passes no
`expert_map`, so the kernel runs with `is_ep` off and reads expert $-1$ as a real one.

One job answers which change makes a boot plus a request succeed, and at what decode rate and
KV pool, before the submission: it installs version 11 once and boots each candidate in turn.

## 2. What it does, step by step

1. **The notebook** (built by `build_fn_notebooks.notebook(config, script="fn_bisect.py")`): cell
   1 writes the placeholder `submission.json` and checks the code dataset's files against the
   build's sha256 (now four: the wheel, `fn_probe.py`, `fn_fake_sglang.py`, `fn_bisect.py`); cell
   2 runs `fn_bisect.py`; cell 3 checks the submission.
2. **The outer process** (the image's Python) runs `fn_probe.py`'s own stages, imported from the
   same folder: `inputs`, `local_inputs`, `python` (P5.2-00's CPython 3.12), `install` (the venv
   from the wheelhouse, then version 11's three hash-guarded edits: `fa2_scratch_dtype`,
   `logits_gather_nccl`, `marlin_moe_torch_sum`), `cuda`. No precache, no tasks, no tool test,
   no argument check.
3. **The inner process** (the venv's Python) runs one stage, `bisect`. For variant $i$ in order:
   - **Rules.** `run_if: none_survived` skips it once an earlier variant survived. With $t_0$
     the first cell, $E$ the development `end_s`, $B$ `budget_s`, $K$ `kill_reserve_s`, it starts
     at $t$ only if $t_0 + E - t \ge m$ (`min_variant_s`), with the deadline
     $$d = \min\left(t + B,\; t_0 + E - K\right).$$
   - **Edits.** Each of its edits through `fn_probe.apply_patch` (sha256 before, exact match
     count); the file's bytes are kept. A refused edit ends the variant as `edit_failed`, no boot.
   - **Boot.** `python -m sglang.launch_server` with version 11's flags, the edits' fallback
     flags, `base_flags`, then the variant's (`null` removes a flag); the environment is the
     inner one (version 11's `engine.env`) with `base_env` and the variant's (`null` removes,
     `{scratch}` is the variant's local folder) and `ARC26_BISECT_VARIANT` as its tag. Health is
     polled until 200, the process's exit (`exited_at_boot`), or $d - $ `request.min_s`
     (`boot_timeout`).
   - **Readings.** `/get_server_info`'s `max_total_num_tokens` (the pool); one chat request
     (`/v1/chat/completions`, about 330 prompt tokens, 64 new, greedy); after `settle_s` the
     liveness check. `survived` needs status 200, at least one completion token, and a live
     server; else `request_failed` (alive) or `crashed_in_request` (dead). A survivor gets the
     rate request: 512 new tokens with `ignore_eos`, its wall rate $r = n / s$ and the server's
     own `gen throughput` lines from that request on (their median).
   - **Stop.** `fn_probe.kill_server` on the group (SIGTERM, 60 s, SIGKILL), then SIGKILL to any
     process still tagged, then a wait for the port; the GPU memory after it is recorded. The
     edited files get their bytes back.
   - **Log.** The first fatal line (`error_patterns`, minus `ignore_patterns`), the traceback
     around it, its message, its rank, the last eight frames, and the launching frame: the
     deepest frame outside `kernel_frame_skip` (torch's module dispatch, `torch/_ops.py`, Triton's
     runtime and compiler), with the Triton kernel launched there if any. Under
     `CUDA_LAUNCH_BLOCKING=1` that frame is the faulting kernel's call. Plus `log_marks` (the
     load's seconds, the pool, Uvicorn, graph capture) and the files named by `logs` (bounded).
   - `bisect.json` is written before each boot (`running` names the variant) and after it.
4. **The variants of `P5.2-06`** (N's ranking, the orchestrator's order): V1 the EP mask edit
   (`marlin.py`, two anchored replacements) with BF16 KV and no graphs; V2 the same edit on
   version 11's settings (FP8 KV, decode graphs); V3 no edit, `--expert-parallel-size 1` (the
   Triton `MoeWNA16` MoE), FP8 KV and graphs; V4, only if none of them survived, the edit with
   `CUDA_LAUNCH_BLOCKING=1`, the fork's kernel API log, and `--disable-overlap-schedule`.

## 3. Why it is built this way

| Choice | Alternative considered | Why it lost |
|---|---|---|
| A new script beside `fn_probe.py`, importing it | A `bisect` mode inside `fn_probe.py` | `fn_probe.py` is version 11's submitted code (`3b3c58b4...`, brief `P5.2-01` section 11); a change would alter the hash every Kaggle brief records and touch a file `task/P5.2` owns. The import keeps the install, the edits, and the CUDA setup exactly those `P5.2-05` ran (`tests/test_modal_probe_p5206.py` asserts the bytes) |
| The run config's `entry`, read by the builder | A new key in the probe job's schema (`modal_job.PROBE_KEYS`) | The schema change would touch `parse_probe` and every probe test; the run config is already embedded and hashed in the notebook, and the gateway rebuilds it from the commit (`job_from_commit`), so `entry` is bound as tightly. The builder refuses any entry outside `PROBE_ENTRIES` |
| Variants in one job, in sequence | One Modal job per variant | Each job pays its container start, the Python unpack (97.6 s), and the install (31.3 s) again (`runs/P5.2-05/vol_fn/state.json`): about 130 s per variant, and each job's bound carries its own preemption restart (0.56 USD), fetch, and driver terms, so three jobs do not fit one 3 USD cap |
| Edits applied per variant and restored | The edits in `patches` for the whole run | V3 must run without the MoE edit, and a variant's result must not depend on the one before; the restore writes the original bytes, so Python recompiles the module at the next import (its mtime changes) |
| `CUDA_LAUNCH_BLOCKING=1` plus the kernel API log, only in V4 | Blocking launches on every variant | Blocking launches slow every kernel and change the stream timing a graph capture depends on; N ranks a fix first, and a pin run only pays when every fix failed (`run_if`) |
| No sanitizer: the pin run uses blocking launches and the kernel API log | `compute-sanitizer --tool memcheck` (N, "Debug settings") | Not in the Pennyroyal wheelhouse (nvcc wheels only) nor the probe image, and 10 to 100 times slower, so one boot would outlast a variant's budget |
| The first fatal line by pattern, then its traceback | The engine log's last traceback | The fault cascades: rank 0 and rank 1 each raise, and SIGQUIT follows (`P5.2-05` boot 2, lines 2475 to 2673); the first names the fault. The import of torchcodec without FFmpeg logs harmless tracebacks at every start (lines 13 to 700), so they are ignored by pattern |
| The chat endpoint for the request | `/generate` with token ids, as `fn_probe.py` solves | The bisect needs only the forward path; the chat endpoint renders the server's own template, so no tokenizer is loaded in the inner process. The prefill and decode kernels are the same |
| The decode rate from the server's `gen throughput` lines | A streaming request timed per chunk | With `--stream-interval 16` a chunk is 16 tokens, and the fake server has no streaming chat; the server's lines measure decode alone, and the wall rate is kept beside them |
| `Timeout` 1,280 s, fetch 600 s, `term_s` 30 | `P5.2-05`'s 3,600, 1,200, 120 | Under the 3 USD cap with one preemption restart in the bound (`modal_job.worst_case_usd`): 2.96 USD at 1,280 s, 3.04 at 1,300 |

## 4. How to run it

Modal, by the orchestrator (every call through `scripts/modal_ops.py`, after the brief's
approval): the commands of `modal_ops.md` section 4 with `P5.2-06`, and the
brief's launch steps.

CPU dry run (no network, no GPU): the same entry point on local stand-ins, with the fake server
and the smoke's six variants (a survivor, a refused flag, a hang, a refused edit, a second
survivor, a skipped pin run):

```bash
.venv/bin/python scripts/modal_ops.py --smoke --config configs/modal_P5.2-06.yaml
.venv/bin/pytest tests/test_fn_bisect.py tests/test_modal_probe_p5206.py --basetemp runs/P5.2m/pt
```

A plain GPU machine with version 11's inputs mounted as on Kaggle: `python scripts/fn_bisect.py
--config <resolved fn_config.json> --root / --session-start $(date +%s)`.

Time (estimate from `P5.2-05`): 140 s before the first variant; per variant the load 61 to 77 s,
then the warm-up (the `is_ep` JIT instance compiles at the first call, tens of seconds, N), a
graph capture where on, the requests (64 tokens in seconds, 512 in 10 to 50 s); a failing variant
ends in about 100 s. The end at 830 s holds V1 to V3.

## 5. Outputs

Under `/kaggle/working/run_logs/fn/` (on Modal `runs/P5.2-06/probe/attempts/1/kaggle_working/
run_logs/fn/`, pulled by `modal_ops.py logs`; all names already in `PROBE_PULL_PATTERN`):

| File | What |
|---|---|
| `bisect.json` | `rows`, one per variant: `name`, `why`, `flags`, `env`, `edits` (each `apply_patch` result), `argv`, `started_s`, `deadline_s`, `boot`, `ready_s`, `pool`, `request`, `rate` (`wall_tok_s`, `log.median_tok_s`), `alive_after`, `outcome`, `exit_code`, `stop`, `restored`, `first_error`, `marks`, `logs`, `seconds`; `running` (the variant booting, if the run stopped mid-way) |
| `bisect_<n>_<name>_tail.log` | The variant's last 300 log lines |
| `bisect_<n>_<name>_engine.log` | The variant's whole log, at most 2 MB (head eighth and tail kept) |
| `bisect_<n>_<name>_kapi_<i>.log` | V4 only: the kernel API log's bounded copy |
| `summary.md` | The stages and one line per variant |
| `run.json`, `state.json`, `inputs.json`, `install.json`, `patches.json`, `versions.txt`, `engine_commands.jsonl` | `fn_probe.py`'s outer records, as in `P5.2-05` |

Beside them, the stage's `step.json`, `env.json`, `host_memory.jsonl`, `notebook.log`.

## 6. How to read the results

- **The fix works:** V1 `survived`, its log with no fatal line. The edit then goes into version
  11's `patches` for every EP above 1 (N: TP 4 + EP 4 too), and `P5.2-01` gets a revision.
- **Which to ship:** among the survivors, V2 (FP8 KV, graphs) has about twice V1's pool and the
  graphs' decode rate; V3 shows whether the Triton MoE is a usable fallback (its rate against
  V2's). Rates are single readings on another GPU; only their order carries to Kaggle's L4.
- **V1 fails:** its `first_error` (`kernel_frame`, `triton_kernel`, `message`) and V4's (the
  traceback stops at the faulting launch; the kapi log's last line names the op) name the next
  kernel; N lists the next suspects (the QSA indexer, the PLE gather, the FA2 decode).
- **V2 fails where V1 survived:** the graph capture or the FP8 path, not the MoE; `marks` shows
  whether the capture began and ended.
- **V3 survives and V1 does not:** the fault is on the Marlin EP path but the edit does not cover
  it; ship EP 1 if its rate allows.
- **`not_run` rows:** the time left (`reason`), or `run_if`.

## 7. Failure modes

| Symptom | Cause | What to do |
|---|---|---|
| Every edited variant `edit_failed`, `sha256 differs` | The wheel's `marlin.py` differs from GitHub's at `d00d88e` | V3 still reads; the row's `edits[].sha256_before` is the installed file's hash; recompute the edit on the wheel's file |
| `edit_failed`, `match count` | The anchors moved | As above |
| V1 `crashed_in_request` with `get_parallel` in the message | The accessor is unset at forward time (N, open risks) | Use N's single-file alternative in `fused_marlin_moe.py` |
| `boot_timeout` on V2 | The graph capture hangs (as `P5.2-04`'s multimem) | `marks.graph_capture_begin` without `_end`; read the tail log |
| A later variant `exited_at_boot` with an out-of-memory or bind error | A rank of the earlier variant outlived its stop | `stop.killed`, `stop.port_free`, `stop.gpu_used_mib` of the row before |
| `bisect.json` with `running` set and fewer rows | The stage's SIGTERM or Modal's `Timeout` cut a variant | The finished rows stand; the running one's log is lost with the container |
| `first_error` null on a dead server | A fault text no pattern matches | Read `bisect_<n>_<name>_tail.log`; add the pattern |
| A pattern was wrong (`P5.2-06`'s `kv_tokens` read null: the fork's line carries the dtype before `#tokens`) | The log line's form | Fix the pattern, then read the pulled logs again locally: `python scripts/fn_bisect.py --config configs/P5.2-06.yaml --reread runs/<run>/probe/attempts/1/kaggle_working/run_logs/fn` writes `bisect_reread.json` |
| A variant starts and ends in 0.1 s as `boot_timeout` (`P5.2-06`'s V4) | Its boot window (the deadline less `request.min_s`) was below zero | Fixed: a variant starts only when that window is at least `min_variant_s` |
| The next variant meets GPU memory still held (`P5.2-06`: 41,881 MiB 0.2 s after V2's kill) | The ranks free their memory after their exit | Fixed: the stop also waits until no GPU holds more than `gpu_free_mib` above its reading before the launch |
| Every variant `boot_timeout` with `health.last_status` 503 | The server listens but its warm-up still compiles (`P5.2-06`'s V1: 44 s of 503) | A longer `budget_s`; `health.non_200_s` says how long it answered 503 |
