# ADR-023: Submission 4: shot v2 (bucket order, refill queue, no short caps)

- Status: Accepted, 2026-10-24 (the human submitted `arc26-submission` version 6, built by
  Amendment 1 and checked by `P6.9-03`; public 30.28). Proposed 2026-10-19 until a check read Pass. Amended 2026-10-24 (Amendment 1, task P6.9c:
  the weights' warm-up, and the waiver of the 12 h check for its version) and 2026-10-25
  (Amendment 2, task P6.9d: MTP and the 69,632 cap, Proposed until `P6.9-04` reads Pass and its
  public score is read), and 2026-10-26 (Amendment 3, run `P6.9-05`: the cap without MTP,
  Proposed until its public score is read). Drafted by the kaggle-engineer of task P6.9
  (`docs/tasks/P6.9.md`; design source `docs/research/beyond_h_options.md` on `task/ideas`,
  options 2a, 2b, 2c, 4b). The plan is the human's of 18 Oct: the re-targeted shot is the main
  track for the 30 Oct final run.
- Changes: [ADR-021](ADR-021-submission-3-shot.md) decision 2's `min_stage_s` (2,700 to 6,500 s),
  decisions 3 (selection) and 4 (waves), and the request and parser under them. Its decisions 1
  (NVARC unchanged), 2 otherwise (the shot only in the time NVARC leaves, the 45-minute
  allowance), 5 (the merge), 6 (the shot's end), and 7's pattern (CPU twin, then a 12 h check
  before any submission) stand.
- Related: ADR-018 (Submission 2's NVARC), ADR-020 (the agent solver and the hybrid build).

## Context

- **Submission 3's queue spent its time where the 27B solves least.** `P6.5-03` shot 30 DEV240
  tasks, all at NVARC confidence exactly 0 (least sure first, ties by id), and none of the 17 in
  $(0, 0.5)$. Pooled over `P6.3-05`, `P6.3-06`, `P6.5-03`: 3.5 of 53 shot tasks solved at 0
  ($g_0 = 0.066$) against 2.5 of 7 in $(0, 0.5)$; and NVARC's attempt 2 was never right in
  $(0, 0.5)$ on DEV120 (`P2.6-01`, `P1.1-01`: `reports/P2.8`), so the merge risks nothing there.
- **Short caps never solved:** 0 of 15 at 26,469 (`P6.5-03`'s second wave), 0 of 29 at 28,805 per
  task (`P6.3-04` arm D).
- **Waves idle their streams:** 4 of 15 turns ended early in `P6.5-03`'s first wave, 7 of 15 in
  `P6.3-05`; each stream waited for the wave's slowest task.
- **`P6.9-01`** (`docs/experiments/P6.9-01.md`, record `cedc610`; 15 DEV120 tasks, 4xL4, the new
  request): no Fault; harness 3 of 5; the shot alone 6.0 of 15, merged 8.5 against NVARC's 7.0
  (noise). Its rows: **4' Keep**, the middle bucket $[0.5, 0.7)$ stays out ($1.0 - 6 \cdot 5.5/46 =
  0.283 < 1$); **5' Revert**, the in-request forcing answered 6 of 7 outputs of its 4 budget-ended
  turns ($\rho = 0.857$) against the separate turn's 13/15 = 0.867. The lenient parser read no
  answer the strict one refused and changed no valid one. Pooled $(0, 0.5)$: 3.0 of 11, posterior
  $g_1 = 0.19$ (Beta(5, 21), 90%: 0.08 to 0.33). Its 5 solves ended their turns at 24,137 to
  68,021 tokens.

## Decision

1. **Selection by bucket** (`selection.order: buckets`, `mid_bucket: false`, `max_confidence`
   0.5): NVARC's confidence in $(0, 0.5)$ first, then exactly 0, each least sure first, then by
   id (`arc26.agent.shot.select_tasks`). Replaces ADR-021 decision 3's ascending order.
2. **A refill queue instead of waves** (`waves.refill: true`): up to 15 streams; when one frees,
   the next queued task starts with the largest cap $C \le 94{,}208$ whose planned one-task wave
   at the streams then running, $T = C / (0.85 \cdot 19.0) + n (o + C) / 958.8 + 6{,}144 / 19.0 +
   300$ s, ends by the generation stop (`plan_start`): ADR-021 decision 4's model, applied per
   start. The KV need must fit what the running tasks leave.
3. **No cap under 65,536** (`waves.min_cap`; ADR-021: 16,384). The stage's `min_stage_s` grows
   to 6,500 s (the smallest start of 15 streams at 65,536 plans 5,812.5 s), so its latest start is
   33,430 s instead of 37,230 s; no start comes after about 34,117 s.
4. **The separate forcing turn kept** (`request.forcing_in_request: false`, row 5'): the request
   and the server's flags are Submission 3's.
5. **The lenient answer parser** (`request.lenient_answers: true`): a block the strict parser
   refuses is read again (digit-string rows, string cells, unbalanced digit-string rows), and
   $S_1$ is a test output's first answer of the shape the demonstrations imply.
6. **Runs before any submission:** the CPU twin `P6.9-02-cpu`, then `P6.9-02` (12 h on DEV240 at
   the competition's times, `alyahya737/arc26-submission`, code dataset
   `alyahya737/arc26-code-p69`); its version is Submission 4 only on its brief's Pass row. No DEV4
   smoke: `P6.9-01` ran the request and engine on 4xL4, the notebook is `P6.5-03`'s.

## Alternatives considered

- **`max_confidence` 0.7 with the middle bucket** (option 2c). Lost on `P6.9-01`'s row 4': the
  net in $[0.5, 0.7)$ does not cover P2.8's attempt-2 loss there.
- **The answer inside the thinking request** (vLLM's `thinking_token_budget`, option 2b (iii)).
  Lost on row 5' by one output; the separate turn's prefill costs about 0.1 attempts per shot (the
  record's Monte Carlo).
- **A looser refill start rule** (the planned rate at 1.0, or a forcing term for the turns that can
  be cut together). Not taken: about 10 more attempts by the record's Monte Carlo, but a code
  change with its own GPU check, which the quota before 2 Nov does not hold beside `P6.9-02`.
- **Waves at `min_cap` 65,536.** Lost: a second wave would rarely fit; the refill uses the streams
  freed by early turns.

## Consequences

- The version keeps NVARC's attempts at every moment, as Submission 3.
- At `P6.5-03`'s times: 15 starts at the full cap in $(0, 0.5)$ and about 1 to 6 refills at
  about 66k to 70k (16 to 21 starts, against Submission 3's 30, of which 15 at a cap that never
  solved); the shot ends near 36,800 to 39,000 s.
- The shot runs only if NVARC ends by about 33,340 s ($u \approx 0.804$ on 240 tasks; Submission
  3: 37,140 s): a hidden set about 11% slower per puzzle than DEV240 runs no shot (Submission 3
  tolerated about 24%, but only short waves would have run there).
- Expected gain against Submission 3 on the hidden set: about $+2.3$ tasks, $+1.0$ point ($+0.2$
  to $+2.1$ over $g_1$'s interval; `docs/prerun/P6.9-02.md`, section 9): under the 2-point noise
  floor.
- The notebook's prose still describes Submission 3's queue (least sure first, in waves); the
  embedded run config governs.

## Overturn criterion

- `P6.9-02` reads Fault or Timing fail: no submission of this version; Submission 3 stays the
  shot build; the cause is fixed and measured again (only on the quota of the Sat 31 Oct 03:00
  reset).
- Submission 4's public score falls 2 points or more below Submission 3's: Submission 3's build
  takes the final picks.

## Amendment 1 (2026-10-24, task P6.9c): the weights' warm-up and the waiver of the 12 h check

- **Context.** `P6.9-02` read Fault at the boot (`docs/experiments/P6.9-02.md`): Kaggle now serves
  the weights' notebook-output input through FUSE.GCSFUSE, vLLM 0.31.0 prefetches only on NFS and
  Lustre (`vllm/model_executor/model_loader/weight_utils.py`, `safetensors_weights_iterator`, at
  `db9527a4`), and its memory-mapped load took 802.41 s against 45.42 s on NFS on 15 Oct, past the
  900 s boot limit. NVARC's 30,050.5 s stage, the final check, and the notebook's end were as
  planned.
- **Decision 8: the warm-up.** Submission 4's build adds `agent.warmup`
  (`configs/submission_P6.9-03.yaml`; runbook `docs/runbooks/build_submission.md`, "Weights
  warm-up"): at the agent check, before NVARC, a background process at the lowest CPU and
  best-effort IO priority copies the 12 weight files to local scratch in parallel pieces and
  checks each at its size and sha256 against the output's manifest; the agent stage gives the
  engine the verified copy, waits up to 900 s for a copy still making progress, and otherwise
  loads from the input with `--safetensors-load-strategy prefetch` (`vllm/engine/arg_utils.py`
  line 1038). Why: the copy costs nothing during NVARC's 8.3 h, reads with 8 sequential readers
  where vLLM's prefetch has one per file, and leaves the boot limit, the shot's times, and the
  wheel (`a5bb333f`) unchanged; the alternatives (prefetch alone, `eager`, a longer boot limit, a
  copy after NVARC, an `arc26` module) and why each lost are in the runbook's "Why it is built
  this way". Measured first in the CPU twin `P6.9-03-cpu`: the copy 88.6 s at 219.8 MB/s, the
  hash 123.5 s, the shot given the local copy with no wait.
- **Decision 9: the waiver (the human's choice A, 24 Oct).** This version is submitted on the
  Pass of a short run of its own (`P6.9-03`, 4 DEV120 tasks, `docs/prerun/P6.9-03.md`) instead of
  decision 7's 12 h check: about 16.9 quota h were left that week, under a full check's about 22.
  The timing argument (the brief, "The waiver"): NVARC and its deadlines are `P6.9-02`'s, measured
  at 12 h; the copy ends hours before NVARC; the folder's choice comes before the shot's timeout
  is computed, so the shot's SIGTERM (40,050 s), SIGKILL (40,170 s), and the notebook's end
  (40,500 s) stand. Accepted with it: the refill path of the queue first runs on a GPU in the
  scored rerun (3 shot tasks on 15 streams in `P6.9-03` start at once), with the file valid and
  NVARC's attempts kept at any failure there.
- **Overturn.** `P6.9-03` reads Fault: no submission of its version; Submission 3 stays the shot
  build. A scored rerun of the submitted version whose shot fails is visible only in its score
  against Submission 3's (the overturn criterion above).

## Amendment 2 (2026-10-25, task P6.9d): MTP with one speculative token and the 69,632 cap

- **Context.** Submission 4 (`arc26-submission` version 6, Code-commit `97e4096`) scored 30.28
  public. Its shot reaches 16 to 21 of NVARC's least sure tasks in the time NVARC leaves
  (Consequences above). `P6.10-01` (Modal, 4 x L4, 15 streams; `docs/experiments/P6.10-01.md` on
  `task/P6.10`) measured MTP with $k = 1$ at 1.21 times the base's per-call rate (the
  pre-registered fallback) and 1.34 times over the pure thinking window, acceptance
  $\alpha = 0.87$, its KV pool 13.7% under the base's. Its Kaggle repeat `P6.10-02` failed on the
  GCSFuse weight load most likely, not on MTP (its record, "Cause found", calls the cause most
  likely: its engine log was not saved); Amendment 1's warm-up fixed that
  load (`P6.9-03`: 1.67 s from the local copy, boot 278.1 s). Of the 7 solves in `P6.5-03`'s and
  `P6.9-01`'s 30 long attempts, 6 ended their thinking under 70k tokens (24,137 to 68,021).
- **Decision 10.** A version of Submission 4's build with `configs/agent_shot_P6.9d.yaml`: P6.9's
  run config with `server.extra_args` `["--speculative-config", '{"method": "mtp",
  "num_speculative_tokens": 1}']` (task/P6.10's, verbatim) and `waves.max_cap` 69,632
  ($17 \times 4{,}096$); `min_cap` stays 65,536, so the shot's minimum stage (6,262.5 s), the
  notebook's `min_stage_s` (6,500 s), the latest start (33,430 s), the generation stop (39,930 s),
  and the notebook's end are unchanged. 15 streams fit MTP's Kaggle pool estimate,
  $15\,(6{,}771 + 69{,}632 + 6{,}144) = 1{,}238{,}205 \le 1{,}467{,}099$ (1,393,779 with MTP's
  measured per-request blocks); at 94,208 only 13 would. The plan's rate stays 19.0 tokens per
  second (MTP at long context on Kaggle is unmeasured), so every start stays inside its planned
  time and MTP's gain shows as turns that end sooner and more refills. Short run `P6.9-04`
  (`docs/prerun/P6.9-04.md`, `P6.9-03`'s 4 tasks), then a leaderboard test by the human.
- **Overturn.** `P6.9-04` reads Fault: not submitted; version 6 stays the shot pick. The submitted
  version's public score more than 2 points below version 6's 30.28, or its run unhealthy:
  version 6 stays the shot pick.

## Amendment 3 (2026-10-26, run P6.9-05): the 69,632 cap without MTP

- **Context.** `P6.9-04` (Amendment 2's build) read Pass by its rows, but its engine log shows
  MTP slower than the base at long context on Kaggle's 4 x L4: one stream near 65k to 70k tokens
  of context decoded 21.1 to 22.7 tokens per second (`runs/P6.9-04/run_logs/agent/engine.log`),
  against 38.9 to 41.3 for one stream from about 48k to 82k without MTP in `P6.9-03`
  (`runs/P6.9-03/run_logs/agent/engine.log`), with acceptance 0.824 (95,133 of 115,444 drafts),
  so a step cost about $83$ ms against the base's $25$ ms. `P6.10-01`'s gain was measured at
  contexts up to about 18k.
- **Decision 11.** A version of Submission 4's build with `configs/agent_shot_P6.9e.yaml`:
  `configs/agent_shot_P6.9.yaml` with `waves.max_cap` 69,632 only (no speculative flag;
  `tests/test_p6_9e.py`). The engine and its flags are version 6's (`P6.9-03`), the cap ran on GPU
  in `P6.9-04`, so only the CPU twin `P6.9-05-cpu` checks it before the push; the human then
  submits the pushed version (`docs/prerun/P6.9-05.md`).
- **Overturn.** Its scored rerun fails or its public score is more than 2 points below version
  6's 30.28 (under 28.28): version 6 stays the shot pick.

## Amendment 4 (2026-10-26, task P4.7b): the rank-256 tail

- **Context.** Version 8's shot (Amendment 3) starts about 18 of the 55 tasks it queues at 240
  tasks: all 17 in $(0, 0.5)$, about 1 of the 38 at confidence 0, then the machine is idle for
  about 3,000 s before the generation stop (`docs/research/shot_tail_screen.md` on `task/ideas`,
  `6e54ba4`). A rank-256 NVARC pass (the reference's LoRA settings) on the tasks rank 16 leaves
  at confidence 0 adds $+1.80$ tasks of 120 in the one pair of runs measured (`P1.1-01` against
  five rank-16 passes; `docs/research/second_pass_screen.md`, `e79a1a0`), 2.4 tasks per 4-GPU
  wall hour against the shot's 0.53 at confidence 0. The human's call (26 Oct): no standalone
  test run; each idea becomes a submission whose own run is a short check.
- **Decision 12.** A version of version 8's build with two changes. The shot's run config
  `configs/agent_shot_P4.7b.yaml` is `configs/agent_shot_P6.9e.yaml` with
  `selection.zero_bucket: false` (no task at confidence 0 in its queue). The builder's
  `agent.tail` adds the tail cell after the agent stage: NVARC's solver again (`python -m
  arc26.tail`, the split solver's workers in its challenges file's order) with `RunConfig`'s
  defaults on NVARC's confidence-0 tasks the shot did not answer, cheapest first by NVARC's
  `puzzle` seconds (recorded in scratch in a rerun), its soft deadline
  $E - w - g = 40{,}050 - 300 - 300 = 39{,}450$ s and hard stop 39,750 s, skipped with less than
  600 s left; its attempts fill only outputs whose attempt 1 is a placeholder in the file and in
  NVARC's copy. The agent stage skips the install and the engine when the shot's queue is empty.
  The shot already returns, and stops its engine, once nothing runs and nothing can start
  (`arc26/agent/shot.py`, `run_refill` and `run`), so it needed no change for that. The
  notebook's end stays by 40,500 s. Own run `P4.7-02` (`docs/prerun/P4.7-02.md`) on 3 DEV120
  tasks, then a leaderboard test by the human.
- **Overturn.** `P4.7-02` reads Fault: not submitted, version 8 stays the shot pick. The
  submitted version's scored run fails, or its public score is more than 2 points below version
  8's: version 8 stays the pick (the human's call, as the brief states).

## Amendment 5 (2026-10-28, task P5.1): LLM first, the shot on every task without NVARC

- **Context.** The human's call of 28 Oct, after the competition's discussion: leading teams run
  Qwen3.8 on every test task with long thinking, and their binding constraint is throughput (about
  365 tokens per second covers 240 tasks at about 64k tokens in 12 h). Version 9's shot gets
  about 2.6 h after NVARC's 8.35 h and starts about 18 tasks. The 27B solved 6 of 15 on
  `P6.9-01`'s mixed set; its rate on a random sample is not measured.
- **Decision 13.** Version 10 (`configs/submission_P5.1-01.yaml`): version 9's build without the
  tail and with the builder's `agent.skip_nvarc`: when the agent check passes, in both modes, the
  Run cell starts no NVARC and the Write cell writes `[[0]]` placeholders; a failed check runs
  NVARC alone. The shot's run config `configs/agent_shot_P5.1.yaml` is version 8's with
  `selection.order: opening` (every task, the smallest opening bound first, no confidence), 22
  streams at `waves.kv_admission` 1.05 ($\lfloor 1.05 \cdot 1{,}699{,}566 / 78{,}451 \rfloor$),
  $C = 65{,}536$, the planner's rate 25, run `P6.9-01`'s in-request forcing (its note and
  `--reasoning-config`), `request.boot_fallback` (a failed first boot boots again with version
  9's flags and request), and `waves.dev_cap` 8,192 for the version's own short check. The
  window runs from about 560 s to the generation stop at 39,930 s, as version 9's. Estimated 179
  (173 to 187) of 240 tasks finished (`docs/research/public_qwen38_notebooks.md` on
  `task/ideas`, section 3). Own run `P5.1-01` (`docs/prerun/P5.1-01.md`) on 2 DEV120 tasks, then
  a leaderboard test by the human. `P6.9-01`'s row 5' Revert of the in-request forcing is
  overturned for this build: at 22 streams the separate turn's prefill stalls every stream
  (147 against 174 finished at 20).
- **Overturn.** `P5.1-01` reads Fault: not submitted. The submitted version's scored run fails, or
  its public score is more than 2 points below version 9's or version 8's: the earlier version
  stays the pick (the human's call).

## Amendment 6 (2026-10-30, task P5.4): Flash-Next on version 10's vLLM 0.31.0

- **Context.** The human wants Qwen3.8-Flash-Next submitted on 30 Oct. The SGLang lineage (task
  P5.2, version 11) fails on sm89 with an illegal memory access (`P5.2-05`), traced to the fork's
  GPTQ Marlin MoE call dropping `expert_map` under expert parallelism
  (`docs/research/fn_sm89_illegal_access.md` on `task/ideas`, `b2a5a9e`). vLLM 0.31.0 has the
  model in tree and passes `expert_map` to its Marlin experts; version 10 ran its wheelhouse on
  4 x L4 (`P5.1-01` Pass, scored rerun complete).
- **Decision 14.** Version 11b (`configs/submission_P5.4-01.yaml`): version 10's build with the
  weights as the Kaggle Model `dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/Transformers/default/1`
  (the builder's `agent.model` with `kaggle_model` and a pinned `files` list; the warm-up copies
  all 124.8 GB to local disk) and the run config `configs/agent_shot_P5.4.yaml`: TP 4 with
  `--enable-expert-parallel` (Marlin MoE), FP8 KV, a 69,632-token context, prefix caching off,
  `gpu_memory_utilization` 0.95, the `qwen4_exp` imports, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 128,
  and a cap of $C = 45{,}056$ with the streams the boot's pool holds at admission 1.0
  ($\lfloor P / 57{,}971 \rfloor$, 4 to 5 at the estimated 0.26M to 0.32M). Own run `P5.4-01`
  (`docs/prerun/P5.4-01.md`) on 3 DEV tasks at the 8,192 cap, then a leaderboard test by the
  human. The research note's no-go (`flash_next_shot.md`, 28 Oct) was for a version with NVARC
  first and the tail; this build is LLM first, which the note did not price.
- **Overturn.** `P5.4-01` reads Fault: not submitted. The submitted version's scored run fails, or
  its public score is more than 2 points below version 10's 13: version 10 stays the pick (the
  human's call).
