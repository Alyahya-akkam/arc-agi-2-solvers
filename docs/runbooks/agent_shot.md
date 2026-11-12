# Runbook: `python -m arc26.agent.shot`

Task P6.5b, Part A (`docs/tasks/P6.5b.md`; ADR-020; BRIEF A8). Config: `configs/agent_shot.yaml`.
Module doc: `docs/modules/agent_shot.md`. The hybrid notebook that runs it after NVARC is built by
`scripts/build_kaggle_notebook.py` (Part B, `docs/runbooks/build_submission.md`); its pre-run
briefs are `docs/prerun/P6.5-02.md` (the DEV4 smoke) and `docs/prerun/P6.5-03.md` (the 12 h timing
check). Measurements cited as "P6.3-05" are that run's pulled logs (`install.json`,
`engine_prep.json`, `boot.json`, `engine_calls.jsonl`, `shot_metrics.json`) and
`docs/prerun/P6.3-05.md`. Task P6.9 (`docs/tasks/P6.9.md`) adds four options behind optional
keys; with none of them set (`configs/agent_shot.yaml`) the module runs as Submission 3 did. The
Submission 4 candidate sets them: `configs/agent_shot_P6.9.yaml`; their measurement wave is
`P6.9-01` (`docs/prerun/P6.9-01.md`, `docs/runbooks/agent_spike.md` section 2.7), whose rows kept
the middle bucket off and turned the in-request forcing off (`docs/experiments/P6.9-01.md`).

## 1. Why this script exists

Run `P6.3-05` solved 4 of the 15 DEV40 tasks NVARC is least sure of (NVARC: 0) with one long xhigh
thinking turn per task and direct answers. Submission 2's NVARC ends a 240-task rerun at about
74% of its soft deadline (`P2.6-01`: 499.5 s per puzzle on four workers), so about 3 h of the 12
are idle. This module spends that time on the shot, on the tasks where NVARC's attempts are least
likely right, and never makes NVARC's file worse: it is the human's approved plan for Submission 3
(14 Oct, 12:50). It feeds the Submission 3 decision through `P6.5-03`'s pre-registered rows
(Fault, Timing fail, Pass).

Task P6.9 re-targets and tightens it (`docs/research/beyond_h_options.md` on `task/ideas`,
options 2a, 2b, 4b): Submission 3's queue spent both DEV240 waves at confidence 0, where the 27B
solved 3.5 of 53 shot tasks, while in $(0, 0.5)$ it solved 2.5 of 7 and NVARC's attempt 2 was
never right; caps under 64k never solved; and every forced reply prefilled about 100k tokens
again. The options feed Submission 4 through `P6.9-01`'s reading and the 12 h check `P6.9-02`.

## 2. What it does, step by step

1. **Arguments and config.** `arc26.agent.run`'s parser and checks, this module's config loader.
   A refusal exits 2 and writes nothing.
2. **NVARC's file first.** Read the tasks, NVARC's submission (the log folder's copy, made once),
   and NVARC's `candidates.json`; write `submission.json` from NVARC's attempts. The file is valid
   from here on.
3. **Selection.** Per test output the softmax at $T = 1$ of the rank 1 grid's `score_kgmon` score
   over the output's distinct grids,
   $\hat p = 1 \big/ \sum_g e^{(s(g) - s_1)/T}$ with
   $s(g) = |C_g| - \frac{1}{|C_g|}\sum_{c \in C_g} \bar s(c)$, exactly `softmax_p` of
   `scripts/calibration_report.py`; an output without candidates is 0 where NVARC's attempt is
   its placeholder, else unknown; per task the minimum over its known outputs.
   Tasks under `max_confidence` (0.5), ascending, ties by id.
4. **Engine.** Only when the queue is not empty and `setup_s` (450 s) plus the smallest wave
   (2,000.1 s for 15 tasks with P6.3-05's longest opening) fit before the generation stop. With `--engine-venv`: the engine environment
   (`clean_env`, the config's variables, the venv's `bin` first on `PATH`), P6.3-05's preparation
   (drop the orphan FlashInfer jit-cache provider, an empty `sitecustomize` first on
   `PYTHONPATH`, import the workers' 16 modules), the GPU preflight, then `vllm serve` with
   P6.3-05's flags; one boot, no restart. The KV pool is read from the engine log (`KV cache size:
   N tokens`), else `/metrics`, else the config (1,699,566).
5. **Sanity.** One low-effort request ("What is 17 + 25?"), waiting at most
   $\min(600, \text{stop} - \text{now})$ s; an error, a timeout, an empty or a degenerate reply
   ends the shot.
6. **Waves.** With $L$ the seconds left before the generation stop, $n$ the wave's tasks and
   $o$ their longest opening bound, the cap is the largest $C \le 94{,}208$ with

   $$
   T(C, n, o) = \frac{C}{\phi r} + \frac{n\,(o + C)}{p} + \frac{F}{r} + m \le L,
   $$

   $\phi = 0.85$, $r = 19.0$ tokens per second per stream, $p = 958.8$ prompt tokens per second
   (P6.3-05's 8 forced replies: 783,425 tokens in 817.1 s), $F = 6{,}144$, $m = 300$ s: the
   turn, then the forced replies with every turn of the wave cut. Below $C = 16{,}384$ no wave
   starts. The wave takes the next 15 tasks and fewer, the cap planned again, while
   $\sum_i (o_i + C + F)$ exceeds the pool ($o_i$ the opening's character bound). A full wave of
   15 is planned at 8,036.5 s. Every task of the wave runs at once with P6.3-05's request but $C$: one
   thinking turn at xhigh, the two-answer direct prompt, an answer-forcing turn after a cut turn.
   Each finished task is logged and merged; the file is rewritten atomically. Waves repeat while
   tasks and time remain.
7. **Merge.** Per test output: no shot answer keeps $(N_1, N_2)$; a real $N_1$ gets the shot's
   first answer as attempt 2 when it differs, or its second when the first equals $N_1$ and
   $N_2$ is a placeholder; NVARC's placeholder gives way to the shot's two answers, or its first
   and $N_2$ (full table: `docs/modules/agent_shot.md`).
8. **Task P6.9's options** (each off when its key is absent):
   - `selection.order: buckets`: the queue is $(0, 0.5)$ first, then, under
     `selection.mid_bucket`, $[0.5, 0.7)$, then $c = 0$ (option 2c's pre-registered order);
     least sure first within a bucket, ties by id; still only tasks under `max_confidence` (0.7
     at least for the middle bucket).
   - `waves.refill`: up to `wave_tasks` streams; when one ends, the next queued task starts at
     once (`plan_start`) with the largest cap whose plan $T(C, n, o)$ ends before the stop, $n$
     the streams running with it, so the forced replies are still planned as all cut at once; no
     task starts under `min_cap` (65,536 in the candidate) or without room in the KV pool left by
     the running tasks (it waits for one to end). Each start is a one-task wave in
     `shot_plan.json`. The stop rules read the last `wave_tasks` finished tasks: more than
     `wave_share` of them degenerate, or all engine errors.
   - `request.forcing_in_request`: the thinking request carries vLLM's `thinking_token_budget`
     $= C$ and `max_tokens` $= C + F$; the server forces `thinking_end_note` and `</think>` at the
     cap and the answer follows in the same request. The plan drops the re-prefill:
     $T(C, n, o) = C/(\phi r) + F/r + m$ (a full wave 6,456.7 s against 8,036.5 s). A reply still
     cut without a readable answer gets the separate forcing turn, so `attempt_tokens` is
     $C + 2F$. The server's `--reasoning-config` must name the same note (the config check).
     After the sanity request one budget probe (`budget_probe`: the sanity question, budget 8
     tokens) checks the server: on an error (vLLM's HTTP 400 without a reasoning config) or no
     forced note the option is off for the whole run (`without_in_request`: Submission 3's
     request and plan), recorded as `budget_probe` in `shot_plan.json`, `shot.in_request` false
     and a problem line in `run.json`. A request refused for the field later is sent once more
     without it (the loop), and a reply cut without a readable answer is forced with its
     reasoning left out.
   - `request.lenient_answers`: the loop's lenient parser (rows as digit strings, one-digit string
     cells, digit-string rows with unbalanced brackets) for blocks the strict parser refuses, and
     per test output $S_1$ is the first valid answer of the shape the demonstrations imply
     (`implied_shapes`; the model's own best-first order, drafts of other sizes skipped), $S_2$
     the next one after it that differs (else the first before it that differs); with no
     implied shape or no answer of it, Submission 3's first two.
9. **End.** At the generation stop (`--deadline` minus `final_margin_s`, 120 s) every running call
   is cancelled and its task keeps NVARC's attempts; the engine is stopped; the file is written
   and checked; development mode scores the hybrid, NVARC alone, and the shot.

## 3. Why it is built this way

- **NVARC first, the shot only in the time left; no share cut.** Considered: give the shot a
  fixed share of the session. Lost: NVARC's marginal task is worth far more than the shot's
  (P6.3-05: 4 solves in 15 of the hardest tasks for 1.6 h of 4x L4; NVARC scored 10.5 of 40 on
  DEV40 in `P2.6-01`), so the human's decision (14 Oct) keeps NVARC's deadlines.
- **P6.3-05's rule from `candidates.json`, not the scheduler's.** Considered: the scheduler's
  `nvarc_confidence` (ADR-014's agreement, mean over outputs). Lost: P6.3-05 selected with the
  softmax minimum (`configs/P6.3-05.yaml`, `direct.task_ids`), P2.8's promotion signal (AUROC up
  to 0.89 on DEV40); a test proves the rule reproduces `softmax_p` on all 142 DEV120 rank 1 rows
  and P6.3-05's 15 tasks. Considered: read the CSV. Lost: a scored rerun has only the notebook's
  own candidates.
- **The forced replies are planned with every turn cut, by prefill rate.** Considered: the
  slowest measured forced reply scaled by $C$ ($815\,C/94{,}208$), relying on the plan factor's
  slack for 15 cut turns instead of P6.3-05's 8. Lost (the coordinator, 14 Oct, after review):
  that spent the slack meant for a slower decode, so a time-bound wave of 15 all cut tolerated
  only about 8% under 19.0 tokens per second, while P6.3-05's busy streams already ran at 16.5
  to 17.6. Now the prefill of $n$ cut prompts, $n(o + C)/p$, and the forced reply's generation,
  $F/r$, are planned on their own, and the plan factor covers the decode: a time-bound wave of 15
  at the full cap ends down to 15.54 tokens per second (18% under), at 16,384 down to 13.75. The
  price is a later start of the last wave: a full wave is planned at 8,036.5 s against 5,784 s
  measured in P6.3-05 (7 of its 15 turns ended early).
- **A KV-bound wave takes fewer tasks, not a smaller cap.** Considered: lower $C$ until all 15
  fit. Lost: two of P6.3-05's four solves came from turns cut at the full 94,208 (80a900e0,
  dd6b8c4b), and the dropped tasks run first in the next wave if time allows (brief, point 4).
- **The opening's bound is one token per character.** Considered: count each opening with
  `/tokenize` before the wave. Lost: extra requests the spike did not send, and the pool does not
  bind: P6.3-05's 15 openings bound at 57,627 tokens and the full wave needs 1,562,907 of
  1,699,566 (92%).
- **No engine restart.** Considered: `arc26.agent.run`'s one restart. Lost: a death cuts every
  call of the lockstep wave, and a new boot costs about 300 s (P6.3-05: 300.1 s) of a window of
  about 3 h; the brief ends the shot with NVARC's file (point 7).
- **Preparation and boot in the main thread.** Considered: `arc26.agent.run`'s boot in a worker
  thread. Lost: a SIGTERM during a threaded boot waits for the thread (up to `boot_timeout_s`)
  inside asyncio's shutdown; in the main thread `SystemExit` reaches the final write at once, and
  `subprocess.run` kills a running preparation step.
- **A call cut at the stop keeps NVARC.** Considered: merge the answers a cut turn may already
  show. Lost: the brief (point 6); a cut forcing reply may hold one of two outputs, and NVARC's
  pair is the measured fallback.
- **The moved code is the spike's, verbatim.** `prepare_engine`, the preflight, the boot log
  parsers, the reply health checks, and `NoRepl`/`NoVerifier` are moved, not rewritten; a test
  compares each source with the script's. `RecordingEngine` is the class the spike's
  `make_engine` defined inside itself, lifted out with its fields as arguments; a test compares
  the client config it builds with the spike's. The requests of a full wave are proven byte-identical
  to the spike's `direct_phase` with `configs/P6.3-05.yaml`.
- **Bucket order, not a lower `max_confidence` (P6.9, 2a).** Considered: keep least sure first
  and lower the ceiling. Lost: confidence 0 and $(0, 0.5)$ are different populations (no
  candidate at all against a split vote), and the 27B's measured rate is higher in the second
  (2.5 of 7 against 3.5 of 53); a ceiling cannot put the middle first. Considered: the middle
  bucket on now. Lost: unmeasured for the 27B and NVARC's attempt 2 was right on 3 of 37 of its
  outputs (`P2.6-01`), so `P6.9-01` decides it.
- **A refill queue, not waves (P6.9, 2b (ii)).** Considered: waves with a later start for short
  tasks. Lost: 4 of 15 turns ended early in `P6.5-03`'s first wave and 7 of 15 in `P6.3-05`,
  each stream idle until the wave's slowest task; the refill keeps the time model and the stop
  unchanged in meaning and only starts work sooner. The plan for each start keeps $n$ the
  streams running, so the separate forcing's prefill stays planned as if every turn were cut.
- **`min_cap` 65,536 (P6.9, 2b (i)).** Considered: 16,384 (Submission 3). Lost: short caps never
  solved: 0 of 15 at 26,469 (`P6.5-03`), 0 of 29 at 28,805 per task (`P6.3-04` arm D), 1.0 of
  36 at 32k against 3.5 to 6.0 at 64k for a third party.
- **In-request forcing (P6.9, 2b (iii)).** Considered: the separate forcing turn (Submission 3);
  vLLM's `continue_final_message`. Lost: both prefill the conversation again (14% to 23% of a
  shot); vLLM 0.31.0's `thinking_token_budget` forces the end in the running request
  (`docs/runbooks/agent_spike.md`, section 3, gives the sources). Kept behind its key until
  `P6.9-01` shows the in-request answers come at least as often as the separate turn's. They did
  not ($\rho = 6/7 = 0.857$ against $13/15 = 0.867$, row 5' Revert): Submission 4 runs the separate
  turn.
- **The first answer of the implied shape as $S_1$ (P6.9, 4b).** Considered: Submission 3's first
  answer whatever its shape; the last answer of the implied shape (fix round 1 dropped it). Lost:
  a draft of another size written before the answer of the implied size would take the merge's
  single shot slot under the first; the last breaks the model's own order, which the prompt asks
  for best first: on `abc82100` (`P6.3-06`, two 20 x 20 answers per output) it put the second
  answer in that slot. The first of the implied shape keeps the order and skips such drafts; on
  every earlier transcript it changes $S_1$ on no output.
- **No `--smoke` flag.** The module must take exactly `arc26.agent.run`'s arguments (Part B's
  contract); its CPU path is the same `--fake-engine` (P6.4's fake OpenAI server, started as vLLM
  would be). The engine is a separate server, so there is no model to shrink: the fake engine
  replays scripted replies per task.

## 4. How to run it

On Kaggle it runs only inside the hybrid notebook (Part B), after NVARC's stage, with 4x L4
(`NvidiaL4`), the engine wheelhouse and the INT4 weights attached; the notebook installs the venv
and passes:

```bash
python -m arc26.agent.shot --mode competition --tasks <challenges.json> \
    --out /kaggle/working/submission.json --config <agent_shot.json> \
    --engine-venv /tmp/arc26-agent/venv --model-dir <weights> \
    --nvarc-submission /tmp/arc26-agent/nvarc_submission.json \
    --nvarc-candidates /tmp/arc26-agent/nvarc_candidates.json \
    --deadline <epoch seconds> --log-dir /tmp/arc26-agent/logs
```

On a plain machine with 4 GPUs of 24 GB and the venv installed (`install_wheelhouse.py`), the
same command with `--mode dev --solutions <solutions.json>`. The CPU smoke:

```bash
.venv/bin/python -m arc26.agent.shot --mode dev --tasks challenges.json --out submission.json \
    --config configs/agent_shot.yaml --fake-engine replies.json --nvarc-submission nvarc.json \
    --nvarc-candidates candidates.json --budget-s 60 --solutions solutions.json
.venv/bin/pytest tests/agent/test_shot.py
```

The Submission 4 candidate runs the same command with `--config configs/agent_shot_P6.9.yaml`
(its build and 12 h check are `P6.9-02`); `tests/agent/test_p6_9_shot.py` runs a refill run with
that config's separate forcing turn end to end on the fake engine, and one with in-request forcing,
which the fake engine honors as vLLM does (`arc26.agent.server.fake_completion`).

Expected: preparation about 40 s, boot about 300 s, then waves of up to $T$ each (a full wave of
15 planned at 8,036.5 s, 5,784 s measured). GPU memory: vLLM takes 0.90 of each L4 (weights 4.3 GiB
per GPU in P6.3-05's `boot.json`, the rest KV); host disk about 10 GB in `/tmp` for the venv and caches.

## 5. Outputs

In `--log-dir` (the notebook copies the small files to `run_logs/agent/` in development runs):

| File | Content |
|---|---|
| `run.json` | mode, deadline, problems, engine (prep, boot seconds, command, exit), the shot's counts, submission validity and slots, config sha256; in development mode `score` and `shot_metrics` |
| `shot_plan.json` | the queue with confidences, the wave model and `stage_arithmetic`, the KV pool and its source, the sanity request, per wave: tasks, $C$, planned turn, forcing and length, planned and actual end, used, cut, degenerate, engine errors, unfinished tasks; why the shot stopped |
| `shot_attempts.jsonl` | one row per finished task: P6.2's attempt record (answers, tokens, turns, stop reason, forcing turns) with wave, cap, finish reasons, errors, degenerate, used |
| `engine_calls.jsonl` | one row per chat call: task, attempt, thinking switch, reasoning location, run length, empty and degenerate flags, tokens, finish, seconds, error |
| `prompt_counts.jsonl` | each opening's `/tokenize` count |
| `timing-engine.jsonl` | `{puzzle_id, op, seconds, tokens}` per engine call |
| `engine.log`, `boot.json` | the server's log; the boot's facts (KV pool, concurrency, load times, NCCL) |
| `engine_prep.json`, `engine_prep.log`, `preflight.json`, `preflight.log` | the preparation as in P6.3-05 |
| `nvarc_submission.json` | NVARC's file as it was at the first start |
| `shot_transcripts.jsonl` | development mode only: every conversation with its reasoning |
| `shot_metrics.json` | development mode only: hybrid, NVARC alone, the shot alone, solves, tasks gained and lost by the merge, per wave |
| `summary.md` | a short summary |

Under task P6.9's options: `shot_plan.json` adds `order` and `mid_bucket`, and with
`waves.refill` its `waves` holds one row per started task; `shot_attempts.jsonl` rows add
`budget_forced` (turns the server ended at the budget) and `answer_lenient` (per answer, read only
by the lenient parser) when not 0 or empty; `engine_calls.jsonl` rows add
`thinking_token_budget`.

## 6. How to read the results

- **Good:** every planned wave ended before its planned end; `used` close to the wave's tasks and
  `deadline` 0; in development mode `gained` not empty and `lost` empty; `run.json` problems
  empty. For `P6.5-03` this is the Pass row with the notebook's timing.
- **Borderline:** the last wave's calls cut at the stop (`deadline` above 0): the plan was too
  tight for this task set; tasks lost are only the shot's. Read `actual_s` against `planned_s`
  before changing `plan_factor` or `margin_s`.
- **Bad:** `lost` not empty (the merge replaced a right NVARC attempt 2: only the $(N_1, S_1)$
  case can), any problem with the engine (boot, death, sanity, degenerate), or no valid file
  (exit 1). For `P6.5-03` these are the Fault or Timing fail rows: fix, nothing submitted.

## 7. Failure modes

- **OOM or a small KV pool:** vLLM preempts and recomputes (a delay); the plan takes fewer tasks
  when the boot reports a smaller pool. First change: `gpu_memory_utilization` back to vLLM's
  0.92 only if the boot log shows free memory; else `wave_tasks`.
- **Boot timeout or a failed preparation:** the problem is in `run.json` with the import or log
  error; NVARC's file stands. First check `engine_prep.json` (the jit-cache state) and the
  engine log's first exception.
- **Timeout (calls cut at the stop):** the wave's tasks keep NVARC's attempts. First check the
  decode rate per stream against `stage_arithmetic`'s `least_rate_full` (15.54) in
  `shot_plan.json`; then change `plan_factor`, or `prefill_tokens_per_s` when the forced replies
  ran long, from the wave's `actual_s` against `planned_s`. The sanity request never waits past
  the stop.
- **NaN or degenerate replies:** their tasks keep NVARC's attempts; more than half of a wave ends
  the shot. Check `engine_calls.jsonl` (`run_source`, `token0_count`) and the engine's flags
  against P6.3-05's.
- **Engine death or hang:** the supervisor cancels the wave; NVARC's file and every task merged
  so far stand.
- **In-request forcing not honored** (`budget_forced` 0 while turns run to the cap, then forcing
  turns): the server lacks the reasoning config or ran the V1 runner. Check the engine log's
  `non-default args` for `reasoning_config` and the `Using V2 Model Runner` line; the separate
  forcing turn answered meanwhile, at the old time cost the plan no longer holds, so the last
  tasks may be cut at the stop. First change: `request.forcing_in_request: false`.
- **A refill start waits on KV** (`last_plan` with `kv:` while tasks run): the pool the boot
  reported is smaller than planned; the start follows the next task's end.
- **No readable candidates file:** only tasks with an output NVARC left empty are shot, by task
  id (`selection.fallback` and a problem line in `run.json`, `fallback` in `shot_plan.json`). A
  partial file leaves the tasks it lacks unknown (never shot) unless NVARC's attempt there is
  its placeholder; `shot_plan.json` lists them under `unknown`.
