# Module `arc26/agent/shot.py` (P6.5b): the long shot after NVARC

Task P6.5b, Part A (`docs/tasks/P6.5b.md`, design points 2 to 7). Config:
`configs/agent_shot.yaml`. Runbook: `docs/runbooks/agent_shot.md`. The notebook stage that runs it
(Part B) is in `docs/runbooks/build_submission.md`. Supporting modules moved from
`scripts/agent_spike.py` without change: `arc26/agent/engine_prep.py` and
`arc26/agent/recording.py`.

## Purpose

Run `P6.3-05` solved 4 of the 15 DEV40 tasks NVARC is least sure of, NVARC 0, with one xhigh
thinking turn of up to 94,208 tokens per task and direct answers. Submission 2's NVARC stage
leaves about 3 of the 12 hours idle on 240 tasks. `python -m arc26.agent.shot` runs that shot in
the time NVARC leaves, on NVARC's least sure tasks, and merges its answers into NVARC's file so
that the file is valid, and at least NVARC's, at every moment.

## Files and interfaces

| File | Main interfaces |
|---|---|
| `shot.py` | `main`, `run`, `build_parser`, `load_config`, `config_from_mapping`, `ShotConfig`, `WaveModel`, `RequestSettings`, `PrepSettings`, `stage_arithmetic`, `output_key`, `output_confidence`, `shot_confidence`, `confidence_without_candidates`, `select_tasks`, `opening_bound`, `WavePlan`, `plan_wave`, `shot_answers`, `merge_output`, `ShotWriter`, `read_shot_log`, `prepare_venv`, `boot_engine`, `url_ready`, `kv_pool`, `sanity_check`, `ShotState`, `run_wave`, `run_waves`, `shot_dev_metrics`, `NoRepl`, `NoVerifier` |
| `engine_prep.py` | `prepare_engine`, `preflight`, `parse_boot_log`, `cache_config_info`, `Stop` (and the probe scripts) |
| `recording.py` | `RecordingEngine`, `recording_engine`, `call_health`, `token_stats`, `reasoning_location`, `JsonlWriter` |

The command line is `arc26.agent.run`'s parser (`build_parser` renames it), its option
resolution (`resolve_options`, with this module's `load_config`), its exit codes (0 a valid file,
1 none, 2 refused), and its SIGTERM handler. The notebook therefore switches between the agent
and the shot by one config key.

## Data flow

1. Arguments and config; a refusal exits 2 and writes nothing.
2. Tasks (`arc26.agent.tasks.load_tasks`); NVARC's submission through the run's own copy
   (`run.nvarc_copy`); NVARC's candidates. `ShotWriter` writes the file at once from NVARC's
   attempts.
3. The queue: `shot_confidence` per task, `select_tasks` under `max_confidence`, least sure first,
   ties by id. A resumed run (an existing `shot_attempts.jsonl` in the log folder) skips the tasks
   it already shot and merges their usable records.
4. Only when the queue is not empty and `setup_s` plus the smallest wave fit before the
   generation stop: with `--engine-venv`, `prepare_venv` (`engine_env`, then
   `engine_prep.prepare_engine` and `engine_prep.preflight`), `vllm_command` started by
   `EngineServer` and `boot_engine` (synchronous, in the main thread); with `--fake-engine`, P6.4's
   fake engine; with `--engine-url`, a health wait. Then `kv_pool` (the engine log's `KV cache
   size`, else `/metrics`, else the config's) and `boot.json`.
5. `run_waves`: an `EngineSupervisor` with no restart, the recording client, `sanity_check`, then
   waves: `plan_wave` on the time left, `run_wave` (P6.2's `run_tasks` with `NoRepl`,
   `NoVerifier`, the cap's `Budget`, phase 0, all of the wave's tasks at once), each finished
   attempt logged and, when usable, merged and the file rewritten. A wave is cancelled when the
   supervisor reports the engine gone.
6. The engine is stopped, the file written once more and checked on disk; in development mode
   `run.score_dev` and `shot_dev_metrics`; `run.json` and `summary.md`.

## The confidence (design point 3)

Per test output, the view candidates of `candidates.json` (file names `<task>_<j>.<view ops>`,
read in sorted order as `scripts/calibration_report.py` reads them) are tallied by
`arc26.loop.tally`, the moved `score_kgmon` selection, whose score of a grid $g$ is

$$
s(g) = |C_g| - \frac{1}{|C_g|} \sum_{c \in C_g} \bar s(c),
$$

with $C_g$ the candidates equal to $g$ and $\bar s(c)$ the mean of a candidate's `score_aug`. The
output's confidence is the softmax at $T = 1$ of its rank 1 grid,

$$
\hat p = \frac{1}{\sum_g e^{(s(g) - s_1)/T}},
$$

which is `softmax_p` of `rank_kgmon` 1 in `reports/P2.8/<run>.csv`. A non-finite $\hat p$ (a NaN
or infinite `score_aug`) counts as 0, least sure, so the minimum below does not depend on the
order of the outputs. An output without
candidates is 0 where NVARC's attempt 1 is its placeholder (NVARC had no candidate: P6.3-05's 0;
also every output when NVARC's file is missing) and unknown where NVARC gave a real attempt (a
partial candidates file). A task's confidence is the minimum over its known outputs, since it
scores only when every output is right; the true minimum is at most that, so a task it puts under
`max_confidence` belongs there. A task with no known output is unknown and never taken. This is
not the scheduler's `nvarc_confidence` (ADR-014's agreement, mean over outputs).

With no readable candidates file (missing, not JSON, or not an object) every output is without
candidates, so only tasks with an output NVARC left empty are shot, all at 0, by task id
(`confidence_without_candidates`). A readable file on which the confidence fails takes the same
fallback. The fallback is logged in `run.json` (`selection.source`
`submission`, `selection.fallback`, and a problem line) and in `shot_plan.json`
(`confidence_source`, `fallback`, and the `unknown` tasks).

## The waves (design point 4)

With $r$ the measured decode rate per stream (19.0 tokens per second), $\phi$ the plan factor
(0.85), $p$ the measured prefill rate of the forced replies (958.8 tokens per second), $F$ the
forcing cap (6,144), and $m$ the margin (300 s), a wave of $n$ tasks with thinking cap $C$ and
longest opening bound $o$ is planned to take

$$
T(C, n, o) = \frac{C}{\phi r} + \frac{n\,(o + C)}{p} + \frac{F}{r} + m :
$$

the thinking turn at the planned rate, then the forced replies assuming every turn of the wave is
cut (the coordinator's decision, 14 Oct): their $n$ prompts of up to $o + C$ tokens prefilled at
$p$, and one forced reply of up to $F$ tokens generated at $r$. $p$ is P6.3-05's 8 forced replies:
783,425 prompt tokens from their common start to the last end, 817.1 s (`engine_calls.jsonl`);
their generation of 520 to 3,820 tokens falls inside that time, so $p$ is low and safe.

`WaveModel.largest_cap` gives the largest $C \le$ `max_cap` with $T \le L$, the seconds left
before the generation stop:

$$
C = \min\!\left(C_{\max}, \left\lfloor \frac{L - m - F/r - n o / p}{1/(\phi r) + n/p} \right\rfloor\right),
$$

a slope of 0.077565 s per token for 15 tasks. Below `min_cap` (16,384) no wave starts. The wave
takes the queue's first `wave_tasks` (15); while

$$
\sum_{i=1}^{n} (o_i + C + F) > K,
$$

it drops its last task and plans $C$ again for the smaller wave (it can only grow). $o_i$ is the
opening's bound (`opening_bound`: one token per character plus the loop's template slack, 1.28
times the `/tokenize` count on P6.3-05's 15 openings, 57,627 against 45,116) and $K$ the pool the
boot reported. Dropped tasks stay first in the queue. A task is shot once.

With the shipped constants and P6.3-05's longest opening bound $o = 6{,}771$
(`waves.opening_tokens`, `stage_arithmetic`):

| Quantity | Value |
|---|---|
| full wave, 15 tasks at 94,208 | $5{,}833.3 + 1{,}579.8 + 323.4 + 300 = 8{,}036.5$ s |
| smallest wave, 15 tasks at 16,384 | $1{,}014.5 + 362.3 + 323.4 + 300 = 2{,}000.1$ s |
| least time before the stop to start the engine | $450 + 2{,}000.1 = 2{,}450.1$ s |
| lowest rate at which a time-bound full wave of 15 ends | 15.54 tokens per second (18% under 19.0) |
| the same at 16,384 | 13.75 tokens per second (28% under) |

The last two rows (`WaveModel.least_rate`) solve
$(C + F)/r' = C/(\phi r) + F/r + m$: the turn and the forced reply's generation run at the real
rate $r'$, the prefill as planned. P6.3-05's streams ran at 16.5 to 17.6 tokens per second each
while 11 were busy at 50k to 70k contexts (`server_metrics.jsonl`), inside both bounds.

Each task's request is P6.3-05's (`RequestSettings.budget`): one thinking turn at xhigh with
`DIRECT_TWO_SYSTEM_PROMPT`, `preserve_thinking`, an `answer2` forcing turn of up to 6,144 tokens
after a cut turn, `NoRepl` and `NoVerifier`, temperature 1.0, the seed of attempt
`<task>/p0/0`, the client's `return_token_ids` and `/tokenize` count; only `turn_tokens` $= C$
and `attempt_tokens` $= C + F$ follow the wave. At $C = 94{,}208$ the budget equals P6.3-05's.

## The merge (design point 5)

Per test output of a shot task, with $N_1, N_2$ NVARC's attempts ($N_1$ a placeholder when NVARC
had no candidate) and $S_1, S_2$ the shot's first two different valid answers (`shot_answers`):

| $N_1$ | $N_2$ | $S_1$ | $S_2$ | Slots |
|---|---|---|---|---|
| any | any | none | none | $(N_1, N_2)$ |
| real | any | $\ne N_1$ | any | $(N_1, S_1)$ |
| real | real | $= N_1$ | any | $(N_1, N_2)$ |
| real | placeholder | $= N_1$ | given | $(N_1, S_2)$ |
| real | placeholder | $= N_1$ | none | $(N_1, N_2)$ |
| placeholder | any | given | given | $(S_1, S_2)$ |
| placeholder | any | given | none | $(S_1, N_2)$ |

The fourth row amends the brief's $(N_1, N_2)$ for $S_1 = N_1$ (the coordinator, 14 Oct): a
placeholder slot never beats a real answer.

A record is merged only when it has answers, did not stop at the generation stop, and no reply of
it was degenerate. `ShotWriter` keeps NVARC's grids in their slots (an empty one written as
`[[0]]`) and rewrites the file atomically and validated after every record
(`SubmissionWriter.write`).

## Deadline and health (design points 6 and 7)

Every call of a wave carries the generation stop (`--deadline` minus `final_margin_s`) as its
deadline: at the stop P6.2's loop cancels it (`asyncio.wait_for`, which closes the connection, so
vLLM aborts the request) and the task keeps NVARC's attempts. SIGTERM raises `SystemExit` in the
main thread; the waves are cancelled, the engine stopped, and the file written. The sanity
request waits at most `min(timeout_s, stop - now)`; a timeout fails it. The engine has no
restart: a failed boot, a death (the supervisor's process and `/health` checks), a failed or
degenerate sanity request, more than half of a wave's tasks degenerate, or a wave whose every
attempt ended with an engine error ends the shot with the file as it is.

## Task P6.9's options

Four optional config keys, each off when absent (`configs/agent_shot.yaml`, Submission 3). The
Submission 4 candidate `configs/agent_shot_P6.9.yaml` sets them as run `P6.9-01`'s rows decided
(`docs/experiments/P6.9-01.md`): `selection.order: buckets` with `mid_bucket` false, `waves.refill`
true, `request.forcing_in_request` false (the separate forcing turn, so no `thinking_end_note` and
no `--reasoning-config`), `request.lenient_answers` true; its 12 h check is `P6.9-02` (runbook,
step 8; `docs/tasks/P6.9.md`):

| Key | Function | Effect |
|---|---|---|
| `selection.order: buckets`, `selection.mid_bucket` | `confidence_bucket`, `select_tasks` | $(0, 0.5)$, then optionally $[0.5, 0.7)$, then $c = 0$ |
| `selection.zero_bucket` (task P4.7b; default true) | `select_tasks` | false: no task at $c = 0$ in the queue, in either order; a tail build's rank-256 tail takes them (`configs/agent_shot_P4.7b.yaml`) |
| `selection.order: opening` (task P5.1) | `select_tasks`, `run` | every task, by ascending opening bound then id; NVARC's confidence is not computed (an LLM-first build runs no NVARC); needs `zero_bucket` true (`configs/agent_shot_P5.1.yaml`) |
| `waves.dev_cap` (task P5.1; default none) | `for_mode`, `run` | in development mode only, `max_cap` and `min_cap` lowered to it (`shot_plan.json`'s `dev_cap`); a competition rerun never reads it |
| `waves.refill` | `plan_start`, `run_refill`, `_refill_stop` | a stream that ends starts the next task with the cap the time left allows |
| `request.forcing_in_request`, `request.thinking_end_note` | `WaveModel.forcing_seconds`, `RequestSettings.budget`, `arc26.agent.loop.Budget` | vLLM's `thinking_token_budget` in the thinking request; the plan's prefill term is 0 |
| `request.lenient_answers` | `implied_shapes`, `shot_answers`, `ShotWriter(shapes=...)`, `arc26.agent.loop.parse_answer(lenient=True)` | lenient answer blocks; $S_1$ the first answer of the implied shape |

The loop records `AttemptRecord.budget_forced` and `AttemptRecord.answer_lenient`;
`record_to_json` leaves both out at their defaults, so Submission 3's logs are unchanged. The
engine client sends `thinking_token_budget` only when the loop passes it; P6.4's fake engine
honors it with the note named by the server's `--reasoning-config`
(`arc26.agent.server.reasoning_end_str`).

## Tests

`tests/agent/test_shot.py` (44 tests, CPU, about 19 s): the confidence equals `softmax_p` on all
142 rank 1 rows of `reports/P2.8/P2.6-01.csv` from a reduced copy of P2.6-01's
`candidates.json` (`tests/testdata/p2_6_01_candidates_reduced.json.gz`: grids replaced by their
index within the output, `beam_score` dropped) and picks P6.3-05's 15 DEV40 tasks; a partial candidates file (absent tasks unknown, NVARC's
placeholders 0) and an unreadable one (the fallback, logged); the requests
of one full wave are byte-identical to those of the spike's own `direct_phase` with
`configs/P6.3-05.yaml` (both run against one recording stand-in for the HTTP client on stand-in
tasks under DEV40's ids: 15 openings, 8 forcing turns, 15 prompt counts); the client config,
budget, vLLM command (P6.3-05's `server_command`), environment, and engine imports are P6.3-05's;
the moved code's source is the spike's; the wave arithmetic (full, reduced, skipped, KV-bound)
and its constants (the prefill rate and the longest opening from P6.3-05's logs, the lowest
rates a wave of 15 tolerates); every merge case; a NaN score; a failing confidence's fallback; a
sanity request bounded by the stop; end to end on the fake engine: a dev run with its metrics,
a call cut at the generation stop, an engine that never answers, degenerate replies, a failed boot, an engine death, SIGTERM,
too little time, competition mode without solutions, the arguments and exit codes.
`tests/agent/test_p6_9_shot.py` covers task P6.9's options: the defaults, the in-request request
and its fallback, the lenient rules and their invariance on strict-valid blocks, the bucket order,
the implied shapes, the refill plans and stop rules, the candidate config, and a refill run with
in-request forcing end to end on the fake engine.

## Known limits

- The plan's rates are P6.3-05's, measured on 15 DEV40 tasks; a hidden task set
  with longer openings or a slower decode runs closer to the margin (runbook, failure modes).
- A task whose forcing call is cut at the generation stop loses the answers its cut turn may have
  shown; design point 6 keeps NVARC's attempts for every call still running.
- The opening bound over-counts 1.28 times on P6.3-05's openings, so a KV-bound wave may take
  fewer tasks than the pool would hold; with P6.3-05's pool the full wave needs 92% of it.
- The plan assumes every turn of a wave is cut, so a wave of tasks that end their thinking early
  (7 of P6.3-05's 15) finishes well before its plan; the next wave then plans on the time left.
