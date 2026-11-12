# Modules `arc26/agent/` (P6.4): engine server, scheduler and governor, run entry

Task P6.4 of the agent solver (ADR-020). The contract is `arc26/agent/types.py`; the sandbox and
verifier are P6.1's (`agent_sandbox.md`), the loop, engine client, and selection P6.2's
(`agent_loop.md`). The design is `docs/research/agent_library_design.md`, sections 5.1, 5.4 and
5.5; the engine settings are the GPU spike's (`configs/P6.3-01.yaml` on `task/P6.3b`). Config:
`configs/agent_default.yaml`.

## Purpose

The scored run has 12 hours for about 240 hidden tasks on one engine whose throughput is known
only on the night. This task builds the parts around the agent loop that spend that time: an
engine that starts offline, is watched, and is restarted once; phases that try every task cheaply
before spending long attempts; budgets that follow the throughput actually measured; and a
`submission.json` that is valid at every moment, NVARC's attempts filling every slot the agent
leaves, so a crash, a kill, or an engine that never boots still scores.

## Files and interfaces

| File | Main interfaces |
|---|---|
| `server.py` | `ServerConfig`, `vllm_command`, `engine_env`, `clean_env`, `EngineServer`, `EngineSupervisor`, `EngineStartError`, `EngineBootTimeout`, `fake_server_main`, `fake_command`, `fake_completion`, `free_port`, `http_get` |
| `scheduler.py` | `PhaseSettings`, `GovernorSettings`, `SchedulerSettings`, `default_phases`, `RateMeter`, `MeteredEngine`, `Governor`, `ease_features`, `ease_order`, `nvarc_confidence`, `task_solved`, `failed_programs`, `clip_program`, `verified_groups`, `SubmissionWriter`, `Scheduler`, `SchedulerReport`, `EngineGate` |
| `run.py` | `main`, `run`, `RunConfig`, `RunOptions`, `config_from_mapping`, `load_config`, `resolve_options`, `nvarc_copy`, `score_dev`, `ConfigError` |

## Engine server

`EngineServer(command, env, log_path, root_url, ...)` runs one engine process in a session of its
own (`start_new_session`), output appended to `engine.log`, working folder beside the log (never
the notebook's output). `start(timeout_s)` waits until `/health` answers 200: it raises
`EngineStartError` when the process exits during its boot (the log tail quoted) and
`EngineBootTimeout` when the time runs out, after stopping the process group in both cases, so a
failed boot holds neither the GPUs nor the port. `stop()` sends SIGTERM to the group, SIGKILL after
`stop_timeout_s`, then SIGKILL to whatever the group still holds once its leader is gone (vLLM's
engine core and workers); no start follows a `stop()`, which a lock orders against a restart in a
worker thread. A pid file records the engine's pid and argv; the next start first stops a process
the file names when its argv still matches (a console script gains its interpreter in front), so
a run killed hard does not leave the next one without GPUs or port. A process group counts as gone
when it has no member other than zombies, since the parent of an orphan (the container's first
process) may never reap it.

The engine command is `vllm_command` (the spike's flags from `ServerConfig`) run from the venv's
`bin`, in `engine_env`: `clean_env` (no `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV`, `PIP_*`, the
config's `unset_env`; `PYTHONNOUSERSITE=1`), the config's `env`, the venv's `bin` first on
`PATH`.

`EngineSupervisor(root_url, server, ...)` watches from the event loop every `monitor_interval_s`:
a crash is the process exiting, a hang is `hang_s` without a healthy `/health` answer while it
runs. Either clears the readiness the scheduler waits on and restarts the engine once (`restarts`,
not after the deadline); a failed restart or a second failure makes the engine gone for good.
`wait_ready()` answers True while the engine is up, waits while it restarts, and answers False once
it is gone. A server given by URL (`server` None) cannot be restarted: `hang_s` without a healthy
answer makes it gone. Between a crash and the next check (at most `monitor_interval_s`) the
engine still counts as ready; attempts started then end with an engine error and are run again.
A check that raises is logged and the watch goes on; a restart that raises counts as failed.

The fake engine (`python -m arc26.agent.server fake --port N --replies R.json`) serves `/health`,
`/v1/models`, and `/v1/chat/completions` with the standard library: the task is the one the first
user message names (`Task <id>: ...`), the turn the number of assistant messages, the reply the
script's entry (text, or text, reasoning, and delay) or a reply without code past the script;
tokens are one per 4 characters, a reply over `max_tokens` is cut reasoning first and finishes
with `length`. Test options: a delay per request, a slow boot, a failed boot (exit 1), and a crash
(exit 3, connections dropped) on the $n$-th completion, once when a marker file is given.

## Phases

The scheduler rolls through four phases (design note 5.4): a free slot takes the first task phase
1 still needs, else phase 2, 3, or 4, so the engine does not idle at a phase boundary while the
last long attempts of the previous phase finish. Within a phase, tasks with fewer attempts made go
first. A task has at most one attempt in flight. The scheduler waits for the engine only when it has
an attempt to start, and picks that attempt (and its budget) after the wait.

| Phase | Tasks | Order | Attempt (default config) |
|---|---|---|---|
| 1 | every task; `attempts` per task, solved or not (more than 1: independent votes) | predicted ease | low effort, 3 turns (2 feedback rounds), 16,384 per turn, 8,000 to 24,000 in all, share 0.5 |
| 2 | no verified program yet | NVARC's confidence, lowest first; else ease | xhigh, 5 turns, 32,768 per turn, 16,000 to 65,536, the best failed programs shown |
| 3 | partial progress (a failed program reproduced at least `min_pairs_passed` demonstrations), or verified programs that disagree on a test output | partial progress first (most demonstrations passed, then cell accuracy), then disagreement | xhigh, up to 2 attempts per task, 16,000 to 100,000; failed programs shown, none for a solved task (an independent vote) |
| 4 | until the generation stop: unsolved tasks, then tasks whose verified programs disagree | round robin by all attempts so far, unsolved first at equal counts | xhigh, no limit per task, 16,000 to 65,536, new seeds; failed programs shown, none for a solved task |

`attempts` 0 turns a phase off; only the last phase may have no limit (`null`). Phase 4 keeps the
engine busy until the generation stop instead of ending early once phases 1 to 3 are done; its
budget spreads the time left over the tasks in the cycle. In phase 4 each attempt of a task is one
round, and a task leaves the cycle only after `cycle_error_rounds` (2) consecutive rounds end with
an engine error (or raise); any other outcome resets the count, so errors hours apart never add
up. A resume rebuilds the count from the attempt log.

**Ease** (`ease_order`): a task's score is the sum of its ranks under three criteria counted
equally, ties sharing their mean rank: fewer cells over all its grids (the prompt's size), every
demonstration keeping its shape, fewer distinct colors. Ties by cells, colors, task id.

**NVARC's confidence** (`nvarc_confidence`): with NVARC's `candidates.json` (view file
`<task>_<i>.<ops>` to `{"solution", "beam_score", "score_aug"}` per beam), a test output scores
ADR-014's agreement $|C_{\hat y}| / n$, the view candidates equal to NVARC's top grid $\hat y$ over
$n = 16$ views, at most 1. $\hat y$ is NVARC's attempt 1 when its submission is given, else the
reference's vote (`arc26.loop.tally`). With the submission alone, an output scores 1 when NVARC
filled attempt 1 and 0 for its `[[0]]` placeholder. A task scores the mean over its outputs (an
output without candidates 0), as the task score averages its outputs.

**Failed programs** (`failed_programs`) are the task's unaccepted programs without a fatal guard
finding, one per source, ordered by demonstrations passed, then mean cell accuracy, so the best
come last, where the prompt shows them. Hardcoded programs are never shown. At most
`prior_programs` (2; P6.2's prompt shows no more) are passed, each clipped by `clip_program` so
that with its report it fits `prior_chars` (6,000): P6.2 clips the report to 3,000 characters and
the source keeps the rest, head and tail around a `[... n characters cut ...]` marker. Uncapped,
two programs of 30,000 characters left a later attempt no room for a reply.

**Attempts without room.** An attempt whose first prompt leaves less than `min_turn_tokens` of
room ends with `budget` after 0 turns; it did no work, so it does not count toward its phase's
`attempts`. The next attempt of that task and phase goes without the earlier programs, and a second
such stop ends the phase for the task (it cannot fit at all).

**Engine errors and the deadline.** In phases 1 to 3, an attempt that ends with `engine_error` is
run again later in its phase, up to `engine_error_retries` (1) times, as is an attempt that raised (a REPL or
verifier fault, which leaves no record). An attempt cut by the deadline counts as not run, so a
resumed run repeats it. Attempt ids are `<task_id>/p<phase>/<k>` with the first free $k$, and
each attempt's seed is `attempt_seed(seed, id)`, so ids stay unique and a re-run attempt samples
as before.

## Governor

`RateMeter` tracks every engine call from start to end (`MeteredEngine` wraps the engine; a failed
call is discarded). Phase 2 to 4 calls take up to tens of minutes, so counting the calls that
complete within a short window swings with each burst of completions (with a constant 400
tokens per second, a 15-minute completion count read 84 to 533). Instead, over a window $W$
(`window_s`, 3,600 s), each completed call's tokens are spread evenly over its duration and count
in proportion to its overlap with $(t - W, t]$, and the calls still in flight, whose tokens arrive
only at their end, count their time in the window at the completed calls' rate per call-second:

$$r(t) = \frac{T_c}{S_c} \cdot \frac{S_c + S_f}{\min(W,\; t - t_0)},$$

with $T_c$ the completed calls' tokens in the window, $S_c$ their call-seconds in it, $S_f$ the
in-flight calls' call-seconds in it, and $t_0$ the first call's start: the rate per call in flight
times the mean number of calls in flight. With a constant-rate engine (400 tokens per second shared
by 32 calls of 1,000 to 60,000 tokens) it stays within 15% of 400 after a 30-minute warm-up. While
fewer than `min_calls` (8) completed calls overlap the window (start-up, after an engine restart),
the governor keeps the last measured rate, and before any measurement it uses `prior_rate` (400,
the design note's planning figure, to replace with P6.3-01's measurement). Each attempt's budget is
set when it starts:

$$B = \min\left(c_p,\; \max\left(f_p,\; \left\lfloor \frac{s_p \, \sigma \, r \, T}{N_p} \right\rfloor\right)\right)$$

with $T$ the seconds to the deadline, $\sigma$ the rate safety (0.9: tails below full concurrency,
a restart, attempts cut mid-call), $s_p$ the phase's share, $f_p$ and $c_p$ its floor and cap,
and $N_p$ the attempts the phase still has to run, in flight included (one per task with the
default of one attempt per phase; phase 2 also counts the unsolved tasks still in phase 1; phase 4
counts the tasks in its cycle). An attempt that finishes under budget leaves more for the tasks
after it, and a slow night shrinks every later budget. `run.json` records each attempt's plan
($N_p$, the rate and its source, the seconds left) beside its budget. The reply cap is $\min(\text{turn\_tokens}, B)$, and P6.2's loop keeps every
prompt and reply under the engine's context (`max_model_len`).

**Deadline.** The session ends at `--deadline` (Unix time), or `session_s` (43,200 s) after
`--session-start` (Unix time, the notebook's start). Competition mode requires one of the two: the
run starts after NVARC's stage and cannot tell how much of the session is gone, so `--budget-s`
(seconds from now) is refused there. Development mode may give `--budget-s` or nothing: the end is
then `session_s - startup_margin_s` from this process's start (`run.json` names the source).
Generation stops `final_margin_s` (1,800 s) before the end, 11.5 hours into the session. No attempt
starts after the deadline; running attempts stop at their next engine call (P6.2's deadline, with the stop
reason `deadline`) and are logged; any still running `grace_s` (60 s) later are cancelled, their
REPLs closed from a daemon thread so a stuck REPL call cannot hold the cut.

## Crash safety and resume

After every finished attempt the record is appended to `attempts.jsonl` (one complete line, a cut
last line terminated before the next append), then `SubmissionWriter.add` re-selects that task with
P6.2's `build_selection` and rewrites the whole `submission.json` through `write_submission`
(validated against the task list, temporary file, `os.replace`). Every slot the agent does not fill
holds NVARC's attempt or `[[0]]`. A failed write leaves the previous file and is counted. On a
restart with the same log folder the run reads `attempts.jsonl` back (`read_records` skips a cut
line), rebuilds the same submission at once, and the scheduler skips every logged attempt: an
attempt in flight at the kill has no record and runs again with its id and seed.

**NVARC's copy.** The notebook may name one file for NVARC's submission and this run's output,
which the run rewrites. The first start therefore copies NVARC's file into the log folder
(`nvarc_submission.json`, before the first write of the output) and every start reads NVARC's
attempts from that copy, which is never overwritten; a resume never takes its own output for
NVARC's. A file unreadable at the first start leaves a copy holding `null`, and later starts run
without NVARC. An unusable log folder is replaced by one under the temporary directory named after
the output path, the same at every start, so a resume still finds its log and its copy.

## Run entry

```text
python -m arc26.agent.run --mode {dev,competition} --tasks <challenges.json> --out <submission.json>
    --config configs/agent_default.yaml
    (--engine-venv <venv> --model-dir <weights> | --engine-url <root> | --fake-engine <replies.json>)
    (--deadline <epoch s> | --session-start <epoch s> | --budget-s <s>, dev mode only)
    [--nvarc-submission <file>] [--nvarc-candidates <file>]
    [--log-dir <dir>] [--port <n>] [--task-ids a,b,c] [--solutions <file>, dev mode only]
```

Order of work: (1) the arguments and the config are checked, every key required and none unknown;
a refusal exits 2 and writes nothing. (2) The tasks are read (P6.2's loader, which refuses a file
named like a solutions file); NVARC's submission and candidates, an unreadable one reported and
skipped (NVARC's submission from the log folder's copy); the attempt log of an earlier run. (3)
`submission.json` is written at once from NVARC's attempts and the resumed records, before the
engine starts; NVARC's file may be the output path itself, since it is copied first. (4) The engine starts (one more start after a crash during its boot, none after a boot
timeout) or the URL must answer healthy within `boot_timeout_s`; the verifier pool starts; the
scheduler runs. (5) The supervisor, the client, the verifier, and the engine are stopped, the
engine again in `run`'s `finally` in case the event loop was interrupted; the final file is written;
development mode scores it with `arc26.evaluate` when `--solutions` is given. SIGTERM raises
`SystemExit` in the main thread, so a notebook that stops the run still gets the cleanup and the
final write.

Exit codes: 0 when a valid `submission.json` is in place at the end, whatever failed on the way
(each problem is listed in `run.json`); 1 when no valid file could be written (the tasks cannot be
read; an existing output is then left untouched); 2 for refused arguments or config.

**Competition mode never opens a solutions file:** `--solutions` is refused there (exit 2 before
anything is read), and `score_dev`, the only reader of solutions, raises `PermissionError` for any
mode but development before touching the file. Competition mode writes no transcripts.

Outputs in the log folder (default `<out folder>/agent_logs`; on Kaggle better a scratch folder,
which survives a process restart within the session): `attempts.jsonl` (resume and analysis),
`transcripts.jsonl` (development mode, `logs.dev_transcripts`), `timing-engine.jsonl`
(`{puzzle_id, op: "engine", seconds, tokens}` per call), `engine.log`, `engine.pid`, and
`nvarc_submission.json` (NVARC's copy), and `run.json`: the deadline and its source, the boot, the
supervisor's events, the scheduler's report (attempts started, finished, and verified per phase,
stop reasons, generated tokens, each attempt's budget and plan, the last rate and its source), the
submission's slots by source, the problems, and the score.

## Tests

`tests/agent/test_server.py`, `test_scheduler.py`, `test_run.py`: 99 tests, about 25 s on the CPU.

- server, against the fake engine as a real subprocess: the clean environment, the vLLM command,
  config checks, the fake's scripted replies and cut; start, health, a chat call through P6.2's
  client, stop (group gone, pid file removed, no start after); a crash during the boot, a boot
  timeout, a command that cannot run; a worker that ignores SIGTERM killed with the group; a stale
  engine of a killed run stopped before a start on the same port, a pid file naming another program
  ignored; a crash restarted once, a second crash gone for good, a hang restarted, no restart after
  the deadline, a URL server not restarted, a check that raises once without ending the watch;
- scheduler, with an in-process REPL and verifier and P6.2's `ScriptedEngine`: settings checks; ease
  order; NVARC's confidence (top-grid count over 16 views, the vote without a submission, the
  submission alone); the rate's overlap counting with calls in flight and discarded calls; a
  constant-rate engine (400 tokens per second, 32 calls in flight, calls of 80 s to 80 minutes)
  estimated within 15% over 8 hours; the last measured rate kept when the window is thin; the budget
  arithmetic (prior, measured, floor, cap, past the deadline); with a fake clock that advances by
  tokens over a fixed rate, the phase order (ease, NVARC's least confident, partial progress), each
  attempt's budget and $N_p$ against the formula, the effort per phase, and the prior programs
  shown; $N_p$ counting attempts in flight and, for phase 2, unsolved tasks still in phase 1; the
  shown programs capped in number and size; an attempt without room run again without them, and two
  such ending the phase; the cycle working until the stop (round robin, new seeds, unsolved tasks
  with their programs, disagreeing tasks without, $N_4$ the tasks in the cycle) with a valid
  submission after every write; a task whose cycle attempts fail every other round kept to the
  stop, one failing every round dropped after 2, and the count rebuilt on a resume; phase
  2 in ease order without NVARC; phase-1 votes whose disagreement reaches phase 3; engine errors
  retried once with new ids; a sandbox that cannot start
  (no endless loop); the deadline with a valid submission and NVARC's fill; attempts stuck past the
  grace cancelled; an engine gone for good, a gate that never answers, and nothing left to start
  while the engine restarts (no wait); a kill (cancelled run, a
  cut log line) and a resume that runs no logged attempt again; the writer's fill, updates, failed
  writes, and resume;
- run entry: the default config, its server flags equal to the spike's stage 1, strict config
  checks; competition mode refusing `--solutions` (nothing opened, nothing written) and `score_dev`
  refusing it; competition mode refusing a run without `--deadline` or `--session-start`, and
  `--budget-s`; the end from `--session-start` and development mode's default; refused arguments;
  a resume whose output is NVARC's file keeping NVARC's attempt 2 from the copy; a NVARC file
  unreadable at the first start never read later; a SIGKILL during the first engine call leaving a
  valid file; an engine that cannot boot, an unreachable URL, an unreadable NVARC
  file, unreadable tasks, a deadline before the boot (NVARC's attempts kept, exit 0); a competition
  run that never reads solutions and a dev run that scores (5 of 6 synthetic tasks); an engine crash
  mid-run restarted; SIGTERM; a sandbox that cannot start; an unusable log folder (temporary logs,
  the run goes on); and the end-to-end CPU run with P6.1's real
  REPL and verifier and the fake engine started by the run: SIGKILL mid-run, the file valid at that
  moment, a resume that stops the killed run's engine, runs no logged attempt again, and ends with a
  valid submission scoring 5 of 6.

## Limits and open questions

- The budgets and the rate prior are design targets; the spike's measured rate (P6.3-01) and solves
  (P6.3-02) should reset `prior_rate`, the phase floors and caps, and the concurrency before the
  scored run.
- With one attempt per task and phase (the default), a task has at most one verified program, so
  the disagreement rules of phases 3 and 4 only act when phase 1 runs more than one attempt per
  task.
- The rate estimate assumes a call's tokens arrive evenly over its duration and that calls in
  flight generate at the completed calls' rate; when the concurrency changes quickly (a phase tail,
  a restart) it lags for up to a window.
- Between a crash and the supervisor's next check (up to 5 s) new attempts can start and end with
  engine errors; each costs a retry.
- A server given by URL is declared gone only after `hang_s` without a healthy answer; attempts
  started meanwhile fail and use their retries.
- A record holding NaN cannot be logged (P6.2); it still reaches the submission, and a resumed run
  repeats that attempt.
