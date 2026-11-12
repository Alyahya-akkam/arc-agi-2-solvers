# ADR-021: Submission 3: NVARC unchanged, then the long shot in the time it leaves

- Status: Accepted, 2026-10-14, on the human's go of 14 Oct 12:50 (task brief
  `docs/tasks/P6.5b.md`): NVARC first with Submission 2's deadlines, then `P6.3-05`'s shot on
  NVARC's least sure tasks in the time NVARC leaves, and a 12 h timing check on 4xL4 before any
  submission. Drafted by the kaggle-engineer of task P6.5b. Decision 2's `min_stage_s`,
  decisions 3 and 4, and with them the minimum cap, the request's forcing, and the answer parser,
  are changed for Submission 4 by [ADR-023](ADR-023-submission-4-shot-v2.md) (19 Oct, Proposed
  until `P6.9-02`); Submission 3's version keeps them.
- Tasks: P6.5b (the shot module `arc26.agent.shot`, Part A; the build and its runs `P6.5-02`,
  `P6.5-03`, Part B).
- Related: ADR-018 (Submission 2, whose NVARC this keeps), ADR-020 (the agent solver and the
  hybrid build this extends), ADR-008 (Kaggle access), ADR-010 (packaging).

## Context

- **The shot works where NVARC fails.** `P6.3-05` (4xL4, Qwen3.8-27B INT4 on vLLM 0.31.0, xhigh,
  direct answers, one thinking turn of up to 94,208 tokens, 15 tasks in one wave) solved 4 of the
  15 DEV40 tasks NVARC is least sure of in `P2.6-01` (`136b0064`, `7b5033c1`, `80a900e0`,
  `dd6b8c4b`; score 4.5 of 15), against NVARC's 0 of 15: its pre-registered Enters row
  (`docs/experiments/P6.3-05.md` on `task/P6.3c`). The wave took 5,784 s; the model wrote one
  answer per test output, so no solve came from a second answer.
- **NVARC leaves time.** Submission 2's NVARC projects $u = 0.736$ to $0.741$ of its soft deadline
  on 240 tasks (`P2.6-01`): it ends near 8.5 h, about 3 h before its own soft deadline.
- **NVARC's confidence picks the tasks.** P2.8's selection signal per test output (the softmax at
  temperature 1 of `score_kgmon` over the output's distinct grids, the top grid's share; AUROC up
  to 0.89 on DEV40), per task the minimum over its outputs, 0 for an output without candidates.
  The buckets on DEV120 (`reports/P6.3-05/merge.md` on `task/P6.3c`, from
  `reports/P2.8/P2.6-01.csv` and `P1.1-01.csv`): below 0.5 NVARC solves 0 of 34 tasks
  (`P2.6-01`) and 1 of 19 (`P1.1-01`), and no output there is right by attempt 2 alone (0 of 56,
  0 of 31); in $[0.5, 0.7)$ it solves 6 of 23 and 4 of 23, with 3 of 37 and 5 of 38 outputs right
  by attempt 2 alone. On about 240 tasks, below 0.5 holds about 68 at `P2.6-01`'s share, more
  than the 15 to 30 the time affords.
- **The merge table** (`reports/P6.3-05/merge.md`): on the 15 tasks the planned rule (below), the
  shot alone, and $(S_1, N_1)$ all score 4.5 of 15 against NVARC's 0; they tie because NVARC is
  right nowhere there and $S_2$ never exists, so the buckets decide the rule.

## Decision

1. **NVARC unchanged.** NVARC's stage is Submission 2's (`configs/submission_P2.6-01.yaml`),
   with the competition's deadlines (soft 41,400 s, hard stop 42,000 s): no share cut. The build
   key `agent.entry: arc26.agent.shot` requires `nvarc_share_s` 41,400, so `nvarc_deadlines`
   never shortens them. NVARC's marginal task is worth far more than the shot's.
2. **The shot only in the time NVARC leaves.** The agent stage starts after NVARC's write and
   runs the shot only with at least `min_stage_s` (2,700 s: the install, the GPU wait, the shot's
   setup and smallest wave) before its generation stop; the builder plans the latest such start
   (`shot_plan`: $L = E - f - m$, 37,230 s in `P6.5-03`) instead of a worst-case start.
   **The shot plans for the time before the first cell** (the coordinator's decision of 14 Oct,
   after `P6.5-02`: Kaggle showed RUNNING 36.7 minutes before the notebook's first cell; 41
   minutes in `P6.3-02` revision 5, about 10 in `P6.3-04`). If the 12 h limit counts from
   RUNNING, a shot planned against 43,200 s of notebook time would be killed mid-wave, so the
   shot plans against 12 h less a 45-minute allowance: its end $E = 43{,}200 - 2{,}700 - 120 -
   30 - 300 = 40{,}050$ s, the notebook's last line by 40,500 s of notebook time. The allowance
   applies to the shot only: NVARC keeps Submission 2's deadlines, which it never came near in
   either scored run (Submissions 1 and 2 both scored; `P2.6-01`'s solver ended at 15,703 s on
   DEV120 and projects about 30,700 s on 240 tasks). A shot build logs, from inside, the clocks
   that could see the minutes before the first cell (`stages.environment.clocks`), so `P6.5-03`
   shows which of them does.
3. **Selection:** the softmax rule above, computed in the notebook from NVARC's candidates,
   ascending, ties by task id, never a task at or above `max_confidence` 0.5
   (`configs/agent_shot.yaml`).
4. **Waves sized by the time model:** up to 15 tasks at once with `P6.3-05`'s request; the
   thinking cap $C$ of each wave the largest whose planned length
   $T = C / (0.85 \cdot 19.0) + n (o + C) / 958.8 + 6{,}144 / 19.0 + 300$ s ends before the
   generation stop and whose KV need fits the pool the boot reports; none below $C = 16{,}384$; a
   task is shot once.
5. **The merge, per test output** ($N_1, N_2$ NVARC's attempts; $S_1, S_2$ the shot's first two
   different answers): no $S_1$: $(N_1, N_2)$; $N_1$ real and $S_1 \ne N_1$: $(N_1, S_1)$; $N_1$
   real and $S_1 = N_1$: $(N_1, N_2)$, or $(N_1, S_2)$ when $N_2$ is a placeholder; $N_1$ a
   placeholder: $(S_1, S_2)$, or $(S_1, N_2)$ without $S_2$. The file is rewritten atomically after
   every task, and the notebook's final check restores NVARC's grid in any invalid slot.
6. **At the shot's end** every call still running is cancelled and its task keeps NVARC's
   attempts; SIGTERM at $E$, SIGKILL 120 s later, the engine's group stopped through
   `engine.pid`.
7. **Runs before any submission:** the free CPU twin `P6.5-02-cpu`, the DEV4 smoke `P6.5-02`,
   then the 12 h check `P6.5-03` on DEV240 (DEV120 and a copy of each task under a derived id) at
   the competition's times on `alyahya737/arc26-submission`; its version is Submission 3 only on
   its brief's Pass row, and the human submits.

## Alternatives considered

- **A share cut for the shot** (the hybrid of ADR-020 with NVARC stopped at a share). Lost:
  NVARC's last tasks on a slow hidden set, each worth more than a shot task.
- **The program-writing agent of ADR-020 in the remaining time.** Lost: `P6.3-02` and `P6.3-04`
  solved 0 and 1 of 29 NVARC misses; the shot solved 4 of 15.
- **`max_confidence` 0.7.** Lost: in $[0.5, 0.7)$ NVARC solves a quarter of the tasks and 8% to
  13% of the outputs are right by attempt 2 alone, which the merge would put at risk, against a
  shot rate measured only below 0.5; and below 0.5 already holds more tasks than the time
  affords.
- **$(S_1, S_2)$, the shot alone, on its tasks.** Lost: NVARC's attempt 1 is right on 1 of 19
  tasks below 0.5 in `P1.1-01`; keeping $N_1$ costs nothing measured.
- **Fixed waves at the full cap.** Lost: a second wave cut before its forcing turns answers
  nothing; the time model sizes it to what is left.

## Consequences

- The version keeps NVARC's attempts at every moment: whatever fails in the shot (install, boot,
  engine, deadline), Submission 3 scores at least what NVARC's attempts score.
- On a hidden set whose NVARC ends after about 37,140 s ($u$ above about 0.896 on 240 tasks) the
  shot does not run, and Submission 3 equals Submission 2 in its attempts.
- At $u = 0.74$ the shot gets about 8,700 s of waves: one full wave of 15, a second of 15 only if
  the first ends early (as `P6.3-05`'s did). The expected gain is small and unmeasured on the
  hidden set: about $s$ tasks for $s$ solved of the 15 to 30 shot, 0.42 points each on 240 tasks; DEV40's least sure 37.5% may be easier than
  the hidden set's least sure 6 to 12%.
- The shot's GPU time comes out of the scored session, not the weekly quota; its development
  runs (`P6.5-02`, `P6.5-03`) cost about 1.3 h and 12 h of 4xL4.

## Overturn criterion

- `P6.5-03` reads Fault or Timing fail: no submission; the design is fixed and measured again.
- The leaderboard score of Submission 3 falls 2 points or more below Submission 2's (the noise
  floor, CLAUDE.md): the shot is dropped from the final picks.
- A measured shot rate in $[0.5, 0.7)$ above NVARC's there (a run on those tasks): raise
  `max_confidence` under a new brief.
