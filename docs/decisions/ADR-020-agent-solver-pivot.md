# ADR-020: An agent solver that writes and verifies Python programs; Phase 3 and 4 paused

- Status: Accepted, 2026-10-09 (the human: "let's go with the pivot"). Source:
  `docs/research/leaders_and_base_models_2026.md`, `qwen38_and_small_models.md`,
  `agent_library_design.md`; `docs/STATE.md`, 9 Oct checkpoints.
- Tasks: P6.1 (REPL sandbox, guards, verifier, helper library), P6.2 (engine client, mock engine,
  agent loop, attempt selection), P6.3 (the GPU spike and the Kaggle plumbing), later P6.x
  (scheduler, library growth, hybrid with NVARC, submissions).
- Related: BRIEF A0 (nothing in the pipeline is textual), B2 Q5 (the base model), ADR-018
  (Submission 2), ADR-019 (P3.7, shelved).

## Context

On 9 Oct the public leaderboard's first place was 88.06, prize places started at 58.47, and gold
ended at 38.47; we were at 29.72. Every public notebook at 31.8 or more runs our base model, and
identical NVARC code spreads from about 28 to 34. No public checkpoint beats ours as a grid
transducer, and retraining one is far beyond our quota. The model that moved the 2026 board is a
long-reasoning open model, Qwen3.8-27B (Apache 2.0): 42.4% on the ARC-AGI-2 semi-private set at
XHigh effort, 1.5% with thinking off. Phase 2's loop lost on DEV40, and Phase 3's hypernetwork can
at best make the transducer cheaper inside the same 30 to 34 band.

## Decision

1. **Phase 6, the agent solver.** A local reasoning model writes ``transform(grid)`` in a per-task
   Python REPL; a least-privilege sandbox re-runs each program on every demonstration pair; only
   programs that reproduce every pair and pass the hardcoding guards become attempts; verified
   programs are grouped by test output and ranked by votes, then by size; NVARC or the model's
   direct answers fill the remaining attempt slots. The design follows Prime Intellect's Prime
   Agent (a REPL as the tool, sub-agents as functions, a library the agent extends) in a minimal
   Python harness of our own.
2. **BRIEF A0's "nothing textual" rule is waived for Phase 6 only.** Its prompts, reasoning, and
   programs are text; only verified grids leave it. Everything else keeps the rule.
3. **The library is a measured switch.** Version 1 ships a frozen module of tested numpy helpers;
   growth during a run (the human's cascade of agents into one global memory of reusable blocks)
   is built behind a switch and ships only if it beats the no-library arm by at least 2 of 40
   tasks at equal tokens.
4. **Phase 3 and 4 are paused.** `MT-03` is not pushed; P3.7's builds (PR #38) are shelved; the
   H code stays on `main` for a later return.
5. **NVARC stays.** B2 Q5 still holds for the transducer; Submission 2 (`sub2-r16`) remains a
   candidate for one of the two final selections.
6. **Model:** Qwen3.8-27B, `RedHatAI/Qwen3.8-27B-INT4` (rev `91bd022d`, Apache 2.0) on vLLM with
   tensor parallelism 4; fallback the official FP8 build; a fan-out model
   (`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`) only if the spike shows it solves tasks. The spike's
   measurements decide the build.
7. **Schedule:** the CPU build and the GPU spike run in parallel, not in sequence (the human:
   expedite). The spike's gate: at least about 5 of 20 tasks that NVARC misses solved by verified
   programs, and at least 300 generated tokens per second on 4x L4.
   **Amended 9 Oct (the P6.3 review):** on DEV40, where `P2.6-01` misses 29 tasks, the solve half
   reads as the same 25%: **Good** when verified programs (only programs that reproduce every
   demonstration; direct answers and unverified fallbacks do not count) solve at least 7 of the 29
   and at least 80% of verified programs are right on the test output; **Borderline** at 5 or 6,
   or at a precision of 60% to 80%: Phase 6 continues, but the agent gets only the time NVARC
   leaves; **Bad** at 4 or fewer, only with a healthy engine (engine errors on under 10% of calls,
   no degenerate outputs); **Fault** otherwise: fix and rerun, never read as a failed gate.

## Consequences

- A second inference stack (vLLM wheels for torch 2.13 and CUDA 13, offline) and model weights of
  about 19.5 GB come onto Kaggle through CPU notebooks with internet on, never through the human's
  machine.
- The 12 h budget is split between the agent and NVARC by measured solves per hour; the agent may
  take all of it.
- New risks: the throughput of a 27B model on 4x L4, programs that pass the demonstrations by
  hardcoding (guards), and engine bugs on the hybrid DeltaNet architecture.

## Overturn

If the spike's gate fails, Phase 6 stops at its CPU parts, and the remaining days go to Phase 3's
validation or to NVARC's selection gap (P2.8: 6.17 tasks of 120 lost in selection).

## Amendment 1 (2026-10-26, tasks P4.8 and P4.9): the waiver covers the judge's choice

- **Context.** Decision 2 waives BRIEF A0's "nothing textual" rule for Phase 6 with one limit:
  only verified grids leave the agent. The judge of task P4.8 (`arc26/agent/judge.py`) and its
  submission stage of task P4.9 (`arc26/agent/rerank.py`, variant (1)) prompt Qwen3.8 with a task
  and NVARC's candidates and read a reply, but no grid of the model's own leaves them: the reply
  only chooses among grids NVARC already produced.
- **Decision 8.** The waiver also covers the judge. What leaves it is a number: the reply's
  letter maps to one of NVARC's ranks 2 to 8 through the output's letter order
  (`judge.label_order`, a permutation fixed by the run's seed and the output key), and that rank's
  grid, NVARC's own, becomes attempt 2. The parse (`judge.parse_label`): the last `ANSWER:` line
  counts, else a reply that is one word; exactly one shown letter is a pick (`label`); `UNSURE`
  alone is an abstention (`unsure`); two letters, or a letter with `UNSURE`, are a `tie`; anything
  else is `unparsed`. The fallback (`judge.picked_rank`): every status but `label` (`unsure`,
  `tie`, `unparsed`, an engine `error`, a judgment `cancelled` at the generation stop, one `not
  started`, an output with `no pool`) keeps NVARC's rank 2, so the file holds NVARC's attempts
  unless the judge named one of NVARC's grids. Attempt 1 is never changed. The replies themselves
  stay in the transcripts, written only in development mode (P4.9's `logs.dev_transcripts`) or
  kept on the Modal volume (P4.8-01); every record the stage writes holds ranks, statuses, token
  counts, and seconds, never a grid or a reply.
- **Overturn.** A judge that emits or edits a grid instead of choosing one of NVARC's would need a
  verifier, as the agent's programs do; until then the judge chooses only.
