# ADR-018: Submission 2 without the loop: rank 16 on q, k, v, o, every pair trained, the split solver

- Status: Proposed, 2026-10-03, until run `P2.6-01` (DEV120 in development mode on 4xL4) reports
  against `P1.1-01`; the brief's rows say what each outcome decides. Source: task brief
  `docs/tasks/P2.6.md`; BRIEF B5 P2.6, B2 Q2, B7 G3.
- Tasks: P2.6 (Submission 2, run `P2.6-01`); P2.7 and P4.4, which compare against it.
- Related: ADR-002 (the scope, confirmed at G2), ADR-010 (packaging), ADR-011 (the split solver),
  ADR-013, ADR-015, ADR-016 (the loop that stays out).

## Context

B5 P2.6 is "Full run and submission 2", acceptance "at least 2 net tasks gained on DEV120 over
P1.1 at the same seed, zero governor violations"; A5's milestone `sub2-loop` names the loop
(gate, promotion, refinement, governor on SGD TTT). What the runs since then measured:

- **The loop loses.** `P2.5-02` (DEV40 on 4xL4, rank 16 on q, k, v, o, a pair held out, one
  gradient round, the final decode with the pair back) scored 7.0 of 40 against 10.5 for the same
  scope with every pair (`P1.3-01` arm 2): its pre-registered Loss row. Its report located the
  loss: all four lost outputs were submitted from the final decodes, whose prompts equal arm 2's
  token for token but whose adapter is round 0's, fit without the held-out pair (43 correct
  candidates against 63). The round changed no output: 5 of 55 test inputs reached
  $\tau = 9/16$, and no DEV40 puzzle had an open input beside a promoted one, so the loop as
  built has no path to a gain there. Its row keeps the loop out of P2.6 until a fix is measured;
  the fix it names (a last fit with the pair) would at best recover arm 2's 10.5 at more than
  twice its time.
- **The measured alternative is a scope.** `P1.3-01` arm 2 (rank 16 on q, k, v, o, `embed_tokens`
  and `lm_head` frozen, every pair trained) scored 10.5 of 40 against 8.5 for the notebook
  configuration, at 490 s per puzzle against 606 s; G2 confirmed it as H's scope (ADR-002).
  Three runs of the notebook configuration on these tasks gave 6.5, 8.5, and 9.5 (`P2.1-03`), so
  the DEV40 gain is at the noise floor; every smaller scope gained the same two outputs.
- **Time.** On a 240-task hidden set (2025's; 2026 unverified) the notebook configuration
  projects $u = 0.909$ (`P1.1-01`), at G1's 90% line; rank 16 on q, k, v, o projects about 0.74
  (estimate, `docs/prerun/P2.6-01.md`, section 6).
- **The builder could not submit a scope.** A build ran the reference's LoRA settings; other
  scopes existed only as arms of a series, which a competition rerun ignores and
  `submission_blocker` refuses.
- **The solver.** `P2.1-03` ran the notebook configuration on the split solver on DEV40: every
  input equal to the moved reference's (training tokens, decode views and prompts, rescoring
  tokens, memory, environment), its score within the reference's run-to-run spread (ADR-011's
  flag cleared).

## Decision

1. **Submission 2 is one run at rank 16 on q, k, v, o.** rsLoRA with the reference's
   $\alpha = 32$ (scale $\alpha / \sqrt{r} = 8$), `embed_tokens` and `lm_head` frozen, every
   demonstration pair trained: no held-out pair, gate, round, or governor. Everything else is
   the reference's (one epoch of 128 training views, 16 decoding views, 8 rescoring views, the
   1200 s decode cap, `score_kgmon`), on the split solver (`solver: split`, ADR-011), with
   `P1.1-01`'s production settings: four workers on `NvidiaL4`, `PYTHONHASHSEED` 0, the
   competition deadlines.
2. **The scope applies in both modes.** A submission config may set `scope` (`lora_r`,
   `lora_targets`), checked as an arm's scope is; the Config cell holds it as `SCOPE`, and the Run
   cell passes it to `arc26.runner.RunConfig`, which writes it into the solver's `--config` JSON,
   so a competition rerun of the submitted version trains exactly what its development run
   trained. A scoped build is fit to submit (`submission_blocker` does not read the scope); a
   scope beside arms is refused (a rerun would train a scope no development run of the build
   trained), and so is the reference's own scope (leaving the key out builds exactly that). A
   build without the key is byte for byte as before (`tests/test_scope_build.py`, golden
   digests of every committed build config). The first cell, the environment check, and
   `MANIFEST.md` state the scope.
3. **The loop stays out of production** until a fix is measured (overturn criterion). Its code
   stays: hold-out and round arms remain series settings, development runs only.
4. **Run `P2.6-01` measures this build** on DEV120 against `P1.1-01`'s 37.17 of 120 at the same
   seed, on its own code dataset `alyahya737/arc26-code-p26`; the human submits that notebook
   version per the brief's rows.

## Alternatives considered

- **Submit the loop as built** (`P2.5-02`'s configuration). Lost: its Loss row ($-3.5$ tasks on
  DEV40, robust outputs among them: `de809cff_0`, `eee78d87_0`), and 1,041 s per puzzle against
  490 s, which on 240 tasks projects $u \approx 1.5$ (`P1.1-01`'s projection with this mean):
  the rerun would leave tasks unsolved.
- **The loop with its fix** (the final decode with an adapter fit on every real pair). Lost for
  Submission 2: unmeasured; for a puzzle with nothing to promote it is this build plus the
  hold-out's fit, decode, and gate, so its best case on DEV40 is arm 2's 10.5 at more than twice
  the time, and DEV40 offers the round no open input to gain on.
- **The in-context round** (`P2.5-01`). Lost: its round 0 and final decode are `P2.5-02`'s, which
  carry the hold-out's price; its rounds face the same promotion screen.
- **The notebook configuration** (rank 256 on all nine target modules). Lost: it is Submission
  1's (`P1.1-01`); submitting it again measures nothing new, and it is the slower scope ($u =
  0.909$ on 240 tasks).
- **Another smaller scope of `P1.3-01`** (r32-attn 10.0, r16-proj 10.0, r64-proj 8.0 of 40). Lost:
  within arm 2's noise and slower per puzzle (497, 567, 574 s); rank 16 on q, k, v, o is also H's
  scope, so P4.4 (H plus polish) compares with this build at the same scope.
- **The moved reference solver** (`nvarc`). Either may submit (ADR-011); lost because the split
  solver is the default since 2 Oct and the code Phases 2 and 4 extend, so Submission 2 runs
  what they build on.
- **The scope as a one-arm series.** Lost: a competition rerun ignores a series and runs the
  reference's settings, so the submitted run would not be the one measured (and a series is
  never fit to submit).
- **The first arm of a series in a competition rerun.** Lost: it changes what every series build
  reruns, and their bytes; a series is built to compare arms.
- **A new default in `arc26.runner.RunConfig`.** Lost: it changes every build's solver, the P0.1
  parity runs' among them, and moves a run setting into the package.

## Consequences

- Submission 2 measures rank 16 on q, k, v, o on the hidden set; P4.5 chooses among Submission
  1 (`P1.2`), P2.6, and P4.4 by DEV120 first, the leaderboard second.
- P4.4 (H at this scope plus polish) has its SGD baseline on DEV120 at the same scope.
- The milestone holds no loop: G3 ("P2.6 shows the lift") reads `P2.6-01`'s row as the scope's
  lift. Whether the loop returns is the overturn criterion's question; until then Phase 4 targets
  round 0 (B7 G3's no-lift branch).
- The time the smaller scope frees ($u \approx 0.74$ against 0.91 on 240 tasks, estimate) stays
  headroom: more decode views or steps would each be a change to measure first.
- This build has a code dataset of its own, `alyahya737/arc26-code-p26`: no queued run of another
  branch can make it wait for a version, and the competition rerun reads the version attached to
  the submitted notebook version.
- The builder now has a build setting, besides `solver`, that reaches a competition rerun; any
  later one must be tested in competition mode on the simulated tree, as
  `tests/test_scope_build.py` tests this one.

## Overturn criterion

A measured fix of the loop on DEV40 or DEV120 that beats this build: on DEV40, a loop arm at least
2 tasks above a no-loop arm at this scope run in the same session, with outputs that have several
correct candidates among those gained (the 3-task spread of three runs of one configuration,
`P2.1-03`, rules out single-run comparisons near the floor); or on DEV120, a loop build at least 2
tasks above `P2.6-01`; in either case with a projected $u \le 0.9$ on 240 tasks. A later
submission then runs the loop. Separately, `P2.6-01`'s Loss row (at least 2 tasks under
`P1.1-01`, robust outputs lost) rejects decision 1's scope before anything is submitted.
