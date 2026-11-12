# ADR-003: Test-time adaptation is H followed by a short SGD polish

- Status: Accepted, 2026-09-26. Source: BRIEF B2 Q3.
- Tasks: P4.1 (providers), P4.3 (overturn test).

## Context

The notebook shows full SGD TTT fits the per-worker budget (about 600 s planning number per
puzzle, BRIEF B3). H's value is fewer steps to the same accuracy and budget freed for refinement
rounds and decode augmentations. H may also be mediocre on some puzzle families.

## Decision

Round-0 adapter $= \mathrm{polish}(H(D_0), s)$: the emitted adapter is loaded as a trainable
initialization and trained for $s$ steps on leave-one-out tasks built from the augmented
demonstrations. Start at $s \approx 10\%$ of the notebook's one-epoch step count; the governor
sets the actual budget. Pure H ($s = 0$) is a measured ablation arm, not the default.

## Alternatives considered

- **Pure H, no polish.** Lost as the default: no safety net if H is mediocre, and a polish is
  affordable within the budget. Measured in P4.3.
- **Full SGD from H's initialization.** Lost: spends the budget H was meant to free.

## Consequences

- `HyperPolishProvider` wraps `SGDProvider`; one decode path consumes PEFT adapters from both.
- The polish is a PEFT fit on exported weights, so the P3.1 injection-to-PEFT parity test guards
  the seam between H's functional injection and PEFT.

## Overturn criterion

P4.3 grid over $s \in \{0, 5\%, 10\%, 25\%\}$ of an epoch. If accuracy is flat in $s$, drop the
polish and spend the time on rounds and decode augmentations.
