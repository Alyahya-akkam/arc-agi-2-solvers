# ADR-004: Refinement recomputes the adapter from the enlarged demo set

- Status: Accepted, 2026-09-26. Source: BRIEF B2 Q4.
- Tasks: P2.3 (loop), P2.5 (in-context fallback), P4.3 (overturn test).

## Context

A refinement round promotes confident test predictions to pseudo-pairs and must produce a new
adapter that uses them, with rollback when the held-out gate fails. The same loop must work with
SGD TTT (Phase 2) and with H (Phase 4).

## Decision

The versioned state is the `DemoSet` (real pairs, held-out pair, promoted pseudo-pairs with
provenance). The adapter provider is injected; the loop code is identical in both phases.

- Phase 2 (no H): $\theta_t = \mathrm{sgd\_continue}(\theta_{t-1}, D_t, s')$, a fraction of an
  epoch on the enlarged set.
- Phase 4: $\theta_t = \mathrm{polish}(H(D_t), s)$, recomputed from H, not continued from
  $\theta_{t-1}$.

The in-context variant (append promoted pairs to the decode prompt, no gradient step) is the
fallback the governor selects when the remaining budget cannot cover a gradient round.

## Alternatives considered

- **Continue from $\theta_{t-1}$ in Phase 4.** Lost as the default: H's test inputs would then
  differ from its training inputs (sets without prior adaptation). Measured in P4.3.
- **In-context only.** Lost as the default: no gradient signal from the new pairs; kept as the
  cheap fallback.

## Consequences

- Rollback is a pointer swap: previous adapter and previous `DemoSet` version.
- Recomputing from H keeps H's training and test distributions aligned (and ADR-007 would train
  H on enlarged sets directly).
- The in-context variant needs a context-length guard at the $30 \times 30$ worst case (P2.5).

## Overturn criterion

P4.3 ablation, continue-from-previous versus recompute, on DEV120. If continue wins by at least
2 net tasks (the same bar as P2.6), switch the Phase 4 default.
