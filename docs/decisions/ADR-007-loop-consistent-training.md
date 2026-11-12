# ADR-007: Loop-consistent training of H (optional, dropped by default)

- Status: Accepted as optional; dropped by default per BRIEF A7. 2026-09-26. Source: BRIEF B2 Q7.
- Tasks: P3.8.

## Context

In Phase 4, H receives demo sets that contain promoted pseudo-pairs (ADR-004). If H only ever
trained on clean demo sets, round 1 inputs are out of distribution for it.

## Decision

After $\mathcal{L}_{\text{task}}$ (ADR-006) converges, continue training on episodes whose $D$
contains model-generated pseudo-pairs that passed the same promotion filter used at test time
(`loop.promote`), so H sees Phase 4 inputs during training.

Scheduled for week 5 only if the project is a full week ahead at G3; first item in the A7 cut
order.

## Alternatives considered

- **No loop-consistent stage.** The default under the compressed schedule; H relies on the
  polish to absorb the distribution shift.
- **Mix pseudo-pairs from the start of meta-training.** Lost: the promotion filter needs a
  trained H or SGD adapter to generate pseudo-pairs, so it cannot run at step 0.

## Consequences

- P3.8 depends on P3.7 and P2.2 and costs extra decode passes to build episodes.
- If dropped, round-1 accuracy in P4.3 is the measurement of how much the shift costs.

## Overturn criterion

Reinstated if at G3 the project is a full week ahead. Kept only if the P3.7 table repeated after
P3.8 shows no regression on H alone and an improvement on H plus polish with one refinement
round.
