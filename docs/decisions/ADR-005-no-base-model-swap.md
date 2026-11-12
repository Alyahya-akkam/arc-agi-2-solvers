# ADR-005: Keep `qwen3_4b_grids15_sft139` as the base model for the whole 2026 entry

- Status: Accepted, 2026-09-26. Source: BRIEF B2 Q5.
- Tasks: all.

## Context

The handover listed a tokenizer redesign and a base swap as options. The entry has five weeks
left (BRIEF A7) and 2 to 3 human hours per week, and 2026 is the final year of this track.

## Decision

No base swap. The handover's constraint 4 (tokenizer redesign) leaves the plan.

## Alternatives considered

- **Reproduce the grid SFT on a different or larger base offline.** Lost: does not fit the
  schedule or the human time; the score is expected to come from H and the loop, not the base.
- **Tokenizer redesign.** Lost for the same reason, and it would break the parity fixture and
  every cached embedding.

## Consequences

- The notebook's tokenization, augmentation, and decode are reused unchanged; the parity fixture
  protects them.
- The pair encoder (ADR-001) and H are tied to this base's hidden size (2560) and depth (36).

## Overturn criterion

None within the 2026 entry.
