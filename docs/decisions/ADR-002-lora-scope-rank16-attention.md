# ADR-002: Generated LoRA scope is rank 16 on attention projections, both factors

- Status: Accepted, 2026-09-26; confirmed at gate G2 by the human, 2026-10-02. Source: BRIEF B2
  Q2.
- Tasks: P1.3 (gate measurement), P3.3 (build).

## Context

The notebook's TTT fits a rank-256 adapter on all modules by SGD. H must *emit* an adapter from
tens of thousands of training puzzles, and its output dimension is the generalization
bottleneck: every emitted value is a regression target learned from the same number of tasks.

## Decision

H emits $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$
with $r = 16$ for `q_proj`, `k_proj`, `v_proj`, `o_proj` on all 36 layers: about 11.8M values per
adapter. Scaling is rsLoRA, $\alpha / \sqrt{r}$ with $\alpha = 32$ as in the notebook. The $B$
head is zero-initialized, so H at initialization reproduces the base model exactly. Embeddings
and `lm_head` are excluded from H and stay frozen.

Expansion order if the gate fails: (1) add `gate_proj`, `up_proj`, `down_proj` (about 33M values),
(2) $r = 32$.

## Alternatives considered

- **Notebook scope (rank 256, all modules).** Lost: orders of magnitude more outputs than H can
  learn from the available puzzles; it is what SGD can use, not what H can emit.
- **Emit only $B$ with a frozen shared $A$.** Lost as the default: halves outputs but constrains
  the subspace per layer. Reconsidered only if H training is unstable.
- **Include embeddings or `lm_head`.** Lost: 16-token vocabulary, tiny benefit, and they change
  the output distribution directly rather than the computation.

## Consequences

- T2L's released hypernetworks use the same pattern (low-rank, attention only), so its head
  design transfers.
- The SGD polish (ADR-003) runs at the same scope, so P1.3's SGD numbers at this scope are the
  ceiling H plus polish is compared against in P3.7.

## Overturn criterion

Gate rule at G2: P1.3 measures SGD TTT at each scope on DEV40. H's scope is the smallest scope
within 2 points of the notebook configuration. If rank-16 attention-only loses more than that,
expand per the order above before building H's final heads.

Open point for G2: on DEV40 one task is 2.5 points, so "within 2 points" means zero tasks lost,
which is stricter than the DEV40 noise floor of 2 tasks. Flagged in `docs/STATE.md` for the
human to confirm or restate at G2.

## Gate G2 (2026-10-02)

Run `P1.3-01` ([`docs/experiments/P1.3-01.md`](../experiments/P1.3-01.md); DEV40, 4xL4, one seed)
measured SGD TTT at five scopes. The notebook configuration (rank 256, all seven projections,
embeddings and `lm_head` trained) scored $S = 8.5$ of 40 tasks (21.25%); rank 16 on q, k, v, o
scored 10.5 of 40 (26.25%), $+2.0$ tasks, the best of the five. That is within the rule under every
reading of "within 2 points" (zero tasks lost, at most one task lost, or literal points), so the
reading question (`docs/STATE.md`, conflicts 1 and 7) does not bear on this gate. The human
confirmed the decision on 2026-10-02: H's scope stays rank 16 on q, k, v, o, and the expansion
order above is not used. The $+2.0$ is at the DEV40 noise floor of 2 tasks, so it shows that the
smaller scope loses nothing, not that it is better.
