# ADR-006: Meta-train H on end-to-end task loss, optional function-space distillation

- Status: Accepted, 2026-09-26. Source: BRIEF B2 Q6.
- Tasks: P3.4 (episodes), P3.5 (training loop), P3.7 (validation), P3.9 (amendment 2).

## Context

H can be trained to reconstruct per-puzzle SGD adapters (weight space), to match an in-context
teacher (D2L's context distillation), or to minimize the task loss directly. ARC scores come from
TTT *beyond* in-context behavior, so matching the in-context model is a floor, not a target.

## Decision

An episode is (demos $D$, queries $Q$), where $Q$ is one left-out pair of the puzzle under
augmentations and $D$ is the rest. The loss is

$$\mathcal{L}_{\text{task}} = -\sum_{(x,y) \in Q} \log p\!\left(y \mid x, D;\ \theta_{\text{base}} + \Delta\theta_H(D)\right)$$

with teacher forcing over output-grid tokens and gradients flowing through the frozen base into
H. Demos stay in context at query time (the notebook's format), so the adapter is a delta on top
of in-context behavior, as in the notebook's TTT.

Optional, behind a flag, only if quota allows after P3.7: add
$\lambda \cdot \mathrm{KL}\!\left(p_{\text{SGD}} \,\|\, p_H\right)$ on output positions, with
$p_{\text{SGD}}$ from cached logits of a notebook-fit adapter. With a 16-token vocabulary a cached
position costs 32 bytes in bf16, a few MB per puzzle.

## Alternatives considered

- **Weight-space reconstruction of SGD adapters.** Lost on three counts. Identifiability:
  $BA = (BQ)(Q^{-1}A)$ for any invertible $Q \in \mathbb{R}^{r \times r}$, so MSE on factors is
  ill-posed without fixing the gauge (a frozen shared $A$, or a loss on the product
  $\|B_H A_H - B^* A^*\|_F^2$ computed through $r \times r$ Gram matrices). Cost: one target
  adapter costs about as much as several hundred end-to-end steps, and H generalizes with the
  number of distinct puzzles seen, not reuse of a fixed target set. Evidence: T2L reports its
  end-to-end (SFT) variant generalizing better than its reconstruction variant.
- **D2L context distillation.** Lost: its target is the in-context model, which the notebook's
  TTT already beats.

## Consequences

- No adapter storage for targets; the training set is the episode stream from ReARC, BARC, and
  the public ARC training sets, split by task ID.
- Backpropagating through the frozen 4B base at up to 9.5k tokens needs gradient checkpointing
  and FlashAttention; P3.6 tests the worst case before any long run.
- The KL term is the first thing cut if behind (BRIEF A7 cut order).
- Token set (amendment, 1 Oct, P3.5): the loss covers the query's output-grid tokens plus the
  closing `<|im_end|>` (`loss_includes_eos`, default true), as the notebook's completion-only
  test-time training does; H's adapter serves the notebook's decode, where that token ends a
  grid. `docs/modules/train.md` records the reading.
- Objective (amendment 2, 18 Oct, P3.9): beside the loss above (`objective: zero_step`, the
  default, unchanged), `objective: fomaml` trains H for the start of the fine-tuning. Per episode,
  $\theta_0 = H(\bar z)$; $k$ inner steps on the demonstrations' views,
  $\theta_{j+1} = \theta_j - \alpha u_j$ ($u_j$ Adam's bias-corrected step with fresh moments,
  or the plain gradient), base frozen, H untouched; the outer loss is
  $\mathcal L_Q(\theta_k)$, and H receives $J_\phi^\top \nabla_\theta \mathcal L_Q(\theta_k)$
  (first-order MAML). Why: `P3.7-03` (H's start tied the fit from zero, 10.5 of 40 each on DEV40,
  and ended at a higher training loss on 38 of 40 puzzles): a start trained for the loss with no
  steps is a bias the fit works against, and the survey
  (`docs/research/h_architecture_survey.md`, sections 2, 9.4, 11a) ranks a generator trained for
  the post-adaptation loss (MMAML, LEOPARD, FOMAML) as the supported way to make a generated start
  help. Validation adds decode-aligned metrics (the share under the decode's $\ln 5$ bound,
  greedy exact match), since `P3.7-01` showed mean CE does not track solving across adapters;
  under `fomaml` the best point is chosen on the post-$k$ exact match. $k = 0$ reproduces
  `zero_step` bit for bit. Exact second-order MAML and Reptile were not chosen: the first needs
  Hessian-vector products through the 4B base, the second gives no query loss to select on.
  `docs/modules/train.md` holds the mechanics; the pilot is `P3.9-01`. A new objective is a new
  training phase: with `fresh_schedule`, `--init-from` loads H's weights and starts fresh AdamW
  moments and a new warmup and cosine (orchestrator, 18 Oct). The objective keys are resume
  keys, absent read as `zero_step`. Review fixes (18 Oct, P3.9): each inner gradient is clipped
  at the fit's global norm 1.0 (`max_grad_norm` in `arc26/solver.py`), since from Adam's second
  step on the clip factors weigh the gradients; the inner rate is not fixed in advance but picked
  at the warm start's start step by a pre-registered rule over a few probed rates
  (`inner_probe`: the most post-$k$ exact matches within 2 episodes, then the lowest mean $S$),
  because Adam's first step moves every value by the full rate whatever the gradient's scale;
  `best.pt` carries the config without the objective keys (`h_config`) so that a loader on the
  code before P3.9 parses it.

## Overturn criterion

P3.7: if H closes less than 25% of the gap between the zero adapter and the SGD adapter (gate
G4), stop Phase 4 and submit the Phase 2 system. If training loss falls but validation episode
CE does not beat the zero adapter within 2k steps on the toy set (P3.5), revisit the objective
before a long run.
