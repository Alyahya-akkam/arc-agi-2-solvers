# Hypernetwork design: what we take from D2L and T2L

Task P3.1 (BRIEF B5, B1 item 1, B8). Evidence, file and line references, and parameter counts are
in the [research note](research/d2l_t2l_reading.md) ("note" below); decisions are ADR-001
(encoder), ADR-002 (scope), ADR-006 (objective). This page records the choices, not the evidence.

## Reused

| Area | Source | What we take | Where it lands |
| --- | --- | --- | --- |
| Injection | D2L (note 1.1) | externally supplied $(A, B)$ added to frozen target `Linear`s, differentiable in the factors | `arc26/hyper/inject.py` (P3.1), but with PEFT's op sequence instead of D2L's einsum ([module doc](modules/inject.md)) |
| Batched hypernetwork | T2L (note 2); D2L (note 1.2) | T2L: one head per module type, evaluated batched over the 36 layers. D2L: a leading per-context axis on the factors for distinct adapters in one batch (unpacked path only) | P3.3 (head), P3.5 (batched distinct adapters) |
| Head design | T2L (note 2) | input concat($z$, layer embedding, module embedding), MLP trunk, per-module-type linear heads emitting $r(d_\text{in} + d_\text{out})$, bias-HyperInit or zero $B$ at step 0; D2L's L2 norm of the head input and zero-initialized $B$ gate as options (note 4) | P3.3, head width $H_\text{in} \le 384$ for the 150M budget (note 2, table) |
| Training loop | D2L (note 1.4) | cross-entropy trainer structure, optional L1 (D2L) or L2 (T2L) penalty on factors, AdamW with cosine schedule, minimum lr, warmup, clipping, no decay on norms, biases, scalers | P3.5, lr and warmup retuned |

No D2L or T2L code is copied in P3.1. Copies in P3.3 or P3.5 keep the license header the note
recommends (note 5).

## Changed

- **16-token vocabulary.** The base reads and writes grid tokens only, so D2L's text data
  pipeline, chat templates, and vLLM self-generation do not transfer (note 4, "Data and
  tokenizer"). Embeddings and `lm_head` stay frozen and outside H (ADR-002). NEFTune is off: noise
  on 16 grid-token embeddings has no validated benefit.
- **Objective.** Task cross-entropy on output-grid tokens with the demonstrations in context
  (ADR-006), not D2L's top-16 KL to a context-in teacher. With 16 tokens, an optional KL term
  uses the full distribution (note 4, "Objective").
- **Encoder.** Mean-pooled layer-24 hidden states of the output-grid positions from the same
  frozen base (ADR-001), cached per pair and augmentation: no second model copy, no Perceiver,
  no per-token read (note 4, "Encoder").
- **Scope and scale.** $r = 16$ on `q_proj`, `k_proj`, `v_proj`, `o_proj`, rsLoRA
  $s = \alpha/\sqrt{r} = 8$ as one constant shared by injection and export (ADR-002;
  `rslora_scaling`). Not D2L's $s = 2r^{3/2}$, and no rank growth across context chunks
  (note 1.1).
- **Memory.** Non-reentrant activation checkpointing per decoder layer, with the factors held in
  place until `backward` returns (note 6, risks 2 and 3). Neither repo checkpoints.

## How D2L reads its context

Confirmed from code, not the paper (note 1.3):

- A **second, frozen instance** of the base (optionally 4-bit) encodes the context; the modulated
  model never sees the context tokens, which reach it only through the adapter.
- Features are **per token, not pooled**: early exit at layer $\lfloor L/4 \rfloor$ after the final
  norm, or one feature map per target layer in the main experiments.
- A **trained Perceiver** resamples them into latents and emits one query vector per
  (layer, module, rank); per-(layer, rank) scalers start $B$ at zero.

Ours differs on each point: one loaded base serves as encoder and target, features are pooled per
pair (ADR-001), and the demonstrations stay in the query context (ADR-006), so the adapter is a
delta on top of in-context behavior.
