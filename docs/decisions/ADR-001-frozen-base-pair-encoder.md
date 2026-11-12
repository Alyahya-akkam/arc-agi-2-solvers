# ADR-001: Pair encoder is the frozen base model with pooled hidden states

- Status: Accepted, 2026-09-26; amended 2026-10-02 (8 views, measured cost; runs `P3.2-01` and
  `P3.2-02`); amended 2026-10-17 (set aggregator as an option, task P3.3b). Source: BRIEF B2 Q1.
- Tasks: P3.2 (build), P3.7 (overturn test).

## Context

H needs a fixed-size task embedding $z$ computed from a variable number of demonstration pairs,
invariant to pair order, cheap at test time, and informative about the *transformation* rather
than the grids. The only grid-aware network available offline is the grid-SFT base model
`qwen3_4b_grids15_sft139`.

## Decision

Render each augmented demonstration pair in the notebook's exact token format and run one
teacher-forced prefill through the frozen base. Take layer-$L$ hidden states and mean-pool over
the output-grid token positions to get a pair embedding $e_i \in \mathbb{R}^{2560}$. The task
embedding is

$$z = \mathrm{MLP}\!\left(\frac{1}{n}\sum_{i=1}^{n} e_i\right)$$

over all pairs and all their augmentations. Start at $L = 24$ of 36. The layer and the pooling
mask are parameters.

**Views (amended 2026-10-02):** 8 augmentations per pair, `view_descriptors(1, 2)`: one color
permutation for each of the 8 dihedral transforms, the first round of the decode path's 16. At 16
views `P3.2-01` failed B5's 10 s bound on 11 of 40 DEV40 tasks; at 8 views `P3.2-02` passed on all
40, and the 8-view pair embedding stays within cosine 0.9983 of the 16-view one (median 0.9996;
the task embedding within 0.9992) while different pairs sit at a median cosine of 0.55.

## Alternatives considered

- **Raster encoder over one-hot grids, trained jointly with H.** Lost: starts without any grid
  prior and adds a second model to train in five weeks. Kept as the fallback.
- **Attention pooling (learned query over pair tokens) or a set-transformer aggregator.** Lost as
  the default because mean pooling is permutation invariant by construction and has no
  parameters to overfit; kept as ablations. The set aggregator is now an option of H (amendment
  of 2026-10-17 above), default still the mean.
- **D2L-style per-token context read.** Lost: longer H inputs for 9.5k-token contexts, and no
  measured need for token-level detail yet.

**Set aggregator (amended 2026-10-17, P3.3b):** the mean stays the default and the input of every
H trained so far. A Set Transformer over the pair embeddings is added as an option
(`aggregator: set` in `configs/hypernet.yaml`; `docs/modules/hypernet.md`, "Set aggregator"): it
adds a zero-initialized term to H's task code per (module, layer) row, so it starts exactly at a
trained mean H and can be warm-started from one (`--init-from`). Why now: `P3.2-03` read Keep (the
layer-24 output-pooled embedding carries the rule), so what is left to gain is what the mean
discards, weighting and comparing pairs (scoping note `docs/research/h_upgrades_scoping.md`,
section 1). It becomes the default only after a pilot (`P3.3-01`) reads Good.

## Consequences

- Embeddings are cacheable per (pair, augmentation), so meta-training and the refinement loop
  only encode new pairs.
- Test-time cost (measured, `P3.2-02`, one L4, bf16, SDPA): $k \times 8$ prefills of at most
  1,866 tokens, $t_\text{prefill} \approx 0.54 + 0.15\,T/1000$ s for $T$ tokens; for $k = 4$ over
  DEV40 a median of 3.46 s, $p_{90}$ 6.77 s, maximum 9.73 s (P3.2 acceptance, at the bound on the
  largest grid). Encoding a pair for the cache costs about 1 GPU-second at 16 views (`P3.5-01`),
  half at 8: for the full meta-training split, about 200k pairs, roughly 28 GPU-hours.
- H depends on the base's internal representation at layer $L$; changing the base (ruled out by
  ADR-005) would invalidate the cache and H.

## Overturn criterion

In P3.7, if a $z$ pooled over *input*-grid positions, or taken from a random layer, performs the
same as the chosen $z$, the encoder carries no transformation information: switch to the raster
encoder trained jointly with H.
