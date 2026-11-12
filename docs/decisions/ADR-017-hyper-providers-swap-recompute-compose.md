# ADR-017: H's adapter is swapped into the live PEFT adapter and recomputed every round; the polish composes SGDProvider

- Status: Proposed, 2026-10-03, until the DEV4 run of B5 P4.1 reports (it needs an H checkpoint,
  P3.7 and P4.2, and a solver that can choose the provider, after P2.5). Source: task brief
  `docs/tasks/P4.1.md`; BRIEF B5 P4.1, B2 Q3 and Q4, B4 (`AdapterProvider`, the `Adapter`).
- Tasks: P4.1 (the providers); P4.2 (the offline hash check); P4.3 (the cost model's H terms,
  the $s$ grid, continue against recompute).
- Related: ADR-001 (the encoder, 8 views), ADR-002 (H's scope), ADR-003 (H, then the polish),
  ADR-004 (recompute from H; rollback), ADR-013 (the held-out pair reaches no adapter), ADR-015
  (`SGDProvider`, the round's mechanics).

## Context

B5 P4.1: "`HyperProvider.adapt` encodes only pairs not in the cache, runs H, exports a PEFT
adapter; `HyperPolishProvider` loads it as trainable init and runs `SGDProvider` for the governor's
step budget. Acceptance: `refine()` integration test passes with all three providers on DEV4;
emitted adapter hashes are deterministic for a fixed `DemoSet`." B2 Q3 makes round 0's adapter
$\mathrm{polish}(H(D_0), s)$ with $s$ about 10% of the notebook's epoch; B2 Q4 makes round $t$'s
$\mathrm{polish}(H(D_t), s)$, recomputed from H. P2.3 built the round around an injected provider
whose adapter is the live PEFT model (`SGDProvider` trains it in place and restores a CPU copy on
rollback), and the solver's decode callables pass that adapter to `decode_puzzle` as its model.
Open were: what an H provider returns, how H's adapter reaches the model, how the encoder sees the
frozen base while the live model holds an adapter, what makes a hash deterministic, how the polish
starts from H's adapter (with or without a change to `arc26/ttt.py`), and what the governor prices.

## Decision

1. **The adapter a provider returns is the live PEFT model**, as `SGDProvider`'s is, so `refine`,
   `run_rounds`, their decode callables, and `decode_puzzle` run unchanged: one decode path for
   every provider (B4). H's output is B4's `Adapter`: the PEFT-keyed float32 tensors of P3.1's
   export plus `{rank, alpha, target_modules}`; `Adapter.save` writes `adapter_model.safetensors`
   and `adapter_config.json` (`inject.save_adapter`'s layout). `adapt` copies it into the live
   model's existing `default` adapter with `set_peft_model_state_dict` (P3.1's parity loader, no
   files), after checking that the live adapter has H's scope: rank, alpha, rsLoRA, targets, and
   every tensor key and shape; otherwise it raises before any change.
2. **Recomputed from H every round** (B2 Q4, ADR-004). `init`'s weights are copied to the CPU for
   the rollback and are never an input to H; the polish starts from H's adapter, not from `init`.
3. **Rollback restores the CPU copy** (a state-dict swap, ADR-004's pointer swap made physical).
   `fingerprint` is a sha256 over each tensor's key, dtype, shape, and raw bytes in sorted key
   order, of a live model's adapter or of an `Adapter`, so equal digests mean bit-for-bit equal
   tensors, and a live float32 adapter's digest equals the emitted one's.
4. **The encoder reads the frozen base; a pair's embedding depends on the pair alone.** The encoder
   is built on the live PEFT model and runs inside its `disable_adapter`, which gives the base's
   outputs bit for bit, as P3.5 encoded. Only the pairs the cache lacks are encoded, each in its own
   encode call, so its prefill batches hold its own views only. $\bar z$ is the mean of the cached
   embeddings in pair-hash order, repeats counted as in training. Only `DemoSet.pairs` is read, so
   the held-out pair cannot reach the encoder or H (ADR-013). Order dependence found: the float32
   mean's summation order ($9.5 \times 10^{-7}$ on the tiny model), removed by the hash order;
   and, on GPU, a pair's batch partners, removed by the per-pair calls. None is left on one device
   and stack.
5. **The polish composes the two providers; `arc26/ttt.py` is unchanged.** `HyperPolishProvider`
   runs `HyperProvider.adapt` (CPU copy of `init`, H's adapter loaded), then `SGDProvider.adapt`,
   whose CPU copy of H's adapter becomes `fit`'s `default_weights` with `max_steps` $= s$:
   `SGDProvider` starts from a given adapter as it stands once the adapter is in the model, so the
   brief's item 4 (an addition to `ttt.py`) is not needed. `rollback` undoes the two in reverse:
   `SGDProvider.rollback` (H's adapter back, inference mode, the trainer's memory released), then
   `HyperProvider.rollback` (the round's starting weights); the first runs only when the last
   `adapt` reached the polish, since an `SGDProvider` that did not run keeps an earlier round's
   copy. $s$ is the `SGDProvider`'s `steps`, `POLISH_STEPS` $= 13 = \lceil 128/10 \rceil$ by
   default (B2 Q3). The polish trains on `fit`'s samples, the pairs under its 128 views, as
   `SGDProvider` does: the reference has no leave-one-out task generation (ADR-009, ADR-011), so
   B2 Q3's "leave-one-out tasks" read as those samples. The governor accepts or declines the round
   at $s$'s price; choosing $s$ from the budget is P4.3's.
6. **`fit_shape` stays the train term; the encode is exposed apart.** `HyperProvider.fit_shape` is
   $(0, 0)$, which `CostModel` prices at 0 s; the polish's is `SGDProvider`'s
   $(\lceil T s / 128 \rceil, s)$. `encode_shape(demos)` returns the cache misses, their prefills,
   and their real tokens, rendered and tokenized on the CPU without the model; H's forward is one
   call per `adapt`. P4.3 adds the encode (seconds per prefill token; `P3.2-02`: 0.181 s per 1,000)
   and H's forward (a constant) to `CostModel` and to `refine`'s plan; `arc26/governor.py` and
   `arc26/loop.py` are not edited here.
7. **H and the encoder's settings come from the checkpoint.** `HyperProvider.from_checkpoint`
   reads H's config and weights, the encoder's layer, pool, views (`n_perm`, `view_seed`), and
   batch, and the cache's `model_id` from P3.5's checkpoint, so test time encodes as training did.
8. **Timing ops.** `encode` (seconds, prefill tokens) and `h_forward` (P3.5's names for the same
   work), `snapshot`, `swap`, `restore`, `fingerprint`; the polish adds `SGDProvider`'s
   `snapshot` and `candidate_train`.

## Alternatives considered

- **`adapt` returns the `Adapter`, and the decode callables load it.** Lost: it edits the
  callables in `arc26/loop.py` (P2.5's file) and gives a second decode path. The `Adapter` stays
  available through `emit`.
- **A second copy of the base for the encoder.** Lost: about 8 GB more GPU memory per worker;
  `disable_adapter` gives the base bit for bit.
- **All misses in one encode call** (`P3.2-02`'s batching). Lost: on GPU a pair's batch partners
  can change its embedding's bits, so a `DemoSet`'s hash would depend on the cache's history
  (which pairs earlier rounds or puzzles encoded). Per-pair calls cost a few more, smaller prefill
  batches.
- **$\bar z$ in `DemoSet.pairs` order.** Lost: its bits depend on the order.
- **A start-weights parameter on `SGDProvider.adapt`** (an addition to `ttt.py`). Lost: the
  composition needs no change; its extra CPU copy of H's adapter is 24 MB in bf16 at H's scope.
- **Price H through `fit_shape`'s tokens.** Lost: the governor's train model, fitted on 128-step
  fits, would price prefill tokens as training tokens.
- **Continue from $\theta_{t-1}$ in Phase 4.** ADR-004's overturn test (P4.3), not the default.

## Consequences

- The H arms need the solver's live adapter at H's scope: `lora_r` 16, `lora_targets` q, k, v, o
  (P1.3's arm settings), alpha 32 and rsLoRA as `arc26.solver.load` sets them. Another scope is
  refused.
- bf16: `arc26.solver.load` casts every float32 parameter to bf16, the adapter's included, so H's
  float32 export is rounded at the swap and the LoRA delta runs in bf16, while P3.5 trained H with
  float32 factors and a float32 delta (P3.1's parity). P3.1 measured logit moves of
  $7.0 \times 10^{-2}$ to $9.4 \times 10^{-2}$ for such rounding on its tiny model. On the CPU,
  under the same cast, the swap is exactly the bf16 rounding of the export (so the live digest is
  not the emitted one's), the rollbacks stay bit for bit, and zero-initialized H keeps the base's
  logits (`tests/test_providers.py`). The effect on a trained H's logits at Qwen3-4B is
  unmeasured: the DEV4 run measures it, or the H arms keep the adapter in float32 (a change to
  `solver.load`'s cast, P2.5's file now).
- The encoder at test time runs unsloth's patched model; P3.5 and P3.2's cache ran transformers'
  (SDPA or FlashAttention 2). $\bar z$'s agreement is unmeasured.
- Per worker: H on the GPU (135.0M parameters, 540 MB in float32), the CPU copy (11.8M values,
  24 MB in bf16), a cache root of its own (`EmbeddingCache` supports one writer), about 10 KB per
  pair on disk.
- A round of H costs the pseudo-pairs' encode, about 0.8 s for a median pair and 2.7 s for a
  $30 \times 30$ one (estimate: `P3.2-02`'s rate at 8 views), and one forward (4.6 ms on an
  L4, `P3.5-01`), unpriced until P4.3; the margin covers it. The polish costs $s = 13$ of the 128 views,
  priced by `fit_shape`.
- B5 P4.1's acceptance holds on the CPU (`tests/test_providers.py`: `refine` with all three
  providers on DEV4-shaped stand-ins, deterministic hashes); the DEV4 GPU run is pending. The PR
  adds `or providers` to the `-k` filter of CI's `hyper-cpu` job (`.github/workflows/ci.yml`), so
  that job runs the module; `.github/` is a protected path, so the PR is labeled `needs-human` and
  the human merges it.

## Overturn criterion

- bf16: if the DEV4 run's candidates or gate verdicts differ between the bf16 live adapter and a
  float32 one, for H or for the polish, the H arms keep the adapter in float32.
- Per-pair encode: if the DEV4 run's `encode` records exceed `P3.2-02`'s rate (0.181 s per 1,000
  tokens) by more than 50%, the misses are encoded in one call, sorted by hash, and hashes are
  compared only with a warm cache.
- Determinism: if two emissions for one `DemoSet` on one GPU differ (P4.2's offline hash check),
  H's forward runs under `torch.use_deterministic_algorithms` before hashes are compared.
- Recompute against continue: ADR-004's P4.3 ablation.
