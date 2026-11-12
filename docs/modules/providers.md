# `arc26.hyper.providers`: adapter providers from H

Task P4.1 (BRIEF B5 P4.1, B2 Q3 and Q4, B4), [ADR-017](../decisions/ADR-017-hyper-providers-swap-recompute-compose.md);
builds on ADR-001 to ADR-004 and ADR-015. Our code. Code:
[`arc26/hyper/providers.py`](../../arc26/hyper/providers.py). Tests:
[`tests/test_providers.py`](../../tests/test_providers.py).

## Purpose

The refinement loop takes its adapter from an injected provider (`arc26.loop.AdapterProvider`, B4,
ADR-004). Phase 2's is `arc26.ttt.SGDProvider`, a continuation of the current adapter (ADR-015).
Phase 4 recomputes the adapter from H on the enlarged demonstration set every round (B2 Q4), with a
short SGD polish by default (B2 Q3, ADR-003). This module holds those two providers,
`HyperProvider` (H alone, the $s = 0$ arm) and `HyperPolishProvider` (H, then the polish), and B4's
`Adapter`. Both plug into `refine` and `run_rounds` as they are; choosing a provider in the solver
is a later task (it edits `arc26/loop.py`, which task P2.5 holds).

## Math

Let $D = (p_1, \dots, p_n)$ be `DemoSet.pairs`, the training and voting set (the held-out pair is
never among them, ADR-013). Each pair has P3.2's embedding $\bar e_i \in \mathbb{R}^{2560}$, the
mean over the encoder's views of the layer-$L$ states pooled over the output-grid tokens of the
frozen base (ADR-001), computed once and cached by its content hash. With $\pi$ the order of the
pair hashes,

$$\bar z(D) = \frac{1}{n} \sum_{j=1}^{n} \bar e_{\pi(j)} .$$

$\bar z$ is a symmetric function of $D$; summing in hash order makes it bitwise independent of the
order of the pairs too (a float32 mean depends on the summation order: $9.5 \times 10^{-7}$ on the
tiny model). Repeated pairs count, as in P3.5's training. H maps $\bar z$ to one pair
$(A_\ell^m, B_\ell^m)$, $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$,
per layer $\ell$ and projection $m$, applied as $W x + s\,B A x$ with $s = \alpha / \sqrt{r}$
(rsLoRA, ADR-002). A round's adapter is

$$\theta_t = H(\bar z(D_t)) \quad \text{(HyperProvider)}, \qquad
\theta_t = \mathrm{polish}(H(\bar z(D_t)), s) \quad \text{(HyperPolishProvider)},$$

recomputed from H, never continued from $\theta_{t-1}$. The polish is `arc26.ttt.fit` from
$H(\bar z(D_t))$ for $s$ AdamW steps at batch 1 on fit's 128 views of $D_t$ (the pairs under 16
color permutations of the 8 dihedral transforms). On the GPU its trainer (transformers' `Trainer`
under unsloth) draws the views in a random order seeded by `train_args`' seed 42, so the $s$ steps
see $s$ shuffled views; the CPU tests' stand-in trainer takes the first $s$. By default
$s = 13 = \lceil 128 / 10 \rceil$ (B2 Q3: about 10% of the notebook's epoch). A rollback restores
$\theta_{t-1}$, the live adapter's weights before the round, exactly.

**Cost.** `fit_shape` is the round's train term for the governor: $(0, 0)$ for H, which trains
nothing (the governor's train model prices zero tokens at zero seconds), and
$(\lceil T s / 128 \rceil, s)$ for the polish, $T$ the tokens of fit's 128 views, as `SGDProvider`
counts it. The encode is exposed apart, for P4.3's cost model: `encode_shape` returns the cache
misses, their prefills ($8$ per pair at the 8 views of ADR-001), and their real tokens, a
$30 \times 30$ pair being $8 \cdot 1866$; `P3.2-02` measured 0.181 s per 1,000 prefill tokens on an
L4. H's forward is one call per `adapt` (4.6 ms per call on an L4, `P3.5-01`).

## Interfaces

| Name | Contract |
|---|---|
| `Adapter(state, rank, alpha, target_modules)` | B4's Adapter: PEFT state-dict key to tensor (`base_model.model.model.layers.{i}.self_attn.{m}.lora_A.weight` $[r, d_\text{in}]$, `...lora_B.weight` $[d_\text{out}, r]$), plus the metadata; identity equality. `digest()` is `adapter_digest(state)`; `save(path)` writes `adapter_model.safetensors` and `adapter_config.json` (`inject.save_adapter`'s layout, loadable offline by `PeftModel.from_pretrained`) |
| `adapter_digest(state) -> str` | sha256 over each tensor's key, dtype, shape, and raw bytes, in sorted key order: equal exactly for bit-for-bit equal tensors |
| `EncodeShape(pairs, prefills, tokens)` | What H's encode would prefill: distinct cache misses, sequences, real tokens |
| `HyperProvider(hypernet, encoder, cache, *, timing_logger=None)` | Checks H's width against the encoder, the cache's namespace (layer, pool, view set) against the encoder's, and that the encoder's model can disable any LoRA layers it holds; puts H in evaluation mode |
| `HyperProvider.from_checkpoint(path, *, model, tokenizer, cache_root, timing_logger=None)` | H's config and weights, the encoder's settings (layer, pool, `n_perm`, `view_seed`, `batch_tokens`), and the cache's `model_id`, all from a P3.5 checkpoint (file or folder with `latest.json`); H on the model's device |
| `emit(demos) -> Adapter` | Encodes the cache misses of `demos.pairs`, then H on $\bar z$; float32, on H's device; the live model untouched |
| `encode_shape(demos) -> EncodeShape` | Renders and tokenizes the misses' views on the CPU; the model does not run, the cache is not written |
| `adapt(demos, init, budget_s) -> model` | Checks that `init`'s `default` adapter has H's scope (r, alpha, rsLoRA, targets, every key and shape; `ValueError`, or `TypeError` for a model without it, before any change), emits, copies `init`'s adapter to the CPU, loads H's (`set_peft_model_state_dict`), returns `init`; `budget_s` is printed |
| `rollback(previous) -> model` | Loads the CPU copy into `previous`; returns it unchanged when the last `adapt` took no copy |
| `fingerprint(adapter) -> str` | `adapter_digest` of a PEFT model's `default` adapter or of an `Adapter` |
| `fit_shape(demos) -> (0, 0)` | No train term |
| `HyperPolishProvider(hyper, sgd)` | `sgd` is the puzzle's `arc26.ttt.SGDProvider` with `steps` $= s$; `steps` property |
| `HyperPolishProvider.adapt(demos, init, budget_s)` | `hyper.adapt`, then `sgd.adapt` with the budget left; returns the model in inference mode, as `fit` does |
| `HyperPolishProvider.rollback(previous)` | `sgd.rollback` (only when the last `adapt` reached it), then `hyper.rollback`: the round's starting weights, bit for bit |
| `fingerprint`, `fit_shape`, `encode_shape` | `hyper`'s digest, `sgd`'s shape, `hyper`'s encode shape |
| `POLISH_STEPS` | 13, $\lceil 128/10 \rceil$, the wiring's default $s$ |
| `ADAPTER_NAME`, `ENCODE`, `H_FORWARD`, `SWAP` | `"default"`, and the timing op names below |

## Data flow

```text
DemoSet.pairs -> task_pair -> pair_hash -> EmbeddingCache.missing
  misses, sorted -> PairEncoder.encode_pairs([pair]), one call per pair,
                    under the live model's disable_adapter -> EmbeddingCache.put
  every hash, sorted, repeats kept -> EmbeddingCache.get -> mean -> z_bar [hidden], fp32, CPU
  -> H (eval, no_grad, H's device) -> LoRAFactors -> to_peft_state_dict (fp32) -> Adapter
live model: get_peft_model_state_dict -> scope check -> CPU copy (rollback)
            -> set_peft_model_state_dict(Adapter.state) -> returned to refine
polish:     SGDProvider.adapt(demos, live model): CPU copy of H's adapter
            -> fit(default_weights=copy, max_steps=s) -> returned
rollback:   [polish: SGDProvider.rollback] -> set_peft_model_state_dict(CPU copy)
```

- **The frozen base.** H was trained on $\bar z$ of the base with nothing injected (P3.5). The
  live model holds an adapter (round 0's, or the last round's), so the encoder runs inside the PEFT
  model's `disable_adapter`, which makes every LoRA layer return its base layer's output, bit for
  bit (tested against the plain base).
- **One encode call per missing pair.** A pair's prefill batches then hold only its own views, so
  its cached embedding does not depend on which other pairs were missing with it, nor on the
  cache's history. On the CPU the batch partners did not change the bits; on a GPU a different
  batch shape can select another kernel.
- **Where things live on Kaggle.** H on the model's GPU (135.0M parameters, 540 MB in float32 at
  H's scope); the CPU copy of the live adapter (11.8M values, 24 MB in bf16); the cache under a root
  of the worker's own, since `EmbeddingCache` supports one writer (about 10 KB per pair).
- **Timing records** (`{puzzle_id, op, seconds, tokens}`, the worker's logger):

  | Op | What | Tokens |
  |---|---|---|
  | `encode` | the cache misses' prefills (P3.5's name for the same work) | real prefill tokens |
  | `h_forward` | H's forward and the export, synchronized (P3.5's name) | none |
  | `snapshot` | the CPU copy of the live adapter | none |
  | `swap` | H's adapter loaded into the live model | none |
  | `restore` | the rollback's load | none |
  | `fingerprint` | a digest (puzzle id of the last `adapt`) | none |

  The polish adds `SGDProvider`'s `snapshot` (its copy of H's adapter) and `candidate_train`.
- **Session log.** `adapt` prints `H's adapter for <task> from <n> pairs (<m> encoded), <s> s
  left`; the polish adds `SGDProvider`'s line.

## Origin

Our code (no notebook cell). It composes P3.1's export (`to_peft_state_dict`, `peft_config`), P3.2's
encoder and cache, P3.3's H, P3.5's checkpoint format (`load_checkpoint`, `EncoderSettings`), and
P2.3's `SGDProvider`, all unchanged.

## Tests

`CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/pytest tests/test_providers.py`: 51 tests, about
11 s on the CPU (3 Oct, torch 2.14.0, transformers 4.55.4, peft 0.17.1). The model is P3.5's tiny
random Qwen3 (hidden 64, 4 layers, GQA, vocabulary 16) wrapped by PEFT as `arc26.solver.load` wraps
the base (rsLoRA, dropout 0, q, k, v, o, $r = 4$, $\alpha = 16$, $s = 8$); H is tiny (latent 16,
head input 8, dropout 0.25) with random non-zero heads, so its adapter depends on $\bar z$; the
encoder runs 2 layers under the 8 views of ADR-001. Skipped without torch, transformers, peft, or
safetensors.

| Group | Checks |
|---|---|
| `refine()` on DEV4 stand-ins, each of `SGDProvider`, `HyperProvider`, `HyperPolishProvider` | Four synthetic puzzles under DEV4's ids and layout: kept, kept (one of two inputs promoted), rolled back (a wrong pseudo-pair fails the gate; round 0's weights back bit for bit, the record's digests equal), nothing to promote; the gate checked the provider's candidate; SGD continued round 0's adapter for $s'$ steps, H holds $H(D_1)$, the polish started from $H(D_1)$, trained $s$ steps, and gives the same weights from another initial adapter; the H providers encoded each training pair once and then only the pseudo-pair, never the held-out pair; the plans price $(0, 0)$, $(T, s)$, $(T, s')$ |
| Protocol | Each provider: `adapt` returns the live model with a new adapter, `fingerprint` stable, `fit_shape`, `rollback` gives the previous weights back |
| Cache | Each miss encoded once, alone; nothing encoded again by later rounds, other providers on the same cache, or another puzzle sharing a pair |
| Fingerprints | One digest for a fixed `DemoSet` whatever the pairs' order, the cache's history, or H's mode; the live model's digest equals the emitted one's; another set gives another; the digest is the documented sha256 |
| Rollback | Each provider, after a kept round; after a swap that changed a weight and raised; after the polish's fit raised mid-training; after H raised in round 2 (the polish's earlier copy is not loaded) |
| Decode path | Zero-initialized H swapped into a live model that held another adapter: logits and `arc26.decode.calc_scores` NLLs bit for bit equal to the base's; the saved `Adapter` through `PeftModel.from_pretrained` too |
| Edges | The frozen base (the live adapter changes nothing cached); refusals of another rank, alpha, targets, or base, of an adapter without rsLoRA (PEFT would scale by $\alpha / r$), and of a model without the adapter, with nothing changed; a repeated pair, encoded once and counted twice in $\bar z$; $k = 2$ with one pair held out; an empty pair set; a $30 \times 30$ pair ($8 \cdot 1866$ tokens, the `encode` record's count); `from_checkpoint`; B4's files; the polish's budget never negative |
| bf16 cast | The live model cast as `arc26.solver.load` casts it: H's swap is exactly the bf16 rounding of the export, so the live digest is not the emitted one's; both providers' rollbacks give the bf16 weights back bit for bit, after round 1 and after a later round; zero-initialized H keeps the cast base's logits and `calc_scores` NLLs bit for bit |

Stubbed, because they cannot run on the CPU: unsloth (`FastLanguageModel.for_training` and
`for_inference` as `train()` and `eval()`; its trainer as a stand-in that runs AdamW over the first
`max_steps` views with the moved collator and `UnslothFixedTrainer.compute_loss`;
`UnslothTrainingArguments` as `dict`), datasets (`Dataset.from_list`), torch.cuda's peak-memory
statistics (`reset_peak_memory_stats` raises without a GPU), and `refine`'s decodes (round 0's
candidates, the re-decode, the gate's decode): a random tiny model decodes no grid, so they return
fixed votes. Everything else is real, `arc26.ttt.fit` and `SGDProvider` included.

Negative controls (not committed), each a one-line change to the module: the encoder with the
adapter active, $\bar z$ in `DemoSet.pairs` order, all misses in one encode call, no CPU copy, H
left in training mode, H's adapter added to `init`'s weights, the scope check skipped, its rsLoRA
term dropped, repeated pairs counted once in $\bar z$, the encode tokens priced as a fit, the
polish without H, the polish's rollback in the other order, and the polish's rollback without its
guard; each failed at least one test.

CI: this branch adds `or providers` to the `-k` filter of the `hyper-cpu` job in
`.github/workflows/ci.yml` (the keyword selects this module's tests and no others), so the job runs
them on CPU torch 2.8.0 with Kaggle's transformers and peft. `.github/` is a protected path, so the
PR carries the `needs-human` label and the human merges it. The other job skips the module.

## Limits

- The live adapter must have H's scope: in the solver, `lora_r` 16 and `lora_targets` q, k, v, o
  (P1.3's arm settings), alpha 32 and rsLoRA as `arc26.solver.load` sets them. The notebook's
  scope (rank 256, nine targets) is refused.
- bf16: `arc26.solver.load` casts every float32 parameter to bf16, the adapter's included, so H's
  float32 export is rounded at the swap and the LoRA delta runs in bf16, while P3.5 trained H with
  float32 factors and a float32 delta. P3.1 measured logit moves of $7.0 \times 10^{-2}$ to
  $9.4 \times 10^{-2}$ for such rounding on its tiny model (`inject.md`, Limits). The CPU tests
  pin the providers under the cast: the swap is exactly the bf16 rounding of the export, every
  rollback stays bit for bit, and zero-initialized H keeps the base's logits. The effect of the
  rounding on a trained H's logits at Qwen3-4B is unmeasured (ADR-017).
- The encoder at test time runs unsloth's patched model; P3.5 and P3.2's cache ran transformers'.
  $\bar z$'s agreement between the two is unmeasured.
- Determinism holds on one device and software stack. Bit-for-bit equality of an emitted adapter
  across sessions on GPU is P4.2's check.
- The governor does not price H's encode and forward yet (P4.3): a round's are the pseudo-pairs'
  prefills, about 0.8 s for a median pair and 2.7 s for a $30 \times 30$ one (estimate from
  `P3.2-02`'s rate), and one forward.
- One `HyperProvider` per worker (its CPU copy and the puzzle id of its records are the last
  `adapt`'s); one cache root per worker.
- The polish trains on fit's samples, the pairs under its 128 views, as `SGDProvider` does; the
  reference has no leave-one-out task generation (ADR-009, ADR-011), which B2 Q3's wording names.
