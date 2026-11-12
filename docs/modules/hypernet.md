# `arc26.hyper.hypernet`: hypernetwork H

Task P3.3 (BRIEF B2 Q1 and Q2, B5 P3.3; ADR-001, ADR-002). Code:
[`arc26/hyper/hypernet.py`](../../arc26/hyper/hypernet.py). Config:
[`configs/hypernet.yaml`](../../configs/hypernet.yaml). Tests:
[`tests/test_hypernet.py`](../../tests/test_hypernet.py). Background: the
[research note](../research/d2l_t2l_reading.md) sections 2 to 5, the
[design page](../hypernet_design.md), and the [injection module](inject.md).

## Purpose

H maps a task's pooled pair embedding $\bar{z}$ to a LoRA adapter for the frozen base in one
forward pass. Meta-training (P3.5) trains it end to end through `inject`; test time (P4.1) exports
its adapter to PEFT. At initialization H reproduces the base exactly, so training starts from the
notebook's model and every early gradient step is meaningful. The design is T2L's hypernetwork
head (research note, section 2), written from the spec; no T2L code is copied.

## Architecture

Notation: $E$ = `embed_dim` (2560), $d$ = `latent` (512), $H_\text{in}$ = `head_in` (384),
$r$ = `rank` (16), $L$ decoder layers (36), $M$ target module types (4), $\mathrm{LN}$ a LayerNorm,
$\sigma$ SiLU, $\mathrm{drop}$ dropout with probability `dropout` (0.05).

Codes. The task code is Q1's $z = \mathrm{MLP}(\bar{z})$ (T2L's `TaskEncoder`); layer $i$ and
module type $m$ get learned embeddings:

$$
t = \mathrm{LN}(W_t \bar{z} + b_t) \in \mathbb{R}^{d/2}, \qquad
\ell_i = \mathrm{LN}(e^\text{layer}_i) \in \mathbb{R}^{d/4}, \qquad
\mu_m = \mathrm{LN}(e^\text{module}_m) \in \mathbb{R}^{d/4}, \qquad
u_{m,i} = [\,t;\ \ell_i;\ \mu_m\,] \in \mathbb{R}^{d}.
$$

Trunk. With $\phi(x) = \sigma\big(W_2\,\mathrm{drop}(\sigma(W_1 x + b_1)) + b_2\big)$, where
$\phi_\text{mix}, \phi_1, \phi_2 : d \to 4d \to d$ and $\phi_3 : d \to 4d \to H_\text{in}$:

$$
u^{(1)} = \mathrm{drop}\big(\phi_\text{mix}(u)\big), \qquad
u^{(k+1)} = u^{(k)} + \mathrm{drop}\big(\phi_k(\mathrm{LN}(u^{(k)}))\big),\ k = 1, 2, \qquad
h = \phi_3\big(\mathrm{LN}(u^{(3)})\big) \in \mathbb{R}^{H_\text{in}}.
$$

Heads. One linear head per module type, shared by all layers:

$$
o_{m,i} = W_m h_{m,i} + c_m \in \mathbb{R}^{r (d^m_\text{in} + d^m_\text{out})}, \qquad
A_{m,i} = \mathrm{mat}_{r \times d^m_\text{in}}\big(o_{m,i}[{:}\,r d^m_\text{in}]\big), \qquad
B_{m,i} = \mathrm{mat}_{r \times d^m_\text{out}}\big(o_{m,i}[r d^m_\text{in}{:}]\big)^\top,
$$

with $\mathrm{mat}$ the row-major reshape, so $A_{m,i} \in \mathbb{R}^{r \times d_\text{in}}$ and
$B_{m,i} \in \mathbb{R}^{d_\text{out} \times r}$ are in PEFT's layout. The adapted projection is
P3.1's $y = W x + s\,B_{m,i} A_{m,i}\,x$ with $s = \alpha/\sqrt{r} = 8$ (`config.scaling`, from
`inject.rslora_scaling`).

Evaluation. The trunk runs once on the grid of all inputs, `[M, batch, L, d]`; each head runs once
on its slice `[batch, L, H_in]`. The output for one task is `LoRAFactors` keyed
`layers.{i}.{block}.{m}`, layer-major, in the order `wrap_targets` returns.

Shapes. H reads six sizes from the base's config (`BaseShapes.from_config(model.config)`) and
derives the widths as `Qwen3Attention` and `Qwen3MLP` build them (transformers 4.55.4,
`modeling_qwen3.py` 164 to 181 and 75 to 77):

| Module | Block | $d_\text{in}$ | $d_\text{out}$ | Qwen3-4B |
| --- | --- | --- | --- | --- |
| `q_proj` | `self_attn` | `hidden_size` | `num_attention_heads * head_dim` | $2560 \to 4096$ |
| `k_proj`, `v_proj` | `self_attn` | `hidden_size` | `num_key_value_heads * head_dim` | $2560 \to 1024$ |
| `o_proj` | `self_attn` | `num_attention_heads * head_dim` | `hidden_size` | $4096 \to 2560$ |
| `gate_proj`, `up_proj` | `mlp` | `hidden_size` | `intermediate_size` | $2560 \to 9728$ |
| `down_proj` | `mlp` | `intermediate_size` | `hidden_size` | $9728 \to 2560$ |

## Initialization

T2L's bias-HyperInit (Q2): $W_m = 0$ and $c_m = [\,\mathrm{vec}(A^0_m);\ 0\,]$, with one
$A^0_m \in \mathbb{R}^{r \times d^m_\text{in}}$ per module type drawn as PEFT draws a new `lora_A`
(`kaiming_uniform_(a=sqrt(5))`, `peft/tuners/lora/layer.py` 264), that is
$A^0_m \sim \mathcal{U}\big(-1/\sqrt{d^m_\text{in}},\ 1/\sqrt{d^m_\text{in}}\big)$. For every task and
every layer, H then emits $A_{m,i} = A^0_m$ and $B_{m,i} = 0$, whatever $h$ is (dropout included).
With $g = \partial L / \partial y$ summed over tokens:

- The delta $s\,B A x$ is exactly zero, so the injected logits equal the base's bitwise, in fp32 and
  bf16, and so do PEFT's with the exported adapter (tests).
- $\partial L / \partial B_{m,i} = s\, g\, (A^0_m x)^\top \neq 0$, so the heads learn from step 0:
  the gradient of the $B$ rows of $W_m$ is $\sum_i \mathrm{vec}\big((\partial L / \partial B_{m,i})^\top\big)\, h_{m,i}^\top \neq 0$,
  and since $h_{m,i}$ differs per layer through $\ell_i$, so does the $B$ emitted after the first
  step.
- $\partial L / \partial A_{m,i} = s\, B^\top g\, x^\top = 0$ and $\partial L / \partial h = W_m^\top\, \partial L / \partial o = 0$:
  at step 0 only the $B$ rows and $B$ bias of the heads move; the $A$ rows, the trunk, and the
  codes learn from step 1 on (test).

Why this initialization: H starts at the base, so no early steps undo a random adapter. Its step-0
adapter is the one a fresh PEFT LoRA starts from (`init_lora_weights=True`: kaiming $A$, zero $B$,
`layer.py` 264 and 269). Zeroing $A$ as well would stall training: with $A = 0$ and $B = 0$ both
gradients above vanish. D2L's alternative, random head weights with zero
per-(layer, rank) scalers on $B$ (research note, section 1.3), also starts at $B = 0$; it is not
used.

Seeding. Parameters are created on the meta device and filled by `reset_parameters`, so PyTorch's
default initialization never runs. Every value is drawn on the CPU from a generator seeded with
`seed` and copied to the parameters' device: the weights depend on the seed only, not on the
device, and the global RNG is untouched (test). Trunk and codes follow PyTorch's defaults
(`nn.Linear`: kaiming-uniform weight, uniform bias; `nn.Embedding`: $\mathcal{N}(0, 1)$;
`nn.LayerNorm`: ones and zeros).

## Set aggregator (P3.3b)

`aggregator.kind` in [`configs/hypernet.yaml`](../../configs/hypernet.yaml): `mean` (the default)
is H as above, reading $\bar z$ only; `set` adds a branch that reads the demonstration set itself
(scoping note `docs/research/h_upgrades_scoping.md` on `task/H-scope`, section 1(a); ADR-001 as
amended 17 Oct). Notation: $\bar e_1, \dots, \bar e_n$ the pair embeddings of a task (their mean is
$\bar z$), $d_s$ = `aggregator.width` (512), $k$ = `aggregator.blocks` (2), `aggregator.heads` (8).

$$
x_i^{(0)} = W_\text{in}\,\mathrm{LN}(\bar e_i) \in \mathbb{R}^{d_s}, \qquad
X^{(j+1)} = \mathrm{SAB}_j(X^{(j)}),\ j < k, \qquad
c_{m,i} = \mathrm{MHA}\big(q_{m,i},\ \mathrm{LN}(X^{(k)}),\ \mathrm{LN}(X^{(k)})\big),
$$

$$
u_{m,i} = [\,t + W_o\, c_{m,i};\ \ell_i;\ \mu_m\,], \qquad W_o \in \mathbb{R}^{(d/2) \times d_s},
$$

with $\mathrm{SAB}(X) = Y + \mathrm{FFN}(\mathrm{LN}(Y))$, $Y = X + \mathrm{MHA}(\mathrm{LN}(X))$ (pre-LN,
FFN $d_s \to 4 d_s \to d_s$ with SiLU), one learned query $q_{m,i}$ per (module, layer) row (144 on
Qwen3-4B), and no positions, so the adapter does not depend on the pairs' order. Padded pairs are
masked out of every attention as keys (`pad_pair_sets`, `mask` True for a real pair). Only the
task code becomes per row; the trunk already runs on the $[M, \text{batch}, L, d]$ grid, so its cost
is unchanged.

Initialization. The aggregator draws after every other parameter, so a set H's shared parameters
equal the mean H's of the same seed; attention as `nn.MultiheadAttention` initializes itself,
Xavier-uniform queries, and $W_o = 0$ (weight and bias). Then $W_o c = 0$ exactly, the set H
computes what the mean H does (base-exact from scratch, and bitwise equal to a loaded mean H in
eval mode), $\partial L / \partial W_o = \sum (\partial L / \partial t_{m,i})\, c_{m,i}^\top \neq 0$
at step 0, and the aggregator's own gradients are non-zero from step 1.

Size on Qwen3-4B, $d_s = 512$: $+3{,}152{,}384$ per block and $+2{,}573{,}056$ for input, pooling,
queries, and $W_o$; with 2 blocks $+8{,}877{,}824$, H 143,836,032 (3 blocks: 146,988,416; 4 would
pass the 150M budget at 150,140,800).

Warm start. `scripts/train_hypernet.py --init-from <full checkpoint>` loads a mean H's weights into
a set H by name, keeps the checkpoint's step, data cursor, RNG states, AdamW state, and cosine for
the loaded parameters, and puts the new ones in two parameter groups of their own with fresh
moments: peak `new_params_lr`, a linear warmup of `new_params_warmup_steps` from the start step,
then a cosine to `min_lr` at `steps` (`train.NewParamsSchedule`). The run validates at its start
step (equal to the loaded H's validation there) and records `init_from` in `run.json` and in
every checkpoint, so `--resume` rebuilds the four groups. `--resume` and `RESUME_KEYS` are
unchanged for the mean path: a mean config serializes with no `aggregator` block and no
`new_params_*` keys, exactly as before.

Checkpoints. A checkpoint's config builds the class it needs (ADR-017 decision 7):
`HypernetConfig(**config["hypernet"])` gives a mean H when the block is absent (every checkpoint
before P3.3b) and a set H otherwise; `HyperProvider` then passes the cached $\bar e_i$ in
pair-hash order beside their mean.

## Parameter count

With $\mathrm{Linear}(a, b) = (a + 1)\,b$, $\mathrm{LN}(n) = 2n$, $\mathrm{Embedding}(n, k) = nk$:

| Part | Modules | Parameters |
| --- | --- | --- |
| Task encoder | $\mathrm{Linear}(E, d/2)$, LN | $(E + 1)\,d/2 + d$ |
| Layer codes | $\mathrm{Embedding}(L, d/4)$, LN | $L d/4 + d/2$ |
| Module codes | $\mathrm{Embedding}(M, d/4)$, LN | $M d/4 + d/2$ |
| Mixer | $d \to 4d \to d$ | $8d^2 + 5d$ |
| Two residual blocks | LN, $d \to 4d \to d$ | $2\,(8d^2 + 7d)$ |
| `mlp3` | LN, $d \to 4d \to H_\text{in}$ | $4d^2 + 6d + (4d + 1)\,H_\text{in}$ |
| Heads | $\mathrm{Linear}(H_\text{in}, r(d^m_\text{in} + d^m_\text{out}))$ per $m$ | $(H_\text{in} + 1)\, r \sum_m (d^m_\text{in} + d^m_\text{out})$ |

$$
N_\text{trunk} = 28 d^2 + d\Big(\frac{E + 1}{2} + \frac{L + M}{4} + 27\Big) + (4d + 1)\,H_\text{in},
\qquad
N_\text{heads} = (H_\text{in} + 1)\, r \sum_m \big(d^m_\text{in} + d^m_\text{out}\big).
$$

Default scope ($E = 2560$, $d = 512$, $L = 36$, $M = 4$, $H_\text{in} = 384$, $r = 16$,
$\sum_m (d_\text{in} + d_\text{out}) = 6656 + 3584 + 3584 + 6656 = 20480$):

$$
N_\text{trunk} = 7{,}340{,}032 + 674{,}560 + 786{,}816 = 8{,}801{,}408, \qquad
N_\text{heads} = 385 \cdot 16 \cdot 20480 = 126{,}156{,}800,
$$

$N = 134{,}958{,}208$, under the 150M budget; the meta-device count in the test equals the closed
form (the research note's 135.0M). The heads are 93% of H, so the budget binds through
$H_\text{in}$: $N_\text{heads} = (H_\text{in} + 1) \cdot N_\text{emit} / L$, where
$N_\text{emit} = L r \sum_m (d^m_\text{in} + d^m_\text{out})$ is the number of values per adapter.
G2's expansions are config changes, but under the cap they need a smaller head input (meta-device
counts):

| Scope | $N_\text{emit}$ | $N$ at $H_\text{in} = 384$ | Largest $H_\text{in}$ within 150M | $N$ there |
| --- | --- | --- | --- | --- |
| q, k, v, o; $r = 16$ (default) | 11,796,480 | 134,958,208 | 429 | 149,796,013 |
| q, k, v, o; $r = 32$ | 23,592,960 | 261,115,008 | 214 | 149,355,478 |
| plus MLP; $r = 16$ | 33,030,144 | 362,040,832 | 153 | 149,624,089 |
| plus MLP; $r = 32$ | 66,060,288 | 715,279,872 | 76 | 149,466,316 |

Whether a narrower head input costs accuracy is not measured; G2 weighs it with this table.

## Interfaces

| Name | Contract |
| --- | --- |
| `HypernetConfig` | frozen dataclass: `embed_dim`, `latent`, `head_in`, `dropout`, `rank`, `alpha`, `target_modules`, `seed`, `aggregator` (default `AggregatorConfig()`, a mapping is converted); validated on construction and by `dataclasses.replace`; `scaling` is `rslora_scaling(rank, alpha)`; `to_dict()` leaves the default aggregator out |
| `AggregatorConfig` | frozen dataclass: `kind` (`mean`, `set`), `width`, `blocks`, `heads`; `from_mapping` refuses unknown keys |
| `HypernetConfig.from_yaml(path)` | loads `configs/hypernet.yaml`; refuses missing and unknown keys (`aggregator` may be absent) |
| `BaseShapes` | frozen dataclass of the base sizes; `from_config(model.config)`; `projection(name) -> (d_in, d_out)` |
| `PROJECTIONS` | projection name to block (`self_attn`, `mlp`), in Qwen3's module order |
| `Hypernet(config, base)` | `nn.Module`, float32 parameters on the default device; under `torch.device("meta")` it stays on meta (counts, shape checks) |
| `Hypernet.forward(z_bar, pairs=None, mask=None) -> list[LoRAFactors]` | `z_bar` `[batch, embed_dim]` in float32 on H's device; a set H also takes `pairs` `[batch, n_max, embed_dim]` and `mask` `[batch, n_max]` (bool, None for all real), a mean H refuses them; one dict per task, keys `layers.{i}.{block}.{m}` in `factor_keys` order, values `(A [r, d_in], B [d_out, r])`, views of the head outputs (B transposed) that `inject` and `to_peft_state_dict` copy |
| `SetAggregator`, `pad_pair_sets(sets)` | the set branch, `[batch, n_max, E] -> [batch, M L, d/2]`; padding of `[n_i, E]` sets to `pairs` and `mask` |
| `Hypernet.factor_keys` | the keys of each emitted dict, equal to `wrap_targets(model, config.target_modules)` |
| `Hypernet.reset_parameters()` | the seeded initialization above; after `to_empty(device=...)` it materializes an H built on meta |

Batched tasks: `forward` returns a list of `LoRAFactors`, one per task, so each element goes
straight into `inject` or `to_peft_state_dict`, and P3.5's one episode per step is
`hypernet(z_bar[None])[0]`. The trunk and heads still run batched over tasks. Factors with a
leading batch axis would pay off only with distinct adapters in one base forward, which `inject`
does not support (P3.1 limits).

## Data flow

P3.2 caches $\bar{z}$ per task; H maps it to `LoRAFactors`. P3.5 calls `wrap_targets(model,
config.target_modules)` once, then per step runs H in training mode and the base's forward and
`loss.backward()` inside one `with inject(model, factors, config.scaling)` block. Its AdamW should
not decay the heads' biases, which hold $A^0$ (D2L also excludes biases and norms from decay).
P4.1 runs H in eval mode and exports with `to_peft_state_dict(factors)` or
`save_adapter(factors, path, config.rank, config.alpha, config.target_modules)`. On a bf16 base it
must load them with a loader that keeps parity with `inject` (`docs/modules/inject.md`, Limits:
`PeftModel.from_pretrained` with its defaults, or `load_adapter` or `set_peft_model_state_dict`
into an existing adapter name); the others round the factors to bf16. H's initial adapter is exempt
($B = 0$ rounds to 0), a trained one is not. A checkpoint is the
state dict plus the config file and the `BaseShapes` it was built with. H has no puzzle id, so it
writes no timing records; its callers time the forward as an op (P0.3 logger), as for `inject`.

## Tests

`CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/pytest tests/test_hypernet.py -rP`: 29 tests,
about 3 s on CPU. The tiny base is a random Qwen3 built from `Qwen3Config` (hidden 64,
intermediate 128, 2 layers, 4 heads, 2 KV heads, `head_dim` 16, eager attention); the tiny H keeps
the default scope ($r = 16$, $\alpha = 32$, q, k, v, o) with $E = 24$, $d = 32$,
$H_\text{in} = 16$. The PEFT checks load with `PeftModel.from_pretrained(base, path)` and its
defaults, a loader that keeps bf16 parity (`docs/modules/inject.md`, Limits). Skipped when torch,
transformers, peft, or safetensors is missing.

| Test | Result (1 Oct, torch 2.14.0 CPU) |
| --- | --- |
| Config file holds the default scope; bad values, missing and unknown keys raise | pass |
| `BaseShapes` from `Qwen3Config` (and the `head_dim` fallback) give the table above | pass |
| Qwen3-4B on the meta device, default scope and MLP with $r = 32$: keys equal `wrap_targets`, shapes equal every module's, `inject` accepts them | 144 and 252 pairs |
| Default-scope parameter count on the meta device equals the closed form, at most 150M | 134,958,208 |
| Tiny base: keys, order, shapes, dtype for a batch of 3 | pass |
| Initial adapter: $B = 0$ exactly, one $A^0$ per module type within PEFT's bound, the same for every task | pass |
| Output layout: with zero head weights and a known bias $[\mathrm{vec}(A^*); \mathrm{vec}(B^{*\top})]$ per head, every layer emits $A^*$ and $B^*$ exactly (including `k_proj`, $d_\text{out} \neq d_\text{in}$) | exact |
| Initial adapter, H in training mode: injected logits bitwise equal to the base, PEFT from the saved adapter too (fp32, bf16); a non-zero $B$ moves them, and the two paths agree bitwise | exact |
| CE loss through `inject`: every $\partial L / \partial B$ non-zero, every $\partial L / \partial A$ zero, head $B$ rows only, trunk zero; after one SGD step every parameter has a non-zero gradient, every row of both code tables too, and $B$ differs between layers and between module types | pass |
| Same seed, same weights and outputs; another seed, other weights; the global RNG unchanged; a meta-built H materializes to the same weights | exact |
| Batch rows are independent tasks (batched equals single within float tolerance) | pass |
| Each residual block is pre-LN: shifting its input by a constant shifts its output by the same constant | pass |
| MLP modules and $r = 32$ by editing the YAML only: keys `layers.{i}.mlp.{m}`, shapes, bitwise base logits, PEFT export | pass |
| Bad `z_bar` shapes raise | pass |

P3.3b's tests are in [`tests/test_hypernet_set.py`](../../tests/test_hypernet_set.py) (25 tests,
about 5 s on CPU): the parameter count above on the meta device; the mean H's state dict at seed 0
equal by sha256 to the one before P3.3b; from scratch the shared parameters equal the mean H's and
$B = 0$; the warm start bitwise equal to the loaded H in eval mode, with and without a mask;
permutation invariance (within $10^{-6}$, and other pairs with the same mean move the adapter);
padded rows change nothing (bit for bit for any finite padding); $\partial L / \partial W_o \neq 0$ and
the rest of the aggregator 0 at step 0, non-zero at step 1; and in training (smoke fixture, one
process) the four parameter groups, the loaded AdamW state, the step and cursor kept, the start
validation equal to the mean run's at that step, the same episodes and loaded-group learning
rates afterwards, an exact resume of a warm-started run, and the refusals. Two provider tests are
in `tests/test_providers.py`.

Negative controls (not committed) mutated the module: a swapped bias layout (part of $A^0$ in $B$)
failed 5 tests, PyTorch's default initialization left on failed the RNG test, module-major keys
failed 4, and an uninitialized trunk failed 8. After review round 1: one layer code for all layers
and module 0's trunk output for every head each fail the gradient test; $B$ read as row-major
$[d_\text{out}, r]$ and a permuted $A$ each fail the layout test; a block without its skip
connection fails the residual test.

Forward time on the dev box CPU (8 threads, batch 1, no grad): tiny H 0.4 ms (median of 50, in
the test output); default-scope H about 60 ms (median of 10; one-off measurement, 0.5 s to build).
The acceptance item "under 50 ms on L4" is GPU-only and belongs to P3.6, the next GPU run that
loads H. For scale (an estimate, not a measurement): the forward is bound by reading the 505 MB
of head weights, about 1.7 ms at the L4's 300 GB/s; the heads' 9.1 GFLOP take about 0.3 ms at its
fp32 peak of about 30 TFLOPS.

## Environment

As for [`inject`](inject.md): torch 2.14.0 (local CPU build from the uv cache),
transformers 4.55.4, peft 0.17.1, safetensors 0.6.2, accelerate 1.11.0, installed offline. The
config loader uses pyyaml, in the `hyper` extra (`pyyaml>=6`). CI installs `.[dev]` only, so the
tests skip there.

## Limits

- Qwen3-style decoders: `PROJECTIONS` and `BaseShapes.projection` encode Qwen3's projection names,
  blocks, and widths; another architecture needs its own table.
- Every target in every layer: `inject` expects factors for every wrapped projection, so H has no
  per-layer subset.
- One adapter per base forward (`inject`); H itself is batched over tasks.
- The 150M budget holds for the default scope; expansions need a smaller `head_in` (table above).
- $\bar{z}$ enters as given, in float32 on H's device, with no normalization before $W_t$: outlier
  dimensions of the layer-24 states would dominate the task code. Standardizing $\bar{z}$ with
  cache statistics is P3.2 or P3.5 work if it shows up.
- Dropout makes training-mode output stochastic; P4.1 must call `eval()` so adapters are
  deterministic.
- GPU timing pending (P3.6); the numbers above are CPU only.
