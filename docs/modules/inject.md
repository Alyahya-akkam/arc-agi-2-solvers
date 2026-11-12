# `arc26.hyper.inject`: LoRA injection and PEFT export

Task P3.1 (BRIEF B5, B8; ADR-002). Code: [`arc26/hyper/inject.py`](../../arc26/hyper/inject.py).
Tests: [`tests/test_inject.py`](../../tests/test_inject.py). Background:
[research note](../research/d2l_t2l_reading.md) sections 3, 6, 7.

## Purpose

Meta-training (P3.5) needs H's adapter applied to the frozen base as a differentiable function of
the factors, so the task loss reaches H. Test time (P4.1) loads the same adapter through PEFT for
the polish and decoding. If the two paths disagreed, H would be trained on one model and evaluated
on another. This module provides the training path, replaying PEFT 0.17.1's unmerged LoRA forward
op for op, and the export to PEFT's format; `tests/test_inject.py` guards the seam.

## Math

For a target projection with frozen weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$
(bias-free in Qwen3) and input $x \in \mathbb{R}^{d_\text{in}}$:

$$
y = W x + s\,B A x, \qquad A \in \mathbb{R}^{r \times d_\text{in}},\quad
B \in \mathbb{R}^{d_\text{out} \times r},\quad s = \frac{\alpha}{\sqrt{r}} \ \text{(rsLoRA)}.
$$

H's scope (Q2): $r = 16$, $\alpha = 32$, $s = 8$, targets `q_proj` ($2560 \to 4096$), `k_proj` and
`v_proj` ($2560 \to 1024$), `o_proj` ($4096 \to 2560$) on 36 layers. With $g = \partial L /
\partial y$ summed over tokens,

$$
\frac{\partial L}{\partial B} = s\, g\, (A x)^\top, \qquad
\frac{\partial L}{\partial A} = s\, B^\top g\, x^\top,
$$

so at H's zero-initialized $B$ (ADR-002) the gradient reaches $B$ but not yet $A$.

Rounding on a bf16 base with float32 factors, where $\mathrm{bf}(\cdot)$ rounds to bf16:

$$
y = \mathrm{bf}\Big(\mathrm{bf}(W x) + s \cdot B\,\big(A\,x\big)\Big),
$$

with $x$ upcast exactly, the delta and the sum in float32, and one final rounding. Merging
$W + sBA$ into a bf16 weight rounds differently (research note, risk 4).

## PEFT 0.17.1 op sequence

Read from the installed source (`.venv/lib/python3.11/site-packages/peft/`, version 0.17.1, the
Kaggle image's version per `P1.1-01`'s `pip_freeze.txt`). `lora.Linear.forward`,
`tuners/lora/layer.py` 744 to 782, unmerged path with one active adapter:

| Step | PEFT 0.17.1 | `InjectableLinear.forward` |
| --- | --- | --- |
| Base output | `result = self.base_layer(x, *args, **kwargs)` (757) | `result = self.base_layer(x)` |
| Output dtype | `torch_result_dtype = result.dtype` (758) | same |
| Input cast | `x = self._cast_input_dtype(x, lora_A.weight.dtype)` (769): `x.to(dtype=dtype)` unless equal (`tuners/tuners_utils.py` 1128 to 1130) | `x = x.to(lora_a.dtype)` |
| Delta, add | `result = result + lora_B(lora_A(dropout(x))) * scaling` (771) | `result + F.linear(F.linear(x, lora_a), lora_b) * scaling` |
| Cast back | `result = result.to(torch_result_dtype)` (780) | same |

- `lora_A`, `lora_B` are bias-free `nn.Linear` (`layer.py` 217 to 218), whose forward is
  `F.linear(input, self.weight, self.bias)` (`torch/nn/modules/linear.py` 134); dropout 0 is
  `nn.Identity` (`layer.py` 213). `scaling` is the Python float `lora_alpha / math.sqrt(r)`
  with `use_rslora` (`layer.py` 221 to 222), the same expression as `rslora_scaling`.
- Adapter dtype: `update_layer` (`layer.py` 249) calls `_move_adapter_to_device_of_base_layer`,
  which casts new adapters to the base layer's dtype (`tuners_utils.py` 1105);
  `PeftModel.__init__` (`peft_model.py` 133) then calls
  `_cast_adapter_dtype`, which upcasts bf16 and fp16 adapter weights to float32
  (`tuners_utils.py` 353 to 394) when `autocast_adapter_dtype=True`, the default of
  `get_peft_model` (`mapping_func.py` 36) and `PeftModel.from_pretrained` (`peft_model.py` 384).
  Loading into such an existing adapter copies the file into its float32 parameters
  (`utils/save_and_load.py` 524), so `inject` computes in float32 by default and the export is
  float32. Other loaders round the factors to bf16 before any upcast: see Limits.
- Keys: `get_peft_model_state_dict` strips the adapter name (`save_and_load.py` 319);
  `set_peft_model_state_dict` reinserts it (`_insert_adapter_name_into_state_dict`, 351).
- Not replayed: the mixed-batch path (`adapter_names`, `_mixed_batch_forward`, `layer.py` 544 to
  574) and merged adapters (`merge`, 629).

## Interfaces

| Name | Contract |
| --- | --- |
| `LoRAFactors` | `dict[str, tuple[Tensor, Tensor]]`, key `layers.{i}.self_attn.{m}`, value `(A [r, d_in], B [d_out, r])` |
| `rslora_scaling(r, alpha) -> float` | $\alpha/\sqrt{r}$, the one source of $s$ for injection and export |
| `wrap_targets(model, target_modules) -> list[str]` | wraps each target `nn.Linear` in `InjectableLinear` once (idempotent, atomic); returns the keys `inject` expects (every wrapped projection, module order); the output is bitwise unchanged while nothing is injected |
| `inject(model, factors, scaling, dtype=float32)` | context manager: checks keys, shapes (transposes and D2L's `[r, d_out]` B raise), dtypes, devices; installs fresh contiguous copies for forward and backward; removes them on exit, also by exception; does not nest |
| `to_peft_state_dict(factors, dtype=float32) -> dict[str, Tensor]` | keys `base_model.model.model.layers.{i}.self_attn.{m}.lora_{A,B}.weight`, detached copies with own storage |
| `peft_config(r, alpha, target_modules) -> dict` | `adapter_config.json`: LORA, `CAUSAL_LM`, `use_rslora`, dropout 0, bias none, no DoRA, no `fan_in_fan_out` |
| `save_adapter(factors, path, r, alpha, target_modules, dtype=float32)` | writes `adapter_model.safetensors` and `adapter_config.json` (B4's Adapter format) without `save_pretrained`, so nothing probes the Hub (risk 6) |

## Data flow

P3.3's H emits `LoRAFactors`. P3.5 calls `wrap_targets` once on the frozen base, enables
`gradient_checkpointing_enable({"use_reentrant": False})`, and per step runs the forward and
`loss.backward()` inside one `with inject(...)` block, so the recompute sees the same factors
(risk 2). P4.1's providers turn factors into the B4 Adapter with `to_peft_state_dict` (or
`save_adapter`) and decode through PEFT. The module adds no operation of its own, so it writes no
timing records; the forward and backward it changes are the caller's ops and are timed there
(P0.3 logger).

## Tests

`CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/pytest tests/test_inject.py -rP`: 31 tests on
a tiny random Qwen3 built from `Qwen3Config` (hidden 64, 2 layers, 4 heads, 2 KV heads,
`head_dim` 16, so `k_proj` is $64 \to 32$), CPU, eager attention, $r = 8$, $\alpha = 16$
($s = 4\sqrt{2}$ rounds, unlike $s = 8$, so the place of the multiply shows). The module is
skipped when torch, transformers, peft, or safetensors is missing.

| Test | Result (1 Oct, torch 2.14.0 CPU) |
| --- | --- |
| Zero $B$ gives logits bitwise equal to the unwrapped base (fp32, bf16) | exact |
| Injected logits against PEFT with the exported state dict: bitwise in fp32 and bf16, max abs diff reported against the brief's $10^{-3}$ | fp32 0, bf16 0 (bitwise) |
| `save_adapter`, then `PeftModel.from_pretrained` offline, reproduces the injected logits bitwise; file keys equal PEFT's own | fp32 0, bf16 0 (bitwise) |
| Each wrapped projection against PEFT's `lora.Linear` on a copy of its base layer, bitwise (fp32, bf16) | exact |
| Loaders on a bf16 base (Limits table): the parity rows give bitwise logits and store the float32 factors; the others store bf16-rounded factors and differ | as documented |
| Gradients reach every $A$ and $B$, non-zero, equal PEFT's `lora_A`/`lora_B` grads (fp32 tolerance) | max abs diff 0 |
| Non-reentrant checkpointing gives bitwise equal gradients; the wrapped layer runs twice | exact |
| `k_proj` layout trap: transposed A, B, or both, swapped A and B, swapped $d_\text{in}$, $d_\text{out}$ raise, nothing installed | raises |
| Bad keys, mixed dtypes, nesting, unwrapped model raise; leaving `inject` (also by exception) restores base logits bitwise | pass |
| `wrap_targets` idempotent, transparent, atomic on bad targets; export copies views of one buffer | pass |

Negative controls (not committed) replaced the forward with plausible deviations. On a bf16 base
these broke bitwise parity with PEFT: scale applied to $A$ before the matmul (max diff
$1.6 \times 10^{-2}$), delta cast to bf16 before the add ($7.8 \times 10^{-2}$), merged weight
($8.6 \times 10^{-2}$), bf16 factors ($9.4 \times 10^{-2}$). The scale applied to $A x$ before
$B$, `F.linear(F.linear(x, A) * s, B)`, changes only float32 roundings: whole-model diff
$3.8 \times 10^{-6}$ in fp32 and 0 in bf16. It passed the first version of the tests (21 of 21)
and fails three of the current ones: the fp32 tiers of the state-dict parity, the
`from_pretrained` round trip, and the single-layer comparison. D2L's einsum gave the same bits
on CPU, which says nothing about GPU kernels.

## Environment

CPU tier run with torch 2.14.0 (the local cp311 build from the uv cache; Kaggle has 2.8.0),
transformers 4.55.4 (works with torch 2.14.0 on CPU, so no other 4.5x release was needed),
peft 0.17.1, safetensors 0.6.2, accelerate 1.11.0 (installed, not imported). Install with
`uv pip install -e ".[dev,hyper]"`. CI installs `.[dev]` only, so these tests skip there until the
CI job with CPU torch lands.

## Limits

- Parity holds against unmerged PEFT only. `merge_and_unload` or kernels such as Unsloth's round
  differently (risk 4); the decode loader is checked in P4.1.
- Parity on a bf16 base also depends on how the adapter is loaded, which matters for P4.1's
  per-puzzle swap. Line numbers are peft 0.17.1 unless marked; `test_peft_loaders_on_bf16_base`
  pins every row (on a float32 base all of them keep parity, as there is nothing to round):

  | Loader | Bitwise parity | Why |
  | --- | --- | --- |
  | `PeftModel.from_pretrained(base, path)`, defaults | yes | `PeftModel.__init__` creates the adapter and upcasts it (`peft_model.py` 539 to 553, 133); `load_adapter` skips `add_adapter` for an existing name (1304) and copies into float32 (1326; `save_and_load.py` 524) |
  | `peft_model.load_adapter(path, name)`, `name` already loaded | yes | same copy into the existing float32 adapter (1304, 1326; 524): the swap from files |
  | `set_peft_model_state_dict(peft_model, to_peft_state_dict(factors))`, existing name | yes | same copy (524): the swap without files |
  | `from_pretrained` or `load_adapter` with `low_cpu_mem_usage=True` | no, factors rounded to bf16 | `load_state_dict(assign=True)` (`save_and_load.py` 518), then `_move_adapter_to_device_of_base_layer` (519 to 522) casts to bf16 (`tuners_utils.py` 1105) before `_cast_adapter_dtype` upcasts (`peft_model.py` 1400 to 1403) |
  | `load_adapter(path, new_name)` | no, factors rounded to bf16 | `add_adapter` (1318; `inject_adapter`, 1014 to 1016) builds bf16 layers (`layer.py` 249, `tuners_utils.py` 1105) with no upcast; the copy (1326) rounds, the upcast (1400 to 1403) comes after |
  | `from_pretrained(..., autocast_adapter_dtype=False)` | no, bf16 adapter | `_cast_adapter_dtype` returns early (`tuners_utils.py` 366 to 367): rounded factors and a bf16 delta |
  | transformers 4.55.4 `PreTrainedModel.load_adapter(path)` | no, bf16 adapter | `inject_adapter_in_model` and `set_peft_model_state_dict` (`transformers/integrations/peft.py` 244, 270) without `_cast_adapter_dtype` |

  On the test model the rounding rows move the logits by $7.0 \times 10^{-2}$ (rounded factors,
  float32 delta) and $9.4 \times 10^{-2}$ (bf16 adapter).
- One adapter per forward, shared by the whole batch. Batched distinct adapters (D2L's unpacked
  einsum, research note 1.2) are P3.5 work and need their own parity check.
- Plain `nn.Linear` targets under a decoder at `model.model` (any `*ForCausalLM`): export keys
  assume that path. No quantized layers, DoRA, dropout, or `fan_in_fan_out`.
- Wrapping renames base parameters to `....base_layer.weight`, as PEFT does, and PEFT cannot adapt
  a wrapped model: keep the PEFT path on an unwrapped base.
- Backward must run inside the block. Reentrant checkpointing (`use_reentrant=True`, the
  transformers default) is not supported (risk 2). Factor state lives on the modules: one block at
  a time, not thread-safe.
- GPU tier passed (2 Oct, [`docs/experiments/P3.1-01.md`](../experiments/P3.1-01.md)): bf16
  logits bitwise equal on all 9 DEV4 sequences, gradients flowing to every factor. The run:
  `P3.1-01` ([brief](../prerun/P3.1-01.md)), `scripts/inject_parity.py`
  with `configs/inject_parity.yaml` ([runbook](../runbooks/inject_parity.md)), bf16 on one L4 of
  `NvidiaL4`, compares the injected path with `PeftModel.from_pretrained` on DEV4's 9 sequences
  and checks the gradient through `inject` with non-reentrant checkpointing. CPU equality does not
  prove GPU equality, because cuBLAS picks kernels by shape, stride, and alignment. `inject` copies
  factors to fresh contiguous storage, as PEFT's parameters are, to keep that choice the same.
