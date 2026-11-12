# D2L and T2L reading for P3.1: injection, head design, reuse

Author: `researcher` subagent. Date: 2026-09-26. Feeds: P3.1 (`arc26/hyper/inject.py`, `docs/hypernet_design.md`), P3.3, P3.5, P3.6.

## Question

1. D2L: how is the generated LoRA injected into the frozen target; how are per-example adapters batched and packed sequences handled; how is the context read; what is the training loop?
2. T2L: hypernetwork head design (inputs, trunk, per-module heads, init, parameter counts, mapping to PEFT keys).
3. Reuse versus change for our setting; Qwen3 support; PEFT key names; rsLoRA scaling.
4. Licenses and pinned versions.
5. Open risks for P3.1 and P3.5.

Plus: confirm Qwen3-4B projection shapes and recompute the emitted value count.

## Sources read

| Source | Revision | License |
| --- | --- | --- |
| `github.com/SakanaAI/doc-to-lora` (D2L, package `ctx_to_lora`) | `baa85db4d5df9b29d618af432d6ebf28b3ad5a29` (2026-05-25) | MIT, "Copyright (c) 2026 Sakana AI" |
| `github.com/SakanaAI/text-to-lora` (T2L, package `hyper_llm_modulator`) | `8ba77493eb8c732bac41c22ca76117dcacfcf2a4` (2025-06-08, single commit) | Apache-2.0, no NOTICE file |
| `huggingface/peft` tag `v0.15.2` | `3c7b6e7f0252cb18386d86b056a9b100a8160792` | Apache-2.0 |
| `huggingface/transformers` tag `v4.51.3` (Qwen3 files only) | `5f4ecf2d9f867a1255131d2461d75793c0cf1db2` | Apache-2.0 |
| `pytorch/pytorch` tag `v2.6.0` (`stateless.py`, `checkpoint.py`) | `1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340` | BSD-3 |
| `huggingface.co/Qwen/Qwen3-4B/raw/main/config.json` | model repo sha `1cfa9a7208912126459214e8b04321603b3df60c` | |
| arXiv 2602.15902 | abstract only (export API) | |

Not reached: the full D2L paper text (only the abstract was read; paper-level claims below come from code, not the paper). The base model `sorokin/qwen3_4b_grids15_sft139` is on Kaggle, which is off limits; every statement about its vocabulary and config is from `docs/BRIEF.md` and is **unverified**.

Paths below are relative to each repo root. D2L source lives in `src/ctx_to_lora/`.

## Short answer

- **D2L injection** is instance-level monkey patching of `forward` on every targeted PEFT-wrapped `Linear` (PEFT adapters disabled), with per-call $(A, B)$ bound through `functools.partial`, and the low-rank delta computed with `einops.einsum`. No forward hooks, no module replacement, no `torch.func.functional_call`.
- **Per-example adapters**: $A, B$ carry a leading context axis; they are `repeat_interleave`d per query (unpacked) or per token (packed, batch size 1, boundaries from `position_ids` resets). The per-token path is unusable for us at 9.5k tokens (about 1.56 GB per module in fp32, saved for backward).
- **Context**: a separately loaded, frozen copy of the base (optionally 4-bit), per-token hidden states (no pooling), read by a *trained* Perceiver that emits one latent per (layer, module, rank).
- **Training**: HF `Trainer`, top-16 KL to a context-in teacher (main) or CE (NIAH), L1 on generated factors, AdamW fused, cosine with min lr, bf16 plus TF32, `torch.compile`, **no gradient checkpointing**, 8 GPUs.
- **T2L head**: concat(task code, layer embedding, module embedding), MLP trunk, one linear head per module type emitting $r(d_\text{in}+d_\text{out})$, zero head weight with bias set to PEFT's initial $(A, 0)$. Output maps 1:1 onto PEFT keys.
- **Qwen3**: supported by D2L's pins and configs; `q_norm`/`k_norm` need no handling. PEFT rsLoRA gives exactly $\alpha/\sqrt{r} = 8$.
- **Biggest P3.1 risk**: the bf16 bound $<10^{-3}$ effectively demands bitwise-identical logits, so `inject.py` must replay PEFT's exact op sequence, not D2L's einsum. **Biggest P3.5 risk**: activation checkpointing silently corrupts gradients if the injected $(A, B)$ are not in place during the backward recompute.

## 1. D2L

### 1.1 Injection mechanism

- The base is wrapped as `PeftModel(model, lora_config)` (`model_loading.py` 159 to 160), frozen (162 to 163), and PEFT's own LoRA computation is turned off with `base_model.disable_adapter_layers()` (`modeling/hypernet.py` 527). PEFT is kept only to locate target modules and read their shapes (`utils.py` 164 to 203).
- `ModulatedPretrainedModel.patch_lora_forward` (`modeling/hypernet.py` 501 to 521) sets `module.forward = partial(lora_forward_fn, self=module, lora_dropout_p=..., scaling=peft_config.lora_alpha)` on each targeted module, keeping `module.forward_orig` for `reset()` (804 to 813).
- `apply_lora_to_layers` (`modeling/lora_layer.py` 75 to 107), called in every `forward` (`hypernet.py` 761 to 767), re-binds the current step's `A=..., B=..., n_qs=..., tot_q=...` (and `seq_lens`, `tot_len` when packed) into that partial. The binding persists until the next call.
- The patched forward (`lora_layer.py` 14 to 37) calls the frozen base through `torch.nn.Linear.forward(self, x)` (works because PEFT's `BaseTunerLayer` exposes `weight` and `bias` properties, `peft/tuners/tuners_utils.py` 682, 698), then:

  ```python
  delta_x = einsum(A, delta_x, "tot_q r d_in, tot_q s_len d_in -> tot_q s_len r")
  delta_x = einsum(B, delta_x, "tot_q r d_out, tot_q s_len r -> tot_q s_len d_out")
  return (base_out + delta_x * scaling).to(base_out.dtype)
  ```

- Layout: D2L stores $B$ as `[r, d_out]` (rank-major), not PEFT's `[d_out, r]`.
- Scaling: `scaling=peft_config.lora_alpha` is used directly, with `lora_alpha = 2 r^{3/2}` (`model_loading.py` 178), so $s_\text{D2L} = 2r^{3/2}$ (45.25 at $r=8$). This is neither $\alpha/r$ nor $\alpha/\sqrt{r}$. D2L never exports generated adapters to PEFT in its main path; `generated_lora_to_state_dict` (`utils.py` 206 to 226) is defined but unused.
- Rank grows with chunks: `combine_lora` (`modeling/lora_merger.py` 15 to 78) concatenates each context chunk's rank-$r$ factors along the rank axis and appends a learned, input-independent "bias" LoRA (`bias_A` random, `bias_B` zero; `hypernet.py` 276 to 293) as $r$ extra ranks. Effective rank is $(n_\text{chunks}+1)\,r$.

### 1.2 Per-example adapters and sequence packing

- Generated factors are `{module: {A: [n_ctx, n_layers, r_eff, d_in], B: [n_ctx, n_layers, r_eff, d_out]}}`; `apply_lora_to_layers` slices `[:, layer_idx]` (`lora_layer.py` 101 to 102). `n_queries[n_ctx]` gives the number of query sequences per context.
- **Unpacked** (`lora_forward`, 14 to 37): `repeat_interleave(n_qs)` expands per-context factors to per-sequence `[tot_q, r, d]`; the einsum is batched over sequences.
- **Packed** (`lora_forward_packed`, 40 to 72): batch size 1; all queries concatenated. Sequence lengths come from resets in `position_ids` (`lora_layer.py` 83 to 90). Factors are expanded twice, to per-sequence and then to **per-token** `[tot_len, r, d]`. The attention side relies on transformers' flash-attention varlen path from `position_ids` (collator `data/collator.py` 13 to 62, forced batch size 1 at `train.py` 350 to 351).
- Packing is mandatory in training (`train.py` 105 to 107); main runs use 4096-token packs (`scripts/main_exp/1-train.sh`).

### 1.3 How D2L reads its context

- **Encoder**: `_init_model` loads a **second instance** of the base (or of `ctx_encoder_model_name_or_path`) via `get_model(..., requires_grad=False)` (`hypernet.py` 535 to 552). It is frozen (checked at `train.py` 237 to 240; `@torch.no_grad` forwards in `modeling/ctx_encoder.py`) and optionally 4-bit NF4 (`--quantize_ctx_encoder=True` in the main script; `model_loading.py` 136 to 148).
- **Layer choice** (`ctx_encoder.py` 53 to 158):
  - `early_exit`: the first `layer_idx` blocks, default $\lfloor L/4 \rfloor$ (`train.py` 189 to 196; 9 for 36 layers). Because the HF model applies its final RMSNorm after the truncated stack, the features are `norm(h_k)`.
  - `per_layer_activations` (main experiments): all `hidden_states` except the last block's output, one feature map per target layer ("layer to layer").
  - `embed_only`: embeddings only.
- **Per token, not pooled**: token features go to a trained Idefics2 Perceiver resampler (`modeling/aggregator.py` 78 to 206; `modeling/idefics2.py`, which is HF Apache-2.0 code). Its encoder has `n_latent_queries` latents (8 main, 208 NIAH) and 8 to 9 blocks. A one-block decoder then emits $n_\text{layers} \cdot n_\text{modules} \cdot r$ query vectors (per-rank generation, 103 to 106). In layer-to-layer mode the same perceiver runs once per layer.
- **Heads** (`hypernet.py` 215 to 437): ResMLP blocks (per-layer weights when `per_layer_processing`), then L2 normalization of each latent (422 to 424). Then a head `EinMix` with weight shape `n_layers n_modules d_latent d_lora`, which emits one row of length $d_\text{in}+d_\text{out}$ per rank (313 to 333). Then per-(layer, rank) learnable scalers: `scaler_A` initialized to 1 and `scaler_B` to 0 (295 to 306, applied at 368 to 370).
- **Init**: head weight is $\mathcal{N}\big(0,\ (0.5/\sqrt{d_\text{latent} + d_\text{lora}\,r})^2\big)$ (`_bias_hyper_init`, 570 to 587). Because `scaler_B` and `bias_B` start at zero, the emitted $B$ is zero at step 0 and the modulated model equals the base.
- **Context is not in the query input**: `add_ctx_to_chat=False` for the modulated model (`train.py` 261). The context reaches the base only through the adapter. Our Q6 keeps demos in context.

### 1.4 Training loop

| Item | D2L | Reference |
| --- | --- | --- |
| Framework | HF `Trainer` subclasses | `trainer.py` 89 to 455 |
| Objective (main) | $-\sum_{k \in \text{top-16}} p_\text{teacher}(k) \log q(k)$ on response tokens; teacher top-16 logprobs from the base with the document in context (self-generated QA, vLLM `logprobs=16`) | `trainer.py` 138 to 239; `data/self_generate_qa.py` 267, 347 |
| Objective (NIAH) | token CE | `trainer.py` 272 to 380 |
| Normalization | per-context mean (`use_per_ctx_average_loss`) | `trainer.py` 14 to 86 |
| Regularizer | L1 on generated $A, B$, coef 0.1 (main), 0.5 (NIAH) | `trainer.py` 208 to 218; `scripts/main_exp/1-train.sh` |
| Optimizer | `adamw_torch_fused`, lr $4\times10^{-5}$, wd 0.01, betas (0.9, 0.999); no decay on scalers, biases, norms, perceiver latents | `configs.py` 198 to 221; `trainer.py` 383 to 397 |
| Schedule | `cosine_with_min_lr`, min $10^{-7}$, 100 warmup steps, 80k steps | `configs.py` 222 to 233; `train.py` 136 to 140 |
| Batch | 1 packed sequence (4096 tokens) per GPU, grad accumulation 8, 8 GPUs | `1-train.sh`; `accelerate_config.yaml` |
| Precision | base bf16; hypernet parameters fp32 (`hypernet.py` 529 to 531), aggregator under bf16 autocast (400); `bf16=True`, `tf32=True` | `configs.py` 136 to 143; `train.py` 415 to 416 |
| Compile | `hypernet.compile` and `base_model.compile(fullgraph=True, mode="max-autotune")` | `train.py` 242, 373 |
| Gradient checkpointing | **none** (no flag anywhere; HF default off) | repo-wide grep |
| NEFTune | default $\alpha = 5$ in D2L's `TrainingArguments` (Qwen NIAH script sets 0) | `configs.py` 194 to 197 |
| Released scope | `down_proj` only, $r = 8$ | all `scripts/*/train*.sh` |

## 2. T2L hypernetwork head

All references are to `src/hyper_llm_modulator/hyper_modulator.py` unless noted.

- **Task input**: a frozen text-embedding vector per task description (`Alibaba-NLP/gte-large-en-v1.5`, CLS pooling, fp32; or the base with last-token pooling), precomputed. `TaskEncoder` = `Linear(E, latent/2)` then `LayerNorm` (71 to 86).
- **Layer embedding** = `Embedding(n_layers, latent/4)` then `LayerNorm`. **Module embedding** = `Embedding(n_modules, latent/4)` then `LayerNorm` (392 to 401). The input is the concatenation, width $d = \text{latent}$ (403, 599).
- **Trunk**, all with dropout 0.05:
  - `mixer` $d \to 4d \to d$, SiLU (405 to 413).
  - Two residual blocks with pre-LN, $d \to 4d \to d$, SiLU (196 to 216, 414 to 427).
  - `mlp3` with LN, $d \to 4d \to$ `head_in_size`, SiLU (429 to 436).
- **Heads**: one `nn.Linear(head_in_size, r (d_in + d_out))` per module type, shared across layers (474 to 502). The output is split into $A$ `[r, d_in]` and $B$ `[r, d_out]`, then $B$ is transposed to PEFT's `[d_out, r]` (684 to 719). Evaluation is batched over layers, one module type at a time (589 to 682).
- **Init** (`create_hypermod`, 877 to 939): for SFT training, `zero_init_head=True` (930), `head_use_bias=True` (933 to 934) and `match_lora_init=True` (921). Head weight is zeroed and head bias set to `cat(flatten(PEFT-init lora_A), flatten(lora_B = 0))` (533 to 541, bias-HyperInit). At step 0 every layer gets the same kaiming-uniform $A$ and $B = 0$, so the output equals the base, and $\partial L / \partial W_B \neq 0$ from the first step.
- **Scaling**: PEFT `use_rslora=True`, `lora_alpha = 2r`, $r = 8$, dropout 0.05 (`utils/model_loading.py` 108 to 127). The hook uses `hypermod.scaling` $= \alpha/\sqrt{r}$ (311 to 315), so training and export agree.
- **Injection**: `register_forward_hook` on each targeted module, which adds `bmm(bmm(x, A), B) * scaling` with per-token `repeat_interleave` (`hooks.py` 150 to 179). Hooks are removed after the forward and before `backward` (`sft_trainer.py` 166).
- **Export to PEFT**: `gen_lora` (721 to 754) writes `lora_state_dict[name]` for names from `get_peft_model_state_dict(model)` (`utils/lora_formatting.py` 214 to 232), with $A$ `[r, d_in]` and $B$ `[d_out, r]`. This is the pattern to copy.
- **Training** (`scripts/train_custom_sft.py`, `sft_trainer.py`): AdamW, lr $2.5\times10^{-5}$, wd $10^{-2}$, linear warmup (20%) and linear decay, clip 1.0, label smoothing 0.1, L2 on generated factors $10^{-3}$, NEFTune 5, Accelerate bf16, no checkpointing. Released config targets `q_proj`, `v_proj` (`configs/hyper_lora_decontam_lol_tasks.yaml` 664 to 666).
- **Parameter counts**, computed from the code (trainable only). The head term $H_\text{in} \cdot \sum_m r(d_\text{in}^m + d_\text{out}^m)$ dominates.

| Config | Trunk | Heads | Total |
| --- | --- | --- | --- |
| T2L defaults, Mistral-7B, $r=8$, q/v, latent 128, $H_\text{in}=512$, $E=1024$ | 0.79M | 54.6M | 55.4M |
| T2L Llama script, latent 512, $H_\text{in}=2048$ | 11.8M | 218.2M | 230.0M |
| Ours, Qwen3-4B, $r=16$, qkvo, $E=2560$, latent 512, $H_\text{in}=512$ | 9.1M | 168.1M | 177.2M (over the 150M budget) |
| Ours, latent 512, $H_\text{in}=384$ | 8.8M | 126.2M | 135.0M |
| Ours, latent 512, $H_\text{in}=256$ | 8.5M | 84.2M | 92.8M |

A D2L-style per-layer head at latent 512 would need $36 \cdot 4 \cdot 512 \cdot 6656 \approx 491\text{M}$ parameters, so it is excluded.

## 3. Qwen3-4B shapes, emitted count, PEFT keys, rsLoRA

- From the HF config (`Qwen/Qwen3-4B`, sha `1cfa9a72`): `hidden_size` 2560, `num_hidden_layers` 36, `num_attention_heads` 32, `num_key_value_heads` 8, `head_dim` 128 (explicit, not $2560/32$), `intermediate_size` 9728, `attention_bias` false, `tie_word_embeddings` true, `transformers_version` 4.51.0.
- Module definitions are in `transformers/models/qwen3/modeling_qwen3.py` 183 to 196: `q_proj` $2560 \to 4096$, `k_proj` and `v_proj` $2560 \to 1024$, `o_proj` $4096 \to 2560$, all bias-free `nn.Linear`.
- `q_norm` and `k_norm` are RMSNorms over `head_dim` applied to the projection outputs (217 to 218). A LoRA on `q_proj`/`k_proj` is therefore normalized per head along with the base output. This is identical in PEFT and in any injected path, so no special handling is needed. Side effect: q/k deltas can rotate per-head queries and keys but cannot rescale them.
- Emitted values at $r = 16$, attention only:

$$
N = L \cdot r \sum_{m} \big(d_\text{in}^m + d_\text{out}^m\big) = 36 \cdot 16 \cdot (6656 + 3584 + 3584 + 6656) = 36 \cdot 327{,}680 = 11{,}796{,}480 \approx 11.8\text{M}.
$$

  Adding `gate_proj`/`up_proj`/`down_proj` contributes $36 \cdot 16 \cdot 3 \cdot 12288 = 21{,}233{,}664$, for a total of $33{,}030{,}144 \approx 33.0\text{M}$. $r = 32$ attention only gives 23.6M. The brief's numbers are confirmed.
- **The fine-tuned base is assumed to share this architecture except for vocabulary (16 rows in embedding and `lm_head`, per the brief): unverified.**
- **PEFT keys** (v0.15.2): parameters live at `base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{A,B}.default.weight`. `get_peft_model_state_dict` strips the adapter name (`peft/utils/save_and_load.py` 268), giving the on-disk and export form:

  ```
  base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight   # [16, 2560]
  base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight   # [4096, 16]
  ...k_proj / v_proj: lora_A [16, 2560], lora_B [1024, 16]
  ...o_proj:          lora_A [16, 4096], lora_B [2560, 16]
  ```

  `set_peft_model_state_dict` re-inserts the adapter name (`save_and_load.py` 300 to 317).
- **rsLoRA**: `update_layer` sets `scaling = lora_alpha / math.sqrt(r)` when `use_rslora` (`peft/tuners/lora/layer.py` 147 to 150). With $\alpha = 32$ and $r = 16$ this gives $s = 32/\sqrt{16} = 8$, exactly $\alpha/\sqrt{r}$. Export config: `{"r": 16, "lora_alpha": 32, "use_rslora": true, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM"}`.
- **PEFT forward order** (`layer.py` 699 to 745): `result = base_layer(x)`; `x = x.to(lora_A.weight.dtype)`; `result = result + lora_B(lora_A(dropout(x))) * scaling`; `result.to(result_dtype)`. By default PEFT keeps adapter weights in fp32 even on a bf16 base (`autocast_adapter_dtype`, `tuners_utils.py` 321 to 357), so the delta is computed in fp32 and cast to bf16 once.

## 4. What to reuse, what to change

| Area | Reuse | Change for us |
| --- | --- | --- |
| Injection | D2L's idea: target `Linear`s return base output plus $s\,B(A x)$ with externally supplied $(A, B)$; D2L's unpacked batched einsum for batch size above 1 with distinct adapters | Replay PEFT's exact op sequence (`F.linear` with A, then with B, fp32 delta, multiply by $s$, add, cast) for the single-adapter path. Store factors in PEFT layout (A `[r, d_in]`, B `[d_out, r]`) so export is a rename. Keep $(A, B)$ installed until `backward` returns. No per-token `repeat_interleave`. No `torch.compile` in P3.1. |
| Scaling | PEFT rsLoRA | $s = \alpha/\sqrt{r} = 8$ as one constant shared by `inject.py` and the export config. Not D2L's $2r^{3/2}$, and no rank growth through `combine_lora`. |
| Head | T2L: concat(z code, layer embedding, module embedding), MLP trunk, per-module-type heads batched over 36 layers, bias-HyperInit | $H_\text{in} \le 384$ to meet the 150M budget. Optionally D2L's L2 normalization of the head input and its zero-initialized per-(layer, rank) $B$ gate as the zero-init alternative. |
| Encoder | Frozen-base features (both repos) | Mean-pooled layer-24 output-grid embeddings from the same loaded base (Q1), cached; no second model copy, no Perceiver. |
| Objective | D2L `CrossEntropyTrainer` structure; L1 (D2L) or L2 (T2L) penalty on generated factors as an option | Task CE on output-grid tokens with demos in context (Q6). With a 16-token vocabulary the optional KL term uses the full distribution; D2L's top-16 truncation is the whole vocabulary here. |
| Optimizer | AdamW, cosine with min lr, warmup, clip 1.0, no decay on norms, biases, scalers | lr and warmup retuned in P3.5. Disable NEFTune: noise on 16 grid-token embeddings has no validated benefit. |
| Memory | Nothing (D2L and T2L run without checkpointing on large GPUs) | Non-reentrant activation checkpointing per decoder layer, FA2 or SDPA, one episode per step, grad accumulation 8. |
| Data and tokenizer | Nothing (chat templates, text tokenizers, vLLM self-generation) | Notebook tokenization (P0.1). |

**Qwen3 support in D2L**: yes. D2L pins transformers 4.51.3, which includes Qwen3, and ships `configs/main_exp/qwen/`, `scripts/niah/train_qwen3_4b.sh` and `chat_templates/Qwen/Qwen3-4B-Instruct-2507.jinja`. Its module lookup `self_attn.{q,k,v,o}_proj` / `mlp.{gate,up,down}_proj` matches Qwen3 (`lora_layer.py` 95 to 100), and `get_layers` recurses through `PeftModel` to `model.layers` (`utils.py` 41 to 44). Its text data pipeline does not transfer to the 16-token base. T2L's pin (transformers 4.46.2) cannot load Qwen3, so only T2L's head code is usable.

## 5. Licenses and pinned versions

- **D2L, MIT**: copying into a private or later public repo is permitted if the copyright and permission notice are kept with substantial portions. `modeling/idefics2.py` inside D2L is HF Apache-2.0 code; we do not need it.
- **T2L, Apache-2.0**: copying is permitted with a copy of the license, retained attribution, and a prominent change notice in modified files (Sec. 4(a) to (c)). There is no NOTICE file, so 4(d) does not apply. The license includes a patent grant and is compatible with a later MIT or Apache-2.0 release of this repo.
- Recommended practice: a file header such as `# Adapted from SakanaAI/text-to-lora@8ba7749 (Apache-2.0), modified` and a `THIRD_PARTY_LICENSES` file. The expected copy volume is small (the T2L head, about 80 lines; the injection forward, about 15 lines). This is not legal advice. The competition's open-source rule is an A5/G5 item.

| Package | D2L | T2L |
| --- | --- | --- |
| python | 3.10 (`install.sh` 3) | 3.10 |
| torch | 2.6.0 cu124 (`install.sh` 4; `uv.lock` 5479) | 2.4.0 (`uv.lock` 4379) |
| transformers | 4.51.3 (`pyproject.toml` 9; `uv.lock` 5638) | 4.46.2 (`pyproject.toml` 9; `uv.lock` 4499) |
| peft | 0.15.2 (`uv.lock` 3883; unpinned in pyproject) | 0.15.2 (`uv.lock` 2820) |
| flash-attn | 2.7.4.post1, cu12 torch2.6 wheel (`install.sh` 7) | 2.6.3, cu123 torch2.3 wheel (README 30; mismatches its own torch 2.4.0 lock) |
| accelerate | 1.6.0 | 1.6.0 |
| other | deepspeed 0.17.1, vllm 0.8.5.post1, tokenizers 0.21.0 | vllm 0.5.4 |

Qwen3 requires transformers 4.51.0 or later. The Kaggle notebook's own torch, transformers, peft and flash-attn versions are **unverified**; P0.1 should record them, because the P3.1 parity must hold on the notebook's stack.

## 6. Open risks for P3.1 and P3.5

1. **The bf16 bound is a bitwise-equality requirement.** Logits are bf16 (`modeling_qwen3.py` 866, no upcast). For $|z| \in [2^k, 2^{k+1})$ the bf16 spacing is $2^{k-7}$, so for any logit with $|z| \ge 0.25$ a single-ULP difference already exceeds $10^{-3}$.
   - D2L's einsum/bmm path, or any change in kernel, shape, stride or dtype relative to PEFT, will flip some bf16 roundings in the residual stream, and these compound over 36 layers.
   - Mitigation: replay PEFT's sequence exactly (fp32 adapter weights, `x.to(fp32)`, `F.linear` with A then with B, multiply by $s$, add to the bf16 base output, cast), use `.contiguous()` factors, run both paths with no autocast, dropout 0 or `eval()`, identical TF32 and attention settings, and bs 1 per DEV4 sequence to avoid padding differences.
   - Proposed test tiers: (a) fp32 base, max abs diff $< 10^{-3}$ (expected about $10^{-5}$); (b) bf16, expect exact equality and report the max diff. If exact equality fails on the L4, fall back to "argmax identical and max diff $\le 2$ ULP at the logit scale", as a human decision at the P3.1 review.
2. **Checkpointing silently corrupts gradients** if $(A, B)$ are not in place during the backward recompute.
   - `torch.func.functional_call` restores parameters on return (`torch/nn/utils/stateless.py` 297 to 300), and T2L removes hooks before `backward`.
   - Non-reentrant checkpointing only checks shape, dtype and device of recomputed tensors (`torch/utils/checkpoint.py` 1032 to 1043), not values, so a reverted adapter with the same shapes passes silently.
   - HF's default `use_reentrant=True` (`modeling_utils.py` 3119 to 3120) detaches everything when inputs do not require grad (`checkpoint.py` 85 to 89). It also nests backward passes through the hypernet graph (**expected failure mode, not tested here**).
   - Mitigation: `gradient_checkpointing_enable({"use_reentrant": False})`; a context manager that keeps $(A, B)$ installed across `loss.backward()`; a test that compares $\nabla_{A,B}$ with checkpointing on and off, and against PEFT's own `lora_A/B.grad` under the exported weights.
3. **Memory at 9.5k tokens on 24 GB** (estimate, **unverified until P3.6**). Base bf16 non-embedding weights: about 3.63B parameters, 7.3 GB (the 16-row vocabulary makes embeddings and logits negligible). H at 150M parameters with fp32 AdamW: 2.4 GB. Checkpoint boundaries: $36 \cdot 9500 \cdot 2560 \cdot 2\,\text{B} \approx 1.75$ GB. One recomputed layer with fp32 LoRA inputs: about 1.2 to 1.5 GB. Total about 13 GB. Without checkpointing, saved activations exceed 40 GB. D2L's per-token factor expansion would add about 1.56 GB per module (q_proj, fp32) and must not be copied.
4. **Stack mismatch with the notebook** (**unverified**): if the notebook's TTT/decode uses merged adapters (`merge_and_unload`) or non-PEFT kernels (for example Unsloth), exported weights will not reproduce injected logits to $10^{-3}$, because merging rounds $W + sBA$ to bf16. P3.1 parity is against unmerged PEFT; P4.1 must check the decode loader separately.
5. **Scope signal**: D2L's released adapters are MLP-only (`down_proj`, $r = 8$), and T2L's are q/v. Attention-only rank 16 is not validated by either repo for our task; the P1.3 gate decides.
6. **Offline export detail**: if `PeftModel.save_pretrained` is used, pass `save_embedding_layers=False`. The `"auto"` mode compares vocabulary size against the base config and may probe the Hub (`save_and_load.py` 228 to 256).

## 7. Suggested `inject.py` surface (for the P3.1 brief)

- `LoRAFactors = dict[str, tuple[Tensor, Tensor]]`, keyed by `layers.{i}.self_attn.{m}`, with A `[r, d_in]` and B `[d_out, r]` (PEFT layout).
- `wrap_targets(model, target_modules)` installs thin `InjectableLinear` wrappers once. `inject(model, factors, scaling)` is a context manager that sets factors for the duration of forward plus backward.
- `to_peft_state_dict(factors) -> dict[str, Tensor]` produces the Section 3 keys, and `peft_config(r, alpha)` produces the Section 3 JSON.
- Tests:
  - zero-$B$ gives bitwise base logits;
  - fp32 and bf16 parity tiers as in risk 1;
  - gradient non-null and matching PEFT `.grad`;
  - checkpointing on versus off;
  - a layout trap on `k_proj` ($d_\text{out} \ne d_\text{in}$) to catch transposes.
