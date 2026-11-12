# Models for the code-writing agent on 4x L4: Qwen3.8-27B builds and small fan-out models

Author: `researcher` subagent. Date: 2026-10-09. Feeds: the human's new-solver idea (a code-writing
agent after Prime Intellect's Prime Agent: a persistent Python REPL as the only tool, sub-agents as
function calls, a growing library of verified skills, possibly many small agents feeding one
memory). Builds on `docs/research/leaders_and_base_models_2026.md` (same day, "yesterday's survey"
below).

## Question

Which models, and which quantized builds of them, are the best candidates for the agent's model on
4x L4 offline (sm_89, 24 GB each, FP8 yes, FP4 no, 12 h for about 240 tasks), for (1) a strong main
solver and (2) small fast sub-agents for fan-out?

## Sources and conventions

| Source | Revision or read time | Kind |
| --- | --- | --- |
| Hugging Face Hub API `https://huggingface.co/api/models/<id>?blobs=true` (file sizes, licenses, `downloads` = 30-day count unless marked all-time), `config.json` and `README.md` at the revisions given inline (first 8 hex digits) | read 2026-10-09 | primary for file facts; benchmark tables in cards are the publisher's own claims |
| arcprize.org `/media/data/evaluations.json` and `/media/data/models.json` (sha256 prefixes `9a568172a0b25495`, `885ea7b52e8f70ad`), `/results/alibaba-qwen3-8-27b` | read 2026-10-09 | primary (organizers) |
| vLLM source at tag `v0.31.0` (commit `db9527a4`, 2026-10-02): `docs/features/quantization/README.md`, `vllm/model_executor/kernels/linear/__init__.py`, `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu`, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `vllm/v1/attention/backends/flash_attn.py` and `triton_attn.py`, `vllm/config/cache.py`, `vllm/reasoning/` | sparse clone in scratch, 2026-10-09 | primary (code) |
| SGLang source at tag `v0.5.21` (commit `e00930c5`, 2026-09-29): `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py` | sparse clone in scratch, 2026-10-09 | primary (code, docs) |
| vLLM GitHub issues #54690, #54691, #55533, #55766, #58241, #58807, #58894, #60174 (body and comments through the GitHub API) | read 2026-10-09 | user reports; open unless stated |
| PyPI JSON: vllm, sglang, sglang-kernel, torch, triton, flashinfer-python, flashinfer-cubin, nvidia-* CUDA 13 wheels, kaggle-vllm | read 2026-10-09 | primary (package metadata) |
| vLLM recipe `https://recipes.vllm.ai/Qwen/Qwen3.8-27B` | read 2026-10-09 | primary (vLLM project) |
| HyperQwen, `github.com/syv-ai/HyperQwen` (formerly `qwen38-27b-rtx3090`), commit `3acb93f7` (2026-10-08): `README.md`, `docs/multi-gpu.md`, `docs/wsl2-4090.md`, `docs/gotchas.md`, `docs/third-party-checkpoints.md` | clone in scratch | third party (RTX 3090 and 4090 measurements) |
| `github.com/nicosuter/Qwen3.8-27B-AWQ` `results/2026-08-23-published-quants/README.md`, commit `dd050a1e` (2026-08-24) | read 2026-10-09 | third party (paired evaluations) |
| `https://vramcalculator.com/qwen3-8-27b-speed/` (dated 2026-10-03, updated 2026-10-04) | read 2026-10-09 | third party (L4 llama.cpp runs) |
| NVIDIA L4 product page `https://www.nvidia.com/en-us/data-center/l4/` | read 2026-10-09 | primary (vendor) |
| HF blog `https://huggingface.co/blog/kaggle-integration` (2025-05-14) | read 2026-10-09 | primary (HF and Kaggle) |
| Kaggle public pages: titles and search snippets only (page bodies are not readable by a plain fetch); no internal Kaggle endpoints used | 2026-10-09 | contents **to check** |
| This repo: `docs/experiments/P0.1.md`, `P2.0-01.md`, `P2.1-03.md` (Kaggle L4 driver), `K0.md` (image), `docs/research/kaggle_first_runs.md`, yesterday's survey (forum statements are cited as quoted there) | worktree `wt-W40-orch` | project |

Units: GB = $10^9$ bytes (HF API sizes), GiB = $2^{30}$ bytes. "Estimate" marks my own arithmetic
from stated assumptions. Third-party measurements and forum statements that carry a conclusion are
marked **unverified**.

## Short answer

- **Main solver: Qwen3.8-27B.** It is still the only open model that fits 4x L4 with a verified
  ARC-AGI-2 score above 20%: 42.4% semi-private at XHigh, 22.8% at Low (arcprize). Build choice,
  to settle by an A/B in the first GPU run:
  - **`RedHatAI/Qwen3.8-27B-INT4`** (rev `91bd022d`, Apache 2.0): W4A16 g128 with static FP8 KV
    scales and a BF16 MTP head, 19.47 GB, so it fits one 20 GB notebook output. Vendor recovery
    against BF16 is 98.5% to 101% on six suites, including AIME 2025 at 98.7% (long reasoning).
    Estimate: about 530 tok/s decode on 4x L4 at a 32k mean context (vLLM 0.31.0, TP=4, no
    speculation), about 22M tokens in 12 h.
  - **`Qwen/Qwen3.8-27B-FP8`** (rev `017b9c7a`, official, Apache 2.0): 30.89 GB, too big for one
    output (needs a public Kaggle mirror, two datasets, or the Hugging Face reference route).
    Qwen calls it "nearly identical" to BF16. On vLLM it runs on L4 as weight-only FP8 (Marlin)
    because CUTLASS block FP8 needs SM90. Estimate: about 420 tok/s at 32k; a participant's
    full-run estimate uses about 350 tok/s on 4x L4 (**unverified**).
- **Fan-out: `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`** (rev `2367e865`, MIT): Qwen3.5-9B
  fine-tuned by Xiaomi on 77.4B tokens of agentic coding data; SWE-bench Pro 44.6 against 32.0
  for its base, Terminal Bench 2.1 37.1 against 27.0 (Xiaomi). 18.85 GB BF16 on disk, served with
  online FP8 (about 11 GB), one replica per L4: estimate 730 tok/s total at 32k contexts, 2,000 at
  8k. Same architecture family and tokenizer as the 27B. Control: `Qwen/Qwen3.5-9B`; lighter
  option: `Qwen/Qwen3.5-4B` (9.34 GB). No small model has any ARC evidence.
- **Fallback:** the official FP8 27B on SGLang 0.5.21 (TP=4; Triton W8A8 block FP8 on sm_89),
  the configuration a participant reports using in two scored Kaggle submissions
  (**unverified**).
- **`empero-ai/Qwen3.8-4B-Distill` is not a Qwen release:** a third-party full SFT of
  Qwen3.5-4B on about 45k traces from Qwen3.8-2.4T-A95B (data private); its only evals are GSM8K
  (down from 0.850 to 0.785) and MMLU-CoT; no code, tool-use or ARC results. Not shortlisted.
- **Biggest risks:** (1) budget: XHigh spends about 175k output tokens per ARC-AGI-2 task
  (derived from arcprize cost), and 12 h on 4x L4 yields about 17M to 22M tokens at 32k contexts,
  so about 100 to 125 tasks at that rate; contexts must stay short and runs must stop early.
  (2) Open vLLM bugs for hybrid DeltaNet models: MTP collapses the batch to about 3 sequences on
  24 GB cards (#55533); NaN logits after prefix-cache hits with speculation (#55766); align-mode
  prefix caching can drop to 0% hits when several long conversations alternate (HyperQwen).
  (3) Offline install of a torch 2.13 / CUDA 13.0 stack (about 3.5 GB of wheels) with FlashInfer
  JIT. (4) The INT4 effect on ARC is unmeasured. (5) Small models are unproven on ARC.

## 1. Hardware and memory arithmetic

L4 (NVIDIA product page): 24 GB GDDR6 at 300 GB/s; BF16 tensor 242 TFLOPS and FP8 485 TFLOPS
with sparsity, half of that dense; no FP4; PCIe Gen4 x16, no NVLink; 72 W TDP. On Kaggle each L4
shows 23 GB and the driver is 580.159.04 or 580.178.04 (`P0.1.md`, `P2.0-01.md`, `P2.1-03.md`).

KV cache and DeltaNet state per sequence, from each `config.json`. KV bytes per token are
$2 \cdot L_{full} \cdot H_{kv} \cdot d_{head} \cdot b$; the Gated DeltaNet state is
$L_{lin} \cdot H_v \cdot 128 \cdot 128 \cdot b$ per sequence and does not grow with length (the
conv state adds under 3 MiB).

| Model (config rev) | Full-attention layers / KV heads / head dim | KV per token BF16 / FP8 | KV per 64k tokens BF16 / FP8 | DeltaNet state fp32 / bf16 |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B (`1d4bf0f2`), Qwen3.6-27B (`6a9e13bd`) | 16 / 4 / 256 (48 DeltaNet layers, 48 value heads) | 64 / 32 KiB | 4.00 / 2.00 GiB | 144 / 72 MiB |
| Qwen3.6-35B-A3B (`995ad96e`) | 10 / 2 / 256 (30 DeltaNet, 32 value heads) | 20 / 10 KiB | 1.25 / 0.62 GiB | 60 / 30 MiB |
| Qwen3.5-9B (`c2022362`), Qwen3.5-4B (`851bf6e8`), their distills | 8 / 4 / 256 (24 DeltaNet, 32 value heads) | 32 / 16 KiB | 2.00 / 1.00 GiB | 48 / 24 MiB |
| Qwen3.5-2B (`15852e8c`) | 6 / 2 / 256 | 12 / 6 KiB | 0.75 / 0.38 GiB | 18 / 9 MiB |
| gpt-oss-20b (`6cee5e81`) | 12 full + 12 sliding-window (128) / 8 / 64 | 24 / 12 KiB | 1.50 / 0.75 GiB | none |
| MiMo-7B-RL (`6299b5a2`) | 36 / 8 / 128 (32,768 native context) | 144 / 72 KiB | 9.0 / 4.5 GiB | none |

The SGLang cookbook states the same geometry for the 27B: 153.9 MB per state slot at fp32 (with
conv state) and 32.8 KB of FP8 KV per token. Engines hold more than one state per request: SGLang
reserves 3 to 5 slots per running request with its radix cache (default `extra_buffer` = 5) plus
draft slots under speculation (cookbook, "Mamba ratio calculator"); vLLM's align-mode prefix
caching keeps snapshots at block boundaries, and MTP adds $1+k$ blocks per DeltaNet group
(#55533 comments). The state is also read and written at every decode step (about 300 MB per
sequence for the 27B, split across TP ranks), a bandwidth cost equal to about 9k tokens of FP8 KV.

## 2. Qwen3.8-27B

### 2.1 Facts and evidence

- **Architecture** (config `1d4bf0f2`): `Qwen3_5ForConditionalGeneration`, 64 layers = 16 x
  (3 Gated DeltaNet + 1 gated attention), hidden 5120, FFN 17,408, vocabulary 248,320 with untied
  head, one MTP layer, a 27-block vision tower (about 0.46B params, about 0.9 GB BF16), 262,144
  native context. 27.78B parameters, 55.6 GB in BF16. Apache 2.0. Created 2026-08-05; 6.78M
  downloads.
- **ARC** (arcprize `evaluations.json`; tested 2026-08-14 through "dedicated inference on
  Baseten", precision not stated; results page):

  | Effort | ARC-AGI-1 semi-private / public eval | ARC-AGI-2 semi-private / public eval | Cost per ARC-AGI-2 task |
  | --- | ---: | ---: | ---: |
  | XHigh | 87.5 / 87.5 | 42.4 / 38.8 | $0.447 |
  | Low | 69.2 / 75.8 | 22.8 / 19.6 | $0.238 |
  | Medium | 68.7 / 76.3 | 13.2 / 17.1 | $0.286 |
  | None | 34.0 / 42.9 | 1.5 / 2.5 | $0.059 |

  The results page explains the Medium dip: the chat template adds effort-specific instructions
  for low and xhigh but not for medium. Prices used: $0.425 per million input and $2.55 per
  million output tokens. Derived (**estimate**): $0.447 / $2.55 per million is about 175k output
  tokens per ARC-AGI-2 task at XHigh, all attempts included (input cost is a fraction of a cent);
  Low is about 93k. With pass@2 and more than one test input on some tasks, that is about 60k to
  70k per generation, consistent with the forum's "median near 64k" (**unverified**).
- **Code and agent benchmarks** (Qwen card `1d4bf0f2`, self-reported): LiveCodeBench v6 90.3;
  SWE-bench Pro 61.7 (Claude Code harness); Terminal Bench 2.1 73.0; NL2Repo 42.3; GPQA Diamond
  89.2. Independent: nicosuter's runs of the official FP8 at XHigh on H200 give LiveCodeBench v6
  87.71, BFCL v4 81.27, GPQA 89.90. BottleCap's base runs (H200, vLLM 0.29, XHigh) give
  LiveCodeBench v6 91.1 at a mean of 28,395 thinking tokens per problem, GPQA 89.9 at 12,772, and
  Terminal-Bench 2.1 75.8 at 72,871 per episode (`bottlecapai/ThinkingCap-Qwen3.8-27B` card).
- **Settings** (card): thinking on by default, `temperature=1.0, top_p=0.95, top_k=20`;
  `reasoning_effort` xhigh (default), medium, low; `preserve_thinking` on by default (keeps old
  thinking in context, which helps prefix caching but grows the context).

### 2.2 Quantized builds on Hugging Face

| Repo (rev, 30-day downloads) | Format and what is quantized | Size | Quality evidence | sm_89 path | One 20 GB output? |
| --- | --- | ---: | --- | --- | --- |
| `Qwen/Qwen3.8-27B-FP8` (`017b9c7a`, 4.54M) | FP8 E4M3, 128x128 blocks, dynamic activations; lm_head, embeddings, DeltaNet `in_proj_a/b`, conv, norms, vision in BF16 | 30.89 GB | Qwen: "nearly identical"; huginnfork harness: KL divergence to BF16 0.052 weight-only, 0.100 with FP8 activations, wikitext perplexity +0.18% | vLLM: Marlin W8A16; SGLang: Triton W8A8 (Section 4.2) | no |
| `unsloth/Qwen3.8-27B-FP8` (`d51e38f6`) | same parameter split as official | 30.89 GB | none | same | no |
| `huginnfork/Qwen3.8-27B-FP8` (`ee0358b3`) | FP8_DYNAMIC W8A8 on the 192 MLP projections only | 38.49 GB | KL 0.036 weight-only, 0.076 deployed; perplexity -0.13%; MTP acceptance 85.4% at k=1 | CUTLASS FP8 W8A8 | no |
| **`RedHatAI/Qwen3.8-27B-INT4`** (`91bd022d`, 571k; 769k all-time) | W4A16 symmetric g128 (AWQ smoothing + GPTQ), all Linear except vision, lm_head, embeddings, DeltaNet `in_proj_a/b`; static FP8 KV scales; 15 MTP tensors BF16 in `model_mtp.safetensors`, `re:^mtp.*` in `ignore` | 19.47 GB | against BF16, 3 seeds (AIME 8): IFEval 99.67%, MMLU-Pro 98.81%, GSM8K Platinum 101.07%, MATH-500 99.52%, GPQA 98.49%, AIME 2025 98.69% (94.17 vs 95.42) | Marlin W4A16 | yes, 0.5 GB spare |
| `nicosuter/Qwen3.8-27B-AWQ` (`6918041e`) | AWQ W4A16 asymmetric g128; DeltaNet `in_proj_qkv/z` INT8; MTP BF16 | 21.57 GB | paired against official FP8 at XHigh (H200): macro recovery 99.84% [98.86, 100.82]; BFCL 100.07%, GPQA 98.60%, LiveCodeBench v6 101.63% | Marlin | no |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` (`6e134bae`, 524k) | W4A16 asymmetric g32, DeltaNet included | 21.04 GB | same harness: 99.81%; RULER 96.77% | Marlin | no |
| `philbert440/Qwen3.8-27B-W4A16-AWQ` (`7908d42a`) | W4A16 asymmetric g128, DeltaNet included, thinking-mode calibration | 19.56 GB | same harness: 99.10% [98.07, 100.13]; LiveCodeBench v6 85.71 vs 87.71, the only build that separated from FP8 | Marlin | yes |
| `dbirks/Qwen3.8-27B-W4A16-AutoRound` (`1f05c441`) | AutoRound W4A16 | 19.47 GB | none published (HyperQwen's starting point) | Marlin | yes |
| GPTQ INT4: `abhishekchohan/...-GPTQ-INT4-FP8KV` (`045e2bb9`), `SergiioB/...-GPTQ-Int4-sym-G128-MTP-BF16` (`e28c5f95`), `sokada4/...-GPTQ-Int4` (`2232b67a`) | GPTQ W4A16 | 20.99 / 19.58 / 19.58 GB | none | Marlin | borderline |
| `lued/Qwen3.8-27B-INT8-W8A16-MTP` (`7c123737`), `Freaksterz/...-SmoothQuant-W8A8-INT8` (`2df4e3b0`) | INT8 weight-only; INT8 W8A8 with rotations | 31.64 / 31.27 GB | none read | Marlin; CUTLASS INT8 | no |
| NVFP4: `nvidia/` (`482ca0f3`), `RedHatAI/` (`2889ddee`), `unsloth/`, `Inferact/` | FP4 MLPs, FP8 elsewhere | 21.95 to 26.40 GB | RedHat NVFP4 98.3% to 100.9% incl. SWE-bench 98.98% (B200) | no FP4 on L4; vLLM's table lists Marlin FP4 for Ada (weight-only), **to check** | n/a |
| GGUF: `unsloth/Qwen3.8-27B-GGUF` (`4ca72078`), also ggml-org, bartowski | llama.cpp | UD-Q4_K_M 16.46, Q8_0 29.05, UD-IQ4_XS 14.25 GB | L4, llama.cpp b11062, one request: decode 13.1 tok/s at 29k context, 11.8 at 59k (vramcalculator) | no useful batching | not for the agent |
| Drafters: `z-lab/Qwen3.8-27B-DFlash2` (`50307d4c`, 3.85 GB), `RadixArk/Qwen3.8-27B-DSpark` (`b9a5dbdf`, 3.71 GB, license "other") | speculative decoding | | DFlash2 card (H200, SGLang, 7 drafts): acceptance 4.1 to 5.5 tokens per step vs 3.7 to 5.0 for built-in MTP; at concurrency 32 the speedup is 0.77x to 1.04x for MTP and 1.01x to 1.45x for DFlash2 | open vLLM bugs, Section 4.3 | |

What the quality evidence covers: long reasoning only through RedHat's AIME 2025 (INT4 98.7%) and
nicosuter's XHigh paired runs (four 4-bit builds within 1% of FP8; putting the DeltaNet input
projections at 4 bits looked weakest, which RedHat's INT4 also does). Nothing on ARC.

### 2.3 Other Qwen3.8 sizes and derivatives

- Official Qwen3.8 repos on HF (Qwen org, 2026-10-09): 27B (+FP8), Flash-Next (+FP8), 2.4T-A95B
  (+FP8). There is no official Qwen3.8 4B or 9B. Flash-Next (`de4b8e4d`): architecture
  `qwen4_exp`, 360 GB BF16, 185.6 GB FP8, Qwen Community License 1.0 (eligibility is an open forum
  question per yesterday's survey); does not fit 96 GB.
- Token-efficient fine-tunes of the 27B: `bottlecapai/ThinkingCap-Qwen3.8-27B` (`7891e03d`, gated,
  PolyForm Small Business 1.0.0) reports 37% fewer thinking tokens for 85.8 vs 86.6 macro accuracy
  (vendor, H200). `ukisai/Swift-Qwen3.8-27b` (`6bc57e4e`, gated, "swift-open-license-1.0") claims
  58.3% fewer at under 1% loss; HyperQwen measured 31.2% fewer on a 6-task battery. Neither license
  is an OSI license, so both are likely prize-ineligible (**unverified**); they show that 30% to
  40% shorter thinking is reachable, not candidates.
- `empero-ai/Qwen3.8-{2B,4B,9B,35B-A3B}-Distill`: third-party SFTs of Qwen3.5 and Qwen3.6 bases
  (Section 3.1).

## 3. Small and mid models for fan-out

### 3.1 empero-ai/Qwen3.8-4B-Distill (rev `c83cb7aa`)

- **What it is:** "a full-parameter distillation of Qwen3.8 2.4T A95B into the Qwen3.5-4B
  architecture" (card): base `Qwen/Qwen3.5-4B`, SFT (off-policy distillation, TRL) on "~45,000
  curated teacher traces from our internal Qwen3.8 distillation datasets" covering mathematics,
  general reasoning and instruction following. Teacher: Qwen3.8-2.4T-A95B. The data is not
  published. The card's title is "Qwen3.8-4B" and its quickstart loads `empero-ai/Qwen3.8-4B`, a
  different id.
- **Files:** 9.34 GB BF16 (4.66B parameters with the vision tower); `config.json` matches
  Qwen3.5-4B; the safetensors header holds the 15 MTP tensors (whether they were trained after the
  SFT is unknown); `training_args.bin` included. Apache 2.0 (inherited). Created 2026-08-15.
- **Card claims** (lm-evaluation-harness, HF backend, CoT protocols): gsm8k_cot 0.785 against
  0.850 for the base (a 6.5-point regression); MMLU CoT flexible 0.553 against 0.354, strict 0.233
  against 0.071. Qwen reports MMLU-Pro 79.1 for Qwen3.5-4B, so the 0.354 baseline points to answer
  extraction failing on the base rather than a real gain (**unverified** reading). No code,
  tool-use or ARC results; the card sends code users to its 9B sibling.
- **Use:** 56,030 downloads (72,113 all-time), 67 likes; its GGUF repo has 569,001. The same org
  publishes distills named after closed models ("Qwythos-9B-Claude-Mythos-5").
- **Verdict:** no evidence for writing or debugging code, a measured regression, private data.
  Not shortlisted; at most a 30-minute side cell in the spike.

### 3.2 Xiaomi MiMo

- **MiMo-7B-RL** (`6299b5a2`, MIT, April 2025) and **RL-0530** (`32340059`): 7.8B dense, all 36
  layers full attention, 32,768 native context, one MTP layer (about 90% acceptance per card).
  LiveCodeBench v5 57.8 / v6 49.3 (0530: 60.9 / 52.2), AIME 2025 55.4 (0530: 70.2). Its KV costs
  4.5x Qwen3.5-9B's per token. Superseded by Qwen3.5-9B (LiveCodeBench v6 65.6); not shortlisted.
- **MiMo-V2.6-Distill-Qwen-9B** (`2367e865`, MIT, created 2026-09-21; 21,061 downloads, 631
  likes): SFT of Qwen3.5-9B on MiMo-generated data, 77.4B total tokens and 27.2B loss-bearing
  (code 23.2B, cyber 11.0B, general 22.0B, visual 21.2B). Card table, from the MiMo-V2.6 technical
  report: SWE-bench Verified 61.1 vs 60.0 (base), SWE-bench Pro 44.6 vs 32.0, Terminal Bench 2.1
  37.1 vs 27.0, Toolathlon-Verified 35.2 vs 25.9, AutomationBench 30.3 vs 5.0. Files: 18.85 GB BF16
  in 4 shards up to 5.28 GB; config max positions 262,144; no MTP tensors in the index (no MTP
  speculation); MiMo v2.6 chat template; the SGLang recipe uses `--reasoning-parser mimo`, and vLLM
  0.31 has no `mimo` parser (parse in our harness). Xiaomi releases it "as a starting point for
  open research in agentic reinforcement learning".
- **MiMo-V2.5, V2.6-Flash, V2.6-Pro, V2-Flash:** 178 GB to 316 GB on disk; do not fit.

### 3.3 Other candidates

| Model (rev) | License | Disk size | Active params | KV per 64k FP8 | Code and agent evidence | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3.5-4B (`851bf6e8`, 8.0M) | Apache 2.0 | 9.34 GB | 4.66B | 1.0 GiB | LiveCodeBench v6 55.8, BFCL-V4 50.3, TAU2 79.9, GPQA 76.2 (Qwen, 9B card) | MTP head; same engine path as the 27B |
| Qwen3.5-9B (`c2022362`, 8.4M) | Apache 2.0 | 19.33 GB | 9.65B | 1.0 GiB | LiveCodeBench v6 65.6, BFCL-V4 66.1, TAU2 79.1, OJBench 29.2 (Qwen); SWE-bench Verified 60.0, Pro 32.0 (Xiaomi harness) | no official FP8 build |
| Qwen3.6-27B (`6a9e13bd`; FP8 `e89b16eb` 30.89 GB) | Apache 2.0 | 55.6 GB | 27.8B | 2.0 GiB | LiveCodeBench v6 83.9, SWE-bench Verified 77.2, Pro 53.5, Terminal-Bench 2.0 59.3 (Qwen) | same cost as Qwen3.8-27B, lower on every shared row |
| Qwen3.6-35B-A3B (`995ad96e`; FP8 `95a723d0` 37.49 GB) | Apache 2.0 | 71.9 GB | about 3B | 0.62 GiB | LiveCodeBench v6 80.4, SWE-bench Verified 73.4, Pro 49.5, Terminal-Bench 2.0 51.5 (Qwen); NVIDIA harness: SWE-bench Verified 70.12, Terminal-Bench 2.1 44.38 | KV 3.2x lighter than the 27B; official GPTQ-Int4 exists only for Qwen3.5-35B-A3B (`3af5ca29`, 24.46 GB); a forum post reports 0 of 64 ARC-AGI-2 evaluation grids for two Qwen3.5-35B-A3B post-trains, method not stated (**unverified**); throughput fallback |
| gpt-oss-20b (`6cee5e81`, 6.0M) | Apache 2.0 | 13.79 GB (repo 41.3 GB with `original/` and `metal/` copies) | 3.6B | 0.75 GiB | LiveCodeBench v6 74.6 (Qwen's table); SWE-bench Verified 52.44, Terminal-Bench 2.1 15.17 (NVIDIA harness) | 131k context, harmony format, Marlin MXFP4 on Ada; a public 2026 notebook titled as gpt-oss-20b "direct" scored 2.92 (yesterday's survey) |
| Gemma 4 12B (`707f0a3b`), 26B-A4B (`4d7ae498`) | Apache 2.0 | 23.95 GB; Google QAT W4A16 build 10.30 GB (`1d2c2d7f`); 51.6 GB | 12B; 4B | 26B-A4B: 5 global layers with 2 KV heads of 512 and shared K and V, about 5 KiB per token FP8, plus 25 sliding-window layers of 1024 | Google: LiveCodeBench v6 72.0 (12B), 77.1 (26B-A4B); Tau2 69.0, 68.2. NVIDIA harness, 26B-A4B: SWE-bench Verified 57.40 (Qwen's harness: 17.4) | different tokenizer and parsers; second line |
| Nemotron 3.5 Lightning 30B-A3B (`a9904d24`; NVFP4 `bee75962` 21.58 GB) | OpenMDW 1.1 | 65.85 GB | 3B | Mamba-2 hybrid | NVIDIA: SWE-bench Verified 51.56, Terminal-Bench 2.1 24.58, GPQA 75.44, all below Qwen3.6-35B-A3B in the same table | card lists A100, H100, Blackwell, not Ada |
| Nemotron 3 Nano 4B (`dfaf35de`) | NVIDIA Nemotron Open Model License (not OSI) | 7.96 GB | 4B | Mamba-2 hybrid | none read | license risk |
| Phi-4-reasoning-plus (`69baf852`), Phi-4-mini-flash-reasoning (`1dff8163`) | MIT | 29.3 / 7.7 GB | 14.7B / 3.8B | full attention, 32k (plus) | 2025 releases; no newer Phi reasoning LLM on HF (only Phi-4-reasoning-vision-15B, Jan 2026) | superseded |
| DeepSeek distills: newest small is DeepSeek-R1-0528-Qwen3-8B (`6e8885a6`, May 2025) | MIT | 16.4 GB | 8.2B | full attention | 2025 | DeepSeek's 2026 releases are V4 Flash and Pro (too large) and drafters |

None of the models in Sections 3.1 to 3.3 has an arcprize.org entry (`evaluations.json`, read
2026-10-09); the only open model there that fits 4x L4 is Qwen3.8-27B.

## 4. Engines on sm_89, offline in Kaggle

### 4.1 Versions and what an offline install needs (PyPI, read 2026-10-09)

- **vLLM 0.31.0** (2026-10-05): wheel `cp38-abi3 manylinux_2_28_x86_64`, 320.6 MB. Pins
  `torch==2.13.0`, `torchvision==0.28.0`, `torchaudio==2.11.0`, `transformers>=5.10.4,<5.18.0`,
  `compressed-tensors==0.17.0`, `flashinfer-python==0.7.0.post1`,
  `nvidia-cutlass-dsl[cu13]==4.7.1`, `xgrammar==0.2.7`, `numba==0.65.0`, `tilelang`,
  `quack-kernels`, `humming-kernels[cu13]`. The Qwen3.5 architecture needs vLLM 0.17.0 or later
  (vLLM recipe). The RedHat INT4 config records compressed-tensors `0.18.1.a20260824` while vLLM
  pins 0.17.0 (loading **to check** at boot).
- **SGLang 0.5.21** (2026-10-01): wheel `manylinux_2_34`, cp310 to cp313, 28.8 MB. Pins
  `torch==2.13.0`, `transformers==5.12.1`, `flashinfer_python[cu13]==0.6.18`,
  `sglang-kernel==0.4.7` (373.1 MB for x86_64), `flash-attn-4`, `sgl-deep-gemm`, `sgl-deep-ep`,
  `cuda-python>=13.0`. Its Qwen3.8-27B cookbook was measured on v0.5.19 on H200, RTX PRO 6000,
  RTX 5090 and DGX Spark, not on Ada. vLLM and SGLang need separate environments (different
  transformers and flashinfer pins).
- **torch 2.13.0:** CUDA 13.0 user-space wheels (`cuda-toolkit==13.0.3`, cuDNN 9.20.0.48, NCCL
  2.29.7, cuSPARSELt 0.8.1, NVSHMEM 3.4.5) and triton 3.7.1; cp312 and cp313 wheels are 526.6 MB.
- **Wheel sizes (x86_64, MB):** torch 527, cuBLAS 421, cuDNN 366, cuFFT 214, NCCL 206, cuSOLVER
  201, triton 198, cuSPARSELt 170, cuSPARSE 146, NVRTC 90, NVSHMEM 60, nvJitLink 41, vLLM 321:
  about 3.0 GB; a complete vLLM wheelhouse is about 3.5 GB (**estimate**; measure with
  `pip download`). SGLang adds about 0.4 GB.
- **Kaggle side:** driver 580.159.04 reports CUDA 13.0 as its maximum (kaggle-vllm, recorded on
  Kaggle's T4 notebooks); our L4 runs show the same driver version, so torch 2.13's CUDA 13.0
  runtime should load (**to check** with an offline import). Ubuntu 22.04 (glibc 2.35, `K0.md`)
  satisfies `manylinux_2_28` and `_2_34`. The latest image had Python 3.12.13 and torch 2.10.0
  (`K0.md`, CPU image built 2026-06-29; kaggle-vllm records torch 2.10.0+cu128 on the GPU image),
  and the image source moved to Python 3.13 on 2026-09-28 (`kaggle_first_runs.md`), so the
  wheelhouse must match the pinned image's Python. The solver needs its own venv or notebook; it
  cannot share a process with the NVARC Unsloth stack.
- **JIT compilation offline:** `flashinfer-python` 0.7.0.post1 is a 33.7 MB pure wheel and
  `flashinfer-cubin` on PyPI stops at 0.6.13 (458 MB), so FlashInfer kernels may need nvcc at
  first use; kaggle-vllm records CUDA toolkit 12.8.93 in the image, not 13 (**to check**).
  Mitigations: `--attention-backend TRITON_ATTN` (handles FP8 KV on sm_89, `triton_attn.py`),
  Triton DeltaNet kernels (the sm_89 default), and caches warmed in a dev run and shipped as a
  dataset.
- **Precedent:** `kaggle-vllm` 0.2.0 (PyPI, 2026-08-31) stages a vLLM 0.18 cu128 wheel on Kaggle's
  T4 notebooks next to the preinstalled torch and documents the profile (Python 3.12.13, torch
  2.10.0+cu128, CUDA 12.8.93, driver 580.159.04).

### 4.2 Kernel paths on sm_89

- **Official block FP8, vLLM 0.31.0:** the CUDA priority list for block FP8 is FlashInfer
  DeepGEMM, DeepGEMM, CUTLASS block, B12x, Marlin FP8, Humming, Triton block, torch
  (`kernels/linear/__init__.py`). `cutlass_scaled_mm_supports_block_fp8` returns true only for
  capability 90 and up (`scaled_mm_entry.cu`), so on L4 the checkpoint runs on Marlin FP8, weights
  FP8 and activations BF16. Issue #58241 (open) reports exactly this with Qwen3.8-27B-FP8 on an RTX
  6000 Ada. Decode cost is unchanged (memory-bound); prefill uses BF16 tensor cores instead of FP8.
- **Block FP8, SGLang 0.5.21:** `_dispatch_auto_backend` falls through to
  `triton_w8a8_block_fp8_linear` on Ada, and Marlin FP8 is auto-enabled only for $80 \le sm < 89$
  (`fp8_utils.py`), so prefill gets real FP8 matmuls.
- **Per-channel FP8** (llm-compressor FP8_DYNAMIC, or vLLM online `--quantization fp8`): CUTLASS
  FP8 W8A8, supported on sm_89 with CUDA 12.4 or later (`scaled_mm_entry.cu`).
- **W4A16** (compressed-tensors, GPTQ, AWQ): Marlin, minimum capability 75; Machete and CUTLASS
  W4A8 require capability 90 (`kernels/linear/mixed_precision/{marlin,machete,cutlass}.py`).
- **DeltaNet:** prefill uses the Triton/FLA kernel on sm_89 (FlashInfer only on SM90, SM100,
  SM120); the fused CUDA decode kernel needs a BF16 model and sm80+ (`qwen_gdn_linear_attn.py`).
  SGLang's DeltaNet backend defaults to Triton.
- **Attention:** FlashAttention 2 (the Ada version) does not take an FP8 KV cache, so FP8 KV runs
  on FlashInfer or the Triton backend (`flash_attn.py`, `triton_attn.py`).

### 4.3 Known issues that touch this design

- **vLLM #55533** (open, 2026-09-06; RTX 4090 D sm89, Qwen3.8-27B GPTQ INT4, vLLM main): with
  MTP the scheduler runs only about 3 sequences at batch 4 or more, and MTP becomes 2x to 3x slower
  than no speculation; at batch 3 or less MTP gives +33% to +72%. Cause per comments: MTP charges
  $1+k$ extra state blocks per request per DeltaNet group, and the 0.88 GiB MTP head shrinks the
  pool. The 9B sibling is unaffected up to batch 20. Consequence: no MTP for batched 27B decoding
  on L4 unless fixed.
- **vLLM #55766** (open): NaN logits after a prefix-cache hit (Qwen3.8-27B, align mode, ngram
  speculation), reproduced on 4x RTX 4090 TP4 on v0.28.0 and main; trigger: a final prefill chunk
  of exactly $1+k$ tokens. Consequence: prefix caching without speculation, plus a guard against
  NaN and repeated-token outputs.
- **vLLM #58807** (open, v0.30.0): MTP fails to load a dense BF16 MTP head on compressed-tensors
  pack-quantized checkpoints when the head is not ignored. RedHat INT4 ignores `re:^mtp.*`, so it
  may load (**to check**); moot while MTP is off.
- **vLLM #54690** (open, v0.28.0, 4x RTX 4090): a draft-model speculator with FP8 draft KV crashes
  hybrid DeltaNet models at startup, with a native crash on sm89.
- **vLLM #60174** (0.30 and 0.31) and **#58894**: DFlash2 or DSpark plus prefix caching corrupts
  output or drops acceptance to 0% after cache hits; **#54691**: DFlash is a net loss at long
  context on 4x RTX 4090.
- **Prefix caching across many agents** (HyperQwen `docs/gotchas.md` items 44 and 60, vLLM 0.28
  and 0.29, **unverified**): align-mode caching can fall to 0% hits every few turns, and two long
  conversations advanced in turn can evict each other completely, which forces a full re-prefill
  per turn. HyperQwen calls this "exactly how a two-agent deployment runs" and sets a snapshot
  retention interval (CLI flag from 0.29). This is the multi-agent pattern.
- **PCIe without P2P** (HyperQwen `docs/multi-gpu.md`, 2x RTX 4090): TP=2 was slower than one
  card (118.7 vs 151.1 tok/s single stream; a 128-request burst took 110.9 s vs 76.2 s), and data
  parallel gave 1.92x. Whether Kaggle's L4 VM allows P2P is unknown (**to check** with
  `NCCL_DEBUG=INFO`).
- **Shapes:** tensor parallel size must divide the 4 KV heads (1, 2 or 4). The huginnfork card
  needed `--max-num-seqs 32` because the default 1024 exceeded the mamba cache blocks during
  CUDA-graph capture (vLLM version not stated).

## 5. Throughput on 4x L4

### 5.1 Measurements

| Source | Hardware and engine | Model and build | Result |
| --- | --- | --- | --- |
| Forum 745787 (Anil Thomas, yesterday's survey) | 4x L4 | Qwen3.8-27B | about 350 tok/s, used in a full-run estimate (**unverified**, conditions unknown) |
| Forum 732723 (Dominic) | 4x L4, vLLM 0.26 | Qwen3.5-35B-A3B post-trains, FP8 | 438 and 452 tok/s (**unverified**) |
| Forum 745787 (NguyenThanhNhan) | 4x L4, SGLang | Qwen3.8-27B FP8, MTP via EAGLE (3 steps), FP8 KV | two scored submissions, at most 32.78 (**unverified**) |
| vramcalculator | 1x L4, llama.cpp b11062 | UD-Q4_K_M, one request | decode 13.1 tok/s at 29k context, 11.8 at 59k; prefill 647 and 495 tok/s |
| HyperQwen README | 1x RTX 3090 (936 GB/s, 250 W), patched vLLM 0.30 | INT4 requant | 127 tok/s single stream; about 1,035 tok/s at 64 concurrent (128 in, 512 out) |
| HyperQwen `docs/wsl2-4090.md` | 1x RTX 4090, batch profile, FP8 KV | INT4 | 55, 195, 384, 676 tok/s at 1, 4, 8, 16 concurrent (4k prompts) |
| `RadixArk/Qwen3.8-27B-DSpark` card | 1x H200, SGLang | official FP8 | no speculation: 94 tok/s single stream, 600 to 670 at 8, 1,300 to 1,860 at 32 (2,048-token outputs) |

Scaling by memory bandwidth (L4 300 GB/s) applies only in the memory-bound regime and ignores the
L4's 72 W cap; use these for ordering, not as L4 numbers.

### 5.2 Estimates for 4x L4 (no speculation)

Method (script in scratch): each decode step, every GPU reads its weight shard, the FP8 KV of every
running sequence, and reads and writes every sequence's DeltaNet state. Assumptions: 240 GB/s
effective per L4 (80% of spec); 92% of 23,034 MiB usable with 2 GB reserved; 2 state slots per
sequence; 1, 3, 5 ms per-step overhead at TP 1, 2, 4; compute checked at 40% of dense peak; batch =
as many sequences as the KV pool holds at the mean context. Calibration: the 27B FP8 TP4 estimate
at a 32k mean context (424 tok/s) sits next to the forum's 350.

| Configuration | Sequences per replica at 8k / 32k / 64k | tok/s at 8k | at 32k | at 64k | Tokens per 12 h at 32k (95% busy) |
| --- | --- | ---: | ---: | ---: | ---: |
| 27B official FP8, TP4 | 89 / 36 / 20 | 1,030 | 424 | 237 | 17.4M |
| 27B RedHat INT4, TP4 | 110 / 45 / 25 | 1,280 | 528 | 296 | 21.7M |
| 27B INT4, 2 replicas of TP2 | 39 / 16 / 9 | 965 | 398 | 224 | 16.3M |
| Qwen3.6-35B-A3B FP8, TP4 | 210 / 95 / 55 | 2,770 | 1,255 | 727 | 51.5M, likely optimistic (measured 438 to 452 above) |
| 9B (MiMo distill or Qwen3.5-9B), online FP8, 4 replicas | 38 / 14 / 7 | 1,980 | 728 | 378 | 29.9M |
| gpt-oss-20b MXFP4, 4 replicas | 63 / 15 / 7 | 3,260 | 788 | 376 | 32.3M |
| Qwen3.5-4B BF16, 4 replicas | 49 / 18 / 9 | 2,300 | 847 | 443 | 34.8M |
| Qwen3.5-4B online FP8, 4 replicas | 64 / 23 / 12 | 3,040 | 1,110 | 596 | 45.6M |

Reading:

- Once the pool is full, each step reads the whole pool, so throughput falls roughly as one over
  the mean context. At an 8k mean context the 27B reaches about 1,000 to 1,300 tok/s instead of
  about 420 to 530 at 32k; context length matters more than FP8 vs INT4 (about +25% for INT4).
- TP4 beats two TP2 replicas for the 27B because the weights are not duplicated. Single-GPU
  replicas only work for the small models (the INT4 27B leaves under 2 GB for KV on one L4).
- Co-locating the 27B INT4 (4.4 GB per GPU at TP4) with a 9B FP8 replica (about 11 GB per GPU)
  leaves about 5 GB per GPU for both KV pools: not practical. Use phases (swap servers, swap time
  to measure) or split GPUs (27B on two, small replicas on two).
- MTP on L4: blocked at batch by #55533 in vLLM; at concurrency 3 or less +33% to +72% (same
  issue); SGLang with MTP ran in a participant's submissions (**unverified**), throughput unknown.

Budget against ARC (**estimate**): at XHigh about 175k output tokens per task, so 17M to 22M tokens
cover about 100 to 125 of 240 tasks; at Low (about 93k per task, 22.8% on arcprize) about 190 to
235. The agent has to beat the arcprize harness on tokens per task (stop when a program passes all
demonstrations, short sub-agent contexts, Low effort where it suffices) to reach every task.

## 6. Offline availability

- **Own download notebook** (output limit about 20 GB, `kaggle_first_runs.md`, **unverified**).
  Fits: RedHat INT4 19.47 GB, dbirks AutoRound 19.47, philbert440 19.56, Qwen3.5-9B 19.33,
  MiMo-V2.6-Distill-Qwen-9B 18.85, gpt-oss-20b 13.79 (top-level files only), Gemma 4 12B QAT 10.30,
  Qwen3.5-4B 9.34, empero 4B 9.34. Does not fit: official FP8 30.89 GB (66 files of at most 6.0 GB,
  so two datasets split by file), nicosuter AWQ 21.57, Qwen3.6-35B-A3B FP8 37.49,
  Qwen3.5-35B-A3B GPTQ-Int4 24.46.
- **Public Kaggle copies** (titles from web search; size, revision and license **to check**;
  verify every file against the HF LFS sha256 from `https://huggingface.co/api/models/<id>/tree/main`
  before trusting a mirror): datasets `saltb0x/qwen3-8-27b-fp8` ("Qwen3.8-27B FP8 (official)"),
  `johnlussier/qwen3-8-27b-fp8-hf-snapshot`, `sergiodefreitas/qwen3-8-27b`,
  `rahim3/qwen3-8-27b-bf16`, `souhardyatalukdar/unslothqwen3-8-27b-nvp4`; models
  `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` (snippet gives the mount
  `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`), `noillum123/qwen3-8-27b-uncensored-fp8`
  (abliterated; avoid); the QwenLM org page `qwen-lm/qwen-3-5`; generated pages
  `refs/hf-model/Qwen/Qwen3.5-4B` and `refs/hf-model/Qwen/Qwen3.5-9B`. A search summary mentions a
  dataset "kcc-qwen38-27b-w4a16" without a URL (**unverified**). Nothing found for RedHat INT4,
  the MiMo 9B distill, or the empero distills.
- **Hugging Face reference route:** Kaggle generates a `refs/hf-model` page when a notebook
  references a HF model; a Kaggle discussion (getting-started 547125, known only by title and
  snippet) says such a model is then attached and works in internet-off competition submissions,
  while the HF blog of 2025-05-14 called offline-competition support "a few more months" away.
  If it works for a 31 GB repo, the 20 GB limit stops mattering for the official FP8 (**to
  check** in a dev session: reference `Qwen/Qwen3.8-27B-FP8`, then list `/kaggle/input` in an
  internet-off session).

## 7. Shortlist and the first GPU run

### 7.1 Shortlist

| Role | Model and build | Size | License | Evidence | Throughput on 4x L4 (estimate) |
| --- | --- | --- | --- | --- | --- |
| Main | Qwen3.8-27B as `RedHatAI/Qwen3.8-27B-INT4` (`91bd022d`); vLLM 0.31.0, TP4, FP8 KV, `--language-model-only`, prefix caching on, no speculation | 19.47 GB; about 4.4 GB of weights per GPU | Apache 2.0 | ARC-AGI-2 42.4 (XHigh) and 22.8 (Low) for the model; INT4 recovery 98.5% or better, AIME 2025 98.7% | 530 tok/s at 32k, 1,280 at 8k; about 22M tokens per 12 h at 32k |
| Fan-out | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (`2367e865`); online FP8, one replica per L4, in a separate phase or on GPUs not used by the 27B | 18.85 GB disk, about 11 GB per GPU | MIT (base Apache 2.0) | SWE-bench Pro 44.6, Terminal Bench 2.1 37.1 (Xiaomi); no ARC | 730 tok/s at 32k, 1,980 at 8k (4 GPUs) |
| Fallback | Qwen3.8-27B as `Qwen/Qwen3.8-27B-FP8` (`017b9c7a`); SGLang 0.5.21, TP4, FP8 KV, Triton attention | 30.89 GB | Apache 2.0 | "nearly identical" (Qwen); KL 0.05 (huginnfork); used in a participant's Kaggle submissions (forum, **unverified**) | 420 tok/s at 32k |

Controls for the same session: Qwen3.5-9B (same architecture as the fan-out pick, no agentic SFT);
Qwen3.5-4B if the 9B is too slow; Qwen3.6-35B-A3B FP8 as the throughput fallback if the 27B
budget does not close. All shortlisted licenses are Apache 2.0 or MIT; the open rules question on
non-OSAID models (yesterday's survey, Section 5.3) applies to them as it does to our current base.

### 7.2 What to measure in the first GPU run

One 4x L4 session (`NvidiaL4`): the 27B needs at least two L4s, and per the human's accelerator
note a 1xL4 session can land on a T4 without BF16.

1. **Environment:** driver and CUDA max (`nvidia-smi`), `nvidia-smi topo -m` and NCCL transport
   (P2P or SHM), offline import of torch 2.13 cu130, Python version of the image; for each engine
   and build: load time from `/kaggle/input`, compile and graph-capture time, KV pool tokens and
   max concurrency printed at boot, peak memory per GPU.
2. **Throughput under batching** (fixed seed, thinking on, 4k-token prompts, `max_tokens` 16k):
   concurrency 1, 8, 16, 32, 48 at mean contexts near 8k and 32k for INT4 vLLM TP4, FP8 vLLM TP4,
   FP8 SGLang TP4, and INT4 as 2x TP2; one cell each with MTP k=1 and k=2 at concurrency 1 to 4.
   Record aggregate output tok/s, TTFT, per-request tok/s and preemptions. Then a multi-turn replay
   (8 agents x 6 turns with REPL outputs) for prefix-cache hit rate and a count of NaN or
   repeated-token outputs.
3. **Solve rate on a fixed 20 to 40 ARC-AGI-2 training tasks** (training split only; mixed
   difficulty; one seed; same REPL agent loop and verification on demonstrations; 2 attempts;
   64k-token cap per attempt): 27B INT4 vs 27B FP8 at XHigh; the 27B at Low on the same tasks;
   the 9B FP8 alone, and as a sub-agent under a 27B planner on 10 tasks. Gate: keep INT4 if it is
   within 1 task of FP8 on 40 tasks (project noise level is 2 tasks on 40) and at least 20% faster
   at concurrency 32; otherwise use FP8.
4. **Tokens per solved task:** completion, prompt and tool-output tokens per task and per solve,
   REPL turns per solve, wall seconds per solve, and false positives (programs that pass every
   demonstration but miss the test output).
5. **Projection:** tokens per solved task x 240 against the 12 h token total from item 2.

Caveat: Qwen3.8's exposure to public ARC training tasks is unknown, so training-set solve rates
may overstate the hidden test.

## 8. Open risks and unverified items

- Every throughput number for 4x L4 is an estimate or a forum statement; no L4 measurement of a
  batched hybrid DeltaNet model was found.
- INT4 against FP8 on ARC (or on any agentic multi-turn task) is unmeasured; the closest evidence
  is AIME 2025 and XHigh LiveCodeBench, GPQA and BFCL.
- vLLM bugs #55533, #55766, #58807, #54690, #60174 are open as of 2026-10-09; HyperQwen's
  prefix-cache findings come from vLLM 0.28 and 0.29 on consumer cards.
- Offline engine install: CUDA 13.0 runtime on the Kaggle driver, FlashInfer JIT without a CUDA 13
  nvcc, the image's Python after the 2026-09-28 switch, and RedHat's compressed-tensors 0.18
  format under vLLM's 0.17 pin are all unchecked on Kaggle.
- Kaggle mirrors are known by title only; the Hugging Face reference route for internet-off
  submissions rests on a search snippet; the 20 GB output limit is itself unverified.
- The fan-out picks rest on vendor benchmarks (Xiaomi, Qwen); none has an ARC result, and the
  empero distill's claims rest on a harness whose base scores look broken.
- Tokens per ARC task at XHigh (about 175k) are derived from arcprize cost and price, not counted.
