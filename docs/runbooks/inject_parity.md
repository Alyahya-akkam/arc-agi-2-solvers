# Runbook: `scripts/inject_parity.py` (P3.1-01, injection parity on the GPU)

Task P3.1 (BRIEF B5 P3.1, A8, A9; `docs/STATE.md`, conflict 3). Config:
`configs/inject_parity.yaml`. Build: `configs/script_P3.1-01.yaml` (script mode,
[`build_submission.md`](build_submission.md)). Pre-run brief:
[`docs/prerun/P3.1-01.md`](../prerun/P3.1-01.md). Module:
[`docs/modules/inject.md`](../modules/inject.md). Background: the research note
[`docs/research/d2l_t2l_reading.md`](../research/d2l_t2l_reading.md), risks 1 to 4 ("risk $n$"
below). Tests: `tests/test_inject_parity.py`.

## 1. Why this script exists

Meta-training (P3.5) trains H through P3.1's differentiable injection (`arc26.hyper.inject`): the
adapter is added to the frozen base inside `inject`. At test time (P4.1) H's adapter is exported and
loaded through PEFT for the polish and the decode. If the two paths computed different logits, H
would be trained on one model and used on another. P3.1's CPU tier (`tests/test_inject.py`) found
them bitwise equal in fp32 and bf16, but a CPU says nothing about GPU kernels: cuBLAS chooses a
kernel by shape, stride, alignment, and dtype, and each kernel rounds in its own order.

The question: on one L4 in bf16, on DEV4's sequences, do the injected path and PEFT loaded from the
exported adapter give the same logits, and does the gradient of P3.5's loss reach every factor
through the injected path with non-reentrant checkpointing? It feeds B5 P3.1's acceptance ("max abs
diff below $10^{-3}$ in bf16 on DEV4 sequences; gradient flows to $(A, B)$"), conflict 3 (the
human's call if the logits are close but not equal), and two diagnostics for P4.1 (adapter swaps).

Conflict 3 in numbers: a bf16 value $z$ with $2^k \le |z| < 2^{k+1}$ has the spacing (one ULP)
$2^{k-7}$. One ULP exceeds $10^{-3}$ from $k = -2$ on ($2^{-9} \approx 1.95 \times 10^{-3}$), so for
$|z| \ge 0.25$ the bound demands bitwise-equal logits; below 0.25 it allows one or a few ULPs.

## 2. What it does, step by step

1. **Config.** Reads the YAML (every key required, none extra) and validates the GPU values and the
   `smoke` values, so a smoke run also checks the GPU settings. `--smoke` applies the `smoke`
   section; `--input-root`, `--model-dir`, `--challenges`, `--out-dir`, and `--device` override.
   Writes the resolved `config.yaml` and a first `parity.json` (status `running`, provenance:
   command line, commit, package versions, the path of the `arc26` copy that runs).
2. **Smoke inputs** (`--smoke` only). Writes a synthetic input tree under `input_root`, laid out as
   Kaggle mounts the real one: the tiny random Qwen3 of `tests/test_inject.py` (hidden 64, 2 layers,
   4 query heads, 2 key-value heads of width 16, vocabulary 16, `initializer_range` 0.2; positions
   raised from 64 to `max_tokens`) saved with `save_pretrained` beside P3.5's stand-in tokenizer in
   `models/<model_slug>/`, and DEV4-shaped tasks (DEV4's ids, its 1, 2, 1, 1 test inputs, random
   grids) in `competitions/<competition>/<challenges_file>`. Everything after this step is the GPU
   run's code.
3. **Device.** `cuda:0` must exist and support bf16 (a T4 does not), else the run stops (Error).
   Records the GPU, compute capability, CUDA and cuDNN versions, and the global settings that pick
   kernels or roundings (TF32, bf16 reduced-precision reduction, deterministic algorithms, the SDPA
   backends, `CUBLAS_WORKSPACE_CONFIG`).
4. **Inputs, by search.** Under `input_root` (`/kaggle/input`), with the script notebook's own
   search (`find_inputs`: depth 8, package trees skipped): the model is the one folder holding a
   `config.json` below a folder named `model_slug`; the challenges file is the one
   `arc-agi_evaluation_challenges.json`, preferring copies below the competition's folder. Zero or
   two candidates stop the run. A name containing `solution` is refused; only DEV4's train pairs and
   test inputs are read, and the file's sha256 is recorded.
5. **Base.** Tokenizer and model from that folder (`local_files_only`), bf16, frozen, evaluation
   mode, with the attention kernel P3.5 uses: FlashAttention 2 when `flash_attn` imports on a bf16
   GPU, else SDPA (`resolve_attention`; a failed FlashAttention 2 load falls back to SDPA). Enables
   non-reentrant activation checkpointing, which transformers applies in training mode only.
6. **Sequences.** Per DEV4 task, built by the notebook's own code (`arc26.nvarc.arc_loader`) under
   the identity view, tokenized with `tokenizer.encode` as the notebook does: the decode prompt of
   each test input, `fmt_train(train) + fmt_query(test)` cut to $8192 - 931$ tokens, and the
   training text `fmt_train(train)` cut to 8192 tokens (931 is `max_new_tokens`, the reply of a
   $30 \times 30$ grid). `cut_to_len` drops the first demonstration pairs until the text fits. For
   the training text the loss positions $O$ are its last pair's output-grid tokens and the closing
   `<|im_end|>` (P3.5's query-output positions, located with `render_pair`, and equal to the last
   segment the notebook's completion-only collator labels). DEV4 gives 9 sequences: 5 decode
   prompts and 4 training texts.
7. **Base pass.** Every sequence alone (batch 1, no padding, no cache) in evaluation mode under
   `torch.no_grad`; the bf16 logits $z^\text{base} \in \mathbb{R}^{T \times 16}$ go to the CPU.
8. **The adapter.** `wrap_targets` wraps `q_proj`, `k_proj`, `v_proj`, `o_proj` of the 36 layers
   (144 projections; the scope is H's, copied from `configs/hypernet.yaml` and checked against it by
   the tests). One seeded generator draws, per projection $m$ with frozen weight
   $W_m \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:
   $$A_m \sim \mathcal{U}\!\left(-\tfrac{1}{\sqrt{d_\text{in}}}, \tfrac{1}{\sqrt{d_\text{in}}}\right)^{r \times d_\text{in}}
   \ (\text{PEFT's } \texttt{lora\_A} \text{ draw}), \qquad
   \tilde B_m \sim \mathcal{N}(0, 1)^{d_\text{out} \times r}, \qquad
   B_m = \tilde B_m \, \frac{\rho \, \|W_m\|_F}{s \, \|\tilde B_m A_m\|_F},$$
   with $s = \alpha / \sqrt{r} = 8$ and $\rho = 0.1$, so that $\|s B_m A_m\|_F = \rho \|W_m\|_F$.
   The norm of the rank-$r$ product comes from two $r \times r$ Gram matrices,
   $\|\tilde B A\|_F^2 = \operatorname{tr}(A^\top \tilde B^\top \tilde B A) = \sum_{ij} (A A^\top)_{ij} (\tilde B^\top \tilde B)_{ij}$.
   The factors are fp32, as PEFT keeps adapters on a bf16 base.
9. **Zero-B pass and injected pass.** With $B = 0$ inside `inject`, then with the adapter: every
   wrapped projection computes $y = \mathrm{bf}\big(\mathrm{bf}(W x) + s\,B (A x)\big)$, the delta in
   fp32 and one rounding, PEFT's op sequence (`docs/modules/inject.md`).
10. **Injected gradients.** For each training text, in training mode (checkpointing on), forward,
    loss, and backward inside one `inject` block, with fresh leaf copies of the factors:
    $$\mathcal{L} = -\frac{1}{|O|} \sum_{t \in O} \log \operatorname{softmax}(z_{t-1})_{y_t},$$
    P3.5's query-output cross-entropy (`arc26.hyper.train.query_ce`). Records whether every
    $\nabla A_m$ and $\nabla B_m$ arrived, is finite, and is non-zero, then runs the backward again
    (repeat control). A forward pre-hook on every projection counts its calls: 2 per projection
    when checkpointing recomputed each layer during the backward, 1 without it; the checkpoint
    function's `use_reentrant` is read back from the model. A backward that raises (an OOM in the
    8k-token text, a loss that reaches no factor) is recorded on its text with the reason and its
    traceback in `error.log`, counts as a gradient that does not flow, and the run goes on.
11. **Unwrap and base repeat.** Puts every original `nn.Linear` back (the same objects) and runs the
    base pass again: it must reproduce step 7 bitwise.
12. **PEFT.** Saves the adapter with `save_adapter` (fp32 safetensors and `adapter_config.json`,
    recorded by sha256) and loads it with `PeftModel.from_pretrained(base, path)` and its defaults,
    the loader with bitwise parity on a bf16 base on the CPU (`inject.md`, Limits). PEFT pass,
    then a repeat (determinism control). PEFT's `lora_A`/`lora_B` weights are then made trainable
    and each training text's backward runs through PEFT (checkpointing on), for the gradient
    comparison.
13. **Swaps** (P4.1's diagnostics). Into the same adapter name, the previous puzzle's adapter
    ($A/2$, $B/2$) and then this one, once with `load_adapter(path, "default")` and once with
    `set_peft_model_state_dict`; a pass after each.
14. **Comparisons.** For a candidate $u$ and a reference $v$ ($T \times 16$ bf16 logits), element
    by element: the largest $|u - v|$; the count of elements with $u \ne v$; the elements at or
    above the bound, $E = \{(t, k) : |u_{tk} - v_{tk}| \ge 10^{-3}\}$, and their largest ULP
    distance $\max_E |\operatorname{ord}(u) - \operatorname{ord}(v)|$ (the same over all elements
    is a diagnostic), where $\operatorname{ord}$ reads the 16 bits as a signed integer $i$ and maps
    it to $i$ for $i \ge 0$ and to $-(i \wedge \texttt{0x7FFF})$ otherwise (consecutive bf16 values
    get consecutive integers, $\pm 0$ both 0); and the positions whose argmax differs. An element
    is within tolerance when $|u - v| < 10^{-3}$ or it is at most 2 ULPs off. A comparison is
    **exact** ($E$ empty, every element below the bound; for the zero-B control, bitwise),
    **within** (not exact, but every element within tolerance, so $\max_E \le 2$ ULPs, and every
    argmax equal), or **beyond** (anything else, a non-finite value included). ULPs count only on
    $E$: a logit near zero has tiny ULPs (0.01 has $2^{-14}$), so rounding noise far below the
    bound there would read as thousands of ULPs and fail every comparison that is not bitwise. The
    effect of the adapter is the mean and the largest $|z^\text{peft} - z^\text{base}|$, the
    fraction of logits it changed, and the fraction of positions whose argmax it changed.
15. **Decision** (`decide`, the brief's section 1), first match wins:
    **Error** when the run did not finish, a sequence lacks a measurement, a reference logit (PEFT
    or the base) is not finite, or the adapter moved some sequence's logits by less than $10^{-2}$
    on average; **Fail** when a comparison (injected against PEFT, or the zero-B control) is beyond,
    or a factor gradient is missing, zero, or not finite, or a backward raised; **Human decides**
    when a comparison is within; else **Pass**. The repeats, the swaps, the gradient differences,
    the all-element ULP distance, and the calls per projection are diagnostics, reported but not
    decided on.
16. **Outputs.** `parity.json` after every stage, then `summary.md`, printed to the session log.
    A wall budget (`max_wall_s`) is checked before every pass: past it the run stops with its logs
    (exit 3, Error). Every GPU op appends a timing record.

## 3. Why it is built this way

- **One base, the paths in turn** (inject, unwrap, PEFT on the same model object). Considered: two
  copies of the base. Lost: 2 times 7.3 GB of weights plus the activations of an 8k-token backward
  leaves little of the L4's 22.5 GB, and a copy on a second GPU would compare two devices. One
  object guarantees one set of weights; PEFT cannot adapt a wrapped model (`inject.md`, Limits),
  so the wrappers are removed first, and the base repeat (step 11) proves the base is untouched.
- **The reference loader is `PeftModel.from_pretrained` with its defaults.** Considered:
  `get_peft_model` plus `set_peft_model_state_dict` in memory (the CPU test's reference), and the
  loaders that round factors to bf16. Lost: P4.1 loads adapters from files, and on a bf16 base only
  the loaders that copy into an existing fp32 adapter keep the factors (`inject.md`, Limits table).
  The two swap paths with CPU parity are measured as diagnostics because P4.1 may swap adapters
  per puzzle; they are not part of the decision (task brief).
- **The adapter: PEFT's $A$, $B$ scaled to $\rho = 0.1$.** Considered: H at initialization ($B = 0$:
  the delta is exactly zero, so the comparison would be vacuous); a fixed $\sigma_B$ (the effect
  then depends on weight norms nobody has measured on this base, so it could vanish or swamp the
  logits); a trained adapter (needs a training run first). With $\|s B A\|_F = 0.1 \|W\|_F$ the
  delta is about 10% of a projection's output in RMS, 13 to 26 bf16 ULPs of a typical output
  element (one ULP is $2^{-8}$ to $2^{-7}$ of the value), so every rounding of every projection
  depends on the delta and a kernel difference in the LoRA path flips some of them. On the smoke's
  tiny model it moves the logits by 0.25 to 0.30 on average and changes 99% of them (**measured**,
  `--smoke`); RMSNorm before every block keeps a 36-layer base in range.
- **$s = 8$, H's scope, and not a scale that rounds.** The CPU tier uses $r = 8$, $\alpha = 16$
  ($s = 4\sqrt 2$) so that moving the multiply by $s$ changes the bits (`inject.md`, Tests). This
  run measures the configuration P3.5 trains and P4.1 decodes; with $s = 8$ the place of the
  multiply cannot change a bit on any hardware (a power of two), and the op order stays pinned by
  the CPU tier.
- **DEV4, identity view, the notebook's own texts and cuts.** B5 asks for "DEV4 sequences". The
  notebook's 16 views would add random color permutations and example orders without changing
  any shape the kernels see; the identity view keeps the run deterministic and the texts exactly
  reproducible from the challenges file. Decode prompts and training texts both enter, so the
  token counts span about 2k to 8k (estimate from `P0.3-01`'s timing records), the range in which
  cuBLAS's choices for $M = T$ rows could change.
- **Batch 1, no padding** (risk 1): padding changes the shapes and adds masks, which could pick
  other kernels in one path only. Each sequence runs alone in both paths.
- **Logits in evaluation mode, gradients in training mode with non-reentrant checkpointing.** The
  decision is about the logits, which P4.1 decodes from, in evaluation mode. The gradient check
  runs as P3.5 trains: training mode (transformers checkpoints only there; Qwen3's attention
  dropout is 0, which `load_base` checks), forward and backward inside one `inject` block (risk 2),
  non-reentrant checkpointing (the reentrant default detaches inputs that do not require grad).
- **Gradients are judged on flow, not on equality.** Backward kernels are not deterministic in
  general (FlashAttention 2's backward accumulates with atomics), so bitwise gradient equality is
  not a fair criterion; B5 asks that the gradient flows. The largest relative difference to PEFT's
  gradients, $\max_m \max|g_m - h_m| / \max|h_m|$, and to a repeat of the injected backward are
  reported: a PEFT difference far above the repeat's would point at the backward op sequence.
- **The attention kernel is meta-training's** (FlashAttention 2 when it imports, else SDPA), shared
  by both paths and recorded. Eager attention would materialize $T \times T$ scores per head
  (about 4 GB per layer at 8k tokens) and is not what P3.5 runs.
- **Thresholds.** $10^{-3}$ is B5's bound. 2 ULPs and equal argmax are conflict 3's fallback, the
  human's call: the decode ranks candidates by log-probabilities, so an argmax flip can change a
  decoded grid, while a 1 to 2 ULP wobble with the same argmax cannot change a greedy step. The
  tolerance is per element, and ULPs count only where the bound does not hold. Considered first:
  the largest ULP distance over all elements. Lost: logits near zero have tiny ULPs. A logit of
  0.01 has ULPs of $2^{-14} \approx 6 \times 10^{-5}$, so a difference of $3 \times 10^{-4}$, well
  below the bound, is 5 ULPs; on a logit of $10^{-4}$ (ULPs of $2^{-21}$) it is over 600. Any
  rounding-level difference would then read as "beyond", and the "Human decides" row could
  practically never be reached (`tests/test_inject_parity.py` pins such a case: one logit 1 ULP
  off above the bound and one 5 ULPs off below it is "within"). $10^{-2}$ for the effect is 10
  times the bound: below it a PEFT path that ignored the adapter could pass.
- **A failed backward is a result, not a lost run.** Considered: letting an exception in a backward
  end the run (the first version). Lost: the injected gradients run before PEFT, so an OOM in the
  8k-token text would discard every logit comparison and read as Error, where the brief's rule
  reads Fail ("no gradient"). The reason and the traceback are kept, so an OOM is not mistaken for
  a broken gradient.
- **The zero-B control is bitwise.** With $B = 0$ the delta $s\,B(Ax)$ is exactly $\pm 0$ (sums
  of products with zero), and adding it to the fp32 copy of a bf16 value and rounding back is
  exact, so any difference there is a change to the base's own computation, not a rounding.
- **Repeats and swaps as diagnostics.** If the injected and PEFT logits differ, the repeats say
  whether the stack itself is non-deterministic (then the difference is not inject's) and the swaps
  say which loaders P4.1 may use.
- **Inputs by search, not by path.** The 2026 mount layout differs from 2025's and moved within
  2026 (`docs/STATE.md`, rules for briefs and scripts); the script uses the notebook's search rule,
  so both find the same files (`tests/test_inject_parity.py` compares them on a decoy tree).
- **No checkpoint or resume.** The run takes minutes (section 4); A8 asks for resumability past
  30 minutes. The wall budget (1800 s) stops it with its logs long before the notebook's hard stop
  (2400 s) and Kaggle's `Timeout` (2700 s).
- **`NvidiaL4`, one L4 used.** `NvidiaL4X1` ran on 2xT4 when requested (`P0.3-01`) and T4 has no
  bf16; meta-training runs one episode per L4, so one L4's kernels are the ones that matter.
- **A Kaggle script notebook, not a `@pytest.mark.gpu` test.** No GPU work runs on the dev box
  (BRIEF A0); the run's kernels must be the Kaggle image's (torch 2.8.0, transformers 4.55.4, peft
  0.17.1, flash_attn 2.8.2, `P1.1-01`'s pip freeze), which the script notebook pins.

## 4. How to run it

**On Kaggle** (the orchestrator, after the human approves `docs/prerun/P3.1-01.md`): build from a
clean tree at the brief's `Code-commit`, version the code dataset from the build, push.

```bash
.venv/bin/python scripts/build_submission.py --tag P3.1-01 --config configs/script_P3.1-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P3.1-01/arc26-code -m "P3.1-01 <sha>"
.venv/bin/python scripts/kaggle_ops.py push dist/P3.1-01 --run-id P3.1-01
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p3-1-parity --task-id P3.1-01
```

The notebook attaches the competition, the base model, the install notebook at version 2, and the
code dataset; it runs `python inject_parity.py --config inject_parity.yaml --commit <sha>` in
`/kaggle/working` and kills it at 2400 s. Wait a few minutes after the dataset version before the
push, so the push attaches it.

**On a plain GPU machine** (bf16 GPU, the same stack):

```bash
python scripts/inject_parity.py --config configs/inject_parity.yaml \
    --model-dir /path/to/qwen3_4b_grids15_sft139 \
    --challenges /path/to/arc-agi_evaluation_challenges.json \
    --out-dir runs/P3.1-01 --device cuda:0
```

**CPU smoke** (seconds; writes `runs/P3.1-smoke/`):

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python scripts/inject_parity.py \
    --config configs/inject_parity.yaml --smoke
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/pytest tests/test_inject_parity.py -rP
```

**Expected cost** (estimates, derived in the brief's section 6): imports and model load 1 to 2.5
minutes; 8 forward passes over 9 sequences (about 37k tokens per pass) about 2 minutes; 12
backward passes about 1.5 minutes; about 5 minutes for the script and 7 to 8 for the notebook.
Peak GPU memory about 11 GB of 22.5 GB (the 8k-token backward with checkpointing); `parity.json`
records the measured peak. Disk: two adapters of 47 MB in a temporary folder, removed at the end;
logs under 1 MB.

## 5. Outputs

All in `out_dir` (`/kaggle/working/run_logs/`, pulled to `runs/P3.1-01/`):

| File | Contents |
|---|---|
| `config.yaml` | The resolved config. |
| `parity.json` | `status` (`running`, `complete`, `failed`, `stopped`), `exit_code`, `decision` (`outcome`, `reasons`), `thresholds`, `provenance` (command line, UTC start, commit, versions, platform, `arc26` path), `device`, `precision`, `inputs` (model folder, challenges path and sha256, task ids), `model` (load time, attention kernel, dtype, shape, `checkpointing`: modules and `use_reentrant`), `adapter` (scope, $s$, projections, values, seed, $\rho$, sha256, PEFT's dtype and scale), `sequences` (per sequence: name, kind, key after the cut, tokens, pairs kept and dropped, loss positions, `zero_b`, `parity`, `effect`, `effect_injected`, `repeat`, `swaps`, each comparison with its bound, `above_bound`, `max_ulp_above_bound`, `max_ulp`, and class; `gradient` with flows, losses, relative differences, `calls_per_projection`, and `error`, `repeat_error`, `peft_error` when a backward raised), `peak_memory_bytes`, `elapsed_s`, `error_log`. Written after every stage, so a killed run leaves its last state. |
| `summary.md` | The outcome and its reasons, the run's facts, three tables (logits, gradients, diagnostics), and how to read them. Also printed to the session log. |
| `timing-parity.jsonl` | One `{puzzle_id, op, seconds, tokens}` record per GPU op: `load`, `peft_load`, `forward_<pass>` per sequence (`base`, `zero_b`, `inject`, `base_repeat`, `peft`, `peft_repeat`, `swap_load_adapter`, `swap_state_dict`), `grad_inject`, `grad_inject_repeat`, `grad_peft` per training text. |
| `error.log` | The traceback of a failed run, and of each backward that raised; only when there is one. |

The script notebook adds its own `env.json`, `pip_freeze.txt`, `candidates.json` (`{}`), and
`stages.json` (whether the environment check and install passed, the script's exit code).

## 6. How to read the results

Read the outcome in `summary.md`, then the tables:

- **Pass**: every "Max abs diff" is below $10^{-3}$ (expected 0, "Differing" 0), every zero-B cell
  says `bitwise`, every training text shows "Flows yes" and "Zero tensors 0", and the effect
  column is well above 0.01; 2 calls per projection show the backward ran with checkpointing.
  P3.1 is accepted on the GPU tier; `docs/modules/inject.md` records the result; P3.5 trains
  through `inject` and P4.1 decodes through `PeftModel.from_pretrained`.
- **Human decides**: some logits are $10^{-3}$ or more off, each by at most 2 ULPs, with every
  argmax equal ("At or above bound" above 0, "Max ULP there" at most 2). Look at the
  repeats: if a repeat differs too, the stack is non-deterministic and the difference is noise
  that would also separate two PEFT runs. The human accepts (P3.1 accepted with the measured
  tolerance) or rejects (as Fail) at review.
- **Fail**: an argmax differs, a logit is more than 2 ULPs off, the zero-B control is beyond, or a
  gradient does not flow. Find the differing op: if the zero-B control differs, the wrapping
  changed the base's computation; if only the parity differs and the repeats are bitwise, the
  LoRA path's op sequence or kernel choice differs from PEFT's. Fix `inject.py` (CPU tier first),
  then a new brief.
- **Error**: no conclusion. A non-finite reference or a vacuous adapter (effect below 0.01) means
  the measurement was not valid; an exception or a stop means it did not finish. Pull the logs,
  fix, rerun under an amended, re-approved brief.

Diagnostics, whatever the outcome: a swap row that differs while `from_pretrained` is exact tells
P4.1 to load adapters with `from_pretrained` (or to fix that swap); a gradient difference to PEFT
far above the repeat's points at the backward op sequence; `peak_memory_bytes` and the
`grad_inject` times are the first measured 8k-token backward of the 4B base on an L4, input to
P3.6.

## 7. Failure modes

| Failure | Sign | First thing to change |
|---|---|---|
| No bf16 GPU (Kaggle ran another machine) | The notebook's bf16 stop, or `RuntimeError: ... has no bf16` | Nothing in code: push again on `NvidiaL4` under the same brief only if nothing changed; otherwise an amended brief. |
| Inputs not found, or found twice | `FileNotFoundError: N model folders ...` or `N files ...` | Check the attached sources in `kernel-metadata.json`; the environment check prints what it found. |
| FlashAttention 2 does not load | `model.attn_implementation` is `sdpa` | Nothing: SDPA is the recorded fallback; both paths share it. |
| The real tokenizer breaks the format | `ValueError` from `render_pair` or "tokens of the training text differ" | Compare with P3.2-01's and P3.5-01's preflights; the notebook vocabulary is in `arc26/hyper/train.py` (`STANDIN_VOCAB`). |
| OOM in a backward | Fail: "its backward failed" with `OutOfMemoryError` on that text, its traceback in `error.log`; the logit comparisons and the other texts are kept | Check the calls per projection (2 means checkpointing ran) and `peak_memory_bytes`; lower `max_tokens` only under an amended brief (it changes the sequences). |
| NaN or inf | Error ("non-finite reference") or Fail (beyond) | Lower `adapter.relative_norm` under an amended brief; a NaN only in the injected path is an `inject` bug. |
| Vacuous adapter | Error, effect below 0.01 | Raise `adapter.relative_norm` under an amended brief. |
| Timeout | Exit 3 at 1800 s with logs; or the notebook's kill at 2400 s (`stages.json`: killed; `parity.json` at its last stage) | Read `timing-parity.jsonl` for the slow op; raise `max_wall_s` and `script_kill_s` together under an amended brief. |
| peft version differs from 0.17.1 | A note in `summary.md` | The op sequence `inject` replays may differ: rerun the CPU tier against that version first. |
| Divergence | n/a: nothing is trained. | |
