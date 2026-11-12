# Module: `arc26/hyper/train.py`

The meta-training loop for the hypernetwork H (task P3.5; BRIEF B2 Q1, Q2, Q3, Q6, B5 P3.5).
Our code. It joins P3.1's injection (`inject.py`), P3.2's encoder (`encoder.py`), P3.3's H
(`hypernet.py`), and P3.4's episodes (`episodes.py`); the notebook's own formatter and
augmentation (`arc26.nvarc.arc_loader`) define the token format. Script and runbook:
`scripts/train_hypernet.py`, [`docs/runbooks/train_hypernet.md`](../runbooks/train_hypernet.md).

## Purpose

Train H end to end on Q6's task loss through the frozen base: H maps a task embedding of the
demonstrations to a rank-16 LoRA adapter, the adapted base predicts the query's output grid with
the demonstrations in context, and the loss's gradient reaches H only. The module also verifies
the episode set it trains on, validates against two controls, checkpoints, resumes, and enforces
its own kill criteria. Task MT adds source weights for the episode order (BARC's share raised for
runs `MT-02` and `MT-03`) and `best.pt`, H's weights at its best validation point.

## Math

An episode is $(D, Q, a)$ with $Q = (x_Q, y_Q)$ and augmentation $a = 8j + 4t + k$. With
$\bar e_i = \frac{1}{|A|} \sum_{v \in A} \frac{1}{|O_{i,v}|} \sum_{t \in O_{i,v}} h^{(L)}_t$
(P3.2, view set $A$, layer $L$) and $\bar z(D) = \frac{1}{|D|} \sum_{i \in D} \bar e_i$ over the
**un-augmented** demonstrations,

$$\mathcal{L}(D, Q, a) = -\frac{1}{|O_Q|} \sum_{t \in O_Q}
\log p\big(y_t \mid y_{<t};\ \theta_\text{base} + \Delta\theta_{H(\bar z(D))}\big),$$

where the sequence is $a(D)$ then $a(Q)$ in `QwenFormatter.fmt_train`'s format and $O_Q$ the
query's output-grid positions plus, with `loss_includes_eos` (the default), the position of the
`<|im_end|>` that closes the grid (`query_ce`: logit row $t - 1$ predicts token $t$).

**Reading of Q6.** Q6 scores "output-grid tokens". The closing `<|im_end|>` is read as one of them:
the notebook's completion-only training (`QwenDataCollatorForCompletionOnlyLM`) trains each
reply's grid tokens and its end token, and H's adapter serves the notebook's decode, where that
token ends a grid, so it carries the grid's last row and with it its shape. `loss_includes_eos:
false` restores the narrower reading, the grid's digits and newlines alone. Each projection
$W$ of q, k, v, o computes $W x + s B A x$, $s = \alpha / \sqrt{r} = 8$, in fp32. An optimizer step
averages $\mathcal{L}$ over $W \cdot A$ episodes: $W$ processes, $A$ micro steps each, slot
$g = (s A + m) W + k$, pass $\lfloor g / N \rfloor$ a seeded permutation of the $N$ episodes.
With source weights $w_s$ (task MT), slots form blocks of $B = 1000$; block $b$ holds
$c_s = \mathrm{round}(B w_s / \sum_t w_t)$ slots of source $s$ (largest remainders, so
$\sum_s c_s = B$) in a seeded shuffle, and the slot that is source $s$'s $k$-th draw,
$k = b c_s + j$ for its rank $j$ in block $b$, takes position $k \bmod N_s$ of a seeded
permutation of that source's $N_s$ episodes for pass $\lfloor k / N_s \rfloor$. Over $D$ draws,
source $s$ is drawn $D c_s / B$ times (exactly, at whole blocks), each of its episodes about
$D c_s / (B N_s)$ times.
Learning rate multiplier $m(n) = n / n_w$ for $n \le n_w$, then
$\rho + (1 - \rho) \tfrac12 (1 + \cos(\pi \frac{n - n_w}{T - n_w}))$. Validation reports, per
episode $i$, $\mathrm{CE}_0(i)$ (no adapter), $\mathrm{CE}_\text{cur}(i)$ (H on $\bar z_i$), and
$\mathrm{CE}_\text{mis}(i)$ (H on $\bar z_{\pi(i)}$, $\pi$ pairing other families), their means,
and the paired differences $\mathrm{CE}_\text{cur} - \mathrm{CE}_0$ and
$\mathrm{CE}_\text{mis} - \mathrm{CE}_\text{cur}$ with standard errors and their ratios $z$
(criterion B and check C). Under `zero_step`, `best.pt` holds H at the lowest
$\overline{\mathrm{CE}}_\text{cur}$ over every validation of every session; under `fomaml`, at
the best post-$k$ point (next section).

## The `fomaml` objective and the decode-aligned metrics (P3.9)

`objective: zero_step` (the default) is the loss above, unchanged bit for bit. `objective:
fomaml` (ADR-006, amendment 2; `docs/tasks/P3.9.md`) trains H for the start of the fine-tuning:
per episode, $\theta_0 = H(\bar z)$, then $k$ = `inner.steps` inner steps on the demonstrations,

$$\theta_{j+1} = \theta_j - \alpha\,u_j, \qquad
g_j = \nabla_\theta \mathcal L_{D}^{(j)}(\theta_j),$$

where $\mathcal L_D^{(j)}$ is the mean CE over every demonstration's output tokens (with its
closing `<|im_end|>` under `loss_includes_eos`) of $b_j(D)$, the demonstrations alone under view
$b_j$: the episode's own $a$ (`inner.views: episode`) or the $j$-th id of a shuffle of the build's
augmentations seeded by the run seed and the episode id (`drawn`, the fit's one view per step).
$\alpha$ = `inner.lr`; $u_j = g_j$ (`sgd`) or Adam's bias-corrected step with moments that start
at zero for every episode (`adam`):
$m_j = \beta_1 m_{j-1} + (1 - \beta_1) g_j$, $v_j = \beta_2 v_{j-1} + (1 - \beta_2) g_j^2$,
$u_j = \frac{m_j / (1 - \beta_1^{j+1})}{\sqrt{v_j / (1 - \beta_2^{j+1})} + \epsilon}$; no weight
decay, as the notebook's fit. Each $g_j$ is first clipped as the fit clips its own
(`inner.clip` $\tau$ = 1.0, the Trainer's `max_grad_norm` in `arc26/solver.py`; torch's
`clip_grad_norm_` formula over every factor of the episode, `clip_inner_grads`):

$$g_j \leftarrow c_j\,g_j, \qquad c_j = \min\Big(1, \frac{\tau}{\lVert g_j\rVert_2 + 10^{-6}}\Big).$$

Under `adam` the first step is $u_0 = g_0 / (|g_0| + \epsilon)$, the sign of $g_0$ whatever $c_0$
is (up to $\epsilon$); from the second step on the clip changes the update, since
$m_1 = \beta_1 (1 - \beta_1)\,c_0 g_0 + (1 - \beta_1)\,c_1 g_1$ weighs the two gradients by
their clip factors (and $v_1$ likewise); under `sgd` the step is $\alpha c_j g_j$. Without the
clip the start would be trained for a step the fit never takes when $c_0 \ne c_1$. The outer
loss is the query CE at $\theta_k$, and the first-order gradient H receives is

$$\nabla_\phi \approx J_\phi^\top\, \nabla_\theta \mathcal L_Q(\theta_k), \qquad
J_\phi = \partial \theta_0 / \partial \phi,$$

that is, $\partial\theta_k / \partial\theta_0$ is taken as $I$ (no second derivatives).

Mechanics (`adapt`, `fomaml_micro_step`): $\theta_0$ is detached and copied into leaf tensors
(47 MB fp32 at H's scope: 11,796,480 values); each inner step's forward and
`torch.autograd.grad` run inside one `inject` block (so activation checkpointing recomputes under
that step's factors, research note risk 2), and the in-place update runs after the block; the
query's forward and `torch.autograd.grad` at $\theta_k$ run inside another; then
`torch.autograd.backward(theta_0, g)` sends $g$ through H once (DDP's all-reduce hooks fire there,
inside `no_sync` on the micro steps before the last). Nothing reaches the base (frozen) or H
during the inner steps (no `.grad` is written). With $k = 0$ the micro step is `micro_step`
itself. `adam` allocates its two moments only when a later step reads them ($k \ge 2$). Inner
state beyond `zero_step`'s, computed: $k = 1$ the factor copy, 47 MB; $k = 2$ the copy and two
moments, $3 \times 47 = 141$ MB (plus the transient gradient, 47 MB, as `zero_step` has),
against about 12.7 GiB reserved of an L4's 24 GiB; accepted by the orchestrator, 18 Oct.

A `--init-from` from a `zero_step` checkpoint into a `fomaml` config with the same parameters
keeps the checkpoint's step, data cursor, AdamW state, and cosine, as a resume would
(`init_from.objective` in `run.json` and the checkpoints). With `fresh_schedule: true` it starts
a new training phase instead: H's weights load from the checkpoint (bitwise), every parameter
gets one AdamW with fresh moments, and `WarmupCosine` runs over the phase's own steps, from the
checkpoint's step $s_0$ to `steps`: phase step $n$ (global step $s_0 + n$, which the logs
show) has learning rate `lr` $\cdot\, n / W$ for $n \le W$ = `warmup_steps`, then the cosine to
`min_lr` at the last step. The schedule keys (`SCHEDULE_KEYS`) may then differ from the
checkpoint's; `init_from.fresh_schedule` records $s_0$, the phase length, the warmup, and the
rates, and `--resume` rebuilds the same optimizer and schedule from it before loading their
state. `objective`, `inner`, and `fresh_schedule` (`OBJECTIVE_KEYS`) are `RESUME_KEYS`: an
absent key reads as its default (`zero_step`, none, false), so the checkpoints written before
P3.9 (`MT-02`, `MT-03`, `X1-01`) resume as before, and a `fomaml` checkpoint never resumes under
`zero_step` or another $k$, inner learning rate, optimizer, or view rule.

Logged per training step (`inner`): the inner rate (`lr`), the demonstrations' CE at
$\theta_0, \ldots, \theta_{k-1}$, each inner step's gradient norm before the clip (`grad_norm`),
the tokens the step's inner steps read (`tokens`, summed over its episodes; the record's own
`tokens` stays the query sequences', comparable with a `zero_step` run), and per module type the
means over layers and the step's episodes of the inner displacement
$\lVert s(B_kA_k - B_0A_0)\rVert_F$ and of the start norm $\lVert sB_0A_0\rVert_F$, both from
$\lVert BA\rVert_F^2 = \mathrm{tr}((B^\top B)(AA^\top))$ on stacked factors
($B_kA_k - B_0A_0 = [B_k, -B_0][A_k; A_0]$, Gram matrices of side $2r$, fp64).

Every validation now also records, per reference, `decode`: the mean $|O_Q|$, the mean summed NLL
$S = |O_Q|\,\bar\ell$, the share with $S < \ln 5$ (the decode's per-view bound), the teacher-forced
greedy exact match (every argmax over $O_Q$'s rows is the target), and the geometric mean CE with
floor $10^{-6}$; `per_episode_exact` holds the matches. Under `fomaml` with $k \ge 1$, `adapted`
scores the current H after its $k$ inner steps on the same episodes (H in evaluation mode, the
same views as training): CE, per source, `decode`, the paired differences against the current H
and the zero adapter, and the displacement and start norm per module type and per projection.
`adapted` also holds each inner step's mean gradient norm (`grad_norm`).
`best.pt` follows the objective: `zero_step` the lowest current-H CE, `fomaml` the highest post-$k$
greedy exact match, ties by the lowest mean $S$, then the earliest. Timing ops added:
`inner_step` (each training inner forward and backward, tokens of $b_j(D)$), `h_backward` (the
outer gradient through H), `validate_inner_step` (validation's inner steps, apart from
training's), `validate_adapted` (the post-$k$ query forward).

**The inner-rate probe** (`inner_probe`, run `P3.9-01`). Adam's first inner step moves every
factor value by about $\alpha$ whatever the gradient's scale, so $\alpha$ is judged on the
warm-started H before training: at the start step of a `--init-from` run, validation scores
`adapted` at every rate of `inner_probe.lrs` (`inner.lr` must be one of them), and the run trains
at

$$\alpha^* = \operatorname*{arg\,min}_{\alpha \in C} \bar S(\alpha), \qquad
C = \{\alpha : E(\alpha) \ge \max_\beta E(\beta) - m\},$$

with $E(\alpha)$ the post-$k$ greedy exact matches (episodes), $\bar S(\alpha)$ the mean summed
NLL, $m$ = `inner_probe.margin`, over the rates with finite scores, ties to the smaller rate
(`choose_inner_lr`). Every rank computes the pick from the same all-reduced scores and rank 0's is
broadcast. The validation record's `probe` holds each rate's scores and full `adapted` block, the
pick, and the probe's extra seconds (left out of the clock's validation mean, which the budget and
the projection read); `init_from.inner_probe` (in `run.json` and every checkpoint) records the
pick, and the checkpoints' config has `inner.lr` = the pick. The run then checkpoints its start
step, so a restart resumes at the pick without probing again (`--resume` applies
`init_from.inner_probe.chosen` to the YAML's `inner.lr` before comparing the resume keys).

**Starts.** `fresh_schedule` and `inner_probe` act at a warm start's start step, so `check_start`
refuses either without `--init-from` (or a `--resume` of such a run), before any data or model
is read; the script exits 2. A new phase may start from a phase's checkpoint (`--init-from` a
P3.9 checkpoint with `fresh_schedule: true`: the longer run a Pass leads to), its
`init_from.previous` keeping the earlier warm start's record; without `fresh_schedule` a
warm-started checkpoint is refused (`--resume` it instead).

**`best.pt` for loaders on older code.** `best.pt`'s `config` is `h_config`: the run's config
without the `OBJECTIVE_KEYS`, which describe how H was trained and not how to build it, its
encoder, or its validation. So a `fomaml` H's `best.pt` holds the config keys of the code before
P3.9, and P3.7's load check (`arc26/hyper/validation.py` on `task/P3.7`, which parses the config
with that branch's strict `TrainConfig.from_dict`) reads it as it reads `MT-03`'s; the objective
keys are kept in `best.pt`'s `objective`. That branch needs no change for `best.pt`; a full
checkpoint (`ckpt-*.pt`) keeps the whole config, so loading one there needs P3.9's `train.py`
(merge `task/P3.9` first). P3.7's `logged_best` reads `status.json`'s `best.step`, which under
`fomaml` is the post-$k$ best point, and its 0-step CE per episode, which every validation
record still carries.

Cost (estimate, survey section 10): an inner step reads $b_j(D)$, 71.6% of an episode's tokens,
so a step costs about $(1 + 0.716k)$ times `MT-02`'s 11.42 s: 19.6 s at $k = 1$, 27.8 s at
$k = 2$, on 4xL4 with 32 episodes; a $k = 2$ validation adds about 170 s for 286 episodes.

## Interfaces

| Name | Contract |
|---|---|
| `TrainConfig`, `EncoderSettings`, `KillCriteria` | Frozen dataclasses, validated on construction; `TrainConfig.from_dict` refuses missing and unknown keys (`loss_includes_eos` may be left out: true; `source_weights` too: null); `to_dict` gives plain values. `kl_weight` must be 0 (Q6's KL term is an interface only). `source_weights` maps sources of `episodes.SOURCES` to weights $\ge 0$ (one positive), stored sorted as floats; a resume setting. |
| `open_episode_set(episodes_root, manifest_sha256, check_root, max_tokens)` | Finds the P3.4 build by `MANIFEST.json`, refuses another sha256, a missing check report, a failed one, or a build cap above `max_tokens`; `EpisodeSetError`. |
| `load_split(episode_set, split, max_tokens)` | Shards checked against the manifest (size, sha256, count), records validated (`read_shards`); over-length records raise `RecordError`. |
| `augmented_pairs`, `demo_hashes`, `build_sequence` | $a(D), a(Q)$ through `forward_mod`; P3.2's `pair_hash` of the un-augmented demonstrations; the token ids `[seq]` and $O_Q$ (with the closing token when `include_eos`), checked against `render_pair` and the whole text's tokens. |
| `query_ce(logits, input_ids, target)` | `logits [seq, vocab]`, `input_ids [seq]`; fp32 mean CE over $O_Q$. |
| `factor_penalty(factors, kind)` | Mean $|v|$ (`l1`) or $v^2$ (`l2`) over every emitted entry. |
| `demo_pairs`, `PairEmbeddings` | Distinct demonstrations by hash; `fill` reads an `EmbeddingCache`, then encodes the rest in chunks shared by the ranks; `z_bar(hashes)` is `[hidden]` fp32. |
| `train_slot`, `EpisodeOrder`, `block_counts`, `mismatch_permutation` | The data order (unweighted, or by source weights in blocks of `SOURCE_BLOCK`; `source_draws(slots)` counts a weighted order's draws per source) and the control's pairing, pure functions of the seed. |
| `param_groups`, `WarmupCosine` | AdamW groups (decay on 2-D weights except the codes); the `LambdaLR` multiplier. |
| `standin_tokenizer`, `tiny_base`, `resolve_attention`, `load_base` | The smoke stand-ins and the real base (frozen, eval mode, `use_cache` off, FlashAttention 2 or SDPA). |
| `save_checkpoint`, `save_best`, `resolve_checkpoint`, `load_checkpoint`, `resume_conflicts` | Atomic checkpoints with `latest.json` and pruning; `best.pt` (kind `best`: H's weights, config, validation record; never pruned, never resumed); the `RESUME_KEYS` that must match. |
| `run_worker`, `train(cfg, resume, started)` | One process of a run; the entry point that spawns `gpus` processes (or runs in this one) and returns the exit code from `status.json` (0, 3, 4). |
| `InnerLoop`, `OBJECTIVES` | The inner loop's settings (`steps`, `lr`, `optimizer` from `INNER_OPTIMIZERS`, `betas`, `eps`, `views` from `INNER_VIEWS`, `clip`, null for none); `TrainConfig.objective` and `TrainConfig.inner` (set exactly for `fomaml`); `to_dict` leaves both out for `zero_step`. |
| `InnerProbe`, `choose_inner_lr` | The probed rates (ascending, distinct, `inner.lr` among them) and the margin; the pick from each rate's exact matches, mean $S$, and mean CE (raises when none is finite). |
| `check_start`, `h_config` | The refusal of `fresh_schedule` or `inner_probe` without a warm start (and of `--resume` with `--init-from`); `best.pt`'s config, `to_dict` without the `OBJECTIVE_KEYS`. |
| `build_demo_sequence`, `demo_ce`, `inner_view_ids` | $b(D)$'s tokens `[seq]` and every demonstration's output positions; the fit's mean CE over them; the views of the $k$ inner steps, a pure function of seed and episode id. |
| `InnerOptimizer`, `clip_inner_grads`, `adapt`, `flat_factors` | The in-place update; the global-norm clip, returning the norm before it; the inner loop from $\theta_0$, returning $\theta_k$ (fresh leaves), the inner CEs, and the gradient norms (timing op `inner_step`, or `op`). |
| `displacement`, `delta_norms`, `by_module` | Per projection $\lVert s(B_kA_k - B_0A_0)\rVert_F$ and $\lVert sBA\rVert_F$ through Gram matrices; means by module type. |
| `query_scores`, `decode_metrics`, `is_better`, `selection_metrics` | CE and greedy exact match from one forward; the decode-aligned summary of a reference; the best-point rule per objective. |
| `objective_conflicts` | The `OBJECTIVE_KEYS` that differ (an absent key reads as its default); `resume_conflicts` uses it for those keys of `RESUME_KEYS`; `init_conflicts` lets them differ, and the `SCHEDULE_KEYS` too under `fresh_schedule`. |
| `MetricsLog`, `validation_record`, `best_validation`, `read_status` | `metrics.jsonl`, rewritten from a checkpoint's history on resume; one validation's record (with `z`; the worker adds `best` and `session_start`); the best validation in a history (`is_better` for the objective); the run's status (with `best`). |

Exit codes: `EXIT_OK` 0, `EXIT_FAILED` 1, `EXIT_CONFIG` 2, `EXIT_STOPPED` 3 (session budget,
resumable), `EXIT_KILLED` 4 (kill criterion).

## Data flow

`open_episode_set` and `load_split` give `EpisodeRecord`s; `load_base` and `wrap_targets` give the
frozen base with injectable projections; `PairEmbeddings.fill` encodes every distinct
demonstration once; per step, each process draws its episodes by slot, `build_sequence` renders
them, H emits the factors from $\bar z$, and forward, loss, and backward run inside `inject`;
DDP all-reduces H's gradients; AdamW steps; rank 0 appends to `metrics.jsonl`, validates on
schedule, rewrites `best.pt` at a new best validation point (`is_better`), and writes
checkpoints; every process writes `timing-rank<k>.jsonl`.

## Origin

New code. The training recipe follows D2L's trainer (AdamW, cosine with a minimum lr, warmup,
clipping, no decay on norms, biases, and codes; research note `docs/research/d2l_t2l_reading.md`,
1.4); no D2L or T2L code is copied. The stand-in tokenizer repeats `scripts/cache_embeddings.py`'s
(a library copy, because a script notebook ships one script and the spawned processes import the
library).

## Tests

`tests/test_train.py` (CPU, tiny random Qwen3, fixture episodes in
`tests/testdata/train_hypernet/`): the loss mask covers exactly the query's output tokens (spans
decode back to the grid text, the whole text tokenizes to that of every pair of $D$ and $Q$ under
the record's augmentation, applied with P3.4's numpy path, not the notebook's that the code
uses), plus the closing `<|im_end|>` with `loss_includes_eos`, its default, which a resume must
share; the CE a train step returns equals an independent forward's CE over $O_Q$ (and differs
from the CE over every token); `query_ce` reads
only the target rows; $\bar z$ ignores $a$ and equals P3.2's mean of per-pair view means; step 0
equals the base bitwise in fp32 (and the step-0 validation's three references agree); gradients
reach every parameter of H and none of the base; checkpointing on and off give bitwise equal
gradients (and the layer does run twice); a resume after 3 of 6 steps equals the uninterrupted
run bitwise (parameters, data order, losses); the session budget stops with a checkpoint and the
rerun finishes identically; gloo DDP with 2 processes matches one process with accumulation 2
(bitwise on the dev box; the test allows rtol $10^{-5}$), validation records included; the zero,
current, and mismatched CEs of a trained H, per episode and per source, equal an independent
recomputation with H in evaluation mode; the encoder's views are the configured set; pairs in an
`EmbeddingCache` of the encoder's namespace are read bitwise and not encoded, a cache of another
namespace is not read; clipping bounds the update; the manifest and check guard (another
manifest, a missing, foreign, or failed report, a `pass` that is not `true`, a tampered shard, a
larger cap); a failure in a worker process names the worker's exception; over-length
episodes are refused; the kill criteria fire (step time, projected wall time, a non-finite loss
whose update is never applied); the smoke run exits 0 well under 2 minutes, its training CE falls
on the memorizable fixture, its outputs match the pull allowlist and the timing schema, and its
`--resume` rerun from step 8 ends with the same H; the full config's smoke runs accumulation with
DDP; both configs validate and copy `configs/hypernet.yaml`; the committed fixture equals its
generator. `tests/test_train_mt.py` (task MT): `block_counts` rounds by the largest
remainder; without weights the order is the unweighted one (a checkpoint written before the key
existed resumes); every aligned block holds the shares exactly and, on pools whose passes span
blocks (2000, 600, 400 episodes), each source shows each of its episodes once per pass of its own
for two passes; the weighted order is a pure function of seed and slot and scale invariant in the
weights; `validation_record`'s means, standard errors, and `z` (signs included) equal a hand
computation, and `z` is None for a constant difference; weights must name exactly the split's sources (and are validated); a
weighted run draws its order and resumes bitwise, and a checkpoint of other weights is refused;
`best.pt` is written at each strict new low only (scripted validation CEs), holds H alone, equals
the full checkpoint of its step, and a resumed session rewrites it only for a new low over both
sessions while the status names the session that holds it; `best.pt` loads as P4.1's provider
loads H and is never resumed, and two DDP processes write it with H's own keys (a plain Hypernet
loads it strictly); the MT config matches its brief, its steps fit two sessions at
P3.6-01's step, its comment states the draws its weights give, its encoder namespace is MT-01's
cache's, every value has a comment, and its smoke runs. `tests/test_train_fomaml.py` (P3.9):
$k = 0$ under `fomaml` equals `zero_step` bitwise (H, losses, gradient norms, validations); the
first-order gradient H receives matches central finite differences of
$\phi \mapsto \mathcal L_Q(H_\phi(\bar z) + \Delta)$ with $\Delta = \theta_k - \theta_0$ held
fixed, along the gradient and a random direction; the inner steps leave the base, H's
parameters and gradients, and $\theta_0$ untouched and return fresh leaves; the `adam` update
equals `torch.optim.Adam` and `sgd` is $\theta - \alpha g$, and $k = 1$ keeps no moments;
checkpointing on and off give bitwise equal `fomaml` gradients (every forward runs again in its
backward); the demonstrations under the episode's view are the prefix of the episode and their
targets hold each augmented output grid and its closing token; the inner views are a pure function
of seed and episode; two runs and a resume after 3 of 6 steps are bitwise equal, and a
`zero_step` resume of a `fomaml` checkpoint is refused; gloo DDP with 2 processes matches one
process with accumulation 2 (post-$k$ validation included); the `inner` and `adapted` blocks,
the timing ops, `status.json`, and `best.pt` carry the new fields, and the post-$k$ CE and
exact match equal an independent recomputation; `--init-from` a `zero_step` checkpoint changes
the objective and continues the data order and the cosine, and the same objective is refused;
a `fresh_schedule` warm start loads H bitwise with empty AdamW state, logs global steps from the
checkpoint's, has the learning rate `lr`/2, `lr`, and `min_lr` at phase steps 1, $W$ = 2, and the
last, records the phase in its checkpoints, resumes mid-phase to the same H and rates, and is
refused when the phase is not longer than its warmup or a resume drops the flag; MT's and
X1-01's configs read as `zero_step` and raise no resume conflict; the P3.9 smoke warm-starts
from the MT smoke; the script's smoke exits 2 without `--init-from` (nothing written) and 0
from the MT smoke's checkpoint, with the decode and probe tables; `fresh_schedule` and
`inner_probe` without a warm start are refused before anything is written; the inner clip equals
torch's `clip_grad_norm_` (and leaves a norm under it alone), scales `sgd`'s step by $c_j$, and
leaves Adam's first step unchanged; `choose_inner_lr` on hand cases (margin, ties, non-finite
rates); a probed warm start logs every rate's block, trains at the rule's pick (steps,
checkpoints, `run.json`), scores the start step as a probe-free run at the pick does and ends
bitwise equal to it, resumes from its start-step checkpoint bitwise without a second probe, and
a resume that drops the probe is refused; a new phase starts from a phase's checkpoint (and
without `fresh_schedule` is refused); a `fomaml` `best.pt` holds only the config keys of the code
before P3.9, parses with `from_dict`, loads H strictly, and reproduces the logged 0-step CE per
episode at `status.json`'s best step, as P3.7's load check does; the training records' inner
tokens equal the `inner_step` timing records', and validation's inner steps are timed apart;
`query_scores`, `decode_metrics`, `displacement`, `delta_norms` (against dense norms), and the
`fomaml` selection rule on hand cases; the P3.9 config keeps MT's resume keys and comments
every key.

## Consumers outside the training script

`scripts/train_throughput.py` (task P3.6, [runbook](../runbooks/train_throughput.md)) times this
module's training step: its `ThroughputWorker` subclasses `_Worker`, replaces only `setup`
(synthetic episodes and one fixed $\bar z$ instead of P3.4's build and the encoder), and calls the
inherited `train_step`. A change to what `train_step` and `micro_step` read from the worker
(`cfg`, `model`, `hypernet`, `h_train`, `optimizer`, `scheduler`, `comm`, `clock`, `timing`,
`metrics`, `train_records`, `order`, `table`, `formatter`, `embeddings`, `step`) or to
`_Worker.setup`'s order and seeds must keep `tests/test_train_throughput.py` green: it checks that
the two setups leave the same model, H, optimizer, schedule, and random state, and that three
steps give the same H, bitwise on the CPU.

## Known limits

- The real tokenizer and FlashAttention 2 path run only on Kaggle; the CPU tests use the
  stand-in tokenizer and SDPA.
- One adapter per forward: no packing of several episodes per micro step.
- A resumed session re-encodes the embeddings unless an `EmbeddingCache` holds them.
- `resume_conflicts` compares settings, not the episode build's content; the manifest sha256 is
  one of the settings.
- The per-pass paired comparison in `summary.md` needs two complete passes; a single-pass run
  (the long-run defaults), or a source-weighted one (no pass then shows every episode once),
  falls back to the first and last 5% of the steps.
- `best.pt` is selected on the validation split it is reported on, so its validation CE is
  optimistic by the selection over about 29 points; P3.7 confirms on DEV40.
- `fomaml` is first order: $\partial\theta_k/\partial\theta_0 = I$, so H is not trained
  through the inner steps' curvature (AvgGradInner enters only through where $g$ is taken). A
  $k \le 2$ loop tests the direction of a start trained for the fit, not the fit's 128 steps
  (short-horizon bias, survey 9.4).
- The greedy exact match is teacher forced: it equals greedy decoding of the target except at
  argmax ties, and it is not the DFS decode with its views and selection.
- The probe picks the inner rate on the validation episodes it is then reported on, so the start
  step's post-$k$ scores at the pick are optimistic by a selection over the probed rates; a
  start-to-end comparison at the pick is therefore biased against the end, not for it.
- The validation's zero-adapter exact matches enter checkpoints from P3.9 on; a run resumed or
  warm-started from an older checkpoint scores the zero adapter once more at its next
  validation (its CE stays the checkpoint's).

## Deviations from the task brief

1. Embeddings are encoded in one sharded pass before step 0 instead of lazily: each pair once per
   run instead of once per process, no encoder time inside steps, and embeddings independent of
   the process count and of a resume (same values, same encoder, same base).
2. The zero-adapter CE is computed once (the base is frozen) and carried by the checkpoint.
3. A session budget (`max_wall_s`, exit 3) beside the kill criteria (exit 4), and a checkpoint at
   every stop, for runs that span Kaggle sessions.
4. The fixture lives in `tests/testdata/train_hypernet/`, not `tests/data/`: `.gitignore`'s
   `data/` rule (for competition data) matches `tests/data/` too, and `tests/testdata/` is the
   repository's test-data folder (P0.5d); its build folder is `episode_build/` because `build/`
   is ignored as well.
