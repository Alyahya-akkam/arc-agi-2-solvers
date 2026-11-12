# Runbook: `scripts/train_hypernet.py` (P3.5, meta-training of the hypernetwork H)

Task P3.5 (BRIEF B2 Q1, Q2, Q3, Q6, B5 P3.5, A8). Library `arc26/hyper/train.py`
([module doc](../modules/train.md)); configs `configs/train_hypernet_toy.yaml` (run `P3.5-01`),
`configs/train_hypernet_toy_P3.5-02.yaml` (run `P3.5-02`: `P3.5-01` with half the steps), and
`configs/train_hypernet.yaml` (long-run defaults, not run in P3.5); pre-run briefs
[`docs/prerun/P3.5-01.md`](../prerun/P3.5-01.md) and
[`docs/prerun/P3.5-02.md`](../prerun/P3.5-02.md); result
[`docs/experiments/P3.5-01.md`](../experiments/P3.5-01.md). Task MT, the full-scale runs:
`configs/train_hypernet_MT.yaml` (runs `MT-02`, `MT-03`, and `MT-04`, three sessions of one
5,600-step run on `MT-01`'s 8-view cache, BARC's share raised; `MT-02` capped at one week's quota
by `--max-wall-s`, amendment of 4 Oct), built from `configs/script_MT-02.yaml`,
`configs/script_MT-03.yaml`, and `configs/script_MT-04.yaml`; briefs
[`docs/prerun/MT-02.md`](../prerun/MT-02.md) (revision 2), [`docs/prerun/MT-03.md`](../prerun/MT-03.md)
(revision 3), and [`docs/prerun/MT-04.md`](../prerun/MT-04.md). Decisions: ADR-001 (encoder), ADR-002
(scope), ADR-006 (objective), ADR-012 (episode sources). Research notes:
`docs/research/d2l_t2l_reading.md` ("D2L note") and `docs/research/meta_training_sources.md`
("episode note").

## 1. Why this script exists

H must learn, from tens of thousands of puzzles, to emit the adapter that test-time training
finds by SGD (Q3, Q6). This script trains H end to end on the task loss through the frozen base.
Run `P3.5-01` answers B5's acceptance question before any long run is priced: on P3.4's 200-puzzle
toy split, does the training loss decrease, and does the validation episode CE on held-out
generator families beat the zero adapter within 2k steps? Its result gates P3.6 (throughput and
memory, which feeds P0.4) and the meta-training launch (A7 week 3), whose checkpoint P3.7 and gate
G4 read. `P3.5-01` was Borderline: the validation gain held at steps 200 to 800 and faded by step
1000 while the training CE fell 88% over four passes. Run `P3.5-02` makes the one change its brief
pre-registered for a faded gain, half the steps (two passes), and asks whether the gain and its
dependence on $\bar z$ then last to the end of training.

**The full-scale runs (`MT-02`, `MT-03`, `MT-04`, task MT).** Both toy runs were Borderline: a
held-out gain early in training that faded, and no dependence on $\bar z$ (check C) at their ends
(`docs/experiments/P3.5-01.md`; `P3.5-02`'s report). At full scale, 51,048 tasks instead of 200,
the question that decides whether Phase 4 is worth its quota is whether H beats the zero adapter
and uses $\bar z$. The run trains a 5,600-step cosine schedule with BARC's share raised and keeps
H at its best validation point, over three sessions: `MT-02` (revision 2) is capped at what the
week of 4 Oct's quota covered and stopped at step 1,028 (`docs/experiments/MT-02.md`); `MT-03`
(revision 3) resumes it for a session of 38,700 s, the config's budget less a 45-minute pre-script
allowance, to about step 4,170 to 4,240. `MT-02`'s step-1,000 read (B and C together) already
meets the Continue row, so `MT-03` decides only Kill or failure (whether `MT-04` may resume its
checkpoint to finish the cosine) and gives the later validations and the best point so far; P3.7
(gate G4) reads the best checkpoint.

## 2. What it does, step by step

1. **Config and data guard.** The YAML config is resolved (smoke section, command-line paths) and
   written to `config.yaml`. P3.4's build is found under `episodes_root` by its `MANIFEST.json`;
   a manifest whose sha256 is not `manifest_sha256` is refused, and so is a build without a
   `leakage.json` naming that sha256 with `pass: true`, or with any such report failing. Every
   shard of the two splits is checked against the manifest's size and sha256, and every record
   against P3.4's schema (`read_shards`), which includes the 8192-token cap: an over-length
   episode stops the run instead of being cut.
2. **Base.** Each process loads the base in bf16, frozen, with FlashAttention 2 when `flash_attn`
   imports (else SDPA), wraps `q_proj`, `k_proj`, `v_proj`, `o_proj` of all 36 layers in P3.1's
   `InjectableLinear`, and enables non-reentrant activation checkpointing per decoder layer.
3. **Task embeddings.** For every distinct un-augmented demonstration pair $i$ of both splits,
   P3.2's encoder on the same base gives
   $$e_{i,a} = \frac{1}{|O_{i,a}|} \sum_{t \in O_{i,a}} h^{(24)}_t, \qquad
   \bar e_i = \frac{1}{16} \sum_{a} e_{i,a}, \qquad
   \bar z(D) = \frac{1}{|D|} \sum_{i \in D} \bar e_i,$$
   with $h^{(24)}$ the residual stream after 24 of 36 layers, $O_{i,a}$ the output-grid
   positions of pair $i$ under view $a$, and the decode path's 16 views. Pairs come from an
   `EmbeddingCache` when one is configured; the rest are encoded once, before step 0, in chunks
   of `chunk_pairs` spread over the processes and all-gathered.
4. **Episode to sequence.** An episode $(D, Q, a)$ gets augmentation
   $a = 8j + 4t + k$ (transpose if $t$, $k$ quarter turns, color permutation $j$ of the build's
   table) through the notebook's `ArcDataset.forward_mod`. The base reads $a(D)$ then $a(Q)$ as
   `QwenFormatter.fmt_train` renders them, pair by pair
   `<|im_start|>user\n{x}<|im_end|><|im_start|>assistant\n{y}<|im_end|>`, and $O_Q$ is the set of
   positions of the query's output-grid tokens (P3.2's `render_pair` spans, offset) and, with
   `loss_includes_eos` (the default), of the `<|im_end|>` that closes the grid.
5. **Adapter.** H maps $\bar z(D)$ to $(A_{\ell,m}, B_{\ell,m})$, $A \in \mathbb{R}^{16 \times
   d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times 16}$, for every layer $\ell$ and module
   $m$ (11,796,480 values). Inside `inject`, each wrapped projection computes
   $W x + s\, B A x$ with the rsLoRA scale $s = \alpha / \sqrt{r} = 32 / 4 = 8$, the delta in fp32.
6. **Loss and gradient (Q6).** Teacher forced, demonstrations in context,
   $$\mathcal{L}(D, Q, a) = -\frac{1}{|O_Q|} \sum_{t \in O_Q} \log p\big(y_t \mid y_{<t};\
   \theta_\text{base} + \Delta\theta_{H(\bar z(D))}\big),$$
   one episode per process per micro step, `accum` micro steps per optimizer step, so a step's
   gradient is that of the mean of $\mathcal{L}$ over $W \cdot A$ episodes ($W$ processes). Forward,
   loss, and backward run inside `inject`; the gradient flows through the frozen base into the
   factors and into H only. An L1 or L2 penalty on the factors and Q6's KL term exist as config
   keys and are off (the KL term is an interface only, A7 cut order item 4).
7. **Update.** AdamW (decay 0.01 on 2-D weights; none on norms, biases, layer and module codes),
   gradient-norm clip 1.0, learning rate $\eta_n = \eta \cdot m(n)$ for step $n$:
   $m(n) = n / n_w$ for $n \le n_w$, else
   $m(n) = \rho + (1 - \rho) \tfrac12 \big(1 + \cos(\pi \frac{n - n_w}{T - n_w})\big)$ with
   $\rho = \eta_\text{min} / \eta$.
8. **Data order.** Draw $m$ of step $s$ on rank $k$ takes slot $g = (s A + m) W + k$; pass
   $e = \lfloor g / N \rfloor$ of the $N$ training episodes is a permutation seeded by
   (seed, pass), so the order is a function of (seed, pass, step, rank) and a resumed run sees the
   same episodes.
9. **Validation**, at step 0, every `eval_every` steps, and at the end: the mean query CE over
   the validation episodes for three references: the zero adapter (the base; computed once, it
   never changes), the current H, and H fed $\bar z$ of an episode of another family (a fixed
   permutation). Overall, per source, and the paired differences current minus zero and
   mismatched minus current with their standard errors.
10. **Checkpoints** every `ckpt_every` steps, at the end, and at every stop: H, AdamW, the
    scheduler, the step, the data cursor, every process's RNG states, the zero-adapter CEs, and
    the metrics history, written to a temporary file, renamed, then pointed at by `latest.json`.
11. **Stops.** The script checks its kill criteria itself (section 7 of the brief): a non-finite
    loss or gradient norm (nothing applied), a step over `kill.max_step_s`, a projected run wall
    time over `kill.max_projected_s`: checkpoint, logs, exit 4. The session budget `max_wall_s`
    checkpoints and exits 3 before a step could cross it; `--resume` continues. `--max-wall-s`
    sets one session's budget on the command line (a finite number of seconds above 0, else exit
    2) without touching the config; it is not a resume setting, so the next session resumes under
    the config's budget or its own `--max-wall-s`.

### Task MT additions

12. **Source weights** (`source_weights`, runs `MT-02` to `MT-04`). Slots form blocks of 1000;
    block $b$ holds $c_s$ slots of source $s$ ($c_s$ the largest-remainder rounding of
    $1000\, w_s / \sum_t w_t$; 650, 150, 200 for BARC, ReARC, ARC-AGI-2 training) in a seeded
    shuffle, and source $s$'s $k$-th draw takes its own pass $\lfloor k / N_s \rfloor$'s seeded
    permutation at position $k \bmod N_s$. `run.json` `order` records the shares, the episodes per
    source, and the draws and passes per source over all steps.
13. **The best checkpoint.** At every validation, a new best point over every validation of
    every session (the first validation is one; under `zero_step` a new low of the current H's
    mean CE, under `fomaml` item 15's rule) rewrites `ckpt_dir/best.pt`: H's
    weights, the config, and the validation record, no optimizer state. Each validation record in
    `metrics.jsonl` carries `z` (both paired differences in standard errors), `best`, and
    `session_start` (the step its session started at); `status.json` `best` names the best point
    and the file when this session wrote it.

### Task P3.9 additions

14. **The `fomaml` objective** (`objective: fomaml` with an `inner` section; run `P3.9-01`,
    `configs/train_hypernet_P3.9-01.yaml`). Each episode takes $k$ = `inner.steps` inner steps
    on the demonstrations from H's adapter (a detached copy; Adam-like by default, fresh moments
    per episode, one drawn view per step), and the query CE at the adapted factors trains H
    first order. `--init-from` a `zero_step` full checkpoint with the same parameters (MT-03's
    `ckpt-004180.pt`) changes the objective and keeps the step, data cursor, AdamW state, and
    cosine; with `fresh_schedule: true` (the pilot's config) it starts a new phase instead: H's
    weights loaded, fresh AdamW moments, and a new warmup (50 steps) and cosine from `lr` (4e-5)
    over steps 4,181 to `steps`, the logged step continuing MT-03's. `--resume` refuses a change
    of the objective, the inner loop, or `fresh_schedule` (all `RESUME_KEYS`; absent reads as
    `zero_step`). Memory: 141 MB of inner state at $k = 2$ (the factor copy and Adam's moments). Training records carry `inner` (inner CEs,
    displacement and start norm per module type); timing adds `inner_step`, `h_backward`,
    `validate_adapted`. A step costs about $(1 + 0.716k)$ of a `zero_step` step (estimate).
15. **Decode-aligned validation** (every objective). Each validation record carries `decode` per
    reference (mean $|O_Q|$, mean $S$, share with $S < \ln 5$, greedy exact match, geometric-mean
    CE) and `per_episode_exact`; under `fomaml` with $k \ge 1$, `adapted` holds the current H
    after its $k$ inner steps. `best.pt` under `fomaml` is the highest post-$k$ exact match, ties
    by the lowest mean $S$. `summary.md` adds a decode-aligned table. Module doc:
    [`docs/modules/train.md`](../modules/train.md), section on P3.9.
16. **The inner clip** (`inner.clip`, 1.0 in `P3.9-01`). Each inner gradient is clipped by its
    global norm over the episode's factors, as the notebook's fit clips its own
    (`max_grad_norm` 1.0): $g_j \leftarrow \min(1, \tau / (\lVert g_j\rVert_2 + 10^{-6}))\,g_j$.
    The training records' `inner.grad_norm` and the validations' `adapted.grad_norm` give each
    inner step's norm before the clip, so the reader sees whether the clip acts.
17. **The inner-rate probe** (`inner_probe`, run `P3.9-01`). At the start step of a
    `--init-from` run, validation scores H after its $k$ inner steps at every rate of
    `inner_probe.lrs` (4e-4, 1.6e-3, 6.4e-3) and the run trains at
    $\alpha^* = \arg\min_{\alpha \in C} \bar S(\alpha)$, $C$ the rates within `margin` (2)
    episodes of the most post-$k$ greedy exact matches (ties to the smaller rate). The record's
    `probe` holds every rate's block; `summary.md` adds a probe table; `run.json` and the
    checkpoints record the pick in `init_from.inner_probe`, and the run checkpoints its start step
    right after, so a restart resumes at the pick. Each probed rate costs one post-$k$
    validation (about 188 s on 4 x L4, estimate).
18. **Starts and phases.** `fresh_schedule` and `inner_probe` without `--init-from` (or a
    `--resume` of such a run) exit 2 before anything is read or written. `--init-from` a P3.9
    checkpoint with `fresh_schedule: true` starts another phase from it (a longer run), its
    `init_from.previous` keeping the first phase's record. `best.pt`'s `config` leaves the
    objective keys out (they move to `best.pt`'s `objective`), so P3.7's load check, which parses
    the config strictly on its branch's code, loads a `fomaml` `best.pt` as it loads `MT-03`'s.

## 3. Why it is built this way

- **Task loss end to end, not weight-space targets** (Q6, ADR-006). Matching per-puzzle LoRA
  factors is ill-posed ($BA = (BQ)(Q^{-1}A)$), and fitting one target costs as much as several
  hundred steps here; T2L's end-to-end variant generalized better than its reconstruction
  variant. Alternative: D2L's top-16 KL to a context-in teacher. Lost: the in-context model is the
  floor here, since TTT beats it (BRIEF B1 item 1).
- **Demonstrations stay in the query's context** (ADR-006): the adapter is a delta on in-context
  behavior, as the notebook's TTT adapter is. Alternative: D2L's adapter-only query. Lost: test
  time decodes with the demonstrations in context, so training without them would shift the
  distribution H is trained for.
- **The loss covers the query's output: its grid and the closing `<|im_end|>`.** The notebook's TTT
  trains every output grid of the sequence with its closing token (the completion-only collator);
  Q6 scores the left-out query given the demonstrations, which is what H must make likely, so the
  demonstrations' outputs are context, not targets. The closing token is read as part of Q6's
  "output-grid tokens" (`loss_includes_eos`, on): H's adapter serves the notebook's decode, where
  that token ends a grid and so fixes its last row and its shape. Alternative: the grid's digits
  and newlines alone. Lost: the adapter would never be trained on where a grid ends.
- **$\bar z$ from un-augmented demonstrations** (task brief). One adapter serves all 16 decode
  views at test time, as the notebook's TTT adapter does; training on $a(Q)$ under random $a$ with
  one $\bar z$ per demonstration set teaches exactly that. Per-pair embeddings are then cacheable
  by pair hash, so a long run pays the encoder once per distinct pair (episode note, open risk 3).
  Alternative: $\bar z$ of the augmented demonstrations. Lost: H's input would depend on the view,
  and the cache would need 16 times the entries.
- **Embeddings encoded before step 0, sharded** (deviation 1 in the module doc). Lazily encoding
  per step on each process would encode every pair four times and put encoder spikes into step
  times, which the step-time kill criterion reads. Sharding by fixed chunks of sorted pair hashes
  makes each pair's prefill batch, so its embedding, independent of the process count and of a
  resume.
- **One episode per process per micro step** (B5). D2L packs 4096-token sequences; episodes here
  reach 8192 tokens, and packing distinct adapters needs D2L's per-segment factor path, which P3.1
  did not build (`inject` installs one adapter per forward). Alternative: batches of episodes.
  Lost: the same, plus padding waste on a 6-to-1 length spread (toy p50 968, max 6252).
- **`accum` 1 on 4 GPUs for the toy run, 8 for the long run** (task brief; B5; D2L uses 8). The
  toy run reads B5's "2k steps" as 2000 optimizer steps of 4 episodes: 8000 draws, four passes over
  the 2000 toy episodes.
- **`P3.5-02`: half the steps, nothing else** (`P3.5-01`'s brief, Borderline row;
  `docs/experiments/P3.5-01.md`). `P3.5-01`'s gain over the zero adapter peaked at step 800, in its
  second pass, and from step 1400 H was worse than the zero adapter while the training CE kept
  falling: H fitting the repeated 2000 episodes. 1000 steps make two passes. The warmup stays 100
  steps, so steps 1 to 100 repeat `P3.5-01`'s learning rates on the same episodes in the same order
  (the order is a function of seed, pass, step, and rank), and the cosine ends at `min_lr` at step
  1000; the budget and the projected-wall kill shrink with the run. Alternatives: half the learning
  rate, the other pre-registered remedy for a faded gain. Lost: four passes over the same 200
  puzzles would remain, and the fit and the memorization would both slow. Truncating `P3.5-01`'s
  2000-step schedule at step 1000. Lost: that run's step 1000 is already measured (current minus
  zero $-0.0004$, SE 0.0024). Raising BARC's share, the remedy for a generic adapter. Lost: C held
  at step 600 and faded with B, so the fade comes first. 8 views (`P3.2-02`). Lost: a second
  change; the long run takes the view set `P3.2-02` decides.
- **DistributedDataParallel on H only.** H with AdamW is 135M fp32 parameters, about 2.7 GB with
  gradients, both moments, and DDP's buckets; it fits each L4 beside the 7.3 GB base, so sharding
  (FSDP, ZeRO) buys nothing. The base is frozen and replicated, so only H's gradients are
  all-reduced (NCCL). Processes come from `torch.multiprocessing.spawn`, as in the reference's
  starter, so the script stays one file (P0.5d script mode).
- **Non-reentrant activation checkpointing** (D2L note, risks 2 and 3). Without it, saved
  activations exceed an L4's memory at long episodes (over 40 GB at 9.5k tokens, estimate in the
  note). The reentrant variant (transformers' default)
  detaches inputs that do not require grad and nests backward passes through H's graph. Backward
  runs inside `inject`, so the recompute sees the same factors; `tests/test_train.py` checks
  that gradients with and without checkpointing are bitwise equal.
- **Precision.** The base in bf16 (the Kaggle variant; T4 has none, so the accelerator is
  `NvidiaL4`), H's parameters and the injected delta in fp32 (`inject`'s default, PEFT's on a bf16
  base), the loss in fp32. Alternative: H in bf16. Lost: 135M parameters cost little in fp32, and
  AdamW's small updates to zero-initialized heads would round away in bf16.
- **Learning rate 4e-5, 100 warmup steps, minimum 1e-7, clip 1.0, decay 0.01** (D2L note 1.4).
  D2L is the closest scaffold (a hypernetwork into a frozen bf16 base, fp32 hypernetwork). T2L's
  2.5e-5 with 20% warmup and linear decay is the alternative; neither is tuned here (task brief).
  The first step uses $\eta / n_w$, not 0, so no step is wasted.
- **The cap is asserted, not applied** (task brief). P3.4 dropped the 17 longer episodes; cutting
  demonstrations, as the notebook's `cut_to_len` does, would change an episode silently.
- **The zero adapter once.** The base is frozen, so its CE is a constant; recomputing it at every
  validation would cost a third of each validation.
- **The mismatched control takes another family.** Episodes of one ReARC family share a
  transformation, and the episodes of one puzzle share their demonstrations, so another puzzle of
  the same family would not be a mismatch.
- **Validation every 200 steps on all 286 episodes.** Eleven points on the curve for about 12% of
  the run (estimate: two references of about 40 s per point); a subsample would add sampling
  noise to a comparison that is already small.
- **Checkpoints every 200 steps, atomic, two kept.** About 11 minutes of training (estimate) is the
  most a lost session costs; each checkpoint is about 1.6 GB, two fit easily in `/kaggle/working`.
- **Kill criteria in the script** (A9): nothing can cancel a running kernel, so the script stops
  itself, after a checkpoint and the logs, and Kaggle's `Timeout` is only the outer bound.

- **Source weights, not a new episode build** (task MT, orchestrator's decision). `P3.5-02`'s
  report pre-registered raising BARC's share; the pool's share was 49.8%. Weights in the order cost
  no Kaggle runs and keep the build, its leakage check, and `MT-01`'s cache. Lost: a new P3.4
  build with BARC raised to all its usable problems, about twice the distinct BARC puzzles, which
  grows `MT-01` by about 60% of its encode quota and delays the launch; it is the fallback if
  check C still fails at full scale.
- **Shares: BARC 65%, ReARC 15%, ARC-AGI-2 training 20% (13 : 3 : 4).** At `P3.5-02`'s step
  1000, H beat the zero adapter on BARC (0.0365 against 0.0429, $-15\%$) and lost on ReARC
  (0.0174 against 0.0160, $+9\%$), where the base already fits (zero CE 2.7 times below BARC's;
  NVARC's SFT used regenerated ReARC). ReARC's share halves, ARC-AGI-2 training keeps its 20%
  (human-written, nearest the evaluation, with no validation measure to move it on), BARC takes
  the rest. Over 5,600 steps: BARC 116,485 draws (2.34 passes over its 49,680 episodes), ReARC
  26,884 (0.90 over 29,997), ARC-AGI-2 training 35,831 (1.79 over 19,997).
- **Stratified blocks, not a draw per slot.** Every aligned block of 1000 slots (31.25 steps)
  holds the rounded shares exactly; a window that straddles two blocks can differ by a few points
  of share (measured on the MT pool and weights over its 179,200 slots: up to 3.9 points over 992
  slots, 1.0 over 3,200, 0.1 over 32,000), and the order stays a pure function of seed and slot
  at the cost of one shuffle per block. A categorical draw per slot would hold the shares only in
  expectation and would need every earlier slot to know a source's draw count. Unweighted (the
  default), the order is P3.5's, bitwise.
- **A weights-only best checkpoint beside two full ones.** H's best validation point is the
  checkpoint P3.7 needs (`P3.5-02`: H from its best point, not its last step). A full checkpoint
  is about 1.6 GB (H and AdamW's two moments); keeping one per validation point would outgrow
  Kaggle's about 20 GB of working output in a session. `best.pt` is H alone, about 0.54 GB, so
  the folder holds about 3.8 GB. Lost: `keep_checkpoints` at every validation point (disk);
  checkpoints every 100 steps with the best one copied out by hand (not possible on Kaggle);
  picking the best point after the run (its H would be gone).
- **5,600 steps, chosen for two sessions, run in three.** Two `max_wall_s` budgets (82,800 s)
  hold, after setup (about 600 s and 480 s), 29 validations, and 56 checkpoints, about 5,950 steps
  at `P3.6-01`'s 13.42 s; 5,600 leaves 4.6%. These weights' shorter mix projects 11.89 s per step
  (`P3.6-01`'s model, estimate). Lost: 6,230 steps (two passes of the pool's size), which need
  83,600 s of steps alone. On 4 Oct the week's quota had 7.87 h left, lost at the reset on 10
  Oct, with 4xL4's rate (once or twice the T4's) still open, so `MT-02` (revision 2) runs a capped
  first session (12,600 s, step 1,028 measured), `MT-03` one of 38,700 s (to about step 4,170 to
  4,240), and `MT-04` the rest (about 1,360 to 1,430 steps): the same steps, one more setup and
  notebook (about 0.2 h).
- **A per-session cap on the command line** (`--max-wall-s`, amendment of 4 Oct). The capped
  session and the sessions that resume it must share every `RESUME_KEYS` value, and the code
  dataset must hold the bytes each notebook checks. A command-line budget leaves the config and
  the wheel byte-identical and costs one script change (so one code dataset version, and a
  rebuild of the notebooks that record the script's sha256). Alternatives: a second config file
  with only `max_wall_s` changed (a 250-line copy to keep in step, another file in the code
  dataset, and a build whose config differs from the resuming sessions'); the notebook's hard
  stop alone (`script_kill_s`), which kills the script with `SIGKILL`: up to 100 steps (about 20
  minutes) lost and no final `status.json` or summary. Lost: nothing the run reads; the budget is
  recorded in `config.yaml` and `run.json` `argv`.
- **A 45-minute pre-script allowance in a full session** (`MT-03` revision 3, amendment of 17
  Oct). Kaggle has spent up to about 41 minutes between RUNNING and a notebook's first cell
  (`P6.5-02` 36.7, `P6.3-02` revision 5 41, `P6.3-04` about 10, `P6.5-03` 0.6; task P6.5b,
  `docs/prerun/P6.5-03.md` on `task/P6.5b`), and whether the 12 h limit and the `Timeout` count
  it is unknown. A full session then takes $a = 2{,}700$ s off both bounds that run on the
  notebook's clock: `--max-wall-s` $41{,}400 - a = 38{,}700$ and `script_kill_s` $42{,}300 - a =
  39{,}600$, so the margin between them is unchanged and the session ends by $a + 39{,}600 + 300 =
  42{,}600$ s from RUNNING, the `Timeout`, which stays 42,600 s because Kaggle's own timer may
  count from RUNNING. The notebook logs the clocks that could show the gap
  (`stages.environment.clocks`: `/proc/uptime`, the ages of PID 1, of the kernel, and of the
  working folder), as P6.5b's builds do. Alternatives: no allowance (revision 2: a 41-minute gap
  would let Kaggle end the session 873 to 1,473 s before the script's own stop, losing up to 100
  steps and the final `status.json`); `max_wall_s` lowered in `train_hypernet_MT.yaml` (a new
  config file, so a new code dataset version, and a config that differs from `MT-02`'s bytes,
  which the notebook's sha256 check and the shared code dataset forbid); a lower `Timeout` alone
  (if Kaggle counts from RUNNING it would kill the script before its stop; if not, it changes
  nothing); waiting to measure whether the gap counts (no measurement can settle it before the
  run, and `P6.5-03` showed none). Lost: about 222 to 227 steps of this session (2,700 s), which
  `MT-04` runs, about 0.75 h more of 4xL4 if the gap does not count.
  **Merge note:** the clock reader is `session_clocks` in `scripts/build_kaggle_notebook.py`
  (defined before `RUNTIME_HELPERS`, listed in `SCRIPT_RUNTIME_HELPERS`), the same function as
  `task/P6.5b`'s, which defines it elsewhere in that file for the shot build's `SHOT_EDITS`. A
  merge of both branches keeps both definitions without a conflict, and ruff's F811 may not flag
  it (the first definition is used, in `SCRIPT_RUNTIME_HELPERS`, before the second rebinds the
  name); keep one definition by hand at that merge.
- **Validation every 200 steps** (29 points, 2,235 s, 3.4% of the steps' time): the best point
  within 100 steps and 29 readings of check C; every 100 steps doubles the cost, every 400 could
  miss a peak as narrow as the toy runs'.
- **Kill criteria from `P3.6-01`.** `max_step_s` 150 s, twice the slowest legitimate step (all 8
  micro steps of one GPU at 8192 tokens, about 72 s); `max_projected_s` 82,000 s, what two
  sessions hold, so a run that cannot finish in two sessions stops after 100 steps instead of
  costing a third.
- **The inner clip at the fit's 1.0, not none** (`P3.9-01`). Alternatives: no clip (the first
  version, read as "Adam's step does not depend on the gradient's scale"): true for the first
  step only; from the second, $m_1 = \beta_1(1-\beta_1)c_0g_0 + (1-\beta_1)c_1g_1$ weighs the two
  gradients by their clip factors, so an unclipped inner loop trains H for a step the fit
  (`arc26/solver.py`, `max_grad_norm` 1.0) never takes whenever one gradient's norm is above 1.
  Correcting the wording instead would have kept that mismatch. Cost: one norm over 11.8 M
  values per inner step.
- **The inner rate picked by a probe at the start step, not fixed** (`P3.9-01`). Alternatives:
  the survey's 1.6e-3 alone (its derivation, the fit's per-coordinate displacement over 128
  steps split over two, is an upper bound, and Adam's first step moves every value by the full
  rate whatever the gradient's scale, so a wrong rate trains H for an inner loop that overshoots
  or barely moves); a separate validation-only job per rate (a second container start, setup,
  and approval: about 0.3 USD and a day more); a learned per-layer rate (Meta-SGD; more
  parameters and a second-order signal FOMAML does not give). The probe costs about 376 s
  (0.41 USD) for the two rates beyond the pick and runs once, on the episodes the start-to-end
  comparison uses, which biases that comparison against the end (the start's scores are the
  best of three). The rule is the `best.pt` rule (post-$k$ exact match, then mean $S$) with a
  noise margin of 2 episodes, the project's two-task noise floor, so the rate is chosen on the
  decode-aligned metric P3.9 selects on, not on mean CE (`P3.7-01`: CE does not track solving).
- **No middle validation in `P3.9-01`** (`eval_every` 1,000): the decision is paired start
  against end at the pick; a middle one costs about 265 s (0.29 USD) of the human's credit and
  falls inside the warmup.
- **One config, `run_id` `MT-02`, for all three sessions.** A resume must match every
  `RESUME_KEYS` setting, and one code dataset version must serve every notebook; `summary.md` then
  points at `MT-02`'s brief, whose section 1 holds every session's rows.

## 4. How to run it

**CPU smoke (local, required before any GPU run; BRIEF A0: CPU only).** From the repository root
(for `P3.9-01`'s config, see the end of this block):

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_toy.yaml --smoke
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_toy.yaml --smoke \
    --resume runs/P3.5-smoke/checkpoints/ckpt-000008.pt
```

The first runs the whole loop in 2 processes on gloo: the tiny random Qwen3 (hidden 64, 4 layers)
and the stand-in tokenizer built in code, the fixture episodes of `tests/testdata/train_hypernet/`
(12 toy, 6 validation), 24 steps (four passes), validation at 0, 8, 16, 24. Measured at
`805dda5`: 7.8 s wall, 0.87 GB peak resident memory, exit 0, training CE per pass 3.20, 2.34,
1.79, 1.64. The second continues from step 8 (6.5 s) and ends with the same H, bitwise.
`--config configs/train_hypernet.yaml --smoke` runs accumulation 2 with DDP. For `P3.5-02` run the
same two commands with `--config configs/train_hypernet_toy_P3.5-02.yaml` and
`--resume runs/P3.5-02-smoke/checkpoints/ckpt-000004.pt`: 12 steps (two passes), validation at 0,
4, 8, 12. Measured (brief `P3.5-02`, section 11): about 7 s wall, 0.87 GB peak resident memory,
exit 0, training CE per pass 3.19 and 2.22; the rerun from step 4 (about 7 s) crosses the pass
boundary and ends with the same H, bitwise; steps 1 to 3 repeat `P3.5-01`'s smoke bitwise (same
episodes, same warmup). Tests: `CUDA_VISIBLE_DEVICES="" .venv/bin/pytest tests/test_train.py` (51
tests, about 39 s).

**`P3.9-01`'s config** warm-starts, so its smoke needs the MT smoke's step-8 checkpoint first
(`fresh_schedule` and `inner_probe` without `--init-from` exit 2, nothing written):

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_MT.yaml --smoke
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_P3.9-01.yaml --smoke \
    --init-from runs/MT-smoke/checkpoints/ckpt-000008.pt
```

Measured on the dev box (18 Oct): the bare P3.9 smoke exits 2 in 2.2 s; the MT smoke 6.0 s; the
P3.9 smoke 7.1 s, exit 0: the probe at 2.5e-3, 1e-2, 4e-2 picks 2.5e-3 on the random base (mean
$S$ 28.9 against 38.0 and 52.8, no exact match anywhere), steps 9 to 12 at the pick with the
phase's warmup, validations at 8 and 12, 0.42 GB peak resident (the parent process). The whole
Modal path of the pilot runs on the CPU with
`python scripts/modal_ops.py --smoke --config configs/modal_P3.9-01.yaml` (14 checks, the probe's
included; modal_ops.md). Tests: `tests/test_train_fomaml.py` (29 tests, about
90 s).

**Modal (`P3.9-01`), after the human approves `docs/prerun/P3.9-01.md`:** build
`dist/P3.9-01/` with `scripts/build_modal_job.py --config configs/modal_P3.9-01.yaml` and launch
it with `scripts/modal_ops.py run dist/P3.9-01 --run-id P3.9-01`
(modal_ops.md, section 4). The job fetches MT-03's `ckpt-004180.pt` and passes
`--init-from` to this script; nothing is attached.

**Kaggle (`P3.5-01`), after the human approves the brief.** The script runs in a script notebook
(P0.5d, `docs/runbooks/build_submission.md`, script mode) built from
`configs/script_P3.5-01.yaml`, on `NvidiaL4`, with the model, the code dataset (wheel, this script,
`train_hypernet_toy.yaml`), the install notebook at version 2, and the two P3.4 outputs it pins in
`notebook_inputs`: the builder output `alyahya737/arc26-p3-4-episodes` version 2 and the checker
output `alyahya737/arc26-p3-4-leakage` version 2, which checked build 2. The notebook finds each
mount as the folder named by its slug and passes it as `--episodes-root` and `--check-root`
(`{input:<owner>/<slug>}` in `args`), with `--commit` (`{commit}`); the hard stop is
`script_kill_s` 15600 s. The code dataset needs a version from this build (new wheel and script):

```bash
.venv/bin/python scripts/build_submission.py --tag P3.5-01 --config configs/script_P3.5-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P3.5-01/arc26-code -m "P3.5-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P3.5-01 --run-id P3.5-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p3-5-train
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p3-5-train --task-id P3.5-01
```

The notebook runs `python train_hypernet.py --config train_hypernet_toy.yaml --commit <sha>
--episodes-root <episodes mount> --check-root <checker mount>` in `/kaggle/working`. If
`NvidiaL4` waits in the queue for 30 minutes, the orchestrator tells the human; there is no
smaller fallback (`NvidiaL4X1` runs on 2xT4, which has no bf16).

**Kaggle (`P3.5-02`), after the human approves its brief.** The same notebook
`alyahya737/arc26-p3-5-train` (its next version), built from `configs/script_P3.5-02.yaml`:
`P3.5-01`'s inputs and arguments, the config `train_hypernet_toy_P3.5-02.yaml`, the hard stop
`script_kill_s` 8400 s (`Timeout` 8700 s). Its wheel is `main`'s, which `P3.2-02`'s build also
carries, so one code dataset version serves both runs: a folder holding both builds'
`arc26-code/` files (the same wheel; both scripts and configs, which each notebook finds by name
beside the wheel and checks by sha256; either build's `dataset-metadata.json`, the two differing
only in the description that names the script), made once the run before it is past its wheel
check and never while a run waits in the queue; push about 5 minutes later:

```bash
.venv/bin/python scripts/build_submission.py --tag P3.5-02 --config configs/script_P3.5-02.yaml
# <shared>/arc26-code: dist/P3.5-02/arc26-code/ plus dist/P3.2-02/arc26-code/ (task/P3.2-01)
.venv/bin/python scripts/kaggle_ops.py dataset-version <shared>/arc26-code \
    -m "P3.2-02 <commit>, P3.5-02 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P3.5-02 --run-id P3.5-02
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p3-5-train
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p3-5-train --task-id P3.5-02
```

**Plain GPU machine** (bf16 GPUs, CUDA, four of them for the toy config's `gpus: 4`):

```bash
uv venv --python 3.11 .venv && uv pip install -e ".[dev,hyper]"
.venv/bin/python scripts/train_hypernet.py --config configs/train_hypernet_toy.yaml \
    --episodes-root /data/arc26-p3-4-episodes --check-root /data/arc26-p3-4-leakage \
    --model-dir /data/qwen3_4b_grids15_sft139 --out-dir runs/P3.5-01/run_logs \
    --ckpt-dir /data/checkpoints/P3.5-01
```

FlashAttention 2 is used when `flash_attn` imports (Kaggle's install notebook ships 2.8.2),
SDPA otherwise. With fewer GPUs, copy the config and set `gpus`; the data order then differs
from the brief's run.

**Expected cost of `P3.5-01`** (estimates, brief section 6): about 2.2 h of wall time on 4xL4
(model load 2 min, embeddings 6 min, 2000 steps of about 3.2 s, eleven validations of about 40 s
plus the step-0 references), about 4.4 GPU quota hours; about 12 GB of each L4's 22.5 GB; disk
about 3.3 GB of checkpoints and a few MB of logs. Measured (`docs/experiments/P3.5-01.md`): a
6446 s session, the script 6388 s (2000 steps of 2.30 s on average, encoding 543 s, validations
107 s at step 0 and 76 s after), about 3.6 quota hours.

**Expected cost of `P3.5-02`** (brief section 6, from `P3.5-01`'s measured times): `P3.5-01`'s
script less what this run leaves out (steps 1001 to 2000, 2314 s; five validations, 382 s; five
checkpoints of about 10 s) is about 3640 s; with the notebook's check, install, and logs, about
3700 s, 1.0 h of session on 4xL4, about 2.1 GPU quota hours at twice the T4 rate. Its steps 1 to
1000 draw `P3.5-01`'s episodes in the same order, so their measured 2276 s carries over.

**CPU smoke of the MT config** (2 gloo processes, accumulation 2, the source weights on the
fixture's toy split, `best.pt`):

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_MT.yaml --smoke
```

Measured at the brief's `Code-commit`: 7.2 s wall, 0.89 GB peak resident memory, exit 0, 12
steps (48 draws: BARC 31, ARC-AGI-2 training 9, ReARC 8), `best.pt` written. Tests:
`tests/test_train_mt.py` and `tests/test_notebook_local_mt.py`.

The session cap of `MT-02` revision 2, its stop, and a resume without it:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_MT.yaml --smoke --max-wall-s 12600
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_MT.yaml --smoke --max-wall-s 1e-6
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/train_hypernet.py \
    --config configs/train_hypernet_MT.yaml --smoke --resume runs/MT-smoke/checkpoints
```

The first runs the 12 steps under the cap (`config.yaml` `max_wall_s` 12600.0); the second stops
before its first step (a budget below the setup), exit 3, with `ckpt-000000.pt`; the third resumes
it under the smoke section's 600 s and ends complete at step 12 (MT-02's brief, section 11, has
the measured times).

**Kaggle (`MT-02`, revision 2), at once after the human approves the brief (both batch slots
free).** The build of `configs/script_MT-02.yaml`: notebook `alyahya737/arc26-mt-train-1` on
`NvidiaL4`, P3.4's two outputs at version 2, `MT-01`'s output `alyahya737/arc26-mt-cache` at
version 1, and the code dataset `alyahya737/arc26-code-mt` at version 2: version 1's merged folder
`dist/MT-code/` (`docs/runbooks/cache_embeddings.md`, section 4: `MT-01`'s `arc26-code/` files
plus `MT-02`'s `train_hypernet.py` and `train_hypernet_MT.yaml`) with this build's
`train_hypernet.py`, which reads `--max-wall-s`; the wheel and the config are version 1's bytes.
The notebook runs `python train_hypernet.py --config train_hypernet_MT.yaml --commit <sha>
--episodes-root <episodes mount> --check-root <checker mount> --embedding-cache <MT-01
mount>/embedding_cache --max-wall-s 12600`, with the hard stop at 13,300 s and `Timeout` 13,600 s.
The orchestrator, with a clean tree at the brief's `Code-commit`:

```bash
.venv/bin/python scripts/build_submission.py --tag MT-02 --config configs/script_MT-02.yaml
cp dist/MT-02/arc26-code/train_hypernet.py dist/MT-code/
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/MT-code -m "MT-02 revision 2 <commit>"
# about 5 minutes for Kaggle to process the version
.venv/bin/python scripts/kaggle_ops.py push dist/MT-02 --run-id MT-02
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-mt-train-1
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-mt-train-1 --task-id MT-02
```

`dist/MT-code/` then holds exactly the five files of version 1 other than the script, unchanged,
and the new `train_hypernet.py` (the brief lists every sha256). A notebook that mounted version 1
stops in its environment check (the script's sha256 differs from the build's), before the
install. The session stops itself at `--max-wall-s` 12,600 s with a checkpoint near step 900
(exit 3, expected, the brief's Proceed row); the quota page before and after it settles whether
4xL4 costs once or twice the T4 rate.

**Kaggle (`MT-03`, revision 3), `MT-02` having read Proceed, in the week from the reset of Sat 17
Oct, after its brief's approval.** The build of `configs/script_MT-03.yaml`: `MT-02`'s with its
`--max-wall-s` at 38,700 s (the config's 41,400 s less the 45-minute pre-script allowance above)
and its hard stop at 39,600 s, with `alyahya737/arc26-mt-train-1` version 1 attached and `--resume
{input:alyahya737/arc26-mt-train-1}/checkpoints`, in notebook `alyahya737/arc26-mt-train-2`, on
the same code dataset version (2):

```bash
.venv/bin/python scripts/build_submission.py --tag MT-03 --config configs/script_MT-03.yaml
.venv/bin/python scripts/kaggle_ops.py push dist/MT-03 --run-id MT-03
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-mt-train-2
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-mt-train-2 --task-id MT-03
```

It stops at its budget (exit 3) near step 4,173 to 4,239 (`MT-02`'s measured 11.90 to 12.15 s
all-in step). Its report reads Kill or failure against Continue (the Continue row is already met
by `MT-02`'s step 1,000, so Generic and No gain cannot occur), gives B and C at its validations as
informative reads and the best point so far, and reads `stages.environment.clocks` against the
RUNNING time the orchestrator records.

**Kaggle (`MT-04`), unless `MT-03` ends in Kill or failure, when the quota page covers it, and
after its brief's approval.** The build of `configs/script_MT-04.yaml` (revision 1; its next
revision takes `MT-03`'s pre-script allowance and pins the version of `alyahya737/arc26-mt-train-2`
that `MT-03`'s GPU session wrote): `MT-03`'s output attached instead of `MT-02`'s and `--resume
{input:alyahya737/arc26-mt-train-2}/checkpoints`, in notebook `alyahya737/arc26-mt-train-3`, on
the same code dataset version:

```bash
.venv/bin/python scripts/build_submission.py --tag MT-04 --config configs/script_MT-04.yaml
.venv/bin/python scripts/kaggle_ops.py push dist/MT-04 --run-id MT-04
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-mt-train-3
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-mt-train-3 --task-id MT-04
```

It ends at step 5,600 (`complete`, exit 0).

**Expected cost** (briefs, section 6): `MT-02` measured about 3.57 h of 4xL4 (7.1 quota h at
twice the T4 rate, 3.6 at the T4 rate); `MT-03` (revision 3, from `MT-02`'s measured step) about
10.8 h from its first cell (21.6 and 10.8; 23.1 and 11.6 if Kaggle counts 45 minutes before it;
at most 23.7 and 11.8 if its `Timeout` counts from RUNNING, 24.0 and 12.0 if it counts from the
first cell); `MT-04` the rest of the 5,600 steps, 4.7 to 5.0 h (9.4
to 10.0, and 4.7 to 5.0). About 12.7 GB of each L4 at the
8192 cap (`P3.6-01`); about 3.8 GB of checkpoints per session.

**Resume on Kaggle** after a stop (exit 3) or a lost session: see "Resuming across sessions" below.

## 5. Outputs

| File (in `out_dir`, pulled) | Contents |
|---|---|
| `config.yaml` | The resolved config. |
| `run.json` | Run id, argv, commit (from `--commit` on Kaggle), versions; world size, GPU name, attention kernel; model facts (load seconds, layers, hidden size, vocabulary, dtype); the build folder, manifest sha256, and check report path; split sizes; per validation episode its source, family, tokens, target tokens, and mismatch partner; H's parameter count and values per adapter; embedding counts and seconds; the start step and the checkpoint resumed from. |
| `status.json` | `complete`, `stopped`, `killed`, or `failed`, the exit code, the reason, the step, the last checkpoint, the wall time over all sessions. |
| `timing-rank<k>.jsonl` | One `{puzzle_id, op, seconds, tokens}` per GPU op of process $k$: `encode` (one chunk of the embedding pass; `puzzle_id` null, prefill tokens), `h_forward` (H for one episode), `train_step` (one episode's forward and backward; its tokens), `validate` (one episode under one reference). |
| `metrics.jsonl` | Per optimizer step (`kind: train`): step, lr, mean query CE and per source, penalty, gradient norm (before clipping), tokens, step seconds, episode ids and their CEs in slot order. Per validation (`kind: val`): step, CE per reference, per source, the paired differences with standard errors, seconds, and the per-episode CEs. |
| `summary.md` | Status, data, model, H; the training CE per pass over the training split and the change from the first to the last complete pass, paired by episode (the first and last 5% of the steps when there are fewer than two passes); the validation table, per-source CEs at the last validation, the step-0 check, and the decision-rule pointer. |
| `loss_curve.csv`, `loss_curve.png` | The training CE per step with its moving mean, and the three validation curves; the CSV holds the data (the PNG needs matplotlib, which the Kaggle image has). |
| `error.log` | The traceback of a failure. |

Checkpoints `ckpt_dir/ckpt-<step>.pt` and `latest.json` stay on Kaggle (`.pt` is outside the pull
allowlist and `.gitignore`d). So does `ckpt_dir/best.pt` (task MT): H's weights at the best
validation point over all sessions (the lowest current-H CE under `zero_step`, the most post-$k$
exact matches, then the lowest mean $S$, under `fomaml`), with the config (without the objective
keys, which `objective` holds) and the validation record; `load_checkpoint` reads it
(kind `best`), as P4.1's provider does, and a resume refuses it. In `MT-02` to `MT-04`,
`metrics.jsonl`'s validation records carry `z`, `best`, and `session_start`; `status.json` adds
`best` (step, CEs, differences, `z`, per source, the session it came from, and the file when this
session wrote it); `run.json` adds `order` (shares, episodes, draws, and passes per source).

## 6. How to read the results

The pre-registered rule is the brief's section 1, read off `summary.md`:

- **A, the training CE falls:** the toy run makes four passes of 500 steps, each showing every
  toy episode once; pass 4's mean CE must be below pass 1's by more than two standard errors of
  the paired differences over the 2000 episodes.
- **B, validation beats the zero adapter:** at step 2000, the paired difference current minus
  zero over the 286 validation episodes is below minus two standard errors. **B'**: the same at
  some earlier validation (a gain that later faded).
- **C, the gain depends on $\bar z$:** at step 2000, mismatched minus current is above two
  standard errors.
- **`P3.5-02`** reads the same criteria at its length: A compares pass 2 (steps 501 to 1000) with
  pass 1 (steps 1 to 500), the summary's "Pass 2 minus pass 1" line; B and C at step 1000; B' at
  steps 200 to 800. Its brief's outcome table says what follows each row.

| Result | Rule | Next |
|---|---|---|
| Good | A, B, and C | P3.6 on this code; then the long run's brief |
| Borderline | A and (B or B'), not Good | one more toy run with one change chosen from the curves |
| Bad | not A, or neither B nor B', or a kill or an error | diagnose on the CPU before any GPU run (section 7) |

A Good with the final CE above its best validation's, or a Borderline from B' alone, points at
memorization (below): the long run's brief then weighs fewer passes or more distinct puzzles.

**`MT-02`, `MT-03`, and `MT-04`** read `MT-02`'s brief, section 1 (revision 2), off
`metrics.jsonl` and `status.json`: at each validation $s$, B($s$) is `z.current_minus_zero` below
$-2$ and C($s$) is `z.mismatched_minus_current` at least 2. `MT-02`, capped, falls in **Proceed**
when it stopped at its cap (`status.json` `stopped`, exit 3, the budget's reason) with
`latest.json` naming the checkpoint of its last step; its B and C are early reads. `MT-03` falls
in **Continue** when B and C hold at one validation point of either session (then `MT-04` may
launch), **Generic** when B holds somewhere but never with C, and **No gain** when B never holds;
`MT-04`'s final reading names the best point over all three sessions (`status.json` `best`) and
reads B and C there. A best point well before the end, with the training CE still falling, is the
memorization pattern below; its `best.pt` is the H that P3.7 evaluates either way, from the output
of the session that wrote it (`best.session_start`).

**`P3.9-01`** reads its brief's section 1 off the start and end validation records
(`metrics.jsonl`, steps 4,180 and 4,260): the probe's pick (`probe.chosen`), then, paired per
validation episode at the pick, the end's post-$k$ exact matches gained and lost
(`adapted.per_episode_exact`), the paired mean of $S_\text{end}(i) - S_\text{start}(i)$ with
its standard error ($S = |O_Q|\,\bar\ell$ from `adapted.per_episode` and `run.json`
`target_tokens`), the count under $\ln 5$, and the CE. The 0-step metrics of the same records
are reported beside them and decide nothing: a start trained for the inner steps may lose some
0-step fit.

## 7. Failure modes

- **Out of memory.** First: check `run.json`'s attention kernel; SDPA instead of FlashAttention 2
  costs memory at long sequences. Then lower `max_tokens` only through a new P3.4 build (episodes
  are never cut here), or move to P3.6's measurements before retrying.
- **Divergence** (training CE rising, gradient norm growing): halve `lr` and double
  `warmup_steps` in a new revision of the brief.
- **Non-finite loss or gradient (exit 4):** the update was not applied and the checkpoint holds
  the last good state; look at the step's episode ids in `metrics.jsonl` and their tokens, then
  lower `lr`.
- **Step time over `kill.max_step_s` or projection over `kill.max_projected_s` (exit 4):**
  compare `timing-rank<k>.jsonl` `train_step` seconds per token with P1.1-01's cost model
  (brief, section 6); a GPU without FlashAttention 2 or a slow process shows there first.
- **Session budget (exit 3) or Kaggle's `Timeout`:** resume (below).
- **`--max-wall-s` refused (exit 2, argparse):** it takes a finite number of seconds above 0;
  `inf`, `nan`, 0, and negative values are refused, so the notebook's hard stop never becomes the
  only stop.
- **Refused episode set (exit 1, `EpisodeSetError`):** the attached build or check report is not
  build 2's; attach the right versions, never edit `manifest_sha256` without a new check.
- **Tokenizer mismatch (exit 1 in `build_sequence` or `render_pair`):** the real tokenizer splits
  the format differently from the notebook vocabulary; stop and compare with P3.2's preflight.
- **Queue wait over 30 minutes:** the orchestrator tells the human; no fallback accelerator.
- **A resume from `best.pt`** (`ValueError: ... best validation point only`): resume from the
  folder (its `latest.json` names a full checkpoint).
- **`source_weights names [...], the training split holds [...]`** (exit 1 at setup): the weights
  must name exactly the training split's sources; a resume from a checkpoint of other weights is
  refused as another setting.
- **The embedding cache not found** (`OSError` at setup, a read-only mount): the attached `MT-01`
  output is not mounted where the argument points, or its namespace differs (other views); check
  `run.json` `embeddings` and the notebook's input folders. Never let a training session encode
  the full pool itself (about 7 h); a partial cache costs about 0.5 GPU-seconds per missing
  pair.
- **Config error (exit 2):** the message on stderr names the key; nothing is written, since the
  config is checked before `out_dir` is created.
- **`inner_probe: no inner rate gives finite post-k scores`** (exit 1 at the start
  validation): every probed rate overflowed; lower the rates by a factor of 4 in a new revision.
- **`... records no inner-rate probe` or `no fresh-schedule phase`** (exit 1 at a resume): the
  checkpoint is not a probed or phased warm start; resume with the config that wrote it.
- **The probe picks the largest or the smallest rate:** the best rate may lie outside the grid;
  the run trains at the edge, and the summary's probe table shows the trend for the next
  revision's grid.
- **A failure in a worker process:** with `gpus` above 1 it reaches the script as torch's
  `ProcessRaisedException`; `status.json` and `summary.md` give its last line, the worker's own
  exception, and `error.log` the whole traceback.

## Meta-training notes

### How episodes are built and why they are split by family

P3.4 (episode note, sections 4 and 4.5; ADR-012) builds each episode from one task: $|D|$ of 1 to
5 demonstrations drawn with $P(|D|) = (0.15, 0.35, 0.30, 0.15, 0.05)$, one query, and one
augmentation for all pairs; ARC tasks cycle the query over their pairs, ReARC and BARC pools take
fresh pairs. The validation split holds 200 puzzles whose families (a ReARC generator with the ARC
task of the same id, or one BARC problem) are absent from training, and no training pair is
canonically equal to a validation pair. Splitting by family rather than by puzzle matters because
a family's puzzles share one transformation: a puzzle-level split would score memorized
transformations on new grids, not new transformations.

### Why gradients flow through the frozen base

The loss is defined on the base's output under the injected adapter, so its gradient with
respect to H's parameters $\phi$ passes through the base:
$\nabla_\phi \mathcal{L} = \sum_{\ell, m} \big(\frac{\partial \mathcal{L}}{\partial A_{\ell,m}}
\frac{\partial A_{\ell,m}}{\partial \phi} + \frac{\partial \mathcal{L}}{\partial B_{\ell,m}}
\frac{\partial B_{\ell,m}}{\partial \phi}\big)$, where
$\partial \mathcal{L} / \partial B_{\ell,m} = s\, \delta_{\ell,m} (A_{\ell,m} x)^\top$ needs the
backpropagated error $\delta_{\ell,m}$ at that projection, which only a backward pass through the
layers above it provides. The base's weights are frozen (no gradient is stored for them), but its
activations and their gradients are computed: that is the cost of a step, about that of LoRA
training. No surrogate loss in weight space would do this (Q6).

### What the zero-initialized $B$ head guarantees

H's heads start with weight 0 and bias $(\mathrm{vec}(A_0), 0)$ (bias-HyperInit, P3.3), so every
emitted adapter at step 0 is $(A_0, B = 0)$ and the delta $s B A x$ is exactly zero: the adapted
base equals the base, bitwise (tested in fp32, and the step-0 validation records
$\max |\text{current} - \text{zero}|$, expected 0 on the GPU too). Training therefore starts from
the zero-adapter baseline, never below it by construction. $\partial \mathcal{L} / \partial B
\neq 0$ from the first step, while $\partial \mathcal{L} / \partial A = 0$ and the trunk receives
no gradient until the head weights have moved (a zero weight passes no gradient back): the first
update moves only the $B$ part of each head, and from the second step every parameter of H
learns.

### Learning a transformation or memorizing

| Pattern | Training CE | Validation, current vs zero | Mismatched vs current |
|---|---|---|---|
| Learning transformations | falls | current below zero, growing | mismatched above current |
| Generic adapter (format, colors) | falls | current below zero | mismatched about current |
| Memorizing the toy set | falls fast, especially after the first pass (step 500) | flat or rising after an early dip | about current |
| Not learning | flat | about zero | about current |

H learning transformations shows as current $<$ mismatched $<$ zero: the gain depends on the
matching $\bar z$. Four passes over 2000 episodes make memorization likely late in the toy run;
the per-pass shape of the training curve (steps 500, 1000, 1500) and the gap between training and
validation CE show it. Per-source validation CEs separate BARC (new to the base) from ReARC (the
base was fine-tuned on regenerated ReARC; episode note 4.6): a gain on ReARC only is weaker
evidence.

### Resuming across sessions

A stop (exit 3), a kill (exit 4), and the end all write a checkpoint; a session that Kaggle ends
leaves the last periodic one. Checkpoints live in `/kaggle/working/checkpoints/` of the notebook's
output. Kaggle refuses a notebook's own output as an input (`invalidKernelSources`,
`docs/research/kaggle_first_runs.md`, section 3.6), and the builder refuses a config that
attaches its own notebook, so a continuation runs in a notebook of another slug, with its own
build config and brief: it pins the stopped notebook's output in `notebook_inputs` (for example
`alyahya737/arc26-p3-5-train: <version of the stopped session>`) and passes
`--resume {input:alyahya737/arc26-p3-5-train}/checkpoints` in `args`; the notebook finds the
mount by search and passes the folder (`docs/runbooks/build_submission.md`, script mode). The
script reads `latest.json`, refuses a checkpoint written with other settings (`RESUME_KEYS`:
data, loss mask, model, H, world size, accumulation, schedule, seed), restores H, AdamW, the
scheduler, the step, the data cursor, and every process's RNG state, rewrites `metrics.jsonl`
from the checkpoint's history, and continues; new checkpoints go to
`/kaggle/working/checkpoints/` of the continuation. The world size must match. Embeddings are
re-encoded (about 6 minutes for the toy set; a long run reads a prebuilt cache instead).

`MT-03` and `MT-04` are this case at full scale: `configs/script_MT-03.yaml` attaches `MT-02`'s
output (version 1) and passes `--resume {input:alyahya737/arc26-mt-train-1}/checkpoints`, and
`configs/script_MT-04.yaml` attaches `MT-03`'s (version 1) and passes `--resume
{input:alyahya737/arc26-mt-train-2}/checkpoints`; their embeddings come from `MT-01`'s cache, so
their setup takes minutes. `MT-02`'s checkpoints were written under its `--max-wall-s` 12,600;
the budget is not a resume setting, so `MT-03` resumes them under its own `--max-wall-s` 38,700
(revision 3: the config's 41,400 s less the pre-script allowance). The
best-point tracking continues over every session: the checkpoint's history holds the earlier
validations, so a session rewrites `best.pt` (in its own output) only for a new low, and
`status.json` `best.session_start` says which session's output holds the best `best.pt` (0:
`MT-02`'s). A resumed session that stopped before its first step would write no checkpoint of
its own (its last is the input's), so the next session would attach the earlier output; with a
budget of hours this does not arise.
