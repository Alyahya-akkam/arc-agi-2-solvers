# Runbook: H's validation (P3.7): `python -m arc26.arms` and `scripts/h_validation_report.py`

Task P3.7 (BRIEF B5 P3.7, B7 G4; task brief `docs/tasks/P3.7.md` with its amendments of 4 and 17
Oct),
[ADR-019](../decisions/ADR-019-h-validation-paired-arms.md). Code: `arc26/arms.py` (the arms
entry), `arc26/hyper/validation.py` (GEN and the load check), `arc26/decode.py` (the `views`
keyword), `arc26/runner.py` (`RunConfig.adapters`), the builder's `adapters` option
(`scripts/build_kaggle_notebook.py`, `scripts/build_submission.py`), and
`scripts/h_validation_report.py` (the table and `--smoke`). Configs: `configs/P3.7-01.yaml`
(DEV40), `configs/P3.7-02.yaml` (GEN), `configs/P3.7-03.yaml` (H as the fit's start), and
`configs/P3.9-02.yaml` (P3.7-03's arms and arm (g) on the FOMAML-trained H, task P3.9b). Pre-run briefs:
`docs/prerun/P3.7-01.md`, `docs/prerun/P3.7-02.md`, `docs/prerun/P3.7-03.md`,
`docs/prerun/P3.9-02.md`. Module docs: `docs/modules/arms.md`, `docs/modules/validation.md`.

## 1. Why this script exists

Gate G4 decides Phase 4: does H, the hypernetwork that maps a puzzle's demonstrations to a LoRA
adapter, do a meaningful share of what test-time training does? B5 P3.7 measures it as $f_b$, the
fraction of the gap between the zero adapter (a) and the SGD adapter (c) that H (b) closes, on
held-out generated puzzles (GEN, the distribution H was trained on, unseen families) and on
DEV40 (the competition's distribution): at least 50% means Phase 4 as planned, 25% to 50% Phase 4
with the heaviest polish that fits, below 25% Submission 2's system is final. Arm (d), H plus a
13-step polish, is ADR-003's production default; its $f_d$ prices the polish. The two notebooks
(`P3.7-01` DEV40, `P3.7-02` GEN) produce the table; the report script turns their pulled logs into
it and into G4's line.

`P3.7-01` read Bad at `MT-02`'s step 1,000 (H 4.0, zero 4.0, SGD 9.5, H plus 13 steps 5.0 of 40).
Those arms test H as a replacement for the fine-tuning. Run `P3.7-03` (task brief amendment 8,
the human's question) tests H as its **start**: arm (f), `hyper_fit`, runs arm (c)'s one-epoch
fit from H's adapter instead of the zero-initialized LoRA, paired with (c) on DEV40, so
$m_f - m_c$ isolates the starting adapter; both arms log their per-step training loss, which
says whether (f) reaches (c)'s final loss sooner.

Run `P3.9-02` (task P3.9b) repeats `P3.7-03` with one change: H is `P3.9-01`'s, trained 80 steps
further from `MT-03`'s step 4,180 with the first-order meta-learning objective (FOMAML: the query
loss after $k = 2$ inner Adam steps on the demonstrations, ADR-006 amendment 2). Its teacher-forced
validation improved after the inner steps ($z_S = -4.48$, 199 to 209 of 286 exact), which cannot
say whether more ARC-AGI-2 tasks are solved; the same four arms on the same DEV40 queue can, each
paired with `P3.7-03`'s.

`P3.9-02` also runs arm (g), `hyper_merge_fit` (task P3.9b amendment 1, the human's choice A of 21
Oct): the human's idea of 17 Oct was merge-then-fine-tune, and arm (f) is not that: it starts the
fit's factors at H's ($A, B$ both nonzero), and in `P3.7-03` it ended with a higher final training
loss than (c) on 38 of 40 puzzles, which points at the factor start rather than the function
start. Arm (g) merges H's adapter into the base weights and trains a fresh LoRA from the standard
start ($B = 0$), HyperDreamBooth's recipe (Ruiz et al. 2024, arXiv 2307.06949;
`docs/research/hypernetwork_literature_generation.md` section 6.3 on `task/ideas`), so the fit has
(c)'s dynamics on a base shifted by H's function. It runs fifth, after `P3.7-03`'s four arms.

## 2. What it does, step by step

The notebook is the submission notebook (environment check, offline install, Run, Write, logs)
with the build's `ADAPTERS`; the Run cell starts `python -m arc26.arms` through
`arc26.runner.run_solver`, with the build's deadlines.

1. **Folders.** The parent clears its candidate root (`/kaggle/working/arm_outputs/`), its log
   folder (`run_logs/arms/`), and the workers' cache root (`/kaggle/h_cache/`, one writable cache
   per worker, `rank<k>`). The caches stay outside `/kaggle/working`, the notebook's output that
   Kaggle keeps and later notebooks attach: they are scratch rebuilt by every run (a few MB),
   written by four workers with one writer each, and DEV40's entries embed evaluation pairs, which
   no attachable output should carry.
2. **GEN** (`P3.7-02` only; `arc26.hyper.validation.gen_from_episodes`). It finds P3.4's build 2
   by its manifest sha256 and its passing leakage report, reads the validation split (286
   episodes of 200 puzzles, every shard checked against the manifest), takes per puzzle the
   episode of the lowest index, and keeps 50 BARC puzzles (sorted id) and 50 ReARC puzzles dealt
   round-robin over the 20 held-out families. Each becomes a competition-format puzzle (the
   episode's demonstrations as `train`, its query's input as the one test input, its output as
   the solution) in `/kaggle/working/gen_data/`, under the competition's file names; the
   workers' `data_dir` is that folder.
3. **The load check** (child process on GPU 0, `python -m arc26.hyper.validation check`). With
   $c_i$ the recomputed and $\ell_i$ the logged query CE of validation episode $i$ (MT-02's best
   point, `status.json` and `metrics.jsonl`),

   $$\mathrm{CE}_i = -\frac{1}{|O_{Q_i}|} \sum_{t \in O_{Q_i}}
   \log p\big(y_t \mid y_{<t};\ \theta_\text{base} + \Delta\theta_{H(\bar z(D_i))}\big),$$

   it passes when $\max_i |c_i - \ell_i| \le 10^{-3}$ and $|\frac{1}{n}\sum_i (c_i - \ell_i)| \le
   10^{-4}$ for the current H and for the zero adapter, the checkpoint's step is the best step,
   its H config equals `configs/hypernet.yaml` as built, and its manifest the attached build's.
   It also measures the bf16 cast, $d_i = \mathrm{CE}^{\text{bf16}}_i - \mathrm{CE}^{\text{fp32}}_i$,
   flagged when $\bar d > 2\,\mathrm{SE}(d)$ with $\mathrm{SE}(d) = s_d/\sqrt{n}$, and H's gain
   $\mathrm{CE}^{\text{fp32}} - \mathrm{CE}^{\text{zero}}$; and on DEV4 emits H's adapter three
   times (two fresh caches, one with the pairs shuffled) and compares the sha256 digests. A failed
   check ends the run with exit 5 before any decode.
4. **The queue.** The puzzles (DEV40's 40 ids, or GEN's keys) in the order of
   $\mathrm{sha256}(\texttt{"0:"} + \text{id})$, then one `None` per worker.
5. **The workers** (one per L4, `arc26.arms.local_worker`, the reference's process body and
   import order). Each loads the base with the build's scope (`arc26.solver.load`: rank 16 on q,
   k, v, o, rsLoRA $s = 32/\sqrt{16} = 8$, cast to bf16), builds H from `best.pt` on the live
   model, and **probes H's path** once: H's adapter for the DEV4 puzzle with the fewest
   demonstration cells (by `P2.6-01`'s token counts `36a08778`, about 2,100 tokens per view)
   through the live encode and the swap on this unsloth-patched model, then the rollback, bit for
   bit (a few seconds; the load check ran transformers' base, so this path is new on GPU). A probe
   that raises stops the run (below). Then it loops: no puzzle starts after the soft deadline $T$;
   per puzzle, every arm in order:
   - `base`: the adapter's initial weights (`lora_B` $= 0$, so $W x + s\,B A x = W x$);
   - `h`: copies the embeddings `MT-01` cached for the puzzle's pairs into the worker's cache,
     encodes the rest (8 views each, on the frozen base under `disable_adapter`), and swaps in
     $H(\bar z(D))$, $\bar z(D) = \frac{1}{|D|}\sum_{i \in D} \bar e_i$ in pair-hash order;
   - `sgd`: `arc26.ttt.fit` from the initial weights, one epoch over the 128 views;
   - `h-polish`: $H(\bar z(D))$, then `SGDProvider` for $s = 13$ steps from it;
   - `h-fit`, source `hyper_fit` (`P3.7-03`): H's adapter as `h` loads it, copied to the CPU
     (`snapshot`), then `arc26.ttt.fit` with that copy as its `default_weights` and every other
     argument `sgd`'s: the same 128 views, one epoch, a fresh AdamW, the same warmup and cosine;
   - `h-merge-fit`, source `hyper_merge_fit` (`P3.9-02`): H's adapter as `h` loads it (its
     digest recorded as `h_fingerprint`); `arc26.arms.BaseMerge` copies the base weight of every
     LoRA layer (q, k, v, o of 36 layers, `base_layer.weight`, the tensor PEFT's forward and
     unsloth's fast paths read) on the GPU, then writes, layer by layer,
     $$W_\ell \leftarrow \mathrm{bf16}\big(W_\ell + s\,B_{H,\ell} A_{H,\ell}\big),
     \qquad s = \alpha/\sqrt{r} = 8,$$
     the product in float32 and one rounding (`merge` op; the record's `merge` keeps the layer
     count, the copy's bytes, $\lVert\Delta\rVert_F$, and the rounding's share
     $\rho = \lVert W' - W - \Delta\rVert_F / \lVert\Delta\rVert_F$); then `arc26.ttt.fit`
     exactly as `sgd` (it resets the LoRA to the initial weights, $B = 0$, before any forward);
     the CE and the decode as every arm; then, in a `finally` around the arm, the copy goes back
     into every weight bit for bit and is checked (`unmerge` op, `restored` true), also when the
     arm raised, before the next arm;
   with `train_log`, every training arm's trainer logs each step (`logging_strategy` `steps`,
   `logging_steps` 1) and its record keeps `train_log`, per step $\{s, L_s, \eta_s,
   \lVert g_s \rVert\}$, read off the trainer's `log` calls around the fit;
   then, for the arm: the adapter's sha256, the CE of every answer (one forward of the identity
   view's prompt and reply under `torch.inference_mode`, `calc_scores`, divided by the reply's
   tokens), and `decode_puzzle`
   with the build's views into `arm_outputs/<arm>/`, on the arm's own clock (the 1200 s cap from
   the arm's start), decoding stopped at $T + $ `finish_s`. Each arm's timing records go to
   `run_logs/arms/<arm>/timing/`, one record per puzzle and arm to
   `run_logs/arms/records-rank<k>.jsonl` (status `done`, `cut`, `not_started`, or `raised`).
   **The stop rule:** if the probe raises, or an H arm (`h`, `h-polish`, `h-fit`, `h-merge-fit`)
   raises on the first puzzle the worker finishes, or the base weights of `h-merge-fit` could not
   be put back (on any puzzle: every later arm would run on H's merge), the worker writes
   `run_logs/arms/stop-rank<k>.json` and exits;
   the parent then writes `arms.json` without a selection and exits 7 (the load check's stop is 5).
   A later H arm that raises is recorded and the run goes on. Its cost: about 1,100 s of wall
   (launch, load check, the workers' loads) for a probe that raises, at most one more puzzle per
   worker (DEV40 about 1,500 s on average, GEN about 560 s) for a first-puzzle failure, instead of
   every H arm `raised` until the deadline.
6. **Selection and scores** (the parent, after the workers): per arm the reference's selection
   (`write_submission` over the arm's folder: `score_kgmon`, two attempts) into
   `run_logs/arms/<arm>/submission.json`, scored on the run's puzzles against the solutions
   (development mode only): per puzzle the task score $c_t = \frac{1}{n_t}\sum_i
   \mathbb{1}[\text{an attempt equals } y_{t,i}]$ and the attempt-1 matches; then
   `run_logs/arms/arms.json`.
7. **The report** (locally, `scripts/h_validation_report.py`). The paired set $P$ is the puzzles
   every arm decoded to the end and scored. Per arm: $m^{\text{acc}} = \sum_{t \in P} c_t$ (tasks),
   $m^{\text{a1}}$ = attempt-1 matches over outputs, $m^{\text{ce}}$ = the mean over $P$ of the
   per-puzzle mean CE; then

   $$f_x = \frac{m_x - m_a}{m_c - m_a}, \qquad x \in \{b, d\},$$

   with the 5th and 95th percentiles of $f_x$ over 10,000 resamples of $P$ with replacement
   (paired: one resample of puzzles for every arm; seed 0) as the 90% interval. On task accuracy
   $f_x$ is undefined unless the SGD arm beats the zero adapter by the noise floor, $m_c - m_a \ge
   2$ (signed; tasks on DEV40, puzzles on GEN), and G4 then reads CE's. GEN also gets BARC and
   ReARC rows. Then G4's rows, No gap first (below, section 6), and the decision over both sets,
   whose consequences depend on `--training-ended`.

## 3. Why it is built this way

| Choice | Alternative considered | Why it lost |
|---|---|---|
| All arms of a puzzle in one worker before the next (paired) | One notebook or one series launch per arm (P1.3's series mode) | A cut would end each arm at its own puzzle; four model loads and queues; ADR-019 |
| A fixed shuffled queue order (sha256 of seed and id) | Sorted ids | GEN's sorted keys put all BARC before ReARC, so a cut would drop a whole source; Python's `hash` is salted, `random` may change across releases (`arc26.data.seeded_key`'s reason) |
| No puzzle after $T$, its arms until $T$ + `finish_s` | One deadline for everything | A puzzle cut mid-arms is lost to every arm; a grace bounded by the longest puzzle (DEV40 about 4,100 s, GEN about 1,800 s, estimates) keeps nearly every started puzzle |
| Each arm's own clock for `decode_puzzle`'s 1200 s cap | The puzzle's start for every arm | P2.6-01's cap counted from its puzzle's start, before the fit: the arm's start reproduces that for the SGD arm and gives every arm the same budget |
| H's adapter in bf16 (the solver's cast) | Float32 for the H arms (ADR-017's open item) | One numerics path for all arms (task brief amendment 3); unsloth's patched forward may cast the factors anyway; the load check measures the rounding on 286 episodes (ADR-019) |
| GEN at 8 views, every dihedral map once, in two full batches | 16 views (7.1 h for 100 puzzles); 64 puzzles at 16 views; the first 8 views in list order | Over budget; a wider interval on $f_b$; no reflections at equal cost (ADR-019) |
| DEV40 at 16 views | 8 views | Its SGD arm is then P2.6-01's build, comparable with `P2.6-01`'s DEV40 subset (10.5 of 40) and `P1.3-01` arm 2 (10.5) |
| GEN: lowest episode, 50 BARC by sorted id, 50 ReARC round-robin over families | All 200 puzzles; random draws | 200 does not fit (14 h, estimate); sorted ids and round-robin make the subset a function of the build alone, every held-out family 2 or 3 puzzles |
| The load check through `train.py`'s validation path, tolerances $10^{-3}$, $10^{-4}$ | Reload H and trust it; compare means only | A wrong checkpoint, config, or injection is caught before 4 h of decode; per-episode bounds sit 40 times under `P3.5-01`'s smallest neighbouring change and 2,000 times over the log's rounding |
| The load check in a child process before the workers | In each worker | It must stop the run before any decode, and frees GPU 0 before worker 0 loads |
| Each worker probes H's path on its own model, and an H arm failing on its first puzzle stops the run (exit 7) | Record failing H arms and go on; probe in the load check's process | A path broken on the unsloth model would fail every H arm for about 2.6 h; the load check's process runs transformers' base, and probing the unsloth model there costs a second model load, while the worker's probe costs seconds (ADR-019, decision 8) |
| The workers' caches at `/kaggle/h_cache/rank<k>` | Under `/kaggle/working/` | `/kaggle/working` is the output later notebooks attach; the caches are scratch, one writer each, and DEV40's embed evaluation pairs |
| H's checkpoint a config-only swap: `adapters.checkpoint`, `adapters.load_check.run_logs`, and their `notebook_inputs` entries | A build flag (`--checkpoint`); the slug repeated in tests and code | A flag makes the build a function of the command line, not of the commit, so the brief's `Code-commit` and hashes would not pin H (A8: config-driven); repeated pins turn a swap into a code change; the builder already refuses a placeholder whose output is not attached and an attached output no path reads, so a half swap cannot build (`tests/test_arms_build.py`); a checkpoint and logs of sessions that do not belong together do build, and the load check stops them (exit 5) |
| Arm (f) as `sgd`'s `fit` call with H's adapter as `default_weights` (`hyper_fit`) | `hyper_polish` with `steps: 128` (no new code) | `SGDProvider` sets `max_steps` 128 and builds its views from `DemoSet.task()`: when `cut_to_len` drops a view over 8,192 tokens, one epoch is fewer than 128 steps, so the trainer would run past it into a second pass and stretch the warmup and the cosine over 128 steps, not (c)'s; the same call guarantees the same data, steps, and schedule (`tests/test_arms_hyper_fit.py` checks every argument) |
| Arm (g) merges H's adapter into the base and fits a fresh LoRA from $B = 0$ (`hyper_merge_fit`, task P3.9b amendment 1) | Arm (f) alone, H's factors as the fit's start | (f) is not the human's merge-then-fine-tune: from $\theta_H$ both factors move from step 1, and in `P3.7-03` (f) ended with a higher final training loss on 38 of 40 puzzles (median ratio 2.0) at the same score; HyperDreamBooth's working recipe merges the prediction and trains a fresh LoRA (arXiv 2307.06949; `hypernetwork_literature_generation.md` 6.3), which keeps (c)'s dynamics |
| The delta computed by `BaseMerge` in float32 from the live factors and written into `base_layer.weight` | PEFT's `merge_adapter` or `merge_and_unload`, then a new adapter | PEFT 0.17.1's `get_delta_weight` computes $B A$ in float32 only on the CPU (`cast_to_fp32` for a CPU bf16 weight), in bf16 on the GPU; `merge_and_unload` drops the LoRA layers the solver's trainer and `default_weights` address; its `unmerge` subtracts |
| The base restored from an exact copy | Restore by subtracting $s\,B A$ | bf16 rounds $W + \Delta$, so $\mathrm{bf16}(W + \Delta) - \Delta \ne W$ for most weights (`tests/test_arms_merge_fit.py` measures it on a bf16 LoRA): the drift would reach every later arm and puzzle, and the next puzzle's encode |
| The copy on the GPU | Pinned CPU memory; reload q, k, v, o from the model files after the arm | `P3.7-03`'s highest per-puzzle peak was 14,190 MiB (its session log), so the copy's 1,800 MiB gives about 15.6 GiB of the L4's 22.5 GiB; on the CPU the four workers would pin 7.5 GB of host RAM for headroom the run does not need; a reload is cheap (a worker's `load` took about 6 s in `P3.7-03`, its timing records) but makes the restore depend on the read-only mount and on `load`'s transforms (unsloth's patches, the bf16 cast) giving the same bits, while the copy is checked against itself in the process with `torch.equal` |
| Weights written in place where every path reads them | A second, frozen LoRA holding H beside the trained one | unsloth 2025.9.7's fast paths (`get_lora_parameters`, `fast_linear_forward`) read `base_layer.weight` and the first active adapter only, so a second active adapter would be dropped from the fast forward; `base_layer.weight` is read by every path (checked on the wheel's source) |
| (g) fifth, after `P3.7-03`'s four arms | (g) elsewhere in the order | The first four arms then run as in `P3.7-03` on every puzzle and pair with its arms; (g)'s encode is a cache hit from (b) |
| Per-step loss read off the trainer's `log` calls (`train_log_capture`), the logging switched on through `train_args` | A callback or a change to `fit` | `fit` is moved reference code checked statement by statement (`tests/test_split_solver.py`, ADR-011) and builds its trainer inside, so neither a callback nor a return value reaches it without a change; logging reads the step's loss and changes no gradient, update, or random draw |
| `P3.7-03` on its own code dataset `alyahya737/arc26-code-p37f` | A new version of `alyahya737/arc26-code-p37` | The notebooks of `P3.7-01` and `P3.7-02` attach that dataset unversioned and check its wheel by sha256, so a new wheel there would stop `P3.7-02` (one code dataset per concurrent build lineage) |
| `P3.9-02` reads H from `P3.9-01`'s Modal-published dataset through `dataset_inputs`, pinned as `dataset_sources` `<owner>/<slug>/<version>` (task P3.9b) | A copy notebook that attaches the dataset and writes `best.pt` and the logs into its output, read through `notebook_inputs`; the dataset as a `kernel_sources` entry; the dataset attached unversioned | The copy is one more Kaggle session, brief, and wait for no new information, and its output would be a second copy of 0.54 GB; `kernel_sources` names notebooks only, so Kaggle would not mount a dataset there; unversioned would read whatever a later version holds, while the pin keeps the build a function of the commit, as `notebook_inputs` does. kaggle-cli 2.2.4 accepts `<owner>/<slug>/<version>` (`validate_dataset_string`); whether the server honors the version is unverified (`docs/research/kaggle_first_runs.md` 3.1). Version 2 (`P3.9-01r`, task P3.9r) is the dataset's latest, so an unversioned reading also mounts it; version 1 holds only four root files |
| `P3.9-02` reads `best.pt`, not `ckpt-004260.pt` | The full checkpoint of the same step | Both hold H at step 4,260 (`status.json` `best.step` is the last step), but `best.pt` is the weights-only file the load check and `HyperProvider.from_checkpoint` were built for, its config without the FOMAML keys (`tests/test_train_fomaml.py`), 0.54 GB against 1.6 GB |
| `--smoke --layout CONFIG` trains the smoke's H as `P3.9-01` trained H and lays it out at the config's paths | The toy H of `--smoke`; `tests/test_train_fomaml.py`'s load test alone | The toy H is zero-step, so a FOMAML file and the dataset's layout (version 2: `best.pt` and the logs at its root, task P3.9r) would first meet the arms on Kaggle; the unit test calls the loader's steps by hand, not `arc26.arms.run` with every arm |
| `MT-01`'s cached embeddings copied per puzzle into a cache per worker | Encode GEN's pairs again; one shared writable cache | H was validated on `MT-01`'s embeddings; `EmbeddingCache` has one writer; copying is bit for bit |
| The CE by one forward of the identity view (`calc_scores`) | The 8 rescoring views; a full sweep | The task brief: one forward, the continuous metric with the smallest noise; the reference's scorer keeps the reply's tokens as decoding scores them |
| Arm (e) implemented, off | Run it | The cut order: with (e) DEV40 takes about 6.0 h (estimate); P4.3 lists it |
| `P3.9-02`'s deadlines re-priced from `P3.7-03`'s measured arm seconds, (g) at (c)'s (soft deadline 21,600 s, `finish_s` 5,200 s, hard stop 27,700 s, `Timeout` 30,700 s) | `P3.7-03`'s deadlines (16,200 s, 3,900 s, 21,000 s); the task brief's 10.4 quota h, five fifths of `P3.7-03`'s | (g) trains and decodes as (c) does, 479.8 s per puzzle against the four arms' mean of 324 s, so the workers' span grows by 1.372, not 1.25; `P3.7-03`'s soft deadline would leave about a fifth of the queue not started (14,453 s of the workers' 17,865 s before it) |
| `Timeout` = the hard stop + 300 s of logs + 2,700 s (25,500 s and 22,500 s), the deadlines counted from the first cell | `Timeout` = the hard stop + 300 s (revision 1) | Kaggle has spent 37 to 41 minutes between RUNNING and the first cell (`P6.5-02`, `P6.3-02` revision 5), and the `Timeout` may count from RUNNING: the 1.4x cases (20,617 s and 17,710 s) would have kept only 36 and 35 minutes; the allowance follows `MT-03` revision 3 (`fde6baa`) |
| Four workers on `NvidiaL4` (4xL4) | 1xL4, 2xT4 | bf16 (T4 has none); `NvidiaL4X1` is not granted; P2.6-01's timings price this run |
| One seed (`hash_seed` 0, `order_seed` 0, the bootstrap's 0) | Several seeds | CLAUDE.md: one fixed seed for all comparisons |

## 4. How to run it

**On Kaggle** (the orchestrator, after the human approves the brief; never by an agent before):

```bash
python scripts/build_submission.py --tag P3.7-01 --config configs/P3.7-01.yaml
python scripts/build_submission.py --tag P3.7-02 --config configs/P3.7-02.yaml
python scripts/kaggle_ops.py dataset-create dist/P3.7-01/arc26-code   # once: alyahya737/arc26-code-p37
python scripts/kaggle_ops.py push dist/P3.7-01 --run-id P3.7-01
python scripts/kaggle_ops.py push dist/P3.7-02 --run-id P3.7-02
python scripts/kaggle_ops.py pull alyahya737/arc26-h-val-dev40 --task-id P3.7-01
python scripts/kaggle_ops.py pull alyahya737/arc26-h-val-gen --task-id P3.7-02
# P3.7-03, arms (c) and (f) on DEV40, from its own code dataset:
python scripts/build_submission.py --tag P3.7-03 --config configs/P3.7-03.yaml
python scripts/kaggle_ops.py dataset-create dist/P3.7-03/arc26-code   # once: alyahya737/arc26-code-p37f
python scripts/kaggle_ops.py push dist/P3.7-03 --run-id P3.7-03
python scripts/kaggle_ops.py pull alyahya737/arc26-h-val-fit --task-id P3.7-03
python scripts/h_validation_report.py --run DEV40=runs/P3.7-03/run_logs \
    --out reports/P3.7-03/fit.md --json reports/P3.7-03/fit.json
python scripts/h_validation_report.py --run DEV40=runs/P3.7-01/run_logs \
    --run GEN=runs/P3.7-02/run_logs --out reports/P3.7/report.md --json reports/P3.7/report.json
# add --training-ended only once H's meta-training chain has stopped for good
# P3.9-02, P3.7-03's arms and arm (g) on P3.9-01's FOMAML-trained H, from its own code dataset
# (created at version 1 from 0c2beed's build; a new build adds a version):
python scripts/build_submission.py --tag P3.9-02 --config configs/P3.9-02.yaml
python scripts/kaggle_ops.py dataset-version dist/P3.9-02/arc26-code -m "P3.9-02 revision 3"
kaggle datasets status alyahya737/arc26-code-p39v    # the human, until ready
# the human: H's dataset after P3.9-01r (docs/runbooks/modal_republish.md): version 2, about 0.54 GB
kaggle datasets status alyahya737/arc26-p3-9-01 --format json
kaggle datasets list --mine -s arc26-p3-9-01
python scripts/kaggle_ops.py push dist/P3.9-02 --run-id P3.9-02
python scripts/kaggle_ops.py pull alyahya737/arc26-h-val-fomaml --task-id P3.9-02
python scripts/h_validation_report.py --run DEV40=runs/P3.9-02/run_logs \
    --out reports/P3.9-02/h_validation.md --json reports/P3.9-02/h_validation.json
```

`P3.9-02` attaches H's output as a dataset, not a notebook output: `P3.9-01` ran on Modal, whose
publish stage uploads a private dataset. Version 1 of `alyahya737/arc26-p3-9-01` holds only the
run folder's four root files: kagglehub 0.3.12 never attaches files below a folder's root to a
version (task P3.9r, `modal_republish.md`). Version 2 (`P3.9-01r`) holds
`best.pt`, attempt 2's 11 run logs, and a `MANIFEST.json` with every file's size, sha256, and
source path, all at its root. The config pins it under `dataset_inputs`, and
`{input:alyahya737/arc26-p3-9-01}` names its mount (`/kaggle/input/datasets/alyahya737/arc26-p3-9-01`,
the code dataset's layout in `P3.7-03`'s session log) as it names a notebook output's. Its
checkpoint and logs lines are:

```yaml
adapters:
  checkpoint: "{input:alyahya737/arc26-p3-9-01}/best.pt"
  load_check:
    run_logs: "{input:alyahya737/arc26-p3-9-01}"
dataset_inputs:
  alyahya737/arc26-p3-9-01: 2
```

Attempt 2's logs, because attempt 1 stopped before a training step; its `status.json` names step
4,260 and its `metrics.jsonl` holds that step's per-episode 0-step CEs, which the load check
reproduces. The load check opens only `status.json`, `metrics.jsonl`, and `run.json` in the logs
folder (`arc26.hyper.validation.logged_best`), so `best.pt`, the manifest, and the other eight
logs beside them are never read.

Both builds hold the same wheel, so one dataset version serves both. `kernel-metadata.json`
attaches the competition, the base model, the install notebook, the code dataset, and the four
notebook outputs: `alyahya737/arc26-mt-train-1` version 1 (`best.pt`, MT-02's run logs),
`alyahya737/arc26-mt-cache` version 1, `alyahya737/arc26-p3-4-episodes` version 2,
`alyahya737/arc26-p3-4-leakage` version 2.

**H's checkpoint** is `MT-02`'s `best.pt` (output version 1): H at step 1,000 of 5,600 (the
session stopped at its cap at step 1,028; `status.json` `best.step`, the last validation that
`metrics.jsonl` flags `best`), validation CE 0.024783 against the zero adapter's 0.029248 over the
286 episodes, the values the load check reproduces per episode. A later best point (`MT-03` or
`MT-04`, `status.json` `best.session_start` naming the session that wrote `best.pt`) is a swap of
these lines in both configs, then a new commit, new builds, and an amended brief:

```yaml
adapters:
  checkpoint: "{input:<owner>/<slug of the session that wrote best.pt>}/checkpoints/best.pt"
  load_check:
    run_logs: "{input:<owner>/<slug of the last session>}/run_logs"
notebook_inputs:
  <owner>/<slug of the session that wrote best.pt>: <its version>
  <owner>/<slug of the last session>: <its version>   # one entry when the two are one output
```

The last session's `status.json` names the best point over all sessions and its `metrics.jsonl`
carries every session's validations (`train.py` restores the history from the checkpoint), so the
load check reads them there. The other three inputs and every other value stay. If a later session
ran but found no new low, `best.pt` stays in `MT-02`'s output and `MT-02`'s logs still name its
step, so nothing has to change; pointing `run_logs` at the last session's output is the
equivalent two-output case. The builder refuses a half swap (a path whose output is not attached,
an attached output no path reads), but a checkpoint and logs of sessions that do not belong
together still build: only the load check catches them (exit 5, the step or the CEs differ).

**On a plain 4-GPU machine** (bf16, the Kaggle image's stack): with the four outputs unpacked and
the paths filled as the Run cell's `adapter_settings` fills them,

```python
from arc26.runner import RunConfig, run_solver
cfg = RunConfig(data_dir=..., model_dir=..., outputs_dir="/tmp/inference_outputs",
                marker_dir="/tmp", subset=DEV40_IDS_OR_NONE, workers=4, hash_seed=0,
                timing_dir="run_logs/timing", lora_r=16,
                lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"), adapters=SETTINGS)
run_solver(cfg, time.time() + 18000, timeout=22500)
```

**CPU smoke** (BRIEF A8; no GPU, a tiny random Qwen3, the stand-ins of unsloth and datasets):

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/h_validation_report.py --smoke
```

It trains a tiny H with `train.py`'s smoke mode on the episode fixture, builds GEN from the
fixture (2 BARC, 1 ReARC), runs the load check, every arm (a) to (g) on one in-process worker
with `train_log` on, the selection, the scores, and the report (with the (f) against (c), (g)
against (c), and (g) against (f) readings)
into `runs/P3.7-smoke/` (about 15 s). With a FOMAML H at a build config's paths (task P3.9b):

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/h_validation_report.py --smoke \
    --layout configs/P3.9-02.yaml
```

It trains the MT smoke (12 steps, `configs/train_hypernet_MT.yaml`), then the P3.9 smoke
(`configs/train_hypernet_P3.9-01.yaml`, `objective: fomaml`, the inner-rate probe) warm-started
from its step 8, as `P3.9-01` started from `MT-03`'s step 4,180; copies `best.pt` and the run
logs to the config's relative paths under `runs/P3.9-02-smoke/inputs/<owner>/<slug>/`; then runs
the load check, every arm (a) to (g), the selection, and the report on that file (about 25 s).

**Expected cost** (estimates, the briefs' section 6): `P3.7-01` about 4.9 h of wall (4.1 to 5.7 h),
`P3.7-02` about 4.3 h (3.8 to 4.9 h), `P3.7-03` about 3.2 h (3.0 to 3.4 h), hard stops 22,500 s,
19,500 s, and 16,200 s after the first cell, `Timeout`s 25,500 s, 22,500 s, and 19,200 s (a 2,700 s
pre-script allowance; up to 45 more minutes of quota if Kaggle counts the time before the first
cell). `P3.9-02` (revision 2, five arms): about 5.5 h from the first cell (5.1 to 5.9 h), hard
stop 27,700 s, `Timeout` 30,700 s; arm (g) holds 1,800 MiB more for its span. GPU memory per worker: the bf16 base, H (540 MB), and P2.6-01's peaks (11.4 GB training, 15.1
GB inference) on 24 GB. Disk: candidates and GEN under 100 MB in `/kaggle/working`, the caches a few
MB.

## 5. Outputs

Pulled into `runs/<task-id>/` (`run_logs/` and `submission.json`):

| File | Content |
|---|---|
| `run_logs/arms/arms.json` | the settings, the queue's order, GEN's summary and index (no grids), the load check's exit code, each arm's selection result and score summary, the exit code, wall seconds |
| `run_logs/arms/load_check.json` | verdict and reasons; per-episode CEs (zero, fp32, bf16); both comparisons; the bf16 measurement and its flag; H's gain; DEV4 digests; the attention kernel |
| `run_logs/arms/gen.json` | GEN's counts per source and family, file hashes, index (`P3.7-02`) |
| `run_logs/arms/records-rank<k>.jsonl` | one line per puzzle and arm: status, seconds, adapter seconds, fingerprint, CEs, cache misses, copied embeddings, error; with `train_log` (`P3.7-03`) the training arms' per-step `train_log`; for `h-merge-fit` (`P3.9-02`) H's adapter digest `h_fingerprint`, the merge's numbers `merge` (layers, copy bytes, device, dtype, $\lVert\Delta\rVert_F$, $\lVert W\rVert_F$, the rounding and $\rho$), and `restored`; the worker's probe as arm `probe` |
| `run_logs/arms/stop-rank<k>.json` | only after a stop on H's path (exit 7): the reason and the failing record |
| `run_logs/arms/<arm>/timing/timing-rank<k>.jsonl` | the arm's ops: `reset`, `cache_seed`, `encode`, `h_forward`, `snapshot`, `swap`, `merge`, `unmerge`, `train`, `candidate_train`, `fingerprint`, `ce`, `decode`, `selection`, and the arm's span as `puzzle` |
| `run_logs/arms/<arm>/submission.json`, `scores.json`, `selection.log` | the arm's two attempts (all tasks of the data), its per-puzzle scores on the run's puzzles, the selection's printout |
| `run_logs/timing/` | `load` and `h_probe` per worker, the whole puzzle's `puzzle` record, the load check's records |
| `run_logs/stages.json`, `env.json`, `pip_freeze.txt`, `candidates.json` | the notebook's own logs (`candidates.json` is `{}`: the arms write elsewhere) |
| `submission.json` | placeholders (`[[0]]`): never submitted |

Stays on Kaggle: `arm_outputs/<arm>/` (the candidates, for P4.6) and `gen_data/`.

## 6. How to read the results

- **The load check first.** On `MT-02`'s `best.pt` it should report step 1,000 and mean CEs near
  0.024783 (current H) and 0.029248 (zero adapter); per source, current and zero, BARC 0.031495 and
  0.042915, ReARC 0.018256 and 0.015959 (`status.json`). `fail`: nothing was decoded; read the
  reasons (step, config, manifest, the per-episode differences), fix the input pin or the code,
  re-brief. `pass` with `bf16_minus_fp32.flagged` true: the H arms are a lower bound; the report
  says so and a float32 path is the follow-up (pre-registered). DEV4 digests unequal: recorded for
  ADR-017 (P4.2's check), the comparison stands. On `P3.9-01`'s `best.pt` (`P3.9-02`) it should
  report step 4,260 and mean CEs near 0.026337 (current H, no inner steps) and 0.029248; per
  source, current and zero, BARC 0.034930 and 0.042915, ReARC 0.017980 and 0.015959
  (`status.json` of attempt 2). Those CEs were logged on Modal's 4xL4 (torch 2.8.0, transformers
  4.55.4, peft 0.17.1, flash-attn 2.8.2, the versions of Kaggle's image): its zero-adapter CEs
  equal `MT-03`'s Kaggle-logged ones on all 286 episodes at the log's six decimals.
- **Exit 7, a stop on H's path.** Nothing more was decoded: read `stop-rank<k>.json` (the probe's
  or the first H arm's error) and the session log; fix H's path on the solver's model (the live
  encode under unsloth's `disable_adapter`, the swap), then an amended, re-approved brief.
- **Completeness.** The paired set should be all 40 (DEV40) and all 100 (GEN). Cut puzzles (a
  slow arm past the deadline) are dropped from every arm; more than 10% cut is reported and the
  next repetition repriced.
- **G4.** Per set, read in this order, each set in exactly one row. **No gap** first: the SGD arm
  beats the zero adapter on neither metric, $m^{\text{acc}}_c - m^{\text{acc}}_a < 2$ (signed)
  and $m^{\text{ce}}_c \ge m^{\text{ce}}_a$; it wins over the others, since $f_b$ on CE can
  then be defined but measures no closure. Else $f_b$ on task accuracy (CE when accuracy's is
  undefined): **Good**, at least 0.5, Phase 4 as planned; **Borderline**, 0.25 to 0.5, Phase 4
  with the heaviest polish that fits ($f_d$ says what 13 steps buy); **Bad**, below 0.25. The
  consequences of Bad and No gap depend on whether H's meta-training has ended (the chain stopped
  for good: `MT-04` at step 5,600, or a Kill or failure row with no resume planned; the report's
  `--training-ended`). Not ended (`MT-02`'s step 1,000, also while `MT-03` runs, and any later
  best point before the chain ends): Bad means H's training continues and P3.7 repeats on the
  chain's final best point, and Phase 4 does not stop. Ended: Bad means Submission 2's system is
  final and Phase 4 stops for the deadline. No gap on one set: G4 reads the other set alone; on
  both: not ended, training continues and P3.7 repeats; ended, H does not enter Phase 4 for the
  deadline. Sets in different rows, or an interval that crosses a boundary: the report says so
  and the human decides at G4.
- **Sanity rows.** DEV40's SGD arm should land within the 3-task spread of 10.5 of 40 (`P2.6-01`'s
  subset, `P1.3-01` arm 2); far outside it means the arms entry changed the build path.
- **`P3.7-03`, (f) against (c).** The report has no G4 line (no zero adapter, no H alone); it
  gives $\Delta = m^{\text{acc}}_f - m^{\text{acc}}_c$ in tasks with a 90% paired bootstrap
  interval: **Start helps** at $\Delta \ge 2$, **Start hurts** at $\Delta \le -2$, **Same
  score** between (the 2-task floor); then the mean training loss of both arms, a trailing mean
  over 8 steps (one view per step, so single steps are noisy), at steps 1 to 128, and per puzzle
  the first step $s^\ast$ at which (f)'s trailing loss reaches (c)'s final one:
  $$s^\ast_p = \min\{s : \bar L^{f}_{p}(s) \le \bar L^{c}_{p}(S_p)\}, \qquad
  \bar L(s) = \frac{1}{\min(8, s)}\sum_{u = \max(1, s - 7)}^{s} L(u),$$
  with $S_p$ (c)'s last step. **Sooner** when the median $s^\ast$ (a puzzle that never gets there
  counts as last) is at most half of $S$. The brief's section 1 says what each reading means.
- **`P3.9-02`, (g) against (c) and against (f).** The same reading with (g) in (f)'s place:
  $\Delta_g = m^{\text{acc}}_g - m^{\text{acc}}_c$ with its interval and the row **Good**
  ($\Delta_g \ge 2$), **Neutral**, or **Bad** ($\Delta_g \le -2$); then
  $m^{\text{acc}}_g - m^{\text{acc}}_f$ (**Merge ahead**, **Same score**, **Merge behind** at
  the same floor), each with the loss curves and the reach step. Before reading them: every
  `h-merge-fit` record `restored`, its `h_fingerprint` equal to the same puzzle's `h`
  fingerprint, and its $\rho$ (the share of H's delta the bf16 merge moved; a $\rho$ near 1 says
  the merge kept little of H's function, which reads (g) as (c) on a perturbed base).

## 7. Failure modes

| Symptom | Likely cause | First thing to change |
|---|---|---|
| Out of memory in an arm (status `raised`, error in the record) | an arm's peak above the L4's 24 GB (P2.6-01 peaked at 15.1 GB; H adds 0.5 GB and the encoder's prefill batches) | the arm is recorded, its puzzle left out of the pairs, and the next arm runs; if it repeats, the next build moves H off the GPU between H arms (a code change) or decodes fewer views per batch |
| Out of memory in `h-merge-fit` only | the copy's 1,800 MiB on top of the arm's peak | the arm is recorded and its base restored (no allocation); if it repeats, the next build keeps the copy in pinned CPU memory (a code change) |
| Exit 7, "the base weights were not restored" | the copy could not be written back, or a layer's base weight was replaced during the arm (the merged weights were then not what the model read) | read the record's error; nothing after it ran on the merge; fix and re-brief |
| Load check `fail`, current H differences large, zero small | another checkpoint, or the config's input version wrong | pin the output that `status.json` names (`best.session_start`) |
| Load check `fail`, zero differences large too | another base, dtype, or attention kernel than MT-02's | compare `attn_implementation` with MT-02's `run.json`; the image or the install notebook changed |
| Load check `error` | a missing mount or file | the environment check lists what it found; attach the pinned versions |
| Environment check: "0 distinct folders of the dataset alyahya737/arc26-p3-9-01" (`P3.9-02`) | Kaggle did not attach the pinned dataset version (kaggle-cli 2.2.4's push prints "The following are not valid dataset sources and could not be added to the kernel" and exits 0; stop the session at once) | read the push output and the session log; version 2 is the dataset's latest, so attaching it unversioned mounts the same files: a config change, a new build, and an amended brief |
| Load check `error`, no `best.pt` or `status.json` under the dataset's mount (`P3.9-02`) | version 1 mounted (its four root files only), or `P3.9-01r` not run | the human's `kaggle datasets status alyahya737/arc26-p3-9-01 --format json` (`current_version_number` 2) and the session log's found dataset line; never relaunch before `P3.9-01r` reads Pass |
| A worker fails at start (exit 1, `workers_error`) | H's config not the build's, the cache namespace missing on the mount | rebuild with the right `configs/hypernet.yaml` or input pin |
| Exit 7, `stop-rank<k>.json` | H's path raised on the solver's model: the probe, or an H arm on a worker's first puzzle (an encode under unsloth's patched forward, a swap, a polish from H's adapter) | reproduce the error from the record; the first thing to try is the encoder on the PEFT model without `disable_adapter` (the base arm's zero adapter is loaded then) |
| Many puzzles `cut` or `not_started` | decode slower than priced (the zero adapter and H decode far longer than the SGD adapter) | report the paired subset; reprice `solver_end_s` and `finish_s` from the records |
| `train_log` empty in the records of a training arm | the trainer did not log (another transformers or unsloth version that logs elsewhere) | the scores stand; read `train` seconds only, and check `pip_freeze.txt` against `P3.7-01`'s |
| NaN CE | a non-finite logit under H's adapter | the CE is NaN-propagated in the report; check the bf16 flag and H's weights |
| The hard stop kills the run before the selection | the arms overran $T$ + `finish_s` + 900 s | the records and timing are pulled, the candidates stay in the output (`arm_outputs/`): select them in a follow-up notebook that attaches this output, and raise `solver_kill_s` in the next build |
