# Runbook: `scripts/cache_embeddings.py`

Task P3.2 (BRIEF B2 Q1, B5 P3.2, A8, A9). Configs: `configs/cache_embeddings.yaml` (run
`P3.2-01`, 16 views) and `configs/cache_embeddings_P3.2-02.yaml` (run `P3.2-02`, 8 views and the
view diagnostic); on Kaggle the script notebooks built from `configs/script_P3.2-01.yaml` and
`configs/script_P3.2-02.yaml` run them. Pre-run briefs:
[`docs/prerun/P3.2-01.md`](../prerun/P3.2-01.md), [`docs/prerun/P3.2-02.md`](../prerun/P3.2-02.md). Episode mode (task MT): `configs/cache_embeddings_MT-01.yaml` (run `MT-01`, the meta-training
cache of P3.4's episodes), built from `configs/script_MT-01.yaml`, brief
[`docs/prerun/MT-01.md`](../prerun/MT-01.md).
Result of `P3.2-01`: [`docs/experiments/P3.2-01.md`](../experiments/P3.2-01.md). Module:
[`docs/modules/encoder.md`](../modules/encoder.md).
Decisions: ADR-001 (frozen-base pair encoder), ADR-008 (Kaggle only through
`scripts/kaggle_ops.py`), ADR-010 (the code dataset). "K0" is `docs/experiments/K0.md`, the
measured facts of the Kaggle environment and of the evaluation challenges, with its pulled
`env_probe.json` (per-task pair counts and cell counts, no grids).

## 1. Why this script exists

H reads a task through $\bar{z}$, the mean of its pairs' embeddings from the frozen base model
(ADR-001). Two questions decide whether that encoder is affordable and how later tasks plan
around it:

1. **Time.** B5 P3.2's acceptance: encoding $k = 4$ demonstration pairs under 16 views takes under
   10 s on one L4. At test time this is paid once per task inside the per-puzzle budget (B3: 600 s
   per puzzle per worker), and ADR-001 claims "well under 10 s". From K0's grid sizes of DEV40's
   tasks the claim is close, not safe: about 5 s for the median task, 9.5 s at the 90th
   percentile, and 14 s for the largest grids at 40% of the L4's bf16 peak (estimate; section 6
   of the brief has the derivation).
2. **Size.** Bytes per pair and per task, which size P3.5's meta-training cache and decide whether
   the per-view embeddings are kept.

The run also builds the first cache (DEV40's demonstration pairs at $L = 24$), and it is the
encoder's first contact with the real tokenizer: before the model loads, every view is checked to
decode back to its grids. Results feed the P3.2 acceptance, ADR-001's cost line, the governor's
encoder cost (P2.4, P4.3), and P3.5's cache plan.

`P3.2-01` measured 16 views: Bad, 11 of DEV40's 40 tasks over 10 s, the time linear in tokens at
about 30% of the L4's peak (`docs/experiments/P3.2-01.md`). `P3.2-02` measures the pre-registered
cut, 8 views, which halves every task's tokens; its view diagnostic shows how far each pair's
embedding moves from the 16-view one, because H must be trained and used with one view set. The
cost also sets meta-training's encoding budget (P0.4): about 200k distinct pairs take about 100
L4-hours at 16 views (0.54 pairs per second, `P3.2-01`) and about half at 8.

**Episode mode (`MT-01`, task MT).** Meta-training reads $\bar z(D)$ for every episode it draws.
Without a cache, each training session encodes the build's distinct demonstration pairs before
step 0 (`PairEmbeddings.fill` in `arc26/hyper/train.py` writes no cache), about 7 h of 4xL4 per
session for about 207k pairs at 8 views. `MT-01` encodes them once into one `EmbeddingCache` that
`MT-02` and `MT-03` attach, and it gives the first exact count: P3.4's checker counted 286,532
distinct pairs over every pair of its three splits, queries included
(`runs/P3.4-check2/run_logs/leakage.json`, `checked.unique_pairs`); the training and validation
demonstrations alone were estimated at 205.1k to 211.8k (`docs/prerun/MT-01.md`, section 2).

## 2. What it does, step by step

1. **Config.** Reads the YAML (every key required, none extra) and validates both the GPU values
   and the `smoke` values, so a smoke run also checks the GPU settings, the model handle
   included. `--smoke` applies the `smoke` section; `--input-root`, `--model-dir`,
   `--challenges`, `--cache-dir`, `--out-dir`, and `--device` override the search root, the two
   inputs, the folders, and the device. A path on the command line names the folder or file
   itself and is made absolute, so it is never searched for. `subset` (DEV40) becomes its task
   ids through `arc26.data`. `reference_n_perm`, when set, must exceed `n_perm`, with `n_perm`
   at least 1, so that the run's views are the first rounds of the reference set (step 11).
   Writes the resolved `config.yaml`, ids included, to `out_dir`.
2. **Inputs.** Finds the base model folder and the challenges file under `input_root`
   (`/kaggle/input`), unless the command line names them. A path $p = (p_1, \dots, p_m)$ ends
   with $e = (e_1, \dots, e_r)$ when $r \le m$ and $p_{m-r+j} = e_j$ for $j = 1, \dots, r$, case
   ignored. The model folder is the one holding `config.json` whose path ends with `model_id`
   after its owner (`qwen3_4b_grids15_sft139/transformers/bfloat16/1`; K0 found it under
   `models/sorokin/`); the challenges file is the one whose path ends with `challenges`
   (`arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json`; K0: under `competitions/`).
   The walk is the script notebook's: files at most 8 levels down, package trees skipped, each
   real folder once. No match, or two, stops the run (exit 1) before the model loads; the message
   lists the matches and the other files of that name. `run.json` records what was found.
3. **Tasks.** Reads the challenges JSON and keeps only each task's `train` pairs, in file order,
   for the subset's ids (DEV40, 40 tasks, sorted; every task of the file when `subset` is null).
   It refuses a file whose name contains `solution`, so the evaluation solutions are never opened
   (BRIEF A0).
4. **Views.** The decode path's 16 views: the 8 dihedral transforms in `ArcDataset.augment`'s
   order, each with 2 color permutations drawn from `RandomState(2)` in augment's order. Applied
   with the notebook's `ArcDataset.forward_mod` to both grids of a pair.
5. **Preflight (CPU, before the model loads).** Renders every view of every pair with
   `render_pair`. The text is the notebook's `fmt_train` of the pair. Token spans come from
   tokenizing prefixes: for a character offset $c$ of the pair text $s$, the prefix $s_{<c}$ must
   tokenize to a prefix of $\tau(s)$, and its length $n(c)$ is the token offset. With $c_1, c_2$
   and $c_3, c_4$ the ends of the input and output grid texts, the spans are
   $I = [n(c_1), n(c_2))$ and $O = [n(c_3), n(c_4))$. Each must decode back to its grid through
   the notebook's `convert_tokens_to_array`, and its decoded text, spaces removed, must equal the
   grid's text exactly. A failure stops the run here. The token count of every view goes to
   `metrics.json`.
6. **Model.** Loads the tokenizer and the model from the model folder (`local_files_only`) in bf16
   with SDPA attention, on `device` (`cuda:0`, one L4), frozen. Writes `run.json`.
7. **Encoder.** Per call: renders the views, sorts the sequences longest first, and groups them so
   that rows $b$ times the batch's longest length $T$ satisfy $bT \le$ `batch_tokens`. Rows are
   right-padded and masked. Decoder layers $1, \dots, L$ run; a forward hook on layer $L$ captures
   $h^{(L)}$ and stops the forward. Then, in fp32,
   $$e_{i,a} = \frac{1}{|O_{i,a}|} \sum_{t \in O_{i,a}} h^{(L)}_t, \qquad
   \bar{e}_i = \frac{1}{16} \sum_{a} e_{i,a}, \qquad \bar{z} = \frac{1}{n} \sum_{i} \bar{e}_i .$$
8. **Throughput phase.** One untimed warm-up encode, then for each task (up to
   `throughput_tasks`): a fresh encode of its first $k = 4$ pairs, repeated in order when it has
   fewer, with the cache bypassed. The wall time $t_4$ covers rendering, the prefills, and the
   pooling; the copy of the result to the CPU synchronizes the GPU before the clock stops. One
   line per task goes to `throughput.jsonl`; tasks already there are skipped on a rerun.
9. **Cache phase.** For each task, encodes the pairs whose `pair_hash` the cache lacks (a pair
   shared by two tasks is encoded once) and stores $\bar{e}_i$ and the 16 rows $e_{i,a}$. Every
   `flush_pairs` (256) pairs, and at the end or on a stop, one shard and the manifest are written
   atomically. A rerun encodes only what is missing.
10. **Stop and report.** Between tasks the script checks the wall-time criterion. In every case,
    including failures, it writes `run.json`, `metrics.json`, and `summary.md`, and `error.log`
    on a failure.
11. **View diagnostic** (`reference_n_perm` set; `P3.2-02`). It decides nothing. With the run's
    views $R$ (`view_descriptors(n_perm, view_seed)`) and the reference set $V$
    (`view_descriptors(reference_n_perm, view_seed)`), `view_descriptors` draws round by round,
    so $R$ is the first $|R|$ views of $V$. For each distinct pair $i$ the run's cached
    $\bar{e}_R(i)$ and the views $C = V \setminus R$, encoded now, give
    $$\bar{e}_V(i) = \frac{|R|\,\bar{e}_R(i) + \sum_{c \in C} e_{i,c}}{|V|},$$
    the embedding under $V$, at the cost of $|C|$ views instead of $|V|$. Reported, each as the
    median, 10th percentile, and minimum: $\cos(\bar{e}_R(i), \bar{e}_V(i))$ over the pairs; the
    same after subtracting $\mu = \frac{1}{n}\sum_i \bar{e}_V(i)$ from both; and
    $\cos(\bar{z}_R(\tau), \bar{z}_V(\tau))$ over the tasks. As the scale, the median cosine between
    different pairs' $\bar{e}_V$ (over at most 2,000 pairs). `view_diagnostic.jsonl` holds each
    pair's two cosines.

**Cost model.** A view of $T$ tokens costs about
$$F(T) \approx 2 P_L T + 4 L\, d_{\text{attn}}\, T^2$$
FLOPs, where $P_L$ is the parameter count of the first $L$ layers and $d_{\text{attn}}$ the query
width. For Qwen3-4B (`docs/research/d2l_t2l_reading.md`: hidden 2560, 32 query heads and 8
key-value heads of size 128, MLP 9728) one layer has
$2560 \cdot 4096 + 2 \cdot 2560 \cdot 1024 + 4096 \cdot 2560 + 3 \cdot 2560 \cdot 9728 \approx
100.9\text{M}$ parameters, so $P_{24} \approx 2.42 \times 10^9$, and $d_{\text{attn}} = 4096$. A
task's time is $t_4 \approx \sum_{\text{views}} F(T) / (\eta \cdot 121 \times 10^{12})$ at a
fraction $\eta$ of the L4's dense bf16 peak; at $\eta = 0.4$ that is about 0.10 s per 1,000
tokens plus the attention term.

### Episode mode (`MT-01`)

A config with an `episodes` section runs this way instead of steps 3 to 11 above.

1. **Config.** Its own exact keys: the encoder and run keys of challenge mode, `episodes`
   (`root`, `manifest_sha256`, `check_root`, `splits`, `max_tokens`), `gpus`, `chunk_pairs`,
   `seed_cache`, `verify_pairs`, `verify_min_cosine`, and a smoke section; none of the challenge
   keys. `device` is `cuda` (rank $k$ uses `cuda:k`) or `cpu`. `--episodes-root`, `--check-root`,
   and `--seed-cache` name folders on the command line (the notebook passes the mounts).
2. **Inputs.** The model folder by search, as above. The episode build through
   `arc26.hyper.train.open_episode_set` and `load_split`, exactly as training reads it: the
   manifest's sha256, a passing leakage report for it, every shard's size and sha256, every
   record's schema and the token cap.
3. **Pairs and the exact count.** Every demonstration $(x, y)$ of every episode of the splits,
   keyed by `pair_hash`, the first occurrence kept: the cache's target set $P$. Per split the
   episodes, demonstration occurrences, and distinct pairs, and per source the distinct pairs, go
   to the log, `metrics.json` `pairs`, and `summary.md`.
4. **Preflight.** A seeded sample of 64 pairs and the pair with the most cells, every view, through
   `render_pair` under the real tokenizer, before any model loads; every other pair is checked as
   it is rendered for its encode.
5. **Cache and seed.** The cache of the encoder's namespace opens in `cache_dir`. With
   `--seed-cache`, an empty namespace first receives the earlier session's files (its index, its
   shards, its part files), copied as contents only, so a read-only mount's permissions stay
   behind. Part files a cut session left are merged.
6. **Chunks.** The pairs of $P$ the cache lacks, in sorted hash order, cut into chunks of
   `chunk_pairs`; chunk $j$ goes to rank $j \bmod W$, $W = \min(\texttt{gpus}, \text{chunks})$.
7. **Encoding processes.** $W$ copies of this script (`--rank-job`), one per GPU, each loading the
   base straight onto its GPU (`device_map`, as training's `load_base` does) and encoding its
   chunks with `PairEncoder.encode_pairs`, which gives
   $$\bar e_i = \frac{1}{8} \sum_{a} \frac{1}{|O_{i,a}|} \sum_{t \in O_{i,a}} h^{(24)}_t$$
   in fp32 for the 8 views $a$. Each chunk becomes one part file
   (`parts/session-<t>/rank-<k>/part-<j>.safetensors`, the namespace in its metadata) by an
   atomic rename. Before each chunk the process checks the deadline (the script's start plus
   `max_wall_s`) and stops there if it has passed. Each writes `timing-rank<k>.jsonl` (one
   `encode` record per chunk) and `encode-rank<k>.json` (its summary).
8. **Merge.** This process, the cache's only writer, moves finished part files into the cache every
   5 minutes and once the processes end, in puts of `flush_pairs` pairs (one shard and one index
   rewrite each), deleting a part only after its put.
9. **Check.** A seeded sample of `verify_pairs` cached pairs of $P$ is encoded again, in chunks of
   other pairs than when it was cached, on `cuda:0`; with $\hat e_i$ the fresh embedding,
   $\cos(\hat e_i, \bar e_i) = \hat e_i^\top \bar e_i / (\lVert \hat e_i \rVert \lVert \bar e_i \rVert)$
   must be at least `verify_min_cosine` for every pair.
10. **Report.** `run.json`, `metrics.json`, `summary.md` in every case. Exit 0 when every pair of
    $P$ is cached and the check passed, 3 when the deadline stopped the run with pairs missing, 1
    on any failure, a failed check included.

## 3. Why it is built this way

**Frozen base, pooled hidden states, no trained encoder.** ADR-001. Lost: a raster encoder over
one-hot grids trained with H (no grid prior, a second model to train in five weeks; kept as the
P3.7 fallback); attention pooling or a set transformer (parameters to overfit; ablations); D2L's
per-token context read (longer H inputs, no measured need). Mean pooling over pairs makes
$\bar{z}$ order invariant by construction.

**$L = 24$ of 36, the residual stream.** BRIEF B2 Q1 and ADR-001; P3.7 ablates 18, 24, 30. The
encoder takes Hugging Face's `hidden_states[L]`. The last entry of that tuple is after the final
norm, not the residual stream, so $L = 36$ is refused rather than special-cased.

**Early exit through a forward hook.** Lost: (a) a full forward with `output_hidden_states=True`,
which runs all 36 layers and the head, 1.5 times the layer FLOPs of 24 layers (estimate: equal
cost per layer); (b) loading a truncated checkpoint (`num_hidden_layers=24`), which saves 2.4 GB
of weights but cannot share the loaded base with test-time training and decoding (P4.1); (c) a
reimplemented decoder loop, which depends on transformers internals (mask creation, the layer
call signature) that differ between 4.55 on the pinned image and 5.0 on the latest image (K0).
The hook depends only on the module structure. Measured on CPU: the captured tensor equals the
full forward's `hidden_states[L]` bitwise for every $L$ of the test model, and no later layer,
norm, or head runs. Output recording is passed as off because transformers 4.55's
`check_model_inputs` restores the forwards it patches only on a normal return.

**Spans from prefix tokenization, then checked.** The task brief rules out assuming one token per
character. Lost: the notebook collator's rule (`QwenDataCollatorForCompletionOnlyLM`: the output
starts two tokens after the `assistant` token), which assumes the role word and the newline are
one token each; offset mappings, which need a fast tokenizer and whose byte-level offsets around
special tokens differ between tokenizers. Prefix tokenization needs only `encode`. Two decode
checks follow: the notebook's own `convert_tokens_to_array`, which reads a span the way the decode
path reads generated grids, and an exact comparison of the span's text with the grid's text,
because the notebook's decoder strips whitespace and skips non-digits. A tokenizer whose
normalizer drops a trailing newline makes the prefix that ends after `user\n` one token short, so
both spans start on the newline; the notebook's check passes that, the exact one does not (a test
reproduces it). Cost, measured on CPU: about 4 ms per $30 \times 30$ view (five encodes, four
decodes); 6% of the smoke run's encode time.

**The decode path's 16 views, from a local generator.** Lost: calling `ArcDataset.augment`
(it reseeds numpy's global generator, changing every other consumer's random stream, and its
`shuffle_ex` has no meaning for one pair); per-pair random views (the cache key would need the
views of each entry; one view set per namespace keeps it content-addressed); 8 views
(`n_perm: 1`, one permutation per transform; or `n_perm: 0`, none), half the cost with less
averaging over colors, which is the fallback if this run is Bad. With seed 2 the views are the
decode path's for a task with one test input: on Kaggle, run A of `P0.1-01` decoded exactly these
16 views for `981571dc` and `aa4ec2a5` (a test checks this against the committed fixture).

**fp32 pooling.** bf16 keeps 8 significant bits, so a bf16 running sum over up to 929 output
tokens would lose precision at every addition; an fp32 sum costs one cast of the span. All later
means are fp32 too.

**Token-budget batches, longest first, right padding.** Lost: a fixed batch size (pairs range
from 10 to 1,866 tokens, so it either runs out of memory on $30 \times 30$ pairs or leaves the GPU
idle on small ones); left padding (Hugging Face's default position ids start at 0 in every row,
so left padding would move every real token; right padding keeps each one's position and
attention context as in an unbatched run); packing with block-diagonal masks (needs
FlashAttention's variable-length kernels; after sorting, padding is already small). Longest first
makes an out-of-memory error appear on the first batch. Measured on CPU: batched equals one at a
time within $2.6 \times 10^{-8}$; with every padded position set to NaN, every embedding is
unchanged.

**`batch_tokens` 16384.** Estimate: one layer's activations at 16,384 tokens (the MLP's three
$16384 \times 9728$ bf16 tensors dominate) are about 1.3 GB beside 7.3 GB of weights on a 22 GB L4,
and GEMMs with 16k rows run near their peak rate. Lost: 4096 (smaller GEMMs, four times the
launches); 65536 (about 5 GB of activations, no gain once GEMMs are large). This run measures the
result; section 7 says what to change.

**bf16 and SDPA.** bf16 is the Kaggle model variant and the notebook's runtime precision; fp32
doubles memory and time. SDPA ships with torch and serves the padding mask with its
memory-efficient kernel. Lost: eager attention (materializes $[b, 32, T, T]$ scores, 2 to 4 GB
for 8 rows of 2k tokens in bf16 with the fp32 softmax, estimate); FlashAttention 2 (needs the
flash-attn wheel from the install notebook; the first change to try if prefill is slow,
section 6).

**Content-addressed cache: one namespace per model id, layer, pool, and view set; safetensors
shards and a JSON manifest; fp32.** Lost: keys by task id and pair index (wrong when a pair occurs
in two tasks or under an augmentation; P3.4's leakage check needs a content hash anyway); one
file per pair (hundreds of thousands of files at meta-training scale); one growing file
(rewritten on every write); `torch.save` pickles (loading runs code; safetensors loads only
tensors and is already in the stack); SQLite or LMDB (a new dependency to store blobs); bf16
storage (half the size, but it rounds the fp32 means, so a round trip would not be exact).
Writes go to a temporary file, `fsync`, and `os.replace`, and the manifest goes last, so a crash
leaves at most an unreferenced shard; a test fails the manifest write and checks this. Per-view
rows are kept here (`store_views: true`, about 21 MB for DEV40) so P3.7 can average any subset of
views without a GPU; a meta-training cache should turn them off (17 times the size).

**Throughput measured fresh, at $k = 4$ by repeating pairs, after a warm-up.** Lost: measuring
only tasks with at least 4 pairs (24 of the 120 evaluation tasks, K0, so about 8 in DEV40: too
few for a 90th percentile); extrapolating from each task's own pair count (assumes cost linear in
pairs; untested). Repeating keeps each task's real grid sizes. The warm-up keeps CUDA context
creation and kernel selection out of the first task's time.

**Separate throughput and cache phases.** Lost: reusing the timed encodes to fill the cache
(saves about 2.5 minutes of L4 time on DEV40, estimate), because a resumed run would then time
fewer pairs, so the measurement would depend on cache state.

**CPU preflight before the model loads.** The real tokenizer is this run's main unknown; a
format problem then fails in seconds instead of after an 8 GB model load.

**Inputs found by the ends of their paths, by the script.** `docs/STATE.md` (rules for scripts)
forbids hardcoded `/kaggle/input` paths: the mounts differ between 2025 (the reference notebook
read `/kaggle/input/arc-prize-2025/` and `/kaggle/input/qwen3_4b_grids15_sft139/...`) and 2026
(K0), and the code dataset moved within 2026 (`docs/runbooks/build_submission.md`). Lost: (a) the
K0 paths in the config, which one more mount change would break; (b) the notebook passing the
paths its environment check found, as `P3.5-01`'s notebook passes the folders of its attached
outputs: the builder fills only `{commit}` and `{input:<owner>/<slug>}` (another notebook's
output), so the model and the competition would need new placeholders in the builder that every
build shares, and an argument would override the config, which would then no longer be the one
source of the setting; (c) matching the model's folder name only, as the environment check does
with `model_slug`: another version of the model would match too, and the cache key `model_id`
could then name weights other than those loaded, whereas matching the handle's ending makes
`model_id` the one setting for both; (d) importing the builder's `find_inputs`: script mode ships
only the script and its config, and a 30-line walk does not justify a new wheel module. Two
matches are refused rather than the first taken, as the notebook refuses two folders of one
attached output: a guess could read a stale copy.

**DEV40 by name.** `arc26.data` defines DEV40 once and freezes it; `tests/test_data.py` recomputes
it from the K0 statistics, and every DEV40 score uses it. The wheel installed on Kaggle carries
it, so naming the subset keeps one source of truth, and the tests check that the config resolves
to it. Lost: the 40 ids listed in the config, a second copy that only a test could keep equal,
and that a reader could not tell from a hand-picked list.

**One L4 of `NvidiaL4`.** The question is a one-L4 time (B5 P3.2), and each of the scored rerun's
four workers owns one L4 (BRIEF B1, B3), so the script runs on `cuda:0`. Lost: `NvidiaL4X1`,
which Kaggle ran on its default 2xT4 (`docs/experiments/P0.3-01.md`), and T4 has no bf16;
spreading the cache phase over the four L4s, which would save about 2 minutes (estimate, brief
section 6) at the price of a second code path, while the throughput phase must stay on one GPU.
The script notebook's bf16 stop refuses substituted T4s before the install.

**A script notebook.** The run reuses the submission notebook's stack and plumbing (P0.5d,
`docs/runbooks/build_submission.md`, script mode): run A's pinned image, the install notebook at
version 2 (torch 2.8.0, transformers 4.55.4), the wheel installed offline, inputs found by
search, the hard stop, and `run_logs/` for the pull. The build ships this script and its config
beside the wheel and the notebook checks both by sha256, so the run executes the reviewed commit.
The alternatives (a builder per script, `!python` cells) and why they lost are in that runbook.

**8 views: the first round of the 16, under the same seed** (`P3.2-02`). `P3.2-01`'s time was
linear in tokens ($R^2 = 0.974$) with a flat cost per 1,000 tokens from 800 to 1,866 tokens per
view, so there is no attention term for FlashAttention 2 to remove, and halving the views halves
the tokens (`docs/experiments/P3.2-01.md`; section 6, item 4). With `view_seed` 2 and `n_perm` 1
the 8 views are the first 8 of `P3.2-01`'s 16, one color permutation per dihedral transform,
views the decode path itself uses. That makes the view diagnostic exact at half the cost (step
11) and lets `P3.2-01`'s cache, which holds the 16 per-view rows of DEV40's pairs, give these
views' embeddings without a GPU. Lost: another seed (its 8 permutations lie outside the 16, so
the diagnostic would encode all 16 views and `P3.2-01`'s rows would not serve); `n_perm` 0, the
8 dihedral views in the original colors (outside the 16 too, and no averaging over colors); a
lower $L$ (P3.7's ablation, a larger change of the embedding than dropping views).

**The view diagnostic in the run, from the other 8 views.** Lost: (a) attaching `P3.2-01`'s output
and reading its 16 per-view rows, which saves about 2 minutes of one L4 but needs a notebook of
another slug (Kaggle refuses a notebook's own output), a mounted notebook output, and code to
read a second cache; (b) a separate CPU notebook on that output, a fourth build and push for 117
cosines; (c) encoding all 16 views in the cache phase, after which the cache phase would no
longer measure the 8-view rate that meta-training's budget needs. The centered cosine is
reported beside the raw one because mean-pooled hidden states share a large common direction,
which lifts every raw cosine: on the CPU stand-in, different pairs already have a median raw
cosine of 0.94.

**Kill criteria in the script, resumable state.** `docs/STATE.md` (rules for this week's briefs)
requires every GPU script to enforce its own kill criteria, since nothing can cancel a running
kernel; the notebook's hard stop and Kaggle's `Timeout` are the outer bounds. A8 requires resuming
for runs over 30 minutes: DEV40 takes about 10 minutes (estimate), but the same script can build
larger caches, so both phases resume in the same folders. On Kaggle a stopped `P3.2-01` is rerun
from the start under an amended brief instead: a resume there needs the previous output attached
to a notebook of another slug (Kaggle refuses a notebook's own output) and copied into
`/kaggle/working` first, which costs more than the 10 minutes it would save.

### Episode mode (`MT-01`)

**A mode of this script.** The encoder, the cache, the input search, and the smoke stand-ins are
this script's, and script mode ships one file. Lost: (a) a new script, which would copy them;
(b) training's own pass (`PairEmbeddings.fill`), which writes no cache, so each training session
would encode again, about 7 h of its 11.5; (c) a CPU notebook writing the pairs as a challenges
file for one-GPU challenge mode: 28 to 33 GPU-hours on one L4 is three sessions, and a second
file to keep equal to the build.

**Processes with part files, one writer.** `EmbeddingCache` has one writer by design: its index is
rewritten whole on every put, so two writers lose each other's entries. Each process therefore
writes only its own part files, which need no collective, cost nothing to merge, and survive a
kill; this process merges them. Lost: (a) every process writing the cache (lost entries);
(b) gathering every chunk to rank 0 over NCCL, a process group for no other use, which also
loses what rank 0 holds unwritten at a kill; (c) `torch.multiprocessing.spawn`, whose children
import the target by module name, which a script loaded from its path (the tests) cannot
provide; copies of this script started with its own path work for the notebook and the tests.

**Chunks of 256 pairs in hash order.** One encode call per 256 pairs (2,048 prefills) pays the
call's fixed cost (0.54 s in `P3.2-01`'s fit) about 810 times; one call per task would pay it for
51,048 tasks, most of them BARC tasks of about 2.6 pairs (about 7.6 GPU-hours, estimate). Hash
order makes the chunks independent of tasks and splits, as training's own pass cuts them. Lost:
larger chunks (more work lost per process at a kill); per-task calls.

**A seed by copy.** Kaggle mounts an earlier notebook's output read-only and refuses a notebook's
own output, so a continuation is another notebook. Copying the namespace keeps one cache that
`MT-02` attaches whole. Lost: reading two caches in layers, which `EmbeddingCache` and training
(one `embedding_cache` root) do not support.

**The check.** Counts pass even when an embedding sits under another pair's hash (a merge or index
error) or a shard is corrupt; a fresh encode of a seeded sample, batched with other pairs,
catches both. The bound 0.999 sits far above two different pairs (median cosine 0.55, `P3.2-02`)
and below what batching can move a bf16 embedding (the 8-to-16-view change is 0.9983 at worst).
Lost: no check; re-encoding every pair (twice the cost); hashes alone (they test the keys, not
the values).

**Means only** (`store_views: false`): per-view rows would make an entry nine times larger, about
19 GB for about 207k pairs, near Kaggle's working limit. **SDPA**, as `P3.2-02` measured it.

**Weights straight onto each GPU** (`device_map` on CUDA). Loading on the CPU first, as challenge
mode's single process does, would stage about 8 GB of bf16 weights in host memory per process,
about 32 GB at once for four, on a machine whose memory no run has measured. The run records the
machine's `MemTotal` and its container's cgroup limit (`run.json` `host`), so `MT-02`, whose four
processes each hold the whole embedding table, can be checked against the smaller of the two.

**Kill criteria and resumable state.** The deadline is checked before each chunk, and each finished
chunk is on disk, so a stop loses nothing finished; the notebook's hard stop and Kaggle's
`Timeout` are the outer bounds.

## 4. How to run it

**CPU smoke (local, required before any GPU run):**

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python scripts/cache_embeddings.py \
    --config configs/cache_embeddings.yaml --smoke
```

It builds the tiny random Qwen3 (hidden 64, 4 layers) and the stand-in tokenizer in code, writes
three synthetic tasks (2, 3, and 4 pairs; one $30 \times 30$ pair) to
`runs/P3.2-smoke/smoke_challenges.json`, and runs every stage but the input search. Measured at
`P3.2-01`'s `Code-commit` (brief, section 11): 6.6 s wall, 0.99 GB peak RSS, exit 0.
`tests/test_cache_embeddings.py` runs it.

**The GPU run's own path on the CPU (local).** Two tests run `P3.2-01`'s settings as committed
(the input search, DEV40, $L = 24$, bf16, SDPA) against a Kaggle-like input tree holding a
36-layer copy of the tiny Qwen3 and stand-in challenges of DEV40's 40 ids, with only the device
and the folders given on the command line. The first calls the script in-process:
`tests/test_cache_embeddings.py::test_the_gpu_runs_settings_run_on_cpu_from_a_kaggle_layout`.
The second,
`tests/test_notebook_local.py::test_the_p3_2_01_build_runs_cache_embeddings_in_the_simulated_tree`,
builds the notebook from `configs/script_P3.2-01.yaml` and runs it against a simulated `/kaggle`
tree with four L4s reported, so the environment check, the install, and the script's command
line are the ones Kaggle will run. Both take seconds.

**GPU test (CUDA machines only; never on the dev box, BRIEF A0).**
`tests/test_encoder.py::test_bf16_sdpa_on_cuda_matches_the_full_forward_and_batches` builds the
tiny Qwen3 in bf16 with SDPA on `cuda:0`, checks that the captured layer equals the full
forward's `hidden_states[2]` bitwise, and that batched encoding equals one sequence at a time
within bf16 tolerance. It downloads nothing and takes seconds. On a plain GPU machine, from the
repository root:

```bash
HF_HUB_OFFLINE=1 .venv/bin/pytest -m gpu -q \
    "tests/test_encoder.py::test_bf16_sdpa_on_cuda_matches_the_full_forward_and_batches"
```

In a Kaggle notebook cell, with the repository's `pyproject.toml` (it registers the `gpu`
marker), `scripts/`, and `tests/` next to the cell, `arc26` installed, and pytest from the image
or the attached wheels (`-p no:cacheprovider` because a checkout under `/kaggle/input` is
read-only):

```text
!HF_HUB_OFFLINE=1 python -m pytest -m gpu -q -p no:cacheprovider tests/test_encoder.py::test_bf16_sdpa_on_cuda_matches_the_full_forward_and_batches
```

Expected: `1 passed`. `1 skipped` means CUDA was not visible, so nothing was checked. On Kaggle it
runs only in a notebook whose approved brief names it (BRIEF A9).

**Kaggle (`P3.2-02`), after the human approves the brief.** As `P3.2-01` below, with
`configs/script_P3.2-02.yaml` (the same notebook `alyahya737/arc26-p3-2-encoder`, its next
version) and the script's config `cache_embeddings_P3.2-02.yaml`; the code dataset needs a version
made from this build (or one shared with a later build carrying the same wheel and this script
and config). Smoke: `CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python
scripts/cache_embeddings.py --config configs/cache_embeddings_P3.2-02.yaml --smoke`.

```bash
.venv/bin/python scripts/build_submission.py --tag P3.2-02 --config configs/script_P3.2-02.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P3.2-02/arc26-code -m "P3.2-02 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P3.2-02 --run-id P3.2-02
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p3-2-encoder
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p3-2-encoder --task-id P3.2-02
```

**Kaggle (`P3.2-01`), after the human approves the brief.** The script runs in a script notebook
(P0.5d, `docs/runbooks/build_submission.md`, script mode) built from `configs/script_P3.2-01.yaml`:
notebook `alyahya737/arc26-p3-2-encoder` on `NvidiaL4`, internet off, run A's pinned image, the
install notebook at version 2, the competition `arc-prize-2026-arc-agi-2`, the model
`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`, and the code dataset
`alyahya737/arc26-code` (wheel, this script, `cache_embeddings.yaml`). The code dataset needs a
version made from this build, after `P3.5-01` has started (whether a queued or running notebook
keeps its dataset version is unverified), and no other version while this run waits or runs. The
orchestrator, from the repository root with a clean tree at the brief's `Code-commit`:

```bash
.venv/bin/python scripts/build_submission.py --tag P3.2-01 --config configs/script_P3.2-01.yaml
.venv/bin/python scripts/kaggle_ops.py dataset-version dist/P3.2-01/arc26-code -m "P3.2-01 <commit>"
.venv/bin/python scripts/kaggle_ops.py push dist/P3.2-01 --run-id P3.2-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-p3-2-encoder
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-p3-2-encoder --task-id P3.2-01
```

Wait a few minutes after `dataset-version` (Kaggle processes the version) before the push. The
notebook checks the environment (bf16 on every GPU, the inputs found, the script and its config
beside the wheel with the build's sha256), installs the wheel, and runs
`python cache_embeddings.py --config cache_embeddings.yaml --commit <sha>` in `/kaggle/working`,
which finds the model and the challenges under `/kaggle/input` itself. The script stops itself at
`max_wall_s` 1800 s; the notebook kills it at `script_kill_s` 2400 s after its start, and the
brief's `Timeout` is 2700 s. If `NvidiaL4` waits in the queue for 30 minutes, the orchestrator
tells the human; there is no smaller fallback (`NvidiaL4X1` runs on 2xT4, which has no bf16).
`pull` brings `run_logs/` back to `runs/P3.2-01/`; the cache stays in the notebook's output
(`/kaggle/working/embedding_cache/`) and reaches later notebooks as an attached input.

**After a stop on Kaggle** (exit 3, or the session lost): report it, and rerun from the start under
an amended, re-approved brief (section 3, kill criteria). Locally, or on any machine that keeps
`out_dir` and `cache_dir`, the same command resumes: measured tasks and cached pairs are skipped.

**Plain GPU machine (bf16-capable GPU, CUDA):** name the inputs, or point `--input-root` at a
folder laid out as Kaggle mounts them:

```bash
uv venv --python 3.11 .venv && uv pip install -e ".[dev]" torch==2.8.0 transformers==4.55.4 \
    safetensors==0.6.2
.venv/bin/python scripts/cache_embeddings.py --config configs/cache_embeddings.yaml \
    --model-dir /data/qwen3_4b_grids15_sft139 \
    --challenges /data/arc-agi_evaluation_challenges.json \
    --cache-dir /data/embedding_cache --out-dir runs/P3.2-01 --device cuda:0
```

**Episode mode, CPU smoke** (two encoding processes on the episode fixture of
`tests/testdata/train_hypernet/`, the check on 8 pairs):

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python scripts/cache_embeddings.py \
    --config configs/cache_embeddings_MT-01.yaml --smoke
```

Measured at the brief's `Code-commit`: 6.8 s, 0.87 GB peak resident memory, exit 0, 21 distinct
pairs cached, the check's minimum cosine 1.0. Tests: `tests/test_cache_embeddings_episodes.py`
(cut sessions, leftover parts, seeds and their temporary files, a misfiled embedding caught by the
check, a part of another namespace refused, a part deleted only after its put) and
`tests/test_notebook_local_mt.py` (the build in the simulated tree on the merged code dataset, its
cache read by `MT-02`'s).

**Kaggle (`MT-01`), after the human approves the brief.** The build of
`configs/script_MT-01.yaml`: notebook `alyahya737/arc26-mt-cache` on `NvidiaL4`, P3.4's two
outputs attached at version 2, and the code dataset `alyahya737/arc26-code-mt`, the MT runs' own,
created once for all three runs from the merged folder `dist/MT-code/`: `MT-01`'s `arc26-code/`
files (the wheel, `cache_embeddings.py`, `cache_embeddings_MT-01.yaml`, and its
`dataset-metadata.json`, which names the dataset) plus `MT-02`'s script and config
(`train_hypernet.py`, `train_hypernet_MT.yaml`; `MT-03` uses the same two). Each notebook takes its
own script and config by name beside the wheel and checks their sha256. Not the
`dataset-create dist/MT-01/arc26-code` line of `dist/MT-01/MANIFEST.md`, which the builder writes
for one build: that version would lack `train_hypernet.py`, and `MT-02`'s and `MT-03`'s notebooks
would stop in their environment check. The orchestrator, with a clean tree at the brief's
`Code-commit`:

```bash
.venv/bin/python scripts/build_submission.py --tag MT-01 --config configs/script_MT-01.yaml
.venv/bin/python scripts/build_submission.py --tag MT-02 --config configs/script_MT-02.yaml
mkdir dist/MT-code
cp dist/MT-01/arc26-code/* dist/MT-code/
cp dist/MT-02/arc26-code/train_hypernet.py dist/MT-02/arc26-code/train_hypernet_MT.yaml dist/MT-code/
.venv/bin/python scripts/kaggle_ops.py dataset-create dist/MT-code
.venv/bin/python scripts/kaggle_ops.py push dist/MT-01 --run-id MT-01
.venv/bin/python scripts/kaggle_ops.py status alyahya737/arc26-mt-cache
.venv/bin/python scripts/kaggle_ops.py pull alyahya737/arc26-mt-cache --task-id MT-01
```

Wait a few minutes after `dataset-create` (Kaggle processes the version) before the push.
`tests/test_notebook_local_mt.py` runs the MT notebooks against this merged folder. Version 2
(`MT-02` revision 2, 4 Oct) replaces only `train_hypernet.py` with the one that reads
`--max-wall-s` (`docs/runbooks/train_hypernet.md`, section 4); `MT-02` to `MT-04` run on it.

The notebook runs `python cache_embeddings.py --config cache_embeddings_MT-01.yaml --commit <sha>
--episodes-root <episodes mount> --check-root <checker mount>` in `/kaggle/working`; the cache
stays in the output (`/kaggle/working/embedding_cache/`), which `MT-02` attaches as
`alyahya737/arc26-mt-cache` version 1.

**A continuation of `MT-01`** (exit 3 with more pairs missing than `MT-02`'s brief accepts): a
notebook of another slug whose build attaches this one's output and adds
`--seed-cache {input:alyahya737/arc26-mt-cache}/embedding_cache` to the arguments; it copies the
cache, merges any part files, and encodes the rest. It needs its own build config and brief,
and `MT-02`'s config then attaches its output instead.

**Episode mode on a plain GPU machine** (four bf16 GPUs; the build and its report in local
folders):

```bash
.venv/bin/python scripts/cache_embeddings.py --config configs/cache_embeddings_MT-01.yaml \
    --model-dir /data/qwen3_4b_grids15_sft139 --episodes-root /data/arc26-p3-4-episodes \
    --check-root /data/arc26-p3-4-leakage --cache-dir /data/embedding_cache \
    --out-dir runs/MT-01/run_logs
```

**Expected cost of `MT-01`** (brief, section 6): 7.1 to 8.4 h of session on 4xL4 for 632M to 652M
prefill tokens (205.1k to 211.8k pairs times 8 views), 0.157 to 0.181 s per 1,000 tokens per L4;
about 8.7 GB of each L4 (as `P3.2-02`); about 2.1 GB of cache.

**Expected cost** (estimates, brief section 6): about 8 to 12 minutes of session wall time for
DEV40 (environment check and install about 1 min, model load 1 to 2 min, preflight seconds,
throughput 3 to 5 min, cache 2 to 3.5 min); GPU memory under 10 GB of one L4 (about 7.3 GB of bf16
weights for 3.63B non-embedding parameters, plus 0.8 GB if the vocabulary is not the 16 tokens
the brief assumes; about 1.5 GB of activations and buffers); disk about 20.4 MB of cache and under
1 MB of logs.

## 5. Outputs

All in `out_dir` (`/kaggle/working/run_logs/` on Kaggle, pulled to `runs/P3.2-01/`):

| File | Contents |
|---|---|
| `config.yaml` | The resolved config: file values, `smoke` section if used, command-line overrides, and the subset's `task_ids`. |
| `run.json` | Start time, argv, commit and dirty flag (`git`, or `--commit`), Python, torch, transformers, safetensors, numpy versions, device and GPU name, `inputs` (the model folder and the challenges file found or named), model facts (layers, hidden size, vocabulary, dtype, load seconds), encoder settings with the 16 view descriptors and the view-set id, cache directory and entries at start. Written once the model is loaded and again at the end, however the run ends. |
| `timing.jsonl` | One line per encode stage: `{puzzle_id, op, seconds, tokens, phase}`, `op` in `encode_render`, `encode_prefill`; `phase` in `warmup`, `throughput`, `cache`. Appended across sessions. |
| `throughput.jsonl` | One line per measured task: `task_id`, `k`, `n_pairs`, `cycled` (pairs repeated), `sequences` (64), `tokens`, `seconds` ($t_4$), `render_seconds`, `prefill_seconds`, `device`. |
| `metrics.json` | `status` (`complete`, `stopped: ...`, `failed: ...`), `elapsed_s`; `tasks` (count, pairs, distinct pairs); `lengths` (tokens per view: p50, p90, p99, max); `model`; `throughput` (median, p90, max, mean of $t_4$ and of tokens, seconds per 1,000 tokens, render share, slowest task, tasks over `acceptance_s`, tasks measured this session); `cache` (entries, run pairs cached, pairs encoded this session, seconds, pairs and tokens per second, bytes, bytes per pair on disk and of tensor data, bytes per task); `memory` (CUDA peak allocated and reserved, weights included). Sections the run did not reach are absent. |
| `summary.md` | The same, readable, with the $t_4$ table. |
| `error.log` | The traceback, only on a failure. |
| `view_diagnostic.jsonl` | With `reference_n_perm` set: one line per distinct pair, `pair` (its hash), `task_id` (the first task holding it), `cosine` and `centered_cosine` of its run-view and reference-view embeddings (step 11). `metrics.json` adds `view_diagnostic` (views, reference views, pairs, tasks, the three spreads, the between-pairs median and its sample, seconds, tokens) and `run.json` the reference set's id and the views encoded. |
| `smoke_challenges.json` | Smoke runs only: the synthetic tasks. |

On Kaggle the script notebook adds its own files to `run_logs/` (`stages.json` with `ok`, the
script's command, exit code, and whether the hard stop killed it; `env.json`; `pip_freeze.txt`;
`candidates.json`, `{}`), and the session log holds the script's printed lines, the inputs found
among them (`docs/runbooks/build_submission.md`, script mode).

In episode mode `out_dir` holds `config.yaml` (the resolved config, `episodes` included),
`run.json` (provenance, `inputs`: the model folder, the build folder, its manifest, the check
report; `encoder`; `cache`: its folder, entries at the start, the seed; `host`: the machine's CPUs,
its `MemTotal`, the container's cgroup memory limit and the file it came from, and the smaller of
the two as `memory_limit_bytes`), `metrics.json` (`mode`
`episodes`; `episodes` with per-split counts; `pairs`: `distinct`, `by_source`,
`cached_at_start`, `seeded`, `leftover_parts`, `to_encode`, `missing`; `encode`: processes,
chunks, pairs, tokens, GPU-seconds, seconds per 1,000 tokens, each process's summary; `cache`;
`check`), `summary.md`, `timing-rank<k>.jsonl` (`{puzzle_id: null, op: encode, seconds, tokens,
pairs, chunk}` per chunk), `encode-rank<k>.json` (each process's status, chunks, pairs, tokens,
seconds, peak memory), and `error.log` or `error-rank<k>.log` on a failure. The cache folder may
hold `parts/` while a run is cut; a finished run leaves it empty.

The cache, in `cache_dir` (`/kaggle/working/embedding_cache/`, stays on Kaggle):
`<pool>-L<layer>-<digest>/namespace.json`, `manifest.json`, and `shard-NNNNNN.safetensors`
(tensors `<pair hash>`, `[2560]`, and `<pair hash>/views`, `[16, 2560]`, fp32).

## 6. How to read the results

1. **`status`** must be `complete`, and on Kaggle `stages.json` `ok` true. Then check `run.json`:
   the GPU name (an L4), torch 2.8.0 and transformers 4.55.4, bf16, 36 layers, hidden size 2560,
   vocabulary 16 (assumed by the brief; another value changes nothing in this run), and `inputs`:
   a model folder ending in `qwen3_4b_grids15_sft139/transformers/bfloat16/1` and the challenges
   file in `arc-prize-2026-arc-agi-2/`. `metrics.json` `tasks.count` must be 40 (DEV40).
2. **Preflight.** `lengths` exists only if every view passed the span checks. Its maximum should
   be at most 1,866 tokens, the $30 \times 30$ pair under the notebook vocabulary.
3. **$t_4$**, against the brief's decision rule (pre-registered there, section 1), read from the
   count of tasks over 10 s (`metrics.json` `throughput`, `summary.md` column "Over 10 s"); the
   median, the 90th percentile, and the maximum are reported, not decided on:
   - *Good*, no task above 10 s: P3.2 is accepted, the defaults stay, and the governor charges the
     measured seconds per 1,000 tokens.
   - *Borderline*, 1 to 4 of the 40 tasks above 10 s (at least 90% at most 10 s): accepted; the
     experiment report records which tasks exceed 10 s and amends ADR-001's cost line there.
   - *Bad*, 5 or more tasks above 10 s: the throughput criterion fails; `P3.2-02` with one cost
     cut, chosen as in item 4.
4. **Why it is slow, if it is.** Compare `seconds_per_1k_tokens` with the cost model (about 0.10 s
   at 40% of peak, more for the longest sequences). Far above it means the GPU is underused:
   check the GPU name, the attention kernel, and `render_share`. A `render_share` above 10% means
   CPU rendering matters, and caching rendered tokens would help. Near the model means the
   arithmetic itself is the cost; then fewer views (8) or a lower $L$ are the levers, and each
   changes the embedding (ADR-001's ablations).
5. **Cache size.** `payload_bytes_per_pair` must be 174,080 ($4 \cdot 2560 \cdot 17$);
   `bytes_per_pair` adds the shard header, about 274 bytes (measured at this width), and under
   1 KB is expected. `bytes_per_task` is the number P3.5 plans with: about 520 KB for a task of
   three pairs with views, about 31 KB without.
6. **Memory.** `memory.max_allocated` under 10 GB leaves room for a larger `batch_tokens` if the
   rate is low.
7. **`P3.2-02`.** The same reading at 8 views, against its own brief's rows; `views` must be 8,
   `sequences` 32 per task, and `payload_bytes_per_pair` 92,160 ($4 \cdot 2560 \cdot 9$). Then
   the view diagnostic, which decides nothing and goes to the human with the run: a pair cosine
   near 1 with a centered cosine also near 1 means the 8 views place each pair where the 16 do;
   a centered cosine well below 1, while the raw cosine stays high, means what distinguishes the
   pairs moves, which matters more to H than the shared direction. Read it against the scale, the
   median cosine between different pairs: a pair whose two embeddings are no closer than two
   different pairs would be has moved as far as the pairs are apart.8. **`MT-01`** (episode mode), against its brief's rows: `status` `complete`, `pairs.missing` 0,
   `cache.entries` equal to `pairs.distinct` (the exact count, expected 205.1k to 211.8k), and the
   check passed (`check.min_cosine` at least 0.999 over 256 pairs; expected above 0.9999). Then
   the rate: `encode.seconds_per_1k_tokens` against 0.157 to 0.181, and each process's
   `max_allocated` under 10 GB. A stop (exit 3) reads `pairs.missing` against the brief's
   threshold for a continuation. `run.json` `host.memory_limit_bytes`, the smaller of the 4xL4
   machine's `MemTotal` and its container's cgroup limit, is the memory `MT-02`'s brief compares
   with its need (about 22 to 27 GB at steady state).

## 7. Failure modes

| Symptom | Likely cause | First thing to change |
|---|---|---|
| `failed: ValueError: task <id>, view <d>: ... not a prefix ...` or `... do not decode back ...`, before any GPU time | The real tokenizer splits the pair text differently from the notebook vocabulary | Read `error.log`; adapt `render_pair` to the tokenizer (for example offsets); a code change, so a new commit and an amended brief. |
| `failed: RuntimeError: device cuda:0 requested but CUDA is not available` | The notebook ran without an accelerator | Check the push's accelerator; the brief's `Accelerator` line. |
| `failed: InputSearchError: found 0 model folders ending in qwen3_4b_grids15_sft139/transformers/bfloat16/1 with config.json ...`, before the model loads | The model is not attached, another version is, or Kaggle mounted it under a path that does not end with the handle | The message lists every `config.json` found. Attach version 1 of the model; if the mount moved, name the folder with `--model-dir` in the build's `args` (a new commit and an amended brief). |
| `failed: InputSearchError: found 0 files ending in arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json ...` (or `found 2`) | The competition is not attached, or a second input holds a copy under a folder of the competition's name | The message lists the files of that name; attach the competition only once, or name the file with `--challenges` in the build's `args` (a new commit and an amended brief). |
| The notebook stops in its environment check (`stages.json` `ok` false, no `run_logs/metrics.json`) | GPUs without bf16 (a substituted 2xT4), a code dataset version from another build, or inputs the notebook did not find | `docs/runbooks/build_submission.md`, failure modes; push again only under an amended, re-approved brief. |
| `failed: OutOfMemoryError` | A batch too large for the GPU; it shows on the first batch, which holds the longest sequences | Halve `batch_tokens`. |
| `failed: FloatingPointError: a pair embedding is not finite` (NaN or inf) | Overflow in a precision without bf16's range, or corrupt weights | Check `dtype` is `bfloat16` and `run.json`'s dtype; check the model files. |
| Divergence | Not applicable: nothing is trained. | |
| `stopped: wall time over max_wall_s`, exit 3 | The run is slower than the estimate | The cache and both ledgers are saved. If the throughput phase alone ran out of time, read section 6, item 4, first; on Kaggle, report and rerun under an amended brief (section 4); elsewhere, rerun to resume. |
| The notebook's hard stop kills the script (`stages.json`: `killed` true, `script_exit` -9) or Kaggle's `Timeout` ends the session | Setup before the script, or a single encode, took longer than the margins | Ledger lines and flushed shards written before the end survive; `metrics.json` does not. Report; raise `script_kill_s` and `Timeout` in an amended brief. |
| `failed: OSError` while loading | The folder found holds no loadable weights or tokenizer (another format) | Read `run.json` `inputs` and the session log's listing; name the right folder with `--model-dir`. |
| `failed: ValueError: ... task ids are not in ...` | The challenges file holds other tasks than the subset (not the evaluation set) | Check `challenges` and `subset` (DEV40 needs `arc-agi_evaluation_challenges.json`). |
| Episode mode: `failed: EpisodeSetError: ...` before any model loads | The attached build or check report is not build 2's, or the report failed | Attach the versions the build pins; never edit `manifest_sha256` without a passing check. |
| Episode mode: `failed: RuntimeError: encoding processes exited non-zero {k: code}` | One process failed (`error-rank<k>.log`: out of memory, a render check, CUDA) | Its finished chunks are merged; fix, then a continuation with `--seed-cache` under an amended brief. |
| Episode mode: `failed: CheckFailed: ...` | A cached embedding differs from a fresh encode: a merge or index error, a corrupt shard, or a nondeterministic kernel | Never attach this cache; `check.worst` names the pairs; diagnose on the CPU (`tests/test_cache_embeddings_episodes.py` reproduces a misfiled entry). |
| Episode mode: `stopped: wall time over max_wall_s, N pairs missing`, exit 3 | The encode ran slower than its estimate | Read `encode.seconds_per_1k_tokens`; under the brief's threshold `MT-02` encodes the rest itself, over it a continuation seeded from this output. |
| Episode mode: `... already holds embeddings: drop --seed-cache` | `--seed-cache` given for a `cache_dir` that has entries | Resume that cache in place, or seed an empty one. |
