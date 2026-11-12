# ADR-019: H's validation runs four adapter sources in one solver, paired by puzzle; GEN at 8 views; a load check first; H's adapter in bf16

- Status: Proposed, 2026-10-04, until runs `P3.7-01` (DEV40) and `P3.7-02` (GEN) report; G4
  reads their table. Source: task brief `docs/tasks/P3.7.md` with its amendments of 4 Oct (the
  orchestrator's decisions on the pricing); BRIEF B5 P3.7, B7 G4, B2 Q3, A8, A9.
- Tasks: P3.7 (the runs); P4.3 (the decode-count axis and arm (e) reuse the views keyword and the
  arms); P4.6 (trains on GEN's candidates, kept in the notebooks' output).
- Related: ADR-002 (H's scope), ADR-003 (H, then the polish), ADR-011 (the split solver),
  ADR-013 (the held-out pair), ADR-017 (the providers, the bf16 cast), ADR-018 (P2.6-01's build).

## Context

Gate G4 reads $f_b$, the share of the gap between the zero adapter (a) and the SGD adapter (c)
that H (b) closes, on the 200 held-out generated puzzles and on DEV40 (B5 P3.7, B7): at least 50%
Phase 4 as planned, 25% to 50% Phase 4 with the heaviest polish that fits, below 25% Submission
2's system is final. The arms differ only in the adapter: P2.6-01's build path for all (rank 16
on q, k, v, o, every pair, no loop, the reference's decode, rescoring, and selection). The
solver could choose one adapter source per run only through the loop (`HoldOut.rounds`), and its
16 decode views were hard-wired in `decode_puzzle` (a protected file).

Pricing (estimates from `P2.6-01`'s measured per-puzzle ops; GEN from DEV120's grid-size strata
matched to P3.4's validation token sizes; four workers replayed in a shuffled order): DEV40 at
arms (a) to (d) 4.9 h (4.1 to 5.7 h for the zero adapter and H decoding 0.6 to 1.4 times the SGD
adapter's time), with arm (e) 6.0 h; GEN at 100 puzzles and 16 views 7.1 h. The orchestrator
fixed the cuts: arm (e) off on both, GEN at 100 puzzles and 8 views, DEV40 at 16 views, and one
protected keyword in `decode_puzzle`.

## Decision

1. **One solver launch, the arms per puzzle.** `arc26.arms` is the split solver's arms entry
   (`python -m arc26.arms`, the solver's arguments); `arc26.runner` starts it when
   `RunConfig.adapters` is set, and `arc26/solver.py` and its AST parity test stay untouched.
   Each worker loads the model once (`arc26.solver.load`), builds H from `best.pt` on the live
   PEFT model (`HyperProvider.from_checkpoint`), and runs every arm on a puzzle before the next:
   `base` loads the adapter's initial weights (`lora_B` zero, the base itself, P3.5); `hyper`
   swaps in $H(\bar z(D))$; `sgd` runs `arc26.ttt.fit` from the initial weights (P2.6-01's
   training, statement for statement); `hyper_polish` runs `HyperPolishProvider` with $s = 13$
   (ADR-017); `polish` (arm (e), implemented, off in both configs) runs `SGDProvider` for $s$
   steps from the initial weights. Each arm decodes on its own clock (`decode_puzzle`'s 1200 s cap
   from the arm's start, as P2.6-01's puzzle clock started before the fit) into its own folder,
   records its timing ops, the sha256 of its adapter, and the teacher-forced CE of every answer
   (one forward under `torch.inference_mode`, as `decode_puzzle` scores, `calc_scores` per reply
   token). The order (a), (b), (c), (d) puts H's encode
   (DEV40's pairs, encoded live) right after the base arm, when the live adapter is the zero
   adapter: the encoder sees the base exactly even if unsloth's kernels ignored
   `disable_adapter` (ADR-017's assumption), and (d) reads (b)'s cached embeddings.
2. **Paired under a cut.** The queue holds the puzzles in a fixed shuffled order (sha256 of the
   seed and the task id). No puzzle starts after the soft deadline; a started puzzle's arms may
   run `finish_s` longer, and an arm still decoding then is cut. The report uses the puzzles every
   arm decoded to the end, so a cut drops the same shuffled puzzles from every arm.
3. **GEN.** P3.4 build 2's validation split, one episode per puzzle (the lowest episode index),
   its query the test pair (its output the answer, the held-out pair), its demonstrations
   un-augmented (the solver applies its own views); the first 50 BARC puzzles by sorted id and 50
   ReARC puzzles dealt round-robin over the 20 held-out families in sorted order (3 each for the
   first 10 families, 2 for the rest). Written under the competition's file names, so the solver
   and the reference's selection read it unchanged; its demonstrations' embeddings are copied from
   `MT-01`'s cache (the ones H was validated on), DEV40's encoded live, in both cases into each
   worker's writable cache `/kaggle/h_cache/rank<k>`: outside `/kaggle/working`, the notebook's
   output, which Kaggle keeps and later notebooks attach, since the caches are scratch rebuilt by
   every run (a few MB), written by four workers with one writer each, and DEV40's entries embed
   evaluation pairs, which no attachable output should carry.
4. **GEN's 8 views.** `decode_puzzle` gains one keyword-only parameter, `views` (default None:
   the reference's 16 views in its batches, byte-identical; the P0.1 fixture and
   `tests/test_split_solver.py` green, only its intended-difference entry amended): batches of
   indices into a test input's 16 view keys in sorted order, view $2d + c$ being dihedral map $d$
   (identity, rot90, rot180, rot270, then the four after a transpose) under color permutation $c$.
   GEN decodes `DIHEDRAL_ONCE` $= ((0, 4, 10, 14), (2, 6, 8, 12))$: every dihedral map once, its
   first color permutation, as two full batches of four. A batch must hold prompts of one length;
   the maps that keep rows as rows (identity, rot180, transpose then rot90, transpose then rot270)
   give one length and the others the other, whatever the grids, so the 8 maps fill exactly two
   batches of the reference's width. The decode cost per test input halves (two of four DFS
   batches); rescoring is unchanged (8 views per candidate); `score_kgmon`'s count term ranges over
   8 views. DEV40 keeps 16 views, so its SGD arm compares with `P2.6-01` and `P1.3-01`.
5. **The load check, before any decode.** The parent runs `python -m arc26.hyper.validation
   check` on GPU 0: H from `best.pt` through `arc26.hyper.train`'s validation path (the verified
   build, `build_sequence` with each episode's augmentation, `load_base` and `wrap_targets`,
   $\bar z$ in record order from `MT-01`'s cache, the query CE under `inject`) must reproduce the
   per-episode CE that `MT-02` logged at its best point, for the current H and the zero adapter:
   $\max_i |c_i - \ell_i| \le 10^{-3}$ and $|\overline{c - \ell}| \le 10^{-4}$. A recomputation on
   the same image repeats the forward exactly up to the log's 6-decimal rounding ($5 \times
   10^{-7}$; the CPU smoke reproduced its log to $9 \times 10^{-7}$), while neighbouring
   validations of `P3.5-01` differ by at least 0.041 per episode and $3.4 \times 10^{-4}$ in the
   mean, so the bounds separate a faithful load from another checkpoint. The checkpoint's step must
   be the logged best step, its H config `configs/hypernet.yaml`'s, its state dict strict, its
   manifest the attached build's; any failure stops the run (exit 5). On DEV4 it emits H's
   adapter three times (two fresh caches, one with the pairs shuffled) and records whether the
   digests agree (evidence for ADR-017; not a stop, since every arm reads one cache).
6. **H's adapter in bf16.** `arc26.solver.load` casts the live adapter to bf16, and the H arms
   use it as every other arm does: H's float32 export is rounded at the swap. The load check
   measures the cost on the 286 episodes: $d_i = \mathrm{CE}^{\text{bf16}}_i -
   \mathrm{CE}^{\text{fp32}}_i$ with the factors cast to bf16 and the delta in bf16 (PEFT's path
   on bf16 weights). Pre-registered: $\bar d > 2\,\mathrm{SE}(d)$ flags the H arms as a lower bound
   and names a float32 path as the follow-up; the run continues.
7. **The report.** `scripts/h_validation_report.py`: per arm and set the task accuracy with
   voting, the attempt-1 exact match, the answer CE, seconds per puzzle by op;
   $f_x = (m_x - m_a)/(m_c - m_a)$ for $x \in \{b, d\}$ with a 90% paired bootstrap interval
   (10,000 resamples, seed 0); undefined on task accuracy unless $m_c - m_a \ge 2$ tasks (puzzles on
   GEN; signed since 17 Oct); G4 reads a No gap row first (the SGD arm beats the zero adapter on
   neither metric), then $f_b$ on task accuracy, on CE when that is undefined; per-source rows on
   GEN; sets in different branches go to the human. The consequences of Bad and No gap depend on
   whether H's meta-training has ended (the briefs' section 1, the report's `--training-ended`).

8. **Fail fast on H's GPU path.** The load check runs transformers' base, so the H arms' path on the
   solver's unsloth-patched model (the live encode under `disable_adapter`, the swap, the rollback)
   is first run by the workers. Each worker probes it once after its model load (`ArmRunner.probe`:
   H's adapter for the DEV4 puzzle with the fewest demonstration cells, by `P2.6-01`'s token counts
   `36a08778`, about 2,100 tokens per view, then the rollback, bit for bit; a few seconds), and a
   probe that raises, or an H arm (`hyper`, `hyper_polish`) that raises on the first puzzle a worker
   finishes, stops the run: the worker writes `stop-rank<k>.json` and exits, the parent writes its
   summary without a selection and exits 7. A later H arm that raises is recorded and the run goes
   on. GEN keeps the guard though its embeddings are cached (the swap, the rollback, and the polish
   still run on the solver's model).

## Alternatives considered

- **Four notebooks, or a series of four launches (P1.3's series mode).** Lost: not paired under a
  cut (each arm would end at its own puzzle), four model loads, and four times the queue; the
  series also requires the notebook configuration first.
- **The arm loop in `arc26/solver.py`'s worker.** Lost: the worker and the entry are pinned to the
  reference by `tests/test_split_solver.py`'s AST checks, and every submission build runs that
  module; a separate entry keeps the default path unchanged by construction.
- **A script notebook (`kind: script`) running the solver itself.** Lost: the task brief asks
  for a builder option refused for the submission notebook; the arms entry and `RunConfig.adapters`
  are what P4.3 and P4.4's builds reuse.
- **GEN at 100 puzzles and 16 views (7.1 h), or 64 puzzles at 16 views (4.8 h), or the whole
  queue cut at 5 h (about 65 puzzles).** Lost to the orchestrator's choice: 100 puzzles keep
  the interval on $f_b$ narrower, and the 8 views are the same for every arm.
- **The first 8 views in list order** (`[[0, 1, 4, 5], [2, 3, 6, 7]]`, also two full batches).
  Lost: the four rotations under both color permutations drop every reflection; covering the
  8 symmetries once keeps the geometric diversity the vote relies on, at equal cost.
- **Fewer views by an earlier decode deadline.** Lost: it cuts by time, not by view, differently
  per arm and puzzle.
- **H's adapter in float32 for the H arms** (ADR-017's open item). Lost: two numerics paths in one
  comparison (the SGD and polish arms train and decode in bf16), a dtype switch of the live
  adapter per arm, and under unsloth's patched forward the factors may be cast to the activation
  dtype anyway, with a cached cast copy for a non-bf16 factor (from unsloth's source as recalled,
  not verified here); the load check prices the rounding instead.
- **The load check inside the workers.** Lost: it would start after a model load per worker and
  could not stop the run before the first decode; a child process on GPU 0 runs before them.
- **Mark a failing H arm `raised` and go on.** Lost: a path broken on the solver's model would
  fail every H arm until the deadline, about 2.6 h of two of four arms' quota for nothing.
- **H's probe in the load check's process.** Lost: that process runs transformers' base; probing
  the unsloth model there needs the GPU stack and a second model load (minutes), while each
  worker's probe uses the model it already loaded (seconds) and tests the exact path.

## Consequences

- Costs (estimates, the briefs' section 6): `P3.7-01` about 4.9 h of wall (4.1 to 5.7 h), quota
  4.9 h at 1x and 9.9 h at 2x; `P3.7-02` about 4.3 h (3.8 to 4.9 h), 4.3 and 8.7 h.
- Seconds per puzzle, DEV40 (estimate): (a) 336, (b) 339, (c) 490, (d) 355; GEN: BARC 67, 67,
  127, 74; ReARC 146, 147, 318, 171.
- Outputs: per arm `run_logs/arms/<arm>/` (timing, selection log, submission, scores), the
  workers' records, the load check's and GEN's reports, a summary; the candidates stay in each
  notebook's output (`arm_outputs/`), GEN's files too (`gen_data/`), for P4.6.
- DEV40's H arms read embeddings encoded live by the unsloth-patched base, GEN's the cached ones
  of transformers' base (`MT-01`): ADR-017's unmeasured agreement enters DEV40 only.
- A stop on H's path (exit 7) costs the launch, the load check, and the workers' loads, about
  1,100 s of wall (0.6 quota h at 2x), for a probe that raises; at most one puzzle per worker
  more (DEV40 about 1,500 s on average, GEN about 560 s) for a first-puzzle H failure.
- The views keyword is general (any batches of one length each); P4.3's decode-count axis can
  decode fewer views with it, and more by a second decode under `view_seed`.

## Overturn criterion

- The bf16 flag ($\bar d > 2\,\mathrm{SE}$): the H arms are a lower bound and a follow-up tests
  a float32 live adapter under unsloth on Kaggle before P4.3.
- A soft-deadline cut of more than 10% of a set's puzzles: the report says so and the next
  repetition is repriced from this run's timing records.
- If the load check fails on a run whose checkpoint and logs are right (a kernel difference above
  $10^{-3}$ per episode), the tolerance is re-derived from the logged differences, in an amended
  brief, never edited after a run.

## Amendment 1 (21 Oct, task P3.9b amendment 1): arm (g), H merged into the base, then a fresh fit

- **Context.** The human's idea of 17 Oct was to merge H's adapter and then fine-tune. Arm (f)
  (`hyper_fit`, `P3.7-03`) instead starts the fit's factors at H's and ended with a higher final
  training loss than (c) on 38 of 40 puzzles at the same score, which points at the factor start.
  HyperDreamBooth (Ruiz et al. 2024, arXiv 2307.06949) merges the predicted adapter into the
  weights and trains a fresh LoRA. The human chose this arm for `P3.9-02` (choice A, 21 Oct).
- **Decision.** Source `hyper_merge_fit` (an H source and a training source): H's adapter as arm
  (b) loads it; `arc26.arms.BaseMerge` keeps an exact GPU copy of every targeted base weight
  (`base_layer.weight` of each PEFT LoRA layer, the tensor PEFT's forward and unsloth 2025.9.7's
  fast paths read), writes $W_\ell \leftarrow \mathrm{bf16}(W_\ell + s\,B_\ell A_\ell)$ with the
  product in float32 and $s$ the layer's rsLoRA scaling; then `arc26.ttt.fit` exactly as (c), from
  $B = 0$; the decode as every arm; the copy written back bit for bit in a `finally` around the
  arm, checked with `torch.equal`, before the next arm; a restore that fails stops the run (exit
  7). The arm runs after the existing arms, so they pair with earlier runs.
- **Alternatives** (the runbook's section 3 gives the evidence): arm (f) alone; PEFT's
  `merge_adapter` or `merge_and_unload` (bf16 product on the GPU, drops the layers the trainer
  addresses, unmerges by subtraction); a restore by subtraction (bf16 drift into every later arm);
  the copy in pinned CPU memory or a reload of the weights from the model files; a second frozen
  LoRA holding H (unsloth's fast path reads the first active adapter only).
- **Consequences.** 1,800 MiB more GPU memory for the arm's span (`P3.7-03`'s per-puzzle peak
  14,190 MiB, so about 15.6 GiB of 22.5); about 482 s per DEV40 puzzle, (c)'s measured 479.8 s
  plus the merge and the restore (estimate); each record keeps H's digest, the merge's numbers
  (with the bf16 rounding's share of the delta, $\rho$), and `restored`.
- **Overturn criterion.** A $\rho$ near 1 on most puzzles (the bf16 merge kept little of H's
  delta): the arm then reads as (c) on a perturbed base, and a follow-up keeps the merged delta in
  float32 (an fp32 residual beside the bf16 weight) before any further merge-then-fit run.
