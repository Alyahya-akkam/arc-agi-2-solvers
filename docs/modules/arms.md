# `arc26.arms`: paired adapter arms, one solver

Task P3.7 (BRIEF B5 P3.7, B7 G4), [ADR-019](../decisions/ADR-019-h-validation-paired-arms.md);
builds on ADR-011 (the split solver), ADR-017 (the providers), ADR-018 (P2.6-01's build). Our code.
Code: [`arc26/arms.py`](../../arc26/arms.py). Tests: [`tests/test_arms.py`](../../tests/test_arms.py);
the builder's side in [`tests/test_arms_build.py`](../../tests/test_arms_build.py). Runbook:
[`docs/runbooks/h_validation.md`](../runbooks/h_validation.md).

## Purpose

The split solver's arms entry: `python -m arc26.arms --end-time T --config JSON` runs every puzzle
under several adapter sources, one after another on the same worker, so the arms compare paired by
puzzle. Every arm is P2.6-01's build path (rank 16 on q, k, v, o; `decode_puzzle`'s views,
rescoring, and the reference's selection); only the adapter differs. `arc26.runner` starts it
instead of `arc26.solver` when `RunConfig.adapters` is set; `arc26/solver.py` is not changed.

## Math

An arm's adapter for puzzle $p$ with demonstrations $D$:

| Source | Adapter | Training |
|---|---|---|
| `base` | $\theta_0$: the initial weights, `lora_B` $= 0$, so the model is the base | none |
| `hyper` | $H(\bar z(D))$, $\bar z(D) = \frac{1}{\lvert D\rvert}\sum_{i \in D} \bar e_i$ (pair-hash order) | none |
| `sgd` | $\mathrm{fit}(\theta_0, D)$: one epoch over the 128 views (`arc26.ttt.fit`) | 128 steps |
| `hyper_polish` | $\mathrm{polish}(H(\bar z(D)), s)$ (`HyperPolishProvider`) | $s$ steps |
| `polish` | $\mathrm{polish}(\theta_0, s)$ (`SGDProvider`) | $s$ steps |
| `hyper_fit` | $\mathrm{fit}(H(\bar z(D)), D)$: `sgd`'s call with H's adapter (bf16, as the live model holds it) as `default_weights` (task brief amendment 8) | 128 steps |
| `hyper_merge_fit` | $\mathrm{fit}_{W'}(\theta_0, D)$ with $W'_\ell = \mathrm{bf16}(W_\ell + s\,B_{H,\ell} A_{H,\ell})$ on every targeted layer $\ell$: H's adapter merged into the base, then `sgd`'s call (task P3.9b amendment 1); the base restored from an exact copy after the decode | 128 steps |

`sgd` and `hyper_fit` call `arc26.ttt.fit` with the same arguments but `default_weights`: the same
128 views (`augment(n=16, shfl_keys=True, seed=1)`, `cut_to_len`), one epoch at batch 1, AdamW
created fresh by the trainer (zero moments), learning rate $5 \times 10^{-5}$ with 10% linear
warmup from 0 and cosine decay over the same step count, seed 42. Only $\theta$ at step 0 differs:
from $\theta_0$ (`lora_B` $= 0$) the first step's gradient reaches $B$ only, since
$\partial L / \partial A \propto B^\top$; from H's adapter both factors move from step 1.

`hyper_merge_fit` starts from H's function with the standard factors (HyperDreamBooth's recipe,
arXiv 2307.06949): `BaseMerge` copies every LoRA layer's `base_layer.weight` (same dtype and
device), then writes $W_\ell \leftarrow \mathrm{dtype}(W_\ell + \Delta_\ell)$ with
$\Delta_\ell = s\,B_\ell A_\ell$ computed in float32 from the live factors and $s$ the layer's
`scaling` (rsLoRA, $32/\sqrt{16} = 8$); `fit` then resets the LoRA to $\theta_0$ before its first
forward. At step 0 the model is $W' x$, which equals $(W + s\,B_H A_H) x$ up to one bf16
rounding per weight; the record keeps that rounding,
$\rho = \lVert W' - W - \Delta \rVert_F / \lVert \Delta \rVert_F$ over all layers. `solve`
copies the base back in a `finally` around the arm, after its decode or its error, never by
subtracting $\Delta$ ($\mathrm{bf16}(W + \Delta) - \Delta \ne W$ in general), and checks
every weight equals its copy; a restore that fails stops the worker (exit 7). On Qwen3-4B's q, k,
v, o the copy is $36 \times 2560 \times (4096 + 1024 + 1024 + 4096) = 943{,}718{,}400$ bf16 values,
1.89 GB (1,800 MiB) of GPU memory for the arm's span.

With `train_log`, every training arm (`TRAINED_SOURCES`) runs with `TRAIN_LOG_ARGS`
(`logging_strategy` `steps`, `logging_steps` 1) and its record keeps `train_log`, per step
`{step, loss, learning_rate, grad_norm}`, captured by `train_log_capture` around the trainer's
`log` (no change to `fit`); logging reads the loss and changes no gradient or random draw.

The answer's CE under an arm, for test output $y$ of input $x$ (the decode's identity view):
$\mathrm{CE} = -\frac{1}{|y|}\sum_t \log p(y_t \mid x, D, y_{<t})$ over the reply's tokens, one
forward (`calc_scores`) under `torch.inference_mode`, the mode `decode_puzzle` scores in.

The queue's order is $\mathrm{sort}_k\, \mathrm{sha256}(\texttt{"}s\texttt{:}k\texttt{"})$ for the
seed $s$; no puzzle starts after the soft deadline $T$, decoding stops at $T + f$ (`finish_s`).

**Views.** A test input's 16 view keys in sorted order are $2d + c$ for dihedral map $d$ of
`DIHEDRAL_ORDER` (identity, rot90, rot180, rot270, transpose, transpose.rot90, transpose.rot180,
transpose.rot270) and color permutation $c \in \{0, 1\}$. A map that keeps rows as rows gives the
original shapes, the others the transposed ones, so the prompts' token counts split into two
groups, `LENGTH_GROUPS` $= \{0,1,4,5,10,11,14,15\}$ and $\{2,3,6,7,8,9,12,13\}$, and a DFS batch must
lie in one. `DIHEDRAL_ONCE` $= ((0, 4, 10, 14), (2, 6, 8, 12))$ decodes each map once (its first
color permutation) in two full batches.

## Interfaces

| Name | Contract |
|---|---|
| `ArmSpec(name, source, steps=None)` | `name` matches `ARM_NAME`; `source` in `SOURCES`; `steps` 1 to 128 for `hyper_polish` and `polish`, None otherwise (`hyper_fit` trains one epoch, as `sgd`) |
| `GenSettings(out_dir, barc, rearc)` | GEN's folder and counts (`arc26.hyper.validation.gen_puzzles`) |
| `LoadCheckSettings(run_logs, max_abs, max_mean, dev4_challenges=None)` | the training run's logs and the tolerances (`arc26.hyper.validation.load_check`) |
| `Arms(arms, outputs_root, log_dir, cache_dir, checkpoint=None, cache=None, hypernet=None, views=None, order_seed=0, ce=True, finish_s=0.0, train_log=False, episodes_root=None, check_root=None, manifest_sha256=None, gen=None, load_check=None)` | the solver JSON's `adapters`; `from_mapping`, `from_cfg`, `to_json` (round trip), `folders()`; an H arm needs `checkpoint` and `hypernet`; `gen` and `load_check` need the episode build's three settings |
| `check_views(views) -> tuple \| None` | None, or batches of at most 4 distinct indices 0 to 15, each inside one length group |
| `queue_order(keys, seed) -> list[str]` | the fixed shuffled order |
| `train_log_capture(sink)` | context manager: every `UnslothFixedTrainer.log` call with a `loss` appends `{step, loss, learning_rate, grad_norm}` to `sink`; the class is restored on exit, also after an error |
| `seed_cache(target, source, hashes) -> int` | copies the entries the read-only cache holds and the worker's lacks, one put, bit for bit |
| `answer_ce(task, solutions, key, *, formatter, tokenizer, model, max_seq_length, max_new_tokens)` | per output `{nll, tokens, ce}` |
| `ArmRunner(settings, loaded, rank, *, hyper=None, source_cache=None, solutions=None)`; `ArmRunner.for_worker(settings, loaded, rank, data_dir)` | one worker's loggers, providers, caches; `for_worker` builds H from the checkpoint and checks its config against `hypernet` |
| `ArmRunner.solve(model, key, end_time) -> model` | every arm on one puzzle, in order; one record per arm (`last` keeps them); an arm that raises is recorded and the next runs |
| `ArmRunner.probe(model) -> model` | H's adapter for the DEV4 puzzle with the fewest demonstration cells (the load check's challenges, else the run's own), then the rollback, bit for bit; recorded as arm `probe` and op `h_probe`; raises `ArmStop` on a failure |
| `BaseMerge(model, adapter_name="default")`; `add() -> dict`; `restore()`; `nbytes` | the adapter's PEFT LoRA layers (plain unquantized `Linear` only, else `ValueError`) and an exact copy of their base weights; `add` merges $s\,B A$ in float32 with one rounding and returns `{layers, copy_bytes, device, dtype, delta_norm, weight_norm, rounding_norm, rounding_rel}`; `restore` copies back, then raises `RuntimeError` if a base weight was replaced after the merge or differs from its copy |
| `ArmRunner.stop(reason, entry)`, `ArmStop` | writes `stop-rank<k>.json` into the log folder and raises: the run stops with exit 7 |
| `run_worker(rank, queue, end_time, cfg, loaded, runner=None)`, `worker`, `local_worker` | the reference worker's loop and process body around `solve`; with an H arm, the probe first, and a stop when an H arm raises on the worker's first finished puzzle |
| `score_arm(predictions, solutions, keys)`, `select_and_score(settings, data_dir, keys)` | the reference's selection per arm, then the competition metric and attempt 1 per puzzle |
| `run_load_check(settings, cfg, end_time) -> int` | `python -m arc26.hyper.validation check` on GPU 0 |
| `spawn_workers(order, end_time, cfg)`, `run(cfg, end_time, *, launch, check) -> int`, `main(argv)` | the parent: clear, GEN, check, queue, workers, selection, summary; exit 0, 1, 2, 5 (the load check), or 7 (`EXIT_H_ARM`: a worker's stop marker, no selection) |

## Data flow

```text
run(cfg): clear outputs_root, log_dir, cache_dir
  gen?        -> arc26.hyper.validation.gen_from_episodes -> gen_data/ (data_dir), gen.json
  load_check? -> child: python -m arc26.hyper.validation check (GPU 0) -> load_check.json; != 0: exit 5
  queue_order(challenges keys & subset) -> mp queue -> spawn local_worker x workers
worker: solver.load(cfg) -> ArmRunner.for_worker (H from best.pt; MT-01's cache read only;
        its writable cache /kaggle/h_cache/rank<k>, outside the notebook's output)
  probe: H's adapter for a small DEV4 puzzle, rollback; raises -> stop-rank<k>.json -> exit 7
  per puzzle, per arm: [cache_seed] -> adapter (reset | H swap | fit | polish
     | H swap, copy, merge, fit) -> fingerprint -> answer_ce
     -> decode_puzzle(views, own clock, end T + finish_s) -> arm_outputs/<arm>/
     -> finally: base weights back from the copy (hyper_merge_fit)
     -> records-rank<k>.jsonl, <arm>/timing/timing-rank<k>.jsonl
  first finished puzzle: an H arm raised -> stop-rank<k>.json -> exit 7 (no selection)
run: select_and_score -> <arm>/submission.json, scores.json, selection.log -> arms.json
```

Timing ops (`{puzzle_id, op, seconds, tokens}`, each arm's own logger): `reset`, `cache_seed`,
`encode`, `h_forward`, `snapshot`, `swap`, `merge` and `unmerge` (`hyper_merge_fit`), `train`,
`candidate_train`, `fingerprint`, `ce`, `decode`, `selection`, and the arm's span as `puzzle`; the process logger (`timing_dir`) keeps
`load` and the whole puzzle's `puzzle`.

## Origin

Our code. `local_worker` and `run_worker` follow `arc26.solver.local_worker` and `worker` (the
reference's process body and loop: marker files, unsloth first, the deadline, the printed lines);
the arms call `arc26.solver.load`, `arc26.ttt.fit`, `arc26.decode.decode_puzzle` (its `views`
keyword is task P3.7's), `calc_scores`, the providers, and the moved selection
(`arc26.nvarc.select.write_submission`), all unchanged.

## Tests

`CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/pytest tests/test_arms.py`: 34 tests, about 12 s
on the CPU (4 Oct, torch 2.14.1, transformers 4.55.4, peft 0.17.1). The tiny random Qwen3 wrapped
as `load` wraps the base, a tiny H with random heads, the smoke path's stand-ins of unsloth and
datasets (`scripts/h_validation_report.py`).

| Group | Checks |
|---|---|
| Settings | `DIHEDRAL_ONCE` covers the 8 maps once in two full batches; the length groups are the maps that keep or swap rows; `check_views`' refusals; `ArmSpec` and `Arms` rules and round trip; the queue order is a fixed permutation |
| Runner | `adapters` start `arc26.arms` and ride in the JSON, which is otherwise the plain run's; the split solver, no hold-out, folders outside `outputs_dir` |
| Views | the real `decode_puzzle` with a recording DFS: the default is the reference's batches; `DIHEDRAL_ONCE` decodes each map once, input by input, one prompt length per batch |
| Arms | each arm decodes with its own adapter (the initial weights, H's emitted adapter exactly, one epoch from zero, the polish from H's adapter for $s$ steps, from zero for $s$), with the run's views, into its own folder, on its own clock; each arm's timing ops; the CE per output |
| Pairing | a puzzle cut mid-arms records `cut` and `not_started` and leaves the paired set; an arm that raises is recorded and the next runs; the loop stops at the soft deadline |
| Scores and parent | `seed_cache` bit for bit; `answer_ce` is `calc_scores` per reply token, under inference mode without grad; `score_arm` on a hand case; the reference's selection per arm; `run` stops at a failed load check (exit 5), launches the shuffled subset on a pass, refuses a competition rerun |
| Arm (f) and `train_log` (`tests/test_arms_hyper_fit.py`, 6 tests) | `hyper_fit` calls `fit` with `sgd`'s objects and values but the start (H's adapter bit for bit); the trainer starts from it over the same views and steps; the same `train` op as `sgd`; `train_log` adds only `TRAIN_LOG_ARGS` and records every step of both arms, none for `base`; the capture is undone, also after an error; an `hyper_fit` arm raising on the first puzzle counts as an H arm's |
| Arm (g) (`tests/test_arms_merge_fit.py`, 13 tests) | `hyper_merge_fit` is an H and a training source without steps; the merged forward equals the base plus H's LoRA forward (float32); the delta is PEFT's and the bf16 sum is rounded once; subtracting would drift in bf16, the copy restores exactly; the fit is `sgd`'s call from the initial weights; the decode reads the merged base and the fitted fresh LoRA, the next arm the original base; the record's H digest, merge numbers, `restored`, `train_log`, and ops; the base restored after a decode, a fit, or a merge that raised; a failed restore stops the worker; a replaced weight and a DoRA layer are refused; (a), (b), (c), (f) give the same fingerprints and CEs with (g) after them |
| Stops on H's path | the probe takes the smallest DEV4 puzzle, emits H's adapter, rolls back bit for bit; a probe that raises stops the worker before any puzzle; an H arm raising on the first puzzle stops it, on a later puzzle does not; `run` exits 7 without a selection on a stop marker, 1 with a selection on any other worker failure |

## Limits

- Development mode only: the solutions are read for the CE and the scores; a competition rerun
  ignores `adapters` (the notebook) and `run` refuses one.
- DEV40's pairs are encoded live by the unsloth-patched base, GEN's come from `MT-01`'s cache
  (transformers' base): their agreement is unmeasured (ADR-017). In the configs' order the H arm
  follows the base arm, so the live adapter during that encode is the zero adapter and the encoder
  sees the base even if `disable_adapter` were ignored; an order with an H arm after a trained arm
  relies on `disable_adapter` alone.
- H's adapter runs in bf16, the solver's cast (ADR-019, decision 6); the load check prices it.
- `hyper_merge_fit` rounds the merged weights to bf16 once; $\rho$ in each record says how much
  of H's delta that rounding moved. Its encode is a cache hit (arm (b) encoded the puzzle's
  pairs), and the next puzzle's encode runs after the restore, on the original base.
- The hard stop kills the parent too: the arms' selection needs `solver_kill_s` at least
  `solver_end_s` + `finish_s` + 900 s (the builder's rule).
