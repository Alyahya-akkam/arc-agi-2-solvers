# `arc26.hyper.validation`: GEN and H's load check

Task P3.7 (BRIEF B5 P3.7, B7 G4), [ADR-019](../decisions/ADR-019-h-validation-paired-arms.md);
builds on P3.4's episodes (ADR-012), P3.5's training loop (`arc26.hyper.train`), P4.1's providers
(ADR-017). Our code. Code: [`arc26/hyper/validation.py`](../../arc26/hyper/validation.py). Tests:
[`tests/test_validation.py`](../../tests/test_validation.py).

## Purpose

Two inputs of H's validation. **GEN** turns P3.4's validation split into competition-format
puzzles, so the solver decodes the held-out generated puzzles as it decodes DEV40. **The load
check** makes sure, before any decode, that the H a run loads is the H meta-training validated:
it recomputes `MT-02`'s logged validation CE at its best point through `train.py`'s own path, and
measures what the solver's bf16 cast of H's adapter costs on the same episodes.

## Math

GEN: P3.4's validation episode $(D, q_i)$ of puzzle $p$ holds all of $p$'s demonstrations $D$ and
its $i$-th test pair $q_i = (x_i, y_i)$, un-augmented. GEN's puzzle is $(D, x_{i^*})$ with answer
$y_{i^*}$, $i^* = \min i$. The subset: the first $n_B$ BARC puzzles by sorted id, and $n_R$ ReARC
puzzles dealt round-robin over the held-out families in sorted order, each family's puzzles by
sorted id ($n_R = 50$ over 20 families: 3 for the first 10, 2 for the rest).

The load check, with $\mathrm{CE}_i$ the query CE of validation episode $i$ (train.py's Q6 loss,
one forward under `inject`), $c_i$ recomputed and $\ell_i$ logged:

$$\text{pass} \iff \max_i |c_i - \ell_i| \le \epsilon_{\max} \ \wedge\
\Big|\frac{1}{n}\sum_i (c_i - \ell_i)\Big| \le \epsilon_{\text{mean}}$$

for the current H and the zero adapter, with the checkpoint's step, H config, and manifest equal
to the log's, the build's, and the attached ones. The bf16 measurement: $d_i =
\mathrm{CE}^{\text{bf16}}_i - \mathrm{CE}^{\text{fp32}}_i$, the factors cast to bf16 and the delta
in bf16 (`inject(dtype=torch.bfloat16)`, PEFT's path on bf16 weights); flagged when $\bar d >
2\,\mathrm{SE}(d)$, $\mathrm{SE}(d) = s_d / \sqrt{n}$.

## Interfaces

| Name | Contract |
|---|---|
| `GenPuzzle` | `key` (letters, digits, `-`), `puzzle_id`, `source`, `family`, `task_id`, `episode_id`, `train`, `test_input`, `test_output`, `tokens`; `challenge()`, `solution()`, `index()` (no grids) |
| `gen_puzzles(records, *, barc, rearc) -> list[GenPuzzle]` | the lowest episode per puzzle, the stratified subset, in key order; refuses a non-validation id, another source, a short split, a repeated key |
| `write_gen(puzzles, out_dir) -> dict` | `arc-agi_evaluation_challenges.json`, `arc-agi_evaluation_solutions.json`, `gen_puzzles.json`; the summary with counts, hashes, index |
| `gen_from_episodes(episodes_root, check_root, manifest_sha256, out_dir, *, barc, rearc, max_tokens=8192)` | `open_episode_set` and `load_split("val")`, then the two above |
| `logged_best(run_logs) -> dict` | the `metrics.jsonl` validation record at `status.json`'s best step, with `run.json`'s episode ids |
| `compare(recomputed, logged) -> dict` | `n`, both means, `max_abs`, `mean_diff` |
| `bf16_flag(differences) -> dict` | `mean`, `se`, `n`, `z`, and `flagged`: $\bar d > 2\,\mathrm{SE}(d)$ (a bf16 CE below the float32 CE is never flagged) |
| `dev4_fingerprints(hypernet, encoder, model_id, tasks, seed=0, timing_logger=None)` | three emissions per task through `HyperProvider` (two fresh caches, one with the pairs shuffled); `equal` per task and overall |
| `load_check(settings) -> dict` | the report: `verdict`, `reasons`, `current`, `zero`, `bf16_minus_fp32` (mean, SE, z, `flagged`), `fp32_minus_zero`, per-episode CEs, embeddings, `dev4`, the attention kernel |
| `main(argv)` | `check` (exit 0 pass, 6 fail, 1 error) or `gen`, from one JSON config; writes the report to its `out` |

## Data flow

```text
gen:   find build by manifest sha256 + passing leakage.json -> val shards (sha256 checked)
       -> lowest episode per puzzle -> 50 BARC sorted, 50 ReARC round-robin -> gen_data/*.json
check: best.pt -> config (hypernet == configs/hypernet.yaml, manifest == attached)
       -> TrainConfig (paths, device, gpus 1) -> load_base + wrap_targets -> build_sequence x 286
       -> PairEmbeddings.fill (MT-01's cache) -> per episode: zero, fp32 inject, bf16 inject
       -> compare with metrics.jsonl at status.json's best step -> DEV4 digests -> load_check.json
```

Timing ops (`timing-load-check.jsonl`): `load`, `embed`, `validate_zero`, `validate_fp32`,
`validate_bf16` (episode tokens), `dev4_emit`.

## Origin

Our code. The check composes `arc26.hyper.train`'s validation path unchanged (`open_episode_set`,
`load_split`, `build_sequence`, `load_base`, `PairEmbeddings`, `query_ce`, `load_checkpoint`) and
repeats `_Worker.episode_ce`'s forward; GEN reads P3.4's records (`arc26.hyper.episodes`).

## Tests

`CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/pytest tests/test_validation.py`: 19 tests,
about 5 s. GEN on synthetic records (the numeric lowest index, the round-robin counts, the keys,
the refusals, the files) and on the episode fixture; the load check on a tiny H trained by
`train.py`'s smoke path (one process): it reproduces the logged best point (max difference under
$10^{-5}$), measures bf16, agrees on DEV4; it fails for another checkpoint (the step and the
current H's CEs, not the zero adapter's), perturbed weights, another H config, another manifest,
and a tolerance below the log's rounding; the CLI's exit codes and reports; the bf16 flag rule on
hand cases (just above and below 2 SE, a negative mean, no spread).

## Limits

- The check runs one process on one GPU: three references on 286 episodes at `P3.5-01`'s 0.5 s
  per forward, about 7 minutes, and 10 with the base's load (estimate); its zero-adapter CE
  repeats MT-02's step-0 values.
- The bf16 measurement is PEFT's bf16 path on transformers' base; the solver's unsloth kernels may
  round the delta differently (ADR-019).
- GEN's keys replace characters other than letters, digits, and `-`; P3.4's ids have none.
