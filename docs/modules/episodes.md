# Module: `arc26/hyper/episodes.py`

Meta-training episodes and the leakage gates (task P3.4; BRIEF B2 Q6, B4, B5 P3.4, B8). Numpy and
the standard library only, so it runs in a Kaggle CPU notebook without an install. Script and
runbook: `scripts/build_episodes.py`, `docs/runbooks/build_episodes.md`. Sources and design:
`docs/research/meta_training_sources.md`.

## Purpose

Build episodes $(D, Q)$ with one augmentation applied to every pair, store them with the
provenance the split and the leakage check need, and test them against the evaluation set.

## Math

- Augmentation id $a = 8 j + 4 t + k$: transpose if $t = 1$, then `np.rot90` $k$ times, then
  color $c \mapsto \pi_j(c)$ from a seeded table whose entry 0 is the identity, the order and
  spelling of the notebook's `ArcDataset.augment`. `PermutationTable.descriptor(a)` is the key
  suffix `forward_mod` applies, e.g. `transpose.rot90.permute3810245679`.
- Tokens per pair in the notebook format: $T(x, y) = (h_x w_x + h_x - 1) + (h_y w_y + h_y - 1) + 8$;
  an episode counts in its longer orientation.
- $H_e$, $K_e$: sha256 of compact JSON ($H_e$ equals P3.2's `pair_hash`).
  $H_c(x, y) = \mathrm{sha256}(\min_{g \in D_4} \mathrm{ser}(\rho(g x, g y)))$ with $\rho$ a joint
  first-appearance relabeling; $K_c$ the same for one grid. $H_c(p) = H_c(p')$ exactly when
  $p' = (g, \pi) \cdot p$.

## Interfaces

- B4 schemas, unchanged: `Pair`, `Episode` (defined here until a shared module holds them).
- Grids and hashes: `validate_grid`, `exact_grid_hash`, `exact_pair_hash`, `canonical_grid_hash`,
  `canonical_pair_hash`, `is_nontrivial`, `pair_tokens`, `episode_tokens`.
- Augmentation: `Augmentation`, `PermutationTable` (`seeded`, `from_strings`, `augmentation`,
  `descriptor`), `augmentation_id`.
- Randomness: `derive_seed`, `rng_for`, `keyed_order`, `allocate`, `ExampleStream`.
- Sources: `SourceTask`, `load_arc_dir`, `load_rearc_zip`, `iter_barc_jsonl`, `barc_task_id`.
- Episodes: `split_families`, `draw_demo_count`, `sample_episodes`, `Puzzle`, `EpisodeRecord`,
  `validate_record`, `write_shards`, `read_shards`.
- Gates: `leakage_report`, `positive_control`, `control_outcome`, `canonical_matches`,
  `pair_set_digests`, `compare_pair_sets`. Every hit enters the report's `drop` list; a provenance
  hit is named `seed:<id>` (`SEED_PREFIX`), and the builder drops every task naming that seed.
  Design decisions and their departures from B5 P3.4: ADR-012.

## Data flow

Loaders give `SourceTask`s of validated `uint8` pairs; the script splits families, samples
`EpisodeRecord`s, and writes JSONL shards plus `MANIFEST.json`; P3.5 reads them with
`read_shards`, decodes `augmentation_id` with the manifest's table, and converts with
`EpisodeRecord.to_episode`; the checker feeds all records to `leakage_report`.

## Origin

New code. Augmentation and format semantics follow the notebook's `arc_loader.py`
(`permute_mod`, `ArcDataset.forward_mod`, `augment`, `QwenFormatter`); this branch does not import
it, and the tests check equality against cell 2 of `reference/reference_notebook.ipynb`.

## Tests

`tests/test_episodes.py`: $H_c$ invariance under all 8 dihedral maps and color permutations (via
the notebook's `forward_mod`), equality of $H_c$ with brute-force equivalence, joint relabeling,
shape sensitivity, `pair_tokens` against `QwenFormatter` under a stand-in word-level tokenizer,
descriptors against `forward_mod` and `augment`'s keys, loaders, sampling modes, over-length
drops, record schema, shards, every gate with positive and negative controls, the positive
control and its failure when $H_c$ is blinded. `tests/test_build_episodes.py`: config, the smoke
loop, split disjointness, manifest determinism, task-local randomness, fetch pins and G0, check
refusals, push folders against `kaggle_ops.py`'s checks.

## Known limits

- Pixel edits, crops, scalings, and concept-level overlap are not hash-detectable (note 2.2).
- R2 (near-duplicate inputs) is not implemented; the note marks it optional.
- Token counts assume NVARC's word-level tokenizer; P3.2's brief asserts it on Kaggle.
- `Pair` and `Episode` live here until the B4 shared module lands; they are unchanged from B4.
