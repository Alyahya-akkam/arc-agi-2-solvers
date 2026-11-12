# Runbook: `scripts/build_episodes.py` (P3.4 episodes and leakage check)

Two Kaggle CPU notebooks built by `scripts/build_episodes_kernel.py`: the builder
(`<owner>/arc26-p3-4-episodes`) and the checker (`<owner>/arc26-p3-4-leakage`). CPU runs spend no
GPU quota, so no pre-run brief applies (BRIEF A9). Plan: `docs/research/meta_training_sources.md`
("the note"), section 5.2; departures from B5 P3.4: ADR-012. Module: `docs/modules/episodes.md`.

## Why this script exists

P3.5 trains H on episodes: a task's demonstrations $D$ and one left-out query $Q$ under one
augmentation (B2 Q6). P3.4 must deliver them from ReARC, BARC, and ARC-AGI-2 training, split by
family with 200 held-out validation puzzles and a 200-puzzle toy split, and prove that no ARC-AGI-2
evaluation pair entered them (B5 P3.4, B8). The checker's `pass` gates P3.5: its training notebook
refuses an episode set whose manifest sha256 has no passing report.

## What it does, step by step

Builder (`build`, internet on, no competition data):

1. Fetch the pins of `configs/episodes.yaml` into `/tmp/arc26-p3-4-sources`: `re_arc.zip` (size,
   sha256, git blob id), the BARC JSONL (size, sha256), and ARC-AGI-2 by a blobless clone whose
   sparse checkout holds only `data/training/` and `data/training.txt`, checked against git's own
   commit, tree, and blob ids. No evaluation blob is downloaded (gate G0).
2. Load and validate every grid (2-D, sides 1 to 30, integer colors 0 to 9; integral floats
   accepted). Invalid pairs are dropped and counted; duplicate BARC programs are merged.
3. Remove the tasks of the config's `drop` list.
4. Split by family (a ReARC generator with the ARC task of the same id, or one BARC problem):
   20 held-out ReARC generators give 5 puzzles each, and 100 BARC problems one each; every puzzle
   draws its (train, test) counts from DEV120's joint counts.
5. Sample training episodes: one per BARC problem (50,000), 30,000 from disjoint ReARC examples,
   20,000 leave-one-out ARC-AGI-2 episodes; $|D| \sim P = (0.15, 0.35, 0.30, 0.15, 0.05)$
   truncated to the pairs available; augmentation id $8 j + 4 t + k$ uniform over the table.
6. Count tokens exactly: $T(x, y) = (h_x w_x + h_x - 1) + (h_y w_y + h_y - 1) + 8$ per pair,
   summed over the episode in its longer orientation; episodes above 8192 are dropped and
   counted per source and split.
7. Drop every training or toy task holding a pair whose $H_c$ equals a validation pair's, then
   fail unless no family and no canonical pair crosses the split.
8. Shuffle the training pool once (a run of $N$ steps reads the first $N$) and write shards,
   validation puzzles in the competition format, ARC-AGI-2 pair-set digests (for R3),
   `MANIFEST.json`, and `run_logs/`.

Checker (`check`, internet off, competition and builder output attached):

1. Find the build by its `MANIFEST.json` under `/kaggle/input`, verify every listed file's size
   and sha256, and validate every record.
2. Hash every unique pair and grid once. With
   $\kappa(p) = \min_{g \in D_4} \mathrm{ser}(\rho(g x, g y))$, where $\rho$ relabels colors in
   order of first appearance jointly over $x$ then $y$, $H_c(p) = \mathrm{sha256}(\kappa(p))$ is
   equal for two pairs exactly when one is a $D_4 \times S_{10}$ image of the other.
3. Gates against the 120 evaluation tasks: G1 ids, G2 $H_e$, G3 $H_c$ over all demonstration and
   test pairs, G4 and G5 ($K_e$, $K_c$) for input and output grids of at least 25 cells and 2
   colors against the test inputs and outputs. Reports: R1 (matches with demonstration grids by
   cell bucket), split overlap, R3 (GitHub vs competition ARC-AGI-2 training pair sets).
4. Positive control: 3 evaluation test pairs moved by a random non-identity dihedral map and color
   permutation, injected into a copy of the first training shard; G3 must catch all 3.
5. `pass` = no G1 to G5 hit and the control caught.

## Why it is built this way

- **Two notebooks.** The builder needs internet and must never see evaluation data (G0); the
  checker reads evaluation solutions, which A0 allows for this check only, so it runs with
  internet off and returns ids and counts. No dataset upload: downstream notebooks attach the
  builder's output by `kernel_sources` (note 3.1).
- **Canonical hashes, not exact ones alone.** Training applies $D_4 \times S_{10}$, third-party
  sets ship augmented copies, and ARC itself has color-permuted duplicates (note 2.2); the
  positive control proves the exact gate alone would miss them (`caught_g2` is 0 by design).
- **Grid gates with floors.** The note measured 38 exact false positives at 9 cells and 1 at 25;
  a hit drops a source task, which costs nothing.
- **Task-local randomness.** Each choice is seeded by sha256 of the seed and its name, so a drop
  list changes only the dropped tasks' episodes and the allocation of the rest (tested).
- **Longer orientation.** The loader may redraw the augmentation; only exchanging rows and columns
  changes the count, so the drop rule bounds every redraw (as the notebook's
  `max_of_transposed`).
- **Family split.** A ReARC generator and the same-id ARC task carry the same concept, so they
  share a side; the cross-split canonical check catches identical pairs across families.
- **Bundled code.** A script kernel has one file, so the push folder carries `episodes.py`, this
  script, and the config as text, with stub package files, and records its commit.

## How to run it

Locally (no data, offline, seconds): `.venv/bin/python scripts/build_episodes.py --smoke`.

On Kaggle, from a clean commit (the folder builder refuses uncommitted bundled files):

```bash
python scripts/build_episodes_kernel.py --owner alyahya737 --kernel builder --out dist/P3.4/builder
python scripts/kaggle_ops.py push dist/P3.4/builder --timeout 7200
python scripts/kaggle_ops.py status alyahya737/arc26-p3-4-episodes
python scripts/kaggle_ops.py pull alyahya737/arc26-p3-4-episodes --task-id P3.4-build
python scripts/build_episodes_kernel.py --owner alyahya737 --kernel checker --builder-version <N> \
    --out dist/P3.4/checker
python scripts/kaggle_ops.py push dist/P3.4/checker --timeout 3600
python scripts/kaggle_ops.py pull alyahya737/arc26-p3-4-leakage --task-id P3.4-check
```

`<N>` is the builder version that completed. Expected (estimates from the note's sizes and local
timings: $H_c$ 0.13 ms per small pair, 0.47 ms at $30 \times 30$; grid validation 0.03 ms):
builder under 30 minutes (2.6 GB download, parsing, about 300k unique pairs hashed), under 4 GB
RAM, about 0.3 GB of output; checker under 15 minutes. No GPU.

## Outputs

Builder, in `/kaggle/working` (only `run_logs/` is pulled):

- `episodes/{train,val,toy}-NNNNN-of-NNNNN.jsonl`: one record per line, keys `episode_id`,
  `source`, `task_id`, `family`, `split`, `augmentation_id`, `tokens`, `n_demos`, `demos`,
  `queries` (un-augmented `{"input", "output"}` pairs).
- `val/val_challenges.json`, `val/val_solutions.json`: the 200 puzzles, competition format.
- `sources/arc2train_pair_sets.json`: exact and canonical pair-set digests per ARC-AGI-2 task.
- `MANIFEST.json`: schema, code commit, config and its sha256, verified input pins, image,
  permutation table, counts, held-out families, BARC seed ids, every file's size and sha256.
- `run_logs/MANIFEST.json` (copy), `run_logs/episodes_stats.{json,md}`: counts per source and
  split, drops (over length, short, too few pairs, cross-split, drop list), token percentiles and
  histogram, demo counts, timings.

Checker: `run_logs/leakage.json` and `.md`: verdict, manifest sha256, per-gate hits and tasks by
source and split, hit tuples `(gate, source, task_id, split, eval_task_id, match)`, the drop list
to paste, positive control, R1, split overlap, R3. Ids and counts only, never grids.

## How to read the results

- **Good:** `pass` true, control caught 3 of 3 by G3 and 0 by G2, split overlap 0, R3 equal for
  all common tasks, drops near the note's figures (about 0.05% over length). P3.5 may attach the
  build at that manifest sha256.
- **Borderline:** gates hit a few source tasks (the note expects on the order of 20 grid-gate
  drops). Paste the report's drop list into `configs/episodes.yaml` with a reason, commit,
  rebuild, re-check; repeat until `pass`. Every hit is on the list; a `seed:<id>` entry (a BARC
  seed that is an evaluation ID) drops every BARC problem naming that seed.
- **Bad:** the control fails (the check cannot detect moved pairs: stop, fix, re-run), G1 hits a
  task id (a source contains evaluation tasks: review the source before dropping), or hits by the
  hundreds (a source is evaluation-derived: remove the source, ADR).
- R3 differences mean the competition's training files differ from GitHub's; record them in
  `docs/experiments/P3.4.md`; they do not fail the check.

## Failure modes

- **Pin mismatch (`PinError`):** a source changed or a download was cut. Re-run once; if it
  repeats, the source moved: a new pin needs the researcher and a config change.
- **git missing or partial clone refused:** the error names the git step; Kaggle images carry
  git; the fallback is a raw-URL fetch of the 1000 files, which needs a code change.
- **Count mismatch (`BuildError` "found N, pinned M"):** a loader or format change; fix the loader,
  never the pin.
- **Timeout:** raise `--timeout` at push (CPU sessions allow 12 h); BARC parsing dominates.
- **Out of memory:** lower `sources.barc.max_pairs` (pairs kept per problem).
- **Tampered or partial output (`differs from the manifest`):** the checker refuses; rebuild.
- **OOM, divergence, NaN:** not applicable (no model, no training).
- Errors write `run_logs/error.log` with the traceback, pulled with the session log.
