# Meta-training sources, leakage check, and the Kaggle build for P3.4

Author: `researcher` subagent. Date: 2026-10-01. Feeds: P3.4 phase 2 (`arc26/hyper/episodes.py`,
`scripts/build_episodes.py`, `tests/test_episodes.py`), P3.5, P3.6, P3.7, B8 (leakage).

## Question

What data do we build meta-training episodes from, how do we get it onto Kaggle, and how do we
prove that no ARC-AGI-2 evaluation pair leaks in? In five parts (brief `docs/tasks/P3.4.md`):
sources, leakage, Kaggle build, episode design, recommendation.

## Sources and conventions

| Source | Revision | License |
| --- | --- | --- |
| `github.com/michaelhodel/re-arc` | `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce` (2025-02-24); `re_arc.zip` sha256 `4a7c3094...9ac4` | MIT |
| `github.com/xu3kev/BARC` | `a7b51a6b1ff969da3a78a71c533b6d79a93966e7` (2024-11-20) | no LICENSE file; GitHub API `license: null` |
| `huggingface.co/datasets/barc0/100k-gpt4-description-gpt4omini-code_generated_problems` | `c2c13194e8e144ae04236c4b12af46616df43adc`; JSONL 2,550,057,070 B, LFS sha256 `b84560f8...2498` | MIT (dataset card) |
| `barc0/100k-gpt4omini-description-gpt4omini-code_generated_problems` | `24555e6d68b69f95a15f78715b98f19c02aef099`; JSONL 3,148,629,747 B, sha256 `836004f5...ba2c` | MIT |
| `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` | `4264798428e3c48cd67ec7a3402e0847f645da52`; `data_100k.jsonl` 2,798,072,950 B (`982aa76c...d8a2`), `data_suggestfunction_100k.jsonl` 3,389,250,697 B (`197a1846...15bf`) | MIT |
| `github.com/fchollet/ARC-AGI` (ARC-AGI-1) | `399030444e0ab0cc8b4e199870fb20b863846f34` (2025-04-04) | Apache-2.0 |
| `github.com/arcprize/ARC-AGI-2` | `f3283f727488ad98fe575ea6a5ac981e4a188e49` (2025-05-15) | Apache-2.0 |
| `github.com/1ytic/NVARC` (`SDG/`, `ARChitects/`, `nvarc_2025.pdf`) | `846d0198efa752534594e321fc3289fc0a06c657` | no license (as in `kaggle_first_runs.md`) |
| `arxiv.org/abs/2505.11831` (ARC-AGI-2 technical report) | read 2026-10-01 | |
| HF Hub API and datasets-server `size`, `statistics`, `rows` endpoints for the BARC sets | read 2026-10-01 | |
| `runs/K0-latest/run_logs/env_probe.{md,json}` (K0, counts only) | 2026-09-30 | |

What was and was not read:

- ARC-AGI-2 was sparse-cloned with only `data/training` checked out plus the ID lists
  `data/training.txt` and `data/evaluation.txt`; no blob of `data/evaluation` was fetched. No
  ARC-AGI-2 evaluation grid or solution was read.
- The ARC-AGI-1 evaluation directory contains 6 ARC-AGI-2 evaluation task IDs (Section 2.1). It
  was hashed programmatically for the aggregate overlap counts below; no grid of it was displayed.
  The ReARC repo's `arc_original.zip` (ARC-AGI-1 training and evaluation) was listed by file name
  only. Both scratch copies were deleted after the analysis.
- BARC was sampled, not downloaded: 600 rows (rows 1000 to 1099, 40000 to 40099, 65000 to 65099 of
  the gpt4-description set and of HEAVY) through the datasets-server `rows` endpoint.
- Every kaggle.com page body is JS-rendered (as in `kaggle_first_runs.md`); Kaggle limits not
  measured by K0 rest on search snippets and are marked **unverified**.

Paths are relative to each repo root. Numbers marked *measured* come from scripts run in the
session scratchpad over the pinned sources (not committed); the methods are described where used.

## Short answer

- **Sources.** ReARC (`re_arc.zip`, 400 generators x 1000 verified examples, MIT), BARC (HF
  JSONL, MIT, 102,895 problems in the first set), ARC-AGI-2 training (1000 tasks, Apache-2.0).
  **Do not use ARC-AGI-1 as a separate source**: ARC-AGI-2 training already holds 391 of its 400
  training tasks and 376 of its 400 evaluation tasks with identical pair sets, and 6 of the
  remaining ARC-AGI-1 evaluation tasks *are* ARC-AGI-2 evaluation tasks (3 of them in DEV4).
- **The base model already saw most of it.** NVARC's SFT of `qwen3_4b_grids15_sft139` used ReARC
  (all 400), ARC-AGI-2 training, ConceptARC, MINI-ARC, and 103k synthetic puzzles, 55,886 of them
  generated from summaries that include all 120 ARC-AGI-2 evaluation puzzles (paper p. 3 to 5, 7).
  BARC was left out. So BARC is the only large source the base has not fit, and the base has a
  concept-level exposure to DEV120 that no choice of ours can remove.
- **Leakage.** No ARC-AGI-2 evaluation ID is a ReARC generator, a BARC seed, or an ARC-AGI-2
  training task. An exact pair hash is not enough: training applies $D_4 \times S_{10}$
  augmentations, third-party sets ship augmented copies (NVARC's `arc2_evaluation6` is the
  evaluation set under random dihedral and color maps), and ARC-AGI-1 itself has color-permuted
  duplicate pairs across its training/evaluation split (`caa06a1f` vs `50a16a69`). Use exact plus
  a $D_4 \times S_{10}$-canonical pair hash, plus grid-level gates on the evaluation test inputs and
  outputs, in a separate internet-off Kaggle notebook that returns counts and IDs only.
- **Kaggle.** One CPU notebook with internet (no competition data attached) fetches the pinned
  sources and writes episodes as its output; a second CPU notebook without internet attaches the
  competition and the first one's output and runs the check. Training chains both outputs by
  `kernel_sources`; no dataset upload and no API call from inside Kaggle are needed. 100k
  episodes are about 0.3 GB uncompressed: far inside every limit.
- **Episodes.** One task, $|D| \in \{1, \dots, 5\}$ demos matched to DEV120's demo counts, one
  query, one augmentation for all pairs. Token count is exact:
  $\sum_{\text{pairs}} (h_i w_i + h_i - 1) + (h_o w_o + h_o - 1) + 8$ (verified against NVARC's
  tokenizer). At 8192, about 0.05% of the mix is dropped; no sampled ReARC or BARC episode of
  this design exceeds it.
- **First run.** A 100k-episode pool (50% BARC, 30% ReARC, 20% ARC-AGI-2 training) that every run
  consumes as a prefix, a 200-puzzle validation split by generator family (100 ReARC from 20
  held-out generators, 100 BARC problems), and a 200-puzzle toy split for P3.5.

## 1. Sources

### 1.1 Summary

Per-pair statistics are *measured* over all pairs (ARC sets, ReARC) or over the 600-row sample
(BARC). Pair tokens use the formula of Section 4.3.

| Source | Tasks | Pairs per task | Max side p50 / p90 | Pair tokens p50 / p90 | How generated | Derives from |
| --- | --- | --- | --- | --- | --- | --- |
| ReARC `re_arc.zip` | 400 generators, 1000 examples each (400,000) | sampled | 23 / 29 | 534 / 1158 | one hand-written generator per task on the ARC-DSL; each example kept only if the task's hand-written verifier reproduces its output, input differs from output, and the input is new | the 400 ARC-AGI-1 training tasks, same IDs |
| BARC 100k gpt4-description | 102,895 problems | 4 to 59, median 30 | 10 / 18 | 188 / 486 | GPT-4 descriptions, GPT-4o-mini code (`main`, `generate_input`) with seed retrieval, 30 inputs run, auto-filtered | remixes of 100 seed programs for ARC-AGI-1 training tasks |
| BARC 100k gpt4omini-description | 102,522 | 4 to 59, median 30 | not sampled | not sampled | GPT-4o-mini for both stages | same 100 seeds |
| BARC 200k HEAVY | about 200k in two files (two code-generation approaches over the same ~110k GPT-4o descriptions) | 4 to 59, median 30 | 11 / 20 | 226 / 690 | as above | 162 seeds plus earlier BARC descriptions |
| ARC-AGI-1 training | 400, 1,718 pairs | 3 to 12 | 10 / 19 | 222 / 618 | human (Chollet 2019) | none |
| ARC-AGI-1 evaluation | 400, 1,782 pairs | 3 to 9 | 12 / 25 | 294 / 1062 | human | none |
| ARC-AGI-2 training | 1000, 4,308 pairs | 3 to 12 (train 2 to 10, test 1 to 4) | 11 / 22 | 226 / 846 | human | 391 ARC-AGI-1 training, 376 ARC-AGI-1 evaluation, 233 new |
| DEV120 (K0 counts, reference only) | 120 | train 2 to 6, test 1 to 3 | per task: 26 (p50), 70% of tasks above 20 | | human, calibrated | 6 ARC-AGI-1 evaluation tasks, rest new or reserves |

Facts behind the table:

- **ReARC.** `main.py` `generate_dataset` (lines 75 to 145): `random.seed(seed)` once, tasks in
  sorted order, the loop at lines 112 to 136 applies the verifier check, the non-degeneracy check,
  and input deduplication. `utils.py` `unifint` (lines 13 to 27) maps the difficulty interval
  $[d_{lb}, d_{ub}]$ to sizes and counts; `is_grid` (lines 30 to 50) bounds sides to 30. Generators
  randomize nuisance constants per example while keeping the defining ones (e.g.
  `generators.py` `generate_00d62c1b`, line 9575: background and object colors are sampled per
  example, the fill color 4 is fixed), so independent examples of one generator form a consistent
  task. `metadata.json` in the zip: 416,283 generations for 400,000 kept; total generation time
  2,063 s single process (p50 2.1 s per task, max 133 s for `447fd412`); verifier pass rate p5
  0.89. Defect: the 129 stored examples of `e26a3af2` have a side above 30 (up to 39); filter them.
- **BARC.** `generate_problems.py` `generate_problem` (lines 164 to 254) rejects a program whose
  output differs over 20 reruns, rejects one that is not equivariant under 20 random color
  permutations (the program's color constants permuted alike), and skips identity examples,
  all-black outputs, and ill-formed grids. Schema: `{"source": <Python>, "examples": [[input,
  output], ...], "seeds": [<seed file or description>, ...]}`; no problem ID, so use
  sha1 of `source`. The 160 hex seed IDs in `seeds/` are all ARC-AGI-1 training tasks (*measured*).
  HEAVY: the HF parquet conversion stores cells as float64, and 42 of 8,892 sampled pairs hold a
  grid with a side above 30; the raw JSONL holds ints (range read of its first 2.5 kB). Validate
  every grid.
- **ARC-AGI-2 training vs ARC-AGI-1** (*measured*): all 767 shared IDs keep the same train/test
  split; 766 have identical pair sets with the train pairs reordered; `ac0c5833` (ARC-AGI-1
  evaluation) changed pairs. The 233 new tasks share no exact or canonical pair with ARC-AGI-1.
  ARC-AGI-2 dropped 9 ARC-AGI-1 training tasks (`0dfd9992`, `29ec7d0e`, `3631a71a`, `40853293`,
  `73251a56`, `9ecd008a`, `a3df8b1e`, `c3f564a4`, `dc0a314f`), reason undocumented; one of them,
  `40853293`, shares three identical pairs with ARC-AGI-1 evaluation task `070dd51e`, which
  ARC-AGI-2 training keeps. It also dropped 24 ARC-AGI-1 evaluation tasks, 6 of which moved to the
  ARC-AGI-2 public evaluation set (Section 2.1). The technical report: "Tasks from the ARC-AGI-1
  Public Training set were excluded from testing" and "tasks previously publicly available
  remained public".

### 1.2 Conversion to the B4 schemas

- ARC JSON (`{"train": [{"input", "output"}], "test": [...]}`, test outputs present in training
  sets): pairs = train + test, each `Pair(input, output, kind="real", round_added=0,
  agreement=None)`.
- ReARC: one pool of `Pair`s per generator ID; a puzzle is a sample from the pool.
- BARC: one pool per problem (`examples`), ID `barc:<set>:<sha1(source)[:16]>`, `seeds` kept as
  provenance.
- `Episode(demos, queries, augmentation_id)` as in B4. The split and the leakage report need
  provenance that B4 does not carry, so store a record next to it: `{episode_id, source, task_id,
  family, split, tokens, n_demos}`. Keep B4 unchanged; the record is P3.4's.

### 1.3 ARC-AGI-1 evaluation: usable?

In principle yes (public, Apache-2.0); in practice no. 376 of its 400 tasks are in ARC-AGI-2
training already. Of the other 24, six are DEV120 tasks: `0934a4d8`, `136b0064`, `16b78196`,
`981571dc`, `aa4ec2a5`, `da515329` (three in DEV4, `136b0064` in DEV40). The remaining 18
(`08573cc6`, `1e97544e`, `3ed85e70`, `477d2879`, `47996f11`, `4aab4007`, `55783887`, `604001fa`,
`67b4a34d`, `79fb03f4`, `903d1b4a`, `929ab4e9`, `af22c60d`, `c663677b`, `ca8f78db`, `e66aafb8`,
`e95e3d8e`, `f4081712`) would add 18 tasks to the 989 ARC tasks of Section 4.4: not worth a
source whose files carry DEV120 tasks. Treat the ARC-AGI-1 evaluation directory, and any third-party file that bundles it, as
evaluation-adjacent data. One such file is in the ReARC repo: `arc_original.zip` holds ARC-AGI-1
training and evaluation, the 6 IDs included (by file name); `re_arc.zip` holds no evaluation task.

### 1.4 NVARC's synthetic data (optional, not in the first run)

`sorokin/nvarc-synthetic-puzzles` (103,253 puzzles, about 30 pairs each) and
`sorokin/nvarc-augmented-puzzles` (3.2M samples) are Kaggle datasets, so they could be attached
without a download. Not recommended now: license **unverified** (page unreadable, repo has none);
the "full" subset is built from ARC-AGI-2 evaluation summaries; the augmented set contains
`arc2_evaluation6`, the evaluation tasks with test outputs under random augmentations
(`ARChitects/README.md` lines 3 to 11, `SDG/scripts/build_datasets.py` lines 249 to 250); and
`color_mapping` (lines 35 to 38) recolors in place sequentially, which is not a permutation (36%
of 3-color grids lose a color in a *measured* simulation). Only the `nvarc_training` subset of the
un-augmented puzzles could be an ablation later.

## 2. Leakage

### 2.1 What the ARC-AGI-2 evaluation set shares with the sources

*Measured* on IDs (evaluation IDs from `data/evaluation.txt`, no content):

| Relation | Result |
| --- | --- |
| Evaluation IDs that are ReARC generators (= ARC-AGI-1 training) | none |
| Evaluation IDs that are BARC seeds | none (seeds are ARC-AGI-1 training) |
| Evaluation IDs in ARC-AGI-2 training | none |
| Evaluation IDs in ARC-AGI-1 evaluation | 6: `0934a4d8`, `136b0064`, `16b78196`, `981571dc`, `aa4ec2a5`, `da515329` |
| Pairs of those 6 (ARC-AGI-1 versions) in ARC-AGI-2 training, exact or canonical | none |

- **Generator.** No ReARC generator and no BARC seed corresponds to an evaluation task.
- **Concept.** ReARC (2024) and BARC (Nov 2024) predate ARC-AGI-2 (24 Mar 2025) and derive from
  ARC-AGI-1 training; ARC-AGI-2 training is the organizers' training split. Shared core-knowledge
  concepts are expected and permitted, and cannot be audited without looking at evaluation tasks,
  which this project does not do. NVARC "full" is the one source that derives from evaluation
  puzzles by construction, and the base model was trained on it (Section 4.6).
- **Pairs.** Only the Kaggle check (Section 2.3) can answer this. Local evidence that the sources
  do not share pairs with tasks they do not derive from (*measured*, exact / canonical pair
  matches): ReARC (first 250 examples per generator) vs ARC-AGI-1 evaluation 0 / 0 and vs the 233
  new ARC-AGI-2 tasks 0 / 0 (vs its own ARC-AGI-1 training tasks 1 / 38); the BARC sample vs
  ARC-AGI-1 evaluation and the new tasks 0 / 0 (vs ARC-AGI-1 training 1 / 6).

The task brief's premise that the evaluation set exists only on Kaggle is not literally true: it is
public in `arcprize/ARC-AGI-2` `data/evaluation` (with solutions), and six tasks are in ARC-AGI-1.
The project rule still holds as policy: the check runs on Kaggle, the dev box never fetches those
files.

### 2.2 Why an exact hash of grid pairs is not enough

1. **Our own augmentations.** Training, TTT, and decode use the notebook's group: transpose times
   $\mathrm{rot90}^k$ (8 dihedral maps) and a random permutation of all 10 colors, background
   included (`arc_loader.py` on `task/P0.1-extraction`: `permute_rnd_all_` lines 41 to 43,
   `augment` lines 255 to 262). A source holding any augmented copy of an evaluation pair passes
   an exact check, and every training-time augmentation of an episode would need its own exact
   check. A hash invariant under $D_4 \times S_{10}$ certifies all $8 \cdot 10!$ variants at once.
2. **Third-party sets ship augmented copies.** NVARC's `arc2_evaluation6` (Section 1.4) holds the
   evaluation tasks once plainly and five times under random dihedral and color maps; an exact
   check misses the five.
3. **Precedent inside ARC.** ARC-AGI-1 training `caa06a1f` and evaluation `50a16a69` share two pairs
   that differ only by a color permutation; the exact hash finds 3 shared pairs between ARC-AGI-1
   training and evaluation (`40853293`, `070dd51e`), the canonical hash finds 5 (*measured*).

What canonical hashing does not catch: pixel-edited variants (the ARC-AGI-2 changelog lists
one-pixel fixes to evaluation pairs), crops, scalings, and concepts. The grid gates and a
near-duplicate report narrow the first two; the third is the accepted residual risk.

### 2.3 Proposed check

Canonical pair key. Let $D_4$ act on positions and $S_{10}$ on colors; both act on a pair
$p = (x, y)$ jointly. Let $\rho(x, y)$ relabel colors $0, 1, 2, \dots$ in order of first appearance
in the row-major scan of $x$ followed by $y$, and let $\mathrm{ser}$ serialize both shapes and
cells. Then

$$\kappa(p) = \min_{g \in D_4} \mathrm{ser}\big(\rho(g x,\, g y)\big), \qquad
H_c(p) = \mathrm{sha256}\big(\kappa(p)\big).$$

$\rho(\pi x, \pi y) = \rho(x, y)$ for every $\pi \in S_{10}$, and $D_4$ commutes with $S_{10}$, so
$H_c(p) = H_c(p')$ if and only if $p' = (g, \pi) \cdot p$ for some $(g, \pi)$. The relabeling must be
joint: per-grid relabeling would merge pairs with different color correspondences. The grid key
$K_c(x)$ is the same construction on one grid; $H_e$ and $K_e$ are plain sha256 of the shapes and
cells (P3.2's `pair_hash` can serve as $H_e$ if it encodes shapes). Cost (*measured*): 0.13 to
0.15 ms per pair for $H_c$, 0.02 ms for $H_e$, single thread.

Evaluation side, inside the Kaggle checker only: $E_{tr}$, the 358 demonstration pairs of the 120
tasks (challenges file); $E_{te}$, the 172 test pairs (challenges plus solutions, read in that
notebook only, as A0 allows for the P3.4 hash check; never written out).

| Gate | Rule over every pair of every episode and validation puzzle | On a hit |
| --- | --- | --- |
| G0 provenance | the builder notebook has no competition data and fetches no evaluation file: ARC-AGI-2 `data/training` only, `re_arc.zip` only (never `arc_original.zip`), no ARC-AGI-1 | n/a |
| G1 ID | no source task ID in `arc26.data.DEV120` | drop the task |
| G2 exact pair | $H_e(p) \notin H_e(E_{tr} \cup E_{te})$ | drop the source task |
| G3 canonical pair | $H_c(p) \notin H_c(E_{tr} \cup E_{te})$ | drop the source task |
| G4 test inputs | for input grids with at least 25 cells and 2 colors, $K_e, K_c \notin$ keys of the 172 test inputs | drop the source task |
| G5 test outputs | the same for output grids against the 172 test outputs | drop the source task |
| R1 report | grid matches against all demonstration grids, by size bucket | count only |
| R2 report (optional) | input grids of equal shape after $D_4$ alignment with at most 2% cells differing under the best color bijection | count only |
| R3 report | ARC-AGI-2 training from GitHub vs the competition's training files, by canonical pair sets | count only |
| P positive control | three evaluation pairs, transformed by a random $(g, \pi)$, injected in memory into one shard copy | G3 must flag all three |

The builder applies a committed drop list (`configs/episodes.yaml`) and the checker must then
report zero for G1 to G5; the training notebook refuses an episode set whose manifest sha256 has no
passing report. Output: `run_logs/leakage.json` and `.md` with counts per gate, source, and split,
and `(source, task_id, eval_task_id, gate)` tuples: IDs only, never grids.

Grid-gate false positives, *measured* as matches of 80,000 ReARC grids against unrelated tasks
(true leakage is implausible there, so every hit is a false positive):

| Minimum cells, colors | vs ARC-AGI-1 evaluation, exact / canonical | vs 233 new ARC-AGI-2 tasks, exact / canonical |
| --- | --- | --- |
| 9, 2 | 38 / 681 | 5 / 262 |
| 25, 2 | 1 / 58 | 0 / 81 |
| 64, 2 | 0 / 14 | 0 / 44 |
| 100, 3 | 0 / 0 | 0 / 31 |
| 225, 3 | 0 / 0 | 0 / 0 |

So grid-level matching is a gate only with a size floor and a drop-the-task response. Scaled to 172
test inputs and about 600k episode grids, G4 and G5 at 25 cells should drop on the order of 20
source tasks (estimate), which costs nothing. Pair-level keys never matched between unrelated sets
in Section 2.1's comparisons; every pair match found was a real duplicate.

## 3. Getting it onto Kaggle

### 3.1 Two CPU notebooks, chained by output

| Notebook | Settings | Does |
| --- | --- | --- |
| `arc26-p3-4-episodes` (builder) | CPU, `enable_internet: true` (allowed for CPU pushes by ADR-008), no competition, no model, `--timeout 7200` | fetch pinned sources into `/kaggle/tmp`, validate, split, sample, write episodes, val and toy puzzles, `MANIFEST.json`, and `run_logs/` statistics to `/kaggle/working` |
| `arc26-p3-4-leakage` (checker) | CPU, `enable_internet: false`, `competition_sources: ["arc-prize-2026-arc-agi-2"]`, `kernel_sources: ["<owner>/arc26-p3-4-episodes/<version>"]` | Section 2.3; writes `run_logs/leakage.{json,md}` |

- **Why two.** The builder cannot read evaluation data (G0), and the notebook that does read it
  has no internet. Neither is a GPU run, so neither needs a pre-run brief (A9); neither slug may be
  named by a brief (gateway CPU slug rule).
- **Dataset or notebook output.** kagglesdk 0.1.37 `ApiCreateDatasetRequest` (local install,
  `kagglesdk/datasets/types/dataset_api_service.py` lines 136 to 165) creates datasets only from
  uploaded file tokens; no API call turns a notebook's output into a dataset. The website's
  Output tab has "New Dataset", and the dataset's "Update" syncs later versions (search snippet of
  Kaggle's datasets docs, **unverified**). Running `kaggle datasets version` inside the notebook
  would need API credentials in the session, which ADR-008 rule 6 excludes, and would fail
  `tests/test_repo_hygiene.py`'s Kaggle-access check. Recommendation: no dataset. Downstream
  notebooks attach the builder's output at a pinned version through `kernel_sources`
  (`owner/slug/version`, as in `kaggle_first_runs.md` 3.6) and resolve the mount by searching
  for `MANIFEST.json` (the 2026 mount layout is **unverified**). If the human wants a named
  dataset later, it is one click on the output.
- **Pulls.** `kaggle_ops.py pull` fetches `run_logs/*.json|md` and the session log only; episodes
  under `episodes/` never match the anchored pattern, so nothing large comes back.
- **Code.** A Kaggle script kernel has one code file. Build each push folder like
  `scripts/build_env_probe.py`: inline `arc26/hyper/episodes.py` and the script into one file plus
  `kernel-metadata.json`.
- **Images.** The latest CPU image (K0) has numpy 2.0.2, huggingface_hub 1.11.0 with hf-xet 1.4.3
  (the BARC files are Xet-backed), pyarrow 24.0.0, orjson, gitpython. Record the image in the
  manifest.

### 3.2 Sizes against limits

| Item | Size | Limit | Status of the limit |
| --- | --- | --- | --- |
| BARC JSONL, first set | 2.55 GB, into `/kaggle/tmp` | `/tmp` 1,119 GiB free | measured (K0) |
| `re_arc.zip` | 51 MB, read in place (724 MB uncompressed) | | |
| ARC-AGI-2, sparse `data/training` | about 4 MB | | |
| Episodes, 100k train plus val and toy | about 0.3 GB JSONL (2.8 kB per episode for the mix, from per-source sizes *measured* on samples); 25 to 50 MB gzipped (estimate; gzip kept 8 to 9% on the ReARC and BARC samples) | `/kaggle/working` 19.5 GiB | measured (K0) |
| Output files | about 30 (at most 16 shards per split) | about 500 per output | **unverified** (kaggle-cli issue 665, product-feedback posts) |
| RAM | under 4 GB if BARC is streamed line by line into uint8 arrays (estimate; Python int lists of the whole file would need tens of GB) | 31.3 GiB | measured (K0) |
| CPU time | builder under 30 min, checker under 15 min (estimate: parsing plus 0.15 ms per pair hash) | 12 h per session | **unverified** |
| A dataset, if made | about 0.05 GB | 200 GB per dataset and private quota | **unverified** (search snippet) |
| Internet in a notebook | needed by the builder | phone-verified account | the account already ran GPU sessions, which need the same verification |

### 3.3 Pins for `configs/episodes.yaml`

ReARC `re_arc.zip` alone at `e5b7f1d` (blob `2365d068`, by raw URL or a blobless checkout of that
one path) plus its sha256; ARC-AGI-2 `f3283f7`, sparse `data/training` only, assert
1000 files and `data/training.txt` equality; BARC `hf_hub_download(..., revision=<sha>)` plus the
LFS sha256 (table above); no ARC-AGI-1. The manifest records every input hash, the code commit,
the seed, and the image. Running ReARC's generators is optional: the zip is the generators' own
verified output at seed 42, and 1000 examples per generator are three and a half times what the
first pool needs. To regenerate (more data, other difficulty), vendor `generators.py`,
`verifiers.py`, `dsl.py`, `utils.py` at `e5b7f1d` with the MIT notice and seed each generator from
a stable integer (sha256 of the ID, not Python's salted `hash`); 1000 examples of every generator
took 2,063 s single process (`metadata.json`), so about 0.6 CPU-hours, less over 4 cores.

## 4. Episode design

### 4.1 Definition

An episode takes one task, draws $|D|$ demos and one query from its pairs, shuffles the demo
order, and applies one augmentation to every pair, so $D$ and $Q$ are consistent (Q6). For ARC
tasks the query cycles over all pairs (leave-one-out) and $D$ is a random subset of the rest, at
most 5. `augmentation_id` $= 8j + 4t + k$: transpose $t \in \{0, 1\}$, $k$ quarter turns, and
color permutation $j$ from a table stored in the manifest ($j = 0$ the identity, the rest seeded,
all 10 colors permuted as in `permute_rnd_all_`). Apply it with the notebook's
`ArcDataset.forward_mod` on a key such as `<task>.transpose.rot90.permute3810245679`, so TTT,
decode, and meta-training share one augmentation code path. Store un-augmented grids plus the id;
the data loader may redraw the id per epoch, and $H_c$ already covers every redraw.

### 4.2 Demo counts

DEV120 (K0): train pairs $\{2: 34, 3: 62, 4: 17, 5: 6, 6: 1\}$, test inputs
$\{1: 71, 2: 46, 3: 3\}$. With P2.1's hold-out (on for $k \ge 3$, flag default on for $k = 2$), H
sees $k - 1$ demos: $\{1: 34, 2: 62, 3: 17, 4: 6, 5: 1\}$. Sample generated episodes with
$P(|D|) = (0.15, 0.35, 0.30, 0.15, 0.05)$ for $|D| = 1, \dots, 5$: the DEV120 shape, flattened so
that 4 and 5 demos are not rare.

### 4.3 Token lengths

In the notebook format (`QwenFormatter.fmt_train`, `arc_loader.py` lines 58 to 71) a pair
renders as `<|im_start|>user\n{x}<|im_end|><|im_start|>assistant\n{y}<|im_end|>`. NVARC's
tokenizer (`ARChitects/qwen3_configs/tokenizer.json`: WordLevel over 10 digits, `\n`, `user`,
`assistant`, plus `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`; ids match `ARC_VOCAB` in
`arc_solver.py`) gives exactly

$$T(x, y) = (h_x w_x + h_x - 1) + (h_y w_y + h_y - 1) + 8,$$

with $h$ the row count after the episode's transform (*measured*: 0 mismatches over 2,000 random
pairs with the `tokenizers` library). A $30 \times 30$ pair is 1,866 tokens; four demos plus a
query 9,330 (B3's "about 9.5k"); five plus a query 11,196. Whether the Kaggle model ships the same
tokenizer files is **unverified** (the P3.2 brief has the script assert it).

*Measured* episode lengths with the design above (one random transpose per episode):

| Source (episodes simulated) | Mean | p50 | p90 | p99 | Max | Above 8192 | Episodes with max side above 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC-AGI-2 training, leave-one-out, at most 5 demos (4,308) | 1,537 | 1,105 | 3,442 | 6,984 | 9,330 | 0.23% | 22% |
| ReARC (4,000) | 2,160 | 1,998 | 3,724 | 5,404 | 7,462 | 0 | 93% |
| BARC gpt4-description (299) | 902 | 678 | 1,829 | 4,244 | 5,412 | 0 | 5% |
| BARC HEAVY (298) | 1,126 | 899 | 2,285 | 4,276 | 5,008 | 0 | 11% |

Keep the training max length at the notebook's 8192 and drop longer episodes with a count (B5);
the 50/30/20 mix loses about 0.05% and averages about 1,400 tokens. DEV120 grids are larger (70%
of tasks above side 20) than BARC's and ARC-AGI-2 training's; ReARC is the closest in size.

### 4.4 Tasks per source and episodes per task

Distinct tasks matter more than repeats (Q6). One pool of 100k training episodes, shuffled with
the sources interleaved, covers the 15k to 70k steps of B3 (one episode per GPU per step) without
reuse; a run of $N$ steps consumes the first $N$.

| Source | Train tasks | Episodes per task | Train episodes | Reserve |
| --- | --- | --- | --- | --- |
| BARC gpt4-description | 50,000 problems | 1 | 50,000 | 52,795 problems; the other two BARC sets |
| ReARC | 380 generators | about 79, disjoint examples (about 285 of 1000 per generator) | 30,000 | 715 examples per generator; regeneration |
| ARC-AGI-2 training | 989 tasks | about 20 (leave-one-out times augmentations) | 20,000 | more augmentations |
| ARC-AGI-1, NVARC | 0 | | 0 | Sections 1.3, 1.4 |

### 4.5 Validation split by generator family

A family is a ReARC generator together with the ARC task of the same ID, or one BARC problem.

- **100 ReARC puzzles** from 20 held-out families, 5 each: the 9 tasks ARC-AGI-2 dropped plus 11
  seeded picks among generators that are not BARC seeds. Remove all 20 generators from ReARC
  training and the 11 that exist in ARC-AGI-2 training from that source (989 left). Two of the 9
  (`0dfd9992`, `a3df8b1e`) are BARC seeds; BARC problems prompted with them stay in training.
- **100 BARC puzzles** from 100 held-out problems. BARC problems remix shared seeds, so
  concept-level near neighbors stay in training; the ReARC half is the clean unseen-generator test.
- Each puzzle draws its demo and test counts from DEV120's joint distribution and is written in the
  competition format (`val_challenges.json`, `val_solutions.json`), so P3.7 runs the DEV40 code
  path on it; validation episodes for P3.5's loss are its test pairs with all its demos.
- After the split, drop any training task with a canonical pair in a validation puzzle, then
  assert zero cross-split $H_c$ matches.
- **Toy split** for P3.5's acceptance: 200 training puzzles (100 BARC, 60 ReARC from distinct
  training generators, 40 ARC-AGI-2), 10 episodes each.

### 4.6 What NVARC trained its base on, and why it matters

`ARChitects/run_sft_4b.sh` (lines 9 to 33) and `sft_mg.yaml` (lines 118 to 127): full fine-tune of
Qwen3-4B-Thinking-2507 cut to 16 tokens, NeMo-RL Megatron, 1 epoch of 12,716 steps at global
batch 256, lr $10^{-4}$ linear to $10^{-7}$, 200 warmup steps, 4 nodes of 8 H100 for 27 h (paper
p. 6). Training data, paper Table 1 (p. 5) and `build_datasets.py` lines 144 to 275:

| Subset | Puzzles | Samples per puzzle | Share |
| --- | --- | --- | --- |
| MINI-ARC | 147 | 256 | 1.2% |
| ConceptARC | 160 | 256 | 1.3% |
| ReARC (2048 regenerated examples per task, 6 or 7 pairs per sample) | 400 | 256 | 3.2% |
| ARC-AGI-2 training minus the 391 ReARC IDs | 609 | 256 | 4.8% |
| NVARC training (synthetic, ARC-AGI-2 training summaries) | 47,337 | 24 | 34.8% |
| NVARC full (synthetic, training and evaluation summaries) | 55,886 | 32 | 54.9% |

Validation was `arc2_evaluation6` (`sft_mg.yaml` line 127), not trained on. The summaries behind
"full" include 29
hand-written and 91 LLM-written summaries of the 120 evaluation puzzles; the paper says "we
'leaked' the descriptions of these puzzles to our synthetic dataset" (p. 7). BARC was tested and
removed (p. 7).

Why it matters:

1. **Signal.** The base fit ReARC and ARC-AGI-2 training for 256 augmented passes per task. Its
   in-context query loss there is likely low, so the gradient into H is small and unlike test time,
   where the adapter matters because the base is unsure. BARC is new to the base; fresh ReARC
   instances are new samples of known concepts. Measure before the long run: zero-adapter query CE
   per source on 500 episodes each, one forward per episode (minutes on one L4), added to the P3.6
   throughput run. If ARC-AGI-2 training's CE is near zero, cut its share to 10%.
2. **Leakage.** The base's exposure to DEV120 concepts is fixed; every arm of P3.7 shares it, so
   comparisons between arms stay fair, but absolute DEV scores are optimistic for the hidden test.
   H must not add to it: no NVARC "full", no `arc2_evaluation6`.

## 5. Recommendation

### 5.1 First meta-training run

The Section 4.4 pool (50,000 BARC, 30,000 ReARC, 20,000 ARC-AGI-2 training), the Section 4.5
validation and toy splits, max length 8192, all six gates passing. A 15k-step run then sees 7.5k
BARC problems, 4.5k ReARC episodes over 380 generators, and 3k ARC episodes. Planned ablations,
each a filter or reweighting of the same pool: no ARC-AGI-2 training; BARC at 25%; plus 10% of the
NVARC training subset (after its license is read).

### 5.2 Phase 2 build plan

Files:

- `arc26/hyper/episodes.py` (numpy and stdlib only): `exact_pair_hash`, `canonical_pair_hash`,
  `exact_grid_hash`, `canonical_grid_hash`, `is_nontrivial(grid, min_cells=25, min_colors=2)`;
  `validate_grid` (2-D, rectangular, sides 1 to 30, integer-valued cells 0 to 9, floats accepted
  only if integral); `pair_tokens(x, y, transpose)`; `augmentation(aug_id)` and
  `descriptor(aug_id)` over the permutation table; loaders `load_arc_dir`, `load_rearc_zip`,
  `iter_barc_jsonl` (streaming); `split_families`; `sample_episodes`; `EpisodeRecord`,
  `write_shards`, `read_shards`, `validate_record`; `leakage_report(records, eval_challenges,
  eval_solutions)`.
- `scripts/build_episodes.py` with `configs/episodes.yaml` (pins, mix, seeds, $P(|D|)$, max
  length, gate thresholds, held-out families, drop list; every value commented). Modes `build`,
  `check`, `--smoke` (synthetic tasks and a synthetic "evaluation" fixture, never real data).
- `scripts/build_episodes_kernel.py`: the two push folders of Section 3.1.
- `tests/test_episodes.py`: $H_c$ invariant under all 8 dihedral maps times random color
  permutations, joint relabeling, shape sensitivity ($2 \times 3$ vs $3 \times 2$), no collision
  between non-equivalent random pairs; `pair_tokens` equals the token count of `QwenFormatter`
  output under the stand-in word-level tokenizer; record schema; family and canonical-pair
  disjointness of splits; every gate with positive and negative synthetic controls; the over-length
  drop counter; float and oversize grids rejected; same seed gives the same manifest hash.
- `docs/runbooks/build_episodes.md` (A8 sections), `docs/modules/episodes.md`.

Kaggle steps: push the builder (CPU, internet on); poll; pull `run_logs/`; push the checker with
the builder's version pinned; pull `leakage.json`; on any hit, add the IDs to the drop list,
rebuild, re-check. P3.5's notebook attaches both outputs and checks the manifest hash and `pass`.
B5's acceptance maps to: schema (`validate_record` over every record), leakage (G1 to G5 zero),
length histogram (`run_logs/episodes_stats.md`), drops (counted per source).

## What to reuse

- ReARC `re_arc.zip` as is; its generators only for regeneration (MIT, vendored at `e5b7f1d`).
- BARC HF JSONL `examples` and `seeds`; nothing from the GitHub repo (no license).
- ARC-AGI-2 `data/training` (Apache-2.0).
- Project: `arc_loader.py` `forward_mod`, `permute_mod`, `QwenFormatter`; P3.2's `pair_hash` as
  $H_e$ if it encodes shapes; `arc26/data.py` `DEV120` (on `task/P0.2-P0.3a-harness`) for G1.
- NVARC: ideas only (no license): degeneracy filters like `validate_pairs`
  (`build_datasets.py` lines 88 to 120), the Table 1 mix as a reference point.

## What to change

- B5 P3.4 leakage spec: exact plus canonical pair hashes, the G4 and G5 grid gates, drop and
  re-check, a positive control, and a separate internet-off checker notebook.
- B5 P3.4 sources: ARC-AGI-1 is not a separate source; NVARC data stays out of the first run.
- A0 wording: the evaluation set is public on GitHub; keeping it off the dev box is policy, and the
  ARC-AGI-1 evaluation directory falls under the same policy.
- Validation split by family (ReARC generator plus same-ID ARC task), not by task ID alone.

## Open risks

1. **Base already fit most sources.** H may learn near-zero adapters from ReARC and ARC-AGI-2
   training. The per-source CE measurement (4.6) decides the mix before the long run.
2. **Size mismatch.** DEV120 is large-grid; BARC is small-grid. If H underperforms on large
   grids, raise ReARC's share or regenerate ReARC with a higher difficulty floor.
3. **Encoder cost, for P3.2 and P3.5.** At P3.2's 16 views per demo pair, prefill of the demos
   (about 1,000 tokens per episode per view for this mix, to layer 24 of 36) costs about 2.5 times
   the training step, and caching saves nothing on a single pass over distinct pairs. For 100k
   episodes that is about 50 L4-hours at 16 views and about 6 at 2 views (estimate:
   $2 \cdot 4 \times 10^9$ FLOPs per token, two thirds of the layers, 40% of 121 TFLOPS). Reusing
   demo pairs across episodes or using fewer views during training are the levers.
4. **BARC quality.** LLM-written programs pass determinism and color-equivariance filters but can
   still be trivial or ambiguous; HEAVY has invalid grids and paired near-duplicates across its two
   files (same descriptions), which must share a split if HEAVY is used.
5. **Residual leakage.** Pixel-edited and concept-level overlap are not hash-detectable; the base
   already carries concept-level exposure through NVARC "full". DEV numbers are optimistic for
   every arm.
6. **Kaggle mechanics.** Mount path of chained outputs in 2026, the 500-file cap, the CPU session
   limit, and the competition training files' identity with GitHub's (R3) are **unverified** until
   the first builder and checker runs.
7. **Pretraining contamination.** Qwen3's web pretraining may include ARC-AGI-1 evaluation JSON,
   which holds three DEV4 tasks; not measurable.
