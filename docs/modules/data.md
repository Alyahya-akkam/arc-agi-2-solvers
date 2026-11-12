# Module: `arc26/data.py`

Task P0.2 (BRIEF B4, B5). Tests: `tests/test_data.py`. Pure Python, no GPU.

## Purpose

Defines the evaluation subsets of the ARC-AGI-2 public evaluation set, frozen as tuples of task
ids, and the function that drew DEV40. The challenge and solution loaders and the
competition-rerun paths join this module with P0.1.

| Subset | Tasks | Test outputs | One task in points | Use |
|---|---|---|---|---|
| `DEV4` | 4 | 5 | 25 | parity with the reference notebook only |
| `DEV40` | 40 | 55 | 2.5 | dev comparisons |
| `DEV120` | 120 | 172 | 0.83 | full public evaluation set |

Differences under 2 tasks on DEV40 are noise (`CLAUDE.md`, code conventions).

## DEV40

DEV40 is a seeded sample of 40 tasks, stratified by the largest grid side, drawn once from the K0
statistics and frozen in `DEV40`. It never changes: DEV40 scores compare only on the same tasks.
`tests/test_data.py` recomputes it from the committed statistics and must get the frozen tuple.

**Statistic.** For task $t$ with grids $G_t$ (train inputs, train outputs, test inputs), the
largest side is $s_t = \max_{g \in G_t} \max(h_g, w_g)$, measured by K0 on Kaggle
(`scripts/env_probe.py`, `evaluation_stats`). The committed file
[`docs/research/eval_task_stats.json`](../research/eval_task_stats.json) holds only
$s_t$ (`max_side`) and the number of test inputs per task, with the sha256 of the source file
(`arc-agi_evaluation_challenges.json`, `e7c62a4b...`) and the K0 provenance. No grids.

**Strata.** Upper edges $e = (15, 20, 25, 29, 30)$: stratum $k$ holds the tasks with
$e_{k-1} < s_t \le e_k$ ($e_0 = 0$). Side 30 is its own stratum: it is the format maximum and the
most common side (38 of 120 tasks). Grid size drives sequence length, hence TTT and decode time
and difficulty, so DEV40 keeps the size distribution of DEV120.

**Allocation.** Proportional, by the largest-remainder (Hamilton) method in exact integers: with
$n_k$ tasks in stratum $k$, $N = 120$, and sample size $M = 40$, stratum $k$ first gets
$q_k = \lfloor M n_k / N \rfloor$; the $M - \sum_k q_k$ units left go to the largest remainders
$M n_k \bmod N$, ties to the lower stratum.

| Stratum | $s_t$ | Tasks in DEV120 | Quota $q_k$ |
|---|---|---|---|
| 1 | 1 to 15 | 8 | 3 |
| 2 | 16 to 20 | 28 | 9 |
| 3 | 21 to 25 | 21 | 7 |
| 4 | 26 to 29 | 25 | 8 |
| 5 | 30 | 38 | 13 |

**Seed and order.** Seed `DEV40_SEED = 0`, the project's one fixed seed (the P0.1 runs use
`PYTHONHASHSEED=0`). Within a stratum, tasks are sorted by
$\mathrm{SHA256}(\texttt{"<seed>:<task\_id>"})$ in hex and the first $q_k$ are taken. A hash is the
same on every platform and Python version, whereas `random.sample` is not guaranteed stable across
Python releases.

**Result.** 40 tasks, 55 test outputs (25 tasks with one test output, 15 with two); mean
$s_t$ 24.2 against 24.5 on DEV120. No DEV4 task was drawn.

## Interfaces

| Name | What |
|---|---|
| `Grid` | `list[list[int]]`, the B4 schema |
| `DEV4`, `DEV40`, `DEV120` | frozen, sorted tuples of task ids |
| `SUBSETS` | read-only mapping from `DEV4`, `DEV40`, `DEV120` to the tuples |
| `subset_ids(name)` | the tuple of a named subset; `ValueError` on an unknown name |
| `DEV40_SEED`, `DEV40_SIZE`, `DEV40_EDGES` | the selection's parameters (0, 40, the edges above) |
| `select_dev40(max_side, *, seed, size, edges)` | the seeded stratified draw, sorted |
| `stratify(max_side, edges)` | task ids grouped by stratum |
| `allocate(sizes, total)` | largest-remainder quotas |
| `seeded_key(task_id, seed)` | the SHA-256 sort key |

## Data flow

K0 probe on Kaggle (`runs/K0-latest/run_logs/env_probe.json`, key `evaluation_stats`, identical in
`K0-pinned`) -> `docs/research/eval_task_stats.json` (task id to `max_side` and `test_inputs`) ->
`select_dev40` -> the frozen `DEV40`. `arc26.evaluate` and the notebook's `DEV_SUBSET` setting
(BRIEF A5) read the subsets through `SUBSETS` and `subset_ids`.

## Origin

`DEV4` is the dev filter hardcoded in the reference notebook's `starter.py` cell
(`reference/reference_notebook.ipynb`, cell 5: `key not in ["0934a4d8", "36a08778", "981571dc",
"aa4ec2a5"]`). Nothing else here comes from the notebook.

## Tests

`tests/test_data.py`: DEV40 recomputed from the committed statistics and pinned by the sha256 of
its sorted, comma-joined ids (so a new seed with a regenerated tuple fails); the stats file's
provenance and fields; subsets sorted, unique, nested, of the right sizes; DEV120 equal to the
statistics' task ids; strata sizes and quotas; the seed changes the draw but input order does
not; the documented hash; edge and allocation checks, including quotas that never exceed their
stratum.

## Known limits

- Stratified on the largest side only, not on the number of test outputs or train pairs; DEV40
  has no task with three test outputs (DEV120 has 3).
- The sample is small: one task is 2.5 points, and two runs of the same code differ by 1 to 2
  points on the full set (NVARC paper, section 3.5).
