"""Evaluation subsets of the ARC-AGI-2 public evaluation set (BRIEF B4, task P0.2).

- ``DEV4``: the reference notebook's hardcoded dev filter (``starter.py``); parity tests only.
- ``DEV40``: a fixed seeded sample of 40 tasks, stratified by the largest grid side. It was
  computed once by :func:`select_dev40` from the K0 statistics committed as
  ``docs/research/eval_task_stats.json`` and is frozen here; ``tests/test_data.py`` recomputes it
  from that file. Never change it: every DEV40 score is comparable only on the same 40 tasks.
- ``DEV120``: the full public evaluation set, 120 tasks and 172 test outputs.
- ``DEV240`` (task P6.5b): DEV120 and a second copy of each task under :func:`dev240_copy_id`,
  for the 12 h timing check of a scored rerun's size; a copy is scored against its original's
  solutions (``DEV240_COPIES``) and reported apart. It is not in ``SUBSETS``: its copies are in
  no competition file, so the notebook writes them (``scripts/build_kaggle_notebook.py``,
  ``dev240_files``).

The challenge and solution loaders and the competition-rerun paths join this module with P0.1.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType

# BRIEF B4 schema: rows of cells, each cell a color in 0..9.
Grid = list[list[int]]

DEV4: tuple[str, ...] = ("0934a4d8", "36a08778", "981571dc", "aa4ec2a5")

# Sorted ids of arc-agi_evaluation_challenges.json (sha256 e7c62a4b..., K0 30 Sep 2026).
DEV120: tuple[str, ...] = (
    "0934a4d8", "135a2760", "136b0064", "13e47133", "142ca369", "16b78196", "16de56c4",
    "1818057f", "195c6913", "1ae2feb7", "20270e3b", "20a9e565", "21897d95", "221dfab4",
    "247ef758", "269e22fb", "271d71e2", "28a6681f", "291dc1e1", "2b83f449", "2ba387bc",
    "2c181942", "2d0172a1", "31f7f899", "332f06d7", "35ab12c3", "36a08778", "38007db0",
    "3a25b0d8", "3dc255db", "3e6067c3", "409aa875", "446ef5d2", "45a5af55", "4a21e3da",
    "4c3d4a41", "4c416de3", "4c7dc4dd", "4e34c42c", "53fb4810", "5545f144", "581f7754",
    "58490d8a", "58f5dbd5", "5961cc34", "5dbc8537", "62593bfd", "64efde09", "65b59efc",
    "67e490f4", "6e453dd6", "6e4f6532", "6ffbe589", "71e489b6", "7491f3cf", "7666fa5d",
    "78332cb0", "7b0280bc", "7b3084d4", "7b5033c1", "7b80bb43", "7c66cb00", "7ed72f31",
    "800d221b", "80a900e0", "8698868d", "88bcf3b4", "88e364bc", "89565ca0", "898e7135",
    "8b7bacbf", "8b9c3697", "8e5c0c38", "8f215267", "8f3a5a89", "9385bd28", "97d7923e",
    "981571dc", "9aaea919", "9bbf930d", "a251c730", "a25697e4", "a32d8b75", "a395ee82",
    "a47bf94d", "a6f40cea", "aa4ec2a5", "abc82100", "b0039139", "b10624e5", "b5ca7ac4",
    "b6f77b65", "b99e7126", "b9e38dc0", "bf45cf4b", "c4d067a0", "c7f57c3e", "cb2d8a2c",
    "cbebaa4b", "d35bdbdc", "d59b0160", "d8e07eb2", "da515329", "db0c5428", "db695cfb",
    "dbff022c", "dd6b8c4b", "de809cff", "dfadab01", "e12f9a14", "e3721c99", "e376de54",
    "e8686506", "e87109e9", "edb79dae", "eee78d87", "f560132c", "f931b4a8", "faa9f03d",
    "fc7cae8d",
)  # fmt: skip

# The project's one fixed seed (the P0.1 runs use PYTHONHASHSEED=0).
DEV40_SEED = 0
DEV40_SIZE = 40
# Upper edges of the max_side strata: [1, 15], [16, 20], [21, 25], [26, 29], [30]. Side 30 is its
# own stratum: it is the format maximum and the most common side (38 of 120 tasks). Stratum sizes
# on DEV120: 8, 28, 21, 25, 38; quotas 3, 9, 7, 8, 13.
DEV40_EDGES: tuple[int, ...] = (15, 20, 25, 29, 30)

# select_dev40 on docs/research/eval_task_stats.json, 30 Sep 2026: 40 tasks, 55 test outputs.
DEV40: tuple[str, ...] = (
    "136b0064", "16de56c4", "195c6913", "20a9e565", "28a6681f", "2ba387bc", "31f7f899",
    "332f06d7", "3e6067c3", "446ef5d2", "45a5af55", "4a21e3da", "581f7754", "5961cc34",
    "64efde09", "7b0280bc", "7b5033c1", "7c66cb00", "80a900e0", "8698868d", "89565ca0",
    "8f3a5a89", "9385bd28", "a25697e4", "b0039139", "b10624e5", "b5ca7ac4", "c7f57c3e",
    "cbebaa4b", "db0c5428", "db695cfb", "dbff022c", "dd6b8c4b", "de809cff", "dfadab01",
    "e12f9a14", "e3721c99", "e376de54", "eee78d87", "f560132c",
)  # fmt: skip

SUBSETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"DEV4": DEV4, "DEV40": DEV40, "DEV120": DEV120}
)


def dev240_copy_id(task_id: str) -> str:
    """Return the id of a task's DEV240 copy (task P6.5b).

    Why: a copy needs an id of the ARC form (8 lowercase hex digits), so every id parser takes
    it, that no real task has; a hash prefix of the original's id is fixed and needs no table.
    The notebook checks it against every id of the competition's files.

    Args:
        task_id: The original task's id.

    Returns:
        The first 8 hex digits of the sha256 of ``task_id`` (UTF-8).
    """
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]


# Each DEV240 copy's id to its original's (no copy id is a DEV120 id or another copy's:
# tests/test_shot_build.py).
DEV240_COPIES: Mapping[str, str] = MappingProxyType({dev240_copy_id(t): t for t in DEV120})
DEV240: tuple[str, ...] = tuple(sorted((*DEV120, *DEV240_COPIES)))


def subset_ids(name: str) -> tuple[str, ...]:
    """Return the task ids of a named evaluation subset.

    Args:
        name: ``DEV4``, ``DEV40``, or ``DEV120``.

    Returns:
        The subset's task ids, sorted.

    Raises:
        ValueError: If the name is not a known subset.
    """
    try:
        return SUBSETS[name]
    except KeyError:
        raise ValueError(f"unknown subset {name!r}; expected one of {', '.join(SUBSETS)}") from None


def stratify(max_side: Mapping[str, int], edges: Sequence[int]) -> list[list[str]]:
    """Group task ids into strata by their largest grid side.

    Stratum ``k`` holds the tasks with ``edges[k-1] < max_side <= edges[k]`` (``edges[-1]`` read
    as 0 for ``k = 0``).

    Args:
        max_side: Largest height or width over each task's grids, by task id.
        edges: Strictly increasing positive upper edges; the last one bounds every side.

    Returns:
        One sorted list of task ids per stratum, in edge order (a stratum may be empty).

    Raises:
        ValueError: If the edges are not strictly increasing positive integers, or a side is
            below 1 or above the last edge.
    """
    if not edges or edges[0] < 1 or any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"edges must be strictly increasing positive integers; got {edges!r}")
    strata: list[list[str]] = [[] for _ in edges]
    for task_id in sorted(max_side):
        side = max_side[task_id]
        if not 1 <= side <= edges[-1]:
            raise ValueError(f"{task_id}: max_side {side} outside [1, {edges[-1]}]")
        strata[next(k for k, edge in enumerate(edges) if side <= edge)].append(task_id)
    return strata


def allocate(sizes: Sequence[int], total: int) -> list[int]:
    """Split a sample size across strata in proportion to their sizes (largest remainder).

    Why: proportional allocation keeps the sample's size distribution equal to the population's
    up to rounding; the Hamilton method in exact integer arithmetic makes the rounding
    deterministic. Stratum ``k`` with ``n_k`` of ``N`` tasks gets ``floor(total * n_k / N)``;
    the units left go to the largest remainders, ties to the lower stratum.

    Args:
        sizes: Number of tasks in each stratum.
        total: Sample size, at most ``sum(sizes)``.

    Returns:
        The quota of each stratum, summing to ``total``; no quota exceeds its stratum's size.

    Raises:
        ValueError: If a size is negative, or ``total`` is negative or exceeds ``sum(sizes)``.
    """
    population = sum(sizes)
    if any(n < 0 for n in sizes) or not 0 <= total <= population:
        raise ValueError(f"cannot draw {total} from strata of sizes {list(sizes)}")
    if total == 0:
        return [0] * len(sizes)
    quotas = [total * n // population for n in sizes]
    remainders = [total * n % population for n in sizes]
    left = total - sum(quotas)
    for k in sorted(range(len(sizes)), key=lambda k: (-remainders[k], k))[:left]:
        quotas[k] += 1
    return quotas


def seeded_key(task_id: str, seed: int) -> str:
    """Return the sort key that orders tasks pseudo-randomly for a given seed.

    Why: a SHA-256 of the seed and the id is the same on every platform and Python version,
    whereas ``random.sample`` is not guaranteed stable across Python releases.

    Args:
        task_id: ARC task id.
        seed: Selection seed.

    Returns:
        The hex digest of ``"<seed>:<task_id>"``.
    """
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def select_dev40(
    max_side: Mapping[str, int],
    *,
    seed: int = DEV40_SEED,
    size: int = DEV40_SIZE,
    edges: Sequence[int] = DEV40_EDGES,
) -> tuple[str, ...]:
    """Draw a seeded sample of tasks, stratified by the largest grid side.

    Why: grid size drives sequence length, hence TTT and decode time and difficulty, so DEV40
    keeps the size distribution of DEV120 (proportional allocation over ``edges``). Within a
    stratum, tasks are taken in the order of :func:`seeded_key`.

    Args:
        max_side: Largest height or width over each task's grids, by task id (the K0 statistic
            ``max_side``).
        seed: Selection seed.
        size: Number of tasks to draw.
        edges: Upper edges of the strata, as in :func:`stratify`.

    Returns:
        The selected task ids, sorted.

    Raises:
        ValueError: As :func:`stratify` and :func:`allocate` raise it.
    """
    strata = stratify(max_side, edges)
    quotas = allocate([len(members) for members in strata], size)
    chosen: list[str] = []
    for members, quota in zip(strata, quotas, strict=True):
        chosen.extend(sorted(members, key=lambda task_id: seeded_key(task_id, seed))[:quota])
    return tuple(sorted(chosen))
