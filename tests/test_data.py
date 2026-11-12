"""DEV subsets and the DEV40 selection of arc26.data (P0.2, BRIEF B4)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from arc26 import data
from arc26.data import (
    DEV4,
    DEV40,
    DEV40_EDGES,
    DEV40_SEED,
    DEV120,
    SUBSETS,
    allocate,
    seeded_key,
    select_dev40,
    stratify,
    subset_ids,
)

REPO = Path(__file__).resolve().parent.parent
STATS = REPO / "docs" / "research" / "eval_task_stats.json"
SOURCE_SHA256 = "e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8"
# sha256 of the sorted DEV40 ids joined by commas, fixed when DEV40 was drawn (30 Sep 2026). A new
# seed, edges, or statistics with a regenerated tuple still passes the recompute test; not this.
DEV40_SHA256 = "c3cac4aeafbb5b00c442a4f7db26ec528142971a14f1ca849df85889b073689e"


def stats():
    return json.loads(STATS.read_text(encoding="utf-8"))


def max_side():
    return {task_id: v["max_side"] for task_id, v in stats()["per_task"].items()}


def test_stats_file_holds_only_the_needed_statistics_with_provenance():
    committed = stats()
    assert committed["source"]["sha256"] == SOURCE_SHA256
    assert committed["source"]["file"] == "arc-agi_evaluation_challenges.json"
    assert "K0" in committed["source"]["measured_by"]
    per_task = committed["per_task"]
    assert len(per_task) == committed["summary"]["tasks"] == 120
    assert all(set(v) == {"max_side", "test_inputs"} for v in per_task.values())
    assert sum(v["test_inputs"] for v in per_task.values()) == 172
    assert max(v["max_side"] for v in per_task.values()) == 30


def test_dev40_is_recomputed_exactly_from_the_committed_statistics():
    assert select_dev40(max_side()) == DEV40


def test_dev40_is_pinned_by_its_digest():
    assert hashlib.sha256(",".join(sorted(DEV40)).encode()).hexdigest() == DEV40_SHA256
    assert (DEV40_SEED, data.DEV40_SIZE, DEV40_EDGES) == (0, 40, (15, 20, 25, 29, 30))


def test_subsets_are_sorted_unique_and_nested():
    assert DEV4 == ("0934a4d8", "36a08778", "981571dc", "aa4ec2a5")
    for subset, size in ((DEV4, 4), (DEV40, 40), (DEV120, 120)):
        assert len(subset) == size and len(set(subset)) == size
        assert list(subset) == sorted(subset)
    assert set(DEV4) <= set(DEV120) and set(DEV40) <= set(DEV120)
    assert DEV120 == tuple(sorted(stats()["per_task"]))


def test_dev40_keeps_the_size_distribution_of_dev120():
    sides = max_side()
    strata = stratify(sides, DEV40_EDGES)
    assert [len(members) for members in strata] == [8, 28, 21, 25, 38]
    chosen = set(DEV40)
    assert [len(chosen & set(members)) for members in strata] == [3, 9, 7, 8, 13]
    per_task = stats()["per_task"]
    assert sum(per_task[task_id]["test_inputs"] for task_id in DEV40) == 55


def test_dev40_depends_on_the_seed_but_not_on_input_order():
    sides = max_side()
    reordered = dict(reversed(list(sides.items())))
    assert select_dev40(reordered) == DEV40
    other = select_dev40(sides, seed=DEV40_SEED + 1)
    assert other != DEV40
    strata = stratify(sides, DEV40_EDGES)
    assert [len(set(other) & set(m)) for m in strata] == [3, 9, 7, 8, 13]


def test_seeded_key_is_the_documented_sha256():
    assert seeded_key("0934a4d8", 0) == hashlib.sha256(b"0:0934a4d8").hexdigest()


def test_stratify_uses_upper_edges():
    sides = {"a": 1, "b": 15, "c": 16, "d": 20, "e": 21, "f": 29, "g": 30}
    assert stratify(sides, DEV40_EDGES) == [["a", "b"], ["c", "d"], ["e"], ["f"], ["g"]]


@pytest.mark.parametrize("edges", [(), (0, 30), (15, 15, 30), (20, 15, 30)])
def test_stratify_refuses_bad_edges(edges):
    with pytest.raises(ValueError, match="strictly increasing"):
        stratify({"a": 5}, edges)


@pytest.mark.parametrize("side", [0, 31])
def test_stratify_refuses_sides_outside_the_edges(side):
    with pytest.raises(ValueError, match="outside"):
        stratify({"a": side}, DEV40_EDGES)


def test_allocate_is_proportional_with_largest_remainders():
    assert allocate([8, 28, 21, 25, 38], 40) == [3, 9, 7, 8, 13]
    assert allocate([30, 60, 30], 40) == [10, 20, 10]
    # Equal remainders go to the lower stratum.
    assert allocate([1, 1, 1], 2) == [1, 1, 0]
    assert allocate([5, 0, 5], 0) == [0, 0, 0]
    assert allocate([2, 3], 5) == [2, 3]


def test_allocate_never_exceeds_a_stratum_and_sums_to_the_total():
    sizes = [1, 2, 3, 5, 8, 13, 21]
    for total in range(sum(sizes) + 1):
        quotas = allocate(sizes, total)
        assert sum(quotas) == total
        assert all(0 <= q <= n for q, n in zip(quotas, sizes, strict=True))


@pytest.mark.parametrize(("sizes", "total"), [([3, 4], 8), ([3, 4], -1), ([3, -1], 1)])
def test_allocate_refuses_impossible_draws(sizes, total):
    with pytest.raises(ValueError, match="cannot draw"):
        allocate(sizes, total)


def test_select_dev40_takes_whole_strata_when_asked_for_everything():
    sides = max_side()
    assert select_dev40(sides, size=120) == DEV120


def test_subset_ids_and_the_read_only_registry():
    assert subset_ids("DEV4") == DEV4 and subset_ids("DEV40") == DEV40
    assert subset_ids("DEV120") == DEV120
    assert tuple(SUBSETS) == ("DEV4", "DEV40", "DEV120")
    with pytest.raises(ValueError, match="unknown subset 'DEV5'"):
        subset_ids("DEV5")
    with pytest.raises(TypeError):
        SUBSETS["DEV5"] = ()


def test_frozen_tuples_are_tuples():
    assert all(isinstance(ids, tuple) for ids in (data.DEV4, data.DEV40, data.DEV120))
    assert Counter(len(task_id) for task_id in DEV120) == Counter({8: 120})
