"""Checks arc26.hyper.episodes on synthetic grids only: no ARC data, no network, no Kaggle.

The notebook's own augmentation and format code (``ArcDataset.forward_mod``, ``augment``,
``QwenFormatter``) runs from cell 2 of the committed reference notebook
(``reference/reference_notebook.ipynb``, Apache License 2.0, attribution in
``reference/README.md``), with a stub for ``transformers``, which that cell imports only for a type
annotation. Token counts use a stand-in word-level tokenizer over NVARC's vocabulary (P3.2's
``smoke_tokenizer`` splits the same way); nothing is downloaded.
"""

import hashlib
import itertools
import json
import random
import re
import sys
import types
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from arc26.hyper import episodes as ep

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "reference" / "reference_notebook.ipynb"
# NVARC's tokenizer: special chat markers, then single digits, newlines, and the role words.
STAND_IN = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\d|\n|[a-z]+")
VOCAB = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "\n", "user", "assistant", *"0123456789"}
TABLE = ep.PermutationTable.seeded(32, 7)


@pytest.fixture(scope="module")
def nb():
    """The notebook's ``arc_loader`` module, executed from cell 2 of the reference notebook."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    magic, body = "".join(cells[2]["source"]).split("\n", 1)
    assert magic.strip() == "%%writefile arc_loader.py"
    stub = types.ModuleType("transformers")
    stub.AutoTokenizer = object
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = stub
    try:
        namespace = {"__name__": "arc_loader"}
        exec(compile(body, "arc_loader.py", "exec"), namespace)
    finally:
        if saved is None:
            del sys.modules["transformers"]
        else:
            sys.modules["transformers"] = saved
    return types.SimpleNamespace(**namespace)


def stand_in_tokens(text):
    tokens = STAND_IN.findall(text)
    assert "".join(tokens) == text, "text outside the stand-in vocabulary"
    assert set(tokens) <= VOCAB
    return len(tokens)


def grid(rng, rows, cols, colors=10):
    return rng.integers(0, colors, size=(rows, cols)).astype(np.uint8)


def random_pair(rng, low=1, high=12, colors=10):
    sides = rng.integers(low, high + 1, size=4)
    return grid(rng, sides[0], sides[1], colors), grid(rng, sides[2], sides[3], colors)


def record(pairs, task_id="t0000001", source="rearc", split="train", family=None, aug=0):
    pairs = [(ep.validate_grid(x), ep.validate_grid(y)) for x, y in pairs]
    return ep.EpisodeRecord.create(
        episode_id=f"{source}/{task_id}/{split}/0",
        source=source,
        task_id=task_id,
        family=family or task_id,
        split=split,
        augmentation_id=aug,
        demos=pairs[:-1] or pairs,
        queries=pairs[-1:],
    )


# Grid validation.


def test_validate_grid_accepts_integer_and_integral_float_cells():
    arr = ep.validate_grid([[0, 9], [3, 4]])
    assert arr.dtype == np.uint8 and arr.shape == (2, 2) and arr.flags.c_contiguous
    assert ep.validate_grid([[1.0, 2.0]]).tolist() == [[1, 2]]
    assert ep.validate_grid(np.arange(6, dtype=np.int64).reshape(2, 3)).tolist() == [
        [0, 1, 2],
        [3, 4, 5],
    ]
    ready = np.zeros((30, 30), dtype=np.uint8)
    assert ep.validate_grid(ready) is ready


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[]],
        [[1, 2], [3]],
        [[[1]]],
        5,
        "0123",
        [[0] * 31],
        [[0]] * 31,
        [[10]],
        [[-1]],
        [[1.5]],
        [[float("nan")]],
        [[float("inf")]],
        [[True, False]],
        [["1"]],
        [[None]],
        [[2**70]],
    ],
)
def test_validate_grid_rejects_without_quoting_cells(bad):
    with pytest.raises(ep.GridError) as info:
        ep.validate_grid(bad)
    assert "[[" not in str(info.value)


# Hashes.


def test_exact_pair_hash_is_p3_2s_pair_hash():
    # P3.2's docstring: {"input": [[1]], "output": [[2]]} hashes these bytes.
    expected = hashlib.sha256(b'{"input":[[1]],"output":[[2]]}').hexdigest()
    assert ep.exact_pair_hash([[1]], [[2]]) == expected
    assert ep.exact_pair_hash(np.array([[1]], dtype=np.int64), [[2.0]]) == expected
    assert ep.exact_grid_hash([[0, 1], [2, 3]]) == hashlib.sha256(b"[[0,1],[2,3]]").hexdigest()


@pytest.mark.parametrize("seed", range(5))
def test_canonical_hashes_are_invariant_under_the_notebooks_augmentations(nb, seed):
    rng = np.random.default_rng(seed)
    x, y = random_pair(rng, 1, 15)
    pair_key, grid_key = ep.canonical_pair_hash(x, y), ep.canonical_grid_hash(x)
    for transpose, turns in ep.DIHEDRAL:
        for _ in range(5):
            perm = "".join(map(str, rng.permutation(10)))
            key = "t." + ".".join(
                ["transpose"] * transpose + ["rot90"] * turns + ["permute" + perm]
            )
            mx, my = nb.ArcDataset.forward_mod(x, key), nb.ArcDataset.forward_mod(y, key)
            assert ep.canonical_pair_hash(mx, my) == pair_key
            assert ep.canonical_grid_hash(mx) == grid_key


def test_relabeling_is_joint_over_both_grids():
    # Each output alone is a recoloring of the other, but the color correspondence with the
    # input differs, so the pairs are not equivalent.
    assert ep.canonical_grid_hash([[1, 2]]) == ep.canonical_grid_hash([[2, 1]])
    assert ep.canonical_pair_hash([[1, 2]], [[1, 2]]) != ep.canonical_pair_hash([[1, 2]], [[2, 1]])
    assert ep.canonical_pair_hash([[1, 2]], [[1, 2]]) == ep.canonical_pair_hash([[5, 7]], [[5, 7]])


def test_shapes_are_part_of_every_hash():
    g = np.array([[1, 1, 2], [2, 2, 2]])
    reshaped = g.reshape(3, 2)  # same row-major cells, 3 by 2
    assert ep.exact_grid_hash(g) != ep.exact_grid_hash(reshaped)
    assert ep.canonical_grid_hash(g) != ep.canonical_grid_hash(reshaped)
    assert ep.canonical_grid_hash(g) == ep.canonical_grid_hash(g.T)
    # The same cell sequence split differently between input and output.
    a = ([[1, 2]], [[3]])
    b = ([[1]], [[2, 3]])
    assert ep.exact_pair_hash(*a) != ep.exact_pair_hash(*b)
    assert ep.canonical_pair_hash(*a) != ep.canonical_pair_hash(*b)


def equivalent(nb, p, q):
    """Brute force: is q = (g, pi) p for a dihedral map g and a color bijection pi?"""
    for transpose, turns in ep.DIHEDRAL:
        key = "t." + ".".join(["transpose"] * transpose + ["rot90"] * turns)
        moved = [nb.ArcDataset.forward_mod(g, key) for g in p] if key != "t." else list(p)
        if [m.shape for m in moved] != [g.shape for g in q]:
            continue
        forward, backward = {}, {}
        cells = zip(
            np.concatenate([m.ravel() for m in moved]),
            np.concatenate([g.ravel() for g in q]),
            strict=True,
        )
        if all(forward.setdefault(a, b) == b and backward.setdefault(b, a) == a for a, b in cells):
            return True
    return False


def test_canonical_hash_equality_is_exactly_augmentation_equivalence(nb):
    rng = np.random.default_rng(3)
    # Small grids with few colors, plus moved copies of 40 of them, so both outcomes occur often.
    pairs = [random_pair(rng, 1, 3, colors=3) for _ in range(120)]
    for x, y in pairs[:40]:
        transpose, turns = ep.DIHEDRAL[int(rng.integers(8))]
        perm = "".join(map(str, rng.permutation(10)))
        key = "t." + ".".join(["transpose"] * transpose + ["rot90"] * turns + ["permute" + perm])
        pairs.append((nb.ArcDataset.forward_mod(x, key), nb.ArcDataset.forward_mod(y, key)))
    keys = [ep.canonical_pair_hash(x, y) for x, y in pairs]
    same = 0
    for i, j in itertools.combinations(range(len(pairs)), 2):
        match = keys[i] == keys[j]
        assert match == equivalent(nb, pairs[i], pairs[j]), (i, j)
        same += match
    assert same >= 40


def test_no_collision_between_random_pairs():
    rng = np.random.default_rng(11)
    pairs = [random_pair(rng, 3, 10) for _ in range(3000)]
    assert len({ep.canonical_pair_hash(x, y) for x, y in pairs}) == 3000


def test_is_nontrivial_floors():
    assert ep.is_nontrivial(np.eye(5, dtype=np.uint8))
    assert not ep.is_nontrivial(np.eye(4, dtype=np.uint8))  # 16 cells
    assert not ep.is_nontrivial(np.zeros((30, 30), dtype=np.uint8))  # one color
    assert ep.is_nontrivial(np.eye(4, dtype=np.uint8), min_cells=16)


# Tokens.


def test_pair_tokens_equal_the_notebook_format_under_the_stand_in_tokenizer(nb):
    rng = np.random.default_rng(5)
    formatter = nb.QwenFormatter(None)
    for _ in range(150):
        pairs = [random_pair(rng, 1, 30) for _ in range(int(rng.integers(1, 5)))]
        aug_id = int(rng.integers(TABLE.n_augmentations))
        key = "t." + TABLE.descriptor(aug_id)
        train = [
            {
                "input": nb.ArcDataset.forward_mod(x, key),
                "output": nb.ArcDataset.forward_mod(y, key),
            }
            for x, y in pairs
        ]
        text = formatter.fmt_train(train, last_is_challenge=True)
        swap = TABLE.augmentation(aug_id).swaps_axes
        assert stand_in_tokens(text) == sum(ep.pair_tokens(x, y, swap) for x, y in pairs)


def test_token_counts_match_the_notes_worst_cases():
    big = np.zeros((30, 30), dtype=np.uint8)
    assert ep.pair_tokens(big, big) == 1866
    assert ep.episode_tokens([(big, big)] * 5) == 9330
    assert ep.episode_tokens([(big, big)] * 6) == 11196


def test_episode_tokens_take_the_longer_orientation():
    tall = np.zeros((10, 2), dtype=np.uint8)
    # Untransposed: 10 * 2 + 9 per grid; transposed: 2 * 10 + 1.
    assert ep.pair_tokens(tall, tall) == 2 * 29 + 8
    assert ep.pair_tokens(tall, tall, True) == 2 * 21 + 8
    assert ep.episode_tokens([(tall, tall)]) == 2 * 29 + 8


# Augmentations.


def test_descriptors_apply_exactly_like_forward_mod(nb):
    rng = np.random.default_rng(2)
    for aug_id in range(TABLE.n_augmentations):
        rows = int(rng.integers(1, 9))
        g = grid(rng, rows, rows + int(rng.integers(1, 5)))  # never square
        expected = nb.ArcDataset.forward_mod(g, "t." + TABLE.descriptor(aug_id))
        aug = TABLE.augmentation(aug_id)
        assert np.array_equal(aug.apply(g), expected)
        assert aug.swaps_axes == (expected.shape == g.shape[::-1])


def test_descriptors_spell_the_keys_of_the_notebooks_augment(nb):
    dataset = nb.ArcDataset(
        {"task": {"train": [{"input": [[1, 2]], "output": [[3]]}], "test": [{"input": [[4]]}]}}
    )
    keys = dataset.augment(n=1, seed=0).keys
    dihedral = set()
    for key in keys:
        ops = key.split(".")[1:-1]  # drop the task id and the trailing ex<order> op
        perm = ops[-1][len("permute") :]
        identity = tuple(range(10))
        perms = [identity] if perm == "0123456789" else [identity, tuple(map(int, perm))]
        table = ep.PermutationTable(tuple(perms))
        transpose, turns = "transpose" in ops, ops.count("rot90")
        aug_id = ep.augmentation_id(transpose, turns, len(perms) - 1)
        assert table.descriptor(aug_id) == ".".join(ops)
        dihedral.add((transpose, turns))
    assert dihedral == set(ep.DIHEDRAL)


def test_swaps_axes_and_ids():
    g = np.arange(6, dtype=np.uint8).reshape(2, 3)
    for transpose, turns in ep.DIHEDRAL:
        aug = ep.Augmentation(transpose, turns, tuple(range(10)))
        assert (aug.apply(g).shape == (3, 2)) == aug.swaps_axes
        assert TABLE.augmentation(ep.augmentation_id(transpose, turns, 3)) == ep.Augmentation(
            transpose, turns, TABLE.permutations[3]
        )
    with pytest.raises(ValueError):
        ep.augmentation_id(False, 4, 0)


def test_permutation_table_is_seeded_distinct_and_round_trips():
    table = ep.PermutationTable.seeded(64, 0)
    assert table == ep.PermutationTable.seeded(64, 0)
    assert table != ep.PermutationTable.seeded(64, 1)
    assert table.permutations[0] == tuple(range(10))
    assert len(set(table.permutations)) == 64
    assert ep.PermutationTable.from_strings(table.to_strings()) == table
    assert table.n_augmentations == 512
    for bad in (-1, 512, True):
        with pytest.raises(ValueError):
            table.augmentation(bad)
    with pytest.raises(ValueError):
        ep.PermutationTable.from_strings(["0123456789", "0123456788"])
    with pytest.raises(ValueError):
        ep.PermutationTable.from_strings(["1023456789"])  # the identity must come first
    with pytest.raises(ValueError):
        ep.PermutationTable.from_strings(["0123456789", "0123456789"])


# Loaders.


def write_task(path, train, test):
    path.write_text(json.dumps({"train": train, "test": test}), encoding="utf-8")


def test_load_arc_dir_reads_train_then_test_and_counts_invalid_pairs(tmp_path):
    write_task(
        tmp_path / "00000001.json",
        [{"input": [[1]], "output": [[2]]}],
        [{"input": [[3]], "output": [[4]]}],
    )
    write_task(
        tmp_path / "00000002.json",
        [{"input": [[0] * 31], "output": [[1]]}],
        [{"input": [[1]], "output": [[1]]}],
    )
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    tasks = ep.load_arc_dir(tmp_path)
    assert sorted(tasks) == ["00000001", "00000002"]
    first = tasks["00000001"]
    assert [(x.tolist(), y.tolist()) for x, y in first.pairs] == [([[1]], [[2]]), ([[3]], [[4]])]
    assert (first.source, first.family, first.examples, first.invalid_pairs) == (
        "arc2train",
        "00000001",
        2,
        0,
    )
    assert (tasks["00000002"].invalid_pairs, len(tasks["00000002"].pairs)) == (1, 1)
    (tmp_path / "00000003.json").write_text('{"train": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="00000003.json"):
        ep.load_arc_dir(tmp_path)


def test_load_rearc_zip_reads_task_members_only(tmp_path):
    path = tmp_path / "re_arc.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("re_arc/metadata.json", "{}")
        archive.writestr("__MACOSX/re_arc/tasks/._0000000a.json", "junk")
        archive.writestr(
            "re_arc/tasks/0000000a.json",
            json.dumps([{"input": [[1]], "output": [[2]]}, {"input": [[0] * 39], "output": [[2]]}]),
        )
        archive.writestr(
            "re_arc/tasks/0000000b.json", json.dumps([{"input": [[3]], "output": [[4]]}])
        )
    tasks = ep.load_rearc_zip(path)
    assert sorted(tasks) == ["0000000a", "0000000b"]
    assert (
        tasks["0000000a"].examples,
        tasks["0000000a"].invalid_pairs,
        len(tasks["0000000a"].pairs),
    ) == (2, 1, 1)
    assert tasks["0000000b"].family == "0000000b" and tasks["0000000b"].source == "rearc"


def test_iter_barc_jsonl_streams_validates_and_names_problems(tmp_path):
    rows = [
        {
            "source": "def a(): pass",
            "examples": [[[[1]], [[2]]], [[[3]], [[4]]], [[[5]], [[6]]]],
            "seeds": ["0a1b2c3d.py", "free text 12345678"],
        },
        {
            "source": "def b(): pass",
            "examples": [[[[1.0]], [[2.0]]], [[[0.5]], [[1]]], "bad"],
            "seeds": [],
        },
    ]
    path = tmp_path / "barc.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\nnot json\n", encoding="utf-8")
    malformed = Counter()
    tasks = list(ep.iter_barc_jsonl(path, "set", max_pairs=2, malformed=malformed))
    assert malformed["lines"] == 1
    first, second = tasks
    digest = hashlib.sha1(b"def a(): pass").hexdigest()[:16]
    assert first.task_id == first.family == f"barc:set:{digest}"
    assert (len(first.pairs), first.examples, first.seeds) == (2, 3, ("0a1b2c3d", "12345678"))
    assert (len(second.pairs), second.invalid_pairs) == (1, 2)
    assert second.pairs[0][0].tolist() == [[1]]


# Sampling.


def pool_task(n, rng, source="rearc", task_id="0000000a", low=2, high=6):
    return ep.SourceTask(
        source,
        task_id,
        task_id,
        [tuple(map(ep.validate_grid, random_pair(rng, low, high))) for _ in range(n)],
    )


def test_cycle_mode_queries_every_pair_once_before_repeating():
    task = pool_task(5, np.random.default_rng(0), source="arc2train")
    sampled = ep.sample_episodes(
        task,
        12,
        split="train",
        rng=random.Random(1),
        demo_probs=[0.2] * 5,
        n_augmentations=64,
        max_tokens=8192,
    )
    assert (sampled.over_length, sampled.short, len(sampled.records)) == (0, 0, 12)
    index = {ep.exact_pair_hash(x, y): i for i, (x, y) in enumerate(task.pairs)}
    queries = [index[ep.exact_pair_hash(*r.queries[0])] for r in sampled.records]
    assert sorted(queries[:5]) == [0, 1, 2, 3, 4] and sorted(queries[5:10]) == [0, 1, 2, 3, 4]
    for r, q in zip(sampled.records, queries, strict=True):
        demos = {index[ep.exact_pair_hash(x, y)] for x, y in r.demos}
        assert q not in demos and 1 <= len(demos) == r.n_demos <= 4
        assert 0 <= r.augmentation_id < 64
    assert len({r.episode_id for r in sampled.records}) == 12


def test_fresh_mode_uses_each_example_once_and_counts_shortfall():
    rng = np.random.default_rng(1)
    task = pool_task(20, rng)
    stream = ep.ExampleStream(20, random.Random(0))
    sampled = ep.sample_episodes(
        task,
        50,
        split="train",
        rng=random.Random(2),
        demo_probs=[0.2] * 5,
        n_augmentations=8,
        max_tokens=8192,
        stream=stream,
    )
    used = [ep.exact_pair_hash(x, y) for r in sampled.records for x, y in r.pairs]
    assert len(used) == len(set(used))
    assert sampled.short > 0 and len(sampled.records) + sampled.short == 50
    assert stream.remaining < 2


def test_demo_counts_are_truncated_to_the_pairs_available():
    rng = random.Random(0)
    assert {ep.draw_demo_count(rng, [0.15, 0.35, 0.3, 0.15, 0.05], 2) for _ in range(200)} == {1, 2}
    assert {ep.draw_demo_count(rng, [0.15, 0.35, 0.3, 0.15, 0.05], 9) for _ in range(500)} == {
        1,
        2,
        3,
        4,
        5,
    }
    with pytest.raises(ValueError):
        ep.draw_demo_count(rng, [1.0], 0)


def test_over_length_episodes_are_dropped_and_counted():
    big = np.ones((30, 30), dtype=np.uint8)
    task = ep.SourceTask("rearc", "0000000a", "0000000a", [(big, big)] * 6)
    only_five = [0, 0, 0, 0, 1]  # |D| = 5: six 30 by 30 pairs, 11,196 tokens
    sampled = ep.sample_episodes(
        task,
        7,
        split="train",
        rng=random.Random(0),
        demo_probs=only_five,
        n_augmentations=8,
        max_tokens=8192,
    )
    assert (len(sampled.records), sampled.over_length) == (0, 7)
    sampled = ep.sample_episodes(
        task,
        7,
        split="train",
        rng=random.Random(0),
        demo_probs=only_five,
        n_augmentations=8,
        max_tokens=11196,
    )
    assert (len(sampled.records), sampled.over_length) == (7, 0)


def test_sampling_is_reproducible():
    task = pool_task(9, np.random.default_rng(4), source="arc2train")
    runs = [
        [
            r.to_json()
            for r in ep.sample_episodes(
                task,
                6,
                split="toy",
                rng=ep.rng_for(0, "toy", task.task_id),
                demo_probs=[0.5, 0.5],
                n_augmentations=16,
                max_tokens=8192,
            ).records
        ]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_keyed_order_and_allocate():
    ids = [f"{i:08x}" for i in range(50)]
    order = ep.keyed_order(ids, 0, "x")
    assert order == ep.keyed_order(reversed(ids), 0, "x") and sorted(order) == ids
    assert order != ep.keyed_order(ids, 1, "x")
    assert [i for i in order if i != order[3]] == ep.keyed_order(set(ids) - {order[3]}, 0, "x")
    counts = ep.allocate(ids, 1234, 0, "y")
    assert sum(counts.values()) == 1234 and set(counts.values()) == {24, 25}
    assert ep.allocate([], 10, 0) == {}
    assert ep.derive_seed(0, "a") == ep.derive_seed(0, "a") != ep.derive_seed(0, "b")


def test_split_families():
    assert ep.split_families(["a", "b", "c"], ["b"]) == {"a": "train", "b": "val", "c": "train"}
    with pytest.raises(ValueError, match="z"):
        ep.split_families(["a"], ["z"])


# Records and shards.


def test_record_round_trip_and_b4_episode():
    rng = np.random.default_rng(6)
    rec = record([random_pair(rng, 2, 5) for _ in range(3)], aug=17)
    assert (
        ep.validate_record(json.loads(json.dumps(rec.to_json())), n_augmentations=64).to_json()
        == rec.to_json()
    )
    assert list(rec.to_json()) == list(ep.RECORD_KEYS)
    episode = rec.to_episode()
    assert isinstance(episode, ep.Episode) and episode.augmentation_id == 17
    assert len(episode.demos) == 2 and len(episode.queries) == 1
    pair = episode.queries[0]
    assert (pair.kind, pair.round_added, pair.agreement) == ("real", 0, None)
    assert pair.input == rec.queries[0][0].tolist()


def _mutations():
    def set_key(key, value):
        return lambda obj: obj.__setitem__(key, value)

    return {
        "missing key": lambda obj: obj.pop("family"),
        "extra key": set_key("note", 1),
        "unknown source": set_key("source", "arc1"),
        "unknown split": set_key("split", "test"),
        "empty id": set_key("episode_id", ""),
        "boolean augmentation": set_key("augmentation_id", True),
        "augmentation out of range": set_key("augmentation_id", 64),
        "negative augmentation": set_key("augmentation_id", -1),
        "n_demos mismatch": set_key("n_demos", 5),
        "tokens mismatch": set_key("tokens", 3),
        "no demos": set_key("demos", []),
        "no queries": set_key("queries", []),
        "bad grid": lambda obj: obj["demos"][0].__setitem__("input", [[1, 2], [3]]),
        "float grid": lambda obj: obj["demos"][0].__setitem__("input", [[0.5]]),
        "oversize grid": lambda obj: obj["queries"][0].__setitem__("output", [[1] * 31]),
        "extra pair key": lambda obj: obj["demos"][0].__setitem__("note", 1),
        "not an object": lambda obj: obj.clear(),
    }


@pytest.mark.parametrize("name", list(_mutations()))
def test_validate_record_rejects(name):
    rng = np.random.default_rng(8)
    obj = json.loads(json.dumps(record([random_pair(rng, 2, 5) for _ in range(3)]).to_json()))
    _mutations()[name](obj)
    with pytest.raises(ep.RecordError):
        ep.validate_record(obj, n_augmentations=64)


def test_validate_record_enforces_the_token_limit():
    obj = record([(np.ones((30, 30)), np.ones((30, 30)))] * 6).to_json()
    with pytest.raises(ep.RecordError, match="above the limit"):
        ep.validate_record(obj, max_tokens=8192)


def test_shards_keep_order_and_hash_their_bytes(tmp_path):
    rng = np.random.default_rng(9)
    records = [
        record([random_pair(rng, 1, 4) for _ in range(2)], task_id=f"{i:08x}") for i in range(10)
    ]
    entries = ep.write_shards(records, tmp_path, "train", 3)
    assert [e["path"] for e in entries] == [
        f"episodes/train-0000{i}-of-00003.jsonl" for i in range(3)
    ]
    assert [e["records"] for e in entries] == [3, 3, 4]
    for e in entries:
        data = (tmp_path / e["path"]).read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == (e["bytes"], e["sha256"])
    back = list(ep.read_shards([tmp_path / e["path"] for e in entries]))
    assert [r.to_json() for r in back] == [r.to_json() for r in records]
    shard = tmp_path / entries[1]["path"]
    shard.write_text(shard.read_text().replace('"split":"train"', '"split":"dev"', 1))
    with pytest.raises(ep.RecordError, match=r"train-00001-of-00003.jsonl:1"):
        list(ep.read_shards([shard]))
    assert ep.write_shards([], tmp_path / "empty", "val", 1)[0]["records"] == 0


# Leakage gates, on a synthetic evaluation set.


@pytest.fixture(scope="module")
def evaluation():
    rng = np.random.default_rng(100)
    challenges, solutions = {}, {}
    for i in range(4):
        train = [
            {"input": grid(rng, 6, 7).tolist(), "output": grid(rng, 5, 6).tolist()}
            for _ in range(2)
        ]
        tests = [(grid(rng, 6, 6), grid(rng, 7, 5))]
        if i == 3:
            tests = [(grid(rng, 4, 4), grid(rng, 4, 4))]  # 16 cells: under the grid-gate floor
        challenges[f"e000000{i}"] = {
            "train": train,
            "test": [{"input": x.tolist()} for x, _ in tests],
        }
        solutions[f"e000000{i}"] = [y.tolist() for _, y in tests]
    return challenges, solutions


PERM = (3, 8, 1, 0, 2, 4, 5, 6, 7, 9)


def moved(g, transpose=True, turns=1):
    return ep.Augmentation(transpose, turns, PERM).apply(g)


def gates_of(report, task_id):
    return {h["gate"] for h in report["hits"] if h["task_id"] == task_id}


def clean_records(n=4, seed=200):
    rng = np.random.default_rng(seed)
    return [
        record([random_pair(rng, 3, 9) for _ in range(3)], task_id=f"c000000{i}") for i in range(n)
    ]


def test_clean_records_pass_every_gate(evaluation):
    report = ep.leakage_report(clean_records(), *evaluation)
    assert report["pass"] and all(v["hits"] == 0 for v in report["gates"].values())
    assert report["eval"] == {"tasks": 4, "test_pairs": 4, "train_pairs": 8}
    assert report["drop"] == [] and report["hits"] == []


def test_g1_flags_evaluation_ids_and_seed_ids(evaluation):
    rng = np.random.default_rng(1)
    positive = record(
        [random_pair(rng, 3, 9) for _ in range(2)], task_id="e0000001", source="arc2train"
    )
    by_family = record(
        [random_pair(rng, 3, 9) for _ in range(2)], task_id="x0000001", family="e0000002"
    )
    report = ep.leakage_report(
        [positive, by_family, *clean_records()],
        *evaluation,
        seed_ids={"barc": ["e0000003", "00000000"]},
    )
    assert gates_of(report, "e0000001") == {"G1"} and gates_of(report, "x0000001") == {"G1"}
    assert gates_of(report, "seed:e0000003") == {"G1"}
    # Every hit enters the drop list, the seed hit too (as task seed:<id> of its source).
    assert {(d["source"], d["task_id"]) for d in report["drop"]} == {
        ("arc2train", "e0000001"),
        ("rearc", "x0000001"),
        ("barc", "seed:e0000003"),
    }
    assert not report["pass"]


def test_g2_and_g3_flag_an_exact_evaluation_pair_and_an_edit_escapes(evaluation):
    challenges, _ = evaluation
    pair = challenges["e0000000"]["train"][0]
    exact = record(
        [(pair["input"], pair["output"]), (pair["output"], pair["input"])], task_id="p0000001"
    )
    edited = np.array(pair["output"])
    edited[0, 0] = (edited[0, 0] + 1) % 10
    near = record([(pair["input"], edited), (pair["output"], pair["input"])], task_id="n0000001")
    report = ep.leakage_report([exact, near], *evaluation)
    assert gates_of(report, "p0000001") == {"G2", "G3"}
    assert gates_of(report, "n0000001") == set()


def test_g3_catches_a_moved_test_pair_that_g2_misses(evaluation):
    challenges, solutions = evaluation
    x, y = np.array(challenges["e0000001"]["test"][0]["input"]), np.array(solutions["e0000001"][0])
    leak = record(
        [(moved(x), moved(y)), random_pair(np.random.default_rng(2), 3, 9)],
        task_id="m0000001",
        source="barc",
    )
    other = record(
        [(moved(x), np.rot90(moved(y))), random_pair(np.random.default_rng(3), 3, 9)],
        task_id="o0000001",
    )
    report = ep.leakage_report([leak, other], *evaluation)
    assert {"G3", "G4", "G5"} <= gates_of(report, "m0000001") and "G2" not in gates_of(
        report, "m0000001"
    )
    assert "G3" not in gates_of(report, "o0000001")
    assert [
        h["match"] for h in report["hits"] if h["task_id"] == "m0000001" and h["gate"] == "G3"
    ] == ["canonical"]


def test_g4_flags_large_test_inputs_only(evaluation):
    challenges, _ = evaluation
    rng = np.random.default_rng(4)
    test_input = np.array(challenges["e0000002"]["test"][0]["input"])
    small_input = np.array(challenges["e0000003"]["test"][0]["input"])  # 16 cells
    demo_input = np.array(challenges["e0000000"]["train"][1]["input"])
    records = [
        record([(test_input, grid(rng, 5, 5)), random_pair(rng, 3, 9)], task_id="g4exact1"),
        record(
            [(moved(test_input, False, 2), grid(rng, 5, 5)), random_pair(rng, 3, 9)],
            task_id="g4canon1",
        ),
        record([(moved(small_input), grid(rng, 5, 5)), random_pair(rng, 3, 9)], task_id="g4small1"),
        record([(demo_input, grid(rng, 5, 5)), random_pair(rng, 3, 9)], task_id="g4demo01"),
    ]
    report = ep.leakage_report(records, *evaluation)
    assert gates_of(report, "g4exact1") == {"G4"} and gates_of(report, "g4canon1") == {"G4"}
    assert {h["match"] for h in report["hits"] if h["task_id"] == "g4exact1"} == {
        "exact",
        "canonical",
    }
    assert gates_of(report, "g4small1") == set() and gates_of(report, "g4demo01") == set()
    assert sum(report["reports"]["R1"]["exact"].values()) == 1  # the demonstration grid, counted


def test_g5_flags_large_test_outputs_and_skips_single_color_grids(evaluation):
    _, solutions = evaluation
    rng = np.random.default_rng(5)
    output = np.array(solutions["e0000002"][0])
    flat = np.zeros((6, 6), dtype=np.uint8)
    flat_eval = {
        "f0000000": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1]]}]}
    }
    records = [
        record([random_pair(rng, 3, 9), (grid(rng, 5, 5), moved(output))], task_id="g5canon1"),
        record([random_pair(rng, 3, 9), (grid(rng, 5, 5), flat)], task_id="g5flat01"),
    ]
    report = ep.leakage_report(records, *evaluation)
    assert gates_of(report, "g5canon1") == {"G5"} and gates_of(report, "g5flat01") == set()
    flat_report = ep.leakage_report(records[1:], flat_eval, {"f0000000": [flat.tolist()]})
    assert flat_report["pass"]  # equal, but one color: under the floor


def test_report_holds_ids_and_counts_never_grids(evaluation):
    challenges, solutions = evaluation
    x, y = np.array(challenges["e0000001"]["test"][0]["input"]), np.array(solutions["e0000001"][0])
    report = ep.leakage_report(
        [record([(moved(x), moved(y)), (x, y)], task_id="leak0001"), *clean_records()], *evaluation
    )

    def no_grids(value):
        if isinstance(value, dict):
            return all(no_grids(v) for v in value.values())
        if isinstance(value, list):
            return all(not isinstance(v, list) and no_grids(v) for v in value)
        return True

    assert no_grids(report)
    json.dumps(report)


def test_positive_control_is_caught_by_g3_and_missed_by_g2(evaluation):
    controls = ep.positive_control(*evaluation, n=3, seed=0)
    assert len(controls) == 3
    report = ep.leakage_report([*clean_records(), *(r for r, _ in controls)], *evaluation)
    outcome = ep.control_outcome(report, controls)
    assert outcome == {"injected": 3, "caught_g3": 3, "caught_g2": 0, "pass": True}
    assert ep.positive_control(*evaluation, n=3, seed=0)[0][0].to_json() == controls[0][0].to_json()
    with pytest.raises(ValueError):
        ep.positive_control(*evaluation, n=5, seed=0)


def test_positive_control_fails_when_the_canonical_gate_is_blind(evaluation, monkeypatch):
    controls = ep.positive_control(*evaluation, n=3, seed=0)
    monkeypatch.setattr(ep, "canonical_pair_hash", ep.exact_pair_hash)
    report = ep.leakage_report([*clean_records(), *(r for r, _ in controls)], *evaluation)
    assert not ep.control_outcome(report, controls)["pass"]


def test_split_overlap_and_canonical_matches(evaluation):
    rng = np.random.default_rng(7)
    shared = random_pair(rng, 3, 9)
    val = record(
        [random_pair(rng, 3, 9), shared], task_id="v0000001", split="val", family="fam00001"
    )
    train = record(
        [(moved(shared[0]), moved(shared[1])), random_pair(rng, 3, 9)],
        task_id="t0000002",
        family="fam00001",
    )
    report = ep.leakage_report([val, train, *clean_records()], *evaluation)
    assert report["reports"]["split_overlap"] == {
        "val_families_in_training": 1,
        "training_tasks_sharing_a_val_pair": 1,
    }
    assert ep.canonical_matches([shared], [train, *clean_records()]) == {("rearc", "t0000002")}


def test_pair_set_digests_compare_sets_not_order(evaluation):
    rng = np.random.default_rng(8)
    pairs = [tuple(map(ep.validate_grid, random_pair(rng, 2, 6))) for _ in range(4)]
    ours = ep.pair_set_digests({"a": pairs, "b": pairs, "c": pairs})
    theirs = ep.pair_set_digests(
        {
            "a": list(reversed(pairs)),
            "b": [(moved(x), moved(y)) for x, y in pairs],
            "c": pairs[:3],
            "d": pairs,
        }
    )
    result = ep.compare_pair_sets(ours, theirs)
    assert result == {
        "ours": 3,
        "theirs": 4,
        "common": 3,
        "only_ours": 0,
        "only_theirs": 1,
        "equal_exact": 1,
        "equal_canonical": 2,
        "canonical_differs": ["c"],
    }


def test_malformed_evaluation_files_raise_without_quoting_grids(evaluation):
    challenges, solutions = evaluation
    with pytest.raises(ValueError, match="different tasks"):
        ep.leakage_report([], challenges, {})
    broken = {**solutions, "e0000000": []}
    with pytest.raises(ValueError, match="e0000000") as info:
        ep.leakage_report([], challenges, broken)
    assert "[[" not in str(info.value)
