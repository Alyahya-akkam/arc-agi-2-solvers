"""Tests for arc26/hyper/encoder.py (task P3.2, BRIEF B2 Q1).

The model is the task brief's tiny random Qwen3 and the tokenizer the stand-in with the notebook's
16 ids, both built in code by scripts/cache_embeddings.py (smoke_model, smoke_tokenizer); nothing
is downloaded. Without torch, transformers, or safetensors the module is skipped. Run with
CUDA_VISIBLE_DEVICES="" (BRIEF A0); the one GPU test is marked and never runs locally.
"""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from tokenizers import normalizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from arc26.hyper import encoder as enc_mod  # noqa: E402
from arc26.hyper.encoder import (  # noqa: E402
    DIHEDRAL,
    EmbeddingCache,
    PairEncoder,
    apply_view,
    augmented_views,
    pair_hash,
    render_pair,
    view_descriptors,
    view_set_id,
)
from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location(
        "cache_embeddings", REPO / "scripts" / "cache_embeddings.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()
LAYERS = 4  # smoke_model's depth


def grid(rng, rows, cols):
    return rng.integers(0, 10, size=(rows, cols)).tolist()


def make_pairs(shapes, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"input": grid(rng, *shape_in), "output": grid(rng, *shape_out)}
        for shape_in, shape_out in shapes
    ]


SMALL = [((3, 4), (2, 2)), ((1, 1), (1, 1)), ((5, 2), (2, 5)), ((4, 4), (6, 3))]


@pytest.fixture(scope="module")
def tokenizer():
    return script.smoke_tokenizer()


@pytest.fixture(scope="module")
def model():
    return script.smoke_model(seed=0)


@pytest.fixture(scope="module")
def formatter(tokenizer):
    return QwenFormatter(tokenizer)


# --- stand-in tokenizer -------------------------------------------------------------------------


def test_standin_tokenizer_has_the_notebook_ids(tokenizer):
    # ARC_VOCAB and the token ids of arc26/nvarc/arc_solver.py (that module imports unsloth).
    ids = tokenizer.encode("<|im_start|>user\n12\n34<|im_end|><|im_start|>assistant\n5<|im_end|>")
    assert ids == [14, 11, 10, 1, 2, 10, 3, 4, 15, 14, 12, 10, 5, 15]
    assert tokenizer.pad_token_id == 13 and tokenizer.eos_token_id == 15
    assert tokenizer.decode([1, 2, 10, 3, 4]) == "12\n34"


def test_standin_tokenizer_refuses_text_outside_the_format(tokenizer):
    with pytest.raises(Exception, match="WordLevel"):
        tokenizer.encode("1 2")


# --- pair_hash ----------------------------------------------------------------------------------


def test_pair_hash_is_sha256_of_the_documented_encoding():
    expected = hashlib.sha256(b'{"input":[[1]],"output":[[2]]}').hexdigest()
    assert pair_hash({"input": [[1]], "output": [[2]]}) == expected


def test_pair_hash_ignores_container_and_integer_dtype():
    pair = make_pairs([((3, 4), (2, 2))])[0]
    reference = pair_hash(pair)
    for dtype in (np.int8, np.uint8, np.int32, np.int64):
        as_arrays = {k: np.asarray(v, dtype=dtype) for k, v in pair.items()}
        assert pair_hash(as_arrays) == reference
    assert pair_hash({"output": pair["output"], "input": pair["input"], "extra": 1}) == reference


def test_pair_hash_separates_different_pairs():
    pair = make_pairs([((3, 4), (2, 3))])[0]
    swapped = {"input": pair["output"], "output": pair["input"]}
    transposed = {k: np.transpose(v).tolist() for k, v in pair.items()}
    hashes = {pair_hash(p) for p in (pair, swapped, transposed)}
    assert len(hashes) == 3
    # Same cells, other shape: the row structure is part of the encoding.
    flat = {"input": [sum(pair["input"], [])], "output": pair["output"]}
    assert pair_hash(flat) != pair_hash(pair)


@pytest.mark.parametrize(
    "bad",
    [
        [[1, 2], [3]],  # ragged
        [1, 2, 3],  # 1-D
        [[[1]]],  # 3-D
        [[]],  # empty
        [[10]],  # not a color
        [[-1]],  # not a color
        [[1.0]],  # not an integer
        [[True]],  # not an integer
    ],
)
def test_pair_hash_rejects_invalid_grids(bad):
    with pytest.raises(ValueError):
        pair_hash({"input": bad, "output": [[0]]})


# --- views --------------------------------------------------------------------------------------


def test_view_descriptors_equal_the_notebook_augment_and_leave_numpy_alone():
    pair = make_pairs([((3, 4), (2, 3))])[0]
    before = np.random.get_state()
    views = view_descriptors(2, 2)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    state = np.random.get_state()
    try:
        dataset = ArcDataset(queries={"t": {"train": [pair], "test": [{"input": pair["input"]}]}})
        augmented = dataset.augment(n=2, seed=2)
    finally:
        np.random.set_state(state)
    # augment's keys are "t.<descriptor>.ex<order>"; shuffle_ex only reorders pairs.
    descriptors = [key.split(".", 1)[1].rsplit(".ex", 1)[0] for key in augmented.keys]
    assert views == descriptors and len(views) == 16
    for key, view in zip(augmented.keys, views, strict=True):
        expected = augmented.queries[key]["train"][0]
        got = apply_view(pair, view)
        assert np.array_equal(got["input"], expected["input"])
        assert np.array_equal(got["output"], expected["output"])


def test_default_views_are_the_views_run_a_decoded_on_kaggle():
    # Run A of P0.1-01 named its decode outputs <task>_<i>.<descriptor>.ex<order>. For a task
    # with one test input, augment(n=2, seed=2) draws exactly the encoder's 16 permutations.
    fixture = REPO / "tests" / "fixtures" / "p0_1_reference_candidates_A.json"
    names = json.loads(fixture.read_text())
    for output in ("981571dc_0", "aa4ec2a5_0"):
        decoded = {
            name.split(".", 1)[1].rsplit(".ex", 1)[0]
            for name in names
            if name.startswith(f"{output}.")
        }
        assert decoded == set(view_descriptors())


def test_view_descriptors_shapes_and_ids():
    assert view_descriptors(0) == list(DIHEDRAL)
    assert len(view_descriptors(1)) == 8 and len(view_descriptors(3)) == 24
    assert view_descriptors(2, 2) == view_descriptors()
    assert view_descriptors(2, 3) != view_descriptors(2, 2)
    assert view_set_id(view_descriptors()) != view_set_id(view_descriptors(2, 3))
    assert len(view_set_id(view_descriptors())) == 16
    with pytest.raises(ValueError):
        view_descriptors(-1)


def test_augmented_views_are_the_dihedral_group_times_color_permutations():
    pair = make_pairs([((2, 3), (3, 2))])[0]
    views = dict(augmented_views(pair))
    assert len(views) == 16
    identity = apply_view(pair, "")
    assert np.array_equal(identity["input"], pair["input"])
    quarter = apply_view(pair, "rot90")
    assert np.array_equal(quarter["output"], np.rot90(np.asarray(pair["output"])))
    # A color permutation maps cell values and keeps the geometry.
    perm = apply_view(pair, "permute9876543210")
    assert np.array_equal(perm["input"], 9 - np.asarray(pair["input"]))


# --- render_pair --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape_in", "shape_out"), [((1, 1), (1, 1)), ((3, 4), (2, 2)), ((30, 30), (30, 30))]
)
def test_render_pair_positions_decode_to_the_grids(formatter, tokenizer, shape_in, shape_out):
    pair = make_pairs([(shape_in, shape_out)])[0]
    rendered = render_pair(pair, formatter)
    ids = rendered.token_ids
    assert ids == tokenizer.encode(formatter.fmt_train([pair]))
    for name, span in (("output", rendered.output_positions), ("input", rendered.input_positions)):
        text = tokenizer.decode(ids[span.start : span.stop])
        rows = [[int(c) for c in line] for line in text.split("\n")]
        assert rows == pair[name]
    # The output span starts right after "assistant\n" and ends at the closing <|im_end|>.
    assert ids[rendered.output_positions.start - 2 : rendered.output_positions.start] == [12, 10]
    assert ids[rendered.output_positions.stop :] == [15]
    assert ids[: rendered.input_positions.start] == [14, 11, 10]
    rows_in, rows_out = shape_in[0], shape_out[0]
    cells = shape_in[0] * shape_in[1] + shape_out[0] * shape_out[1]
    assert len(ids) == 8 + cells + (rows_in - 1) + (rows_out - 1)


def test_a_30_by_30_pair_is_1866_tokens(formatter):
    # BRIEF B3: a 30 by 30 grid is 930 tokens with its newlines; a pair about 1.9k.
    pair = make_pairs([((30, 30), (30, 30))])[0]
    assert len(render_pair(pair, formatter).token_ids) == 1866


class _AppendsEnd:
    """A tokenizer that appends an end token to every encode, so prefixes stop being prefixes."""

    def __init__(self, inner):
        self.inner = inner

    def encode(self, text):
        return self.inner.encode(text) + [15]

    def decode(self, ids):
        return self.inner.decode(ids)


class _MisreadsOnes:
    """A tokenizer whose decode reads every 1 as a 2."""

    def __init__(self, inner):
        self.inner = inner

    def encode(self, text):
        return self.inner.encode(text)

    def decode(self, ids):
        return self.inner.decode(ids).replace("1", "2")


def test_render_pair_refuses_a_tokenizer_that_breaks_the_prefix_property(tokenizer):
    pair = {"input": [[1, 2]], "output": [[3]]}
    with pytest.raises(ValueError, match="not a prefix"):
        render_pair(pair, QwenFormatter(_AppendsEnd(tokenizer)))


def test_render_pair_refuses_spans_that_do_not_decode_back(tokenizer):
    pair = {"input": [[0, 2]], "output": [[1]]}
    with pytest.raises(ValueError, match="output positions do not decode"):
        render_pair(pair, QwenFormatter(_MisreadsOnes(tokenizer)))


def stripping_tokenizer(tokenizer):
    """The stand-in with a normalizer that strips trailing whitespace from every text segment.

    The prefix that ends after "user\\n" then tokenizes without its newline, and its tokens are
    still a prefix of the pair's, so the spans start one token early, on the newline. The
    notebook's decoder strips that newline again; only the exact text check sees it.
    """
    inner = copy.deepcopy(tokenizer.backend_tokenizer)
    inner.normalizer = normalizers.Strip(left=False, right=True)
    return PreTrainedTokenizerFast(
        tokenizer_object=inner, eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )


def test_render_pair_refuses_spans_that_hold_a_neighboring_newline(tokenizer):
    stripping = stripping_tokenizer(tokenizer)
    pair = {"input": [[1, 2], [3, 4]], "output": [[5, 6, 7]]}
    ids = stripping.encode(QwenFormatter(stripping).fmt_train([pair]))
    assert ids == tokenizer.encode(QwenFormatter(tokenizer).fmt_train([pair]))
    assert len(stripping.encode("<|im_start|>user\n")) == 2  # the newline is gone
    with pytest.raises(ValueError, match="exactly the input grid text"):
        render_pair(pair, QwenFormatter(stripping))


# --- PairEncoder: the forward -------------------------------------------------------------------


def padded_batch(formatter, pairs):
    rendered = [render_pair(p, formatter) for p in pairs]
    width = max(len(r.token_ids) for r in rendered)
    ids = torch.full((len(rendered), width), 13, dtype=torch.long)
    mask = torch.zeros((len(rendered), width), dtype=torch.long)
    for row, r in enumerate(rendered):
        ids[row, : len(r.token_ids)] = torch.tensor(r.token_ids)
        mask[row, : len(r.token_ids)] = 1
    return rendered, ids, mask


@pytest.mark.parametrize("layer", range(LAYERS))
def test_layer_truncation_equals_the_full_forward(model, tokenizer, formatter, layer):
    _, ids, mask = padded_batch(formatter, make_pairs(SMALL))
    encoder = PairEncoder(model, tokenizer, layer=layer)
    with torch.no_grad():
        truncated = encoder.hidden_states(ids, mask)
        full = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[layer]
    assert torch.equal(truncated, full)


def test_no_later_layer_norm_or_head_runs(model, tokenizer, formatter):
    layer = 2
    calls = []
    watched = [*model.model.layers[layer:], model.model.norm, model.lm_head]
    handles = [
        module.register_forward_pre_hook(lambda m, a, i=i: calls.append(i))
        for i, module in enumerate(watched)
    ]
    try:
        encoder = PairEncoder(model, tokenizer, layer=layer, views=["", "rot90"])
        encoder.encode_views(make_pairs(SMALL[:2]))
        assert calls == []
        # Sanity: the watch works on a full forward.
        _, ids, mask = padded_batch(formatter, make_pairs(SMALL[:1]))
        with torch.no_grad():
            model(ids, attention_mask=mask)
        assert len(calls) == len(watched)
    finally:
        for handle in handles:
            handle.remove()


def test_encoder_leaves_no_hooks(model, tokenizer):
    encoder = PairEncoder(model, tokenizer, layer=3, views=[""])
    encoder.encode_views(make_pairs(SMALL[:1]))
    assert all(not m._forward_hooks for m in model.modules())


def test_view_embedding_is_the_mean_over_the_pooled_positions(model, tokenizer, formatter):
    pairs = make_pairs(SMALL[:3])
    views = ["", "transpose.rot90.permute3012456789"]
    for pool in ("output", "input"):
        encoder = PairEncoder(model, tokenizer, layer=2, pool=pool, views=views)
        got = encoder.encode_views(pairs)
        assert got.shape == (3, 2, 64) and got.dtype == torch.float32
        for i, pair in enumerate(pairs):
            for a, view in enumerate(views):
                rendered = render_pair(apply_view(pair, view), formatter)
                span = rendered.output_positions if pool == "output" else rendered.input_positions
                with torch.no_grad():
                    hidden = model(
                        torch.tensor([rendered.token_ids]), output_hidden_states=True
                    ).hidden_states[2][0]
                expected = hidden[span.start : span.stop].mean(dim=0)
                torch.testing.assert_close(got[i, a], expected, atol=1e-6, rtol=0)


def test_input_and_output_pooling_differ(model, tokenizer):
    pairs = make_pairs(SMALL[:2])
    out = PairEncoder(model, tokenizer, layer=2, pool="output").encode_pairs(pairs)
    inp = PairEncoder(model, tokenizer, layer=2, pool="input").encode_pairs(pairs)
    assert not torch.allclose(out, inp)


def test_batched_equals_one_at_a_time(model, tokenizer):
    pairs = make_pairs(SMALL)
    batched = PairEncoder(model, tokenizer, layer=2, batch_tokens=10**6).encode_views(pairs)
    single = PairEncoder(model, tokenizer, layer=2, batch_tokens=1).encode_views(pairs)
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=0)


def test_padding_never_pooled(model, tokenizer, monkeypatch):
    pairs = make_pairs(SMALL)
    encoder = PairEncoder(model, tokenizer, layer=2, batch_tokens=10**6)
    clean = encoder.encode_views(pairs)
    original = encoder.hidden_states
    padded = []

    def poisoned(input_ids, attention_mask):
        hidden = original(input_ids, attention_mask).clone()
        hidden[attention_mask == 0] = float("nan")
        padded.append(int((attention_mask == 0).sum()))
        return hidden

    monkeypatch.setattr(encoder, "hidden_states", poisoned)
    dirty = encoder.encode_views(pairs)
    assert sum(padded) > 0, "the batch must contain padding for this test to mean anything"
    assert torch.equal(dirty, clean)


def test_non_finite_embeddings_raise(model, tokenizer, monkeypatch):
    encoder = PairEncoder(model, tokenizer, layer=2, views=[""])
    monkeypatch.setattr(
        encoder, "hidden_states", lambda ids, mask: torch.full((*ids.shape, 64), float("inf"))
    )
    with pytest.raises(FloatingPointError):
        encoder.encode_views(make_pairs(SMALL[:1]))


# --- PairEncoder: task embedding ----------------------------------------------------------------


def test_task_embedding_is_invariant_to_pair_order(model, tokenizer):
    pairs = make_pairs(SMALL, seed=3)
    encoder = PairEncoder(model, tokenizer, layer=2, batch_tokens=2048)
    z = encoder.task_embedding(pairs)
    for order in ([3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]):
        shuffled = encoder.task_embedding([pairs[i] for i in order])
        torch.testing.assert_close(shuffled, z, atol=1e-5, rtol=0)
    assert z.shape == (64,)


def test_k_equals_2(model, tokenizer):
    pairs = make_pairs(SMALL[:2])
    encoder = PairEncoder(model, tokenizer, layer=1)
    per_view = encoder.encode_views(pairs)
    assert per_view.shape == (2, 16, 64)
    pair_embeddings = encoder.encode_pairs(pairs)
    torch.testing.assert_close(pair_embeddings, per_view.mean(dim=1), atol=1e-6, rtol=0)
    z = encoder.task_embedding(pairs)
    torch.testing.assert_close(z, pair_embeddings.mean(dim=0), atol=1e-6, rtol=0)


def test_30_by_30_pairs_encode(model, tokenizer):
    pairs = make_pairs([((30, 30), (30, 30)), ((30, 30), (1, 1))])
    encoder = PairEncoder(model, tokenizer, layer=2, batch_tokens=4096)
    per_view = encoder.encode_views(pairs)
    assert per_view.shape == (2, 16, 64) and torch.isfinite(per_view).all()


def test_timing_records_follow_the_b4_schema(model, tokenizer, formatter):
    records = []
    pairs = make_pairs(SMALL[:2])
    encoder = PairEncoder(model, tokenizer, layer=2, timing=records.append)
    encoder.encode_views(pairs, puzzle_id="abcd1234")
    assert [r["op"] for r in records] == ["encode_render", "encode_prefill"]
    tokens = sum(
        len(render_pair(apply_view(p, v), formatter).token_ids)
        for p in pairs
        for v in encoder.views
    )
    for record in records:
        assert set(record) == {"puzzle_id", "op", "seconds", "tokens"}
        assert record["puzzle_id"] == "abcd1234" and record["tokens"] == tokens
        assert record["seconds"] >= 0


def test_encoder_settings_are_validated(model, tokenizer):
    for bad in (
        {"layer": LAYERS},
        {"layer": -1},
        {"pool": "pair"},
        {"batch_tokens": 0},
        {"views": []},
        {"views": ["flip"]},
        {"views": ["permute0123"]},
    ):
        with pytest.raises(ValueError):
            PairEncoder(model, tokenizer, **bad)
    with pytest.raises(TypeError):
        PairEncoder(torch.nn.Linear(2, 2), tokenizer)
    with pytest.raises(ValueError):
        PairEncoder(model, tokenizer, layer=1).encode_views([])


# --- EmbeddingCache -----------------------------------------------------------------------------


def hashes_of(pairs):
    return [pair_hash(p) for p in pairs]


def open_cache(root, layer=24):
    return EmbeddingCache(
        root, model_id="m/test", layer=layer, pool="output", view_set_id=view_set_id(["a"])
    )


def random_entries(n, views=3, hidden=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    keys = hashes_of(make_pairs([((1, i + 1), (1, 1)) for i in range(n)], seed=seed))
    per_view = {k: torch.randn(views, hidden, generator=gen) for k in keys}
    return keys, {k: v.mean(dim=0) for k, v in per_view.items()}, per_view


def test_cache_round_trip_is_exact_and_survives_reopening(tmp_path):
    keys, means, views = random_entries(5)
    cache = open_cache(tmp_path)
    assert cache.missing(keys) == keys
    assert cache.put(means, views) == 5
    for reopened in (cache, open_cache(tmp_path)):
        assert len(reopened) == 5 and reopened.missing(keys) == []
        assert torch.equal(reopened.get(keys), torch.stack([means[k] for k in keys]))
        assert torch.equal(
            reopened.get_views(keys[::-1]), torch.stack([views[k] for k in keys[::-1]])
        )
        assert reopened.hidden_size == 8 and reopened.n_views == 3


def test_cache_put_skips_known_keys_and_resumes(tmp_path):
    keys, means, views = random_entries(6)
    first = open_cache(tmp_path)
    first.put({k: means[k] for k in keys[:3]}, {k: views[k] for k in keys[:3]})
    resumed = open_cache(tmp_path)
    assert resumed.missing(keys + keys[:2]) == keys[3:]
    assert resumed.put(means, views) == 3
    assert resumed.put(means, views) == 0
    shards = sorted(p.name for p in resumed.path.glob("shard-*.safetensors"))
    assert shards == ["shard-000000.safetensors", "shard-000001.safetensors"]
    assert resumed.size_bytes == sum((resumed.path / s).stat().st_size for s in shards)


def test_cache_means_without_views(tmp_path):
    keys, means, _ = random_entries(2)
    cache = open_cache(tmp_path)
    cache.put(means)
    assert torch.equal(cache.get(keys[1:]), means[keys[1]].unsqueeze(0))
    assert not cache.has_views(keys[0])
    with pytest.raises(KeyError):
        cache.get_views(keys)


def test_cache_namespaces_are_separate(tmp_path):
    keys, means, views = random_entries(2)
    open_cache(tmp_path, layer=24).put(means, views)
    other = open_cache(tmp_path, layer=18)
    assert other.path != open_cache(tmp_path, layer=24).path
    assert other.missing(keys) == keys
    record = json.loads((other.path / "namespace.json").read_text())
    assert record["layer"] == 18 and record["model_id"] == "m/test"
    (other.path / "namespace.json").write_text(json.dumps({**record, "layer": 30}))
    with pytest.raises(ValueError, match="another namespace"):
        open_cache(tmp_path, layer=18)


def test_cache_refuses_bad_entries(tmp_path):
    keys, means, views = random_entries(2)
    cache = open_cache(tmp_path)
    with pytest.raises(ValueError, match="not a pair hash"):
        cache.put({"abc": means[keys[0]]})
    with pytest.raises(ValueError, match="exactly the keys"):
        cache.put(means, {keys[0]: views[keys[0]]})
    with pytest.raises(ValueError, match="non-finite"):
        cache.put({keys[0]: torch.full((8,), float("nan"))})
    cache.put({keys[0]: means[keys[0]]}, {keys[0]: views[keys[0]]})
    with pytest.raises(ValueError, match="width"):
        cache.put({keys[1]: torch.zeros(9)})
    with pytest.raises(ValueError, match="views"):
        cache.put({keys[1]: means[keys[1]]}, {keys[1]: torch.zeros(4, 8)})
    with pytest.raises(KeyError):
        cache.get([keys[1]])
    assert cache.get([]).shape == (0, 8)


def test_cache_write_is_atomic_when_the_manifest_write_fails(tmp_path, monkeypatch):
    keys, means, views = random_entries(4)
    cache = open_cache(tmp_path)
    cache.put({k: means[k] for k in keys[:2]}, {k: views[k] for k in keys[:2]})
    manifest_before = (cache.path / "manifest.json").read_bytes()
    real_replace = os.replace

    def failing_replace(src, dst):
        if Path(dst).name == "manifest.json":
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(enc_mod.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        cache.put(means, views)
    monkeypatch.setattr(enc_mod.os, "replace", real_replace)
    assert (cache.path / "manifest.json").read_bytes() == manifest_before
    assert not list(cache.path.glob(".*.tmp"))
    assert len(cache) == 2 and cache.missing(keys) == keys[2:]
    reopened = open_cache(tmp_path)
    assert reopened.missing(keys) == keys[2:]
    # The orphan shard from the failed put is overwritten by the next one.
    assert reopened.put(means, views) == 2
    assert torch.equal(reopened.get(keys), torch.stack([means[k] for k in keys]))


def test_cache_stores_encoder_output_exactly(tmp_path, model, tokenizer):
    pairs = make_pairs(SMALL)
    encoder = PairEncoder(model, tokenizer, layer=2)
    per_view = encoder.encode_views(pairs)
    keys = hashes_of(pairs)
    cache = EmbeddingCache(
        tmp_path, model_id="smoke", layer=2, pool="output", view_set_id=encoder.view_set_id
    )
    cache.put(
        {k: e.mean(dim=0) for k, e in zip(keys, per_view, strict=True)},
        dict(zip(keys, per_view, strict=True)),
    )
    assert torch.equal(cache.get_views(keys), per_view)
    assert torch.equal(cache.get(keys), per_view.mean(dim=1))
    torch.testing.assert_close(
        cache.get(keys).mean(dim=0), encoder.task_embedding(pairs), atol=1e-6, rtol=0
    )


# --- GPU ----------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_sdpa_on_cuda_matches_the_full_forward_and_batches(tokenizer, formatter):
    model = script.smoke_model(seed=0, attn_implementation="sdpa").to("cuda", torch.bfloat16)
    pairs = make_pairs(SMALL)
    _, ids, mask = padded_batch(formatter, pairs)
    encoder = PairEncoder(model, tokenizer, layer=2, batch_tokens=10**6)
    with torch.no_grad():
        truncated = encoder.hidden_states(ids.cuda(), mask.cuda())
        full = model(ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True)
    assert torch.equal(truncated, full.hidden_states[2])
    batched = encoder.encode_views(pairs)
    single = PairEncoder(model, tokenizer, layer=2, batch_tokens=1).encode_views(pairs)
    torch.testing.assert_close(batched, single, atol=2e-2, rtol=2e-2)
