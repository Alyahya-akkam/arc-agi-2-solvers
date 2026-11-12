"""P3.3: hypernetwork H on Qwen3-4B's shapes (meta device) and on a tiny random Qwen3 (CPU).

Skipped when torch, transformers, peft, or safetensors is missing: CI installs `.[dev]` only.
Nothing touches the Hugging Face Hub: models come from `Qwen3Config`, adapters from `tmp_path`.
`pytest -rP` shows the parameter count and the CPU forward time (docs/modules/hypernet.md).
"""

import copy
import dataclasses
import math
import os
import time
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

import yaml  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from arc26.hyper.hypernet import PROJECTIONS, BaseShapes, Hypernet, HypernetConfig  # noqa: E402
from arc26.hyper.inject import inject, rslora_scaling, save_adapter, wrap_targets  # noqa: E402

CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "hypernet.yaml"
CONFIG = HypernetConfig.from_yaml(CONFIG_FILE)
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
BUDGET = 150_000_000
# Qwen/Qwen3-4B config.json (sha 1cfa9a72, research note section 3). The fine-tuned base is
# assumed to differ only in its 16-token vocabulary, which H does not touch.
QWEN3_4B = Qwen3Config(
    vocab_size=151936,
    hidden_size=2560,
    intermediate_size=9728,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    tie_word_embeddings=True,
)
# The default scope (r = 16, alpha = 32, q, k, v, o), with H's widths shrunk for a tiny base.
TINY = dataclasses.replace(CONFIG, embed_dim=24, latent=32, head_in=16)
DTYPES = pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])


def tiny_qwen3(dtype=torch.float32):
    """Random Qwen3 with Qwen3-4B's shape relations: explicit head_dim, GQA, d_out != d_in."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        initializer_range=0.2,  # logits of order 1, so a changed adapter shows in bf16
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model.requires_grad_(False)
    return model


def prepared(config=TINY, dtype=torch.float32):
    """A tiny base wrapped for the config's targets, an unwrapped copy, its keys, and H."""
    model = tiny_qwen3(dtype)
    reference = copy.deepcopy(model)
    keys = wrap_targets(model, config.target_modules)
    return model, reference, keys, Hypernet(config, BaseShapes.from_config(model.config))


def z_bar(batch=1, seed=3):
    return torch.randn(batch, TINY.embed_dim, generator=torch.Generator().manual_seed(seed))


def token_ids():
    return torch.randint(0, 16, (2, 24), generator=torch.Generator().manual_seed(2))


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def ce_backward(model, factors, scaling, ids):
    with inject(model, factors, scaling):
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()


def test_config_file_holds_the_default_scope():
    assert CONFIG == HypernetConfig(
        embed_dim=2560,
        latent=512,
        head_in=384,
        dropout=0.05,
        rank=16,
        alpha=32,
        target_modules=ATTENTION,
        seed=0,
    )
    assert CONFIG.scaling == rslora_scaling(16, 32) == 8.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"latent": 30}, "divisible by 4"),
        ({"rank": 0}, "rank"),
        ({"head_in": 2.5}, "head_in"),
        ({"dropout": 1.0}, "dropout"),
        ({"alpha": 0}, "alpha"),
        ({"seed": 1.5}, "seed"),
        ({"target_modules": ()}, "target_modules"),
        ({"target_modules": ("q_proj", "q_proj")}, "target_modules"),
        ({"target_modules": ("q_proj", "lm_head")}, "target_modules"),
        ({"target_modules": "q_proj"}, "target_modules"),
    ],
)
def test_config_rejects_bad_values(change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(CONFIG, **change)


def test_config_file_must_name_every_field(tmp_path):
    data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    path = tmp_path / "hypernet.yaml"
    path.write_text(yaml.safe_dump({**data, "ranks": 32}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown keys \['ranks'\]"):
        HypernetConfig.from_yaml(path)
    path.write_text(yaml.safe_dump({k: v for k, v in data.items() if k != "seed"}), "utf-8")
    with pytest.raises(ValueError, match=r"missing keys \['seed'\]"):
        HypernetConfig.from_yaml(path)


def test_base_shapes_come_from_the_model_config():
    base = BaseShapes.from_config(QWEN3_4B)
    assert base == BaseShapes(
        n_layers=36,
        hidden_size=2560,
        intermediate_size=9728,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
    )
    assert {name: base.projection(name) for name in PROJECTIONS} == {
        "q_proj": (2560, 4096),
        "k_proj": (2560, 1024),
        "v_proj": (2560, 1024),
        "o_proj": (4096, 2560),
        "gate_proj": (2560, 9728),
        "up_proj": (2560, 9728),
        "down_proj": (9728, 2560),
    }
    no_head_dim = types.SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    assert BaseShapes.from_config(no_head_dim).head_dim == 16  # Qwen3Attention's fallback
    with pytest.raises(ValueError, match="unknown projection"):
        base.projection("lm_head")
    with pytest.raises(ValueError, match="positive"):
        dataclasses.replace(base, head_dim=0)


@pytest.mark.parametrize("scope", ["default", "MLP and r = 32"])
def test_shapes_and_keys_on_qwen3_4b(scope):
    """Every (layer, module) of the real shapes, against the modules and inject's checks (meta)."""
    config = CONFIG
    if scope != "default":
        config = dataclasses.replace(CONFIG, rank=32, target_modules=ATTENTION + MLP)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(QWEN3_4B)
        hypernet = Hypernet(config, BaseShapes.from_config(model.config))
    keys = wrap_targets(model, config.target_modules)
    assert len(keys) == 36 * len(config.target_modules)
    assert hypernet.factor_keys == keys
    emitted = hypernet(torch.empty(2, 2560, device="meta"))
    assert len(emitted) == 2
    for factors in emitted:
        assert list(factors) == keys
        for key, (lora_a, lora_b) in factors.items():
            linear = model.get_submodule(f"model.{key}").base_layer
            assert lora_a.shape == (config.rank, linear.in_features)
            assert lora_b.shape == (linear.out_features, config.rank)
        with inject(model, factors, config.scaling):  # inject's key, shape, and dtype checks
            pass


def test_parameter_count_of_the_default_scope():
    with torch.device("meta"):
        hypernet = Hypernet(CONFIG, BaseShapes.from_config(QWEN3_4B))
    assert all(p.is_meta and p.requires_grad for p in hypernet.parameters())  # nothing allocated
    count = sum(p.numel() for p in hypernet.parameters())
    # Closed form from docs/modules/hypernet.md.
    e, d, h, r, n_codes = 2560, 512, 384, 16, 36 + 4  # layer codes plus module codes
    trunk = 28 * d**2 + d * (e + 1) // 2 + n_codes * d // 4 + 27 * d + (4 * d + 1) * h
    heads = (h + 1) * r * sum(d_in + d_out for d_in, d_out in hypernet.shapes.values())
    print(f"P3.3 default scope: {count:,} trainable parameters (trunk {trunk:,}, heads {heads:,})")
    assert count == trunk + heads == 134_958_208
    assert count <= BUDGET


def test_shapes_and_keys_on_a_tiny_base():
    model, _, keys, hypernet = prepared()
    emitted = hypernet(z_bar(batch=3))
    assert len(emitted) == 3
    for factors in emitted:
        assert list(factors) == keys == hypernet.factor_keys
        for key, (lora_a, lora_b) in factors.items():
            linear = model.get_submodule(f"model.{key}").base_layer
            assert lora_a.shape == (TINY.rank, linear.in_features)
            assert lora_b.shape == (linear.out_features, TINY.rank)
            assert lora_a.dtype == lora_b.dtype == torch.float32


def test_initial_adapter_is_peft_init_for_every_task():
    _, _, _, hypernet = prepared()
    first, second = hypernet(z_bar(batch=2))
    for key, (lora_a, lora_b) in first.items():
        name = key.rpartition(".")[2]
        assert not lora_b.any()  # B = 0 exactly
        assert torch.equal(lora_a, first[f"layers.0.{PROJECTIONS[name]}.{name}"][0])  # one A_0
        assert torch.equal(lora_a, second[key][0]) and not second[key][1].any()  # input-free
        bound = 1 / math.sqrt(lora_a.shape[1])  # PEFT's kaiming_uniform_(a=sqrt(5)) bound
        assert 0.9 * bound < lora_a.abs().max() <= bound
    q_proj, k_proj = first["layers.0.self_attn.q_proj"][0], first["layers.0.self_attn.k_proj"][0]
    assert not torch.equal(q_proj, k_proj)  # each module type draws its own A_0


def test_head_output_is_vec_a_then_vec_b_transposed():
    """With zero head weights a head emits its bias: [vec(A [r, d_in]); vec(B^T [r, d_out])]."""
    _, _, _, hypernet = prepared()
    r, expected = TINY.rank, {}
    with torch.no_grad():
        for j, (name, head) in enumerate(hypernet.heads.items()):
            d_in, d_out = hypernet.shapes[name]
            lora_a = torch.arange(r * d_in, dtype=torch.float32).view(r, d_in) + 1e4 * j
            lora_b = -1 - torch.arange(d_out * r, dtype=torch.float32).view(d_out, r) - 1e4 * j
            head.bias.copy_(torch.cat((lora_a.flatten(), lora_b.T.flatten())))
            expected[name] = (lora_a, lora_b)
    assert hypernet.shapes["k_proj"] == (64, 32)  # d_out != d_in: a transposed layout shows
    for factors in hypernet(z_bar(batch=2)):
        for key, (lora_a, lora_b) in factors.items():
            want_a, want_b = expected[key.rpartition(".")[2]]
            assert torch.equal(lora_a, want_a) and torch.equal(lora_b, want_b), key


@DTYPES
def test_initial_adapter_reproduces_the_base_bitwise(dtype, tmp_path):
    model, reference, _, hypernet = prepared(dtype=dtype)
    hypernet.train()  # dropout on, as in P3.5: the zero head weights make it irrelevant
    z, ids = z_bar(), token_ids()
    factors = hypernet(z)[0]
    base = logits(reference, ids)
    with inject(model, factors, TINY.scaling):
        assert torch.equal(logits(model, ids), base)
    save_adapter(factors, tmp_path / "init", TINY.rank, TINY.alpha, TINY.target_modules)
    peft_model = PeftModel.from_pretrained(copy.deepcopy(reference), str(tmp_path / "init"))
    assert torch.equal(logits(peft_model, ids), base)
    # Not vacuous: once H emits a non-zero B, both paths move the logits, and they agree.
    with torch.no_grad():
        for head in hypernet.heads.values():
            head.bias.normal_(std=0.05, generator=torch.Generator().manual_seed(5))
    factors = hypernet.eval()(z)[0]
    with inject(model, factors, TINY.scaling):
        moved = logits(model, ids)
    assert (moved.float() - base.float()).abs().max() > 0.1
    save_adapter(factors, tmp_path / "moved", TINY.rank, TINY.alpha, TINY.target_modules)
    peft_model = PeftModel.from_pretrained(copy.deepcopy(reference), str(tmp_path / "moved"))
    assert torch.equal(logits(peft_model, ids), moved)


def test_gradients_reach_h_through_inject():
    model, _, _, hypernet = prepared()
    hypernet.eval()  # no dropout, so both steps see one function
    z, ids, r = z_bar(), token_ids(), TINY.rank
    factors = hypernet(z)[0]
    for pair in factors.values():
        for factor in pair:
            factor.retain_grad()
    ce_backward(model, factors, TINY.scaling, ids)
    for key, (lora_a, lora_b) in factors.items():
        assert lora_b.grad.any(), key  # dL/dB = s g (A_0 x)^T
        assert not lora_a.grad.any(), key  # dL/dA = s B^T g x^T = 0 while B = 0
    for name, head in hypernet.heads.items():
        rows_a = r * hypernet.shapes[name][0]  # the head emits vec(A), then vec(B^T)
        assert head.weight.grad[rows_a:].any() and head.bias.grad[rows_a:].any()
        assert not head.weight.grad[:rows_a].any() and not head.bias.grad[:rows_a].any()
    trunk = [p for n, p in hypernet.named_parameters() if not n.startswith("heads.")]
    assert not any(p.grad.any() for p in trunk)  # zero head weights block the trunk at step 0
    torch.optim.SGD(hypernet.parameters(), lr=1.0).step()
    hypernet.zero_grad()
    factors = hypernet(z)[0]
    ce_backward(model, factors, TINY.scaling, ids)
    silent = [n for n, p in hypernet.named_parameters() if p.grad is None or not p.grad.any()]
    assert not silent  # after one step every parameter learns
    for codes in (hypernet.layer_embedding, hypernet.module_embedding):
        assert codes.weight.grad.ne(0).any(dim=1).all()  # every layer and module code is used
    lora_b = factors["layers.0.self_attn.k_proj"][1]
    assert not torch.equal(lora_b, factors["layers.1.self_attn.k_proj"][1])  # B per layer
    assert not torch.equal(lora_b, factors["layers.0.self_attn.v_proj"][1])  # B per module type


def test_seeded_init_is_deterministic_and_leaves_the_global_rng_alone():
    base = BaseShapes.from_config(tiny_qwen3().config)
    state = torch.get_rng_state()
    first, second = Hypernet(TINY, base), Hypernet(TINY, base)
    other = Hypernet(dataclasses.replace(TINY, seed=1), base)
    assert torch.equal(torch.get_rng_state(), state)
    weights, again, reseeded = first.state_dict(), second.state_dict(), other.state_dict()
    assert all(torch.isfinite(value).all() for value in weights.values())
    assert all(torch.equal(value, again[name]) for name, value in weights.items())
    changed = {name for name, value in weights.items() if not torch.equal(value, reseeded[name])}
    assert {"task_encoder.0.weight", "layer_embedding.weight", "heads.q_proj.bias"} <= changed
    z = z_bar()
    pairs = zip(first.eval()(z)[0].values(), second.eval()(z)[0].values(), strict=True)
    assert all(torch.equal(a, b) for pair, twin in pairs for a, b in zip(pair, twin, strict=True))


def test_meta_built_h_materializes_to_the_seeded_weights():
    base = BaseShapes.from_config(tiny_qwen3().config)
    with torch.device("meta"):
        lazy = Hypernet(TINY, base)
    assert all(p.is_meta for p in lazy.parameters())
    lazy.to_empty(device="cpu")
    lazy.reset_parameters()
    eager = Hypernet(TINY, base).state_dict()
    assert all(torch.equal(value, eager[name]) for name, value in lazy.state_dict().items())


def test_batch_axis_holds_independent_tasks():
    _, _, _, hypernet = prepared()
    with torch.no_grad():  # leave the init, so the adapter depends on z_bar
        for head in hypernet.heads.values():
            head.weight.normal_(std=0.1, generator=torch.Generator().manual_seed(4))
    z = z_bar(batch=3)
    batched = hypernet.eval()(z)
    for b in range(3):
        for key, (lora_a, lora_b) in hypernet(z[b : b + 1])[0].items():
            torch.testing.assert_close(batched[b][key][0], lora_a)
            torch.testing.assert_close(batched[b][key][1], lora_b)
    key = "layers.1.self_attn.v_proj"
    assert not torch.equal(batched[0][key][1], batched[1][key][1])


def test_trunk_blocks_are_pre_ln_residual():
    """u + f(LN(u)) carries a constant shift of u through: LN removes it, the skip adds it."""
    _, _, _, hypernet = prepared()
    u = torch.randn(3, TINY.latent, generator=torch.Generator().manual_seed(6))
    assert len(hypernet.blocks) == 2
    with torch.no_grad():
        for block in hypernet.eval().blocks:
            torch.testing.assert_close(block(u + 2.0), block(u) + 2.0)


def test_scope_change_needs_only_the_config_file(tmp_path):
    """G2's expansions, MLP modules and r = 32, by editing configs/hypernet.yaml only."""
    data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    widths = {"embed_dim": TINY.embed_dim, "latent": TINY.latent, "head_in": TINY.head_in}
    data.update(widths, rank=32, target_modules=[*ATTENTION, *MLP])
    path = tmp_path / "hypernet.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = HypernetConfig.from_yaml(path)
    model, reference, keys, hypernet = prepared(config)
    factors = hypernet(z_bar())[0]
    assert list(factors) == keys
    assert len(keys) == 2 * 7
    lora_a, lora_b = factors["layers.1.mlp.down_proj"]
    assert lora_a.shape == (32, 128) and lora_b.shape == (64, 32)
    assert config.scaling == rslora_scaling(32, 32)
    ids = token_ids()
    base = logits(reference, ids)
    with inject(model, factors, config.scaling):
        assert torch.equal(logits(model, ids), base)
    save_adapter(factors, tmp_path / "adapter", config.rank, config.alpha, config.target_modules)
    peft_model = PeftModel.from_pretrained(copy.deepcopy(reference), str(tmp_path / "adapter"))
    assert torch.equal(logits(peft_model, ids), base)


def test_forward_rejects_a_bad_task_embedding():
    _, _, _, hypernet = prepared()
    width = TINY.embed_dim
    for bad in (torch.zeros(width), torch.zeros(1, width + 1), torch.zeros(1, 1, width)):
        with pytest.raises(ValueError, match="z_bar"):
            hypernet(bad)


def test_cpu_forward_time_of_a_tiny_h():
    """Recorded in the module doc; the 50 ms L4 target is measured by P3.6, which loads H."""
    _, _, _, hypernet = prepared()
    hypernet.eval()
    z, seconds = z_bar(), []
    with torch.no_grad():
        hypernet(z)  # warm-up
        for _ in range(50):
            start = time.perf_counter()
            hypernet(z)
            seconds.append(time.perf_counter() - start)
    median = sorted(seconds)[len(seconds) // 2]
    pairs = len(hypernet.factor_keys)
    print(f"P3.3 tiny H forward, CPU, batch 1, {pairs} pairs: median {median * 1e3:.2f} ms of 50")
    assert median < 1.0  # sanity bound only
