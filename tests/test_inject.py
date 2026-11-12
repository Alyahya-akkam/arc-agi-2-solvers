"""P3.1: LoRA injection against PEFT on a tiny random Qwen3, CPU only (docs/modules/inject.md).

Skipped when torch, transformers, peft, or safetensors is missing: CI installs `.[dev]` only.
Nothing touches the Hugging Face Hub: the model comes from `Qwen3Config`, adapters from `tmp_path`.
The parity tests print the measured max abs logit difference (`pytest -rP` shows it).
"""

import copy
import math
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from peft import (  # noqa: E402
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.tuners.lora import Linear as LoraLinear  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from arc26.hyper.inject import (  # noqa: E402
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    InjectableLinear,
    inject,
    peft_config,
    rslora_scaling,
    save_adapter,
    to_peft_state_dict,
    wrap_targets,
)

DEVICE = torch.device("cpu")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
# s = 16 / sqrt(8) is not a power of two, so the scale multiply rounds and its place in the op
# order matters (H's s = 8 would hide a reordering).
R, ALPHA = 8, 16
SCALING = rslora_scaling(R, ALPHA)
K_PROJ = "layers.0.self_attn.k_proj"  # d_in 64, d_out 32: catches transposes a square layer hides
DTYPES = pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])


def tiny_qwen3(dtype):
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
        initializer_range=0.2,  # logits of order 1, so a bf16 ULP is well above zero
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).to(device=DEVICE, dtype=dtype).eval()
    model.requires_grad_(False)
    return model


def prepared(dtype):
    """A wrapped model, an unwrapped bitwise copy of it, and the factor keys."""
    model = tiny_qwen3(dtype)
    reference = copy.deepcopy(model)
    return model, reference, wrap_targets(model, TARGETS)


def random_factors(reference, keys, *, zero_b=False, requires_grad=False):
    """float32 factors in PEFT layout, large enough to move the logits."""
    gen = torch.Generator().manual_seed(1)
    factors = {}
    for key in keys:
        linear = reference.get_submodule(f"model.{key}")
        d_in, d_out = linear.in_features, linear.out_features
        lora_a = torch.randn(R, d_in, generator=gen) / math.sqrt(d_in)
        lora_b = torch.randn(d_out, R, generator=gen) * 0.05
        if zero_b:
            lora_b = torch.zeros_like(lora_b)
        factors[key] = (lora_a.requires_grad_(requires_grad), lora_b.requires_grad_(requires_grad))
    return factors


def token_ids():
    return torch.randint(0, 16, (2, 24), generator=torch.Generator().manual_seed(2)).to(DEVICE)


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def peft_in_memory(reference, factors):
    """PEFT on a copy of the base, loaded with the exported state dict (trainable adapters)."""
    config = PeftConfig.from_peft_type(**peft_config(R, ALPHA, TARGETS))
    model = get_peft_model(copy.deepcopy(reference), config)
    result = set_peft_model_state_dict(model, to_peft_state_dict(factors))
    assert not result.unexpected_keys
    assert not [k for k in result.missing_keys if "lora_" in k]
    return model


def check_parity(injected, expected, dtype, label):
    diff = (injected.float() - expected.float()).abs().max().item()
    print(f"P3.1 {label} {dtype}: max abs logit diff {diff:.3e} (brief bound 1e-3)")
    assert diff < 1e-3
    # Bitwise in both tiers. In bf16 the bound means bitwise anyway (conflict 3), but a bf16 base
    # hides a reordering that changes only float32 roundings; the fp32 tier sees it.
    assert torch.equal(injected, expected)


def test_rslora_scaling():
    assert rslora_scaling(16, 32) == 8.0
    assert SCALING == ALPHA / math.sqrt(R)
    with pytest.raises(ValueError):
        rslora_scaling(0, 32)


@DTYPES
def test_wrap_targets_is_idempotent_and_transparent(dtype):
    model, reference, keys = prepared(dtype)
    assert keys == [f"layers.{i}.self_attn.{m}" for i in range(2) for m in TARGETS]
    assert wrap_targets(model, TARGETS) == keys
    assert wrap_targets(model, ["q_proj"]) == keys  # always the full set inject expects
    for key in keys:
        wrapper = model.get_submodule(f"model.{key}")
        assert isinstance(wrapper, InjectableLinear)
        assert type(wrapper.base_layer) is torch.nn.Linear
    ids = token_ids()
    assert torch.equal(logits(model, ids), logits(reference, ids))


def test_wrap_targets_rejects_bad_targets_atomically():
    model = tiny_qwen3(torch.float32)
    with pytest.raises(ValueError, match="empty"):
        wrap_targets(model, [])
    with pytest.raises(ValueError, match="no module"):
        wrap_targets(model, ["gate"])
    with pytest.raises(ValueError, match="outside the decoder"):
        wrap_targets(model, ["q_proj", "lm_head"])
    with pytest.raises(TypeError, match="nn.Linear"):
        wrap_targets(model, ["q_proj", "self_attn"])
    assert not any(isinstance(m, InjectableLinear) for m in model.modules())


@DTYPES
def test_zero_b_gives_bitwise_base_logits(dtype):
    model, reference, keys = prepared(dtype)
    ids = token_ids()
    with inject(model, random_factors(reference, keys, zero_b=True), SCALING):
        injected = logits(model, ids)
    assert torch.equal(injected, logits(reference, ids))


@DTYPES
def test_logits_match_peft_with_exported_weights(dtype):
    model, reference, keys = prepared(dtype)
    factors = random_factors(reference, keys)
    ids = token_ids()
    with inject(model, factors, SCALING):
        injected = logits(model, ids)
    peft_model = peft_in_memory(reference, factors)
    k_proj = peft_model.get_submodule(f"base_model.model.model.{K_PROJ}")
    assert k_proj.lora_A["default"].weight.dtype == torch.float32  # also on a bf16 base
    assert k_proj.scaling["default"] == SCALING
    effect = (injected.float() - logits(reference, ids).float()).abs().max().item()
    assert effect > 0.1  # the adapter moves the logits, so parity is not vacuous
    check_parity(injected, logits(peft_model, ids), dtype, "inject vs PEFT state dict")


@DTYPES
def test_each_layer_matches_peft_lora_linear(dtype):
    """Each wrapped projection against PEFT's own `lora.Linear` on a copy of its base layer.

    One layer compared bitwise sees a change of a single float32 rounding, for example the scale
    applied to `A x` instead of `B A x`; the whole-model bf16 tier sees only bf16 flips.
    """
    model, reference, keys = prepared(dtype)
    factors = random_factors(reference, keys)
    gen = torch.Generator().manual_seed(3)
    with inject(model, factors, SCALING), torch.no_grad():
        for key in keys:
            wrapper = model.get_submodule(f"model.{key}")
            base = copy.deepcopy(wrapper.base_layer)
            layer = LoraLinear(base, "default", r=R, lora_alpha=ALPHA, use_rslora=True)
            assert layer.scaling["default"] == SCALING
            lora_a, lora_b = factors[key]
            # PeftModel upcasts adapters to float32 (autocast_adapter_dtype); a bare layer does not.
            layer.lora_A["default"].to(torch.float32).weight.copy_(lora_a)
            layer.lora_B["default"].to(torch.float32).weight.copy_(lora_b)
            x = (torch.randn(3, 11, base.in_features, generator=gen) * 2).to(dtype)
            assert torch.equal(wrapper(x), layer(x)), key


def test_gradients_reach_factors_and_match_peft():
    model, reference, keys = prepared(torch.float32)
    factors = random_factors(reference, keys, requires_grad=True)
    ids = token_ids()
    with inject(model, factors, SCALING):
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
    peft_model = peft_in_memory(reference, factors)
    peft_loss = peft_model(input_ids=ids, labels=ids, use_cache=False).loss
    peft_loss.backward()
    torch.testing.assert_close(loss, peft_loss)
    worst = 0.0
    for key, (lora_a, lora_b) in factors.items():
        layer = peft_model.get_submodule(f"base_model.model.model.{key}")
        pairs = (
            (lora_a.grad, layer.lora_A["default"].weight.grad),
            (lora_b.grad, layer.lora_B["default"].weight.grad),
        )
        for mine, theirs in pairs:
            assert mine is not None and mine.abs().max() > 0
            torch.testing.assert_close(mine, theirs)
            worst = max(worst, (mine - theirs).abs().max().item())
    print(f"P3.1 gradients inject vs PEFT: max abs diff {worst:.3e}")


def test_checkpointing_does_not_change_gradients():
    model, reference, keys = prepared(torch.float32)
    model.train()  # transformers checkpoints only in training mode; Qwen3 dropout is 0
    ids = token_ids()
    calls = []
    model.get_submodule(f"model.{keys[0]}").register_forward_pre_hook(lambda *_: calls.append(1))

    def factor_grads():
        factors = random_factors(reference, keys, requires_grad=True)
        with inject(model, factors, SCALING):
            model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        return [t.grad for pair in factors.values() for t in pair]

    plain = factor_grads()
    assert len(calls) == 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpointed = factor_grads()
    assert len(calls) == 3  # forward, then the recompute during backward
    assert all(torch.equal(p, c) for p, c in zip(plain, checkpointed, strict=True))


@pytest.mark.parametrize(
    "variant",
    ["A transposed", "B transposed", "both transposed", "A and B swapped", "dims swapped"],
)
def test_k_proj_layout_trap(variant):
    model, reference, keys = prepared(torch.float32)
    factors = random_factors(reference, keys)
    lora_a, lora_b = factors[K_PROJ]
    assert lora_a.shape == (R, 64) and lora_b.shape == (32, R)
    factors[K_PROJ] = {
        "A transposed": (lora_a.T, lora_b),
        "B transposed": (lora_a, lora_b.T),  # also D2L's B layout [r, d_out]
        "both transposed": (lora_a.T, lora_b.T),
        "A and B swapped": (lora_b, lora_a),
        "dims swapped": (torch.randn(R, 32), torch.randn(64, R)),  # passes on a square q_proj
    }[variant]
    with pytest.raises(ValueError, match="k_proj"), inject(model, factors, SCALING):
        pass
    ids = token_ids()
    assert torch.equal(logits(model, ids), logits(reference, ids))  # nothing was installed


def test_inject_rejects_bad_keys_dtypes_and_nesting():
    model, reference, keys = prepared(torch.float32)
    factors = random_factors(reference, keys)
    missing = {k: v for k, v in factors.items() if k != keys[0]}
    with pytest.raises(ValueError, match="missing"), inject(model, missing, SCALING):
        pass
    unknown = {**factors, "layers.9.self_attn.q_proj": factors[keys[0]]}
    with pytest.raises(ValueError, match="unknown"), inject(model, unknown, SCALING):
        pass
    lora_a, lora_b = factors[K_PROJ]
    mixed = {**factors, K_PROJ: (lora_a, lora_b.to(torch.bfloat16))}
    with pytest.raises(ValueError, match="dtype"), inject(model, mixed, SCALING):
        pass
    with pytest.raises(RuntimeError, match="wrap_targets"), inject(reference, factors, SCALING):
        pass
    with inject(model, factors, SCALING):
        with pytest.raises(RuntimeError, match="nest"), inject(model, factors, SCALING):
            pass


@DTYPES
def test_leaving_inject_restores_base_output(dtype):
    model, reference, keys = prepared(dtype)
    factors = random_factors(reference, keys)
    ids = token_ids()
    base = logits(reference, ids)

    class Boom(Exception):
        pass

    with inject(model, factors, SCALING):
        assert not torch.equal(logits(model, ids), base)
    assert torch.equal(logits(model, ids), base)
    with pytest.raises(Boom), inject(model, factors, SCALING):
        raise Boom
    assert torch.equal(logits(model, ids), base)


@DTYPES
def test_save_adapter_then_peft_from_pretrained(dtype, tmp_path):
    model, reference, keys = prepared(dtype)
    factors = random_factors(reference, keys)
    ids = token_ids()
    with inject(model, factors, SCALING):
        injected = logits(model, ids)
    save_adapter(factors, tmp_path, R, ALPHA, TARGETS)
    peft_model = PeftModel.from_pretrained(copy.deepcopy(reference), str(tmp_path))
    check_parity(injected, logits(peft_model, ids), dtype, "inject vs PEFT from_pretrained")
    saved = load_file(tmp_path / ADAPTER_WEIGHTS)
    assert set(saved) == set(get_peft_model_state_dict(peft_model))
    assert all(t.dtype == torch.float32 for t in saved.values())
    config = peft_model.peft_config["default"]
    assert (config.r, config.lora_alpha, config.use_rslora) == (R, ALPHA, True)
    assert set(config.target_modules) == set(TARGETS)
    assert (config.lora_dropout, config.bias, config.task_type) == (0.0, "none", "CAUSAL_LM")
    assert (tmp_path / ADAPTER_CONFIG).is_file()


# Loaders of a saved adapter on a bf16 base, and whether they keep its float32 factors bitwise
# (docs/modules/inject.md, Limits); the others store the factors rounded to bf16.
LOADERS = {
    "from_pretrained": True,
    "swap load_adapter": True,
    "swap set_peft_model_state_dict": True,
    "from_pretrained low_cpu_mem_usage": False,
    "swap load_adapter low_cpu_mem_usage": False,
    "load_adapter new name": False,
    "from_pretrained no autocast": False,
    "transformers load_adapter": False,
}


def load(loader, base, path, previous, factors):
    """Load the adapter at `path` into `base` by `loader`; return the model and the adapter name.

    Swap loaders start from a PEFT model that holds the `previous` puzzle's adapter (P4.1).
    """
    match loader:
        case "from_pretrained":
            return PeftModel.from_pretrained(base, str(path)), "default"
        case "from_pretrained low_cpu_mem_usage":
            return PeftModel.from_pretrained(base, str(path), low_cpu_mem_usage=True), "default"
        case "from_pretrained no autocast":
            model = PeftModel.from_pretrained(base, str(path), autocast_adapter_dtype=False)
            return model, "default"
        case "transformers load_adapter":
            base.load_adapter(str(path))
            return base, "default"
    model = PeftModel.from_pretrained(base, str(previous))
    match loader:
        case "swap load_adapter":
            model.load_adapter(str(path), "default")
        case "swap set_peft_model_state_dict":
            set_peft_model_state_dict(model, to_peft_state_dict(factors))
        case "swap load_adapter low_cpu_mem_usage":
            model.load_adapter(str(path), "default", low_cpu_mem_usage=True)
        case "load_adapter new name":
            model.load_adapter(str(path), "puzzle")
            model.set_adapter("puzzle")
            return model, "puzzle"
    return model, "default"


@pytest.mark.parametrize("loader", list(LOADERS))
def test_peft_loaders_on_bf16_base(loader, tmp_path):
    model, reference, keys = prepared(torch.bfloat16)
    factors = random_factors(reference, keys)
    ids = token_ids()
    with inject(model, factors, SCALING):
        injected = logits(model, ids)
    path, previous = tmp_path / "puzzle", tmp_path / "previous"
    save_adapter(factors, path, R, ALPHA, TARGETS)
    save_adapter({k: (a / 2, b / 2) for k, (a, b) in factors.items()}, previous, R, ALPHA, TARGETS)
    loaded, name = load(loader, copy.deepcopy(reference), path, previous, factors)
    layer = next(m for n, m in loaded.named_modules() if n.endswith(K_PROJ))
    lora_a = factors[K_PROJ][0]
    keeps = LOADERS[loader]
    stored = lora_a if keeps else lora_a.to(torch.bfloat16).float()
    assert torch.equal(layer.lora_A[name].weight.float(), stored)
    assert torch.equal(logits(loaded, ids), injected) == keeps


def test_export_copies_views_and_validates(tmp_path):
    _, reference, keys = prepared(torch.float32)
    factors = random_factors(reference, keys)
    flat = torch.cat([t.reshape(-1) for pair in factors.values() for t in pair])
    views, offset = {}, 0
    for key, (lora_a, lora_b) in factors.items():
        a_end = offset + lora_a.numel()
        b_end = a_end + lora_b.numel()
        views[key] = (flat[offset:a_end].view_as(lora_a), flat[a_end:b_end].view_as(lora_b))
        offset = b_end
    state = to_peft_state_dict(views)
    assert len({t.untyped_storage().data_ptr() for t in state.values()}) == len(state)
    save_adapter(views, tmp_path, R, ALPHA, TARGETS)  # safetensors refuses shared storage
    with pytest.raises(ValueError, match="ranks"):
        save_adapter(factors, tmp_path, R + 1, ALPHA, TARGETS)
    with pytest.raises(ValueError, match="target_modules"):
        save_adapter(factors, tmp_path, R, ALPHA, ["q_proj", "v_proj"])
    with pytest.raises(ValueError, match="key such as"):
        to_peft_state_dict({f"model.{keys[0]}": factors[keys[0]]})
