"""Differentiable LoRA injection into the frozen base, and export to PEFT (BRIEF B5 P3.1, B8).

Meta-training (P3.5) applies H's adapter to the frozen base as a differentiable function of the
factors; test time (P4.1) loads the same adapter through PEFT. `InjectableLinear.forward` replays
the op sequence of PEFT 0.17.1's unmerged `lora.Linear.forward`, so the two paths give bitwise
equal outputs, also on a bf16 base (docs/STATE.md, conflict 3). Math, PEFT citations, and limits:
docs/modules/inject.md.
"""

import json
import os
import re
from collections.abc import Iterator, Sequence, Set
from contextlib import contextmanager
from math import sqrt
from pathlib import Path
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import Tensor, nn

# One (A [r, d_in], B [d_out, r]) pair, PEFT layout, per key `layers.{i}.self_attn.{m}`.
LoRAFactors: TypeAlias = dict[str, tuple[Tensor, Tensor]]

DECODER_PREFIX = "model."  # a *ForCausalLM holds its decoder at `.model`; factor keys drop it
PEFT_PREFIX = "base_model.model."  # PeftModelForCausalLM -> LoraModel -> the wrapped causal LM
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
_KEY = re.compile(r"layers\.\d+\.\w+\.\w+")


class InjectableLinear(nn.Module):
    """A frozen `nn.Linear` that adds an externally supplied LoRA delta while one is installed.

    The original layer is kept as `base_layer`, as PEFT's `lora.Linear` keeps it, so its
    parameters, dtype, and device are untouched. With nothing installed the output is the base
    output, bitwise. Only `inject` installs factors, for the duration of one block.

    Args:
        base_layer: The projection to wrap.
    """

    def __init__(self, base_layer: nn.Linear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self._lora: tuple[Tensor, Tensor, float] | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Return `base_layer(x)`, plus `scaling * B A x` while factors are installed.

        Replays PEFT 0.17.1 `lora.Linear.forward` (`peft/tuners/lora/layer.py` 757 to 780) op for
        op: base output; input cast to the factor dtype; `F.linear` with A, then with B; times the
        scale; add, which runs in float32 when the factors are float32; cast back to the dtype of
        the base output.

        Args:
            x: Input activations, `[..., d_in]`.

        Returns:
            Output activations in the base output dtype, `[..., d_out]`.
        """
        result = self.base_layer(x)
        if self._lora is None:
            return result
        lora_a, lora_b, scaling = self._lora
        torch_result_dtype = result.dtype
        x = x.to(lora_a.dtype)
        result = result + F.linear(F.linear(x, lora_a), lora_b) * scaling
        return result.to(torch_result_dtype)


def rslora_scaling(r: int, alpha: float) -> float:
    """Return the rsLoRA scale s = alpha / sqrt(r), the one source of s for injection and export.

    The expression is PEFT's (`peft/tuners/lora/layer.py` 221 to 222 with `use_rslora=True`), so
    the float equals, bit for bit, the scale PEFT derives from `peft_config(r, alpha, ...)`.

    Args:
        r: LoRA rank.
        alpha: LoRA alpha.

    Returns:
        The scale s; 8.0 for H's scope (r = 16, alpha = 32).

    Raises:
        ValueError: If `r` is not positive.
    """
    if r <= 0:
        raise ValueError(f"rank must be positive, got {r}")
    return alpha / sqrt(r)


def wrap_targets(model: nn.Module, target_modules: Sequence[str]) -> list[str]:
    """Wrap each target projection in an `InjectableLinear`, once.

    A module is a target when the last component of its qualified name is in `target_modules`,
    the rule PEFT applies to a list of names, so the wrapped set is the set PEFT adapts for the
    same config. Idempotent: wrapped modules are kept. Parameters, dtypes, devices, and
    `requires_grad` flags are untouched (the caller freezes the base); with nothing injected the
    model output is bitwise unchanged. Every match is checked before any is wrapped.

    Args:
        model: Causal LM with its decoder at `model.model`, for example `Qwen3ForCausalLM`.
        target_modules: Projection names, for example `["q_proj", "k_proj", "v_proj", "o_proj"]`.

    Returns:
        The keys `inject` expects: every wrapped projection of the model in module order, such
        as `layers.0.self_attn.q_proj`.

    Raises:
        ValueError: If `target_modules` is empty, nothing matches, or a match lies outside
            `model.model` (its PEFT key would differ from the one `to_peft_state_dict` writes).
        TypeError: If a match is neither a plain `nn.Linear` nor already wrapped; the replayed op
            sequence is the one of PEFT's `lora.Linear`.
    """
    targets = set(target_modules)
    if not targets:
        raise ValueError("target_modules is empty")
    matches = [(n, m) for n, m in model.named_modules() if n.rpartition(".")[2] in targets]
    if not matches:
        raise ValueError(f"no module is named any of {sorted(targets)}")
    for name, module in matches:
        if not name.startswith(DECODER_PREFIX):
            raise ValueError(f"{name}: target outside the decoder at {DECODER_PREFIX!r}")
        if not isinstance(module, InjectableLinear) and type(module) is not nn.Linear:
            raise TypeError(f"{name}: expected nn.Linear, got {type(module).__name__}")
    for name, module in matches:
        if not isinstance(module, InjectableLinear):
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, InjectableLinear(module))
    return list(_wrapped(model))


@contextmanager
def inject(
    model: nn.Module,
    factors: LoRAFactors,
    scaling: float,
    dtype: torch.dtype | None = torch.float32,
) -> Iterator[None]:
    """Install LoRA factors on every wrapped projection for one forward and backward.

    Inside the block each wrapped projection computes `base(x) + scaling * B A x` (see
    `InjectableLinear.forward`), differentiably in the factors. Call `loss.backward()` inside the
    block: activation checkpointing recomputes layer forwards during backward and checks only the
    shape, dtype, and device of what it recomputes, so a recompute under other factors would
    corrupt the gradients silently (research note, risk 2). On exit, also by exception, all
    factors are removed.

    Keys and shapes are checked before anything is installed. Each factor is cast to `dtype` and
    copied to fresh contiguous storage (differentiably, once per block), as PEFT's parameters are:
    GEMM kernel choice, and with it rounding, can depend on strides and alignment.

    Args:
        model: Model prepared by `wrap_targets`.
        factors: One `(A [r, d_in], B [d_out, r])` per wrapped projection, keyed exactly as
            `wrap_targets` returned; A and B share a dtype and the base layer's device.
        scaling: LoRA scale s, from `rslora_scaling`.
        dtype: Compute dtype of the delta. float32, the default, matches PEFT, which keeps adapter
            weights in float32 on a bf16 base (`autocast_adapter_dtype`); None keeps the dtype of
            the factors.

    Yields:
        None; the factors stay installed until the block exits.

    Raises:
        RuntimeError: If the model has no wrapped projections, or factors are already installed
            (blocks do not nest).
        ValueError: If the keys differ from the wrapped set, or a pair has the wrong shapes (for
            example a transposed factor or D2L's `[r, d_out]` layout for B), mismatched dtypes,
            or a device other than its base layer's.
    """
    wrapped = _wrapped(model)
    if not wrapped:
        raise RuntimeError("the model has no wrapped projections: call wrap_targets first")
    if any(module._lora is not None for module in wrapped.values()):
        raise RuntimeError("factors are already installed: inject blocks do not nest")
    _check_keys(factors.keys(), wrapped.keys())
    for key, module in wrapped.items():
        _check_pair(key, *factors[key], module.base_layer)
    try:
        for key, module in wrapped.items():
            lora_a, lora_b = factors[key]
            module._lora = (_fresh(lora_a, dtype), _fresh(lora_b, dtype), scaling)
        yield
    finally:
        for module in wrapped.values():
            module._lora = None


def to_peft_state_dict(
    factors: LoRAFactors, dtype: torch.dtype | None = torch.float32
) -> dict[str, Tensor]:
    """Rename factors to PEFT state-dict keys, the form PEFT saves and loads.

    `layers.{i}.self_attn.{m}` becomes `base_model.model.model.layers.{i}.self_attn.{m}` plus
    `.lora_A.weight` (`[r, d_in]`) and `.lora_B.weight` (`[d_out, r]`): the keys of
    `get_peft_model_state_dict`, which strips the adapter name (`peft/utils/save_and_load.py`
    319). Tensors are detached, cast to `dtype`, and copied to their own contiguous storage, so
    safetensors can save them even when the factors are views of one buffer. float32 by default,
    the dtype PEFT keeps adapter weights in and `inject` computes in. Parity then depends on the
    loader: on a bf16 base only loaders that copy into an existing float32 adapter keep the
    values; the others (`low_cpu_mem_usage=True`, a new adapter name,
    `autocast_adapter_dtype=False`, transformers' `load_adapter`) round them to bf16
    (docs/modules/inject.md, Limits).

    Args:
        factors: `(A [r, d_in], B [d_out, r])` per key `layers.{i}.self_attn.{m}`.
        dtype: Export dtype; None keeps each factor's dtype.

    Returns:
        PEFT key to tensor, two entries per factor key.

    Raises:
        ValueError: If a key does not look like `layers.{i}.{block}.{module}`, or a pair is not
            two matrices `[r, d_in]`, `[d_out, r]` with one r.
    """
    state: dict[str, Tensor] = {}
    for key, (lora_a, lora_b) in factors.items():
        if not _KEY.fullmatch(key):
            raise ValueError(f"{key!r}: expected a key such as 'layers.0.self_attn.q_proj'")
        if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != lora_b.shape[1]:
            raise ValueError(
                f"{key}: expected A [r, d_in] and B [d_out, r], "
                f"got A {list(lora_a.shape)} and B {list(lora_b.shape)}"
            )
        prefix = f"{PEFT_PREFIX}{DECODER_PREFIX}{key}"
        state[f"{prefix}.lora_A.weight"] = _fresh(lora_a.detach(), dtype)
        state[f"{prefix}.lora_B.weight"] = _fresh(lora_b.detach(), dtype)
    return state


def peft_config(r: int, alpha: float, target_modules: Sequence[str]) -> dict[str, Any]:
    """Return the `adapter_config.json` content whose PEFT forward `inject` replays.

    rsLoRA (PEFT then scales by `rslora_scaling(r, alpha)`), dropout 0, no bias, causal LM, plain
    LoRA (no DoRA) on `[out, in]` weights. PEFT fills the other fields with its defaults when it
    loads the file.

    Args:
        r: LoRA rank, shared by all targets.
        alpha: LoRA alpha.
        target_modules: Projection names, as passed to `wrap_targets`.

    Returns:
        JSON-serializable `LoraConfig` fields.

    Raises:
        ValueError: If `r` is not positive or `target_modules` is empty.
    """
    if r <= 0:
        raise ValueError(f"rank must be positive, got {r}")
    if not target_modules:
        raise ValueError("target_modules is empty")
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "use_rslora": True,
        "target_modules": list(target_modules),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_dora": False,
    }


def save_adapter(
    factors: LoRAFactors,
    path: str | os.PathLike[str],
    r: int,
    alpha: float,
    target_modules: Sequence[str],
    dtype: torch.dtype | None = torch.float32,
) -> None:
    """Write an adapter directory that `PeftModel.from_pretrained` loads offline.

    B4's Adapter format: `adapter_model.safetensors` with PEFT key names, and
    `adapter_config.json` with the metadata (rank, alpha, target modules) and the fixed LoRA
    settings of `peft_config`. Both files are written directly, not through
    `PeftModel.save_pretrained`, so nothing probes the Hub (research note, risk 6). Parity with
    `inject` on a bf16 base needs `PeftModel.from_pretrained` or `load_adapter` into an existing
    adapter name, each with its defaults; the other loaders round the factors
    (docs/modules/inject.md).

    Args:
        factors: `(A [r, d_in], B [d_out, r])` per key `layers.{i}.self_attn.{m}`.
        path: Output directory, created if missing; existing adapter files are overwritten.
        r: Rank of every factor pair.
        alpha: LoRA alpha; PEFT scales by `rslora_scaling(r, alpha)`.
        target_modules: Projection names; must equal the module names in the factor keys.
        dtype: Stored dtype, as in `to_peft_state_dict`.

    Raises:
        ValueError: If a factor's rank is not `r`, the factor keys name other modules than
            `target_modules`, or `to_peft_state_dict` or `peft_config` rejects the input.
    """
    config = peft_config(r, alpha, target_modules)
    state = to_peft_state_dict(factors, dtype)
    names = {key.rpartition(".")[2] for key in factors}
    if names != set(target_modules):
        targets = sorted(set(target_modules))
        raise ValueError(f"factor keys name {sorted(names)}, target_modules {targets}")
    ranks = sorted({lora_a.shape[0] for lora_a, _ in factors.values()})
    if ranks != [r]:
        raise ValueError(f"factor ranks {ranks}, expected [{r}]")
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    save_file(state, out / ADAPTER_WEIGHTS, metadata={"format": "pt"})
    (out / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _wrapped(model: nn.Module) -> dict[str, InjectableLinear]:
    return {
        name.removeprefix(DECODER_PREFIX): module
        for name, module in model.named_modules()
        if isinstance(module, InjectableLinear)
    }


def _check_keys(given: Set[str], wrapped: Set[str]) -> None:
    missing, unknown = sorted(wrapped - given), sorted(given - wrapped)
    if missing or unknown:
        raise ValueError(
            "factor keys differ from the wrapped projections: "
            f"missing {missing[:3]} ({len(missing)}), unknown {unknown[:3]} ({len(unknown)})"
        )


def _check_pair(key: str, lora_a: Tensor, lora_b: Tensor, base: nn.Linear) -> None:
    d_in, d_out = base.in_features, base.out_features
    r = lora_a.shape[0] if lora_a.ndim == 2 else 0
    if r < 1 or lora_a.shape != (r, d_in) or lora_b.shape != (d_out, r):
        raise ValueError(
            f"{key}: expected A [r, {d_in}] and B [{d_out}, r] (PEFT layout), "
            f"got A {list(lora_a.shape)} and B {list(lora_b.shape)}"
        )
    if lora_a.dtype != lora_b.dtype or not lora_a.is_floating_point():
        raise ValueError(f"{key}: A and B need one float dtype, got {lora_a.dtype}, {lora_b.dtype}")
    device = base.weight.device
    if lora_a.device != device or lora_b.device != device:
        raise ValueError(f"{key}: factors on {lora_a.device}, {lora_b.device}; base on {device}")


def _fresh(t: Tensor, dtype: torch.dtype | None) -> Tensor:
    return t.to(
        dtype=t.dtype if dtype is None else dtype, memory_format=torch.contiguous_format, copy=True
    )
