r"""Hypernetwork H: a task embedding to a LoRA adapter for the frozen base (BRIEF B2 Q1, Q2; P3.3).

H maps the pooled pair embedding of a task (P3.2) to one `(A, B)` pair per target projection of
every decoder layer, in one forward pass. The design is T2L's hypernetwork head (research note
`docs/research/d2l_t2l_reading.md`, section 2), written from the spec, not copied: a task code,
a layer code, and a module code feed an MLP trunk, and one linear head per module type, shared
across layers, emits the factors. At initialization every layer gets PEFT's initial `lora_A` and
`B = 0`, so the adapted base equals the base exactly (bias-HyperInit). The output is P3.1's
`LoRAFactors`, so it goes straight into `inject` (P3.5) or `to_peft_state_dict` (P4.1). Math,
parameter count, and limits: docs/modules/hypernet.md.

The task code reads the mean $\bar z$ of the pair embeddings (`aggregator: mean`, the default and
the H of every checkpoint before P3.3b). With `aggregator: set` (P3.3b, scoping note
`docs/research/h_upgrades_scoping.md` section 1(a)) a Set Transformer (Lee et al. 2019, arXiv
1810.00825) also reads the set of pair embeddings $\bar e_i$ and adds a zero-initialized term per
(module, layer) row to the task code, so a set H starts exactly at the mean H it is loaded from.
"""

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import torch
import yaml
from torch import Tensor, nn

from arc26.hyper.inject import LoRAFactors, rslora_scaling

# Projection name -> the decoder block that holds it, in Qwen3's module order (transformers
# 4.55.4 `modeling_qwen3.py`: Qwen3Attention 170 to 181, Qwen3MLP 75 to 77), the order in which
# `wrap_targets` returns the factor keys.
PROJECTIONS: dict[str, str] = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


AGGREGATORS = ("mean", "set")


@dataclass(frozen=True)
class AggregatorConfig:
    r"""How H reads the demonstration set (P3.3b); `configs/hypernet.yaml` gives the reasons.

    `mean` is H before P3.3b: the task code reads $\bar z$ only, and the other fields are unused.
    `set` adds the Set Transformer branch of `Hypernet` (`SetAggregator`). Validated on
    construction.

    Attributes:
        kind: `mean` or `set`.
        width: Width $d_s$ of the set tokens and of the attention.
        blocks: Pre-LN self-attention blocks (SAB) over the pair tokens.
        heads: Attention heads of every SAB and of the pooling (PMA); must divide `width`.

    Raises:
        ValueError: If `kind` is unknown, a size is not a positive int, or `heads` does not
            divide `width`.
    """

    kind: str = "mean"
    width: int = 512
    blocks: int = 2
    heads: int = 8

    def __post_init__(self) -> None:
        if self.kind not in AGGREGATORS:
            raise ValueError(f"aggregator.kind must be one of {AGGREGATORS}, got {self.kind!r}")
        for name in ("width", "blocks", "heads"):
            if not _is_positive_int(getattr(self, name)):
                raise ValueError(
                    f"aggregator.{name} must be a positive int, got {getattr(self, name)!r}"
                )
        if self.width % self.heads:
            raise ValueError(
                f"aggregator.heads {self.heads} must divide aggregator.width {self.width}"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Build the config from a mapping, refusing unknown keys (missing keys take defaults).

        Args:
            data: Some of the fields above, such as the `aggregator` block of a YAML file.

        Returns:
            The validated config.

        Raises:
            ValueError: If a key is unknown or a value is invalid.
        """
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"aggregator: unknown keys {unknown}")
        return cls(**data)


@dataclass(frozen=True)
class HypernetConfig:
    """H's architecture and adapter scope; `configs/hypernet.yaml` holds the values and reasons.

    The scope (rank, alpha, target modules) is data, so G2's expansions (MLP modules, r = 32)
    change this config only. Validated on construction, also by `dataclasses.replace`.

    Attributes:
        embed_dim: Width E of the task embedding `z_bar` (P3.2's pooled pair embedding).
        latent: Trunk width d; the task code takes d / 2, the layer and module codes d / 4 each.
        head_in: Head input width H_in; each head holds (H_in + 1) r (d_in + d_out) parameters.
        dropout: Dropout probability in the trunk (active in training mode only).
        rank: LoRA rank r of every emitted pair.
        alpha: LoRA alpha; the adapter scale is `rslora_scaling(rank, alpha)`.
        target_modules: Projection names adapted in every decoder layer, keys of `PROJECTIONS`.
        seed: Seed of the initialization (`Hypernet.reset_parameters`).
        aggregator: How H reads the demonstration set; a mapping is converted. The default,
            `mean`, is H before P3.3b, so a config or checkpoint without the block builds it.

    Raises:
        ValueError: If a width or the rank is not a positive int, `latent` is not divisible by
            4, `dropout` is outside [0, 1), `alpha` is not positive, the seed is not an int,
            `target_modules` is empty, repeats a name, or names an unknown projection, or the
            aggregator is invalid.
    """

    embed_dim: int
    latent: int
    head_in: int
    dropout: float
    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    seed: int
    aggregator: AggregatorConfig = AggregatorConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if isinstance(self.aggregator, Mapping):
            object.__setattr__(self, "aggregator", AggregatorConfig.from_mapping(self.aggregator))
        if not isinstance(self.aggregator, AggregatorConfig):
            raise ValueError(f"aggregator must be a mapping, got {self.aggregator!r}")
        for name in ("embed_dim", "latent", "head_in", "rank"):
            if not _is_positive_int(getattr(self, name)):
                raise ValueError(f"{name} must be a positive int, got {getattr(self, name)!r}")
        if self.latent % 4:
            raise ValueError(f"latent must be divisible by 4 (d/2 + d/4 + d/4), got {self.latent}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must lie in [0, 1), got {self.dropout}")
        if isinstance(self.alpha, bool) or not self.alpha > 0:
            raise ValueError(f"alpha must be positive, got {self.alpha!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        targets = self.target_modules
        known = set(targets) <= PROJECTIONS.keys()
        if not targets or len(set(targets)) != len(targets) or not known:
            raise ValueError(
                f"target_modules must be distinct names from {list(PROJECTIONS)}, "
                f"got {list(targets)}"
            )

    @property
    def scaling(self) -> float:
        """The adapter scale s = alpha / sqrt(rank), from P3.1's single source `rslora_scaling`."""
        return rslora_scaling(self.rank, self.alpha)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain values, the form checkpoints store.

        The `aggregator` block is left out when it is the default (`mean` at default sizes), so
        a mean H serializes exactly as before P3.3b and `--resume` compares it unchanged.

        Returns:
            The fields as a mapping; `target_modules` as a list.
        """
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        if self.aggregator == AggregatorConfig():
            del data["aggregator"]
        return data

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> Self:
        """Load a config file such as `configs/hypernet.yaml`.

        Every field without a default must be present and no other key may be, so a misspelled
        key fails here instead of leaving a value at a silent default; `aggregator` may be left
        out (then `mean`).

        Args:
            path: YAML file with one mapping of the fields above.

        Returns:
            The validated config.

        Raises:
            ValueError: If the file is not one mapping, has missing or unknown keys, or holds an
                invalid value.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping, got {type(data).__name__}")
        names = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING}
        missing, unknown = sorted(required - data.keys()), sorted(data.keys() - names)
        if missing or unknown:
            raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
        return cls(**data)


@dataclass(frozen=True)
class BaseShapes:
    """The sizes of the frozen base that fix H's output shapes, read from the base's config.

    Build it with `from_config(model.config)`, so the shapes are the ones transformers builds the
    projections with, never literals.

    Attributes:
        n_layers: Decoder layers (`num_hidden_layers`); H emits an adapter for every one.
        hidden_size: Residual width (`hidden_size`).
        intermediate_size: MLP width (`intermediate_size`).
        n_heads: Query heads (`num_attention_heads`).
        n_kv_heads: Key and value heads (`num_key_value_heads`).
        head_dim: Width of one head (`head_dim`; Qwen3-4B sets 128, not 2560 / 32).

    Raises:
        ValueError: If a size is not a positive int.
    """

    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        bad = {f.name: getattr(self, f.name) for f in fields(self)}
        bad = {name: value for name, value in bad.items() if not _is_positive_int(value)}
        if bad:
            raise ValueError(f"base sizes must be positive ints, got {bad}")

    @classmethod
    def from_config(cls, config: Any) -> Self:
        """Read the sizes from a transformers config of the base, such as `model.config`.

        `head_dim` is read as `Qwen3Attention` reads it (`modeling_qwen3.py` 164): the config
        value if the attribute exists, else `hidden_size // num_attention_heads`.

        Args:
            config: A `Qwen3Config` or any object with the same attribute names.

        Returns:
            The base sizes.

        Raises:
            AttributeError: If the config lacks one of the attributes.
            ValueError: If a size is not a positive int.
        """
        return cls(
            n_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        )

    def projection(self, name: str) -> tuple[int, int]:
        """Return `(d_in, d_out)` of a projection as Qwen3 builds it (`modeling_qwen3.py`).

        Args:
            name: A key of `PROJECTIONS`, such as `"k_proj"`.

        Returns:
            Input and output width; `k_proj` of Qwen3-4B is `(2560, 1024)`.

        Raises:
            ValueError: If the name is not a key of `PROJECTIONS`.
        """
        q_width, kv_width = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        shapes = {
            "q_proj": (self.hidden_size, q_width),
            "k_proj": (self.hidden_size, kv_width),
            "v_proj": (self.hidden_size, kv_width),
            "o_proj": (q_width, self.hidden_size),
            "gate_proj": (self.hidden_size, self.intermediate_size),
            "up_proj": (self.hidden_size, self.intermediate_size),
            "down_proj": (self.intermediate_size, self.hidden_size),
        }
        if name not in shapes:
            raise ValueError(f"unknown projection {name!r}, expected one of {list(shapes)}")
        return shapes[name]


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
        nn.SiLU(),
    )


class _ResidualBlock(nn.Module):
    """Pre-LN residual MLP, `x + dropout(mlp(norm(x)))`, width d -> 4d -> d."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = _mlp(width, 4 * width, width, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.mlp(self.norm(x)))


class _SetBlock(nn.Module):
    """Pre-LN self-attention block (SAB) over a padded set, width d_s -> 4 d_s -> d_s."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * width, width)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, ignore: Tensor) -> Tensor:
        h = self.attn_norm(x)
        attended = self.attn(h, h, h, key_padding_mask=ignore, need_weights=False)[0]
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class SetAggregator(nn.Module):
    r"""Set Transformer over the pair embeddings: one context vector per (module, layer) row.

    Tokens $x_i = W_\text{in}\,\mathrm{LN}(\bar e_i)$ carry no position, so the output does not
    depend on the pairs' order; `blocks` pre-LN SABs and a final LayerNorm follow, then pooling by
    multihead attention (PMA) from one learned query $q_{m,\ell}$ per row, and the projection
    $W_o$ to the task code's width: $\delta_{m,\ell} = W_o\, c_{m,\ell}$ (Lee et al. 2019,
    arXiv 1810.00825, SAB and PMA). `Hypernet.reset_parameters` sets $W_o = 0$, so $\delta = 0$
    at initialization. Padded pairs are masked out of every attention as keys.

    Args:
        embed_dim: Width E of a pair embedding.
        config: The aggregator's sizes (`kind` must be `set`).
        n_queries: Rows, one per (module, layer): n_modules * n_layers, module-major.
        out_dim: Width of the task code, d / 2.
        dropout: Dropout in the attention and the feed-forward layers (training mode only).

    Attributes:
        input_norm, input: $\mathrm{LN}$ and $W_\text{in}$.
        blocks: The SABs.
        norm: LayerNorm of the encoded set.
        queries: $q_{m,\ell}$, `[n_queries, width]`.
        pool: The PMA attention.
        output: $W_o$.
    """

    def __init__(
        self, embed_dim: int, config: AggregatorConfig, n_queries: int, out_dim: int, dropout: float
    ) -> None:
        super().__init__()
        width = config.width
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input = nn.Linear(embed_dim, width)
        self.blocks = nn.ModuleList(
            _SetBlock(width, config.heads, dropout) for _ in range(config.blocks)
        )
        self.norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.empty(n_queries, width))
        self.pool = nn.MultiheadAttention(width, config.heads, dropout=dropout, batch_first=True)
        self.output = nn.Linear(width, out_dim)

    def forward(self, pairs: Tensor, mask: Tensor) -> Tensor:
        r"""Pool the set into one task-code offset per row.

        Args:
            pairs: Pair embeddings, `[batch, n_max, embed_dim]`; padded rows must be finite
                (`pad_pair_sets` pads with zeros).
            mask: `[batch, n_max]` bool, True for a real pair; at least one per task.

        Returns:
            $\delta$, `[batch, n_queries, out_dim]`.
        """
        ignore = ~mask  # [batch, n_max], MultiheadAttention's key_padding_mask convention
        x = self.input(self.input_norm(pairs))  # [batch, n_max, width]
        for block in self.blocks:
            x = block(x, ignore)
        x = self.norm(x)
        queries = self.queries[None].expand(pairs.shape[0], -1, -1)  # [batch, n_queries, width]
        pooled = self.pool(queries, x, x, key_padding_mask=ignore, need_weights=False)[0]
        return self.output(pooled)  # [batch, n_queries, out_dim]


def pad_pair_sets(sets: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Stack variable-size sets of pair embeddings into `Hypernet.forward`'s `pairs` and `mask`.

    Why: a batch of tasks holds different numbers of demonstrations; zero rows behind a mask keep
    every padded value finite, and the mask removes them from every attention.

    Args:
        sets: One `[n_i, E]` tensor per task, `n_i >= 1`, all of one dtype and device.

    Returns:
        `pairs [batch, n_max, E]` (zeros after row `n_i`) and `mask [batch, n_max]` (bool, True
        for the first `n_i` rows).

    Raises:
        ValueError: If `sets` is empty, a set is empty or not 2-D, or the widths differ.
    """
    if not sets:
        raise ValueError("pad_pair_sets needs at least one set")
    widths = {tuple(s.shape[1:]) for s in sets}
    if any(s.ndim != 2 or s.shape[0] == 0 for s in sets) or len(widths) != 1:
        raise ValueError(f"expected non-empty [n_i, E] sets of one width, got {widths}")
    first, n_max = sets[0], max(s.shape[0] for s in sets)
    pairs = first.new_zeros(len(sets), n_max, first.shape[1])
    mask = torch.zeros(len(sets), n_max, dtype=torch.bool, device=first.device)
    for b, rows in enumerate(sets):
        pairs[b, : rows.shape[0]] = rows
        mask[b, : rows.shape[0]] = True
    return pairs, mask


class Hypernet(nn.Module):
    """H: one task embedding to one LoRA pair per target projection of every decoder layer.

    For a task code t = LN(W z_bar + b), a layer code per layer i, and a module code per module
    type m, the trunk maps concat(t, layer_i, module_m) to a head input; the head of m, shared
    by all layers, emits A [r, d_in] and B [r, d_out], and B is transposed to PEFT's [d_out, r].
    The trunk runs once over all (module type, layer) pairs and each head once over all layers.
    With `config.aggregator.kind == "set"`, the task code of row (m, i) is
    t + W_o c_{m,i}, where c_{m,i} is `SetAggregator`'s pooling of the pair embeddings; W_o = 0
    at initialization, so the set H computes exactly what the mean H with the same weights does.

    Parameters are created on the meta device and filled by `reset_parameters`, so PyTorch's
    default initialization never runs: the global RNG is untouched, and the weights depend on
    `config.seed` only, not on the device. Built under `torch.device("meta")`, H stays on meta
    (parameter counts and shape checks without allocation).

    Args:
        config: Architecture and scope.
        base: Sizes of the frozen base, `BaseShapes.from_config(model.config)`.

    Attributes:
        config: As given.
        base: As given.
        module_names: Target modules in `PROJECTIONS` order; index of each module's code.
        shapes: `(d_in, d_out)` per module name.
        aggregator: The `SetAggregator`, or None for `mean` (then the state dict is unchanged).
    """

    def __init__(self, config: HypernetConfig, base: BaseShapes) -> None:
        super().__init__()
        self.config = config
        self.base = base
        self.module_names = tuple(name for name in PROJECTIONS if name in config.target_modules)
        self.shapes = {name: base.projection(name) for name in self.module_names}
        d, p = config.latent, config.dropout
        device = torch.get_default_device()
        with torch.device("meta"):
            self.task_encoder = nn.Sequential(
                nn.Linear(config.embed_dim, d // 2), nn.LayerNorm(d // 2)
            )
            self.layer_embedding = nn.Embedding(base.n_layers, d // 4)
            self.layer_norm = nn.LayerNorm(d // 4)
            self.module_embedding = nn.Embedding(len(self.module_names), d // 4)
            self.module_norm = nn.LayerNorm(d // 4)
            self.mixer = nn.Sequential(_mlp(d, 4 * d, d, p), nn.Dropout(p))
            self.blocks = nn.Sequential(_ResidualBlock(d, p), _ResidualBlock(d, p))
            self.mlp3 = nn.Sequential(nn.LayerNorm(d), _mlp(d, 4 * d, config.head_in, p))
            self.heads = nn.ModuleDict(
                {
                    name: nn.Linear(config.head_in, config.rank * (d_in + d_out))
                    for name, (d_in, d_out) in self.shapes.items()
                }
            )
            # Registered last, so the other parameters keep their order and their keys.
            self.aggregator: SetAggregator | None = None
            if config.aggregator.kind == "set":
                rows = len(self.module_names) * base.n_layers
                self.aggregator = SetAggregator(
                    config.embed_dim, config.aggregator, rows, d // 2, p
                )
        self.to_empty(device=device)
        self.reset_parameters()

    @property
    def factor_keys(self) -> list[str]:
        """The keys of each emitted `LoRAFactors`, layer-major, in `wrap_targets` order."""
        return [
            f"layers.{i}.{PROJECTIONS[name]}.{name}"
            for i in range(self.base.n_layers)
            for name in self.module_names
        ]

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Fill every parameter from `config.seed`; a no-op while the parameters are on meta.

        Values are drawn on the CPU from a generator seeded with `config.seed` and copied to the
        parameters' device, so they are the same on any device and the global RNG is untouched.
        Trunk and codes: PyTorch's defaults (`nn.Linear`: kaiming-uniform weight with
        a = sqrt(5) and a uniform bias in +-1/sqrt(fan_in); `nn.Embedding`: N(0, 1);
        `nn.LayerNorm`: ones and zeros). Heads, bias-HyperInit (T2L): weight 0, bias
        concat(vec(A_0), 0) with one A_0 [r, d_in] per module type, drawn as PEFT draws a new
        `lora_A` (`kaiming_uniform_(a=sqrt(5))`, `peft/tuners/lora/layer.py` 264). Every layer
        then gets A_0 and B = 0: the adapter's delta is exactly zero, and dL/dB is not. The set
        aggregator draws last, so every other parameter equals the mean H's of the same seed:
        `nn.Linear` and `nn.LayerNorm` as above, attention as `nn.MultiheadAttention` initializes
        itself (Xavier-uniform input projection, zero biases), Xavier-uniform queries (as the Set
        Transformer's seeds), and W_o = 0 (weight and bias).

        Call it after `to_empty(device=...)` to materialize an H built on meta.
        """
        if any(param.is_meta for param in self.parameters()):
            return
        generator = torch.Generator().manual_seed(self.config.seed)

        def fill(param: Tensor, init: Callable[..., Tensor], **kwargs: float) -> None:
            values = torch.empty(param.shape, dtype=param.dtype, device="cpu")
            init(values, generator=generator, **kwargs)
            param.copy_(values)

        for name, module in self.named_modules():
            if name.startswith("aggregator."):
                continue
            if isinstance(module, nn.Linear) and not name.startswith("heads."):
                fill(module.weight, nn.init.kaiming_uniform_, a=math.sqrt(5))
                bound = 1 / math.sqrt(module.in_features)
                fill(module.bias, nn.init.uniform_, a=-bound, b=bound)
            elif isinstance(module, nn.Embedding):
                fill(module.weight, nn.init.normal_)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
        r = self.config.rank
        for name, head in self.heads.items():
            d_in, d_out = self.shapes[name]
            lora_a = torch.empty(r, d_in, dtype=head.bias.dtype, device="cpu")
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5), generator=generator)
            lora_b = torch.zeros(r * d_out, dtype=head.bias.dtype, device="cpu")
            head.weight.zero_()
            head.bias.copy_(torch.cat((lora_a.flatten(), lora_b)))
        if self.aggregator is None:
            return
        for module in self.aggregator.modules():
            if isinstance(module, nn.MultiheadAttention):
                fill(module.in_proj_weight, nn.init.xavier_uniform_)
                module.in_proj_bias.zero_()
            elif isinstance(module, nn.Linear):
                fill(module.weight, nn.init.kaiming_uniform_, a=math.sqrt(5))
                bound = 1 / math.sqrt(module.in_features)
                fill(module.bias, nn.init.uniform_, a=-bound, b=bound)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
        for attention in (*(block.attn for block in self.aggregator.blocks), self.aggregator.pool):
            attention.out_proj.bias.zero_()
        fill(self.aggregator.queries, nn.init.xavier_uniform_)
        self.aggregator.output.weight.zero_()
        self.aggregator.output.bias.zero_()

    def forward(
        self, z_bar: Tensor, pairs: Tensor | None = None, mask: Tensor | None = None
    ) -> list[LoRAFactors]:
        """Emit one adapter per task.

        Args:
            z_bar: Task embeddings, `[batch, embed_dim]`, in H's parameter dtype and device: one
                pooled pair embedding per task (P3.2), the mean of its pair embeddings.
            pairs: The pair embeddings themselves, `[batch, n_max, embed_dim]`, in H's dtype and
                device, zero-padded (`pad_pair_sets`); required by a `set` H, refused by `mean`.
            mask: `[batch, n_max]` bool, True for a real pair; None means every row is real.

        Returns:
            One `LoRAFactors` per task, in batch order: `(A [r, d_in], B [d_out, r])` per key
            `layers.{i}.{block}.{m}`, keys in `factor_keys` order. The tensors are views of the
            heads' outputs (B is a transposed view); `inject` and `to_peft_state_dict` copy them.

        Raises:
            ValueError: If `z_bar` is not `[batch, embed_dim]`; if a `mean` H gets `pairs` or
                `mask`, or a `set` H gets no `pairs`, `pairs` is not `[batch, n_max, embed_dim]`,
                or `mask` is not `[batch, n_max]` bool with a real pair in every task.
        """
        if z_bar.ndim != 2 or z_bar.shape[1] != self.config.embed_dim:
            raise ValueError(
                f"expected z_bar [batch, {self.config.embed_dim}], got {list(z_bar.shape)}"
            )
        n_modules, batch, n_layers = len(self.module_names), z_bar.shape[0], self.base.n_layers
        grid = (n_modules, batch, n_layers, -1)
        task = self.task_encoder(z_bar)  # [batch, d/2]
        if self.aggregator is None:
            if pairs is not None or mask is not None:
                raise ValueError("a mean H reads z_bar only; pairs and mask are for aggregator set")
            task_rows = task[None, :, None].expand(grid)  # [n_modules, batch, n_layers, d/2]
        else:
            mask = self._check_set(pairs, mask, batch)
            delta = self.aggregator(pairs, mask)  # [batch, n_modules * n_layers, d/2]
            delta = delta.unflatten(1, (n_modules, n_layers)).transpose(0, 1)
            task_rows = task[None, :, None] + delta  # [n_modules, batch, n_layers, d/2]
        layer = self.layer_norm(self.layer_embedding.weight)  # [n_layers, d/4]
        module = self.module_norm(self.module_embedding.weight)  # [n_modules, d/4]
        codes = torch.cat(
            (
                task_rows,
                layer[None, None].expand(grid),
                module[:, None, None].expand(grid),
            ),
            dim=-1,
        )  # [n_modules, batch, n_layers, d]
        head_in = self.mlp3(self.blocks(self.mixer(codes)))  # [n_modules, batch, n_layers, H_in]
        r = self.config.rank
        emitted = {}
        for j, name in enumerate(self.module_names):
            d_in, d_out = self.shapes[name]
            out = self.heads[name](head_in[j])  # [batch, n_layers, r (d_in + d_out)]
            out_a, out_b = out.split((r * d_in, r * d_out), dim=-1)
            lora_a = out_a.unflatten(-1, (r, d_in))  # [batch, n_layers, r, d_in]
            lora_b = out_b.unflatten(-1, (r, d_out)).mT  # [batch, n_layers, d_out, r]
            emitted[name] = (lora_a, lora_b)
        return [
            {
                f"layers.{i}.{PROJECTIONS[name]}.{name}": (lora_a[b, i], lora_b[b, i])
                for i in range(n_layers)
                for name, (lora_a, lora_b) in emitted.items()
            }
            for b in range(batch)
        ]

    def _check_set(self, pairs: Tensor | None, mask: Tensor | None, batch: int) -> Tensor:
        """Validate a `set` H's `pairs` and `mask`; return the mask (all True when None)."""
        width = self.config.embed_dim
        if pairs is None:
            raise ValueError("a set H needs pairs [batch, n_max, embed_dim] beside z_bar")
        if pairs.ndim != 3 or pairs.shape[0] != batch or pairs.shape[2] != width:
            raise ValueError(f"expected pairs [{batch}, n_max, {width}], got {list(pairs.shape)}")
        if mask is None:
            return torch.ones(pairs.shape[:2], dtype=torch.bool, device=pairs.device)
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(pairs.shape[:2]):
            raise ValueError(
                f"expected a bool mask {list(pairs.shape[:2])}, got {mask.dtype} {list(mask.shape)}"
            )
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every task needs at least one real pair in mask")
        return mask
