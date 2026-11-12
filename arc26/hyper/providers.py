r"""Adapter providers from the hypernetwork H (task P4.1; BRIEF B5 P4.1, B2 Q3, Q4, B4; ADR-017).

The refinement loop takes its adapters from an injected ``AdapterProvider`` (``arc26.loop``,
ADR-004). Phase 2's is ``arc26.ttt.SGDProvider``, a continuation of the current adapter. Phase 4
recomputes the adapter from H on the enlarged demonstration set every round (B2 Q4), and polishes
it with a short SGD fit by default (B2 Q3, ADR-003):

$$\theta_t = H(\bar z(D_t)) \ \text{(HyperProvider)}, \qquad
\theta_t = \mathrm{polish}(H(\bar z(D_t)), s) \ \text{(HyperPolishProvider)},$$

where $\bar z(D)$ is the mean of P3.2's pair embeddings $\bar e_i$ over the pairs of $D$ (ADR-001),
each pair encoded once on the frozen base and kept in an ``EmbeddingCache``, and
$\mathrm{polish}(\cdot, s)$ is ``SGDProvider``'s fit for $s$ optimizer steps from H's adapter.

H's output is B4's ``Adapter``: PEFT-keyed float32 tensors plus ``{rank, alpha, target_modules}``.
``adapt`` copies it into the live PEFT model (``set_peft_model_state_dict``, P3.1's parity loader)
and returns that model, as ``SGDProvider`` returns its trained model, so every provider's adapter
goes through one decode path (``arc26.decode.decode_puzzle``). The live model's previous weights
are copied to the CPU first; ``rollback`` loads them back. ``fingerprint`` is a sha256 over the
adapter's tensors in key order. Math, data flow, and limits: ``docs/modules/providers.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import save_file
from torch import Tensor

from arc26 import timing
from arc26.hyper.encoder import (
    ArcPair,
    EmbeddingCache,
    PairEncoder,
    apply_view,
    pair_hash,
    render_pair,
    view_descriptors,
)
from arc26.hyper.hypernet import BaseShapes, Hypernet, HypernetConfig
from arc26.hyper.inject import ADAPTER_CONFIG, ADAPTER_WEIGHTS, peft_config, to_peft_state_dict
from arc26.loop import EPOCH_VIEWS, FINGERPRINT, RESTORE, SNAPSHOT

if TYPE_CHECKING:
    from arc26.loop import DemoSet
    from arc26.ttt import SGDProvider

# The PEFT adapter that arc26.solver.load wraps and arc26.ttt.fit resets.
ADAPTER_NAME = "default"
# Timing ops of H's GPU work. ENCODE (the cache misses' prefills, with their tokens) and H_FORWARD
# are P3.5's names for the same work in meta-training (arc26/hyper/train.py); SWAP is the copy of
# H's adapter into the live model. The snapshot, restore, and fingerprint keep the loop's names.
ENCODE = "encode"
H_FORWARD = "h_forward"
SWAP = "swap"
# s, the polish's optimizer steps: about 10% of the notebook's one-epoch count (B2 Q3, ADR-003),
# 128 steps at batch 1 (arc26/ttt.py), rounded up. The polish's SGDProvider takes it as `steps`;
# the governor accepts or declines the round at that price (choosing s from the budget is P4.3's).
POLISH_STEPS = math.ceil(EPOCH_VIEWS / 10)


def adapter_digest(state: Mapping[str, Tensor]) -> str:
    """Return the sha256 of an adapter's tensors: key, dtype, shape, and raw bytes, in key order.

    Why: the round's record shows a rollback was bit for bit by equal digests (ADR-015), and
    B5 P4.1 asks for deterministic emitted-adapter hashes. Sorted keys make the digest
    independent of the dict's insertion order, so H's export and the live model's state dict
    give the same digest exactly when they hold the same tensors.

    Args:
        state: PEFT state-dict key to tensor, on any device and of any dtype.

    Returns:
        64 lowercase hex characters.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().to("cpu").contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.view(torch.uint8).numpy())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class Adapter:
    """B4's Adapter: PEFT-keyed LoRA tensors plus the metadata that fixes their scale.

    Attributes:
        state: PEFT state-dict key to tensor, the keys of ``get_peft_model_state_dict``:
            ``base_model.model.model.layers.{i}.self_attn.{m}.lora_A.weight`` [r, d_in] and
            ``....lora_B.weight`` [d_out, r].
        rank: LoRA rank r of every pair.
        alpha: LoRA alpha; PEFT scales the delta by ``rslora_scaling(rank, alpha)``.
        target_modules: The projections adapted in every decoder layer.
    """

    state: Mapping[str, Tensor]
    rank: int
    alpha: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", dict(self.state))
        object.__setattr__(self, "target_modules", tuple(self.target_modules))

    def digest(self) -> str:
        """Return ``adapter_digest`` of the tensors.

        Returns:
            64 lowercase hex characters.
        """
        return adapter_digest(self.state)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write B4's files: ``adapter_model.safetensors`` and ``adapter_config.json``.

        Why: B4 saves the Adapter as safetensors; the layout is ``inject.save_adapter``'s, which
        ``PeftModel.from_pretrained`` loads offline with P3.1's parity.

        Args:
            path: Output directory, created if missing; existing adapter files are overwritten.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        tensors = {
            key: tensor.detach().to("cpu").contiguous() for key, tensor in self.state.items()
        }
        save_file(tensors, out / ADAPTER_WEIGHTS, metadata={"format": "pt"})
        config = peft_config(self.rank, self.alpha, self.target_modules)
        (out / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class EncodeShape:
    """What H's encode of a ``DemoSet`` would prefill: the inputs of P4.3's encode price.

    Attributes:
        pairs: Distinct pairs of ``DemoSet.pairs`` that the cache lacks.
        prefills: Sequences to prefill: ``pairs`` times the encoder's views.
        tokens: Real tokens over those sequences, as the encoder's ``encode_prefill`` record
            counts them.
    """

    pairs: int
    prefills: int
    tokens: int


@contextmanager
def _frozen_base(model: Any) -> Iterator[None]:
    # The encoder must see the frozen base (ADR-001; P3.5 encoded on the base, nothing injected):
    # PEFT's disable_adapter makes every LoRA layer return its base layer's output, bit for bit.
    disable = getattr(model, "disable_adapter", None)
    if disable is None:
        yield
        return
    with disable():
        yield


def _sync(device: torch.device) -> None:
    # A GPU op's span ends when its kernels have run.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _targets(value: Any) -> set[str] | None:
    # PEFT keeps target_modules as a set of names, or a regex string (no fixed set).
    if isinstance(value, str) or value is None:
        return None
    return set(value)


class HyperProvider:
    r"""The pure H provider: $\theta = H(\bar z(D))$, swapped into the live adapter (B5 P4.1).

    Why: H's adapter for the enlarged set, recomputed every round (B2 Q4, ADR-004), is the measured
    $s = 0$ arm of ADR-003 and the first half of the polish. ``adapt`` encodes the pairs of
    ``demos.pairs`` that the cache lacks on the frozen base (the live model's adapter disabled),
    each in its own encode call, so a cached embedding depends on its pair alone; takes $\bar z$ as
    the mean of the cached embeddings in pair-hash order, so the result does not depend on the
    pairs' order (a checkpoint with ``aggregator: set``, P3.3b, also gets the embeddings themselves
    in that order); runs H in evaluation mode; exports B4's ``Adapter``; copies the live adapter's
    weights to the CPU; and loads H's into it. Only ``demos.pairs`` is read, so the held-out pair
    reaches neither the encoder nor H (ADR-013). ``init``'s weights are kept for the rollback and
    are never an input to H. With a logger, the encode is recorded as ``encode`` (seconds, prefilled
    tokens), H's forward and export as ``h_forward``, the CPU copy as ``snapshot``, the load as
    ``swap``, ``rollback``'s load as ``restore``, and each digest as ``fingerprint``.

    Attributes:
        hypernet: H, in evaluation mode, on the device its forward runs on.
        encoder: P3.2's encoder on the live PEFT model, with training's layer, pool, and views.
        cache: Pair embeddings by pair hash, in the encoder's namespace.
        timing_logger: The worker's timing logger, or None.
        shapes: The live adapter's expected tensor shapes by PEFT key: H's scope.
    """

    def __init__(
        self,
        hypernet: Hypernet,
        encoder: PairEncoder,
        cache: EmbeddingCache,
        *,
        timing_logger: timing.TimingLogger | None = None,
    ) -> None:
        """Build the provider from H, the encoder, and the cache.

        Args:
            hypernet: H; set to evaluation mode here.
            encoder: P3.2's encoder, built on the live PEFT model (its ``disable_adapter`` gives
                the frozen base) or on a model without adapters.
            cache: An ``EmbeddingCache`` of the encoder's layer, pool, and view set.
            timing_logger: The worker's timing logger, or None to record nothing.

        Raises:
            ValueError: If H's input width is not the encoder's hidden size, the cache's
                namespace is not the encoder's, or the encoder's model holds LoRA layers that it
                cannot disable.
        """
        if hypernet.config.embed_dim != encoder.hidden_size:
            raise ValueError(
                f"H reads z_bar of width {hypernet.config.embed_dim}; the encoder pools "
                f"{encoder.hidden_size}"
            )
        expected = {
            "layer": encoder.layer,
            "pool": encoder.pool,
            "view_set_id": encoder.view_set_id,
        }
        found = {name: cache.namespace.get(name) for name in expected}
        if found != expected:
            raise ValueError(f"the cache holds embeddings of {found}; the encoder makes {expected}")
        model = encoder.model
        if getattr(model, "disable_adapter", None) is None and any(
            isinstance(module, BaseTunerLayer) for module in model.modules()
        ):
            raise ValueError(
                "the encoder's model holds LoRA layers it cannot disable: build the encoder on the "
                "PEFT model, whose disable_adapter gives the frozen base"
            )
        self.hypernet = hypernet.eval()
        self.encoder = encoder
        self.cache = cache
        self.timing_logger = timing_logger
        # H's scope as PEFT keys and shapes, named by the export itself on meta tensors.
        rank, meta = hypernet.config.rank, {}
        for key in hypernet.factor_keys:
            d_in, d_out = hypernet.shapes[key.rpartition(".")[2]]
            meta[key] = (
                torch.empty(rank, d_in, device="meta"),
                torch.empty(d_out, rank, device="meta"),
            )
        self.shapes = {key: tuple(t.shape) for key, t in to_peft_state_dict(meta).items()}
        # The live adapter's weights before the last adapt, on the CPU; and the puzzle of the
        # last adapt, for the timing records of rollback and fingerprint.
        self._snapshot: dict[str, Tensor] | None = None
        self._puzzle_id: str | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        *,
        model: Any,
        tokenizer: Any,
        cache_root: str | os.PathLike[str],
        timing_logger: timing.TimingLogger | None = None,
    ) -> HyperProvider:
        r"""Build the provider from a P3.5 checkpoint, with training's encoder settings.

        Why: H was trained on $\bar z$ from the checkpoint's encoder settings (layer, pool,
        views; ``EncoderSettings``: "they must equal the test-time encoder's"), so they are read
        from it, never restated; the cache's namespace takes the checkpoint's ``model_id``.

        Args:
            path: A checkpoint file of ``arc26.hyper.train.save_checkpoint``, or its folder
                (``latest.json``).
            model: The live PEFT model; H goes to its device, and its config gives H's shapes.
            tokenizer: The base model's tokenizer.
            cache_root: Folder of the embedding cache; one per worker (one writer at a time).
            timing_logger: The worker's timing logger, or None.

        Returns:
            The provider.

        Raises:
            FileNotFoundError: If there is no checkpoint at ``path``.
            ValueError: As the checkpoint, the configs, and the constructor raise it.
        """
        from arc26.hyper.train import EncoderSettings, load_checkpoint, resolve_checkpoint

        payload = load_checkpoint(resolve_checkpoint(Path(path)))
        config = payload["config"]
        settings = EncoderSettings(**config["encoder"])
        hypernet = Hypernet(
            HypernetConfig(**config["hypernet"]), BaseShapes.from_config(model.config)
        )
        hypernet.load_state_dict(payload["hypernet"])
        hypernet.to(_device(model))
        encoder = PairEncoder(
            model,
            tokenizer,
            layer=settings.layer,
            pool=settings.pool,
            views=view_descriptors(settings.n_perm, settings.view_seed),
            batch_tokens=settings.batch_tokens,
        )
        cache = EmbeddingCache(
            cache_root,
            model_id=config["model_id"],
            layer=settings.layer,
            pool=settings.pool,
            view_set_id=encoder.view_set_id,
        )
        return cls(hypernet, encoder, cache, timing_logger=timing_logger)

    def _record(
        self, puzzle_id: str | None, op: str, start: float, tokens: int | None = None
    ) -> None:
        if self.timing_logger is not None:
            self.timing_logger.record(puzzle_id, op, time.perf_counter() - start, tokens)

    def _pairs(self, demos: DemoSet) -> tuple[list[str], dict[str, ArcPair]]:
        # The training and voting set only: DemoSet.pairs never holds the held-out pair.
        pairs = [pair.task_pair() for pair in demos.pairs]
        if not pairs:
            raise ValueError(f"{demos.puzzle_id}: no pair to emit an adapter from")
        hashes = [pair_hash(pair) for pair in pairs]
        return hashes, dict(zip(hashes, pairs, strict=True))

    def encode_shape(self, demos: DemoSet) -> EncodeShape:
        """Count what ``adapt`` would encode for ``demos``, for P4.3's cost model.

        Why: the encode is priced per prefilled token (``P3.2-02``: 0.181 s per 1,000 tokens on
        an L4), and only cache misses are encoded. The views are rendered and tokenized on the
        CPU as the encoder renders them; the model does not run and the cache is not written.

        Args:
            demos: The state ``adapt`` would read, without its held-out pair.

        Returns:
            The cache misses, their prefills, and their real tokens.

        Raises:
            ValueError: If ``demos`` has no pair.
        """
        hashes, by_hash = self._pairs(demos)
        missing = self.cache.missing(hashes)
        views = self.encoder.views
        tokens = sum(
            len(render_pair(apply_view(by_hash[key], view), self.encoder.formatter).token_ids)
            for key in missing
            for view in views
        )
        return EncodeShape(pairs=len(missing), prefills=len(missing) * len(views), tokens=tokens)

    def _encode_missing(
        self, puzzle_id: str, hashes: Sequence[str], pairs: Mapping[str, ArcPair]
    ) -> int:
        missing = sorted(self.cache.missing(hashes))
        if not missing:
            return 0
        tokens: list[int] = []
        previous = self.encoder.timing

        def count(record: dict[str, Any], previous: Callable[..., Any] | None = previous) -> None:
            if record["op"] == "encode_prefill":
                tokens.append(int(record["tokens"]))
            if previous is not None:
                previous(record)

        start = time.perf_counter()
        self.encoder.timing = count
        try:
            with _frozen_base(self.encoder.model):
                # One call per pair: its prefill batches hold its own views only, so its
                # embedding does not depend on which other pairs were missing.
                means = {
                    key: self.encoder.encode_pairs([pairs[key]], puzzle_id)[0] for key in missing
                }
        finally:
            self.encoder.timing = previous
        self.cache.put(means)
        self._record(puzzle_id, ENCODE, start, sum(tokens))
        return len(missing)

    def _emit(self, demos: DemoSet) -> tuple[Adapter, int]:
        hashes, pairs = self._pairs(demos)
        encoded = self._encode_missing(demos.puzzle_id, hashes, pairs)
        # Pair-hash order fixes the float32 summation order; repeats count as in training.
        rows = self.cache.get(sorted(hashes))  # [n, hidden]
        z_bar = rows.mean(dim=0)  # [hidden]
        start = time.perf_counter()
        param = next(self.hypernet.parameters())
        inputs = (z_bar[None].to(device=param.device, dtype=param.dtype),)
        if self.hypernet.aggregator is not None:
            # A set H reads the pairs themselves, in the same pair-hash order (one task, so the
            # mask is all real pairs).
            inputs += (rows[None].to(device=param.device, dtype=param.dtype),)
        with torch.no_grad():
            factors = self.hypernet.eval()(*inputs)
            state = to_peft_state_dict(factors[0])
        _sync(param.device)
        self._record(demos.puzzle_id, H_FORWARD, start)
        config = self.hypernet.config
        adapter = Adapter(state, config.rank, config.alpha, config.target_modules)
        return adapter, encoded

    def emit(self, demos: DemoSet) -> Adapter:
        r"""Return H's adapter for ``demos``: B4's ``Adapter``, float32, on H's device.

        Encodes the pairs the cache lacks (then caches them) and runs H once on $\bar z$ (a
        ``set`` H also on the pair embeddings); the live model is not touched.

        Args:
            demos: The state, without its held-out pair; only ``demos.pairs`` is read.

        Returns:
            The adapter, one ``(lora_A, lora_B)`` pair per H factor key.

        Raises:
            ValueError: If ``demos`` has no pair.
        """
        return self._emit(demos)[0]

    def _live_state(self, model: Any) -> dict[str, Tensor]:
        configs = getattr(model, "peft_config", None)
        if not isinstance(configs, Mapping) or ADAPTER_NAME not in configs:
            raise TypeError(
                f"the live model must be a PEFT model with the {ADAPTER_NAME!r} adapter, as "
                "arc26.solver.load wraps it"
            )
        live = configs[ADAPTER_NAME]
        config = self.hypernet.config
        names = ("r", "lora_alpha", "use_rslora", "target_modules")
        r, alpha, rslora, targets = (getattr(live, name, None) for name in names)
        if (r, alpha, rslora, _targets(targets)) != (
            config.rank, config.alpha, True, set(config.target_modules)
        ):  # fmt: skip
            raise ValueError(
                f"the live adapter is r {r}, alpha {alpha}, rsLoRA {rslora}, targets {targets}; "
                f"H emits r {config.rank}, alpha {config.alpha}, rsLoRA, targets "
                f"{list(config.target_modules)}"
            )
        state = get_peft_model_state_dict(
            model, adapter_name=ADAPTER_NAME, save_embedding_layers=False
        )
        shapes = {key: tuple(tensor.shape) for key, tensor in state.items()}
        if shapes != self.shapes:
            missing = sorted(self.shapes.keys() - shapes.keys())
            extra = sorted(shapes.keys() - self.shapes.keys())
            other = sorted(
                k for k in shapes.keys() & self.shapes.keys() if shapes[k] != self.shapes[k]
            )
            raise ValueError(
                f"the live adapter's tensors are not H's: missing {missing[:2]} ({len(missing)}), "
                f"extra {extra[:2]} ({len(extra)}), other shapes {other[:2]} ({len(other)})"
            )
        return state

    def adapt(self, demos: DemoSet, init: Any, budget_s: float) -> Any:
        """Load H's adapter for ``demos`` into the live model, keeping its weights for a rollback.

        Args:
            demos: The state without its held-out pair (``refine`` passes it so).
            init: The live PEFT model, holding the current adapter; its weights are copied to the
                CPU and replaced, never read by H (B2 Q4).
            budget_s: Seconds left to the puzzle's deadline; printed, not enforced (the governor
                priced the round within it).

        Returns:
            ``init``, now holding H's adapter.

        Raises:
            TypeError: If ``init`` is not a PEFT model with the ``default`` adapter.
            ValueError: If its adapter is not H's scope (rank, alpha, rsLoRA, targets, tensor
                keys and shapes), or ``demos`` has no pair.
        """
        # An earlier round's copy must never be restored onto this round's adapter: a copy that
        # raises leaves no snapshot, and the rollback then has nothing to load.
        self._snapshot = None
        self._puzzle_id = demos.puzzle_id
        live = self._live_state(init)
        adapter, encoded = self._emit(demos)
        start = time.perf_counter()
        self._snapshot = {key: tensor.detach().to("cpu", copy=True) for key, tensor in live.items()}
        self._record(demos.puzzle_id, SNAPSHOT, start)
        start = time.perf_counter()
        set_peft_model_state_dict(init, dict(adapter.state), adapter_name=ADAPTER_NAME)
        _sync(_device(init))
        self._record(demos.puzzle_id, SWAP, start)
        print(
            f"H's adapter for {demos.puzzle_id} from {len(demos.pairs)} pairs ({encoded} "
            f"encoded), {budget_s:.0f} s left"
        )
        return init

    def rollback(self, previous: Any) -> Any:
        """Load the weights the last ``adapt`` replaced back into the live model.

        Args:
            previous: The live model; returned unchanged when the last ``adapt`` took no copy
                (it raised before, or none ran).

        Returns:
            ``previous``, holding the adapter it held before the last ``adapt``, bit for bit.
        """
        if self._snapshot is None:
            return previous
        start = time.perf_counter()
        set_peft_model_state_dict(previous, dict(self._snapshot), adapter_name=ADAPTER_NAME)
        _sync(_device(previous))
        self._record(self._puzzle_id, RESTORE, start)
        return previous

    def fingerprint(self, adapter: Any) -> str:
        """Return ``adapter_digest`` of a live model's adapter or of an emitted ``Adapter``.

        Args:
            adapter: A PEFT model (its ``default`` adapter is read) or an ``Adapter``.

        Returns:
            64 lowercase hex characters; equal exactly for bit-for-bit equal tensors.
        """
        start = time.perf_counter()
        if isinstance(adapter, Adapter):
            state = adapter.state
        else:
            state = get_peft_model_state_dict(
                adapter, adapter_name=ADAPTER_NAME, save_embedding_layers=False
            )
        digest = adapter_digest(state)
        self._record(self._puzzle_id, FINGERPRINT, start)
        return digest

    def fit_shape(self, demos: DemoSet) -> tuple[int, int]:
        """Return the train term of the round's price: none, since H trains nothing.

        Why: ``refine`` prices ``fit_shape`` with the governor's train model, fitted on SGD fits;
        H's encode and forward are other ops, exposed by ``encode_shape`` for P4.3's cost model
        (ADR-017). Zero tokens cost zero seconds in ``CostModel``.

        Args:
            demos: The state ``adapt`` would read.

        Returns:
            ``(0, 0)``: no tokens trained, no optimizer step.
        """
        return 0, 0


class HyperPolishProvider:
    r"""H's adapter polished by SGD: $\theta = \mathrm{polish}(H(\bar z(D)), s)$ (B2 Q3, ADR-003).

    Why: the production default of ADR-003, a safety net when H is mediocre at the price of $s$
    steps. It composes the two providers: ``HyperProvider.adapt`` copies the live adapter to the
    CPU and loads H's adapter for ``demos``; ``SGDProvider.adapt`` then copies that (H's adapter as
    the live model holds it) and runs ``fit`` from it with ``max_steps`` $s$, its ``steps``, on
    ``demos.task()`` (the pairs under fit's 128 views). Every round starts from H again, never
    from ``init`` (B2 Q4). ``rollback`` undoes the two in reverse: ``SGDProvider.rollback`` (H's
    adapter back, inference mode, the trainer's memory released), then ``HyperProvider.rollback``
    (the weights the round started from), so the result is the snapshot of ``init`` bit for bit
    whatever the fit did. ``arc26/ttt.py`` is used as it stands (ADR-017).

    Attributes:
        hyper: The H provider (one per worker: H, the encoder, and the cache are shared).
        sgd: The polish: an ``arc26.ttt.SGDProvider`` of the puzzle with ``steps`` $= s$.
    """

    def __init__(self, hyper: HyperProvider, sgd: SGDProvider) -> None:
        """Compose the two providers.

        Args:
            hyper: The H provider.
            sgd: The puzzle's ``SGDProvider``; its ``steps`` is the polish's $s$ (``POLISH_STEPS``
                by default in the wiring).

        Raises:
            TypeError: If ``hyper`` is not a ``HyperProvider`` or ``sgd`` lacks ``adapt``,
                ``rollback``, ``fit_shape``, or an integer ``steps``.
        """
        if not isinstance(hyper, HyperProvider):
            raise TypeError(f"hyper must be a HyperProvider; got {hyper!r}")
        for name in ("adapt", "rollback", "fit_shape"):
            if not callable(getattr(sgd, name, None)):
                raise TypeError(f"sgd must have a {name} method (arc26.ttt.SGDProvider)")
        if type(getattr(sgd, "steps", None)) is not int:
            raise TypeError("sgd must have integer steps, the polish's s")
        self.hyper = hyper
        self.sgd = sgd
        # Whether the last adapt reached the polish, so that its rollback applies: an SGDProvider
        # that did not run this round may hold an earlier round's copy.
        self._polishing = False

    @property
    def steps(self) -> int:
        """$s$, the polish's optimizer steps: the ``SGDProvider``'s ``steps``."""
        return self.sgd.steps

    def adapt(self, demos: DemoSet, init: Any, budget_s: float) -> Any:
        """Load H's adapter for ``demos``, then polish it for $s$ steps.

        Args:
            demos: The state without its held-out pair.
            init: The live PEFT model, holding the current adapter.
            budget_s: Seconds left to the puzzle's deadline; the polish gets what is left after
                H's step.

        Returns:
            The model in inference mode with the polished adapter, as ``fit`` returns it.

        Raises:
            TypeError: As ``HyperProvider.adapt`` raises it.
            ValueError: As ``HyperProvider.adapt`` raises it.
        """
        start = time.perf_counter()
        self._polishing = False
        model = self.hyper.adapt(demos, init, budget_s)
        self._polishing = True
        left = max(0.0, budget_s - (time.perf_counter() - start))
        return self.sgd.adapt(demos, model, left)

    def rollback(self, previous: Any) -> Any:
        """Undo the polish, then H's swap: the live model holds the round's starting weights.

        Args:
            previous: The model the last ``adapt`` started from.

        Returns:
            The live model in inference mode with the weights it held before the last ``adapt``.
        """
        model = self.sgd.rollback(previous) if self._polishing else previous
        return self.hyper.rollback(model)

    def fingerprint(self, adapter: Any) -> str:
        """Return the H provider's sha256 digest of the adapter (``adapter_digest``).

        Args:
            adapter: A PEFT model or an ``Adapter``.

        Returns:
            64 lowercase hex characters.
        """
        return self.hyper.fingerprint(adapter)

    def fit_shape(self, demos: DemoSet) -> tuple[int, int]:
        """Return the polish's tokens and steps for the governor, as ``SGDProvider`` counts them.

        Args:
            demos: The state ``adapt`` would train on, without its held-out pair.

        Returns:
            ``(tokens, steps)``: the tokens of $s$ of fit's 128 views, and $s$.
        """
        return self.sgd.fit_shape(demos)

    def encode_shape(self, demos: DemoSet) -> EncodeShape:
        """Return what H's encode of ``demos`` would prefill (``HyperProvider.encode_shape``).

        Args:
            demos: The state ``adapt`` would read.

        Returns:
            The cache misses, their prefills, and their real tokens.
        """
        return self.hyper.encode_shape(demos)
