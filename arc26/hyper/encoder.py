"""Frozen-base pair encoder and content-addressed embedding cache (task P3.2, BRIEF B2 Q1).

H reads a task through a fixed-size embedding of its demonstration pairs (ADR-001). Each pair is
rendered in the reference notebook's exact format under a fixed set of augmented views, run
through the first ``L`` decoder layers of the frozen base model, and mean-pooled over its
output-grid token positions ``O``::

    e[i, a] = mean over t in O[i, a] of h_L[t]        one pair i under view a, [hidden]
    e_bar[i] = mean over views a of e[i, a]            the pair embedding, [hidden]
    z_bar = mean over pairs i of e_bar[i]              the task embedding, [hidden]

``h_L`` is the residual stream after ``L`` decoder layers, Hugging Face's ``hidden_states[L]``.
The MLP that maps ``z_bar`` to ``z`` belongs to H (P3.3). Rendering and augmentation reuse the
notebook's own functions (``arc26.nvarc.arc_loader``); nothing is reimplemented. Pair embeddings
are cached by content (``EmbeddingCache``), so meta-training (P3.5) and the refinement loop
encode each pair once. Math and data flow: ``docs/modules/encoder.md``.

Timing: ``PairEncoder`` reports each encode call through an optional ``timing`` callback that
receives one record ``{puzzle_id, op, seconds, tokens}`` (BRIEF B4) per stage. The default is no
callback; ``arc26/timing.py`` (P0.3, another branch) will supply the JSONL logger to pass here.
"""

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save as safetensors_bytes
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter, convert_grid_to_string

ArcPair = Mapping[str, Any]
"""A demonstration pair in the ARC JSON form, ``{"input": grid, "output": grid}``.

A grid is a non-empty rectangular 2-D list or array of integer colors 0 to 9. A B4 ``Pair``
converts with ``{"input": p.input, "output": p.output}``.
"""

TimingCallback = Callable[[dict[str, Any]], None]
"""Receives one timing record ``{"puzzle_id", "op", "seconds", "tokens"}`` per encode stage."""

POOLS = ("output", "input")
# The 8 dihedral transforms in the order ArcDataset.augment builds them: mod(np.transpose,
# keep=True) gives [identity, transpose], then mod(np.rot90, n=3, keep=True) appends one, two,
# and three quarter turns of each. tests/test_encoder.py checks this against augment itself.
DIHEDRAL = (
    "",
    "transpose",
    "rot90",
    "transpose.rot90",
    "rot90.rot90",
    "transpose.rot90.rot90",
    "rot90.rot90.rot90",
    "transpose.rot90.rot90.rot90",
)
# The decode path draws its 16 views with augment(n=2, seed=2) (arc_solver.worker); with the same
# seed, the encoder's default views are that path's views of a task with one test input.
DEFAULT_N_PERM = 2
DEFAULT_VIEW_SEED = 2
DEFAULT_LAYER = 24
# Padded tokens per prefill batch; see docs/runbooks/cache_embeddings.md for the derivation.
DEFAULT_BATCH_TOKENS = 16384
# QwenFormatter.fmt_query starts every pair with this text; render_pair checks it.
USER_HEAD = "<|im_start|>user\n"
CACHE_SCHEMA = 1
_HEX = frozenset("0123456789abcdef")


def _grid(grid: Any) -> np.ndarray:
    """Return ``grid`` as a validated int64 array.

    Args:
        grid: A non-empty rectangular 2-D list or array of integer colors 0 to 9.

    Returns:
        The grid as an int64 array, ``[rows, cols]``.

    Raises:
        ValueError: If the grid is ragged, not 2-D, empty, not integer, or outside 0 to 9.
    """
    try:
        arr = np.asarray(grid)
    except ValueError as err:
        raise ValueError(f"grid is not rectangular: {err}") from err
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"a grid must be a non-empty 2-D array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"grid cells must be integers, got dtype {arr.dtype}")
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError("grid cells must be colors 0 to 9")
    return arr.astype(np.int64, copy=False)


def pair_hash(pair: ArcPair) -> str:
    """Return the content address of a pair: sha256 of a canonical encoding of its two grids.

    The encoding is the compact JSON object ``{"input":[[...]],"output":[[...]]}``: keys sorted,
    no spaces, cells as integers, rows in order. For example ``{"input": [[1]], "output": [[2]]}``
    hashes the bytes ``{"input":[[1]],"output":[[2]]}``. Lists and numpy arrays of any integer
    dtype hash alike. It is the cache key here; P3.4's leakage check may reuse it.

    Args:
        pair: A pair with ``"input"`` and ``"output"`` grids.

    Returns:
        64 lowercase hex characters.

    Raises:
        ValueError: If a grid is not a valid ARC grid (see ``ArcPair``).
        KeyError: If ``pair`` lacks ``"input"`` or ``"output"``.
    """
    payload = json.dumps(
        {"input": _grid(pair["input"]).tolist(), "output": _grid(pair["output"]).tolist()},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def view_descriptors(n_perm: int = DEFAULT_N_PERM, seed: int = DEFAULT_VIEW_SEED) -> list[str]:
    """Return the augmentation descriptors of the encoder's view set, in the notebook's order.

    Why: ``ArcDataset.augment`` builds the same views but reseeds numpy's global generator, which
    would change every other consumer's random stream. Here the color permutations come from a
    local ``RandomState(seed)``, drawn in augment's order (round by round, the 8 dihedral
    transforms within each round) and spelled as ``permute_rnd_all_`` spells them, so
    ``view_descriptors(2, 2)`` equals the descriptors of ``augment(n=2, seed=2)`` for one key.

    Args:
        n_perm: Color permutations per dihedral transform; 0 gives the 8 dihedral views alone.
        seed: Seed of the permutation draws.

    Returns:
        ``8 * max(n_perm, 1)`` descriptors such as ``"transpose.rot90.permute3012456789"``; the
        identity is ``""``. ``apply_view`` applies one.

    Raises:
        ValueError: If ``n_perm`` or ``seed`` is negative.
    """
    if n_perm < 0 or seed < 0:
        raise ValueError(f"n_perm and seed must be non-negative, got {n_perm} and {seed}")
    if n_perm == 0:
        return list(DIHEDRAL)
    rng = np.random.RandomState(seed)
    views = []
    for _ in range(n_perm):
        for transform in DIHEDRAL:
            permute = "permute" + "".join(map(str, rng.permutation(10).tolist()))
            views.append(f"{transform}.{permute}" if transform else permute)
    return views


def view_set_id(views: Sequence[str]) -> str:
    """Return a short content id of a view set, part of the cache key.

    Args:
        views: Augmentation descriptors, in order.

    Returns:
        The first 16 hex characters of the sha256 of the JSON list of descriptors.
    """
    return hashlib.sha256(json.dumps(list(views)).encode("ascii")).hexdigest()[:16]


def apply_view(pair: ArcPair, descriptor: str) -> dict[str, np.ndarray]:
    """Apply one augmentation descriptor to both grids of a pair with the notebook's code.

    Args:
        pair: A pair with ``"input"`` and ``"output"`` grids.
        descriptor: Dot-separated operations as ``view_descriptors`` returns them, applied left
            to right by ``ArcDataset.forward_mod``; ``""`` is the identity.

    Returns:
        ``{"input": array, "output": array}``, the transformed grids.

    Raises:
        ValueError: If a grid is not a valid ARC grid.
        NotImplementedError: If the descriptor names an unknown operation (``forward_mod``).
        AssertionError: If a ``permute`` operation is not a permutation of 0 to 9
            (``permute_mod``).
    """
    key = f"view.{descriptor}" if descriptor else "view"
    return {name: ArcDataset.forward_mod(_grid(pair[name]), key) for name in ("input", "output")}


def augmented_views(
    pair: ArcPair, n_perm: int = DEFAULT_N_PERM, seed: int = DEFAULT_VIEW_SEED
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Return a pair under every view of ``view_descriptors(n_perm, seed)``.

    Args:
        pair: A pair with ``"input"`` and ``"output"`` grids.
        n_perm: Color permutations per dihedral transform.
        seed: Seed of the permutation draws.

    Returns:
        ``(descriptor, pair)`` for each view, in descriptor order.
    """
    return [(d, apply_view(pair, d)) for d in view_descriptors(n_perm, seed)]


class RenderedPair(NamedTuple):
    r"""One pair in the notebook's token format, with the token spans of its two grids.

    Attributes:
        token_ids: The tokens of ``QwenFormatter.fmt_train([pair])``, ``[seq]``.
        output_positions: Positions of the output-grid tokens: after ``assistant\n``, before
            the closing ``<|im_end|>``.
        input_positions: Positions of the input-grid tokens: after ``user\n``, before the
            first ``<|im_end|>``.
    """

    token_ids: list[int]
    output_positions: range
    input_positions: range


def render_pair(pair: ArcPair, formatter: QwenFormatter) -> RenderedPair:
    r"""Tokenize one pair exactly as the notebook trains on it and locate its grid tokens.

    The text is ``<|im_start|>user\n{input}<|im_end|><|im_start|>assistant\n{output}<|im_end|>``
    (``QwenFormatter.fmt_train``). Spans come from tokenizing the text up to each grid boundary:
    the tokens of every such prefix must be a prefix of the pair's tokens, so no assumption of one
    token per character is made. Each span must then decode back to its grid through the
    notebook's own ``QwenFormatter.convert_tokens_to_array``, and its decoded text, spaces
    removed, must equal the grid's text exactly; the second check catches a span that holds a
    neighboring token, which the notebook's whitespace-tolerant decoder accepts.

    Args:
        pair: A pair with ``"input"`` and ``"output"`` grids, each at most 30 by 30 (the limit of
            ``convert_tokens_to_array``).
        formatter: The notebook's formatter around the base model's tokenizer.

    Returns:
        The token ids and the output and input spans.

    Raises:
        ValueError: If a grid is invalid, the formatter's text is not the notebook's pair
            format, a prefix does not tokenize to a prefix of the pair's tokens, or a span does
            not decode back to exactly its grid.
    """
    grids = {"input": _grid(pair["input"]), "output": _grid(pair["output"])}
    query = formatter.fmt_query([grids])
    text = formatter.fmt_train([grids])
    in_text = convert_grid_to_string(grids["input"])
    out_text = convert_grid_to_string(grids["output"])
    if not (
        text.startswith(query)
        and query.startswith(USER_HEAD + in_text)
        and text[len(query) :].startswith(out_text)
    ):
        raise ValueError("formatter text is not the notebook's pair format")
    ids = list(formatter.tokenizer.encode(text))

    def boundary(n_chars: int) -> int:
        head = list(formatter.tokenizer.encode(text[:n_chars]))
        if ids[: len(head)] != head:
            raise ValueError(
                f"tokens of the first {n_chars} characters are not a prefix of the pair's tokens; "
                "the tokenizer merges across a grid boundary"
            )
        return len(head)

    in_start = len(USER_HEAD)
    spans = {
        "input": range(boundary(in_start), boundary(in_start + len(in_text))),
        "output": range(boundary(len(query)), boundary(len(query) + len(out_text))),
    }
    for name, span in spans.items():
        # convert_tokens_to_array drops its last token (the end marker it expects), so pass one.
        decoded = formatter.convert_tokens_to_array(ids[span.start : span.stop + 1])
        if decoded is None or not np.array_equal(decoded, grids[name]):
            raise ValueError(f"the {name} positions do not decode back to the {name} grid")
        # The notebook's decoder strips whitespace and skips non-digits, so a span holding one
        # token too many (a neighboring newline, say) would pass it; the exact text would not.
        text_of_span = formatter.tokenizer.decode(ids[span.start : span.stop]).replace(" ", "")
        if text_of_span != (in_text if name == "input" else out_text):
            raise ValueError(f"the {name} positions do not decode to exactly the {name} grid text")
    return RenderedPair(ids, spans["output"], spans["input"])


class _StopForward(Exception):
    """Raised by the capture hook once layer L has run, so no later layer runs."""


def _token_batches(order: Sequence[int], lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Group sequence indices, longest first, so that rows times width stays within ``budget``.

    Args:
        order: Sequence indices sorted by descending length.
        lengths: Token count of every sequence.
        budget: Maximum padded tokens per batch; a longer sequence runs alone.

    Returns:
        Batches of indices; each batch's width is its first sequence's length.
    """
    batches: list[list[int]] = []
    for i in order:
        if batches and (len(batches[-1]) + 1) * lengths[batches[-1][0]] <= budget:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


class PairEncoder:
    """Embed demonstration pairs with the frozen base model (ADR-001).

    For each pair and each view, one prefill runs through the first ``layer`` decoder layers
    only: a forward hook on the last of them captures its output and stops the forward, so later
    layers, the final norm, and the LM head never run. The captured residual stream equals
    Hugging Face's ``hidden_states[layer]``. Prefills are batched by a padded-token budget,
    longest sequence first, right-padded; pooling reads only each sequence's own grid span, so
    padding never enters an embedding. Pooling and every later mean accumulate in fp32.

    The model is used as given: keep it in eval mode and without active adapters, since its
    current weights define the embedding.

    Args:
        model: A decoder-only Hugging Face model whose decoder (``get_decoder()``, or the model
            itself) has ``layers``; Qwen3 in this project.
        tokenizer: The base model's tokenizer.
        layer: ``L``, the number of decoder layers to run, ``0 <= L < num_hidden_layers``. 0
            pools the token embeddings. The final layer is excluded because Hugging Face's last
            ``hidden_states`` entry is after the final norm, not the residual stream.
        pool: ``"output"`` pools the output-grid tokens (the default, Q1); ``"input"`` the
            input-grid tokens (P3.7's overturn test).
        views: Augmentation descriptors (see ``view_descriptors``); None means
            ``view_descriptors()``, the decode path's 16 views.
        batch_tokens: Maximum padded tokens per prefill batch (rows times the longest length).
        timing: Optional callback for the timing records; None records nothing.

    Raises:
        TypeError: If the model has no decoder layer list.
        ValueError: If ``layer``, ``pool``, ``batch_tokens``, or a view descriptor is invalid.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        layer: int = DEFAULT_LAYER,
        pool: str = "output",
        views: Sequence[str] | None = None,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
        timing: TimingCallback | None = None,
    ) -> None:
        decoder = model.get_decoder() if hasattr(model, "get_decoder") else model
        layers = getattr(decoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("the model's decoder has no `layers` ModuleList")
        if not 0 <= layer < len(layers):
            raise ValueError(f"layer must be in [0, {len(layers) - 1}], got {layer}")
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        if batch_tokens < 1:
            raise ValueError(f"batch_tokens must be positive, got {batch_tokens}")
        self.views = list(view_descriptors() if views is None else views)
        if not self.views:
            raise ValueError("the view set is empty")
        probe = {"input": [[0, 1, 2], [3, 4, 5]], "output": [[6, 7], [8, 9]]}
        for descriptor in self.views:
            try:
                apply_view(probe, descriptor)
            except (AssertionError, NotImplementedError) as err:
                raise ValueError(f"invalid view descriptor {descriptor!r}") from err
        self.model = model
        self.formatter = QwenFormatter(tokenizer)
        self.layer = layer
        self.pool = pool
        self.batch_tokens = batch_tokens
        self.timing = timing
        self.hidden_size = int(model.config.hidden_size)
        self.view_set_id = view_set_id(self.views)
        self._decoder = decoder
        self._capture = decoder.get_input_embeddings() if layer == 0 else layers[layer - 1]
        pad = tokenizer.pad_token_id
        # Any id works: right padding puts pads after every real token, and the attention mask
        # hides them, so they never reach a pooled position.
        self._pad_id = 0 if pad is None else int(pad)

    @property
    def device(self) -> torch.device:
        """The device of the model's parameters, where prefills run."""
        return next(self.model.parameters()).device

    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Run the first ``layer`` decoder layers and return their output.

        Args:
            input_ids: Token ids, right-padded, ``[batch, seq]``.
            attention_mask: 1 at real tokens, 0 at padding, ``[batch, seq]``.

        Returns:
            The residual stream after ``layer`` layers, ``hidden_states[layer]`` in Hugging
            Face's convention, in the model's dtype, ``[batch, seq, hidden]``.

        Raises:
            RuntimeError: If the forward ends without reaching the capture layer.
        """
        captured: list[Tensor] = []

        def capture(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            captured.append(output[0] if isinstance(output, tuple) else output)
            raise _StopForward

        handle = self._capture.register_forward_hook(capture)
        try:
            # Output recording stays off: transformers 4.55 patches layer forwards to record
            # them and restores the patches only on a normal return.
            self._decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
        except _StopForward:
            pass
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("the forward ended before the capture layer ran")
        return captured[0]

    def encode_views(self, pairs: Sequence[ArcPair], puzzle_id: str | None = None) -> Tensor:
        """Embed every pair under every view.

        Emits two timing records: ``encode_render`` (CPU: views, text, tokens, span checks) and
        ``encode_prefill`` (the batched forward and pooling, synchronized by the copy of the
        result to the CPU). ``tokens`` is the number of real tokens over all views.

        Args:
            pairs: Demonstration pairs, at least one.
            puzzle_id: Task id for the timing records.

        Returns:
            ``e[i, a]``, fp32 on the CPU, ``[n_pairs, n_views, hidden]``.

        Raises:
            ValueError: If ``pairs`` is empty or a pair fails ``render_pair``.
            FloatingPointError: If an embedding is not finite.
        """
        if not pairs:
            raise ValueError("no pairs to encode")
        start = time.perf_counter()
        rendered = [
            render_pair(apply_view(pair, view), self.formatter)
            for pair in pairs
            for view in self.views
        ]
        tokens = sum(len(r.token_ids) for r in rendered)
        rendered_at = time.perf_counter()
        self._emit(puzzle_id, "encode_render", rendered_at - start, tokens)
        pooled = self._prefill(rendered)
        self._emit(puzzle_id, "encode_prefill", time.perf_counter() - rendered_at, tokens)
        return pooled.view(len(pairs), len(self.views), self.hidden_size)

    def encode_pairs(self, pairs: Sequence[ArcPair], puzzle_id: str | None = None) -> Tensor:
        """Return the pair embeddings, each the mean over its views.

        Args:
            pairs: Demonstration pairs, at least one.
            puzzle_id: Task id for the timing records.

        Returns:
            ``e_bar[i]``, fp32 on the CPU, ``[n_pairs, hidden]``.
        """
        return self.encode_views(pairs, puzzle_id).mean(dim=1)

    def task_embedding(self, pairs: Sequence[ArcPair], puzzle_id: str | None = None) -> Tensor:
        """Return the task embedding, the mean of the pair embeddings (order invariant).

        Args:
            pairs: The task's demonstration pairs, at least one.
            puzzle_id: Task id for the timing records.

        Returns:
            ``z_bar``, fp32 on the CPU, ``[hidden]``.
        """
        return self.encode_pairs(pairs, puzzle_id).mean(dim=0)

    def _prefill(self, rendered: Sequence[RenderedPair]) -> Tensor:
        """Pool every rendered sequence; returns fp32 ``[n_sequences, hidden]`` on the CPU."""
        lengths = [len(r.token_ids) for r in rendered]
        order = sorted(range(len(rendered)), key=lambda i: lengths[i], reverse=True)
        device = self.device
        with torch.no_grad():
            out = torch.empty(len(rendered), self.hidden_size, dtype=torch.float32, device=device)
            for batch in _token_batches(order, lengths, self.batch_tokens):
                width = lengths[batch[0]]
                ids = torch.full((len(batch), width), self._pad_id, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):
                    ids[row, : lengths[i]] = torch.as_tensor(rendered[i].token_ids)
                    mask[row, : lengths[i]] = 1
                hidden = self.hidden_states(ids.to(device), mask.to(device))
                for row, i in enumerate(batch):
                    span = self._span(rendered[i])
                    out[i] = hidden[row, span.start : span.stop].mean(dim=0, dtype=torch.float32)
            result = out.cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("a pair embedding is not finite")
        return result

    def _span(self, rendered: RenderedPair) -> range:
        """The pooled positions of one rendered pair under this encoder's ``pool``."""
        return rendered.output_positions if self.pool == "output" else rendered.input_positions

    def _emit(self, puzzle_id: str | None, op: str, seconds: float, tokens: int) -> None:
        """Send one timing record to the callback, if there is one."""
        if self.timing is not None:
            self.timing({"puzzle_id": puzzle_id, "op": op, "seconds": seconds, "tokens": tokens})


def _check_hash(value: str) -> str:
    """Return ``value`` if it is a ``pair_hash`` (64 lowercase hex characters)."""
    if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX):
        raise ValueError(f"not a pair hash: {value!r}")
    return value


def _fsync_dir(path: Path) -> None:
    """Make a rename inside ``path`` durable (POSIX); a no-op where directories cannot open."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so that readers see the old file or the new one, never part.

    The bytes go to a temporary file in the same directory, are flushed to disk, and replace
    ``path`` with ``os.replace``, which is atomic on POSIX file systems.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)


def _json_bytes(value: Any) -> bytes:
    """Deterministic compact JSON, the form of every file the cache writes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class EmbeddingCache:
    """Content-addressed store of pair embeddings, one namespace per encoder configuration.

    The key of an entry is (``pair_hash``, model id, layer, pool, view-set id). The last four form
    the namespace: a directory under ``root`` named ``<pool>-L<layer>-<digest>``, where the digest
    is the sha256 of the namespace record, which ``namespace.json`` holds. Inside it, every
    ``put`` writes one immutable safetensors shard and then rewrites ``manifest.json``, the index
    from pair hash to shard. Both writes are atomic and the manifest goes last, so a crash leaves
    at most an unreferenced shard: every entry the manifest lists is complete. Tensors are stored
    in fp32, so a round trip is exact.

    Each entry holds ``e_bar[i]`` (tensor ``<hash>``, ``[hidden]``) and optionally the per-view
    ``e[i, a]`` (tensor ``<hash>/views``, ``[n_views, hidden]``). One process writes a namespace
    at a time; concurrent writers are not supported. The manifest is rewritten on every ``put``,
    so callers put many pairs at once.

    Args:
        root: Directory holding the namespaces; created if missing.
        model_id: Identifier of the base weights and their precision, for example the Kaggle
            handle ``sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1``.
        layer: The encoder's ``layer``.
        pool: The encoder's ``pool``.
        view_set_id: The encoder's ``view_set_id``.

    Raises:
        ValueError: If an existing ``namespace.json`` differs from this namespace, or an
            argument is empty or of the wrong kind.
    """

    def __init__(
        self, root: str | Path, *, model_id: str, layer: int, pool: str, view_set_id: str
    ) -> None:
        if not model_id or not view_set_id or pool not in POOLS or layer < 0:
            raise ValueError("model_id and view_set_id must be non-empty, pool valid, layer >= 0")
        self.namespace = {
            "schema": CACHE_SCHEMA,
            "model_id": model_id,
            "layer": layer,
            "pool": pool,
            "view_set_id": view_set_id,
        }
        digest = hashlib.sha256(_json_bytes(self.namespace)).hexdigest()[:16]
        self.path = Path(root) / f"{pool}-L{layer}-{digest}"
        self.path.mkdir(parents=True, exist_ok=True)
        record = self.path / "namespace.json"
        if record.exists():
            stored = json.loads(record.read_text(encoding="utf-8"))
            if stored != self.namespace:
                raise ValueError(f"{record} holds another namespace: {stored}")
        else:
            _atomic_write(record, _json_bytes(self.namespace))
        manifest = self.path / "manifest.json"
        if manifest.exists():
            self._manifest = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            self._manifest = {
                "schema": CACHE_SCHEMA,
                "hidden_size": None,
                "n_views": None,
                "shards": {},
                "entries": {},
            }

    def __len__(self) -> int:
        return len(self._manifest["entries"])

    def __contains__(self, pair_hash: object) -> bool:
        return pair_hash in self._manifest["entries"]

    @property
    def hidden_size(self) -> int | None:
        """Embedding width, fixed by the first ``put``; None while the cache is empty."""
        return self._manifest["hidden_size"]

    @property
    def n_views(self) -> int | None:
        """Views per entry with per-view tensors, fixed by the first ``put`` that stores them."""
        return self._manifest["n_views"]

    @property
    def size_bytes(self) -> int:
        """Bytes of all shards the manifest lists (shard headers included)."""
        return sum(shard["bytes"] for shard in self._manifest["shards"].values())

    def has_views(self, pair_hash: str) -> bool:
        """Whether the entry of ``pair_hash`` stores its per-view embeddings.

        Args:
            pair_hash: A key present in the cache.

        Returns:
            True if ``get_views`` can return it.

        Raises:
            KeyError: If the pair is not cached.
        """
        return self._manifest["entries"][pair_hash]["views"]

    def missing(self, pair_hashes: Iterable[str]) -> list[str]:
        """Return the hashes that have no entry, once each, in order of first appearance.

        Args:
            pair_hashes: Keys to look up.

        Returns:
            The keys not in the cache.
        """
        entries = self._manifest["entries"]
        return [h for h in dict.fromkeys(pair_hashes) if h not in entries]

    def put(self, means: Mapping[str, Tensor], views: Mapping[str, Tensor] | None = None) -> int:
        """Store new entries in one shard; keys already cached are skipped.

        Args:
            means: ``e_bar[i]`` per pair hash, each ``[hidden]``.
            views: ``e[i, a]`` per pair hash, each ``[n_views, hidden]``, for exactly the keys
                of ``means``; None stores the means alone.

        Returns:
            The number of entries written.

        Raises:
            ValueError: If a key is not a pair hash, the keys of ``views`` differ from those of
                ``means``, a shape disagrees with the cache, or a value is not finite.
        """
        if views is not None and set(views) != set(means):
            raise ValueError("views must have exactly the keys of means")
        new = [_check_hash(h) for h in means if h not in self._manifest["entries"]]
        if not new:
            return 0
        hidden = self.hidden_size
        n_views = self.n_views
        tensors: dict[str, Tensor] = {}
        for h in new:
            mean = means[h].detach().to("cpu", torch.float32).contiguous()
            if mean.dim() != 1 or mean.numel() == 0:
                raise ValueError(f"mean of {h} must be 1-D and non-empty")
            if hidden is not None and mean.shape[0] != hidden:
                raise ValueError(f"mean of {h} has width {mean.shape[0]}, the cache {hidden}")
            hidden = mean.shape[0]
            tensors[h] = mean
            if views is None:
                continue
            per_view = views[h].detach().to("cpu", torch.float32).contiguous()
            if per_view.dim() != 2 or per_view.shape[0] == 0 or per_view.shape[1] != hidden:
                raise ValueError(f"views of {h} must be [n_views, {hidden}]")
            if n_views is not None and per_view.shape[0] != n_views:
                raise ValueError(f"views of {h}: {per_view.shape[0]} views, the cache {n_views}")
            n_views = per_view.shape[0]
            tensors[f"{h}/views"] = per_view
        if not all(torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("refusing to cache a non-finite embedding")
        name = f"shard-{len(self._manifest['shards']):06d}.safetensors"
        metadata = {"namespace": _json_bytes(self.namespace).decode("utf-8")}
        data = safetensors_bytes(tensors, metadata=metadata)
        _atomic_write(self.path / name, data)
        manifest = {
            **self._manifest,
            "hidden_size": hidden,
            "n_views": n_views,
            "shards": {**self._manifest["shards"], name: {"bytes": len(data), "entries": len(new)}},
            "entries": {
                **self._manifest["entries"],
                **{h: {"shard": name, "views": views is not None} for h in new},
            },
        }
        _atomic_write(self.path / "manifest.json", _json_bytes(manifest))
        self._manifest = manifest
        return len(new)

    def get(self, pair_hashes: Sequence[str]) -> Tensor:
        """Load the pair embeddings ``e_bar[i]``.

        Args:
            pair_hashes: Cached keys, in the order wanted.

        Returns:
            fp32, ``[n, hidden]``, bit-identical to what ``put`` stored.

        Raises:
            KeyError: If a key is not cached.
        """
        return self._load(pair_hashes, suffix="", shape=(self.hidden_size or 0,))

    def get_views(self, pair_hashes: Sequence[str]) -> Tensor:
        """Load the per-view embeddings ``e[i, a]``.

        Args:
            pair_hashes: Cached keys stored with views, in the order wanted.

        Returns:
            fp32, ``[n, n_views, hidden]``.

        Raises:
            KeyError: If a key is not cached or was stored without views.
        """
        bare = [h for h in pair_hashes if h in self and not self.has_views(h)]
        if bare:
            raise KeyError(f"stored without views: {bare[:3]}")
        return self._load(
            pair_hashes, suffix="/views", shape=(self.n_views or 0, self.hidden_size or 0)
        )

    def _load(self, pair_hashes: Sequence[str], suffix: str, shape: tuple[int, ...]) -> Tensor:
        """Read the tensors ``<hash><suffix>``, one open per shard; ``[n, *shape]``."""
        entries = self._manifest["entries"]
        absent = [h for h in pair_hashes if h not in entries]
        if absent:
            raise KeyError(f"{len(absent)} pairs are not cached, for example {absent[:3]}")
        out = torch.empty((len(pair_hashes), *shape), dtype=torch.float32)
        by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, h in enumerate(pair_hashes):
            by_shard[entries[h]["shard"]].append((i, h))
        for shard, items in by_shard.items():
            with safe_open(str(self.path / shard), framework="pt") as f:
                for i, h in items:
                    out[i] = f.get_tensor(h + suffix)
        return out
