r"""Meta-training episodes and the leakage gates (task P3.4; BRIEF B2 Q6, B4, B5 P3.4, B8).

An episode is one task's demonstrations $D$ and one left-out query pair $Q$, with one augmentation
applied to every pair (Q6). This module holds what the builder and checker notebooks share
(``scripts/build_episodes.py``): grid validation, the token count of the notebook format, the
augmentation table, the source loaders, episode sampling, the record format, and the leakage
gates. Method and sources: ``docs/research/meta_training_sources.md`` (sections 2.3 and 4) and
``docs/modules/episodes.md``.

Numpy and the standard library only, so it runs in a Kaggle CPU notebook without an install.

- Grids become ``uint8`` arrays ``[rows, cols]`` through ``validate_grid``.
- Augmentation id $8 j + 4 t + k$: transpose $t$, then $k$ quarter turns (``np.rot90``), then
  color permutation $j$ of a ``PermutationTable``, the order of the notebook's
  ``ArcDataset.augment``. ``PermutationTable.descriptor`` spells an id as the key suffix that the
  notebook's ``ArcDataset.forward_mod`` applies, so meta-training, TTT, and decode share one
  augmentation.
- Hashes: $H_e$ and $K_e$ are sha256 of a pair's or grid's compact JSON ($H_e$ equals P3.2's
  ``pair_hash``); $H_c$ and $K_c$ take the minimum over $D_4$ of the serialization after relabeling
  colors in order of first appearance, jointly over a pair's two grids, so they are invariant
  under $D_4 \times S_{10}$ and separate everything else.
- Randomness: each choice draws from a ``random.Random`` seeded with sha256 of the run seed and the
  choice's name (``rng_for``), so results do not depend on iteration order, Python's salted
  ``hash``, or numpy's generator versions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

Grid = list[list[int]]
"""BRIEF B4: rows of cells, each cell a color 0 to 9."""

ArrayPair = tuple[np.ndarray, np.ndarray]
"""An (input, output) pair of validated grids, each ``uint8`` ``[rows, cols]``."""

MAX_SIDE = 30
N_COLORS = 10
# <|im_start|> user \n <|im_end|> on the input side, the same with "assistant" on the output side
# (QwenFormatter.fmt_train); the grids add rows * cols digits and rows - 1 newlines each.
PAIR_OVERHEAD_TOKENS = 8
SPLITS = ("train", "val", "toy")
SOURCES = ("arc2train", "rearc", "barc")
GATES = ("G1", "G2", "G3", "G4", "G5")
# (transpose, quarter turns) in augmentation-id order 4 t + k: the 8 elements of D4.
DIHEDRAL: tuple[tuple[bool, int], ...] = tuple((t, k) for t in (False, True) for k in range(4))
RECORD_KEYS = (
    "episode_id",
    "source",
    "task_id",
    "family",
    "split",
    "augmentation_id",
    "tokens",
    "n_demos",
    "demos",
    "queries",
)
# Cell-count buckets of report R1, edged at the minimum sizes of the note's false-positive table.
R1_BUCKETS = ((1, 8), (9, 24), (25, 63), (64, 99), (100, 224), (225, MAX_SIDE * MAX_SIDE))
_REARC_MEMBER = re.compile(r"re_arc/tasks/([0-9a-f]{8})\.json")
_ARC_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")
_COLORS = np.arange(N_COLORS, dtype=np.uint8)
# Drop-list and hit ids of provenance ids (a BARC seed): seed:<arc id>.
SEED_PREFIX = "seed:"
_IDENTITY = tuple(range(N_COLORS))


class GridError(ValueError):
    """A grid is not a valid ARC grid. The message never quotes cells."""


class RecordError(ValueError):
    """An episode record breaks the schema that ``validate_record`` checks."""


@dataclass
class Pair:
    """A demonstration pair (BRIEF B4 schema, unchanged; P3.4 builds real pairs only).

    Attributes:
        input: Input grid.
        output: Output grid.
        kind: ``real``, ``held_out``, or ``pseudo``.
        round_added: Refinement round that added the pair; 0 for real pairs.
        agreement: Augmentation agreement of a pseudo pair; None otherwise.
    """

    input: Grid
    output: Grid
    kind: Literal["real", "held_out", "pseudo"]
    round_added: int
    agreement: float | None


@dataclass
class Episode:
    """A meta-training episode (BRIEF B4 schema, unchanged).

    Attributes:
        demos: The demonstrations $D$, un-augmented.
        queries: The query pairs $Q$, un-augmented; P3.4 builds exactly one.
        augmentation_id: $8 j + 4 t + k$ over the run's ``PermutationTable``; the loader applies it
            to every pair of $D$ and $Q$ alike.
    """

    demos: list[Pair]
    queries: list[Pair]
    augmentation_id: int


def validate_grid(grid: Any) -> np.ndarray:
    """Return a grid as a ``uint8`` array after checking that it is a valid ARC grid.

    Why: every source is third-party data (BARC's parquet view stores floats, ReARC stores sides up
    to 39), so each grid is checked once, on load, and later steps can trust the arrays.

    Args:
        grid: Rows of cells: nested lists or tuples, or a 2-D array.

    Returns:
        The grid, C-contiguous ``uint8`` ``[rows, cols]``; the input itself if it already is one.

    Raises:
        GridError: If the grid is not 2-D and rectangular, a side is outside 1 to 30, or a cell is
            not an integer 0 to 9 (floats pass only when integral; booleans and strings never).
    """
    if isinstance(grid, np.ndarray):
        arr = grid
    else:
        try:
            arr = np.asarray(grid)
        except (ValueError, TypeError, OverflowError) as err:
            raise GridError("grid is not a rectangular 2-D array") from err
    if arr.ndim != 2:
        raise GridError(f"grid must be 2-D, got {arr.ndim} dimensions")
    rows, cols = arr.shape
    if not (1 <= rows <= MAX_SIDE and 1 <= cols <= MAX_SIDE):
        raise GridError(f"grid sides must be 1 to {MAX_SIDE}, got {rows} by {cols}")
    if arr.dtype.kind == "f":
        if not (np.isfinite(arr).all() and (arr == np.floor(arr)).all()):
            raise GridError("float cells must be integral")
    elif arr.dtype.kind not in "iu":
        raise GridError(f"cells must be integers, got dtype {arr.dtype}")
    if arr.min() < 0 or arr.max() >= N_COLORS:
        raise GridError(f"cells must be colors 0 to {N_COLORS - 1}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compact(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("ascii")


def exact_grid_hash(grid: Any) -> str:
    """Return $K_e$, the sha256 of a grid's compact JSON, for example ``[[0,1],[2,3]]``.

    Args:
        grid: A valid grid (``validate_grid``).

    Returns:
        64 lowercase hex characters.

    Raises:
        GridError: If the grid is invalid.
    """
    return _sha256(_compact(validate_grid(grid).tolist()))


def exact_pair_hash(x: Any, y: Any) -> str:
    """Return $H_e$, the sha256 of the pair's compact JSON ``{"input":...,"output":...}``.

    Why: the bytes equal those of P3.2's ``arc26.hyper.encoder.pair_hash``, so the leakage report
    and the embedding cache name a pair alike. The JSON nesting encodes both shapes.

    Args:
        x: Input grid.
        y: Output grid.

    Returns:
        64 lowercase hex characters.

    Raises:
        GridError: If a grid is invalid.
    """
    return _sha256(
        _compact({"input": validate_grid(x).tolist(), "output": validate_grid(y).tolist()})
    )


def _dihedral(grid: np.ndarray, transpose: bool, quarter_turns: int) -> np.ndarray:
    return np.rot90(np.swapaxes(grid, 0, 1) if transpose else grid, quarter_turns)


def _first_appearance_lut(cells: np.ndarray) -> np.ndarray:
    # lut[c] is the rank of color c's first appearance in the scan; absent colors rank last.
    hits = cells[:, None] == _COLORS[None, :]
    first = np.where(hits.any(axis=0), hits.argmax(axis=0), cells.size)
    lut = np.empty(N_COLORS, dtype=np.uint8)
    lut[np.argsort(first, kind="stable")] = _COLORS
    return lut


def _canonical_key(grids: Sequence[np.ndarray]) -> bytes:
    best = b""
    for transpose, turns in DIHEDRAL:
        moved = [_dihedral(grid, transpose, turns) for grid in grids]
        # One joint relabeling over all grids: per-grid relabeling would merge pairs whose
        # input and output colors correspond differently.
        cells = np.concatenate([grid.ravel() for grid in moved])
        key = bytes(side for grid in moved for side in grid.shape)
        key += _first_appearance_lut(cells)[cells].tobytes()
        if not best or key < best:
            best = key
    return best


def canonical_grid_hash(grid: Any) -> str:
    r"""Return $K_c$, a grid hash invariant under the 8 dihedral maps and all color permutations.

    The serialization is the shape followed by the cells after relabeling colors 0, 1, ... in order
    of first appearance in the row-major scan; $K_c$ is the sha256 of its minimum over $D_4$.

    Args:
        grid: A valid grid.

    Returns:
        64 lowercase hex characters; equal for two grids exactly when one is a $D_4 \times S_{10}$
        image of the other.

    Raises:
        GridError: If the grid is invalid.
    """
    return _sha256(b"grid" + _canonical_key([validate_grid(grid)]))


def canonical_pair_hash(x: Any, y: Any) -> str:
    r"""Return $H_c(x, y) = \mathrm{sha256}(\min_{g \in D_4} \mathrm{ser}(\rho(g x, g y)))$.

    $\rho$ relabels colors in order of first appearance over the row-major scan of $x$ followed by
    $y$, and ``ser`` writes both shapes and the relabeled cells. $\rho(\pi x, \pi y) = \rho(x, y)$
    for every color permutation $\pi$ and $D_4$ commutes with $S_{10}$, so $H_c(p) = H_c(p')$
    exactly when $p' = (g, \pi) \cdot p$ (note, section 2.3). It certifies all $8 \cdot 10!$
    augmentations of a pair at once.

    Args:
        x: Input grid.
        y: Output grid.

    Returns:
        64 lowercase hex characters.

    Raises:
        GridError: If a grid is invalid.
    """
    return _sha256(b"pair" + _canonical_key([validate_grid(x), validate_grid(y)]))


def is_nontrivial(grid: Any, min_cells: int = 25, min_colors: int = 2) -> bool:
    """Return whether a grid is large and varied enough for the grid gates G4 and G5.

    Why: small or single-color grids match unrelated tasks by chance (note, section 2.3: 38 exact
    false positives at 9 cells, 1 at 25), so only grids above both floors gate.

    Args:
        grid: A valid grid.
        min_cells: Smallest cell count that gates.
        min_colors: Smallest number of distinct colors that gates.

    Returns:
        True when the grid has at least ``min_cells`` cells and ``min_colors`` colors.

    Raises:
        GridError: If the grid is invalid.
    """
    arr = validate_grid(grid)
    colors = int(np.count_nonzero(np.bincount(arr.ravel(), minlength=N_COLORS)))
    return arr.size >= min_cells and colors >= min_colors


def grid_tokens(rows: int, cols: int) -> int:
    """Return the tokens of one grid in the notebook format: one per cell, a newline between rows.

    Args:
        rows: Rows of the grid as rendered (after any transform).
        cols: Columns as rendered.

    Returns:
        ``rows * cols + rows - 1``.
    """
    return rows * cols + rows - 1


def pair_tokens(x: Any, y: Any, transpose: bool = False) -> int:
    r"""Return the tokens of one pair in the notebook format.

    ``QwenFormatter.fmt_train`` renders a pair as
    ``<|im_start|>user\n{x}<|im_end|><|im_start|>assistant\n{y}<|im_end|>``, so under NVARC's
    word-level tokenizer $T(x, y) = (h_x w_x + h_x - 1) + (h_y w_y + h_y - 1) + 8$ (note, 4.3).

    Args:
        x: Input grid as an array or rectangular rows.
        y: Output grid as an array or rectangular rows.
        transpose: Whether the augmentation exchanges rows and columns: a transpose or an odd
            number of quarter turns, not both (``Augmentation.swaps_axes``).

    Returns:
        The token count.
    """
    (x_rows, x_cols), (y_rows, y_cols) = np.shape(x), np.shape(y)
    if transpose:
        x_rows, x_cols, y_rows, y_cols = x_cols, x_rows, y_cols, y_rows
    return grid_tokens(x_rows, x_cols) + grid_tokens(y_rows, y_cols) + PAIR_OVERHEAD_TOKENS


def episode_tokens(pairs: Iterable[ArrayPair]) -> int:
    """Return an episode's tokens in its longer orientation.

    Why: the loader may redraw the augmentation id per epoch, and only the exchange of rows and
    columns changes the count, so the larger of the two orientations bounds every redraw (the
    notebook's ``get_length(max_of_transposed=True)`` does the same).

    Args:
        pairs: The demos followed by the queries.

    Returns:
        The larger of the episode's untransposed and transposed token counts.
    """
    pairs = list(pairs)
    return max(sum(pair_tokens(x, y, swap) for x, y in pairs) for swap in (False, True))


@dataclass(frozen=True)
class Augmentation:
    r"""One element of $D_4 \times S_{10}$, applied as ``ArcDataset.forward_mod`` applies it.

    Attributes:
        transpose: Swap rows and columns first.
        quarter_turns: Then ``np.rot90`` this many times, 0 to 3 (counterclockwise).
        permutation: Then map color ``c`` to ``permutation[c]`` (``permute_mod``).
    """

    transpose: bool
    quarter_turns: int
    permutation: tuple[int, ...]

    @property
    def swaps_axes(self) -> bool:
        """Whether the result has the input's columns as rows."""
        return self.transpose != (self.quarter_turns % 2 == 1)

    def apply(self, grid: Any) -> np.ndarray:
        """Return the augmented grid.

        Args:
            grid: A valid grid.

        Returns:
            A new ``uint8`` array, ``[cols, rows]`` when ``swaps_axes`` else ``[rows, cols]``.

        Raises:
            GridError: If the grid is invalid.
        """
        moved = _dihedral(validate_grid(grid), self.transpose, self.quarter_turns)
        return np.asarray(self.permutation, dtype=np.uint8)[moved]


def _check_permutation(perm: Sequence[int]) -> tuple[int, ...]:
    perm = tuple(int(c) for c in perm)
    if sorted(perm) != list(_IDENTITY):
        raise ValueError(f"not a permutation of the {N_COLORS} colors: {perm}")
    return perm


@dataclass(frozen=True)
class PermutationTable:
    """The color permutations behind augmentation ids $8 j + 4 t + k$.

    Entry 0 is the identity; the rest are distinct seeded permutations of all 10 colors,
    background included, as the notebook's ``permute_rnd_all_`` draws them. The table is written
    into the episode manifest, so a loader decodes ids without regenerating it.

    Attributes:
        permutations: The table; ``permutations[j][c]`` is the new color of color ``c``.
    """

    permutations: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        perms = tuple(_check_permutation(p) for p in self.permutations)
        if not perms or perms[0] != _IDENTITY:
            raise ValueError("a permutation table starts with the identity")
        if len(set(perms)) != len(perms):
            raise ValueError("permutation table entries must be distinct")
        object.__setattr__(self, "permutations", perms)

    @classmethod
    def seeded(cls, size: int, seed: int) -> PermutationTable:
        """Draw a table of ``size`` distinct permutations, the identity first.

        Args:
            size: Number of permutations, at least 1.
            seed: Seed of ``random.Random``.

        Returns:
            The table.

        Raises:
            ValueError: If ``size`` is below 1 or above the number of permutations.
        """
        if not 1 <= size <= 3628800:
            raise ValueError(f"table size must be 1 to 10!, got {size}")
        rng = random.Random(seed)
        perms = [_IDENTITY]
        seen = {_IDENTITY}
        while len(perms) < size:
            perm = list(_IDENTITY)
            rng.shuffle(perm)
            if tuple(perm) not in seen:
                seen.add(tuple(perm))
                perms.append(tuple(perm))
        return cls(tuple(perms))

    @classmethod
    def from_strings(cls, items: Sequence[str]) -> PermutationTable:
        """Read a table written by ``to_strings``, for example from ``MANIFEST.json``.

        Args:
            items: One 10-digit string per permutation.

        Returns:
            The table.

        Raises:
            ValueError: If an item is not a permutation of the digits 0 to 9, or the table is not
                valid (``PermutationTable``).
        """
        perms = []
        for item in items:
            if not isinstance(item, str) or not re.fullmatch(r"[0-9]{10}", item):
                raise ValueError(f"a permutation is 10 digits, got {item!r}")
            perms.append(tuple(int(c) for c in item))
        return cls(tuple(perms))

    def to_strings(self) -> list[str]:
        """Return the table as 10-digit strings, the spelling ``permute_mod`` reads."""
        return ["".join(map(str, perm)) for perm in self.permutations]

    @property
    def n_augmentations(self) -> int:
        """Number of valid augmentation ids: 8 dihedral maps per permutation."""
        return len(DIHEDRAL) * len(self.permutations)

    def augmentation(self, aug_id: int) -> Augmentation:
        """Decode an augmentation id $8 j + 4 t + k$.

        Args:
            aug_id: An id from 0 to ``n_augmentations - 1``.

        Returns:
            The augmentation.

        Raises:
            ValueError: If the id is out of range.
        """
        if isinstance(aug_id, bool) or not 0 <= aug_id < self.n_augmentations:
            raise ValueError(f"augmentation id must be 0 to {self.n_augmentations - 1}")
        j, dihedral = divmod(aug_id, len(DIHEDRAL))
        transpose, turns = DIHEDRAL[dihedral]
        return Augmentation(transpose, turns, self.permutations[j])

    def descriptor(self, aug_id: int) -> str:
        """Spell an id as the notebook's key suffix, e.g. ``transpose.rot90.permute3810245679``.

        ``ArcDataset.forward_mod(grid, f"{task_id}.{descriptor}")`` equals
        ``augmentation(aug_id).apply(grid)``; the ops come in ``ArcDataset.augment``'s order.

        Args:
            aug_id: An id from 0 to ``n_augmentations - 1``.

        Returns:
            The ops joined by ``.``: ``transpose`` if $t = 1$, ``rot90`` $k$ times, then
            ``permute`` with the 10 digits of permutation $j$.

        Raises:
            ValueError: If the id is out of range.
        """
        aug = self.augmentation(aug_id)
        ops = ["transpose"] * aug.transpose + ["rot90"] * aug.quarter_turns
        return ".".join([*ops, "permute" + "".join(map(str, aug.permutation))])


def augmentation_id(transpose: bool, quarter_turns: int, permutation_index: int) -> int:
    """Return the id $8 j + 4 t + k$ of an augmentation.

    Args:
        transpose: $t$.
        quarter_turns: $k$, 0 to 3.
        permutation_index: $j$, the permutation's index in the table.

    Returns:
        The id.

    Raises:
        ValueError: If ``quarter_turns`` is outside 0 to 3 or ``permutation_index`` is negative.
    """
    if not 0 <= quarter_turns <= 3 or permutation_index < 0:
        raise ValueError("quarter turns must be 0 to 3 and the permutation index at least 0")
    return len(DIHEDRAL) * permutation_index + 4 * int(transpose) + quarter_turns


def derive_seed(*parts: object) -> int:
    """Return a 64-bit seed from sha256 of the parts joined with a unit separator.

    Why: Python's ``hash`` of a string changes per process, and seeding each choice by its name
    keeps one task's episodes the same when other tasks are added or dropped.

    Args:
        *parts: The run seed and the names of the choice.

    Returns:
        An integer in ``[0, 2**64)``.
    """
    data = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def rng_for(*parts: object) -> random.Random:
    """Return a ``random.Random`` seeded with ``derive_seed(*parts)``.

    Args:
        *parts: The run seed and the names of the choice.

    Returns:
        A fresh generator.
    """
    return random.Random(derive_seed(*parts))


def keyed_order(items: Iterable[str], *salt: object) -> list[str]:
    """Order distinct items by sha256 of the salt and the item: a seeded shuffle.

    Why: selecting the first $n$ in this order is reproducible and independent of the input order,
    and dropping one item leaves the others' order unchanged.

    Args:
        items: Distinct strings; duplicates are merged.
        *salt: The run seed and the name of the selection.

    Returns:
        The items in keyed order.
    """
    prefix = "\x1f".join(str(part) for part in salt)

    def key(item: str) -> bytes:
        return hashlib.sha256(f"{prefix}\x1f{item}".encode()).digest()

    return sorted(set(items), key=key)


def allocate(task_ids: Iterable[str], total: int, *salt: object) -> dict[str, int]:
    """Spread ``total`` episodes over tasks as evenly as possible.

    Args:
        task_ids: Distinct task ids.
        total: Episodes to allocate, at least 0.
        *salt: The run seed and the name of the allocation; the first ``total mod n`` tasks in
            ``keyed_order`` get one more.

    Returns:
        Episodes per task id, summing to ``total`` (empty when there are no tasks).
    """
    ids = keyed_order(task_ids, *salt)
    if not ids:
        return {}
    base, extra = divmod(total, len(ids))
    return {task_id: base + (rank < extra) for rank, task_id in enumerate(ids)}


class ExampleStream:
    """Hands out a pool's pair indices in a seeded order, each index at most once.

    Why: a ReARC generator's episodes use disjoint examples (note, 4.4), and its toy or validation
    puzzles draw from the same stream after them.

    Args:
        size: Number of pairs in the pool; the indices are 0 to ``size - 1``.
        rng: Generator of the order, consumed once by a shuffle.
    """

    def __init__(self, size: int, rng: random.Random) -> None:
        self._order = list(range(size))
        rng.shuffle(self._order)
        self._next = 0

    @property
    def remaining(self) -> int:
        """Indices not yet handed out."""
        return len(self._order) - self._next

    def take(self, count: int) -> list[int]:
        """Hand out the next ``count`` indices.

        Args:
            count: Indices wanted, at most ``remaining``.

        Returns:
            The indices, in stream order.

        Raises:
            ValueError: If fewer than ``count`` remain.
        """
        if not 0 <= count <= self.remaining:
            raise ValueError(f"asked for {count} examples, {self.remaining} remain")
        chunk = self._order[self._next : self._next + count]
        self._next += count
        return chunk


@dataclass
class SourceTask:
    """One task of a source: its valid pairs plus provenance.

    Attributes:
        source: ``arc2train``, ``rearc``, or ``barc``.
        task_id: The ARC id, the ReARC generator id, or ``barc:<set>:<sha1(source)[:16]>``.
        family: The split unit (note, 4.5): a ReARC generator together with the ARC task of the
            same id, or one BARC problem.
        pairs: Valid (input, output) pairs in source order; for ARC files train then test.
        examples: Pairs in the source before validation.
        invalid_pairs: Pairs dropped by ``validate_grid`` (or malformed).
        seeds: ARC ids named in a BARC problem's ``seeds`` field.
    """

    source: str
    task_id: str
    family: str
    pairs: list[ArrayPair]
    examples: int = 0
    invalid_pairs: int = 0
    seeds: tuple[str, ...] = ()


def _arc_pair(item: Any) -> tuple[Any, Any]:
    return item["input"], item["output"]


def _list_pair(item: Any) -> tuple[Any, Any]:
    x, y = item
    return x, y


def _valid_pairs(
    items: Iterable[Any], split: Callable[[Any], tuple[Any, Any]], limit: int | None = None
) -> tuple[list[ArrayPair], int]:
    pairs: list[ArrayPair] = []
    invalid = 0
    for item in items:
        if limit is not None and len(pairs) >= limit:
            break
        try:
            x, y = split(item)
            pairs.append((validate_grid(x), validate_grid(y)))
        except (GridError, KeyError, TypeError, ValueError):
            invalid += 1
    return pairs, invalid


def load_arc_dir(path: Path, source: str = "arc2train") -> dict[str, SourceTask]:
    """Load every ``<task_id>.json`` of an ARC directory such as ARC-AGI-2 ``data/training``.

    Args:
        path: The directory; subdirectories are not read.
        source: Source name for the tasks.

    Returns:
        Tasks by id; each task's pairs are its train pairs then its test pairs (training sets
        carry test outputs), and its family is its id.

    Raises:
        ValueError: If a file is not an ARC task (``train`` and ``test`` lists).
    """
    tasks: dict[str, SourceTask] = {}
    for file in sorted(Path(path).glob("*.json")):
        try:
            task = json.loads(file.read_text(encoding="utf-8"))
            items = [*task["train"], *task["test"]]
        except (OSError, ValueError, KeyError, TypeError) as err:
            raise ValueError(f"{file.name} is not an ARC task file ({type(err).__name__})") from err
        pairs, invalid = _valid_pairs(items, _arc_pair)
        tasks[file.stem] = SourceTask(source, file.stem, file.stem, pairs, len(items), invalid)
    return tasks


def load_rearc_zip(path: Path) -> dict[str, SourceTask]:
    """Load ReARC's ``re_arc.zip``: one pool of examples per generator.

    Only ``re_arc/tasks/<id>.json`` members are read (a list of ``{"input", "output"}``);
    ``re_arc/metadata.json`` and the ``__MACOSX/`` resource forks are skipped. Invalid examples,
    such as the 129 of ``e26a3af2`` with a side above 30, are dropped and counted.

    Args:
        path: The zip file.

    Returns:
        Generators by id, each a task whose family is its id.

    Raises:
        ValueError: If a task member is not a JSON list.
    """
    tasks: dict[str, SourceTask] = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            match = _REARC_MEMBER.fullmatch(name)
            if match is None:
                continue
            try:
                examples = json.loads(archive.read(name))
            except ValueError as err:
                raise ValueError(f"{name} is not JSON") from err
            if not isinstance(examples, list):
                raise ValueError(f"{name} must hold a list of examples")
            pairs, invalid = _valid_pairs(examples, _arc_pair)
            gen = match.group(1)
            tasks[gen] = SourceTask("rearc", gen, gen, pairs, len(examples), invalid)
    return tasks


def barc_task_id(name: str, source_code: str) -> str:
    """Return a BARC problem's id: ``barc:<name>:<sha1(source)[:16]>`` (BARC rows have no id).

    Args:
        name: Short name of the BARC set, e.g. ``gpt4desc``.
        source_code: The problem's ``source`` field, the generating program.

    Returns:
        The id.
    """
    digest = hashlib.sha1(source_code.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"barc:{name}:{digest[:16]}"


def iter_barc_jsonl(
    path: Path, name: str, *, max_pairs: int | None = None, malformed: Counter | None = None
) -> Iterator[SourceTask]:
    """Stream the problems of a BARC JSONL file, one line at a time.

    Each line is ``{"source": <program>, "examples": [[input, output], ...], "seeds": [...]}``.
    Streaming keeps memory to the problems the caller keeps, not the 2.5 GB file.

    Args:
        path: The JSONL file.
        name: Short name of the set, part of every id.
        max_pairs: Keep at most this many valid pairs per problem (the first in file order);
            None keeps all.
        malformed: Counter that receives ``lines`` for each line that is not a BARC problem.

    Yields:
        One task per problem in file order; its family is its id, ``seeds`` the 8-hex ARC ids
        named in its ``seeds`` field.
    """
    with open(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                code, examples = row["source"], row["examples"]
                if not isinstance(code, str) or not isinstance(examples, list):
                    raise TypeError("source must be a string and examples a list")
            except (ValueError, KeyError, TypeError):
                if malformed is not None:
                    malformed["lines"] += 1
                continue
            seeds = row.get("seeds") or []
            seed_ids = sorted({m for s in seeds if isinstance(s, str) for m in _ARC_ID.findall(s)})
            task_id = barc_task_id(name, code)
            pairs, invalid = _valid_pairs(examples, _list_pair, max_pairs)
            yield SourceTask(
                "barc", task_id, task_id, pairs, len(examples), invalid, tuple(seed_ids)
            )


def split_families(families: Iterable[str], held_out: Iterable[str]) -> dict[str, str]:
    """Assign every family to ``train`` or ``val``.

    Why: splitting by family rather than task id keeps a ReARC generator and the ARC task it was
    written for on the same side, so no generator leaks across the split (note, 4.5).

    Args:
        families: All families of a source.
        held_out: The families that go to ``val``.

    Returns:
        Split by family.

    Raises:
        ValueError: If a held-out family is not among ``families``.
    """
    families = set(families)
    held = set(held_out)
    unknown = sorted(held - families)
    if unknown:
        raise ValueError(f"held-out families are not among the families: {unknown}")
    return {family: "val" if family in held else "train" for family in sorted(families)}


def draw_demo_count(rng: random.Random, probs: Sequence[float], available: int) -> int:
    r"""Draw $|D|$ from $P(|D| = 1, \dots)$, truncated to the demos available and renormalized.

    Args:
        rng: The episode's generator.
        probs: $P(|D| = d)$ for $d = 1, \dots, $ ``len(probs)``; all positive.
        available: Pairs available as demos.

    Returns:
        A demo count from 1 to ``min(len(probs), available)``.

    Raises:
        ValueError: If no demo is available.
    """
    top = min(len(probs), available)
    if top < 1:
        raise ValueError("an episode needs at least one demo")
    return rng.choices(range(1, top + 1), weights=list(probs[:top]))[0]


class Sampled(NamedTuple):
    """Result of ``sample_episodes``.

    Attributes:
        records: Episodes that fit the token limit, in sampling order.
        over_length: Episodes dropped because their tokens exceed the limit.
        short: Episodes not formed because the stream ran out or the task has under 2 pairs.
    """

    records: list[EpisodeRecord]
    over_length: int
    short: int


def sample_episodes(
    task: SourceTask,
    n_episodes: int,
    *,
    split: str,
    rng: random.Random,
    demo_probs: Sequence[float],
    n_augmentations: int,
    max_tokens: int,
    stream: ExampleStream | None = None,
) -> Sampled:
    r"""Sample episodes of one task: $|D|$ demos and one query, one augmentation for all (Q6).

    Cycle mode (``stream`` None, ARC tasks and puzzles): episode $e$ queries pair
    ``order[e mod n]`` of a seeded order of the task's $n$ pairs, so every pair is a query once
    before any repeats (leave-one-out), and draws its demos from the other pairs. Fresh mode
    (ReARC and BARC pools): each episode takes $|D| + 1$ unused pairs from ``stream``, the last
    as the query. Either way $|D|$ comes from ``draw_demo_count``, the demo order is random, and
    the augmentation id is uniform over the table. Episodes above ``max_tokens``
    (``episode_tokens``) are dropped and counted (B5 P3.4); their pairs stay consumed.

    Args:
        task: The task; fresh mode indexes its pairs through ``stream``.
        n_episodes: Episodes to form.
        split: ``train``, ``val``, or ``toy``; part of each episode id.
        rng: The task's generator for this split.
        demo_probs: $P(|D| = d)$ for $d = 1, \dots$.
        n_augmentations: Ids are drawn from 0 to ``n_augmentations - 1``.
        max_tokens: Longest episode kept.
        stream: The pool's stream for fresh mode, or None for cycle mode.

    Returns:
        The kept episodes with ids ``<source>/<task_id>/<split>/<e>``, and the drop counts.
    """
    pairs = task.pairs
    records: list[EpisodeRecord] = []
    over_length = 0
    if stream is None:
        if len(pairs) < 2:
            return Sampled([], 0, n_episodes)
        order = list(range(len(pairs)))
        rng.shuffle(order)
    for e in range(n_episodes):
        if stream is None:
            query = order[e % len(pairs)]
            others = [i for i in range(len(pairs)) if i != query]
            demos = rng.sample(others, draw_demo_count(rng, demo_probs, len(others)))
        else:
            if stream.remaining < 2:
                return Sampled(records, over_length, n_episodes - e)
            chosen = stream.take(draw_demo_count(rng, demo_probs, stream.remaining - 1) + 1)
            demos, query = chosen[:-1], chosen[-1]
        record = EpisodeRecord.create(
            episode_id=f"{task.source}/{task.task_id}/{split}/{e}",
            source=task.source,
            task_id=task.task_id,
            family=task.family,
            split=split,
            augmentation_id=rng.randrange(n_augmentations),
            demos=[pairs[i] for i in demos],
            queries=[pairs[query]],
        )
        if record.tokens > max_tokens:
            over_length += 1
        else:
            records.append(record)
    return Sampled(records, over_length, 0)


@dataclass
class Puzzle:
    """A validation puzzle in the competition's shape: demonstrations and test pairs.

    Attributes:
        puzzle_id: Key in ``val_challenges.json``; letters, digits, and ``-`` only, because the
            notebook splits keys on ``_`` and ``.``.
        source: Source of the pairs.
        task_id: Source task.
        family: Source family (held out of training).
        train: Demonstration pairs.
        test: Test pairs; their outputs go to ``val_solutions.json``.
    """

    puzzle_id: str
    source: str
    task_id: str
    family: str
    train: list[ArrayPair]
    test: list[ArrayPair]

    def episodes(self, rng: random.Random, n_augmentations: int) -> list[EpisodeRecord]:
        """Return the validation episodes: each test pair as the query with all demonstrations.

        Args:
            rng: Generator of the augmentation ids.
            n_augmentations: Ids are drawn from 0 to ``n_augmentations - 1``.

        Returns:
            One ``val`` record per test pair, ids ``<source>/<puzzle_id>/val/<i>``.
        """
        return [
            EpisodeRecord.create(
                episode_id=f"{self.source}/{self.puzzle_id}/val/{i}",
                source=self.source,
                task_id=self.task_id,
                family=self.family,
                split="val",
                augmentation_id=rng.randrange(n_augmentations),
                demos=self.train,
                queries=[pair],
            )
            for i, pair in enumerate(self.test)
        ]

    def challenge(self) -> dict[str, Any]:
        """Return the puzzle as a challenges-file entry (test outputs withheld)."""
        return {
            "train": [_pair_json(pair) for pair in self.train],
            "test": [{"input": x.tolist()} for x, _ in self.test],
        }

    def solution(self) -> list[Grid]:
        """Return the test outputs as a solutions-file entry."""
        return [y.tolist() for _, y in self.test]


def _pair_json(pair: ArrayPair) -> dict[str, Grid]:
    return {"input": pair[0].tolist(), "output": pair[1].tolist()}


@dataclass(frozen=True, eq=False)
class EpisodeRecord:
    """An episode with the provenance B4's ``Episode`` lacks (note, 1.2); one JSONL line.

    Attributes:
        episode_id: ``<source>/<task_id or puzzle_id>/<split>/<index>``, unique in a build.
        source: ``arc2train``, ``rearc``, or ``barc``.
        task_id: Source task (``SourceTask.task_id``).
        family: Split unit (``SourceTask.family``).
        split: ``train``, ``val``, or ``toy``.
        augmentation_id: $8 j + 4 t + k$ over the build's ``PermutationTable``.
        tokens: ``episode_tokens`` of demos then queries.
        n_demos: Number of demos.
        demos: Un-augmented demonstration pairs.
        queries: Un-augmented query pairs (one in every P3.4 episode).
    """

    episode_id: str
    source: str
    task_id: str
    family: str
    split: str
    augmentation_id: int
    tokens: int
    n_demos: int
    demos: tuple[ArrayPair, ...]
    queries: tuple[ArrayPair, ...]

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        source: str,
        task_id: str,
        family: str,
        split: str,
        augmentation_id: int,
        demos: Iterable[ArrayPair],
        queries: Iterable[ArrayPair],
    ) -> EpisodeRecord:
        """Build a record, computing ``tokens`` and ``n_demos`` from the pairs.

        Args:
            episode_id: See the class attributes.
            source: See the class attributes.
            task_id: See the class attributes.
            family: See the class attributes.
            split: See the class attributes.
            augmentation_id: See the class attributes.
            demos: Validated demonstration pairs.
            queries: Validated query pairs.

        Returns:
            The record.
        """
        demos, queries = tuple(demos), tuple(queries)
        tokens = episode_tokens(demos + queries)
        return cls(
            episode_id, source, task_id, family, split, augmentation_id, tokens, len(demos),
            demos, queries,
        )  # fmt: skip

    @property
    def pairs(self) -> tuple[ArrayPair, ...]:
        """Demos followed by queries."""
        return self.demos + self.queries

    def to_json(self) -> dict[str, Any]:
        """Return the record as its JSONL object, keys in ``RECORD_KEYS`` order."""
        return {
            "episode_id": self.episode_id,
            "source": self.source,
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "augmentation_id": self.augmentation_id,
            "tokens": self.tokens,
            "n_demos": self.n_demos,
            "demos": [_pair_json(pair) for pair in self.demos],
            "queries": [_pair_json(pair) for pair in self.queries],
        }

    def to_episode(self) -> Episode:
        """Return the B4 ``Episode``: real pairs, round 0, the record's augmentation id."""
        demos = [Pair(x.tolist(), y.tolist(), "real", 0, None) for x, y in self.demos]
        queries = [Pair(x.tolist(), y.tolist(), "real", 0, None) for x, y in self.queries]
        return Episode(demos, queries, self.augmentation_id)


def _record_pairs(value: Any, what: str) -> tuple[ArrayPair, ...]:
    if not isinstance(value, list) or not value:
        raise RecordError(f"{what} must be a non-empty list of pairs")
    pairs = []
    for i, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"input", "output"}:
            raise RecordError(f"{what}[{i}] must be an object with input and output only")
        try:
            pairs.append((validate_grid(item["input"]), validate_grid(item["output"])))
        except GridError as err:
            raise RecordError(f"{what}[{i}]: {err}") from err
    return tuple(pairs)


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_record(
    obj: Any, *, n_augmentations: int | None = None, max_tokens: int | None = None
) -> EpisodeRecord:
    """Check one JSONL object against the episode schema and return it as a record.

    Checked: exactly the ``RECORD_KEYS``; non-empty string ids; a known source and split; an
    integer augmentation id in range; one or more demos and queries of valid grids; ``n_demos``
    equal to the demo count; ``tokens`` equal to ``episode_tokens`` and within ``max_tokens``.

    Args:
        obj: The parsed JSON object.
        n_augmentations: Valid ids are below this; None skips the upper bound.
        max_tokens: Longest episode allowed; None skips the check.

    Returns:
        The record.

    Raises:
        RecordError: On the first rule the object breaks.
    """
    if not isinstance(obj, dict):
        raise RecordError("a record must be a JSON object")
    if set(obj) != set(RECORD_KEYS):
        raise RecordError(f"record keys must be exactly {list(RECORD_KEYS)}, got {sorted(obj)}")
    for key in ("episode_id", "task_id", "family"):
        if not isinstance(obj[key], str) or not obj[key]:
            raise RecordError(f"{key} must be a non-empty string")
    if obj["source"] not in SOURCES:
        raise RecordError(f"source must be one of {SOURCES}, got {obj['source']!r}")
    if obj["split"] not in SPLITS:
        raise RecordError(f"split must be one of {SPLITS}, got {obj['split']!r}")
    aug = obj["augmentation_id"]
    if not _plain_int(aug) or aug < 0 or (n_augmentations is not None and aug >= n_augmentations):
        raise RecordError(f"augmentation_id must be an integer from 0 below {n_augmentations}")
    demos = _record_pairs(obj["demos"], "demos")
    queries = _record_pairs(obj["queries"], "queries")
    if not _plain_int(obj["n_demos"]) or obj["n_demos"] != len(demos):
        raise RecordError(f"n_demos must equal the {len(demos)} demos")
    tokens = episode_tokens(demos + queries)
    if not _plain_int(obj["tokens"]) or obj["tokens"] != tokens:
        raise RecordError(f"tokens must equal the episode's {tokens} tokens")
    if max_tokens is not None and tokens > max_tokens:
        raise RecordError(f"episode has {tokens} tokens, above the limit {max_tokens}")
    return EpisodeRecord(
        obj["episode_id"], obj["source"], obj["task_id"], obj["family"], obj["split"], aug,
        tokens, len(demos), demos, queries,
    )  # fmt: skip


def shard_name(split: str, index: int, count: int) -> str:
    """Return a shard's path relative to the output root.

    Args:
        split: ``train``, ``val``, or ``toy``.
        index: Shard index from 0.
        count: Number of shards of the split.

    Returns:
        ``episodes/<split>-<index:05d>-of-<count:05d>.jsonl``.
    """
    return f"episodes/{split}-{index:05d}-of-{count:05d}.jsonl"


def write_shards(
    records: Sequence[EpisodeRecord], root: Path, split: str, count: int
) -> list[dict[str, Any]]:
    """Write a split's records as ``count`` JSONL shards of contiguous runs, in order.

    Why: a run of $N$ steps consumes the first $N$ training episodes, so shard order is pool
    order; each line is compact JSON with keys in ``RECORD_KEYS`` order, so equal records give
    equal bytes.

    Args:
        records: The split's records in pool order.
        root: Output root; shards go to ``root/episodes/``.
        split: Split name, part of the file names.
        count: Number of shards, at least 1 (some may be empty when records are few).

    Returns:
        Per shard, in order: ``path`` (relative to ``root``), ``split``, ``records``, ``bytes``,
        ``sha256``.

    Raises:
        ValueError: If ``count`` is below 1.
    """
    if count < 1:
        raise ValueError("a split needs at least one shard")
    entries = []
    for index in range(count):
        chunk = records[index * len(records) // count : (index + 1) * len(records) // count]
        data = "".join(
            json.dumps(record.to_json(), separators=(",", ":")) + "\n" for record in chunk
        ).encode("ascii")
        rel = shard_name(split, index, count)
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(
            {"path": rel, "split": split, "records": len(chunk), "bytes": len(data),
             "sha256": _sha256(data)}
        )  # fmt: skip
    return entries


def read_shards(
    paths: Iterable[Path], *, n_augmentations: int | None = None, max_tokens: int | None = None
) -> Iterator[EpisodeRecord]:
    """Read records from JSONL shards, validating each line (``validate_record``).

    Args:
        paths: Shard files, read in order.
        n_augmentations: Passed to ``validate_record``.
        max_tokens: Passed to ``validate_record``.

    Yields:
        The records in file order.

    Raises:
        RecordError: If a line is not JSON or breaks the schema; the message names file and line.
    """
    for path in paths:
        with open(path, encoding="ascii") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    record = validate_record(
                        json.loads(line), n_augmentations=n_augmentations, max_tokens=max_tokens
                    )
                except ValueError as err:
                    raise RecordError(f"{Path(path).name}:{number}: {err}") from err
                yield record


# The leakage gates (note, section 2.3).


def _eval_items(
    challenges: Mapping[str, Any], solutions: Mapping[str, Any]
) -> Iterator[tuple[str, str, np.ndarray, np.ndarray]]:
    # Yields (task_id, "train" or "test", input, output). Messages never quote grids.
    if set(challenges) != set(solutions):
        raise ValueError("evaluation challenges and solutions name different tasks")
    for task_id in sorted(challenges):
        task = challenges[task_id]
        try:
            train, test, outputs = task["train"], task["test"], solutions[task_id]
            if len(test) != len(outputs):
                raise ValueError("test inputs and solutions differ in number")
            for pair in train:
                yield task_id, "train", validate_grid(pair["input"]), validate_grid(pair["output"])
            for pair, output in zip(test, outputs, strict=True):
                yield task_id, "test", validate_grid(pair["input"]), validate_grid(output)
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(
                f"evaluation task {task_id} is malformed ({type(err).__name__})"
            ) from err


class _EvalKeys:
    """Hash indexes of the evaluation set: each key maps to the evaluation task ids holding it."""

    def __init__(self, challenges: Mapping[str, Any], solutions: Mapping[str, Any]) -> None:
        self.ids = set(challenges)
        self.pair_exact: dict[str, set[str]] = defaultdict(set)
        self.pair_canonical: dict[str, set[str]] = defaultdict(set)
        # Grid indexes: (role, kind) -> key -> task ids; roles test_input, test_output, demo.
        self.grids: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.counts = Counter(tasks=len(challenges))
        for task_id, part, x, y in _eval_items(challenges, solutions):
            self.counts[f"{part}_pairs"] += 1
            self.pair_exact[exact_pair_hash(x, y)].add(task_id)
            self.pair_canonical[canonical_pair_hash(x, y)].add(task_id)
            roles = (
                [("test_input", x), ("test_output", y)]
                if part == "test"
                else [("demo", x), ("demo", y)]
            )
            for role, grid in roles:
                self.grids[role, "exact"][exact_grid_hash(grid)].add(task_id)
                self.grids[role, "canonical"][canonical_grid_hash(grid)].add(task_id)


@dataclass
class _Seen:
    arrays: tuple[np.ndarray, ...]
    where: set[tuple[str, str, str]]


def _bucket(cells: int) -> str:
    for low, high in R1_BUCKETS:
        if low <= cells <= high:
            return f"{low}-{high}"
    raise ValueError(f"no bucket for {cells} cells")


def leakage_report(
    records: Iterable[EpisodeRecord],
    eval_challenges: Mapping[str, Any],
    eval_solutions: Mapping[str, Any],
    *,
    min_cells: int = 25,
    min_colors: int = 2,
    seed_ids: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Run gates G1 to G5 and report R1 over every pair of every record (note, section 2.3).

    - G1: a record's task id or family is an evaluation task id; ``seed_ids`` (e.g. BARC seeds)
      are checked the same way.
    - G2: $H_e(p)$ equals that of an evaluation demonstration or test pair.
    - G3: $H_c(p)$ equals that of an evaluation demonstration or test pair.
    - G4: an input grid with at least ``min_cells`` cells and ``min_colors`` colors has the $K_e$
      or $K_c$ of an evaluation test input.
    - G5: the same for output grids against the evaluation test outputs.
    - R1 (count only): unique grids whose $K_e$ or $K_c$ equals that of any evaluation
      demonstration grid, by cell-count bucket.
    - Split overlap: families, and canonical pairs, shared by ``val`` and ``train`` or ``toy``.

    Every pair and grid is hashed once however many episodes hold it. The result names tasks
    by id and never holds a grid.

    Args:
        records: All episodes to check (every split).
        eval_challenges: Evaluation challenges: task id to ``{"train": [{"input", "output"}],
            "test": [{"input"}]}``.
        eval_solutions: Task id to the test outputs, one per test input.
        min_cells: G4 and G5 cell floor (``is_nontrivial``).
        min_colors: G4 and G5 color floor.
        seed_ids: Provenance ids per source (a BARC problem's seeds), checked by G1 and reported
            as task ``seed:<id>`` of that source, split ``provenance``.

    Returns:
        A JSON-ready dict: ``eval`` and ``checked`` counts; ``gates`` with, per gate, the hit
        tuples, distinct source tasks, and tasks by source and split; ``hits`` as
        ``{gate, source, task_id, split, eval_task_id, match}``; ``drop``, the distinct
        ``{source, task_id}`` of every hit, ready for the drop list (a ``seed:<id>`` entry drops
        every task of that source that names the seed); ``reports`` with ``R1`` and
        ``split_overlap``; and ``pass``, true when G1 to G5 have no hit.

    Raises:
        ValueError: If the evaluation files are malformed.
    """
    ev = _EvalKeys(eval_challenges, eval_solutions)
    hits: set[tuple[str, str, str, str, str, str]] = set()
    pairs: dict[str, _Seen] = {}
    grids: dict[tuple[str, str], _Seen] = {}
    checked: Counter = Counter()
    family_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        where = (rec.source, rec.task_id, rec.split)
        checked[f"records_{rec.split}"] += 1
        family_splits[rec.family].add(rec.split)
        for ident in {rec.task_id, rec.family} & ev.ids:
            hits.add(("G1", *where, ident, "id"))
        for x, y in rec.pairs:
            checked["pair_occurrences"] += 1
            key = exact_pair_hash(x, y)
            pairs.setdefault(key, _Seen((x, y), set())).where.add(where)
            for role, grid in (("input", x), ("output", y)):
                grid_key = (role, exact_grid_hash(grid))
                grids.setdefault(grid_key, _Seen((grid,), set())).where.add(where)
    for source, ids in (seed_ids or {}).items():
        for ident in sorted(set(ids) & ev.ids):
            hits.add(("G1", source, f"{SEED_PREFIX}{ident}", "provenance", ident, "id"))

    canonical_splits: dict[str, set[str]] = defaultdict(set)
    canonical_where: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for key, seen in pairs.items():
        canonical = canonical_pair_hash(*seen.arrays)
        canonical_splits[canonical] |= {split for _, _, split in seen.where}
        canonical_where[canonical] |= seen.where
        for gate, match, eval_ids in (
            ("G2", "exact", ev.pair_exact.get(key, ())),
            ("G3", "canonical", ev.pair_canonical.get(canonical, ())),
        ):
            hits |= {(gate, *w, e, match) for w in seen.where for e in eval_ids}

    r1: dict[str, Counter] = {kind: Counter() for kind in ("exact", "canonical")}
    for (role, key), seen in grids.items():
        grid = seen.arrays[0]
        canonical = canonical_grid_hash(grid)
        keys = {"exact": key, "canonical": canonical}
        for kind in ("exact", "canonical"):
            if keys[kind] in ev.grids["demo", kind]:
                r1[kind][_bucket(grid.size)] += 1
        if not is_nontrivial(grid, min_cells, min_colors):
            continue
        gate, eval_role = ("G4", "test_input") if role == "input" else ("G5", "test_output")
        for kind in ("exact", "canonical"):
            for e in ev.grids[eval_role, kind].get(keys[kind], ()):
                hits |= {(gate, *w, e, kind) for w in seen.where}

    hit_rows = [
        dict(zip(("gate", "source", "task_id", "split", "eval_task_id", "match"), h, strict=True))
        for h in sorted(hits)
    ]
    gates = {}
    for gate in GATES:
        rows = [h for h in hit_rows if h["gate"] == gate]
        tasks = {(h["source"], h["task_id"]) for h in rows}
        gates[gate] = {
            "hits": len({(h["source"], h["task_id"], h["eval_task_id"]) for h in rows}),
            "tasks": len(tasks),
            "by_source": dict(sorted(Counter(s for s, _ in tasks).items())),
            "by_split": dict(sorted(Counter(h["split"] for h in rows).items())),
            "eval_tasks": len({h["eval_task_id"] for h in rows}),
        }
    # Every hit enters the drop list, so pasting it and rebuilding clears every gate.
    drop = sorted({(h["source"], h["task_id"]) for h in hit_rows})
    training = {"train", "toy"}
    val_shared = set()
    for canonical, splits in canonical_splits.items():
        if "val" in splits and splits & training:
            val_shared |= {
                (s, t) for s, t, split in canonical_where[canonical] if split in training
            }
    checked.update(unique_pairs=len(pairs), unique_grids=len(grids))
    return {
        "schema": 1,
        "eval": dict(sorted(ev.counts.items())),
        "thresholds": {"min_cells": min_cells, "min_colors": min_colors},
        "checked": dict(sorted(checked.items())),
        "gates": gates,
        "hits": hit_rows,
        "drop": [{"source": s, "task_id": t} for s, t in drop],
        "reports": {
            "R1": {kind: dict(sorted(counts.items())) for kind, counts in r1.items()},
            "split_overlap": {
                "val_families_in_training": sum(
                    "val" in splits and bool(splits & training) for splits in family_splits.values()
                ),
                "training_tasks_sharing_a_val_pair": len(val_shared),
            },
        },
        "pass": all(gates[gate]["tasks"] == 0 for gate in GATES),
    }


def positive_control(
    eval_challenges: Mapping[str, Any],
    eval_solutions: Mapping[str, Any],
    *,
    n: int,
    seed: int,
) -> list[tuple[EpisodeRecord, str]]:
    r"""Build records that each hide an evaluation test pair under a random $(g, \pi)$.

    Why: the check must show it can fail. Each control record holds one evaluation test pair moved
    by a random non-identity dihedral map and color permutation, redrawn until it differs from the
    original, so the exact gate G2 misses it and only G3 can catch it (note, section 2.3, row P).

    Args:
        eval_challenges: Evaluation challenges.
        eval_solutions: Evaluation test outputs.
        n: Number of controls, at most the number of test pairs.
        seed: Run seed.

    Returns:
        ``(record, eval_task_id)`` per control: source ``control``, task id ``control-<k>``, split
        ``train``, the moved pair as its one demo and its query.

    Raises:
        ValueError: If there are fewer than ``n`` test pairs or no transform moves a chosen pair.
    """
    rng = rng_for(seed, "positive-control")
    tests = [
        (t, x, y)
        for t, part, x, y in _eval_items(eval_challenges, eval_solutions)
        if part == "test"
    ]
    if len(tests) < n:
        raise ValueError(f"{n} controls need {n} test pairs, the evaluation set has {len(tests)}")
    controls = []
    for k, (task_id, x, y) in enumerate(rng.sample(tests, n)):
        original = exact_pair_hash(x, y)
        for _ in range(100):
            transpose, turns = rng.choice(DIHEDRAL[1:])
            perm = list(_IDENTITY)
            while tuple(perm) == _IDENTITY:
                rng.shuffle(perm)
            aug = Augmentation(transpose, turns, tuple(perm))
            moved = (aug.apply(x), aug.apply(y))
            if exact_pair_hash(*moved) != original:
                break
        else:
            raise ValueError(f"no transform moves the chosen test pair of {task_id}")
        name = f"control-{k}"
        record = EpisodeRecord.create(
            episode_id=f"control/{name}/train/0", source="control", task_id=name, family=name,
            split="train", augmentation_id=0, demos=[moved], queries=[moved],
        )  # fmt: skip
        controls.append((record, task_id))
    return controls


def control_outcome(
    report: Mapping[str, Any], controls: Sequence[tuple[EpisodeRecord, str]]
) -> dict[str, Any]:
    """Score the positive control in a ``leakage_report`` over records that include it.

    Args:
        report: The report.
        controls: ``positive_control``'s output.

    Returns:
        ``injected``; ``caught_g3`` and ``caught_g2``, the controls flagged by each gate with
        their own evaluation task; and ``pass``, true when G3 caught every control.
    """
    flagged = {
        gate: {(h["task_id"], h["eval_task_id"]) for h in report["hits"] if h["gate"] == gate}
        for gate in ("G2", "G3")
    }
    expected = [(record.task_id, eval_id) for record, eval_id in controls]
    caught = {gate: sum(item in flagged[gate] for item in expected) for gate in flagged}
    return {
        "injected": len(controls),
        "caught_g3": caught["G3"],
        "caught_g2": caught["G2"],
        "pass": caught["G3"] == len(controls),
    }


def pair_set_digests(tasks: Mapping[str, Iterable[ArrayPair]]) -> dict[str, dict[str, str]]:
    """Return per task the sha256 of its sorted exact and canonical pair-hash sets (report R3).

    Args:
        tasks: Pairs by task id.

    Returns:
        ``{task_id: {"exact": ..., "canonical": ...}}``.
    """
    digests = {}
    for task_id, task_pairs in tasks.items():
        task_pairs = list(task_pairs)
        exact = sorted({exact_pair_hash(x, y) for x, y in task_pairs})
        canonical = sorted({canonical_pair_hash(x, y) for x, y in task_pairs})
        digests[task_id] = {
            "exact": _sha256(",".join(exact).encode()),
            "canonical": _sha256(",".join(canonical).encode()),
        }
    return digests


def compare_pair_sets(
    ours: Mapping[str, Mapping[str, str]], theirs: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    """Compare two ``pair_set_digests`` results, e.g. GitHub's ARC-AGI-2 training with Kaggle's.

    Args:
        ours: Digests of the copy the episodes were built from.
        theirs: Digests of the other copy.

    Returns:
        Task counts (``ours``, ``theirs``, ``common``, ``only_ours``, ``only_theirs``), the common
        tasks with equal exact and equal canonical pair sets, and the sorted ids of common tasks
        whose canonical sets differ.
    """
    common = sorted(set(ours) & set(theirs))
    differ = [t for t in common if ours[t]["canonical"] != theirs[t]["canonical"]]
    return {
        "ours": len(ours),
        "theirs": len(theirs),
        "common": len(common),
        "only_ours": len(set(ours) - set(theirs)),
        "only_theirs": len(set(theirs) - set(ours)),
        "equal_exact": sum(ours[t]["exact"] == theirs[t]["exact"] for t in common),
        "equal_canonical": len(common) - len(differ),
        "canonical_differs": differ,
    }


def canonical_matches(
    reference: Iterable[ArrayPair], records: Iterable[EpisodeRecord]
) -> set[tuple[str, str]]:
    """Return the ``(source, task_id)`` of records holding a pair canonically equal to a reference.

    Why: the builder drops every training task with a pair in a validation puzzle, under any
    augmentation, before it writes the splits (note, 4.5).

    Args:
        reference: Pairs to match, e.g. every pair of every validation puzzle.
        records: Records to search.

    Returns:
        The matching tasks.
    """
    keys = {canonical_pair_hash(x, y) for x, y in reference}
    cache: dict[str, str] = {}
    found = set()
    for rec in records:
        for x, y in rec.pairs:
            exact = exact_pair_hash(x, y)
            if exact not in cache:
                cache[exact] = canonical_pair_hash(x, y)
            if cache[exact] in keys:
                found.add((rec.source, rec.task_id))
    return found
