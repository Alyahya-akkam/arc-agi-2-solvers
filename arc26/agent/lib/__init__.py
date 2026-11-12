"""Frozen helper library v0 for the agent's programs (P6.1; ADR-020 decision 3).

Why: the agent writes ``transform(grid)`` in numpy; tested helpers for the operations ARC solvers
use most (the map is Hodel's ARC DSL, MIT: connected objects, boxes and crops, fill and recolor,
palettes, rotations and flips, tiling and scaling, gravity and shifts, symmetry, partitions)
spare it from rewriting and debugging them in every attempt. Written from scratch, after the
DSL's ideas, not its code. The REPL and the verifier preload this module as ``lib``; programs may
also write ``from arc26.agent import lib``. The prompt carries :func:`index_text`.

Conventions:

- A grid is a 2-D integer numpy array; nested lists are accepted wherever a grid is.
- Coordinates are ``(row, col)`` from the top-left cell.
- A box is ``(top, left, bottom, right)`` with ``bottom`` and ``right`` exclusive, so
  ``grid[top:bottom, left:right]`` is the box.
- A mask is a boolean array of the grid's shape.
- Every helper returns new arrays and never changes its arguments.
"""

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Object",
    "adjacent",
    "anti_transpose",
    "background_color",
    "bbox",
    "color_counts",
    "complete_symmetry",
    "complete_tiling",
    "coords",
    "crop",
    "crop_to_content",
    "dihedral",
    "dilate",
    "doc",
    "downscale",
    "draw_line",
    "draw_rect",
    "fill",
    "fill_enclosed",
    "find_period",
    "flip_lr",
    "flip_ud",
    "flood_fill",
    "gravity",
    "hconcat",
    "hsplit",
    "index_text",
    "is_symmetric",
    "least_common_color",
    "mask_of",
    "most_common_color",
    "move",
    "objects",
    "outline",
    "paint",
    "palette",
    "ray",
    "recolor",
    "rot180",
    "rot270",
    "rot90",
    "shift",
    "split_grid",
    "tile",
    "transpose",
    "upscale",
    "vconcat",
    "vsplit",
]

GridLike = np.ndarray | Sequence[Sequence[int]]
Box = tuple[int, int, int, int]
Cell = tuple[int, int]

_ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(grid: GridLike) -> np.ndarray:
    array = np.asarray(grid)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got {array.ndim} dimensions")
    return array


def _mask(mask: GridLike) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got {array.ndim} dimensions")
    return array


def _steps(diagonal: bool) -> tuple[tuple[int, int], ...]:
    return _ORTHOGONAL + _DIAGONAL if diagonal else _ORTHOGONAL


def _components(mask: np.ndarray, diagonal: bool) -> list[np.ndarray]:
    """Connected components of a mask, in row-major order of their first cell."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for r0, c0 in zip(*np.nonzero(mask), strict=True):
        if seen[r0, c0]:
            continue
        component = np.zeros_like(mask, dtype=bool)
        queue = deque([(int(r0), int(c0))])
        seen[r0, c0] = True
        while queue:
            r, c = queue.popleft()
            component[r, c] = True
            for dr, dc in _steps(diagonal):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        found.append(component)
    return found


# Colors and background.


def palette(grid: GridLike) -> list[int]:
    """Return the distinct colors of a grid, ascending.

    Args:
        grid: The grid.

    Returns:
        The sorted colors.

    Example:
        >>> palette([[0, 3], [3, 1]])
        [0, 1, 3]
    """
    return [int(c) for c in np.unique(_grid(grid))]


def color_counts(grid: GridLike) -> dict[int, int]:
    """Count the cells of each color.

    Args:
        grid: The grid.

    Returns:
        A mapping from color to cell count, colors ascending.

    Example:
        >>> color_counts([[0, 3], [3, 3]])
        {0: 1, 3: 3}
    """
    colors, counts = np.unique(_grid(grid), return_counts=True)
    return {int(c): int(n) for c, n in zip(colors, counts, strict=True)}


def most_common_color(grid: GridLike, exclude: Iterable[int] = ()) -> int:
    """Return the most frequent color, ties broken by the smaller color.

    Args:
        grid: The grid.
        exclude: Colors to ignore, for example the background.

    Returns:
        The color.

    Raises:
        ValueError: If every color is excluded.

    Example:
        >>> most_common_color([[0, 0, 2], [2, 5, 2]])
        2
        >>> most_common_color([[0, 0, 2], [2, 5, 2]], exclude=[2])
        0
    """
    counts = {c: n for c, n in color_counts(grid).items() if c not in set(exclude)}
    if not counts:
        raise ValueError("no color left after exclusion")
    return min(counts, key=lambda c: (-counts[c], c))


def least_common_color(grid: GridLike, exclude: Iterable[int] = ()) -> int:
    """Return the least frequent color, ties broken by the smaller color.

    Args:
        grid: The grid.
        exclude: Colors to ignore.

    Returns:
        The color.

    Raises:
        ValueError: If every color is excluded.

    Example:
        >>> least_common_color([[0, 0, 2], [2, 5, 2]])
        5
    """
    counts = {c: n for c, n in color_counts(grid).items() if c not in set(exclude)}
    if not counts:
        raise ValueError("no color left after exclusion")
    return min(counts, key=lambda c: (counts[c], c))


def background_color(grid: GridLike) -> int:
    """Guess the background: the most frequent color, ties broken by frequency on the border.

    Args:
        grid: The grid.

    Returns:
        The background color (remaining ties go to the smaller color).

    Example:
        >>> background_color([[7, 7, 7], [7, 1, 1], [7, 1, 1]])
        7
    """
    g = _grid(grid)
    counts = color_counts(g)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    on_border = {int(c): int(np.sum(border == c)) for c in counts}
    return min(counts, key=lambda c: (-counts[c], -on_border[c], c))


def recolor(grid: GridLike, mapping: Mapping[int, int]) -> np.ndarray:
    """Replace colors by a mapping, all at once (so swaps work).

    Args:
        grid: The grid.
        mapping: Old color to new color; colors not in it stay.

    Returns:
        The recolored grid.

    Example:
        >>> recolor([[1, 2], [2, 0]], {1: 2, 2: 1}).tolist()
        [[2, 1], [1, 0]]
    """
    g = _grid(grid)
    out = g.copy()
    for old, new in mapping.items():
        out[g == old] = new
    return out


# Masks.


def mask_of(grid: GridLike, colors: int | Iterable[int]) -> np.ndarray:
    """Return the mask of the cells with a color, or with any of several colors.

    Args:
        grid: The grid.
        colors: One color or an iterable of colors.

    Returns:
        A boolean mask.

    Example:
        >>> mask_of([[1, 2], [2, 3]], 2).tolist()
        [[False, True], [True, False]]
        >>> mask_of([[1, 2], [2, 3]], [1, 3]).astype(int).tolist()
        [[1, 0], [0, 1]]
    """
    g = _grid(grid)
    if isinstance(colors, int | np.integer):
        return g == colors
    return np.isin(g, list(colors))


def fill(grid: GridLike, mask: GridLike, color: int) -> np.ndarray:
    """Set the masked cells to one color.

    Args:
        grid: The grid.
        mask: Boolean mask of the grid's shape.
        color: The new color.

    Returns:
        The filled grid.

    Example:
        >>> fill([[0, 0], [0, 0]], [[True, False], [False, True]], 4).tolist()
        [[4, 0], [0, 4]]
    """
    out = _grid(grid).copy()
    out[_mask(mask)] = color
    return out


def coords(mask: GridLike) -> list[Cell]:
    """List the coordinates of the true cells of a mask, row by row.

    Args:
        mask: Boolean mask.

    Returns:
        ``(row, col)`` pairs.

    Example:
        >>> coords([[False, True], [True, False]])
        [(0, 1), (1, 0)]
    """
    return [(int(r), int(c)) for r, c in np.argwhere(_mask(mask))]


def dilate(mask: GridLike, diagonal: bool = False, steps: int = 1) -> np.ndarray:
    """Grow a mask by its neighbors.

    Args:
        mask: Boolean mask.
        diagonal: Grow into the 8 neighbors instead of the 4 orthogonal ones.
        steps: How many times to grow.

    Returns:
        The grown mask.

    Example:
        >>> dilate([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(int).tolist()
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    """
    out = _mask(mask).copy()
    for _ in range(steps):
        grown = out.copy()
        for dr, dc in _steps(diagonal):
            grown |= shift(out.astype(np.int8), dr, dc, fill=0).astype(bool)
        out = grown
    return out


def outline(mask: GridLike, diagonal: bool = False) -> np.ndarray:
    """Return the ring of cells just outside a mask.

    Args:
        mask: Boolean mask.
        diagonal: Include diagonal neighbors.

    Returns:
        Cells adjacent to the mask but not in it.

    Example:
        >>> outline([[0, 0, 0], [0, 1, 0], [0, 0, 0]], diagonal=True).astype(int).tolist()
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    """
    m = _mask(mask)
    return dilate(m, diagonal) & ~m


def flood_fill(grid: GridLike, start: Cell, color: int, diagonal: bool = False) -> np.ndarray:
    """Recolor the same-colored region that contains a cell.

    Args:
        grid: The grid.
        start: ``(row, col)`` of a cell in the region.
        color: The new color.
        diagonal: Connect through diagonal neighbors too.

    Returns:
        The filled grid.

    Example:
        >>> flood_fill([[0, 0, 1], [1, 1, 1], [0, 0, 0]], (0, 0), 5).tolist()
        [[5, 5, 1], [1, 1, 1], [0, 0, 0]]
    """
    g = _grid(grid)
    region = g == g[start]
    for component in _components(region, diagonal):
        if component[start]:
            return fill(g, component, color)
    return g.copy()


def fill_enclosed(
    grid: GridLike, color: int, background: int | None = None, diagonal: bool = False
) -> np.ndarray:
    """Fill the background cells that cannot reach the border (holes inside shapes).

    Args:
        grid: The grid.
        color: The fill color.
        background: The background color; None detects it with :func:`background_color`.
        diagonal: Let background escape through diagonal gaps too.

    Returns:
        The filled grid.

    Example:
        >>> fill_enclosed([[0, 0, 0, 0], [0, 2, 2, 2], [0, 2, 0, 2], [0, 2, 2, 2]], 4).tolist()
        [[0, 0, 0, 0], [0, 2, 2, 2], [0, 2, 4, 2], [0, 2, 2, 2]]
    """
    g = _grid(grid)
    bg = background_color(g) if background is None else background
    out = g.copy()
    for component in _components(g == bg, diagonal):
        touches = component[0].any() or component[-1].any()
        touches = touches or component[:, 0].any() or component[:, -1].any()
        if not touches:
            out[component] = color
    return out


# Objects.


@dataclass(frozen=True, eq=False)
class Object:
    """A connected object found by :func:`objects`.

    Attributes:
        color: The object's most frequent color (ties to the smaller color).
        colors: Its distinct colors, ascending.
        size: Its number of cells.
        bbox: Its box ``(top, left, bottom, right)``, bottom and right exclusive.
        mask: Its cells as a boolean mask of the whole grid's shape.
        patch: The grid inside the box, with cells outside the object set to the background.
    """

    color: int
    colors: tuple[int, ...]
    size: int
    bbox: Box
    mask: np.ndarray
    patch: np.ndarray

    @property
    def top(self) -> int:
        """Return the first row of the box."""
        return self.bbox[0]

    @property
    def left(self) -> int:
        """Return the first column of the box."""
        return self.bbox[1]

    @property
    def height(self) -> int:
        """Return the number of rows of the box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def width(self) -> int:
        """Return the number of columns of the box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def cells(self) -> list[Cell]:
        """Return the object's ``(row, col)`` cells, row by row."""
        return coords(self.mask)

    @property
    def shape_mask(self) -> np.ndarray:
        """Return the object's mask cropped to its box (its shape, independent of position)."""
        top, left, bottom, right = self.bbox
        return self.mask[top:bottom, left:right].copy()


def objects(
    grid: GridLike,
    diagonal: bool = False,
    multicolor: bool = False,
    background: int | None = None,
) -> list[Object]:
    """Find the connected objects of a grid.

    Args:
        grid: The grid.
        diagonal: Connect cells through diagonal neighbors too (8-connectivity).
        multicolor: Connect cells of any non-background colors; False keeps one color per
            object.
        background: The color that is not part of objects; None detects it with
            :func:`background_color`; a color absent from the grid (for example -1) keeps
            every cell.

    Returns:
        The objects, ordered by the top row, then the left column, of their boxes.

    Example:
        >>> grid = [[0, 1, 0, 2], [0, 1, 0, 2], [0, 0, 0, 2]]
        >>> [(o.color, o.size, o.bbox) for o in objects(grid)]
        [(1, 2, (0, 1, 2, 2)), (2, 3, (0, 3, 3, 4))]
    """
    g = _grid(grid)
    bg = background_color(g) if background is None else background
    foreground = g != bg
    groups = (
        [foreground] if multicolor else [foreground & (g == c) for c in np.unique(g[foreground])]
    )
    found = []
    for group in groups:
        for component in _components(group, diagonal):
            top, left, bottom, right = bbox(component)
            values = g[component]
            patch = np.where(component, g, bg)[top:bottom, left:right]
            found.append(
                Object(
                    color=most_common_color(values.reshape(1, -1)),
                    colors=tuple(int(c) for c in np.unique(values)),
                    size=int(component.sum()),
                    bbox=(top, left, bottom, right),
                    mask=component,
                    patch=patch.astype(g.dtype),
                )
            )
    found.sort(key=lambda o: (o.bbox[0], o.bbox[1], o.color))
    return found


def bbox(mask: GridLike) -> Box:
    """Return the box of the true cells of a mask.

    Args:
        mask: Boolean mask (for a grid, pass for example ``grid != background``).

    Returns:
        ``(top, left, bottom, right)``, bottom and right exclusive.

    Raises:
        ValueError: If the mask is empty.

    Example:
        >>> bbox([[0, 0, 0], [0, 1, 1], [0, 0, 1]])
        (1, 1, 3, 3)
    """
    m = _mask(mask)
    rows, cols = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
    if rows.size == 0:
        raise ValueError("empty mask has no box")
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def crop(grid: GridLike, box: Box) -> np.ndarray:
    """Return a copy of the grid inside a box.

    Args:
        grid: The grid.
        box: ``(top, left, bottom, right)``, bottom and right exclusive.

    Returns:
        The cropped grid.

    Example:
        >>> crop([[1, 2, 3], [4, 5, 6]], (0, 1, 2, 3)).tolist()
        [[2, 3], [5, 6]]
    """
    top, left, bottom, right = box
    return _grid(grid)[top:bottom, left:right].copy()


def crop_to_content(grid: GridLike, background: int | None = None) -> np.ndarray:
    """Crop away the background around the content.

    Args:
        grid: The grid.
        background: The background color; None detects it.

    Returns:
        The smallest box holding every non-background cell (a copy of the grid if there is
        none).

    Example:
        >>> crop_to_content([[0, 0, 0], [0, 3, 0], [0, 3, 3]]).tolist()
        [[3, 0], [3, 3]]
    """
    g = _grid(grid)
    bg = background_color(g) if background is None else background
    if not (g != bg).any():
        return g.copy()
    return crop(g, bbox(g != bg))


def adjacent(mask_a: GridLike, mask_b: GridLike, diagonal: bool = False) -> bool:
    """Return whether two masks touch.

    Args:
        mask_a: Boolean mask.
        mask_b: Boolean mask of the same shape.
        diagonal: Count diagonal contact.

    Returns:
        True when a cell of one is next to (or on) a cell of the other.

    Example:
        >>> adjacent([[1, 0, 0]], [[0, 1, 0]]), adjacent([[1, 0, 0]], [[0, 0, 1]])
        (True, False)
    """
    return bool((dilate(mask_a, diagonal) & _mask(mask_b)).any())


def paint(
    grid: GridLike, patch: GridLike, top: int, left: int, transparent: int | None = None
) -> np.ndarray:
    """Copy a patch onto a grid at a position, clipped at the borders.

    Args:
        grid: The grid.
        patch: The grid to paint.
        top: Row of the patch's top-left cell (may be negative).
        left: Column of the patch's top-left cell (may be negative).
        transparent: A patch color that is not painted, or None to paint every cell.

    Returns:
        The painted grid.

    Example:
        >>> paint([[0, 0, 0], [0, 0, 0]], [[5, 0], [5, 5]], 0, 1, transparent=0).tolist()
        [[0, 5, 0], [0, 5, 5]]
    """
    out = _grid(grid).copy()
    p = _grid(patch)
    rows, cols = out.shape
    r0, c0 = max(top, 0), max(left, 0)
    r1, c1 = min(top + p.shape[0], rows), min(left + p.shape[1], cols)
    if r0 >= r1 or c0 >= c1:
        return out
    piece = p[r0 - top : r1 - top, c0 - left : c1 - left]
    target = out[r0:r1, c0:c1]
    keep = np.ones_like(piece, dtype=bool) if transparent is None else piece != transparent
    target[keep] = piece[keep]
    return out


def move(
    grid: GridLike, mask: GridLike, dr: int, dc: int, background: int | None = None
) -> np.ndarray:
    """Move the masked cells by an offset, leaving background behind.

    Args:
        grid: The grid.
        mask: The cells to move.
        dr: Rows to move down (negative moves up).
        dc: Columns to move right (negative moves left).
        background: The color left behind; None detects it.

    Returns:
        The grid after the move; cells moved off the grid are lost.

    Example:
        >>> move([[3, 0, 0]], [[True, False, False]], 0, 2).tolist()
        [[0, 0, 3]]
    """
    g = _grid(grid)
    m = _mask(mask)
    bg = background_color(g) if background is None else background
    out = g.copy()
    out[m] = bg
    for r, c in coords(m):
        nr, nc = r + dr, c + dc
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
            out[nr, nc] = g[r, c]
    return out


# Rotations, flips, and transposes.


def rot90(grid: GridLike) -> np.ndarray:
    """Rotate a quarter turn clockwise.

    Args:
        grid: The grid.

    Returns:
        The rotated grid.

    Example:
        >>> rot90([[1, 2], [3, 4]]).tolist()
        [[3, 1], [4, 2]]
    """
    return np.rot90(_grid(grid), -1).copy()


def rot180(grid: GridLike) -> np.ndarray:
    """Rotate a half turn.

    Args:
        grid: The grid.

    Returns:
        The rotated grid.

    Example:
        >>> rot180([[1, 2], [3, 4]]).tolist()
        [[4, 3], [2, 1]]
    """
    return np.rot90(_grid(grid), 2).copy()


def rot270(grid: GridLike) -> np.ndarray:
    """Rotate a quarter turn counterclockwise (the inverse of :func:`rot90`).

    Args:
        grid: The grid.

    Returns:
        The rotated grid.

    Example:
        >>> rot270([[1, 2], [3, 4]]).tolist()
        [[2, 4], [1, 3]]
    """
    return np.rot90(_grid(grid), 1).copy()


def flip_lr(grid: GridLike) -> np.ndarray:
    """Mirror left to right (reverse each row).

    Args:
        grid: The grid.

    Returns:
        The mirrored grid.

    Example:
        >>> flip_lr([[1, 2], [3, 4]]).tolist()
        [[2, 1], [4, 3]]
    """
    return _grid(grid)[:, ::-1].copy()


def flip_ud(grid: GridLike) -> np.ndarray:
    """Mirror top to bottom (reverse the row order).

    Args:
        grid: The grid.

    Returns:
        The mirrored grid.

    Example:
        >>> flip_ud([[1, 2], [3, 4]]).tolist()
        [[3, 4], [1, 2]]
    """
    return _grid(grid)[::-1, :].copy()


def transpose(grid: GridLike) -> np.ndarray:
    """Mirror across the main diagonal (rows become columns).

    Args:
        grid: The grid.

    Returns:
        The transposed grid.

    Example:
        >>> transpose([[1, 2], [3, 4]]).tolist()
        [[1, 3], [2, 4]]
    """
    return _grid(grid).T.copy()


def anti_transpose(grid: GridLike) -> np.ndarray:
    """Mirror across the anti-diagonal.

    Args:
        grid: The grid.

    Returns:
        The mirrored grid.

    Example:
        >>> anti_transpose([[1, 2], [3, 4]]).tolist()
        [[4, 2], [3, 1]]
    """
    return _grid(grid)[::-1, ::-1].T.copy()


_DIHEDRAL = (lambda g: g.copy(), rot90, rot180, rot270, flip_lr, flip_ud, transpose, anti_transpose)


def dihedral(grid: GridLike, k: int) -> np.ndarray:
    """Apply one of the 8 rotations and reflections of the square.

    Args:
        grid: The grid.
        k: 0 identity, 1 :func:`rot90`, 2 :func:`rot180`, 3 :func:`rot270`, 4 :func:`flip_lr`,
            5 :func:`flip_ud`, 6 :func:`transpose`, 7 :func:`anti_transpose`. Each is its own
            inverse except 1 and 3, which invert each other.

    Returns:
        The transformed grid.

    Raises:
        ValueError: If k is not in 0..7.

    Example:
        >>> [dihedral([[1, 2]], k).tolist() for k in (0, 1, 4)]
        [[[1, 2]], [[1], [2]], [[2, 1]]]
    """
    if not 0 <= k < 8:
        raise ValueError(f"k must be in 0..7, got {k}")
    return _DIHEDRAL[k](_grid(grid))


# Scaling, tiling, and concatenation.


def upscale(grid: GridLike, factor: int | tuple[int, int]) -> np.ndarray:
    """Scale up: each cell becomes a block.

    Args:
        grid: The grid.
        factor: Block size, or ``(rows, cols)`` for a rectangular block.

    Returns:
        The scaled grid.

    Example:
        >>> upscale([[1, 2]], 2).tolist()
        [[1, 1, 2, 2], [1, 1, 2, 2]]
    """
    fr, fc = (factor, factor) if isinstance(factor, int) else factor
    return np.repeat(np.repeat(_grid(grid), fr, axis=0), fc, axis=1)


def downscale(grid: GridLike, factor: int | tuple[int, int]) -> np.ndarray:
    """Scale down: each block becomes its top-left cell.

    Args:
        grid: The grid.
        factor: Block size, or ``(rows, cols)``.

    Returns:
        The scaled grid.

    Raises:
        ValueError: If the grid's shape is not a multiple of the block size.

    Example:
        >>> downscale([[1, 1, 2, 2], [1, 1, 2, 2]], 2).tolist()
        [[1, 2]]
    """
    g = _grid(grid)
    fr, fc = (factor, factor) if isinstance(factor, int) else factor
    if g.shape[0] % fr or g.shape[1] % fc:
        raise ValueError(f"shape {g.shape} is not a multiple of {(fr, fc)}")
    return g[::fr, ::fc].copy()


def tile(grid: GridLike, rows: int, cols: int) -> np.ndarray:
    """Repeat a grid in a rows by cols arrangement.

    Args:
        grid: The grid.
        rows: Copies stacked vertically.
        cols: Copies side by side.

    Returns:
        The tiled grid.

    Example:
        >>> tile([[1, 2]], 2, 2).tolist()
        [[1, 2, 1, 2], [1, 2, 1, 2]]
    """
    return np.tile(_grid(grid), (rows, cols))


def hconcat(*grids: GridLike) -> np.ndarray:
    """Join grids of equal height side by side.

    Args:
        *grids: The grids, left to right.

    Returns:
        The joined grid.

    Example:
        >>> hconcat([[1], [2]], [[3], [4]]).tolist()
        [[1, 3], [2, 4]]
    """
    return np.concatenate([_grid(g) for g in grids], axis=1)


def vconcat(*grids: GridLike) -> np.ndarray:
    """Stack grids of equal width top to bottom.

    Args:
        *grids: The grids, top to bottom.

    Returns:
        The stacked grid.

    Example:
        >>> vconcat([[1, 2]], [[3, 4]]).tolist()
        [[1, 2], [3, 4]]
    """
    return np.concatenate([_grid(g) for g in grids], axis=0)


# Shifts and gravity.


def shift(grid: GridLike, dr: int, dc: int, fill: int | None = None) -> np.ndarray:
    """Translate the whole grid content, keeping the grid's size.

    Args:
        grid: The grid.
        dr: Rows to move down (negative moves up).
        dc: Columns to move right (negative moves left).
        fill: Color of the vacated cells; None uses :func:`background_color`.

    Returns:
        The shifted grid; content moved off the grid is lost.

    Example:
        >>> shift([[1, 2], [3, 4]], 0, 1, fill=0).tolist()
        [[0, 1], [0, 3]]
    """
    g = _grid(grid)
    color = background_color(g) if fill is None else fill
    out = np.full_like(g, color)
    rows, cols = g.shape
    if abs(dr) >= rows or abs(dc) >= cols:
        return out
    src_r = slice(max(-dr, 0), rows - max(dr, 0))
    src_c = slice(max(-dc, 0), cols - max(dc, 0))
    dst_r = slice(max(dr, 0), rows - max(-dr, 0))
    dst_c = slice(max(dc, 0), cols - max(-dc, 0))
    out[dst_r, dst_c] = g[src_r, src_c]
    return out


def gravity(grid: GridLike, direction: str = "down", background: int | None = None) -> np.ndarray:
    """Let every non-background cell fall as far as it can, keeping the order in its line.

    Args:
        grid: The grid.
        direction: ``"down"``, ``"up"``, ``"left"``, or ``"right"``.
        background: The background color; None detects it.

    Returns:
        The grid after the fall.

    Raises:
        ValueError: If the direction is unknown.

    Example:
        >>> gravity([[1, 0], [0, 0], [2, 3]], "down", background=0).tolist()
        [[0, 0], [1, 0], [2, 3]]
    """
    turns = {"down": 0, "left": 1, "up": 2, "right": 3}
    if direction not in turns:
        raise ValueError(f"unknown direction {direction!r}")
    g = _grid(grid)
    bg = background_color(g) if background is None else background
    # Rotate so the fall is downward, settle each column, rotate back.
    work = np.rot90(g, turns[direction])
    out = np.full_like(work, bg)
    for c in range(work.shape[1]):
        column = work[:, c]
        cells = column[column != bg]
        if cells.size:
            out[-cells.size :, c] = cells
    return np.rot90(out, -turns[direction]).copy()


# Symmetry and periodicity.

_SQUARE_ONLY = frozenset({"diag", "anti", "rot90"})


def _mirror(g: np.ndarray, kind: str) -> np.ndarray:
    images = {
        "lr": lambda: g[:, ::-1],
        "ud": lambda: g[::-1, :],
        "rot180": lambda: g[::-1, ::-1],
        "diag": lambda: g.T,
        "anti": lambda: g[::-1, ::-1].T,
        "rot90": lambda: np.rot90(g, -1),
    }
    if kind not in images:
        raise ValueError(f"unknown symmetry {kind!r}")
    return images[kind]()


def is_symmetric(grid: GridLike, kind: str) -> bool:
    """Return whether a grid equals its image under a symmetry.

    Args:
        grid: The grid.
        kind: ``"lr"`` (left-right mirror), ``"ud"`` (top-bottom mirror), ``"rot180"``,
            ``"diag"`` (transpose), ``"anti"`` (anti-transpose), or ``"rot90"`` (quarter-turn
            symmetry); the last three need a square grid.

    Returns:
        True when the grid is symmetric.

    Raises:
        ValueError: If the kind is unknown.

    Example:
        >>> is_symmetric([[1, 2, 1]], "lr"), is_symmetric([[1, 2, 1]], "ud")
        (True, True)
    """
    g = _grid(grid)
    image = _mirror(g, kind)
    return image.shape == g.shape and bool(np.array_equal(image, g))


def complete_symmetry(
    grid: GridLike, unknown: int, kinds: Sequence[str] = ("lr", "ud", "rot180", "diag", "anti")
) -> np.ndarray:
    """Fill cells of an unknown color from their mirror images under the symmetries that hold.

    A symmetry holds when every pair of known cells it maps onto each other agrees. Cells whose
    images are all unknown stay unknown. The symmetry axes are the grid's center lines.

    Args:
        grid: The grid.
        unknown: The color that marks missing cells.
        kinds: Symmetries to try, as in :func:`is_symmetric`.

    Returns:
        The completed grid.

    Example:
        >>> complete_symmetry([[1, 2, 2, 1], [3, 4, 0, 3]], unknown=0).tolist()
        [[1, 2, 2, 1], [3, 4, 4, 3]]
    """
    out = _grid(grid).copy()
    for _ in range(len(kinds) + 1):
        changed = False
        for kind in kinds:
            if kind in _SQUARE_ONLY and out.shape[0] != out.shape[1]:
                continue
            image = _mirror(out, kind)
            known = (out != unknown) & (image != unknown)
            if np.any(out[known] != image[known]):
                continue
            missing = (out == unknown) & (image != unknown)
            if missing.any():
                out[missing] = image[missing]
                changed = True
        if not changed:
            break
    return out


def _period(rows: np.ndarray, unknown: int | None) -> int:
    height = rows.shape[0]
    for p in range(1, height):
        a, b = rows[p:], rows[:-p]
        same = a == b
        if unknown is not None:
            same |= (a == unknown) | (b == unknown)
        if same.all():
            return p
    return height


def find_period(grid: GridLike, unknown: int | None = None) -> tuple[int, int]:
    """Find the smallest row and column periods of a repeating pattern.

    Args:
        grid: The grid.
        unknown: A color that matches anything (missing cells), or None.

    Returns:
        ``(row_period, col_period)``: the grid equals itself shifted by that many rows, or
        columns, wherever both cells are known. A period equal to the grid's size means none.

    Example:
        >>> find_period([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2]])
        (2, 2)
    """
    g = _grid(grid)
    return _period(g, unknown), _period(g.T, unknown)


def complete_tiling(grid: GridLike, unknown: int) -> np.ndarray:
    """Fill cells of an unknown color from the grid's repeating pattern.

    Each cell takes the most frequent known color among the cells that share its position in
    the period found by :func:`find_period`.

    Args:
        grid: The grid.
        unknown: The color that marks missing cells.

    Returns:
        The completed grid; cells with no known counterpart stay unknown.

    Example:
        >>> complete_tiling([[1, 2, 1, 2], [3, 4, 0, 4], [1, 0, 1, 2]], unknown=0).tolist()
        [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2]]
    """
    g = _grid(grid)
    pr, pc = find_period(g, unknown)
    out = g.copy()
    for r in range(pr):
        for c in range(pc):
            cells = g[r::pr, c::pc]
            known = cells[cells != unknown]
            if known.size:
                out[r::pr, c::pc] = most_common_color(known.reshape(1, -1))
    return out


# Partitions.


def _separator_lines(g: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(g.shape[0]) if (g[r] == color).all()]
    cols = [c for c in range(g.shape[1]) if (g[:, c] == color).all()]
    return rows, cols


def _separator_color(g: np.ndarray) -> int | None:
    bg = background_color(g)
    best = None
    for color in palette(g):
        rows, cols = _separator_lines(g, color)
        blocks = len(_spans(rows, g.shape[0])) * len(_spans(cols, g.shape[1]))
        # Lines that leave a single block are a frame or a one-cell-wide grid, not a partition.
        if (rows or cols) and blocks >= 2:
            # Prefer a non-background color: separators are drawn, empty rows of content are not.
            key = (color == bg, -(len(rows) + len(cols)), color)
            if best is None or key < best[0]:
                best = (key, color)
    return None if best is None else best[1]


def _spans(lines: list[int], size: int) -> list[tuple[int, int]]:
    edges = [-1, *lines, size]
    return [(a + 1, b) for a, b in zip(edges, edges[1:], strict=False) if b > a + 1]


def split_grid(grid: GridLike, separator: int | None = None) -> list[list[np.ndarray]]:
    """Split a grid into the cells between full separator lines.

    Args:
        grid: The grid.
        separator: Color of the separator rows and columns; None picks the color with the most
            full-width rows and full-height columns that split the grid into at least two
            blocks, preferring non-background colors.

    Returns:
        Rows of blocks: ``blocks[i][j]`` is the block in the i-th band from the top and j-th
        from the left. Without separator lines, ``[[copy of grid]]``.

    Example:
        >>> blocks = split_grid([[1, 5, 2], [5, 5, 5], [3, 5, 4]])
        >>> [[b.tolist() for b in row] for row in blocks]
        [[[[1]], [[2]]], [[[3]], [[4]]]]
    """
    g = _grid(grid)
    color = _separator_color(g) if separator is None else separator
    if color is None:
        return [[g.copy()]]
    rows, cols = _separator_lines(g, color)
    return [
        [g[r0:r1, c0:c1].copy() for c0, c1 in _spans(cols, g.shape[1])]
        for r0, r1 in _spans(rows, g.shape[0])
    ]


def hsplit(grid: GridLike, n: int) -> list[np.ndarray]:
    """Split a grid into n equal parts side by side.

    Args:
        grid: The grid.
        n: Number of parts.

    Returns:
        The parts, left to right.

    Raises:
        ValueError: If the width is not a multiple of n.

    Example:
        >>> [p.tolist() for p in hsplit([[1, 2, 3, 4]], 2)]
        [[[1, 2]], [[3, 4]]]
    """
    g = _grid(grid)
    if g.shape[1] % n:
        raise ValueError(f"width {g.shape[1]} is not a multiple of {n}")
    return [part.copy() for part in np.split(g, n, axis=1)]


def vsplit(grid: GridLike, n: int) -> list[np.ndarray]:
    """Split a grid into n equal parts stacked top to bottom.

    Args:
        grid: The grid.
        n: Number of parts.

    Returns:
        The parts, top to bottom.

    Raises:
        ValueError: If the height is not a multiple of n.

    Example:
        >>> [p.tolist() for p in vsplit([[1], [2], [3], [4]], 2)]
        [[[1], [2]], [[3], [4]]]
    """
    g = _grid(grid)
    if g.shape[0] % n:
        raise ValueError(f"height {g.shape[0]} is not a multiple of {n}")
    return [part.copy() for part in np.split(g, n, axis=0)]


# Drawing.


def draw_line(grid: GridLike, start: Cell, end: Cell, color: int) -> np.ndarray:
    """Draw a horizontal, vertical, or 45 degree line, both ends included, clipped.

    Args:
        grid: The grid.
        start: ``(row, col)`` of one end.
        end: ``(row, col)`` of the other end.
        color: The line color.

    Returns:
        The grid with the line.

    Raises:
        ValueError: If the line is neither straight nor diagonal.

    Example:
        >>> draw_line([[0, 0, 0], [0, 0, 0], [0, 0, 0]], (0, 0), (2, 2), 4).tolist()
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    """
    out = _grid(grid).copy()
    (r0, c0), (r1, c1) = start, end
    if r0 != r1 and c0 != c1 and abs(r1 - r0) != abs(c1 - c0):
        raise ValueError(f"line from {start} to {end} is neither straight nor diagonal")
    dr, dc = int(np.sign(r1 - r0)), int(np.sign(c1 - c0))
    for i in range(max(abs(r1 - r0), abs(c1 - c0)) + 1):
        r, c = r0 + i * dr, c0 + i * dc
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = color
    return out


def draw_rect(grid: GridLike, box: Box, color: int, filled: bool = False) -> np.ndarray:
    """Draw a rectangle's outline, or fill it.

    Args:
        grid: The grid.
        box: ``(top, left, bottom, right)``, bottom and right exclusive.
        color: The color.
        filled: Fill the inside too.

    Returns:
        The grid with the rectangle.

    Example:
        >>> draw_rect([[0, 0, 0], [0, 0, 0], [0, 0, 0]], (0, 0, 3, 3), 2).tolist()
        [[2, 2, 2], [2, 0, 2], [2, 2, 2]]
    """
    out = _grid(grid).copy()
    top, left, bottom, right = box
    if filled:
        out[top:bottom, left:right] = color
        return out
    out[top, left:right] = color
    out[bottom - 1, left:right] = color
    out[top:bottom, left] = color
    out[top:bottom, right - 1] = color
    return out


def ray(
    grid: GridLike, start: Cell, direction: Cell, color: int, until: Iterable[int] = ()
) -> np.ndarray:
    """Draw from the cell after ``start`` in a direction to the border, or to a stop color.

    Args:
        grid: The grid.
        start: ``(row, col)`` the ray leaves from (not painted).
        direction: ``(dr, dc)`` step, for example ``(0, 1)`` right or ``(-1, -1)`` up-left.
        color: The ray color.
        until: Colors that stop the ray before the cell that has them.

    Returns:
        The grid with the ray.

    Example:
        >>> ray([[1, 0, 0, 3, 0]], (0, 0), (0, 1), 2, until=[3]).tolist()
        [[1, 2, 2, 3, 0]]
    """
    out = _grid(grid).copy()
    stop = set(until)
    r, c = start
    dr, dc = direction
    if dr == 0 and dc == 0:
        return out
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]) or int(out[r, c]) in stop:
            return out
        out[r, c] = color


# The prompt index.


def _signature(obj: object) -> str:
    import inspect

    parts = []
    for p in inspect.signature(obj).parameters.values():
        prefix = "*" if p.kind is p.VAR_POSITIONAL else ""
        default = "" if p.default is p.empty else f"={p.default!r}"
        parts.append(f"{prefix}{p.name}{default}")
    return f"({', '.join(parts)})"


def doc(name: str) -> str:
    """Return a helper's signature and full docstring.

    Args:
        name: The helper's name, for example ``"objects"``.

    Returns:
        The text.

    Raises:
        KeyError: If the name is not a public helper.

    Example:
        >>> doc("rot90").splitlines()[0]
        'rot90(grid)'
    """
    if name not in __all__:
        raise KeyError(f"no helper named {name!r}")
    import inspect

    obj = globals()[name]
    return f"{name}{_signature(obj)}\n{inspect.getdoc(obj)}"


def index_text() -> str:
    """Return the compact index of the helpers for the agent's prompt.

    Why: one line per helper (name, signature, summary) keeps the whole library within about
    1,500 tokens of the cached prompt prefix; ``lib.doc(name)`` gives the rest on demand.

    Returns:
        The index text.

    Example:
        >>> index_text().splitlines()[1].split(":")[0]
        'Object'
    """
    lines = [
        "lib (numpy helpers; grids are 2-D int arrays, boxes (top, left, bottom, right) with "
        "bottom/right exclusive, masks bool arrays; lib.doc(name) shows the full docstring):",
        "Object: color, colors, size, bbox, mask, patch; top, left, height, width, cells, "
        "shape_mask",
    ]
    for name in __all__:
        if name in {"Object", "doc", "index_text"}:
            continue
        obj = globals()[name]
        summary = (obj.__doc__ or "").strip().splitlines()[0]
        summary = summary.replace(":func:", "").replace("``", "").replace("`", "")
        lines.append(f"{name}{_signature(obj)}: {summary}")
    return "\n".join(lines)
