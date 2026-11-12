"""Tests of arc26.agent.lib: one case per helper, the docstring examples, purity, the index."""

import ast
import doctest
import inspect
import types

import numpy as np
import pytest

from arc26.agent import lib

G = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 1, 0, 0, 2],
        [0, 0, 0, 3, 0],
    ]
)


def eq(array, expected):
    return np.asarray(array).tolist() == expected


def case_palette():
    assert lib.palette(G) == [0, 1, 2, 3]


def case_color_counts():
    assert lib.color_counts(G) == {0: 14, 1: 3, 2: 2, 3: 1}


def case_most_common_color():
    assert lib.most_common_color(G) == 0
    assert lib.most_common_color(G, exclude=[0]) == 1
    assert lib.most_common_color([[4, 2, 2, 4]]) == 2
    with pytest.raises(ValueError):
        lib.most_common_color([[1]], exclude=[1])


def case_least_common_color():
    assert lib.least_common_color(G) == 3
    assert lib.least_common_color([[4, 2, 2, 4]]) == 2
    with pytest.raises(ValueError):
        lib.least_common_color([[1]], exclude=[1])


def case_background_color():
    assert lib.background_color(G) == 0
    # A tie (2 of each) goes to the color that dominates the border.
    assert lib.background_color([[5, 5, 5], [5, 1, 1], [1, 1, 5]]) == 5
    assert lib.background_color([[1, 1], [3, 3]]) == 1


def case_recolor():
    assert eq(lib.recolor([[1, 2, 3]], {1: 2, 2: 1}), [[2, 1, 3]])


def case_mask_of():
    assert lib.mask_of(G, 2).sum() == 2
    assert lib.mask_of(G, {1, 3}).sum() == 4
    assert lib.mask_of(G, np.int64(3)).sum() == 1


def case_fill():
    assert eq(lib.fill([[1, 1]], [[False, True]], 0), [[1, 0]])


def case_coords():
    assert lib.coords(G == 2) == [(1, 4), (2, 4)]
    assert lib.coords(np.zeros((2, 2), bool)) == []


def case_dilate():
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    assert lib.dilate(mask).sum() == 5
    assert lib.dilate(mask, diagonal=True).sum() == 9
    assert lib.dilate(mask, steps=2).sum() == 13
    assert lib.dilate([[True]]).tolist() == [[True]]


def case_outline():
    assert eq(lib.outline([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def case_flood_fill():
    grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert eq(lib.flood_fill(grid, (1, 1), 5), [[0, 1, 0], [1, 5, 1], [0, 1, 0]])
    assert eq(lib.flood_fill(grid, (1, 1), 5, diagonal=True), [[5, 1, 5], [1, 5, 1], [5, 1, 5]])


def case_fill_enclosed():
    ring = [[3, 3, 3], [3, 7, 3], [3, 3, 3]]
    assert eq(lib.fill_enclosed(ring, 4, background=7), [[3, 3, 3], [3, 4, 3], [3, 3, 3]])
    # A diagonal gap lets the inside escape only when diagonal moves count.
    leaky = [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 0, 2], [0, 0, 2, 2]]
    assert lib.fill_enclosed(leaky, 9, background=0)[2, 2] == 9
    assert lib.fill_enclosed(leaky, 9, background=0, diagonal=True)[2, 2] == 0


def case_objects():
    found = lib.objects(G)
    assert [(o.color, o.size, o.bbox) for o in found] == [
        (1, 3, (1, 1, 3, 3)),
        (2, 2, (1, 4, 3, 5)),
        (3, 1, (3, 3, 4, 4)),
    ]
    first = found[0]
    assert (first.top, first.left, first.height, first.width) == (1, 1, 2, 2)
    assert first.cells == [(1, 1), (1, 2), (2, 1)]
    assert eq(first.patch, [[1, 1], [1, 0]])
    assert eq(first.shape_mask, [[True, True], [True, False]])
    assert first.mask.shape == G.shape and first.colors == (1,)
    # 8-connectivity joins the 2s with the 3 through the diagonal at (2, 4) and (3, 3).
    assert len(lib.objects(G, diagonal=True, multicolor=True)) == 2
    two_colors = [[0, 4, 5], [0, 0, 0]]
    assert len(lib.objects(two_colors)) == 2
    (joined,) = lib.objects(two_colors, multicolor=True)
    assert joined.colors == (4, 5) and joined.size == 2
    assert len(lib.objects([[1, 1], [1, 1]], background=-1)) == 1


def case_bbox():
    assert lib.bbox(G == 1) == (1, 1, 3, 3)
    with pytest.raises(ValueError):
        lib.bbox(np.zeros((2, 2), bool))


def case_crop():
    assert eq(lib.crop(G, (1, 1, 3, 3)), [[1, 1], [1, 0]])


def case_crop_to_content():
    assert eq(lib.crop_to_content(G), [[1, 1, 0, 2], [1, 0, 0, 2], [0, 0, 3, 0]])
    assert eq(lib.crop_to_content([[4, 4], [4, 4]]), [[4, 4], [4, 4]])
    assert eq(lib.crop_to_content([[6, 6], [6, 1]], background=1), [[6, 6], [6, 1]])


def case_adjacent():
    assert lib.adjacent(G == 1, G == 3, diagonal=True) is False
    assert lib.adjacent(G == 2, G == 3, diagonal=True) is True
    assert lib.adjacent(G == 2, G == 3) is False


def case_paint():
    base = np.zeros((2, 2), int)
    assert eq(lib.paint(base, [[1, 2], [3, 4]], -1, -1), [[4, 0], [0, 0]])
    assert eq(lib.paint(base, [[1, 2], [3, 4]], 1, 1), [[0, 0], [0, 1]])
    assert eq(lib.paint(base, [[1]], 5, 5), [[0, 0], [0, 0]])


def case_move():
    moved = lib.move(G, G == 3, -3, 1)
    assert moved[0, 4] == 3 and moved[3, 3] == 0
    assert lib.move(G, G == 3, 0, 5).tolist().count([0, 0, 0, 0, 0]) == 2


def case_rot90():
    assert eq(lib.rot90([[1, 2, 3]]), [[1], [2], [3]])


def case_rot180():
    assert eq(lib.rot180([[1, 2, 3]]), [[3, 2, 1]])


def case_rot270():
    assert eq(lib.rot270(lib.rot90(G)), G.tolist())


def case_flip_lr():
    assert eq(lib.flip_lr([[1, 2, 3]]), [[3, 2, 1]])


def case_flip_ud():
    assert eq(lib.flip_ud([[1], [2]]), [[2], [1]])


def case_transpose():
    assert lib.transpose(G).shape == (5, 4)
    assert lib.transpose(G)[4, 1] == G[1, 4]


def case_anti_transpose():
    t = lib.anti_transpose(G)
    assert t.shape == (5, 4) and t[0, 0] == G[-1, -1]


def case_dihedral():
    images = [lib.dihedral(G, k) for k in range(8)]
    assert len({im.tobytes() + bytes(im.shape) for im in images}) == 8
    for k in (0, 2, 4, 5, 6, 7):
        assert eq(lib.dihedral(lib.dihedral(G, k), k), G.tolist())
    assert eq(lib.dihedral(lib.dihedral(G, 1), 3), G.tolist())
    with pytest.raises(ValueError):
        lib.dihedral(G, 8)


def case_upscale():
    assert lib.upscale(G, (2, 3)).shape == (8, 15)
    assert eq(lib.downscale(lib.upscale(G, 3), 3), G.tolist())


def case_downscale():
    assert eq(lib.downscale([[1, 2, 3, 4]], (1, 2)), [[1, 3]])
    with pytest.raises(ValueError):
        lib.downscale(G, 2)


def case_tile():
    assert lib.tile(G, 2, 3).shape == (8, 15)


def case_hconcat():
    assert lib.hconcat(G, G, G).shape == (4, 15)


def case_vconcat():
    assert lib.vconcat(G, G).shape == (8, 5)


def case_shift():
    assert eq(lib.shift([[1, 2], [3, 4]], -1, 0, fill=9), [[3, 4], [9, 9]])
    assert eq(lib.shift([[1, 2], [3, 4]], 5, 0, fill=0), [[0, 0], [0, 0]])
    assert lib.shift(G, 0, -1)[1, 0] == 1


def case_gravity():
    grid = [[1, 0], [0, 2], [0, 0]]
    assert eq(lib.gravity(grid, "down", background=0), [[0, 0], [0, 0], [1, 2]])
    assert eq(lib.gravity(grid, "up", background=0), [[1, 2], [0, 0], [0, 0]])
    assert eq(lib.gravity([[0, 1, 0, 2]], "left", background=0), [[1, 2, 0, 0]])
    assert eq(lib.gravity([[1, 0, 2, 0]], "right", background=0), [[0, 0, 1, 2]])
    with pytest.raises(ValueError):
        lib.gravity(grid, "sideways")


def case_is_symmetric():
    assert lib.is_symmetric([[1, 2], [2, 1]], "diag")
    assert lib.is_symmetric([[1, 2], [2, 1]], "rot180")
    assert not lib.is_symmetric([[1, 2, 3]], "diag")
    assert lib.is_symmetric([[1, 2], [4, 3]], "rot90") is False
    assert lib.is_symmetric([[5, 5], [5, 5]], "rot90")
    with pytest.raises(ValueError):
        lib.is_symmetric(G, "spiral")


def case_complete_symmetry():
    # Only the half-turn symmetry holds on the known cells, and it restores the corner.
    damaged = [[1, 2, 3], [4, 5, 4], [3, 2, 0]]
    assert eq(lib.complete_symmetry(damaged, unknown=0), [[1, 2, 3], [4, 5, 4], [3, 2, 1]])
    # No symmetry holds, so nothing is filled.
    asymmetric = [[1, 2, 3], [4, 0, 6], [3, 2, 0]]
    assert eq(lib.complete_symmetry(asymmetric, unknown=0), asymmetric)
    square = [[1, 2], [0, 1]]
    assert eq(lib.complete_symmetry(square, unknown=0, kinds=("diag",)), [[1, 2], [2, 1]])
    assert eq(lib.complete_symmetry([[1, 0]], unknown=0, kinds=("ud",)), [[1, 0]])


def case_find_period():
    assert lib.find_period([[1, 2, 3]]) == (1, 3)
    assert lib.find_period([[1, 0, 1, 2]], unknown=0) == (1, 2)


def case_complete_tiling():
    assert eq(lib.complete_tiling([[1, 2, 1, 0, 1, 2]], unknown=0), [[1, 2, 1, 2, 1, 2]])
    assert eq(lib.complete_tiling([[0, 0]], unknown=0), [[0, 0]])


def case_split_grid():
    grid = [[1, 2, 5, 3], [5, 5, 5, 5], [4, 4, 5, 6]]
    blocks = lib.split_grid(grid)
    assert [[b.tolist() for b in row] for row in blocks] == [
        [[[1, 2]], [[3]]],
        [[[4, 4]], [[6]]],
    ]
    assert eq(lib.split_grid([[1, 2]])[0][0], [[1, 2]])
    # Lines of the background color are content, not separators, when a drawn color exists.
    framed = [[0, 0, 0], [8, 8, 8], [0, 1, 0]]
    assert [len(row) for row in lib.split_grid(framed)] == [1, 1]
    assert len(lib.split_grid(framed, separator=0)) == 1


def case_hsplit():
    assert [p.tolist() for p in lib.hsplit([[1, 2, 3, 4, 5, 6]], 3)] == [
        [[1, 2]],
        [[3, 4]],
        [[5, 6]],
    ]
    with pytest.raises(ValueError):
        lib.hsplit(G, 2)


def case_vsplit():
    assert len(lib.vsplit(G, 2)) == 2
    with pytest.raises(ValueError):
        lib.vsplit(G, 3)


def case_draw_line():
    base = np.zeros((3, 4), int)
    assert eq(lib.draw_line(base, (1, 0), (1, 3), 2)[1], [2, 2, 2, 2])
    assert lib.draw_line(base, (2, 0), (0, 2), 3)[0, 2] == 3
    assert lib.draw_line(base, (0, 0), (0, 9), 1)[0].tolist() == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        lib.draw_line(base, (0, 0), (1, 2), 1)


def case_draw_rect():
    filled = lib.draw_rect(np.zeros((4, 4), int), (1, 1, 3, 4), 6, filled=True)
    assert filled.sum() == 6 * 6
    outline = lib.draw_rect(np.zeros((4, 4), int), (0, 0, 4, 4), 1)
    assert outline.sum() == 12


def case_ray():
    base = np.zeros((4, 4), int)
    assert lib.ray(base, (3, 0), (-1, 1), 7).tolist()[0] == [0, 0, 0, 7]
    assert lib.ray(base, (0, 0), (0, 0), 7).sum() == 0
    stop = lib.ray([[0, 0, 4, 0]], (0, 0), (0, 1), 7, until=(4,))
    assert eq(stop, [[0, 7, 4, 0]])


CASES = {name[len("case_") :]: fn for name, fn in globals().items() if name.startswith("case_")}
HELPERS = [name for name in lib.__all__ if name not in {"Object", "doc", "index_text"}]


def test_every_helper_has_a_case():
    assert sorted(CASES) == sorted(HELPERS)


@pytest.mark.parametrize("name", HELPERS)
def test_helper(name):
    CASES[name]()


def test_docstring_examples_run():
    results = doctest.testmod(lib, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0
    assert results.attempted >= len(HELPERS)


@pytest.mark.parametrize("name", [*HELPERS, "doc", "index_text"])
def test_helper_is_documented_and_typed(name):
    fn = getattr(lib, name)
    text = inspect.getdoc(fn)
    assert "Example:" in text and ">>>" in text
    signature = inspect.signature(fn)
    assert signature.return_annotation is not inspect.Signature.empty
    assert all(p.annotation is not inspect.Parameter.empty for p in signature.parameters.values())


def test_all_lists_exactly_the_public_names():
    public = {
        name
        for name, value in vars(lib).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and getattr(value, "__module__", None) == lib.__name__
    }
    assert public == set(lib.__all__)


def test_lib_exposes_no_module_but_numpy():
    modules = {name for name, value in vars(lib).items() if isinstance(value, types.ModuleType)}
    assert modules == {"np"}


def test_lib_imports_only_numpy_and_the_standard_library():
    # numpy-only, so the helpers behave the same on the dev machine (no scipy) and on Kaggle.
    tree = ast.parse(inspect.getsource(lib))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots == {"collections", "dataclasses", "inspect", "numpy"}


PURITY_CALLS = [
    ("recolor", lambda g: lib.recolor(g, {0: 9})),
    ("fill", lambda g: lib.fill(g, g == 0, 9)),
    ("flood_fill", lambda g: lib.flood_fill(g, (0, 0), 9)),
    ("fill_enclosed", lambda g: lib.fill_enclosed(g, 9)),
    ("paint", lambda g: lib.paint(g, [[9, 9]], 0, 0)),
    ("move", lambda g: lib.move(g, g == 1, 1, 1)),
    ("shift", lambda g: lib.shift(g, 1, 1)),
    ("gravity", lambda g: lib.gravity(g)),
    ("complete_symmetry", lambda g: lib.complete_symmetry(g, 0)),
    ("complete_tiling", lambda g: lib.complete_tiling(g, 0)),
    ("draw_line", lambda g: lib.draw_line(g, (0, 0), (0, 3), 9)),
    ("draw_rect", lambda g: lib.draw_rect(g, (0, 0, 2, 2), 9)),
    ("ray", lambda g: lib.ray(g, (0, 0), (1, 1), 9)),
    ("crop", lambda g: lib.crop(g, (0, 0, 2, 2))),
    ("rot90", lib.rot90),
    ("flip_lr", lib.flip_lr),
    ("dihedral", lambda g: lib.dihedral(g, 0)),
    ("split_grid", lambda g: lib.split_grid(g)),
]


@pytest.mark.parametrize(("name", "call"), PURITY_CALLS)
def test_helpers_never_change_their_input(name, call):
    grid = G.copy()
    result = call(grid)
    assert eq(grid, G.tolist()), name
    outputs = result if isinstance(result, list) else [result]
    for out in (o for row in outputs for o in (row if isinstance(row, list) else [row])):
        out[...] = 7
    assert eq(grid, G.tolist()), f"{name} returned a view of its input"


def test_index_text_is_compact_and_complete():
    text = lib.index_text()
    lines = text.splitlines()
    # About 1,500 tokens at about 4 characters per token.
    assert len(text) <= 6_000
    assert len(lines) == len(HELPERS) + 2
    for name in HELPERS:
        assert any(line.startswith(f"{name}(") for line in lines), name
    assert ":func:" not in text and "``" not in text


def test_doc_returns_signature_and_docstring():
    text = lib.doc("objects")
    assert text.startswith("objects(grid, diagonal=False, multicolor=False, background=None)")
    assert "Example:" in text
    with pytest.raises(KeyError):
        lib.doc("np")
