"""Tests of arc26.agent.guards: every rule catches a hostile program; clean programs pass."""

import textwrap

import pytest

from arc26.agent import guards
from arc26.agent.types import AgentTask, GuardFinding, Pair


def rot90(grid):
    return [list(row) for row in zip(*grid[::-1], strict=True)]


IN_0 = [[1, 2, 0, 0], [0, 3, 0, 5], [4, 0, 6, 5], [7, 8, 9, 5]]
IN_1 = [[2, 2, 0, 1], [3, 0, 4, 1], [0, 5, 0, 1], [6, 6, 6, 1]]
# A 5 by 6 input: its sides differ from every square input, for the shape tests.
IN_2 = [
    [3, 1, 4, 1, 5, 9],
    [2, 6, 5, 3, 5, 8],
    [9, 7, 9, 3, 2, 3],
    [8, 4, 6, 2, 6, 4],
    [3, 3, 8, 3, 2, 7],
]
TEST_IN = [[1, 0, 2, 0, 3], [0, 4, 0, 5, 0], [6, 0, 7, 0, 8]]
TASK = AgentTask(
    task_id="guards",
    train=(Pair(IN_0, rot90(IN_0)), Pair(IN_1, rot90(IN_1)), Pair(IN_2, rot90(IN_2))),
    test_inputs=(TEST_IN,),
)
OUT_0 = rot90(IN_0)

CLEAN = """
import numpy as np
from collections import Counter
from arc26.agent import lib


def transform(grid):
    \"\"\"Rotate the grid a quarter turn clockwise.\"\"\"
    if __name__ == "__main__":
        print(lib.rot90.__doc__)
    h, w = grid.shape
    if h > 1 and w > 1:
        counts = Counter(grid.flatten().tolist())
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    return np.rot90(grid, -1)
"""


def rules(findings):
    return [f.rule for f in findings]


def program(body):
    return textwrap.dedent(body)


# Safety policy.


def test_clean_program_has_no_findings():
    assert guards.check_program(CLEAN, TASK) == ()


def test_syntax_error_is_fatal():
    findings = guards.check_policy("def transform(grid):\n    return (")
    assert rules(findings) == ["syntax_error"]
    assert findings[0].fatal and "line" in findings[0].detail


def test_null_byte_is_a_syntax_error():
    assert rules(guards.check_policy("x = 1\0")) == ["syntax_error"]


def test_source_too_large():
    findings = guards.check_policy("x = 1\n" * 10, max_chars=20)
    assert rules(findings) == ["source_too_large"]


HOSTILE_POLICY = [
    ("import os", "forbidden_import"),
    ("from os import path", "forbidden_import"),
    ("import subprocess", "forbidden_import"),
    ("import importlib", "forbidden_import"),
    ("import sys", "forbidden_import"),
    ("from . import sibling", "forbidden_import"),
    ("from numpy import *", "forbidden_import"),
    ("from numpy import load", "forbidden_import"),
    ("import numpy.ctypeslib", "forbidden_import"),
    ("import scipy.io", "forbidden_import"),
    ("from __future__ import annotations", "forbidden_import"),
    ("import arc26.agent.lib", "forbidden_import"),
    ("from arc26 import evaluate", "forbidden_import"),
    ("open('/etc/passwd').read()", "forbidden_builtin"),
    ("eval('1 + 1')", "forbidden_builtin"),
    ("exec('x = 1')", "forbidden_builtin"),
    ("compile('1', 'f', 'eval')", "forbidden_builtin"),
    ("__import__('os')", "forbidden_builtin"),
    ("globals()", "forbidden_builtin"),
    ("vars(np)", "forbidden_builtin"),
    ("locals()", "forbidden_builtin"),
    ("breakpoint()", "forbidden_builtin"),
    ("input()", "forbidden_builtin"),
    ("getattr(np, '__dict__')", "dunder_access"),
    ("().__class__.__bases__[0].__subclasses__()", "dunder_access"),
    ("print(__builtins__)", "dunder_access"),
    ("print(__loader__)", "dunder_access"),
    ("match grid:\n    case object(__class__=c):\n        pass", "dunder_access"),
    ("m = np.ctypeslib", "forbidden_attribute"),
    ("a = np.load('x.npy')", "forbidden_attribute"),
    ("np.zeros(3).tofile('x.bin')", "forbidden_attribute"),
    ("np.savetxt('x.txt', np.zeros(3))", "forbidden_attribute"),
    ("v = np.lib.stride_tricks.as_strided(np.zeros(1), (9,), (8,))", "forbidden_attribute"),
    ("import typing\nmods = typing.sys.modules", "forbidden_attribute"),
    ("import collections\ns = collections._sys", "forbidden_attribute"),
    ("import typing\ntyping.get_type_hints(transform)", "forbidden_attribute"),
    ("import functools\nfunctools.singledispatch(len)", "forbidden_attribute"),
    ("hasattr(np, 'ctypeslib')", "forbidden_attribute"),
    ("np.test()", "forbidden_attribute"),
    # A string that parses as code is checked too: typing evaluates string annotations.
    ("def f(x: \"__import__('os').system('true')\"):\n    pass", "forbidden_builtin"),
    ("NAME = 'np.ctypeslib.ctypes'", "forbidden_attribute"),
]


@pytest.mark.parametrize(("source", "rule"), HOSTILE_POLICY)
def test_policy_catches_hostile_program(source, rule):
    findings = guards.check_policy(source)
    assert rule in rules(findings), findings
    assert all(f.fatal for f in findings)
    assert all(f.rule in guards.SAFETY_RULES for f in findings)


ALLOWED_POLICY = [
    "import numpy as np",
    "import numpy.linalg",
    "from numpy.lib.stride_tricks import sliding_window_view",
    "from arc26.agent import lib",
    "import arc26.agent.lib as L",
    "from arc26.agent.lib import objects, rot90",
    "from collections import Counter, deque",
    "import itertools, math, functools, copy",
    "from typing import List, Tuple",
    "import scipy.ndimage as ndi",
    "from scipy import ndimage",
    "x = np.select([np.zeros(2, bool)], [np.ones(2)])",
    "if __name__ == '__main__':\n    pass",
    "class A:\n    def __init__(self):\n        super().__init__()",
    "print(lib.rot90.__doc__, lib.rot90.__name__)",
    "def transform(input):\n    return input",
    "def compile(rows):\n    return rows\ncompile([1])",
    "s = 'Rotate the grid (np.rot90) and return it.'",
    "s = 'grid.shape'",
    "def f(x: 'np.ndarray') -> 'list[list[int]]':\n    return x",
]


@pytest.mark.parametrize("source", ALLOWED_POLICY)
def test_policy_allows_ordinary_code(source):
    assert guards.check_policy(source) == ()


def test_one_finding_per_hit_with_line_numbers():
    findings = guards.check_policy("import os\nx = 1\nopen('f')\n")
    assert rules(findings) == ["forbidden_import", "forbidden_builtin"]
    assert "line 1" in findings[0].detail and "line 3" in findings[1].detail


def test_module_and_attribute_rules():
    assert guards.is_allowed_module("numpy.linalg")
    assert guards.is_allowed_module("arc26.agent.lib")
    assert not guards.is_allowed_module("arc26.agent")
    assert not guards.is_allowed_module("arc26.agent.librarian")
    assert not guards.is_allowed_module("numpy.testing")
    assert not guards.is_allowed_module("scipy.io.wavfile")
    assert not guards.is_allowed_module("numpy..linalg")
    assert guards.is_forbidden_attribute("__class__")
    assert guards.is_forbidden_attribute("tofile")
    assert not guards.is_forbidden_attribute("__doc__")
    assert not guards.is_forbidden_attribute("shape")


# Hardcoding guards.


def test_literal_whole_output_is_fatal():
    source = program(
        f"""
        def transform(grid):
            return {OUT_0!r}
        """
    )
    findings = guards.check_program(source, TASK)
    assert "literal_output" in rules(findings)
    assert all(f.fatal for f in findings if f.rule == "literal_output")


def test_a_partial_block_of_an_output_is_demoted_not_fatal():
    block = [row[1:4] for row in rot90(IN_2)[0:3]]
    padded = [[0] + row + [0] for row in block]
    source = program(
        f"""
        import numpy as np
        PATCH = {padded!r}

        def transform(grid):
            return np.rot90(grid, -1)
        """
    )
    findings = guards.check_program(source, TASK)
    assert rules(findings) == ["partial_output_literal"]
    assert not findings[0].fatal


CROSS_TASK = AgentTask(
    "cross",
    train=(
        Pair(
            [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 4, 0, 0, 0], [4, 2, 4, 0, 0], [0, 4, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ),
        Pair(
            [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0], [0, 0, 0, 4, 2, 4], [0, 0, 0, 0, 4, 0]],
        ),
    ),
    test_inputs=([[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]],),
)
# Stamps a fixed 3 by 3 cross at every red cell: the cross is a block of every output.
CROSS_STAMP = """
import numpy as np

CROSS = np.array([[0, 4, 0], [4, 2, 4], [0, 4, 0]])


def transform(grid):
    out = grid.copy()
    height, width = grid.shape
    for r, c in zip(*np.nonzero(grid == 2)):
        for dr in range(3):
            for dc in range(3):
                rr, cc = r + dr - 1, c + dc - 1
                if CROSS[dr, dc] and 0 <= rr < height and 0 <= cc < width:
                    out[rr, cc] = CROSS[dr, dc]
    return out
"""


def test_a_stencil_program_is_demoted_not_rejected():
    findings = guards.check_program(CROSS_STAMP, CROSS_TASK)
    assert rules(findings) == ["partial_output_literal"]
    assert not any(f.fatal for f in findings)


def test_a_whole_output_literal_shared_by_every_demonstration_is_demoted():
    shared = [[1, 2], [3, 4]]
    task = AgentTask("c", (Pair(IN_0, shared), Pair(IN_1, shared)), (TEST_IN,))
    source = f"def transform(grid):\n    return {shared!r}\n"
    findings = guards.check_program(source, task)
    assert rules(findings) == ["constant_output_literal"]
    assert not findings[0].fatal
    flat = [7, 1, 7, 1, 7, 1, 7, 1, 7]
    task = AgentTask("f", (Pair(IN_0, [flat[0:3], flat[3:6], flat[6:9]]),) * 2, (TEST_IN,))
    source = (
        f"import numpy as np\ndef transform(grid):\n    return np.array({flat!r}).reshape(3, 3)\n"
    )
    assert rules(guards.check_program(source, task)) == ["constant_output_literal"]
    # One demonstration is no evidence that the output is constant.
    single = AgentTask("s", (Pair(IN_0, shared),), (TEST_IN,))
    findings = guards.check_program(f"def transform(grid):\n    return {shared!r}\n", single)
    assert rules(findings) == ["literal_output"] and findings[0].fatal


def test_flat_literal_of_an_output_is_fatal():
    flat = [c for row in OUT_0 for c in row]
    source = program(
        f"""
        import numpy as np

        def transform(grid):
            return np.array({flat!r}).reshape(4, 4)
        """
    )
    assert "literal_output" in rules(guards.check_program(source, TASK))


def test_a_padded_output_literal_is_fatal():
    # The output inside extra rows and columns, cropped in code, is still the whole output.
    padded = [[0] * 6] + [[0, *row, 0] for row in OUT_0] + [[0] * 6]
    source = program(
        f"""
        import numpy as np
        PADDED = {padded!r}

        def transform(grid):
            return np.array(PADDED)[1:5, 1:5]
        """
    )
    findings = guards.check_program(source, TASK)
    assert rules(findings) == ["literal_output"]
    assert findings[0].fatal and "contains demonstration output 0" in findings[0].detail


@pytest.mark.parametrize("order", ["row", "column"])
def test_a_flat_literal_with_extra_values_is_fatal(order):
    flat = (
        [c for row in OUT_0 for c in row]
        if order == "row"
        else [row[c] for c in range(4) for row in OUT_0]
    )
    source = program(
        f"""
        import numpy as np

        def transform(grid):
            return np.array({[5, 5, *flat, 7]!r})[2:18].reshape(4, 4)
        """
    )
    findings = guards.check_program(source, TASK)
    assert "literal_output" in rules(findings)
    assert all(f.fatal for f in findings if f.rule == "literal_output")


def test_a_padded_constant_output_literal_is_demoted():
    shared = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    task = AgentTask("c", (Pair(IN_0, shared), Pair(IN_1, shared)), (TEST_IN,))
    padded = [[0] * 5] + [[0, *row, 0] for row in shared] + [[0] * 5]
    source = (
        f"import numpy as np\ndef transform(grid):\n    return np.array({padded!r})[1:4, 1:4]\n"
    )
    findings = guards.check_program(source, task)
    assert rules(findings) == ["constant_output_literal"]
    assert not findings[0].fatal


def test_small_outputs_count_only_when_a_literal_equals_them():
    # A 2 by 2 output sits inside a 0/1 kernel by chance; only an equal literal is hardcoding.
    task = AgentTask(
        "small", (Pair(IN_0, [[1, 0], [0, 1]]), Pair(IN_1, [[0, 1], [1, 0]])), (TEST_IN,)
    )
    kernel = "KERNEL = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]\ndef transform(grid):\n    return grid\n"
    assert guards.check_program(kernel, task) == ()
    equal = "def transform(grid):\n    return [[1, 0], [0, 1]]\n"
    assert rules(guards.check_program(equal, task)) == ["literal_output"]


def test_kernels_and_uniform_literals_are_not_output_literals():
    # The output holds a 0/1 plus and a uniform block; literals shaped like them are stencils.
    output = [[0, 1, 0, 5, 5, 5], [1, 1, 1, 5, 5, 5], [0, 1, 0, 5, 5, 5]]
    task = AgentTask("k", (Pair([[0] * 6] * 3, output),), ([[0] * 6] * 3,))
    source = program(
        """
        import numpy as np
        PLUS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        FIVE = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]

        def transform(grid):
            return grid
        """
    )
    found = rules(guards.check_program(source, task))
    assert "literal_output" not in found and "partial_output_literal" not in found


HOSTILE_INPUT_COMPARE = [
    f"""
    def transform(grid):
        if grid.tolist() == {IN_0!r}:
            return grid
        return grid
    """,
    f"""
    import numpy as np

    def transform(grid):
        if np.array_equal(grid, {IN_1!r}):
            return grid
        return grid
    """,
    f"""
    KNOWN = {IN_0!r}

    def transform(grid):
        g = grid.copy()
        return g if g.tolist() != KNOWN else g
    """,
    # Recognizing the input via an embedded copy, without any comparison in sight.
    f"""
    TABLE = [({IN_2!r}, 1)]

    def transform(grid):
        return grid
    """,
]


@pytest.mark.parametrize("source", HOSTILE_INPUT_COMPARE)
def test_input_literal_compare_is_fatal(source):
    findings = guards.check_program(program(source), TASK)
    assert "input_literal_compare" in rules(findings), findings
    assert all(f.fatal for f in findings if f.rule == "input_literal_compare")


HOSTILE_HASH = [
    """
    TABLE = {}

    def transform(grid):
        return TABLE[grid.tobytes()]
    """,
    """
    def transform(grid):
        key = hash(grid.tobytes())
        return grid if key == 12345 else grid
    """,
    """
    def transform(grid):
        if str(grid) == "[[1 2]]":
            return grid
        return grid
    """,
    """
    KNOWN = {}

    def transform(grid):
        return KNOWN.get(tuple(map(tuple, grid)), grid)
    """,
    """
    KNOWN = set()

    def transform(grid):
        g = grid.tolist()
        if tuple(map(tuple, g)) in KNOWN:
            return grid
        return grid
    """,
    """
    LOOKUP = {}

    def transform(grid):
        return LOOKUP[f"{grid}"]
    """,
    """
    def helper(g):
        return {"x": 1}[repr(g)]

    def transform(grid):
        return helper(grid)
    """,
]


@pytest.mark.parametrize("source", HOSTILE_HASH)
def test_input_hash_key_is_fatal(source):
    findings = guards.check_program(program(source), TASK)
    assert "input_hash_key" in rules(findings), findings
    assert all(f.fatal for f in findings if f.rule == "input_hash_key")


def test_legitimate_uses_of_input_forms_pass():
    source = program(
        """
        from collections import Counter
        import numpy as np

        def transform(grid):
            out = np.rot90(grid, -1)
            print(str(grid), f"{grid}")
            seen = Counter()
            for row in grid:
                seen[tuple(row)] += 1
            tiles = {grid[0:2, 0:2].tobytes(): 1}
            if tuple(map(tuple, out)) == tuple(map(tuple, grid)):
                return grid
            return out
        """
    )
    assert guards.check_program(source, TASK) == ()


HOSTILE_SHAPE = [
    """
    def transform(grid):
        if grid.shape == (4, 4):
            return grid
        return grid
    """,
    """
    def transform(grid):
        h, w = grid.shape
        if w == 6:
            return grid
        return grid
    """,
    """
    def transform(grid):
        rows, cols = len(grid), len(grid[0])
        return grid if rows in (5, 7) else grid
    """,
    """
    def transform(grid):
        match grid.shape:
            case (5, 6):
                return grid
        return grid
    """,
    """
    SIZES = {(4, 4): 1, (5, 6): 2}

    def transform(grid):
        mode = SIZES[grid.shape]
        return grid
    """,
    """
    def is_big(g):
        return g.shape[0] == 5

    def transform(grid):
        return grid if is_big(grid) else grid
    """,
    """
    def transform(grid):
        return grid if grid.size == 30 else grid
    """,
]


@pytest.mark.parametrize("source", HOSTILE_SHAPE)
def test_shape_dispatch_is_demoted_not_fatal(source):
    findings = guards.check_program(program(source), TASK)
    hits = [f for f in findings if f.rule == "shape_dispatch"]
    assert hits, findings
    assert not any(f.fatal for f in hits)


def test_shape_checks_that_are_not_dispatch_pass():
    source = program(
        """
        import numpy as np
        from arc26.agent import lib

        def transform(grid):
            if grid.shape == (13, 17):
                return grid
            for obj in lib.objects(grid):
                if obj.patch.shape == (4, 4):
                    return obj.patch
            h, w = grid.shape
            if h > 2 and w != h:
                return np.rot90(grid, -1)
            return grid
        """
    )
    assert guards.check_program(source, TASK) == ()


def test_check_program_skips_hardcoding_when_policy_fails():
    source = f"import os\ndef transform(grid):\n    return {OUT_0!r}\n"
    assert rules(guards.check_program(source, TASK)) == ["forbidden_import"]


# Test-output checks.


def test_output_equals_input_is_demoted_unless_a_demo_is_identity():
    findings = guards.check_outputs(TASK, [TEST_IN])
    assert rules(findings) == ["output_equals_input"]
    assert not findings[0].fatal
    identity = AgentTask("i", (Pair(IN_0, IN_0), Pair(IN_1, rot90(IN_1))), (TEST_IN,))
    assert "output_equals_input" not in rules(guards.check_outputs(identity, [TEST_IN]))


def test_output_equals_demo_is_demoted_unless_outputs_are_constant():
    assert rules(guards.check_outputs(TASK, [OUT_0])) == ["output_equals_demo"]
    constant = AgentTask("c", (Pair(IN_0, [[3]]), Pair(IN_1, [[3]])), (TEST_IN,))
    assert guards.check_outputs(constant, [[[3]]]) == ()


def test_novel_colors_are_demoted():
    task = AgentTask("n", (Pair([[1, 2]], [[2, 1]]),), ([[3, 4]],))
    findings = guards.check_outputs(task, [[[1, 7]]])
    assert rules(findings) == ["novel_colors"]
    assert "[7]" in findings[0].detail
    assert guards.check_outputs(task, [[[1, 2, 3, 4]]]) == ()


def test_failed_test_outputs_are_skipped():
    assert guards.check_outputs(TASK, [None]) == ()


# Rule bookkeeping and transform detection.


def test_every_rule_is_documented_and_classified():
    emitted = {
        "syntax_error",
        "source_too_large",
        "forbidden_import",
        "forbidden_builtin",
        "dunder_access",
        "forbidden_attribute",
        "literal_output",
        "constant_output_literal",
        "partial_output_literal",
        "input_literal_compare",
        "input_hash_key",
        "shape_dispatch",
        "output_equals_input",
        "output_equals_demo",
        "novel_colors",
    }
    assert set(guards.RULES) == emitted
    assert guards.SAFETY_RULES < guards.FATAL_RULES <= emitted
    assert "shape_dispatch" not in guards.FATAL_RULES
    assert "literal_output" in guards.FATAL_RULES
    assert not {"constant_output_literal", "partial_output_literal"} & guards.FATAL_RULES
    assert GuardFinding("x", "y", True).fatal


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def transform(grid):\n    return grid", True),
        ("transform = lambda grid: grid", True),
        ("from arc26.agent.lib import rot90 as transform", True),
        ("transform: object = None", True),
        ("def helper():\n    def transform(g):\n        return g", False),
        ("x = transform(train[0][0])", False),
        ("def transform(grid):\n    return (", False),
    ],
)
def test_defines_transform(source, expected):
    assert guards.defines_transform(source) is expected
