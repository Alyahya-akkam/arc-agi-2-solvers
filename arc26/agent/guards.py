"""Safety policy and hardcoding guards for the programs the agent writes (P6.1, ADR-020).

Why: a program counts only if it reproduces every demonstration pair, and that test means
something only when two things hold: the program could reach nothing but numpy and the helper
library (the safety policy), and it did not pass by embedding or keying on the demonstrations
(the hardcoding guards). The rules follow ``docs/research/agent_library_design.md`` section 4;
``docs/modules/agent_sandbox.md`` explains each one.

Every hit is one :class:`~arc26.agent.types.GuardFinding` whose ``rule`` is a stable name:

- Safety, fatal, and the program is never executed: ``syntax_error``, ``source_too_large``,
  ``forbidden_import``, ``forbidden_builtin``, ``dunder_access``, ``forbidden_attribute``.
- Hardcoding, fatal, but the program still runs so its report shows what it did:
  ``literal_output``, ``input_literal_compare``, ``input_hash_key``.
- Demotion, non-fatal (ranks the program lower): ``constant_output_literal``,
  ``partial_output_literal``, ``shape_dispatch``, ``output_equals_input``, ``output_equals_demo``,
  ``novel_colors``.

The static policy is the first of several layers: the sandbox processes also drop the forbidden
builtins, check imports and dynamic attribute names at run time, deny dangerous operations with an
audit hook, and run under resource limits (:mod:`arc26.agent.sandbox`, :mod:`arc26.agent.verify`).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType

from arc26.agent.types import AgentTask, GuardFinding
from arc26.data import Grid

# Section 4 of the design note: imports are allowed only from these modules and their submodules.
ALLOWED_MODULES: frozenset[str] = frozenset({
    "numpy", "scipy", "math", "itertools", "collections", "functools", "copy", "typing",
    "arc26.agent.lib",
})  # fmt: skip

# Builtins a program may not name; the sandbox also removes them from the program's builtins.
FORBIDDEN_BUILTINS: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "__import__", "globals", "vars", "locals", "input",
    "breakpoint", "help", "exit", "quit",
})  # fmt: skip

# Dunder names a program may use: `if __name__ == ...`, docstrings, and `super().__init__()`.
ALLOWED_DUNDERS: frozenset[str] = frozenset({"__name__", "__doc__", "__init__"})

# Attribute names that lead from an allowed module to files, processes, frames, native code, or
# eval of strings (typing.sys, collections._sys, np.ctypeslib, np.load, typing.get_type_hints, ...).
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
    # Modules reachable as attributes of allowed modules.
    "os", "sys", "_sys", "posix", "nt", "io", "_io", "codecs", "shutil", "pathlib", "glob",
    "tempfile", "subprocess", "multiprocessing", "threading", "_thread", "socket", "selectors",
    "mmap", "resource", "ctypes", "_ctypes", "ctypeslib", "importlib", "builtins", "types",
    "inspect", "gc", "pickle", "marshal", "copyreg", "operator", "linecache", "tokenize",
    "traceback", "runpy", "codeop", "pdb", "atexit", "dataclasses", "site", "sysconfig",
    "platform", "asyncio", "concurrent", "urllib", "http",
    # numpy and scipy entry points to files, tests, downloads, or raw memory.
    "load", "loads", "loadtxt", "genfromtxt", "recfromtxt", "recfromcsv", "fromfile",
    "fromtextfile", "fromregex", "memmap", "open_memmap", "save", "savez", "savez_compressed",
    "savetxt", "tofile", "dump", "dumps", "DataSource", "_datasource", "npyio", "f2py",
    "testing", "test", "_pytesttester", "PytestTester", "show_config", "datasets", "_lib",
    "LowLevelCallable", "as_strided",
    # Processes and file handles.
    "open", "fdopen", "popen", "system", "spawn", "fork", "kill",
    # Frames and code objects.
    "_getframe", "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "gi_frame",
    "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "tb_frame", "tb_next", "co_code",
    "co_consts",
    # Library functions that eval strings (forward references) or exec generated code.
    "get_type_hints", "ForwardRef", "_eval_type", "NamedTuple", "TypedDict", "singledispatch",
    "singledispatchmethod", "_create_fn",
})  # fmt: skip

MAX_SOURCE_CHARS = 100_000

RULES: Mapping[str, str] = MappingProxyType(
    {
        "syntax_error": "the source does not parse",
        "source_too_large": "the source is longer than the size limit",
        "forbidden_import": "an import outside the allowlist, a relative or star import, or one "
        "that binds the arc26 package",
        "forbidden_builtin": "a use of open, eval, exec, compile, __import__, globals, vars, "
        "locals, input, breakpoint, help, exit, or quit",
        "dunder_access": "access to a dunder attribute or name other than __name__, __doc__, "
        "__init__",
        "forbidden_attribute": "access to an attribute that reaches files, processes, frames, "
        "native code, or eval (np.load, tofile, ctypeslib, sys, get_type_hints, ...)",
        "literal_output": "a literal equal to a whole demonstration output, or one containing an "
        "output of at least 9 cells as a block (or its flattening as a run)",
        "constant_output_literal": "a literal equal to the output that every demonstration "
        "shares, which may be the rule itself",
        "partial_output_literal": "a literal holding a 3 by 3 block of a demonstration output, "
        "for example a stencil the program stamps",
        "input_literal_compare": "a literal equal to a task input, or a comparison of the whole "
        "input with a grid literal",
        "input_hash_key": "a lookup or comparison keyed on the input's hash, bytes, string, or "
        "tuple form",
        "shape_dispatch": "a comparison of the input's exact dimensions with a demonstration or "
        "test input size",
        "output_equals_input": "a test output equal to its test input while no demonstration "
        "output equals its input",
        "output_equals_demo": "a test output equal to a demonstration output",
        "novel_colors": "a test output color seen neither in demonstration outputs nor in the "
        "test input",
    }
)

# A finding of these rules means the program is never executed.
SAFETY_RULES: frozenset[str] = frozenset({
    "syntax_error", "source_too_large", "forbidden_import", "forbidden_builtin",
    "dunder_access", "forbidden_attribute",
})  # fmt: skip
FATAL_RULES: frozenset[str] = SAFETY_RULES | {
    "literal_output",
    "input_literal_compare",
    "input_hash_key",
}

_ATTR_BUILTINS = frozenset({"getattr", "setattr", "delattr", "hasattr"})
# Calls that keep the whole input (copies, conversions, transposes) for the input-form rules.
_WHOLE_FUNCS = frozenset(
    {"array", "asarray", "asanyarray", "ascontiguousarray", "copy", "deepcopy", "list", "tuple"}
)
_WHOLE_METHODS = frozenset({"copy", "tolist", "astype", "flatten", "ravel", "view"})
_EQUALITY_CALLS = frozenset({"array_equal", "array_equiv", "allclose"})
_BYTES_METHODS = frozenset({"tobytes", "tostring"})
_STRING_CALLS = frozenset({"str", "repr", "format", "array2string", "array_str", "array_repr"})
_KEY_METHODS = frozenset({"get", "setdefault", "pop", "count", "index"})
_MAX_PAYLOAD_CHARS = 2_000


def is_dunder(name: str) -> bool:
    """Return whether a name has the ``__x__`` form.

    Args:
        name: An identifier.

    Returns:
        True for names such as ``__class__``.
    """
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def is_forbidden_attribute(name: str) -> bool:
    """Return whether a program may not access an attribute of this name.

    Why: the sandbox's run-time ``getattr`` and import checks use the same rule as the static
    policy, so a name built at run time is refused like a literal one.

    Args:
        name: An attribute name.

    Returns:
        True for dunder names outside :data:`ALLOWED_DUNDERS` and for
        :data:`FORBIDDEN_ATTRIBUTES`.
    """
    return (is_dunder(name) and name not in ALLOWED_DUNDERS) or name in FORBIDDEN_ATTRIBUTES


def is_allowed_module(name: str) -> bool:
    """Return whether a program may import a module.

    Args:
        name: A dotted module name, for example ``"scipy.ndimage"``.

    Returns:
        True when the module is in :data:`ALLOWED_MODULES` or below one, and no component is a
        dunder or a forbidden attribute name (``numpy.ctypeslib``, ``scipy.io``, ...).
    """
    parts = name.split(".")
    if not all(parts) or any(is_dunder(p) or p in FORBIDDEN_ATTRIBUTES for p in parts):
        return False
    return any(name == m or name.startswith(m + ".") for m in ALLOWED_MODULES)


def check_policy(source: str, *, max_chars: int = MAX_SOURCE_CHARS) -> tuple[GuardFinding, ...]:
    """Check a program or REPL cell against the safety policy.

    Why: a program that fails the policy must not run at all; the verifier and the REPL call this
    in the parent process, before any model-written code reaches a sandbox.

    Args:
        source: Python source.
        max_chars: The size limit; longer sources are rejected unparsed.

    Returns:
        One fatal finding per hit, empty when the source passes.
    """
    if len(source) > max_chars:
        detail = f"{len(source)} characters; the limit is {max_chars}"
        return (GuardFinding("source_too_large", detail, True),)
    tree, error = _parse(source)
    if tree is None:
        return (GuardFinding("syntax_error", error, True),)
    return tuple(_policy_findings(tree, _bound_names(tree), outer_line=None))


def check_hardcoding(source: str, task: AgentTask) -> tuple[GuardFinding, ...]:
    """Check a program for hardcoded demonstrations and for dispatch on the input's identity.

    Why: a program can reproduce every demonstration without inducing the rule, by embedding the
    outputs, by recognizing the inputs (a literal comparison, a hash, bytes, or string key), or by
    branching on their exact sizes. The first two are fatal; size dispatch only demotes.

    Args:
        source: Python source that defines ``transform``.
        task: The task the program is verified on.

    Returns:
        One finding per hit (``literal_output``, ``input_literal_compare``, ``input_hash_key``
        fatal; ``shape_dispatch`` not), empty when the source does not parse.
    """
    tree, _ = _parse(source)
    if tree is None:
        return ()
    outputs = [pair.output for pair in task.train]
    inputs = [pair.input for pair in task.train] + list(task.test_inputs)
    findings = list(_literal_findings(tree, outputs, inputs))
    literal_names = _literal_grid_names(tree)
    module_dicts = _module_dicts(tree)
    for function, aliases in _tracked_functions(tree).items():
        findings.extend(_compare_findings(function, aliases, literal_names))
        findings.extend(_hash_findings(function, aliases))
        findings.extend(_shape_findings(function, aliases, inputs, module_dicts))
    return tuple(_dedupe(findings))


def check_program(source: str, task: AgentTask) -> tuple[GuardFinding, ...]:
    """Run the safety policy and, when it passes, the hardcoding guards.

    Args:
        source: Python source that defines ``transform``.
        task: The task the program is verified on.

    Returns:
        The policy findings, or when there are none, the hardcoding findings.
    """
    findings = check_policy(source)
    if findings:
        return findings
    return check_hardcoding(source, task)


def check_outputs(task: AgentTask, test_outputs: Sequence[Grid | None]) -> tuple[GuardFinding, ...]:
    """Check a program's test outputs for signs of a wrong rule (all non-fatal).

    Why: section 4 of the design note demotes test outputs that copy the test input when no
    demonstration does, that copy a demonstration output, or that use colors from nowhere.

    Args:
        task: The task.
        test_outputs: The program's output on each test input, None where it failed.

    Returns:
        One ``output_equals_input``, ``output_equals_demo``, or ``novel_colors`` finding per hit.
    """
    demo_outputs = [pair.output for pair in task.train]
    identity_task = any(pair.input == pair.output for pair in task.train)
    constant_task = _constant(demo_outputs)
    output_colors = {c for grid in demo_outputs for row in grid for c in row}
    findings = []
    for index, output in enumerate(test_outputs):
        if output is None:
            continue
        test_input = task.test_inputs[index]
        if not identity_task and output == test_input:
            detail = f"test output {index} equals test input {index}"
            findings.append(GuardFinding("output_equals_input", detail, False))
        if not constant_task and output in demo_outputs:
            detail = f"test output {index} equals demonstration output {demo_outputs.index(output)}"
            findings.append(GuardFinding("output_equals_demo", detail, False))
        input_colors = {c for row in test_input for c in row}
        novel = sorted({c for row in output for c in row} - output_colors - input_colors)
        if novel:
            detail = f"test output {index} uses colors {novel}, absent from the demonstration "
            detail += "outputs and the test input"
            findings.append(GuardFinding("novel_colors", detail, False))
    return tuple(findings)


def defines_transform(source: str) -> bool:
    """Return whether a source defines a top-level ``transform``.

    Args:
        source: Python source of a REPL cell or program.

    Returns:
        True when a module-level statement binds ``transform`` (a ``def``, an assignment, or an
        import alias); False otherwise and when the source does not parse.
    """
    tree, _ = _parse(source)
    if tree is None:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "transform":
            return True
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if any(isinstance(n, ast.Name) and n.id == "transform" for n in ast.walk(target)):
                    return True
        if isinstance(node, ast.Import | ast.ImportFrom) and any(
            (a.asname or a.name) == "transform" for a in node.names
        ):
            return True
    return False


def _parse(source: str) -> tuple[ast.Module | None, str]:
    try:
        return ast.parse(source, filename="<program>", mode="exec"), ""
    except SyntaxError as exc:
        return None, f"line {exc.lineno}: {exc.msg}"
    except (ValueError, RecursionError, MemoryError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _bound_names(tree: ast.AST) -> frozenset[str]:
    """Names the program binds itself; a forbidden builtin name it binds refers to its own."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            names.update(node.names)
    return frozenset(names)


def _where(node: ast.AST, outer_line: int | None) -> str:
    if outer_line is not None:
        return f"line {outer_line} (inside a string)"
    return f"line {getattr(node, 'lineno', '?')}"


def _policy_findings(
    tree: ast.AST, bound: frozenset[str], outer_line: int | None
) -> Iterator[GuardFinding]:
    for node in ast.walk(tree):
        where = _where(node, outer_line)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not is_allowed_module(alias.name):
                    detail = f"{where}: import {alias.name} (allowed: numpy, scipy, math, "
                    detail += "itertools, collections, functools, copy, typing, arc26.agent.lib)"
                    yield GuardFinding("forbidden_import", detail, True)
                elif alias.asname is None and not is_allowed_module(alias.name.split(".")[0]):
                    detail = f"{where}: `import {alias.name}` binds the package "
                    detail += f"{alias.name.split('.')[0]}; write `from arc26.agent import lib`"
                    yield GuardFinding("forbidden_import", detail, True)
        elif isinstance(node, ast.ImportFrom):
            yield from _import_from_findings(node, where)
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_BUILTINS and (node.id == "__import__" or node.id not in bound):
                if isinstance(node.ctx, ast.Load):
                    yield GuardFinding("forbidden_builtin", f"{where}: {node.id}", True)
            elif (
                is_dunder(node.id)
                and node.id not in ALLOWED_DUNDERS
                and not isinstance(node.ctx, ast.Store)
            ):
                yield GuardFinding("dunder_access", f"{where}: name {node.id}", True)
        elif isinstance(node, ast.Attribute):
            yield from _attribute_findings(node.attr, where)
        elif isinstance(node, ast.MatchClass):
            for name in node.kwd_attrs:
                yield from _attribute_findings(name, where)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Name)
                and func.id in _ATTR_BUILTINS
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                yield from _attribute_findings(node.args[1].value, where)
        elif (
            outer_line is None
            and isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (payload := _string_payload(node.value)) is not None
        ):
            # A string that parses as code can reach eval through library calls (forward
            # references in typing); it is held to the same policy as the program.
            yield from _policy_findings(payload, frozenset(), outer_line=node.lineno)


def _import_from_findings(node: ast.ImportFrom, where: str) -> Iterator[GuardFinding]:
    module = node.module or ""
    if node.level:
        yield GuardFinding("forbidden_import", f"{where}: relative import", True)
        return
    for alias in node.names:
        if alias.name == "*":
            yield GuardFinding("forbidden_import", f"{where}: from {module} import *", True)
        elif is_allowed_module(module):
            if is_forbidden_attribute(alias.name) or is_dunder(alias.name):
                detail = f"{where}: from {module} import {alias.name}"
                yield GuardFinding("forbidden_import", detail, True)
        elif not is_allowed_module(f"{module}.{alias.name}"):
            detail = f"{where}: from {module} import {alias.name}"
            yield GuardFinding("forbidden_import", detail, True)


def _attribute_findings(name: str, where: str) -> Iterator[GuardFinding]:
    if is_dunder(name) and name not in ALLOWED_DUNDERS:
        yield GuardFinding("dunder_access", f"{where}: attribute {name}", True)
    elif name in FORBIDDEN_ATTRIBUTES:
        yield GuardFinding("forbidden_attribute", f"{where}: attribute {name}", True)


def _string_payload(text: str) -> ast.AST | None:
    if len(text) > _MAX_PAYLOAD_CHARS:
        return None
    for mode in ("eval", "exec"):
        try:
            tree = ast.parse(text, mode=mode)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        code_like = (ast.Call, ast.Attribute, ast.Subscript, ast.Import, ast.ImportFrom)
        if any(isinstance(n, code_like) for n in ast.walk(tree)):
            return tree
        return None
    return None


def _int_value(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _literal_vector(node: ast.AST) -> list[int] | None:
    """A non-empty list or tuple of int constants."""
    if not isinstance(node, ast.List | ast.Tuple) or not node.elts:
        return None
    values = []
    for element in node.elts:
        value = _int_value(element)
        if value is None:
            return None
        values.append(value)
    return values


def _literal_matrix(node: ast.AST) -> list[list[int]] | None:
    """A rectangular list or tuple of lists or tuples of int constants."""
    if not isinstance(node, ast.List | ast.Tuple) or not node.elts:
        return None
    rows = []
    for element in node.elts:
        row = _literal_vector(element)
        if row is None:
            return None
        rows.append(row)
    return rows if len({len(row) for row in rows}) == 1 else None


def _is_color_grid(rows: list[list[int]]) -> bool:
    return all(0 <= v <= 9 for row in rows for v in row)


def _uniform(grid: Grid) -> bool:
    return len({c for row in grid for c in row}) <= 1


def _blocks(grid: Grid) -> set[tuple[int, ...]]:
    """The 3 by 3 windows of a grid that carry information: neither uniform nor 0/1-valued.

    Why: uniform windows and 0/1 windows are what masks, kernels, and stencils look like, so they
    are not taken as embedded output.
    """
    found = set()
    for r in range(len(grid) - 2):
        for c in range(len(grid[0]) - 2):
            window = tuple(grid[r + i][c + j] for i in range(3) for j in range(3))
            values = set(window)
            if len(values) > 1 and not values <= {0, 1}:
                found.add(window)
    return found


def _constant(outputs: Sequence[Grid]) -> bool:
    """Whether at least two demonstrations share one output, which is then the rule's answer."""
    return len(outputs) >= 2 and all(o == outputs[0] for o in outputs)


def _whole_output_finding(detail: str, constant: bool) -> GuardFinding:
    if constant:
        detail += ", which every demonstration shares"
        return GuardFinding("constant_output_literal", detail, False)
    return GuardFinding("literal_output", detail, True)


# Outputs this large are found inside a longer literal too (padded, then cropped or reshaped in
# code); a smaller output could sit inside a kernel or a color table by chance, so only a literal
# equal to it counts. The floor is the 3 by 3 block of the partial rule.
_CONTAINED_MIN_CELLS = 9


def _literal_findings(
    tree: ast.AST, outputs: Sequence[Grid], inputs: Sequence[Grid]
) -> Iterator[GuardFinding]:
    constant = _constant(outputs)
    output_blocks = [_blocks(o) for o in outputs]
    flat_outputs = [
        (k, [c for row in o for c in row], [row[c] for c in range(len(o[0])) for row in o])
        for k, o in enumerate(outputs)
        if not _uniform(o) and len(o) * len(o[0]) >= _CONTAINED_MIN_CELLS
    ]
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        matrix = _literal_matrix(node)
        if matrix is not None and _is_color_grid(matrix) and not _uniform(matrix):
            finding = _output_literal_finding(matrix, outputs, output_blocks, constant, line)
            if finding is not None:
                yield finding
            for k, grid in enumerate(inputs):
                if matrix == grid:
                    detail = f"line {line}: a literal equals task input {k} (demonstrations first)"
                    yield GuardFinding("input_literal_compare", detail, True)
                    break
            continue
        vector = _literal_vector(node)
        if vector is not None and len(vector) >= _CONTAINED_MIN_CELLS:
            for k, row_major, column_major in flat_outputs:
                if _contains_run(vector, row_major) or _contains_run(vector, column_major):
                    detail = f"line {line}: a flat literal holds demonstration output {k}"
                    yield _whole_output_finding(detail, constant)
                    break


def _output_literal_finding(
    matrix: list[list[int]],
    outputs: Sequence[Grid],
    output_blocks: Sequence[set[tuple[int, ...]]],
    constant: bool,
    line: object,
) -> GuardFinding | None:
    for k, output in enumerate(outputs):
        if matrix == output:
            return _whole_output_finding(
                f"line {line}: a literal equals demonstration output {k}", constant
            )
    for k, output in enumerate(outputs):
        cells = len(output) * len(output[0])
        if cells >= _CONTAINED_MIN_CELLS and not _uniform(output) and _contains(matrix, output):
            detail = f"line {line}: a literal contains demonstration output {k}"
            return _whole_output_finding(detail, constant)
    if len(matrix) >= 3 and len(matrix[0]) >= 3:
        blocks = _blocks(matrix)
        for k, found in enumerate(output_blocks):
            if blocks & found:
                detail = f"line {line}: a literal holds a 3 by 3 block of demonstration output {k}"
                return GuardFinding("partial_output_literal", detail, False)
    return None


def _contains(matrix: list[list[int]], grid: Grid) -> bool:
    """Whether a grid appears in a matrix as a contiguous block."""
    height, width = len(grid), len(grid[0])
    for r in range(len(matrix) - height + 1):
        for c in range(len(matrix[0]) - width + 1):
            if all(matrix[r + i][c : c + width] == grid[i] for i in range(height)):
                return True
    return False


def _contains_run(values: list[int], run: list[int]) -> bool:
    """Whether a run appears in a sequence contiguously."""
    size = len(run)
    return any(values[i : i + size] == run for i in range(len(values) - size + 1))


def _literal_grid_names(tree: ast.Module) -> frozenset[str]:
    """Module-level names bound to a grid literal (or np.array of one)."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and _is_grid_literal(node.value, frozenset()):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return frozenset(names)


def _module_dicts(tree: ast.Module) -> dict[str, ast.Dict]:
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value
    return found


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_grid_literal(node: ast.AST, literal_names: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in literal_names
    if isinstance(node, ast.Call) and _call_name(node) in {"array", "asarray"} and node.args:
        return _is_grid_literal(node.args[0], literal_names)
    return _literal_matrix(node) is not None


def _is_whole_input(node: ast.AST, aliases: frozenset[str]) -> bool:
    """Whether an expression is the whole input, a copy of it, or a conversion of it."""
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, ast.Attribute) and node.attr == "T":
        return _is_whole_input(node.value, aliases)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _WHOLE_METHODS:
            return _is_whole_input(func.value, aliases)
        name = _call_name(node)
        if name in _WHOLE_FUNCS and node.args:
            return _is_whole_input(node.args[0], aliases)
        if name == "map" and len(node.args) == 2:
            return _is_whole_input(node.args[1], aliases)
    if isinstance(node, ast.ListComp | ast.GeneratorExp) and len(node.generators) == 1:
        return _is_whole_input(node.generators[0].iter, aliases)
    return False


def _function_aliases(function: ast.AST, seeds: set[str]) -> frozenset[str]:
    aliases = set(seeds)
    changed = True
    while changed:
        changed = False
        frozen = frozenset(aliases)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if _is_whole_input(node.value, frozen):
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in aliases:
                            aliases.add(target.id)
                            changed = True
    return frozenset(aliases)


def _tracked_functions(tree: ast.Module) -> dict[ast.AST, frozenset[str]]:
    """``transform`` and every top-level function it passes the whole input to, with aliases."""
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    functions[target.id] = node.value
    tracked: dict[ast.AST, frozenset[str]] = {}
    queue: list[tuple[str, int]] = [("transform", 0)]
    while queue:
        name, position = queue.pop()
        function = functions.get(name)
        if function is None:
            continue
        params = [*function.args.posonlyargs, *function.args.args]
        if position >= len(params):
            continue
        seed = params[position].arg
        if seed in tracked.get(function, frozenset()):
            continue
        aliases = _function_aliases(function, {seed, *tracked.get(function, frozenset())})
        tracked[function] = aliases
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                for index, arg in enumerate(node.args):
                    if node.func.id in functions and _is_whole_input(arg, aliases):
                        queue.append((node.func.id, index))
    return tracked


def _compare_findings(
    function: ast.AST, aliases: frozenset[str], literal_names: frozenset[str]
) -> Iterator[GuardFinding]:
    for node in ast.walk(function):
        operands: list[ast.AST] = []
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
        elif isinstance(node, ast.Call) and _call_name(node) in _EQUALITY_CALLS:
            operands = list(node.args)
        if (
            operands
            and any(_is_whole_input(o, aliases) for o in operands)
            and any(_is_grid_literal(o, literal_names) for o in operands)
        ):
            detail = f"line {node.lineno}: compares the whole input with a grid literal"
            yield GuardFinding("input_literal_compare", detail, True)


def _input_form(node: ast.AST, aliases: frozenset[str]) -> str | None:
    """Name the hashable form of the whole input an expression is, if any."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _BYTES_METHODS:
            if _is_whole_input(func.value, aliases):
                return "bytes"
        name = _call_name(node)
        if name in {"hash", "bytes"} and node.args and _mentions(node.args[0], aliases):
            return name
        if name in _STRING_CALLS and node.args and _is_whole_input(node.args[0], aliases):
            return "string"
        if name in {"tuple", "frozenset"} and node.args and _is_whole_input(node.args[0], aliases):
            return "tuple"
    if isinstance(node, ast.JoinedStr):
        if any(
            isinstance(v, ast.FormattedValue) and _is_whole_input(v.value, aliases)
            for v in node.values
        ):
            return "string"
    return None


def _mentions(node: ast.AST, aliases: frozenset[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in aliases for n in ast.walk(node))


def _is_constant_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return all(_is_constant_like(e) for e in node.elts)
    return False


def _hash_findings(function: ast.AST, aliases: frozenset[str]) -> Iterator[GuardFinding]:
    for node in ast.walk(function):
        keyed: list[ast.AST] = []
        if isinstance(node, ast.Call):
            name = _call_name(node)
            func = node.func
            # The bytes form and the hash of the input have no use but recognizing it.
            if name == "hash" or (isinstance(func, ast.Attribute) and name in _BYTES_METHODS):
                keyed.append(node)
            elif name in _KEY_METHODS and isinstance(func, ast.Attribute) and node.args:
                keyed.append(node.args[0])
        elif isinstance(node, ast.Subscript):
            keyed.append(node.slice)
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for op, left, right in zip(node.ops, operands, operands[1:], strict=False):
                if isinstance(op, ast.In | ast.NotIn):
                    keyed.append(left)
                elif _is_constant_like(right):
                    keyed.append(left)
                elif _is_constant_like(left):
                    keyed.append(right)
        elif isinstance(node, ast.Dict):
            keyed.extend(k for k in node.keys if k is not None)
        for expr in keyed:
            form = _input_form(expr, aliases)
            if form is not None:
                detail = f"line {node.lineno}: keys on the input's {form} form"
                yield GuardFinding("input_hash_key", detail, True)
                break


def _shape_names(function: ast.AST, aliases: frozenset[str]) -> tuple[set[str], set[str]]:
    """Names bound to one input dimension, and names bound to the input's shape tuple."""
    dims: set[str] = set()
    shapes: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if isinstance(target, ast.Name):
            if _is_dim(value, aliases, dims):
                dims.add(target.id)
            elif _is_shape(value, aliases, dims, shapes):
                shapes.add(target.id)
        elif isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts):
            names = [e.id for e in target.elts if isinstance(e, ast.Name)]
            if len(names) == 2 and _is_shape(value, aliases, dims, shapes):
                dims.update(names)
            elif isinstance(value, ast.Tuple) and len(value.elts) == len(names):
                pairs = zip(names, value.elts, strict=True)
                dims.update(n for n, v in pairs if _is_dim(v, aliases, dims))
    return dims, shapes


def _is_dim(node: ast.AST, aliases: frozenset[str], dims: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in dims
    if isinstance(node, ast.Subscript) and _int_value(node.slice) is not None:
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            return _is_whole_input(value.value, aliases)
    if isinstance(node, ast.Call) and _call_name(node) == "len" and node.args:
        arg = node.args[0]
        if _is_whole_input(arg, aliases):
            return True
        return isinstance(arg, ast.Subscript) and _is_whole_input(arg.value, aliases)
    return False


def _is_shape(node: ast.AST, aliases: frozenset[str], dims: set[str], shapes: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in shapes
    if isinstance(node, ast.Attribute) and node.attr == "shape":
        return _is_whole_input(node.value, aliases)
    if isinstance(node, ast.Call) and _call_name(node) == "shape" and node.args:
        return _is_whole_input(node.args[0], aliases)
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        return all(_is_dim(e, aliases, dims) for e in node.elts)
    return False


def _constant_tuple(node: ast.AST) -> tuple[int, ...] | None:
    values = _literal_vector(node)
    return None if values is None else tuple(values)


def _shape_findings(
    function: ast.AST,
    aliases: frozenset[str],
    inputs: Sequence[Grid],
    module_dicts: Mapping[str, ast.Dict],
) -> Iterator[GuardFinding]:
    sizes = {len(g) for g in inputs} | {len(g[0]) for g in inputs}
    shapes = {(len(g), len(g[0])) for g in inputs}
    cells = {len(g) * len(g[0]) for g in inputs}
    dim_names, shape_names = _shape_names(function, aliases)

    def hit(expr: ast.AST, other: ast.AST) -> str | None:
        is_shape = _is_shape(expr, aliases, dim_names, shape_names)
        is_dim = _is_dim(expr, aliases, dim_names)
        is_size = (
            isinstance(expr, ast.Attribute)
            and expr.attr == "size"
            and _is_whole_input(expr.value, aliases)
        )
        # `grid.shape == (3, 3)` names one shape; `h in (3, 5)` and dict keys list candidates.
        candidates = [other]
        if isinstance(other, ast.Tuple | ast.List | ast.Set) and not (
            is_shape and _constant_tuple(other) is not None
        ):
            candidates = list(other.elts)
        for value in candidates:
            number = _int_value(value)
            if is_dim and number in sizes:
                return f"the input dimension {number}"
            if is_size and number in cells:
                return f"the input size {number}"
            if is_shape and _constant_tuple(value) in shapes:
                return f"the input shape {_constant_tuple(value)}"
        return None

    for node in ast.walk(function):
        found = None
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for op, left, right in zip(node.ops, operands, operands[1:], strict=False):
                if isinstance(op, ast.Eq | ast.NotEq | ast.In | ast.NotIn):
                    found = hit(left, right) or hit(right, left)
                    if found:
                        break
        elif isinstance(node, ast.Match):
            for case in node.cases:
                pattern = case.pattern
                if isinstance(pattern, ast.MatchValue):
                    found = hit(node.subject, pattern.value)
                elif isinstance(pattern, ast.MatchSequence) and all(
                    isinstance(p, ast.MatchValue) for p in pattern.patterns
                ):
                    found = hit(node.subject, ast.Tuple([p.value for p in pattern.patterns]))
                if found:
                    break
        elif isinstance(node, ast.Subscript | ast.Call):
            container, key = _lookup(node)
            if isinstance(container, ast.Name):
                container = module_dicts.get(container.id)
            if isinstance(container, ast.Dict) and key is not None:
                keys = [k for k in container.keys if k is not None]
                found = hit(key, ast.Tuple(keys))
        if found:
            detail = f"line {node.lineno}: branches on {found}, a demonstration or test size"
            yield GuardFinding("shape_dispatch", detail, False)


def _lookup(node: ast.Subscript | ast.Call) -> tuple[ast.AST | None, ast.AST | None]:
    if isinstance(node, ast.Subscript):
        return node.value, node.slice
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "get" and node.args:
        return func.value, node.args[0]
    return None, None


def _dedupe(findings: Sequence[GuardFinding]) -> list[GuardFinding]:
    seen = set()
    unique = []
    for finding in findings:
        if finding not in seen:
            seen.add(finding)
            unique.append(finding)
    return unique
