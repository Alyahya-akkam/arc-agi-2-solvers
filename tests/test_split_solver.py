"""The split solver (task P2.0): moved, not rewritten, and wired as the reference.

``arc26/solver.py``, ``arc26/ttt.py``, and ``arc26/decode.py`` hold the statements of the generated
``arc26/nvarc/arc_solver.py`` and ``starter.py`` in functions (ADR-011). The first part compares
them on the AST. Every moved block's statement list, after the new docstring, is
``ast.dump``-equal to its slice of the reference; the parameters of each new function are exactly
the reference's locals that the block reads, in order of first use; its return values are exactly
the locals that the rest of the reference reads afterwards; and the split worker calls the blocks
in the reference's order with those names; the new functions take no decorator, default, or
``*args``/``**kwargs``; and the imports, as ``(module, name, asname)``, are the reference's.
``MOVES`` lists every moved block with every intended difference, the imports' added and removed
triples included; the tests allow no other. The second part runs both solvers' ``local_worker`` in a
subprocess against the stand-ins for the GPU stack of ``tests/test_solver_timing.py`` on its two
synthetic tasks: the candidate files, numpy's random state, the token counts the stand-ins
record, the timing records, and the printed lines must be the same. The third part checks that
importing ``arc26.solver``, as the solver's parent process does, imports no GPU stack, and runs
``python -m arc26.solver`` under a real spawn.
"""

import ast
import builtins
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parent.parent
ARC26 = REPO / "arc26"
NVARC = ARC26 / "nvarc"
BUILTINS = frozenset(dir(builtins))


class Move(NamedTuple):
    """One moved block: where it is, where it came from, and every intended difference."""

    module: str  # arc26/<module>.py, or several joined by ", "
    name: str  # the new function or class, "__main__" (entry block), or an import block
    source: str  # arc26/nvarc/<source>
    lines: tuple[tuple[int, int], ...]  # line ranges in the source
    intended: str
    # Import blocks only: the (module, name, asname) triples the split adds to the source's
    # imports or drops from them; module is None for ``import name``.
    added: frozenset = frozenset()
    removed: frozenset = frozenset()


MOVES = (
    Move(
        "decode, ttt, solver", "imports", "arc_solver.py", ((10, 34),),
        "the reference's imports, spread over decode.py and ttt.py (module level, outside "
        "if TYPE_CHECKING:) and the first statements of load, plus added, less removed",
        added=frozenset({
            ("__future__", "annotations", None),  # annotations stay strings, never evaluated
            ("typing", "TYPE_CHECKING", None),  # ttt.py: imports for annotations only
            (None, "unsloth", None),  # decode.py: unsloth before arc_loader's transformers
            ("arc26.decode", "USER_TOKEN_ID", None),  # ttt.py: the collator's constants,
            ("arc26.decode", "ASSISTANT_TOKEN_ID", None),  # which moved to decode.py
            ("arc26.decode", "EOS_ID", None),
            ("arc26.ttt", "QwenDataCollatorForCompletionOnlyLM", None),  # load: the class moved
        }),
        removed=frozenset({("tqdm", "tqdm", None)}),  # imported by arc_solver.py, never used
    ),
    Move(
        "solver", "entry imports", "starter.py", ((10, 17),),
        "the starter's module-level imports, plus added",
        added=frozenset({
            ("__future__", "annotations", None),
            ("typing", "TYPE_CHECKING", None),  # solver.py: imports for annotations only
            ("typing", "Any", None),
            ("typing", "NamedTuple", None),  # Loaded
        }),
    ),
    Move("decode", "ARC_VOCAB to EOS_ID", "arc_solver.py", ((38, 57),), "none"),
    Move(
        "decode", "turbo_dfs", "arc_solver.py", ((108, 175),),
        "type hints (the return type refined from dict) and a docstring",
    ),
    Move(
        "decode", "inference_turbo_dfs", "arc_solver.py", ((178, 197),),
        "type hints and a docstring",
    ),
    Move("decode", "calc_scores", "arc_solver.py", ((200, 228),), "type hints and a docstring"),
    Move(
        "decode", "decode_puzzle", "arc_solver.py", ((384, 495),),
        "a function of the 13 worker locals the block reads, in order of first use, with type "
        "hints and a docstring; it returns nothing, since the worker reads none of its locals; "
        "task P2.2 adds one keyword-only parameter, view_seed=None, and the seeds of its two "
        "augment calls become '<the reference's seed> if view_seed is None else <a seed from "
        "view_seed>'; task P3.7 adds a second, views=None, for decoding a subset of the 16 views "
        "(GEN's 8 views, P4.3's decode-count axis; ADR-019), and the batch loop's iterable "
        "becomes 'batches if views is None else <the views' batches per test input>'",
    ),
    Move("ttt", "logging.disable", "arc_solver.py", ((36, 36),), "none"),
    Move(
        "ttt", "UnslothFixedTrainer", "arc_solver.py", ((60, 85),),
        "a class docstring; compute_loss: type hints, and its one-line docstring extended to "
        "Google style",
    ),
    Move(
        "ttt", "QwenDataCollatorForCompletionOnlyLM", "arc_solver.py", ((88, 105),),
        "a class docstring; torch_call: a docstring",
    ),
    Move(
        "ttt", "fit", "arc_solver.py", ((331, 382),),
        "a function of the 11 worker locals the block reads, in order of first use, with type "
        "hints and a docstring; it returns (model, puzzle_ds), the locals the worker reads after "
        "it",
    ),
    Move(
        "solver", "load", "arc_solver.py", ((233, 316),),
        "a function of cfg with type hints and a docstring; it first imports the GPU names the "
        "block reads (unsloth first, then numpy, peft, arc_loader, the collator), which "
        "arc_solver.py imports at module level (the imports move above); it returns Loaded, "
        "the 12 locals the per-puzzle loop reads",
    ),
    Move(
        "solver", "worker", "arc_solver.py", ((318, 329), (497, 503)),
        "type hints and a docstring; it imports fit and decode_puzzle, unpacks load(cfg) into "
        "the 12 locals, and calls fit and decode_puzzle with keyword arguments of the same names "
        "in place of lines 331-382 and 384-495; task P2.1 adds the hold-out settings (None "
        "unless the config holds a held-out pair) and a branch around the two calls, whose "
        "else runs arc26.loop.solve_puzzle with the same locals; task P2.3 passes it the "
        "queue's size and the worker count as keyword-only arguments, the governor's inputs",
    ),
    Move(
        "solver", "local_worker", "starter.py", ((20, 42),),
        "type hints and a docstring; it imports arc26.decode and arc26.ttt (the GPU stack, "
        "unsloth first) where the reference imports worker from arc_solver, before the marker",
    ),
    Move("solver", "__main__", "starter.py", ((45, 73),), "none"),
)
# Top-level functions and classes of each new module: the moved ones, Loaded, and task P2.3's
# SGDProvider beside fit (our code, not moved: it calls fit), nothing else.
DEFINED = {
    "decode": ["turbo_dfs", "inference_turbo_dfs", "calc_scores", "decode_puzzle"],
    "ttt": ["UnslothFixedTrainer", "QwenDataCollatorForCompletionOnlyLM", "fit", "SGDProvider"],
    "solver": ["Loaded", "load", "worker", "local_worker"],
}
NOT_MOVED = {"Loaded", "SGDProvider"}
CONSTANTS = ["ARC_VOCAB", "ARC_TOKENS", "USER_TOKEN_ID", "ASSISTANT_TOKEN_ID", "PAD_ID", "EOS_ID"]
IMPORTS = next(move for move in MOVES if move.name == "imports")
ENTRY_IMPORTS = next(move for move in MOVES if move.name == "entry imports")
# The new functions whose signatures are new code: no decorator, default, *args, or **kwargs,
# but for the keyword-only parameters listed here with their defaults.
NEW_FUNCTIONS = [
    ("ttt", "fit"), ("decode", "decode_puzzle"), ("solver", "load"), ("solver", "worker")
]  # fmt: skip
# Task P2.2: decode_puzzle's view seed for the promotion's re-decode (ADR-014). With its default
# the two seeds it reads are the reference's: "<reference> if view_seed is None else <other>".
VIEW_SEED = "view_seed"
# Task P3.7: decode_puzzle's view subset (ADR-019). With its default the batch loop iterates the
# reference's batches: "batches if views is None else <the views' batches>".
VIEWS = "views"
KEYWORD_ONLY = {("decode", "decode_puzzle"): [(VIEW_SEED, None), (VIEWS, None)]}
VIEW_SEED_EXPRESSIONS = [
    ("2", "view_seed"), ("hash(bk) % 1024 ** 2", "hash((bk, view_seed)) % 1024 ** 2")
]  # fmt: skip
VIEWS_EXPRESSIONS = [
    (
        "batches",
        "[[keys[i] for i in batch] for keys in test_id_to_subkeys.values() for batch in views]",
    )
]
# What the solver's parent process must never import: the GPU stack and the modules that load it.
GPU_STACK = {"unsloth", "unsloth_zoo", "transformers", "peft", "datasets", "trl", "numpy"}


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference():
    return parse(NVARC / "arc_solver.py")


@pytest.fixture(scope="module")
def starter():
    return parse(NVARC / "starter.py")


@pytest.fixture(scope="module")
def split():
    return {name: parse(ARC26 / f"{name}.py") for name in DEFINED}


def top(module, name):
    (node,) = [
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name == name
    ]
    return node


def dumps(statements):
    return [ast.dump(statement) for statement in statements]


def is_docstring(statement):
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def after_docstring(node):
    """The body of a new function or class: it must start with a docstring, which is dropped."""
    assert node.body and is_docstring(node.body[0]), f"{node.name} has no docstring"
    return node.body[1:]


def span(node):
    first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    return first, node.end_lineno


def block(statements, first, last):
    """The statements of one body that lie in the lines first to last, which they must fill."""
    found = [s for s in statements if first <= s.lineno <= last]
    assert found, (first, last)
    assert (found[0].lineno, max(s.end_lineno for s in found)) == (first, last)
    index = statements.index(found[0])
    assert statements[index : index + len(found)] == found
    return found


def params(function):
    arguments = function.args
    assert not (arguments.posonlyargs or arguments.vararg or arguments.kwonlyargs)
    assert not (arguments.kwarg or arguments.defaults or arguments.kw_defaults)
    return [a.arg for a in arguments.args]


def positional(function):
    """The parameters a caller passes the worker's locals to: every one but the keyword-only
    parameters ``KEYWORD_ONLY`` lists, which keep their defaults in the worker's call."""
    arguments = function.args
    assert not (arguments.posonlyargs or arguments.vararg or arguments.kwarg or arguments.defaults)
    return [a.arg for a in arguments.args]


class ReferenceSeeds(ast.NodeTransformer):
    """Replaces each ``<a> if view_seed is None else <b>`` (and, task P3.7, each ``<a> if views
    is None else <b>``) by ``<a>``: what decode_puzzle runs with both keywords at their defaults.
    The replaced expressions are kept, in source order, by keyword."""

    def __init__(self):
        self.replaced = []
        self.by_name = {VIEW_SEED: [], VIEWS: []}

    def visit_IfExp(self, node):
        self.generic_visit(node)
        for name in (VIEW_SEED, VIEWS):
            if ast.unparse(node.test) == f"{name} is None":
                self.replaced.append(node)
                self.by_name[name].append(node)
                return node.body
        return node


def signature_names(function):
    """The parameter names, defaults, and decorators: everything of a signature but types."""
    arguments = function.args
    return (
        [a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs],
        getattr(arguments.vararg, "arg", None),
        getattr(arguments.kwarg, "arg", None),
        dumps(arguments.defaults),
        [ast.dump(d) if d else None for d in arguments.kw_defaults],
        dumps(function.decorator_list),
    )


# Name flow: which names a list of statements reads before binding them, in evaluation order.


def first_uses(statements):
    """Each name's first use in evaluation order: True if it is read first, False if bound.
    Names bound by a comprehension or a lambda are local to it and skipped."""
    uses = {}

    def name(node, reads, local):
        if node.id not in local:
            uses.setdefault(node.id, reads)

    def visit(node, local):
        if isinstance(node, ast.Name):
            name(node, isinstance(node.ctx, ast.Load), local)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            if node.value is not None:
                visit(node.value, local)
            for target in getattr(node, "targets", None) or [node.target]:
                visit(target, local)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                name(node.target, True, local)
            visit(node.value, local)
            visit(node.target, local)
        elif isinstance(node, ast.For):
            visit(node.iter, local)
            visit(node.target, local)
            for statement in node.body + node.orelse:
                visit(statement, local)
        elif isinstance(node, ast.With):
            for item in node.items:
                visit(item.context_expr, local)
                if item.optional_vars is not None:
                    visit(item.optional_vars, local)
            for statement in node.body:
                visit(statement, local)
        elif isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            bound = {
                n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)
            }
            visit(node.generators[0].iter, local)
            inner = local | bound
            for generator in node.generators[1:]:
                visit(generator.iter, inner)
            for generator in node.generators:
                for condition in generator.ifs:
                    visit(condition, inner)
            for part in ("elt", "key", "value"):
                if hasattr(node, part):
                    visit(getattr(node, part), inner)
        elif isinstance(node, ast.Lambda):
            visit(node.body, local | {a.arg for a in node.args.args})
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                uses.setdefault((alias.asname or alias.name).partition(".")[0], False)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, local)

    for statement in statements:
        visit(statement, frozenset())
    return uses


def reads(statements, bound=frozenset()):
    """Names the statements read before binding them, in order, less ``bound`` and builtins."""
    return [
        name
        for name, read in first_uses(statements).items()
        if read and name not in bound and name not in BUILTINS
    ]


def binds(statements):
    return {name for name, read in first_uses(statements).items() if not read} | {
        n.id
        for statement in statements
        for n in ast.walk(statement)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def module_globals(module):
    """Names a module binds at run time: imports, assignments, functions, and classes. The
    ``if TYPE_CHECKING:`` and ``__main__`` blocks bind nothing a function may use."""
    names = set()
    for node in module.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(a.asname or a.name).partition(".")[0] for a in node.names}
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return names


def imports_at_start(body):
    """The import statements a function body starts with."""
    count = 0
    while count < len(body) and isinstance(body[count], ast.Import | ast.ImportFrom):
        count += 1
    return body[:count]


class Reference(NamedTuple):
    """The reference worker cut into the blocks the split moves."""

    load: list  # lines 233-316
    loop: ast.While  # line 318
    head: list  # lines 318-329: deadline, queue, start times
    fit: list  # lines 331-382
    decode: list  # lines 384-495
    tail: list  # lines 497-503: inference memory, finish line, puzzle record


@pytest.fixture(scope="module")
def ref(reference):
    worker = top(reference, "worker")
    loop = worker.body[-1]
    assert isinstance(loop, ast.While) and loop.lineno == 318 and not loop.orelse
    parts = Reference(
        block(worker.body, 233, 316),
        loop,
        [s for s in loop.body if 318 <= s.lineno <= 329],
        block(loop.body, 331, 382),
        block(loop.body, 384, 495),
        block(loop.body, 497, 503),
    )
    # The blocks cover the reference worker: nothing is left out.
    assert worker.body == parts.load + [loop]
    assert loop.body == parts.head + parts.fit + parts.decode + parts.tail
    assert (parts.head[-1].end_lineno, parts.tail[-1].end_lineno) == (329, worker.end_lineno)
    return parts


def live_out(parts, before, after):
    """Names a block binds that the reference reads later: in the rest of the puzzle, then in
    the next puzzle (the loop carries model from one puzzle to the next)."""
    later = after + parts.loop.body
    return binds(before) & set(reads(later))


# The intended-difference list itself


def test_every_move_names_its_line_ranges_in_the_module_header():
    for move in MOVES:
        for module in move.module.split(", "):
            text = (ARC26 / f"{module}.py").read_text(encoding="utf-8")
            header = text.split("\nfrom __future__", 1)[0]
            for first, last in move.lines:
                where = f"line {first}" if first == last else f"{first}-{last}"
                assert where in header, f"{module}.py names no {where} for {move.name}"
            assert move.source in header


def test_the_new_modules_define_the_moved_code_and_nothing_else(split):
    for module, names in DEFINED.items():
        defined = [
            n.name for n in split[module].body if isinstance(n, ast.FunctionDef | ast.ClassDef)
        ]
        assert defined == names
        moved = {move.name for move in MOVES if move.module == module}
        assert set(names) - NOT_MOVED <= moved
    for module, tree in split.items():
        others = [
            n
            for n in tree.body[1:]
            if not isinstance(n, ast.Import | ast.ImportFrom | ast.FunctionDef | ast.ClassDef)
        ]
        kinds = [ast.unparse(n).split("\n")[0] for n in others]
        expected = {
            "decode": [f"{name} = " for name in CONSTANTS],
            "ttt": ["if TYPE_CHECKING:", "logging.disable(logging.WARNING)"],
            "solver": ["if TYPE_CHECKING:", "if __name__ == '__main__':"],
        }[module]
        assert len(kinds) == len(expected)
        assert all(kind.startswith(start) for kind, start in zip(kinds, expected, strict=True))


# Moved verbatim: module-level statements, functions, and classes


def test_the_token_constants_and_the_logging_switch_are_the_reference_statements(
    reference, split
):
    constants = block(reference.body, 38, 57)
    moved = [
        n
        for n in split["decode"].body
        if isinstance(n, ast.Assign) and n.targets[0].id in CONSTANTS
    ]
    assert dumps(moved) == dumps(constants)
    (switch,) = block(reference.body, 36, 36)
    (moved,) = [
        n for n in split["ttt"].body if isinstance(n, ast.Expr) and not is_docstring(n)
    ]
    assert ast.dump(moved) == ast.dump(switch)


@pytest.mark.parametrize("name", ["turbo_dfs", "inference_turbo_dfs", "calc_scores"])
def test_the_decode_functions_are_the_reference_functions(reference, split, name):
    original, moved = top(reference, name), top(split["decode"], name)
    assert signature_names(moved) == signature_names(original)
    assert dumps(after_docstring(moved)) == dumps(original.body)
    move = next(m for m in MOVES if m.name == name)
    assert move.lines == (span(original),)


@pytest.mark.parametrize("name", ["UnslothFixedTrainer", "QwenDataCollatorForCompletionOnlyLM"])
def test_the_training_classes_are_the_reference_classes(reference, split, name):
    original, moved = top(reference, name), top(split["ttt"], name)
    assert [dumps(n.bases) for n in (moved, original)] == [dumps(original.bases)] * 2
    assert dumps(moved.keywords) == dumps(original.keywords)
    assert dumps(moved.decorator_list) == dumps(original.decorator_list)
    methods = after_docstring(moved)
    assert [m.name for m in methods] == [m.name for m in original.body]
    for new, old in zip(methods, original.body, strict=True):
        assert isinstance(new, ast.FunctionDef) and isinstance(old, ast.FunctionDef)
        assert signature_names(new) == signature_names(old)
        body = old.body[1:] if is_docstring(old.body[0]) else old.body
        assert dumps(after_docstring(new)) == dumps(body)
    move = next(m for m in MOVES if m.name == name)
    assert move.lines == (span(original),)


# Moved blocks: the statements, the parameters, and the returns


def test_fit_is_the_training_block_of_a_puzzle(ref, reference, split):
    fit = top(split["ttt"], "fit")
    body = after_docstring(fit)
    assert dumps(body[:-1]) == dumps(ref.fit)
    # Parameters: the worker's locals the block reads, in order of first use. The block reads
    # the same globals in ttt.py as in arc_solver.py: ttt.py imports all of them.
    expected = reads(ref.fit, module_globals(reference))
    assert reads(ref.fit, module_globals(split["ttt"])) == expected == params(fit)
    assert len(expected) == 11
    # Returns: the locals the worker reads after the block.
    returned = body[-1]
    assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Tuple)
    names = [n.id for n in returned.value.elts]
    assert names == ["model", "puzzle_ds"]
    assert set(names) == live_out(ref, ref.fit, ref.decode + ref.tail)


def test_decode_puzzle_is_the_decode_block_of_a_puzzle(ref, reference, split):
    decode = top(split["decode"], "decode_puzzle")
    body = after_docstring(decode)
    # Task P2.2: with view_seed at its default (None), the block is the reference's statement
    # for statement; view_seed is read only in the two seeds. Task P3.7: with views at its
    # default too; views is read only in the batch loop's iterable.
    seeds = ReferenceSeeds()
    with_default = [seeds.visit(statement) for statement in copy.deepcopy(body)]
    assert dumps(with_default) == dumps(ref.decode)
    assert [(ast.unparse(n.body), ast.unparse(n.orelse)) for n in seeds.by_name[VIEW_SEED]] == (
        VIEW_SEED_EXPRESSIONS
    )
    assert [(ast.unparse(n.body), ast.unparse(n.orelse)) for n in seeds.by_name[VIEWS]] == (
        VIEWS_EXPRESSIONS
    )
    assert not [
        n
        for n in ast.walk(ast.Module(with_default, []))
        if getattr(n, "id", None) in (VIEW_SEED, VIEWS)
    ]
    # The views' batches replace the iterable of the loop over the reference's batches, nothing
    # else: the loop whose body decodes each batch.
    (loop,) = [
        n
        for n in ast.walk(ast.Module(body, []))
        if isinstance(n, ast.For) and ast.unparse(n.target) == "subkeys"
        and isinstance(n.iter, ast.IfExp)
    ]
    assert [ast.dump(n) for n in seeds.by_name[VIEWS]] == [ast.dump(loop.iter)]
    augment_seeds = [
        keyword.value
        for n in ast.walk(ast.Module(body, []))
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(".augment")
        for keyword in n.keywords
        if keyword.arg == "seed"
    ]
    assert [ast.unparse(seed) for seed in augment_seeds] == [
        f"{reference_form} if {VIEW_SEED} is None else {other}"
        for reference_form, other in VIEW_SEED_EXPRESSIONS
    ]
    expected = reads(ref.decode, module_globals(reference))
    assert reads(ref.decode, module_globals(split["decode"])) == expected == positional(decode)
    assert len(expected) == 13
    # No return: the worker reads none of the block's locals afterwards.
    assert not [n for n in ast.walk(decode) if isinstance(n, ast.Return)]
    assert live_out(ref, ref.decode, ref.tail) == set()


def test_load_is_the_start_of_the_worker(ref, reference, split):
    load = top(split["solver"], "load")
    body = after_docstring(load)
    imported = imports_at_start(body)
    assert dumps(body[len(imported) : -1]) == dumps(ref.load)
    assert params(load) == ["cfg"]
    # The leading imports bind exactly the GPU names the block reads that arc_solver.py
    # imports at module level and solver.py does not.
    bound = {(a.asname or a.name) for node in imported for a in node.names}
    reference_reads = set(reads(ref.load, {"cfg"})) - BUILTINS
    assert bound == reference_reads - module_globals(split["solver"])
    assert bound <= module_globals(reference)
    assert reads(ref.load, module_globals(split["solver"]) | bound) == ["cfg"]
    # Returns: Loaded, the locals the per-puzzle loop reads, as keywords of the same names.
    returned = body[-1]
    assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Call)
    assert ast.unparse(returned.value.func) == "Loaded" and not returned.value.args
    keywords = [(k.arg, ast.unparse(k.value)) for k in returned.value.keywords]
    assert all(arg == value for arg, value in keywords)
    fields = [
        n.target.id for n in top(split["solver"], "Loaded").body if isinstance(n, ast.AnnAssign)
    ]
    assert [arg for arg, _ in keywords] == fields
    assert set(fields) == binds(ref.load) & set(reads([ref.loop]))
    assert len(fields) == 12


def keyword_call(statement, function, names):
    """Whether the statement's call is ``function(name=name, ...)`` for ``names`` in order."""
    call = statement.value
    return (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) == function
        and not call.args
        and [(k.arg, ast.unparse(k.value)) for k in call.keywords] == [(n, n) for n in names]
    )


def test_the_worker_calls_load_fit_and_decode_puzzle_in_the_reference_order(ref, reference, split):
    worker = top(split["solver"], "worker")
    body = after_docstring(worker)
    assert params(worker) == ["rank", "queue", "end_time", "cfg"]
    imported = imports_at_start(body)
    unpack, settings, loop = body[len(imported) :]
    assert [ast.unparse(n) for n in imported] == [
        "from arc26.decode import decode_puzzle",
        "from arc26.loop import HoldOut, solve_puzzle",
        "from arc26.ttt import fit",
    ]
    # The loop's locals from load, by name.
    fields = [
        n.target.id for n in top(split["solver"], "Loaded").body if isinstance(n, ast.AnnAssign)
    ]
    assert isinstance(unpack, ast.Assign) and ast.unparse(unpack.value) == "load(cfg)"
    (target,) = unpack.targets
    assert [n.id for n in target.elts] == fields
    # Task P2.1: the hold-out settings, None unless the config holds "holdout".
    assert ast.unparse(settings) == "holdout = HoldOut.from_cfg(cfg)"
    # The loop: the reference's, with the two blocks replaced by the calls, which run whenever
    # no pair is held out; solve_puzzle runs instead with a held-out pair.
    assert isinstance(loop, ast.While) and not loop.orelse
    assert ast.dump(loop.test) == ast.dump(ref.loop.test)
    head, rest = loop.body[: len(ref.head)], loop.body[len(ref.head) :]
    (branch,), tail = rest[:1], rest[1:]
    assert dumps(head) == dumps(ref.head) and dumps(tail) == dumps(ref.tail)
    assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "holdout is None"
    (fit_call, decode_call), (solve_call,) = branch.body, branch.orelse
    fit = top(split["ttt"], "fit")
    decode = top(split["decode"], "decode_puzzle")
    assert isinstance(fit_call, ast.Assign) and keyword_call(fit_call, "fit", params(fit))
    assert ast.unparse(fit_call.targets[0]) == ast.unparse(after_docstring(fit)[-1].value)
    assert isinstance(decode_call, ast.Expr)
    # The worker passes the 13 locals and leaves view_seed at its default: the reference's views.
    assert keyword_call(decode_call, "decode_puzzle", positional(decode))
    # solve_puzzle takes the settings and every worker local the two blocks read, in order of
    # first use, and returns the model, the one local the loop carries to the next puzzle. Task
    # P2.3 adds two keyword-only arguments, the governor's inputs: the queue's size right after
    # the worker took the puzzle and the worker count.
    solve = top(parse(ARC26 / "loop.py"), "solve_puzzle")
    locals_read = reads(ref.fit + ref.decode, module_globals(reference))
    assert positional(solve) == ["holdout", *locals_read] and len(locals_read) == 16
    assert [a.arg for a in solve.args.kwonlyargs] == ["queue_size", "workers"]
    assert isinstance(solve_call, ast.Assign) and not solve_call.value.args
    keywords = [(k.arg, ast.unparse(k.value)) for k in solve_call.value.keywords]
    assert keywords == [(n, n) for n in positional(solve)] + [
        ("queue_size", "queue.qsize()"), ("workers", "cfg['workers']")
    ]  # fmt: skip
    assert ast.unparse(solve_call.value.func) == "solve_puzzle"
    assert ast.unparse(solve_call.targets[0]) == "model"


def test_local_worker_is_the_starters_but_imports_the_split_modules(starter, split):
    original, moved = top(starter, "local_worker"), top(split["solver"], "local_worker")
    assert signature_names(moved) == signature_names(original)
    body = after_docstring(moved)
    assert len(body) == len(original.body)
    differing = [
        (ast.unparse(old), ast.unparse(new))
        for old, new in zip(original.body, body, strict=True)
        if ast.dump(old) != ast.dump(new)
    ]
    assert differing == [
        ("from arc26.nvarc.arc_solver import worker", "from arc26 import decode, ttt")
    ]
    # The import comes before the marker that lets the next rank import, as in the reference.
    position = [ast.unparse(s) for s in body].index("from arc26 import decode, ttt")
    assert isinstance(body[position + 1], ast.With) and "worker{rank}" in ast.unparse(
        body[position + 1]
    )
    assert next(m for m in MOVES if m.name == "local_worker").lines == (span(original),)


def test_the_entry_block_is_the_starters(starter, split):
    original, moved = starter.body[-1], split["solver"].body[-1]
    assert ast.unparse(original.test) == "__name__ == '__main__'"
    assert ast.dump(moved) == ast.dump(original)
    assert next(m for m in MOVES if m.name == "__main__").lines == ((45, 73),)
    assert span(original) == (45, 73)


# Imports: unsloth first wherever the GPU stack is imported, none of it in the parent process.


def first_import(tree):
    statements = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    return next(s for s in statements if getattr(s, "module", None) != "__future__")


def test_unsloth_is_imported_before_anything_else_in_the_gpu_modules(reference, split):
    assert ast.unparse(first_import(reference)).startswith("from unsloth import")
    assert ast.unparse(first_import(split["decode"])) == "import unsloth"
    assert ast.unparse(first_import(split["ttt"])).startswith("from unsloth import")
    # load too, so that a caller outside the worker process keeps the order.
    (first, *_) = imports_at_start(after_docstring(top(split["solver"], "load")))
    assert ast.unparse(first).startswith("from unsloth import")


def import_triples(statements):
    """``(module, name, asname)`` of every name the import statements bind; module is None for
    ``import name``. Statements of other kinds, ``if TYPE_CHECKING:`` blocks included, add none."""
    found = set()
    for node in statements:
        if isinstance(node, ast.Import):
            found |= {(None, alias.name, alias.asname) for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            found |= {(node.module, alias.name, alias.asname) for alias in node.names}
    return found


def test_the_split_imports_are_the_references_with_the_listed_differences(reference, split):
    """Compared by source module and alias, so that an import from another module under the same
    name, or an alias that swaps two names, cannot pass for the reference's."""
    imports = [n for n in reference.body if isinstance(n, ast.Import | ast.ImportFrom)]
    assert (imports[0].lineno, imports[-1].end_lineno) == IMPORTS.lines[0]
    original = import_triples(reference.body)
    load = top(split["solver"], "load")
    moved = (
        import_triples(split["decode"].body)
        | import_triples(split["ttt"].body)
        | import_triples(imports_at_start(after_docstring(load)))
    )
    assert moved == (original - IMPORTS.removed) | IMPORTS.added
    # Every listed difference is one: an addition the reference lacks, a removal it has.
    assert not IMPORTS.added & original and IMPORTS.removed <= original


def test_solver_imports_at_module_level_what_the_starter_imports(starter, split):
    imports = [n for n in starter.body if isinstance(n, ast.Import | ast.ImportFrom)]
    assert (imports[0].lineno, imports[-1].end_lineno) == ENTRY_IMPORTS.lines[0]
    original = import_triples(starter.body)
    assert import_triples(split["solver"].body) == original | ENTRY_IMPORTS.added
    assert not ENTRY_IMPORTS.added & original and not ENTRY_IMPORTS.removed


@pytest.mark.parametrize(("module", "name"), NEW_FUNCTIONS)
def test_the_new_functions_take_plain_parameters(split, module, name):
    """The calls pass every local by name, and nothing wraps the moved statements: a
    ``torch.no_grad`` on fit, for one, would break training. The only keyword-only parameters
    are those ``KEYWORD_ONLY`` lists, with their defaults (decode_puzzle's view_seed=None)."""
    function = top(split[module], name)
    arguments = function.args
    assert function.decorator_list == []
    assert not arguments.posonlyargs
    assert arguments.vararg is None and arguments.kwarg is None
    assert arguments.defaults == []
    keyword_only = [
        (a.arg, ast.literal_eval(default))
        for a, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
    ]
    assert keyword_only == KEYWORD_ONLY.get((module, name), [])


IMPORT_PROBE = r"""
import json, sys, types

attempted = []


class Recorder:
    def find_spec(self, name, path=None, target=None):
        attempted.append(name)
        return None


sys.meta_path.insert(0, Recorder())
# Stand-ins: the CPU test environment has no torch; both entries import it at module level.
torch = types.ModuleType("torch")
torch.multiprocessing = types.ModuleType("torch.multiprocessing")
sys.modules.update({"torch": torch, "torch.multiprocessing": torch.multiprocessing})
import importlib

importlib.import_module(sys.argv[1])
print(json.dumps(attempted))
"""


@pytest.mark.parametrize("entry", ["arc26.nvarc.starter", "arc26.solver"])
def test_importing_a_solver_entry_imports_no_gpu_stack(entry):
    """The solver's parent process and every worker process import the entry module before
    local_worker runs, so importing it must not import the GPU stack: local_worker imports it,
    after CUDA_VISIBLE_DEVICES is set and in rank order (the spawn test below)."""
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    probe = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE, entry],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    attempted = json.loads(probe.stdout)
    assert not {name.partition(".")[0] for name in attempted} & GPU_STACK
    packages = {entry.rsplit(".", count)[0] for count in range(entry.count(".") + 1)}
    assert [n for n in attempted if n.startswith("arc26.") and n not in packages] == [
        "arc26.timing"
    ]


# The wired split solver against the reference, on stand-ins for the GPU stack


def load_timing_tests():
    path = REPO / "tests" / "test_solver_timing.py"
    spec = importlib.util.spec_from_file_location("_split_solver_stand_ins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAND_INS = load_timing_tests()
# test_solver_timing's driver imports the reference's two modules under these names, replaces
# their DFS and rescoring with deterministic fakes, and runs starter.local_worker. For the split
# solver, the same names point at arc26.decode, which holds the DFS and the rescoring, and at
# arc26.solver, which holds local_worker.
REFERENCE_IMPORT = "from arc26.nvarc import arc_solver, starter"
SPLIT_IMPORT = "from arc26 import decode as arc_solver, solver as starter"
RUNS = {"plain-off": ("plain", False), "plain-on": ("plain", True), "cap-on": ("cap", True)}
SECONDS = re.compile(r"\d+\.\d+s\b")


@pytest.fixture(scope="module")
def solver_runs(tmp_path_factory):
    assert STAND_INS.DRIVER.count(REFERENCE_IMPORT) == 1
    drivers = {
        "nvarc": STAND_INS.DRIVER,
        "split": STAND_INS.DRIVER.replace(REFERENCE_IMPORT, SPLIT_IMPORT),
    }
    root = tmp_path_factory.mktemp("split")
    runs = {}
    try:
        for solver, driver in drivers.items():
            STAND_INS.DRIVER = driver
            for name, (mode, timing) in RUNS.items():
                runs[solver, name] = STAND_INS.run_worker(root / f"{solver}-{name}", mode, timing)
    finally:
        STAND_INS.DRIVER = drivers["nvarc"]
    return runs


@pytest.mark.parametrize("name", list(RUNS))
def test_the_split_solver_writes_the_reference_candidates(solver_runs, name):
    nvarc, split = solver_runs["nvarc", name], solver_runs["split", name]
    assert split.summary["candidates"] and split.summary["candidates"] == nvarc.summary[
        "candidates"
    ]
    # numpy's global random state (the augmentations reseed it) and the token counts the
    # stand-ins record: per trainer.train, per DFS call, and per calc_scores call.
    assert split.summary["rng"] == nvarc.summary["rng"]
    assert split.summary["calls"] == nvarc.summary["calls"]


@pytest.mark.parametrize("name", list(RUNS))
def test_the_split_solver_prints_the_reference_lines(solver_runs, name):
    nvarc, split = solver_runs["nvarc", name], solver_runs["split", name]
    assert SECONDS.sub("<s>", split.output) == SECONDS.sub("<s>", nvarc.output)
    assert "[Rank 0] done!" in split.output


@pytest.mark.parametrize("name", ["plain-on", "cap-on"])
def test_the_split_solver_writes_the_reference_timing_records(solver_runs, name):
    nvarc, split = solver_runs["nvarc", name], solver_runs["split", name]
    assert split.summary["logger"].endswith("run_logs/timing/timing-rank0.jsonl")

    def ops(run):
        return [(r["puzzle_id"], r["op"], r["tokens"]) for r in run.records]

    assert ops(split) == ops(nvarc) and len(ops(split)) > 2


def test_the_split_solver_without_a_timing_dir_writes_no_record(solver_runs):
    split = solver_runs["split", "plain-off"]
    assert split.summary["logger"] is None
    written = sorted(p.relative_to(split.work).as_posix() for p in split.work.rglob("*"))
    assert not [name for name in written if "timing" in name]


# The process entry under a real spawn: python -m arc26.solver, two workers, no task


STAND_IN_PACKAGES = {
    "torch/__init__.py": """
import contextlib, types

float32, bfloat16 = "float32", "bfloat16"
cuda = types.SimpleNamespace(
    reset_peak_memory_stats=lambda: None, empty_cache=lambda: None, max_memory_allocated=lambda: 0
)
no_grad = lambda: (lambda function: function)
inference_mode = contextlib.nullcontext
set_default_device = lambda device: None
""",
    "torch/multiprocessing.py": """
import multiprocessing

Manager = multiprocessing.Manager


def spawn(fn, args=(), nprocs=1):
    # As torch.multiprocessing.spawn: one process per rank, spawn start method, fn(rank, *args).
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=fn, args=(rank, *args)) for rank in range(nprocs)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    codes = [process.exitcode for process in processes]
    if any(codes):
        raise SystemExit(f"worker exit codes {codes}")
""",
    "unsloth/__init__.py": """
import os

# Reported once per process that imports the GPU stack.
print(f"unsloth imported with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")


class Tokenizer:
    def encode(self, text):
        return [ord(c) for c in text]


class Model:
    def named_parameters(self):
        return []


class FastLanguageModel:
    @staticmethod
    def from_pretrained(**kwargs):
        return Model(), Tokenizer()

    @staticmethod
    def get_peft_model(model, **kwargs):
        return model


class UnslothTrainer:
    pass


UnslothTrainingArguments = dict
""",
    "datasets/__init__.py": "class Dataset:\n    pass\n",
    "transformers/__init__.py": (
        "class DataCollatorForLanguageModeling:\n"
        "    def __init__(self, tokenizer, mlm):\n"
        "        self.tokenizer = tokenizer\n\n\n"
        "AutoTokenizer = object\n"
    ),
    "peft/__init__.py": (
        "get_peft_model_state_dict = lambda model, adapter_name: {}\n"
        "set_peft_model_state_dict = lambda model, weights, adapter_name: None\n"
    ),
}


def test_python_m_arc26_solver_spawns_its_workers_as_the_starter_does(tmp_path):
    """Only the two spawned workers import the GPU stack, each after it is pinned to its GPU,
    rank 1 after rank 0 (the markers); both write their markers, record the model load, and stop
    at the queue's None (the queue holds no task, since the subset matches none). Output is
    unbuffered, as run_solver makes it, so the lines arrive in the order they are printed."""
    pytest.importorskip("numpy")
    packages = tmp_path / "stand_ins"
    for name, text in STAND_IN_PACKAGES.items():
        (packages / name).parent.mkdir(parents=True, exist_ok=True)
        (packages / name).write_text(text, encoding="utf-8")
    data, markers, timing = tmp_path / "data", tmp_path / "markers", tmp_path / "timing"
    for folder in (data, markers):
        folder.mkdir()
    challenges = {"00000001": STAND_INS.TASKS["00000001"]}
    (data / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    cfg = {
        "data_dir": str(data),
        "model_dir": str(tmp_path / "model"),
        "outputs_dir": str(tmp_path / "outputs"),
        "marker_dir": str(markers),
        "subset": ["ffffffff"],
        "workers": 2,
        "timing_dir": str(timing),
        "lora_r": 256,
        "lora_targets": ["q_proj"],
    }
    env = {key: value for key, value in os.environ.items() if key != "KAGGLE_IS_COMPETITION_RERUN"}
    env.update(
        PYTHONPATH=os.pathsep.join([str(packages), str(REPO)]),
        PYTHONHASHSEED="0",
        PYTHONUNBUFFERED="1",
    )
    command = [sys.executable, "-m", "arc26.solver", "--end-time", "1e12"]
    result = subprocess.run(
        [*command, "--config", json.dumps(cfg)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    imported = [line for line in result.stdout.splitlines() if line.startswith("unsloth imported")]
    assert imported == [f"unsloth imported with CUDA_VISIBLE_DEVICES={r}" for r in (0, 1)]
    for rank in (0, 1):
        assert f"[Rank {rank}] start!" in result.stdout
        assert f"[Rank {rank}] done!" in result.stdout
        assert (markers / f"worker{rank}").read_text() == "Ok"
        lines = (timing / f"timing-rank{rank}.jsonl").read_text().splitlines()
        (record,) = [json.loads(line) for line in lines]
        assert (record["puzzle_id"], record["op"]) == (None, "load")
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []
