r"""Move the reference notebook's code cells into ``arc26/nvarc/`` (task P0.1, ADR-009).

Purpose: the 2025 first-place notebook keeps its code in four ``%%writefile`` cells and one cell
it runs inline. P0.1 moves each of them into one module of ``arc26/nvarc/``, unchanged except for
the literal replacements ``configs/extraction.yaml`` lists with a reason each, so every deviation
from the reference is one reviewable list and the modules can be regenerated at any time.

Inputs: the spec (default ``configs/extraction.yaml``) and the notebook it names (default: the
spec's ``notebook``, relative to the repository root). Output: one module per spec entry in
``--out`` (default ``arc26/nvarc/``), each starting with a header comment that names its source
cell. Nothing else is read or written: no git, no network.

Method: the notebook's sha256 must equal the spec's. For a ``%%writefile`` cell the module body is
what the magic writes: the cell after its first line, ending in a newline. For an inline cell
(``writefile: null``) the first ``top_lines`` lines, its imports, stay at module top and the rest
becomes the body of the spec's ``function``, indented four spaces. Replacements are literal and
applied in one pass by ``apply_replacements`` of ``scripts/build_reference_run.py``, which refuses
any count other than the spec's. Every module is built and compiled before the first write, so a
refused spec writes nothing, and reruns are byte-identical. ``--check`` writes nothing and exits 1
when a module on disk differs from what the spec produces.

Usage::

    python scripts/extract_reference.py            # write arc26/nvarc/*.py
    python scripts/extract_reference.py --check    # exit 1 if a module differs from the spec
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "extraction.yaml"
DEFAULT_OUT = REPO / "arc26" / "nvarc"
# Names written into the module headers, independent of the paths given on the command line.
SCRIPT_NAME = "scripts/extract_reference.py"
SPEC_NAME = "configs/extraction.yaml"
TOP_KEYS = frozenset({"notebook", "sha256", "origin", "kaggle_notebook", "license", "modules"})
MODULE_KEYS = frozenset({"module", "cell", "writefile", "replacements"})
WRAP_KEYS = frozenset({"function", "top_lines"})
ENTRY_KEYS = frozenset({"old", "new", "count"})
SHA256 = re.compile(r"[0-9a-f]{64}")
MODULE_NAME = re.compile(r"[a-z_][a-z0-9_]*\.py")
FUNCTION = re.compile(r"def [A-Za-z_][A-Za-z0-9_]*\(.*\):")
IMPORT_LINE = re.compile(r"(?:import|from) \S")
WRITEFILE = "%%writefile "
INDENT = "    "


def _load_builder() -> ModuleType:
    # scripts/ is not a package, so the P0.1-01 builder is loaded by path, as its tests do.
    path = Path(__file__).resolve().parent / "build_reference_run.py"
    spec = importlib.util.spec_from_file_location("build_reference_run", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILDER = _load_builder()


class ExtractionError(Exception):
    """The notebook or the spec breaks a rule of the extraction; no module was written."""


class Module(NamedTuple):
    """One spec entry: a module of ``arc26/nvarc/`` and how its source cell becomes it.

    Attributes:
        name: File name in the output folder, for example ``starter.py``.
        cell: Index of the source cell in the notebook.
        writefile: Name after the cell's ``%%writefile`` magic; None for an inline cell.
        function: Inline cells only: the ``def`` line whose body the cell becomes.
        top_lines: Inline cells only: leading lines (the imports) kept at module top.
        replacements: Literal replacements, as ``build_reference_run.Replacement`` tuples.
    """

    name: str
    cell: int
    writefile: str | None
    function: str | None
    top_lines: int | None
    replacements: tuple[Any, ...]


class Spec(NamedTuple):
    """The extraction settings read from ``configs/extraction.yaml``.

    Attributes:
        notebook: Notebook path relative to the repository root, as written in the headers.
        sha256: The notebook's expected sha256.
        origin: Who wrote the notebook, for the headers.
        kaggle_notebook: The notebook's Kaggle id, for the headers.
        license: License and pointer to the full attribution, for the headers.
        modules: The module entries, in spec order.
    """

    notebook: str
    sha256: str
    origin: str
    kaggle_notebook: str
    license: str
    modules: tuple[Module, ...]


def _read(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as err:
        raise ExtractionError(f"cannot read the {what} {path}: {err}") from err


def _mapping(
    value: Any, required: frozenset[str], optional: frozenset[str], where: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExtractionError(f"{where} must be a mapping")
    keys = {str(key) for key in value}
    missing, unknown = sorted(required - keys), sorted(keys - required - optional)
    if missing or unknown:
        raise ExtractionError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return value


def _string(value: Any, where: str, multiline: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\r" in value
        or (not multiline and "\n" in value)
    ):
        kind = "a non-empty string" if multiline else "a non-empty single-line string"
        raise ExtractionError(f"{where} must be {kind}")
    return value


def _count(value: Any, where: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ExtractionError(f"{where} must be an integer >= {minimum}; got {value!r}")
    return value


def _replacement(entry: Any, where: str) -> Any:
    entry = _mapping(entry, ENTRY_KEYS, frozenset(), where)
    old = _string(entry["old"], f"{where}.old")
    new = _string(entry["new"], f"{where}.new", multiline=True)
    if old == new:
        raise ExtractionError(f"{where}: old and new are the same literal")
    return _BUILDER.Replacement(old, new, _count(entry["count"], f"{where}.count", 1))


def _module(entry: Any, where: str) -> Module:
    entry = _mapping(entry, MODULE_KEYS, WRAP_KEYS, where)
    name = entry["module"]
    if not isinstance(name, str) or not MODULE_NAME.fullmatch(name) or name == "__init__.py":
        raise ExtractionError(
            f"{where}.module must be a module file name such as starter.py (not __init__.py, "
            f"which is written by hand); got {name!r}"
        )
    cell = _count(entry["cell"], f"{where}.cell", 0)
    writefile, function, top_lines = entry["writefile"], None, None
    wrap = WRAP_KEYS & set(entry)
    if writefile is not None:
        writefile = _string(writefile, f"{where}.writefile")
        if wrap:
            raise ExtractionError(
                f"{where}: {sorted(wrap)} apply only to an inline cell (writefile: null)"
            )
    else:
        if wrap != WRAP_KEYS:
            raise ExtractionError(
                f"{where}: an inline cell (writefile: null) needs function and top_lines"
            )
        function = _string(entry["function"], f"{where}.function")
        if not FUNCTION.fullmatch(function):
            raise ExtractionError(f"{where}.function must be a def line; got {function!r}")
        top_lines = _count(entry["top_lines"], f"{where}.top_lines", 0)
    entries = entry["replacements"]
    if not isinstance(entries, list):
        raise ExtractionError(f"{where}.replacements must be a list (empty when none)")
    replacements = tuple(
        _replacement(item, f"{where}.replacements[{i}]") for i, item in enumerate(entries)
    )
    for i, first in enumerate(replacements):
        for j, second in enumerate(replacements):
            if i != j and first.old in second.old:
                raise ExtractionError(
                    f"{where}: replacements[{i}].old occurs inside replacements[{j}].old, so "
                    "their counts are ambiguous; anchor them with more context"
                )
    return Module(name, cell, writefile, function, top_lines, replacements)


def parse_spec(data: bytes, path: Path) -> Spec:
    """Parse and check the extraction spec.

    Why: the spec is the complete list of deviations from the reference, so it is checked
    strictly. Unknown keys are refused (a misspelt ``replacements`` would otherwise move a cell
    unpatched), and ``old`` literals that contain one another are refused because their counts
    would be ambiguous.

    Args:
        data: Raw bytes of ``configs/extraction.yaml``.
        path: Its path, used in messages.

    Returns:
        The parsed spec.

    Raises:
        ExtractionError: If the text is not UTF-8 or not YAML, a key is missing or unknown, a
            value has the wrong type, a module name repeats or is ``__init__.py``, an inline cell
            lacks ``function`` or ``top_lines`` (or a ``%%writefile`` cell has them), or an
            ``old`` occurs inside another ``old`` of the same module.
    """
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except UnicodeDecodeError as err:
        raise ExtractionError(f"{path} is not UTF-8 text: {err}") from err
    except yaml.YAMLError as err:
        raise ExtractionError(f"{path} is not valid YAML: {err}") from err
    raw = _mapping(raw, TOP_KEYS, frozenset(), str(path))
    sha256 = raw["sha256"]
    if not isinstance(sha256, str) or not SHA256.fullmatch(sha256):
        raise ExtractionError(f"{path}: sha256 must be 64 lowercase hex digits, quoted")
    entries = raw["modules"]
    if not isinstance(entries, list) or not entries:
        raise ExtractionError(f"{path}: modules must be a non-empty list")
    modules = tuple(_module(entry, f"modules[{i}]") for i, entry in enumerate(entries))
    names = [module.name for module in modules]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ExtractionError(f"{path}: modules repeat {repeated}")
    return Spec(
        _string(raw["notebook"], "notebook"),
        sha256,
        _string(raw["origin"], "origin"),
        _string(raw["kaggle_notebook"], "kaggle_notebook"),
        _string(raw["license"], "license"),
        modules,
    )


def _cell_source(notebook: Any, module: Module) -> str:
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list) or not 0 <= module.cell < len(cells):
        raise ExtractionError(f"{module.name}: the notebook has no cell {module.cell}")
    cell = cells[module.cell]
    if not isinstance(cell, dict) or cell.get("cell_type") != "code":
        raise ExtractionError(f"{module.name}: cell {module.cell} is not a code cell")
    source = cell.get("source", "")
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        source = "".join(source)
    if not isinstance(source, str):
        raise ExtractionError(f"{module.name}: cell {module.cell} has a source that is not text")
    return source


def module_header(module: Module, spec: Spec) -> str:
    """Return the header comment that starts a generated module, ending in a blank line.

    Why: provenance must be readable in the file itself (CLAUDE.md: a header names the source
    cell), and Apache 2.0 asks for the license, attribution, and a notice on changed files.

    Args:
        module: The spec entry.
        spec: The spec, for the notebook, its hash, origin, and license.

    Returns:
        ``#`` comment lines naming the notebook, cell, ``%%writefile`` name, sha256, origin,
        license, the changes, and the generator, then one empty line.
    """
    count = len(module.replacements)
    changes = [f"Changes: {count} literal replacement{'' if count == 1 else 's'}"]
    if module.writefile is not None:
        cell = f"cell {module.cell} (%%writefile {module.writefile})"
    else:
        cell = f"cell {module.cell} (runs inline, no %%writefile)"
        name = (module.function or "def (").removeprefix("def ").split("(")[0]
        changes[0] += f"; the code after the imports is the body of {name}"
    if count or module.writefile is None:
        changes[0] += ","
        changes.append(f"  listed with their reasons in {SPEC_NAME}")
    else:
        changes = ["Changes: none, moved verbatim"]
    lines = [
        f"Moved from {spec.notebook}, {cell}.",
        f"Notebook sha256 {spec.sha256}",
        f"Origin: {spec.origin},",
        f"  Kaggle {spec.kaggle_notebook}",
        f"License: {spec.license}",
        *changes,
        f"generated by {SCRIPT_NAME} from {SPEC_NAME}; edit the spec, not this file",
    ]
    return "".join(f"# {line}\n" for line in lines) + "\n"


def build_module(source: str, module: Module, spec: Spec) -> str:
    """Build one module's text from its cell: header, then the patched (and wrapped) code.

    Why: this is the whole transformation from cell to module, so ``--check`` and the writer
    produce the same bytes and a reviewer can read every rule in one place.

    Args:
        source: The cell's source text.
        module: The spec entry.
        spec: The spec, for the header.

    Returns:
        The module text: the header, then the code, ending in exactly one newline.

    Raises:
        ExtractionError: If a ``%%writefile`` cell does not start with the spec's magic line, an
            inline cell starts with a magic, an inline cell's top lines are not all imports or
            nothing follows them, a replacement count does not match, or the result does not
            compile.
    """
    if module.writefile is not None:
        first, _, body = source.partition("\n")
        if first != f"{WRITEFILE}{module.writefile}":
            raise ExtractionError(
                f"{module.name}: cell {module.cell} starts with {first!r}, not "
                f"{WRITEFILE}{module.writefile!r}"
            )
        parts = [body]
    else:
        if source.lstrip().startswith(("%", "!")):
            raise ExtractionError(
                f"{module.name}: cell {module.cell} starts with a magic or shell line; "
                "only a plain code cell can be moved inline"
            )
        lines = source.split("\n")
        top, rest = lines[: module.top_lines], lines[module.top_lines :]
        if not all(IMPORT_LINE.match(line) for line in top) or not "".join(rest).strip():
            raise ExtractionError(
                f"{module.name}: the first {module.top_lines} lines of cell {module.cell} must "
                "be imports, followed by code"
            )
        # Top and rest are patched as two parts of one cell: counts span both, and a
        # multi-line new text cannot shift the split.
        parts = ["\n".join(top), "\n".join(rest)]
    try:
        patched, _ = _BUILDER.apply_replacements(
            [{"cell_type": "code", "source": part} for part in parts], module.replacements
        )
    except _BUILDER.BuildError as err:
        raise ExtractionError(f"{module.name} (cell {module.cell}): {err}") from err
    texts = [cell["source"] for cell in patched]
    if module.writefile is not None:
        # IPython's cell magic receives the cell with a final newline added when missing.
        code = texts[0] if texts[0].endswith("\n") else texts[0] + "\n"
    else:
        body = texts[1].split("\n")
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        indented = "\n".join(INDENT + line if line else line for line in body)
        code = f"{texts[0]}\n\n\n{module.function}\n{indented}\n"
    text = module_header(module, spec) + code
    try:
        compile(text, module.name, "exec")
    except SyntaxError as err:
        message = f"{module.name}: the generated module does not compile: {err}"
        raise ExtractionError(message) from err
    return text


def build_modules(spec: Spec, notebook_bytes: bytes, notebook_path: Path) -> dict[str, bytes]:
    """Check the notebook against the spec and build every module in memory.

    Args:
        spec: The parsed spec.
        notebook_bytes: Raw bytes of the notebook.
        notebook_path: Its path, used in messages.

    Returns:
        UTF-8 module bytes keyed by file name, in spec order.

    Raises:
        ExtractionError: If the notebook's sha256 differs from the spec's, it is not a JSON
            object, or a module cannot be built (``build_module``).
    """
    digest = hashlib.sha256(notebook_bytes).hexdigest()
    if digest != spec.sha256:
        raise ExtractionError(
            f"{notebook_path} has sha256 {digest}, the spec expects {spec.sha256}; a changed "
            "notebook needs a reviewed spec change"
        )
    try:
        notebook = json.loads(notebook_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ExtractionError(f"{notebook_path} is not valid JSON: {err}") from err
    return {
        module.name: build_module(_cell_source(notebook, module), module, spec).encode("utf-8")
        for module in spec.modules
    }


def compare(out: Path, modules: dict[str, bytes]) -> list[str]:
    """List every way the output folder differs from what the spec produces.

    Args:
        out: The output folder, for example ``arc26/nvarc``.
        modules: The expected module bytes (``build_modules``).

    Returns:
        One message per missing or differing module and per other ``.py`` file in ``out``
        (except ``__init__.py``); empty when the folder matches.
    """
    problems = []
    for name, data in modules.items():
        path = out / name
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != data:
            problems.append(f"{path}: differs from the spec's output")
    if out.is_dir():
        expected = set(modules) | {"__init__.py"}
        problems += [
            f"{path}: not produced by the spec"
            for path in sorted(out.glob("*.py"))
            if path.name not in expected
        ]
    return problems


def write_modules(out: Path, modules: dict[str, bytes]) -> None:
    """Write the modules into the output folder, creating it if needed.

    Args:
        out: The output folder.
        modules: Module bytes keyed by file name (``build_modules``).

    Raises:
        ExtractionError: If a file cannot be written.
    """
    try:
        out.mkdir(parents=True, exist_ok=True)
        for name, data in modules.items():
            (out / name).write_bytes(data)
    except OSError as err:
        raise ExtractionError(f"cannot write the modules to {out}: {err}") from err


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="extraction spec (configs/extraction.yaml)"
    )
    parser.add_argument(
        "--notebook", help="the reference notebook (default: the spec's notebook, repo-relative)"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (arc26/nvarc)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if a module differs from what the spec produces",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the moved modules, or check them against the spec.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success (``--check``: every module matches), 1 when ``--check`` finds a
        difference, 2 when the spec or the notebook is refused or a write fails.
    """
    args = build_parser().parse_args(argv)
    config, out = Path(args.config), Path(args.out)
    try:
        spec = parse_spec(_read(config, "spec"), config)
        notebook = Path(args.notebook) if args.notebook else REPO / spec.notebook
        modules = build_modules(spec, _read(notebook, "notebook"), notebook)
        if args.check:
            problems = compare(out, modules)
            for problem in problems:
                print(problem, file=sys.stderr)
            if problems:
                print(
                    f"check failed: edit {SPEC_NAME}, not the modules, and rerun {SCRIPT_NAME}",
                    file=sys.stderr,
                )
                return 1
            print(f"check passed: {len(modules)} modules in {out} match {config}")
            return 0
        write_modules(out, modules)
    except ExtractionError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print(f"wrote {', '.join(str(out / name) for name in modules)}")
    for problem in compare(out, modules):
        print(f"warning: {problem}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
