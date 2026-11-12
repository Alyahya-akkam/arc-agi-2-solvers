"""Acceptance tests for scripts/build_reference_run.py (task P0.1-01, deliverable 3).

The builder runs on a synthetic pulled notebook (with ``metadata.kaggle``, a ``%%writefile`` cell
holding path literals, and a ``!python starter.py`` cell) and a synthetic pulled
``kernel-metadata.json`` (with ``id_no``, a foreign ``id``, a ``docker_image``, and versioned
kernel sources), all under ``tmp_path``. Its output folder is then pushed through
``scripts/kaggle_ops.py`` with the gateway's seams replaced as in ``tests/test_kaggle_ops.py``.
No test starts a process, opens a connection, or reads ``~/.kaggle/`` or ``~/.config/arc26/``.
"""

import ast
import bz2
import importlib.util
import io
import json
import os
import pickle
import re
import socket
import subprocess
import tokenize
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "scripts" / "build_reference_run.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ops = load("kaggle_ops", REPO / "scripts" / "kaggle_ops.py")


@pytest.fixture(scope="module")
def builder():
    """The builder, loaded by file path like the gateway; a missing or broken script fails these
    tests one by one instead of stopping collection of the whole suite."""
    return load("build_reference_run", BUILDER)


OWNER = "tester"
SLUG = "arc26-p0-1-reference"
RUN_ID = "P0.1-01"
COMMIT = "0123abcd4567"
COMPETITION = "arc-prize-2026-arc-agi-2"
OUTPUTS = ("reference-run.ipynb", "kernel-metadata.json", "patch.diff")
DROP = object()  # a field to leave out
TODO = "TODO(orchestrator): set from the pulled notebook and the K0 listing"
# Config keys. The brief names `old` and `new`; these are this file's names for the competition
# slug, the replacement list, each entry's expected match count, the DEV4 ids, and the hash seed.
COMPETITION_KEY = "competition"
REPLACEMENTS_KEY = "replacements"
COUNT_KEY = "count"
DEV4_KEY = "dev4"
SEED_KEY = "hash_seed"
DEV4 = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
# Mount literals of the 2025 notebook and their 2026 replacements (research note, section 1.3).
DATA_OLD = "/kaggle/input/arc-prize-2025/"
DATA_NEW = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/"
MODEL_OLD = "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
MODEL_NEW = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
PAIRS = [(DATA_OLD, DATA_NEW), (MODEL_OLD, MODEL_NEW)]
DOCKER_IMAGE = "gcr.io/kaggle-gpu-images/python@sha256:" + "5f2c" * 16
SEED = re.compile(r"""os\.environ\[\s*["']PYTHONHASHSEED["']\s*\]\s*=\s*["']0["']""")
# Top-level notebook metadata as Kaggle saves it (research note, section 1.3).
KAGGLE_SAVED = {
    "accelerator": "nvidiaL4",
    "dataSources": [
        {"sourceId": 133469, "sourceType": "competition"},
        {"modelInstanceId": 472720, "sourceId": 627758, "sourceType": "modelInstanceVersion"},
        {"sourceId": 272622465, "sourceType": "kernelVersion"},
    ],
    "dockerImageVersionId": 31090,
    "isGpuEnabled": True,
    "isInternetEnabled": False,
    "language": "python",
    "sourceType": "notebook",
}


def rows(*lines):
    """nbformat's list form of a cell source: every line but the last ends with a newline."""
    return [f"{line}\n" for line in lines[:-1]] + list(lines[-1:])


def cell(kind, *lines, metadata=None, execution_count=None, outputs=()):
    doc = {"cell_type": kind, "metadata": metadata or {}, "source": rows(*lines)}
    if kind == "code":
        doc["execution_count"] = execution_count
        doc["outputs"] = list(outputs)
    return doc


def saved(n):
    """Cell metadata as Kaggle saves it."""
    return {"_cell_guid": f"guid-{n}", "_uuid": f"uuid-{n}", "trusted": True}


def stream(output):
    return {"name": "stdout", "output_type": "stream", "text": [output]}


def reference_cells():
    """Cells of the pulled notebook. In code cells each configured literal matches twice; prose,
    the raw cell, and unconfigured paths such as /kaggle/working must survive as written."""
    return [
        cell(
            "markdown",
            "# ARC Prize 2025, first place (reference)",
            "",
            f"Reads `{DATA_OLD}` and `{MODEL_OLD}`; the 2026 mounts differ → patched, café.",
        ),
        cell(
            "code",
            "!pip uninstall -y tensorflow",
            metadata=saved(1),
            execution_count=1,
            outputs=[stream("Successfully uninstalled tensorflow-2.18.0\n")],
        ),
        cell(
            "code",
            "%%writefile starter.py",
            "import os, sys",
            f'DATA = "{DATA_OLD}arc-agi_evaluation_challenges.json"',
            f'TEST = "{DATA_OLD}arc-agi_test_challenges.json"',
            f'MODEL = "{MODEL_OLD}"',
            'OUT = "/kaggle/working/submission.json"',
            f"DEV4 = {DEV4}",
            "print(DATA, TEST, MODEL, OUT, DEV4)",
            metadata=saved(2),
            execution_count=2,
            outputs=[stream("Writing starter.py\n")],
        ),
        cell("raw", f"Raw notes keep {DATA_OLD} and {MODEL_OLD} as written."),
        cell(
            "code",
            "import os  # list the attached inputs",
            "print(sorted(os.listdir('/kaggle/input')))",
            metadata=saved(4),
            execution_count=3,
        ),
        cell(
            "code",
            f"!python starter.py --model {MODEL_OLD} --end-time 41400",
            metadata=saved(5),
            execution_count=4,
        ),
        cell(
            "code",
            "import json  # read the submission back",
            "with open('/kaggle/working/submission.json') as f:",
            "    print(len(json.load(f)))",
            metadata=saved(6),
            execution_count=5,
        ),
        cell("markdown", "Scored in dev mode against `arc-agi_evaluation_solutions.json`."),
        cell("code", metadata=saved(8)),  # the empty cell Kaggle leaves at the end
    ]


def notebook(cells=None, **metadata):
    """The pulled notebook: Kaggle's ``metadata.kaggle`` beside the usual kernel metadata."""
    meta = {
        "kaggle": KAGGLE_SAVED,
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.13"},
    }
    meta.update(metadata)
    return {
        "cells": reference_cells() if cells is None else cells,
        "metadata": {key: value for key, value in meta.items() if value is not DROP},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def pulled_meta(**fields):
    """kernel-metadata.json as a pull with metadata writes it, updated with ``fields``. Every
    field the output must set differs here from the value the output needs."""
    meta = {
        "id": "sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue",
        "id_no": 245678901,
        "title": "ARC2 Qwen3 Unsloth Flash LoRA batch4 queue",
        "code_file": "arc2-qwen3-unsloth-flash-lora-batch4-queue.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": ["sorokin/arc-agi-2-extras/3"],
        "kernel_sources": [
            "sorokin/pip-install-unsloth-flash-patch/12",
            "sorokin/arc-grid-utils/4",
        ],
        "competition_sources": ["arc-prize-2025"],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
        "docker_image": DOCKER_IMAGE,
        "docker_image_pinning_type": "latest",
        "machine_shape": "NvidiaTeslaT4",
    }
    meta.update(fields)
    return {key: value for key, value in meta.items() if value is not DROP}


def replacement(old, new, count):
    return {"old": old, "new": new, COUNT_KEY: count}


def replacements():
    return [replacement(DATA_OLD, DATA_NEW, 2), replacement(MODEL_OLD, MODEL_NEW, 2)]


def config(entries=None, competition=COMPETITION):
    return {
        COMPETITION_KEY: competition,
        REPLACEMENTS_KEY: replacements() if entries is None else entries,
        DEV4_KEY: DEV4,
        SEED_KEY: 0,
    }


class Inputs(NamedTuple):
    notebook: Path
    metadata: Path
    config: Path
    out: Path


class Built(NamedTuple):
    notebook: dict
    meta: dict
    diff: str


def write_json(path, doc):
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_inputs(root, notebook_doc=None, meta=None, conf=None):
    """Write the builder's three inputs under ``root``; the output folder does not exist yet."""
    (root / "reference").mkdir(parents=True)
    (root / "configs").mkdir()
    paths = Inputs(
        root / "reference" / "reference_notebook.ipynb",
        root / "reference" / "kernel-metadata.json",
        root / "configs" / "reference_run.yaml",
        root / "dist" / "P0.1-01",
    )
    write_json(paths.notebook, notebook() if notebook_doc is None else notebook_doc)
    write_json(paths.metadata, pulled_meta() if meta is None else meta)
    paths.config.write_text(yaml.safe_dump(config() if conf is None else conf), encoding="utf-8")
    return paths


def run_build(builder, paths, owner=OWNER):
    builder.build(
        notebook=paths.notebook,
        metadata=paths.metadata,
        config=paths.config,
        owner=owner,
        out=paths.out,
    )


def argv(paths, owner=OWNER):
    return [
        "--notebook", str(paths.notebook),
        "--metadata", str(paths.metadata),
        "--config", str(paths.config),
        "--owner", owner,
        "--out", str(paths.out),
    ]  # fmt: skip


def outputs(out):
    return Built(
        read_json(out / "reference-run.ipynb"),
        read_json(out / "kernel-metadata.json"),
        (out / "patch.diff").read_text(encoding="utf-8"),
    )


def written(paths):
    return [name for name in OUTPUTS if (paths.out / name).exists()]


def text(doc):
    source = doc.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def patched(source, pairs=PAIRS):
    for old, new in pairs:
        source = source.replace(old, new)
    return source


def changes(doc):
    """Whether the fixture's replacements change this original cell: code cells only."""
    return doc["cell_type"] == "code" and patched(text(doc)) != text(doc)


def without_source(doc):
    return {key: value for key, value in doc.items() if key != "source"}


def assert_originals_kept(original, cells):
    """Every original cell stays, in order: equal, or for a patched cell equal apart from the
    configured literals in its source."""
    assert len(cells) == len(original)
    for index, (before, after) in enumerate(zip(original, cells, strict=True)):
        if changes(before):
            assert without_source(after) == without_source(before), f"original cell {index}"
            assert text(after) == patched(text(before)), f"original cell {index}"
        else:
            assert after == before, f"original cell {index}"


def slugify(title):
    """Kaggle's slug of a notebook title (python-slugify's defaults on ASCII text)."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower().replace("'", "")).strip("-")


def as_python(source):
    """A cell's source as IPython runs it: `!` lines and line magics become shell calls."""
    lines = []
    for line in source.split("\n"):
        body = line.lstrip()
        indent = line[: len(line) - len(body)]
        if body.startswith("!"):
            lines.append(f"{indent}get_ipython().system({body[1:]!r})")
        elif body.startswith("%") and not body.startswith("%%"):
            name, _, rest = body[1:].partition(" ")
            lines.append(f"{indent}get_ipython().run_line_magic({name!r}, {rest!r})")
        else:
            lines.append(line)
    return "\n".join(lines)


class Shell:
    """IPython's shell for cells run with exec: shell commands and magics do nothing."""

    def system(self, command):
        return None

    def getoutput(self, command):
        return []

    def run_line_magic(self, name, line):
        return None


def run_cell(doc, kaggle=None):
    """Run a code cell with exec; with ``kaggle``, its /kaggle paths point there instead."""
    source = as_python(text(doc))
    if kaggle is not None:
        for quote in ("'", '"'):
            source = source.replace(f"{quote}/kaggle", f"{quote}{kaggle}")
    exec(compile(source, "<cell>", "exec"), {"__name__": "__main__", "get_ipython": Shell})


def no_subprocess(*args, **kwargs):
    raise AssertionError(f"a test tried to start a process: {args}")


def no_network(*args, **kwargs):
    raise AssertionError(f"a test tried to open a connection: {args}")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No processes, no network, and a home directory without ~/.kaggle/ or ~/.config/arc26/."""
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def paths(tmp_path):
    return write_inputs(tmp_path / "in")


@pytest.fixture
def built(builder, paths):
    run_build(builder, paths)
    return outputs(paths.out)


# Brief check: kaggle metadata removed.


@pytest.mark.parametrize("kaggle", [KAGGLE_SAVED, {}, None], ids=["kaggle-saved", "empty", "null"])
def test_kaggle_metadata_removed(builder, tmp_path, kaggle):
    paths = write_inputs(tmp_path / "in", notebook_doc=notebook(kaggle=kaggle))
    run_build(builder, paths)
    doc = read_json(paths.out / "reference-run.ipynb")
    assert "kaggle" not in doc["metadata"]
    kept = {key: value for key, value in notebook()["metadata"].items() if key != "kaggle"}
    assert doc["metadata"] == kept
    assert doc["nbformat"] == 4


@pytest.mark.parametrize("metadata", [DROP, None], ids=["absent", "null"])
def test_notebook_without_top_level_metadata_is_built_or_refused(builder, tmp_path, metadata):
    doc = notebook()
    if metadata is DROP:
        del doc["metadata"]
    else:
        doc["metadata"] = metadata
    paths = write_inputs(tmp_path / "in", notebook_doc=doc)
    try:
        run_build(builder, paths)
    except builder.BuildError:
        return  # a refusal is fine; any other exception is not
    out = read_json(paths.out / "reference-run.ipynb")
    settings = out.get("metadata")
    assert not (isinstance(settings, dict) and "kaggle" in settings)
    assert_originals_kept(reference_cells(), out["cells"][2:-2])


# Brief check: the seed cell comes first, before the launch cell.


def test_seed_cell_comes_first_before_the_launch_cell(built):
    cells = built.notebook["cells"]
    kinds = [c["cell_type"] for c in cells]
    assert kinds[:2] == ["markdown", "code"]
    assert kinds.index("code") == 1
    assert SEED.search(text(cells[1]))
    launch = [
        i
        for i, c in enumerate(cells)
        if c["cell_type"] == "code" and text(c).lstrip().startswith("!python starter.py")
    ]
    assert launch == [7]  # original cell 5, after the two prepended cells


def test_seed_cell_sets_pythonhashseed_0(built, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    run_cell(built.notebook["cells"][1])
    assert os.environ["PYTHONHASHSEED"] == "0"


def test_seed_cell_comments_its_reason(built):
    source = as_python(text(built.notebook["cells"][1]))
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    assert [tok.string for tok in tokens if tok.type == tokenize.COMMENT]


# Brief checks: replacements applied only where configured, and a count mismatch refused.


def test_replacements_applied_only_where_configured(built):
    cells = built.notebook["cells"]
    assert len(cells) == len(reference_cells()) + 4
    for before, after in zip(reference_cells(), cells[2:-2], strict=True):
        expected = patched(text(before)) if before["cell_type"] == "code" else text(before)
        assert text(after) == expected
    code = "".join(text(c) for c in cells[2:-2] if c["cell_type"] == "code")
    assert [code.count(old) for old, _ in PAIRS] == [0, 0]
    assert [code.count(new) for _, new in PAIRS] == [2, 2]
    prose = "".join(text(c) for c in cells[2:-2] if c["cell_type"] != "code")
    assert [prose.count(old) for old, _ in PAIRS] == [2, 2]
    assert [prose.count(new) for _, new in PAIRS] == [0, 0]


@pytest.mark.parametrize(
    "entries",
    [
        [replacement(DATA_OLD, DATA_NEW, 1), replacement(MODEL_OLD, MODEL_NEW, 2)],
        [replacement(DATA_OLD, DATA_NEW, 2), replacement(MODEL_OLD, MODEL_NEW, 3)],
        [replacement(DATA_OLD, DATA_NEW, 2), replacement(MODEL_OLD, MODEL_NEW, 0)],
        [*replacements(), replacement("/kaggle/input/not-in-the-notebook/", DATA_NEW, 1)],
    ],
    ids=[
        "more-matches-than-expected",
        "fewer-matches-than-expected",
        "expected-none",
        "literal-absent",
    ],
)
def test_count_mismatch_refused(builder, tmp_path, entries):
    paths = write_inputs(tmp_path / "in", conf=config(entries))
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)


def test_literal_in_markdown_only_does_not_count(builder, tmp_path):
    prose = [cell("markdown", f"Inputs live in `{DATA_OLD}`."), cell("raw", DATA_OLD)]
    launch = cell("code", "!python starter.py")
    conf = config([replacement(DATA_OLD, DATA_NEW, 1)])
    paths = write_inputs(tmp_path / "refused", notebook_doc=notebook([*prose, launch]), conf=conf)
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)
    # Control: with one match in a code cell the same count holds and the prose stays as written.
    cells = [*prose, cell("code", f"DATA = '{DATA_OLD}'"), launch]
    paths = write_inputs(tmp_path / "control", notebook_doc=notebook(cells), conf=conf)
    run_build(builder, paths)
    out = read_json(paths.out / "reference-run.ipynb")["cells"][2:-2]
    expected = [f"Inputs live in `{DATA_OLD}`.", DATA_OLD, f"DATA = '{DATA_NEW}'", text(launch)]
    assert [text(c) for c in out] == expected


def test_replacements_are_literal_not_regex(builder, tmp_path):
    cells = [
        cell("code", "A = '/kaggle/input/arc.prize/'"),
        cell("code", "B = '/kaggle/input/arcXprize/'"),
        cell("code", "!python starter.py"),
    ]
    conf = config([replacement("/kaggle/input/arc.prize/", "/kaggle/input/arc.prize.2026/", 1)])
    paths = write_inputs(tmp_path / "in", notebook_doc=notebook(cells), conf=conf)
    run_build(builder, paths)
    out = read_json(paths.out / "reference-run.ipynb")["cells"][2:-2]
    assert [text(c) for c in out] == [
        "A = '/kaggle/input/arc.prize.2026/'",
        "B = '/kaggle/input/arcXprize/'",
        "!python starter.py",
    ]


def todo_cases():
    cases = {}
    for marker_id, marker in (("bare", "TODO"), ("tagged", TODO)):
        for field in ("old", "new", COUNT_KEY):
            entry = replacement(DATA_OLD, DATA_NEW, 2)
            entry[field] = marker
            cases[f"{field}-{marker_id}"] = entry
    cases["every-field"] = replacement(TODO, TODO, TODO)
    cases["bare-string"] = "TODO"
    cases["todo-key"] = {"TODO": "data path, after the notebook is pulled"}
    return cases


TODO_CASES = todo_cases()


@pytest.mark.parametrize("entry", list(TODO_CASES.values()), ids=list(TODO_CASES))
def test_todo_replacement_refused(builder, tmp_path, entry):
    conf = config([entry, replacement(MODEL_OLD, MODEL_NEW, 2)])
    paths = write_inputs(tmp_path / "in", conf=conf)
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)


MALFORMED_ENTRIES = {
    "no-new": {"old": DATA_OLD, COUNT_KEY: 2},
    "null-new": replacement(DATA_OLD, None, 2),
    "no-count": {"old": DATA_OLD, "new": DATA_NEW},
    "empty-old": replacement("", DATA_NEW, 2),  # would match between every two characters
}


@pytest.mark.parametrize("entry", list(MALFORMED_ENTRIES.values()), ids=list(MALFORMED_ENTRIES))
def test_malformed_replacement_refused(builder, tmp_path, entry):
    conf = config([entry, replacement(MODEL_OLD, MODEL_NEW, 2)])
    paths = write_inputs(tmp_path / "in", conf=conf)
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)


def test_patch_diff_holds_every_changed_cell(built):
    removed, added, changed_rows, unchanged_rows = [], [], set(), set()
    for doc in reference_cells():
        lines = text(doc).splitlines()
        if not changes(doc):
            unchanged_rows.update(line for line in lines if line.strip())
            continue
        changed_rows.update(lines)
        for line in lines:
            if patched(line) != line:
                removed.append(line)
                added.append(patched(line))
    diff = built.diff.splitlines()
    minus = [row[1:] for row in diff if row.startswith("-") and not row.startswith("---")]
    plus = [row[1:] for row in diff if row.startswith("+") and not row.startswith("+++")]
    assert any(row.startswith("@@") for row in diff)
    assert sorted(minus) == sorted(removed)
    assert [line for line in added if line not in plus] == []
    assert [row for row in plus if any(old in row for old, _ in PAIRS)] == []
    assert [line for line in unchanged_rows - changed_rows if line in built.diff] == []


# Brief check: all other cells byte-identical.


def test_all_other_cells_byte_identical(built):
    cells = built.notebook["cells"]
    assert len(cells) == len(reference_cells()) + 4
    assert_originals_kept(reference_cells(), cells[2:-2])


def test_string_source_cells_stay_byte_identical(builder, tmp_path):
    cells = reference_cells()
    for doc in cells:
        doc["source"] = text(doc)  # the single-string form nbformat also allows
    paths = write_inputs(tmp_path / "in", notebook_doc=notebook(cells))
    run_build(builder, paths)
    assert_originals_kept(cells, read_json(paths.out / "reference-run.ipynb")["cells"][2:-2])


# Brief check: the dump cell last.


def test_dump_cell_last(built):
    cells = built.notebook["cells"]
    assert [c["cell_type"] for c in cells[-2:]] == ["markdown", "code"]
    source = text(cells[-1])
    names = [
        "inference_outputs",
        "run_logs",
        "candidates.json",
        "pip_freeze.txt",
        "env.json",
        "KAGGLE_KERNEL_RUN_TYPE",
        "nvidia-smi",
    ]
    assert [name for name in names if name not in source] == []
    tree = ast.parse(as_python(source))
    assert any(isinstance(node, (ast.Try, ast.TryStar)) for node in ast.walk(tree))


def write_unexpected_inference_outputs(root):
    """Inference outputs that fit no reading of the documented format, so every parse fails."""
    (root / "0934a4d8_0").mkdir(parents=True)
    (root / "0934a4d8_0" / "0").write_bytes(b"not bz2")
    (root / "36a08778_0").write_bytes(bz2.compress(b"not a pickle"))
    (root / "981571dc_0").write_bytes(bz2.compress(pickle.dumps(["not", "a", "dict"])))
    (root / "aa4ec2a5_0").write_bytes(bz2.compress(pickle.dumps({"beam_score": None})))
    (root / "aa4ec2a5_1").write_bytes(b"")


def missing_tool(*args, **kwargs):
    raise FileNotFoundError(2, "no such tool in tests", str(args[0]) if args else None)


@pytest.mark.parametrize("inference", ["absent", "unexpected"])
def test_dump_cell_never_fails_the_notebook(built, tmp_path, monkeypatch, inference):
    kaggle = tmp_path / "kaggle"
    (kaggle / "working").mkdir(parents=True)
    if inference == "unexpected":
        write_unexpected_inference_outputs(kaggle / "inference_outputs")
    monkeypatch.chdir(kaggle / "working")
    # Still no process starts: pip and nvidia-smi fail as when missing from the image.
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    monkeypatch.setattr(subprocess, "Popen", missing_tool)
    monkeypatch.setattr(os, "system", lambda command: 127)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    existed = Path("/kaggle").exists()
    run_cell(built.notebook["cells"][-1], kaggle=kaggle)  # raises if the cell raises
    assert Path("/kaggle").exists() == existed


def test_added_cells_are_well_formed(built):
    cells = built.notebook["cells"]
    for doc in (cells[0], cells[1], cells[-2], cells[-1]):
        assert isinstance(doc["metadata"], dict)
        assert isinstance(doc["source"], (str, list)) and text(doc).strip()
        if doc["cell_type"] == "code":
            assert doc["outputs"] == [] and doc["execution_count"] is None
            compile(as_python(text(doc)), "<cell>", "exec")


# Brief checks: kernel-metadata.json, with id, id_no, and title not copied.


def test_kernel_metadata_fields_match_the_brief(built):
    pulled = pulled_meta()
    expected = {
        "id": f"{OWNER}/{SLUG}",
        "code_file": "reference-run.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "machine_shape": "NvidiaL4",
        "docker_image_pinning_type": "original",
        "docker_image": DOCKER_IMAGE,
        "competition_sources": [COMPETITION],
        "model_sources": pulled["model_sources"],
        "dataset_sources": pulled["dataset_sources"],
        "kernel_sources": pulled["kernel_sources"],
    }
    assert {key: built.meta.get(key) for key in expected} == expected
    assert built.meta["is_private"] is True and built.meta["enable_internet"] is False


def test_id_id_no_and_title_not_copied(built):
    pulled = pulled_meta()
    assert built.meta["id"] == f"{OWNER}/{SLUG}"
    assert "id_no" not in built.meta
    assert built.meta["title"] != pulled["title"]
    assert slugify(built.meta["title"]) == SLUG
    dumped = json.dumps(built.meta)
    assert pulled["id"] not in dumped and str(pulled["id_no"]) not in dumped


def test_competition_sources_come_from_the_config(builder, tmp_path):
    paths = write_inputs(tmp_path / "in", conf=config(competition="arc-prize-2026-dry-run"))
    run_build(builder, paths)
    meta = read_json(paths.out / "kernel-metadata.json")
    assert meta["competition_sources"] == ["arc-prize-2026-dry-run"]


@pytest.mark.parametrize("image", [DROP, None, ""], ids=["absent", "null", "empty"])
def test_refusal_without_docker_image(builder, tmp_path, image):
    paths = write_inputs(tmp_path / "in", meta=pulled_meta(docker_image=image))
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)


UNVERSIONED = {
    "owner-slug": ["sorokin/pip-install-unsloth-flash-patch"],
    "trailing-slash": ["sorokin/pip-install-unsloth-flash-patch/"],
    "one-of-two": ["sorokin/pip-install-unsloth-flash-patch/12", "sorokin/arc-grid-utils"],
}


@pytest.mark.parametrize("sources", list(UNVERSIONED.values()), ids=list(UNVERSIONED))
def test_refusal_with_an_unversioned_kernel_source(builder, tmp_path, sources):
    paths = write_inputs(tmp_path / "in", meta=pulled_meta(kernel_sources=sources))
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)


# Brief check: the output passes the gateway's checks for an accelerated push with a matching
# brief. Seams replaced as in tests/test_kaggle_ops.py; brief rules from docs/prerun/README.md.


def approved_brief():
    """docs/prerun/P0.1-01.md as approved: the TEMPLATE.md header and a signed Approval block."""
    header = {
        "Run-id": RUN_ID,
        "Revision": "1",
        "Task": "P0.1 reference capture",
        "Title": "reference notebook on DEV4, runs A and B",
        "Kind": "evaluation",
        "Decides": "the P0.1 parity fixture and its acceptance rule (conflict 5)",
        "Code-commit": COMMIT,
        "Config": "configs/reference_run.yaml",
        "Kaggle notebook": f"{OWNER}/{SLUG}",
        "Accelerator": "NvidiaL4",
        "Timeout": "3600",
        "Internet": "off",
        "GPU hours": "2",
        "Quota left after": "18",
        "Previous run": "none",
    }
    lines = [f"{key + ':':<18}{value}" for key, value in header.items()]
    approval = ["## 14. Approval", "", "```", "Approved-by: tester, 2026-09-30", "Notes:", "```"]
    return "\n".join([f"# Pre-run brief: {RUN_ID}", "", "```", *lines, "```", "", *approval, ""])


class Gateway:
    """kaggle_ops.py in a temporary repository whose docs/prerun/ holds the approved brief, with
    the owner set, git clean at the approved commit, and a kaggle CLI that records its arguments."""

    def __init__(self, tmp_path, monkeypatch):
        prerun = tmp_path / "repo" / "docs" / "prerun"
        prerun.mkdir(parents=True)
        (prerun / f"{RUN_ID}.md").write_text(approved_brief(), encoding="utf-8")
        self.calls = []
        monkeypatch.setattr(ops, "REPO", tmp_path / "repo")
        monkeypatch.setattr(ops, "PRERUN_DIR", prerun)
        monkeypatch.setattr(ops, "KAGGLE_OWNER", OWNER)
        monkeypatch.setattr(ops, "_git", self.git)
        monkeypatch.setattr(ops, "_kaggle", self.kaggle)

    def git(self, *args):
        # Only the approved commit is unchanged, so a misread Code-commit fails the push.
        if args[:1] == ("diff",) and not any(COMMIT in arg for arg in args):
            return "arc26/other.py\n"
        return ""

    def kaggle(self, *args):
        self.calls.append(list(args))
        return 0

    def push(self, folder, *extra):
        return ops.main(["push", str(folder), "--run-id", RUN_ID, *extra])


def accelerated_push(folder):
    folder = str(folder.resolve())
    return ["kernels", "push", "-p", folder, "--accelerator", "NvidiaL4", "-t", "3600"]


# enable_gpu and enable_tpu as a pulled file may carry them: from a GPU version, from a version
# saved on CPU, or not at all.
PULLED_FLAGS = {
    "gpu-version": {},
    "cpu-version": {"enable_gpu": False},
    "no-flags": {"enable_gpu": DROP, "enable_tpu": DROP},
}


@pytest.mark.parametrize(
    "extra", [[], ["--accelerator", "NvidiaL4"]], ids=["machine-shape", "accelerator-flag"]
)
@pytest.mark.parametrize("flags", list(PULLED_FLAGS.values()), ids=list(PULLED_FLAGS))
def test_output_passes_the_gateway_for_an_accelerated_push_with_a_matching_brief(
    builder, tmp_path, monkeypatch, flags, extra
):
    paths = write_inputs(tmp_path / "in", meta=pulled_meta(**flags))
    run_build(builder, paths)
    gateway = Gateway(tmp_path, monkeypatch)
    assert gateway.push(paths.out, *extra) == 0  # run A
    assert gateway.push(paths.out, *extra) == 0  # run B: same folder, same brief
    assert gateway.calls == [accelerated_push(paths.out)] * 2


def test_gateway_refuses_the_built_folder_with_the_unpatched_notebook(
    builder, paths, tmp_path, monkeypatch
):
    run_build(builder, paths)
    gateway = Gateway(tmp_path, monkeypatch)
    assert gateway.push(paths.out) == 0
    (paths.out / "reference-run.ipynb").write_bytes(paths.notebook.read_bytes())
    assert gateway.push(paths.out) == 2  # metadata.kaggle is back
    assert gateway.calls == [accelerated_push(paths.out)]


# Refusals, inputs, and the command line.


REFUSALS = {
    "count-mismatch": {"conf": config([replacement(DATA_OLD, DATA_NEW, 1), *replacements()[1:]])},
    "todo-entry": {"conf": config([replacement(DATA_OLD, TODO, 2), *replacements()[1:]])},
    "no-docker-image": {"meta": pulled_meta(docker_image=DROP)},
    "unversioned-kernel-source": {
        "meta": pulled_meta(kernel_sources=["sorokin/pip-install-unsloth-flash-patch"])
    },
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_refused_build_writes_no_files(builder, tmp_path, case):
    paths = write_inputs(tmp_path / "in", **REFUSALS[case])
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)
    assert written(paths) == []


MALFORMED_INPUTS = {
    "notebook-missing": lambda p: p.notebook.unlink(),
    "notebook-not-json": lambda p: p.notebook.write_text('{"cells": [', encoding="utf-8"),
    "notebook-without-cells": lambda p: write_json(p.notebook, {"metadata": {}, "nbformat": 4}),
    "metadata-not-json": lambda p: p.metadata.write_text("{", encoding="utf-8"),
    "metadata-not-an-object": lambda p: write_json(p.metadata, [pulled_meta()]),
    "config-not-yaml": lambda p: p.config.write_text("replacements: [open\n", encoding="utf-8"),
    "config-not-a-mapping": lambda p: p.config.write_text(f"- {COMPETITION}\n", encoding="utf-8"),
}


@pytest.mark.parametrize("case", list(MALFORMED_INPUTS))
def test_malformed_input_is_refused(builder, tmp_path, case):
    paths = write_inputs(tmp_path / "in")
    MALFORMED_INPUTS[case](paths)
    with pytest.raises(builder.BuildError):
        run_build(builder, paths)
    assert builder.main(argv(paths)) == 2


def test_inputs_are_left_unchanged(builder, paths):
    before = {path: path.read_bytes() for path in paths[:3]}
    run_build(builder, paths)
    assert {path: path.read_bytes() for path in paths[:3]} == before


def test_rebuild_is_byte_identical(builder, paths, tmp_path):
    again = paths._replace(out=tmp_path / "again" / "P0.1-01")
    run_build(builder, paths)
    run_build(builder, again)
    first = {name: (paths.out / name).read_bytes() for name in OUTPUTS}
    assert {name: (again.out / name).read_bytes() for name in OUTPUTS} == first


def test_main_returns_0_and_writes_the_three_files(builder, paths):
    assert builder.main(argv(paths, owner="akkam")) == 0
    assert written(paths) == list(OUTPUTS)
    assert read_json(paths.out / "kernel-metadata.json")["id"] == f"akkam/{SLUG}"


@pytest.mark.parametrize("case", list(REFUSALS))
def test_main_returns_2_on_refusal(builder, tmp_path, case):
    paths = write_inputs(tmp_path / "in", **REFUSALS[case])
    assert builder.main(argv(paths)) == 2
    assert written(paths) == []


# Fix round 1, item 4 (review of a4d20a8): TODO anywhere in the config's text refuses the build,
# comments included, although a comment changes no parsed value.

CONFIG_COMMENT_PLACES = {
    "own-first-line": lambda text, comment: f"{comment}\n{text}",
    "end-of-first-line": lambda text, comment: text.replace("\n", f"  {comment}\n", 1),
    "in-replacements": lambda text, comment: text.replace(
        f"{REPLACEMENTS_KEY}:\n", f"{REPLACEMENTS_KEY}:\n  {comment}\n", 1
    ),
    "last-line": lambda text, comment: f"{text}{comment}\n",
}
TODO_COMMENTS = {
    "bare": "# TODO",
    "tagged": "# TODO(orchestrator): cell and line of the pulled notebook, K0 listing line",
}


def comment_config(paths, place, comment):
    """Put ``comment`` into the config at ``place``; every parsed value stays the same."""
    text = paths.config.read_text(encoding="utf-8")
    commented = CONFIG_COMMENT_PLACES[place](text, comment)
    assert commented != text
    assert yaml.safe_load(commented) == yaml.safe_load(text)
    paths.config.write_text(commented, encoding="utf-8")


@pytest.mark.parametrize("comment", list(TODO_COMMENTS.values()), ids=list(TODO_COMMENTS))
@pytest.mark.parametrize("place", list(CONFIG_COMMENT_PLACES))
def test_todo_in_a_config_comment_refused(builder, paths, place, comment):
    comment_config(paths, place, comment)
    with pytest.raises(builder.BuildError, match="TODO"):
        run_build(builder, paths)
    assert written(paths) == []
    assert builder.main(argv(paths)) == 2
    assert written(paths) == []


def test_todo_refusal_names_every_marked_line(builder, paths):
    comment_config(paths, "own-first-line", "# TODO")
    comment_config(paths, "last-line", "# TODO")
    last = len(paths.config.read_text(encoding="utf-8").splitlines())
    with pytest.raises(builder.BuildError, match=rf"\blines 1, {last}\b"):
        run_build(builder, paths)


@pytest.mark.parametrize("place", list(CONFIG_COMMENT_PLACES))
def test_config_comment_without_todo_builds(builder, paths, place):
    comment_config(paths, place, "# cell 2, line 3 of the pulled notebook; K0 listing line 7")
    run_build(builder, paths)
    assert written(paths) == list(OUTPUTS)


# Install-notebook version pins in the config (30 Sep): the pulled metadata attaches
# sorokin/pip-install-unsloth-flash-patch without a version; the config pins the run the reference
# recorded, and the committed metadata stays verbatim.


def test_pin_versions_an_unversioned_source(builder):
    pins = {"someone/install-notebook": 2}
    assert builder.pin_kernel_sources(["someone/install-notebook"], pins) == [
        "someone/install-notebook/2"
    ]


def test_pin_keeps_a_versioned_source(builder):
    sources = ["someone/install-notebook/3"]
    assert builder.pin_kernel_sources(sources, {}) == sources


@pytest.mark.parametrize(
    "sources, pins",
    [
        (["someone/install-notebook"], {}),
        (["someone/install-notebook/3"], {"someone/install-notebook": 2}),
        (["someone/install-notebook"], {"someone/install-notebook": 2, "other/notebook-x": 1}),
    ],
    ids=["no-version-no-pin", "pin-contradicts-pull", "stale-pin"],
)
def test_pin_refuses(builder, sources, pins):
    with pytest.raises(builder.BuildError):
        builder.pin_kernel_sources(sources, pins)


@pytest.mark.parametrize(
    "pins",
    ["x", {"someone/install-notebook": 0}, {"someone/install-notebook": "2"}, {"bad": 2}],
    ids=["not-a-mapping", "zero", "string-version", "no-slug"],
)
def test_malformed_kernel_source_versions_refused(builder, pins):
    settings = {
        "competition": "arc-prize-2026-arc-agi-2",
        "hash_seed": 0,
        "replacements": [],
        "kernel_source_versions": pins,
    }
    with pytest.raises(builder.BuildError, match="kernel_source_versions"):
        builder.parse_config(yaml.safe_dump(settings).encode(), Path("c.yaml"))


def test_config_pin_reaches_the_built_metadata(builder, tmp_path):
    settings = {
        "competition": "arc-prize-2026-arc-agi-2",
        "hash_seed": 0,
        "replacements": [],
        "kernel_source_versions": {"someone/install-notebook": 2},
    }
    config = builder.parse_config(yaml.safe_dump(settings).encode(), Path("c.yaml"))
    pulled = {
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:" + "0" * 64,
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
        "kernel_sources": ["someone/install-notebook"],
    }
    pins = config.kernel_source_versions
    meta = builder.kernel_metadata(pulled, "tester", config.competition, pins)
    assert meta["kernel_sources"] == ["someone/install-notebook/2"]
    assert meta["is_private"] is True
