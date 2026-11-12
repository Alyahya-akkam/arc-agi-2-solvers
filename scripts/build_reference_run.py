r"""Build the P0.1-01 reference capture notebook from the pulled reference (BRIEF B5 P0.1, A8).

Purpose: turn the 2025 first-place notebook, as the human pulled it with its metadata, into a
private notebook that runs on 4xL4 in the 2026 Kaggle environment with its method unchanged, so
its DEV4 attempt grids can become the P0.1 parity fixture. Task P0.1-01; pre-run brief
``docs/prerun/P0.1-01.md``; flagged conflict 5 in ``docs/STATE.md`` (parity with a
nondeterministic reference); ADR-008 (the push goes through ``kaggle_ops.py push``); runbook
``docs/runbooks/build_reference_run.md``.

Inputs, all given on the command line: the pulled ``.ipynb``, the pulled ``kernel-metadata.json``,
and ``configs/reference_run.yaml``. No other input is read: no git, no network, no credentials.
Outputs, in ``--out``: ``reference-run.ipynb``, ``kernel-metadata.json``, and ``patch.diff``,
which together form the folder ``kaggle_ops.py push`` sends.

Method: every original cell is copied unchanged and in order, except for literal path
replacements listed in the config. Each entry states how many matches it must find in code
cells, so a notebook that differs from the one the config was written for is refused. The
notebook's top-level ``metadata.kaggle`` (Kaggle editor settings, refused by the gateway) is
dropped; ``kernel-metadata.json`` sets those explicitly instead. One prepended cell fixes
``PYTHONHASHSEED`` for the processes the reference launches, which removes the salted ``hash()``
from its rescoring seeds. One appended cell (``dump_run_logs``, embedded verbatim) saves the
per-candidate outputs the reference leaves outside ``/kaggle/working``, together with the
software environment. ``kernel-metadata.json`` is written from scratch. Identity comes from
``--owner``; accelerator, internet, and image pinning are fixed here. Only the image and the
model, dataset, and pinned notebook sources are copied from the pulled file. ``--smoke`` runs the
build and both added cells on CPU with synthetic inputs.

Usage::

    python scripts/build_reference_run.py --notebook reference/reference_notebook.ipynb \
        --metadata reference/kernel-metadata.json --config configs/reference_run.yaml \
        --owner <username> --out dist/P0.1-01
    python scripts/build_reference_run.py --smoke
"""

from __future__ import annotations

import argparse
import bz2
import difflib
import hashlib
import inspect
import json
import os
import pickle
import pprint
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import yaml

# The title slugifies to the slug (python-slugify, which kaggle-cli 2.2.4 uses for its check),
# so the CLI prints no title mismatch warning.
SLUG = "arc26-p0-1-reference"
TITLE = "arc26 P0.1 reference"
NOTEBOOK_FILE = "reference-run.ipynb"
METADATA_FILE = "kernel-metadata.json"
DIFF_FILE = "patch.diff"
# 4xL4: the reference runs one worker per GPU on four L4s, the hardware of the scored rerun.
ACCELERATOR = "NvidiaL4"
# Top-level notebook metadata key under which Kaggle's editor saves accelerator and internet.
SETTINGS_KEY = "kaggle"
# The reference's workers write candidate pickles here, outside /kaggle/working, which is the
# only folder Kaggle keeps after the session.
INFERENCE_DIR = "/kaggle/inference_outputs"
RUN_LOGS_DIR = "/kaggle/working/run_logs"
# Cap on any pickle the appended cell reads and any log it writes: pulls stay small for the
# human's connection, well under the gateway's 50 MB warning (MAX_PULL_FILE_BYTES).
MAX_LOG_FILE_BYTES = 20 * 1024**2
# Cells added before the first original cell: original cell i becomes built cell i + PREPENDED.
PREPENDED = 2
# Python accepts PYTHONHASHSEED values from 0 to 2**32 - 1.
MAX_HASH_SEED = 2**32 - 1
ENTRY_KEYS = frozenset({"old", "new", "count"})
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")
COMPETITION_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# owner/slug/version. kaggle-cli accepts an optional version and a slug of at least five
# characters; the version is required here so the install notebook, and with it the package
# stack, is pinned.
VERSIONED_KERNEL = re.compile(r"[^/\s]+/[^/\s]{5,}/[0-9]+")
UNVERSIONED_KERNEL = re.compile(r"[^/\s]+/[^/\s]{5,}")

SEED_NOTE = """\
### Added for P0.1-01: fixed hash seed

The next cell sets `PYTHONHASHSEED` before any reference cell runs. The reference seeds its \
rescoring augmentations with Python's salted `hash()` and fixes no hash seed, so two runs of the \
same notebook rescore with different augmentations (conflict 5 in `docs/STATE.md`). The variable \
reaches `!python starter.py` and the four workers it spawns, the processes that run the method. \
Every original cell follows unchanged, except the path literals listed in `patch.diff`."""

DUMP_NOTE = """\
### Added for P0.1-01: run logs

The next cell runs after every reference cell. It converts the candidate pickles the reference \
writes to `/kaggle/inference_outputs/`, which Kaggle does not keep, into \
`/kaggle/working/run_logs/candidates.json` (per test output: each candidate's `solution` grid, \
`beam_score`, and `score_aug`). It also writes `pip_freeze.txt` and `env.json` (Python, image \
build, GPU names and driver, `KAGGLE_KERNEL_RUN_TYPE`, package versions, build provenance). Files \
over 20 MB are skipped and listed in `env.json`. Every error is caught and printed, so this cell \
cannot fail the run."""


class BuildError(Exception):
    """The inputs break a rule of the reference build; no output file was written."""


class Replacement(NamedTuple):
    """One literal path replacement: ``old`` must occur ``count`` times across all code cells."""

    old: str
    new: str
    count: int


class Config(NamedTuple):
    """The build settings read from ``configs/reference_run.yaml``."""

    competition: str
    hash_seed: int
    replacements: tuple[Replacement, ...]
    kernel_source_versions: dict[str, int]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_id(value: str) -> bool:
    return bool(SAFE_ID.fullmatch(value)) and value not in {".", ".."}


def _read(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as err:
        raise BuildError(f"cannot read the {what} {path}: {err}") from err


def _json_object(data: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise BuildError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(value, dict):
        raise BuildError(f"{path} must hold a JSON object")
    return value


def _todo_paths(value: Any, where: str) -> list[str]:
    if isinstance(value, str):
        return [where or "the top level"] if "TODO" in value else []
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            path = f"{where}.{key}" if where else str(key)
            found += [path] if "TODO" in str(key) else _todo_paths(item, path)
        return found
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in _todo_paths(item, f"{where}[{i}]")]
    return []


def _replacement(entry: Any, where: str) -> Replacement:
    if not isinstance(entry, dict):
        raise BuildError(f"{where} must be a mapping with the keys old, new, count")
    keys = {str(key) for key in entry}
    if keys != ENTRY_KEYS:
        raise BuildError(
            f"{where} must have exactly the keys old, new, count; missing "
            f"{sorted(ENTRY_KEYS - keys)}, unknown {sorted(keys - ENTRY_KEYS)}"
        )
    for key in ("old", "new"):
        text = entry[key]
        if not isinstance(text, str) or not text or "\n" in text or "\r" in text:
            raise BuildError(f"{where}.{key} must be a non-empty single-line string")
    if entry["old"] == entry["new"]:
        raise BuildError(f"{where}: old and new are the same literal")
    count = entry["count"]
    if type(count) is not int or count < 1:
        raise BuildError(
            f"{where}.count must be a positive integer, the number of matches expected in code "
            f"cells; got {count!r}"
        )
    return Replacement(entry["old"], entry["new"], count)


def parse_config(data: bytes, path: Path) -> Config:
    """Parse and check the build config.

    Why: apart from the owner, every choice this build makes comes from the config, so the config
    is checked strictly. It is refused outright while a placeholder remains, since the path
    replacements stay TODO entries until the reference notebook is pulled and the K0 probe
    confirms the mounts. Comments count as well: each entry's comment must name the notebook line
    it patches and the K0 listing line that confirms it.

    Args:
        data: Raw bytes of ``configs/reference_run.yaml``.
        path: Its path, used in messages.

    Returns:
        The competition slug, the hash seed, the replacements in config order, and the version
        pins for unversioned kernel sources (empty when the key is absent).

    Raises:
        BuildError: If the text is not UTF-8 or contains ``TODO`` on any line, comments included;
            the YAML is invalid; a parsed key or value contains ``TODO`` (possible through a YAML
            escape); a required key is missing or has the wrong type; two ``old`` literals are
            equal or one contains the other (their counts would be ambiguous); or
            ``kernel_source_versions`` is not a mapping of ``<owner>/<slug>`` to a positive integer.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as err:
        raise BuildError(f"{path} is not UTF-8 text: {err}") from err
    marked = [str(n) for n, line in enumerate(text.splitlines(), start=1) if "TODO" in line]
    if marked:
        raise BuildError(
            f"{path} still has TODO on line{'s' if len(marked) > 1 else ''} {', '.join(marked)} "
            "(comments count); fill each entry and its source comment from the pulled notebook "
            "and the K0 listing before building"
        )
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise BuildError(f"{path} is not valid YAML: {err}") from err
    if not isinstance(raw, dict):
        raise BuildError(f"{path} must hold a mapping")
    # A YAML escape such as "\x54ODO" spells TODO in a parsed value but not in the text.
    todos = _todo_paths(raw, "")
    if todos:
        raise BuildError(
            f"{path} still has TODO entries ({', '.join(todos)}); fill them from the pulled "
            "notebook and the K0 listing before building"
        )
    competition = raw.get("competition")
    if not isinstance(competition, str) or not COMPETITION_SLUG.fullmatch(competition):
        raise BuildError(f"{path}: competition must be a Kaggle competition slug")
    hash_seed = raw.get("hash_seed")
    if type(hash_seed) is not int or not 0 <= hash_seed <= MAX_HASH_SEED:
        raise BuildError(f"{path}: hash_seed must be an integer from 0 to {MAX_HASH_SEED}")
    entries = raw.get("replacements")
    if not isinstance(entries, list):
        raise BuildError(f"{path}: replacements must be a list (empty when no path changes)")
    replacements = tuple(
        _replacement(entry, f"replacements[{i}]") for i, entry in enumerate(entries)
    )
    for i, first in enumerate(replacements):
        for j, second in enumerate(replacements):
            if i != j and first.old in second.old:
                raise BuildError(
                    f"{path}: replacements[{i}].old occurs inside replacements[{j}].old, so their "
                    "match counts are ambiguous; make each old literal distinct"
                )
    pins = raw.get("kernel_source_versions", {})
    if not isinstance(pins, dict) or not all(
        isinstance(name, str)
        and UNVERSIONED_KERNEL.fullmatch(name)
        and type(version) is int
        and version >= 1
        for name, version in pins.items()
    ):
        raise BuildError(
            f"{path}: kernel_source_versions must map <owner>/<slug> to a version number >= 1"
        )
    return Config(competition, hash_seed, replacements, dict(pins))


def _sources(pulled: dict[str, Any], key: str) -> list[str]:
    value = pulled.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
        raise BuildError(f"the pulled {key} must be a list of source names; got {value!r}")
    return list(value)


def pin_kernel_sources(sources: list[str], pins: dict[str, int]) -> list[str]:
    """Give every kernel source a version, from the pulled name or from the config's pins.

    Why: the Kaggle CLI's metadata pull writes attached notebooks without a version, while the
    notebook records the exact run it used; the config pins that run's version number so the
    committed metadata stays verbatim.

    Args:
        sources: Kernel sources as pulled, ``<owner>/<slug>`` or ``<owner>/<slug>/<version>``.
        pins: Version numbers keyed by ``<owner>/<slug>``, from ``kernel_source_versions``.

    Returns:
        The sources, each ``<owner>/<slug>/<version>``, in pulled order.

    Raises:
        BuildError: If a source has no version and no pin, a pin contradicts a pulled version, or
            a pin names no pulled source.
    """
    pinned: list[str] = []
    unversioned: list[str] = []
    for source in sources:
        if VERSIONED_KERNEL.fullmatch(source):
            name, _, version = source.rpartition("/")
            if name in pins and pins[name] != int(version):
                raise BuildError(
                    f"kernel_source_versions pins {name} to {pins[name]}, but the pulled metadata "
                    f"attaches version {version}"
                )
            pinned.append(source)
        elif source in pins:
            pinned.append(f"{source}/{pins[source]}")
        else:
            unversioned.append(source)
    if unversioned:
        raise BuildError(
            f"kernel sources without a version: {unversioned}; each must be "
            "<owner>/<slug>/<version> so the package stack is pinned (pin it in the config's "
            "kernel_source_versions)"
        )
    names = {s.rpartition("/")[0] if VERSIONED_KERNEL.fullmatch(s) else s for s in sources}
    stale = sorted(set(pins) - names)
    if stale:
        raise BuildError(
            f"kernel_source_versions names sources the notebook does not attach: {stale}"
        )
    return pinned


def kernel_metadata(
    pulled: dict[str, Any], owner: str, competition: str, pins: dict[str, int] | None = None
) -> dict[str, Any]:
    """Write ``kernel-metadata.json`` from scratch, copying only what the run must inherit.

    Why: kaggle-cli pushes to ``id_no`` in preference to ``id``, so a copied identity could push
    to the reference author's notebook (ADR-008). The run needs the reference's Python 3.11 image
    (the current default image cannot import its cp311 packages) and its install notebook at the
    version it ran with, since that notebook's unpinned install resolves differently over time.

    Args:
        pulled: The reference's metadata as pulled with the notebook.
        owner: Kaggle username that owns the new notebook.
        competition: Slug of the competition to attach, from the config.
        pins: Version numbers for unversioned kernel sources (``pin_kernel_sources``).

    Returns:
        The metadata: private, internet off, 4xL4, the pulled image pinned, the competition from
        the config, and the pulled model, dataset, and versioned kernel sources. ``id``,
        ``id_no``, and ``title`` are never copied.

    Raises:
        BuildError: If the pulled file has no ``docker_image``, no model source, no kernel
            source, a kernel source without a version, or a source list that is not a list of
            names.
    """
    image = pulled.get("docker_image")
    if not isinstance(image, str) or not image.strip():
        raise BuildError(
            "the pulled kernel-metadata.json has no docker_image; the reference needs its original "
            "Python 3.11 image pinned, and the current default image cannot import its packages"
        )
    model_sources = _sources(pulled, "model_sources")
    if not model_sources:
        raise BuildError(
            "the pulled kernel-metadata.json has no model_sources; the reference loads its base "
            "model from an attached Kaggle model"
        )
    kernel_sources = _sources(pulled, "kernel_sources")
    if not kernel_sources:
        raise BuildError(
            "the pulled kernel-metadata.json has no kernel_sources; the reference installs "
            "nothing itself and needs its install notebook attached at a pinned version"
        )
    kernel_sources = pin_kernel_sources(kernel_sources, pins or {})
    return {
        "id": f"{owner}/{SLUG}",
        "title": TITLE,
        "code_file": NOTEBOOK_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "machine_shape": ACCELERATOR,
        "docker_image_pinning_type": "original",
        "docker_image": image,
        "competition_sources": [competition],
        "model_sources": model_sources,
        "dataset_sources": _sources(pulled, "dataset_sources"),
        "kernel_sources": kernel_sources,
    }


def _cells(notebook: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        raise BuildError(f"{path} is not an nbformat 4 notebook: no list of cells")
    for index, cell in enumerate(cells):
        source = cell.get("source", "")
        lines_ok = isinstance(source, list) and all(isinstance(line, str) for line in source)
        if cell.get("cell_type") == "code" and not (isinstance(source, str) or lines_ok):
            raise BuildError(f"{path}: code cell {index} has a source that is not text")
    return cells


def apply_replacements(
    cells: list[dict[str, Any]], replacements: Sequence[Replacement]
) -> tuple[list[dict[str, Any]], list[tuple[int, str, str]]]:
    """Apply the configured literal replacements to the sources of code cells.

    Why: the 2026 mounts differ from 2025 and nothing else in a cell may change. Each literal is
    therefore replaced where it occurs, in code cells only, in one pass, so a replacement's new
    text is never searched again. The build is refused unless every literal matched as often as
    the config says. A list source is patched element by element, so unchanged lines keep their
    exact form.

    Args:
        cells: The notebook's cells, in order; not modified.
        replacements: The config's entries.

    Returns:
        The cells, with changed code cells replaced by patched copies (other cells are the same
        objects), and for every changed cell its index and its source text before and after.

    Raises:
        BuildError: If a literal matched a different number of times than its ``count``, or its
            occurrences overlapped another literal's matches.
    """
    if not replacements:
        return list(cells), []
    by_old = {entry.old: entry for entry in replacements}
    pattern = re.compile("|".join(map(re.escape, sorted(by_old, key=len, reverse=True))))
    found = dict.fromkeys(by_old, 0)
    replaced = dict.fromkeys(by_old, 0)

    def swap(match: re.Match[str]) -> str:
        replaced[match.group(0)] += 1
        return by_old[match.group(0)].new

    patched: list[dict[str, Any]] = []
    changed: list[tuple[int, str, str]] = []
    for index, cell in enumerate(cells):
        source = cell.get("source", "")
        if cell.get("cell_type") != "code":
            patched.append(cell)
            continue
        parts = [source] if isinstance(source, str) else source
        for part in parts:
            for old in by_old:
                found[old] += part.count(old)
        new_parts = [pattern.sub(swap, part) for part in parts]
        if new_parts == parts:
            patched.append(cell)
            continue
        patched.append({**cell, "source": new_parts[0] if isinstance(source, str) else new_parts})
        changed.append((index, "".join(parts), "".join(new_parts)))
    problems = []
    for old, entry in by_old.items():
        if replaced[old] != found[old]:
            problems.append(
                f"{old!r} overlaps another literal ({found[old]} found, {replaced[old]} replaced)"
            )
        elif replaced[old] != entry.count:
            problems.append(
                f"{old!r} matches {replaced[old]} times in code cells, the config expects "
                f"{entry.count}"
            )
    if problems:
        raise BuildError("path replacements refused: " + "; ".join(problems))
    return patched, changed


def seed_cell_source(hash_seed: int) -> str:
    """Return the source of the prepended code cell that fixes ``PYTHONHASHSEED``.

    Why: the reference seeds rescoring with ``hash(bk) % 1024**2`` and fixes no hash seed
    (conflict 5). A variable set in the kernel before ``!python starter.py`` reaches that process
    and the workers it spawns, so no original cell has to change.

    Args:
        hash_seed: Value for ``PYTHONHASHSEED``, from the config.

    Returns:
        Python source that sets the variable, with a comment giving the reason.
    """
    return (
        "# Added by scripts/build_reference_run.py (P0.1-01). Conflict 5 in docs/STATE.md: the\n"
        "# reference seeds rescoring with Python's salted hash(), seed=hash(bk) % 1024**2, and\n"
        "# fixes no hash seed, so two runs rescore with different augmentations. The value set\n"
        "# here is inherited by `!python starter.py` and the workers it spawns, which start\n"
        "# later, so their hash() and the rescoring seeds repeat across runs. This kernel is\n"
        "# already running and keeps its own salt.\n"
        "import os\n"
        "\n"
        f'os.environ["PYTHONHASHSEED"] = "{hash_seed}"'
    )


def dump_run_logs(inference_dir: str, run_logs: str, max_file_bytes: int, build: dict) -> dict:
    """Save the reference's candidates and the software environment as small files.

    Why: the reference's workers write their candidates as pickles under ``inference_dir``, one
    file or folder per test output (subkey), outside ``/kaggle/working``. Kaggle drops them when
    the session ends, and they hold the scores behind every attempt grid. The built notebook's
    last cell runs this function verbatim (``build_reference_run.py`` embeds its source) after
    every original cell. It therefore imports what it needs itself and never raises: each step
    records its error and the next step still runs.

    Args:
        inference_dir: Folder of candidate pickles, bz2-compressed or plain; every file under it
            is read and keyed by its relative path.
        run_logs: Folder for ``candidates.json``, ``pip_freeze.txt``, and ``env.json``.
        max_file_bytes: A pickle or log file larger than this is skipped and the skip recorded.
        build: Provenance of the built notebook, copied into ``env.json``.

    Returns:
        The report also stored under ``"dump"`` in ``env.json``: bytes per file written, skipped
        files with their size and reason, and error messages.
    """
    import bz2
    import importlib.metadata
    import json
    import os
    import pickle
    import platform
    import shutil
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    report: dict = {"written": {}, "skipped": [], "errors": []}
    out = Path(run_logs)

    def plain(value: object) -> object:
        # numpy arrays and scalars and torch tensors all convert to Python values via tolist().
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, set | frozenset):
            value = sorted(value, key=repr)
        if isinstance(value, list | tuple):
            return [plain(item) for item in value]
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return repr(value)

    def skip(name: str, size: int) -> None:
        report["skipped"].append({"file": name, "bytes": size, "reason": "over the size cap"})

    def write(name: str, text: str) -> None:
        data = text.encode("utf-8")
        if len(data) > max_file_bytes:
            skip(name, len(data))
            return
        (out / name).write_bytes(data)
        report["written"][name] = len(data)

    try:
        out.mkdir(parents=True, exist_ok=True)
        candidates = {}
        root = Path(inference_dir)
        if not root.is_dir():
            report["errors"].append(f"{inference_dir} does not exist")
        for path in sorted(root.rglob("*")) if root.is_dir() else []:
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > max_file_bytes:
                skip(str(path), size)
                continue
            try:
                data = path.read_bytes()
                if data[:3] == b"BZh":
                    data = bz2.decompress(data)
                candidates[path.relative_to(root).as_posix()] = plain(pickle.loads(data))
            except Exception as err:
                report["errors"].append(f"{path}: {err!r}")
        # Compact: about 2 bytes per grid cell, so DEV4's candidates stay well under 1 MB.
        write("candidates.json", json.dumps(candidates, separators=(",", ":")))
    except Exception:
        report["errors"].append(traceback.format_exc())

    # `!python starter.py` resolves python on PATH, so its packages are the ones that matter.
    python = shutil.which("python") or sys.executable
    try:
        freeze = subprocess.run(
            [python, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=300
        )
        if freeze.returncode:
            report["errors"].append(f"pip freeze exit {freeze.returncode}: {freeze.stderr[-500:]}")
        write("pip_freeze.txt", freeze.stdout)
    except Exception:
        report["errors"].append(traceback.format_exc())

    gpus = None
    try:
        smi = shutil.which("nvidia-smi")
        if smi:
            query = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            fields = ("name", "driver_version", "memory_total")
            gpus = [
                dict(zip(fields, (part.strip() for part in line.split(",")), strict=False))
                for line in query.stdout.splitlines()
                if line.strip()
            ]
    except Exception:
        report["errors"].append(traceback.format_exc())

    packages = {}
    for name in (
        "torch", "transformers", "peft", "accelerate", "trl", "datasets", "unsloth",
        "unsloth_zoo", "xformers", "flash_attn", "triton", "bitsandbytes", "numpy",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:
            packages[name] = None

    # Kaggle's docker-python images keep their build commit and date in these files, so runs with
    # equal values ran the same image (unverified until run A; None when a file is absent).
    image = {}
    for name in ("/etc/git_commit", "/etc/build_date"):
        try:
            image[name] = Path(name).read_text(encoding="utf-8").strip()
        except Exception:
            image[name] = None

    try:
        variables = (
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_IS_COMPETITION_RERUN",
            "KAGGLE_DOCKER_IMAGE",
            "PYTHONHASHSEED",
        )
        env = {
            "python": sys.version,
            "executable": sys.executable,
            "python_on_path": python,
            "platform": platform.platform(),
            "image": image,
            "environment": {name: os.environ.get(name) for name in variables},
            "gpus": gpus,
            "packages": packages,
            "sys_path": list(sys.path),
            "build": build,
            "dump": report,
        }
        write("env.json", json.dumps(env, indent=1, default=str))
    except Exception:
        report["errors"].append(traceback.format_exc())

    print(
        f"run_logs: wrote {sorted(report['written'])}, skipped {len(report['skipped'])}, "
        f"errors {len(report['errors'])}"
    )
    for error in report["errors"]:
        print(error)
    return report


def dump_cell_source(build: dict[str, Any]) -> str:
    """Return the source of the appended code cell that runs ``dump_run_logs``.

    Why: the cell must be exactly the code the smoke path and tests run, so it embeds this
    module's ``dump_run_logs`` source verbatim (``inspect.getsource`` reads this script's own
    file) and calls it inside ``try``/``except``, so the cell cannot fail the run.

    Args:
        build: Provenance recorded in ``env.json`` (input hashes, hash seed, replacements).

    Returns:
        Python source for the cell; it compiles, which ``build`` checks.
    """
    literal = pprint.pformat(build, sort_dicts=False, width=96)
    return (
        "# Added by scripts/build_reference_run.py (P0.1-01): saves the candidates the reference\n"
        "# writes outside /kaggle/working, and the software environment, to run_logs/.\n"
        f"{inspect.getsource(dump_run_logs)}\n\n"
        f"ARC26_BUILD = {literal}\n\n"
        "try:\n"
        "    dump_run_logs(\n"
        f"        {INFERENCE_DIR!r}, {RUN_LOGS_DIR!r}, {MAX_LOG_FILE_BYTES}, ARC26_BUILD\n"
        "    )\n"
        "except Exception:\n"
        "    import traceback\n"
        "\n"
        "    traceback.print_exc()"
    )


def _new_cell(kind: str, text: str, as_lines: bool, cell_id: str | None) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": kind}
    if cell_id is not None:
        cell["id"] = cell_id
    cell["metadata"] = {}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cell["source"] = text.splitlines(keepends=True) if as_lines else text
    return cell


def _added_cells(
    notebook: dict[str, Any], cells: list[dict[str, Any]], hash_seed: int, build: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # New cells follow the notebook's own conventions: list or string sources, and cell ids when
    # the notebook is nbformat 4.5 or later or already has them.
    as_lines = any(isinstance(cell.get("source"), list) for cell in cells) or not cells
    minor = notebook.get("nbformat_minor")
    with_ids = (isinstance(minor, int) and minor >= 5) or any("id" in cell for cell in cells)
    names = ("arc26-hash-seed-note", "arc26-hash-seed", "arc26-run-logs-note", "arc26-run-logs")
    taken = sorted({cell.get("id") for cell in cells} & set(names))
    if with_ids and taken:
        raise BuildError(f"the notebook already has cells with the ids {taken}")
    ids = names if with_ids else (None,) * len(names)
    seed = seed_cell_source(hash_seed)
    dump = dump_cell_source(build)
    for label, source in (("seed", seed), ("dump", dump)):
        try:
            compile(source, f"<{label} cell>", "exec")
        except SyntaxError as err:
            raise BuildError(f"the generated {label} cell does not compile: {err}") from err
    first = [
        _new_cell("markdown", SEED_NOTE, as_lines, ids[0]),
        _new_cell("code", seed, as_lines, ids[1]),
    ]
    last = [
        _new_cell("markdown", DUMP_NOTE, as_lines, ids[2]),
        _new_cell("code", dump, as_lines, ids[3]),
    ]
    return first, last


def render_diff(header: list[str], changed: list[tuple[int, str, str]]) -> str:
    """Render ``patch.diff``: a commented summary, then a unified diff of every changed cell.

    Args:
        header: Summary lines, written as ``#`` comments before the diffs.
        changed: Per changed cell, its original index and its source before and after.

    Returns:
        The file's text. Cell indices name the pulled notebook and the built notebook.
    """
    lines = [f"# {line}" for line in header]
    for index, before, after in changed:
        lines.extend(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"pulled:cell[{index}]",
                tofile=f"{NOTEBOOK_FILE}:cell[{index + PREPENDED}]",
                lineterm="",
            )
        )
    return "\n".join(lines) + "\n"


def build(notebook: Path, metadata: Path, config: Path, owner: str, out: Path) -> None:
    """Build the reference capture folder: notebook, ``kernel-metadata.json``, and ``patch.diff``.

    Why: P0.1's fixture must come from the reference's own code, so the pushed notebook differs
    from the pulled one only in the changes this function makes and records (see the module
    docstring and ``docs/runbooks/build_reference_run.md``). Every check runs before the first
    write, so a refused build writes nothing.

    Args:
        notebook: The pulled reference ``.ipynb``.
        metadata: The ``kernel-metadata.json`` pulled with it.
        config: ``configs/reference_run.yaml``.
        owner: Kaggle username that owns the new notebook.
        out: Output folder, created if needed; the three files in it are overwritten.

    Raises:
        BuildError: If the config is refused (``parse_config``), including while it has TODO
            entries; the owner is not a Kaggle username; an input is also an output; an input
            cannot be read or parsed; the pulled metadata is refused (``kernel_metadata``); a
            replacement count does not match (``apply_replacements``); or an output cannot be
            written.
    """
    config_bytes = _read(config, "config")
    settings = parse_config(config_bytes, config)
    if not _safe_id(owner):
        raise BuildError(
            f"owner must be a Kaggle username (letters, digits, '.', '_', '-'); got {owner!r}"
        )
    targets = {out / NOTEBOOK_FILE, out / METADATA_FILE, out / DIFF_FILE}
    resolved = {target.resolve() for target in targets}
    for given in (notebook, metadata, config):
        if given.resolve() in resolved:
            raise BuildError(f"{given} is also an output of this build; choose another --out")
    notebook_bytes = _read(notebook, "notebook")
    pulled_bytes = _read(metadata, "pulled metadata")
    source = _json_object(notebook_bytes, notebook)
    pulled = _json_object(pulled_bytes, metadata)
    kernel_meta = kernel_metadata(
        pulled, owner, settings.competition, settings.kernel_source_versions
    )
    cells, changed = apply_replacements(_cells(source, notebook), settings.replacements)

    built = dict(source)
    settings_meta = source.get("metadata")
    removed = "top-level metadata.kaggle: not present"
    if isinstance(settings_meta, dict) and SETTINGS_KEY in settings_meta:
        saved = settings_meta[SETTINGS_KEY]
        keys = ", ".join(sorted(map(str, saved))) if isinstance(saved, dict) else ""
        removed = f"top-level metadata.kaggle ({f'keys: {keys}' if keys else repr(saved)})"
        built["metadata"] = {k: v for k, v in settings_meta.items() if k != SETTINGS_KEY}
    provenance = {
        "task": "P0.1-01",
        "builder": "scripts/build_reference_run.py",
        "pulled_notebook_sha256": _sha256(notebook_bytes),
        "pulled_metadata_sha256": _sha256(pulled_bytes),
        "config_sha256": _sha256(config_bytes),
        "hash_seed": settings.hash_seed,
        "replacements": [entry._asdict() for entry in settings.replacements],
    }
    first, last = _added_cells(source, cells, settings.hash_seed, provenance)
    built["cells"] = first + cells + last
    built_bytes = (json.dumps(built, indent=1, ensure_ascii=False) + "\n").encode("utf-8")

    after = len(cells) + PREPENDED
    header = [
        "P0.1-01 reference capture: what scripts/build_reference_run.py changed.",
        f"pulled notebook  {notebook.name}  sha256 {provenance['pulled_notebook_sha256']}",
        f"pulled metadata  {metadata.name}  sha256 {provenance['pulled_metadata_sha256']}",
        f"config           {config.name}  sha256 {provenance['config_sha256']}",
        f"built notebook   {NOTEBOOK_FILE}  sha256 {_sha256(built_bytes)}",
        f"removed          {removed}",
        f"prepended        cells 0 (markdown) and 1 (code): PYTHONHASHSEED={settings.hash_seed}",
        f"appended         cells {after} (markdown) and {after + 1} (code): run_logs dump",
        *(
            f"replacement      {entry.old!r} -> {entry.new!r}: {entry.count} matches"
            for entry in settings.replacements
        ),
        "changed cells    "
        + (
            ", ".join(f"{i} -> {i + PREPENDED}" for i, _, _ in changed)
            + " (pulled index -> built index); unified diffs follow"
            if changed
            else "none"
        ),
    ]
    files = {
        NOTEBOOK_FILE: built_bytes,
        METADATA_FILE: (json.dumps(kernel_meta, indent=2) + "\n").encode("utf-8"),
        DIFF_FILE: render_diff(header, changed).encode("utf-8"),
    }
    try:
        out.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (out / name).write_bytes(data)
    except OSError as err:
        raise BuildError(f"cannot write the build folder {out}: {err}") from err


# Synthetic inputs of the smoke path: a notebook shaped like the reference (Kaggle settings in its
# metadata, a %%writefile cell with two data path literals, a launch cell), and its metadata.
SMOKE_OWNER = "smoke-owner"
SMOKE_OLD = "/kaggle/input/arc-prize-2025/"
SMOKE_NEW = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/"
SMOKE_PATCHED_CELL = 2
SMOKE_CANDIDATES = {
    "0934a4d8_0": [
        {"solution": ((1, 2), (3, 4)), "beam_score": -0.25, "score_aug": [-0.5, -0.75]},
        {"solution": ((0,),), "beam_score": -1.5, "score_aug": [-2.0]},
    ],
    "aa4ec2a5_0": {"solution": [[5]], "beam_score": -0.1, "score_aug": (-0.2,)},
}


def _smoke_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    sources = [
        ("markdown", f"Reads the evaluation files from {SMOKE_OLD}."),
        ("code", "!pip uninstall -y tensorflow"),
        (
            "code",
            "%%writefile arc_loader.py\n"
            f"EVAL = '{SMOKE_OLD}arc-agi_evaluation_challenges.json'\n"
            f"TEST = '{SMOKE_OLD}arc-agi_test_challenges.json'\n",
        ),
        ("code", "!python starter.py --end-time 0"),
    ]
    cells = [
        _new_cell(kind, text, as_lines=True, cell_id=f"cell-{i}")
        for i, (kind, text) in enumerate(sources)
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            SETTINGS_KEY: {"accelerator": "nvidiaL4", "isInternetEnabled": False},
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    pulled = {
        "id": "someone/their-notebook",
        "id_no": 12345,
        "title": "Their notebook",
        "docker_image": "gcr.io/kaggle-gpu-images/python@sha256:0123",
        "competition_sources": ["arc-prize-2025"],
        "model_sources": ["someone/some-model/Transformers/bfloat16/1"],
        "dataset_sources": [],
        "kernel_sources": ["someone/install-notebook/3"],
    }
    entry = {"old": SMOKE_OLD, "new": SMOKE_NEW, "count": 2}
    settings = {"competition": "arc-prize-2026-arc-agi-2", "hash_seed": 0, "replacements": [entry]}
    todo = {**settings, "replacements": [{**entry, "old": "TODO(orchestrator)"}]}
    names = ("notebook.ipynb", "pulled.json", "config.yaml", "todo.yaml")
    texts = (
        json.dumps(notebook, indent=1),
        json.dumps(pulled, indent=2),
        yaml.safe_dump(settings, sort_keys=False),
        yaml.safe_dump(todo, sort_keys=False),
    )
    for name, text in zip(names, texts, strict=True):
        (root / name).write_text(text, encoding="utf-8")
    return root / names[0], root / names[1], root / names[2], root / names[3]


def _expect(condition: bool, what: str) -> None:
    if not condition:
        raise BuildError(f"smoke check failed: {what}")


def _text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def smoke() -> list[str]:
    """Run the build and both added cells on CPU with synthetic inputs (BRIEF A8 smoke path).

    Why: the reference's own code needs four GPUs and stays unchanged, so the smoke covers what
    this script adds: the build, and the appended cell, which on Kaggle runs only after half an
    hour of GPU time. Everything is written to a temporary folder, and the environment variable
    the seed cell sets is restored afterwards.

    Returns:
        One line per group of checks that passed.

    Raises:
        BuildError: If the build refuses the synthetic inputs or a check fails.
    """
    passed = []
    with tempfile.TemporaryDirectory(prefix="reference-run-smoke-") as tmp:
        root = Path(tmp)
        notebook, metadata, config, todo = _smoke_inputs(root)
        out = root / "out"
        try:
            build(notebook, metadata, todo, SMOKE_OWNER, out)
        except BuildError as err:
            _expect("TODO" in str(err), f"the TODO refusal says so: {err}")
        else:
            raise BuildError("smoke check failed: a config with a TODO entry was built")
        _expect(not out.exists(), "a refused build writes nothing")
        passed.append("config with a TODO entry refused, nothing written")

        build(notebook, metadata, config, SMOKE_OWNER, out)
        original = json.loads(notebook.read_text(encoding="utf-8"))
        built = json.loads((out / NOTEBOOK_FILE).read_text(encoding="utf-8"))
        cells = built["cells"]
        _expect(SETTINGS_KEY not in built["metadata"], "metadata.kaggle removed")
        _expect(built["metadata"]["kernelspec"] == original["metadata"]["kernelspec"], "kernelspec")
        _expect(len(cells) == len(original["cells"]) + 4, "two cells prepended, two appended")
        _expect(
            'os.environ["PYTHONHASHSEED"] = "0"' in _text(cells[1]), "seed cell sets the seed"
        )
        for index, cell in enumerate(original["cells"]):
            want = cell
            if index == SMOKE_PATCHED_CELL:
                lines = [line.replace(SMOKE_OLD, SMOKE_NEW) for line in cell["source"]]
                want = {**cell, "source": lines}
            _expect(cells[index + PREPENDED] == want, f"original cell {index} kept in place")
        _expect("def dump_run_logs(" in _text(cells[-1]), "dump cell last")
        passed.append("notebook: settings dropped, seed first, originals kept, dump last")

        meta = json.loads((out / METADATA_FILE).read_text(encoding="utf-8"))
        _expect(meta["id"] == f"{SMOKE_OWNER}/{SLUG}" and meta["title"] == TITLE, "identity")
        _expect("id_no" not in meta and meta["docker_image"].endswith(":0123"), "id_no, image")
        _expect(meta["competition_sources"] == ["arc-prize-2026-arc-agi-2"], "competition")
        _expect(meta["kernel_sources"] == ["someone/install-notebook/3"], "kernel source")
        diff = (out / DIFF_FILE).read_text(encoding="utf-8")
        _expect(f"+EVAL = '{SMOKE_NEW}" in diff and f"-EVAL = '{SMOKE_OLD}" in diff, "patch.diff")
        passed.append("kernel-metadata.json and patch.diff as specified")

        saved = os.environ.get("PYTHONHASHSEED")
        try:
            exec(compile(_text(cells[1]), "<seed cell>", "exec"), {})
            _expect(os.environ.get("PYTHONHASHSEED") == "0", "seed cell sets PYTHONHASHSEED")
        finally:
            if saved is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = saved
        passed.append("seed cell runs and sets PYTHONHASHSEED=0")

        inference = root / "inference_outputs"
        inference.mkdir()
        for name, value in SMOKE_CANDIDATES.items():
            data = pickle.dumps(value)
            (inference / name).write_bytes(bz2.compress(data) if name.startswith("0934") else data)
        logs = root / "working" / "run_logs"
        dump = _text(cells[-1])
        dump = dump.replace(repr(INFERENCE_DIR), repr(str(inference)))
        dump = dump.replace(repr(RUN_LOGS_DIR), repr(str(logs)))
        exec(compile(dump, "<dump cell>", "exec"), {"__name__": "dump_cell"})
        candidates = json.loads((logs / "candidates.json").read_text(encoding="utf-8"))
        _expect(candidates == json.loads(json.dumps(SMOKE_CANDIDATES)), "candidates.json")
        env = json.loads((logs / "env.json").read_text(encoding="utf-8"))
        _expect(env["build"]["hash_seed"] == 0 and {"python", "image"} <= set(env), "env.json")
        _expect((logs / "pip_freeze.txt").is_file(), "pip_freeze.txt written")
        passed.append("dump cell wrote candidates.json (2 subkeys), pip_freeze.txt, env.json")

        cap = 64
        report = dump_run_logs(str(inference), str(root / "capped"), cap, {})
        skipped = {item["file"] for item in report["skipped"]}
        large = {str(p) for p in inference.iterdir() if p.stat().st_size > cap}
        _expect(bool(large) and large <= skipped, "pickles over the cap skipped, not read")
        _expect("env.json" in skipped, "a log over the cap is skipped and recorded")
        _expect(not (root / "capped" / "env.json").exists(), "a skipped log is not written")
        passed.append("files over the size cap are skipped and recorded")
    return passed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--notebook", help="the pulled reference .ipynb")
    parser.add_argument("--metadata", help="the kernel-metadata.json pulled with it")
    parser.add_argument("--config", help="configs/reference_run.yaml")
    parser.add_argument("--owner", help="Kaggle username that owns the new notebook")
    parser.add_argument("--out", help="output folder, for example dist/P0.1-01")
    parser.add_argument(
        "--smoke", action="store_true", help="build synthetic inputs on CPU and check the result"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the reference capture folder, or run the smoke path.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 when the build or a smoke check is refused.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    names = ("notebook", "metadata", "config", "owner", "out")
    given = [name for name in names if getattr(args, name) is not None]
    if args.smoke and given:
        parser.error("--smoke takes no other arguments")
    if not args.smoke and len(given) < len(names):
        missing = [f"--{name}" for name in names if name not in given]
        parser.error(f"missing {', '.join(missing)}")
    if args.smoke:
        try:
            checks = smoke()
        except BuildError as err:
            print(f"refused: {err}", file=sys.stderr)
            return 2
        print("smoke passed (CPU, synthetic inputs):")
        for check in checks:
            print(f"- {check}")
        return 0
    out = Path(args.out)
    try:
        build(Path(args.notebook), Path(args.metadata), Path(args.config), args.owner, out)
    except BuildError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print(f"wrote {out / NOTEBOOK_FILE}, {out / METADATA_FILE}, {out / DIFF_FILE}")
    print("review patch.diff; after the brief is approved: kaggle_ops.py push with --run-id")
    return 0


if __name__ == "__main__":
    sys.exit(main())
