"""K0 environment probe: what a Kaggle session offers, measured on Kaggle (P0.5a, BRIEF A5).

Runs as a private CPU script kernel sent by ``kaggle_ops.py push`` (no GPU quota, so no pre-run
brief). It writes ``/kaggle/working/run_logs/env_probe.json`` and a readable ``env_probe.md``,
which ``kaggle_ops.py pull`` brings back into ``runs/K0/``. The fields answer the A5 questions and
feed P0.1-01, P0.2, P0.5, and P3.1: mount paths of the competition data, the base model, and
attached notebooks; image, Python, and package versions; the ``KAGGLE_*`` variables; CPU, RAM,
and disk; and per-task size statistics of the evaluation challenges.

Safety, by construction:

- A value is written only for the ``KAGGLE_*`` names in ``SHOWN_VALUES``, and never for a name
  that contains one of ``SECRET_MARKERS``; every other ``KAGGLE_*`` variable appears by name only.
  As a second layer, the value of every secret-named environment variable, ``KAGGLE_*`` or not,
  is scrubbed from the report, the Markdown, and the printed lines, since pip's error output and
  ``sys.path`` can carry any of them; ``pip list`` runs without those variables.
- The search of ``/kaggle/input`` keeps one set of visited real paths, so a symbolic link to
  ``/`` or a loop is recorded and skipped instead of turning the walk into a filesystem scan.
- Every file the probe reads goes through ``read_bytes``, which refuses a file whose name contains
  ``solution``, so the evaluation solutions are never opened. Sizes come from ``stat``.
- The statistics are counts and sizes; no grid content is written.

Standard library only. Packages are never imported: versions come from ``importlib.metadata`` and
``pip list``, since importing a GPU stack on a CPU session can fail or have side effects. Each
section runs on its own, so a failure is recorded under ``errors`` and costs no other section.

Usage::

    python env_probe.py                     # on Kaggle, as the pushed script kernel
    python env_probe.py --root /tmp/tree    # every absolute path is read or written under it
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath

SCHEMA = 1
INPUT_DIR = "/kaggle/input"
USR_LIB_DIR = "/kaggle/usr/lib"
WORKING_DIR = "/kaggle/working"
LOG_DIR = "/kaggle/working/run_logs"
TMP_DIR = "/tmp"
OS_RELEASE = "/etc/os-release"
# Build commit and date that the Kaggle image keeps (unverified until K0 runs); the P0.1-01
# decision rule compares them between runs to tell whether two runs used the same image.
IMAGE_FILES = ("/etc/git_commit", "/etc/build_date")
IMAGE_VALUE_CHARS = 200
MEMINFO = "/proc/meminfo"
JSON_NAME = "env_probe.json"
MD_NAME = "env_probe.md"
# A child of a listed folder has depth 1, so depth 4 reaches /kaggle/input/competitions/<slug>/
# <file> and /kaggle/input/models/<owner>/<model>/<framework>; deeper files, such as the model's
# config.json, come from the unbounded search in find_inputs.
LIST_DEPTH = 4
# Per listed folder, so an attached tree of installed packages cannot flood the report.
LIST_CAP = 2000
SHOWN_VALUES = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_IS_COMPETITION_RERUN",
    "KAGGLE_DOCKER_IMAGE",
    "KAGGLE_URL_BASE",
)
# Matched without case anywhere in the name; a match withholds the value even for a name in
# SHOWN_VALUES. Kaggle sessions carry, for example, KAGGLE_USER_SECRETS_TOKEN.
SECRET_MARKERS = ("TOKEN", "SECRET", "KEY", "PASS", "AUTH", "CREDENTIAL")
# Shorter values are not scrubbed: replacing a short string everywhere would corrupt the report.
MIN_SCRUB_CHARS = 8
REDACTED = "<redacted>"
# Both spellings of flash-attn, as the task brief lists them.
DISTRIBUTIONS = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "trl",
    "datasets",
    "unsloth",
    "unsloth_zoo",
    "xformers",
    "flash_attn",
    "flash-attn",
    "triton",
    "bitsandbytes",
    "vllm",
    "numpy",
)
PIP_TIMEOUT_S = 300
MEMINFO_KEYS = ("MemTotal", "MemAvailable", "SwapTotal")
COMPETITION = "arc-prize-2026-arc-agi-2"
MODEL_SLUG = "qwen3_4b_grids15_sft139"
ARC_FILES = "arc-agi_*.json"
EVAL_CHALLENGES = "arc-agi_evaluation_challenges.json"
# A file whose name contains this, in any case, is never opened.
FORBIDDEN = "solution"


class ProbeError(Exception):
    """A read the probe refuses to make, or an input it cannot use."""


def under(root: Path, path: str) -> Path:
    """Return where an absolute session path lives under ``root``.

    Args:
        root: The ``--root`` folder; ``/`` on Kaggle.
        path: An absolute path such as ``/kaggle/input``.

    Returns:
        ``root`` joined with ``path`` without its leading ``/``.
    """
    return root / path.lstrip("/")


def _shown(root: Path, path: Path) -> str:
    # The path as the session sees it, so reports read the same in tests and on Kaggle.
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
    return "/" if rel == "." else f"/{rel}"


def _describe(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def read_bytes(path: Path) -> bytes:
    """Read a file, refusing any file whose name contains ``solution``.

    Why: the evaluation solutions serve dev-mode scoring only (BRIEF A0) and this probe has no use
    for them; routing every read through here makes "never opened" a property of the code.

    Args:
        path: File to read.

    Returns:
        The file's bytes.

    Raises:
        ProbeError: If the name, or the name of the file a symbolic link resolves to, contains
            ``solution`` in any case. The file is not opened.
        OSError: If the file cannot be read.
    """
    for name in (path.name, Path(os.path.realpath(path)).name):
        if FORBIDDEN in name.lower():
            raise ProbeError(f"refused to open {name}: solutions files are never read")
    return path.read_bytes()


def python_info() -> dict[str, object]:
    """Describe the interpreter and platform the session runs.

    Returns:
        ``sys.version``, the version triple, implementation, executable, ``platform.platform()``,
        and machine type.
    """
    return {
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def os_release(root: Path) -> dict[str, str]:
    """Parse ``/etc/os-release``, which names the distribution under the Kaggle image.

    Args:
        root: The ``--root`` folder.

    Returns:
        The ``KEY=value`` pairs, quotes removed.

    Raises:
        OSError: If the file cannot be read.
    """
    info: dict[str, str] = {}
    text = read_bytes(under(root, OS_RELEASE)).decode("utf-8", errors="replace")
    for line in text.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            info[key] = _unquote(value)
    return info


def image_info(root: Path) -> dict[str, str | None]:
    """Read the build commit and date the Kaggle image keeps in ``/etc``.

    Why: the P0.1-01 brief matches two runs' images through these files; recording them here
    shows before any GPU run whether the image has them.

    Args:
        root: The ``--root`` folder.

    Returns:
        Each file's first line, stripped and capped at ``IMAGE_VALUE_CHARS``, keyed by its path
        as the reference run's ``env.json`` keys it; None when the file is absent.

    Raises:
        OSError: If a file exists but cannot be read.
    """
    info: dict[str, str | None] = {}
    for name in IMAGE_FILES:
        try:
            text = read_bytes(under(root, name)).decode("utf-8", errors="replace").strip()
        except FileNotFoundError:
            info[name] = None
            continue
        info[name] = text.splitlines()[0][:IMAGE_VALUE_CHARS] if text else ""
    return info


def meminfo(root: Path) -> dict[str, int]:
    """Read total and available RAM and swap from ``/proc/meminfo``.

    Args:
        root: The ``--root`` folder.

    Returns:
        The values of ``MEMINFO_KEYS`` that are present, in bytes.

    Raises:
        OSError: If the file cannot be read.
    """
    info: dict[str, int] = {}
    text = read_bytes(under(root, MEMINFO)).decode("ascii", errors="replace")
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if key in MEMINFO_KEYS and fields and fields[0].isdigit():
            scale = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
            info[key] = int(fields[0]) * scale
    return info


def cpu_info() -> dict[str, int | None]:
    """Count the CPUs of the machine and those this process may use.

    Returns:
        ``os.cpu_count()`` and the size of the scheduler affinity set (None where unsupported);
        the two differ when a container restricts the CPUs.
    """
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {"cpu_count": os.cpu_count(), "affinity": affinity}


def disk_usage(root: Path) -> dict[str, dict[str, object]]:
    """Measure the space of ``/kaggle/working`` (saved outputs) and ``/tmp`` (scratch).

    Args:
        root: The ``--root`` folder.

    Returns:
        Per path, ``total``, ``used``, and ``free`` in bytes, or ``error`` if it cannot be read.
    """
    usage: dict[str, dict[str, object]] = {}
    for path in (WORKING_DIR, TMP_DIR):
        try:
            total, used, free = shutil.disk_usage(under(root, path))
        except OSError as err:
            usage[path] = {"error": _describe(err)}
        else:
            usage[path] = {"total": total, "used": used, "free": free}
    return usage


def looks_secret(name: str) -> bool:
    """Return whether an environment variable name suggests a secret value.

    Args:
        name: Variable name.

    Returns:
        True if the name contains one of ``SECRET_MARKERS``, compared without case.
    """
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def _is_kaggle(name: str) -> bool:
    return name.upper().startswith("KAGGLE_")


def kaggle_env(environ: Mapping[str, str]) -> dict[str, object]:
    """List the ``KAGGLE_*`` variables by name, with values only for ``SHOWN_VALUES``.

    Why: the names show what a session provides, and the four shown values decide mode detection
    and identify the image. A value is never taken from a name that ``looks_secret``, even when
    ``SHOWN_VALUES`` lists it.

    Args:
        environ: The environment, ``os.environ`` on Kaggle.

    Returns:
        ``names``: every ``KAGGLE_*`` name, sorted; ``values``: each name in ``SHOWN_VALUES`` that
        does not look secret, mapped to its value or None when unset.
    """
    names = sorted(name for name in environ if _is_kaggle(name))
    values = {name: environ.get(name) for name in SHOWN_VALUES if not looks_secret(name)}
    return {"names": names, "values": values}


def secret_values(environ: Mapping[str, str]) -> list[str]:
    """Return the values of secret-named environment variables, to scrub from every output.

    Why: any of them, ``KAGGLE_*`` or not, can reach the report through pip's error output or
    ``sys.path``.

    Args:
        environ: The environment.

    Returns:
        Distinct values of at least ``MIN_SCRUB_CHARS`` characters of the variables whose names
        ``looks_secret``, longest first, so a secret that contains another is replaced whole.
    """
    found = {
        value
        for name, value in environ.items()
        if looks_secret(name) and len(value) >= MIN_SCRUB_CHARS
    }
    return sorted(found, key=len, reverse=True)


def scrub(value: object, secrets: Sequence[str]) -> object:
    """Replace every occurrence of a secret in the strings of a JSON-like value.

    Args:
        value: A report or any part of it: dicts, lists, strings, numbers, None.
        secrets: Strings to replace with ``REDACTED``.

    Returns:
        A copy of ``value`` with the secrets replaced in every string, keys included.
    """
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, dict):
        return {scrub(key, secrets): scrub(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(item, secrets) for item in value]
    return value


def _entry(root: Path, child: os.DirEntry[str]) -> dict[str, object]:
    path = _shown(root, Path(child.path))
    try:
        if child.is_symlink():
            return {"path": path, "type": "link", "target": os.readlink(child.path)}
        if child.is_dir(follow_symlinks=False):
            return {"path": path, "type": "dir"}
        return {"path": path, "type": "file", "size": child.stat(follow_symlinks=False).st_size}
    except OSError as err:
        return {"path": path, "type": "unknown", "error": _describe(err)}


def list_tree(
    root: Path, path: str, depth: int = LIST_DEPTH, cap: int = LIST_CAP
) -> dict[str, object]:
    """List a folder breadth first to a depth, keeping at most ``cap`` entries.

    Why: the shallow levels show the mount layout, which moved within 2026, so they must survive
    the cap; breadth-first order keeps them before anything deeper. Symbolic links are recorded
    with their targets and never descended, so the listing cannot leave the listed tree.

    Args:
        root: The ``--root`` folder.
        path: Absolute session path to list, such as ``/kaggle/input``.
        depth: Deepest level listed; a child of ``path`` has depth 1.
        cap: Most entries kept.

    Returns:
        ``path``, ``exists``, ``depth``, ``cap``, ``total`` (entries within the depth),
        ``truncated`` (whether the cap dropped any), ``entries`` sorted by path (each with
        ``path``, ``type``, and ``size`` for a file or ``target`` for a link), and ``errors``
        for folders that could not be read.
    """
    base = under(root, path)
    exists = base.is_dir()
    entries: list[dict[str, object]] = []
    errors: list[str] = []
    total = 0
    queue = deque([(base, 1)] if exists else [])
    while queue:
        folder, level = queue.popleft()
        try:
            with os.scandir(folder) as scan:
                children = sorted(scan, key=lambda item: item.name)
        except OSError as err:
            errors.append(_describe(err))
            continue
        for child in children:
            total += 1
            entry = _entry(root, child)
            if len(entries) < cap:
                entries.append(entry)
            if entry["type"] == "dir" and level < depth:
                queue.append((Path(child.path), level + 1))
    entries.sort(key=lambda item: str(item["path"]))
    return {
        "path": path,
        "exists": exists,
        "depth": depth,
        "cap": cap,
        "total": total,
        "truncated": total > len(entries),
        "entries": entries,
        "errors": errors,
    }


def _above(base: Path) -> set[str]:
    # Real paths of the folders above ``base``, by name and after resolving links. Entering one of
    # them (a link to /, say) would walk everything around the base.
    real = Path(os.path.realpath(base))
    return {os.path.realpath(parent) for parent in base.parents} | {
        str(parent) for parent in real.parents
    }


def _walk_files(base: Path, errors: list[str], skipped: list[tuple[str, str]]) -> Iterator[Path]:
    # A mount may be reachable only through a symbolic link, so links to folders are followed, but
    # only after the real tree, so a file is reported under its real path when it has one. One set
    # of visited real paths covers the whole walk: a link to a folder above the base, to a folder
    # already walked, or back into a loop is appended to ``skipped`` as (link, real path) and not
    # entered, so no folder is walked twice and no link reaches the rest of the filesystem.
    visited = _above(base)
    tops = deque([(str(base), os.path.realpath(base))])
    while tops:
        top, real = tops.popleft()
        if real in visited:
            skipped.append((top, real))
            continue
        visited.add(real)
        for folder, dirnames, filenames in os.walk(
            top, onerror=lambda err: errors.append(_describe(err))
        ):
            kept = []
            for name in sorted(dirnames):
                child = os.path.join(folder, name)
                child_real = os.path.realpath(child)
                if os.path.islink(child):
                    tops.append((child, child_real))
                elif child_real in visited:
                    skipped.append((child, child_real))
                else:
                    visited.add(child_real)
                    kept.append(name)
            dirnames[:] = kept
            for name in sorted(filenames):
                yield Path(folder, name)


def _sized(root: Path, path: Path) -> dict[str, object]:
    try:
        size: int | None = path.stat().st_size
    except OSError:
        size = None
    return {"path": _shown(root, path), "size": size}


def find_inputs(root: Path) -> dict[str, object]:
    """Resolve the competition files and the base model's config under ``/kaggle/input``.

    Why: nothing may hardcode ``/kaggle/input`` paths, since the 2026 layout differs from 2025
    and moved within 2026; the notebook and the P0.1-01 run take them from here. Files are
    located with ``stat`` only; none is opened.

    Args:
        root: The ``--root`` folder.

    Returns:
        ``arc_agi_files``: path and size of every ``arc-agi_*.json``; ``model_configs``: path and
        size of every ``config.json`` whose parent path contains ``MODEL_SLUG``;
        ``skipped_links``: path and real target of every folder not entered because its real path
        was already walked or lies above ``/kaggle/input``; ``errors``: folders that could not be
        read, including a missing ``/kaggle/input``.
    """
    base = under(root, INPUT_DIR)
    arc_files: list[dict[str, object]] = []
    model_configs: list[dict[str, object]] = []
    errors: list[str] = []
    skipped: list[tuple[str, str]] = []
    for path in _walk_files(base, errors, skipped):
        if fnmatch.fnmatchcase(path.name, ARC_FILES):
            arc_files.append(_sized(root, path))
        elif path.name == "config.json" and MODEL_SLUG in _shown(root, path.parent).lower():
            model_configs.append(_sized(root, path))
    real_root = Path(os.path.realpath(root))
    skipped_links = [
        {"path": _shown(root, Path(link)), "target": _shown(real_root, Path(real))}
        for link, real in skipped
    ]
    return {
        "arc_agi_files": arc_files,
        "model_configs": model_configs,
        "skipped_links": skipped_links,
        "errors": errors,
    }


def _side(grid: Sequence[Sequence[int]]) -> int:
    return max([len(grid), *(len(row) for row in grid)])


def _cells(grid: Sequence[Sequence[int]]) -> int:
    return sum(len(row) for row in grid)


def challenge_stats(tasks: Mapping[str, object]) -> dict[str, object]:
    """Summarize the size of each task: counts and sides, never grid contents.

    Args:
        tasks: A challenges file: task id to ``{"train": [{"input", "output"}], "test":
            [{"input"}]}``.

    Returns:
        ``per_task``: per task id, ``train_pairs``, ``test_inputs``, ``max_side`` (the largest
        height or width over train inputs, train outputs, and test inputs), and ``cells`` (their
        total cell count), or ``error`` with the exception type for a malformed task; ``summary``:
        the task count, malformed count, and totals over well-formed tasks. Test outputs, where a
        file has them, are not counted.
    """
    per_task: dict[str, dict[str, object]] = {}
    for task_id in sorted(tasks):
        task = tasks[task_id]
        try:
            train, test = task["train"], task["test"]
            grids = [pair["input"] for pair in train]
            grids += [pair["output"] for pair in train]
            grids += [pair["input"] for pair in test]
            per_task[task_id] = {
                "train_pairs": len(train),
                "test_inputs": len(test),
                "max_side": max((_side(grid) for grid in grids), default=0),
                "cells": sum(_cells(grid) for grid in grids),
            }
        except (KeyError, TypeError) as err:
            # Only the type: a message could quote part of the task.
            per_task[task_id] = {"error": type(err).__name__}
    good = [stats for stats in per_task.values() if "error" not in stats]
    summary = {
        "tasks": len(per_task),
        "malformed": len(per_task) - len(good),
        "train_pairs": sum(stats["train_pairs"] for stats in good),
        "test_inputs": sum(stats["test_inputs"] for stats in good),
        "max_side": max((stats["max_side"] for stats in good), default=0),
        "cells": sum(stats["cells"] for stats in good),
    }
    return {"summary": summary, "per_task": per_task}


def evaluation_stats(root: Path, arc_files: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compute ``challenge_stats`` of the evaluation challenges file that ``find_inputs`` found.

    Why: P0.2 selects DEV40 from these per-task sizes. With several copies, the file in the
    competition's folder wins, then the shortest path.

    Args:
        root: The ``--root`` folder.
        arc_files: ``find_inputs(root)["arc_agi_files"]``.

    Returns:
        ``source`` (session path), ``bytes``, ``sha256`` of the file, then the ``summary`` and
        ``per_task`` of ``challenge_stats``.

    Raises:
        ProbeError: If no ``arc-agi_evaluation_challenges.json`` was found, or the file does not
            hold a JSON object.
        OSError: If the file cannot be read.
        ValueError: If the file is not valid JSON.
    """
    candidates = [
        str(item["path"])
        for item in arc_files
        if PurePosixPath(str(item["path"])).name == EVAL_CHALLENGES
    ]
    if not candidates:
        raise ProbeError(f"no {EVAL_CHALLENGES} under {INPUT_DIR}")
    source = min(candidates, key=lambda path: (COMPETITION not in path, path.count("/"), path))
    data = read_bytes(under(root, source))
    tasks = json.loads(data)
    if not isinstance(tasks, dict):
        raise ProbeError(f"{source} does not hold a JSON object of tasks")
    digest = hashlib.sha256(data).hexdigest()
    return {"source": source, "bytes": len(data), "sha256": digest, **challenge_stats(tasks)}


def package_versions() -> dict[str, str | None]:
    """Read installed versions from package metadata, without importing any package.

    Returns:
        Each name in ``DISTRIBUTIONS`` mapped to its version, or None when not installed.
    """
    versions: dict[str, str | None] = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def pip_list(environ: Mapping[str, str]) -> list[dict[str, str]]:
    """Return every installed distribution, as ``pip list --format=json`` of this interpreter.

    Why: pip needs no credentials, and its error output lands in the report, so it runs without
    the secret-named variables.

    Args:
        environ: The environment; pip gets it without the names that ``looks_secret``.

    Returns:
        pip's JSON list of ``{"name", "version"}`` objects.

    Raises:
        ProbeError: If pip exits with an error.
        OSError: If the interpreter cannot be started.
        subprocess.TimeoutExpired: If pip runs longer than ``PIP_TIMEOUT_S``.
        ValueError: If pip's output is not JSON.
    """
    env = {name: value for name, value in environ.items() if not looks_secret(name)}
    # No version check: with internet off, pip would wait on PyPI before answering.
    command = [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"]
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=PIP_TIMEOUT_S, check=False, env=env
    )
    if done.returncode != 0:
        raise ProbeError(f"pip list exited {done.returncode}: {done.stderr.strip()[-2000:]}")
    return json.loads(done.stdout)


def _run_section(
    report: dict[str, object], errors: dict[str, str], key: str, section: Callable[[], object]
) -> None:
    try:
        report[key] = section()
    except Exception as err:  # one failing section must not cost the others
        report[key] = None
        errors[key] = _describe(err)


def probe(root: Path, environ: Mapping[str, str]) -> dict[str, object]:
    """Gather every section of the report, then scrub secret values from all of it.

    Args:
        root: The ``--root`` folder; ``/`` on Kaggle.
        environ: The environment, ``os.environ`` on Kaggle.

    Returns:
        The report: ``schema``, ``written_utc``, ``root``, one key per section (None when the
        section failed), and ``errors`` mapping each failed section to its error.
    """
    report: dict[str, object] = {
        "schema": SCHEMA,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
    }
    errors: dict[str, str] = {}
    sections: list[tuple[str, Callable[[], object]]] = [
        ("python", python_info),
        ("os_release", lambda: os_release(root)),
        ("image", lambda: image_info(root)),
        ("cpu", cpu_info),
        ("memory", lambda: meminfo(root)),
        ("disk", lambda: disk_usage(root)),
        ("kaggle_env", lambda: kaggle_env(environ)),
        ("sys_path", lambda: [str(entry) for entry in sys.path]),
        ("listings", lambda: {path: list_tree(root, path) for path in (INPUT_DIR, USR_LIB_DIR)}),
        ("versions", package_versions),
        ("pip_list", lambda: pip_list(environ)),
        ("resolved", lambda: find_inputs(root)),
    ]
    for key, section in sections:
        _run_section(report, errors, key, section)
    resolved = report["resolved"] if isinstance(report["resolved"], dict) else {}
    arc_files = resolved.get("arc_agi_files", [])
    _run_section(report, errors, "evaluation_stats", lambda: evaluation_stats(root, arc_files))
    report["errors"] = errors
    return scrub(report, secret_values(environ))


def _cell(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    if not rows:
        return ["None found.", ""]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return [*lines, ""]


def _gib(size: object) -> str:
    return f"{size / 1024**3:.1f} GiB" if isinstance(size, int) else str(size)


def _usage(usage: Mapping[str, object]) -> str:
    if "error" in usage:
        return str(usage["error"])
    return f"{_gib(usage['free'])} free of {_gib(usage['total'])}"


def _bullets(items: Sequence[str]) -> list[str]:
    return [*(f"- {item}" for item in items), ""] if items else []


def _unreadable(errors: Sequence[str]) -> list[str]:
    return _bullets([f"unreadable: {error}" for error in errors])


def _entry_line(entry: Mapping[str, object]) -> str:
    kind = entry["type"]
    if kind == "dir":
        return f"{entry['path']}/"
    if kind == "file":
        return f"{entry['path']}  {entry['size']}"
    if kind == "link":
        return f"{entry['path']} -> {entry['target']}"
    return f"{entry['path']}  ({entry.get('error')})"


def _listing_lines(listing: Mapping[str, object]) -> list[str]:
    lines = [f"## Listing {listing['path']}", ""]
    if not listing["exists"]:
        return [*lines, "Not present.", ""]
    shown = len(listing["entries"])
    note = f"{shown} of {listing['total']} entries within depth {listing['depth']}"
    if listing["truncated"]:
        note += f", truncated at the cap of {listing['cap']}."
    else:
        note += f" (cap {listing['cap']})."
    lines += [note, "", *_unreadable(listing["errors"])]
    lines += ["```", *(_entry_line(entry) for entry in listing["entries"]), "```", ""]
    return lines


def render_markdown(report: Mapping[str, object]) -> str:
    """Render the report for a human: decision-relevant facts first, full listings last.

    Args:
        report: The scrubbed report from ``probe``.

    Returns:
        Markdown text. It holds only what the JSON holds; ``pip list`` is summarized by count.
    """
    errors = report.get("errors") or {}
    python = report.get("python") or {}
    release = report.get("os_release") or {}
    cpu = report.get("cpu") or {}
    lines = [
        "# K0 environment probe",
        "",
        f"Written {report.get('written_utc')}, schema {report.get('schema')}. "
        f"Sections with errors: {', '.join(errors) or 'none'}.",
        "",
        "## Session",
        "",
    ]
    rows: list[tuple[object, object]] = [
        ("Python", python.get("version")),
        ("Platform", python.get("platform")),
        ("OS", release.get("PRETTY_NAME")),
        *[(f"Image {name}", value) for name, value in (report.get("image") or {}).items()],
        ("CPUs", f"{cpu.get('cpu_count')}, usable {cpu.get('affinity')}"),
    ]
    rows += [(key, _gib(size)) for key, size in (report.get("memory") or {}).items()]
    rows += [(f"Disk {path}", _usage(usage)) for path, usage in (report.get("disk") or {}).items()]
    lines += _table(("Field", "Value"), rows)

    env = report.get("kaggle_env") or {}
    values = env.get("values") or {}
    lines += [
        "## KAGGLE_* variables",
        "",
        f"Values are recorded only for {', '.join(SHOWN_VALUES)}; other names are listed alone.",
        "",
    ]
    rows = []
    for name in sorted(set(env.get("names") or []) | set(values)):
        if name not in values:
            rows.append((name, "(value not recorded)"))
        else:
            rows.append((name, "(unset)" if values[name] is None else f"`{values[name]}`"))
    lines += _table(("Variable", "Value"), rows)

    resolved = report.get("resolved") or {}
    lines += ["## Resolved paths", "", f"### {ARC_FILES} under {INPUT_DIR}", ""]
    files = resolved.get("arc_agi_files") or []
    lines += _table(("Path", "Bytes"), [(f"`{item['path']}`", item["size"]) for item in files])
    lines += [f"### config.json under a {MODEL_SLUG} folder", ""]
    files = resolved.get("model_configs") or []
    lines += _table(("Path", "Bytes"), [(f"`{item['path']}`", item["size"]) for item in files])
    lines += _bullets(
        [
            f"not entered: `{item['path']}` -> `{item['target']}` (already walked, or above "
            f"{INPUT_DIR})"
            for item in resolved.get("skipped_links") or []
        ]
    )
    lines += _unreadable(resolved.get("errors") or [])

    lines += ["## Evaluation challenges", ""]
    stats = report.get("evaluation_stats")
    if isinstance(stats, dict):
        summary = stats["summary"]
        lines += [
            f"Source `{stats['source']}`, {stats['bytes']} bytes, sha256 `{stats['sha256']}`.",
            f"{summary['tasks']} tasks ({summary['malformed']} malformed), "
            f"{summary['train_pairs']} train pairs, {summary['test_inputs']} test inputs, "
            f"largest side {summary['max_side']}, {summary['cells']} cells.",
            "",
        ]
        rows = []
        for task_id, task in stats["per_task"].items():
            if "error" in task:
                rows.append((task_id, f"malformed ({task['error']})", "", "", ""))
            else:
                cells = (task["train_pairs"], task["test_inputs"], task["max_side"], task["cells"])
                rows.append((task_id, *cells))
        header = ("Task", "Train pairs", "Test inputs", "Max side", "Cells")
        lines += _table(header, rows)
    else:
        lines += [f"Not available: {errors.get('evaluation_stats', 'not computed')}.", ""]

    lines += ["## Package versions", ""]
    versions = report.get("versions")
    if isinstance(versions, dict):
        rows = [(name, version or "not installed") for name, version in versions.items()]
        lines += _table(("Distribution", "Version"), rows)
    else:
        lines += [f"Not available: {errors.get('versions')}.", ""]
    packages = report.get("pip_list")
    if isinstance(packages, list):
        lines += [f"`pip list`: {len(packages)} distributions; the full list is in {JSON_NAME}."]
    else:
        lines += [f"`pip list` failed: {errors.get('pip_list')}."]
    lines += ["", "## sys.path", ""]
    lines += [f"- `{entry}`" for entry in report.get("sys_path") or []]
    lines += [""]

    for listing in (report.get("listings") or {}).values():
        lines += _listing_lines(listing)

    lines += ["## Errors", ""]
    lines += [f"- {key}: {message}" for key, message in errors.items()] or ["None."]
    return "\n".join(lines) + "\n"


def write_report(
    root: Path, report: Mapping[str, object], secrets: Sequence[str] = ()
) -> tuple[Path, Path]:
    """Write the report as JSON and Markdown to ``/kaggle/working/run_logs/``.

    Args:
        root: The ``--root`` folder.
        report: The scrubbed report from ``probe``.
        secrets: ``secret_values`` of the environment, scrubbed from the rendered Markdown too, so
            the guarantee does not rest on what the renderer adds.

    Returns:
        The paths of ``env_probe.json`` and ``env_probe.md``.

    Raises:
        OSError: If the folder or files cannot be written.
    """
    folder = under(root, LOG_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    json_path, md_path = folder / JSON_NAME, folder / MD_NAME
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = scrub(render_markdown(report), secrets)
    # The JSON is ASCII-escaped; a file name that is not valid UTF-8 must not fail the Markdown.
    md_path.write_text(markdown, encoding="utf-8", errors="replace")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser: only ``--root``, since Kaggle runs the script bare."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], allow_abbrev=False)
    parser.add_argument(
        "--root", default="/", help="folder that stands for / (tests use a synthetic tree)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Probe the session and write both reports.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``. Unknown arguments are ignored,
            because a script kernel may run inside an IPython kernel whose argv carries
            ``-f <connection file>``.

    Returns:
        0 once both files are written; section failures are reported inside them.
    """
    args, _ = build_parser().parse_known_args(argv)
    root = Path(os.path.abspath(args.root))
    secrets = secret_values(os.environ)
    report = probe(root, os.environ)
    json_path, md_path = write_report(root, report, secrets)
    lines = [f"wrote {_shown(root, json_path)} and {_shown(root, md_path)}"]
    lines += [f"section {key} failed: {message}" for key, message in report["errors"].items()]
    for line in lines:
        print(scrub(line, secrets), flush=True)
    return 0


if __name__ == "__main__":
    status = main()
    if status:  # only on failure: under an IPython kernel, SystemExit(0) fails the cell too
        sys.exit(status)
