r"""Build the two download notebooks of task P6.3: the INT4 weights and the engine wheelhouse.

Purpose: ADR-020's agent solver needs Qwen3.8-27B (``RedHatAI/Qwen3.8-27B-INT4``, 19.47 GB) and a
vLLM 0.31.0 stack for torch 2.13 and CUDA 13 (about 4 GB of wheels) inside Kaggle's offline GPU
sessions. Neither may pass through the human's machine (BRIEF A0), so two private CPU notebooks
with internet on fetch them on Kaggle; their outputs are attached to the GPU runs as notebook
inputs. CPU sessions spend no GPU quota and need no pre-run brief (BRIEF A9). Task P6.3, build
item 1 (``docs/tasks/P6.3.md``); runbook ``docs/runbooks/agent_spike.md``.

The existing notebook builder (``scripts/build_kaggle_notebook.py``) writes notebooks with
internet off around the arc26 wheel and the reference's model, so it cannot express these; this
module writes two script kernels instead, each a generated ``.py`` file with its settings baked
in, beside a ``kernel-metadata.json`` that passes the gateway's CPU-push checks.

- ``model`` (``alyahya737/arc26-dl-qwen38-int4``): ``huggingface_hub.snapshot_download`` of the
  pinned revision into ``/kaggle/working/<folder>/``; every file's size, sha256, and git blob sha1
  checked against the pinned list, which is also compared with the Hub API's answer at run time;
  ``MANIFEST.json``; on any mismatch the folder is deleted, so no unverified weights survive.
- ``wheels`` (``alyahya737/arc26-dl-agent-wheels``): ``pip download`` of the requirements under
  the pinned constraints (binary wheels only) into ``/kaggle/working/<folder>/``, the optional
  extras, ``install_wheelhouse.py`` (the offline installer, generated from this module), then the
  CPU install check: that installer run into a scratch venv exactly as the GPU run will run it,
  with import checks; ``MANIFEST.json`` with every file's name, version, size, and sha256.

Both kernels write ``run_logs/`` (``status.json``, ``MANIFEST.json``, ``summary.md``, logs),
which ``scripts/kaggle_ops.py pull`` brings back; a failure is recorded there, never raised, so
the logs survive (whether Kaggle keeps a failed session's outputs is unverified).

Runtime functions below run inside the kernels: the generated files hold their source verbatim
(``inspect.getsource``), so the tests call the same code with fakes for the network and pip.

Usage::

    python scripts/build_download_notebooks.py --config configs/P6.3-download.yaml \
        --out dist/P6.3-download
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import pprint
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent


def _load(name: str) -> ModuleType:
    # Sibling scripts are loaded by path, as the tests load them: scripts/ is not a package.
    spec = importlib.util.spec_from_file_location(f"_arc26_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The gateway's metadata checks and KAGGLE_OWNER: every folder written here must pass them.
gateway = _load("kaggle_ops")

KINDS = ("model", "wheels")
CODE_FILES = {"model": "dl_qwen38_int4.py", "wheels": "dl_agent_wheels.py"}
INSTALLER_FILE = "install_wheelhouse.py"
CONFIG_KEYS = ("owner", "reference_metadata", "model", "wheels")
MODEL_KEYS = (
    "kaggle_notebook", "repo_id", "revision", "folder", "files", "free_margin_bytes",
    "max_workers", "attempts", "timeout_s",
)  # fmt: skip
WHEELS_KEYS = (
    "kaggle_notebook", "python", "requirements", "extras", "index_url", "folder", "imports",
    "optional_imports", "expected_versions", "timeout_s", "constraints",
)  # fmt: skip
FILE_KEYS = ("name", "size", "sha256", "git_sha1")
EXTRA_KEYS = ("requirement", "find_links")
SLUG = re.compile(r"([A-Za-z0-9._-]+)/([a-z0-9]+(?:-[a-z0-9]+)*)")
SHA256 = re.compile(r"[0-9a-f]{64}")
SHA1 = re.compile(r"[0-9a-f]{40}")
PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+!_-]+")
MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Kaggle's session cap for CPU pushes, as the gateway enforces it.
MIN_TIMEOUT_S = gateway.MIN_TIMEOUT_S
MAX_TIMEOUT_S = gateway.MAX_TIMEOUT_S


class BuildError(Exception):
    """The config or the repository breaks a rule of this build; nothing was written."""


# --- Runtime functions: they run inside the generated kernels and the installer. -----------------


def write_json(path: str, value: object) -> None:
    """Write ``value`` as JSON through a temporary file and ``os.replace``.

    Why: a session killed mid-write leaves the old file or the new one, never half of one.

    Args:
        path: The file to write; its folder is created when missing.
        value: A JSON value; other objects are written as their ``str``.
    """
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    temporary = os.path.join(folder, f".{os.path.basename(path)}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, default=str) + "\n")
    os.replace(temporary, path)


def file_digests(path: str, block: int = 1 << 24) -> dict[str, Any]:
    r"""Return a file's size, sha256, and git blob sha1, reading it once.

    Why: the Hub names a large file by the sha256 of its content (the LFS object id) and a small
    one by its git blob sha1, so both are computed in one pass over the bytes.

    Args:
        path: The file.
        block: Bytes read per step.

    Returns:
        ``size``, ``sha256``, and ``git_sha1`` (``sha1(b"blob <size>\0" + content)``).
    """
    size = os.path.getsize(path)
    sha256 = hashlib.sha256()
    git = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            sha256.update(chunk)
            git.update(chunk)
    return {"size": size, "sha256": sha256.hexdigest(), "git_sha1": git.hexdigest()}


def check_files(folder: str, expected: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    """Check a folder against the expected files: names, sizes, hashes, and nothing else.

    Args:
        folder: The downloaded folder.
        expected: One mapping per file: ``name``, ``size``, ``sha256``, and ``git_sha1`` (None
            when the Hub gives none).

    Returns:
        Whether every check passed, and one row per file (expected and unexpected) with its
        measured size and hashes, seconds, and problems.
    """
    rows: list[dict[str, Any]] = []
    ok = True
    names = {item["name"] for item in expected}
    for item in expected:
        path = os.path.join(folder, item["name"])
        row: dict[str, Any] = {"name": item["name"], "problems": []}
        if os.path.islink(path) or not os.path.isfile(path):
            row["problems"].append("missing or not a regular file")
        else:
            start = time.time()
            row.update(file_digests(path))
            row["seconds"] = round(time.time() - start, 2)
            for key in ("size", "sha256", "git_sha1"):
                if item.get(key) is not None and row[key] != item[key]:
                    row["problems"].append(f"{key} {row[key]} is not the pinned {item[key]}")
        ok = ok and not row["problems"]
        rows.append(row)
    for directory, subdirectories, files in os.walk(folder):
        subdirectories.sort()
        for name in sorted(files):
            relative = os.path.relpath(os.path.join(directory, name), folder).replace(os.sep, "/")
            if relative not in names:
                rows.append({"name": relative, "problems": ["unexpected file"]})
                ok = False
    return ok, rows


def compare_listing(pinned: list[dict[str, Any]], hub: dict[str, Any], revision: str) -> list[str]:
    """Compare the pinned file list with the Hub API's listing of the revision.

    Why: the pinned list was read on 9 Oct; the run-time listing proves the revision still holds
    exactly those files before 19 GB are fetched.

    Args:
        pinned: The config's files.
        hub: ``hub_files``'s answer: ``sha`` and ``files``.
        revision: The pinned commit.

    Returns:
        The differences, empty when the listings agree.
    """
    problems = []
    if hub.get("sha") != revision:
        problems.append(f"the Hub resolved the revision to {hub.get('sha')}, not {revision}")
    theirs = {row["name"]: row for row in hub.get("files", [])}
    ours = {row["name"]: row for row in pinned}
    for name in sorted(set(ours) ^ set(theirs)):
        where = "pinned only" if name in ours else "on the Hub only"
        problems.append(f"{name}: {where}")
    for name in sorted(set(ours) & set(theirs)):
        for key in ("size", "sha256", "git_sha1"):
            if theirs[name].get(key) is not None and theirs[name][key] != ours[name].get(key):
                problems.append(f"{name}: Hub {key} {theirs[name][key]} differs from the pin")
    return problems


def hub_files(repo_id: str, revision: str) -> dict[str, Any]:
    """List a model revision's files with sizes and hashes through the Hub API.

    Args:
        repo_id: The model repository.
        revision: The commit.

    Returns:
        ``sha`` (the resolved commit) and ``files``: name, size, ``sha256`` for LFS files (else
        None), and ``git_sha1`` for the others (else None).
    """
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    rows = []
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        rows.append(
            {
                "name": sibling.rfilename,
                "size": sibling.size,
                "sha256": sha,
                "git_sha1": None if lfs else getattr(sibling, "blob_id", None),
            }
        )
    return {"sha": getattr(info, "sha", None), "files": rows}


def snapshot(repo_id: str, revision: str, local_dir: str, max_workers: int) -> None:
    """Download a model revision into a folder with ``huggingface_hub.snapshot_download``.

    Why: it resumes partly written files and uses the Xet or HTTP backend the image provides.
    Caches go to ``/tmp`` (scratch, never saved) and progress bars are off, so the session log
    stays readable.

    Args:
        repo_id: The model repository.
        revision: The commit.
        local_dir: The target folder (real files, no cache symlinks, since huggingface_hub 0.23).
        max_workers: Files fetched in parallel.
    """
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id, revision=revision, local_dir=local_dir, max_workers=max_workers
    )


def distribution_version(name: str) -> str | None:
    """Return an installed distribution's version, or None.

    Args:
        name: The distribution name.

    Returns:
        The version string, or None when it is not installed.
    """
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def download_model(
    settings: dict[str, Any],
    working: str = "/kaggle/working",
    snapshot_fn: Callable[[str, str, str, int], None] | None = None,
    hub_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Download, verify, and describe the model revision: the model notebook's work.

    Why: the GPU runs load these weights offline, so nothing unverified may be left behind: the
    free space is checked first, the pinned list is compared with the Hub's listing, every file
    is hashed after the download, and the folder is deleted on any failure.

    Args:
        settings: The config's ``model`` section.
        working: The output root, ``/kaggle/working`` on Kaggle.
        snapshot_fn: The downloader (``snapshot``); tests pass a fake.
        hub_fn: The Hub listing (``hub_files``); tests pass a fake.

    Returns:
        The status written to ``run_logs/status.json``, with ``ok``.
    """
    start = time.time()
    # Read by huggingface_hub at import: caches on the scratch disk, no progress bars, no
    # telemetry.
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    working = os.path.abspath(working)
    run_logs = os.path.join(working, "run_logs")
    os.makedirs(run_logs, exist_ok=True)
    target = os.path.join(working, settings["folder"])
    files = settings["files"]
    total = sum(item["size"] for item in files)
    status: dict[str, Any] = {
        "kind": "model",
        "repo_id": settings["repo_id"],
        "revision": settings["revision"],
        "folder": settings["folder"],
        "total_bytes": total,
        "python": sys.version.split()[0],
        "huggingface_hub": distribution_version("huggingface_hub"),
        "hf_xet": distribution_version("hf_xet") or distribution_version("hf-xet"),
        "stages": {},
        "problems": [],
        "ok": False,
    }
    try:
        free = shutil.disk_usage(working).free
        status["free_bytes_before"] = free
        if free < total + settings["free_margin_bytes"]:
            raise RuntimeError(
                f"{free} bytes free in {working}, under the {total} the files need plus the "
                f"{settings['free_margin_bytes']}-byte margin"
            )
        began = time.time()
        listing = None
        try:
            listing = (hub_fn or hub_files)(settings["repo_id"], settings["revision"])
        except Exception as err:  # the pinned hashes still check every file
            status["hub_listing"] = f"not read ({type(err).__name__}: {err}); pinned hashes used"
        status["stages"]["hub_listing_s"] = round(time.time() - began, 1)
        if listing is not None:
            differences = compare_listing(files, listing, settings["revision"])
            status["hub_listing"] = "agrees with the pinned files" if not differences else "differs"
            if differences:
                raise RuntimeError("Hub listing: " + "; ".join(differences))
        began = time.time()
        attempts = []
        for attempt in range(1, settings["attempts"] + 1):
            try:
                (snapshot_fn or snapshot)(
                    settings["repo_id"], settings["revision"], target, settings["max_workers"]
                )
                attempts.append({"attempt": attempt, "ok": True})
                break
            except Exception as err:
                attempts.append({"attempt": attempt, "ok": False, "error": f"{err}"[:500]})
                print(f"download attempt {attempt} failed: {err}", flush=True)
                if attempt == settings["attempts"]:
                    raise
        status["download_attempts"] = attempts
        seconds = time.time() - began
        status["stages"]["download_s"] = round(seconds, 1)
        status["download_mb_per_s"] = round(total / 1e6 / max(seconds, 1e-6), 1)
        # snapshot_download keeps its bookkeeping in <folder>/.cache/huggingface; only the
        # repository files belong in the output.
        shutil.rmtree(os.path.join(target, ".cache"), ignore_errors=True)
        began = time.time()
        ok, rows = check_files(target, files)
        status["files"] = rows
        status["stages"]["verify_s"] = round(time.time() - began, 1)
        if not ok:
            bad = [
                f"{row['name']}: {', '.join(row['problems'])}" for row in rows if row["problems"]
            ]
            raise RuntimeError("verification failed: " + "; ".join(bad))
        status["ok"] = True
    except Exception as err:
        status["problems"].append(f"{type(err).__name__}: {err}")
        status["traceback"] = traceback.format_exc()[-4000:]
    finally:
        if not status["ok"] and os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            status["deleted_folder"] = True
        status["free_bytes_after"] = shutil.disk_usage(working).free
        status["seconds"] = round(time.time() - start, 1)
        manifest = {
            key: status.get(key)
            for key in ("kind", "repo_id", "revision", "folder", "total_bytes", "ok")
        }
        manifest["files"] = [
            {key: row.get(key) for key in ("name", "size", "sha256", "git_sha1")}
            for row in status.get("files", [])
        ]
        write_json(os.path.join(working, "MANIFEST.json"), manifest)
        write_json(os.path.join(run_logs, "MANIFEST.json"), manifest)
        write_json(os.path.join(run_logs, "status.json"), status)
        write_summary(os.path.join(run_logs, "summary.md"), status)
    return status


def write_summary(path: str, status: dict[str, Any]) -> None:
    """Write a short human-readable summary of a download notebook's status.

    Args:
        path: The ``summary.md`` to write.
        status: The kernel's status mapping.
    """
    lines = [
        f"# Download notebook: {status.get('kind')}",
        "",
        f"- ok: {status.get('ok')}",
        f"- seconds: {status.get('seconds')}",
    ]
    for key in ("repo_id", "revision", "folder", "total_bytes", "download_mb_per_s", "hub_listing"):
        if key in status:
            lines.append(f"- {key}: {status[key]}")
    for key in ("wheel_count", "wheel_bytes", "python", "pip"):
        if key in status:
            lines.append(f"- {key}: {status[key]}")
    if status.get("install_check"):
        lines.append(f"- install check: {json.dumps(status['install_check'], default=str)}")
    for problem in status.get("problems", []):
        lines.append(f"- PROBLEM: {problem}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def clean_env(environ: Mapping[str, str], venv: str | None = None) -> dict[str, str]:
    """Return an environment for the engine's venv, cut off from the image's packages.

    Why: Kaggle puts the attached install notebook (torch 2.8 and its stack) and its own folders
    on ``PYTHONPATH``, which would shadow the venv's torch 2.13; pip settings from the image would
    change what pip resolves.

    Args:
        environ: The current environment.
        venv: The venv folder; its ``bin`` goes first on ``PATH`` when given.

    Returns:
        A copy without ``PYTHONPATH``, ``PYTHONHOME``, ``PYTHONSTARTUP``, ``PYTHONUSERBASE``,
        ``VIRTUAL_ENV``, and any ``PIP_*`` variable, with ``PYTHONNOUSERSITE=1``.
    """
    dropped = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"}
    env = {k: v for k, v in environ.items() if k not in dropped and not k.startswith("PIP_")}
    env["PYTHONNOUSERSITE"] = "1"
    if venv:
        env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = venv
    return env


def run_logged(
    command: list[str], log_path: str, timeout: float, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Run a command, copy its output to a log file and to stdout, kill it at a timeout.

    Args:
        command: Program and arguments; no shell.
        log_path: The log file, appended to.
        timeout: Seconds before the command is killed.
        env: The command's environment; None inherits.

    Returns:
        ``command``, ``exit`` (127 when it could not start, negative for a signal),
        ``seconds``, ``timed_out``, and ``tail``, its last 40 output lines.
    """
    start = time.time()
    shown = "$ " + shlex.join(command)
    print(shown, flush=True)
    tail: list[str] = []
    fired = threading.Event()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(shown + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=None if env is None else dict(env),
                text=True,
                errors="replace",
            )
        except OSError as err:
            log.write(f"could not run: {err}\n")
            return {
                "command": command, "exit": 127, "seconds": 0.0, "timed_out": False,
                "tail": [str(err)],
            }  # fmt: skip

        def kill() -> None:
            fired.set()
            process.kill()

        timer = threading.Timer(timeout, kill)
        timer.start()
        try:
            for line in process.stdout:
                log.write(line)
                tail.append(line.rstrip("\n"))
                del tail[:-40]
                print(line, end="", flush=True)
            code = process.wait()
        finally:
            timer.cancel()
        log.write(f"exit {code} after {time.time() - start:.1f} s\n")
    return {
        "command": command,
        "exit": code,
        "seconds": round(time.time() - start, 1),
        "timed_out": fired.is_set(),
        "tail": tail,
    }


def wheel_inventory(folder: str) -> list[dict[str, Any]]:
    """List every file of a wheelhouse with its distribution name, version, size, and sha256.

    Args:
        folder: The wheelhouse.

    Returns:
        One row per file, sorted by name; ``name`` and ``version`` only for ``.whl`` files, the
        name normalized as pip compares names (lowercase, runs of ``-_.`` as ``-``).
    """
    rows = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if not os.path.isfile(path):
            continue
        row: dict[str, Any] = {"file": entry}
        if entry.endswith(".whl"):
            parts = entry[: -len(".whl")].split("-")
            row["name"] = re.sub(r"[-_.]+", "-", parts[0]).lower()
            row["version"] = parts[1] if len(parts) > 1 else None
        digests = file_digests(path)
        row["size"] = digests["size"]
        row["sha256"] = digests["sha256"]
        rows.append(row)
    return rows


def install_wheelhouse(
    wheelhouse: str,
    venv: str,
    imports: list[str],
    optional_imports: list[str],
    log_path: str,
    run: Callable[..., dict[str, Any]] | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Install a wheelhouse offline into a fresh venv and check the imports: the installer.

    Why: the GPU run must get the engine with no index and no image package in the way, and the
    CPU check must prove the same procedure works, so both run this one function (as
    ``install_wheelhouse.py``). Steps: ``python -m venv --without-pip`` (no ensurepip needed);
    pip bootstrapped from the wheelhouse's own pip wheel (``python pip-*.whl/pip install``, a
    form pip supports); ``pip install --no-index --find-links`` of ``requirements.txt`` under
    ``constraints.txt`` without byte-compiling; each line of ``extras.txt`` (optional);
    ``pip check``; one import per module in a subprocess; ``pip freeze``. Every command runs in
    ``clean_env``.

    Args:
        wheelhouse: The folder with the wheels and the requirement files.
        venv: The venv to create; an existing one is replaced.
        imports: Modules that must import.
        optional_imports: Modules whose import is only recorded.
        log_path: The log file every command appends to.
        run: The command runner (``run_logged``); tests pass a fake.
        python: The interpreter that creates the venv; defaults to ``sys.executable``.

    Returns:
        The report: ``ok``, ``problems``, every step, the imports, the freeze, and ``seconds``.
    """
    run = run or run_logged
    python = python or sys.executable
    start = time.time()
    wheelhouse = os.path.abspath(wheelhouse)
    venv = os.path.abspath(venv)
    report: dict[str, Any] = {
        "wheelhouse": wheelhouse,
        "venv": venv,
        "python": python,
        "steps": [],
        "imports": {},
        "problems": [],
        "ok": False,
    }
    env = clean_env(os.environ, venv)
    probe = (
        "import importlib, json, sys\n"
        "name = sys.argv[1]\n"
        "module = importlib.import_module(name)\n"
        "info = {'version': getattr(module, '__version__', None),"
        " 'file': getattr(module, '__file__', None)}\n"
        "if name == 'torch':\n"
        "    info['cuda'] = module.version.cuda\n"
        "    info['cuda_available'] = module.cuda.is_available()\n"
        "print('IMPORT_PROBE ' + json.dumps(info, default=str))\n"
    )

    def step(name: str, command: list[str], timeout: float, fatal: bool = True) -> dict[str, Any]:
        result = run(command, log_path, timeout, env)
        result = {"step": name, **result, "fatal": fatal}
        report["steps"].append(result)
        if result["exit"] != 0 and fatal:
            report["problems"].append(f"{name} exited {result['exit']}")
        return result

    try:
        requirements = os.path.join(wheelhouse, "requirements.txt")
        constraints = os.path.join(wheelhouse, "constraints.txt")
        extras = os.path.join(wheelhouse, "extras.txt")
        with open(requirements, encoding="utf-8") as handle:
            pins = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
        pip_pin = next((pin for pin in pins if pin.startswith("pip==")), None)
        pip_wheels = sorted(
            entry for entry in os.listdir(wheelhouse)
            if entry.startswith("pip-") and entry.endswith(".whl")
        )  # fmt: skip
        if pip_pin is None or len(pip_wheels) != 1:
            raise RuntimeError(
                f"the wheelhouse needs pip==<version> in requirements.txt and one pip wheel; "
                f"found {pip_pin!r} and {pip_wheels}"
            )
        if os.path.exists(venv):
            shutil.rmtree(venv)
        venv_python = os.path.join(venv, "bin", "python")
        offline = ["--no-index", "--find-links", wheelhouse, "--disable-pip-version-check"]
        quiet = ["--no-compile", "--progress-bar", "off", "--no-warn-script-location"]
        if step("venv", [python, "-m", "venv", "--without-pip", venv], 600)["exit"]:
            raise RuntimeError("venv creation failed")
        bootstrap = [venv_python, os.path.join(wheelhouse, pip_wheels[0], "pip"), "install"]
        if step("pip", bootstrap + offline + quiet + [pip_pin], 600)["exit"]:
            raise RuntimeError("pip bootstrap failed")
        main = [venv_python, "-m", "pip", "install", *offline, *quiet, "-r", requirements]
        if os.path.isfile(constraints):
            main += ["-c", constraints]
        if step("install", main, 3600)["exit"]:
            raise RuntimeError("install of requirements.txt failed")
        if os.path.isfile(extras):
            with open(extras, encoding="utf-8") as handle:
                for extra in [line.strip() for line in handle if line.strip()]:
                    command = [venv_python, "-m", "pip", "install", *offline, *quiet, extra]
                    step(f"extra {extra}", command, 1200, fatal=False)
        step("pip check", [venv_python, "-m", "pip", "check"], 600, fatal=False)
        for name in [*imports, *optional_imports]:
            required = name in imports
            result = step(f"import {name}", [venv_python, "-c", probe, name], 900, required)
            info = None
            for line in result.get("tail", []):
                if line.startswith("IMPORT_PROBE "):
                    info = json.loads(line[len("IMPORT_PROBE ") :])
            report["imports"][name] = {
                "ok": result["exit"] == 0,
                "required": required,
                **(info or {}),
            }
        freeze = step("pip freeze", [venv_python, "-m", "pip", "freeze"], 600, fatal=False)
        report["freeze_tail"] = freeze.get("tail", [])
        report["ok"] = not report["problems"]
    except Exception as err:
        report["problems"].append(f"{type(err).__name__}: {err}")
    report["seconds"] = round(time.time() - start, 1)
    return report


def installer_main(argv: list[str]) -> int:
    """Command line of ``install_wheelhouse.py``.

    Args:
        argv: ``--wheelhouse``, ``--venv``, ``--report``, ``--log``, ``--imports``, and
            ``--optional-imports`` (comma-separated module names); unknown arguments are ignored.

    Returns:
        0 when the install and the required imports succeeded, else 1.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Install a wheelhouse offline into a venv.")
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--imports", default="")
    parser.add_argument("--optional-imports", default="")
    args, _ = parser.parse_known_args(argv)
    log = args.log or os.path.splitext(args.report)[0] + ".log"
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    report = install_wheelhouse(
        args.wheelhouse,
        args.venv,
        [name for name in args.imports.split(",") if name],
        [name for name in args.optional_imports.split(",") if name],
        log,
    )
    write_json(args.report, report)
    print(f"install {'ok' if report['ok'] else 'FAILED'} in {report['seconds']} s", flush=True)
    for problem in report["problems"]:
        print(f"PROBLEM: {problem}", flush=True)
    return 0 if report["ok"] else 1


def build_wheelhouse(
    settings: dict[str, Any],
    installer_source: str,
    working: str = "/kaggle/working",
    run: Callable[..., dict[str, Any]] | None = None,
    scratch: str = "/tmp",
) -> dict[str, Any]:
    """Download the wheelhouse, write its installer, and check the install: the wheels notebook.

    Why: the GPU run installs offline from exactly these files, so the notebook downloads binary
    wheels for its own interpreter (the GPU run's), under the pinned constraints, and then runs
    the installer it ships into a scratch venv, as the GPU run will.

    Args:
        settings: The config's ``wheels`` section.
        installer_source: The text of ``install_wheelhouse.py``.
        working: The output root, ``/kaggle/working`` on Kaggle.
        run: The command runner (``run_logged``); tests pass a fake.
        scratch: Folder for the pip bootstrap, pip's cache, and the check venv, never saved:
            ``/tmp`` on Kaggle (about 1.1 TB free, K0).

    Returns:
        The status written to ``run_logs/status.json``, with ``ok``.
    """
    run = run or run_logged
    start = time.time()
    working = os.path.abspath(working)
    run_logs = os.path.join(working, "run_logs")
    os.makedirs(run_logs, exist_ok=True)
    wheelhouse = os.path.join(working, settings["folder"])
    log = os.path.join(run_logs, "pip.log")
    status: dict[str, Any] = {
        "kind": "wheels",
        "folder": settings["folder"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libc": "-".join(platform.libc_ver()),
        "requirements": settings["requirements"],
        "constraints": len(settings["constraints"]),
        "extras": [],
        "stages": {},
        "problems": [],
        "ok": False,
    }
    env = clean_env(os.environ)
    inventory: list[dict[str, Any]] | None = None
    try:
        wanted = settings["python"]
        have = "{}.{}".format(*sys.version_info[:2])
        if have != wanted or platform.machine() != "x86_64":
            raise RuntimeError(
                f"this interpreter is Python {have} on {platform.machine()}; the wheelhouse is "
                f"for Python {wanted} on x86_64, the GPU run's interpreter"
            )
        os.makedirs(wheelhouse, exist_ok=True)
        index = ["--isolated", "--disable-pip-version-check", "--progress-bar", "off"]
        cache = ["--cache-dir", os.path.join(scratch, "pip-cache")]
        pip_pin = next(pin for pin in settings["requirements"] if pin.startswith("pip=="))
        boot = os.path.join(scratch, "pip-boot")
        began = time.time()
        fetch = [sys.executable, "-m", "pip", "download", *index, *cache, "--no-deps"]
        fetch += ["--only-binary", ":all:", "--index-url", settings["index_url"], "--dest", boot]
        if run(fetch + [pip_pin], log, 900, env)["exit"]:
            raise RuntimeError(f"could not fetch {pip_pin} to run it")
        boot_wheels = sorted(e for e in os.listdir(boot) if e.startswith("pip-"))
        if len(boot_wheels) != 1:
            raise RuntimeError(f"expected one pip wheel in {boot}, found {boot_wheels}")
        pip = [sys.executable, os.path.join(boot, boot_wheels[0], "pip")]
        status["pip"] = boot_wheels[0]
        requirements = os.path.join(wheelhouse, "requirements.txt")
        constraints = os.path.join(wheelhouse, "constraints.txt")
        for path, lines in ((requirements, settings["requirements"]),
                            (constraints, settings["constraints"])):  # fmt: skip
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        download = [*pip, "download", *index, *cache, "--only-binary", ":all:"]
        download += ["--index-url", settings["index_url"], "--dest", wheelhouse]
        result = run(download + ["-r", requirements, "-c", constraints], log, 2400, env)
        status["stages"]["download_s"] = round(time.time() - began, 1)
        if result["exit"]:
            raise RuntimeError(f"pip download exited {result['exit']}: {result['tail'][-5:]}")
        installed_extras = []
        for extra in settings["extras"]:
            command = [*pip, "download", *index, *cache, "--no-deps", "--only-binary", ":all:"]
            command += ["--no-index", "--find-links", extra["find_links"], "--dest", wheelhouse]
            result = run(command + [extra["requirement"]], log, 900, env)
            row = {"requirement": extra["requirement"], "ok": result["exit"] == 0}
            status["extras"].append(row)
            if row["ok"]:
                installed_extras.append(extra["requirement"])
        with open(os.path.join(wheelhouse, "extras.txt"), "w", encoding="utf-8") as handle:
            handle.write("".join(f"{line}\n" for line in installed_extras))
        installer = os.path.join(wheelhouse, "install_wheelhouse.py")
        with open(installer, "w", encoding="utf-8") as handle:
            handle.write(installer_source)
        inventory = wheel_inventory(wheelhouse)
        wheels = [row for row in inventory if "name" in row]
        status["wheel_count"] = len(wheels)
        status["wheel_bytes"] = sum(row["size"] for row in wheels)
        versions = {row["name"]: row["version"] for row in wheels}
        for name, version in settings["expected_versions"].items():
            if versions.get(name) != version:
                raise RuntimeError(
                    f"the wheelhouse holds {name} {versions.get(name)}, not {version}"
                )
        began = time.time()
        report_path = os.path.join(run_logs, "install_check.json")
        venv = os.path.join(scratch, "check-venv")
        check = [sys.executable, installer, "--wheelhouse", wheelhouse, "--venv", venv]
        check += ["--report", report_path, "--log", os.path.join(run_logs, "install_check.log")]
        check += ["--imports", ",".join(settings["imports"])]
        check += ["--optional-imports", ",".join(settings["optional_imports"])]
        result = run(check, log, 3000, env)
        status["stages"]["install_check_s"] = round(time.time() - began, 1)
        report: dict[str, Any] = {}
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        status["install_check"] = {
            "ok": bool(report.get("ok")) and result["exit"] == 0,
            "seconds": report.get("seconds"),
            "imports": report.get("imports"),
            "steps": {s["step"]: [s["exit"], s["seconds"]] for s in report.get("steps", [])},
            "problems": report.get("problems"),
        }
        if not status["install_check"]["ok"]:
            raise RuntimeError(
                f"install check failed: {report.get('problems') or result['tail'][-5:]}"
            )
        status["ok"] = True
    except Exception as err:
        status["problems"].append(f"{type(err).__name__}: {err}")
        status["traceback"] = traceback.format_exc()[-4000:]
    finally:
        status["seconds"] = round(time.time() - start, 1)
        manifest = {
            key: status.get(key)
            for key in ("kind", "folder", "python", "machine", "libc", "requirements", "pip", "ok")
        }
        manifest["extras"] = status["extras"]
        manifest["install_check"] = status.get("install_check")
        if inventory is None and os.path.isdir(wheelhouse):
            inventory = wheel_inventory(wheelhouse)
        # The installer and the requirement files were hashed with the wheels; MANIFEST.json
        # itself is written last and so is not listed.
        manifest["files"] = inventory or []
        if os.path.isdir(wheelhouse):
            write_json(os.path.join(wheelhouse, "MANIFEST.json"), manifest)
        write_json(os.path.join(run_logs, "MANIFEST.json"), manifest)
        write_json(os.path.join(run_logs, "status.json"), status)
        write_summary(os.path.join(run_logs, "summary.md"), status)
    return status


def kernel_main(kind: str, settings: dict[str, Any], build: dict[str, Any]) -> dict[str, Any]:
    """Run one download notebook and print its verdict; never raises.

    Args:
        kind: ``model`` or ``wheels``.
        settings: The kind's config section.
        build: Provenance: the commit and config the kernel was built from.

    Returns:
        The status mapping.
    """
    print(f"build: {json.dumps(build)}", flush=True)
    try:
        if kind == "model":
            status = download_model(settings)
        else:
            status = build_wheelhouse(settings, INSTALLER_SOURCE)  # noqa: F821, in the kernel
    except Exception as err:  # the statuses catch their own errors; this is the last guard
        status = {"kind": kind, "ok": False, "problems": [f"{type(err).__name__}: {err}"]}
        traceback.print_exc()
    verdict = "OK" if status.get("ok") else "FAILED: " + "; ".join(status.get("problems", []))
    print(f"{kind} notebook {verdict} after {status.get('seconds')} s", flush=True)
    return status


# Functions copied into each generated file, in this order.
COMMON_FUNCTIONS = (write_json, file_digests, write_summary)
MODEL_FUNCTIONS = (
    check_files, compare_listing, hub_files, snapshot, distribution_version, download_model,
)  # fmt: skip
INSTALLER_FUNCTIONS = (write_json, clean_env, run_logged, install_wheelhouse)
WHEELS_FUNCTIONS = (clean_env, run_logged, wheel_inventory, build_wheelhouse)
IMPORTS = """\
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from typing import Any
"""


# --- Build side: config, generated files, push folders. ------------------------------------------


def _fail(path: Path, message: str) -> BuildError:
    return BuildError(f"{path}: {message}")


def _keys(section: object, keys: tuple[str, ...], where: str, path: Path) -> dict[str, Any]:
    if not isinstance(section, dict) or set(map(str, section)) != set(keys):
        found = sorted(map(str, section)) if isinstance(section, dict) else section
        raise _fail(path, f"{where} must have exactly the keys {', '.join(keys)}; got {found}")
    return section


def _notebook(value: object, owner: str, where: str, path: Path) -> str:
    match = SLUG.fullmatch(value) if isinstance(value, str) else None
    if match is None or match.group(1) != owner or not 5 <= len(match.group(2)) <= 50:
        raise _fail(path, f"{where} must be {owner}/<slug of 5 to 50 characters>; got {value!r}")
    return value


def _timeout(value: object, where: str, path: Path) -> int:
    if type(value) is not int or not MIN_TIMEOUT_S <= value <= MAX_TIMEOUT_S:
        raise _fail(path, f"{where} must be an integer from {MIN_TIMEOUT_S} to {MAX_TIMEOUT_S}")
    return value


def _check_model(model: dict[str, Any], owner: str, path: Path) -> None:
    _keys(model, MODEL_KEYS, "model", path)
    _notebook(model["kaggle_notebook"], owner, "model.kaggle_notebook", path)
    if not isinstance(model["repo_id"], str) or model["repo_id"].count("/") != 1:
        raise _fail(path, "model.repo_id must be <org>/<name>")
    if not isinstance(model["revision"], str) or not SHA1.fullmatch(model["revision"]):
        raise _fail(path, "model.revision must be a full 40-character commit")
    if not isinstance(model["folder"], str) or not FOLDER.fullmatch(model["folder"]):
        raise _fail(path, "model.folder must be one folder name")
    files = model["files"]
    if not isinstance(files, list) or not files:
        raise _fail(path, "model.files must be a non-empty list")
    names = []
    for item in files:
        _keys(item, FILE_KEYS, "every model file", path)
        name, size, sha256, git_sha1 = (item[key] for key in FILE_KEYS)
        if not isinstance(name, str) or not name or name.startswith("/") or ".." in name.split("/"):
            raise _fail(path, f"model file name {name!r} must be a relative path")
        if type(size) is not int or size < 0:
            raise _fail(path, f"model file {name}: size must be a non-negative integer")
        if not isinstance(sha256, str) or not SHA256.fullmatch(sha256):
            raise _fail(path, f"model file {name}: sha256 must be 64 hex characters")
        if git_sha1 is not None and not (isinstance(git_sha1, str) and SHA1.fullmatch(git_sha1)):
            raise _fail(path, f"model file {name}: git_sha1 must be null or 40 hex characters")
        names.append(name)
    if len(set(names)) != len(names):
        raise _fail(path, "model.files repeats a name")
    for key in ("free_margin_bytes", "max_workers", "attempts"):
        if type(model[key]) is not int or model[key] < (0 if key == "free_margin_bytes" else 1):
            raise _fail(path, f"model.{key} must be a positive integer")
    _timeout(model["timeout_s"], "model.timeout_s", path)


def _check_wheels(wheels: dict[str, Any], owner: str, path: Path) -> None:
    _keys(wheels, WHEELS_KEYS, "wheels", path)
    _notebook(wheels["kaggle_notebook"], owner, "wheels.kaggle_notebook", path)
    if not isinstance(wheels["python"], str) or not re.fullmatch(r"3\.[0-9]+", wheels["python"]):
        raise _fail(path, 'wheels.python must be a quoted major.minor such as "3.11"')
    for key in ("requirements", "constraints"):
        pins = wheels[key]
        if (
            not isinstance(pins, list)
            or not pins
            or not all(isinstance(pin, str) and PIN.fullmatch(pin) for pin in pins)
        ):
            raise _fail(path, f"wheels.{key} must be a non-empty list of name==version pins")
    if not any(pin.startswith("pip==") for pin in wheels["requirements"]):
        raise _fail(path, "wheels.requirements must pin pip, which bootstraps the offline venv")
    pinned = {pin.split("==")[0].lower(): pin for pin in wheels["constraints"]}
    for pin in wheels["requirements"]:
        if pinned.get(pin.split("==")[0].lower()) != pin:
            raise _fail(path, f"wheels.requirements {pin} is not the same pin in constraints")
    if not isinstance(wheels["extras"], list):
        raise _fail(path, "wheels.extras must be a list")
    for extra in wheels["extras"]:
        _keys(extra, EXTRA_KEYS, "every wheels extra", path)
        if not PIN.fullmatch(str(extra["requirement"])) or not str(extra["find_links"]).startswith(
            "https://"
        ):
            raise _fail(path, f"wheels extra {extra} needs a name==version pin and an https URL")
    if not str(wheels["index_url"]).startswith("https://"):
        raise _fail(path, "wheels.index_url must be an https URL")
    if not isinstance(wheels["folder"], str) or not FOLDER.fullmatch(wheels["folder"]):
        raise _fail(path, "wheels.folder must be one folder name")
    for key in ("imports", "optional_imports"):
        names = wheels[key]
        if not isinstance(names, list) or not all(
            isinstance(n, str) and MODULE.fullmatch(n) for n in names
        ):
            raise _fail(path, f"wheels.{key} must be a list of module names")
    if not wheels["imports"]:
        raise _fail(path, "wheels.imports must name at least one module")
    expected = wheels["expected_versions"]
    if (
        not isinstance(expected, dict)
        or not expected
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in expected.items())
    ):
        raise _fail(path, "wheels.expected_versions must map names to version strings")
    for name, version in expected.items():
        if pinned.get(name.lower()) != f"{name}=={version}":
            raise _fail(path, f"wheels.expected_versions {name} {version} is not pinned so")
    _timeout(wheels["timeout_s"], "wheels.timeout_s", path)


def load_config(path: Path) -> dict[str, Any]:
    """Read and check the download config.

    Args:
        path: ``configs/P6.3-download.yaml``.

    Returns:
        The parsed config.

    Raises:
        BuildError: If the file is not a YAML mapping with exactly the known keys, or a value
            breaks its rule (see the config's comments).
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        raise BuildError(f"{path}: cannot read the config: {err}") from err
    config = _keys(raw, CONFIG_KEYS, "the config", path)
    owner = config["owner"]
    if owner != gateway.KAGGLE_OWNER:
        raise _fail(path, f"owner {owner!r} is not KAGGLE_OWNER {gateway.KAGGLE_OWNER!r}")
    if not isinstance(config["reference_metadata"], str):
        raise _fail(path, "reference_metadata must be a path")
    _check_model(config["model"], owner, path)
    _check_wheels(config["wheels"], owner, path)
    if config["model"]["kaggle_notebook"] == config["wheels"]["kaggle_notebook"]:
        raise _fail(path, "the two notebooks need two slugs")
    return config


def docker_image(repo: Path, relative: str) -> str:
    """Return the image of the reference notebook's pulled metadata.

    Args:
        repo: The repository root.
        relative: ``reference_metadata`` from the config.

    Returns:
        The ``docker_image`` value.

    Raises:
        BuildError: If the file is missing, not JSON, or has no single-word image.
    """
    path = repo / relative
    try:
        image = json.loads(path.read_text(encoding="utf-8")).get("docker_image")
    except (OSError, ValueError, AttributeError) as err:
        raise BuildError(f"{path}: cannot read docker_image: {err}") from err
    if not isinstance(image, str) or len(image.split()) != 1:
        raise BuildError(f"{path}: docker_image must be one image name; got {image!r}")
    return image


def kernel_metadata(notebook: str, code_file: str, image: str) -> dict[str, Any]:
    """Return the ``kernel-metadata.json`` of one download notebook.

    Why: every field the gateway checks is explicit: private, internet on (allowed for CPU
    pushes only), no accelerator in any field (``uses_gpu`` false, so no brief is needed), the
    reference image pinned, no attached source. The title slugifies to the slug.

    Args:
        notebook: ``<owner>/<slug>``.
        code_file: The generated script's file name.
        image: The image to pin.

    Returns:
        The metadata, in the order it is written.

    Raises:
        BuildError: If the gateway would refuse it or treat it as an accelerated push.
    """
    slug = notebook.split("/")[1]
    meta = {
        "id": notebook,
        "title": slug.replace("-", " "),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "docker_image_pinning_type": "original",
        "docker_image": image,
        "competition_sources": [],
        "model_sources": [],
        "kernel_sources": [],
        "dataset_sources": [],
    }
    try:
        if gateway.uses_gpu(meta, None):
            raise BuildError(f"{notebook} would count as an accelerated push")
        gateway.check_kernel_metadata(meta, False)
        gateway.check_flags(meta, gateway.accelerator_id(meta, None))
    except gateway.GatewayError as err:
        raise BuildError(f"the gateway would refuse {notebook}: {err}") from err
    return meta


def _sources(functions: tuple[Callable[..., Any], ...]) -> str:
    return "\n\n".join(inspect.getsource(function) for function in functions)


def installer_source(build: Mapping[str, Any]) -> str:
    """Return the text of ``install_wheelhouse.py``, which the wheels notebook ships.

    Args:
        build: Provenance printed in its header.

    Returns:
        A standalone script (standard library only) running ``installer_main``.
    """
    header = (
        f"# Generated by scripts/build_download_notebooks.py at commit {build['commit']} "
        f"(task P6.3); do not edit.\n"
        '"""Install the agent engine\'s wheelhouse offline into a fresh venv (task P6.3).\n\n'
        "Usage: python install_wheelhouse.py --wheelhouse <dir> --venv <dir> --report <json>\n"
        '[--log <file>] [--imports torch,vllm] [--optional-imports flashinfer]\n"""\n\n'
    )
    body = _sources(INSTALLER_FUNCTIONS + (installer_main,))
    tail = '\n\nif __name__ == "__main__":\n    sys.exit(installer_main(sys.argv[1:]))\n'
    return header + IMPORTS + "\n\n" + body + tail


def kernel_source(kind: str, settings: Mapping[str, Any], build: Mapping[str, Any]) -> str:
    """Return the text of one generated kernel script.

    Args:
        kind: ``model`` or ``wheels``.
        settings: The kind's config section, baked in as ``SETTINGS``.
        build: Provenance (commit, config, its sha256), baked in as ``BUILD``.

    Returns:
        The script: header, imports, ``KIND``, ``SETTINGS``, ``BUILD``, the runtime functions
        verbatim (and ``INSTALLER_SOURCE`` for the wheels kernel), and a call of ``kernel_main``.
    """
    what = {
        "model": "download and verify the INT4 weights into /kaggle/working",
        "wheels": "download the engine wheelhouse and check its offline install",
    }[kind]
    header = (
        f"# Generated by scripts/build_download_notebooks.py at commit {build['commit']} from "
        f"{build['config']} (sha256 {build['config_sha256']}); do not edit, rebuild instead.\n"
        f'"""Download notebook of task P6.3 ({kind}): {what}."""\n\n'
    )
    parts = [header + IMPORTS]
    parts.append(f"KIND = {kind!r}")
    parts.append("SETTINGS = " + pprint.pformat(dict(settings), width=99, sort_dicts=False))
    parts.append("BUILD = " + pprint.pformat(dict(build), width=99, sort_dicts=False))
    functions = COMMON_FUNCTIONS + (MODEL_FUNCTIONS if kind == "model" else WHEELS_FUNCTIONS)
    if kind == "wheels":
        parts.append(f"INSTALLER_SOURCE = {installer_source(build)!r}")
    parts.append(_sources(functions + (kernel_main,)))
    parts.append(
        'if __name__ == "__main__":\n'
        "    # No sys.exit: under Kaggle's IPython kernel SystemExit fails the cell, and the\n"
        "    # verdict is in run_logs/status.json and the last line of the session log.\n"
        "    kernel_main(KIND, SETTINGS, BUILD)\n"
    )
    return "\n\n\n".join(parts)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        raise BuildError(f"git {' '.join(args)} failed: {err}") from err


def build(config_path: Path, out: Path, commit: str) -> dict[str, Path]:
    """Write both push folders and ``MANIFEST.md``.

    Args:
        config_path: The download config.
        out: The build folder, for example ``dist/P6.3-download``; replaced if present.
        commit: The commit the build records.

    Returns:
        The push folder of each kind.

    Raises:
        BuildError: If the config, the reference metadata, or a generated file is refused;
            nothing is written then.
    """
    config = load_config(config_path)
    image = docker_image(REPO, config["reference_metadata"])
    data = config_path.read_bytes()
    try:
        shown = config_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        shown = config_path.as_posix()
    provenance = {
        "commit": commit,
        "config": shown,
        "config_sha256": hashlib.sha256(data).hexdigest(),
    }
    files: dict[str, dict[str, str]] = {}
    for kind in KINDS:
        settings = config[kind]
        notebook = settings["kaggle_notebook"]
        source = kernel_source(kind, settings, {**provenance, "kaggle_notebook": notebook})
        try:
            compile(source, CODE_FILES[kind], "exec")
        except SyntaxError as err:
            raise BuildError(f"generated {CODE_FILES[kind]} does not compile: {err}") from err
        meta = kernel_metadata(notebook, CODE_FILES[kind], image)
        files[kind] = {
            CODE_FILES[kind]: source,
            "kernel-metadata.json": json.dumps(meta, indent=2) + "\n",
        }
    if out.exists():
        shutil.rmtree(out)
    folders = {}
    for kind in KINDS:
        folder = out / config[kind]["kaggle_notebook"].split("/")[1]
        folder.mkdir(parents=True)
        for name, text in files[kind].items():
            (folder / name).write_text(text, encoding="utf-8")
        folders[kind] = folder
    (out / "MANIFEST.md").write_text(manifest_md(config, folders, provenance), encoding="utf-8")
    return folders


def manifest_md(
    config: Mapping[str, Any], folders: Mapping[str, Path], provenance: Mapping[str, str]
) -> str:
    """Return ``MANIFEST.md``: provenance, every file's sha256, and the push commands.

    Args:
        config: The checked config.
        folders: The push folder of each kind.
        provenance: Commit, config, and its sha256.

    Returns:
        The Markdown text.
    """
    model, wheels = config["model"], config["wheels"]
    shown = {
        kind: folder.relative_to(REPO) if folder.is_relative_to(REPO) else folder
        for kind, folder in folders.items()
    }
    push = ".venv/bin/python scripts/kaggle_ops.py push"
    lines = [
        "# P6.3 download notebooks",
        "",
        f"Built from commit `{provenance['commit']}` and `{provenance['config']}` (sha256 "
        f"`{provenance['config_sha256']}`) by `scripts/build_download_notebooks.py`. CPU "
        "sessions with internet on: no GPU quota, no pre-run brief (BRIEF A9).",
        "",
        "| Folder | File | sha256 |",
        "|---|---|---|",
    ]
    for kind in KINDS:
        for path in sorted(folders[kind].iterdir()):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"| `{folders[kind].name}` | `{path.name}` | `{digest}` |")
    total = sum(item["size"] for item in model["files"])
    lines += [
        "",
        "## What each notebook leaves on Kaggle",
        "",
        f"- `{model['kaggle_notebook']}`: `/kaggle/working/{model['folder']}/` with the "
        f"{len(model['files'])} files of `{model['repo_id']}` at `{model['revision']}` "
        f"({total:,} bytes, all hashed), `MANIFEST.json`, `run_logs/`.",
        f"- `{wheels['kaggle_notebook']}`: `/kaggle/working/{wheels['folder']}/` with the "
        f"wheels of {', '.join(wheels['requirements'])} under {len(wheels['constraints'])} "
        "constraints (about 3.95 GB, PyPI sizes), the extras, `requirements.txt`, "
        "`constraints.txt`, `extras.txt`, `install_wheelhouse.py`, `MANIFEST.json`; "
        "`run_logs/install_check.json`.",
        "",
        "## Push (the human runs these; the agent's Kaggle writes are refused)",
        "",
        "```bash",
        f"cd {REPO}",
        f"{push} {shown['model']} --timeout {model['timeout_s']}",
        f"{push} {shown['wheels']} --timeout {wheels['timeout_s']}",
        "```",
        "",
        "Then poll with `scripts/kaggle_ops.py status <owner>/<slug>` and pull the logs with "
        "`scripts/kaggle_ops.py pull <owner>/<slug> --task-id P6.3-dl`; `run_logs/status.json` "
        "holds `ok`. The GPU runs attach version 1 of each output (configs/P6.3-01-script.yaml).",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="configs/P6.3-download.yaml")
    parser.add_argument("--out", required=True, help="build folder, e.g. dist/P6.3-download")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build from a working tree with changes (the build then names HEAD plus changes)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build both push folders from a clean commit.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the folders are written, 2 when the build is refused.
    """
    args = build_parser().parse_args(argv)
    try:
        dirty = _git("status", "--porcelain", "--untracked-files=all").strip()
        if dirty and not args.allow_dirty:
            raise BuildError("the working tree is not clean; commit first:\n" + dirty)
        commit = _git("rev-parse", "HEAD").strip() + ("+dirty" if dirty else "")
        folders = build(Path(args.config), Path(args.out).resolve(), commit)
    except BuildError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    for kind, folder in folders.items():
        print(f"{kind}: {folder}")
    print(f"commands: {Path(args.out) / 'MANIFEST.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
