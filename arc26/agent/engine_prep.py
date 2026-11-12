"""Make an offline vLLM venv bootable, check the GPUs, and read the boot's facts (P6.5b).

Why: run ``P6.3-05`` (``docs/prerun/P6.3-05.md``) ran the long shot after these steps, between
the wheelhouse install and the engine's boot, and the hybrid's shot (:mod:`arc26.agent.shot`)
must start its engine the same way. Without :func:`prepare_engine` the wheelhouse's orphan
FlashInfer provider kills every tensor-parallel worker at the boot (``P6.3-01``, attempt 2).

- :func:`prepare_engine`: drop the orphan jit-cache provider, put an empty ``sitecustomize``
  first on the engine's ``PYTHONPATH``, import what the workers import, write
  ``engine_prep.json``.
- :func:`preflight`: torch from the venv sees the GPUs the server needs (``preflight.json``).
- :func:`parse_boot_log` and :func:`cache_config_info`: the KV pool and the boot's facts from the
  engine log and ``/metrics``.
"""

# Moved from scripts/agent_spike.py (task P6.3, commit acac69d, the code run P6.3-05 ran) without
# change: Stop, write_json, run_command, PREFLIGHT, preflight, JIT_CACHE_PROBE,
# ENGINE_IMPORT_CHECK, QUIET_SITECUSTOMIZE, probe_line, jit_cache_state, drop_orphan_jit_cache,
# quiet_sitecustomize, check_engine_imports, prepare_engine, BOOT_PATTERNS, parse_boot_log, and
# cache_config_info. tests/agent/test_shot.py checks that each source is identical to the
# script's. The script keeps its copies: its past runs' hashes name them.

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class Stop(Exception):
    """A problem that ends the run as an Error (exit 1): missing input, failed install or boot."""


def write_json(path: Path, value: object) -> None:
    """Write JSON through a temporary file and ``os.replace``, so a reader never sees half a file.

    Args:
        path: The file; its folder is created when missing.
        value: A JSON value; other objects are written as their ``str``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_command(
    command: list[str], log: Path, timeout_s: float, env: Mapping[str, str]
) -> tuple[int, list[str]]:
    """Run a command with its output appended to a log; return the exit code and the last lines.

    Args:
        command: Program and arguments; no shell.
        log: The log file.
        timeout_s: Seconds before the command is killed (exit 124).
        env: Its environment.

    Returns:
        The exit code (127 when it cannot start, 124 at the timeout) and its last 50 lines.
    """
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(env),
                text=True, errors="replace", timeout=timeout_s, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired as err:
            text = err.stdout if isinstance(err.stdout, str) else ""
            handle.write(text + f"\ntimeout after {timeout_s} s\n")
            return 124, text.splitlines()[-50:]
        except OSError as err:
            handle.write(f"could not run: {err}\n")
            return 127, [str(err)]
        handle.write(result.stdout)
    return result.returncode, result.stdout.splitlines()[-50:]


PREFLIGHT = """\
import json, sys, torch
want = int(sys.argv[1])
info = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "available": torch.cuda.is_available(), "count": torch.cuda.device_count()}
devices = []
for index in range(info["count"]):
    props = torch.cuda.get_device_properties(index)
    x = torch.randn(512, 512, device=f"cuda:{index}", dtype=torch.bfloat16)
    devices.append({"name": props.name, "capability": f"{props.major}.{props.minor}",
                    "memory_bytes": props.total_memory, "matmul": float((x @ x).float().sum())})
info["devices"] = devices
if info["available"]:
    info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    info["p2p"] = {f"{a}->{b}": torch.cuda.can_device_access_peer(a, b)
                   for a in range(info["count"]) for b in range(info["count"]) if a != b}
keys = ("libnccl", "libcudart", "libcublas", "libcudnn")
with open("/proc/self/maps") as maps:
    info["libraries"] = sorted({line.split()[-1] for line in maps
                                if "/" in line and any(k in line for k in keys)})
info["ok"] = info["available"] and info["count"] >= want
print("PREFLIGHT " + json.dumps(info))
"""


def preflight(python: Path, tp: int, out: Path, env: Mapping[str, str]) -> dict[str, Any]:
    """Check from the venv that torch sees enough working GPUs; record NCCL, P2P, and libraries.

    Args:
        python: The venv's python.
        tp: GPUs the server needs.
        out: The output folder (``preflight.json``).
        env: The engine's environment.

    Returns:
        The probe's findings.

    Raises:
        Stop: If CUDA is unavailable, fewer than ``tp`` GPUs answer, or the probe fails.
    """
    code, tail = run_command(
        [str(python), "-c", PREFLIGHT, str(tp)], out / "preflight.log", 600, env
    )
    info: dict[str, Any] = {"exit": code}
    for line in tail:
        if line.startswith("PREFLIGHT "):
            info.update(json.loads(line[len("PREFLIGHT ") :]))
    write_json(out / "preflight.json", info)
    if code != 0 or not info.get("ok"):
        raise Stop(f"GPU preflight failed (exit {code}): {info or tail[-5:]}")
    return info


# Run by the venv's python: what FlashInfer's import will find as ``flashinfer_jit_cache``.
JIT_CACHE_PROBE = """\
import importlib, importlib.metadata, importlib.util, json, re, sys
shim, providers = None, []
for dist in importlib.metadata.distributions():
    name = re.sub(r"[-_.]+", "-", dist.metadata["Name"] or "").lower()
    if name == "flashinfer-jit-cache":
        shim = dist.version
    elif name.startswith("flashinfer-jit-cache-"):
        providers.append(name)
spec = importlib.util.find_spec("flashinfer_jit_cache")
info = {"prefix": sys.prefix, "shim": shim, "providers": sorted(set(providers)),
        "found": spec is not None, "has_version": False, "locations": []}
if spec is not None:
    info["locations"] = list(spec.submodule_search_locations or [])
    info["has_version"] = hasattr(importlib.import_module("flashinfer_jit_cache"), "__version__")
print("JIT_CACHE " + json.dumps(info))
"""


# Run by the venv's python with module names as arguments: import each, stop at the first error.
ENGINE_IMPORT_CHECK = """\
import importlib, json, sys, time, traceback
report = {"ok": True, "modules": [], "python": sys.executable}
for name in sys.argv[1:]:
    began = time.monotonic()
    try:
        importlib.import_module(name)
    except BaseException as err:
        report.update(ok=False, failed=name, error=f"{type(err).__name__}: {err}",
                      traceback=traceback.format_exc().splitlines()[-40:])
        break
    report["modules"].append([name, round(time.monotonic() - began, 2)])
far = sys.modules.get("vllm.distributed.device_communicators.flashinfer_all_reduce")
if far is not None:
    report["flashinfer_all_reduce_available"] = bool(getattr(far, "fi_ar_available", False))
print("ENGINE_IMPORTS " + json.dumps(report), flush=True)
sys.exit(0 if report["ok"] else 1)
"""


QUIET_SITECUSTOMIZE = '''\
"""Empty on purpose: written by scripts/agent_spike.py (quiet_sitecustomize).

Kaggle's image installs its sitecustomize as /usr/lib/python3.11/sitecustomize.py, on the path of
every Python 3.11, venvs included. Its second statement imports ``log`` from the image's
dist-packages, which a venv does not see, so in the engine's processes it stopped there and only
printed "Error in sitecustomize ... No module named 'log'". This module, first on PYTHONPATH,
takes its place.
"""
'''


def probe_line(tail: Sequence[str], tag: str) -> dict[str, Any] | None:
    """Return the JSON a probe printed after ``tag``, from the last such line, or None."""
    for line in reversed(tail):
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1 :])
    return None


def jit_cache_state(python: Path, env: Mapping[str, str], log: Path) -> dict[str, Any]:
    """Report what ``flashinfer_jit_cache`` is in a venv, as FlashInfer's import will see it.

    Args:
        python: The venv's python.
        env: The engine's environment.
        log: The log the probe appends to.

    Returns:
        ``prefix`` (the venv), ``shim`` (the ``flashinfer-jit-cache`` version or None),
        ``providers`` (installed ``flashinfer-jit-cache-*`` distributions), ``found`` (the name
        imports), ``has_version`` (it has ``__version__``), and its ``locations``.

    Raises:
        Stop: If the probe does not run.
    """
    code, tail = run_command([str(python), "-c", JIT_CACHE_PROBE], log, 300, env)
    state = probe_line(tail, "JIT_CACHE")
    if code != 0 or state is None:
        raise Stop(f"the jit-cache probe failed (exit {code}): {tail[-5:]}")
    return state


def drop_orphan_jit_cache(python: Path, env: Mapping[str, str], log: Path) -> dict[str, Any]:
    """Remove FlashInfer jit-cache providers installed without their shim; else do nothing.

    Why: P6.3-01's attempt 2 died at the boot in every tensor-parallel worker. vLLM 0.31.0's
    ``CudaCommunicator`` always imports its FlashInfer all-reduce module, which runs ``import
    flashinfer.comm`` inside ``except ImportError``; FlashInfer 0.7.0.post1 (``jit/env.py``) then
    finds a module ``flashinfer_jit_cache`` and reads its ``__version__``. The wheelhouse holds
    the sm_89 provider ``flashinfer-jit-cache-sm89`` (package
    ``flashinfer_jit_cache.providers.sm89``) but not the shim ``flashinfer-jit-cache`` that owns
    ``flashinfer_jit_cache/__init__.py``, its ``__version__``, and the provider registry. Alone,
    the provider leaves a namespace package, and the read raises an AttributeError that the
    guard does not catch. FlashInfer reaches a provider only through the shim, so this one was
    never usable: removing it changes nothing else. ``pip uninstall`` keeps the empty namespace
    folders, which still import, so empty ones inside the venv go too.

    Args:
        python: The venv's python (pip is installed in it).
        env: The engine's environment.
        log: The log every command appends to.

    Returns:
        The record: ``state`` ("absent", "ready" with the shim, or "orphan"), ``before``,
        ``removed`` (distributions), ``folders`` (removed), and ``after``.

    Raises:
        Stop: If pip fails, or ``flashinfer_jit_cache`` still imports without ``__version__``.
    """
    before = jit_cache_state(python, env, log)
    record: dict[str, Any] = {"before": before, "removed": [], "folders": []}
    if not before["found"] or before["has_version"]:
        record["state"] = "ready" if before["found"] else "absent"
        return record
    record["state"] = "orphan"
    if before["providers"]:
        command = [
            str(python), "-m", "pip", "uninstall", "--yes", "--disable-pip-version-check",
            *before["providers"],
        ]  # fmt: skip
        code, tail = run_command(command, log, 300, env)
        if code != 0:
            raise Stop(f"pip could not remove {before['providers']} (exit {code}): {tail[-5:]}")
        record["removed"] = before["providers"]
    prefix = Path(before["prefix"]).resolve()
    for location in before["locations"]:
        folder = Path(location).resolve()
        inside = prefix in folder.parents and folder.is_dir() and not folder.is_symlink()
        if inside and all(p.is_dir() and not p.is_symlink() for p in folder.rglob("*")):
            shutil.rmtree(folder)
            record["folders"].append(str(folder))
    after = record["after"] = jit_cache_state(python, env, log)
    if after["found"] and not after["has_version"]:
        raise Stop(
            f"flashinfer_jit_cache still imports without __version__ from {after['locations']} "
            f"after removing {record['removed']} and {record['folders']}; FlashInfer's import "
            "would fail in every worker"
        )
    return record


def quiet_sitecustomize(folder: Path, env: Mapping[str, str]) -> dict[str, str]:
    """Return the engine's environment with an empty ``sitecustomize`` first on its path.

    Why: the image's ``sitecustomize`` (see ``QUIET_SITECUSTOMIZE``) fails in the venv and prints
    two lines of noise into every engine process's log. Replacing it is safe because it already
    stopped at its second statement there, so the engine loses nothing, and the folder holds only
    this file. The alternatives do not work: ``PYTHONNOUSERSITE`` does not reach the standard
    library's folder, and ``-S`` would also drop the venv's site-packages.

    Args:
        folder: Where to write ``sitecustomize.py`` (scratch).
        env: The engine's environment, without ``PYTHONPATH`` (``clean_env``).

    Returns:
        A copy of ``env`` with ``PYTHONPATH`` set to ``folder``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sitecustomize.py").write_text(QUIET_SITECUSTOMIZE, encoding="utf-8")
    return {**env, "PYTHONPATH": str(folder)}


def check_engine_imports(
    python: Path, env: Mapping[str, str], modules: Sequence[str], log: Path, timeout_s: float
) -> dict[str, Any]:
    """Import, with the venv's python and the engine's environment, what its workers import.

    Why: in attempt 2 the install's probes (``import torch``, ``import vllm``) passed and the
    workers then failed on an import vLLM makes only inside them. One process importing the same
    modules stops a broken venv in under a minute, with the error, before a boot that loads
    19.5 GB.

    Args:
        python: The venv's python.
        env: The engine's environment.
        modules: Module names, imported in this order.
        log: The log the check appends to.
        timeout_s: Seconds allowed.

    Returns:
        The report: ``ok``, each module's import seconds, and on a failure the module, the error,
        and its traceback; ``flashinfer_all_reduce_available`` once vLLM's FlashInfer all-reduce
        module imported.
    """
    code, tail = run_command(
        [str(python), "-c", ENGINE_IMPORT_CHECK, *modules], log, timeout_s, env
    )
    report: dict[str, Any] = {"exit": code, **(probe_line(tail, "ENGINE_IMPORTS") or {})}
    if "ok" not in report:
        report.update(ok=False, error=f"no report from the check: {tail[-5:]}")
    report["ok"] = bool(report["ok"]) and code == 0
    return report


def prepare_engine(
    python: Path,
    env: Mapping[str, str],
    settings: Mapping[str, Any],
    out: Path,
    scratch: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Make an installed engine venv bootable and prove it: the step between install and boot.

    In order: ``drop_orphan_jit_cache`` (attempt 2's fault), ``quiet_sitecustomize`` (the
    image's failing ``sitecustomize``), and ``check_engine_imports`` (the workers' imports, in
    the environment the engine gets). Every engine launched from this wheelhouse needs it after
    the install.

    Args:
        python: The venv's python.
        env: The engine's environment so far (``clean_env`` plus ``server.env``).
        settings: The config's ``wheels`` section (``engine_imports``,
            ``engine_check_timeout_s``).
        out: The output folder (``engine_prep.json``, ``engine_prep.log``).
        scratch: The scratch folder (the empty ``sitecustomize``).

    Returns:
        The environment to start the engine with, and the record (also ``engine_prep.json``).

    Raises:
        Stop: If a step fails, with the error; the record is written first.
    """
    log = out / "engine_prep.log"
    record: dict[str, Any] = {"python": str(python)}
    began = time.monotonic()
    try:
        record["jit_cache"] = drop_orphan_jit_cache(python, env, log)
        env = quiet_sitecustomize(scratch / "engine_site", env)
        record["pythonpath"] = env["PYTHONPATH"]
        imports = record["imports"] = check_engine_imports(
            python, env, settings["engine_imports"], log, settings["engine_check_timeout_s"]
        )
        if not imports["ok"]:
            raise Stop(
                f"engine import check failed at {imports.get('failed')} (exit "
                f"{imports['exit']}): {imports.get('error')}"
            )
    except Stop as err:
        record["error"] = str(err)
        raise
    finally:
        record["seconds"] = round(time.monotonic() - began, 1)
        write_json(out / "engine_prep.json", record)
    return dict(env), record


BOOT_PATTERNS = {
    "kv_cache_tokens": r"KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
    "weights_load_s": r"Loading weights took ([\d.]+) seconds",
    "model_load": r"Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds",
    "compile_s": r"torch\.compile takes ([\d.]+) s in total",
    "graph_capture": r"Graph capturing finished in ([\d.]+) secs?(?:, took ([\d.-]+) GiB)?",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
    "attention_backend": r"Using ([A-Z][A-Z_0-9]+) (?:attention )?backend",
    "nccl_version": r"NCCL version ([\d.]+[^\s]*)",
    "nccl_transport": r"via (P2P|SHM|NET|NVLS|CollNet)",
    "custom_allreduce": r"(?i)(custom all-?reduce[^\n]{0,160})",
    "marlin": r"(?i)(using [^\n]{0,40}marlin[^\n]{0,80})",
}


def parse_boot_log(text: str) -> dict[str, Any]:
    """Extract the engine's boot facts from its log.

    Args:
        text: The server log.

    Returns:
        For each of ``BOOT_PATTERNS``, up to 8 matches (a group tuple or a string), plus counts of
        tracebacks, ERROR lines, and lines mentioning NaN, and ``kv_cache_tokens_first`` and
        ``max_concurrency_first`` as numbers when found.
    """
    found: dict[str, Any] = {}
    for name, pattern in BOOT_PATTERNS.items():
        matches = re.findall(pattern, text)
        if name == "nccl_transport":
            found[name] = {kind: matches.count(kind) for kind in sorted(set(matches))}
        else:
            found[name] = matches[:8]
    if found["kv_cache_tokens"]:
        found["kv_cache_tokens_first"] = int(found["kv_cache_tokens"][0].replace(",", ""))
    if found["max_concurrency"]:
        length, factor = found["max_concurrency"][0]
        found["max_concurrency_first"] = {
            "tokens": int(length.replace(",", "")),
            "x": float(factor),
        }
    found["tracebacks"] = text.count("Traceback (most recent call last)")
    found["error_lines"] = len(re.findall(r"(?m)^.*\bERROR\b.*$", text))
    found["nan_lines"] = len(re.findall(r"(?mi)^.*\bnan\b.*$", text))
    return found


def cache_config_info(text: str) -> dict[str, str]:
    """Return the labels of ``vllm:cache_config_info`` (block size, GPU blocks), if present.

    Args:
        text: The body of ``/metrics``.

    Returns:
        Label names to values; empty when the sample is absent.
    """
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            return dict(re.findall(r'(\w+)="([^"]*)"', line))
    return {}
