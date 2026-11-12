r"""Bisect probe of version 11's engine faults: boot several engine variants, one request each.

Purpose: run ``P5.2-06`` (task P5.2m, brief ``docs/prerun/P5.2-06.md``). ``P5.2-05`` on Modal's
2 x L40S (sm89) hit ``an illegal memory access`` in both boots: the first in the decode CUDA graph
capture, the second (BF16 KV, no graphs) at the first prefill after ``Uvicorn running``. CUDA
reports such a fault at a later launch, so neither traceback names the kernel. This script boots a
list of engine variants in sequence on the same install and sends each one tiny chat request; per
variant it records whether the server survived, the first error with its full traceback, and the
frame that launched the failing kernel (exact under ``CUDA_LAUNCH_BLOCKING=1``). Runbook
``docs/runbooks/fn_bisect.md``.

It runs where version 11's ``fn_probe.py`` runs (the notebook's cell 2 names it, by the run
config's ``entry``) and reuses ``fn_probe.py`` unchanged for everything before the server: the
inputs, the local copy, Python 3.12, the venv from the wheelhouse with version 11's three
hash-guarded edits, and the CUDA toolkit links. Only the inner process differs: no tasks, no
tool self-test, no solve.

A variant is a set of extra server flags (``null`` removes a flag), environment variables
(``null`` removes one), and optional hash-guarded edits (``fn_probe.apply_patch``'s items), all
on top of version 11's flags and the edits' fallback flags. An edit applies only for its variant:
the file's bytes are restored after the server stops, so the next variant sees version 11's file.

Time. With $t_0$ the notebook's first cell, $E$ the development ``end_s``, $B$ a variant's
``budget_s``, and $K$ the kill reserve, a variant starting at $t$ runs only if
$E - t \ge m$ (``min_variant_s``), and its deadline is

.. math:: d = \min\left(t + B,\; t_0 + E - K\right),

so the bisect ends by $t_0 + E$ whatever the variants do. ``bisect.json`` is rewritten before
each boot (the running row) and after each variant, so a stop at any moment keeps every finished
row.

Outputs under ``/kaggle/working/run_logs/fn/`` (pulled): ``bisect.json``, per variant
``bisect_<n>_<name>_tail.log`` and a bounded ``bisect_<n>_<name>_engine.log``, plus
``fn_probe.py``'s outer records (``run.json``, ``state.json``, ``inputs.json``,
``install.json``, ``patches.json``, ``versions.txt``) and ``summary.md``.

Usage::

    python fn_bisect.py --config fn_config.json --root / --session-start <t0>   # cell 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# fn_probe.py from the same folder (the code dataset, or scripts/ in tests), unchanged.
_spec = importlib.util.spec_from_file_location("_fn_probe_for_bisect", HERE / "fn_probe.py")
fp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fp
_spec.loader.exec_module(fp)

SCHEMA = 1
BISECT_FILE = "bisect.json"
# The variable that tags one variant's server processes, so a rank that outlived its group's
# kill is found by its environment (as modal_probe.marked_pids finds an attempt's).
VARIANT_ENV = "ARC26_BISECT_VARIANT"
# A variant's name becomes part of pulled file names (modal_probe.PROBE_PULL_PATTERN's segment).
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,47}")
TRACEBACK = "Traceback (most recent call last)"
# A log line of SGLang's own logger: "[2026-10-30 09:32:28 TP0 EP0] ...".
STAMPED = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")
FRAME = re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')
RANK = re.compile(r"\b(TP\d+(?: [A-Z]+\d+)*)\]")
# A Triton launch in a frame's code line: kernel_name[grid](...).
TRITON_LAUNCH = re.compile(r"\b([A-Za-z_]\w*)\[[^\]]*\]\(")
# When a variant runs: always, or only when no earlier variant survived (a diagnostic run that
# only pays when every fix failed).
RUN_IF = ("always", "none_survived")
OUTCOMES = ("survived", "exited_at_boot", "boot_timeout", "request_failed", "crashed_in_request",
            "died_after_request", "edit_failed", "not_run")


# ---- the log ------------------------------------------------------------------------------------


def _short(path: str) -> str:
    """Return a frame's file from its package on (the part after ``site-packages/``)."""
    return path.split("site-packages/", 1)[-1]


def first_error(text: str, patterns: Sequence[str], ignore: Sequence[str],
                skip_frames: Sequence[str], max_lines: int = 400) -> dict[str, Any] | None:
    """Return the first fatal error of an engine log with its traceback and launching frame.

    Why: the fork logs harmless tracebacks at import (torchcodec's missing FFmpeg) before any
    fault, and a CUDA fault cascades into one traceback per rank; the first fatal line names the
    fault, its traceback the call that met it. Under ``CUDA_LAUNCH_BLOCKING=1`` every launch
    synchronizes, so the deepest frame outside the launch machinery is the faulting kernel's.

    Args:
        text: The engine log.
        patterns: Regexes of a fatal line (any one matches).
        ignore: Regexes of lines never taken as fatal.
        skip_frames: Path parts of frames that only dispatch a launch (``torch/nn/modules/``,
            ``triton/runtime/``), skipped when naming the kernel's frame.
        max_lines: Lines searched back for the traceback's start, and kept in the excerpt.

    Returns:
        None when no line matches; else ``line`` (1-based), ``fatal_line``, ``message`` (the
        exception's line and its continuation), ``rank``, ``traceback`` (the excerpt),
        ``frames`` (the last eight, ``file:line func``), ``kernel_frame``, ``kernel_call`` (its
        code line), ``triton_kernel`` (the name launched there, if any), ``fatal_lines`` (how
        many lines match in all).
    """
    lines = text.splitlines()
    fatal = re.compile("|".join(f"(?:{p})" for p in patterns))
    skip = re.compile("|".join(f"(?:{p})" for p in ignore)) if ignore else None
    hits = [i for i, line in enumerate(lines)
            if fatal.search(line) and not (skip and skip.search(line))]
    if not hits:
        return None
    first = hits[0]
    start = first
    for j in range(first, max(-1, first - max_lines), -1):
        if TRACEBACK in lines[j]:
            start = j
            break
        if j < first and STAMPED.match(lines[j]):
            break
    message_at = first
    if TRACEBACK in lines[first] or start < first:
        for k in range(max(start + 1, first), min(len(lines), start + max_lines)):
            line = lines[k]
            if line and not line[0].isspace() and TRACEBACK not in line:
                message_at = k
                break
    end = message_at
    while end + 1 < len(lines) and end - message_at < 8:
        nxt = lines[end + 1]
        if not nxt.strip() or STAMPED.match(nxt) or nxt.startswith(("During handling",
                                                                     "The above exception")):
            break
        end += 1
    frames = []
    for k in range(start, message_at):
        found = FRAME.match(lines[k])
        if found:
            code = lines[k + 1].strip() if k + 1 < message_at else ""
            frames.append((found["file"], int(found["line"]), found["func"].strip(), code))
    launch = [f for f in frames if not any(part in f[0] for part in skip_frames)]
    kernel = launch[-1] if launch else (frames[-1] if frames else None)
    rank = RANK.search(lines[start]) or RANK.search(lines[first])
    triton = TRITON_LAUNCH.search(kernel[3]) if kernel else None
    excerpt = lines[start:min(end + 1, start + max_lines)]
    return {
        "line": first + 1,
        "fatal_line": lines[first][:500],
        "message": "\n".join(lines[message_at:end + 1])[:2000],
        "rank": rank.group(1) if rank else None,
        "traceback": "\n".join(excerpt)[-20000:],
        "frames": [f"{_short(f)}:{n} {fn}" for f, n, fn, _ in frames[-8:]],
        "kernel_frame": f"{_short(kernel[0])}:{kernel[1]} {kernel[2]}" if kernel else None,
        "kernel_call": kernel[3][:300] if kernel else None,
        "triton_kernel": triton.group(1) if triton else None,
        "fatal_lines": len(hits),
    }


def log_marks(text: str, marks: Mapping[str, str]) -> dict[str, Any]:
    """Return each mark's first match: its first group as a number or text, else True; or None.

    Args:
        text: The engine log.
        marks: Name to regex.

    Returns:
        Name to value.
    """
    out: dict[str, Any] = {}
    for name, pattern in marks.items():
        found = re.search(pattern, text)
        if found is None:
            out[name] = None
        elif found.groups():
            value = found.group(1)
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                out[name] = value
        else:
            out[name] = True
    return out


def bounded(data: bytes, limit: int) -> bytes:
    """Return ``data`` cut to ``limit`` bytes: the first eighth and the rest from the end."""
    if len(data) <= limit:
        return data
    return data[: limit // 8] + b"\n[... cut ...]\n" + data[-(limit - limit // 8):]


# ---- one variant --------------------------------------------------------------------------------


def check_variants(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the variant list and return it as plain values.

    Raises:
        ValueError: If a name is unsafe for a file name or repeated, or a field is unknown.
    """
    seen: set[str] = set()
    allowed = {"name", "why", "flags", "env", "edits", "run_if", "logs"}
    variants = json.loads(json.dumps(list(spec["variants"])))
    if not variants:
        raise ValueError("bisect.variants is empty")
    for variant in variants:
        name = str(variant.get("name", ""))
        if not NAME.fullmatch(name) or name in seen:
            raise ValueError(f"variant name {name!r} is unsafe or repeated")
        if set(variant) - allowed:
            raise ValueError(f"variant {name}: unknown {sorted(set(variant) - allowed)}")
        seen.add(name)
        if variant.get("run_if", "always") not in RUN_IF:
            raise ValueError(f"variant {name}: run_if must be one of {RUN_IF}")
        for item in variant.get("edits") or []:
            if not {"name", "file", "sha256_before", "pattern", "replacement"} <= set(item):
                raise ValueError(f"variant {name}: an edit needs name, file, sha256_before, "
                                 "pattern, replacement (hash-guarded)")
    return variants


def fill(text: str, values: Mapping[str, str]) -> str:
    """Replace ``{name}`` placeholders (``out``, ``scratch``, ``variant``) in a value."""
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def variant_env(base: Mapping[str, str], extra: Mapping[str, Any] | None, tag: str,
                values: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the server's environment: ``base`` with the variant's values (None removes one).

    Args:
        base: The inner process's environment (version 11's engine variables among it).
        extra: The variant's variables; a ``{placeholder}`` in a value is filled from ``values``.
        tag: The value of ``VARIANT_ENV``.
        values: Placeholder values.

    Returns:
        The environment.
    """
    env = dict(base)
    for key, value in (extra or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[str(key)] = fill(str(value), values or {})
    env[VARIANT_ENV] = tag
    return env


def tagged_pids(tag: str, proc: Path = Path("/proc")) -> list[int]:
    """Return the processes whose environment holds ``VARIANT_ENV=<tag>``."""
    needle = f"{VARIANT_ENV}={tag}".encode()
    found = []
    for environ in proc.glob("[0-9]*/environ"):
        try:
            if needle in environ.read_bytes().split(b"\0"):
                found.append(int(environ.parent.name))
        except (OSError, ValueError):
            continue
    return sorted(found)


def port_free(port: int) -> bool:
    """Whether nothing listens on ``127.0.0.1:port``."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def gpu_used() -> list[float] | str:
    """Return each GPU's used MiB from ``nvidia-smi``, or why it is unreadable."""
    gpus = fp.gpu_memory()
    return [g["used_mib"] for g in gpus] if isinstance(gpus, list) else gpus


def stop_variant(inner: Any, tag: str, settle_s: float,
                 before: list[float] | str | None = None) -> dict[str, Any]:
    """Stop the variant's server, every process it tagged, and wait for its port and GPUs.

    Why: the next variant binds the same port and needs the same GPU memory; a rank that
    outlived its process group would make the next boot fail for a reason that is not its own.

    Args:
        inner: The inner state.
        tag: The variant's ``VARIANT_ENV`` value.
        settle_s: Seconds the wait may take.
        before: Each GPU's used MiB before the variant's launch; the wait ends when no GPU holds
            more than ``bisect.gpu_free_mib`` above it.

    Returns:
        ``killed`` (tagged pids still alive after the group's stop), ``port_free``,
        ``gpu_used_mib`` (per GPU after the stop, or the reason it is unreadable), ``seconds``.
    """
    began = fp._now()
    fp.kill_server(inner)
    left = tagged_pids(tag)
    for pid in left:
        try:
            os.kill(pid, 9)
        except OSError:
            continue
    port = int(inner.run.config["engine"]["port"])
    margin = float(inner.run.config["bisect"].get("gpu_free_mib", 2048))

    def held(used: list[float] | str) -> bool:
        if not isinstance(used, list) or not isinstance(before, list) or len(used) != len(before):
            return False
        return any(u > b + margin for u, b in zip(used, before, strict=True))

    # The ranks' memory is freed after their exit, not at it (P5.2-06: V2's stop read 41,881 MiB
    # held 0.2 s after its kill), so the wait also covers the GPUs, when nvidia-smi reads them.
    used = gpu_used()
    while fp._now() - began < settle_s and (tagged_pids(tag) or not port_free(port)
                                            or held(used)):
        time.sleep(0.5)
        used = gpu_used()
    return {"killed": left, "port_free": port_free(port), "gpu_used_mib": used,
            "seconds": round(fp._now() - began, 1)}


def apply_edits(site: Path | None, edits: Sequence[Mapping[str, Any]],
                restore: list[tuple[Path, bytes]]) -> tuple[list[dict[str, Any]], bool]:
    """Apply a variant's hash-guarded edits; record each file's bytes for the restore.

    Args:
        site: The venv's site-packages, or None (no venv: the smoke, whose interpreter is the
            developer's and is never edited).
        edits: ``fn_probe.apply_patch`` items.
        restore: Filled with ``(path, original bytes)`` for every file an edit wrote.

    Returns:
        The results and whether every edit is in place (``applied`` or ``already applied``).
    """
    results = []
    for item in edits:
        if site is None:
            results.append({"name": item["name"], "file": item["file"], "status": "no venv"})
            return results, False
        path = site / item["file"]
        original = path.read_bytes() if path.is_file() else None
        result = fp.apply_patch(site, item)
        result.pop("context", None)
        if result["status"] == "applied" and original is not None:
            restore.append((path, original))
        results.append(result)
        if result["status"] not in {"applied", "already applied"}:
            return results, False
    return results, True


def restore_edits(restore: list[tuple[Path, bytes]]) -> list[str]:
    """Write back every edited file's original bytes, last edit first; return the files."""
    done = []
    for path, original in reversed(restore):
        path.write_bytes(original)
        done.append(path.name)
    restore.clear()
    return done


def edit_site(run: Any) -> Path | None:
    """Return the venv's site-packages, or None when the install did not build one."""
    venv_python = run.state.get("venv_python")
    if not venv_python or not run.config["install"].get("enabled", True):
        return None
    sites = sorted(Path(venv_python).parent.parent.glob("lib/python*/site-packages"))
    return sites[0] if sites else None


def wait_healthy(inner: Any, deadline: float, poll_s: float,
                 seen: dict[str, Any] | None = None) -> str:
    """Poll ``/health`` until 200, the process's exit, or the deadline.

    Args:
        inner: The inner state (``proc``, ``base``).
        deadline: ``fn_probe._now()`` value at which the wait ends.
        poll_s: Seconds between polls.
        seen: Filled with ``last_status`` (the last HTTP status, None before the server listens)
            and ``first_answer_s`` / ``non_200_s`` (when it first answered, and for how long it
            answered with another status: P5.2-06's eager boot answered 503 for 44 s while its
            warm-up compiled).

    Returns:
        ``healthy``, ``exited``, or ``timeout``.
    """
    seen = {} if seen is None else seen
    began = fp._now()
    seen.update(last_status=None, first_answer_s=None, non_200_s=0.0)
    while fp._now() < deadline:
        if inner.proc.poll() is not None:
            return "exited"
        try:
            status, _ = fp.http_json(inner.base, "GET", "/health", timeout=5)
            seen["last_status"] = status
            if seen["first_answer_s"] is None:
                seen["first_answer_s"] = round(fp._now() - began, 1)
            if status == 200:
                return "healthy"
            seen["non_200_s"] = round(fp._now() - began - seen["first_answer_s"], 1)
        except OSError:
            pass
        time.sleep(poll_s)
    return "exited" if inner.proc.poll() is not None else "timeout"


def server_alive(inner: Any) -> bool:
    """Whether the variant's server still runs and answers ``/health`` with 200."""
    if inner.proc is None or inner.proc.poll() is not None:
        return False
    try:
        return fp.http_json(inner.base, "GET", "/health", timeout=10)[0] == 200
    except OSError:
        return False


def send_request(inner: Any, messages: Sequence[Mapping[str, str]], max_tokens: int,
                 extra: Mapping[str, Any] | None, timeout: float) -> dict[str, Any]:
    """Send one chat request; return its status, token counts, seconds, and tokens per second.

    The rate is $r = n / s$ for $n$ completion tokens in $s$ seconds of wall time, the prefill
    included (a few hundred tokens, served from the radix cache after the first request).
    """
    config = inner.run.config
    body = {"model": config["engine"]["served_model_name"], "messages": list(messages),
            "max_tokens": int(max_tokens), **(extra or {})}
    began = fp._now()
    try:
        status, reply = fp.http_json(inner.base, "POST", "/v1/chat/completions", body,
                                     timeout=timeout)
    except OSError as err:
        return {"status": None, "error": fp._describe(err),
                "seconds": round(fp._now() - began, 1)}
    usage = reply.get("usage") if isinstance(reply, dict) else None
    out: dict[str, Any] = {"status": status, "seconds": round(fp._now() - began, 1),
                           "prompt_tokens": (usage or {}).get("prompt_tokens"),
                           "completion_tokens": (usage or {}).get("completion_tokens")}
    made = out["completion_tokens"]
    out["wall_tok_s"] = (round(made / out["seconds"], 2)
                         if isinstance(made, int) and made > 0 and out["seconds"] > 0 else None)
    if status != 200:
        out["error"] = fp._tail(json.dumps(reply) if not isinstance(reply, str) else reply, 1000)
    return out


def log_decode_rate(text: str) -> dict[str, Any]:
    """Return the server's own decode rate lines in ``text``: count, median, and last value.

    Why: SGLang logs ``gen throughput (token/s)`` every decode-log interval; with one request
    running it is the decode rate alone, without the prefill the wall rate includes.
    """
    rates = sorted(float(m.group(2)) for m in fp.DECODE_LINE.finditer(text)
                   if int(m.group(1)) >= 1 and float(m.group(2)) > 0)
    if not rates:
        return {"lines": 0, "median_tok_s": None}
    return {"lines": len(rates), "median_tok_s": rates[len(rates) // 2]}


def pool_reading(inner: Any) -> dict[str, Any]:
    """Return the KV pool and request limit from ``/get_server_info`` (None when unreadable)."""
    try:
        _, info = fp.http_json(inner.base, "GET", "/get_server_info", timeout=30)
    except OSError as err:
        return {"pool_tokens": None, "error": fp._describe(err)}
    pool = fp.find_key(info, "max_total_num_tokens")
    running = fp.find_key(info, "max_running_requests")
    return {"pool_tokens": None if pool is None else int(pool),
            "max_running_requests": None if running is None else int(running)}


def run_variant(inner: Any, variant: Mapping[str, Any], index: int, site: Path | None,
                deadline: float) -> dict[str, Any]:
    """Boot one variant, send one request, stop it, and read its log.

    Args:
        inner: ``fn_probe.Inner`` (the run, the base URL; ``proc`` set here).
        variant: ``name``, ``why``, ``flags``, ``env``, ``edits``.
        index: The variant's position (file names).
        site: The venv's site-packages for the edits, or None.
        deadline: ``fn_probe._now()`` value by which the request must have ended.

    Returns:
        The row of ``bisect.json``.
    """
    run = inner.run
    config = run.config
    spec = config["bisect"]
    name = variant["name"]
    began = fp._now()
    row: dict[str, Any] = {
        "index": index, "name": name, "why": variant.get("why", ""),
        "flags": variant.get("flags") or {}, "env": variant.get("env") or {},
        "started_s": round(began - run.start, 1), "deadline_s": round(deadline - run.start, 1),
    }
    restore: list[tuple[Path, bytes]] = []
    tag = f"{name}-{uuid.uuid4().hex[:8]}"
    folder = run.path(config["paths"]["scratch"]) / "bisect" / f"{index}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "engine.log"
    values = {"out": str(run.out), "scratch": str(folder), "variant": name}
    try:
        row["edits"], ready = apply_edits(site, variant.get("edits") or [], restore)
        if not ready:
            row["outcome"] = "edit_failed"
            return row
        overrides = {**run.state.get("flag_overrides", {}), **(spec.get("base_flags") or {}),
                     **(variant.get("flags") or {})}
        argv = fp.server_argv(config, sys.executable, fp.server_values(run), overrides)
        env = variant_env(os.environ, {**(spec.get("base_env") or {}),
                                       **(variant.get("env") or {})}, tag, values)
        row["argv"] = argv
        fp.append_jsonl(run.out / "engine_commands.jsonl",
                        {"t": round(fp._now() - run.start, 1), "variant": name, "argv": argv})
        row["gpu_before_mib"] = gpu_used()
        with open(log_path, "wb", buffering=0) as handle:
            inner.proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True)
        request_s = float(spec["request"]["timeout_s"])
        # The boot may use the variant's time but the request's floor (request.min_s).
        row["health"] = {}
        boot = wait_healthy(inner, deadline - float(spec["request"]["min_s"]),
                            float(spec["poll_s"]), row["health"])
        row["boot"] = boot
        row["ready_s"] = round(fp._now() - began, 1) if boot == "healthy" else None
        if boot != "healthy":
            row["outcome"] = "exited_at_boot" if boot == "exited" else "boot_timeout"
            return row
        row["pool"] = pool_reading(inner)
        ask = spec["request"]
        timeout = max(min(request_s, deadline - fp._now()), 5.0)
        row["request"] = send_request(inner, ask["messages"], ask["max_tokens"], ask.get("body"),
                                      timeout)
        time.sleep(float(spec["settle_s"]))
        alive = server_alive(inner)
        row["alive_after"] = alive
        made = row["request"].get("completion_tokens") or 0
        if row["request"].get("status") == 200 and made >= 1:
            row["outcome"] = "survived" if alive else "died_after_request"
        else:
            row["outcome"] = "request_failed" if alive else "crashed_in_request"
        rate = spec.get("rate") or {}
        if row["outcome"] == "survived" and rate.get("enabled"):
            left = deadline - fp._now()
            if left < float(rate["min_s"]):
                row["rate"] = {"skipped": f"{round(left)} s left before the deadline"}
            else:
                offset = log_path.stat().st_size
                row["rate"] = send_request(inner, ask["messages"], rate["max_tokens"],
                                           rate.get("body"), min(float(rate["timeout_s"]), left))
                with open(log_path, "rb") as handle:
                    handle.seek(offset)
                    row["rate"]["log"] = log_decode_rate(
                        handle.read().decode("utf-8", errors="replace"))
                if not server_alive(inner):
                    row["outcome"] = "died_after_request"
                    row["alive_after"] = False
        return row
    except OSError as err:
        row.setdefault("outcome", "request_failed")
        row["error"] = fp._describe(err)
        return row
    finally:
        if inner.proc is not None:
            row["exit_code"] = inner.proc.poll()
            row["stop"] = stop_variant(inner, tag, float(spec["stop_settle_s"]),
                                       row.get("gpu_before_mib"))
            inner.proc = None
        row["restored"] = restore_edits(restore)
        row["seconds"] = round(fp._now() - began, 1)
        if log_path.is_file():
            data = log_path.read_bytes()
            text = data.decode("utf-8", errors="replace")
            row["first_error"] = first_error(text, spec["error_patterns"],
                                             spec.get("ignore_patterns") or [],
                                             spec["kernel_frame_skip"])
            row["marks"] = log_marks(text, spec.get("log_marks") or {})
            stem = f"bisect_{index}_{name}"
            lines = text.splitlines()
            (run.out / f"{stem}_tail.log").write_text(
                "\n".join(lines[-int(spec["tail_lines"]):]) + "\n", encoding="utf-8")
            (run.out / f"{stem}_engine.log").write_bytes(bounded(data, int(spec["max_log_bytes"])))
        row["logs"] = collect_logs(run, variant, index, folder)


def collect_logs(run: Any, variant: Mapping[str, Any], index: int, folder: Path) -> list[str]:
    """Copy a bounded copy of each file the variant's ``logs`` globs name into the pulled folder.

    Why: a diagnostic log (the fork's kernel API log at ``SGLANG_KERNEL_API_LOGLEVEL=1``, one
    line per kernel call) grows to hundreds of MB; it is written to the variant's scratch folder,
    which is never mirrored, and only its head and tail come back (the last call before a fault
    is its last line).

    Returns:
        The pulled files' names.
    """
    spec = run.config["bisect"]
    values = {"out": str(run.out), "scratch": str(folder), "variant": variant["name"]}
    names = []
    for pattern in variant.get("logs") or []:
        for path in sorted(Path("/").glob(fill(pattern, values).lstrip("/")))[:4]:
            if not path.is_file():
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)
            name = f"bisect_{index}_{variant['name']}_{safe}.log"
            (run.out / name).write_bytes(bounded(path.read_bytes(), int(spec["max_log_bytes"])))
            names.append(name)
    return names


def write_bisect(run: Any, rows: Sequence[Mapping[str, Any]], running: str | None) -> None:
    """Write ``bisect.json``: the run, the request, the finished rows, the variant running."""
    spec = run.config["bisect"]
    fp.write_json(run.out / BISECT_FILE, {
        "schema": SCHEMA, "run_id": run.config["run_id"], "running": running,
        "request": {"max_tokens": spec["request"]["max_tokens"],
                    "messages": len(spec["request"]["messages"])},
        "rows": list(rows)})


def stage_bisect(inner: Any) -> dict[str, Any]:
    """Run every variant that fits the time left, in order; write ``bisect.json`` as it goes.

    Returns:
        ``variants`` run, the outcome of each, the survivors, and the first named kernel.

    Raises:
        StageSkipped: If ``bisect.enabled`` is off.
    """
    run = inner.run
    spec = run.config["bisect"]
    if not spec.get("enabled"):
        raise fp.StageSkipped("bisect.enabled is off")
    variants = check_variants(spec)
    site = edit_site(run)
    end = run.start + float(run.config["time_plan"][run.state.get("mode", "development")]["end_s"])
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        now = fp._now()
        survived = [r["name"] for r in rows if r["outcome"] == "survived"]
        if variant.get("run_if") == "none_survived" and survived:
            rows.append({"index": index, "name": variant["name"], "why": variant.get("why", ""),
                         "outcome": "not_run", "reason": f"run_if none_survived: {survived}"})
            continue
        deadline = min(now + float(spec["budget_s"]), end - float(spec["kill_reserve_s"]))
        # The boot's own window, not only the time to the end: P5.2-06's V4 started with 163 s
        # to the end but a boot window below zero (the kill reserve and the request's floor).
        window = deadline - float(spec["request"]["min_s"]) - now
        if window < float(spec["min_variant_s"]):
            rows.append({"index": index, "name": variant["name"], "why": variant.get("why", ""),
                         "outcome": "not_run",
                         "reason": f"a boot window of {round(window)} s, under min_variant_s"})
            continue
        write_bisect(run, rows, variant["name"])
        print(f"== variant {index} {variant['name']} (t={round(now - run.start)} s, "
              f"deadline {round(deadline - run.start)} s)", flush=True)
        row = run_variant(inner, variant, index, site, deadline)
        rows.append(row)
        print(f"   {variant['name']}: {row['outcome']} in {row['seconds']} s "
              f"{(row.get('first_error') or {}).get('fatal_line', '')[:160]}", flush=True)
        write_bisect(run, rows, None)
    write_bisect(run, rows, None)
    named = next((r["first_error"] for r in rows if r.get("first_error")), None)
    return {"variants": sum(r["outcome"] != "not_run" for r in rows),
            "outcomes": {r["name"]: r["outcome"] for r in rows},
            "survived": [r["name"] for r in rows if r["outcome"] == "survived"],
            "first_kernel": None if named is None else {
                k: named[k] for k in ("kernel_frame", "triton_kernel", "fatal_line")}}


def summary_markdown(run: Any) -> str:
    """Return ``summary.md``: the stages and one line per variant."""
    lines = [f"# {run.config['run_id']}: engine bisect", "", "| stage | status | seconds |",
             "|---|---|---|"]
    for name, record in run.state.items():
        if isinstance(record, dict) and "status" in record:
            lines.append(f"| {name} | {record['status']} | {record.get('seconds')} |")
    bisect = {}
    if (run.out / BISECT_FILE).is_file():
        bisect = json.loads((run.out / BISECT_FILE).read_text(encoding="utf-8"))
    lines += ["", "| n | variant | outcome | load s | KV tokens | last health | ready s | pool | "
              "tok/s (log, wall) | seconds | kernel frame | first error |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in bisect.get("rows", []):
        err = row.get("first_error") or {}
        fatal = str(err.get("fatal_line", "")).replace("|", "/")[:120]
        rate = row.get("rate") or {}
        speed = f"{(rate.get('log') or {}).get('median_tok_s')}, {rate.get('wall_tok_s')}"
        marks = row.get("marks") or {}
        health = row.get("health") or {}
        lines.append(f"| {row['index']} | {row['name']} | {row['outcome']} | "
                     f"{marks.get('load_s')} | {marks.get('kv_tokens')} | "
                     f"{health.get('last_status')} | "
                     f"{row.get('ready_s')} | {(row.get('pool') or {}).get('pool_tokens')} | "
                     f"{speed} | {row.get('seconds')} | {err.get('kernel_frame')} | {fatal} |")
    return "\n".join(lines) + "\n"


# ---- the processes ------------------------------------------------------------------------------


def inner_main(state_path: Path) -> int:
    """The inner process (the venv's Python): the variants.

    Returns:
        0 when the bisect stage ran (its rows hold the outcomes), else 1.
    """
    out = state_path.parent
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    run = fp.Run(config=meta["config"], root=Path(meta["root"]), out=out, start=meta["start"],
                 state=json.loads(state_path.read_text(encoding="utf-8")))
    inner = fp.Inner(run=run, base=f"http://127.0.0.1:{run.config['engine']['port']}", tok=None,
                     mode=run.state.get("mode", fp.DEVELOPMENT))
    try:
        record = run.stage("bisect", lambda: stage_bisect(inner))
    finally:
        fp.kill_server(inner)
        run.save()
    return 0 if record["status"] == "ok" else 1


def outer_main(config: dict[str, Any], root: Path, commit: str,
               session_start: float | None = None) -> int:
    """The outer process: ``fn_probe.py``'s stages up to the server, then the inner process.

    Args:
        config: The resolved config (``bisect`` among it).
        root: Prefix of absolute paths.
        commit: The code's commit.
        session_start: Wall time of the notebook's first cell; None counts from now.

    Returns:
        The inner process's exit code, or 1 when a stage before it failed.
    """
    out = root / config["paths"]["out"].lstrip("/")
    out.mkdir(parents=True, exist_ok=True)
    for name in (BISECT_FILE, "engine_commands.jsonl"):
        (out / name).unlink(missing_ok=True)
    mode = fp.run_mode(config, os.environ)
    run = fp.Run(config=config, root=root, out=out, start=session_start or fp._now(),
                 state={"mode": mode})
    fp.write_json(out / "run.json", {"schema": fp.SCHEMA, "run_id": config["run_id"],
                                     "entry": "fn_bisect.py", "commit": commit,
                                     "start": run.start, "mode": mode, "root": str(root),
                                     "config": config, "python": sys.version,
                                     "outer_started_s": round(fp._now() - run.start, 1)})
    inputs = run.stage("inputs", lambda: fp.stage_inputs(run))
    run.state["scratch"] = str(run.path(config["paths"]["scratch"]))
    if inputs["status"] == "ok":
        run.stage("local_inputs", lambda: fp.stage_local_inputs(run))
        run.stage("python", lambda: fp.stage_python(run))
        run.stage("install", lambda: fp.stage_install(run))
        run.stage("cuda", lambda: fp.stage_cuda(run))
    venv_python = run.state.get("venv_python")
    code = 1
    if venv_python:
        paths = run.state.get("paths", {})
        script = Path(paths["code"]) / "fn_bisect.py" if paths.get("code") else Path(__file__)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(run.state.get("env_overrides", {}))
        if not config["install"].get("enabled", True):
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        run.save()
        timeout = max(run.left(), 60)
        print(f"== inner process ({round(timeout)} s left)", flush=True)
        try:
            done = subprocess.run([venv_python, str(script), "--inner", str(out / "state.json")],
                                  env=env, timeout=timeout, check=False)
            code = done.returncode
        except subprocess.TimeoutExpired:
            code = 124
        run.state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    run.state["outer"] = {"exit_inner": code, "wall_s": round(fp._now() - run.start, 1)}
    run.save()
    (out / "summary.md").write_text(summary_markdown(run), encoding="utf-8")
    return code


def reread(logs: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read a pulled bisect folder's variant logs again with ``config``'s patterns.

    Why: a pattern fixed after a run (P5.2-06's KV mark) must not need another GPU run; the
    pulled ``bisect_<n>_<name>_engine.log`` copies hold what the marks and the first error read.

    Args:
        logs: The pulled ``run_logs/fn`` folder.
        config: A run config with a ``bisect`` section.

    Returns:
        ``bisect.json`` with each row's ``marks`` and ``first_error`` read again, and
        ``reread`` naming the config.
    """
    spec = config["bisect"]
    data = json.loads((logs / BISECT_FILE).read_text(encoding="utf-8"))
    for row in data.get("rows", []):
        path = logs / f"bisect_{row['index']}_{row['name']}_engine.log"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        row["marks"] = log_marks(text, spec.get("log_marks") or {})
        row["first_error"] = first_error(text, spec["error_patterns"],
                                         spec.get("ignore_patterns") or [],
                                         spec["kernel_frame_skip"])
    data["reread"] = True
    return data


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (``fn_probe.py``'s, which the notebook's cell 2 calls)."""
    parser = fp.build_parser()
    parser.add_argument("--reread", type=Path, default=None,
                        help="a pulled run_logs/fn folder: read its variant logs again with "
                        "--config's patterns into bisect_reread.json and summary_reread.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bisect (outer process), or its inner process with ``--inner``.

    Returns:
        The exit code: 0 when the variants ran, 1 when a stage before them failed.
    """
    args = build_parser().parse_args(argv)
    if args.inner is not None:
        return inner_main(args.inner)
    config = fp.load_config(args.config)
    if args.reread is not None:
        data = reread(args.reread, config)
        fp.write_json(args.reread / "bisect_reread.json", data)
        print(json.dumps({r["name"]: [r["outcome"], r.get("marks")] for r in data["rows"]}))
        return 0
    root = args.root.resolve()
    if args.smoke:
        if root == Path("/"):
            raise SystemExit("--smoke needs --root, a scratch folder for the fake inputs")
        config = fp.apply_smoke(config)
        fp.build_smoke_tree(root, config)
    commit = args.commit or config.get("build", {}).get("commit") or "unknown"
    code = outer_main(config, root, commit, args.session_start)
    print(f"exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
