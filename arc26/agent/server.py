"""Start, watch, and stop the agent's OpenAI-compatible engine, and a fake engine for tests (P6.4).

Why: the scored run talks to a vLLM server on localhost for up to 12 hours. The server must start
from its own offline venv without the image's Python packages, be healthy before the first call,
be restarted once if it dies, and never outlive the run. A failure that cannot be recovered must
not cost the submission: the caller keeps NVARC's attempts and exits cleanly
(:mod:`arc26.agent.run`).

- :class:`EngineServer`: one engine process in a session of its own, its output appended to a
  log file. :meth:`EngineServer.start` waits for ``/health`` with a timeout and stops a process
  that exits or never answers; :meth:`EngineServer.stop` signals the whole process group
  (SIGTERM, then SIGKILL), so the engine's worker processes go with it. A pid file lets the next
  run stop an engine that a killed run left behind.
- :class:`EngineSupervisor`: watches the engine from the event loop (process alive, ``/health``),
  restarts it once after a crash or a hang, and tells the scheduler whether the engine is ready
  or gone for good. It also watches a server given by URL, which it cannot restart.
- :func:`fake_server_main`: a standard-library HTTP server with vLLM's endpoints that replays
  scripted replies per task. Tests and the CPU smoke start it as they would start vLLM
  (``python -m arc26.agent.server fake ...``).

Module-level imports are the standard library only, so the fake starts fast and needs nothing
but the package.
"""

# Engine start, health wait, clean environment, and stop: reused from scripts/agent_spike.py on
# task/P6.3b (commit 74ac92f, reviewed in P6.3). clean_env, free_port, and http_get are copied
# verbatim; server_command is vllm_command here (same flags, from a ServerConfig); the Server class
# (start, alive, wait_healthy, stop) is EngineServer, with a pid file, a stop after a failed boot,
# a kill of the group's leftovers, and a restart added.

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http.client
import json
import logging
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

HEALTH_PATH = "/health"
# Seconds a health probe may take; vLLM answers /health from its API process at once.
PROBE_TIMEOUT_S = 10.0
# Seconds allowed for SIGKILL to take a process group down.
KILL_WAIT_S = 30.0
# Lines of the engine log quoted in an error.
LOG_TAIL_LINES = 20
# The fake engine's exit code when it crashes on purpose.
FAKE_CRASH_CODE = 3
FAKE_FALLBACK = "I need to look at the examples again before I write code."
FAKE_REASONING = "Let me work this out step by step. " * 8
_TASK_LINE = re.compile(r"^Task (\S+?):", re.MULTILINE)


class EngineStartError(RuntimeError):
    """The engine did not become healthy: it could not run, exited during its boot, or timed out."""


class EngineBootTimeout(EngineStartError):
    """The engine ran but did not answer healthy in time; another try would wait as long."""


def clean_env(environ: Mapping[str, str], unset: Sequence[str] = ()) -> dict[str, str]:
    """Return an environment cut off from the image's Python packages and pip settings.

    Why: Kaggle's ``PYTHONPATH`` carries the attached install notebook (torch 2.8 and its stack),
    which would shadow the venv's torch 2.13 in every engine process.

    Args:
        environ: The current environment.
        unset: More names to remove.

    Returns:
        A copy without ``PYTHONPATH``, ``PYTHONHOME``, ``PYTHONSTARTUP``, ``PYTHONUSERBASE``,
        ``VIRTUAL_ENV``, any ``PIP_*``, and ``unset``, with ``PYTHONNOUSERSITE=1``.
    """
    dropped = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"}
    dropped |= set(unset)
    env = {k: v for k, v in environ.items() if k not in dropped and not k.startswith("PIP_")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def free_port() -> int:
    """Return a free local TCP port (for the fake server in tests and the CPU smoke)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_get(base_url: str, path: str, timeout_s: float = 10.0) -> tuple[int, str]:
    """GET a path from a local server.

    Args:
        base_url: For example ``http://127.0.0.1:8000``.
        path: For example ``/health``.
        timeout_s: Socket timeout.

    Returns:
        The status (0 when the connection failed) and the body as text.
    """
    parts = urllib.parse.urlsplit(base_url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException):
        return 0, ""
    finally:
        connection.close()


@dataclass(frozen=True)
class ServerConfig:
    """How to start and watch the vLLM server (the run config's ``server`` section).

    Why: the flags are the spike's stage 1 settings (``configs/P6.3-01.yaml``, checked against the
    vLLM 0.31.0 sources); the run config states each value's reason.

    Attributes:
        host: The address the server binds and the client calls.
        port: Its port.
        served_model_name: The model name requests use.
        tensor_parallel_size: GPUs of the one replica.
        kv_cache_dtype: The KV cache type (``fp8``).
        enable_prefix_caching: Prefix caching on or off.
        max_model_len: The context limit per request; also the attempts' context bound.
        max_num_seqs: The most sequences in flight.
        max_num_batched_tokens: Prompt tokens per scheduler step.
        gpu_memory_utilization: Share of each GPU's memory vLLM may use.
        attention_backend: The attention kernels (``TRITON_ATTN`` on sm_89 with FP8 KV).
        language_model_only: Skip the vision tower.
        reasoning_parser: Splits the thinking from the answer.
        generation_config: Where default sampling comes from (``auto``: the checkpoint's).
        extra_args: Further command-line arguments, appended as given.
        env: Variables added to the engine's clean environment.
        unset_env: Variables removed from it.
        boot_timeout_s: Seconds from the start until ``/health`` must answer 200.
        stop_timeout_s: Seconds between SIGTERM and SIGKILL of the process group.
        health_interval_s: Seconds between health probes during a boot.
        monitor_interval_s: Seconds between the supervisor's checks during the run.
        hang_s: Seconds without a healthy answer, the process alive, before the engine counts as
            hung and is restarted.
        restarts: Restarts allowed after a crash or a hang.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str = "qwen38-27b-int4"
    tensor_parallel_size: int = 4
    kv_cache_dtype: str = "fp8"
    enable_prefix_caching: bool = True
    max_model_len: int = 65536
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.90
    attention_backend: str = "TRITON_ATTN"
    language_model_only: bool = True
    reasoning_parser: str = "qwen3"
    generation_config: str = "auto"
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    unset_env: tuple[str, ...] = ()
    boot_timeout_s: float = 1800.0
    stop_timeout_s: float = 60.0
    health_interval_s: float = 2.0
    monitor_interval_s: float = 5.0
    hang_s: float = 600.0
    restarts: int = 1

    def __post_init__(self) -> None:
        """Reject settings that cannot work.

        Raises:
            ValueError: If the port is out of range, a size or a time is not positive, or
                ``restarts`` is negative.
        """
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be from 1 to 65535; got {self.port}")
        sizes = ("tensor_parallel_size", "max_model_len", "max_num_seqs", "max_num_batched_tokens")
        for name in sizes:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1; got {getattr(self, name)}")
        times = (
            "boot_timeout_s",
            "stop_timeout_s",
            "health_interval_s",
            "monitor_interval_s",
            "hang_s",
        )
        for name in times:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.restarts < 0:
            raise ValueError(f"restarts must be at least 0; got {self.restarts}")

    @property
    def root_url(self) -> str:
        """Return the server's root URL, where ``/health`` lives."""
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        """Return the OpenAI API root the engine client posts to."""
        return f"{self.root_url}/v1"


THINK_END = "</think>"


def reasoning_end_str(extra_args: Sequence[str]) -> str | None:
    """Return the ``reasoning_end_str`` of a ``--reasoning-config`` among vLLM's extra flags.

    Why: the in-request forcing (task P6.9) needs the string vLLM forces when a request's
    thinking budget runs out; the shot's config check and the fake engine read it from here.

    Args:
        extra_args: ``ServerConfig.extra_args``.

    Returns:
        The string, or None when no such flag is there (or its JSON names none).

    Raises:
        ValueError: If the flag appears twice, has no value, or its value is not a JSON object.
    """
    found = [i for i, arg in enumerate(extra_args) if arg == "--reasoning-config"]
    if not found:
        return None
    if len(found) > 1 or found[0] + 1 >= len(extra_args):
        raise ValueError("server.extra_args: one --reasoning-config with one JSON value")
    try:
        value = json.loads(extra_args[found[0] + 1])
    except ValueError as err:
        raise ValueError(f"server.extra_args: --reasoning-config is not JSON: {err}") from err
    if not isinstance(value, dict):
        raise ValueError("server.extra_args: --reasoning-config must be a JSON object")
    end = value.get("reasoning_end_str")
    return end if isinstance(end, str) else None


def vllm_command(
    executable: str | os.PathLike[str], model_dir: str, config: ServerConfig
) -> list[str]:
    """Return the ``vllm serve`` command line for a server config.

    Args:
        executable: The venv's ``vllm`` console script.
        model_dir: The weights folder.
        config: The server settings.

    Returns:
        Program and arguments.
    """
    command = [
        str(executable), "serve", str(model_dir),
        "--host", config.host,
        "--port", str(config.port),
        "--served-model-name", config.served_model_name,
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--kv-cache-dtype", config.kv_cache_dtype,
        "--max-model-len", str(config.max_model_len),
        "--max-num-seqs", str(config.max_num_seqs),
        "--max-num-batched-tokens", str(config.max_num_batched_tokens),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--attention-backend", config.attention_backend,
        "--reasoning-parser", config.reasoning_parser,
        "--generation-config", config.generation_config,
        "--seed", "0",
        "--disable-uvicorn-access-log",
        "--enable-prefix-caching" if config.enable_prefix_caching
        else "--no-enable-prefix-caching",
    ]  # fmt: skip
    if config.language_model_only:
        command.append("--language-model-only")
    return command + list(config.extra_args)


def engine_env(
    environ: Mapping[str, str], config: ServerConfig, bin_dir: str | None = None
) -> dict[str, str]:
    """Return the engine's environment: :func:`clean_env`, the config's variables, the venv first.

    Args:
        environ: The current environment.
        config: The server settings (``env`` and ``unset_env``).
        bin_dir: The venv's ``bin`` folder, put first on ``PATH`` so the engine finds the venv's
            tools (ninja, console scripts) before the image's; None leaves ``PATH`` as it is.

    Returns:
        The environment.
    """
    env = clean_env(environ, config.unset_env)
    env.update({str(k): str(v) for k, v in config.env.items()})
    if bin_dir:
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    return env


def _cmdline(pid: int) -> list[str] | None:
    """A process's argument vector from ``/proc``, or None when it is gone or unreadable."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    parts = raw.decode("utf-8", errors="replace").split("\0")
    return parts[:-1] if parts and parts[-1] == "" else parts


def _argv_matches(found: Sequence[str] | None, argv: Sequence[str]) -> bool:
    """Whether a running process's argv is ``argv`` (a console script gains its interpreter)."""
    if not found or not argv:
        return False
    if list(found) == list(argv):
        return True
    rest = list(argv[1:])
    return bool(rest) and len(found) > len(rest) and list(found[-len(rest) :]) == rest


def _group_alive(pgid: int) -> bool:
    """Whether a process group has a member that is not a zombie.

    Why: a killed member stays a zombie until its parent reaps it, and the parent of an orphaned
    engine is the container's first process, which may never reap; signal 0 still reaches
    zombies, so it alone would wait out every timeout.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        entries = os.listdir("/proc")
    except OSError:
        return True
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
            return True
    return False


def _kill_group(pgid: int, term_wait_s: float) -> None:
    """SIGTERM a process group that is not our child, wait, then SIGKILL what is left."""
    for sig, wait_s in ((signal.SIGTERM, term_wait_s), (signal.SIGKILL, KILL_WAIT_S)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            if not _group_alive(pgid):
                return
            time.sleep(0.1)


class EngineServer:
    """An engine process in a session of its own, logging to a file.

    Thread-safe: the supervisor restarts the engine from a worker thread while the main thread
    may stop it at the end of the run; a lock orders the two, and once :meth:`stop` has run no
    new process starts.

    Args:
        command: Program and arguments (:func:`vllm_command`, or the fake's).
        env: Its environment (:func:`engine_env`).
        log_path: The log file; each start appends to it.
        root_url: The server's root URL (``http://127.0.0.1:8000``), for ``/health``.
        boot_timeout_s: Seconds from a start until ``/health`` must answer 200.
        stop_timeout_s: Seconds between SIGTERM and SIGKILL.
        health_interval_s: Seconds between health probes during a boot.
        pid_path: A file recording the running engine's pid and argv, or None. A start first
            stops a process the file names when its argv still matches (left by a killed run).
        cwd: The engine's working folder; the log's folder when None, so nothing the engine
            writes there lands in the notebook's output.
    """

    def __init__(
        self,
        command: Sequence[str],
        env: Mapping[str, str],
        log_path: str | os.PathLike[str],
        root_url: str,
        *,
        boot_timeout_s: float = 1800.0,
        stop_timeout_s: float = 60.0,
        health_interval_s: float = 2.0,
        pid_path: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> None:
        self.command = [str(c) for c in command]
        self.env = dict(env)
        self.log_path = Path(log_path)
        self.root_url = root_url.rstrip("/")
        self.boot_timeout_s = boot_timeout_s
        self.stop_timeout_s = stop_timeout_s
        self.health_interval_s = health_interval_s
        self.pid_path = None if pid_path is None else Path(pid_path)
        self.cwd = Path(cwd) if cwd is not None else self.log_path.parent
        self.process: subprocess.Popen[bytes] | None = None
        self.started = 0.0
        self.boots = 0
        self._log: IO[bytes] | None = None
        self._lock = threading.Lock()
        self._closed = False

    def alive(self) -> bool:
        """Return whether the engine process is running."""
        process = self.process
        return process is not None and process.poll() is None

    def returncode(self) -> int | None:
        """Return the last process's exit code, or None while it runs or before any start."""
        process = self.process
        return None if process is None else process.poll()

    def healthy(self, timeout_s: float = PROBE_TIMEOUT_S) -> bool:
        """Return whether ``/health`` answers 200.

        Args:
            timeout_s: Socket timeout of the probe.

        Returns:
            True on a 200 answer.
        """
        return http_get(self.root_url, HEALTH_PATH, timeout_s)[0] == 200

    def log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        """Return the last lines of the engine log, for error messages.

        Args:
            lines: How many lines.

        Returns:
            The lines joined, or an empty string when the log cannot be read.
        """
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def start(self, timeout_s: float | None = None) -> float:
        """Start the engine and wait until ``/health`` answers 200.

        Args:
            timeout_s: A tighter limit than ``boot_timeout_s`` (the run's deadline), or None.

        Returns:
            Seconds from the start to the first healthy answer.

        Raises:
            EngineStartError: If the process cannot run, exits during its boot, or is not healthy
                in time; the process group is stopped first, so nothing holds the GPUs or the
                port.
        """
        limit = self.boot_timeout_s if timeout_s is None else min(self.boot_timeout_s, timeout_s)
        self._spawn()
        try:
            return self.wait_healthy(limit)
        except EngineStartError:
            self._terminate()
            raise

    def restart(self, timeout_s: float | None = None) -> float:
        """Stop the engine's process group, then start it again.

        Args:
            timeout_s: As in :meth:`start`.

        Returns:
            Seconds from the new start to the first healthy answer.

        Raises:
            EngineStartError: As in :meth:`start`, or when :meth:`stop` was called before.
        """
        self._terminate()
        return self.start(timeout_s)

    def wait_healthy(self, timeout_s: float) -> float:
        """Wait until ``/health`` answers 200.

        Args:
            timeout_s: Seconds allowed from the start.

        Returns:
            Seconds from the start to the first healthy answer.

        Raises:
            EngineStartError: If the process exits first.
            EngineBootTimeout: If the time runs out first.
        """
        process = self.process
        if process is None:
            raise EngineStartError("the engine was never started")
        while True:
            code = process.poll()
            if code is not None:
                raise EngineStartError(
                    f"the engine exited with code {code} during its boot; log tail:\n"
                    f"{self.log_tail()}"
                )
            if self.healthy(min(PROBE_TIMEOUT_S, max(timeout_s, 0.1))):
                return time.monotonic() - self.started
            elapsed = time.monotonic() - self.started
            if elapsed >= timeout_s:
                raise EngineBootTimeout(f"the engine was not healthy after {timeout_s:.0f} s")
            time.sleep(min(self.health_interval_s, max(0.01, timeout_s - elapsed)))

    def stop(self) -> int | None:
        """Stop the engine for good: no start after this.

        Returns:
            The last process's exit code, or None if none ever started.
        """
        with self._lock:
            self._closed = True
        return self._terminate()

    def stop_stale(self) -> bool:
        """Stop an engine that the pid file names, when its argv still matches.

        Why: a run killed hard (out of memory, a notebook kill) cannot stop its engine, which then
        holds the GPUs and the port; the next run's start must clear it first.

        Returns:
            Whether a process group was signalled.
        """
        path = self.pid_path
        if path is None or not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pid = int(data["pid"])
            argv = [str(a) for a in data["argv"]]
        except (OSError, ValueError, KeyError, TypeError):
            path.unlink(missing_ok=True)
            return False
        stopped = False
        if pid > 1 and _argv_matches(_cmdline(pid), argv):
            logger.warning("stopping the engine a previous run left behind (pid %d)", pid)
            _kill_group(pid, self.stop_timeout_s)
            stopped = True
        path.unlink(missing_ok=True)
        return stopped

    def _spawn(self) -> None:
        with self._lock:
            if self._closed:
                raise EngineStartError("the engine was stopped for good")
            if self.alive():
                raise RuntimeError("the engine is already running")
            self.stop_stale()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.cwd.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "ab")  # closed in _terminate
            log.write(f"=== start {self.boots + 1}: {' '.join(self.command)}\n".encode())
            log.flush()
            self.started = time.monotonic()
            try:
                self.process = subprocess.Popen(
                    self.command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=self.env,
                    start_new_session=True,
                    cwd=self.cwd,
                )
            except OSError as err:
                log.close()
                raise EngineStartError(f"cannot run {self.command[0]}: {err}") from err
            self._log = log
            self.boots += 1
            if self.pid_path is not None:
                self.pid_path.parent.mkdir(parents=True, exist_ok=True)
                record = {"pid": self.process.pid, "argv": self.command}
                self.pid_path.write_text(json.dumps(record), encoding="utf-8")

    def _terminate(self) -> int | None:
        """Stop the current process group (idempotent); a later start stays possible."""
        with self._lock:
            process, log = self.process, self._log
            self._log = None
            if process is not None:
                steps = ((signal.SIGTERM, self.stop_timeout_s), (signal.SIGKILL, KILL_WAIT_S))
                for sig, wait_s in steps:
                    if process.poll() is not None:
                        break
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        break
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(wait_s)
                # The leader may be gone while its children live on (vLLM's engine core and
                # workers); the kernel keeps the group id while members remain, so this reaches
                # only them.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
            if log is not None:
                log.close()
            if self.pid_path is not None:
                self.pid_path.unlink(missing_ok=True)
            return None if process is None else process.poll()


class EngineSupervisor:
    """Watches the engine during the run and restarts it once after a crash or a hang.

    The scheduler awaits :meth:`wait_ready` before each attempt: it returns True while the
    engine is up, waits while it restarts, and returns False once it is gone for good. A crash
    is the process exiting; a hang is ``hang_s`` seconds without a healthy ``/health`` answer
    while it runs. A server given by URL (``server`` None) cannot be restarted: ``hang_s``
    seconds without a healthy answer make it gone for good.

    Args:
        root_url: The server's root URL, for ``/health``.
        server: The managed engine, or None for a server given by URL.
        interval_s: Seconds between checks.
        hang_s: Seconds without a healthy answer before the engine counts as down.
        restarts: Restarts allowed.
        deadline: A ``clock()`` value after which no restart starts, or None.
        clock: The time source (``time.monotonic``).
        probe: Returns whether the engine answers healthy; ``/health`` of ``root_url`` when None.
    """

    def __init__(
        self,
        root_url: str,
        server: EngineServer | None = None,
        *,
        interval_s: float = 5.0,
        hang_s: float = 600.0,
        restarts: int = 1,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        probe: Callable[[], bool] | None = None,
    ) -> None:
        self.root_url = root_url.rstrip("/")
        self.server = server
        self.interval_s = interval_s
        self.hang_s = hang_s
        self.restarts_left = restarts
        self.deadline = deadline
        self.clock = clock
        self._probe = probe if probe is not None else self._http_probe
        self._ready = asyncio.Event()
        self._failed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._last_ok = clock()
        self.reason: str | None = None
        self.events: list[dict[str, Any]] = []

    def _http_probe(self) -> bool:
        return http_get(self.root_url, HEALTH_PATH, PROBE_TIMEOUT_S)[0] == 200

    @property
    def failed(self) -> bool:
        """Return whether the engine is gone for good."""
        return self._failed.is_set()

    @property
    def ready(self) -> bool:
        """Return whether the engine is up now."""
        return self._ready.is_set() and not self._failed.is_set()

    def start(self) -> None:
        """Mark the engine ready and start watching it; call inside the event loop.

        Call it once the engine has answered healthy (after :meth:`EngineServer.start`, or a
        first probe of a server given by URL).
        """
        self._last_ok = self.clock()
        self._ready.set()
        self._task = asyncio.get_running_loop().create_task(self._watch())

    async def wait_ready(self) -> bool:
        """Wait until the engine can take calls.

        Returns:
            True when it is up, False once it is gone for good.
        """
        if self._failed.is_set():
            return False
        if self._ready.is_set():
            return True
        ready = asyncio.ensure_future(self._ready.wait())
        failed = asyncio.ensure_future(self._failed.wait())
        try:
            await asyncio.wait({ready, failed}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in (ready, failed):
                waiter.cancel()
        return not self._failed.is_set()

    async def close(self) -> None:
        """Stop watching (the engine itself is stopped by its owner)."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _fail(self, reason: str) -> None:
        self.reason = reason
        self.events.append({"event": "failed", "reason": reason, "t": self.clock()})
        logger.error("engine gone for good: %s", reason)
        self._ready.clear()
        self._failed.set()

    async def _problem(self) -> str | None:
        """The reason the engine counts as down now, or None while it is up."""
        server = self.server
        if server is not None and server.process is not None and server.returncode() is not None:
            return f"the engine exited with code {server.returncode()}"
        healthy = await asyncio.to_thread(self._probe)
        now = self.clock()
        if healthy:
            self._last_ok = now
            return None
        if now - self._last_ok >= self.hang_s:
            return f"no healthy answer for {now - self._last_ok:.0f} s"
        return None

    async def _watch(self) -> None:
        while not self._failed.is_set():
            await asyncio.sleep(self.interval_s)
            try:
                problem = await self._problem()
            except Exception:  # a broken check must not end the watch
                logger.exception("the engine check failed; checking again")
                continue
            if problem is None:
                continue
            self._ready.clear()
            self.events.append({"event": "down", "reason": problem, "t": self.clock()})
            logger.error("engine down: %s", problem)
            if self.server is None:
                self._fail(f"{problem} (a server given by URL is not restarted)")
                return
            if self.restarts_left < 1:
                self._fail(f"{problem}; no restart left")
                return
            left = None if self.deadline is None else self.deadline - self.clock()
            if left is not None and left <= 0:
                self._fail(f"{problem}; the deadline has passed")
                return
            self.restarts_left -= 1
            try:
                seconds = await asyncio.to_thread(self.server.restart, left)
            except Exception as err:
                self._fail(f"{problem}; the restart failed: {err}")
                return
            self._last_ok = self.clock()
            self.events.append({"event": "restarted", "seconds": seconds, "t": self._last_ok})
            logger.warning("engine restarted in %.1f s", seconds)
            self._ready.set()


# --- the fake engine ----------------------------------------------------------------------------


def _fake_tokens(text: str, chars_per_token: float = 4.0) -> int:
    return math.ceil(len(text) / chars_per_token)


def fake_completion(
    replies: Mapping[str, Sequence[Any]], body: Mapping[str, Any], budget_note: str = ""
) -> tuple[dict[str, Any], float]:
    """Build the fake engine's chat completion for one request.

    The task is the one the first user message names (``Task <id>: ...``, as the agent's task
    message starts); the turn is the number of assistant messages sent. A reply is a string (the
    visible text) or ``{"text", "reasoning", "delay_s"}``; past the script, or for an unknown
    task, the reply has no code. Tokens are one per 4 characters; a reply over ``max_tokens`` is
    cut, reasoning first, and finishes with ``length``. A request's ``thinking_token_budget``
    cuts the reasoning there and appends ``budget_note``, the text then following, as vLLM
    0.31 forces its ``reasoning_end_str`` (task P6.9).

    Args:
        replies: Task id to its replies, one per turn.
        body: The request body.
        budget_note: What a ``--reasoning-config`` would force before ``</think>``.

    Returns:
        The response object (vLLM 0.31's: the reasoning in ``message.reasoning``) and the seconds
        to wait before sending it.
    """
    messages = body.get("messages") or []
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    match = _TASK_LINE.search(users[0]) if users else None
    script = replies.get(match.group(1), ()) if match else ()
    turn = sum(1 for m in messages if m.get("role") == "assistant")
    reply = script[turn] if turn < len(script) else FAKE_FALLBACK
    if isinstance(reply, str):
        reply = {"text": reply}
    text = str(reply.get("text") or "")
    reasoning = str(reply.get("reasoning") or "")
    delay = float(reply.get("delay_s") or 0.0)
    budget = body.get("thinking_token_budget")
    if isinstance(budget, int) and not reasoning:
        # A thinking model reasons before it answers: an unscripted budgeted request (the budget
        # probe) gets some reasoning, which the budget then cuts.
        reasoning = FAKE_REASONING
    if isinstance(budget, int) and _fake_tokens(reasoning) > budget:
        reasoning = reasoning[: budget * 4] + budget_note
    limit = int(body.get("max_tokens") or 1)
    used = _fake_tokens(reasoning) + _fake_tokens(text)
    finish = "stop"
    if used > limit:
        room = limit * 4
        text = text[: max(0, room - len(reasoning))]
        reasoning = reasoning[:room]
        used, finish = limit, "length"
    prompt = sum(_fake_tokens(str(m.get("content") or "")) + 4 for m in messages)
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning"] = reasoning
    response = {
        "id": f"fake-{turn}",
        "object": "chat.completion",
        "model": body.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": used},
    }
    return response, delay


def fake_server_main(
    port: int,
    replies_path: str | None = None,
    *,
    host: str = "127.0.0.1",
    model: str = "fake",
    boot_delay_s: float = 0.0,
    delay_s: float = 0.0,
    crash_after: int | None = None,
    crash_marker: str | None = None,
    fail_boot: bool = False,
    budget_note: str = "",
) -> int:
    """Serve the fake engine until SIGTERM: ``/health``, ``/v1/models``, ``/v1/chat/completions``.

    Args:
        port: The port.
        replies_path: JSON of task id to replies (:func:`fake_completion`), or None.
        host: The address to bind.
        model: The served model name.
        boot_delay_s: Seconds before the port opens, as a booting engine.
        delay_s: Seconds every completion waits before it is sent.
        crash_after: Exit abruptly (code 3, connections dropped) on this completion request,
            as a crashing engine; None never crashes.
        crash_marker: With ``crash_after``, crash only while this file does not exist, and create
            it first: the restarted fake then serves on.
        fail_boot: Exit with code 1 at once, as an engine that cannot load.
        budget_note: As in :func:`fake_completion`.

    Returns:
        The exit code: 0 after SIGTERM, 1 for ``fail_boot``.
    """
    if fail_boot:
        print("fake engine: failing its boot on purpose", flush=True)
        return 1
    replies: dict[str, Any] = {}
    if replies_path:
        replies = json.loads(Path(replies_path).read_text(encoding="utf-8"))
    time.sleep(boot_delay_s)
    counter = {"completions": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
            print("fake engine: " + format % args, flush=True)

        def _send(self, status: int, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802, D102
            if self.path == HEALTH_PATH:
                self._send(200, "")
            elif self.path == "/v1/models":
                self._send(200, json.dumps({"object": "list", "data": [{"id": model}]}))
            else:
                self._send(404, "{}")

        def do_POST(self) -> None:  # noqa: N802, D102
            if self.path != "/v1/chat/completions":
                self._send(404, "{}")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError:
                self._send(400, json.dumps({"error": "not JSON"}))
                return
            with lock:
                counter["completions"] += 1
                count = counter["completions"]
            if crash_after is not None and count >= crash_after:
                marker = None if crash_marker is None else Path(crash_marker)
                if marker is None or not marker.exists():
                    if marker is not None:
                        marker.write_text("crashed\n", encoding="utf-8")
                    print(f"fake engine: crashing on request {count}", flush=True)
                    os._exit(FAKE_CRASH_CODE)
            response, wait_s = fake_completion(replies, body, budget_note)
            time.sleep(delay_s + wait_s)
            self._send(200, json.dumps(response))

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(f"fake engine on {host}:{port}", flush=True)
    server.serve_forever()
    server.server_close()
    return 0


def fake_command(
    port: int,
    replies_path: str | os.PathLike[str] | None = None,
    *,
    python: str = sys.executable,
    model: str = "fake",
    boot_delay_s: float = 0.0,
    delay_s: float = 0.0,
    crash_after: int | None = None,
    crash_marker: str | os.PathLike[str] | None = None,
    fail_boot: bool = False,
    budget_note: str | None = None,
) -> list[str]:
    """Return the command line that starts the fake engine.

    Args:
        port: Its port.
        replies_path: Its replies file, or None.
        python: The interpreter (one that can import ``arc26``).
        model: The served model name.
        boot_delay_s: As in :func:`fake_server_main`.
        delay_s: As in :func:`fake_server_main`.
        crash_after: As in :func:`fake_server_main`.
        crash_marker: As in :func:`fake_server_main`.
        fail_boot: As in :func:`fake_server_main`.
        budget_note: As in :func:`fake_server_main`; None passes none.

    Returns:
        Program and arguments.
    """
    command = [python, "-m", "arc26.agent.server", "fake", "--port", str(port), "--model", model]
    if replies_path is not None:
        command += ["--replies", str(replies_path)]
    if boot_delay_s:
        command += ["--boot-delay-s", str(boot_delay_s)]
    if delay_s:
        command += ["--delay-s", str(delay_s)]
    if crash_after is not None:
        command += ["--crash-after", str(crash_after)]
    if crash_marker is not None:
        command += ["--crash-marker", str(crash_marker)]
    if fail_boot:
        command.append("--fail-boot")
    if budget_note:
        command += ["--budget-note", budget_note]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fake engine (``python -m arc26.agent.server fake --port N ...``).

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m arc26.agent.server", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fake = commands.add_parser("fake", help="serve the fake OpenAI-compatible engine")
    fake.add_argument("--port", type=int, required=True)
    fake.add_argument("--host", default="127.0.0.1")
    fake.add_argument("--replies", help="JSON: task id to its replies, one per turn")
    fake.add_argument("--model", default="fake")
    fake.add_argument("--boot-delay-s", type=float, default=0.0)
    fake.add_argument("--delay-s", type=float, default=0.0)
    fake.add_argument("--crash-after", type=int)
    fake.add_argument("--crash-marker")
    fake.add_argument("--fail-boot", action="store_true")
    fake.add_argument("--budget-note", default="")
    args = parser.parse_args(argv)
    return fake_server_main(
        args.port,
        args.replies,
        host=args.host,
        model=args.model,
        boot_delay_s=args.boot_delay_s,
        delay_s=args.delay_s,
        crash_after=args.crash_after,
        crash_marker=args.crash_marker,
        fail_boot=args.fail_boot,
        budget_note=args.budget_note,
    )


if __name__ == "__main__":
    sys.exit(main())
