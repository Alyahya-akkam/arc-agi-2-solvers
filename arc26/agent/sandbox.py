"""A persistent per-task Python REPL in a resource-limited subprocess (P6.1, ADR-020).

Why: the agent explores a task by running Python cells on its grids. Cells are model-written code,
so they never run in the solver's process: each REPL is a child process running this module (see
:func:`spawn_command`: no ``site`` module, an import path of the standard library followed by the
folders the parent loaded arc26, numpy, and scipy from) that starts with the task's grids, numpy,
and the helper library loaded, and has these layers between the code and the machine:

1. The AST policy of :mod:`arc26.agent.guards`, checked in the parent before a cell is sent.
2. Restricted builtins in the child: no ``open``, ``eval``, ``exec``, ``compile``, ``input``, ...;
   an ``__import__`` that honors the import allowlist; ``getattr`` and its relatives refuse
   dunder and forbidden names built at run time.
3. An audit hook (PEP 578) that denies process creation, signals, sockets, ctypes, file writes,
   filesystem changes, and reads outside the Python installation.
4. Resource limits: address space (default 2 GiB), no file data written (``RLIMIT_FSIZE`` 0), no
   core files, no new processes (``RLIMIT_NPROC`` 0, which binds non-root users); a wall timeout
   per cell that kills the child's process group and starts a fresh child with the grids loaded.
5. Output capped (default 8,000 characters; head and tail kept, the cut marked). Replies are
   size-capped JSON lines on private pipes, validated by the parent; never pickles.

:mod:`arc26.agent.verify` reuses the child-side pieces defined here.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import io
import json
import linecache
import os
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

from arc26.agent import guards
from arc26.agent.types import AgentTask, CellResult

GIB = 1024**3
DEFAULT_MEMORY_BYTES = 2 * GIB
DEFAULT_MAX_OUTPUT_CHARS = 8_000
# Deep enough for a recursive flood fill over a 30 by 30 grid (900 frames plus callers).
RECURSION_LIMIT = 4_000
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
ERROR_CHARS = 500
# Imported by every sandbox process at startup when installed, before the audit hook and the
# process limit, so model code that imports them behaves the same in the REPL and the verifier
# and starts fast. scipy is optional: absent on the dev machine and in CI, present on Kaggle.
DEFAULT_PRELOAD: tuple[str, ...] = ("scipy.ndimage",)

# PEP 578 events the sandboxed code may never cause (prefixes); file reads are checked by path.
DENIED_EVENTS: tuple[str, ...] = (
    # Processes and signals.
    "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.kill",
    "subprocess.", "pty.", "signal.pthread_kill",
    # Network, native code, and memory maps.
    "socket.", "ctypes.", "mmap.", "urllib.", "http.client.", "ftplib.", "smtplib.",
    "imaplib.", "poplib.", "nntplib.", "telnetlib.", "webbrowser.", "sqlite3.", "syslog.",
    # Filesystem changes and the environment.
    "shutil.", "tempfile.", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod",
    "os.chown", "os.chflags", "os.lchflags", "os.lchmod", "os.lchown", "os.link", "os.symlink",
    "os.truncate", "os.utime", "os.chdir", "os.chroot", "os.putenv", "os.unsetenv",
    "os.setxattr", "os.removexattr", "os.mkfifo", "os.mknod",
    # Limits, tracing, heap walks, code construction, unpickling, stdin.
    "resource.setrlimit", "resource.prlimit", "sys.settrace", "sys.setprofile",
    "gc.get_objects", "gc.get_referrers", "gc.get_referents", "function.__new__",
    "code.__new__", "pickle.find_class", "builtins.input",
)  # fmt: skip
_PATH_EVENTS = frozenset({"open", "os.listdir", "os.scandir", "glob.glob"})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_DROPPED_BUILTINS = frozenset({"copyright", "credits", "license", "__loader__", "__spec__"})
_MISSING = object()
# Top-level packages every sandbox process imports: arc26 (the REPL, the verifier, the helper
# library), numpy, and scipy where installed (DEFAULT_PRELOAD; the import allowlist names it).
SANDBOX_PACKAGES: tuple[str, ...] = ("arc26", "numpy", "scipy")
# Run with ``python -c`` as every sandbox process's first code: the import path becomes the
# standard library followed by the package folders given after the module name (the working
# directory's "" entry dropped), then the module runs as ``__main__``, as ``python -m`` runs it.
_BOOTSTRAP = (
    "import sys\n"
    "sys.path[:] = [p for p in sys.path if p] + sys.argv[2:]\n"
    "import runpy\n"
    "runpy.run_module(sys.argv[1], run_name='__main__', alter_sys=True)\n"
)


class SandboxError(RuntimeError):
    """A sandbox process failed to start, died unexpectedly, or broke the message protocol."""


# The JSON-lines protocol, used by both sides.


def encode_message(message: Mapping[str, Any]) -> bytes:
    """Encode one protocol message as a JSON line.

    Args:
        message: A JSON-serializable mapping.

    Returns:
        ASCII JSON followed by a newline (strings escape their own newlines).
    """
    return json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def write_message(fd: int, message: Mapping[str, Any]) -> None:
    """Write one protocol message to a file descriptor, completely.

    Args:
        fd: A pipe's write end.
        message: The message.

    Raises:
        OSError: If the pipe is closed (``BrokenPipeError``) or the write fails.
    """
    data = memoryview(encode_message(message))
    while data:
        written = os.write(fd, data)
        data = data[written:]


class LineReader:
    """Reads JSON-line messages from a pipe with deadlines and a size cap.

    Why: the other end may be model-controlled (a sandboxed process), so a reply may be late,
    endless, oversized, or not JSON; none of that may block or exhaust the reader.

    Args:
        fd: The pipe's read end.
        max_bytes: The longest accepted line.
    """

    def __init__(self, fd: int, max_bytes: int = MAX_MESSAGE_BYTES) -> None:
        self._fd = fd
        self._max = max_bytes
        self._buffer = bytearray()
        self._poll = select.poll()
        self._poll.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)

    def read(self, deadline: float | None) -> dict[str, Any] | None:
        """Return the next message, waiting at most until a deadline.

        Args:
            deadline: A ``time.monotonic()`` value, or None to wait indefinitely.

        Returns:
            The decoded message, or None at end of file.

        Raises:
            TimeoutError: If no complete line arrived by the deadline.
            SandboxError: If a line is too long, is not JSON, or is not a JSON object.
        """
        while True:
            end = self._buffer.find(b"\n")
            if end >= 0:
                line = bytes(self._buffer[:end])
                del self._buffer[: end + 1]
                return _decode(line)
            if len(self._buffer) > self._max:
                raise SandboxError(f"message longer than {self._max} bytes")
            if deadline is None:
                ready = self._poll.poll()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                ready = self._poll.poll(max(1, int(remaining * 1000)))
            if not ready:
                continue
            chunk = os.read(self._fd, 65536)
            if not chunk:
                return None
            self._buffer += chunk


def _decode(line: bytes) -> dict[str, Any]:
    try:
        message = json.loads(line)
    except (ValueError, RecursionError) as exc:
        raise SandboxError(f"malformed message: {type(exc).__name__}") from None
    if not isinstance(message, dict):
        raise SandboxError("message is not a JSON object")
    return message


# Child side: limits, audit hook, builtins, output capture.


def child_env(workdir: str) -> dict[str, str]:
    """Return the minimal environment of a sandbox process.

    Why: nothing of the parent's environment (tokens, paths) reaches model code, except
    ``LD_LIBRARY_PATH``, which some installations need to load numpy's libraries; one BLAS thread
    keeps each process small under the address-space limit and the CPUs shared fairly. There is
    no ``PYTHONPATH``: its folders would come before the standard library (see
    :func:`spawn_command`).

    Args:
        workdir: The process's empty working and home directory.

    Returns:
        The environment.
    """
    env = {
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return env


def import_roots(packages: Sequence[str] = SANDBOX_PACKAGES) -> list[str]:
    """Return the import-path folders this interpreter loads some packages from, in path order.

    Why: a sandbox process must import arc26, numpy, and scipy from the folders the parent
    imports them from, and take nothing else from the parent's path. Each folder is the one that
    holds the package; ordered as the parent's ``sys.path`` orders them, a package present in two
    of them resolves as it does in the parent. Found with :func:`importlib.util.find_spec`, so a
    package the parent has not imported (scipy) is not imported here.

    Args:
        packages: Top-level package names; one that is not installed is skipped.

    Returns:
        Real paths of the folders, each once, in ``sys.path`` order; a folder only a meta-path
        finder supplies (absent from ``sys.path``) comes last.
    """
    roots: list[str] = []
    for name in packages:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        locations = list(spec.submodule_search_locations or ())
        if locations:
            root = os.path.dirname(locations[0])
        elif spec.has_location and spec.origin:
            root = os.path.dirname(spec.origin)
        else:
            continue
        root = os.path.realpath(root)
        if root not in roots:
            roots.append(root)
    order: dict[str, int] = {}
    for index, entry in enumerate(sys.path):
        order.setdefault(os.path.realpath(entry or os.curdir), index)
    # sorted() is stable: folders absent from sys.path keep the packages' order, after the rest.
    return sorted(roots, key=lambda root: order.get(root, len(sys.path)))


def spawn_command(module: str) -> list[str]:
    """Return the command line that starts a sandbox child running a module.

    Why: the child imports only the standard library and the packages the parent passes, and the
    standard library first. Run ``P6.3-02`` (12 Oct) failed every attempt because a child's
    ``PYTHONPATH`` was the parent's site folder: it came before the standard library, and Kaggle's
    image holds the obsolete ``pathlib`` 1.0.1 backport there, which fails to import on Python
    3.10 and later. So the child runs without the ``site`` module (``-S``: no site folders, no
    ``.pth`` files, no ``sitecustomize``, which on Kaggle imports the image's logging and Google
    modules), a bootstrap sets its path to the standard library followed by
    :func:`import_roots`, and ``-B`` writes no bytecode.

    Args:
        module: The module whose ``__main__`` block is the child, for example
            ``"arc26.agent.sandbox"``.

    Returns:
        The argument list.
    """
    return [sys.executable, "-B", "-S", "-c", _BOOTSTRAP, module, *import_roots()]


def take_protocol_fds() -> tuple[int, int]:
    """Move the protocol off standard input and output (child side).

    Why: model code that prints, or writes to file descriptor 1, must not corrupt the replies.

    Returns:
        ``(command_fd, reply_fd)``: private duplicates of the parent's pipes; standard input and
        output then point at ``/dev/null``.
    """
    command_fd, reply_fd = os.dup(0), os.dup(1)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.close(devnull)
    return command_fd, reply_fd


def silence_stderr() -> None:
    """Point standard error at ``/dev/null`` once startup is over (child side)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)


def apply_limits(memory_bytes: int) -> None:
    """Set the resource limits every sandbox process has (child side).

    Args:
        memory_bytes: The address-space limit (``RLIMIT_AS``), soft and hard.
    """
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def forbid_processes() -> None:
    """Forbid new processes and threads (child side, after startup).

    Why: ``RLIMIT_NPROC`` 0 makes every ``fork`` and thread start fail for a non-root user; the
    audit hook denies ``os.fork`` and its relatives for root as well.
    """
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def preload_modules(names: Sequence[str]) -> list[str]:
    """Import the installed modules among some names (child side, before the hook and limits).

    Args:
        names: Dotted module names; a missing or failing one is skipped, so the program's own
            import of it fails later with a clear error.

    Returns:
        The names that were imported.
    """
    loaded = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            continue
        loaded.append(name)
    return loaded


def read_roots() -> list[str]:
    """Return the directories sandboxed code may read: the Python installation and import path.

    Returns:
        Real paths of ``sys.prefix``, ``sys.base_prefix``, the exec prefixes, and the existing
        directories on ``sys.path`` (lazy imports must still find their modules).
    """
    candidates = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, *sys.path]
    return sorted({os.path.realpath(p) for p in candidates if p and os.path.isdir(p)})


def install_audit_hook(roots: Sequence[str]) -> None:
    """Install the audit hook that denies dangerous operations (child side, irreversible).

    Why: it binds even code that slips past the static policy, for example through a library
    function that evaluates a string; PEP 578 hooks cannot be removed.

    Args:
        roots: Directories that may be read (see :func:`read_roots`); everything else may not.
    """
    allowed = tuple(os.path.realpath(r) for r in roots)

    def readable(args: tuple[Any, ...], event: str) -> bool:
        path = args[0] if args else None
        if path is None:
            return event != "open"
        if isinstance(path, int):
            return False
        if event == "open":
            mode, flags = (args + (None, None))[1:3]
            if isinstance(mode, str) and any(ch in mode for ch in "wax+"):
                return False
            if isinstance(flags, int) and flags & _WRITE_FLAGS:
                return False
        try:
            real = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        return any(real == root or real.startswith(root + os.sep) for root in allowed)

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event in _PATH_EVENTS:
            if not readable(args, event):
                raise PermissionError(f"sandbox: {event} {args[0]!r} is not allowed")
        elif event.startswith(DENIED_EVENTS):
            raise PermissionError(f"sandbox: {event} is not allowed")

    sys.addaudithook(hook)


def _guarded_import(
    name: str,
    globals: Mapping[str, Any] | None = None,
    locals: Mapping[str, Any] | None = None,
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> Any:
    if level != 0:
        raise ImportError("sandbox: relative imports are not allowed")
    items = tuple(fromlist or ())
    if not guards.is_allowed_module(name):
        # `from arc26.agent import lib` asks for arc26.agent with lib in the fromlist.
        if not items or not all(guards.is_allowed_module(f"{name}.{i}") for i in items):
            raise ImportError(f"sandbox: import of {name!r} is not allowed")
    elif any(i == "*" or guards.is_forbidden_attribute(i) or guards.is_dunder(i) for i in items):
        raise ImportError(f"sandbox: from {name} import {', '.join(items)} is not allowed")
    return builtins.__import__(name, globals, locals, items, level)


def _checked_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("attribute name must be a string")
    if guards.is_forbidden_attribute(name):
        raise AttributeError(f"sandbox: attribute {name!r} is not allowed")
    return name


def _safe_getattr(obj: object, name: str, default: object = _MISSING) -> Any:
    if default is _MISSING:
        return getattr(obj, _checked_name(name))
    return getattr(obj, _checked_name(name), default)


def _safe_hasattr(obj: object, name: str) -> bool:
    return hasattr(obj, _checked_name(name))


def _safe_setattr(obj: object, name: str, value: object) -> None:
    setattr(obj, _checked_name(name), value)


def _safe_delattr(obj: object, name: str) -> None:
    delattr(obj, _checked_name(name))


def restricted_builtins() -> dict[str, Any]:
    """Return the builtins of sandboxed code (child side).

    Returns:
        Python's builtins without the forbidden ones (:data:`arc26.agent.guards.FORBIDDEN_BUILTINS`
        and the site helpers), with ``__import__``, ``getattr``, ``hasattr``, ``setattr``, and
        ``delattr`` replaced by versions that apply the import allowlist and the attribute rule.
    """
    safe = {
        name: value
        for name, value in vars(builtins).items()
        if name not in guards.FORBIDDEN_BUILTINS and name not in _DROPPED_BUILTINS
    }
    safe.update(
        __import__=_guarded_import,
        getattr=_safe_getattr,
        hasattr=_safe_hasattr,
        setattr=_safe_setattr,
        delattr=_safe_delattr,
    )
    return safe


class CappedWriter(io.TextIOBase):
    """A text sink that keeps the head and tail of what is written, up to a limit.

    Why: a cell may print without end; memory stays bounded and the model still sees the start
    of its output and the end, where the traceback is.

    Args:
        limit: Characters kept: the first three quarters as head, the last quarter as tail.
    """

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._head_limit = limit - limit // 4
        self._tail_limit = limit // 4
        self._head: list[str] = []
        self._head_len = 0
        self._tail: deque[str] = deque()
        self._tail_len = 0
        self._total = 0

    def writable(self) -> bool:
        """Return True: this stream accepts writes."""
        return True

    def write(self, text: str) -> int:
        """Record text, keeping only the head and tail.

        Args:
            text: The text.

        Returns:
            The number of characters accepted (all of them).

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        size = len(text)
        self._total += size
        room = self._head_limit - self._head_len
        if room > 0:
            piece = text[:room]
            self._head.append(piece)
            self._head_len += len(piece)
            text = text[room:]
        if text and self._tail_limit:
            piece = text[-self._tail_limit :]
            self._tail.append(piece)
            self._tail_len += len(piece)
            while self._tail_len - len(self._tail[0]) >= self._tail_limit:
                self._tail_len -= len(self._tail.popleft())
        return size

    def getvalue(self) -> str:
        """Return the kept text, with a marker where text was dropped.

        Returns:
            The head, the marker if any characters were dropped, and the tail.
        """
        head = "".join(self._head)
        tail = "".join(self._tail)[-self._tail_limit :] if self._tail_limit else ""
        dropped = self._total - len(head) - len(tail)
        if dropped <= 0:
            return head + tail
        return f"{head}\n[... {dropped} characters truncated ...]\n{tail}"


def clean_text(text: str) -> str:
    """Make text from a sandboxed process safe to encode as UTF-8.

    Why: model code can print lone surrogates; JSON carries them, but they would break any later
    UTF-8 encoding (the engine request, a log file).

    Args:
        text: The text.

    Returns:
        The text with unencodable characters replaced.
    """
    return text.encode("utf-8", "replace").decode("utf-8")


def error_line(exc: BaseException) -> str:
    """Summarize an exception as its traceback's last line.

    Args:
        exc: The exception.

    Returns:
        ``"memory"`` for a ``MemoryError``, else for example ``"ZeroDivisionError: division by
        zero"``, at most :data:`ERROR_CHARS` characters.
    """
    if isinstance(exc, MemoryError):
        return "memory"
    try:
        lines = "".join(traceback.format_exception_only(type(exc), exc)).strip().splitlines()
        text = lines[-1] if lines else type(exc).__name__
    except Exception:
        # str(exc) runs the exception's own code, which may fail.
        text = type(exc).__name__
    return text[:ERROR_CHARS]


def _user_traceback(exc: BaseException) -> str:
    """Format a cell's traceback without the sandbox's own frames."""
    try:
        report = traceback.TracebackException(type(exc), exc, exc.__traceback__)
        for part in _chain(report):
            part.stack = traceback.StackSummary.from_list(
                [frame for frame in part.stack if frame.filename != __file__]
            )
        return "".join(report.format())
    except Exception:
        return f"{error_line(exc)}\n"


def _chain(report: traceback.TracebackException) -> list[traceback.TracebackException]:
    parts, seen = [], set()
    pending = [report]
    while pending:
        part = pending.pop()
        if id(part) in seen:
            continue
        seen.add(id(part))
        parts.append(part)
        pending.extend(p for p in (part.__cause__, part.__context__) if p is not None)
    return parts


# Parent side: a child process speaking the protocol.


def _signal_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _reap(proc: subprocess.Popen[bytes]) -> None:
    _signal_group(proc.pid)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass


class ChildProcess:
    """A sandbox child process, its pipes, and its process group (parent side).

    The child runs in a new session, so its process group can be killed as one; a finalizer kills
    it when this object is collected or the interpreter exits.

    Args:
        module: The module the child runs (see :func:`spawn_command`).
        workdir: The child's working directory.
    """

    def __init__(self, module: str, workdir: str) -> None:
        try:
            self.proc = subprocess.Popen(
                spawn_command(module),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                env=child_env(workdir),
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            raise SandboxError(f"cannot start a sandbox process: {exc}") from None
        if self.proc.stdin is None or self.proc.stdout is None:
            raise SandboxError("child pipes missing")
        self._command_fd = self.proc.stdin.fileno()
        self._reader = LineReader(self.proc.stdout.fileno())
        self._finalizer = weakref.finalize(self, _reap, self.proc)

    @property
    def pid(self) -> int:
        """Return the child's process id (also its process group id)."""
        return self.proc.pid

    @property
    def alive(self) -> bool:
        """Return whether the child is still running."""
        return self.proc.poll() is None

    def send(self, message: Mapping[str, Any]) -> None:
        """Send one message.

        Args:
            message: The message.

        Raises:
            SandboxError: If the child is gone.
        """
        try:
            write_message(self._command_fd, message)
        except OSError as exc:
            raise SandboxError(f"child {self.pid} is gone: {exc}") from None

    def receive(self, deadline: float | None) -> dict[str, Any]:
        """Receive one message.

        Args:
            deadline: A ``time.monotonic()`` deadline, or None.

        Returns:
            The message.

        Raises:
            TimeoutError: If none arrived in time.
            SandboxError: If the child closed its pipe or broke the protocol.
        """
        message = self._reader.read(deadline)
        if message is None:
            raise SandboxError(f"child {self.pid} exited ({self.exit_description()})")
        return message

    def exit_description(self) -> str:
        """Describe how the child ended, waiting briefly for it.

        Returns:
            For example ``"signal SIGSEGV"`` or ``"exit code 1"``, or ``"running"``.
        """
        try:
            code = self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return "running"
        if code < 0:
            try:
                return f"signal {signal.Signals(-code).name}"
            except ValueError:
                return f"signal {-code}"
        return f"exit code {code}"

    def startup_errors(self) -> str:
        """Return what the child wrote to standard error, once it has exited.

        Returns:
            The text (empty while the child runs, to avoid blocking).
        """
        if self.alive or self.proc.stderr is None or self.proc.stderr.closed:
            return ""
        return self.proc.stderr.read().decode("utf-8", "replace")[-4000:]

    def close_stderr(self) -> None:
        """Close the parent's end of standard error once the child has started."""
        if self.proc.stderr is not None and not self.proc.stderr.closed:
            self.proc.stderr.close()

    def terminate(self) -> None:
        """Kill the child's process group but keep the pipes open.

        Why: another thread may be reading the child's pipe; it sees end of file and calls
        :meth:`kill` itself, so no descriptor is closed under a reader (and reused).
        """
        _signal_group(self.proc.pid)

    def kill(self) -> None:
        """Kill the child's process group, reap it, and release its pipes."""
        self._finalizer()


# The REPL.


class SubprocessRepl:
    """A persistent Python REPL for one task in a sandboxed subprocess; implements ``Repl``.

    The namespace starts with ``train`` (a list of ``(input, output)`` numpy integer arrays),
    ``test_inputs`` (a list of arrays), ``np``, and ``lib`` (:mod:`arc26.agent.lib`). When a
    cell's last statement is an expression, its value's ``repr`` is printed, as in an interactive
    session.

    One instance per agent attempt, never shared between threads, so it needs no locking; an
    internal lock only serializes accidental concurrent calls.

    Args:
        task: The task whose grids are loaded.
        memory_bytes: The child's address-space limit.
        max_output_chars: The output kept per cell.
        policy: Apply the AST policy and the restricted builtins (the Python-level layer). False
            only in tests of the process-level containment (resource limits, audit hook), which
            is always on.
        preload: Modules the child imports at startup when installed (see
            :data:`DEFAULT_PRELOAD`); the verifier's workers load the same ones.
        startup_timeout_s: How long a fresh child may take to load numpy and the grids.
        cpu_seconds: CPU seconds a child may use over its life (``RLIMIT_CPU``; ``SIGXCPU`` then
            ends it and the next cell starts a fresh child), or None for no cap. A backstop
            behind the per-cell wall timeout.

    Raises:
        SandboxError: If the child cannot be started.
    """

    def __init__(
        self,
        task: AgentTask,
        *,
        memory_bytes: int = DEFAULT_MEMORY_BYTES,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        policy: bool = True,
        preload: Sequence[str] = DEFAULT_PRELOAD,
        startup_timeout_s: float = 60.0,
        cpu_seconds: int | None = None,
    ) -> None:
        self._init = {
            "op": "init",
            "cpu_seconds": cpu_seconds,
            "train": [[pair.input, pair.output] for pair in task.train],
            "test_inputs": [list(grid) for grid in task.test_inputs],
            "memory_bytes": memory_bytes,
            "max_output_chars": max_output_chars,
            "policy": policy,
            "preload": list(preload),
        }
        self._max_output_chars = max_output_chars
        self._policy = policy
        self._startup_timeout_s = startup_timeout_s
        self._lock = threading.Lock()
        self._workdir = tempfile.mkdtemp(prefix="arc26-repl-")
        self._closed = False
        self._ready = False
        self.restarts = 0
        self._child = self._spawn()
        self._wait_ready()

    @property
    def pid(self) -> int:
        """Return the current child's process id (for diagnostics and tests)."""
        return self._child.pid

    def _spawn(self) -> ChildProcess:
        child = ChildProcess("arc26.agent.sandbox", self._workdir)
        child.send(self._init)
        self._ready = False
        return child

    def _wait_ready(self) -> None:
        if self._ready:
            return
        try:
            reply = self._child.receive(time.monotonic() + self._startup_timeout_s)
        except (TimeoutError, SandboxError) as exc:
            errors = self._child.startup_errors()
            self._child.kill()
            raise SandboxError(f"REPL failed to start: {exc!r}\n{errors}") from None
        if reply.get("ready") is not True:
            self._child.kill()
            raise SandboxError(f"REPL sent {reply!r} instead of ready")
        self._child.close_stderr()
        self._ready = True

    def _restart(self) -> None:
        self._child.kill()
        self.restarts += 1
        self._child = self._spawn()

    def run_cell(self, code: str, timeout_s: float) -> CellResult:
        """Run one cell in the persistent namespace.

        Args:
            code: Python source of the cell.
            timeout_s: Wall-time limit; on breach the child is killed and restarted with the grids
                loaded again, and the cell's state is lost.

        Returns:
            The cell's result. ``error`` is the exception's last traceback line, ``"timeout"``,
            ``"memory"``, ``"PolicyError: ..."`` (the cell was not run), or ``"crashed: ..."``
            (the child died; it was restarted).

        Raises:
            SandboxError: If the REPL is closed or a fresh child cannot start.
        """
        with self._lock:
            if self._closed:
                raise SandboxError("the REPL is closed")
            defines = guards.defines_transform(code)
            if self._policy:
                findings = guards.check_policy(code)
                if findings:
                    report = "".join(f"PolicyError: {f.rule}: {f.detail}\n" for f in findings)
                    first = f"PolicyError: {findings[0].rule}: {findings[0].detail}"
                    if findings[0].rule == "syntax_error":
                        first = f"SyntaxError: {findings[0].detail}"
                    return CellResult(report, first[:ERROR_CHARS], defines, 0.0, False)
            restarted = False
            if not self._child.alive:
                self._restart()
                restarted = True
            self._wait_ready()
            start = time.monotonic()
            try:
                self._child.send({"op": "run", "code": code})
                reply = self._child.receive(start + timeout_s)
            except TimeoutError:
                self._restart()
                return CellResult("", "timeout", defines, time.monotonic() - start, True)
            except SandboxError:
                status = self._child.exit_description()
                self._restart()
                elapsed = time.monotonic() - start
                return CellResult("", f"crashed: {status}", defines, elapsed, True)
            elapsed = time.monotonic() - start
            stdout, error = reply.get("stdout"), reply.get("error")
            if not isinstance(stdout, str) or not (error is None or isinstance(error, str)):
                self._restart()
                return CellResult("", "crashed: malformed reply", defines, elapsed, True)
            limit = self._max_output_chars + 200
            stdout = clean_text(stdout)
            error = None if error is None else clean_text(error)
            return CellResult(
                stdout[:limit], error and error[:ERROR_CHARS], defines, elapsed, restarted
            )

    def close(self) -> None:
        """Stop the child process and remove its working directory."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._child.kill()
            shutil.rmtree(self._workdir, ignore_errors=True)

    def __enter__(self) -> SubprocessRepl:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# The REPL child (this module as __main__, started by spawn_command).


def _run_cell(code: str, namespace: dict[str, Any], number: int, limit: int) -> dict[str, Any]:
    filename = f"<cell-{number}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    writer = CappedWriter(limit)
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = writer
    error = None
    try:
        tree = ast.parse(code, filename, "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, filename, "exec"), namespace)
        if last is not None:
            value = eval(compile(last, filename, "eval"), namespace)
            if value is not None:
                print(repr(value))
    except BaseException as exc:
        error = error_line(exc)
        if error != "memory":
            writer.write(_user_traceback(exc))
    finally:
        sys.stdout, sys.stderr = saved
    return {"stdout": writer.getvalue(), "error": error}


def _repl_main() -> None:
    command_fd, reply_fd = take_protocol_fds()
    reader = LineReader(command_fd)
    init = reader.read(None)
    if init is None or init.get("op") != "init":
        os._exit(2)
    apply_limits(int(init["memory_bytes"]))
    if init.get("cpu_seconds"):
        cpu = int(init["cpu_seconds"])
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
    import numpy as np

    from arc26.agent import lib

    preload_modules(init.get("preload", ()))
    np.random.seed(0)
    np.set_printoptions(linewidth=250)
    namespace: dict[str, Any] = {
        "__builtins__": restricted_builtins() if init["policy"] else builtins,
        "__name__": "__main__",
        "np": np,
        "lib": lib,
        "train": [
            (np.array(i, dtype=np.int64), np.array(o, dtype=np.int64)) for i, o in init["train"]
        ],
        "test_inputs": [np.array(g, dtype=np.int64) for g in init["test_inputs"]],
    }
    sys.setrecursionlimit(RECURSION_LIMIT)
    forbid_processes()
    silence_stderr()
    install_audit_hook(read_roots())
    write_message(reply_fd, {"ready": True})
    number = 0
    while True:
        message = reader.read(None)
        if message is None:
            os._exit(0)
        if message.get("op") == "run" and isinstance(message.get("code"), str):
            number += 1
            reply = _run_cell(message["code"], namespace, number, int(init["max_output_chars"]))
            write_message(reply_fd, reply)


if __name__ == "__main__":
    _repl_main()
