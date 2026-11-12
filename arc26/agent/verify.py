"""The verifier: judges a program on a task in fresh, least-privilege processes (P6.1, ADR-020).

Why: a program becomes an attempt only if it reproduces every demonstration pair, so the judgement
must not depend on anything but the program's source: not on REPL state, files, or the
demonstration outputs themselves. :class:`PoolVerifier` keeps a pool of worker processes (this
module, started as the REPL is: :func:`arc26.agent.sandbox.spawn_command`); each loads numpy, the
helper library, and the sandbox's restricted builtins once, then forks a fresh child per program.
The child runs the source in a new namespace holding only ``np`` and ``lib``, calls ``transform``
on every demonstration input and every test input, and streams one JSON line per call back to the
worker, which enforces the per-call time limit, kills a child that overruns or crashes, and
continues with the next input in a new child. The child never sees an expected output: the parent
compares, so even code that escapes the Python-level guards cannot read the answers from its own
memory.

Layers (see :mod:`arc26.agent.sandbox` for the shared ones): the AST policy and hardcoding guards
(parent, before running; a safety finding means the program never runs), restricted builtins, an
audit hook, ``RLIMIT_AS`` (default 2 GiB), no file writes, no new processes, a process group per
child (killed on timeout), and JSON replies validated by the parent.
"""

from __future__ import annotations

import builtins
import ctypes
import gc
import hashlib
import math
import os
import queue
import resource
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Any, NoReturn

from arc26.agent import guards
from arc26.agent.sandbox import (
    DEFAULT_MEMORY_BYTES,
    DEFAULT_PRELOAD,
    ERROR_CHARS,
    RECURSION_LIMIT,
    CappedWriter,
    ChildProcess,
    LineReader,
    SandboxError,
    apply_limits,
    clean_text,
    error_line,
    forbid_processes,
    install_audit_hook,
    preload_modules,
    read_roots,
    restricted_builtins,
    silence_stderr,
    take_protocol_fds,
    write_message,
)
from arc26.agent.types import AgentTask, PairResult, Verification
from arc26.data import Grid

DEFAULT_WORKERS = 8
DEFAULT_CALL_TIMEOUT_S = 2.0
MAX_SIDE = 30
_PR_SET_PDEATHSIG = 1
_MAX_RESULT_BYTES = 256 * 1024
_SLACK_S = 1.0


def to_grid(value: object) -> tuple[Grid | None, str | None]:
    """Convert a ``transform`` result to a grid, or say why it is not one.

    Why: programs return numpy arrays or nested lists; a valid grid is rectangular, 1 to 30 cells
    per side, with integer colors 0 to 9 (integral floats are accepted, booleans are not).

    Args:
        value: What ``transform`` returned.

    Returns:
        ``(grid, None)`` for a valid grid as lists of Python ints, else ``(None, reason)``.
    """
    import numpy as np

    rows: list[list[int]] = []
    if isinstance(value, np.ndarray):
        # Shape first, so a huge array is refused before any pass over its cells.
        if value.ndim != 2:
            return None, f"invalid output: a {value.ndim}-D array, not 2-D"
        if not 1 <= value.shape[0] <= MAX_SIDE or not 1 <= value.shape[1] <= MAX_SIDE:
            return None, f"invalid output: shape {tuple(value.shape)}; sides must be 1 to 30"
        kind = value.dtype.kind
        if kind == "b":
            return None, "invalid output: a boolean array, not colors"
        if kind == "O":
            return to_grid(value.tolist())
        if kind not in "iuf":
            return None, f"invalid output: an array of dtype {value.dtype}"
        if kind == "f" and not bool(np.all(np.isfinite(value) & (value == np.round(value)))):
            return None, "invalid output: non-integral float colors"
        # Range before the cast, which would wrap values beyond int64.
        if not bool(np.all((value >= 0) & (value <= 9))):
            return None, "invalid output: colors must be 0 to 9"
        return value.astype(np.int64).tolist(), None
    elif isinstance(value, list | tuple):
        if not 1 <= len(value) <= MAX_SIDE:
            return None, f"invalid output: {len(value)} rows; sides must be 1 to 30"
        for row in value:
            if isinstance(row, np.ndarray) and row.ndim == 1:
                row = row.tolist()
            if not isinstance(row, list | tuple):
                return None, f"invalid output: a row is {type(row).__name__}, not a list"
            cells = []
            for item in row:
                cell = _cell(item)
                if cell is None:
                    return None, "invalid output: a cell is not an integer color"
                cells.append(cell)
            rows.append(cells)
        if len({len(row) for row in rows}) != 1:
            return None, "invalid output: rows of different lengths"
        if not 1 <= len(rows[0]) <= MAX_SIDE:
            return None, f"invalid output: {len(rows[0])} columns; sides must be 1 to 30"
    else:
        return None, f"invalid output: {type(value).__name__}, not a 2-D grid"
    if any(not 0 <= c <= 9 for row in rows for c in row):
        return None, "invalid output: colors must be 0 to 9"
    return rows, None


def _cell(value: object) -> int | None:
    import numpy as np

    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    if (
        isinstance(value, float | np.floating)
        and math.isfinite(value)
        and float(value).is_integer()
    ):
        return int(value)
    return None


def strict_grid(value: object) -> Grid | None:
    """Accept a decoded JSON value only if it is exactly a valid grid (parent side).

    Why: replies come from a process that ran model code, so the parent trusts no structure it did
    not check itself.

    Args:
        value: A decoded JSON value.

    Returns:
        A fresh copy of the grid, or None.
    """
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SIDE:
        return None
    width = None
    grid: Grid = []
    for row in value:
        if not isinstance(row, list) or not 1 <= len(row) <= MAX_SIDE:
            return None
        if width is None:
            width = len(row)
        if len(row) != width:
            return None
        if not all(type(c) is int and 0 <= c <= 9 for c in row):
            return None
        grid.append(list(row))
    return grid


def _clean_text(value: object) -> str:
    text = value if isinstance(value, str) else "invalid error text"
    return clean_text(text)[:ERROR_CHARS]


def _pair_result(index: int, expected: Grid, item: dict[str, Any]) -> PairResult:
    grid = item.get("grid")
    if grid is None:
        return PairResult(index, False, None, item.get("error") or "no output", False, 0.0)
    shape_match = len(grid) == len(expected) and len(grid[0]) == len(expected[0])
    accuracy = 0.0
    if shape_match:
        same = sum(
            a == b
            for got, want in zip(grid, expected, strict=True)
            for a, b in zip(got, want, strict=True)
        )
        accuracy = same / (len(grid) * len(grid[0]))
    return PairResult(index, grid == expected, grid, None, shape_match, accuracy)


class _Worker:
    """One worker process (parent side) and its startup handshake."""

    def __init__(self, workdir: str, init: dict[str, Any], startup_timeout_s: float) -> None:
        self.child = ChildProcess("arc26.agent.verify", workdir)
        self.child.send(init)
        self._ready = False
        self._startup_timeout_s = startup_timeout_s

    def ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            reply = self.child.receive(time.monotonic() + self._startup_timeout_s)
        except (TimeoutError, SandboxError) as exc:
            errors = self.child.startup_errors()
            self.child.kill()
            raise SandboxError(f"verifier worker failed to start: {exc!r}\n{errors}") from None
        if reply.get("ready") is not True:
            self.child.kill()
            raise SandboxError(f"verifier worker sent {reply!r} instead of ready")
        self.child.close_stderr()
        self._ready = True


class PoolVerifier:
    """Verifies programs in a pool of sandboxed worker processes; implements ``Verifier``.

    Thread-safe: callers may run :meth:`verify` from many threads at once (the agent runner does,
    through ``asyncio.to_thread``). Each call checks one worker out of a queue for its exclusive
    use, waiting while all are busy, and keeps all per-call state local; the set of live workers
    and the replacement count change only under a lock. A worker that crashes, hangs, or breaks
    the protocol is killed and replaced, so the pool keeps its size; none is created after
    :meth:`close`.

    Args:
        workers: Number of worker processes.
        call_timeout_s: Wall-time limit of each ``transform`` call, and of the program's top-level
            code.
        memory_bytes: Address-space limit of each program's process.
        policy: Apply the AST policy (programs with a safety finding never run) and the restricted
            builtins. False only in tests of the process-level containment, which is always on.
        preload: Modules each worker imports at startup when installed, before any program runs
            (see :data:`arc26.agent.sandbox.DEFAULT_PRELOAD`); the REPL loads the same ones.
        startup_timeout_s: How long a worker may take to start.
        hang_grace_s: Margin beyond a program's own time budget, ``(inputs + 1)`` call limits,
            after which a silent worker counts as hung and is replaced.

    Raises:
        ValueError: If ``workers`` is less than 1.
    """

    def __init__(
        self,
        workers: int = DEFAULT_WORKERS,
        *,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        memory_bytes: int = DEFAULT_MEMORY_BYTES,
        policy: bool = True,
        preload: Sequence[str] = DEFAULT_PRELOAD,
        startup_timeout_s: float = 60.0,
        hang_grace_s: float = 10.0,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._call_timeout_s = call_timeout_s
        self._hang_grace_s = hang_grace_s
        self._policy = policy
        self._startup_timeout_s = startup_timeout_s
        self._init = {
            "op": "init",
            "memory_bytes": memory_bytes,
            "policy": policy,
            "preload": list(preload),
        }
        self._workdir = tempfile.mkdtemp(prefix="arc26-verify-")
        self._lock = threading.Lock()
        self._closed = False
        self._live: set[_Worker] = set()
        self._replaced = 0
        self._idle: queue.Queue[_Worker] = queue.Queue()
        for _ in range(workers):
            self._idle.put(self._new_worker())

    @property
    def replaced(self) -> int:
        """Return how many workers were replaced after a crash, a hang, or a protocol error."""
        with self._lock:
            return self._replaced

    def worker_pids(self) -> list[int]:
        """Return the process ids of the live workers (for diagnostics and tests).

        Returns:
            The pids, one per worker in the pool.
        """
        with self._lock:
            return [w.child.pid for w in self._live]

    def _new_worker(self) -> _Worker:
        with self._lock:
            if self._closed:
                raise SandboxError("the verifier is closed")
            worker = _Worker(self._workdir, self._init, self._startup_timeout_s)
            self._live.add(worker)
            return worker

    def _discard(self, worker: _Worker) -> None:
        worker.child.kill()
        with self._lock:
            self._live.discard(worker)

    def _replace(self, worker: _Worker) -> _Worker:
        self._discard(worker)
        with self._lock:
            self._replaced += 1
        return self._new_worker()

    def verify(self, task: AgentTask, source: str) -> Verification:
        """Verify one self-contained program on one task.

        Args:
            task: The task.
            source: Python source that defines ``transform(grid)``. ``transform`` receives each
                input as a fresh 2-D ``numpy.int64`` array and returns a grid (an array or nested
                lists). The namespace starts with ``np`` and ``lib``; imports follow the AST policy.

        Returns:
            The verification: one result per demonstration pair, the test outputs, and every guard
            finding.

        Raises:
            SandboxError: If the verifier is closed or a worker cannot start.
        """
        start = time.monotonic()
        digest = hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()
        findings = list(guards.check_program(source, task) if self._policy else ())
        inputs = [pair.input for pair in task.train] + list(task.test_inputs)
        safety = [f for f in findings if f.rule in guards.SAFETY_RULES]
        if safety:
            items = [{"error": f"not run: {safety[0].rule}"}] * len(inputs)
        else:
            items = self._run(source, inputs)
        n_train = len(task.train)
        pairs = tuple(_pair_result(k, pair.output, items[k]) for k, pair in enumerate(task.train))
        test_outputs = tuple(item.get("grid") for item in items[n_train:])
        findings.extend(guards.check_outputs(task, test_outputs))
        return Verification(
            source=source,
            source_sha256=digest,
            pairs=pairs,
            test_outputs=test_outputs,
            guards=tuple(findings),
            seconds=time.monotonic() - start,
        )

    def _acquire(self) -> _Worker:
        while True:
            if self._closed:
                raise SandboxError("the verifier is closed")
            try:
                # A bounded wait, so a caller blocked here notices close().
                worker = self._idle.get(timeout=0.2)
            except queue.Empty:
                continue
            break
        try:
            if self._closed:
                raise SandboxError("the verifier is closed")
            if not worker.child.alive:
                worker = self._replace(worker)
            worker.ensure_ready()
        except SandboxError:
            self._discard(worker)
            if not self._closed:
                self._idle.put(self._new_worker())
            raise
        return worker

    def _release(self, worker: _Worker) -> None:
        if self._closed:
            self._discard(worker)
        else:
            self._idle.put(worker)

    def _run(self, source: str, inputs: Sequence[Grid]) -> list[dict[str, Any]]:
        n = len(inputs)
        job = {
            "op": "verify",
            "source": source,
            "inputs": [list(grid) for grid in inputs],
            "call_timeout": self._call_timeout_s,
        }
        worker = self._acquire()
        try:
            # The worker's own budget is (n + 1) call limits plus slack; this backstop, counted
            # from checkout, adds margin for forking and a loaded machine, then calls it hung.
            budget = (n + 1) * (self._call_timeout_s + 0.5)
            deadline = time.monotonic() + budget + self._hang_grace_s
            worker.child.send(job)
            items = _parse_items(worker.child.receive(deadline), n)
        except (TimeoutError, SandboxError) as exc:
            failure = "hung" if isinstance(exc, TimeoutError) else str(exc)
            worker = self._replace(worker)
            items = [{"error": _clean_text(f"worker failure: {failure}")}] * n
        finally:
            self._release(worker)
        return items

    def close(self) -> None:
        """Kill every worker and remove the working directory; later calls raise.

        Idle workers are discarded here. A busy worker's process group is killed, but its pipes
        stay open for the thread that is reading them: that thread sees end of file, raises
        ``SandboxError``, and discards the worker itself.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            busy = set(self._live)
        while True:
            try:
                worker = self._idle.get_nowait()
            except queue.Empty:
                break
            busy.discard(worker)
            self._discard(worker)
        for worker in busy:
            worker.child.terminate()
        shutil.rmtree(self._workdir, ignore_errors=True)

    def __enter__(self) -> PoolVerifier:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _parse_items(reply: dict[str, Any], n: int) -> list[dict[str, Any]]:
    results = reply.get("results")
    if not isinstance(results, list) or len(results) != n:
        raise SandboxError("malformed results")
    items = []
    for item in results:
        if isinstance(item, dict) and "grid" in item:
            grid = strict_grid(item["grid"])
            items.append({"grid": grid} if grid is not None else {"error": "invalid result"})
        elif isinstance(item, dict) and "error" in item:
            items.append({"error": _clean_text(item["error"])})
        else:
            items.append({"error": "invalid result"})
    return items


# The worker process (this module as __main__, started by spawn_command).


class _Context:
    """What a worker prepares once and every program child inherits through fork."""

    def __init__(self, program_builtins: Any, roots: list[str], libc: Any) -> None:
        self.program_builtins = program_builtins
        self.roots = roots
        self.libc = libc
        self.pid = os.getpid()


def _program_child(job: dict[str, Any], start: int, write_fd: int, context: _Context) -> NoReturn:
    """Run the program on inputs ``start..`` in this forked child; never returns."""
    try:
        os.setpgid(0, 0)
        # Die with the worker, so no program outlives the pool.
        context.libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        if os.getppid() != context.pid:
            os._exit(1)
        _close_fds_except(write_fd)
        inputs = job["inputs"]
        timeout = float(job["call_timeout"])
        cpu = math.ceil(timeout * (len(inputs) - start + 1)) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        forbid_processes()
        sys.stdout = sys.stderr = CappedWriter(0)
        sys.setrecursionlimit(RECURSION_LIMIT)
        install_audit_hook(context.roots)
        import numpy as np

        from arc26.agent import lib

        namespace: dict[str, Any] = {
            "__builtins__": context.program_builtins,
            "__name__": "program",
            "np": np,
            "lib": lib,
        }
        try:
            exec(compile(job["source"], "<program>", "exec"), namespace)
            transform = namespace.get("transform")
            if not callable(transform):
                raise NameError("the program defines no callable transform")
        except BaseException as exc:
            write_message(write_fd, {"exec_error": error_line(exc)})
            os._exit(0)
        write_message(write_fd, {"exec": "ok"})
        for index in range(start, len(inputs)):
            try:
                grid, problem = to_grid(transform(np.array(inputs[index], dtype=np.int64)))
            except BaseException as exc:
                grid, problem = None, error_line(exc)
            write_message(write_fd, {"grid": grid} if grid is not None else {"error": problem})
    except BaseException:
        os._exit(3)
    os._exit(0)


def _close_fds_except(keep: int) -> None:
    """Close every descriptor above stderr but ``keep`` (the worker's pipes above all)."""
    # Listing /proc/self/fd is O(open fds); os.closerange up to RLIMIT_NOFILE costs a quarter
    # second per fork when the interpreter lacks close_range(2).
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        if fd > 2 and fd != keep:
            try:
                os.close(fd)
            except OSError:
                pass


class _ProgramChild:
    """A forked program child: killed and reaped exactly once, so its pid is never reused."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._status: str | None = None

    def end(self) -> str:
        """Kill the child's group (a no-op once reaped), reap it, and describe how it ended."""
        if self._status is None:
            # The child is unreaped here, so its pid and group id still name it alone.
            for kill in (os.killpg, os.kill):
                try:
                    kill(self.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            _, status = os.waitpid(self.pid, 0)
            if os.WIFSIGNALED(status):
                try:
                    self._status = f"signal {signal.Signals(os.WTERMSIG(status)).name}"
                except ValueError:
                    self._status = f"signal {os.WTERMSIG(status)}"
            else:
                self._status = f"exit code {os.WEXITSTATUS(status)}"
        return self._status


def _sanitize(message: dict[str, Any]) -> dict[str, Any]:
    """Reduce a program child's reply to a valid grid or a short error text."""
    if "grid" in message:
        grid = strict_grid(message["grid"])
        return {"grid": grid} if grid is not None else {"error": "invalid result"}
    if isinstance(message.get("error"), str):
        return {"error": message["error"][:ERROR_CHARS]}
    return {"error": "invalid result"}


def _collect(
    reader: LineReader,
    child: _ProgramChild,
    results: list[dict[str, Any] | None],
    start: int,
    timeout: float,
    deadline: float,
) -> int:
    """Read one child's replies into ``results``; return the index the next child starts at."""
    n = len(results)

    def fill(first: int, text: str) -> int:
        for k in range(first, n):
            results[k] = {"error": text}
        return n

    def step() -> float:
        return min(time.monotonic() + timeout, deadline)

    try:
        message = reader.read(step())
    except TimeoutError:
        return fill(start, "timeout (top-level code)")
    except SandboxError:
        return fill(start, "invalid result")
    if message is None:
        return fill(start, f"crashed: {child.end()}")
    if "exec_error" in message:
        return fill(start, _sanitize({"error": message["exec_error"]})["error"])
    for index in range(start, n):
        try:
            message = reader.read(step())
        except TimeoutError:
            results[index] = {"error": "timeout"}
            return index + 1
        except SandboxError:
            results[index] = {"error": "invalid result"}
            return index + 1
        if message is None:
            results[index] = {"error": f"crashed: {child.end()}"}
            return index + 1
        results[index] = _sanitize(message)
    return n


def _run_job(job: dict[str, Any], context: _Context) -> list[dict[str, Any]]:
    """Run one verification: a child per stretch of inputs, a new one after each failure."""
    n = len(job["inputs"])
    timeout = float(job["call_timeout"])
    deadline = time.monotonic() + (n + 1) * timeout + _SLACK_S
    results: list[dict[str, Any] | None] = [None] * n
    index = 0
    while index < n:
        if time.monotonic() >= deadline:
            for k in range(index, n):
                results[k] = {"error": "timeout"}
            break
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            _program_child(job, index, write_fd, context)
        os.close(write_fd)
        child = _ProgramChild(pid)
        try:
            os.setpgid(pid, pid)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            reader = LineReader(read_fd, _MAX_RESULT_BYTES)
            index = _collect(reader, child, results, index, timeout, deadline)
        finally:
            child.end()
            os.close(read_fd)
    return [r if r is not None else {"error": "not run"} for r in results]


def _worker_main() -> None:
    command_fd, reply_fd = take_protocol_fds()
    reader = LineReader(command_fd)
    init = reader.read(None)
    if init is None or init.get("op") != "init":
        os._exit(2)
    apply_limits(int(init["memory_bytes"]))
    import numpy as np

    from arc26.agent import lib  # noqa: F401 (loaded once, inherited by every program child)

    preload_modules(init.get("preload", ()))
    np.random.seed(0)
    program_builtins = restricted_builtins() if init["policy"] else builtins
    context = _Context(program_builtins, read_roots(), ctypes.CDLL(None, use_errno=True))
    gc.collect()
    # Objects that exist now are never collected in the children, so forks share their pages.
    gc.freeze()
    silence_stderr()
    write_message(reply_fd, {"ready": True})
    while True:
        job = reader.read(None)
        if job is None:
            os._exit(0)
        if job.get("op") == "verify":
            write_message(reply_fd, {"results": _run_job(job, context)})


if __name__ == "__main__":
    _worker_main()
