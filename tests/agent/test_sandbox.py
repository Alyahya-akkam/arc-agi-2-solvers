"""Tests of arc26.agent.sandbox: the REPL, its limits and restarts, and the shared plumbing."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arc26.agent import sandbox
from arc26.agent.sandbox import (
    CappedWriter,
    LineReader,
    SandboxError,
    SubprocessRepl,
    clean_text,
    encode_message,
    error_line,
    restricted_builtins,
)
from arc26.agent.types import AgentTask, Pair

TASK = AgentTask(
    task_id="repl",
    train=(Pair([[1, 2], [3, 4]], [[3, 1], [4, 2]]), Pair([[5, 0, 0]], [[5], [0], [0]])),
    test_inputs=([[7, 8], [9, 0]],),
)
ROOT = os.geteuid() == 0
REPO = Path(__file__).resolve().parents[2]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; it no longer runs.
    status = Path(f"/proc/{pid}/status")
    return status.exists() and "State:\tZ" not in status.read_text()


def children_of(pid):
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(fields[1]) == pid:
                found.append(int(entry.name))
    return found


@pytest.fixture(scope="module")
def repl():
    with SubprocessRepl(TASK) as instance:
        yield instance


@pytest.fixture(scope="module")
def open_repl():
    """A REPL without the Python-level layer, to test the process-level containment."""
    with SubprocessRepl(TASK, policy=False) as instance:
        yield instance


# The REPL.


def test_namespace_holds_the_grids_numpy_and_lib(repl):
    result = repl.run_cell(
        "print(type(train).__name__, len(train), type(train[0]).__name__)\n"
        "print(train[0][0].tolist(), train[0][1].tolist(), train[1][1].shape)\n"
        "print(train[0][0].dtype, [g.tolist() for g in test_inputs])\n"
        "print(np.__name__, lib.rot90(train[0][0]).tolist())",
        10,
    )
    assert result.error is None
    assert result.stdout.splitlines() == [
        "list 2 tuple",
        "[[1, 2], [3, 4]] [[3, 1], [4, 2]] (3, 1)",
        "int64 [[[7, 8], [9, 0]]]",
        "numpy [[3, 1], [4, 2]]",
    ]
    assert not result.restarted


def test_state_persists_and_the_last_expression_is_echoed(repl):
    assert repl.run_cell("counter = 41", 10).stdout == ""
    result = repl.run_cell("counter += 1\ncounter", 10)
    assert (result.stdout, result.error) == ("42\n", None)
    assert repl.run_cell("print('x')", 10).stdout == "x\n"


def test_exception_gives_last_line_and_traceback(repl):
    result = repl.run_cell("def f():\n    return 1 / 0\nf()", 10)
    assert result.error == "ZeroDivisionError: division by zero"
    assert "Traceback" in result.stdout and "return 1 / 0" in result.stdout
    assert "sandbox.py" not in result.stdout
    assert not result.restarted


def test_defines_transform_flag(repl):
    assert repl.run_cell("def transform(grid):\n    return grid", 10).defines_transform
    assert not repl.run_cell("y = transform(train[0][0])", 10).defines_transform


def test_policy_rejects_before_running(repl):
    result = repl.run_cell("marker = 1\nimport os", 10)
    assert result.error.startswith("PolicyError: forbidden_import")
    assert result.seconds == 0.0
    assert repl.run_cell("marker", 10).error == "NameError: name 'marker' is not defined"
    assert repl.run_cell("x = (", 10).error.startswith("SyntaxError:")
    assert repl.run_cell("open('/etc/passwd')", 10).error.startswith(
        "PolicyError: forbidden_builtin"
    )
    assert repl.run_cell("__import__('os')", 10).error.startswith("PolicyError")


def test_runtime_checks_catch_names_built_at_run_time(repl):
    result = repl.run_cell("getattr(np, 'c' + 'typeslib')", 10)
    assert result.error == "AttributeError: sandbox: attribute 'ctypeslib' is not allowed"
    result = repl.run_cell("getattr(np, '_' + '_dict_' + '_')", 10)
    assert "is not allowed" in result.error


def test_huge_output_is_truncated_keeping_head_and_tail(repl):
    result = repl.run_cell("print('a' * 100_000)\nprint('END')", 10)
    assert result.error is None
    assert len(result.stdout) < 8_100
    assert result.stdout.startswith("aaaa")
    assert result.stdout.rstrip().endswith("END")
    assert "characters truncated" in result.stdout


def test_memory_limit_reports_memory_and_keeps_state(repl):
    repl.run_cell("kept = 7", 10)
    result = repl.run_cell("block = bytearray(3 * 1024**3)", 20)
    assert result.error == "memory"
    assert not result.restarted
    result = repl.run_cell("big = np.ones((40_000, 40_000))", 20)
    assert result.error == "memory"
    assert repl.run_cell("kept", 10).stdout == "7\n"


def test_lone_surrogates_are_made_safe(repl):
    result = repl.run_cell("print('a\\ud800b')", 10)
    assert result.stdout == "a?b\n"
    result.stdout.encode("utf-8")


def test_timeout_kills_and_restarts_with_grids_reloaded():
    with SubprocessRepl(TASK) as instance:
        instance.run_cell("state = 1\ntrain[0][0][0, 0] = 9", 10)
        old_pid = instance.pid
        result = instance.run_cell("while True:\n    pass", 0.5)
        assert result.error == "timeout"
        assert result.restarted
        assert 0.5 <= result.seconds < 5
        assert not alive(old_pid)
        assert instance.restarts == 1
        after = instance.run_cell("print(train[0][0][0, 0])\nstate", 10)
        assert after.error == "NameError: name 'state' is not defined"
        assert after.stdout.startswith("1\n")
        assert not after.restarted
        assert instance.pid != old_pid


def test_endless_printing_times_out_with_bounded_memory():
    with SubprocessRepl(TASK) as instance:
        result = instance.run_cell("while True:\n    print('spam' * 1000)", 1.0)
        assert result.error == "timeout" and result.restarted
        assert instance.run_cell("1 + 1", 10).stdout == "2\n"


def test_close_kills_the_child_and_rejects_cells():
    instance = SubprocessRepl(TASK)
    pid = instance.pid
    workdir = Path(instance._workdir)
    assert workdir.is_dir()
    instance.close()
    instance.close()
    assert not alive(pid)
    assert not workdir.exists()
    with pytest.raises(SandboxError):
        instance.run_cell("1", 10)


def test_a_dead_child_is_replaced_before_the_next_cell():
    with SubprocessRepl(TASK) as instance:
        old_pid = instance.pid
        os.kill(old_pid, 9)
        time.sleep(0.2)
        result = instance.run_cell("train[0][1].tolist()", 10)
        assert result.restarted
        assert result.stdout == "[[3, 1], [4, 2]]\n"


# Process-level containment, with the Python-level layer off.


def test_limits_are_set_in_the_child(open_repl):
    result = open_repl.run_cell(
        "import resource\n"
        "print(resource.getrlimit(resource.RLIMIT_AS)[0] == 2 * 1024**3)\n"
        "print(resource.getrlimit(resource.RLIMIT_FSIZE))\n"
        "print(resource.getrlimit(resource.RLIMIT_NPROC))",
        10,
    )
    assert result.stdout.splitlines() == ["True", "(0, 0)", "(0, 0)"]


CONTAINED = [
    ("open('/etc/hostname').read()", "PermissionError: sandbox: open"),
    ("open('out.txt', 'w')", "PermissionError: sandbox: open"),
    ("import os\nos.listdir('/')", "PermissionError: sandbox: os.listdir"),
    ("import os\nos.fork()", "PermissionError: sandbox: os.fork"),
    ("import os\nwhile True:\n    os.fork()", "PermissionError: sandbox: os.fork"),
    ("import os\nos.system('true')", "PermissionError: sandbox: os.system"),
    ("import subprocess\nsubprocess.run(['true'])", "PermissionError: sandbox: subprocess.Popen"),
    ("import socket\nsocket.socket()", "PermissionError: sandbox: socket"),
    ("import ctypes\nctypes.CDLL(None)", "PermissionError: sandbox: ctypes.dlopen"),
    ("import os\nos.kill(os.getppid(), 9)", "PermissionError: sandbox: os.kill"),
    ("import os\nos.remove('/tmp/nothing')", "PermissionError: sandbox: os.remove"),
    ("import gc\ngc.get_objects()", "PermissionError: sandbox: gc.get_objects"),
    (
        "import resource\nresource.setrlimit(resource.RLIMIT_AS, (-1, -1))",
        "PermissionError: sandbox: resource.setrlimit",
    ),
    (
        "import typing\n"
        "def f(x: \"__import__('os').system('true')\"):\n    pass\n"
        "typing.get_type_hints(f)",
        "PermissionError: sandbox: os.system",
    ),
]


@pytest.mark.parametrize(("code", "error"), CONTAINED)
def test_audit_hook_contains_hostile_cells(open_repl, code, error):
    result = open_repl.run_cell(code, 10)
    assert result.error.startswith(error), result
    assert not result.restarted


def test_fork_bomb_creates_no_process(open_repl):
    result = open_repl.run_cell("import os\nfor _ in range(100):\n    os.fork()", 10)
    assert "os.fork" in result.error
    assert children_of(open_repl.pid) == []


@pytest.mark.skipif(ROOT, reason="RLIMIT_NPROC does not bind root")
def test_rlimit_nproc_alone_stops_fork(open_repl):
    # posix_spawn is denied by the hook too; the raw clone through ctypes is not reachable, so
    # check the limit through a thread start, which it also forbids.
    result = open_repl.run_cell("import threading\nthreading.Thread(target=print).start()", 10)
    assert result.error.startswith("RuntimeError: can't start new thread")


def test_lazy_imports_of_installed_modules_still_work(open_repl):
    result = open_repl.run_cell("import fractions\nprint(fractions.Fraction(1, 3))", 10)
    assert (result.error, result.stdout) == (None, "1/3\n")


def test_a_crashing_cell_restarts_the_child(open_repl):
    old_pid = open_repl.pid
    result = open_repl.run_cell("import os\nos.abort()", 10)
    assert result.error == "crashed: signal SIGABRT"
    assert result.restarted
    assert not alive(old_pid)
    assert open_repl.run_cell("len(train)", 10).stdout == "2\n"


def test_the_environment_is_minimal(open_repl):
    result = open_repl.run_cell("import os\nprint(sorted(os.environ))", 10)
    names = eval(result.stdout)  # the parent's own eval of the child's printed list
    assert "PATH" not in names and "HOME" in names
    assert not any("KAGGLE" in n or "TOKEN" in n for n in names)


def test_preloaded_modules_load_before_the_hook_and_missing_ones_are_skipped():
    with SubprocessRepl(TASK, policy=False, preload=("colorsys", "arc26_no_such_module")) as r:
        result = r.run_cell("import sys\nprint('colorsys' in sys.modules)", 10)
        assert (result.error, result.stdout) == (None, "True\n")


def test_scipy_in_the_repl_where_installed():
    # scipy is absent on the dev machine and in CI, present on Kaggle: there it is preloaded
    # before the audit hook, and this test checks that cells can use it.
    pytest.importorskip("scipy.ndimage")
    with SubprocessRepl(TASK) as instance:
        result = instance.run_cell(
            "import scipy.ndimage as ndi\nprint(ndi.label(train[0][0] > 2)[1])", 30
        )
        assert (result.error, result.stdout) == (None, "1\n")


# The child's import path (run P6.3-02: the parent's site folder came before the stdlib).


def test_child_path_is_the_standard_library_then_the_import_roots(open_repl):
    result = open_repl.run_cell("import sys\nprint(sys.path)\nprint(sys.flags.no_site)", 10)
    printed, no_site = result.stdout.splitlines()
    path = eval(printed)  # the parent's own eval of the child's printed list
    roots = sandbox.import_roots()
    stdlib, extra = path[: len(path) - len(roots)], path[len(path) - len(roots) :]
    assert extra == roots and stdlib and "" not in path
    bases = {os.path.realpath(p) for p in (sys.base_prefix, sys.base_exec_prefix)}
    assert all(any(os.path.realpath(p).startswith(b) for b in bases) for p in stdlib)
    # No site module: no site folders, .pth files, or sitecustomize in the child.
    assert no_site == "1"


def test_import_roots_follow_the_parents_path_and_skip_missing_packages():
    import arc26

    roots = sandbox.import_roots(("numpy", "arc26", "arc26_no_such_package"))
    assert roots == sandbox.import_roots(("arc26", "numpy"))
    assert str(Path(arc26.__file__).resolve().parents[1]) in roots
    first = {}
    for index, entry in enumerate(sys.path):
        first.setdefault(os.path.realpath(entry or os.curdir), index)
    positions = [first.get(root, len(sys.path)) for root in roots]
    assert positions == sorted(positions)


# Kaggle's image: pip installs the arc26 wheel into the site folder that also holds the obsolete
# pathlib 1.0.1 backport, whose module header fails on Python 3.10 and later.
STALE_PATHLIB = "from collections import Sequence  # the header of pathlib 1.0.1 (2014)\n"
KAGGLE_PARENT = """
import json, sys
sys.path[:] = [p for p in sys.path if p] + sys.argv[1:]
import arc26
from arc26.agent.sandbox import SubprocessRepl
from arc26.agent.types import AgentTask, Pair
from arc26.agent.verify import PoolVerifier
task = AgentTask("t", (Pair([[1, 2]], [[2, 1]]),), ([[3, 4]],))
with SubprocessRepl(task) as repl:
    cell = repl.run_cell("print(np.flip(train[0][0], 1).tolist())", 30)
with PoolVerifier(1) as verifier:
    check = verifier.verify(task, "def transform(grid):\\n    return grid[:, ::-1]\\n")
print(json.dumps({"arc26": arc26.__file__, "cell": [cell.stdout, cell.error],
                  "verified": check.verified, "test": check.test_outputs}))
"""


def test_a_stale_backport_in_the_parents_site_folder_never_shadows_the_childs_stdlib(tmp_path):
    site = tmp_path / "dist-packages"
    shutil.copytree(REPO / "arc26", site / "arc26", ignore=shutil.ignore_patterns("__pycache__"))
    (site / "pathlib.py").write_text(STALE_PATHLIB)
    numpy_root = str(Path(importlib.util.find_spec("numpy").origin).resolve().parents[1])
    # The hazard: that folder first on the path, as PYTHONPATH put it, breaks the import.
    hazard = subprocess.run(
        [sys.executable, "-S", "-c", "import pathlib"], env={"PYTHONPATH": str(site)},
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert "cannot import name 'Sequence'" in hazard.stderr
    # A parent laid out as on Kaggle (its site folder after the stdlib) runs a REPL and a verifier.
    result = subprocess.run(
        [sys.executable, "-S", "-c", KAGGLE_PARENT, str(site), numpy_root],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["arc26"].startswith(str(site))
    assert report["cell"] == ["[[2, 1]]\n", None]
    assert report["verified"] is True and report["test"] == [[[4, 3]]]


# Shared plumbing.


def test_capped_writer_keeps_head_and_tail():
    writer = CappedWriter(100)
    for i in range(1000):
        writer.write(f"{i:04d}\n")
    text = writer.getvalue()
    assert text.startswith("0000\n0001\n")
    assert text.endswith("0998\n0999\n")
    assert "characters truncated" in text
    small = CappedWriter(100)
    small.write("short")
    assert small.getvalue() == "short"
    with pytest.raises(TypeError):
        small.write(b"bytes")
    assert CappedWriter(0).write("anything") == 8


def test_line_reader_deadlines_size_cap_and_bad_json():
    read_fd, write_fd = os.pipe()
    try:
        reader = LineReader(read_fd, max_bytes=64)
        with pytest.raises(TimeoutError):
            reader.read(time.monotonic() + 0.05)
        os.write(write_fd, encode_message({"a": 1}) + b'{"b": 2}\n')
        assert reader.read(None) == {"a": 1}
        assert reader.read(None) == {"b": 2}
        os.write(write_fd, b"not json\n")
        with pytest.raises(SandboxError):
            reader.read(None)
        os.write(write_fd, b"[1, 2]\n")
        with pytest.raises(SandboxError):
            reader.read(None)
        os.write(write_fd, b"x" * 200)
        with pytest.raises(SandboxError):
            reader.read(None)
    finally:
        os.close(write_fd)
        os.close(read_fd)
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    assert LineReader(read_fd).read(None) is None
    os.close(read_fd)


def test_error_line_and_clean_text():
    assert error_line(MemoryError()) == "memory"
    assert error_line(ValueError("bad")) == "ValueError: bad"
    assert len(error_line(ValueError("x" * 5000))) == sandbox.ERROR_CHARS

    class Nasty(Exception):
        def __str__(self):
            raise RuntimeError("no")

    assert "Nasty" in error_line(Nasty())
    assert clean_text("a\ud800") == "a?"


def test_restricted_builtins_drop_and_guard():
    safe = restricted_builtins()
    for name in ("open", "eval", "exec", "compile", "globals", "vars", "input", "breakpoint"):
        assert name not in safe
    assert "print" in safe and "__build_class__" in safe
    with pytest.raises(ImportError):
        safe["__import__"]("os")
    with pytest.raises(ImportError):
        safe["__import__"]("numpy", fromlist=("load",))
    with pytest.raises(ImportError):
        safe["__import__"]("numpy", fromlist=("*",))
    with pytest.raises(ImportError):
        safe["__import__"]("numpy", level=1)
    assert safe["__import__"]("numpy").__name__ == "numpy"
    assert safe["__import__"]("arc26.agent", fromlist=("lib",)).lib.__name__ == "arc26.agent.lib"
    with pytest.raises(ImportError):
        safe["__import__"]("arc26.agent", fromlist=("sandbox",))
    with pytest.raises(AttributeError):
        safe["getattr"](sandbox, "o" + "s")
    with pytest.raises(AttributeError):
        safe["setattr"](TASK, "__class__", object)
    assert safe["getattr"](TASK, "missing", 5) == 5
    assert safe["hasattr"](TASK, "task_id")
