"""Tests of arc26.agent.verify: results, limits, guards, isolation, pool recovery, concurrency."""

import hashlib
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from arc26.agent import sandbox
from arc26.agent.sandbox import SandboxError
from arc26.agent.types import AgentTask, Pair
from arc26.agent.verify import PoolVerifier, strict_grid, to_grid


def rot90(grid):
    return [list(row) for row in zip(*grid[::-1], strict=True)]


A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 7], [0, 7]]
C = [[1, 1, 2], [3, 5, 8], [1, 3, 2], [1, 3, 4]]
TESTS = ([[9, 8, 7]], [[2, 4], [6, 8], [1, 3]])
TASK = AgentTask("verify", (Pair(A, rot90(A)), Pair(B, rot90(B)), Pair(C, rot90(C))), TESTS)

ROTATE = "def transform(grid):\n    return np.rot90(grid, -1)\n"
IDENTITY = "def transform(grid):\n    return grid\n"
HANG = "def transform(grid):\n    while True:\n        pass\n"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/status")
    return status.exists() and "State:\tZ" not in status.read_text()


def children_of(pids):
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(fields[1]) in pids:
                found.append(int(entry.name))
    return found


@pytest.fixture(scope="module")
def verifier():
    with PoolVerifier(4, call_timeout_s=1.0) as pool:
        yield pool


@pytest.fixture(scope="module")
def open_verifier():
    """A verifier without the Python-level layer, to test the process-level containment."""
    with PoolVerifier(2, call_timeout_s=1.0, policy=False) as pool:
        yield pool


# Results.


def test_correct_program_is_verified(verifier):
    result = verifier.verify(TASK, ROTATE)
    assert result.verified and result.passed == 3
    assert [p.exact for p in result.pairs] == [True, True, True]
    assert [p.index for p in result.pairs] == [0, 1, 2]
    assert result.pairs[2].output == rot90(C)
    assert all(p.shape_match and p.cell_accuracy == 1.0 and p.error is None for p in result.pairs)
    assert result.test_outputs == (rot90(TESTS[0]), rot90(TESTS[1]))
    assert result.guards == ()
    assert result.source == ROTATE
    assert result.source_sha256 == hashlib.sha256(ROTATE.encode()).hexdigest()
    assert result.seconds > 0


def test_wrong_program_scores_shape_and_cells(verifier):
    result = verifier.verify(TASK, IDENTITY)
    assert not result.verified and result.passed == 0
    first, second, third = result.pairs
    assert (first.shape_match, first.cell_accuracy) == (False, 0.0)
    assert (second.shape_match, second.cell_accuracy) == (True, 0.5)
    assert second.output == B and second.error is None
    assert not third.shape_match
    # Identity outputs equal the test inputs although no demonstration is an identity.
    assert [g.rule for g in result.guards] == ["output_equals_input", "output_equals_input"]
    assert not any(g.fatal for g in result.guards)


def test_partially_correct_program(verifier):
    source = (
        "def transform(grid):\n"
        "    if grid.shape[0] == grid.shape[1]:\n"
        "        return np.rot90(grid, -1)\n"
        "    return grid\n"
    )
    result = verifier.verify(TASK, source)
    assert result.passed == 1
    assert [p.exact for p in result.pairs] == [False, True, False]
    assert not result.verified


def test_exception_in_transform(verifier):
    result = verifier.verify(TASK, "def transform(grid):\n    return 1 // 0\n")
    assert all(
        p.error == "ZeroDivisionError: integer division or modulo by zero" for p in result.pairs
    )
    assert all(p.output is None and not p.shape_match for p in result.pairs)
    assert result.test_outputs == (None, None)


def test_verified_needs_every_test_output(verifier):
    source = (
        "def transform(grid):\n"
        "    if grid.shape[0] == 1:\n"
        "        raise ValueError('no rule for this input')\n"
        "    return np.rot90(grid, -1)\n"
    )
    result = verifier.verify(TASK, source)
    assert result.passed == 3
    assert result.test_outputs[0] is None and result.test_outputs[1] == rot90(TESTS[1])
    assert not result.verified


def test_lib_is_available_to_programs(verifier):
    for source in (
        "def transform(grid):\n    return lib.rot90(grid)\n",
        "from arc26.agent.lib import rot90\ndef transform(grid):\n    return rot90(grid)\n",
        "from arc26.agent import lib as L\ntransform = L.rot90\n",
    ):
        assert verifier.verify(TASK, source).verified, source


INVALID = [
    ("None", "invalid output: NoneType"),
    ("'abc'", "invalid output: str"),
    ("[]", "invalid output: 0 rows"),
    ("[[1, 2], [3]]", "invalid output: rows of different lengths"),
    ("[[10]]", "invalid output: colors must be 0 to 9"),
    ("[[-1]]", "invalid output: colors must be 0 to 9"),
    ("[[1.5]]", "invalid output: a cell is not an integer color"),
    ("[[True]]", "invalid output: a cell is not an integer color"),
    ("[[[1]]]", "invalid output: a cell is not an integer color"),
    ("[1, 2]", "invalid output: a row is int"),
    ("np.zeros((31, 1), int)", "invalid output: shape (31, 1)"),
    ("np.zeros((2, 2), bool)", "invalid output: a boolean array"),
    ("np.zeros((2, 2, 2), int)", "invalid output: a 3-D array"),
    ("np.full((2, 2), 0.5)", "invalid output: non-integral float colors"),
    ("np.full((2, 2), np.nan)", "invalid output: non-integral float colors"),
    ("np.array([['a']])", "invalid output: an array of dtype <U1"),
]


@pytest.mark.parametrize(("value", "error"), INVALID)
def test_invalid_outputs_are_rejected(verifier, value, error):
    result = verifier.verify(TASK, f"def transform(grid):\n    return {value}\n")
    assert all(p.error.startswith(error) and p.output is None for p in result.pairs), result
    assert result.test_outputs == (None, None)


def test_to_grid_converts_valid_grids():
    assert to_grid(np.array([[1, 2]], dtype=np.int8)) == ([[1, 2]], None)
    assert to_grid(np.zeros((2, 2))) == ([[0, 0], [0, 0]], None)
    assert to_grid(((1, 2), (3, 4))) == ([[1, 2], [3, 4]], None)
    assert to_grid([np.array([1, 2]), [np.int64(3), 4.0]]) == ([[1, 2], [3, 4]], None)
    assert to_grid(np.array([[1, 2]], dtype=object)) == ([[1, 2]], None)
    grid, error = to_grid(np.zeros((30, 30), dtype=np.uint8))
    assert error is None and len(grid) == 30
    assert to_grid(np.full((1, 1), 1e300)) == (None, "invalid output: colors must be 0 to 9")
    assert to_grid(np.full((1, 1), 2**64 - 1, dtype=np.uint64))[0] is None
    assert to_grid(np.zeros((10_000, 10_000), dtype=np.float32))[1].startswith(
        "invalid output: shape"
    )


def test_strict_grid_rejects_anything_else():
    assert strict_grid([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    for value in ([], [[]], [[1], [2, 3]], [[True]], [[1.0]], [[10]], [["1"]], "x", [[0] * 31]):
        assert strict_grid(value) is None, value


# Limits.


def test_a_hang_on_one_input_spares_the_others(verifier):
    source = (
        "def transform(grid):\n"
        "    if grid[0, 0] == 7:\n"
        "        while True:\n"
        "            pass\n"
        "    return np.rot90(grid, -1)\n"
    )
    result = verifier.verify(TASK, source)
    assert [p.error for p in result.pairs] == [None, "timeout", None]
    assert [p.exact for p in result.pairs] == [True, False, True]
    assert result.test_outputs == (rot90(TESTS[0]), rot90(TESTS[1]))
    assert 1.0 <= result.seconds < 4.0


def test_top_level_code_is_time_limited_too(verifier):
    result = verifier.verify(TASK, "while True:\n    pass\n" + ROTATE)
    assert all(p.error == "timeout (top-level code)" for p in result.pairs)


def test_top_level_error_and_missing_transform(verifier):
    result = verifier.verify(TASK, "x = [][0]\n" + ROTATE)
    assert all(p.error == "IndexError: list index out of range" for p in result.pairs)
    result = verifier.verify(TASK, "def solve(grid):\n    return grid\n")
    assert all(
        p.error == "NameError: the program defines no callable transform" for p in result.pairs
    )


def test_memory_limit(verifier):
    source = "def transform(grid):\n    block = bytearray(3 * 1024**3)\n    return grid\n"
    result = verifier.verify(TASK, source)
    assert all(p.error == "memory" for p in result.pairs)


def test_runtime_attribute_check(verifier):
    source = "def transform(grid):\n    return getattr(np, 'c' + 'typeslib')\n"
    result = verifier.verify(TASK, source)
    assert result.pairs[0].error == "AttributeError: sandbox: attribute 'ctypeslib' is not allowed"


# Guards.


def test_a_policy_violation_never_runs(verifier):
    start = time.monotonic()
    result = verifier.verify(TASK, "import os\nwhile True:\n    pass\n" + ROTATE)
    assert time.monotonic() - start < 0.5
    assert all(p.error == "not run: forbidden_import" for p in result.pairs)
    assert [g.rule for g in result.guards] == ["forbidden_import"]
    assert result.guards[0].fatal and not result.verified


def test_a_hardcoded_program_passes_the_demos_but_is_rejected(verifier):
    table = {(2, 3): rot90(A), (2, 2): rot90(B), (4, 3): rot90(C)}
    source = f"OUT = {table!r}\n\ndef transform(grid):\n    return OUT[grid.shape]\n"
    result = verifier.verify(TASK, source)
    assert result.passed == 3
    rules = {g.rule for g in result.guards}
    assert {"literal_output", "shape_dispatch"} <= rules
    assert [g.fatal for g in result.guards if g.rule == "literal_output"] == [True, True, True]
    assert not result.verified


def test_a_padded_hardcoded_program_passes_the_demos_but_is_rejected(verifier):
    def pad(grid):
        width = len(grid[0]) + 2
        return [[0] * width] + [[0, *row, 0] for row in grid] + [[0] * width]

    table = {(2, 3): pad(rot90(A)), (2, 2): pad(rot90(B)), (4, 3): pad(rot90(C))}
    source = (
        f"OUT = {table!r}\n\n"
        "def transform(grid):\n"
        "    return np.array(OUT[grid.shape])[1:-1, 1:-1]\n"
    )
    result = verifier.verify(TASK, source)
    assert result.passed == 3
    # Output 2 has 12 cells; outputs 0 and 1 are below the 9-cell floor of the containment test.
    assert [(g.rule, g.fatal) for g in result.guards if g.rule == "literal_output"] == [
        ("literal_output", True)
    ]
    assert not result.verified


def test_a_padded_constant_output_literal_verifies_with_a_demotion(verifier):
    shared = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    task = AgentTask("const", (Pair(A, shared), Pair(B, shared)), TESTS)
    padded = [[0] * 5] + [[0, *row, 0] for row in shared] + [[0] * 5]
    source = f"def transform(grid):\n    return np.array({padded!r})[1:4, 1:4]\n"
    result = verifier.verify(task, source)
    assert result.verified
    assert [(g.rule, g.fatal) for g in result.guards] == [("constant_output_literal", False)]


def test_a_stencil_program_verifies_with_a_demotion(verifier):
    # Stamp a fixed 3 by 3 cross at every red cell (2): the cross is a block of every output.
    def stamp(grid):
        out = [row[:] for row in grid]
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == 2:
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= rr < len(grid) and 0 <= cc < len(row):
                            out[rr][cc] = 4
        return out

    inputs = [
        [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0]],
    ]
    test_input = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [2, 0, 0, 0]]
    task = AgentTask("cross", tuple(Pair(g, stamp(g)) for g in inputs), (test_input,))
    source = (
        "CROSS = np.array([[0, 4, 0], [4, 2, 4], [0, 4, 0]])\n"
        "def transform(grid):\n"
        "    out = grid.copy()\n"
        "    height, width = grid.shape\n"
        "    for r, c in zip(*np.nonzero(grid == 2)):\n"
        "        for dr in range(3):\n"
        "            for dc in range(3):\n"
        "                rr, cc = r + dr - 1, c + dc - 1\n"
        "                if CROSS[dr, dc] and 0 <= rr < height and 0 <= cc < width:\n"
        "                    out[rr, cc] = CROSS[dr, dc]\n"
        "    return out\n"
    )
    result = verifier.verify(task, source)
    assert result.verified and result.passed == 2
    assert result.test_outputs == (stamp(test_input),)
    assert [(g.rule, g.fatal) for g in result.guards] == [("partial_output_literal", False)]


def test_a_constant_output_literal_verifies_with_a_demotion(verifier):
    shared = [[1, 2], [3, 4]]
    task = AgentTask("const", (Pair(A, shared), Pair(B, shared), Pair(C, shared)), TESTS)
    result = verifier.verify(task, f"def transform(grid):\n    return {shared!r}\n")
    assert result.verified
    assert result.test_outputs == (shared, shared)
    assert [(g.rule, g.fatal) for g in result.guards] == [("constant_output_literal", False)]


# Isolation.


def test_the_worker_never_receives_demonstration_outputs(monkeypatch):
    sent = []
    original = sandbox.ChildProcess.send

    def spy(self, message):
        sent.append(json.loads(json.dumps(message)))
        original(self, message)

    monkeypatch.setattr(sandbox.ChildProcess, "send", spy)
    with PoolVerifier(1) as pool:
        assert pool.verify(TASK, ROTATE).verified
    jobs = [m for m in sent if m.get("op") == "verify"]
    assert len(jobs) == 1
    assert set(jobs[0]) == {"op", "source", "inputs", "call_timeout"}
    assert jobs[0]["inputs"] == [A, B, C, *TESTS]
    text = json.dumps(sent)
    for pair in TASK.train:
        assert json.dumps(pair.output) not in text


def test_programs_run_outside_the_parent_and_workers(open_verifier):
    source = "import os\ndef transform(grid):\n    return [[int(d) for d in str(os.getpid())]]\n"
    result = open_verifier.verify(TASK, source)
    pid = int("".join(str(d) for d in result.pairs[0].output[0]))
    assert pid != os.getpid()
    assert pid not in open_verifier.worker_pids()


def test_programs_do_not_share_state():
    with PoolVerifier(1) as pool:
        pool.verify(TASK, "np.SENTINEL = 1\nlib.SENTINEL = 1\n" + ROTATE)
        source = (
            "def transform(grid):\n"
            "    seen = hasattr(np, 'SENTINEL') or hasattr(lib, 'SENTINEL')\n"
            "    return [[1]] if seen else [[2]]\n"
        )
        result = pool.verify(TASK, source)
        assert result.pairs[0].output == [[2]]


CONTAINED = [
    ("open('/etc/hostname').read()", "PermissionError: sandbox: open"),
    ("open('out.txt', 'w')", "PermissionError: sandbox: open"),
    ("os.fork()", "PermissionError: sandbox: os.fork"),
    ("[os.fork() for _ in range(100)]", "PermissionError: sandbox: os.fork"),
    ("os.system('true')", "PermissionError: sandbox: os.system"),
    ("os.kill(os.getppid(), 9)", "PermissionError: sandbox: os.kill"),
    ("__import__('socket').socket()", "PermissionError: sandbox: socket"),
    ("__import__('subprocess').run(['true'])", "PermissionError: sandbox: subprocess.Popen"),
    ("os.listdir('/')", "PermissionError: sandbox: os.listdir"),
]


@pytest.mark.parametrize(("expression", "error"), CONTAINED)
def test_containment_of_program_children(open_verifier, expression, error):
    source = f"import os\ndef transform(grid):\n    {expression}\n    return grid\n"
    result = open_verifier.verify(TASK, source)
    assert result.pairs[0].error.startswith(error), result.pairs[0]
    assert children_of(set(open_verifier.worker_pids())) == []


@pytest.mark.parametrize("payload", ["b'garbage\\n'", "b'x' * 300_000 + b'\\n'", "b'[1]\\n'"])
def test_garbage_on_the_result_pipe_is_contained(open_verifier, payload):
    # The program finds its result pipe by brute force and writes junk instead of a reply.
    source = (
        "import os\n"
        "def transform(grid):\n"
        "    if grid[0, 0] == 7:\n"
        "        for fd in range(3, 64):\n"
        "            try:\n"
        f"                os.write(fd, {payload})\n"
        "            except OSError:\n"
        "                pass\n"
        "        while True:\n"
        "            pass\n"
        "    return np.rot90(grid, -1)\n"
    )
    result = open_verifier.verify(TASK, source)
    assert [p.error for p in result.pairs] == [None, "invalid result", None]
    assert result.test_outputs == (rot90(TESTS[0]), rot90(TESTS[1]))


def test_a_crashing_program_spares_the_other_inputs(open_verifier):
    before = open_verifier.replaced
    source = (
        "import os\n"
        "def transform(grid):\n"
        "    if grid[0, 0] == 7:\n"
        "        os.abort()\n"
        "    return np.rot90(grid, -1)\n"
    )
    result = open_verifier.verify(TASK, source)
    assert [p.error for p in result.pairs] == [None, "crashed: signal SIGABRT", None]
    assert result.test_outputs == (rot90(TESTS[0]), rot90(TESTS[1]))
    assert open_verifier.replaced == before


def test_preloaded_modules_reach_program_children():
    with PoolVerifier(1, policy=False, preload=("colorsys", "arc26_no_such_module")) as pool:
        source = (
            "import sys\n"
            "def transform(grid):\n"
            "    return [[1]] if 'colorsys' in sys.modules else [[2]]\n"
        )
        assert pool.verify(TASK, source).pairs[0].output == [[1]]


def test_scipy_programs_verify_where_installed(verifier):
    # Skipped where scipy is absent (the dev machine, CI); on Kaggle the workers preload it.
    pytest.importorskip("scipy.ndimage")
    source = (
        "import scipy.ndimage as ndi\n"
        "def transform(grid):\n"
        "    labels, count = ndi.label(grid > 0)\n"
        "    return np.rot90(grid, -1) if count >= 1 else grid\n"
    )
    assert verifier.verify(TASK, source).verified


# Pool recovery.


def test_a_killed_worker_is_replaced():
    with PoolVerifier(2) as pool:
        pool.verify(TASK, ROTATE)
        victim = pool.worker_pids()[0]
        os.kill(victim, signal.SIGKILL)
        time.sleep(0.2)
        assert all(pool.verify(TASK, ROTATE).verified for _ in range(6))
        assert pool.replaced >= 1
        pids = pool.worker_pids()
        assert len(pids) == 2 and victim not in pids


def test_a_hung_worker_is_replaced():
    with PoolVerifier(1, call_timeout_s=0.2, hang_grace_s=0.5) as pool:
        pool.verify(TASK, ROTATE)
        (victim,) = pool.worker_pids()
        os.kill(victim, signal.SIGSTOP)
        start = time.monotonic()
        result = pool.verify(TASK, ROTATE)
        assert time.monotonic() - start < 10
        assert all(p.error == "worker failure: hung" for p in result.pairs)
        assert pool.replaced == 1
        assert not alive(victim)
        assert pool.verify(TASK, ROTATE).verified


# Concurrency (the agent runner calls verify from up to 64 threads at once).


def test_concurrent_mixed_verifications_are_correct_and_leak_nothing():
    kinds = ["correct", "wrong", "hang"] * 16
    sources = {"correct": ROTATE, "wrong": IDENTITY, "hang": HANG}
    with PoolVerifier(8, call_timeout_s=0.3) as pool:
        pids = set(pool.worker_pids())
        with ThreadPoolExecutor(32) as threads:
            results = list(threads.map(lambda k: pool.verify(TASK, sources[k]), kinds))
        for kind, result in zip(kinds, results, strict=True):
            if kind == "correct":
                assert result.verified and result.passed == 3
                assert result.test_outputs == (rot90(TESTS[0]), rot90(TESTS[1]))
            elif kind == "wrong":
                assert result.passed == 0 and result.pairs[1].cell_accuracy == 0.5
                assert result.test_outputs == TESTS
            else:
                assert all(p.error == "timeout" for p in result.pairs)
                assert result.test_outputs == (None, None)
        assert set(pool.worker_pids()) == pids
        assert pool.replaced == 0
        assert all(alive(pid) for pid in pids)
        assert children_of(pids) == []
    assert not any(alive(pid) for pid in pids)


def test_close_releases_waiting_callers_and_kills_workers():
    pool = PoolVerifier(1, call_timeout_s=0.5)
    pids = pool.worker_pids()
    errors = []

    def call():
        try:
            pool.verify(TASK, HANG)
        except SandboxError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(3)]
    for thread in threads:
        thread.start()
    time.sleep(0.3)
    pool.close()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert len(errors) == 3
    assert pool.worker_pids() == []
    assert not any(alive(pid) for pid in pids)
    with pytest.raises(SandboxError):
        pool.verify(TASK, ROTATE)
    pool.close()


def test_throughput_of_trivial_programs():
    with PoolVerifier(8) as pool:
        with ThreadPoolExecutor(8) as threads:
            list(threads.map(lambda _: pool.verify(TASK, ROTATE), range(16)))
            start = time.monotonic()
            results = list(threads.map(lambda _: pool.verify(TASK, ROTATE), range(200)))
            rate = len(results) / (time.monotonic() - start)
    assert all(r.verified for r in results)
    print(f"verifier throughput, 8 workers, trivial program: {rate:.0f} per second")
    # The brief's target is 50 per second; the margin absorbs slow CI machines.
    assert rate >= 50
