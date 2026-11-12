"""The ``check(fn)`` helper of the "think with Python" tool loop (task P5.3).

Why: the model tests a candidate rule on the demonstrations far more often than it writes any
other code, so one call should say per demonstration whether ``fn`` reproduces it and, for a
same-shape miss, which cells differ. The cell form is Poetiq's ``_array_diff``
(``arc_agi/solve_coding.py``, MIT): every cell as its value, a wrong cell as ``yours/expected``.

Where it lives: :data:`CHECK_SOURCE` is plain Python that :mod:`arc26.agent.toolshot` runs as the
first cell of the task's :class:`arc26.agent.sandbox.SubprocessRepl` (and again after every
restart), so ``check`` sits in the REPL namespace beside ``train``, ``test_inputs``, ``np`` and
``lib``. It runs under the sandbox's policy like any model cell; the sandbox itself is unchanged.

Each call also names the function it checked (its ``__name__``) on the summary line and appends
``{"name", "passed", "total"}`` to the REPL's ``check_log`` list (task P5.8): the harness reads
the summary lines (:func:`check_records`) and takes the latest fully passing function for the
program slot whatever its name, since the model does not always call its rule ``transform``.
"""

from __future__ import annotations

import re

# The REPL's prelude cell. It must pass guards.check_policy (no dunders but __name__, no
# forbidden builtins) and print nothing when defined.
CHECK_SOURCE = '''\
check_log = []


def check(fn):
    """Run fn on every training input; print pass or fail per demonstration."""
    try:
        name = str(fn.__name__)
    except Exception:
        name = "?"
    passed = 0
    for k, (grid_in, want) in enumerate(train):
        try:
            got = np.asarray(fn(grid_in.copy()))
        except Exception as err:
            print(f"demo {k}: fail, raised {type(err).__name__}: {err}")
            continue
        if got.ndim != 2:
            print(f"demo {k}: fail, returned shape {got.shape}, not a 2-D grid")
            continue
        if got.shape != want.shape:
            print(f"demo {k}: fail, shape {got.shape[0]}x{got.shape[1]}, "
                  f"expected {want.shape[0]}x{want.shape[1]}")
            continue
        yours = got.tolist()
        expected = want.tolist()
        wrong = sum(1 for r in range(len(expected)) for c in range(len(expected[0]))
                    if yours[r][c] != expected[r][c])
        if wrong == 0:
            passed += 1
            print(f"demo {k}: pass")
            continue
        print(f"demo {k}: fail, {wrong} of {want.size} cells differ (yours/expected):")
        for r in range(len(expected)):
            print(" ".join(str(yours[r][c]) if yours[r][c] == expected[r][c]
                           else f"{yours[r][c]}/{expected[r][c]}"
                           for c in range(len(expected[0]))))
    print(f"check: {passed} of {len(train)} demonstrations pass for {name}")
    check_log.append({"name": name, "passed": passed, "total": len(train)})
'''

SUMMARY = re.compile(r"^check: (\d+) of (\d+) demonstrations pass(?: for (\S+))?$", re.MULTILINE)


def check_summaries(stdout: str) -> list[list[int]]:
    """Return the ``check`` summaries a cell printed.

    Args:
        stdout: The cell's captured output.

    Returns:
        ``[passed, total]`` per ``check`` call, in order (empty when the cell called none).
    """
    return [[int(a), int(b)] for a, b, _ in SUMMARY.findall(stdout)]


def check_records(stdout: str) -> list[tuple[str, int, int]]:
    """Return each ``check`` call's function name and result from a cell's output.

    Args:
        stdout: The cell's captured output.

    Returns:
        ``(name, passed, total)`` per ``check`` call, in order; ``name`` is the checked
        function's ``__name__`` (``"?"`` when it has none, empty for a summary line without one).
    """
    return [(name, int(a), int(b)) for a, b, name in SUMMARY.findall(stdout)]
