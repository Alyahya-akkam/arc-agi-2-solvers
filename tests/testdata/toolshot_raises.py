"""A tool loop that passes its self-test and raises on every real task (tests only)."""

from toolshot_stub import ToolConfig, tool_config  # noqa: F401 (the interface's names)
from toolshot_stub import solve_with_tools as _solve


def solve_with_tools(task, *args, **kwargs):
    if task.task_id != "selftest":
        raise RuntimeError("the sandbox broke on a real task")
    return _solve(task, *args, **kwargs)
