"""Replay of the program slot over versions 16 and 17's real tool calls (task P5.8b).

Each run's calls are parsed from its pulled ``transcripts.jsonl`` and paired with the call rows
of ``attempts.jsonl``, as ``docs/experiments/P5.2-01-r21.md``'s replay did; the candidates are
verified by the real :class:`arc26.agent.verify.PoolVerifier` on the demonstrations only (the
public evaluation challenges, never the solutions). The run logs and the challenges stay outside
the repository, so the test skips where they are missing; ``ARC26_R20_LOGS``, ``ARC26_R21_LOGS``
and ``ARC26_EVAL_CHALLENGES`` override the paths.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from arc26.agent import toolshot
from arc26.agent.types import AgentTask, Pair
from arc26.agent.verify import PoolVerifier

HOME = Path.home()
RUNS = {
    "r20": Path(os.environ.get(
        "ARC26_R20_LOGS", HOME / "wt-P5.5/runs/P5.2-01-r20/run_logs/fn")),
    "r21": Path(os.environ.get(
        "ARC26_R21_LOGS", HOME / "wt-P5.9/runs/P5.2-01-r21/run_logs/fn")),
}
CHALLENGES = Path(os.environ.get(
    "ARC26_EVAL_CHALLENGES", "/tmp/harness/pivotools/data/arc-agi_evaluation_challenges.json"))
# The slots P5.8 (98343c2) fills on each run's calls, and the ones the retry without top-level
# assignments adds: every one a load NameError of a kept assignment. Version 17's 6e453dd6
# stays empty: its passing rule is a closure, check(make_transform(mode)), that no top-level
# statement binds, so no joined program holds it.
BEFORE = {
    "r20": {"1818057f", "247ef758", "7c66cb00", "7ed72f31", "88e364bc", "97d7923e", "a395ee82",
            "b0039139", "b5ca7ac4"},
    "r21": {"1818057f", "247ef758", "88e364bc", "b0039139"},
}
FIXED = {"r20": {"6e453dd6"}, "r21": {"97d7923e", "7c66cb00", "a395ee82", "b5ca7ac4"}}


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _inputs(logs: Path, task_id: str, text: str, rows: list[dict[str, Any]]) -> tuple[
        list[str], list[int], list[tuple[int, str]]]:
    parsed = []
    for piece in text.split("<tool_call>")[1:]:
        body = piece.split("</tool_call>")[0]
        closed = "</tool_call>" if "</tool_call>" in piece else ""
        parsed.append(toolshot.parse_call("<tool_call>" + body + closed))
    codes: list[str] = []
    defining: list[int] = []
    checked: list[tuple[int, str]] = []
    for found, row in zip(parsed, rows, strict=False):
        if not row.get("ran"):
            continue
        codes.append(found.arguments.get("code", "") if found else "")
        if row.get("defines_transform"):
            defining.append(len(codes) - 1)
        for name, passed, total in row.get("checked") or ():
            if total > 0 and passed == total:
                checked.append((len(codes) - 1, name))
    return codes, defining, checked


def _replay(run: str, verifier: PoolVerifier, retry: bool,
            monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any] | None]:
    logs = RUNS[run]
    challenges = json.loads(CHALLENGES.read_text())
    texts = {r["task_id"]: r["text"] for r in _jsonl(logs / "transcripts.jsonl")}
    out: dict[str, dict[str, Any] | None] = {}
    with monkeypatch.context() as patch:
        if not retry:
            patch.setattr(toolshot, "_load_name_error", lambda result: False)
        for attempt in _jsonl(logs / "attempts.jsonl"):
            task_id = attempt["task_id"]
            spec = challenges[task_id]
            task = AgentTask(task_id, tuple(Pair(p["input"], p["output"]) for p in spec["train"]),
                             tuple(t["input"] for t in spec["test"]))
            codes, defining, checked = _inputs(logs, task_id, texts[task_id], attempt["calls"])
            errors: list[str] = []
            rows: list[dict[str, Any]] = []
            program = toolshot._verify(task, codes, defining, checked, toolshot.ToolConfig(),
                                       verifier, errors, rows, float("inf"), lambda: 0.0)
            assert not [e for e in errors if "no time" in e]
            for row in rows:
                if not row["verified"]:
                    assert row["passed"] < row["demos"] or row["fatal"]
            if program is not None:
                assert program["demos_passed"] == program["demos"] == len(task.train)
            out[task_id] = program
    return out


@pytest.mark.parametrize("run", ["r20", "r21"])
def test_replay_fills_the_load_fault_tasks(run: str, monkeypatch: pytest.MonkeyPatch) -> None:
    logs = RUNS[run]
    if not (logs / "attempts.jsonl").exists() or not CHALLENGES.exists():
        pytest.skip(f"{run}'s run logs or the evaluation challenges are not on this machine")
    verifier = PoolVerifier(1, call_timeout_s=2.0, memory_bytes=1 << 30)
    try:
        before = _replay(run, verifier, False, monkeypatch)
        after = _replay(run, verifier, True, monkeypatch)
    finally:
        verifier.close()
    filled_before = {t for t, p in before.items() if p is not None}
    filled_after = {t for t, p in after.items() if p is not None}
    print(f"{run}: {len(before)} tasks, filled before {sorted(filled_before)}, "
          f"after {sorted(filled_after)}")
    assert filled_before == BEFORE[run]
    assert filled_after == BEFORE[run] | FIXED[run]
    assert all(after[t]["without_assignments"] for t in FIXED[run])  # type: ignore[index]
