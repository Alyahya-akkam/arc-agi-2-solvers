"""A stand-in for ``arc26.agent.toolshot`` (task P5.3) with its interface, for tests only.

The smokes run the real module; ``toolshot_raises.py``, a loop that fails on purpose, builds on
this one through ``tool.module`` and ``tool.extra_path``. It runs one call loop in the simplest
way: the code of each ``<tool_call>`` is executed in one in-process namespace (no sandbox),
``check(fn)`` reports per demonstration, and the answers are the final turn's blocks with a
passing ``transform``'s output in the second slot.
"""

from __future__ import annotations

import contextlib
import io
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

CALL = re.compile(r"<tool_call>\s*<function=python>\s*<parameter=code>\n?(.*?)</parameter>",
                  re.DOTALL)
TOOL_TURN = ("<|im_end|>\n<|im_start|>user\n<tool_response>\n{reply}\n</tool_response><|im_end|>"
             "\n<|im_start|>assistant\n<think>\n")


@dataclass(frozen=True)
class ToolConfig:
    """The loop's budgets (the P5.3 defaults)."""

    cap: int = 65536
    context_tokens: int = 86016
    max_calls: int = 40
    reply_tokens: int = 16384
    forcing_tokens: int = 6144
    reply_chars: int = 2000
    call_timeout_s: float = 10.0
    verify_call_timeout_s: float = 2.0
    thinking_end_note: str = ("\n\nI have used my whole thinking budget, so I stop reasoning here "
                              "and write my two final answer blocks for each test input now, best "
                              "first.\n")


@dataclass
class ToolOutcome:
    """What one task's loop produced (the P5.3 interface's fields)."""

    text: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    direct: list[list[Any]] = field(default_factory=list)
    program: dict[str, Any] | None = None
    attempts: list[list[Any]] = field(default_factory=list)
    forced: bool = False
    completion_tokens: int = 0
    tool_reply_tokens: int = 0
    ended_by: str = ""
    errors: list[str] = field(default_factory=list)


def tool_config(mapping: Mapping[str, Any]) -> ToolConfig:
    """Return the config from the YAML ``tool:`` block (unknown keys ignored)."""
    names = {f.name for f in fields(ToolConfig)}
    return ToolConfig(**{k: v for k, v in mapping.items() if k in names})


def _check(task: Any) -> Callable[[Callable[[Any], Any]], str]:
    def check(fn: Callable[[Any], Any]) -> str:
        lines = []
        for i, pair in enumerate(task.train):
            try:
                ok = [list(map(int, r)) for r in fn([list(r) for r in pair.input])] == pair.output
            except Exception as err:  # the model's code may raise anything
                ok = False
                lines.append(f"demo {i}: error {type(err).__name__}")
                continue
            lines.append(f"demo {i}: {'pass' if ok else 'fail'}")
        return "\n".join(lines)

    return check


def solve_with_tools(task: Any, tok: Any, generate: Callable[..., Any], cfg: ToolConfig, *,
                     force_at: float, end: float, now: Callable[[], float] = time.time,
                     verifier: Any = None) -> ToolOutcome:
    """Run the call loop on one task (the interface's signature)."""
    from arc26.agent.prompts import DIRECT_TWO_SYSTEM_PROMPT, task_text

    messages = [{"role": "system", "content": DIRECT_TWO_SYSTEM_PROMPT},
                {"role": "user", "content": task_text(task)}]
    ids = tok.encode(tok.render(messages, tools=[{"name": "python"}], reasoning_effort="xhigh"))
    out = ToolOutcome()
    namespace: dict[str, Any] = {"train": task.train, "test_inputs": task.test_inputs,
                                 "check": _check(task)}
    verified = None
    kind = "think"
    while True:
        budget = cfg.cap - out.completion_tokens
        res = generate(ids, budget, force_at, kind, {"stop": ["</tool_call>"],
                                                      "no_stop_trim": True})
        out.turns.append({"kind": kind, **res.row()})
        out.completion_tokens += res.completion_tokens
        out.text += res.text
        ids = ids + (list(res.ids) or tok.encode(res.text))
        match = CALL.search(res.text)
        if res.finish == "stop" and match and len(out.calls) < cfg.max_calls:
            buffer = io.StringIO()
            status = "ok"
            try:
                with contextlib.redirect_stdout(buffer):
                    exec(match.group(1), namespace)
            except Exception as err:  # the model's code may raise anything
                status = "error"
                buffer.write(f"{type(err).__name__}: {err}")
            reply = buffer.getvalue()[: cfg.reply_chars]
            if "transform" in namespace and "fail" not in reply and "error" not in reply:
                verified = namespace["transform"]
            turn = tok.encode(TOOL_TURN.format(reply=reply))
            out.tool_reply_tokens += len(turn)
            out.calls.append({"chars": len(match.group(1)), "status": status,
                              "reply_tokens": len(turn)})
            out.text += TOOL_TURN.format(reply=reply)
            ids = ids + turn
            kind = "tool"
            continue
        if res.finish != "stop" and now() < end:
            suffix = cfg.thinking_end_note + "</think>\n\n"
            ids = ids + tok.encode(suffix)
            res = generate(ids, cfg.forcing_tokens, end, "force", {})
            out.turns.append({"kind": "force", **res.row()})
            out.completion_tokens += res.completion_tokens
            out.text += suffix + res.text
            out.forced = True
        break
    from arc26.agent.loop import fenced_blocks, parse_answer

    visible = out.text.rsplit("</think>", 1)[-1].replace("<|im_end|>", "\n")
    direct: list[list[Any]] = [[] for _ in task.test_inputs]
    for tag, body in fenced_blocks(visible):
        if tag == "answer":
            try:
                index, grid = parse_answer(body, len(task.test_inputs), lenient=True)
                direct[index].append(grid)
            except ValueError as err:
                out.errors.append(str(err))
    program_outputs = [None] * len(task.test_inputs)
    if verified is not None:
        program_outputs = [verified([list(r) for r in g]) for g in task.test_inputs]
        out.program = {"demos_passed": len(task.train), "test_outputs": program_outputs}
    for j in range(len(task.test_inputs)):
        slots: list[Any] = []
        a1 = direct[j][:1]
        a2 = direct[j][1:2]
        for grid in [*a1, program_outputs[j], *a2]:
            if grid is not None and grid not in slots:
                slots.append(grid)
        out.attempts.append(slots[:2])
        out.direct.append(direct[j][:2])
    out.ended_by = "force" if out.forced else "stop"
    return out
