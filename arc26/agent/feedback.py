"""Executor feedback for the agent loop: grids as text and NVIDIA-style program reports (P6.2).

Why: the model improves a program from what the executor tells it, and Pang found that showing
the failed examples and the output differences "significantly improves performance"
(``docs/research/agent_library_design.md``, section 2.1). The report follows the executor of
NVIDIA's ARC traces: "k/n training examples passed", then per example the exact match, the shape,
the cell accuracy and a compact diff of the first mismatching rows, then the program's outputs on
the test inputs. Every text here is capped in length, so a runaway program cannot flood the
model's context.

Indices are 0-based everywhere, as ``train[i]`` and ``test_inputs[j]`` are in the REPL.
"""

from __future__ import annotations

from collections.abc import Sequence

from arc26.agent.types import AgentTask, CellResult, GuardFinding, PairResult, Verification
from arc26.data import Grid

# Rows of the diff shown per failing example; more rarely helps and costs context.
DIFF_ROWS = 3
# Default caps in characters (one token is at least one character, so these bound tokens too).
REPORT_CHARS = 3000
CELL_CHARS = 1500

# The cap on a forcing message's report of the cut reply's cells: a reply cut while reasoning has
# none, and the forcing turn's room is reserved before the cut reply runs (``forcing_chars``).
FORCING_REPORT_CHARS = 1500
# Revision 6: one of revision 5's three forced replies explored instead ("Let me analyze the
# examples more carefully first." and a block of prints), so the note asks for the program alone.
FORCING_NOTE = (
    "Your reasoning reached the limit of {limit} tokens and was cut off. Stop reasoning now. "
    "Reply with exactly one ```python block that defines transform(grid), your best program from "
    "the analysis so far, and nothing else: no exploration, no prints, no tests, no prose. It is "
    "checked on every training example."
)

# The answer-forcing note (run P6.3-04's direct-answer arm): after a cut reply, the test outputs
# alone, as the direct-answer system prompt's answer blocks.
ANSWER_FORCING_NOTE = (
    "Your reasoning reached the limit of {limit} tokens and was cut off. Stop reasoning now. "
    "Reply with exactly one ```answer block per test input, each holding "
    '{{"test_index": j, "grid": [[...], ...]}}, your best answer from the analysis so far, and '
    "nothing else: no code, no prose."
)
# Run P6.3-05: one or two answers per test input (pass@2 from one reply), best first.
ANSWER2_FORCING_NOTE = (
    "Your reasoning reached the limit of {limit} tokens and was cut off. Stop reasoning now. "
    "Reply with one or two ```answer blocks per test input, your best answer first, each holding "
    '{{"test_index": j, "grid": [[...], ...]}}, from the analysis so far, and nothing else: no '
    "code, no prose."
)
FORCING_NOTES = {
    "program": FORCING_NOTE, "answer": ANSWER_FORCING_NOTE, "answer2": ANSWER2_FORCING_NOTE,
}  # fmt: skip

NO_CODE_NOTE = (
    "No code cell found. Reply with a ```python block; a block that defines transform(grid) is "
    "checked on every training example."
)


def grid_text(grid: Grid) -> str:
    """Return a grid as rows of digits, one row per line.

    Why: Qwen's tokenizer splits digits one by one, so a row of ``w`` cells costs ``w`` tokens
    plus a newline; separators would double that.

    Args:
        grid: The grid.

    Returns:
        The rows joined by newlines, for example two lines ``012`` and ``120``.
    """
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def shape_text(grid: Grid) -> str:
    """Return a grid's size as ``"<rows>x<columns>"``.

    Args:
        grid: The grid.

    Returns:
        For example ``"3x4"``.
    """
    return f"{len(grid)}x{len(grid[0]) if grid else 0}"


def clip(text: str, limit: int) -> str:
    """Shorten a text to at most ``limit`` characters, keeping its head and its tail.

    Why: the start of an output shows what a program printed and the end shows where it failed,
    so both are kept around a marker that says how much was cut.

    Args:
        text: The text.
        limit: The maximum length, at least 1.

    Returns:
        ``text`` itself when it fits, else two thirds of the room from its head, a marker
        ``[... n characters cut ...]``, and the rest from its tail.

    Raises:
        ValueError: If ``limit`` is below 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1; got {limit}")
    if len(text) <= limit:
        return text
    reserve = len(f"\n[... {len(text)} characters cut ...]\n")
    if limit <= reserve:
        return text[:limit]
    keep = limit - reserve
    head = keep * 2 // 3
    tail = keep - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n[... {cut} characters cut ...]\n{text[len(text) - tail :]}"


def mismatch_rows(got: Grid, want: Grid, limit: int = DIFF_ROWS) -> list[str]:
    """Return the first rows where two grids of equal shape differ, side by side.

    Args:
        got: The program's output.
        want: The expected output, of the same shape.
        limit: The number of rows to show.

    Returns:
        Lines like ``"row 2: 0120 | 0110"`` (yours, then expected).
    """
    lines = []
    for r, (row_got, row_want) in enumerate(zip(got, want, strict=False)):
        if row_got != row_want:
            lines.append(f"row {r}: {''.join(map(str, row_got))} | {''.join(map(str, row_want))}")
            if len(lines) == limit:
                break
    return lines


def _guard_line(finding: GuardFinding) -> str:
    effect = "rejects the program" if finding.fatal else "ranks it lower"
    return f"{finding.rule} ({effect}): {finding.detail}"


def _pair_lines(pair: PairResult, want: Grid, diff_rows: int) -> list[str]:
    if pair.exact:
        return [f"Example {pair.index}: exact_match=True"]
    if pair.output is None:
        return [f"Example {pair.index}: exact_match=False, error: {pair.error or 'no valid grid'}"]
    if not pair.shape_match:
        lines = [
            f"Example {pair.index}: exact_match=False, shape {shape_text(pair.output)} "
            f"(expected {shape_text(want)}), cell_accuracy=0.00",
            "  your output starts:",
        ]
        lines.extend(f"  {row}" for row in grid_text(pair.output[:diff_rows]).splitlines())
        return lines
    lines = [
        f"Example {pair.index}: exact_match=False, shape {shape_text(pair.output)} (as expected), "
        f"cell_accuracy={pair.cell_accuracy:.2f}",
        "  first mismatching rows (yours | expected):",
    ]
    lines.extend(f"  {line}" for line in mismatch_rows(pair.output, want, diff_rows))
    return lines


def verification_report(
    task: AgentTask,
    verification: Verification,
    *,
    max_chars: int = REPORT_CHARS,
    diff_rows: int = DIFF_ROWS,
) -> str:
    """Describe the verifier's judgement of one program, in NVIDIA's executor style.

    Args:
        task: The task the program was verified on (for the expected outputs).
        verification: The verifier's result.
        max_chars: The cap on the report's length.
        diff_rows: Mismatching rows shown per failing example.

    Returns:
        The report: the passed count, one entry per example (identical errors on every example
        are reported once), the test outputs, the guard findings, and a verdict line.
    """
    n = len(verification.pairs)
    lines = [f"Program check: {verification.passed}/{n} training examples passed."]
    errors = {p.error for p in verification.pairs if p.output is None}
    if n > 1 and all(p.output is None for p in verification.pairs) and len(errors) == 1:
        lines.append(f"All examples: error: {errors.pop() or 'no valid grid'}")
    else:
        for pair in verification.pairs:
            want = task.train[pair.index].output if 0 <= pair.index < len(task.train) else []
            lines.extend(_pair_lines(pair, want, diff_rows))
    failed_tests = []
    for j, output in enumerate(verification.test_outputs):
        if output is None:
            failed_tests.append(j)
            lines.append(f"Test input {j}: no valid output (the call raised or returned no grid)")
        else:
            lines.append(f"Test input {j}: output {shape_text(output)}")
            lines.append(grid_text(output))
    if verification.guards:
        lines.append("Guards:")
        lines.extend(f"  {_guard_line(g)}" for g in verification.guards)
    if verification.verified:
        verdict = "accepted: it reproduces every training example and runs on every test input."
    elif any(g.fatal for g in verification.guards):
        verdict = "rejected by a guard; write a general rule, not stored grids or sizes."
    elif verification.passed == n and failed_tests:
        verdict = f"not accepted: no valid output on test input(s) {failed_tests}."
    else:
        verdict = "not accepted."
    lines.append(f"Verdict: {verdict}")
    return clip("\n".join(lines), max_chars)


def cell_report(
    index: int, result: CellResult, *, timeout_s: float, max_chars: int = CELL_CHARS
) -> str:
    """Describe what one REPL cell printed and how it ended.

    Args:
        index: The cell's position in the reply, from 0.
        result: The REPL's result.
        timeout_s: The cell's time limit, named when the cell was stopped.
        max_chars: The cap on the printed output shown.

    Returns:
        The cell's output (clipped), its error if any, and a note when the REPL was restarted.
    """
    lines = []
    output = result.stdout.rstrip("\n")
    if output:
        lines.append(f"[cell {index}] output:\n{clip(output, max_chars)}")
    if result.error == "timeout":
        lines.append(f"[cell {index}] error: timeout (stopped after {timeout_s:g} s)")
    elif result.error == "memory":
        lines.append(f"[cell {index}] error: memory limit exceeded")
    elif result.error is not None:
        lines.append(f"[cell {index}] error: {clip(result.error, max_chars)}")
    if result.restarted:
        lines.append(
            f"[cell {index}] the REPL was restarted: earlier variables are lost; train, "
            "test_inputs, np and lib are loaded again."
        )
    if not lines:
        lines.append(f"[cell {index}] ran without output.")
    return "\n".join(lines)


def truncated_note(max_tokens: int) -> str:
    """Return the note for a reply cut off at the token limit.

    Args:
        max_tokens: The reply's token limit.

    Returns:
        The note; it asks for a shorter reply.
    """
    return (
        f"Your reply hit the limit of {max_tokens} tokens and was cut off; an unfinished code "
        "block is not run. Think more briefly and send the code sooner."
    )


def forcing_message(sections: Sequence[str], *, limit: int, kind: str = "program") -> str:
    """Return the user message of a forcing turn, sent after a reply cut at its cap.

    Why: run ``P6.3-02`` (revision 4) ended 156 of its 201 engine calls at the turn cap while the
    model was still reasoning, so most attempts never wrote a program. The forcing turn asks for
    the program at once, with the cut reasoning kept in the context and thinking off for that one
    call (``docs/runbooks/agent_spike.md``).

    Args:
        sections: What the cut reply's complete code blocks printed and how they were judged
            (usually none: a reply cut while reasoning has no visible text).
        limit: The cut reply's token cap, named in the note.
        kind: ``"program"`` (:data:`FORCING_NOTE`) or ``"answer"``
            (:data:`ANSWER_FORCING_NOTE`, the direct-answer arm of run ``P6.3-04``).

    Returns:
        The sections clipped to :data:`FORCING_REPORT_CHARS`, then the kind's note; at most
        :func:`forcing_chars` of the kind characters for a limit of up to 12 digits.
    """
    note = FORCING_NOTES[kind].format(limit=limit)
    body = clip("\n\n".join(sections), FORCING_REPORT_CHARS) if sections else ""
    return f"{body}\n\n{note}" if body else note


def forcing_chars(kind: str = "program") -> int:
    """Return an upper bound on the length of :func:`forcing_message`'s text, in characters.

    Why: the loop reserves the forcing turn's context before the reply that may be cut, so it
    needs the message's size in advance; one token per character bounds its tokens.

    Args:
        kind: The forcing note's kind, as in :func:`forcing_message`.

    Returns:
        The report's cap, two separator characters, and the note with a 12-digit limit.
    """
    return FORCING_REPORT_CHARS + 2 + len(FORCING_NOTES[kind].format(limit=10**11))


def compose_feedback(
    sections: Sequence[str],
    *,
    next_round: int,
    max_rounds: int,
    tokens_left: int,
    max_chars: int,
) -> str:
    """Assemble the user message that follows a reply.

    Args:
        sections: Cell outputs, program reports and notes, in order.
        next_round: The number of the coming reply, from 1.
        max_rounds: The attempt's reply limit.
        tokens_left: Generated tokens left in the attempt.
        max_chars: The cap on the sections' total length.

    Returns:
        A header with the round counter and the tokens left (as NVIDIA's "Round 2/5"), then the
        sections, clipped together to ``max_chars``.
    """
    header = f"Round {next_round}/{max_rounds}. About {tokens_left} tokens left for this task."
    body = clip("\n\n".join(sections), max_chars) if sections else ""
    return f"{header}\n\n{body}" if body else header
