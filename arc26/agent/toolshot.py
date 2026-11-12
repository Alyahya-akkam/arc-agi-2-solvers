r"""Think with Python: one long-thinking shot with a ``python`` tool (task P5.3, version 11).

Why: the model may run Python on the task's grids in the middle of its reasoning, see whether its
code reproduces the demonstrations, and fix it; a program that reproduces every demonstration
takes one of the two answer slots, and the model's own first answer always keeps the other
(``docs/research/flash_next_harness.md``, section "Recommendation", on ``task/ideas``).

The loop per task (:func:`solve_with_tools`):

1. The prompt is :data:`arc26.agent.prompts.DIRECT_TWO_SYSTEM_PROMPT` with its "There is no code
   interpreter" sentence replaced by :data:`TOOL_PARAGRAPH` (:func:`system_text`), rendered by
   the injected tokenizer with ``tools=[PYTHON_TOOL]`` and ``reasoning_effort``; thinking stays
   on (the template's default).
2. Each request goes to the injected ``generate`` as token ids with ``stop=["</tool_call>"]`` and
   ``no_stop_trim``. A reply ending in a call (also one inside the reasoning, or one whose closing
   tags are missing when the turn ends at ``<|im_end|>``) is parsed, run in the task's
   :class:`arc26.agent.sandbox.SubprocessRepl`, and the next request is the previous ids, the
   generated ids, and the ids of :data:`TOOL_TURN` with the reply, so the server's radix cache
   serves everything but the reply.
3. Budgets: :attr:`ToolConfig.cap` generated tokens over all turns, the forced answer included
   (thinking and tool turns stop :attr:`ToolConfig.forcing_tokens` short of it), except a second
   forcing request (``force2``), which may add :attr:`ToolConfig.forcing_tokens` more; every request
   bounded by :attr:`ToolConfig.context_tokens` with room held for the forcing turn; at most
   :attr:`ToolConfig.max_calls` calls and :attr:`ToolConfig.reply_tokens` tool-turn tokens, after
   which each call is answered with :attr:`ToolConfig.budget_message`. A reply cut at the cap, the
   context bound, or ``force_at`` gets the forcing turn (:func:`forcing_parts`): a cut call is
   closed and answered with the budget message; then the note and ``</think>`` unless the turn's
   answer blocks have begun, then :attr:`ToolConfig.forcing_tokens`. A call in the forced reply
   is never run. A forced reply with a call and no complete answer block is cut before the call
   (its ids kept, :func:`_strip_forced_call`) and continued once more with
   :attr:`ToolConfig.force_again_note` and :attr:`ToolConfig.forcing_tokens` (turn kind
   ``force2``, task P5.8).
4. Answers per test input: the first two distinct grids of :math:`(A_1, P, A_2)`, with
   :math:`A_1, A_2` from :func:`arc26.agent.shot.shot_answers` on the answer blocks of the last
   turn that has any (every refused block is an error, never repaired), and :math:`P` the test
   outputs of the first program :class:`arc26.agent.verify.PoolVerifier` (guards on) verifies,
   run as the task's code joined (:func:`joined_program`): the latest ``transform`` first, then
   the latest function of any name that passed every demonstration in ``check(fn)``
   (:func:`program_candidates`).

Template lines matched (P5.2-00's saved ``model_chat_template.jinja.txt`` and ``template.json``):

- Line 68, the tools system block: a call is ``<tool_call>\n<function=NAME>\n<parameter=P>\n
  value\n</parameter>\n</function>\n</tool_call>`` "with NO suffix"; lines 130, 137 to 140 and
  143 render an assistant call the same way, and line 146 closes the assistant turn with
  ``<|im_end|>\n``.
- Lines 148 to 157, a tool message after an assistant one: ``<|im_start|>user``, then
  ``\n<tool_response>\n``, the content trimmed (line 108, ``|trim``), ``\n</tool_response>``,
  and ``<|im_end|>\n`` as the last message.
- Lines 164 and 168, the generation prompt with thinking on: ``<|im_start|>assistant\n<think>\n``
  (``template.json``'s ``xhigh`` render ends the same way; ``<think>`` is one id, 248068).

Together these give :data:`TOOL_TURN`; rendering the saved template with jinja2 3.1.6 (an
assistant call, then a tool message, ``preserve_thinking``) gives the same text after
``</tool_call>``, character for character. The call parser follows the ARChitects'
``_recover_tool_calls_from_markup`` and ``_extract_leaked_tool_calls`` (``tool_agent.py``,
Apache 2.0): the qwen3_coder markup is read from the text wherever it sits, also without its
closing tags. ``check(fn)`` comes from :mod:`arc26.agent.toolcheck`.

The REPL cannot take the engine down (on Kaggle the SGLang server holds most of the RAM): each
REPL is a child process of :class:`arc26.agent.sandbox.SubprocessRepl` with ``RLIMIT_AS``
:attr:`ToolConfig.repl_memory_bytes` (1 GiB, enough for numpy, scipy, and 30 x 30 grids; an
allocation past it is a ``MemoryError`` in the cell, and the session goes on), ``RLIMIT_CPU``
:attr:`ToolConfig.repl_cpu_s` (600 s per process), ``RLIMIT_NPROC`` 0 and an audit hook that
denies fork, exec, and spawn (the hook also binds root, which ``RLIMIT_NPROC`` does not),
``RLIMIT_FSIZE`` 0 (no file data written) and ``RLIMIT_CORE`` 0, a wall timeout of
:attr:`ToolConfig.call_timeout_s` per call that kills its process group, and an empty working
and home directory from ``tempfile.mkdtemp`` (``/tmp`` on Kaggle, never ``/kaggle/working``),
removed at the end. A child that dies (timeout, crash, CPU cap) is replaced by a fresh one with
the grids and ``check`` loaded; the call's reply says so and the task goes on. The verifier's
children have the same memory (1 GiB), file, and process limits. A verification starts only
when its worst case (one call limit per input plus one, and the verifier's 10 s hang allowance)
ends before ``end``; the startup of a verifier this module creates is not counted (about a
second, at most its 60 s startup timeout), so that is the bound past ``end``.

No HTTP here: ``scripts/fn_probe.py`` builds ``generate`` around its ``/generate`` client, and
that client writes the timing record of every generation (the GPU op); this module records each
tool call as op ``tool_python``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from arc26 import timing
from arc26.agent import guards
from arc26.agent.feedback import clip
from arc26.agent.loop import fenced_blocks, parse_answer, program_source
from arc26.agent.prompts import DIRECT_TWO_SYSTEM_PROMPT, task_text
from arc26.agent.sandbox import SandboxError, SubprocessRepl
from arc26.agent.shot import implied_shapes, shot_answers
from arc26.agent.toolcheck import CHECK_SOURCE, check_records, check_summaries
from arc26.agent.types import AgentTask, Verification, Verifier
from arc26.data import Grid
from arc26.submit import grid_problem

THINK_CLOSE = "</think>"
IM_END = "<|im_end|>"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
TOOL_NAME = "python"
BUDGET_MESSAGE = "Tool budget used: write your answer blocks now."
# The template's own tool turn after a call (module docstring); {reply} is the trimmed reply.
TOOL_TURN = (
    "<|im_end|>\n<|im_start|>user\n<tool_response>\n{reply}\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
# The forcing note: P5.2-01's (configs/P5.2-01.yaml), reworded to ask for both answer blocks.
# Unlike P5.2-01, thinking and tool turns stop at cap less forcing_tokens, and the forced answer
# uses the rest of the cap.
THINKING_END_NOTE = (
    "\n\nI have used my whole thinking budget, so I stop reasoning here and write both of my "
    "answer blocks for each test input now, best first.\n"
)
# The second forcing request's text (task P5.8): it continues a forced reply whose tool call was
# stripped, in the model's voice after </think>, as the first note does; the loop puts it two
# newlines after the kept text.
FORCE_AGAIN_NOTE = (
    "The python tool is no longer available, so I write no more code. I write both of my "
    "answer blocks for each test input now, best first.\n\n"
)
# PoolVerifier's default hang_grace_s (verify.py): a silent worker is replaced after this.
VERIFY_HANG_GRACE_S = 10.0
# Closes a call cut before its end, longest form (the reserve counts it).
CALL_CLOSERS = "\n</parameter>\n</function>\n</tool_call>"
FORCING_SUFFIX = "</think>\n\n"

PYTHON_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Run Python code in this task's persistent session, which holds train, test_inputs, "
            "np, lib and check(fn). Returns what the code printed; the value of a last bare "
            "expression is printed too."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to run."}},
            "required": ["code"],
        },
    },
}

# DIRECT_TWO_SYSTEM_PROMPT's sentence that the tool arm replaces with TOOL_PARAGRAPH (versions 8
# to 10 keep the prompt itself unchanged).
NO_INTERPRETER = "There is no code interpreter: reason, then answer. "
TOOL_PARAGRAPH = (
    "You have a Python tool: while you think you may call the python function with code to run. "
    "It runs in one persistent Python session for this task that already holds train (a list of "
    "(input, output) pairs of 2-D numpy int arrays), test_inputs (a list of arrays), np, and "
    "check(fn), which runs fn on every training input and prints pass or fail per example, with "
    "yours/expected for the cells that differ when the size is right. Each call may run "
    "{timeout:g} seconds and its printed output is cut to {chars} characters. Name your rule "
    "transform(grid): if transform, written as self-contained functions that do not read train "
    "or test_inputs, reproduces every training example, its test outputs become one of your two "
    "attempts. Then answer. "
)

_FUNCTION = re.compile(r"<function=([^>\n]*)>", re.IGNORECASE)
_PARAMETER = re.compile(r"<parameter=([^>\n]*)>", re.IGNORECASE)
_PARAMETER_END = re.compile(
    r"</parameter>|<parameter=|</function>|</tool_call>|<\|im_end\|>|<\|endoftext\|>",
    re.IGNORECASE,
)


class GenLike(Protocol):
    """What ``generate`` returns (as ``scripts/fn_probe.py``'s ``GenResult``)."""

    text: str
    ids: list[int]
    finish: str
    completion_tokens: int
    cached_tokens: int | None

    def row(self) -> dict[str, Any]:
        """Return the request's record without its text and ids."""
        ...


class TokenizerLike(Protocol):
    """The model's tokenizer and chat template (as ``fn_probe``'s ``HFTokenizer``)."""

    def encode(self, text: str) -> list[int]:
        """Return the ids of ``text`` without added special tokens."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text of ``ids``, special tokens kept."""
        ...

    def render(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        """Render the chat template with a generation prompt."""
        ...


# generate(input_ids, max_new_tokens, cut_at_wall_time, kind, extra_sampling) -> GenLike.
GenerateFn = Callable[[list[int], int, float, str, Mapping[str, Any]], "GenLike"]


@dataclass(frozen=True)
class ToolConfig:
    """The YAML ``tool:`` block; the defaults are the research note's values.

    Attributes:
        cap: Generated tokens per task over all turns before forcing (version 10's 65,536).
        context_tokens: Bound on prompt plus generated ids per request (P5.2-01's
            ``--context-length``, 86,016); the forcing turn's room is held back.
        max_calls: Calls run per task; later ones get :attr:`budget_message`.
        reply_tokens: Tool-turn tokens per task (the :data:`TOOL_TURN` frame and the reply);
            once reached, later calls get :attr:`budget_message`.
        call_timeout_s: Wall seconds per call; the REPL is restarted on a breach.
        reply_chars: The reply's length cap, head and tail kept (``feedback.clip``).
        forcing_tokens: The forcing turn's cap (P5.2-01's 6,144).
        thinking_end_note: Appended before ``</think>`` when a turn is cut inside its thinking.
        forcing_suffix: The tag that closes the thinking.
        force_again_note: The text of the second forcing request, sent after a forced reply's
            tool call is stripped (:data:`FORCE_AGAIN_NOTE`).
        budget_message: The reply to a call over a budget.
        lenient_answers: P6.9-01's lenient answer parser and first-attempt rule.
        structured_forcing: Constrain forced replies to a complete JSON answer object.
        strip_tokens: End-of-turn tokens removed before the answer blocks are read.
        stop: Stop strings of the thinking and tool turns, kept in the output.
        reasoning_effort: The template's ``reasoning_effort``.
        verify_call_timeout_s: The verifier's limit per ``transform`` call.
        max_verify: Joined programs verified per task at most, latest first.
        repl_memory_bytes: The address-space cap (``RLIMIT_AS``, 1 GiB) of the REPL and of each
            verifier process.
        repl_cpu_s: CPU seconds per REPL process (``RLIMIT_CPU``): a backstop behind the
            per-call wall timeout, above :attr:`max_calls` times :attr:`call_timeout_s`.
    """

    cap: int = 65_536
    context_tokens: int = 86_016
    max_calls: int = 40
    reply_tokens: int = 16_384
    call_timeout_s: float = 10.0
    reply_chars: int = 2_000
    forcing_tokens: int = 6_144
    thinking_end_note: str = THINKING_END_NOTE
    forcing_suffix: str = FORCING_SUFFIX
    force_again_note: str = FORCE_AGAIN_NOTE
    budget_message: str = BUDGET_MESSAGE
    lenient_answers: bool = True
    structured_forcing: bool = False
    strip_tokens: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>")
    stop: tuple[str, ...] = (CALL_CLOSE,)
    reasoning_effort: str = "xhigh"
    verify_call_timeout_s: float = 2.0
    max_verify: int = 8
    repl_memory_bytes: int = 1024**3
    repl_cpu_s: int = 600


def tool_config(mapping: Mapping[str, Any]) -> ToolConfig:
    """Build the loop's settings from the YAML ``tool:`` block.

    Args:
        mapping: The block; a missing key keeps its default.

    Returns:
        The settings.

    Raises:
        ValueError: For an unknown key, a value of the wrong type, a budget below 1, or an
            effort the template refuses.
    """
    fields = {f.name: f for f in dataclasses.fields(ToolConfig)}
    unknown = sorted(set(mapping) - set(fields))
    if unknown:
        raise ValueError(f"unknown tool keys: {unknown}")
    defaults = ToolConfig()
    values: dict[str, Any] = {}
    for name, value in mapping.items():
        kind = type(getattr(defaults, name))
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"tool.{name} must be true or false; got {value!r}")
        elif kind is tuple:
            if isinstance(value, str) or not isinstance(value, Iterable):
                raise ValueError(f"tool.{name} must be a list of strings; got {value!r}")
            value = tuple(str(v) for v in value)
        elif kind in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"tool.{name} must be a number; got {value!r}")
            if kind is int and value != int(value):
                raise ValueError(f"tool.{name} must be an integer; got {value!r}")
            value = kind(value)
            if value <= 0:
                raise ValueError(f"tool.{name} must be positive; got {value!r}")
        elif not isinstance(value, str):
            raise ValueError(f"tool.{name} must be a string; got {value!r}")
        values[name] = value
    config = dataclasses.replace(defaults, **values)
    if config.reasoning_effort not in ("xhigh", "medium", "low"):
        raise ValueError(f"tool.reasoning_effort {config.reasoning_effort!r} is not supported")
    return config


@dataclass
class ToolOutcome:
    """One task's run of the tool loop.

    Attributes:
        text: The transcript after the prompt: generated text, tool turns, forcing suffix.
        turns: One row per ``generate`` call: ``kind`` (``think``, ``tool``, ``force``,
            ``force2``), ``input_tokens``, ``max_new_tokens``, and the result's ``row()``; a
            forced reply whose call was stripped also has ``stripped_call_chars``.
        calls: One row per parsed call: ``turn``, ``chars``, ``seconds``, ``status`` (``ok``,
            ``error``, ``timeout``, ``budget``), ``reply_tokens``, ``check`` (``[passed,
            total]`` per ``check`` call), ``checked`` (``[name, passed, total]`` per ``check``
            call), ``in_reasoning``, ``closed``, ``defines_transform``,
            ``restarted``, ``error``.
        direct: Per test input, the model's :math:`A_1, A_2` (fewer when it gave fewer).
        program: The verified program, ``source``, ``call`` (the call that defined its
            ``transform`` or checked its function), ``name`` (``transform`` or the checked
            function's name), ``demos_passed``, ``demos``, ``test_outputs``; or None.
        attempts: Per test input, the first two distinct grids of :math:`(A_1, P, A_2)`.
        forced: Whether the forcing turn ran.
        completion_tokens: Generated tokens over all turns.
        tool_reply_tokens: Tool-turn tokens appended over all calls.
        ended_by: ``stop`` (the model ended), ``end`` (no time left), ``error``, or the cut's
            reason with ``+force`` (``length``, ``cancelled``, ``abort``, ``error``, ``cap``,
            ``context``, ``force_time``).
        errors: Engine, sandbox, and parse errors.
        verifications: One row per joined program verified: ``call``, ``name``, ``passed``,
            ``demos``, ``verified``, ``fatal`` guard rules.
        opening_tokens: The rendered prompt's ids.
    """

    text: str
    turns: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    direct: list[list[Grid]]
    program: dict[str, Any] | None
    attempts: list[list[Grid]]
    forced: bool
    completion_tokens: int
    tool_reply_tokens: int
    ended_by: str
    errors: list[str]
    verifications: list[dict[str, Any]] = field(default_factory=list)
    opening_tokens: int = 0

    def record(self) -> dict[str, Any]:
        """Return the outcome as a JSON-ready dict without the transcript."""
        row = dataclasses.asdict(self)
        row.pop("text")
        return row


@dataclass(frozen=True)
class ParsedCall:
    """One tool call read from a turn's text.

    Attributes:
        name: The function name (empty when the markup has no ``<function=...>``).
        arguments: Parameter name to value.
        start: The call's offset in the turn's text.
        closed: Whether ``</tool_call>`` closes it.
    """

    name: str
    arguments: dict[str, str]
    start: int
    closed: bool


def system_text(cfg: ToolConfig) -> str:
    """Return the tool arm's system prompt: ``DIRECT_TWO_SYSTEM_PROMPT`` with the tool paragraph.

    Why: the plain prompt says there is no code interpreter; the tool arm replaces that one
    sentence with :data:`TOOL_PARAGRAPH`, so the model never reads both.

    Args:
        cfg: The settings (the paragraph names the call limits).

    Returns:
        The prompt.

    Raises:
        ValueError: If the plain prompt does not hold :data:`NO_INTERPRETER` exactly once.
    """
    count = DIRECT_TWO_SYSTEM_PROMPT.count(NO_INTERPRETER)
    if count != 1:
        raise ValueError(f"DIRECT_TWO_SYSTEM_PROMPT holds the sentence {count} times, not once")
    paragraph = TOOL_PARAGRAPH.format(timeout=cfg.call_timeout_s, chars=f"{cfg.reply_chars:,}")
    return DIRECT_TWO_SYSTEM_PROMPT.replace(NO_INTERPRETER, paragraph)


def parse_call(text: str) -> ParsedCall | None:
    """Read the last tool call of a turn, closed or not, inside the reasoning or after it.

    Why: the model writes calls in the qwen3_coder markup but may leave the closing tags out or
    call before ``</think>``; as the ARChitects do, the markup is recovered from the text
    itself, not from a server-side parser.

    Args:
        text: One assistant turn's generated text.

    Returns:
        The call, or None when the turn has no ``<tool_call>`` or ``<function=`` markup, when
        text other than end-of-turn tokens follows the call (only a call that ends the turn is
        one), or when the turn's visible part holds a complete answer block before the call (that
        turn is final).
    """
    lowered = text.lower()
    start = lowered.rfind(CALL_OPEN)
    if start < 0:
        start = lowered.rfind("<function=")
        if start < 0:
            return None
    body = text[start:]
    lowered_body = body.lower()
    closed = CALL_CLOSE in lowered_body
    closer = CALL_CLOSE if closed else ("</function>" if CALL_OPEN not in lowered_body else None)
    if closer is not None and closer in lowered_body:
        after = body[lowered_body.index(closer) + len(closer) :]
        for token in (IM_END, "<|endoftext|>"):
            after = after.replace(token, "")
        if after.strip():
            return None
    head = text[:start]
    if THINK_CLOSE in head and any(
        tag == "answer" for tag, _ in fenced_blocks(head.rsplit(THINK_CLOSE, 1)[1])
    ):
        return None
    function = _FUNCTION.search(body)
    if CALL_OPEN not in lowered_body and (function is None or not _PARAMETER.search(body)):
        return None
    if function is None:
        return ParsedCall("", {}, start, closed)
    rest = body[function.end() :]
    arguments: dict[str, str] = {}
    for match in _PARAMETER.finditer(rest):
        name = match.group(1).strip()
        after = rest[match.end() :]
        end = _PARAMETER_END.search(after)
        value = after[: end.start()] if end else after
        if value.startswith("\n"):
            value = value[1:]
        if name:
            arguments[name] = value.rstrip()
    return ParsedCall(function.group(1).strip(), arguments, start, closed)


def joined_program(codes: Sequence[str], drop: Iterable[str] = ()) -> str:
    """Join the program statements of a task's calls; the last definition of a name wins.

    Why: in a persistent REPL the model defines helpers in one call and ``transform`` in a
    later one; the verifier needs them in one self-contained source (the ARChitects retain
    functions the same way).

    Args:
        codes: The calls' sources, oldest first.
        drop: Names whose definitions and plain assignments are left out.

    Returns:
        The imports (first occurrence each, in order), then the function and class definitions
        and the assignments to plain names that :func:`arc26.agent.loop.program_source` keeps,
        each at the place of its last occurrence; every other statement is left out.
    """
    imports: dict[str, str] = {}
    others: dict[tuple[str, ...], str] = {}
    dropped = frozenset(drop)
    for code in codes:
        source = program_source(code)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        lines = source.splitlines(keepends=True)
        for node in tree.body:
            decorators = getattr(node, "decorator_list", ())
            first = min([node.lineno, *(d.lineno for d in decorators)])
            text = "".join(lines[first - 1 : node.end_lineno or node.lineno])
            if not text.endswith("\n"):
                text += "\n"
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Try)):
                imports.setdefault(text, text)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                key: tuple[str, ...] = ("def", node.name)
            elif (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and node.value is not None
                and _one_statement(lines, node)
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if not all(isinstance(t, ast.Name) for t in targets):
                    continue
                key = ("assign", *sorted(t.id for t in targets if isinstance(t, ast.Name)))
            else:
                continue
            if dropped.intersection(key[1:]):
                continue
            others.pop(key, None)
            others[key] = text
    return "".join(imports.values()) + "".join(others.values())


def _one_statement(lines: Sequence[str], node: ast.stmt) -> bool:
    """Whether a statement's lines hold it alone (not ``k = 3; print(k)``)."""
    try:
        text = "".join(lines[node.lineno - 1 : node.end_lineno or node.lineno])
        tree = ast.parse(text.strip())
    except SyntaxError:
        return False
    return len(tree.body) == 1


def final_answers(
    turn_texts: Sequence[str], task: AgentTask, cfg: ToolConfig
) -> tuple[list[list[Grid]], list[str]]:
    """Return :math:`A_1, A_2` per test input from the last turn with answer blocks.

    Why: the final turn normally holds the answers; a turn that wrote its blocks and then a call
    still counts when no later turn answers, so no answer the model gave is lost. Every refused
    block is reported, also in a turn whose blocks are all refused (``1818057f`` in P5.2-01
    revision 20 lost a ragged hand-copied grid without a record); no grid is repaired.

    Args:
        turn_texts: The assistant turns' texts, oldest first (the forced reply belongs to the
            turn it continues).
        task: The task.
        cfg: The settings (the lenient parser, the end tokens).

    Returns:
        The grids per test input (two or fewer, from :func:`arc26.agent.shot.shot_answers`) and
        one error per refused block, ``turn T block B: reason``, of the turn used and of every
        later turn.
    """
    count = len(task.test_inputs)
    implied = implied_shapes(task)
    errors: list[str] = []
    for index in range(len(turn_texts) - 1, -1, -1):
        text = turn_texts[index]
        if THINK_CLOSE not in text:
            continue
        visible = text.rsplit(THINK_CLOSE, 1)[1]
        for token in cfg.strip_tokens:
            visible = visible.replace(token, "\n")
        answers: list[tuple[int, Grid]] = []
        refused: list[str] = []
        blocks = [body for tag, body in fenced_blocks(visible) if tag == "answer"]
        for number, body in enumerate(blocks):
            try:
                answers.append(parse_answer(body, count, lenient=cfg.lenient_answers))
            except ValueError as err:
                refused.append(f"turn {index} block {number}: {str(err)[:200]}")
        errors = refused + errors
        if not answers:
            continue
        holder = type("Record", (), {"direct_answers": answers})()
        picks = [
            shot_answers([holder], j, implied[j] if cfg.lenient_answers else None)
            for j in range(count)
        ]
        return picks, errors
    return [[] for _ in range(count)], errors


def slots(
    direct: Sequence[Sequence[Grid]], program: Sequence[Grid | None] | None
) -> list[list[Grid]]:
    """Return per test input the first two distinct grids of :math:`(A_1, P, A_2)`.

    Args:
        direct: Per test input, :math:`A_1, A_2` (fewer when missing).
        program: Per test input, the verified program's output, or None.

    Returns:
        Two grids or fewer per test input; nothing is padded.
    """
    out: list[list[Grid]] = []
    for j, pair in enumerate(direct):
        order = [
            pair[0] if len(pair) > 0 else None,
            program[j] if program is not None and j < len(program) else None,
            pair[1] if len(pair) > 1 else None,
        ]
        picked: list[Grid] = []
        for grid in order:
            if grid is None or grid_problem(grid) is not None:
                continue
            grid = [[int(v) for v in row] for row in grid]
            if grid not in picked:
                picked.append(grid)
            if len(picked) == 2:
                break
        out.append(picked)
    return out


def _answer_schema(count: int) -> dict[str, Any]:
    """Build the bounded forced-answer schema; four or more inputs get one attempt each."""
    grid = {"anyOf": [
        {"type": "array", "minItems": 1, "maxItems": 30,
         "items": {"type": "string", "pattern": f"^[0-9]{{{width}}}$"}}
        for width in range(1, 31)
    ]}
    choices = {"type": "array", "minItems": 1, "maxItems": 1 if count >= 4 else 2,
               "items": grid}
    keys = [str(i) for i in range(count)]
    return {"type": "object", "properties": {"answers": {
        "type": "object", "properties": {key: choices for key in keys},
        "required": keys, "additionalProperties": False,
    }}, "required": ["answers"], "additionalProperties": False}


def _schema_answers(reply: str, count: int, finish: str, strip_tokens: Sequence[str]) -> str:
    """Validate a complete object and translate it to ordinary answer blocks."""
    if finish not in ("stop", "length", "cancelled"):
        raise ValueError("failed forced reply")
    text = reply.strip()
    for token in strip_tokens:
        if text.endswith(token):
            text = text[:-len(token)].rstrip()

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate key")
            obj[key] = value
        return obj

    try:
        obj = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError) as err:
        raise ValueError(f"invalid JSON: {err}") from err
    keys = {str(i) for i in range(count)}
    if not isinstance(obj, dict) or set(obj) != {"answers"}:
        raise ValueError("expected answers object")
    answers = obj["answers"]
    if not isinstance(answers, dict) or set(answers) != keys:
        raise ValueError("test indices do not match")
    blocks = []
    for i in range(count):
        grids = answers[str(i)]
        if not isinstance(grids, list) or not 1 <= len(grids) <= (1 if count >= 4 else 2):
            raise ValueError("invalid attempt count")
        for rows in grids:
            if (not isinstance(rows, list) or not 1 <= len(rows) <= 30
                    or any(not isinstance(row, str) or re.fullmatch(r"[0-9]{1,30}", row)
                           is None for row in rows)):
                raise ValueError("invalid grid rows")
            grid = [[int(c) for c in row] for row in rows]
            if grid_problem(grid) is not None:
                raise ValueError("invalid grid")
            blocks.append("```answer\n" + json.dumps({"test_index": i, "grid": grid}) + "\n```")
    return THINK_CLOSE + "\n" + "\n".join(blocks)


def forcing_parts(turn: str, cfg: ToolConfig) -> tuple[str, str, str, ParsedCall | None]:
    """Return what the forcing turn appends to a cut turn, so that the model answers.

    Why: in this template a call follows ``</think>``, so a cut inside a call (or after
    ``</think>`` before one) is the common case; continued as is, the model finishes the call
    and ends its turn without answer blocks. Such a call is closed and answered with the budget
    message, and the next turn starts with the note and ``</think>``.

    Args:
        turn: The cut assistant turn's text.
        cfg: The settings (the budget message, the note, the closing tag).

    Returns:
        ``(closers, tool_turn, tail, call)``: the tags that close a cut call, the tool turn with
        the budget message (empty when none is added), the note and ``</think>`` (empty when the
        cut turn is continued as is), and the cut call or None.
    """
    note = cfg.thinking_end_note + cfg.forcing_suffix
    budget_turn = TOOL_TURN.format(reply=cfg.budget_message)
    call = parse_call(turn)
    if call is not None:
        return _closers(turn[call.start :]), budget_turn, note, call
    if THINK_CLOSE not in turn:
        return "", "", note, None
    visible = turn.rsplit(THINK_CLOSE, 1)[1]
    if "```" in visible:
        return "", "", "", None
    return "", budget_turn, note, None


def _closers(body: str) -> str:
    """The tags that close a cut call's markup (see :data:`CALL_CLOSERS`)."""
    lowered = body.lower()
    if CALL_CLOSE in lowered:
        return ""
    out = ""
    opened = lowered.rfind("<parameter=")
    if opened >= 0 and "</parameter>" not in lowered[opened:]:
        out += "\n</parameter>"
    if "</function>" not in lowered:
        out += "\n</function>"
    return out + "\n" + CALL_CLOSE


def _cut_row(call: ParsedCall, turn: str, index: int) -> dict[str, Any]:
    """The call row of a call cut before its end: never run, answered with the budget message."""
    return {
        "name": call.name,
        "chars": len(call.arguments.get("code", "")),
        "seconds": 0.0,
        "status": "budget",
        "check": None,
        "in_reasoning": THINK_CLOSE not in turn[: call.start],
        "closed": call.closed,
        "defines_transform": False,
        "restarted": False,
        "error": "cut",
        "ran": False,
        "cut": True,
        "turn": index,
        "reply_tokens": 0,
    }


def _reply_text(
    stdout: str, error: str | None, restarted: bool, timeout_s: float, memory_bytes: int
) -> str:
    """The python tool's reply: the output, then the exception as the last line."""
    parts = []
    output = stdout.rstrip()
    if output:
        parts.append(output)
    lost = (
        "the REPL was restarted, so earlier variables are lost (train, test_inputs, np, lib "
        "and check are loaded again)"
    )
    if error == "timeout":
        parts.append(
            f"TimeoutError: the call ran over {timeout_s:g} seconds and was stopped; {lost}."
        )
    elif error == "memory":
        limit = f"{memory_bytes / 1024**3:g} GiB"
        if restarted:
            parts.append(f"MemoryError: the call ran over the {limit} memory limit; {lost}.")
        else:
            parts.append(
                f"MemoryError: the call ran over the {limit} memory limit and was stopped; the "
                "session goes on, and variables set before the failing line are kept."
            )
    elif error is not None:
        if restarted:
            parts.append(f"Note: {lost}.")
        if not output.endswith(error):
            parts.append(error)
    elif restarted:
        parts.append(f"Note: {lost}.")
    return "\n".join(parts) if parts else "(no output)"


class _Session:
    """The REPL of one task, with ``check`` loaded at the start and after every restart."""

    def __init__(self, task: AgentTask, cfg: ToolConfig, errors: list[str]) -> None:
        self._task = task
        self._cfg = cfg
        self._timeout_s = cfg.call_timeout_s
        self._errors = errors
        self.repl: SubprocessRepl | None = None
        self.failure: str | None = None

    def _prelude(self) -> None:
        assert self.repl is not None
        result = self.repl.run_cell(CHECK_SOURCE, self._timeout_s)
        if result.error is not None:
            self._errors.append(f"check prelude: {result.error}")

    def run(self, code: str, timeout_s: float) -> tuple[str, str | None, bool, float]:
        """Run one cell; return its output, error, restart flag, and seconds."""
        if self.failure is None and self.repl is None:
            try:
                self.repl = SubprocessRepl(
                    self._task,
                    memory_bytes=self._cfg.repl_memory_bytes,
                    cpu_seconds=self._cfg.repl_cpu_s,
                )
                self._prelude()
            except SandboxError as err:
                self.failure = str(err)[:300]
                self._errors.append(f"sandbox: {self.failure}")
        if self.repl is None:
            return "", f"SandboxError: the Python tool is unavailable: {self.failure}", False, 0.0
        try:
            result = self.repl.run_cell(code, timeout_s)
            if result.restarted:
                self._prelude()
        except SandboxError as err:
            self.failure = str(err)[:300]
            self._errors.append(f"sandbox: {self.failure}")
            self.close()
            return "", f"SandboxError: {self.failure}", True, 0.0
        return result.stdout, result.error, result.restarted, result.seconds

    def close(self) -> None:
        """Stop the REPL if one was started."""
        if self.repl is not None:
            self.repl.close()
            self.repl = None


def _grids(outputs: Iterable[Grid | None]) -> list[Grid | None]:
    return [None if g is None else [[int(v) for v in row] for row in g] for g in outputs]


def _binds(source: str, name: str) -> bool:
    """Whether a joined program binds ``name`` at its top level (a def, class, or assignment)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return True
    return False


def defined_functions(code: str) -> list[str]:
    """Return the names of the top-level functions a call's program statements define.

    Args:
        code: One call's source.

    Returns:
        The ``def`` names in order (``transform`` included); empty when the source does not
        parse.
    """
    try:
        tree = ast.parse(program_source(code))
    except SyntaxError:
        return []
    return [
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def program_candidates(
    codes: Sequence[str],
    defining: Sequence[int],
    checked: Sequence[tuple[int, str]],
    limit: int | None = None,
) -> list[tuple[int, str, str]]:
    """Return the joined programs to verify for the program slot, in order of preference.

    Why: the model does not always name its rule ``transform``: on ``1818057f`` (P5.2-01
    revision 20) it wrote a correct ``plus_transform``, tested it with its own
    ``np.array_equal`` loop, never called ``check``, and lost the slot. So after ``transform``,
    any top-level function of the task's calls is tried, each run through a wrapper
    ``def transform(grid): return NAME(grid)``, which the hardcoding guards follow into
    ``NAME``'s body (``guards._tracked_functions``); every candidate still has to pass
    :class:`arc26.agent.verify.PoolVerifier` on every demonstration.

    Args:
        codes: The calls' sources that ran, oldest first.
        defining: The indexes in ``codes`` of the calls that define ``transform``.
        checked: ``(index in codes, function name)`` per ``check`` call that passed every
            demonstration, oldest first.
        limit: The most candidates returned (None: all).

    Returns:
        ``(call index, name, source)`` per candidate, each source once: every ``transform``
        (latest first); then the functions that passed ``check`` (latest first); then every
        other top-level function the calls define (latest definition first). A named
        candidate's source is the calls up to its index joined without any ``transform``, then
        the wrapper.
    """
    out: list[tuple[int, str, str]] = []
    seen: set[str] = set()

    def add(index: int, name: str, source: str) -> None:
        if source and source not in seen and (limit is None or len(out) < limit):
            seen.add(source)
            out.append((index, name, source))

    for index in reversed(defining):
        add(index, "transform", joined_program(codes[: index + 1]))
    named: list[tuple[int, str]] = list(reversed(checked))
    for index in range(len(codes) - 1, -1, -1):
        named.extend((index, name) for name in reversed(defined_functions(codes[index])))
    tried: set[str] = set()
    for index, name in named:
        if name == "transform" or not name.isidentifier() or name in tried:
            continue
        source = joined_program(codes[: index + 1], drop=("transform",))
        if not _binds(source, name):
            continue
        tried.add(name)
        add(index, name, source + f"\n\ndef transform(grid):\n    return {name}(grid)\n")
    return out


def without_assignments(source: str) -> str:
    """Return a joined program with its top-level assignments removed.

    Why: :func:`joined_program` keeps one-line assignments to plain names, and such an
    assignment can read a name that only a dropped REPL statement bound (a ``for`` loop, a tuple
    target): ``d=np.argwhere(inp!=out)`` then raises ``NameError`` at load and fails every pair
    (``a395ee82``, ``b5ca7ac4``, ``6e453dd6`` in P5.2-01 revision 21).

    Args:
        source: A joined program.

    Returns:
        The imports, function and class definitions and every other statement but the
        module-level ``Assign``, ``AnnAssign`` and ``AugAssign`` ones, in order; the source
        unchanged when it does not parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    kept: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        decorators = getattr(node, "decorator_list", ())
        first = min([node.lineno, *(d.lineno for d in decorators)])
        text = "".join(lines[first - 1 : node.end_lineno or node.lineno])
        kept.append(text if text.endswith("\n") else text + "\n")
    return "".join(kept)


def _load_name_error(result: Verification) -> bool:
    """Whether every pair failed with ``NameError`` (the program's load, or a missing global)."""
    return bool(result.pairs) and all(
        not p.exact and (p.error or "").startswith("NameError") for p in result.pairs
    )


def verify_worst_s(task: AgentTask, cfg: ToolConfig) -> float:
    """Return the longest time one verification may take.

    That is one call limit (plus 0.5 s) per input plus one, and the verifier's hang allowance;
    its worker's startup is not counted.

    Args:
        task: The task (its demonstration and test input counts).
        cfg: The settings (:attr:`ToolConfig.verify_call_timeout_s`).

    Returns:
        Seconds.
    """
    inputs = len(task.train) + len(task.test_inputs)
    return (inputs + 1) * (cfg.verify_call_timeout_s + 0.5) + VERIFY_HANG_GRACE_S


def _verify(
    task: AgentTask,
    codes: Sequence[str],
    defining: Sequence[int],
    checked: Sequence[tuple[int, str]],
    cfg: ToolConfig,
    verifier: Verifier | None,
    errors: list[str],
    rows: list[dict[str, Any]],
    end: float,
    now: Callable[[], float],
) -> dict[str, Any] | None:
    """The first verified program of :func:`program_candidates`, or None.

    A candidate whose every pair fails with ``NameError`` is verified once more right after it
    without its top-level assignments (:func:`without_assignments`); at most
    :attr:`ToolConfig.max_verify` verifications in all, retries counted.

    Never raises: a failure of the verifier or of the joining is an error row. A verification
    starts only when its worst case, one call limit (plus 0.5 s) per input plus one and the
    verifier's hang allowance, ends before ``end``; a worker's startup is not counted.
    """
    if not codes:
        return None
    own = None
    worst = verify_worst_s(task, cfg)
    tried: set[str] = set()
    try:
        candidates = program_candidates(codes, defining, checked)
        queue = [(index, name, source, False) for index, name, source in candidates]
        while queue and len(tried) < cfg.max_verify:
            index, name, source, stripped = queue.pop(0)
            if source in tried:
                continue
            if now() + worst > end:
                errors.append("verifier: no time left before the end")
                break
            if verifier is None:
                from arc26.agent.verify import PoolVerifier

                own = verifier = PoolVerifier(
                    1,
                    call_timeout_s=cfg.verify_call_timeout_s,
                    memory_bytes=cfg.repl_memory_bytes,
                )
            tried.add(source)
            result: Verification = verifier.verify(task, source)
            rows.append(
                {
                    "call": index,
                    "name": name,
                    "passed": result.passed,
                    "demos": len(result.pairs),
                    "verified": result.verified,
                    "fatal": [g.rule for g in result.guards if g.fatal],
                    "without_assignments": stripped,
                }
            )
            if result.verified:
                return {
                    "source": source,
                    "call": index,
                    "name": name,
                    "demos_passed": result.passed,
                    "demos": len(result.pairs),
                    "test_outputs": _grids(result.test_outputs),
                    "without_assignments": stripped,
                }
            if not stripped and _load_name_error(result):
                # A kept top-level assignment may read a name that only a dropped REPL
                # statement bound (P5.8b): the same candidate once more without assignments,
                # next in line, inside the same cap and time limit.
                retry = without_assignments(source)
                if retry and retry != source:
                    queue.insert(0, (index, name, retry, True))
    except Exception as err:  # model content must never cost the task its direct answers
        errors.append(f"verifier: {type(err).__name__}: {str(err)[:300]}")
    finally:
        if own is not None:
            try:
                own.close()
            except Exception as err:
                errors.append(f"verifier close: {type(err).__name__}: {str(err)[:200]}")
    return None


def solve_with_tools(
    task: AgentTask,
    tok: TokenizerLike,
    generate: GenerateFn,
    cfg: ToolConfig,
    *,
    force_at: float,
    end: float,
    now: Callable[[], float] = time.time,
    verifier: Verifier | None = None,
) -> ToolOutcome:
    """Run one task: thinking with ``python`` calls, the forcing turn if cut, then the slots.

    Args:
        task: The task.
        tok: The tokenizer (``encode``, ``decode``, ``render``).
        generate: One request: ``generate(input_ids, max_new_tokens, cut_at_wall_time, kind,
            extra_sampling)``; ``extra_sampling`` carries ``stop`` and ``no_stop_trim`` for the
            thinking and tool turns.
        cfg: The settings (:func:`tool_config`).
        force_at: Wall time (on ``now``'s clock) from which a running request is cut and forced.
        end: Wall time at which every request is cut.
        now: The clock of ``force_at`` and ``end``.
        verifier: A shared verifier; None starts a one-worker
            :class:`arc26.agent.verify.PoolVerifier` for this task when a ``transform`` exists.

    Returns:
        The outcome; the answers are grids, never scored here.
    """
    errors: list[str] = []
    messages = [
        {"role": "system", "content": system_text(cfg)},
        {"role": "user", "content": task_text(task)},
    ]
    prompt = tok.render(messages, tools=[PYTHON_TOOL], reasoning_effort=cfg.reasoning_effort)
    ids = tok.encode(prompt)
    opening = len(ids)
    im_end = tok.encode(IM_END)
    schema_note = (
        "\n\nI stop reasoning. Write only one JSON object with an answers object mapping "
        "each test index (as a string) to a list of one or two grids, best first. Each grid "
        "is a list of row strings containing only digits 0 through 9. For four or more test "
        "inputs write one grid per input. Include every test index.\n"
    )
    force_cfg = (
        dataclasses.replace(cfg, thinking_end_note=schema_note, forcing_suffix=FORCING_SUFFIX)
        if cfg.structured_forcing else cfg
    )
    force_text = force_cfg.thinking_end_note + force_cfg.forcing_suffix
    # The longest forcing suffix (a cut call closed, the budget tool turn, the note) and the
    # forced answer stay free in the context; the forced answer also stays inside the cap.
    budget_turn = TOOL_TURN.format(reply=cfg.budget_message)
    reserve = cfg.forcing_tokens + len(tok.encode(CALL_CLOSERS + budget_turn + force_text))
    think_cap = cfg.cap - cfg.forcing_tokens
    sampling = {"stop": list(cfg.stop), "no_stop_trim": True}
    logger = timing.get_logger()

    transcript: list[str] = []
    turn_texts: list[str] = [""]
    turns: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    codes: list[str] = []
    defining: list[int] = []
    checked: list[tuple[int, str]] = []
    generated = reply_tokens = run_calls = 0
    ended_by = "stop"
    cut: str | None = None
    session = _Session(task, cfg, errors)
    kind = "think"
    verifications: list[dict[str, Any]] = []
    program: dict[str, Any] | None = None
    verified = False
    translated: str | None = None

    def request(
        input_ids: list[int], max_new: int, cut_at: float, label: str, extra: Mapping[str, Any]
    ) -> tuple[GenLike | None, list[int]]:
        head = {"kind": label, "input_tokens": len(input_ids), "max_new_tokens": max_new}
        try:
            result = generate(list(input_ids), max_new, cut_at, label, extra)
        except Exception as err:  # an engine failure ends the task, its text so far is kept
            errors.append(f"generate ({label}): {type(err).__name__}: {str(err)[:300]}")
            turns.append({**head, "finish": "error", "error": str(err)[:300]})
            return None, []
        out = list(result.ids or ()) or (tok.encode(result.text) if result.text else [])
        turns.append({**result.row(), **head})
        return result, out

    try:
        while True:
            if now() >= end:
                ended_by = "end"
                break
            if now() >= force_at:
                cut = "force_time"
                break
            room = cfg.context_tokens - len(ids) - reserve
            max_new = min(think_cap - generated, room)
            if max_new <= 0:
                cut = "cap" if think_cap - generated <= room else "context"
                break
            result, out = request(ids, max_new, force_at, kind, sampling)
            if result is None:
                ended_by = "error"
                break
            ids += out
            generated += result.completion_tokens or len(out)
            transcript.append(result.text)
            turn_texts[-1] += result.text
            if result.finish != "stop":
                if result.finish in ("error", "abort") and not out:
                    errors.append(f"generate ({kind}): finish {result.finish}")
                    ended_by = result.finish
                else:
                    cut = result.finish
                break
            call = parse_call(turn_texts[-1])
            if call is None:
                ended_by = "stop"
                break
            try:
                reply, row = _run_call(call, session, cfg, run_calls, reply_tokens, end, now)
            except Exception as err:  # model content must never end the task
                errors.append(f"tool: {type(err).__name__}: {str(err)[:300]}")
                reply = f"Error: the call could not be run ({type(err).__name__})."
                row = {
                    "name": call.name,
                    "chars": len(call.arguments.get("code", "")),
                    "seconds": 0.0,
                    "status": "error",
                    "check": None,
                    "closed": call.closed,
                    "defines_transform": False,
                    "restarted": False,
                    "ran": False,
                    "error": f"{type(err).__name__}: {str(err)[:200]}",
                }
            row["in_reasoning"] = THINK_CLOSE not in turn_texts[-1][: call.start]
            if row["status"] != "budget":
                run_calls += 1
                code = call.arguments.get("code", "")
                if row["ran"]:
                    codes.append(code)
                    if row["defines_transform"]:
                        defining.append(len(codes) - 1)
                    checked.extend(
                        (len(codes) - 1, name)
                        for name, passed, total in row.get("checked") or ()
                        if total > 0 and passed == total
                    )
            if logger is not None and row["status"] != "budget":
                logger.record(task.task_id, "tool_python", row["seconds"], None)
            suffix = TOOL_TURN.format(reply=reply.strip())
            if ids[-len(im_end) :] == im_end:
                suffix = suffix[len(IM_END) :]
            suffix_ids = tok.encode(suffix)
            if len(ids) + len(suffix_ids) + reserve >= cfg.context_tokens:
                # No room for the reply: the forcing turn answers the call with the budget
                # message (the reserve holds it), so the forced answer still fits.
                row["context_full"] = True
                calls.append({**row, "turn": len(turns) - 1, "reply_tokens": 0})
                cut = "context"
                break
            reply_tokens += len(suffix_ids)
            calls.append({**row, "turn": len(turns) - 1, "reply_tokens": len(suffix_ids)})
            ids += suffix_ids
            transcript.append(suffix)
            turn_texts.append("")
            kind = "tool"

        forced = False
        if cut is not None:
            ended_by = f"{cut}+force"
            if now() < end:
                closers, turn, tail, call = forcing_parts(turn_texts[-1], force_cfg)
                if cfg.structured_forcing and not tail:
                    turn = TOOL_TURN.format(reply=cfg.budget_message)
                    tail = force_text
                if turn and ids[-len(im_end) :] == im_end:
                    turn = turn[len(IM_END) :]
                if call is not None and cut != "context":
                    calls.append(_cut_row(call, turn_texts[-1], len(turns) - 1))
                suffix = closers + turn + tail
                force_ids = ids + (tok.encode(suffix) if suffix else [])
                max_new = min(cfg.forcing_tokens, cfg.context_tokens - len(force_ids))
                if max_new > 0 and cfg.structured_forcing and not verified:
                    session.close()
                    program = _verify(
                        task, codes, defining, checked, cfg, verifier, errors, verifications,
                        end, now,
                    )
                    verified = True
                if max_new > 0 and (not cfg.structured_forcing or now() < end):
                    transcript.append(suffix)
                    turn_texts[-1] += closers
                    if turn:
                        turn_texts.append("")
                    turn_texts[-1] += tail
                    extra = ({"json_schema": json.dumps(_answer_schema(len(task.test_inputs))),
                              "stop": [], "no_stop_trim": False}
                             if cfg.structured_forcing else {"no_stop_trim": True})
                    result, out = request(force_ids, max_new, end, "force", extra)
                    forced = True
                    if result is not None:
                        generated += result.completion_tokens or len(out)
                        if cfg.structured_forcing:
                            transcript.append(result.text)
                            try:
                                translated = _schema_answers(
                                    result.text, len(task.test_inputs), result.finish,
                                    cfg.strip_tokens,
                                )
                                turns[-1]["schema_translation"] = "accepted"
                            except ValueError as err:
                                turns[-1]["schema_translation"] = "rejected"
                                errors.append(f"answer: structured force: {err}")
                            continue_forced = False
                        else:
                            continue_forced = True
                        again = (_strip_forced_call(turn_texts[-1], result.text, out, tok)
                                 if continue_forced else None)
                        if again is None:
                            if continue_forced:
                                transcript.append(result.text)
                                turn_texts[-1] += result.text
                        else:
                            # The reply wrote a call and no answer: the call is dropped (never
                            # run) and one more request asks for the answer blocks.
                            kept_ids, kept_text = again
                            turns[-1]["stripped_call_chars"] = len(result.text) - len(kept_text)
                            transcript.append(kept_text)
                            turn_texts[-1] += kept_text
                            # P is settled before force2, so force2 cannot use up its time;
                            # force2 is cut a verification's worst case before the end.
                            session.close()
                            program = _verify(
                                task,
                                codes,
                                defining,
                                checked,
                                cfg,
                                verifier,
                                errors,
                                verifications,
                                end,
                                now,
                            )
                            verified = True
                            # At most two newlines between the kept text and the note.
                            gap = len(turn_texts[-1]) - len(turn_texts[-1].rstrip("\n"))
                            note = "\n" * max(0, 2 - gap) + cfg.force_again_note.lstrip("\n")
                            again_ids = force_ids + kept_ids + tok.encode(note)
                            max_new = min(cfg.forcing_tokens, cfg.context_tokens - len(again_ids))
                            cut_again = end - verify_worst_s(task, cfg)
                            if max_new > 0 and now() < cut_again:
                                transcript.append(note)
                                turn_texts[-1] += note
                                result, out = request(
                                    again_ids,
                                    max_new,
                                    cut_again,
                                    "force2",
                                    {"no_stop_trim": True},
                                )
                                if result is not None:
                                    generated += result.completion_tokens or len(out)
                                    transcript.append(result.text)
                                    turn_texts[-1] += result.text
            else:
                ended_by = f"{cut}+end"
    finally:
        session.close()

    direct, parse_errors = final_answers(turn_texts, task, cfg)
    if translated is not None:
        forced_direct, forced_errors = final_answers([translated], task, cfg)
        parse_errors.extend(forced_errors)
        for natural, forced_grids in zip(direct, forced_direct, strict=True):
            for grid in forced_grids:
                if grid not in natural and len(natural) < 2:
                    natural.append(grid)
    errors.extend(f"answer: {e}" for e in parse_errors)
    if not verified:
        program = _verify(
            task, codes, defining, checked, cfg, verifier, errors, verifications, end, now
        )
    attempts = slots(direct, program["test_outputs"] if program else None)
    return ToolOutcome(
        text="".join(transcript),
        turns=turns,
        calls=calls,
        direct=direct,
        program=program,
        attempts=attempts,
        forced=forced,
        completion_tokens=generated,
        tool_reply_tokens=reply_tokens,
        ended_by=ended_by,
        errors=errors,
        verifications=verifications,
        opening_tokens=opening,
    )


def _strip_forced_call(
    turn: str, reply: str, reply_ids: Sequence[int], tok: TokenizerLike
) -> tuple[list[int], str] | None:
    """The forced reply up to its tool call, when it wrote a call and no answer block.

    Why: P5.2-01 revision 20's forced replies wrote a ``<tool_call>`` instead of answer blocks on
    3 of 4 tasks. Such a reply is cut before its first call and continued; the ids kept are a
    prefix of the engine's own ids, so the next request continues token for token.

    Args:
        turn: The turn's text before the forced reply (the forcing suffix included).
        reply: The forced reply's text.
        reply_ids: The forced reply's ids.
        tok: The tokenizer (``decode``).

    Returns:
        ``(ids, text)`` of the longest prefix of ``reply_ids`` whose text ends at or before the
        call; None when the reply has no call, or when the turn's visible part holds a complete
        answer block.
    """
    visible = (turn + reply).rsplit(THINK_CLOSE, 1)[-1]
    if any(tag == "answer" for tag, _ in fenced_blocks(visible)):
        return None
    lowered = reply.lower()
    at = lowered.find(CALL_OPEN)
    if at < 0:
        call = parse_call(turn + reply)
        if call is None or call.start < len(turn):
            return None
        at = call.start - len(turn)
    head = reply[:at]
    # The longest id prefix whose text is a prefix of the reply before the call: a binary search
    # on the decoded length, then a step back past any token that decodes differently alone.
    low, high = 0, len(reply_ids)
    while low < high:
        mid = (low + high + 1) // 2
        if len(tok.decode(reply_ids[:mid])) <= at:
            low = mid
        else:
            high = mid - 1
    while low > 0 and not head.startswith(tok.decode(reply_ids[:low])):
        low -= 1
    ids = list(reply_ids[:low])
    return ids, tok.decode(ids) if ids else ""


def _run_call(
    call: ParsedCall,
    session: _Session,
    cfg: ToolConfig,
    run_calls: int,
    reply_tokens: int,
    end: float,
    now: Callable[[], float],
) -> tuple[str, dict[str, Any]]:
    """Run one parsed call, or answer it with the budget message; return the reply and row."""
    code = call.arguments.get("code")
    row: dict[str, Any] = {
        "name": call.name,
        "chars": len(code or ""),
        "seconds": 0.0,
        "status": "budget",
        "check": None,
        "checked": None,
        "in_reasoning": False,
        "closed": call.closed,
        "defines_transform": False,
        "restarted": False,
        "error": None,
        "ran": False,
    }
    if run_calls >= cfg.max_calls or reply_tokens >= cfg.reply_tokens:
        return cfg.budget_message, row
    if call.name != TOOL_NAME:
        row.update(status="error", error="unknown function")
        return (
            f"Unknown function {call.name!r}: the only function is python, with one "
            "parameter, code."
        ), row
    if code is None:
        row.update(status="error", error="no code parameter")
        return "The python function needs its code parameter: <parameter=code>.", row
    timeout_s = min(cfg.call_timeout_s, max(end - now(), 1.0))
    stdout, error, restarted, seconds = session.run(code, timeout_s)
    not_run = error is not None and (
        error.startswith(("PolicyError", "SyntaxError", "SandboxError"))
    )
    status = "ok" if error is None else ("timeout" if error == "timeout" else "error")
    row.update(
        seconds=round(seconds, 3),
        status=status,
        check=check_summaries(stdout) or None,
        checked=[[n, a, b] for n, a, b in check_records(stdout)] or None,
        defines_transform=guards.defines_transform(code),
        restarted=restarted,
        error=None if error is None else error[:200],
        ran=not not_run,
    )
    reply = clip(
        _reply_text(stdout, error, restarted, timeout_s, cfg.repl_memory_bytes), cfg.reply_chars
    )
    return reply, row
