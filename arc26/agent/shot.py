r"""The long shot after NVARC: direct answers on NVARC's least sure tasks in the time left (P6.5b).

Why: run ``P6.3-05`` (``docs/prerun/P6.3-05.md``) solved 4 of the 15 DEV40 tasks NVARC is least
sure of, against NVARC's 0, with one long xhigh thinking turn per task and direct answers.
Submission 2's NVARC ends a 240-task rerun at about 74% of its soft deadline, so hours of the 12
are idle. This entry point fills them with that shot, NVARC's file the fallback at every moment
(``docs/tasks/P6.5b.md``, design points 2 to 7). Runbook: ``docs/runbooks/agent_shot.md``;
module doc: ``docs/modules/agent_shot.md``; config: ``configs/agent_shot.yaml``.

Usage (exactly :mod:`arc26.agent.run`'s arguments, exit codes, and meanings, so the notebook
switches modules by one config key)::

    python -m arc26.agent.shot --mode competition --tasks <challenges.json> \
        --out /kaggle/working/submission.json --config configs/agent_shot.yaml \
        --engine-venv <venv> --model-dir <weights> --deadline <epoch seconds> \
        --nvarc-submission <file> --nvarc-candidates <candidates.json> --log-dir <dir>
    python -m arc26.agent.shot --mode dev ... --fake-engine <replies.json> \
        --solutions <solutions.json>                                  # the CPU smoke

Order of work:

1. Read the config and the arguments (:func:`arc26.agent.run.resolve_options`); a refusal exits 2
   and writes nothing.
2. Read the tasks, NVARC's submission (the run's own copy, :func:`arc26.agent.run.nvarc_copy`),
   and NVARC's candidates; write ``submission.json`` at once with NVARC's attempts. From here on
   the file is valid at every moment.
3. Rank the tasks by NVARC's confidence (:func:`shot_confidence`, P6.3-05's rule) and keep those
   under ``max_confidence``.
4. With a vLLM venv: prepare it as P6.3-05 did (:func:`arc26.agent.engine_prep.prepare_engine`,
   the GPU preflight), boot the engine once, read its KV pool, and send a sanity request.
5. Waves: up to ``wave_tasks`` tasks at once, each with P6.3-05's request but the thinking cap
   :math:`C` the time left allows (:func:`plan_wave`). Each finished task's answers are merged
   with NVARC's (:func:`merge_output`) and the file is rewritten atomically.
6. At the generation stop every call still running is cancelled and its task keeps NVARC's
   attempts; the engine is stopped and the file written once more. In development mode the run
   is scored (``run.json``, ``shot_metrics.json``).

Wave arithmetic: with :math:`r` the measured decode rate per stream, :math:`\phi` the plan factor,
:math:`p` the measured prefill rate of the forced replies, :math:`F` the forcing cap, and
:math:`m` the margin, a wave of :math:`n` tasks with cap :math:`C` and longest opening :math:`o`
is planned to take

.. math::

    T(C, n, o) = \frac{C}{\phi r} + \frac{n (o + C)}{p} + \frac{F}{r} + m,

every turn of the wave assumed cut. :math:`C` is the largest value up to ``max_cap`` with
:math:`T` at most the seconds left; below ``min_cap`` no wave starts. Its KV need
:math:`\sum_i (o_i + C + F)` over its tasks (:math:`o_i` the opening's bound) must fit the pool
the boot reported, else the wave takes fewer tasks and its cap is planned again.

Task P6.9 adds four options and task P4.7b a fifth, each behind an optional config key whose
absence keeps Submission 3's behavior (``configs/agent_shot.yaml`` sets none of them):

- ``selection.order: buckets`` (option 2a): the queue by NVARC confidence bucket, :math:`(0,
  0.5)` first, then :math:`[0.5, 0.7)` only under ``selection.mid_bucket``, then exactly 0
  (:func:`select_tasks`).
- ``waves.refill`` (option 2b (ii)): a refill queue instead of waves; a stream that ends starts
  the next task at once, its cap planned from the time left as a wave's is
  (:func:`plan_start`, :func:`run_refill`).
- ``request.forcing_in_request`` (option 2b (iii)): the forced answer inside the thinking
  request through vLLM's ``thinking_token_budget``; the plan then drops the forced replies'
  prefill, :math:`T(C, n, o) = C/(\phi r) + F/r + m`.
- ``selection.zero_bucket: false`` (task P4.7b): no task at confidence exactly 0 in the queue,
  in either order; the notebook's rank-256 tail takes them after the shot (:func:`select_tasks`).
- ``request.lenient_answers`` (option 4b): the loop's lenient answer parser
  (:func:`arc26.agent.loop.parse_answer`), and per test output the first valid answer of the
  shape the demonstrations imply as :math:`S_1` (:func:`implied_shapes`, :func:`shot_answers`).

Task P5.1 (LLM first, no NVARC) adds two more, absent in every earlier config:

- ``selection.order: opening``: every task of the run in the queue, by ascending opening bound
  (:func:`opening_bound`), then id; NVARC's confidence is not computed, since an LLM-first build
  runs no NVARC (:func:`select_tasks`).
- ``waves.dev_cap``: in development mode only, ``max_cap`` and ``min_cap`` are lowered to it, so
  a version's short check thinks less than its scored rerun; a competition rerun never reads it.

Exit codes: 0 when a valid ``submission.json`` is in place at the end, whatever failed on the way;
1 when no valid file could be written; 2 when the arguments or the config are refused.
Competition mode never opens a solutions file: ``--solutions`` is refused there, and the two
functions that read solutions (:func:`arc26.agent.run.score_dev` and :func:`shot_dev_metrics`)
refuse any mode but development.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import signal
import sys
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc26.agent import prompts
from arc26.agent import run as agent_run
from arc26.agent import server as server_module
from arc26.agent.loop import MESSAGE_SLACK, PROMPT_SLACK, Budget
from arc26.agent.run import DEV, EXIT_NO_SUBMISSION, EXIT_OK, EXIT_REFUSED, ConfigError
from arc26.agent.scheduler import SubmissionWriter
from arc26.agent.server import (
    EngineBootTimeout,
    EngineServer,
    EngineStartError,
    ServerConfig,
    engine_env,
    http_get,
    vllm_command,
)
from arc26.agent.types import AgentTask, AttemptRecord, Candidate
from arc26.data import Grid

logger = logging.getLogger(__name__)

PROG = "python -m arc26.agent.shot"
SHOT_ATTEMPTS = "shot_attempts.jsonl"
SHOT_PLAN = "shot_plan.json"
SHOT_METRICS = "shot_metrics.json"
SHOT_TRANSCRIPTS = "shot_transcripts.jsonl"
BOOT_FILE = "boot.json"
SUMMARY_MD = "summary.md"
# The phase recorded with every shot attempt: 0, as P6.3-05's direct phase ran it, so the
# attempt ids (``<task>/p0/0``) and their seeds are P6.3-05's.
SHOT_PHASE = 0
# The softmax's temperature over an output's distinct grids (scripts/calibration_report.py,
# TEMPERATURE): fixed, not fit.
TEMPERATURE = 1.0
# P6.3-05's sanity request (scripts/agent_spike.py, sanity_check).
SANITY_PROMPT = "What is 17 + 25? Reply with the number only."
SYSTEM_PROMPTS = {
    "direct_two": prompts.DIRECT_TWO_SYSTEM_PROMPT,
    "direct": prompts.DIRECT_SYSTEM_PROMPT,
}
EFFORTS = (None, "low", "medium", "xhigh")

TOP_KEYS = (
    "seed", "deadline", "selection", "waves", "request", "sanity", "degenerate", "engine",
    "engine_prep", "server", "logs",
)  # fmt: skip
DEADLINE_KEYS = ("session_s", "startup_margin_s", "final_margin_s", "grace_s")
SELECTION_KEYS = ("max_confidence",)
# Optional keys (task P6.9) with the defaults that keep Submission 3's behavior.
SELECTION_OPTIONAL = {"order": "confidence", "mid_bucket": False, "zero_bucket": True}
WAVE_KEYS = (
    "wave_tasks", "max_cap", "min_cap", "plan_factor", "rate_tokens_per_s",
    "prefill_tokens_per_s", "opening_tokens", "margin_s", "kv_pool_tokens", "setup_s",
)  # fmt: skip
# waves.dev_cap (task P5.1): None keeps the caps in development mode too.
WAVE_OPTIONAL = {"refill": False, "dev_cap": None}
REQUEST_KEYS = (
    "system_prompt", "reasoning_effort", "temperature", "max_turns", "forcing_tokens",
    "forcing_kind", "preserve_thinking", "context_tokens", "min_turn_tokens", "cell_timeout_s",
    "max_cells", "feedback_chars",
)  # fmt: skip
REQUEST_OPTIONAL = {"forcing_in_request": False, "thinking_end_note": "", "lenient_answers": False}
# "opening" (task P5.1): every task by ascending opening bound, NVARC's confidence unread.
OPENING_ORDER = "opening"
ORDERS = ("confidence", "buckets", OPENING_ORDER)
# The bucket edges of the queue order (docs/research/beyond_h_options.md, section 1.5): the 27B
# solved 2.5 of 7 shot tasks in (0, 0.5) and 3.5 of 53 at exactly 0; [0.5, 0.7) is unmeasured.
BUCKET_EDGE = 0.5
MID_EDGE = 0.7
# The tag that closes Qwen3.8's reasoning: vLLM's reasoning_end_str is the end note and this.
THINK_END = server_module.THINK_END
SANITY_KEYS = ("max_tokens", "reasoning_effort", "timeout_s")
DEGENERATE_KEYS = ("max_run", "special_min_id", "eos_ids", "wave_share")
ENGINE_KEYS = (
    "connect_timeout_s", "max_retries", "backoff_s", "backoff_max_s", "return_token_ids",
    "count_prompts",
)  # fmt: skip
PREP_KEYS = ("enabled", "imports", "check_timeout_s", "preflight", "scratch_dir")
LOGS_KEYS = ("dev_transcripts",)


# --- the direct-answer arm's stand-ins, moved from scripts/agent_spike.py without change ------


class NoRepl:
    """The direct-answer arm's REPL: none. A code block is answered with a note, never run."""

    def run_cell(self, code: str, timeout_s: float) -> Any:
        """Refuse a code block.

        Args:
            code: The block (not run).
            timeout_s: Unused.

        Returns:
            A ``CellResult`` with the note as its error and no ``transform``.
        """
        from arc26.agent.types import CellResult

        note = "no code interpreter in this mode: give the output grids as answer blocks"
        return CellResult("", note, False, 0.0, False)

    def close(self) -> None:
        """Nothing to stop."""


class NoVerifier:
    """The direct-answer arm verifies no program (``NoRepl`` never reports a ``transform``)."""

    replaced = 0

    def verify(self, task: Any, source: str) -> Any:
        """Refuse: the arm has no programs.

        Args:
            task: Unused.
            source: Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("the direct-answer arm verifies no program")

    def close(self) -> None:
        """Nothing to stop."""


# --- config -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WaveModel:
    r"""The waves' time and memory arithmetic (``configs/agent_shot.yaml``, ``waves``).

    A wave of :math:`n` tasks with thinking cap :math:`C` and longest opening :math:`o` is
    planned to take

    .. math::

        T(C, n, o) = \frac{C}{\phi r} + \frac{n (o + C)}{p} + \frac{F}{r} + m,

    the thinking turn at the planned rate, then the forced replies assuming every turn is cut:
    their prompts (the opening and the cut turn) prefilled at :math:`p`, and one forced reply of
    up to :math:`F` tokens generated at :math:`r`.

    Attributes:
        wave_tasks: Tasks per wave at most.
        max_cap: The largest thinking cap :math:`C` (P6.3-05's 94,208).
        min_cap: Below this cap no wave starts.
        plan_factor: :math:`\phi`, the share of the measured rate the plan assumes for the turn.
        rate_tokens_per_s: :math:`r`, the measured decode rate per stream while thinking.
        prefill_tokens_per_s: :math:`p`, the measured prefill rate of the forced replies.
        opening_tokens: :math:`o` for the fixed arithmetic (:func:`stage_arithmetic`); a wave's
            plan uses its own tasks' longest opening bound.
        margin_s: :math:`m`, seconds added to every planned wave.
        kv_pool_tokens: The KV pool when the boot reports none.
        setup_s: Seconds from the start to the first wave (preparation, boot, sanity).
        forcing_tokens: :math:`F`, the forced reply's cap (the request's ``forcing_tokens``).
        refill: Run a refill queue instead of waves (:func:`run_refill`).
        forcing_in_request: The answer comes inside the thinking request (the request's
            ``forcing_in_request``): no forced reply prefills the conversation again, so the
            :math:`n (o + C) / p` term is 0.
    """

    wave_tasks: int
    max_cap: int
    min_cap: int
    plan_factor: float
    rate_tokens_per_s: float
    prefill_tokens_per_s: float
    opening_tokens: int
    margin_s: float
    kv_pool_tokens: int
    setup_s: float
    forcing_tokens: int
    refill: bool = False
    forcing_in_request: bool = False

    def turn_seconds(self, cap: int) -> float:
        r"""Return the planned thinking turn, :math:`C / (\phi r)`.

        Args:
            cap: The thinking cap :math:`C`.

        Returns:
            Seconds.
        """
        return cap / (self.plan_factor * self.rate_tokens_per_s)

    def forcing_seconds(self, cap: int, tasks: int, opening: int) -> float:
        """Return the planned forced replies, every turn cut: :math:`n (o + C) / p + F / r`.

        Under ``forcing_in_request`` the answer is generated in the thinking request itself, so
        nothing is prefilled again and this is :math:`F / r`.

        Args:
            cap: The thinking cap :math:`C`.
            tasks: The wave's tasks :math:`n`.
            opening: Its longest opening :math:`o` in tokens.

        Returns:
            Seconds.
        """
        prefill = 0.0
        if not self.forcing_in_request:
            prefill = tasks * (opening + cap) / self.prefill_tokens_per_s
        return prefill + self.forcing_tokens / self.rate_tokens_per_s

    def wave_seconds(self, cap: int, tasks: int, opening: int) -> float:
        """Return a wave's planned length :math:`T(C, n, o)`.

        Args:
            cap: The thinking cap.
            tasks: The wave's tasks.
            opening: Its longest opening in tokens.

        Returns:
            Seconds.
        """
        return self.turn_seconds(cap) + self.forcing_seconds(cap, tasks, opening) + self.margin_s

    def largest_cap(self, left_s: float, tasks: int, opening: int) -> int:
        r"""Return the largest cap whose planned wave ends within ``left_s``, at most ``max_cap``.

        :math:`T` is linear in :math:`C`, so the cap is
        :math:`\lfloor (L - m - F/r - n o / p) / (1/(\phi r) + n/p) \rfloor`, lowered while
        float rounding would put :math:`T` past :math:`L` (under ``forcing_in_request`` both
        :math:`n/p` terms are 0).

        Args:
            left_s: :math:`L`, the seconds left before the generation stop.
            tasks: The wave's tasks :math:`n`.
            opening: Its longest opening :math:`o`.

        Returns:
            The cap, 0 when not even one token fits.
        """
        prefill = 0.0 if self.forcing_in_request else tasks / self.prefill_tokens_per_s
        per_token = 1 / (self.plan_factor * self.rate_tokens_per_s) + prefill
        fixed = self.wave_seconds(0, tasks, opening)
        if left_s <= fixed:
            return 0
        cap = min(self.max_cap, math.floor((left_s - fixed) / per_token))
        while cap > 0 and self.wave_seconds(cap, tasks, opening) > left_s:
            cap -= 1
        return max(cap, 0)

    def least_rate(self, cap: int, tasks: int, opening: int) -> float:
        r"""Return the lowest decode rate at which a wave planned at :math:`L = T(C)` still ends.

        A time-bound wave runs its turn and its forced reply's generation at the real rate
        :math:`r'`, its prefill as planned: it ends before the stop while
        :math:`(C + F) / r' \le C / (\phi r) + F / r + m`.

        Args:
            cap: The thinking cap.
            tasks: The wave's tasks.
            opening: Its longest opening.

        Returns:
            :math:`r'` in tokens per second per stream.
        """
        room = self.turn_seconds(cap) + self.forcing_tokens / self.rate_tokens_per_s
        return (cap + self.forcing_tokens) / (room + self.margin_s)


@dataclass(frozen=True)
class RequestSettings:
    """Each task's request, P6.3-05's but for the thinking cap (``configs/agent_shot.yaml``).

    Attributes:
        system_prompt: The system prompt's name in :data:`SYSTEM_PROMPTS`.
        reasoning_effort: The effort level.
        temperature: Sampling temperature.
        max_turns: Thinking turns per task.
        forcing_tokens: The answer-forcing reply's cap :math:`F`.
        forcing_kind: The forcing note (``answer2``: one or two answers per test input).
        preserve_thinking: Keep the cut reasoning in the forcing turn's context.
        context_tokens: The context per request (``server.max_model_len``).
        min_turn_tokens: No reply is requested with less room.
        cell_timeout_s: The loop's cell timeout (no cell runs).
        max_cells: Code blocks per reply (none runs).
        feedback_chars: Characters of one feedback message.
        forcing_in_request: Force the answer inside the thinking request (``Budget``'s).
        thinking_end_note: The phrase the server forces before ``</think>`` (``Budget``'s).
        lenient_answers: The loop's lenient answer parser and the implied-shape choice of
            :math:`S_1` (:func:`shot_answers`).
    """

    system_prompt: str
    reasoning_effort: str | None
    temperature: float
    max_turns: int
    forcing_tokens: int
    forcing_kind: str
    preserve_thinking: bool
    context_tokens: int
    min_turn_tokens: int
    cell_timeout_s: float
    max_cells: int
    feedback_chars: int
    forcing_in_request: bool = False
    thinking_end_note: str = ""
    lenient_answers: bool = False

    @property
    def system(self) -> str:
        """Return the system prompt's text."""
        return SYSTEM_PROMPTS[self.system_prompt]

    def budget(self, cap: int) -> Budget:
        """Return one task's attempt limits for a thinking cap.

        Args:
            cap: The thinking cap :math:`C`.

        Returns:
            P6.3-05's ``Budget`` with ``turn_tokens`` :math:`C` and ``attempt_tokens``
            :math:`C + F`; at :math:`C` 94,208 it equals P6.3-05's. Under
            ``forcing_in_request`` ``attempt_tokens`` is :math:`C + 2F`: the answer inside the
            thinking request and, should the server ignore the budget, the separate forcing
            turn's :math:`F`.
        """
        extra = self.forcing_tokens if self.forcing_in_request else 0
        return Budget(
            max_turns=self.max_turns,
            attempt_tokens=cap + self.forcing_tokens + extra,
            turn_tokens=cap,
            min_turn_tokens=self.min_turn_tokens,
            context_tokens=self.context_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            cell_timeout_s=self.cell_timeout_s,
            max_cells=self.max_cells,
            feedback_chars=self.feedback_chars,
            preserve_thinking=self.preserve_thinking,
            forcing_tokens=self.forcing_tokens,
            forcing_kind=self.forcing_kind,
            forcing_in_request=self.forcing_in_request,
            thinking_end_note=self.thinking_end_note,
            lenient_answers=self.lenient_answers,
        )


@dataclass(frozen=True)
class PrepSettings:
    """The engine preparation P6.3-05 ran between the install and the boot.

    Attributes:
        enabled: Run :func:`arc26.agent.engine_prep.prepare_engine`.
        imports: Modules the engine's workers import, checked in this order.
        check_timeout_s: Seconds for the import check.
        preflight: Check that torch from the venv sees the GPUs first.
        scratch_dir: Scratch for the empty ``sitecustomize`` (its ``engine_site`` folder).
    """

    enabled: bool
    imports: tuple[str, ...]
    check_timeout_s: float
    preflight: bool
    scratch_dir: str


@dataclass(frozen=True)
class ShotConfig:
    """The shot's run config (``configs/agent_shot.yaml``), checked.

    Attributes:
        seed: The run's seed.
        session_s: The session's length (Kaggle: 12 h).
        startup_margin_s: Development mode's default time gone before this process starts.
        final_margin_s: Seconds kept after the generation stop.
        grace_s: Seconds calls may run past the generation stop (0: none).
        max_confidence: Tasks at or above this confidence are never taken.
        order: The queue's order, ``confidence`` (least sure first, Submission 3) or
            ``buckets`` (:func:`select_tasks`).
        mid_bucket: Under ``buckets``, queue :math:`[0.5, 0.7)` before confidence 0.
        zero_bucket: Queue tasks at confidence exactly 0 (task P4.7b: False leaves them to the
            notebook's rank-256 tail).
        dev_cap: In development mode only, the thinking cap's ceiling (task P5.1): ``max_cap``
            and ``min_cap`` are lowered to it by :func:`for_mode`; None keeps them.
        waves: The wave arithmetic.
        request: Each task's request.
        sanity: The sanity request (``max_tokens``, ``reasoning_effort``, ``timeout_s``).
        degenerate: The degenerate rule (``max_run``, ``special_min_id``, ``eos_ids``,
            ``wave_share``).
        engine: ``EngineConfig`` fields of the client (timeouts and retries).
        return_token_ids: Ask vLLM for each reply's token ids.
        count_prompts: Count each opening prompt with ``/tokenize``.
        prep: The engine preparation.
        server: The engine server's settings.
        dev_transcripts: Write every conversation to a JSONL in development mode.
        source_sha256: The sha256 of the file it was read from (:func:`load_config`), or None.
    """

    seed: int
    session_s: float
    startup_margin_s: float
    final_margin_s: float
    grace_s: float
    max_confidence: float
    waves: WaveModel
    request: RequestSettings
    sanity: Mapping[str, Any]
    degenerate: Mapping[str, Any]
    engine: Mapping[str, Any]
    return_token_ids: bool
    count_prompts: bool
    prep: PrepSettings
    server: ServerConfig
    dev_transcripts: bool
    source_sha256: str | None = None
    order: str = "confidence"
    mid_bucket: bool = False
    zero_bucket: bool = True
    dev_cap: int | None = None


def _optional_section(
    raw: object, keys: Sequence[str], optional: Mapping[str, Any], where: str
) -> dict[str, Any]:
    """A section with its required keys checked and its optional keys filled with defaults."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    required = {k: v for k, v in raw.items() if str(k) not in optional}
    out = agent_run._section(required, keys, where)
    out.update({k: raw.get(k, default) for k, default in optional.items()})
    return out


def _flag(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false; got {value!r}")
    return value


def reasoning_end_str(extra_args: Sequence[str]) -> str | None:
    """Return the ``reasoning_end_str`` of a ``--reasoning-config`` among vLLM's extra flags.

    Args:
        extra_args: ``server.extra_args``.

    Returns:
        The string, or None when no such flag is there (or it names none).

    Raises:
        ConfigError: If the flag appears twice, has no value, or its value is not a JSON object
            (:func:`arc26.agent.server.reasoning_end_str`).
    """
    from arc26.agent.server import reasoning_end_str as parse

    try:
        return parse(extra_args)
    except ValueError as err:
        raise ConfigError(str(err)) from err


def _number(value: object, where: str, low: float, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not low <= value <= high:
        raise ConfigError(f"{where} must be a number from {low} to {high}; got {value!r}")
    return float(value)


def _integer(value: object, where: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ConfigError(f"{where} must be an integer of at least {low}; got {value!r}")
    return value


def config_from_mapping(raw: object) -> ShotConfig:
    """Check a parsed shot config and build its settings.

    Every key of every section is required and no other key is accepted, so a typo fails at once
    instead of running with a default.

    Args:
        raw: The parsed YAML.

    Returns:
        The config.

    Raises:
        ConfigError: If a section or a key is missing or unknown, or a value is out of range.
    """
    section = agent_run._section
    top = section(raw, TOP_KEYS, "config")
    try:
        deadline = section(top["deadline"], DEADLINE_KEYS, "deadline")
        selection = _optional_section(
            top["selection"], SELECTION_KEYS, SELECTION_OPTIONAL, "selection"
        )
        waves = _optional_section(top["waves"], WAVE_KEYS, WAVE_OPTIONAL, "waves")
        request = _optional_section(top["request"], REQUEST_KEYS, REQUEST_OPTIONAL, "request")
        sanity = section(top["sanity"], SANITY_KEYS, "sanity")
        degenerate = section(top["degenerate"], DEGENERATE_KEYS, "degenerate")
        engine = section(top["engine"], ENGINE_KEYS, "engine")
        prep = section(top["engine_prep"], PREP_KEYS, "engine_prep")
        server = section(top["server"], agent_run.SERVER_KEYS, "server")
        logs = section(top["logs"], LOGS_KEYS, "logs")
        server["extra_args"] = tuple(str(a) for a in server["extra_args"] or ())
        server["unset_env"] = tuple(str(a) for a in server["unset_env"] or ())
        server["env"] = {str(k): str(v) for k, v in (server["env"] or {}).items()}
        server_config = ServerConfig(**server)
        model = WaveModel(
            wave_tasks=_integer(waves["wave_tasks"], "waves.wave_tasks", 1),
            max_cap=_integer(waves["max_cap"], "waves.max_cap", 1),
            min_cap=_integer(waves["min_cap"], "waves.min_cap", 1),
            plan_factor=_number(waves["plan_factor"], "waves.plan_factor", 1e-6, 1.0),
            rate_tokens_per_s=_number(waves["rate_tokens_per_s"], "waves.rate_tokens_per_s", 1e-6),
            prefill_tokens_per_s=_number(
                waves["prefill_tokens_per_s"], "waves.prefill_tokens_per_s", 1e-6
            ),
            opening_tokens=_integer(waves["opening_tokens"], "waves.opening_tokens", 0),
            margin_s=_number(waves["margin_s"], "waves.margin_s", 0.0),
            kv_pool_tokens=_integer(waves["kv_pool_tokens"], "waves.kv_pool_tokens", 1),
            setup_s=_number(waves["setup_s"], "waves.setup_s", 0.0),
            forcing_tokens=_integer(request["forcing_tokens"], "request.forcing_tokens", 1),
            refill=_flag(waves["refill"], "waves.refill"),
            forcing_in_request=_flag(request["forcing_in_request"], "request.forcing_in_request"),
        )
        note = request["thinking_end_note"]
        if not isinstance(note, str) or "<think>" in note or THINK_END in note:
            raise ConfigError("request.thinking_end_note must be a string without think tags")
        if request["system_prompt"] not in SYSTEM_PROMPTS:
            raise ConfigError(f"request.system_prompt must be one of {sorted(SYSTEM_PROMPTS)}")
        for where, effort in (("request", request["reasoning_effort"]),
                              ("sanity", sanity["reasoning_effort"])):  # fmt: skip
            if effort not in EFFORTS:
                raise ConfigError(f"{where}.reasoning_effort must be one of {EFFORTS}")
        settings = RequestSettings(
            system_prompt=str(request["system_prompt"]),
            reasoning_effort=request["reasoning_effort"],
            temperature=_number(request["temperature"], "request.temperature", 0.0),
            max_turns=_integer(request["max_turns"], "request.max_turns", 1),
            forcing_tokens=_integer(request["forcing_tokens"], "request.forcing_tokens", 1),
            forcing_kind=str(request["forcing_kind"]),
            preserve_thinking=bool(request["preserve_thinking"]),
            context_tokens=_integer(request["context_tokens"], "request.context_tokens", 1),
            min_turn_tokens=_integer(request["min_turn_tokens"], "request.min_turn_tokens", 1),
            cell_timeout_s=_number(request["cell_timeout_s"], "request.cell_timeout_s", 1e-6),
            max_cells=_integer(request["max_cells"], "request.max_cells", 1),
            feedback_chars=_integer(request["feedback_chars"], "request.feedback_chars", 1),
            forcing_in_request=model.forcing_in_request,
            thinking_end_note=note,
            lenient_answers=_flag(request["lenient_answers"], "request.lenient_answers"),
        )
        if selection["order"] not in ORDERS:
            raise ConfigError(f"selection.order must be one of {ORDERS}")
        _flag(selection["mid_bucket"], "selection.mid_bucket")
        _flag(selection["zero_bucket"], "selection.zero_bucket")
        full = settings.budget(model.max_cap)  # raises ValueError on limits that cannot work
        settings.budget(model.min_cap)
        dev_cap = waves["dev_cap"]
        if dev_cap is not None:
            dev_cap = _integer(dev_cap, "waves.dev_cap", 1)
            if dev_cap > model.max_cap:
                raise ConfigError("waves.dev_cap must not exceed waves.max_cap")
            settings.budget(dev_cap)
        imports = prep["imports"]
        if not isinstance(imports, list) or not all(isinstance(m, str) and m for m in imports):
            raise ConfigError("engine_prep.imports must be a list of module names")
        eos = degenerate["eos_ids"]
        if not isinstance(eos, list) or not all(_integer(e, "eos_ids", 0) >= 0 for e in eos):
            raise ConfigError("degenerate.eos_ids must be a list of token ids")
        client = {k: engine[k] for k in ("connect_timeout_s", "max_retries", "backoff_s",
                                         "backoff_max_s")}  # fmt: skip
        from arc26.agent.engine import EngineConfig

        EngineConfig(**client)  # raises ValueError on settings that cannot work
        config = ShotConfig(
            seed=_integer(top["seed"], "seed", 0),
            session_s=_number(deadline["session_s"], "deadline.session_s", 1.0),
            startup_margin_s=_number(deadline["startup_margin_s"], "deadline.startup_margin_s", 0),
            final_margin_s=_number(deadline["final_margin_s"], "deadline.final_margin_s", 0.0),
            grace_s=_number(deadline["grace_s"], "deadline.grace_s", 0.0),
            max_confidence=_number(selection["max_confidence"], "selection.max_confidence", 0, 1),
            waves=model,
            request=settings,
            sanity={
                "max_tokens": _integer(sanity["max_tokens"], "sanity.max_tokens", 16),
                "reasoning_effort": sanity["reasoning_effort"],
                "timeout_s": _number(sanity["timeout_s"], "sanity.timeout_s", 1e-6),
            },
            degenerate={
                "max_run": _integer(degenerate["max_run"], "degenerate.max_run", 1),
                "special_min_id": _integer(degenerate["special_min_id"], "special_min_id", 0),
                "eos_ids": [int(e) for e in eos],
                "wave_share": _number(degenerate["wave_share"], "degenerate.wave_share", 0, 1),
            },
            engine=client,
            return_token_ids=bool(engine["return_token_ids"]),
            count_prompts=bool(engine["count_prompts"]),
            prep=PrepSettings(
                enabled=bool(prep["enabled"]),
                imports=tuple(imports),
                check_timeout_s=_number(prep["check_timeout_s"], "check_timeout_s", 1e-6),
                preflight=bool(prep["preflight"]),
                scratch_dir=str(prep["scratch_dir"]),
            ),
            server=server_config,
            dev_transcripts=bool(logs["dev_transcripts"]),
            order=str(selection["order"]),
            mid_bucket=selection["mid_bucket"],
            zero_bucket=selection["zero_bucket"],
            dev_cap=dev_cap,
        )
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError) as err:
        raise ConfigError(f"config: {err}") from err
    if config.session_s <= config.startup_margin_s + config.final_margin_s:
        raise ConfigError("deadline: session_s must exceed startup_margin_s + final_margin_s")
    if model.min_cap > model.max_cap:
        raise ConfigError("waves: min_cap must not exceed max_cap")
    if model.wave_tasks > server_config.max_num_seqs:
        raise ConfigError("waves.wave_tasks must not exceed server.max_num_seqs")
    if settings.context_tokens != server_config.max_model_len:
        raise ConfigError("request.context_tokens must equal server.max_model_len")
    if model.max_cap + full.in_request_tokens() + full.forcing_reserve() >= settings.context_tokens:
        raise ConfigError("waves.max_cap plus the forcing turn's reserve must fit the context")
    end = reasoning_end_str(server_config.extra_args)
    if model.forcing_in_request and end != f"{settings.thinking_end_note}{THINK_END}":
        raise ConfigError(
            "request.forcing_in_request needs server.extra_args to hold --reasoning-config with "
            "reasoning_end_str equal to request.thinking_end_note followed by </think>"
        )
    if config.mid_bucket and (config.order != "buckets" or config.max_confidence < MID_EDGE):
        raise ConfigError(
            f"selection.mid_bucket needs order buckets and max_confidence of at least {MID_EDGE}"
        )
    if config.order == OPENING_ORDER and not config.zero_bucket:
        raise ConfigError(
            "selection.order opening queues every task, so selection.zero_bucket must be true"
        )
    return config


def for_mode(config: ShotConfig, mode: str) -> ShotConfig:
    """Return the config a run of this mode uses: ``waves.dev_cap`` applied in development mode.

    Why: a version's own Kaggle run in development mode is a short check of the build that the
    scored rerun runs in competition mode (task P5.1), so its thinking cap is lowered there and
    only there; a competition rerun keeps ``max_cap`` and ``min_cap`` as configured.

    Args:
        config: The checked config.
        mode: The run's mode (:data:`arc26.agent.run.DEV` or competition).

    Returns:
        The config, with ``max_cap`` and ``min_cap`` at most ``dev_cap`` in development mode
        when ``dev_cap`` is set; else ``config`` itself.
    """
    if mode != DEV or config.dev_cap is None:
        return config
    model = config.waves
    cap = config.dev_cap
    waves = dataclasses.replace(
        model, max_cap=min(model.max_cap, cap), min_cap=min(model.min_cap, cap)
    )
    return dataclasses.replace(config, waves=waves)


def load_config(path: str | os.PathLike[str]) -> ShotConfig:
    """Read and check a shot config.

    Args:
        path: The YAML file (or its JSON copy, which the notebook writes).

    Returns:
        The config.

    Raises:
        ConfigError: If the file cannot be read or parsed, or :func:`config_from_mapping`
            refuses it.
    """
    import yaml

    try:
        data = Path(path).read_bytes()
        raw = yaml.safe_load(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        raise ConfigError(f"cannot read the config {path}: {err}") from err
    config = config_from_mapping(raw)
    return dataclasses.replace(config, source_sha256=hashlib.sha256(data).hexdigest())


def stage_arithmetic(config: ShotConfig) -> dict[str, float]:
    """Return the shot's fixed times, for the notebook's ``min_stage_s`` and the reports.

    Every wave here has ``wave_tasks`` tasks and the config's ``opening_tokens``.

    Args:
        config: The config.

    Returns:
        ``setup_s``; ``full_wave_s`` and ``smallest_wave_s``, :math:`T` at ``max_cap`` and
        ``min_cap``; ``min_shot_s``, the setup plus the smallest wave (the least time before the
        generation stop with which this process starts an engine); ``final_margin_s``; and
        ``least_rate_full`` and ``least_rate_smallest``, the lowest decode rates per stream at
        which a time-bound wave at those caps still ends before the stop
        (:meth:`WaveModel.least_rate`).
    """
    model = config.waves
    n, o = model.wave_tasks, model.opening_tokens
    smallest = model.wave_seconds(model.min_cap, n, o)
    return {
        "setup_s": model.setup_s,
        "full_wave_s": round(model.wave_seconds(model.max_cap, n, o), 1),
        "full_turn_s": round(model.turn_seconds(model.max_cap), 1),
        "full_forcing_s": round(model.forcing_seconds(model.max_cap, n, o), 1),
        "smallest_wave_s": round(smallest, 1),
        "min_shot_s": round(model.setup_s + smallest, 1),
        "final_margin_s": config.final_margin_s,
        "least_rate_full": round(model.least_rate(model.max_cap, n, o), 2),
        "least_rate_smallest": round(model.least_rate(model.min_cap, n, o), 2),
    }


# --- selection: NVARC's confidence --------------------------------------------------------------


def output_key(name: str) -> tuple[str, int] | None:
    """Return ``(task_id, test_index)`` of a candidate file name ``<task>_<index>.<view ops>``.

    Args:
        name: The key in ``candidates.json`` (a path under NVARC's output folder).

    Returns:
        The output, or None for a name of another form.
    """
    base = name.rsplit("/", 1)[-1].split(".", 1)[0]
    task_id, sep, index = base.rpartition("_")
    if not sep or not task_id or not index.isdigit():
        return None
    return task_id, int(index)


def output_confidence(samples: Sequence[Any]) -> float:
    r"""Return :math:`\hat p` of one output's rank 1 grid, as ``calibration_report.py`` computes.

    The output's view candidates are tallied by ``arc26.loop.tally`` (the moved ``score_kgmon``
    selection, :math:`s(g) = |C_g| - \frac{1}{|C_g|} \sum_{c \in C_g} \bar s(c)`), and

    .. math::

        \hat p = \frac{1}{\sum_g e^{(s(g) - s_1)/T}}, \qquad T = 1,

    with :math:`s_1` the rank 1 grid's score.

    Args:
        samples: The output's candidates as ``arc26.loop.Candidate`` objects, in the order of
            their file names (``sorted``), then their order in each file.

    Returns:
        :math:`\hat p`, 0 without candidates or when a score is not finite (NaN or infinite
        ``score_aug``): such an output counts as least sure, so the task's minimum does not
        depend on the order of its outputs.
    """
    if not samples:
        return 0.0
    from arc26.loop import tally

    votes = tally(list(samples))
    if not votes.scores:
        return 0.0
    top = max(votes.scores)
    weights = [math.exp((s - top) / TEMPERATURE) for s in votes.scores]
    p = weights[0] / sum(weights)
    # A non-finite score (a NaN or infinite score_aug) gives no confidence: 0, least sure.
    return p if math.isfinite(p) else 0.0


def _placeholder(nvarc: Mapping[str, Sequence[Any]] | None, task_id: str, index: int) -> bool:
    """Whether NVARC's attempt 1 for a test output is its placeholder (no candidate, no file)."""
    if nvarc is None:
        return True
    pairs = list(nvarc.get(task_id, ()))
    return index >= len(pairs) or pairs[index][0] is None


def shot_confidence(
    counts: Mapping[str, int],
    candidates: Mapping[str, Any] | None,
    nvarc: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, float | None]:
    """Return NVARC's confidence in each task: P6.3-05's rule from ``candidates.json``.

    Per test output :func:`output_confidence` (``softmax_p`` of the rank 1 grid in
    ``reports/P2.8/<run>.csv``). An output without candidates is 0 where NVARC's attempt 1 is
    its placeholder (no candidate, P6.3-05's 0; also every output when NVARC's file is missing),
    and unknown where NVARC gave a real attempt (a partial file). Per task the minimum over its
    known outputs, since a task scores only when every output is right: the true minimum is at
    most that, so a task it puts under ``max_confidence`` is under it. A task with no known
    output is unknown (None) and never taken. This is not the scheduler's ``nvarc_confidence``
    (agreement, mean over outputs).

    Args:
        counts: Task id to its number of test inputs.
        candidates: The parsed candidates file (``<task>_<i>.<ops>`` to a list of
            ``{"solution", "beam_score", "score_aug"}``), or None (every output without
            candidates).
        nvarc: NVARC's attempts per task (``nvarc_from_submission``), or None.

    Returns:
        Task id to a confidence in [0, 1], or None when unknown.
    """
    from arc26.loop import Candidate as ViewCandidate

    grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for name in sorted(candidates or {}):
        key = output_key(str(name))
        samples = (candidates or {})[name]
        if key is None or key[0] not in counts or not isinstance(samples, list):
            continue
        for sample in samples:
            try:
                grouped[key].append(
                    ViewCandidate(
                        sample["solution"],
                        sample.get("beam_score", 0.0),
                        sample["score_aug"],
                        str(name),
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    out: dict[str, float | None] = {}
    for task_id, count in counts.items():
        known = []
        for j in range(count):
            found = grouped.get((task_id, j))
            if found:
                known.append(output_confidence(found))
            elif _placeholder(nvarc, task_id, j):
                known.append(0.0)
        out[task_id] = min(known) if known else None
    return out


def confidence_without_candidates(
    counts: Mapping[str, int], nvarc: Mapping[str, Sequence[Any]] | None
) -> dict[str, float | None]:
    """Return the confidence when no candidates file is readable: known only where NVARC had none.

    Why: without ``candidates.json`` P6.3-05's rule would give every output 0 and rank tasks
    NVARC answered with confidence among the shot's by id alone. An output NVARC left without an
    attempt has no candidate, so its 0 is P6.3-05's; any other output is unknown. So a task with
    an empty output is 0 (the queue then runs by task id) and the others are unknown, never taken.
    It equals :func:`shot_confidence` with no candidates, without reading any candidate, so it
    also serves when that function fails on an odd file.

    Args:
        counts: Task id to its number of test inputs.
        nvarc: NVARC's attempts per task, or None (then NVARC answered nothing: every task 0).

    Returns:
        Task id to 0, or None when unknown.
    """
    return {
        task_id: 0.0 if any(_placeholder(nvarc, task_id, j) for j in range(count)) else None
        for task_id, count in counts.items()
    }


def confidence_bucket(confidence: float, mid_bucket: bool = False) -> int | None:
    r"""Return a task's place in the ``buckets`` order, or None when that order never takes it.

    Args:
        confidence: NVARC's confidence :math:`c`.
        mid_bucket: Whether :math:`[0.5, 0.7)` is queued.

    Returns:
        0 for :math:`0 < c < 0.5` (no right attempt 2 of NVARC's there, the 27B's best measured
        rate), 1 for :math:`0.5 \le c < 0.7` under ``mid_bucket`` (option 2c's pre-registered
        place, before confidence 0), 2 for :math:`c = 0`, else None.
    """
    if 0 < confidence < BUCKET_EDGE:
        return 0
    if mid_bucket and BUCKET_EDGE <= confidence < MID_EDGE:
        return 1
    if confidence == 0:
        return 2
    return None


def select_tasks(
    confidence: Mapping[str, float | None],
    max_confidence: float,
    exclude: Iterable[str] = (),
    order: str = "confidence",
    mid_bucket: bool = False,
    zero_bucket: bool = True,
    openings: Mapping[str, int] | None = None,
) -> list[tuple[str, float]]:
    """Return the shot's queue: tasks under ``max_confidence``, in the configured order.

    ``confidence`` (Submission 3): least sure first, ties by id. ``buckets`` (task P6.9, option
    2a): by :func:`confidence_bucket`, then least sure first, then id: :math:`(0, 0.5)`, then
    :math:`[0.5, 0.7)` under ``mid_bucket``, then confidence 0; a task outside every bucket is
    not taken. Without ``zero_bucket`` (task P4.7b) no task at confidence exactly 0 is taken, in
    either order: the notebook's rank-256 tail takes those. ``opening`` (task P5.1): ascending
    opening bound, then id, the smallest prompts first; :func:`run` gives every task confidence
    0 in that order, so every task is queued.

    Args:
        confidence: Task id to NVARC's confidence, None when unknown (never taken).
        max_confidence: Tasks at or above this are never taken.
        exclude: Tasks already shot (a resumed run).
        order: ``confidence`` or ``buckets``.
        mid_bucket: Under ``buckets``, queue :math:`[0.5, 0.7)` before confidence 0.
        zero_bucket: Take tasks at confidence exactly 0.
        openings: Task id to its opening bound in tokens; the ``opening`` order needs it.

    Returns:
        ``(task_id, confidence)`` in the order the waves take them.

    Raises:
        ValueError: If ``order`` is unknown, or ``opening`` comes without ``openings``.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}; got {order!r}")
    if order == OPENING_ORDER and openings is None:
        raise ValueError("order opening needs the tasks' opening bounds")
    skip = set(exclude)
    chosen = [
        (t, c)
        for t, c in confidence.items()
        if c is not None and c < max_confidence and t not in skip and (zero_bucket or c > 0)
    ]
    if order == "confidence":
        return sorted(chosen, key=lambda item: (item[1], item[0]))
    if order == OPENING_ORDER:
        return sorted(chosen, key=lambda item: (openings[item[0]], item[0]))
    ranked = [(confidence_bucket(c, mid_bucket), c, t) for t, c in chosen]
    return [(t, c) for _, c, t in sorted(r for r in ranked if r[0] is not None)]


# --- the wave plan ------------------------------------------------------------------------------


def opening_bound(task: AgentTask, system: str) -> int:
    """Return an upper bound of a task's opening prompt in tokens: one per character plus slack.

    Why: the plan must know the KV need before any request; the loop's own first bound (one
    token per character, as ``arc26.agent.loop`` bounds it) over-counts 1.28 times on P6.3-05's
    15 openings (57,627 against 45,116 ``/tokenize`` tokens), so the plan is safe.

    Args:
        task: The task.
        system: The system prompt.

    Returns:
        Tokens.
    """
    messages = prompts.initial_messages(task, system=system)
    return sum(len(m.content) + MESSAGE_SLACK for m in messages) + PROMPT_SLACK


@dataclass(frozen=True)
class WavePlan:
    """One wave's plan, or why no wave starts.

    Attributes:
        tasks: The wave's tasks, empty when none starts.
        cap: The thinking cap :math:`C`, or the largest the time allows when below ``min_cap``.
        planned_s: The planned length :math:`T(C)`, 0 when none starts.
        turn_s: Its thinking turn.
        forcing_s: Its forced replies.
        left_s: The seconds left before the generation stop.
        kv_need: The KV need of the wave's tasks.
        kv_pool: The pool it had to fit.
        dropped: Tasks the pool left for a later wave.
        reason: Why no wave starts, or why it is smaller than ``wave_tasks``; None otherwise.
        opening: The wave's longest opening bound :math:`o` in tokens.
    """

    tasks: tuple[str, ...]
    cap: int
    planned_s: float
    turn_s: float
    forcing_s: float
    left_s: float
    kv_need: int
    kv_pool: int
    dropped: tuple[str, ...] = ()
    reason: str | None = None
    opening: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return the plan for ``shot_plan.json``."""
        return {
            "tasks": list(self.tasks),
            "cap": self.cap,
            "planned_s": round(self.planned_s, 1),
            "turn_s": round(self.turn_s, 1),
            "forcing_s": round(self.forcing_s, 1),
            "left_s": round(self.left_s, 1),
            "kv_need": self.kv_need,
            "kv_pool": self.kv_pool,
            "dropped": list(self.dropped),
            "reason": self.reason,
            "opening": self.opening,
        }


def plan_wave(
    queue: Sequence[tuple[str, int]],
    left_s: float,
    model: WaveModel,
    kv_pool: int,
) -> WavePlan:
    r"""Plan the next wave: the cap from the time left, then as many tasks as the pool holds.

    With :math:`n` the queue's first ``wave_tasks`` tasks and :math:`o` their longest opening
    bound, the cap is :meth:`WaveModel.largest_cap`; below ``min_cap`` no wave starts. While the
    KV need :math:`\sum_i (o_i + C + F)` exceeds the pool, the wave drops its last task and the cap
    is planned again for the smaller wave (it can only grow, up to ``max_cap``); the dropped tasks
    stay first in the queue for a later wave.

    Args:
        queue: ``(task_id, opening bound o_i)`` in the shot's order.
        left_s: Seconds left before the generation stop.
        model: The wave arithmetic.
        kv_pool: The KV pool the boot reported, in tokens.

    Returns:
        The plan.
    """
    head = list(queue[: model.wave_tasks])
    if not head:
        return WavePlan((), 0, 0.0, 0.0, 0.0, left_s, 0, kv_pool, reason="no tasks left")
    taken = list(head)
    forcing = model.forcing_tokens
    while True:
        opening = max(o for _, o in taken)
        cap = model.largest_cap(left_s, len(taken), opening)
        if cap < model.min_cap:
            return WavePlan(
                (), cap, 0.0, 0.0, 0.0, left_s, 0, kv_pool,
                reason=f"time: {left_s:.0f} s left allow a cap of {cap} for {len(taken)} tasks, "
                f"under min_cap {model.min_cap}",
            )  # fmt: skip
        need = sum(o + cap + forcing for _, o in taken)
        if need <= kv_pool:
            break
        taken.pop()
        if not taken:
            reason = f"kv: one task needs more than the pool's {kv_pool} tokens at cap {cap}"
            dropped = tuple(t for t, _ in head)
            return WavePlan((), cap, 0.0, 0.0, 0.0, left_s, 0, kv_pool, dropped, reason)
    dropped = tuple(t for t, _ in head[len(taken) :])
    reason = f"kv: {len(dropped)} tasks left for a later wave to fit the pool" if dropped else None
    n = len(taken)
    return WavePlan(
        tasks=tuple(t for t, _ in taken),
        cap=cap,
        planned_s=model.wave_seconds(cap, n, opening),
        turn_s=model.turn_seconds(cap),
        forcing_s=model.forcing_seconds(cap, n, opening),
        left_s=left_s,
        kv_need=need,
        kv_pool=kv_pool,
        dropped=dropped,
        reason=reason,
        opening=opening,
    )


def plan_start(
    item: tuple[str, int], left_s: float, model: WaveModel, kv_free: int, streams: int
) -> WavePlan:
    """Plan one task's start in the refill queue: a wave of one with the streams' arithmetic.

    The cap is :meth:`WaveModel.largest_cap` for :math:`n` the streams running once it starts,
    so the forced replies' prefill (without in-request forcing) still assumes every running
    turn is cut at once, the meaning of a wave's plan. Below ``min_cap`` it does not start; its
    KV need :math:`o + C + F` must fit what the running tasks leave of the pool.

    Args:
        item: ``(task_id, opening bound o)``.
        left_s: Seconds left before the generation stop.
        model: The wave arithmetic.
        kv_free: The pool less the running tasks' needs, in tokens.
        streams: The streams running once this task starts (itself included).

    Returns:
        The plan of a one-task wave, or no tasks with the reason (``time: ...`` or ``kv: ...``).
    """
    task_id, opening = item
    cap = model.largest_cap(left_s, streams, opening)
    if cap < model.min_cap:
        return WavePlan(
            (), cap, 0.0, 0.0, 0.0, left_s, 0, kv_free,
            reason=f"time: {left_s:.0f} s left allow a cap of {cap} at {streams} streams, under "
            f"min_cap {model.min_cap}",
        )  # fmt: skip
    need = opening + cap + model.forcing_tokens
    if need > kv_free:
        return WavePlan(
            (), cap, 0.0, 0.0, 0.0, left_s, need, kv_free, (task_id,),
            f"kv: {task_id} needs {need} tokens at cap {cap}, {kv_free} free",
        )  # fmt: skip
    return WavePlan(
        tasks=(task_id,),
        cap=cap,
        planned_s=model.wave_seconds(cap, streams, opening),
        turn_s=model.turn_seconds(cap),
        forcing_s=model.forcing_seconds(cap, streams, opening),
        left_s=left_s,
        kv_need=need,
        kv_pool=kv_free,
        opening=opening,
    )


# --- the merge ----------------------------------------------------------------------------------


def implied_shapes(task: AgentTask) -> tuple[tuple[int, int] | None, ...]:
    """Return the output shape the demonstrations imply for each test input, if any.

    Why: option 4b's choice of :math:`S_1` (:func:`shot_answers`); run ``P6.3-06`` wrote drafts
    of other sizes before (or after) the answer of the size its task implies.

    Args:
        task: The task.

    Returns:
        Per test input: its own shape when every demonstration's output has its input's shape;
        else the one shape every demonstration output has; else None.
    """

    def shape(grid: Grid) -> tuple[int, int]:
        return len(grid), len(grid[0]) if grid else 0

    pairs = task.train
    if pairs and all(shape(p.output) == shape(p.input) for p in pairs):
        return tuple(shape(g) for g in task.test_inputs)
    outputs = {shape(p.output) for p in pairs}
    if len(outputs) == 1:
        return (next(iter(outputs)),) * len(task.test_inputs)
    return (None,) * len(task.test_inputs)


def shot_answers(
    records: Iterable[AttemptRecord],
    test_index: int,
    implied: tuple[int, int] | None = None,
) -> list[Grid]:
    """Return the shot's two answers :math:`S_1, S_2` for one test output.

    Without ``implied`` (Submission 3): the first two different valid answers, in order. With
    the shape the demonstrations imply (``request.lenient_answers``, option 4b), the model's own
    order is kept (the prompt asks for the best first) but a draft of another size is skipped:
    :math:`S_1` is the first valid answer of that shape, or the first valid answer when none
    has it; :math:`S_2` the next valid answer after it that differs, else the first one before
    it that differs.

    Args:
        records: The task's usable shot records.
        test_index: The test output.
        implied: The output shape :func:`implied_shapes` gives, or None.

    Returns:
        :math:`S_1, S_2` (fewer when the replies gave fewer).
    """
    from arc26.submit import grid_problem

    valid = [
        [list(row) for row in grid]
        for record in records
        for index, grid in record.direct_answers
        if index == test_index and grid_problem(grid) is None
    ]
    if implied is None:
        found: list[Grid] = []
        for grid in valid:
            if grid not in found:
                found.append(grid)
                if len(found) == 2:
                    break
        return found
    if not valid:
        return []
    at = next((k for k, g in enumerate(valid) if (len(g), len(g[0])) == implied), 0)
    first = valid[at]
    later = [g for g in valid[at + 1 :] if g != first]
    second = later[0] if later else next((g for g in valid[:at] if g != first), None)
    return [first] if second is None else [first, second]


def merge_output(
    n1: Grid | None, n2: Grid | None, s1: Grid | None, s2: Grid | None
) -> tuple[tuple[Grid | None, str], tuple[Grid | None, str]]:
    r"""Merge NVARC's attempts and the shot's answers for one test output (design point 5).

    No :math:`S_1`: :math:`(N_1, N_2)`. :math:`N_1` real and :math:`S_1 \ne N_1`:
    :math:`(N_1, S_1)`. :math:`N_1` real and :math:`S_1 = N_1`: :math:`(N_1, N_2)`, or
    :math:`(N_1, S_2)` when :math:`N_2` is a placeholder and :math:`S_2` exists. :math:`N_1` a
    placeholder: :math:`(S_1, S_2)`, or :math:`(S_1, N_2)` without :math:`S_2`.

    Args:
        n1: NVARC's attempt 1, None when it is NVARC's placeholder (no candidate).
        n2: NVARC's attempt 2, None when missing.
        s1: The shot's first answer, or None.
        s2: Its second, different answer, or None.

    Returns:
        The two slots, each a grid (None for an empty slot, written as ``[[0]]``) and its source,
        ``nvarc`` or ``shot``.
    """
    nvarc = ((n1, "nvarc"), (n2, "nvarc"))
    if s1 is None:
        return nvarc
    if n1 is not None:
        if s1 != n1:
            return (n1, "nvarc"), (s1, "shot")
        # A placeholder slot never beats a real answer: S2 takes an empty N2's place.
        return nvarc if n2 is not None or s2 is None else ((n1, "nvarc"), (s2, "shot"))
    if s2 is not None:
        return (s1, "shot"), (s2, "shot")
    return (s1, "shot"), (n2, "nvarc")


# The shot's answers are direct answers (arc26.agent.types.CandidateSource); slots() reports them
# as "shot".
_SOURCE = {"nvarc": "nvarc", "shot": "direct"}


def _entry(grids: Sequence[Grid | None]) -> dict[str, Grid]:
    """One test input's ``{"attempt_1", "attempt_2"}``, ``[[0]]`` for a missing grid."""
    from arc26.submit import ATTEMPTS, fallback_grid

    return {a: fallback_grid() if g is None else g for a, g in zip(ATTEMPTS, grids, strict=True)}


class ShotWriter(SubmissionWriter):
    """Keeps ``submission.json`` valid: NVARC's attempts, merged with the shot's answers.

    :class:`~arc26.agent.scheduler.SubmissionWriter`'s atomic, validated rewrite after every
    record, with :func:`merge_output` instead of P6.2's selection. A task without a usable shot
    record keeps NVARC's two attempts in their slots, a missing or invalid one written as
    ``[[0]]``.

    Args:
        path: The submission file.
        test_counts: Task id to its number of test inputs.
        nvarc: NVARC's attempts per task, or None.
        records: Usable shot records already logged (a resumed run).
        shapes: Per task the implied output shapes (:func:`implied_shapes`) for option 4b's
            choice of :math:`S_1`, or None for Submission 3's.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        test_counts: Mapping[str, int],
        nvarc: Mapping[str, Sequence[Any]] | None = None,
        records: Iterable[AttemptRecord] = (),
        shapes: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    ) -> None:
        # Set before the base class selects the resumed records.
        self.shapes = None if shapes is None else {k: tuple(v) for k, v in shapes.items()}
        super().__init__(path, test_counts, nvarc, records)

    def implied(self, task_id: str, test_index: int) -> tuple[int, int] | None:
        """Return the implied shape of one test output, None without option 4b or a shape.

        Args:
            task_id: The task.
            test_index: The test output.

        Returns:
            The shape, or None.
        """
        found = (self.shapes or {}).get(task_id, ())
        return found[test_index] if test_index < len(found) else None

    def _select(self, task_id: str) -> list[Any]:
        count = self.counts[task_id]
        pairs = list(self.nvarc.get(task_id, ()))
        own = self._records.get(task_id, [])
        slots = []
        for j in range(count):
            n1, n2 = pairs[j] if j < len(pairs) else (None, None)
            shot = shot_answers(own, j, self.implied(task_id, j)) + [None, None]
            merged = merge_output(n1, n2, shot[0], shot[1])
            slots.append(
                [
                    None
                    if grid is None
                    else Candidate(task_id, j, grid, _SOURCE[source], 1, 0, False)
                    for grid, source in merged
                ]
            )
        return slots

    def submission(self) -> dict[str, list[dict[str, Grid]]]:
        """Return the current submission, every slot in its place.

        Returns:
            Task id to one ``{"attempt_1", "attempt_2"}`` per test input.
        """
        from arc26.submit import ATTEMPTS, fallback_grid

        out = {}
        for task_id in self.counts:
            rows = []
            for slots in self._selection.get(task_id) or []:
                grids = [fallback_grid() if c is None else c.grid for c in slots]
                rows.append(dict(zip(ATTEMPTS, grids, strict=True)))
            out[task_id] = rows
        return out

    def nvarc_submission(self) -> dict[str, list[dict[str, Grid]]]:
        """Return NVARC's attempts alone, written as :meth:`submission` writes them.

        Returns:
            Task id to one ``{"attempt_1", "attempt_2"}`` per test input.
        """
        out = {}
        for task_id, count in self.counts.items():
            pairs = list(self.nvarc.get(task_id, ()))
            rows = []
            for j in range(count):
                grids = pairs[j] if j < len(pairs) else (None, None)
                rows.append(_entry(grids))
            out[task_id] = rows
        return out

    def slots(self) -> dict[str, int]:
        """Return how many attempt slots each source fills now.

        Returns:
            Counts for ``nvarc``, ``shot``, and ``empty``.
        """
        counts = {"nvarc": 0, "shot": 0, "empty": 0}
        for task_id in self.counts:
            for slots in self._selection.get(task_id) or []:
                for candidate in slots:
                    key = "empty" if candidate is None else candidate.source
                    counts["shot" if key == "direct" else key] += 1
        return counts


# --- logs ---------------------------------------------------------------------------------------


def read_shot_log(path: Path) -> tuple[list[AttemptRecord], list[dict[str, Any]], int]:
    """Read ``shot_attempts.jsonl``: the records, their rows, and the lines skipped.

    Args:
        path: The file.

    Returns:
        The records, each record's full row (``used`` says whether the merge took it), and the
        number of lines that are not valid rows (a cut last line).
    """
    from arc26.agent.loop import record_from_json

    records, rows, skipped = [], [], 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                records.append(record_from_json(row))
            except (ValueError, TypeError, KeyError):
                skipped += 1
                continue
            rows.append(row)
    return records, rows, skipped


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    from arc26.agent.loop import _ensure_line_end

    _ensure_line_end(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: object) -> None:
    from arc26.submit import write_json_atomic

    try:
        write_json_atomic(path, value)
    except (OSError, TypeError, ValueError) as err:
        logger.error("cannot write %s: %s", path, err)


# --- the engine ---------------------------------------------------------------------------------


def _vllm_server(options: agent_run.RunOptions, env: Mapping[str, str]) -> EngineServer:
    """The vLLM engine from the venv, in the prepared environment."""
    config = options.config.server
    command = vllm_command(options.engine_venv / "bin" / "vllm", str(options.model_dir), config)
    return EngineServer(
        command,
        env,
        options.log_dir / agent_run.ENGINE_LOG,
        config.root_url,
        boot_timeout_s=config.boot_timeout_s,
        stop_timeout_s=config.stop_timeout_s,
        health_interval_s=config.health_interval_s,
        pid_path=options.log_dir / agent_run.ENGINE_PID,
    )


def prepare_venv(
    options: agent_run.RunOptions, config: ShotConfig, summary: dict[str, Any]
) -> dict[str, str] | None:
    """Prepare the engine venv as P6.3-05 did and return the engine's environment.

    The environment is :func:`arc26.agent.server.engine_env` (``clean_env``, the config's
    variables, the venv's ``bin`` first on ``PATH``), as the spike built it; then
    :func:`arc26.agent.engine_prep.prepare_engine` and the GPU preflight, each writing its JSON
    to the log folder.

    Args:
        options: The run's options (``engine_venv``).
        config: The config.
        summary: ``run.json``'s dict (``engine`` and ``problems``).

    Returns:
        The environment, or None when a step failed (the problem is in ``summary``).
    """
    from arc26.agent import engine_prep

    bin_dir = options.engine_venv / "bin"
    python = bin_dir / "python"
    env = engine_env(os.environ, config.server, str(bin_dir))
    if not config.prep.enabled:
        return env
    settings = {
        "engine_imports": list(config.prep.imports),
        "engine_check_timeout_s": config.prep.check_timeout_s,
    }
    began = time.monotonic()
    try:
        env, record = engine_prep.prepare_engine(
            python, env, settings, options.log_dir, Path(config.prep.scratch_dir)
        )
        summary["engine"]["prep"] = {
            "seconds": round(time.monotonic() - began, 1),
            "jit_cache": record["jit_cache"]["state"],
            "removed": record["jit_cache"]["removed"],
            "modules": len(record["imports"].get("modules", [])),
        }
        if config.prep.preflight:
            began = time.monotonic()
            engine_prep.preflight(python, config.server.tensor_parallel_size, options.log_dir, env)
            summary["engine"]["preflight_s"] = round(time.monotonic() - began, 1)
    except engine_prep.Stop as err:
        summary["problems"].append(f"engine preparation: {err}")
        return None
    return env


def boot_engine(
    server: EngineServer, deadline: float, restarts: int, summary: dict[str, Any]
) -> bool:
    """Start the engine and wait for ``/health``; one more start after a crash if allowed.

    Why: run synchronously in the main thread, so a SIGTERM during the boot reaches the final
    write at once instead of waiting for a boot thread.

    Args:
        server: The engine.
        deadline: The generation stop (``time.monotonic()``).
        restarts: Starts allowed after a crash during the boot (at most 1 is used).
        summary: ``run.json``'s dict.

    Returns:
        Whether the engine is up.
    """
    for attempt in range(1 + min(restarts, 1)):
        left = deadline - time.monotonic()
        if left <= 0:
            summary["problems"].append("the generation stop passed before the engine was up")
            return False
        try:
            seconds = server.start(left)
        except EngineBootTimeout as err:
            summary["problems"].append(f"engine start {attempt + 1}: {err}")
            return False
        except EngineStartError as err:
            summary["problems"].append(f"engine start {attempt + 1}: {err}")
            continue
        summary["engine"]["boot_s"] = round(seconds, 1)
        return True
    return False


def url_ready(root: str, timeout_s: float, interval_s: float) -> bool:
    """Wait until a server given by URL answers ``/health`` with 200.

    Args:
        root: Its root URL.
        timeout_s: Seconds allowed.
        interval_s: Seconds between probes.

    Returns:
        Whether it answered in time.
    """
    end = time.monotonic() + max(0.0, timeout_s)
    while True:
        if http_get(root, "/health", 10.0)[0] == 200:
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(min(interval_s, max(0.0, end - time.monotonic())))


def kv_pool(log_path: Path, root: str, fallback: int) -> tuple[int, str, dict[str, Any]]:
    """Return the KV pool in tokens and where it came from, with the boot's facts.

    Args:
        log_path: The engine log (vLLM prints ``KV cache size: N tokens``).
        root: The server's root URL (``/metrics`` holds ``kv_cache_size_tokens``).
        fallback: The config's pool, used when neither reports one.

    Returns:
        The pool, its source (``engine log``, ``/metrics``, or ``config``), and the facts for
        ``boot.json``.
    """
    from arc26.agent.engine_prep import cache_config_info, parse_boot_log

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    facts = parse_boot_log(text)
    status, body = http_get(root, "/metrics")
    facts["cache_config"] = cache_config_info(body) if status == 200 else {}
    if facts.get("kv_cache_tokens_first"):
        return int(facts["kv_cache_tokens_first"]), "engine log", facts
    try:
        return int(facts["cache_config"]["kv_cache_size_tokens"]), "/metrics", facts
    except (KeyError, TypeError, ValueError):
        return fallback, "config", facts


async def sanity_check(engine: Any, config: ShotConfig, deadline: float) -> dict[str, Any]:
    """Send P6.3-05's sanity request and judge the reply (design point 7).

    The request waits at most ``timeout_s`` and never past the generation stop: a timeout, or a
    stop already passed, fails the check (design point 6), and NVARC's file stands.

    Args:
        engine: The recording engine.
        config: The config (``sanity``, ``seed``).
        deadline: The generation stop (``time.monotonic()``).

    Returns:
        ``ok`` (an answer or reasoning came back, not degenerate, not empty), the answer's tail,
        ``answer_ok`` (it holds 42, recorded only), the tokens, and an ``error`` when it failed.
    """
    from arc26.agent.loop import AttemptContext, attempt_context
    from arc26.agent.types import Message

    settings = config.sanity
    context = AttemptContext("sanity", "sanity/0", SHOT_PHASE)
    began = time.monotonic()
    timeout = min(settings["timeout_s"], deadline - began)
    if timeout <= 0:
        return {"ok": False, "error": "the generation stop passed before the sanity request"}
    try:
        with attempt_context(context):
            completion = await asyncio.wait_for(
                engine.complete(
                    (Message("user", SANITY_PROMPT),),
                    max_tokens=settings["max_tokens"],
                    temperature=config.request.temperature,
                    seed=config.seed,
                    reasoning_effort=settings["reasoning_effort"],
                ),
                timeout=timeout,
            )
    except Exception as err:
        what = "timed out" if isinstance(err, TimeoutError) else f"{type(err).__name__}: {err}"
        return {"ok": False, "seconds": round(time.monotonic() - began, 2), "error": what[:300]}
    rows = getattr(engine, "calls", {}).get(context.attempt_id) or [{}]
    health = rows[-1]
    replied = bool(completion.text.strip() or (completion.reasoning or "").strip())
    ok = replied and not health.get("degenerate") and not health.get("empty")
    return {
        "ok": ok,
        "seconds": round(time.monotonic() - began, 2),
        "answer": completion.text[-200:],
        "answer_ok": "42" in completion.text,
        "tokens": completion.completion_tokens,
        "degenerate": bool(health.get("degenerate")),
        "empty": bool(health.get("empty")),
    }


# The budget probe (task P6.9): the sanity question at low effort with a budget too short for any
# reasoning, so a server that honors the budget forces its end note at once.
PROBE_TOKENS = 8
PROBE_MAX_TOKENS = 512
PROBE_ID = "budget-check"


async def budget_probe(
    engine: Any, note: str, seed: int, timeout_s: float, deadline: float | None = None
) -> dict[str, Any]:
    """Check once, after the boot, that the server ends the thinking at a request's budget.

    Why: vLLM answers HTTP 400 to a ``thinking_token_budget`` when it runs without a reasoning
    config, and a server that ignores the field would let every thinking turn run to its cap and
    the answer room; either way the in-request forcing must be off for the run, and this one
    request tells before any task is shot.

    Args:
        engine: The engine (its ``complete`` takes ``thinking_token_budget``).
        note: The end note the server should force (``request.thinking_end_note``).
        seed: The run's seed.
        timeout_s: Seconds the request may take.
        deadline: The generation stop (``time.monotonic()``), or None.

    Returns:
        ``ok`` (the reasoning ends with the note and an answer follows), ``forced``, the reply's
        tail, the reasoning's tail, tokens, finish, seconds, and an ``error`` when the call failed.
    """
    from arc26.agent.loop import AttemptContext, attempt_context
    from arc26.agent.types import Message

    began = time.monotonic()
    timeout = timeout_s if deadline is None else min(timeout_s, deadline - began)
    if timeout <= 0:
        return {"ok": False, "forced": False, "error": "no time left for the budget probe"}
    try:
        with attempt_context(AttemptContext(PROBE_ID, f"{PROBE_ID}/0", SHOT_PHASE)):
            completion = await asyncio.wait_for(
                engine.complete(
                    (Message("user", SANITY_PROMPT),), max_tokens=PROBE_MAX_TOKENS,
                    temperature=1.0, seed=seed, reasoning_effort="low",
                    thinking_token_budget=PROBE_TOKENS,
                ),
                timeout=timeout,
            )  # fmt: skip
    except Exception as err:
        what = "timed out" if isinstance(err, TimeoutError) else f"{type(err).__name__}: {err}"
        return {"ok": False, "forced": False, "error": what[:300],
                "seconds": round(time.monotonic() - began, 2)}  # fmt: skip
    reasoning = (completion.reasoning or "").rstrip()
    forced = reasoning.endswith(note.strip())
    return {
        "ok": forced and bool(completion.text.strip()), "forced": forced,
        "answer": completion.text[-200:], "reasoning_tail": reasoning[-200:],
        "completion_tokens": completion.completion_tokens,
        "finish_reason": completion.finish_reason,
        "seconds": round(time.monotonic() - began, 2),
    }  # fmt: skip


def without_in_request(config: ShotConfig) -> ShotConfig:
    """Return the config with the in-request forcing off: Submission 3's request and plan.

    Args:
        config: The config.

    Returns:
        The same config with ``forcing_in_request`` off in the request and the wave model (the
        forced replies' prefill planned again), ``turn_cut_s`` gone with it.
    """
    return dataclasses.replace(
        config,
        waves=dataclasses.replace(config.waves, forcing_in_request=False),
        request=dataclasses.replace(config.request, forcing_in_request=False),
    )


# --- the waves ----------------------------------------------------------------------------------


@dataclass
class ShotState:
    """What the waves need and record, shared with :func:`run`.

    Attributes:
        options: The run's options.
        config: The config.
        writer: The submission writer.
        tasks: The run's tasks by id.
        queue: ``(task_id, opening bound)`` still to shoot, in order.
        confidence: NVARC's confidence of every queued task.
        kv_pool: The KV pool in tokens.
        kv_source: Where it came from.
        started: The process's start (``time.monotonic()``).
        deadline: The generation stop (``time.monotonic()``).
        plan: ``shot_plan.json``'s dict.
        records: Every shot record of this run, resumed ones first.
        rows: Their ``shot_attempts.jsonl`` rows.
    """

    options: agent_run.RunOptions
    config: ShotConfig
    writer: ShotWriter
    tasks: Mapping[str, AgentTask]
    queue: list[tuple[str, int]]
    confidence: Mapping[str, float]
    kv_pool: int
    kv_source: str
    started: float
    deadline: float
    plan: dict[str, Any]
    records: list[AttemptRecord] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def since_start(self, now: float | None = None) -> float:
        """Return seconds since the process's start.

        Args:
            now: A ``time.monotonic()`` value; now when None.

        Returns:
            Seconds, rounded to 0.1.
        """
        return round((time.monotonic() if now is None else now) - self.started, 1)

    def save_plan(self) -> None:
        """Write ``shot_plan.json`` atomically (a failure is logged)."""
        _write_json(self.options.log_dir / SHOT_PLAN, self.plan)


async def _until_failed(supervisor: Any, interval_s: float) -> None:
    while not supervisor.failed:
        await asyncio.sleep(interval_s)


async def run_wave(
    state: ShotState,
    engine: Any,
    number: int,
    plan: WavePlan,
    supervisor: Any,
) -> dict[str, Any]:
    """Run one wave: P6.3-05's request on each task at once, merged as each one finishes.

    Each finished attempt is logged to ``shot_attempts.jsonl`` and, when usable (it was not cut
    by the generation stop and no reply of it was degenerate), merged into the file, which is
    rewritten atomically. The wave ends when every task has finished, at the generation stop,
    or when the engine is gone for good (its calls are cancelled; their tasks keep NVARC's
    attempts).

    Args:
        state: The shot's state.
        engine: The recording engine.
        number: The wave's number, from 1.
        plan: Its plan.
        supervisor: The engine's supervisor.

    Returns:
        The wave's row for ``shot_plan.json``: the plan, the actual end, and the counts.
    """
    from arc26.agent.loop import record_to_json, run_tasks

    config, options = state.config, state.options
    began = time.monotonic()
    row: dict[str, Any] = {
        "wave": number,
        **plan.as_dict(),
        "confidence": {t: round(state.confidence.get(t, 0.0), 6) for t in plan.tasks},
        "start_s": state.since_start(began),
        "planned_end_s": state.since_start(began + plan.planned_s),
        "planned_end_unix": round(time.time() + plan.planned_s, 1),
    }
    counts = {"finished": 0, "used": 0, "deadline": 0, "degenerate": 0, "engine_error": 0}
    finished: set[str] = set()

    def on_record(record: AttemptRecord) -> None:
        calls = engine.calls.get(record.attempt_id, [])
        degenerate = any(c.get("degenerate") for c in calls)
        used = bool(record.direct_answers) and record.stop_reason != "deadline" and not degenerate
        counts["finished"] += 1
        finished.add(record.task_id)
        counts["used"] += used
        counts["deadline"] += record.stop_reason == "deadline"
        counts["degenerate"] += degenerate
        counts["engine_error"] += record.stop_reason == "engine_error"
        line = {
            **record_to_json(record),
            "wave": number,
            "cap": plan.cap,
            "finish": [c.get("finish_reason") for c in calls],
            "errors": sum(1 for c in calls if c.get("error")),
            "degenerate": degenerate,
            "used": used,
        }
        state.records.append(record)
        state.rows.append(line)
        try:
            _append_row(options.log_dir / SHOT_ATTEMPTS, line)
        except (OSError, TypeError, ValueError) as err:
            logger.error("cannot log %s: %s", record.attempt_id, err)
        if used:
            state.writer.add(record)

    transcripts = None
    if options.mode == DEV and config.dev_transcripts:
        transcripts = options.log_dir / SHOT_TRANSCRIPTS
    tasks = [state.tasks[t] for t in plan.tasks]
    wave = asyncio.ensure_future(
        run_tasks(
            tasks,
            engine,
            lambda task: NoRepl(),
            NoVerifier(),
            config.request.budget(plan.cap),
            concurrency=len(tasks),
            phase=SHOT_PHASE,
            seed=config.seed,
            deadline=state.deadline,
            log_path=None,
            transcript_path=transcripts,
            system=config.request.system,
            on_record=on_record,
        )  # fmt: skip
    )
    watch = asyncio.ensure_future(_until_failed(supervisor, config.server.monitor_interval_s))
    try:
        await asyncio.wait({wave, watch}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for job in (wave, watch):
            if not job.done():
                job.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await job
    if wave.done() and not wave.cancelled() and wave.exception() is not None:
        row["error"] = f"{type(wave.exception()).__name__}: {wave.exception()}"
    ended = time.monotonic()
    row.update(
        counts,
        unfinished=[t for t in plan.tasks if t not in finished],
        end_s=state.since_start(ended),
        actual_s=round(ended - began, 1),
        engine_gone=bool(supervisor.failed),
    )
    return row


def _refill_stop(recent: Sequence[tuple[bool, bool]], started: int, model: WaveModel,
                 share: float) -> str | None:  # fmt: skip
    """The refill queue's stop rules, a wave's read over the last ``wave_tasks`` finished."""
    degenerate = sum(d for d, _ in recent)
    if degenerate > share * model.wave_tasks:
        return f"{degenerate} of the last {len(recent)} finished tasks degenerate"
    if recent and len(recent) >= min(model.wave_tasks, started) and all(e for _, e in recent):
        return f"the last {len(recent)} finished tasks all ended with an engine error"
    return None


async def run_refill(state: ShotState, engine: Any, supervisor: Any) -> None:
    """Run the refill queue (``waves.refill``, task P6.9, option 2b (ii)).

    Up to ``wave_tasks`` tasks run at once; whenever one ends, the next queued task starts at
    once with the cap :func:`plan_start` gives for the time left, instead of waiting for a whole
    wave. Each start is a one-task :func:`run_wave` (its row in ``shot_plan.json``'s ``waves``,
    its records merged as they finish). No task starts once its cap would fall under
    ``min_cap``, the engine is gone, or a stop rule fires (more than ``wave_share`` of the last
    ``wave_tasks`` finished tasks degenerate, or all of them engine errors); the running ones
    then finish, as a wave's do. A task that does not fit the free KV waits for a running one
    to end.

    Args:
        state: The shot's state (its ``plan`` gets ``waves`` and ``stopped``).
        engine: The recording engine.
        supervisor: The engine's supervisor.
    """
    config, plan, model = state.config, state.plan, state.config.waves
    running: dict[asyncio.Future[Any], WavePlan] = {}
    recent: deque[tuple[bool, bool]] = deque(maxlen=model.wave_tasks)
    rows: dict[int, int] = {}
    number, stopped = 0, None
    while True:
        while stopped is None and state.queue and len(running) < model.wave_tasks:
            if supervisor.failed:
                stopped = f"engine gone: {supervisor.reason}"
                break
            left = state.deadline - time.monotonic()
            free = state.kv_pool - sum(p.kv_need for p in running.values())
            start = plan_start(state.queue[0], left, model, free, len(running) + 1)
            if not start.tasks:
                if (start.reason or "").startswith("kv") and running:
                    break  # a running task's end frees its KV
                stopped = start.reason
                plan["last_plan"] = start.as_dict()
                break
            number += 1
            state.queue.pop(0)
            logger.info(
                "start %d: %s, cap %d, planned %.0f s of %.0f s left, %d streams",
                number, start.tasks[0], start.cap, start.planned_s, left, len(running) + 1,
            )  # fmt: skip
            rows[number] = len(plan["waves"])
            plan["waves"].append({"wave": number, **start.as_dict(), "running": True})
            state.save_plan()
            job = asyncio.ensure_future(run_wave(state, engine, number, start, supervisor))
            running[job] = start
        if not running:
            if stopped is None:
                stopped = "no tasks left" if not state.queue else "no task could start"
            plan["stopped"] = stopped
            return
        done, _ = await asyncio.wait(set(running), return_when=asyncio.FIRST_COMPLETED)
        for job in done:
            start = running.pop(job)
            try:
                row = job.result()
            except Exception as err:  # run_wave catches its own failures; this is a bug
                row = {"wave": None, **start.as_dict(), "error": f"{type(err).__name__}: {err}",
                       "finished": 0, "degenerate": 0, "engine_error": 0}  # fmt: skip
            number_done = row.get("wave")
            if number_done in rows:
                plan["waves"][rows[number_done]] = row
            state.save_plan()
            recent.append((row.get("degenerate", 0) > 0,
                           bool(row.get("finished")) and row.get("engine_error") == row.get(
                               "finished")))  # fmt: skip
            rule = _refill_stop(recent, number, model, config.degenerate["wave_share"])
            if stopped is None and rule is not None:
                stopped = rule
        if stopped is None and supervisor.failed:
            stopped = f"engine gone: {supervisor.reason}"


async def run_waves(state: ShotState, server: EngineServer | None, root: str) -> None:
    """Watch the engine, send the sanity request, and run waves while time and tasks remain.

    Args:
        state: The shot's state (its ``plan`` gets ``sanity``, ``waves``, and ``stopped``).
        server: The managed engine, or None for a server given by URL.
        root: The server's root URL.
    """
    from arc26.agent.recording import recording_engine
    from arc26.agent.server import EngineSupervisor

    config, plan = state.config, state.plan
    supervisor = EngineSupervisor(
        root,
        server,
        interval_s=config.server.monitor_interval_s,
        hang_s=config.server.hang_s,
        restarts=config.server.restarts,
        deadline=state.deadline,
    )
    supervisor.start()
    engine = recording_engine(
        root,
        config.server.served_model_name,
        config.waves.wave_tasks,
        state.options.log_dir,
        config.degenerate,
        return_token_ids=config.return_token_ids,
        count_prompts=config.count_prompts,
        **config.engine,
    )
    try:
        plan["sanity"] = await sanity_check(engine, config, state.deadline)
        if not plan["sanity"]["ok"]:
            plan["stopped"] = f"the sanity request failed: {plan['sanity']}"
            return
        if config.request.forcing_in_request:
            probe = await budget_probe(
                engine, config.request.thinking_end_note, config.seed,
                config.sanity["timeout_s"], state.deadline,
            )  # fmt: skip
            probe["in_request"] = probe["ok"]
            plan["budget_probe"] = probe
            if not probe["ok"]:
                # The whole run falls back to the separate forcing turn and plans its prefill.
                logger.warning("the budget probe failed (%s): in-request forcing off", probe)
                state.config = config = without_in_request(config)
                plan["model"] = dataclasses.asdict(config.waves)
            state.save_plan()
        if config.waves.refill:
            await run_refill(state, engine, supervisor)
            return
        number = 0
        while True:
            if supervisor.failed:
                plan["stopped"] = f"engine gone: {supervisor.reason}"
                return
            left = state.deadline - time.monotonic()
            wave = plan_wave(state.queue, left, config.waves, state.kv_pool)
            if not wave.tasks:
                plan["stopped"] = wave.reason
                plan["last_plan"] = wave.as_dict()
                return
            number += 1
            taken = set(wave.tasks)
            state.queue = [item for item in state.queue if item[0] not in taken]
            logger.info(
                "wave %d: %d tasks, cap %d, planned %.0f s of %.0f s left",
                number, len(wave.tasks), wave.cap, wave.planned_s, left,
            )  # fmt: skip
            plan["waves"].append({"wave": number, **wave.as_dict(), "running": True})
            state.save_plan()
            row = await run_wave(state, engine, number, wave, supervisor)
            plan["waves"][-1] = row
            state.save_plan()
            size = len(wave.tasks)
            if row["degenerate"] > config.degenerate["wave_share"] * size:
                plan["stopped"] = f"wave {number}: {row['degenerate']} of {size} degenerate"
                return
            if row["finished"] and row["engine_error"] == row["finished"] == size:
                plan["stopped"] = f"wave {number}: every attempt ended with an engine error"
                return
    finally:
        await supervisor.close()
        state.plan["engine_events"] = supervisor.events
        if supervisor.reason:
            state.plan.setdefault("engine_problem", supervisor.reason)
        with contextlib.suppress(Exception):
            await engine.aclose()


# --- development scoring ------------------------------------------------------------------------


def shot_dev_metrics(
    mode: str,
    solutions: Path,
    state: ShotState,
    label: str,
) -> dict[str, Any]:
    """Score the hybrid, NVARC alone, and the shot alone on the run, in development mode only.

    Why: the second place a solutions file is opened (after :func:`arc26.agent.run.score_dev`);
    it refuses any other mode before touching the file (BRIEF A0).

    Args:
        mode: The run's mode; anything but ``dev`` is refused.
        solutions: The solutions file.
        state: The shot's state (the writer, the records, the plan).
        label: The subset's name for the report.

    Returns:
        ``hybrid`` and ``nvarc`` (score, tasks, solved outputs, test outputs on the same tasks),
        ``shot`` (the shot's own two answers per output on the tasks it shot), the shot's solves,
        the tasks the merge gained and lost against NVARC, and per wave the cap and its solves.

    Raises:
        PermissionError: If ``mode`` is not ``dev``.
    """
    if mode != DEV:
        raise PermissionError("competition mode never opens a solutions file")
    from arc26.evaluate import load_predictions, load_solutions, score_submission

    answers = load_solutions(solutions)
    writer = state.writer
    ids = [t for t in writer.counts if t in answers]
    hybrid = score_submission(load_predictions(writer.path), answers, ids, label)
    nvarc = score_submission(writer.nvarc_submission(), answers, ids, label)
    by_task: dict[str, list[AttemptRecord]] = defaultdict(list)
    used = {row["attempt_id"] for row in state.rows if row.get("used")}
    for record in state.records:
        if record.attempt_id in used:
            by_task[record.task_id].append(record)
    shot_ids = sorted({r.task_id for r in state.records} & set(ids))
    own = {}
    for task_id in shot_ids:
        rows = []
        for j in range(writer.counts[task_id]):
            implied = writer.implied(task_id, j)
            grids = (shot_answers(by_task.get(task_id, ()), j, implied) + [None, None])[:2]
            rows.append(_entry(grids))
        own[task_id] = rows
    shot = score_submission(own, answers, shot_ids, f"{label}, shot tasks")
    h = {t.task_id: t.score for t in hybrid.per_task}
    n = {t.task_id: t.score for t in nvarc.per_task}
    s = {t.task_id: t.score for t in shot.per_task}

    def block(score: Any) -> dict[str, Any]:
        return {
            "score": round(score.score, 3),
            "tasks": score.tasks,
            "solved_outputs": score.solved_outputs,
            "test_outputs": score.test_outputs,
        }

    waves = []
    for row in state.plan.get("waves", []):
        tasks = [t for t in row.get("tasks", []) if t in s]
        waves.append({
            "wave": row.get("wave"), "cap": row.get("cap"), "tasks": len(row.get("tasks", [])),
            "solved": sorted(t for t in tasks if s[t] >= 1),
            "planned_s": row.get("planned_s"), "actual_s": row.get("actual_s"),
        })  # fmt: skip
    return {
        "subset": label,
        "hybrid": block(hybrid),
        "nvarc": block(nvarc),
        "shot": block(shot),
        "shot_tasks": shot_ids,
        "shot_solved": sorted(t for t in shot_ids if s[t] >= 1),
        "nvarc_solved_on_shot_tasks": sorted(t for t in shot_ids if n.get(t, 0) >= 1),
        "gained": sorted(t for t in ids if h[t] > n[t]),
        "lost": sorted(t for t in ids if h[t] < n[t]),
        "net_tasks": round(hybrid.score - nvarc.score, 3),
        "waves": waves,
    }


# --- the run ------------------------------------------------------------------------------------


def _summary_markdown(summary: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    lines = [
        "# The long shot (arc26.agent.shot)",
        "",
        f"- mode {summary.get('mode')}, {summary.get('tasks', 0)} tasks, "
        f"{plan.get('queued', 0)} queued under max_confidence {plan.get('max_confidence')}",
        f"- KV pool {plan.get('kv_pool')} tokens ({plan.get('kv_source')})",
        f"- stopped: {plan.get('stopped')}",
    ]
    for row in plan.get("waves", []):
        lines.append(
            f"- wave {row.get('wave')}: {len(row.get('tasks', []))} tasks, cap {row.get('cap')}, "
            f"planned {row.get('planned_s')} s, actual {row.get('actual_s')} s, used "
            f"{row.get('used')}, cut {row.get('deadline')}, degenerate {row.get('degenerate')}"
        )
    sub = summary.get("submission", {})
    lines.append(f"- submission valid {sub.get('valid')}, slots {sub.get('slots')}")
    metrics = summary.get("shot_metrics")
    if metrics:
        lines.append(
            f"- {metrics['subset']}: hybrid {metrics['hybrid']['score']} of "
            f"{metrics['hybrid']['tasks']}, NVARC alone {metrics['nvarc']['score']}, shot solves "
            f"{len(metrics['shot_solved'])} of {len(metrics['shot_tasks'])}, gained "
            f"{metrics['gained']}, lost {metrics['lost']}"
        )
    for problem in summary.get("problems", []):
        lines.append(f"- problem: {str(problem)[:300]}")
    return "\n".join(lines) + "\n"


def run(options: agent_run.RunOptions) -> int:
    """Run the shot with resolved options; always leave a valid submission if possible.

    Args:
        options: The options (``config`` a :class:`ShotConfig`).

    Returns:
        The exit code: 0 with a valid submission at the end, 1 without one.
    """
    options_in = options
    config: ShotConfig = for_mode(options.config, options.mode)
    if config is not options.config:
        options = dataclasses.replace(options, config=config)
    started = time.monotonic()
    deadline = started + (options.session_end - time.time()) - config.final_margin_s
    summary: dict[str, Any] = {
        "module": "arc26.agent.shot",
        "mode": options.mode,
        "tasks_file": str(options.tasks),
        "out": str(options.out),
        "engine": {"kind": options.engine_kind},
        "deadline_in_s": round(deadline - started, 1),
        "deadline_source": options.deadline_source,
        "problems": [],
    }
    problems: list[str] = summary["problems"]
    try:
        options.log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:  # the logs are not worth the submission: run with temporary ones
        fallback = agent_run._fallback_log_dir(options.out)
        problems.append(f"log folder {options.log_dir} unusable ({err}); logs in {fallback}")
        options = dataclasses.replace(options, log_dir=fallback)
    from arc26.agent.select import nvarc_from_submission
    from arc26.agent.tasks import count_test_inputs, load_tasks

    try:
        tasks = load_tasks(options.tasks, options.task_ids)
    except (OSError, ValueError) as err:
        problems.append(f"tasks: {err}")
        logger.error("cannot read the tasks: %s", err)
        agent_run._write_summary(options, summary, started)
        return EXIT_NO_SUBMISSION
    counts = count_test_inputs(tasks)
    summary["tasks"] = len(counts)
    nvarc = None
    copy, raw = agent_run.nvarc_copy(options.nvarc_submission, options.log_dir, problems)
    if raw is not None:
        try:
            nvarc = nvarc_from_submission(raw)
        except ValueError as err:
            problems.append(f"NVARC's submission skipped: {err}")
    summary["nvarc"] = None if copy is None else {"copy": str(copy), "tasks": len(nvarc or {})}
    candidates = agent_run._read_json(options.nvarc_candidates, "NVARC's candidates", problems)
    if candidates is not None and not isinstance(candidates, Mapping):
        problems.append("NVARC's candidates skipped: not a JSON object")
        candidates = None
    resumed: list[AttemptRecord] = []
    rows: list[dict[str, Any]] = []
    log = options.log_dir / SHOT_ATTEMPTS
    if log.exists():
        try:
            resumed, rows, skipped = read_shot_log(log)
            summary["resume"] = {"records": len(resumed), "skipped_lines": skipped}
        except OSError as err:
            problems.append(f"the shot log could not be read: {err}")
    used = {row["attempt_id"] for row in rows if row.get("used")}
    shapes = None
    if config.request.lenient_answers:
        shapes = {task.task_id: implied_shapes(task) for task in tasks}
    writer = ShotWriter(
        options.out, counts, nvarc, [r for r in resumed if r.attempt_id in used], shapes
    )
    if not writer.write():
        problems.append(f"the first write of {options.out} failed: {writer.last_error}")
    fallback = None
    system = config.request.system
    openings = None
    if config.order == OPENING_ORDER:
        openings = {task.task_id: opening_bound(task, system) for task in tasks}
    try:
        if config.order == OPENING_ORDER:
            # LLM first (task P5.1): no NVARC ran, so no confidence; every task, smallest first.
            confidence = dict.fromkeys(counts, 0.0)
            source = "none (order opening)"
        elif candidates is not None:
            confidence = shot_confidence(counts, candidates, nvarc)
            source = "candidates"
        else:
            confidence = confidence_without_candidates(counts, nvarc)
            source = "submission"
            fallback = (
                "no readable NVARC candidates file: only tasks with an output NVARC left empty "
                "are shot, by task id"
            )
            problems.append(fallback)
        chosen = select_tasks(
            confidence, config.max_confidence, {r.task_id for r in resumed}, config.order,
            config.mid_bucket, config.zero_bucket, openings,
        )  # fmt: skip
    except Exception as err:  # a readable but odd file: the documented no-candidates fallback
        fallback = (
            f"NVARC's candidates failed ({type(err).__name__}: {err}): only tasks with an output "
            "NVARC left empty are shot, by task id"
        )
        problems.append(fallback)
        try:
            confidence = confidence_without_candidates(counts, nvarc)
            chosen = select_tasks(
                confidence, config.max_confidence, {r.task_id for r in resumed}, config.order,
                config.mid_bucket, config.zero_bucket, openings,
            )  # fmt: skip
            source = "submission"
        except Exception as again:  # the shot is an extra: never stop the final write
            problems.append(f"NVARC's confidence failed: {type(again).__name__}: {again}")
            confidence, chosen, source = {}, [], "none"
    unknown = sorted(t for t, c in confidence.items() if c is None)
    summary["selection"] = {
        "source": source,
        "fallback": fallback,
        "queued": len(chosen),
        "unknown": len(unknown),
    }
    by_id = {task.task_id: task for task in tasks}
    state = ShotState(
        options=options,
        config=config,
        writer=writer,
        tasks=by_id,
        queue=[(t, opening_bound(by_id[t], system)) for t, _ in chosen],
        confidence=dict(chosen),
        kv_pool=config.waves.kv_pool_tokens,
        kv_source="config",
        started=started,
        deadline=deadline,
        plan={
            "max_confidence": config.max_confidence,
            **(
                {"order": config.order, "mid_bucket": config.mid_bucket}
                if config.order != "confidence"
                else {}
            ),
            **({} if config.zero_bucket else {"zero_bucket": False}),
            **({"dev_cap": config.dev_cap} if config is not options_in.config else {}),
            "confidence_source": source,
            "fallback": fallback,
            "unknown": unknown,
            "queued": len(chosen),
            "queue": [[t, round(c, 6)] for t, c in chosen],
            "model": dataclasses.asdict(config.waves),
            "arithmetic": stage_arithmetic(config),
            "waves": [],
            "stopped": None,
        },
        records=list(resumed),
        rows=list(rows),
    )
    state.save_plan()
    server: EngineServer | None = None
    try:
        left = deadline - time.monotonic()
        model = config.waves
        longest = max((o for _, o in state.queue), default=0)
        size = min(model.wave_tasks, len(state.queue))
        smallest = model.setup_s + model.wave_seconds(model.min_cap, size, longest)
        if not state.queue:
            state.plan["stopped"] = "no task under max_confidence"
        elif left < smallest:
            state.plan["stopped"] = (
                f"{left:.0f} s left before the generation stop, under the setup and the smallest "
                f"wave ({smallest:.0f} s): no engine started"
            )
        else:
            root = config.server.root_url
            env: Mapping[str, str] | None = None
            if options.engine_kind == "vllm":
                env = prepare_venv(options, config, summary)
                if env is not None:
                    server = _vllm_server(options, env)
            elif options.engine_kind == "fake":
                server = agent_run._engine_server(options)
            else:
                root = str(options.engine_url)
            up = False
            if server is not None:
                up = boot_engine(server, deadline, config.server.restarts, summary)
                summary["engine"]["command"] = server.command
            elif options.engine_kind == "url":
                limit = min(config.server.boot_timeout_s, deadline - time.monotonic())
                up = url_ready(root, limit, config.server.health_interval_s)
                if not up:
                    problems.append(f"the engine at {root} did not answer healthy")
            if up:
                state.kv_pool, state.kv_source, facts = kv_pool(
                    options.log_dir / agent_run.ENGINE_LOG, root, config.waves.kv_pool_tokens
                )
                facts.update(kv_pool=state.kv_pool, kv_source=state.kv_source,
                             boot_s=summary["engine"].get("boot_s"))  # fmt: skip
                _write_json(options.log_dir / BOOT_FILE, facts)
                state.plan.update(kv_pool=state.kv_pool, kv_source=state.kv_source)
                asyncio.run(run_waves(state, server, root))
            else:
                state.plan["stopped"] = "the engine was not up (see run.json's problems)"
    except BaseException as err:  # a bug, a signal, an interrupt: the final write still happens
        problems.append(f"{type(err).__name__}: {err}")
        state.plan.setdefault("interrupted", f"{type(err).__name__}: {err}")
        logger.error("the shot stopped early:\n%s", traceback.format_exc())
    finally:
        if server is not None:
            with contextlib.suppress(Exception):
                summary["engine"]["exit_code"] = server.stop()
        writer.write()
        state.save_plan()
    valid = agent_run._valid_on_disk(options.out, counts)
    summary["submission"] = {
        "valid": valid,
        "writes": writer.writes,
        "failed_writes": writer.failures,
        "slots": writer.slots(),
    }
    probe = state.plan.get("budget_probe")
    if probe is not None and not probe["in_request"]:
        problems.append(f"budget probe failed, in-request forcing off for the run: {probe}")
    summary["shot"] = {
        **({} if probe is None else {"in_request": probe["in_request"]}),
        "queued": state.plan["queued"],
        "waves": len(state.plan["waves"]),
        "shot_tasks": len({r.task_id for r in state.records}),
        "used": sum(1 for row in state.rows if row.get("used")),
        "stopped": state.plan.get("stopped"),
    }
    if options.mode == DEV and options.solutions is not None:
        try:
            summary["score"] = agent_run.score_dev(
                options.mode, options.out, options.solutions, list(counts)
            )
            label = f"{len(counts)} tasks"
            summary["shot_metrics"] = shot_dev_metrics(
                options.mode, options.solutions, state, label
            )
            _write_json(options.log_dir / SHOT_METRICS, summary["shot_metrics"])
        except (OSError, ValueError) as err:
            problems.append(f"scoring failed: {err}")
    summary["config_sha256"] = config.source_sha256
    agent_run._write_summary(options, summary, started)
    with contextlib.suppress(OSError):
        (options.log_dir / SUMMARY_MD).write_text(
            _summary_markdown(summary, state.plan), encoding="utf-8"
        )
    if not valid:
        logger.error("no valid submission at %s", options.out)
        return EXIT_NO_SUBMISSION
    return EXIT_OK


def build_parser() -> Any:
    """Build the command-line parser: :mod:`arc26.agent.run`'s, under this module's name.

    Returns:
        The parser.
    """
    parser = agent_run.build_parser()
    parser.prog = PROG
    parser.description = __doc__.split("\n", 1)[0]
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shot from the command line.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        The exit code (``EXIT_*`` of :mod:`arc26.agent.run`).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        options = agent_run.resolve_options(args, load=load_config)
    except SystemExit as err:  # argparse's own refusal
        return EXIT_REFUSED if err.code else EXIT_OK
    except ConfigError as err:
        print(f"refused: {err}", file=sys.stderr)
        return EXIT_REFUSED
    # SIGTERM (the notebook stopping the stage) runs the cleanup and the final write.
    previous = signal.signal(signal.SIGTERM, agent_run._raise_exit)
    try:
        return run(options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
