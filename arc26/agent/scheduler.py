r"""The agent's scheduler, governor, and crash-safe submission writer (P6.4).

Why: the scored run has 12 hours for about 240 hidden tasks on one engine whose throughput is
known only on the night (``docs/research/agent_library_design.md``, sections 5.4 and 5.5). The
scheduler spends that time in phases, sets each attempt's token budget from the throughput it
measures, stops at one deadline, and keeps ``submission.json`` valid after every attempt, so a
crash or a kill at any moment still scores.

Phases (design note 5.4; ADR-020):

1. Every task once, easiest first (:func:`ease_order`: smaller grids, the same input and output
   shape, fewer colors): a short attempt at low effort with at most 2 feedback rounds.
2. Tasks no verified program solves, NVARC's least confident first (:func:`nvarc_confidence`,
   ADR-014's count over 16 views): one long attempt that sees the task's best failed programs.
3. Tasks with partial progress (a program reproduced some demonstrations, most progress
   first), then tasks whose verified programs disagree on a test output.
4. Until the generation stop: fresh attempts (new seeds) on the unsolved tasks, then on tasks
   whose verified programs disagree, round robin by fewest attempts so far, so no time is left
   unused. Each attempt of a task is one round; a task leaves the cycle only after
   ``cycle_error_rounds`` consecutive rounds end with an engine error.

The phases roll: a free slot takes the first task that phase 1 still needs, else phase 2, 3, or
4, so the engine does not idle at a phase boundary while the last long attempts of the previous
phase finish. Within a phase, tasks with fewer attempts made go first, so every task gets its
first attempt before any gets a second. A task has at most one attempt in flight. With more than
one phase-1 attempt per task (``attempts``), solved tasks get independent votes, which is how
verified programs come to disagree. An attempt that ends with an engine error is run again, up
to ``engine_error_retries`` times; one cut by the deadline counts as not run, so a resumed run
repeats it. An attempt whose first prompt left no room for a reply (stop ``budget`` after 0
turns) does not count either: the next one goes without the earlier programs, and a second such
stop ends the task's phase. The earlier programs shown are at most ``prior_programs``, each
clipped with its report to ``prior_chars`` characters. Attempt ids are
``<task_id>/p<phase>/<k>`` with the first free ``k``, so ids stay unique across a kill and a
resume.

Governor: one deadline. :class:`RateMeter` measures the aggregate generated tokens per second over
a time window: each completed call's tokens count in proportion to its overlap with the window,
and the calls still in flight count their time in it at the completed calls' rate per
call-second (:class:`MeteredEngine` reports the calls). While fewer than ``min_calls`` completed
calls overlap the window, the last measured rate stays, and before any, a configured prior. Each
attempt's budget is set when it starts:

.. math::

    B = \min\left(c_p,\; \max\left(f_p,\; \left\lfloor
        \frac{s_p \, \sigma \, r \, T}{N_p} \right\rfloor\right)\right)

with :math:`r` the rate, :math:`\sigma` the rate safety factor, :math:`T` the seconds left,
:math:`N_p` the attempts phase :math:`p` still has to run (in flight included; one per task with
the default of one attempt per task and phase; for phase 4, the tasks in its cycle), :math:`s_p`
the phase's share of the remaining tokens, and :math:`f_p`, :math:`c_p` its floor and cap.
Attempts that finish under budget leave more for the tasks after them; a slow night shrinks every
later budget.

Deadline: no attempt starts after it; running attempts stop at their next engine call
(:func:`~arc26.agent.loop.run_attempt`'s deadline) and are logged; any still running
``grace_s`` seconds later are cancelled.

Crash safety: every finished attempt is appended to the attempt JSONL first, then
:class:`SubmissionWriter` rewrites the whole ``submission.json`` atomically, NVARC's attempts
(or ``[[0]]``) filling every slot the agent leaves. A resumed run reads the JSONL back, skips the
finished attempts, and rebuilds the same submission.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from arc26.agent import prompts
from arc26.agent.feedback import REPORT_CHARS, clip
from arc26.agent.loop import Budget, append_record, attempt_seed, run_attempt
from arc26.agent.select import (
    NvarcPair,
    build_selection,
    submission_from_selection,
    write_submission,
)
from arc26.agent.types import (
    AgentTask,
    AttemptRecord,
    Candidate,
    Completion,
    Engine,
    Message,
    Repl,
    Verification,
    Verifier,
)
from arc26.data import Grid

logger = logging.getLogger(__name__)

PHASES = (1, 2, 3, 4)
# Stop reasons of an attempt that did not run its course; the task's phase is not done.
INCOMPLETE = frozenset({"engine_error", "deadline"})
# Attempts of one task and phase whose first prompt left no room for a reply (``budget`` after
# 0 turns): the second ends the phase for the task, after a retry without the earlier programs.
EMPTY_LIMIT = 2
# Characters a shown program keeps beyond its report, at least.
MIN_PRIOR_SOURCE = 100
# NVARC decodes each test input under 16 views (ADR-014, ``augment(n=2, seed=2)``).
NVARC_VIEWS = 16
# Why the scheduler stopped starting attempts.
STOP_DONE, STOP_DEADLINE, STOP_ENGINE = "done", "deadline", "engine"


@dataclass(frozen=True)
class PhaseSettings:
    """Budget and sampling of one phase's attempts.

    Attributes:
        attempts: Attempts that ran their course a task gets in this phase; 0 turns the phase
            off, and None (the last phase only) means no limit, until the generation stop.
            Phase 1 runs them on every task, solved or not (independent votes); phase 2 stops
            once the task is solved; phases 3 and 4 run them while the task stays eligible.
        min_attempt_tokens: The floor of an attempt's generated tokens.
        max_attempt_tokens: The cap of an attempt's generated tokens.
        turn_tokens: The cap on one reply (lowered to the attempt's budget).
        max_turns: Replies per attempt (phase 1: 3, the first answer and 2 feedback rounds).
        reasoning_effort: The engine's effort level (Qwen3.8: ``low``, ``medium``, ``xhigh``), or
            None for the model's default.
        temperature: Sampling temperature.
        share: The share of the remaining tokens the phase plans on, in (0, 1]; phase 1 below 1
            keeps time for the later phases when tokens are scarce.
        preserve_thinking: Keep each reply's reasoning in the conversation.
        prior: Show the task's best failed programs in the first message (not for a task that
            already has a verified program, whose extra attempt must be an independent vote).
    """

    attempts: int | None = 1
    min_attempt_tokens: int = 8_000
    max_attempt_tokens: int = 24_000
    turn_tokens: int = 16_384
    max_turns: int = 3
    reasoning_effort: str | None = "low"
    temperature: float = 1.0
    share: float = 1.0
    preserve_thinking: bool = False
    prior: bool = False

    def __post_init__(self) -> None:
        """Reject settings that cannot work.

        Raises:
            ValueError: If ``attempts`` is negative, a token limit or the turn count is below 1,
                the cap is below the floor, the share is outside (0, 1], or the temperature is
                negative.
        """
        if self.attempts is not None and self.attempts < 0:
            raise ValueError(f"attempts must be at least 0 or None; got {self.attempts}")
        for name in ("min_attempt_tokens", "turn_tokens", "max_turns"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1; got {getattr(self, name)}")
        if self.max_attempt_tokens < self.min_attempt_tokens:
            raise ValueError("max_attempt_tokens must be at least min_attempt_tokens")
        if not 0 < self.share <= 1:
            raise ValueError(f"share must be in (0, 1]; got {self.share}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative; got {self.temperature}")


@dataclass(frozen=True)
class GovernorSettings:
    """How the governor turns measured throughput into budgets.

    Attributes:
        prior_rate: Aggregate generated tokens per second assumed until a rate is measured.
        window_s: The measuring window in seconds; long enough to hold many of the phases'
            long calls (tens of minutes each).
        min_calls: Completed calls that must overlap the window for a new measurement; with
            fewer, the last measured rate stays.
        rate_safety: The share of the rate the budgets plan on, in (0, 1]: a phase's last
            attempts run below full concurrency and the deadline cuts attempts mid-call.
    """

    prior_rate: float = 400.0
    window_s: float = 3600.0
    min_calls: int = 8
    rate_safety: float = 0.9

    def __post_init__(self) -> None:
        """Reject settings that cannot work.

        Raises:
            ValueError: If the prior rate or the window is not positive, ``min_calls`` is below
                1, or the safety factor is outside (0, 1].
        """
        if not self.prior_rate > 0 or not self.window_s > 0:
            raise ValueError("prior_rate and window_s must be positive")
        if self.min_calls < 1:
            raise ValueError(f"min_calls must be at least 1; got {self.min_calls}")
        if not 0 < self.rate_safety <= 1:
            raise ValueError(f"rate_safety must be in (0, 1]; got {self.rate_safety}")


def default_phases() -> tuple[PhaseSettings, ...]:
    """Return the four phases' defaults (``configs/agent_default.yaml`` states the reasons).

    Returns:
        Phase 1 (short, low effort), phase 2 (long, the failed programs shown), phase 3 (long,
        up to 2 attempts per eligible task), phase 4 (long, cycling until the stop).
    """
    return (
        PhaseSettings(
            attempts=1,
            min_attempt_tokens=8_000,
            max_attempt_tokens=24_000,
            turn_tokens=16_384,
            max_turns=3,
            reasoning_effort="low",
            share=0.5,
            prior=False,
        ),
        PhaseSettings(
            attempts=1,
            min_attempt_tokens=16_000,
            max_attempt_tokens=65_536,
            turn_tokens=32_768,
            max_turns=5,
            reasoning_effort="xhigh",
            share=1.0,
            prior=True,
        ),
        PhaseSettings(
            attempts=2,
            min_attempt_tokens=16_000,
            max_attempt_tokens=100_000,
            turn_tokens=32_768,
            max_turns=5,
            reasoning_effort="xhigh",
            share=1.0,
            prior=True,
        ),
        PhaseSettings(
            attempts=None,
            min_attempt_tokens=16_000,
            max_attempt_tokens=65_536,
            turn_tokens=32_768,
            max_turns=5,
            reasoning_effort="xhigh",
            share=1.0,
            prior=True,
        ),
    )


@dataclass(frozen=True)
class SchedulerSettings:
    """Everything the scheduler needs besides the tasks and the harness.

    Attributes:
        phases: Phases 1 to 4, in order.
        governor: The governor's settings.
        concurrency: Attempts in flight at once.
        seed: The run's seed; each attempt's seed derives from it and its id.
        context_tokens: The engine's context window (vLLM's ``max_model_len``), or None.
        cell_timeout_s: Wall-time limit per REPL cell.
        max_cells: Code blocks run per reply.
        feedback_chars: Cap on one feedback message's body.
        min_turn_tokens: No reply is requested with less room than this.
        engine_error_retries: Extra attempts per task and phase after attempts that ended with an
            engine error or raised (phases with an attempt limit).
        cycle_error_rounds: In the phase without an attempt limit, where each attempt of a task
            is one round: consecutive rounds ending with an engine error (or raising) after
            which the task leaves the cycle; any other outcome resets the count.
        min_pairs_passed: Demonstrations a failed program must reproduce for phase 3.
        prior_programs: Earlier failed programs shown, at most (P6.2's prompt shows up to
            ``prompts.MAX_PRIOR``).
        prior_chars: Characters of one shown program with its report (P6.2 clips the report to
            ``feedback.REPORT_CHARS``; the program gets the rest).
        grace_s: Seconds after the deadline (or after the engine is gone) before running
            attempts are cancelled.
    """

    phases: tuple[PhaseSettings, ...] = field(default_factory=default_phases)
    governor: GovernorSettings = field(default_factory=GovernorSettings)
    concurrency: int = 32
    seed: int = 0
    context_tokens: int | None = None
    cell_timeout_s: float = 10.0
    max_cells: int = 4
    feedback_chars: int = 6_000
    min_turn_tokens: int = 256
    engine_error_retries: int = 1
    cycle_error_rounds: int = 2
    min_pairs_passed: int = 1
    prior_programs: int = 2
    prior_chars: int = 6_000
    grace_s: float = 60.0

    def __post_init__(self) -> None:
        """Reject settings that cannot work.

        Raises:
            ValueError: If there are not exactly four phases, a phase before the last has no
                attempt limit, or a count, a size, or a time is out of range.
        """
        if len(self.phases) != len(PHASES):
            raise ValueError(f"expected {len(PHASES)} phases; got {len(self.phases)}")
        if any(p.attempts is None for p in self.phases[:-1]):
            raise ValueError("only the last phase may run without an attempt limit")
        if not 0 <= self.prior_programs <= prompts.MAX_PRIOR:
            raise ValueError(f"prior_programs must be from 0 to {prompts.MAX_PRIOR}")
        if self.prior_chars < REPORT_CHARS + MIN_PRIOR_SOURCE:
            raise ValueError(f"prior_chars must be at least {REPORT_CHARS + MIN_PRIOR_SOURCE}")
        for name in ("concurrency", "max_cells", "feedback_chars", "min_turn_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1; got {getattr(self, name)}")
        if self.context_tokens is not None and self.context_tokens < 1:
            raise ValueError(f"context_tokens must be at least 1; got {self.context_tokens}")
        if self.engine_error_retries < 0 or self.min_pairs_passed < 1:
            raise ValueError("engine_error_retries must be >= 0 and min_pairs_passed >= 1")
        if self.cycle_error_rounds < 1:
            raise ValueError(f"cycle_error_rounds must be >= 1; got {self.cycle_error_rounds}")
        if not self.cell_timeout_s > 0 or self.grace_s < 0:
            raise ValueError("cell_timeout_s must be positive and grace_s non-negative")


# --- throughput and budgets --------------------------------------------------------------------


class RateMeter:
    r"""Aggregate generated tokens per second over a time window.

    Why: the phases' calls take minutes to tens of minutes, so counting the calls that complete
    within a window swings with every burst of completions. Here each completed call's tokens are
    spread evenly over its duration and counted in proportion to its overlap with the window
    :math:`(t - W, t]`; the calls still in flight, whose tokens arrive only at their end, count
    their time in the window at the completed calls' rate per call-second:

    .. math::

        r(t) = \frac{T_c}{S_c} \cdot \frac{S_c + S_f}{\min(W,\; t - t_0)}

    with :math:`T_c` the completed calls' tokens in the window, :math:`S_c` their call-seconds in
    it, :math:`S_f` the in-flight calls' call-seconds in it, and :math:`t_0` the first call's
    start. The first factor is the generation rate per call in flight, the second the mean number
    of calls in flight. A failed call counts neither tokens nor time.

    Args:
        window_s: The window :math:`W` in seconds.
        clock: The time source.
    """

    def __init__(self, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = window_s
        self.clock = clock
        self._open: dict[int, float] = {}
        self._done: deque[tuple[float, float, int]] = deque()
        self._next = 0
        self._since: float | None = None
        self.total_calls = 0
        self.total_tokens = 0

    def begin(self) -> int:
        """Note that a call starts.

        Returns:
            The call's handle, for :meth:`end` or :meth:`discard`.
        """
        now = self.clock()
        if self._since is None:
            self._since = now
        handle = self._next
        self._next += 1
        self._open[handle] = now
        return handle

    def end(self, handle: int, tokens: int) -> None:
        """Record a completed call.

        Args:
            handle: From :meth:`begin`.
            tokens: Its generated tokens.
        """
        start = self._open.pop(handle, None)
        if start is None:
            return
        self._done.append((start, self.clock(), int(tokens)))
        self.total_calls += 1
        self.total_tokens += int(tokens)

    def discard(self, handle: int) -> None:
        """Forget a call that failed or was cancelled.

        Args:
            handle: From :meth:`begin`.
        """
        self._open.pop(handle, None)

    def in_flight(self) -> int:
        """Return the calls started and not yet ended or discarded."""
        return len(self._open)

    def rate(self, min_calls: int = 1) -> float | None:
        """Return the measured rate, or None while the window is too thin.

        Args:
            min_calls: Completed calls that must overlap the window.

        Returns:
            Generated tokens per second, or None while fewer than ``min_calls`` completed calls
            overlap the window.
        """
        now = self.clock()
        low = now - self.window_s
        while self._done and self._done[0][1] <= low:
            self._done.popleft()
        if self._since is None:
            return None
        tokens = seconds = 0.0
        calls = 0
        for start, end, count in self._done:
            overlap = end - max(start, low)
            if overlap <= 0:
                continue
            tokens += count * overlap / (end - start)
            seconds += overlap
            calls += 1
        span = now - max(low, self._since)
        if calls < min_calls or seconds <= 0 or span <= 0:
            return None
        flight = sum(now - max(start, low) for start in self._open.values())
        return tokens / seconds * (seconds + flight) / span


class MeteredEngine:
    """An :class:`~arc26.agent.types.Engine` that reports each completed call to a meter.

    Args:
        inner: The engine.
        meter: The meter.
    """

    def __init__(self, inner: Engine, meter: RateMeter) -> None:
        self.inner = inner
        self.meter = meter

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion:
        """Generate one reply with the inner engine and record its generated tokens.

        Args:
            messages: The conversation so far.
            max_tokens: Cap on generated tokens.
            temperature: Sampling temperature.
            seed: Sampling seed, or None.
            reasoning_effort: The effort level, or None.

        Returns:
            The inner engine's reply.
        """
        handle = self.meter.begin()
        try:
            completion = await self.inner.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                reasoning_effort=reasoning_effort,
            )
        except BaseException:
            self.meter.discard(handle)
            raise
        self.meter.end(handle, completion.completion_tokens)
        return completion


class Governor:
    """Sets each attempt's token budget from the deadline and the measured rate.

    Args:
        deadline: A ``clock()`` value: the end of generation.
        settings: The governor's settings.
        meter: The rate meter of the run's engine.
        clock: The time source.
    """

    def __init__(
        self,
        deadline: float,
        settings: GovernorSettings,
        meter: RateMeter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.deadline = deadline
        self.settings = settings
        self.meter = meter
        self.clock = clock
        self._last: float | None = None

    def time_left(self) -> float:
        """Return the seconds until the deadline, at least 0."""
        return max(0.0, self.deadline - self.clock())

    def rate(self) -> tuple[float, str]:
        """Return the rate the budgets use and where it comes from.

        Returns:
            Generated tokens per second, and ``"measured"``; ``"last"`` (the last measurement,
            while the window is too thin for a new one); or ``"prior"`` (none yet).
        """
        measured = self.meter.rate(self.settings.min_calls)
        if measured is not None and measured > 0:
            self._last = measured
            return measured, "measured"
        if self._last is not None:
            return self._last, "last"
        return self.settings.prior_rate, "prior"

    def attempt_tokens(self, phase: PhaseSettings, remaining: int) -> int:
        r"""Return the token budget of an attempt that starts now.

        Args:
            phase: The attempt's phase.
            remaining: The attempts the phase still has to run, this one and those in flight
                included.

        Returns:
            :math:`s_p \sigma r T / N_p`, rounded down, within the phase's floor and cap.
        """
        rate, _ = self.rate()
        planned = phase.share * self.settings.rate_safety * rate * self.time_left()
        planned /= max(1, remaining)
        return int(min(phase.max_attempt_tokens, max(phase.min_attempt_tokens, planned)))


# --- task order ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class EaseFeatures:
    """What predicts a task's ease (design note 5.4).

    Attributes:
        cells: Cells over all its grids (demonstrations and test inputs): the prompt's size.
        same_shape: Whether every demonstration's output has its input's shape.
        colors: Distinct colors over all its grids.
    """

    cells: int
    same_shape: bool
    colors: int


def ease_features(task: AgentTask) -> EaseFeatures:
    """Return a task's ease features.

    Args:
        task: The task.

    Returns:
        Its features.
    """
    grids = [g for pair in task.train for g in (pair.input, pair.output)] + list(task.test_inputs)
    same = all(
        len(p.input) == len(p.output) and len(p.input[0]) == len(p.output[0]) for p in task.train
    )
    return EaseFeatures(
        cells=sum(len(g) * len(g[0]) for g in grids),
        same_shape=same,
        colors=len({cell for g in grids for row in g for cell in row}),
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1 in ascending order; tied values share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def ease_order(tasks: Sequence[AgentTask]) -> list[AgentTask]:
    """Order tasks from the easiest predicted to the hardest.

    Why: easy tasks first means early solves (and, with a growing library, early blocks). The
    three criteria of the design note count equally: a task's score is the sum of its ranks
    under fewer cells, the same input and output shape, and fewer colors (ties share their mean
    rank), so no weight has to be tuned.

    Args:
        tasks: The tasks.

    Returns:
        The tasks by rank sum, then cells, colors, and task id.
    """
    features = [ease_features(task) for task in tasks]
    cells = _average_ranks([f.cells for f in features])
    shapes = _average_ranks([0 if f.same_shape else 1 for f in features])
    colors = _average_ranks([f.colors for f in features])
    keys = {
        task.task_id: (cells[i] + shapes[i] + colors[i], f.cells, f.colors, task.task_id)
        for i, (task, f) in enumerate(zip(tasks, features, strict=True))
    }
    return sorted(tasks, key=lambda task: keys[task.task_id])


def _output_key(name: str) -> tuple[str, int] | None:
    """``(task_id, test_index)`` of a candidate file name ``<task>_<index>.<view ops>``."""
    base = name.rsplit("/", 1)[-1].split(".", 1)[0]
    task_id, sep, index = base.rpartition("_")
    if not sep or not task_id or not index.isdigit():
        return None
    return task_id, int(index)


def nvarc_confidence(
    test_counts: Mapping[str, int],
    candidates: Mapping[str, Any] | None = None,
    nvarc: Mapping[str, Sequence[NvarcPair]] | None = None,
    views: int = NVARC_VIEWS,
) -> dict[str, float]:
    r"""Return NVARC's confidence in each task, for phase 2's order.

    With NVARC's candidates (``candidates.json``: view file name ``<task>_<i>.<ops>`` to a list
    of ``{"solution", "beam_score", "score_aug"}``), a test output's confidence is ADR-014's
    agreement :math:`|C_{\hat y}| / n`: the view candidates equal to NVARC's top grid
    :math:`\hat y` over the :math:`n = 16` views, at most 1. :math:`\hat y` is NVARC's attempt 1
    when its submission is given, else the reference's vote (``arc26.loop.tally``). With only
    the submission, an output scores 1 when NVARC filled its attempt 1 and 0 when it left the
    ``[[0]]`` placeholder. A task's confidence is the mean over its test outputs, since the task
    score is the mean over them; an output without candidates scores 0.

    Args:
        test_counts: Task id to its number of test inputs.
        candidates: The parsed candidates file, or None.
        nvarc: NVARC's attempts per task (:func:`arc26.agent.select.load_nvarc`), or None.
        views: :math:`n`.

    Returns:
        Task id to a confidence in [0, 1]; empty when neither input is given.
    """
    if candidates is None and nvarc is None:
        return {}
    from arc26.loop import Candidate as ViewCandidate
    from arc26.loop import tally

    grouped: dict[tuple[str, int], list[ViewCandidate]] = defaultdict(list)
    for name, samples in (candidates or {}).items():
        key = _output_key(str(name))
        if key is None or key[0] not in test_counts or not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            try:
                grouped[key].append(
                    ViewCandidate(
                        sample["solution"],
                        sample.get("beam_score", 0.0),
                        sample.get("score_aug") or (0.0,),
                        str(name),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    out = {}
    for task_id, count in test_counts.items():
        pairs = (nvarc or {}).get(task_id, ())
        values = []
        for index in range(count):
            top: Any = pairs[index][0] if index < len(pairs) else None
            if candidates is None:
                values.append(0.0 if top is None else 1.0)
                continue
            found = grouped.get((task_id, index), [])
            if top is None and found:
                try:
                    top = tally(found, views).voted
                except (TypeError, ValueError):
                    top = Counter(c.grid for c in found).most_common(1)[0][0]
            target = None if top is None else tuple(tuple(row) for row in top)
            agree = sum(1 for c in found if c.grid == target)
            values.append(min(1.0, agree / views))
        out[task_id] = sum(values) / len(values) if values else 0.0
    return out


# --- what the records say about a task ---------------------------------------------------------


def _mean_accuracy(verification: Verification) -> float:
    pairs = verification.pairs
    return sum(p.cell_accuracy for p in pairs) / len(pairs) if pairs else 0.0


def task_solved(records: Iterable[AttemptRecord]) -> bool:
    """Return whether any attempt produced a verified program.

    Args:
        records: One task's records.

    Returns:
        True when a verification is verified.
    """
    return any(v.verified for r in records for v in r.verifications)


def failed_programs(records: Iterable[AttemptRecord]) -> list[Verification]:
    """Return a task's programs that were not accepted and have no fatal guard finding.

    Why: a later phase starts from what failed; the prompt shows the last two of this list
    (``prompts.task_message``), so the best come last. Hardcoded programs are left out: showing
    them would invite more of the same.

    Args:
        records: One task's records.

    Returns:
        One verification per distinct source, ordered by demonstrations passed, then mean cell
        accuracy, ascending (ties keep their first-seen order).
    """
    seen: dict[str, Verification] = {}
    for record in records:
        for v in record.verifications:
            if v.verified or any(g.fatal for g in v.guards):
                continue
            seen.setdefault(v.source_sha256, v)
    return sorted(seen.values(), key=lambda v: (v.passed, _mean_accuracy(v)))


def verified_groups(records: Iterable[AttemptRecord], test_count: int) -> int:
    """Return the most distinct verified outputs any one test input has.

    Args:
        records: One task's records.
        test_count: Its number of test inputs.

    Returns:
        0 without verified programs, 1 when they all agree, more when they disagree.
    """
    verified = [v for r in records for v in r.verifications if v.verified]
    most = 0
    for index in range(test_count):
        outputs = {
            json.dumps(v.test_outputs[index])
            for v in verified
            if index < len(v.test_outputs) and v.test_outputs[index] is not None
        }
        most = max(most, len(outputs))
    return most


# --- the crash-safe submission -----------------------------------------------------------------


class SubmissionWriter:
    """Keeps ``submission.json`` valid and current during the run.

    Every task starts with NVARC's attempts (or ``[[0]]``); each new record re-selects its task
    (:func:`arc26.agent.select.build_selection`) and the whole file is rewritten atomically
    (:func:`arc26.agent.select.write_submission`: validated, temporary file, rename), so a
    reader, or a run killed at any moment, sees a complete valid file. A failed write leaves the
    previous file and is counted; the run goes on.

    Args:
        path: The submission file.
        test_counts: Task id to its number of test inputs (the challenges' tasks).
        nvarc: NVARC's attempts per task, or None.
        records: Records already logged (a resumed run).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        test_counts: Mapping[str, int],
        nvarc: Mapping[str, Sequence[NvarcPair]] | None = None,
        records: Iterable[AttemptRecord] = (),
    ) -> None:
        self.path = Path(path)
        self.counts = dict(test_counts)
        self.nvarc = dict(nvarc or {})
        self._records: dict[str, list[AttemptRecord]] = defaultdict(list)
        for record in records:
            if record.task_id in self.counts:
                self._records[record.task_id].append(record)
        self._selection: dict[str, list[list[Candidate]]] = {}
        for task_id in self.counts:
            self._selection[task_id] = self._select(task_id) or []
        self.writes = 0
        self.failures = 0
        self.last_error: str | None = None

    def _select(self, task_id: str) -> list[list[Candidate]] | None:
        try:
            selection = build_selection(
                {task_id: self.counts[task_id]}, self._records.get(task_id, ()), self.nvarc
            )
        except Exception:
            logger.exception("selecting the attempts of %s failed; the previous ones stay", task_id)
            return None
        return selection[task_id]

    def _chosen(self, task_id: str) -> list[list[Candidate]]:
        """The task's chosen candidates, one list per test input (empty where none)."""
        count = self.counts[task_id]
        return (list(self._selection.get(task_id) or []) + [[] for _ in range(count)])[:count]

    def add(self, record: AttemptRecord) -> None:
        """Add a finished attempt: re-select its task and rewrite the file.

        Args:
            record: The record; a record of a task outside the challenges is ignored.
        """
        if record.task_id not in self.counts:
            return
        self._records[record.task_id].append(record)
        selected = self._select(record.task_id)
        if selected is not None:
            self._selection[record.task_id] = selected
        self.write()

    def submission(self) -> dict[str, list[dict[str, Grid]]]:
        """Return the current submission, in the challenges' task order.

        Returns:
            Task id to one ``{"attempt_1", "attempt_2"}`` per test input.
        """
        chosen = {task_id: self._chosen(task_id) for task_id in self.counts}
        return submission_from_selection(chosen)

    def write(self) -> bool:
        """Write the current submission atomically.

        Returns:
            Whether the write succeeded; on failure the previous file stays in place.
        """
        try:
            write_submission(self.path, self.submission(), self.counts)
        except Exception as err:
            self.failures += 1
            self.last_error = f"{type(err).__name__}: {err}"
            logger.exception("writing %s failed; the previous file stays", self.path)
            return False
        self.writes += 1
        return True

    def slots(self) -> dict[str, int]:
        """Return how many attempt slots each source fills now.

        Returns:
            Counts for ``program``, ``direct``, ``nvarc``, and ``empty``.
        """
        slots = {"program": 0, "direct": 0, "nvarc": 0, "empty": 0}
        for task_id in self.counts:
            for entry in self._chosen(task_id):
                for candidate in entry[:2]:
                    slots[candidate.source] += 1
                slots["empty"] += 2 - len(entry[:2])
        return slots


# --- the scheduler ------------------------------------------------------------------------------


class EngineGate(Protocol):
    """Tells the scheduler whether the engine can take calls (``EngineSupervisor``)."""

    async def wait_ready(self) -> bool:
        """Wait until the engine is up.

        Returns:
            True when it is up, False once it is gone for good.
        """
        ...


@dataclass
class SchedulerReport:
    """What the scheduler did, for ``run.json``.

    Attributes:
        resumed: Records read back from an earlier run of the same log.
        started: Attempts started, per phase.
        finished: Attempts logged, per phase.
        verified: Attempts that produced a verified program, per phase.
        stops: Stop reasons per phase.
        tokens: Generated tokens per phase.
        budgets: The token budget each attempt started with, by attempt id.
        plans: What each budget was computed from, by attempt id: the attempts the phase still
            had to run (:math:`N_p`), the rate and its source, and the seconds left.
        failed: Attempts that raised (a REPL or verifier fault) and left no record.
        cancelled: Attempts cancelled after the grace period.
        stopped: Why the scheduler stopped: ``done``, ``deadline``, or ``engine``.
        rate: The last rate the governor used, and its source.
    """

    resumed: int = 0
    started: Counter[int] = field(default_factory=Counter)
    finished: Counter[int] = field(default_factory=Counter)
    verified: Counter[int] = field(default_factory=Counter)
    stops: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    tokens: Counter[int] = field(default_factory=Counter)
    budgets: dict[str, int] = field(default_factory=dict)
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: int = 0
    cancelled: int = 0
    stopped: str = ""
    rate: tuple[float, str] = (0.0, "prior")

    def as_dict(self) -> dict[str, Any]:
        """Return the report as plain JSON values.

        Returns:
            The fields, phases as string keys.
        """
        return {
            "resumed": self.resumed,
            "started": {str(p): self.started[p] for p in PHASES},
            "finished": {str(p): self.finished[p] for p in PHASES},
            "verified": {str(p): self.verified[p] for p in PHASES},
            "stops": {str(p): dict(self.stops[p]) for p in PHASES},
            "completion_tokens": {str(p): self.tokens[p] for p in PHASES},
            "budgets": dict(self.budgets),
            "plans": dict(self.plans),
            "failed": self.failed,
            "cancelled": self.cancelled,
            "stopped": self.stopped,
            "rate": {"tokens_per_s": round(self.rate[0], 3), "source": self.rate[1]},
        }


@dataclass
class _TaskState:
    task: AgentTask
    rank: int
    records: list[AttemptRecord] = field(default_factory=list)
    busy: bool = False
    raised: Counter[int] = field(default_factory=Counter)
    # Consecutive attempts per phase that ended with an engine error or raised, at the end of
    # the task's history (any other outcome resets it): the cycle's drop rule.
    streak: Counter[int] = field(default_factory=Counter)


@dataclass(frozen=True)
class _Job:
    state: _TaskState
    phase: int
    attempt_id: str
    budget: Budget
    prior: tuple[Verification, ...]


def _empty(record: AttemptRecord) -> bool:
    """Whether the attempt's first prompt left no room for a reply: no work was done."""
    return record.stop_reason == "budget" and record.turns == 0


def _complete(record: AttemptRecord) -> bool:
    """Whether the attempt ran its course (it counts toward its phase's ``attempts``)."""
    return record.stop_reason not in INCOMPLETE and not _empty(record)


def clip_program(verification: Verification, chars: int) -> Verification:
    """Return a failed program for the prompt, clipped to ``chars`` characters with its report.

    Why: a model can write programs of tens of thousands of characters; shown in full, two of
    them leave a later attempt's prompt no room for a reply.

    Args:
        verification: The program's verification.
        chars: The characters of the program and its report together (P6.2 clips the report to
            ``REPORT_CHARS``; the source gets the rest, at least ``MIN_PRIOR_SOURCE``).

    Returns:
        The verification with its source clipped around a ``[... n characters cut ...]`` marker,
        or the verification itself when it fits.
    """
    room = max(MIN_PRIOR_SOURCE, chars - REPORT_CHARS)
    if len(verification.source) <= room:
        return verification
    return dataclasses.replace(verification, source=clip(verification.source, room))


def _count_streak(state: _TaskState, record: AttemptRecord) -> None:
    """Extend the task's engine-error streak in the record's phase, or reset it."""
    if record.stop_reason == "engine_error":
        state.streak[record.phase] += 1
    else:
        state.streak[record.phase] = 0


def _close_quietly(repl: Repl, attempt_id: str) -> None:
    try:
        repl.close()
    except Exception:
        logger.exception("closing the REPL of %s failed", attempt_id)


def _terminate_partial_line(path: Path) -> None:
    """End a JSONL's cut last line (a write killed mid-way) so the next record starts clean."""
    try:
        with open(path, "rb+") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    except FileNotFoundError:
        return


class Scheduler:
    """Runs the phases over the tasks until they are done, the deadline, or the engine is gone.

    Args:
        tasks: The tasks (any order; the scheduler orders them).
        engine: The chat engine; the scheduler meters it.
        repl_factory: Starts a REPL for a task; called in a worker thread.
        verifier: The program verifier, called from worker threads.
        settings: The phases, the governor, and the limits.
        deadline: A ``clock()`` value: no attempt starts after it, and running ones stop.
        clock: The time source (``time.monotonic``; tests pass a fake). Attempts get the
            deadline converted to ``time.monotonic()`` when they start.
        records: Records of an earlier run of the same log (resume): their attempts count as
            done and feed the order, the prior programs, and the budgets.
        confidence: NVARC's confidence per task (:func:`nvarc_confidence`), or None.
        log_path: The attempt JSONL, appended after every attempt, or None.
        transcript_path: A JSONL of conversations, one line per attempt, or None.
        on_record: Called with each record after it is logged (the submission writer); its
            exceptions are logged.
        gate: The engine's readiness (``EngineSupervisor``), or None for always ready.
        system: The system prompt; :func:`arc26.agent.prompts.system_prompt` when None.
    """

    def __init__(
        self,
        tasks: Sequence[AgentTask],
        engine: Engine,
        repl_factory: Callable[[AgentTask], Repl],
        verifier: Verifier,
        settings: SchedulerSettings,
        *,
        deadline: float,
        clock: Callable[[], float] = time.monotonic,
        records: Iterable[AttemptRecord] = (),
        confidence: Mapping[str, float] | None = None,
        log_path: str | os.PathLike[str] | None = None,
        transcript_path: str | os.PathLike[str] | None = None,
        on_record: Callable[[AttemptRecord], None] | None = None,
        gate: EngineGate | None = None,
        system: str | None = None,
    ) -> None:
        self.settings = settings
        self.deadline = deadline
        self.clock = clock
        self.meter = RateMeter(settings.governor.window_s, clock)
        self.governor = Governor(deadline, settings.governor, self.meter, clock)
        self._engine = MeteredEngine(engine, self.meter)
        self._repl_factory = repl_factory
        self._verifier = verifier
        self._confidence = dict(confidence or {})
        self._log_path = None if log_path is None else Path(log_path)
        self._transcript_path = None if transcript_path is None else Path(transcript_path)
        self._on_record = on_record
        self._gate = gate
        self._system = prompts.system_prompt() if system is None else system
        self._states: dict[str, _TaskState] = {
            task.task_id: _TaskState(task, rank) for rank, task in enumerate(ease_order(tasks))
        }
        self._ids: set[str] = set()
        self.report = SchedulerReport()
        for record in records:
            self._ids.add(record.attempt_id)
            state = self._states.get(record.task_id)
            if state is not None:
                state.records.append(record)
                _count_streak(state, record)
                self.report.resumed += 1
        self._executor: ThreadPoolExecutor | None = None

    # The phase rules.

    def _made(self, state: _TaskState, phase: int) -> int:
        """Attempts of this phase logged for the task, whatever their stop reason."""
        return sum(1 for r in state.records if r.phase == phase)

    def _left(self, state: _TaskState, phase: int) -> int:
        """Attempts the task still has to run in this phase (at least 1 while it needs one).

        A phase without an attempt limit counts one per task: the attempt of the current round.
        """
        attempts = self.settings.phases[phase - 1].attempts
        if attempts is None:
            return 1
        complete = sum(1 for r in state.records if r.phase == phase and _complete(r))
        return max(1, attempts - complete)

    def _exhausted(self, state: _TaskState, phase: int) -> bool:
        own = [r for r in state.records if r.phase == phase]
        empties = sum(1 for r in own if _empty(r))
        attempts = self.settings.phases[phase - 1].attempts
        if attempts is None:
            # The cycle runs for hours: errors count per round (one attempt of the task), and only
            # consecutive errored rounds drop the task, never errors hours apart.
            errored = state.streak[phase] >= self.settings.cycle_error_rounds
            return errored or empties >= EMPTY_LIMIT
        if sum(1 for r in own if _complete(r)) >= attempts:
            return True
        errors = sum(1 for r in own if r.stop_reason == "engine_error") + state.raised[phase]
        return errors > self.settings.engine_error_retries or empties >= EMPTY_LIMIT

    def _disagree(self, state: _TaskState) -> bool:
        """Whether the task's verified programs disagree on a test output."""
        return verified_groups(state.records, len(state.task.test_inputs)) > 1

    def _eligible_late(self, state: _TaskState) -> bool:
        """Phase 3's rule: partial progress, or verified programs that disagree."""
        if task_solved(state.records):
            return self._disagree(state)
        failed = failed_programs(state.records)
        return bool(failed) and failed[-1].passed >= self.settings.min_pairs_passed

    def _eligible_cycle(self, state: _TaskState) -> bool:
        """Phase 4's rule: unsolved, or verified programs that disagree."""
        return not task_solved(state.records) or self._disagree(state)

    def _needs(self, state: _TaskState, phase: int) -> bool:
        """Whether the task still needs an attempt of this phase (busy or not).

        Phase 1 serves every task, solved or not, so ``attempts`` above 1 gives independent
        votes; phase 2 only unsolved tasks; phases 3 and 4 only their eligible ones, each after
        the phases before it.
        """
        if phase == 1:
            return not self._exhausted(state, 1)
        if self._needs(state, 1):
            return False
        if phase == 2:
            return not task_solved(state.records) and not self._exhausted(state, 2)
        if self._needs(state, 2):
            return False
        if phase == 3:
            return self._eligible_late(state) and not self._exhausted(state, 3)
        return (
            not self._needs(state, 3)
            and self._eligible_cycle(state)
            and not self._exhausted(state, 4)
        )

    def _remaining(self, phase: int) -> int:
        """:math:`N_p`: the attempts phase ``phase`` still has to run, in flight included.

        Phase 2 also counts the unsolved tasks still in phase 1, since each may need it.
        """
        if phase == 2:
            return sum(
                self._left(s, 2)
                for s in self._states.values()
                if not task_solved(s.records) and not self._exhausted(s, 2)
            )
        return sum(self._left(s, phase) for s in self._states.values() if self._needs(s, phase))

    def _order(self, state: _TaskState, phase: int) -> tuple[float, ...]:
        """Sort key of the phase's candidates: fewest attempts made first, then the phase's rule.

        Phase 4 is a round robin over all the task's attempts so far, unsolved tasks before
        tasks with disagreeing programs at equal counts.
        """
        made = self._made(state, phase)
        if phase == 1:
            return (made, state.rank)
        if phase == 2:
            return (made, self._confidence.get(state.task.task_id, 0.0), state.rank)
        if phase == 4:
            return (len(state.records), float(task_solved(state.records)), state.rank)
        if task_solved(state.records):
            groups = verified_groups(state.records, len(state.task.test_inputs))
            return (made, 1.0, -groups, 0.0, state.rank)
        best = failed_programs(state.records)[-1]
        share = best.passed / len(best.pairs) if best.pairs else 0.0
        return (made, 0.0, -share, -_mean_accuracy(best), state.rank)

    def _next(self) -> tuple[_TaskState, int] | None:
        """The task and phase to start next, or None when nothing can start now."""
        for phase in PHASES:
            ready = [s for s in self._states.values() if not s.busy and self._needs(s, phase)]
            if ready:
                return min(ready, key=lambda s: self._order(s, phase)), phase
        return None

    def _job(self, state: _TaskState, phase: int) -> _Job:
        settings = self.settings.phases[phase - 1]
        k = 0
        while f"{state.task.task_id}/p{phase}/{k}" in self._ids:
            k += 1
        remaining = self._remaining(phase)
        tokens = self.governor.attempt_tokens(settings, remaining)
        rate, source = self.governor.rate()
        self.report.plans[f"{state.task.task_id}/p{phase}/{k}"] = {
            "remaining": remaining,
            "rate": round(rate, 3),
            "rate_source": source,
            "time_left_s": round(self.governor.time_left(), 1),
        }
        budget = Budget(
            max_turns=settings.max_turns,
            attempt_tokens=tokens,
            turn_tokens=min(settings.turn_tokens, tokens),
            min_turn_tokens=min(self.settings.min_turn_tokens, tokens),
            context_tokens=self.settings.context_tokens,
            temperature=settings.temperature,
            reasoning_effort=settings.reasoning_effort,
            cell_timeout_s=self.settings.cell_timeout_s,
            max_cells=self.settings.max_cells,
            feedback_chars=self.settings.feedback_chars,
            preserve_thinking=settings.preserve_thinking,
        )
        prior: tuple[Verification, ...] = ()
        # After an attempt whose prompt left no room for a reply, the next goes without them.
        crowded = any(_empty(r) for r in state.records if r.phase == phase)
        shown = self.settings.prior_programs
        if settings.prior and shown and not crowded and not task_solved(state.records):
            prior = tuple(
                clip_program(v, self.settings.prior_chars)
                for v in failed_programs(state.records)[-shown:]
            )
        return _Job(state, phase, f"{state.task.task_id}/p{phase}/{k}", budget, prior)

    # Running attempts.

    def _attempt_deadline(self) -> float:
        """The deadline as a ``time.monotonic()`` value, which ``run_attempt`` uses."""
        return time.monotonic() + max(0.0, self.deadline - self.clock())

    async def _engine_stop(self) -> str | None:
        """None when the engine is up; else why to stop starting attempts."""
        if self._gate is None:
            return None
        left = self.deadline - self.clock()
        if left <= 0:
            return STOP_DEADLINE
        try:
            ready = await asyncio.wait_for(self._gate.wait_ready(), timeout=left)
        except TimeoutError:
            return STOP_DEADLINE
        return None if ready else STOP_ENGINE

    async def _close(self, repl: Repl, attempt_id: str) -> None:
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, repl.close)
        except Exception:
            logger.exception("closing the REPL of %s failed", attempt_id)

    async def _run_job(self, job: _Job) -> None:
        state = job.state
        loop = asyncio.get_running_loop()
        transcript: list[dict[str, Any]] | None = None
        if self._transcript_path is not None:
            transcript = []
        try:
            repl = await loop.run_in_executor(self._executor, self._repl_factory, state.task)
            try:
                record = await run_attempt(
                    state.task,
                    self._engine,
                    repl,
                    self._verifier,
                    job.budget,
                    attempt_id=job.attempt_id,
                    phase=job.phase,
                    seed=attempt_seed(self.settings.seed, job.attempt_id),
                    deadline=self._attempt_deadline(),
                    system=self._system,
                    prior=job.prior,
                    executor=self._executor,
                    transcript=transcript,
                )
            except asyncio.CancelledError:
                # Cancelled past the grace: the executor's threads may all be stuck in REPL
                # calls, so a daemon thread closes this REPL without holding up the cut.
                closer = threading.Thread(target=_close_quietly, args=(repl, job.attempt_id))
                closer.daemon = True
                closer.start()
                raise
            except BaseException:
                await self._close(repl, job.attempt_id)
                raise
            await self._close(repl, job.attempt_id)
        except Exception:
            logger.exception("attempt %s failed", job.attempt_id)
            state.raised[job.phase] += 1
            state.streak[job.phase] += 1
            self.report.failed += 1
            return
        finally:
            state.busy = False
        self._accept(state, record, transcript)

    def _accept(
        self,
        state: _TaskState,
        record: AttemptRecord,
        transcript: list[dict[str, Any]] | None,
    ) -> None:
        state.records.append(record)
        _count_streak(state, record)
        self._ids.add(record.attempt_id)
        report = self.report
        report.finished[record.phase] += 1
        report.stops[record.phase][record.stop_reason] += 1
        report.tokens[record.phase] += record.completion_tokens
        if any(v.verified for v in record.verifications):
            report.verified[record.phase] += 1
        try:
            if self._log_path is not None:
                append_record(self._log_path, record)
            if self._transcript_path is not None and transcript is not None:
                line = {
                    "task_id": record.task_id,
                    "attempt_id": record.attempt_id,
                    "messages": transcript,
                }
                self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._transcript_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line, separators=(",", ":")) + "\n")
        except Exception:
            logger.exception("writing the logs of %s failed", record.attempt_id)
        if self._on_record is not None:
            try:
                self._on_record(record)
            except Exception:
                logger.exception("on_record failed for %s", record.attempt_id)

    async def _fill(self, running: set[asyncio.Task[None]]) -> str | None:
        """Start attempts while slots are free; return why to stop starting, or None."""
        while len(running) < self.settings.concurrency:
            if self.clock() >= self.deadline:
                return STOP_DEADLINE
            # Nothing to start: do not wait on an engine that is restarting.
            if self._next() is None:
                return None
            stop = await self._engine_stop()
            if stop is not None:
                return stop
            # Chosen after the wait, so the budget uses the time then left.
            chosen = self._next()
            if chosen is None:
                return None
            job = self._job(*chosen)
            job.state.busy = True
            self.report.started[job.phase] += 1
            self.report.budgets[job.attempt_id] = job.budget.attempt_tokens
            plan = self.report.plans[job.attempt_id]
            self.report.rate = (plan["rate"], plan["rate_source"])
            running.add(asyncio.get_running_loop().create_task(self._run_job(job)))
        return None

    async def run(self) -> SchedulerReport:
        """Run until every phase is done, the deadline passes, or the engine is gone.

        Returns:
            The report; :attr:`report` holds the same object.
        """
        for path in (self._log_path, self._transcript_path):
            if path is not None:
                _terminate_partial_line(path)
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.concurrency, thread_name_prefix="arc26-agent"
        )
        running: set[asyncio.Task[None]] = set()
        stop: str | None = None
        hard_end: float | None = None
        try:
            while True:
                if stop is None:
                    # Checked here too: with every slot busy, _fill starts nothing and returns.
                    if self.clock() >= self.deadline:
                        stop = STOP_DEADLINE
                    else:
                        stop = await self._fill(running)
                    if stop is not None:
                        hard_end = time.monotonic() + self.settings.grace_s
                if not running:
                    break
                if hard_end is None:
                    timeout = max(0.0, self.deadline - self.clock())
                else:
                    timeout = max(0.0, hard_end - time.monotonic())
                done, _ = await asyncio.wait(
                    running, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                running -= done
                if not done and hard_end is not None and time.monotonic() >= hard_end:
                    self.report.cancelled += len(running)
                    await self._cancel(running)
                    break
        finally:
            if running:
                await self._cancel(running)
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.report.stopped = stop or STOP_DONE
        logger.info("scheduler stopped: %s", self.report.stopped)
        return self.report

    @staticmethod
    async def _cancel(running: set[asyncio.Task[None]]) -> None:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        running.clear()
