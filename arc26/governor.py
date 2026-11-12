r"""Time governor for the refinement loop (task P2.4; BRIEF B5 P2.4, B2 Q4, B3, B7 G1).

The refinement loop (P2.3) and later H's polish (P4) may only spend the time the baseline leaves
free, and a run that passes the 12 h limit loses tasks. The governor prices each loop operation
with a conservative cost model fitted on a measured run (``configs/governor.yaml``, from run
``P1.1-01``), gives each puzzle a share of the remaining wall time, and picks one round variant
per puzzle from a fallback ladder: a gradient round if it fits, else an in-context round if it
fits, else none (B2 Q4). It can also cap the baseline's decode batches to make room for a round,
an explicit trade that the human enables at G1.

Cost model, per op (op names from :mod:`arc26.timing`):

- ``train``, per fit: :math:`a + bT + cT^2/S` for :math:`T` tokens in :math:`S` optimizer steps
  (the quadratic term is the attention cost per step), plus an allowance for the fit's residual.
- ``decode``, per DFS batch (one ``inference_turbo_dfs`` call on 4 augmented views): a high
  quantile of the measured seconds per batch, since decode time does not follow a size axis.
- ``selection``, per rescored candidate: a high quantile of the measured seconds per candidate.

Every estimate is multiplied by :math:`1 + m` for the margin :math:`m`. The per-puzzle budget is
:math:`(t_\text{end} - r - t) / (1 + \lceil q / w \rceil)` for the clock :math:`t`, deadline
:math:`t_\text{end}`, straggler reserve :math:`r`, :math:`q` queued task ids (the reference's queue
also holds one ``None`` per worker, which the budget leaves out), and :math:`w` workers. See
``docs/modules/governor.md`` for the derivation, the call pattern, and the limits.
Pure Python, no GPU code; the governor never tokenizes (token counts are the caller's inputs).
"""

from __future__ import annotations

import math
import operator
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from arc26.timing import DECODE, SELECTION, TRAIN

GRADIENT = "gradient"
IN_CONTEXT = "in_context"
NONE = "none"
# Tried in this order (B2 Q4): the in-context round is the fallback when the gradient round does
# not fit, not the cheaper first choice.
LADDER = (GRADIENT, IN_CONTEXT)
VARIANTS = (*LADDER, NONE)
REFERENCE = "reference"
TRADE = "trade"
POLICIES = (REFERENCE, TRADE)
# The reference decodes each test input in 4 DFS batches of 4 augmented views (arc_solver.worker).
REFERENCE_BATCHES_PER_INPUT = 4
OPS = (TRAIN, DECODE, SELECTION)
_UNITS = {TRAIN: "fit", DECODE: "batch", SELECTION: "candidate"}


def _count(name: str, value: object) -> int:
    """A non-negative integer; numpy and torch integer scalars convert via ``__index__``."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    try:
        number = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be an integer; got {value!r}") from None
    if number < 0:
        raise ValueError(f"{name} must be non-negative; got {number}")
    return number


def _number(name: str, value: object, *, minimum: float | None = None) -> float:
    """A finite float, at least ``minimum`` when given; booleans are refused."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value!r}")
    return number


@dataclass(frozen=True)
class OpCost:
    r"""Conservative seconds for one unit of an op, before the margin.

    With a token axis, a unit of :math:`T` tokens in :math:`S` optimizer steps costs
    :math:`\max(0, a + bT + cT^2/S) + \text{allowance}`; zero tokens cost 0 (no fit runs).
    Without one, every unit costs :math:`a + \text{allowance}`, whatever its tokens.

    Attributes:
        unit: What one unit is: ``fit``, ``batch``, or ``candidate``.
        a: Intercept in seconds; without a token axis, the seconds per unit (a high quantile).
        b: Seconds per token; 0 without a token axis.
        c: Seconds per squared token per step (the attention term); 0 for a linear fit.
        steps: Default optimizer steps :math:`S` of the quadratic term.
        allowance: Seconds added per unit, for example the fit's worst residual.
        token_axis: Whether the cost depends on tokens.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is not finite, ``steps < 1``, ``allowance < 0``, ``b`` or ``c``
            is negative, an op without a token axis has ``b`` or ``c``, or ``a < 0`` without a
            token axis.
    """

    unit: str
    a: float
    b: float = 0.0
    c: float = 0.0
    steps: int = 1
    allowance: float = 0.0
    token_axis: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str) or not self.unit:
            raise TypeError(f"unit must be a non-empty string; got {self.unit!r}")
        if not isinstance(self.token_axis, bool):
            raise TypeError(f"token_axis must be a bool; got {self.token_axis!r}")
        for name in ("a", "b", "c", "allowance"):
            object.__setattr__(self, name, _number(name, getattr(self, name)))
        object.__setattr__(self, "steps", _count("steps", self.steps))
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1; got {self.steps}")
        if self.b < 0 or self.c < 0 or self.allowance < 0:
            raise ValueError("b, c, and allowance must be non-negative")
        if not self.token_axis and (self.b or self.c):
            raise ValueError("an op without a token axis has no b or c")
        if not self.token_axis and self.a < 0:
            raise ValueError(f"seconds per unit must be non-negative; got {self.a}")

    def seconds(self, tokens: int | None = None, steps: int | None = None) -> float:
        """Conservative seconds of one unit, before the margin.

        Args:
            tokens: Tokens of the unit; required with a token axis, ignored without one.
            steps: Optimizer steps of the quadratic term; ``self.steps`` when None.

        Returns:
            The seconds, never negative.

        Raises:
            TypeError: If ``tokens`` or ``steps`` is not an integer.
            ValueError: If ``tokens`` is missing with a token axis, or ``steps < 1``.
        """
        if tokens is not None:
            tokens = _count("tokens", tokens)
        if not self.token_axis:
            return self.a + self.allowance
        if tokens is None:
            raise ValueError(f"the {self.unit} cost needs a token count")
        if tokens == 0:
            return 0.0
        n_steps = self.steps if steps is None else _count("steps", steps)
        if n_steps < 1:
            raise ValueError(f"steps must be at least 1; got {n_steps}")
        fit = self.a + self.b * tokens + self.c * tokens * tokens / n_steps
        return max(0.0, fit) + self.allowance


class CostModel:
    r"""Conservative per-op cost model with a margin (B5 P2.4).

    Why: the governor must never spend time it does not have, so every decision uses a
    conservative estimate, a high quantile or a fit plus an allowance, times :math:`1 + m`, never
    the mean alone. The coefficients come from a committed config with their provenance.

    Attributes:
        ops: :class:`OpCost` per op name (``train``, ``decode``, ``selection``).
        margin: Multiplicative margin :math:`m`; every estimate is :math:`(1 + m)` times the raw
            conservative cost.
        candidates_per_batch: Rescored candidates assumed per decode batch when the caller cannot
            count them yet (before decoding).
        overhead: Seconds per round outside train, decode, and selection (dataset build, adapter
            swap), before the margin.
        provenance: Free-form record of where the coefficients come from (run id, commits, fit
            :math:`R^2`, :math:`n`), for the loop's log.
    """

    def __init__(
        self,
        ops: Mapping[str, OpCost],
        margin: float,
        candidates_per_batch: int,
        overhead: float = 0.0,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Build a cost model.

        Args:
            ops: An :class:`OpCost` for each of ``train``, ``decode``, and ``selection``.
            margin: Multiplicative margin, at least 0.
            candidates_per_batch: Candidates assumed per decode batch, at least 0.
            overhead: Seconds per round, at least 0.
            provenance: Where the coefficients come from; copied.

        Raises:
            TypeError: If an op is not an :class:`OpCost` or a number has the wrong type.
            ValueError: If an op is missing or unknown, ``train`` has no token axis, ``decode``
                or ``selection`` has one, or a number is out of range.
        """
        missing = [op for op in OPS if op not in ops]
        unknown = [op for op in ops if op not in OPS]
        if missing or unknown:
            raise ValueError(f"ops must be exactly {OPS}; missing {missing}, unknown {unknown}")
        for op, cost in ops.items():
            if not isinstance(cost, OpCost):
                raise TypeError(f"ops[{op!r}] must be an OpCost; got {cost!r}")
            if cost.token_axis != (op == TRAIN):
                raise ValueError(f"{op}: only train has a token axis in this model")
        self.ops: Mapping[str, OpCost] = MappingProxyType(dict(ops))
        self.margin = _number("margin", margin, minimum=0.0)
        self.candidates_per_batch = _count("candidates_per_batch", candidates_per_batch)
        self.overhead = _number("overhead", overhead, minimum=0.0)
        self.provenance: Mapping[str, Any] = MappingProxyType(dict(provenance or {}))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> CostModel:
        """Build the model from a parsed ``configs/governor.yaml``.

        Args:
            config: The parsed config: ``ops`` (``train`` with ``a``, ``b``, optional ``c``,
                ``steps``, optional ``allowance``; ``decode`` and ``selection`` with ``seconds``),
                ``margin``, ``candidates_per_batch``, optional ``overhead`` and ``provenance``.
                Other top-level keys (the deadline, the decode policy) are read by
                :meth:`Governor.from_config`.

        Returns:
            The cost model.

        Raises:
            KeyError: If a required key is missing.
            TypeError: As :class:`OpCost` and :class:`CostModel` raise it.
            ValueError: As :class:`OpCost` and :class:`CostModel` raise it, or for an unknown
                key in an op's section.
        """
        sections = config["ops"]
        ops: dict[str, OpCost] = {}
        train = _section(sections, TRAIN, {"a", "b", "c", "steps", "allowance"}, {"a", "b"})
        ops[TRAIN] = OpCost(
            unit=_UNITS[TRAIN],
            a=train["a"],
            b=train["b"],
            c=train.get("c", 0.0),
            steps=train.get("steps", 1),
            allowance=train.get("allowance", 0.0),
            token_axis=True,
        )
        for op in (DECODE, SELECTION):
            section = _section(sections, op, {"seconds", "allowance"}, {"seconds"})
            ops[op] = OpCost(
                unit=_UNITS[op], a=section["seconds"], allowance=section.get("allowance", 0.0)
            )
        return cls(
            ops,
            margin=config["margin"],
            candidates_per_batch=config["candidates_per_batch"],
            overhead=config.get("overhead", 0.0),
            provenance=config.get("provenance"),
        )

    @classmethod
    def from_config(cls, path: str | os.PathLike[str]) -> CostModel:
        """Load the model from a YAML config such as ``configs/governor.yaml``.

        Args:
            path: The config file.

        Returns:
            The cost model.

        Raises:
            ImportError: If PyYAML is missing (it ships with the Kaggle image as a dependency of
                transformers, and with the ``dev`` extras).
            OSError: If the file cannot be read.
            KeyError: As :meth:`from_mapping` raises it.
            TypeError: As :meth:`from_mapping` raises it.
            ValueError: As :meth:`from_mapping` raises it.
        """
        return cls.from_mapping(_load_yaml(path))

    def estimate(
        self,
        op: str,
        tokens: int | None = None,
        *,
        count: int = 1,
        steps: int | None = None,
    ) -> float:
        r"""Conservative seconds for ``count`` units of an op, margin included.

        Args:
            op: ``train`` (unit: one fit), ``decode`` (unit: one DFS batch of 4 views), or
                ``selection`` (unit: one rescored candidate).
            tokens: Tokens per unit. Required for ``train`` (all tokens of the fit; 0 means no
                fit runs and costs 0); ignored by ``decode`` and ``selection``, which have no
                token axis in run ``P1.1-01``.
            count: Number of units, for example decode batches; 0 costs 0.
            steps: Optimizer steps of a ``train`` fit; the config's steps (the reference's 128)
                when None.

        Returns:
            :math:`(1 + m) \cdot \text{count}` times the op's seconds per unit.

        Raises:
            ValueError: If ``op`` is unknown, a count is negative, or ``train`` has no tokens.
            TypeError: If ``tokens``, ``count``, or ``steps`` is not an integer.
        """
        if op not in self.ops:
            raise ValueError(f"unknown op {op!r}; expected one of {OPS}")
        units = _count("count", count)
        per_unit = self.ops[op].seconds(tokens, steps)
        return (1.0 + self.margin) * units * per_unit

    def decode_cost(
        self, batches: int, length_ratio: float = 1.0, observed_batch: float | None = None
    ) -> float:
        """Conservative seconds to decode ``batches`` DFS batches and rescore their candidates.

        Why: decode and its rescoring always run together, batch by batch, so the decode trade
        and the round costs price them as one unit. Decode time is mostly a property of the
        puzzle (in ``P1.1-01``, 77% of the variance of the seconds per batch lies between
        puzzles), so once a puzzle has decoded in round 0, a round prices each batch at no less
        than the slowest batch it ran.

        Args:
            batches: DFS batches.
            length_ratio: Prompt tokens over the round-0 prompt tokens (at least 1), for prompts
                that carry extra pairs (the in-context round). It scales decode and selection
                linearly, which overestimates: a DFS step's cost is affine in the prompt length
                with a positive intercept (the weights' read).
            observed_batch: Seconds of the slowest DFS batch this puzzle ran in round 0, or None
                before its first batch.

        Returns:
            Seconds, margin included.

        Raises:
            TypeError: If ``batches`` is not an integer or a number has the wrong type.
            ValueError: If ``batches`` or ``observed_batch`` is negative, or
                ``length_ratio < 1``.
        """
        ratio = _number("length_ratio", length_ratio, minimum=1.0)
        units = _count("batches", batches)
        decode = self.estimate(DECODE, count=units)
        if observed_batch is not None:
            seen = _number("observed_batch", observed_batch, minimum=0.0)
            decode = max(decode, (1.0 + self.margin) * units * seen)
        selection = self.estimate(SELECTION, count=units * self.candidates_per_batch)
        return ratio * (decode + selection)

    def round_costs(
        self,
        *,
        train_tokens: int,
        test_batches: int,
        gate_batches: int = 0,
        train_steps: int | None = None,
        length_ratio: float = 1.0,
        observed_batch: float | None = None,
        redecode_batches: int = 0,
    ) -> dict[str, float]:
        """Conservative seconds of each round variant, for :meth:`Governor.choose`.

        A gradient round re-decodes each test input that passed the promotion's free checks
        under fresh views (P2.2), trains on the enlarged pair set, runs the held-out gate's decode
        (P2.1), and decodes and rescores the test inputs. An in-context round re-decodes the same
        inputs, then decodes the gate and the test inputs with the extra pairs in the prompt
        (``length_ratio``) and rescores; it has no training. The re-decodes run before any pair
        is added, so their prompts keep the round-0 length in both variants. Both pay the
        per-round overhead. P2.3 runs the gate check in every round, for every adapter provider,
        so the in-context round pays the gate's decode too.

        Args:
            train_tokens: Tokens the gradient round's fit trains on (for ``sgd_continue`` on the
                enlarged set, P2.3: the tokens of its ``train_steps`` training samples), counted
                by the caller from the ``DemoSet``; 0 for no fit.
            test_batches: DFS batches to decode the test inputs (the reference: 4 per input).
            gate_batches: DFS batches of the held-out gate; 0 without a held-out pair.
            train_steps: Optimizer steps of the fit; the config's steps when None.
            length_ratio: In-context prompt tokens over the round-0 prompt tokens, at least 1.
            observed_batch: Seconds of the slowest DFS batch this puzzle ran in round 0, or
                None; see :meth:`decode_cost`.
            redecode_batches: DFS batches of the promotion's re-decodes (P2.2, P2.3): 4 per test
                input that passed the free checks; 0 by default, as before P2.3.

        Returns:
            ``{"gradient": seconds, "in_context": seconds}``, margin included.

        Raises:
            TypeError: If a count is not an integer or ``length_ratio`` not a number.
            ValueError: If a count is negative or ``length_ratio < 1``.
        """
        batches = _count("test_batches", test_batches) + _count("gate_batches", gate_batches)
        redecode = _count("redecode_batches", redecode_batches)
        overhead = (1.0 + self.margin) * self.overhead
        train = self.estimate(TRAIN, train_tokens, steps=train_steps)
        again = self.decode_cost(redecode, 1.0, observed_batch)
        return {
            GRADIENT: train + self.decode_cost(batches + redecode, 1.0, observed_batch) + overhead,
            IN_CONTEXT: again + self.decode_cost(batches, length_ratio, observed_batch) + overhead,
        }


def _section(
    sections: Mapping[str, Any], op: str, allowed: set[str], required: set[str]
) -> Mapping[str, Any]:
    """One op's section of the config: parameters only (provenance lives in ``provenance``)."""
    section = sections[op]
    if not isinstance(section, Mapping):
        raise TypeError(f"ops.{op} must be a mapping; got {section!r}")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"ops.{op}: unknown keys {unknown}")
    missing = sorted(required - set(section))
    if missing:
        raise KeyError(f"ops.{op}: missing {missing}")
    return section


def _load_yaml(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Parse a YAML mapping; PyYAML is imported here, so importing this module never needs it."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"{os.fspath(path)}: the config must be a mapping")
    return config


@dataclass(frozen=True)
class RoundPlan:
    r"""One governor decision for one puzzle, as the loop logs it.

    Attributes:
        puzzle_id: Task id, or None.
        decision: ``gradient``, ``in_context``, or ``none``.
        at: Clock time of the decision.
        budget: The puzzle's budget from :meth:`Governor.puzzle_budget`, seconds.
        deadline: Clock time by which the puzzle, its round included, should end.
        remaining: Seconds from ``at`` to ``deadline``; negative when the puzzle is late.
        estimates: Conservative seconds of each round variant offered, margin included.
        decode_batches: Baseline decode batches to run (the decode trade's cap).
        pending_batches: Baseline decode batches the reference would still run.
        decode_seconds: Conservative seconds of the ``decode_batches`` and their rescoring.
        overhead_seconds: Conservative seconds held for the baseline's own overhead after its
            decode, :math:`(1 + m)\,o`, in a plan made before a baseline batch; 0 after it.
        policy: Decode policy, ``reference`` (never cut) or ``trade`` (cut to fit a round).
        margin: The cost model's margin.
    """

    puzzle_id: str | None
    decision: str
    at: float
    budget: float
    deadline: float
    remaining: float
    estimates: Mapping[str, float]
    decode_batches: int
    pending_batches: int
    decode_seconds: float
    overhead_seconds: float
    policy: str
    margin: float

    @property
    def cut_batches(self) -> int:
        """Baseline decode batches given up for the round (the logged trade); 0 when none."""
        return self.pending_batches - self.decode_batches

    def to_record(self) -> dict[str, object]:
        """Return the plan as a JSON-ready dict, for the loop's JSONL log.

        Returns:
            Every field, plus ``cut_batches``; seconds rounded to milliseconds.
        """
        return {
            "puzzle_id": self.puzzle_id,
            "decision": self.decision,
            "at": round(self.at, 3),
            "budget": round(self.budget, 3),
            "deadline": round(self.deadline, 3),
            "remaining": round(self.remaining, 3),
            "estimates": {k: round(v, 3) for k, v in sorted(self.estimates.items())},
            "decode_batches": self.decode_batches,
            "pending_batches": self.pending_batches,
            "cut_batches": self.cut_batches,
            "decode_seconds": round(self.decode_seconds, 3),
            "overhead_seconds": round(self.overhead_seconds, 3),
            "policy": self.policy,
            "margin": self.margin,
        }


class Governor:
    r"""Per-puzzle budgets and the round decision for one worker (B5 P2.4).

    Each worker process holds its own governor; the workers share only the queue and the
    deadline. The budget of a puzzle is this worker's share of the time left before
    :math:`t_\text{end} - r`; a round runs only if its conservative cost fits what is left of
    the puzzle's budget.

    Attributes:
        cost_model: The conservative cost model.
        end_time: Global deadline on the clock's scale: the solver's end time, 12 h minus
            30 min after the notebook starts.
        workers: Workers draining the shared queue, as the solver started them; also the number
            of ``None`` sentinels the reference queues after the task ids.
        clock: Returns the current time; ``time.time`` (the scale of the reference's
            ``end_time``) unless a test injects one.
        reserve: Seconds before ``end_time`` that no round may use: the straggler reserve for
            the baseline of each worker's last puzzle.
        policy: ``reference`` keeps the reference's decode batches; ``trade`` may cap them to
            make room for a round.
        min_batches_per_input: Under ``trade``, the fewest baseline decode batches kept per
            test input.
    """

    def __init__(
        self,
        cost_model: CostModel,
        end_time: float,
        workers: int,
        clock: Callable[[], float] = time.time,
        *,
        reserve: float = 0.0,
        policy: str = REFERENCE,
        min_batches_per_input: int = 2,
    ) -> None:
        """Build a governor.

        Args:
            cost_model: The conservative cost model.
            end_time: Global deadline on the clock's scale.
            workers: Workers the solver started, at least 1.
            clock: Returns the current time.
            reserve: Straggler reserve in seconds, at least 0.
            policy: ``reference`` or ``trade``.
            min_batches_per_input: Trade floor per test input, 1 to 4.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``end_time`` or ``reserve`` is not finite, ``reserve < 0``,
                ``workers < 1``, ``policy`` is unknown, or ``min_batches_per_input`` is outside
                ``[1, 4]``.
        """
        if not isinstance(cost_model, CostModel):
            raise TypeError(f"cost_model must be a CostModel; got {cost_model!r}")
        if not callable(clock):
            raise TypeError(f"clock must be callable; got {clock!r}")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
        self.cost_model = cost_model
        self.end_time = _number("end_time", end_time)
        self.workers = _count("workers", workers)
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1; got {self.workers}")
        self.clock = clock
        self.reserve = _number("reserve", reserve, minimum=0.0)
        self.policy = policy
        self.min_batches_per_input = _count("min_batches_per_input", min_batches_per_input)
        if not 1 <= self.min_batches_per_input <= REFERENCE_BATCHES_PER_INPUT:
            raise ValueError(
                f"min_batches_per_input must be 1 to {REFERENCE_BATCHES_PER_INPUT}; "
                f"got {self.min_batches_per_input}"
            )

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike[str],
        end_time: float,
        workers: int,
        clock: Callable[[], float] = time.time,
    ) -> Governor:
        """Build a governor from ``configs/governor.yaml``.

        Args:
            path: The config file; see :meth:`from_mapping`.
            end_time: The solver's end time on the clock's scale.
            workers: Workers the solver started.
            clock: The clock.

        Returns:
            The governor.

        Raises:
            ImportError: If PyYAML is missing.
            OSError: If the file cannot be read.
            KeyError: As :meth:`from_mapping` raises it.
            TypeError: As :meth:`from_mapping` raises it.
            ValueError: As :meth:`from_mapping` raises it.
        """
        return cls.from_mapping(_load_yaml(path), end_time, workers, clock)

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        end_time: float,
        workers: int,
        clock: Callable[[], float] = time.time,
    ) -> Governor:
        """Build a governor from a parsed config, without reading a file.

        Why: ``configs/`` is not in the ``arc26`` wheel, so on Kaggle the loop gets the config's
        content some other way (for example embedded by the notebook builder) and passes it here.

        Args:
            config: The parsed config: the cost model's keys (:meth:`CostModel.from_mapping`),
                plus ``deadline.reserve_s`` and ``decode_policy`` (``mode``,
                ``min_batches_per_input``).
            end_time: The solver's end time on the clock's scale; the config's
                ``deadline.end_s`` records its offset from the notebook start.
            workers: Workers the solver started.
            clock: The clock.

        Returns:
            The governor.

        Raises:
            KeyError: If a required key is missing.
            TypeError: As the constructors raise it.
            ValueError: As the constructors raise it.
        """
        policy = config.get("decode_policy", {})
        return cls(
            cost_model=CostModel.from_mapping(config),
            end_time=end_time,
            workers=workers,
            clock=clock,
            reserve=config["deadline"].get("reserve_s", 0.0),
            policy=policy.get("mode", REFERENCE),
            min_batches_per_input=policy.get("min_batches_per_input", 2),
        )

    def puzzle_budget(self, queue_size: int) -> float:
        r"""Seconds this worker may spend on the puzzle it has just taken.

        The remaining wall time before the reserve, over the puzzles this worker still runs: the
        current one plus its share :math:`\lceil q / w \rceil` of the :math:`q` queued puzzles.
        The reference's queue (``arc26/nvarc/starter.py``) holds the task ids followed by one
        ``None`` per worker, which tells a worker to stop. The queue is first in, first out, so
        while task ids remain all :math:`w` sentinels are still queued; once the ids are gone
        the size is at most :math:`w`. Hence :math:`q = \max(0, \text{size} - w)` exactly.
        Failure mode (stragglers at the end of the queue): the share assumes the workers drain
        the queue evenly, but while one worker is stuck on a long puzzle the others take more
        than their share, on budgets sized for fewer puzzles; and a worker's last puzzle
        (:math:`q = 0`) gets all the time left, so an overrun there has no later puzzle to absorb
        it. ``docs/modules/governor.md`` lists what guards against it.

        Args:
            queue_size: ``queue.qsize()`` of the reference's queue right after this worker took
                its puzzle: the queued task ids plus the ``None`` sentinels still queued.

        Returns:
            :math:`\max(0, t_\text{end} - r - t) / (1 + \lceil q / w \rceil)`; 0 at or past
            the reserve.

        Raises:
            TypeError: If ``queue_size`` is not an integer.
            ValueError: If ``queue_size`` is negative.
        """
        queued = max(0, _count("queue_size", queue_size) - self.workers)
        share = 1 + -(-queued // self.workers)
        left = self.end_time - self.reserve - self.clock()
        return max(0.0, left) / share

    def choose(self, round_costs: Mapping[str, float], remaining: float) -> str:
        """Pick the round variant from the fallback ladder.

        The ladder is B2 Q4's order, not the cheapest variant first: a gradient round if its
        cost fits, else an in-context round if its cost fits, else none.

        Args:
            round_costs: Conservative seconds per offered variant (``gradient``,
                ``in_context``); a missing variant is not offered.
            remaining: Seconds left in the puzzle's budget; zero or negative leaves only none.

        Returns:
            ``"gradient"``, ``"in_context"``, or ``"none"``.

        Raises:
            TypeError: If a cost or ``remaining`` is not a number.
            ValueError: If a key is not a ladder variant, or a cost is negative or not finite,
                or ``remaining`` is not finite.
        """
        costs = _costs(round_costs)
        left = _number("remaining", remaining)
        for variant in LADDER:
            if variant in costs and costs[variant] <= left:
                return variant
        return NONE

    def plan(
        self,
        round_costs: Mapping[str, float],
        deadline: float,
        *,
        budget: float,
        pending_batches: int = 0,
        test_inputs: int = 1,
        done_batches: int = 0,
        observed_batch: float | None = None,
        puzzle_id: str | None = None,
    ) -> RoundPlan:
        r"""Decide the round and, under ``trade``, how many baseline decode batches still run.

        With ``pending_batches=0`` (after the baseline) the decision is :meth:`choose` on the
        time left to ``deadline``. Before a baseline decode batch, the pending batches are
        priced too: under ``reference`` all of them run and the decision is what they leave;
        under ``trade`` the ladder is walked in order and each variant may stop the baseline
        decode early, keeping at least ``min_batches_per_input`` batches per test input in all:
        the first variant that fits wins, with the most batches that let it fit. When no
        variant fits, no batch is cut. The loop calls it before each baseline batch, as the
        reference checks its 1200 s cap, and stops the baseline decode when
        ``decode_batches`` is 0. A plan made before a baseline batch also holds
        :math:`(1 + m)\,o` for the baseline's own overhead after its decode, so that a variant
        the decode was cut for still fits at the final decision when that overhead stays within
        the margin; otherwise the cut views would be lost for no round.

        Args:
            round_costs: Conservative seconds per offered variant.
            deadline: Clock time by which the puzzle should end: its start plus its budget.
            budget: The puzzle's budget, for the record.
            pending_batches: Baseline decode batches the reference would still run.
            test_inputs: Test inputs of the puzzle; with ``done_batches`` it sets the floor.
            done_batches: Baseline decode batches already run.
            observed_batch: Seconds of the slowest baseline batch run so far, or None; prices
                the pending batches as :meth:`CostModel.decode_cost` does.
            puzzle_id: Task id, for the record.

        Returns:
            The plan; log it with :meth:`RoundPlan.to_record`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range.
        """
        costs = _costs(round_costs)
        end = _number("deadline", deadline)
        pending = _count("pending_batches", pending_batches)
        inputs = _count("test_inputs", test_inputs)
        done = _count("done_batches", done_batches)
        at = self.clock()
        remaining = end - at
        if self.policy == TRADE:
            floor = min(pending, max(0, self.min_batches_per_input * inputs - done))
        else:
            floor = pending
        per_batch = self.cost_model.decode_cost(1, observed_batch=observed_batch)
        model = self.cost_model
        overhead = (1.0 + model.margin) * model.overhead if pending else 0.0
        decision, kept = NONE, pending
        for variant in LADDER:
            if variant not in costs:
                continue
            room = remaining - costs[variant] - overhead
            if room < 0 or floor * per_batch > room:
                continue
            batches = pending
            if per_batch > 0:
                # Floor division can land one off; settle on the largest count that fits.
                batches = min(pending, max(floor, int(room // per_batch)))
                while batches < pending and (batches + 1) * per_batch <= room:
                    batches += 1
                while batches > floor and batches * per_batch > room:
                    batches -= 1
            decision, kept = variant, batches
            break
        return RoundPlan(
            puzzle_id=puzzle_id,
            decision=decision,
            at=at,
            budget=_number("budget", budget),
            deadline=end,
            remaining=remaining,
            estimates=MappingProxyType(dict(costs)),
            decode_batches=kept,
            pending_batches=pending,
            decode_seconds=kept * per_batch,
            overhead_seconds=overhead,
            policy=self.policy,
            margin=self.cost_model.margin,
        )


def _costs(round_costs: Mapping[str, float]) -> dict[str, float]:
    """Validated round costs: ladder variants only, finite and non-negative."""
    if not isinstance(round_costs, Mapping):
        raise TypeError(f"round_costs must be a mapping; got {round_costs!r}")
    unknown = sorted(set(round_costs) - set(LADDER))
    if unknown:
        raise ValueError(f"round_costs keys must be in {LADDER}; got {unknown}")
    return {k: _number(f"round_costs[{k!r}]", v, minimum=0.0) for k, v in round_costs.items()}
