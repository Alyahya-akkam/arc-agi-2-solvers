r"""The refinement loop: state, held-out gate, promotion, and rounds (P2.1 to P2.5; B5, B2 Q4).

A promoted test prediction is only as good as the adapter trained with it, so the loop holds one
real demonstration pair out of everything the adapter learns from and checks, after each fit,
that the adapter still solves it. This module holds that state, that check, the decision to
promote, and the rounds around it:

- ``DemoSet`` (ADR-004): the versioned state of one puzzle, its real pairs, the held-out pair, and
  (from P2.2) promoted pseudo-pairs with provenance. ``DemoSet.pairs`` is the training and voting
  set; the held-out pair reaches no adapter and no prompt (ADR-013).
- ``held_out_index``: the deterministic hold-out rule, the last real pair by index by default. It
  applies for :math:`k \ge 3` real pairs, and for :math:`k = 2` behind the flag ``k2`` (default on,
  ablated in P2.7).
- ``HeldOutGate``: B4's ``Gate``. It takes the fit and decode callables (injected, so the tests run
  on CPU with mocks): ``fit`` trains on ``pairs`` only, and ``passes`` decodes the held-out input
  under :math:`n` augmented views and requires the voted answer, the reference's selection
  (``score_kgmon``, attempt 1 of the submission), to equal the held-out output exactly.
- ``promote`` (P2.2, ADR-014): a test input's voted answer becomes a pseudo-pair when its
  agreement (the views that produced it, the count term of the reference's ``score_kgmon``) is at
  least :math:`\tau`, it survives a re-decode under fresh views that share no view with the
  decode that produced it, and the candidate adapter fit with it passes the gate.
- ``refine`` (P2.3, ADR-015): B4's one round with rollback. The governor prices the round and
  may decline it; ``promote`` runs with the provider's ``adapt`` as the candidate fit (Phase 2:
  ``arc26.ttt.SGDProvider``, ``sgd_continue`` from a snapshot); a failing gate rolls the
  provider back to the previous adapter, and the state keeps its version. ``run_rounds`` chains
  up to ``max_rounds`` of them (hard cap 2) and decodes the open test inputs after them: by
  default once more with the held-out pair back in the prompts, the decode that is submitted
  (``Rounds.final_in_prompts``, ADR-013 as amended), which each round's plan prices.
- The in-context round (P2.5, ADR-016, B2 Q4's fallback): ``InContextProvider`` returns the
  current adapter unchanged, so a round with it promotes into the prompts only: the gate decodes
  with the enlarged set in its prompts, and a failed gate rolls back the version alone.
  ``refine`` offers it beside the gradient round when given the provider (``Rounds.variants``
  forces either), and the governor picks it when the gradient round does not fit.
- The length guard (P2.5, ADR-016): ``context_fit``, a pure function of prompt lengths, keeps a
  prompt's pseudo-pairs only as far as the prompt fits the reference's limit, latest dropped
  first, so the reference's ``cut_to_len`` never drops a real pair for a pseudo-pair;
  ``PromptGuard`` applies it to every decode and rescoring prompt that carries pseudo-pairs.
- ``solve_puzzle``: the split solver's per-puzzle body with a held-out pair (``HoldOut``), wiring
  ``arc26.ttt.fit`` and ``arc26.decode.decode_puzzle`` (ADR-011) into the gate and, with
  ``HoldOut.rounds``, into the rounds. Its GPU imports happen inside it, so importing this module
  needs only the standard library, ``arc26.timing``, and ``arc26.governor``; ``vote`` and
  ``tally`` import numpy through the moved selection cell, and the view checks import the moved
  ``arc_loader``.

Grids here are B4's ``Grid`` in immutable form, a tuple of row tuples, so pairs hash and compare
by value. The reference has no leave-one-out task generation (ADR-009, ADR-011): its fit trains
on the puzzle's pairs under 128 augmented views, so excluding the held-out pair from the pairs
handed to ``fit`` excludes it from every training path at test time (P3's episodes are offline
meta-training, not test time). See ``docs/modules/loop.md``.
"""

from __future__ import annotations

import json
import math
import operator
import os
import shutil
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from arc26 import timing
from arc26.governor import GRADIENT, IN_CONTEXT, LADDER, REFERENCE, Governor, RoundPlan

if TYPE_CHECKING:
    from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter

Grid = tuple[tuple[int, ...], ...]

REAL = "real"
HELD_OUT = "held_out"
PSEUDO = "pseudo"
KINDS = (REAL, HELD_OUT, PSEUDO)
# The ARC format: sides 1 to 30, colors 0 to 9.
MAX_SIDE = 30
COLORS = 10
# The views decode_puzzle decodes one input under: augment(n=2, seed=2), the 8 dihedral transforms
# with 2 color permutations each, in 4 DFS batches of 4 views (arc26/decode.py).
DECODE_VIEWS = 16
# B5 P2.1: hold-out applies from 3 real pairs; at exactly 2 it is the k2 flag.
MIN_PAIRS = 3
DEFAULT_RULE = "last"
# Op names of the gate's timing records (arc26.timing): its DFS batches and rescorings, which
# decode_puzzle records as decode and selection, and its whole span. The governor prices the
# gate's batches apart from the test inputs' (P2.4, CostModel.round_costs's gate_batches).
GATE = "gate"
GATE_DECODE = "gate_decode"
GATE_SELECTION = "gate_selection"
GATE_OPS: Mapping[str, str] = MappingProxyType(
    {timing.DECODE: GATE_DECODE, timing.SELECTION: GATE_SELECTION}
)
VERDICT_PREFIX = "gate-"
# Op names of the promotion's GPU work (P2.2): the re-decode's DFS batches and rescorings, which
# decode_puzzle records as decode and selection, its span per test input, and the candidate
# adapter's fit, whose trainer records train. The wiring renames them with RenamedOps, so the
# timing report and the governor (P2.4) price them apart from the test inputs' own ops.
REDECODE = "redecode"
REDECODE_DECODE = "redecode_decode"
REDECODE_SELECTION = "redecode_selection"
REDECODE_OPS: Mapping[str, str] = MappingProxyType(
    {timing.DECODE: REDECODE_DECODE, timing.SELECTION: REDECODE_SELECTION}
)
CANDIDATE_TRAIN = "candidate_train"
CANDIDATE_OPS: Mapping[str, str] = MappingProxyType({timing.TRAIN: CANDIDATE_TRAIN})
PROMOTION_PREFIX = "promote-"
# ADR-014: tau's starting value. The reference has no agreement threshold; on run P1.1-01's
# candidates the isotonic estimate of P(voted answer correct | views that produced it) first
# reaches 1/2 at 9 of the 16 views, a strict majority.
TAU = 9 / 16
# The first view seed a re-decode tries: the reference seeds its fit's views with 1 and its
# decode's with 2 (arc26/ttt.py, arc26/decode.py).
FRESH_SEED = 3
# Seeds tried, from the first, for a view set that shares no view with the first decode. A
# two-color prompt collides at about one seed in two; a prompt that no view set avoids (every
# grid one color, say) gives up here.
VIEW_SEED_TRIES = 64
# numpy.random.seed takes seeds below 2**32.
SEED_LIMIT = 2**32
# Why a test input was or was not promoted, in the order promote checks them (ADR-014).
PROMOTED = "promoted"
NO_CANDIDATES = "no_candidates"
TIED = "tied"
BELOW_TAU = "agreement_below_tau"
ALREADY_PROMOTED = "already_promoted"
NO_HELD_OUT = "no_held_out"
VIEWS_NOT_DISJOINT = "views_not_disjoint"
REDECODE_RAISED = "redecode_raised"
# P2.3: a re-decode stopped by decode_puzzle's 1200 s cap or the run's end time before all its
# DFS batches ran votes on fewer views than the check needs, so it fails (ADR-015).
REDECODE_CUT = "redecode_cut"
REDECODE_DISAGREES = "redecode_disagrees"
FIT_RAISED = "fit_raised"
GATE_RAISED = "gate_raised"
GATE_FAILED = "gate_failed"
# P2.5: the in-context round's prompts cannot hold the input's pseudo-pair: the gate's prompt with
# it would pass the reference's limit, so cut_to_len could drop a real pair for it (the length
# guard, ADR-016). Not re-decoded; when no input's pair fits, the round is not offered (the
# round's outcome has the same name).
TOO_LONG = "too_long"
# P2.3: the governor declined the round after the free checks, before any re-decode; refine
# records it for the inputs that passed them (the round's outcome has the same name), so that
# every voted answer's agreement is logged (ADR-014, P2.8).
DECLINED = "declined"
REASONS = (
    PROMOTED, NO_CANDIDATES, TIED, BELOW_TAU, ALREADY_PROMOTED, NO_HELD_OUT, VIEWS_NOT_DISJOINT,
    TOO_LONG, DECLINED, REDECODE_RAISED, REDECODE_CUT, REDECODE_DISAGREES, FIT_RAISED,
    GATE_RAISED, GATE_FAILED,
)  # fmt: skip
# Task P2.3, the rounds around promote (ADR-015). B5 P2.3: "Maximum rounds is a parameter,
# default 1, hard cap 2."
DEFAULT_ROUNDS = 1
MAX_ROUNDS = 2
# The training views of one fit: augment(n=16) over the 8 dihedral transforms, one epoch of 128
# steps at batch 1 (arc26/ttt.py).
EPOCH_VIEWS = 128
# s', the optimizer steps of a round's sgd_continue (ADR-004: a fraction of an epoch): a quarter
# of the epoch. At P2.1-01's DEV40 sizes (121.9 s per 128-step fit with a pair held out) the
# governor's attention model gives about 61 s for it on the enlarged set, against about 170 s for
# a full epoch (ADR-015); P2.7 may vary it.
CONTINUE_STEPS = 32
# decode_puzzle decodes 4 views per DFS batch, so an input's 16 views are 4 batches.
VIEWS_PER_BATCH = 4
# Op names of the round's GPU work (P2.3), recorded apart from round 0's: the span of a round
# that ran its promotion; the test decode after a kept round, and the final decode with the
# held-out pair in the prompts (final_in_prompts), whose DFS batches and rescorings
# decode_puzzle records as decode and selection; and the provider's copies of the adapter.
ROUND = "round"
ROUND_DECODE = "round_decode"
ROUND_SELECTION = "round_selection"
ROUND_OPS: Mapping[str, str] = MappingProxyType(
    {timing.DECODE: ROUND_DECODE, timing.SELECTION: ROUND_SELECTION}
)
FINAL_DECODE = "final_decode"
FINAL_SELECTION = "final_selection"
FINAL_OPS: Mapping[str, str] = MappingProxyType(
    {timing.DECODE: FINAL_DECODE, timing.SELECTION: FINAL_SELECTION}
)
SNAPSHOT = "snapshot"
RESTORE = "restore"
FINGERPRINT = "fingerprint"
# The round's JSONL records, beside the gate's verdicts (P2.3): the governor's plans, the rounds,
# and the test decodes after them; the promotions keep PROMOTION_PREFIX. P2.5 adds one line per
# decode whose prompts left out pseudo-pairs (the length guard), written only when one did.
PLAN_PREFIX = "plan-"
ROUND_PREFIX = "round-"
DECODE_PREFIX = "decode-"
CONTEXT_PREFIX = "context-"
# How a round ended (RoundRecord.outcome): kept (version + 1); rolled back (the candidate fit ran
# and the gate failed or a check raised after it); not promoted (no fit reached: every re-decode
# disagreed, was cut, or raised); declined by the governor; nothing to promote (no test input
# passed the free checks, so no plan was made); no held-out pair (no gate, so no promotion);
# too long (P2.5: the in-context round alone was offered and none of its pseudo-pairs fits the
# gate's prompts, so no plan was made); or raised (an error outside promote's checks; rolled back).
KEPT = "kept"
ROLLED_BACK = "rolled_back"
NOT_PROMOTED = "not_promoted"
NOTHING_TO_PROMOTE = "nothing_to_promote"
RAISED = "raised"
OUTCOMES = (
    KEPT, ROLLED_BACK, NOT_PROMOTED, DECLINED, NOTHING_TO_PROMOTE, NO_HELD_OUT, TOO_LONG, RAISED
)  # fmt: skip
# The round variants a run offers (Rounds.variants, P2.5): today's runs offer the gradient round
# alone; P2.7 and run P2.5-01 force the in-context one, and production may offer both, in the
# governor's order (LADDER: the in-context round is the fallback when the gradient one does not
# fit, B2 Q4).
DEFAULT_VARIANTS = (GRADIENT,)
# Why a test decode after the rounds did not run (DecodeRecord.skipped): every test input is a
# pseudo-pair, whose candidates are the ones that promoted it; the last round's rollback did not
# restore the weights, so no adapter of known state is left for the final decode; or the
# governor's time left cannot cover the final decode.
ALL_PROMOTED = "all_promoted"
NOT_RESTORED = "not_restored"
NO_TIME = "no_time"
# The prompts of a test decode after a round (DecodeRecord.prompts): DemoSet.pairs only, or with
# the held-out pair put back (final_in_prompts).
PAIRS_PROMPTS = "pairs"
HELD_OUT_PROMPTS = "held_out"


def as_grid(value: Any) -> Grid:
    """Return ``value`` as an immutable ARC grid, checking its shape and colors.

    Why: pairs and candidates are hashed and compared by their grids, and a grid is the only
    signal the loop passes around (B4). A list of lists, a tuple of tuples, or a 2-D numpy array
    of integers converts to the same tuple of row tuples of plain ``int``.

    Args:
        value: Rows of integer colors.  # [height, width]

    Returns:
        ``height`` rows, each a tuple of ``width`` ints from 0 to 9.

    Raises:
        TypeError: If ``value`` is not a sequence of sequences of integers; strings and booleans
            are refused.
        ValueError: If the rows differ in length, a side is outside 1 to 30, or a color is outside
            0 to 9.
    """
    if isinstance(value, str | bytes):
        raise TypeError(f"a grid is rows of integer colors; got {value!r}")
    try:
        rows = [row for row in value]
    except TypeError:
        raise TypeError(f"a grid is rows of integer colors; got {value!r}") from None
    if not 1 <= len(rows) <= MAX_SIDE:
        raise ValueError(f"a grid has 1 to {MAX_SIDE} rows; got {len(rows)}")
    grid = []
    for row in rows:
        if isinstance(row, str | bytes):
            raise TypeError(f"a grid row is a sequence of integer colors; got {row!r}")
        try:
            cells = list(row)
        except TypeError:
            raise TypeError(f"a grid row is a sequence of integer colors; got {row!r}") from None
        colors = []
        for cell in cells:
            if isinstance(cell, bool | str | bytes):
                raise TypeError(f"a color is an integer; got {cell!r}")
            try:
                color = operator.index(cell)
            except TypeError:
                raise TypeError(f"a color is an integer; got {cell!r}") from None
            if not 0 <= color < COLORS:
                raise ValueError(f"a color is 0 to {COLORS - 1}; got {color}")
            colors.append(color)
        grid.append(tuple(colors))
    width = len(grid[0])
    if not 1 <= width <= MAX_SIDE:
        raise ValueError(f"a grid has 1 to {MAX_SIDE} columns; got {width}")
    if any(len(row) != width for row in grid):
        raise ValueError(f"a grid is rectangular; got rows of {sorted({len(r) for r in grid})}")
    return tuple(grid)


def _grid_lists(grid: Grid) -> list[list[int]]:
    # The challenges file's form, which the reference's ArcDataset and formatter read.
    return [list(row) for row in grid]


@dataclass(frozen=True)
class Pair:
    """One demonstration pair of a puzzle, with its provenance (B4 schema).

    Attributes:
        input: The input grid.
        output: The output grid.
        kind: ``"real"``, a demonstration of the puzzle; ``"held_out"``, the real pair the gate
            checks; ``"pseudo"``, a promoted test prediction (P2.2).
        round_added: The refinement round that added the pair: 0 for real and held-out pairs, at
            least 1 for a pseudo-pair.
        agreement: The augmentation agreement of a pseudo-pair's output (P2.2, ``Tally``); None
            for real and held-out pairs.
        test_index: Which test input of the puzzle a pseudo-pair's input is (P2.2); None for
            real and held-out pairs.
        verdicts: The checks that promoted a pseudo-pair (P2.2): its view check, re-decode, and
            gate verdicts; None for real and held-out pairs.

    Raises:
        TypeError: If a grid is malformed (``as_grid``), ``round_added`` or ``test_index`` is
            not an integer, ``agreement`` is not None or a number, or ``verdicts`` is not None
            or ``Verdicts``.
        ValueError: If a grid is malformed, ``kind`` is unknown, a real or held-out pair has a
            round, an agreement, a test index, or verdicts, or a pseudo-pair's round is under 1,
            its agreement not finite, or its test index negative.
    """

    input: Grid
    output: Grid
    kind: str = REAL
    round_added: int = 0
    agreement: float | None = None
    test_index: int | None = None
    verdicts: Verdicts | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", as_grid(self.input))
        object.__setattr__(self, "output", as_grid(self.output))
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}; got {self.kind!r}")
        if type(self.round_added) is not int:
            raise TypeError(f"round_added must be an integer; got {self.round_added!r}")
        if self.agreement is not None and (
            isinstance(self.agreement, bool) or not isinstance(self.agreement, int | float)
        ):
            raise TypeError(f"agreement must be None or a number; got {self.agreement!r}")
        if self.test_index is not None and type(self.test_index) is not int:
            raise TypeError(f"test_index must be None or an integer; got {self.test_index!r}")
        if self.verdicts is not None and not isinstance(self.verdicts, Verdicts):
            raise TypeError(f"verdicts must be None or Verdicts; got {self.verdicts!r}")
        if self.kind == PSEUDO:
            if self.round_added < 1:
                raise ValueError(
                    f"a pseudo-pair is added in round 1 or later; got {self.round_added}"
                )
            if self.agreement is not None and not math.isfinite(self.agreement):
                raise ValueError(f"agreement must be finite; got {self.agreement!r}")
            if self.test_index is not None and self.test_index < 0:
                raise ValueError(f"test_index must be at least 0; got {self.test_index}")
        elif (
            self.round_added != 0
            or self.agreement is not None
            or self.test_index is not None
            or self.verdicts is not None
        ):
            raise ValueError(
                f"a {self.kind} pair comes with the puzzle: round 0, no agreement, test index, "
                f"or verdicts; got round {self.round_added}, agreement {self.agreement!r}, "
                f"test index {self.test_index!r}"
            )

    def task_pair(self) -> dict[str, list[list[int]]]:
        """Return the pair as the challenges file holds it, ``{"input": ..., "output": ...}``.

        Returns:
            Both grids as lists of lists of ints, the form the reference's ``ArcDataset`` reads.
        """
        return {"input": _grid_lists(self.input), "output": _grid_lists(self.output)}


Rule = Callable[[Sequence[Pair]], int]


def _last(pairs: Sequence[Pair]) -> int:
    return len(pairs) - 1


def _first(pairs: Sequence[Pair]) -> int:
    return 0


# Hold-out rules by name, for configs (B5 P2.1: "last real pair by index; rule is a parameter").
# Python callers may pass any callable that maps the real pairs to an index.
HOLDOUT_RULES: Mapping[str, Rule] = MappingProxyType({"last": _last, "first": _first})


def held_out_index(
    pairs: Sequence[Pair], rule: str | Rule = DEFAULT_RULE, k2: bool = True
) -> int | None:
    r"""Return the index of the real pair to hold out, or None when the puzzle keeps them all.

    Why: B5 P2.1's deterministic rule. Holding out one of :math:`k` pairs removes :math:`1/k` of
    the supervision, so it applies for :math:`k \ge 3`; for :math:`k = 2` it would leave one pair
    to train on, and the flag ``k2`` decides (default on, ablated in P2.7). With fewer than 2
    pairs nothing is held out, since training needs a pair.

    Args:
        pairs: The puzzle's real demonstration pairs, in the challenges file's order.
        rule: A name in ``HOLDOUT_RULES`` (``"last"``, the default, or ``"first"``), or a callable
            that maps ``pairs`` to an index.
        k2: Whether to hold out a pair when the puzzle has exactly 2.

    Returns:
        An index into ``pairs``, or None.

    Raises:
        TypeError: If ``k2`` is not a bool, ``rule`` is neither a name nor a callable, or the
            rule returns a non-integer.
        ValueError: If ``rule`` names no rule, or the rule returns an index outside ``pairs``.
    """
    if not isinstance(k2, bool):
        raise TypeError(f"k2 must be true or false; got {k2!r}")
    if isinstance(rule, str):
        if rule not in HOLDOUT_RULES:
            raise ValueError(f"rule must be one of {sorted(HOLDOUT_RULES)}; got {rule!r}")
        choose = HOLDOUT_RULES[rule]
    elif callable(rule):
        choose = rule
    else:
        raise TypeError(f"rule must be a rule name or a callable; got {rule!r}")
    count = len(pairs)
    if not (count >= MIN_PAIRS or (count == 2 and k2)):
        return None
    index = choose(pairs)
    if type(index) is not int:
        raise TypeError(f"the hold-out rule must return an integer index; got {index!r}")
    if not 0 <= index < count:
        raise ValueError(f"the hold-out rule chose index {index} of {count} pairs")
    return index


@dataclass(frozen=True)
class DemoSet:
    """The refinement loop's versioned state for one puzzle (B2 Q4, ADR-004, B4 schema).

    Why: the loop's state is data plus an adapter (Q4), and rollback is a pointer swap to the
    previous version, so a ``DemoSet`` is immutable: a promotion (P2.2) builds a new one with
    ``version + 1``. Every training set and every prompt is built from ``pairs`` only
    (``task``), so the held-out pair can reach neither (ADR-013).

    Attributes:
        puzzle_id: The task id.
        pairs: The training and voting set: the real pairs less the held-out one, in the
            challenges file's order, then promoted pseudo-pairs (P2.2).
        held_out: The real pair the gate checks, never trained on and never in a prompt; None
            when the puzzle keeps every pair.
        test_inputs: The puzzle's test inputs, in order.
        version: 0 for the puzzle as loaded, incremented on every promotion.
        held_out_index: Where the held-out pair stood among the puzzle's real pairs; None
            without one.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``puzzle_id`` is empty, ``pairs`` holds a held-out pair, ``held_out`` is
            not one, ``held_out`` and ``held_out_index`` disagree, or ``version`` is negative.
    """

    puzzle_id: str
    pairs: tuple[Pair, ...]
    held_out: Pair | None
    test_inputs: tuple[Grid, ...]
    version: int = 0
    held_out_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.puzzle_id, str) or not self.puzzle_id:
            raise ValueError(f"puzzle_id must be a non-empty string; got {self.puzzle_id!r}")
        if not isinstance(self.pairs, list | tuple) or not all(
            isinstance(pair, Pair) for pair in self.pairs
        ):
            raise TypeError(f"pairs must be a sequence of Pair; got {self.pairs!r}")
        object.__setattr__(self, "pairs", tuple(self.pairs))
        if any(pair.kind == HELD_OUT for pair in self.pairs):
            raise ValueError("pairs must not hold the held-out pair: it is never trained on")
        if self.held_out is not None:
            if not isinstance(self.held_out, Pair) or self.held_out.kind != HELD_OUT:
                raise ValueError(f"held_out must be a held-out Pair; got {self.held_out!r}")
        if isinstance(self.test_inputs, str | bytes) or not isinstance(
            self.test_inputs, list | tuple
        ):
            raise TypeError(f"test_inputs must be a sequence of grids; got {self.test_inputs!r}")
        object.__setattr__(self, "test_inputs", tuple(as_grid(g) for g in self.test_inputs))
        if type(self.version) is not int or self.version < 0:
            raise ValueError(f"version must be an integer >= 0; got {self.version!r}")
        index = self.held_out_index
        if (index is None) != (self.held_out is None):
            raise ValueError("held_out_index must be set exactly when a pair is held out")
        if index is not None and (type(index) is not int or not 0 <= index < self.real_pairs):
            raise ValueError(f"held_out_index must index the {self.real_pairs} real pairs")

    @property
    def real_pairs(self) -> int:
        """The puzzle's real demonstration pairs, :math:`k`, the held-out one included."""
        return sum(pair.kind == REAL for pair in self.pairs) + (self.held_out is not None)

    @classmethod
    def from_task(
        cls,
        puzzle_id: str,
        task: Mapping[str, Any],
        *,
        rule: str | Rule = DEFAULT_RULE,
        k2: bool = True,
    ) -> DemoSet:
        """Build a puzzle's ``DemoSet`` from its challenges entry, holding out one real pair.

        Args:
            puzzle_id: The task id.
            task: The challenges file's entry: ``{"train": [{"input", "output"}, ...],
                "test": [{"input"}, ...]}``, as the reference's ``ArcDataset.queries`` holds it.
            rule: The hold-out rule (``held_out_index``).
            k2: Whether to hold out a pair when the puzzle has exactly 2.

        Returns:
            Version 0: every real pair but the held-out one in ``pairs``, in order.

        Raises:
            KeyError: If ``task`` lacks ``train`` or ``test``, or an entry lacks a grid.
            TypeError: As ``as_grid`` and ``held_out_index`` raise it.
            ValueError: As ``as_grid`` and ``held_out_index`` raise it.
        """
        real = [Pair(item["input"], item["output"]) for item in task["train"]]
        tests = tuple(as_grid(item["input"]) for item in task["test"])
        index = held_out_index(real, rule, k2)
        if index is None:
            return cls(puzzle_id, tuple(real), None, tests)
        held = replace(real[index], kind=HELD_OUT)
        kept = tuple(pair for position, pair in enumerate(real) if position != index)
        return cls(puzzle_id, kept, held, tests, held_out_index=index)

    def without_held_out(self) -> DemoSet:
        """Return this state without its held-out pair, as fit and decode callables receive it.

        Returns:
            The same pairs, test inputs, and version, with ``held_out`` None.
        """
        return replace(self, held_out=None, held_out_index=None)

    def enlarged(self, pairs: Iterable[Pair]) -> DemoSet:
        """Return the next version: this state with promoted pseudo-pairs appended (P2.2).

        Why: B4 increments the version on every promotion, and the state is immutable, so the
        previous version stays as it was for a rollback (ADR-004).

        Args:
            pairs: The pseudo-pairs to add, in order.

        Returns:
            ``pairs`` appended to this state's pairs, ``version + 1``; the held-out pair and the
            test inputs unchanged.

        Raises:
            ValueError: If ``pairs`` is empty or holds a pair that is not a pseudo-pair.
        """
        added = tuple(pairs)
        if not added:
            raise ValueError("a promotion adds at least one pseudo-pair")
        if not all(isinstance(pair, Pair) and pair.kind == PSEUDO for pair in added):
            raise ValueError("only pseudo-pairs are promoted into a DemoSet")
        return replace(self, pairs=self.pairs + added, version=self.version + 1)

    @property
    def pseudo_pairs(self) -> tuple[Pair, ...]:
        """The promoted pseudo-pairs, in the order they were added (P2.2)."""
        return tuple(pair for pair in self.pairs if pair.kind == PSEUDO)

    def first_pseudo(self, count: int) -> DemoSet:
        """Return this state as one decode's prompts hold it: its first ``count`` pseudo-pairs only.

        Why: the length guard (task P2.5, ADR-016) drops the pseudo-pairs a prompt cannot hold,
        latest first, and never a real pair. The result is what a decode callable receives in
        place of the state; the state itself does not change, so the version is kept, and the
        held-out pair stays for the gate's comparison.

        Args:
            count: Pseudo-pairs to keep, from the first added, 0 to ``len(pseudo_pairs)``.

        Returns:
            The same state with every real pair, in place, and the first ``count`` pseudo-pairs;
            ``self`` when it keeps them all.

        Raises:
            TypeError: If ``count`` is not an integer.
            ValueError: If ``count`` is negative or above the pseudo-pairs held.
        """
        held = len(self.pseudo_pairs)
        if type(count) is not int:
            raise TypeError(f"count must be an integer; got {count!r}")
        if not 0 <= count <= held:
            raise ValueError(f"count must be 0 to the {held} pseudo-pairs held; got {count}")
        if count == held:
            return self
        kept, seen = [], 0
        for pair in self.pairs:
            if pair.kind == PSEUDO:
                seen += 1
                if seen > count:
                    continue
            kept.append(pair)
        return replace(self, pairs=tuple(kept))

    def task(
        self, test_inputs: Iterable[Any] | None = None, *, held_out: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Return the reference's task entry built from ``pairs`` and the given test inputs.

        Why: every dataset the solver trains or decodes on is built here, from the training and
        voting set only, so the held-out pair reaches no adapter and no prompt (ADR-013). The
        reference formats ``train`` into the training samples and the prompts, and needs at
        least one test input to format a query. ``held_out`` puts the pair back for the one
        decode allowed to see it, the final decode after the rounds, whose candidates are
        submitted and feed no promotion (P2.3, ``Rounds.final_in_prompts``, ADR-013 as amended
        after ``P2.1-02``).

        Args:
            test_inputs: The inputs to query, as grids; None queries the puzzle's own.
            held_out: Whether the held-out pair joins the demonstrations, at its place among the
                real pairs (the challenges file's order), before the pseudo-pairs.

        Returns:
            ``{"train": [pair.task_pair() for pair in pairs], "test": [{"input": grid}, ...]}``.

        Raises:
            ValueError: If there is no pair or no test input, or ``held_out`` is asked of a
                state without a held-out pair.
        """
        grids = self.test_inputs if test_inputs is None else tuple(map(as_grid, test_inputs))
        if not self.pairs:
            raise ValueError(f"{self.puzzle_id}: no pair to train on or to prompt with")
        if not grids:
            raise ValueError(f"{self.puzzle_id}: no test input to query")
        pairs = self.pairs
        if held_out:
            if self.held_out is None or self.held_out_index is None:
                raise ValueError(f"{self.puzzle_id}: no held-out pair to put back in the prompts")
            real = [pair for pair in self.pairs if pair.kind == REAL]
            pseudo = [pair for pair in self.pairs if pair.kind == PSEUDO]
            index = self.held_out_index
            pairs = (*real[:index], self.held_out, *real[index:], *pseudo)
        return {
            "train": [pair.task_pair() for pair in pairs],
            "test": [{"input": _grid_lists(grid)} for grid in grids],
        }


@dataclass(frozen=True)
class Candidate:
    """One decoded answer of one augmented view, mapped back to the puzzle's orientation.

    Attributes:
        grid: The answer grid.
        beam_score: The DFS beam's negative log-likelihood.
        score_aug: Its negative log-likelihood under each rescoring view (8 in the reference).
        view: The view's key, for the record.
    """

    grid: Grid
    beam_score: float
    score_aug: tuple[float, ...]
    view: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", as_grid(self.grid))
        object.__setattr__(self, "beam_score", float(self.beam_score))
        object.__setattr__(self, "score_aug", tuple(float(s) for s in self.score_aug))
        if not self.score_aug:
            raise ValueError("score_aug needs at least one rescoring view")


Select = Callable[[Mapping[str, Mapping[str, Any]]], list[Any]]


def _guesses(candidates: Sequence[Candidate]) -> dict[str, dict[str, Any]]:
    # The selection's input, as ArcDecoder builds it from the candidate files: one entry per view
    # candidate, under a key unique to it, in the candidates' order.
    return {
        f"{index}.{candidate.view}": {
            "solution": candidate.grid,
            "beam_score": candidate.beam_score,
            "score_aug": list(candidate.score_aug),
        }
        for index, candidate in enumerate(candidates)
    }


def vote(candidates: Sequence[Candidate], select: Select | None = None) -> list[Grid]:
    """Rank the distinct grids among ``candidates`` with the reference's selection.

    Why: the gate's "voted answer" must be what the pipeline would submit for that input, so it
    is attempt 1 of the reference's submission selection, ``score_kgmon`` (cell 3, moved as
    ``arc26.nvarc.arc_decoder``), applied to the candidates of one input as ``ArcDecoder``
    applies it to one test output: views that agree raise a grid, the mean rescoring NLL breaks
    the tie. Ties keep the order of ``candidates``.

    Args:
        candidates: The view candidates of one input.
        select: A selection algorithm of ``arc_decoder`` (``score_kgmon`` when None), called with
            ``{name: {"solution", "beam_score", "score_aug"}}`` as ``ArcDecoder`` calls it.

    Returns:
        The distinct grids, best first; empty without candidates.
    """
    if not candidates:
        return []
    if select is None:
        # The moved selection cell imports numpy; imported here so this module needs only the
        # standard library.
        from arc26.nvarc.arc_decoder import score_kgmon as select
    return [as_grid(grid) for grid in select(_guesses(candidates))]


@dataclass(frozen=True)
class Tally:
    r"""The reference's vote over one input's view candidates, with the terms of its score (P2.2).

    For a distinct grid :math:`g` with view candidates :math:`C_g`, ``score_kgmon`` (cell 3,
    ``arc26/nvarc/arc_decoder.py`` lines 37-43) scores
    :math:`|C_g| - \frac{1}{|C_g|} \sum_{c \in C_g} \bar s(c)`, with :math:`\bar s(c)` the mean
    of candidate :math:`c`'s rescoring NLLs, and ranks the grids by it (``score_sum``, lines
    17-27; ties keep the candidates' order). Its first term is the augmentation agreement: how
    many of the :math:`n` views produced :math:`g`.

    Attributes:
        grids: The distinct grids, best first: ``score_kgmon``'s order, ``vote``'s.
        counts: View candidates per grid, :math:`|C_g|` (``getter_kgmon``'s ``len(guesses)``).
        scores: ``getter_kgmon``'s value per grid.
        views: :math:`n`, the views the input was decoded under.
    """

    grids: tuple[Grid, ...]
    counts: tuple[int, ...]
    scores: tuple[float, ...]
    views: int

    @property
    def voted(self) -> Grid | None:
        """The voted answer, the reference's attempt 1; None without candidates."""
        return self.grids[0] if self.grids else None

    @property
    def agreement(self) -> float | None:
        r"""The voted answer's agreement, :math:`|C_{\hat y}| / n`; None without candidates."""
        return self.counts[0] / self.views if self.counts else None

    @property
    def tied(self) -> bool:
        """Whether the top two grids tie in the vote, so that the candidates' order picked one."""
        return len(self.scores) > 1 and self.scores[0] == self.scores[1]


def tally(candidates: Sequence[Candidate], views: int = DECODE_VIEWS) -> Tally:
    """Vote over one input's view candidates as the reference does, keeping each grid's terms.

    Why: the promotion's agreement must be the reference's own vote, not a new score (B5 P2.2).
    The moved ``score_sum`` groups the candidates by grid and ranks the groups by
    ``getter_kgmon``, which is recorded as it runs, so the order and the scores are the
    reference's exactly.

    Args:
        candidates: The view candidates of one input.
        views: :math:`n`, the views the input was decoded under (``decode_puzzle``: 16).

    Returns:
        The tally; empty without candidates.

    Raises:
        TypeError: If ``views`` is not an integer.
        ValueError: If ``views`` is under 1.
    """
    if type(views) is not int:
        raise TypeError(f"views must be an integer; got {views!r}")
    if views < 1:
        raise ValueError(f"views must be at least 1; got {views}")
    if not candidates:
        return Tally((), (), (), views)
    from arc26.nvarc.arc_decoder import getter_kgmon, hashable, score_sum

    terms: dict[Any, tuple[int, float]] = {}

    def recorded(group: list[Mapping[str, Any]]) -> Any:
        score = getter_kgmon(group)
        terms[hashable(group[0]["solution"])] = (len(group), float(score))
        return score

    ranked = [as_grid(grid) for grid in score_sum(_guesses(candidates), recorded)]
    return Tally(
        grids=tuple(ranked),
        counts=tuple(terms[grid][0] for grid in ranked),
        scores=tuple(terms[grid][1] for grid in ranked),
        views=views,
    )


@dataclass(frozen=True)
class GateVerdict:
    """The result of one gate check, as the solver logs it (one JSON line per puzzle).

    Attributes:
        puzzle_id: The task id.
        applied: Whether a pair was held out; False when the puzzle keeps them all (no decode).
        passed: Whether the voted answer equals the held-out output exactly; None when not
            applied.
        real_pairs: The puzzle's real pairs, :math:`k`.
        held_out_index: Which real pair was held out; None when not applied.
        views: Augmented views the held-out input was decoded under, :math:`n`.
        candidates: View candidates decoded (every parsed beam of every view).
        votes: For each distinct grid, best first, the view candidates that produced it.
        expected_rank: Place of the held-out output in the vote, 1 for the voted answer; None
            when no candidate equals it.
        seconds: Wall time of the check: decode, read back, and vote.
        error: Why the check raised, ``"<type>: <message>"``, with ``passed`` None; None when it
            ran to its verdict.
    """

    puzzle_id: str
    applied: bool
    passed: bool | None
    real_pairs: int
    held_out_index: int | None = None
    views: int = 0
    candidates: int = 0
    votes: tuple[int, ...] = field(default_factory=tuple)
    expected_rank: int | None = None
    seconds: float = 0.0
    error: str | None = None

    @classmethod
    def not_applied(cls, demos: DemoSet) -> GateVerdict:
        """Return the verdict of a puzzle that keeps every pair: no gate ran.

        Args:
            demos: The puzzle's state, without a held-out pair.

        Returns:
            ``applied`` False, ``passed`` None.
        """
        return cls(demos.puzzle_id, applied=False, passed=None, real_pairs=demos.real_pairs)

    @classmethod
    def raised(cls, demos: DemoSet, error: BaseException, seconds: float) -> GateVerdict:
        """Return the verdict of a check that raised before its vote: neither passed nor failed.

        Args:
            demos: The puzzle's state, with its held-out pair.
            error: What the check raised.
            seconds: Wall time until it raised.

        Returns:
            ``applied`` True, ``passed`` None, ``error`` set.
        """
        return cls(
            demos.puzzle_id,
            applied=True,
            passed=None,
            real_pairs=demos.real_pairs,
            held_out_index=demos.held_out_index,
            seconds=seconds,
            error=f"{type(error).__name__}: {error}",
        )

    def to_record(self) -> dict[str, Any]:
        """Return the verdict as a JSON-ready dict, for the gate's JSONL log.

        Returns:
            Every field; seconds rounded to milliseconds.
        """
        return {
            "puzzle_id": self.puzzle_id,
            "applied": self.applied,
            "passed": self.passed,
            "real_pairs": self.real_pairs,
            "held_out_index": self.held_out_index,
            "views": self.views,
            "candidates": self.candidates,
            "votes": list(self.votes),
            "expected_rank": self.expected_rank,
            "seconds": round(self.seconds, 3),
            "error": self.error,
        }


Fit = Callable[[DemoSet], Any]
Decode = Callable[[Any, DemoSet, Grid, int], Iterable[Candidate]]


class HeldOutGate:
    """B4's ``Gate`` for P2.1: does the adapter solve the real pair it never saw?

    Why: the refinement loop promotes a test prediction to a pseudo-pair only if the adapter
    trained with it still solves a held-out real pair (B5 P2.1, P2.2). The gate owns the two
    places the held-out pair could leak: training (``fit`` receives the state without it) and
    the held-out input's prompts (``passes`` decodes it with ``pairs`` as the only
    demonstrations). The adapter provider and the decoder are injected (ADR-004), so the gate
    runs unchanged with SGD TTT (``solve_puzzle``), later providers, and CPU mocks.

    Attributes:
        views: Augmented views the held-out input is decoded under, :math:`n`.
    """

    def __init__(
        self,
        fit: Fit,
        decode: Decode,
        *,
        views: int = DECODE_VIEWS,
        select: Select | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Build a gate.

        Args:
            fit: Trains an adapter on a state's ``pairs`` and returns it:
                ``fit(demos) -> adapter``; it receives the state without its held-out pair.
            decode: Decodes one input grid with an adapter under ``views`` augmented views, with
                the state's ``pairs`` as the prompt's demonstrations, and returns the view
                candidates: ``decode(adapter, demos, input, views) -> [Candidate, ...]``; it
                receives the state without its held-out pair.
            views: :math:`n`, at least 1; ``DECODE_VIEWS`` (16) is what ``decode_puzzle``
                decodes.
            select: The selection algorithm of ``vote``; the reference's ``score_kgmon`` when
                None.
            clock: Seconds for the verdict's wall time.

        Raises:
            TypeError: If a callable is not callable or ``views`` is not an integer.
            ValueError: If ``views`` is under 1.
        """
        if not callable(fit) or not callable(decode) or not callable(clock):
            raise TypeError("fit, decode, and clock must be callable")
        if select is not None and not callable(select):
            raise TypeError(f"select must be callable; got {select!r}")
        if type(views) is not int:
            raise TypeError(f"views must be an integer; got {views!r}")
        if views < 1:
            raise ValueError(f"views must be at least 1; got {views}")
        self._fit = fit
        self._decode = decode
        self._select = select
        self._clock = clock
        self.views = views

    def fit(self, demos: DemoSet) -> Any:
        """Train an adapter on the state's training pairs; the held-out pair is not passed on.

        Args:
            demos: The puzzle's state.

        Returns:
            The adapter the fit callable returns.
        """
        return self._fit(demos.without_held_out())

    def verdict(self, adapter: Any, demos: DemoSet) -> GateVerdict:
        """Decode the held-out input with the adapter, vote, and compare with the held-out output.

        Args:
            adapter: The adapter to check, as the fit callable returned it.
            demos: The puzzle's state, with its held-out pair.

        Returns:
            The verdict: passed when the voted answer equals the held-out output exactly. No
            candidate fails (for example a decode stopped at the deadline).

        Raises:
            ValueError: If the state holds no held-out pair.
            TypeError: If the decode callable yields something other than a ``Candidate``.
        """
        held = demos.held_out
        if held is None:
            raise ValueError(f"{demos.puzzle_id}: no held-out pair to check")
        start = self._clock()
        candidates = list(self._decode(adapter, demos.without_held_out(), held.input, self.views))
        if not all(isinstance(candidate, Candidate) for candidate in candidates):
            raise TypeError("the decode callable must yield Candidate objects")
        ranked = vote(candidates, self._select)
        counts: dict[Grid, int] = {}
        for candidate in candidates:
            counts[candidate.grid] = counts.get(candidate.grid, 0) + 1
        rank = ranked.index(held.output) + 1 if held.output in ranked else None
        return GateVerdict(
            puzzle_id=demos.puzzle_id,
            applied=True,
            passed=rank == 1,
            real_pairs=demos.real_pairs,
            held_out_index=demos.held_out_index,
            views=self.views,
            candidates=len(candidates),
            votes=tuple(counts[grid] for grid in ranked),
            expected_rank=rank,
            seconds=self._clock() - start,
        )

    def passes(self, adapter: Any, demos: DemoSet) -> bool:
        """B4's ``Gate.passes``: whether the voted answer for the held-out input is exact.

        Args:
            adapter: The adapter to check.
            demos: The puzzle's state, with its held-out pair.

        Returns:
            ``verdict(adapter, demos).passed``.
        """
        return bool(self.verdict(adapter, demos).passed)


def _seed(name: str, value: object, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if not 0 <= value < SEED_LIMIT:
        raise ValueError(f"{name} must be in [0, 2**32); got {value}")


def view_keys(
    key: str, task: Mapping[str, Any], index: int, seed: int | None = None
) -> tuple[str, ...]:
    """Return the augmented views under which ``decode_puzzle`` decodes and rescores one input.

    Why: P2.2's re-decode must share no view with the decode whose candidates it checks, and
    that is checked, not assumed from the seed (B5 P2.2, ADR-014). ``decode_puzzle`` decodes test
    input ``index`` of a one-key dataset under the 16 views that ``augment(n=2, seed=2)`` draws
    for it (the permutations are drawn for every test input of the call in turn, so the input's
    views depend on the inputs before it), and rescores each new grid under the 8 views of
    ``augment(seed=hash(bk) % 1024**2)``; with ``view_seed`` the seeds are ``view_seed`` and
    ``hash((bk, view_seed)) % 1024**2`` (``arc26/decode.py``). This runs the same calls of the
    moved ``ArcDataset`` on the same task and keeps numpy's global random state as it was.

    Args:
        key: The dataset's key, the task id ``decode_puzzle`` gets.
        task: The task entry decoded, ``{"train": [...], "test": [...]}``, as ``DemoSet.task``
            builds it.
        index: Which of its test inputs.
        seed: ``decode_puzzle``'s ``view_seed``; None for the reference's views.

    Returns:
        The 16 decoding views' keys in ``decode_puzzle``'s order, then the 8 rescoring views'
        keys, each ``<key>_<index>.<transforms>.permute<colors>.ex<order>``. A prompt too long
        for the model is later cut by ``cut_to_len``, which renames the ``ex`` part and
        drops the leading demonstrations.

    Raises:
        TypeError: If ``index`` or ``seed`` is not an integer.
        ValueError: If ``index`` is not a test input of ``task``, or ``seed`` is outside
            ``[0, 2**32)``.
    """
    if type(index) is not int:
        raise TypeError(f"index must be an integer; got {index!r}")
    if not 0 <= index < len(task["test"]):
        raise ValueError(f"index {index} is not one of the task's {len(task['test'])} inputs")
    _seed("seed", seed, optional=True)
    import numpy as np

    from arc26.nvarc.arc_loader import ArcDataset

    base = f"{key}_{index}"
    state = np.random.get_state()
    try:
        multi = ArcDataset(queries={key: task}).split_multi_replies()
        decoded = multi.augment(n=2, seed=2 if seed is None else seed)
        rescoring = ArcDataset(keys=[base], queries={base: multi.queries[base]}).augment(
            seed=hash(base) % 1024**2 if seed is None else hash((base, seed)) % 1024**2
        )
    finally:
        np.random.set_state(state)
    own = sorted(name for name in decoded.keys if name.split(".")[0] == base)
    return tuple(own) + tuple(rescoring.keys)


def view_signatures(keys: Iterable[str], grids: Sequence[Grid]) -> frozenset[tuple[Grid, ...]]:
    """Return what each view shows the model of the given grids.

    Why: two views that differ only in how they recolor colors the puzzle does not use, or in a
    transform its grids are symmetric under, show the model the same prompt; a few-color puzzle
    has few distinct recolorings, so permutations that differ often agree on its colors. A view
    is therefore compared by its image of the prompt's grids (the moved ``forward_mod``: its
    transforms, then its permutation), the demonstrations' order aside. Equal (transform,
    permutation) views have equal images, so disjoint images imply disjoint views.

    Args:
        keys: Augmented keys, as ``view_keys`` returns them.
        grids: The prompt's grids: the demonstrations' inputs and outputs, then the query.

    Returns:
        The distinct images, each the grids under one view, in the given order.
    """
    import numpy as np

    from arc26.nvarc.arc_loader import ArcDataset

    arrays = [np.asarray(grid) for grid in grids]
    return frozenset(
        tuple(tuple(map(tuple, ArcDataset.forward_mod(array, name).tolist())) for array in arrays)
        for name in keys
    )


@dataclass(frozen=True)
class ViewCheck:
    """Whether a re-decode's views share none with the decode that produced the candidates.

    Attributes:
        seed: The re-decode's ``view_seed``: the first seed tried that shares no view, or the
            last one tried when none did.
        tries: Seeds tried, from the first one upward.
        first: Distinct views of the first decode, by their image of the prompt's grids.
        fresh: Distinct views of the re-decode at ``seed``.
        common: Views the two share; 0 when disjoint.
    """

    seed: int
    tries: int
    first: int
    fresh: int
    common: int

    @property
    def disjoint(self) -> bool:
        """Whether no view is shared."""
        return self.common == 0

    def to_record(self) -> dict[str, int]:
        """Return the check as a JSON-ready dict.

        Returns:
            Every field.
        """
        return {
            "seed": self.seed,
            "tries": self.tries,
            "first": self.first,
            "fresh": self.fresh,
            "common": self.common,
        }


def _first_inputs(value: object, inputs: int) -> tuple[int, ...] | None:
    # The test inputs one decode decoded together, in order (P2.3); None means every input.
    if value is None:
        return None
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"first_inputs must be None or a sequence of test indices; got {value!r}")
    found = tuple(value)
    if not all(type(index) is int for index in found):
        raise TypeError(f"first_inputs must hold integer test indices; got {found!r}")
    if not found or list(found) != sorted(set(found)) or not all(0 <= i < inputs for i in found):
        raise ValueError(
            f"first_inputs must be distinct test indices in increasing order, below {inputs}; "
            f"got {found!r}"
        )
    return found


def fresh_views(
    demos: DemoSet,
    index: int,
    *,
    first_seed: int | None = None,
    fresh_seed: int = FRESH_SEED,
    tries: int = VIEW_SEED_TRIES,
    first_inputs: Sequence[int] | None = None,
) -> ViewCheck:
    """Find the first view seed whose re-decode of one test input shares no view with the first.

    Why: B5 P2.2's fresh, disjoint augmentation set, checked (ADR-014). The first decode is the
    one that produced the candidates: by default every test input of the puzzle in one
    ``decode_puzzle`` call under ``first_seed`` (round 0's decode); P2.3's decode after a kept
    round decodes only the inputs that are not pseudo-pairs, ``first_inputs``, in one call, and
    an input's views depend on the inputs decoded before it. The re-decode decodes the input
    alone, as test input 0 of the puzzle's key, under each seed from ``fresh_seed`` upward.
    Views are compared by their image of the re-decode's prompt (``view_signatures``): the
    demonstrations ``demos.pairs`` and the input.

    Args:
        demos: The puzzle's state; its held-out pair is in no prompt (ADR-013).
        index: The test input.
        first_seed: The first decode's ``view_seed``; None for the reference's views.
        fresh_seed: The first seed to try.
        tries: Seeds to try, at least 1.
        first_inputs: The test inputs the first decode decoded, in increasing order; None for
            every input.

    Returns:
        The check at the first disjoint seed, or at the last one tried.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``index`` is not a test input or not among ``first_inputs``, a seed is
            outside ``[0, 2**32)``, ``tries`` is under 1, ``first_inputs`` is malformed, or the
            state holds no pair.
    """
    _seed("first_seed", first_seed, optional=True)
    _seed("fresh_seed", fresh_seed)
    if type(tries) is not int:
        raise TypeError(f"tries must be an integer; got {tries!r}")
    if tries < 1:
        raise ValueError(f"tries must be at least 1; got {tries}")
    _seed("the last seed tried", fresh_seed + tries - 1)
    context = demos.without_held_out()
    if type(index) is not int:
        raise TypeError(f"index must be an integer; got {index!r}")
    if not 0 <= index < len(context.test_inputs):
        raise ValueError(f"index {index} is not one of {len(context.test_inputs)} test inputs")
    layout = _first_inputs(first_inputs, len(context.test_inputs))
    query = context.test_inputs[index]
    grids = [grid for pair in context.pairs for grid in (pair.input, pair.output)] + [query]
    key = context.puzzle_id
    if layout is None:
        first_task, position = context.task(), index
    else:
        if index not in layout:
            raise ValueError(f"index {index} was not decoded by the first decode, {layout}")
        first_task = context.task([context.test_inputs[i] for i in layout])
        position = layout.index(index)
    first = view_signatures(view_keys(key, first_task, position, first_seed), grids)
    alone = context.task([query])
    for seed in range(fresh_seed, fresh_seed + tries):
        fresh = view_signatures(view_keys(key, alone, 0, seed), grids)
        common = len(first & fresh)
        if not common:
            break
    return ViewCheck(
        seed=seed, tries=seed - fresh_seed + 1, first=len(first), fresh=len(fresh), common=common
    )


@dataclass(frozen=True)
class RedecodeVerdict:
    """The re-decode of one test input under fresh views, with the adapter that voted (P2.2).

    Attributes:
        candidates: View candidates the re-decode yielded.
        votes: For each distinct grid, best first, the view candidates that produced it.
        expected_rank: Place of the voted answer under check in the re-decode's vote, 1 when it
            is the re-decode's voted answer; None when no candidate equals it.
        tied: Whether the re-decode's top two grids tie in the vote.
        agreed: Whether the re-decode's voted answer is the one under check, untied, from all its
            views.
        seconds: Wall time of the re-decode and its vote.
        error: Why the re-decode raised, ``"<type>: <message>"``; None when it ran.
        cut: Whether the re-decode stopped before all its DFS batches ran (``DecodeCut``, P2.3):
            its candidates are counted, and it never agrees.
    """

    candidates: int = 0
    votes: tuple[int, ...] = field(default_factory=tuple)
    expected_rank: int | None = None
    tied: bool = False
    agreed: bool = False
    seconds: float = 0.0
    error: str | None = None
    cut: bool = False

    def to_record(self) -> dict[str, Any]:
        """Return the verdict as a JSON-ready dict.

        Returns:
            Every field; seconds rounded to milliseconds.
        """
        return {
            "candidates": self.candidates,
            "votes": list(self.votes),
            "expected_rank": self.expected_rank,
            "tied": self.tied,
            "agreed": self.agreed,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "cut": self.cut,
        }


@dataclass(frozen=True)
class Verdicts:
    """The checks that promoted a pseudo-pair, its provenance in the ``DemoSet`` (P2.2, B4).

    Attributes:
        views: The re-decode's view check.
        redecode: The re-decode under those views.
        gate: The gate's verdict on the candidate adapter fit with the pair.
    """

    views: ViewCheck
    redecode: RedecodeVerdict
    gate: GateVerdict


@dataclass(frozen=True)
class PromotionRecord:
    """Why one test input's voted answer was or was not promoted, one JSON line each (P2.2).

    Attributes:
        puzzle_id: The task id.
        round_added: The refinement round the promotion is for.
        test_index: The test input.
        candidates: View candidates of the decode that produced the voted answer.
        votes: For each distinct grid, best first, the view candidates that produced it.
        agreement: The voted answer's agreement (``Tally.agreement``); None without candidates.
        tau: The agreement threshold.
        tied: Whether the vote's top two grids tie.
        views: The re-decode's view check; None when not reached.
        redecode: The re-decode's verdict; None when not run.
        gate: The gate's verdict on the candidate adapter; None when not run.
        fitted: The test inputs whose pseudo-pairs the candidate adapter was fit with; empty when
            no fit was reached.
        promoted: Whether the voted answer became a pseudo-pair.
        reason: ``PROMOTED`` or the first check that failed, one of ``REASONS``.
        error: Why the candidate adapter's fit raised; None otherwise (a raising re-decode or
            gate keeps its error in its verdict).
    """

    puzzle_id: str
    round_added: int
    test_index: int
    candidates: int
    votes: tuple[int, ...]
    agreement: float | None
    tau: float
    tied: bool
    views: ViewCheck | None = None
    redecode: RedecodeVerdict | None = None
    gate: GateVerdict | None = None
    fitted: tuple[int, ...] = ()
    promoted: bool = False
    reason: str = NO_CANDIDATES
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the record as a JSON-ready dict, for the promotion's JSONL log.

        Returns:
            Every field, the verdicts as nested dicts (null when not reached).
        """
        return {
            "puzzle_id": self.puzzle_id,
            "round": self.round_added,
            "test_index": self.test_index,
            "candidates": self.candidates,
            "votes": list(self.votes),
            "agreement": self.agreement,
            "tau": self.tau,
            "tied": self.tied,
            "views": None if self.views is None else self.views.to_record(),
            "redecode": None if self.redecode is None else self.redecode.to_record(),
            "gate": None if self.gate is None else self.gate.to_record(),
            "fitted": list(self.fitted),
            "promoted": self.promoted,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class Promotion:
    """What ``promote`` decided for one puzzle in one round (P2.2).

    Attributes:
        demos: The next version, with the promoted pseudo-pairs appended, when any was promoted;
            otherwise the state given.
        adapter: The candidate adapter, fit with the promoted pairs and passed by the gate; None
            when nothing was promoted (keep the previous adapter; with a provider that trains in
            place, restore its weights).
        pairs: The promoted pseudo-pairs in test input order: B4's ``promote`` return value.
        records: One record per test input, in order.
        gate: The gate's verdict on the candidate adapter; None when no adapter was fit.
    """

    demos: DemoSet
    adapter: Any
    pairs: tuple[Pair, ...]
    records: tuple[PromotionRecord, ...]
    gate: GateVerdict | None = None

    @property
    def promoted(self) -> bool:
        """Whether any pseudo-pair was promoted."""
        return bool(self.pairs)


Adapt = Callable[[DemoSet, Any], Any]
Redecode = Callable[[Any, DemoSet, Grid, int], Iterable[Candidate]]


class DecodeCut(Exception):
    """A decode that ran fewer DFS batches than its views need (task P2.3, ADR-015).

    Why: ``decode_puzzle`` checks its 1200 s cap and the run's end time before each DFS batch and
    stops there, so a decode can return with some of its views never decoded. A re-decode or a
    gate check that voted on what is left would pass on its top grid alone; the round's GPU
    callables raise this instead, so the check fails (``REDECODE_CUT``, or the gate's
    ``GATE_RAISED`` with this error).

    Attributes:
        candidates: The view candidates of the batches that ran.
        batches: DFS batches that ran.
        expected: DFS batches the decode's views need.
    """

    def __init__(self, candidates: Iterable[Candidate], batches: int, expected: int) -> None:
        self.candidates = tuple(candidates)
        self.batches = batches
        self.expected = expected
        super().__init__(f"{batches} of {expected} DFS batches ran before the time cap")


def _error(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def _promote_args(gate: object, adapt: object, redecode: object, clock: object) -> None:
    if not callable(getattr(gate, "verdict", None)):
        raise TypeError(f"gate must have a verdict method (HeldOutGate); got {gate!r}")
    if not callable(adapt) or not callable(redecode) or not callable(clock):
        raise TypeError("adapt, redecode, and clock must be callable")


def _screen_args(
    candidates: object,
    demos: object,
    tau: object,
    round_added: object,
    views: object,
    first_seed: object,
    fresh_seed: object,
    tries: object,
    first_inputs: object = None,
) -> list[list[Candidate]]:
    # The checks screen and promote share; returns the candidates as lists, one per test input.
    if not isinstance(demos, DemoSet):
        raise TypeError(f"demos must be a DemoSet; got {demos!r}")
    if isinstance(candidates, str | bytes) or not isinstance(candidates, Sequence):
        raise TypeError(f"candidates must hold one sequence per test input; got {candidates!r}")
    if len(candidates) != len(demos.test_inputs):
        raise ValueError(
            f"{demos.puzzle_id}: candidates for {len(candidates)} test inputs; the puzzle has "
            f"{len(demos.test_inputs)}"
        )
    if isinstance(tau, bool) or not isinstance(tau, int | float):
        raise TypeError(f"tau must be a number; got {tau!r}")
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1]; got {tau!r}")
    if type(round_added) is not int:
        raise TypeError(f"round_added must be an integer; got {round_added!r}")
    if round_added < 1:
        raise ValueError(f"pseudo-pairs are added in round 1 or later; got {round_added}")
    if type(views) is not int:
        raise TypeError(f"views must be an integer; got {views!r}")
    if views < 1:
        raise ValueError(f"views must be at least 1; got {views}")
    found = [list(entry) for entry in candidates]
    if not all(isinstance(c, Candidate) for entry in found for c in entry):
        raise TypeError("candidates must hold Candidate objects")
    _seed("first_seed", first_seed, optional=True)
    _seed("fresh_seed", fresh_seed)
    if type(tries) is not int:
        raise TypeError(f"tries must be an integer; got {tries!r}")
    if tries < 1:
        raise ValueError(f"tries must be at least 1; got {tries}")
    _seed("the last seed tried", fresh_seed + tries - 1)  # type: ignore[operator]
    _first_inputs(first_inputs, len(demos.test_inputs))
    return found


@dataclass(frozen=True)
class Screened:
    r"""One test input after a promotion's free checks, which cost no GPU work (task P2.3).

    Why: the governor prices a round before any of its GPU ops runs, so the round needs to know
    which inputs can still be promoted (each costs a re-decode and joins the candidate fit) from
    the checks that cost nothing: a candidate exists, the vote is not tied, the agreement reaches
    :math:`\tau`, the input is not a pseudo-pair yet, a pair is held out, and a fresh view seed
    exists (ADR-014's order). ``promote`` runs the same checks through ``screen``.

    Attributes:
        index: The test input.
        tally: Its vote.
        record: Its promotion record so far: final, with the failing check's reason, unless
            ``passed``.
        check: The re-decode's view check, once reached.
        passed: Whether every free check passed, so that the re-decode runs.
    """

    index: int
    tally: Tally
    record: PromotionRecord
    check: ViewCheck | None = None
    passed: bool = False


def _screen(
    demos: DemoSet,
    found: Sequence[Sequence[Candidate]],
    tau: float,
    views: int,
    round_added: int,
    first_seed: int | None,
    fresh_seed: int,
    tries: int,
    first_inputs: Sequence[int] | None = None,
) -> tuple[Screened, ...]:
    # The free checks per test input, cheapest first; the first that fails ends the input's.
    promoted_inputs = {pair.input for pair in demos.pairs if pair.kind == PSEUDO}
    screened = []
    for index, (query, entry) in enumerate(zip(demos.test_inputs, found, strict=True)):
        votes = tally(entry, views)
        record = PromotionRecord(
            puzzle_id=demos.puzzle_id,
            round_added=round_added,
            test_index=index,
            candidates=len(entry),
            votes=votes.counts,
            agreement=votes.agreement,
            tau=float(tau),
            tied=votes.tied,
        )
        reason = None
        if not entry:
            reason = NO_CANDIDATES
        elif votes.tied:
            reason = TIED
        elif votes.agreement is None or votes.agreement < tau:
            reason = BELOW_TAU
        elif query in promoted_inputs:
            reason = ALREADY_PROMOTED
        elif demos.held_out is None:
            reason = NO_HELD_OUT
        if reason is not None:
            screened.append(Screened(index, votes, replace(record, reason=reason)))
            continue
        check = fresh_views(
            demos,
            index,
            first_seed=first_seed,
            fresh_seed=fresh_seed,
            tries=tries,
            first_inputs=first_inputs,
        )
        record = replace(record, views=check)
        if not check.disjoint:
            failed = replace(record, reason=VIEWS_NOT_DISJOINT)
            screened.append(Screened(index, votes, failed, check))
            continue
        screened.append(Screened(index, votes, record, check, passed=True))
    return tuple(screened)


def screen(
    candidates: Sequence[Iterable[Candidate]],
    demos: DemoSet,
    tau: float = TAU,
    *,
    round_added: int = 1,
    fresh_seed: int = FRESH_SEED,
    first_seed: int | None = None,
    views: int = DECODE_VIEWS,
    tries: int = VIEW_SEED_TRIES,
    first_inputs: Sequence[int] | None = None,
) -> tuple[Screened, ...]:
    """Apply a promotion's free checks to every test input: no GPU work (task P2.3).

    Why: ``refine`` prices a round from the inputs that pass them (``Screened``): each costs a
    re-decode, and together they bound the candidate fit's training set. The checks are
    ``promote``'s own, in its order.

    Args:
        candidates: Per test input of ``demos``, the view candidates of the decode that produced
            the voted answers (``promote``'s ``candidates``).
        demos: The puzzle's state.
        tau: The agreement threshold, in [0, 1].
        round_added: The round, at least 1, for the records.
        fresh_seed: The first view seed a re-decode would try.
        first_seed: The ``view_seed`` of the decode that produced ``candidates``.
        views: :math:`n`, the views each input was decoded under.
        tries: View seeds tried per input before giving up.
        first_inputs: The test inputs the decode that produced ``candidates`` decoded together,
            in increasing order; None for every input (``fresh_views``).

    Returns:
        One entry per test input, in order.

    Raises:
        TypeError: As ``promote`` raises it for these arguments.
        ValueError: As ``promote`` raises it for these arguments.
    """
    found = _screen_args(
        candidates, demos, tau, round_added, views, first_seed, fresh_seed, tries, first_inputs
    )
    return _screen(
        demos, found, tau, views, round_added, first_seed, fresh_seed, tries, first_inputs
    )


def promote(
    candidates: Sequence[Iterable[Candidate]],
    gate: HeldOutGate,
    adapter: Any,
    demos: DemoSet,
    tau: float = TAU,
    *,
    adapt: Adapt,
    redecode: Redecode,
    round_added: int = 1,
    fresh_seed: int = FRESH_SEED,
    first_seed: int | None = None,
    views: int = DECODE_VIEWS,
    tries: int = VIEW_SEED_TRIES,
    timing_logger: timing.TimingLogger | None = None,
    clock: Callable[[], float] = time.perf_counter,
    first_inputs: Sequence[int] | None = None,
) -> Promotion:
    r"""Promote the test inputs' voted answers that pass three independent checks (B5 P2.2).

    Why: a wrong pseudo-pair teaches the adapter a wrong rule, so a voted answer
    :math:`\hat y_i` becomes a pseudo-pair :math:`(x_i, \hat y_i)` only when (1) its agreement,
    the share of the :math:`n` views that produced it (the count term of the reference's
    ``score_kgmon``, ``Tally``), is at least :math:`\tau`; (2) it survives a re-decode with the
    same adapter under fresh views that share none with the decode that produced it
    (``fresh_views``): the re-decode's voted answer is :math:`\hat y_i`, untied, from all its
    views; (3) the candidate adapter, fit with it from the current one (Phase 2:
    ``sgd_continue``), passes the gate on the real pair held out of everything. The checks run
    cheapest first and stop at the first that fails (ADR-014): the free ones (``screen``: a
    candidate exists, the vote is not tied, agreement, not already promoted, a held-out pair
    exists, a fresh view seed exists), then the re-decode per input, then one fit and one gate
    check for every input that passed both: one candidate adapter per round, fit on the state
    with all their pseudo-pairs, so a failing gate promotes none of them. A re-decode, fit, or
    gate that raises fails its check, with its error logged, and the worker goes on; so does a
    re-decode cut short by the time cap (``DecodeCut``, P2.3). The re-decode uses ``adapter``,
    not the candidate adapter, which has seen the pair. With a logger, each re-decode's span is
    recorded as ``redecode`` and the candidate adapter's gate check as ``gate``.

    Args:
        candidates: Per test input of ``demos``, in order, the view candidates of the decode that
            produced the voted answers: every input in one ``decode_puzzle`` call under
            ``first_seed``, with ``demos.pairs`` as the prompts' demonstrations (ADR-013).
        gate: The held-out gate; its ``verdict(adapter, demos)`` checks the candidate adapter.
        adapter: The current adapter, which produced ``candidates``; the re-decode uses it and
            the candidate fit starts from it.
        demos: The puzzle's state.
        tau: The agreement threshold, in [0, 1]; ``TAU`` (9/16) by default (ADR-014).
        adapt: Fits the candidate adapter: ``adapt(state, init) -> adapter``, with ``state`` the
            enlarged state without its held-out pair and ``init`` the current adapter.
        redecode: Decodes one input under fresh views: ``redecode(adapter, state, query, seed) ->
            [Candidate, ...]``, with ``state`` without its held-out pair (its ``pairs`` are the
            demonstrations), ``query`` decoded alone as test input 0 of the puzzle's key, and
            ``seed`` passed to ``decode_puzzle`` as ``view_seed``; it raises ``DecodeCut`` when
            the time cap stopped the decode before all its batches.
        round_added: The refinement round, at least 1; the pseudo-pairs carry it.
        fresh_seed: The first view seed the re-decode tries.
        first_seed: The ``view_seed`` of the decode that produced ``candidates``; None for the
            reference's views.
        views: :math:`n`, the views each input was decoded under (16).
        tries: View seeds tried per input before giving up.
        timing_logger: Records the ``redecode`` and ``gate`` spans; None records nothing.
        clock: Seconds for the verdicts' wall times and the spans.
        first_inputs: The test inputs the decode that produced ``candidates`` decoded together,
            in increasing order (P2.3: the decode after a kept round decodes only the inputs that
            are not pseudo-pairs); None for every input, round 0's decode.

    Returns:
        The promotion: the next ``DemoSet`` version and the candidate adapter when a pair was
        promoted, and one record per test input.

    Raises:
        TypeError: If an argument has the wrong type, or ``candidates`` or the re-decode yields
            something other than a ``Candidate``.
        ValueError: If an argument is out of range or ``candidates`` does not hold one entry
            per test input.
    """
    _promote_args(gate, adapt, redecode, clock)
    found = _screen_args(
        candidates, demos, tau, round_added, views, first_seed, fresh_seed, tries, first_inputs
    )
    screened = _screen(
        demos, found, tau, views, round_added, first_seed, fresh_seed, tries, first_inputs
    )
    return _promote(
        screened,
        gate,
        adapter,
        demos,
        adapt=adapt,
        redecode=redecode,
        round_added=round_added,
        timing_logger=timing_logger,
        clock=clock,
    )


def _shown(
    guard: PromptGuard | None,
    log: RoundLog | None,
    state: DemoSet,
    queries: Sequence[int] | None,
    round_added: int,
    name: str,
    held_out: bool = False,
) -> DemoSet:
    # The state as one decode's prompts hold it (the length guard, task P2.5, ADR-016): its
    # pseudo-pairs as far as the longest prompt fits the limit, the latest dropped first. A
    # decode that drops one writes a context record; one without a pseudo-pair is not measured.
    # queries: the test indices decoded, or None for the held-out input (a gate).
    if guard is None or not state.pseudo_pairs:
        return state
    if queries is None:
        if state.held_out is None:
            raise ValueError(f"{state.puzzle_id}: a gate's prompts need the held-out input")
        grids = [state.held_out.input]
    else:
        grids = [state.test_inputs[index] for index in queries]
    fit = guard.fit(state, grids, held_out=held_out)
    if not fit.dropped:
        return state
    if log is not None:
        log.context(
            ContextRecord(
                puzzle_id=state.puzzle_id,
                round_added=round_added,
                name=name,
                prompts=HELD_OUT_PROMPTS if held_out else PAIRS_PROMPTS,
                queries=None if queries is None else tuple(queries),
                kept=fit.kept,
                dropped=fit.dropped,
                lengths=fit.lengths,
                limit=fit.limit,
            )
        )
    return state.first_pseudo(len(fit.kept))


def _promote(
    screened: Sequence[Screened],
    gate: HeldOutGate,
    adapter: Any,
    demos: DemoSet,
    *,
    adapt: Adapt,
    redecode: Redecode,
    round_added: int,
    timing_logger: timing.TimingLogger | None,
    clock: Callable[[], float],
    guard: PromptGuard | None = None,
    log: RoundLog | None = None,
) -> Promotion:
    # promote after its free checks: the re-decode of every input that passed them, then one
    # candidate fit and one gate check for every input whose re-decode agreed (ADR-014). With a
    # guard (P2.5), each re-decode's and the gate's prompts hold the pseudo-pairs that fit.
    context = demos.without_held_out()
    records: dict[int, PromotionRecord] = {item.index: item.record for item in screened}
    passed: dict[int, tuple[Tally, ViewCheck, RedecodeVerdict]] = {}
    for item in screened:
        if not item.passed or item.check is None:
            continue
        query = demos.test_inputs[item.index]
        show = partial(_show_redecode, guard, log, round_added, item.index)
        verdict = _redecode(
            redecode, adapter, context, query, item.check, item.tally, timing_logger, clock, show
        )
        record = replace(item.record, redecode=verdict)
        if verdict.error is not None:
            records[item.index] = replace(record, reason=REDECODE_RAISED)
        elif verdict.cut:
            records[item.index] = replace(record, reason=REDECODE_CUT)
        elif not verdict.agreed:
            records[item.index] = replace(record, reason=REDECODE_DISAGREES)
        else:
            records[item.index] = record
            passed[item.index] = (item.tally, item.check, verdict)
    if not passed:
        ordered = tuple(records[index] for index in sorted(records))
        return Promotion(demos=demos, adapter=None, pairs=(), records=ordered)

    fitted = tuple(sorted(passed))
    drafts = [
        Pair(
            demos.test_inputs[index],
            passed[index][0].voted,
            kind=PSEUDO,
            round_added=round_added,
            agreement=passed[index][0].agreement,
            test_index=index,
        )
        for index in fitted
    ]
    enlarged = demos.enlarged(drafts)
    candidate: Any = None
    gate_verdict: GateVerdict | None = None
    error: str | None = None
    try:
        candidate = adapt(enlarged.without_held_out(), adapter)
    except Exception as err:
        traceback.print_exc()
        error, reason = _error(err), FIT_RAISED
    else:
        start = clock()
        try:
            shown = _shown(guard, log, enlarged, None, round_added, f"round{round_added}-gate")
            gate_verdict = gate.verdict(candidate, shown)
        except Exception as err:
            traceback.print_exc()
            gate_verdict = GateVerdict.raised(enlarged, err, clock() - start)
            reason = GATE_RAISED
        else:
            if timing_logger is not None:
                timing_logger.record(demos.puzzle_id, GATE, clock() - start)
            reason = PROMOTED if gate_verdict.passed else GATE_FAILED
    for index in fitted:
        records[index] = replace(
            records[index],
            gate=gate_verdict,
            fitted=fitted,
            promoted=reason == PROMOTED,
            reason=reason,
            error=error,
        )
    ordered = tuple(records[index] for index in sorted(records))
    if reason != PROMOTED or gate_verdict is None:
        return Promotion(demos=demos, adapter=None, pairs=(), records=ordered, gate=gate_verdict)
    pairs = tuple(
        replace(
            draft,
            verdicts=Verdicts(views=passed[index][1], redecode=passed[index][2], gate=gate_verdict),
        )
        for index, draft in zip(fitted, drafts, strict=True)
    )
    return Promotion(
        demos=demos.enlarged(pairs),
        adapter=candidate,
        pairs=pairs,
        records=ordered,
        gate=gate_verdict,
    )


def _show_redecode(
    guard: PromptGuard | None, log: RoundLog | None, round_added: int, index: int, state: DemoSet
) -> DemoSet:
    # The prompts of round round_added's re-decode of test input index (the length guard, P2.5).
    return _shown(guard, log, state, (index,), round_added, f"round{round_added}-redecode{index}")


def _redecode(
    redecode: Redecode,
    adapter: Any,
    context: DemoSet,
    query: Grid,
    check: ViewCheck,
    votes: Tally,
    timing_logger: timing.TimingLogger | None,
    clock: Callable[[], float],
    show: Callable[[DemoSet], DemoSet] | None = None,
) -> RedecodeVerdict:
    # The re-decode of one input under the check's seed, voted as the first decode was. A
    # re-decode that raises (out of memory, say) fails the check and writes no timing record; one
    # the time cap cut short (DecodeCut) ran, so its span is recorded, but it never agrees. show
    # (P2.5) gives the state as its prompts hold it, the length guard applied.
    start = clock()
    cut = False
    try:
        prompt = context if show is None else show(context)
        found = list(redecode(adapter, prompt, query, check.seed))
    except DecodeCut as short:
        found, cut = list(short.candidates), True
    except Exception as err:
        traceback.print_exc()
        return RedecodeVerdict(seconds=clock() - start, error=_error(err))
    if not all(isinstance(candidate, Candidate) for candidate in found):
        raise TypeError("the redecode callable must yield Candidate objects")
    again = tally(found, votes.views)
    seconds = clock() - start
    if timing_logger is not None:
        timing_logger.record(context.puzzle_id, REDECODE, seconds)
    rank = again.grids.index(votes.voted) + 1 if votes.voted in again.grids else None
    return RedecodeVerdict(
        candidates=len(found),
        votes=again.counts,
        expected_rank=rank,
        tied=again.tied,
        agreed=rank == 1 and not again.tied and not cut,
        seconds=seconds,
        cut=cut,
    )


def _directory(name: str, value: object, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string path; got {value!r}")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _variants(value: object) -> tuple[str, ...]:
    # The round variants a run offers (P2.5): gradient, in_context, or both in the governor's
    # order, so that the governor's ladder is the only order a run can mean (B2 Q4).
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"variants must be a sequence of variant names; got {value!r}")
    found = tuple(value)
    if not all(isinstance(name, str) for name in found):
        raise TypeError(f"variants must hold variant names; got {found!r}")
    if (
        not found
        or len(set(found)) != len(found)
        or not set(found) <= set(LADDER)
        or list(found) != sorted(found, key=LADDER.index)
    ):
        raise ValueError(
            f"variants must be {GRADIENT!r}, {IN_CONTEXT!r}, or both in the governor's order "
            f"{list(LADDER)}; got {list(found)}"
        )
    return found


def _governor_config(value: object) -> dict[str, Any]:
    # A copy of a parsed configs/governor.yaml that the governor accepts and JSON can carry: the
    # settings reach the workers inside the solver's --config JSON.
    if not isinstance(value, Mapping):
        raise TypeError(f"governor must be a mapping (configs/governor.yaml); got {value!r}")
    try:
        config = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as err:
        raise TypeError(f"governor must hold JSON values only: {err}") from None
    try:
        governor = Governor.from_mapping(config, end_time=0.0, workers=1)
    except KeyError as err:
        raise ValueError(f"governor: missing key {err}") from None
    if governor.policy != REFERENCE:
        raise ValueError(
            f"governor: decode_policy mode {governor.policy!r} needs a plan before each baseline "
            "DFS batch, which decode_puzzle (protected) does not offer; the rounds plan after "
            f"round 0's decode, so the mode must be {REFERENCE!r}"
        )
    return config


@dataclass(frozen=True)
class Rounds:
    r"""The refinement rounds of one solver run (task P2.3, ADR-015): ``HoldOut.rounds``.

    Why: the rounds' settings reach every worker in the solver's ``--config`` JSON, inside the
    hold-out settings they need (no gate without a held-out pair). The governor's config cannot
    be read from ``configs/`` on Kaggle (the wheel carries ``arc26/`` only), so the notebook
    builder embeds its parsed content here.

    Attributes:
        round_dir: Folder of the rounds' candidate files, one subfolder per puzzle (each round's
            test decode, re-decodes, and gate, and round 0's candidates once replaced, for P2.8);
            never the run's ``outputs_dir``.
        governor: The parsed ``configs/governor.yaml``, for ``Governor.from_mapping``; its
            ``decode_policy`` mode must be ``reference``.
        max_rounds: Rounds per puzzle at most, 1 to ``MAX_ROUNDS`` (B5 P2.3: default 1, hard cap
            2); round 2 runs only after a kept round 1.
        tau: The promotion's agreement threshold (ADR-014), in [0, 1]; P2.7 varies it.
        continue_steps: :math:`s'`, the optimizer steps of a round's ``sgd_continue``, 1 to
            ``EPOCH_VIEWS`` (a fraction of an epoch, ADR-004).
        final_in_prompts: Whether the decode whose candidates are submitted holds the held-out
            pair in its prompts. True, the default: one more decode of the open test inputs runs
            after the rounds, the pair back among the real pairs, whatever the rounds' outcome;
            run ``P2.1-02`` (3 Oct) recovered 2.0 DEV40 tasks with the pair in the test prompts
            (7.0 against 5.0 of 40, its Recovers row), so ADR-013 as amended keeps it in the
            decode that is submitted and out of every decode that feeds a promotion. False
            submits the last kept round's decode, which left it out, as ``P2.1-01`` did.
        variants: The round variants offered (task P2.5, ADR-016), in the governor's order:
            ``("gradient",)``, the default and P2.3's round; ``("in_context",)``, the
            in-context round alone; or both, the governor's ladder (the in-context round when
            the gradient one does not fit, B2 Q4). P2.7 and run ``P2.5-01`` force one to measure
            it. It joins the JSON only when not the default, so P2.3's settings keep their JSON.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``round_dir`` is empty, a number is out of range, the governor config
            is incomplete or not in the ``reference`` decode policy, or ``variants`` is not one
            of the three.
    """

    # P2.3's fields, every one in the JSON; OPTIONAL (task P2.5) joins only when not the default.
    FIELDS: ClassVar[tuple[str, ...]] = (
        "round_dir", "governor", "max_rounds", "tau", "continue_steps", "final_in_prompts"
    )  # fmt: skip
    OPTIONAL: ClassVar[tuple[str, ...]] = ("variants",)

    round_dir: str
    governor: Mapping[str, Any] = field(hash=False)
    max_rounds: int = DEFAULT_ROUNDS
    tau: float = TAU
    continue_steps: int = CONTINUE_STEPS
    # ADR-013 as amended after P2.1-02's Recovers row: the submitted decode keeps the pair.
    final_in_prompts: bool = True
    # P2.3's round alone, so that today's runs are unchanged (ADR-016).
    variants: tuple[str, ...] = DEFAULT_VARIANTS

    def __post_init__(self) -> None:
        _directory("round_dir", self.round_dir)
        object.__setattr__(self, "governor", _governor_config(self.governor))
        for name, high in (("max_rounds", MAX_ROUNDS), ("continue_steps", EPOCH_VIEWS)):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer; got {value!r}")
            if not 1 <= value <= high:
                raise ValueError(f"{name} must be 1 to {high}; got {value}")
        if isinstance(self.tau, bool) or not isinstance(self.tau, int | float):
            raise TypeError(f"tau must be a number; got {self.tau!r}")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1]; got {self.tau!r}")
        object.__setattr__(self, "tau", float(self.tau))
        if not isinstance(self.final_in_prompts, bool):
            raise TypeError(
                f"final_in_prompts must be true or false; got {self.final_in_prompts!r}"
            )
        object.__setattr__(self, "variants", _variants(self.variants))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Rounds:
        """Build the settings from a mapping, such as the solver JSON's ``holdout.rounds``.

        Args:
            mapping: ``round_dir``, ``governor``, and any of the other fields.

        Returns:
            The settings; missing optional fields take their defaults.

        Raises:
            TypeError: If ``mapping`` is not a mapping, or as the constructor raises it.
            ValueError: If a key is unknown, ``round_dir`` or ``governor`` is missing, or as the
                constructor raises it.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"rounds must be a mapping; got {mapping!r}")
        unknown = sorted(set(mapping) - set(cls.FIELDS) - set(cls.OPTIONAL))
        if unknown:
            raise ValueError(
                f"rounds: unknown keys {unknown}; expected {list(cls.FIELDS)} and optionally "
                f"{list(cls.OPTIONAL)}"
            )
        missing = [name for name in ("round_dir", "governor") if name not in mapping]
        if missing:
            raise ValueError(f"rounds needs {', '.join(missing)}")
        return cls(**mapping)

    def to_json(self) -> dict[str, Any]:
        """Return the settings as the solver JSON's ``holdout.rounds`` value.

        Returns:
            Every field of ``FIELDS``, by name, the governor config as plain JSON data; and
            ``variants`` only when not the default, so P2.3's settings keep their JSON.
        """
        payload = {
            name: json.loads(json.dumps(value)) if name == "governor" else value
            for name, value in ((name, getattr(self, name)) for name in self.FIELDS)
        }
        if self.variants != DEFAULT_VARIANTS:
            payload["variants"] = list(self.variants)
        return payload


@dataclass(frozen=True)
class HoldOut:
    """The hold-out settings of one solver run (task P2.1): the solver JSON's ``holdout``.

    Attributes:
        gate_dir: Folder of the gate's candidate files, one subfolder per puzzle; never the
            run's ``outputs_dir``, so no gate file reaches the submission.
        log_dir: Folder of the verdicts, ``gate-rank<rank>.jsonl`` per worker; None writes none.
        rule: The hold-out rule's name in ``HOLDOUT_RULES``.
        k2: Whether a puzzle with exactly 2 real pairs holds one out (B5 P2.1, default on).
        in_test_prompts: Whether the held-out pair stays in the decode and rescoring prompts of
            the real test inputs. False, the loop's choice (ADR-013): the test inputs are
            decoded with ``pairs`` only, as the held-out input is.
        rounds: The refinement rounds after the gate (task P2.3, ``Rounds``), or None, the
            default, to run none. Their records go to ``log_dir`` beside the verdicts. They need
            ``in_test_prompts`` False: round 0's test decode feeds their promotion, and no decode
            that feeds a promotion may see the held-out pair (ADR-013); their final decode, whose
            candidates are submitted, puts it back (``Rounds.final_in_prompts``, default on).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a folder is empty, ``rule`` names no rule, or ``rounds`` is set with
            ``in_test_prompts``.
    """

    # The fields of task P2.1, every one in the JSON; rounds (task P2.3) joins only when set.
    FIELDS: ClassVar[tuple[str, ...]] = ("gate_dir", "log_dir", "rule", "k2", "in_test_prompts")
    ROUNDS: ClassVar[str] = "rounds"

    gate_dir: str
    log_dir: str | None = None
    rule: str = DEFAULT_RULE
    k2: bool = True
    in_test_prompts: bool = False
    rounds: Rounds | None = None

    def __post_init__(self) -> None:
        _directory("gate_dir", self.gate_dir)
        _directory("log_dir", self.log_dir, optional=True)
        if not isinstance(self.rule, str):
            raise TypeError(f"rule must be a rule name; got {self.rule!r}")
        if self.rule not in HOLDOUT_RULES:
            raise ValueError(f"rule must be one of {sorted(HOLDOUT_RULES)}; got {self.rule!r}")
        for name in ("k2", "in_test_prompts"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be true or false; got {getattr(self, name)!r}")
        if isinstance(self.rounds, Mapping):
            object.__setattr__(self, "rounds", Rounds.from_mapping(self.rounds))
        if self.rounds is not None:
            if not isinstance(self.rounds, Rounds):
                raise TypeError(f"rounds must be None or Rounds settings; got {self.rounds!r}")
            if self.in_test_prompts:
                raise ValueError(
                    "rounds need in_test_prompts False: round 0's test decode feeds their "
                    "promotion, and no decode that feeds a promotion may hold the held-out pair "
                    "(ADR-013); Rounds.final_in_prompts puts it back in the final decode"
                )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> HoldOut:
        """Build the settings from a mapping, such as the solver JSON's ``holdout``.

        Args:
            mapping: ``gate_dir`` and any of the other fields, ``rounds`` as a mapping of
                ``Rounds``' fields.

        Returns:
            The settings; missing optional fields take their defaults.

        Raises:
            TypeError: If ``mapping`` is not a mapping, or as the constructor raises it.
            ValueError: If a key is unknown or ``gate_dir`` is missing, or as the constructor
                raises it.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"holdout must be a mapping; got {mapping!r}")
        unknown = sorted(set(mapping) - set(cls.FIELDS) - {cls.ROUNDS})
        if unknown:
            raise ValueError(
                f"holdout: unknown keys {unknown}; expected {list(cls.FIELDS)} and optionally "
                f"{cls.ROUNDS}"
            )
        if "gate_dir" not in mapping:
            raise ValueError("holdout needs gate_dir, the folder of the gate's candidates")
        return cls(**mapping)

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> HoldOut | None:
        """Read the settings from a solver config (``arc26.runner.solver_command``'s JSON).

        Args:
            cfg: The solver config; its optional ``holdout`` is read.

        Returns:
            The settings, or None when the run holds no pair out (the key is absent or null).
        """
        raw = cfg.get("holdout")
        return None if raw is None else cls.from_mapping(raw)

    def to_json(self) -> dict[str, Any]:
        """Return the settings as the solver JSON's ``holdout`` value.

        Returns:
            Every field of ``FIELDS``, by name, and ``rounds`` (``Rounds.to_json()``) only when
            set, so a run without rounds passes the JSON of task P2.1.
        """
        payload = {name: getattr(self, name) for name in self.FIELDS}
        if self.rounds is not None:
            payload[self.ROUNDS] = self.rounds.to_json()
        return payload


class RenamedOps:
    """A timing logger that records some ops under other names: the gate's decode (P2.1).

    Why: ``decode_puzzle`` records its DFS batches as ``decode`` and its rescorings as
    ``selection``; the gate calls the same function on the held-out input, and those records
    must not mix with the test inputs' (the governor prices the gate's batches apart, P2.4).

    Attributes:
        logger: The worker's logger.
        names: New op name by old one; other ops keep theirs.
    """

    def __init__(self, logger: timing.TimingLogger, names: Mapping[str, str]) -> None:
        """Wrap a logger.

        Args:
            logger: The logger that writes the records.
            names: New op name by old one.
        """
        self.logger = logger
        self.names = dict(names)

    def record(
        self, puzzle_id: str | None, op: str, seconds: float, tokens: int | None = None
    ) -> dict[str, object]:
        """Record one op under its new name, as ``TimingLogger.record`` does.

        Args:
            puzzle_id: Task id, or None.
            op: The op's name as the caller records it.
            seconds: Wall time.
            tokens: Tokens processed, or None.

        Returns:
            The record as written.
        """
        return self.logger.record(puzzle_id, self.names.get(op, op), seconds, tokens)


def read_candidates(folder: str | os.PathLike[str]) -> list[Candidate]:
    """Read the candidate files ``decode_puzzle`` wrote into ``folder``.

    Why: ``decode_puzzle`` writes one bz2 pickle per view that yielded a grid, a list of
    ``{"beam_score", "score_aug", "solution"}``, as the moved selection reads them
    (``ArcDecoder.load_decoded_results``); the gate reads its own folder back to vote. Files
    are read in name order, so the vote does not depend on the directory's listing order.

    Args:
        folder: The gate's folder for one puzzle; a missing folder holds no candidate.

    Returns:
        Every candidate of every file, in file order, then in each file's order.
    """
    if not os.path.isdir(folder):
        return []
    found = []
    for name in sorted(os.listdir(folder)):
        found.extend(_read_file(folder, name))
    return found


def _read_file(folder: str | os.PathLike[str], name: str) -> list[Candidate]:
    # One candidate file of decode_puzzle: a bz2 pickle of {"beam_score", "score_aug",
    # "solution"} rows, named by its view's key.
    import bz2
    import pickle

    with bz2.BZ2File(os.path.join(folder, name)) as handle:
        rows = pickle.load(handle)
    return [
        Candidate(
            grid=row["solution"],
            beam_score=row["beam_score"],
            score_aug=row["score_aug"],
            view=name,
        )
        for row in rows
    ]


class VerdictLog:
    """Appends gate verdicts to ``<log_dir>/gate-<worker>.jsonl``, one line each.

    Why: one file per worker process, as the timing records, so concurrent workers never share
    a file; each line is appended and the file closed at once, so a hard kill loses at most the
    line being written. The promotion's records (P2.2) go to ``promote-<worker>.jsonl`` the same
    way (``prefix=PROMOTION_PREFIX``).

    Attributes:
        path: The worker's file.
    """

    def __init__(
        self, log_dir: str | os.PathLike[str], worker: str, prefix: str = VERDICT_PREFIX
    ) -> None:
        """Create a log; nothing is written before the first verdict.

        Args:
            log_dir: The folder of the run's verdicts.
            worker: The worker's label, ``rank<rank>``.
            prefix: The file name's prefix: ``VERDICT_PREFIX`` for gate verdicts,
                ``PROMOTION_PREFIX`` for promotion records.
        """
        self.path = Path(log_dir) / f"{prefix}{worker}.jsonl"

    def write(self, verdict: GateVerdict | PromotionRecord) -> dict[str, Any]:
        """Append one verdict or promotion record as one line.

        Args:
            verdict: The verdict or record.

        Returns:
            The record as written.
        """
        record = verdict.to_record()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return record


def _clear(folder: str) -> None:
    # The gate's folder of one puzzle: files an earlier launch left would join this vote.
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                os.remove(path)
    os.makedirs(folder, exist_ok=True)


def _test_index(name: str, key: str) -> int | None:
    # The test input a candidate file of puzzle key belongs to: decode_puzzle names each file by
    # its view's key, "<task>_<test index>.<view>", and the selection groups by the part before
    # the first dot (ArcDecoder.load_decoded_results).
    task, _, index = name.split(".")[0].rpartition("_")
    return int(index) if task == key and index.isdigit() else None


def read_puzzle(
    folder: str | os.PathLike[str], key: str, inputs: int
) -> tuple[tuple[Candidate, ...], ...]:
    """Read one puzzle's candidate files back from a folder, per test input (task P2.3).

    Why: the run's candidate folder holds the files of every puzzle the workers solved; a round
    promotes from round 0's candidates of one puzzle, one entry per test input, as the selection
    groups them by the file name's ``<task>_<test index>`` part.

    Args:
        folder: A candidate folder; a missing folder holds none.
        key: The task id.
        inputs: The puzzle's test inputs.

    Returns:
        Per test input, in order, its candidates in file-name order; other puzzles' files and
        test indices beyond ``inputs`` are skipped.
    """
    found: list[list[Candidate]] = [[] for _ in range(inputs)]
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            index = _test_index(name, key)
            if index is not None and index < inputs:
                found[index].extend(_read_file(folder, name))
    return tuple(tuple(entry) for entry in found)


def install_candidates(
    source: str | os.PathLike[str],
    outputs: str | os.PathLike[str],
    key: str,
    indices: Iterable[int],
    backup: str | os.PathLike[str] | None = None,
) -> tuple[int, ...]:
    """Replace some test inputs' candidate files in the run's folder with a decode's (task P2.3).

    Why: after the rounds, the test inputs that are not pseudo-pairs take the candidates of a
    test decode after them, the final decode or the decode after a kept round (ADR-015); the
    selection reads every file of the run's folder, and an input's files are those named
    ``<task>_<index>.*``. Each new file is copied in before the input's old files that it does
    not overwrite are removed, so a stop in between leaves both, never none; an input the decode
    found no candidate for keeps its files. Before the first change the puzzle's files are copied
    to ``backup``, which keeps round 0's candidates for P2.8.

    Args:
        source: The decode's folder.
        outputs: The run's candidate folder (``outputs_dir``).
        key: The task id.
        indices: The test inputs whose files to replace.
        backup: Where to copy the puzzle's files first, when it does not exist yet; None copies
            nothing.

    Returns:
        The inputs whose files were replaced, in the order given.
    """
    current = [n for n in sorted(os.listdir(outputs)) if _test_index(n, key) is not None]
    if backup is not None and not os.path.isdir(backup):
        os.makedirs(backup)
        for name in current:
            shutil.copyfile(os.path.join(outputs, name), os.path.join(backup, name))
    new = sorted(os.listdir(source)) if os.path.isdir(source) else []
    replaced = []
    for index in indices:
        files = [name for name in new if _test_index(name, key) == index]
        if not files:
            continue
        for name in files:
            shutil.copyfile(os.path.join(source, name), os.path.join(outputs, name))
        for name in current:
            if _test_index(name, key) == index and name not in files:
                os.remove(os.path.join(outputs, name))
        replaced.append(index)
    return tuple(replaced)


def renumber_candidates(folder: str | os.PathLike[str], key: str, inputs: Sequence[int]) -> None:
    """Name a decode's candidate files by the puzzle's test indices (task P2.3).

    Why: ``decode_puzzle`` names each file by its view's key, ``<task>_<j>.<view>``, with ``j``
    the input's position in the task entry it decoded. A decode of some test inputs only (the
    open ones after a kept round) numbers them from 0, so its files are renamed to
    ``<task>_<inputs[j]>.<view>`` before they meet the run's folder or the next round. The
    positions are renamed from the last to the first; since ``inputs`` increases,
    ``inputs[j] >= j``, and no name is taken twice.

    Args:
        folder: The decode's folder.
        key: The task id.
        inputs: The test indices decoded, in the order decoded (increasing).
    """
    names = sorted(os.listdir(folder))
    for position in reversed(range(len(inputs))):
        target = inputs[position]
        if target == position:
            continue
        stem = f"{key}_{position}"
        for name in names:
            if _test_index(name, key) == position:
                renamed = f"{key}_{target}{name[len(stem) :]}"
                os.rename(os.path.join(folder, name), os.path.join(folder, renamed))


class OpWatch:
    """A timing logger that counts and times the ops passing through it (task P2.3).

    Why: ``decode_puzzle`` and ``fit`` report each DFS batch, rescoring, and fit only through
    their ``timing_logger``, and a round needs two of those facts as it runs: how many DFS
    batches a decode ran (fewer than its views need means the 1200 s cap or the end time cut it,
    ``DecodeCut``), and how long the slowest took (the governor's floor for the puzzle's batches,
    ``CostModel.decode_cost``). The watch counts the ops under the caller's names, renames them
    as ``RenamedOps`` does, and hands them to the worker's logger; without one (a run that records
    no timing) it validates each record and keeps nothing else.

    Attributes:
        logger: The worker's logger, or None.
        names: New op name by old one; other ops keep theirs.
        counts: Records per op, by the caller's name.
        slowest: Seconds of the slowest record per op, by the caller's name.
    """

    def __init__(
        self, logger: timing.TimingLogger | None, names: Mapping[str, str] | None = None
    ) -> None:
        """Wrap a logger.

        Args:
            logger: The logger that writes the records, or None to write none.
            names: New op name by old one; None renames nothing.
        """
        self.logger = logger
        self.names = dict(names or {})
        self.counts: dict[str, int] = {}
        self.slowest: dict[str, float] = {}

    def record(
        self, puzzle_id: str | None, op: str, seconds: float, tokens: int | None = None
    ) -> dict[str, object]:
        """Count one op and record it under its new name, as ``TimingLogger.record`` does.

        Args:
            puzzle_id: Task id, or None.
            op: The op's name as the caller records it.
            seconds: Wall time.
            tokens: Tokens processed, or None.

        Returns:
            The record, as written when there is a logger.
        """
        renamed = self.names.get(op, op)
        if self.logger is None:
            entry = timing.make_record(puzzle_id, renamed, seconds, tokens)
        else:
            entry = self.logger.record(puzzle_id, renamed, seconds, tokens)
        self.counts[op] = self.counts.get(op, 0) + 1
        self.slowest[op] = max(self.slowest.get(op, 0.0), float(seconds))
        return entry


@dataclass(frozen=True)
class Decoded:
    """A decode of some test inputs of a puzzle after a round (task P2.3).

    Attributes:
        candidates: One entry per test input of the puzzle, in order: the view candidates the
            decode wrote for it, empty for an input it did not decode.
        batches: DFS batches that ran.
        expected: DFS batches its views need: 4 per input decoded.
        slowest: Seconds of its slowest DFS batch; None when none ran.
    """

    candidates: tuple[tuple[Candidate, ...], ...]
    batches: int
    expected: int
    slowest: float | None = None

    @property
    def cut(self) -> bool:
        """Whether the time cap stopped it before all its batches ran."""
        return self.batches < self.expected


def context_fit(lengths: Sequence[int], limit: int) -> int:
    r"""Return how many pseudo-pairs, in order, a prompt holds without a real pair lost (P2.5).

    Why: the reference's ``ArcDataset.cut_to_len`` drops a prompt's leading demonstrations until
    its tokens are at most ``limit`` (``max_seq_length - max_new_tokens``), and the views shuffle
    the demonstrations (``shuffle_ex``), so a pseudo-pair that pushes a prompt past the limit can
    cost it a real pair. The guard (ADR-016) keeps the first :math:`k` pseudo-pairs with
    :math:`L_j \le` ``limit`` for every :math:`j \le k`, :math:`L_j` the longest prompt with the
    first :math:`j`: the others are dropped, latest first, and ``cut_to_len`` then cuts nothing.
    A prompt over the limit without any pseudo-pair (the reference cuts real pairs from it
    itself) holds none, since each pseudo-pair would cost one more pair.

    Args:
        lengths: :math:`L_0, \dots, L_n`: the tokens of the longest prompt (over the views,
            the rescoring's included, and the queries) holding the first :math:`j` pseudo-pairs.
        limit: The tokens a prompt may have, ``max_seq_length - max_new_tokens``.

    Returns:
        :math:`k`, from 0 to :math:`n`.

    Raises:
        TypeError: If a length or ``limit`` is not an integer.
        ValueError: If ``lengths`` is empty, a length is negative, or ``limit`` is under 1.
    """
    if isinstance(lengths, str | bytes) or not isinstance(lengths, Sequence):
        raise TypeError(f"lengths must be a sequence of token counts; got {lengths!r}")
    found = tuple(lengths)
    if not found:
        raise ValueError("lengths needs L_0, the prompt without a pseudo-pair")
    if not all(type(length) is int for length in found) or type(limit) is not int:
        raise TypeError(f"lengths and limit must be integers; got {found!r} and {limit!r}")
    if any(length < 0 for length in found):
        raise ValueError(f"lengths must be at least 0; got {found!r}")
    if limit < 1:
        raise ValueError(f"limit must be at least 1; got {limit}")
    if found[0] > limit:
        return 0
    kept = 0
    for count in range(1, len(found)):
        if found[count] > limit:
            break
        kept = count
    return kept


def longest_prompt(entry: Mapping[str, Any], formatter: QwenFormatter) -> int:
    """Return the tokens of the longest prompt that ``decode_puzzle`` and its rescoring build.

    Why: the length guard (task P2.5) must measure what ``cut_to_len`` measures: the prompt's
    ``input`` (the demonstrations and the query) under the model's tokenizer. Every view of a
    decode (16) and of its rescoring (8) is a dihedral transform, a color permutation, and an
    order of the demonstrations. Each color is one token and each turn sits between special
    tokens, so a view's length depends only on whether its transform exchanges rows and columns;
    the moved ``ArcDataset.get_length(..., max_of_transposed=True)`` takes the longer of the two.
    numpy's random state is not touched (no augmentation is drawn).

    Args:
        entry: A task entry, ``{"train": [...], "test": [...]}``, as ``DemoSet.task`` builds it.
        formatter: The model's formatter (``QwenFormatter``, with its tokenizer).

    Returns:
        The longest prompt's tokens over the entry's test inputs and both orientations.
    """
    from arc26.nvarc.arc_loader import ArcDataset

    multi = ArcDataset(queries={"prompt": entry}).split_multi_replies()
    return max(
        multi.get_length(key, formatter, "input", max_of_transposed=True) for key in multi.keys
    )


@dataclass(frozen=True)
class PromptFit:
    """The pseudo-pairs one decode's prompts hold under the length guard (task P2.5).

    Attributes:
        kept: The test indices of the pseudo-pairs held, the first ones in order.
        dropped: The test indices of the pseudo-pairs left out, in order: the latest added last.
        lengths: The longest prompt's tokens with the first 0 to all pseudo-pairs.
        limit: The token limit.
    """

    kept: tuple[int | None, ...]
    dropped: tuple[int | None, ...]
    lengths: tuple[int, ...]
    limit: int


@dataclass(frozen=True)
class PromptGuard:
    """The length guard of every prompt that carries pseudo-pairs (task P2.5, ADR-016).

    Why: B5 P2.5's "no unbounded appends". A pseudo-pair joins a prompt only where the prompt
    still fits the reference's limit, so ``cut_to_len`` never drops a real pair, or an earlier
    pseudo-pair, for it (``context_fit``). Applied by ``refine`` (the in-context round's choice
    of pairs, the re-decodes', and the gate's prompts) and ``run_rounds`` (the test decodes
    after the rounds); the decode callables then receive the state as the prompts hold it
    (``DemoSet.first_pseudo``).

    Attributes:
        limit: The tokens a prompt may have: ``max_seq_length - max_new_tokens`` (8192 minus a
            30 by 30 reply), what ``decode_puzzle`` passes to ``cut_to_len``.
        measure: The tokens of the longest prompt of a task entry (``longest_prompt`` with the
            model's formatter; the tests pass a count without a tokenizer).
    """

    limit: int
    measure: Callable[[Mapping[str, Any]], int] = field(compare=False)

    def __post_init__(self) -> None:
        if type(self.limit) is not int:
            raise TypeError(f"limit must be an integer; got {self.limit!r}")
        if self.limit < 1:
            raise ValueError(f"limit must be at least 1; got {self.limit}")
        if not callable(self.measure):
            raise TypeError(f"measure must be callable; got {self.measure!r}")

    def tokens(self, entry: Mapping[str, Any]) -> int:
        """Measure one task entry's longest prompt.

        Args:
            entry: The task entry.

        Returns:
            Its tokens.

        Raises:
            TypeError: If the measure returns something other than an integer.
            ValueError: If it returns less than 1.
        """
        tokens = self.measure(entry)
        if type(tokens) is not int:
            raise TypeError(f"the measure must return an integer; got {tokens!r}")
        if tokens < 1:
            raise ValueError(f"a prompt has at least one token; the measure returned {tokens}")
        return tokens

    def fit(self, demos: DemoSet, queries: Sequence[Grid], *, held_out: bool = False) -> PromptFit:
        """Choose the pseudo-pairs a decode's prompts hold.

        Args:
            demos: The state the decode would prompt with.
            queries: The inputs it decodes, at least one.
            held_out: Whether its prompts put the held-out pair back (the final decode).

        Returns:
            The pairs held and dropped, and the lengths measured.
        """
        pseudo = demos.pseudo_pairs
        lengths = tuple(
            self.tokens(demos.first_pseudo(count).task(queries, held_out=held_out))
            for count in range(len(pseudo) + 1)
        )
        kept = context_fit(lengths, self.limit)
        return PromptFit(
            kept=tuple(pair.test_index for pair in pseudo[:kept]),
            dropped=tuple(pair.test_index for pair in pseudo[kept:]),
            lengths=lengths,
            limit=self.limit,
        )


class AdapterProvider(Protocol):
    """B4's ``AdapterProvider``, with what a round needs besides ``adapt`` (task P2.3, ADR-015).

    Phase 2's provider is ``arc26.ttt.SGDProvider`` (``sgd_continue`` from a snapshot); Phase 4's
    will build new adapters from H (ADR-004). The tests use CPU stand-ins.
    """

    def adapt(self, demos: DemoSet, init: Any, budget_s: float) -> Any:
        """Return the adapter for ``demos``, started from ``init`` (B4).

        Args:
            demos: The enlarged state, without its held-out pair.
            init: The current adapter.
            budget_s: Seconds left to the puzzle's deadline when the fit starts.

        Returns:
            The candidate adapter.
        """

    def rollback(self, previous: Any) -> Any:
        """Make ``previous``, the adapter the last ``adapt`` started from, the live one again.

        Args:
            previous: The adapter ``adapt`` started from.

        Returns:
            The adapter to use from here on: ``previous`` itself for a provider that builds new
            adapters (a pointer swap, ADR-004); for one that trains in place, the model with
            its previous weights restored.
        """

    def fingerprint(self, adapter: Any) -> str | None:
        """Return a digest of the adapter's weights for the round's record, or None.

        Args:
            adapter: An adapter this provider returned or started from.

        Returns:
            Equal digests exactly for bit-for-bit equal weights.
        """

    def fit_shape(self, demos: DemoSet) -> tuple[int, int]:
        """Return the tokens and optimizer steps of ``adapt`` on ``demos``, for the governor.

        Args:
            demos: The state ``adapt`` would train on, without its held-out pair.

        Returns:
            ``(tokens, steps)``: the tokens the fit trains on and its optimizer steps, the train
            term of ``CostModel.round_costs``.
        """


class InContextProvider:
    """B4's ``AdapterProvider`` for the in-context round: no fit (task P2.5, B2 Q4, ADR-016).

    Why: B2 Q4's fallback appends the promoted pairs to the decode prompt with no gradient step.
    Every decode after a round already builds its prompts from ``DemoSet.pairs``, so a kept
    round's pseudo-pairs reach the prompts by being in the state; what the in-context round
    leaves out is the fit. ``adapt`` therefore returns the current adapter as the candidate,
    and the gate checks it with the enlarged set in its prompts; ``rollback`` has nothing to
    restore, so a failed gate rolls back the state's version alone.

    Attributes:
        digest: The fingerprint function, as the gradient provider's (``SGDProvider
            .fingerprint``), so the round's record shows the adapter unchanged; None records
            no fingerprint.
    """

    def __init__(self, fingerprint: Callable[[Any], str | None] | None = None) -> None:
        """Build the provider.

        Args:
            fingerprint: Digests an adapter's weights, as the gradient provider's; None for no
                digest.

        Raises:
            TypeError: If ``fingerprint`` is neither None nor callable.
        """
        if fingerprint is not None and not callable(fingerprint):
            raise TypeError(f"fingerprint must be None or callable; got {fingerprint!r}")
        self.digest = fingerprint

    def adapt(self, demos: DemoSet, init: Any, budget_s: float) -> Any:
        """Return ``init`` unchanged: the in-context round fits nothing.

        Args:
            demos: The enlarged state, without its held-out pair (unused: its pairs reach the
                prompts through the state).
            init: The current adapter.
            budget_s: Seconds left to the puzzle's deadline (unused).

        Returns:
            ``init`` itself.
        """
        return init

    def rollback(self, previous: Any) -> Any:
        """Return ``previous``: nothing was changed, so nothing is restored.

        Args:
            previous: The adapter ``adapt`` returned unchanged.

        Returns:
            ``previous`` itself.
        """
        return previous

    def fingerprint(self, adapter: Any) -> str | None:
        """Digest the adapter as the gradient provider does, or None without a digest function.

        Args:
            adapter: The adapter.

        Returns:
            The digest, or None.
        """
        return None if self.digest is None else self.digest(adapter)

    def fit_shape(self, demos: DemoSet) -> tuple[int, int]:
        """Return ``(0, 0)``: no token is trained, in no step.

        Args:
            demos: The state (unused).

        Returns:
            ``(0, 0)``.
        """
        return 0, 0


@dataclass(frozen=True)
class RoundRecord:
    """One round of one puzzle, one JSON line in ``round-<worker>.jsonl`` (task P2.3).

    Attributes:
        puzzle_id: The task id.
        round_added: The round, 1 to ``MAX_ROUNDS``.
        version: The state's version when the round started.
        next_version: Its version after the round: ``version + 1`` when kept, else ``version``.
        outcome: How the round ended, one of ``OUTCOMES``.
        screened: The test inputs that passed the free checks (``screen``).
        promoted: The test inputs whose pseudo-pairs the round added.
        decision: The governor's decision, ``gradient``, ``in_context`` (P2.5), or ``none``;
            None without a plan.
        estimate: The plan's seconds for the variant decided, margin included, or for the first
            variant offered when none was (the gradient round's when offered); None without a
            plan.
        remaining: Seconds left to the puzzle's deadline at the plan; None without one.
        weights_before: The provider's fingerprint of the adapter the candidate fit started
            from; None when no fit ran (or the in-context provider has no fingerprint).
        weights_after: The fingerprint after the round: the kept adapter's, or the restored
            one's after a rollback; None when no fit ran. An in-context round's two are equal:
            it changes no weight.
        restored: After a rollback, whether the restored weights equal the previous ones bit
            for bit (equal fingerprints); None otherwise, and after an in-context round's
            rollback without fingerprints (its adapter was never changed).
        seconds: Wall time of the round.
        error: Why the round raised, ``"<type>: <message>"``; None otherwise.
    """

    puzzle_id: str
    round_added: int
    version: int
    next_version: int
    outcome: str
    screened: tuple[int, ...] = ()
    promoted: tuple[int, ...] = ()
    decision: str | None = None
    estimate: float | None = None
    remaining: float | None = None
    weights_before: str | None = None
    weights_after: str | None = None
    restored: bool | None = None
    seconds: float = 0.0
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the record as a JSON-ready dict.

        Returns:
            Every field; seconds rounded to milliseconds.
        """
        return {
            "puzzle_id": self.puzzle_id,
            "round": self.round_added,
            "version": self.version,
            "next_version": self.next_version,
            "outcome": self.outcome,
            "screened": list(self.screened),
            "promoted": list(self.promoted),
            "decision": self.decision,
            "estimate": None if self.estimate is None else round(self.estimate, 3),
            "remaining": None if self.remaining is None else round(self.remaining, 3),
            "weights_before": self.weights_before,
            "weights_after": self.weights_after,
            "restored": self.restored,
            "seconds": round(self.seconds, 3),
            "error": self.error,
        }


@dataclass(frozen=True)
class DecodeRecord:
    """One test decode after the rounds of one puzzle, one line in ``decode-<worker>.jsonl``.

    Attributes:
        puzzle_id: The task id.
        round_added: The last round before it.
        name: Its folder in the puzzle's round folder: ``round<r>`` after kept round ``r``, or
            ``final``.
        prompts: ``pairs`` (``DemoSet.pairs``) or ``held_out`` (the held-out pair put back).
        inputs: The test inputs whose candidates it may replace: those that are not
            pseudo-pairs.
        installed: The inputs whose candidates in the run's folder it replaced.
        batches: DFS batches that ran; None when it did not run.
        expected: DFS batches its views need.
        cut: Whether the time cap stopped it, so that nothing was replaced.
        estimate: The final decode's seconds, its own time check; None for a round's decode.
        remaining: Seconds left to the puzzle's deadline at that check; None for a round's.
        skipped: Why it did not run (``ALL_PROMOTED``, ``NOT_RESTORED``, ``NO_TIME``); None
            when it ran.
        seconds: Wall time.
        error: Why it raised, ``"<type>: <message>"``; None otherwise.
    """

    puzzle_id: str
    round_added: int
    name: str
    prompts: str
    inputs: tuple[int, ...]
    expected: int
    installed: tuple[int, ...] = ()
    batches: int | None = None
    cut: bool = False
    estimate: float | None = None
    remaining: float | None = None
    skipped: str | None = None
    seconds: float = 0.0
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return the record as a JSON-ready dict.

        Returns:
            Every field; seconds rounded to milliseconds.
        """
        return {
            "puzzle_id": self.puzzle_id,
            "round": self.round_added,
            "name": self.name,
            "prompts": self.prompts,
            "inputs": list(self.inputs),
            "installed": list(self.installed),
            "batches": self.batches,
            "expected": self.expected,
            "cut": self.cut,
            "estimate": None if self.estimate is None else round(self.estimate, 3),
            "remaining": None if self.remaining is None else round(self.remaining, 3),
            "skipped": self.skipped,
            "seconds": round(self.seconds, 3),
            "error": self.error,
        }


@dataclass(frozen=True)
class ContextRecord:
    """A decode whose prompts left out pseudo-pairs, one line in ``context-<worker>.jsonl``.

    Why: the length guard (task P2.5, ADR-016) changes what a decode sees, so every decode it
    trimmed is recorded with the pseudo-pairs it left out and the lengths that decided it. A
    decode whose prompts hold every pseudo-pair writes no line, so a run where nothing was
    dropped writes the records of P2.3.

    Attributes:
        puzzle_id: The task id.
        round_added: The round the decode belongs to, or that it follows.
        name: The decode, as its folder in the puzzle's round folder: ``round<r>-redecode<i>``,
            ``round<r>-gate``, ``round<r>``, or ``final``.
        prompts: ``pairs`` (``DemoSet.pairs``) or ``held_out`` (the held-out pair put back).
        queries: The inputs it decodes: the test indices, or None for the held-out input.
        kept: The test indices of the pseudo-pairs its prompts hold, in order.
        dropped: The test indices of those it left out, in order (the latest added last).
        lengths: The longest prompt's tokens with the first 0 to all of the state's pseudo-pairs.
        limit: The token limit, ``max_seq_length - max_new_tokens``.
    """

    puzzle_id: str
    round_added: int
    name: str
    prompts: str
    queries: tuple[int, ...] | None
    kept: tuple[int | None, ...]
    dropped: tuple[int | None, ...]
    lengths: tuple[int, ...]
    limit: int

    def to_record(self) -> dict[str, Any]:
        """Return the record as a JSON-ready dict.

        Returns:
            Every field.
        """
        return {
            "puzzle_id": self.puzzle_id,
            "round": self.round_added,
            "name": self.name,
            "prompts": self.prompts,
            "queries": None if self.queries is None else list(self.queries),
            "kept": list(self.kept),
            "dropped": list(self.dropped),
            "lengths": list(self.lengths),
            "limit": self.limit,
        }


def _append_line(path: Path, record: Mapping[str, Any]) -> None:
    # One JSON line appended and the file closed at once: a hard kill loses at most this line.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


class RoundLog:
    """The round records of one worker (task P2.3): one JSONL file per kind, beside the verdicts.

    Why: the rounds' decisions must be readable after the run without the session log: the
    governor's plans (``plan-<worker>.jsonl``, each ``RoundPlan`` with its round and the shape it
    priced), the promotion records (``promote-<worker>.jsonl``, P2.2's), the rounds with the
    fingerprints that show a rollback restored the weights (``round-<worker>.jsonl``), the
    test decodes after them (``decode-<worker>.jsonl``), and the decodes whose prompts left out
    pseudo-pairs (``context-<worker>.jsonl``, P2.5's length guard). One file per worker process,
    each line appended and the file closed at once, as ``VerdictLog`` does. Every record is also
    kept in memory, and ``say`` prints a one-line summary of each to the session log.

    Attributes:
        log_dir: The folder of the run's verdicts, or None to keep the records in memory only.
        worker: The worker's label, ``rank<rank>``.
        plans: The plan records written, in order.
        promotions: The promotion records written, in order.
        rounds: The round records written, in order.
        decodes: The decode records written, in order.
        contexts: The context records written, in order (P2.5).
    """

    def __init__(
        self,
        log_dir: str | os.PathLike[str] | None,
        worker: str,
        say: Callable[[str], None] | None = None,
    ) -> None:
        """Create a log; nothing is written before the first record.

        Args:
            log_dir: The folder of the run's verdicts (``HoldOut.log_dir``), or None.
            worker: The worker's label, ``rank<rank>``.
            say: Prints a summary line of each record, for example to the session log; None
                prints nothing.
        """
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.worker = worker
        self._say = say
        self.plans: list[dict[str, Any]] = []
        self.promotions: list[PromotionRecord] = []
        self.rounds: list[RoundRecord] = []
        self.decodes: list[DecodeRecord] = []
        self.contexts: list[ContextRecord] = []

    def path(self, prefix: str) -> Path | None:
        """Return the worker's file of one kind of record.

        Args:
            prefix: ``PLAN_PREFIX``, ``PROMOTION_PREFIX``, ``ROUND_PREFIX``, ``DECODE_PREFIX``,
                or ``CONTEXT_PREFIX``.

        Returns:
            ``<log_dir>/<prefix><worker>.jsonl``, or None without a folder.
        """
        return None if self.log_dir is None else self.log_dir / f"{prefix}{self.worker}.jsonl"

    def _write(self, prefix: str, record: Mapping[str, Any], line: str) -> None:
        path = self.path(prefix)
        if path is not None:
            _append_line(path, record)
        if self._say is not None:
            self._say(line)

    def plan(self, round_added: int, plan: RoundPlan, shape: Mapping[str, Any]) -> dict[str, Any]:
        """Record the governor's plan of one round.

        Args:
            round_added: The round.
            plan: The governor's plan.
            shape: What it priced: the screened inputs, the fit's tokens and steps, the decode
                batches, and the observed batch seconds.

        Returns:
            The record as written: the round, the plan's fields, and the shape.
        """
        record = {"round": round_added, **plan.to_record(), "shape": dict(shape)}
        self.plans.append(record)
        # The decided variant's estimate, or the first offered one's (the ladder's order).
        estimate = plan.estimates.get(plan.decision, next(iter(plan.estimates.values()), 0.0))
        self._write(
            PLAN_PREFIX,
            record,
            f"round {round_added} for {plan.puzzle_id}: plan {plan.decision}, estimated "
            f"{estimate:.0f} s, {plan.remaining:.0f} s left of the puzzle's budget",
        )
        return record

    def promotion(self, record: PromotionRecord) -> dict[str, Any]:
        """Record one test input's promotion record.

        Args:
            record: The record.

        Returns:
            The record as written.
        """
        payload = record.to_record()
        self.promotions.append(record)
        self._write(
            PROMOTION_PREFIX,
            payload,
            f"round {record.round_added} for {record.puzzle_id}: test input {record.test_index} "
            f"{record.reason}",
        )
        return payload

    def round(self, record: RoundRecord) -> dict[str, Any]:
        """Record one round.

        Args:
            record: The record.

        Returns:
            The record as written.
        """
        payload = record.to_record()
        self.rounds.append(record)
        detail = ""
        if record.outcome == KEPT:
            detail = f", version {record.next_version}, promoted {list(record.promoted)}"
        elif record.restored is not None:
            detail = f", weights {'restored' if record.restored else 'NOT restored'}"
        if record.error is not None:
            detail += f", {record.error}"
        self._write(
            ROUND_PREFIX,
            payload,
            f"round {record.round_added} for {record.puzzle_id}: {record.outcome}{detail}",
        )
        return payload

    def decode(self, record: DecodeRecord) -> dict[str, Any]:
        """Record one test decode after the rounds.

        Args:
            record: The record.

        Returns:
            The record as written.
        """
        payload = record.to_record()
        self.decodes.append(record)
        if record.skipped is not None:
            state = f"skipped, {record.skipped}"
        elif record.error is not None:
            state = f"raised {record.error}"
        else:
            state = (
                f"{record.batches} of {record.expected} DFS batches"
                + (", cut by the time cap" if record.cut else "")
                + f", replaced the candidates of test inputs {list(record.installed)}"
            )
        self._write(
            DECODE_PREFIX,
            payload,
            f"test decode {record.name} for {record.puzzle_id} ({record.prompts} prompts): {state}",
        )
        return payload

    def context(self, record: ContextRecord) -> dict[str, Any]:
        """Record one decode whose prompts left out pseudo-pairs (the length guard, P2.5).

        Args:
            record: The record.

        Returns:
            The record as written.
        """
        payload = record.to_record()
        self.contexts.append(record)
        self._write(
            CONTEXT_PREFIX,
            payload,
            f"round {record.round_added} for {record.puzzle_id}: the prompts of {record.name} "
            f"left out the pseudo-pairs of test inputs {list(record.dropped)} "
            f"({record.lengths[-1]} tokens with every one, limit {record.limit})",
        )
        return payload


def _refine_args(
    provider: object,
    gate: object,
    governor: object,
    redecode: object,
    deadline: object,
    budget: object,
    observed_batch: object,
    clock: object,
) -> None:
    _provider_args("provider", provider)
    if not callable(getattr(gate, "verdict", None)):
        raise TypeError(f"gate must have a verdict method (HeldOutGate); got {gate!r}")
    if not isinstance(governor, Governor):
        raise TypeError(f"governor must be a Governor; got {governor!r}")
    if not callable(redecode) or not callable(clock):
        raise TypeError("redecode and clock must be callable")
    for name, value in (("deadline", deadline), ("budget", budget)):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number; got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
    if budget < 0:  # type: ignore[operator]
        raise ValueError(f"budget must be at least 0; got {budget!r}")
    if observed_batch is not None:
        if isinstance(observed_batch, bool) or not isinstance(observed_batch, int | float):
            raise TypeError(f"observed_batch must be None or a number; got {observed_batch!r}")
        if not math.isfinite(observed_batch) or observed_batch < 0:
            raise ValueError(f"observed_batch must be finite and at least 0; got {observed_batch}")


def _provider_args(name: str, provider: object) -> None:
    for method in ("adapt", "rollback", "fingerprint", "fit_shape"):
        if not callable(getattr(provider, method, None)):
            raise TypeError(f"{name} must have a {method} method (AdapterProvider)")


def _offered(variants: object, in_context: object, guard: object) -> tuple[str, ...]:
    # The variants refine offers (P2.5): by default the gradient round, and the in-context round
    # too when its provider is given; variants restricts them. The in-context round needs its
    # provider and the length guard: no unbounded appends (B5 P2.5).
    if variants is None:
        offered = DEFAULT_VARIANTS if in_context is None else LADDER
    else:
        offered = _variants(variants)
    if IN_CONTEXT in offered:
        if in_context is None:
            raise ValueError("variants offer the in-context round: refine needs its provider")
        _provider_args("in_context", in_context)
        if not isinstance(guard, PromptGuard):
            raise TypeError(
                f"the in-context round needs the length guard (PromptGuard); got {guard!r}"
            )
    elif guard is not None and not isinstance(guard, PromptGuard):
        raise TypeError(f"guard must be None or a PromptGuard; got {guard!r}")
    return offered


def _in_context_fit(
    guard: PromptGuard, demos: DemoSet, drafts: Sequence[Pair], held_input: Grid
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    # The in-context round's pairs (P2.5, ADR-016): the drafts its gate's prompts hold (the
    # length guard, latest dropped first, earlier rounds' pseudo-pairs kept before this round's),
    # those they cannot, and the prompts' length ratio rho for the governor's price: the largest
    # over the gate's query and the open test inputs of the longest prompt with the pairs held
    # over the longest without them.
    fit = guard.fit(demos.enlarged(drafts), [held_input])
    fitting = tuple(pair.test_index for pair in drafts if pair.test_index in fit.kept)
    too_long = tuple(pair.test_index for pair in drafts if pair.test_index not in fit.kept)
    if not fitting:
        return (), too_long, 1.0  # type: ignore[return-value]
    held = demos.enlarged([pair for pair in drafts if pair.test_index in fitting])
    queries = [held_input] + [demos.test_inputs[index] for index in _open_inputs(demos)]
    ratio = max(
        guard.tokens(held.task([query])) / guard.tokens(demos.task([query])) for query in queries
    )
    return fitting, too_long, round(max(1.0, ratio), 6)  # type: ignore[return-value]


def refine(
    demos: DemoSet,
    adapter: Any,
    provider: AdapterProvider,
    gate: HeldOutGate,
    governor: Governor,
    *,
    candidates: Sequence[Iterable[Candidate]],
    redecode: Redecode,
    deadline: float,
    budget: float,
    observed_batch: float | None = None,
    round_added: int = 1,
    tau: float = TAU,
    fresh_seed: int = FRESH_SEED,
    first_seed: int | None = None,
    views: int = DECODE_VIEWS,
    tries: int = VIEW_SEED_TRIES,
    first_inputs: Sequence[int] | None = None,
    decode_batches: int | None = None,
    in_context: AdapterProvider | None = None,
    variants: Sequence[str] | None = None,
    guard: PromptGuard | None = None,
    log: RoundLog | None = None,
    timing_logger: timing.TimingLogger | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[DemoSet, Any]:
    r"""B4's ``refine``: one round of promotion, kept or rolled back (tasks P2.3, P2.5).

    Why: B5 P2.3: "one round = promote, enlarge ``DemoSet`` (version + 1),
    ``provider.adapt(demos, init=previous, budget)``, gate check, keep or roll back (previous
    adapter and previous ``DemoSet`` version)". Before any GPU work the round is priced and the
    governor may decline it (B5 P2.4): the free checks (``screen``) bound it, each input that
    passes them costing a re-decode of 4 DFS batches and joining the candidate fit, priced at
    ``provider.fit_shape`` of the state with all of them; the gate decodes 4 batches, and the
    test decodes that follow the round (``decode_batches``; by default the decode after a kept
    round, 4 per input still open after it, at most one fewer than are open now) are priced with
    it (``CostModel.round_costs``). The free checks' records are written whatever follows, a
    declined round's with the reason ``declined`` for the inputs that passed them, so that every
    voted answer's agreement is logged (ADR-014). With ``gradient`` the round runs ``promote``
    with the provider's ``adapt`` as the candidate fit, from ``adapter``, with the time left to
    the deadline as its budget: Phase 2's :math:`\theta_t = \mathrm{sgd\_continue}(\theta_{t-1},
    D_t, s')`. A kept round returns the enlarged state and the candidate adapter. Otherwise the
    round returns ``demos`` itself and the previous adapter, after ``provider.rollback`` when the
    candidate fit ran: a provider that trains in place then holds the previous weights again,
    which the record shows with the fingerprints before the fit and after the restore. A round
    that raises outside ``promote``'s own checks is rolled back the same way and recorded with
    its error, so the worker goes on.

    The in-context round (task P2.5, ADR-016, B2 Q4) is offered beside it when ``in_context`` is
    given (``variants`` forces either): its provider returns ``adapter`` unchanged, so the
    pseudo-pairs reach only the prompts, and the gate decodes with them there. Its pairs are the
    inputs that passed the free checks whose pseudo-pairs the gate's prompts can hold (the
    length guard, ``guard``; the others are recorded ``too_long`` and not re-decoded), and when
    none can, it is not offered. The governor prices it without a fit, its gate and test decodes
    at the prompts' length ratio (``CostModel.round_costs``' ``in_context``), and picks it when
    the gradient round does not fit (its ladder, B2 Q4). Kept: the enlarged state and the same
    adapter. A failed gate: ``demos`` itself and the same adapter, the version rolled back
    alone. With ``guard``, every re-decode's and gate's prompts hold the pseudo-pairs that fit,
    whatever the variant, each decode that drops one recorded (``ContextRecord``).

    Args:
        demos: The puzzle's state, with its held-out pair; without one no round runs.
        adapter: The current adapter, which produced ``candidates``.
        provider: The gradient round's provider: fits the candidate adapter and rolls it back
            (``AdapterProvider``).
        gate: The held-out gate whose ``verdict`` checks the candidate adapter.
        governor: The worker's governor; its cost model prices the round and ``plan`` decides.
        candidates: Per test input of ``demos``, the view candidates of the decode that produced
            the voted answers (``promote``'s).
        redecode: ``promote``'s re-decode callable.
        deadline: The puzzle's deadline on the governor's clock: its start plus its budget.
        budget: The puzzle's budget in seconds (``Governor.puzzle_budget``), for the record.
        observed_batch: Seconds of the slowest DFS batch this puzzle ran, or None.
        round_added: The round, 1 to ``MAX_ROUNDS``; the pseudo-pairs carry it.
        tau: The agreement threshold.
        fresh_seed: The first view seed a re-decode tries.
        first_seed: The ``view_seed`` of the decode that produced ``candidates``.
        views: :math:`n`, the views each input was decoded under (16).
        tries: View seeds tried per input before giving up.
        first_inputs: The test inputs the decode that produced ``candidates`` decoded together;
            None for every input (``fresh_views``).
        decode_batches: DFS batches of the test decodes that may follow the round, at most,
            which its plan reserves: ``run_rounds`` counts the final decode with the held-out
            pair here too (``Rounds.final_in_prompts``); None prices the decode after a kept
            round alone.
        in_context: The in-context round's provider (``InContextProvider``, task P2.5), or None,
            the default: the round is not offered, as before P2.5.
        variants: The variants offered, in the governor's order (``Rounds.variants``); None
            offers the gradient round, and the in-context one too when ``in_context`` is given.
        guard: The length guard (``PromptGuard``, task P2.5) of the prompts that carry
            pseudo-pairs; the in-context round needs it. None leaves the prompts as built.
        log: Receives the plan, the promotion records, the round record, and the context
            records; None keeps them nowhere.
        timing_logger: Records ``promote``'s spans and the round's own, ``round``, for a round
            whose promotion ran to its end.
        clock: Seconds for the spans and the round's wall time.

    Returns:
        ``(state, adapter)``: the next version and the candidate adapter (the same adapter for
        an in-context round) when the round was kept; otherwise ``demos`` itself and the previous
        adapter, restored when the fit had trained it in place.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range, or ``variants`` offers the in-context round
            without its provider.
    """
    _refine_args(provider, gate, governor, redecode, deadline, budget, observed_batch, clock)
    offered = _offered(variants, in_context, guard)
    found = _screen_args(
        candidates, demos, tau, round_added, views, first_seed, fresh_seed, tries, first_inputs
    )
    if decode_batches is not None:
        if type(decode_batches) is not int:
            raise TypeError(f"decode_batches must be None or an integer; got {decode_batches!r}")
        if decode_batches < 0:
            raise ValueError(f"decode_batches must be at least 0; got {decode_batches}")
    if round_added > MAX_ROUNDS:
        raise ValueError(f"round_added must be at most {MAX_ROUNDS} (B5 P2.3); got {round_added}")
    log = log if log is not None else RoundLog(None, "")
    start = clock()
    record = RoundRecord(
        puzzle_id=demos.puzzle_id,
        round_added=round_added,
        version=demos.version,
        next_version=demos.version,
        outcome=NO_HELD_OUT,
    )
    # The free checks run first, on the CPU, and their records are written whatever follows.
    screened = _screen(
        demos, found, tau, views, round_added, first_seed, fresh_seed, tries, first_inputs
    )
    passing = tuple(item.index for item in screened if item.passed)
    if demos.held_out is None or not passing:
        for item in screened:
            log.promotion(item.record)
        outcome = NO_HELD_OUT if demos.held_out is None else NOTHING_TO_PROMOTE
        log.round(replace(record, outcome=outcome, seconds=clock() - start))
        return demos, adapter
    record = replace(record, screened=passing)
    fit: dict[str, Any] = {"ran": False, "before": None}
    current = adapter
    chosen: Any = provider
    try:
        drafts = [
            Pair(
                demos.test_inputs[item.index],
                item.tally.voted,
                kind=PSEUDO,
                round_added=round_added,
                agreement=item.tally.agreement,
                test_index=item.index,
            )
            for item in screened
            if item.passed
        ]
        per_input = -(-views // VIEWS_PER_BATCH)
        # The decode after a kept round decodes the inputs still open after it: at least one
        # screened input is then a pseudo-pair, so at most one fewer than are open now.
        still_open = max(0, len(_open_inputs(demos)) - 1)
        follow = per_input * still_open if decode_batches is None else decode_batches
        costs: dict[str, float] = {}
        if GRADIENT in offered:
            train_tokens, train_steps = provider.fit_shape(
                demos.enlarged(drafts).without_held_out()
            )
            shape: dict[str, Any] = {
                "screened": list(passing),
                "train_tokens": train_tokens,
                "train_steps": train_steps,
                "test_batches": follow,
                "gate_batches": per_input,
                "redecode_batches": per_input * len(passing),
                "observed_batch": observed_batch,
            }
            costs[GRADIENT] = governor.cost_model.round_costs(
                train_tokens=train_tokens,
                train_steps=train_steps,
                test_batches=shape["test_batches"],
                gate_batches=shape["gate_batches"],
                redecode_batches=shape["redecode_batches"],
                observed_batch=observed_batch,
            )[GRADIENT]
        else:
            shape = {
                "screened": list(passing),
                "test_batches": follow,
                "gate_batches": per_input,
                "observed_batch": observed_batch,
            }
        fitting, too_long = passing, ()
        if IN_CONTEXT in offered:
            # The in-context round: no fit; its pairs are those its gate's prompts can hold.
            fitting, too_long, ratio = _in_context_fit(
                guard,
                demos,
                drafts,
                demos.held_out.input,  # type: ignore[arg-type]
            )
            shape[IN_CONTEXT] = {
                "screened": list(fitting),
                "too_long": list(too_long),
                "redecode_batches": per_input * len(fitting),
                "length_ratio": ratio,
            }
            if fitting:
                costs[IN_CONTEXT] = governor.cost_model.round_costs(
                    train_tokens=0,
                    test_batches=follow,
                    gate_batches=per_input,
                    redecode_batches=per_input * len(fitting),
                    length_ratio=ratio,
                    observed_batch=observed_batch,
                )[IN_CONTEXT]

        def reasons(fits: bool, reason: str) -> list[PromotionRecord]:
            # The records of a round that ran no promotion: an input that passed the free
            # checks takes the reason, or too_long when only the in-context round could have
            # promoted it and its pair does not fit.
            return [
                item.record
                if not item.passed
                else replace(
                    item.record,
                    reason=TOO_LONG if not fits and item.index in too_long else reason,
                )
                for item in screened
            ]

        if not costs:
            # Only the in-context round was offered, and none of its pairs fits: no plan.
            for promotion_record in reasons(False, TOO_LONG):
                log.promotion(promotion_record)
            log.round(replace(record, outcome=TOO_LONG, seconds=clock() - start))
            return demos, adapter
        plan = governor.plan(
            costs,
            deadline,
            budget=budget,
            observed_batch=observed_batch,
            puzzle_id=demos.puzzle_id,
        )
        log.plan(round_added, plan, shape)
        record = replace(
            record,
            decision=plan.decision,
            estimate=costs.get(plan.decision, next(iter(costs.values()))),
            remaining=plan.remaining,
        )
        if plan.decision not in costs:
            for promotion_record in reasons(GRADIENT in costs, DECLINED):
                log.promotion(promotion_record)
            log.round(replace(record, outcome=DECLINED, seconds=clock() - start))
            return demos, adapter
        if plan.decision == IN_CONTEXT:
            chosen = in_context
            screened = tuple(
                replace(item, passed=False, record=replace(item.record, reason=TOO_LONG))
                if item.passed and item.index not in fitting
                else item
                for item in screened
            )

        def adapt(enlarged: DemoSet, init: Any) -> Any:
            # The candidate fit: the chosen provider's adapt from the current adapter (the
            # in-context one returns it unchanged), whose weights are fingerprinted first for
            # the rollback's record. Only then can the fit change them, so a fingerprint that
            # raises leaves nothing to roll back.
            fit["before"] = chosen.fingerprint(init)
            fit["ran"] = True
            return chosen.adapt(enlarged, init, max(0.0, deadline - governor.clock()))

        promotion = _promote(
            screened,
            gate,
            adapter,
            demos,
            adapt=adapt,
            redecode=redecode,
            round_added=round_added,
            timing_logger=timing_logger,
            clock=clock,
            guard=guard,
            log=log,
        )
        for promoted in promotion.records:
            log.promotion(promoted)
        record = replace(record, weights_before=fit["before"])
        if promotion.promoted:
            result = (promotion.demos, promotion.adapter)
            record = replace(
                record,
                outcome=KEPT,
                next_version=promotion.demos.version,
                promoted=tuple(pair.test_index for pair in promotion.pairs),
                weights_after=chosen.fingerprint(promotion.adapter),
            )
        elif fit["ran"]:
            current = chosen.rollback(adapter)
            after = chosen.fingerprint(current)
            result = (demos, current)
            record = replace(
                record,
                outcome=ROLLED_BACK,
                weights_after=after,
                restored=_restored(fit["before"], after, chosen is in_context),
            )
        else:
            result = (demos, adapter)
            record = replace(record, outcome=NOT_PROMOTED)
        seconds = clock() - start
        if timing_logger is not None:
            timing_logger.record(demos.puzzle_id, ROUND, seconds)
        log.round(replace(record, seconds=seconds))
        return result
    except Exception as err:
        traceback.print_exc()
        after = None
        if fit["ran"]:
            try:
                current = chosen.rollback(adapter)
                after = chosen.fingerprint(current)
            except Exception:
                traceback.print_exc()
        log.round(
            replace(
                record,
                outcome=RAISED,
                next_version=demos.version,
                promoted=(),
                weights_before=fit["before"],
                weights_after=after,
                restored=(
                    None
                    if not fit["ran"]
                    else _restored(fit["before"], after, chosen is in_context)
                ),
                seconds=clock() - start,
                error=_error(err),
            )
        )
        return demos, current


def _restored(before: str | None, after: str | None, in_context: bool) -> bool | None:
    # Whether a rollback gave back the weights: equal fingerprints. The in-context provider
    # changes no weight, so without fingerprints there is nothing to compare (None), which the
    # final decode does not take as a failed restore (P2.5).
    if in_context and before is None and after is None:
        return None
    return after is not None and after == before


DecodeTests = Callable[[Any, DemoSet, str, bool, tuple[int, ...]], Decoded]
Install = Callable[[str, Sequence[int]], tuple[int, ...]]


def _open_inputs(demos: DemoSet) -> tuple[int, ...]:
    # The test inputs that are not pseudo-pairs: a later decode may replace their candidates.
    promoted = {pair.test_index for pair in demos.pairs if pair.kind == PSEUDO}
    return tuple(index for index in range(len(demos.test_inputs)) if index not in promoted)


def _decode_after(
    record: DecodeRecord,
    adapter: Any,
    demos: DemoSet,
    held_out: bool,
    decode_tests: DecodeTests,
    install: Install,
    log: RoundLog,
    clock: Callable[[], float],
    guard: PromptGuard | None = None,
) -> Decoded | None:
    # One test decode after the rounds, of the inputs that are not pseudo-pairs, then the install
    # of their candidates; a cut or raising decode installs nothing and ends the rounds. With a
    # guard (P2.5) its prompts hold the pseudo-pairs that fit, a drop recorded.
    start = clock()
    try:
        shown = _shown(
            guard, log, demos, record.inputs, record.round_added, record.name, held_out=held_out
        )
        decoded = decode_tests(adapter, shown, record.name, held_out, record.inputs)
        if decoded.cut:
            log.decode(replace(record, batches=decoded.batches, cut=True, seconds=clock() - start))
            return None
        installed = install(record.name, record.inputs)
    except Exception as err:
        traceback.print_exc()
        log.decode(replace(record, seconds=clock() - start, error=_error(err)))
        return None
    log.decode(
        replace(record, batches=decoded.batches, installed=installed, seconds=clock() - start)
    )
    return decoded


def run_rounds(
    demos: DemoSet,
    adapter: Any,
    settings: Rounds,
    *,
    provider: AdapterProvider,
    governor: Governor,
    candidates: Sequence[Iterable[Candidate]],
    decode_tests: DecodeTests,
    install: Install,
    make_gate: Callable[[int], HeldOutGate],
    make_redecode: Callable[[int], Redecode],
    deadline: float,
    budget: float,
    observed_batch: float | None = None,
    in_context: AdapterProvider | None = None,
    guard: PromptGuard | None = None,
    log: RoundLog | None = None,
    timing_logger: timing.TimingLogger | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[DemoSet, Any]:
    """Run a puzzle's rounds and the test decodes after them (tasks P2.3, P2.5; ADR-015, ADR-016).

    Why: B5 P2.3 caps the rounds at ``max_rounds`` (default 1, hard cap 2). Round ``r + 1``
    runs only after a kept round ``r``, on the candidates of the test decode that follows it: a
    round that promotes nothing (declined, rolled back, or nothing to promote) ends the rounds,
    since its state and adapter are the ones that produced the candidates it saw. After a kept
    round the test inputs that are not pseudo-pairs are decoded with the kept adapter, the
    enlarged set in the prompts, in one call, and their candidates replace theirs in the run's
    folder: a pseudo-pair's input keeps the candidates that promoted it, since its own pair in
    the prompt would answer it, so it is not decoded. The next round's view check takes that
    call's layout (``first_inputs``), and its free checks see a pseudo-pair's input with the
    candidates that promoted it, which stop at ``already_promoted``. A decode that the time cap
    cut short or that raised replaces nothing and ends the rounds. With ``final_in_prompts``, the
    default (ADR-013 as amended), the last round decodes nothing; one final decode of the open
    inputs follows the rounds instead, whatever their outcome, with the held-out pair back in the
    prompts, when the time left covers it and the adapter's state is known: after a rollback
    whose restore did not give back the weights (``restored`` not true), the open inputs keep
    their candidates (``NOT_RESTORED``). Each round's plan reserves the decodes that may follow
    it (``refine``'s ``decode_batches``): with ``final_in_prompts`` the final decode, of every
    open input when nothing is kept, and before the last round also the round's own decode;
    without it, the decode after a kept round. Each round offers ``settings.variants`` (task
    P2.5): the gradient round by default, the in-context round (``in_context``, no fit) forced or
    as the governor's fallback; with ``guard`` every prompt that carries pseudo-pairs holds those
    that fit (the length guard), the test decodes' included.

    Args:
        demos: The puzzle's state after round 0, with its held-out pair.
        adapter: Round 0's adapter, which produced ``candidates``.
        settings: The run's ``Rounds``.
        provider: The gradient round's adapter provider.
        governor: The worker's governor.
        candidates: Round 0's view candidates per test input.
        decode_tests: ``decode_tests(adapter, state, name, held_out, inputs) -> Decoded``:
            decodes the test inputs ``inputs`` of ``state`` (increasing indices) in one call,
            prompts ``state.task(held_out=held_out)``'s pairs, into the round folder ``name``,
            its files named by the puzzle's test indices; ``Decoded.candidates`` holds one entry
            per test input of the puzzle, empty for those not decoded.
        install: ``install(name, indices) -> replaced``: copies the decode ``name``'s
            candidates of those test inputs into the run's folder (``install_candidates``).
        make_gate: The gate of round ``r``, whose decode writes into that round's folder.
        make_redecode: The re-decode callable of round ``r``.
        deadline: The puzzle's deadline on the governor's clock.
        budget: The puzzle's budget in seconds.
        observed_batch: Seconds of the slowest DFS batch of round 0's test decode, or None.
        in_context: The in-context round's provider (``InContextProvider``), needed when
            ``settings.variants`` offers it; ignored otherwise.
        guard: The length guard (``PromptGuard``), needed with the in-context round; None
            leaves every prompt as built.
        log: Receives every record; None keeps them nowhere.
        timing_logger: The worker's timing logger, or None.
        clock: Seconds for spans and wall times.

    Returns:
        ``(state, adapter)`` after the last round: the latest kept version and its adapter, or
        ``demos`` and the restored round 0 adapter.

    Raises:
        TypeError: As ``refine`` raises it.
        ValueError: As ``refine`` raises it, or when ``settings.variants`` offers the in-context
            round without its provider.
    """
    if not isinstance(settings, Rounds):
        raise TypeError(f"settings must be Rounds; got {settings!r}")
    offers = IN_CONTEXT in settings.variants
    if offers and in_context is None:
        raise ValueError("the settings offer the in-context round: run_rounds needs its provider")
    log = log if log is not None else RoundLog(None, "")
    first_record = len(log.rounds)
    per_input = -(-DECODE_VIEWS // VIEWS_PER_BATCH)
    state, current = demos, adapter
    found: list[list[Candidate]] = [list(entry) for entry in candidates]
    # Round 0's decode decoded every test input in one call.
    layout: tuple[int, ...] | None = None
    observed = observed_batch
    last = 0
    for round_added in range(1, settings.max_rounds + 1):
        last = round_added
        before = state
        open_now = len(_open_inputs(state))
        after = max(0, open_now - 1)  # inputs still open after a kept round, at most
        if settings.final_in_prompts and state.held_out is not None:
            # The final decode follows whatever this round does: of every open input when the
            # round keeps nothing; before the last round a kept round also decodes on its own.
            last_round = round_added == settings.max_rounds
            follow = open_now if last_round else max(open_now, 2 * after)
        else:
            follow = after
        state, current = refine(
            state,
            current,
            provider,
            make_gate(round_added),
            governor,
            candidates=found,
            redecode=make_redecode(round_added),
            deadline=deadline,
            budget=budget,
            observed_batch=observed,
            round_added=round_added,
            tau=settings.tau,
            first_inputs=layout,
            decode_batches=per_input * follow,
            in_context=in_context if offers else None,
            variants=settings.variants,
            guard=guard,
            log=log,
            timing_logger=timing_logger,
            clock=clock,
        )
        if state.version == before.version:
            break
        if settings.final_in_prompts and round_added == settings.max_rounds:
            break
        inputs = _open_inputs(state)
        name = f"round{round_added}"
        record = DecodeRecord(
            demos.puzzle_id, round_added, name, PAIRS_PROMPTS, inputs, per_input * len(inputs)
        )
        if not inputs:
            log.decode(replace(record, skipped=ALL_PROMOTED))
            break
        decoded = _decode_after(
            record, current, state, False, decode_tests, install, log, clock, guard
        )
        if decoded is None:
            break
        # A pseudo-pair's input keeps the candidates that promoted it.
        found = [
            list(decoded.candidates[index]) if index in inputs else found[index]
            for index in range(len(found))
        ]
        layout = inputs
        if decoded.slowest is not None:
            observed = decoded.slowest if observed is None else max(observed, decoded.slowest)
    if settings.final_in_prompts and state.held_out is not None:
        inputs = _open_inputs(state)
        record = DecodeRecord(
            demos.puzzle_id, last, "final", HELD_OUT_PROMPTS, inputs, per_input * len(inputs)
        )
        cost = governor.cost_model
        estimate = cost.decode_cost(record.expected, observed_batch=observed)
        estimate += (1.0 + cost.margin) * cost.overhead
        remaining = deadline - governor.clock()
        record = replace(record, estimate=estimate, remaining=remaining)
        rounds = log.rounds[first_record:]
        if not record.inputs:
            log.decode(replace(record, skipped=ALL_PROMOTED))
        elif rounds and rounds[-1].restored is False:
            log.decode(replace(record, skipped=NOT_RESTORED))
        elif estimate > remaining:
            log.decode(replace(record, skipped=NO_TIME))
        else:
            _decode_after(record, current, state, True, decode_tests, install, log, clock, guard)
    return state, current


def solve_puzzle(
    holdout: HoldOut,
    model: Any,
    default_weights: dict[str, Any],
    arc_test_set: ArcDataset,
    key: str,
    formatter: QwenFormatter,
    max_seq_length: int,
    tokenizer: Any,
    collator: Any,
    train_args: dict[str, Any],
    timing_logger: timing.TimingLogger | None,
    rank: int,
    max_new_tokens: int,
    start_time: float,
    end_time: float,
    max_score: float,
    dir_outputs: str,
    *,
    queue_size: int | None = None,
    workers: int = 1,
) -> Any:
    """Solve one puzzle with a held-out pair: fit without it, decode the test inputs, gate.

    Why: the split solver's per-puzzle body (``arc26.solver.worker``) when the run holds a pair
    out. ``arc26.ttt.fit`` and ``arc26.decode.decode_puzzle`` run unchanged; the pair is excluded
    by what they are given (ADR-011): ``fit`` trains on a one-puzzle dataset built from
    ``DemoSet.pairs``, the test inputs are decoded with the dataset ``fit`` returns (or with the
    puzzle as loaded when ``holdout.in_test_prompts``), and the gate decodes the held-out input
    with ``pairs`` as its demonstrations into ``holdout.gate_dir/<key>/``, never into
    ``dir_outputs``. The test decode runs first, on the puzzle's clock, exactly as the reference
    decodes; the gate then gets a clock of its own (``decode_puzzle``'s 1200 s cap from its own
    start) and the run's ``end_time``. A puzzle the rule keeps whole (``held_out_index`` None)
    runs the reference's body and logs a verdict that was not applied. A gate that raises is
    logged with its error and the worker goes on, since ``torch.multiprocessing.spawn`` would stop
    every worker. With a logger, the gate's DFS batches and rescorings are recorded as
    ``gate_decode`` and ``gate_selection`` and its span as ``gate``; the verdict goes to
    ``holdout.log_dir`` when set. With ``holdout.rounds`` (task P2.3, ADR-015), the governor gives
    the puzzle its budget from the queue as the worker took it, and after round 0's gate the
    rounds run (``run_rounds``) with ``arc26.ttt.SGDProvider``, their candidate files in
    ``rounds.round_dir/<key>/`` and their records beside the verdicts; round 0's test decode is
    then watched (``OpWatch``) for its slowest DFS batch, the governor's floor. With
    ``rounds.variants`` offering it (task P2.5, ADR-016), the rounds may run the in-context round
    (``InContextProvider``: no fit), and every prompt that carries pseudo-pairs holds those that
    fit the model's limit (``PromptGuard`` with ``longest_prompt``). A puzzle without a held-out
    pair records its free checks and runs no round. Rounds that raise leave the run's candidates
    as they were and the worker going. Without rounds nothing of this runs, and the puzzle is
    solved exactly as in task P2.1.

    Args:
        holdout: The run's hold-out settings.
        model: The model as ``arc26.solver.load`` or the previous puzzle left it.
        default_weights: The adapter's weights right after wrapping.
        arc_test_set: The run's challenges; ``key`` selects the puzzle.
        key: The task id.
        formatter: Prompt and reply formatter of the model.
        max_seq_length: Token limit of a sample (8192).
        tokenizer: The model's tokenizer.
        collator: The completion-only collator.
        train_args: Keyword arguments of ``UnslothTrainingArguments``.
        timing_logger: The worker's timing logger, or None.
        rank: Worker rank.
        max_new_tokens: Longest reply in tokens.
        start_time: ``time.time()`` when the puzzle started.
        end_time: Absolute deadline, ``time.time()`` scale.
        max_score: NLL bound of the DFS.
        dir_outputs: Folder of the run's candidate files.
        queue_size: ``queue.qsize()`` right after the worker took the puzzle (task P2.3): the
            governor's share of the time left (``Governor.puzzle_budget``); needed with rounds.
        workers: The run's workers, the governor's share divisor (task P2.3).

    Returns:
        The model as the last fit (or rollback) left it, for the next puzzle.

    Raises:
        ValueError: If the settings hold rounds and ``queue_size`` is None.
    """
    from arc26.decode import decode_puzzle
    from arc26.nvarc.arc_loader import ArcDataset
    from arc26.ttt import fit

    demos = DemoSet.from_task(key, arc_test_set.queries[key], rule=holdout.rule, k2=holdout.k2)
    log = None if holdout.log_dir is None else VerdictLog(holdout.log_dir, f"rank{rank}")
    trained: dict[str, Any] = {}
    rounds = holdout.rounds
    governor: Governor | None = None
    budget = 0.0
    watch: OpWatch | None = None
    if rounds is not None:
        if queue_size is None:
            raise ValueError("rounds need the queue size, the governor's share of the time left")
        # The puzzle's budget is its share of the time left when the worker took it (P2.4).
        governor = Governor.from_mapping(rounds.governor, end_time=end_time, workers=workers)
        budget = governor.puzzle_budget(queue_size)
        if demos.held_out is not None:
            watch = OpWatch(timing_logger)

    def fit_pairs(training: DemoSet) -> Any:
        # The reference's fit on a dataset of the training pairs alone (ADR-011: training on a
        # changed pair set means passing another ArcDataset as arc_test_set).
        dataset = ArcDataset(queries={key: training.task()}, replies={})
        trained["model"], trained["puzzle_ds"] = fit(
            model=model,
            default_weights=default_weights,
            arc_test_set=dataset,
            key=key,
            formatter=formatter,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            collator=collator,
            train_args=train_args,
            timing_logger=timing_logger,
            rank=rank,
        )
        return trained["model"]

    def decode_input(adapter: Any, context: DemoSet, query: Grid, views: int) -> list[Candidate]:
        # The reference's decode of one input with the training pairs as demonstrations, into
        # the gate's own folder, its records renamed for the gate.
        if views != DECODE_VIEWS:
            raise ValueError(f"decode_puzzle decodes {DECODE_VIEWS} views; got {views}")
        folder = os.path.join(holdout.gate_dir, key)
        _clear(folder)
        decode_puzzle(
            puzzle_ds=ArcDataset(queries={key: context.task([query])}, replies={}),
            formatter=formatter,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            start_time=time.time(),
            end_time=end_time,
            rank=rank,
            key=key,
            tokenizer=tokenizer,
            model=adapter,
            max_score=max_score,
            timing_logger=None if timing_logger is None else RenamedOps(timing_logger, GATE_OPS),
            dir_outputs=folder,
        )
        return read_candidates(folder)

    if demos.held_out is None:
        model, puzzle_ds = fit(
            model=model,
            default_weights=default_weights,
            arc_test_set=arc_test_set,
            key=key,
            formatter=formatter,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            collator=collator,
            train_args=train_args,
            timing_logger=timing_logger,
            rank=rank,
        )
        tests = puzzle_ds
    else:
        gate = HeldOutGate(fit_pairs, decode_input)
        model = gate.fit(demos)
        tests = trained["puzzle_ds"]
        if holdout.in_test_prompts:
            tests = arc_test_set.change_keys([key])
    decode_puzzle(
        puzzle_ds=tests,
        formatter=formatter,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        start_time=start_time,
        end_time=end_time,
        rank=rank,
        key=key,
        tokenizer=tokenizer,
        model=model,
        max_score=max_score,
        timing_logger=timing_logger if watch is None else watch,
        dir_outputs=dir_outputs,
    )
    if demos.held_out is None:
        verdict = GateVerdict.not_applied(demos)
        print(f"[Rank {rank}] gate for {key}: not applied, {demos.real_pairs} real pairs")
    else:
        print(
            f"[Rank {rank}] gate for {key}: decoding the held-out pair, index "
            f"{demos.held_out_index} of {demos.real_pairs} real pairs"
        )
        gate_start = time.perf_counter()
        try:
            verdict = gate.verdict(model, demos)
        except Exception as err:
            # The test inputs' candidates are written by now. torch.multiprocessing.spawn stops
            # every worker when one raises, so a gate that fails (out of memory, say) is recorded
            # and the worker goes on to its next puzzle; like any failed op, it writes no timing
            # record.
            traceback.print_exc()
            verdict = GateVerdict.raised(demos, err, time.perf_counter() - gate_start)
            print(f"[Rank {rank}] gate for {key}: raised {verdict.error}")
        else:
            if timing_logger is not None:
                timing_logger.record(key, GATE, time.perf_counter() - gate_start)
            place = (
                "not among them"
                if verdict.expected_rank is None
                else f"ranked {verdict.expected_rank}"
            )
            print(
                f"[Rank {rank}] gate for {key}: {'passed' if verdict.passed else 'failed'}, "
                f"{verdict.candidates} candidates, the held-out output {place}"
            )
    if log is not None:
        log.write(verdict)
    if rounds is None or governor is None:
        return model
    rounds_log = RoundLog(
        holdout.log_dir, f"rank{rank}", say=lambda line: print(f"[Rank {rank}] {line}")
    )
    # A puzzle without a held-out pair goes through the rounds too: its free checks are
    # recorded (no_held_out), and no GPU op runs.
    try:
        model = _refine_puzzle(
            rounds=rounds,
            demos=demos,
            model=model,
            key=key,
            formatter=formatter,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            collator=collator,
            train_args=train_args,
            timing_logger=timing_logger,
            rank=rank,
            max_new_tokens=max_new_tokens,
            end_time=end_time,
            max_score=max_score,
            dir_outputs=dir_outputs,
            governor=governor,
            budget=budget,
            deadline=start_time + budget,
            observed_batch=None if watch is None else watch.slowest.get(timing.DECODE),
            log=rounds_log,
        )
    except Exception as err:
        # Outside refine's own handling: the rounds' setup or their records. The run's candidate
        # files are as round 0 or the last installed decode left them, and the worker goes on,
        # even when the record of the error cannot be written either.
        traceback.print_exc()
        try:
            rounds_log.round(
                RoundRecord(key, 1, demos.version, demos.version, RAISED, error=_error(err))
            )
        except Exception:
            traceback.print_exc()
    return model


def _fit_nothing(demos: DemoSet) -> Any:
    # A round's gate only checks the candidate adapter that promote fits through the provider.
    raise RuntimeError(f"{demos.puzzle_id}: a round's gate checks a candidate adapter, fits none")


def _refine_puzzle(
    *,
    rounds: Rounds,
    demos: DemoSet,
    model: Any,
    key: str,
    formatter: QwenFormatter,
    max_seq_length: int,
    tokenizer: Any,
    collator: Any,
    train_args: dict[str, Any],
    timing_logger: timing.TimingLogger | None,
    rank: int,
    max_new_tokens: int,
    end_time: float,
    max_score: float,
    dir_outputs: str,
    governor: Governor,
    budget: float,
    deadline: float,
    observed_batch: float | None,
    log: RoundLog,
) -> Any:
    # The GPU callables of run_rounds, built as solve_puzzle builds the gate's (task P2.3): every
    # decode is decode_puzzle on a one-key dataset of the state's task entry, with a clock of its
    # own and the run's end time, into its own folder of rounds.round_dir/<key>/, its records
    # renamed and watched for the DFS batches it ran. The state they get is the one the length
    # guard left (task P2.5): refine and run_rounds apply it before each call.
    from arc26.decode import decode_puzzle
    from arc26.nvarc.arc_loader import ArcDataset
    from arc26.ttt import SGDProvider

    folder = os.path.join(rounds.round_dir, key)
    # Files an earlier launch left would join this puzzle's votes or its round 0 backup.
    shutil.rmtree(folder, ignore_errors=True)
    per_input = DECODE_VIEWS // VIEWS_PER_BATCH

    def decode_into(
        adapter: Any,
        entry: dict[str, Any],
        target: str,
        names: Mapping[str, str],
        view_seed: int | None = None,
    ) -> OpWatch:
        _clear(target)
        watched = OpWatch(timing_logger, names)
        decode_puzzle(
            puzzle_ds=ArcDataset(queries={key: entry}, replies={}),
            formatter=formatter,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            start_time=time.time(),
            end_time=end_time,
            rank=rank,
            key=key,
            tokenizer=tokenizer,
            model=adapter,
            max_score=max_score,
            timing_logger=watched,
            dir_outputs=target,
            view_seed=view_seed,
        )
        return watched

    def make_redecode(round_added: int) -> Redecode:
        def redecode(adapter: Any, context: DemoSet, query: Grid, seed: int) -> list[Candidate]:
            index = context.test_inputs.index(query)
            target = os.path.join(folder, f"round{round_added}-redecode{index}")
            watched = decode_into(adapter, context.task([query]), target, REDECODE_OPS, seed)
            found = read_candidates(target)
            ran = watched.counts.get(timing.DECODE, 0)
            if ran < per_input:
                raise DecodeCut(found, ran, per_input)
            return found

        return redecode

    def make_gate(round_added: int) -> HeldOutGate:
        def gate_decode(adapter: Any, context: DemoSet, query: Grid, views: int) -> list[Candidate]:
            if views != DECODE_VIEWS:
                raise ValueError(f"decode_puzzle decodes {DECODE_VIEWS} views; got {views}")
            target = os.path.join(folder, f"round{round_added}-gate")
            watched = decode_into(adapter, context.task([query]), target, GATE_OPS)
            found = read_candidates(target)
            ran = watched.counts.get(timing.DECODE, 0)
            if ran < per_input:
                raise DecodeCut(found, ran, per_input)
            return found

        return HeldOutGate(_fit_nothing, gate_decode)

    def decode_tests(
        adapter: Any, state: DemoSet, name: str, held_out: bool, inputs: tuple[int, ...]
    ) -> Decoded:
        target = os.path.join(folder, name)
        names = FINAL_OPS if held_out else ROUND_OPS
        queries = [state.test_inputs[index] for index in inputs]
        watched = decode_into(adapter, state.task(queries, held_out=held_out), target, names)
        renumber_candidates(target, key, inputs)
        return Decoded(
            candidates=read_puzzle(target, key, len(state.test_inputs)),
            batches=watched.counts.get(timing.DECODE, 0),
            expected=per_input * len(inputs),
            slowest=watched.slowest.get(timing.DECODE),
        )

    def install(name: str, indices: Sequence[int]) -> tuple[int, ...]:
        return install_candidates(
            os.path.join(folder, name),
            dir_outputs,
            key,
            indices,
            backup=os.path.join(folder, "round0"),
        )

    provider = SGDProvider(
        key=key,
        formatter=formatter,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        collator=collator,
        train_args=train_args,
        timing_logger=timing_logger,
        rank=rank,
        steps=rounds.continue_steps,
    )
    # Task P2.5: the in-context round fits nothing; its records fingerprint the adapter as the
    # gradient provider does, so they show it unchanged. The length guard measures what
    # decode_puzzle's cut_to_len measures, against the same limit.
    in_context = (
        InContextProvider(fingerprint=provider.fingerprint)
        if IN_CONTEXT in rounds.variants
        else None
    )
    guard = PromptGuard(
        limit=max_seq_length - max_new_tokens,
        measure=partial(longest_prompt, formatter=formatter),
    )
    _, model = run_rounds(
        demos,
        model,
        rounds,
        provider=provider,
        governor=governor,
        candidates=read_puzzle(dir_outputs, key, len(demos.test_inputs)),
        decode_tests=decode_tests,
        install=install,
        make_gate=make_gate,
        make_redecode=make_redecode,
        deadline=deadline,
        budget=budget,
        observed_batch=observed_batch,
        in_context=in_context,
        guard=guard,
        log=log,
        timing_logger=timing_logger,
    )
    return model
