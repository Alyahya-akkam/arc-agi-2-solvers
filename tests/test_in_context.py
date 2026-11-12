"""Acceptance tests of the in-context round and the length guard (task P2.5; BRIEF B5 P2.5).

B5 P2.5: "provider ``InContextProvider`` that returns the current adapter unchanged.
Acceptance: DEV40 lift versus P1 measured and reported; context length guard tested at the 30x30
worst case (no unbounded appends)." The DEV40 lift is runs ``P2.5-01`` and ``P2.5-02``
(``docs/prerun/``); this file holds the round's logic with the CPU stand-ins of
``tests/test_refine.py`` (a numpy weight vector trained in place by the gradient provider): a
kept and a rolled-back in-context round (the adapter's fingerprint unchanged in both, the version
up by one or unchanged), the governor's ladder (in context when the gradient round does not fit,
none when neither does), the guard at the 30 by 30 worst case (real pairs never dropped,
pseudo-pairs dropped latest first, the in-context round not offered when none fits, the
reference's ``cut_to_len`` cutting nothing behind it), the settings (``Rounds.variants``), and the
records. The split solver's stand-in runs are ``tests/test_in_context_solver.py``. The synthetic
tasks are never competition data.
"""

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from arc26 import loop
from arc26.governor import GRADIENT, IN_CONTEXT, LADDER, NONE
from arc26.loop import (
    CONTEXT_PREFIX,
    DECLINED,
    DEFAULT_VARIANTS,
    GATE_FAILED,
    HELD_OUT_PROMPTS,
    KEPT,
    NO_TIME,
    NOT_RESTORED,
    OUTCOMES,
    PAIRS_PROMPTS,
    PROMOTED,
    REASONS,
    ROLLED_BACK,
    TOO_LONG,
    ContextRecord,
    DemoSet,
    HeldOutGate,
    HoldOut,
    InContextProvider,
    Pair,
    PromptGuard,
    RoundLog,
    Rounds,
    context_fit,
    longest_prompt,
    refine,
    run_rounds,
)
from arc26.runner import RunConfig, solver_command

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parent.parent
MOVED = ("arc26.nvarc.arc_loader",)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


# The stand-ins of P2.3's round tests: the model, the providers, the gate's decode, the
# re-decode, the decoder of the test decodes after the rounds, the governor, the synthetic tasks.
REFINE = load("_in_context_refine", REPO / "tests" / "test_refine.py")
A, B, C = REFINE.A, REFINE.B, REFINE.C
T0, T1, Y0, Y1, W = REFINE.T0, REFINE.T1, REFINE.Y0, REFINE.Y1, REFINE.W
AGREE, GOVERNOR, PUZZLE = REFINE.AGREE, REFINE.GOVERNOR, REFINE.PUZZLE
Model, InPlaceProvider = REFINE.Model, REFINE.InPlaceProvider
GateDecode, Redecode, Decoder = REFINE.GateDecode, REFINE.Redecode, REFINE.Decoder
demos, votes, grid, governor, Clock = (
    REFINE.demos, REFINE.votes, REFINE.grid, REFINE.governor, REFINE.Clock
)  # fmt: skip


@pytest.fixture(autouse=True, scope="module")
def moved_loader():
    """The moved arc_loader imported with a stub transformers (it imports AutoTokenizer for an
    annotation), removed again afterwards, as in tests/test_refine.py."""
    with pytest.MonkeyPatch.context() as patch:
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = type("AutoTokenizer", (), {})
        patch.setitem(sys.modules, "transformers", stub)
        for name in MOVED:
            patch.delitem(sys.modules, name, raising=False)
        yield
        for name in MOVED:
            sys.modules.pop(name, None)


def cells(entry):
    """A token count without a tokenizer: one per cell, and ten per prompt."""
    grids = [g for pair in entry["train"] for g in pair.values()]
    grids += [test["input"] for test in entry["test"]]
    return 10 + sum(len(row) for g in grids for row in g)


def loose(measure=cells):
    """A guard whose limit no stand-in task reaches."""
    return PromptGuard(limit=100_000, measure=measure)


def digest(model):
    return hashlib.blake2b(model.weights.tobytes(), digest_size=16).hexdigest()


def run(state=None, *, candidates=None, model=None, provider=None, in_context="fingerprint",
        gate_decode=None, redecode=None, gov=None, budget=None, guard="loose",
        **kwargs):  # fmt: skip
    """One refine call with the stand-ins, the in-context provider given (fingerprinting as the
    gradient provider does) and, by default, forced, with a guard no stand-in task reaches;
    returns the result and the stand-ins."""
    state = state if state is not None else demos()
    model = model or Model()
    provider = provider or InPlaceProvider(model)
    if in_context == "fingerprint":
        in_context = InContextProvider(fingerprint=provider.fingerprint)
    gate_decode = gate_decode or GateDecode()
    redecode = redecode or Redecode()
    gov = gov or governor()
    if budget is None:
        budget = gov.puzzle_budget(0)
    if candidates is None:
        candidates = [AGREE[query] for query in state.test_inputs]
    kwargs.setdefault("variants", (IN_CONTEXT,))
    log = kwargs.pop("log", None) or RoundLog(None, "rank0")
    result = refine(
        state,
        model,
        provider,
        HeldOutGate(lambda s: None, gate_decode),
        gov,
        candidates=candidates,
        redecode=redecode,
        deadline=gov.clock() + budget,
        budget=budget,
        in_context=in_context,
        guard=loose() if guard == "loose" else guard,
        log=log,
        **kwargs,
    )
    return result, types.SimpleNamespace(
        model=model, provider=provider, in_context=in_context, gate=gate_decode,
        redecode=redecode, log=log, gov=gov,
    )  # fmt: skip


# B5 acceptance: a kept and a rolled-back in-context round, the adapter unchanged in both.


def test_a_kept_in_context_round_enlarges_the_state_with_the_adapter_unchanged():
    """Promote, enlarge (version + 1), no fit: the candidate adapter is the current one, and the
    gate decodes with the pseudo-pairs in its prompts. The fingerprints before and after are
    equal, and the gradient provider never runs."""
    state = demos(T0, T1)
    model = Model()
    before = model.weights.copy()
    (kept, adapter), stand = run(state, model=model)
    assert kept.version == state.version + 1 and kept is not state and state.version == 0
    assert [(p.input, p.output, p.test_index) for p in kept.pseudo_pairs] == [
        (T0, Y0, 0), (T1, Y1, 1)
    ]  # fmt: skip
    assert adapter is model and model.weights.tobytes() == before.tobytes()
    assert stand.provider.adapts == [] and stand.provider.rollbacks == []
    # The gate checked the unchanged adapter with the enlarged set in its prompts.
    ((weights, gate_state, query, _),) = stand.gate.calls
    assert weights.tobytes() == before.tobytes() and query == C.input
    assert [p.kind for p in gate_state.pairs] == ["real", "real", "pseudo", "pseudo"]
    (record,) = stand.log.rounds
    assert (record.outcome, record.version, record.next_version) == (KEPT, 0, 1)
    assert record.decision == IN_CONTEXT and record.promoted == (0, 1)
    assert record.weights_before == record.weights_after == digest(model)
    assert record.restored is None
    (plan,) = stand.log.plans
    assert plan["decision"] == IN_CONTEXT and set(plan["estimates"]) == {IN_CONTEXT}
    assert [p.reason for p in stand.log.promotions] == [PROMOTED, PROMOTED]


def test_a_failed_in_context_gate_rolls_back_the_version_only():
    """The voted answer W is wrong but confident (14 of 16 views, the re-decode agrees); the
    gate, decoding with it in its prompts, fails. The state is returned as given, at its
    version; the adapter was never changed, so nothing is restored, and the fingerprints match."""
    state = demos()
    model = Model()
    before = model.weights.copy()
    wrong = {T0: votes((W, 14), (Y0, 2))}
    (rolled, adapter), stand = run(
        state, model=model, candidates=[wrong[T0]], redecode=Redecode(wrong),
        gate_decode=GateDecode(wrong={W}),
    )  # fmt: skip
    assert rolled is state and rolled.version == 0 and rolled.pairs == (A, B)
    assert adapter is model and model.weights.tobytes() == before.tobytes()
    assert stand.provider.adapts == [] and stand.provider.rollbacks == []
    ((_, gate_state, _, _),) = stand.gate.calls
    assert gate_state.pairs[-1].output == W  # the wrong pair was in the gate's prompts
    (record,) = stand.log.rounds
    assert (record.outcome, record.version, record.next_version) == (ROLLED_BACK, 0, 0)
    assert record.decision == IN_CONTEXT and record.restored is True
    assert record.weights_before == record.weights_after == digest(model)
    (promotion,) = stand.log.promotions
    assert promotion.reason == GATE_FAILED and promotion.gate.passed is False


def test_the_in_context_provider_returns_the_adapter_unchanged():
    model, other = Model(), Model()
    plain = InContextProvider()
    assert plain.adapt(demos(), model, 12.0) is model and plain.rollback(model) is model
    assert plain.fingerprint(model) is None and plain.fit_shape(demos()) == (0, 0)
    digested = InContextProvider(fingerprint=digest)
    other.weights += np.float32(1.0)
    assert digested.fingerprint(model) == digest(model) != digested.fingerprint(other)
    assert digested.adapt(demos(), other, 0.0) is other and digested.rollback(other) is other
    with pytest.raises(TypeError, match="fingerprint must be None or callable"):
        InContextProvider(fingerprint="blake2b")


def test_without_fingerprints_a_rollback_leaves_restored_unset_and_the_final_decode_runs():
    """The in-context provider changes no weight: without a fingerprint there is nothing to
    compare, so restored stays None, which the final decode does not take as a failed restore
    (P2.3's not_restored is for a restore that did not give the weights back)."""
    model = Model()
    gov = governor()
    log = RoundLog(None, "rank0")
    decoder = Decoder()
    state, adapter = run_rounds(
        demos(T0, T1), model, rounds(variants=[IN_CONTEXT]), provider=InPlaceProvider(model),
        governor=gov, candidates=[AGREE[T0], AGREE[T1]], decode_tests=decoder.decode_tests,
        install=decoder.install, make_gate=lambda r: HeldOutGate(lambda s: None,
        GateDecode(wrong={Y0})), make_redecode=lambda r: Redecode(), deadline=gov.clock() + 1e6,
        budget=1e6, in_context=InContextProvider(), guard=loose(), log=log,
    )  # fmt: skip
    (record,) = log.rounds
    assert (record.outcome, record.restored, record.weights_before) == (ROLLED_BACK, None, None)
    (final,) = log.decodes
    assert (final.name, final.skipped, final.installed) == ("final", None, (0, 1))
    assert state.version == 0 and adapter is model


# The governor's ladder (B2 Q4): in context when the gradient round does not fit, none when
# neither does, gradient whenever it fits.


class CostlyProvider(InPlaceProvider):
    """A gradient provider whose fit the governor prices high: 2.4M tokens in 32 steps."""

    def fit_shape(self, state):
        return 2_400_000, self.steps


def costs(state, guard):
    """The two estimates refine prices for a one-input puzzle with both rounds offered."""
    (_, _), stand = run(
        state, provider=CostlyProvider(Model()), variants=None, guard=guard,
        gov=governor(end=1e9),
    )  # fmt: skip
    (plan,) = stand.log.plans
    return plan["estimates"][GRADIENT], plan["estimates"][IN_CONTEXT]


def big_prompt(entry):
    """A measure that makes every prompt about 20k tokens, so the pseudo-pair adds little."""
    return 20_000 + cells(entry)


def test_the_governor_picks_in_context_when_the_gradient_round_does_not_fit():
    guard = PromptGuard(limit=100_000, measure=big_prompt)
    gradient, in_context = costs(demos(), guard)
    assert in_context < gradient  # a costly fit, a prompt barely longer
    gov = governor(end=1e9, clock=Clock(1000.0))
    model = Model()
    before = model.weights.copy()
    (kept, adapter), stand = run(
        demos(), model=model, provider=CostlyProvider(model), variants=None, guard=guard,
        gov=gov, budget=(gradient + in_context) / 2,
    )  # fmt: skip
    (plan,) = stand.log.plans
    assert plan["decision"] == IN_CONTEXT and list(plan["estimates"]) == list(LADDER)
    assert stand.log.rounds[0].estimate == pytest.approx(in_context)
    assert kept.version == 1 and adapter is model and np.array_equal(model.weights, before)
    assert stand.provider.adapts == []


def test_the_governor_picks_none_when_neither_round_fits():
    guard = PromptGuard(limit=100_000, measure=big_prompt)
    gradient, in_context = costs(demos(), guard)
    (state, _), stand = run(
        demos(), provider=CostlyProvider(Model()), variants=None, guard=guard,
        gov=governor(end=1e9, clock=Clock(1000.0)), budget=min(gradient, in_context) / 2,
    )  # fmt: skip
    (plan,) = stand.log.plans
    assert plan["decision"] == NONE and set(plan["estimates"]) == {GRADIENT, IN_CONTEXT}
    (record,) = stand.log.rounds
    assert record.outcome == DECLINED and record.estimate == pytest.approx(gradient)
    assert [p.reason for p in stand.log.promotions] == [DECLINED] and state.version == 0
    assert stand.redecode.calls == [] and stand.gate.calls == [] and stand.provider.adapts == []


def test_the_governor_picks_the_gradient_round_whenever_it_fits():
    """B2 Q4's ladder, not the cheapest first: the in-context round is the fallback, so the
    gradient round runs whenever it fits, here though the in-context one is cheaper."""
    guard = PromptGuard(limit=100_000, measure=big_prompt)
    (kept, _), stand = run(demos(), variants=None, guard=guard)
    (plan,) = stand.log.plans
    assert plan["estimates"][IN_CONTEXT] < plan["estimates"][GRADIENT]
    assert plan["decision"] == GRADIENT and len(stand.provider.adapts) == 1
    assert stand.log.rounds[0].decision == GRADIENT and kept.version == 1


def test_the_in_context_price_has_no_fit_and_scales_the_gate_and_test_decodes():
    """No train term; the re-decodes at the round-0 length; the gate's and the test decodes'
    batches at the length ratio of the prompts with the pairs held over those without."""
    state = demos(T0, T1)
    gov = governor()
    (_, _), stand = run(state, gov=gov, observed_batch=200.0, decode_batches=8)
    (plan,) = stand.log.plans
    shape = plan["shape"]
    assert shape["screened"] == [0, 1] and "train_tokens" not in shape
    held = state.enlarged([
        Pair(T0, Y0, kind="pseudo", round_added=1, test_index=0),
        Pair(T1, Y1, kind="pseudo", round_added=1, test_index=1),
    ])  # fmt: skip
    ratio = max(
        cells(held.task([query])) / cells(state.task([query])) for query in (C.input, T0, T1)
    )
    assert shape[IN_CONTEXT] == {
        "screened": [0, 1], "too_long": [], "redecode_batches": 8,
        "length_ratio": round(ratio, 6),
    }  # fmt: skip
    expected = gov.cost_model.round_costs(
        train_tokens=0, test_batches=8, gate_batches=4, redecode_batches=8,
        length_ratio=round(ratio, 6), observed_batch=200.0,
    )[IN_CONTEXT]  # fmt: skip
    assert plan["estimates"] == {IN_CONTEXT: round(expected, 3)}
    no_pairs = gov.cost_model.round_costs(
        train_tokens=0, test_batches=8, gate_batches=4, redecode_batches=8, observed_batch=200.0
    )[IN_CONTEXT]  # fmt: skip
    assert expected > no_pairs and ratio > 1


# The length guard at the 30 by 30 worst case (B5 P2.5: no unbounded appends).


class WordTokenizer:
    """The reference model's tokenization of a prompt: one token per special token, per role
    word, per newline, and per color (NVARC's word-level vocabulary, arc26/decode.py ARC_VOCAB)."""

    WORDS = re.compile(r"<\|im_start\|>|<\|im_end\|>|user|assistant|\n|\d")

    def encode(self, text):
        return [hash(word) % 16 for word in self.WORDS.findall(text)]


def formatter():
    from arc26.nvarc.arc_loader import QwenFormatter

    return QwenFormatter(tokenizer=WordTokenizer())


def square(base, side=30):
    """A side by side grid of every color, its top-left cell ``base``: so its views differ (the
    view check needs it), and a view's pair maps back to its ``base``."""
    return grid(*[[(base + row + col) % 10 for col in range(side)] for row in range(side)])


def max_seq_guard():
    """The solver's guard: 8192 tokens less the longest reply (arc26.solver.load)."""
    fmt = formatter()
    return PromptGuard(limit=8192 - fmt.max_new_tokens(), measure=lambda e: longest_prompt(e, fmt))


def big_state(real=3, tests=1, pseudo=0):
    """A puzzle of 30 by 30 grids, the pairs' bases 1, 2, ... (the last held out); test inputs of
    bases 8, 9, 7, and pseudo-pairs on them with outputs of bases 5, 6."""
    pairs = [Pair(square(i + 1), square(i + 1)) for i in range(real)]
    queries = [square(8), square(9), square(7)][:tests]
    task = {
        "train": [pair.task_pair() for pair in pairs],
        "test": [{"input": [list(r) for r in q]} for q in queries],
    }
    state = DemoSet.from_task(PUZZLE, task)
    if pseudo:
        state = state.enlarged([
            Pair(queries[i], square(5 + i), kind="pseudo", round_added=1, test_index=i)
            for i in range(pseudo)
        ])  # fmt: skip
    return state


def test_a_30_by_30_prompt_costs_what_the_reference_tokenizer_counts():
    """B3: a 30 by 30 grid is 929 tokens and its newlines, a pair 1,866 with the turns, a query
    936; the limit is 8,192 less a 30 by 30 reply and its end (931)."""
    fmt = formatter()
    assert fmt.max_new_tokens() == 931 and max_seq_guard().limit == 7261
    one = {"train": [Pair(square(1), square(1)).task_pair()], "test": [{"input": square(8)}]}
    two = {"train": one["train"] * 2, "test": one["test"]}
    assert longest_prompt(one, fmt) == 1866 + 936
    assert longest_prompt(two, fmt) - longest_prompt(one, fmt) == 1866
    # The longer orientation counts: a 10 by 30 grid has 9 newlines as it stands, 29 in the views
    # that exchange rows and columns (rot90, transpose), so its prompts' longest view is those.
    wide = {"train": [Pair(grid(*[[1] * 30] * 10), grid(*[[1] * 30] * 10)).task_pair()],
            "test": [{"input": grid(*[[2] * 30] * 10)}]}  # fmt: skip
    assert longest_prompt(wide, fmt) == 2 * (300 + 29) + 8 + (300 + 29) + 7
    from arc26.nvarc.arc_loader import ArcDataset

    as_given = ArcDataset(queries={"prompt": wide}).split_multi_replies()
    assert as_given.get_length("prompt_0", fmt, "input") == 2 * (300 + 9) + 8 + (300 + 9) + 7


def test_context_fit_keeps_the_longest_prefix_of_pseudo_pairs_that_fits():
    limit = 7261
    # Two real pairs and a query (4,668), then 30 by 30 pseudo-pairs of 1,866 each.
    lengths = [4668, 6534, 8400, 10266]
    assert context_fit(lengths, limit) == 1  # the latest two dropped
    assert context_fit(lengths[:2], limit) == 1 and context_fit(lengths[:1], limit) == 0
    assert context_fit([4668, 7261], limit) == 1  # the limit itself fits (cut_to_len: >)
    assert context_fit([4668, 7262], limit) == 0
    # A prompt over the limit without a pseudo-pair holds none (the reference cuts it itself).
    assert context_fit([8400, 8401], limit) == 0
    # A length above the limit ends the prefix even if a later one would fit.
    assert context_fit([100, 9000, 200], limit) == 0
    for bad, error in (
        (([], limit), ValueError), (([1, -2], limit), ValueError), (([1], 0), ValueError),
        (([1.0], limit), TypeError), (([1], 7261.0), TypeError), (("12", limit), TypeError),
    ):  # fmt: skip
        with pytest.raises(error):
            context_fit(*bad)


def test_the_guard_at_the_30_by_30_worst_case_drops_pseudo_pairs_latest_first_never_a_real_pair():
    """Three real 30 by 30 pairs (one held out) and two pseudo-pairs: the gate's prompts (two
    real pairs, the held-out input as query) hold the first pseudo-pair only; the final decode's
    (the held-out pair back) hold none; every real pair stays in both."""
    guard = max_seq_guard()
    state = big_state(real=3, tests=2, pseudo=2)
    gate = guard.fit(state, [state.held_out.input])
    assert gate.lengths == (4668, 6534, 8400) and (gate.kept, gate.dropped) == ((0,), (1,))
    final = guard.fit(state, [state.test_inputs[1]], held_out=True)
    assert final.lengths == (6534, 8400, 10266) and (final.kept, final.dropped) == ((), (0, 1))
    shown = state.first_pseudo(len(gate.kept))
    assert [p.kind for p in shown.pairs] == ["real", "real", "pseudo"]
    assert shown.pseudo_pairs == state.pseudo_pairs[:1] and shown.version == state.version
    assert shown.held_out == state.held_out
    assert [p.kind for p in state.first_pseudo(0).pairs] == ["real", "real"]
    assert state.first_pseudo(2) is state
    with pytest.raises(ValueError, match="count must be 0 to the 2 pseudo-pairs"):
        state.first_pseudo(3)
    with pytest.raises(TypeError, match="count must be an integer"):
        state.first_pseudo(1.0)


def views_of(entry, guard):
    """The pairs left in each decoding view after the reference's cut, by their input's base:
    what decode_puzzle builds (split_multi_replies, augment(n=2, seed=2), cut_to_len), each
    view's demonstrations mapped back to the puzzle's orientation and colors."""
    from arc26.nvarc.arc_loader import ArcDataset

    fmt = formatter()
    multi = ArcDataset(queries={PUZZLE: entry}).split_multi_replies()
    views = multi.augment(n=2, seed=2).cut_to_len(formatter=fmt, name="input", max_len=guard.limit)
    found = {}
    for key in views.keys:
        colors = []
        for pair in views.queries[key]["train"]:
            back = ArcDataset.invert_mod(np.array(pair["input"]), key, inv_perm=True)
            colors.append(int(back[0][0]))
        found[key] = sorted(colors)
    return found


def test_behind_the_guard_cut_to_len_drops_no_pair_and_without_it_real_pairs_are_lost():
    """The hazard the guard closes: two real pairs (inputs of bases 1, 2) and two 30 by 30
    pseudo-pairs (on the test inputs, bases 8, 9) exceed the limit, so the reference's cut drops
    each view's first demonstration, and the views shuffle the demonstrations, so real pairs go
    too. Behind the guard every view keeps both real pairs and the pseudo-pair that fits."""
    guard = max_seq_guard()
    state = big_state(real=3, tests=2, pseudo=2)
    query = state.held_out.input
    unguarded = views_of(state.without_held_out().task([query]), guard)
    assert len(unguarded) == 16 and all(len(colors) == 3 for colors in unguarded.values())
    lost_real = [key for key, colors in unguarded.items() if not {1, 2} <= set(colors)]
    assert lost_real, "with seed 2 some view loses a real pair to the cut"
    fit = guard.fit(state, [query])
    shown = state.first_pseudo(len(fit.kept)).without_held_out()
    guarded = views_of(shown.task([query]), guard)
    assert set(map(tuple, guarded.values())) == {(1, 2, 8)}


def test_the_in_context_round_is_not_offered_when_no_pseudo_pair_fits():
    """Four real 30 by 30 pairs, one held out: three real pairs and the query fill the gate's
    prompt (6,534 of 7,261 tokens), so no pseudo-pair fits. Forced, the round is not offered:
    no plan, no GPU op, the input's reason too_long. Offered beside the gradient round, the
    governor gets the gradient round alone."""
    state = big_state(real=4, tests=1)
    answer = square(4)
    candidates = [votes((answer, 16))]
    redecode = Redecode({state.test_inputs[0]: votes((answer, 16))})
    (same, _), stand = run(
        state, candidates=candidates, redecode=redecode, guard=max_seq_guard(),
        gate_decode=GateDecode(right=state.held_out.output),
    )  # fmt: skip
    assert same is state and stand.log.plans == [] and stand.redecode.calls == []
    assert stand.gate.calls == [] and stand.provider.adapts == []
    (record,) = stand.log.rounds
    assert (record.outcome, record.decision, record.screened) == (TOO_LONG, None, (0,))
    assert [p.reason for p in stand.log.promotions] == [TOO_LONG]
    (kept, _), both = run(
        state, candidates=candidates, redecode=Redecode({state.test_inputs[0]: votes((answer,
        16))}), guard=max_seq_guard(), gate_decode=GateDecode(right=state.held_out.output),
        variants=None,
    )  # fmt: skip
    (plan,) = both.log.plans
    assert set(plan["estimates"]) == {GRADIENT} and plan["decision"] == GRADIENT
    assert plan["shape"][IN_CONTEXT] == {
        "screened": [], "too_long": [0], "redecode_batches": 0, "length_ratio": 1.0
    }  # fmt: skip
    # The gradient round trains on the pair, and its gate's prompts, full, leave it out.
    assert kept.version == 1 and len(both.provider.adapts) == 1
    (context,) = both.log.contexts
    assert (context.name, context.kept, context.dropped) == ("round1-gate", (), (0,))
    assert context.lengths == (6534, 8400)


def test_the_in_context_round_holds_the_pairs_that_fit_and_records_the_others_too_long():
    """Three real 30 by 30 pairs and two test inputs, both confident: the gate's prompt holds one
    pseudo-pair, the first. Input 1 is recorded too_long and never re-decoded; input 0 is
    re-decoded, gated with its pair in the prompts, and promoted."""
    state = big_state(real=3, tests=2)
    first, second = state.test_inputs
    answers = {first: votes((square(5), 16)), second: votes((square(6), 16))}
    redecode = Redecode(answers)
    (kept, _), stand = run(
        state, candidates=[answers[first], answers[second]], redecode=redecode,
        guard=max_seq_guard(), gate_decode=GateDecode(right=state.held_out.output),
    )  # fmt: skip
    assert [call[2] for call in redecode.calls] == [first]
    assert [p.reason for p in stand.log.promotions] == [PROMOTED, TOO_LONG]
    assert kept.version == 1 and [p.test_index for p in kept.pseudo_pairs] == [0]
    (plan,) = stand.log.plans
    assert plan["shape"][IN_CONTEXT]["screened"] == [0]
    assert plan["shape"][IN_CONTEXT]["too_long"] == [1]
    assert plan["shape"]["screened"] == [0, 1]
    # The price re-decodes the input whose pair fits (4 batches), not both that passed.
    in_context = plan["shape"][IN_CONTEXT]
    assert in_context["redecode_batches"] == 4
    expected = stand.gov.cost_model.round_costs(
        train_tokens=0, test_batches=4, gate_batches=4, redecode_batches=4,
        length_ratio=in_context["length_ratio"],
    )[IN_CONTEXT]  # fmt: skip
    assert plan["estimates"] == {IN_CONTEXT: round(expected, 3)}
    assert stand.log.rounds[0].estimate == pytest.approx(expected)
    ((_, gate_state, _, _),) = stand.gate.calls
    assert [p.kind for p in gate_state.pairs] == ["real", "real", "pseudo"]
    assert stand.log.contexts == []  # the gate held every pair it was given


def test_the_in_context_round_chooses_its_pairs_on_the_gates_prompt():
    """The gate decodes the held-out input, so the pairs are chosen on that prompt: here the
    pseudo-pair would fit beside the test input (4 cells) but not beside the held-out input (12),
    so the in-context round is not offered."""
    state = demos()
    draft = Pair(T0, Y0, kind="pseudo", round_added=1, test_index=0)
    gate_without = cells(state.without_held_out().task([C.input]))
    gate_with = cells(state.enlarged([draft]).without_held_out().task([C.input]))
    test_with = cells(state.enlarged([draft]).without_held_out().task([T0]))
    assert test_with <= gate_without < gate_with and (gate_without, gate_with) == (36, 44)
    (same, _), stand = run(state, guard=PromptGuard(limit=gate_with - 1, measure=cells))
    assert same is state and stand.log.plans == [] and stand.redecode.calls == []
    (record,) = stand.log.rounds
    assert (record.outcome, record.screened) == (TOO_LONG, (0,))
    assert [p.reason for p in stand.log.promotions] == [TOO_LONG]
    # One token more and the gate's prompt holds it: the round runs and is kept.
    (kept, _), roomy = run(state, guard=PromptGuard(limit=gate_with, measure=cells))
    assert kept.version == 1 and roomy.log.rounds[0].outcome == KEPT


def test_every_prompt_with_pseudo_pairs_holds_those_that_fit_and_a_drop_is_recorded(tmp_path):
    """A gradient round (the default variants) on three real 30 by 30 pairs and two test inputs:
    input 0 is promoted and trained on; the gate's prompts hold its pair, but the final decode's,
    with the held-out pair back, cannot: the decode gets the state without it, and a context
    record names it. The kept state still holds it (the adapter learned it)."""
    state = big_state(real=3, tests=2)
    first, second = state.test_inputs
    answers = {first: votes((square(5), 16)), second: votes((square(6), 4))}
    model = Model()
    decoder = Decoder(answers={second: votes((square(6), 4))})
    log = RoundLog(tmp_path, "rank2")
    gov = governor()
    kept, _ = run_rounds(
        state, model, rounds(), provider=InPlaceProvider(model), governor=gov,
        candidates=[answers[first], answers[second]], decode_tests=decoder.decode_tests,
        install=decoder.install,
        make_gate=lambda r: HeldOutGate(lambda s: None, GateDecode(right=state.held_out.output)),
        make_redecode=lambda r: Redecode(answers), deadline=gov.clock() + 1e6, budget=1e6,
        guard=max_seq_guard(), log=log,
    )  # fmt: skip
    assert kept.version == 1 and [p.test_index for p in kept.pseudo_pairs] == [0]
    ((_, decoded, name, held_out, inputs),) = decoder.decodes
    assert (name, held_out, inputs) == ("final", True, (1,))
    assert decoded.pseudo_pairs == () and decoded.version == 1
    assert [p.kind for p in decoded.pairs] == ["real", "real"]
    (context,) = log.contexts
    assert (context.name, context.prompts, context.queries) == ("final", HELD_OUT_PROMPTS, (1,))
    assert (context.kept, context.dropped, context.limit) == ((), (0,), 7261)
    assert context.lengths == (6534, 8400) and context.round_added == 1
    (line,) = [json.loads(x) for x in (tmp_path / "context-rank2.jsonl").read_text().splitlines()]
    assert line == context.to_record() and line["dropped"] == [0]
    assert log.path(CONTEXT_PREFIX) == tmp_path / "context-rank2.jsonl"
    # The decode record itself is P2.3's: the drop is in the context record only.
    (decode,) = log.decodes
    assert "dropped" not in decode.to_record() and decode.installed == (1,)


def test_the_guard_trims_a_round_two_gate_and_records_it(tmp_path):
    """The gate of a round sees the earlier rounds' pseudo-pairs and its own: with a limit that
    fits one, a gradient round 2 decodes its gate with round 1's pair only, recorded."""
    state = demos(T0, T1).enlarged([Pair(T0, Y0, kind="pseudo", round_added=1, test_index=0)])
    one = cells(state.without_held_out().task([C.input]))
    guard = PromptGuard(limit=one + 1, measure=cells)  # round 1's pair fits, no more
    log = RoundLog(None, "rank0")
    gate = GateDecode()
    (kept, _), stand = run(
        state, candidates=[AGREE[T0], AGREE[T1]], guard=guard, variants=(GRADIENT,),
        round_added=2, log=log, gate_decode=gate,
    )  # fmt: skip
    ((_, gate_state, _, _),) = gate.calls
    assert [p.test_index for p in gate_state.pseudo_pairs] == [0]
    (context,) = log.contexts
    assert (context.name, context.queries, context.kept, context.dropped) == (
        "round2-gate", None, (0,), (1,)
    )  # fmt: skip
    assert kept.version == 2 and [p.test_index for p in kept.pseudo_pairs] == [0, 1]
    # Round 2's re-decode of input 1 prompts with round 1's pair, which fits: no record.
    ((_, redecode_state, query, _),) = stand.redecode.calls
    assert query == T1 and [p.test_index for p in redecode_state.pseudo_pairs] == [0]


def test_the_guard_trims_a_round_two_redecode_and_records_it(tmp_path):
    """A round-2 re-decode prompts with round 1's pseudo-pair beside the real pairs: when the
    prompt fits only without it, the re-decode decodes with the real pairs alone (never one of
    them dropped), and a ``round2-redecode<i>`` context record names the pair it left out. The
    gate's prompts, longer, keep no pseudo-pair either (over the limit without any), recorded."""
    state = demos(T0, T1).enlarged([Pair(T0, Y0, kind="pseudo", round_added=1, test_index=0)])
    real = cells(state.first_pseudo(0).without_held_out().task([T1]))
    with_pair = cells(state.without_held_out().task([T1]))
    assert (real, with_pair) == (27, 35)
    log = RoundLog(tmp_path, "rank1")
    (kept, _), stand = run(
        state, candidates=[AGREE[T0], AGREE[T1]], guard=PromptGuard(limit=real, measure=cells),
        variants=(GRADIENT,), round_added=2, log=log,
    )  # fmt: skip
    ((_, redecode_state, query, _),) = stand.redecode.calls
    assert query == T1 and redecode_state.pseudo_pairs == ()
    assert redecode_state.pairs == (A, B) and redecode_state.version == state.version
    redecode, gate = log.contexts
    assert (redecode.name, redecode.prompts) == ("round2-redecode1", "pairs")
    assert redecode.queries == (1,)
    assert (redecode.kept, redecode.dropped, redecode.lengths, redecode.limit) == (
        (), (0,), (27, 35), 27
    )  # fmt: skip
    assert (gate.name, gate.kept, gate.dropped) == ("round2-gate", (), (0, 1))
    lines = (tmp_path / "context-rank1.jsonl").read_text().splitlines()
    assert [json.loads(line)["name"] for line in lines] == ["round2-redecode1", "round2-gate"]
    # The re-decode agreed, the fit trained on both pseudo-pairs, and the gate passed.
    assert kept.version == 2 and [p.test_index for p in kept.pseudo_pairs] == [0, 1]
    ((fitted, _, _),) = stand.provider.adapts
    assert [p.test_index for p in fitted.pairs if p.kind == "pseudo"] == [0, 1]


# The settings: Rounds.variants, the solver JSON, the refusals.


def rounds(**changes):
    return Rounds(round_dir="/kaggle/round_outputs/arm", governor=GOVERNOR, **changes)


def test_variants_default_to_the_gradient_round_and_keep_p2_3s_json():
    settings = rounds()
    assert settings.variants == DEFAULT_VARIANTS == (GRADIENT,)
    assert list(settings.to_json()) == list(Rounds.FIELDS) and Rounds.OPTIONAL == ("variants",)
    # Naming the default changes neither the settings nor the JSON.
    assert rounds(variants=["gradient"]) == settings
    assert rounds(variants=["gradient"]).to_json() == settings.to_json()


@pytest.mark.parametrize("variants", [[IN_CONTEXT], [GRADIENT, IN_CONTEXT]])
def test_variants_force_either_round_or_offer_both_through_the_solver_json(variants):
    settings = HoldOut(
        gate_dir="/kaggle/gate_outputs/a",
        log_dir="/kaggle/run_logs/a/gate",
        rounds={"round_dir": "/kaggle/round_outputs/a", "governor": GOVERNOR, "variants": variants},
    )
    assert settings.rounds.variants == tuple(variants)
    payload = json.loads(json.dumps(settings.to_json()))
    assert list(payload["rounds"]) == [*Rounds.FIELDS, "variants"]
    assert payload["rounds"]["variants"] == variants
    assert HoldOut.from_mapping(payload) == settings
    cfg = RunConfig(data_dir="/d", model_dir="/m", holdout=payload)
    assert json.loads(solver_command(cfg, 1.0, "python")[6])["holdout"] == payload


@pytest.mark.parametrize(
    ("variants", "error"),
    [
        ([IN_CONTEXT, GRADIENT], ValueError), ([GRADIENT, GRADIENT], ValueError), ([], ValueError),
        (["sgd"], ValueError), ("gradient", TypeError), ([1], TypeError), (None, TypeError),
    ],
)  # fmt: skip
def test_bad_variants_are_refused(variants, error):
    with pytest.raises(error, match="variants"):
        Rounds.from_mapping({"round_dir": "/r", "governor": GOVERNOR, "variants": variants})


def test_refine_refuses_the_in_context_round_without_its_provider_or_its_guard():
    with pytest.raises(ValueError, match="needs its provider"):
        run(in_context=None)
    with pytest.raises(TypeError, match="needs the length guard"):
        run(guard=False)
    with pytest.raises(TypeError, match="in_context must have a adapt method"):
        run(in_context=object())
    with pytest.raises(ValueError, match="variants must be"):
        run(variants=(IN_CONTEXT, GRADIENT))
    with pytest.raises(TypeError, match="guard must be None or a PromptGuard"):
        run(variants=(GRADIENT,), in_context=None, guard="7261")
    with pytest.raises(ValueError, match="run_rounds needs its provider"):
        run_rounds(
            demos(), Model(), rounds(variants=[IN_CONTEXT]), provider=InPlaceProvider(Model()),
            governor=governor(), candidates=[AGREE[T0]], decode_tests=None, install=None,
            make_gate=None, make_redecode=None, deadline=1.0, budget=1.0,
        )  # fmt: skip
    for bad in ({"limit": 0}, {"limit": 1.5}, {"measure": "cells"}):
        with pytest.raises((TypeError, ValueError)):
            PromptGuard(**{"limit": 10, "measure": cells, **bad})
    for value, error in ((0, ValueError), (2.0, TypeError)):
        with pytest.raises(error):
            PromptGuard(limit=10, measure=lambda entry, v=value: v).tokens({})


def test_refine_without_the_in_context_provider_is_p2_3s():
    """The default offers the gradient round alone: the plan, its shape, and the records carry
    P2.3's keys, nothing is measured, and no context record is written."""
    measured = []

    def measure(entry):
        measured.append(entry)
        return cells(entry)

    (_, _), stand = run(demos(T0, T1), in_context=None, variants=None, guard=None)
    (plan,) = stand.log.plans
    assert set(plan["estimates"]) == {GRADIENT} and plan["decision"] == GRADIENT
    assert list(plan["shape"]) == [
        "screened", "train_tokens", "train_steps", "test_batches", "gate_batches",
        "redecode_batches", "observed_batch",
    ]  # fmt: skip
    assert set(stand.log.rounds[0].to_record()) == {
        "puzzle_id", "round", "version", "next_version", "outcome", "screened", "promoted",
        "decision", "estimate", "remaining", "weights_before", "weights_after", "restored",
        "seconds", "error",
    }  # fmt: skip
    # A guard on the gradient round measures the gate's prompt (it carries the pseudo-pairs) and
    # drops nothing on these small grids.
    (_, _), guarded = run(demos(T0, T1), in_context=None, variants=None, guard=loose(measure))
    assert measured and guarded.log.contexts == []
    assert guarded.log.plans[0]["shape"] == plan["shape"]


@pytest.mark.parametrize(
    ("variants", "decision"),
    [((GRADIENT,), GRADIENT), ((IN_CONTEXT,), IN_CONTEXT), ((GRADIENT, IN_CONTEXT), GRADIENT)],
)
def test_run_rounds_offers_the_variants_of_its_settings(variants, decision):
    model = Model()
    gov = governor()
    log = RoundLog(None, "rank0")
    decoder = Decoder()
    in_context = InContextProvider(fingerprint=REFINE.InPlaceProvider(model).fingerprint)
    kept, adapter = run_rounds(
        demos(T0, T1), model, rounds(variants=list(variants)), provider=InPlaceProvider(model),
        governor=gov, candidates=[AGREE[T0], votes((Y1, 4))], decode_tests=decoder.decode_tests,
        install=decoder.install, make_gate=lambda r: HeldOutGate(lambda s: None, GateDecode()),
        make_redecode=lambda r: Redecode(), deadline=gov.clock() + 1e6, budget=1e6,
        in_context=in_context, guard=loose(), log=log,
    )  # fmt: skip
    (plan,) = log.plans
    assert tuple(plan["estimates"]) == variants and plan["decision"] == decision
    assert kept.version == 1 and log.rounds[0].decision == decision
    # The final decode (the default) decodes the open input with the pseudo-pair in its prompts.
    ((_, state, name, held_out, inputs),) = decoder.decodes
    assert (name, held_out, inputs, state.pseudo_pairs) == ("final", True, (1,), kept.pseudo_pairs)
    if decision == IN_CONTEXT:
        assert log.rounds[0].weights_before == log.rounds[0].weights_after


def test_two_in_context_rounds_chain_on_the_decode_after_the_first():
    """Round 2 runs after a kept in-context round 1, on the decode of the input still open, with
    round 1's pair in that decode's prompts and the adapter never changed."""
    model = Model()
    before = model.weights.copy()
    gov = governor()
    log = RoundLog(None, "rank0")
    decoder = Decoder(answers={T0: votes((Y0, 16)), T1: votes((Y1, 15), (W, 1))})
    state, adapter = run_rounds(
        demos(T0, T1), model, rounds(variants=[IN_CONTEXT], max_rounds=2, final_in_prompts=False),
        provider=InPlaceProvider(model), governor=gov, candidates=[AGREE[T0], votes((Y1, 4))],
        decode_tests=decoder.decode_tests, install=decoder.install,
        make_gate=lambda r: HeldOutGate(lambda s: None, GateDecode()),
        make_redecode=lambda r: Redecode(), deadline=gov.clock() + 1e6, budget=1e6,
        in_context=InContextProvider(), guard=loose(), log=log,
    )  # fmt: skip
    assert [r.outcome for r in log.rounds] == [KEPT, KEPT]
    assert [r.decision for r in log.rounds] == [IN_CONTEXT, IN_CONTEXT]
    assert state.version == 2 and [p.round_added for p in state.pseudo_pairs] == [1, 2]
    assert adapter is model and np.array_equal(model.weights, before)
    ((_, decoded, name, _, inputs), *_) = decoder.decodes
    assert (name, inputs) == ("round1", (1,)) and len(decoded.pseudo_pairs) == 1


def test_an_in_context_round_without_the_time_for_its_final_decode_is_planned_with_it():
    """The plan reserves the final decode for the in-context round too: with less time left than
    the round and its final decode cost, the round is declined and the final decode skipped."""
    clock = Clock(1000.0)
    gov = governor(end=1000.0 + 600.0, clock=clock)
    model = Model()
    log = RoundLog(None, "rank0")
    decoder = Decoder()
    run_rounds(
        demos(T0, T1), model, rounds(variants=[IN_CONTEXT]), provider=InPlaceProvider(model),
        governor=gov, candidates=[AGREE[T0], AGREE[T1]], decode_tests=decoder.decode_tests,
        install=decoder.install, make_gate=lambda r: HeldOutGate(lambda s: None, GateDecode()),
        make_redecode=lambda r: Redecode(), deadline=gov.clock() + 600.0, budget=600.0,
        in_context=InContextProvider(), guard=loose(), log=log,
    )  # fmt: skip
    (plan,) = log.plans
    assert plan["decision"] == NONE and plan["shape"]["test_batches"] == 8
    assert log.rounds[0].outcome == DECLINED and log.decodes[0].skipped == NO_TIME


# The records.


def test_the_reasons_and_outcomes_name_the_guard():
    assert TOO_LONG in REASONS and TOO_LONG in OUTCOMES
    assert REASONS.index(TOO_LONG) == REASONS.index(loop.VIEWS_NOT_DISJOINT) + 1
    record = ContextRecord(PUZZLE, 1, "final", HELD_OUT_PROMPTS, (1,), (), (0,), (6534, 8400), 7261)
    assert record.to_record() == {
        "puzzle_id": PUZZLE, "round": 1, "name": "final", "prompts": "held_out", "queries": [1],
        "kept": [], "dropped": [0], "lengths": [6534, 8400], "limit": 7261,
    }  # fmt: skip
    gate = ContextRecord(PUZZLE, 2, "round2-gate", PAIRS_PROMPTS, None, (0,), (1,), (1, 2, 3), 2)
    assert gate.to_record()["queries"] is None
    lines = []
    log = RoundLog(None, "rank0", say=lines.append)
    log.context(record)
    assert lines == [
        f"round 1 for {PUZZLE}: the prompts of final left out the pseudo-pairs of test inputs "
        "[0] (8400 tokens with every one, limit 7261)"
    ]


def test_a_rollback_that_did_not_restore_keeps_p2_3s_rule_for_the_gradient_round():
    """The in-context exception to restored (None without fingerprints) leaves the gradient
    round's rule as it was: a restore whose fingerprints differ is not restored."""
    assert loop._restored("a", "a", False) is True and loop._restored("a", "b", False) is False
    assert loop._restored(None, None, False) is False and loop._restored(None, None, True) is None
    assert loop._restored("a", "a", True) is True and loop._restored("a", None, True) is False
    assert NOT_RESTORED == "not_restored"
