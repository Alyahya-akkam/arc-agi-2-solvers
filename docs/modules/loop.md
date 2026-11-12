# Module: `arc26/loop.py`

Tasks P2.1 (BRIEF B5 P2.1, B2 Q4, B8 "Gate cost and gate price"), P2.2 (B5 P2.2, promotion),
P2.3 (B5 P2.3, the rounds and their rollback), and P2.5 (B5 P2.5, the in-context round and the
length guard); ADR-004 (the `DemoSet`), ADR-011 (the split solver it calls), ADR-013 (what the
held-out pair may reach), ADR-014 (promotion), ADR-015 (round mechanics), ADR-016 (the in-context
round, the length guard). Tests: `tests/test_gate.py`, `tests/test_promote.py`,
`tests/test_refine.py`, `tests/test_rounds_solver.py`, `tests/test_in_context.py`,
`tests/test_in_context_solver.py`, `tests/test_split_solver.py` (the worker's branch,
`decode_puzzle`'s `view_seed`), `tests/test_runner.py` (`RunConfig.holdout`). Importing it needs
the standard library, `arc26.timing`, and `arc26.governor` only; `vote` and `tally` import numpy
through the moved selection cell, the view checks the moved `arc_loader` (numpy, and
transformers for an annotation), and `solve_puzzle` the GPU stack.

## Purpose

The refinement loop (P2.2, P2.3) promotes a test prediction to a pseudo-pair only if the adapter
trained with it still solves a real pair it never saw. P2.1 builds that check, the held-out gate,
and the loop's state, the `DemoSet`, and wires both into the split solver so that run `P2.1-01`
can measure the gate's price: holding out one of $k$ pairs removes $1/k$ of the supervision, and
P2.3 cannot start until that price is known (B5).

P2.2 builds the decision to promote, `promote`: a test input's voted answer becomes a pseudo-pair
only when three independent checks agree that it is right, since a wrong pseudo-pair teaches the
adapter a wrong rule. P2.3 builds the rounds around it (`refine`, `run_rounds`): the governor
prices a round before it runs, the candidate adapter continues from the current one, a failing
gate restores the previous weights bit for bit, and after the rounds the test inputs that are
still open are decoded once more, by default with the held-out pair back in the prompts: the
decode that is submitted keeps the pair, every decode that feeds a promotion leaves it out
(ADR-013 as amended after `P2.1-02`).

P2.5 adds the round B2 Q4 keeps as the fallback, the in-context round: the promoted pairs reach
the prompts and no gradient step is taken, so it costs no fit, and the governor picks it when
the gradient round does not fit the time left. It adds the length guard every prompt with
pseudo-pairs needs (B5: "no unbounded appends"): the reference cuts a prompt over its limit by
dropping demonstrations from the front of each view's shuffled order, so a pseudo-pair joins a
prompt only where the prompt still fits, and a real pair is never lost to one (ADR-016).

## Math

A puzzle has real demonstration pairs $P = (p_0, \dots, p_{k-1})$, $p_i = (x_i, y_i)$, in the
challenges file's order, and test inputs $t_1, \dots, t_m$. The hold-out rule $\rho$ picks an
index $h = \rho(P)$; B5's rule, the default, is the last pair, $h = k - 1$ (`"first"`, $h = 0$, and
any callable are the parameter's other values). It applies when

$$k \ge 3 \quad \text{or} \quad (k = 2 \text{ and the flag } k_2),$$

with $k_2$ on by default (B5: ablated in P2.7); otherwise no pair is held out and the puzzle runs
as the reference. The training and voting set is $D = (p_i)_{i \ne h}$, the held-out pair
$p_h = (x_h, y_h)$.

**Training.** $\theta_D = \mathrm{fit}(D)$: the reference's test-time training (`arc26.ttt.fit`,
`docs/modules/ttt.md`) on 128 augmented views of $D$ alone, each view one conversation of the
$k - 1$ pairs, loss on their output grids.

**The gate.** The held-out input is decoded with $\theta_D$ under $n$ augmented views
(`decode_puzzle`: $n = 16$, the 8 dihedral transforms with 2 color permutations each), each prompt
holding $D$ as its demonstrations and $x_h$ as its query. Each view yields candidates $c$ (parsed
DFS beams, mapped back by the view's inverse), each with its rescoring NLLs $s_1(c), \dots, s_8(c)$
under 8 more views. For a distinct grid $g$ with candidates $C_g$, the reference's submission
selection (`score_kgmon`, cell 3) scores

$$\mathrm{score}(g) = |C_g| - \frac{1}{|C_g|} \sum_{c \in C_g} \frac{1}{8} \sum_{i=1}^{8} s_i(c),$$

each view that produced $g$ adding 1 and the mean rescoring NLL subtracted (so a grid that fewer
views produced can win when the rescoring views find it much likelier); grids with equal scores
keep the candidates' order. The voted answer is $\hat y = \arg\max_g \mathrm{score}(g)$,
attempt 1 of the submission for that input, and the gate passes iff $\hat y = y_h$ cell for cell
(shape included); no candidate fails.

**The price.** On DEV40, with $S$ the score in tasks (per-task mean over test outputs, two
attempts), $\Delta = S_\text{on} - S_\text{off}$ for the same settings with and without the
hold-out; run `P2.1-01` measures $S_\text{on}$ (`docs/prerun/P2.1-01.md`).

**Promotion (P2.2, ADR-014).** In round $r \ge 1$ the state is $D$ (version $v$), the adapter
$\theta$, and for each test input $x_i$ the view candidates $C^{(i)}$ of one `decode_puzzle` call
over all test inputs, $D$ in the prompts (ADR-013). With $\hat y_i$ the voted answer and $n = 16$,
the agreement is the count term of $\mathrm{score}$, the share of the views that produced it:

$$a_i = \frac{|C^{(i)}_{\hat y_i}|}{n} \in [0, 1].$$

$\hat y_i$ is promoted iff, in this order (cheapest first; the first failure stops it):

1. $C^{(i)} \ne \emptyset$; the top two grids do not tie in $\mathrm{score}$; $a_i \ge \tau$
   ($\tau = 9/16$ by default, a strict majority of the views); $x_i$ is not already a
   pseudo-pair's input; a pair is held out (otherwise there is no gate).
2. **Re-decode.** With the same $\theta$, $x_i$ alone is decoded under views $V'$ (`decode_puzzle`'s
   `view_seed` $= s$, its 16 decoding and 8 rescoring views), where $s$ is the first seed from 3
   upward with $\phi(V') \cap \phi(V) = \emptyset$: $V$ is the first decode's 16 decoding and 8
   rescoring views of $x_i$, and $\phi(v)$ the image under view $v$ of the prompt's grids
   ($D$'s inputs and outputs, then $x_i$). The re-decode's voted answer is $\hat y_i$, untied.
3. **Gate with the candidate adapter.** With $I$ the inputs that passed 1 and 2,
   $\theta' = \mathrm{adapt}(D \cup \{(x_i, \hat y_i) : i \in I\}, \theta)$ (Phase 2:
   `sgd_continue`) passes the gate, its prompts holding the enlarged set.

If the gate passes, $D' = D \cup \{(x_i, \hat y_i) : i \in I\}$ is version $v + 1$ and $\theta'$
the round's adapter; otherwise $D$ and $\theta$ stay (ADR-004). Comparing views by $\phi$ is
stricter than comparing (transform, permutation) pairs, which have equal images when equal; it
also counts as one view two permutations that agree on the puzzle's colors. A re-decode that
ran fewer DFS batches than its 16 views need (the 1200 s cap or the end time) fails
(`redecode_cut`, P2.3).

**Rounds (P2.3, ADR-015).** Round $r$ starts from $(D_{r-1}, \theta_{r-1})$ and the candidates
$C_{r-1}$ of the decode that $\theta_{r-1}$ made with $D_{r-1}$ in the prompts (round 0's test
decode for $r = 1$). With $I_0$ the inputs that pass the free checks (step 1 above, no GPU work),
the governor prices the round as

$$C = (1 + m)\left(\tau_\text{train}(P, s') + o\right) + D(4|I_0| + 4 + 4 f, 1, s),
\qquad P = \left\lceil T' \, s' / 128 \right\rceil,$$

the re-decodes, the gate, and the test decodes that may follow the round, $f$ inputs' worth: with
the final decode (`final_in_prompts`, the default), $f = n_\text{open}$ at the last round (the
final decode follows whatever the round does, and a round that keeps nothing leaves every input
open) and $f = \max(n_\text{open}, 2(n_\text{open} - 1))$ before it (a kept round's own decode
too); without it, $f = n_\text{open} - 1$, the decode after a kept round of the inputs still open
(a kept round promotes at least one of the $n_\text{open}$ open now). All at the slowest round-0
batch $s$ (`docs/modules/governor.md`), with $T'$ the tokens of the 128 training views of
$D_{r-1} \cup \{(x_i, \hat y_i) : i \in I_0\}$, of which $s'$ steps at batch 1 train on $s'$.
The round runs when $C$ fits the time left to the puzzle's deadline. The candidate fit is Phase
2's $\theta' = \mathrm{sgd\_continue}(\theta_{r-1}, D', s')$: the reference's fit started from a
copy $\bar\theta = \theta_{r-1}$ instead of the default weights, $s' = 32$ steps. A failing gate
restores $\bar\theta$, so the weights after the round equal $\theta_{r-1}$ bit for bit, and the state
stays $D_{r-1}$ at its version. After a kept round, $\theta_r$ decodes the test inputs that are
not pseudo-pairs, in one call with $D_r$ in the prompts, and they take these candidates; the
others keep the candidates that promoted them. Round 2 (`max_rounds` 2) runs only after a kept
round 1, its view check on that call's layout (`first_inputs`).

**The in-context round (P2.5, ADR-016).** The same round with a provider that fits nothing:
$\theta' = \theta_{r-1}$. Its pairs $P \subseteq \{(x_i, \hat y_i) : i \in I_0\}$ are those the
gate's prompts can hold (the guard below), the others recorded `too_long`; each is re-decoded,
and the gate decodes $x_h$ with $\theta_{r-1}$ and $D_{r-1} \cup P'$ in its prompts ($P'$: the pairs
whose re-decode agreed). A passing gate keeps $D_r = D_{r-1} \cup P'$ at version $v + 1$ with
$\theta_r = \theta_{r-1}$; a failing one keeps $D_{r-1}$, and there is nothing to restore. The
governor prices it as

$$C_\text{in-context} = (1 + m)\,o + D(4|P|, 1, s) + D(4 + 4f, \rho, s),
\qquad \rho = \max_{q \in \{x_h\} \cup O} \frac{L(D_{r-1} \cup P, q)}{L(D_{r-1}, q)},$$

$O$ the open test inputs, $L$ the guard's measure, $f$ as above, and picks it when
$C_\text{gradient} > R \ge C_\text{in-context}$ (both offered), or whenever it fits (forced,
`Rounds.variants`).

**The length guard (P2.5, ADR-016).** A prompt's tokens may be at most
$\ell = $ `max_seq_length` $-$ `max_new_tokens` ($8192 - 931 = 7261$ under the reference's
tokenizer). For a decode whose state holds pseudo-pairs $p_1, \dots, p_n$ (in the order added),
with $L_j$ the tokens of its longest prompt holding the real pairs (and the held-out pair, for the
final decode) and $p_1, \dots, p_j$, over its views (every dihedral transform: each color one
token, so only the orientation matters), its rescoring's, and its queries, the prompts hold

$$p_1, \dots, p_k, \qquad k = \max\{j \le n : L_i \le \ell \ \forall\, i \le j\}, \qquad
k = 0 \text{ if } L_0 > \ell,$$

so `cut_to_len` cuts nothing it would not cut without them. At $30 \times 30$ a pair is 1,866
tokens and a query 936: two real pairs and the query take 4,668 tokens, so one pseudo-pair fits
(6,534) and a second does not (8,400); with three real pairs, none does.

## Interfaces

| Name | What |
|---|---|
| `Pair(input, output, kind="real", round_added=0, agreement=None, test_index=None, verdicts=None)` | B4's pair; `kind` `real`, `held_out`, or `pseudo` (round at least 1, P2.2); a promoted pseudo-pair carries its provenance: round, agreement, test input, and `Verdicts`; grids stored as tuples of row tuples (`as_grid`) |
| `DemoSet(puzzle_id, pairs, held_out, test_inputs, version=0, held_out_index=None)` | the versioned state; `pairs` is the training and voting set and never holds the held-out pair; immutable (a promotion builds version $+1$, rollback is a pointer swap, ADR-004) |
| `DemoSet.from_task(puzzle_id, task, *, rule="last", k2=True)` | from a challenges entry, the rule applied |
| `DemoSet.without_held_out()` | the state the fit and decode callables receive |
| `DemoSet.task(test_inputs=None, *, held_out=False)` | the reference's task entry from `pairs` only: every dataset the solver trains or decodes on is built here; `held_out=True` puts the held-out pair back at its place among the real pairs, for the final decode (P2.3, `final_in_prompts`, the default) |
| `DemoSet.enlarged(pairs)` | the next version: pseudo-pairs appended, version $+1$ (P2.2) |
| `DemoSet.pseudo_pairs`, `DemoSet.first_pseudo(count)` | P2.5: the pseudo-pairs in the order added; the state as one decode's prompts hold it, every real pair in place and the first `count` pseudo-pairs, the version and held-out pair kept |
| `held_out_index(pairs, rule="last", k2=True)` | the rule: an index, or None when the puzzle keeps every pair |
| `HOLDOUT_RULES` | `{"last", "first"}`, names for configs; Python callers may pass a callable |
| `Candidate(grid, beam_score, score_aug, view="")` | one view's decoded answer |
| `vote(candidates, select=None)` | the distinct grids, best first, by `score_kgmon` (or `select`) |
| `tally(candidates, views=16) -> Tally` | `score_kgmon`'s vote with its terms (P2.2): `grids`, `counts` $\lvert C_g \rvert$, `scores`, `views`; `voted`, `agreement` ($\lvert C_{\hat y} \rvert / n$), `tied` |
| `HeldOutGate(fit, decode, *, views=16, select=None, clock=time.perf_counter)` | B4's `Gate`; `fit(demos)` trains on `pairs` only; `verdict(adapter, demos)` decodes $x_h$ and votes; `passes(adapter, demos)` is `verdict(...).passed` |
| `GateVerdict` | `puzzle_id`, `applied`, `passed`, `real_pairs`, `held_out_index`, `views`, `candidates`, `votes` (view candidates per distinct grid, in vote order), `expected_rank` (place of $y_h$ in the vote, None when absent), `seconds`, `error`; `to_record()`, `not_applied(demos)`, `raised(demos, error, seconds)` |
| `HoldOut(gate_dir, log_dir=None, rule="last", k2=True, in_test_prompts=False, rounds=None)` | a run's settings, the solver JSON's `holdout` (`from_mapping`, `from_cfg`, `to_json`); `rounds` (P2.3, `Rounds`) joins the JSON only when set, and needs `in_test_prompts` False |
| `solve_puzzle(holdout, model, ..., *, queue_size=None, workers=1)` | the split solver's per-puzzle body with a held-out pair; takes the 16 worker locals that `fit` and `decode_puzzle` read, and with rounds the queue's size and the worker count (P2.3); returns the model |
| `RenamedOps(logger, names)`, `GATE_OPS` | records `decode` and `selection` as `gate_decode` and `gate_selection` |
| `read_candidates(folder)`, `VerdictLog(log_dir, worker, prefix="gate-")` | the gate's candidate files read back in name order; one verdict per line in `gate-<worker>.jsonl`, or one promotion record per line in `promote-<worker>.jsonl` (`prefix=PROMOTION_PREFIX`) |
| `promote(candidates, gate, adapter, demos, tau=TAU, *, adapt, redecode, round_added=1, fresh_seed=3, first_seed=None, views=16, tries=64, timing_logger=None, clock=time.perf_counter) -> Promotion` | B4's `promote` (P2.2): the three checks per test input, cheapest first; one candidate adapter for every input that passes the first two |
| `Promotion` | `demos` (version $+1$ with the promoted pairs, or the state given), `adapter` (the candidate adapter, or None), `pairs` (B4's return value), `records`, `gate`; `promoted` |
| `PromotionRecord` | per test input: `puzzle_id`, `round_added`, `test_index`, `candidates`, `votes`, `agreement`, `tau`, `tied`, `views`, `redecode`, `gate` (None when not reached), `fitted` (the inputs of the candidate fit), `promoted`, `reason` (one of `REASONS`), `error` (a raising fit); `to_record()` |
| `view_keys(key, task, index, seed=None)` | the 16 decoding and 8 rescoring view keys `decode_puzzle` uses for test input `index` of `task` with `view_seed=seed`, from the moved `ArcDataset` calls; numpy's state kept |
| `view_signatures(keys, grids)` | each view's image of the grids (moved `forward_mod`), as a set |
| `fresh_views(demos, index, *, first_seed=None, fresh_seed=3, tries=64) -> ViewCheck` | the first seed whose re-decode of the input shares no image with the first decode; `ViewCheck(seed, tries, first, fresh, common)`, `disjoint` |
| `RedecodeVerdict`, `Verdicts` | the re-decode's `candidates`, `votes`, `expected_rank`, `tied`, `agreed`, `seconds`, `error`, `cut` (P2.3); a promoted pair's `views`, `redecode`, `gate` |
| `TAU`, `FRESH_SEED`, `VIEW_SEED_TRIES`, `REASONS` | $9/16$; 3; 64; `promoted`, `no_candidates`, `tied`, `agreement_below_tau`, `already_promoted`, `no_held_out`, `views_not_disjoint`, `too_long` (P2.5: the in-context round's prompts cannot hold the pair), `declined` (P2.3, `refine`), `redecode_raised`, `redecode_cut` (P2.3), `redecode_disagrees`, `fit_raised`, `gate_raised`, `gate_failed` |
| `REDECODE_OPS`, `CANDIDATE_OPS` | `RenamedOps` maps for the promotion's GPU ops: `decode` and `selection` to `redecode_decode` and `redecode_selection`; `train` to `candidate_train` |
| `screen(candidates, demos, tau=TAU, *, round_added=1, fresh_seed=3, first_seed=None, views=16, tries=64, first_inputs=None) -> tuple[Screened, ...]` | promote's free checks per test input, no GPU work (P2.3: they price the round); `promote` runs them through it |
| `Screened` | `index`, `tally`, `record` (final with its reason unless `passed`), `check` (the re-decode's views), `passed` |
| `DecodeCut(candidates, batches, expected)` | raised by a round's GPU decode callable that ran fewer DFS batches than its views need; a re-decode fails with `redecode_cut`, a gate with `gate_raised` |
| `refine(demos, adapter, provider, gate, governor, *, candidates, redecode, deadline, budget, observed_batch=None, round_added=1, tau=TAU, fresh_seed=3, first_seed=None, views=16, tries=64, first_inputs=None, decode_batches=None, in_context=None, variants=None, guard=None, log=None, timing_logger=None, clock=time.perf_counter) -> (DemoSet, adapter)` | B4's `refine` (P2.3): one round, kept (version $+1$, the candidate adapter) or rolled back (`demos` itself, `provider.rollback`); the free checks' records are written in every case; `decode_batches`, the DFS batches of the test decodes that may follow the round, which its plan prices (None: the decode after a kept round, 4 per input still open after it). P2.5: `in_context` (an `InContextProvider`) offers the in-context round beside the gradient one, `variants` restricts the offer (None: the gradient round, and the in-context one too when its provider is given), `guard` (a `PromptGuard`, needed with the in-context round) trims the re-decodes' and the gate's prompts |
| `InContextProvider(fingerprint=None)` | P2.5, B4's `AdapterProvider` without a fit: `adapt` returns `init`, `rollback` returns `previous`, `fingerprint` digests with the function given (the gradient provider's) or returns None, `fit_shape` is `(0, 0)` |
| `context_fit(lengths, limit) -> int` | P2.5, the guard, pure: how many pseudo-pairs, in order, a prompt holds; `lengths[j]` the longest prompt with the first `j` |
| `longest_prompt(entry, formatter) -> int` | P2.5: the tokens of the longest prompt `decode_puzzle` and its rescoring build from a task entry (the moved `get_length(..., max_of_transposed=True)`); numpy's state untouched |
| `PromptGuard(limit, measure)`, `PromptFit` | P2.5: `fit(demos, queries, held_out=False) -> PromptFit` (`kept`, `dropped`: test indices; `lengths`, `limit`); `tokens(entry)` checks the measure's count |
| `run_rounds(demos, adapter, settings, *, provider, governor, candidates, decode_tests, install, make_gate, make_redecode, deadline, budget, observed_batch=None, in_context=None, guard=None, log=None, timing_logger=None, clock=...) -> (DemoSet, adapter)` | up to `settings.max_rounds` rounds and the test decodes of the open inputs after them, `decode_tests(adapter, state, name, held_out, inputs) -> Decoded`, each round's plan pricing the decodes that may follow it (the final decode included); returns the last kept state and its adapter. P2.5: each round offers `settings.variants` (`in_context` needed when they hold the in-context round), and with `guard` every test decode receives the state as its prompts hold it |
| `fresh_views(..., first_inputs=None)`, `promote(..., first_inputs=None)` | P2.3: the test inputs the first decode decoded together, in increasing order; None for every input, round 0's decode |
| `renumber_candidates(folder, key, inputs)` | P2.3: names a decode of some inputs' files by the puzzle's test indices, `<task>_<inputs[j]>.<view>` |
| `AdapterProvider` | B4's protocol with what a round needs: `adapt(demos, init, budget_s)`, `rollback(previous)`, `fingerprint(adapter)`, `fit_shape(demos) -> (tokens, steps)`; Phase 2's is `arc26.ttt.SGDProvider` |
| `Rounds(round_dir, governor, max_rounds=1, tau=TAU, continue_steps=32, final_in_prompts=True, variants=("gradient",))` | a run's round settings, `HoldOut.rounds` (`from_mapping`, `to_json`); `governor` is the parsed `configs/governor.yaml` (policy `reference` only); `final_in_prompts` true by default: ADR-013 as amended after `P2.1-02`'s Recovers row (7.0 against 5.0 of 40) keeps the held-out pair in the decode that is submitted; `variants` (P2.5, `OPTIONAL`): `("gradient",)`, `("in_context",)`, or both in the governor's order, in the JSON only when not the default |
| `ContextRecord` | P2.5: one line per decode whose prompts left out pseudo-pairs (`name`, `prompts`, `queries`, `kept`, `dropped`, `lengths`, `limit`), in `context-<worker>.jsonl`; none when nothing was dropped |
| `RoundRecord`, `DecodeRecord` | one line per round (`outcome` in `OUTCOMES`, the screened and promoted inputs, the plan's decision (`in_context` too, P2.5), estimate, and time left, `weights_before`, `weights_after`, `restored`, `seconds`, `error`) and per test decode after the rounds (`name`, `prompts`, `inputs`, `installed`, `batches`, `expected`, `cut`, `estimate`, `remaining`, `skipped`, `seconds`, `error`); `skipped` is `all_promoted` (no open input), `not_restored` (the final decode after a rollback whose restore did not give back the weights), or `no_time` (the final decode's own check) |
| `RoundLog(log_dir, worker, say=None)` | the round records of one worker, `plan-`, `promote-`, `round-`, `decode-`, and (P2.5) `context-<worker>.jsonl` beside the verdicts, kept in memory too; `say` prints one line per record |
| `OpWatch(logger, names=None)` | a timing logger that counts and times the ops it forwards (renamed as `RenamedOps`): the DFS batches a decode ran and its slowest; with `logger` None it records nothing |
| `Decoded` | a test decode after a round: `candidates` per test input of the puzzle (empty when not decoded), `batches`, `expected`, `slowest`; `cut` |
| `read_puzzle(folder, key, inputs)`, `install_candidates(source, outputs, key, indices, backup=None)` | one puzzle's candidate files per test input; replacing some inputs' files in the run's folder (new ones copied in first, an input without a new candidate keeping its own, round 0's copied to `backup` once) |
| `DEFAULT_ROUNDS`, `MAX_ROUNDS`, `CONTINUE_STEPS`, `EPOCH_VIEWS`, `VIEWS_PER_BATCH`, `OUTCOMES` | 1; 2; $s' = 32$; 128; 4; `kept`, `rolled_back`, `not_promoted`, `declined`, `nothing_to_promote`, `no_held_out`, `too_long` (P2.5: only the in-context round was offered and none of its pairs fits), `raised` |
| `DEFAULT_VARIANTS`, `TOO_LONG`, `CONTEXT_PREFIX` | P2.5: `("gradient",)`; `too_long`; `context-` |
| `ROUND`, `ROUND_OPS`, `FINAL_OPS`, `SNAPSHOT`, `RESTORE`, `FINGERPRINT` | the round's op names: `round`; `decode` and `selection` of the decode after a kept round as `round_decode` and `round_selection`, of the final decode as `final_decode` and `final_selection`; the provider's `snapshot`, `restore`, and `fingerprint` |

The callables, injected so that the tests run on CPU (ADR-004, B4):

```python
fit(demos: DemoSet) -> adapter                       # demos.held_out is None
decode(adapter, demos: DemoSet, query: Grid, views: int) -> list[Candidate]   # demos.held_out is None
# P2.2, promote:
adapt(demos: DemoSet, init: adapter) -> adapter      # the enlarged state, held_out None; init the current adapter
redecode(adapter, demos: DemoSet, query: Grid, seed: int) -> list[Candidate]   # demos.held_out is None
```

`redecode` must decode `query` as `fresh_views` computed its views: alone, as test input 0 of a
one-key dataset keyed by the puzzle id, `ArcDataset(queries={demos.puzzle_id:
demos.task([query])})`, through `decode_puzzle(..., view_seed=seed)`, with its records through
`RenamedOps(logger, REDECODE_OPS)` and its candidate files in a folder of its own.

## Data flow

`arc26.solver.worker` reads `HoldOut.from_cfg(cfg)` once (`RunConfig.holdout`, passed in the
solver JSON only when set). Without it each puzzle runs `fit` and `decode_puzzle` as moved
(ADR-011). With it, each puzzle runs `solve_puzzle`:

1. `DemoSet.from_task(key, arc_test_set.queries[key], rule, k2)`. A puzzle the rule keeps whole
   runs the reference's two calls and logs a verdict with `applied` false.
2. `HeldOutGate.fit`: `arc26.ttt.fit` on `ArcDataset({key: demos.task()})`, the one-puzzle
   dataset of $D$ (ADR-011: "training on a changed pair set means passing another `ArcDataset` as
   `arc_test_set`"). Records `train`.
3. The test decode, first and on the puzzle's clock, as the reference decodes: `decode_puzzle` on
   the dataset `fit` returned ($D$ as demonstrations, ADR-013), or on the puzzle as loaded with
   `in_test_prompts`; candidates into the run's `outputs_dir`. Records `decode` and `selection`.
4. The gate: `decode_puzzle` on `ArcDataset({key: demos.task([x_h])})` with its own clock (the
   1200 s cap from the gate's start) and the run's `end_time`, into
   `<gate_dir>/<key>/`, never `outputs_dir`, so no gate file reaches the submission; its records
   go through `RenamedOps` as `gate_decode` and `gate_selection`. The files are read back
   (`read_candidates`), voted, and compared. Records `gate` over the whole check.
5. The verdict goes to `<log_dir>/gate-rank<rank>.jsonl`; the session log gets
   `[Rank r] gate for <task>: decoding the held-out pair, index i of k real pairs` before the
   decode and `passed|failed, N candidates, the held-out output ranked R` (or `not among them`)
   after it.

A gate that raises (an out-of-memory error on a large held-out input, say) is logged with its
error (`passed` null) and the worker goes on: `torch.multiprocessing.spawn` stops every worker
when one raises, and the test candidates are written by then. Like any failed op it writes no
timing record.

**Every training path.** At test time the reference trains only in `fit`, on the views of the
pairs it is given; it has no leave-one-out task generation (the training text of a view is its
pairs as one conversation, `fmt_train(..., last_is_challenge=True)`). So the held-out pair is
excluded from every training path by never being in a dataset handed to `fit`, and from
leave-one-out generation trivially. P3's episodes (`arc26.hyper.episodes`) are offline
meta-training, not test time. The gate's own prompts hold $D$ only, the held-out input as the
query.

**Timing records** (`arc26.timing`, per worker): per puzzle `train`, the test inputs' `decode`
and `selection`, then `gate_decode` (per DFS batch of the held-out input, prompt tokens) and
`gate_selection` (per rescored candidate), `gate` (null tokens), and `puzzle`. The governor
(P2.4) prices the gate's batches as `CostModel.round_costs(gate_batches=...)`.
`scripts/timing_report.py` counts the gate ops as overhead in its per-puzzle table and fits them
in its cost-model section.

**Verdict record**, one JSON line per puzzle:

```json
{"puzzle_id": "136b0064", "applied": true, "passed": false, "real_pairs": 3,
 "held_out_index": 2, "views": 16, "candidates": 21, "votes": [9, 7, 3, 2],
 "expected_rank": 2, "seconds": 284.511, "error": null}
```

No grid is logged: the verdict carries counts and ranks only.

### Promotion (P2.2)

P2.3's round (`refine`) calls `promote` after round 0's gate, or after the decode that follows
a kept round, with the GPU callables built as `solve_puzzle` builds the gate's (the contract
above). For each test input it tallies the candidates (`tally`), applies the free checks and
finds a fresh view seed (`screen`, `fresh_views`, CPU, milliseconds), and re-decodes; then it
fits one candidate adapter with every pseudo-pair that passed and checks it with `gate.verdict`.
A re-decode, fit, or gate that raises fails its check: its error is in the record (the fit's in
`error`, the others' in their verdicts), the traceback is printed, and the worker goes on. A
re-decode cut short by the time cap fails as `redecode_cut` (P2.3).

**Timing records.** With `timing_logger`, `promote` records the span of each re-decode,
`redecode`, and of the candidate adapter's gate check, `gate`; a check that raises writes none.
The GPU callables record the ops inside them under their own names through `RenamedOps`:
`redecode_decode` (per DFS batch of the re-decode, prompt tokens), `redecode_selection` (per
rescored candidate), `candidate_train` (the candidate fit's trainer, tokens), and the gate's
`gate_decode` and `gate_selection`. The governor (P2.4) prices a round with promotion as 4 DFS
batches and their rescoring per re-decoded input (`CostModel.decode_cost`), one fit on the
enlarged set (`train`'s model), the gate's batches (`round_costs(gate_batches=...)`), and, from
P2.3, the test decodes that may follow the round (`test_batches`, below).

**Promotion record**, one JSON line per test input in `promote-<worker>.jsonl` (`VerdictLog`
with `prefix=PROMOTION_PREFIX`); `views`, `redecode`, and `gate` are null when not reached:

```json
{"puzzle_id": "0a0b0c0d", "round": 1, "test_index": 0, "candidates": 16, "votes": [12, 4],
 "agreement": 0.75, "tau": 0.5625, "tied": false,
 "views": {"seed": 3, "tries": 1, "first": 24, "fresh": 24, "common": 0},
 "redecode": {"candidates": 16, "votes": [14, 2], "expected_rank": 1, "tied": false,
              "agreed": true, "seconds": 301.2, "error": null, "cut": false},
 "gate": {"puzzle_id": "0a0b0c0d", "applied": true, "passed": true, "real_pairs": 3,
          "held_out_index": 2, "views": 16, "candidates": 16, "votes": [16],
          "expected_rank": 1, "seconds": 290.4, "error": null},
 "fitted": [0], "promoted": true, "reason": "promoted", "error": null}
```

### Rounds (P2.3)

With `HoldOut.rounds`, `solve_puzzle` takes the puzzle's budget from the governor first
(`Governor.from_mapping(rounds.governor, end_time, workers)`, `puzzle_budget(queue_size)`; the
deadline is the puzzle's start plus it), watches round 0's test decode for its slowest DFS batch
(`OpWatch`, which forwards the records unchanged), and after round 0's gate runs `run_rounds`
with `arc26.ttt.SGDProvider` and these GPU callables, each a `decode_puzzle` call on a one-key
dataset with a clock of its own and the run's end time, into its own folder of
`<round_dir>/<task>/` (cleared at the puzzle's start):

| Callable | Decodes | Folder | Ops | Cut |
|---|---|---|---|---|
| round $r$'s re-decode | $x_i$ alone, `view_seed` | `round<r>-redecode<i>/` | `redecode_decode`, `redecode_selection` | `DecodeCut`, `redecode_cut` |
| round $r$'s gate | $x_h$ alone, $D'$ in the prompts | `round<r>-gate/` | `gate_decode`, `gate_selection` | `DecodeCut`, `gate_raised` |
| the decode after kept round $r$ | the open test inputs, $D_r$ in the prompts | `round<r>/` | `round_decode`, `round_selection` | nothing installed |
| the final decode (`final_in_prompts`, the default) | the open test inputs, the held-out pair back | `final/` | `final_decode`, `final_selection` | nothing installed |

A decode after the rounds decodes the open inputs in one call, renames its files by the
puzzle's test indices (`renumber_candidates`), and installs them into the run's folder
(`install_candidates`), round 0's being copied to `round0/` first. With `final_in_prompts`, the
default (ADR-013 as amended: the decode that is submitted keeps the held-out pair, every decode
that feeds a promotion leaves it out), the last round decodes nothing itself and the final decode
follows the rounds whatever their outcome, with the current adapter; before the last round, the
decode after a kept round still runs without the pair, since the next round promotes from it. It
is skipped when every input is promoted (`all_promoted`), after a rollback whose restore did not
give back the weights (`not_restored`), or when the time left is below its own estimate,
$D(4 n_\text{open}, 1, s)$ plus the overhead (`no_time`). Each round's plan reserves the decodes
that may follow it, as its `test_batches` (`run_rounds` counts them, `refine`'s
`decode_batches`): with the final decode, $4 n_\text{open}$ at the last round and
$4 \max(n_\text{open}, 2(n_\text{open} - 1))$ before it; without it, $4(n_\text{open} - 1)$
for the decode after a kept round. The provider records
`snapshot` (the CPU copy before the candidate fit), `candidate_train` (the fit), `fingerprint`
(before the fit and after the round), and `restore` (a rollback); `refine` records `round`. A
puzzle without a held-out pair logs its free checks (`no_held_out`) and a round `no_held_out`;
an error in the rounds' setup is recorded as a round `raised`, and the worker goes on with the
candidates as they are.

**Records**, beside the verdicts in `holdout.log_dir` (`RoundLog`): `plan-<worker>.jsonl`, the
governor's `RoundPlan.to_record()` with `round` and the `shape` it priced (`screened`,
`train_tokens`, `train_steps`, `test_batches`, `gate_batches`, `redecode_batches`,
`observed_batch`); `promote-<worker>.jsonl` as above; `round-<worker>.jsonl`; and
`decode-<worker>.jsonl`:

```json
{"puzzle_id": "981571dc", "round": 1, "version": 0, "next_version": 0,
 "outcome": "rolled_back", "screened": [0], "promoted": [], "decision": "gradient",
 "estimate": 2917.524, "remaining": 41210.3, "weights_before": "c5ccdac5d0be606d38213d6e0d25deb8",
 "weights_after": "c5ccdac5d0be606d38213d6e0d25deb8", "restored": true, "seconds": 812.4,
 "error": null}
{"puzzle_id": "36a08778", "round": 1, "name": "round1", "prompts": "pairs", "inputs": [1],
 "installed": [1], "batches": 4, "expected": 4, "cut": false, "estimate": null,
 "remaining": null, "skipped": null, "seconds": 401.7, "error": null}
{"puzzle_id": "981571dc", "round": 1, "name": "final", "prompts": "held_out", "inputs": [0],
 "installed": [0], "batches": 4, "expected": 4, "cut": false, "estimate": 958.5,
 "remaining": 6482.0, "skipped": null, "seconds": 398.6, "error": null}
```

The session log gets one line per record, `[Rank r] round 1 for <task>: <outcome>, ...`.

### The in-context round and the length guard (P2.5)

With `Rounds.variants` holding `in_context`, `_refine_puzzle` builds an `InContextProvider` whose
fingerprint is the `SGDProvider`'s, and in every run with rounds a `PromptGuard` with the limit
`max_seq_length - max_new_tokens` and `longest_prompt` on the model's formatter. `refine` then:

1. Runs the free checks as before; no input passing them, or no held-out pair, ends the round
   with no plan, as in P2.3.
2. Prices each offered variant: the gradient round as in P2.3 (its shape keys unchanged); the
   in-context round on the inputs whose pseudo-pairs the gate's prompts can hold
   (`shape.in_context`: `screened`, `too_long`, `redecode_batches`, `length_ratio`), without a
   train term (`CostModel.round_costs`' `in_context`). The in-context round is not offered when
   none fits; offered alone, that ends the round `too_long` with no plan.
3. The governor's plan picks a variant (its ladder: gradient first). Declined: every input that
   passed the free checks is recorded `declined` (or `too_long`, when only the in-context round
   could have taken it).
4. `in_context`: the inputs that do not fit are recorded `too_long` and never re-decoded; the
   others are re-decoded, `InContextProvider.adapt` returns the adapter, and the gate decodes
   with them in its prompts. Kept: version $+1$, the adapter unchanged (equal fingerprints).
   Failed: the state as it was; `rollback` restores nothing.

Every decode whose state holds pseudo-pairs gets the state as its prompts hold it, the guard
applied before the call (the GPU callables are P2.3's): a round's re-decodes (round 2) and its
gate in `refine`, the decode after a kept round and the final decode in `run_rounds`. A decode
that drops a pseudo-pair writes `context-<worker>.jsonl` and a session-log line:

```json
{"puzzle_id": "e3721c99", "round": 1, "name": "final", "prompts": "held_out", "queries": [1],
 "kept": [], "dropped": [0], "lengths": [6534, 8400], "limit": 7261}
```

`lengths` are the longest prompt's tokens with the first 0, 1, ... pseudo-pairs; `queries` is
None for a gate (the held-out input). The round's own records are P2.3's, `decision`
`in_context` for an in-context round; its timing ops are P2.3's less `snapshot`,
`candidate_train`, and `restore` (two `fingerprint` records, the digest of the adapter before
and after).

## Origin

Our code (B4's `loop.py`), not moved. It calls the moved functions unchanged: `arc26.ttt.fit` and
`arc26.decode.decode_puzzle` (cell 4 through ADR-011), `ArcDataset` (cell 2), and `score_kgmon`
(cell 3, the submission's selection). The pair is excluded by what they are given, never by
editing them (`arc26/decode.py` and `arc26/nvarc/` are protected). P2.2 reuses `score_sum` and
`getter_kgmon` (cell 3, `arc26/nvarc/arc_decoder.py` lines 17-27 and 37-40) for the agreement,
and `ArcDataset.split_multi_replies`, `augment`, and `forward_mod` (cell 2) for the views; its
only change to moved code is `decode_puzzle`'s keyword-only `view_seed` (ADR-014), whose default
runs the reference's statements.

## Tests

`tests/test_gate.py`, named after B5's acceptance:

- Pass and fail with a mocked decoder: the voted answer equal to the held-out output passes;
  another grid ranked first fails even when the right one is second; no candidate fails; a cell,
  a row, a column, or the transpose off fails; the vote is `score_kgmon` (ties broken by the
  rescoring NLL) or a given selection; the decoder gets $n$.
- Exclusion from every training path: the mocked fit receives the state without the held-out pair
  and `task()` holds exactly the other pairs; the mocked decoder gets the held-out input only as
  the query; a held-out pair can never enter `pairs`.
- The rule: last by default, `first` and a callable as parameters, bad rules refused; $k \ge 3$,
  $k = 2$ with the flag on (default) and off, $k = 1$.
- Edge cases: $30 \times 30$ grids, grid validation, provenance of pseudo-pairs, the state's
  invariants, the verdict record, a raising check.
- The split solver's worker with a hold-out on the GPU stand-ins of `tests/test_solver_timing.py`
  and an identity DFS (each view answers its query's input): none of the 128 training samples per
  puzzle holds the held-out pair (the run without hold-out, where every sample holds it, is the
  control); every gate prompt, DFS and rescoring, holds the other pairs and the held-out input as
  the query; the test prompts leave the pair out, or keep it with `in_test_prompts`, whose
  candidate folder is then byte for byte the run without hold-out; the gate's files are in their
  own folder; the identity task passes and the color-swap task fails; a $k = 2$ puzzle with the
  flag off runs no gate; the gate's timing ops; a raising gate leaves the worker running.

`tests/test_promote.py` (P2.2), named after B5's acceptance, on synthetic vote tallies with mocks
for the candidate fit, the re-decode, and the gate's decode:

- One test per condition that fails only it: agreement below $\tau$, the gate failing with the
  candidate adapter, the re-decode voting another grid (or none); and all three passing. The
  checks after a failing one never run; the re-decode gets the current adapter, the fit starts
  from it, and the gate checks the candidate adapter.
- The disjointness check: all 24 views of each decode compared, by key and by image; a re-decode
  under the first decode's own seed is found and the next seed tried; the seed the search found is
  the one decoded; one-color grids, whose differing permutations all show one prompt, give up; the
  first decode's views follow its layout of test inputs (one shared view each at seed 2).
- The agreement is `score_kgmon`'s order and `getter_kgmon`'s terms, the count over 16; $\tau$
  starts at $9/16$ and is inclusive.
- Edge cases: $k = 2$ with the flag on (one real pair, the fit never holds the held-out one) and
  off (no promotion), several test inputs (one by one, then one shared fit and gate, a failing gate
  promoting neither), no candidate, a tie in the vote and in the re-decode, $30 \times 30$ grids,
  an input promoted in an earlier round, a raising re-decode, fit, or gate, rollback, provenance.
- The held-out pair in no fit and no prompt; the records and `promote-<worker>.jsonl`; the
  `redecode` and `gate` spans; the op names; argument errors; numpy's random state unchanged.
- `decode_puzzle` on the GPU stand-ins of `tests/test_solver_timing.py`: with `view_seed` at its
  default and with a seed, on two test inputs and on one alone, it decodes and rescores under
  exactly the views `view_keys` predicts, with no candidate file in common, and its records go
  under `redecode_decode` and `redecode_selection`.

`tests/test_refine.py` (P2.3), named after B5's acceptance, with a provider stand-in that trains a
numpy weight vector in place: one round kept (version $+1$, the candidate adapter from the previous
one, the gate on it, the plan and records); a wrong pseudo-pair (confident and re-decoded alike)
failing the gate, rolled back to the previous weights bit for bit and the same version; a fit that
raises mid-training restored; a fingerprint that raises before the fit rolling nothing back; the
governor declining (no GPU op, a plan record, the free checks' records as `declined`); nothing to
promote (no plan); no held-out pair (its free checks recorded); the plan's terms; the provider's
budget; a cut re-decode (`redecode_cut`) and a cut gate; a raising re-decode; an error outside the
checks rolled back; argument errors; $k = 2$; two test inputs; $30 \times 30$ grids; the `round`
span. `run_rounds`: the default and the cap; the kept adapter decoding the open input only and its
candidates installed; every input promoted (no decode); round 2 on the decode's candidates after a
kept round 1; a copy that raises in round 2 keeping round 1's adapter; no promotion or a declined
round ending the rounds; a cut or raising decode installing nothing; both values of
`final_in_prompts`, true the default (after a kept and a rolled-back round, without the time for
it, cut or raising, after a rollback that did not restore the weights, and after two rounds); each
round's plan reserving the decodes that may follow it, under both values and with two rounds.
The first decode's layout in `fresh_views` and `screen`; `renumber_candidates`.
The settings' path (`Rounds`, `HoldOut`, `RunConfig`, the solver JSON unchanged without rounds,
refusals), `round_costs`' re-decode term, `DemoSet.task(held_out=True)`, `screen`, the record
files, `OpWatch`, `read_puzzle`, `install_candidates`.

`tests/test_in_context.py` (P2.5), named after B5's acceptance and the brief's, on P2.3's CPU
stand-ins: a kept and a rolled-back in-context round (the adapter's fingerprints unchanged in
both, version $+1$ or unchanged; the gradient provider never called; the gate's prompts holding
the pairs); no fingerprint leaving `restored` None and the final decode running; the governor
picking `in_context` when the gradient round does not fit, `none` when neither does, and the
gradient round whenever it fits; the in-context price (no train term, the length ratio); the guard
at the $30 \times 30$ worst case with the reference's word-level token counts (a pair 1,866, a
query 936, the limit 7,261): `context_fit` on its lengths, real pairs never dropped, pseudo-pairs
dropped latest first, the moved `cut_to_len` cutting nothing behind the guard while without it
some views lose a real pair, the in-context round not offered when no pair fits (forced:
`too_long`; beside the gradient round: the gradient round alone), the round holding the pairs
that fit and recording the others `too_long`, the final decode of a gradient round leaving out the
pair it cannot hold (a context record), a round-2 gate and a round-2 re-decode trimmed, the
in-context pairs chosen on the gate's prompt (the held-out input as its query), the longer
orientation of a wide grid counted; the settings (`variants`' default
and JSON, the three values through the solver JSON, refusals); refine's default unchanged
(P2.3's plan, shape, and record keys); `run_rounds` offering its settings' variants; two chained
in-context rounds; the plan reserving the final decode; the records.
`tests/test_in_context_solver.py` (P2.5): every run of `tests/test_rounds_solver.py` with the
defaults reproduces the digests recorded at P2.3's commit
(`tests/testdata/p2_3_rounds_stand_in.json`: candidates, round files, records, timing ops,
prompts, fits, weights); an in-context arm on the
split solver's worker (no fit, snapshot, or restore; fingerprints unchanged; the gate's and the
final decode's prompts holding the pair); and the guard in `decode_puzzle`'s own prompts at
$30 \times 30$ (the final decode holding the three real pairs, not the pseudo-pair, recorded).

`tests/test_rounds_solver.py` (P2.3): the split solver's worker with rounds on the GPU stand-ins,
extended with adapter weights that the stand-in trainer changes in place, on four synthetic tasks
with DEV4's ids and layout: one round end to end under `final_in_prompts` false (kept; the
candidate fit continuing from round 0's weights for 32 of the 128 views; the kept adapter decoding
the test inputs, its candidates replacing the open input's, which alone is decoded, round 0's in
`round0/`); a wrong pseudo-pair rolled back with the weights restored float for float; nothing to
promote; a governor declining every round (its free checks recorded, the final decode skipped for
time), with round 0's candidates, verdicts, and timing ops as without rounds; a raising fit,
re-decode, and decode after the round, leaving round 0's candidates and the worker going; a cut
re-decode; the round's ops inside the puzzle span; the default, the final decode with the
held-out pair, its ops inside the span; two rounds, the round-1 decode then the final one; a
puzzle without a held-out pair (no final decode). Then `SGDProvider` on its
own (snapshot, $s'$ steps, restore, fingerprints, its price and numpy's state, a copy that raises
in round 2 restoring nothing onto round 1's adapter), and one GPU test of the restore on CUDA bf16
PEFT weights, `embed_tokens` and `lm_head` included (`@pytest.mark.gpu`).

## Known limits

- $n = 16$ is what `decode_puzzle` decodes (`augment(n=2, seed=2)`, in `arc26/decode.py`, a
  protected path); `HeldOutGate` takes $n$, but its GPU decoder refuses any other value. P2.2's
  `view_seed` changes which 16 views, not how many.
- The gate decodes one more input per puzzle, about as long as a test input (4 DFS batches and
  their rescoring): about 340 s per DEV40 puzzle at `P1.3-01` arm 1's batch times (**estimate**,
  `docs/prerun/P2.1-01.md`). With its own 1200 s cap, a puzzle can take up to about 2400 s.
- Hold-out reaches a run only as a series arm (`docs/runbooks/build_submission.md`), so a
  competition rerun never holds a pair out; P2.6 decides how the loop enters production.
- If another real pair equals the held-out one, the gate is easy; such duplicates are not
  filtered (rare in ARC).
- `"last"` depends on the challenges file's order, which the reference keeps.
- `decode_puzzle` reseeds numpy (seed 2 and the rescoring seeds), so after the gate numpy's state
  differs from a run without it; the next puzzle's `fit` reseeds with seed 1, so no puzzle's
  computation depends on it.
- P2.2: $\tau$'s calibration comes from `P1.1-01` (DEV120, no hold-out); on DEV40 the voted
  answers are right less often at every agreement, and the loop's adapter trains on one pair fewer
  (ADR-014). P2.7 ablates $\tau$; P2.8 recalibrates.
- One candidate adapter per round: a failing gate cannot say which of several pseudo-pairs was
  wrong, so it promotes none of them.
- A provider that trains the model in place (the GPU fit does) leaves the candidate's weights in
  it when nothing is promoted; P2.3's `SGDProvider` restores the previous weights from its CPU
  copy. The re-decodes run before the fit, so they use the previous adapter.
- The view check assumes the first decode decoded every test input in one call and the re-decode
  decodes one input alone (the contract above); a caller that decodes otherwise must compute its
  views with `view_keys` on its own layout. `cut_to_len` renames a view's `ex` part, which the images
  ignore, and drops the leading demonstrations of a prompt too long for the model, so a cut view
  shows fewer pairs than its key implies.
- A puzzle whose views all show few distinct prompts (every grid one color, say) finds no disjoint
  seed and promotes nothing.
- P2.3: the governor prices a round from the inputs that pass the free checks, before knowing
  which re-decodes agree, so its estimate is an upper bound: a round declined for time might have
  fit. The trade (`decode_policy` `trade`) is refused.
- P2.3: a promoted input keeps the candidates that promoted it, so a round's adapter changes the
  submission only for a puzzle's open inputs (49 of DEV120's 120 puzzles have two or more test
  inputs); with `final_in_prompts` (the default) every open input is decoded once more, with the
  held-out pair, whatever the round's outcome.
- P2.3: with two rounds and the final decode, an input still open after a kept round 1 is decoded
  twice after it, without the pair for round 2's promotion and with it for the submission; round
  1's plan reserves both.
- P2.3: `install_candidates` copies files one by one; a stop in between leaves an input with both
  old and new files, never none. The round folders stay on Kaggle (not in the pull allowlist).
- P2.3: the snapshot is 1.1 GB of CPU memory per worker during a round, at the notebook's scope.
- P2.5: the in-context round's pairs are chosen on its gate's prompts; the final decode's hold
  the held-out pair as well, so it may leave out a pair the gate held (its context record shows
  it), and the round then changes the submission only through the inputs it promoted.
- P2.5: the guard's measure assumes the reference's tokenization (one token per color, turns
  framed by special tokens), under which a view's length depends on its orientation only; the
  context record's lengths can be checked against the decode's `decode` timing tokens.
- P2.5: the guard covers prompts; the gradient round's training samples are cut by the
  reference's rule, as in P2.3, and the view check counts a pseudo-pair the guard dropped from a
  round-2 re-decode's prompt.
