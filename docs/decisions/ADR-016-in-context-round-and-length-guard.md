# ADR-016: The in-context round: no fit, the gate on the enlarged prompts, the version rolled back; a length guard on every prompt that carries pseudo-pairs

- Status: Proposed, 2026-10-03, until runs `P2.5-01` (the in-context round forced) and `P2.5-02`
  (the gradient round forced), DEV40 on 4xL4, report. Source: task brief `docs/tasks/P2.5.md`;
  BRIEF B5 P2.5, B2 Q4, B4 (`AdapterProvider`, `refine`).
- Tasks: P2.5; P2.6 (the ladder in production), P2.7 (its gradient versus in-context ablation).
- Related: ADR-004 (the in-context fallback, and its consequence "a context-length guard at the
  $30 \times 30$ worst case"), ADR-013 (the held-out pair, the final decode), ADR-014 (promotion),
  ADR-015 (round mechanics, which this round reuses).

## Context

B2 Q4: "The in-context variant (append promoted pairs to the decode prompt, no gradient step) is
the fallback the governor selects when the remaining budget cannot cover a gradient round." B5
P2.5: a provider `InContextProvider` "that returns the current adapter unchanged. Acceptance:
DEV40 lift versus P1 measured and reported; context length guard tested at the 30x30 worst case
(Duck harness lesson: no unbounded appends)."

What P2.3 left: the governor already prices both variants (`CostModel.round_costs`, the ladder
`(gradient, in_context)`); every decode after a round builds its prompts from `DemoSet.pairs`
(`DemoSet.task`), so a kept round's pseudo-pairs already reach the prompts; `refine` offered the
gradient round only.

The hazard: `decode_puzzle` (protected) cuts every prompt with the reference's
`ArcDataset.cut_to_len`, which drops a prompt's first demonstration until its tokens are at most
`max_seq_length - max_new_tokens`, $8192 - 931 = 7261$ under the reference's tokenizer (one token
per color, per newline, per special token and role word). Each view shuffles the demonstrations
(`shuffle_ex`), so the pair dropped is a random one. A $30 \times 30$ pair is 1,866 tokens and a
$30 \times 30$ query 936 (B3: "a pair about 1.9k"): three such pairs and a query take 6,534 tokens,
a fourth makes 8,400, and the cut then drops one of the four, a real pair three times in four
when one of them is a pseudo-pair.

## Decision

1. **The in-context round is P2.3's round with a provider that fits nothing.**
   `arc26.loop.InContextProvider`: `adapt` returns the current adapter, `rollback` returns it
   (nothing was changed), `fingerprint` digests as the gradient provider does (or None),
   `fit_shape` is $(0, 0)$. The round (`refine`, ADR-015 decision 1) runs the free checks, the
   re-decodes under fresh views (ADR-014), enlarges the `DemoSet` with the pairs that fit
   (version $+1$), skips the fit, and gates: the held-out input decoded with the adapter of round
   0 and the enlarged set in its prompts. Kept: the enlarged state and the same adapter. A failed
   gate: the previous state, at its version, and the same adapter; the version alone is rolled
   back. The round record's `decision` is `in_context`; its fingerprints before and after are
   equal, and `restored` is true, or None when the provider has no digest (the final decode reads
   None as "nothing to restore", not as `not_restored`).
2. **Offered beside the gradient round, forced by a setting.** `refine` offers the in-context
   round when given its provider (`in_context`); `variants` restricts what is offered:
   `("gradient",)`, the default, P2.3's offer, plan, and records unchanged; `("in_context",)`; or
   both, in the governor's order. The governor's ladder (P2.4) then picks the gradient round
   whenever it fits, the in-context round when only it fits, else none. `Rounds.variants` carries
   the setting to the workers (in the solver JSON only when not the default, so P2.3's settings
   keep their JSON), and the notebook builder takes it per arm, so one run can measure each.
3. **Priced without a fit, its decodes at the prompts' length ratio.** The plan prices the
   in-context round as P2.4's model says: the re-decodes of its pairs at the round-0 length, the
   gate's and the following test decodes' batches (the final decode reserved, ADR-015 decision 6)
   at $\rho$, the overhead, and no train term:
   $$C_\text{in-context} = (1 + m)\,o + D(4|P|, 1, s) + D(n_\text{test} + 4, \rho, s), \qquad
   \rho = \max_{q} \frac{L(D \cup P, q)}{L(D, q)},$$
   with $P$ the pairs the gate's prompts hold, $L$ the guard's measure, and $q$ over the held-out
   input and the open test inputs.
4. **The length guard: pseudo-pairs as far as the prompt fits, the latest dropped first.** For a
   prompt whose pseudo-pairs are $p_1, \dots, p_n$ in the order added, with $L_j$ the longest
   prompt holding the first $j$ (over the decode's 16 views, its rescoring's 8, and its queries),
   the prompt holds $p_1, \dots, p_k$ for the largest $k$ with $L_j \le \text{limit}$ for every
   $j \le k$; a prompt over the limit without any pseudo-pair holds none
   (`arc26.loop.context_fit`, a pure function of the lengths). `cut_to_len` then cuts no pair it
   would not cut without them: a real pair, or an earlier round's pseudo-pair, is never lost to a
   later pseudo-pair. The limit is `max_seq_length - max_new_tokens`, what `decode_puzzle` passes
   to `cut_to_len`. The measure (`longest_prompt`) is the moved
   `ArcDataset.get_length(..., max_of_transposed=True)` on the entry decoded: each color is one
   token and each turn sits between special tokens, so a view's length depends only on whether
   it exchanges rows and columns, and the rescoring's prompts have the decode's input.
5. **Applied to every decode and rescoring prompt that carries pseudo-pairs, for both
   variants.** The re-decodes of round 2, every gate, the decode after a kept round, and the
   final decode (the held-out pair back before the pseudo-pairs, ADR-013 as amended) receive the
   state as their prompts hold it (`DemoSet.first_pseudo`: every real pair in place, the first
   $k$ pseudo-pairs, the version kept); `refine` and `run_rounds` apply the guard before each
   call, so the GPU callables are P2.3's. A decode that drops one writes a line to
   `context-<worker>.jsonl` (the decode, its prompts, its queries, the pairs held and dropped,
   the lengths, the limit); a decode that drops none writes nothing, so P2.3's records stay as
   they were.
6. **The in-context round's pairs are those its gate's prompts hold.** The inputs that passed the
   free checks are drafted in test-index order after the state's pseudo-pairs; the guard on the
   gate's prompts (the round's own decode) keeps a prefix of them. The others are recorded
   `too_long` and not re-decoded; when none is kept, the in-context round is not offered: forced
   alone, the round ends `too_long` with no plan and no GPU work; beside the gradient round, the
   governor is offered the gradient round alone. The decodes after the round apply their own
   guard: the final decode's prompts hold the held-out pair too, so they may drop a pseudo-pair
   the gate held, which its context record shows.
7. **The gradient round promotes as before.** Every input that passes its checks is trained on
   (ADR-014 decision 4), whatever the prompts can hold; the guard trims only its prompts.

## Alternatives considered

- **A `decode_puzzle` parameter for extra pairs** (B5's deliverable reads "`decode.py` accepts
  extra in-context pairs"). Lost: `arc26/decode.py` is protected and under the P0.1 parity
  fixture; the pairs reach `decode_puzzle` through the puzzle it is given, as the held-out pair
  does in the final decode, so no statement changes.
- **Let `cut_to_len` cut.** Lost: it drops the first demonstration of each view's shuffled order,
  a real pair as often as a pseudo-pair; the CPU test at $30 \times 30$ shows views losing a real
  pair (`tests/test_in_context.py`).
- **Drop the earliest pseudo-pair first.** Lost: the brief's order, and an earlier round's pair
  has passed a kept round's gate already; dropping the latest keeps the prompt a prefix of the
  `DemoSet`.
- **Measure every augmented view.** Equal to the two orientations under the token model, at 24
  tokenizations per query instead of 2.
- **Price the in-context round at $\rho = 1$.** Lost: P2.4's model scales the longer prompts; the
  scaling overestimates (`P2.1-02`'s test prompts, one pair longer than `P2.1-01`'s, decoded in
  0.98 times the seconds), which errs toward declining.
- **Choose the in-context round's pairs on every prompt they will enter** (the final decode's
  too). Lost: a pair the final decode cannot hold still serves the gate and keeps its input's
  checked candidates (ADR-015 decision 4); the final decode records what it drops.
- **`restored` false for an in-context rollback without fingerprints.** Lost: the final decode
  would be skipped as `not_restored` although no weight changed.
- **A separate function for the in-context round.** Lost: B2 Q4 and ADR-004 keep one loop with
  an injected provider; the round's checks, records, and rollback stay in one place.

## Consequences

- Without the in-context provider `refine` offers what it offered in P2.3; the split solver's CPU
  stand-in runs reproduce P2.3's outputs digest for digest (`tests/test_in_context_solver.py`).
  With the guard on, a P2.3-style run's prompts change only where a pseudo-pair would push a
  prompt past the limit, where the reference's cut would have dropped a random pair: on DEV40,
  for example, the final decode of a $30 \times 30$ puzzle with three real pairs and a promoted
  input cannot hold the pseudo-pair (four pairs and a query, 8,400 tokens).
- New: the reason and outcome `too_long`; the plan's `shape.in_context` (`screened`, `too_long`,
  `redecode_batches`, `length_ratio`); `context-<worker>.jsonl`; `Rounds.variants`.
- An in-context round costs no fit and no snapshot; its two fingerprints read the adapter as the
  gradient round's do (a few seconds); the guard tokenizes on the CPU (milliseconds).
- Priced by P2.4's model, the in-context round costs more than the gradient round on every one of
  the 13 DEV40 puzzles whose inputs pass the free checks in `P2.1-01`'s or `P1.3-01` arm 2's
  candidates (**estimate**: 3.0 to 7.0 ks against 2.9 to 4.8 ks, `docs/prerun/P2.5-01.md`,
  section 6): $\rho$ scales the gate's and the final decode's batches, while the 32-step fit adds
  80 to 206 s to the price. Offered beside the gradient round, the ladder then runs it only when
  the gradient round does not fit and it does, which these prices never allow: until P2.7
  measures it (or `P2.5-01`'s seconds refit $\rho$'s scaling), the fallback runs only when forced.
- The gradient round's training samples are cut by the reference's rule (`cut_to_len` on the
  text, 8192 tokens), as in P2.3: a pseudo-pair can push a real pair out of a training sample.
  The guard covers prompts only.
- The view check (ADR-014) compares the images of the state's pairs: a pseudo-pair the guard
  drops from a round-2 re-decode's prompt still counts in them.
- A kept in-context round on a one-input puzzle submits round 0's candidates for that input (its
  checked answer first, ADR-015 decision 4), and no final decode runs.

## Overturn criterion

- `P2.5-01`'s Loss row ($S \le 8.5$ of 40, at least 2 tasks under `P1.3-01` arm 2's 10.5, the same
  scope, rank 16 on q, k, v, o, without the loop): the in-context round leaves P2.6's variants
  (`[gradient]`); its Lift row ($S \ge 12.5$) keeps it as the fallback; P2.7 compares the two
  rounds on DEV40.
- The pairs' choice on the gate's prompts: if `P2.5-01`'s context records show the final decode
  dropping a kept in-context round's pair on more than 2 puzzles, measure the choice on the final
  decode's prompts.
- The measure: if a context record's lengths disagree with the decode's `decode` timing tokens
  (the prompts' tokens per batch) for a decode that held every pair, fix `longest_prompt`.
