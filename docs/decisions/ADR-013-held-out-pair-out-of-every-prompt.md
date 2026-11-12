# ADR-013: The held-out pair reaches no adapter and no prompt that feeds a promotion; the final test decode keeps it; the gate votes as the submission does

- Status: Accepted as amended, 2026-10-03: decision 4 (the final round's test decode keeps the
  held-out pair), as the overturn criterion prescribes, on the evidence of runs `P2.1-01` (the
  pair out of every prompt: $\Delta = -3.5$ tasks on DEV40, 5.0 against 8.5 of 40, Measurable)
  and `P2.1-02` (the pair kept in the test prompts: 7.0 of 40, a recovery of 2.0 tasks, Recovers;
  `docs/experiments/P2.1-01.md`, `docs/experiments/P2.1-02.md`). Caveat: the recovery equals
  DEV40's 2-task noise floor exactly; `P2.1-03`, the hold-out-off arm on the split solver, restates
  both prices when it runs. Proposed 2026-10-02, until `P2.1-01` measured the price (as ADR-011);
  on 2026-10-02 `P2.1-01` measured $\Delta = -3.5$ (Measurable), so the overturn arm `P2.1-02`
  ran. Source: task brief `docs/tasks/P2.1.md`; BRIEF B5 P2.1, B2 Q4, B4 (the `DemoSet` schema).
- Tasks: P2.1 (the gate; its runs `P2.1-01`, `P2.1-02`, and `P2.1-03`); P2.2 and P2.3, which
  build on it (P2.3: `Rounds.final_in_prompts`).
- Related: ADR-004 (the `DemoSet`, the refinement mechanics), ADR-011 (the split solver it calls),
  ADR-014 (the promotion, which decodes without the pair).

## Context

B5 P2.1 excludes the held-out pair "from TTT training, from leave-one-out task generation, and
from voting", and `passes()` "decodes the held-out input under $n$ augmentations and requires the
voted answer to match exactly". Two readings of "voting" were open (task brief):

1. Minimal: the pair reaches no adapter and no prompt that decodes or rescores the held-out input
   itself. Every reading needs this, or the gate would score the adapter on an answer it was
   shown.
2. Full: the pair also stays out of the decode and rescoring prompts of the real test inputs.
   The reference's prompts hold every demonstration pair, and its vote over augmented views is
   how a test output is chosen, so these prompts are where the test inputs are "voted with".

The choice has a price. Keeping the pair in the test prompts gives the test decode $k$
demonstrations, where the adapter learned from $k - 1$; leaving it out gives it $k - 1$. And
"the voted answer" needs a definition: the reference decodes 16 views, rescores each new grid
under 8 more, and chooses two attempts with `score_kgmon`.

Measured (amendment, 3 Oct): leaving the pair out of training and of every prompt cost
$\Delta = -3.5$ tasks on DEV40 (`P2.1-01`); keeping it in the test prompts, with the same fit and
gate, recovered 2.0 of them (`P2.1-02`, 7.0 against 5.0 of 40), leaving $-1.5$ against the
hold-out-off arm, under the noise floor.

## Decision

1. **Out of every prompt that feeds a promotion (the full reading).** Every test decode whose
   predictions a promotion can take (P2.2, ADR-014) decodes and rescores the test inputs with the
   training and voting set `DemoSet.pairs` only, as the held-out input is
   (`HoldOut.in_test_prompts = False`, as in `P2.1-01`).
2. **One place builds every dataset.** `DemoSet.task()` builds the reference's task entry from
   `pairs` only, and every dataset the solver trains or decodes on comes from it, or from the
   puzzle as loaded for a decode that keeps the pair (`in_test_prompts`, decision 4). The fit and
   decode callables receive the state without its held-out pair.
3. **The voted answer is attempt 1 of the submission.** The gate decodes the held-out input with
   `decode_puzzle` (16 views, its rescoring) and votes with the reference's submission selection,
   `score_kgmon`, over all its candidates; it passes iff the top grid equals the held-out output
   exactly. A correct grid ranked second fails.
4. **The final round's test decode keeps the held-out pair** (amended 2026-10-03, the overturn
   criterion). The test decode after which nothing is promoted, the one that gives the
   submission's attempts, decodes and rescores the test inputs with every real pair, the held-out
   one included, as the reference does (`in_test_prompts` for that decode; `P2.1-02`). The adapter
   is still never trained on the pair, and the gate still decodes the held-out input without it.
   P2.3 implements it as `Rounds.final_in_prompts`, true by default. The governor prices the
   extra decode: a puzzle whose last test decode fed a promotion check needs one more test decode
   for its final attempts, about 370 s per DEV40 puzzle (`P1.3-01` arm 1, `P2.1-02`: the same
   prompts cost the same with the adapter fit on $k - 1$ pairs, 0.997 times arm 1's seconds on the
   36 puzzles that decoded the same batches).

## Alternatives considered

- **Keep the held-out pair in the test prompts of every decode** (the minimal reading). Cheaper in
  accuracy: the test decode sees all $k$ demonstrations, so part of the price is recovered for
  free. Lost, for three reasons. (a) In the loop the round-0 test predictions become pseudo-labels
  that the next adapter is trained on (P2.2, P2.3); a pseudo-label decoded with the held-out pair
  in context carries the pair's information into training, and the gate then checks an adapter
  that has learned from the pair it is checked on. B5's reason for the gate, "a real pair it never
  saw", would no longer hold. (b) B2 Q4 and B4 make the state the data plus the adapter: `pairs`
  is "the training and voting set" and the held-out pair is "never trained on, never voted with";
  a prompt holding a pair outside `pairs` puts state outside the versioned `DemoSet`, so a rollback
  (a pointer swap, ADR-004) would not describe what produced a prediction. (c) Phase 4 keeps the
  adapter's source and the prompt's demonstrations the same set,
  $\theta = \mathrm{polish}(H(D), s)$ with $D$ in context (B2 Q6), and Q4 recomputes the adapter to
  keep H's training and test distributions aligned; a held-out pair in the prompt but not in $H$'s
  input would break that. Reasons (a) and (b) concern decodes whose predictions are promoted or
  rolled back, so they do not apply to the final decode (decision 4); reason (c) is Phase 4's and
  is not decided by this amendment, which `P2.1-02` measured with the SGD adapter only.
- **Hold out only for the gate's decode, and decode the final attempts with every pair in a
  separate pass.** The final round's predictions are never trained on, so they could use all
  $k$ pairs without leaking. Lost on 2 Oct for its cost: a second test decode per puzzle (about
  370 s per DEV40 puzzle, `P1.3-01` arm 1) on top of the gate's (about 260 s, `P2.1-01`), which the
  governor could not afford at 240 tasks. Taken on 3 Oct as decision 4, the overturn criterion's
  first candidate, since `P2.1-02` measured its recovery; the governor prices the decode.
- **A last fit with the held-out pair before the final decode.** Not taken: it adds a fit (about
  187 s per DEV40 puzzle, `P1.3-01` arm 1) to the final decode, and the decode alone recovered 2.0
  of the 3.5 tasks. It would remove a mismatch `P2.1-02` showed: with the pair in context but not
  in the fit, 41 of the 55 test outputs got a candidate, against 49 (`P2.1-01`) and 51 (arm 1). It
  stays a P2.3 option, measured on DEV40 before P2.6.
- **Pass when the held-out output is either of the two attempts.** Lost: B5 asks for "the voted
  answer", and a looser gate admits more wrong pseudo-pairs. The verdict logs the output's rank
  (`expected_rank`), so a top-2 gate can be read off the price run's logs without another run.
- **Vote with `score_full_probmul_3`, the reference's other selection.** Lost: the submission uses
  `score_kgmon` (`run_selection_algo`'s default), and the gate should certify what would be
  submitted. The selection is a parameter of `HeldOutGate`.
- **Decode the held-out input under fewer views** (the governor's simulator assumed 2 batches, 8
  views). Lost for P2.1: `decode_puzzle` fixes 16 views, and changing it is a change to the
  protected `arc26/decode.py` (B4: reuse the notebook's decode as it is); P2.2 needs a view
  parameter for its fresh augmentation set and can add one under the parity fixture.

## Consequences

- The price `P2.1-01` measured is the price of the full exclusion, training and context: $-3.5$
  tasks on DEV40. With the final decode keeping the pair, `P2.1-02` measured $-1.5$ against the
  hold-out-off arm, under the noise floor; the gate's pass rate stayed 10 of 40, with 38 of the 40
  verdicts equal.
- The test inputs' prompts of every decode that feeds a promotion are one demonstration shorter
  than the reference's, so that decode is slightly cheaper; the gate adds about one test input's
  decode per puzzle (261.8 s on average, `P2.1-01`); the final decode keeps the reference's
  prompts and cost.
- The final attempts come from prompts holding every real pair and an adapter fit on all but one:
  `P2.1-02`'s coverage fell to 41 of 55 outputs, without a loss of score, a mismatch to watch in
  P2.3's runs.
- The gate's verdict depends only on $D$, the adapter, and $x_h$, so its pass rate is a clean
  estimate of how often the adapter solves an unseen real pair (P2.2's evidence, with the
  association between a passing gate and a correct test prediction: 3 of 10 against 2 of 30 in
  `P2.1-01`, 3 of 10 against 4 of 30 in `P2.1-02`).
- P2.2 promotes from test predictions made without the held-out pair, and checks the candidate
  adapter with a gate that is still independent of its training data.
- The governor's round cost adds the final decode, $n_\text{test}$ batches, for a puzzle whose
  last test decode fed a promotion check; the gate costs 4 batches (`P2.1-01`).

## Overturn criterion

If `P2.1-01` measures $\Delta = S_\text{on} - S_\text{off} \le -2$ tasks on DEV40 (the noise
floor), run one DEV40 arm with `in_test_prompts: true` at the same settings and seed. If it
recovers at least 2 tasks over `P2.1-01`, amend this ADR so that the final round's test decode
(after which nothing is promoted) keeps the held-out pair, while every decode that feeds a
promotion still leaves it out; the governor then prices the extra decode.

Met on 2026-10-03: `P2.1-01` measured $\Delta = -3.5$ and `P2.1-02` recovered 2.0 tasks (7.0
against 5.0 of 40), exactly the threshold; decision 4 is the amendment. Should `P2.1-03` restate
the recovery below the floor, the final decode's policy (the pair in its prompts, a last fit with
it, or neither) is measured again on DEV40 before P2.6, since a price inside DEV40's noise floor
can still be 3 to 4.5 DEV120 tasks, more than P2.6's bar. This ADR fixes the exclusion for the
gate and for every decode that feeds a promotion.
