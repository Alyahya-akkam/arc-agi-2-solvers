# Module: `arc26/governor.py`

Task P2.4 (BRIEF B5 P2.4, B2 Q4, B3, B7 G1). Config `configs/governor.yaml`; tests
`tests/test_governor.py`. Pure Python, no GPU code. The governor never tokenizes: token counts
are the caller's inputs.

## Purpose

The refinement loop (P2.3) and later H's polish (P4) can only spend the time the baseline leaves
free, and a run that passes the 12 h limit loses tasks. Run `P1.1-01` used $u = 0.465$ of the
solving window on DEV120's 120 tasks; at the same per-puzzle cost a 240-task rerun projects to
$u = 0.909$, at G1's 90% line (`reports/P1.1-01/summary.md`, section 8). The governor prices
each loop operation conservatively, gives each puzzle a share of the time left (computed at run
time from the queue, so it adapts to the rerun's task count), and picks one round variant per
puzzle from a fallback ladder. Under the `trade` policy it can also stop the baseline's decode
early to make room for a round: G1's "buy time by reducing decode augmentations".

## Cost model

Ops are those of `arc26.timing`: `train`, `decode`, `selection`. Raw conservative costs per unit,
before the margin:

$$\tau_\text{train}(T, S) = \begin{cases} 0 & T = 0, \\ \max\!\left(0,\ a + bT + c\,T^2/S\right) + \varepsilon & T > 0, \end{cases} \qquad \tau_\text{decode} = q_\text{dec}, \qquad \tau_\text{selection} = q_\text{sel}.$$

A fit of $T$ tokens in $S$ optimizer steps has mean sequence length $T/S$ per step, so $cT^2/S$
is its attention cost. Every estimate is $(1 + m)$ times the raw cost: `estimate(op, tokens,
count=n)` returns $(1 + m)\,n\,\tau_\text{op}$. Coefficients (`summary.md` lines; the test checks
each against its report row and recomputes it from `reports/P1.1-01/timing-rank*.jsonl`):

| Symbol | Value | Source |
|---|---:|---|
| $a, b, c$ | 24.35 s, $5.812 \times 10^{-4}$ s/token, $3.487 \times 10^{-8}$ s/token$^2$ | line 90, the attention model ($R^2 = 0.998$, $n = 120$); also `docs/experiments/P1.1-01.md` line 145 |
| $S$ (default) | 128 | line 104: the reference's one epoch |
| $\varepsilon$ | 22.1 s | line 90: the model's worst residual (`5961cc34`) |
| $q_\text{dec}$ | 143.9 s per DFS batch | line 48: p90 of 669 batches (mean 66.8 s) |
| $q_\text{sel}$ | 7.8 s per rescored candidate | line 49: p90 of 1181 candidates (mean 4.7 s) |
| $\kappa$ | 4 candidates per batch | line 214: $\lceil 13/4 \rceil$, the p90 of distinct grids per test output over 4 batches |
| $o$ | 9.6 s per round | line 43: the largest per-puzzle overhead |
| $m$ | 0.35 | "Margin" below |

Decode and selection have no token axis: `P1.1-01` logged no decode or selection tokens, and on
K0's `cells` proxy a line explains $R^2 = 0.243$ of decode per batch (line 80). Their inputs are
therefore what the caller can count before the op runs: DFS batches, and $\kappa$ candidates per
batch, since candidates exist only after decoding. Decisions never use a mean: the mean decode
batch is less than half the p90.

Decode and its rescoring run together, batch by batch, so they are priced together. For $n$
batches, a prompt-length ratio $\rho \ge 1$, and $s$ the seconds of the slowest batch this
puzzle ran in round 0 (0 before its first):

$$D(n, \rho, s) = \rho\,(1 + m)\,n\left(\max(q_\text{dec}, s) + \kappa\,q_\text{sel}\right).$$

The floor $s$ matters: 77% of the variance of the seconds per batch lies between puzzles (85% of
the log seconds: a between-puzzle sd of 0.83 against 0.35 within), so a puzzle that decoded
slowly in round 0 decodes slowly in its round too. Without it the simulator's slowest puzzles took twice their round estimate at
the end of the queue and passed the deadline. Scaling by $\rho$ overestimates: a DFS step's cost
is affine in the prompt length with a positive intercept (reading the weights).

Round costs (`round_costs`), with $n_\text{test}$ test-input batches (the reference: 4 per test
input), $n_\text{gate}$ held-out gate batches (P2.1), $n_\text{re}$ re-decode batches of the
promotion (P2.2; 4 per test input that passed its free checks, P2.3), and the gradient round's
fit of $T'$ tokens in $S'$ steps:

$$C_\text{gradient} = (1 + m)\left(\tau_\text{train}(T', S') + o\right) + D(n_\text{test} + n_\text{gate} + n_\text{re}, 1, s),$$
$$C_\text{in-context} = (1 + m)\,o + D(n_\text{re}, 1, s) + D(n_\text{test} + n_\text{gate}, \rho, s).$$

The re-decodes run before any pair is added, so their prompts keep the round-0 length in both
variants. For a fit of part of an epoch (P2.3's $\mathrm{sgd\_continue}$, $S' = s'$ of the 128
views at batch 1), $T'$ is the tokens of the samples it trains on, $s'/128$ of the dataset's
(`arc26.ttt.SGDProvider.fit_shape`), so that $T'/S'$ stays the mean tokens per step as in the
fits the model was fitted on.

The in-context round pays the gate's decode too, since P2.3 runs the gate check in every round
for every provider. It is the cheaper variant only when
$(1 + m)\,\tau_\text{train}(T', S') > (\rho - 1)\,D(n, 1, s)$: the gradient round's fit must cost
more than the longer prompts add to decode. With a quarter-epoch fit that rarely holds; in the
simulator (a full epoch per gradient round) the in-context round was chosen for 0.0% to 0.3% of
puzzles.

### Margin

In-sample: priced with these coefficients as a round of its own $T$ and decode batches, 8 of the
120 `P1.1-01` puzzles take longer than their estimate at $m = 0$; the worst takes 1.302 times it
(`7b0280bc`: four batches of 177 to 266 s, so batches of one puzzle are correlated). With
$m = 0.35$ the estimate covers all 120 (`test_margin_covers_every_p11_puzzle_and_is_needed`).

Stated noise level: the margin keeps the governor safe when the model is optimistic by up to 30%.
In the simulator below, with true costs $k$ times the calibrated ones (400 runs per row):

| Tasks | Policy | $k$ | Runs past the deadline, $m = 0.35$ / $m = 0$ | Rounds over their estimate, $m = 0.35$ / $m = 0$ |
|---:|---|---:|---|---|
| 120 | reference | 1.25 | 0 / 0 | 0.02% / 0.73% |
| 120 | trade | 1.25 | 0 / 0 | 0.03% / 1.06% |
| 200 | reference | 1.25 | 0 / 0 | 0.05% / 0.48% |
| 200 | trade | 1.25 | 0 / 1 (by 501 s) | 0.00% / 0.97% |
| 120 | reference | 1.3 | 0 / 0 | 0.04% / 1.04% |
| 120 | trade | 1.3 | 0 / 0 | 0.04% / 1.25% |
| 200 | reference | 1.3 | 0 / 1 (by 235 s) | 0.16% / 0.57% |
| 200 | trade | 1.3 | 0 / 3 (by up to 594 s) | 0.14% / 0.96% |

At 200 tasks and $k = 1.3$ the baseline alone fills most of the window, so rounds decided late
in the queue are where an optimistic model passes the deadline. The five runs that pass it
without a margin (seeds 259 at $k = 1.25$; 87, 173, 259, 289 at $k = 1.3$) stay inside with it
(`test_margin_is_what_keeps_these_runs_inside_the_deadline`). At $k = 1$ no round cost more
than its estimate in 400 runs of 120 tasks under either policy.

## Budgets

When a worker takes a puzzle at time $t_0$, with $q$ task ids still queued, $w$ workers,
deadline $t_\text{end}$, and reserve $r$:

$$B = \frac{\max(0,\ t_\text{end} - r - t_0)}{1 + \lceil q / w \rceil}, \qquad d = t_0 + B, \qquad R(t) = d - t.$$

The workers share one queue, so this worker's share of the $q$ queued ids is $\lceil q / w \rceil$
if they drain it evenly, plus the puzzle it holds. `puzzle_budget` takes the raw `queue.qsize()`:
the reference's queue (`arc26/nvarc/starter.py`) holds the task ids followed by one `None` per
worker, which stops the worker that takes it. The queue is first in, first out, so while ids
remain all $w$ sentinels are queued, and once they are gone the size is at most $w$; hence
$q = \max(0, \text{size} - w)$ exactly. Counting the sentinels as puzzles would shrink every
budget: in the simulator, 13% fewer rounds at 120 tasks and 66% fewer at 240. $d$ is the puzzle's
deadline and $R$ the time left to it at a decision. $t_\text{end}$ is the solver's end time, 12 h
minus 30 min after the notebook starts (41,400 s; `deadline.end_s`). For one worker, a puzzle that
ends within its budget never shrinks the next one: with $n = 1 + \lceil q/w \rceil$ and $nB$ left,
using at most $B$ leaves at least $(n - 1)B$ for $n - 1$ puzzles. Only a baseline longer than its
budget shrinks later budgets, and then the governor refuses rounds.

Budgets follow the load. With spare time they grow toward the end, as unused time passes to fewer
remaining puzzles: the median budget rose from 1,429 s in the first quarter of a simulated
120-task run to 4,529 s in the last (240 tasks: 687 to 1,135 s). When the work outgrows the
window (360 tasks) they shrink (439 to 167 s) and the late puzzles get no round.

### Stragglers

The formula's failure mode is the end of the queue:

1. The even split fails while one worker is stuck on a long puzzle: the others take more than
   their share, on budgets sized for fewer puzzles.
2. A worker's last puzzle ($q = 0$) gets all the time left, so an overrun of its round has no
   later puzzle to absorb it, and the other workers idle meanwhile.
3. With $q < w$, every worker plans one more puzzle though only $q$ get one: those budgets are
   smaller than needed (safe, but rounds are lost).
4. $w$ is fixed: a dead worker leaves the others' shares underestimated.

Guards: the margin; the observed-batch floor; the reference's own end-time checks, which stop the
baseline decode at the deadline; and the reserve $r$, which keeps rounds out of the last $r$
seconds. The simulator found no run past the deadline without a reserve, while $r = 1380.7$ s (the
longest `P1.1-01` puzzle, line 44) cost 7% of the rounds at 120 tasks, 16% at 200, and 46% at
240. The config therefore keeps $r = 0$, the brief's formula; the human may raise it at G1. The
hard stop at 42,000 s, 600 s after the deadline, bounds what remains.

## The ladder

`choose(round_costs, remaining)`: gradient if $C_\text{gradient} \le R$, else in-context if
$C_\text{in-context} \le R$, else none. B5 P2.4 says "the cheapest permitted variant"; B2 Q4's
intent is a fallback ladder, with in-context the variant the governor selects "when the remaining
budget cannot cover a gradient round". The code follows Q4: gradient wins whenever it fits, even
when in-context is cheaper, and an in-context round that costs more than the gradient round never
replaces it. A missing key means the variant is not offered; $R \le 0$ leaves only none.

## The decode trade

The reference decodes each test input in 4 DFS batches of 4 augmented views, in this order:
batches 0 and 1 of every test input, then batches 2 and 3 of every test input; it checks its
1200 s cap before each batch. `plan(...)` prices $p$ pending baseline batches at
$\delta = D(1, 1, s)$ each, with $j$ batches done and $n_\text{in}$ test inputs:

- `reference` (default): the floor is $f = p$, so no batch is cut; the decision is what the
  pending batches leave.
- `trade`: the floor is $f = \min(p, \max(0, 2 n_\text{in} - j))$, keeping the first 2 batches of
  every test input (its 8 views without a transpose). For the first ladder variant $v$ with
  $C_v + f\delta + h \le R$, the plan keeps $k = \max\{k \in [f, p] : C_v + k\delta + h \le R\}$
  batches; when no variant fits, it keeps all $p$.

A plan made before a baseline batch ($p > 0$) holds $h = (1 + m)\,o$ for the baseline's own
overhead after its decode (`overhead_seconds`); after the baseline $h = 0$. Without it, a cut left
less than one batch of slack, the unpriced overhead then used it, and the final decision fell to
none in 2.6% to 3.5% of the cut puzzles: views given up for no round. The alternative, keeping the
cut's decision at the final plan whatever time is left, would let that round overrun its
deadline and break the rule that every final decision is `choose` on its own record. With $h$,
a cut is followed by the variant it was made for (or gradient) whenever the overhead stays within
$(1 + m)\,o$: in the simulator, none of the cut puzzles ended without a round at true costs up
to 1.3 times the model (`test_simulated_cuts_are_never_followed_by_none`).

The loop asks before each baseline batch and stops the baseline decode when `decode_batches` is
0; `cut_batches` in the logged plan is the trade. In the simulator `trade` ran 9% more rounds than
`reference` at 120 tasks (17,451 against 16,024 in 400 runs) and 14% more at 240 tasks (2,283
against 2,010). The
default keeps the reference's decode; the human sets the policy at G1.

## Interfaces

| Name | What |
|---|---|
| `CostModel.from_config(path)`, `CostModel.from_mapping(config)` | load `configs/governor.yaml` (PyYAML, imported only by `from_config`), or build from its parsed content |
| `CostModel.estimate(op, tokens=None, *, count=1, steps=None)` | $(1 + m)\,n\,\tau_\text{op}$; `train` needs tokens (0 costs 0), `decode` and `selection` ignore them |
| `CostModel.decode_cost(batches, length_ratio=1.0, observed_batch=None)` | $D(n, \rho, s)$ |
| `CostModel.round_costs(*, train_tokens, test_batches, gate_batches=0, train_steps=None, length_ratio=1.0, observed_batch=None, redecode_batches=0)` | `{"gradient": ..., "in_context": ...}`; `redecode_batches` (P2.3) defaults to 0, which leaves every earlier value unchanged |
| `Governor(cost_model, end_time, workers, clock=time.time, *, reserve=0.0, policy="reference", min_batches_per_input=2)` | one per worker process; `clock` injectable |
| `Governor.from_config(path, end_time, workers, clock=time.time)`, `Governor.from_mapping(config, end_time, workers, clock=time.time)` | also read `deadline.reserve_s` and `decode_policy`; `from_mapping` needs no file |
| `Governor.puzzle_budget(queue_size)` | $B$, from the raw `queue.qsize()` (task ids plus sentinels) |
| `Governor.choose(round_costs, remaining)` | `"gradient"`, `"in_context"`, or `"none"` |
| `Governor.plan(round_costs, deadline, *, budget, pending_batches=0, test_inputs=1, done_batches=0, observed_batch=None, puzzle_id=None)` | a `RoundPlan`; with no pending batches its decision is `choose` on $R$ |
| `RoundPlan` | `puzzle_id`, `decision`, `at`, `budget`, `deadline`, `remaining`, `estimates`, `decode_batches`, `pending_batches`, `decode_seconds`, `overhead_seconds`, `policy`, `margin`; `cut_batches`; `to_record()` returns a JSON-ready dict |
| `OpCost` | one op's unit cost: `a`, `b`, `c`, `steps`, `allowance`, `token_axis` |

Constants: `GRADIENT`, `IN_CONTEXT`, `NONE`, `LADDER`, `VARIANTS`, `REFERENCE`, `TRADE`,
`POLICIES`, `REFERENCE_BATCHES_PER_INPUT`. Bad arguments raise `TypeError` or `ValueError`;
booleans are refused, and integer scalars that are not `int` (numpy, torch) convert through
`__index__`.

## How the loop calls it

```python
from arc26.governor import GRADIENT, IN_CONTEXT, TRADE, Governor

# config: the parsed configs/governor.yaml, which the wheel does not carry ("Known limits")
governor = Governor.from_mapping(config, end_time=end_time, workers=4)
cost = governor.cost_model
while not queue.empty():                    # the reference's worker loop
    if time.time() > end_time:
        break
    key = queue.get()
    if key is None:                         # this worker's sentinel
        break
    start = time.time()
    budget = governor.puzzle_budget(queue.qsize())         # raw size: ids plus sentinels
    deadline = start + budget
    ...                                     # round-0 fit, as the reference
    shape = dict(train_tokens=t_round, train_steps=s_round, test_batches=4 * n_test,
                 gate_batches=n_gate, length_ratio=rho)    # counted from the DemoSet
    slowest = None
    for done, subkeys in enumerate(batches):               # the reference's order
        if time.time() - start > 1200 or time.time() > end_time:
            break                                          # the reference's checks
        if governor.policy == TRADE:
            plan = governor.plan(cost.round_costs(**shape, observed_batch=slowest), deadline,
                                 budget=budget, pending_batches=len(batches) - done,
                                 test_inputs=n_test, done_batches=done,
                                 observed_batch=slowest, puzzle_id=key)
            if plan.decode_batches == 0:
                log(plan.to_record())                      # the trade
                break
        ...                                 # decode the batch (timed), rescore its candidates
        slowest = batch_seconds if slowest is None else max(slowest, batch_seconds)
    plan = governor.plan(cost.round_costs(**shape, observed_batch=slowest), deadline,
                         budget=budget, puzzle_id=key)
    log(plan.to_record())
    if plan.decision == GRADIENT:
        ...                                 # refine() with the SGD provider
    elif plan.decision == IN_CONTEXT:
        ...                                 # refine() with the in-context provider
```

P2.3 (`arc26.loop.solve_puzzle` and `refine`, ADR-015) calls it this way with the reference
policy only (the trade needs a plan before each baseline batch inside the protected
`decode_puzzle`, so `Rounds` refuses it): the governor comes from the mapping that the notebook
builder embeds in the solver JSON (`holdout.rounds.governor`); the budget from the queue's size
right after the worker took the puzzle; one plan per round, after round 0's decode (or the decode
after the previous round), with `{"gradient": C}` only, priced from the inputs that passed the
promotion's free checks, `gate_batches` 4, `redecode_batches` 4 per screened input, the slowest
batch the puzzle ran so far, and as `test_batches` the test decodes that may follow the round
(`arc26.loop.run_rounds` counts them, `refine`'s `decode_batches`). With `final_in_prompts`, the
default (ADR-013 as amended after `P2.1-02`: the decode that is submitted keeps the held-out
pair, so a puzzle needs one more test decode after the rounds), the last round's `test_batches`
are the final decode's, $4 n_\text{open}$, since it follows whatever the round does, and an
earlier round's $4 \max(n_\text{open}, 2(n_\text{open} - 1))$, its own decode after a kept
round as well; without it, $4 (n_\text{open} - 1)$ (the decode after a kept round decodes only
the test inputs still open, at least one fewer than are open before it). The plan goes to
`plan-<worker>.jsonl` beside the gate's verdicts. The final decode is also checked on its own
before it runs, $D(4 n_\text{open}, 1, s) + (1 + m)\,o$ against the time left, which a round
the governor accepted leaves unless it overran its price.

P2.5 (ADR-016) offers the in-context round too, as `Rounds.variants` says: the default
`("gradient",)` keeps P2.3's call above, `{"gradient": C}` alone; `("in_context",)` offers
`{"in_context": C}` alone, and both offer both, so `choose` walks the ladder: the gradient round
whenever it fits, the in-context round when only it fits. Its price is `round_costs(...)
["in_context"]` with no train tokens, `redecode_batches` 4 per input whose pseudo-pair the gate's
prompts can hold (the loop's length guard), the same `gate_batches` and `test_batches` as the
gradient round's, and `length_ratio` $\rho$ measured by the guard: the largest over the gate's
query and the open test inputs of the longest prompt with the pairs over the longest without
them (the plan records it as `shape.in_context.length_ratio`). When no pseudo-pair fits, the
in-context round is not offered; offered alone, no plan is made. On DEV40 `P2.1-02`'s test
prompts, one pair longer than `P2.1-01`'s, decoded in 0.98 times the seconds, so the linear
scaling by $\rho$ errs toward declining the in-context round.

`end_time` is the solver's `--end-time`; the notebook passes 41,400 s after its start in a rerun
(`scripts/build_kaggle_notebook.py`, `COMPETITION_SOLVER_END_S`), which the test checks against
`deadline.end_s`. `t_round` is the token count of the round's fit, counted as P0.3b counts train
tokens, and `s_round` its optimizer steps; `rho` is the in-context prompt's tokens over the
round-0 prompt's. Where the plan records go is P2.3's choice (one JSONL line per plan, next to
the timing records).

## Simulator

`tests/test_governor.py::simulate`: four workers (or one) drain one shared queue under the
reference's loop. The queue holds the task ids and then one `None` per worker, and the governor
gets its raw size. Each worker takes an item while the queue is not empty and the clock is before
the deadline, stops at its `None`, fits, decodes 4 batches per test input with the reference's checks (a DFS call stops at
540 s and at the deadline), and rescores; the governor then plans one round, whose ops run to
completion (the strictest assumption about P2.3's loop). True costs are the cost model's central
line times heavy-tailed noise, calibrated to the `P1.1-01` records:

- $\log T \sim \mathcal{N}(12.304, 0.544^2)$; test inputs 1, 2, 3 in proportion 71 : 46 : 3
  (line 106).
- Decode, log seconds per batch: $3.852$ plus a puzzle effect (sd 0.829, Student $t$ with 5
  degrees of freedom, correlated 0.553 with $\log T$) plus batch noise (sd 0.352, $t$ with 4),
  capped at the DFS's 540 s.
- Train: the attention model times $e^{0.03\,t_3}$ (measured log sd 0.031).
- Selection: log-normal per candidate (log mean 1.393, sd 0.576); candidates per batch from the
  measured histogram.
- Overhead uniform on 5.3 to 9.6 s; worker starts from `P1.1-01`'s startup and model load.

Assumptions, not measured: 2 to 5 demonstration pairs (weights 3 : 4 : 2 : 1); one held-out pair;
a gradient round fits one epoch (128 steps) on the enlarged set; the gate decodes 2 batches; the
in-context prompt grows by $\rho = 1 + 2 n_\text{in} / (2 \cdot \text{pairs} + 1)$, and its true
decode cost by $\sqrt{\rho}$.

With the shipped config (400 runs per row) no worker passed the deadline:

| Tasks | Policy | Rounds | Gradient / in-context / none | Rounds past their puzzle deadline |
|---:|---|---:|---|---:|
| 120 | reference | 16,024 | 33.1% / 0.2% / 66.6% | 0 |
| 120 | trade | 17,451 | 36.0% / 0.3% / 63.6% | 0 |
| 200 | reference | 6,513 | 8.1% / 0.1% / 91.9% | 0 |
| 200 | trade | 7,190 | 8.9% / 0.1% / 91.0% | 1 |
| 240 | reference | 2,010 | 2.1% / 0.0% / 97.9% | 0 |
| 240 | trade | 2,283 | 2.4% / 0.0% / 97.6% | 1 |

A round past its own puzzle's deadline costs the next budgets, not the run.

## Tests

`tests/test_governor.py` (77 tests, about 4.5 s): the config against its report rows and
recomputed from the records (quantiles, candidates and rescoring seconds per batch, overhead, the
attention fit by least squares, its $R^2$ and worst residual); the in-sample margin check; the
deadline against the notebook builder; `from_mapping` without a file; estimates (the quadratic
term with a round's steps, zero tokens, zero counts, the clamped negative fit, quantile ops, the
observed-batch floor, round composition) and their errors; budgets (the share from a raw queue
size, a `queue.Queue` with the reference's sentinels, an empty queue, one worker, shrinking to 0
with the clock, the reserve) and errors; the ladder (order, fallback, no room, an expensive
in-context round) and errors; plans and the trade (reference never cuts, the fewest cuts for the
first fitting variant, the floor with batches done, the observed floor, the held overhead).
Simulator: no worker passes the deadline at 120 and 240 tasks under both policies; the ladder is
exercised and every decision equals `choose` on its own record; the last puzzle gets all the time
left; budgets shrink under overload; trade cuts keep the floor, add rounds, and are never followed
by none; the margin catches a 25% optimistic model, keeps the deadline at 30%, and is what keeps
the five pinned runs inside it; one worker; rounds stay out of a reserve.

## Known limits

- One measured run: the coefficients come from `P1.1-01` (DEV120 on 4xL4); the hidden set may be
  larger or slower, which is what the margin covers.
- Before a puzzle's first decode batch (the trade's first decisions) decode is priced at the
  population p90, which underprices slow puzzles; their own batches correct it from the second
  decision on.
- No token axis for decode and selection until P0.3b logs their tokens; the in-context length
  scaling is an argument, not a fit: P2.5's loop measures $\rho$ from the prompts' tokens and
  scales the decode linearly, an overestimate.
- A round's fit with fewer steps or longer sequences than the reference's extrapolates the
  attention model, which was fitted on 128-step fits of 58k to 955k tokens.
- The governor cannot stop the reference from starting a puzzle it cannot finish: when the
  baseline alone outgrows the window, workers pass the deadline by up to one fit, as the
  reference does. The trade cuts decode only to make room for a round; cutting to fit the budget
  without a round would be a G1 decision.
- Wall time, not GPU time; the worker count is fixed for the run.
- Packaging: the wheel carries only `arc26/`, not `configs/`, so on Kaggle the loop cannot read
  `configs/governor.yaml`. P2.3 passes the parsed content instead: the notebook builder reads the
  built commit's file and embeds it in the arm's round settings, which reach the workers in the
  solver JSON (`Rounds.governor`, `Governor.from_mapping`); the tests keep checking the YAML
  against its sources.
- `from_config` needs PyYAML (a dependency of transformers in the Kaggle image, and in the `dev`
  extras); `pyproject.toml` declares no runtime dependency.
