# ADR-015: Round mechanics: continue from a snapshot, roll back by restore, decode the open inputs after a kept round

- Status: Accepted, 2026-10-03: run `P2.3-01` passed its decision rule
  (`docs/experiments/P2.3-01.md`): on the GPU, two rounds kept (`36a08778`, `981571dc`, a
  32-step fit each), two with nothing to promote, no raised or cut op, every final decode
  installed or skipped `all_promoted`, timing records complete; snapshot and fingerprints 5.4 s
  per round, under the overturn criterion's 10 s. No round rolled back there; decision 3's
  restore was then shown on the GPU by run `P2.5-02` (3 Oct, rank 16 on q, k, v, o): its two
  rolled-back rounds (`b0039139`, `dfadab01`) have `restored` true and equal fingerprints, 0.2 to
  0.7 s per round (`docs/experiments/P2.5-02.md` on `task/P2.5`). Proposed 2026-10-03. Source:
  task brief `docs/tasks/P2.3.md`; BRIEF B5 P2.3, B2 Q4, B4 (`refine`, `AdapterProvider`).
- Tasks: P2.3 (the rounds); P2.5 (the in-context provider, which plugs into the same round);
  P2.6 and P2.7 (the settings this ADR makes run parameters); P2.8 (the round folders).
- Related: ADR-004 (the `DemoSet`, rollback, $\mathrm{sgd\_continue}$), ADR-011 (the moved `fit`
  and `decode_puzzle`, unchanged), ADR-013 (the held-out pair out of every prompt that feeds a
  promotion, and, as amended on 3 Oct, kept in the decode that is submitted), ADR-014
  (promotion, one candidate adapter per round).

## Context

B5 P2.3: "one round = promote, enlarge `DemoSet` (version + 1), `provider.adapt(demos,
init=previous, budget)`, gate check, keep or roll back (previous adapter and previous `DemoSet`
version). Maximum rounds is a parameter, default 1, hard cap 2." `promote` (P2.2) already makes
one round's decision. Missing were the provider, the rollback of weights that the GPU fit trains
in place, what is decoded and submitted after a kept round, the governor's decision, the wiring
into the solver, and the settings' path to Kaggle, where the wheel carries `arc26/` only. P2.2's
review added two points: the re-decode's batches had no term in the governor's round cost, and a
re-decode cut short by `decode_puzzle`'s 1200 s cap still passed on its top grid.

## Decision

1. **One round (`arc26.loop.refine`).** The free checks of `promote` run first, on the CPU
   (`screen`: a candidate, no tie, $a \ge \tau$, not yet a pseudo-pair, a held-out pair, a fresh
   view seed), and their records are written whatever follows, so P2.8 gets every voted
   answer's agreement (ADR-014): a puzzle without a held-out pair records `no_held_out`, and a
   declined round records `declined` for the inputs that passed them. No input passing them ends
   the round without a plan. Otherwise the governor prices the round from them and plans it
   (decision 6); with `gradient`, `promote` runs with the provider's `adapt` as its candidate
   fit. A passing gate keeps the round: the state moves to version $v + 1$ and the candidate
   adapter is the round's. Anything else returns the state as given (version $v$, the same
   immutable object) and, when the candidate fit ran, calls `provider.rollback`. An error
   outside `promote`'s own checks rolls back the same way and is recorded; the worker goes on.
2. **$\mathrm{sgd\_continue}$ is `fit` from a snapshot (`arc26.ttt.SGDProvider`).** `fit` resets
   the adapter to the weights it is given, then trains. `adapt` copies the current adapter's
   weights to the CPU (`get_peft_model_state_dict`: every LoRA factor, and the trained copies of
   `embed_tokens` and `lm_head`) and calls `fit` with that copy as `default_weights` and
   `train_args` plus `max_steps` $= s'$. No moved statement changes (ADR-011's AST test stays as
   it was). $s' = 32$ (`Rounds.continue_steps`): a quarter of the 128-view epoch, as ADR-004's "a
   fraction of an epoch". Its cost, from `P2.1-01`'s fits (**measured**: 121.9 s per 128-step fit
   on 151,074 tokens on average, DEV40 with a pair held out; the governor's attention model gives
   118.4 s for that fit): with one pseudo-pair the enlarged set has about $k/(k - 1)$ times the
   tokens, about 226,600 at $k = 3$, of which 32 steps train on a quarter; the model gives
   $24.35 + 0.0005812 \cdot 56{,}653 + 3.487 \cdot 10^{-8} \cdot 56{,}653^2 / 32 \approx 61$ s
   (**estimate**), against about 170 s for a full epoch. The learning-rate schedule is `fit`'s
   (cosine, 10% warmup) over the 32 steps, with a fresh AdamW state.
3. **Rollback by restore.** The CPU copy is the rollback: `rollback` loads it back with
   `set_peft_model_state_dict`, the call `fit` uses for its own reset, and returns the model in
   inference mode. The round's record holds a blake2b digest of the adapter's raw bytes before the
   candidate fit and after the restore (`fingerprint`); equal digests show the restore was bit for
   bit. The digest is taken before the fit can change anything, and the candidate fit counts as
   run only after it, so a digest that raises rolls nothing back; `adapt` drops the previous
   round's copy before taking its own, so a copy that raises leaves nothing to restore, and the
   adapter, unchanged, stays as the previous round kept it. The copy stays on the CPU: about
   0.53G values, 1.1 GB in bf16 per worker at the notebook's rank 256 (`P2.1-01`'s scope); the
   GPU's headroom for it (13.1 GB training and 15.6 GB inference peaks of 22 GB, `P1.1-01`) is
   not measured.
4. **After a kept round, the open inputs take the kept adapter's candidates.** The kept adapter
   decodes the test inputs that are not pseudo-pairs, in one `decode_puzzle` call with the
   enlarged set's pairs in the prompts, into the round's folder, whose files are then named by the
   puzzle's test indices; their candidates replace those inputs' files in the run's candidate
   folder. A pseudo-pair's input keeps the candidates that promoted it and is not decoded: its own
   pair in the prompt would answer its query (the views would copy it, and its second attempt
   would become the placeholder `[[0]]`). When every input is a pseudo-pair, no decode runs. An
   input's views depend on the inputs decoded before it in the same call, so the next round's
   view check (ADR-014) takes that call's layout (`first_inputs`), and its free checks see a
   pseudo-pair's input with the candidates that promoted it, which stop at `already_promoted`.
   Before the first replacement, the puzzle's files are copied to `<round_dir>/<task>/round0/`
   for P2.8. With the final decode (decision 8, the default) this decode runs only after a kept
   round before the last, where it feeds the next round's promotion; after the last round the
   final decode takes its place, with the same inputs and the held-out pair back.
5. **A cut decode fails.** `decode_puzzle` (protected) returns nothing, so the round counts its
   DFS batches through the timing logger it is given (`OpWatch`): a decode that ran fewer than 4
   per input was stopped by the 1200 s cap or the end time. A cut re-decode fails with its own
   reason, `redecode_cut` (its candidates are recorded, and it never agrees); a cut gate decode
   raises `DecodeCut`, which fails the gate (`gate_raised`); a cut decode after a round installs
   nothing and ends the rounds.
6. **The governor decides before any GPU op of the round.** The round is priced as the free
   checks bound it: 4 re-decode batches per input that passed them, the candidate fit on the state
   with all of their pseudo-pairs (`provider.fit_shape`: the tokens of $s'$ of the fit's 128 views,
   in $s'$ steps), the gate's 4 batches, and the test decodes that may follow the round, at the
   puzzle's slowest round-0 batch. Those are counted by `run_rounds` and passed to `refine` as
   `decode_batches`, the plan's `test_batches`. With the final decode (decision 8, the default),
   the last round reserves it for every open input, $4 n_\text{open}$ (a round that keeps
   nothing leaves them all open, and the final decode follows whatever the round does), and an
   earlier round also its own decode after a kept round, $4 \max(n_\text{open},
   2(n_\text{open} - 1))$; without it, the decode after a kept round, 4 per input still open
   after it, at most $4(n_\text{open} - 1)$ (a kept round promotes at least one). A round the
   governor accepts thus leaves the final decode the estimate its own check asks for (decision 8)
   unless the round overruns its price. `CostModel.round_costs` gains `redecode_batches` for the
   first term, priced in both variants at the round-0 prompt length (the re-decodes run before
   any pair is added). Only the gradient variant is offered (the in-context one is P2.5's), so
   the plan is `gradient` or `none`; it is recorded beside the verdicts, with the shape it
   priced. The puzzle's budget comes from the queue as the worker took the puzzle
   (`Governor.puzzle_budget`, the raw `qsize()`). The decode trade (`decode_policy` `trade`) is
   refused: it needs a plan before each baseline DFS batch, inside the protected
   `decode_puzzle`.
7. **Rounds.** `max_rounds` is 1 by default, at most 2 (`MAX_ROUNDS`). Round 2 runs only after a
   kept round 1, on the candidates of the decode that followed it; a round that keeps nothing ends
   the rounds, since its state and adapter are the ones that produced the candidates it saw.
8. **The decode that is submitted keeps the held-out pair** (`Rounds.final_in_prompts`, true by
   default). Run `P2.1-02` recovered 2.0 DEV40 tasks with the pair in the test prompts (7.0
   against `P2.1-01`'s 5.0 of 40, its Recovers row, at the noise floor), so ADR-013 as amended
   (3 Oct, on `task/P2.1-02`) keeps it in the decode whose candidates are submitted, leaves it
   out of every decode that feeds a promotion, and has the governor price the extra decode
   (decision 6). The last round decodes nothing itself; after the rounds, whatever their outcome,
   one more decode runs with the current adapter (the last kept one, or round 0's, restored or
   untouched) and the held-out pair back among the real pairs at its place, then the pseudo-pairs
   (`DemoSet.task(held_out=True)`), of the open inputs only; their candidates replace those
   inputs', and an input it gives no candidate keeps its own. Every decode that feeds a
   promotion (round 0's test decode, the decode after a kept round before the last, the
   re-decodes, the gates) still leaves the pair out. The final decode runs when the time left
   covers its own estimate ($D(4 n_\text{open}, 1, s)$ plus the overhead; otherwise `no_time`),
   and not after a rollback whose restore did not give back the weights (`not_restored`: the
   adapter's state is then unknown, and the open inputs keep their candidates). A puzzle without
   a held-out pair has none: round 0's decode already held every pair. False keeps `P2.1-01`'s
   prompts: the last kept round's decode is submitted, without the pair, as round 0's. Under both
   values a pseudo-pair's input keeps the candidates that promoted it.
9. **The settings' path to Kaggle.** `Rounds` (`round_dir`, the governor's config, `max_rounds`,
   `tau`, `continue_steps`, `final_in_prompts`) rides inside the hold-out settings,
   `HoldOut.rounds`, so the solver JSON carries it as `holdout.rounds` only when set: a run
   without rounds passes task P2.1's JSON, and the worker's change stays inside its hold-out
   branch (it passes `queue.qsize()` and the worker count). The governor's config is the parsed
   `configs/governor.yaml` of the built commit, which the notebook builder embeds in the arm's
   settings; `Rounds` checks it with `Governor.from_mapping`.

## Alternatives considered

- **Recompute $\theta_t$ with a full fit from the default weights on $D_t$** (the reference's `fit`
  as is). Lost: ADR-004's Phase 2 continues from $\theta_{t-1}$, and a full epoch costs about
  170 s against 61 s (estimates above); P4.3 tests continue against recompute for H.
- **Keep the snapshot on the GPU.** A faster restore, but 1.1 GB more on a GPU whose inference peak
  is 15.6 GB of 22, unmeasured; the CPU copy costs about a second. Revisited if `P2.3-01`'s
  `snapshot`, `restore`, and `fingerprint` records exceed 10 s per round.
- **Roll back by refitting $\theta_{t-1}$.** A second fit, and not bit for bit (training on GPU
  is not deterministic).
- **Replace every input's candidates after a kept round** (the brief's literal reading). Lost:
  for a promoted input the decode answers from its own pair in the prompt (decision 4).
- **Decode each input alone, its own pseudo-pair left out of its prompt.** A real decode of the
  promoted inputs too, but one call per input, and the promoted inputs' attempt 1 is already the
  checked answer. P2.7 may measure it.
- **Decode every test input after a kept round, the promoted ones too, as round 0 does.** Keeps
  round 0's layout for the next round's view check without a layout parameter, but decodes
  candidates that are thrown away: about 400 s per promoted input of a DEV4 puzzle (`P2.0-01`).
  Lost to `first_inputs`.
- **Plan after the re-decodes, on the exact set of pseudo-pairs.** Lost: the re-decodes are the
  round's first GPU ops, and B5 P2.4 prices an op before it runs; the upper bound costs at most
  a declined round.
- **Let a cut re-decode vote on the views it decoded.** Lost (P2.2's review): one batch of 4 views
  could carry the check.
- **Ship the governor's YAML in the wheel as package data.** Lost: a second copy of the config and
  a `pyproject.toml` change; the builder embeds the commit's file, which the tests check.
- **`RunConfig.rounds` as a top-level JSON key.** Lost: rounds need the gate, and nesting keeps the
  worker's change inside the hold-out branch.

## Consequences

- `promote` runs its free checks through `screen`: the same checks in the same order, the same
  records, plus the reasons `redecode_cut` and `declined` (`refine`'s), the re-decode verdict's
  `cut`, and the first decode's layout (`first_inputs`, default every input).
- A round's GPU work, at `P2.1-01`'s DEV40 means (**estimate**): a re-decode about as long as
  the gate (261.8 s, measured), the fit about 61 s, the gate 261.8 s: about 0.6 ks, against
  0.75 ks of puzzle work in round 0 (fit, test decode, gate). The final decode (the default)
  then decodes the open inputs once more, about 262 s per input (the gate's decode of one
  input), about 370 s per DEV40 puzzle (ADR-013 as amended), unless every input was promoted: it
  takes the place of the decode after a kept last round, so a kept round costs no more than
  before, and a puzzle whose round keeps nothing, or that has nothing to promote, pays one test
  decode more. The governor prices a one-input puzzle's round, the final decode included, at its
  p90 batch and margin: about 3.0 to 3.2 ks for DEV4's (`P2.3-01`'s brief, section 6).
- Per worker during a round: 1.1 GB of CPU memory for the snapshot; two fingerprints (a CPU copy
  and a hash of the adapter each), and a restore after a failed gate.
- New timing ops: `redecode_decode`, `redecode_selection`, `redecode`, `candidate_train`,
  `gate_decode`, `gate_selection`, and `gate` of the candidate gate, `round` (a round whose
  promotion ran), `round_decode` and `round_selection` (the decode after a kept round),
  `final_decode` and `final_selection`, `snapshot`, `restore`, `fingerprint`. `candidate_train`
  carries P0.3b's count, the tokens of the 128 views the trainer prepared, not the $s'$ trained.
- New records beside the gate's verdicts: `plan-`, `promote-`, `round-`, and
  `decode-<worker>.jsonl`.
- The round folders (`<round_dir>/<task>/round0/`, `round<r>/`, `round<r>-gate/`,
  `round<r>-redecode<i>/`, `final/`) stay on Kaggle; P2.8 needs a pulled summary of them, which
  P2.6's build adds.
- A rerun never runs rounds until P2.6 decides how the loop enters production: the series that
  carries them is ignored in a competition rerun.

## Overturn criterion

- $s'$: if P2.7's ablation of `continue_steps` (32 against 128) gains at least 2 tasks on DEV40,
  the default moves.
- Promoted inputs keeping their candidates: if P2.7 measures that decoding them alone, without their
  own pseudo-pair in the prompt, gains at least 2 tasks on DEV40, decision 4 changes for them.
- The CPU snapshot: if `P2.3-01` or P2.6 measures more than 10 s per round in `snapshot`,
  `restore`, and `fingerprint` together, measure the GPU's headroom and move the copy there.
- Rollback: any rolled-back round of `P2.3-01` whose fingerprints differ (`restored` false)
  rejects decision 3 until the restore is fixed.
- The final decode's prompts: decision 8 follows ADR-013's overturn criterion; should `P2.1-03`
  restate `P2.1-02`'s recovery below the noise floor, the policy (the pair in its prompts, a last
  fit with it, or neither) is measured again on DEV40 before P2.6.
