# ADR-014: Promotion: the reference's vote count as agreement, a re-decode under views checked disjoint, one candidate adapter per round

- Status: Proposed, 2026-10-02; accepted with the human's merge of its protected change
  (`arc26/decode.py`, `needs-human`). $\tau$'s value is a starting value that P2.7 ablates.
  Source: task brief `docs/tasks/P2.2.md`; BRIEF B5 P2.2, B2 Q4, B4 (`promote`, the `DemoSet`).
- Tasks: P2.2; P2.3 (the rounds around it), P2.7 (the $\tau$ ablation), P2.8 (calibration).
- Related: ADR-004 (the `DemoSet`, rollback), ADR-011 (the split decode), ADR-013 (the held-out
  pair out of every prompt that feeds a promotion; the voted answer is attempt 1).

## Context

B5 P2.2: promote a test input's voted answer when agreement $\ge \tau$ "(reusing the notebook's
agreement score)", the gate passes with the candidate adapter, and the answer survives
re-decoding under "a fresh, disjoint augmentation set"; $\tau$ starts "at the notebook's
confidence threshold". Four points were open: what that score and threshold are; how to decode
under other views when `decode_puzzle` (protected) fixes its own, and what disjoint means; which
adapter re-decodes; and what "the candidate adapter" is when a puzzle has several test inputs.

## Decision

1. **Agreement: the count term of the reference's vote.** The voted answer $\hat y$ is
   `score_kgmon`'s first grid, the submission's attempt 1 (`arc26/nvarc/select.py` line 29,
   `run_selection_algo`'s default, `arc_decoder.py` line 68; ADR-013). For a distinct grid $g$
   with view candidates $C_g$ (`decode_puzzle` writes one candidate per parsed beam of a view,
   `arc26/decode.py` lines 443-447), `score_kgmon` (`arc26/nvarc/arc_decoder.py` lines 37-43,
   grouped and ranked by `score_sum`, lines 17-27) scores

   $$\mathrm{score}(g) = |C_g| - \frac{1}{|C_g|} \sum_{c \in C_g} \bar s(c),$$

   with $\bar s(c)$ the mean of $c$'s 8 rescoring NLLs. Its first term, `inf_score = len(guesses)`
   (line 38), is the augmentation agreement. The promotion's agreement is
   $a(\hat y) = |C_{\hat y}| / n$ over the $n = 16$ views decoded (`augment(n=2, seed=2)`),
   computed by running the moved `score_sum` with `getter_kgmon` (`arc26.loop.tally`), so the
   order and the terms are the reference's. It lies in $[0, 1]$ unless one view parses two beams
   into the same grid, which no view did for its voted grid among `P1.1-01`'s 2,335 candidates or
   `P1.3-01`'s.
2. **$\tau = 9/16$ to start.** The reference has no agreement or confidence threshold. Its
   thresholds are the DFS bound, a beam probability of at least 0.2 (`max_score = -np.log(0.2)`,
   `arc26/nvarc/arc_solver.py` line 306), which every candidate already meets, and
   `getter_full_probmul_3`'s `baseline=3` (`arc_decoder.py` line 29), an offset in the selection
   the submission does not use. $\tau$ is therefore proposed from run `P1.1-01`'s candidate files
   (DEV120, the notebook configuration, no hold-out), labeled with the correctness the notebook
   printed in development mode (its `ALL_CORRECT` lines, read as
   `scripts/session_log_timing.py` reads them; no solutions file was read). Voted answers by the
   views that produced them, of 161 test outputs with candidates:

   | Views | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | right | 5 | 1 | 0 | 4 | 3 | 1 | 1 | 2 | 3 | 2 | 4 | 2 | 2 | 2 | 1 | 7 |
   | wrong | 37 | 27 | 12 | 11 | 4 | 8 | 4 | 7 | 1 | 2 | 4 | 0 | 1 | 1 | 1 | 1 |

   The isotonic (pool adjacent violators) estimate of $P(\hat y \text{ right} \mid |C_{\hat y}|)$
   is 0.073 for 1 to 3 views, 0.244 for 4 to 8, 0.562 for 9 to 11, 0.700 for 12 to 15, and 0.875
   for 16. $\tau$ is the smallest agreement at which it reaches $1/2$: 9 of 16 views, a strict
   majority. At $\tau = 9/16$, 34 of the 161 outputs pass, 23 of them right (40 of 161 overall).
   Calibration is weaker on DEV40, a harder subset (`P1.1-01` was right on 8 of its 52 DEV40
   outputs): on `P1.3-01`'s notebook arm (51 outputs, 7 right) the estimate reaches $1/2$ only at
   16 views, and the loop's adapter trains on one pair fewer. Proposed for P2.7's three values:
   $\tau \in \{9/16, 3/4, 1\}$; P2.8 recalibrates on P2.6's logs.
   Caveats (review, 2 Oct): $\tau$ was chosen in sample, on DEV120's dev-mode labels, and DEV120
   contains DEV40 and is P2.6's subset, so the loop's DEV scores are optimistic about $\tau$. On
   DEV40 the voted answer is right less often at every level: over `P1.3-01`'s five arms pooled,
   18 of the 53 outputs at $\tau = 9/16$ are right (0.34, 95% CI 0.23 to 0.47), 16 of 31 at 12/16,
   11 of 19 at 16/16; on DEV120 without DEV40, 20 of 26 at 9/16. So on DEV40 the gate and the
   re-decode do most of the filtering. $\tau$ stays a parameter; P2.3 puts it in the run config,
   so P2.7 varies it without a code change.
3. **A re-decode under views checked disjoint, with the current adapter.** `decode_puzzle` gains
   one keyword-only parameter, `view_seed` (default None, the reference's views): its DFS views
   become `augment(n=2, seed=view_seed)` and its rescoring views
   `augment(seed=hash((bk, view_seed)) % 1024**2)`. Both change, since the vote subtracts the
   rescoring NLLs. The re-decode decodes $x_i$ alone (test input 0 of the puzzle's key) with the
   adapter $\theta$ that produced the candidates, not the candidate adapter, which was trained on
   $(x_i, \hat y_i)$ and would reproduce it. Disjointness is computed, not assumed:
   `arc26.loop.view_keys` runs the moved `ArcDataset` calls that `decode_puzzle` runs (the first
   decode's draws depend on how many test inputs it decoded), and views are compared by their
   image of the re-decode's prompt grids, $D$'s pairs and $x_i$, under the moved `forward_mod`.
   Equal (transform, permutation) views have equal images, so this is stricter than the brief's
   definition; it also catches permutations that differ only on colors the puzzle does not use
   and transforms its grids are symmetric under. The re-decode uses the first seed from
   `FRESH_SEED` (3) upward, at most 64 tried, whose views share no image with the first decode's
   16 decoding and 8 rescoring views; when none does, nothing is promoted. Its voted answer must
   equal $\hat y$ exactly and not be tied.
4. **One candidate adapter per round.** The inputs that pass the free checks and the re-decode are
   fit together: $\theta' = \mathrm{adapt}(D \cup \{(x_i, \hat y_i)\}, \theta)$ (Phase 2:
   `sgd_continue`, injected), on the state without its held-out pair, and $\theta'$ is gated once,
   its prompt holding the enlarged $D$. If it passes, every such pair is promoted: the `DemoSet`
   moves to version $+1$ and $\theta'$ is the round's adapter. If not, none is, and the state and
   adapter stay the previous ones (ADR-004).
5. **Order and failures.** The checks run cheapest first and stop at the first that fails: a
   candidate exists, the vote's top two grids do not tie (the candidates' order would have picked
   the answer), $a(\hat y) \ge \tau$, the input is not already a pseudo-pair, a held-out pair
   exists (without one there is no gate, so no promotion: $k = 1$, or $k = 2$ with the flag
   off), the views are disjoint, the re-decode agrees, the fit, the gate. A re-decode, fit, or
   gate that raises fails its check with its error logged. Each test input gets one record
   (`promote-<worker>.jsonl`), and a promoted pair carries its round, test input, agreement, and
   verdicts.

## Alternatives considered

- **The full `score_kgmon` value as the agreement.** Where a threshold sits, both select the same
  outputs on `P1.1-01` (for the top 8, 10, 13, 16, and 18 outputs, as many right); over all
  outputs its AUROC is 0.83 against the count's 0.79, a difference in the low-agreement region no
  threshold reaches. Lost: not in $[0, 1]$ (P2.8's reliability diagram), and its scale moves with
  $n$.
- **The DFS bound, 0.2, as $\tau$.** A beam's probability under its own view, met by every
  candidate; another quantity than agreement.
- **Disjointness on (transform, permutation) alone.** With $10!$ permutations it practically never
  fails, so the check would pass while the model sees prompts it has seen: a two-color prompt has
  $10 \cdot 9 = 90$ recolorings, and on a synthetic two-color puzzle 241 of 400 seeds share a view
  with the first decode. Even the reference's own seed shares views with a re-decode of one input
  alone: one view when the puzzle has two test inputs, all 16 when it has one
  (`tests/test_promote.py`).
- **Fresh decoding views, the reference's rescoring.** The re-decode's vote would subtract the
  same NLLs as the first decode's for every grid both found: 8 views in common.
- **Re-decode with the candidate adapter.** It has trained on the pair under 128 views.
- **A candidate adapter per test input.** Attributes a failing gate to one pair, but costs one fit
  and one gate decode per input where the governor prices one per round (P2.4), and P2.3 would
  still fit the union. 49 of DEV120's 120 puzzles have two or more test inputs.
- **A function beside `decode_puzzle`.** Duplicates the 112 moved lines of the decode block; the
  parameter changes two seeds, and the AST test checks that its default is the reference.
- **Promote without a gate when no pair is held out.** Two checks, not three.

## Consequences

- `arc26/decode.py` changes (protected, `needs-human`): with `view_seed` at its default every
  statement is the reference's (`tests/test_split_solver.py` checks it on the AST, and the split
  solver's stand-in runs stay byte for byte equal to the reference); the P0.1 parity fixture and
  the CPU replay do not read it.
- Cost per round, for P2.3 and the governor (P2.4): per input that passes the free checks, one
  re-decode, 4 DFS batches (`redecode_decode`) and their rescoring (`redecode_selection`), priced
  as `CostModel.decode_cost(4)`; then one fit (`candidate_train`, the train model on the
  enlarged set's tokens and steps) and one gate check (`gate_batches`). `redecode` and `gate`
  record the spans.
- P2.3 builds the round from `Promotion`: the next `DemoSet` and $\theta'$, or the previous ones.
  A provider that trains in place must keep the previous weights for the rollback.
- P2.8 gets the agreement of every voted answer with candidates, promoted or not.

## Overturn criterion

- $\tau$: P2.7's ablation. If $3/4$ or $1$ scores at least 2 tasks above $9/16$ on DEV40, the
  default moves there.
- One candidate adapter: if P2.6's logs show joint gates failing in at least 2 tasks where one
  input's pseudo-pair was right and another's wrong (dev scoring), measure a per-input fit on
  DEV40 before P2.7's table.
- Disjointness: if the seed search exhausts its tries on real puzzles more than once in a DEV120
  run, report those puzzles; no promotion stays the default for them.
