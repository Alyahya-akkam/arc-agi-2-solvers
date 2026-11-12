# Module: `arc26/decode.py`

Task P2.0 (BRIEF B4, B5 P2.1 to P2.5), ADR-011; `view_seed`, task P2.2 (ADR-014). Protected path
(BRIEF A0): every change must keep the P0.1 parity fixture green, and its PRs carry `needs-human`.
Tests: `tests/test_split_solver.py`, `tests/test_promote.py` (`view_seed`). Imports unsloth (first), torch, and transformers through
`arc_loader`, which only the Kaggle image provides.

## Purpose

The reference notebook's decoding (B4: "DFS decode, voting, the two selection functions (from the
notebook)"), moved out of `arc_solver.worker` without change so that the refinement loop can
decode inputs that are not test inputs, such as the held-out pair's input. `arc26.solver.worker`
calls `decode_puzzle` once per puzzle, after `arc26.ttt.fit`. The voting and selection over the
candidate files stay in the moved cells 3 and 7 (`arc26/nvarc/arc_decoder.py`, `select.py`).

## Math

DFS keeps a continuation while its cumulative negative log-likelihood stays under a bound,

$$-\sum_{k=1}^{m} \log p(y_k \mid x, y_{<k}) < -\ln 0.2 \approx 1.61,$$

over the tokens of `ARC_TOKENS` (the digits, the newline, and the end token); a suffix ends at
`EOS_ID`. Each test input is decoded under 16 views (the 8 dihedral transforms, each with 2
color permutations, `augment(n=2, seed=2)`), four per batch: per test input the identity and 180
degree views, then the 90 and 270 degree ones, then the same for the transposed views, so each
batch holds prompts of one length. A parsed grid is mapped back by its view's inverse. Each new
grid of test output $b$ (the key `<task>_<index>`) is rescored under the 8 views of
`augment(seed=hash(b) % 1024**2)` with `calc_scores`, which returns
$-\sum_t \log p(a_t \mid q, a_{<t})$ for each view's answer $a$ given its query $q$;
`PYTHONHASHSEED` fixes the seeds, and a grid already scored for $b$ reuses its scores. Selection
(cell 7) combines these scores.

With `view_seed` $= s$ (P2.2) the same code decodes under other views: `augment(n=2, seed=s)` and
rescoring under `augment(seed=hash((b, s)) % 1024**2)`, in the same batch order. The promotion's
re-decode uses it; `arc26.loop.view_keys` computes the views of any call and `promote` checks
that they share none with the decode whose candidates it promotes.

With `views` (task P3.7, ADR-019) it decodes a subset of each test input's 16 views: batches of
indices $2d + c$ into the input's sorted view keys ($d$ the dihedral map of
`arc26.arms.DIHEDRAL_ORDER`, $c$ the color permutation), each batch's prompts of one length (the
maps that keep rows as rows, or those that swap them; `arc26.arms.check_views`), decoded input by
input; rescoring is unchanged. `arc26.arms.DIHEDRAL_ONCE` $= ((0, 4, 10, 14), (2, 6, 8, 12))$ is
GEN's: every map once, two full batches. The default None runs the reference's statements: the
batch loop reads `batches if views is None else <the views' batches>`.

## Interfaces

| Name | What |
|---|---|
| `decode_puzzle(puzzle_ds, formatter, max_seq_length, max_new_tokens, start_time, end_time, rank, key, tokenizer, model, max_score, timing_logger, dir_outputs, *, view_seed=None, views=None) -> None` | Decodes every test input of the one-key `puzzle_ds`, rescores each new grid, and writes one bz2 pickle per view that yields a grid to `dir_outputs`, named by the view's key and holding a list of `{"beam_score", "score_aug", "solution"}`. Stops at 1200 s after `start_time` or at `end_time`. Records `decode` per DFS batch (prompt tokens) and `selection` per rescored grid (query and answer tokens) when `timing_logger` is set. `view_seed` (P2.2): None, the reference's views; an integer, the views it seeds (Math). `views` (P3.7): None, the reference's 16 views and batches; batches of view indices, those views only (Math). |
| `inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time)` | One prefill over `n` prompts of equal length (`prefix_tokens`, `[n, prompt_len]`), then `turbo_dfs`; returns `(row, beams)` with the beams in ascending NLL. |
| `turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time)` | The batched depth-first search from `logits` (`[n, vocab]`) and the cache; stops after 540 s or at `end_time`. |
| `calc_scores(queries, answers, tokenizer, model)` | The NLL of each answer given its query, in one batched forward pass. |
| `ARC_VOCAB`, `ARC_TOKENS`, `USER_TOKEN_ID`, `ASSISTANT_TOKEN_ID`, `PAD_ID`, `EOS_ID` | The token ids of the model's grid vocabulary; `arc26.ttt`'s collator reads three of them. |

`decode_puzzle`'s parameters are the reference worker's locals, with their names;
`arc26.solver.load` returns all of them but the puzzle's own (`puzzle_ds` from `fit`, `key`,
`start_time`, `rank`) and `end_time`. The worker leaves `view_seed` and `views` at their
defaults; `arc26.arms` passes `views`.

## Data flow

`fit` returns the trained model and `puzzle_ds`; `decode_puzzle` writes the candidate files that
`arc26.runner.make_submission` (the moved cell 7) selects from after the workers finish.

## Origin

`reference/reference_notebook.ipynb`, cell 4 (`%%writefile arc_solver.py`), through the
generated `arc26/nvarc/arc_solver.py`: the constants (lines 38-57), `turbo_dfs` (lines 108-175),
`inference_turbo_dfs` (lines 178-197), `calc_scores` (lines 200-228), and `decode_puzzle` (lines
384-495, the decode block of the worker's per-puzzle loop). Changes: type hints and docstrings;
long lines wrapped; task P2.2's keyword-only `view_seed=None`, with which the seeds of the two
`augment` calls read `<the reference's seed> if view_seed is None else <a seed from view_seed>`.
`tests/test_split_solver.py` lists them and checks every statement on the AST, the two seeds with
their first branch, the reference's.

## Tests

`tests/test_split_solver.py`: the functions and the constants equal the reference's; the
parameters of `decode_puzzle` are the locals the block reads, in order of first use, and it
returns nothing because the worker reads none of its locals afterwards; against stand-ins for
the GPU stack, the split solver writes the reference's candidate files byte for byte, with the
same `decode` and `selection` records, also when the 1200 s cap cuts a puzzle.
`tests/test_promote.py`: on the same stand-ins, `decode_puzzle` with `view_seed` at its default
and with a seed decodes and rescores under exactly the views `arc26.loop.view_keys` predicts.

## Known limits

- Importing the module imports unsloth, so that its patches of transformers apply before
  `arc_loader` imports transformers, whichever of `arc26.decode` and `arc26.ttt` a process
  imports first.
- `augment(seed=...)` reseeds numpy's global random generator, as in the reference.
- Decoding stops 1200 s into a puzzle (`start_time` is the puzzle's start, not the decode's), and
  each DFS call stops after 540 s: the reference's limits, unchanged.
- The module holds no voting or selection: those stay in `arc26/nvarc/` until a later task moves
  them under the same parity rule.
