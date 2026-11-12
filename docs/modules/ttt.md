# Module: `arc26/ttt.py`

Task P2.0 (BRIEF B4, B5 P2.1 to P2.3), ADR-011; `SGDProvider`, task P2.3 (ADR-015). Tests:
`tests/test_split_solver.py`, `tests/test_rounds_solver.py`. Imports unsloth (first), torch,
peft, datasets, and transformers, which only the Kaggle image provides.

## Purpose

The per-puzzle test-time training of the reference notebook (B4: "per-puzzle LoRA fit (from the
notebook)"), moved out of `arc_solver.worker` without change so that the refinement loop can
train on a changed pair set and decode separately. `arc26.solver.worker` calls `fit` once per
puzzle, before `arc26.decode.decode_puzzle`. The loop's `SGDProvider` (B4, task P2.3) wraps it
in Phase 2: a refinement round's candidate adapter continues from the current one,
$\theta_t = \mathrm{sgd\_continue}(\theta_{t-1}, D_t, s')$ (ADR-004), and is rolled back when the
gate fails.

## Math

The adapter is rsLoRA on the target modules: $W' = W + \frac{\alpha}{\sqrt{r}} B A$ with
$\alpha = 32$ and $r$ = `lora_r` (256 by default; `embed_tokens` and `lm_head`, when targeted,
are trained as full copies). Before every puzzle the adapter is reset to `default_weights`, its
weights right after PEFT wrapping, so no puzzle starts from another's training.

The training set is the puzzle's demonstration pairs under $8 \times 16 = 128$ views (the 8
dihedral transforms, each with 16 random color permutations, `augment(n=16, shfl_keys=True,
seed=1)`), each formatted as one conversation and cut to 8192 tokens. One epoch with batch 1 is
128 steps of AdamW (learning rate $5 \cdot 10^{-5}$, cosine schedule, 10% warmup, bf16). The
collator keeps the labels of the assistant turns only, so the loss is the mean token
cross-entropy over the output grids:

$$\mathcal{L}(\theta) = -\frac{1}{|R|} \sum_{t \in R} \log p_\theta(x_t \mid x_{<t}),$$

where $R$ is the set of positions from the third token of each assistant reply through its end
token.

## Interfaces

| Name | What |
|---|---|
| `fit(model, default_weights, arc_test_set, key, formatter, max_seq_length, tokenizer, collator, train_args, timing_logger, rank) -> (model, puzzle_ds)` | Resets the adapter, trains on puzzle `key` of `arc_test_set`, prints the peak training memory and the trainer's statistics, resets the peak for decoding, and records `train` (seconds, the prepared dataset's token count) when `timing_logger` is set. Returns the model in inference mode, which the caller uses from then on, and the puzzle as a one-key `ArcDataset` for `decode_puzzle`. |
| `UnslothFixedTrainer` | `UnslothTrainer` whose `compute_loss` clones the loss before DDP scaling (unsloth issue 2435). |
| `QwenDataCollatorForCompletionOnlyLM` | Causal-LM collator that sets every label outside the assistant replies to -100. |
| `SGDProvider(*, key, formatter, max_seq_length, tokenizer, collator, train_args, timing_logger, rank, steps)` | Task P2.3, our code beside `fit` (ADR-015), `arc26.loop.AdapterProvider`. `adapt(demos, init, budget_s)` copies the adapter's weights (`get_peft_model_state_dict`) to the CPU and calls `fit` with that copy as `default_weights` and `train_args` plus `max_steps` $= s'$, on `demos.task()`: $s'$ of the 128 views, from the current adapter; records `snapshot` and, through `RenamedOps`, `candidate_train`. `rollback(previous)` loads the copy back (`set_peft_model_state_dict`, `for_inference`, cache release; records `restore`). `fingerprint(adapter)` is a blake2b digest of every adapter tensor's name, dtype, shape, and bytes (records `fingerprint`). `fit_shape(demos)` builds the 128 views as `fit` does, tokenizes them, and returns $(\lceil T s'/128 \rceil, s')$ for the governor, numpy's random state kept. |

The parameters are the reference worker's locals, with their names; `arc26.solver.load` returns
all of them but `key` and `rank`.

## Data flow

`arc26.solver.worker` takes a task id from the queue, calls `fit`, then
`arc26.decode.decode_puzzle` with the model and `puzzle_ds` that `fit` returns. `fit` reads
numpy's global random state through `augment(seed=1)`, which reseeds it, as the reference does.

## Origin

`reference/reference_notebook.ipynb`, cell 4 (`%%writefile arc_solver.py`), through the
generated `arc26/nvarc/arc_solver.py`: `logging.disable` (line 36), `UnslothFixedTrainer` (lines
60-85), `QwenDataCollatorForCompletionOnlyLM` (lines 88-105), and `fit` (lines 331-382, the
training block of the worker's per-puzzle loop). Changes: type hints, docstrings, and the return;
long lines wrapped. `tests/test_split_solver.py` lists them and checks every statement on the AST.

## Tests

`tests/test_split_solver.py`: the statements equal the reference's; the parameters are the
locals the block reads, in order of first use, and the return values the locals the worker
reads afterwards; `fit` takes no decorator, default, `*args`, or `**kwargs`; the module's
imports, by module and alias, are the reference's with the listed differences; the wired solver,
against stand-ins for the GPU stack, gives the reference's candidates, random state, stand-in
token counts, timing records, and printed lines.

## Known limits

- Importing the module disables logging at `WARNING` and below for the whole process, as
  importing `arc_solver` does.
- `augment(seed=1)` reseeds numpy's global random generator; a caller that relies on numpy's
  random state must reseed after `fit`.
- The peak-memory counters are process-wide: `fit` resets them on entry and before it returns.
- The signature is the reference worker's locals; P2.x may wrap it, but a change to the moved
  statements needs its own parity check (ADR-011). `SGDProvider` wraps it without changing it: a
  copy of the current adapter as `default_weights` and `max_steps` in `train_args`.
- `SGDProvider`'s copy is about 1.1 GB of CPU memory at rank 256 on the seven projections; the
  continuation starts a fresh AdamW state and `fit`'s cosine schedule over $s'$ steps.
