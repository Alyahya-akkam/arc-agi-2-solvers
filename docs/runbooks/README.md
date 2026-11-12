# Runbooks

One file per GPU or training script (BRIEF A8), with these sections in this order:

1. Why this script exists: the question it answers and the gate or decision it feeds.
2. What it does, step by step, with math in LaTeX where there is any.
3. Why it is built this way: each non-obvious choice, the alternative considered, why it lost
   (cite the ADR, paper, or measurement).
4. How to run it: exact commands for a Kaggle notebook cell and for a plain GPU machine; what to
   attach; expected wall time, GPU memory, and disk.
5. Outputs: every file written under `runs/<task-id>/` and what it contains.
6. How to read the results: good, borderline, and bad, and the decision each triggers.
7. Failure modes: OOM, divergence, timeout, NaN, and the first thing to change for each.

Before each run, a pre-run brief `docs/prerun/<run-id>.md` (BRIEF A9) must be approved by the
human; `scripts/kaggle_ops.py` then launches it on Kaggle, or, for a training job with
`Provider: modal`, `scripts/modal_ops.py` on Modal (ADR-022, modal_ops.md). After the run, `kaggle_ops.py pull`
brings the small outputs into `runs/<task-id>/`; the analysis lands in
`docs/experiments/<task-id>.md` and `RESULTS.md`.
