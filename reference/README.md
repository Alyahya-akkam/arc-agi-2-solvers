# Reference notebook

`reference_notebook.ipynb` is the 2025 ARC Prize (ARC-AGI-2) first-place test-time-training
notebook of the NVARC team, `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue` on Kaggle,
pulled unmodified with its metadata on 30 Sep 2026 by the human (`kaggle kernels pull ... -m`).
Its Kaggle page states that it is released under the Apache License 2.0 (confirmed by the human,
30 Sep 2026). It is the source of `ttt.py`, `decode.py`, `augment.py`, `tokenize.py`, and
`runner.py` (BRIEF B9); code extracted from it keeps a header naming its source cell and this
license.

| File | SHA-256 |
|---|---|
| `reference_notebook.ipynb` | `506b85db66f9949022e87935ccb06a73fd78cb690883bd600bfb5c7d36c2b3af` |
| `kernel-metadata.json` | `b1d9ce0f7c0137db9e80a85b57f5721decc0bebf21847c13f275525137271c9a` |

`kernel-metadata.json` is the pulled metadata (sources, pinned image), read by
`scripts/build_reference_run.py`. Neither file is edited: the P0.1-01 build writes its patched copy
under `dist/`. The Apache License 2.0 text is the repository's `LICENSE`; `NOTICE` lists this
notebook and the files derived from it.
