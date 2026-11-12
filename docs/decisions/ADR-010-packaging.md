# ADR-010: Upload only the arc26 wheel; the stack is the pinned image plus the install notebook

- Status: Accepted, 2026-09-30 (task P0.5). Resolves flagged conflict 4 in `docs/STATE.md`.
- Tasks: P0.5 (builder), every milestone build (P1.2, P2.6, P4.4, P4.5).

## Context

- BRIEF A5 describes `dist/<tag>/arc26-code/` as the "wheel of arc26 plus pinned dependency
  wheels". ADR-008 caps a local upload at 200 MB and keeps large artifacts on Kaggle, and the
  human has little disk and a slow connection.
- The method runs on NVARC's stack, as run A of `P0.1-01` recorded it (`docs/experiments/P0.1.md`):
  Python 3.11, torch 2.8.0 (CUDA 12.8), transformers 4.55.4, peft 0.17.1, unsloth 2025.9.7,
  unsloth_zoo 2025.9.9, flash-attn 2.8.2, xformers 0.0.32.post2, triton 3.4.0. A wheelhouse of it
  is several GB (**estimate**: torch's CUDA wheel and the CUDA libraries it pulls in are each
  hundreds of MB), and flash-attn needs a prebuilt wheel that matches torch, CUDA, and Python.
- K0 (`docs/experiments/K0.md`): Kaggle's latest image is Python 3.12 with transformers 5 and no
  unsloth or flash-attn, so it cannot run that stack. The reference's own image, pinned by digest,
  still runs.
- Runs A and B of `P0.1-01` ran on that pinned image with the install notebook
  `sorokin/pip-install-unsloth-flash-patch` at version 2 attached. Both recorded exactly the 2025
  stack, with identical `pip freeze` output.
- `arc26` is pure Python. Today it imports the standard library, numpy, and the stack.

## Decision

1. The only file uploaded from the local machine is the `arc26` wheel (tens of KB), as the private
   dataset `alyahya737/arc26-code`. `scripts/build_submission.py` writes it to
   `dist/<tag>/arc26-code/`.
2. Every other package comes from the reference's Python 3.11 image, pinned by digest
   (`docker_image_pinning_type: original`), plus the install notebook
   `sorokin/pip-install-unsloth-flash-patch` at version 2, attached as a kernel source. This is run
   A's stack, and `kernel-metadata.json` takes the image and the model source from
   `reference/kernel-metadata.json` through the P0.1-01 builder's `kernel_metadata`.
3. The notebook checks the wheel's sha256 against the value baked into it, then installs it with
   `python -m pip install --no-index --no-deps --force-reinstall`, run by the `python` on PATH
   (the interpreter that runs the solver). Because of `--no-deps`, the dependency list in
   `pyproject.toml` never changes the stack on Kaggle; pins there are lower bounds that the stack
   satisfies.
4. The notebook finds the wheel, the data, and the model by name under `/kaggle/input`, never by a
   hardcoded mount path.
5. Builds are reproducible: a clean tree, `git archive`, and `SOURCE_DATE_EPOCH` set to the time of
   the last commit that changed the wheel's inputs, so commits that touch nothing the wheel holds
   keep its sha256. `MANIFEST.md` records the commit, the wheel builder, and the hash of every
   file.

## Alternatives considered

- **A full wheelhouse in `arc26-code/`, as A5 describes.** Lost: several GB, far over ADR-008's
  200 MB cap and the human's connection. A set resolved on this machine could also differ from
  run A's.
- **A wheelhouse built on Kaggle by our own internet-on CPU notebook, kept as a Kaggle dataset.**
  Not now: it would remove the dependency on a third-party notebook, but flash-attn needs a
  prebuilt wheel that matches torch, CUDA, and Python, and it costs a session and a review while
  giving no parity benefit over run A's proven stack. It is the fallback named in the overturn
  criterion.
- **The latest image with our own installs.** Lost: Python 3.12 or 3.13 and transformers 5 break
  NVARC's code (K0), and the parity check needs run A's stack.
- **Code written by notebook cells (`%%writefile`), as the reference does.** Lost: A5 requires an
  installable package built from a commit, and a notebook that carries the whole package cannot
  be read on its own.

## Consequences

- Submissions depend on a pinned third-party notebook version and an image built in July 2025
  (`/etc/build_date` 20250701). Both are fixed references, but Kaggle could retire them.
- A new runtime dependency of `arc26` (for example in Phase 3) must already be in the stack; the
  environment-check cell prints the versions the stack provides. Anything missing needs a
  Kaggle-side source and a new ADR, since `--no-deps` fetches nothing.
- Uploads stay tiny, and each code dataset version holds one wheel. A development run refuses a
  wheel whose sha256 differs from its build; a competition rerun logs the mismatch and installs
  it anyway, to still score.
- The comment on runtime dependencies in `pyproject.toml` now points here.
- Task P0.5d (script notebooks): a script build's `arc26-code/` also holds its script and the
  script's config, byte for byte beside the wheel, and the notebook runs them only when their
  sha256 match its build. A dataset version made from a script build therefore serves that
  script notebook and every notebook of the same wheel; a version made from a submission build,
  or from another script's build, lacks the script, and the script notebook stops in its
  environment check. Version the dataset from the build that runs next; script notebooks queued
  at the same time need one version that holds all their files, or code datasets of their own
  (`code_dataset` in the config).

## Overturn criterion

The pinned image or version 2 of the install notebook becomes unavailable, or no longer reproduces
run A's `pip freeze`; or a milestone needs a package the stack lacks. Then build our own wheelhouse
on Kaggle (the second alternative) and supersede this ADR.
