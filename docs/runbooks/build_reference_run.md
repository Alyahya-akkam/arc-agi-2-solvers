# Runbook: `scripts/build_reference_run.py`

Task P0.1-01 (BRIEF B5 P0.1, A8, A9). Config: `configs/reference_run.yaml`. Pre-run brief:
`docs/prerun/P0.1-01.md`. Decisions: ADR-008 (Kaggle only through `scripts/kaggle_ops.py`) and
flagged conflict 5 in `docs/STATE.md` (parity with a nondeterministic reference). "Research" is
`docs/research/kaggle_first_runs.md` (branch `task/W40-plan`). It read third-party copies of the
notebook and Kaggle logs of forks; the pulled notebook and the K0 probe confirm its facts.

## Why this script exists

P0.1 moves the 2025 first-place notebook (`sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`)
into `arc26/` without behavior change. It proves this against a parity fixture: the reference's
own attempt grids on DEV4 (BRIEF A1, `runs/P0.1/reference_output.json`). Two facts stand in the
way:

1. The reference cannot run untouched in 2026. The competition data and the model mount at new
   paths, and Kaggle's current default image cannot import the Python 3.11 packages that the
   reference's install notebook provides (research, sections 1.3 and 1.4, open risk 1).
2. The reference is not deterministic. Its rescoring seeds come from Python's salted `hash()`,
   batched DFS is nondeterministic, and wall-clock caps cut work short (research, section 1.5;
   NVARC paper, sections 3.3 and 3.5). Two runs of the reference itself can therefore disagree,
   so "identical attempt grids" alone is not a usable acceptance rule (conflict 5).

The script builds the smallest patched copy of the reference that runs in 2026. It removes the one
source of nondeterminism that can go without touching the reference's code, the hash salt, and it
keeps the per-candidate outputs needed to analyze the fixture. Two runs of the built notebook, A
and B, give the fixture (A) and the noise floor (B against A). The result feeds the P0.1 fixture
and its acceptance rule, and through them gate G0 ("parity fixture green").

## What it does, step by step

### Build (local, CPU, under a second)

1. **Config.** Reads `configs/reference_run.yaml` and refuses to build while its text contains
   `TODO` on any line, comments included, so every entry's source comment is filled in too. After
   parsing with `yaml.safe_load` it refuses a key or value that still spells `TODO` (possible
   through a YAML escape). It then checks the competition slug, the hash seed
   ($0 \le s \le 2^{32} - 1$), and every replacement entry (`old`, `new`, `count`). The `old`
   literals must be distinct, and none may occur inside another.
2. **Inputs.** Reads the pulled notebook and the pulled `kernel-metadata.json`. With the config,
   these are the only inputs.
3. **Metadata.** Writes `kernel-metadata.json` from scratch (table in section 5 below).
4. **Path literals.** Replaces each configured literal in the sources of code cells, in one pass.
   Let $n_k$ be the number of occurrences of literal $k$ across all code cells and $c_k$ its
   configured `count`. The build continues only if $n_k = c_k$ for every $k$, and no occurrence of
   one literal overlaps a match of another. A source stored as a list of lines is patched line by
   line, so unchanged lines keep their exact bytes.
5. **Kaggle settings.** Drops the notebook's top-level `metadata.kaggle`.
6. **Added cells.** Prepends a markdown cell and the seed cell, and appends a markdown cell and the
   dump cell. They follow the notebook's conventions: list or string sources, and cell ids when
   the notebook is nbformat 4.5 or later or already has ids. Both code cells must compile.
7. **Write.** Writes `reference-run.ipynb`, `kernel-metadata.json`, and `patch.diff`. Every check
   runs before the first write, so a refused build writes nothing, and a rebuild from the same
   inputs is byte-identical.

### On Kaggle (each of runs A and B)

1. The seed cell sets `PYTHONHASHSEED` in the kernel's environment.
2. The reference's cells run unchanged. The `%%writefile` cells write `arc_loader.py`,
   `arc_decoder.py`, `arc_solver.py`, and `starter.py`. The launch line `!python starter.py ...`
   starts four workers, one per L4, which share a queue of DEV4 test outputs. For each task, a
   worker trains a LoRA adapter on the task's own demonstration pairs. It then decodes candidates
   with DFS over 16 views and rescores them under 8 views, writes them as bz2 pickles under
   `/kaggle/inference_outputs/<subkey>`, and contributes its attempts to `submission.json`. The
   last reference cell scores DEV4 against the evaluation solutions (development mode).
3. The dump cell writes `run_logs/candidates.json`, `run_logs/pip_freeze.txt`, and
   `run_logs/env.json` under `/kaggle/working`.

### The seed

The reference seeds the rescoring augmentations of each subkey $b$ (a string) with

$$s(b) = \mathrm{hash}(b) \bmod 1024^2 .$$

For strings, $\mathrm{hash}$ is a keyed hash. Python draws the key when an interpreter starts,
unless `PYTHONHASHSEED` is set; with a fixed value, $s(b)$ is the same function of $b$ in every run.
The variable is read only at interpreter start. It therefore covers `starter.py` and its workers,
which start after the seed cell. It does not cover the notebook kernel, which is already running;
in the research copy the kernel only writes files, launches `starter.py`, and scores, and none of
that is seeded from `hash()`.

## Why it is built this way

The rule: each change is needed either to run in 2026 or for the fixture, and none may change
results except through the hash seed. The variation that remains between runs (batched DFS,
wall-clock caps) is what run B measures.

### 1. Path literals, replaced with checked counts

- **Needed:** the 2026 mounts differ from 2025 (research, sections 1.3 and 2).
- **Cannot change results:** a path changes where a file is read, not what the file holds. Two
  assumptions, both listed in the brief: the 2026 competition's evaluation file holds the same
  DEV4 tasks as 2025's (both are the public ARC-AGI-2 evaluation set), and the model mount holds
  the same weights (model version 1 in every copy, research section 1.3).
- **Alternatives:**
  - *Symlinks from the 2025 paths, created in a prepended cell,* which would leave every original
    cell byte-identical. Lost: Kaggle mounts `/kaggle/input` read-only (whether new top-level
    entries can be created there is unverified), and a symlink hides the change from `patch.diff`.
  - *A path resolver that searches `/kaggle/input`,* as a June 2026 fork added (research, section
    1.1). Lost: it adds code to the reference's cells.
  - *Regular expressions, or replacement without a count.* Lost: a pattern can match text nobody
    reviewed, and without a count a different version of the notebook is patched silently. A
    literal with a count fails closed.
  - *Sequential `str.replace`, one entry at a time.* Lost: a later literal could match inside an
    earlier entry's replacement. One pass never searches replaced text again.
- **Code cells only:** markdown and raw cells do not run, and patching them would only add noise
  to `patch.diff`.

### 2. `metadata.kaggle` removed

- **Needed:** the gateway refuses a notebook that carries it (ADR-008, implementation notes).
  Kaggle's editor saves the accelerator and the internet setting there, and kaggle-cli sends the
  file as is, next to the checked `kernel-metadata.json`.
- **Cannot change results:** it holds editor settings, not code. `kernel-metadata.json` sets the
  accelerator, internet, and image explicitly.

### 3. The seed cell

- **Needed:** conflict 5. Among the known sources of nondeterminism, the hash salt is the only one
  that can be removed without changing the reference's code.
- **What it changes, by design:** `hash()` of strings and bytes in the processes the reference
  launches. This fixes the rescoring seeds, and the iteration order of any set of strings in those
  processes, and is the one intended effect. The torch and numpy generators, which the reference
  seeds itself, are untouched.
- **Alternatives:**
  - *Prefix the launch line,* `PYTHONHASHSEED=0 python starter.py`. Lost: it edits an original
    cell.
  - *Replace `hash()` with a stable hash,* the "stable seed" of a June 2026 fork. Lost: it changes
    the reference's code and its seed values. That fork scored 2.5/4, against 3.0/4 twice with the
    original seeds (research, section 1.6).
  - *Deterministic kernels or batch-invariant DFS.* Lost: they change numerics and speed; the
    authors dropped their batch-invariant fix as 17% slower (paper, section 3.3).
  - *No seed.* Lost: it leaves a removable source of run-to-run difference in the noise floor.
- **Value 0:** any fixed value works, and 0 is conventional. Runs A and B, and every later
  comparison against the fixture, must use the same value.

### 4. The dump cell, appended

- **Needed:** the pickles hold `solution`, `beam_score`, and `score_aug` for every candidate of
  each test output (research, section 1.5). When the extracted code disagrees with the fixture,
  these show whether the difference lies in decoding or in ranking. The pickles live outside
  `/kaggle/working`, so Kaggle discards them at the end of the session. `pip_freeze.txt` and
  `env.json` record the stack the run actually used, including the image's build commit and date,
  which the brief's environment rule compares between runs A and B. Only the pinned image and the pinned install
  notebook fix that stack, because the notebook installs flash-attn unpinned and it drifts
  (research, section 1.4). P0.1 must also record the torch, transformers, peft, and flash-attn
  versions (`docs/STATE.md`, next actions).
- **Cannot change results:** it runs after the last original cell, when `submission.json` is
  final and the score is printed. It only reads, and it writes only under `run_logs/`.
- **Never fails the run:** a crash here would mark a good run as failed. So each step catches and
  records its own error, the call sits in `try`/`except`, and the build checks that the cell
  compiles.
- **Embedded with `inspect.getsource`:** the cell holds exactly the `dump_run_logs` function that
  `--smoke` and the tests run, not a second copy kept in a string.
- **20 MB cap per file:** keeps pulls small on the human's connection, below the gateway's 50 MB
  warning (`MAX_PULL_FILE_BYTES`). DEV4 needs far less; `candidates.json` is written as compact
  JSON, about 2 bytes per grid cell.
- **Alternatives:**
  - *Copy the raw pickles into `run_logs/`.* Lost: the pull allowlist (`PULL_PATTERN`) takes only
    `jsonl`, `json`, `md`, `csv`, `txt`, `log`, `yaml`, and `png`, and loading a pickle needs the
    reference's classes and numpy.
  - *Make the reference write into `/kaggle/working`.* Lost: it edits original cells.
  - *An `atexit` hook in the first cell, so the dump also runs after a failed cell.* Lost: it adds
    behavior to the kernel before the reference runs, it still does not run when Kaggle kills the
    session, and a failed run A stops the brief anyway.

### 5. `kernel-metadata.json`, written from scratch

| Field | Value | Why |
|---|---|---|
| `id`, `title` | `<owner>/arc26-p0-1-reference`, `arc26 P0.1 reference` | The owner's slug. The title slugifies to it under python-slugify, which kaggle-cli 2.2.4 uses for its title check. |
| pulled `id`, `id_no`, `title` | never copied | kaggle-cli pushes to `id_no` in preference to `id`, so a copy would push a version of the author's notebook (ADR-008). |
| `code_file`, `language`, `kernel_type` | `reference-run.ipynb`, `python`, `notebook` | The built notebook. |
| `is_private` | `true` | ADR-008: nothing is published. |
| `enable_internet` | `false` | Competition rule for L4 sessions (research, section 2, overview copy); the gateway requires it for accelerated pushes. |
| `machine_shape` | `NvidiaL4` (4xL4) | The reference runs one worker per GPU on four L4s, the hardware of the scored rerun. `enable_gpu` is left out: kagglesdk marks it deprecated in favor of `machine_shape`, and the gateway forwards `--accelerator NvidiaL4`. |
| `docker_image_pinning_type`, `docker_image` | `original`, the pulled image | The reference's Python 3.11 image. The current default image cannot import its cp311 packages (research, P0.1-01 section). |
| `competition_sources` | the config's slug | Attaches the 2026 data and makes L4 available. |
| `model_sources`, `dataset_sources` | copied; an empty `model_sources` is refused | The run needs the reference's attachments, and the reference loads its base model from an attached Kaggle model. |
| `kernel_sources` | copied, each `<owner>/<slug>/<version>` | The install notebook at a fixed version fixes the package stack; an unversioned source is refused. |

### 6. Two runs under one brief

One run cannot separate an extraction bug from run-to-run noise. With the hash seed fixed, batched
DFS and the 1200 s per-puzzle cap still vary. Two DEV4 tasks took 1203 s and 1280 s in a
third-party run (research, section 1.6), right at that cap. Two runs cost about 2 quota hours. A
third would add 1 more hour for a slightly sharper noise floor that is only 5 test outputs wide.

### 7. Not applicable here

- **Resume:** a run takes about 28 minutes (research, section 1.6), under A8's 30-minute
  threshold, and the reference has no resume. A failed run is rerun under an amended brief.
- **A plain GPU machine:** the reference depends on Kaggle mounts, the attached install notebook,
  and four GPUs. Running it elsewhere needs changes that would break parity.

## How to run it

1. **Human:** pull the reference notebook with its metadata into `reference/`. Keep the metadata as
   `reference/kernel-metadata.json` and save the notebook as `reference/reference_notebook.ipynb`
   (gitignored). Set `KAGGLE_OWNER` in `scripts/kaggle_ops.py`.
2. **Orchestrator:** run K0 (task P0.5a, `docs/runbooks/env_probe.md`). Fill the TODO entries in
   `configs/reference_run.yaml` from the pulled notebook and the K0 listing, and fill the TODO
   values in the brief.
3. **Build and check:**

   ```bash
   .venv/bin/python scripts/build_reference_run.py --smoke
   .venv/bin/python scripts/build_reference_run.py --notebook reference/reference_notebook.ipynb \
       --metadata reference/kernel-metadata.json --config configs/reference_run.yaml \
       --owner <username> --out dist/P0.1-01
   .venv/bin/ruff check . && .venv/bin/pytest -m "not gpu"
   ```

   Read `dist/P0.1-01/patch.diff` in full. It holds every changed line of the reference, and its
   header lists the input hashes and the replacement counts.
4. **Commit** the config and `reference/kernel-metadata.json`. The pulled metadata file holds only
   public fields and is a build input; left untracked, it fails the gateway's clean-tree check.
   Write that commit's SHA into the brief's `Code-commit`. Copy the two hashes from the
   `patch.diff` header, the pulled notebook's and the built `reference-run.ipynb`'s, into the
   brief's section 10 lines: `reference/*.ipynb` and `dist/` are gitignored, so `Code-commit` alone
   does not pin what runs. Commit the brief (a docs-only change), then export it with
   `scripts/prerun_sync.py export`, notify, and wait for approval.
5. **Run A**, after approval. First check that the folder is the one the brief names:
   `sha256sum dist/P0.1-01/reference-run.ipynb` must print the brief's built-notebook hash. If it
   does not, do not push; rebuild from the committed inputs, or amend the brief. Then:

   ```bash
   .venv/bin/python scripts/kaggle_ops.py push dist/P0.1-01 --run-id P0.1-01
   .venv/bin/python scripts/kaggle_ops.py status <owner>/arc26-p0-1-reference  # until done
   .venv/bin/python scripts/kaggle_ops.py pull <owner>/arc26-p0-1-reference --task-id P0.1-01-A
   ```

   Pull A before pushing B: kaggle-cli 2.2.4 downloads the latest run's outputs, whichever
   version is named (research, section 3.4). After the pull, check that
   `build.pulled_notebook_sha256` in `run_logs/env.json` equals the brief's pulled-notebook hash.
   Apply the brief's kill criteria to run A before going on.
6. **Run B:** the same hash check, push, and `pull ... --task-id P0.1-01-B`, then the same
   `env.json` check. The folder and the brief stay the same. Compare B's `env.json` and
   `pip_freeze.txt` with A's under the brief's environment rule (section 1) before using B.
7. **Fixture:** copy `runs/P0.1-01-A/submission.json` to `runs/P0.1/reference_output.json`
   (BRIEF A1). P0.1 commits the fixture under `tests/fixtures/`, a protected path.

Expected cost per run, from third-party logs (research, section 1.6, unverified): 1606 to 1694 s
wall time on `NvidiaL4`, and a GPU memory peak of 13.0 GB in training and 17.1 GB in inference,
out of 22 GB per L4. Disk use is a few MB. Locally, the build folder is about the notebook's size.

## Outputs

Build folder (`dist/P0.1-01/`, untracked):

| File | Contents |
|---|---|
| `reference-run.ipynb` | The seed cells, the original cells in order (only the configured path literals changed), and the dump cells; no `metadata.kaggle`. |
| `kernel-metadata.json` | The fields in the table above. |
| `patch.diff` | A `#` header first: hashes of the pulled notebook, pulled metadata, config, and built notebook; the removed settings; the added cells; each replacement with its count. Then a unified diff of every changed cell, labeled with its pulled index and built index. |

Per run, pulled into `runs/P0.1-01-A/` and `runs/P0.1-01-B/`:

| File | Contents |
|---|---|
| `submission.json` | The reference's attempt grids: per task, one `{"attempt_1", "attempt_2"}` per test output. Run A's copy becomes the fixture. |
| `run_logs/candidates.json` | Every file under `/kaggle/inference_outputs/`, keyed by its relative path (the subkey, one per test output), converted from its pickle: each candidate's `solution` grid, `beam_score`, and `score_aug`, with numpy values turned into lists and numbers. |
| `run_logs/pip_freeze.txt` | `pip freeze` of the `python` on `PATH`, the interpreter `!python starter.py` uses. |
| `run_logs/env.json` | Python version and interpreter paths; platform; the image's build commit and date (`image`, read from `/etc/git_commit` and `/etc/build_date`, null when absent); `KAGGLE_KERNEL_RUN_TYPE`, `KAGGLE_IS_COMPETITION_RERUN`, `KAGGLE_DOCKER_IMAGE`, and `PYTHONHASHSEED`; GPU names, driver, and memory from `nvidia-smi`; versions of the key packages; `sys.path`; build provenance (input hashes, hash seed, replacements); and the dump report (files written, files skipped with size and reason, errors). |
| `arc26-p0-1-reference.log` | The session log, always downloaded: the reference's own prints, including per-task timings and the DEV4 score. |

## How to read the results

1. **Did it run as built?** Check `env.json` for Python 3.11 (the image pin held), four NVIDIA L4
   GPUs, `PYTHONHASHSEED` `"0"`, and an empty `dump.errors`. Compare its `build` hashes with the
   brief after each run: `build.pulled_notebook_sha256` must equal the brief's pulled-notebook
   hash, and `build.pulled_metadata_sha256` and `build.config_sha256` the `patch.diff` header. The
   built notebook's own hash is checked before each push (How to run it, step 5), since the
   notebook cannot record it. Compare `packages` with the third-party stack: torch 2.8.0+cu128,
   transformers 4.55.4, unsloth 2025.9.7, triton 3.4.0, xformers 0.0.32.post2 (research, section
   1.4). For run B, compare `image`, `python`, and the package versions with run A's under the
   brief's environment rule.
2. **Score.** The session log prints the DEV4 score: the per-task mean with fractional credit.
   With $c_{t,i} = 1$ when either attempt equals test output $i$ of task $t$ exactly,
   $$S = \sum_{t \in \mathrm{DEV4}} \frac{1}{n_t} \sum_{i=1}^{n_t} c_{t,i}
   \quad \text{out of 4 tasks (5 test outputs)}.$$
3. **Agreement.** For runs $X$ and $A$, over DEV4's 5 test outputs $j$,
   $$g(X, A) = \frac{1}{5} \sum_{j=1}^{5}
   \mathbb{1}\!\left[\left(x^{(1)}_j, x^{(2)}_j\right) = \left(a^{(1)}_j, a^{(2)}_j\right)\right],$$
   so both attempt grids must match exactly and in order. $g(B, A)$ is the noise floor.
4. **Decision.** Apply the pre-registered table in `docs/prerun/P0.1-01.md`, section 1 (Good,
   Borderline, Bad, B fails, B's environment differs), with its rules for run B's environment
   and for the later extracted-code run. The brief is the only authority; this runbook does not
   restate it, so the two cannot diverge.
5. **When A and B disagree,** look up the test output in both runs' `candidates.json`. Different
   candidate sets point to decoding (batched DFS, the time cap). The same candidates in a
   different order point to scores that differ only in their last bits.

## Failure modes

| Failure | Sign | First thing to do |
|---|---|---|
| Config still has TODO text | `refused: ... still has TODO on line(s) ... (comments count)` | Fill each entry from the pulled notebook and the K0 listing, put the cell, line, and listing line in its comment, and delete the instruction paragraph. |
| A count does not match | `path replacements refused: '<old>' matches N times ...` | The pulled notebook differs from the one the entry was written for. Find every occurrence in the pulled file, then correct the entry and its comment. Never loosen a count to get a build through. |
| No image, no model or kernel source, or an unversioned kernel source | `refused: the pulled kernel-metadata.json has no ...` or `kernel sources without a version ...` | Take all of them from the reference's pulled metadata. If the pull really lacks one, such as the version, stop and amend the task brief (for example, a pinned version in the config with its source); never guess one. |
| Image not pinnable | The run errors within minutes on an import of the cp311 packages, or the session log shows another Python version | Kaggle rejected or replaced the old image. Check the Python version in the K0 pinned variant. This script has no workaround: the P0.5 stack decision (research, P0.5 section) comes first, then an amended brief. |
| A path not found | `FileNotFoundError` for a `/kaggle/input/...` path in the session log | A mount differs from the K0 listing, or a literal is missing from the config. Add or correct the entry, rebuild, and amend the brief: the config changed, so it needs re-approval. |
| Run over the timeout | Kaggle ends the session at 3600 s; outputs may be missing, and the dump cell never runs | The expected wall time is about 1700 s. Before any rerun, check the session log for fewer than four GPUs or a slower stack. A rerun needs an amended brief that gives the reason for any new `Timeout`. |
| Out of memory | CUDA out-of-memory error in the session log | Third-party peaks were 17.1 of 22 GB per L4, so an OOM means other hardware or another stack. Compare the log and `env.json` with the third-party run; never change the reference. |
| Divergence or NaN | NaN losses in the session log, or a score far below 2/4 | The reference's training is unchanged, so suspect the stack, for example another flash-attn build. The score kill criterion stops the brief. |
| Dump errors | Entries under `dump.errors` in `env.json`, or no `candidates.json` | `submission.json`, and with it the fixture, is unaffected. Fix `dump_run_logs`, and check the fix with `--smoke`, before the next brief. |
| B's pull replaces A's | A's files are missing locally | Pull A before pushing B, into separate `--task-id` folders. |
