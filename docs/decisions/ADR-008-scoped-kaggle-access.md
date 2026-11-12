# ADR-008: Scoped Kaggle access through one gateway, gated by approved pre-run briefs

- Status: Accepted, 2026-09-29. Supersedes the "no connection to Kaggle" rule of BRIEF v3 (A0).
  Amended 2026-10-17 by ADR-022 (Kaggle reads and writes from inside Modal).
- Tasks: all GPU tasks (A8), P0.1, P0.5, P4.2, every milestone tag.

## Context

BRIEF v3 forbade any agent connection to Kaggle, so the human uploaded every build, launched
every run, and downloaded every output by hand. The human does not want to train on the local
machine, has little disk space and a slow connection, and asked for the agent to be linked to
Kaggle. The human also asked for a brief before each training run, detailed enough to judge the
data, objective, architecture, and technical settings before any quota is spent.

## Decision

1. The agent reaches Kaggle only through `scripts/kaggle_ops.py`, which wraps the official
   `kaggle` CLI with five actions: push a private notebook version, poll its status, pull
   allowlisted small outputs, create a private dataset, and version a dataset.
2. The gateway has no path to submit to the competition, make anything public, or delete
   anything. Those stay with the human on the website.
3. Every GPU push requires an approved pre-run brief `docs/prerun/<run-id>.md` (BRIEF A9). The
   gateway checks the `Approved-by:` line, a clean working tree, and that no file outside `docs/`
   changed since the brief's `Code-commit:`. The agent never approves its own brief.
4. No GPU work runs on the local machine. Large artifacts (weights, checkpoints, episode sets,
   embedding caches) are built and kept on Kaggle and chained between notebooks as inputs. Pulls
   are limited to log, metric, report, and small plot file types; uploads are capped at 200 MB.
5. Briefs are reviewed on an owner-only approval page (https://claude.ai/artifact/Vo68JfLjqsLvKnkHxBpqer) and announced by email
   (`scripts/prerun_sync.py`), so the human can decide from any device. A page decision reaches
   the gateway only through `prerun_sync.py record`, which binds it to the brief's `Revision` and
   `Code-commit`.
6. The token stays in `~/.kaggle/` and is used only by the CLI. The agent never reads, prints, or
   commits it; `.gitignore` covers `.kaggle/` and `access_token`.

## Alternatives considered

- **Keep v3 (manual everything).** Lost: every run costs the human uploads, downloads, and disk
  space they do not have, and the 2 to 3 hours per week go to file shuffling instead of judgment.
- **Full CLI access without a gateway.** Lost: nothing would stop a stray `competitions submit`,
  a public dataset, or a multi-GB download; rules held only in prose are easy to break.
- **Rented GPUs instead of Kaggle.** Not chosen now: costs money and still needs a transfer path.
  The P0.4 decision can still select it for Phase 3.

## Consequences

- The human's weekly work becomes: approve briefs, decide at gates, click Submit.
- GPU quota can now be spent without a human click, so the approval gate and the kill criteria in
  each brief are the only brakes. The weekly brief reports quota used and left.
- `tests/test_repo_hygiene.py` still bans Kaggle access from `arc26/`, `scripts/`, and
  `notebooks/`, with `scripts/kaggle_ops.py` as the single exemption. `tests/test_kaggle_ops.py`
  checks the gateway's refusals.

### Implementation notes (29 and 30 Sep)

Before the first push, the gateway was checked against the installed kaggle-cli 2.2.4
(`kaggle/api/kaggle_api_extended.py`) and hardened. The decision is unchanged.
`scripts/kaggle_ops.py` enforces:

- **Accelerated pushes.** A push is accelerated when `enable_gpu` or `enable_tpu` is true (a bool,
  or `"true"` in any case), `machine_shape` is non-empty, or `--accelerator` is given, because the
  CLI sends `machine_shape` whenever `--accelerator` is absent. Each needs `--run-id` and an
  approved brief (decision 3).
- **Brief reading.** The gateway and `prerun_sync.py` read header fields only from the brief's
  first fenced block and refuse a repeated `Run-id`, `Revision`, `Code-commit`, `Kaggle notebook`,
  `Accelerator`, or `Timeout`, so the approval page and the gateway see the same values. The file
  holds exactly one `Approved-by:` line, inside the Approval block (the fenced block holding
  `Approved-by:` and `Notes:`). A `revision N` stamp on that line, as `prerun_sync.py record`
  writes it, must equal the header's `Revision`, and `Run-id` must equal the file name.
- **Notebook.** The metadata `id` must equal the brief's `Kaggle notebook:`, and a CPU push may
  not use the `Kaggle notebook:` of any brief, because Kaggle may keep a slug's previous
  accelerator. The code file must exist inside the pushed folder. A notebook whose top-level
  `metadata` has a `kaggle` object is refused on every push: Kaggle-saved notebooks keep
  `accelerator` and `isInternetEnabled` there, and the CLI sends the file with its metadata
  intact (lines 6416 to 6426).
- **Accelerator IDs.** `--accelerator` and `machine_shape` must be in `ACCELERATORS`. Retired IDs
  (`NvidiaTeslaP100`, `Gpu`, which Kaggle silently runs on 2xT4), names such as `4xL4`, and two
  different IDs are refused. The ID must equal the brief's `Accelerator:` and is always forwarded
  as `--accelerator`. `enable_gpu` and `enable_tpu`, when present, must be booleans that agree
  with the ID: a GPU ID (`Nvidia*`) with `enable_gpu` false or `enable_tpu` true is refused, and
  a TPU ID (`Tpu*`) the other way round.
- **Internet.** `enable_internet` must be an explicit boolean, because the CLI defaults it to
  true; accelerated pushes require false.
- **Timeout.** The brief's `Timeout:` (an integer, 60 to 43200 seconds) is forwarded as `-t`. It
  is the only programmatic brake, since the CLI has no cancel command: kill criteria are enforced
  inside the script, and `Timeout` is the outer bound. A CPU push may pass `--timeout` in the same
  range.
- **Image.** `docker_image_pinning_type` must be `original` or `latest`, so no push changes the
  image implicitly.
- **Identity.** Metadata with `id_no` is refused: the CLI prefers it over `id` for notebooks and
  datasets, and `kernels pull -m` writes it. `id` in `kernel-metadata.json` and
  `dataset-metadata.json` must be `<KAGGLE_OWNER>/<slug>`; while `KAGGLE_OWNER` is empty, every
  push and dataset action is refused.
- **Pulls.** The CLI applies an unanchored `re.search` to paths relative to `/kaggle/working` and
  does not sanitize server file names. `PULL_PATTERN` is anchored: `submission.json` at the root,
  and log, metric, report, and plot files under `run_logs/`, every path segment starting with a
  letter, digit, `_`, or `-`, so no `.` or `..` segment. After the download, any file it added
  that `pull_allowed` rejects, other than the `<slug>.log` session log, is deleted and reported,
  with the directories that leaves empty, and the pull fails. Files from earlier pulls are left
  alone.
- **Paths.** Run and task ids are limited to letters, digits, `.`, `_`, and `-`, so the brief
  lookup stays in `docs/prerun/` and the pull folder in `runs/`.

Known limits: an edit to a brief's body after approval is caught only through `Revision` (there
is no content hash); build folders under `dist/` are untracked by design, so the clean-tree and
`Code-commit` checks do not cover the pushed folder itself; the approval page shows `Timeout`
only inside the brief text; the CPU slug guard reads only this checkout's briefs, so each
notebook slug belongs to one task branch.

### Amendment (17 Oct, ADR-022): Kaggle reads and writes made from inside Modal

Training jobs may run on Modal (ADR-022). Their Kaggle access does not go through
`scripts/kaggle_ops.py`, because it happens inside Modal's containers, with the human's
credentials, and never on the local machine:

- **Credentials.** The human's Kaggle username and key are the Modal secret `kaggle`, created by
  the human. Only a job's CPU `fetch` and `publish` containers receive it; the GPU container, the
  driver, and the local gateway never do. No code reads, prints, or logs it; the attempt's
  `env.json` records only an allowlist of Modal placement variables.
- **Reads.** `fetch` downloads exactly the inputs the approved job lists, each pinned to a
  version (`owner/slug/versions/N`, `owner/model/framework/variation/N`; an unpinned handle is
  refused), with kagglehub 0.3.12, into the Modal volume. Nothing is downloaded to the local
  machine, so decision 4's "never download weights" holds.
- **Writes.** `publish` writes one dataset only, the brief's `Output dataset`
  (`<KAGGLE_OWNER>/<slug>`, checked by the gateway and again in the job), created private by
  kagglehub (`isPrivate` true) or versioned if it exists, capped at 20 GiB. Decision 2 holds:
  nothing is submitted, made public, or deleted.
- **Gate.** All of it runs only after `scripts/modal_ops.py run` accepted an approved brief with
  `Provider: modal` (decision 3's checks plus ADR-022's); `kaggle_ops.py push` refuses such a
  brief.
- **Hygiene.** `tests/test_repo_hygiene.py` allows Kaggle client imports (`kaggle`, `kagglehub`)
  in `scripts/kaggle_ops.py` and `scripts/modal_job.py` only, and Modal client use in
  `scripts/modal_ops.py` and `scripts/modal_app.py` only.

## Overturn criterion

Any unapproved GPU push, public artifact, or unwanted download is treated as an incident: access
reverts to v3 (manual) until the gateway is fixed and the human re-approves ADR-008.
