# Runbook: `scripts/env_probe.py` (K0 environment probe)

Task P0.5a (K0). Builder: `scripts/build_env_probe.py`. Tests: `tests/test_env_probe.py`. CPU
only: it spends no GPU quota and needs no pre-run brief (BRIEF A9 covers GPU runs).

## 1. Why this script exists

Before the first GPU run, several facts must be measured on Kaggle instead of inferred from 2025
notebooks and third-party logs (`docs/research/kaggle_first_runs.md`, sections 1.3, 2, 3.1, 3.6).
The probe answers them in one CPU session per image:

- Mount paths of the competition data, the base model, and attached notebooks: the A5 checklist,
  the path literals of the reference capture run (P0.1-01), and the path search of the P0.5
  notebook. The 2026 layout differs from 2025 and moved within 2026.
- Image, Python, and package versions, on the latest image and on NVARC's pinned image: the P0.5
  stack decision (flagged conflict 4 in `docs/STATE.md`) and the P3.1 pins.
- The `KAGGLE_*` variables: mode detection in the notebook's config cell (A5).
- CPU, RAM, and disk of a session.
- Per-task size statistics of the evaluation challenges: the DEV40 selection (P0.2).

K0 is also the first end-to-end use of the gateway (`scripts/kaggle_ops.py` push, status, pull).

## 2. What it does, step by step

1. **Session facts.** `sys.version` and platform; `/etc/os-release`; the image's build commit and date from `/etc/git_commit` and
   `/etc/build_date` (null when absent; the P0.1-01 brief matches runs A and B on them); `os.cpu_count()` and the
   usable CPUs (`os.sched_getaffinity`); `MemTotal`, `MemAvailable`, `SwapTotal` from
   `/proc/meminfo`; `shutil.disk_usage` of `/kaggle/working` and `/tmp`.
2. **`KAGGLE_*` variables.** Every name. Values only for `KAGGLE_KERNEL_RUN_TYPE`,
   `KAGGLE_IS_COMPETITION_RERUN`, `KAGGLE_DOCKER_IMAGE`, `KAGGLE_URL_BASE`, and never for a name
   containing `TOKEN`, `SECRET`, `KEY`, `PASS`, `AUTH`, or `CREDENTIAL`.
3. **`sys.path`**, which shows whether an attached notebook adds packages.
4. **Listings** of `/kaggle/input` and `/kaggle/usr/lib`, breadth first to depth 4 (a child of
   the listed folder has depth 1), at most 2000 entries each: type, size of each file, target of
   each symbolic link (not descended). `total` counts every entry within the depth and
   `truncated` says whether the cap dropped any.
5. **Versions** from `importlib.metadata` for torch, transformers, peft, accelerate, trl,
   datasets, unsloth, unsloth_zoo, xformers, flash_attn, flash-attn, triton, bitsandbytes, vllm,
   numpy, without importing any of them; then `pip list --format=json` in a fresh interpreter,
   with the version check off and every secret-named variable removed from its environment.
6. **Resolved paths.** A walk of all of `/kaggle/input` collects every `arc-agi_*.json` and every
   `config.json` whose parent path contains `qwen3_4b_grids15_sft139`, with sizes from `stat`.
   Nothing is opened. The real tree is walked first, then symbolic links to folders, with one set
   of visited real paths seeded with the folders above `/kaggle/input`: a link to `/`, to a folder
   above, to a folder already walked, or into a loop is recorded in `resolved.skipped_links` and
   not entered, so no folder is walked twice and no link reaches the rest of the filesystem.
7. **Evaluation statistics.** Among the resolved files named `arc-agi_evaluation_challenges.json`,
   the one in the competition's folder wins, then the shortest path. For task $t$ with train
   pairs $(x_i, y_i)_{i=1}^{n_t}$, test inputs $(u_j)_{j=1}^{m_t}$, and
   $G_t = \{x_i\} \cup \{y_i\} \cup \{u_j\}$ (a multiset of grids, each of height $h$ and width
   $w$), it writes $n_t$, $m_t$,
   $$\text{max\_side}_t = \max_{g \in G_t} \max\bigl(h(g), w(g)\bigr), \qquad
   \text{cells}_t = \sum_{g \in G_t} h(g)\,w(g),$$
   plus the file's size and SHA-256. Test outputs, where a file has them, are not counted. A
   malformed task is flagged with its exception type only.
8. **Scrub and write.** The value of every secret-named environment variable, `KAGGLE_*` or not
   (8 characters or more), is replaced by `<redacted>` anywhere in the report, the Markdown, and
   the printed lines; then `env_probe.json` and `env_probe.md` are written to
   `/kaggle/working/run_logs/`.

## 3. Why it is built this way

- **CPU script kernel with the GPU runs' attachments.** It sees the mounts a GPU run will see
  (competition, model, install notebook, internet off) at no quota cost. A notebook kernel was
  the alternative; a script carries no notebook JSON, whose Kaggle-saved settings the gateway
  would have to police, and runs unchanged in the tests.
- **Standard library only, no package imports.** It must run on any image, including a latest
  image whose packages do not match, and importing a GPU stack on a CPU session can fail or have
  side effects. Metadata gives the versions without executing package code.
- **Two variants, two slugs.** `latest` shows what an unpinned notebook gets today; `pinned`
  shows whether NVARC's image (Python 3.11) is still served. Separate slugs keep their histories
  apart, and neither may appear as the `Kaggle notebook:` of a pre-run brief, since Kaggle may
  keep a slug's previous accelerator (ADR-008, the gateway's CPU slug rule).
- **Secrets.** Sessions carry tokens such as `KAGGLE_USER_SECRETS_TOKEN`, and the report is
  pulled to the dev box. Values come from an allowlist of four names; a secret-looking name is
  withheld even if someone adds it there; the final scrub covers every secret-named variable,
  not only `KAGGLE_*`, because pip's error output and `sys.path` can carry any of them, and pip
  runs without them. Tests check each layer.
- **One visited set for the walk.** Links are followed because a mount may be reachable only
  through one, but a per-route cycle check would still let a link to `/` walk the whole
  filesystem. One set of visited real paths, seeded with the folders above `/kaggle/input`, bounds
  the walk by the folders it may legitimately reach. The depth listing never descends links, so
  it cannot leave its tree.
- **Solutions never opened.** BRIEF A0 limits the evaluation solutions to dev-mode scoring. Every
  read goes through one function that refuses a file whose name, or whose link target's name,
  contains `solution`; resolved paths and sizes come from `stat`.
- **Counts, not grids.** Competition data must never reach the repository, and summaries of runs
  are committed under `docs/experiments/`.
- **Depth 4, cap 2000, breadth first.** Depth 4 reaches `/kaggle/input/competitions/<slug>/<file>`
  and `/kaggle/input/models/<owner>/<model>/<framework>`; breadth-first order keeps the shallow
  levels, which show the layout, when an installed-package tree hits the cap. Deeper files come
  from the unbounded search in step 6.
- **Isolated sections.** One failure (a missing mount, an unreadable file, pip missing) is
  recorded under `errors` and the rest of the report stays valid.
- **Kaggle execution details.** `--root` (default `/`) prefixes every absolute path, so the tests
  run on a synthetic tree. Unknown arguments are ignored and success raises no `SystemExit`,
  because a script kernel may run inside an IPython kernel.

## 4. How to run it

Prerequisites:

- `KAGGLE_OWNER` in `scripts/kaggle_ops.py` is the human's Kaggle username (while it is empty the
  gateway refuses every push), and `--owner` below is the same name.
- The Kaggle account can attach the competition `arc-prize-2026-arc-agi-2` (joined, rules
  accepted), the model, and the install notebook.
- For `pinned`: `docker_image` from `reference/kernel-metadata.json`, the human's pull of the
  reference notebook with its metadata, and the install notebook from the same file as
  `<owner>/<slug>/<version>`. The builder refuses a handle without a version.

Build both push folders (`dist/` is gitignored). `--kernel-source` is optional; without it,
`kernel_sources` is `[]`.

```bash
.venv/bin/python scripts/build_env_probe.py --owner <username> --variant latest \
    --out dist/K0/latest --kernel-source <owner>/<slug>/<version>
.venv/bin/python scripts/build_env_probe.py --owner <username> --variant pinned \
    --out dist/K0/pinned --docker-image <docker_image> --kernel-source <owner>/<slug>/<version>
```

Push, poll, and pull one variant at a time. Both variants write the same file names, so move each
pull aside before the next one:

```bash
python scripts/kaggle_ops.py push dist/K0/latest --timeout 1800
python scripts/kaggle_ops.py status <username>/arc26-k0-env-probe-latest   # until COMPLETE or ERROR
python scripts/kaggle_ops.py pull <username>/arc26-k0-env-probe-latest --task-id K0
mkdir -p runs/K0/latest && mv runs/K0/run_logs runs/K0/latest/
```

Then the same with `pinned` in place of `latest`.

Local check, no Kaggle involved:

```bash
.venv/bin/pytest tests/test_env_probe.py
.venv/bin/python scripts/env_probe.py --root <scratch folder>   # a synthetic or empty tree
```

Expected cost: no GPU and no GPU memory. The probe reads metadata, walks `/kaggle/input`, and
reads one challenges file, so it should finish within minutes of the session starting;
`--timeout 1800` is the hard stop. Outputs stay under 1 MB (the listings are capped).

## 5. Outputs

| File | Contents |
|---|---|
| `runs/K0/<variant>/run_logs/env_probe.json` | the full report, keys below |
| `runs/K0/<variant>/run_logs/env_probe.md` | the same facts for reading; `pip list` as a count |
| `runs/K0/arc26-k0-env-probe-<variant>.log` | the session log, which every pull brings |

| Key | Contents |
|---|---|
| `schema`, `written_utc`, `root` | report format 1, UTC time of the run, `/` on Kaggle |
| `python` | `sys.version`, version triple, implementation, executable, `platform.platform()`, machine |
| `os_release` | the `KEY=value` pairs of `/etc/os-release` |
| `cpu` | `cpu_count` and `affinity` (usable CPUs) |
| `memory` | `MemTotal`, `MemAvailable`, `SwapTotal` in bytes |
| `disk` | `total`, `used`, `free` bytes of `/kaggle/working` and `/tmp` |
| `kaggle_env` | `names`: every `KAGGLE_*` name; `values`: the four shown values, null when unset |
| `sys_path` | the interpreter's `sys.path` |
| `listings` | per folder: `exists`, `depth`, `cap`, `total`, `truncated`, `entries`, `errors` |
| `versions` | version per listed distribution, null when not installed |
| `pip_list` | the output of `pip list --format=json` |
| `resolved` | `arc_agi_files` and `model_configs` as path and size; `skipped_links` as path and real target; `errors` of the walk |
| `evaluation_stats` | `source`, `bytes`, `sha256`, `summary`, `per_task` |
| `errors` | failed section to its error; empty when every section ran |

## 6. How to read the results

| Field | Check | Feeds |
|---|---|---|
| `resolved.arc_agi_files` | folder of the evaluation challenges, solutions, and test challenges; expected `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` | competition data mount path (A5 table in `docs/STATE.md`); P0.1-01 path literals; P0.5 path search |
| `resolved.model_configs` | folder of `config.json`; expected `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1` | base model mount path; same consumers |
| `listings`, `sys_path` | where the install notebook mounts (`/kaggle/input/notebooks/...` or `/kaggle/usr/lib/...`) and whether it is on `sys.path` | P0.5 stack decision (conflict 4) |
| `python`, `os_release`, `kaggle_env.values.KAGGLE_DOCKER_IMAGE` | Python and image per variant: 3.11 expected for `pinned`, 3.12 or newer for `latest` | P0.5 stack decision; whether P0.1-01 can pin NVARC's image |
| `versions`, `pip_list` | torch, transformers, peft, unsloth, flash-attn, triton, xformers per variant; whether the install notebook changes them (`versions` reads the probe's own `sys.path`, `pip_list` a fresh interpreter's) | P3.1 pins; lower bounds in `pyproject.toml` (conflict 4) |
| `kaggle_env` | `KAGGLE_KERNEL_RUN_TYPE`; `KAGGLE_IS_COMPETITION_RERUN` unset in a commit run; the full list of names | mode detection in the notebook's config cell (A5) |
| `cpu`, `memory`, `disk` | size of a CPU session; free space in `/kaggle/working` | CPU jobs on Kaggle; output size budgets |
| `evaluation_stats` | 120 well-formed tasks; DEV4 (`0934a4d8`, `36a08778`, `981571dc`, `aa4ec2a5`) present with 5 test inputs in total | DEV40 selection (P0.2) |

- **Good:** `errors` is empty for both variants; data and model resolve at the expected paths; the
  statistics cover 120 well-formed tasks; `pinned` runs Python 3.11 and `KAGGLE_DOCKER_IMAGE`
  names the pinned image. Record the facts in `docs/experiments/K0.md` and the A5 table of
  `docs/STATE.md`.
- **Borderline:** paths resolve elsewhere or only through links (the resolved paths win over the
  research note; update the P0.1-01 config and the P0.5 search roots); versions differ from the
  third-party logs (pin from K0); a listing is `truncated` (fine, deep paths come from
  `resolved`); `pip_list` failed but `versions` is present.
- **Bad:** no `arc-agi_*.json` or no model `config.json` resolved: an attachment is missing, so
  fix it and rerun before any pre-run brief cites a path. `pinned` fails to start or reports
  another Python or image: NVARC's image is no longer served, which P0.5 must resolve (conflict
  4) before the reference run is briefed.

## 7. Failure modes

- **The gateway refuses the push** (`refused: ...`, exit 2): `KAGGLE_OWNER` is empty or differs
  from `--owner`; a pre-run brief names the probe's slug (remove it from the brief; never reuse
  the slug for a GPU run); `--timeout` is outside 60 to 43200. Fix the input and rebuild.
- **Kaggle rejects the push:** the CLI prints the reason. Usually an unknown `docker_image` (copy
  it verbatim from the pulled metadata) or a source the account cannot attach (competition not
  joined, model or install notebook handle or version mistyped).
- **A missing mount:** `listings["/kaggle/input"].exists` is false, a `resolved` list is empty,
  or `errors.evaluation_stats` reads `no arc-agi_evaluation_challenges.json under /kaggle/input`.
  Compare `kernel-metadata.json` in the push folder with the listing (links show their targets),
  `resolved.skipped_links` (a mount behind a link to a folder above `/kaggle/input` is not
  entered), and the session log, fix the attachment, and push again.
- **An unreadable file or folder:** `errors.<section>` holds the exception, such as
  `PermissionError` for the challenges file or `JSONDecodeError` if its format changed;
  unreadable folders appear in `listings[...].errors` and `resolved.errors`. The other sections
  stay valid. A changed challenges format blocks P0.2 until it is understood.
- **`pip list` fails or runs past 300 s:** `errors.pip_list`; `versions` still holds the listed
  distributions.
- **The session ends in ERROR or hits the timeout:** read the session log. The probe writes its
  files at the end and every section is guarded, so the likely cause is a failed write to
  `/kaggle/working/run_logs/`.
- **The pull returns 2 or removes files:** the pull keeps only allowlisted files under
  `run_logs/`, `submission.json`, and the session log. The probe writes only `env_probe.json` and
  `env_probe.md`, so a removed file means the session wrote something unexpected.
