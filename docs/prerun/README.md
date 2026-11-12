# Pre-run briefs

One brief per accelerated (GPU or TPU) run on Kaggle, `<run-id>.md`, copied from
[TEMPLATE.md](TEMPLATE.md) (BRIEF A9, ADR-008). A run starts only after the human writes an
`Approved-by:` line in the brief's Approval section, or approves it in chat and the orchestrator
records that verbatim with the date.

`scripts/kaggle_ops.py push` enforces this for every accelerated push: `enable_gpu` or
`enable_tpu` true, a non-empty `machine_shape`, or `--accelerator`. The push needs
`--run-id <run-id>`. The gateway and `scripts/prerun_sync.py` read a brief the same way: header
fields come only from its first fenced block, a repeated `Run-id`, `Revision`, `Code-commit`,
`Kaggle notebook`, `Accelerator`, or `Timeout` there refuses the brief, and the file holds exactly
one `Approved-by:` line, inside the Approval block (the fenced block holding `Approved-by:` and
`Notes:`). The gateway then checks:

| Field | Gateway check |
|---|---|
| `Approved-by:` | filled in; a `revision N` stamp, as `prerun_sync.py record` writes it, must equal `Revision` |
| `Run-id:` | equals the brief's file name |
| `Code-commit:` | the working tree is clean and nothing outside `docs/` changed since this commit |
| `Kaggle notebook:` | equals the `id` in `kernel-metadata.json`; no CPU push may use it, since Kaggle may keep a slug's previous accelerator |
| `Accelerator:` | exactly one ID from `ACCELERATORS`, equal to the push's `--accelerator` or `machine_shape`; always forwarded as `--accelerator` |
| `Timeout:` | an integer from 60 to 43200 seconds, forwarded as `-t`; the command line cannot change it |

Accelerator IDs: `NvidiaL4` (4xL4), `NvidiaL4X1` (1xL4), `NvidiaTeslaT4` (2xT4),
`NvidiaTeslaA100`, `NvidiaH100`, `NvidiaRtxPro6000`, `TpuV5E8`, `TpuV6E8`. Retired IDs
(`NvidiaTeslaP100`, `Gpu`, which Kaggle silently runs on 2xT4) and names such as `4xL4` are
refused. `Internet:` is informational: the gateway checks `"enable_internet": false` in
`kernel-metadata.json` instead. A notebook whose top-level `metadata` has a `kaggle` object is
refused on every push, because Kaggle-saved notebooks keep `accelerator` and `isInternetEnabled`
there and the push sends the file as is. The other metadata checks (owner, `id_no`,
`enable_gpu` and `enable_tpu`, image pinning) are in ADR-008's implementation notes. A code or
config change after approval needs an updated brief and a new approval.

**Modal briefs** (`Provider: modal`, ADR-022) are launched only by `scripts/modal_ops.py run
<dist> --run-id <run-id>`, after the same approval checks (`kaggle_ops.approved_header`, clean
tree, `Code-commit`); `kaggle_ops.py push` refuses any `Provider` but `kaggle`. The Modal gateway
also checks:

| Field | Gateway check |
|---|---|
| `Provider:` | exactly `modal` |
| `GPU:`, `GPU count:` | a type of `modal_job.GPU_RATES_USD_S` (such as `H100!`) and 1 to 8 GPUs, equal to `job.json` |
| `Timeout:` | 60 to 86400 seconds, equal to `job.json`'s GPU-stage timeout; also the ledger's total over attempts (an evaluation job: per step) |
| `Max attempts:` | 1 to 3, equal to `job.json` (an evaluation job: per step) |
| `Cost cap USD:` | a positive number at least the job's worst case at every hard stop (`modal_job.worst_case_usd`) |
| `Output dataset:` | `<KAGGLE_OWNER>/<slug>`, equal to `job.json`; the only dataset the job may write. An evaluation job (task X2) publishes nothing: `none`, and its `job.json` has none |

The dist folder (`scripts/build_modal_job.py`) is checked against `Code-commit`: the folder holds exactly the files `job.json` lists, each with its sha256; `job.json` is
regenerated from `configs/modal_<run-id>.yaml`, the training script, and its config as
`git show <Code-commit>` gives them (`build_modal_job.job_from_commit`, the builder's own code)
and must equal the folder's byte for byte, so every argument, input, volume, resource, and
timeout is the approved commit's; and the wheel holds exactly the commit's `arc26/` files, byte
for byte, plus `METADATA`, `WHEEL`, and `RECORD` of `arc26-<version>.dist-info` and nothing else,
with `METADATA` naming the commit's project and version, `WHEEL` declaring a pure-Python wheel,
and `RECORD` listing every other member with its size and sha256. This closes the "dist is untracked" limit below for Modal runs. `Cost expected USD` is informational,
shown by `prerun_sync.py export` and in the email.

Known limits:

- An edit to a brief's body after approval is caught only through `Revision`; there is no content
  hash.
- Build folders under `dist/` are untracked by design, so the clean-tree and `Code-commit` checks
  do not cover the pushed folder itself.
- The approval page shows `Timeout` only inside the brief text.
- The CPU slug guard reads the briefs in this checkout's `docs/prerun/` only; a brief that exists
  only on another task branch does not protect its notebook's slug. Each notebook slug therefore
  belongs to one task branch.

Decisions are made on the owner-only approval page, https://claude.ai/artifact/Vo68JfLjqsLvKnkHxBpqer, which shows every brief in full and
works from a phone. `scripts/prerun_sync.py` exports a brief to the page (`export`), emails the
summary and link (`notify`), and copies the decision back into the brief (`record`), refusing a
decision made on another revision or code commit.

After the run, `docs/experiments/<task-id>.md` links the brief and states which branch of its
decision rule the result fell in.

Page access: the database rules are `read: "admin", write: "admin"` (the share menu's Owner and
Editors); the page is private and has no editors, so only the owner can read or decide. On 30 Sep
the owner's view showed an empty queue while the briefs were stored, and the rules were raised from
`owner` to `admin`, which did not help. The cause (found 1 Oct, fixed in page version 3) was the
page's snapshot handler: it assigned defaults onto the snapshot's `data()` object, which the runtime
delivers frozen, so in strict mode the first brief threw and the queue never rendered. `P0.1-01`,
`P0.5-01`, and `P1.1-01` were approved in chat under BRIEF A9 meanwhile; their page entries record
that in a decision note and show them as finished.
