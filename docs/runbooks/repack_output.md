# Runbook: `scripts/repack_output.py`

CPU script, run in a Kaggle script notebook (`configs/X2-repack-script.yaml`), never on the dev
box but for its `--smoke`. Task X2, after `X2-01` attempt 3.

## 1. Why this script exists

The Modal fetch stage downloads Kaggle inputs with kagglehub. A notebook output downloaded
whole comes as one zip that Kaggle builds while it streams it (chunked transfer, no
`Content-Length`, no `Range`). For `sorokin/pip-install-unsloth-flash-patch` version 2 (the
install notebook every build attaches, about 8 GB, **estimate**) Kaggle ended that stream with
`ChunkedEncodingError: Response ended prematurely` on every try: 185 s in attempt 2, then 193,
193, 193, and 190 s in attempt 3's four retries (`modal app logs --timestamps`). The smaller
inputs fetched fine (`X1-01`: a 2.18 GB output archive in 114 s, about 19 MB/s; a 1.62 GB
single file in 19.7 s, about 82 MB/s). So a large output must not come as one archive. The
weights' and wheels' outputs (this repository's download notebooks) carry manifests and are
fetched file by file. The install notebook's output has no manifest and thousands of files; this
script turns it, inside a Kaggle session that mounts it exactly as every run does, into a few
tar parts and a manifest, which the fetch downloads part by part and checks.

## 2. What it does, step by step

1. Reads its config (`configs/repack_kaggle_lib.yaml`, flat `key: value` lines: the image's
   Python may lack PyYAML).
2. Walks the mounted folder (`source`) without following links and records every entry
   (`tree_record`): each file's size, sha256, and owner execute bit; each link's target; each
   empty folder.
3. Writes the entries, sorted, as one uncompressed PAX tar into parts of at most `part_bytes`
   (`parts/<name>.tar.000`, `.001`, ...), owners cleared, modes and links kept.
4. Writes `MANIFEST.json`: `kind` `repack`, the source, each part's path, size, and sha256 in
   order, the tree, and the totals.
5. On Modal, `modal_job.fetch_listed` downloads `repack/MANIFEST.json`, then each part as one
   file (kagglehub's single-file download, retried, checked against its size and sha256), joins
   and extracts them with `tarfile`'s `data` filter, and refuses the folder unless its tree
   equals the manifest's (`unpack_tar`); then the `kaggle_lib` check runs as before (the 13
   package versions, the unsloth patch, the execute bits).

## 3. Why it is built this way

| Choice | Alternative considered | Why it lost |
|---|---|---|
| Repack in a Kaggle session that attaches version 2 | Download the install output file by file through Kaggle's list API | The list API (`kernels/output/list`) names the latest session, not a version, so the files could be another version's; and thousands of requests |
| Tar parts of 1 GiB | One tar file | One file of about 8 GB may hit the same time limit; a part takes about 56 s at the archive's measured rate and 13 s at a single file's |
| Uncompressed | gzip | The bytes are shared objects and wheels' contents; compression barely shrinks them at a large CPU cost on both sides |
| A tree of every entry in the manifest | Trust the tar | The extracted folder is checked file for file against what Kaggle mounted, so a download or extraction error cannot pass |
| Standard library only | PyYAML, the arc26 wheel | The script notebook runs the script with the image's Python; nothing else is needed |

## 4. How to run it

```bash
python scripts/repack_output.py --config configs/repack_kaggle_lib.yaml --smoke   # dev box, CPU
python scripts/build_submission.py --tag X2-repack --config configs/X2-repack-script.yaml
# the human, from the repository root (MANIFEST.md of the build lists the same commands):
python scripts/kaggle_ops.py dataset-create dist/X2-repack/arc26-code
python scripts/kaggle_ops.py push dist/X2-repack --timeout 3900                     # CPU, no brief
python scripts/kaggle_ops.py status alyahya737/arc26-repack-kaggle-lib             # until COMPLETE
```

Expected: about 5 to 10 minutes of CPU session (two reads of about 8 GB), an output of about
8 GB: `repack/parts/kaggle_lib.tar.000` to about `.007` and `repack/MANIFEST.json`.

The weights' output (`X2-01` attempt 5): Kaggle serves a single file of an output without
`Range` and cut the 18.6 GB `model.safetensors` at most 6.9 GB in on each of 7 tries, each from
byte 0, so `alyahya737/arc26-dl-qwen38-int4` version 1 is repacked the same way. It is attached
as a notebook input (`configs/X2-repack-weights-script.yaml`), whose mount the session finds by
search, so the build passes the folder as `--source` and `configs/repack_qwen38_int4.yaml` has
no `source` line:

```bash
python scripts/build_submission.py --tag X2-repack-w --config configs/X2-repack-weights-script.yaml
python scripts/kaggle_ops.py dataset-version dist/X2-repack-w/arc26-code -m "X2-repack-w"
python scripts/kaggle_ops.py push dist/X2-repack-w --timeout 5700                   # CPU, no brief
python scripts/kaggle_ops.py status alyahya737/arc26-repack-qwen38-int4            # until COMPLETE
```

Expected: about 20 minutes (two reads and one write of about 19.5 GB), an output of about
19.5 GB: `repack/parts/qwen38_int4.tar.000` to about `.018` and `repack/MANIFEST.json`.

## 5. Outputs

| File | Size | Where |
|---|---|---|
| `repack/parts/kaggle_lib.tar.NNN` | at most 1 GiB each | The notebook's output (version 1 the first time); fetched by Modal, never pulled locally |
| `repack/MANIFEST.json` | under 10 MB | The same output; fetched by Modal |
| `run_logs/` | small | The script notebook's own logs, pullable with `kaggle_ops.py pull` |

## 6. How to read the results

The session's log prints one JSON line: `files`, `bytes`, `seconds`, `parts`. `files` should be
in the thousands and `bytes` near the output's size. On Modal, `fetch.json` records the input's
`listed` block: the manifest's sha256, the parts, any retries per part, and `unpacked` (entries,
files, seconds) or `reused`.

## 7. Failure modes

| Symptom | Likely cause | First thing to change |
|---|---|---|
| `... is not a folder` | The install notebook is not attached at version 2 | The build's `kernel_source_versions` |
| `... is not empty` | A rerun in the same session folder | Push a fresh session |
| Fetch: `differs from the manifest's tree` | A part or the extraction lost a file, a mode, or a link | Read the named entries; a link outside the folder is refused by the `data` filter |
| Fetch: a part's `DownloadMismatch` four times in a row | The part changed on Kaggle (a new session overwrote the output) | Pin the version the brief names |
