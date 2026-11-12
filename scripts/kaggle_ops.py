"""Single gateway between this repository and Kaggle (BRIEF A0, A9; ADR-008).

Every Kaggle action the agent takes goes through this script, so the limits live in code instead
of memory:

- ``push``: a private notebook version. Accelerated (GPU or TPU) pushes need an approved pre-run
  brief and push exactly its ``Kaggle notebook`` on its ``Accelerator`` with its ``Timeout`` as
  the hard stop.
- ``status``: poll a notebook run.
- ``pull``: download run outputs matching the small-file allowlist into ``runs/<task-id>/``.
- ``dataset-create`` and ``dataset-version``: private datasets from a local folder, size capped.
- ``competition-submissions``: read the allowlisted competition's submission history.
- ``competition-submit``: submit an explicitly named, owned notebook version and its output.

The owner explicitly authorized competition submission on 1 Nov 2026. Submission requires the
competition, notebook, positive version, output file and description on the command line; no
default or latest version is selected. Publication and deletion remain unavailable. The API
token stays in ``~/.kaggle/`` and is read only by the ``kaggle`` CLI itself. The checks follow
the behavior of kaggle-cli 2.2.4 (ADR-008, implementation notes).

Usage::

    python scripts/kaggle_ops.py push <folder> --run-id P3.5-01 [--accelerator <ID>]
    python scripts/kaggle_ops.py push <folder> [--timeout <seconds>]    # CPU
    python scripts/kaggle_ops.py status <owner>/<slug>
    python scripts/kaggle_ops.py pull <owner>/<slug> --task-id P3.5
    python scripts/kaggle_ops.py dataset-create <folder>
    python scripts/kaggle_ops.py dataset-version <folder> -m "<notes>"
    python scripts/kaggle_ops.py competition-submissions --competition <competition>
    python scripts/kaggle_ops.py competition-submit --competition <competition> \
        --kernel <owner>/<slug> --version <number> --file submission.json -m "<description>"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
PRERUN_DIR = REPO / "docs" / "prerun"
NOT_BRIEFS = {"TEMPLATE.md", "README.md"}
# Kaggle username that owns every notebook and dataset pushed from here, set by the human. While
# it is empty, every push and dataset action is refused.
KAGGLE_OWNER = "alyahya737"
SUBMISSION_COMPETITIONS = frozenset({"arc-prize-2026-arc-agi-2"})
# IDs Kaggle accepts (kaggle-cli docs/kernels.md, Sep 2026). Retired IDs such as NvidiaTeslaP100
# and Gpu are left out on purpose: Kaggle silently runs them on 2xT4.
ACCELERATORS = {
    "NvidiaTeslaT4",
    "NvidiaL4",
    "NvidiaL4X1",
    "NvidiaTeslaA100",
    "NvidiaH100",
    "NvidiaRtxPro6000",
    "TpuV5E8",
    "TpuV6E8",
}
PINNING_TYPES = ("original", "latest")
# Hard stop of a session: at least a minute, at most Kaggle's 12 h session cap.
MIN_TIMEOUT_S = 60
MAX_TIMEOUT_S = 43200
# Output paths are relative to /kaggle/working and matched with re.search, so the pattern anchors
# itself. A segment uses POSIX portable file name characters and starts with a letter, digit, "_",
# or "-", so "." and ".." never match.
_SEGMENT = r"[A-Za-z0-9_-][A-Za-z0-9._-]*"
# submission.json at the root plus logs, metrics, reports, and small plots under run_logs/;
# weights and checkpoints never leave Kaggle.
PULL_PATTERN = (
    r"^(?:submission\.json"
    rf"|run_logs/(?:{_SEGMENT}/)*{_SEGMENT}\.(?:jsonl|json|md|csv|txt|log|yaml|png))\Z"
)
# Flag anything unexpectedly large after a pull; the human's disk and connection are limited.
MAX_PULL_FILE_BYTES = 50 * 1024**2
# Large artifacts are built on Kaggle, so a big local upload is almost certainly a mistake.
MAX_UPLOAD_BYTES = 200 * 1024**2
# Header keys this gateway or scripts/modal_ops.py reads (ADR-022 added the Modal ones); a brief
# that repeats one of them in its header is refused.
GATED_KEYS = (
    "Run-id",
    "Revision",
    "Code-commit",
    "Accelerator",
    "Timeout",
    "Kaggle notebook",
    "Provider",
    "GPU",
    "GPU count",
    "Max attempts",
    "Cost cap USD",
    "Output dataset",
)
# A brief without a Provider line is a Kaggle brief; a Modal brief is for scripts/modal_ops.py.
PROVIDER = "kaggle"
APPROVED = re.compile(r"Approved-by:[ \t]*\S")
COMMIT_SHA = re.compile(r"[0-9a-f]{7,40}")
# The stamp prerun_sync.py record appends to the Approved-by line: "(..., revision N)".
STAMPED_REVISION = re.compile(r"\brevision[ \t]+([0-9]+)", re.IGNORECASE)
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


class GatewayError(Exception):
    """A requested Kaggle action breaks an ADR-008 rule and was not sent."""


class Approval(NamedTuple):
    """What an approved pre-run brief allows: one commit, accelerator, hard stop, and notebook."""

    code_commit: str
    accelerator: str
    timeout: int
    notebook: str


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, check=True, text=True
    ).stdout


def _shown(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def _safe_id(value: str) -> bool:
    return bool(SAFE_ID.fullmatch(value)) and value not in {".", ".."}


def _fenced_blocks(text: str) -> list[list[str]]:
    # A fence opens on any line starting with ``` and closes on a bare ``` line, so a ```bash
    # block in the body cannot shift which lines count as the header or the Approval block.
    blocks: list[list[str]] = []
    block: list[str] | None = None
    for line in text.split("\n"):
        if block is None:
            if line.startswith("```"):
                block = []
        elif line.rstrip() == "```":
            blocks.append(block)
            block = None
        else:
            block.append(line)
    return blocks


def _header(text: str) -> dict[str, str]:
    blocks = _fenced_blocks(text)
    if not blocks:
        raise GatewayError("brief has no header block (its first fenced block)")
    header: dict[str, str] = {}
    for line in blocks[0]:
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or not key:
            continue
        if key in header and key in GATED_KEYS:
            raise GatewayError(f"brief header repeats {key!r}")
        header.setdefault(key, value.strip())
    return header


def _approval_line(text: str) -> str:
    lines = [line for line in text.split("\n") if line.lstrip(" \t").startswith("Approved-by:")]
    if len(lines) != 1:
        raise GatewayError(
            f"brief is not approved: it has {len(lines)} 'Approved-by:' lines instead of one"
        )
    line = lines[0]
    for block in _fenced_blocks(text)[1:]:
        notes = any(row.startswith("Notes:") for row in block)
        if line.startswith("Approved-by:") and line in block and notes:
            return line
    raise GatewayError(
        "brief is not approved: its 'Approved-by:' line is outside the Approval block, the "
        "fenced block holding 'Approved-by:' and 'Notes:'"
    )


def brief_header(text: str) -> dict[str, str]:
    """Return the header fields of a well-formed pre-run brief.

    Why: this script and ``prerun_sync.py`` must read the same values from a brief, so both read
    header fields only from its first fenced block, refuse a header that repeats one of
    ``GATED_KEYS``, and refuse a brief whose ``Approved-by:`` line is missing, repeated, or
    outside the Approval block (the fenced block holding ``Approved-by:`` and ``Notes:``).

    Args:
        text: Full text of a pre-run brief.

    Returns:
        The ``Key: value`` lines of the header block, stripped; for a repeated key that is not in
        ``GATED_KEYS``, the first value.

    Raises:
        GatewayError: If the brief breaks one of those rules.
    """
    _approval_line(text)
    return _header(text)


def _approved_header(text: str) -> dict[str, str]:
    line = _approval_line(text)
    if not APPROVED.match(line):
        raise GatewayError("brief is not approved: no filled 'Approved-by:' line")
    header = _header(text)
    if not COMMIT_SHA.fullmatch(header.get("Code-commit", "")):
        raise GatewayError("brief has no 'Code-commit: <sha>' line in its header")
    # Read as prerun_sync.py reads it, which stamped the approval: leading digits, default 1.
    revision = re.match(r"\s*([0-9]+)", header.get("Revision", ""))
    current = int(revision.group(1)) if revision else 1
    stamped = [int(n) for n in STAMPED_REVISION.findall(line)]
    if any(n != current for n in stamped):
        raise GatewayError(
            f"brief is not approved at revision {current}: the approval is stamped revision "
            f"{', '.join(map(str, stamped))}"
        )
    return header


def approved_header(brief_text: str) -> dict[str, str]:
    """Return the header of an approved pre-run brief, for either gateway.

    Why: ``scripts/modal_ops.py`` must apply exactly this script's approval rules, so it reads
    the header through this function instead of a copy of them.

    Args:
        brief_text: Full text of ``docs/prerun/<run-id>.md``.

    Returns:
        The header fields (``brief_header``).

    Raises:
        GatewayError: As ``approved_code_commit``.
    """
    return _approved_header(brief_text)


def approved_code_commit(brief_text: str) -> str:
    """Return the code commit of an approved pre-run brief.

    Args:
        brief_text: Full text of ``docs/prerun/<run-id>.md``.

    Returns:
        The SHA on the header's ``Code-commit:`` line.

    Raises:
        GatewayError: If ``brief_header`` refuses the brief, its ``Approved-by:`` line is blank,
            its header has no ``Code-commit`` SHA, or a ``revision N`` stamp on the
            ``Approved-by:`` line differs from the header's ``Revision``.
    """
    return _approved_header(brief_text)["Code-commit"]


def files_changed_outside_docs(commit: str) -> list[str]:
    """List files outside ``docs/`` that differ between ``commit`` and ``HEAD``.

    Why: an approved brief covers exactly one version of the code and config; only documentation
    (including the brief itself) may change between approval and launch.

    Args:
        commit: The brief's ``Code-commit`` SHA.

    Returns:
        Repository-relative paths changed outside ``docs/``; empty when the code is unchanged.
    """
    changed = _git("diff", "--name-only", commit, "HEAD").split()
    return [p for p in changed if not p.startswith("docs/")]


def check_brief(run_id: str) -> Approval:
    """Read what the approved pre-run brief of an accelerated push allows.

    Why: the human approves one code commit, notebook, accelerator, and hard stop, and the push
    must use exactly those. Only the brief is read here, so ``cmd_push`` can refuse a mismatch
    before it runs git.

    Args:
        run_id: Run identifier, the brief's file stem, for example ``P3.5-01``.

    Returns:
        The brief's ``Code-commit``, ``Accelerator`` ID, ``Timeout`` in seconds, and
        ``Kaggle notebook``.

    Raises:
        GatewayError: If the run id could leave ``docs/prerun/``, the brief is missing, not
            approved (``approved_code_commit``), names another ``Run-id`` or a ``Provider``
            other than ``kaggle``, ``Accelerator`` is
            not one ID from ``ACCELERATORS``, ``Timeout`` is not an integer from
            ``MIN_TIMEOUT_S`` to ``MAX_TIMEOUT_S``, or ``Kaggle notebook`` is missing.
    """
    if not _safe_id(run_id):
        raise GatewayError(f"bad run id {run_id!r}")
    brief = PRERUN_DIR / f"{run_id}.md"
    if not brief.is_file():
        raise GatewayError(f"no pre-run brief at {_shown(brief)}")
    header = _approved_header(brief.read_text(encoding="utf-8"))
    if header.get("Run-id") != run_id:
        raise GatewayError(f"brief header Run-id {header.get('Run-id')!r} is not {run_id!r}")
    if header.get("Provider", PROVIDER) != PROVIDER:
        raise GatewayError(
            f"brief names Provider {header['Provider']!r}: it is launched by its own gateway "
            "(scripts/modal_ops.py for modal), never pushed to Kaggle"
        )
    accelerator = header.get("Accelerator")
    if accelerator not in ACCELERATORS:
        raise GatewayError(
            f"brief Accelerator {accelerator!r} is not one Kaggle ID from ACCELERATORS: "
            f"{', '.join(sorted(ACCELERATORS))}"
        )
    timeout = header.get("Timeout", "")
    if not re.fullmatch(r"[0-9]+", timeout) or not (
        MIN_TIMEOUT_S <= int(timeout) <= MAX_TIMEOUT_S
    ):
        raise GatewayError(
            f"brief needs 'Timeout: <seconds>', an integer from {MIN_TIMEOUT_S} to "
            f"{MAX_TIMEOUT_S}; got {timeout!r}"
        )
    notebook = header.get("Kaggle notebook", "")
    if not notebook:
        raise GatewayError("brief needs 'Kaggle notebook: <owner>/<slug>'")
    return Approval(header["Code-commit"], accelerator, int(timeout), notebook)


def check_code_unchanged(commit: str) -> None:
    """Refuse a push unless ``HEAD`` still holds the code and config of the approved commit.

    Args:
        commit: The brief's ``Code-commit`` SHA.

    Raises:
        GatewayError: If the working tree is dirty or files outside ``docs/`` changed since
            ``commit``.
    """
    if _git("status", "--porcelain").strip():
        raise GatewayError("working tree is not clean; commit first so the run is reproducible")
    changed = files_changed_outside_docs(commit)
    if changed:
        raise GatewayError(f"code changed since the approved commit {commit}: {changed}")


def briefed_notebooks() -> set[str]:
    """Return the ``Kaggle notebook`` of every pre-run brief in ``docs/prerun/``.

    Why: Kaggle may keep a slug's previous accelerator, so a CPU push must never reuse the slug of
    a notebook that runs accelerated.

    Returns:
        The values from the header of every ``*.md`` in ``PRERUN_DIR`` except ``TEMPLATE.md`` and
        ``README.md``; a brief without the line adds nothing.

    Raises:
        GatewayError: If a brief has no header block or repeats a gated key, so its notebook
            cannot be read unambiguously.
    """
    notebooks: set[str] = set()
    for path in sorted(PRERUN_DIR.glob("*.md")):
        if path.name in NOT_BRIEFS:
            continue
        try:
            header = _header(path.read_text(encoding="utf-8"))
        except GatewayError as err:
            raise GatewayError(f"{_shown(path)}: {err}") from err
        if header.get("Kaggle notebook"):
            notebooks.add(header["Kaggle notebook"])
    return notebooks


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise GatewayError(f"{path.name} is not valid JSON: {err}") from err
    if not isinstance(data, dict):
        raise GatewayError(f"{path.name} must hold a JSON object")
    return data


def load_kernel_metadata(folder: Path) -> dict:
    """Read ``kernel-metadata.json`` and require the notebook to be private.

    Args:
        folder: Folder holding the notebook and its ``kernel-metadata.json``.

    Returns:
        The parsed metadata.

    Raises:
        GatewayError: If the file is missing or malformed, or ``is_private`` is not ``true``.
    """
    path = folder / "kernel-metadata.json"
    if not path.is_file():
        raise GatewayError(f"missing {path}")
    meta = _read_json(path)
    if meta.get("is_private") is not True:
        raise GatewayError("kernel-metadata.json must set \"is_private\": true")
    return meta


def _flag(value: object) -> bool | None:
    # kaggle-cli's get_bool: a JSON bool, or the string "true" or "false" in any case.
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def uses_gpu(meta: dict, accelerator: str | None) -> bool:
    """Return whether a push runs on an accelerator (GPU or TPU) and so spends quota.

    Why: kaggle-cli 2.2.4 sends ``machine_shape`` from the metadata whenever ``--accelerator`` is
    absent, so the metadata alone can select an accelerator.

    Args:
        meta: Parsed ``kernel-metadata.json``.
        accelerator: The ``--accelerator`` value, if any.

    Returns:
        True when ``enable_gpu`` or ``enable_tpu`` is true (a bool, or ``"true"`` in any case),
        ``machine_shape`` is non-empty, or an accelerator is requested.
    """
    return (
        _flag(meta.get("enable_gpu")) is True
        or _flag(meta.get("enable_tpu")) is True
        or meta.get("machine_shape") not in (None, "")
        or bool(accelerator)
    )


def accelerator_id(meta: dict, accelerator: str | None) -> str | None:
    """Return the accelerator ID a push names, refusing unknown or conflicting IDs.

    Why: kaggle-cli forwards the ID unchecked, and Kaggle silently replaces retired IDs such as
    ``NvidiaTeslaP100`` and ``Gpu`` with 2xT4, so only IDs in ``ACCELERATORS`` pass.

    Args:
        meta: Parsed ``kernel-metadata.json``.
        accelerator: The ``--accelerator`` value, if any.

    Returns:
        The ID from ``--accelerator`` or ``machine_shape``, or None when neither names one.

    Raises:
        GatewayError: If a named ID is not in ``ACCELERATORS`` or the two name different IDs.
    """
    named = {}
    shape = meta.get("machine_shape")
    for source, value in (("--accelerator", accelerator), ("machine_shape", shape)):
        if value in (None, ""):
            continue
        if not isinstance(value, str) or value not in ACCELERATORS:
            raise GatewayError(
                f"{source} {value!r} is not in ACCELERATORS: {', '.join(sorted(ACCELERATORS))}"
            )
        named[source] = value
    if len(set(named.values())) > 1:
        raise GatewayError(
            f"--accelerator {named['--accelerator']!r} differs from machine_shape "
            f"{named['machine_shape']!r}"
        )
    return next(iter(named.values()), None)


def check_flags(meta: dict, accelerator: str | None) -> None:
    """Refuse ``enable_gpu`` and ``enable_tpu`` values that are unreadable or contradict the ID.

    Why: kaggle-cli fails on a flag that is not a boolean, and a flag that disagrees with the
    accelerator ID sends Kaggle contradictory settings.

    Args:
        meta: Parsed ``kernel-metadata.json``.
        accelerator: The push's accelerator ID (``accelerator_id``), or None.

    Raises:
        GatewayError: If a flag is present but is not a bool or ``"true"`` or ``"false"``, or a
            GPU ID (``Nvidia*``) comes with ``enable_gpu`` false or ``enable_tpu`` true, or a TPU
            ID (``Tpu*``) with ``enable_tpu`` false or ``enable_gpu`` true. Absent flags pass.
    """
    flags = {}
    for key in ("enable_gpu", "enable_tpu"):
        if key in meta:
            flags[key] = _flag(meta[key])
            if flags[key] is None:
                raise GatewayError(
                    f'kernel-metadata.json "{key}" must be true or false; got {meta[key]!r}'
                )
    if accelerator is None:
        return
    gpu = accelerator.startswith("Nvidia")
    wanted, other = ("enable_gpu", "enable_tpu") if gpu else ("enable_tpu", "enable_gpu")
    if flags.get(wanted) is False or flags.get(other) is True:
        raise GatewayError(
            f'{accelerator} is a {"GPU" if gpu else "TPU"} ID, so kernel-metadata.json may not set '
            f'"{wanted}" false or "{other}" true'
        )


def check_owner(meta: dict, name: str) -> None:
    """Refuse metadata that could act on a notebook or dataset outside ``KAGGLE_OWNER``.

    Why: kaggle-cli acts on ``id_no`` in preference to ``id`` (``kernels pull -m`` writes it), so a
    copied metadata file could push to or version someone else's notebook or dataset.

    Args:
        meta: Parsed ``kernel-metadata.json`` or ``dataset-metadata.json``.
        name: The metadata file name, used in messages.

    Raises:
        GatewayError: If the metadata has ``id_no``, ``KAGGLE_OWNER`` is empty, or ``id`` is not
            ``<KAGGLE_OWNER>/<slug>``.
    """
    if "id_no" in meta:
        raise GatewayError(f'{name} has "id_no", which kaggle-cli prefers over "id"; remove it')
    if not KAGGLE_OWNER:
        raise GatewayError(
            "KAGGLE_OWNER in scripts/kaggle_ops.py is empty; every push and dataset action is "
            "refused until the human sets it to the Kaggle username"
        )
    ref = meta.get("id")
    owner, _, slug = ref.partition("/") if isinstance(ref, str) else ("", "", "")
    if owner != KAGGLE_OWNER or not _safe_id(slug):
        raise GatewayError(
            f'{name} "id" must be "<owner>/<slug>" with owner KAGGLE_OWNER '
            f"({KAGGLE_OWNER!r}); got {ref!r}"
        )


def check_kernel_metadata(meta: dict, accelerated: bool) -> None:
    """Refuse notebook metadata that would leave owner, internet, or image to kaggle-cli defaults.

    Why: kaggle-cli 2.2.4 pushes to ``id_no`` in preference to ``id``, turns internet on when
    ``enable_internet`` is absent, and without ``docker_image_pinning_type`` a push may change the
    image implicitly. Competition sessions must run with internet off.

    Args:
        meta: Parsed ``kernel-metadata.json``.
        accelerated: Whether the push runs on a GPU or TPU (``uses_gpu``).

    Raises:
        GatewayError: If ``check_owner`` refuses, ``enable_internet`` is not an explicit boolean
            or is true on an accelerated push, or ``docker_image_pinning_type`` is not
            ``original`` or ``latest``.
    """
    check_owner(meta, "kernel-metadata.json")
    internet = _flag(meta.get("enable_internet"))
    if internet is None:
        raise GatewayError(
            'kernel-metadata.json must set "enable_internet" to true or false; kaggle-cli '
            "defaults it to true"
        )
    if accelerated and internet:
        raise GatewayError('accelerated pushes need "enable_internet": false')
    if meta.get("docker_image_pinning_type") not in PINNING_TYPES:
        raise GatewayError(
            'kernel-metadata.json must set "docker_image_pinning_type" to "original" or "latest"'
        )


def check_code_file(folder: Path, meta: dict) -> None:
    """Refuse a code file outside the folder, and a notebook that carries Kaggle settings.

    Why: kaggle-cli sends the code file as is, clearing only cell outputs, and a Kaggle-saved
    notebook keeps ``accelerator`` and ``isInternetEnabled`` in its top-level ``metadata.kaggle``,
    beside the checked ``kernel-metadata.json``.

    Args:
        folder: Resolved folder holding the code file and ``kernel-metadata.json``.
        meta: Parsed ``kernel-metadata.json``.

    Raises:
        GatewayError: If ``code_file`` is missing or leaves the folder, or, for a notebook kernel
            (``kernel_type`` notebook or a ``.ipynb`` code file), is not a JSON notebook or has a
            ``kaggle`` key in its top-level ``metadata``.
    """
    code_file = meta.get("code_file")
    is_notebook = meta.get("kernel_type") == "notebook" or (
        isinstance(code_file, str) and code_file.lower().endswith(".ipynb")
    )
    if code_file is None and not is_notebook:
        return
    if not isinstance(code_file, str) or not code_file:
        raise GatewayError('kernel-metadata.json needs a "code_file"')
    path = (folder / code_file).resolve()
    if not path.is_relative_to(folder):
        raise GatewayError(f"code_file {code_file!r} resolves outside {folder}")
    if not path.is_file():
        raise GatewayError(f"code_file {code_file!r} is missing")
    if not is_notebook:
        return
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise GatewayError(f"code_file {code_file!r} is not a readable notebook: {err}") from err
    if not isinstance(notebook, dict):
        raise GatewayError(f"code_file {code_file!r} is not a notebook: not a JSON object")
    settings = notebook.get("metadata")
    if isinstance(settings, dict) and "kaggle" in settings:
        raise GatewayError(
            f'code_file {code_file!r} has a "kaggle" object in its metadata (Kaggle-saved '
            "accelerator and internet settings), which the push would send as is; remove it"
        )


def folder_bytes(folder: Path) -> int:
    """Return the total size of the files under ``folder``.

    Args:
        folder: Folder to measure.

    Returns:
        Size in bytes.
    """
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def check_dataset_folder(folder: Path) -> None:
    """Require owned dataset metadata and keep uploads under the size cap.

    Args:
        folder: Folder holding the files and ``dataset-metadata.json``.

    Raises:
        GatewayError: If the metadata is missing, the folder exceeds ``MAX_UPLOAD_BYTES``, or
            ``check_owner`` refuses the metadata.
    """
    path = folder / "dataset-metadata.json"
    if not path.is_file():
        raise GatewayError(f"missing {path}")
    size = folder_bytes(folder)
    if size > MAX_UPLOAD_BYTES:
        raise GatewayError(
            f"{size / 1024**2:.0f} MB exceeds the {MAX_UPLOAD_BYTES // 1024**2} MB upload cap; "
            "build large artifacts on Kaggle instead"
        )
    check_owner(_read_json(path), path.name)


def pull_allowed(rel_path: str) -> bool:
    """Return whether a pull may keep an output file (``PULL_PATTERN``).

    Args:
        rel_path: Path relative to ``/kaggle/working``, with ``/`` separators.

    Returns:
        True for ``submission.json`` at the root and for log, metric, report, and plot files under
        ``run_logs/``; False for anything else, including ``.`` and ``..`` segments.
    """
    return re.search(PULL_PATTERN, rel_path) is not None


def _kaggle(*args: str) -> int:
    if shutil.which("kaggle") is None:
        raise GatewayError("kaggle CLI not found on PATH")
    print("+ kaggle " + " ".join(args), flush=True)
    return subprocess.run(["kaggle", *args], cwd=REPO).returncode


def cmd_push(args: argparse.Namespace) -> int:
    """Push a private notebook version; accelerated pushes need an approved, matching brief.

    Why: an accelerated push spends GPU or TPU quota without a human click, so it must run the
    approved code on the approved notebook and accelerator with the approved hard stop. Checks run
    cheapest first: metadata, IDs, and the code file, then the brief's text, then git.

    Args:
        args: Parsed ``push`` arguments: ``folder``, ``run_id``, ``accelerator``, ``timeout``.

    Returns:
        The kaggle CLI exit code.

    Raises:
        GatewayError: If any check refuses the push; nothing is sent.
    """
    if args.run_id is not None and not _safe_id(args.run_id):
        raise GatewayError(f"bad run id {args.run_id!r}")
    folder = Path(args.folder).resolve()
    meta = load_kernel_metadata(folder)
    accelerated = uses_gpu(meta, args.accelerator)
    accelerator = accelerator_id(meta, args.accelerator)
    if accelerated and not args.run_id:
        raise GatewayError(
            "accelerated (GPU or TPU) push needs --run-id and an approved docs/prerun/<run-id>.md"
        )
    check_kernel_metadata(meta, accelerated)
    check_flags(meta, accelerator)
    check_code_file(folder, meta)
    if not accelerated:
        if args.timeout is not None and not MIN_TIMEOUT_S <= args.timeout <= MAX_TIMEOUT_S:
            raise GatewayError(f"--timeout must be from {MIN_TIMEOUT_S} to {MAX_TIMEOUT_S} seconds")
        # Compared without case to fail closed on a slug that differs only in case.
        if meta["id"].lower() in {notebook.lower() for notebook in briefed_notebooks()}:
            raise GatewayError(
                f"{meta['id']} is the Kaggle notebook of a pre-run brief; CPU pushes need their "
                "own slug, since Kaggle may keep a slug's previous accelerator"
            )
        extra = [] if args.timeout is None else ["-t", str(args.timeout)]
        return _kaggle("kernels", "push", "-p", str(folder), *extra)
    if accelerator is None:
        raise GatewayError(
            "accelerated push names no accelerator ID: pass --accelerator <ID> or set machine_shape"
        )
    approval = check_brief(args.run_id)
    if meta["id"] != approval.notebook:
        raise GatewayError(
            f"kernel-metadata.json id {meta['id']!r} differs from the brief's Kaggle notebook "
            f"{approval.notebook!r}"
        )
    if accelerator != approval.accelerator:
        raise GatewayError(
            f"accelerator {accelerator} differs from the brief's Accelerator {approval.accelerator}"
        )
    if args.timeout is not None and args.timeout != approval.timeout:
        raise GatewayError(
            f"--timeout {args.timeout} differs from the brief's Timeout {approval.timeout}; "
            "accelerated pushes take the timeout from the approved brief"
        )
    check_code_unchanged(approval.code_commit)
    limits = ["--accelerator", accelerator, "-t", str(approval.timeout)]
    return _kaggle("kernels", "push", "-p", str(folder), *limits)


def cmd_status(args: argparse.Namespace) -> int:
    """Print the run status of a notebook."""
    return _kaggle("kernels", "status", args.kernel)


def _file_stamps(folder: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in sorted(folder.rglob("*")) if p.is_file()}


def cmd_pull(args: argparse.Namespace) -> int:
    """Download allowlisted outputs of a run into ``runs/<task-id>/`` and police the result.

    Why: kaggle-cli 2.2.4 does not sanitize server file names and always writes the session log
    ``<slug>.log``. Any other file the download added or changed that ``pull_allowed`` rejects is
    removed and reported, with the directories that leaves empty. Files from earlier pulls, such
    as another notebook's session log, are left alone.

    Args:
        args: Parsed ``pull`` arguments: ``kernel`` (``<owner>/<slug>``) and ``task_id``.

    Returns:
        The kaggle CLI exit code, or 2 when the CLI succeeded but files had to be removed.

    Raises:
        GatewayError: If the task id could leave ``runs/``.
    """
    if not _safe_id(args.task_id):
        raise GatewayError(f"bad task id {args.task_id!r}")
    dest = REPO / "runs" / args.task_id
    dest.mkdir(parents=True, exist_ok=True)
    before = _file_stamps(dest)
    code = _kaggle(
        "kernels", "output", args.kernel, "-p", str(dest), "--file-pattern", PULL_PATTERN
    )
    slug = args.kernel.split("/")[1] if "/" in args.kernel else args.kernel
    removed = []
    for path, stamp in _file_stamps(dest).items():
        rel = path.relative_to(dest).as_posix()
        if before.get(path) != stamp and rel != f"{slug}.log" and not pull_allowed(rel):
            path.unlink()
            removed.append(path)
            print(f"removed {_shown(path)}: not in the pull allowlist", file=sys.stderr)
    for path in removed:
        parent = path.parent
        while parent != dest and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    large = [p for p in dest.rglob("*") if p.is_file() and p.stat().st_size > MAX_PULL_FILE_BYTES]
    for p in large:
        print(f"warning: {p.relative_to(REPO)} is {p.stat().st_size / 1024**2:.0f} MB", flush=True)
    return (code or 2) if removed else code


def cmd_dataset_create(args: argparse.Namespace) -> int:
    """Create a private dataset from a local folder."""
    folder = Path(args.folder).resolve()
    check_dataset_folder(folder)
    return _kaggle("datasets", "create", "-p", str(folder))


def cmd_dataset_version(args: argparse.Namespace) -> int:
    """Add a version to an existing dataset, keeping old versions."""
    folder = Path(args.folder).resolve()
    check_dataset_folder(folder)
    return _kaggle("datasets", "version", "-p", str(folder), "-m", args.message)


def _check_submission_competition(competition: str) -> None:
    if competition not in SUBMISSION_COMPETITIONS:
        raise GatewayError(f"competition {competition!r} is not in the submission allowlist")


def cmd_competition_submissions(args: argparse.Namespace) -> int:
    """Read the allowlisted competition's submission history, without changing submissions."""
    _check_submission_competition(args.competition)
    if not 1 <= args.page_size <= 200:
        raise GatewayError("submission page size must be from 1 to 200")
    command = [
        "competitions", "submissions", args.competition, "--format", "json",
        "--page-size", str(args.page_size),
    ]
    if args.page_token:
        command.extend(["--page-token", args.page_token])
    return _kaggle(*command)


def cmd_competition_submit(args: argparse.Namespace) -> int:
    """Submit one explicitly pinned owned notebook output to the allowlisted competition.

    The installed CLI routes ``kernel`` plus ``version`` to ``competition_submit_code``;
    ``file`` names that version's output, never a file uploaded from this machine. No retry is
    made after an uncertain response, since a repeated request can consume another submission.

    Args:
        args: Explicit competition, kernel, version, file_name and message arguments.

    Returns:
        The Kaggle CLI exit code.

    Raises:
        GatewayError: If the competition, notebook owner, version, output name or message is
            outside the permitted request.
    """
    _check_submission_competition(args.competition)
    check_owner({"id": args.kernel}, "competition submission notebook")
    if isinstance(args.version, bool) or not isinstance(args.version, int) or args.version < 1:
        raise GatewayError("competition submission needs an explicit positive notebook version")
    if args.file_name != "submission.json":
        raise GatewayError("competition submission output must be exactly submission.json")
    if not isinstance(args.message, str) or not args.message.strip() or "\x00" in args.message:
        raise GatewayError("competition submission needs a nonempty description without NUL")
    return _kaggle(
        "competitions", "submit", args.competition, "-k", args.kernel,
        "-v", str(args.version), "-f", args.file_name, "--message=" + args.message,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; its subcommands are the only Kaggle actions allowed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="push a private notebook version")
    push.add_argument("folder")
    push.add_argument("--run-id", help="pre-run brief id, required for GPU and TPU pushes")
    push.add_argument("--accelerator", help="Kaggle accelerator ID; must equal the brief's")
    push.add_argument(
        "--timeout",
        type=int,
        help="hard stop in seconds (60 to 43200) for CPU pushes; accelerated pushes use the "
        "brief's Timeout",
    )
    push.set_defaults(func=cmd_push)

    status = sub.add_parser("status", help="poll a notebook run")
    status.add_argument("kernel", help="<owner>/<slug>")
    status.set_defaults(func=cmd_status)

    pull = sub.add_parser("pull", help="download allowlisted outputs into runs/<task-id>/")
    pull.add_argument("kernel", help="<owner>/<slug>")
    pull.add_argument("--task-id", required=True)
    pull.set_defaults(func=cmd_pull)

    create = sub.add_parser("dataset-create", help="create a private dataset")
    create.add_argument("folder")
    create.set_defaults(func=cmd_dataset_create)

    version = sub.add_parser("dataset-version", help="version an existing dataset")
    version.add_argument("folder")
    version.add_argument("-m", "--message", required=True)
    version.set_defaults(func=cmd_dataset_version)

    submissions = sub.add_parser("competition-submissions", help="read submission history")
    submissions.add_argument("--competition", required=True)
    submissions.add_argument("--page-size", type=int, default=200)
    submissions.add_argument("--page-token")
    submissions.set_defaults(func=cmd_competition_submissions)

    submit = sub.add_parser("competition-submit", help="submit an explicit owned notebook version")
    submit.add_argument("--competition", required=True)
    submit.add_argument("--kernel", required=True, help="owned <owner>/<slug>")
    submit.add_argument("--version", required=True, type=int)
    submit.add_argument("--file", required=True, dest="file_name", choices=["submission.json"])
    submit.add_argument("-m", "--message", required=True)
    submit.set_defaults(func=cmd_competition_submit)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one gateway command.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The kaggle CLI exit code, or 2 when the gateway refuses the action or a pull had to
        remove files outside the allowlist.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except GatewayError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
