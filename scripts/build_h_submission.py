"""Wrap the reference submission build with a pinned, checked H initialization.

The original builder still produces the wheel, install, v1 scope and submission plumbing.
This wrapper adds a CPU checkpoint preflight and the runner's H fields in both modes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

SPEC = importlib.util.spec_from_file_location(
    "h_base_builder", Path(__file__).with_name("build_submission.py")
)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
BuildError = BASE.BuildError
REPO = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = Path("configs/h_submission_source.yaml")
RUN_ANCHOR = "    return RunConfig(\n"


def parse_source(text: str) -> dict[str, Any]:
    """Validate the immutable checkpoint dataset identity and digest.

    Args:
        text: Source YAML from the committed tree.

    Returns:
        The four checked source fields.

    Raises:
        BuildError: If fields, identifiers, version, filename or digest are invalid.
    """
    try:
        source = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise BuildError(f"invalid H source YAML: {error}") from error
    if not isinstance(source, dict) or set(source) != {"dataset", "version", "filename", "sha256"}:
        raise BuildError("H source needs exactly dataset, version, filename and sha256")
    if not isinstance(source["dataset"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*", source["dataset"]
    ):
        raise BuildError("invalid H dataset identity")
    if type(source["version"]) is not int or source["version"] < 1:
        raise BuildError("H dataset version must be a positive integer")
    if not isinstance(source["filename"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*\.pt", source["filename"]
    ):
        raise BuildError("H filename must be one controlled .pt basename")
    if not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise BuildError("H sha256 must contain 64 lowercase hexadecimal characters")
    return source


def checkpoint_cell(source: dict[str, Any]) -> str:
    """Return the fail-closed CPU preflight cell for one pinned checkpoint.

    Args:
        source: Checked source fields from parse_source.

    Returns:
        Compilable notebook Python using the original install and log paths.
    """
    return f'''H_SOURCE = {source!r}
H_CHECKPOINT = None
H_CHECK_PASSED = False
with stage("h_checkpoint") as h_record:
    if not INSTALLED or PYTHON is None:
        raise RuntimeError("H checkpoint preflight requires the successful offline install")
    h_owner, h_slug = H_SOURCE["dataset"].split("/")
    h_roots = [INPUT / "datasets" / h_owner / h_slug, INPUT / h_slug]
    h_candidates = {{path.resolve() for root in h_roots if root.is_dir()
                    for path in root.rglob(H_SOURCE["filename"]) if path.is_file()}}
    h_matches = sorted(path for path in h_candidates
                       if file_sha256(str(path)) == H_SOURCE["sha256"])
    if len(h_matches) != 1:
        raise RuntimeError(f"expected one pinned H checkpoint, found {{len(h_matches)}} matching "
                           f"files in {{len(h_candidates)}} candidates")
    H_CHECKPOINT = str(h_matches[0])
    h_report = RUN_LOGS / "h_checkpoint.json"
    h_check = run_logged([PYTHON, "-m", "arc26.h_submission", "--check-only", "--checkpoint",
                          H_CHECKPOINT, "--sha256", H_SOURCE["sha256"], "--report",
                          str(h_report)], timeout=900)
    if h_check.returncode != 0:
        raise RuntimeError(f"H checkpoint preflight failed with exit {{h_check.returncode}}")
    h_result = json.loads(h_report.read_text())
    if h_result.get("ok") is not True:
        raise RuntimeError("H checkpoint preflight did not report ok=true")
    h_record.update(checkpoint=H_CHECKPOINT, sha256=H_SOURCE["sha256"], ok=True)
    H_CHECK_PASSED = True
# stage() deliberately swallows competition errors. This guard prevents running plain v1
# after an H-preflight failure; the environment cell already wrote valid placeholders.
if not H_CHECK_PASSED:
    raise RuntimeError("H preflight failed; refusing to run or label this as v1 plus H")
'''


def patch_notebook(notebook: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Add H preflight and runner fields without changing the reference scope.

    Args:
        notebook: Original submission notebook mapping.
        source: Checked checkpoint identity.

    Returns:
        A copied notebook with exactly one preflight and RunConfig injection.

    Raises:
        BuildError: If the expected original cells or single anchor have changed.
    """
    result = copy.deepcopy(notebook)
    cells = result.get("cells", [])
    anchor_cells = [cell for cell in cells if RUN_ANCHOR in "".join(cell.get("source", []))]
    if len(anchor_cells) != 1:
        raise BuildError("H wrapper requires exactly one RunConfig cell")
    run = anchor_cells[0]
    code = "".join(run["source"])
    if code.count(RUN_ANCHOR) != 1:
        raise BuildError("H wrapper requires exactly one RunConfig anchor")
    run["source"] = code.replace(
        RUN_ANCHOR,
        RUN_ANCHOR + "        hyper_checkpoint=H_CHECKPOINT,\n"
        "        hyper_sha256=H_SOURCE[\"sha256\"],\n"
        "        hyper_log_dir=str(RUN_LOGS / \"h-start\"),\n",
    ).splitlines(keepends=True)
    installs = [index for index, cell in enumerate(cells)
                if "INSTALLED = False\n" in "".join(cell.get("source", []))]
    if len(installs) != 1:
        raise BuildError("H wrapper requires exactly one original install cell")
    note = ("## Pinned H checkpoint preflight\n\n"
            "CPU validation checks the attached checkpoint hash, architecture and finite weights "
            "before either mode starts. Failure stops this H build; valid placeholders remain.")
    at = installs[0] + 1
    cells[at:at] = [BASE.notebook_builder._cell(at, "h-checkpoint-note", "markdown", note),
                   BASE.notebook_builder._cell(at + 1, "h-checkpoint", "code",
                                              checkpoint_cell(source))]
    heading = ("# V1 plus H: production H initialization\n\n"
               "The predicted rank-16 attention delta is merged into each puzzle's base before "
               "the unchanged v1 rank-256 training scope and decode. Both modes validate the "
               "pinned checkpoint and use the H runner. This build is a new measured candidate.")
    cells.insert(0, BASE.notebook_builder._cell(0, "h-production-note", "markdown", heading))
    for cell in cells:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), "h-notebook", "exec")
    return result


def build(tag: str, config_path: Path, repo: Path = REPO) -> Path:
    """Build the original package, then add the explicit H runtime and complete hashes.

    Args:
        tag: Output folder name under dist.
        config_path: Committed original-builder config.
        repo: Repository with a clean committed tree.

    Returns:
        The complete upload folder.

    Raises:
        BuildError: If the base build, source validation or guarded injection fails.
    """
    repo = repo.resolve()
    BASE.require_clean(repo)
    commit, _ = BASE.head_commit(repo)
    source = parse_source(BASE._git(repo, "show", f"{commit}:{SOURCE_CONFIG.as_posix()}"))
    target = BASE.build(tag, config_path, repo)
    files = {path.relative_to(target).as_posix(): path.read_bytes()
             for path in target.rglob("*") if path.is_file() and path.name != "MANIFEST.md"}
    metadata = json.loads(files["kernel-metadata.json"])
    notebook_name = metadata["code_file"]
    if notebook_name != "kaggle_submission.ipynb":
        raise BuildError("H wrapper accepts only the original submission notebook")
    notebook = patch_notebook(json.loads(files[notebook_name]), source)
    files[notebook_name] = BASE.notebook_builder.notebook_bytes(notebook)
    pinned = f"{source['dataset']}/{source['version']}"
    metadata["dataset_sources"] = list(dict.fromkeys([*metadata["dataset_sources"], pinned]))
    files["kernel-metadata.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
    lines = [f"# V1 plus H build: {tag}", "", f"Code commit: `{commit}`.", "",
             "V1 rank-256 scope and decode are preserved. H initializes the base in both modes.",
             "H CPU preflight is mandatory, including in the free no-solver twin.",
             f"Pinned checkpoint dataset: `{pinned}`; filename `{source['filename']}`.",
             f"Checkpoint sha256: `{source['sha256']}`.", "",
             f"Config: `{config_path}`. Notebook: `{metadata['id']}`.", "",
             "Every artifact hash below describes the final wrapped output. The manifest itself "
             "is excluded.", "", "| File | SHA-256 |", "|---|---|"]
    lines.extend(f"| `{name}` | `{hashlib.sha256(data).hexdigest()}` |"
                 for name, data in sorted(files.items()))
    files["MANIFEST.md"] = ("\n".join(lines) + "\n").encode()
    BASE.write_folder(target, files)
    return target


def main(argv: list[str] | None = None) -> int:
    """Build the explicitly pinned H submission from command-line arguments.

    Args:
        argv: Arguments, or the process arguments.

    Returns:
        Zero on success; two on a refused build.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        folder = build(args.tag, args.config)
    except (BuildError, OSError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(f"wrote {folder}; MANIFEST.md lists the final H build hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
