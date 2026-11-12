"""Build the push folders of task P5.2's Kaggle runs from one commit (BRIEF A5, A8; ADR-008).

Purpose: turn a run config (``configs/P5.2-00.yaml``, ``configs/P5.2-00-fa2.yaml``,
``configs/P5.2-01.yaml``) into exactly what ``scripts/kaggle_ops.py`` pushes, reproducibly: the
same commit gives the same bytes. Runbook ``docs/runbooks/fn_probe.md``.

Outputs, in ``--out``:

- The kernel's code file: ``scripts/fn_env.py`` or ``scripts/fn_probe.py`` as committed, with
  the run's config embedded as JSON on its ``EMBEDDED_CONFIG`` line (a script kernel has one
  file, and the image's Python may lack PyYAML).
- ``kernel-metadata.json``: private, a script kernel, the config's image (a digest pinned as
  ``original``, or ``latest`` when the config gives none), internet and accelerator as the
  config says, and its inputs; checked with the gateway's own functions before anything is
  written.
- For the probe only, ``arc26-code-fn/``: the code dataset (the ``arc26`` wheel built with
  ``SOURCE_DATE_EPOCH`` from the last commit that changed the package, ``fn_probe.py``,
  ``fn_code.json`` with every file's sha256, ``dataset-metadata.json``). The embedded config's
  ``build`` section carries the commit and the same hashes, which the probe checks on Kaggle.
- ``MANIFEST.md``: the commit, every file's sha256, and the gateway commands.

Method: the tree must be clean (``--allow-dirty`` for tests), the commit is exported with
``git archive``, and every input is read from the export, so nothing uncommitted reaches Kaggle;
the folder is written beside ``--out`` and swapped in (``build_submission.write_folder``).

Usage::

    python scripts/build_fn_notebooks.py --config configs/P5.2-00.yaml --out dist/P5.2-00
    python scripts/build_fn_notebooks.py --config configs/P5.2-01.yaml --out dist/P5.2-01
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
EMBED_LINE = "EMBEDDED_CONFIG: dict[str, Any] | None = None\n"
CODE_MARKER = "fn_code.json"
DATASET_METADATA = "dataset-metadata.json"
# Code files the builder knows, and the script each is made from; a notebook's cells run
# fn_probe.py from the code dataset.
SOURCES = {"fn_env.py": "scripts/fn_env.py", "fn_probe.py": "scripts/fn_probe.py",
           "kaggle_submission.ipynb": "scripts/fn_probe.py"}
# Files of the code dataset besides the wheel: the solver and the CPU twin's fake server.
DATASET_SCRIPTS = ("scripts/fn_probe.py", "scripts/fn_fake_sglang.py")
# Packages the code dataset carries as one zip archive each, importable by zipimport with the
# file itself on sys.path (the engine's PYTHONPATH, configs/P5.2-01.yaml engine.env). One file,
# not a folder: the dataset upload (kaggle_ops.py dataset-version) skips subfolders. Named
# .pyz, not .zip: Kaggle extracts an uploaded .zip by its extension (revision 17's fn_kvq.zip
# arrived in alyahya737/arc26-code-fn-kv4 version 1 as the folder fn_kvq/fn_kvq/, 31 Oct 05:52)
# and keeps other names, as it kept the .whl. zipimport reads the archive whatever its name.
# P5.2-01 revision 18: the KV4 cache's fn_kvq (task P5.5), which the hash-guarded edits import.
DATASET_ZIPS = {"fn_kvq.pyz": "scripts/fn_kvq"}
# Archive extensions Kaggle unpacks on upload; no file of the code dataset may end with one.
KAGGLE_EXTRACTS = (".zip", ".tar", ".gz", ".tgz", ".7z", ".bz2", ".xz")
# A zip member's fixed time (the zip format's earliest), so the same files give the same bytes.
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def package_zip(tree: Path, folder: str, commit: str) -> bytes:
    """Return a byte-reproducible zip of a package's ``.py`` files from the exported commit.

    Members are ``<package>/<file>.py`` in name order, deflated, at ``ZIP_TIME``, mode 0644, so
    Python's zipimport finds the package with the zip itself on ``sys.path`` (and ``inspect``
    reads the sources through the zip's loader, as Triton's JIT needs).

    Args:
        tree: The exported commit.
        folder: The package's folder in the repository (``scripts/fn_kvq``).
        commit: The commit, for messages.

    Returns:
        The zip's bytes.

    Raises:
        BuildError: If the folder holds no ``.py`` file at the commit.
    """
    import io
    import zipfile

    base = tree / folder
    names = sorted(p.name for p in base.glob("*.py")) if base.is_dir() else []
    if not names:
        raise BuildError(f"{folder} holds no .py file at {commit[:12]}")
    package = Path(folder).name
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            info = zipfile.ZipInfo(f"{package}/{name}", date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, submission._read_export(tree, f"{folder}/{name}", commit))
    return buffer.getvalue()


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_arc26_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


submission = _load("build_submission")
gateway = submission.gateway
BuildError = submission.BuildError


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def embed(source: str, config: dict[str, Any]) -> str:
    """Return a script's text with the config on its ``EMBEDDED_CONFIG`` line.

    Args:
        source: The script as committed; it must hold :data:`EMBED_LINE` exactly once.
        config: The config to embed.

    Returns:
        The script with that line set to ``json.loads(<the config as JSON>)``.

    Raises:
        BuildError: If the line is missing or repeated.
    """
    if source.count(EMBED_LINE) != 1:
        raise BuildError("the script must hold its EMBEDDED_CONFIG line exactly once")
    text = json.dumps(config, sort_keys=False, separators=(",", ":"))
    line = f"EMBEDDED_CONFIG: dict[str, Any] | None = json.loads({text!r})\n"
    return source.replace(EMBED_LINE, line)


def resolve_extends(read: Any, rel: str) -> dict[str, Any]:
    """Return a config with its ``extends`` chain merged, base first (``deep_merge``).

    Why: P5.6-01's twin extends P5.6-01, which extends P5.2-01 (revision 20); up to P5.2-01's
    own twin a config extended one file at most, and that one-level chain gives the same result.

    Args:
        read: Returns a file of the commit by its repository path.
        rel: The config's path.

    Returns:
        The merged config, without ``extends``.

    Raises:
        BuildError: If the chain loops.
    """
    import yaml

    chain: list[dict[str, Any]] = []
    seen: list[str] = []
    name: str | None = rel
    while name:
        if name in seen:
            raise BuildError(f"extends loops: {seen + [name]}")
        seen.append(name)
        config = yaml.safe_load(read(name))
        name = config.pop("extends", None)
        chain.append(config)
    merged = chain.pop()
    while chain:
        merged = deep_merge(merged, chain.pop())
    return merged


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``over`` merged in: mappings merge key by key, other values replace.

    Args:
        base: The config a twin extends.
        over: The twin's own values.

    Returns:
        A new mapping.
    """
    out = json.loads(json.dumps(base))
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = json.loads(json.dumps(value))
    return out


# The notebook's cells. Cell 1 runs before anything else can fail: it records the session's
# start, checks the code dataset, and writes an all-[[0]] submission.json with arc26.submit
# (standard library only, imported from the wheel as a zip). Cell 2 runs fn_probe.py, which fills
# the file as tasks finish. Cell 3 checks the file once more, whatever cell 2 did.
CELL_SETUP = """import hashlib, json, os, subprocess, sys, time
from pathlib import Path

SESSION_START = time.time()
CONFIG = json.loads({config!r})
# FN_ROOT prefixes every path in tests on CPU; unset on Kaggle.
ROOT = Path(os.environ.get("FN_ROOT", "/"))
WORKING = ROOT / "kaggle" / "working"
INPUT = ROOT / "kaggle" / "input"
LOGS = WORKING / "run_logs" / "fn"
LOGS.mkdir(parents=True, exist_ok=True)


def find(marker, hints):
    for current, folders, files in os.walk(INPUT, followlinks=True):
        if len(Path(current).relative_to(INPUT).parts) >= 7:
            folders[:] = []
        if marker in files and any(h in current for h in hints):
            return Path(current)
    return None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


CODE = find("fn_code.json", ["arc26-code-fn"])
COMPETITION = find("arc-agi_test_challenges.json", ["arc-prize-2026"])
CHALLENGES = COMPETITION / "arc-agi_test_challenges.json"
CODE_OK = CODE is not None and all(
    (CODE / name).is_file() and sha256(CODE / name) == digest
    for name, digest in CONFIG["build"]["files"].items())
sys.path.insert(0, str(CODE / CONFIG["build"]["wheel"]))
from arc26.submit import finalize_submission

SETUP = {{"session_start": SESSION_START, "code": str(CODE), "code_ok": CODE_OK,
         "challenges": str(CHALLENGES), "commit": CONFIG["build"]["commit"],
         "rerun": os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""),
         "placeholder": finalize_submission(WORKING / "submission.json", CHALLENGES)}}
(LOGS / "notebook_setup.json").write_text(json.dumps(SETUP, indent=1))
print(json.dumps(SETUP, indent=1))
"""

CELL_SOLVE = """RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
MODE = "competition" if RERUN not in ("", "0", "false", "no") else "development"
if CONFIG.get("mode", "auto") != "auto":
    MODE = CONFIG["mode"]
END_S = float(CONFIG["time_plan"][MODE]["end_s"])
(WORKING / "fn_config.json").write_text(json.dumps(CONFIG))
SOLVE = {{"mode": MODE, "end_s": END_S}}
if CODE_OK:
    # The solver stops itself at END_S; the kill 120 s later only guards a hang.
    limit = max(END_S + 120 - (time.time() - SESSION_START), 60)
    try:
        done = subprocess.run(
            [sys.executable, str(CODE / "{script}"), "--config", str(WORKING / "fn_config.json"),
             "--root", str(ROOT), "--session-start", str(SESSION_START),
             "--commit", CONFIG["build"]["commit"]],
            timeout=limit, check=False)
        SOLVE["exit"] = done.returncode
    except subprocess.TimeoutExpired:
        SOLVE["exit"] = "timeout"
    except Exception as err:
        SOLVE["exit"] = f"{{type(err).__name__}}: {{err}}"
else:
    SOLVE["exit"] = "code dataset check failed: the placeholders stay"
print(SOLVE)
"""

CELL_FINAL = """FINAL = finalize_submission(WORKING / "submission.json", CHALLENGES,
                            raw_path=WORKING / "submission.json")
FINAL.update(SOLVE, notebook_s=round(time.time() - SESSION_START, 1))
(LOGS / "final_check.json").write_text(json.dumps(FINAL, indent=1))
print(json.dumps(FINAL, indent=1))
assert FINAL["valid"], "submission.json is not valid"
"""


def notebook(config: dict[str, Any], script: str = "fn_probe.py") -> bytes:
    """Return version 11's notebook: a title and the three cells, the config embedded.

    Why: a code competition submits a notebook; the notebook only frames ``fn_probe.py`` (from
    the code dataset, checked against the build's sha256) between a placeholder
    ``submission.json`` before anything can fail and a final check after anything happened.

    Args:
        config: The resolved config, ``build`` included.
        script: The code dataset's script that cell 2 runs: ``fn_probe.py``, or a Modal probe's
            ``fn_bisect.py`` (run ``P5.2-06``); the default gives the same bytes as before.

    Returns:
        The ``.ipynb`` bytes (nbformat 4, no Kaggle-saved metadata).
    """
    text = json.dumps(config, sort_keys=False, separators=(",", ":"))
    title = (f"# arc26 version 11: Qwen3.8-Flash-Next on SGLang, LLM first\n\nBuilt by "
             f"`scripts/build_fn_notebooks.py` from commit `{config['build']['commit']}` "
             f"(`{config['build']['config']}`). Runbook `docs/runbooks/fn_probe.md`.")

    def cell(kind: str, source: str) -> dict[str, Any]:
        lines = source.splitlines(keepends=True)
        if kind == "markdown":
            return {"cell_type": "markdown", "metadata": {}, "source": lines}
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": lines}

    cells = [cell("markdown", title), cell("code", CELL_SETUP.format(config=text)),
             cell("code", CELL_SOLVE.format(script=script)), cell("code", CELL_FINAL)]
    book = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    return (json.dumps(book, indent=1) + "\n").encode("utf-8")


def server_flags(probe: dict[str, Any]) -> list[str]:
    """Return the flags the probe passes to the server (values not null or false).

    Args:
        probe: The probe's config (``engine.flags``).

    Returns:
        The flag names in order.
    """
    return [f for f, v in probe["engine"]["flags"].items() if v is not None and v is not False]


def kernel_metadata(config: dict[str, Any], extra_datasets: list[str]) -> dict[str, Any]:
    """Return the kernel's ``kernel-metadata.json`` and check it with the gateway.

    Args:
        config: The run config (its ``kernel`` section).
        extra_datasets: Datasets added after the config's (the probe's code dataset).

    Returns:
        The metadata in the order it is written.

    Raises:
        BuildError: If the gateway would refuse it, or a kernel source has no integer version.
    """
    spec = config["kernel"]
    accelerator = spec.get("accelerator")
    sources = []
    for name, version in spec.get("kernel_sources", {}).items():
        if not isinstance(version, int) or isinstance(version, bool):
            raise BuildError(f"kernel source {name} needs an integer version; got {version!r}")
        sources.append(f"{name}/{version}")
    meta: dict[str, Any] = {
        "id": spec["id"],
        "title": spec["title"],
        "code_file": spec["code_file"],
        "language": "python",
        "kernel_type": spec.get("kernel_type", "script"),
        "is_private": True,
        "enable_gpu": accelerator is not None,
        "enable_internet": bool(spec["enable_internet"]),
    }
    if accelerator is not None:
        meta["machine_shape"] = accelerator
    # A digest pins that image ("original"); null takes Kaggle's newest ("latest").
    image = spec.get("docker_image")
    meta["docker_image_pinning_type"] = "original" if image else "latest"
    if image:
        meta["docker_image"] = image
    meta.update({
        "competition_sources": list(spec.get("competition_sources", [])),
        "model_sources": list(spec.get("model_sources", [])),
        "dataset_sources": list(spec.get("dataset_sources", [])) + extra_datasets,
        "kernel_sources": sources,
    })
    if spec["title"].lower().replace(" ", "-") != spec["id"].split("/", 1)[1]:
        raise BuildError(f"title {spec['title']!r} does not slugify to {spec['id']!r}")
    try:
        accelerated = gateway.uses_gpu(meta, None)
        gateway.check_kernel_metadata(meta, accelerated)
        gateway.check_flags(meta, gateway.accelerator_id(meta, None))
    except gateway.GatewayError as err:
        raise BuildError(f"kernel-metadata.json would be refused by the gateway: {err}") from err
    return meta


def code_dataset(slug: str, files: dict[str, bytes]) -> dict[str, bytes]:
    """Return the probe's code dataset: its files, ``fn_code.json``, and its metadata.

    Args:
        slug: ``<owner>/<slug>``.
        files: Name to bytes (the wheel and ``fn_probe.py``).

    Returns:
        Name to bytes, the metadata included.

    Raises:
        BuildError: If the gateway's owner check refuses the metadata.
    """
    meta = {
        "title": slug.split("/", 1)[1].replace("-", " "),
        "id": slug,
        "licenses": [{"name": "other"}],
        "description": (
            "The arc26 wheel and scripts/fn_probe.py for the arc26 Flash-Next probe (task P5.2), "
            "built by scripts/build_fn_notebooks.py from one commit; fn_code.json lists every "
            "file's sha256, which the probe checks before use."
        ),
    }
    try:
        gateway.check_owner(meta, DATASET_METADATA)
    except gateway.GatewayError as err:
        raise BuildError(str(err)) from err
    # No commit here: a docs-only commit leaves the dataset's bytes, so no new version is needed.
    listing = {"files": {n: _sha256(d) for n, d in sorted(files.items())}}
    out = dict(files)
    out[CODE_MARKER] = (json.dumps(listing, indent=1) + "\n").encode()
    out[DATASET_METADATA] = (json.dumps(meta, indent=2) + "\n").encode()
    return out


def manifest(config: dict[str, Any], commit: str, files: dict[str, bytes], out: Path) -> str:
    """Return ``MANIFEST.md``: the commit, every file's sha256, and the gateway commands.

    Args:
        config: The run config.
        commit: The build's commit.
        files: Every file of the folder but the manifest.
        out: The folder (for the commands).

    Returns:
        The Markdown.
    """
    spec = config["kernel"]
    rows = [f"| `{name}` | {len(data):,} | `{_sha256(data)}` |" for name, data in
            sorted(files.items())]
    lines = [f"# {config['run_id']} push folder", "",
             f"- Commit: `{commit}`; kernel `{spec['id']}`; accelerator "
             f"{spec.get('accelerator') or 'none (CPU)'}; internet "
             f"{'on' if spec['enable_internet'] else 'off'}.", "",
             "| File | Bytes | sha256 |", "|---|---:|---|", *rows, "", "Gateway commands:", "",
             "```bash"]
    folder = out.as_posix()
    if spec.get("code_dataset"):
        lines.append(f"python scripts/kaggle_ops.py dataset-create {folder}/arc26-code-fn")
    if spec.get("accelerator"):
        lines.append(f"python scripts/kaggle_ops.py push {folder} --run-id {config['run_id']}")
    else:
        lines.append(f"python scripts/kaggle_ops.py push {folder} --timeout {spec['timeout_s']}")
    lines += [f"python scripts/kaggle_ops.py status {spec['id']}",
              f"python scripts/kaggle_ops.py pull {spec['id']} --task-id {config['run_id']}",
              "```", ""]
    return "\n".join(lines)


def build(config_path: Path, out: Path, repo: Path = REPO, allow_dirty: bool = False) -> Path:
    """Build one push folder from ``HEAD``.

    Args:
        config_path: The run config, relative to the repository (read from the commit).
        out: The folder to write (replaced).
        repo: The repository root.
        allow_dirty: Build from ``HEAD`` even when the working tree has changes (tests only;
            the inputs still come from the commit).

    Returns:
        ``out``.

    Raises:
        BuildError: If the tree is dirty, a file is missing from the commit, or a check fails;
            nothing is written then.
    """
    import yaml

    if not allow_dirty:
        submission.require_clean(repo)
    commit, _ = submission.head_commit(repo)
    rel = config_path.resolve().relative_to(repo.resolve()).as_posix() if config_path.is_absolute(
    ) else config_path.as_posix()
    with tempfile.TemporaryDirectory(prefix="fn-build-") as tmp:
        tree = Path(tmp) / "tree"
        tree.mkdir()
        submission.export_tree(repo, commit, tree)
        config = resolve_extends(lambda name: submission._read_export(tree, name, commit), rel)
        spec = config["kernel"]
        code_file = spec["code_file"]
        if code_file not in SOURCES:
            raise BuildError(f"kernel.code_file must be one of {sorted(SOURCES)}")
        source = submission._read_export(tree, SOURCES[code_file], commit).decode("utf-8")
        files: dict[str, bytes] = {}
        extra: list[str] = []
        config = dict(config)
        config["build"] = {"commit": commit, "config": rel}
        helper = config.get("server_help")
        if isinstance(helper, dict) and helper.get("flags_from"):
            probe = yaml.safe_load(submission._read_export(tree, helper["flags_from"], commit))
            helper["flags"] = server_flags(probe)
        if spec.get("code_dataset"):
            wheel_commit, epoch = submission.wheel_input_commit(repo, commit)
            wheel_dir = Path(tmp) / "wheel"
            wheel_dir.mkdir()
            wheel, builder = submission.build_wheel(tree, wheel_dir, epoch)
            dataset_files = {wheel.name: wheel.read_bytes()}
            for name in DATASET_SCRIPTS:
                dataset_files[Path(name).name] = submission._read_export(tree, name, commit)
            for name, folder in DATASET_ZIPS.items():
                dataset_files[name] = package_zip(tree, folder, commit)
            config["build"].update(
                wheel=wheel.name, wheel_commit=wheel_commit, wheel_builder=builder,
                files={n: _sha256(d) for n, d in sorted(dataset_files.items())},
            )
            for name, data in code_dataset(spec["code_dataset"], dataset_files).items():
                files[f"arc26-code-fn/{name}"] = data
            extra.append(spec["code_dataset"])
        meta = kernel_metadata(config, extra)
        if code_file.endswith(".ipynb"):
            files[code_file] = notebook(config)
        else:
            files[code_file] = embed(source, config).encode("utf-8")
        files["kernel-metadata.json"] = (json.dumps(meta, indent=2) + "\n").encode()
        files["MANIFEST.md"] = manifest(config, commit, files, out).encode("utf-8")
        submission.write_folder(out, files)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, required=True, help="configs/P5.2-0*.yaml")
    parser.add_argument("--out", type=Path, required=True, help="push folder, e.g. dist/P5.2-00")
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build one push folder.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when written, 2 when refused.
    """
    args = build_parser().parse_args(argv)
    try:
        folder = build(args.config, args.out, allow_dirty=args.allow_dirty)
    except BuildError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print((folder / "MANIFEST.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
