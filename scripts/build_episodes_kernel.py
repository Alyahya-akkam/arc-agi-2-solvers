"""Build the two push folders of P3.4: the episode builder and the leakage checker (ADR-008).

- ``builder``: ``<owner>/arc26-p3-4-episodes``, a CPU script kernel with internet on (allowed for
  CPU pushes), no competition, no model, no kernel source (gate G0). It runs
  ``build_episodes.py build`` and leaves the episodes in ``/kaggle/working``.
- ``checker``: ``<owner>/arc26-p3-4-leakage``, a CPU script kernel with internet off, the
  competition attached, and the builder's output attached at a pinned version through
  ``kernel_sources``. It runs ``build_episodes.py check`` and writes ``run_logs/leakage.*``.

No accelerator field is written, so ``kaggle_ops.py push`` sends each as a CPU push, which needs
no pre-run brief; neither slug may be the ``Kaggle notebook`` of a brief (the gateway's CPU slug
rule). A script kernel has one code file, so the folder's script carries its modules as text:
``arc26/hyper/episodes.py``, ``scripts/build_episodes.py``, and ``configs/episodes.yaml`` as
JSON, with stub package files. At run time it writes them under ``/tmp/arc26-p3-4`` and calls
``build_episodes.main``. The commit of those files is embedded; a tree whose bundled files differ
from ``HEAD`` is refused unless ``--allow-dirty``, which marks the commit ``+dirty``.

Usage::

    python scripts/build_episodes_kernel.py --owner alyahya737 --kernel builder \
        --out dist/P3.4/builder
    python scripts/build_episodes_kernel.py --owner alyahya737 --kernel checker \
        --builder-version 1 --out dist/P3.4/checker

Then ``python scripts/kaggle_ops.py push <folder> --timeout <seconds>`` (the runbook gives both).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "episodes.yaml"
# Source files bundled into each push folder, by their path inside the bundle.
BUNDLED = {
    "arc26/hyper/episodes.py": REPO / "arc26" / "hyper" / "episodes.py",
    "build_episodes.py": REPO / "scripts" / "build_episodes.py",
}
# The bundle's packages are stubs, so a later package-level import (torch, say) cannot break a
# CPU kernel that needs numpy only.
STUB = '"""Bundled subset of arc26 for the P3.4 kernels (scripts/build_episodes_kernel.py)."""\n'
COMPETITION = "arc-prize-2026-arc-agi-2"
KERNELS = {
    "builder": {
        "slug": "arc26-p3-4-episodes",
        "title": "arc26 P3.4 episodes",
        "code_file": "arc26_p3_4_episodes.py",
        "internet": True,
        "argv": [
            "build", "--config", "{root}/episodes.json", "--out", "/kaggle/working",
            "--work", "/tmp/arc26-p3-4-sources",
        ],
    },
    "checker": {
        "slug": "arc26-p3-4-leakage",
        "title": "arc26 P3.4 leakage",
        "code_file": "arc26_p3_4_leakage.py",
        "internet": False,
        "argv": [
            "check", "--config", "{root}/episodes.json", "--episodes", "/kaggle/input",
            "--competition", "/kaggle/input", "--out", "/kaggle/working",
        ],
    },
}  # fmt: skip
BUNDLE_ROOT = "/tmp/arc26-p3-4"
# The gateway's rule for ids: letters, digits, ".", "_", "-", and never "." or "..".
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")

TEMPLATE = '''"""arc26 P3.4 {kernel} kernel from scripts/build_episodes_kernel.py; do not edit.

Code commit {commit}.
Kaggle runs this file as a script kernel. It writes the bundled modules under {root},
puts that folder first on sys.path, and runs build_episodes.main with ARGV. Outputs go
to /kaggle/working; kaggle_ops.py pull brings back run_logs/ only.
"""

import sys
from pathlib import Path

COMMIT = {commit!r}
ARGV = {argv!r}
FILES = {files!r}


def run(argv, root):
    """Write the bundle under root and run build_episodes.main; return its exit code."""
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sys.path.insert(0, str(root))
    import build_episodes

    return build_episodes.main([arg.replace("{{root}}", str(root)) for arg in argv])


if __name__ == "__main__":
    sys.exit(run([*ARGV, "--code-commit", COMMIT], Path({root!r})))
'''


class BuildError(Exception):
    """The requested push folder would break a P3.4 or ADR-008 rule; nothing was written."""


def _safe_id(value: str) -> bool:
    return bool(SAFE_ID.fullmatch(value)) and value not in {".", ".."}


def kernel_metadata(
    owner: str, kernel: str, builder_version: int | None = None
) -> dict[str, object]:
    """Return the ``kernel-metadata.json`` of one kernel.

    Why: every field the gateway checks is explicit (private, internet, image pinning, owner), and
    no ``enable_gpu``, ``enable_tpu``, or ``machine_shape`` appears, so the push runs on CPU. The
    title slugifies to the slug, as kaggle-cli expects.

    Args:
        owner: Kaggle username; must equal ``KAGGLE_OWNER`` in ``kaggle_ops.py`` at push time.
        kernel: ``builder`` or ``checker``.
        builder_version: The builder notebook's version whose output the checker attaches;
            required for ``checker``, refused for ``builder``.

    Returns:
        The metadata, in the key order it is written.

    Raises:
        BuildError: If the owner is not a plain id, the kernel is unknown, or the builder version
            is missing, not a positive integer, or given for the builder.
    """
    if not _safe_id(owner):
        raise BuildError(f"--owner must be a Kaggle username; got {owner!r}")
    if kernel not in KERNELS:
        raise BuildError(f"--kernel must be one of {', '.join(KERNELS)}; got {kernel!r}")
    if kernel == "checker":
        if isinstance(builder_version, bool) or not isinstance(builder_version, int):
            raise BuildError("the checker needs --builder-version, the builder run to check")
        if builder_version < 1:
            raise BuildError(f"--builder-version must be at least 1; got {builder_version}")
    elif builder_version is not None:
        raise BuildError("--builder-version applies to the checker only")
    spec = KERNELS[kernel]
    builder = f"{owner}/{KERNELS['builder']['slug']}"
    return {
        "id": f"{owner}/{spec['slug']}",
        "title": spec["title"],
        "code_file": spec["code_file"],
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": spec["internet"],
        # The newest CPU image; the manifest records the image each run used (note 3.1).
        "docker_image_pinning_type": "latest",
        "competition_sources": [COMPETITION] if kernel == "checker" else [],
        "model_sources": [],
        "kernel_sources": [f"{builder}/{builder_version}"] if kernel == "checker" else [],
        "dataset_sources": [],
    }


def _build_episodes() -> ModuleType:
    # scripts/ is not a package; register the module before running it, as dataclasses with
    # postponed annotations look their module up in sys.modules.
    name = "build_episodes"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BUNDLED["build_episodes.py"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def config_json(path: Path = CONFIG) -> str:
    """Return the YAML config as JSON text, after validating it as the build would.

    Args:
        path: The YAML config.

    Returns:
        Indented JSON of the whole file, ``smoke`` section included.

    Raises:
        BuildError: If the config is not valid.
    """
    import yaml

    build_episodes = _build_episodes()
    try:
        build_episodes.read_config(path)
    except build_episodes.ConfigError as err:
        raise BuildError(str(err)) from err
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.dumps(raw, indent=1, sort_keys=True) + "\n"


def bundle_files(config: Path = CONFIG) -> dict[str, str]:
    """Return the files a kernel writes at run time, by path inside the bundle.

    Args:
        config: The YAML config to carry as ``episodes.json``.

    Returns:
        Stub ``arc26/__init__.py`` and ``arc26/hyper/__init__.py``, the bundled sources, and
        ``episodes.json``.
    """
    files = {"arc26/__init__.py": STUB, "arc26/hyper/__init__.py": STUB}
    files |= {rel: path.read_text(encoding="utf-8") for rel, path in BUNDLED.items()}
    files["episodes.json"] = config_json(config)
    return files


def code_commit(allow_dirty: bool = False, config: Path = CONFIG) -> str:
    """Return ``HEAD``'s commit, refusing bundled files that differ from it.

    Args:
        allow_dirty: Accept local changes and mark the commit ``+dirty``.
        config: The config being bundled.

    Returns:
        The 40-hex commit, with ``+dirty`` when changes were allowed and present.

    Raises:
        BuildError: If git fails, or a bundled file or the config has uncommitted changes and
            ``allow_dirty`` is false.
    """
    paths = [str(p) for p in (*BUNDLED.values(), config)]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        raise BuildError(f"cannot read the code commit from git: {err}") from err
    if status and not allow_dirty:
        raise BuildError(
            "bundled files differ from HEAD; commit them first, or pass --allow-dirty:\n" + status
        )
    return head + "+dirty" if status else head


def render(kernel: str, commit: str, files: dict[str, str]) -> str:
    """Return the code file of a kernel.

    Args:
        kernel: ``builder`` or ``checker``.
        commit: The embedded code commit.
        files: ``bundle_files()``.

    Returns:
        Python source that writes ``files`` under ``BUNDLE_ROOT`` and runs the kernel's mode.
    """
    argv = KERNELS[kernel]["argv"]
    return TEMPLATE.format(kernel=kernel, commit=commit, root=BUNDLE_ROOT, argv=argv, files=files)


def build(
    out: Path,
    owner: str,
    kernel: str,
    builder_version: int | None = None,
    commit: str | None = None,
    config: Path = CONFIG,
) -> Path:
    """Write one push folder: the kernel's code file and its metadata, nothing else.

    Args:
        out: Folder to write, created if missing; other files in it are left alone.
        owner: Kaggle username.
        kernel: ``builder`` or ``checker``.
        builder_version: The builder run the checker attaches (checker only).
        commit: Code commit to embed; None reads it with ``code_commit()``.
        config: The YAML config to bundle.

    Returns:
        ``out``.

    Raises:
        BuildError: If ``kernel_metadata`` or ``code_commit`` refuses, or the config is invalid;
            nothing is written then.
    """
    meta = kernel_metadata(owner, kernel, builder_version)
    commit = code_commit(config=config) if commit is None else commit
    code = render(kernel, commit, bundle_files(config))
    out.mkdir(parents=True, exist_ok=True)
    (out / KERNELS[kernel]["code_file"]).write_text(code, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--owner", required=True, help="Kaggle username (KAGGLE_OWNER)")
    parser.add_argument("--kernel", required=True, choices=sorted(KERNELS))
    parser.add_argument("--out", required=True, type=Path, help="push folder, e.g. dist/P3.4/x")
    parser.add_argument("--builder-version", type=int, help="builder run the checker attaches")
    parser.add_argument("--allow-dirty", action="store_true", help="bundle uncommitted changes")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build one push folder.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the folder is written, 2 when the arguments or the tree are refused.
    """
    args = build_parser().parse_args(argv)
    try:
        kernel_metadata(args.owner, args.kernel, args.builder_version)
        commit = code_commit(args.allow_dirty)
        folder = build(args.out, args.owner, args.kernel, args.builder_version, commit)
    except BuildError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    timeout = 7200 if args.kernel == "builder" else 3600
    print(f"wrote {folder}; push: python scripts/kaggle_ops.py push {folder} --timeout {timeout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
