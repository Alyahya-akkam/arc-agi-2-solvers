"""Build the push folders of the K0 environment probe (P0.5a, BRIEF A5, ADR-008).

Writes ``<out>/env_probe.py``, a copy of ``scripts/env_probe.py``, and
``<out>/kernel-metadata.json`` for the private script kernel
``<owner>/arc26-k0-env-probe-<variant>``: internet off, the competition and the base model
attached, and optionally the install notebook at a fixed version.
No accelerator field is written, so ``kaggle_ops.py push`` sends it as a CPU push, which needs no
pre-run brief. Each variant has its own slug, and no pre-run brief may name either one, because
Kaggle may keep a slug's previous accelerator (the gateway's CPU slug rule).

- ``latest``: ``docker_image_pinning_type`` ``latest``, the image a new notebook gets today.
- ``pinned``: ``original`` with ``--docker-image``, the image the reference notebook's pulled
  metadata names (NVARC's Python 3.11 stack).

Usage::

    python scripts/build_env_probe.py --owner <username> --variant latest --out dist/K0/latest
    python scripts/build_env_probe.py --owner <username> --variant pinned --out dist/K0/pinned
        --docker-image <image> [--kernel-source <owner>/<slug>/<version>]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parent / "env_probe.py"
SLUG_PREFIX = "arc26-k0-env-probe"
VARIANTS = ("latest", "pinned")
PINNING = {"latest": "latest", "pinned": "original"}
COMPETITION = "arc-prize-2026-arc-agi-2"
# The reference notebook's model handle (docs/research/kaggle_first_runs.md, section 1.3).
MODEL = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
# The gateway's rule for ids: letters, digits, ".", "_", "-", and never "." or "..".
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")
VERSION = re.compile(r"[0-9]+")


class BuildError(Exception):
    """The requested push folder would break a K0 or ADR-008 rule; nothing was written."""


def _safe_id(value: str) -> bool:
    return bool(SAFE_ID.fullmatch(value)) and value not in {".", ".."}


def _versioned_source(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 3 and all(map(_safe_id, parts[:2])) and bool(VERSION.fullmatch(parts[2]))


def kernel_metadata(
    owner: str, variant: str, docker_image: str | None = None, kernel_source: str | None = None
) -> dict[str, object]:
    """Return the ``kernel-metadata.json`` of one probe variant.

    Why: every field the gateway checks is explicit (private, internet off, image pinning, owner),
    and no ``enable_gpu``, ``enable_tpu``, or ``machine_shape`` appears, so the push can only run
    on CPU. The title slugifies to the id's slug, as kaggle-cli expects.

    Args:
        owner: Kaggle username, which must equal ``KAGGLE_OWNER`` in ``kaggle_ops.py`` at push time.
        variant: ``latest`` or ``pinned``.
        docker_image: The image to pin; required for ``pinned`` and refused for ``latest``.
        kernel_source: The install notebook as ``<owner>/<slug>/<version>``, or None for none.

    Returns:
        The metadata as a dict, in the key order it is written.

    Raises:
        BuildError: If the owner is not a plain id, the variant is unknown, ``pinned`` lacks a
            docker image or ``latest`` has one, the image contains whitespace, or the kernel
            source is not ``<owner>/<slug>/<version>`` with an integer version.
    """
    if not _safe_id(owner):
        raise BuildError(f"--owner must be a Kaggle username; got {owner!r}")
    if variant not in VARIANTS:
        raise BuildError(f"--variant must be one of {', '.join(VARIANTS)}; got {variant!r}")
    if variant == "pinned" and not docker_image:
        raise BuildError(
            "the pinned variant needs --docker-image, the docker_image of the reference "
            "notebook's pulled metadata"
        )
    if variant == "latest" and docker_image is not None:
        raise BuildError("--docker-image contradicts the latest variant (the newest image)")
    if docker_image is not None and len(docker_image.split()) != 1:
        raise BuildError(f"--docker-image must be one image name; got {docker_image!r}")
    if kernel_source is not None and not _versioned_source(kernel_source):
        raise BuildError(
            f"--kernel-source must be <owner>/<slug>/<version> with an integer version; "
            f"got {kernel_source!r}"
        )
    slug = f"{SLUG_PREFIX}-{variant}"
    meta: dict[str, object] = {
        "id": f"{owner}/{slug}",
        "title": f"arc26 K0 env probe {variant}",
        "code_file": PROBE.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": False,
        "docker_image_pinning_type": PINNING[variant],
    }
    if variant == "pinned":
        meta["docker_image"] = docker_image
    meta.update(
        {
            "competition_sources": [COMPETITION],
            "model_sources": [MODEL],
            "kernel_sources": [kernel_source] if kernel_source else [],
            "dataset_sources": [],
        }
    )
    return meta


def build(
    out: Path,
    owner: str,
    variant: str,
    docker_image: str | None = None,
    kernel_source: str | None = None,
) -> Path:
    """Write one push folder: the probe and its metadata, and nothing else.

    Args:
        out: Folder to write, created if missing; other files in it are left alone.
        owner: Kaggle username.
        variant: ``latest`` or ``pinned``.
        docker_image: The image to pin, for ``pinned`` only.
        kernel_source: The install notebook as ``<owner>/<slug>/<version>``, or None.

    Returns:
        ``out``.

    Raises:
        BuildError: If ``kernel_metadata`` refuses the arguments; nothing is written then.
    """
    meta = kernel_metadata(owner, variant, docker_image, kernel_source)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROBE, out / PROBE.name)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--owner", required=True, help="Kaggle username (KAGGLE_OWNER)")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--out", required=True, help="push folder to write, e.g. dist/K0/latest")
    parser.add_argument("--docker-image", help="image to pin (pinned variant only)")
    parser.add_argument("--kernel-source", help="install notebook as <owner>/<slug>/<version>")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build one push folder.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the folder is written, 2 when the arguments are refused.
    """
    args = build_parser().parse_args(argv)
    try:
        folder = build(
            Path(args.out), args.owner, args.variant, args.docker_image, args.kernel_source
        )
    except BuildError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print(f"wrote {folder}; push: python scripts/kaggle_ops.py push {folder} --timeout 1800")
    return 0


if __name__ == "__main__":
    sys.exit(main())
