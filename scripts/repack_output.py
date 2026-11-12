r"""Repack an attached notebook output into tar parts with a manifest (task X2, X2-01 attempt 3).

Why: Kaggle serves a whole notebook output as one zip it builds while it streams (chunked, no
``Content-Length``, no ``Range``), and cut that stream after about 190 s on each of five tries
for ``sorokin/pip-install-unsloth-flash-patch`` version 2 (``X2-01`` attempts 2 and 3); a single
file of an output downloads with its length and at about four times the rate (``X1-01``: a
1.62 GB file in 19.7 s). This script, run in a CPU Kaggle notebook that attaches the output as
every build does, writes the mounted folder as one uncompressed tar split into parts of at most
``part_bytes``, and ``MANIFEST.json`` with every part's size and sha256 and every file's size,
sha256, and mode, so the Modal fetch stage can download the parts one by one, join and extract
them, and check the tree file for file against what Kaggle mounts (``modal_job.unpack_tar``).

Standard library only: the script notebook copies this file beside the wheel and runs it with
the image's Python. Usage::

    python scripts/repack_output.py --config configs/repack_kaggle_lib.yaml
    python scripts/repack_output.py --config configs/repack_qwen38_int4.yaml --source <mount>
    python scripts/repack_output.py --config configs/repack_kaggle_lib.yaml --smoke

``--source`` names the folder when only the session knows it: a notebook output attached as a
script build's ``notebook_inputs`` is found by search, and the build passes its folder as
``{input:<owner>/<slug>}`` (``X2-01`` attempt 5: Kaggle serves a single file without ``Range``
and cut the 18.6 GB ``model.safetensors`` at most 6.9 GB in, so the weights' output is repacked
too).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

MANIFEST = "MANIFEST.json"
PARTS = "parts"
KIND = "repack"
# The part size of the smoke: small, so a few kilobytes of fixture make several parts.
SMOKE_PART_BYTES = 4096


class PartWriter(io.RawIOBase):
    """A write-only stream that splits what it is given into numbered part files.

    Args:
        folder: Where the parts go.
        stem: Each part is ``<stem>.<NNN>``.
        part_bytes: Most bytes in one part.
    """

    def __init__(self, folder: Path, stem: str, part_bytes: int) -> None:
        super().__init__()
        self.folder, self.stem, self.part_bytes = folder, stem, part_bytes
        self.parts: list[dict[str, Any]] = []
        self._handle: BinaryIO | None = None
        self._digest = hashlib.sha256()
        self._size = 0

    def writable(self) -> bool:
        """Return True: this stream only writes."""
        return True

    def _open(self) -> None:
        path = self.folder / f"{self.stem}.{len(self.parts):03d}"
        self._handle = path.open("wb")
        self._digest, self._size = hashlib.sha256(), 0
        self.parts.append({"path": f"{PARTS}/{path.name}"})

    def _seal(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self.parts[-1].update(size=self._size, sha256=self._digest.hexdigest())
            self._handle = None

    def write(self, data: bytes) -> int:  # type: ignore[override]
        """Write ``data`` across parts; returns its length."""
        view = memoryview(data)
        while view:
            if self._handle is None or self._size == self.part_bytes:
                self._seal()
                self._open()
            take = min(len(view), self.part_bytes - self._size)
            assert self._handle is not None
            self._handle.write(view[:take])
            self._digest.update(view[:take])
            self._size += take
            view = view[take:]
        return len(data)

    def close(self) -> None:
        """Seal the last part."""
        self._seal()
        super().close()


def file_sha256(path: Path) -> str:
    """Return a file's sha256, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tree_record(source: Path) -> dict[str, dict[str, Any]]:
    """Return every entry under ``source`` by relative path: files, links, and empty folders.

    Args:
        source: The folder.

    Returns:
        ``{"type": "file", "size", "sha256", "exec"}``, ``{"type": "link", "target"}``, or
        ``{"type": "dir"}`` for a folder with no entries, by POSIX relative path.
    """
    tree: dict[str, dict[str, Any]] = {}
    for root, folders, files in os.walk(source, followlinks=False):
        folders.sort()
        base = Path(root)
        rel_root = base.relative_to(source)
        for name in list(folders):
            path = base / name
            if path.is_symlink():
                folders.remove(name)
                tree[(rel_root / name).as_posix()] = {"type": "link", "target": os.readlink(path)}
        if not folders and not files and base != source:
            tree[rel_root.as_posix()] = {"type": "dir"}
        for name in sorted(files):
            path = base / name
            rel = (rel_root / name).as_posix()
            if path.is_symlink():
                tree[rel] = {"type": "link", "target": os.readlink(path)}
                continue
            mode = path.stat().st_mode
            tree[rel] = {
                "type": "file",
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
                "exec": bool(mode & stat.S_IXUSR),
            }
    return tree


def repack(source: Path, out: Path, part_bytes: int, name: str) -> dict[str, Any]:
    """Write ``source`` as tar parts and ``MANIFEST.json`` under ``out``.

    Args:
        source: The mounted notebook output (read only).
        out: An empty folder in ``/kaggle/working``.
        part_bytes: Most bytes in one part.
        name: The parts' stem, ``<name>.tar.<NNN>``.

    Returns:
        The manifest: ``kind``, ``source``, ``tar`` (format, members, bytes), ``parts`` (path,
        size, sha256, in order), ``tree`` (``tree_record``), ``files``, ``bytes``, ``seconds``.

    Raises:
        ValueError: If ``source`` is not a folder, ``out`` is not empty, or ``part_bytes`` < 1.
    """
    if not source.is_dir():
        raise ValueError(f"{source} is not a folder")
    if part_bytes < 1:
        raise ValueError("part_bytes must be positive")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    started = time.monotonic()
    (out / PARTS).mkdir(parents=True, exist_ok=True)
    tree = tree_record(source)
    writer = PartWriter(out / PARTS, f"{name}.tar", part_bytes)
    members = 0
    # Uncompressed PAX: modes, long names, and links kept; the parts are mostly wheels' and
    # shared objects' bytes, which compression would barely shrink at a large CPU cost.
    with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        for rel in sorted(tree):
            info = archive.gettarinfo(str(source / rel), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if info.isfile():
                with (source / rel).open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
            members += 1
    writer.close()
    files = [entry for entry in tree.values() if entry["type"] == "file"]
    manifest = {
        "kind": KIND,
        "source": str(source),
        "tar": {"format": "pax", "members": members,
                "bytes": sum(p["size"] for p in writer.parts)},  # fmt: skip
        "parts": writer.parts,
        "tree": tree,
        "files": len(files),
        "bytes": sum(entry["size"] for entry in files),
        "seconds": round(time.monotonic() - started, 1),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def read_config(path: Path) -> dict[str, Any]:
    """Read the script's config: ``source``, ``out``, ``part_bytes``, ``name`` (simple YAML).

    Why: the image's Python has no guaranteed PyYAML, so the config is flat ``key: value``
    lines with ``#`` comments, which this reads itself. ``source`` may be left out when the
    command line gives ``--source``.

    Raises:
        ValueError: If a key is missing or unknown, or a line is not ``key: value``.
    """
    keys = {"source", "out", "part_bytes", "name", "kind"}
    data: dict[str, Any] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        key, sep, value = text.partition(":")
        if not sep or key.strip() not in keys:
            raise ValueError(f"{path}:{number}: expected one of {sorted(keys)} as key: value")
        data[key.strip()] = value.strip().strip('"')
    missing = {"out", "part_bytes", "name"} - set(data)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    data["part_bytes"] = int(data["part_bytes"])
    return data


def smoke() -> int:
    """Repack a small generated folder (a link, an executable, an empty folder) and check it."""
    with tempfile.TemporaryDirectory(prefix="repack-smoke-") as work:
        source = Path(work) / "src"
        (source / "pkg" / "bin").mkdir(parents=True)
        (source / "empty").mkdir()
        (source / "pkg" / "a.py").write_text("x = 1\n" * 2000)
        tool = source / "pkg" / "bin" / "tool"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
        os.symlink("a.py", source / "pkg" / "alias.py")
        manifest = repack(source, Path(work) / "out", SMOKE_PART_BYTES, "smoke")
        ok = len(manifest["parts"]) > 1 and manifest["tree"]["pkg/bin/tool"]["exec"]
        ok = ok and manifest["tree"]["pkg/alias.py"] == {"type": "link", "target": "a.py"}
        ok = ok and manifest["tree"]["empty"] == {"type": "dir"}
        print(json.dumps({"ok": ok, "parts": len(manifest["parts"]), "files": manifest["files"]}))
        return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Repack the configured folder; with ``--smoke`` run the self-check instead."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="the folder to repack, over the config's")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        return smoke()
    config = read_config(args.config)
    source = args.source or config.get("source")
    if not source:
        raise ValueError(f"{args.config}: no source, and no --source")
    manifest = repack(Path(source), Path(config["out"]), config["part_bytes"],
                      config["name"])  # fmt: skip
    summary = {k: manifest[k] for k in ("files", "bytes", "seconds")}
    summary["parts"] = len(manifest["parts"])
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
