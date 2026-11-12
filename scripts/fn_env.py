r"""Flash-Next environment check on a Kaggle CPU session (task P5.2, runs ``P5.2-00``).

Purpose: before any GPU quota is spent on Qwen3.8-Flash-Next served by SGLang on 4 x L4 (run
``P5.2-01``, ``scripts/fn_probe.py``), find out on a free CPU session whether the public stack
installs and whether its compiled kernels carry code the L4 (sm89) can run. Task P5.2
(``docs/tasks/P5.2.md``); runbook ``docs/runbooks/fn_probe.md``.

Inputs (Kaggle mounts, found by listing ``/kaggle/input`` first): the SGLang wheelhouse
``dfranzen/pennyroyal-v253`` (cp312, CUDA 13), the model
``dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple``, its MTP drafter, and the competition.
Outputs: ``/kaggle/working/run_logs/<report_dir>/`` (``report.json`` after every step, the
listing, versions, the per-file CUDA architectures, the model's configs and sizes, the chat
template, ``summary.md``), which ``kaggle_ops.py pull`` brings back, and
``/kaggle/working/wheels/`` plus ``python312.tar.gz`` when a step produced them, which the GPU
probe attaches as a kernel source.

Method. Each configured step runs on its own; a failure is recorded with its error and the next
step still runs (a step whose input is missing records why it was skipped). The steps:

1. ``listing``: the input tree to a bounded depth, so the mount paths are read, not assumed.
2. ``python``: ``sys.version``; when it is not the wanted 3.12, a python-build-standalone 3.12
   through ``uv python install`` (internet), kept in the output as a tarball.
3. ``venv``: dfranzen's launcher: uv from the wheelhouse's uv wheel, ``uv venv``, then
   ``uv pip install --no-index --find-links`` of the lock and the SGLang wheel ``--no-deps``.
4. ``imports``: each module imported in the venv in its own process (no CUDA here: errors are
   recorded, not fatal).
5. ``arch``: every ``.so``, ``.cubin``, and ``.fatbin`` of the listed packages, read for the CUDA
   architectures they carry (cubins by ``sm``, PTX by ``compute``), with ``cuobjdump`` as a
   cross-check when the wheels bundle it. A cubin for ``sm_XY`` runs on ``sm_89`` when
   :math:`X = 8` and :math:`Y \le 9` (binary compatibility within a major version); PTX for
   ``compute_Z`` runs after a JIT when :math:`Z \le 89`.
6. ``model``: the model's and the drafter's small JSON files, every file's size, and the bytes of
   each component (experts, n-gram table, the rest) summed from the safetensors headers alone.
7. ``template``: the chat template saved, and rendered by the venv's tokenizer at the configured
   effort, so the probe's own rendering can be checked against it.
8. ``fetch``: wheels the recipe says are missing, downloaded by URL or ``pip download``, each
   install-tested in the venv.
9. ``fa2`` (the separate FA2 notebook only): FlashAttention 2 built from its sdist for the
   configured architectures.
10. ``package``: the produced wheels' manifest, the Python tarball, and ``summary.md``.

Safety: standard library only (the image's Python runs it); no file whose name contains
``solution`` is ever opened; no secret is read or printed.

Usage::

    python fn_env.py                                   # on Kaggle, config embedded by the builder
    python scripts/fn_env.py --config configs/P5.2-00.yaml --root /tmp/tree
    python scripts/fn_env.py --config configs/P5.2-00.yaml --smoke --root <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The builder replaces this line with the run's config as JSON (scripts/build_fn_notebooks.py).
EMBEDDED_CONFIG: dict[str, Any] | None = None

SCHEMA = 1
REPORT_FILE = "report.json"
SUMMARY_FILE = "summary.md"
OUTPUT_MARKER = "fn_env_output.json"
# A fat binary's header (cuda fatbinary.h): magic, version 1, header size 16, payload size.
FATBIN_MAGIC = b"\x50\xed\x55\xba"
FATBIN_HEADER = struct.Struct("<IHHQ")
# One entry's header: kind (1 PTX, 2 ELF), version, header size, payload size, compressed size,
# a word, minor, major, arch, name offset, name length, flags, a word, decompressed size.
FATBIN_ENTRY = struct.Struct("<HHIQIIHHIIIQQQ")
FATBIN_KINDS = {1: "ptx", 2: "elf"}
ELF_MAGIC = b"\x7fELF"
EM_CUDA = 190
# A cuda ELF with ABI version 8 or more (CUDA 12.8 and later) keeps sm in e_flags bits 8 to 15;
# older ones in bits 0 to 7 (LLVM's ELF.h, EF_CUDA_SM and its V2 layout).
CUDA_ABI_V2 = 8
TARGET_SM = 89
DEVICE_SUFFIXES = (".so", ".cubin", ".fatbin")
TAIL_CHARS = 4000


class StepSkipped(Exception):
    """A step could not run because an earlier step did not give it what it needs."""


def _now() -> float:
    return time.time()


def _tail(text: str | bytes | None, limit: int | None = TAIL_CHARS) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text if limit is None else text[-limit:]


def _describe(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def read_bytes(path: Path, limit: int | None = None) -> bytes:
    """Read a file, refusing any file whose name contains ``solution``.

    Why: the evaluation solutions sit beside the challenges on the competition mount, and this
    script never opens them (CLAUDE.md, competition mode rules).

    Args:
        path: File to read.
        limit: Bytes to read from the start, or None for the whole file.

    Returns:
        The bytes read.

    Raises:
        PermissionError: If the name contains ``solution``.
        OSError: If the file cannot be read.
    """
    if "solution" in path.name.lower():
        raise PermissionError(f"refusing to read {path.name}: solutions are never opened")
    with open(path, "rb") as handle:
        return handle.read() if limit is None else handle.read(limit)


def read_json(path: Path) -> Any:
    """Read a JSON file through :func:`read_bytes`.

    Args:
        path: The file.

    Returns:
        The parsed value.
    """
    return json.loads(read_bytes(path).decode("utf-8"))


def sha256_file(path: Path) -> str:
    """Return the sha256 of a file, read in 8 MiB blocks.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    if "solution" in path.name.lower():
        raise PermissionError(f"refusing to read {path.name}: solutions are never opened")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- CUDA architectures in compiled files --------------------------------------------------


def parse_fatbin(data: bytes | mmap.mmap, start: int) -> tuple[list[tuple[str, int]], int]:
    """Parse one fat binary at ``start`` and return its entries and its end offset.

    Why: ``cuobjdump`` is not always bundled, and the header of every entry names its kind and
    architecture without decompressing anything.

    Args:
        data: The file's bytes (or a memory map of them).
        start: Offset of the fat binary's magic.

    Returns:
        ``[(kind, arch), ...]`` with kind ``ptx``, ``elf``, or ``kind<N>``, and the offset just
        past the fat binary; an empty list and ``start + 4`` when the header is not sane.
    """
    end_of_data = len(data)
    if start + FATBIN_HEADER.size > end_of_data:
        return [], start + 4
    magic, version, header_size, size = FATBIN_HEADER.unpack_from(data, start)
    if version != 1 or header_size != FATBIN_HEADER.size or size == 0:
        return [], start + 4
    end = start + header_size + size
    if end > end_of_data:
        return [], start + 4
    entries: list[tuple[str, int]] = []
    offset = start + header_size
    while offset + FATBIN_ENTRY.size <= end:
        fields = FATBIN_ENTRY.unpack_from(data, offset)
        kind, entry_header, payload, arch = fields[0], fields[2], fields[3], fields[8]
        if entry_header < FATBIN_ENTRY.size or entry_header > 4096 or not 10 <= arch <= 300:
            break
        entries.append((FATBIN_KINDS.get(kind, f"kind{kind}"), arch))
        offset += entry_header + payload
    if not entries:
        return [], start + 4
    return entries, end


def cubin_arch(head: bytes) -> int | None:
    """Return the sm of a CUDA ELF (a standalone cubin) from its header, or None.

    Args:
        head: At least the first 64 bytes of the file.

    Returns:
        The architecture number (89 for ``sm_89``), or None when the bytes are not a 64-bit
        little-endian CUDA ELF.
    """
    if len(head) < 64 or not head.startswith(ELF_MAGIC) or head[4] != 2 or head[5] != 1:
        return None
    (machine,) = struct.unpack_from("<H", head, 18)
    if machine != EM_CUDA:
        return None
    (flags,) = struct.unpack_from("<I", head, 48)
    return (flags >> 8) & 0xFF if head[8] >= CUDA_ABI_V2 else flags & 0xFF


def scan_device_code(path: Path) -> dict[str, Any]:
    """Return the CUDA architectures a compiled file carries.

    Args:
        path: A shared library, cubin, or fatbin.

    Returns:
        ``{"elf": [...], "ptx": [...], "other": [...], "fatbins": n}``: sorted distinct sm
        numbers of the cubins, compute numbers of the PTX, other entry kinds, and the count of
        fat binaries found.
    """
    elf: set[int] = set()
    ptx: set[int] = set()
    other: set[str] = set()
    fatbins = 0
    size = path.stat().st_size
    if size == 0:
        return {"elf": [], "ptx": [], "other": [], "fatbins": 0}
    with open(path, "rb") as handle:
        head = handle.read(64)
        arch = cubin_arch(head)
        if arch is not None:
            elf.add(arch)
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            position = data.find(FATBIN_MAGIC)
            while position != -1:
                entries, after = parse_fatbin(data, position)
                if entries:
                    fatbins += 1
                for kind, number in entries:
                    if kind == "elf":
                        elf.add(number)
                    elif kind == "ptx":
                        ptx.add(number)
                    else:
                        other.add(f"{kind}:{number}")
                position = data.find(FATBIN_MAGIC, after)
    return {"elf": sorted(elf), "ptx": sorted(ptx), "other": sorted(other), "fatbins": fatbins}


def runs_on(scan: Mapping[str, Sequence[int]], target: int = TARGET_SM) -> str | None:
    r"""Return how a file's device code reaches ``sm_<target>``, if at all.

    A cubin for ``sm_XY`` runs on ``sm_target`` when its major equals the target's and
    :math:`Y` is at most the target's minor; PTX for ``compute_Z`` runs after a JIT when
    :math:`Z \le` target.

    Args:
        scan: :func:`scan_device_code`'s result.
        target: The architecture number, 89 for the L4.

    Returns:
        ``"native"`` (a compatible cubin), ``"ptx"`` (only through a JIT), or None.
    """
    major, minor = divmod(target, 10)
    if any(a // 10 == major and a % 10 <= minor for a in scan.get("elf", [])):
        return "native"
    if any(a <= target for a in scan.get("ptx", [])):
        return "ptx"
    return None


def parse_cuobjdump(text: str) -> list[int]:
    """Return the distinct architecture numbers named in ``cuobjdump --list-*`` output.

    Args:
        text: The tool's output, lines such as ``ELF file 1: x.sm_89.cubin`` or
            ``PTX file 1: x.sm_80.ptx``.

    Returns:
        Sorted distinct numbers.
    """
    return sorted({int(n) for n in re.findall(r"(?:sm|compute)_(\d+)[a-z]?", text)})


def device_files(root: Path, limit: int) -> list[Path]:
    """List a package folder's compiled files, largest first, at most ``limit``.

    Args:
        root: The package folder in site-packages.
        limit: The most files returned.

    Returns:
        ``.so``, ``.cubin``, and ``.fatbin`` files (also ``.so.N``), by size descending, then
        path.
    """
    found = []
    for path in root.rglob("*"):
        name = path.name
        if path.is_file() and (name.endswith(DEVICE_SUFFIXES) or ".so." in name):
            found.append(path)
    found.sort(key=lambda p: (-p.stat().st_size, str(p)))
    return found[:limit]


def summarize_package(files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the per-file scans of one package against sm89.

    Args:
        files: Per file: ``{"path", "elf", "ptx", "runs_on"}``.

    Returns:
        Counts, the union of architectures, the files without an sm89 path, and a verdict:
        ``native`` (every file with device code has a compatible cubin), ``ptx`` (some only by
        JIT), ``partial`` (some have no path), ``none`` (device code, none for sm89), or
        ``no device code`` (a JIT package such as triton).
    """
    with_code = [f for f in files if f["elf"] or f["ptx"]]
    elf = sorted({a for f in with_code for a in f["elf"]})
    ptx = sorted({a for f in with_code for a in f["ptx"]})
    missing = [f["path"] for f in with_code if f["runs_on"] is None]
    by_ptx = [f["path"] for f in with_code if f["runs_on"] == "ptx"]
    if not with_code:
        verdict = "no device code"
    elif len(missing) == len(with_code):
        verdict = "none"
    elif missing:
        verdict = "partial"
    elif by_ptx:
        verdict = "ptx"
    else:
        verdict = "native"
    return {
        "files": len(files),
        "with_device_code": len(with_code),
        "elf_archs": elf,
        "ptx_archs": ptx,
        "sm89_native": sum(1 for f in with_code if f["runs_on"] == "native"),
        "sm89_ptx_only": len(by_ptx),
        "sm89_missing": missing,
        "verdict": verdict,
    }


# ---- safetensors headers and model components ------------------------------------------------

DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2,
    "U16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1, "F4": 1,
}


def safetensors_header(path: Path, max_header: int = 256 * 1024 * 1024) -> dict[str, Any]:
    """Read a safetensors file's JSON header only (8-byte length, then the JSON).

    Args:
        path: The shard.
        max_header: Largest header accepted, in bytes.

    Returns:
        Tensor name to ``{"dtype", "shape", "data_offsets"}`` (``__metadata__`` removed).

    Raises:
        ValueError: If the length is not sane.
    """
    head = read_bytes(path, 8)
    (length,) = struct.unpack("<Q", head)
    if not 2 <= length <= max_header:
        raise ValueError(f"{path.name}: header length {length} is not sane")
    with open(path, "rb") as handle:
        handle.seek(8)
        header = json.loads(handle.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def tensor_bytes(info: Mapping[str, Any]) -> int:
    """Return a tensor's byte size from its header entry (the data offsets when present).

    Args:
        info: ``{"dtype", "shape", "data_offsets"}``.

    Returns:
        Bytes.
    """
    offsets = info.get("data_offsets")
    if isinstance(offsets, list) and len(offsets) == 2:
        return int(offsets[1]) - int(offsets[0])
    count = 1
    for dim in info.get("shape", []):
        count *= int(dim)
    return count * DTYPE_BYTES.get(str(info.get("dtype")), 1)


def component_of(name: str, rules: Sequence[tuple[str, re.Pattern[str]]]) -> str:
    """Return the first component whose pattern matches a tensor name, or ``other``.

    Args:
        name: Tensor name.
        rules: ``(component, compiled regex)`` in priority order.

    Returns:
        The component.
    """
    for component, pattern in rules:
        if pattern.search(name):
            return component
    return "other"


def model_sizes(root: Path, rules: Sequence[tuple[str, str]], max_shards: int) -> dict[str, Any]:
    """Sum a checkpoint's tensor bytes by component from its safetensors headers.

    Args:
        root: The checkpoint folder.
        rules: ``(component, regex)`` pairs in priority order.
        max_shards: The most shards read.

    Returns:
        Bytes and tensor counts by component, by dtype, the shard count, and errors per shard.
    """
    compiled = [(name, re.compile(pattern)) for name, pattern in rules]
    shards = sorted(root.rglob("*.safetensors"))[:max_shards]
    by_component: dict[str, int] = {}
    counts: dict[str, int] = {}
    by_dtype: dict[str, int] = {}
    errors = {}
    for shard in shards:
        try:
            header = safetensors_header(shard)
        except (OSError, ValueError) as err:
            errors[str(shard.relative_to(root))] = _describe(err)
            continue
        for name, info in header.items():
            size = tensor_bytes(info)
            component = component_of(name, compiled)
            by_component[component] = by_component.get(component, 0) + size
            counts[component] = counts.get(component, 0) + 1
            dtype = str(info.get("dtype"))
            by_dtype[dtype] = by_dtype.get(dtype, 0) + size
    return {
        "shards_read": len(shards),
        "bytes_by_component": dict(sorted(by_component.items())),
        "tensors_by_component": dict(sorted(counts.items())),
        "bytes_by_dtype": dict(sorted(by_dtype.items())),
        "gib_by_component": {k: round(v / 1024**3, 3) for k, v in sorted(by_component.items())},
        "errors": errors,
    }


# ---- the run context -------------------------------------------------------------------------


@dataclass
class Context:
    """State shared by the steps of one run.

    Attributes:
        config: The run's config.
        root: Prefix for every absolute path (``/`` on Kaggle, a temporary tree in tests).
        report: The report being written.
        out: The report folder.
        python: The 3.12 interpreter the venv is built on, once found.
        standalone: The fetched python-build-standalone interpreter, if any (kept as a tarball).
        uv: The uv binary, once extracted.
        venv_python: The venv's interpreter, once built.
        inputs: Input name to its folder, once found.
        runner: Runs a command; replaced in tests.
    """

    config: Mapping[str, Any]
    root: Path
    report: dict[str, Any]
    out: Path
    python: str | None = None
    standalone: str | None = None
    uv: str | None = None
    venv_python: str | None = None
    inputs: dict[str, Path] = field(default_factory=dict)
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    def path(self, value: str) -> Path:
        """Map a config path under :attr:`root` (absolute paths only)."""
        if not value.startswith("/"):
            return Path(value)
        return self.root / value.lstrip("/")

    def run(
        self, command: Sequence[str], timeout: float, env: Mapping[str, str] | None = None,
        limit: int | None = TAIL_CHARS,
    ) -> dict[str, Any]:
        """Run a command and return its exit code, seconds, and output tails.

        Args:
            command: The argument list.
            timeout: Seconds before it is killed.
            env: The environment, or None for this process's.
            limit: Characters kept of each output's end; None keeps all.

        Returns:
            ``{"command", "returncode", "seconds", "stdout", "stderr"}``; a timeout gives
            return code None.
        """
        start = _now()
        record: dict[str, Any] = {"command": [str(c) for c in command]}
        try:
            done = self.runner(
                [str(c) for c in command],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=None if env is None else dict(env),
                check=False,
            )
            record.update(returncode=done.returncode, stdout=_tail(done.stdout, limit),
                          stderr=_tail(done.stderr, limit))
        except subprocess.TimeoutExpired as err:
            record.update(returncode=None, stdout=_tail(err.stdout), stderr=f"timeout {timeout}s")
        except OSError as err:
            record.update(returncode=None, stdout="", stderr=_describe(err))
        record["seconds"] = round(_now() - start, 3)
        return record

    def save(self) -> None:
        """Write :attr:`report` as ``report.json`` (after every step, so a kill keeps it)."""
        self.out.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.report, indent=2, sort_keys=False, default=str) + "\n"
        (self.out / REPORT_FILE).write_text(text, encoding="utf-8")


def site_packages(venv_python: str) -> Path:
    """Return the site-packages folder of a venv from its interpreter path.

    Args:
        venv_python: ``<venv>/bin/python``.

    Returns:
        ``<venv>/lib/python3.X/site-packages`` (the first match).

    Raises:
        StepSkipped: If there is none.
    """
    venv = Path(venv_python).parent.parent
    found = sorted(venv.glob("lib/python*/site-packages"))
    if not found:
        raise StepSkipped(f"no site-packages under {venv}")
    return found[0]


def cuda_env(ctx: Context) -> dict[str, str]:
    """Return the environment dfranzen's launcher sets for the venv's bundled CUDA 13 toolkit.

    Args:
        ctx: The run context (``venv_python`` set).

    Returns:
        ``os.environ`` with ``PYTHONPATH`` removed and ``CUDA_HOME``, ``PATH``,
        ``LD_LIBRARY_PATH`` pointed at ``nvidia/cu13`` when the venv has it, plus the config's
        ``env`` values.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONNOUSERSITE"] = "1"
    if ctx.venv_python:
        try:
            cu13 = sorted(site_packages(ctx.venv_python).glob("nvidia/cu1*"))
        except StepSkipped:
            cu13 = []
        if cu13:
            home = str(cu13[-1])
            env["CUDA_HOME"] = home
            env["PATH"] = f"{home}/bin:{Path(ctx.venv_python).parent}:" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = f"{home}/lib64:{home}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env.update({str(k): str(v) for k, v in ctx.config.get("env", {}).items()})
    return env


# ---- steps -----------------------------------------------------------------------------------


def list_tree(base: Path, depth: int, max_entries: int) -> list[str]:
    """List a folder tree to ``depth`` levels as ``<size> <path>/`` lines, bounded.

    Args:
        base: The folder.
        depth: Levels shown; a child of ``base`` is level 1.
        max_entries: The most lines.

    Returns:
        Lines relative to ``base``'s parent, folders with a trailing ``/``.
    """
    lines: list[str] = []
    seen: set[str] = set()

    def walk(folder: Path, level: int) -> None:
        real = os.path.realpath(folder)
        if real in seen or len(lines) >= max_entries:
            return
        seen.add(real)
        try:
            children = sorted(folder.iterdir(), key=lambda p: p.name)
        except OSError as err:
            lines.append(f"? {folder} ({_describe(err)})")
            return
        for child in children:
            if len(lines) >= max_entries:
                lines.append("... (truncated)")
                return
            if child.is_dir():
                lines.append(f"- {child}/")
                if level < depth:
                    walk(child, level + 1)
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = -1
                lines.append(f"{size} {child}")

    if base.is_dir():
        walk(base, 1)
    return lines


def mem_total_gib() -> float | None:
    """Return the machine's RAM in GiB from ``/proc/meminfo``, or None off Linux."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        return None
    return None


def find_input(ctx: Context, name: str, spec: Mapping[str, Any]) -> Path | None:
    """Return an input's folder: its configured path, else a search by marker file.

    Args:
        ctx: The run context.
        name: The input's name in the config.
        spec: ``{"path", "marker", "search_depth"}``.

    Returns:
        The folder, or None when neither finds it.
    """
    path = ctx.path(spec["path"])
    marker = spec.get("marker")
    if path.is_dir() and (not marker or (path / marker).exists() or any(path.rglob(marker))):
        return path
    base = ctx.path(ctx.config["paths"]["input_root"])
    if not marker or not base.is_dir():
        return None
    hint = spec.get("hint", "")
    depth = int(spec.get("search_depth", 6))
    for current, folders, files in os.walk(base, followlinks=True):
        level = len(Path(current).relative_to(base).parts)
        if level >= depth:
            folders[:] = []
        folders[:] = [f for f in sorted(folders) if f not in {"site-packages", "__pycache__"}]
        if marker in files or marker in folders:
            if not hint or hint in current:
                return Path(current)
    return None


def step_listing(ctx: Context) -> dict[str, Any]:
    """List ``/kaggle/input`` and locate every configured input."""
    paths = ctx.config["paths"]
    listing = ctx.config.get("listing", {})
    lines = list_tree(
        ctx.path(paths["input_root"]), int(listing.get("depth", 6)),
        int(listing.get("max_entries", 4000)),
    )
    (ctx.out / "input_listing.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    disk = {}
    for name in listing.get("disk_paths", []):
        folder = ctx.path(name)
        if folder.exists():
            usage = shutil.disk_usage(folder)
            disk[name] = {"free_gib": round(usage.free / 1024**3, 1),
                          "total_gib": round(usage.total / 1024**3, 1)}
    found = {}
    for name, spec in ctx.config.get("inputs", {}).items():
        folder = find_input(ctx, name, spec)
        if folder is not None:
            ctx.inputs[name] = folder
        found[name] = {
            "configured": spec["path"],
            "found": None if folder is None else str(folder),
            "as_configured": folder is not None and folder == ctx.path(spec["path"]),
        }
    return {"lines": len(lines), "inputs": found, "disk": disk, "cpus": os.cpu_count(),
            "mem_total_gib": mem_total_gib()}


def find_wheel(folder: Path, pattern: str) -> list[Path]:
    """Return the wheels under ``folder`` matching a glob, sorted.

    Args:
        folder: Where to search (recursively).
        pattern: A glob such as ``uv-*.whl``.

    Returns:
        The paths.
    """
    return sorted(folder.rglob(pattern))


def extract_uv(wheel: Path, dest: Path) -> Path:
    """Extract the Linux ``uv`` executable from its wheel, as dfranzen's launcher does.

    Args:
        wheel: The ``uv-*.whl``.
        dest: The binary's path.

    Returns:
        ``dest``, executable.

    Raises:
        ValueError: If the wheel holds no single ``uv`` member or it is not an ELF.
    """
    with zipfile.ZipFile(wheel) as archive:
        members = [n for n in archive.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"{wheel.name}: expected one uv executable, found {members}")
        binary = archive.read(members[0])
    if not binary.startswith(ELF_MAGIC):
        raise ValueError(f"{wheel.name}: the uv member is not a Linux ELF binary")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(binary)
    dest.chmod(0o755)
    return dest


def ensure_uv(ctx: Context) -> str:
    """Return the uv binary: from the wheelhouse's uv wheel, else ``uv.fallback_path``.

    Args:
        ctx: The run context.

    Returns:
        The binary's path.

    Raises:
        StepSkipped: If neither is available.
    """
    if ctx.uv:
        return ctx.uv
    spec = ctx.config.get("uv", {})
    wheelhouse = ctx.inputs.get("wheelhouse")
    errors = []
    if wheelhouse is not None:
        wheels = find_wheel(wheelhouse, spec.get("wheel_glob", "uv-*.whl"))
        if len(wheels) == 1:
            try:
                dest = ctx.path(ctx.config["paths"]["scratch"]) / "bin" / "uv"
                ctx.uv = str(extract_uv(wheels[0], dest))
                return ctx.uv
            except (OSError, ValueError, zipfile.BadZipFile) as err:
                errors.append(_describe(err))
        else:
            errors.append(f"uv wheels found: {[w.name for w in wheels]}")
    fallback = spec.get("fallback_path")
    if fallback == "PATH":
        fallback = shutil.which("uv")
    if fallback and Path(fallback).is_file():
        ctx.uv = str(fallback)
        return ctx.uv
    raise StepSkipped(f"no uv binary: {'; '.join(errors) or 'no wheelhouse'}")


def find_python(folder: Path, version: str) -> Path | None:
    """Return a ``python<version>`` executable under ``folder``, if any.

    Args:
        folder: Where uv installed or extracted the interpreter.
        version: ``3.12``.

    Returns:
        The first ``bin/python<version>`` found, sorted, or None.
    """
    hits = sorted(p for p in folder.rglob(f"python{version}") if p.parent.name == "bin")
    return hits[0] if hits else None


def step_python(ctx: Context) -> dict[str, Any]:
    """Record the image's Python; fetch a 3.12 with uv when the image's is another version."""
    spec = ctx.config["python"]
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    want = have if spec["want"] == "auto" else str(spec["want"])
    result: dict[str, Any] = {
        "sys_version": sys.version, "executable": sys.executable, "want": want,
        "platform": platform.platform(), "glibc": platform.libc_ver(),
    }
    mode = spec.get("standalone", "auto")
    if have == want:
        ctx.python = sys.executable
        result["source"] = "image"
        if mode != "always":
            return result
    elif mode == "never":
        raise StepSkipped(f"image Python is {have}, want {want}, and standalone is never")
    install_dir = ctx.path(ctx.config["paths"]["python_out"])
    uv = ensure_uv(ctx)
    env = dict(os.environ, UV_PYTHON_DOWNLOADS="automatic", UV_PYTHON_INSTALL_DIR=str(install_dir))
    run = ctx.run(
        [uv, "python", "install", spec.get("uv_request", want), "--install-dir", str(install_dir)],
        float(spec.get("timeout_s", 600)), env,
    )
    result["uv_python_install"] = run
    found = find_python(install_dir, want)
    if found is None:
        url = spec.get("fallback_url")
        if url:
            archive = ctx.path(ctx.config["paths"]["scratch"]) / "python-standalone.tar.gz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            result["fallback_url"] = url
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(install_dir, filter="tar")
            found = find_python(install_dir, want)
    if found is None:
        raise RuntimeError(f"no python{want} after uv python install and the fallback URL")
    ctx.standalone = str(found)
    if ctx.python is None:
        ctx.python = str(found)
        result["source"] = "standalone"
    version = ctx.run([str(found), "-c", "import sys; print(sys.version)"], 60)
    result.update(standalone=str(found), standalone_version=version["stdout"].strip(),
                  python=ctx.python)
    return result


def step_venv(ctx: Context) -> dict[str, Any]:
    """Build the venv from the wheelhouse offline, as dfranzen's launcher does."""
    if ctx.python is None:
        raise StepSkipped("no Python 3.12 (step python)")
    wheelhouse = ctx.inputs.get("wheelhouse")
    if wheelhouse is None:
        raise StepSkipped("no wheelhouse input")
    spec = ctx.config["venv"]
    uv = ensure_uv(ctx)
    prefix = ctx.path(spec["prefix"])
    venv_python = prefix / "bin" / "python"
    wheels_name = spec.get("wheels_dir", "wheels")
    wheels_dirs = sorted(p for p in wheelhouse.rglob(wheels_name) if p.is_dir())
    wheels = wheels_dirs[0] if wheels_dirs else wheelhouse
    locks = sorted(wheelhouse.rglob(spec.get("lock", "requirements.lock")))
    sglang = find_wheel(wheels, spec.get("sglang_glob", "sglang-*.whl"))
    result: dict[str, Any] = {
        "wheels_dir": str(wheels), "lock": [str(p) for p in locks],
        "wheel_count": len(list(wheels.glob("*.whl"))), "sglang_wheels": [p.name for p in sglang],
    }
    env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    timeout = float(spec.get("timeout_s", 1800))
    steps = []
    if not venv_python.exists():
        steps.append(ctx.run([uv, "venv", "--python", ctx.python, str(prefix)], 300, env))
    install = [uv, "pip", "install", "--python", str(venv_python), "--no-index",
               "--find-links", str(wheels)]
    for extra in spec.get("extra_find_links", []):
        install += ["--find-links", str(ctx.path(extra))]
    if locks:
        steps.append(ctx.run([*install, "-r", str(locks[0])], timeout, env))
    else:
        steps.append({"skipped": "no requirements.lock in the wheelhouse"})
    if len(sglang) == 1 and spec.get("reinstall_sglang_no_deps", True):
        steps.append(ctx.run([*install, "--reinstall", "--no-deps", str(sglang[0])], 600, env))
    result["commands"] = steps
    failed = [s for s in steps if "command" in s and s.get("returncode") != 0]
    if not venv_python.exists():
        raise RuntimeError(f"venv not created: {steps[-1] if steps else 'no command'}")
    ctx.venv_python = str(venv_python)
    check = ctx.run([uv, "pip", "check", "--python", str(venv_python)], 120, env)
    result["pip_check"] = {"returncode": check["returncode"], "output": check["stdout"][-2000:]}
    freeze = ctx.run([uv, "pip", "freeze", "--python", str(venv_python)], 120, env)
    (ctx.out / "versions.txt").write_text(freeze.get("stdout", ""), encoding="utf-8")
    versions = {}
    for line in freeze.get("stdout", "").splitlines():
        name, sep, version = line.partition("==")
        if sep:
            versions[name.strip().lower()] = version.strip()
    keys = spec.get("report_versions", [])
    result["versions"] = {k: versions.get(k.lower()) for k in keys}
    result["installed"] = len(versions)
    if failed:
        result["failed_commands"] = len(failed)
        raise RuntimeError(f"{len(failed)} install command(s) failed; see commands")
    return result


IMPORT_PROBE = (
    "import importlib, json, sys\n"
    "name = sys.argv[1]\n"
    "module = importlib.import_module(name)\n"
    "print(json.dumps({'version': str(getattr(module, '__version__', None)),"
    " 'file': str(getattr(module, '__file__', None))}))\n"
)


def step_imports(ctx: Context) -> dict[str, Any]:
    """Import each configured module in the venv, one process each, recording errors."""
    if ctx.venv_python is None:
        raise StepSkipped("no venv (step venv)")
    spec = ctx.config["imports"]
    env = cuda_env(ctx)
    results = {}
    for name in spec["modules"]:
        run = ctx.run([ctx.venv_python, "-c", IMPORT_PROBE, name], float(spec["timeout_s"]), env)
        entry: dict[str, Any] = {"ok": run["returncode"] == 0, "seconds": run["seconds"]}
        if run["returncode"] == 0:
            try:
                entry.update(json.loads(run["stdout"].strip().splitlines()[-1]))
            except (ValueError, IndexError):
                entry["stdout"] = run["stdout"][-500:]
        else:
            lines = [ln for ln in run["stderr"].splitlines() if ln.strip()]
            entry["error"] = lines[-1] if lines else run["stderr"]
            entry["stderr_tail"] = run["stderr"][-1500:]
        results[name] = entry
    return {"modules": results, "ok": sorted(k for k, v in results.items() if v["ok"])}


def apply_patch(site: Path, spec: Mapping[str, Any], context_chars: int = 1500,
                dry_run: bool = False) -> dict[str, Any]:
    """Apply one hash-guarded regex edit to an installed file (the recipe's R4).

    The same function is in ``scripts/fn_probe.py``: each Kaggle script is a single file.

    Args:
        site: The venv's site-packages.
        spec: ``{"name", "file", "sha256_before", "sha256_after" (optional), "pattern",
            "replacement", "count"}``: the file must hash to ``sha256_before`` and the pattern
            must match exactly ``count`` times.
        context_chars: Characters kept around the first match, for the report.
        dry_run: Check and report without writing.

    Returns:
        ``status`` (``applied``, ``checked``, ``already applied``, ``sha256 differs``,
        ``match count``, ``missing``), the hashes, the match count, and the context.
    """
    path = site / spec["file"]
    out: dict[str, Any] = {"name": spec.get("name"), "file": spec["file"]}
    if not path.is_file():
        return {**out, "status": "missing"}
    data = path.read_bytes()
    before = hashlib.sha256(data).hexdigest()
    out["sha256_before"] = before
    if spec.get("sha256_after") and before == spec["sha256_after"]:
        return {**out, "status": "already applied"}
    if spec.get("sha256_before") and before != spec["sha256_before"]:
        return {**out, "status": "sha256 differs", "expected": spec["sha256_before"]}
    text = data.decode("utf-8")
    matches = list(re.finditer(spec["pattern"], text))
    out["matches"] = len(matches)
    if matches:
        first = matches[0]
        out["context"] = text[max(0, first.start() - context_chars): first.end() + context_chars]
    if len(matches) != int(spec.get("count", 1)):
        return {**out, "status": "match count"}
    new = re.sub(spec["pattern"], spec["replacement"], text, count=len(matches))
    out["sha256_after"] = hashlib.sha256(new.encode("utf-8")).hexdigest()
    if dry_run:
        return {**out, "status": "checked"}
    path.write_text(new, encoding="utf-8")
    return {**out, "status": "applied"}


def step_patch(ctx: Context) -> dict[str, Any]:
    """Check (or apply) the recipe's guarded edits on the venv; save the code around them."""
    if ctx.venv_python is None:
        raise StepSkipped("no venv (step venv)")
    site = site_packages(ctx.venv_python)
    spec = ctx.config.get("patches", {})
    results = []
    for item in spec.get("items", []):
        result = apply_patch(site, item, dry_run=not spec.get("apply", False))
        context = result.pop("context", "")
        (ctx.out / f"patch_{item['name']}_context.txt").write_text(context, encoding="utf-8")
        results.append(result)
    ok = all(r["status"] in {"applied", "checked", "already applied"} for r in results)
    return {"status": "ok" if ok else "failed", "patches": results}


def step_server_help(ctx: Context) -> dict[str, Any]:
    """Run the server's ``--help`` in the venv and check the probe's flags against it.

    Why: an unknown flag stops SGLang at argument parsing, which on the GPU run would cost the
    boot's quota; here it costs nothing. The builder copies the probe's flag names into
    ``server_help.flags`` (``scripts/build_fn_notebooks.py``).
    """
    if ctx.venv_python is None:
        raise StepSkipped("no venv (step venv)")
    spec = ctx.config["server_help"]
    run = ctx.run([ctx.venv_python, "-m", spec["module"], "--help"], float(spec["timeout_s"]),
                  cuda_env(ctx), limit=None)
    text = run["stdout"] + "\n" + run["stderr"]
    (ctx.out / "server_help.txt").write_text(text, encoding="utf-8")
    pattern = re.compile(r"(?<![\w-])(--[A-Za-z0-9][A-Za-z0-9_-]*)")
    known = set(pattern.findall(text))
    # The argument definitions' source as well: --help may fail on a session without CUDA.
    sources = {}
    for rel in spec.get("source_files", []):
        path = site_packages(ctx.venv_python) / rel
        if path.is_file():
            found = set(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
            sources[rel] = len(found)
            known |= found
    flags = list(spec.get("flags", []))
    unknown = [f for f in flags if f not in known]
    ok = (run["returncode"] == 0 or sources) and not unknown
    return {"status": "ok" if ok else "failed", "returncode": run["returncode"],
            "flags_checked": len(flags), "known_flags": len(known), "unknown_flags": unknown,
            "source_files": sources,
            "error_tail": "" if run["returncode"] == 0 else run["stderr"][-1500:]}


def find_cuobjdump(ctx: Context) -> str | None:
    """Return a ``cuobjdump`` from the venv's CUDA wheels or ``PATH``, if any."""
    if ctx.venv_python:
        try:
            hits = sorted(site_packages(ctx.venv_python).glob("nvidia/**/bin/cuobjdump"))
        except StepSkipped:
            hits = []
        if hits:
            return str(hits[0])
    return shutil.which("cuobjdump")


def step_arch(ctx: Context) -> dict[str, Any]:
    """Read the CUDA architectures of every compiled file of the configured packages."""
    if ctx.venv_python is None:
        raise StepSkipped("no venv (step venv)")
    spec = ctx.config["arch"]
    target = int(spec.get("target_sm", TARGET_SM))
    sp = site_packages(ctx.venv_python)
    tool = find_cuobjdump(ctx)
    packages: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    for package in spec["packages"]:
        checks = int(spec.get("cuobjdump_files", 0))
        folder = sp / package
        # A package folder, or a glob of top-level extension modules such as flash_attn_2_cuda.
        candidates = (device_files(folder, int(spec.get("max_files", 5000))) if folder.is_dir()
                      else sorted(p for p in sp.glob(package) if p.is_file()))
        if not candidates:
            packages[package] = {"verdict": "not installed"}
            continue
        files = []
        for path in candidates:
            rel = str(path.relative_to(sp))
            try:
                scan = scan_device_code(path)
            except (OSError, ValueError) as err:
                files.append({"path": rel, "elf": [], "ptx": [], "runs_on": None,
                              "error": _describe(err)})
                continue
            entry = {"path": rel, "size": path.stat().st_size, **scan,
                     "runs_on": runs_on(scan, target)}
            if tool and checks > 0 and (scan["elf"] or scan["ptx"]):
                checks -= 1
                listed = ctx.run([tool, "--list-elf", "--list-ptx", str(path)], 600)
                entry["cuobjdump"] = parse_cuobjdump(listed["stdout"])
                entry["cuobjdump_agrees"] = set(entry["cuobjdump"]) == set(
                    scan["elf"] + scan["ptx"])
            files.append(entry)
        summary = summarize_package(files)
        summary["jit_note"] = spec.get("jit_notes", {}).get(package)
        packages[package] = summary
        detail[package] = files
    (ctx.out / "arch_files.json").write_text(
        json.dumps(detail, indent=1, default=str) + "\n", encoding="utf-8")
    return {"target_sm": target, "cuobjdump": tool, "packages": packages}


SMALL_JSON = (
    "config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json",
    "preprocessor_config.json", "quantization_config.json", "hybrid_sources.json",
)


def describe_checkpoint(ctx: Context, folder: Path, label: str) -> dict[str, Any]:
    """Record a checkpoint's small configs, file sizes, and component bytes.

    Args:
        ctx: The run context.
        folder: The checkpoint's mount folder.
        label: ``model`` or ``drafter``; names the files written.

    Returns:
        The summary written to ``<label>.json``.
    """
    spec = ctx.config["model"]
    files = []
    total = 0
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            files.append({"path": str(path.relative_to(folder)), "bytes": size})
    configs: dict[str, Any] = {}
    for name in SMALL_JSON:
        hits = sorted(folder.rglob(name))
        if hits:
            try:
                configs[str(hits[0].relative_to(folder))] = read_json(hits[0])
            except (OSError, ValueError) as err:
                configs[name] = {"error": _describe(err)}
    indexes = sorted(folder.rglob("model.safetensors.index.json"))
    index_info: dict[str, Any] = {}
    if indexes:
        index = read_json(indexes[0])
        weight_map = index.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        base = indexes[0].parent
        index_info = {
            "path": str(indexes[0].relative_to(folder)), "tensors": len(weight_map),
            "shards": len(shards), "metadata": index.get("metadata"),
            "missing_shards": [n for n in shards if not (base / n).is_file()],
        }
    sizes = model_sizes(folder, [tuple(r) for r in spec["components"]], int(spec["max_shards"]))
    template = None
    for name in ("chat_template.jinja",):
        hits = sorted(folder.rglob(name))
        if hits:
            template = read_bytes(hits[0]).decode("utf-8")
    tok_config = next((v for k, v in configs.items() if k.endswith("tokenizer_config.json")), {})
    if template is None and isinstance(tok_config, dict):
        template = tok_config.get("chat_template")
    if isinstance(template, str):
        (ctx.out / f"{label}_chat_template.jinja.txt").write_text(template, encoding="utf-8")
    tokenizer = sorted(folder.rglob("tokenizer.json"))
    summary = {
        "folder": str(folder), "files": len(files), "total_bytes": total,
        "total_gib": round(total / 1024**3, 3), "index": index_info, "components": sizes,
        "has_chat_template": isinstance(template, str),
        "tokenizer_sha256": sha256_file(tokenizer[0]) if tokenizer else None,
        "configs": configs,
    }
    (ctx.out / f"{label}_files.json").write_text(json.dumps(files, indent=1) + "\n", "utf-8")
    (ctx.out / f"{label}.json").write_text(json.dumps(summary, indent=1) + "\n", "utf-8")
    return summary


def step_model(ctx: Context) -> dict[str, Any]:
    """Describe the model and the drafter: configs, sizes, components, the chat template."""
    result = {}
    for label in ("model", "drafter"):
        folder = ctx.inputs.get(label)
        if folder is None:
            result[label] = {"error": "input not found"}
            continue
        summary = describe_checkpoint(ctx, folder, label)
        config = next((v for k, v in summary["configs"].items() if k.endswith("config.json")
                       and k.split("/")[-1] == "config.json"), {})
        text = config.get("text_config", config) if isinstance(config, dict) else {}
        keys = ctx.config["model"].get("report_keys", [])
        result[label] = {
            "total_gib": summary["total_gib"], "files": summary["files"],
            "gib_by_component": summary["components"]["gib_by_component"],
            "tokenizer_sha256": summary["tokenizer_sha256"],
            "quantization": {k: (config.get("quantization_config") or {}).get(k)
                             for k in ("quant_method", "bits", "group_size", "sym")}
            if isinstance(config, dict) else None,
            "missing_shards": summary["index"].get("missing_shards"),
            "text_config": {k: text.get(k) for k in keys if isinstance(text, dict)},
        }
    want = ctx.config["model"].get("tokenizer_sha256")
    if want and isinstance(result.get("model"), dict):
        result["model"]["tokenizer_matches_27b"] = result["model"].get("tokenizer_sha256") == want
    return result


TEMPLATE_PROBE = r"""
import json, sys
from transformers import AutoTokenizer
spec = json.loads(sys.argv[1])
tok = AutoTokenizer.from_pretrained(spec["model"], trust_remote_code=True)
messages = [{"role": "system", "content": spec["system"]},
            {"role": "user", "content": spec["user"]}]
out = {"renders": {}, "ids": {}}
for name, kwargs in spec["variants"].items():
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       **kwargs)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out["renders"][name] = {"text": text, "tokens": len(ids)}
    except Exception as err:
        out["renders"][name] = {"error": f"{type(err).__name__}: {err}"}
for marker in spec["markers"]:
    out["ids"][marker] = tok(marker, add_special_tokens=False)["input_ids"]
note = spec["note"] + "</think>\n\n"
out["note_tokens"] = len(tok(note, add_special_tokens=False)["input_ids"])
out["eos_token"] = tok.eos_token
print(json.dumps(out))
"""


def step_template(ctx: Context) -> dict[str, Any]:
    """Render the chat template with the venv's tokenizer at the configured effort."""
    if ctx.venv_python is None:
        raise StepSkipped("no venv (step venv)")
    model = ctx.inputs.get("model")
    if model is None:
        raise StepSkipped("no model input")
    spec = dict(ctx.config["template"])
    spec["model"] = str(model)
    run = ctx.run([ctx.venv_python, "-c", TEMPLATE_PROBE, json.dumps(spec)],
                  float(spec.get("timeout_s", 300)), cuda_env(ctx))
    if run["returncode"] != 0:
        raise RuntimeError(f"template probe failed: {run['stderr'][-1500:]}")
    data = json.loads(run["stdout"].strip().splitlines()[-1])
    (ctx.out / "template.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return {
        "variants": {k: v.get("tokens", v.get("error")) for k, v in data["renders"].items()},
        "ids": data["ids"], "note_tokens": data["note_tokens"],
        "generation_prompt_tail": {
            k: v["text"][-80:] for k, v in data["renders"].items() if "text" in v
        },
    }


def _download(url: str, dest: Path, sha256: str | None) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = _now()
    urllib.request.urlretrieve(url, dest)
    digest = sha256_file(dest)
    if sha256 and digest != sha256:
        dest.unlink()
        raise ValueError(f"{dest.name}: sha256 {digest} is not the configured {sha256}")
    return {"url": url, "file": dest.name, "bytes": dest.stat().st_size, "sha256": digest,
            "seconds": round(_now() - start, 3)}


def install_check(ctx: Context, wheel: Path, module: str | None) -> dict[str, Any]:
    """Install a fetched wheel into the venv ``--no-deps`` and import a module from it.

    Args:
        ctx: The run context.
        wheel: The wheel.
        module: The module to import, or None.

    Returns:
        The install and import records.
    """
    if ctx.venv_python is None:
        return {"skipped": "no venv"}
    uv = ensure_uv(ctx)
    env = dict(os.environ, UV_OFFLINE="1")
    out = {"install": ctx.run([uv, "pip", "install", "--python", ctx.venv_python, "--no-deps",
                               str(wheel)], 600, env)}
    if module:
        out["import"] = ctx.run([ctx.venv_python, "-c", IMPORT_PROBE, module], 300, cuda_env(ctx))
    return out


def step_fetch(ctx: Context) -> dict[str, Any]:
    """Download the wheels the recipe names (URLs, then ``pip download`` specs)."""
    spec = ctx.config.get("fetch", {})
    out_dir = ctx.path(ctx.config["paths"]["wheels_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in spec.get("urls", []):
        entry: dict[str, Any] = {"name": item["name"]}
        try:
            url = item["url"].replace("{root}", str(ctx.root).rstrip("/"))
            name = item.get("file") or url.rsplit("/", 1)[-1]
            entry.update(_download(url, out_dir / name, item.get("sha256")))
            entry["check"] = install_check(ctx, out_dir / name, item.get("module"))
            entry["ok"] = True
        except Exception as err:
            entry.update(ok=False, error=_describe(err))
        results.append(entry)
    for item in spec.get("pip", []):
        entry = {"name": item["spec"]}
        before = set(out_dir.glob("*.whl"))
        command = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                   "--dest", str(out_dir), *item.get("args", []), item["spec"]]
        run = ctx.run(command, float(item.get("timeout_s", 900)))
        new = sorted(set(out_dir.glob("*.whl")) - before)
        entry.update(ok=run["returncode"] == 0 and bool(new), files=[p.name for p in new],
                     run=run)
        if new:
            entry["check"] = install_check(ctx, new[0], item.get("module"))
        results.append(entry)
    ok = sum(1 for r in results if r.get("ok"))
    # A failed fetch marks the step failed, with every item's record kept.
    return {"status": "ok" if ok == len(results) else "failed", "fetched": results, "ok": ok}


def step_fa2(ctx: Context) -> dict[str, Any]:
    """Build FlashAttention 2 from its sdist against the venv's torch, for the set archs."""
    if ctx.venv_python is None:
        raise StepSkipped("no venv (step venv)")
    spec = ctx.config["fa2"]
    scratch = ctx.path(ctx.config["paths"]["scratch"]) / "fa2"
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir = ctx.path(ctx.config["paths"]["wheels_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    meta_url = spec["pypi_json"].format(version=spec["version"])
    with urllib.request.urlopen(meta_url, timeout=60) as response:
        meta = json.loads(response.read().decode("utf-8"))
    sdists = [u for u in meta.get("urls", []) if u.get("packagetype") == "sdist"]
    if not sdists:
        raise RuntimeError(f"no sdist for flash-attn {spec['version']} on PyPI")
    sdist = sdists[0]
    archive = scratch / sdist["filename"]
    result["sdist"] = _download(sdist["url"], archive, sdist.get("digests", {}).get("sha256"))
    with tarfile.open(archive) as tar:
        tar.extractall(scratch, filter="tar")
    sources = sorted(p for p in scratch.iterdir() if p.is_dir() and (p / "setup.py").exists())
    if not sources:
        raise RuntimeError("the sdist holds no setup.py folder")
    uv = ensure_uv(ctx)
    online = {k: v for k, v in os.environ.items() if k not in {"UV_OFFLINE", "PYTHONPATH"}}
    result["build_deps"] = ctx.run(
        [uv, "pip", "install", "--python", ctx.venv_python, *spec["build_deps"]], 900, online)
    env = cuda_env(ctx)
    env.update({str(k): str(v) for k, v in spec["build_env"].items()})
    log = ctx.out / "fa2_build.log"
    start = _now()
    with open(log, "w", encoding="utf-8") as handle:
        done = subprocess.run(
            [ctx.venv_python, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
             "--wheel-dir", str(out_dir), str(sources[0])],
            stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=float(spec["timeout_s"]),
            check=False,
        )
    result["build"] = {"returncode": done.returncode, "seconds": round(_now() - start, 1),
                       "log_tail": _tail(log.read_text(encoding="utf-8", errors="replace"))}
    wheels = sorted(out_dir.glob("flash_attn-*.whl"))
    result["wheels"] = [w.name for w in wheels]
    if done.returncode != 0 or not wheels:
        raise RuntimeError(f"the build failed (exit {done.returncode}); see fa2_build.log")
    unpacked = scratch / "wheel"
    with zipfile.ZipFile(wheels[-1]) as archive_zip:
        archive_zip.extractall(unpacked)
    scans = {}
    for path in sorted(unpacked.rglob("*.so")):
        scan = scan_device_code(path)
        scans[path.name] = {**scan, "runs_on": runs_on(scan, int(spec.get("target_sm", 89)))}
    result["arch"] = scans
    result["check"] = install_check(ctx, wheels[-1], spec.get("module"))
    return result


def step_package(ctx: Context) -> dict[str, Any]:
    """Write the wheels' manifest, the Python tarball, and the output marker."""
    paths = ctx.config["paths"]
    working = ctx.path(paths["working"])
    wheels_dir = ctx.path(paths["wheels_out"])
    wheels = sorted(wheels_dir.glob("*.whl")) if wheels_dir.is_dir() else []
    manifest = {w.name: {"bytes": w.stat().st_size, "sha256": sha256_file(w)} for w in wheels}
    if wheels:
        (wheels_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    tarball = None
    python_dir = ctx.path(paths["python_out"])
    if ctx.standalone and Path(ctx.standalone).is_relative_to(python_dir):
        tarball = working / ctx.config["python"].get("tarball", "python312.tar.gz")
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(python_dir, arcname="python")
    marker = {
        "run_id": ctx.config.get("run_id"), "schema": SCHEMA, "wheels": manifest,
        "python_tarball": None if tarball is None else tarball.name,
        "python_tarball_sha256": None if tarball is None else sha256_file(tarball),
        "python_version": ctx.report.get("steps", {}).get("python", {}).get("sys_version"),
    }
    (working / OUTPUT_MARKER).write_text(json.dumps(marker, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(working / OUTPUT_MARKER, ctx.out / OUTPUT_MARKER)
    return {"wheels": list(manifest), "python_tarball": marker["python_tarball"]}


STEPS: dict[str, Callable[[Context], dict[str, Any]]] = {
    "listing": step_listing,
    "python": step_python,
    "venv": step_venv,
    "imports": step_imports,
    "patch": step_patch,
    "server_help": step_server_help,
    "arch": step_arch,
    "model": step_model,
    "template": step_template,
    "fetch": step_fetch,
    "fa2": step_fa2,
    "package": step_package,
}


def run_step(ctx: Context, name: str) -> dict[str, Any]:
    """Run one step and record its result, error, or skip; never raise.

    Args:
        ctx: The run context.
        name: A key of :data:`STEPS`.

    Returns:
        The record stored under ``report["steps"][name]``.
    """
    start = _now()
    print(f"== step {name}", flush=True)
    try:
        record: dict[str, Any] = {"status": "ok", **STEPS[name](ctx)}
    except StepSkipped as err:
        record = {"status": "skipped", "reason": str(err)}
    except Exception as err:
        record = {"status": "failed", "error": _describe(err),
                  "traceback": _tail(traceback.format_exc(), 3000)}
    record["seconds"] = round(_now() - start, 3)
    ctx.report["steps"][name] = record
    ctx.save()
    print(f"   {name}: {record['status']} in {record['seconds']} s", flush=True)
    return record


def render_summary(report: Mapping[str, Any]) -> str:
    """Render ``summary.md``: one line per step and the key answers.

    Args:
        report: The finished report.

    Returns:
        Markdown.
    """
    steps = report.get("steps", {})
    lines = [f"# {report.get('run_id')}: Flash-Next environment check", "",
             "| Step | Status | Seconds | Note |", "|---|---|---:|---|"]
    for name, record in steps.items():
        note = " ".join(str(record.get("error") or record.get("reason") or "").split())
        lines.append(f"| {name} | {record['status']} | {record['seconds']} | {note[:120]} |")
    python = steps.get("python", {})
    lines += ["", f"- Python: {python.get('sys_version', 'n/a')} ({python.get('source', 'n/a')})"]
    imports = steps.get("imports", {}).get("modules", {})
    for name, entry in imports.items():
        state = entry.get("version") if entry.get("ok") else f"error: {entry.get('error')}"
        lines.append(f"- import {name}: {state}")
    for package, summary in steps.get("arch", {}).get("packages", {}).items():
        lines.append(
            f"- sm89 {package}: {summary.get('verdict')} (cubins {summary.get('elf_archs')}, "
            f"PTX {summary.get('ptx_archs')}, missing {len(summary.get('sm89_missing', []))})"
        )
    for label, entry in steps.get("model", {}).items():
        if isinstance(entry, dict) and "total_gib" in entry:
            lines.append(f"- {label}: {entry['total_gib']} GiB, {entry['gib_by_component']}")
    return "\n".join(lines) + "\n"


def load_config(path: Path | None) -> dict[str, Any]:
    """Return the run's config: from ``path`` (YAML or JSON), else the embedded one.

    Args:
        path: The config file, or None on Kaggle.

    Returns:
        The config.

    Raises:
        SystemExit: If neither is available.
    """
    if path is None:
        if EMBEDDED_CONFIG is None:
            raise SystemExit("no --config and no embedded config")
        return dict(EMBEDDED_CONFIG)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def apply_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Return the config with its ``smoke`` section merged over it (one level deep per key).

    Args:
        config: The full config.

    Returns:
        A new config for the CPU smoke run.
    """
    merged = json.loads(json.dumps(config))
    for key, value in config.get("smoke", {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def run(config: Mapping[str, Any], root: Path, resume: bool = False,
        runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Run every configured step and return the report.

    Args:
        config: The run's config.
        root: Prefix of every absolute config path.
        resume: Skip steps an earlier ``report.json`` in the report folder marks ``ok``,
            reusing what they found (a rerun in the same session).
        runner: Replaces ``subprocess.run`` in tests.

    Returns:
        The report (also written as ``report.json`` and ``summary.md``).
    """
    out = root / config["paths"]["report_dir"].lstrip("/")
    report: dict[str, Any] = {
        "schema": SCHEMA, "run_id": config.get("run_id"), "started": _now(),
        "config": config, "steps": {},
    }
    previous: dict[str, Any] = {}
    if resume and (out / REPORT_FILE).is_file():
        previous = json.loads((out / REPORT_FILE).read_text(encoding="utf-8")).get("steps", {})
    ctx = Context(config=config, root=root, report=report, out=out, runner=runner)
    ctx.save()
    for name in config["steps"]:
        if previous.get(name, {}).get("status") == "ok" and name not in {"listing", "package"}:
            report["steps"][name] = {**previous[name], "resumed": True}
            if name == "python":
                ctx.python = previous[name].get("python") or sys.executable
            if name == "venv":
                ctx.venv_python = str(ctx.path(config["venv"]["prefix"]) / "bin" / "python")
            continue
        run_step(ctx, name)
    report["finished"] = _now()
    report["seconds"] = round(report["finished"] - report["started"], 3)
    ctx.save()
    (out / SUMMARY_FILE).write_text(render_summary(report), encoding="utf-8")
    return report


def build_smoke_tree(root: Path, config: Mapping[str, Any]) -> None:
    """Write a tiny fake ``/kaggle/input`` under ``root`` for the CPU smoke run.

    The wheelhouse holds one pure-Python wheel (``fnfake``) with a shared library carrying a
    synthetic fat binary (an ``sm_89`` cubin and ``compute_80`` PTX), its lock, and no uv wheel
    (the smoke config takes uv from ``PATH``); the model and the drafter hold configs, a
    template, and safetensors shards whose headers name experts, an n-gram table, and the rest.

    Args:
        root: The smoke root.
        config: The smoke config (its ``inputs`` paths).
    """
    inputs = config["inputs"]
    wheelhouse = root / inputs["wheelhouse"]["path"].lstrip("/")
    wheels = wheelhouse / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    (wheelhouse / "requirements.lock").write_text("fnfake==0.1\n", encoding="utf-8")
    write_fake_wheel(wheels / "fnfake-0.1-py3-none-any.whl", "fnfake", fake_fatbin([(2, 89),
                                                                                   (1, 80)]))
    for label in ("model", "drafter"):
        folder = root / inputs[label]["path"].lstrip("/")
        folder.mkdir(parents=True, exist_ok=True)
        cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"],
               "quantization_config": {"quant_method": "auto-round", "bits": 4},
               "text_config": {"num_hidden_layers": 2, "num_experts": 4,
                               "ple_embedding_dtype": "float8_e4m3fn"}}
        (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (folder / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
        tensors = {"model.layers.0.mlp.experts.0.up_proj.qweight": ("I32", [8, 4]),
                   "model.ngram_embedding.weight": ("F8_E4M3", [16, 8]),
                   "model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4])}
        write_safetensors(folder / "model-00001-of-00001.safetensors", tensors)
        index = {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors"
                                                for k in tensors}}
        (folder / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    extra = root / "kaggle/input/fake-extra"
    extra.mkdir(parents=True, exist_ok=True)
    write_fake_wheel(extra / "fnextra-0.2-py3-none-any.whl", "fnextra", fake_fatbin([(2, 90)]))
    competition = root / inputs["competition"]["path"].lstrip("/")
    competition.mkdir(parents=True, exist_ok=True)
    (competition / "arc-agi_evaluation_challenges.json").write_text("{}", encoding="utf-8")
    (competition / "arc-agi_evaluation_solutions.json").write_text("{}", encoding="utf-8")


def fake_fatbin(entries: Iterable[tuple[int, int]]) -> bytes:
    """Return a minimal fat binary with one empty-payload entry per ``(kind, arch)``.

    Args:
        entries: ``(kind, arch)``: kind 1 PTX, 2 ELF.

    Returns:
        The bytes, header first.
    """
    body = b""
    for kind, arch in entries:
        payload = b"\0" * 16
        body += FATBIN_ENTRY.pack(kind, 0x101, FATBIN_ENTRY.size, len(payload), 0, 0, 0, 0,
                                  arch, 0, 0, 0, 0, 0) + payload
    return FATBIN_HEADER.pack(0xBA55ED50, 1, FATBIN_HEADER.size, len(body)) + body


def write_fake_wheel(path: Path, name: str, device: bytes) -> None:
    """Write a minimal installable wheel with one module and one ``.so`` holding ``device``.

    Args:
        path: The wheel's path; its name gives the version.
        name: The distribution and module name.
        device: Bytes of the shared library.
    """
    version = path.name.split("-")[1]
    info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f"__version__ = '{version}'\n".encode(),
        f"{name}/_kernels.so": b"\x7fELF\x02\x01\x01" + b"\0" * 57 + device,
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: fn_env\nRoot-Is-Purelib: true\n"
        b"Tag: py3-none-any\n",
    }
    record = "".join(f"{k},,\n" for k in files) + f"{info}/RECORD,,\n"
    files[f"{info}/RECORD"] = record.encode()
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in files.items():
            archive.writestr(zipfile.ZipInfo(member, (2020, 1, 1, 0, 0, 0)), data)


def write_safetensors(path: Path, tensors: Mapping[str, tuple[str, list[int]]]) -> None:
    """Write a valid safetensors file of zero-filled tensors.

    Args:
        path: The file.
        tensors: Name to ``(dtype, shape)``.
    """
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        size = count * DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, help="YAML or JSON config; default: embedded")
    parser.add_argument("--root", type=Path, default=Path("/"),
                        help="prefix of every absolute path (tests and the smoke run)")
    parser.add_argument("--smoke", action="store_true",
                        help="CPU smoke: a fake input tree under --root and the smoke section")
    parser.add_argument("--resume", action="store_true",
                        help="skip steps an earlier report.json marks ok")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when every step ran (failures are recorded in the report, not raised); 2 when the
        config is missing.
    """
    args = build_parser().parse_args(argv)
    args.root = args.root.resolve()
    config = load_config(args.config)
    if args.smoke:
        if args.root == Path("/"):
            raise SystemExit("--smoke needs --root, a scratch folder for the fake input tree")
        config = apply_smoke(config)
        build_smoke_tree(args.root, config)
    report = run(config, args.root, resume=args.resume)
    failed = [k for k, v in report["steps"].items() if v["status"] != "ok"]
    print(f"done in {report['seconds']} s; not ok: {failed or 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
