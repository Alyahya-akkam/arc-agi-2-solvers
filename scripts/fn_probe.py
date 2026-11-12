r"""Version 11's solver: Qwen3.8-Flash-Next on SGLang on Kaggle's 4 x L4, LLM first.

Purpose: task P5.2 (``docs/tasks/P5.2.md``, round 2). Version 11 (``alyahya737/arc26-submission``,
its notebook built by ``scripts/build_fn_notebooks.py``) runs this script between a placeholder
``submission.json`` and a final check. In a scored rerun (**competition mode**,
``KAGGLE_IS_COMPETITION_RERUN`` set) it answers every hidden task with one arm, by default "think
with Python" (``arc26.agent.toolshot``, task P5.3), plain when the sandbox fails its self-test, and
fills ``submission.json`` as tasks finish. In every other session (**development mode**, run
``P5.2-01`` revision 2 and the CPU twin) it runs the paired A/B of the harness note on
``P6.9-01``'s DEV120 tasks: arm A plain and arm B tool on one server, the streams split evenly.
Runbook ``docs/runbooks/fn_probe.md``; pre-run brief ``docs/prerun/P5.2-01.md``.

Inputs (Kaggle mounts, found by marker files): the wheelhouse, ``P5.2-00``'s output (the CPython
3.12 tarball, the FA2 wheel), the code dataset (this script, the ``arc26`` wheel, the fake
server), the model, and the competition's challenges. The solutions are never opened here:
``scripts/fn_report.py`` scores the pulled attempts locally.

Outputs under ``/kaggle/working/run_logs/fn/`` (pulled by ``kaggle_ops.py pull``): ``run.json``,
``state.json``, ``inputs.json``, ``install.json``, ``patches.json``, ``arg_check.json``,
``boot.json``, ``server_info.json``, ``memory.jsonl``, ``server_metrics.jsonl``,
``template_check.json``, ``solve_plan.json``, ``attempts.jsonl``, ``transcripts.jsonl``,
``restarts.jsonl``, ``engine_failures.jsonl``, ``timing-inner.jsonl`` (``{puzzle_id, op,
seconds, tokens}``), ``engine.log`` (bounded), ``engine_tail.log``, ``decode_lines.json``,
``summary.md``; in competition mode also ``/kaggle/working/submission.json``.

Method. An outer process (the image's Python, standard library only) finds the inputs, unpacks
Python 3.12, builds the venv from the wheelhouse with uv as dfranzen's launcher does, adds the
FA2 wheel and the ``arc26`` wheel ``--no-deps``, applies the recipe's guarded edit, and runs the
inner process with the venv's Python. The inner process loads the tasks (competition: smallest
opening first), self-tests the tool loop, resolves the server's flags with the fork's own
``ServerArgs`` in a child process (a refusal fails development mode), boots
``python -m sglang.launch_server`` (once more with
fallback flags if the boot or the first generation fails), and runs the refill queue over the
streams the pool allows:

.. math:: n = \min\left(L, \left\lfloor \frac{P}{o_{\max} + C + F + R + S} \right\rfloor, 8\right),

with :math:`L` the request limit, :math:`P` the pool, :math:`o_{\max}` the longest opening,
:math:`C` the cap, :math:`F` the forcing reply, :math:`R` the tool replies (arm B), :math:`S` the
linear-attention state, the bound capped at the context length. Requests go to SGLang's native
``/generate`` with token ids: the chat template is applied here, the thinking cap is exact, and
the forcing turn (and every tool continuation) is the previous ids plus the new ones, which the
radix cache serves. A health watch restarts a dead server and queues its tasks again. Rates per
request: :math:`r = (n - 1)/(e - f)` for :math:`n` tokens, first token at :math:`f`, end at
:math:`e`. Every stage records its result; ``--resume`` skips the tasks already written.

Usage::

    python fn_probe.py --config fn_config.json --session-start <t0>   # the notebook's cell 2
    python scripts/fn_probe.py --config configs/P5.2-01.yaml --smoke --root <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.parse
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The builder replaces this line with the run's config as JSON (scripts/build_fn_notebooks.py).
EMBEDDED_CONFIG: dict[str, Any] | None = None

SCHEMA = 1
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ENV_MARKER = "fn_env_output.json"
# Where the image keeps the driver's libcuda.so (dfranzen's launcher, cell 12).
DRIVER_DIRS = ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu",
               "/usr/lib64")
TAIL_CHARS = 4000


def _now() -> float:
    return time.time()


def _describe(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def _tail(text: str | bytes | None, limit: int = TAIL_CHARS) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-limit:]


def write_json(path: Path, value: object) -> None:
    """Write JSON atomically (a temporary file, then a rename), so a kill keeps the last copy.

    Args:
        path: The file.
        value: Any JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock | None = None) -> None:
    """Append one JSON line (one write of one complete line).

    Args:
        path: The file.
        row: The record.
        lock: Serializes writers in this process, if given.
    """
    line = json.dumps(row, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock is None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
        return
    with lock, open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping a torn last line.

    Args:
        path: The file; a missing file gives no rows.

    Returns:
        The rows.
    """
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def sha256_file(path: Path) -> str:
    """Return a file's sha256, refusing any file whose name contains ``solution``.

    Args:
        path: The file.

    Returns:
        The hex digest.

    Raises:
        PermissionError: For a solutions file, which this script never opens.
    """
    if "solution" in path.name.lower():
        raise PermissionError(f"refusing to read {path.name}: solutions are never opened")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- config --------------------------------------------------------------------------------------


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
        return json.loads(json.dumps(EMBEDDED_CONFIG))
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def apply_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the config with its ``smoke`` section merged over it (one level per key).

    Args:
        config: The full config.

    Returns:
        The smoke config.
    """
    merged = json.loads(json.dumps(config))
    for key, value in config.get("smoke", {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


# ---- tokenizers ----------------------------------------------------------------------------------


class FakeTokenizer:
    """Byte-level tokenizer with Qwen's markers as single ids, for the CPU smoke only.

    Why: the smoke has no model files and no ``transformers``; the fake server
    (``scripts/fn_fake_sglang.py``) uses the same class, so ids round-trip exactly.
    """

    SPECIAL = {"<think>": 256, "</think>": 257, "<|im_start|>": 258, "<|im_end|>": 259,
               "<|endoftext|>": 260}
    _SPLIT = re.compile("(" + "|".join(re.escape(k) for k in SPECIAL) + ")")

    def encode(self, text: str) -> list[int]:
        """Return the ids of ``text`` (markers as one id, everything else as UTF-8 bytes)."""
        ids: list[int] = []
        for part in self._SPLIT.split(text):
            if part in self.SPECIAL:
                ids.append(self.SPECIAL[part])
            elif part:
                ids.extend(part.encode("utf-8"))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text of ``ids``."""
        inverse = {v: k for k, v in self.SPECIAL.items()}
        out: list[str] = []
        buffer = bytearray()
        for token in ids:
            if token in inverse:
                out.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()
                out.append(inverse[token])
            elif 0 <= token < 256:
                buffer.append(token)
        out.append(buffer.decode("utf-8", errors="replace"))
        return "".join(out)

    def render(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        """Render Qwen's chat format with an open think block (the template's frame)."""
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        return "".join(parts) + "<|im_start|>assistant\n<think>\n"


class HFTokenizer:
    """The model's own tokenizer and chat template through ``transformers`` (in the venv).

    Why: SGLang renders chat requests with the same ``apply_chat_template``, so the prompt sent
    to ``/generate`` is the one its chat endpoint would build (the template check compares them).
    """

    def __init__(self, model_dir: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def encode(self, text: str) -> list[int]:
        """Return the ids of ``text`` without added special tokens."""
        return list(self._tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text of ``ids``, special tokens kept."""
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def render(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        """Render the chat template with a generation prompt and the template's variables."""
        return self._tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **kwargs)


def make_tokenizer(kind: str, model_dir: str | None) -> FakeTokenizer | HFTokenizer:
    """Return the configured tokenizer.

    Args:
        kind: ``hf`` (the model's) or ``fake`` (the smoke's).
        model_dir: The model folder, for ``hf``.

    Returns:
        The tokenizer.

    Raises:
        ValueError: For an unknown kind or ``hf`` without a model folder.
    """
    if kind == "fake":
        return FakeTokenizer()
    if kind == "hf" and model_dir:
        return HFTokenizer(model_dir)
    raise ValueError(f"tokenizer {kind!r} needs a model folder or is unknown")


# ---- HTTP ----------------------------------------------------------------------------------------


def http_json(base: str, method: str, path: str, body: Any = None,
              timeout: float = 30.0) -> tuple[int, Any]:
    """Send one request to the server and parse a JSON reply when there is one.

    Args:
        base: ``http://127.0.0.1:<port>``.
        method: ``GET`` or ``POST``.
        path: The endpoint.
        body: A JSON body, or None.
        timeout: Seconds.

    Returns:
        The status and the parsed JSON (the text when it is not JSON).

    Raises:
        OSError: On a connection failure.
    """
    url = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        try:
            return response.status, json.loads(raw)
        except ValueError:
            return response.status, raw
    finally:
        conn.close()


@dataclass
class GenResult:
    """One ``/generate`` call's outcome.

    Attributes:
        text: Generated text (special tokens kept).
        ids: Generated token ids (empty when the server sent none).
        prompt_tokens: Prompt tokens by the server's count.
        completion_tokens: Generated tokens by the server's count.
        cached_tokens: Prompt tokens served from the radix cache.
        finish: ``stop``, ``length``, ``abort``, ``cancelled``, or ``error``.
        start: Wall time the request was sent.
        first: Wall time of the first chunk with text, or None.
        end: Wall time of the last chunk.
        error: The error, if any.
    """

    text: str = ""
    ids: list[int] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int = 0
    cached_tokens: int | None = None
    finish: str = "error"
    start: float = 0.0
    first: float | None = None
    end: float = 0.0
    error: str | None = None

    def decode_rate(self) -> float | None:
        r"""Return :math:`(n - 1) / (e - f)`, the decode rate, or None if undefined."""
        if self.first is None or self.end <= self.first or self.completion_tokens < 2:
            return None
        return (self.completion_tokens - 1) / (self.end - self.first)

    def row(self) -> dict[str, Any]:
        """Return the record without the text and ids."""
        return {
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens, "finish": self.finish,
            "seconds": round(self.end - self.start, 3),
            "first_token_s": None if self.first is None else round(self.first - self.start, 3),
            "decode_tok_s": None if self.decode_rate() is None else round(self.decode_rate(), 3),
            "error": self.error,
        }


def _merge(current: Any, chunk: Any, incremental: bool | None = None) -> Any:
    # Incremental chunks (--incremental-streaming-output) are appended; cumulative ones replace.
    # None guesses from the content (a chunk that extends the current text is cumulative).
    if incremental is None:
        incremental = not (chunk[: len(current)] == current and len(chunk) >= len(current))
    if incremental:
        return current + chunk if isinstance(current, str) else current + list(chunk)
    return chunk if isinstance(current, str) else list(chunk)


class Generation:
    """A streaming ``/generate`` call that another thread can cut.

    Why: a reply cut at the solve's deadline keeps its tokens, which the forcing turn then
    continues; closing the socket makes SGLang abort the request and free its KV.
    """

    def __init__(self, base: str, body: Mapping[str, Any], timeout: float,
                 incremental: bool | None = None) -> None:
        self.base = base
        self.body = dict(body, stream=True)
        self.timeout = timeout
        self.incremental = incremental
        self.result = GenResult()
        self._conn: http.client.HTTPConnection | None = None
        # http.client hands the socket to the response, so it is kept here for cancel().
        self._sock: socket.socket | None = None
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Cut the request; :meth:`run` then returns with finish ``cancelled``."""
        self._cancelled.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def run(self) -> GenResult:
        """Send the request and read the stream to its end, a cut, or an error.

        Returns:
            The result (also kept as :attr:`result`).
        """
        res = self.result
        res.start = _now()
        url = urllib.parse.urlsplit(self.base)
        try:
            self._conn = http.client.HTTPConnection(url.hostname, url.port, timeout=self.timeout)
            if self._cancelled.is_set():
                raise ConnectionAbortedError("cancelled before the start")
            self._conn.request("POST", "/generate", body=json.dumps(self.body).encode(),
                               headers={"Content-Type": "application/json"})
            self._sock = self._conn.sock
            if self._cancelled.is_set():
                self.cancel()
            response = self._conn.getresponse()
            if response.status != 200:
                res.error = f"HTTP {response.status}: {_tail(response.read(), 500)}"
                res.finish = "error"
                return res
            finish = None
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                now = _now()
                if "error" in chunk:
                    # SGLang refuses a request (for example input + max_new at the context
                    # length) with an error chunk: recorded, not taken for an abort.
                    err = chunk["error"]
                    res.error = str(err.get("message", err) if isinstance(err, Mapping)
                                    else err)[:500]
                    finish = "error"
                    res.end = now
                    break
                text = chunk.get("text", "")
                if text and res.first is None:
                    res.first = now
                res.text = _merge(res.text, text, self.incremental)
                if chunk.get("output_ids"):
                    res.ids = _merge(res.ids, chunk["output_ids"], self.incremental)
                meta = chunk.get("meta_info", {})
                res.prompt_tokens = meta.get("prompt_tokens", res.prompt_tokens)
                res.completion_tokens = meta.get("completion_tokens", res.completion_tokens)
                res.cached_tokens = meta.get("cached_tokens", res.cached_tokens)
                reason = meta.get("finish_reason")
                if isinstance(reason, Mapping):
                    finish = reason.get("type")
                elif isinstance(reason, str):
                    finish = reason
                res.end = now
            res.finish = "cancelled" if self._cancelled.is_set() and finish is None else (
                finish or "abort")
        except (OSError, http.client.HTTPException, ValueError) as err:
            res.finish = "cancelled" if self._cancelled.is_set() else "error"
            if not self._cancelled.is_set():
                res.error = _describe(err)
        finally:
            res.end = max(res.end, res.start) if res.end else _now()
            if self._conn is not None:
                self._conn.close()
        return res


def sampling(config: Mapping[str, Any], max_new: int, ignore_eos: bool = False) -> dict[str, Any]:
    """Return ``/generate``'s ``sampling_params``: the model's sampling plus the cap.

    Args:
        config: The run's config (``sampling``).
        max_new: ``max_new_tokens``.
        ignore_eos: Generate to the cap whatever the model writes (unused by the solve).

    Returns:
        The parameters.
    """
    params = dict(config["sampling"])
    params.update(max_new_tokens=int(max_new), ignore_eos=ignore_eos, skip_special_tokens=False,
                  no_stop_trim=True)
    return params


# ---- the server ----------------------------------------------------------------------------------


def server_argv(config: Mapping[str, Any], python: str, values: Mapping[str, str],
                overrides: Mapping[str, Any] | None = None) -> list[str]:
    """Return the server's command line from ``engine``.

    A flag's value ``true`` writes the bare flag; ``false`` or null leaves it out; a list writes
    its items after the flag; a mapping writes it as JSON; anything else as text. ``{name}``
    placeholders take ``values`` (model, port, chat_template, drafter, scratch).

    Args:
        config: The run's config.
        python: The interpreter that runs the server.
        values: Placeholder values.
        overrides: Flag values that replace the config's (a failed edit's fallback flags).

    Returns:
        The argument list.
    """
    engine = config["engine"]
    if engine.get("script"):
        argv = [python, str(HERE / engine["script"])]
    else:
        argv = [python, "-m", engine["module"]]
    return argv + server_flags({**engine["flags"], **(overrides or {})}, values)


def server_flags(flags: Mapping[str, Any], values: Mapping[str, str]) -> list[str]:
    """Return the flag part of the server's command line, by :func:`server_argv`'s value rules.

    Args:
        flags: Flag to value, in order.
        values: Placeholder values.

    Returns:
        The flags and their values as text.
    """

    def fill(value: Any) -> str:
        text = json.dumps(value) if isinstance(value, Mapping) else str(value)
        for key, val in values.items():
            text = text.replace("{" + key + "}", str(val))
        return text

    argv: list[str] = []
    for flag, value in flags.items():
        if value is None or value is False:
            continue
        argv.append(flag)
        if value is True:
            continue
        if isinstance(value, list):
            argv.extend(fill(v) for v in value)
        else:
            argv.append(fill(value))
    return argv


def find_key(value: Any, key: str) -> Any:
    """Return the first value stored under ``key`` anywhere in nested dicts and lists.

    Args:
        value: Parsed JSON.
        key: The key.

    Returns:
        The value, or None.
    """
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        children = list(value.values())
    elif isinstance(value, list):
        children = value
    else:
        return None
    for child in children:
        found = find_key(child, key)
        if found is not None:
            return found
    return None


def parse_log(text: str, patterns: Mapping[str, str]) -> dict[str, list[float]]:
    """Return every number each pattern's first group captures in the server log.

    Args:
        text: The log.
        patterns: Name to regex with one numeric group.

    Returns:
        Name to the captured numbers in log order (one per tensor-parallel rank, often).
    """
    found: dict[str, list[float]] = {}
    for name, pattern in patterns.items():
        found[name] = [float(m.group(1)) for m in re.finditer(pattern, text)]
    return found


DECODE_LINE = re.compile(r"#running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)")


def decode_lines(text: str) -> dict[str, dict[str, float]]:
    """Summarize SGLang's decode log lines by running requests.

    Args:
        text: The server log.

    Returns:
        Running requests to ``{"lines", "median_tok_s", "max_tok_s"}``.
    """
    by_running: dict[int, list[float]] = {}
    for match in DECODE_LINE.finditer(text):
        by_running.setdefault(int(match.group(1)), []).append(float(match.group(2)))
    out = {}
    for running, rates in sorted(by_running.items()):
        rates.sort()
        out[str(running)] = {"lines": len(rates), "median_tok_s": rates[len(rates) // 2],
                             "max_tok_s": rates[-1]}
    return out


def read_meminfo() -> dict[str, float]:
    """Return ``MemTotal`` and ``MemAvailable`` in GiB from ``/proc/meminfo`` (empty off Linux)."""
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(":")
            if name in {"MemTotal", "MemAvailable"}:
                out[name] = round(int(rest.split()[0]) / 1024**2, 2)
    except OSError:
        pass
    return out


def gpu_memory() -> list[dict[str, float]] | str:
    """Return per-GPU memory from ``nvidia-smi`` in MiB, or the error text.

    Returns:
        ``[{"index", "used_mib", "total_mib", "util"}]``, or a string when the tool is absent.
    """
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return "nvidia-smi not found"
    try:
        out = subprocess.run(
            [tool, "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as err:
        return _describe(err)
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                rows.append({"index": int(parts[0]), "used_mib": float(parts[1]),
                             "total_mib": float(parts[2]), "util": float(parts[3])})
            except ValueError:
                continue
    return rows


METRIC_LINE = re.compile(r"^(sglang[:_][A-Za-z_:]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")


def parse_metrics(text: str, names: Sequence[str]) -> dict[str, float]:
    """Return the selected gauges of a Prometheus ``/metrics`` page (the sum over labels).

    Args:
        text: The page.
        names: Metric names without the ``sglang:`` prefix, such as ``num_running_reqs``.

    Returns:
        Name to value for the names present.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_LINE.match(line.strip())
        if not match:
            continue
        name = re.sub(r"^sglang[:_]", "", match.group(1))
        if name in names:
            out[name] = out.get(name, 0.0) + float(match.group(2))
    return out


class Monitor(threading.Thread):
    """Samples GPU memory, RAM, and the server's gauges every few seconds into JSONL files."""

    def __init__(self, out: Path, base: str, interval: float, metrics: Sequence[str]) -> None:
        super().__init__(daemon=True)
        self.out, self.base, self.interval, self.metrics = out, base, interval, list(metrics)
        self.stop = threading.Event()
        self.peak_used_mib: dict[int, float] = {}
        self.peaks: dict[str, float] = {}
        self.min_ram_available_gib: float | None = None

    def sample(self) -> None:
        """Take one sample of every source."""
        now = _now()
        gpus = gpu_memory()
        ram = read_meminfo()
        if isinstance(gpus, list):
            for gpu in gpus:
                old = self.peak_used_mib.get(gpu["index"], 0.0)
                self.peak_used_mib[gpu["index"]] = max(old, gpu["used_mib"])
        available = ram.get("MemAvailable")
        if available is not None:
            low = self.min_ram_available_gib
            self.min_ram_available_gib = available if low is None else min(low, available)
        append_jsonl(self.out / "memory.jsonl", {"t": round(now, 3), "gpus": gpus, "ram": ram})
        try:
            status, page = http_json(self.base, "GET", "/metrics", timeout=5)
        except OSError:
            return
        if status == 200 and isinstance(page, str):
            values = parse_metrics(page, self.metrics)
            for name, value in values.items():
                self.peaks[name] = max(self.peaks.get(name, value), value)
            append_jsonl(self.out / "server_metrics.jsonl", {"t": round(now, 3), **values})

    def run(self) -> None:
        """Sample until :attr:`stop` is set."""
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as err:  # a sampler must never end the run
                append_jsonl(self.out / "memory.jsonl", {"t": _now(), "error": _describe(err)})
            self.stop.wait(self.interval)


# ---- precache ------------------------------------------------------------------------------------


class Precache(threading.Thread):
    """Reads the model's files in parallel chunks to warm the mount's cache (dfranzen, cell 6).

    Why: the weights are about 125 GB on a network mount; P6.9-02's GCSFuse read 24 MB/s with
    one reader, and parallel reads are how dfranzen's notebook prepares the load. The thread
    also measures the mount's read rate, a reading of its own.
    """

    def __init__(self, folders: Sequence[Path], threads: int, chunk: int, max_s: float) -> None:
        super().__init__(daemon=True)
        self.folders, self.threads, self.chunk, self.max_s = folders, threads, chunk, max_s
        self.bytes_read = 0
        self.seconds = 0.0
        self.done = False
        self._lock = threading.Lock()

    def run(self) -> None:
        """Read every file once, largest work split into chunks, until done or ``max_s``."""
        start = _now()
        chunks = []
        for folder in self.folders:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and "solution" not in path.name.lower():
                    size = path.stat().st_size
                    chunks += [(path, off, min(self.chunk, size - off))
                               for off in range(0, max(size, 1), self.chunk)]

        def read(item: tuple[Path, int, int]) -> None:
            if _now() - start > self.max_s:
                return
            path, offset, length = item
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read(length)
            with self._lock:
                self.bytes_read += len(data)

        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            list(pool.map(read, chunks))
        self.seconds = _now() - start
        self.done = True

    def reading(self) -> dict[str, Any]:
        """Return the bytes read, the seconds, and the rate in MB/s."""
        seconds = self.seconds or 1e-9
        return {"bytes": self.bytes_read, "seconds": round(self.seconds, 1), "done": self.done,
                "mb_per_s": round(self.bytes_read / 1e6 / seconds, 1) if self.done else None}


# ---- the run -------------------------------------------------------------------------------------


@dataclass
class Run:
    """State of one probe run (both processes).

    Attributes:
        config: The resolved config.
        root: Prefix of every absolute config path (``/`` on Kaggle).
        out: The output folder (``run_logs/fn``).
        start: Wall time the outer process started (the wall budget counts from it).
        state: Stage name to its record, saved as ``state.json``.
        lock: Serializes JSONL writers.
    """

    config: dict[str, Any]
    root: Path
    out: Path
    start: float
    state: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def path(self, value: str) -> Path:
        """Map an absolute config path under :attr:`root`."""
        return self.root / value.lstrip("/") if value.startswith("/") else Path(value)

    def left(self) -> float:
        """Seconds left before the mode's ``end_s`` (``time_plan``), from the session start."""
        mode = self.state.get("mode", "development")
        return float(self.config["time_plan"][mode]["end_s"]) - (_now() - self.start)

    def save(self) -> None:
        """Write ``state.json``."""
        write_json(self.out / "state.json", self.state)

    def stage(self, name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Run one stage, record its result or error, and never raise.

        Args:
            name: The stage.
            fn: Its body.

        Returns:
            The record (``status`` ``ok``, ``failed``, or ``skipped``).
        """
        began = _now()
        print(f"== {name} (t={round(began - self.start)} s)", flush=True)
        before = self.state.get(name)
        try:
            record = {"status": "ok", **fn()}
            # A stage's record lives under its name; a stage that stores a value there itself
            # would have it overwritten here (the P5.2-01-cpu install fault), so it is an error.
            if self.state.get(name) is not before:
                raise RuntimeError(f"stage {name} wrote state[{name!r}], its own record's key")
        except StageSkipped as err:
            record = {"status": "skipped", "reason": str(err)}
        except Exception as err:  # every failure is recorded; later stages decide if they run
            record = {"status": "failed", "error": _describe(err),
                      "traceback": _tail(traceback.format_exc(), 3000)}
        record["seconds"] = round(_now() - began, 3)
        self.state[name] = record
        self.save()
        print(f"   {name}: {record['status']} in {record['seconds']} s "
              f"{record.get('error', '')}", flush=True)
        return record


class StageSkipped(Exception):
    """A stage cannot run because an earlier one did not give it what it needs."""


def find_dir(run: Run, spec: Mapping[str, Any]) -> Path | None:
    """Return an input folder: the configured path if it holds the marker, else a search.

    Args:
        run: The run.
        spec: ``{"path", "marker", "hint", "search_depth"}``; ``hint`` is a text or a list of
            texts, one of which the folder's path must contain.

    Returns:
        The folder holding the marker, or None.
    """
    path = run.path(spec["path"]) if spec.get("path") else None
    marker = spec["marker"]
    if path is not None and (path / marker).exists():
        return path
    base = run.path(run.config["paths"]["input_root"])
    if not base.is_dir():
        return None
    depth = int(spec.get("search_depth", 7))
    hint = spec.get("hint") or ""
    hints = [hint] if isinstance(hint, str) else list(hint)
    for current, folders, files in os.walk(base, followlinks=True):
        level = len(Path(current).relative_to(base).parts)
        folders[:] = [] if level >= depth else sorted(
            f for f in folders if f not in {"site-packages", "__pycache__", "venv"})
        if (marker in files or marker in folders) and any(h in current for h in hints):
            return Path(current)
    return None


def stage_inputs(run: Run) -> dict[str, Any]:
    """Find every input; the code dataset's files are checked against the build's sha256."""
    found: dict[str, Any] = {}
    for name, spec in run.config["inputs"].items():
        folder = find_dir(run, spec)
        found[name] = None if folder is None else str(folder)
        if folder is None and spec.get("required", True):
            raise RuntimeError(f"input {name} not found (marker {spec['marker']})")
    build = run.config.get("build", {})
    checks = {}
    if found.get("code") and build.get("files"):
        for name, digest in build["files"].items():
            path = Path(found["code"]) / name
            checks[name] = path.is_file() and sha256_file(path) == digest
        if not all(checks.values()):
            raise RuntimeError(f"code dataset files differ from the build: {checks}")
    run.state["paths"] = found
    write_json(run.out / "inputs.json", {"found": found, "code_checks": checks})
    return {"found": found, "code_checks": checks}


def copy_tree(source: Path, target: Path, workers: int, piece_bytes: int, max_s: float,
              stall_s: float, block_bytes: int = 8 << 20) -> dict[str, Any]:
    r"""Copy a folder to local disk with parallel readers; a file not copied in time is linked.

    Why: ``P5.2-01-cpu6`` (30 Oct) lost its install to the input mount: ``uv pip install`` of the
    6.6 GB wheelhouse straight from GCSFuse passed its 1,800 s timeout, where twins 2 to 5 took
    104 to 121 s; ``P6.9-02`` read 24 MB/s with one reader. Version 9's weights warm-up
    (``scripts/build_kaggle_notebook.py``, ``start_warmup``) copied 18.6 GB at 469.2 MB/s with 8
    readers of 256 MiB pieces (``P4.7-02``). Same pattern here: every file is cut into pieces
    :math:`[k p, \min((k+1) p, n))`, each piece read in order through its own handle and written
    with ``os.pwrite`` into ``<name>.part``, renamed when its last piece is in and its size
    matches. Workers are daemon threads, so a read blocked on the mount cannot hold the process.
    At ``max_s``, or ``stall_s`` without a byte, the copy stops and every file not done becomes a
    symlink to its mount file: the install then reads those from the mount, as before.

    Args:
        source: The input folder on the mount.
        target: The local folder (removed first).
        workers: Parallel readers.
        piece_bytes: Bytes per piece.
        max_s: Most seconds for the copy.
        stall_s: Seconds without progress after which the copy stops.
        block_bytes: Bytes per read inside a piece.

    Returns:
        ``files``, ``bytes``, ``copied_files``, ``copied_bytes``, ``linked`` (names left on the
        mount), ``errors``, ``seconds``, ``mb_per_s``, ``stopped`` (``max_s``, ``stall``, or None).
    """
    import queue

    start = _now()
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for folder in sorted(source.rglob("*")):
        if folder.is_dir():
            (target / folder.relative_to(source)).mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(source.rglob("*")) if p.is_file()]
    sizes = {p: p.stat().st_size for p in files}
    pieces: queue.Queue[tuple[Path, int, int]] = queue.Queue()
    left = {}
    for path in files:
        rel = path.relative_to(source)
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        part = target / f"{rel}.part"
        with open(part, "wb") as handle:
            handle.truncate(sizes[path])
        offsets = list(range(0, sizes[path], piece_bytes)) or [0]
        left[path] = len(offsets)
        for offset in offsets:
            pieces.put((path, offset, min(piece_bytes, sizes[path] - offset)))
    lock = threading.Lock()
    stop = threading.Event()
    done: set[Path] = set()
    errors: dict[str, str] = {}
    progress = {"bytes": 0, "t": _now()}

    def finish(path: Path) -> None:
        rel = path.relative_to(source)
        part = target / f"{rel}.part"
        if part.stat().st_size != sizes[path]:
            raise OSError(f"{rel}: {part.stat().st_size} bytes, want {sizes[path]}")
        part.rename(target / rel)
        done.add(path)

    def work() -> None:
        while not stop.is_set():
            try:
                path, offset, length = pieces.get_nowait()
            except queue.Empty:
                return
            try:
                part = target / f"{path.relative_to(source)}.part"
                with open(path, "rb") as src:
                    src.seek(offset)
                    fd = os.open(part, os.O_WRONLY)
                    try:
                        at = offset
                        while at < offset + length and not stop.is_set():
                            data = src.read(min(block_bytes, offset + length - at))
                            if not data:
                                raise OSError(f"short read at {at} of {path.name}")
                            os.pwrite(fd, data, at)
                            at += len(data)
                            with lock:
                                progress["bytes"] += len(data)
                                progress["t"] = _now()
                    finally:
                        os.close(fd)
                if stop.is_set():
                    return
                with lock:
                    left[path] -= 1
                    if left[path] == 0 and not stop.is_set():
                        finish(path)
            except OSError as err:
                with lock:
                    errors[str(path.relative_to(source))] = _describe(err)

    threads = [threading.Thread(target=work, daemon=True) for _ in range(max(1, workers))]
    for thread in threads:
        thread.start()
    stopped = None
    while any(t.is_alive() for t in threads):
        with lock:
            idle = _now() - progress["t"]
            finished = len(done) == len(files)
        if finished:
            break
        if _now() - start > max_s:
            stopped = "max_s"
        elif idle > stall_s:
            stopped = "stall"
        if stopped:
            stop.set()
            break
        time.sleep(0.05)
    with lock:
        linked = []
        for path in files:
            if path in done:
                continue
            rel = path.relative_to(source)
            (target / f"{rel}.part").unlink(missing_ok=True)
            (target / rel).symlink_to(path)
            linked.append(str(rel))
        copied = sum(sizes[p] for p in done)
    seconds = _now() - start
    return {"files": len(files), "bytes": sum(sizes.values()), "copied_files": len(done),
            "copied_bytes": copied, "linked": linked, "errors": errors,
            "seconds": round(seconds, 1), "stopped": stopped,
            "mb_per_s": round(progress["bytes"] / 1e6 / max(seconds, 1e-9), 1)}


def stage_local_inputs(run: Run) -> dict[str, Any]:
    """Copy the install's inputs (the wheelhouse, ``P5.2-00``'s output) from the mount to disk.

    Why: the install then reads local disk (``P5.2-01-cpu6``'s mount read too slowly for the
    lock's 1,800 s); each copied input's path in ``state["paths"]`` becomes the local folder, the
    mount's is kept under ``state["mount_paths"]``. Without the room for a copy (its bytes plus
    ``reserve_gb``) or on an error, the input stays on the mount, the earlier behaviour.

    Args:
        run: The run (``local_inputs`` of the config, the found paths).

    Returns:
        Per input its :func:`copy_tree` record, or why it stayed on the mount.

    Raises:
        StageSkipped: If ``local_inputs.enabled`` is off.
    """
    spec = run.config.get("local_inputs") or {}
    if not spec.get("enabled"):
        raise StageSkipped("local_inputs.enabled is off")
    paths = run.state.get("paths", {})
    root = run.path(run.config["paths"]["scratch"]) / "inputs"
    records: dict[str, Any] = {}
    started = _now()
    for name in spec["inputs"]:
        folder = paths.get(name)
        if not folder:
            records[name] = {"local": False, "reason": "not found"}
            continue
        source = Path(folder)
        try:
            total = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
            root.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(root).free
            if free < total + float(spec["reserve_gb"]) * 1e9:
                records[name] = {"local": False, "bytes": total, "free_bytes": free,
                                 "reason": "no room for the copy and the reserve"}
                continue
            left = float(spec["max_s"]) - (_now() - started)
            record = copy_tree(source, root / name, int(spec["workers"]),
                               int(spec["piece_mib"]) << 20, max(left, 0.0),
                               float(spec["stall_s"]))
        except OSError as err:
            shutil.rmtree(root / name, ignore_errors=True)
            records[name] = {"local": False, "reason": _describe(err)}
            continue
        run.state.setdefault("mount_paths", {})[name] = folder
        paths[name] = str(root / name)
        records[name] = {"local": True, **record}
    run.state["local_inputs_ended_s"] = round(_now() - run.start, 3)
    write_json(run.out / "local_inputs.json", records)
    return {"inputs": records}


def stage_python(run: Run) -> dict[str, Any]:
    """Return the 3.12 interpreter: the image's, else ``P5.2-00``'s tarball unpacked."""
    spec = run.config["python"]
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    want = have if spec["want"] == "auto" else str(spec["want"])
    if have == want:
        run.state["python_path"] = sys.executable
        return {"python": sys.executable, "version": sys.version, "source": "image"}
    env_out = run.state.get("paths", {}).get("env_output")
    if not env_out:
        raise StageSkipped(f"image Python is {have}, want {want}, and no P5.2-00 output")
    tarball = Path(env_out) / spec["tarball"]
    if not tarball.is_file():
        raise RuntimeError(f"image Python is {have} and {tarball} is missing")
    dest = run.path(run.config["paths"]["scratch"]) / "python"
    with tarfile.open(tarball) as tar:
        tar.extractall(dest, filter="tar")
    hits = sorted(p for p in dest.rglob(f"python{want}") if p.parent.name == "bin")
    if not hits:
        raise RuntimeError(f"no python{want} in {tarball}")
    run.state["python_path"] = str(hits[0])
    return {"python": str(hits[0]), "source": "P5.2-00 tarball", "image_version": sys.version}


def _run_cmd(command: Sequence[str], timeout: float, env: Mapping[str, str]) -> dict[str, Any]:
    start = _now()
    try:
        done = subprocess.run([str(c) for c in command], capture_output=True, text=True,
                              timeout=timeout, env=dict(env), check=False)
        out = {"returncode": done.returncode, "stdout": _tail(done.stdout, 2000),
               "stderr": _tail(done.stderr, 2000)}
    except (OSError, subprocess.TimeoutExpired) as err:
        out = {"returncode": None, "stdout": "", "stderr": _describe(err)}
    out.update(command=shlex.join(str(c) for c in command), seconds=round(_now() - start, 2))
    return out


def apply_patch(site: Path, spec: Mapping[str, Any], context_chars: int = 1500,
                dry_run: bool = False) -> dict[str, Any]:
    """Apply one hash-guarded regex edit to an installed file (the recipe's R4).

    The same function is in ``scripts/fn_env.py``: each Kaggle script is a single file.

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


def stage_install(run: Run) -> dict[str, Any]:
    """Build the venv: the wheelhouse's lock, SGLang ``--no-deps``, extras, the arc26 wheel."""
    spec = run.config["install"]
    if not spec.get("enabled", True):
        run.state["venv_python"] = sys.executable
        raise StageSkipped("install disabled (the smoke runs in the current Python)")
    paths = run.state.get("paths", {})
    python = run.state.get("python_path")
    if not python or not paths.get("wheelhouse"):
        raise StageSkipped("no Python 3.12 or no wheelhouse")
    wheelhouse = Path(paths["wheelhouse"])
    scratch = run.path(run.config["paths"]["scratch"])
    uv_wheels = sorted(wheelhouse.rglob("uv-*.whl"))
    if len(uv_wheels) != 1:
        raise RuntimeError(f"expected one uv wheel, found {[w.name for w in uv_wheels]}")
    uv = scratch / "bin" / "uv"
    with zipfile.ZipFile(uv_wheels[0]) as archive:
        members = [n for n in archive.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        if len(members) != 1:
            raise RuntimeError(f"no single uv executable in {uv_wheels[0].name}")
        uv.parent.mkdir(parents=True, exist_ok=True)
        uv.write_bytes(archive.read(members[0]))
    uv.chmod(0o755)
    venv = scratch / "venv"
    venv_python = venv / "bin" / "python"
    wheels = sorted(p for p in wheelhouse.rglob("wheels") if p.is_dir())
    wheels_dir = wheels[0] if wheels else wheelhouse
    lock = sorted(wheelhouse.rglob(spec["lock"]))
    sglang = sorted(wheels_dir.glob(spec["sglang_glob"]))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
    timeout = float(spec["timeout_s"])
    links = ["--find-links", str(wheels_dir)]
    extra_dirs = []
    for name in ("env_output", "fa2_output"):
        if paths.get(name) and (Path(paths[name]) / "wheels").is_dir():
            extra_dirs.append(Path(paths[name]) / "wheels")
    for folder in extra_dirs:
        links += ["--find-links", str(folder)]
    install = [str(uv), "pip", "install", "--python", str(venv_python), "--no-index", *links]
    commands = []
    if not venv_python.exists():
        commands.append(_run_cmd([str(uv), "venv", "--python", python, str(venv)], 300, env))
    if lock:
        commands.append(_run_cmd([*install, "-r", str(lock[0])], timeout, env))
    if len(sglang) == 1:
        commands.append(_run_cmd([*install, "--reinstall", "--no-deps", str(sglang[0])], 600, env))
    for pattern in spec.get("extra_wheels", []):
        for folder in extra_dirs:
            for wheel in sorted(folder.glob(pattern)):
                commands.append(_run_cmd([*install, "--no-deps", str(wheel)], 600, env))
    code = paths.get("code")
    arc26 = sorted(Path(code).glob("arc26-*.whl")) if code else []
    if len(arc26) == 1:
        commands.append(_run_cmd([*install, "--no-deps", str(arc26[0])], 300, env))
    failed = [c for c in commands if c["returncode"] != 0]
    freeze = _run_cmd([str(uv), "pip", "freeze", "--python", str(venv_python)], 120, env)
    (run.out / "versions.txt").write_text(freeze["stdout"], encoding="utf-8")
    write_json(run.out / "install.json", {"commands": commands})
    if failed or not venv_python.exists() or len(arc26) != 1:
        raise RuntimeError(f"install failed: {[c['command'][:120] for c in failed]}, "
                           f"arc26 wheels {len(arc26)}")
    run.state["venv_python"] = str(venv_python)
    run.state["uv"] = str(uv)
    patches = apply_patches(run, venv)
    return {"venv_python": str(venv_python), "commands": len(commands),
            "seconds_by_command": [c["seconds"] for c in commands], "patches": patches}


def apply_patches(run: Run, venv: Path) -> list[dict[str, Any]]:
    """Apply the config's guarded edits; on a failed one, set its fallback flags instead.

    Args:
        run: The run (``patches`` in its config; ``flag_overrides`` set in its state).
        venv: The venv folder.

    Returns:
        One result per edit (its context is written beside the other outputs).
    """
    site = sorted(venv.glob("lib/python*/site-packages"))
    results = []
    overrides = dict(run.state.get("flag_overrides", {}))
    env_fallback = dict(run.state.get("env_fallback", {}))
    spec = run.config.get("patches", {})
    items = spec.get("items", [])
    # The KV4 group (revision 20): every edit from patches.kv4_from on. Before its first edit,
    # every file the group edits is copied as it stands (revision 16's five edits applied), so a
    # KV4 fault restores revision 16's engine byte for byte (restore_kv4).
    names = [i["name"] for i in items]
    first = names.index(spec["kv4_from"]) if spec.get("kv4_from") in names else len(items)
    failed_kv4 = False
    for index, item in enumerate(items):
        if index == first and site:
            snapshot_kv4(run, site[0], items[first:])
        result = apply_patch(site[0], item) if site else {"name": item["name"], "status": "missing"}
        context = result.pop("context", "")
        (run.out / f"patch_{item['name']}_context.txt").write_text(context, encoding="utf-8")
        if result["status"] not in {"applied", "already applied"}:
            overrides.update(item.get("fallback_flags", {}))
            result["fallback_flags"] = item.get("fallback_flags", {})
            # An edit whose path is chosen by a variable (the KV4 cache, ARC26_KV4) turns its
            # path off instead (P5.2-01 revision 17): stage_cuda applies these last.
            if item.get("fallback_env"):
                env_fallback.update({str(k): str(v) for k, v in item["fallback_env"].items()})
                result["fallback_env"] = item["fallback_env"]
            if index >= first:
                failed_kv4 = True
        results.append(result)
    run.state["flag_overrides"] = overrides
    run.state["env_fallback"] = env_fallback
    write_json(run.out / "patches.json", results)
    if failed_kv4:
        # The group's other edits came out with it: revision 16's files, ARC26_KV4 off.
        restore_kv4(run, "a KV4 edit did not apply")
    return results


def snapshot_kv4(run: Run, site: Path, items: Sequence[Mapping[str, Any]]) -> None:
    """Copy every file the KV4 edits touch, before the first of them, into the scratch folder.

    Records ``run.state["kv4_snapshot"]``: the site, the folder, and each file's sha256.
    """
    folder = run.path(run.config["paths"]["scratch"]) / "kv4_snapshot"
    shutil.rmtree(folder, ignore_errors=True)
    files = {}
    for rel in dict.fromkeys(i["file"] for i in items):
        source = site / rel
        if source.is_file():
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files[rel] = sha256_file(source)
    run.state["kv4_snapshot"] = {"site": str(site), "folder": str(folder), "files": files}


def restore_kv4(run: Run, reason: str) -> dict[str, Any]:
    """Undo the KV4 edits: write back the snapshot's files and turn ``ARC26_KV4`` off.

    Why: version 15 (P5.2-01 revision 19) failed at the import of a KV4-edited module in every
    boot, whatever ARC26_KV4 was, so switching the variable off was not enough; restoring the
    files gives revision 16's engine byte for byte. Idempotent; recorded in ``kv4_restore.json``
    and ``state.json``.

    Args:
        run: The run (``kv4_snapshot`` in its state).
        reason: Why.

    Returns:
        The record: the files restored with their sha256 and whether each matches the snapshot.
    """
    snap = run.state.get("kv4_snapshot") or {}
    record: dict[str, Any] = {"t": round(_now() - run.start, 1), "reason": reason, "files": {}}
    for rel, digest in (snap.get("files") or {}).items():
        target = Path(snap["site"]) / rel
        source = Path(snap["folder"]) / rel
        if source.is_file():
            shutil.copy2(source, target)
            record["files"][rel] = sha256_file(target) == digest
    for key in ("env_fallback", "boot_env"):
        run.state.setdefault(key, {})["ARC26_KV4"] = "0"
    if isinstance(run.state.get("env_overrides"), dict):
        run.state["env_overrides"]["ARC26_KV4"] = "0"
    # boot_env reaches every later boot (launch_server) and the import check; the process's own
    # environment is left as it is.
    run.state.setdefault("kv4_restores", []).append(record)
    append_jsonl(run.out / "kv4_restore.jsonl", record)
    return record


def stage_cuda(run: Run) -> dict[str, Any]:
    """Point the bundled CUDA 13 toolkit at the driver and compilers (dfranzen's cell 12)."""
    venv_python = run.state.get("venv_python")
    # "{code}" in a value is the code dataset's folder as found (the mount path differs from the
    # configured one on Kaggle: /kaggle/input/arc26-code-fn in revision 15's run).
    code = run.state.get("paths", {}).get("code") or ""
    env = {str(k): str(v).replace("{code}", code)
           for k, v in run.config["engine"]["env"].items()}
    scratch = run.path(run.config["paths"]["scratch"])
    cache = scratch / "cache"
    for name in ("huggingface", "torch", "triton", "tilelang", "flashinfer", "cuda", "sglang/jit"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env.update({
        "HF_HOME": str(cache / "huggingface"), "XDG_CACHE_HOME": str(cache),
        "TORCH_HOME": str(cache / "torch"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "TILELANG_CACHE_DIR": str(cache / "tilelang"),
        "CUDA_CACHE_PATH": str(cache / "cuda"), "FLASHINFER_WORKSPACE_BASE":
        str(cache / "flashinfer"), "SGLANG_CACHE_DIR": str(cache / "sglang"),
        "SGLANG_JIT_CACHE_DIR": str(cache / "sglang/jit"),
    })
    result: dict[str, Any] = {}
    if venv_python and venv_python != sys.executable:
        venv = Path(venv_python).parent.parent
        found = sorted(venv.glob("lib/python*/site-packages/nvidia/cu1*"))
        if found:
            home = found[-1]
            lib, lib64 = home / "lib", home / "lib64"
            if not lib64.exists() and not lib64.is_symlink():
                lib64.symlink_to("lib")
            for so in sorted(lib.glob("*.so.*")):
                link = lib / re.sub(r"\.so\..*$", ".so", so.name)
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(so.name)
            if not (lib / "libcuda.so").exists():
                for folder in DRIVER_DIRS:
                    hits = sorted(p for p in Path(folder).glob("libcuda.so*") if p.is_file())
                    if hits:
                        (lib / "libcuda.so").symlink_to(hits[0].resolve())
                        break
            env.update({
                "CUDA_HOME": str(home), "CUDACXX": str(home / "bin" / "nvcc"),
                "PATH": f"{home}/bin:{Path(venv_python).parent}:" + os.environ.get("PATH", ""),
                "LD_LIBRARY_PATH": f"{home}/lib64:{home}/lib:"
                + os.environ.get("LD_LIBRARY_PATH", ""),
            })
            result["cuda_home"] = str(home)
        cxx = next((c for c in ("g++-14", "g++-13", "g++-12", "g++-11", "g++")
                    if shutil.which(c)), None)
        if cxx:
            env.update(CXX=cxx, CC=cxx.replace("g++", "gcc"), CUDAHOSTCXX=cxx)
        result["cxx"] = cxx
    # A failed edit's variables last (apply_patches): its path stays off in every boot.
    env.update(run.state.get("env_fallback", {}))
    # Only the values this run sets are kept (state.json is pulled; the session's own
    # environment may hold tokens and is never written).
    run.state["env_overrides"] = env
    return {**result, "overrides": sorted(env)}


# ---- inner stages -------------------------------------------------------------------------------

COMPETITION = "competition"
DEVELOPMENT = "development"
# Words of an error that point at the engine, not the request: a dead or restarting server.
ENGINE_ERRORS = ("refused", "reset", "RemoteDisconnected", "BrokenPipe", "HTTP 50", "Connection",
                 "timed out")


def run_mode(config: Mapping[str, Any], environ: Mapping[str, str]) -> str:
    """Return ``competition`` or ``development`` for this session.

    Why: Kaggle sets ``KAGGLE_IS_COMPETITION_RERUN`` only in the scored rerun of a submitted
    version; every other session (the version's own run, the CPU twin) is development mode, which
    reads evaluation tasks for local scoring. ``mode`` in the config overrides unless ``auto``.

    Args:
        config: The run's config.
        environ: The environment.

    Returns:
        The mode.
    """
    mode = config.get("mode", "auto")
    if mode in (COMPETITION, DEVELOPMENT):
        return mode
    flag = environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
    return COMPETITION if flag not in ("", "0", "false", "no") else DEVELOPMENT


@dataclass
class Inner:
    """State of the inner process: the server, the tokenizer, the tasks, the readings."""

    run: Run
    base: str
    tok: Any
    mode: str = DEVELOPMENT
    proc: subprocess.Popen[bytes] | None = None
    monitor: Monitor | None = None
    pool: int | None = None
    max_running: int | None = None
    tasks: tuple[Any, ...] = ()
    openings: dict[str, int] = field(default_factory=dict)
    plan_openings: list[int] = field(default_factory=list)
    challenges: dict[str, Any] = field(default_factory=dict)
    context: int = 86016
    plan: dict[str, Any] | None = None
    rates: list[float] = field(default_factory=list)
    gates: dict[str, Gate] = field(default_factory=dict)
    restarts: int = 0
    # The engine instance: launch_server counts each process it starts (P5.6-02, the stale
    # health probe of P5.6-01's restart 2), so a failure seen on one instance never kills the next.
    generation: int = 0
    # The solve's health watch of the current instance (None outside the solve).
    watch: Any = None
    engine_dead: bool = False
    tool_ok: bool | None = None
    verifier: Any = None
    writer: SubmissionWriter | None = None
    engine_lock: threading.Lock = field(default_factory=threading.Lock)
    plan_lock: threading.Lock = field(default_factory=threading.Lock)
    engine_ready: threading.Event = field(default_factory=threading.Event)
    arg_refused: list[str] = field(default_factory=list)


def time_plan(config: Mapping[str, Any], mode: str) -> dict[str, float]:
    """Return the mode's generation stop and end, in seconds after the session start.

    Args:
        config: The run's config (``time_plan``).
        mode: ``competition`` or ``development``.

    Returns:
        ``{"generation_stop_s", "end_s", "min_start_s"}``.
    """
    return {k: float(v) for k, v in config["time_plan"][mode].items()}


def server_values(run: Run) -> dict[str, str]:
    """Return the placeholder values of the server's flags (model, port, chat template, ...)."""
    paths = run.state["paths"]
    model = paths.get("model") or ""
    template = Path(model) / "chat_template.jinja" if model else Path("")
    return {"model": model, "port": str(run.config["engine"]["port"]),
            "scratch": run.state.get("scratch", ""), "chat_template": str(template),
            "drafter": paths.get("drafter") or ""}


def boot_rungs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the boot ladder after the first boot: each rung's flags and variables.

    Why: P5.2-01 revision 17 boots the KV4 cache first; a failure there (the boot, or the first
    generation) must fall back to revision 16's FP8 path (``ARC26_KV4=0``) before the BF16 eager
    one. ``engine.boot_ladder`` lists the rungs in order, each ``{"flags": {...}, "env":
    {...}}``, cumulative (a rung keeps the earlier rungs' changes); without it the one rung is
    ``engine.boot_retry_flags``, as up to revision 16.

    Args:
        config: The run's config.

    Returns:
        The rungs, each with ``flags`` and ``env`` mappings.
    """
    engine = config["engine"]
    ladder = engine.get("boot_ladder")
    if ladder is None:
        retry = engine.get("boot_retry_flags") or {}
        ladder = [{"flags": retry}] if retry else []
    # A variable set to null removes it for the rung's boot (P5.6-01's rung 1 takes 8B's
    # variables back out); the rest are text.
    return [{"flags": dict(r.get("flags") or {}), "env": {str(k): None if v is None else str(v)
                                                          for k, v in (r.get("env") or {}).items()}}
            for r in ladder]


def next_rung(run: Run, reason: str) -> dict[str, Any] | None:
    """Move the run to the next rung of the boot ladder; return it, or None when none is left.

    Its flags join ``flag_overrides`` and its variables ``boot_env`` (both cumulative); the row is
    appended to ``restarts.jsonl``.
    """
    rungs = boot_rungs(run.config)
    index = int(run.state.get("rung", 0))
    if index >= len(rungs):
        return None
    rung = rungs[index]
    run.state["rung"] = index + 1
    run.state.setdefault("flag_overrides", {}).update(rung["flags"])
    run.state.setdefault("boot_env", {}).update(rung["env"])
    if rung["env"].get("ARC26_KV4") == "0" and run.state.get("kv4_snapshot"):
        restore_kv4(run, f"the ladder's rung {index + 1}: {reason}")
    append_jsonl(run.out / "restarts.jsonl", {"t": round(_now() - run.start, 1),
                                              "reason": reason, "rung": index + 1,
                                              "flags": rung["flags"], "env": rung["env"]})
    return rung


def boot_environment(run: Run) -> dict[str, str]:
    """Return the boot's environment: the process's, with ``boot_env`` (None removes a key)."""
    env = dict(os.environ)
    for key, value in run.state.get("boot_env", {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def launch_server(inner: Inner, overrides: Mapping[str, Any] | None = None) -> float:
    """Start the server and wait for ``/health``; return the seconds it took.

    Args:
        inner: The inner state (``proc`` set here).
        overrides: Flag values replacing the config's on top of the edit's fallback flags.

    Returns:
        Seconds to healthy.

    Raises:
        RuntimeError: If it exits or is not healthy within the boot timeout (bounded by the time
            left before the generation stop).
    """
    run = inner.run
    config = run.config
    engine = config["engine"]
    flags = {**run.state.get("flag_overrides", {}), **(overrides or {})}
    argv = server_argv(config, sys.executable, server_values(run), flags)
    log_path = run.path(config["paths"]["engine_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    boot_env = run.state.get("boot_env", {})
    append_jsonl(run.out / "engine_commands.jsonl", {"t": round(_now() - run.start, 1),
                                                       "argv": argv, "env": boot_env})
    start = _now()
    with open(log_path, "ab", buffering=0) as handle:
        inner.proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                      env=boot_environment(run), start_new_session=True)
    inner.generation += 1
    stop = run.start + time_plan(config, inner.mode)["generation_stop_s"]
    timeout = min(float(engine["boot_timeout_s"]), max(stop - _now(), 0))
    while _now() - start < timeout:
        if inner.proc.poll() is not None:
            break
        try:
            status, _ = http_json(inner.base, "GET", "/health", timeout=5)
            if status == 200:
                inner.engine_ready.set()
                return _now() - start
        except OSError:
            pass
        time.sleep(float(engine.get("poll_s", 3)))
    raise RuntimeError(f"server not healthy after {round(_now() - start)} s "
                       f"(exit {inner.proc.poll()}); see engine_tail.log")


def kill_server(inner: Inner) -> None:
    """Stop the server's process group (SIGTERM, then SIGKILL after 60 s)."""
    inner.engine_ready.clear()
    proc = inner.proc
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, 15)
        proc.wait(timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, 9)
            proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass


def read_limits(inner: Inner) -> dict[str, Any]:
    """Read the pool and the request limit from ``/get_server_info``, else the server log."""
    run = inner.run
    log_text = run.path(run.config["paths"]["engine_log"]).read_text(
        encoding="utf-8", errors="replace")
    readings = parse_log(log_text, run.config["engine"]["log_patterns"])
    try:
        _, info = http_json(inner.base, "GET", "/get_server_info", timeout=30)
    except OSError as err:
        info = {"error": _describe(err)}
    write_json(run.out / "server_info.json", info)
    pool = find_key(info, "max_total_num_tokens")
    running = find_key(info, "max_running_requests")
    if pool is None and readings.get("max_total_num_tokens"):
        pool = min(readings["max_total_num_tokens"])
    if running is None and readings.get("max_running_requests"):
        running = min(readings["max_running_requests"])
    inner.pool = None if pool is None else int(pool)
    inner.max_running = None if running is None else int(running)
    return readings


# The child of the argument check (stage_arg_check): standard library plus the checked module.
# It prints one marked JSON line, so the module's own prints cannot be mistaken for the verdict.
ARG_CHECK_MARK = "ARC26_ARG_CHECK "
ARG_CHECK_CHILD = r"""
import argparse, importlib, json, os, re, sys, traceback
module_name, argv, code_dir = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
sys.path.insert(0, code_dir)
DEVICE = re.compile(r"cuda|nvidia|nvml|gpu|device|driver|nccl|accelerator", re.I)
MARK = "ARC26_ARG_CHECK "


def report(record):
    print(MARK + json.dumps(record), flush=True)


phase = "import"
try:
    module = importlib.import_module(module_name)
    scope = os.path.dirname(os.path.abspath(module.__file__)) + os.sep
    phase = "parse"
    if hasattr(module, "prepare_server_args"):
        args = module.prepare_server_args(argv)
    else:
        parser = argparse.ArgumentParser()
        module.ServerArgs.add_cli_args(parser)
        args = module.ServerArgs.from_cli_args(parser.parse_args(argv))
    phase = "resolve"
    resolve = getattr(args, "resolve_once", None)
    if callable(resolve):
        resolve()
    report({"verdict": "accepted", "phase": "done", "resolve_once": callable(resolve)})
except SystemExit as err:
    # argparse exits 2 on an unknown flag or a value outside its choices (message on stderr).
    bad = phase == "parse" and err.code not in (0, None)
    report({"verdict": "refused" if bad else "error", "phase": phase,
            "error": f"SystemExit: {err.code}"})
except BaseException as err:
    frames = traceback.extract_tb(err.__traceback__)
    text = f"{type(err).__name__}: {err}"
    inside = [f for f in frames if phase != "import"
              and os.path.abspath(f.filename).startswith(scope)]
    raised_inside = bool(frames) and bool(inside) and frames[-1] is inside[-1]
    try:
        import torch
        gpu = bool(torch.cuda.is_available())
    except BaseException:
        gpu = False
    device = bool(DEVICE.search(text))
    if isinstance(err, ValueError) and raised_inside and (gpu or not device):
        verdict = "refused"
    elif device:
        verdict = "device_stop"
    else:
        verdict = "error"
    report({"verdict": verdict, "phase": phase, "error": text[:2000], "gpu": gpu,
            "stopped_in": (f"{os.path.relpath(inside[-1].filename, scope)}:{inside[-1].lineno} "
                           f"{inside[-1].name}") if inside else None,
            "traceback": traceback.format_exc()[-3000:]})
"""


def check_arguments(python: str, module: str, argv: Sequence[str], timeout: float,
                    env: Mapping[str, str]) -> dict[str, Any]:
    """Run the fork's argument parsing and resolution on ``argv`` in a child process.

    Why: P5.2-03 (Modal, 2 x L40S, 29 Oct) lost both boots to a ``ValueError`` that the fork's
    ``ServerArgs`` raises during resolution, before any GPU work; the same resolution on CPU
    finds such a refusal in the free twin. A child process, so no CUDA context of the check
    outlives it into the server's memory.

    Args:
        python: The venv's interpreter.
        module: The module holding ``prepare_server_args`` (``sglang.srt.server_args``).
        argv: The server's flags (no interpreter or module).
        timeout: Seconds before the child's process group is killed.
        env: The child's environment (the server's).

    Returns:
        ``verdict``: ``accepted``; ``refused`` (a ``ValueError`` raised inside the module's
        package, or the parser's exit), the failure the check exists for; ``device_stop`` (the
        resolution stopped at a CUDA or device query, as on a CPU box: the flags before it were
        accepted, the rest unverified); ``error`` (anything else, or no verdict line). Plus the
        phase, the error, where it stopped, and the child's output tail.
    """
    start = _now()
    proc = subprocess.Popen([python, "-c", ARG_CHECK_CHILD, module, json.dumps(list(argv)),
                             str(HERE)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(env), start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
        output, _ = proc.communicate()
        return {"verdict": "error", "error": f"no verdict in {timeout} s",
                "seconds": round(_now() - start, 1), "output_tail": _tail(output)}
    text = output.decode("utf-8", errors="replace")
    record: dict[str, Any] = {"verdict": "error", "error": f"no verdict (exit {proc.returncode})"}
    for line in text.splitlines():
        if line.startswith(ARG_CHECK_MARK):
            record = json.loads(line[len(ARG_CHECK_MARK):])
    record["seconds"] = round(_now() - start, 1)
    record["output_tail"] = _tail("\n".join(
        line for line in text.splitlines() if not line.startswith(ARG_CHECK_MARK)))
    return record


def stage_schema_check(inner: Inner) -> dict[str, Any]:
    """Compile and validate the forcing grammar offline before starting an engine."""
    run = inner.run
    spec = run.config.get("schema_check") or {}
    if not spec.get("enabled"):
        raise StageSkipped("schema_check.enabled is off")
    report = run.out / "schema_check.json"
    command = [sys.executable, "-m", "arc26.agent.grammar_check", "--model",
               str(run.state["paths"]["model"]), "--report", str(report)]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=float(spec.get("timeout_s", 600)),
                            env=boot_environment(run))
    (run.out / "schema_check.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    record = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
    if result.returncode or record.get("ok") is not True:
        raise RuntimeError(f"schema preflight failed: {record.get('error', result.stderr[-1000:])}")
    return {"ok": True, "xgrammar_version": record["xgrammar_version"],
            "schemas": list(record["schemas"]), "cuda_initialized": record["cuda_initialized"]}


def stage_arg_check(inner: Inner) -> dict[str, Any]:
    """Resolve the boot's flags, and the fallback boot's, with the fork's own ``ServerArgs``.

    Why: a flag the fork refuses for this model (P5.2-03's ``--language-model-only``) fails
    every boot in seconds, after the install and the queue; resolving the flags here names the
    refusal in the stage table, in the CPU twin before any GPU quota and in the GPU run before
    the boot log. Development mode reads a refusal as a failed run (exit 1); competition mode
    only records it, and the boot decides, as for the first generation.

    Args:
        inner: The inner state (the run's config and state).

    Returns:
        ``status`` ``failed`` when a flag set is refused or the check errs, else ``ok``; per
        flag set its verdict (:func:`check_arguments`). Also written to ``arg_check.json``.

    Raises:
        StageSkipped: If ``arg_check.enabled`` is off.
    """
    run = inner.run
    config = run.config
    spec = config.get("arg_check") or {}
    if not spec.get("enabled"):
        raise StageSkipped("arg_check.enabled is off")
    engine = config["engine"]
    drop = set(spec.get("drop_flags") or [])
    base = {k: v for k, v in {**engine["flags"], **run.state.get("flag_overrides", {})}.items()
            if k not in drop}
    variants: dict[str, Mapping[str, Any]] = {"boot": {}}
    if "retry_flags" in spec:
        if spec["retry_flags"]:
            variants["boot_retry"] = spec["retry_flags"]
    else:
        # Every distinct flag set of the boot ladder (a rung that changes only variables has
        # the previous rung's flags): boot_retry, boot_retry_2, ...
        cumulative: dict[str, Any] = {}
        for rung in boot_rungs(config):
            if not rung["flags"] or all(cumulative.get(k) == v for k, v in rung["flags"].items()):
                continue
            cumulative = {**cumulative, **rung["flags"]}
            name = "boot_retry" if "boot_retry" not in variants else f"boot_retry_{len(variants)}"
            variants[name] = dict(cumulative)
    stress = config.get("stress") or {}
    if stress.get("enabled") and stress.get("fallback_flags"):
        # Revision 22: the stress check's fallback set (the smaller prefill), so a flag the fork
        # does not know is refused here, never dropped.
        variants["stress_fallback"] = dict(stress["fallback_flags"])
    values = server_values(run)
    results = {}
    for name, overrides in variants.items():
        argv = server_flags({**base, **overrides}, values)
        results[name] = {"argv": argv, **check_arguments(
            sys.executable, spec["module"], argv, float(spec["timeout_s"]), os.environ)}
    verdicts = {name: r["verdict"] for name, r in results.items()}
    refused = [name for name, v in verdicts.items() if v == "refused"]
    inner.arg_refused = refused
    record = {"status": "failed" if refused or "error" in verdicts.values() else "ok",
              "module": spec["module"], "verdicts": verdicts,
              "errors": {name: r.get("error") for name, r in results.items() if r.get("error")}}
    write_json(run.out / "arg_check.json", {**record, "results": results})
    return record


# The child of the import check (stage_import_check): imports each module and prints one marked
# JSON line, as the argument check's child does.
IMPORT_CHECK_MARK = "ARC26_IMPORT_CHECK "
IMPORT_CHECK_CHILD = r"""
import importlib, json, re, sys, traceback
modules, code_dir = json.loads(sys.argv[1]), sys.argv[2]
sys.path.insert(0, code_dir)
DEVICE = re.compile(r"cuda|nvidia|nvml|gpu|device|driver|nccl|accelerator|libcuda", re.I)
out = {}
for name in modules:
    try:
        importlib.import_module(name)
        out[name] = {"status": "ok"}
    except BaseException as err:
        text = f"{type(err).__name__}: {err}"
        out[name] = {"status": "error", "error": text[:2000], "device": bool(DEVICE.search(text)),
                     "traceback": traceback.format_exc()[-3000:]}
try:
    import torch
    gpu = bool(torch.cuda.is_available())
except BaseException:
    gpu = False
print("ARC26_IMPORT_CHECK " + json.dumps({"modules": out, "gpu": gpu}), flush=True)
"""
KV4_OFF = ("", "0", "off", "false")


def import_modules(modules: Sequence[str], env: Mapping[str, str], timeout: float
                   ) -> dict[str, Any]:
    """Import ``modules`` in a child process with ``env``; return each one's status.

    Returns:
        ``modules`` (name to ``status`` ``ok`` or ``error``, with the error, a device flag, and
        the traceback's tail), ``gpu``, ``failed`` (the errors that count: every error on a GPU
        box, and on a CPU box those that do not name a device, as the argument check's
        ``device_stop``), ``seconds``.
    """
    start = _now()
    proc = subprocess.Popen([sys.executable, "-c", IMPORT_CHECK_CHILD, json.dumps(list(modules)),
                             str(HERE)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(env), start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
        output, _ = proc.communicate()
        return {"modules": {}, "gpu": None, "failed": ["(timeout)"],
                "error": f"no verdict in {timeout} s", "seconds": round(_now() - start, 1),
                "output_tail": _tail(output)}
    text = output.decode("utf-8", errors="replace")
    record: dict[str, Any] = {"modules": {}, "gpu": None, "error": f"no verdict (exit "
                                                                     f"{proc.returncode})"}
    for line in text.splitlines():
        if line.startswith(IMPORT_CHECK_MARK):
            record = json.loads(line[len(IMPORT_CHECK_MARK):])
    found = record.get("modules") or {}
    record["failed"] = [m for m in modules if (found.get(m) or {}).get("status") != "ok"
                        and (record.get("gpu") or not (found.get(m) or {}).get("device"))]
    record["seconds"] = round(_now() - start, 1)
    record["output_tail"] = _tail("\n".join(
        line for line in text.splitlines() if not line.startswith(IMPORT_CHECK_MARK)), 2000)
    return record


def stage_import_check(inner: Inner) -> dict[str, Any]:
    """Import the edited engine modules as the boot will, before the boot (revision 20).

    Why: version 15 (P5.2-01 revision 19) applied every edit, yet the edited
    ``kv_cache_configurator.py`` failed to import in all three boots (a decorator wrapped the
    KV4 helper), and the CPU twin's fake server never imports the engine. Here a child process,
    with the boot's environment (``ARC26_KV4``, ``PYTHONPATH``), imports ``import_check.modules``
    and, with KV4 on, ``import_check.kv4_modules``. A failure with KV4 on restores revision 16's
    files and turns KV4 off (``restore_kv4``), then checks again without KV4.

    Returns:
        ``kv4``, the first check, and after a fallback ``restored`` and ``after_restore``;
        ``status`` failed only when the engine's own modules fail without KV4.

    Raises:
        StageSkipped: If ``import_check.enabled`` is off.
    """
    run = inner.run
    spec = run.config.get("import_check") or {}
    if not spec.get("enabled"):
        raise StageSkipped("import_check.enabled is off")
    timeout = float(spec["timeout_s"])

    def env() -> dict[str, str]:
        return boot_environment(run)

    kv4 = env().get("ARC26_KV4", "0").strip().lower() not in KV4_OFF
    modules = list(spec["modules"]) + (list(spec.get("kv4_modules") or []) if kv4 else [])
    first = import_modules(modules, env(), timeout)
    record: dict[str, Any] = {"kv4": kv4, "check": first}
    final = first
    if first["failed"] and kv4:
        record["restored"] = restore_kv4(run, f"import check failed: {first['failed']}")
        final = import_modules(list(spec["modules"]), env(), timeout)
        record["after_restore"] = final
    write_json(run.out / "import_check.json", record)
    if final["failed"]:
        raise RuntimeError(f"the engine's modules fail to import: {final['failed']}")
    return {"kv4": kv4, "failed_first": first["failed"], "fell_back": "restored" in record,
            "kv4_after": env().get("ARC26_KV4", "0")}


def stage_boot(inner: Inner) -> dict[str, Any]:
    """Start the server, wait for health, and read the pool and limits from its log and info."""
    run = inner.run
    config = run.config
    inner.monitor = Monitor(run.out, inner.base, float(config["monitor"]["interval_s"]),
                            config["monitor"]["metrics"])
    inner.monitor.start()
    try:
        ready = launch_server(inner)
    except RuntimeError as err:
        # The next rungs of the boot ladder before the run gives up (boot_rungs: revision 17's
        # FP8 path without KV4, then the recipe's R7, BF16 KV and no CUDA graphs).
        while True:
            if next_rung(run, f"boot: {err}") is None:
                raise
            kill_server(inner)
            try:
                ready = launch_server(inner)
                break
            except RuntimeError as again:
                err = again
    readings = read_limits(inner)
    inner.monitor.sample()
    boot = {
        "ready_s": round(ready, 1), "exit_code": inner.proc.poll(), "pool_tokens": inner.pool,
        "max_running_requests": inner.max_running, "log": readings,
        "flag_overrides": run.state.get("flag_overrides", {}),
        "gpu_peak_used_mib": inner.monitor.peak_used_mib,
        "ram_available_gib": read_meminfo().get("MemAvailable"),
        "since_start_s": round(_now() - run.start, 1),
    }
    write_json(run.out / "boot.json", boot)
    return boot


def engine_alive(inner: Inner, timeout: float | None = None) -> bool:
    """Whether the server process runs and answers one ``/health`` probe.

    Args:
        inner: The inner state.
        timeout: The probe's seconds; None: ``health.probe_timeout_s`` (10 when unset, the value
            up to P5.6-01).
    """
    if inner.proc is None or inner.proc.poll() is not None:
        return False
    if timeout is None:
        timeout = float(inner.run.config.get("health", {}).get("probe_timeout_s", 10))
    try:
        status, _ = http_json(inner.base, "GET", "/health", timeout=timeout)
    except OSError:
        return False
    return status == 200


def engine_responsive(inner: Inner) -> bool:
    """Whether the server answers any of ``health.kill_probes`` probes in a row.

    Why: P5.6-01's restart 2 killed an engine that was stalled for about 10 s at 0.05 GiB free
    on a single 10 s probe (``docs/experiments/P5.6-01-forensics.md``); a kill now needs every
    one of ``kill_probes`` probes (``probe_timeout_s`` each) to fail. A process that exited is
    dead at once.

    Returns:
        True on the first answered probe; False when the process exited or every probe failed.
    """
    spec = inner.run.config.get("health", {})
    for _ in range(max(1, int(spec.get("kill_probes", 1)))):
        if inner.proc is None or inner.proc.poll() is not None:
            return False
        if engine_alive(inner):
            return True
    return False


def restart_engine(inner: Inner, reason: str, seen: int | None = None) -> bool:
    """Restart a dead server once more if the time and the restart budget allow.

    Why: the version 10 review found that a server that dies mid-run zeroes every task after it;
    a restart costs one boot (with a warm page cache and compile caches, shorter than the first)
    and the tasks in flight are queued again. P5.6-02: a caller names the engine instance it saw
    fail (``seen``, :attr:`Inner.generation`); when another instance has started since, the
    caller's failure is not this engine's, so it is neither probed nor killed (P5.6-01's restart 2
    was a thread waiting on the lock since restart 1).

    Args:
        inner: The inner state.
        reason: Recorded in ``restarts.jsonl``.
        seen: The instance the caller saw fail; None: the current one.

    Returns:
        True when the server is healthy afterwards (also when another thread restarted it).
    """
    run = inner.run
    with inner.engine_lock:
        if seen is not None and seen != inner.generation:
            if not inner.engine_dead:
                inner.engine_ready.set()
            return not inner.engine_dead
        if engine_responsive(inner):
            inner.engine_ready.set()
            return True
        plan = time_plan(run.config, inner.mode)
        left = run.start + plan["generation_stop_s"] - _now()
        spec = run.config["health"]
        row: dict[str, Any] = {"t": round(_now() - run.start, 1), "reason": reason,
                               "restart": inner.restarts + 1}
        if inner.engine_dead:
            return False
        if inner.restarts >= int(spec["max_restarts"]) or left < float(spec["min_left_s"]):
            row["skipped"] = "restart budget or time used"
            append_jsonl(run.out / "restarts.jsonl", row)
            inner.engine_dead = True
            inner.engine_ready.clear()
            return False
        kill_server(inner)
        inner.restarts += 1
        try:
            row["ready_s"] = round(launch_server(inner), 1)
        except RuntimeError as err:
            row["error"] = _describe(err)
            append_jsonl(run.out / "restarts.jsonl", row)
            inner.engine_dead = True
            inner.engine_ready.clear()
            return False
        read_limits(inner)
        row["generation"] = inner.generation
        append_jsonl(run.out / "restarts.jsonl", row)
        if inner.watch is not None:
            # The new instance's own watch; the old one exits on its next poll.
            inner.watch.stop.set()
            inner.watch = HealthWatch(inner)
            inner.watch.start()
        return True


class HealthWatch(threading.Thread):
    """Polls one engine instance; after ``failures`` misses in a row, :func:`restart_engine` runs.

    The watch is tied to the instance it started on (:attr:`Inner.generation`): when another
    instance has started, it exits without probing it (the restart started that instance's
    watch).
    """

    def __init__(self, inner: Inner) -> None:
        super().__init__(daemon=True)
        self.inner = inner
        self.generation = inner.generation
        self.stop = threading.Event()

    def run(self) -> None:
        """Watch until :attr:`stop` is set, the instance is replaced, or a restart fails."""
        spec = self.inner.run.config["health"]
        misses = 0
        while not self.stop.wait(float(spec["interval_s"])):
            if self.inner.generation != self.generation:
                return
            if engine_alive(self.inner):
                misses = 0
                continue
            if self.inner.generation != self.generation:
                return
            misses += 1
            if misses >= int(spec["failures"]):
                self.inner.engine_ready.clear()
                restart_engine(self.inner, f"{misses} failed health checks", self.generation)
                return


class SubmissionWriter:
    """Keeps ``submission.json`` valid at every moment and fills it as tasks finish.

    Why: Kaggle scores whatever the file holds when the session ends. It starts as every
    attempt ``[[0]]`` (or the valid parts of an earlier file); a finished task replaces its
    placeholders; a test input the model answered is never overwritten by a later attempt with
    nothing, so an answered task never reverts.
    """

    def __init__(self, path: Path, challenges: Mapping[str, Any]) -> None:
        from arc26.submit import repair

        self.path = path
        self.challenges = challenges
        old = None
        if path.is_file():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = None
        self.data = repair(old, challenges)
        self.answered: set[tuple[str, int]] = set()
        self.lock = threading.Lock()
        self.write()

    def write(self) -> None:
        """Write the file atomically and check it."""
        from arc26.submit import validate, write_json_atomic

        problems = validate(self.data, self.challenges)
        if problems:
            raise RuntimeError(f"submission would be invalid: {problems[:3]}")
        write_json_atomic(self.path, self.data)

    def update(self, task_id: str, picks: Sequence[Sequence[Any]]) -> int:
        """Put a task's answers in place; return how many test inputs gained an answer.

        Args:
            task_id: A task of the challenges file.
            picks: Per test input, up to two grids, best first.

        Returns:
            Test inputs filled now.
        """
        from arc26.submit import fallback_grid, grid_problem

        if task_id not in self.data:
            return 0
        filled = 0
        with self.lock:
            for index, pick in enumerate(picks[: len(self.data[task_id])]):
                grids = [g for g in pick if grid_problem(g) is None]
                if not grids or (task_id, index) in self.answered:
                    continue
                self.data[task_id][index] = {
                    "attempt_1": [list(r) for r in grids[0]],
                    "attempt_2": [list(r) for r in grids[1]] if len(grids) > 1 else fallback_grid(),
                }
                self.answered.add((task_id, index))
                filled += 1
            if filled:
                self.write()
        return filled


def task_messages(task: Any, system: str) -> list[dict[str, str]]:
    """Return the system and user messages of a direct-answer attempt (P6.9-01's prompt)."""
    from arc26.agent.prompts import task_text

    return [{"role": "system", "content": system}, {"role": "user", "content": task_text(task)}]


def challenges_path(inner: Inner) -> Path:
    """The challenges file of the mode: the hidden test tasks, or the evaluation tasks."""
    spec = inner.run.config["solve"]["challenges"]
    return Path(inner.run.state["paths"]["competition"]) / spec[inner.mode]


def load_solve_tasks(inner: Inner) -> None:
    """Load the mode's tasks and count their openings; competition order: smallest first."""
    from arc26.agent.shot import SYSTEM_PROMPTS
    from arc26.agent.tasks import load_tasks

    config = inner.run.config
    path = challenges_path(inner)
    if "solution" in path.name.lower():
        raise PermissionError(f"refusing {path.name}: the solutions are never opened")
    inner.challenges = json.loads(path.read_text(encoding="utf-8"))
    ids = None if inner.mode == COMPETITION else config["solve"]["task_ids"]
    loaded = {t.task_id: t for t in load_tasks(path, ids)}
    system = SYSTEM_PROMPTS[config["solve"]["system"]]
    kwargs = config["template"]["kwargs"]
    for task_id, task in loaded.items():
        text = inner.tok.render(task_messages(task, system), **kwargs)
        inner.openings[task_id] = len(inner.tok.encode(text))
    if inner.mode == COMPETITION:
        order = sorted(loaded, key=lambda t: (inner.openings[t], t))
        inner.plan_openings = [inner.openings[t] for t in order]
    else:
        order = list(ids)
        # The plan of a development run is the scored rerun's: the visible test file's tasks.
        test_path = Path(inner.run.state["paths"]["competition"]) / config["solve"][
            "challenges"][COMPETITION]
        inner.plan_openings = [
            len(inner.tok.encode(inner.tok.render(task_messages(t, system), **kwargs)))
            for t in load_tasks(test_path)]
    inner.tasks = tuple(loaded[t] for t in order)


def stage_tasks(inner: Inner) -> dict[str, Any]:
    """Load the tasks and their opening token counts (:func:`load_solve_tasks`)."""
    load_solve_tasks(inner)
    values = sorted(inner.openings.values())
    return {"mode": inner.mode, "tasks": len(inner.tasks),
            "test_inputs": sum(len(t.test_inputs) for t in inner.tasks),
            "opening_min": values[0], "opening_max": values[-1],
            "opening_median": values[len(values) // 2],
            "plan_tasks": len(inner.plan_openings),
            "plan_opening_max": max(inner.plan_openings) if inner.plan_openings else None}


def stage_template_check(inner: Inner) -> dict[str, Any]:
    """Check that our rendering equals the server's: prompt token counts of the first task."""
    from arc26.agent.shot import SYSTEM_PROMPTS

    config = inner.run.config
    task = inner.tasks[0]
    messages = task_messages(task, SYSTEM_PROMPTS[config["solve"]["system"]])
    kwargs = config["template"]["kwargs"]
    ids = inner.tok.encode(inner.tok.render(messages, **kwargs))
    body = {"model": config["engine"]["served_model_name"], "messages": messages,
            "max_tokens": 1, **kwargs}
    try:
        status, reply = http_json(inner.base, "POST", "/v1/chat/completions", body,
                                  timeout=float(config["engine"]["first_generation_timeout_s"]))
    except OSError as err:
        status, reply = None, {"error": _describe(err)}
    chat_tokens = find_key(reply, "prompt_tokens") if status == 200 else None
    gen = Generation(inner.base, {"input_ids": ids, "sampling_params": sampling(config, 1)},
                     float(config["engine"]["first_generation_timeout_s"]),
                     config["engine"]["incremental_stream"])
    result = gen.run()
    out = {"task_id": task.task_id, "ours": len(ids), "chat_endpoint": chat_tokens,
           "chat_status": status, "generate_prompt_tokens": result.prompt_tokens,
           "equal": chat_tokens == len(ids), "generate": result.row(),
           "rendered_tail": inner.tok.render(messages, **kwargs)[-120:]}
    write_json(inner.run.out / "template_check.json", out)
    return out


def most_streams(inner: Inner, tokens_per_stream: int, cap: int) -> int:
    r"""Return the most streams the pool and the request limit allow.

    .. math:: n = \min\left(L, \left\lfloor \frac{P}{t} \right\rfloor, c\right)

    with :math:`L` the running-request limit, :math:`P` the pool, :math:`t` the tokens one
    stream holds, and :math:`c` the config's cap.

    Args:
        inner: The inner state (pool and limit from the boot).
        tokens_per_stream: :math:`t`.
        cap: :math:`c`.

    Returns:
        At least 1.
    """
    limit = inner.max_running or cap
    pool = inner.pool or tokens_per_stream
    return max(1, min(limit, pool // max(tokens_per_stream, 1), cap))


def task_cap(context: int, opening: int, spec: Mapping[str, Any]) -> int:
    r"""Return a task's total generated-token cap at the per-stream context :math:`c`.

    .. math:: g_t = \min\left(C_{\max},\ c - o_t - m\right)

    with :math:`o_t` the task's opening, :math:`m` the context margin, and :math:`C_{\max}` the
    config's ``solve.cap``. Both arms generate at most :math:`g_t`; thinking stops
    ``forcing_tokens`` short of it, and the forced reply takes the rest.

    Args:
        context: :math:`c`.
        opening: :math:`o_t`.
        spec: ``solve`` of the config.

    Returns:
        :math:`g_t`, at least ``forcing_tokens`` more than nothing (1).
    """
    room = context - opening - int(spec["context_margin_tokens"])
    return max(1, min(int(spec["cap"]), room))


def plan_budget(pool: int | None, limit: int | None, rate: float, time_s: float, n_tasks: int,
                openings: Sequence[int], arms: int, spec: Mapping[str, Any],
                planner: Mapping[str, Any]) -> dict[str, Any]:
    r"""Choose the streams and the context per stream that keep a long thinking budget.

    Why: the 27B's solves need long thinking (``docs/research/budget_vs_solves.md`` on
    task/ideas: :math:`S(b)` 1.5% at 16k, 3.0% at 24k and 32k, 5.2% at 40k, 8.2% at 48k, 9.7% at
    56k and 65.5k, 17.9% at 94.2k; forced answers below the natural end scored 0 of 56), so the
    expected solves :math:`N(c)\,S(b(c))` peak at long budgets, not at full coverage. With
    :math:`s` streams sharing the pool :math:`P`,

    .. math:: c(s) = \min\left(X,\ \left\lfloor \frac{P}{s} \right\rfloor - S\right), \qquad
       b(s) = c(s) - o_{\max} - F - m,

    and the plan takes the most streams :math:`k \le s \le \min(L, 8)` (:math:`k` the arms) with
    :math:`b(s) \ge` ``planner.min_thinking_tokens``; if even :math:`k` streams miss it, :math:`k`
    streams at :math:`c(k)`, recorded as not met. The
    coverage :math:`N_{\text{cov}} = s\,r\,T / \bar g` (every task to its cap
    :math:`g_t = \min(C_{\max}, c - o_t - m)`) is reported, not targeted.

    Args:
        pool: :math:`P`, the server's KV pool in tokens (None: one stream).
        limit: :math:`L`, the server's running-request limit (None: ``solve.max_streams``).
        rate: :math:`r`, decode tokens per second per stream (for the coverage only).
        time_s: :math:`T`, the generation window in seconds (for the coverage only).
        n_tasks: :math:`N`, the tasks to cover (reported against the coverage).
        openings: The openings of the tasks planned for (:math:`o_{\max}`, :math:`\bar g`).
        arms: Arms sharing the streams (split evenly).
        spec: ``solve`` of the config (``context_tokens`` :math:`X`, ``state_reserve_tokens``
            :math:`S`, ``forcing_tokens`` :math:`F`, ``context_margin_tokens`` :math:`m`).
        planner: ``planner`` of the config (``min_thinking_tokens``).

    Returns:
        ``context``, ``streams``, ``per_arm``, ``thinking``, ``mean_cap``, ``coverage``,
        ``met`` (the floor reached), and the inputs.
    """
    top = int(spec["context_tokens"])
    state = int(spec["state_reserve_tokens"])
    reserve = int(spec["forcing_tokens"]) + int(spec["context_margin_tokens"])
    floor = int(planner["min_thinking_tokens"])
    values = list(openings) or [0]
    o_max = max(values)
    cap_streams = max(arms, min(int(limit or spec["max_streams"]), int(spec["max_streams"])))
    available = pool if pool is not None else (top + state) * arms

    def context_for(streams: int) -> int:
        return min(top, available // streams - state)

    chosen = None
    # At least one stream per arm (development runs the arms side by side), each stream at the
    # context the pool allows, so no plan holds more than the pool.
    for streams in range(cap_streams, arms - 1, -1):
        if context_for(streams) - o_max - reserve >= floor:
            chosen = (streams, context_for(streams), True)
            break
    if chosen is None:
        # Below the floor even at one stream per arm: one stream per arm at the largest context
        # the pool allows, recorded as not met (the run goes on with the shorter thinking).
        chosen = (arms, context_for(arms), False)
    streams, context, met = chosen
    mean_cap = sum(task_cap(context, o, spec) for o in values) / len(values)
    coverage = streams * rate * time_s / mean_cap
    return {"context": context, "streams": streams, "per_arm": max(1, streams // arms),
            "thinking": context - o_max - reserve, "mean_cap": round(mean_cap, 1),
            "coverage": round(coverage, 1), "covered": round(min(coverage, n_tasks), 1),
            "met": met, "floor": floor, "pool": pool, "limit": limit, "rate": rate,
            "time_s": round(time_s, 1), "n_tasks": n_tasks, "openings": len(values),
            "opening_max": o_max, "opening_mean": round(sum(values) / len(values), 1)}


class Gate:
    """A counting gate whose limit can change: each arm's streams, re-planned once."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.cond = threading.Condition()

    def set_limit(self, limit: int) -> None:
        """Change the limit and wake the waiters."""
        with self.cond:
            self.limit = limit
            self.cond.notify_all()

    def acquire(self, deadline: float) -> bool:
        """Wait for a free stream until ``deadline``; return whether one was taken."""
        with self.cond:
            while self.active >= self.limit:
                left = deadline - _now()
                if left <= 0:
                    return False
                self.cond.wait(timeout=min(left, 5))
            self.active += 1
            return True

    def release(self) -> None:
        """Give a stream back."""
        with self.cond:
            self.active -= 1
            self.cond.notify_all()


def tool_module(config: Mapping[str, Any]) -> Any:
    """Import the tool loop named by ``tool.module`` (``arc26.agent.toolshot``).

    ``tool.extra_path``, relative to this script's folder, is put first on ``sys.path`` when set
    (the tests' stand-in before ``task/P5.3`` lands).
    """
    import importlib

    spec = config["tool"]
    if spec.get("extra_path"):
        path = str((HERE / spec["extra_path"]).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module(spec["module"])


SELF_TEST_CALL = ("<tool_call>\n<function=python>\n<parameter=code>\n"
                  "def transform(grid):\n    return grid\nprint(check(transform))\n"
                  "</parameter>\n</function>\n</tool_call>")


def tool_self_test(inner: Inner) -> dict[str, Any]:
    """Run the tool loop once on a tiny task with a scripted model: does the sandbox work?

    Why: the tool arm needs the REPL subprocess, the verifier, and the call parser on this
    machine; a broken sandbox must cost the arm, not the run (version 11 then runs plain).

    Args:
        inner: The inner state (its tokenizer).

    Returns:
        ``ok`` and what the loop recorded.
    """
    from arc26.agent.types import AgentTask, Pair

    module = tool_module(inner.run.config)
    cfg = module.tool_config(inner.run.config["tool"].get("loop", {}))
    grid = [[1, 2], [3, 4]]
    task = AgentTask("selftest", (Pair(grid, grid), Pair([[5]], [[5]])), ([[7, 8]],))
    replies = iter([SELF_TEST_CALL, "Checked.\n</think>\n\n```answer\n"
                    '{"test_index": 0, "grid": [[7, 8]]}\n```<|im_end|>'])

    def generate(ids: list[int], max_new: int, cut_at: float, kind: str,
                 extra: Mapping[str, Any]) -> GenResult:
        text = next(replies, "</think>\n\n<|im_end|>")
        out = inner.tok.encode(text)
        now = _now()
        return GenResult(text=text, ids=out, prompt_tokens=len(ids), completion_tokens=len(out),
                         cached_tokens=0, finish="stop", start=now, first=now, end=now)

    start = _now()
    outcome = module.solve_with_tools(task, inner.tok, generate, cfg, force_at=_now() + 120,
                                      end=_now() + 180)
    calls = list(outcome.calls)
    ok = bool(calls) and calls[0].get("status") == "ok" and bool(outcome.attempts) and bool(
        outcome.attempts[0])
    inner.tool_ok = ok
    return {"status": "ok" if ok else "failed", "calls": calls,
            "program": outcome.program is not None, "seconds": round(_now() - start, 2)}


def make_generate(inner: Inner, task_id: str, turns: list[dict[str, Any]]) -> Callable[..., Any]:
    """Return ``generate(ids, max_new, cut_at, kind, extra)`` over streaming ``/generate``.

    The interface of ``arc26.agent.toolshot.GenerateFn``; the plain arm uses it too. A request
    still running at ``cut_at`` is cut and returns what it generated (finish ``cancelled``).
    Each call appends its row to ``turns`` and a timing record.
    """
    from arc26 import timing

    config = inner.run.config
    spec = config["solve"]
    logger = timing.get_logger()

    room_total = int(spec["context_tokens"]) - int(spec["context_margin_tokens"])

    def generate(ids: list[int], max_new: int, cut_at: float, kind: str,
                 extra: Mapping[str, Any] | None = None) -> GenResult:
        # Input plus max_new stays a margin under the server's context: SGLang refuses a
        # request at the context length (without --allow-auto-truncate).
        max_new = min(int(max_new), room_total - len(ids))
        if max_new < 1:
            now = _now()
            result = GenResult(finish="error", error="no room left in the context",
                               start=now, end=now)
            turns.append({"kind": kind, **result.row()})
            return result
        params = sampling(config, max_new)
        params.update(extra or {})
        body = {"input_ids": list(ids), "sampling_params": params}
        if "json_schema" in params:
            body["require_reasoning"] = False
        gen = Generation(inner.base, body,
                         float(spec["request_timeout_s"]), config["engine"]["incremental_stream"])
        timer = threading.Timer(max(cut_at - _now(), 0), gen.cancel)
        timer.daemon = True
        timer.start()
        try:
            result = gen.run()
        finally:
            timer.cancel()
        turns.append({"kind": kind, **result.row()})
        if logger is not None:
            logger.record(task_id, f"fn_{kind}", result.end - result.start,
                          result.completion_tokens)
        rate = result.decode_rate()
        if kind in ("think", "tool") and rate and result.completion_tokens >= int(
                config["planner"]["replan_min_tokens"]):
            observe_rate(inner, rate)
        return result

    return generate


def stage_schema_probe(inner: Inner) -> dict[str, Any]:
    """Exercise the native constrained endpoint on one and two tiny output indices."""
    module = tool_module(inner.run.config)
    turns: list[dict[str, Any]] = []
    generate = make_generate(inner, "schema-probe", turns)
    cases = []
    for count in (1, 2):
        expected = {"answers": {str(i): [["12"]] for i in range(count)}}
        prompt = inner.tok.render([
            {"role": "system", "content": "Return exactly the requested JSON object."},
            {"role": "user", "content": "Copy this object exactly, without commentary: "
             + json.dumps(expected)},
        ], reasoning_effort="xhigh") + "\n</think>\n\n"
        extra = {"json_schema": json.dumps(module._answer_schema(count)),
                 "ignore_eos": False, "stop": [], "stop_token_ids": [], "no_stop_trim": False}
        result = generate(inner.tok.encode(prompt), 512, min(_now() + 60,
                          _now() + inner.run.left()), "schema_probe", extra)
        module._schema_answers(result.text, count, result.finish,
                               inner.run.config["tool"].get("loop", {}).get(
                                   "strip_tokens", ["<|im_end|>", "<|endoftext|>"]))
        clean = result.text.strip()
        for token in ("<|im_end|>", "<|endoftext|>"):
            clean = clean.removesuffix(token).rstrip()
        if result.finish in ("error", "abort") or json.loads(clean) != expected:
            raise RuntimeError(f"native schema probe failed for {count} test indices")
        cases.append({"test_inputs": count, "finish": result.finish,
                      "completion_tokens": result.completion_tokens, "ok": True})
    record = {"ok": True, "cases": cases, "turns": turns,
              "fake_server": inner.run.config["engine"].get("script") == "fn_fake_sglang.py"}
    write_json(inner.run.out / "schema_probe.json", record)
    return record


@dataclass
class Clock:
    """The solve phase's two deadlines (wall time).

    Attributes:
        force_at: Thinking requests still running are cut and forced from here.
        end: Every request is cut here.
    """

    force_at: float
    end: float


def max_repeat_run(ids: Sequence[int], special_min_id: int) -> int:
    """Return the longest run of one repeated token, special tokens left out.

    Why: a run of 64 or more identical tokens is the symptom of NaN logits or a broken kernel
    (P6.3-02's degenerate rule), the silent corruption the recipe warns of (#37111).

    Args:
        ids: Generated token ids.
        special_min_id: Ids at or above this are special and never counted.

    Returns:
        The run length (0 for no ids).
    """
    best = run = 0
    previous = None
    for token in ids:
        if token >= special_min_id:
            previous, run = None, 0
            continue
        run = run + 1 if token == previous else 1
        previous = token
        best = max(best, run)
    return best


def answers_from(text: str, task: Any, lenient: bool,
                 strip: Sequence[str] = ()) -> tuple[list[tuple[int, Any]], list[str]]:
    """Parse the answer blocks after the last ``</think>``.

    Args:
        text: The attempt's whole generated text (special tokens kept).
        task: The task (its test count).
        lenient: P6.9-01's lenient parser.
        strip: End-of-turn tokens removed first, so a closing fence followed by
            ``<|im_end|>`` still ends its line.

    Returns:
        ``[(test_index, grid)]`` in reply order, and the parse errors.
    """
    from arc26.agent.loop import fenced_blocks, parse_answer

    if THINK_CLOSE not in text:
        return [], ["no </think> in the reply"]
    visible = text.rsplit(THINK_CLOSE, 1)[1]
    for token in strip:
        visible = visible.replace(token, "\n")
    answers, errors = [], []
    for tag, body in fenced_blocks(visible):
        if tag != "answer":
            continue
        try:
            answers.append(parse_answer(body, len(task.test_inputs), lenient=lenient))
        except ValueError as err:
            errors.append(str(err)[:200])
    return answers, errors


def _plain(value: Any) -> Any:
    # Grids as plain lists of ints (a program's outputs may be numpy arrays).
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return int(value) if hasattr(value, "__index__") and not isinstance(value, bool) else value


def tool_budget(inner: Inner, opening: int) -> dict[str, Any]:
    """Return the tool loop's ``tool_config`` mapping at the planned context.

    The cap is :func:`task_cap`'s (the plain arm's rule), the context bound the planned context
    less the margin, the tool replies at most a fifth of the context, and the forcing turn and
    note the plain arm's.

    Args:
        inner: The inner state (its planned context).
        opening: The task's opening.

    Returns:
        ``tool.loop`` updated with those values.
    """
    config = inner.run.config
    spec = config["solve"]
    loop = dict(config["tool"].get("loop", {}))
    replies = int(loop.get("reply_tokens", spec["tool_reply_reserve_tokens"]))
    loop.update(cap=task_cap(inner.context, opening, spec),
                context_tokens=inner.context - int(spec["context_margin_tokens"]),
                reply_tokens=max(1, min(replies, inner.context // 5)),
                forcing_tokens=int(spec["forcing_tokens"]),
                thinking_end_note=spec["thinking_end_note"])
    return loop


def observe_rate(inner: Inner, rate: float) -> None:
    """Keep a measured decode rate; after ``planner.replan_after_requests``, re-plan once.

    The re-plan records the coverage at the measured rate (``replan.json``).
    """
    planner = inner.run.config["planner"]
    # Done before any lock: a restart holds engine_lock through a whole boot, and a request it
    # cut must reach its requeue at once (fix round D).
    if inner.plan is None or inner.plan.get("replanned"):
        return
    with inner.plan_lock:
        if inner.plan is None or inner.plan.get("replanned"):
            return
        inner.rates.append(rate)
        if len(inner.rates) < int(planner["replan_after_requests"]):
            return
        rates = sorted(inner.rates)
        measured = rates[len(rates) // 2]
        first = inner.plan
        plan = plan_budget(first["pool"], first["limit"], measured, first["time_s"],
                           first["n_tasks"], inner.plan_openings, first["arms"],
                           inner.run.config["solve"], planner)
        plan.update(arms=first["arms"], replanned=True, rates=[round(r, 2) for r in rates],
                    at_s=round(_now() - inner.run.start, 1))
        inner.plan = plan
        # The streams and context follow the pool and the openings only; the measured rate
        # changes the reported coverage. Applied anyway, so a rule that reads the rate holds.
        inner.context = plan["context"]
        for gate in inner.gates.values():
            gate.set_limit(plan["per_arm"])
    write_json(inner.run.out / "replan.json", plan)


def attempt_plain(inner: Inner, task: Any, clock: Clock, turns: list[dict[str, Any]]
                  ) -> dict[str, Any]:
    """Arm A: the thinking turn to the cap, then the forcing turn if it was cut.

    The thinking turn sends the rendered prompt's ids with ``max_new_tokens`` at the cap. A
    reply that ends by itself is final. A reply cut at the cap or at ``clock.force_at`` is
    continued from its own ids: inside the thinking block with the note and ``</think>``
    appended, else (cut inside the answer) as is, with ``forcing_tokens`` more.

    Args:
        inner: The inner state.
        task: The task.
        clock: The deadlines.
        turns: Receives one row per request.

    Returns:
        The arm's part of the record (answers as grids, never scored here) and the text.
    """
    from arc26.agent.shot import SYSTEM_PROMPTS, implied_shapes, shot_answers

    config = inner.run.config
    spec = config["solve"]
    system = SYSTEM_PROMPTS[spec["system"]]
    prompt_text = inner.tok.render(task_messages(task, system), **config["template"]["kwargs"])
    prompt_ids = inner.tok.encode(prompt_text)
    call = make_generate(inner, task.task_id, turns)
    second: GenResult | None = None
    # One budget rule for both arms: at most the task's cap in all, thinking stopping
    # forcing_tokens short of it, and the forced reply taking the rest.
    cap = task_cap(inner.context, len(prompt_ids), spec)
    forcing = int(spec["forcing_tokens"])
    first = call(prompt_ids, max(1, cap - forcing), clock.force_at, "think")
    text = first.text
    out_ids = list(first.ids) or inner.tok.encode(first.text)
    forced = False
    # Forced only after a cut (the cap or the force time): after an abort or an error the
    # server may be gone, and the task is queued again instead (engine_failed).
    if first.finish in ("length", "cancelled") and _now() < clock.end and (
            first.completion_tokens > 0):
        closed = THINK_CLOSE in first.text
        suffix = "" if closed else spec["thinking_end_note"] + spec["forcing_suffix"]
        ids = prompt_ids + out_ids + (inner.tok.encode(suffix) if suffix else [])
        second = call(ids, max(1, min(forcing, cap - first.completion_tokens)), clock.end,
                      "force")
        text = first.text + suffix + second.text
        forced = True
    answers, errors = answers_from(text, task, bool(spec["lenient_answers"]),
                                   spec["strip_tokens"])
    implied = implied_shapes(task)
    holder = type("Record", (), {"direct_answers": answers})()
    picks = [shot_answers([holder], j, implied[j] if spec["lenient_answers"] else None)
             for j in range(len(task.test_inputs))]
    all_ids = out_ids + (list(second.ids) if second is not None else [])
    close_id = inner.tok.encode(THINK_CLOSE)
    thinking = None
    if len(close_id) == 1 and close_id[0] in out_ids:
        thinking = out_ids.index(close_id[0])
    return {
        "opening_tokens": len(prompt_ids), "cap": cap, "context": inner.context,
        "forced": forced,
        "continuation": "ids" if first.ids else "text",
        "ended_by": first.finish if not forced else f"{first.finish}+force",
        "thinking_tokens": thinking,
        "max_repeat_run": max_repeat_run(all_ids, config["kill"]["special_min_id"]),
        "blocks": len(answers), "parse_errors": errors,
        "attempts": [[g for g in pick] for pick in picks], "text": text,
    }


def attempt_tool(inner: Inner, task: Any, clock: Clock, turns: list[dict[str, Any]]
                 ) -> dict[str, Any]:
    """Arm B: "think with Python" (``arc26.agent.toolshot.solve_with_tools``) on this server.

    Args:
        inner: The inner state.
        task: The task.
        clock: The deadlines (``force_at``, ``end``).
        turns: Receives one row per request.

    Returns:
        The arm's part of the record and the text.
    """
    config = inner.run.config
    module = tool_module(config)
    cfg = module.tool_config(tool_budget(inner, inner.openings.get(task.task_id, 0)))
    generate = make_generate(inner, task.task_id, turns)
    extra = {"verifier": inner.verifier} if inner.verifier is not None else {}
    outcome = module.solve_with_tools(task, inner.tok, generate, cfg, force_at=clock.force_at,
                                      end=clock.end, **extra)
    ids = inner.tok.encode(outcome.text)
    program = outcome.program
    if program is not None and not isinstance(program, Mapping):
        program = getattr(program, "__dict__", {"repr": repr(program)})
    return {
        "opening_tokens": inner.openings.get(task.task_id), "cap": cfg.cap,
        "context": inner.context, "reply_budget": cfg.reply_tokens,
        "forced": bool(outcome.forced),
        "continuation": "ids", "ended_by": outcome.ended_by,
        "tool_turns": list(outcome.turns), "calls": list(outcome.calls),
        "tool_reply_tokens": outcome.tool_reply_tokens,
        "program": _plain(dict(program)) if program else None,
        "direct": _plain(outcome.direct), "tool_errors": list(outcome.errors),
        "blocks": sum(len(d) for d in outcome.direct),
        "max_repeat_run": max_repeat_run(ids, config["kill"]["special_min_id"]),
        "attempts": _plain(outcome.attempts), "text": outcome.text,
    }


def engine_failed(record: Mapping[str, Any], restarted: bool = False) -> bool:
    """Whether an attempt's requests failed in a way that points at the engine.

    Args:
        record: The attempt's record.
        restarted: The server restarted (or died) while the attempt ran: then a request that
            ended in ``abort`` (the stream cut without a reason) counts too.

    Returns:
        True for an engine error in any request, or an aborted request across a restart.
    """
    texts = [str(t.get("error") or "") for t in record.get("turns", [])]
    texts.append(str(record.get("error") or ""))
    if any(word in text for text in texts for word in ENGINE_ERRORS):
        return True
    return restarted and any(t.get("finish") in ("abort", "error")
                             for t in record.get("turns", []))


def attempt_task(inner: Inner, task: Any, clock: Clock, stream: int, arm: str
                 ) -> dict[str, Any]:
    """Run one task on one arm and return its record (the text goes to ``transcripts.jsonl``).

    Args:
        inner: The inner state.
        task: The task.
        clock: The deadlines.
        stream: The worker's index.
        arm: ``plain`` or ``tool``.

    Returns:
        The record: arm, stream, times, every request's row, tokens, the forcing, the answers.
    """
    started = _now()
    turns: list[dict[str, Any]] = []
    record: dict[str, Any] = {"task_id": task.task_id, "arm": arm, "stream": stream,
                              "started_s": round(started - inner.run.start, 1),
                              "test_outputs": len(task.test_inputs)}
    try:
        part = (attempt_tool if arm == "tool" else attempt_plain)(inner, task, clock, turns)
        text = part.pop("text")
        record.update(part)
        append_jsonl(inner.run.out / "transcripts.jsonl",
                     {"task_id": task.task_id, "arm": arm, "text": text}, inner.run.lock)
    except Exception as err:  # one task's failure is recorded; the stream goes on
        record.update(error=_describe(err), traceback=_tail(traceback.format_exc(), 2000))
    record["turns"] = turns
    record["completion_tokens"] = sum(t.get("completion_tokens") or 0 for t in turns)
    record["ended_s"] = round(_now() - inner.run.start, 1)
    record["wall_s"] = round(_now() - started, 1)
    return record


def solve_arms(inner: Inner) -> list[str]:
    """Return the arms this run uses: development's pair, or competition's one arm.

    A tool arm whose self-test failed is dropped (competition falls back to plain).
    """
    spec = inner.run.config["solve"]
    arms = list(spec["dev_arms"]) if inner.mode == DEVELOPMENT else [spec["arm"]]
    if "tool" in arms and inner.tool_ok is False:
        arms = [a for a in arms if a != "tool"] or ["plain"]
    return arms


def stage_sweep(inner: Inner) -> dict[str, Any]:
    r"""Measure decode at each batch size before the solve; write ``sweep.json``.

    Why: P5.6-01's question is the per-stream step cost at 8 streams (``fn_many_streams.md``
    section 1, ``fn_900_toks.md``): version 14 measured only batch 1 to 3. At each size $n$, $n$
    requests with the same short prompt run at once, each to ``max_new`` tokens with
    ``ignore_eos``; the radix cache serves the prompt after the first, so the window is decode.
    Per size: each request's rate $(k - 1)/(e - f)$ (``GenResult.decode_rate``), their median,
    the aggregate $\sum_i (k_i - 1) / (\max_i e_i - \min_i f_i)$, the step time
    $1000\,n / \text{aggregate}$ ms, and the server's own ``gen throughput`` lines in the window.

    Returns:
        Per size the readings; sizes above the server's running limit, or past the time left
        before ``sweep.stop_before_s`` of the generation stop, are skipped and named.

    Raises:
        StageSkipped: If ``sweep.enabled`` is off.
    """
    run = inner.run
    spec = run.config.get("sweep") or {}
    if not spec.get("enabled"):
        raise StageSkipped("sweep.enabled is off")
    config = run.config
    prompt = inner.tok.encode(inner.tok.render([{"role": "user", "content": spec["prompt"]}],
                                               **config["template"]["kwargs"]))
    stop = run.start + time_plan(config, inner.mode)["generation_stop_s"]
    log_path = run.path(config["paths"]["engine_log"])
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for n in [int(b) for b in spec["batches"]]:
        if inner.max_running is not None and n > inner.max_running:
            skipped.append(f"batch {n}: above the server's limit {inner.max_running}")
            continue
        if stop - _now() < float(spec["stop_before_s"]):
            skipped.append(f"batch {n}: {round(stop - _now())} s before the generation stop")
            continue
        offset = log_path.stat().st_size if log_path.is_file() else 0
        body = {"input_ids": prompt,
                "sampling_params": sampling(config, int(spec["max_new"]), ignore_eos=True)}
        gens = [Generation(inner.base, body, float(spec["timeout_s"]),
                           config["engine"]["incremental_stream"]) for _ in range(n)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(lambda g: g.run(), gens))
        firsts = [r.first for r in results if r.first is not None]
        made = sum(max(r.completion_tokens - 1, 0) for r in results)
        window = (max(r.end for r in results) - min(firsts)) if firsts else 0.0
        aggregate = made / window if window > 0 else None
        rates = sorted(r.decode_rate() for r in results if r.decode_rate() is not None)
        text = ""
        if log_path.is_file():
            with open(log_path, "rb") as handle:
                handle.seek(offset)
                text = handle.read().decode("utf-8", errors="replace")
        rows.append({
            "batch": n, "requests": [r.row() for r in results],
            "median_tok_s": rates[len(rates) // 2] if rates else None,
            "aggregate_tok_s": None if aggregate is None else round(aggregate, 2),
            "step_ms": None if not aggregate else round(1000.0 * n / aggregate, 2),
            "server_lines": decode_lines(text),
            "errors": [r.error for r in results if r.error],
        })
        write_json(run.out / "sweep.json", {"rows": rows, "skipped": skipped,
                                            "prompt_tokens": len(prompt)})
    write_json(run.out / "sweep.json", {"rows": rows, "skipped": skipped,
                                        "prompt_tokens": len(prompt)})
    return {"batches": [r["batch"] for r in rows],
            "median_tok_s": {str(r["batch"]): r["median_tok_s"] for r in rows},
            "step_ms": {str(r["batch"]): r["step_ms"] for r in rows}, "skipped": skipped}


def free_device_min(text: str, pattern: str) -> dict[str, Any]:
    """Return the smallest free device memory the engine logged, in GiB, and the line count.

    Why: the fork logs ``free device mem: X GiB`` whenever a kernel loads after serving started;
    P5.6-01's minimum was 0.02 GiB before its out-of-memory fault, version 17's 0.22
    (``fn_p56_suspects.md`` section 3), a reading ``memory.jsonl``'s ``nvidia-smi`` use cannot
    give.

    Args:
        text: The server log.
        pattern: A regular expression whose first group is the GiB value.

    Returns:
        ``min_gib`` (None without a match) and ``lines``.
    """
    values = [float(m.group(1)) for m in re.finditer(pattern, text)]
    return {"min_gib": min(values) if values else None, "lines": len(values)}


def stage_stress(inner: Inner) -> dict[str, Any]:
    """Run the solve's largest prefill and its full decode batch once before the solve.

    Why: P5.6-01's only out-of-memory fault came at the solve's first prefill (8 openings queued,
    0 running), after the sweep's short prompts had left the long-prefill kernels unloaded
    (``docs/experiments/P5.6-01-forensics.md``). Here the $n$ largest openings of the plan's tasks
    ($n$ the streams the server and ``solve.max_streams`` allow) are sent at once, each a distinct
    prompt of that length (so the radix cache shares nothing), then every request decodes
    ``stress.max_new`` tokens with ``ignore_eos`` (about 30 s). The engine chunks the prefill at
    its ``--chunked-prefill-size`` and ``--prefill-max-requests``, as in the solve. A failure here
    moves the boot one rung down the ladder (``inner_main``) instead of failing mid-solve.

    Returns:
        The streams, the prompt lengths, each request's row, the seconds, the free device memory
        the engine logged in the window, and ``memory.jsonl``'s peak; also ``stress.json``.

    Raises:
        StageSkipped: If ``stress.enabled`` is off.
        RuntimeError: If a request errs or ends short, or the engine stops answering.
    """
    run = inner.run
    config = run.config
    spec = config.get("stress") or {}
    if not spec.get("enabled"):
        raise StageSkipped("stress.enabled is off")
    limit = int(config["solve"]["max_streams"])
    n = min(limit, inner.max_running or limit)
    max_new = int(spec["max_new"])
    room = int(config["solve"]["context_tokens"]) - max_new - int(
        config["solve"]["context_margin_tokens"])
    openings = sorted(inner.plan_openings or list(inner.openings.values()), reverse=True)[:n]
    lengths = [max(16, min(int(o), room)) for o in openings] or [16] * n
    filler = inner.tok.encode(str(spec.get("filler", "0 1 2 3 4 5 6 7 8 9\n")) * (
        max(lengths) // 4 + 64))
    prompts = []
    for i, length in enumerate(lengths):
        head = inner.tok.encode(f"Stress request {i}:\n")
        prompts.append((head + filler)[:length])
    log_path = run.path(config["paths"]["engine_log"])
    offset = log_path.stat().st_size if log_path.is_file() else 0
    start = _now()
    gens = [Generation(inner.base, {"input_ids": ids, "sampling_params": sampling(
        config, max_new, ignore_eos=True)}, float(spec["timeout_s"]),
        config["engine"]["incremental_stream"]) for ids in prompts]
    with ThreadPoolExecutor(max_workers=max(len(gens), 1)) as pool:
        results = list(pool.map(lambda g: g.run(), gens))
    seconds = round(_now() - start, 1)
    text = ""
    if log_path.is_file():
        with open(log_path, "rb") as handle:
            handle.seek(offset)
            text = handle.read().decode("utf-8", errors="replace")
    alive = engine_responsive(inner)
    if inner.monitor is not None:
        inner.monitor.sample()
    errors = [r.error for r in results if r.error]
    short = [i for i, r in enumerate(results) if r.completion_tokens < max_new]
    record = {
        "streams": len(prompts), "prompt_tokens": lengths, "max_new": max_new,
        "seconds": seconds, "requests": [r.row() for r in results], "errors": errors,
        "short": short, "engine_alive": alive,
        "free_device": free_device_min(text, str(config["engine"].get(
            "free_pattern", r"free device mem: ([\d.]+) GiB"))),
        "gpu_peak_used_mib": None if inner.monitor is None else inner.monitor.peak_used_mib,
        "out_of_memory": "out of memory" in text.lower(),
    }
    write_json(run.out / "stress.json", record)
    if errors or short or not alive or record["out_of_memory"]:
        raise RuntimeError(f"stress check failed: {len(errors)} errors, short {short}, "
                           f"alive {alive}, out of memory {record['out_of_memory']}")
    return {k: v for k, v in record.items() if k != "requests"}


def stage_solve(inner: Inner) -> dict[str, Any]:
    """Run the tasks on the streams the pool allows until the generation stop.

    Competition: one arm, every hidden task, smallest opening first, through one refill queue;
    each finished task fills ``submission.json``. Development: the arms of ``dev_arms`` side by
    side, the streams split evenly, each arm taking the tasks in their order (so the arms of a
    task start close together). A task whose requests failed because the engine died is queued
    again once the server is back (at most ``health.max_requeue`` times).
    """
    run = inner.run
    spec = run.config["solve"]
    plan_s = time_plan(run.config, inner.mode)
    done = {(r["task_id"], r.get("arm")) for r in read_jsonl(run.out / "attempts.jsonl")
            if "error" not in r}
    arms = solve_arms(inner)
    planner = run.config["planner"]
    # The plan: competition's generation window in both modes, so a development run's pairs run
    # at the budget the scored rerun will use; N the hidden tasks (development: the visible
    # test file's), r the planner's rate until the first requests measure it.
    window = run.start + time_plan(run.config, COMPETITION)["generation_stop_s"] - _now()
    plan = plan_budget(inner.pool, inner.max_running, float(planner["rate_tokens_per_s"]),
                       max(window, 1.0), len(inner.plan_openings), inner.plan_openings,
                       len(arms), spec, planner)
    plan["arms"] = len(arms)
    inner.plan = plan
    inner.context = plan["context"]
    per_arm = plan["per_arm"]
    streams = plan["streams"]
    # Workers enough for any re-plan; the gates hold each arm to the plan's streams.
    workers = max(per_arm, min(int(inner.max_running or spec["max_streams"]),
                               int(spec["max_streams"])) // len(arms))
    inner.gates = {arm: Gate(per_arm) for arm in arms}
    end = run.start + plan_s["generation_stop_s"]
    clock = Clock(force_at=end - float(spec["force_reserve_s"]), end=end)
    queues = {arm: [t for t in inner.tasks if (t.task_id, arm) not in done] for arm in arms}
    requeued: dict[tuple[str, str], int] = {}
    lock = threading.Lock()
    skipped: list[str] = []
    stopped: list[str] = []
    kill = run.config["kill"]
    finished: list[dict[str, Any]] = []

    def killed() -> str | None:
        # The run's own stop (docs/runbooks/fn_probe.md, failure modes): too many failed or
        # degenerate attempts among the first ones means the stack is broken, not the tasks.
        # Only in the modes kill.modes names: in a scored rerun a stop saves nothing.
        if inner.mode not in kill["modes"] or len(finished) < int(kill["min_attempts"]):
            return None
        bad = [r for r in finished if "error" in r]
        degenerate = [r for r in finished if r.get("max_repeat_run", 0) >= int(kill["max_run"])]
        if len(bad) / len(finished) > float(kill["error_share"]):
            return f"{len(bad)} of {len(finished)} attempts failed"
        if len(degenerate) / len(finished) > float(kill["degenerate_share"]):
            return f"{len(degenerate)} of {len(finished)} attempts degenerate"
        return None

    def worker(arm: str, index: int) -> None:
        queue = queues[arm]
        gate = inner.gates[arm]
        while True:
            if not gate.acquire(clock.force_at - plan_s["min_start_s"]):
                return
            try:
                if not run_one(arm, index, queue):
                    return
            finally:
                gate.release()

    def run_one(arm: str, index: int, queue: list[Any]) -> bool:
        # One task on this stream; False when the stream should stop.
        while True:
            with lock:
                if not queue:
                    return False
                reason = killed() or ("the engine is dead and cannot restart"
                                      if inner.engine_dead else None)
                if reason or clock.force_at - _now() < plan_s["min_start_s"]:
                    if reason and reason not in stopped:
                        stopped.append(reason)
                    skipped.extend(f"{t.task_id}:{arm}" for t in queue)
                    queue.clear()
                    return False
                task = queue.pop(0)
            while not inner.engine_ready.wait(timeout=5) and not inner.engine_dead and (
                    _now() < clock.force_at):
                pass
            if not inner.engine_ready.is_set():
                with lock:
                    queue.insert(0, task)
                    skipped.extend(f"{t.task_id}:{arm}" for t in queue)
                    queue.clear()
                return False
            restarts_before = inner.restarts
            generation_before = inner.generation
            record = attempt_task(inner, task, clock, index, arm)
            hit = inner.restarts != restarts_before or not engine_alive(inner)
            if (arm == "tool" and "error" in record and inner.mode == COMPETITION and not hit
                    and spec["plain_after_tool_error"] and engine_alive(inner)):
                # The tool loop itself failed on this task: the plain request still answers it.
                append_jsonl(run.out / "tool_errors.jsonl", record, run.lock)
                record = attempt_task(inner, task, clock, index, "plain")
                record["after_tool_error"] = True
            if engine_failed(record, hit) and hit:
                key = (task.task_id, arm)
                # A server the health watch already restarted is alive: restart_engine then
                # returns at once.
                back = restart_engine(inner, f"engine failed during {task.task_id} ({arm})",
                                      generation_before)
                record["requeued"] = back and requeued.get(key, 0) < int(run.config["health"][
                    "max_requeue"])
                append_jsonl(run.out / "engine_failures.jsonl", record, run.lock)
                if record["requeued"]:
                    requeued[key] = requeued.get(key, 0) + 1
                    with lock:
                        queue.insert(0, task)
                    continue
            append_jsonl(run.out / "attempts.jsonl", record, run.lock)
            if inner.writer is not None and record.get("attempts"):
                inner.writer.update(task.task_id, record["attempts"])
            with lock:
                finished.append(record)
            return True

    if "tool" in arms and run.config["tool"].get("shared_verifier"):
        # One verifier pool for every tool stream (P5.3: share it across streams), a worker per
        # stream, instead of a process started per task.
        from arc26.agent.verify import PoolVerifier

        cfg = tool_module(run.config).tool_config(run.config["tool"].get("loop", {}))
        inner.verifier = PoolVerifier(workers, call_timeout_s=float(
            getattr(cfg, "verify_call_timeout_s", 2.0)))
    inner.watch = HealthWatch(inner)
    inner.watch.start()
    layout = {"mode": inner.mode, "arms": arms, "streams": streams, "per_arm": per_arm,
              "workers_per_arm": workers, "context": inner.context, "budget": plan,
              "pool_tokens": inner.pool, "max_running_requests": inner.max_running,
              "force_at_s": round(clock.force_at - run.start, 1),
              "end_s": round(clock.end - run.start, 1),
              "queued": {arm: len(q) for arm, q in queues.items()},
              "order": [t.task_id for t in inner.tasks]}
    write_json(run.out / "solve_plan.json", layout)
    jobs = [(arm, i) for arm in arms for i in range(workers)]
    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            list(pool.map(lambda job: worker(*job), jobs))
    finally:
        watch, inner.watch = inner.watch, None
        if watch is not None:
            watch.stop.set()
        if inner.verifier is not None:
            inner.verifier.close()
    rows = read_jsonl(run.out / "attempts.jsonl")
    return {**layout, "replan": inner.plan if inner.plan is not plan else None,
            "attempted": len(rows),
            "with_answers": sum(1 for r in rows if any(r.get("attempts") or [])),
            "not_started": skipped, "killed": stopped[0] if stopped else None,
            "restarts": inner.restarts,
            "submission_filled": None if inner.writer is None else len(inner.writer.answered)}


def summary_markdown(run: Run) -> str:
    """Render ``summary.md`` from the state and the readings written so far.

    Args:
        run: The run.

    Returns:
        Markdown.
    """
    lines = [f"# {run.config['run_id']}: Flash-Next on SGLang, 4 x L4", "",
             "| Stage | Status | Seconds | Note |", "|---|---|---:|---|"]
    for name, record in run.state.items():
        if isinstance(record, dict) and "status" in record:
            note = " ".join(str(record.get("error") or record.get("reason") or "").split())
            lines.append(f"| {name} | {record['status']} | {record['seconds']} | {note[:100]} |")
    boot = run.state.get("boot", {})
    lines += ["", f"- Mode: {run.state.get('mode')}; boot to healthy: {boot.get('ready_s')} s; "
              f"pool {boot.get('pool_tokens')} tokens; running-request limit "
              f"{boot.get('max_running_requests')}"]
    attempts = read_jsonl(run.out / "attempts.jsonl")
    for arm in sorted({a.get("arm") for a in attempts}, key=str):
        mine = [a for a in attempts if a.get("arm") == arm]
        lines.append(f"- Arm {arm}: {len(mine)} tasks attempted, "
                     f"{sum(1 for a in mine if a.get('forced'))} forced, "
                     f"{sum(1 for a in mine if 'error' in a)} errors; scored locally by "
                     "scripts/fn_report.py")
    solve = run.state.get("solve", {})
    if solve.get("submission_filled") is not None:
        lines.append(f"- submission.json: {solve['submission_filled']} test inputs answered")
    return "\n".join(lines) + "\n"


def finish_logs(run: Run) -> None:
    """Copy the engine log's tail and a bounded copy into the pulled folder; write the summary."""
    log = run.path(run.config["paths"]["engine_log"])
    if log.is_file():
        data = log.read_bytes()
        limit = int(run.config["engine"]["max_log_bytes"])
        kept = data if len(data) <= limit else data[: limit // 8] + b"\n[... cut ...]\n" + data[
            -(limit - limit // 8):]
        (run.out / "engine.log").write_bytes(kept)
        lines = data.decode("utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-int(run.config["engine"]["tail_lines"]):]) + "\n"
        (run.out / "engine_tail.log").write_text(tail, encoding="utf-8")
        write_json(run.out / "decode_lines.json", decode_lines("\n".join(lines)))
    (run.out / "summary.md").write_text(summary_markdown(run), encoding="utf-8")


def stop_server(inner: Inner) -> None:
    """Stop the monitor and the server's process group."""
    if inner.monitor is not None:
        inner.monitor.stop.set()
    kill_server(inner)


def stress_check(inner: Inner, spec: Mapping[str, Any], boot: dict[str, Any]) -> dict[str, Any]:
    """Run the boot stress check and its fallbacks; return the boot record that serves.

    Steps: ``stress`` (:func:`stage_stress`); on a failure, once, ``stress.fallback_flags`` (when
    set) join ``flag_overrides`` (a row in ``restarts.jsonl``), the server reboots
    (``boot_stress_fallback``), the template check and ``stress_fallback`` run again; on a
    failure without fallback flags or after them, the ladder's next rung boots
    (``boot_stress_retry``) with its template check, and no further stress check runs.

    Args:
        inner: The inner state.
        spec: The ``stress`` section.
        boot: The serving boot's stage record.

    Returns:
        The stage record of the boot that serves (``status`` not ``ok`` when none does).
    """
    run = inner.run
    tried = run.stage("stress", lambda: stage_stress(inner))
    if tried["status"] == "ok" or run.state.get("retried_stress"):
        return boot
    run.state["retried_stress"] = True
    fallback = dict(spec.get("fallback_flags") or {})
    if fallback:
        run.state.setdefault("flag_overrides", {}).update(fallback)
        append_jsonl(run.out / "restarts.jsonl", {"t": round(_now() - run.start, 1),
                                                  "reason": "the boot stress check failed",
                                                  "stress_fallback": fallback})
        kill_server(inner)
        boot = run.stage("boot_stress_fallback", lambda: stage_boot(inner))
        if boot["status"] != "ok":
            return boot
        run.stage("template_check_stress_fallback", lambda: stage_template_check(inner))
        again = run.stage("stress_fallback", lambda: stage_stress(inner))
        if again["status"] == "ok":
            return boot
    if next_rung(run, "the boot stress check failed") is None:
        return boot
    kill_server(inner)
    boot = run.stage("boot_stress_retry", lambda: stage_boot(inner))
    if boot["status"] == "ok":
        run.stage("template_check_stress_retry", lambda: stage_template_check(inner))
    return boot


def inner_main(state_path: Path) -> int:
    """The inner process: tasks, the tool self-test, boot, the first generation, the solve.

    Args:
        state_path: The outer process's ``state.json``, beside ``run.json``.

    Returns:
        0 when the stages ran (their failures are recorded), 1 when the boot or the first
        generation failed.
    """
    from arc26 import timing

    out = state_path.parent
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    run = Run(config=meta["config"], root=Path(meta["root"]), out=out, start=meta["start"],
              state=json.loads(state_path.read_text(encoding="utf-8")))
    timing.configure(out, worker="inner")
    paths = run.state.get("paths", {})
    inner = Inner(run=run, base=f"http://127.0.0.1:{run.config['engine']['port']}",
                  tok=make_tokenizer(run.config["tokenizer"], paths.get("model")),
                  mode=run.state.get("mode", DEVELOPMENT))
    code = 0
    try:
        load = run.stage("tasks", lambda: stage_tasks(inner))
        if load["status"] == "ok" and inner.mode == COMPETITION:
            target = run.path(run.config["paths"]["submission"])
            inner.writer = SubmissionWriter(target, inner.challenges)
        arms = run.config["solve"]["dev_arms"] if inner.mode == DEVELOPMENT else [
            run.config["solve"]["arm"]]
        if "tool" in arms:
            test = run.stage("tool_self_test", lambda: tool_self_test(inner))
            if test["status"] != "ok":
                inner.tool_ok = False
        # Before the boot, never instead of it: a refused flag set fails the boot anyway, and a
        # check that errs on a GPU box must not cost a run the boot would have served.
        # The edited engine's own import, KV4 first, revision 16's files after a fault.
        run.stage("import_check", lambda: stage_import_check(inner))
        run.stage("arg_check", lambda: stage_arg_check(inner))
        if (run.config.get("schema_check") or {}).get("enabled"):
            grammar = run.stage("schema_check", lambda: stage_schema_check(inner))
            if grammar["status"] != "ok":
                run.state["killed"] = "structured forcing grammar preflight failed"
                return 1
        boot = run.stage("boot", lambda: stage_boot(inner))
        if boot["status"] == "ok" and load["status"] == "ok":
            check = run.stage("template_check", lambda: stage_template_check(inner))
            # The first generation is where #36968 fails on sm89: without one token back, the
            # solve would only spend quota (development) or time (competition, which keeps its
            # placeholders and its retry below).
            generated = check.get("generate", {}).get("completion_tokens", 0)
            if ((check["status"] != "ok" or not generated)
                    and not run.state.get("retried_first_generation")
                    and next_rung(run, "the first generation failed") is not None):
                # The next rung of the boot ladder (boot_rungs), once.
                run.state["retried_first_generation"] = True
                kill_server(inner)
                boot = run.stage("boot_retry", lambda: stage_boot(inner))
                if boot["status"] == "ok":
                    check = run.stage("template_check_retry",
                                      lambda: stage_template_check(inner))
                    generated = check.get("generate", {}).get("completion_tokens", 0)
            if (check["status"] != "ok" or not generated) and inner.mode != COMPETITION:
                run.state["killed"] = "the first generation failed (template_check)"
                code = 1
            else:
                # Competition solves whatever the check said: the placeholders are the floor,
                # and a failed check costs nothing that a solve could lose.
                if check["status"] != "ok" or not generated:
                    run.state["first_generation_failed"] = True
                # P5.6-02's boot stress check: the solve's largest prefill and its decode batch
                # once, before any task. A failure reboots with stress.fallback_flags when set
                # (revision 22: the smaller prefill) and checks once more, else, or after that,
                # moves the boot one rung down the ladder.
                stress = run.config.get("stress") or {}
                if inner.mode in stress.get("modes", []) and stress.get("enabled"):
                    boot = stress_check(inner, stress, boot)
                if boot["status"] != "ok":
                    code = 1
                else:
                    # The batch sweep (P5.6-01): decode rates at 1 to N concurrent requests,
                    # before the solve and only in the modes listed (development).
                    if inner.mode in (run.config.get("sweep") or {}).get("modes", []):
                        run.stage("sweep", lambda: stage_sweep(inner))
                    if run.config.get("tool", {}).get("loop", {}).get("structured_forcing"):
                        probe = run.stage("schema_probe", lambda: stage_schema_probe(inner))
                        if probe["status"] != "ok":
                            run.state["killed"] = "live structured forcing probe failed"
                            return 1
                    run.stage("solve", lambda: stage_solve(inner))
        else:
            code = 1
        if inner.arg_refused and inner.mode != COMPETITION:
            run.state["arg_check_refused"] = inner.arg_refused
            code = 1
    finally:
        stop_server(inner)
        if inner.monitor is not None:
            run.state["monitor"] = {"gpu_peak_used_mib": inner.monitor.peak_used_mib,
                                    "metric_peaks": inner.monitor.peaks,
                                    "min_ram_available_gib": inner.monitor.min_ram_available_gib}
            pattern = run.config["engine"].get("free_pattern")
            log_path = run.path(run.config["paths"]["engine_log"])
            if pattern and log_path.is_file():
                # The engine's own free device memory while serving (P5.6-02's row).
                run.state["monitor"]["free_device"] = free_device_min(
                    log_path.read_text(encoding="utf-8", errors="replace"), pattern)
        run.save()
        finish_logs(run)
    return code


def outer_main(config: dict[str, Any], root: Path, commit: str, resume: bool,
               session_start: float | None = None) -> int:
    """The outer process: inputs, Python, install, CUDA, then the inner process.

    Args:
        config: The resolved config.
        root: Prefix of absolute paths.
        commit: The code's commit (recorded in ``run.json``).
        resume: Keep the earlier attempts in the output folder.
        session_start: Wall time the notebook started (its first cell), from which the time
            plan counts; None counts from this process's start.

    Returns:
        The inner process's exit code, or 1 when a stage before it failed.
    """
    out = root / config["paths"]["out"].lstrip("/")
    if out.exists() and not resume:
        for name in ("attempts.jsonl", "transcripts.jsonl", "memory.jsonl",
                     "server_metrics.jsonl", "engine_failures.jsonl", "restarts.jsonl",
                     "engine_commands.jsonl"):
            (out / name).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    mode = run_mode(config, os.environ)
    run = Run(config=config, root=root, out=out, start=session_start or _now(),
              state={"mode": mode})
    write_json(out / "run.json", {"schema": SCHEMA, "run_id": config["run_id"], "commit": commit,
                                  "start": run.start, "mode": mode, "root": str(root),
                                  "config": config, "python": sys.version,
                                  "outer_started_s": round(_now() - run.start, 1)})
    precache = None
    pre = config["precache"]
    inputs = run.stage("inputs", lambda: stage_inputs(run))
    run.state["scratch"] = str(run.path(config["paths"]["scratch"]))
    if inputs["status"] == "ok":
        # The wheel copy first, alone on the mount; the model's precache only after it ends, so
        # the two never share the mount's bandwidth (the install then reads local disk).
        run.stage("local_inputs", lambda: stage_local_inputs(run))
    if inputs["status"] == "ok" and pre["enabled"]:
        folders = [Path(run.state["paths"][name]) for name in pre["inputs"]
                   if run.state["paths"].get(name)]
        precache = Precache(folders, int(pre["threads"]), int(pre["chunk_mib"]) * 1024**2,
                            float(pre["max_s"]))
        run.state["precache_started_s"] = round(_now() - run.start, 3)
        precache.start()
    if inputs["status"] == "ok":
        run.stage("python", lambda: stage_python(run))
        run.stage("install", lambda: stage_install(run))
        run.stage("cuda", lambda: stage_cuda(run))
    venv_python = run.state.get("venv_python")
    code = 1
    if venv_python:
        paths = run.state.get("paths", {})
        script = Path(paths["code"]) / "fn_probe.py" if paths.get("code") else Path(__file__)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(run.state.get("env_overrides", {}))
        if not config["install"].get("enabled", True):
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        run.save()
        timeout = max(run.left(), 60)
        print(f"== inner process ({round(timeout)} s left)", flush=True)
        try:
            done = subprocess.run([venv_python, str(script), "--inner", str(out / "state.json")],
                                  env=env, timeout=timeout, check=False)
            code = done.returncode
        except subprocess.TimeoutExpired:
            code = 124
        run.state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    if precache is not None:
        run.state["precache"] = precache.reading()
    run.state["outer"] = {"exit_inner": code, "wall_s": round(_now() - run.start, 1)}
    run.save()
    finish_logs(run)
    return code


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, help="YAML or JSON config; default: embedded")
    parser.add_argument("--root", type=Path, default=Path("/"),
                        help="prefix of every absolute path (the smoke run and tests)")
    parser.add_argument("--smoke", action="store_true",
                        help="CPU smoke: the smoke section, a fake SGLang server, fake inputs")
    parser.add_argument("--resume", action="store_true",
                        help="keep the attempts already written")
    parser.add_argument("--session-start", type=float, default=None,
                        help="wall time the notebook started; the time plan counts from it")
    parser.add_argument("--commit", default=None, help="the code's commit for run.json")
    parser.add_argument("--inner", type=Path, help=argparse.SUPPRESS)
    return parser


def build_smoke_tree(root: Path, config: Mapping[str, Any]) -> None:
    """Write the smoke's fake inputs: a challenges file of tiny tasks and empty folders.

    Args:
        root: The smoke root.
        config: The smoke config.
    """
    inputs = config["inputs"]
    competition = root / inputs["competition"]["path"].lstrip("/")
    competition.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for k, task_id in enumerate(config["solve"]["task_ids"]):
        grid = [[(k + i + j) % 10 for j in range(3)] for i in range(2)]
        tasks[task_id] = {"train": [{"input": grid, "output": grid}],
                          "test": [{"input": [[(k + 1) % 10, 0], [0, 1]]}]}
    for name in set(config["solve"]["challenges"].values()) | {inputs["competition"]["marker"]}:
        (competition / name).write_text(json.dumps(tasks), encoding="utf-8")
    for name in ("model",):
        (root / inputs[name]["path"].lstrip("/")).mkdir(parents=True, exist_ok=True)
        (root / inputs[name]["path"].lstrip("/") / inputs[name]["marker"]).write_text(
            "{}", encoding="utf-8")
    # A fake wheelhouse for the local copy (never installed: the smoke's install is off).
    wheelhouse = root / inputs["wheelhouse"]["path"].lstrip("/")
    (wheelhouse / "wheels").mkdir(parents=True, exist_ok=True)
    (wheelhouse / inputs["wheelhouse"]["marker"]).write_text("fake==0.0\n", encoding="utf-8")
    (wheelhouse / "wheels" / "fake-0.0-py3-none-any.whl").write_bytes(
        bytes(range(256)) * (3 * 4096 + 1))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe (outer process), or its inner process with ``--inner``.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The exit code: 0 when the stages ran, 1 when the boot or a stage before it failed.
    """
    args = build_parser().parse_args(argv)
    if args.inner is not None:
        return inner_main(args.inner)
    config = load_config(args.config)
    root = args.root.resolve()
    if args.smoke:
        if root == Path("/"):
            raise SystemExit("--smoke needs --root, a scratch folder for the fake inputs")
        config = apply_smoke(config)
        build_smoke_tree(root, config)
    commit = args.commit or config.get("build", {}).get("commit") or "unknown"
    code = outer_main(config, root, commit, args.resume, args.session_start)
    print(f"exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
