"""A fake SGLang server for the CPU smoke of ``scripts/fn_probe.py`` (task P5.2).

Why: the probe's every path (boot and its log lines, ``/get_server_info``, ``/metrics``, the
chat endpoint's prompt count, streaming ``/generate`` with token ids, a cut at the cap, the
forcing continuation, a cut at the deadline) must run on CPU before any quota is spent
(BRIEF A8). It speaks SGLang's native API closely enough for the probe and nothing more.

The "model" is deterministic: it thinks ``--think-tokens`` bytes of filler per request (more
than the cap for every second task, so the forcing turn runs), then closes the think block and
writes one answer block per test input with the test input itself as the grid. Token ids are
:class:`fn_probe.FakeTokenizer`'s, so the probe's ids round-trip exactly. Unknown flags (the real
server's) are accepted and ignored.

Usage (the probe starts it from the smoke config)::

    python scripts/fn_fake_sglang.py --port 30000 --model-path <dir> [--think-tokens 300]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location("_fn_probe", Path(__file__).with_name("fn_probe.py"))
_probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _probe
_spec.loader.exec_module(_probe)
TOK = _probe.FakeTokenizer()
EOS_TEXT = "<|im_end|>"
# The identity program the fake "model" tests with its tool: the smokes' demonstrations map each
# input to itself, so it passes every one.
TOOL_CALL = ("<tool_call>\n<function=python>\n<parameter=code>\n"
             "def transform(grid):\n    return grid\nprint(check(transform))\n"
             "</parameter>\n</function>\n</tool_call>")
TEST_GRID = re.compile(r"Test input (\d+) \((\d+)x(\d+)\):\n((?:[0-9]+\n?)+)")


def answer_text(prompt: str) -> str:
    """Return answer blocks echoing each test input of the prompt's task.

    Args:
        prompt: The decoded prompt.

    Returns:
        One ``answer`` block per test input found, as the direct-answer prompt asks.
    """
    blocks = []
    for match in TEST_GRID.finditer(prompt):
        rows = [[int(c) for c in row] for row in match.group(4).strip().splitlines()]
        body = json.dumps({"test_index": int(match.group(1)), "grid": rows})
        blocks.append(f"```answer\n{body}\n```")
    return "\n".join(blocks) or "```answer\n{\"test_index\": 0, \"grid\": [[0]]}\n```"


def schema_answer_text(prompt: str, schema: dict[str, Any]) -> str:
    """Echo test inputs in the structured schema for plumbing checks, without masking claims."""
    grids = {match.group(1): match.group(4).strip().splitlines()
             for match in TEST_GRID.finditer(prompt)}
    keys = schema["properties"]["answers"]["required"]
    return json.dumps({"answers": {key: [grids.get(key, ["12"])] for key in keys}})


class State:
    """Counters the fake reports through ``/metrics``."""

    def __init__(self, pool: int, running_limit: int) -> None:
        self.pool = pool
        self.running_limit = running_limit
        self.running = 0
        self.generated = 0
        self.requests = 0
        # /generate requests in flight (--die-over-inflight).
        self.inflight = 0
        self.lock = threading.Lock()
        self.cache: list[list[int]] = []


def make_handler(state: State, args: argparse.Namespace) -> type[BaseHTTPRequestHandler]:
    """Return the request handler class bound to the server's state and options."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *values: Any) -> None:
            return

        def _send(self, status: int, body: Any, kind: str = "application/json") -> None:
            data = (json.dumps(body) if kind == "application/json" else body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {})
            elif self.path == "/get_server_info":
                self._send(200, {"model_path": args.model_path, "internal_states": [
                    {"max_total_num_tokens": state.pool,
                     "max_running_requests": state.running_limit}]})
            elif self.path == "/metrics":
                text = (f"sglang:num_running_reqs{{model=\"fake\"}} {state.running}\n"
                        f"sglang:token_usage{{model=\"fake\"}} 0.1\n"
                        f"sglang:generation_tokens_total{{model=\"fake\"}} {state.generated}\n")
                self._send(200, text, "text/plain")
            else:
                self._send(404, {"error": self.path})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_POST(self) -> None:
            body = self._body()
            if self.path == "/v1/chat/completions":
                kwargs = {k: body[k] for k in ("reasoning_effort",) if k in body}
                text = TOK.render(body["messages"], **kwargs)
                self._send(200, {"usage": {"prompt_tokens": len(TOK.encode(text)),
                                           "completion_tokens": 1}})
            elif self.path == "/generate":
                self.generate(body)
            else:
                self._send(404, {"error": self.path})

        def reply_ids(self, prompt_ids: list[int], max_new: int, ignore_eos: bool,
                      stop: Sequence[str] = ()) -> list[int]:
            prompt = TOK.decode(prompt_ids)
            calls = prompt.count("<tool_response>")
            if prompt.endswith("</think>\n\n") or "</think>" in prompt.rsplit("<think>", 1)[-1]:
                ids = TOK.encode(answer_text(prompt) + EOS_TEXT)
            elif "</tool_call>" in stop and calls < args.tool_calls:
                # The tool arm: think a little, then call python; the request stops after the
                # call's closing tag (kept, as no_stop_trim).
                ids = TOK.encode("think " * 5 + "\n" + TOOL_CALL)
            else:
                # Every second task (by its first test input) thinks past the cap.
                match = TEST_GRID.search(prompt)
                long = match is not None and match.group(4)[0] in "13579"
                think = args.think_tokens * (10 if long else 1)
                filler = "a" * think if args.degenerate else "think " * (think // 6)
                ids = TOK.encode(filler) + TOK.encode("\n</think>\n\n")
                ids += TOK.encode(answer_text(prompt) + EOS_TEXT)
            if ignore_eos:
                ids = (ids * (max_new // max(len(ids), 1) + 1))
            return ids[:max_new]

        def generate(self, body: dict[str, Any]) -> None:
            with state.lock:
                state.inflight += 1
            if args.die_over_inflight:
                # Requests sent together overlap here, whatever the thread timing.
                time.sleep(0.5)
            with state.lock:
                crowded = args.die_over_inflight and state.inflight > args.die_over_inflight
            if crowded:
                # P5.6-01's fault: an out-of-memory death when many prefills arrive at once.
                print("torch.OutOfMemoryError: CUDA out of memory (fake)", flush=True)
                os._exit(3)
            try:
                self._generate(body)
            finally:
                with state.lock:
                    state.inflight -= 1

        def _generate(self, body: dict[str, Any]) -> None:
            with state.lock:
                state.requests += 1
                dying = args.die_after and state.requests > args.die_after
            if dying:
                # A dead engine: the process ends without answering (os._exit skips cleanup).
                print("fake engine: dying on request", state.requests, flush=True)
                os._exit(3)
            params = body.get("sampling_params", {})
            prompt_ids = body.get("input_ids") or TOK.encode(body.get("text", ""))
            max_new = int(params.get("max_new_tokens", 16))
            refusal = None
            if args.context_length and len(prompt_ids) + max_new >= args.context_length:
                # SGLang's refusal without --allow-auto-truncate, as an error chunk.
                refusal = (f"Requested token count exceeds the model's maximum context length "
                           f"of {args.context_length} tokens.")
            elif args.fail_one_token and max_new == 1:
                refusal = "the first generation failed (fake)"
            if refusal is not None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {"error": {"message": refusal, "code": 400}}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return
            stop = list(params.get("stop") or [])
            if "json_schema" in params:
                schema = json.loads(params["json_schema"])
                reply = schema_answer_text(TOK.decode(prompt_ids), schema) + EOS_TEXT
                ids = TOK.encode(reply)[:max_new]
            else:
                ids = self.reply_ids(list(prompt_ids), max_new,
                                     bool(params.get("ignore_eos")), stop)
            text = TOK.decode(ids)
            ended = text.endswith(EOS_TEXT) or any(text.endswith(s) for s in stop)
            finish = "stop" if ended and not params.get("ignore_eos") else "length"
            cached = 0
            with state.lock:
                for old in state.cache:
                    n = 0
                    while n < min(len(old), len(prompt_ids)) and old[n] == prompt_ids[n]:
                        n += 1
                    cached = max(cached, n)
                state.cache = (state.cache + [list(prompt_ids) + ids])[-32:]
                state.running += 1
            try:
                if not body.get("stream"):
                    meta = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(ids),
                            "cached_tokens": cached, "finish_reason": {"type": finish}}
                    self._send(200, {"text": TOK.decode(ids), "output_ids": ids,
                                     "meta_info": meta})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                step = args.stream_interval
                marker = Path(args.model_path or ".") / ".fake_died_mid_stream"
                with state.lock:
                    # Once per model folder: the restarted fake serves normally.
                    dies_here = (args.die_mid_stream and state.requests == args.die_mid_stream
                                 and not marker.exists())
                if dies_here:
                    marker.write_text("1")
                for end in range(step, len(ids) + step, step):
                    if dies_here and end > len(ids) // 2:
                        # A hang or crash mid-reply: the stream ends without a finish reason.
                        print("fake engine: dying mid-stream", flush=True)
                        os._exit(3)
                    part = ids[:end]
                    done = end >= len(ids)
                    meta = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(part),
                            "cached_tokens": cached,
                            "finish_reason": {"type": finish} if done else None}
                    # SGLang's default repeats the reply so far; the flag sends only the new part.
                    sent = ids[end - step:end] if args.incremental_streaming_output else part
                    chunk = {"text": TOK.decode(sent), "output_ids": sent, "meta_info": meta}
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                    with state.lock:
                        state.generated += min(step, len(part))
                    time.sleep(args.token_delay * step)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                # One decode line per finished request as well, so a short smoke always has one.
                print(f"Decode batch. #running-req: {state.running}, #token: {len(ids)}, token "
                      "usage: 0.10, gen throughput (token/s): 100.00, #queue-req: 0", flush=True)
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            finally:
                with state.lock:
                    state.running -= 1

    return Handler


class FakeServerArgs:
    """The fake's stand-in for the fork's ``ServerArgs``, for the probe's argument check.

    Why: the smoke must run ``stage_arg_check`` end to end without SGLang; this class refuses
    the one flag the fork was seen to refuse for Flash-Next (P5.2-03, 29 Oct), with the fork's
    message, and accepts every other flag.

    Args:
        argv: The server's flags.
    """

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = list(argv)

    def resolve_once(self) -> None:
        """Resolve the flags as the fork's ``resolve_once`` does, for the refused one only.

        Raises:
            ValueError: If ``--language-model-only`` is set.
        """
        if "--language-model-only" in self.argv:
            raise ValueError("--language-model-only does not support "
                             "['Qwen4ExpForConditionalGeneration']. Supported: "
                             "['MuseGlimmerForConditionalGeneration'].")


def prepare_server_args(argv: Sequence[str]) -> FakeServerArgs:
    """Return the fake's server arguments (``sglang.srt.server_args.prepare_server_args``).

    Args:
        argv: The server's flags.

    Returns:
        The unresolved arguments.
    """
    return FakeServerArgs(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Serve until killed, printing SGLang-like boot and decode log lines."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--think-tokens", type=int, default=120)
    parser.add_argument("--token-delay", type=float, default=0.0002)
    parser.add_argument("--fake-pool", type=int, default=200000)
    parser.add_argument("--boot-delay", type=float, default=0.3)
    parser.add_argument("--stream-interval", type=int, default=1)
    # A broken kernel's symptom: one token repeated (the probe's degenerate stop).
    parser.add_argument("--degenerate", action="store_true")
    # Tool calls per task when the request stops at </tool_call> (the tool arm).
    parser.add_argument("--tool-calls", type=int, default=1)
    # The model's tokenizer (the CPU twin), else the byte-level fake of fn_probe.py.
    parser.add_argument("--tokenizer-dir", default="")
    # The process exits after this many /generate requests (the engine-death test); 0: never.
    parser.add_argument("--die-after", type=int, default=0)
    # The process exits halfway through the reply of this /generate request (1-based); 0: never.
    parser.add_argument("--die-mid-stream", type=int, default=0)
    # The process exits, printing an out-of-memory line, when more than this many /generate
    # requests are in flight at once (P5.6-02's stress check test); 0: never.
    parser.add_argument("--die-over-inflight", type=int, default=0)
    # SGLang's context refusal: input + max_new at or above this gets an error chunk; 0: off.
    parser.add_argument("--context-length", type=int, default=0)
    # Every one-token request gets an error chunk (the template check's first generation).
    parser.add_argument("--fail-one-token", action="store_true")
    parser.add_argument("--incremental-streaming-output", action="store_true")
    # NAME=VALUE: the process exits at its start when the variable has that value (a stand-in for
    # a boot fault of one engine path, P5.2-01 revision 17's KV4 cache: ARC26_KV4=1).
    parser.add_argument("--fail-if-env", default="")
    args, _unknown = parser.parse_known_args(argv)
    if args.fail_if_env:
        name, _, value = args.fail_if_env.partition("=")
        if os.environ.get(name) == value:
            print(f"fake engine: refusing to start with {name}={value}", flush=True)
            return 4
    global TOK
    if args.tokenizer_dir:
        TOK = _probe.HFTokenizer(args.tokenizer_dir)
    state = State(args.fake_pool, args.max_running_requests)
    time.sleep(args.boot_delay)
    print(f"KV Cache is allocated. #tokens: {state.pool}, K size: 0.50 GB, V size: 0.50 GB",
          flush=True)
    print(f"max_total_num_tokens={state.pool}, chunked_prefill_size=4096, "
          f"max_prefill_tokens=8192, max_running_requests={state.running_limit}, "
          "context_len=81920, available_gpu_mem=1.20 GB", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args))
    server.daemon_threads = True

    def ticker() -> None:
        last = 0
        while True:
            time.sleep(0.2)
            with state.lock:
                running, made = state.running, state.generated
            if running:
                rate = (made - last) / 0.2
                print(f"Decode batch. #running-req: {running}, #token: {made}, token usage: "
                      f"0.10, gen throughput (token/s): {rate:.2f}, #queue-req: 0", flush=True)
            last = made

    threading.Thread(target=ticker, daemon=True).start()
    print("The server is fired up and ready to roll!", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
