r"""The CPU smoke's fake server for ``scripts/fn_kvab.py``: ``fn_fake_sglang.py`` plus logprobs.

Why: run ``P5.5-01``'s every path (the teacher-forced pass with prompt log-probabilities and
top-k lists, non-streaming greedy requests, a larger pool for the 4-bit variant) must run on the
CPU before any credit is spent (BRIEF A8), and ``fn_fake_sglang.py`` is part of version 11's
code dataset, whose bytes stay as they are. This file wraps that fake's handler: it answers the
non-streaming ``/generate`` requests itself; every streaming request is the base fake's.

The fake "model" is deterministic. At position $t$ of the prompt $x$, the top-$k$ list has
log-probabilities $\lambda_i = -(i + 1) \ln 2 - \epsilon_t$ for $i = 0, \dots, k - 1$ over
tokens $(x_t + i) \bmod 256$ (probabilities $2^{-(i+1)}$, summing under 1), so the sent token is
the top-1 and $\ell_t = \lambda_0$; with
``ARC26_KV4`` set in the server's environment, $\epsilon_t = 10^{-3} \cdot (t \bmod 7)$, a small
shift that the A/B comparison reads as a difference, else $\epsilon_t = 0$. The pool follows
the same settings as the real boots' (``pool_argv``).

Usage (the smoke's run config names it as the engine's script)::

    python scripts/fn_fake_kvab.py --port 30000 --model-path <dir> [--fake-pool 20000]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "_fn_fake_sglang_for_kvab", Path(__file__).with_name("fn_fake_sglang.py"))
fake = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fake
_spec.loader.exec_module(fake)
_base_handler = fake.make_handler
LN2 = 0.6931471805599453


def shift(t: int) -> float:
    """Return the fake 4-bit cache's log-probability shift at position ``t`` (zero without it)."""
    return 1e-3 * (t % 7) if os.environ.get("ARC26_KV4") else 0.0


def logprob_meta(ids: Sequence[int], top_k: int, start: int) -> dict[str, Any]:
    """Return ``input_token_logprobs`` and ``input_top_logprobs`` as SGLang shapes them.

    Args:
        ids: The prompt's ids.
        top_k: ``top_logprobs_num``.
        start: ``logprob_start_len``; positions before it are left out.

    Returns:
        The two lists (the first position's log-probability and top list are None).
    """
    tokens, tops = [], []
    for t in range(start, len(ids)):
        if t == 0:
            tokens.append([None, int(ids[0]), None])
            tops.append(None)
            continue
        eps = shift(t)
        top = [[-(i + 1) * LN2 - eps, (int(ids[t]) + i) % 256, None]
               for i in range(max(top_k, 1))]
        tokens.append([top[0][0], int(ids[t]), None])
        tops.append(top[:top_k] if top_k else None)
    return {"input_token_logprobs": tokens, "input_top_logprobs": tops}


def make_handler(state: Any, args: Any) -> type[BaseHTTPRequestHandler]:
    """Return the base fake's handler with ``/generate``'s ``return_logprob`` answered."""
    base = _base_handler(state, args)

    class Handler(base):  # type: ignore[misc, valid-type]
        def do_POST(self) -> None:
            if self.path != "/generate":
                return super().do_POST()
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) or b"{}"
            body = json.loads(raw)
            if body.get("stream"):
                real, self.rfile = self.rfile, _Replay(raw)
                try:
                    return super().do_POST()
                finally:
                    self.rfile = real
            ids = body.get("input_ids") or fake.TOK.encode(body.get("text", ""))
            if not body.get("return_logprob"):
                # A non-streaming reply takes the base fake's per-token delay, so a decode
                # request has a finite wall rate (the base answers these at once).
                params = body.get("sampling_params", {})
                ignore = bool(params.get("ignore_eos"))
                out = self.reply_ids(list(ids), int(params.get("max_new_tokens", 16)), ignore,
                                     list(params.get("stop") or []))
                time.sleep(args.token_delay * len(out))
                self._send(200, {"text": fake.TOK.decode(out), "output_ids": out, "meta_info": {
                    "prompt_tokens": len(ids), "completion_tokens": len(out), "cached_tokens": 0,
                    "finish_reason": {"type": "length" if ignore else "stop"}}})
                return None
            meta = logprob_meta(ids, int(body.get("top_logprobs_num") or 0),
                                max(int(body.get("logprob_start_len", -1)), 0))
            meta.update(prompt_tokens=len(ids), completion_tokens=1, cached_tokens=0,
                        finish_reason={"type": "length"})
            self._send(200, {"text": "", "output_ids": [0], "meta_info": meta})
            return None

    return Handler


class _Replay:
    """A request body read once, offered again to the base handler's ``rfile.read``."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self, size: int = -1) -> bytes:
        """Return the body (all of it, whatever ``size``: the base reads exactly its length)."""
        data, self.data = self.data, b""
        return data


# The fake pool's factors, as the real boots differ: the 4-bit cache by its variable (KV4
# 6,912 / 4,224 B per token is 1.64, V only 1.24), the vision skip by its config key.
POOL_FACTORS = {"1": 1.7, "kv": 1.7, "v": 1.25}
VISION_FACTOR = 1.18


def pool_argv(argv: Sequence[str], environ: dict[str, str] | None = None) -> list[str]:
    """Return the server's flags with ``--fake-pool`` scaled as the real cache and flags would.

    Why: the real boots differ only by ``ARC26_KV4`` and C's vision flags (the edits select the
    packed pool by the variable), so the smoke's variants carry the same settings and the fake
    derives its pool from them.
    """
    environ = dict(os.environ if environ is None else environ)
    argv = list(argv)
    factor = POOL_FACTORS.get(environ.get("ARC26_KV4", ""), 1.0)
    if "--json-model-override-args" in argv:
        value = argv[argv.index("--json-model-override-args") + 1]
        if json.loads(value).get("language_model_only"):
            factor *= VISION_FACTOR
    if factor == 1.0:
        return argv
    pool = int(argv[argv.index("--fake-pool") + 1]) if "--fake-pool" in argv else 200000
    argv = [a for i, a in enumerate(argv)
            if a != "--fake-pool" and (i == 0 or argv[i - 1] != "--fake-pool")]
    return [*argv, "--fake-pool", str(round(pool * factor))]


def main(argv: Sequence[str] | None = None) -> int:
    """Serve as ``fn_fake_sglang.py`` does, with this file's handler and the scaled pool."""
    fake.make_handler = make_handler
    return fake.main(pool_argv(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
