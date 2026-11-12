"""Tests of the OpenAI-compatible engine client (arc26/agent/engine.py, P6.2).

A stdlib HTTP server on localhost plays vLLM: each request pops the next scripted behavior (a
JSON reply, an HTTP error, a delay, or a dropped connection), and the server records the request
bodies and headers and its peak number of concurrent requests.
"""

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

httpx = pytest.importorskip("httpx")

from arc26.agent.engine import (  # noqa: E402
    QWEN38_EFFORT_NOTES,
    EngineConfig,
    EngineError,
    OpenAIEngine,
    parse_completion,
)
from arc26.agent.loop import AttemptContext, attempt_context  # noqa: E402
from arc26.agent.types import Message  # noqa: E402
from arc26.timing import TimingLogger, read_records  # noqa: E402

MESSAGES = (Message("system", "rules"), Message("user", "task"))


def reply(text="ok", reasoning=None, field="reasoning", usage=(12, 34), finish="stop"):
    message = {"role": "assistant", "content": text}
    if reasoning is not None:
        message[field] = reasoning
    data = {"id": "x", "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage is not None:
        data["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        }
    return data


class FakeServer:
    """A threaded HTTP/1.1 server that answers POSTs with scripted behaviors."""

    def __init__(self, behaviors=(), port=0, default=None):
        self.behaviors = list(behaviors)
        self.default = default or ("json", reply(), 200, 0.0)
        self.bodies = []
        self.headers = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                with fake.lock:
                    fake.bodies.append(body)
                    fake.headers.append(dict(self.headers))
                    behavior = fake.behaviors.pop(0) if fake.behaviors else fake.default
                    fake.active += 1
                    fake.peak = max(fake.peak, fake.active)
                try:
                    fake.respond(self, behavior)
                finally:
                    with fake.lock:
                        fake.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )

    @staticmethod
    def respond(handler, behavior):
        kind = behavior[0]
        if kind == "drop":
            handler.close_connection = True
            return
        _, payload, status, delay = behavior
        if delay:
            time.sleep(delay)
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def config(url, **overrides):
    values = {
        "base_url": url,
        "model": "qwen38",
        "max_retries": 3,
        "backoff_s": 0.01,
        "backoff_max_s": 0.05,
        "read_timeout_s": 10.0,
        "connect_timeout_s": 2.0,
    }
    values.update(overrides)
    return EngineConfig(**values)


async def complete(engine, **kwargs):
    async with engine:
        return await engine.complete(
            MESSAGES,
            max_tokens=kwargs.pop("max_tokens", 100),
            temperature=kwargs.pop("temperature", 1.0),
            **kwargs,
        )


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_completion_fields_and_request_body():
    with FakeServer(
        [("json", reply("```python\nx\n```", "thinking...", usage=(100, 250)), 200, 0)]
    ) as srv:
        engine = OpenAIEngine(config(srv.url))
        out = asyncio.run(
            complete(engine, max_tokens=512, temperature=0.7, seed=7, reasoning_effort="low")
        )
    assert out.text == "```python\nx\n```"
    assert out.reasoning == "thinking..."
    assert (out.prompt_tokens, out.completion_tokens) == (100, 250)
    assert out.finish_reason == "stop"
    assert out.seconds >= 0
    body = srv.bodies[0]
    assert body["model"] == "qwen38"
    assert body["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    assert (body["max_tokens"], body["temperature"], body["seed"]) == (512, 0.7, 7)
    # Default transport: the card's top-level field; preserve_thinking off (no reasoning field
    # in Message to send back).
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"preserve_thinking": False}


def test_effort_in_template_kwargs_and_thinking_switch():
    engine = OpenAIEngine(
        EngineConfig(
            effort_transport="template",
            thinking=False,
            template_kwargs={},
            sampling={"top_p": 0.95, "top_k": 20},
            extra_body={"include_reasoning": True},
        )
    )
    body = engine.request_body(MESSAGES, max_tokens=10, temperature=1.0, reasoning_effort="xhigh")
    assert "reasoning_effort" not in body
    assert body["chat_template_kwargs"] == {
        "preserve_thinking": False,
        "enable_thinking": False,
        "reasoning_effort": "xhigh",
    }
    assert (body["top_p"], body["top_k"], body["include_reasoning"]) == (0.95, 20, True)
    assert "seed" not in body


def test_no_effort_and_no_template_kwargs_sends_neither():
    engine = OpenAIEngine(EngineConfig(preserve_kwarg=None))
    body = engine.request_body(MESSAGES, max_tokens=10, temperature=1.0)
    assert "reasoning_effort" not in body
    assert "chat_template_kwargs" not in body


def test_custom_parameter_names():
    engine = OpenAIEngine(
        EngineConfig(thinking=True, thinking_kwarg="thinking", effort_kwarg="effort")
    )
    body = engine.request_body(MESSAGES, max_tokens=10, temperature=1.0, reasoning_effort="low")
    assert body["effort"] == "low"
    assert body["chat_template_kwargs"]["thinking"] is True


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_read_from_either_field(field):
    out = parse_completion(reply("a", "why", field=field), 0.5)
    assert out.reasoning == "why"
    assert out.seconds == 0.5


def test_null_content_becomes_empty_text_and_finish_reason_kept():
    data = reply(None, "all budget spent thinking", finish="length")
    out = parse_completion(data, 0.0)
    assert out.text == ""
    assert out.finish_reason == "length"


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"choices": [], "usage": {}},
        reply(usage=None),
        {"choices": [{"message": {}}]},
        {"choices": [{"message": "hi"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        None,
    ],
)
def test_malformed_responses_raise(data):
    with pytest.raises(EngineError, match="malformed"):
        parse_completion(data, 0.0)


def test_timing_record_per_call_with_puzzle_id(tmp_path):
    timing = TimingLogger(tmp_path, "engine")
    with FakeServer() as srv:
        engine = OpenAIEngine(config(srv.url), timing=timing)

        async def run():
            async with engine:
                with attempt_context(AttemptContext("t1", "t1/p1/0", 1)):
                    await engine.complete(MESSAGES, max_tokens=10, temperature=1.0)
                await engine.complete(MESSAGES, max_tokens=10, temperature=1.0)

        asyncio.run(run())
    records, skipped = read_records([timing.path])
    assert skipped == 0
    assert [(r.puzzle_id, r.op, r.tokens) for r in records] == [
        ("t1", "engine", 34),
        (None, "engine", 34),
    ]
    assert all(r.seconds >= 0 for r in records)


def test_retries_until_the_server_comes_up():
    port = free_port()
    srv = FakeServer(port=port)
    timer = threading.Timer(0.3, srv.thread.start)
    timer.start()
    try:
        engine = OpenAIEngine(
            config(
                f"http://127.0.0.1:{port}/v1", max_retries=50, backoff_s=0.02, backoff_max_s=0.05
            )
        )
        out = asyncio.run(complete(engine))
    finally:
        timer.join()
        srv.__exit__(None, None, None)
    assert out.text == "ok"
    assert len(srv.bodies) == 1


def test_connection_errors_exhaust_retries():
    engine = OpenAIEngine(config(f"http://127.0.0.1:{free_port()}/v1", max_retries=2))
    with pytest.raises(EngineError, match="no connection after 3 tries"):
        asyncio.run(complete(engine))


def test_dropped_connection_is_retried():
    with FakeServer([("drop",)]) as srv:
        out = asyncio.run(complete(OpenAIEngine(config(srv.url))))
    assert out.text == "ok"
    assert len(srv.bodies) == 2


def test_http_errors_are_not_retried():
    error = ("json", {"error": "This model's maximum context length is 1000 tokens"}, 400, 0)
    with FakeServer([error]) as srv:
        with pytest.raises(EngineError, match="HTTP 400.*maximum context length"):
            asyncio.run(complete(OpenAIEngine(config(srv.url))))
    assert len(srv.bodies) == 1


def test_server_errors_are_not_retried():
    with FakeServer([("json", {"error": "boom"}, 500, 0)]) as srv:
        with pytest.raises(EngineError, match="HTTP 500"):
            asyncio.run(complete(OpenAIEngine(config(srv.url))))
    assert len(srv.bodies) == 1


def test_read_timeout_is_not_retried():
    with FakeServer([("json", reply(), 200, 1.0)]) as srv:
        engine = OpenAIEngine(config(srv.url, read_timeout_s=0.2))
        with pytest.raises(EngineError, match="request failed"):
            asyncio.run(complete(engine))
    assert len(srv.bodies) == 1


def test_non_json_response_raises():
    with FakeServer([("json", b"<html>oops</html>", 200, 0)]) as srv:
        with pytest.raises(EngineError, match="not JSON"):
            asyncio.run(complete(OpenAIEngine(config(srv.url))))


def test_pool_bounds_concurrent_requests():
    with FakeServer(default=("json", reply(), 200, 0.1)) as srv:
        engine = OpenAIEngine(config(srv.url, max_connections=2))

        async def run():
            async with engine:
                return await asyncio.gather(
                    *(engine.complete(MESSAGES, max_tokens=5, temperature=1.0) for _ in range(6))
                )

        outs = asyncio.run(run())
    assert len(outs) == 6
    assert len(srv.bodies) == 6
    assert srv.peak <= 2


def test_api_key_is_sent_as_bearer_token():
    with FakeServer() as srv:
        asyncio.run(complete(OpenAIEngine(config(srv.url, api_key="secret"))))
    assert srv.headers[0]["Authorization"] == "Bearer secret"


def test_aclose_releases_the_client():
    with FakeServer() as srv:
        engine = OpenAIEngine(config(srv.url))

        async def run():
            await engine.complete(MESSAGES, max_tokens=5, temperature=1.0)
            assert engine._client is not None
            await engine.aclose()
            assert engine._client is None

        asyncio.run(run())


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_connections": 0},
        {"read_timeout_s": 0},
        {"pool_timeout_s": -1.0},
        {"connect_timeout_s": 0},
        {"max_retries": -1},
        {"effort_transport": "header"},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ValueError):
        EngineConfig(**overrides)


def test_reasoning_of_earlier_turns_is_sent_back():
    engine = OpenAIEngine()
    plain_turns = (*MESSAGES, Message("assistant", "a1"), Message("user", "report"))
    body = engine.request_body(plain_turns, max_tokens=10, temperature=1.0)
    assert body["messages"][2] == {"role": "assistant", "content": "a1"}
    assert body["chat_template_kwargs"] == {"preserve_thinking": False}
    kept = (
        Message("system", "rules", reasoning="never sent for a system message"),
        Message("user", "task"),
        Message("assistant", "a1", reasoning="r1"),
        Message("user", "report"),
    )
    body = engine.request_body(kept, max_tokens=10, temperature=1.0)
    assert body["messages"][0] == {"role": "system", "content": "rules"}
    assert body["messages"][2] == {"role": "assistant", "content": "a1", "reasoning_content": "r1"}
    assert body["chat_template_kwargs"] == {"preserve_thinking": True}
    explicit = OpenAIEngine(EngineConfig(template_kwargs={"preserve_thinking": True}))
    body = explicit.request_body(plain_turns, max_tokens=10, temperature=1.0)
    assert body["chat_template_kwargs"] == {"preserve_thinking": True}


def test_reasoning_reaches_the_server():
    turns = (*MESSAGES, Message("assistant", "a1", reasoning="r1"), Message("user", "report"))
    with FakeServer() as srv:
        engine = OpenAIEngine(config(srv.url))

        async def run():
            async with engine:
                return await engine.complete(turns, max_tokens=5, temperature=1.0)

        asyncio.run(run())
    assert srv.bodies[0]["messages"][2]["reasoning_content"] == "r1"
    assert srv.bodies[0]["chat_template_kwargs"]["preserve_thinking"] is True


def test_engine_survives_a_new_event_loop():
    with FakeServer() as srv:
        engine = OpenAIEngine(config(srv.url))
        first = asyncio.run(engine.complete(MESSAGES, max_tokens=5, temperature=1.0))
        second = asyncio.run(engine.complete(MESSAGES, max_tokens=5, temperature=1.0))
        asyncio.run(engine.aclose())  # a pool of a finished loop is dropped, not awaited
    assert first.text == second.text == "ok"
    assert len(srv.bodies) == 2


def test_default_timeouts_leave_reads_to_the_deadline():
    engine = OpenAIEngine()
    assert (engine.config.read_timeout_s, engine.config.pool_timeout_s) == (None, None)

    async def timeouts():
        timeout = engine._http().timeout
        await engine.aclose()
        return timeout

    timeout = asyncio.run(timeouts())
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (10.0, None, 10.0, None)


@pytest.mark.parametrize(
    ("content", "reasoning", "text"),
    [
        (
            "let me think\n</think>\n\n```python\nx = 1\n```",
            "let me think",
            "```python\nx = 1\n```",
        ),
        ("<think>\nA</think>B</think>\n\nC", "A</think>B", "C"),
        ("no tags here", None, "no tags here"),
    ],
)
def test_think_tags_in_content_are_split(content, reasoning, text):
    out = parse_completion(reply(content), 0.0)
    assert (out.reasoning, out.text) == (reasoning, text)


def test_think_tags_are_kept_when_a_reasoning_field_arrives():
    out = parse_completion(reply("answer with a literal </think> in it", "why"), 0.0)
    assert (out.reasoning, out.text) == ("why", "answer with a literal </think> in it")


# --- the per-call thinking switch (P6.3-02, revision 5: the loop's forcing turn) ----------------


def test_a_call_with_thinking_off_keeps_the_earlier_reasoning_and_the_prefix():
    engine = OpenAIEngine(EngineConfig())
    messages = (*MESSAGES, Message("assistant", "", "cut reasoning"), Message("user", "now"))
    forced = engine.request_body(
        messages, max_tokens=10, temperature=1.0, reasoning_effort="low", thinking=False
    )
    assert forced["chat_template_kwargs"] == {"preserve_thinking": True, "enable_thinking": False}
    assert forced["messages"][2] == {
        "role": "assistant", "content": "", "reasoning_content": "cut reasoning",
    }  # fmt: skip
    # No effort sent; its note heads the system message as the template writes it with thinking
    # on, so the rendered system block is the thinking turns' own.
    assert "reasoning_effort" not in forced
    assert forced["messages"][0]["content"] == QWEN38_EFFORT_NOTES["low"] + "\n\nrules"
    assert forced["messages"][1:] == [{"role": "user", "content": "task"}, *forced["messages"][2:3],
                                      {"role": "user", "content": "now"}]  # fmt: skip
    medium = engine.request_body(
        messages, max_tokens=10, temperature=1.0, reasoning_effort="medium", thinking=False
    )
    assert medium["messages"][0]["content"] == "rules"  # the template adds none for medium
    normal = engine.request_body(
        messages, max_tokens=10, temperature=1.0, reasoning_effort="low"
    )
    assert normal["chat_template_kwargs"] == {"preserve_thinking": True}
    assert normal["reasoning_effort"] == "low"
    # The call's switch wins over the engine's.
    on = OpenAIEngine(EngineConfig(thinking=False)).request_body(
        MESSAGES, max_tokens=10, temperature=1.0, thinking=True
    )
    assert on["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.parametrize(
    "thinking, text, reasoning, want_text, want_reasoning",
    [
        # A parser that files output without </think> as reasoning: with thinking off, the text
        # is the answer.
        (False, None, "```python\nx\n```", "```python\nx\n```", None),
        (False, "answer", "stray", "answer", "stray"),
        # With thinking on, a cut reply's reasoning stays reasoning.
        (None, None, "still thinking", "", "still thinking"),
    ],
)  # fmt: skip
def test_a_reply_with_thinking_off_has_its_text_as_the_answer(
    thinking, text, reasoning, want_text, want_reasoning
):
    with FakeServer([("json", reply(text, reasoning, finish="length"), 200, 0)]) as srv:
        engine = OpenAIEngine(config(srv.url))
        out = asyncio.run(complete(engine, thinking=thinking))
    assert (out.text, out.reasoning) == (want_text, want_reasoning)
    kwargs = srv.bodies[0]["chat_template_kwargs"]
    assert kwargs.get("enable_thinking") == (False if thinking is False else None)


# --- the opening prompt's count (P6.3-02, revision 5) ----------------------------------------


def test_count_prompt_tokens_posts_the_rendered_request_to_tokenize():
    with FakeServer([("json", {"count": 1234, "tokens": []}, 200, 0)]) as srv:
        engine = OpenAIEngine(config(srv.url, tokenize_url=srv.url[:-3] + "/tokenize"))

        async def go():
            async with engine:
                return await engine.count_prompt_tokens(MESSAGES, reasoning_effort="low")

        assert asyncio.run(go()) == 1234
    body = srv.bodies[0]
    assert body["messages"] == [
        {"role": "system", "content": "rules"}, {"role": "user", "content": "task"},
    ]  # fmt: skip
    assert body["add_generation_prompt"] is True and body["add_special_tokens"] is False
    # The top-level effort goes into the template's variables, as vLLM forwards it.
    assert body["chat_template_kwargs"] == {"preserve_thinking": False, "reasoning_effort": "low"}


@pytest.mark.parametrize(
    "behavior, match",
    [
        (("json", {"tokens": []}, 200, 0), "malformed"),
        (("json", {"count": "12"}, 200, 0), "not a positive integer"),
        (("json", {"error": "no route"}, 404, 0), "HTTP 404"),
    ],
)  # fmt: skip
def test_count_prompt_tokens_raises_on_a_bad_reply(behavior, match):
    with FakeServer([behavior]) as srv:
        engine = OpenAIEngine(config(srv.url, tokenize_url=srv.url[:-3] + "/tokenize"))

        async def go():
            async with engine:
                return await engine.count_prompt_tokens(MESSAGES)

        with pytest.raises(EngineError, match=match):
            asyncio.run(go())


def test_count_prompt_tokens_needs_a_url():
    async def go():
        return await OpenAIEngine(EngineConfig()).count_prompt_tokens(MESSAGES)

    with pytest.raises(EngineError, match="no tokenize_url"):
        asyncio.run(go())
