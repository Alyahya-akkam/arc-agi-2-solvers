"""Checks that a stream's failure is one request's error, never the run's (P6.3-01, attempt 3).

Attempt 3 of ``P6.3-01`` ended with exit 1 when its 32k, 64-request cell reached its timeout:
the wave's cancel closed every ``HTTPConnection`` from the main thread while worker threads were
inside ``readline`` on chunked streams, and ``http.client`` raised ``AttributeError`` on its own
``fp``, which ``stream_chat`` did not catch. These tests cut waves of real chunked streams (as
vLLM sends them), break single reads, and fail the preserve check's calls, and require every
failure to be recorded. No GPU, no network beyond localhost.
"""

import concurrent.futures
import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike_streams")

EVENT = b'data: {"choices": [{"index": 0, "delta": {"content": "x"}, "token_ids": [7]}]}\n\n'


class Endless(BaseHTTPRequestHandler):
    """Streams chunked server-sent events until the client goes away, as vLLM does."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(EVENT), EVENT))
                self.wfile.flush()
                time.sleep(0.002)
        except OSError:
            pass


class Server(ThreadingHTTPServer):
    # The default backlog of 5 drops the connects of a large wave, which then wait out a SYN
    # retransmit (1 s) and hit their socket timeout before the cut.
    request_queue_size = 128


@pytest.fixture
def endless():
    server = Server(("127.0.0.1", 0), Endless)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}


def test_a_cut_wave_of_chunked_streams_records_every_request_as_cancelled(endless):
    # Attempt 3's crash: the cut of a wave whose requests are all mid-stream. Repeated, since
    # the old cross-thread close failed only when a read and the close interleaved.
    for _ in range(4):
        began = time.monotonic()
        records, cut = spike.run_wave(endless, [PAYLOAD] * 32, 0.5)
        assert cut and time.monotonic() - began < 10
        assert [r["error"] for r in records] == ["cancelled"] * 32
        assert all(r["chunks"] and r["content"].startswith("x") for r in records)


def test_cancel_shuts_sockets_down_and_leaves_the_close_to_the_reader():
    class Sock:
        def __init__(self):
            self.shut = []

        def shutdown(self, how):
            self.shut.append(how)

    class Connection:
        def __init__(self, sock):
            self.sock = sock

        def close(self):
            raise AssertionError("only the reading thread may close its connection")

    inflight = spike.Inflight()
    first = Connection(Sock())
    inflight.add(first)
    inflight.add(Connection(None))  # not connected: nothing to shut down
    inflight.cancel()
    assert first.sock.shut == [socket.SHUT_RDWR] and inflight.cancelled.is_set()
    late = Connection(Sock())
    inflight.add(late)  # registered after the cancel: shut down at once
    assert late.sock.shut == [socket.SHUT_RDWR]


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("'NoneType' object has no attribute 'close'"),  # attempt 3's error
        RuntimeError("anything else"),
        ValueError("invalid literal for int() with base 16: b''"),
    ],
)
def test_any_error_while_reading_is_the_requests_error(endless, monkeypatch, error):
    calls = {"n": 0}
    real = spike.http.client.HTTPResponse.readline

    def broken(self, *args):
        calls["n"] += 1
        if calls["n"] > 3:
            raise error
        return real(self, *args)

    monkeypatch.setattr(spike.http.client.HTTPResponse, "readline", broken)
    record = spike.stream_chat(endless, PAYLOAD, 5.0)
    assert record["error"] == f"{type(error).__name__}: {error}"
    assert record["content"].startswith("x") and record["t_first"] is not None


def test_a_refused_connection_is_an_error_not_an_exception():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    record = spike.stream_chat(f"http://127.0.0.1:{port}", PAYLOAD, 2.0)
    assert record["error"].startswith("ConnectionRefusedError")
    assert (record["content"], record["chunks"]) == ("", [])


def test_a_wave_records_a_raising_request_instead_of_raising(monkeypatch):
    def explode(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(spike, "stream_chat", explode)
    records, cut = spike.run_wave("http://127.0.0.1:9", [PAYLOAD] * 3, 5.0)
    assert not cut and [r["error"] for r in records] == ["RuntimeError: boom"] * 3
    rows = [spike.request_summary(r, {"max_run": 64, "special_min_id": 9, "eos_ids": []})
            for r in records]  # fmt: skip
    assert all(row["error"] == "RuntimeError: boom" for row in rows)


def test_a_probe_task_that_raises_is_recorded_with_one_error():
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        future = pool.submit(lambda: 1 / 0)
        concurrent.futures.wait([future])
    record = spike.task_outcome(future, "aaaaaaaa")
    assert record["task_id"] == "aaaaaaaa" and record["errors"] == 1 and record["rounds"] == []
    assert record["error"].startswith("ZeroDivisionError") and not record["verified"]


# --- the preserve check: a failure is recorded and the run goes on -------------------------------


class FailingEngine:
    async def complete(self, *args, **kwargs):
        raise ConnectionResetError("the stream broke")

    async def aclose(self):
        pass


class ReasoningEngine:
    async def complete(self, *args, **kwargs):
        return SimpleNamespace(
            text="ok", reasoning="Let me look at every example. " * 20, completion_tokens=120,
            finish_reason="stop", prompt_tokens=50,
        )  # fmt: skip

    async def aclose(self):
        pass


def preserve_run(tmp_path, challenges, base_url="http://127.0.0.1:9"):
    settings = {"n_tasks": 2, "turn1_max_tokens": 64, "reasoning_effort": "low",
                "tolerance_tokens": 16, "budget_s": 30}  # fmt: skip
    return SimpleNamespace(
        config={"preserve": settings, "seed": 0, "server": {"served_model_name": "m"}},
        args=SimpleNamespace(smoke=True, smoke_challenges=str(challenges)),
        deadline=time.monotonic() + 60, out=tmp_path, metrics={}, base_url=base_url,
    )  # fmt: skip


def test_preserve_records_an_engine_failure_per_task_and_returns(monkeypatch, tmp_path):
    challenges, _, _ = spike.write_smoke_inputs(tmp_path / "inputs")
    monkeypatch.setattr(spike, "make_engine", lambda *args, **kwargs: FailingEngine())
    run = preserve_run(tmp_path, challenges)
    spike.preserve_phase(run)
    check = json.loads((tmp_path / "preserve_check.json").read_text())
    assert check["verdict"] == "inconclusive" and len(check["tasks"]) == 2
    for row in check["tasks"]:
        assert row["error"] == "ConnectionResetError: the stream broke"
        assert row["verdict"] == "inconclusive: ConnectionResetError: the stream broke"
    assert run.metrics["preserve"] == check


def test_preserve_records_a_failed_tokenize_call(monkeypatch, tmp_path):
    challenges, _, _ = spike.write_smoke_inputs(tmp_path / "inputs")
    monkeypatch.setattr(spike, "make_engine", lambda *args, **kwargs: ReasoningEngine())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    spike.preserve_phase(preserve_run(tmp_path, challenges, f"http://127.0.0.1:{port}"))
    check = json.loads((tmp_path / "preserve_check.json").read_text())
    assert check["verdict"] == "inconclusive"
    assert all(row["error"].startswith("Stop: POST /tokenize failed") for row in check["tasks"])


def test_preserve_records_a_failure_outside_the_tasks(tmp_path):
    spike.preserve_phase(preserve_run(tmp_path, tmp_path / "absent.json"))
    check = json.loads((tmp_path / "preserve_check.json").read_text())
    assert check["verdict"] == "inconclusive" and check["tasks"] == []
    assert check["error"]
