"""Tests of the engine server (P6.4): start, health, crash, restart, hang, stale engines, stop.

Every test runs the fake engine of :mod:`arc26.agent.server` as a real subprocess on a free local
port, so the same start, health, and stop code paths run as for vLLM.
"""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from arc26.agent.engine import EngineConfig, OpenAIEngine
from arc26.agent.loop import AttemptContext, attempt_context
from arc26.agent.server import (
    FAKE_CRASH_CODE,
    FAKE_FALLBACK,
    EngineBootTimeout,
    EngineServer,
    EngineStartError,
    EngineSupervisor,
    ServerConfig,
    clean_env,
    engine_env,
    fake_command,
    fake_completion,
    free_port,
    http_get,
    vllm_command,
)
from arc26.agent.types import Message

REPLY = "The rows are mirrored.\n\n```python\ndef transform(grid):\n    return grid\n```\n"


def make_server(tmp_path, *, replies=None, boot_timeout_s=20.0, **options):
    port = free_port()
    path = None
    if replies is not None:
        path = tmp_path / "replies.json"
        path.write_text(json.dumps(replies), encoding="utf-8")
    return EngineServer(
        fake_command(port, path, **options),
        engine_env(os.environ, ServerConfig(port=port)),
        tmp_path / "engine.log",
        f"http://127.0.0.1:{port}",
        boot_timeout_s=boot_timeout_s,
        stop_timeout_s=5.0,
        health_interval_s=0.05,
        pid_path=tmp_path / "engine.pid",
    )


@pytest.fixture
def servers():
    started = []
    yield started
    for server in started:
        server.stop()


def chat(root, task_id="t1", timeout=5.0):
    """POST one chat completion; return (status, body) or (0, error text) on a dropped call."""
    body = {
        "model": "fake",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": f"Task {task_id}: 1 training example."}],
    }
    request = urllib.request.Request(
        f"{root}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except (urllib.error.URLError, ConnectionError, OSError) as err:
        return 0, str(err)


def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def gone(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


# ---------------------------------------------------------------- environment and command line


def test_clean_env_drops_python_and_pip_settings():
    environ = {
        "PYTHONPATH": "/kaggle/lib",
        "PYTHONHOME": "/x",
        "VIRTUAL_ENV": "/venv",
        "PIP_INDEX_URL": "https://example.invalid",
        "PIP_NO_INDEX": "1",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "HOME": "/root",
        "PATH": "/usr/bin",
    }
    env = clean_env(environ, ["TRITON_PTXAS_PATH"])
    assert env == {"HOME": "/root", "PATH": "/usr/bin", "PYTHONNOUSERSITE": "1"}
    assert "PYTHONPATH" in environ  # the input is not changed


def test_engine_env_adds_the_config_and_puts_the_venv_first():
    config = ServerConfig(env={"HF_HUB_OFFLINE": "1"}, unset_env=("TRITON_PTXAS_PATH",))
    environ = {"PATH": "/usr/bin", "PYTHONPATH": "/x", "TRITON_PTXAS_PATH": "/p"}
    env = engine_env(environ, config, "/venv/bin")
    assert env["PATH"] == "/venv/bin" + os.pathsep + "/usr/bin"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "PYTHONPATH" not in env and "TRITON_PTXAS_PATH" not in env


def test_vllm_command_carries_the_config():
    config = ServerConfig(port=8123, extra_args=("--enforce-eager",))
    command = vllm_command("/venv/bin/vllm", "/models/q", config)
    assert command[:3] == ["/venv/bin/vllm", "serve", "/models/q"]
    pairs = dict(zip(command[3::2], command[4::2], strict=False))
    assert pairs["--port"] == "8123" and pairs["--tensor-parallel-size"] == "4"
    assert pairs["--kv-cache-dtype"] == "fp8" and pairs["--max-model-len"] == "65536"
    assert pairs["--attention-backend"] == "TRITON_ATTN" and pairs["--reasoning-parser"] == "qwen3"
    assert "--enable-prefix-caching" in command and "--language-model-only" in command
    assert command[-1] == "--enforce-eager"
    plain = ServerConfig(enable_prefix_caching=False, language_model_only=False)
    off = vllm_command("vllm", "m", plain)
    assert "--no-enable-prefix-caching" in off and "--language-model-only" not in off


@pytest.mark.parametrize(
    "kwargs",
    [{"port": 0}, {"max_model_len": 0}, {"boot_timeout_s": 0}, {"restarts": -1},
     {"gpu_memory_utilization": 1.5}],
)  # fmt: skip
def test_server_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        ServerConfig(**kwargs)


def test_server_config_urls():
    config = ServerConfig(host="127.0.0.1", port=8001)
    assert config.root_url == "http://127.0.0.1:8001"
    assert config.api_url == "http://127.0.0.1:8001/v1"


# ---------------------------------------------------------------- the fake engine's replies


def test_fake_completion_follows_the_script_by_task_and_turn():
    replies = {"t1": ["first", {"text": "second", "reasoning": "think " * 10}]}
    first = {"max_tokens": 100, "messages": [{"role": "user", "content": "Task t1: 2 examples."}]}
    response, delay = fake_completion(replies, first)
    assert response["choices"][0]["message"]["content"] == "first" and delay == 0.0
    assert response["choices"][0]["finish_reason"] == "stop"
    second = dict(first)
    second["messages"] = first["messages"] + [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "feedback"},
    ]
    response, _ = fake_completion(replies, second)
    message = response["choices"][0]["message"]
    assert message["content"] == "second" and message["reasoning"].startswith("think")
    assert response["usage"]["completion_tokens"] == 15 + 2
    third = dict(second)
    third["messages"] = second["messages"] + [{"role": "assistant", "content": "x"}]
    assert fake_completion(replies, third)[0]["choices"][0]["message"]["content"] == FAKE_FALLBACK
    unknown = {"max_tokens": 100, "messages": [{"role": "user", "content": "Task zz: 1 example."}]}
    assert fake_completion(replies, unknown)[0]["choices"][0]["message"]["content"] == FAKE_FALLBACK


def test_fake_completion_cuts_a_reply_at_max_tokens():
    replies = {"t1": [{"text": "x" * 40, "reasoning": "r" * 40}]}
    body = {"max_tokens": 5, "messages": [{"role": "user", "content": "Task t1: 1 example."}]}
    response, _ = fake_completion(replies, body)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 5
    assert choice["message"]["reasoning"] == "r" * 20 and choice["message"]["content"] == ""


# ---------------------------------------------------------------- start, health, stop


def test_start_health_chat_and_stop(tmp_path, servers):
    server = make_server(tmp_path, replies={"t1": [REPLY]})
    servers.append(server)
    seconds = server.start()
    assert 0 < seconds < 20 and server.alive() and server.healthy()
    status, body = http_get(server.root_url, "/v1/models")
    assert status == 200 and json.loads(body)["data"][0]["id"] == "fake"
    pid = server.process.pid
    assert json.loads((tmp_path / "engine.pid").read_text())["pid"] == pid

    async def one_call():
        engine = OpenAIEngine(EngineConfig(base_url=server.root_url + "/v1", model="fake"))
        try:
            with attempt_context(AttemptContext("t1", "t1/p1/0", 1)):
                return await engine.complete(
                    [Message("system", "rules"), Message("user", "Task t1: 1 training example.")],
                    max_tokens=1000,
                    temperature=1.0,
                )
        finally:
            await engine.aclose()

    completion = asyncio.run(one_call())
    assert completion.text == REPLY and completion.finish_reason == "stop"
    assert server.stop() is not None
    assert not server.alive() and gone(pid)
    assert not (tmp_path / "engine.pid").exists()
    assert "fake engine on 127.0.0.1" in (tmp_path / "engine.log").read_text()
    with pytest.raises(EngineStartError, match="stopped for good"):
        server.start()


def test_a_crash_during_the_boot_is_reported_and_cleaned_up(tmp_path, servers):
    server = make_server(tmp_path, fail_boot=True)
    servers.append(server)
    began = time.monotonic()
    with pytest.raises(EngineStartError, match="exited with code 1") as caught:
        server.start()
    assert not isinstance(caught.value, EngineBootTimeout)
    assert time.monotonic() - began < 10
    assert not server.alive() and not (tmp_path / "engine.pid").exists()
    assert "failing its boot" in str(caught.value)  # the log tail is quoted


def test_a_boot_timeout_stops_the_process(tmp_path, servers):
    server = make_server(tmp_path, boot_timeout_s=0.5, boot_delay_s=60)
    servers.append(server)
    with pytest.raises(EngineBootTimeout):
        server.start()
    assert not server.alive()
    assert gone(server.process.pid)


def test_a_command_that_cannot_run(tmp_path):
    server = EngineServer(
        [str(tmp_path / "no-such-vllm"), "serve"], {}, tmp_path / "engine.log", "http://127.0.0.1:9"
    )
    with pytest.raises(EngineStartError, match="cannot run"):
        server.start()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_stop_kills_what_the_engine_leaves_in_its_group(tmp_path, servers):
    # The engine starts a worker that ignores SIGTERM, as a stuck vLLM worker would: the leader
    # exits on SIGTERM, and the worker must still be killed with the group.
    port = free_port()
    worker_pid = tmp_path / "worker.pid"
    script = (
        "import pathlib, subprocess, sys\n"
        "code = 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'\n"
        "worker = subprocess.Popen([sys.executable, '-c', code])\n"
        f"pathlib.Path({str(worker_pid)!r}).write_text(str(worker.pid))\n"
        "from arc26.agent.server import main\n"
        f"sys.exit(main(['fake', '--port', '{port}']))\n"
    )
    server = EngineServer(
        [sys.executable, "-c", script],
        engine_env(os.environ, ServerConfig(port=port)),
        tmp_path / "engine.log",
        f"http://127.0.0.1:{port}",
        boot_timeout_s=20,
        stop_timeout_s=5,
        health_interval_s=0.05,
    )
    servers.append(server)
    server.start()
    pid = int(worker_pid.read_text())
    assert alive(pid)
    server.stop()
    assert wait_for(lambda: not alive(pid), 10)


# ---------------------------------------------------------------- stale engines


def test_a_stale_engine_of_a_killed_run_is_stopped_first(tmp_path, servers):
    server = make_server(tmp_path)
    stale = subprocess.Popen(
        server.command, env=server.env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        assert wait_for(server.healthy, 10)
        (tmp_path / "engine.pid").write_text(
            json.dumps({"pid": stale.pid, "argv": server.command}), encoding="utf-8"
        )
        servers.append(server)
        server.start()  # the same port: only possible once the stale engine is gone
        assert stale.wait(10) is not None
        assert server.process.pid != stale.pid and server.healthy()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(stale.pid, signal.SIGKILL)
        stale.wait()


def test_a_pid_file_naming_another_program_is_ignored(tmp_path):
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        server = make_server(tmp_path)
        (tmp_path / "engine.pid").write_text(
            json.dumps({"pid": other.pid, "argv": server.command}), encoding="utf-8"
        )
        assert server.stop_stale() is False
        assert other.poll() is None
        assert not (tmp_path / "engine.pid").exists()
        (tmp_path / "engine.pid").write_text("not json", encoding="utf-8")
        assert server.stop_stale() is False and not (tmp_path / "engine.pid").exists()
    finally:
        other.kill()
        other.wait()


# ---------------------------------------------------------------- crash, restart, hang


def supervise(supervisor, action):
    """Run ``action(supervisor)`` (a coroutine function) with the supervisor watching."""

    async def main():
        supervisor.start()
        try:
            return await action(supervisor)
        finally:
            await supervisor.close()

    return asyncio.run(main())


async def until(predicate, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


def test_a_crash_is_detected_and_the_engine_restarted_once(tmp_path, servers):
    marker = tmp_path / "crashed"
    server = make_server(tmp_path, replies={"t1": [REPLY]}, crash_after=1, crash_marker=marker)
    servers.append(server)
    server.start()
    first_pid = server.process.pid
    supervisor = EngineSupervisor(server.root_url, server, interval_s=0.05, hang_s=10, restarts=1)

    async def action(sup):
        status, _ = await asyncio.to_thread(chat, server.root_url)
        assert status == 0  # the engine died with the call in flight
        assert await until(lambda: any(e["event"] == "restarted" for e in sup.events))
        assert await sup.wait_ready()
        return await asyncio.to_thread(chat, server.root_url)

    status, body = supervise(supervisor, action)
    assert status == 200 and body["choices"][0]["message"]["content"] == REPLY
    assert marker.exists() and server.process.pid != first_pid and server.boots == 2
    assert [e["event"] for e in supervisor.events] == ["down", "restarted"]
    assert f"code {FAKE_CRASH_CODE}" in supervisor.events[0]["reason"]
    assert supervisor.restarts_left == 0 and not supervisor.failed


def test_a_second_crash_leaves_the_engine_gone_for_good(tmp_path, servers):
    server = make_server(tmp_path, crash_after=1)  # no marker: every start crashes again
    servers.append(server)
    server.start()
    supervisor = EngineSupervisor(server.root_url, server, interval_s=0.05, hang_s=10, restarts=1)

    async def action(sup):
        await asyncio.to_thread(chat, server.root_url)
        assert await until(lambda: any(e["event"] == "restarted" for e in sup.events))
        await asyncio.to_thread(chat, server.root_url)
        assert await until(lambda: sup.failed)  # wait_ready says ready until the check runs
        return await sup.wait_ready()

    assert supervise(supervisor, action) is False
    assert supervisor.failed and "no restart left" in supervisor.reason


def test_a_hang_is_detected_and_the_engine_restarted(tmp_path, servers):
    server = make_server(tmp_path)
    servers.append(server)
    server.start()
    first_pid = server.process.pid
    answers = iter([False] * 10)

    def probe():
        return next(answers, True)  # unhealthy for a while, then healthy after the restart

    supervisor = EngineSupervisor(
        server.root_url, server, interval_s=0.02, hang_s=0.1, restarts=1, probe=probe
    )

    async def action(sup):
        assert await until(lambda: any(e["event"] == "restarted" for e in sup.events))
        return await sup.wait_ready()

    assert supervise(supervisor, action) is True
    assert "no healthy answer" in supervisor.events[0]["reason"]
    assert server.process.pid != first_pid and gone(first_pid)


def test_no_restart_after_the_deadline(tmp_path, servers):
    server = make_server(tmp_path, crash_after=1)
    servers.append(server)
    server.start()
    supervisor = EngineSupervisor(
        server.root_url, server, interval_s=0.05, restarts=1, deadline=time.monotonic() - 1
    )

    async def action(sup):
        await asyncio.to_thread(chat, server.root_url)
        assert await until(lambda: sup.failed)
        return await sup.wait_ready()

    assert supervise(supervisor, action) is False
    assert "deadline" in supervisor.reason and server.boots == 1


def test_a_server_given_by_url_is_not_restarted():
    root = f"http://127.0.0.1:{free_port()}"  # nothing listens here
    supervisor = EngineSupervisor(root, None, interval_s=0.02, hang_s=0.2, restarts=1)

    async def action(sup):
        assert sup.ready  # ready until the checks say otherwise
        assert await until(lambda: sup.failed, 10)
        return await sup.wait_ready()

    assert supervise(supervisor, action) is False
    assert "not restarted" in supervisor.reason
    assert [e["event"] for e in supervisor.events] == ["down", "failed"]


def test_wait_ready_blocks_while_down_and_answers_false_when_gone():
    supervisor = EngineSupervisor("http://127.0.0.1:9", None, interval_s=0.02, hang_s=0.05,
                                  probe=lambda: False)  # fmt: skip

    async def main():
        supervisor.start()
        try:
            return await asyncio.wait_for(supervisor.wait_ready(), 0.01), await until(
                lambda: supervisor.failed, 5
            )
        finally:
            await supervisor.close()

    ready_at_start, failed_later = asyncio.run(main())
    assert ready_at_start is True and failed_later is True
    assert asyncio.run(supervisor.wait_ready()) is False
    assert "not restarted" in supervisor.reason


def test_a_broken_check_does_not_end_the_watch(tmp_path, servers):
    server = make_server(tmp_path)
    servers.append(server)
    server.start()
    calls = []

    def probe():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("a probe that breaks once")
        return True

    supervisor = EngineSupervisor(server.root_url, server, interval_s=0.02, probe=probe)

    async def action(sup):
        assert await until(lambda: len(calls) >= 3)
        return await sup.wait_ready()

    assert supervise(supervisor, action) is True and not supervisor.events
