"""Check native structured sampling and the startup gates without downloading model files."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fn_schema_probe", ROOT / "scripts/fn_probe.py")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_schema_goes_in_sampling_while_reasoning_is_top_level(monkeypatch):
    bodies = []

    class Generation:
        def __init__(self, base, body, timeout, incremental):
            bodies.append(body)

        def cancel(self):
            pass

        def run(self):
            return probe.GenResult(text="{}", ids=[1], finish="stop")

    monkeypatch.setattr(probe, "Generation", Generation)
    config = {"solve": {"context_tokens": 4096, "context_margin_tokens": 128,
                        "request_timeout_s": 2}, "sampling": {"temperature": 0},
              "engine": {"incremental_stream": False}, "planner": {"replan_min_tokens": 256}}
    inner = SimpleNamespace(run=SimpleNamespace(config=config), base="http://fake")
    generate = probe.make_generate(inner, "schema-test", [])
    generate([1], 10, probe._now() + 2, "force", {"json_schema": "{}", "stop": []})
    assert bodies[0]["require_reasoning"] is False
    assert bodies[0]["sampling_params"]["json_schema"] == "{}"
    assert "require_reasoning" not in bodies[0]["sampling_params"]
    generate([1], 10, probe._now() + 2, "force", {})
    assert "require_reasoning" not in bodies[1]


@pytest.mark.parametrize("ok", [True, False])
def test_real_library_preflight_is_a_fail_closed_child(monkeypatch, tmp_path, ok):
    run = SimpleNamespace(config={"schema_check": {"enabled": True, "timeout_s": 600}},
                          state={"paths": {"model": "/local/model"}}, out=tmp_path)
    monkeypatch.setattr(probe, "boot_environment", lambda run: {})

    def child(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "arc26.agent.grammar_check"]
        assert kwargs["timeout"] == 600
        assert command[command.index("--model") + 1] == "/local/model"
        Path(command[-1]).write_text(json.dumps({"ok": ok, "xgrammar_version": "0.2.1",
                                               "schemas": {"1": []}, "cuda_initialized": False,
                                               "error": "bad grammar"}))
        return SimpleNamespace(returncode=0 if ok else 1, stdout="", stderr="")

    monkeypatch.setattr(probe.subprocess, "run", child)
    if ok:
        assert probe.stage_schema_check(SimpleNamespace(run=run))["ok"] is True
    else:
        with pytest.raises(RuntimeError, match="schema preflight failed"):
            probe.stage_schema_check(SimpleNamespace(run=run))


@pytest.mark.parametrize("wrong", [False, True])
def test_live_probe_requires_exact_valid_json_for_both_index_counts(monkeypatch, tmp_path, wrong):
    import arc26.agent.toolshot as module

    monkeypatch.setattr(probe, "tool_module", lambda config: module)
    calls = []

    def generate(ids, max_new, deadline, kind, extra):
        count = len(json.loads(extra["json_schema"])["properties"]["answers"]["required"])
        assert extra["ignore_eos"] is False and extra["stop"] == []
        assert kind == "schema_probe" and max_new == 512
        calls.append(count)
        text = json.dumps({"answers": {str(i): [["34" if wrong else "12"]]
                                      for i in range(count)}})
        return probe.GenResult(text=text, finish="stop", completion_tokens=len(text))

    monkeypatch.setattr(probe, "make_generate", lambda *args: generate)
    inner = SimpleNamespace(tok=probe.FakeTokenizer(), run=SimpleNamespace(
        config={"tool": {"loop": {}}, "engine": {"script": "fn_fake_sglang.py"}},
        out=tmp_path, left=lambda: 30))
    if wrong:
        with pytest.raises(RuntimeError, match="native schema probe failed"):
            probe.stage_schema_probe(inner)
    else:
        assert probe.stage_schema_probe(inner)["ok"] is True
        assert calls == [1, 2]
