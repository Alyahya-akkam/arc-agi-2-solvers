"""Restrict the optional pilot to its six-stream experiment and separate artifacts."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p582_builder", ROOT / "scripts/build_fn_notebooks.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def config(name):
    return BUILDER.resolve_extends(lambda path: (ROOT / path).read_bytes(), name)


def test_optional_six_stream_pilot_changes_only_the_declared_experiment():
    base = config("configs/P5.2-01.yaml")
    pilot = config("configs/P5.8-02.yaml")
    twin = config("configs/P5.8-02-cpu.yaml")
    allowed_flags = {
        "--mamba-radix-cache-strategy": "extra_buffer_lazy",
        "--mem-fraction-static": 0.95,
        "--max-running-requests": 7,
        "--max-mamba-cache-size": 21,
        "--cuda-graph-max-bs-decode": 6,
        "--cuda-graph-bs-decode": [1, 2, 3, 4, 5, 6],
    }
    expected_engine = {**base["engine"], "flags": {**base["engine"]["flags"], **allowed_flags},
                       "env": {**base["engine"]["env"],
                               "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1"}}
    assert pilot["engine"] == expected_engine
    assert pilot["solve"] == {**base["solve"], "max_streams": 6}
    for key in ("planner", "time_plan", "sampling", "stress", "arg_check", "patches", "mode",
                "template", "tool"):
        assert pilot[key] == base[key], key
    assert pilot["planner"]["min_thinking_tokens"] == 29000
    assert len(pilot["solve"]["task_ids"]) == 15
    for key in ("model_sources", "dataset_sources", "kernel_sources", "competition_sources",
                "accelerator", "enable_internet", "kernel_type", "code_file", "timeout_s"):
        assert pilot["kernel"][key] == base["kernel"][key], key
    assert pilot["mode"] == "auto"
    assert pilot["kernel"]["timeout_s"] == 7500
    assert not pilot["engine"]["flags"].get("--json-model-override-args")
    assert "SGLANG_VLM_CACHE_SIZE_MB" not in pilot["engine"]["env"]
    assert pilot["kernel"]["id"] != base["kernel"]["id"]
    assert pilot["kernel"]["code_dataset"] != base["kernel"]["code_dataset"]
    assert twin["kernel"]["id"] not in (pilot["kernel"]["id"], base["kernel"]["id"])
    assert twin["kernel"]["code_dataset"] == pilot["kernel"]["code_dataset"]
    assert twin["inputs"]["code"] == pilot["inputs"]["code"]
    assert twin["kernel"]["accelerator"] is None
    assert twin["mode"] == "development"
    assert twin["engine"]["script"] == "fn_fake_sglang.py"
    for flag, value in allowed_flags.items():
        assert twin["engine"]["flags"][flag] == value, flag
    assert twin["engine"]["env"] == pilot["engine"]["env"]
    assert twin["solve"]["max_streams"] == 6
    assert twin["planner"]["min_thinking_tokens"] == 29000
