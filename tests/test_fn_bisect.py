"""The engine bisect (scripts/fn_bisect.py, run P5.2-06, task P5.2m).

Covers the first-error reading on P5.2-05's own tracebacks (the cascade, the torchcodec noise,
the launching frame), the variant checks and environment, the hash-guarded edits applied for one
variant and restored after it, and the variant loop on the fake SGLang server: the time rule, the
run_if rule, a survivor's pool and rate request, a crash at boot, a hang cut at its deadline. No
network, no GPU.
"""
# The engine log excerpts below are P5.2-05's lines verbatim, so they keep their length.
# ruff: noqa: E501

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fb = _load("fn_bisect_under_test", SCRIPTS / "fn_bisect.py")
fp = fb.fp

SPEC_KEYS = dict(
    error_patterns=["Scheduler hit an exception", "CUDA error", "illegal memory access",
                    "Triton Error", ": error: argument"],
    ignore=["torchcodec", "libavutil", "Ignore import error"],
    skip_frames=["/torch/nn/modules/", "/torch/_ops.py", "/triton/runtime/", "/triton/compiler/"],
)
SITE = "/tmp/arc26-probe/root/tmp/fn/venv/lib/python3.12/site-packages"

# P5.2-05's engine.log, boot 2 (BF16 KV, no graphs): the torchcodec noise first, then rank 1's
# fault at the first prefill, reported while a Triton kernel loaded (lines 2459 to 2574, cut).
BOOT2 = f"""[2026-10-30 09:33:01 TP0 EP0] Ignore import error when loading mimo_audio: Could not load libtorchcodec.
Traceback (most recent call last):
  File "{SITE}/torchcodec/_core/ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
OSError: libavutil.so.60: cannot open shared object file: No such file or directory
[2026-10-30 09:34:08] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2026-10-30 09:34:25 TP1 EP1] Scheduler hit an exception: Traceback (most recent call last):
  File "{SITE}/sglang/srt/managers/scheduler.py", line 5259, in run_scheduler_process
    scheduler.run_event_loop()
  File "{SITE}/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "{SITE}/sglang/srt/models/qwen2_moe.py", line 687, in forward
    fused_gate_sigmoid_mul_add(
  File "{SITE}/sglang/kernels/ops/elementwise/elementwise.py", line 540, in fused_gate_sigmoid_mul_add
    _fused_gate_sigmoid_mul_add_kernel[(num_tokens,)](
  File "{SITE}/triton/runtime/jit.py", line 370, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "{SITE}/triton/compiler/compiler.py", line 468, in _init_handles
    self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered

[2026-10-30 09:34:25] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
[2026-10-30 09:34:25 TP0 EP0] Scheduler hit an exception: Traceback (most recent call last):
  File "{SITE}/sglang/srt/managers/scheduler.py", line 5259, in run_scheduler_process
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
"""

# P5.2-05's boot 1: the fault met in the capture at our torch.sum, then the chained exception.
BOOT1 = f"""[2026-10-30 09:32:28 TP0 EP0] Scheduler hit an exception: Traceback (most recent call last):
  File "{SITE}/sglang/srt/layers/moe/moe_runner/marlin.py", line 174, in fused_experts_none_to_marlin
    output = fused_marlin_moe(
             ^^^^^^^^^^^^^^^^^
  File "{SITE}/torch/_ops.py", line 1279, in __call__
    return self._op(*args, **kwargs)
  File "{SITE}/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py", line 423, in fused_marlin_moe
    torch.sum(intermediate_cache3, dim=1, out=output)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
Exception: Capture cuda graph failed: CUDA error: an illegal memory access was encountered
"""


def read_error(text):
    return fb.first_error(text, SPEC_KEYS["error_patterns"], SPEC_KEYS["ignore"],
                          SPEC_KEYS["skip_frames"])


# -- the log ---------------------------------------------------------------------------------


def test_the_first_error_skips_the_import_noise_and_names_the_launching_triton_kernel():
    err = read_error(BOOT2)
    assert err["rank"] == "TP1 EP1" and err["line"] == 7 and err["fatal_lines"] == 4
    assert err["message"] == "RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered"
    assert err["kernel_frame"] == ("sglang/kernels/ops/elementwise/elementwise.py:540 "
                                   "fused_gate_sigmoid_mul_add")
    assert err["triton_kernel"] == "_fused_gate_sigmoid_mul_add_kernel"
    assert "torchcodec" not in err["traceback"] and err["traceback"].startswith("[2026-10-30 09:34:25")
    assert err["frames"][-1] == "triton/compiler/compiler.py:468 _init_handles"


def test_the_first_error_keeps_the_first_exception_of_a_chain_with_its_message_lines():
    err = read_error(BOOT1)
    assert err["message"].startswith("torch.AcceleratorError: CUDA error: an illegal memory")
    assert err["message"].endswith("For debugging consider passing CUDA_LAUNCH_BLOCKING=1")
    assert err["kernel_frame"].startswith("sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py:423")
    assert err["kernel_call"] == "torch.sum(intermediate_cache3, dim=1, out=output)"
    assert err["triton_kernel"] is None and "Capture cuda graph failed" not in err["traceback"]


def test_a_line_without_a_traceback_is_its_own_error_and_a_clean_log_has_none():
    text = "booting\nfn_fake_sglang.py: error: argument --fake-pool: invalid int value: 'x'\n"
    err = read_error(text)
    assert err["line"] == 2 and err["message"].startswith("fn_fake_sglang.py: error: argument")
    assert err["kernel_frame"] is None and err["frames"] == []
    assert read_error("[2026-10-30 09:34:08] INFO: Uvicorn running\n") is None


def test_log_marks_and_the_decode_rate_read_the_servers_lines():
    text = ("Load weight end. elapsed=76.85 s, type=Qwen4Exp\nKV Cache is allocated. #tokens: 869184\n"
            "Decode batch. #running-req: 1, #token: 40, gen throughput (token/s): 31.50\n"
            "Decode batch. #running-req: 1, #token: 80, gen throughput (token/s): 33.00\n"
            "Decode batch. #running-req: 1, #token: 120, gen throughput (token/s): 32.25\n")
    marks = fb.log_marks(text, {"load_s": r"Load weight end\. elapsed=([\d.]+) s",
                                "kv_tokens": r"KV Cache is allocated\. #tokens: (\d+)",
                                "uvicorn": "Uvicorn running", "decode": "Decode batch"})
    assert marks == {"load_s": 76.85, "kv_tokens": 869184.0, "uvicorn": None, "decode": True}
    assert fb.log_decode_rate(text) == {"lines": 3, "median_tok_s": 32.25}
    assert fb.log_decode_rate("") == {"lines": 0, "median_tok_s": None}
    assert fb.bounded(b"x" * 100, 40) .count(b"[... cut ...]") == 1 and fb.bounded(b"ab", 40) == b"ab"


# -- variants, environment, edits ------------------------------------------------------------


def test_the_variant_checks_refuse_unsafe_names_unknown_keys_and_unguarded_edits():
    good = {"variants": [{"name": "v1_a", "flags": {}}, {"name": "v2_b", "run_if": "none_survived"}]}
    assert [v["name"] for v in fb.check_variants(good)] == ["v1_a", "v2_b"]
    for bad in ([{"name": "../x"}], [{"name": "a"}, {"name": "a"}], [{"name": "a", "flag": {}}],
                [{"name": "a", "run_if": "sometimes"}],
                [{"name": "a", "edits": [{"name": "e", "file": "f", "pattern": "p",
                                          "replacement": "r"}]}], []):
        with pytest.raises(ValueError):
            fb.check_variants({"variants": bad})


def test_the_variant_environment_sets_removes_fills_and_tags():
    env = fb.variant_env({"NCCL_DEBUG": "INFO", "KEEP": "1"},
                         {"NCCL_DEBUG": None, "CUDA_LAUNCH_BLOCKING": 1,
                          "SGLANG_KERNEL_API_LOGDEST": "{scratch}/kapi_%i.log"}, "v-1",
                         {"scratch": "/s/v1"})
    assert env == {"KEEP": "1", "CUDA_LAUNCH_BLOCKING": "1",
                   "SGLANG_KERNEL_API_LOGDEST": "/s/v1/kapi_%i.log", fb.VARIANT_ENV: "v-1"}


def _edit(site, text):
    target = site / "pkg" / "marlin.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    before = hashlib.sha256(text.encode()).hexdigest()
    return target, {"name": "e1", "file": "pkg/marlin.py", "sha256_before": before,
                    "pattern": r"expert_map=quant_info\.expert_map,",
                    "replacement": "expert_map=EMPTY,", "count": 1}


def test_an_edit_applies_for_its_variant_and_the_original_bytes_come_back(tmp_path):
    original = "a\n        expert_map=quant_info.expert_map,\nb\n"
    target, item = _edit(tmp_path, original)
    restore = []
    results, ready = fb.apply_edits(tmp_path, [item], restore)
    assert ready and results[0]["status"] == "applied" and "EMPTY" in target.read_text()
    assert "context" not in results[0]
    assert fb.restore_edits(restore) == ["marlin.py"] and target.read_text() == original
    assert restore == []


def test_a_guarded_edit_on_another_file_is_refused_and_nothing_is_written(tmp_path):
    target, item = _edit(tmp_path, "expert_map=quant_info.expert_map,\n")
    item["sha256_before"] = "0" * 64
    restore = []
    results, ready = fb.apply_edits(tmp_path, [item], restore)
    assert not ready and results[0]["status"] == "sha256 differs" and restore == []
    assert fb.apply_edits(None, [item], [])[0][0]["status"] == "no venv"


def test_the_runs_ep_mask_edits_chain_their_hashes_and_match_the_forks_file_when_present(tmp_path):
    import yaml

    config = yaml.safe_load((REPO / "configs" / "P5.2-06.yaml").read_text())
    variants = {v["name"]: v for v in config["bisect"]["variants"]}
    first, second = variants["v1_ep_mask_eager"]["edits"]
    assert first["sha256_after"] == second["sha256_before"] and first["file"] == second["file"]
    assert variants["v2_ep_mask_fp8_graphs"]["edits"] == variants["v1_ep_mask_eager"]["edits"]
    # The fork's file at d00d88e, where a local clone has it (read only, never fetched).
    fork = Path("/tmp/fnsg/penny/python") / first["file"]
    if not fork.is_file():
        pytest.skip("no local copy of the fork at d00d88e")
    site = tmp_path / "site"
    (site / first["file"]).parent.mkdir(parents=True)
    (site / first["file"]).write_bytes(fork.read_bytes())
    restore = []
    results, ready = fb.apply_edits(site, [first, second], restore)
    assert ready and [r["sha256_after"] for r in results] == [first["sha256_after"],
                                                              second["sha256_after"]]
    fb.restore_edits(restore)
    assert (site / first["file"]).read_bytes() == fork.read_bytes()


# -- the loop on the fake server -------------------------------------------------------------


def make_inner(tmp_path, variants, end_s=60.0, **spec):
    config = fp.apply_smoke(fp.load_config(REPO / "configs" / "P5.2-01.yaml"))
    port = 30990
    config["engine"]["port"] = port
    config["engine"]["flags"]["--port"] = "{port}"
    config["paths"]["scratch"] = str(tmp_path / "scratch")
    config["time_plan"]["development"]["end_s"] = end_s
    config["bisect"] = {
        "enabled": True, "budget_s": 6, "min_variant_s": 3, "kill_reserve_s": 1,
        "stop_settle_s": 5, "poll_s": 0.1, "settle_s": 0.1, "base_flags": {}, "base_env": {},
        "request": {"max_tokens": 64, "timeout_s": 5, "min_s": 2,
                    "messages": [{"role": "user", "content": "0 1\n1 0"}]},
        "rate": {"enabled": True, "max_tokens": 512, "timeout_s": 5, "min_s": 1},
        "tail_lines": 50, "max_log_bytes": 100000, "error_patterns": [": error: argument"],
        "ignore_patterns": [], "kernel_frame_skip": [], "log_marks": {}, "variants": variants,
        **spec,
    }
    out = tmp_path / "out"
    out.mkdir()
    run = fp.Run(config=config, root=tmp_path, out=out, start=time.time(),
                 state={"mode": "development", "paths": {"model": str(tmp_path)},
                        "flag_overrides": {}})
    return fp.Inner(run=run, base=f"http://127.0.0.1:{port}", tok=None)


def test_the_loop_cycles_survivor_crash_hang_and_skips_by_rule(tmp_path):
    inner = make_inner(tmp_path, [
        {"name": "ok", "env": {"CUDA_LAUNCH_BLOCKING": "1"}},
        {"name": "refused", "flags": {"--fake-pool": "x"}},
        {"name": "hang", "flags": {"--boot-delay": 30}},
        {"name": "pin", "run_if": "none_survived"},
    ])
    record = fb.stage_bisect(inner)
    rows = json.loads((inner.run.out / "bisect.json").read_text())["rows"]
    assert record["outcomes"] == {"ok": "survived", "refused": "exited_at_boot",
                                  "hang": "boot_timeout", "pin": "not_run"}
    ok, refused, hang = rows[:3]
    assert ok["pool"]["pool_tokens"] == 20000 and ok["rate"]["status"] == 200
    assert ok["request"]["completion_tokens"] == 1 and ok["alive_after"] is True
    assert ok["health"]["last_status"] == 200 and hang["health"]["last_status"] is None
    assert refused["exit_code"] == 2 and "invalid int value" in refused["first_error"]["message"]
    assert hang["seconds"] < 6 + 3 and hang["stop"]["port_free"]
    assert len({json.dumps(r["argv"]) for r in rows[:3]}) == 3
    for row in rows[:3]:
        assert (inner.run.out / f"bisect_{row['index']}_{row['name']}_tail.log").is_file()
    assert "run_if none_survived" in rows[3]["reason"]


def test_a_variant_that_cannot_fit_before_the_end_is_not_run(tmp_path):
    inner = make_inner(tmp_path, [{"name": "late"}], end_s=2.0)
    record = fb.stage_bisect(inner)
    assert record["outcomes"] == {"late": "not_run"} and record["variants"] == 0


def test_the_stage_skips_when_off(tmp_path):
    inner = make_inner(tmp_path, [{"name": "x"}], enabled=False)
    with pytest.raises(fp.StageSkipped):
        fb.stage_bisect(inner)


def test_collected_logs_come_back_bounded_under_the_variants_name(tmp_path):
    inner = make_inner(tmp_path, [{"name": "x"}])
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "kapi_7.log").write_bytes(b"y" * 300000)
    names = fb.collect_logs(inner.run, {"name": "v4", "logs": ["{scratch}/kapi_*.log"]}, 4, folder)
    assert names == ["bisect_4_v4_kapi_7.log"]
    assert (inner.run.out / names[0]).stat().st_size < 100100


def test_a_variant_without_a_boot_window_is_not_run(tmp_path):
    """P5.2-06's V4 started with 163 s to the end but a boot window below zero."""
    inner = make_inner(tmp_path, [{"name": "late"}], end_s=5.0, min_variant_s=2, kill_reserve_s=2)
    record = fb.stage_bisect(inner)
    rows = json.loads((inner.run.out / "bisect.json").read_text())["rows"]
    assert record["outcomes"] == {"late": "not_run"} and "boot window" in rows[0]["reason"]


def test_the_reread_fills_the_marks_from_the_pulled_logs(tmp_path):
    out = tmp_path / "fn"
    out.mkdir()
    (out / "bisect.json").write_text(json.dumps({"rows": [
        {"index": 0, "name": "v1", "outcome": "boot_timeout", "marks": {}}]}))
    (out / "bisect_0_v1_engine.log").write_text(
        "[x TP0 EP0] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 449024, K\n")
    config = {"bisect": {"log_marks": {"kv_tokens": r"KV Cache is allocated\.[^\n]*?#tokens: (\d+)"},
                         **{k: SPEC_KEYS[k] for k in ("error_patterns",)},
                         "ignore_patterns": [], "kernel_frame_skip": []}}
    data = fb.reread(out, config)
    assert data["rows"][0]["marks"] == {"kv_tokens": 449024.0} and data["reread"] is True
