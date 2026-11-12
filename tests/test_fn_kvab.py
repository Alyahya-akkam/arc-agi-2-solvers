"""Run P5.5-01 (task P5.5): the KV-cache A/B probe, scripts/fn_kvab.py and its Modal job.

Checks the readings' arithmetic and rules (the brief's pre-registered rows), the teacher-forced
text's source, the builder's extends lists and kernel bundle, the probe image's pytest, the job
(its cost bound under the 4.5 USD cap, P5.2-07's inputs and boot settings for A, the KV4
variables only in B and C, every KV4 edit required), and the CPU smoke of the whole path on the
fake server. No network, no GPU, no Modal client.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kv = _load("fn_kvab_under_test", SCRIPTS / "fn_kvab.py")
fake = _load("fn_fake_kvab_under_test", SCRIPTS / "fn_fake_kvab.py")
fn_builder = _load("build_fn_notebooks_kvab_under_test", SCRIPTS / "build_fn_notebooks.py")

SHA = "a" * 40
WHEEL = "arc26-0.0.1-py3-none-any.whl"
READINGS = {"pool_ratio_min": 1.6, "top1_min": 0.99, "mean_abs_lp_max": 0.02,
            "entropy_ratio_max": 1.05, "length_ratio_max": 1.20, "length_min_tasks": 2,
            "decode_ratio_min": 0.95}
# P5.2-07's server flags (its fn_config.json and engine_commands.jsonl, 30 Oct).
P5207_FLAGS = {
    "--model-path": "{model}", "--load-format": "safetensors",
    "--model-loader-extra-config": {"enable_multithread_load": False},
    "--weight-loader-prefetch-checkpoints": True, "--weight-loader-drop-cache-after-load": True,
    "--served-model-name": "flash-next-w4a16", "--host": "127.0.0.1", "--port": "{port}",
    "--tensor-parallel-size": 2, "--expert-parallel-size": 2, "--moe-a2a-backend": "none",
    "--disable-custom-all-reduce": True, "--dtype": "bfloat16", "--quantization": "auto-round",
    "--moe-runner-backend": "auto", "--ple-offload-embedding": True, "--trust-remote-code": True,
    "--attention-backend": "triton", "--linear-attn-backend": "triton",
    "--linear-attn-decode-backend": "triton", "--linear-attn-prefill-backend": "triton",
    "--mamba-backend": "triton", "--kv-cache-dtype": "fp8_e4m3", "--page-size": 64,
    "--mamba-ssm-dtype": "bfloat16", "--mamba-radix-cache-strategy": "extra_buffer",
    "--mamba-track-interval": 64, "--max-mamba-cache-size": 44, "--mem-fraction-static": 0.92,
    "--context-length": 86016, "--max-running-requests": 8, "--chunked-prefill-size": 4096,
    "--max-prefill-tokens": 8192, "--cuda-graph-max-bs-decode": 8,
    "--cuda-graph-bs-decode": [1, 2, 3, 4, 5, 6, 7, 8], "--chat-template": "{chat_template}",
    "--reasoning-parser": "qwen3", "--default-chat-template-kwargs": {"preserve_thinking": True},
    "--schedule-policy": "lpm", "--watchdog-timeout": 1800, "--enable-metrics": True,
    "--enable-cache-report": True, "--incremental-streaming-output": True,
    "--stream-interval": 16,
}
KV4_VARS = ("ARC26_KV4", "ARC26_KV4_HADAMARD", "ARC26_KV4_GROUP")


def read(name):
    return (REPO / name).read_bytes()


def tf(lp, top1, entropy):
    return {"lp": list(lp), "top1": list(top1), "entropy": list(entropy)}


def row(name, role="candidate", **extra):
    base = {"name": name, "role": role, "outcome": "measured", "alive_after": True,
            "first_error": None, "tf": {"status": "ok", "compare": {
                "top1_agreement": 1.0, "mean_abs_lp_diff": 0.001, "p99_abs_lp_diff": 0.01,
                "entropy_ratio": 1.0}},
            "pool": {"pool_tokens": 1000}, "request_errors": [],
            "decode": {"1": {"log": {"per_stream_tok_s": 60.0}},
                       "3": {"log": {"per_stream_tok_s": 50.0}}},
            "dev_lengths": {"lengths": {"t1": 100, "t2": 200, "t3": 300}, "median": 200,
                            "cut": 0}}
    base.update(extra)
    return base


# -- the arithmetic ------------------------------------------------------------------------------


def test_percentile_is_the_nearest_rank_and_median_averages_the_middle_pair():
    values = list(range(1, 101))
    assert kv.percentile(values, 0.99) == 99 and kv.percentile(values, 1.0) == 100
    assert kv.percentile([5.0], 0.99) == 5.0 and kv.percentile([], 0.5) is None
    assert kv.median([3, 1, 2]) == 2 and kv.median([4, 1, 3, 2]) == 2.5 and kv.median([]) is None


def test_tf_positions_reads_sglangs_shapes_and_refuses_a_short_reply():
    ids = [7, 8, 9]
    meta = fake.logprob_meta(ids, 3, 0)
    out = kv.tf_positions(meta, 3)
    assert out["ids_back"] == [8, 9] and out["top1"] == [8, 9]
    ln2 = fake.LN2
    assert out["lp"] == pytest.approx([-ln2, -ln2])
    # Top-3 at 1/2, 1/4, 1/8: H = sum p (-log p), mass 7/8.
    h = sum(2.0 ** -(i + 1) * (i + 1) * ln2 for i in range(3))
    assert out["entropy"] == pytest.approx([h, h]) and out["mass"] == pytest.approx([0.875] * 2)
    with pytest.raises(ValueError, match="covers 3 and 3 positions, not 4"):
        kv.tf_positions(meta, 4)
    with pytest.raises(ValueError, match="no input_token_logprobs"):
        kv.tf_positions({}, 3)
    meta["input_top_logprobs"][2] = None
    with pytest.raises(ValueError, match="position 2"):
        kv.tf_positions(meta, 3)


def test_compare_tf_gives_the_brief_formulas():
    base = tf([-1.0, -2.0, -0.5, -0.1], [1, 2, 3, 4], [1.0, 2.0, 1.0, 0.0])
    other = tf([-1.1, -2.0, -0.4, -0.1], [1, 2, 9, 4], [1.2, 2.0, 1.0, 0.2])
    got = kv.compare_tf(base, other)
    assert got["positions"] == 4 and got["top1_agreement"] == 0.75
    assert got["mean_abs_lp_diff"] == pytest.approx(0.05)
    assert got["p99_abs_lp_diff"] == pytest.approx(0.1) == got["max_abs_lp_diff"]
    assert got["entropy_ratio"] == pytest.approx(4.4 / 4.0)
    assert got["mean_nll"] == pytest.approx(3.6 / 4) and got["base_mean_nll"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="positions differ"):
        kv.compare_tf(base, tf([-1.0], [1], [1.0]))


def test_decode_rate_lines_takes_the_median_line_at_the_stream_count_per_stream():
    text = "\n".join([
        "Decode batch, #running-req: 3, #full token: 7104, gen throughput (token/s): 150.00",
        "Decode batch, #running-req: 3, #full token: 7104, gen throughput (token/s): 180.00",
        "Decode batch, #running-req: 3, #full token: 7104, gen throughput (token/s): 171.00",
        "Decode batch, #running-req: 2, #full token: 7104, gen throughput (token/s): 120.00",
        "Decode batch, #running-req: 3, #full token: 7104, gen throughput (token/s): 0.00",
    ])
    assert kv.decode_rate_lines(text, 3) == {"lines": 3, "per_stream_tok_s": 57.0}
    assert kv.decode_rate_lines(text, 2) == {"lines": 1, "per_stream_tok_s": 60.0}
    assert kv.decode_rate_lines(text, 1) == {"lines": 0, "per_stream_tok_s": None}


def test_dev_lengths_records_the_natural_end_or_the_cap_and_counts_cuts():
    def record(task, finish, tokens, **extra):
        return {"task_id": task, "turns": [{"kind": "think", "finish": finish,
                                            "completion_tokens": tokens},
                                           {"kind": "force", "finish": "stop",
                                            "completion_tokens": 500}], **extra}

    got = kv.dev_lengths([record("a", "stop", 900), record("b", "length", 16384),
                          record("c", "cancelled", 4000), record("d", "stop", 1, error="x")],
                         16384)
    assert got["lengths"] == {"a": 900, "b": 16384, "c": 4000}
    assert got["ended"] == {"a": "natural", "b": "cap", "c": "cut", "d": "error"}
    assert (got["median"], got["capped"], got["cut"]) == (4000, 1, 2)


def test_speed_ratios_prefer_the_log_rate_and_fall_back_to_the_wall_rate():
    base = {"1": {"log": {"per_stream_tok_s": 80.0}, "wall_per_stream_tok_s": 70.0},
            "2": {"log": {"per_stream_tok_s": None}, "wall_per_stream_tok_s": 50.0}}
    other = {"1": {"log": {"per_stream_tok_s": 76.0}, "wall_per_stream_tok_s": 70.0},
             "2": {"log": {"per_stream_tok_s": 70.0}, "wall_per_stream_tok_s": 45.0}}
    assert kv.speed_ratios(base, other) == {"by_streams": {"1": 0.95, "2": 0.9}, "min": 0.9}


# -- the readings (the brief's section 1) --------------------------------------------------------


def test_a_candidate_that_meets_every_rule_reads_good_and_passes_on_quality():
    base = row("a_fp8", role="baseline")
    b = row("b_kv4", pool={"pool_tokens": 1636})
    got = kv.judge(b, base, [base], READINGS)
    assert got["reading"] == "good" and got["quality_pass"] is True and got["misses"] == []
    assert got["values"]["pool_ratio"] == 1.636 and got["values"]["length_ratio"] == 1.0
    assert got["values"]["decode_ratio_min"] == 1.0


@pytest.mark.parametrize("change, miss, quality", [
    ({"tf": {"status": "ok", "compare": {"top1_agreement": 0.98, "mean_abs_lp_diff": 0.001,
                                         "p99_abs_lp_diff": 0.0, "entropy_ratio": 1.0}}},
     "top-1 agreement", False),
    ({"tf": {"status": "ok", "compare": {"top1_agreement": 1.0, "mean_abs_lp_diff": 0.03,
                                         "p99_abs_lp_diff": 0.0, "entropy_ratio": 1.0}}},
     "mean |dlogp|", False),
    ({"tf": {"status": "ok", "compare": {"top1_agreement": 1.0, "mean_abs_lp_diff": 0.001,
                                         "p99_abs_lp_diff": 0.0, "entropy_ratio": 1.06}}},
     "entropy ratio", False),
    ({"dev_lengths": {"lengths": {"t1": 300, "t2": 241, "t3": 400}, "median": 241, "cut": 0}},
     "thinking length", False),
    ({"pool": {"pool_tokens": 1599}}, "pool ratio", True),
    ({"decode": {"1": {"log": {"per_stream_tok_s": 56.9}},
                 "3": {"log": {"per_stream_tok_s": 50.0}}}}, "decode ratio", True),
    ({"request_errors": ["dev t1: boom"]}, "request errors", True),
])
def test_each_missed_rule_reads_bad_and_names_itself(change, miss, quality):
    base = row("a_fp8", role="baseline")
    b = row("b_kv4", pool={"pool_tokens": 1700})
    b.update(change)
    got = kv.judge(b, base, [base], READINGS)
    assert got["reading"] == "bad" and got["quality_pass"] is quality
    assert len(got["misses"]) == 1 and miss in got["misses"][0]


def test_a_cut_dev_task_leaves_the_length_unjudged_not_bad():
    base = row("a_fp8", role="baseline")
    b = row("b_kv4", pool={"pool_tokens": 1700},
            dev_lengths={"lengths": {"t1": 900, "t2": 900}, "median": 900, "cut": 1})
    got = kv.judge(b, base, [base], READINGS)
    assert got["reading"] == "good" and got["values"]["length_ratio"] is None
    assert "length_not_judged" in got["values"]


@pytest.mark.parametrize("change", [
    {"outcome": "boot_timeout"}, {"outcome": "patch_missing"}, {"alive_after": False},
    {"first_error": {"fatal_line": "CUDA error: an illegal memory access"}},
    {"tf": {"status": "failed", "error": "HTTP 500"}}, {"tf": {"status": "ok"}},
])
def test_a_candidate_that_did_not_serve_reads_fault(change):
    base = row("a_fp8", role="baseline")
    b = row("b_kv4")
    b.update(change)
    assert kv.judge(b, base, [base], READINGS)["reading"] == "fault"


def test_without_a_serving_baseline_every_candidate_reads_fault_and_the_run_fault():
    base = row("a_fp8", role="baseline", outcome="exited_at_boot")
    assert kv.judge(base, None, [], READINGS)["reading"] == "fault"
    assert kv.judge(row("b_kv4"), base, [base], READINGS)["reading"] == "fault"
    base["judged"] = kv.judge(base, None, [], READINGS)
    assert kv.decision([base]) == "fault"


def test_the_pool_is_read_against_the_variant_named_by_pool_against():
    base = row("a_fp8", role="baseline", pool={"pool_tokens": 1000})
    b = row("b_kv4", pool={"pool_tokens": 1700})
    c = row("c_vision", pool={"pool_tokens": 1800}, pool_against="b_kv4", pool_ratio_min=1.05)
    got = kv.judge(c, base, [base, b], READINGS)
    assert got["values"]["pool_against"] == "b_kv4" and got["values"]["pool_ratio"] == 1.0588
    assert got["reading"] == "good"
    c["pool"] = {"pool_tokens": 1750}
    assert "pool ratio" in kv.judge(c, base, [base, b], READINGS)["misses"][0]


def test_run_if_chooses_c_by_bs_quality_and_skips_after_a_fault():
    b = {"name": "b_kv4", "judged": {"reading": "bad", "quality_pass": True}}
    vision = {"name": "c_vision", "run_if": "quality_pass:b_kv4"}
    vonly = {"name": "c_vonly", "run_if": "quality_fail:b_kv4"}
    assert kv.should_run(vision, [b]) is None and kv.should_run(vonly, [b]) is not None
    b["judged"]["quality_pass"] = False
    assert kv.should_run(vision, [b]) is not None and kv.should_run(vonly, [b]) is None
    b["judged"] = {"reading": "fault", "quality_pass": None}
    assert "fault" in kv.should_run(vision, [b]) and "fault" in kv.should_run(vonly, [b])
    assert kv.should_run({"name": "x"}, []) is None
    assert "no reading" in kv.should_run({"name": "b", "run_if": "healthy:a_fp8"}, [])


def test_the_decision_is_the_first_candidates_reading():
    base = {"name": "a", "role": "baseline", "outcome": "measured", "judged": {"reading": "ok"}}
    b = {"name": "b", "role": "candidate", "outcome": "measured", "judged": {"reading": "bad"}}
    c = {"name": "c", "role": "candidate", "outcome": "measured", "judged": {"reading": "good"}}
    assert kv.decision([base, b, c]) == "bad"
    b["outcome"] = "not_run"
    assert kv.decision([base, b, c]) == "fault"


def test_check_variants_refuses_what_the_run_cannot_mean():
    good = [{"name": "a", "role": "baseline"}, {"name": "b", "role": "candidate",
                                                "run_if": "healthy:a"}]
    assert [v["name"] for v in kv.check_variants({"variants": good})] == ["a", "b"]
    bad = [
        ([{"name": "a", "role": "candidate"}], "one baseline, first"),
        ([{"name": "a", "role": "baseline"}, {"name": "a", "role": "candidate"}], "repeated"),
        ([{"name": "a", "role": "baseline", "flag": 1}], "unknown"),
        ([{"name": "a", "role": "baseline"}, {"name": "b", "role": "candidate",
                                              "run_if": "healthy:c"}], "earlier variant"),
        ([{"name": "a", "role": "baseline"}, {"name": "b", "role": "candidate",
                                              "edits": [{"name": "e"}]}], "hash guard"),
    ]
    for variants, match in bad:
        with pytest.raises(ValueError, match=match):
            kv.check_variants({"variants": variants})


# -- the teacher-forced text ---------------------------------------------------------------------


def test_the_transcript_text_is_the_rendered_prompt_plus_the_reasoning_cut_and_hashed(tmp_path):
    tok = kv.fp.FakeTokenizer()
    record = {"attempt_id": "t/p0/0", "messages": [
        {"role": "system", "content": "S"}, {"role": "user", "content": "U"},
        {"role": "assistant", "content": "ANSWER", "reasoning": "think think"}]}
    other = {"attempt_id": "o/p0/0", "messages": record["messages"]}
    folder = tmp_path / "out"
    (folder / "run_logs").mkdir(parents=True)
    path = folder / "run_logs" / "t.jsonl"
    path.write_text(json.dumps(other) + "\n" + json.dumps(record) + "\n")
    source = {"kind": "transcript", "input": "transcripts", "file": "run_logs/t.jsonl",
              "sha256": kv.fp.sha256_file(path), "attempt_id": "t/p0/0"}
    ids, got = kv.tf_text_ids(tok, source, {}, {"transcripts": str(folder)}, 10_000)
    want = tok.encode(tok.render(record["messages"][:2]) + "think think")
    assert ids == want and "ANSWER" not in tok.decode(ids)
    assert got["tokens"] == len(want) and got["prompt_messages"] == 2
    assert len(got["ids_sha256"]) == 64
    cut, _ = kv.tf_text_ids(tok, source, {}, {"transcripts": str(folder)}, 5)
    assert cut == want[:5]
    with pytest.raises(ValueError, match="sha256"):
        kv.tf_text_ids(tok, {**source, "sha256": "0" * 64}, {}, {"transcripts": str(folder)}, 9)
    with pytest.raises(ValueError, match="no record"):
        kv.tf_text_ids(tok, {**source, "attempt_id": "x"}, {}, {"transcripts": str(folder)}, 9)
    with pytest.raises(ValueError, match="not found"):
        kv.tf_text_ids(tok, source, {}, {}, 9)


# -- the builder ---------------------------------------------------------------------------------


# -- the job --------------------------------------------------------------------------------------


# -- the whole path on the CPU -------------------------------------------------------------------


def test_a_measurement_bug_is_recorded_and_the_server_still_stopped(tmp_path, monkeypatch):
    port = 30959
    config = {
        "run_id": "t", "paths": {"scratch": str(tmp_path / "scratch")},
        # A server process that exits at once: the health wait is replaced below, and the stop
        # must cope with a process already gone.
        "engine": {"port": port, "module": "this", "script": None, "flags": {}},
        "kvab": {"boots": {"measure_floor_s": 1, "poll_s": 0.1, "stop_settle_s": 5,
                           "gpu_free_mib": 2048, "settle_s": 0},
                 "measure": {}, "base_flags": {}, "base_env": {}, "error_patterns": ["boom"],
                 "ignore_patterns": [], "kernel_frame_skip": [], "log_marks": {},
                 "tail_lines": 10, "max_log_bytes": 10000},
    }
    out = tmp_path / "out"
    out.mkdir()
    run = kv.fp.Run(config=config, root=tmp_path, out=out, start=kv.fp._now(),
                    state={"paths": {"model": ""}, "scratch": str(tmp_path)})
    inner = kv.fp.Inner(run=run, base=f"http://127.0.0.1:{port}", tok=None)
    monkeypatch.setattr(kv.fb, "wait_healthy", lambda *a, **k: "healthy")

    def boom(_inner):
        raise KeyError("a field the reply lacks")

    monkeypatch.setattr(kv.fb, "pool_reading", boom)
    row = kv.boot_variant(inner, {"name": "b", "role": "candidate"}, 1, None,
                          kv.fp._now() + 30, [1, 2], [])
    assert row["outcome"] == "error" and "KeyError" in row["error"] and row["traceback"]
    assert inner.proc is None and row["stop"]["port_free"]
    assert kv.judge(row, {"name": "a", "role": "baseline", "outcome": "measured",
                          "tf": {"status": "ok"}}, [], READINGS)["reading"] == "fault"


def test_the_fake_pool_follows_the_runs_settings():
    argv = ["--port", "1", "--fake-pool", "20000"]
    assert fake.pool_argv(argv, {}) == argv
    assert fake.pool_argv(argv, {"ARC26_KV4": "1"})[-2:] == ["--fake-pool", "34000"]
    assert fake.pool_argv(argv, {"ARC26_KV4": "v"})[-2:] == ["--fake-pool", "25000"]
    vision = [*argv, "--json-model-override-args", '{"language_model_only": true}']
    got = fake.pool_argv(vision, {"ARC26_KV4": "1"})
    assert got[-2:] == ["--fake-pool", str(round(20000 * 1.7 * 1.18))]
    assert got.count("--fake-pool") == 1


