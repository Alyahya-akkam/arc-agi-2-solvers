"""P3.1-01: scripts/inject_parity.py on the CPU (docs/runbooks/inject_parity.md).

The smoke end to end, the decision logic on constructed bf16 logits (exact, within 2 ULPs with
every argmax equal, beyond), the input search against the script notebook's own, the sequences
against the notebook's construction, the adapter, the configs, and the build of
configs/script_P3.1-01.yaml: its metadata through the gateway's checks, and its script notebook
accepting it in a simulated Kaggle tree, where the script runs on a tiny saved model.

Skipped when torch, transformers, peft, or safetensors is missing (CI's main job). Nothing touches
the Hugging Face Hub or a GPU: every subprocess gets CUDA_VISIBLE_DEVICES="" and HF_HUB_OFFLINE=1.
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402
import yaml  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from arc26.hyper.inject import (  # noqa: E402
    InjectableLinear,
    inject,
    rslora_scaling,
    save_adapter,
    wrap_targets,
)
from arc26.hyper.train import standin_tokenizer  # noqa: E402
from arc26.nvarc.arc_loader import QwenFormatter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = load("inject_parity", REPO / "scripts" / "inject_parity.py")
nbb = load("build_kaggle_notebook", REPO / "scripts" / "build_kaggle_notebook.py")
ops = load("kaggle_ops", REPO / "scripts" / "kaggle_ops.py")

CONFIG = REPO / "configs" / "inject_parity.yaml"
BUILD_CONFIG = "configs/script_P3.1-01.yaml"
HYPERNET = REPO / "configs" / "hypernet.yaml"
NOTEBOOK = "alyahya737/arc26-p3-1-parity"
DEV4 = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
# The names of the 9 sequences of the smoke and of the GPU run: per task its decode prompts (one
# per test input, DEV4 has 1, 2, 1, 1), then its training text.
NAMES = [
    "0934a4d8_0", "0934a4d8", "36a08778_0", "36a08778_1", "36a08778", "981571dc_0", "981571dc",
    "aa4ec2a5_0", "aa4ec2a5",
]  # fmt: skip
OUTPUTS = {"config.yaml", "parity.json", "summary.md", "timing-parity.jsonl"}
NOTEBOOK_LOGS = {"candidates.json", "env.json", "pip_freeze.txt", "stages.json"}
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
THRESHOLDS = parity.Thresholds(max_abs_diff=1e-3, max_ulp=2, min_mean_effect=1e-2)
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_main(*args):
    """The script in process; the smoke and every test here run on the CPU."""
    return parity.main([str(a) for a in args])


def smoke(tmp_path, config=CONFIG, *extra):
    logs = tmp_path / "logs"
    code = run_main(
        "--config", config, "--smoke", "--input-root", tmp_path / "input", "--out-dir", logs,
        *extra,
    )  # fmt: skip
    return code, logs


def config_with(tmp_path, change):
    """A copy of configs/inject_parity.yaml with ``change`` applied to its parsed mapping."""
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "inject_parity.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# The smoke end to end


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    code, logs = smoke(tmp)
    return code, logs, tmp


def test_the_smoke_passes_with_bitwise_parity_and_flowing_gradients(smoke_run):
    code, logs, tmp = smoke_run
    assert code == parity.EXIT_OK
    assert {p.name for p in logs.iterdir()} == OUTPUTS  # no error.log, none of the notebook's
    report = read_json(logs / "parity.json")
    assert report["status"] == "complete" and report["exit_code"] == 0
    assert report["decision"]["outcome"] == parity.PASS, report["decision"]
    assert report["smoke"] is True and report["run_id"] == "P3.1-smoke"
    records = report["sequences"]
    assert [r["name"] for r in records] == NAMES
    assert [r["kind"] for r in records].count("decode") == 5
    assert [r["kind"] for r in records].count("train") == 4
    worst_effect = math.inf
    for r in records:
        assert r["parity"]["differing"] == 0 and r["parity"]["max_abs_diff"] == 0.0
        assert r["parity"]["max_ulp"] == 0 and r["parity"]["argmax_mismatch"] == 0
        assert r["parity"]["class"] == parity.EXACT
        assert r["zero_b"]["differing"] == 0 and r["zero_b"]["class"] == parity.EXACT
        assert r["repeat"]["base"]["differing"] == 0 and r["repeat"]["peft"]["differing"] == 0
        for swap in ("load_adapter", "set_peft_model_state_dict"):
            assert r["swaps"][swap]["differing"] == 0, (r["name"], swap)
        assert r["effect"] == r["effect_injected"]  # parity: the same logits
        assert r["effect"]["mean_abs"] >= THRESHOLDS.min_mean_effect
        assert r["effect"]["changed_fraction"] > 0.5
        worst_effect = min(worst_effect, r["effect"]["mean_abs"])
        g = r["gradient"]
        if r["kind"] == "decode":
            assert g is None and r["target_tokens"] is None
            continue
        assert g["flows"] and g["finite"] and g["zero_tensors"] == 0 and g["missing"] == 0
        assert g["tensors"] == 2 * 2 * len(TARGETS)  # A and B, 2 layers, 4 projections
        assert g["loss_equal"] and g["loss_injected"] == g["loss_injected_repeat"]
        assert g["rel_diff_peft"] == 0.0 and g["rel_diff_repeat"] == 0.0
        assert set(g) == set(parity.GRADIENT_KEYS)
        assert g["error"] is None and g["repeat_error"] is None and g["peft_error"] is None
        # Checkpointing recomputed every layer: each projection ran twice per backward.
        assert g["calls_per_projection"] == {"inject": 2.0, "inject_repeat": 2.0, "peft": 2.0}
        assert r["target_tokens"] > 0
    assert report["model"]["checkpointing"] == {"modules": 3, "use_reentrant": False}
    assert report["error_log"] is False
    print(f"P3.1-01 smoke: 9 sequences bitwise; smallest mean adapter effect {worst_effect:.3f}")


def test_the_smoke_records_its_provenance_inputs_and_adapter(smoke_run):
    _, logs, tmp = smoke_run
    report = read_json(logs / "parity.json")
    inputs = report["inputs"]
    challenges = Path(inputs["challenges"])
    assert challenges == (
        tmp / "input" / "competitions" / "arc-prize-2026-arc-agi-2"
        / "arc-agi_evaluation_challenges.json"
    )  # fmt: skip
    assert inputs["challenges_sha256"] == hashlib.sha256(challenges.read_bytes()).hexdigest()
    assert Path(inputs["model_dir"]) == tmp / "input" / "models" / "qwen3_4b_grids15_sft139"
    assert inputs["task_ids"] == DEV4 and inputs["subset"] == "DEV4"
    adapter = report["adapter"]
    assert adapter["projections"] == 8 and adapter["scaling"] == 8.0
    assert adapter["rank"] == 16 and adapter["alpha"] == 32
    assert adapter["peft_dtype"] == "float32" and adapter["peft_scaling"] == 8.0
    assert re.fullmatch(r"[0-9a-f]{64}", adapter["sha256"])
    model = report["model"]
    assert model["dtype"] == "bfloat16" and model["attn_implementation"] == "sdpa"
    assert model["num_hidden_layers"] == 2 and model["vocab_size"] == 16
    prov = report["provenance"]
    assert prov["versions"]["peft"] == parity.PEFT_REPLAYED
    assert Path(prov["arc26"]).is_relative_to(REPO / "arc26")
    assert report["device"]["type"] == "cpu" and "cuda_matmul_allow_tf32" in report["precision"]
    config = yaml.safe_load((logs / "config.yaml").read_text(encoding="utf-8"))
    assert config["device"] == "cpu" and config["run_id"] == "P3.1-smoke"
    assert config["out_dir"] == str(logs)


def test_the_smoke_writes_a_timing_record_per_gpu_op(smoke_run):
    _, logs, _ = smoke_run
    records = [json.loads(line) for line in (logs / "timing-parity.jsonl").read_text().split("\n")
               if line]  # fmt: skip
    assert all(list(r) == ["puzzle_id", "op", "seconds", "tokens"] for r in records)
    ops_seen = {r["op"] for r in records}
    passes = ("base", "zero_b", "inject", "base_repeat", "peft", "peft_repeat")
    passes += ("swap_load_adapter", "swap_state_dict")
    assert ops_seen == {
        "load", "peft_load", "grad_inject", "grad_inject_repeat", "grad_peft",
        *(f"forward_{p}" for p in passes),
    }  # fmt: skip
    report = read_json(logs / "parity.json")
    tokens = {(r["task_id"], r["tokens"]) for r in report["sequences"]}
    for p in passes:
        forwards = [r for r in records if r["op"] == f"forward_{p}"]
        assert len(forwards) == 9 and {(r["puzzle_id"], r["tokens"]) for r in forwards} == tokens
    assert sum(r["op"] == "grad_peft" for r in records) == 4


def test_the_smoke_summary_reads_the_decision(smoke_run):
    _, logs, _ = smoke_run
    summary = (logs / "summary.md").read_text(encoding="utf-8")
    assert "- Outcome: **Pass**. Every sequence below 0.001 (largest 0)" in summary
    assert "Smoke run:" in summary and "only the GPU run decides P3.1" in summary
    for name in NAMES:
        assert f"| {name} |" in summary
    assert chr(0x2014) not in summary


def test_the_smoke_cuts_sequences_as_the_notebook_does(smoke_run):
    _, logs, _ = smoke_run
    records = {r["name"]: r for r in read_json(logs / "parity.json")["sequences"]}
    # 0934a4d8: 5 pairs of 426 tokens; training cap 2048 keeps 4; decode cap 2048 - 931 = 1117
    # with a 216-token query keeps 2.
    assert (records["0934a4d8"]["pairs"], records["0934a4d8"]["dropped_pairs"]) == (4, 1)
    assert records["0934a4d8"]["key"] == "0934a4d8.ex1234"
    assert (records["0934a4d8_0"]["pairs"], records["0934a4d8_0"]["dropped_pairs"]) == (2, 3)
    assert all(r["tokens"] <= 2048 for r in records.values())
    assert all(r["tokens"] <= 2048 - 931 for r in records.values() if r["kind"] == "decode")
    assert records["981571dc_0"]["dropped_pairs"] == 0


def test_a_rerun_replaces_the_previous_outputs(tmp_path):
    code, logs = smoke(tmp_path)
    assert code == 0
    first = (logs / "timing-parity.jsonl").read_text().count("\n")
    (logs / "error.log").write_text("stale")
    code, logs = smoke(tmp_path)
    assert code == 0 and not (logs / "error.log").exists()
    assert (logs / "timing-parity.jsonl").read_text().count("\n") == first


def test_a_vacuous_adapter_is_an_error_not_a_pass(tmp_path):
    def tiny_adapter(data):
        data["smoke"]["adapter"] = {"relative_norm": 1e-6}

    code, logs = smoke(tmp_path, config_with(tmp_path, tiny_adapter))
    assert code == parity.EXIT_OK  # measured; the decision is the outcome
    report = read_json(logs / "parity.json")
    assert report["decision"]["outcome"] == parity.ERROR
    assert "moved the logits by less than 0.01" in report["decision"]["reasons"][0]
    assert all(r["parity"]["differing"] == 0 for r in report["sequences"])


def test_the_wall_budget_stops_the_run_with_its_logs(tmp_path):
    def tight(data):
        data["smoke"]["max_wall_s"] = 1e-6

    code, logs = smoke(tmp_path, config_with(tmp_path, tight))
    assert code == parity.EXIT_STOPPED
    report = read_json(logs / "parity.json")
    assert report["status"] == "stopped" and report["exit_code"] == 3
    assert report["decision"]["outcome"] == parity.ERROR
    assert "wall budget" in report["decision"]["reasons"][0]
    assert "**Error**" in (logs / "summary.md").read_text(encoding="utf-8")


def test_a_missing_input_is_an_error_with_a_traceback(tmp_path):
    (tmp_path / "input").mkdir()
    logs = tmp_path / "logs"
    code = run_main(
        "--config", CONFIG, "--input-root", tmp_path / "input", "--out-dir", logs,
        "--device", "cpu", "--commit", "abc123",
    )  # fmt: skip
    assert code == parity.EXIT_FAILED
    assert {p.name for p in logs.iterdir()} == {"config.yaml", "parity.json", "summary.md",
                                                 "error.log"}  # fmt: skip
    report = read_json(logs / "parity.json")
    assert report["status"] == "failed" and report["provenance"]["commit"] == "abc123"
    assert report["error_log"] is True
    (reason,) = report["decision"]["reasons"]
    assert reason.startswith("The run did not finish: FileNotFoundError: 0 model folders")
    assert "Traceback" in (logs / "error.log").read_text()


def test_a_failed_backward_is_recorded_on_its_text_and_the_run_goes_on(tmp_path, monkeypatch):
    """An OOM in a backward (simulated) no longer ends the run: the logit comparisons and the
    swaps are all measured, the text's gradient does not flow, and the decision is Fail with
    the reason, as the brief's section 1 reads."""
    original = parity.ParityRun.gradient_pass
    failing = {("inject", "981571dc"), ("inject_repeat", "0934a4d8"), ("peft", "aa4ec2a5")}

    def gradient_pass(self, model, label, factors, seq):
        if (label, seq.name) in failing:
            raise RuntimeError(f"CUDA out of memory ({label}, simulated)")
        return original(self, model, label, factors, seq)

    monkeypatch.setattr(parity.ParityRun, "gradient_pass", gradient_pass)
    code, logs = smoke(tmp_path)
    assert code == parity.EXIT_OK
    report = read_json(logs / "parity.json")
    assert report["status"] == "complete" and report["error_log"] is True
    records = {r["name"]: r for r in report["sequences"]}
    for r in records.values():
        assert r["parity"]["class"] == parity.EXACT and r["zero_b"]["class"] == parity.EXACT
        assert set(r["swaps"]) == {"load_adapter", "set_peft_model_state_dict"}
    failed = records["981571dc"]["gradient"]
    assert set(failed) == set(parity.GRADIENT_KEYS)
    assert failed["flows"] is False and failed["loss_injected"] is None
    assert failed["error"] == "RuntimeError: CUDA out of memory (inject, simulated)"
    assert failed["calls_per_projection"] == {"peft": 2.0}  # PEFT's backward still ran
    assert failed["loss_peft"] is not None and failed["rel_diff_peft"] is None
    repeat = records["0934a4d8"]["gradient"]
    assert repeat["flows"] and repeat["rel_diff_repeat"] is None
    assert repeat["repeat_error"] == "RuntimeError: CUDA out of memory (inject_repeat, simulated)"
    peft = records["aa4ec2a5"]["gradient"]
    assert peft["flows"] and peft["loss_peft"] is None and peft["rel_diff_peft"] is None
    assert peft["peft_error"] == "RuntimeError: CUDA out of memory (peft, simulated)"
    decision = report["decision"]
    assert decision["outcome"] == parity.FAIL
    assert decision["reasons"] == [
        "A factor gradient is missing, zero, or not finite, or its backward failed, on "
        "['981571dc (RuntimeError: CUDA out of memory (inject, simulated))']"
    ]
    log = (logs / "error.log").read_text(encoding="utf-8")
    for label, name in sorted(failing):
        assert f"{label} gradient on {name}:" in log
    summary = (logs / "summary.md").read_text(encoding="utf-8")
    assert "- 981571dc: the injected backward failed: RuntimeError" in summary
    assert "- 0934a4d8: the repeated injected backward failed" in summary
    assert "- aa4ec2a5: the PEFT backward failed" in summary
    assert "config.yaml, error.log." in summary


def test_without_checkpointing_each_projection_runs_once(tmp_path, monkeypatch):
    """The calls per projection are the evidence that checkpointing ran: with transformers'
    checkpointing switched off, every projection runs once per backward, and parity.json says so."""
    from transformers import PreTrainedModel

    monkeypatch.setattr(PreTrainedModel, "gradient_checkpointing_enable", lambda self, **kw: None)
    code, logs = smoke(tmp_path)
    assert code == parity.EXIT_OK
    report = read_json(logs / "parity.json")
    assert report["model"]["checkpointing"] == {"modules": 0, "use_reentrant": None}
    for r in report["sequences"]:
        if r["kind"] == "train":
            calls = r["gradient"]["calls_per_projection"]
            assert calls == {"inject": 1.0, "inject_repeat": 1.0, "peft": 1.0}


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs CUDA hidden (BRIEF A0)")
def test_cuda_requested_without_a_gpu_is_an_error(tmp_path):
    """The GPU config as committed, with CUDA hidden: the run refuses before any input is read."""
    root = tmp_path / "input"
    cfg = parity.load_config(CONFIG, True, {"input_root": str(root)})
    parity.prepare_smoke_inputs(cfg)
    logs = tmp_path / "logs"
    code = run_main("--config", CONFIG, "--input-root", root, "--out-dir", logs)
    assert code == parity.EXIT_FAILED
    (reason,) = read_json(logs / "parity.json")["decision"]["reasons"]
    assert "RuntimeError: cuda:0 requested, 0 CUDA devices visible" in reason


@pytest.mark.parametrize(
    "args",
    [["--smoke", "--model-dir", "m"], ["--smoke", "--challenges", "c"], ["--device", "cuda"]],
    ids=["smoke-model-dir", "smoke-challenges", "bad-device"],
)
def test_bad_command_lines_are_refused(tmp_path, args):
    assert run_main("--config", CONFIG, "--out-dir", tmp_path, *args) == parity.EXIT_CONFIG
    assert not (tmp_path / "parity.json").exists()


# The decision logic on constructed logits


def bf16(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)


def step(value, ulps):
    """``value`` (bf16, positive) moved by ``ulps`` representable values."""
    bits = bf16([value]).view(torch.int16) + ulps
    return float(bits.view(torch.bfloat16).float())


def test_bf16_ordinal_counts_representable_values():
    ordinal = parity.bf16_ordinal
    one = bf16([1.0])
    assert int(ordinal(bf16([step(1.0, 1)])) - ordinal(one)) == 1
    assert int(ordinal(bf16([step(2.0, 2)])) - ordinal(bf16([2.0]))) == 2
    assert int(ordinal(bf16([-1.0])) - ordinal(bf16([-step(1.0, 1)]))) == 1
    assert int(ordinal(bf16([-0.0])) - ordinal(bf16([0.0]))) == 0
    tiny = bf16([0.0]).view(torch.int16) + 1  # the smallest positive subnormal
    up, down = tiny.view(torch.bfloat16), (-tiny.view(torch.bfloat16))
    assert int(ordinal(up) - ordinal(down)) == 2  # through zero
    with pytest.raises(ValueError, match="bf16"):
        ordinal(torch.zeros(2))


BOUND = THRESHOLDS.max_abs_diff


def logits_pair(changes):
    """A reference [3 positions, 4 vocab] with margins, and a copy with ``changes`` applied."""
    reference = bf16([[4.0, 2.0, 1.0, -3.0], [0.5, 2.5, -1.0, 0.25], [0.1, 0.2, 0.0625, 3.0]])
    candidate = reference.clone()
    for (i, j), value in changes.items():
        candidate[i, j] = value
    return candidate, reference


def check(candidate, reference):
    return parity.compare(candidate, reference, BOUND)


def test_identical_logits_are_exact_under_both_criteria():
    candidate, reference = logits_pair({})
    stats = check(candidate, reference)
    assert stats["differing"] == 0 and stats["max_abs_diff"] == 0.0 and stats["max_ulp"] == 0
    assert stats["above_bound"] == 0 and stats["max_ulp_above_bound"] == 0
    assert stats["bound"] == BOUND
    assert parity.classify(stats, False, 2) == parity.EXACT
    assert parity.classify(stats, True, 2) == parity.EXACT


def test_a_small_logit_one_ulp_off_is_below_the_bound_but_not_bitwise():
    # 0.0625 is below 0.25, where one bf16 ULP (2^-11) is below the 1e-3 bound (conflict 3).
    candidate, reference = logits_pair({(2, 2): step(0.0625, 1)})
    stats = check(candidate, reference)
    assert stats["differing"] == 1 and stats["max_ulp"] == 1 and stats["above_bound"] == 0
    assert 0 < stats["max_abs_diff"] < BOUND
    assert parity.classify(stats, False, 2) == parity.EXACT
    assert parity.classify(stats, True, 2) == parity.WITHIN


def test_near_zero_logits_many_ulps_off_but_below_the_bound_do_not_count_ulps():
    # 0.01 has ULPs of 2^-14: 5 of them are 3.1e-4, below the bound, so no tolerance is needed.
    reference = bf16([[4.0, 2.0, 0.01, -3.0]])
    candidate = reference.clone()
    candidate[0, 2] = step(0.01, 5)
    stats = check(candidate, reference)
    assert (stats["max_ulp"], stats["above_bound"], stats["max_ulp_above_bound"]) == (5, 0, 0)
    assert parity.classify(stats, False, 2) == parity.EXACT
    assert parity.classify(stats, True, 2) == parity.WITHIN  # the zero-B control: not bitwise


def test_the_reviewers_case_is_for_the_human():
    """2.0 one ULP off (above the bound) and 0.01 five ULPs off (below it): every element is
    below B5's bound or within conflict 3's 2 ULPs, so the human decides. Counting the ULPs of
    every element, the first rule read 5 and failed it."""
    reference = bf16([[4.0, 2.0, 0.01, -3.0]])
    candidate = reference.clone()
    candidate[0, 1] = step(2.0, 1)
    candidate[0, 2] = step(0.01, 5)
    stats = check(candidate, reference)
    assert (stats["differing"], stats["above_bound"]) == (2, 1)
    assert (stats["max_ulp"], stats["max_ulp_above_bound"]) == (5, 1)
    assert stats["max_abs_diff"] == 2**-6 and stats["argmax_mismatch"] == 0
    assert parity.classify(stats, False, 2) == parity.WITHIN
    rec = {
        "name": "x",
        "kind": "decode",
        "parity": stats,
        "zero_b": check(reference, reference),
        "effect": {"mean_abs": 0.3},
        "gradient": None,
    }
    decision = parity.decide([rec], THRESHOLDS, expected=1)
    assert decision["outcome"] == parity.HUMAN, decision


@pytest.mark.parametrize("ulps", [1, 2])
def test_within_two_ulps_with_equal_argmax_is_for_the_human(ulps):
    # 2.0 and 2.5 moved by one or two ULPs (2^-7 below 2, 2^-6 above) are above the bound.
    candidate, reference = logits_pair({(1, 1): step(2.5, ulps), (0, 1): step(2.0, -ulps)})
    stats = check(candidate, reference)
    assert stats["above_bound"] == 2 and stats["max_ulp_above_bound"] == ulps
    assert stats["argmax_mismatch"] == 0 and stats["max_abs_diff"] >= BOUND
    assert parity.classify(stats, False, 2) == parity.WITHIN


def test_three_ulps_off_above_the_bound_is_beyond():
    candidate, reference = logits_pair({(0, 1): step(2.0, 3)})
    stats = check(candidate, reference)
    assert stats["max_ulp_above_bound"] == 3 and stats["argmax_mismatch"] == 0
    assert parity.classify(stats, False, 2) == parity.BEYOND
    assert parity.classify(stats, True, 2) == parity.BEYOND


def test_an_argmax_flip_by_one_ulp_is_beyond():
    reference = bf16([[1.0, step(1.0, 1)], [0.0, 1.0]])
    candidate = reference.clone()
    candidate[0, 0] = step(1.0, 2)  # one ULP above position 0's winner
    stats = check(candidate, reference)
    assert stats["max_ulp_above_bound"] == 2 and stats["argmax_mismatch"] == 1
    assert parity.classify(stats, False, 2) == parity.BEYOND


def test_non_finite_logits_are_beyond_and_flag_the_reference():
    candidate, reference = logits_pair({(0, 0): float("nan")})
    stats = check(candidate, reference)
    assert not stats["finite"] and stats["reference_finite"]
    assert stats["max_abs_diff"] is None and stats["max_ulp"] is None
    assert stats["above_bound"] is None and stats["max_ulp_above_bound"] is None
    assert parity.classify(stats, False, 2) == parity.BEYOND
    stats = check(reference, candidate)
    assert stats["finite"] and not stats["reference_finite"]
    with pytest.raises(ValueError, match="shapes"):
        check(reference, reference[:2])


def comparison(differing=0, max_abs=0.0, ulp=0, argmax=0, finite=True, reference_finite=True):
    """``compare``'s fields for a constructed case: the differing elements count as at or above
    the bound when the largest difference is, and then ``ulp`` is theirs."""
    ok = finite and reference_finite
    above = differing if max_abs >= BOUND else 0
    return {
        "bound": BOUND,
        "max_abs_diff": max_abs if ok else None,
        "differing": differing,
        "elements": 64,
        "above_bound": above if ok else None,
        "max_ulp": ulp if ok else None,
        "max_ulp_above_bound": (ulp if above else 0) if ok else None,
        "argmax_mismatch": argmax,
        "positions": 16,
        "finite": finite,
        "reference_finite": reference_finite,
    }


def record(
    name="t", kind="train", parity_stats=None, zero_b=None, mean=0.3, flows=True, error=None
):
    return {
        "name": name,
        "kind": kind,
        "parity": comparison() if parity_stats is None else parity_stats,
        "zero_b": comparison() if zero_b is None else zero_b,
        "effect": {"mean_abs": mean, "max_abs": 2.0},
        "gradient": {"flows": flows, "error": error} if kind == "train" else None,
    }


WITHIN_STATS = comparison(differing=3, max_abs=0.0625, ulp=2)
BEYOND_STATS = comparison(differing=3, max_abs=0.0625, ulp=3)
NEAR_ZERO = comparison(differing=2, max_abs=3e-4, ulp=5)  # many ULPs, all below the bound
OOM = "OutOfMemoryError: CUDA out of memory"


@pytest.mark.parametrize(
    "records, outcome, phrase",
    [
        ([record("d", "decode"), record()], parity.PASS, "Every sequence below 0.001"),
        (
            [record("d", "decode", parity_stats=comparison(1, 2e-4, 1)), record()],
            parity.PASS,
            "largest 0.0002",
        ),
        ([record("d", "decode", parity_stats=NEAR_ZERO), record()], parity.PASS, "largest 0.0003"),
        ([record("d", "decode", parity_stats=WITHIN_STATS), record()], parity.HUMAN, "d parity"),
        ([record("d", "decode", zero_b=comparison(1, 2e-4, 1)), record()], parity.HUMAN, "zero_b"),
        ([record("d", "decode", zero_b=NEAR_ZERO), record()], parity.HUMAN, "d zero_b"),
        ([record("d", "decode", parity_stats=BEYOND_STATS), record()], parity.FAIL, "ULPs off"),
        (
            [record("d", "decode", parity_stats=comparison(1, 0.03, 1, argmax=1)), record()],
            parity.FAIL,
            "An argmax differs",
        ),
        ([record("d", "decode"), record(flows=False)], parity.FAIL, "factor gradient"),
        (
            [record("d", "decode"), record(flows=False, error=OOM)],
            parity.FAIL,
            f"its backward failed, on ['t ({OOM})']",
        ),
        ([record("d", "decode", mean=0.005), record()], parity.ERROR, "less than 0.01"),
        (
            [record("d", "decode", parity_stats=BEYOND_STATS, mean=0.0), record()],
            parity.ERROR,
            "less than 0.01",
        ),
        (
            [record("d", "decode", parity_stats=comparison(reference_finite=False)), record()],
            parity.ERROR,
            "Non-finite reference",
        ),
        ([record("d", "decode")], parity.ERROR, "Measured 1 of 2"),
        (
            [record("d", "decode"), {**record(), "gradient": None}],
            parity.ERROR,
            "incomplete: ['t']",
        ),
    ],
    ids=[
        "pass", "pass-below-bound", "pass-near-zero", "human-parity", "human-zero-b",
        "human-zero-b-near-zero", "fail-ulp", "fail-argmax", "fail-gradient",
        "fail-gradient-error", "error-vacuous", "error-before-fail", "error-reference",
        "error-count", "error-gradient-missing",
    ],
)  # fmt: skip
def test_decide_rows(records, outcome, phrase):
    decision = parity.decide(records, THRESHOLDS, expected=2)
    assert decision["outcome"] == outcome
    assert phrase in " ".join(decision["reasons"])


def test_decide_reports_a_failure_first():
    decision = parity.decide([record()], THRESHOLDS, expected=1, failure="OSError: disk")
    assert decision == {
        "outcome": parity.ERROR,
        "reasons": ["The run did not finish: OSError: disk"],
    }


def test_gradient_report_and_relative_difference():
    a, b = torch.ones(2, 3), torch.full((4, 2), 2.0)
    good = {"k": (a, b)}
    assert parity.gradient_report(good)["flows"]
    assert parity.gradient_report(good)["min_max_abs"] == 1.0
    zero = parity.gradient_report({"k": (a, torch.zeros_like(b))})
    assert not zero["flows"] and zero["zero_tensors"] == 1 and zero["finite"]
    nan = parity.gradient_report({"k": (a, b * float("nan"))})
    assert not nan["flows"] and not nan["finite"]
    missing = parity.gradient_report({"k": (a, None)})
    assert not missing["flows"] and missing["missing"] == 1
    assert parity.relative_difference(good, good) == (0.0, "k.lora_A")
    shifted = {"k": (a, b + 0.5)}
    assert parity.relative_difference(shifted, good) == (0.25, "k.lora_B")
    assert parity.relative_difference({"k": (None, b)}, good) == (None, "k.lora_A")


def test_effect_measures_how_far_the_adapter_moved_the_logits():
    base = bf16([[1.0, 2.0], [3.0, 0.0]])
    moved = bf16([[1.5, 2.0], [3.0, 4.0]])
    result = parity.effect(moved, base)
    assert result["mean_abs"] == 1.125 and result["max_abs"] == 4.0
    assert result["changed_fraction"] == 0.5 and result["argmax_changed_fraction"] == 0.5
    assert parity.effect(base, base)["mean_abs"] == 0.0


# The inputs and the sequences


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_the_search_finds_what_the_script_notebook_finds(tmp_path):
    name = "arc-agi_evaluation_challenges.json"
    model = tmp_path / "models" / "sorokin" / "qwen3_4b_grids15_sft139" / "transformers" / "1"
    write_json(model / "config.json", {})
    write_json(tmp_path / "competitions" / "arc-prize-2026-arc-agi-2" / name, {})
    # Decoys: an older copy in a dataset, another model's config, a package tree, a deep file.
    write_json(tmp_path / "datasets" / "someone" / "old-arc" / name, {})
    write_json(tmp_path / "datasets" / "someone" / "old-arc" / "config.json", {})
    write_json(tmp_path / "datasets" / "x" / "site-packages" / "qwen3_4b_grids15_sft139" /
               "config.json", {})  # fmt: skip
    write_json(tmp_path.joinpath(*"abcdefgh") / name, {})
    names = [name, "config.json"]
    mine = parity.find_files(tmp_path, names)
    theirs = nbb.find_inputs(str(tmp_path), names, parity.SEARCH_DEPTH)
    assert {k: [str(p) for p in v] for k, v in mine.items()} == theirs
    assert parity.find_model_dir(tmp_path, "qwen3_4b_grids15_sft139") == model
    found = parity.find_challenges(tmp_path, name, "arc-prize-2026-arc-agi-2")
    assert found == tmp_path / "competitions" / "arc-prize-2026-arc-agi-2" / name
    # Without a competition copy the two remaining copies are ambiguous.
    with pytest.raises(FileNotFoundError, match="2 files"):
        parity.find_challenges(tmp_path, name, "another-competition")
    write_json(tmp_path / "models" / "other" / "qwen3_4b_grids15_sft139" / "config.json", {})
    with pytest.raises(FileNotFoundError, match="2 model folders"):
        parity.find_model_dir(tmp_path, "qwen3_4b_grids15_sft139")
    with pytest.raises(FileNotFoundError, match="0 model folders"):
        parity.find_model_dir(tmp_path, "qwen3_4b_other")


def test_read_tasks_refuses_solutions_and_missing_tasks(tmp_path):
    path = tmp_path / "challenges.json"
    parity.write_synthetic_challenges(path)
    dataset = parity.read_tasks(path, DEV4)
    assert dataset.keys == DEV4 and dataset.replies == {}
    solutions = tmp_path / "arc-agi_evaluation_solutions.json"
    shutil.copyfile(path, solutions)
    with pytest.raises(ValueError, match="solutions file"):
        parity.read_tasks(solutions, DEV4)
    with pytest.raises(ValueError, match="lacks the tasks"):
        parity.read_tasks(path, [*DEV4, "00000000"])


def test_sequences_are_the_notebooks_texts_and_loss_positions(tmp_path):
    path = tmp_path / "challenges.json"
    parity.write_synthetic_challenges(path)
    tasks = json.loads(path.read_text())
    tokenizer = standin_tokenizer()
    formatter = QwenFormatter(tokenizer)
    max_tokens = 2048
    sequences = parity.build_sequences(parity.read_tasks(path, DEV4), DEV4, formatter, max_tokens)
    assert [s.name for s in sequences] == NAMES
    decode_cap = max_tokens - formatter.max_new_tokens()
    for seq in sequences:
        train = tasks[seq.task_id]["train"]
        ids = seq.input_ids.tolist()
        # cut_to_len drops pairs from the start until the text fits: recomputed independently.
        if seq.kind == "decode":
            test = tasks[seq.task_id]["test"][int(seq.name.split("_")[1])]
            query = formatter.fmt_query([test])

            def text(k, train=train, query=query):
                return formatter.fmt_train(train[k:]) + query

            cap = decode_cap
        else:

            def text(k, train=train):
                return formatter.fmt_train(train[k:])

            cap = max_tokens
        kept = next(k for k in range(len(train)) if len(tokenizer.encode(text(k))) <= cap)
        assert seq.dropped == kept and seq.pairs == len(train) - kept
        assert ids == tokenizer.encode(text(kept))
        if seq.kind == "decode":
            assert seq.target is None
            continue
        # The loss positions: the last pair's reply, as the notebook's completion-only collator
        # labels its last assistant segment (from 2 after "assistant" through <|im_end|>).
        reply = tokenizer.encode(formatter.fmt_reply([train[-1]["output"]]))
        assert ids[seq.target.start : seq.target.stop] == reply
        start = max(i for i, t in enumerate(ids) if t == 12) + 2
        stop = max(i for i, t in enumerate(ids) if t == 15) + 1
        assert seq.target == range(start, stop) and stop == len(ids)


def test_a_test_input_above_the_decode_cap_is_refused_instead_of_looping(tmp_path):
    path = tmp_path / "challenges.json"
    parity.write_synthetic_challenges(path)
    formatter = QwenFormatter(standin_tokenizer())
    # Decode cap 1000 - 931 = 69 tokens: the 216-token query of 0934a4d8 alone exceeds it, and
    # the notebook's cut_to_len would drop pairs forever.
    with pytest.raises(ValueError, match="the test input alone exceeds the decode cap 69"):
        parity.build_sequences(parity.read_tasks(path, DEV4), DEV4, formatter, 1000)


# The adapter and the unwrap


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny") / "qwen3_4b_grids15_sft139"
    parity.write_tiny_model(folder, 2048)
    model = AutoModelForCausalLM.from_pretrained(
        folder, torch_dtype=torch.bfloat16, local_files_only=True
    )
    return folder, model.eval().requires_grad_(False)


def test_the_tiny_model_is_the_one_of_test_inject(tiny):
    folder, model = tiny
    config = model.config
    assert (config.hidden_size, config.num_hidden_layers, config.vocab_size) == (64, 2, 16)
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (4, 2, 16)
    assert config.initializer_range == 0.2 and config.intermediate_size == 128
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    assert tokenizer.get_vocab() == standin_tokenizer().get_vocab()


def test_the_adapter_has_the_configured_relative_norm(tiny):
    _, base = tiny
    model = copy.deepcopy(base)
    keys = wrap_targets(model, TARGETS)
    scope = parity.Scope(rank=16, alpha=32, target_modules=TARGETS)
    settings = parity.AdapterSettings(seed=0, relative_norm=0.1)
    factors = parity.make_adapter(model, keys, scope, settings)
    s = rslora_scaling(16, 32)
    for key, (lora_a, lora_b) in factors.items():
        weight = model.get_submodule(f"model.{key}").base_layer.weight.double()
        d_out, d_in = weight.shape
        assert lora_a.shape == (16, d_in) and lora_b.shape == (d_out, 16)
        assert lora_a.dtype == lora_b.dtype == torch.float32
        assert lora_a.abs().max() <= 1 / math.sqrt(d_in)  # PEFT's kaiming bound
        ratio = (s * lora_b.double() @ lora_a.double()).norm() / weight.norm()
        assert abs(float(ratio) - 0.1) < 1e-5, key
    again = parity.make_adapter(model, keys, scope, settings)
    assert all(torch.equal(x, y) for k in keys for x, y in zip(factors[k], again[k], strict=True))
    other = parity.make_adapter(model, keys, scope, parity.AdapterSettings(1, 0.1))
    assert not torch.equal(other[keys[0]][0], factors[keys[0]][0])


def test_unwrap_restores_the_base_for_peft(tiny, tmp_path):
    _, base = tiny
    model = copy.deepcopy(base)
    linears = {n: m for n, m in model.named_modules() if n.rpartition(".")[2] in TARGETS}
    ids = torch.arange(16)[None]
    before = model(input_ids=ids).logits
    keys = wrap_targets(model, TARGETS)
    factors = parity.make_adapter(model, keys, parity.Scope(16, 32, TARGETS),
                                  parity.AdapterSettings(0, 0.1))  # fmt: skip
    with inject(model, factors, rslora_scaling(16, 32)):
        injected = model(input_ids=ids).logits
    assert parity.unwrap_targets(model) == len(keys) == 8
    assert not any(isinstance(m, InjectableLinear) for m in model.modules())
    assert all(model.get_submodule(n) is m for n, m in linears.items())
    assert torch.equal(model(input_ids=ids).logits, before)
    save_adapter(factors, tmp_path, 16, 32, list(TARGETS))
    peft_model = PeftModel.from_pretrained(model, str(tmp_path))
    assert torch.equal(peft_model(input_ids=ids).logits, injected)


# The configs


def test_every_config_value_has_a_comment():
    for name in ("configs/inject_parity.yaml", BUILD_CONFIG):
        lines = (REPO / name).read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if re.match(r"\s*[a-z_]+:", line):
                assert lines[index - 1].strip().startswith("#"), f"{name}: {line!r}"


def test_the_scope_is_hypernet_yaml():
    mine = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["scope"]
    theirs = yaml.safe_load(HYPERNET.read_text(encoding="utf-8"))
    assert mine == {k: theirs[k] for k in ("rank", "alpha", "target_modules")}


def test_the_configs_agree_and_the_build_config_is_accepted():
    path = REPO / BUILD_CONFIG
    build = nbb.parse_config(path.read_bytes(), path)
    assert isinstance(build, nbb.ScriptConfig)
    assert build.kaggle_notebook == NOTEBOOK and build.accelerator == "NvidiaL4"
    assert build.script == "scripts/inject_parity.py"
    assert build.script_config == "configs/inject_parity.yaml"
    assert build.args == ("--commit", "{commit}") and build.notebook_inputs == {}
    gpu = parity.load_config(CONFIG, False, {})
    assert (gpu.competition, gpu.model_slug) == (build.competition, build.model_slug)
    # The script's budget stops it before the notebook's hard stop, which leaves the brief's
    # Timeout (script_kill_s + 300 s) inside the 12 h session.
    assert gpu.max_wall_s + 300 <= build.script_kill_s <= 43200 - 300
    assert NOTEBOOK in ops.briefed_notebooks()


def test_the_gpu_and_smoke_configs_resolve():
    gpu = parity.load_config(CONFIG, False, {})
    assert (gpu.device, gpu.dtype, gpu.attn_implementation) == ("cuda:0", "bfloat16", "auto")
    assert gpu.input_root == "/kaggle/input" and gpu.out_dir == "run_logs"
    assert gpu.model_dir is None and gpu.challenges is None and gpu.subset == "DEV4"
    assert gpu.max_tokens == 8192 and gpu.decision == THRESHOLDS
    assert gpu.adapter == parity.AdapterSettings(seed=0, relative_norm=0.1)
    smoke_cfg = parity.load_config(CONFIG, True, {})
    assert (smoke_cfg.device, smoke_cfg.dtype, smoke_cfg.max_tokens) == ("cpu", "bfloat16", 2048)
    assert smoke_cfg.out_dir == str(REPO / "runs" / "P3.1-smoke" / "run_logs")
    assert smoke_cfg.scope == gpu.scope and smoke_cfg.decision == gpu.decision
    over = parity.load_config(CONFIG, False, {"device": "cpu", "out_dir": "x", "model_dir": None})
    assert (over.device, over.out_dir, over.model_dir) == ("cpu", "x", None)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(extra=1), "unknown keys \\['extra'\\]"),
        (lambda d: d.pop("subset"), "missing keys \\['subset'\\]"),
        (lambda d: d.update(device="cuda"), "device must be cpu or cuda:<index>"),
        (lambda d: d.update(subset="DEV5"), "subset"),
        (lambda d: d.update(challenges_file="arc-agi_evaluation_solutions.json"), "A0"),
        (lambda d: d["scope"].update(target_modules=["q_proj", "lm_head"]), "target_modules"),
        (lambda d: d["decision"].update(max_ulp=-1), "max_ulp"),
        (lambda d: d["adapter"].update(relative_norm=0), "relative_norm"),
        (lambda d: d["smoke"].update(nonsense=1), "smoke.nonsense is not a config key"),
        (lambda d: d["smoke"].update(decision={"bound": 1}), "smoke.decision must map"),
        (lambda d: d["smoke"].update(device="gpu"), "smoke section: device"),
    ],
    ids=[
        "unknown", "missing", "device", "subset", "solutions", "targets", "ulp", "norm",
        "smoke-key", "smoke-section", "smoke-value",
    ],
)  # fmt: skip
def test_bad_configs_are_refused(tmp_path, change, message):
    path = config_with(tmp_path, change)
    with pytest.raises(parity.ConfigError, match=message):
        parity.load_config(path, False, {})


# The build and its script notebook in a simulated Kaggle tree


GIT_ENV = {
    "GIT_AUTHOR_NAME": "tester",
    "GIT_AUTHOR_EMAIL": "tester@example.com",
    "GIT_COMMITTER_NAME": "tester",
    "GIT_COMMITTER_EMAIL": "tester@example.com",
    "GIT_AUTHOR_DATE": "2026-10-02T12:00:00Z",
    "GIT_COMMITTER_DATE": "2026-10-02T12:00:00Z",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
BUILDER = shutil.which("uv") or importlib.util.find_spec("pip")


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, env={**os.environ, **GIT_ENV}, capture_output=True, text=True,
        check=True,
    ).stdout  # fmt: skip


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """configs/script_P3.1-01.yaml built from a git snapshot of the working tree."""
    if not BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    bs = load("build_submission", REPO / "scripts" / "build_submission.py")
    repo = tmp_path_factory.mktemp("repo")
    listed = git(REPO, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    for name in filter(None, listed.split("\0")):
        if (REPO / name).is_file():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, repo / name)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "snapshot")
    folder = bs.build("P3.1-01", repo / BUILD_CONFIG, repo)
    return folder, git(repo, "rev-parse", "HEAD").strip()


def test_the_build_carries_the_script_and_its_config(built):
    folder, commit = built
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    wheel = "arc26-0.0.1-py3-none-any.whl"
    assert files == {
        "MANIFEST.md", "kaggle_script.ipynb", "kernel-metadata.json", f"arc26-code/{wheel}",
        "arc26-code/dataset-metadata.json", "arc26-code/inject_parity.py",
        "arc26-code/inject_parity.yaml",
    }  # fmt: skip
    for name, source in (("inject_parity.py", "scripts"), ("inject_parity.yaml", "configs")):
        assert (folder / "arc26-code" / name).read_bytes() == (REPO / source / name).read_bytes()
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == NOTEBOOK and meta["machine_shape"] == "NvidiaL4"
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False
    assert ops.uses_gpu(meta, None) and ops.accelerator_id(meta, None) == "NvidiaL4"
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, "NvidiaL4")
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder(folder / "arc26-code")
    namespace = {}
    exec(config_cell(folder), namespace)
    assert namespace["ARGS"] == ["--commit", "{commit}"] and namespace["SCRIPT_KILL_S"] == 2400
    assert namespace["BUILD"]["commit"] == commit
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "with the brief's `Timeout` at least 2700 s" in manifest


def config_cell(folder):
    notebook = json.loads((folder / "kaggle_script.ipynb").read_text(encoding="utf-8"))
    (cell,) = [c for c in notebook["cells"] if c.get("id") == "arc26-02-config"]
    return "".join(cell["source"])


def test_the_script_notebook_accepts_the_build_and_runs_the_script(built, tmp_path):
    """The built notebook, wheel, script, and config in a simulated /kaggle tree with four fake
    L4s, the tiny model saved where Kaggle mounts the base, and DEV4-shaped tasks where it mounts
    the competition. A human edits only ARGS in the Config cell: --input-root at the simulated
    mount and --device cpu (no GPU work here). The script then finds its inputs by search, runs
    from the dataset folder against the installed wheel, and writes its logs beside the
    notebook's."""
    folder, commit = built
    nbl = load("notebook_local", REPO / "tests" / "test_notebook_local.py")
    root = tmp_path / "kaggle"
    dataset = root.joinpath("input", *nbl.DATASET_DIRS["2026"])
    shutil.copytree(folder / "arc26-code", dataset)
    model_dir = root.joinpath("input", *nbl.MODEL_DIR)
    parity.write_tiny_model(model_dir, 8192)
    competition = root / "input" / "competitions" / nbl.COMPETITION
    parity.write_synthetic_challenges(competition / "arc-agi_evaluation_challenges.json")
    (root / "working").mkdir(parents=True)
    extra = ["--input-root", str(root / "input"), "--device", "cpu"]
    source = (folder / "kaggle_script.ipynb").read_text(encoding="utf-8")
    old = json.dumps("ARGS = ['--commit', '{commit}']\n")[1:-1]
    assert source.count(old) == 1
    notebook = tmp_path / "edited.ipynb"
    edited = json.dumps(f"ARGS = {['--commit', '{commit}', *extra]!r}\n")[1:-1]
    notebook.write_text(source.replace(old, edited), encoding="utf-8")

    bin_dir, target = tmp_path / "bin", tmp_path / "target"
    for path in (bin_dir, target, tmp_path / "home"):
        path.mkdir()
    for name, text in (
        ("python", nbl.SHIM.replace("@PYTHON@", sys.executable)),
        ("nvidia-smi", nbl.nvidia_smi(4)),
    ):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    driver, record_path, log = tmp_path / "driver.py", tmp_path / "record.json", tmp_path / "log"
    driver.write_text(nbl.DRIVER)
    env = {
        "PATH": str(bin_dir),
        # The unpacked wheel first, as Kaggle's site-packages; then the test interpreter's
        # site-packages as a plain folder for torch and transformers (with -S its .pth files are
        # not read, so the repository's editable install stays invisible).
        "PYTHONPATH": os.pathsep.join([str(target), sysconfig.get_paths()["purelib"]]),
        "ARC26_KAGGLE_ROOT": str(root),
        "ARC26_SHIM_LOG": str(log),
        "ARC26_SHIM_TARGET": str(target),
        "ARC26_SHIM_SUBMISSION": str(root / "working" / "submission.json"),
        "HOME": str(tmp_path / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        **CPU_ENV,
    }
    result = subprocess.run(
        [sys.executable, "-S", str(driver), str(notebook), str(record_path)],
        cwd=root / "working", env=env, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    output = result.stdout + result.stderr
    record = read_json(record_path)
    assert record["error"] is None, output
    assert record["cells"] == nbl.SCRIPT_CODE_CELLS
    assert record["PROBLEMS"] == [] and record["SCRIPT_EXIT"] == 0, output
    calls = [json.loads(line)["argv"] for line in log.read_text().splitlines()]
    scripts = [argv for argv in calls if argv and argv[0].endswith("inject_parity.py")]
    assert scripts == [[
        str(dataset / "inject_parity.py"), "--config", str(dataset / "inject_parity.yaml"),
        "--commit", commit, *extra,
    ]]  # fmt: skip
    run_logs = root / "working" / "run_logs"
    assert {p.name for p in run_logs.iterdir()} == OUTPUTS | NOTEBOOK_LOGS
    stages = read_json(run_logs / "stages.json")
    assert stages["ok"] is True and stages["script_exit"] == 0
    report = read_json(run_logs / "parity.json")
    assert report["decision"]["outcome"] == parity.PASS, report["decision"]
    assert report["smoke"] is False and report["run_id"] == "P3.1-01"
    assert report["provenance"]["commit"] == commit
    assert Path(report["provenance"]["arc26"]).is_relative_to(target)
    assert report["inputs"]["model_dir"] == str(model_dir)
    assert Path(report["inputs"]["challenges"]).parent == competition
    assert [r["name"] for r in report["sequences"]] == NAMES
    assert all(r["parity"]["differing"] == 0 for r in report["sequences"])
    assert "P3.1-01: injected LoRA against PEFT" in output  # the summary reached the session log
