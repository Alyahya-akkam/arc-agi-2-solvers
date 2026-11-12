"""P3.9b amendment 1: arm (g), ``hyper_merge_fit``, H merged into the base, then a fresh fit.

Arm (g) merges H's delta $s\\,B A$ into every targeted base weight (after an exact copy), runs arm
(c)'s ``arc26.ttt.fit`` from the initial weights ($B = 0$), decodes, and puts the copy back bit for
bit before the next arm, also when the arm raises. Checked on the CPU with ``tests/test_arms.py``'s
tiny stack (a random Qwen3 wrapped by PEFT, a tiny H, the smoke path's stand-ins for unsloth and
datasets, the decode recorded instead of run): the merged forward equals the base plus H's LoRA
forward; the delta is PEFT's in float32 and the sum is rounded once; the fit starts from the
initial weights with ``sgd``'s arguments; the decode reads the merged base and the next arm the
original one; the restore after a decode, a fit, or a merge that raised; a restore that fails
stops the worker; a restore by subtraction would not be exact in bf16, the copy is; the other
arms are unchanged. Skipped without torch, transformers, peft, or safetensors.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("_t_arms_for_merge", REPO / "tests" / "test_arms.py")
T = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_t_arms_for_merge", T)
_spec.loader.exec_module(T)

torch = T.torch
am, report, records = T.am, T.report, T.records
stack, runner_env = T.stack, T.runner_env  # the fixtures, shared
MERGE_ARMS = [
    {"name": "sgd", "source": "sgd"},
    {"name": "h-merge-fit", "source": "hyper_merge_fit"},
    {"name": "base", "source": "base"},
]


def base_weights(model):
    """Every LoRA layer's base weight, cloned, by module name."""
    from peft.tuners.lora import LoraLayer

    return {
        name: module.get_base_layer().weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def demos_of(key):
    task = T.PUZZLES[key]
    return T.DemoSet(
        key, tuple(T.Pair(p["input"], p["output"]) for p in task["train"]), None,
        tuple(t["input"] for t in task["test"]),
    )  # fmt: skip


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def recording_decode(env, monkeypatch, seen):
    """Wrap the runner's recorded decode: keep each arm's base weights and adapter digest."""
    decode = sys.modules["arc26.decode"]
    inner = decode.decode_puzzle

    def wrapped(**kwargs):
        seen.append(
            {
                "arm": Path(kwargs["dir_outputs"]).name,
                "base": base_weights(kwargs["model"]),
                "adapter": T.digest(kwargs["model"]),
            }
        )
        return inner(**kwargs)

    monkeypatch.setattr(decode, "decode_puzzle", wrapped)


def test_hyper_merge_fit_is_an_h_source_that_trains_without_steps():
    assert am.HYPER_MERGE_FIT == "hyper_merge_fit"
    assert am.HYPER_MERGE_FIT in am.SOURCES and am.HYPER_MERGE_FIT in am.H_SOURCES
    assert am.HYPER_MERGE_FIT in am.TRAINED_SOURCES and am.HYPER_MERGE_FIT not in am.POLISH_SOURCES
    assert am.ArmSpec("g", am.HYPER_MERGE_FIT).steps is None
    with pytest.raises(ValueError, match="steps apply to a polish only"):
        am.ArmSpec("g", am.HYPER_MERGE_FIT, steps=128)
    # The earlier sources keep their order: P3.7's configs and records name them.
    assert am.SOURCES[:6] == ("base", "hyper", "sgd", "hyper_polish", "polish", "hyper_fit")


def test_the_merged_forward_equals_the_base_plus_hs_lora_forward(runner_env):
    env = runner_env
    model, loaded = env["loaded"].model, env["loaded"]
    before = base_weights(model)
    ids = torch.tensor([[1, 5, 9, 3, 7, 2, 11, 4]])
    model = env["hyper"].adapt(demos_of("aaaa0001"), model, 0.0)
    with_lora = logits(model, ids)
    merge = am.BaseMerge(model)
    stats = merge.add()
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, loaded.default_weights.copy(), adapter_name="default")
    merged = logits(model, ids)
    # Float32 on the CPU: one rounding of W + s B A per weight, so equal to float32's precision.
    assert torch.allclose(merged, with_lora, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(merged, logits_base(model, before, ids), atol=1e-3)
    n_layers = len(before)
    assert stats["layers"] == n_layers == 4 * model.config.num_hidden_layers
    assert stats["copy_bytes"] == sum(t.numel() * t.element_size() for t in before.values())
    assert stats["dtype"] == "float32" and stats["device"] == "cpu"
    assert stats["delta_norm"] > 0 and stats["rounding_rel"] < 1e-5
    merge.restore()
    assert same(base_weights(model), before)


def logits_base(model, weights, ids):
    """The model's logits on the given base weights, its current adapter kept."""
    from peft.tuners.lora import LoraLayer

    layers = {n: m for n, m in model.named_modules() if isinstance(m, LoraLayer)}
    saved = {n: m.get_base_layer().weight.detach().clone() for n, m in layers.items()}
    with torch.no_grad():
        for n, m in layers.items():
            m.get_base_layer().weight.copy_(weights[n])
        out = logits(model, ids)
        for n, m in layers.items():
            m.get_base_layer().weight.copy_(saved[n])
    return out


def bf16_lora(seed=0):
    """A bf16 PEFT LoRA on two Linear layers, factors random (rsLoRA, rank 4, alpha 16)."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    base = torch.nn.Sequential(torch.nn.Linear(64, 48), torch.nn.Linear(48, 32))
    config = LoraConfig(r=4, lora_alpha=16, target_modules=["0", "1"], use_rslora=True)
    model = get_peft_model(base, config).to(torch.bfloat16)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn_like(param) * 0.3)
    return model


def test_the_delta_is_peft_s_in_float32_and_the_sum_is_rounded_once():
    from peft.tuners.lora import LoraLayer

    model = bf16_lora()
    layers = {n: m for n, m in model.named_modules() if isinstance(m, LoraLayer)}
    before = {n: m.get_base_layer().weight.detach().clone() for n, m in layers.items()}
    expected = {}
    for name, module in layers.items():
        assert module.scaling["default"] == 16 / 4 ** 0.5  # rsLoRA, alpha / sqrt(r)
        a = module.lora_A["default"].weight.float()
        b = module.lora_B["default"].weight.float()
        delta = module.scaling["default"] * (b @ a)
        # PEFT's own delta, computed in float32 on the CPU for bf16 weights.
        assert torch.allclose(module.get_delta_weight("default").float(), delta, 1e-2, 2e-2)
        expected[name] = (before[name].float() + delta).to(torch.bfloat16)
    stats = am.BaseMerge(model).add()
    for name, module in layers.items():
        assert torch.equal(module.get_base_layer().weight, expected[name]), name
    assert stats["dtype"] == "bfloat16" and stats["layers"] == 2
    assert 0 < stats["rounding_rel"] < 0.05  # bf16 rounds W + delta: measured, not zero


def test_a_restore_by_subtraction_would_drift_in_bf16_the_copy_does_not():
    from peft.tuners.lora import LoraLayer

    model = bf16_lora(seed=1)
    layers = {n: m for n, m in model.named_modules() if isinstance(m, LoraLayer)}
    before = {n: m.get_base_layer().weight.detach().clone() for n, m in layers.items()}
    merge = am.BaseMerge(model)
    merge.add()
    drift = 0
    for name, module in layers.items():
        a = module.lora_A["default"].weight.float()
        b = module.lora_B["default"].weight.float()
        delta = module.scaling["default"] * (b @ a)
        undone = (module.get_base_layer().weight.float() - delta).to(torch.bfloat16)
        drift += int((undone != before[name]).sum())
    assert drift > 0  # what subtracting would leave
    merge.restore()
    assert all(torch.equal(m.get_base_layer().weight, before[n]) for n, m in layers.items())


def test_the_arm_fits_from_the_initial_weights_on_the_merged_base_and_restores_it(
    runner_env, monkeypatch
):
    env = runner_env
    ttt = sys.modules["arc26.ttt"]
    original, calls = ttt.fit, []

    def recording_fit(**kwargs):
        calls.append({**kwargs, "start": T.adapter_digest(dict(kwargs["default_weights"]))})
        return original(**kwargs)

    monkeypatch.setattr(ttt, "fit", recording_fit)
    seen = []
    recording_decode(env, monkeypatch, seen)
    model = env["loaded"].model
    before = base_weights(model)
    default = T.adapter_digest(env["loaded"].default_weights)
    runner, arms = T.make_runner(env, arms=MERGE_ARMS, train_log=True)
    model = runner.solve(model, "aaaa0001", end_time=1e9)
    sgd, fit = calls
    # (g) is sgd's call exactly: the same objects, values, and start (the initial weights).
    for name in ("arc_test_set", "formatter", "tokenizer", "collator"):
        assert fit[name] is sgd[name], name
    for name in ("key", "max_seq_length", "rank", "train_args", "start"):
        assert fit[name] == sgd[name], name
    assert fit["start"] == default
    # The trainer starts both from B = 0, the same views and steps; (g) on the merged base.
    c, g = env["trainings"]
    assert c["before"] == g["before"] == default and c["views"] == g["views"] > 3
    rows = {r["arm"]: r for r in records(arms)}
    assert [r["status"] for r in rows.values()] == ["done"] * 3
    # The decode of (g) reads the merged base and its fitted fresh LoRA; the next arm the base.
    decoded = {s["arm"]: s for s in seen}
    assert same(decoded["sgd"]["base"], before) and same(decoded["base"]["base"], before)
    assert not any(
        torch.equal(decoded["h-merge-fit"]["base"][k], before[k]) for k in before
    )
    assert decoded["h-merge-fit"]["adapter"] == rows["h-merge-fit"]["fingerprint"] != default
    assert rows["h-merge-fit"]["fingerprint"] != rows["sgd"]["fingerprint"]
    assert same(base_weights(model), before)
    # The record: H's adapter (as arm (b) loads it), the merge's numbers, the restore, the log.
    g_row = rows["h-merge-fit"]
    assert g_row["h_fingerprint"] == env["hyper"].emit(demos_of("aaaa0001")).digest()
    assert g_row["merge"]["layers"] == len(before) and g_row["restored"] is True
    assert len(g_row["train_log"]) == g["views"] and "restored" not in rows["sgd"]
    assert runner.merged is None


def test_the_arm_records_its_merge_ops_and_hs(runner_env):
    env = runner_env
    runner, arms = T.make_runner(env, arms=MERGE_ARMS)
    runner.solve(env["loaded"].model, "aaaa0002", end_time=1e9)
    lines = (Path(arms.log_dir) / "h-merge-fit" / "timing" / "timing-rank0.jsonl").read_text()
    ops = {T.json.loads(line)["op"] for line in lines.splitlines()}
    assert {"encode", "h_forward", "swap", "merge", "train", "unmerge", "ce", "puzzle"} <= ops
    sgd = (Path(arms.log_dir) / "sgd" / "timing" / "timing-rank0.jsonl").read_text()
    assert "merge" not in {T.json.loads(line)["op"] for line in sgd.splitlines()}


@pytest.mark.parametrize("where", ["decode", "fit", "merge"])
def test_the_base_weights_come_back_when_the_arm_raises(runner_env, monkeypatch, where):
    env = runner_env
    model = env["loaded"].model
    before = base_weights(model)
    if where == "decode":
        env["plan"][("aaaa0001", "h-merge-fit")] = "raise"
    elif where == "fit":

        def failing_fit(**kwargs):
            raise MemoryError("CUDA out of memory (stand-in fit)")

        monkeypatch.setattr(sys.modules["arc26.ttt"], "fit", failing_fit)
    else:

        def half(self):
            # The first layer written, then a failure: a merge that raised midway.
            with torch.no_grad():
                self.params[0].add_(1.0)
            raise MemoryError("CUDA out of memory (stand-in merge)")

        monkeypatch.setattr(am.BaseMerge, "add", half)
    seen = []
    recording_decode(env, monkeypatch, seen)
    runner, arms = T.make_runner(env, arms=MERGE_ARMS)
    runner.solve(model, "aaaa0001", end_time=1e9)
    rows = {r["arm"]: r for r in records(arms)}
    assert rows["h-merge-fit"]["status"] == "raised" and rows["h-merge-fit"]["restored"] is True
    assert rows["base"]["status"] == "done"
    assert same(base_weights(model), before)
    assert same({s["arm"]: s for s in seen}["base"]["base"], before)
    assert runner.merged is None


def test_a_restore_that_fails_stops_the_worker(runner_env, monkeypatch):
    env = runner_env

    def broken(self):
        raise RuntimeError("stand-in: the copy could not be written back")

    monkeypatch.setattr(am.BaseMerge, "restore", broken)
    runner, arms = T.make_runner(env, arms=MERGE_ARMS)
    with pytest.raises(am.ArmStop, match="base weights were not restored"):
        runner.solve(env["loaded"].model, "aaaa0001", end_time=1e9)
    rows = records(arms)
    # The arm's record is written, the next arm never runs on the merged base.
    assert [r["arm"] for r in rows] == ["sgd", "h-merge-fit"]
    assert rows[1]["status"] == "raised" and "could not be written back" in rows[1]["error"]
    marker = Path(arms.log_dir) / "stop-rank0.json"
    assert "not restored" in T.json.loads(marker.read_text())["reason"]


def test_the_restore_refuses_a_base_weight_replaced_after_the_merge():
    from peft.tuners.lora import LoraLayer

    model = bf16_lora(seed=2)
    merge = am.BaseMerge(model)
    merge.add()
    layer = next(m for _, m in model.named_modules() if isinstance(m, LoraLayer))
    layer.get_base_layer().weight = torch.nn.Parameter(layer.get_base_layer().weight.clone())
    with pytest.raises(RuntimeError, match="replaced after the merge"):
        merge.restore()


def test_the_merge_refuses_a_layer_it_cannot_merge_into():
    from peft import LoraConfig, get_peft_model

    base = torch.nn.Sequential(torch.nn.Linear(16, 8))
    dora = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["0"], use_dora=True))
    with pytest.raises(ValueError, match="not a plain unquantized Linear LoRA"):
        am.BaseMerge(dora)
    with pytest.raises(ValueError, match="no LoRA layer"):
        am.BaseMerge(torch.nn.Sequential(torch.nn.Linear(4, 4)))


def test_the_other_arms_are_unchanged_by_arm_g(runner_env):
    """(a), (b), (c), (f) give the same fingerprints and CEs with (g) after them as without."""
    env = runner_env
    four = [
        {"name": "base", "source": "base"}, {"name": "h", "source": "hyper"},
        {"name": "sgd", "source": "sgd"}, {"name": "h-fit", "source": "hyper_fit"},
    ]  # fmt: skip
    runner, arms = T.make_runner(env, arms=four)
    model = runner.solve(env["loaded"].model, "aaaa0001", end_time=1e9)
    model = runner.solve(model, "aaaa0002", end_time=1e9)
    alone = {(r["puzzle_id"], r["arm"]): r for r in records(arms)}
    Path(arms.log_dir, "records-rank0.jsonl").unlink()
    runner, arms = T.make_runner(env, arms=[*four, {"name": "g", "source": "hyper_merge_fit"}])
    model = runner.solve(model, "aaaa0001", end_time=1e9)
    model = runner.solve(model, "aaaa0002", end_time=1e9)
    paired = {(r["puzzle_id"], r["arm"]): r for r in records(arms)}
    for key, row in alone.items():
        assert paired[key]["fingerprint"] == row["fingerprint"], key
        assert paired[key]["ce"] == row["ce"], key
