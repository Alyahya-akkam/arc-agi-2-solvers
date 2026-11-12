"""CPU checks of H merge precision, fail-closed validation, and exact puzzle isolation."""

import hashlib
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from arc26 import h_submission as h  # noqa: E402
from arc26.solver import Loaded  # noqa: E402


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.zeros(256, 3, dtype=torch.bfloat16))
        self.lora_B = torch.nn.Parameter(torch.zeros(4, 256, dtype=torch.bfloat16))
        self.merged = False
        self.fan_in_fan_out = False
        self.use_dora = {"default": False}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        attention = torch.nn.Module()
        for target in h.ATTENTION_TARGETS:
            attention.add_module(target, Layer())
        layer = torch.nn.Module()
        layer.add_module("self_attn", attention)
        decoder = torch.nn.Module()
        decoder.add_module("layers", torch.nn.ModuleList([layer]))
        causal = torch.nn.Module()
        causal.add_module("model", decoder)
        base = torch.nn.Module()
        base.add_module("model", causal)
        self.add_module("base_model", base)
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.disabled = False

    @contextmanager
    def disable_adapter(self):
        previous = self.disabled
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = previous


def layers(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, Layer)}


def adapter(model, value=0.010031):
    state = {}
    for name in layers(model):
        state[f"{name}.lora_A.weight"] = torch.full((16, 3), value)
        state[f"{name}.lora_B.weight"] = torch.full((4, 16), value * 1.3)
    return SimpleNamespace(state=state, rank=16, alpha=32, target_modules=h.ATTENTION_TARGETS)


def snapshot(model):
    return {name: module.base_layer.weight.detach().clone()
            for name, module in layers(model).items()}


def assert_original(model, original):
    for name, module in layers(model).items():
        assert torch.equal(module.base_layer.weight, original[name])


def loaded(model):
    tasks = {key: {"train": [{"input": [[color]], "output": [[color]]}],
                   "test": [{"input": [[9]], "output": [[8]]}]}
             for key, color in [("puzzle_a", 1), ("puzzle_b", 2)]}
    return Loaded({}, 8192, None, model, None, {"sentinel": torch.ones(1)}, None, None,
                  100, 1.0, SimpleNamespace(queries=tasks), "/tmp/unused")


def test_merge_rounds_factors_before_fp32_product_and_preserves_rank256():
    model = Model()
    emitted = adapter(model)
    before = snapshot(model)
    live = {name: (layer.lora_A.clone(), layer.lora_B.clone())
            for name, layer in layers(model).items()}
    merge = h.HBaseMerge(model, emitted)
    assert all(t.device.type == "cpu" for t in merge.copies)
    info = merge.add()
    assert info["scale"] == 8.0 and info["delta_norm"] > 0
    assert info["changed_elements"] > 0 and info["effective_delta_norm"] > 0
    effective = torch.cat([(layer.base_layer.weight.float() - before[name].float()).flatten()
                           for name, layer in layers(model).items()])
    assert info["changed_elements"] == int(torch.count_nonzero(effective))
    assert info["effective_delta_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(effective.double()))
    )
    for name, layer in layers(model).items():
        a = emitted.state[f"{name}.lora_A.weight"].bfloat16().float()
        b = emitted.state[f"{name}.lora_B.weight"].bfloat16().float()
        expected = (before[name].float() + 8 * (b @ a)).bfloat16()
        assert torch.equal(layer.base_layer.weight, expected)
        assert layer.lora_A.shape[0] == layer.lora_B.shape[1] == 256
        assert torch.equal(layer.lora_A, live[name][0])
        assert torch.equal(layer.lora_B, live[name][1])
    merge.restore(model)
    assert_original(model, before)
    with pytest.raises(RuntimeError, match="once"):
        merge.add()


@pytest.mark.parametrize("broken", ["rank", "alpha", "missing", "extra", "shape", "nan",
                                   "bf16_overflow", "targets", "trainable", "merged"])
def test_bad_h_factors_never_modify_base(broken):
    model = Model()
    before = snapshot(model)
    emitted = adapter(model)
    first = next(iter(emitted.state))
    if broken == "rank":
        emitted.rank = 256
    elif broken == "alpha":
        emitted.alpha = float("nan")
    elif broken == "missing":
        emitted.state.pop(first)
    elif broken == "extra":
        emitted.state["unexpected.lora_A.weight"] = torch.zeros(16, 3)
    elif broken == "shape":
        emitted.state[first] = torch.zeros(15, 3)
    elif broken == "nan":
        emitted.state[first][0, 0] = float("nan")
    elif broken == "bf16_overflow":
        emitted.state[first].fill_(3.4e38)
    elif broken == "targets":
        emitted.target_modules = ("gate_proj", *h.ATTENTION_TARGETS[1:])
    elif broken == "trainable":
        next(iter(layers(model).values())).base_layer.weight.requires_grad_(True)
    elif broken == "merged":
        next(iter(layers(model).values())).merged = True
    with pytest.raises((ValueError, TypeError)):
        h.HBaseMerge(model, emitted)
    assert_original(model, before)


def test_successive_puzzles_emit_on_original_base_with_disabled_adapters():
    model = Model()
    before = snapshot(model)
    state = loaded(model)
    seen = []

    def emit(demos):
        assert_original(model, before)
        assert model.disabled
        assert demos.test_inputs == () and demos.held_out is None
        assert len(demos.pairs) == 1
        seen.append(demos.puzzle_id)
        return adapter(model, 0.1 if demos.puzzle_id == "puzzle_a" else 0.2)

    def fit(**kw):
        assert kw["default_weights"] is state.default_weights
        assert not model.disabled
        assert kw["train_args"] == {} and kw["max_seq_length"] == 8192
        for name, layer in layers(model).items():
            assert not torch.equal(layer.base_layer.weight, before[name])
            with torch.no_grad():
                layer.lora_B.fill_(7)  # Simulate trained live residual, disabled for next emit.
        return model, "original-puzzle-dataset"

    def decode(**kw):
        assert kw["puzzle_ds"] == "original-puzzle-dataset"
        assert kw["end_time"] == 1234 and kw["rank"] == 0
        assert kw["model"] is model

    digests = []
    for key in ("puzzle_a", "puzzle_b", "puzzle_a"):
        record = {}
        state = h.solve_one(state, SimpleNamespace(emit=emit), key, 0, 1234, fit, decode, record)
        assert record["restored"] and record["decoded"] and record["h_used"]
        digests.append(record["h_digest"])
        assert_original(model, before)
    assert digests[0] == digests[2] != digests[1]
    assert seen == ["puzzle_a", "puzzle_b", "puzzle_a"]


@pytest.mark.parametrize("phase", ["emit", "merge", "fit", "decode"])
def test_every_failure_restores_base_before_any_other_puzzle(phase):
    model = Model()
    before = snapshot(model)

    def emit(demos):
        if phase == "emit":
            raise RuntimeError("emit failure")
        value = adapter(model)
        if phase == "merge":
            name = list(layers(model))[1]
            value.state[f"{name}.lora_A.weight"].fill_(1e30)
            value.state[f"{name}.lora_B.weight"].fill_(1e30)
        return value

    def fit(**kw):
        if phase == "fit":
            raise RuntimeError("fit failure")
        return model, None

    def decode(**kw):
        if phase == "decode":
            raise RuntimeError("decode failure")

    record = {}
    with pytest.raises((ValueError, RuntimeError)):
        h.solve_one(loaded(model), SimpleNamespace(emit=emit), "puzzle_a", 0, 1234,
                    fit, decode, record)
    assert_original(model, before)
    assert not model.disabled
    if phase != "emit":
        assert record["restored"]


def test_checkpoint_mismatch_fails_before_loading(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not a torch checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h.checkpoint_digest(path, digest) == digest
    with pytest.raises(ValueError, match="mismatch"):
        h.checkpoint_digest(path, "0" * 64)
    with pytest.raises(ValueError, match="hexadecimal"):
        h.checkpoint_digest(path, "invalid")


def test_reference_config_rejects_rank_scope_and_refinement_changes():
    cfg = {"lora_r": 256, "lora_targets": list(h.REFERENCE_TARGETS), "workers": 2,
           "hyper_checkpoint": "H.pt", "hyper_sha256": "0" * 64, "hyper_log_dir": "/tmp/h"}
    h.validate_config(cfg)
    for changes in ({"lora_r": 16}, {"lora_targets": list(h.ATTENTION_TARGETS)},
                    {"holdout": {}}, {"adapters": {}}, {"workers": 0}):
        with pytest.raises(ValueError):
            h.validate_config({**cfg, **changes})


def test_zero_h_is_rejected_and_restored():
    model = Model()
    before = snapshot(model)
    record = {}
    with pytest.raises(ValueError, match="zero H"):
        h.solve_one(loaded(model), SimpleNamespace(emit=lambda demos: adapter(model, 0.0)),
                    "puzzle_a", 0, 1234, None, None, record)
    assert record["restored"]
    assert_original(model, before)


def test_nonzero_h_that_rounds_away_is_rejected_and_restored():
    model = Model()
    with torch.no_grad():
        for layer in layers(model).values():
            layer.base_layer.weight.fill_(1.0)
    before = snapshot(model)
    emitted = adapter(model, 0.0001)
    first = next(iter(layers(model)))
    a = emitted.state[f"{first}.lora_A.weight"].bfloat16().float()
    b = emitted.state[f"{first}.lora_B.weight"].bfloat16().float()
    assert bool(((b @ a) * 8 > 0).all())
    record = {}
    with pytest.raises(ValueError, match="changes no bfloat16 base elements"):
        h.solve_one(loaded(model), SimpleNamespace(emit=lambda demos: emitted),
                    "puzzle_a", 0, 1234, None, None, record)
    assert record["restored"] and not record.get("h_used", False)
    assert_original(model, before)


@pytest.fixture
def checkpoint_payload():
    from arc26.hyper.hypernet import BaseShapes, Hypernet, HypernetConfig

    cfg = {"embed_dim": 2560, "latent": 4, "head_in": 1, "dropout": 0.05,
           "rank": 16, "alpha": 32, "target_modules": list(h.ATTENTION_TARGETS), "seed": 0}
    with torch.device("meta"):
        net = Hypernet(HypernetConfig(**cfg), BaseShapes(36, 2560, 9728, 32, 8, 128))
    state = {name: torch.ones(()).expand(tensor.shape) for name, tensor in net.state_dict().items()}
    return {"schema": 1, "kind": "best", "step": 4260, "hypernet": state,
            "config": {"model_id": h.MODEL_ID, "hypernet": cfg,
                       "encoder": {"layer": 24, "pool": "output", "n_perm": 1,
                                   "view_seed": 2, "batch_tokens": 16384, "chunk_pairs": 256}}}


def write_checkpoint(tmp_path, payload):
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_cpu_preflight_checks_full_h_without_base_or_cuda(
    tmp_path, checkpoint_payload, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU preflight must not initialize CUDA or load the base")

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr("arc26.solver.load", forbidden)
    path, digest = write_checkpoint(tmp_path, checkpoint_payload)
    report_path = tmp_path / "report.json"
    assert h.main(["--check-only", "--checkpoint", str(path), "--sha256", digest,
                   "--report", str(report_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(report_path.read_text())
    assert report["layers"] == 36 and report["rank"] == 16 and report["step"] == 4260
    assert report["device"] == "cpu" and report["parameter_count"] > 0


@pytest.mark.parametrize("bad", ["hash", "model", "step", "rank", "layers", "finite", "keys"])
def test_cpu_preflight_fails_closed(tmp_path, checkpoint_payload, bad, capsys):
    if bad == "model":
        checkpoint_payload["config"]["model_id"] = "other"
    elif bad == "step":
        checkpoint_payload["step"] = 0
    elif bad == "rank":
        checkpoint_payload["config"]["hypernet"]["rank"] = 256
    elif bad == "layers":
        checkpoint_payload["hypernet"]["layer_embedding.weight"] = torch.ones(35, 1)
    elif bad == "finite":
        checkpoint_payload["hypernet"]["layer_embedding.weight"] = torch.full((36, 1), float("nan"))
    elif bad == "keys":
        checkpoint_payload["hypernet"].pop("layer_embedding.weight")
    path, digest = write_checkpoint(tmp_path, checkpoint_payload)
    assert h.main(["--check-only", "--checkpoint", str(path),
                   "--sha256", "0" * 64 if bad == "hash" else digest]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


@pytest.mark.parametrize("rerun", [False, True])
def test_main_queues_sorted_all_hidden_on_rerun_and_only_dev_subset_otherwise(
    tmp_path, monkeypatch, rerun
):
    import torch.multiprocessing as mp

    (tmp_path / "arc-agi_evaluation_challenges.json").write_text('{"dev":{},"excluded":{}}')
    (tmp_path / "arc-agi_test_challenges.json").write_text('{"z_hidden":{},"a_hidden":{}}')
    # A solutions file must never be opened by this entry, including the development path.
    (tmp_path / "arc-agi_evaluation_solutions.json").write_text("not valid JSON")
    checkpoint = tmp_path / "hash-only.pt"
    checkpoint.write_bytes(b"pinned checkpoint bytes; worker creation is intercepted")
    cfg = {"lora_r": 256, "lora_targets": list(h.REFERENCE_TARGETS), "workers": 2,
           "hyper_checkpoint": str(checkpoint),
           "hyper_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           "hyper_log_dir": str(tmp_path / "logs"), "marker_dir": str(tmp_path / "markers"),
           "data_dir": str(tmp_path), "subset": ["dev"]}
    queued = []
    queue = SimpleNamespace(put=queued.append)

    @contextmanager
    def manager():
        yield SimpleNamespace(Queue=lambda: queue)

    calls = []
    monkeypatch.setattr(mp, "Manager", manager)
    monkeypatch.setattr(mp, "spawn", lambda fn, **kw: calls.append((fn, kw)))
    if rerun:
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    else:
        monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    assert h.main(["--config", json.dumps(cfg), "--end-time", str(time.time() + 60)]) == 0
    assert queued == (["a_hidden", "z_hidden"] if rerun else ["dev"]) + [None, None]
    assert calls[0][0] is h.local_worker and calls[0][1]["nprocs"] == 2
