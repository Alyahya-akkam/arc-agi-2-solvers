"""P3.3b: H's set aggregator, the warm start (``--init-from``), and the mean path unchanged.

CPU only: tiny H shapes, the tiny random Qwen3 and the fixture episodes of tests/test_train.py,
and Qwen3-4B's shapes on the meta device for the parameter count. Skipped without torch,
transformers, peft, or safetensors (CI's main job). Run with CUDA_VISIBLE_DEVICES="" (BRIEF A0).
"""

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

import yaml  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.hypernet import (  # noqa: E402
    AggregatorConfig,
    BaseShapes,
    Hypernet,
    HypernetConfig,
    pad_pair_sets,
)
from arc26.hyper.inject import inject  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO / "configs" / "hypernet.yaml"
TOY = REPO / "configs" / "train_hypernet_toy.yaml"
CONFIG = HypernetConfig.from_yaml(CONFIG_FILE)
QWEN3_4B = Qwen3Config(
    vocab_size=151936,
    hidden_size=2560,
    intermediate_size=9728,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    tie_word_embeddings=True,
)
BASE = BaseShapes(
    n_layers=2, hidden_size=64, intermediate_size=128, n_heads=4, n_kv_heads=2, head_dim=16
)
MEAN = dataclasses.replace(CONFIG, embed_dim=24, latent=32, head_in=16)
SET = dataclasses.replace(
    MEAN, aggregator=AggregatorConfig(kind="set", width=16, blocks=2, heads=2)
)
# sha256 of the tiny mean H's state dict at seed 0, computed with arc26/hyper/hypernet.py at
# 0b4ebc2 (before P3.3b): the mean path draws exactly what it drew.
MEAN_DIGEST = "ec30b9d6fe32b57cc9b8578bd03d69f391485590ed6f5bbe5f78aa44212d7c1a"


def digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].contiguous().numpy().tobytes())
    return h.hexdigest()


def randomize(module, seed, scale=0.1, prefix=""):
    """Non-zero values everywhere under ``prefix`` (a trained H, and W_o != 0)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.startswith(prefix):
                param.copy_(torch.randn(param.shape, generator=generator) * scale)


def warm(mean_h):
    """A set H loaded from ``mean_h`` by name, as --init-from loads it."""
    h = Hypernet(SET, BASE)
    missing, unexpected = h.load_state_dict(mean_h.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("aggregator.") for k in missing)
    return h


def same_factors(a, b):
    return all(
        torch.equal(x[key][0], y[key][0]) and torch.equal(x[key][1], y[key][1])
        for x, y in zip(a, b, strict=True)
        for key in x
    )


def close_factors(a, b, atol=1e-6):
    return all(
        torch.allclose(x[key][i], y[key][i], atol=atol, rtol=0)
        for x, y in zip(a, b, strict=True)
        for key in x
        for i in (0, 1)
    )


def max_diff(a, b):
    return max(
        float((x[key][i] - y[key][i]).detach().abs().max())
        for x, y in zip(a, b, strict=True)
        for key in x
        for i in (0, 1)
    )


def inputs(batch=3, n=4, seed=7):
    generator = torch.Generator().manual_seed(seed)
    pairs = torch.randn(batch, n, MEAN.embed_dim, generator=generator)
    return pairs.mean(dim=1), pairs


# --- configuration and size ----------------------------------------------------------------------


def test_parameter_count_matches_the_scoping_note():
    with torch.device("meta"):
        shapes = BaseShapes.from_config(QWEN3_4B)
        counts = {}
        for blocks in (2, 3):
            agg = AggregatorConfig(kind="set", blocks=blocks)
            h = Hypernet(dataclasses.replace(CONFIG, aggregator=agg), shapes)
            counts[blocks] = sum(p.numel() for p in h.parameters())
        mean = sum(p.numel() for p in Hypernet(CONFIG, shapes).parameters())
    assert mean == 134_958_208
    assert counts[2] - mean == 8_877_824 and counts[2] == 143_836_032
    assert counts[3] == 146_988_416


def test_config_file_keeps_mean_as_the_default():
    assert CONFIG.aggregator == AggregatorConfig() and CONFIG.aggregator.kind == "mean"
    data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    assert data["aggregator"] == dataclasses.asdict(AggregatorConfig())


def test_a_config_without_the_block_is_the_mean_h(tmp_path):
    data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    data.pop("aggregator")
    path = tmp_path / "hypernet.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert HypernetConfig.from_yaml(path) == CONFIG
    # The checkpoint form leaves the default block out, and reads back.
    plain = CONFIG.to_dict()
    assert "aggregator" not in plain and HypernetConfig(**plain) == CONFIG
    as_set = dataclasses.replace(CONFIG, aggregator={"kind": "set"})
    assert as_set.to_dict()["aggregator"] == {"kind": "set", "width": 512, "blocks": 2, "heads": 8}
    assert HypernetConfig(**as_set.to_dict()) == as_set


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({"kind": "lm"}, "kind"),
        ({"kind": "set", "width": 0}, "width"),
        ({"kind": "set", "blocks": 1.5}, "blocks"),
        ({"kind": "set", "width": 30, "heads": 4}, "divide"),
        ({"kind": "set", "depth": 2}, "unknown"),
    ],
)
def test_aggregator_rejects_bad_values(block, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(CONFIG, aggregator=block)


def test_pad_pair_sets():
    a, b = torch.ones(2, 3), torch.full((1, 3), 2.0)
    pairs, mask = pad_pair_sets([a, b])
    assert pairs.shape == (2, 2, 3) and mask.tolist() == [[True, True], [True, False]]
    assert torch.equal(pairs[1, 1], torch.zeros(3)) and torch.equal(pairs[0], a)
    for bad in ([], [torch.ones(0, 3)], [torch.ones(2, 3), torch.ones(2, 4)]):
        with pytest.raises(ValueError):
            pad_pair_sets(bad)


# --- the mean path, unchanged -----------------------------------------------------------------


def test_the_mean_h_draws_what_it_drew_before_p3_3b():
    h = Hypernet(MEAN, BASE)
    assert h.aggregator is None and not any("aggregator" in k for k in h.state_dict())
    assert digest(h.state_dict()) == MEAN_DIGEST


def test_the_mean_h_refuses_pairs():
    z, pairs = inputs()
    with pytest.raises(ValueError, match="mean H"):
        Hypernet(MEAN, BASE)(z, pairs)


# --- the set H ---------------------------------------------------------------------------------


def test_from_scratch_the_set_h_is_the_mean_h_plus_a_zero_branch():
    """Same seed: every shared parameter equal; W_o = 0; so the factors equal the mean H's, and
    B = 0 (base-exact at initialization, H's existing property)."""
    mean_h, set_h = Hypernet(MEAN, BASE), Hypernet(SET, BASE)
    mean_state, set_state = mean_h.state_dict(), set_h.state_dict()
    assert all(torch.equal(mean_state[k], set_state[k]) for k in mean_state)
    assert set(set_state) - set(mean_state) == {k for k in set_state if k.startswith("aggregator.")}
    out = set_h.aggregator.output
    assert not out.weight.any() and not out.bias.any()
    assert set_h.aggregator.queries.abs().sum() > 0
    z, pairs = inputs()
    for mode in (False, True):
        mean_h.train(mode), set_h.train(mode)
        torch.manual_seed(0)
        expected = mean_h(z)
        torch.manual_seed(0)
        got = set_h(z, pairs)
        if not mode:
            assert same_factors(expected, got)
        assert all(not b.any() for factors in got for _, b in factors.values())


def test_the_warm_start_is_bitwise_the_loaded_h_in_eval_mode():
    trained = Hypernet(MEAN, BASE)
    randomize(trained, seed=1)
    trained.eval()
    h = warm(trained).eval()
    z, pairs = inputs()
    _, mask = pad_pair_sets([pairs[0], pairs[1, :2], pairs[2, :1]])
    assert same_factors(trained(z), h(z, pairs))
    assert same_factors(trained(z), h(z, pairs, mask))


def test_the_set_h_is_permutation_invariant_over_pairs():
    h = Hypernet(SET, BASE)
    randomize(h, seed=2)
    h.eval()
    z, pairs = inputs()
    order = torch.tensor([2, 0, 3, 1])
    permuted = max_diff(h(z, pairs), h(z, pairs[:, order]))
    assert permuted < 1e-6
    # Not vacuous: other pairs with the same z_bar give another adapter.
    other = pairs + torch.randn(pairs.shape, generator=torch.Generator().manual_seed(9))
    other = other - other.mean(dim=1, keepdim=True) + z[:, None]
    assert max_diff(h(z, pairs), h(z, other)) > 100 * max(permuted, 1e-8)


def test_padded_pairs_change_nothing():
    h = Hypernet(SET, BASE)
    randomize(h, seed=3)
    h.eval()
    z, pairs = inputs(batch=2, n=3)
    alone = [h(z[b : b + 1], pairs[b : b + 1, : b + 2]) for b in range(2)]
    padded, mask = pad_pair_sets([pairs[0, :2], pairs[1, :3]])
    batched = h(z, padded, mask)
    assert close_factors(batched[:1], alone[0]) and close_factors(batched[1:], alone[1])
    # Other values in the padded rows: the same adapter, bit for bit.
    noisy = padded.clone()
    noisy[0, 2] = 1e3
    assert same_factors(batched, h(z, noisy, mask))


def test_the_set_h_checks_its_inputs():
    h = Hypernet(SET, BASE)
    z, pairs = inputs()
    with pytest.raises(ValueError, match="needs pairs"):
        h(z)
    with pytest.raises(ValueError, match="pairs"):
        h(z, pairs[:, :, :5])
    with pytest.raises(ValueError, match="bool mask"):
        h(z, pairs, torch.ones(3, 4))
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[1] = False
    with pytest.raises(ValueError, match="at least one"):
        h(z, pairs, mask)


def test_gradient_reaches_w_o_at_step_0_and_the_aggregator_at_step_1():
    model = tr.tiny_base(0)
    keys = tr.wrap_targets(model, MEAN.target_modules)
    assert keys
    base = BaseShapes.from_config(model.config)
    config = dataclasses.replace(SET, embed_dim=model.config.hidden_size)
    trained = Hypernet(dataclasses.replace(config, aggregator=AggregatorConfig()), base)
    randomize(trained, seed=4, scale=0.02, prefix="heads.")  # B != 0, as after training
    h = Hypernet(config, base)
    h.load_state_dict(trained.state_dict(), strict=False)
    generator = torch.Generator().manual_seed(5)
    pairs = torch.randn(1, 3, config.embed_dim, generator=generator)
    ids = torch.randint(0, 16, (1, 24), generator=generator)
    optimizer = torch.optim.SGD(h.parameters(), lr=0.5)

    def step():
        optimizer.zero_grad()
        with inject(model, h(pairs.mean(dim=1), pairs)[0], config.scaling):
            model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        return {n: p.grad for n, p in h.named_parameters() if n.startswith("aggregator.")}

    grads = step()
    assert grads["aggregator.output.weight"].abs().max() > 0
    assert all(not g.any() for n, g in grads.items() if not n.startswith("aggregator.output."))
    optimizer.step()
    grads = step()
    for name in ("aggregator.input.weight", "aggregator.queries", "aggregator.pool.in_proj_weight"):
        assert grads[name].abs().max() > 0, name


# --- training: groups, schedule, warm start, resume --------------------------------------------


def load_script(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script = load_script("train_hypernet")
SETTINGS = {"gpus": 1, "steps": 6, "eval_every": 3, "ckpt_every": 3, "keep_checkpoints": 6}
WARM = {"new_params_lr": 2e-3, "new_params_warmup_steps": 2}


def smoke_cfg(root, **changes):
    cfg = script.load_config(
        TOY, True, {"out_dir": str(root / "logs"), "ckpt_dir": str(root / "ckpt")}
    )
    return dataclasses.replace(cfg, **{**SETTINGS, **changes})


def set_cfg(root, **changes):
    cfg = smoke_cfg(root, **WARM, **changes)
    agg = AggregatorConfig(kind="set", width=16, blocks=2, heads=2)
    return dataclasses.replace(cfg, hypernet=dataclasses.replace(cfg.hypernet, aggregator=agg))


@pytest.fixture(scope="module")
def mean_run(tmp_path_factory):
    """The toy smoke config's mean H for 6 steps on one process, checkpoints at 3 and 6."""
    root = tmp_path_factory.mktemp("mean")
    assert tr.train(smoke_cfg(root)) == tr.EXIT_OK
    return root


def test_the_mean_config_serializes_as_before(tmp_path):
    plain = smoke_cfg(tmp_path).to_dict()
    assert "aggregator" not in plain["hypernet"]
    assert not set(tr.NEW_PARAMS_KEYS) & plain.keys()
    assert tr.TrainConfig.from_dict(plain) == smoke_cfg(tmp_path)
    assert "new_params_lr" not in tr.RESUME_KEYS and "aggregator" not in tr.RESUME_KEYS


def test_param_groups_put_new_parameters_last():
    h = Hypernet(SET, BASE)
    new = [n for n, _ in h.named_parameters() if n.startswith("aggregator.")]
    plain = tr.param_groups(Hypernet(MEAN, BASE), 0.01)
    groups = tr.param_groups(h, 0.01, new, 1e-3)
    assert len(groups) == 4 and [g["lr"] for g in groups[2:]] == [1e-3, 1e-3]
    shapes = [[tuple(p.shape) for p in g["params"]] for g in groups]
    assert shapes[:2] == [[tuple(p.shape) for p in g["params"]] for g in plain]
    names = {id(p): n for n, p in h.named_parameters()}
    assert {names[id(p)] for g in groups[2:] for p in g["params"]} == set(new)
    assert "aggregator.queries" in {names[id(p)] for p in groups[3]["params"]}
    with pytest.raises(ValueError, match="new_lr"):
        tr.param_groups(h, 0.01, new)


def test_new_params_schedule():
    schedule = tr.NewParamsSchedule(start=10, warmup=2, total=20, min_ratio=0.1)
    assert [schedule(s) for s in (8, 9, 10, 11)] == [0.0, 0.0, 0.5, 1.0]
    values = [schedule(s) for s in range(11, 20)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert schedule(19) == pytest.approx(0.1)
    assert tr.NewParamsSchedule(10, 0, 20, 0.0)(10) == pytest.approx(
        0.5 * (1 + math.cos(math.pi / 10))
    )


def test_init_from_keeps_step_cursor_and_state_and_adds_fresh_groups(mean_run, tmp_path):
    ckpt = mean_run / "ckpt" / "ckpt-000003.pt"
    payload = tr.load_checkpoint(ckpt)
    w = tr._Worker(0, 1, set_cfg(tmp_path), None, time.time(), init_from=str(ckpt))
    w.setup()
    assert w.step == 3 and w.history == [] and w.zero_ce is not None
    assert w.init_info["step"] == 3 and w.init_info["settings"] == WARM
    assert all(n.startswith("aggregator.") for n in w.init_info["parameters"])
    groups = w.optimizer.param_groups
    saved = payload["optimizer"]
    assert len(groups) == 4
    assert [g["lr"] for g in groups[:2]] == [g["lr"] for g in saved["param_groups"]]
    assert [g["lr"] for g in groups[2:]] == [WARM["new_params_lr"] / 2] * 2  # warmup step 1 of 2
    state = w.optimizer.state_dict()["state"]
    assert state.keys() == saved["state"].keys()
    for index, values in saved["state"].items():
        assert all(torch.equal(values[k], state[index][k]) for k in values)
    assert w.scheduler.last_epoch == 3
    loaded = {k: v for k, v in w.hypernet.state_dict().items() if not k.startswith("aggregator.")}
    assert loaded.keys() == payload["hypernet"].keys()
    assert all(torch.equal(loaded[k], payload["hypernet"][k]) for k in loaded)
    assert not w.hypernet.aggregator.output.weight.any()


def test_init_from_continues_the_mean_run_exactly_at_its_start(mean_run, tmp_path):
    """The warm-started H's validation at step 3 is the mean run's, bit for bit (W_o = 0); its
    data order and the loaded groups' learning rates continue the mean run's."""
    ckpt = mean_run / "ckpt" / "ckpt-000003.pt"
    assert tr.train(set_cfg(tmp_path), init_from=str(ckpt)) == tr.EXIT_OK
    mean = script.read_metrics(mean_run / "logs" / "metrics.jsonl")
    warm_run = script.read_metrics(tmp_path / "logs" / "metrics.jsonl")
    val = {r["step"]: r for r in mean if r["kind"] == "val"}
    first = warm_run[0]
    assert first["kind"] == "val" and first["step"] == 3
    assert first["per_episode"] == val[3]["per_episode"]
    trains = [(r["step"], r["episodes"], r["lr"]) for r in warm_run if r["kind"] == "train"]
    assert trains == [(r["step"], r["episodes"], r["lr"]) for r in mean if r["kind"] == "train"][3:]
    final = tr.load_checkpoint(tmp_path / "ckpt" / "ckpt-000006.pt")
    assert final["init_from"]["checkpoint"] == str(ckpt) and final["step"] == 6
    assert final["config"]["hypernet"]["aggregator"]["kind"] == "set"
    assert final["hypernet"]["aggregator.output.weight"].abs().max() > 0
    info = json.loads((tmp_path / "logs" / "run.json").read_text())
    assert info["init_from"]["step"] == 3 and info["start_step"] == 3


def test_a_warm_started_run_resumes_exactly(mean_run, tmp_path):
    ckpt = str(mean_run / "ckpt" / "ckpt-000003.pt")
    whole = set_cfg(tmp_path / "whole", ckpt_every=1)
    assert tr.train(whole, init_from=ckpt) == tr.EXIT_OK
    resumed = set_cfg(tmp_path / "resumed")
    start = tmp_path / "whole" / "ckpt" / "ckpt-000004.pt"
    assert tr.train(resumed, resume=str(start)) == tr.EXIT_OK
    a = tr.load_checkpoint(tmp_path / "whole" / "ckpt" / "ckpt-000006.pt")
    b = tr.load_checkpoint(tmp_path / "resumed" / "ckpt" / "ckpt-000006.pt")
    assert all(torch.equal(a["hypernet"][k], b["hypernet"][k]) for k in a["hypernet"])
    assert [g["lr"] for g in a["optimizer"]["param_groups"]] == [
        g["lr"] for g in b["optimizer"]["param_groups"]
    ]
    other = dataclasses.replace(resumed, new_params_lr=1e-3, out_dir=str(tmp_path / "x"))
    with pytest.raises(ValueError, match="warm-started"):
        tr.train(other, resume=str(start))


def test_init_from_refusals(mean_run, tmp_path):
    ckpt = str(mean_run / "ckpt" / "ckpt-000003.pt")
    cases = [
        (set_cfg(tmp_path / "a", lr=1e-2), ckpt, "other settings"),
        (smoke_cfg(tmp_path / "b"), ckpt, "use --resume"),
        (dataclasses.replace(set_cfg(tmp_path / "c"), new_params_lr=None), ckpt, "needs"),
        (set_cfg(tmp_path / "d"), str(mean_run / "ckpt" / "best.pt"), "full checkpoint"),
    ]
    for cfg, path, match in cases:
        with pytest.raises(ValueError, match=match):
            tr._Worker(0, 1, cfg, None, time.time(), init_from=path).setup()
    with pytest.raises(ValueError, match="exclude"):
        tr._Worker(0, 1, set_cfg(tmp_path / "e"), ckpt, time.time(), init_from=ckpt)
    code = script.main(["--config", str(TOY), "--smoke", "--resume", ckpt, "--init-from", ckpt])
    assert code == tr.EXIT_CONFIG


def test_the_mismatched_control_gets_the_other_familys_set(mean_run, tmp_path, monkeypatch):
    ckpt = str(mean_run / "ckpt" / "ckpt-000003.pt")
    w = tr._Worker(0, 1, set_cfg(tmp_path), None, time.time(), init_from=ckpt)
    w.setup()
    seen = []
    monkeypatch.setattr(w, "episode_ce", lambda seq, z, pairs: seen.append((z, pairs)) or 0.0)
    w.evaluate()
    for i in range(len(w.val)):
        (z, pairs), (z_other, pairs_other) = seen[2 * i : 2 * i + 2]
        j = w.mismatch[i]
        assert torch.equal(pairs, w.val_sets[i]) and torch.equal(pairs_other, w.val_sets[j])
        assert torch.equal(z, pairs.mean(dim=0)) and torch.equal(z_other, pairs_other.mean(dim=0))
