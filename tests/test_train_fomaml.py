"""P3.9: the fomaml objective and the decode-aligned validation metrics (arc26/hyper/train.py).

On the CPU with the tiny random Qwen3 and the fixture build of tests/test_train.py, as the other
training tests. The first-order gradient is checked against finite differences of
$\\mathcal L_Q(H_\\phi(\\bar z) + \\Delta)$ with $\\Delta = \\theta_k - \\theta_0$ held fixed, which
is $\\theta_k$ as a function of $\\theta_0$ under the first-order approximation
($\\partial\\theta_k / \\partial\\theta_0 = I$). Run with CUDA_VISIBLE_DEVICES="" (BRIEF A0).
"""

import dataclasses
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.inject import inject  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TOY = REPO / "configs" / "train_hypernet_toy.yaml"
MT = REPO / "configs" / "train_hypernet_MT.yaml"
P39 = REPO / "configs" / "train_hypernet_P3.9-01.yaml"
HYPERNET = REPO / "configs" / "hypernet.yaml"


def load_script(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script = load_script("train_hypernet")


def inner(steps=2, lr=1e-2, optimizer="adam", views="drawn", clip=1.0, eps=1e-8):
    return tr.InnerLoop(
        steps=steps,
        lr=lr,
        optimizer=optimizer,
        betas=(0.9, 0.999),
        eps=eps,
        views=views,
        clip=clip,
    )


def toy_cfg(tmp_path, **changes):
    cfg = script.load_config(
        TOY, True, {"out_dir": str(tmp_path / "logs"), "ckpt_dir": str(tmp_path / "ckpt")}
    )
    return dataclasses.replace(cfg, **changes)


def fomaml_cfg(tmp_path, k=2, **changes):
    return toy_cfg(tmp_path, objective="fomaml", inner=inner(steps=k), **changes)


def no_dropout(cfg):
    return dataclasses.replace(cfg, hypernet=dataclasses.replace(cfg.hypernet, dropout=0.0))


def worker(cfg):
    w = tr._Worker(0, 1, cfg, None, time.time())
    w.setup()
    return w


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def h_state(path):
    return tr.load_checkpoint(Path(path))["hypernet"]


def assert_same_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def train_records(path):
    return [r for r in read_jsonl(path) if r["kind"] == "train"]


def random_factors(keys, d_in=6, d_out=5, r=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        key: (torch.randn(r, d_in, generator=g), torch.randn(d_out, r, generator=g)) for key in keys
    }


# --- configuration --------------------------------------------------------------------------------


def test_zero_step_configs_write_the_config_of_the_code_before_p3_9(tmp_path):
    cfg = toy_cfg(tmp_path)
    assert cfg.objective == "zero_step" and cfg.inner is None
    data = cfg.to_dict()
    assert "objective" not in data and "inner" not in data
    assert tr.TrainConfig.from_dict(data) == cfg
    # a checkpoint config written before P3.9 (no objective keys) resumes as before
    assert tr.resume_conflicts(data, cfg) == []
    # the objective keys are resume keys whose absence reads as zero_step: every checkpoint
    # written before P3.9 (MT-02, MT-03, X1-01) resumes as before
    assert set(tr.OBJECTIVE_KEYS) <= set(tr.RESUME_KEYS)
    old = {k: v for k, v in data.items() if k not in tr.OBJECTIVE_KEYS}
    assert tr.resume_conflicts(old, cfg) == []
    for name in ("MT", "X1-01"):
        raw = yaml.safe_load((REPO / "configs" / f"train_hypernet_{name}.yaml").read_text())
        raw.pop("smoke")
        assert tr.resume_conflicts(raw, tr.TrainConfig.from_dict(raw)) == []


def test_fomaml_needs_its_inner_loop_and_is_a_resume_setting(tmp_path):
    cfg = fomaml_cfg(tmp_path)
    data = cfg.to_dict()
    assert data["objective"] == "fomaml" and data["inner"]["steps"] == 2
    assert tr.TrainConfig.from_dict(data) == cfg
    zero = toy_cfg(tmp_path)
    assert tr.resume_conflicts(zero.to_dict(), cfg) == ["objective", "inner"]
    assert tr.resume_conflicts(data, zero) == ["objective", "inner"]
    changes = ({"steps": 1}, {"lr": 2e-2}, {"optimizer": "sgd"}, {"views": "episode"},
               {"clip": None})  # fmt: skip
    for change in changes:
        other = dataclasses.replace(cfg, inner=dataclasses.replace(cfg.inner, **change))
        assert tr.resume_conflicts(data, other) == ["inner"], change
    phase = dataclasses.replace(cfg, fresh_schedule=True)
    assert tr.resume_conflicts(data, phase) == ["fresh_schedule"]
    assert phase.to_dict()["fresh_schedule"] is True
    assert tr.init_conflicts(zero.to_dict(), cfg) == []  # --init-from may change the objective
    with pytest.raises(ValueError, match="inner must be set"):
        dataclasses.replace(cfg, inner=None)
    with pytest.raises(ValueError, match="inner must be set"):
        dataclasses.replace(zero, inner=inner())
    with pytest.raises(ValueError, match="objective"):
        dataclasses.replace(zero, objective="maml")
    for bad in ({"steps": -1}, {"lr": 0.0}, {"optimizer": "lion"}, {"views": "all"},
                {"clip": 0.0}):  # fmt: skip
        with pytest.raises(ValueError):
            dataclasses.replace(cfg.inner, **bad)
    raw = {**data, "inner": {**data["inner"], "momentum": 0.9}}
    with pytest.raises(ValueError, match="unknown"):
        tr.TrainConfig.from_dict(raw)


def test_the_p3_9_config_keeps_mt_s_resume_keys_and_comments_every_key():
    p39, mt = yaml.safe_load(P39.read_text()), yaml.safe_load(MT.read_text())
    # a new phase: the objective and the schedule are its own, every other resume key is MT's
    for key in set(tr.RESUME_KEYS) - set(tr.OBJECTIVE_KEYS) - set(tr.SCHEDULE_KEYS):
        assert p39[key] == mt[key], key
    assert p39["fresh_schedule"] is True
    assert (p39["lr"], p39["warmup_steps"], p39["min_lr"]) == (4.0e-5, 50, mt["min_lr"])
    assert p39["steps"] > 4180 + p39["warmup_steps"]  # the phase starts at MT-03's step 4,180
    for key in set(tr.RESUME_KEYS) & set(mt["smoke"]):
        if key != "hypernet":
            assert p39["smoke"][key] == mt["smoke"][key], key
    assert p39["smoke"]["hypernet"] == mt["smoke"]["hypernet"]
    assert "aggregator" not in p39["hypernet"]
    hypernet = yaml.safe_load(HYPERNET.read_text())
    assert {k: v for k, v in hypernet.items() if k != "aggregator"} == p39["hypernet"]
    assert p39["objective"] == "fomaml" and p39["inner"]["steps"] == 2
    for smoke in (False, True):
        cfg = script.load_config(P39, smoke, {})
        assert cfg.objective == "fomaml" and cfg.inner.optimizer == "adam"
    assert script.load_config(P39, True, {}).inner.lr == 1.0e-2
    covered = False
    for line in P39.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            covered = False
        elif stripped.startswith("#"):
            covered = True
        elif re.match(r"[a-z_0-9]+:", stripped):
            assert covered or "#" in line, line


# --- sequences, views, and the inner update -------------------------------------------------------


def test_the_episode_view_of_the_demonstrations_is_the_prefix_of_the_episode(tmp_path):
    w = worker(fomaml_cfg(tmp_path, gpus=1))
    for record in w.train_records:
        seq = tr.build_sequence(record, w.table, w.formatter, 8192)
        demo = tr.build_demo_sequence(record, w.table, w.formatter, record.augmentation_id, 8192)
        n = demo.input_ids.numel()
        assert torch.equal(seq.input_ids[:n], demo.input_ids)
        assert len(demo.targets) == record.n_demos
        # every target ends with the closing <|im_end|> and holds the augmented output grid
        pairs = tr.augmented_pairs(record, w.table)[: record.n_demos]
        for target, pair in zip(demo.targets, pairs, strict=True):
            tokens = demo.input_ids[target.start : target.stop].tolist()
            assert tokens[-1] == tr.STANDIN_VOCAB["<|im_end|>"]
            digits = [t for t in tokens[:-1] if t < 10]
            assert digits == pair["output"].flatten().tolist()
        # the query's target lies after the demonstrations
        assert seq.target.start > n


def test_inner_views_are_a_pure_function_of_the_seed_and_the_episode():
    ids = tr.inner_view_ids(0, "barc/x/toy/0", own=5, steps=4, n_augmentations=32, views="drawn")
    assert ids == tr.inner_view_ids(0, "barc/x/toy/0", 5, 4, 32, "drawn")
    assert len(set(ids)) == 4 and all(0 <= i < 32 for i in ids)
    assert ids != tr.inner_view_ids(1, "barc/x/toy/0", 5, 4, 32, "drawn")
    assert ids != tr.inner_view_ids(0, "barc/x/toy/1", 5, 4, 32, "drawn")
    assert tr.inner_view_ids(0, "barc/x/toy/0", 5, 3, 32, "episode") == [5, 5, 5]
    assert tr.inner_view_ids(0, "e", 0, 0, 32, "drawn") == []
    # past n_augmentations the shuffle cycles
    cycled = tr.inner_view_ids(0, "e", 0, 5, 2, "drawn")
    assert sorted(cycled[:2]) == [0, 1] and cycled == cycled[:2] * 2 + cycled[:1]


def test_the_adam_inner_step_is_torch_adam_and_sgd_is_plain():
    torch.manual_seed(0)
    start = [torch.randn(3, 4), torch.randn(5, 3)]
    grads = [[torch.randn(3, 4), torch.randn(5, 3)] for _ in range(3)]
    settings = inner(steps=3, lr=1e-2)
    ours = [t.clone() for t in start]
    optimizer = tr.InnerOptimizer(settings)
    reference = [t.clone().requires_grad_(True) for t in start]
    adam = torch.optim.Adam(reference, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    for j in range(3):
        optimizer.step(ours, grads[j], j)
        for param, grad in zip(reference, grads[j], strict=True):
            param.grad = grad.clone()
        adam.step()
        for a, b in zip(ours, reference, strict=True):
            torch.testing.assert_close(a, b.detach(), rtol=1e-6, atol=1e-7)
    # k = 1 keeps no moments (the last step's are temporaries); k = 3 keeps two per tensor
    single = tr.InnerOptimizer(inner(steps=1))
    single.step([t.clone() for t in start], grads[0], 0)
    assert single.state is None and optimizer.state is not None
    sgd = [t.clone() for t in start]
    tr.InnerOptimizer(inner(steps=1, lr=0.5, optimizer="sgd")).step(sgd, grads[0], 0)
    for a, s, g in zip(sgd, start, grads[0], strict=True):
        assert torch.equal(a, s - 0.5 * g)


def test_inner_steps_touch_neither_the_base_nor_h(tmp_path):
    cfg = no_dropout(fomaml_cfg(tmp_path, gpus=1))
    w = worker(cfg)
    assert w.train_step() == (0, "")  # B leaves zero, so both factors move
    w.optimizer.zero_grad(set_to_none=True)
    base = {key: value.clone() for key, value in w.model.state_dict().items()}
    h_before = {key: value.clone() for key, value in w.hypernet.state_dict().items()}
    record = w.train_records[2]
    seq = tr.build_sequence(record, w.table, w.formatter, 8192)
    w._train_modes()
    factors = w.hypernet(w.embeddings.z_bar(seq.demo_hashes)[None])[0]  # in H's graph
    emitted = {key: (a.detach().clone(), b.detach().clone()) for key, (a, b) in factors.items()}
    adapted, losses, norms = tr.adapt(
        w.model, factors, w.inner_demos(record), cfg.inner, cfg.hypernet.scaling, w.device
    )
    assert len(norms) == 2 and all(v > 0 for v in norms)
    assert len(losses) == 2 and all(math.isfinite(v) for v in losses)
    for key, (a, b) in adapted.items():
        assert a.is_leaf and b.is_leaf and a.grad is None and b.grad is None
        assert not torch.equal(a, emitted[key][0]) and not torch.equal(b, emitted[key][1])
        assert torch.equal(factors[key][0], emitted[key][0])  # theta_0 is not moved in place
        assert torch.equal(factors[key][1], emitted[key][1])
    for name, param in w.hypernet.named_parameters():
        assert param.grad is None, name
    for param in w.model.parameters():
        assert not param.requires_grad and param.grad is None
    for key, value in w.model.state_dict().items():
        assert torch.equal(value, base[key]), key
    assert_same_state(w.hypernet.state_dict(), h_before)
    # the micro step accumulates into H only, and nothing in the base moves
    w.fomaml_micro_step(record, seq, 0)
    for name, param in w.hypernet.named_parameters():
        assert param.grad is not None and param.grad.abs().max() > 0, name
    for key, value in w.model.state_dict().items():
        assert torch.equal(value, base[key]), key


# --- the gradient ---------------------------------------------------------------------------------


def outer_loss(w, seq, delta):
    r"""$\mathcal L_Q(H_\phi(\bar z) + \Delta)$ under H's current parameters."""
    ids = seq.input_ids
    with torch.no_grad():
        factors = w.hypernet(w.embeddings.z_bar(seq.demo_hashes)[None])[0]
        moved = {k: (a + delta[k][0], b + delta[k][1]) for k, (a, b) in factors.items()}
        with inject(w.model, moved, w.cfg.hypernet.scaling):
            logits = w.model(input_ids=ids[None], use_cache=False).logits[0]
        return float(tr.query_ce(logits, ids, seq.target))


def test_the_first_order_gradient_matches_finite_differences(tmp_path):
    cfg = no_dropout(fomaml_cfg(tmp_path, gpus=1, accum=1))
    w = worker(cfg)
    for _ in range(2):
        assert w.train_step() == (0, "")
    w.optimizer.zero_grad(set_to_none=True)
    record = w.train_records[4]
    seq = tr.build_sequence(record, w.table, w.formatter, 8192)
    w._train_modes()
    ce, _, stats = w.fomaml_micro_step(record, seq, 0)
    names = [n for n, _ in w.hypernet.named_parameters()]
    grads = {n: p.grad.clone() for n, p in w.hypernet.named_parameters()}
    # Delta = theta_k - theta_0 of the same inner loop, recomputed (deterministic on the CPU)
    with torch.no_grad():
        start = w.hypernet(w.embeddings.z_bar(seq.demo_hashes)[None])[0]
    adapted, losses, norms = tr.adapt(
        w.model, start, w.inner_demos(record), cfg.inner, cfg.hypernet.scaling, w.device
    )
    assert losses == stats[:2] and norms == stats[2:4]
    delta = {k: (adapted[k][0] - a, adapted[k][1] - b) for k, (a, b) in start.items()}
    assert outer_loss(w, seq, delta) == pytest.approx(ce, abs=1e-5)
    params = dict(w.hypernet.named_parameters())
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads.values()))
    torch.manual_seed(1)
    random_dir = {n: torch.randn_like(params[n]) for n in names}
    random_norm = math.sqrt(sum(float(v.square().sum()) for v in random_dir.values()))
    for direction in (
        {n: grads[n] / norm for n in names},
        {n: v / random_norm for n, v in random_dir.items()},
    ):
        eps = 1e-3
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                for n in names:
                    params[n].add_(direction[n], alpha=sign * eps)
            values.append(outer_loss(w, seq, delta))
            with torch.no_grad():
                for n in names:
                    params[n].sub_(direction[n], alpha=sign * eps)
        numeric = (values[0] - values[1]) / (2 * eps)
        analytic = sum(float((grads[n] * direction[n]).sum()) for n in names)
        assert numeric == pytest.approx(analytic, rel=2e-2, abs=2e-3)
    assert norm > 1e-3


def test_checkpointing_on_and_off_give_equal_fomaml_gradients(tmp_path):
    grads, calls = [], []
    for checkpointing in (False, True):
        cfg = fomaml_cfg(
            tmp_path / str(checkpointing), gpus=1, gradient_checkpointing=checkpointing
        )
        w = worker(cfg)
        assert w.train_step() == (0, "")
        q_proj = w.model.get_submodule(f"model.{w.keys[0]}")
        count = []
        handle = q_proj.register_forward_pre_hook(lambda *_, count=count: count.append(1))
        w._train_modes()
        record = w.train_records[1]
        w.fomaml_micro_step(record, tr.build_sequence(record, w.table, w.formatter, 8192), 0)
        handle.remove()
        calls.append(len(count))
        grads.append({n: p.grad.clone() for n, p in w.hypernet.named_parameters()})
    # two inner steps and the outer pass: 3 forwards, each run again during its backward
    assert calls == [3, 6]
    for name in grads[0]:
        assert torch.equal(grads[0][name], grads[1][name]), name


# --- runs -----------------------------------------------------------------------------------------


def test_k_0_under_fomaml_equals_zero_step_bitwise(tmp_path):
    settings = {"gpus": 1, "steps": 4, "eval_every": 2, "ckpt_every": 4}
    assert tr.train(toy_cfg(tmp_path / "zero", **settings)) == tr.EXIT_OK
    k0 = toy_cfg(tmp_path / "k0", objective="fomaml", inner=inner(steps=0), **settings)
    assert tr.train(k0) == tr.EXIT_OK
    assert_same_state(
        h_state(tmp_path / "zero" / "ckpt" / "ckpt-000004.pt"),
        h_state(tmp_path / "k0" / "ckpt" / "ckpt-000004.pt"),
    )
    a = read_jsonl(tmp_path / "zero" / "logs" / "metrics.jsonl")
    b = read_jsonl(tmp_path / "k0" / "logs" / "metrics.jsonl")
    keys = ("kind", "step", "episodes", "loss", "grad_norm", "ce", "per_episode")
    assert [{k: r.get(k) for k in keys} for r in a] == [{k: r.get(k) for k in keys} for r in b]
    assert all("inner" not in r and "adapted" not in r for r in b)


def test_fomaml_runs_are_deterministic_and_resume_bit_identically(tmp_path):
    settings = {"gpus": 1, "steps": 6, "eval_every": 3, "ckpt_every": 3, "keep_checkpoints": 3}
    for name in ("a", "b"):
        assert tr.train(fomaml_cfg(tmp_path / name, **settings)) == tr.EXIT_OK
    start = tmp_path / "a" / "ckpt" / "ckpt-000003.pt"
    assert tr.train(fomaml_cfg(tmp_path / "r", **settings), resume=str(start)) == tr.EXIT_OK
    final = [h_state(tmp_path / n / "ckpt" / "ckpt-000006.pt") for n in ("a", "b", "r")]
    assert_same_state(final[0], final[1])
    assert_same_state(final[0], final[2])
    runs = [read_jsonl(tmp_path / n / "logs" / "metrics.jsonl") for n in ("a", "b", "r")]

    def stable(records):
        drop = {"step_s", "seconds", "best_write_s", "session_start"}
        return [{k: v for k, v in r.items() if k not in drop} for r in records]

    assert stable(runs[0]) == stable(runs[1])
    assert stable(runs[0])[-3:] == stable(runs[2])[-3:]  # steps 4 to 6 and the last validation
    # a resume under zero_step from a fomaml checkpoint is refused
    with pytest.raises(ValueError, match="objective"):
        tr.train(toy_cfg(tmp_path / "z", **settings), resume=str(start))


def test_gloo_ddp_matches_one_process_with_accumulation_under_fomaml(tmp_path):
    """Two gloo ranks against one process with accumulation 2, as test_train.py's zero_step check:
    the inner loop runs per episode on each rank, and H's gradients meet in DDP's all-reduce."""
    hypernet = dataclasses.replace(toy_cfg(tmp_path).hypernet, dropout=0.0)
    settings = {"steps": 3, "eval_every": 3, "ckpt_every": 3, "hypernet": hypernet}
    ddp = fomaml_cfg(tmp_path / "ddp", gpus=2, accum=1, **settings)
    acc = fomaml_cfg(tmp_path / "acc", gpus=1, accum=2, **settings)
    assert tr.train(ddp) == tr.EXIT_OK and tr.train(acc) == tr.EXIT_OK
    a = h_state(tmp_path / "ddp" / "ckpt" / "ckpt-000003.pt")
    b = h_state(tmp_path / "acc" / "ckpt" / "ckpt-000003.pt")
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6, msg=key)
    steps_a, steps_b = (
        train_records(tmp_path / n / "logs" / "metrics.jsonl") for n in ("ddp", "acc")
    )
    for ra, rb in zip(steps_a, steps_b, strict=True):
        assert ra["episodes"] == rb["episodes"]
        assert ra["loss"] == pytest.approx(rb["loss"], rel=1e-5)
        assert ra["inner"]["demo_ce"] == pytest.approx(rb["inner"]["demo_ce"], rel=1e-5)
    val_a, val_b = (
        [r for r in read_jsonl(tmp_path / n / "logs" / "metrics.jsonl") if r["kind"] == "val"]
        for n in ("ddp", "acc")
    )
    for ra, rb in zip(val_a, val_b, strict=True):
        assert ra["adapted"]["per_episode"] == pytest.approx(rb["adapted"]["per_episode"], abs=1e-6)
        assert ra["adapted"]["per_episode_exact"] == rb["adapted"]["per_episode_exact"]


def test_training_logs_the_inner_loop_and_validation_the_post_k_scores(tmp_path):
    cfg = fomaml_cfg(tmp_path, gpus=1, steps=4, eval_every=4, ckpt_every=4)
    assert tr.train(cfg) == tr.EXIT_OK
    metrics = read_jsonl(tmp_path / "logs" / "metrics.jsonl")
    step = [r for r in metrics if r["kind"] == "train"][-1]
    block = step["inner"]
    modules = sorted(cfg.hypernet.target_modules)
    assert block["steps"] == 2 and len(block["demo_ce"]) == 2 and block["lr"] == cfg.inner.lr
    assert len(block["grad_norm"]) == 2 and all(v > 0 for v in block["grad_norm"])
    assert sorted(block["displacement"]) == modules == sorted(block["start_norm"])
    assert all(v > 0 for v in block["displacement"].values())
    val = [r for r in metrics if r["kind"] == "val"]
    first = val[0]["adapted"]
    # at step 0 B = 0, so the start norm is 0 and the inner steps still move the adapter
    assert all(v == 0 for v in first["start_norm"].values())
    assert all(v > 0 for v in first["displacement"].values())
    adapted = val[-1]["adapted"]
    n = val[-1]["n"]
    assert adapted["steps"] == 2 and len(adapted["per_episode"]) == n
    assert len(adapted["per_projection"]["displacement"]) == 4 * len(modules)
    assert set(adapted["decode"]) == {
        "tokens",
        "nll_sum",
        "under_bound",
        "exact_match",
        "geomean_ce",
    }
    assert set(val[-1]["decode"]) == set(tr.REFERENCES)
    assert adapted["minus_current"]["mean"] == pytest.approx(
        adapted["ce"] - val[-1]["ce"]["current"], abs=1e-6
    )
    assert len(adapted["grad_norm"]) == 2
    timing = read_jsonl(tmp_path / "logs" / "timing-rank0.jsonl")
    ops = {r["op"] for r in timing}
    assert {"inner_step", "h_backward", "validate_adapted", "train_step", "h_forward"} <= ops
    assert "validate_inner_step" in ops  # validation's inner steps apart from training's
    assert all(set(r) >= {"puzzle_id", "op", "seconds", "tokens"} for r in timing)
    # one process: the training records' inner tokens are the inner_step records' tokens, and
    # each validation runs k inner steps per episode under its own op
    train = [r for r in metrics if r["kind"] == "train"]
    inner_tokens = sum(r["tokens"] for r in timing if r["op"] == "inner_step")
    assert sum(r["inner"]["tokens"] for r in train) == inner_tokens > 0
    counted = sum(1 for r in timing if r["op"] == "validate_inner_step")
    assert counted == 2 * n * len(val)
    status = tr.read_status(tmp_path / "logs")
    assert status["best"]["adapted"]["steps"] == 2
    assert "per_episode" not in status["best"]["adapted"]
    best = tr.load_checkpoint(tmp_path / "ckpt" / tr.BEST_FILE)
    assert best["validation"]["adapted"]["ce"] == pytest.approx(
        [r for r in val if r["best"]][-1]["adapted"]["ce"]
    )


def test_the_post_k_validation_matches_an_independent_recomputation(tmp_path):
    cfg = fomaml_cfg(tmp_path / "run", gpus=1, steps=4, eval_every=4, ckpt_every=4)
    assert tr.train(cfg) == tr.EXIT_OK
    record = [
        r for r in read_jsonl(tmp_path / "run" / "logs" / "metrics.jsonl") if r["kind"] == "val"
    ][-1]
    fresh = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "re" / "logs"), ckpt_dir=str(tmp_path / "re" / "ckpt")
    )
    w = worker(fresh)
    w.hypernet.load_state_dict(h_state(tmp_path / "run" / "ckpt" / "ckpt-000004.pt"))
    w.hypernet.eval()
    scaling = cfg.hypernet.scaling
    for i, seq in enumerate(w.val):
        with torch.no_grad():
            start = w.hypernet(w.val_z[i][None])[0]
        w.model.train(cfg.gradient_checkpointing)
        demos = w.inner_demos(w.val_records[i])
        assert [d.augmentation_id for d in demos] == tr.inner_view_ids(
            cfg.seed, seq.episode_id, w.val_records[i].augmentation_id, 2, w.table.n_augmentations,
            "drawn",
        )  # fmt: skip
        adapted, _, _ = tr.adapt(w.model, start, demos, cfg.inner, scaling, w.device)
        w.model.eval()
        with torch.no_grad(), inject(w.model, adapted, scaling):
            logits = w.model(input_ids=seq.input_ids[None]).logits[0]
        ce, exact = tr.query_scores(logits, seq.input_ids, seq.target)
        assert record["adapted"]["per_episode"][i] == pytest.approx(float(ce), abs=2e-6)
        assert record["adapted"]["per_episode_exact"][i] == int(exact)


def test_init_from_a_zero_step_checkpoint_changes_the_objective(tmp_path):
    settings = {"gpus": 1, "steps": 8, "eval_every": 4, "ckpt_every": 4, "keep_checkpoints": 2}
    assert tr.train(toy_cfg(tmp_path / "zero", **settings)) == tr.EXIT_OK
    ckpt = tmp_path / "zero" / "ckpt" / "ckpt-000004.pt"
    # the same objective and parameters: --resume is the tool
    with pytest.raises(ValueError, match="use --resume"):
        tr.train(toy_cfg(tmp_path / "same", **settings), init_from=str(ckpt))
    cfg = fomaml_cfg(tmp_path / "f", **settings)
    assert tr.train(cfg, init_from=str(ckpt)) == tr.EXIT_OK
    info = json.loads((tmp_path / "f" / "logs" / "run.json").read_text())
    assert info["start_step"] == 4 and info["init_from"]["parameters"] == []
    assert info["init_from"]["objective"] == {"from": "zero_step", "to": "fomaml"}
    zero = read_jsonl(tmp_path / "zero" / "logs" / "metrics.jsonl")
    warm = read_jsonl(tmp_path / "f" / "logs" / "metrics.jsonl")
    # the start step's 0-step validation is the checkpoint's H; the data order continues
    zero_val4 = [r for r in zero if r["kind"] == "val" and r["step"] == 4][0]
    warm_val4 = [r for r in warm if r["kind"] == "val"][0]
    assert warm_val4["step"] == 4 and "adapted" in warm_val4
    assert warm_val4["per_episode"]["current"] == zero_val4["per_episode"]["current"]
    assert [r["episodes"] for r in warm if r["kind"] == "train"] == [
        r["episodes"] for r in zero if r["kind"] == "train" and r["step"] > 4
    ]
    # the optimizer and the cosine continue: the first warm step's lr is the zero run's step 5
    lr = {r["step"]: r["lr"] for r in zero if r["kind"] == "train"}
    assert [r["lr"] for r in warm if r["kind"] == "train"][0] == lr[5]
    # the warm-started run resumes from its own checkpoints
    final = tmp_path / "f" / "ckpt" / "ckpt-000008.pt"
    assert tr.load_checkpoint(final)["init_from"]["objective"]["to"] == "fomaml"


def test_a_fresh_schedule_starts_a_new_phase_and_resumes_exactly(tmp_path):
    settings = {"gpus": 1, "eval_every": 4, "ckpt_every": 2, "keep_checkpoints": 5}
    assert tr.train(toy_cfg(tmp_path / "zero", steps=8, **settings)) == tr.EXIT_OK
    ckpt = tmp_path / "zero" / "ckpt" / "ckpt-000004.pt"
    # phase: global steps 5 to 10, warmup 2, peak 2e-3, cosine to min_lr at step 10
    phase = {"steps": 10, "lr": 2e-3, "min_lr": 1e-4, "warmup_steps": 2, **settings}
    cfg = fomaml_cfg(tmp_path / "p", fresh_schedule=True, **phase)
    assert tr.init_conflicts(tr.load_checkpoint(ckpt)["config"], cfg) == []
    w = tr._Worker(0, 1, cfg, None, time.time(), init_from=str(ckpt))
    w.setup()
    assert_same_state(w.hypernet.state_dict(), tr.load_checkpoint(ckpt)["hypernet"])
    assert w.step == 4 and w.optimizer.state_dict()["state"] == {}  # fresh moments
    assert len(w.optimizer.param_groups) == 2
    assert w.optimizer.param_groups[0]["lr"] == pytest.approx(2e-3 / 2)  # phase step 1 of 2
    assert tr.train(cfg, init_from=str(ckpt)) == tr.EXIT_OK
    steps = train_records(tmp_path / "p" / "logs" / "metrics.jsonl")
    assert [r["step"] for r in steps] == [5, 6, 7, 8, 9, 10]  # the global step continues
    lrs = [r["lr"] for r in steps]
    assert lrs[0] == pytest.approx(1e-3) and lrs[1] == pytest.approx(2e-3)  # warmup, then peak
    assert lrs[-1] == pytest.approx(1e-4)  # min_lr at the phase's last step
    assert all(a > b for a, b in zip(lrs[1:], lrs[2:], strict=False))
    info = tr.load_checkpoint(tmp_path / "p" / "ckpt" / "ckpt-000010.pt")["init_from"]
    assert info["fresh_schedule"] == {
        "start": 4, "phase_steps": 6, "warmup_steps": 2, "lr": 2e-3, "min_lr": 1e-4,
    }  # fmt: skip
    # a resume mid-phase rebuilds the phase schedule and ends bitwise equal
    resumed = fomaml_cfg(tmp_path / "r", fresh_schedule=True, **phase)
    mid = tmp_path / "p" / "ckpt" / "ckpt-000006.pt"
    assert tr.train(resumed, resume=str(mid)) == tr.EXIT_OK
    assert_same_state(
        h_state(tmp_path / "p" / "ckpt" / "ckpt-000010.pt"),
        h_state(tmp_path / "r" / "ckpt" / "ckpt-000010.pt"),
    )
    again = train_records(tmp_path / "r" / "logs" / "metrics.jsonl")
    assert [r["lr"] for r in again] == lrs
    # without fresh_schedule the resume is refused; a phase longer than its warmup is required
    with pytest.raises(ValueError, match="fresh_schedule"):
        tr.train(fomaml_cfg(tmp_path / "x", **phase), resume=str(mid))
    short = fomaml_cfg(tmp_path / "s", fresh_schedule=True, **{**phase, "steps": 6})
    with pytest.raises(ValueError, match="warmup_steps"):
        tr.train(short, init_from=str(ckpt))


def test_the_p3_9_smoke_warm_starts_from_the_mt_smoke(tmp_path):
    def cfg(path, out):
        return script.load_config(
            path, True, {"out_dir": str(out / "logs"), "ckpt_dir": str(out / "ckpt"), "gpus": 1}
        )

    mt = dataclasses.replace(cfg(MT, tmp_path / "mt"), steps=12, ckpt_every=4, eval_every=4)
    assert tr.train(mt) == tr.EXIT_OK
    p39 = cfg(P39, tmp_path / "p39")
    assert tr.init_conflicts(mt.to_dict(), p39) == []
    assert tr.train(p39, init_from=str(tmp_path / "mt" / "ckpt" / "ckpt-000008.pt")) == 0
    status = tr.read_status(tmp_path / "p39" / "logs")
    assert status["step"] == 12 and status["best"]["adapted"]["steps"] == 2


def test_the_script_smoke_of_the_p3_9_config_exits_0(tmp_path):
    """The A8 smoke as the runbook runs it: the MT smoke's checkpoint, then the P3.9 smoke."""

    def run(config, out, *extra):
        return script.main(
            [
                "--config", str(config), "--smoke", "--commit", "test",
                "--out-dir", str(out / "logs"), "--ckpt-dir", str(out / "ckpt"), *extra,
            ]
        )  # fmt: skip

    # fresh_schedule and inner_probe without a warm start: refused before any work, exit 2
    assert run(P39, tmp_path / "bare") == tr.EXIT_CONFIG
    assert not (tmp_path / "bare" / "ckpt").exists()
    assert run(MT, tmp_path / "mt") == 0
    ckpt = tmp_path / "mt" / "ckpt" / "ckpt-000008.pt"
    assert run(P39, tmp_path / "p39", "--init-from", str(ckpt)) == 0
    summary = (tmp_path / "p39" / "logs" / "summary.md").read_text()
    assert "Decode-aligned" in summary and "After k" in summary
    assert "Inner-rate probe at step 8" in summary and "| yes |" in summary


# --- the metrics ----------------------------------------------------------------------------------


def test_query_scores_exact_match_on_hand_logits():
    ids = torch.tensor([3, 1, 2, 0])
    target = range(2, 4)  # tokens 2 and 0, predicted by rows 1 and 2
    logits = torch.zeros(4, 5)
    logits[1, 2] = 5.0
    logits[2, 0] = 5.0
    ce, exact = tr.query_scores(logits, ids, target)
    assert exact and torch.equal(ce, tr.query_ce(logits, ids, target))
    logits[2, 4] = 6.0
    assert tr.query_scores(logits, ids, target)[1] is False
    logits[2, 4] = 0.0
    logits[0, 1] = -9.0  # a row outside O_Q's predictions does not count
    assert tr.query_scores(logits, ids, target)[1] is True


def test_decode_metrics_on_hand_cases():
    ces = torch.tensor([0.1, 0.02, 0.0, 1.0])
    lengths = torch.tensor([10.0, 100.0, 5.0, 2.0])
    exact = torch.tensor([1.0, 0.0, 1.0, 0.0])
    m = tr.decode_metrics(ces, lengths, exact)
    assert m["tokens"] == pytest.approx(117 / 4)
    # S = 1.0, 2.0, 0.0, 2.0; ln 5 = 1.609: two of four under the bound
    assert m["nll_sum"] == pytest.approx(5.0 / 4)
    assert m["under_bound"] == 0.5 and m["exact_match"] == 0.5
    floor = tr.GEOMEAN_FLOOR
    expected = math.exp((math.log(0.1) + math.log(0.02) + math.log(floor) + 0.0) / 4)
    assert m["geomean_ce"] == pytest.approx(expected)


def test_displacement_and_delta_norms_equal_the_dense_frobenius_norms():
    keys = [f"layers.{i}.self_attn.{m}" for i in range(2) for m in ("q_proj", "v_proj")]
    start, end = random_factors(keys, seed=0), random_factors(keys, seed=1)
    s = 8.0
    moved, norms = tr.displacement(start, end, s), tr.delta_norms(start, s)
    for key in keys:
        (a0, b0), (ak, bk) = start[key], end[key]
        dense = torch.linalg.norm(s * (bk @ ak - b0 @ a0)).item()
        assert moved[key] == pytest.approx(dense, rel=1e-5)
        assert norms[key] == pytest.approx(torch.linalg.norm(s * b0 @ a0).item(), rel=1e-5)
    assert tr.displacement(start, start, s) == pytest.approx(dict.fromkeys(keys, 0.0), abs=1e-4)
    grouped = tr.by_module(norms)
    assert sorted(grouped) == ["q_proj", "v_proj"]
    assert grouped["q_proj"] == pytest.approx(
        (norms["layers.0.self_attn.q_proj"] + norms["layers.1.self_attn.q_proj"]) / 2
    )


def test_fomaml_selects_on_post_k_exact_match_then_mean_s():
    def rec(exact, nll, ce):
        decode = {"exact_match": exact, "nll_sum": nll}
        return {"kind": "val", "ce": {"current": ce}, "adapted": {"decode": decode}}

    a, b, c, d = rec(0.1, 5.0, 0.5), rec(0.2, 9.0, 0.9), rec(0.2, 4.0, 0.9), rec(0.2, 4.0, 0.1)
    assert tr.is_better(a, None, "fomaml")
    assert tr.is_better(b, a, "fomaml") and not tr.is_better(a, b, "fomaml")
    assert tr.is_better(c, b, "fomaml")  # tie on exact match, lower S
    assert not tr.is_better(d, c, "fomaml")  # an exact tie keeps the earlier point
    assert tr.best_validation([a, b, c, d], "fomaml") == c
    assert tr.best_validation([a, b, c, d]) == d  # zero_step: the lowest current CE
    # k = 0 under fomaml selects on the 0-step decode metrics of the current H
    e = {"kind": "val", "ce": {"current": 1.0}, "decode": {"current": a["adapted"]["decode"]}}
    assert tr.selection_metrics(e) == a["adapted"]["decode"]


# --- the reviewer's fixes and the pilot's probe (P3.9, second part) -------------------------------


def test_fresh_schedule_and_inner_probe_need_a_warm_start(tmp_path):
    phase = fomaml_cfg(tmp_path / "a", gpus=1, fresh_schedule=True)
    with pytest.raises(ValueError, match="warm start's start step"):
        tr.train(phase)
    probe = tr.InnerProbe(lrs=(5e-3, 1e-2), margin=0)
    probed = fomaml_cfg(tmp_path / "b", gpus=1, inner_probe=probe)
    with pytest.raises(ValueError, match="warm start's start step"):
        tr._Worker(0, 1, probed, None, time.time())
    assert not (tmp_path / "a" / "logs").exists()  # refused before anything is written
    with pytest.raises(ValueError, match="exclude each other"):
        tr.check_start(phase, "x", "y")
    tr.check_start(phase, None, "ckpt")  # a warm start, or a resume of one, is allowed
    tr.check_start(phase, "ckpt", None)
    # the probe's settings: rates ascending and distinct, inner.lr among them, an inner step
    for bad in ({"lrs": (1e-2,)}, {"lrs": (1e-2, 5e-3)}, {"lrs": (1e-2, 1e-2)}, {"margin": -1}):
        with pytest.raises(ValueError, match="inner_probe"):
            dataclasses.replace(probe, **bad)
    with pytest.raises(ValueError, match="one of inner_probe.lrs"):
        dataclasses.replace(probed, inner_probe=tr.InnerProbe(lrs=(1e-3, 2e-3), margin=0))
    with pytest.raises(ValueError, match="inner.steps >= 1"):
        dataclasses.replace(probed, inner=inner(steps=0))
    data = probed.to_dict()
    assert data["inner_probe"] == {"lrs": [5e-3, 1e-2], "margin": 0}
    assert tr.TrainConfig.from_dict(data) == probed
    assert tr.resume_conflicts(data, dataclasses.replace(probed, inner_probe=None)) == [
        "inner_probe"
    ]


def test_the_inner_clip_is_torchs_clip_grad_norm():
    torch.manual_seed(0)
    grads = [torch.randn(3, 4), torch.randn(5, 3)]
    params = [torch.zeros_like(g, requires_grad=True) for g in grads]
    for param, grad in zip(params, grads, strict=True):
        param.grad = grad.clone()
    expected = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    ours = [g.clone() for g in grads]
    assert tr.clip_inner_grads(ours, 1.0) == pytest.approx(expected, rel=1e-6)
    for a, param in zip(ours, params, strict=True):
        torch.testing.assert_close(a, param.grad, rtol=1e-6, atol=0)
    total = math.sqrt(sum(float(a.square().sum()) for a in ours))
    assert total == pytest.approx(1.0, rel=1e-5) and expected > 1.0
    # None and a norm under the clip leave the gradients as they are
    unclipped = [g.clone() for g in grads]
    assert tr.clip_inner_grads(unclipped, None) == pytest.approx(expected, rel=1e-6)
    assert all(torch.equal(a, g) for a, g in zip(unclipped, grads, strict=True))
    small = [g / 100 for g in grads]
    kept = [g.clone() for g in small]
    tr.clip_inner_grads(kept, 1.0)
    assert all(torch.equal(a, g) for a, g in zip(kept, small, strict=True))


def test_the_clip_reaches_the_inner_update(tmp_path):
    """sgd moves by the clipped gradient; Adam's first step is the gradient's sign at any clip."""
    w = worker(no_dropout(fomaml_cfg(tmp_path, gpus=1)))
    w._train_modes()
    record = w.train_records[0]
    seq = tr.build_sequence(record, w.table, w.formatter, 8192)
    with torch.no_grad():
        start = w.hypernet(w.embeddings.z_bar(seq.demo_hashes)[None])[0]
    demos = w.inner_demos(record)[:1]
    s = w.cfg.hypernet.scaling

    def moved(settings):
        end, _, norms = tr.adapt(w.model, start, demos, settings, s, w.device)
        pairs = [(e, b) for k in start for e, b in zip(end[k], start[k], strict=True)]
        step = torch.cat([(e - b).flatten() for e, b in pairs])
        return step, norms[0]

    full, norm = moved(inner(steps=1, lr=1.0, optimizer="sgd", clip=None))
    tau = norm / 4
    clipped, _ = moved(inner(steps=1, lr=1.0, optimizer="sgd", clip=tau))
    torch.testing.assert_close(clipped, full * (tau / (norm + 1e-6)), rtol=1e-4, atol=1e-7)
    # up to epsilon: with eps 1e-8 the coordinates whose gradient is near 1e-8 differ, so the
    # check takes a negligible one
    adam_full, _ = moved(inner(steps=1, lr=1e-2, clip=None, eps=1e-30))
    adam_clipped, _ = moved(inner(steps=1, lr=1e-2, clip=tau, eps=1e-30))
    torch.testing.assert_close(adam_clipped, adam_full, rtol=1e-5, atol=1e-9)


def test_choose_inner_lr_on_hand_cases():
    def c(exact, nll, ce=0.1):
        return {"exact": exact, "nll_sum": nll, "ce": ce}

    # the most exact matches wins when no other is within the margin
    assert tr.choose_inner_lr([c(10, 1.0), c(14, 2.0), c(11, 0.5)], margin=2) == 1
    # within the margin the lowest mean S wins
    assert tr.choose_inner_lr([c(10, 1.0), c(12, 2.0), c(11, 0.5)], margin=2) == 2
    # an exact tie keeps the smaller rate
    assert tr.choose_inner_lr([c(12, 1.0), c(12, 1.0)], margin=0) == 0
    # margin 0: the exact match alone decides, then S
    assert tr.choose_inner_lr([c(12, 0.1), c(13, 9.0)], margin=0) == 1
    # a non-finite rate never wins, and none finite is refused
    nan = float("nan")
    assert tr.choose_inner_lr([c(10, 1.0), c(40, nan), c(9, 0.9)], margin=2) == 2
    with pytest.raises(ValueError, match="no inner rate"):
        tr.choose_inner_lr([c(1, nan), c(1, 1.0, ce=float("inf"))], margin=2)


def probed_settings():
    return {"gpus": 1, "steps": 8, "eval_every": 4, "ckpt_every": 2, "keep_checkpoints": 5}


def probe_cfg(path, **changes):
    phase = {**probed_settings(), "lr": 2e-3, "min_lr": 1e-4, "warmup_steps": 1, **changes}
    return fomaml_cfg(
        path,
        fresh_schedule=True,
        inner_probe=tr.InnerProbe(lrs=(2.5e-3, 1e-2, 4e-2), margin=1),
        **phase,
    )


def test_a_probed_warm_start_trains_at_the_pick_and_resumes_bitwise(tmp_path):
    settings = probed_settings()
    assert tr.train(toy_cfg(tmp_path / "zero", **{**settings, "steps": 4})) == tr.EXIT_OK
    ckpt = tmp_path / "zero" / "ckpt" / "ckpt-000004.pt"
    cfg = probe_cfg(tmp_path / "p")
    assert tr.train(cfg, init_from=str(ckpt)) == tr.EXIT_OK
    metrics = read_jsonl(tmp_path / "p" / "logs" / "metrics.jsonl")
    start = [r for r in metrics if r["kind"] == "val"][0]
    probe = start["probe"]
    assert start["step"] == 4 and probe["lrs"] == [2.5e-3, 1e-2, 4e-2]
    assert probe["configured_lr"] == 1e-2 and len(probe["results"]) == 3
    pick = tr.choose_inner_lr(probe["scores"], 1)
    chosen = probe["lrs"][pick]
    assert probe["chosen"] == chosen
    # the scores are the blocks' own, and the start record's adapted block is the pick's
    for score, block in zip(probe["scores"], probe["results"], strict=True):
        assert score["exact"] == sum(block["per_episode_exact"])
        assert score["ce"] == pytest.approx(block["ce"])
        assert score["nll_sum"] == pytest.approx(block["decode"]["nll_sum"], rel=1e-9)
    assert start["adapted"]["per_episode"] == probe["results"][pick]["per_episode"]
    assert len({tuple(b["per_episode"]) for b in probe["results"]}) == 3  # the rates differ
    # training runs at the pick: the steps' inner blocks, the checkpoints' config, run.json
    assert {r["inner"]["lr"] for r in metrics if r["kind"] == "train"} == {chosen}
    first = tr.load_checkpoint(tmp_path / "p" / "ckpt" / "ckpt-000004.pt")  # after the probe
    assert first["config"]["inner"]["lr"] == chosen
    assert first["init_from"]["inner_probe"]["chosen"] == chosen
    info = json.loads((tmp_path / "p" / "logs" / "run.json").read_text())
    assert info["init_from"]["inner_probe"]["chosen"] == chosen
    # a probe-free run at the picked rate scores the start step as the probe did
    plain = toy_cfg(
        tmp_path / "q",
        objective="fomaml",
        fresh_schedule=True,
        inner=inner(lr=chosen),
        **{**probed_settings(), "lr": 2e-3, "min_lr": 1e-4, "warmup_steps": 1},
    )
    assert tr.train(plain, init_from=str(ckpt)) == tr.EXIT_OK
    plain_start = [
        r for r in read_jsonl(tmp_path / "q" / "logs" / "metrics.jsonl") if r["kind"] == "val"
    ][0]
    assert plain_start["adapted"]["per_episode"] == start["adapted"]["per_episode"]
    assert_same_state(
        h_state(tmp_path / "p" / "ckpt" / "ckpt-000008.pt"),
        h_state(tmp_path / "q" / "ckpt" / "ckpt-000008.pt"),
    )
    # a restart resumes from the start step's checkpoint at the pick, without probing again
    again = probe_cfg(tmp_path / "r")
    assert tr.train(again, resume=str(tmp_path / "p" / "ckpt" / "ckpt-000004.pt")) == 0
    assert_same_state(
        h_state(tmp_path / "p" / "ckpt" / "ckpt-000008.pt"),
        h_state(tmp_path / "r" / "ckpt" / "ckpt-000008.pt"),
    )
    resumed = read_jsonl(tmp_path / "r" / "logs" / "metrics.jsonl")
    assert sum(1 for r in resumed if "probe" in r) == 1  # the history's, not a second probe
    # a resume whose config drops the probe sees the configured rate and is refused
    with pytest.raises(ValueError, match="inner"):
        tr.train(
            dataclasses.replace(again, inner_probe=None),
            resume=str(tmp_path / "p" / "ckpt" / "ckpt-000006.pt"),
        )
    # the clock's validation mean leaves the other rates out
    assert probe["seconds"] > 0


def test_a_new_phase_starts_from_a_phases_checkpoint(tmp_path):
    settings = probed_settings()
    assert tr.train(toy_cfg(tmp_path / "zero", **{**settings, "steps": 4})) == tr.EXIT_OK
    first = probe_cfg(tmp_path / "p")
    assert tr.train(first, init_from=str(tmp_path / "zero" / "ckpt" / "ckpt-000004.pt")) == 0
    phase_ckpt = tmp_path / "p" / "ckpt" / "ckpt-000008.pt"
    # a longer run: steps 9 to 12 as a new phase, the probe again at its start step
    longer = probe_cfg(tmp_path / "l", steps=12, warmup_steps=1)
    assert tr.train(longer, init_from=str(phase_ckpt)) == tr.EXIT_OK
    info = tr.load_checkpoint(tmp_path / "l" / "ckpt" / "ckpt-000012.pt")["init_from"]
    assert info["step"] == 8 and info["fresh_schedule"]["phase_steps"] == 4
    assert info["previous"]["fresh_schedule"]["start"] == 4  # the earlier phase, kept
    steps = train_records(tmp_path / "l" / "logs" / "metrics.jsonl")
    assert [r["step"] for r in steps] == [9, 10, 11, 12]
    start = [r for r in read_jsonl(tmp_path / "l" / "logs" / "metrics.jsonl") if r["kind"] == "val"]
    end = [r for r in read_jsonl(tmp_path / "p" / "logs" / "metrics.jsonl") if r["kind"] == "val"]
    assert start[0]["step"] == 8 and start[0]["per_episode"] == end[-1]["per_episode"]
    # without a new phase a warm-started checkpoint is refused
    same = dataclasses.replace(probe_cfg(tmp_path / "x"), fresh_schedule=False, inner_probe=None)
    with pytest.raises(ValueError, match="itself warm-started"):
        tr.train(same, init_from=str(phase_ckpt))


def test_a_fomaml_best_pt_loads_as_p3_7s_load_check_loads_it(tmp_path):
    """P3.7's ``load_check`` (task/P3.7, arc26/hyper/validation.py) on a ``fomaml`` best.pt.

    It reads ``payload["config"]``, compares its ``hypernet`` and ``manifest_sha256``, builds
    ``TrainConfig.from_dict({**config, **overrides})``, loads H strictly, and compares H's 0-step
    validation CE per episode with ``metrics.jsonl`` at ``status.json``'s best step. The config
    must parse on the code before P3.9 too, whose ``from_dict`` refuses unknown keys.
    """
    settings = probed_settings()
    assert tr.train(toy_cfg(tmp_path / "zero", **{**settings, "steps": 4})) == tr.EXIT_OK
    cfg = probe_cfg(tmp_path / "p")
    assert tr.train(cfg, init_from=str(tmp_path / "zero" / "ckpt" / "ckpt-000004.pt")) == 0
    payload = tr.load_checkpoint(tmp_path / "p" / "ckpt" / tr.BEST_FILE)
    config = dict(payload["config"])
    # the keys of the code before P3.9: every field but the objective keys, none unknown
    before = {f.name for f in dataclasses.fields(tr.TrainConfig)} - set(tr.OBJECTIVE_KEYS)
    required = {
        f.name for f in dataclasses.fields(tr.TrainConfig) if f.default is dataclasses.MISSING
    }
    assert set(config) <= before and required <= set(config)
    assert payload["objective"]["objective"] == "fomaml"
    assert payload["objective"]["fresh_schedule"] is True
    assert payload["objective"]["inner"]["lr"] in cfg.inner_probe.lrs  # the probe's pick
    status = tr.read_status(tmp_path / "p" / "logs")
    assert payload["step"] == status["best"]["step"]
    logged = [
        r
        for r in read_jsonl(tmp_path / "p" / "logs" / "metrics.jsonl")
        if r["kind"] == "val" and r["step"] == status["best"]["step"]
    ][-1]
    overrides = {"out_dir": str(tmp_path / "lc"), "ckpt_dir": str(tmp_path / "lc"), "gpus": 1}
    loaded = tr.TrainConfig.from_dict({**config, **overrides})
    assert loaded.objective == "zero_step" and loaded.hypernet == cfg.hypernet
    w = worker(loaded)
    w.hypernet.load_state_dict(payload["hypernet"])  # strict
    w.hypernet.eval()
    for i, seq in enumerate(w.val):
        ce = w.episode_ce(seq, w.val_z[i])
        assert ce == pytest.approx(logged["per_episode"]["current"][i], abs=2e-6)
