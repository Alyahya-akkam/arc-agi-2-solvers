"""P3.5: the meta-training loop (arc26/hyper/train.py) and scripts/train_hypernet.py, on the CPU.

The base is the tiny random Qwen3 and the tokenizer the stand-in with the notebook's 16 ids, both
built in code by arc26.hyper.train; nothing is downloaded. The episodes are the fixture under
tests/testdata/train_hypernet/: a P3.4 build in the real format (MANIFEST.json, JSONL shards) with a
passing check report, 12 toy and 6 validation episodes of small grids under simple rules, written
by ``write_fixture`` below. After a deliberate change, regenerate it with

    .venv/bin/python -c "import sys; sys.path.insert(0, 'tests'); import test_train as t; \
        print(t.write_fixture(t.FIXTURE))"

and put the printed sha256 into the smoke ``manifest_sha256`` of the three configs (``TOY``,
``TOY_02``, ``FULL``). Without torch, transformers, or safetensors the module is skipped (CI's
main job). Run with CUDA_VISIBLE_DEVICES="" (BRIEF A0).
"""

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from arc26.hyper.episodes import EpisodeRecord, PermutationTable, write_shards  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "testdata" / "train_hypernet"
TOY = REPO / "configs" / "train_hypernet_toy.yaml"
# Run P3.5-02: P3.5-01's toy run with half the steps (its brief's one change for a faded gain).
TOY_02 = REPO / "configs" / "train_hypernet_toy_P3.5-02.yaml"
FULL = REPO / "configs" / "train_hypernet.yaml"
HYPERNET = REPO / "configs" / "hypernet.yaml"

# (source, task id, rule, grid shape): one family per task. Colors 0 to 3 keep the rules simple.
FIXTURE_RULES = {
    "flip": np.fliplr,
    "transpose": np.transpose,
    "recolor": lambda x: (x + 1) % 4,
    "rot90": np.rot90,
    "flipud": np.flipud,
    "invert": lambda x: 3 - x,
}
FIXTURE_TOY = (
    ("barc", "barc:fixture:flip", "flip", (3, 3)),
    ("rearc", "fixture-transpose", "transpose", (3, 4)),
    ("arc2train", "fixture-recolor", "recolor", (2, 3)),
)
FIXTURE_VAL = (
    ("rearc", "fixture-rot90", "rot90", (3, 3)),
    ("barc", "barc:fixture:flipud", "flipud", (2, 4)),
    ("barc", "barc:fixture:invert", "invert", (3, 2)),
)


def write_fixture(root):
    """Write the fixture build and its check report under ``root``; return the manifest sha256."""
    rng = random.Random(0)
    table = PermutationTable.seeded(4, seed=0)

    def episodes(spec, split, count):
        records = []
        for source, task_id, rule, shape in spec:
            pairs = []
            for _ in range(5):
                x = np.array([[rng.randrange(4) for _ in range(shape[1])] for _ in range(shape[0])])
                pairs.append((x.astype(np.uint8), FIXTURE_RULES[rule](x).astype(np.uint8)))
            for e in range(count):
                query = e % len(pairs)
                demos = [i for i in range(len(pairs)) if i != query][e % 2 : e % 2 + 2]
                records.append(
                    EpisodeRecord.create(
                        episode_id=f"{source}/{task_id}/{split}/{e}",
                        source=source,
                        task_id=task_id,
                        family=task_id,
                        split=split,
                        augmentation_id=rng.randrange(table.n_augmentations),
                        demos=[pairs[i] for i in demos],
                        queries=[pairs[query]],
                    )
                )
        return records

    root = Path(root)
    build, check = root / "episode_build", root / "check" / "run_logs"
    toy, val = episodes(FIXTURE_TOY, "toy", 4), episodes(FIXTURE_VAL, "val", 2)
    files = write_shards(toy, build, "toy", 1) + write_shards(val, build, "val", 1)
    manifest = {
        "schema": 1,
        "task": "P3.4",
        "code": {"commit": "fixture", "dirty": False},
        "note": "synthetic fixture of tests/test_train.py: small grids under simple rules",
        "augmentation": {
            "id": "8 * j + 4 * t + k",
            "ops": "transpose if t, then np.rot90 k times, then color c -> permutations[j][c]",
            "permutations": table.to_strings(),
        },
        "max_tokens": 8192,
        "counts": {"toy": len(toy), "val": len(val), "train": 0, "val_puzzles": 3},
        "files": files,
    }
    data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    (build / "MANIFEST.json").write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    report = {
        "schema": 1,
        "pass": True,
        "manifest_sha256": sha,
        "note": "synthetic: no evaluation data exists for the fixture; the pass lets it load",
    }
    check.mkdir(parents=True, exist_ok=True)
    (check / "leakage.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return sha


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.encoder import EmbeddingCache, view_descriptors, view_set_id  # noqa: E402
from arc26.hyper.episodes import RecordError  # noqa: E402
from arc26.hyper.hypernet import HypernetConfig  # noqa: E402
from arc26.hyper.inject import inject  # noqa: E402
from arc26.nvarc.arc_loader import QwenFormatter, convert_grid_to_string  # noqa: E402


def load_script(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script = load_script("train_hypernet")
kaggle_ops = load_script("kaggle_ops")
SHA = hashlib.sha256((FIXTURE / "episode_build" / "MANIFEST.json").read_bytes()).hexdigest()


def smoke_cfg(tmp_path, **changes):
    """The toy config's smoke settings with run outputs under ``tmp_path``."""
    cfg = script.load_config(
        TOY, True, {"out_dir": str(tmp_path / "logs"), "ckpt_dir": str(tmp_path / "ckpt")}
    )
    return dataclasses.replace(cfg, **changes)


def worker(cfg):
    w = tr._Worker(0, 1, cfg, None, time.time())
    w.setup()
    return w


def fixture_set():
    return tr.open_episode_set(FIXTURE / "episode_build", SHA, FIXTURE / "check", 8192)


def records(split):
    return tr.load_split(fixture_set(), split, 8192)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def h_state(path):
    return tr.load_checkpoint(Path(path))["hypernet"]


def assert_same_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def trajectory(metrics):
    """What must match between two runs of one config: data order, losses, validation CEs."""
    return [
        (r["kind"], r["step"], r.get("episodes"), r.get("loss"), r.get("ce"))
        for r in metrics
        if r["kind"] in ("train", "val")
    ]


def write_config(tmp_path, **smoke):
    """A copy of the toy config whose smoke section takes ``smoke`` (nested sections merge)."""
    raw = yaml.safe_load(TOY.read_text())
    for key, value in smoke.items():
        if isinstance(value, dict):
            raw["smoke"].setdefault(key, {}).update(value)
        else:
            raw["smoke"][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


# --- the fixture and the configs -----------------------------------------------------------------


def test_committed_fixture_matches_its_generator(tmp_path):
    sha = write_fixture(tmp_path)
    committed = {p.relative_to(FIXTURE) for p in FIXTURE.rglob("*") if p.is_file()}
    rebuilt = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    assert committed == rebuilt
    for rel in committed:
        assert (FIXTURE / rel).read_bytes() == (tmp_path / rel).read_bytes(), rel
    assert sha == SHA
    for path in (TOY, TOY_02, FULL):
        assert yaml.safe_load(path.read_text())["smoke"]["manifest_sha256"] == SHA


def test_configs_validate_in_both_modes_and_copy_configs_hypernet_yaml():
    expected = HypernetConfig.from_yaml(HYPERNET)
    for path in (TOY, TOY_02, FULL):
        cfg = script.load_config(path, False, {})
        assert cfg.hypernet == expected, path.name
        assert cfg.manifest_sha256.startswith("b1f2cd04")
        assert (cfg.device, cfg.dtype, cfg.val_split, cfg.max_tokens) == (
            "cuda", "bfloat16", "val", 8192,
        )  # fmt: skip
        assert cfg.encoder == tr.EncoderSettings(24, "output", 2, 2, 16384, 256)
        smoke = script.load_config(path, True, {})
        assert (smoke.device, smoke.gpus, smoke.model_dir) == ("cpu", 2, None)
        assert Path(smoke.episodes_root).is_absolute() and Path(smoke.episodes_root).is_dir()
    toy = script.load_config(TOY, False, {})
    assert (toy.train_split, toy.gpus, toy.accum, toy.steps) == ("toy", 4, 1, 2000)
    assert toy.kill.max_projected_s < toy.max_wall_s
    full = script.load_config(FULL, False, {})
    assert (full.train_split, full.gpus, full.accum) == ("train", 4, 8)


def test_config_errors_are_reported(tmp_path, capsys):
    bad = write_config(tmp_path, kl_weight=0.5)
    assert script.main(["--config", str(bad), "--smoke"]) == tr.EXIT_CONFIG
    assert "kl_weight must be 0" in capsys.readouterr().err
    with pytest.raises(script.ConfigError, match="smoke.nonsense"):
        script.load_config(write_config(tmp_path, nonsense=1), True, {})
    with pytest.raises(script.ConfigError, match="penalty_weight"):
        script.load_config(write_config(tmp_path, penalty="l2"), True, {})
    assert script.main(["--config", str(TOY), "--smoke", "--model-dir", "x"]) == tr.EXIT_CONFIG


# --- sequences and the loss mask -----------------------------------------------------------------


@pytest.mark.parametrize("include_eos", [False, True], ids=["grid", "grid_and_eos"])
def test_loss_mask_covers_exactly_the_query_output_tokens(include_eos):
    """The query's grid tokens (spans decode back to the grid text, as P3.2 checks), plus the
    closing <|im_end|> with loss_includes_eos, the notebook's completion-only targets."""
    tokenizer = tr.standin_tokenizer()
    formatter = QwenFormatter(tokenizer)
    table = fixture_set().table
    for record in records("toy") + records("val"):
        seq = tr.build_sequence(record, table, formatter, 8192, include_eos)
        # a(D) then a(Q) from P3.4's numpy path, independent of the notebook's forward_mod that
        # build_sequence uses: every pair of D and Q under the record's own augmentation.
        aug = table.augmentation(record.augmentation_id)
        pairs = [{"input": aug.apply(x), "output": aug.apply(y)} for x, y in record.pairs]
        ids = seq.input_ids.tolist()
        assert ids == list(tokenizer.encode(formatter.fmt_train(pairs)))
        output = pairs[-1]["output"]
        rows, cols = output.shape
        grid = range(seq.target.start, seq.target.stop - include_eos)
        assert tokenizer.decode(ids[grid.start : grid.stop]).replace(" ", "") == (
            convert_grid_to_string(output)
        )
        assert len(seq.target) == rows * cols + rows - 1 + include_eos
        # assistant, newline, then the grid; the closing <|im_end|> ends the sequence
        assert ids[seq.target.start - 2 : seq.target.start] == [12, 10]
        assert grid.stop == len(ids) - 1 and ids[-1] == 15
        assert seq.target.stop == len(ids) - (not include_eos)
        assert len(ids) <= record.tokens


def test_loss_includes_eos_is_the_default_and_a_resume_setting(tmp_path):
    cfg = smoke_cfg(tmp_path)
    data = cfg.to_dict()
    del data["loss_includes_eos"]
    assert tr.TrainConfig.from_dict(data).loss_includes_eos is True
    record = records("toy")[0]
    formatter = QwenFormatter(tr.standin_tokenizer())
    assert tr.build_sequence(record, fixture_set().table, formatter, 8192).target == (
        tr.build_sequence(record, fixture_set().table, formatter, 8192, True).target
    )
    saved = dataclasses.replace(cfg, loss_includes_eos=False).to_dict()
    assert tr.resume_conflicts(saved, cfg) == ["loss_includes_eos"]
    with pytest.raises(ValueError, match="loss_includes_eos must be true or false"):
        dataclasses.replace(cfg, loss_includes_eos=1)


def test_query_ce_averages_the_target_predictions_only():
    torch.manual_seed(0)
    logits = torch.randn(12, 16)
    ids = torch.randint(0, 16, (12,))
    target = range(7, 11)
    expected = -sum(F.log_softmax(logits[t - 1], -1)[ids[t]] for t in target) / len(target)
    torch.testing.assert_close(tr.query_ce(logits, ids, target), expected)
    poisoned = logits.clone()
    poisoned[:6] = float("nan")  # predictions outside O_Q never enter the loss
    poisoned[10:] = float("nan")
    torch.testing.assert_close(tr.query_ce(poisoned, ids, target), expected)


def test_over_length_episodes_are_refused_not_cut():
    formatter = QwenFormatter(tr.standin_tokenizer())
    record = records("toy")[0]
    with pytest.raises(ValueError, match="above max_tokens"):
        tr.build_sequence(record, fixture_set().table, formatter, 40)
    with pytest.raises(RecordError, match="above the limit"):
        tr.load_split(fixture_set(), "toy", 40)


# --- the manifest and check guard ----------------------------------------------------------------


@pytest.fixture
def fixture_copy(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "fx")
    return tmp_path / "fx" / "episode_build", tmp_path / "fx" / "check"


def test_guard_refuses_another_manifest(fixture_copy):
    build, check = fixture_copy
    with pytest.raises(tr.EpisodeSetError, match="has manifest sha256"):
        tr.open_episode_set(build, "0" * 64, check, 8192)
    with pytest.raises(tr.EpisodeSetError, match="no P3.4 MANIFEST.json"):
        tr.open_episode_set(check, SHA, check, 8192)


def test_guard_refuses_a_missing_or_failed_check_report(fixture_copy, tmp_path):
    build, check = fixture_copy
    with pytest.raises(tr.EpisodeSetError, match="no leakage.json"):
        tr.open_episode_set(build, SHA, tmp_path / "empty", 8192)
    other = {"pass": True, "manifest_sha256": "1" * 64}
    (check / "run_logs" / "leakage.json").write_text(json.dumps(other))
    with pytest.raises(tr.EpisodeSetError, match="no leakage.json"):
        tr.open_episode_set(build, SHA, check, 8192)
    failed = {"pass": False, "manifest_sha256": SHA}
    (check / "run_logs" / "leakage.json").write_text(json.dumps(failed))
    with pytest.raises(tr.EpisodeSetError, match="failed"):
        tr.open_episode_set(build, SHA, check, 8192)
    # a failed check of the same manifest refuses the build even beside a passing one
    shutil.copy(FIXTURE / "check" / "run_logs" / "leakage.json", check / "passed.json")
    (check / "second").mkdir()
    shutil.copy(FIXTURE / "check" / "run_logs" / "leakage.json", check / "second" / "leakage.json")
    with pytest.raises(tr.EpisodeSetError, match="failed"):
        tr.open_episode_set(build, SHA, check, 8192)


@pytest.mark.parametrize("value", ["absent", "true", 1, None, False])
def test_guard_refuses_a_report_whose_pass_is_not_true(fixture_copy, value):
    build, check = fixture_copy
    report = {"manifest_sha256": SHA} | ({} if value == "absent" else {"pass": value})
    (check / "run_logs" / "leakage.json").write_text(json.dumps(report))
    with pytest.raises(tr.EpisodeSetError, match="failed"):
        tr.open_episode_set(build, SHA, check, 8192)


def test_guard_refuses_a_tampered_shard_and_a_larger_cap(fixture_copy):
    build, check = fixture_copy
    episode_set = tr.open_episode_set(build, SHA, check, 8192)
    shard = build / "episodes" / "toy-00000-of-00001.jsonl"
    shard.write_bytes(shard.read_bytes().replace(b'"augmentation_id":5', b'"augmentation_id":6'))
    with pytest.raises(tr.EpisodeSetError, match="differs from the manifest"):
        tr.load_split(episode_set, "toy", 8192)
    assert len(tr.load_split(episode_set, "val", 8192)) == 6
    with pytest.raises(tr.EpisodeSetError, match="keeps episodes up to 8192"):
        tr.open_episode_set(build, SHA, check, 4096)


@pytest.mark.parametrize("gpus", [1, 2])
def test_script_exits_1_with_logs_when_the_guard_refuses(tmp_path, gpus):
    """With gpus 2 the worker's error arrives as a ProcessRaisedException whose message starts
    with a blank line; status.json and summary.md still name the worker's own exception."""
    config = write_config(tmp_path, manifest_sha256="0" * 64, gpus=gpus)
    out = tmp_path / "out"
    code = script.main(["--config", str(config), "--smoke", "--out-dir", str(out)])
    assert code == tr.EXIT_FAILED
    assert "EpisodeSetError" in (out / "error.log").read_text()
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "failed" and status["step"] is None
    reason = status["reason"]
    assert "EpisodeSetError: no P3.4 build under " in reason and "0" * 64 in reason
    if gpus == 2:
        assert reason.startswith("ProcessRaisedException: arc26.hyper.train.EpisodeSetError: ")
    summary = (out / "summary.md").read_text()
    assert f"**failed** (exit 1): {reason}" in summary and "step n/a of" in summary


def test_failure_reason_keeps_the_type_and_the_last_line():
    assert script.failure_reason(ValueError("one line")) == "ValueError: one line"
    assert script.failure_reason(KeyError()) == "KeyError"
    err = RuntimeError("\n\n-- Process 1 terminated:\nTraceback\n  x\nmod.Error: what failed\n\n")
    assert script.failure_reason(err) == "RuntimeError: mod.Error: what failed"


# --- z_bar, step 0, gradients --------------------------------------------------------------------


def test_z_bar_ignores_the_augmentation(tmp_path):
    w = worker(smoke_cfg(tmp_path, gpus=1))
    record = w.train_records[0]
    other = dataclasses.replace(
        record, augmentation_id=(record.augmentation_id + 9) % w.table.n_augmentations
    )
    first = tr.build_sequence(record, w.table, w.formatter, 8192)
    second = tr.build_sequence(other, w.table, w.formatter, 8192)
    assert not torch.equal(first.input_ids, second.input_ids)  # the base sees a(D), a(Q)
    assert first.demo_hashes == second.demo_hashes
    assert torch.equal(
        w.embeddings.z_bar(first.demo_hashes), w.embeddings.z_bar(second.demo_hashes)
    )
    # z_bar is the mean over D of P3.2's per-pair view means of the un-augmented demonstrations
    pairs = [{"input": x, "output": y} for x, y in record.demos]
    direct = w.encoder.encode_pairs(pairs).mean(dim=0)
    torch.testing.assert_close(w.embeddings.z_bar(first.demo_hashes), direct)


def test_step_0_equals_the_base_bitwise(tmp_path):
    w = worker(smoke_cfg(tmp_path, gpus=1))
    scaling = w.cfg.hypernet.scaling
    for train_mode in (False, True):
        w.hypernet.train(train_mode)
        for seq, z_bar in zip(w.val, w.val_z, strict=True):
            ids = seq.input_ids[None]
            with torch.no_grad():
                base = w.model(input_ids=ids).logits
                with inject(w.model, w.hypernet(z_bar[None])[0], scaling):
                    adapted = w.model(input_ids=ids).logits
            assert torch.equal(base, adapted)
    w.evaluate()
    record = w.metrics.history[-1]
    assert record["step"] == 0
    assert record["per_episode"]["current"] == record["per_episode"]["zero"]
    assert record["per_episode"]["mismatched"] == record["per_episode"]["zero"]


def test_gradients_reach_h_and_only_h(tmp_path):
    w = worker(smoke_cfg(tmp_path, gpus=1))
    base = {key: value.clone() for key, value in w.model.state_dict().items()}
    assert w.train_step() == (0, "")  # one update, so that B is no longer zero
    w._train_modes()
    seq = tr.build_sequence(w.train_records[1], w.table, w.formatter, 8192)
    assert not w.embeddings.z_bar(seq.demo_hashes).requires_grad
    w.micro_step(seq, 0)
    for name, param in w.hypernet.named_parameters():
        assert param.grad is not None and param.grad.abs().max() > 0, name
    for param in w.model.parameters():
        assert not param.requires_grad and param.grad is None
    w.optimizer.step()
    for key, value in w.model.state_dict().items():
        assert torch.equal(value, base[key]), key


def test_checkpointing_on_and_off_give_equal_gradients(tmp_path):
    grads, calls = [], []
    for checkpointing in (False, True):
        cfg = smoke_cfg(tmp_path / str(checkpointing), gpus=1, gradient_checkpointing=checkpointing)
        w = worker(cfg)
        assert w.train_step() == (0, "")
        q_proj = w.model.get_submodule(f"model.{w.keys[0]}")
        count = []
        handle = q_proj.register_forward_pre_hook(lambda *_, count=count: count.append(1))
        w._train_modes()
        w.micro_step(tr.build_sequence(w.train_records[1], w.table, w.formatter, 8192), 0)
        handle.remove()
        calls.append(len(count))
        grads.append({n: p.grad.clone() for n, p in w.hypernet.named_parameters()})
    assert calls == [1, 2]  # with checkpointing the layer runs again during backward
    for name in grads[0]:
        assert torch.equal(grads[0][name], grads[1][name]), name


# --- data order, controls, schedule --------------------------------------------------------------


def test_episode_order_is_a_pure_function_and_each_pass_a_permutation():
    order = tr.EpisodeOrder(7, seed=0)
    passes = [[order(e * 7 + i) for i in range(7)] for e in range(3)]
    assert all(sorted(p) == list(range(7)) for p in passes)
    assert passes[0] != passes[1]
    assert [tr.EpisodeOrder(7, seed=0)(g) for g in (20, 3, 15)] == [order(20), order(3), order(15)]
    two_ranks = [tr.train_slot(s, 0, k, 1, 2) for s in range(3) for k in range(2)]
    accumulate = [tr.train_slot(s, m, 0, 2, 1) for s in range(3) for m in range(2)]
    assert two_ranks == accumulate == list(range(6))


def test_mismatch_permutation_pairs_other_families():
    families = ["a"] * 3 + ["b"] * 2 + ["c"] * 4 + ["d"]
    perm = tr.mismatch_permutation(families, seed=0)
    assert sorted(perm) == list(range(len(families)))
    assert all(families[perm[i]] != families[i] for i in range(len(families)))
    assert perm == tr.mismatch_permutation(families, seed=0)
    with pytest.raises(ValueError, match="below half"):
        tr.mismatch_permutation(["a", "a", "a", "b"], seed=0)


def test_warmup_cosine_schedule():
    schedule = tr.WarmupCosine(warmup=4, total=20, min_ratio=0.01)
    assert [schedule(s) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]
    values = [schedule(s) for s in range(4, 20)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert schedule(19) == pytest.approx(0.01)
    assert tr.WarmupCosine(0, 10, 0.0)(0) == pytest.approx(0.5 * (1 + math.cos(math.pi / 10)))


def test_param_groups_spare_norms_biases_and_codes(tmp_path):
    w = worker(smoke_cfg(tmp_path, gpus=1))
    names = {id(p): n for n, p in w.hypernet.named_parameters()}
    decay, keep = (
        {names[id(p)] for p in group["params"]} for group in tr.param_groups(w.hypernet, 0.01)
    )
    assert decay and keep and decay.isdisjoint(keep)
    assert decay | keep == set(names.values())
    assert all(n.endswith("weight") and "norm" not in n and "embedding" not in n for n in decay)
    assert {"layer_embedding.weight", "module_embedding.weight", "heads.q_proj.bias"} <= keep


def test_factor_penalty():
    factors = {"layers.0.self_attn.q_proj": (torch.full((2, 3), -2.0), torch.ones(4, 2))}
    assert tr.factor_penalty(factors, "l1").item() == pytest.approx((6 * 2 + 8) / 14)
    assert tr.factor_penalty(factors, "l2").item() == pytest.approx((6 * 4 + 8) / 14)
    with pytest.raises(ValueError):
        tr.factor_penalty(factors, "l3")


# --- checkpoints, resume, budget, kill criteria --------------------------------------------------


def test_resume_after_k_steps_equals_an_uninterrupted_run(tmp_path):
    settings = {"gpus": 1, "steps": 6, "eval_every": 3, "ckpt_every": 3, "keep_checkpoints": 3}
    whole = smoke_cfg(tmp_path / "whole", **settings)
    assert tr.train(whole) == tr.EXIT_OK
    resumed = smoke_cfg(tmp_path / "resumed", **settings)
    start = tmp_path / "whole" / "ckpt" / "ckpt-000003.pt"
    assert tr.train(resumed, resume=str(start)) == tr.EXIT_OK
    assert_same_state(
        h_state(tmp_path / "whole" / "ckpt" / "ckpt-000006.pt"),
        h_state(tmp_path / "resumed" / "ckpt" / "ckpt-000006.pt"),
    )
    a = trajectory(read_jsonl(tmp_path / "whole" / "logs" / "metrics.jsonl"))
    b = trajectory(read_jsonl(tmp_path / "resumed" / "logs" / "metrics.jsonl"))
    assert a == b and [r[1] for r in a if r[0] == "train"] == [1, 2, 3, 4, 5, 6]
    # a checkpoint written with other settings is refused
    other = dataclasses.replace(resumed, lr=resumed.lr * 2, out_dir=str(tmp_path / "x"))
    with pytest.raises(ValueError, match="lr"):
        tr.train(other, resume=str(start))


def test_checkpoints_are_pruned_and_latest_points_at_the_newest(tmp_path):
    cfg = smoke_cfg(tmp_path, gpus=1, steps=6, ckpt_every=1, eval_every=6, keep_checkpoints=2)
    assert tr.train(cfg) == tr.EXIT_OK
    ckpt = tmp_path / "ckpt"
    assert sorted(p.name for p in ckpt.glob("ckpt-*.pt")) == ["ckpt-000005.pt", "ckpt-000006.pt"]
    assert tr.resolve_checkpoint(ckpt) == ckpt / "ckpt-000006.pt"
    payload = tr.load_checkpoint(ckpt / "ckpt-000006.pt")
    assert payload["step"] == 6 and payload["cursor"] == 6 and payload["world_size"] == 1
    assert {"hypernet", "optimizer", "scheduler", "rng", "history", "zero_ce"} <= payload.keys()
    assert not list(ckpt.glob(".*tmp"))


def test_session_budget_stops_with_a_checkpoint_and_the_rerun_finishes(tmp_path):
    stopped = smoke_cfg(tmp_path / "s", gpus=1, steps=4, max_wall_s=1e-6)
    assert tr.train(stopped) == tr.EXIT_STOPPED
    status = tr.read_status(tmp_path / "s" / "logs")
    assert (status["status"], status["step"]) == ("stopped", 0)
    assert Path(status["checkpoint"]).is_file()
    finished = dataclasses.replace(stopped, max_wall_s=600.0)
    assert tr.train(finished, resume=str(tmp_path / "s" / "ckpt")) == tr.EXIT_OK
    reference = smoke_cfg(tmp_path / "r", gpus=1, steps=4)
    assert tr.train(reference) == tr.EXIT_OK
    assert_same_state(
        h_state(tmp_path / "s" / "ckpt" / "ckpt-000004.pt"),
        h_state(tmp_path / "r" / "ckpt" / "ckpt-000004.pt"),
    )


@pytest.mark.parametrize(
    ("smoke", "reason"),
    [
        ({"kill": {"max_step_s": 1e-9}}, "over kill.max_step_s"),
        ({"kill": {"max_projected_s": 1e-6, "projection_after_steps": 1}}, "projected wall"),
    ],
    ids=["step_time", "projected_wall"],
)
def test_kill_criteria_fire(tmp_path, smoke, reason):
    config = write_config(tmp_path, gpus=1, **smoke)
    out = tmp_path / "out"
    args = ["--config", str(config), "--smoke", "--out-dir", str(out), "--ckpt-dir", str(tmp_path)]
    assert script.main(args) == tr.EXIT_KILLED
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "killed" and reason in status["reason"] and status["step"] == 1
    assert Path(status["checkpoint"]).is_file()
    assert "**killed**" in (out / "summary.md").read_text()


def test_kill_on_a_non_finite_loss(tmp_path, monkeypatch):
    real = tr.query_ce
    monkeypatch.setattr(tr, "query_ce", lambda *args: real(*args) * float("nan"))
    config = write_config(tmp_path, gpus=1)
    out = tmp_path / "out"
    args = ["--config", str(config), "--smoke", "--out-dir", str(out), "--ckpt-dir", str(tmp_path)]
    assert script.main(args) == tr.EXIT_KILLED
    status = json.loads((out / "status.json").read_text())
    assert status["step"] == 0 and "non-finite" in status["reason"]
    payload = tr.load_checkpoint(Path(status["checkpoint"]))
    assert payload["step"] == 0  # the poisoned update was never applied
    assert all(torch.isfinite(v).all() for v in payload["hypernet"].values())


# --- multi-process -------------------------------------------------------------------------------


def test_gloo_ddp_with_two_processes_matches_one_process_with_accumulation_2(tmp_path):
    """DDP averages two ranks' gradients, accumulation adds two halves: equal up to fp32 rounding.

    The episodes and the embeddings are bitwise the same in both runs (one slot order, encode
    chunks independent of the world size), and halving is exact in binary floating point, so the
    two differ at most in the order of fp32 sums. Measured bitwise equal on the dev box; the test
    allows rtol 1e-5 and atol 1e-6 after three AdamW steps, for BLAS builds whose reductions
    depend on the thread count.
    """
    hypernet = dataclasses.replace(smoke_cfg(tmp_path).hypernet, dropout=0.0)
    settings = {"steps": 3, "eval_every": 3, "ckpt_every": 3, "hypernet": hypernet}
    ddp = smoke_cfg(tmp_path / "ddp", gpus=2, accum=1, **settings)
    acc = smoke_cfg(tmp_path / "acc", gpus=1, accum=2, **settings)
    assert tr.train(ddp) == tr.EXIT_OK and tr.train(acc) == tr.EXIT_OK
    a = h_state(tmp_path / "ddp" / "ckpt" / "ckpt-000003.pt")
    b = h_state(tmp_path / "acc" / "ckpt" / "ckpt-000003.pt")
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6, msg=key)
    ma = read_jsonl(tmp_path / "ddp" / "logs" / "metrics.jsonl")
    mb = read_jsonl(tmp_path / "acc" / "logs" / "metrics.jsonl")
    steps_a = [r for r in ma if r["kind"] == "train"]
    steps_b = [r for r in mb if r["kind"] == "train"]
    assert [r["episodes"] for r in steps_a] == [r["episodes"] for r in steps_b]
    for ra, rb in zip(steps_a, steps_b, strict=True):
        assert ra["loss"] == pytest.approx(rb["loss"], rel=1e-5)
    # Validation splits the episodes over the ranks and sums: the records must not depend on it.
    val_a = [r for r in ma if r["kind"] == "val"]
    val_b = [r for r in mb if r["kind"] == "val"]
    assert [r["step"] for r in val_a] == [r["step"] for r in val_b] == [0, 3]
    for ra, rb in zip(val_a, val_b, strict=True):
        assert ra["n"] == rb["n"] == 6
        for ref in tr.REFERENCES:
            assert ra["ce"][ref] == pytest.approx(rb["ce"][ref], rel=1e-6, abs=1e-7)
            assert ra["per_episode"][ref] == pytest.approx(rb["per_episode"][ref], abs=1e-6)
            for source, row in ra["by_source"].items():
                assert row["n"] == rb["by_source"][source]["n"]
                assert row[ref] == pytest.approx(rb["by_source"][source][ref], rel=1e-6)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """The toy config's --smoke (2 processes on gloo), then its rerun with --resume."""
    root = tmp_path_factory.mktemp("smoke")
    args = ["--config", str(TOY), "--smoke", "--out-dir", str(root / "run_logs")]
    args += ["--ckpt-dir", str(root / "checkpoints")]
    began = time.monotonic()
    first = script.main(args)
    seconds = time.monotonic() - began
    snapshot = {
        "metrics": read_jsonl(root / "run_logs" / "metrics.jsonl"),
        "final": h_state(root / "checkpoints" / "ckpt-000024.pt"),
        "timing": {p.name for p in (root / "run_logs").glob("timing-*.jsonl")},
    }
    second = script.main(args + ["--resume", str(root / "checkpoints" / "ckpt-000008.pt")])
    return root, first, seconds, snapshot, second


def test_smoke_exits_0_in_under_two_minutes(smoke_run):
    root, first, seconds, _, _ = smoke_run
    assert first == tr.EXIT_OK
    assert seconds < 120
    print(f"P3.5 smoke: {seconds:.1f} s")


def test_smoke_training_loss_falls_on_the_memorizable_fixture(smoke_run):
    _, _, _, snapshot, _ = smoke_run
    losses = [r["loss"] for r in snapshot["metrics"] if r["kind"] == "train"]
    assert len(losses) == 24
    first, last = sum(losses[:4]) / 4, sum(losses[-4:]) / 4
    assert last < 0.7 * first, (first, last)
    val = [r for r in snapshot["metrics"] if r["kind"] == "val"]
    assert [r["step"] for r in val] == [0, 8, 16, 24]
    assert val[0]["ce"]["current"] == val[0]["ce"]["zero"]
    assert val[-1]["ce"]["current"] < val[-1]["ce"]["zero"]


def test_smoke_writes_the_a8_outputs_under_the_pull_allowlist(smoke_run):
    root, _, _, snapshot, _ = smoke_run
    logs = root / "run_logs"
    expected = {"config.yaml", "run.json", "status.json", "metrics.jsonl", "summary.md"}
    expected |= {"loss_curve.csv", "timing-rank0.jsonl", "timing-rank1.jsonl"}
    names = {p.name for p in logs.iterdir()}
    assert expected <= names
    assert snapshot["timing"] == {"timing-rank0.jsonl", "timing-rank1.jsonl"}
    for name in names:
        assert kaggle_ops.pull_allowed(f"run_logs/{name}"), name
    records = read_jsonl(logs / "timing-rank0.jsonl") + read_jsonl(logs / "timing-rank1.jsonl")
    assert {r["op"] for r in records} == {"encode", "h_forward", "train_step", "validate"}
    assert all(set(r) == {"puzzle_id", "op", "seconds", "tokens"} for r in records)
    info = json.loads((logs / "run.json").read_text())
    assert info["world_size"] == 2 and info["episodes"]["manifest_sha256"] == SHA
    assert info["hypernet"]["emitted_per_adapter"] == 4 * 16 * (128 + 96 + 96 + 128)
    summary = (logs / "summary.md").read_text()
    assert "**complete**" in summary and "Step-0 check" in summary
    assert chr(0x2014) not in summary  # built at runtime: this file passes the dash check


def test_smoke_resume_continues_to_the_same_result(smoke_run):
    root, _, _, snapshot, second = smoke_run
    assert second == tr.EXIT_OK
    assert_same_state(snapshot["final"], h_state(root / "checkpoints" / "ckpt-000024.pt"))
    after = read_jsonl(root / "run_logs" / "metrics.jsonl")
    assert trajectory(after) == trajectory(snapshot["metrics"])
    info = json.loads((root / "run_logs" / "run.json").read_text())
    assert info["start_step"] == 8 and info["resumed_from"].endswith("ckpt-000008.pt")


def test_full_config_smoke_runs_accumulation_with_ddp(tmp_path):
    out = tmp_path / "run_logs"
    args = ["--config", str(FULL), "--smoke", "--out-dir", str(out), "--ckpt-dir", str(tmp_path)]
    assert script.main(args) == tr.EXIT_OK
    steps = [r for r in read_jsonl(out / "metrics.jsonl") if r["kind"] == "train"]
    assert len(steps) == 12 and all(len(r["episodes"]) == 4 for r in steps)


def test_training_passes_pair_each_episode_with_itself():
    pool, rng, train, step = ["e0", "e1", "e2", "e3"], random.Random(0), [], 0
    for epoch in range(3):
        order = pool[:]
        rng.shuffle(order)
        for i in (0, 2):
            step += 1
            ces = [1.0 - 0.1 * epoch + 0.01 * int(e[1]) for e in order[i : i + 2]]
            train.append({"step": step, "episodes": order[i : i + 2], "episode_ce": ces})
    train.append({"step": step + 1, "episodes": ["e1", "e3"], "episode_ce": [0.0, 0.0]})
    result = script.training_passes(train, 4)
    assert [(r["pass"], r["steps"], r["episodes"]) for r in result["passes"]] == [
        (1, (1, 2), 4), (2, (3, 4), 4), (3, (5, 6), 4),
    ]  # fmt: skip
    paired = result["paired"]
    assert (paired["first"], paired["last"]) == (1, 3)  # the incomplete fourth pass is left out
    assert paired["mean"] == pytest.approx(-0.2) and paired["se"] == pytest.approx(0, abs=1e-9)
    assert script.training_passes(train[:1], 4) is None


# --- the train step's targets, the validation references, the view set, the cache, clipping ----


def no_dropout(cfg):
    return dataclasses.replace(cfg, hypernet=dataclasses.replace(cfg.hypernet, dropout=0.0))


def independent_ce(model, seq, factors, scaling):
    """Query CE from a plain forward: rows target - 1 predict the target tokens."""
    ids, target = seq.input_ids, seq.target
    with torch.no_grad():
        if factors is None:
            logits = model(input_ids=ids[None]).logits[0]
        else:
            with inject(model, factors, scaling):
                logits = model(input_ids=ids[None]).logits[0]
    rows = logits[target.start - 1 : target.stop - 1].float()
    return F.cross_entropy(rows, ids[target.start : target.stop]).item(), logits


def test_the_train_step_scores_exactly_the_query_targets(tmp_path):
    w = worker(no_dropout(smoke_cfg(tmp_path, gpus=1)))
    assert w.train_step() == (0, "")  # B leaves zero, so the adapter changes the logits
    seq = tr.build_sequence(w.train_records[1], w.table, w.formatter, 8192)
    with torch.no_grad():
        factors = w.hypernet(w.embeddings.z_bar(seq.demo_hashes)[None])[0]
    expected, logits = independent_ce(w.model, seq, factors, w.cfg.hypernet.scaling)
    ids = seq.input_ids
    every_token = F.cross_entropy(logits[:-1].float(), ids[1:]).item()
    assert abs(expected - every_token) > 1e-3
    w._train_modes()
    ce, penalty = w.micro_step(seq, 0)
    assert ce == pytest.approx(expected, rel=1e-6) and penalty == 0.0


def test_validation_references_match_an_independent_recomputation(tmp_path):
    """Zero adapter, current H, and mismatched z at a trained step, per episode and per source,
    recomputed with H in evaluation mode (H's dropout is on, so a validation in training mode
    would differ)."""
    cfg = smoke_cfg(tmp_path / "run", gpus=1, steps=4, eval_every=4, ckpt_every=4)
    assert cfg.hypernet.dropout > 0
    assert tr.train(cfg) == tr.EXIT_OK
    metrics = read_jsonl(tmp_path / "run" / "logs" / "metrics.jsonl")
    record = [r for r in metrics if r["kind"] == "val"][-1]
    assert record["step"] == 4
    fresh = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "re" / "logs"), ckpt_dir=str(tmp_path / "re" / "ckpt")
    )
    w = worker(fresh)
    w.hypernet.load_state_dict(h_state(tmp_path / "run" / "ckpt" / "ckpt-000004.pt"))
    w.hypernet.eval()
    w.model.eval()
    families = [seq.family for seq in w.val]
    pairing = tr.mismatch_permutation(families, cfg.seed)
    assert sorted(pairing) == list(range(len(w.val)))
    assert all(families[j] != families[i] for i, j in enumerate(pairing))
    scaling = cfg.hypernet.scaling

    def under_h(seq, z_bar):
        with torch.no_grad():
            factors = w.hypernet(z_bar[None])[0]
        return independent_ce(w.model, seq, factors, scaling)[0]

    z_bars = [w.embeddings.z_bar(seq.demo_hashes) for seq in w.val]
    expected = {
        "zero": [independent_ce(w.model, seq, None, scaling)[0] for seq in w.val],
        "current": [under_h(seq, z_bars[i]) for i, seq in enumerate(w.val)],
        "mismatched": [under_h(seq, z_bars[pairing[i]]) for i, seq in enumerate(w.val)],
    }
    # The three references are distinct here, so a swap between them shows.
    for a, b in (("zero", "current"), ("current", "mismatched")):
        assert max(abs(x - y) for x, y in zip(expected[a], expected[b], strict=True)) > 1e-4
    sources = sorted({seq.source for seq in w.val})
    assert sorted(record["by_source"]) == sources
    for ref, values in expected.items():
        assert record["per_episode"][ref] == pytest.approx(values, abs=2e-6)
        assert record["ce"][ref] == pytest.approx(sum(values) / len(values), abs=2e-6)
        for source in sources:
            index = [i for i, seq in enumerate(w.val) if seq.source == source]
            row = record["by_source"][source]
            assert row["n"] == len(index)
            assert row[ref] == pytest.approx(sum(values[i] for i in index) / len(index), abs=2e-6)
    diffs = [c - z for c, z in zip(expected["current"], expected["zero"], strict=True)]
    assert record["current_minus_zero"]["mean"] == pytest.approx(sum(diffs) / len(diffs), abs=2e-6)


def test_the_encoder_uses_the_configured_view_set(tmp_path):
    cfg = smoke_cfg(tmp_path / "a", gpus=1)
    views = view_descriptors(cfg.encoder.n_perm, cfg.encoder.view_seed)
    assert len(views) == 16
    w = worker(cfg)
    assert w.encoder.views == views and w.encoder.view_set_id == view_set_id(views)
    eight = dataclasses.replace(
        cfg,
        encoder=dataclasses.replace(cfg.encoder, n_perm=0),
        out_dir=str(tmp_path / "b" / "logs"),
        ckpt_dir=str(tmp_path / "b" / "ckpt"),
    )
    assert worker(eight).encoder.views == view_descriptors(0, cfg.encoder.view_seed)


def test_embeddings_are_read_from_an_embedding_cache(tmp_path):
    """Pairs in the cache of the encoder's namespace are read bitwise and not encoded; the rest
    are encoded; a cache of another namespace is not read."""
    cfg = smoke_cfg(tmp_path / "live", gpus=1)
    live = worker(cfg)
    pairs = sorted(tr.demo_pairs([*live.train_records, *live.val_records]))
    assert live.embeddings.encoded == len(pairs) and live.embeddings.cache_hits == 0
    root = tmp_path / "cache"
    cache = EmbeddingCache(
        root,
        model_id=cfg.model_id,
        layer=cfg.encoder.layer,
        pool=cfg.encoder.pool,
        view_set_id=live.encoder.view_set_id,
    )
    stored = {key: live.embeddings.z_bar([key]) for key in pairs[: len(pairs) // 2]}
    marked = pairs[0]
    stored[marked] = stored[marked] + 1.0  # no encoder gives this: it can only come from the cache
    cache.put(stored)

    def run(name, **changes):
        return worker(
            dataclasses.replace(
                cfg,
                embedding_cache=str(root),
                out_dir=str(tmp_path / name / "logs"),
                ckpt_dir=str(tmp_path / name / "ckpt"),
                **changes,
            )
        )

    cached = run("cached")
    assert cached.embeddings.cache_hits == len(stored)
    assert cached.embeddings.encoded == len(pairs) - len(stored)
    for key in pairs:
        if key in stored:
            assert torch.equal(cached.embeddings.z_bar([key]), stored[key])
        else:
            torch.testing.assert_close(
                cached.embeddings.z_bar([key]), live.embeddings.z_bar([key]), rtol=1e-5, atol=1e-6
            )
    other = run("other", model_id="smoke/another-model")
    assert other.embeddings.cache_hits == 0 and other.embeddings.encoded == len(pairs)
    assert torch.equal(other.embeddings.z_bar([marked]), live.embeddings.z_bar([marked]))


def test_gradient_clipping_bounds_the_update(tmp_path):
    """AdamW moves every weight by about the learning rate whatever the gradient's scale, unless
    the gradient falls far below its epsilon: a clip of 1e-12 then shrinks the step about 1e4
    times. Weight decay is off so that only the gradient moves the weights."""
    moved = {}
    for clip in (1e-12, 1.0):
        cfg = smoke_cfg(tmp_path / str(clip), gpus=1, clip=clip, weight_decay=0.0)
        w = worker(cfg)
        before = {n: p.detach().clone() for n, p in w.hypernet.named_parameters()}
        assert w.train_step() == (0, "")
        moved[clip] = max(
            (p.detach() - before[n]).abs().max().item() for n, p in w.hypernet.named_parameters()
        )
    assert moved[1.0] > 1e-4 and moved[1e-12] < 1e-3 * moved[1.0]


# --- run P3.5-02: P3.5-01's toy run with half the steps ------------------------------------------


def test_p3_5_02_config_is_p3_5_01s_with_half_the_steps():
    """P3.5-01's brief pre-registered one change for a faded gain: steps 2000 to 1000. Only the
    session budget and the projected-wall kill follow the run's length; the smoke section scales
    its steps, cadence, and folders the same way and keeps every other value."""
    first, second = (script.load_config(path, False, {}) for path in (TOY, TOY_02))
    assert (first.run_id, first.steps) == ("P3.5-01", 2000)
    budgets = {"max_wall_s": 7800, "kill": dataclasses.replace(first.kill, max_projected_s=7200)}
    assert second == dataclasses.replace(first, run_id="P3.5-02", steps=1000, **budgets)
    assert second.kill.max_projected_s < second.max_wall_s
    # Two passes over the 2000 toy episodes (P3.4 build 2), where P3.5-01 made four.
    assert second.steps * second.gpus * second.accum == 2 * 2000
    smoke_first, smoke_second = (script.load_config(path, True, {}) for path in (TOY, TOY_02))
    runs = REPO / "runs" / "P3.5-02-smoke"
    scaled = {
        "run_id": "P3.5-02-smoke", "steps": 12, "eval_every": 4, "ckpt_every": 4,
        "ckpt_dir": str(runs / "checkpoints"), "out_dir": str(runs / "run_logs"),
    }  # fmt: skip
    assert smoke_second == dataclasses.replace(smoke_first, **scaled)
    assert smoke_second.steps * smoke_second.gpus * smoke_second.accum == 2 * len(records("toy"))


def test_p3_5_02_keeps_p3_5_01s_warmup_and_ends_its_cosine_at_step_1000():
    """The schedule's shape over the new length: steps 1 to 100 take P3.5-01's learning rates
    exactly; the cosine then reaches min_lr at step 1000, where P3.5-01's was still above half its
    peak, and every later step takes a smaller rate than P3.5-01's at the same step."""
    first, second = (script.load_config(path, False, {}) for path in (TOY, TOY_02))
    assert (second.lr, second.min_lr, second.warmup_steps) == (first.lr, first.min_lr, 100)
    rho = second.min_lr / second.lr
    old = tr.WarmupCosine(first.warmup_steps, first.steps, rho)
    new = tr.WarmupCosine(second.warmup_steps, second.steps, rho)
    # LambdaLR passes the steps taken, so step n runs at lr * schedule(n - 1).
    assert [new(n - 1) for n in range(1, 101)] == [old(n - 1) for n in range(1, 101)]
    assert new(1000 - 1) == pytest.approx(rho) and old(1000 - 1) > 0.5
    assert new(550 - 1) == pytest.approx((1 + rho) / 2) == old(1050 - 1)
    assert all(new(n - 1) < old(n - 1) for n in range(101, 1001))


@pytest.fixture(scope="module")
def smoke_run_02(tmp_path_factory):
    """P3.5-02's --smoke (2 processes on gloo, 12 steps), then its rerun from step 4."""
    root = tmp_path_factory.mktemp("smoke_02")
    args = ["--config", str(TOY_02), "--smoke", "--out-dir", str(root / "run_logs")]
    args += ["--ckpt-dir", str(root / "checkpoints")]
    began = time.monotonic()
    first = script.main(args)
    seconds = time.monotonic() - began
    snapshot = {
        "metrics": read_jsonl(root / "run_logs" / "metrics.jsonl"),
        "final": h_state(root / "checkpoints" / "ckpt-000012.pt"),
        "summary": (root / "run_logs" / "summary.md").read_text(),
    }
    second = script.main(args + ["--resume", str(root / "checkpoints" / "ckpt-000004.pt")])
    return root, first, seconds, snapshot, second


def test_p3_5_02_smoke_makes_two_passes_and_ends_at_min_lr(smoke_run_02):
    """Criterion A compares pass 2 with pass 1 (the summary's paired line); validation runs at
    steps 0, 4, 8, 12 (B at the end, B' before); the last step runs at min_lr."""
    _, first, seconds, snapshot, _ = smoke_run_02
    assert first == tr.EXIT_OK and seconds < 120
    print(f"P3.5-02 smoke: {seconds:.1f} s")
    train = [r for r in snapshot["metrics"] if r["kind"] == "train"]
    assert [r["step"] for r in train] == list(range(1, 13))
    assert train[-1]["lr"] == pytest.approx(script.load_config(TOY_02, True, {}).min_lr)
    passes = script.training_passes(train, len(records("toy")))
    assert [(p["pass"], p["steps"]) for p in passes["passes"]] == [(1, (1, 6)), (2, (7, 12))]
    paired = passes["paired"]
    assert (paired["first"], paired["last"]) == (1, 2)
    assert paired["mean"] < -2 * paired["se"]  # the memorizable fixture: A holds
    assert "Pass 2 minus pass 1, paired over the same episodes" in snapshot["summary"]
    val = [r for r in snapshot["metrics"] if r["kind"] == "val"]
    assert [r["step"] for r in val] == [0, 4, 8, 12]
    assert val[0]["ce"]["current"] == val[0]["ce"]["zero"] == val[0]["ce"]["mismatched"]
    assert val[-1]["ce"]["current"] < val[-1]["ce"]["zero"]


def test_p3_5_02_smoke_resumed_in_the_first_pass_ends_with_the_same_h(smoke_run_02):
    """The rerun from step 4 crosses the pass boundary at step 6 and ends bitwise as the run."""
    root, _, _, snapshot, second = smoke_run_02
    assert second == tr.EXIT_OK
    assert_same_state(snapshot["final"], h_state(root / "checkpoints" / "ckpt-000012.pt"))
    after = read_jsonl(root / "run_logs" / "metrics.jsonl")
    assert trajectory(after) == trajectory(snapshot["metrics"])
    info = json.loads((root / "run_logs" / "run.json").read_text())
    assert info["start_step"] == 4 and info["resumed_from"].endswith("ckpt-000004.pt")


def test_p3_5_02_smoke_repeats_p3_5_01s_until_the_learning_rates_part(smoke_run, smoke_run_02):
    """One change, end to end on the CPU: the same episodes in the same order and the same H
    through the smoke's 2 warmup steps, bitwise, so step 3's loss (taken before its update) is
    P3.5-01's too; from step 3 the shorter cosine runs at lower rates, and step 4's loss differs."""
    before = [r for r in smoke_run[3]["metrics"] if r["kind"] == "train"]
    after = [r for r in smoke_run_02[3]["metrics"] if r["kind"] == "train"]
    assert [r["episodes"] for r in after] == [r["episodes"] for r in before[:12]]
    assert [r["lr"] for r in after[:2]] == [r["lr"] for r in before[:2]]
    assert all(a["lr"] < b["lr"] for a, b in zip(after[2:], before[2:12], strict=True))
    assert [r["loss"] for r in after[:3]] == [r["loss"] for r in before[:3]]
    assert after[3]["loss"] != before[3]["loss"]
    val_before = [r for r in smoke_run[3]["metrics"] if r["kind"] == "val"]
    val_after = [r for r in smoke_run_02[3]["metrics"] if r["kind"] == "val"]
    assert val_after[0]["per_episode"] == val_before[0]["per_episode"]
