"""Task MT: source weights and the best checkpoint of arc26/hyper/train.py, and the MT config.

On the CPU (BRIEF A0) with the tiny random Qwen3 and the stand-in tokenizer built in code, on the
fixture of tests/testdata/train_hypernet/ (its toy split holds four episodes of each source, its
validation split six). ``configs/train_hypernet_MT.yaml`` is the config of runs MT-02, MT-03, and
MT-04, three sessions of one run: these tests check its values against the briefs' derivations
(steps that fit MT-02's capped session and two full ones, the draws per source its comment
states, the encoder namespace of MT-01's cache), and that ``--max-wall-s``, MT-02 revision 2's
per-session cap, changes that session's budget only and never blocks a resume. Without torch,
transformers, or safetensors the module is skipped. Run with CUDA_VISIBLE_DEVICES="".
"""

import dataclasses
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.encoder import view_descriptors, view_set_id  # noqa: E402
from arc26.hyper.hypernet import BaseShapes, Hypernet, HypernetConfig  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TOY = REPO / "configs" / "train_hypernet_toy.yaml"
MT = REPO / "configs" / "train_hypernet_MT.yaml"
MT_02_BUILD = REPO / "configs" / "script_MT-02.yaml"
CACHE_MT = REPO / "configs" / "cache_embeddings_MT-01.yaml"
HYPERNET = REPO / "configs" / "hypernet.yaml"
# P3.4 build 2's training pool by source (P3.4 stats), which MT-02 and MT-03 draw from.
POOL = {"arc2train": 19997, "barc": 49680, "rearc": 29997}
WEIGHTS = {"arc2train": 0.20, "barc": 0.65, "rearc": 0.15}
# P3.6-01's calibrated step on the pool's own mix, and P3.5-01's measured validation (two
# references over the 286 episodes) and checkpoint seconds, which MT-02's brief prices with.
STEP_S, VAL_S, VAL0_S, CKPT_S = 13.42, 76.0, 107.0, 10.0


def load_script(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


script = load_script("train_hypernet")


def toy_cfg(tmp_path, **changes):
    """The toy config's smoke settings on one process, outputs under ``tmp_path``."""
    cfg = script.load_config(
        TOY, True, {"out_dir": str(tmp_path / "logs"), "ckpt_dir": str(tmp_path / "ckpt")}
    )
    return dataclasses.replace(cfg, gpus=1, **changes)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def val_records(root):
    return [r for r in read_jsonl(root / "logs" / "metrics.jsonl") if r["kind"] == "val"]


def h_state(path):
    return tr.load_checkpoint(Path(path))["hypernet"]


def assert_same_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def mixed_pool(seed=0, sizes=(50, 30, 20)):
    """BARC, ReARC, and ARC-AGI-2 training episodes in these numbers, interleaved by a seeded
    shuffle."""
    sources = ["barc"] * sizes[0] + ["rearc"] * sizes[1] + ["arc2train"] * sizes[2]
    random.Random(seed).shuffle(sources)
    return sources


# --- the weighted episode order ------------------------------------------------------------------


def test_block_counts_round_by_the_largest_remainder():
    assert tr.block_counts([0.65, 0.15, 0.20], 1000) == [650, 150, 200]
    assert tr.block_counts([1 / 3, 1 / 3, 1 / 3], 1000) == [334, 333, 333]
    assert tr.block_counts([0.5, 0.5], 3) == [2, 1]
    assert tr.block_counts([0.0, 1.0], 7) == [0, 7]
    shares = [w / sum(range(1, 6)) for w in range(1, 6)]
    counts = tr.block_counts(shares, 1000)
    assert sum(counts) == 1000
    assert all(abs(c - s * 1000) < 1 for c, s in zip(counts, shares, strict=True))


def test_without_weights_the_order_is_the_unweighted_one():
    """The default keeps P3.5's order exactly: the toy runs' data order and tests stand."""
    plain = tr.EpisodeOrder(7, seed=0)
    sourced = tr.EpisodeOrder(7, seed=0, sources=list("abcabca"), weights=None)
    assert [plain(g) for g in range(40)] == [sourced(g) for g in range(40)]
    assert plain.source_draws(100) is None
    cfg = script.load_config(TOY, False, {})
    assert cfg.source_weights is None and cfg.to_dict()["source_weights"] is None
    # A checkpoint written before the key existed resumes: its config lacks it, which reads None.
    saved = cfg.to_dict()
    del saved["source_weights"]
    assert tr.resume_conflicts(saved, cfg) == []


def test_each_block_holds_the_shares_and_each_source_its_own_passes():
    """Pools larger than a block's share, so each source's passes span blocks: every aligned
    block holds the shares exactly, and each source shows each of its episodes once per pass of
    its own, for two passes each, across block boundaries."""
    sources = mixed_pool(sizes=(2000, 600, 400))
    order = tr.EpisodeOrder(len(sources), seed=0, sources=sources, weights=WEIGHTS)
    assert order.counts == {"arc2train": 200, "barc": 650, "rearc": 150}
    blocks = 12  # BARC 7,800 draws of 2,000 episodes; ReARC 1,800 of 600; ARC-AGI-2 2,400 of 400
    draws = [order(g) for g in range(blocks * tr.SOURCE_BLOCK)]
    for b in range(blocks):
        block = draws[b * tr.SOURCE_BLOCK : (b + 1) * tr.SOURCE_BLOCK]
        assert Counter(sources[i] for i in block) == order.counts
    for name, size in Counter(sources).items():
        assert order.counts[name] < size  # a pass of this source spans several blocks
        mine = [i for i in draws if sources[i] == name]
        pool = sorted(i for i, s in enumerate(sources) if s == name)
        first, second = mine[:size], mine[size : 2 * size]
        assert sorted(first) == pool and sorted(second) == pool  # once per pass of its own
        assert first != second
    assert order.source_draws(11500) == dict(Counter(sources[i] for i in draws[:11500]))


def test_the_weighted_order_is_a_pure_function_of_the_seed_and_the_slot():
    sources = mixed_pool()
    order = tr.EpisodeOrder(len(sources), seed=0, sources=sources, weights=WEIGHTS)
    sequence = [order(g) for g in range(2600)]
    slots = [2599, 3, 1999, 1000, 0, 1500, 999]  # across blocks and source passes, out of order
    fresh = tr.EpisodeOrder(len(sources), seed=0, sources=sources, weights=WEIGHTS)
    assert [fresh(g) for g in slots] == [sequence[g] for g in slots]
    scaled = {"arc2train": 4, "barc": 13, "rearc": 3}  # the same shares at another scale
    same = tr.EpisodeOrder(len(sources), seed=0, sources=sources, weights=scaled)
    assert [same(g) for g in range(2600)] == sequence
    other = tr.EpisodeOrder(len(sources), seed=1, sources=sources, weights=WEIGHTS)
    assert [other(g) for g in range(100)] != sequence[:100]


def test_weights_must_name_exactly_the_splits_sources():
    sources = mixed_pool()
    with pytest.raises(ValueError, match="training split holds"):
        tr.EpisodeOrder(100, 0, sources=sources, weights={"barc": 1.0, "rearc": 1.0})
    with pytest.raises(ValueError, match="training split holds"):
        tr.EpisodeOrder(100, 0, sources=["barc"] * 100, weights=WEIGHTS)
    with pytest.raises(ValueError, match="source of every pool episode"):
        tr.EpisodeOrder(100, 0, sources=None, weights=WEIGHTS)
    cfg = toy_cfg(Path("unused"))
    for bad, message in (
        ({"arc1": 1.0}, "unknown sources"),
        ({"barc": -0.1, "rearc": 1.0}, ">= 0"),
        ({"barc": 0.0}, "one weight must be positive"),
        ({"barc": True}, ">= 0"),
        ([("barc", 1.0)], "mapping"),
        ({}, "mapping"),
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(cfg, source_weights=bad)


def test_source_weights_are_a_resume_setting_in_one_form():
    cfg = toy_cfg(Path("unused"))
    weighted = dataclasses.replace(cfg, source_weights={"rearc": 3, "barc": 13, "arc2train": 4})
    assert weighted.source_weights == {"arc2train": 4.0, "barc": 13.0, "rearc": 3.0}
    assert list(weighted.source_weights) == ["arc2train", "barc", "rearc"]
    assert tr.resume_conflicts(weighted.to_dict(), cfg) == ["source_weights"]
    assert tr.TrainConfig.from_dict(weighted.to_dict()) == weighted


def test_a_weighted_run_draws_its_order_and_resumes_bitwise(tmp_path):
    weights = {"arc2train": 1.0, "barc": 2.0, "rearc": 1.0}
    settings = {"steps": 6, "eval_every": 3, "ckpt_every": 3, "keep_checkpoints": 3}
    whole = toy_cfg(tmp_path / "whole", source_weights=weights, **settings)
    assert tr.train(whole) == tr.EXIT_OK
    records = tr.load_split(
        tr.open_episode_set(
            Path(whole.episodes_root), whole.manifest_sha256, Path(whole.check_root), 8192
        ),
        whole.train_split,
        8192,
    )
    order = tr.EpisodeOrder(
        len(records), whole.seed, sources=[r.source for r in records], weights=weights
    )
    steps = [
        r for r in read_jsonl(tmp_path / "whole" / "logs" / "metrics.jsonl") if r["kind"] == "train"
    ]
    assert [r["episodes"] for r in steps] == [[records[order(s)].episode_id] for s in range(6)]
    info = json.loads((tmp_path / "whole" / "logs" / "run.json").read_text())
    assert info["order"]["per_block"] == {"arc2train": 250, "barc": 500, "rearc": 250}
    assert sum(info["order"]["draws"].values()) == 6
    resumed = toy_cfg(tmp_path / "resumed", source_weights=weights, **settings)
    start = tmp_path / "whole" / "ckpt" / "ckpt-000003.pt"
    assert tr.train(resumed, resume=str(start)) == tr.EXIT_OK
    assert_same_state(
        h_state(tmp_path / "whole" / "ckpt" / "ckpt-000006.pt"),
        h_state(tmp_path / "resumed" / "ckpt" / "ckpt-000006.pt"),
    )
    again = [
        r["episodes"]
        for r in read_jsonl(tmp_path / "resumed" / "logs" / "metrics.jsonl")
        if r["kind"] == "train"
    ]
    assert again == [r["episodes"] for r in steps]
    unweighted = toy_cfg(tmp_path / "other", **settings)
    with pytest.raises(ValueError, match="source_weights"):
        tr.train(unweighted, resume=str(start))


def test_validation_record_signs_and_scales_both_differences():
    """MT-02's and MT-03's rows read ``z``: criterion B ``z.current_minus_zero`` (negative when H
    beats the zero adapter) and check C ``z.mismatched_minus_current`` (positive when the
    matched z_bar beats another family's). Hand-computed means, standard errors (n - 1), and z on
    varied CEs; None where a difference is constant (SE 0)."""

    def sequence(i, source):
        return tr.EpisodeSequence(f"e{i}", source, f"f{i}", torch.zeros(1, dtype=torch.long),
                                  range(0), ())  # fmt: skip

    sources = ["barc", "barc", "rearc", "rearc", "barc"]
    seqs = [sequence(i, source) for i, source in enumerate(sources)]
    zero = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    current = zero - torch.tensor([0.5, 0.1, 0.4, 0.2, 0.3], dtype=torch.float64)
    mismatched = current + torch.tensor([0.3, 0.1, 0.2, 0.0, 0.4], dtype=torch.float64)
    ces = {"zero": zero, "current": current, "mismatched": mismatched}
    record = tr.validation_record(7, ces, seqs, 1.5)
    # current - zero = (-0.5, -0.1, -0.4, -0.2, -0.3): mean -0.3, sample variance 0.025;
    # mismatched - current = (0.3, 0.1, 0.2, 0.0, 0.4): mean 0.2, sample variance 0.025.
    se = (0.025 / 5) ** 0.5
    assert record["current_minus_zero"] == pytest.approx({"mean": -0.3, "se": se})
    assert record["mismatched_minus_current"] == pytest.approx({"mean": 0.2, "se": se})
    assert record["z"]["current_minus_zero"] == pytest.approx(-0.3 / se)  # B: below -2
    assert record["z"]["mismatched_minus_current"] == pytest.approx(0.2 / se)  # C: above 2
    assert record["ce"] == pytest.approx({"zero": 3.0, "current": 2.7, "mismatched": 2.9})
    assert record["by_source"]["rearc"] == pytest.approx(
        {"n": 2, "zero": 3.5, "current": 3.2, "mismatched": 3.3}
    )
    assert (record["step"], record["n"], record["seconds"]) == (7, 5, 1.5)
    flat = tr.validation_record(0, {"zero": zero, "current": zero, "mismatched": zero + 0.5},
                                seqs, 0.0)  # fmt: skip
    assert flat["z"] == {"current_minus_zero": None, "mismatched_minus_current": None}
    one = {name: values[:1] for name, values in ces.items()}
    assert tr.validation_record(0, one, seqs[:1], 0.0)["z"]["current_minus_zero"] is None


# --- the best checkpoint -------------------------------------------------------------------------


def scripted_validation(monkeypatch, current_by_step):
    """Make every validation return scripted CEs: zero 1.0, current from the table by step, and
    mismatched 0.05 above current; training runs as usual. Returns the steps best.pt was written
    at."""

    def episode_ce(self, seq, z_bar):
        if z_bar is None:
            return 1.0
        index = next(i for i, s in enumerate(self.val) if s.episode_id == seq.episode_id)
        current = current_by_step[self.step]
        return current if z_bar is self.val_z[index] else current + 0.05

    written = []
    real = tr.save_best

    def save_best(ckpt_dir, payload):
        written.append(payload["step"])
        return real(ckpt_dir, payload)

    monkeypatch.setattr(tr._Worker, "episode_ce", episode_ce)
    monkeypatch.setattr(tr, "save_best", save_best)
    return written


def test_best_pt_holds_h_at_each_new_low_of_the_validation_ce(tmp_path, monkeypatch):
    written = scripted_validation(monkeypatch, {0: 1.0, 2: 0.8, 4: 0.9, 6: 0.7})
    settings = {"steps": 6, "eval_every": 2, "ckpt_every": 2, "keep_checkpoints": 4}
    cfg = toy_cfg(tmp_path, **settings)
    assert tr.train(cfg) == tr.EXIT_OK
    assert written == [0, 2, 6]
    val = val_records(tmp_path)
    assert [(r["step"], r["best"], r["session_start"]) for r in val] == [
        (0, True, 0), (2, True, 0), (4, False, 0), (6, True, 0),
    ]  # fmt: skip
    assert val[1]["z"]["mismatched_minus_current"] is None  # constant differences: SE 0
    ckpt = tmp_path / "ckpt"
    best = tr.load_checkpoint(ckpt / tr.BEST_FILE)
    assert best["kind"] == "best" and best["step"] == 6
    assert {"optimizer", "scheduler", "rng", "history"}.isdisjoint(best)
    assert best["validation"]["ce"]["current"] == pytest.approx(0.7)
    assert_same_state(best["hypernet"], h_state(ckpt / "ckpt-000006.pt"))
    status = tr.read_status(tmp_path / "logs")
    assert status["best"]["step"] == 6 and status["best"]["file"] == str(ckpt / tr.BEST_FILE)
    # best.pt is never pruned and never taken for a resume of the folder.
    assert sorted(p.name for p in ckpt.glob("ckpt-*.pt"))[-1] == "ckpt-000006.pt"
    assert tr.resolve_checkpoint(ckpt) == ckpt / "ckpt-000006.pt"


def test_a_resumed_session_keeps_the_best_point_of_the_earlier_one(tmp_path, monkeypatch):
    """MT-03 resumes MT-02: best.pt is rewritten only for a new low over both sessions, and the
    status names the session whose output holds it."""
    settings = {"steps": 6, "eval_every": 2, "ckpt_every": 2, "keep_checkpoints": 4}
    written = scripted_validation(monkeypatch, {0: 1.0, 2: 0.6, 4: 0.9, 6: 0.8})
    assert tr.train(toy_cfg(tmp_path / "a", **settings)) == tr.EXIT_OK
    start = tmp_path / "a" / "ckpt" / "ckpt-000004.pt"
    written.clear()
    assert tr.train(toy_cfg(tmp_path / "b", **settings), resume=str(start)) == tr.EXIT_OK
    assert written == [] and not (tmp_path / "b" / "ckpt" / tr.BEST_FILE).exists()
    val = val_records(tmp_path / "b")
    assert [(r["step"], r["best"], r["session_start"]) for r in val] == [
        (0, True, 0), (2, True, 0), (4, False, 0), (6, False, 4),
    ]  # fmt: skip
    status = tr.read_status(tmp_path / "b" / "logs")
    assert (status["best"]["step"], status["best"]["session_start"]) == (2, 0)
    assert status["best"]["file"] is None  # in the first session's output
    written = scripted_validation(monkeypatch, {0: 1.0, 2: 0.6, 4: 0.9, 6: 0.5})
    assert tr.train(toy_cfg(tmp_path / "c", **settings), resume=str(start)) == tr.EXIT_OK
    assert written == [6]
    status = tr.read_status(tmp_path / "c" / "logs")
    assert (status["best"]["step"], status["best"]["session_start"]) == (6, 4)
    assert status["best"]["file"] == str(tmp_path / "c" / "ckpt" / tr.BEST_FILE)


def test_best_pt_loads_as_p4_1_loads_h_and_is_never_resumed(tmp_path):
    cfg = toy_cfg(tmp_path, steps=4, eval_every=2, ckpt_every=2)
    assert tr.train(cfg) == tr.EXIT_OK
    path = tmp_path / "ckpt" / tr.BEST_FILE
    payload = tr.load_checkpoint(tr.resolve_checkpoint(path))
    shapes = BaseShapes.from_config(tr.tiny_base().config)
    hypernet = Hypernet(HypernetConfig(**payload["config"]["hypernet"]), shapes)
    hypernet.load_state_dict(payload["hypernet"])  # HyperProvider.from_checkpoint's two reads
    again = toy_cfg(tmp_path / "again", steps=4, eval_every=2, ckpt_every=2)
    with pytest.raises(ValueError, match="best validation point only"):
        tr.train(again, resume=str(path))


# --- the MT config -------------------------------------------------------------------------------


def test_the_mt_config_matches_its_brief():
    cfg = script.load_config(MT, False, {})
    assert cfg.run_id == "MT-02" and cfg.manifest_sha256.startswith("b1f2cd04")
    assert (cfg.train_split, cfg.val_split, cfg.max_tokens) == ("train", "val", 8192)
    assert (cfg.gpus, cfg.accum, cfg.steps, cfg.warmup_steps) == (4, 8, 5600, 100)
    assert (cfg.lr, cfg.min_lr, cfg.clip, cfg.seed) == (4e-5, 1e-7, 1.0, 0)
    assert (cfg.eval_every, cfg.ckpt_every, cfg.keep_checkpoints) == (200, 100, 2)
    assert cfg.max_wall_s == 41400 and cfg.kill == tr.KillCriteria(150, 82000, 100)
    assert cfg.encoder == tr.EncoderSettings(24, "output", 1, 2, 16384, 256)
    assert cfg.hypernet == HypernetConfig.from_yaml(HYPERNET)
    assert cfg.source_weights == WEIGHTS
    assert cfg.embedding_cache.endswith("/alyahya737/arc26-mt-cache/embedding_cache")
    assert (cfg.device, cfg.dtype, cfg.attn_implementation) == ("cuda", "bfloat16", "auto")
    smoke = script.load_config(MT, True, {})
    assert (smoke.device, smoke.gpus, smoke.model_dir, smoke.embedding_cache) == (
        "cpu", 2, None, None,
    )  # fmt: skip
    assert smoke.source_weights == WEIGHTS and smoke.keep_checkpoints == 2


def mt_02_cap():
    """MT-02's session budget: the ``--max-wall-s`` value in its build's arguments."""
    args = yaml.safe_load(MT_02_BUILD.read_text())["args"]
    return float(args[args.index("--max-wall-s") + 1])


def test_the_mt_steps_fit_mt_02s_capped_session_and_two_full_ones_at_p3_6_01s_step():
    """5,600 steps, their validations, and their checkpoints, with three sessions' setup (600 s,
    then 480 s twice, estimate), fit MT-02's capped budget and two full ones at 13.42 s per step
    with room to spare, and not the first two sessions alone (MT-04 is needed); the
    projected-wall kill sits above that cost, so it fires only at a mean step above about 13.9 s,
    and below what the three budgets hold."""
    cfg = script.load_config(MT, False, {})
    per_step = STEP_S + VAL_S / cfg.eval_every + CKPT_S / cfg.ckpt_every
    setup = 600 + 480 + 480
    cost = setup + cfg.steps * per_step + (VAL0_S - VAL_S)
    budgets = mt_02_cap() + 2 * cfg.max_wall_s
    assert mt_02_cap() < cfg.max_wall_s
    assert cost < 0.85 * budgets
    assert cost - 480 > mt_02_cap() + cfg.max_wall_s  # two sessions do not hold the run
    assert cost < cfg.kill.max_projected_s < budgets
    threshold = (cfg.kill.max_projected_s - setup - (VAL0_S - VAL_S)) / cfg.steps
    assert 13.85 < threshold - VAL_S / cfg.eval_every - CKPT_S / cfg.ckpt_every < 13.9


def test_max_wall_s_on_the_command_line_is_this_sessions_budget_only():
    """MT-02 revision 2 caps its session with ``--max-wall-s``: the resolved config differs from
    the file's in ``max_wall_s`` alone, which no resume compares, and only a finite positive
    number of seconds is accepted (argparse's exit 2, the config-error code)."""
    parser = script.build_parser()
    assert parser.parse_args(["--config", str(MT)]).max_wall_s is None
    assert parser.parse_args(["--config", str(MT), "--max-wall-s", "12600"]).max_wall_s == 12600.0
    for bad in ("0", "-1", "nan", "inf", "-inf", "1e400", "12,600", "x"):
        with pytest.raises(SystemExit) as stop:
            parser.parse_args(["--config", str(MT), "--max-wall-s", bad])
        assert stop.value.code == tr.EXIT_CONFIG, bad
    base = script.load_config(MT, False, {})
    capped = script.load_config(MT, False, {"max_wall_s": mt_02_cap()})
    assert capped == dataclasses.replace(base, max_wall_s=12600.0)
    assert "max_wall_s" not in tr.RESUME_KEYS and "kill" not in tr.RESUME_KEYS
    assert tr.resume_conflicts(capped.to_dict(), base) == []
    assert tr.resume_conflicts(base.to_dict(), capped) == []
    # In a smoke run the command line wins over the smoke section's 600 s.
    assert script.load_config(MT, True, {"max_wall_s": 5.0}).max_wall_s == 5.0


def test_a_session_stopped_at_its_command_line_budget_resumes_under_the_configs(tmp_path):
    """The MT smoke (two gloo processes, the source weights) stopped by a budget below its setup
    stops before its first step with a checkpoint (exit 3); resumed without ``--max-wall-s`` it
    runs to the end, and its H equals an uninterrupted run's, bitwise."""

    def run(name, *extra):
        out, ckpt = tmp_path / name / "run_logs", tmp_path / name / "checkpoints"
        args = ["--config", str(MT), "--smoke", "--out-dir", str(out), "--ckpt-dir", str(ckpt)]
        return script.main([*args, *extra]), out, ckpt

    code, out, ckpt = run("capped", "--max-wall-s", "1e-6")
    assert code == tr.EXIT_STOPPED
    status = json.loads((out / "status.json").read_text())
    assert (status["status"], status["step"]) == ("stopped", 0)
    assert "max_wall_s 1e-06" in status["reason"]
    assert yaml.safe_load((out / "config.yaml").read_text())["max_wall_s"] == 1e-6
    assert tr.resolve_checkpoint(ckpt) == ckpt / "ckpt-000000.pt"
    assert tr.load_checkpoint(ckpt / "ckpt-000000.pt")["config"]["max_wall_s"] == 1e-6
    code, out2, ckpt2 = run("resumed", "--resume", str(ckpt))
    assert code == tr.EXIT_OK
    status = json.loads((out2 / "status.json").read_text())
    assert (status["status"], status["step"]) == ("complete", 12)
    assert yaml.safe_load((out2 / "config.yaml").read_text())["max_wall_s"] == 600
    val = [r for r in read_jsonl(out2 / "metrics.jsonl") if r["kind"] == "val"]
    assert [r["step"] for r in val] == [0, 4, 8, 12]
    code, _, ckpt3 = run("whole")
    assert code == tr.EXIT_OK
    assert_same_state(h_state(ckpt2 / "ckpt-000012.pt"), h_state(ckpt3 / "ckpt-000012.pt"))


def test_the_mt_config_states_the_draws_its_weights_give():
    sources = [s for s, n in POOL.items() for _ in range(n)]
    order = tr.EpisodeOrder(len(sources), 0, sources=sources, weights=WEIGHTS)
    cfg = script.load_config(MT, False, {})
    draws = order.source_draws(cfg.steps * cfg.gpus * cfg.accum)
    assert sum(draws.values()) == 179200
    text = re.sub(r"\s+", " ", MT.read_text())
    for name, size in POOL.items():
        assert f"{draws[name]:,} draws" in text or f"{draws[name]:,}," in text, name
        assert f"{draws[name] / size:.2f} passes" in text, name
        assert f"{size:,}" in text, name


def test_the_mt_encoder_reads_mt_01s_cache_namespace():
    """MT-02 reads MT-01's cache only if both use one namespace: model id, layer, pool, views."""
    cfg = script.load_config(MT, False, {})
    cache = yaml.safe_load(CACHE_MT.read_text())
    assert cache["model_id"] == cfg.model_id
    assert (cache["layer"], cache["pool"]) == (cfg.encoder.layer, cfg.encoder.pool)
    assert view_set_id(view_descriptors(cache["n_perm"], cache["view_seed"])) == view_set_id(
        view_descriptors(cfg.encoder.n_perm, cfg.encoder.view_seed)
    )
    assert len(view_descriptors(cfg.encoder.n_perm, cfg.encoder.view_seed)) == 8
    assert cache["episodes"]["manifest_sha256"] == cfg.manifest_sha256
    assert cache["episodes"]["splits"] == [cfg.train_split, cfg.val_split]


def test_every_mt_config_value_has_a_comment():
    """BRIEF A8: every value carries its reason, in a comment above it or above its group (keys
    on consecutive lines below one comment, as configs/train_hypernet.yaml groups lr and its
    schedule); a blank line ends a group."""
    covered = False
    for line in MT.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            covered = False
        elif stripped.startswith("#"):
            covered = True
        elif re.match(r"[a-z_0-9]+:", stripped):
            assert covered or "#" in line, line


def test_the_mt_smoke_runs_the_weighted_order_and_writes_best_pt(tmp_path):
    out, ckpt = tmp_path / "run_logs", tmp_path / "checkpoints"
    args = ["--config", str(MT), "--smoke", "--out-dir", str(out), "--ckpt-dir", str(ckpt)]
    began = time.monotonic()
    assert script.main(args) == tr.EXIT_OK
    assert time.monotonic() - began < 120
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "complete" and status["step"] == 12
    assert status["best"]["file"] == str(ckpt / tr.BEST_FILE) and (ckpt / tr.BEST_FILE).is_file()
    # Two DDP processes wrote it: H's own keys, which a plain Hypernet loads strictly.
    payload = tr.load_checkpoint(ckpt / tr.BEST_FILE)
    plain = Hypernet(
        HypernetConfig(**payload["config"]["hypernet"]),
        BaseShapes.from_config(tr.tiny_base().config),
    )
    plain.load_state_dict(payload["hypernet"], strict=True)
    assert not any(key.startswith("module.") for key in payload["hypernet"])
    info = json.loads((out / "run.json").read_text())
    assert info["order"]["per_block"] == {"arc2train": 200, "barc": 650, "rearc": 150}
    assert sorted(p.name for p in ckpt.glob("ckpt-*.pt")) == ["ckpt-000008.pt", "ckpt-000012.pt"]
    val = [r for r in read_jsonl(out / "metrics.jsonl") if r["kind"] == "val"]
    assert [r["step"] for r in val] == [0, 4, 8, 12] and val[0]["best"] is True
    assert all(set(r["z"]) == {"current_minus_zero", "mismatched_minus_current"} for r in val)
