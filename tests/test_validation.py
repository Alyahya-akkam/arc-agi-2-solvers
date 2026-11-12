"""P3.7: GEN and H's load check (arc26.hyper.validation) on the CPU.

GEN: synthetic validation records in P3.4's format and the episode fixture of tests/test_train.py.
The load check: a tiny H trained by ``train.py``'s smoke path on that fixture (the toy config's
smoke section, one process), whose logged best point the check must reproduce, and the failure
path: another checkpoint, perturbed weights, another H config, another manifest, a tolerance
below the log's rounding. Skipped without torch, transformers, peft, or safetensors (GEN's tests
need numpy only).
"""

import dataclasses
import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from arc26.hyper.episodes import EpisodeRecord  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "testdata" / "train_hypernet"


def grid(seed, rows=2, cols=3):
    return np.random.default_rng(seed).integers(0, 10, size=(rows, cols)).astype(np.uint8)


def record(source, puzzle, index, family, seed, demos=2, augmentation=5):
    pairs = [(grid(seed + 10 * k), grid(seed + 10 * k + 1)) for k in range(demos + 1)]
    return EpisodeRecord.create(
        episode_id=f"{source}/{puzzle}/val/{index}",
        source=source,
        task_id=family,
        family=family,
        split="val",
        augmentation_id=augmentation,
        demos=pairs[:demos],
        queries=pairs[demos:],
    )


def synthetic():
    """6 BARC puzzles of 2 episodes (indices 10 and 2: the numeric lowest is 2) and 3 ReARC
    families of 5 puzzles; the records come in a scrambled order."""
    records = []
    for i in range(6):
        for index in (10, 2):
            records.append(record("barc", f"barc-p{i}", index, f"barc:p{i}", 100 * i + index))
    for f, family in enumerate(("gen-c", "gen-a", "gen-b")):
        for i in range(5):
            records.append(record("rearc", f"rearc-{family}-{i}", 0, family, 1000 + 50 * f + i))
    return list(reversed(records))


def test_gen_takes_the_lowest_episode_of_each_puzzle_and_a_stratified_subset():
    from arc26.hyper.validation import gen_puzzles

    records = synthetic()
    puzzles = gen_puzzles(records, barc=3, rearc=7)
    by_key = {p.key: p for p in puzzles}
    barc = sorted(k for k in by_key if k.startswith("barc"))
    assert barc == ["barc-p0", "barc-p1", "barc-p2"]  # the first 3 by sorted id
    assert all(by_key[k].episode_id.endswith("/val/2") for k in barc)  # numeric, not lexical
    rearc = [p for p in puzzles if p.source == "rearc"]
    counts = {f: sum(p.family == f for p in rearc) for f in ("gen-a", "gen-b", "gen-c")}
    assert counts == {"gen-a": 3, "gen-b": 2, "gen-c": 2}  # round-robin over sorted families
    assert {p.puzzle_id for p in rearc if p.family == "gen-a"} == {
        "rearc-gen-a-0", "rearc-gen-a-1", "rearc-gen-a-2"
    }  # fmt: skip
    assert [p.key for p in puzzles] == sorted(p.key for p in puzzles)
    # The query is the test pair, the demonstrations un-augmented (the record's own grids).
    source = next(r for r in records if r.episode_id == "barc/barc-p0/val/2")
    p = by_key["barc-p0"]
    assert p.test_input == source.queries[0][0].tolist()
    assert p.test_output == source.queries[0][1].tolist()
    assert [pair["input"] for pair in p.train] == [x.tolist() for x, _ in source.demos]
    assert p.challenge() == {"train": list(p.train), "test": [{"input": p.test_input}]}
    assert p.solution() == [p.test_output]


def test_gen_keys_hold_only_letters_digits_and_dashes():
    from arc26.hyper.validation import gen_puzzles

    records = [record("barc", "barc:odd_id.x", 0, "barc:odd", 1)]
    (p,) = gen_puzzles(records, barc=1, rearc=0)
    assert p.key == "barc-odd-id-x" and p.puzzle_id == "barc:odd_id.x"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(barc=7, rearc=0), "asks 7 BARC"),
        (dict(barc=0, rearc=16), "asks 16 ReARC"),
        (dict(barc=-1, rearc=0), "integers"),
    ],
)
def test_gen_refuses_what_the_split_cannot_give(change, message):
    from arc26.hyper.validation import gen_puzzles

    with pytest.raises(ValueError, match=message):
        gen_puzzles(synthetic(), **change)


def test_gen_refuses_a_record_that_is_not_a_validation_episode():
    from arc26.hyper.validation import gen_puzzles

    bad = dataclasses.replace(record("barc", "barc-x", 0, "f", 1), episode_id="barc/barc-x/toy/0")
    with pytest.raises(ValueError, match="not a validation episode"):
        gen_puzzles([bad], barc=1, rearc=0)
    other = dataclasses.replace(record("barc", "a", 0, "f", 1), source="arc2train")
    with pytest.raises(ValueError, match="barc and rearc"):
        gen_puzzles([other], barc=1, rearc=0)


def test_write_gen_writes_the_competition_files_and_an_index(tmp_path):
    from arc26.hyper.validation import (
        CHALLENGES_FILE,
        INDEX_FILE,
        SOLUTIONS_FILE,
        gen_puzzles,
        write_gen,
    )

    puzzles = gen_puzzles(synthetic(), barc=2, rearc=3)
    summary = write_gen(puzzles, tmp_path / "gen")
    challenges = json.loads((tmp_path / "gen" / CHALLENGES_FILE).read_text(encoding="utf-8"))
    solutions = json.loads((tmp_path / "gen" / SOLUTIONS_FILE).read_text(encoding="utf-8"))
    index = json.loads((tmp_path / "gen" / INDEX_FILE).read_text(encoding="utf-8"))
    assert set(challenges) == set(solutions) == {p.key for p in puzzles}
    assert all("output" not in t for c in challenges.values() for t in c["test"])
    assert summary["puzzles"] == 5 and summary["by_source"] == {"barc": 2, "rearc": 3}
    assert summary["rearc_families"] == {"gen-a": 1, "gen-b": 1, "gen-c": 1}
    assert index == summary["index"] and {e["source"] for e in index} == {"barc", "rearc"}
    assert "test_output" not in json.dumps(index)


def test_gen_from_the_episode_fixture(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from arc26.hyper.train import EpisodeSetError
    from arc26.hyper.validation import gen_from_episodes

    sha = _fixture_sha()
    summary = gen_from_episodes(
        FIXTURE / "episode_build", FIXTURE / "check", sha, tmp_path / "gen", barc=2, rearc=1
    )
    assert summary["puzzles"] == 3 and summary["val_episodes"] == 6
    assert sorted(e["key"] for e in summary["index"]) == [
        "barc-fixture-flipud", "barc-fixture-invert", "fixture-rot90"
    ]  # fmt: skip
    with pytest.raises(EpisodeSetError):
        gen_from_episodes(
            FIXTURE / "episode_build", FIXTURE / "check", "0" * 64, tmp_path / "x", barc=1, rearc=0
        )


def _fixture_sha():
    import hashlib

    return hashlib.sha256((FIXTURE / "episode_build" / "MANIFEST.json").read_bytes()).hexdigest()


# --- the load check ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny H from train.py's smoke path (one process), its checkpoints and logs."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    from arc26.hyper.train import train

    spec = importlib.util.spec_from_file_location(
        "_t_train_hypernet", REPO / "scripts" / "train_hypernet.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    root = tmp_path_factory.mktemp("h")
    cfg = script.load_config(
        REPO / "configs" / "train_hypernet_toy.yaml",
        True,
        {"out_dir": str(root / "run_logs"), "ckpt_dir": str(root / "checkpoints")},
    )
    cfg = dataclasses.replace(cfg, gpus=1)
    assert train(cfg) == 0
    return {"root": root, "cfg": cfg}


def check_settings(trained, **changes):
    cfg = trained["cfg"]
    values = {
        "checkpoint": str(trained["root"] / "checkpoints" / "best.pt"),
        "episodes_root": cfg.episodes_root,
        "check_root": cfg.check_root,
        "manifest_sha256": cfg.manifest_sha256,
        "run_logs": str(trained["root"] / "run_logs"),
        "hypernet": cfg.to_dict()["hypernet"],
        "model_dir": None,
        "cache": None,
        "device": "cpu",
        "max_abs": 1e-3,
        "max_mean": 1e-4,
        "dev4_challenges": None,
        "timing_dir": None,
    }
    values.update(changes)
    return values


def test_the_load_check_reproduces_the_logged_best_point(trained, tmp_path):
    from arc26.hyper.validation import load_check, logged_best

    best = logged_best(trained["root"] / "run_logs")
    (tmp_path / "dev4.json").write_text(
        json.dumps({"0934a4d8": {"train": [{"input": [[1, 2]], "output": [[2, 1]]},
                                           {"input": [[3]], "output": [[3]]}],
                                 "test": [{"input": [[4, 5]]}]}}),
        encoding="utf-8",
    )  # fmt: skip
    report = load_check(check_settings(trained, dev4_challenges=str(tmp_path / "dev4.json"),
                                       timing_dir=str(tmp_path / "timing")))  # fmt: skip
    assert report["verdict"] == "pass", report["reasons"]
    assert report["logged_best_step"] == best["step"] == report["checkpoint_step"]
    assert report["current"]["max_abs"] < 1e-5 and report["zero"]["max_abs"] < 1e-5
    assert report["episodes"] == 6 == len(report["per_episode"]["fp32"])
    # The bf16 cast is measured on every episode; H's gain over the zero adapter is reported.
    assert report["bf16_minus_fp32"]["n"] == 6 and "flagged" in report["bf16_minus_fp32"]
    assert report["fp32_minus_zero"]["mean"] == pytest.approx(
        best["ce"]["current"] - best["ce"]["zero"], abs=1e-5
    )
    # DEV4's three emissions agree; the stand-in file holds one of the four ids.
    assert report["dev4"]["equal"] is True
    assert report["dev4"]["missing"] == ["36a08778", "981571dc", "aa4ec2a5"]
    ops = {json.loads(line)["op"] for line in
           (tmp_path / "timing" / "timing-load-check.jsonl").read_text().splitlines()}  # fmt: skip
    assert {"load", "embed", "validate_zero", "validate_fp32", "validate_bf16", "dev4_emit"} <= ops


def test_another_checkpoint_fails_the_load_check(trained):
    from arc26.hyper.validation import load_check, logged_best

    best = logged_best(trained["root"] / "run_logs")["step"]
    other = next(
        p for p in sorted((trained["root"] / "checkpoints").glob("ckpt-*.pt"))
        if int(p.stem.split("-")[1]) != best
    )  # fmt: skip
    report = load_check(check_settings(trained, checkpoint=str(other)))
    assert report["verdict"] == "fail"
    assert any("the log's best point" in reason for reason in report["reasons"])
    assert any("current H" in reason for reason in report["reasons"])
    assert not any("zero adapter" in reason for reason in report["reasons"])


def test_perturbed_weights_fail_the_load_check(trained, tmp_path):
    import torch

    from arc26.hyper.validation import load_check

    payload = torch.load(trained["root"] / "checkpoints" / "best.pt", weights_only=False)
    name = next(k for k in payload["hypernet"] if k.startswith("heads.") and k.endswith("weight"))
    payload["hypernet"][name] = payload["hypernet"][name] + 1e-2
    torch.save(payload, tmp_path / "best.pt")
    report = load_check(check_settings(trained, checkpoint=str(tmp_path / "best.pt")))
    assert report["verdict"] == "fail" and report["current"]["max_abs"] > 1e-3


def test_the_config_manifest_and_tolerance_rules(trained):
    from arc26.hyper.validation import load_check

    hypernet = {**trained["cfg"].to_dict()["hypernet"], "dropout": 0.5}
    report = load_check(check_settings(trained, hypernet=hypernet))
    assert report["verdict"] == "fail" and "hypernet config" in report["reasons"][0]
    # The run attaches another build than the one H was validated on: a reason, so a fail.
    report = load_check(check_settings(trained, manifest_sha256="0" * 64))
    assert report["verdict"] == "fail" and "manifest" in report["reasons"][0]
    # The log rounds to 6 decimals: a tolerance below that fails a faithful recomputation.
    strict = load_check(check_settings(trained, max_abs=1e-12, max_mean=1e-12))
    assert strict["verdict"] == "fail"


def test_the_cli_writes_the_report_and_exits_by_verdict(trained, tmp_path):
    from arc26.hyper.validation import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main

    out = tmp_path / "check.json"
    settings = check_settings(trained, out=str(out))
    assert main(["check", "--config", json.dumps(settings)]) == EXIT_PASS
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "pass"
    settings["max_abs"] = 1e-12
    assert main(["check", "--config", json.dumps(settings)]) == EXIT_FAIL
    settings["checkpoint"] = str(tmp_path / "missing.pt")
    assert main(["check", "--config", json.dumps(settings)]) == EXIT_ERROR
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "error"
    shutil.rmtree(tmp_path / "gen", ignore_errors=True)
    gen = {
        "episodes_root": str(FIXTURE / "episode_build"), "check_root": str(FIXTURE / "check"),
        "manifest_sha256": _fixture_sha(), "out_dir": str(tmp_path / "gen"), "barc": 1,
        "rearc": 1, "out": str(tmp_path / "gen.json"),
    }  # fmt: skip
    assert main(["gen", "--config", json.dumps(gen)]) == EXIT_PASS
    assert json.loads((tmp_path / "gen.json").read_text(encoding="utf-8"))["puzzles"] == 2


# --- fix round 1: the bf16 flag rule (task brief amendment 3) ------------------------------------


@pytest.mark.parametrize(
    ("mean", "spread", "z", "flagged"),
    [
        (1.0, 0.5, 3.464, True),  # well above 2 SE
        (1.0, 0.85, 2.038, True),  # just above 2 SE
        (1.0, 0.88, 1.968, False),  # just below
        (1.0, 1.0, 1.732, False),
        (-1.0, 0.5, -3.464, False),  # bf16 below float32 is never flagged
    ],
)
def test_the_bf16_flag_is_a_mean_cost_above_two_paired_standard_errors(mean, spread, z, flagged):
    from arc26.hyper.validation import bf16_flag

    # Four differences mean +- spread: sample SD spread * sqrt(4/3), SE spread / sqrt(3).
    stats = bf16_flag([mean + spread, mean - spread, mean + spread, mean - spread])
    assert stats["mean"] == pytest.approx(mean) and stats["n"] == 4
    assert stats["se"] == pytest.approx(spread / 3**0.5)
    assert stats["z"] == pytest.approx(z, abs=1e-3) and stats["flagged"] is flagged


def test_the_bf16_flag_without_spread():
    from arc26.hyper.validation import bf16_flag

    assert bf16_flag([1e-4] * 3) == {"mean": 1e-4, "se": 0.0, "n": 3, "z": None, "flagged": True}
    assert bf16_flag([0.0] * 3)["flagged"] is False
