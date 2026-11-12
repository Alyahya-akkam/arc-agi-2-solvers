"""Checks scripts/build_episodes.py and scripts/build_episodes_kernel.py. No network, no Kaggle.

Every source is synthetic: the smoke run's files in the real formats, served by ``file://`` URLs
and a local git repository, and a synthetic evaluation set. The push folders go through the
gateway's own CPU-push checks, with ``scripts/kaggle_ops.py`` loaded as the other tests load it,
``KAGGLE_OWNER`` set, and the kaggle CLI replaced by a recorder.
"""

import copy
import hashlib
import importlib.util
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from arc26.hyper import episodes as ep

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
CONFIG = REPO / "configs" / "episodes.yaml"
OWNER = "tester"


def load(name, register=True):
    if register and name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    if register:  # dataclasses with postponed annotations look their module up
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


be = load("build_episodes")
kb = load("build_episodes_kernel")
ops = load("kaggle_ops", register=False)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    return out, be.smoke(out)


@pytest.fixture(scope="module")
def smoke_inputs(tmp_path_factory):
    """Synthetic sources and their config, written once, for builds the tests run themselves."""
    root = tmp_path_factory.mktemp("inputs")
    cfg = be.smoke_config()
    cfg["sources"], competition, expected = be.smoke_sources(root / "sources", cfg["seed"])
    be.validate_config(cfg)
    return cfg, competition, expected


def manifest_bytes(folder):
    return (folder / be.MANIFEST).read_bytes()


def all_records(folder):
    manifest = json.loads(manifest_bytes(folder))
    paths = [folder / e["path"] for e in manifest["files"] if e["path"].startswith("episodes/")]
    return list(ep.read_shards(paths))


CODE = {"commit": "0" * 40, "dirty": False}


# Configuration.


def test_committed_config_is_valid_and_carries_the_notes_pins():
    cfg = be.read_config(CONFIG)
    src = cfg["sources"]
    assert src["arc2train"]["commit"] == "f3283f727488ad98fe575ea6a5ac981e4a188e49"
    assert src["rearc"]["commit"] == "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce"
    assert src["barc"]["revision"] == "c2c13194e8e144ae04236c4b12af46616df43adc"
    # The note prints the first 8 and last 4 hex digits of each sha256.
    for sha, head, tail in (
        (src["rearc"]["sha256"], "4a7c3094", "9ac4"),
        (src["barc"]["sha256"], "b84560f8", "2498"),
    ):
        assert sha.startswith(head) and sha.endswith(tail)
    assert "arc_original" not in json.dumps(cfg)  # gate G0: never ARC-AGI-1 evaluation tasks
    held = cfg["val"]["rearc_held_out"]
    dropped_by_arc2 = {
        "0dfd9992",
        "29ec7d0e",
        "3631a71a",
        "40853293",
        "73251a56",
        "9ecd008a",
        "a3df8b1e",
        "c3f564a4",
        "dc0a314f",
    }
    assert len(held) == 20 and dropped_by_arc2 <= set(held)
    assert sum(w for _, _, w in cfg["val"]["sizes"]) == 120  # DEV120's tasks
    assert (cfg["max_tokens"], cfg["demo_probs"]) == (8192, [0.15, 0.35, 0.30, 0.15, 0.05])
    # The drop list grows by the checker's drop, rebuild, re-check loop (note 2.3); every
    # entry names a configured source, a task id, and its reason.
    for d in cfg["drop"]:
        assert set(d) == {"source", "task_id", "reason"} and d["source"] in src
        assert d["task_id"] and d["reason"]


def test_every_config_value_has_a_comment():
    lines = CONFIG.read_text(encoding="utf-8").split("\n")
    key = re.compile(r"^(\s*)([a-z_0-9]+):(.*)$")
    for i, line in enumerate(lines):
        if line.startswith("smoke:"):
            break
        match = key.match(line)
        if not match:
            continue
        rest = match.group(3).split("#")[0].strip()
        following = next(
            (x.strip() for x in lines[i + 1 :] if x.strip() and not x.strip().startswith("#")), ""
        )
        if not rest and not following.startswith("-"):
            continue  # a mapping: its values carry the comments
        previous = next(x.strip() for x in reversed(lines[:i]) if x.strip())
        assert previous.startswith("#") or "#" in line, (
            f"line {i + 1} ({match.group(2)}) has no comment"
        )


def _set(path, value):
    def apply(cfg):
        node = cfg
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


@pytest.mark.parametrize(
    "mutate",
    [
        _set(("extra",), 1),
        lambda cfg: cfg.pop("seed"),
        _set(("sources", "rearc", "sha256"), "4a7c3094"),
        _set(("sources", "rearc", "url"), "https://example.com/re_arc.zip"),
        _set(
            ("sources", "barc", "url"),
            "http://huggingface.co/c2c13194e8e144ae04236c4b12af46616df43adc",
        ),
        _set(("demo_probs",), [0.5, 0.4]),
        _set(("demo_probs",), [1.0, 0.0]),
        _set(("train", "barc_tasks"), -1),
        _set(("train", "barc_tasks"), True),
        _set(("val", "rearc_held_out"), [40853293]),
        _set(("val", "rearc_held_out"), ["0dfd9992", "0dfd9992"]),
        _set(("val", "sizes"), [[2, 1]]),
        _set(("toy", "pairs_per_puzzle"), 10),
        _set(("drop",), [{"source": "arc2train", "task_id": "x"}]),
        _set(("drop",), [{"source": "arc1", "task_id": "x", "reason": "r"}]),
        _set(("shards", "train"), 0),
        _set(("drop",), [{"source": "barc", "task_id": "seed:zz", "reason": "r"}]),
    ],
)
def test_config_rules(mutate):
    cfg = be.read_config(CONFIG)
    mutate(cfg)
    with pytest.raises(be.ConfigError):
        be.validate_config(cfg)


def test_smoke_section_may_only_override_existing_keys(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["smoke"]["train"]["typo"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(be.ConfigError, match="typo"):
        be.read_config(path, smoke=True)


# The smoke loop.


def test_smoke_flags_exactly_the_planted_leaks_then_passes(smoke_run):
    out, summary = smoke_run
    assert summary["ok"], summary["failures"]
    assert summary["first"]["pass"] is False and summary["second"]["pass"] is True
    first = json.loads((out / "check-1" / "run_logs" / "leakage.json").read_text())
    assert first["positive_control"] == {
        "injected": 3,
        "caught_g3": 3,
        "caught_g2": 0,
        "pass": True,
    }
    flagged = {(h["source"], h["task_id"]) for h in first["hits"]}
    assert {(d["source"], d["task_id"]) for d in first["drop"]} == flagged
    assert all(v["tasks"] > 0 for v in first["gates"].values())  # every gate fired once
    second = json.loads((out / "check-2" / "run_logs" / "leakage.json").read_text())
    assert second["pass"] and second["hits"] == []
    assert second["reports"]["R3"]["canonical_differs"] == ["a1000001"]


def test_main_smoke_prints_its_summary(tmp_path, capsys):
    assert be.main(["--smoke", "--out", str(tmp_path / "s")]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


# Build outputs.


def test_build_layout_and_manifest(smoke_run):
    out, _ = smoke_run
    folder = out / "build-2"
    manifest = json.loads(manifest_bytes(folder))
    listed = {e["path"] for e in manifest["files"]}
    written = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert written - listed == {
        "MANIFEST.json",
        "run_logs/MANIFEST.json",
        "run_logs/episodes_stats.json",
        "run_logs/episodes_stats.md",
    }
    for entry in manifest["files"]:
        data = (folder / entry["path"]).read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == (entry["bytes"], entry["sha256"])
    assert (folder / "run_logs" / be.MANIFEST).read_bytes() == manifest_bytes(folder)
    records = all_records(folder)
    assert len(records) == sum(manifest["counts"][s] for s in ep.SPLITS)
    table = ep.PermutationTable.from_strings(manifest["augmentation"]["permutations"])
    assert len(table.permutations) == 16 and all(
        r.augmentation_id < table.n_augmentations for r in records
    )
    assert manifest["config_sha256"] == be.config_sha256(manifest["config"])
    assert manifest["inputs"]["arc2train"]["git"].startswith("git version")
    stats = json.loads((folder / "run_logs" / "episodes_stats.json").read_text())
    assert stats["manifest_sha256"] == hashlib.sha256(manifest_bytes(folder)).hexdigest()
    assert "Token lengths" in (folder / "run_logs" / "episodes_stats.md").read_text()


def test_only_run_logs_match_the_pull_allowlist(smoke_run):
    out, _ = smoke_run
    for folder in (out / "build-2", out / "check-2"):
        for path in folder.rglob("*"):
            if path.is_file():
                rel = path.relative_to(folder).as_posix()
                assert ops.pull_allowed(rel) == rel.startswith("run_logs/"), rel


def test_splits_are_disjoint_by_family_and_by_canonical_pair(smoke_run):
    out, _ = smoke_run
    records = all_records(out / "build-2")
    by_split = {s: [r for r in records if r.split == s] for s in ep.SPLITS}
    assert all(by_split.values())
    training = by_split["train"] + by_split["toy"]
    assert not {r.family for r in by_split["val"]} & {r.family for r in training}
    val_keys = {ep.canonical_pair_hash(x, y) for r in by_split["val"] for x, y in r.pairs}
    assert not val_keys & {ep.canonical_pair_hash(x, y) for r in training for x, y in r.pairs}
    # The held-out generator a1000000 takes the ARC task of the same id out of training.
    assert "a1000000" not in {r.task_id for r in training}
    assert {r.task_id for r in by_split["val"] if r.source == "rearc"} == {"a1000000", "b1000000"}


def test_validation_puzzles_are_in_the_competition_format(smoke_run):
    out, _ = smoke_run
    folder = out / "build-2"
    challenges = json.loads((folder / "val" / "val_challenges.json").read_text())
    solutions = json.loads((folder / "val" / "val_solutions.json").read_text())
    sizes = {(k, m) for k, m, _ in be.read_config(CONFIG)["val"]["sizes"]}
    assert set(challenges) == set(solutions)
    for puzzle_id, task in challenges.items():
        assert re.fullmatch(r"[a-z0-9-]+", puzzle_id)  # the notebook splits keys on "_" and "."
        assert (len(task["train"]), len(task["test"])) in sizes
        assert len(solutions[puzzle_id]) == len(task["test"])
    val = [r for r in all_records(folder) if r.split == "val"]
    assert len(val) == sum(len(s) for s in solutions.values())
    for r in val:
        puzzle_id = r.episode_id.split("/")[1]
        assert [x.tolist() for x, _ in r.demos] == [
            p["input"] for p in challenges[puzzle_id]["train"]
        ]


def test_records_validate_and_fit_the_token_limit(smoke_run):
    out, _ = smoke_run
    records = all_records(out / "build-2")
    assert max(r.tokens for r in records) <= 8192
    assert all(r.n_demos >= 1 and len(r.queries) == 1 for r in records)


def test_same_seed_gives_the_same_manifest_hash(smoke_inputs, tmp_path):
    cfg, _, _ = smoke_inputs
    first = be.run_build(cfg, tmp_path / "a", tmp_path / "work-a", CODE)
    be.run_build(cfg, tmp_path / "b", tmp_path / "work-b", CODE)
    assert manifest_bytes(tmp_path / "a") == manifest_bytes(tmp_path / "b")
    other = copy.deepcopy(cfg)
    other["seed"] = 1
    be.run_build(other, tmp_path / "c", tmp_path / "work-c", CODE)
    assert manifest_bytes(tmp_path / "c") != manifest_bytes(tmp_path / "a")
    assert first["counts"]["train"] > 0


def test_dropping_a_task_leaves_every_other_episode_unchanged(smoke_inputs, tmp_path):
    cfg, _, _ = smoke_inputs
    be.run_build(cfg, tmp_path / "a", tmp_path / "work-a", CODE)
    dropped = copy.deepcopy(cfg)
    dropped["drop"] = [{"source": "arc2train", "task_id": "a1000004", "reason": "test"}]
    be.run_build(dropped, tmp_path / "b", tmp_path / "work-b", CODE)
    before = {r.episode_id: r.to_json() for r in all_records(tmp_path / "a")}
    after = {r.episode_id: r.to_json() for r in all_records(tmp_path / "b")}
    assert not any(i.startswith("arc2train/a1000004/") for i in after)
    assert any(i.startswith("rearc/a1000004/") for i in after)  # another source keeps its own
    common = set(before) & set(after)
    assert len(common) > 0.8 * len(after)
    assert all(before[i] == after[i] for i in common)


def _task(source, task_id, pairs):
    return ep.SourceTask(
        source, task_id, task_id, [tuple(map(ep.validate_grid, p)) for p in pairs], len(pairs)
    )


def _random_pairs(rng, n, low=3, high=8):
    return [
        (
            rng.integers(0, 10, size=rng.integers(low, high + 1, size=2)),
            rng.integers(0, 10, size=rng.integers(low, high + 1, size=2)),
        )
        for _ in range(n)
    ]


def _small_config(**val):
    cfg = be.smoke_config()
    cfg["val"].update(
        {
            "rearc_held_out": ["h0000000"],
            "rearc_puzzles_per_family": 1,
            "barc_puzzles": 0,
            "sizes": [[2, 1, 1]],
            **val,
        }
    )
    cfg["toy"].update({"barc": 0, "rearc": 0, "arc2train": 0})
    cfg["train"].update({"barc_tasks": 0, "rearc_episodes": 20, "arc2train_episodes": 12})
    return cfg


def test_a_g1_seed_hit_enters_the_drop_list_and_the_rebuild_passes(smoke_run):
    out, _ = smoke_run
    first = json.loads((out / "check-1" / "run_logs" / "leakage.json").read_text())
    seed_hits = [h for h in first["hits"] if h["task_id"] == "seed:e1000004"]
    assert [(h["gate"], h["source"], h["split"]) for h in seed_hits] == [
        ("G1", "barc", "provenance")
    ]
    assert {"source": "barc", "task_id": "seed:e1000004"} in first["drop"]
    seeded = ep.barc_task_id("smoke", "def main(grid):  # smoke program 5\n    return grid\n")
    builds = [json.loads(manifest_bytes(out / b)) for b in ("build-1", "build-2")]
    assert "e1000004" in builds[0]["provenance"]["barc_seed_ids"]
    assert "e1000004" not in builds[1]["provenance"]["barc_seed_ids"]
    assert {"source": "barc", "task_id": "seed:e1000004", "reason": "smoke leakage check"} in (
        builds[1]["config"]["drop"]
    )
    assert seeded not in {r.task_id for r in all_records(out / "build-2")}
    second = json.loads((out / "check-2" / "run_logs" / "leakage.json").read_text())
    assert second["pass"] and second["gates"]["G1"]["tasks"] == 0


def test_a_seed_entry_drops_every_problem_that_names_the_seed():
    rng = np.random.default_rng(2)
    barc = {}
    for i, seeds in enumerate([("e0000000",), ("e0000000", "a0000000"), ("a0000000",)]):
        task = _task("barc", f"barc:t:{i:016x}", _random_pairs(rng, 6))
        task.seeds = seeds
        barc[task.task_id] = task
    sources = be.Sources(
        arc2train={"a0000001": _task("arc2train", "a0000001", _random_pairs(rng, 4))},
        rearc={"h0000000": _task("rearc", "h0000000", _random_pairs(rng, 9))},
        barc=barc,
    )
    cfg = _small_config()
    cfg["train"]["barc_tasks"] = 10
    cfg["drop"] = [{"source": "barc", "task_id": "seed:e0000000", "reason": "test"}]
    result = be.build_episode_set(cfg, sources)
    assert result.stats["dropped_by_list"]["barc"] == 2
    assert {r.task_id for r in result.train if r.source == "barc"} == {"barc:t:0000000000000002"}
    assert result.provenance["barc_seed_ids"] == ["a0000000"]


def test_code_version_reads_the_dirty_suffix():
    assert be.code_version("c" * 40) == {"commit": "c" * 40, "dirty": False}
    assert be.code_version("c" * 40 + "+dirty") == {"commit": "c" * 40, "dirty": True}


def test_training_tasks_with_a_validation_pair_are_dropped():
    rng = np.random.default_rng(0)
    held = _random_pairs(rng, 3)  # exactly one 2 + 1 puzzle: every pair is a validation pair
    x, y = held[1]
    copy_of_val = (
        ep.Augmentation(True, 1, (3, 8, 1, 0, 2, 4, 5, 6, 7, 9)).apply(x),
        ep.Augmentation(True, 1, (3, 8, 1, 0, 2, 4, 5, 6, 7, 9)).apply(y),
    )
    sources = be.Sources(
        arc2train={
            "a0000000": _task("arc2train", "a0000000", [copy_of_val, *_random_pairs(rng, 3)]),
            "a0000001": _task("arc2train", "a0000001", _random_pairs(rng, 4)),
        },
        rearc={
            "h0000000": _task("rearc", "h0000000", held),
            "r0000000": _task("rearc", "r0000000", _random_pairs(rng, 40)),
        },
        barc={},
    )
    result = be.build_episode_set(_small_config(), sources)
    assert result.stats["splits"]["train"]["arc2train"]["cross_split_tasks"] == 1
    tasks = {r.task_id for r in result.train + result.toy}
    assert "a0000000" not in tasks and {"a0000001", "r0000000"} <= tasks
    assert [p.puzzle_id for p in result.puzzles] == ["rearc-h0000000-0"]


def test_over_length_episodes_are_counted_per_source_and_split():
    rng = np.random.default_rng(1)
    big = [
        (rng.integers(0, 10, size=(30, 30)), rng.integers(0, 10, size=(30, 30))) for _ in range(40)
    ]
    sources = be.Sources(
        arc2train={"a0000001": _task("arc2train", "a0000001", _random_pairs(rng, 4))},
        rearc={
            "h0000000": _task("rearc", "h0000000", _random_pairs(rng, 9)),
            "r0000000": _task("rearc", "r0000000", big),
        },
        barc={},
    )
    cfg = _small_config()
    cfg["demo_probs"] = [0.05, 0.05, 0.1, 0.4, 0.4]  # mostly 4 or 5 demos: 9,330 tokens or more
    counts = be.build_episode_set(cfg, sources).stats["splits"]["train"]["rearc"]
    assert counts["over_length"] > 0
    assert (
        counts["episodes"] + counts["over_length"] + counts.get("short", 0)
        == counts["requested"]
        == 20
    )


# Fetching.


def test_download_verifies_size_sha256_and_git_blob(tmp_path):
    data = b"synthetic source bytes\n"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    url = source.as_uri()
    sha, blob = hashlib.sha256(data).hexdigest(), be._git_blob(data)
    info = be.download(url, tmp_path / "out" / "a.bin", sha256=sha, size=len(data), git_blob=blob)
    assert (tmp_path / "out" / "a.bin").read_bytes() == data and info["git_blob"] == blob
    for kwargs in (
        {"sha256": "0" * 64, "size": len(data)},
        {"sha256": sha, "size": len(data) + 1},
        {"sha256": sha, "size": len(data), "git_blob": "0" * 40},
    ):
        with pytest.raises(be.PinError):
            be.download(url, tmp_path / "out" / "b.bin", **kwargs)
        assert (
            not (tmp_path / "out" / "b.bin").exists()
            and not (tmp_path / "out" / "b.bin.part").exists()
        )
    with pytest.raises(be.PinError, match="after 2 attempts"):
        be.download(
            (tmp_path / "missing").as_uri(), tmp_path / "c.bin", sha256=sha, size=1, attempts=2
        )


def test_fetch_checks_out_training_only_and_never_an_evaluation_blob(smoke_inputs, tmp_path):
    cfg, _, _ = smoke_inputs
    spec = cfg["sources"]["arc2train"]
    training, info = be.fetch_arc_training(spec, tmp_path / "clone")
    clone = tmp_path / "clone"
    assert sorted(p.name for p in (clone / "data").iterdir()) == ["training", "training.txt"]
    assert len(list(training.glob("*.json"))) == spec["tasks"] == info["files"]
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=clone, env=env, capture_output=True, text=True, check=True
        ).stdout

    present = {
        line.split()[0]
        for line in git(
            "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)"
        ).splitlines()
        if line.endswith(" blob")
    }
    evaluation = {
        line.split()[2] for line in git("ls-tree", "-r", "HEAD", "data/evaluation").splitlines()
    }
    wanted = {
        line.split()[2] for line in git("ls-tree", "-r", "HEAD", "data/training").splitlines()
    }
    assert len(evaluation) == 6 and not present & evaluation
    assert present == wanted | {git("rev-parse", "HEAD:data/training.txt").strip()}
    for key in ("tree", "index_blob"):
        with pytest.raises(be.PinError, match=key):
            be.fetch_arc_training({**spec, key: "0" * 40}, tmp_path / f"bad-{key}")
    with pytest.raises(be.PinError):
        be.fetch_arc_training({**spec, "commit": "0" * 40}, tmp_path / "bad-commit")


# Checking.


def test_check_refuses_a_tampered_build(smoke_run, tmp_path):
    out, _ = smoke_run
    folder = tmp_path / "build"
    shutil.copytree(out / "build-2", folder)
    shard = next((folder / "episodes").glob("train-*.jsonl"))
    shard.write_bytes(shard.read_bytes().replace(b'"augmentation_id":', b'"augmentation_id": ', 1))
    with pytest.raises(be.BuildError, match="differs from the manifest"):
        be.verify_build(folder)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))), encoding="utf-8")
    argv = [
        "check",
        "--config",
        str(cfg),
        "--episodes",
        str(folder),
        "--competition",
        str(out / "sources" / "competition"),
        "--out",
        str(tmp_path / "check"),
    ]
    assert be.main(argv) == 1
    assert (
        "differs from the manifest" in (tmp_path / "check" / "run_logs" / "error.log").read_text()
    )


def test_locate_build_and_competition(smoke_run, tmp_path):
    out, _ = smoke_run
    root = tmp_path / "input"
    shutil.copytree(out / "build-2", root / "notebooks" / "tester" / "arc26-p3-4-episodes")
    shutil.copytree(out / "sources" / "competition", root / "competitions")
    decoy = root / "datasets" / "copy"
    decoy.mkdir(parents=True)
    (decoy / be.EVAL_CHALLENGES).write_text("{}", encoding="utf-8")
    assert be.locate_build(root) == root / "notebooks" / "tester" / "arc26-p3-4-episodes"
    paths = be.locate_competition(root)
    assert be.COMPETITION in str(paths["eval_challenges"]) and paths["train_solutions"] is not None
    shutil.copytree(out / "build-1", root / "second")
    with pytest.raises(be.BuildError, match="found 2"):
        be.locate_build(root)
    with pytest.raises(be.BuildError, match="no arc-agi_evaluation_challenges.json"):
        be.locate_competition(tmp_path / "nowhere")


def test_check_reports_hold_ids_and_counts_never_grids(smoke_run):
    out, _ = smoke_run

    def no_grids(value):
        if isinstance(value, dict):
            return all(no_grids(v) for v in value.values())
        if isinstance(value, list):
            return all(not isinstance(v, list) and no_grids(v) for v in value)
        return True

    for check in ("check-1", "check-2"):
        logs = out / check / "run_logs"
        assert sorted(p.name for p in logs.iterdir()) == ["leakage.json", "leakage.md"]
        assert no_grids(json.loads((logs / "leakage.json").read_text()))
        assert "[[" not in (logs / "leakage.md").read_text()
    assert "**FAIL**" in (out / "check-1" / "run_logs" / "leakage.md").read_text()
    assert "**PASS**" in (out / "check-2" / "run_logs" / "leakage.md").read_text()


def test_main_build_then_check_writes_the_report_whatever_the_verdict(smoke_inputs, tmp_path):
    cfg, competition, _ = smoke_inputs
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw.update({k: v for k, v in cfg.items()})
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    build = [
        "build",
        "--config",
        str(path),
        "--out",
        str(tmp_path / "b"),
        "--work",
        str(tmp_path / "w"),
        "--code-commit",
        "1" * 40,
    ]
    assert be.main(build) == 0
    check = [
        "--config",
        str(path),
        "check",
        "--episodes",
        str(tmp_path),
        "--competition",
        str(competition.parent),
        "--out",
        str(tmp_path / "c"),
    ]
    assert be.main(check) == 0  # the planted leaks fail the gates; the verdict is in the report
    report = json.loads((tmp_path / "c" / "run_logs" / "leakage.json").read_text())
    assert report["pass"] is False and report["build"]["code"] == {
        "commit": "1" * 40,
        "dirty": False,
    }


@pytest.mark.parametrize("argv", [[], ["--smoke", "build", "--out", "x", "--work", "y"]])
def test_main_needs_exactly_one_mode(argv):
    assert be.main(argv) == 2


def test_main_refuses_an_invalid_config(tmp_path, capsys):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["seed"] = -1
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert (
        be.main(
            [
                "build",
                "--config",
                str(path),
                "--out",
                str(tmp_path / "o"),
                "--work",
                str(tmp_path / "w"),
            ]
        )
        == 2
    )
    assert "seed" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


# Push folders.


def slugify(title):
    """python-slugify's default for an ASCII title, which kaggle-cli compares with the slug."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_folder(tmp_path, kernel):
    version = 3 if kernel == "checker" else None
    return kb.build(tmp_path / kernel, OWNER, kernel, version, commit="a" * 40)


def no_subprocess(*args, **kwargs):
    raise AssertionError(f"a test tried to start a process: {args}")


@pytest.mark.parametrize("kernel", sorted(kb.KERNELS))
def test_kernel_metadata(tmp_path, kernel):
    meta = json.loads((build_folder(tmp_path, kernel) / "kernel-metadata.json").read_text())
    checker = kernel == "checker"
    assert meta == {
        "id": f"{OWNER}/arc26-p3-4-{'leakage' if checker else 'episodes'}",
        "title": f"arc26 P3.4 {'leakage' if checker else 'episodes'}",
        "code_file": f"arc26_p3_4_{'leakage' if checker else 'episodes'}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": not checker,
        "docker_image_pinning_type": "latest",
        "competition_sources": ["arc-prize-2026-arc-agi-2"] if checker else [],
        "model_sources": [],
        "kernel_sources": [f"{OWNER}/arc26-p3-4-episodes/3"] if checker else [],
        "dataset_sources": [],
    }
    assert slugify(meta["title"]) == meta["id"].split("/")[1]
    assert sorted(p.name for p in (tmp_path / kernel).iterdir()) == sorted(
        [meta["code_file"], "kernel-metadata.json"]
    )


@pytest.mark.parametrize("kernel", sorted(kb.KERNELS))
def test_push_folders_pass_the_gateway_checks_as_cpu_pushes(tmp_path, monkeypatch, kernel):
    calls = []
    monkeypatch.setattr(ops, "KAGGLE_OWNER", OWNER)
    monkeypatch.setattr(ops, "_kaggle", lambda *args: calls.append(list(args)) or 0)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    folder = build_folder(tmp_path, kernel).resolve()
    meta = ops.load_kernel_metadata(folder)
    assert ops.uses_gpu(meta, None) is False and ops.accelerator_id(meta, None) is None
    ops.check_kernel_metadata(meta, False)
    ops.check_flags(meta, None)
    ops.check_code_file(folder, meta)
    timeout = "7200" if kernel == "builder" else "3600"
    assert ops.main(["push", str(folder), "--timeout", timeout]) == 0
    assert calls == [["kernels", "push", "-p", str(folder), "-t", timeout]]


def test_slugs_are_new_and_named_by_no_brief():
    slugs = {spec["slug"] for spec in kb.KERNELS.values()}
    assert slugs == {"arc26-p3-4-episodes", "arc26-p3-4-leakage"}
    assert not slugs & {n.split("/")[-1].lower() for n in ops.briefed_notebooks()}


@pytest.mark.parametrize(
    "argv",
    [
        ["--owner", OWNER, "--kernel", "checker"],
        ["--owner", OWNER, "--kernel", "checker", "--builder-version", "0"],
        ["--owner", OWNER, "--kernel", "builder", "--builder-version", "1"],
        ["--owner", "a/b", "--kernel", "builder"],
        ["--owner", "..", "--kernel", "builder"],
        ["--owner", OWNER, "--kernel", "gpu"],
    ],
)
def test_bad_arguments_are_refused_before_anything_is_written(tmp_path, argv):
    out = tmp_path / "refused"
    try:
        code = kb.main([*argv, "--out", str(out), "--allow-dirty"])
    except SystemExit as exc:  # argparse refused the value
        code = exc.code
    assert code == 2 and not out.exists()


def test_a_dirty_tree_is_refused_unless_allowed(monkeypatch):
    def fake_run(args, **kwargs):
        stdout = "b" * 40 + "\n" if "rev-parse" in args else " M scripts/build_episodes.py\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(kb.subprocess, "run", fake_run)
    with pytest.raises(kb.BuildError, match="differ from HEAD"):
        kb.code_commit()
    assert kb.code_commit(allow_dirty=True) == "b" * 40 + "+dirty"


@pytest.mark.parametrize("kernel", sorted(kb.KERNELS))
def test_bundle_carries_the_repo_files_and_the_config(tmp_path, kernel):
    folder = build_folder(tmp_path, kernel)
    bundle = runpy.run_path(str(folder / kb.KERNELS[kernel]["code_file"]), run_name="bundle")
    files = bundle["FILES"]
    assert (
        files["arc26/hyper/episodes.py"] == (REPO / "arc26" / "hyper" / "episodes.py").read_text()
    )
    assert files["build_episodes.py"] == (SCRIPTS / "build_episodes.py").read_text()
    assert json.loads(files["episodes.json"]) == yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert bundle["COMMIT"] == "a" * 40
    argv = bundle["ARGV"]
    assert argv[0] == ("check" if kernel == "checker" else "build")
    assert argv[argv.index("--out") + 1] == "/kaggle/working"
    if kernel == "builder":
        assert argv[argv.index("--work") + 1].startswith("/tmp/")
    else:
        assert (
            argv[argv.index("--episodes") + 1]
            == argv[argv.index("--competition") + 1]
            == "/kaggle/input"
        )
    spec = importlib.util.spec_from_file_location(
        "repo_hygiene", REPO / "tests" / "test_repo_hygiene.py"
    )
    hygiene = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hygiene)
    assert not hygiene.KAGGLE_ACCESS.search((folder / kb.KERNELS[kernel]["code_file"]).read_text())


def test_bundle_runs_standalone_with_its_own_modules(tmp_path):
    folder = build_folder(tmp_path, "builder")
    kernel, out, root = folder / "arc26_p3_4_episodes.py", tmp_path / "out", tmp_path / "root"
    argv = ["--smoke", "--config", "{root}/episodes.json", "--out", str(out)]
    script = (
        "import runpy, sys\n"
        "from pathlib import Path\n"
        f"ns = runpy.run_path({str(kernel)!r}, run_name='bundle')\n"
        f"code = ns['run']({argv!r}, Path({str(root)!r}))\n"
        "mods = sys.modules['arc26.hyper.episodes'], sys.modules['build_episodes']\n"
        "print('MODULES', *(m.__file__ for m in mods))\n"
        "sys.exit(code)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    modules = next(
        line for line in result.stdout.splitlines() if line.startswith("MODULES")
    ).split()[1:]
    assert all(m.startswith(str(tmp_path / "root")) for m in modules)
    assert '"ok": true' in result.stdout
