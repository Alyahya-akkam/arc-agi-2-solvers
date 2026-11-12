"""P3.7: the adapter arms (arc26.arms) on the CPU: settings, views, every arm's adapter, pairing.

Real: a tiny random Qwen3 wrapped by PEFT as ``arc26.solver.load`` wraps the base (rsLoRA, dropout
0, q, k, v, o), P3.2's encoder and cache, a tiny H with random heads (so its adapter depends on
z_bar), ``HyperProvider``, ``HyperPolishProvider``, ``arc26.ttt.fit`` and ``SGDProvider``,
``arc26.decode.decode_puzzle`` (its views through the real batching) and ``calc_scores``, the
reference's selection, ``arc26.evaluate``. Stubbed: unsloth and datasets (the stand-ins of
``scripts/h_validation_report.py``, the CPU smoke path's), and in the runner tests the decode
itself (a random tiny model decodes no grid; the tests record its calls instead). The synthetic
puzzles are never competition data. Skipped without torch, transformers, peft, or safetensors.
"""

import bz2
import importlib.util
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from peft import get_peft_model_state_dict  # noqa: E402

from arc26 import arms as am  # noqa: E402
from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.encoder import (  # noqa: E402
    EmbeddingCache,
    PairEncoder,
    pair_hash,
    view_descriptors,
)
from arc26.hyper.hypernet import BaseShapes, Hypernet, HypernetConfig  # noqa: E402
from arc26.hyper.providers import HyperProvider, adapter_digest  # noqa: E402
from arc26.loop import DemoSet, Pair  # noqa: E402
from arc26.runner import ARMS_MODULE, RunConfig, solver_command  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
H_CONFIG = {
    "embed_dim": 64, "latent": 16, "head_in": 8, "dropout": 0.25, "rank": 4, "alpha": 16,
    "target_modules": TARGETS, "seed": 0,
}  # fmt: skip
LAYER = 2
VIEWS = view_descriptors(1, 2)  # ADR-001 as amended: 8 views
MODEL_ID = "tiny-random-qwen3"
SOURCES = ("base", "hyper", "sgd", "hyper_polish", "polish")
ARMS = [
    {"name": "base", "source": "base"},
    {"name": "h", "source": "hyper"},
    {"name": "sgd", "source": "sgd"},
    {"name": "h-polish", "source": "hyper_polish", "steps": 3},
    {"name": "polish", "source": "polish", "steps": 2},
]
# Two synthetic puzzles: a flip (two test inputs) and a recolor (one), small grids.
PUZZLES = {
    "aaaa0001": {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
            {"input": [[5, 6, 7], [8, 9, 0]], "output": [[7, 6, 5], [0, 9, 8]]},
            {"input": [[1, 1], [2, 3]], "output": [[1, 1], [3, 2]]},
        ],
        "test": [{"input": [[4, 5], [6, 7]]}, {"input": [[2, 3, 4]]}],
    },
    "aaaa0002": {
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
            {"input": [[1, 1, 0]], "output": [[2, 2, 0]]},
        ],
        "test": [{"input": [[0, 1], [1, 1]]}],
    },
}
SOLUTIONS = {
    "aaaa0001": [[[5, 4], [7, 6]], [[4, 3, 2]]],
    "aaaa0002": [[[0, 2], [2, 2]]],
}


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_script("h_validation_report")


@pytest.fixture(scope="module")
def stack():
    """arc26.ttt and arc26.decode against the smoke path's stand-ins; restored afterwards."""
    with report.standins():
        import arc26.decode  # noqa: F401
        import arc26.ttt  # noqa: F401

        yield sys.modules


def write_data(folder, puzzles=PUZZLES, solutions=SOLUTIONS):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / am.CHALLENGES_FILE).write_text(json.dumps(puzzles), encoding="utf-8")
    (folder / am.SOLUTIONS_FILE).write_text(json.dumps(solutions), encoding="utf-8")
    return folder


def settings(tmp, **changes):
    values = {
        "arms": ARMS,
        "outputs_root": str(tmp / "arm_outputs"),
        "log_dir": str(tmp / "logs"),
        "cache_dir": str(tmp / "cache"),
        "checkpoint": str(tmp / "best.pt"),
        "hypernet": H_CONFIG,
        "views": [list(b) for b in am.DIHEDRAL_ONCE],
        "ce": True,
    }
    values.update(changes)
    return am.Arms.from_mapping(values)


def tiny_hyper(model, root, seed=5):
    """H with random non-zero heads on the live model's base, its encoder on the PEFT model."""
    h = Hypernet(HypernetConfig(**H_CONFIG), BaseShapes.from_config(model.config))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in h.heads.values():
            head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.2)
            head.bias.copy_(torch.randn(head.bias.shape, generator=generator) * 0.05)
    encoder = PairEncoder(model, tr.standin_tokenizer(), layer=LAYER, views=VIEWS)
    cache = EmbeddingCache(
        root, model_id=MODEL_ID, layer=LAYER, pool="output", view_set_id=encoder.view_set_id
    )
    return HyperProvider(h, encoder, cache)


def digest(model):
    return adapter_digest(
        get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    )


# --- settings ------------------------------------------------------------------------------------


def test_the_dihedral_once_views_cover_every_map_once_in_two_full_batches():
    flat = [i for batch in am.DIHEDRAL_ONCE for i in batch]
    assert sorted(i // 2 for i in flat) == list(range(len(am.DIHEDRAL_ORDER)))
    assert all(i % 2 == 0 for i in flat)  # the first color permutation of each map
    assert [len(batch) for batch in am.DIHEDRAL_ONCE] == [am.MAX_BATCH, am.MAX_BATCH]
    assert am.check_views(am.DIHEDRAL_ONCE) == am.DIHEDRAL_ONCE
    # The reference's own batches pass the same rules.
    assert am.check_views(am.REFERENCE_VIEWS) == am.REFERENCE_VIEWS


def test_the_length_groups_are_the_maps_that_keep_or_swap_rows():
    keep = {"identity", "rot180", "transpose.rot90", "transpose.rot270"}
    for index in range(am.VIEW_COUNT):
        name = am.DIHEDRAL_ORDER[index // 2]
        assert (index in am.LENGTH_GROUPS[0]) == (name in keep)
        assert (index in am.LENGTH_GROUPS[1]) == (name not in keep)


@pytest.mark.parametrize(
    "views",
    [[], [[]], [[0, 2]], [[0, 1, 4, 5, 10]], [[0], [0]], [[16]], [[-1]], [[True]], "0", [[0.0]]],
)
def test_check_views_refuses_a_bad_selection(views):
    with pytest.raises(ValueError):
        am.check_views(views)


def test_arm_spec_rules():
    assert am.ArmSpec("h-polish", "hyper_polish", 13).steps == 13
    for bad in (
        dict(name="H", source="hyper"),
        dict(name="h", source="other"),
        dict(name="p", source="polish"),
        dict(name="p", source="polish", steps=129),
        dict(name="s", source="sgd", steps=3),
    ):
        with pytest.raises(ValueError):
            am.ArmSpec(**bad)


def test_arms_settings_rules_and_round_trip(tmp_path):
    arms = settings(tmp_path)
    assert am.Arms.from_mapping(arms.to_json()) == arms
    assert json.loads(json.dumps(arms.to_json())) == arms.to_json()
    assert arms.views == am.DIHEDRAL_ONCE
    with pytest.raises(ValueError, match="distinct"):
        settings(tmp_path, arms=[ARMS[0], ARMS[0]])
    with pytest.raises(ValueError, match="H arm needs"):
        settings(tmp_path, checkpoint=None)
    with pytest.raises(ValueError, match="H arm needs"):
        settings(tmp_path, hypernet=None)
    with pytest.raises(ValueError, match="episodes_root"):
        settings(tmp_path, gen={"out_dir": str(tmp_path), "barc": 1, "rearc": 0})
    with pytest.raises(ValueError, match="unknown keys"):
        am.Arms.from_mapping({**arms.to_json(), "extra": 1})
    with pytest.raises(ValueError, match="finish_s"):
        settings(tmp_path, finish_s=-1)
    # Without an H arm, no checkpoint is needed.
    assert settings(tmp_path, arms=ARMS[2:3], checkpoint=None, hypernet=None).checkpoint is None


def test_the_queue_order_is_a_fixed_shuffle():
    keys = [f"{i:08x}" for i in range(40)]
    order = am.queue_order(keys, 0)
    assert sorted(order) == keys and order != keys
    assert am.queue_order(list(reversed(keys)), 0) == order
    assert am.queue_order(keys, 1) != order


# --- the runner's choice of entry ----------------------------------------------------------------


def run_config(tmp_path, **changes):
    values = {"data_dir": str(tmp_path / "data"), "model_dir": str(tmp_path / "model"),
              "outputs_dir": str(tmp_path / "inference_outputs"), "lora_r": 4,
              "lora_targets": TARGETS}  # fmt: skip
    values.update(changes)
    return RunConfig(**values)


def test_adapters_start_the_arms_entry_and_ride_in_the_json(tmp_path):
    plain = run_config(tmp_path)
    command = solver_command(plain, 1.0, "python")
    assert command[2] == "arc26.solver" and "adapters" not in json.loads(command[-1])
    arms = settings(tmp_path)
    cfg = run_config(tmp_path, adapters=arms.to_json())
    assert cfg.adapters == arms
    command = solver_command(cfg, 1.0, "python")
    assert command[2] == ARMS_MODULE == "arc26.arms"
    sent = json.loads(command[-1])
    assert sent["adapters"] == arms.to_json()
    assert {k: v for k, v in sent.items() if k != "adapters"} == json.loads(
        solver_command(plain, 1.0, "python")[-1]
    )


def test_adapters_need_the_split_solver_no_holdout_and_their_own_folders(tmp_path):
    arms = settings(tmp_path).to_json()
    with pytest.raises(ValueError, match="split"):
        run_config(tmp_path, adapters=arms, solver="nvarc")
    with pytest.raises(ValueError, match="holdout"):
        run_config(tmp_path, adapters=arms, holdout={"gate_dir": str(tmp_path / "gate")})
    inside = {**arms, "outputs_root": str(tmp_path / "inference_outputs" / "arms")}
    with pytest.raises(ValueError, match="outside outputs_dir"):
        run_config(tmp_path, adapters=inside)


# --- decode_puzzle's views (the protected keyword), through the real batching ---------------------


def decode_batches(stack, monkeypatch, tmp_path, views, capsys):
    """Run the real decode_puzzle on one puzzle with a recording DFS; return each batch's
    subkeys and prompt lengths."""
    decode = stack["arc26.decode"]
    from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter

    lengths = []

    def fake_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
        lengths.append([len(t) for t in prefix_tokens])
        return []

    monkeypatch.setattr(decode, "inference_turbo_dfs", fake_dfs)
    tokenizer = tr.standin_tokenizer()
    formatter = QwenFormatter(tokenizer)
    puzzle = ArcDataset(queries={"aaaa0001": PUZZLES["aaaa0001"]})
    decode.decode_puzzle(
        puzzle_ds=puzzle.change_keys(["aaaa0001"]),
        formatter=formatter,
        max_seq_length=8192,
        max_new_tokens=formatter.max_new_tokens(),
        start_time=time.time(),
        end_time=1e18,
        rank=0,
        key="aaaa0001",
        tokenizer=tokenizer,
        model=None,
        max_score=1.6,
        timing_logger=None,
        dir_outputs=str(tmp_path),
        views=views,
    )
    printed = capsys.readouterr().out
    batches = [json.loads(m.replace("'", '"')) for m in re.findall(r"decoding (\[.*\])", printed)]
    return batches, lengths


def test_default_views_are_the_reference_batches(stack, monkeypatch, tmp_path, capsys):
    batches, lengths = decode_batches(stack, monkeypatch, tmp_path, None, capsys)
    # Two test inputs: the first two batches of each, then the transposed two of each.
    assert len(batches) == 8 and all(len(b) == 4 for b in batches)
    order = [b[0].split(".")[0] for b in batches]
    assert order == ["aaaa0001_0", "aaaa0001_0", "aaaa0001_1", "aaaa0001_1"] * 2
    assert all(len(set(lens)) == 1 for lens in lengths)


def test_the_dihedral_once_views_decode_each_map_once_in_equal_length_batches(
    stack, monkeypatch, tmp_path, capsys
):
    batches, lengths = decode_batches(stack, monkeypatch, tmp_path, am.DIHEDRAL_ONCE, capsys)
    # Per test input, two batches of four, input by input.
    assert [b[0].split(".")[0] for b in batches] == ["aaaa0001_0"] * 2 + ["aaaa0001_1"] * 2
    assert all(len(b) == 4 for b in batches)
    assert all(len(set(lens)) == 1 for lens in lengths)  # one prompt length per batch

    def mapping(subkey):
        ops = [op for op in subkey.split(".")[1:] if op in ("transpose", "rot90")]
        return ".".join(ops) or "identity"

    for test in ("aaaa0001_0", "aaaa0001_1"):
        mine = [b for b in batches if b[0].startswith(test)]
        names = [mapping(k) for b in mine for k in b]
        assert sorted(names) == sorted(
            ["identity", "rot90", "rot90.rot90", "rot90.rot90.rot90", "transpose",
             "transpose.rot90", "transpose.rot90.rot90", "transpose.rot90.rot90.rot90"]
        )  # fmt: skip
        assert [mapping(k) for k in mine[0]] == [
            "identity", "rot90.rot90", "transpose.rot90", "transpose.rot90.rot90.rot90"
        ]  # fmt: skip


# --- the arm runner on the tiny model ------------------------------------------------------------


class Clock:
    """A clock the runner reads (time.time, time.perf_counter); the fake decode advances it."""

    def __init__(self, start=1000.0):
        self.now = start

    def time(self):
        return self.now

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def runner_env(stack, monkeypatch, tmp_path):
    data = write_data(tmp_path / "data")
    loaded = report.cpu_loaded(data, 0, H_CONFIG, tmp_path / "timing")
    hyper = tiny_hyper(loaded.model, tmp_path / "hcache")
    decode = stack["arc26.decode"]
    calls = []
    clock = Clock()
    plan = {}

    def fake_decode(**kwargs):
        calls.append(
            {
                "arm": Path(kwargs["dir_outputs"]).name,
                "key": kwargs["key"],
                "views": kwargs["views"],
                "start_time": kwargs["start_time"],
                "end_time": kwargs["end_time"],
                "digest": digest(kwargs["model"]),
            }
        )
        action = plan.get((kwargs["key"], Path(kwargs["dir_outputs"]).name))
        if action == "raise":
            raise MemoryError("CUDA out of memory (stand-in decode)")
        clock.sleep(action if isinstance(action, int | float) else 10.0)

    monkeypatch.setattr(decode, "decode_puzzle", fake_decode)
    trainings = []
    original = report.StandinTrainer.train

    def recording_train(self):
        trainings.append(
            {"before": digest(self.model), "steps": self.args.get("max_steps", -1),
             "views": len(self.train_dataset["input_ids"])}  # fmt: skip
        )
        return original(self)

    monkeypatch.setattr(report.StandinTrainer, "train", recording_train)
    monkeypatch.setattr(am, "time", clock)
    return {
        "data": data, "loaded": loaded, "hyper": hyper, "calls": calls, "clock": clock,
        "plan": plan, "trainings": trainings, "tmp": tmp_path,
    }  # fmt: skip


def make_runner(env, **changes):
    arms = settings(env["tmp"], **changes)
    return am.ArmRunner(
        arms, env["loaded"], 0, hyper=env["hyper"], solutions=SOLUTIONS
    ), arms


def records(arms):
    path = Path(arms.log_dir) / "records-rank0.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_arm_decodes_with_its_own_adapter(runner_env):
    env = runner_env
    runner, arms = make_runner(env)
    model = runner.solve(env["loaded"].model, "aaaa0001", end_time=1e9)
    rows = records(arms)
    assert [r["arm"] for r in rows] == [a["name"] for a in ARMS]
    assert all(r["status"] == "done" for r in rows)
    by = {r["arm"]: r for r in rows}
    calls = {c["arm"]: c for c in env["calls"]}
    default = adapter_digest(env["loaded"].default_weights)
    task = PUZZLES["aaaa0001"]
    demos = DemoSet(
        "aaaa0001", tuple(Pair(p["input"], p["output"]) for p in task["train"]), None,
        tuple(t["input"] for t in task["test"]),
    )  # fmt: skip
    emitted = env["hyper"].emit(demos).digest()
    # (a) the initial weights (lora_B zero); (b) H's adapter exactly (fp32 here, no cast).
    assert by["base"]["fingerprint"] == calls["base"]["digest"] == default
    assert by["h"]["fingerprint"] == calls["h"]["digest"] == emitted
    # (c) one epoch from the initial weights; (d) the polish from H's adapter; (e) from zero.
    sgd, polish_h, polish_0 = env["trainings"]
    assert sgd == {"before": default, "steps": -1, "views": sgd["views"]} and sgd["views"] > 3
    assert polish_h["before"] == emitted and polish_h["steps"] == 3
    assert polish_0["before"] == default and polish_0["steps"] == 2
    assert len({by[name]["fingerprint"] for name in ("sgd", "h-polish", "polish")} | {default}) == 4
    # Every arm decodes with the run's views into its own folder, on its own clock.
    assert all(c["views"] == am.DIHEDRAL_ONCE for c in env["calls"])
    assert all(c["end_time"] == 1e9 + arms.finish_s for c in env["calls"])
    assert all(c["key"] == "aaaa0001" for c in env["calls"])
    starts = [c["start_time"] for c in env["calls"]]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    # The CE of each answer under each arm: the base's differs from H's.
    assert all(len(r["ce"]) == 2 and all(i["ce"] > 0 for i in r["ce"]) for r in rows)
    assert by["base"]["ce"] != by["h"]["ce"]
    # The live model is returned for the next puzzle, in inference mode.
    assert not model.training


def test_each_arm_writes_its_own_timing_records(runner_env):
    env = runner_env
    runner, arms = make_runner(env)
    runner.solve(env["loaded"].model, "aaaa0002", end_time=1e9)
    ops = {}
    for arm in ARMS:
        folder = Path(arms.log_dir) / arm["name"] / "timing"
        lines = (folder / "timing-rank0.jsonl").read_text(encoding="utf-8").splitlines()
        ops[arm["name"]] = {json.loads(line)["op"] for line in lines}
    common = {"fingerprint", "ce", "puzzle"}
    assert ops["base"] >= common | {"reset"}
    assert ops["h"] >= common | {"encode", "h_forward", "snapshot", "swap"}
    assert ops["sgd"] >= common | {"train"}
    assert ops["h-polish"] >= common | {"h_forward", "candidate_train"}
    assert "encode" not in ops["h-polish"]  # the H arm cached the puzzle's pairs
    assert ops["polish"] >= common | {"reset", "candidate_train"}


def test_a_cut_puzzle_is_incomplete_in_every_arm_and_left_out_of_the_pairs(runner_env):
    env = runner_env
    runner, arms = make_runner(env, finish_s=0)
    start = env["clock"].now
    # The first puzzle runs in full; in the second, the sgd arm's decode passes the deadline.
    runner.solve(env["loaded"].model, "aaaa0001", end_time=start + 1e6)
    env["plan"][("aaaa0002", "sgd")] = 1e7
    runner.solve(env["loaded"].model, "aaaa0002", end_time=env["clock"].now + 30)
    rows = records(arms)
    second = [r["status"] for r in rows if r["puzzle_id"] == "aaaa0002"]
    assert second == ["done", "done", "cut", "not_started", "not_started"]
    run = {
        "arms": {a["name"]: a["source"] for a in ARMS},
        "order": ["aaaa0001", "aaaa0002"],
        "records": {},
        "scores": {a["name"]: {"aaaa0001": {}, "aaaa0002": {}} for a in ARMS},
    }
    for r in rows:
        run["records"].setdefault(r["puzzle_id"], {})[r["arm"]] = r
    assert report.paired(run) == ["aaaa0001"]


def test_an_arm_that_raises_is_recorded_and_the_next_arms_run(runner_env):
    env = runner_env
    runner, arms = make_runner(env)
    env["plan"][("aaaa0002", "h")] = "raise"
    runner.solve(env["loaded"].model, "aaaa0002", end_time=1e9)
    rows = records(arms)
    assert [r["status"] for r in rows] == ["done", "raised", "done", "done", "done"]
    assert "out of memory" in rows[1]["error"]


def test_the_worker_loop_stops_at_the_soft_deadline(runner_env, monkeypatch):
    import queue as queues

    env = runner_env
    arms = settings(env["tmp"], arms=ARMS[:1], checkpoint=None, hypernet=None)
    cfg = {"adapters": arms.to_json(), "data_dir": str(env["data"])}
    work = queues.Queue()
    for key in ("aaaa0001", "aaaa0002"):
        work.put(key)
    work.put(None)
    env["plan"][("aaaa0001", "base")] = 100.0
    am.run_worker(0, work, env["clock"].now + 50, cfg, env["loaded"])
    assert [r["puzzle_id"] for r in records(arms)] == ["aaaa0001"]
    assert work.get() == "aaaa0002"  # never taken


def test_seed_cache_copies_what_the_source_holds_bit_for_bit(tmp_path):
    def cache(root):
        return EmbeddingCache(root, model_id="m", layer=1, pool="output", view_set_id="v")

    source, target = cache(tmp_path / "s"), cache(tmp_path / "t")
    pairs = [{"input": [[i]], "output": [[i + 1]]} for i in range(3)]
    hashes = [pair_hash(p) for p in pairs]
    values = torch.randn(3, 5)
    source.put({hashes[0]: values[0], hashes[1]: values[1]})
    assert am.seed_cache(target, source, hashes + [hashes[0]]) == 2
    assert torch.equal(target.get(hashes[:2]), source.get(hashes[:2]))
    assert target.missing(hashes) == [hashes[2]]
    assert am.seed_cache(target, source, hashes) == 0


def test_answer_ce_is_calc_scores_per_reply_token(stack):
    from arc26.decode import calc_scores
    from arc26.nvarc.arc_loader import QwenFormatter

    tokenizer = tr.standin_tokenizer()
    formatter = QwenFormatter(tokenizer)
    base = tr.tiny_base(0)
    out = am.answer_ce(
        PUZZLES["aaaa0002"], SOLUTIONS["aaaa0002"], "aaaa0002", formatter=formatter,
        tokenizer=tokenizer, model=base, max_seq_length=8192,
        max_new_tokens=formatter.max_new_tokens(),
    )  # fmt: skip
    query = formatter.fmt_train(PUZZLES["aaaa0002"]["train"]) + formatter.fmt_query(
        PUZZLES["aaaa0002"]["test"]
    )
    reply = formatter.fmt_reply(SOLUTIONS["aaaa0002"])
    nll = calc_scores([query], [reply], tokenizer, base)[0]
    tokens = len(tokenizer.encode(reply))
    assert out == [{"nll": nll, "tokens": tokens, "ce": nll / tokens}]


# --- scores and the parent -----------------------------------------------------------------------


def test_score_arm_on_a_hand_case():
    predictions = {
        "aaaa0001": [
            {"attempt_1": [[0]], "attempt_2": [[5, 4], [7, 6]]},  # solved by attempt 2
            {"attempt_1": [[4, 3, 2]], "attempt_2": [[0]]},  # solved by attempt 1
        ],
        "aaaa0002": [{"attempt_1": "bad", "attempt_2": [[1]]}],  # malformed, wrong
    }
    scores = am.score_arm(predictions, SOLUTIONS, ["aaaa0001", "aaaa0002"])
    assert scores["puzzles"]["aaaa0001"] == {
        "correct": [True, True], "attempt_1": [False, True], "score": 1.0
    }  # fmt: skip
    assert scores["puzzles"]["aaaa0002"] == {"correct": [False], "attempt_1": [False], "score": 0}
    assert (scores["tasks"], scores["score"], scores["outputs"]) == (2, 1.0, 3)
    assert (scores["solved_outputs"], scores["attempt_1_outputs"]) == (2, 1)
    # A missing task scores 0 in both.
    assert am.score_arm({}, SOLUTIONS, ["aaaa0002"])["puzzles"]["aaaa0002"]["attempt_1"] == [False]


def test_select_and_score_selects_each_arm_with_the_reference_selection(stack, tmp_path):
    data = write_data(tmp_path / "data")
    arms = settings(tmp_path)
    # One candidate for aaaa0002's test input in the h arm's folder, as decode_puzzle writes it.
    folder = Path(arms.outputs_root) / "h"
    folder.mkdir(parents=True)
    import numpy as np

    solution = np.array(SOLUTIONS["aaaa0002"][0])
    row = {"beam_score": 0.1, "score_aug": [0.1] * 8, "solution": solution}
    with bz2.BZ2File(folder / "aaaa0002_0.permute0123456789.ex01", "w") as handle:
        pickle.dump([row], handle)
    results = am.select_and_score(arms, data, ["aaaa0001", "aaaa0002"])
    assert set(results) == {a["name"] for a in ARMS}
    h = json.loads((Path(arms.log_dir) / "h" / "scores.json").read_text(encoding="utf-8"))
    assert h["puzzles"]["aaaa0002"] == {"correct": [True], "attempt_1": [True], "score": 1.0}
    base = json.loads((Path(arms.log_dir) / "base" / "scores.json").read_text(encoding="utf-8"))
    assert base["score"] == 0
    assert (Path(arms.log_dir) / "h" / am.SELECTION_LOG).is_file()


def test_run_stops_before_any_decode_when_the_load_check_fails(tmp_path, monkeypatch):
    data = write_data(tmp_path / "data")
    lc = {"run_logs": str(tmp_path), "max_abs": 1e-3, "max_mean": 1e-4}
    arms = settings(
        tmp_path, load_check=lc, episodes_root="e", check_root="c", manifest_sha256="0" * 64
    )
    cfg = {"adapters": arms.to_json(), "data_dir": str(data), "subset": None, "workers": 1}
    launched = []
    code = am.run(cfg, 1e18, launch=lambda *a: launched.append(a), check=lambda *a: 6)
    assert code == am.EXIT_LOAD_CHECK and not launched
    summary = json.loads((Path(arms.log_dir) / am.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["load_check_exit"] == 6 and "order" not in summary
    # A pass launches the workers on the subset in the shuffled order, then selects every arm.
    cfg["subset"] = ["aaaa0002", "aaaa0001"]
    code = am.run(cfg, 1e18, launch=lambda *a: launched.append(a), check=lambda *a: 0)
    assert code == 0 and launched[0][0] == am.queue_order(cfg["subset"], 0)
    assert launched[0][2]["data_dir"] == str(data)
    summary = json.loads((Path(arms.log_dir) / am.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert set(summary["selection"]) == {a["name"] for a in ARMS}
    # A competition rerun never runs the arms.
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    assert am.run(cfg, 1e18, launch=lambda *a: launched.append(a)) == am.EXIT_CONFIG


# --- fix round 1: the CE's mode, the probe, the first-puzzle guard, exit 7 ----------------------


def test_the_answer_ce_runs_under_inference_mode_without_grad(stack, monkeypatch):
    decode = stack["arc26.decode"]
    from arc26.nvarc.arc_loader import QwenFormatter

    modes = []
    real = decode.calc_scores

    def recording(queries, answers, tokenizer, model):
        modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        return real(queries, answers, tokenizer, model)

    monkeypatch.setattr(decode, "calc_scores", recording)
    tokenizer = tr.standin_tokenizer()
    formatter = QwenFormatter(tokenizer)
    out = am.answer_ce(
        PUZZLES["aaaa0001"], SOLUTIONS["aaaa0001"], "aaaa0001", formatter=formatter,
        tokenizer=tokenizer, model=tr.tiny_base(0).requires_grad_(True), max_seq_length=8192,
        max_new_tokens=formatter.max_new_tokens(),
    )  # fmt: skip
    assert modes == [(True, False), (True, False)] and len(out) == 2
    assert torch.is_grad_enabled()  # the mode ends with the call


def dev4_file(tmp):
    """Two DEV4 ids: 36a08778 has the fewer demonstration cells, so the probe takes it."""
    data = {
        "0934a4d8": PUZZLES["aaaa0001"],
        "36a08778": PUZZLES["aaaa0002"],
        "00000000": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]},
    }
    path = tmp / "dev4.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def with_load_check(tmp):
    lc = {"run_logs": str(tmp), "max_abs": 1e-3, "max_mean": 1e-4,
          "dev4_challenges": str(dev4_file(tmp))}  # fmt: skip
    return {"load_check": lc, "episodes_root": "e", "check_root": "c", "manifest_sha256": "0" * 64}


def test_the_probe_runs_h_on_the_smallest_dev4_puzzle_and_rolls_back(runner_env):
    env = runner_env
    runner, arms = make_runner(env, **with_load_check(env["tmp"]))
    default = adapter_digest(env["loaded"].default_weights)
    model = runner.probe(env["loaded"].model)
    (row,) = records(arms)
    assert (row["arm"], row["status"], row["puzzle_id"]) == ("probe", "done", "36a08778")
    task = PUZZLES["aaaa0002"]  # the probe's task, under its DEV4 id
    demos = DemoSet(
        "36a08778", tuple(Pair(p["input"], p["output"]) for p in task["train"]), None,
        tuple(t["input"] for t in task["test"]),
    )  # fmt: skip
    assert row["fingerprint"] == env["hyper"].emit(demos).digest() != default
    assert digest(model) == default  # the live adapter is back, bit for bit
    assert row["cache_misses"] == 2  # both pairs encoded live, as DEV40's H arms will


def test_a_probe_that_raises_stops_the_worker_before_any_puzzle(runner_env, monkeypatch):
    import queue as queues

    env = runner_env
    runner, arms = make_runner(env)

    def broken(*args, **kwargs):
        raise RuntimeError("unsloth's disable_adapter (stand-in failure)")

    monkeypatch.setattr(env["hyper"], "adapt", broken)
    work = queues.Queue()
    work.put("aaaa0001")
    cfg = {"adapters": arms.to_json(), "data_dir": str(env["data"])}
    with pytest.raises(am.ArmStop, match="probe"):
        am.run_worker(0, work, 1e9, cfg, env["loaded"], runner)
    marker = json.loads((Path(arms.log_dir) / "stop-rank0.json").read_text(encoding="utf-8"))
    assert marker["record"]["arm"] == "probe" and "stand-in failure" in marker["record"]["error"]
    assert [r["arm"] for r in records(arms)] == ["probe"] and work.get() == "aaaa0001"


def test_an_h_arm_raising_on_the_first_puzzle_stops_the_worker_later_ones_do_not(runner_env):
    import queue as queues

    env = runner_env
    cfg_of = lambda arms: {"adapters": arms.to_json(), "data_dir": str(env["data"])}  # noqa: E731

    def queue_of(*keys):
        work = queues.Queue()
        for key in keys:
            work.put(key)
        work.put(None)
        return work

    runner, arms = make_runner(env)
    env["plan"][("aaaa0001", "h-polish")] = "raise"
    with pytest.raises(am.ArmStop, match="first puzzle"):
        am.run_worker(0, queue_of("aaaa0001", "aaaa0002"), 1e9, cfg_of(arms), env["loaded"], runner)
    assert {r["puzzle_id"] for r in records(arms) if r["arm"] != "probe"} == {"aaaa0001"}
    # On a later puzzle the arm is recorded and the worker goes on.
    env["plan"].clear()
    env["plan"][("aaaa0002", "h")] = "raise"
    tmp = env["tmp"] / "second"
    runner, arms = make_runner(env, log_dir=str(tmp / "logs"), outputs_root=str(tmp / "out"))
    am.run_worker(0, queue_of("aaaa0001", "aaaa0002"), 1e9, cfg_of(arms), env["loaded"], runner)
    statuses = {(r["puzzle_id"], r["arm"]): r["status"] for r in records(arms)}
    assert statuses[("aaaa0002", "h")] == "raised" and statuses[("aaaa0002", "sgd")] == "done"
    assert not (tmp / "logs" / "stop-rank0.json").exists()


def test_run_exits_7_without_a_selection_when_a_worker_stops(tmp_path):
    data = write_data(tmp_path / "data")
    arms = settings(tmp_path)
    cfg = {"adapters": arms.to_json(), "data_dir": str(data), "subset": None, "workers": 1}

    def launch(order, end_time, worker_cfg):
        Path(arms.log_dir, "stop-rank2.json").write_text(
            json.dumps({"rank": 2, "reason": "H's probe on 36a08778 raised", "record": {}})
        )
        raise RuntimeError("process 2 terminated with exit code 1")

    assert am.run(cfg, 1e18, launch=launch) == am.EXIT_H_ARM == 7
    summary = json.loads((Path(arms.log_dir) / am.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["exit"] == 7 and summary["stopped"][0]["rank"] == 2
    assert "selection" not in summary and "order" in summary
    # Any other worker failure is exit 1, and the arms are still selected.
    assert am.run(cfg, 1e18, launch=lambda *a: 1 / 0) == am.EXIT_FAILED
    summary = json.loads((Path(arms.log_dir) / am.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert "stopped" not in summary and set(summary["selection"]) == {a["name"] for a in ARMS}
