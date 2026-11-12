"""P3.7 amendment 8: arm (f), ``hyper_fit``, and the per-step training log (``train_log``).

Arm (f) runs arm (c)'s ``arc26.ttt.fit`` with H's adapter as its ``default_weights``; every other
argument must be (c)'s, so (f) minus (c) isolates the starting adapter. Checked on the CPU with
``tests/test_arms.py``'s tiny stack (a random Qwen3 wrapped by PEFT, a tiny H, the smoke path's
stand-ins for unsloth and datasets, the decode recorded instead of run): the fit's arguments,
the start it trains from, the same views and steps, the trainer's arguments (``train_log`` adds
only per-step logging), the per-step records, the capture's cleanup, and the arm's place among
H's arms (the first-puzzle stop). Skipped without torch, transformers, peft, or safetensors.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("_t_arms_for_fit", REPO / "tests" / "test_arms.py")
T = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_t_arms_for_fit", T)
_spec.loader.exec_module(T)

am, report, digest, records = T.am, T.report, T.digest, T.records
stack, runner_env = T.stack, T.runner_env  # the fixtures, shared
FIT_ARMS = [{"name": "sgd", "source": "sgd"}, {"name": "h-fit", "source": "hyper_fit"}]


def emitted(env, key):
    task = T.PUZZLES[key]
    demos = T.DemoSet(
        key, tuple(T.Pair(p["input"], p["output"]) for p in task["train"]), None,
        tuple(t["input"] for t in task["test"]),
    )  # fmt: skip
    return env["hyper"].emit(demos).digest()


def test_hyper_fit_is_an_h_source_that_trains_without_steps():
    assert am.HYPER_FIT in am.SOURCES and am.HYPER_FIT in am.H_SOURCES
    assert am.HYPER_FIT in am.TRAINED_SOURCES and am.HYPER_FIT not in am.POLISH_SOURCES
    assert am.ArmSpec("f", am.HYPER_FIT).steps is None
    with pytest.raises(ValueError, match="steps apply to a polish only"):
        am.ArmSpec("f", am.HYPER_FIT, steps=128)
    # Only training arms log, and logging changes nothing the optimizer reads.
    assert am.TRAINED_SOURCES == {
        am.SGD, am.HYPER_POLISH, am.POLISH, am.HYPER_FIT, am.HYPER_MERGE_FIT
    }  # fmt: skip
    assert am.TRAIN_LOG_ARGS == {"logging_strategy": "steps", "logging_steps": 1}


def test_hyper_fit_calls_fit_with_sgds_arguments_but_the_start(runner_env, monkeypatch):
    env = runner_env
    ttt = sys.modules["arc26.ttt"]
    original, calls = ttt.fit, []

    def recording_fit(**kwargs):
        calls.append({**kwargs, "start": T.adapter_digest(dict(kwargs["default_weights"]))})
        return original(**kwargs)

    monkeypatch.setattr(ttt, "fit", recording_fit)
    runner, arms = T.make_runner(env, arms=FIT_ARMS)
    runner.solve(env["loaded"].model, "aaaa0001", end_time=1e9)
    sgd, fit = calls
    # The start: the initial weights (lora_B zero) for (c), H's adapter for (f), bit for bit.
    assert sgd["start"] == T.adapter_digest(env["loaded"].default_weights)
    assert fit["start"] == emitted(env, "aaaa0001") != sgd["start"]
    # Everything else is (c)'s: the same objects (data, formatter, tokenizer, collator) and the
    # same values (train_args: epochs, batch, learning rate, schedule, warmup, seed).
    for name in ("arc_test_set", "formatter", "tokenizer", "collator"):
        assert fit[name] is sgd[name], name
    for name in ("key", "max_seq_length", "rank", "train_args"):
        assert fit[name] == sgd[name], name
    assert fit["train_args"] == env["loaded"].train_args and "max_steps" not in fit["train_args"]
    assert type(fit["timing_logger"]) is type(sgd["timing_logger"])
    # The trainer starts (f) from H's adapter and (c) from zero, over the same views and steps.
    c, f = env["trainings"]
    assert c["before"] == sgd["start"] and f["before"] == fit["start"]
    assert c["views"] == f["views"] > 3 and c["steps"] == f["steps"] == -1
    rows = {r["arm"]: r for r in records(arms)}
    assert all(r["status"] == "done" for r in rows.values())
    trained = rows["h-fit"]["fingerprint"]
    assert trained not in (sgd["start"], fit["start"], rows["sgd"]["fingerprint"])
    assert "train_log" not in rows["sgd"] and "train_log" not in rows["h-fit"]


def test_hyper_fit_records_the_same_train_op_as_sgd_and_hs_ops(runner_env):
    env = runner_env
    runner, arms = T.make_runner(env, arms=FIT_ARMS)
    runner.solve(env["loaded"].model, "aaaa0002", end_time=1e9)
    ops = {}
    for arm in FIT_ARMS:
        lines = (Path(arms.log_dir) / arm["name"] / "timing" / "timing-rank0.jsonl").read_text()
        ops[arm["name"]] = {T.json.loads(line)["op"] for line in lines.splitlines()}
    assert {"train", "fingerprint", "ce", "puzzle"} <= ops["sgd"]
    assert {"train", "encode", "h_forward", "swap", "snapshot", "ce", "puzzle"} <= ops["h-fit"]
    assert "candidate_train" not in ops["h-fit"]


def test_train_log_records_every_step_of_both_arms(runner_env, monkeypatch):
    env = runner_env
    seen = []
    train = report.StandinTrainer.train

    def args_train(self):
        seen.append(dict(self.args))
        return train(self)

    monkeypatch.setattr(report.StandinTrainer, "train", args_train)
    runner, arms = T.make_runner(
        env, arms=[*FIT_ARMS, {"name": "base", "source": "base"}], train_log=True
    )
    runner.solve(env["loaded"].model, "aaaa0001", end_time=1e9)
    # The trainer's arguments are (c)'s plus the logging, for both arms.
    expected = {**env["loaded"].train_args, **am.TRAIN_LOG_ARGS}
    assert seen == [expected, expected]
    rows = {r["arm"]: r for r in records(arms)}
    views = env["trainings"][0]["views"]
    for name in ("sgd", "h-fit"):
        log = rows[name]["train_log"]
        assert [entry["step"] for entry in log] == list(range(1, views + 1)), name
        assert all(entry["loss"] > 0 and entry["learning_rate"] is not None for entry in log)
    assert rows["sgd"]["train_log"] != rows["h-fit"]["train_log"]
    assert "train_log" not in rows["base"]
    # The capture is undone: the trainer class has its own log method no more.
    assert "log" not in vars(sys.modules["arc26.ttt"].UnslothFixedTrainer)


def test_the_capture_restores_the_trainer_after_an_error(stack):
    trainer = stack["arc26.ttt"].UnslothFixedTrainer
    sink = []
    with pytest.raises(RuntimeError), am.train_log_capture(sink):
        assert "log" in vars(trainer)
        raise RuntimeError("a fit that raised")
    assert "log" not in vars(trainer) and sink == []


def test_an_h_fit_arm_raising_on_the_first_puzzle_stops_the_worker(runner_env, monkeypatch):
    env = runner_env

    def broken(*args, **kwargs):
        raise MemoryError("CUDA out of memory (stand-in H)")

    monkeypatch.setattr(env["hyper"], "adapt", broken)
    runner, arms = T.make_runner(env, arms=FIT_ARMS)
    model = runner.solve(env["loaded"].model, "aaaa0001", end_time=1e9)
    rows = {r["arm"]: r for r in records(arms)}
    assert rows["sgd"]["status"] == "done" and rows["h-fit"]["status"] == "raised"
    failed = [e for e in runner.last if e["source"] in am.H_SOURCES and e["status"] == "raised"]
    assert [e["arm"] for e in failed] == ["h-fit"]
    assert model is not None
