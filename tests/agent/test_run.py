"""Tests of the agent's run entry (P6.4): config, modes, engine failures, kill and resume.

The end-to-end tests run the command line with P6.1's real REPL and verifier and the fake engine
of :mod:`arc26.agent.server` (scripted replies per task over HTTP), started and stopped by the
run itself as it would start vLLM. One of them kills the run with SIGKILL mid-way and resumes it.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from arc26.agent import run as agent_run
from arc26.agent.loop import read_records
from arc26.agent.run import (
    ConfigError,
    config_from_mapping,
    load_config,
    main,
    score_dev,
)
from arc26.agent.server import free_port
from arc26.submit import validate

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "agent_default.yaml"

RULES = {
    "fliplr01": ("np.fliplr(g)", lambda g: np.fliplr(g)),
    "flipud02": ("np.flipud(g)", lambda g: np.flipud(g)),
    "transp03": ("g.T", lambda g: g.T),
    "rot90004": ("np.rot90(g)", lambda g: np.rot90(g)),
    "recolr05": ("np.where(g == 1, 7, g)", lambda g: np.where(g == 1, 7, g)),
    "tiling06": ("np.tile(g, (2, 2))", lambda g: np.tile(g, (2, 2))),
}


def program(expr):
    return (
        "```python\nimport numpy as np\n\ndef transform(grid):\n"
        f"    g = np.asarray(grid)\n    return {expr}\n```\n"
    )


def make_case(tmp_path, delay_scripts=False):
    """Six synthetic tasks, their solutions, the fake engine's replies, and NVARC's file."""
    rng = np.random.default_rng(7)
    challenges, solutions = {}, {}
    for name, (_, rule) in RULES.items():

        def grid():
            g = rng.integers(0, 4, size=tuple(rng.integers(3, 6, size=2)))
            g[0, 0], g[-1, -1] = 1, 2  # no accidental symmetry in the corners
            return g

        pairs = [grid() for _ in range(3)]
        test = grid()
        challenges[name] = {
            "train": [{"input": g.tolist(), "output": rule(g).tolist()} for g in pairs],
            "test": [{"input": test.tolist()}],
        }
        solutions[name] = [rule(test).tolist()]
    replies = {
        "fliplr01": [program(RULES["fliplr01"][0])],  # right at once
        "flipud02": [program("np.fliplr(g)"), program(RULES["flipud02"][0])],  # fixed later
        "transp03": [program(RULES["transp03"][0])],
        "rot90004": ["I am not sure yet.", program(RULES["rot90004"][0])],  # no code, then right
        "recolr05": [],  # never any code: only NVARC has it
        "tiling06": [program("np.tile(g, (1, 2))")] * 4,  # always wrong: unsolved
    }
    nvarc = {
        "recolr05": [{"attempt_1": solutions["recolr05"][0], "attempt_2": [[0]]}],
        "tiling06": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
    }
    paths = {}
    for name, value in (("challenges", challenges), ("solutions", solutions),
                        ("replies", replies), ("nvarc", nvarc)):  # fmt: skip
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    return challenges, solutions, paths


def write_config(tmp_path, **sections):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["concurrency"] = 2
    config["sandbox"]["verifier_workers"] = 2
    config["deadline"].update(startup_margin_s=0, final_margin_s=2, grace_s=5)
    config["governor"]["min_calls"] = 1
    config["server"].update(boot_timeout_s=30, stop_timeout_s=5, health_interval_s=0.05,
                            monitor_interval_s=0.1, hang_s=10)  # fmt: skip
    for number, phase in enumerate(config["phases"]):
        phase.update(max_turns=2, min_attempt_tokens=2000, max_attempt_tokens=20_000)
        phase.update(turn_tokens=8000, attempts=1)
        phase["reasoning_effort"] = "low" if number == 0 else "xhigh"
    for name, values in sections.items():
        if isinstance(values, dict):
            config[name].update(values)
        else:
            config[name] = values
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def challenges_stub(challenges):
    return {task_id: {"test": task["test"]} for task_id, task in challenges.items()}


def arguments(paths, out, config, logs, *extra, mode="dev", port=None, deadline=True,
              nvarc=None):  # fmt: skip
    """The command line; ``deadline`` adds the session's end 300 s from now (absolute)."""
    end = ["--deadline", str(time.time() + 300)] if deadline else []
    return [
        "--mode", mode, "--tasks", str(paths["challenges"]), "--out", str(out),
        "--config", str(config), "--log-dir", str(logs), "--port", str(port or free_port()),
        *end, "--nvarc-submission", str(nvarc or paths["nvarc"]), *extra,
    ]  # fmt: skip


def score(written, solutions):
    return sum(
        any(written[t][0][a] == want[0] for a in ("attempt_1", "attempt_2"))
        for t, want in solutions.items()
    )


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def stop_engine(logs):
    """Kill an engine a test run may have left (its pid file names it)."""
    path = logs / "engine.pid"
    if path.exists():
        pid = json.loads(path.read_text())["pid"]
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------- config


def test_the_default_config_loads():
    config = load_config(DEFAULT_CONFIG)
    assert config.concurrency == 32 and config.scheduler.context_tokens == 65536
    phases = config.scheduler.phases
    assert [p.reasoning_effort for p in phases] == ["low", "xhigh", "xhigh", "xhigh"]
    assert phases[3].attempts is None  # the cycle runs until the generation stop
    assert config.scheduler.governor.window_s >= 3600
    assert config.scheduler.prior_programs == 2 and config.scheduler.prior_chars == 6000
    assert config.server.restarts == 1 and config.server.port == 8000
    assert config.session_s - config.startup_margin_s - config.final_margin_s == 11 * 3600


def test_the_server_flags_equal_the_spike_stage_1_settings():
    server = yaml.safe_load(DEFAULT_CONFIG.read_text())["server"]
    expected = {
        "tensor_parallel_size": 4, "kv_cache_dtype": "fp8", "enable_prefix_caching": True,
        "max_model_len": 65536, "max_num_seqs": 64, "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.9, "attention_backend": "TRITON_ATTN",
        "language_model_only": True, "reasoning_parser": "qwen3", "generation_config": "auto",
        "extra_args": [],
    }  # fmt: skip
    assert {k: server[k] for k in expected} == expected


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(extra=1),
        lambda c: c.pop("governor"),
        lambda c: c["server"].update(prot=1),
        lambda c: c["phases"].pop(),
        lambda c: c["phases"][0].update(attempts=None),
        lambda c: c["scheduler"].update(prior_chars=100),
        lambda c: c["scheduler"].pop("prior_programs"),
        lambda c: c["scheduler"].update(cycle_error_rounds=0),
        lambda c: c["phases"][0].update(share=2.0),
        lambda c: c["deadline"].update(final_margin_s=50_000),
        lambda c: c["engine"].update(effort_transport="carrier pigeon"),
        lambda c: c["sandbox"].update(verifier_workers=0),
    ],
)
def test_the_config_is_checked_strictly(change):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    change(config)
    with pytest.raises(ConfigError):
        config_from_mapping(config)


# ---------------------------------------------------------------- refusals


def test_competition_mode_refuses_a_solutions_file(tmp_path, monkeypatch):
    challenges, solutions, paths = make_case(tmp_path)
    opened = []
    monkeypatch.setattr("arc26.evaluate.load_solutions", lambda path: opened.append(path))
    out = tmp_path / "submission.json"
    code = main(
        arguments(paths, out, write_config(tmp_path), tmp_path / "logs", "--fake-engine",
                  str(paths["replies"]), "--solutions", str(paths["solutions"]),
                  mode="competition")  # fmt: skip
    )
    assert code == 2 and not out.exists() and not opened
    assert not (tmp_path / "logs").exists()


def test_score_dev_refuses_every_other_mode_before_opening_anything(tmp_path):
    missing = tmp_path / "no-such-solutions.json"  # opening it would raise FileNotFoundError
    with pytest.raises(PermissionError, match="never opens a solutions file"):
        score_dev("competition", tmp_path / "s.json", missing, ["a"])


@pytest.mark.parametrize(
    "extra",
    [["--engine-venv", "/venv"], ["--fake-engine", "r.json", "--engine-url", "http://x"], []],
)
def test_refused_arguments_exit_2(tmp_path, extra):
    _, _, paths = make_case(tmp_path)
    out = tmp_path / "submission.json"
    argv = arguments(paths, out, write_config(tmp_path), tmp_path / "logs", *extra)
    assert main(argv) == 2 and not out.exists()


def test_a_bad_config_exits_2(tmp_path):
    _, _, paths = make_case(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 0\n", encoding="utf-8")
    out = tmp_path / "submission.json"
    argv = arguments(paths, out, bad, tmp_path / "logs", "--fake-engine", str(paths["replies"]))
    assert main(argv) == 2 and not out.exists()


# ---------------------------------------------------------------- engine failures keep NVARC


def run_main(tmp_path, *extra, mode="dev", config=None):
    challenges, solutions, paths = make_case(tmp_path)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    config = config or write_config(tmp_path)
    code = main(arguments(paths, out, config, logs, *extra, mode=mode))
    written = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    summary = json.loads((logs / "run.json").read_text(encoding="utf-8"))
    return code, written, summary, challenges, solutions, paths


def test_an_engine_that_cannot_boot_leaves_nvarc_and_exits_0(tmp_path):
    code, written, summary, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", "unused.json", "--fake-fail-boot"
    )
    assert code == 0
    assert validate(written, challenges_stub(challenges)) == []
    assert written["recolr05"][0]["attempt_1"] == solutions["recolr05"][0]
    assert written["fliplr01"][0] == {"attempt_1": [[0]], "attempt_2": [[0]]}
    starts = [p for p in summary["problems"] if p.startswith("engine start")]
    assert len(starts) == 2  # one more start after a crash during the boot
    assert summary["submission"]["valid"] and "scheduler" not in summary


def test_an_unreachable_engine_url_leaves_nvarc_and_exits_0(tmp_path):
    config = write_config(tmp_path, server={"boot_timeout_s": 0.5})
    url = f"http://127.0.0.1:{free_port()}"
    code, written, summary, challenges, solutions, _ = run_main(
        tmp_path, "--engine-url", url, config=config
    )
    assert code == 0 and validate(written, challenges_stub(challenges)) == []
    assert written["recolr05"][0]["attempt_1"] == solutions["recolr05"][0]
    assert any("did not answer healthy" in p for p in summary["problems"])


def test_an_unreadable_nvarc_file_is_reported_and_the_run_goes_on(tmp_path):
    challenges, solutions, paths = make_case(tmp_path)
    paths["nvarc"].write_text("{not json", encoding="utf-8")
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]))  # fmt: skip
    assert main(argv) == 0
    summary = json.loads((logs / "run.json").read_text())
    assert any("NVARC's submission" in p for p in summary["problems"])
    assert validate(json.loads(out.read_text()), challenges_stub(challenges)) == []


def test_unreadable_tasks_exit_1_and_leave_the_old_file(tmp_path):
    _, _, paths = make_case(tmp_path)
    paths["challenges"].write_text("[]", encoding="utf-8")
    out = tmp_path / "submission.json"
    out.write_text('{"kept": true}', encoding="utf-8")  # NVARC's file, written before
    argv = arguments(paths, out, write_config(tmp_path), tmp_path / "logs", "--fake-engine",
                     str(paths["replies"]))  # fmt: skip
    assert main(argv) == 1
    assert json.loads(out.read_text()) == {"kept": True}


def test_the_deadline_before_the_boot_keeps_nvarc(tmp_path):
    config = write_config(tmp_path, deadline={"final_margin_s": 300})  # the budget: past
    code, written, summary, challenges, solutions, paths = run_main(
        tmp_path, "--fake-engine", "unused.json", config=config
    )
    assert code == 0 and written["recolr05"][0]["attempt_1"] == solutions["recolr05"][0]
    assert any("deadline" in p for p in summary["problems"])


# ---------------------------------------------------------------- full runs


def test_a_competition_run_never_reads_solutions(tmp_path, monkeypatch):
    def refuse(*args):
        raise AssertionError("a solutions file was opened")

    monkeypatch.setattr("arc26.evaluate.load_solutions", refuse)
    monkeypatch.setattr(agent_run, "score_dev", refuse)
    challenges, _, paths = make_case(tmp_path)
    code, written, summary, challenges, solutions, _ = run_main(
        tmp_path, "--fake-engine", str(paths["replies"]), mode="competition"
    )
    assert code == 0 and summary["problems"] == []
    assert score(written, solutions) == 5  # every task but tiling06
    assert not (tmp_path / "logs" / "transcripts.jsonl").exists()  # dev only
    assert summary["scheduler"]["stopped"] == "done"


def test_a_dev_run_scores_and_keeps_transcripts(tmp_path):
    _, _, paths = make_case(tmp_path)
    code, written, summary, challenges, solutions, _ = run_main(
        tmp_path, "--fake-engine", str(paths["replies"]), "--solutions", str(paths["solutions"])
    )
    assert code == 0
    assert summary["score"]["score"] == 5.0 and summary["score"]["tasks"] == 6
    assert summary["submission"]["slots"]["nvarc"] >= 1
    lines = (tmp_path / "logs" / "transcripts.jsonl").read_text().splitlines()
    assert len(lines) == len(read_records(tmp_path / "logs" / "attempts.jsonl")[0])
    timing = (tmp_path / "logs" / "timing-engine.jsonl").read_text().splitlines()
    assert timing and all(json.loads(line)["op"] == "engine" for line in timing)


def test_an_engine_crash_mid_run_is_restarted_and_the_run_finishes(tmp_path):
    _, _, paths = make_case(tmp_path)
    marker = tmp_path / "crashed"
    code, written, summary, challenges, solutions, _ = run_main(
        tmp_path, "--fake-engine", str(paths["replies"]), "--fake-crash-after", "3",
        "--fake-crash-marker", str(marker),
    )  # fmt: skip
    assert code == 0 and marker.exists()
    assert [e["event"] for e in summary["engine"]["events"]] == ["down", "restarted"]
    assert score(written, solutions) == 5  # attempts cut by the crash were run again


def test_end_to_end_kill_and_resume_keeps_a_valid_submission(tmp_path):
    challenges, solutions, paths = make_case(tmp_path)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]), "--fake-delay-s", "0.3", "--solutions",
                     str(paths["solutions"]))  # fmt: skip
    command = [sys.executable, "-m", "arc26.agent.run", *argv]
    attempts = logs / "attempts.jsonl"
    with open(tmp_path / "first.log", "wb") as log:
        first = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=tmp_path)
    try:
        end = time.monotonic() + 90
        while time.monotonic() < end and first.poll() is None:
            if attempts.exists() and len(read_records(attempts)[0]) >= 2:
                break
            time.sleep(0.05)
        assert first.poll() is None, (tmp_path / "first.log").read_text()[-3000:]
        first.send_signal(signal.SIGKILL)  # no cleanup, no final write
        first.wait(10)
        # Valid at the moment of the kill: NVARC's fill plus what was finished.
        assert validate(json.loads(out.read_text()), challenges_stub(challenges)) == []
        before = [r.attempt_id for r in read_records(attempts)[0]]
        assert 2 <= len(before) < 8
        stale = json.loads((logs / "engine.pid").read_text())["pid"]
        assert alive(stale)  # the killed run's engine is still up

        second = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=tmp_path)
        assert second.returncode == 0, second.stderr[-3000:]
        assert not alive(stale)  # the resumed run stopped the stale engine first
        summary = json.loads((logs / "run.json").read_text())
        assert summary["resume"]["records"] == len(before) and summary["problems"] == []
        records, _ = read_records(attempts)
        ids = [r.attempt_id for r in records]
        assert len(ids) == len(set(ids)) and ids[: len(before)] == before
        assert {i.split("/")[0] for i in ids if "/p1/" in i} == set(challenges)
        written = json.loads(out.read_text())
        assert validate(written, challenges_stub(challenges)) == []
        assert summary["score"]["score"] == 5.0 and score(written, solutions) == 5
    finally:
        if first.poll() is None:
            first.kill()
        stop_engine(logs)


def test_sigterm_stops_the_engine_and_writes_the_final_file(tmp_path):
    challenges, _, paths = make_case(tmp_path)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]), "--fake-delay-s", "5")  # fmt: skip
    process = subprocess.Popen(
        [sys.executable, "-m", "arc26.agent.run", *argv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_path,
    )  # fmt: skip
    try:
        pid_file = logs / "engine.pid"
        end = time.monotonic() + 30
        while time.monotonic() < end and not (pid_file.exists() and out.exists()):
            time.sleep(0.05)
        engine = json.loads(pid_file.read_text())["pid"]
        time.sleep(0.5)  # attempts are waiting on the slow engine
        process.send_signal(signal.SIGTERM)
        assert process.wait(30) == 0
        assert not alive(engine)
        summary = json.loads((logs / "run.json").read_text())
        assert "SystemExit: 143" in summary["problems"]
        assert summary["submission"]["valid"]
        assert validate(json.loads(out.read_text()), challenges_stub(challenges)) == []
    finally:
        if process.poll() is None:
            process.kill()
        stop_engine(logs)


def test_a_sandbox_that_cannot_start_is_reported(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("cannot start a sandbox")

    monkeypatch.setattr("arc26.agent.sandbox.SubprocessRepl", broken)
    _, _, paths = make_case(tmp_path)
    code, written, summary, challenges, solutions, _ = run_main(
        tmp_path, "--fake-engine", str(paths["replies"])
    )
    assert code == 0 and validate(written, challenges_stub(challenges)) == []
    assert written["recolr05"][0]["attempt_1"] == solutions["recolr05"][0]
    assert any("attempts raised" in p for p in summary["problems"])
    assert summary["scheduler"]["stopped"] == "done"


def test_an_unusable_log_folder_falls_back_to_a_temporary_one(tmp_path):
    challenges, solutions, paths = make_case(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the log folder should be", encoding="utf-8")
    out = tmp_path / "submission.json"
    argv = arguments(paths, out, write_config(tmp_path), blocker / "logs", "--fake-engine",
                     str(paths["replies"]))  # fmt: skip
    assert main(argv) == 0
    written = json.loads(out.read_text())
    assert validate(written, challenges_stub(challenges)) == []
    assert score(written, solutions) == 5


# ---------------------------------------------------------------- fix round 1


def test_competition_mode_needs_the_sessions_deadline(tmp_path):
    _, _, paths = make_case(tmp_path)
    out = tmp_path / "submission.json"
    config = write_config(tmp_path)
    fake = ["--fake-engine", str(paths["replies"])]
    bare = arguments(paths, out, config, tmp_path / "logs", *fake, mode="competition",
                     deadline=False)  # fmt: skip
    assert main(bare) == 2  # this process starts after NVARC's stage: it cannot guess the end
    assert main([*bare, "--budget-s", "300"]) == 2  # relative to this process: refused too
    assert not out.exists()


def test_session_start_and_the_development_default_set_the_end(tmp_path):
    _, _, paths = make_case(tmp_path)
    config = write_config(tmp_path)  # session_s 43200, startup margin 0, final margin 2
    fake = ["--fake-engine", str(paths["replies"])]
    started = time.time() - 100.0  # the notebook started 100 s ago
    out, logs = tmp_path / "a" / "submission.json", tmp_path / "a" / "logs"
    argv = arguments(paths, out, config, logs, *fake, "--session-start", str(started),
                     mode="competition", deadline=False)  # fmt: skip
    assert main(argv) == 0
    summary = json.loads((logs / "run.json").read_text())
    assert summary["deadline_source"] == "session_start"
    assert summary["deadline_in_s"] == pytest.approx(43200 - 100 - 2, abs=5)
    out, logs = tmp_path / "b" / "submission.json", tmp_path / "b" / "logs"
    assert main(arguments(paths, out, config, logs, *fake, deadline=False)) == 0
    summary = json.loads((logs / "run.json").read_text())
    assert summary["deadline_source"] == "default"  # from this process's start
    assert summary["deadline_in_s"] == pytest.approx(43200 - 0 - 2, abs=5)


def test_a_resume_reads_nvarc_from_its_copy_when_the_output_is_nvarcs_file(tmp_path):
    challenges, solutions, paths = make_case(tmp_path)
    nvarc_other = [[5, 5], [5, 5]]  # NVARC's attempt 1 for flipud02, unlike both programs
    nvarc = json.loads(paths["nvarc"].read_text())
    nvarc["flipud02"] = [{"attempt_1": nvarc_other, "attempt_2": [[0]]}]
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    out.write_text(json.dumps(nvarc), encoding="utf-8")  # NVARC's stage wrote the output path
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]), nvarc=out)  # fmt: skip
    assert main(argv) == 0
    first = json.loads(out.read_text())
    # flipud02: its verified program first, NVARC's attempt 1 second; the wrong first program
    # (unverified) ranks after NVARC's.
    right = solutions["flipud02"][0]
    assert first["flipud02"][0] == {"attempt_1": right, "attempt_2": nvarc_other}
    assert json.loads((logs / "nvarc_submission.json").read_text()) == nvarc  # the copy
    before = read_records(logs / "attempts.jsonl")[0]
    assert main(argv) == 0  # the resume: the output now holds this run's own attempts
    summary = json.loads((logs / "run.json").read_text())
    assert summary["resume"]["records"] == len(before) and summary["problems"] == []
    assert json.loads(out.read_text()) == first  # attempt 2 is still NVARC's
    assert len(read_records(logs / "attempts.jsonl")[0]) == len(before)  # nothing run again


def test_a_nvarc_file_unreadable_at_the_first_start_is_never_read_later(tmp_path):
    challenges, solutions, paths = make_case(tmp_path)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    out.write_text("{cut", encoding="utf-8")  # NVARC's stage died mid-write
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]), nvarc=out)  # fmt: skip
    assert main(argv) == 0
    assert json.loads((logs / "nvarc_submission.json").read_text()) is None
    assert main(argv) == 0  # the output is now this run's file: never taken for NVARC's
    summary = json.loads((logs / "run.json").read_text())
    assert any("holds no submission" in p for p in summary["problems"])
    written = json.loads(out.read_text())
    assert written["recolr05"][0] == {"attempt_1": [[0]], "attempt_2": [[0]]}


def test_a_kill_during_the_first_call_leaves_a_valid_file(tmp_path):
    challenges, solutions, paths = make_case(tmp_path)
    out, logs = tmp_path / "submission.json", tmp_path / "logs"
    argv = arguments(paths, out, write_config(tmp_path), logs, "--fake-engine",
                     str(paths["replies"]), "--fake-delay-s", "30")  # fmt: skip
    process = subprocess.Popen(
        [sys.executable, "-m", "arc26.agent.run", *argv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_path,
    )  # fmt: skip
    try:
        end = time.monotonic() + 30
        while time.monotonic() < end and not (logs / "engine.pid").exists():
            time.sleep(0.05)
        time.sleep(1.0)  # the first calls are in flight; each takes 30 s
        assert process.poll() is None
        process.send_signal(signal.SIGKILL)
        process.wait(10)
        attempts = logs / "attempts.jsonl"
        assert not attempts.exists() or not read_records(attempts)[0]  # nothing finished
        written = json.loads(out.read_text())  # so the first write came before the first call
        assert validate(written, challenges_stub(challenges)) == []
        assert written["recolr05"][0]["attempt_1"] == solutions["recolr05"][0]
    finally:
        if process.poll() is None:
            process.kill()
        stop_engine(logs)
