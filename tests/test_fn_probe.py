"""Checks version 11's solver (scripts/fn_probe.py, run P5.2-01) and its local report.

The solver runs as its CPU smoke against the fake SGLang server (scripts/fn_fake_sglang.py):
development mode's paired arms (plain and tool, the tool loop through arc26.agent.toolshot of
task P5.3), competition mode's live submission, an engine that
dies mid-run, the forcing turn at the cap and at the deadline, the stop rules, and the outputs
the pull keeps; then version 11's notebook runs its three cells on a fake /kaggle tree, and
scripts/fn_report.py scores the arms and applies the rule. No network beyond localhost, no GPU.
"""

import importlib.util
import json
import socket
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "P5.2-01.yaml"


def load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load("fn_probe")
report = load("fn_report")
gateway = load("kaggle_ops")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def smoke_config(tmp_path, edit=None):
    """The config file of a smoke run: a free port, and ``edit(smoke)`` applied."""
    config = yaml.safe_load(CONFIG.read_text())
    config["smoke"]["engine"]["port"] = free_port()
    if edit is not None:
        edit(config["smoke"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_flags_follow_the_value_rules():
    config = {"engine": {"module": "sglang.launch_server", "script": None, "flags": {
        "--model-path": "{model}", "--tp": 4, "--bare": True, "--off": False, "--unset": None,
        "--sizes": [1, 2], "--extra": {"a": 1}}}}
    argv = probe.server_argv(config, "py", {"model": "/m"})
    assert argv == ["py", "-m", "sglang.launch_server", "--model-path", "/m", "--tp", "4",
                    "--bare", "--sizes", "1", "2", "--extra", '{"a": 1}']


def test_log_and_metric_readings():
    log = ("max_total_num_tokens=301234, chunked_prefill_size=4096, max_running_requests=12\n"
           "max_total_num_tokens=301230, max_running_requests=12\n"
           "Decode batch. #running-req: 3, #token: 9, token usage: 0.1, gen throughput "
           "(token/s): 120.5, #queue-req: 0\n"
           "Decode batch. #running-req: 3, #token: 9, gen throughput (token/s): 99.0\n")
    patterns = yaml.safe_load(CONFIG.read_text())["engine"]["log_patterns"]
    found = probe.parse_log(log, patterns)
    assert found["max_total_num_tokens"] == [301234.0, 301230.0]
    assert found["max_running_requests"] == [12.0, 12.0]
    assert probe.decode_lines(log) == {"3": {"lines": 2, "median_tok_s": 120.5,
                                             "max_tok_s": 120.5}}
    page = 'sglang:num_running_reqs{model="x",tp="0"} 3\nsglang:token_usage{a="b"} 0.25\n# HELP\n'
    assert probe.parse_metrics(page, ["num_running_reqs", "token_usage"]) == {
        "num_running_reqs": 3.0, "token_usage": 0.25}
    assert probe.find_key({"a": [{"b": {"max_total_num_tokens": 7}}]}, "max_total_num_tokens") == 7


def test_stream_chunks_merge_whether_cumulative_or_incremental():
    assert probe._merge("ab", "abc") == "abc"
    assert probe._merge("ab", "c") == "abc"
    assert probe._merge([1, 2], [1, 2, 3]) == [1, 2, 3]
    assert probe._merge([1, 2], [3]) == [1, 2, 3]


def test_fake_tokenizer_round_trips_markers():
    tok = probe.FakeTokenizer()
    text = "<|im_start|>user\nhé<|im_end|>\n<think>\nx</think>\n\n"
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode("</think>") == [257]


def test_answers_are_read_after_the_last_think_close():
    task = type("T", (), {"test_inputs": [[[1]], [[2]]]})()
    text = ('<think>\n```answer\n{"test_index": 0, "grid": [[9]]}\n```\n</think>\n\n'
            '```answer\n{"test_index": 1, "grid": [[2]]}\n```<|im_end|>')
    answers, errors = probe.answers_from(text, task, True, ["<|im_end|>"])
    assert answers == [(1, [[2]])] and errors == []
    assert probe.answers_from("no close", task, True)[1] == ["no </think> in the reply"]


def test_most_streams_takes_the_tightest_bound():
    inner = probe.Inner(run=None, base="", tok=None, pool=300_000, max_running=16)
    assert probe.most_streams(inner, 78_451, 16) == 3
    inner.pool = 2_000_000
    assert probe.most_streams(inner, 78_451, 16) == 16
    assert probe.most_streams(inner, 78_451, 8) == 8


def run_smoke(tmp_path, edit=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "root"
    code = probe.main(["--config", str(smoke_config(tmp_path, edit)), "--smoke", "--root",
                       str(root), "--commit", "abc123"])
    return code, root / "kaggle/working/run_logs/fn"


def rows(out, name="attempts.jsonl"):
    path = out / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_development_mode_runs_both_arms_on_one_server(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", "sentinel-secret-7f3a")
    code, out = run_smoke(tmp_path)
    assert code == 0
    state = json.loads((out / "state.json").read_text())
    for stage in ("inputs", "cuda", "tasks", "tool_self_test", "arg_check", "boot",
                  "template_check", "solve"):
        assert state[stage]["status"] == "ok", (stage, state[stage])
    assert state["mode"] == "development"
    plan = json.loads((out / "solve_plan.json").read_text())
    assert plan["arms"] == ["plain", "tool"] and plan["streams"] == 4 and plan["per_arm"] == 2
    # The plan at competition's window: the visible test file's 6 tasks; 4 streams of the
    # fake's 20,000-token pool keep the smoke's thinking floor, at context 5,000 (no state).
    assert plan["context"] == 5000 and plan["budget"]["met"] is True
    assert plan["budget"]["n_tasks"] == 6
    assert json.loads((out / "replan.json").read_text())["replanned"] is True
    boot = json.loads((out / "boot.json").read_text())
    assert boot["pool_tokens"] == 20000 and boot["max_running_requests"] == 8
    assert json.loads((out / "template_check.json").read_text())["equal"] is True
    attempts = rows(out)
    by_arm = {arm: [a for a in attempts if a["arm"] == arm] for arm in ("plain", "tool")}
    assert [len(v) for v in by_arm.values()] == [6, 6]
    # One budget rule in both arms: the task's cap (1,000 here), thinking stopping the forcing
    # reserve (300) short of it. The fake thinks past it on every second task; those are forced
    # from their own ids with the prompt and the reply served from the cache.
    forced = [a for a in by_arm["plain"] if a["forced"]]
    assert len(forced) == 3
    for a in forced:
        think, force = a["turns"]
        assert a["cap"] == 1000
        assert think["finish"] == "length" and think["completion_tokens"] == 700
        assert force["cached_tokens"] >= a["opening_tokens"] + 700
    for a in by_arm["tool"]:
        if a["forced"]:
            assert sum(t["completion_tokens"] for t in a["turns"] if t["kind"] != "force") == 700
    assert all(a["completion_tokens"] <= a["cap"] for a in attempts)
    # Tool: one python call per task, continued token for token from the cache.
    for a in by_arm["tool"]:
        assert [c["status"] for c in a["calls"]] == ["ok"]
        tool_turn = next(t for t in a["turns"] if t["kind"] == "tool")
        assert tool_turn["cached_tokens"] >= a["turns"][0]["prompt_tokens"]
    assert all(a["attempts"][0] and a["blocks"] == 1 for a in attempts)
    assert json.loads((out / "run.json").read_text())["commit"] == "abc123"
    timing = rows(out, "timing-inner.jsonl")
    assert {r["op"] for r in timing} >= {"fn_think", "fn_force", "fn_tool"}
    assert all(set(r) == {"puzzle_id", "op", "seconds", "tokens"} for r in timing)
    assert "Decode batch" in (out / "engine_tail.log").read_text()
    for path in out.rglob("*"):
        rel = path.relative_to(out.parent.parent).as_posix()
        assert gateway.pull_allowed(rel), rel
        # The session's environment never reaches a pulled file (it may hold tokens).
        assert "sentinel-secret-7f3a" not in path.read_text(errors="replace"), path
    # Development mode leaves submission.json to the notebook.
    assert not (out.parent.parent / "submission.json").exists()


def test_competition_mode_fills_the_submission_as_tasks_finish(tmp_path):
    def edit(smoke):
        smoke["mode"] = "competition"

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    state = json.loads((out / "state.json").read_text())
    plan = json.loads((out / "solve_plan.json").read_text())
    assert plan["arms"] == ["tool"] and plan["per_arm"] == plan["streams"]
    # Every task of the test file, the smallest opening first (ties by id).
    openings = state["tasks"]
    assert openings["tasks"] == 6 and len(plan["order"]) == 6
    submission = json.loads((out.parent.parent / "submission.json").read_text())
    assert all(entry[0]["attempt_1"] != [[0]] for entry in submission.values())
    assert state["solve"]["submission_filled"] == 6


def test_a_dead_engine_is_restarted_and_its_task_queued_again(tmp_path):
    def edit(smoke):
        smoke["mode"] = "competition"
        smoke["solve"]["max_streams"] = 1
        smoke["engine"]["flags"]["--die-after"] = 2
        smoke["solve"]["dev_arms"] = ["plain"]
        smoke["solve"]["arm"] = "plain"

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    restarts = rows(out, "restarts.jsonl")
    assert restarts[0]["restart"] == 1 and "ready_s" in restarts[0]
    failures = rows(out, "engine_failures.jsonl")
    assert failures[0]["requeued"] is True
    attempts = rows(out)
    # The requeued task finished on the restarted engine; the second death ends the run (the
    # smoke allows one restart) and the placeholders of the rest stay valid.
    assert failures[0]["task_id"] in {a["task_id"] for a in attempts if "error" not in a}
    state = json.loads((out / "state.json").read_text())
    assert state["solve"]["killed"] == "the engine is dead and cannot restart"
    from arc26.submit import validate

    submission = json.loads((out.parent.parent / "submission.json").read_text())
    challenges = json.loads(next((out.parent.parent.parent / "input").rglob(
        "arc-agi_test_challenges.json")).read_text())
    assert validate(submission, challenges) == []


def test_a_failed_tool_self_test_runs_plain(tmp_path):
    def edit(smoke):
        smoke["mode"] = "competition"
        smoke["tool"]["module"] = "no_such_toolshot"

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    state = json.loads((out / "state.json").read_text())
    assert state["tool_self_test"]["status"] == "failed"
    assert json.loads((out / "solve_plan.json").read_text())["arms"] == ["plain"]
    assert state["solve"]["submission_filled"] == 6


def test_a_tool_error_in_competition_runs_the_task_plain(tmp_path):
    def edit(smoke):
        smoke["mode"] = "competition"
        smoke["tool"].update(module="toolshot_raises", extra_path="../tests/testdata")

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    attempts = rows(out)
    assert len(attempts) == 6 and all(a["arm"] == "plain" for a in attempts)
    assert all(a["after_tool_error"] for a in attempts)
    assert len(rows(out, "tool_errors.jsonl")) == 6
    state = json.loads((out / "state.json").read_text())
    assert state["solve"]["submission_filled"] == 6 and state["solve"]["killed"] is None


def test_the_writer_never_reverts_an_answered_input(tmp_path):
    challenges = {"t1": {"train": [], "test": [{"input": [[1]]}, {"input": [[2]]}]}}
    path = tmp_path / "submission.json"
    writer = probe.SubmissionWriter(path, challenges)
    assert json.loads(path.read_text())["t1"][0] == {"attempt_1": [[0]], "attempt_2": [[0]]}
    assert writer.update("t1", [[[[5]], [[6]]], []]) == 1
    assert writer.update("t1", [[[[7]]], [[[8]]]]) == 1
    data = json.loads(path.read_text())
    assert data["t1"][0] == {"attempt_1": [[5]], "attempt_2": [[6]]}
    assert data["t1"][1] == {"attempt_1": [[8]], "attempt_2": [[0]]}
    assert writer.update("unknown", [[[[1]]]]) == 0
    # A second writer keeps what the first wrote.
    probe.SubmissionWriter(path, challenges)
    assert json.loads(path.read_text()) == data


@pytest.mark.parametrize(("environ", "expected"), [
    ({}, "development"), ({"KAGGLE_IS_COMPETITION_RERUN": "True"}, "competition"),
    ({"KAGGLE_IS_COMPETITION_RERUN": "0"}, "development")])
def test_the_mode_follows_kaggles_rerun_flag(environ, expected):
    assert probe.run_mode({"mode": "auto"}, environ) == expected
    assert probe.run_mode({"mode": "competition"}, {}) == "competition"


def test_a_reply_cut_at_the_deadline_is_forced(tmp_path):
    # A model that thinks far past the cap and a budget that ends first: the thinking turn is
    # cut at force_at, and the forcing turn continues it from its own tokens.
    def edit(smoke):
        smoke["engine"]["flags"]["--think-tokens"] = 100000
        smoke["engine"]["flags"]["--token-delay"] = 0.002
        smoke["time_plan"]["development"] = {"generation_stop_s": 9, "end_s": 12,
                                             "min_start_s": 1}
        smoke["solve"].update(task_ids=["s0000001"], cap=60000, force_reserve_s=4,
                              max_streams=1, dev_arms=["plain"])

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    (attempt,) = rows(out)
    assert attempt["ended_by"] == "cancelled+force"
    think, force = attempt["turns"]
    assert 0 < think["completion_tokens"] < 60000 - 300 and force["finish"] == "stop"
    assert attempt["attempts"][0]


def test_degenerate_attempts_stop_new_starts(tmp_path):
    def edit(smoke):
        smoke["engine"]["flags"]["--degenerate"] = True
        smoke["solve"].update(max_streams=1, dev_arms=["plain"])

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    state = json.loads((out / "state.json").read_text())
    # The fake repeats one byte for its whole thinking (80 bytes or more, over the 64 rule):
    # the first two attempts are degenerate, and the run stops starting tasks.
    assert state["solve"]["killed"] == "2 of 2 attempts degenerate"
    assert len(state["solve"]["not_started"]) == 4
    assert probe.max_repeat_run([5, 5, 5, 300, 5, 5], 256) == 3


def test_report_scores_each_arm_and_applies_the_rule(tmp_path):
    code, out = run_smoke(tmp_path)
    assert code == 0
    # The fake answers each test input with itself: right where the solution is the input.
    solutions = {f"s{k:07d}": [[[k % 10, 0], [0, 1]]] if k % 2 else [[[9]]] for k in range(1, 7)}
    sol = tmp_path / "solutions.json"
    sol.write_text(json.dumps(solutions))
    dest = tmp_path / "report"
    assert report.main(["--run-logs", str(out), "--config", str(CONFIG), "--solutions", str(sol),
                        "--out", str(dest), "--smoke"]) == 0
    values = json.loads((dest / "fn_report.json").read_text())
    assert len(values["paired"]) == 6
    assert values["arms"]["plain"]["paired_score"] == values["arms"]["tool"]["paired_score"] == 3.0
    assert values["arms"]["tool"]["calls"] == 6
    assert values["arms"]["tool"]["continuation_cached_share"] > 0.9
    assert values["decision"] in {"A", "B", "human"}
    assert "Decision" in (dest / "fn_report.md").read_text()


def arms(s_a, s_b, wall_a=100.0, wall_b=110.0, cached=0.95):
    return {"fault": None, "paired": ["t"], "arms": {
        "plain": {"paired_score": s_a, "median_wall_s": wall_a},
        "tool": {"paired_score": s_b, "median_wall_s": wall_b,
                 "continuation_cached_share": cached}}}


@pytest.mark.parametrize(("values", "expected"), [
    (arms(3, 3), "B"), (arms(3, 5), "B"), (arms(5, 3), "A"), (arms(4, 3), "human"),
    (arms(3, 4, wall_b=140.0), "human"), (arms(3, 4, wall_b=151.0), "A"),
    (arms(3, 4, cached=0.8), "human"), ({**arms(3, 9), "fault": "boot failed"}, "A"),
    ({**arms(3, 3), "paired": []}, "A")])
def test_the_pre_registered_rule(values, expected):
    rule = yaml.safe_load(CONFIG.read_text())["report"]["rule"]
    assert report.decide(values, rule)[0] == expected


def test_report_reads_a_folder_of_task_files(tmp_path):
    (tmp_path / "t1.json").write_text(json.dumps({"train": [], "test": [{"output": [[1]]}]}))
    assert report.load_solutions(tmp_path, ["t1", "t2"]) == {"t1": [[[1]]]}


def test_report_reference_matches_the_p6_9_01_record():
    config = yaml.safe_load(CONFIG.read_text())
    scores = config["report"]["reference_scores"]
    # Revision 16 runs P6.9-01's 15 tasks again (revisions 11 to 15 ran the first).
    assert list(scores) == config["solve"]["task_ids"] and len(scores) == 15
    assert sum(scores.values()) == 6.0  # docs/experiments/P6.9-01.md: 6.0 of 15


def test_solve_request_is_version_10s_with_both_blocks_asked():
    config = yaml.safe_load(CONFIG.read_text())
    shot = yaml.safe_load((REPO / "configs" / "agent_shot_P5.1.yaml").read_text())
    p69 = yaml.safe_load((REPO / "configs" / "P6.9-01.yaml").read_text())
    solve = config["solve"]
    # Revision 20: 49,152 instead of version 10's 65,536 (the budget note, b54410b).
    assert solve["cap"] == 49_152 < shot["waves"]["max_cap"] == 65_536
    assert solve["forcing_tokens"] == p69["direct"]["forcing_tokens"]
    # Revision 16: P6.9-01's 15 tasks again, in its order.
    assert solve["system"] == "direct_two" and solve["task_ids"] == p69["direct"]["task_ids"]
    # P5.1-01's forced replies wrote one block per output: the note now asks for two.
    assert "two final answer blocks" in solve["thinking_end_note"]
    # The tool loop takes the same note, cap, and forcing reserve per task (tool_budget).
    inner = probe.Inner(run=probe.Run(config=config, root=REPO, out=REPO, start=0.0), base="",
                        tok=None, context=57_000)
    loop = probe.tool_budget(inner, 3_000)
    assert loop["thinking_end_note"] == solve["thinking_end_note"]
    # Revision 20: at a 57,000 context the cap 49,152 binds (57,000 - 3,000 - 128 is above it).
    assert loop["cap"] == probe.task_cap(57_000, 3_000, solve) == 49_152
    assert loop["forcing_tokens"] == solve["forcing_tokens"]
    # The tool replies scale with the context (at most c / 5), not P5.3's fixed 16,384.
    assert loop["reply_tokens"] == 57_000 // 5 and loop["context_tokens"] == 57_000 - 128
    from arc26.agent import toolshot

    cfg = toolshot.tool_config(loop)
    assert cfg.reply_tokens == 11_400 and cfg.cap == loop["cap"]
    assert cfg.thinking_end_note == solve["thinking_end_note"]
    # Revision 16: development runs competition's arm only.
    assert solve["arm"] == "tool" and solve["dev_arms"] == ["tool"]


def test_the_probe_and_the_check_share_one_patch_function():
    import inspect

    env = load("fn_env")
    strip = lambda f: inspect.getsource(f).split('"""', 2)[2]  # noqa: E731
    assert strip(probe.apply_patch) == strip(env.apply_patch)


def test_a_failed_edit_sets_its_fallback_flags(tmp_path):
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    target = site / "sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
    target.parent.mkdir(parents=True)
    target.write_text("not the fork's file\n")
    config = yaml.safe_load(CONFIG.read_text())
    run = probe.Run(config=config, root=tmp_path, out=tmp_path / "out", start=0.0)
    run.out.mkdir()
    results = {r["name"]: r for r in probe.apply_patches(run, tmp_path / "venv")}
    assert results["fa2_scratch_dtype"]["status"] == "sha256 differs"
    # The other files are absent here: recorded as missing; only the FP8 edit's and the EP
    # mask's fallback flags apply (EP 1, never EP > 1 without the mask).
    assert results["logits_gather_nccl"]["status"] == "missing"
    assert results["marlin_moe_torch_sum"]["status"] == "missing"
    assert results["marlin_ep_mask_import"]["status"] == "missing"
    assert run.state["flag_overrides"] == {"--kv-cache-dtype": "bfloat16",
                                           "--expert-parallel-size": 1}
    assert json.loads((run.out / "patches.json").read_text())[2]["name"] == "marlin_moe_torch_sum"
    argv = probe.server_argv(config, "py", {"model": "/m", "port": 1, "chat_template": "t"},
                             run.state["flag_overrides"])
    assert argv[argv.index("--kv-cache-dtype") + 1] == "bfloat16"


def test_the_outer_stages_hand_the_python_path_to_the_install(tmp_path, monkeypatch):
    # The P5.2-01-cpu fault: the python stage's record overwrote the path the install reads.
    # Here the stages run in sequence through the real Run.stage wrapper, with a 3.12 tarball
    # (the image's Python counts as another version) and the commands recorded, not run.
    import tarfile
    import zipfile

    config = yaml.safe_load(CONFIG.read_text())
    config["python"]["want"] = "9.99"
    root = tmp_path / "root"
    env_out = root / "kaggle/input/notebooks/alyahya737/arc26-fn-env"
    wheelhouse = root / "kaggle/input/datasets/dfranzen/pennyroyal-v253"
    code = root / "kaggle/input/datasets/alyahya737/arc26-code-fn"
    stage_dir = tmp_path / "py/python/cpython-9.99/bin"
    stage_dir.mkdir(parents=True)
    (stage_dir / "python9.99").write_text("")
    env_out.mkdir(parents=True)
    with tarfile.open(env_out / "python312.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "py/python", arcname="python")
    (env_out / "fn_env_output.json").write_text("{}")
    (wheelhouse / "wheels").mkdir(parents=True)
    (wheelhouse / "requirements.lock").write_text("")
    with zipfile.ZipFile(wheelhouse / "wheels/uv-0.1-py3-none-any.whl", "w") as archive:
        archive.writestr("uv-0.1.data/scripts/uv", b"\x7fELF")
    (wheelhouse / "wheels/sglang-0.1-py3-none-any.whl").write_text("")
    code.mkdir(parents=True)
    (code / "fn_code.json").write_text("{}")
    (code / "arc26-0.0.1-py3-none-any.whl").write_text("")
    for name in ("model", "competition"):
        spec = config["inputs"][name]
        folder = root / spec["path"].lstrip("/")
        folder.mkdir(parents=True)
        (folder / spec["marker"]).write_text("{}")
    commands = []

    def fake_run(command, timeout, env):
        commands.append([str(c) for c in command])
        if command[1] == "venv":
            venv = Path(command[-1])
            (venv / "bin").mkdir(parents=True)
            (venv / "bin/python").write_text("")
        return {"returncode": 0, "stdout": "", "stderr": "", "command": " ".join(command),
                "seconds": 0.0}

    monkeypatch.setattr(probe, "_run_cmd", fake_run)
    config["local_inputs"]["reserve_gb"] = 0
    run = probe.Run(config=config, root=root, out=tmp_path / "out", start=probe._now(),
                    state={"mode": "development"})
    run.out.mkdir()
    for name, fn in (("inputs", probe.stage_inputs), ("local_inputs", probe.stage_local_inputs),
                     ("python", probe.stage_python), ("install", probe.stage_install)):
        record = run.stage(name, lambda fn=fn: fn(run))
        assert record["status"] == "ok", (name, record)
    venv_command = next(c for c in commands if c[1] == "venv")
    python = venv_command[venv_command.index("--python") + 1]
    assert python.endswith("cpython-9.99/bin/python9.99") and Path(python).is_file()
    assert run.state["python"]["status"] == "ok" and run.state["python_path"] == python
    # The install reads the local copies (P5.2-01-cpu6: the lock from the mount timed out); the
    # mount's paths are kept beside them.
    local = root / "tmp/fn/inputs"
    lock = next(c for c in commands if "-r" in c)
    assert lock[lock.index("-r") + 1] == str(local / "wheelhouse/requirements.lock")
    links = [lock[i + 1] for i, c in enumerate(lock) if c == "--find-links"]
    assert links == [str(local / "wheelhouse/wheels"), str(local / "env_output/wheels")] or \
        links == [str(local / "wheelhouse/wheels")]
    assert run.state["mount_paths"]["wheelhouse"] == str(wheelhouse)
    assert run.state["paths"]["wheelhouse"] == str(local / "wheelhouse")
    assert not (local / "wheelhouse/wheels/sglang-0.1-py3-none-any.whl").is_symlink()


def test_a_stage_that_writes_its_own_key_is_refused(tmp_path):
    run = probe.Run(config={}, root=tmp_path, out=tmp_path, start=0.0)

    def clash():
        run.state["clash"] = "a value the record would overwrite"
        return {}

    record = run.stage("clash", clash)
    assert record["status"] == "failed" and "its own record's key" in record["error"]


# The twin's measured openings (runs/P5.2-01-cpu3, solve_plan.json: the visible test file's 240
# tasks, largest 6,896, mean 1,782); no state reserve (SGLang's Mamba slots are outside the pool).
OPENINGS = [6_896] + [1_782] * 239


@pytest.mark.parametrize(("pool", "streams", "context"), [
    (70_000, 1, 70_000), (89_856, 2, 44_928), (129_000, 3, 43_000), (169_728, 3, 56_576)])
def test_the_planner_keeps_the_thinking_floor(pool, streams, context):
    """Revision 15's floor of 29,000 at 4 x L4's pools (fn_tonight_settings.md, d92431a)."""
    config = yaml.safe_load(CONFIG.read_text())
    spec, planner = config["solve"], config["planner"]
    assert spec["state_reserve_tokens"] == 0
    # Revision 15's 3-stream plans (the server's limit passed as 3; revision 17 allows 4).
    assert planner["min_thinking_tokens"] == 29_000 and spec["max_streams"] == 5
    plan = probe.plan_budget(pool, 3, 30.0, 36_900.0, 240, OPENINGS, 1, spec, planner)
    assert (plan["streams"], plan["context"]) == (streams, context)
    assert plan["met"] is True and plan["thinking"] >= 29_000
    assert plan["streams"] * plan["context"] <= pool
    # One more stream would break the floor, up to the server's limit.
    if streams < 3:  # the limit passed above
        tighter = pool // (streams + 1)
        assert tighter - 6_896 - spec["forcing_tokens"] - spec["context_margin_tokens"] < 29_000
    assert plan["coverage"] < 240  # coverage is reported, not met
    two = probe.plan_budget(pool, 3, 30.0, 36_900.0, 240, OPENINGS, 2, spec, planner)
    assert two["per_arm"] == max(1, two["streams"] // 2) and two["streams"] * two["context"] <= pool


def test_below_the_floor_each_arm_keeps_a_stream_inside_the_pool():
    config = yaml.safe_load(CONFIG.read_text())
    spec, planner = config["solve"], config["planner"]
    # Development's two arms at a small pool: two streams at half the pool each, the floor
    # (29,000 since revision 15: 40,000 - 13,168 = 26,832 under it) recorded as missed, never two
    # streams of 86,016.
    plan = probe.plan_budget(80_000, 4, 30.0, 36_900.0, 240, OPENINGS, 2, spec, planner)
    assert (plan["streams"], plan["context"], plan["per_arm"], plan["met"]) == (2, 40_000, 1,
                                                                                False)
    assert plan["streams"] * plan["context"] <= 80_000
    # Competition's one arm: one stream at what the pool allows, not 86,016.
    one = probe.plan_budget(60_000, 8, 30.0, 36_900.0, 240, OPENINGS, 1, spec, planner)
    assert (one["streams"], one["context"], one["met"]) == (1, 60_000, True)
    tiny = probe.plan_budget(40_000, 8, 30.0, 36_900.0, 240, OPENINGS, 1, spec, planner)
    assert (tiny["streams"], tiny["context"], tiny["met"]) == (1, 40_000, False)
    assert probe.task_cap(50_000, 3_000, spec) == 50_000 - 3_000 - 128
    # Revision 20: the cap 49,152 binds at a long context.
    assert probe.task_cap(86_016, 1_000, spec) == 49_152


def test_a_reply_cut_by_a_dying_engine_is_queued_again_not_forced(tmp_path):
    def edit(smoke):
        smoke["solve"].update(dev_arms=["plain"], max_streams=1)
        # Request 2 is the first task's thinking turn (request 1 is the template check); no
        # stress check (revision 22) before it, whose requests would come first.
        smoke["engine"]["flags"]["--die-mid-stream"] = 2
        smoke["stress"]["enabled"] = False

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    (failure,) = rows(out, "engine_failures.jsonl")
    assert failure["requeued"] is True
    assert [t["kind"] for t in failure["turns"]] == ["think"]
    assert failure["turns"][0]["finish"] in ("abort", "error")
    restarts = rows(out, "restarts.jsonl")
    assert len(restarts) == 1 and "ready_s" in restarts[0]
    attempts = rows(out)
    assert len(attempts) == 6 and all("error" not in a for a in attempts)
    assert failure["task_id"] in {a["task_id"] for a in attempts}


def test_the_rate_report_never_waits_on_a_restart():
    config = yaml.safe_load(CONFIG.read_text())
    inner = probe.Inner(run=probe.Run(config=config, root=REPO, out=REPO, start=0.0), base="",
                        tok=None)
    inner.plan = {"replanned": True}
    import threading

    with inner.engine_lock:  # a restart in progress holds it through the boot
        done = threading.Event()
        threading.Thread(target=lambda: (probe.observe_rate(inner, 30.0), done.set()),
                         daemon=True).start()
        assert done.wait(2.0)


def test_a_partial_run_reads_fault(tmp_path):
    code, out = run_smoke(tmp_path)
    assert code == 0
    state = json.loads((out / "state.json").read_text())
    del state["solve"]  # a Timeout kill leaves no finished solve stage
    (out / "state.json").write_text(json.dumps(state))
    solutions = {f"s{k:07d}": [[[k % 10, 0], [0, 1]]] for k in range(1, 7)}
    sol = tmp_path / "solutions.json"
    sol.write_text(json.dumps(solutions))
    config = yaml.safe_load(CONFIG.read_text())
    config = probe.apply_smoke(config)
    values = report.report(out, config, sol)
    assert values["decision"] == "A" and "solve missing" in values["decision_reason"]


def test_requests_stay_under_the_context_and_refusals_are_errors(tmp_path):
    def edit(smoke):
        smoke["engine"]["flags"]["--context-length"] = 2000
        smoke["solve"]["dev_arms"] = ["plain"]

    code, out = run_smoke(tmp_path, edit)
    assert code == 0
    # The smoke's planned context 4,096 would pass the fake's 2,000: each request is clamped
    # to the config's context (86,016) less the margin, so the fake refuses, and the refusal is
    # recorded as an error with its message, not an abort.
    turns = [t for a in rows(out) for t in a["turns"]]
    refused = [t for t in turns if t["finish"] == "error"]
    assert refused and all("maximum context length" in t["error"] for t in refused)


def test_every_request_keeps_a_margin_under_the_context(monkeypatch):
    config = yaml.safe_load(CONFIG.read_text())
    sent = []

    class Recorder:
        def __init__(self, base, body, timeout, incremental):
            sent.append(body["sampling_params"]["max_new_tokens"])

        def cancel(self):
            pass

        def run(self):
            return probe.GenResult(finish="stop", start=1.0, end=1.0)

    monkeypatch.setattr(probe, "Generation", Recorder)
    inner = probe.Inner(run=probe.Run(config=config, root=REPO, out=REPO, start=0.0), base="",
                        tok=None)
    turns = []
    generate = probe.make_generate(inner, "t", turns)
    generate([0] * 85_000, 65_536, probe._now() + 60, "think", {})
    assert sent == [86_016 - 128 - 85_000]
    result = generate([0] * 85_900, 6_144, probe._now() + 60, "force", {})
    assert result.finish == "error" and "no room" in result.error and len(sent) == 1


def test_a_failed_first_generation_stops_development_but_not_competition(tmp_path):
    def development(smoke):
        smoke["engine"]["flags"]["--fail-one-token"] = True

    code, out = run_smoke(tmp_path / "dev", development)
    state = json.loads((out / "state.json").read_text())
    assert code == 1 and state["killed"].startswith("the first generation failed")
    assert "solve" not in state

    def competition(smoke):
        smoke["engine"]["flags"]["--fail-one-token"] = True
        smoke["mode"] = "competition"

    code, out = run_smoke(tmp_path / "comp", competition)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["first_generation_failed"] is True
    assert state["solve"]["submission_filled"] == 6


def test_a_refused_flag_fails_development_but_not_competition(tmp_path):
    """P5.2-03's boot failure: the fork refuses --language-model-only for Flash-Next."""

    def development(smoke):
        smoke["engine"]["flags"]["--language-model-only"] = True

    code, out = run_smoke(tmp_path / "dev", development)
    state = json.loads((out / "state.json").read_text())
    assert code == 1 and state["arg_check"]["status"] == "failed"
    # Revision 22: the stress check's fallback set is checked too, and refused with the boot's.
    assert state["arg_check"]["verdicts"] == {"boot": "refused", "stress_fallback": "refused"}
    assert "Qwen4ExpForConditionalGeneration" in state["arg_check"]["errors"]["boot"]
    assert state["arg_check_refused"] == ["boot", "stress_fallback"]
    # The check never replaces the boot: the fake boots and the run still solves.
    assert state["boot"]["status"] == "ok" and state["solve"]["status"] == "ok"
    check = json.loads((out / "arg_check.json").read_text())
    assert "--language-model-only" in check["results"]["boot"]["argv"]
    assert check["results"]["boot"]["stopped_in"].endswith("resolve_once")

    def competition(smoke):
        development(smoke)
        smoke["mode"] = "competition"

    code, out = run_smoke(tmp_path / "comp", competition)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["arg_check"]["verdicts"]["boot"] == "refused"
    assert state["solve"]["submission_filled"] == 6


ARG_MODULES = {
    # The parser refuses an unknown flag (no prepare_server_args: ServerArgs' own CLI path).
    "argmod_parser": ("import argparse\nclass ServerArgs:\n"
                      "    @staticmethod\n    def add_cli_args(p):\n"
                      "        p.add_argument('--tp', type=int)\n"
                      "    @classmethod\n    def from_cli_args(cls, ns):\n        return cls()\n"),
    # A device query on a box without a GPU stops the resolution: not a refusal.
    "argmod_device": ("def prepare_server_args(argv):\n"
                      "    raise RuntimeError('Found no NVIDIA driver on your system.')\n"),
    # A ValueError that names the device on a CPU box is the box, not the flags.
    "argmod_cuda_value": ("class A:\n    def resolve_once(self):\n"
                          "        raise ValueError('FP8 KV needs a CUDA device')\n"
                          "def prepare_server_args(argv):\n    return A()\n"),
    # Anything else is an error of the check, never accepted.
    "argmod_other": "def prepare_server_args(argv):\n    raise KeyError('x')\n",
    "argmod_ok": ("class A:\n    def resolve_once(self):\n        pass\n"
                  "def prepare_server_args(argv):\n    return A()\n"),
}


@pytest.mark.parametrize(("module", "argv", "verdict"), [
    ("argmod_parser", ["--tp", "2"], "accepted"),
    ("argmod_parser", ["--tp", "2", "--no-such-flag"], "refused"),
    ("argmod_device", [], "device_stop"),
    ("argmod_cuda_value", [], "device_stop"),
    ("argmod_other", [], "error"),
    ("argmod_ok", [], "accepted"),
    ("no_such_argmod", [], "error"),
])
def test_the_argument_check_classifies_what_stopped_it(tmp_path, module, argv, verdict):
    for name, source in ARG_MODULES.items():
        (tmp_path / f"{name}.py").write_text(source)
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path)}
    result = probe.check_arguments(sys.executable, module, argv, 60, env)
    assert result["verdict"] == verdict, result


def test_a_hung_argument_check_is_killed(tmp_path):
    (tmp_path / "argmod_hang.py").write_text("import time\ntime.sleep(60)\n")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path)}
    result = probe.check_arguments(sys.executable, "argmod_hang", [], 2, env)
    assert result["verdict"] == "error" and "no verdict" in result["error"]


def test_version_11_drops_the_refused_flag_and_the_twin_checks_both_flag_sets():
    base = yaml.safe_load(CONFIG.read_text())
    twin = yaml.safe_load((REPO / "configs" / "P5.2-01-cpu.yaml").read_text())
    assert "--language-model-only" not in base["engine"]["flags"]
    assert base["arg_check"]["enabled"] and base["arg_check"]["module"] == "sglang.srt.server_args"
    # The twin checks the GPU flags only: every flag it adds for the fake is dropped.
    fake_only = set(twin["engine"]["flags"]) - set(base["engine"]["flags"])
    assert fake_only <= set(twin["arg_check"]["drop_flags"])
    assert not set(twin["arg_check"]["drop_flags"]) & set(base["engine"]["flags"])
    assert twin["arg_check"]["retry_flags"] == base["engine"]["boot_retry_flags"]


def make_tree(folder):
    """A small input folder: nested files of several pieces, an empty file, an empty folder."""
    (folder / "wheels").mkdir(parents=True)
    (folder / "empty_dir").mkdir()
    (folder / "requirements.lock").write_text("a==1\n")
    (folder / "wheels/big-1.0-py3-none-any.whl").write_bytes(bytes(range(256)) * 5000)
    (folder / "wheels/zero.whl").write_bytes(b"")
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_the_copy_reads_every_piece_and_verifies_each_size(tmp_path):
    want = make_tree(tmp_path / "mount")
    record = probe.copy_tree(tmp_path / "mount", tmp_path / "local", workers=3,
                             piece_bytes=100_000, max_s=30, stall_s=10, block_bytes=30_000)
    assert record["copied_files"] == record["files"] == 3 and record["linked"] == []
    assert record["copied_bytes"] == record["bytes"] == 1_280_005 and record["errors"] == {}
    assert record["stopped"] is None and record["mb_per_s"] > 0
    got = {p.relative_to(tmp_path / "local"): p.read_bytes()
           for p in (tmp_path / "local").rglob("*") if p.is_file()}
    assert got == want and (tmp_path / "local/empty_dir").is_dir()
    assert not any(p.is_symlink() for p in (tmp_path / "local").rglob("*"))


def slow_writes(monkeypatch, seconds):
    real = probe.os.pwrite

    def pwrite(fd, data, offset):
        time.sleep(seconds)
        return real(fd, data, offset)

    monkeypatch.setattr(probe.os, "pwrite", pwrite)


def test_a_copy_out_of_time_links_the_rest_to_the_mount(tmp_path, monkeypatch):
    want = make_tree(tmp_path / "mount")
    slow_writes(monkeypatch, 0.01)
    record = probe.copy_tree(tmp_path / "mount", tmp_path / "local", workers=1,
                             piece_bytes=1_000, max_s=0.3, stall_s=10, block_bytes=1_000)
    monkeypatch.undo()
    assert record["stopped"] == "max_s"
    assert "wheels/big-1.0-py3-none-any.whl" in record["linked"]
    assert record["copied_files"] + len(record["linked"]) == record["files"]
    for rel, data in want.items():
        path = tmp_path / "local" / rel
        assert path.read_bytes() == data
        if str(rel) in record["linked"]:
            assert path.is_symlink() and path.resolve() == (tmp_path / "mount" / rel).resolve()
    assert not list((tmp_path / "local").rglob("*.part"))


def test_a_stalled_copy_stops_and_links_the_rest(tmp_path, monkeypatch):
    make_tree(tmp_path / "mount")
    slow_writes(monkeypatch, 1.0)
    record = probe.copy_tree(tmp_path / "mount", tmp_path / "local", workers=1,
                             piece_bytes=1 << 20, max_s=30, stall_s=0.3, block_bytes=1 << 20)
    monkeypatch.undo()
    assert record["stopped"] == "stall" and record["linked"]
    assert record["seconds"] < 5


def test_a_file_that_fails_to_read_is_linked_and_recorded(tmp_path, monkeypatch):
    make_tree(tmp_path / "mount")
    real_open = open

    def flaky(path, *args, **kwargs):
        if str(path).endswith("big-1.0-py3-none-any.whl") and "mount" in str(path):
            raise OSError("Input/output error")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky)
    record = probe.copy_tree(tmp_path / "mount", tmp_path / "local", workers=2,
                             piece_bytes=1 << 20, max_s=30, stall_s=10)
    monkeypatch.undo()
    assert record["linked"] == ["wheels/big-1.0-py3-none-any.whl"]
    assert "Input/output error" in record["errors"]["wheels/big-1.0-py3-none-any.whl"]
    assert (tmp_path / "local/wheels/big-1.0-py3-none-any.whl").is_symlink()
    assert (tmp_path / "local/requirements.lock").read_text() == "a==1\n"


def local_run(tmp_path, edit):
    config = yaml.safe_load(CONFIG.read_text())
    config["local_inputs"].update(reserve_gb=0, workers=2, piece_mib=1)
    edit(config["local_inputs"])
    mount = tmp_path / "mount"
    make_tree(mount)
    run = probe.Run(config=config, root=tmp_path / "root", out=tmp_path / "out",
                    start=probe._now(), state={"paths": {"wheelhouse": str(mount),
                                                         "env_output": None}})
    run.out.mkdir()
    return run, mount


def test_without_room_for_the_copy_the_install_keeps_the_mount(tmp_path):
    run, mount = local_run(tmp_path, lambda spec: spec.update(reserve_gb=1e9))
    record = run.stage("local_inputs", lambda: probe.stage_local_inputs(run))
    assert record["status"] == "ok"
    assert record["inputs"]["wheelhouse"]["local"] is False
    assert "no room" in record["inputs"]["wheelhouse"]["reason"]
    assert record["inputs"]["env_output"] == {"local": False, "reason": "not found"}
    assert run.state["paths"]["wheelhouse"] == str(mount) and "mount_paths" not in run.state


def test_a_copy_error_keeps_the_mount(tmp_path, monkeypatch):
    run, mount = local_run(tmp_path, lambda spec: None)

    def broken(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(probe, "copy_tree", broken)
    record = run.stage("local_inputs", lambda: probe.stage_local_inputs(run))
    assert record["inputs"]["wheelhouse"]["local"] is False
    assert "No space left" in record["inputs"]["wheelhouse"]["reason"]
    assert run.state["paths"]["wheelhouse"] == str(mount)
    assert not (tmp_path / "root/tmp/fn/inputs/wheelhouse").exists()


def test_the_precache_starts_only_after_the_wheel_copy(tmp_path, monkeypatch):
    starts = []
    monkeypatch.setattr(probe.Precache, "start",
                        lambda self: starts.append(probe._now()) or self.run())

    def edit(smoke):
        smoke["precache"] = {"enabled": True, "inputs": ["model"], "threads": 2, "chunk_mib": 1,
                             "max_s": 5}

    code, out = run_smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["local_inputs"]["status"] == "ok" and starts
    assert state["local_inputs"]["inputs"]["wheelhouse"]["local"] is True
    assert state["local_inputs_ended_s"] <= state["precache_started_s"]
    assert state["precache"]["done"] is True
    # The precache reads the model's mount, the copy only the install's inputs.
    assert state["paths"]["model"].endswith("models/fake/1")
    assert state["paths"]["wheelhouse"].endswith("tmp/fn/inputs/wheelhouse")


def test_the_install_waits_an_hour_and_the_copy_comes_first():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["install"]["timeout_s"] == 3600
    spec = config["local_inputs"]
    assert spec["enabled"] and spec["inputs"] == ["wheelhouse", "env_output"]
    assert (spec["workers"], spec["piece_mib"]) == (8, 256)


MARLIN_TAIL = '''    if is_mxfp4_marlin:
        if ok:
            moe_topk_sum(intermediate_cache3, output)
        else:
            moe_sum_reduce(intermediate_cache3, output, 1.0)
        return output
    else:
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        moe_sum_reduce(
            intermediate_cache3,
            output,
            routed_scaling_factor,
        )
        return output
'''
LOGITS_INIT = '''        self._logits_gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=triton_symm_mem_ag.recommended_max_tokens(
                include_prefill=False, floor=128
            ),
            enabled=self.do_tensor_parallel_all_gather and not self.use_attn_tp_group,
            skip_entry_sync=True,
        )
'''


def patch_item(name):
    items = yaml.safe_load(CONFIG.read_text())["patches"]["items"]
    return next(item for item in items if item["name"] == name)


def run_edit(tmp_path, name, body):
    """Apply a config edit to the fork's snippet (the hash guard is checked separately)."""
    item = {k: v for k, v in patch_item(name).items() if k not in ("sha256_before", "sha256_after")}
    target = tmp_path / item["file"]
    target.parent.mkdir(parents=True)
    target.write_text(body)
    result = probe.apply_patch(tmp_path, item)
    return result, target.read_text()


def test_the_marlin_moe_sum_becomes_torch_sum_in_the_non_mxfp4_branch_only(tmp_path):
    import ast

    result, text = run_edit(tmp_path, "marlin_moe_torch_sum", MARLIN_TAIL)
    assert result["status"] == "applied" and result["matches"] == 1
    ast.parse("def f():\n" + text)
    assert "        torch.sum(intermediate_cache3, dim=1, out=output)\n" in text
    assert "        if routed_scaling_factor != 1.0:\n            output.mul_(" in text
    # The mxfp4 branch's own call stays: it is not this model's path.
    assert text.count("moe_sum_reduce(") == 1
    assert "moe_sum_reduce(intermediate_cache3, output, 1.0)" in text


def test_the_torch_sum_equals_the_kernels_reduce():
    torch = pytest.importorskip("torch")  # not in the dev venv; the GPU image has it

    x = torch.randn(5, 8, 16, dtype=torch.bfloat16)
    for scale in (1.0, 2.5):
        out = torch.empty(5, 16, dtype=torch.bfloat16)
        torch.sum(x, dim=1, out=out)
        if scale != 1.0:
            out.mul_(scale)
        want = (x.float().sum(dim=1) * scale).to(torch.bfloat16)
        assert torch.allclose(out.float(), want.float(), rtol=1e-2, atol=1e-2)


def test_the_logits_gather_stays_on_nccl(tmp_path):
    result, text = run_edit(tmp_path, "logits_gather_nccl", LOGITS_INIT)
    assert result["status"] == "applied" and "enabled=False," in text
    assert "use_attn_tp_group" not in text


def test_the_edits_are_hash_guarded_and_the_fallback_keeps_every_fix():
    config = yaml.safe_load(CONFIG.read_text())
    items = {i["name"]: i for i in config["patches"]["items"]}
    assert set(items) == {"fa2_scratch_dtype", "logits_gather_nccl", "marlin_moe_torch_sum",
                          "marlin_ep_mask_import", "marlin_ep_mask_map", *KV4_EDITS}
    for name in ("logits_gather_nccl", "marlin_moe_torch_sum"):
        assert len(items[name]["sha256_before"]) == 64 and len(items[name]["sha256_after"]) == 64
        assert items[name]["fallback_flags"] == {}
    assert config["engine"]["flags"]["--disable-custom-all-reduce"] is True
    # The fallback boot differs only in its KV dtype and its graphs.
    retry = config["engine"]["boot_retry_flags"]
    assert set(retry) == {"--kv-cache-dtype", "--cuda-graph-backend-decode",
                          "--cuda-graph-backend-prefill"}
    argv = probe.server_argv(config, "py", {"model": "/m", "port": "1", "chat_template": "/t"},
                             retry)
    assert "--disable-custom-all-reduce" in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "bfloat16"


def test_an_edit_on_another_file_is_refused_by_its_hash(tmp_path):
    item = patch_item("marlin_moe_torch_sum")
    target = tmp_path / item["file"]
    target.parent.mkdir(parents=True)
    target.write_text(MARLIN_TAIL)
    result = probe.apply_patch(tmp_path, item)
    assert result["status"] == "sha256 differs" and target.read_text() == MARLIN_TAIL


# The fork's moe_runner/marlin.py at d00d88e, the lines the EP mask edits anchor on (cut).
MARLIN_RUNNER = """\
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    output = fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        topk_ids=topk_output.topk_ids,
        global_num_experts=quant_info.global_num_experts,
        expert_map=quant_info.expert_map,
        g_idx1=quant_info.w13_g_idx,
    )
"""
FORK = Path("/tmp/fnsg/penny/python")


def test_the_ep_mask_edits_chain_and_pass_an_empty_map_only_at_ep_above_1(tmp_path):
    import ast

    first, second = patch_item("marlin_ep_mask_import"), patch_item("marlin_ep_mask_map")
    assert first["file"] == second["file"] == "sglang/srt/layers/moe/moe_runner/marlin.py"
    assert first["sha256_after"] == second["sha256_before"]
    assert first["fallback_flags"] == second["fallback_flags"] == {"--expert-parallel-size": 1}
    target = tmp_path / first["file"]
    target.parent.mkdir(parents=True)
    target.write_text(MARLIN_RUNNER)
    for item in (first, second):
        unguarded = {k: v for k, v in item.items() if k not in ("sha256_before", "sha256_after")}
        assert probe.apply_patch(tmp_path, unguarded)["status"] == "applied"
    text = target.read_text()
    ast.parse("def f():\n" + text)
    assert "    from sglang.srt.runtime_context import get_parallel\n" in text
    assert ("expert_map=(quant_info.expert_map if quant_info.expert_map is not None or "
            "get_parallel().moe_ep_size == 1 else topk_output.topk_ids.new_empty(0)),") in text
    # Applied at the install, so the fallback boot (BF16, no graphs) carries it too.
    config = yaml.safe_load(CONFIG.read_text())
    assert set(config["engine"]["boot_retry_flags"]).isdisjoint({"--expert-parallel-size"})


def test_the_ep_mask_edits_apply_to_the_forks_own_file(tmp_path):
    first, second = patch_item("marlin_ep_mask_import"), patch_item("marlin_ep_mask_map")
    source = FORK / first["file"]
    if not source.is_file():
        pytest.skip("no local copy of the fork at d00d88e")
    target = tmp_path / first["file"]
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    results = [probe.apply_patch(tmp_path, item) for item in (first, second)]
    assert [r["status"] for r in results] == ["applied", "applied"]
    import hashlib

    assert hashlib.sha256(target.read_bytes()).hexdigest() == second["sha256_after"]


def test_competition_keeps_valid_placeholders_when_every_boot_fails(tmp_path):
    """Tonight's submission may go in without a GPU check: a boot that never serves leaves the
    notebook's placeholders, valid for every task, and the run ends without a solve."""

    def edit(smoke):
        smoke["mode"] = "competition"
        # The fake's argparse refuses the value at its start, in both boots.
        smoke["engine"]["flags"]["--fake-pool"] = "not-a-number"
        smoke["engine"]["boot_retry_flags"] = {"--kv-cache-dtype": "bfloat16"}

    code, out = run_smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 1 and state["boot"]["status"] == "failed" and "solve" not in state
    assert rows(out, "restarts.jsonl")[0]["reason"].startswith("boot:")
    from arc26.submit import validate

    submission = json.loads((out.parent.parent / "submission.json").read_text())
    challenges = json.loads(next((out.parent.parent.parent / "input").rglob(
        "arc-agi_test_challenges.json")).read_text())
    assert validate(submission, challenges) == []
    assert all(entry[0]["attempt_1"] == [[0]] for entry in submission.values())


# Revision 14's memory tuning, which competition gets too (the revision's point).
# Revision 15's values (fn_l4_memory.md, 3d82150) for revision 14's memory flags, plus the chunk.
# Revision 17: 4 streams (24 Mamba slots, graphs 1 to 4) after version 14 read Good.
# Revision 20: 5 streams (29 Mamba slots, graphs 1 to 5), the decode-speed note (f270a86).
# Revision 21: 6 server slots for the 5 streams (30 Mamba slots, the fork's floor 5 x 6; graphs
# still 1 to 5), the N+1 admission fix (fn_many_streams.md section 3.4).
REVISION_14_ENGINE = {"--max-mamba-cache-size": 30, "--mem-fraction-static": 0.95,
                      "--max-running-requests": 6, "--cuda-graph-max-bs-decode": 5,
                      "--cuda-graph-bs-decode": [1, 2, 3, 4, 5], "--chunked-prefill-size": 2048}
# Revision 22's keys (P5.6-01's forensics): the health probe and the boot stress check.
REVISION_22_KEYS = {"health.probe_timeout_s", "health.kill_probes", "stress", "engine.free_pattern",
                    "smoke.health.probe_timeout_s", "smoke.health.kill_probes", "smoke.stress"}
# Revision 17's stream keys, which differ from revisions 15 and 16 (3 streams).
REVISION_17_STREAMS = {"engine.flags.--max-mamba-cache-size", "engine.flags.--max-running-requests",
                       "engine.flags.--cuda-graph-max-bs-decode",
                       "engine.flags.--cuda-graph-bs-decode", "solve.max_streams"}
REVISION_17_KEYS = {"engine.env.ARC26_KV4", "engine.env.ARC26_KV4_HADAMARD",
                    "engine.env.ARC26_KV4_GROUP", "engine.env.PYTHONPATH", "engine.boot_ladder",
                    "patches.items", "kernel.code_dataset", "inputs.code.path",
                    "inputs.code.hint", "tool.loop.reply_chars", *REVISION_17_STREAMS,
                    # Revision 20: the KV4 group's snapshot boundary and the import check.
                    "patches.kv4_from", "import_check", "smoke.import_check",
                    # Revision 20: the cap of 49,152 (the budget note).
                    "solve.cap",
                    # Revision 22: the health probe tied to the engine instance and the boot
                    # stress check (P5.6-01's forensics).
                    *REVISION_22_KEYS}
REVISION_10_DIFFERS_ONLY_IN = {"kernel.timeout_s", "solve.task_ids", "solve.dev_arms",
                               # Revision 15: the floor and the streams, both modes; the
                               # audit's competition window, health, first generation.
                               "planner.min_thinking_tokens", "solve.max_streams",
                               "time_plan.competition.generation_stop_s",
                               "time_plan.competition.end_s", "health.failures",
                               "health.max_restarts", "engine.first_generation_timeout_s",
                               "smoke.solve.dev_arms",
                               *(f"engine.flags.{flag}" for flag in REVISION_14_ENGINE),
                               "time_plan.development.end_s",
                               "time_plan.development.generation_stop_s",
                               "time_plan.development.min_start_s",
                               # Revision 17: the KV4 cache, its variables and ladder.
                               *REVISION_17_KEYS}


def test_revisions_11_to_15_change_competition_only_by_the_memory_tuning_and_the_plan():
    """P5.2-01 revisions 11 to 15 change development keys, the push's timeout, (revision 14)
    the five memory flags, and (revision 15) the planner's floor and the stream limit, which
    competition gets too; nothing else."""
    config = yaml.safe_load(CONFIG.read_text())
    # Revision 15: 60 minutes before the first cell (was 45), the end 450 s after the stop.
    assert config["time_plan"]["competition"] == {"generation_stop_s": 38700, "end_s": 39150,
                                                  "min_start_s": 300}
    assert 43_200 - 3_600 - 450 == 39_150
    assert (config["health"]["failures"], config["health"]["max_restarts"],
            config["health"]["min_left_s"]) == (10, 6, 2400)
    assert config["engine"]["first_generation_timeout_s"] == 1800
    # Revision 20's cap (the budget note, b54410b); the forcing turn stays 6,144.
    assert config["solve"]["cap"] == 49152 and config["solve"]["forcing_tokens"] == 6144
    assert config["solve"]["force_reserve_s"] == 360 and config["engine"]["boot_timeout_s"] == 2700
    # Revision 16's five first; revision 17's KV4 edits after them.
    assert [i["name"] for i in config["patches"]["items"]][:5] == [
        "fa2_scratch_dtype", "logits_gather_nccl", "marlin_moe_torch_sum",
        "marlin_ep_mask_import", "marlin_ep_mask_map"]
    # Revision 16: the engine test in development, competition's arm, inside 7,500 s.
    dev = config["time_plan"]["development"]
    assert dev == {"generation_stop_s": 6600, "end_s": 6900, "min_start_s": 300}
    assert config["solve"]["dev_arms"] == ["tool"] == [config["solve"]["arm"]]
    assert dev["end_s"] + 120 + 480 == config["kernel"]["timeout_s"] == 7500
    flags = config["engine"]["flags"]
    assert {flag: flags[flag] for flag in REVISION_14_ENGINE} == REVISION_14_ENGINE
    assert config["planner"]["min_thinking_tokens"] == 29000
    # Revision 21: one server slot above the client's streams (the N+1 admission fix).
    assert config["solve"]["max_streams"] == flags["--max-running-requests"] - 1 == 5
    assert config["solve"]["context_tokens"] == flags["--context-length"] == 86016
    # extra_buffer: the fork's 5 Mamba slots per server slot (revisions 14 to 20 added a spare of
    # 4; revision 21's 6th server slot, never used by the 5 streams, leaves 5 free instead).
    assert flags["--max-mamba-cache-size"] == 5 * flags["--max-running-requests"]
    # The wheel's cap max_mamba_cache_size // 5 keeps the 6 server slots.
    assert flags["--max-mamba-cache-size"] // 5 == flags["--max-running-requests"]
    assert flags["--cuda-graph-bs-decode"][-1] == config["solve"]["max_streams"]
    # The fallback boot keeps them (it changes only the KV dtype and the graphs).
    assert set(config["engine"]["boot_retry_flags"]) == {
        "--kv-cache-dtype", "--cuda-graph-backend-decode", "--cuda-graph-backend-prefill"}
    # Competition reads its own plan, never the development one.
    assert probe.time_plan(config, probe.COMPETITION)["end_s"] == 39150
    # The key-by-key diff against revision 10 (ce66204), when the history is at hand.
    import subprocess

    old = subprocess.run(["git", "show", "ce66204:configs/P5.2-01.yaml"], cwd=REPO,
                         capture_output=True, text=True)
    if old.returncode != 0:
        pytest.skip("revision 10's commit is not in this clone")

    def diff(a, b, path=""):
        keys = set(a) | set(b)
        out = set()
        for key in keys:
            if a.get(key) == b.get(key):
                continue
            if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
                out |= diff(a[key], b[key], f"{path}{key}.")
            else:
                out.add(f"{path}{key}")
        return out

    assert diff(yaml.safe_load(old.stdout), config) <= REVISION_10_DIFFERS_ONLY_IN


def test_a_generation_stop_already_passed_skips_the_boot_and_keeps_the_placeholders(tmp_path):
    """Revision 12's plan: a development generation stop already passed: every stage before the
    boot runs, both boots end at once, no template check or solve, development exits 1, and the
    notebook's own placeholders stay (cell 3 checks them)."""

    def edit(smoke):
        smoke["time_plan"]["development"] = {"generation_stop_s": 0, "end_s": 60,
                                             "min_start_s": 1}
        smoke["engine"]["boot_retry_flags"] = {"--kv-cache-dtype": "bfloat16"}

    began = time.monotonic()
    code, out = run_smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 1 and time.monotonic() - began < 40
    for stage in ("inputs", "python", "cuda", "tasks", "tool_self_test", "arg_check"):
        assert state[stage]["status"] == "ok", (stage, state[stage])
    assert state["boot"]["status"] == "failed" and "after 0 s" in state["boot"]["error"]
    assert "template_check" not in state and "solve" not in state
    assert rows(out, "attempts.jsonl") == []


def test_a_plain_only_smoke_runs_one_arm_and_writes_its_logs(tmp_path):
    """The development run with dev_arms [plain] and one task: one plain attempt, no tool
    self-test, the engine's tail and the GPU monitor's samples written before the end."""

    def edit(smoke):
        smoke["solve"]["dev_arms"] = ["plain"]
        smoke["solve"]["task_ids"] = ["s0000001"]

    code, out = run_smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and "tool_self_test" not in state and state["solve"]["status"] == "ok"
    assert [(a["task_id"], a["arm"]) for a in rows(out)] == [("s0000001", "plain")]
    for name in ("engine_tail.log", "engine.log", "boot.json", "memory.jsonl", "summary.md"):
        assert (out / name).is_file(), name


def test_the_planner_never_plans_past_the_pool_on_4_x_l4():
    """Revision 14: the smoke's 15,808-token pool, the tuned pool's estimate, the fallback's half.

    No plan holds more than the pool; a small pool gives fewer streams and a shorter context;
    below the floor it keeps one stream per arm at the pool's own size (recorded as not met), and
    a task whose opening leaves no room is refused before sending (its placeholders stay)."""
    config = yaml.safe_load(CONFIG.read_text())
    solve, planner = config["solve"], config["planner"]
    openings = [1457, 2017, 6896]
    for pool, streams, met in ((128_000, 3, True), (64_000, 1, True), (15_808, 1, False)):
        plan = probe.plan_budget(pool, 4, 30.0, 39_000.0, 240, openings, 1, solve, planner)
        assert plan["streams"] == streams and plan["met"] is met, (pool, plan)
        assert plan["streams"] * plan["context"] <= pool
        for opening in openings:
            assert opening + probe.task_cap(plan["context"], opening, solve) <= plan["context"]
    # Two arms (development) at the smoke's pool: one stream each, still within the pool.
    plan = probe.plan_budget(15_808, 4, 30.0, 300.0, 1, openings, 2, solve, planner)
    assert plan["streams"] == 2 and 2 * plan["context"] <= 15_808 and plan["met"] is False


def test_revision_15s_plans_at_the_estimated_pools_and_their_thresholds():
    """The memory note's pools (fn_l4_memory.md, 3d82150) at the floor 29,000 and 3 streams at
    most, with the visible test file's largest opening (c >= 29,000 + 6,896 + 6,144 + 128 =
    42,168 per stream)."""
    config = yaml.safe_load(CONFIG.read_text())
    spec, planner = config["solve"], config["planner"]
    plan = lambda pool: probe.plan_budget(pool, 3, 30.0, 39_000.0, 240, OPENINGS, 1,  # noqa: E731
                                          spec, planner)
    # The FP8 estimate, the BF16 fallback's, the settings note's 70k.
    assert (plan(169_728)["streams"], plan(169_728)["context"]) == (3, 56_576)
    assert plan(169_728)["thinking"] == 43_408
    assert (plan(89_856)["streams"], plan(89_856)["context"]) == (2, 44_928)
    assert (plan(70_000)["streams"], plan(70_000)["context"]) == (1, 70_000)
    assert plan(70_000)["met"] and 70_000 // 2 - 6_896 - 6_144 - 128 == 21_832 < 29_000
    # The thresholds: 3 streams from 126,504, 2 from 84,336.
    assert (plan(126_504)["streams"], plan(126_503)["streams"]) == (3, 2)
    assert (plan(84_336)["streams"], plan(84_335)["streams"]) == (2, 1)
    # The context reaches the engine's 86,016 only from 3 x 86,016 = 258,048.
    assert plan(258_047)["context"] < 86_016 == plan(258_048)["context"]
    # The server's limit binds: never more streams than --max-running-requests.
    # The limit passed binds (revision 15's 3 here; revision 17 passes 4, its own test).
    assert plan(1_000_000)["streams"] == 3
    assert config["engine"]["flags"]["--max-mamba-cache-size"] == 5 * 6


def test_revision_16_changes_only_development_keys_against_revision_15():
    """The engine test keeps every competition key of revision 15 (58f53a8), the submitted one."""
    import subprocess

    old = subprocess.run(["git", "show", "58f53a8:configs/P5.2-01.yaml"], cwd=REPO,
                         capture_output=True, text=True)
    if old.returncode != 0:
        pytest.skip("revision 15's commit is not in this clone")
    before, after = yaml.safe_load(old.stdout), yaml.safe_load(CONFIG.read_text())

    def diff(a, b, path=""):
        out = set()
        for key in set(a) | set(b):
            if a.get(key) == b.get(key):
                continue
            if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
                out |= diff(a[key], b[key], f"{path}{key}.")
            else:
                out.add(f"{path}{key}")
        return out

    assert diff(before, after) == {"kernel.timeout_s", "solve.dev_arms", "solve.task_ids",
                                   "time_plan.development.generation_stop_s",
                                   "time_plan.development.end_s",
                                   "time_plan.development.min_start_s", *REVISION_17_KEYS}
    assert before["time_plan"]["competition"] == after["time_plan"]["competition"]
    assert before["planner"] == after["planner"]


# -- revision 17: the KV4 cache, its boot ladder, its code dataset zip ------------------------

REVISION_16 = "068a1fc"
KV4_EDITS = ["kv4_backend_scratch_dim", "kv4_backend_extract", "kv4_backend_extend_gather",
             "kv4_backend_helpers", "kv4_pool_class_helper", "kv4_pool_class",
             "kv4_cell_size_helper", "kv4_cell_size", "kv4_hybrid_v_head_dim"]


def _config_at(commit, path="configs/P5.2-01.yaml"):
    import subprocess

    old = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True,
                         text=True)
    if old.returncode != 0:
        pytest.skip(f"{commit} is not in this clone")
    return yaml.safe_load(old.stdout)


def _diff(a, b, path=""):
    out = set()
    for key in set(a) | set(b):
        if a.get(key) == b.get(key):
            continue
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            out |= _diff(a[key], b[key], f"{path}{key}.")
        else:
            out.add(f"{path}{key}")
    return out


def test_revision_17_adds_only_the_kv4_cache_to_revision_16():
    before, after = _config_at(REVISION_16), yaml.safe_load(CONFIG.read_text())
    assert _diff(before, after) == {
        "engine.env.ARC26_KV4", "engine.env.ARC26_KV4_HADAMARD", "engine.env.ARC26_KV4_GROUP",
        "engine.env.PYTHONPATH", "engine.boot_ladder", "patches.items",
        # The KV4 lineage's own code dataset (one per concurrent build lineage).
        "kernel.code_dataset", "inputs.code.path", "inputs.code.hint",
        # 4 streams after version 14 read Good; the tool replies' clip at 4,000.
        *REVISION_17_STREAMS, "tool.loop.reply_chars",
        # Revision 20: the snapshot boundary and the import check.
        "patches.kv4_from", "import_check", "smoke.import_check", "solve.cap",
        # Revision 22: the health probe and the boot stress check.
        *REVISION_22_KEYS}
    # Revision 21: the wheel changes (P5.8), so a new slug; version 16 keeps arc26-code-fn-kv4.
    # Revision 22: P5.8b and the stress check, a new slug again; version 17 keeps kv4b.
    assert after["kernel"]["code_dataset"] == "alyahya737/arc26-code-fn-kv4c"
    assert after["inputs"]["code"]["hint"] == ["arc26-code-fn-kv4c", "arc26-code-fn"]
    old_items, items = before["patches"]["items"], after["patches"]["items"]
    # Revision 16's five, the FA2 one with its after-hash the KV4 chain starts from.
    assert items[:5] == [{**old_items[0], "sha256_after": items[0]["sha256_after"]},
                         *old_items[1:]]
    assert [i["name"] for i in items[5:]] == KV4_EDITS
    kvab = yaml.safe_load((REPO / "configs" / "P5.5-01.yaml").read_text())["patches"]["items"]
    assert [{k: v for k, v in i.items() if k != "fallback_env"} for i in items] == kvab
    assert all(i["fallback_env"] == {"ARC26_KV4": "0"} for i in items[5:])
    # The chain inside each file: each edit starts from the previous one's hash.
    last = {}
    for item in items:
        if item["file"] in last:
            assert item["sha256_before"] == last[item["file"]], item["name"]
        last[item["file"]] = item.get("sha256_after")
    env = after["engine"]["env"]
    assert (env["ARC26_KV4"], env["ARC26_KV4_HADAMARD"], env["ARC26_KV4_GROUP"]) == ("1", "1", "32")
    assert env["PYTHONPATH"] == "{code}/fn_kvq.pyz"
    # The flags, the planner, both time plans are revision 16's (the switch is the variable).
    # The flags but the four stream ones, the planner, both time plans are revision 16's (the
    # cache's switch is the variable).
    streams = {k.split("engine.flags.", 1)[1] for k in REVISION_17_STREAMS if "flags" in k}
    assert ({k: v for k, v in before["engine"]["flags"].items() if k not in streams}
            == {k: v for k, v in after["engine"]["flags"].items() if k not in streams})
    assert before["planner"] == after["planner"] and before["time_plan"] == after["time_plan"]


def test_the_boot_ladder_falls_back_to_fp8_then_bf16_eager():
    config = yaml.safe_load(CONFIG.read_text())
    assert probe.boot_rungs(config) == [
        {"flags": {}, "env": {"ARC26_KV4": "0"}},
        {"flags": config["engine"]["boot_retry_flags"], "env": {}}]
    # Without a ladder, the one rung is boot_retry_flags (revisions up to 16).
    old = {"engine": {"boot_retry_flags": {"--kv-cache-dtype": "bfloat16"}}}
    assert probe.boot_rungs(old) == [{"flags": {"--kv-cache-dtype": "bfloat16"}, "env": {}}]
    assert probe.boot_rungs({"engine": {"boot_retry_flags": {}}}) == []
    run = probe.Run(config=config, root=Path("/"), out=Path("/nonexistent"), start=0.0)
    run.out = Path(__import__("tempfile").mkdtemp())
    assert probe.next_rung(run, "x")["env"] == {"ARC26_KV4": "0"}
    assert probe.next_rung(run, "y")["flags"]["--kv-cache-dtype"] == "bfloat16"
    assert probe.next_rung(run, "z") is None
    # Cumulative: the second rung keeps KV4 off.
    assert run.state["boot_env"] == {"ARC26_KV4": "0"}
    assert run.state["flag_overrides"]["--cuda-graph-backend-decode"] == "disabled"
    assert [r["rung"] for r in rows(run.out, "restarts.jsonl")] == [1, 2]


def test_a_kv4_boot_fault_boots_again_without_kv4_and_reads_its_pool(tmp_path):
    """The smoke with the fake refusing ARC26_KV4=1: the first boot fails, the ladder's first
    rung (ARC26_KV4=0) serves, the plan reads that boot's pool, and every task is filled."""

    def edit(smoke):
        smoke["mode"] = "competition"
        smoke["engine"]["flags"]["--fail-if-env"] = "ARC26_KV4=1"
        smoke["engine"]["boot_ladder"] = [{"env": {"ARC26_KV4": "0"}},
                                          {"flags": {"--kv-cache-dtype": "bfloat16"}}]
        smoke["engine"]["env"] = {"ARC26_KV4": "1"}

    code, out = run_smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["boot"]["status"] == "ok" and state["solve"]["status"] == "ok"
    restarts = rows(out, "restarts.jsonl")
    assert [(r["rung"], r["env"]) for r in restarts] == [(1, {"ARC26_KV4": "0"})]
    commands = rows(out, "engine_commands.jsonl")
    assert [c["env"] for c in commands] == [{}, {"ARC26_KV4": "0"}]
    boot = json.loads((out / "boot.json").read_text())
    assert boot["pool_tokens"] == 20000 and state["solve"]["submission_filled"] == 6
    assert "refusing to start with ARC26_KV4=1" in (out / "engine.log").read_text()


def test_a_failed_kv4_edit_turns_kv4_off_for_every_boot(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    run = probe.Run(config=config, root=tmp_path, out=tmp_path / "out", start=0.0)
    run.out.mkdir()
    (tmp_path / "venv" / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    results = {r["name"]: r for r in probe.apply_patches(run, tmp_path / "venv")}
    assert results["kv4_pool_class"]["status"] == "missing"
    assert results["kv4_pool_class"]["fallback_env"] == {"ARC26_KV4": "0"}
    assert run.state["env_fallback"] == {"ARC26_KV4": "0"}
    run.state["paths"] = {"code": "/kaggle/input/arc26-code-fn"}
    probe.stage_cuda(run)
    env = run.state["env_overrides"]
    assert env["ARC26_KV4"] == "0" and env["PYTHONPATH"] == "/kaggle/input/arc26-code-fn/fn_kvq.pyz"


def test_the_code_dataset_carries_fn_kvq_as_one_importable_zip(tmp_path):
    import subprocess
    import zipfile

    nb = load("build_fn_notebooks")
    first = nb.package_zip(REPO, "scripts/fn_kvq", "HEAD")
    assert first == nb.package_zip(REPO, "scripts/fn_kvq", "HEAD")
    names = zipfile.ZipFile(__import__("io").BytesIO(first)).namelist()
    assert names == sorted(f"fn_kvq/{p.name}" for p in (REPO / "scripts/fn_kvq").glob("*.py"))
    listing = json.loads(nb.code_dataset("alyahya737/arc26-code-fn", {"fn_kvq.pyz": first,
                                                                      "fn_probe.py": b"x"})[
        nb.CODE_MARKER])
    assert "fn_kvq.pyz" in json.dumps(listing)
    # No name Kaggle would extract (revision 17's fn_kvq.zip arrived as a folder).
    assert all(not name.lower().endswith(nb.KAGGLE_EXTRACTS) for name in nb.DATASET_ZIPS)
    # Importable from the zip alone, with its sources readable (Triton's JIT reads them).
    path = tmp_path / "fn_kvq.pyz"
    path.write_bytes(first)
    code = ("import inspect, fn_kvq.layout as m; print(m.__file__); "
            "print(len(inspect.getsource(m.bytes_per_token)) > 0)")
    done = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(path)}, check=False)
    assert done.returncode == 0, done.stderr
    assert "fn_kvq.pyz" in done.stdout and done.stdout.strip().endswith("True")


def test_revision_21s_plans_with_kv4_and_on_the_fp8_fallback():
    """5 streams at the KV4 pool's estimate, 3 on the FP8 fallback, with 6 server slots and 30
    Mamba slots (revision 20's: 242,876 and 148,896 with 29 slots; revision 17's: 260,286 and
    159,536 with 24)."""
    config = yaml.safe_load(CONFIG.read_text())
    spec, planner = config["solve"], config["planner"]
    flags = config["engine"]["flags"]
    # The N+1 admission fix: one server slot above the client's streams.
    assert spec["max_streams"] == 5 and flags["--max-running-requests"] == 6
    # The fork's floor 5 x 6, no spare; its cap max_mamba_cache_size // 5 gives 6.
    assert flags["--max-mamba-cache-size"] == 5 * 6 == 30
    assert flags["--mem-fraction-static"] == 0.95
    assert flags["--cuda-graph-bs-decode"] == [1, 2, 3, 4, 5] == list(
        range(1, flags["--cuda-graph-max-bs-decode"] + 1))
    # Version 16 locked 4 slots per running request (20 at 5): 10 of 30 stay for cached states.
    assert flags["--max-mamba-cache-size"] - 4 * spec["max_streams"] == 10 > 29 - 20
    plan = lambda pool, limit=6: probe.plan_budget(  # noqa: E731
        pool, limit, 36.8, 39_000.0, 240, OPENINGS, 1, spec, planner)
    # KV4: 277,696 (19 slots) less 11 more Mamba slots of 3,482 KV4 tokens (estimate).
    kv4 = 277_696 - 11 * 3_482
    assert kv4 == 242_876 - 3_482 == 239_394
    assert (plan(kv4)["streams"], plan(kv4)["context"], plan(kv4)["thinking"]) == (5, 47_878,
                                                                                   34_710)
    # From version 16's measured 243,712 at 29 slots instead: 240,230.
    measured = 243_712 - 3_482
    assert (plan(measured)["streams"], plan(measured)["context"]) == (5, 48_046)
    # The server's 6 never gives a 6th stream: the planner takes solve.max_streams.
    assert plan(10**7)["streams"] == 5
    assert {k: v for k, v in plan(kv4, 5).items() if k != "limit"} == {
        k: v for k, v in plan(kv4).items() if k != "limit"}
    # The FP8 fallback: version 14's measured 170,176 (19 slots) less 11 slots of 2,128.
    fp8 = 170_176 - 11 * 2_128
    assert fp8 == 148_896 - 2_128 == 146_768
    assert (plan(fp8)["streams"], plan(fp8)["context"], plan(fp8)["thinking"]) == (3, 48_922,
                                                                                   35_754)
    assert plan(fp8)["met"]
    # 4 streams there would think 36,692 - 13,168 = 23,524, under the floor.
    assert fp8 // 4 - 6_896 - 6_144 - 128 == 23_524 < 29_000
    # 5 streams from 5 x 42,168.
    assert (plan(210_840)["streams"], plan(210_839)["streams"]) == (5, 4)
    for pool in (kv4, fp8, 15_808):
        assert plan(pool)["streams"] * plan(pool)["context"] <= pool


def test_revision_17s_tool_replies_are_clipped_at_4000_characters():
    """Version 14 clipped 11 of 93 tool replies at P5.3's default 2,000 characters."""
    from arc26.agent.toolshot import tool_config

    config = yaml.safe_load(CONFIG.read_text())
    assert config["tool"]["loop"] == {"reply_chars": 4000}
    assert tool_config(config["tool"]["loop"]).reply_chars == 4000
    assert tool_config({}).reply_chars == 2000


# -- revision 20: the KV4 snapshot, the restore, the import check ----------------------------


def _kv4_site(tmp_path, config, broken=None):
    """A venv whose site holds every edited file at the fork's bytes (the wheel peek)."""
    peek = REPO / "runs" / "wheel-peek" / "run_logs" / "peek"
    if not (peek / "peek.json").is_file():
        pytest.skip("no wheel peek in this clone")
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    for path in peek.glob("*.txt"):
        target = site / path.name[: -len(".txt")].replace("__", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    if broken:
        for item in config["patches"]["items"]:
            if item["name"] == broken:
                item["pattern"] = "this anchor is not in the file"
    return site


def test_a_broken_kv4_edit_restores_revision_16s_files_and_turns_kv4_off(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["paths"]["scratch"] = str(tmp_path / "scratch")
    site = _kv4_site(tmp_path, config, broken="kv4_cell_size")
    run = probe.Run(config=config, root=tmp_path, out=tmp_path / "out", start=0.0)
    run.out.mkdir()
    results = {r["name"]: r["status"] for r in probe.apply_patches(run, tmp_path / "venv")}
    assert results["kv4_cell_size"] == "match count"
    assert all(results[n] == "applied" for n in results if n != "kv4_cell_size")
    # Every file the KV4 group edits is back at the snapshot: revision 16's five edits only.
    snap = run.state["kv4_snapshot"]
    assert set(snap["files"]) == {i["file"] for i in config["patches"]["items"][5:]}
    for rel, digest in snap["files"].items():
        assert probe.sha256_file(site / rel) == digest
    fa2 = next(i for i in config["patches"]["items"] if i["name"] == "fa2_scratch_dtype")
    assert snap["files"][fa2["file"]] == fa2["sha256_after"]
    assert run.state["env_fallback"]["ARC26_KV4"] == "0"
    restores = rows(run.out, "kv4_restore.jsonl")
    assert len(restores) == 1 and all(restores[0]["files"].values())


def test_the_ladders_kv4_off_rung_restores_the_snapshot(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["paths"]["scratch"] = str(tmp_path / "scratch")
    site = _kv4_site(tmp_path, config)
    run = probe.Run(config=config, root=tmp_path, out=tmp_path / "out", start=0.0)
    run.out.mkdir()
    probe.apply_patches(run, tmp_path / "venv")
    edited = site / "sglang/srt/mem_cache/kv_cache_configurator.py"
    assert "_arc26_kv4_pool_class" in edited.read_text()
    rung = probe.next_rung(run, "the KV4 boot failed")
    assert rung["env"] == {"ARC26_KV4": "0"}
    assert "_arc26_kv4_pool_class" not in edited.read_text()
    assert run.state["boot_env"]["ARC26_KV4"] == "0"


def test_a_failed_import_check_restores_and_the_fp8_boot_serves(tmp_path):
    """On the fake path: the KV4 module fails to import, the check restores and turns KV4 off,
    the import of the engine's own modules then passes, and the boot (which the fake refuses
    with ARC26_KV4=1) serves."""

    def edit(smoke):
        smoke["mode"] = "competition"
        smoke["engine"]["flags"]["--fail-if-env"] = "ARC26_KV4=1"
        smoke["engine"]["env"] = {"ARC26_KV4": "1"}
        smoke["import_check"] = {"enabled": True, "timeout_s": 60, "modules": ["fn_fake_sglang"],
                                 "kv4_modules": ["arc26_no_such_kv4_module"]}

    code, out = run_smoke(tmp_path, edit)
    state = json.loads((out / "state.json").read_text())
    assert code == 0 and state["import_check"]["status"] == "ok"
    assert state["import_check"]["fell_back"] is True and state["import_check"]["kv4_after"] == "0"
    check = json.loads((out / "import_check.json").read_text())
    assert check["check"]["failed"] == ["arc26_no_such_kv4_module"]
    assert check["after_restore"]["failed"] == []
    assert state["boot"]["status"] == "ok" and not rows(out, "restarts.jsonl")
    assert [c["env"] for c in rows(out, "engine_commands.jsonl")] == [{"ARC26_KV4": "0"}]
    assert state["solve"]["submission_filled"] == 6


def test_the_import_check_reads_every_edited_module():
    config = yaml.safe_load(CONFIG.read_text())
    edited = {i["file"].removesuffix(".py").replace("/", ".") for i in config["patches"]["items"]}
    assert set(config["import_check"]["modules"]) == edited
    assert config["import_check"]["kv4_modules"] == ["fn_kvq.sglang_kv4"]
    assert config["patches"]["kv4_from"] == "kv4_backend_scratch_dim"


def test_revision_21s_cap_and_context_at_both_plans():
    """The cap 49,152 (docs/experiments/P5.2-01-r16-budget.md, b54410b) with 5 streams at
    revision 21's pools: the context binds under it at both plans, for every opening."""
    config = yaml.safe_load(CONFIG.read_text())
    spec, planner = config["solve"], config["planner"]
    assert spec["cap"] == 49_152 and spec["forcing_tokens"] == 6_144
    plan = lambda pool: probe.plan_budget(pool, 6, 36.8, 39_000.0, 240, OPENINGS, 1,  # noqa: E731
                                          spec, planner)
    kv4, fp8 = plan(239_394), plan(146_768)
    assert (kv4["streams"], kv4["context"]) == (5, 47_878)
    assert (fp8["streams"], fp8["context"]) == (3, 48_922)
    for opening in (0, 1_782, 6_896):
        assert probe.task_cap(kv4["context"], opening, spec) == 47_878 - opening - 128 < 49_152
        assert probe.task_cap(fp8["context"], opening, spec) == 48_922 - opening - 128 < 49_152
    # The forcing turn stays inside every cap; thinking stops 6,144 short of it.
    assert probe.task_cap(kv4["context"], 6_896, spec) - spec["forcing_tokens"] == 34_710
