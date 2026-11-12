"""Checks the agent spike (task P6.3): config, its pieces, and the CPU smoke path end to end.

No GPU, no network beyond localhost, no Kaggle. The smoke runs ``scripts/agent_spike.py --smoke``
in a subprocess against its own fake OpenAI-compatible server; the pieces are tested directly:
the token-run detector, the steady-window rate, program extraction, the subprocess checker, the
feedback text, the task sample, the parsers of vLLM's log and metrics, and the input checks.
"""

import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.3-01.yaml"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike")


def smoke(out, *extra, timeout=300):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--smoke", "--out-dir", str(out),
         "--commit", "c0ffee", *extra],
        capture_output=True, text=True, timeout=timeout, cwd=REPO,
    )  # fmt: skip


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    result = smoke(out)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out, result


# --- the smoke path ------------------------------------------------------------------------------


def test_smoke_completes_with_self_describing_outputs(smoke_run):
    out, _ = smoke_run
    for name in (
        "run.json", "config.yaml", "metrics.json", "summary.md", "timing.jsonl",
        "throughput.jsonl", "throughput_cells.json", "probe.jsonl", "boot.json",
        "server_metrics.jsonl", "vllm_server.log", "engine_prep.json",
    ):  # fmt: skip
        assert (out / name).is_file(), name
    # The notebook's own names in run_logs/ are never written by the script.
    for name in spike.NOTEBOOK_LOGS:
        assert not (out / name).exists()
    run = json.loads((out / "run.json").read_text())
    assert run["exit_code"] == 0 and run["commit"] == "c0ffee" and run["smoke"] is True
    assert run["phases"] == ["throughput-8k", "probe", "throughput-32k"]
    assert set(run["stages"]) >= {
        "inputs", "engine_prep", "boot", "sanity", "throughput-8k", "probe"
    }  # fmt: skip
    # The smoke prepares its own interpreter: no jit cache there, its two harmless imports pass.
    prep = json.loads((out / "engine_prep.json").read_text())
    assert prep["jit_cache"]["state"] == "absent" and prep["imports"]["ok"] is True
    resolved = yaml.safe_load((out / "config.yaml").read_text())
    assert resolved["run_id"] == "P6.3-01" and resolved["out_dir"] == str(out)


def test_smoke_measures_every_cell(smoke_run):
    out, _ = smoke_run
    cells = json.loads((out / "throughput_cells.json").read_text())
    assert [(c["context"], c["concurrency"]) for c in cells] == [
        ("8k", 1), ("8k", 4), ("32k", 1), ("32k", 4),
    ]  # fmt: skip
    for cell in cells:
        assert cell["complete"] and cell["errors"] == 0 and cell["degenerate"] == 0
        assert cell["output_tokens"] == 24 * cell["concurrency"]
        assert cell["agg_tps"] > 0 and cell["ttft_p50_s"] is not None
        assert cell["server"]["generation_tokens"] == cell["output_tokens"]
    rows = [json.loads(line) for line in (out / "throughput.jsonl").read_text().splitlines()]
    assert len(rows) == 10 and all(row["finish_reason"] == "length" for row in rows)


def test_smoke_probe_verifies_two_fixtures_one_per_round(smoke_run):
    out, _ = smoke_run
    probe = json.loads((out / "metrics.json").read_text())["probe"]
    assert probe["tasks"] == ["smoke-mirror", "smoke-recolor", "smoke-transpose"]
    assert (probe["verified"], probe["verified_round1"], probe["verified_round2"]) == (2, 1, 1)
    assert probe["test_correct"] == 2 and probe["precision"] == 1.0
    assert probe["no_program_rounds"] == 1 and probe["errors"] == 0
    records = {
        json.loads(line)["task_id"]: json.loads(line)
        for line in (out / "probe.jsonl").read_text().splitlines()
    }
    assert records["smoke-transpose"]["verified_round"] == 2
    assert records["smoke-recolor"]["rounds_used"] == 2 and not records["smoke-recolor"]["verified"]
    assert "zip(*grid)" in records["smoke-transpose"]["rounds"][1]["program"]


def test_smoke_timing_records_follow_the_project_schema(smoke_run):
    out, _ = smoke_run
    records = [json.loads(line) for line in (out / "timing.jsonl").read_text().splitlines()]
    assert all(set(r) == {"puzzle_id", "op", "seconds", "tokens"} for r in records)
    assert records[0]["op"] == "boot"
    assert {r["op"] for r in records[1:]} == {"engine"}
    assert sum(1 for r in records if str(r["puzzle_id"]).startswith("smoke-")) == 5


def test_smoke_rerun_resumes_finished_cells_and_tasks(smoke_run, tmp_path):
    first, _ = smoke_run
    result = smoke(tmp_path, "--resume-from", str(first))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert all(cell.get("resumed") for cell in metrics["throughput"])
    assert metrics["probe"]["verified"] == 2
    assert not (tmp_path / "probe.jsonl").exists()  # nothing new was run


def test_degenerate_engine_fires_the_kill_criterion(tmp_path):
    result = smoke(tmp_path, "--fake-mode", "degenerate")
    assert result.returncode == 1
    run = json.loads((tmp_path / "run.json").read_text())
    assert "sanity request failed" in run["problems"][0]
    assert (tmp_path / "summary.md").is_file() and (tmp_path / "vllm_server.log").is_file()


def test_wall_time_budget_exits_3_with_logs(tmp_path):
    result = smoke(tmp_path, "--max-wall-s", "1")
    assert result.returncode == 3, result.stdout[-2000:]
    run = json.loads((tmp_path / "run.json").read_text())
    assert "wall-time budget" in run["problems"][0]
    assert (tmp_path / "metrics.json").is_file()


def test_relative_paths_still_reach_the_engine(tmp_path):
    # The engine runs in the scratch folder; every path on its command line must be absolute.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--smoke", "--out-dir", "out",
         "--max-wall-s", "120"],
        capture_output=True, text=True, timeout=300, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert json.loads((tmp_path / "out" / "run.json").read_text())["exit_code"] == 0


def test_gpu_run_without_inputs_is_refused(tmp_path, capsys):
    assert spike.main(["--config", str(CONFIG), "--out-dir", str(tmp_path)]) == 2
    assert "--model-root" in capsys.readouterr().err


# --- the config ----------------------------------------------------------------------------------


def test_every_config_value_has_a_comment():
    previous = ""
    for line in CONFIG.read_text().splitlines():
        if re.match(r"\s*[A-Za-z_0-9]+:", line):
            assert "#" in line or previous.lstrip().startswith("#"), line
        if line.strip():
            previous = line


def test_real_config_matches_the_brief():
    config = spike.load_config(CONFIG)
    server, throughput, probe = config["server"], config["throughput"], config["probe"]
    assert (server["tensor_parallel_size"], server["kv_cache_dtype"]) == (4, "fp8")
    assert server["enable_prefix_caching"] is True and server["attention_backend"] == "TRITON_ATTN"
    assert not any("speculative" in arg for arg in server["extra_args"])
    assert throughput["concurrency"] == [1, 8, 16, 32, 64]
    means = {
        name: c["prompt_tokens"] + throughput["output_tokens"] / 2
        for name, c in throughput["contexts"].items()
    }
    assert means == {"8k": 8192, "32k": 32768}
    assert (probe["n_tasks"], probe["rounds"], probe["reasoning_effort"]) == (10, 2, "low")
    assert "training" in probe["challenges"] and "training" in probe["solutions"]
    assert config["phases"] == ["throughput-8k", "probe", "throughput-32k"]
    # The script stops itself before the notebook's hard stop (the script config's 6900 s),
    # which comes before the brief's 7200 s Timeout.
    build = yaml.safe_load((REPO / "configs" / "P6.3-01-script.yaml").read_text())
    assert config["max_wall_s"] + 120 < build["script_kill_s"] <= 7200 - 300


def test_server_command_carries_the_brief_settings():
    config = spike.load_config(CONFIG)
    command = spike.server_command("/v/bin/vllm", Path("/m/qwen"), config["server"])
    assert command[:3] == ["/v/bin/vllm", "serve", "/m/qwen"]
    pairs = dict(zip(command[3::1], command[4::1], strict=False))
    assert pairs["--tensor-parallel-size"] == "4" and pairs["--kv-cache-dtype"] == "fp8"
    assert pairs["--attention-backend"] == "TRITON_ATTN" and pairs["--max-num-seqs"] == "64"
    assert "--enable-prefix-caching" in command and "--language-model-only" in command
    assert not any("speculative" in part for part in command)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["server"].update(extra_args=["--speculative-config", "{}"]), "speculative"),
        (lambda c: c["probe"].update(challenges="arc-agi_evaluation_challenges.json"), "training"),
        (lambda c: c.update(phases=["throughput-4k", "probe"]), "names no context"),
        (lambda c: c.update(phases=["warmup"]), "must be probe"),
        (lambda c: c["throughput"]["contexts"]["32k"].update(prompt_tokens=64000), "max_model_len"),
        (lambda c: c["server"]["env"].update(NCCL_DEBUG=1), "string values"),
        (lambda c: c["probe"].pop("rounds"), "exactly the keys"),
        (lambda c: c.pop("probe"), "exactly the keys"),
    ],
)
def test_config_rules_refuse_bad_values(edit, message):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    with pytest.raises(spike.ConfigError, match=message):
        spike.check_config(raw)


# --- pieces --------------------------------------------------------------------------------------


def test_token_stats_flags_long_runs_and_ignores_special_tokens():
    assert spike.token_stats([5, 5, 5, 7], 248044, 3)["degenerate"] is True
    assert spike.token_stats([5, 5, 7, 5], 248044, 3)["max_run"] == 2
    special = [248046] * 200 + [1, 2]
    assert spike.token_stats(special, 248044, 64)["degenerate"] is False
    nan_like = spike.token_stats([0] * 80, 248044, 64)
    assert nan_like["degenerate"] and nan_like["token0_count"] == 80
    by_text = spike.token_stats([], 248044, 64, "ok " + "!" * 70)
    assert by_text["degenerate"] and by_text["source"] == "characters"
    # After the first end-of-sequence id (ignore_eos keeps generating) nothing counts.
    after_eos = spike.token_stats([1, 2, 248046] + [9] * 100, 248044, 64, eos_ids=[248046])
    assert after_eos["degenerate"] is False and after_eos["eos_at"] == 2
    before_eos = spike.token_stats([9] * 70 + [248046], 248044, 64, eos_ids=[248046])
    assert before_eos["degenerate"] is True


def record(send, first, last, chunks):
    return {"t_send": send, "t_first": first, "t_last": last, "chunks": chunks}


def test_steady_rate_counts_only_while_every_request_decodes():
    a = record(0.0, 1.0, 20.0, [(t, 1) for t in range(1, 21)])
    b = record(0.0, 5.0, 30.0, [(t, 2) for t in range(5, 31)])
    rate = spike.steady_rate([a, b])
    # Window (5, 20]: a has 15 chunks of 1 token, b has 15 chunks of 2.
    assert rate == {"steady_tps": 3.0, "window_s": 15.0, "window_tokens": 45}
    assert spike.steady_rate([a, record(0.0, None, None, [])])["steady_tps"] is None
    assert spike.steady_rate([record(0.0, 1.0, 3.0, [(2.0, 1)])])["steady_tps"] is None


def test_extract_program_takes_the_last_block_that_defines_transform():
    text = (
        "Try:\n```python\ndef transform(g):\n    return g\n```\nBetter:\n"
        "```\nimport numpy as np\ndef transform(grid):\n    return grid[::-1]\n```\n"
        "```python\nprint('no function')\n```"
    )
    assert spike.extract_program(text).startswith("import numpy as np")
    assert spike.extract_program("no code here") is None
    assert spike.extract_program(None) is None


PY = sys.executable


def test_checker_runs_programs_in_a_limited_subprocess():
    numpy_program = "import numpy as np\ndef transform(grid):\n    return np.array(grid).T\n"
    check = spike.check_program(numpy_program, [[[1, 2], [3, 4]]], PY, 20, 2048)
    assert check["results"] == [{"output": [[1, 3], [2, 4]]}] and not check["timeout"]
    raising = "def transform(grid):\n    raise ValueError('boom')\n"
    assert (
        "ValueError: boom"
        in spike.check_program(raising, [[[1]]], PY, 20, 2048)["results"][0]["error"]
    )
    not_grid = "def transform(grid):\n    return [[1, 12]]\n"
    assert spike.check_program(not_grid, [[[1]]], PY, 20, 2048)["results"][0] == {
        "error": "not a grid"
    }
    broken = "def transform(grid)\n    return grid\n"
    assert "SyntaxError" in spike.check_program(broken, [[[1]]], PY, 20, 2048)["load_error"]


def test_checker_kills_a_hung_or_hungry_program():
    loop = "def transform(grid):\n    while True:\n        pass\n"
    assert spike.check_program(loop, [[[1]]], PY, 1.5, 2048)["timeout"] is True
    hungry = "def transform(grid):\n    return [[0] * (10 ** 10)]\n"
    result = spike.check_program(hungry, [[[1]]], PY, 20, 512)
    assert result.get("timeout") or "MemoryError" in result["results"][0]["error"]


TASK = {
    "train": [
        {"input": [[1, 2]], "output": [[2, 1]]},
        {"input": [[3, 4], [5, 6]], "output": [[4, 3], [6, 5]]},
    ],
    "test": [{"input": [[7, 8]]}],
}


def test_judge_needs_every_demonstration_and_checks_the_test_output():
    good = {"results": [{"output": [[2, 1]]}, {"output": [[4, 3], [6, 5]]}, {"output": [[8, 7]]}]}
    assert spike.judge(good, TASK, [[[8, 7]]]) == {
        "passed": 2, "total": 2, "verified": True, "test_correct": True,
    }  # fmt: skip
    wrong_test = spike.judge(good, TASK, [[[7, 8]]])
    assert wrong_test["verified"] and not wrong_test["test_correct"]
    half = {"results": [{"output": [[2, 1]]}, {"error": "x"}, {"output": [[8, 7]]}]}
    assert spike.judge(half, TASK, [[[8, 7]]])["verified"] is False
    assert spike.judge(None, TASK, [[[8, 7]]])["passed"] == 0


def test_feedback_reports_each_example_and_shows_the_first_wrong_grid():
    check = {"results": [{"output": [[2, 1]]}, {"output": [[4, 3], [5, 6]]}, {"output": [[8, 7]]}]}
    text = spike.feedback_text(check, TASK, True)
    assert "Example 1: correct." in text and "Example 2: wrong: 2 of 4 cells." in text
    assert "Expected:\n43\n65\nYour output:\n43\n56" in text
    assert text.endswith("defines transform.")
    assert "no Python code block" in spike.feedback_text(None, TASK, False)
    assert "time limit" in spike.feedback_text({"timeout": True}, TASK, True)
    shape = {"results": [{"output": [[2, 1], [0, 0]]}, {"error": "IndexError: x"}]}
    text = spike.feedback_text(shape, TASK, True)
    assert "your output is 2x2, expected 1x2" in text and "Example 2: error: IndexError" in text


def test_select_tasks_is_a_seeded_sample_of_small_single_test_tasks():
    rng = random.Random(1)
    challenges = {}
    for index in range(60):
        side = rng.randint(1, 30)
        grid = [[0] * side for _ in range(side)]
        tests = [{"input": grid}] * (2 if index % 7 == 0 else 1)
        challenges[f"{index:08x}"] = {"train": [{"input": grid, "output": grid}], "test": tests}
    chosen, eligible = spike.select_tasks(challenges, 5, 0, 1500)
    assert chosen == spike.select_tasks(challenges, 5, 0, 1500)[0] == sorted(chosen)
    assert all(len(challenges[t]["test"]) == 1 for t in chosen)
    assert all(len(spike.task_text(challenges[t])) <= 1500 for t in chosen)
    assert eligible < 60 and chosen != spike.select_tasks(challenges, 5, 1, 1500)[0]
    with pytest.raises(spike.Stop):
        spike.select_tasks(challenges, 59, 0, 1500)


def test_task_text_renders_digit_rows_and_never_test_outputs():
    task = {**TASK, "test": [{"input": [[7, 8]], "output": [[9, 9]]}]}
    text = spike.task_text(task)
    assert (
        "Example 2\nInput (2 rows, 2 columns):\n34\n56\nOutput (2 rows, 2 columns):\n43\n65" in text
    )
    assert "Test input (1 rows, 2 columns):\n78" in text and "99" not in text


def test_synthetic_prompts_land_near_their_target_and_differ():
    for target in (7168, 31744):
        messages = spike.synthetic_messages(random.Random(target), target)
        digits = sum(ch.isdigit() or ch == "\n" for ch in messages[1]["content"])
        assert 0.85 * target < digits < 1.1 * target
    one = spike.synthetic_messages(random.Random(1), 2000)[1]["content"]
    two = spike.synthetic_messages(random.Random(2), 2000)[1]["content"]
    assert one[:200] != two[:200]


BOOT_LOG = """\
INFO 10-10 12:00:01 [loader.py:200] Loading weights took 51.20 seconds
INFO 10-10 12:00:02 [gpu_model_runner.py:3000] Model loading took 4.9154 GiB memory and 55.31 seconds
INFO 10-10 12:00:30 [backends.py:500] torch.compile takes 88.4 s in total
INFO 10-10 12:01:00 [kv_cache_utils.py:2465] GPU KV cache size: 1,234,560 tokens, Maximum concurrency for 65,536 tokens per request: 18.84x
INFO 10-10 12:01:20 [gpu_model_runner.py:4000] Graph capturing finished in 21 secs, took 0.61 GiB
INFO 10-10 12:01:21 [core.py:300] init engine (profile, create kv cache, warmup model) took 140.2 seconds
x:1:2 [0] NCCL INFO NCCL version 2.29.7+cuda13.0
x:1:2 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
x:1:2 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via SHM/direct/direct
ERROR 10-10 12:02:00 [x.py:1] something failed
"""  # noqa: E501


def test_parse_boot_log_reads_vllm_lines():
    boot = spike.parse_boot_log(BOOT_LOG)
    assert boot["kv_cache_tokens_first"] == 1_234_560
    assert boot["max_concurrency_first"] == {"tokens": 65536, "x": 18.84}
    assert boot["weights_load_s"] == ["51.20"] and boot["compile_s"] == ["88.4"]
    assert boot["model_load"] == [("4.9154", "55.31")]
    assert boot["graph_capture"] == [("21", "0.61")] and boot["init_engine_s"] == ["140.2"]
    assert boot["nccl_version"] == ["2.29.7+cuda13.0"]
    assert boot["nccl_transport"] == {"P2P": 1, "SHM": 1}
    assert boot["error_lines"] == 1 and boot["tracebacks"] == 0


METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="m"} 12.0
vllm:num_requests_waiting{engine="0",model_name="m"} 3.0
vllm:generation_tokens_total{engine="0",model_name="m"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 40.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
vllm:num_preemptions_total{engine="0",model_name="m"} 2.0
vllm:time_to_first_token_seconds_bucket{le="0.1"} 5.0
vllm:cache_config_info{block_size="1056",num_gpu_blocks="1170"} 1.0
"""


def test_parse_metrics_keeps_the_counters_of_interest():
    values = spike.parse_metrics(METRICS)
    assert values == {
        "num_requests_running": 12.0, "num_requests_waiting": 3.0, "generation_tokens": 1000.0,
        "prefix_cache_hits": 40.0, "prefix_cache_queries": 100.0, "num_preemptions": 2.0,
    }  # fmt: skip
    assert spike.cache_config_info(METRICS) == {"block_size": "1056", "num_gpu_blocks": "1170"}


def write_manifest(folder, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "MANIFEST.json").write_text(json.dumps(fields))


def test_resolve_model_checks_the_manifest_and_every_size(tmp_path):
    settings = {"folder": "m", "revision": "a" * 40, "verify_hashes": True}
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "config.json").write_text("{}")
    import hashlib

    row = {"name": "config.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
    write_manifest(tmp_path, ok=True, revision="a" * 40, files=[row])
    folder, facts = spike.resolve_model(tmp_path, settings)
    assert folder == tmp_path / "m" and facts["files_checked"] == 1
    write_manifest(tmp_path, ok=True, revision="a" * 40, files=[{**row, "size": 3}])
    with pytest.raises(spike.Stop, match="has 2 bytes"):
        spike.resolve_model(tmp_path, settings)
    write_manifest(tmp_path, ok=True, revision="a" * 40, files=[{**row, "sha256": "0" * 64}])
    with pytest.raises(spike.Stop, match="sha256 differs"):
        spike.resolve_model(tmp_path, settings)
    write_manifest(tmp_path, ok=True, revision="b" * 40, files=[row])
    with pytest.raises(spike.Stop, match="revision"):
        spike.resolve_model(tmp_path, settings)
    write_manifest(tmp_path, ok=False, revision="a" * 40, files=[row])
    with pytest.raises(spike.Stop, match="did not pass"):
        spike.resolve_model(tmp_path, settings)


def test_resolve_wheelhouse_needs_an_ok_manifest_the_engine_and_the_installer(tmp_path):
    settings = {"folder": "wheelhouse", "vllm": "0.31.0"}
    house = tmp_path / "wheelhouse"
    files = [{"file": "vllm-0.31.0-x.whl", "name": "vllm", "version": "0.31.0"}]
    write_manifest(house, ok=True, files=files, install_check={"seconds": 99.0})
    with pytest.raises(spike.Stop, match="install_wheelhouse.py"):
        spike.resolve_wheelhouse(tmp_path, settings)
    (house / "install_wheelhouse.py").write_text("")
    folder, facts = spike.resolve_wheelhouse(tmp_path, settings)
    assert folder == house and facts["cpu_install_check_s"] == 99.0
    write_manifest(house, ok=True, files=[{**files[0], "version": "0.30.0"}])
    with pytest.raises(spike.Stop, match="vllm 0.30.0"):
        spike.resolve_wheelhouse(tmp_path, settings)


def test_resolve_task_files_prefers_the_competition_folder(tmp_path):
    settings = {"competition": "arc-prize-2026-arc-agi-2", "challenges": "c.json",
                "solutions": "s.json"}  # fmt: skip
    for folder in ("competitions/arc-prize-2026-arc-agi-2", "datasets/other"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "c.json").write_text("{}")
    (tmp_path / "competitions/arc-prize-2026-arc-agi-2/s.json").write_text("{}")
    challenges, solutions = spike.resolve_task_files(tmp_path, settings)
    assert challenges.parent.name == solutions.parent.name == "arc-prize-2026-arc-agi-2"
    (tmp_path / "competitions/arc-prize-2026-arc-agi-2/s.json").unlink()
    with pytest.raises(spike.Stop, match="0 files named s.json"):
        spike.resolve_task_files(tmp_path, settings)


def test_engine_environment_drops_the_image_python_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/kaggle/usr/lib/notebooks/sorokin/x")
    monkeypatch.setenv("PIP_TARGET", "/x")
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    env = spike.clean_env(os.environ, ["TRITON_PTXAS_PATH"])
    assert not {"PYTHONPATH", "PIP_TARGET", "TRITON_PTXAS_PATH"} & set(env)
    assert env["PYTHONNOUSERSITE"] == "1"


def test_install_runs_the_shipped_installer_and_reads_its_report(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, log, timeout, env):
        calls.append(command)
        report = Path(command[command.index("--report") + 1])
        report.write_text(json.dumps({"ok": True, "imports": {"torch": {"ok": True}}}))
        return 0, []

    monkeypatch.setattr(spike, "run_command", fake_run)
    settings = {"base_python": "/nonexistent/python3.11", "venv": str(tmp_path / "venv"),
                "timeout_s": 60}  # fmt: skip
    python, report = spike.install_engine(tmp_path / "house", settings, tmp_path, {})
    assert python == tmp_path / "venv" / "bin" / "python" and report["ok"]
    command = calls[0]
    assert command[0] == sys.executable  # the configured interpreter is missing here
    assert command[1] == str(tmp_path / "house" / "install_wheelhouse.py")
    assert command[command.index("--imports") + 1] == "torch,vllm"
    monkeypatch.setattr(spike, "run_command", lambda *a: (1, ["boom"]))
    with pytest.raises(spike.Stop, match="offline install failed"):
        spike.install_engine(tmp_path / "house", settings, tmp_path, {})


def test_script_build_config_feeds_the_spike_both_download_outputs():
    builder = load(REPO / "scripts" / "build_kaggle_notebook.py", "_test_notebook_builder")
    path = REPO / "configs" / "P6.3-01-script.yaml"
    build = builder.parse_config(path.read_bytes(), path)
    assert build.kind == "script" and build.accelerator == "NvidiaL4"
    assert (build.script, build.script_config) == ("scripts/agent_spike.py", "configs/P6.3-01.yaml")
    download = yaml.safe_load((REPO / "configs" / "P6.3-download.yaml").read_text())
    model_notebook = download["model"]["kaggle_notebook"]
    wheels_notebook = download["wheels"]["kaggle_notebook"]
    assert build.notebook_inputs == {model_notebook: 1, wheels_notebook: 1}
    mounts = {model_notebook: "/kaggle/input/m", wheels_notebook: "/kaggle/input/w"}
    filled = builder.fill_args(list(build.args), "f" * 40, mounts)
    args, _ = spike.build_parser().parse_known_args(["--config", "x", *filled])
    assert (args.commit, args.model_root, args.wheelhouse_root) == (
        "f" * 40,
        "/kaggle/input/m",
        "/kaggle/input/w",
    )
    config = spike.load_config(CONFIG)
    assert config["model"]["folder"] == download["model"]["folder"]
    assert config["model"]["revision"] == download["model"]["revision"]
    assert config["wheels"]["folder"] == download["wheels"]["folder"]
    assert f"vllm=={config['wheels']['vllm']}" in download["wheels"]["requirements"]
