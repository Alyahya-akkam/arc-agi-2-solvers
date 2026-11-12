"""Tests for scripts/cache_embeddings.py (task P3.2, BRIEF A8; runs P3.2-01 and P3.2-02).

The script runs in-process through ``main`` on its CPU ``--smoke`` path: the tiny random Qwen3,
the stand-in tokenizer, and synthetic tasks, all built in the script. The GPU run's own path
(``P3.2-01``'s config as committed: the input search, DEV40, layer 24, bf16, SDPA) also runs on
the CPU, against a Kaggle-like input tree holding a 36-layer copy of the tiny Qwen3 and stand-in
challenges of DEV40's ids, with only the device and the folders given on the command line.
``P3.2-02``'s config (8 views and the view diagnostic) runs the same ways, and the diagnostic is
checked against a direct 16-view encode. Outputs go under ``tmp_path``. Without torch,
transformers, or safetensors the module is skipped. Run with CUDA_VISIBLE_DEVICES="" (BRIEF A0).
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from arc26.data import DEV4, DEV40  # noqa: E402
from arc26.hyper.encoder import PairEncoder, pair_hash, view_descriptors, view_set_id  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "cache_embeddings.yaml"
# Run P3.2-02: P3.2-01's settings with 8 views and the view diagnostic against the 16.
CONFIG_02 = REPO / "configs" / "cache_embeddings_P3.2-02.yaml"
# Where Kaggle mounts the base model and the competition data, in the 2026 layout (K0) and in the
# 2025 layout of the reference notebook.
MODEL_MOUNT = {
    "2026": ("models", "sorokin", "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1"),
    "2025": ("qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1"),
}
DATA_MOUNT = {
    "2026": ("competitions", "arc-prize-2026-arc-agi-2"),
    "2025": ("arc-prize-2026-arc-agi-2",),
}
CHALLENGES = "arc-agi_evaluation_challenges.json"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "cache_embeddings", REPO / "scripts" / "cache_embeddings.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()
OUTPUTS = ("config.yaml", "run.json", "timing.jsonl", "throughput.jsonl", "metrics.json")


def smoke_args(out, config=CONFIG):
    return [
        "--config",
        str(config),
        "--smoke",
        "--out-dir",
        str(out),
        "--cache-dir",
        str(out / "embedding_cache"),
    ]


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_config(tmp_path, edit):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def touch(root, *parts):
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


def standin_challenges(task_ids):
    """Challenges of stand-in tasks: task i has 1 + i % 3 small pairs, all distinct."""
    return {
        task_id: {
            "train": [
                {"input": [[i % 10, i // 10 % 10], [j, 1]], "output": [[j, i % 10]]}
                for j in range(1 + i % 3)
            ],
            "test": [{"input": [[1]]}],
        }
        for i, task_id in enumerate(task_ids)
    }


def kaggle_inputs(root, layout="2026", task_ids=DEV40):
    """A Kaggle-like input tree: a 36-layer copy of the tiny Qwen3 in bf16 with the stand-in
    tokenizer at the model mount; the stand-in challenges at the data mount, beside a solutions
    file that is not JSON, so reading it would fail the run."""
    model = root.joinpath(*MODEL_MOUNT[layout])
    script.smoke_model(0, "sdpa", layers=36).to(torch.bfloat16).save_pretrained(model)
    script.smoke_tokenizer().save_pretrained(model)
    data = root.joinpath(*DATA_MOUNT[layout])
    data.mkdir(parents=True)
    (data / CHALLENGES).write_text(json.dumps(standin_challenges(task_ids)))
    (data / "arc-agi_evaluation_solutions.json").write_text("never read")
    return model, data / CHALLENGES


def gpu_args(tmp_path, inputs, *extra, config=CONFIG):
    """The GPU run's command line on this machine: a committed config, the input root of a
    Kaggle-like tree, the CPU, and folders under tmp_path."""
    return [
        "--config", str(config), "--input-root", str(inputs), "--device", "cpu",
        "--out-dir", str(tmp_path / "out"), "--cache-dir", str(tmp_path / "cache"), *extra,
    ]  # fmt: skip


def search_config(root, **changes):
    cfg = script.load_config(CONFIG, smoke=False, overrides={"input_root": str(root)})
    return script.replace(cfg, **changes)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One smoke run, then a second one in the same directories (the resume)."""
    out = tmp_path_factory.mktemp("smoke")
    first = script.main(smoke_args(out))
    snapshot = {
        "metrics": read(out / "metrics.json"),
        "throughput": (out / "throughput.jsonl").read_text(),
        "shards": sorted(p.name for p in (out / "embedding_cache").rglob("shard-*")),
    }
    second = script.main(smoke_args(out))
    return out, first, snapshot, second


# --- the smoke path -----------------------------------------------------------------------------


def test_smoke_completes_with_self_describing_outputs(smoke_run):
    out, code, first, _ = smoke_run
    assert code == script.EXIT_OK
    for name in (*OUTPUTS, "summary.md"):
        assert (out / name).is_file(), name
    metrics = first["metrics"]
    assert metrics["status"] == "complete"
    assert metrics["tasks"] == {"count": 3, "pairs": 9, "unique_pairs": 8}
    assert metrics["lengths"]["sequences"] == 9 * 16 and metrics["lengths"]["max"] == 1866
    throughput = metrics["throughput"]
    assert throughput["tasks"] == 3 and throughput["k"] == 4 and throughput["views"] == 16
    assert throughput["cycled"] == 2 and throughput["measured_this_session"] == 3
    assert 0 < throughput["seconds"]["median"] <= throughput["seconds"]["max"]
    assert throughput["slowest"]["tokens"] == throughput["tokens"]["max"]
    cache = metrics["cache"]
    assert cache["entries"] == 8 and cache["run_pairs_cached"] == 8
    assert cache["encoded_this_session"] == 8 and cache["pairs_per_second"] > 0
    # 64-wide fp32 means plus 16 per-view rows, then safetensors header bytes.
    assert cache["payload_bytes_per_pair"] == 4 * 64 * 17
    assert cache["payload_bytes_per_pair"] < cache["bytes_per_pair"] < 4 * 64 * 17 + 1024
    assert cache["bytes_per_task"]["max"] == 4 * cache["bytes_per_pair"]


def test_smoke_provenance_and_resolved_config(smoke_run):
    out = smoke_run[0]
    run = read(out / "run.json")
    assert run["run_id"] == "P3.2-smoke" and run["smoke"] is True
    assert run["device_name"] == "cpu" and run["model"]["dtype"] == "float32"
    assert run["model"]["num_hidden_layers"] == 4 and run["model"]["hidden_size"] == 64
    assert run["encoder"]["layer"] == 2 and len(run["encoder"]["views"]) == 16
    assert set(run["versions"]) == {"python", "torch", "transformers", "safetensors", "numpy"}
    assert run["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", run["commit"])
    assert run["inputs"]["challenges"] == str(out / "smoke_challenges.json")
    resolved = yaml.safe_load((out / "config.yaml").read_text())
    assert resolved["layer"] == 2 and resolved["device"] == "cpu"
    assert resolved["challenges"] == str(out / "smoke_challenges.json")
    assert resolved["out_dir"] == str(out)


def test_smoke_timing_and_throughput_records(smoke_run):
    out = smoke_run[0]
    timing = lines(out / "timing.jsonl")
    assert {r["phase"] for r in timing} == {"warmup", "throughput", "cache"}
    for record in timing:
        assert set(record) == {"puzzle_id", "op", "seconds", "tokens", "phase"}
        assert record["op"] in ("encode_render", "encode_prefill")
    throughput = lines(out / "throughput.jsonl")
    assert [r["task_id"] for r in throughput] == ["00000000", "00000001", "00000002"]
    for record in throughput:
        assert record["sequences"] == 64 and record["k"] == 4
        assert record["seconds"] >= record["render_seconds"] + record["prefill_seconds"] * 0.99
    summary = (out / "summary.md").read_text()
    assert "Status: complete" in summary and "| 3 | 2 |" in summary


def test_smoke_rerun_resumes_without_new_work(smoke_run):
    out, _, first, code = smoke_run
    assert code == script.EXIT_OK
    metrics = read(out / "metrics.json")
    assert metrics["status"] == "complete"
    assert metrics["throughput"]["measured_this_session"] == 0
    assert metrics["throughput"]["tasks"] == 3
    assert metrics["cache"]["encoded_this_session"] == 0
    assert metrics["cache"]["entries"] == 8
    assert (out / "throughput.jsonl").read_text() == first["throughput"]
    assert sorted(p.name for p in (out / "embedding_cache").rglob("shard-*")) == first["shards"]
    assert read(out / "run.json")["cache"]["entries_at_start"] == 8


def test_smoke_cache_holds_the_encoder_output(smoke_run):
    out = smoke_run[0]
    cfg = script.load_config(CONFIG, smoke=True, overrides={})
    tasks = script.load_tasks(out / "smoke_challenges.json", None)
    encoder = PairEncoder(
        script.smoke_model(cfg.smoke_seed), script.smoke_tokenizer(), layer=cfg.layer
    )
    cache = script.EmbeddingCache(
        out / "embedding_cache",
        model_id=cfg.model_id,
        layer=cfg.layer,
        pool=cfg.pool,
        view_set_id=encoder.view_set_id,
    )
    pairs = tasks[0].pairs
    keys = [pair_hash(p) for p in pairs]
    torch.testing.assert_close(
        cache.get_views(keys), encoder.encode_views(pairs), atol=1e-6, rtol=0
    )


# --- stops, failures, and resumes ---------------------------------------------------------------


def test_smoke_ignores_the_gpu_runs_subset(tmp_path):
    # DEV4's ids are evaluation ids, absent from the synthetic tasks.
    config = write_config(tmp_path, lambda raw: raw.update(subset="DEV4"))
    assert script.load_config(config, smoke=False, overrides={}).task_ids == DEV4
    assert script.load_config(config, smoke=True, overrides={}).task_ids is None
    out = tmp_path / "out"
    assert script.main(smoke_args(out, config)) == script.EXIT_OK
    metrics = read(out / "metrics.json")
    assert metrics["status"] == "complete" and metrics["tasks"]["count"] == 3
    resolved = yaml.safe_load((out / "config.yaml").read_text())
    assert resolved["subset"] is None and resolved["task_ids"] is None


def test_wall_time_stop_writes_logs_and_exits_3(tmp_path):
    config = write_config(tmp_path, lambda raw: raw["smoke"].update(max_wall_s=1e-6))
    out = tmp_path / "out"
    assert script.main(smoke_args(out, config)) == script.EXIT_STOPPED
    metrics = read(out / "metrics.json")
    assert metrics["status"].startswith("stopped: wall time")
    assert "lengths" in metrics and "cache" not in metrics
    assert "Status: stopped" in (out / "summary.md").read_text()


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="checks the refusal on a machine without CUDA"
)
def test_cuda_requested_without_cuda_fails_before_any_work(tmp_path):
    out = tmp_path / "out"
    assert script.main([*smoke_args(out), "--device", "cuda"]) == script.EXIT_FAILED
    metrics = read(out / "metrics.json")
    assert "CUDA is not available" in metrics["status"]
    assert "RuntimeError" in (out / "error.log").read_text()
    assert not (out / "timing.jsonl").exists()


def small_tasks(n=3):
    challenges = script.smoke_challenges(n, seed=1)
    tasks = [script.Task(t, v["train"]) for t, v in challenges.items()]
    # Drop the 30 by 30 pair: these tests check bookkeeping, not lengths.
    return [script.Task(t.task_id, [p for p in t.pairs if len(p["input"]) < 30]) for t in tasks]


def phase_setup(tmp_path, **changes):
    cfg = script.load_config(CONFIG, smoke=True, overrides={})
    cfg = script.replace(cfg, **changes)
    log = script.TimingLog(tmp_path / "timing.jsonl")
    encoder = PairEncoder(
        script.smoke_model(0), script.smoke_tokenizer(), layer=cfg.layer, timing=log
    )
    cache = script.EmbeddingCache(
        tmp_path / "cache",
        model_id=cfg.model_id,
        layer=cfg.layer,
        pool=cfg.pool,
        view_set_id=encoder.view_set_id,
    )
    return cfg, encoder, cache, log


def stop_on_call(n):
    calls = []

    def check(deadline):
        calls.append(deadline)
        if len(calls) >= n:
            raise script.WallTimeExceeded("wall time over max_wall_s")

    return check


def test_cache_phase_flushes_on_a_stop_and_a_rerun_encodes_only_the_rest(tmp_path, monkeypatch):
    tasks = small_tasks()
    unique = {pair_hash(p) for t in tasks for p in t.pairs}
    first_task = {pair_hash(p) for p in tasks[0].pairs}
    cfg, encoder, cache, log = phase_setup(tmp_path, flush_pairs=10**6)
    state = script.RunState(tasks=tasks)
    monkeypatch.setattr(script, "_check_deadline", stop_on_call(2))
    with pytest.raises(script.WallTimeExceeded):
        script.fill_cache(encoder, cache, tasks, cfg, log, state, deadline=0.0)
    assert len(cache) == len(first_task) == state.encoded_pairs
    monkeypatch.undo()
    reopened = script.EmbeddingCache(
        cache.path.parent,
        model_id=cfg.model_id,
        layer=cfg.layer,
        pool=cfg.pool,
        view_set_id=encoder.view_set_id,
    )
    rerun = script.RunState(tasks=tasks)
    script.fill_cache(encoder, reopened, tasks, cfg, log, rerun, deadline=float("inf"))
    assert rerun.encoded_pairs == len(unique - first_task)
    assert reopened.missing(unique) == [] and len(reopened) == len(unique)


def test_cache_phase_writes_every_flush_pairs(tmp_path):
    tasks = small_tasks()
    cfg, encoder, cache, log = phase_setup(tmp_path, flush_pairs=1, store_views=False)
    script.fill_cache(encoder, cache, tasks, cfg, log, script.RunState(), deadline=float("inf"))
    shards = list(cache.path.glob("shard-*.safetensors"))
    assert len(shards) == len(tasks)  # one write per task: each task adds at least one pair
    key = pair_hash(tasks[0].pairs[0])
    assert not cache.has_views(key) and cache.n_views is None


def test_phases_refuse_a_log_that_is_not_the_encoders(tmp_path):
    tasks = small_tasks()
    cfg, encoder, cache, _ = phase_setup(tmp_path)
    other = script.TimingLog(tmp_path / "other.jsonl")
    with pytest.raises(ValueError, match="timing callback"):
        script.fill_cache(encoder, cache, tasks, cfg, other, script.RunState(), float("inf"))
    with pytest.raises(ValueError, match="timing callback"):
        script.measure_throughput(
            encoder, tasks, cfg, tmp_path / "t.jsonl", other, script.RunState(), float("inf")
        )


def test_throughput_phase_skips_measured_tasks(tmp_path, monkeypatch):
    tasks = small_tasks()
    cfg, encoder, _, log = phase_setup(tmp_path, throughput_tasks=2)
    ledger = tmp_path / "throughput.jsonl"
    ledger.write_text(json.dumps({"task_id": tasks[0].task_id}) + "\n" + '{"task_id": "0000')
    state = script.RunState()
    script.measure_throughput(encoder, tasks, cfg, ledger, log, state, deadline=float("inf"))
    records = script.read_jsonl(ledger)
    assert [r["task_id"] for r in records] == [tasks[0].task_id, tasks[1].task_id]
    assert state.measured_tasks == 1
    assert records[1]["cycled"] is (len(tasks[1].pairs) < 4)


# --- config and inputs --------------------------------------------------------------------------


@pytest.mark.parametrize("config", [CONFIG, CONFIG_02], ids=["P3.2-01", "P3.2-02"])
def test_every_config_value_has_a_comment(config):
    previous = ""
    for line in config.read_text().splitlines():
        if re.match(r"\s*[a-z_]+:", line):
            assert "#" in line or previous.lstrip().startswith("#"), line
        if line.strip():
            previous = line


def test_real_config_matches_the_brief():
    cfg = script.load_config(CONFIG, smoke=False, overrides={})
    assert (cfg.layer, cfg.pool, cfg.n_perm, cfg.view_seed) == (24, "output", 2, 2)
    assert (cfg.throughput_k, cfg.acceptance_s, cfg.device, cfg.dtype) == (
        4,
        10.0,
        "cuda:0",
        "bfloat16",
    )
    # DEV40 comes from arc26.data, its one definition, and the inputs are searched for.
    assert cfg.subset == "DEV40" and cfg.task_ids == DEV40 and cfg.reference_n_perm is None
    assert cfg.input_root == "/kaggle/input" and cfg.model_dir is None
    assert script.model_folder_parts(cfg.model_id) == MODEL_MOUNT["2025"]
    assert cfg.challenges == "/".join((*DATA_MOUNT["2025"], CHALLENGES))
    assert cfg.out_dir == "/kaggle/working/run_logs" and cfg.run_id == "P3.2-01"
    smoke = script.load_config(CONFIG, smoke=True, overrides={"device": "cpu", "out_dir": "x"})
    assert smoke.layer == 2 and smoke.out_dir == "x" and smoke.n_perm == cfg.n_perm


def test_no_input_path_is_hardcoded():
    """docs/STATE.md, rules for scripts: only the search root names /kaggle/input."""
    raw = yaml.safe_load(CONFIG.read_text())
    smoke = raw.pop("smoke")
    values = [value for key, value in raw.items() if key != "input_root"] + list(smoke.values())
    assert [v for v in values if isinstance(v, str) and "/kaggle/input" in v] == []


def test_model_folder_parts_follow_the_kaggle_mount():
    handle = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
    assert script.model_folder_parts(handle) == MODEL_MOUNT["2025"]
    for bad in ("sorokin/qwen3_4b_grids15_sft139", "a/b/c/d/e/f", "a//c/d/e"):
        with pytest.raises(script.ConfigError, match="Kaggle Models handle"):
            script.model_folder_parts(bad)


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw.update(extra=1),
        lambda raw: raw.pop("layer"),
        lambda raw: raw.update(layer=-1),
        lambda raw: raw.update(layer=True),
        lambda raw: raw.update(pool="pair"),
        lambda raw: raw.update(dtype="int8"),
        lambda raw: raw.update(task_ids=["0934a4d8"]),
        lambda raw: raw.update(subset="DEV41"),
        lambda raw: raw.update(input_root=""),
        lambda raw: raw.update(model_dir=""),
        lambda raw: raw.update(challenges="../arc-agi_evaluation_challenges.json"),
        # The GPU settings search for the model by its handle, so the handle is checked even when
        # the smoke run replaces it.
        lambda raw: raw.update(model_id="sorokin/qwen3_4b_grids15_sft139"),
        lambda raw: raw.update(max_wall_s=0),
        lambda raw: raw["smoke"].pop("seed"),
        lambda raw: raw["smoke"].update(n_perm=1),
        lambda raw: raw["smoke"].update(subset="DEV5"),
        # The diagnostic's reference set must hold the run's views as its first rounds.
        lambda raw: raw.pop("reference_n_perm"),
        lambda raw: raw.update(reference_n_perm=2),
        lambda raw: raw.update(reference_n_perm=1),
        lambda raw: raw.update(reference_n_perm=True),
        lambda raw: raw.update(n_perm=0, reference_n_perm=2),
    ],
)
def test_invalid_configs_exit_2(tmp_path, edit):
    config = write_config(tmp_path, edit)
    assert script.main(smoke_args(tmp_path / "out", config)) == script.EXIT_CONFIG
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "flag", [("--model-dir", "/models/x"), ("--challenges", "c.json"), ("--input-root", "/k")]
)
def test_smoke_refuses_model_and_data_paths(tmp_path, flag):
    assert script.main([*smoke_args(tmp_path / "out"), *flag]) == script.EXIT_CONFIG


# --- the input search and the GPU run's settings on the CPU -------------------------------------


@pytest.mark.parametrize("layout", sorted(MODEL_MOUNT))
def test_find_inputs_finds_the_model_and_the_challenges_in_either_layout(tmp_path, layout):
    model = touch(tmp_path, *MODEL_MOUNT[layout], "config.json").parent
    challenges = touch(tmp_path, *DATA_MOUNT[layout], CHALLENGES)
    # Never matched: the solutions file beside the challenges, another dataset's copy of the
    # challenges, another version of the model, and a model folder inside a package tree.
    touch(tmp_path, *DATA_MOUNT[layout], "arc-agi_evaluation_solutions.json")
    touch(tmp_path, "datasets", "someone", "old-arc", CHALLENGES)
    touch(tmp_path, *MODEL_MOUNT[layout][:-1], "2", "config.json")
    touch(tmp_path, "datasets", "x", "site-packages", *MODEL_MOUNT["2025"], "config.json")
    assert script.find_inputs(search_config(tmp_path)) == (model, challenges)


def test_find_inputs_refuses_a_missing_or_doubled_input_and_lists_what_it_found(tmp_path):
    touch(tmp_path, *DATA_MOUNT["2026"], CHALLENGES)
    other = touch(tmp_path, "models", "x", "other", "transformers", "bfloat16", "1", "config.json")
    with pytest.raises(script.InputSearchError, match="found 0 model folders") as err:
        script.find_inputs(search_config(tmp_path))
    assert str(other) in str(err.value)
    touch(tmp_path, *MODEL_MOUNT["2026"], "config.json")
    second = touch(tmp_path, *DATA_MOUNT["2025"], CHALLENGES)  # another input with that name
    with pytest.raises(script.InputSearchError, match="found 2 files ending in") as err:
        script.find_inputs(search_config(tmp_path))
    assert str(second) in str(err.value)


def test_explicit_paths_skip_the_search(tmp_path):
    cfg = search_config(tmp_path / "absent", model_dir="/models/x", challenges="/data/c.json")
    assert script.find_inputs(cfg) == (Path("/models/x"), Path("/data/c.json"))
    with pytest.raises(script.InputSearchError, match="found 0 files"):
        script.find_inputs(search_config(tmp_path / "absent", model_dir="/models/x"))


def test_find_files_is_bounded_and_walks_each_folder_once(tmp_path):
    deep = touch(tmp_path, *"abcdefgh", "x.json")  # level 9
    assert script.find_files(tmp_path, {"x": ("x.json",)}) == {"x": []}
    assert script.find_files(tmp_path, {"x": ("X.JSON",)}, max_depth=9) == {"x": [deep]}
    touch(tmp_path, "real", "y.json")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    assert len(script.find_files(tmp_path, {"y": ("y.json",)})["y"]) == 1


def test_command_line_paths_name_the_folder_and_the_file_themselves(tmp_path, monkeypatch):
    """Relative --model-dir and --challenges are paths under the working directory, never a
    searched ending; the run records them before it loads anything (here a missing folder)."""
    monkeypatch.chdir(tmp_path)
    args = gpu_args(tmp_path, tmp_path / "empty", "--model-dir", "m", "--challenges", "c.json")
    assert script.main(args) == script.EXIT_FAILED
    inputs = read(tmp_path / "out" / "run.json")["inputs"]
    assert inputs == {"model_dir": str(tmp_path / "m"), "challenges": str(tmp_path / "c.json")}


def test_the_gpu_runs_settings_run_on_cpu_from_a_kaggle_layout(tmp_path):
    """P3.2-01's config as committed (input search, DEV40, layer 24, bf16, SDPA), with only the
    device and the folders on the command line, and the commit as the script notebook passes it."""
    inputs = tmp_path / "input"
    model, challenges = kaggle_inputs(inputs)
    code = script.main(gpu_args(tmp_path, inputs, "--commit", "c" * 40))
    out = tmp_path / "out"
    assert code == script.EXIT_OK, (out / "metrics.json").read_text()
    run = read(out / "run.json")
    assert run["inputs"] == {"model_dir": str(model), "challenges": str(challenges)}
    assert run["model"]["model_dir"] == str(model) and run["challenges"] == str(challenges)
    assert run["commit"] == "c" * 40 and run["dirty"] is None and run["smoke"] is False
    assert run["model"]["num_hidden_layers"] == 36 and run["model"]["dtype"] == "bfloat16"
    assert run["encoder"]["layer"] == 24 and len(run["encoder"]["views"]) == 16
    resolved = yaml.safe_load((out / "config.yaml").read_text())
    assert resolved["subset"] == "DEV40" and resolved["task_ids"] == list(DEV40)
    assert resolved["model_dir"] is None and resolved["input_root"] == str(inputs)
    metrics = read(out / "metrics.json")
    assert metrics["status"] == "complete"
    assert metrics["tasks"] == {"count": 40, "pairs": 79, "unique_pairs": 79}
    assert metrics["throughput"]["tasks"] == 40 and metrics["throughput"]["cycled"] == 40
    assert [r["task_id"] for r in lines(out / "throughput.jsonl")] == list(DEV40)
    assert metrics["cache"]["entries"] == 79 and metrics["cache"]["run_pairs_cached"] == 79
    summary = (out / "summary.md").read_text()
    assert "Tasks: 40 (DEV40)" in summary and f"model `{model}`" in summary


def test_an_input_not_found_stops_the_run_before_the_model_loads(tmp_path):
    inputs = tmp_path / "input"
    touch(inputs, *DATA_MOUNT["2026"], CHALLENGES)
    assert script.main(gpu_args(tmp_path, inputs)) == script.EXIT_FAILED
    out = tmp_path / "out"
    assert read(out / "metrics.json")["status"].startswith("failed: InputSearchError: found 0")
    assert "InputSearchError" in (out / "error.log").read_text()
    run = read(out / "run.json")  # written however the run ends
    assert run["device_name"] == "cpu" and "inputs" not in run and "model" not in run
    assert not (out / "timing.jsonl").exists()


def test_load_tasks_reads_demonstrations_only(tmp_path):
    challenges = {
        "b": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]},
        "a": {"train": [{"input": [[4]], "output": [[5]]}] * 2, "test": [{"input": [[6]]}]},
    }
    path = tmp_path / "challenges.json"
    path.write_text(json.dumps(challenges))
    tasks = script.load_tasks(path, None)
    assert [t.task_id for t in tasks] == ["a", "b"]
    assert tasks[1].pairs == [{"input": [[1]], "output": [[2]]}]
    assert [t.task_id for t in script.load_tasks(path, ["b"])] == ["b"]
    with pytest.raises(ValueError, match="not in"):
        script.load_tasks(path, ["c"])
    solutions = tmp_path / "arc-agi_evaluation_solutions.json"
    solutions.write_text(json.dumps(challenges))
    with pytest.raises(ValueError, match="only challenges files"):
        script.load_tasks(solutions, None)


def test_preflight_names_the_failing_task():
    tokenizer = script.smoke_tokenizer()
    tasks = [script.Task("deadbeef", [{"input": [[1, 2]], "output": [[12]]}])]
    with pytest.raises(ValueError, match="task deadbeef"):
        script.preflight(tasks, script.QwenFormatter(tokenizer), [""])


def test_summarize_throughput_counts_tasks_over_the_bound():
    records = [
        {
            "task_id": f"t{i}",
            "k": 4,
            "n_pairs": n,
            "cycled": n < 4,
            "sequences": 64,
            "tokens": 1000 * (i + 1),
            "seconds": s,
            "render_seconds": 0.1,
            "device": "L4",
        }
        for i, (n, s) in enumerate([(2, 4.0), (4, 9.0), (5, 12.0)])
    ]
    summary = script.summarize_throughput(records, acceptance_s=10.0)
    assert summary["over_acceptance"] == 1 and summary["cycled"] == 1
    assert summary["seconds"]["max"] == 12.0 and summary["seconds"]["median"] == 9.0
    assert summary["slowest"]["task_id"] == "t2" and summary["views"] == 16
    assert summary["seconds_per_1k_tokens"] == pytest.approx(25.0 / 6.0)
    assert script.summarize_throughput([], 10.0) == {"tasks": 0, "acceptance_s": 10.0}


def test_git_commit_outside_a_checkout(tmp_path):
    assert script.git_commit(tmp_path) == (None, None)


# --- run P3.2-02: 8 views and the view diagnostic ------------------------------------------------


def test_the_p3_2_02_config_is_p3_2_01s_with_the_cut():
    """Only the run id, the view count, and the diagnostic change, and the smoke's names."""
    old, new = yaml.safe_load(CONFIG.read_text()), yaml.safe_load(CONFIG_02.read_text())
    old_smoke, new_smoke = old.pop("smoke"), new.pop("smoke")
    assert sorted(new) == sorted(old)
    assert {key for key in new if new[key] != old[key]} == {"run_id", "n_perm", "reference_n_perm"}
    assert (new["run_id"], new["n_perm"], new["reference_n_perm"]) == ("P3.2-02", 1, 2)
    assert {key for key in new_smoke if new_smoke[key] != old_smoke[key]} == {
        "run_id", "cache_dir", "out_dir",
    }  # fmt: skip
    cfg = script.load_config(CONFIG_02, smoke=False, overrides={})
    assert cfg.task_ids == DEV40 and cfg.device == "cuda:0" and cfg.layer == 24


def test_the_8_views_are_the_first_8_of_the_16_under_seed_2():
    """view_descriptors draws round by round, so one round under seed 2 is the first round of
    P3.2-01's two; any other seed draws other permutations."""
    sixteen = view_descriptors(2, 2)
    assert view_descriptors(1, 2) == sixteen[:8]
    assert len(set(sixteen)) == 16
    for seed in (0, 1, 3):
        assert not set(view_descriptors(1, seed)) <= set(sixteen)


def distinct_pairs(tasks):
    pairs = {}
    for task in tasks:
        for pair in task.pairs:
            pairs.setdefault(pair_hash(pair), pair)
    return pairs


def test_the_view_diagnostic_equals_a_direct_16_view_encode(tmp_path):
    """The 16-view embedding built from the cached 8-view mean and the other 8 views equals a
    direct encode of all 16, so the reported cosines are those of the direct encodes."""
    cfg = script.load_config(CONFIG_02, smoke=True, overrides={})
    tasks = small_tasks()
    model, tokenizer = script.smoke_model(0), script.smoke_tokenizer()
    sixteen = view_descriptors(2, 2)
    log = script.TimingLog(tmp_path / "timing.jsonl")
    encoder = PairEncoder(model, tokenizer, layer=cfg.layer, views=sixteen[:8], timing=log)
    cache = script.EmbeddingCache(
        tmp_path / "cache",
        model_id=cfg.model_id,
        layer=cfg.layer,
        pool=cfg.pool,
        view_set_id=encoder.view_set_id,
    )
    state = script.RunState(tasks=tasks)
    script.fill_cache(encoder, cache, tasks, cfg, log, state, deadline=float("inf"))
    complement = PairEncoder(model, tokenizer, layer=cfg.layer, views=sixteen[8:], timing=log)
    out = tmp_path / "view_diagnostic.jsonl"
    script.diagnose_views(complement, cache, tasks, 16, log, state, float("inf"), out)
    pairs = distinct_pairs(tasks)
    rows = PairEncoder(model, tokenizer, layer=cfg.layer, views=sixteen).encode_views(
        list(pairs.values())
    )
    run, ref = rows[:, :8].mean(dim=1).double(), rows.mean(dim=1).double()
    cosine = torch.nn.functional.cosine_similarity(run, ref, dim=1)
    centre = ref.mean(dim=0, keepdim=True)
    centered = torch.nn.functional.cosine_similarity(run - centre, ref - centre, dim=1)
    records = lines(out)
    assert [r["pair"] for r in records] == list(pairs)
    torch.testing.assert_close(
        torch.tensor([r["cosine"] for r in records], dtype=torch.float64), cosine, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        torch.tensor([r["centered_cosine"] for r in records], dtype=torch.float64),
        centered,
        atol=1e-5,
        rtol=0,
    )
    d = state.diagnostic
    assert (d["views"], d["reference_views"], d["pairs"], d["tasks"]) == (8, 16, len(pairs), 3)
    assert d["pair_cosine"]["min"] == pytest.approx(float(cosine.min()), abs=1e-6)
    assert d["pair_cosine"]["median"] == pytest.approx(float(cosine.quantile(0.5)), abs=1e-6)
    assert -1 <= d["between_pairs_cosine_median"] <= 1 and d["seconds"] > 0 and d["tokens"] > 0
    assert {r["phase"] for r in lines(tmp_path / "timing.jsonl")} == {"cache", "diagnostic"}


@pytest.fixture(scope="module")
def smoke_02(tmp_path_factory):
    """One P3.2-02 smoke run (8 views, the diagnostic), then a second in the same directories."""
    out = tmp_path_factory.mktemp("smoke02")
    first = script.main(smoke_args(out, CONFIG_02))
    snapshot = read(out / "metrics.json")
    second = script.main(smoke_args(out, CONFIG_02))
    return out, first, snapshot, second


def test_the_p3_2_02_smoke_measures_8_views_and_reports_the_diagnostic(smoke_02):
    out, code, metrics, _ = smoke_02
    assert code == script.EXIT_OK and metrics["status"] == "complete"
    throughput = metrics["throughput"]
    assert throughput["views"] == 8 and throughput["tasks"] == 3
    assert {r["sequences"] for r in lines(out / "throughput.jsonl")} == {32}
    assert metrics["lengths"]["sequences"] == 9 * 8
    # 64-wide fp32 means plus 8 per-view rows.
    assert metrics["cache"]["payload_bytes_per_pair"] == 4 * 64 * 9
    d = metrics["view_diagnostic"]
    assert (d["views"], d["reference_views"], d["pairs"], d["tasks"]) == (8, 16, 8, 3)
    for name in ("pair_cosine", "pair_cosine_centered", "task_cosine"):
        assert -1 <= d[name]["min"] <= d[name]["p10"] <= d[name]["median"] <= 1, name
    assert len(lines(out / script.DIAGNOSTIC_FILE)) == 8
    run = read(out / "run.json")
    sixteen = view_descriptors(2, 2)
    assert run["encoder"]["views"] == sixteen[:8]
    assert run["view_diagnostic"] == {
        "reference_n_perm": 2,
        "reference_view_set_id": view_set_id(sixteen),
        "views_encoded": sixteen[8:],
    }
    summary = (out / "summary.md").read_text()
    assert "## View diagnostic: 8 views against 16 (reported, decides nothing)" in summary


def test_the_p3_2_02_rerun_redoes_only_the_diagnostic(smoke_02):
    out, _, first, code = smoke_02
    assert code == script.EXIT_OK
    metrics = read(out / "metrics.json")
    assert metrics["throughput"]["measured_this_session"] == 0
    assert metrics["cache"]["encoded_this_session"] == 0
    assert metrics["view_diagnostic"]["pairs"] == 8
    assert metrics["view_diagnostic"]["pair_cosine"] == first["view_diagnostic"]["pair_cosine"]


def test_a_stop_before_the_diagnostic_ends_reports_none(tmp_path, monkeypatch):
    """The wall-time stop in the diagnostic phase leaves no partial diagnostic in metrics.json."""
    calls = []
    checker = script._check_deadline

    def stop_in_the_diagnostic(deadline):
        # Calls 1 to 7: after the preflight, then 3 throughput and 3 cache tasks.
        calls.append(deadline)
        if len(calls) > 7:
            raise script.WallTimeExceeded("wall time over max_wall_s")
        checker(deadline)

    monkeypatch.setattr(script, "_check_deadline", stop_in_the_diagnostic)
    out = tmp_path / "out"
    assert script.main(smoke_args(out, CONFIG_02)) == script.EXIT_STOPPED
    metrics = read(out / "metrics.json")
    assert metrics["status"].startswith("stopped") and "view_diagnostic" not in metrics
    assert metrics["cache"]["entries"] == 8


def test_the_p3_2_02_settings_run_on_cpu_from_a_kaggle_layout(tmp_path):
    """P3.2-02's config as committed, on the 36-layer stand-in: 8 views, DEV40, the diagnostic
    over every distinct pair."""
    inputs = tmp_path / "input"
    kaggle_inputs(inputs)
    code = script.main(gpu_args(tmp_path, inputs, "--commit", "c" * 40, config=CONFIG_02))
    out = tmp_path / "out"
    assert code == script.EXIT_OK, (out / "metrics.json").read_text()
    metrics = read(out / "metrics.json")
    assert metrics["tasks"] == {"count": 40, "pairs": 79, "unique_pairs": 79}
    assert metrics["throughput"]["tasks"] == 40 and metrics["throughput"]["views"] == 8
    assert metrics["cache"]["entries"] == 79
    d = metrics["view_diagnostic"]
    assert (d["pairs"], d["tasks"], d["views"], d["reference_views"]) == (79, 40, 8, 16)
    assert len(lines(out / script.DIAGNOSTIC_FILE)) == 79
    assert read(out / "run.json")["encoder"]["layer"] == 24
