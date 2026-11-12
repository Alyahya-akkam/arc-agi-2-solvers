"""Task MT: episode mode of scripts/cache_embeddings.py (run MT-01), on the CPU.

The script runs in-process through ``main`` on the ``--smoke`` path of
``configs/cache_embeddings_MT-01.yaml``: the tiny random Qwen3 and the stand-in tokenizer built in
the script, and the episode fixture of tests/testdata/train_hypernet/ (a P3.4 build in the real
format, its toy split standing in for train). The config's two encoding processes run as child
processes of the script; tests that reach inside a process (a deadline, a failure, a kill before
the merge) run one process in-process. Outputs go under ``tmp_path``. Without torch,
transformers, or safetensors the module is skipped. Run with CUDA_VISIBLE_DEVICES="".
"""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from arc26.hyper import train as tr  # noqa: E402
from arc26.hyper.encoder import (  # noqa: E402
    EmbeddingCache,
    PairEncoder,
    pair_hash,
    view_descriptors,
    view_set_id,
)

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "cache_embeddings_MT-01.yaml"
CHALLENGE_CONFIG = REPO / "configs" / "cache_embeddings.yaml"
FIXTURE = REPO / "tests" / "testdata" / "train_hypernet"
FIXTURE_SHA = "15317ba48d7efefab0b37aaa69727454cb3fd71141140a1cd1d018c4013b0550"


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load("cache_embeddings")
kaggle_ops = load("kaggle_ops")


def smoke_args(out, config=CONFIG, *extra):
    return [
        "--config", str(config), "--smoke", "--out-dir", str(out),
        "--cache-dir", str(out / "embedding_cache"), *extra,
    ]  # fmt: skip


def read(path):
    return json.loads(Path(path).read_text())


def write_config(tmp_path, edit, source=CONFIG):
    raw = yaml.safe_load(source.read_text())
    edit(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def one_process(raw):
    raw["smoke"]["gpus"] = 1


def fixture_pairs():
    """The fixture's distinct demonstration pairs of the toy and validation splits, by hash."""
    episodes = tr.open_episode_set(FIXTURE / "episode_build", FIXTURE_SHA, FIXTURE / "check", 8192)
    pairs = {}
    for split in ("toy", "val"):
        for record in tr.load_split(episodes, split, 8192):
            for x, y in record.demos:
                pairs.setdefault(pair_hash({"input": x, "output": y}), {"input": x, "output": y})
    return pairs


def open_cache(root):
    views = view_descriptors(1, 2)
    return EmbeddingCache(
        root,
        model_id="smoke/tiny-random-qwen3",
        layer=2,
        pool="output",
        view_set_id=view_set_id(views),
    )


def direct_encoder():
    model = script.smoke_model(0, "eager").to(torch.float32)
    return PairEncoder(
        model, script.smoke_tokenizer(), layer=2, pool="output", views=view_descriptors(1, 2),
        batch_tokens=4096,
    )  # fmt: skip


def parts_left(cache_root):
    return sorted(Path(cache_root).rglob("part-*.safetensors"))


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("mt01") / "out"
    code = script.main(smoke_args(out))
    return out, code


# --- the smoke: two processes, every distinct pair, the check ----------------------------------


def test_the_smoke_caches_every_distinct_demonstration_pair(smoke_run):
    out, code = smoke_run
    assert code == script.EXIT_OK
    metrics = read(out / "metrics.json")
    expected = fixture_pairs()
    assert metrics["status"] == "complete" and metrics["mode"] == "episodes"
    pairs = metrics["pairs"]
    assert pairs["distinct"] == len(expected) and pairs["missing"] == 0
    assert pairs["to_encode"] == len(expected) and pairs["cached_at_start"] == 0
    splits = metrics["episodes"]["splits"]
    assert (splits["toy"]["episodes"], splits["val"]["episodes"]) == (12, 6)
    assert sum(pairs["by_source"].values()) >= pairs["distinct"]
    assert metrics["cache"]["entries"] == len(expected)
    assert metrics["check"]["pass"] is True and metrics["check"]["pairs"] == 8
    encode = metrics["encode"]
    assert encode["gpus"] == 2 and encode["pairs"] == len(expected)
    assert {r["status"] for r in encode["ranks"]} == {"complete"}
    cache = open_cache(out / "embedding_cache")
    assert sorted(cache.missing(expected)) == [] and len(cache) == len(expected)
    assert parts_left(out / "embedding_cache") == []
    summary = (out / "summary.md").read_text()
    assert f"Distinct demonstration pairs (the cache's target): {len(expected)}" in summary
    assert chr(0x2014) not in summary


def test_the_smoke_writes_its_logs_under_the_pull_allowlist(smoke_run):
    out, _ = smoke_run
    expected = {"config.yaml", "run.json", "metrics.json", "summary.md"}
    expected |= {
        "timing-rank0.jsonl",
        "timing-rank1.jsonl",
        "encode-rank0.json",
        "encode-rank1.json",
    }
    logs = {p.name for p in out.iterdir() if p.is_file()}
    assert logs == expected
    for name in logs:
        assert kaggle_ops.pull_allowed(f"run_logs/{name}"), name
    records = [json.loads(x) for x in (out / "timing-rank0.jsonl").read_text().splitlines()]
    assert records and all(r["op"] == "encode" and r["puzzle_id"] is None for r in records)
    assert all({"puzzle_id", "op", "seconds", "tokens"} <= set(r) for r in records)
    run = read(out / "run.json")
    assert run["inputs"]["manifest_sha256"] == FIXTURE_SHA and run["encoder"]["gpus"] == 2
    host = run["host"]  # the machine's memory, which MT-02's memory plan reads
    assert host["cpus"] >= 1 and (host["memory_bytes"] is None or host["memory_bytes"] > 2**20)
    assert "- Host: " in (out / "summary.md").read_text()
    assert len(run["encoder"]["views"]) == 8
    resolved = yaml.safe_load((out / "config.yaml").read_text())
    assert resolved["episodes"]["splits"] == ["toy", "val"] and resolved["gpus"] == 2


def test_cached_embeddings_equal_a_direct_encode_of_each_pair(smoke_run):
    """Every entry is its own pair's embedding: a misfiled row would differ."""
    out, _ = smoke_run
    pairs = fixture_pairs()
    keys = sorted(pairs)
    cached = open_cache(out / "embedding_cache").get(keys)
    fresh = direct_encoder().encode_pairs([pairs[k] for k in keys])
    torch.testing.assert_close(cached, fresh, rtol=1e-5, atol=1e-6)


def test_a_rerun_encodes_nothing_and_completes(smoke_run, tmp_path):
    out, _ = smoke_run
    copy = tmp_path / "out"
    shutil.copytree(out / "embedding_cache", copy / "embedding_cache")
    assert script.main(smoke_args(copy)) == script.EXIT_OK
    metrics = read(copy / "metrics.json")
    assert metrics["pairs"]["to_encode"] == 0 and "encode" not in metrics
    assert metrics["check"]["pass"] is True


# --- cut sessions, leftover parts, seeds -------------------------------------------------------


def deadline_after(monkeypatch, chunks):
    """Let ``chunks`` chunks start, then report the clock past every deadline."""
    calls = {"n": 0}
    real = script._now

    def now():
        calls["n"] += 1
        # Call 1 is main's start; then one call per chunk before it starts.
        return real() if calls["n"] <= 1 + chunks else real() + 1e9

    monkeypatch.setattr(script, "_now", now)


def test_a_cut_session_keeps_every_finished_chunk_and_a_rerun_finishes(tmp_path, monkeypatch):
    config = write_config(tmp_path, one_process)
    out = tmp_path / "out"
    deadline_after(monkeypatch, 2)
    assert script.main(smoke_args(out, config)) == script.EXIT_STOPPED
    metrics = read(out / "metrics.json")
    expected = len(fixture_pairs())
    assert metrics["status"].startswith("stopped") and metrics["pairs"]["missing"] == expected - 8
    assert metrics["cache"]["entries"] == 8 and parts_left(out / "embedding_cache") == []
    assert read(out / "encode-rank0.json")["status"].startswith("stopped")
    monkeypatch.undo()
    assert script.main(smoke_args(out, config)) == script.EXIT_OK
    again = read(out / "metrics.json")
    assert again["pairs"]["cached_at_start"] == 8 and again["pairs"]["to_encode"] == expected - 8
    assert again["cache"]["entries"] == expected


def test_parts_of_a_session_killed_before_its_merge_are_merged_first(tmp_path, monkeypatch):
    """A hard stop (the notebook's kill, Kaggle's timeout) can end a session after its processes
    wrote parts and before the merge: the next session merges them and encodes only the rest."""
    config = write_config(tmp_path, one_process)
    out = tmp_path / "out"
    monkeypatch.setattr(script, "merge_parts", lambda cache, flush: {"files": 0, "entries": 0})
    assert script.main(smoke_args(out, config)) == script.EXIT_STOPPED
    written = parts_left(out / "embedding_cache")
    assert written and len(open_cache(out / "embedding_cache")) == 0
    monkeypatch.undo()
    assert script.main(smoke_args(out, config)) == script.EXIT_OK
    metrics = read(out / "metrics.json")
    assert metrics["pairs"]["leftover_parts"] == len(written)
    assert metrics["pairs"]["to_encode"] == 0 and metrics["cache"]["entries"] == len(
        fixture_pairs()
    )
    assert parts_left(out / "embedding_cache") == []


def test_a_failed_process_keeps_what_it_wrote(tmp_path, monkeypatch):
    config = write_config(tmp_path, one_process)
    out = tmp_path / "out"
    real = PairEncoder.encode_pairs
    calls = {"n": 0}

    def fail_second(self, pairs, puzzle_id=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure in the second chunk")
        return real(self, pairs, puzzle_id)

    monkeypatch.setattr(PairEncoder, "encode_pairs", fail_second)
    assert script.main(smoke_args(out, config)) == script.EXIT_FAILED
    metrics = read(out / "metrics.json")
    assert metrics["status"].startswith("failed: RuntimeError: encoding processes exited")
    assert metrics["cache"]["entries"] == 4  # the first chunk, merged although the process failed
    assert "injected failure" in (out / "error-rank0.log").read_text()


def test_a_seeded_session_continues_an_earlier_one(tmp_path, monkeypatch):
    """The earlier session's cache, read-only as Kaggle mounts it, is copied, not written."""
    config = write_config(tmp_path, one_process)
    first, second = tmp_path / "first", tmp_path / "second"
    deadline_after(monkeypatch, 2)
    assert script.main(smoke_args(first, config)) == script.EXIT_STOPPED
    monkeypatch.undo()
    seed = first / "embedding_cache"
    before = {p: p.read_bytes() for p in seed.rglob("*") if p.is_file()}
    for path in [*seed.rglob("*")]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        args = smoke_args(second, config, "--seed-cache", str(seed))
        assert script.main(args) == script.EXIT_OK
    finally:
        for path in [*seed.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
    metrics = read(second / "metrics.json")
    expected = fixture_pairs()
    assert metrics["pairs"]["seeded"] == 8 and metrics["pairs"]["cached_at_start"] == 8
    assert metrics["pairs"]["to_encode"] == len(expected) - 8 and metrics["pairs"]["missing"] == 0
    assert {p: p.read_bytes() for p in seed.rglob("*") if p.is_file()} == before
    keys = sorted(expected)
    seeded = open_cache(second / "embedding_cache").get(keys)
    fresh = direct_encoder().encode_pairs([expected[k] for k in keys])
    torch.testing.assert_close(seeded, fresh, rtol=1e-5, atol=1e-6)
    # Seeding a cache that already holds embeddings, or from a seed of another namespace, fails.
    assert script.main(smoke_args(second, config, "--seed-cache", str(seed))) == 1
    assert "already holds embeddings" in read(second / "metrics.json")["status"]
    empty = tmp_path / "empty"
    empty.mkdir()
    third = tmp_path / "third"
    assert script.main(smoke_args(third, config, "--seed-cache", str(empty))) == 1
    assert "has no namespace" in read(third / "metrics.json")["status"]


def test_the_check_catches_an_embedding_filed_under_another_pair(tmp_path):
    pairs = fixture_pairs()
    keys = sorted(pairs)
    rows = direct_encoder().encode_pairs([pairs[k] for k in keys])
    cosine = torch.nn.functional.cosine_similarity(rows[:1], rows, dim=1)
    far = int(torch.argmin(cosine))
    assert cosine[far] < 0.999
    swapped = {key: rows[i] for i, key in enumerate(keys)}
    swapped[keys[0]], swapped[keys[far]] = rows[far], rows[0]
    out = tmp_path / "out"
    open_cache(out / "embedding_cache").put(swapped)

    def check_all(raw):
        raw["smoke"]["verify_pairs"] = len(keys)
        one_process(raw)

    config = write_config(tmp_path, check_all)
    assert script.main(smoke_args(out, config)) == script.EXIT_FAILED
    metrics = read(out / "metrics.json")
    assert metrics["status"].startswith("failed: CheckFailed")
    assert metrics["check"]["pass"] is False and metrics["check"]["min_cosine"] < 0.999
    worst = {row["pair"] for row in metrics["check"]["worst"][:2]}
    assert worst == {keys[0], keys[far]}


def write_one_part(cache, namespace, key="a" * 64, chunk=0):
    """One part file of one pair, as an encoding process writes it."""
    folder = cache.path / script.PARTS_DIR / "session-1" / "rank-0"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"part-{chunk:06d}.safetensors"
    script.write_part(path, [key], torch.ones(1, 64), None, namespace)
    return path


def test_a_part_of_another_namespace_stops_the_merge(tmp_path):
    """Embeddings of other encoder settings never enter the cache, and the part stays."""
    cache = open_cache(tmp_path / "cache")
    other = EmbeddingCache(
        tmp_path / "other", model_id="another/model", layer=2, pool="output", view_set_id="v"
    )
    part = write_one_part(cache, script.namespace_text(other))
    with pytest.raises(ValueError, match="another namespace"):
        script.merge_parts(cache, 16)
    assert len(cache) == 0 and part.is_file()


def test_a_part_is_deleted_only_after_the_put_that_holds_it(tmp_path, monkeypatch):
    """A failure between the two (a full disk, a kill) leaves the part for the next merge."""
    cache = open_cache(tmp_path / "cache")
    part = write_one_part(cache, script.namespace_text(cache))

    def fail(self, means, views=None):
        raise OSError("no space left on device")

    monkeypatch.setattr(EmbeddingCache, "put", fail)
    with pytest.raises(OSError, match="no space"):
        script.merge_parts(cache, 16)
    assert part.is_file() and len(cache) == 0
    monkeypatch.undo()
    assert script.merge_parts(cache, 16) == {"files": 1, "entries": 1}
    assert not part.exists() and "a" * 64 in cache


def test_the_seed_copy_leaves_temporary_files_behind(tmp_path):
    """A cut session's half-written files (``.name.tmp``) are never copied into the new cache."""
    seed = open_cache(tmp_path / "seed")
    seed.put({"b" * 64: torch.ones(64)})
    (seed.path / ".manifest.json.tmp").write_text("half written")
    write_one_part(seed, script.namespace_text(seed), key="c" * 64)
    hidden = seed.path / script.PARTS_DIR / "session-1" / "rank-0" / ".part-000001.safetensors.tmp"
    hidden.write_bytes(b"half written")
    cache = open_cache(tmp_path / "work")
    copied = script.seed_from(cache, tmp_path / "seed")
    assert copied["entries"] == 1 and copied["files"] == 4  # namespace, index, shard, one part
    names = {p.name for p in cache.path.rglob("*") if p.is_file()}
    assert not any(name.startswith(".") for name in names)
    assert {"namespace.json", "manifest.json", "part-000000.safetensors"} <= names


class FakeModel:
    """Stands in for a loaded model: records where it was asked to go."""

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self


def recorded_loads(monkeypatch):
    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return FakeModel()

    monkeypatch.setattr(script, "AutoModelForCausalLM", Loader)
    monkeypatch.setattr(script.AutoTokenizer, "from_pretrained", lambda path, **kwargs: "tokenizer")
    return calls


def test_a_cuda_process_loads_the_weights_straight_onto_its_gpu(monkeypatch):
    """Four processes staging about 8 GB of weights each in host memory could exhaust it: on CUDA
    the loader gets device_map on the process's GPU, as training's load_base does; the CPU (the
    smoke tests) loads without one; the smoke builds its stand-in without loading."""
    calls = recorded_loads(monkeypatch)
    cfg = script.load_config(CONFIG, smoke=False, overrides={})
    model, tokenizer = script.load_encoder_model(cfg, False, "/models/x", torch.device("cuda:2"))
    ((path, kwargs),) = calls
    assert path == "/models/x" and tokenizer == "tokenizer"
    assert kwargs["device_map"] == {"": torch.device("cuda:2")}
    assert kwargs["torch_dtype"] is torch.bfloat16 and kwargs["local_files_only"] is True
    assert kwargs["attn_implementation"] == "sdpa" and model.device == torch.device("cuda:2")
    calls.clear()
    script.load_encoder_model(cfg, False, "/models/x", torch.device("cpu"))
    ((_, kwargs),) = calls
    assert "device_map" not in kwargs
    calls.clear()
    smoke = script.load_config(CONFIG, smoke=True, overrides={})
    script.load_encoder_model(smoke, True, "unused", torch.device("cpu"))
    assert calls == []


@pytest.mark.parametrize(
    ("meminfo", "cgroup", "expected"),
    [
        ("MemTotal:  4096 kB\n", None, (4 * 2**20, None, 4 * 2**20)),
        ("MemTotal:  4096 kB\n", ("v2", "max"), (4 * 2**20, None, 4 * 2**20)),
        ("MemTotal:  4096 kB\n", ("v2", str(2**20)), (4 * 2**20, 2**20, 2**20)),
        ("MemTotal:  1024 kB\n", ("v2", str(8 * 2**20)), (2**20, 8 * 2**20, 2**20)),
        ("MemTotal:  4096 kB\n", ("v1", "9223372036854771712"), (4 * 2**20, None, 4 * 2**20)),
        ("MemTotal:  4096 kB\n", ("v1", str(3 * 2**20)), (4 * 2**20, 3 * 2**20, 3 * 2**20)),
        (None, ("v2", str(2**20)), (None, 2**20, 2**20)),
        (None, None, (None, None, None)),
    ],
    ids=[
        "no-cgroup",
        "v2-max",
        "v2-limit",
        "meminfo-smaller",
        "v1-unlimited",
        "v1-limit",
        "no-meminfo",
        "nothing",
    ],  # fmt: skip
)
def test_the_host_record_gives_the_smaller_of_memtotal_and_the_cgroup_limit(
    tmp_path, meminfo, cgroup, expected
):
    """MT-02's memory gate reads memory_limit_bytes: a container's cgroup limit (v2 memory.max,
    else v1 memory.limit_in_bytes, whose "no limit" is a huge number) can sit below MemTotal."""
    info = tmp_path / "meminfo"
    if meminfo is not None:
        info.write_text(meminfo)
    v2, v1 = tmp_path / "memory.max", tmp_path / "memory.limit_in_bytes"
    if cgroup is not None:
        (v2 if cgroup[0] == "v2" else v1).write_text(cgroup[1] + "\n")
    host = script.host_facts(info, (v2, v1))
    total, limit, usable = expected
    assert (host["memory_bytes"], host["cgroup_memory_limit_bytes"]) == (total, limit)
    assert host["memory_limit_bytes"] == usable and host["cpus"] >= 1
    assert host["cgroup_file"] == (None if cgroup is None else str(v2 if cgroup[0] == "v2" else v1))


# --- the config ----------------------------------------------------------------------------------


def test_the_mt_01_config_matches_its_brief():
    cfg = script.load_config(CONFIG, smoke=False, overrides={})
    assert cfg.run_id == "MT-01" and cfg.episodes is not None
    assert (cfg.layer, cfg.pool, cfg.n_perm, cfg.view_seed) == (24, "output", 1, 2)
    assert len(view_descriptors(cfg.n_perm, cfg.view_seed)) == 8
    assert (cfg.device, cfg.dtype, cfg.attn_implementation) == ("cuda", "bfloat16", "sdpa")
    assert (cfg.gpus, cfg.chunk_pairs, cfg.store_views, cfg.flush_pairs) == (4, 256, False, 4096)
    assert (cfg.verify_pairs, cfg.verify_min_cosine, cfg.seed_cache) == (256, 0.999, None)
    assert cfg.episodes.manifest_sha256.startswith("b1f2cd04")
    assert cfg.episodes.splits == ("train", "val") and cfg.episodes.max_tokens == 8192
    assert cfg.input_root == "/kaggle/input" and cfg.model_dir is None
    assert cfg.cache_dir == "/kaggle/working/embedding_cache"
    assert cfg.out_dir == "/kaggle/working/run_logs" and cfg.max_wall_s == 40800
    assert cfg.challenges is None and cfg.throughput_k == 0
    smoke = script.load_config(CONFIG, smoke=True, overrides={})
    assert smoke.episodes.root == str(FIXTURE / "episode_build") and smoke.gpus == 2


def test_every_mt_01_config_value_has_a_comment():
    """BRIEF A8: a comment above each key or above its group (consecutive keys)."""
    covered = False
    for line in CONFIG.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            covered = False
        elif stripped.startswith("#"):
            covered = True
        elif re.match(r"[a-z_0-9]+:", stripped):
            assert covered or "#" in line, line


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw.update(extra=1),
        lambda raw: raw.pop("gpus"),
        lambda raw: raw.update(challenges="x.json"),
        lambda raw: raw.update(device="cuda:0"),
        lambda raw: raw.update(gpus=0),
        lambda raw: raw.update(chunk_pairs=0),
        lambda raw: raw.update(verify_min_cosine=1.5),
        lambda raw: raw.update(verify_pairs=-1),
        lambda raw: raw.update(seed_cache=""),
        lambda raw: raw["episodes"].update(splits=["train", "train"]),
        lambda raw: raw["episodes"].update(splits=["test"]),
        lambda raw: raw["episodes"].update(splits=[]),
        lambda raw: raw["episodes"].update(manifest_sha256="b1f2"),
        lambda raw: raw["episodes"].pop("check_root"),
        lambda raw: raw["episodes"].update(extra=1),
        lambda raw: raw["smoke"].pop("seed"),
        lambda raw: raw["smoke"].update(subset=None),
        lambda raw: raw["smoke"].update(device="cuda:1"),
        # The GPU settings search for the model by its handle even when the smoke replaces it.
        lambda raw: raw.update(model_id="sorokin/qwen3_4b_grids15_sft139"),
    ],
)
def test_invalid_episode_configs_exit_2(tmp_path, edit):
    config = write_config(tmp_path, edit)
    assert script.main(smoke_args(tmp_path / "out", config)) == script.EXIT_CONFIG
    assert not (tmp_path / "out").exists()


def test_episode_flags_and_challenge_flags_stay_in_their_modes(tmp_path):
    out = tmp_path / "out"
    assert script.main([*smoke_args(out), "--challenges", "c.json"]) == script.EXIT_CONFIG
    challenge = smoke_args(out, CHALLENGE_CONFIG)
    assert script.main([*challenge, "--episodes-root", str(tmp_path)]) == script.EXIT_CONFIG
    assert script.main([*challenge, "--seed-cache", str(tmp_path)]) == script.EXIT_CONFIG
    assert not out.exists()


def test_the_command_line_names_the_episode_roots(tmp_path):
    overrides = {"episodes_root": "/mnt/build", "check_root": "/mnt/check", "seed_cache": "/mnt/c"}
    cfg = script.load_config(CONFIG, smoke=False, overrides=overrides)
    assert (cfg.episodes.root, cfg.episodes.check_root, cfg.seed_cache) == (
        "/mnt/build", "/mnt/check", "/mnt/c",
    )  # fmt: skip
    record = script.config_record(cfg)
    assert script.config_from_record(json.loads(json.dumps(record))) == cfg
    challenge = script.load_config(CHALLENGE_CONFIG, smoke=False, overrides={})
    assert set(script.config_record(challenge)).isdisjoint(script.EPISODE_FIELDS)
