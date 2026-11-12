"""The weights' warm-up of a shot build (task P6.9c): the copy, its checks, and the engine's path.

Run ``P6.9-02`` read Fault: the shot's weights, on a notebook-output mount Kaggle now serves through
FUSE.GCSFUSE, loaded in 802.41 s, past the 900 s boot limit. A shot build with ``agent.warmup``
copies them to local scratch while NVARC solves (``copy_weights``, run as the script
``WARMUP_SCRIPT`` by ``start_warmup``), and the agent stage gives the engine the verified copy or
the input mount with vLLM's ``--safetensors-load-strategy prefetch`` (``warmup_weights``). Against
the task brief (``docs/tasks/P6.9c.md``) this file checks, on a fake model folder on CPU:

- the copy: every file in parallel pieces, at its size and its sha256, the status JSON's fields;
- a wrong sha256, a missing source file, too little free scratch: no local copy;
- the script the notebook embeds runs on its own, and the path choice takes the copy it made;
- the fallback: no warm-up, a failed copy, a stalled copy (stopped), each the input mount with the
  prefetch strategy;
- the engine's command line (``arc26.agent.server.vllm_command`` on the shot's own config) with the
  local path, and with the fallback's flag;
- the build: ``agent.warmup``'s checks, and Submission 4's configs: ``P6.9-03`` is ``P6.9-02``
  with only the subset and the warm-up added, its CPU twin ``P6.9-02-cpu`` with only the warm-up.

The notebook's runs on the simulated Kaggle tree are ``tests/test_warmup_notebook.py``'s.
"""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


SB = load("_warmup_shot_build", REPO / "tests" / "test_shot_build.py")
HYB, nbb = SB.HYB, SB.nbb
P5_CONFIG = "configs/submission_P6.9-03.yaml"
P5_CPU_CONFIG = "configs/submission_P6.9-03-cpu.yaml"
P5_CONFIGS = (P5_CPU_CONFIG, P5_CONFIG)
FAST = {
    "delay_s": 0, "workers": 3, "piece_mib": 1, "nice": 0, "progress_s": 1, "reserve_gb": 0,
    "wait_s": 30, "stall_s": 5, "fallback_load_strategy": "prefetch",
}  # fmt: skip


def model_folder(root, sizes=None, wrong=()):
    """A fake weights output: files of the given sizes in ``qwen38-27b-int4`` and the output's
    MANIFEST.json beside it with each file's size and sha256 (``wrong`` names files whose sha256
    the manifest misstates)."""
    sizes = sizes or {"config.json": 2, "model.safetensors": 3 * (1 << 20) + 12345, "sub/t.json": 0}
    folder = root / "qwen38-27b-int4"
    rows = []
    for name, size in sizes.items():
        data = (hashlib.sha256(name.encode()).digest() * (size // 32 + 1))[:size]
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        rows.append({"name": name, "size": size, "sha256": "0" * 64 if name in wrong else digest})
    (root / "MANIFEST.json").write_text(json.dumps({"ok": True, "files": rows}))
    return folder


def spec_file(tmp_path, source, target, **changes):
    rows = json.loads((source.parent / "MANIFEST.json").read_text())["files"]
    spec = {
        "source": str(source), "target": str(target), "files": rows,
        "status": str(tmp_path / "warmup.json"), "delay_s": 0, "workers": 3,
        "piece_bytes": 1 << 20, "block_bytes": 1 << 16, "nice": 0, "progress_s": 1, **changes,
    }  # fmt: skip
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def status(tmp_path):
    return json.loads((tmp_path / "warmup.json").read_text())


def script():
    """The copier as the notebook embeds it: WARMUP_SCRIPT's text."""
    namespace = {}
    exec(nbb.warmup_constants_source(), namespace)  # noqa: S102
    return namespace["WARMUP_SCRIPT"]


# The copy and its checks.


def test_the_copy_holds_every_file_at_its_size_and_sha256_and_says_so(tmp_path):
    source = model_folder(tmp_path / "input")
    target = tmp_path / "local"
    target.mkdir()
    assert nbb.copy_weights(str(spec_file(tmp_path, source, target))) == 0
    for name in ("config.json", "model.safetensors", "sub/t.json"):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    found = status(tmp_path)
    assert (found["state"], found["ok"], found["errors"]) == ("done", True, [])
    assert found["files"] == 3 and found["bytes"] == 2 + 3 * (1 << 20) + 12345
    assert found["copied_bytes"] == found["bytes"] == found["hashed_bytes"]
    # Sorted in the final status, whatever order the threads finished in.
    assert found["verified"] == ["config.json", "model.safetensors", "sub/t.json"]
    assert found["size_only"] == []
    assert found["seconds"] >= found["copy_s"] >= 0 and found["verify_s"] >= 0
    assert found["start"] <= found["end"] and found["priority"]["nice"] == os.nice(0)
    assert "ionice" in found["priority"]


def test_a_wrong_sha256_or_a_missing_source_file_leaves_no_usable_copy(tmp_path):
    source = model_folder(tmp_path / "input", wrong=("model.safetensors",))
    target = tmp_path / "local"
    target.mkdir()
    assert nbb.copy_weights(str(spec_file(tmp_path, source, target))) == 1
    found = status(tmp_path)
    assert found["state"] == "done" and found["ok"] is False
    assert any("model.safetensors: sha256" in error for error in found["errors"])
    (source / "config.json").unlink()
    assert nbb.copy_weights(str(spec_file(tmp_path, source, target))) == 1
    assert any("config.json" in error for error in status(tmp_path)["errors"])


def test_a_manifest_row_without_sha256_is_checked_at_its_size_and_named(tmp_path):
    source = model_folder(tmp_path / "input")
    manifest = json.loads((source.parent / "MANIFEST.json").read_text())
    manifest["files"][0]["sha256"] = None
    (source.parent / "MANIFEST.json").write_text(json.dumps(manifest))
    target = tmp_path / "local"
    target.mkdir()
    assert nbb.copy_weights(str(spec_file(tmp_path, source, target))) == 0
    assert status(tmp_path)["size_only"] == ["config.json"]


def test_the_embedded_script_copies_and_the_engine_takes_the_local_copy(tmp_path):
    """start_warmup runs WARMUP_SCRIPT in a session of its own; warmup_weights then names the
    verified copy, with no load strategy, and the copier has exited 0."""
    source = model_folder(tmp_path / "input")
    scratch = tmp_path / "scratch"
    process, record = nbb.start_warmup(
        str(source.parent / "MANIFEST.json"), str(source), str(scratch), FAST, script(),
        dict(os.environ),
    )  # fmt: skip
    assert record["started"] and record["files"] == 3 and process is not None
    assert record["target"] == str(scratch / "weights" / "qwen38-27b-int4")
    assert os.getpgid(process.pid) == process.pid  # a session of its own
    weights = nbb.warmup_weights(process, record, str(source), FAST, 0.05, 5)
    assert weights["source"] == "local" and weights["reason"] is None, weights
    assert weights["model_dir"] == record["target"] and weights["load_strategy"] is None
    assert weights["warmup"]["exit"] == 0 and weights["warmup"]["stopped"] is False
    assert weights["warmup"]["status"]["ok"] is True
    assert (Path(record["target"]) / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()


def test_too_little_free_scratch_starts_no_copy_and_the_engine_falls_back(tmp_path):
    source = model_folder(tmp_path / "input")
    settings = {**FAST, "reserve_gb": 1000}
    process, record = nbb.start_warmup(
        str(source.parent / "MANIFEST.json"), str(source), str(tmp_path / "scratch"), settings,
        script(), dict(os.environ),
    )  # fmt: skip
    if record.get("free_bytes", 0) >= 10**12:
        pytest.skip("this machine has 1,000 GB free")
    assert process is None and not record["started"] and "free" in record["reason"]
    weights = nbb.warmup_weights(process, record, str(source), settings, 0.05, 5)
    assert (weights["source"], weights["model_dir"]) == ("input", str(source))
    assert weights["load_strategy"] == "prefetch" and "did not start" in weights["reason"]


def test_a_failed_copy_falls_back_to_the_input_with_the_prefetch_strategy(tmp_path):
    source = model_folder(tmp_path / "input", wrong=("config.json",))
    process, record = nbb.start_warmup(
        str(source.parent / "MANIFEST.json"), str(source), str(tmp_path / "scratch"), FAST,
        script(), dict(os.environ),
    )  # fmt: skip
    weights = nbb.warmup_weights(process, record, str(source), FAST, 0.05, 5)
    assert (weights["source"], weights["load_strategy"]) == ("input", "prefetch")
    assert weights["model_dir"] == str(source) and "sha256" in weights["reason"]
    assert weights["warmup"]["exit"] == 1


def test_a_stalled_copy_is_stopped_and_the_engine_falls_back(tmp_path):
    """A copy that moves no byte for stall_s (here: still in its delay) is killed with its
    process group, and the engine loads from the input."""
    source = model_folder(tmp_path / "input")
    settings = {**FAST, "delay_s": 600, "stall_s": 5}
    process, record = nbb.start_warmup(
        str(source.parent / "MANIFEST.json"), str(source), str(tmp_path / "scratch"), settings,
        script(), dict(os.environ),
    )  # fmt: skip
    began = time.monotonic()
    weights = nbb.warmup_weights(process, record, str(source), settings, 0.2, 5)
    assert 5 <= time.monotonic() - began < 20
    assert weights["source"] == "input" and "no progress for 5 s" in weights["reason"]
    assert weights["warmup"]["stopped"] is True and process.poll() is not None
    assert weights["warmup"]["status"]["state"] == "waiting"


def test_a_copy_past_wait_s_is_stopped_and_the_engine_falls_back(tmp_path):
    source = model_folder(tmp_path / "input")
    settings = {**FAST, "delay_s": 600, "wait_s": 1, "stall_s": 600}
    process, record = nbb.start_warmup(
        str(source.parent / "MANIFEST.json"), str(source), str(tmp_path / "scratch"), settings,
        script(), dict(os.environ),
    )  # fmt: skip
    weights = nbb.warmup_weights(process, record, str(source), settings, 0.2, 5)
    assert weights["source"] == "input" and "not done after 1 s" in weights["reason"]
    assert process.poll() is not None


def test_a_copy_changed_after_its_check_is_not_used(tmp_path):
    source = model_folder(tmp_path / "input")
    process, record = nbb.start_warmup(
        str(source.parent / "MANIFEST.json"), str(source), str(tmp_path / "scratch"), FAST,
        script(), dict(os.environ),
    )  # fmt: skip
    process.wait(timeout=30)
    (Path(record["target"]) / "config.json").write_bytes(b"{}{}")
    weights = nbb.warmup_weights(process, record, str(source), FAST, 0.05, 5)
    assert weights["source"] == "input" and "config.json" in weights["reason"]


def test_stop_warmup_without_a_warmup_reports_nothing_ran():
    assert nbb.stop_warmup(None, None, 1) == {"exit": None, "stopped": False, "status": None}


def test_the_copier_script_is_standard_library_only_and_runs_with_site_off(tmp_path):
    source = model_folder(tmp_path / "input")
    target = tmp_path / "local"
    target.mkdir()
    path = tmp_path / "warm_weights.py"
    path.write_text(script())
    done = subprocess.run(
        [sys.executable, "-S", "-I", str(path), str(spec_file(tmp_path, source, target))],
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert status(tmp_path)["ok"] is True


# The engine's command line on the shot's own config.


@pytest.mark.parametrize("fallback", [False, True], ids=["local", "input"])
def test_the_engine_command_takes_the_chosen_folder_and_the_fallback_flag(tmp_path, fallback):
    """What the notebook writes for the shot (the run config, with the fallback's flag in
    server.extra_args) gives vLLM's command line through the shot's own config loader."""
    shot = pytest.importorskip("arc26.agent.shot")
    from arc26.agent.server import vllm_command

    data = copy.deepcopy(SB.resolved(P5_CONFIG).agent.config_data)
    folder = "/kaggle/input/arc26-dl-qwen38-int4/qwen38-27b-int4" if fallback else "/tmp/w/q"
    if fallback:
        data["server"]["extra_args"] = [
            *(data["server"].get("extra_args") or []), "--safetensors-load-strategy", "prefetch"
        ]
    path = tmp_path / "agent_config.json"
    path.write_text(json.dumps(data))
    config = shot.load_config(path)
    command = vllm_command("/tmp/v/bin/vllm", folder, config.server)
    assert command[:3] == ["/tmp/v/bin/vllm", "serve", folder]
    assert command[command.index("--served-model-name") + 1] == "qwen38-27b-int4"
    flags = command[command.index("--safetensors-load-strategy") :] if fallback else []
    assert flags == (["--safetensors-load-strategy", "prefetch"] if fallback else [])
    assert ("--safetensors-load-strategy" in command) is fallback
    assert config.server.boot_timeout_s == 900


# The build: agent.warmup's checks and Submission 4's configs.


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"entry": "arc26.agent.run"}, "warmup needs entry arc26.agent.shot"),
        ({"warmup": {**FAST, "fallback_load_strategy": "eager"}}, "fallback_load_strategy"),
        ({"warmup": {**FAST, "workers": 0}}, "workers must be an integer from 1"),
        ({"warmup": {**FAST, "nice": True}}, "nice must be an integer"),
        ({"warmup": {k: v for k, v in FAST.items() if k != "wait_s"}}, "exactly the keys"),
    ],
    ids=["not-a-shot", "strategy", "workers", "bool", "missing"],
)
def test_a_malformed_warmup_is_refused(change, message):
    agent = {"warmup": FAST, **change}
    with pytest.raises(nbb.BuildError, match=message):
        SB.shot_config(agent)


def test_a_warmup_shot_build_differs_only_by_the_warmup_edits_and_helpers():
    """Without warmup a shot build is as before (the P6.5 goldens pin the hybrids); with it,
    every cell but the agent check is the shot's with WARMUP_EDITS applied, and the agent check
    also gains the warm-up's helpers and constants."""
    plain = SB.cells(SB.shot_config())
    warm = SB.cells(SB.shot_config({"warmup": FAST}))
    assert set(plain) == set(warm)
    for name, text in plain.items():
        edited = nbb._edited(text, nbb.WARMUP_EDITS.get(name, ()), name, "WARMUP_EDITS")
        if name == "agent-check":
            helpers = "\n\n\n".join(
                SB.inspect.getsource(h).rstrip() for h in nbb.WARMUP_RUNTIME_HELPERS
            )
            assert helpers in warm[name] and nbb.warmup_constants_source() in warm[name]
            assert "def copy_weights(" in warm[name] and "WARMUP_SCRIPT = '''" in warm[name]
        elif name in ("config", "config-note"):
            continue
        else:
            assert warm[name] == edited, name
    assert SB.cells(SB.shot_config())["agent"] == plain["agent"]


def test_the_warmup_cells_compile():
    for name, text in SB.cells(SB.shot_config({"warmup": FAST})).items():
        if not name.endswith("note"):
            compile(text, name, "exec")


@pytest.mark.parametrize("name", P5_CONFIGS)
def test_every_p6_9_03_build_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if HYB.re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == 18 + 13 + 8 + 10, len(keys)
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


def test_p6_9_03_is_p6_9_02_with_only_the_subset_and_the_warmup():
    """The scored config: P6.9-02's (Code-commit 261d554's shot v2 and NVARC, every deadline,
    the code dataset, the notebook) with only agent.warmup added and the development subset;
    still fit to submit."""
    new, old = HYB.parsed(P5_CONFIG), HYB.parsed(SB.P4_CONFIG)
    assert new.subset == ("291dc1e1", "4e34c42c", "97d7923e", "981571dc") and not new.dev240
    assert new.agent.warmup == {
        "delay_s": 180, "workers": 8, "piece_mib": 256, "nice": 19, "progress_s": 30,
        "reserve_gb": 40, "wait_s": 900, "stall_s": 120, "fallback_load_strategy": "prefetch",
    }  # fmt: skip
    assert new._replace(subset=None, dev240=True, agent=new.agent._replace(warmup=None)) == old
    assert nbb.submission_blocker(new) is None
    assert new.agent.warmup["wait_s"] == SB.resolved(P5_CONFIG).agent.config_data["server"][
        "boot_timeout_s"
    ]


def test_p6_9_03_cpu_is_p6_9_02_cpu_with_the_warmup_at_no_delay():
    cpu, old = HYB.parsed(P5_CPU_CONFIG), HYB.parsed(SB.P4_CPU_CONFIG)
    gpu = HYB.parsed(P5_CONFIG)
    assert cpu.agent.warmup == {**gpu.agent.warmup, "delay_s": 0}
    assert cpu._replace(agent=cpu.agent._replace(warmup=None)) == old
    assert "no accelerator" in nbb.submission_blocker(cpu)
    assert cpu.kaggle_notebook not in HYB.ops.briefed_notebooks()


def test_the_subset_applies_in_development_mode_only():
    """A competition rerun of this version solves every hidden task with Submission 4's times:
    the solver's queue skips the subset in rerun mode and the agent stage passes --task-ids only
    outside a competition rerun."""
    solver = (REPO / "arc26" / "solver.py").read_text(encoding="utf-8")
    assert (
        "        if not rerun_mode:\n"
        '            if cfg["subset"] is not None and key not in cfg["subset"]:\n'
    ) in solver
    agent = SB.cells(SB.shot_config({"warmup": FAST}))["agent"]
    assert agent.count("--task-ids") == 1 and "    if SUBSET and not COMPETITION:\n" in agent


@pytest.mark.parametrize("name", P5_CONFIGS)
def test_two_builds_of_each_p6_9_03_config_are_byte_identical(name):
    first = SB.hybrid_golden_files(SB.bs, name)
    assert first == SB.hybrid_golden_files(SB.bs, name)
    sources = HYB.cell_sources(json.loads(first[nbb.NOTEBOOK_FILE]))
    assert "start_warmup(" in sources["agent-check"] and "warmup_weights(" in sources["agent"]
    meta = json.loads(first[SB.bs.METADATA_FILE])
    assert "alyahya737/arc26-code-p69" in meta["dataset_sources"]
