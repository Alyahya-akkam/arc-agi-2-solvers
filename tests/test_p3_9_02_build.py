"""P3.9b: the build of run P3.9-02, P3.7-03's arms on the FOMAML-trained H of P3.9-01.

Checked here: ``configs/P3.9-02.yaml`` is ``configs/P3.7-03.yaml`` with H swapped (the same four
arms in the same order, DEV40, seed, queue order, scope, load-check tolerances), then amendment 1's
arm (g), ``hyper_merge_fit``, fifth, and the deadlines re-priced for it (the soft deadline, the
hard stop, ``finish_s``; the brief's Timeout from the hard stop), and nothing else; H's
``best.pt`` and attempt 2's run logs at the root of the dataset ``alyahya737/arc26-p3-9-01``
version 2, ``P3.9-01r``'s flat republish (task P3.9r: version 1 holds only the Modal run folder's
root files), its own notebook and code dataset. ``dataset_inputs`` (task P3.9b): an adapter
build's pinned datasets join ``dataset_sources`` as ``<owner>/<slug>/<version>``,
``{input:<owner>/<slug>}`` names their mount as it names a notebook output's, the environment
check finds a dataset under ``datasets/<owner>/<slug>``, the Run cell fills the arms' paths
from it, and the builder refuses malformed or unused ones; every earlier build keeps its
metadata. The build is byte-identical on a rebuild and passes the gateway; the brief's header
matches the config. The CPU smoke with
``--layout`` trains a FOMAML H as P3.9-01 did and runs the load check and every arm on it at this
config's paths. Nothing reaches Kaggle.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_t_arms_build_for_p3_9_02", REPO / "tests" / "test_arms_build.py"
)
B = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_t_arms_build_for_p3_9_02", B)
_spec.loader.exec_module(B)

nbb, bs, ops, am = B.nbb, B.bs, B.ops, B.am
CONFIG = "configs/P3.9-02.yaml"
BASELINE = "configs/P3.7-03.yaml"
BRIEF = REPO / "docs" / "prerun" / "P3.9-02.md"
# P3.9-01's output on Kaggle (task P3.9r, configs/modal_P3.9-01r.yaml): a dataset, version 2,
# best.pt and the training attempt's 11 run logs at its root beside MANIFEST.json.
P39_DATASET = "alyahya737/arc26-p3-9-01"
P39_VERSION = 2
P39_CHECKPOINT = "best.pt"
P39_LOGS = ""  # the dataset's root
P39_LOGS_PATH = f"{{input:{P39_DATASET}}}"


def test_the_config_is_p3_7_03s_with_h_swapped_arm_g_added_and_nothing_else():
    raw, base = B.raw_config(CONFIG), B.raw_config(BASELINE)
    changed = {key for key in {*raw, *base} if raw.get(key) != base.get(key)}
    assert changed == {
        "kaggle_notebook", "code_dataset", "adapters", "notebook_inputs", "dataset_inputs",
        "solver_end_s", "solver_kill_s",
    }  # fmt: skip
    adapters, before = raw["adapters"], base["adapters"]
    assert {k for k in adapters if adapters[k] != before[k]} == {
        "arms", "checkpoint", "load_check", "finish_s",
    }  # fmt: skip
    # The arms' reasons are text; the first four's names, sources, steps, and order are
    # P3.7-03's, so they pair with its arms; arm (g) comes last.
    strip = [{k: v for k, v in arm.items() if k != "reason"} for arm in adapters["arms"]]
    assert strip[:4] == [{k: v for k, v in arm.items() if k != "reason"} for arm in before["arms"]]
    assert strip[4:] == [{"name": "h-merge-fit", "source": "hyper_merge_fit"}]
    lc, lc_before = adapters["load_check"], before["load_check"]
    assert {k for k in lc if lc[k] != lc_before[k]} == {"run_logs"}
    # MT-03's output leaves; the fixed inputs stay; P3.9-01's dataset joins at version 2.
    assert raw["notebook_inputs"] == B.FIXED_INPUTS
    assert raw["dataset_inputs"] == {P39_DATASET: P39_VERSION}
    assert adapters["checkpoint"] == f"{{input:{P39_DATASET}}}/{P39_CHECKPOINT}"
    assert lc["run_logs"] == P39_LOGS_PATH


def test_the_config_parses_with_p3_7_03s_arms_subset_and_seeds_then_arm_g():
    config, base = B.parsed(CONFIG), B.parsed(BASELINE)
    arms = [(a.name, a.source, a.steps) for a in config.adapters.arms]
    assert arms[:4] == [(a.name, a.source, a.steps) for a in base.adapters.arms] == [
        ("base", "base", None), ("h", "hyper", None), ("sgd", "sgd", None),
        ("h-fit", "hyper_fit", None),
    ]  # fmt: skip
    assert arms[4:] == [("h-merge-fit", "hyper_merge_fit", None)]
    for key in ("subset", "workers", "hash_seed", "scope", "accelerator", "solver",
                "competition", "kernel_source_versions"):  # fmt: skip
        assert getattr(config, key) == getattr(base, key), key
    for key in ("views", "order_seed", "ce", "train_log", "cache", "episodes_root",
                "check_root", "manifest_sha256", "hypernet_config", "gen"):  # fmt: skip
        assert getattr(config.adapters, key) == getattr(base.adapters, key), key
    # Re-priced for one more trained arm per puzzle (the config's reasons, the brief section 6):
    # the soft deadline, finish_s, and the hard stop at the builder's minimum above them.
    assert (config.solver_end_s, config.adapters.finish_s) == (21600, 5200)
    assert config.solver_kill_s == 21600 + 5200 + nbb.ADAPTER_SELECTION_RESERVE_S == 27700
    assert config.solver_end_s > base.solver_end_s and config.solver_kill_s > base.solver_kill_s
    assert config.notebook_inputs == B.FIXED_INPUTS
    assert config.dataset_inputs == {P39_DATASET: P39_VERSION}
    assert config.kaggle_notebook == "alyahya737/arc26-h-val-fomaml" != base.kaggle_notebook
    assert config.code_dataset == "alyahya737/arc26-code-p39v" != base.code_dataset
    assert "adapter arm" in nbb.submission_blocker(config)


def test_the_metadata_attach_p3_9_01s_dataset_at_version_2():
    config = B.parsed(CONFIG)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)  # runs the gateway's metadata checks
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p39v", f"{P39_DATASET}/2"]
    assert meta["kernel_sources"] == ["sorokin/pip-install-unsloth-flash-patch/2"] + [
        f"{name}/{version}" for name, version in B.FIXED_INPUTS.items()
    ]
    info = nbb.build_info(
        config, tag="P3.9-02", commit="0" * 40, config_file=CONFIG, config_sha256="2" * 64,
        wheel="arc26-0.0.1-py3-none-any.whl", wheel_sha256="1" * 64,
    )  # fmt: skip
    assert info["dataset_inputs"] == {P39_DATASET: P39_VERSION}
    assert info["notebook_inputs"] == B.FIXED_INPUTS
    rows = bs._adapters_summary(config)
    assert rows[-1] == (
        f"| dataset_inputs (`{{input:<owner>/<slug>}}` is its folder) | `{P39_DATASET}` "
        "version 2 |"
    )


@pytest.mark.parametrize("run", [*sorted(B.CONFIGS.values()), BASELINE])
def test_the_earlier_adapter_builds_attach_what_they_did(run):
    config = B.parsed(run)
    assert config.dataset_inputs == {}
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    assert bs.kernel_metadata(config, pulled)["dataset_sources"] == [config.code_dataset]
    info = nbb.build_info(
        config, tag="t", commit="0" * 40, config_file=run, config_sha256="2" * 64,
        wheel="arc26-0.0.1-py3-none-any.whl", wheel_sha256="1" * 64,
    )  # fmt: skip
    assert "dataset_inputs" not in info
    assert not any(row.startswith("| dataset_inputs") for row in bs._adapters_summary(config))


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"dataset_inputs": {P39_DATASET: 0}}, "dataset_inputs must map"),
        ({"dataset_inputs": ["alyahya737/arc26-p3-9-01"]}, "dataset_inputs must map"),
        ({"dataset_inputs": {P39_DATASET: 1, "alyahya737/arc26-unused": 1}}, "never reach"),
        ({"dataset_inputs": {}}, "neither notebook_inputs nor dataset_inputs|does not attach"),
        (
            {"dataset_inputs": {P39_DATASET: 1, "alyahya737/arc26-code-p39v": 1}},
            "the code dataset",
        ),
        (
            {"notebook_inputs": {**B.FIXED_INPUTS, P39_DATASET: 1}},
            "in both notebook_inputs and dataset_inputs",
        ),
    ],
)
def test_the_dataset_input_refusals(changes, match):
    B.refused(match, changes, name=CONFIG)


def test_dataset_inputs_need_adapters():
    raw = yaml.safe_load((REPO / "configs" / "submission_P2.6-01.yaml").read_text())
    raw["dataset_inputs"] = {P39_DATASET: 1}
    with pytest.raises(nbb.BuildError, match="dataset_inputs attach datasets for the adapter"):
        nbb.parse_config(yaml.safe_dump(raw).encode(), Path("c.yaml"))


def namespace_for(tmp_path, monkeypatch):
    """The Config and Helpers cells of this build, run in process on a simulated tree."""
    config = B.parsed(CONFIG)
    info = nbb.build_info(
        config, tag="P3.9-02", commit="0" * 40, config_file=CONFIG, config_sha256="2" * 64,
        wheel="arc26-0.0.1-py3-none-any.whl", wheel_sha256="1" * 64,
    )  # fmt: skip
    notebook = nbb.build_notebook(config, info, B.SUBMIT_STAND_IN)
    code = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
    monkeypatch.setenv("ARC26_KAGGLE_ROOT", str(tmp_path))
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    namespace = {"__name__": "__main__"}
    for cell in ("arc26-02-config", "arc26-04-helpers"):
        exec(compile(code[cell], cell, "exec"), namespace)
    return config, namespace, code


def kaggle_mounts(tmp_path):
    """The inputs as Kaggle mounts them: notebooks/<owner>/<slug> and datasets/<owner>/<slug>
    (P3.7-03's session log: ``/kaggle/input/datasets/alyahya737/arc26-code-p37f/``)."""
    found = {}
    for name in B.FIXED_INPUTS:
        folder = tmp_path / "input" / "notebooks" / name
        folder.mkdir(parents=True)
        (folder / "marker").write_text(name)
        found[name] = folder
    dataset = tmp_path / "input" / "datasets" / P39_DATASET
    (dataset / P39_CHECKPOINT).parent.mkdir(parents=True)
    (dataset / P39_CHECKPOINT).write_bytes(b"stand-in")
    (dataset / P39_LOGS).mkdir(parents=True, exist_ok=True)
    (dataset / P39_LOGS / "status.json").write_text("{}")
    (dataset / "MANIFEST.json").write_text("{}")  # version 2's root holds it too
    found[P39_DATASET] = dataset
    return found


def test_the_environment_check_finds_the_dataset_and_the_outputs(tmp_path, monkeypatch, capsys):
    _, namespace, code = namespace_for(tmp_path, monkeypatch)
    folders = kaggle_mounts(tmp_path)
    exec(compile(code["arc26-06-validator"], "validator", "exec"), namespace)
    with pytest.raises(RuntimeError):  # no GPU, no data on the simulated tree: the check stops
        exec(compile(code["arc26-08-environment"], "environment", "exec"), namespace)
    assert namespace["INPUT_DIRS"] == folders
    printed = capsys.readouterr().out
    assert f"found dataset {P39_DATASET} (version 2)" in printed
    assert "found notebook output alyahya737/arc26-mt-cache (version 1)" in printed


def test_the_run_cell_reads_best_pt_and_the_logs_from_the_dataset(tmp_path, monkeypatch):
    import arc26.runner as runner

    config, namespace, code = namespace_for(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        runner, "run_solver",
        lambda cfg, end_time, python=None, timeout=None: calls.append(cfg) or 0,
    )  # fmt: skip
    namespace.update(
        INSTALLED=True, PYTHON="/usr/local/bin/python", DATA_DIR=tmp_path / "data",
        MODEL_DIR=tmp_path / "model",
        CHALLENGES=tmp_path / "data" / "arc-agi_evaluation_challenges.json",
    )  # fmt: skip
    namespace["INPUT_DIRS"].update(kaggle_mounts(tmp_path))
    exec(compile(code["arc26-12-run"], "run", "exec"), namespace)
    (cfg,) = calls
    dataset = tmp_path / "input" / "datasets" / P39_DATASET
    assert cfg.adapters.checkpoint == str(dataset / P39_CHECKPOINT)
    assert cfg.adapters.load_check.run_logs == str(dataset / P39_LOGS) == str(dataset)
    assert cfg.adapters.cache.endswith("notebooks/alyahya737/arc26-mt-cache/embedding_cache")
    assert [a.source for a in cfg.adapters.arms] == [
        "base", "hyper", "sgd", "hyper_fit", "hyper_merge_fit",
    ]  # fmt: skip
    assert len(cfg.subset) == 40 and cfg.adapters.train_log
    assert config.solver_kill_s == (
        config.solver_end_s + config.adapters.finish_s + nbb.ADAPTER_SELECTION_RESERVE_S
    )


@pytest.fixture(scope="module")
def p39_builds(tmp_path_factory):
    """dist/P3.9-02/ twice from one throwaway repository."""
    if not B.BUILD_TESTS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = B.BUILD_TESTS.snapshot(tmp_path_factory.mktemp("repo"))
    first = B.BUILD_TESTS.read_files(bs.build("P3.9-02", repo / CONFIG, repo))
    again = bs.build("P3.9-02", repo / CONFIG, repo)
    return again, first


def test_the_build_is_rebuilt_byte_for_byte_and_passes_the_gateway(p39_builds):
    folder, first = p39_builds
    assert B.BUILD_TESTS.read_files(folder) == first
    meta = ops.load_kernel_metadata(folder.resolve())
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder((folder / "arc26-code").resolve())
    assert meta["id"] == "alyahya737/arc26-h-val-fomaml"
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p39v", f"{P39_DATASET}/2"]
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | no, not for submission:" in manifest
    assert f"- dataset `{P39_DATASET}/2`, passed to the adapter arms as" in manifest
    namespace = {}
    exec(B.BUILD_TESTS.code(folder)["arc26-02-config"], namespace)
    assert namespace["BUILD"]["dataset_inputs"] == {P39_DATASET: P39_VERSION}
    assert namespace["ADAPTERS"]["checkpoint"] == f"{{input:{P39_DATASET}}}/{P39_CHECKPOINT}"


def test_the_brief_header_matches_its_config():
    if not BRIEF.is_file():
        pytest.skip("the brief is not in this tree")
    text = BRIEF.read_text(encoding="utf-8")
    assert "TODO" not in text
    header = ops.brief_header(text)
    config = B.parsed(CONFIG)
    assert header["Run-id"] == "P3.9-02" and header["Config"] == CONFIG
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Accelerator"] == config.accelerator == "NvidiaL4"
    assert header["Internet"] == "off"
    assert int(header["Timeout"]) == config.solver_kill_s + 300 + 2700


def test_the_layout_smoke_reads_a_fomaml_best_pt_at_the_configs_paths(tmp_path):
    """The A8 smoke with ``--layout``: P3.9-01's lineage on the CPU (the MT smoke, then the P3.9
    smoke warm-started from its step 8), best.pt and the logs at this config's relative paths,
    then the load check and every arm on that file."""
    for module in ("torch", "transformers", "peft", "safetensors"):
        pytest.importorskip(module)
    rp = B.load("_t_h_validation_report_layout", REPO / "scripts" / "h_validation_report.py")
    assert rp.layout_paths(REPO / CONFIG) == (P39_DATASET, P39_CHECKPOINT, P39_LOGS)
    result = rp.smoke(tmp_path / "smoke", REPO / CONFIG)
    assert result["tables"][0]["puzzles"] == 3
    run = rp.read_run(tmp_path / "smoke" / "run_logs")
    check = run["load_check"]
    root = tmp_path / "smoke" / "inputs" / P39_DATASET
    assert check["verdict"] == "pass" and check["checkpoint"] == str(root / P39_CHECKPOINT)
    assert check["checkpoint_step"] == check["logged_best_step"]
    import torch

    payload = torch.load(root / P39_CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["kind"] == "best" and payload["objective"]["objective"] == "fomaml"
    assert "objective" not in payload["config"]  # the config P3.7's loader parses
    assert {r["status"] for rows in run["records"].values() for r in rows.values()} == {"done"}
    assert run["summary"]["exit"] == 0
