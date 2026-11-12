"""P3.7 amendment 8: the build of run P3.7-03 (arms (c) and (f)) and the adapters' write stage.

Checked here: ``configs/P3.7-03.yaml`` parses and resolves (arms ``sgd`` and ``hyper_fit``, no
steps, ``train_log`` on, H's scope, P3.7-01's inputs and DEV40, its own notebook and code
dataset, the deadlines' rule); ``train_log`` is optional, so P3.7-01's and P3.7-02's settings keep
their keys; the builds are byte-identical on a rebuild and pass the gateway; the brief's header
matches the config with the pre-script allowance. The write stage of an adapters build makes no
top-level selection (``P3.7-01``'s record: ``make_submission`` over the empty
``inference_outputs/`` raised a zero-size ``ValueError``), a plain build still does, and a
competition rerun of an adapters build does too. Nothing reaches Kaggle.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_t_arms_build_for_fit", REPO / "tests" / "test_arms_build.py"
)
B = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_t_arms_build_for_fit", B)
_spec.loader.exec_module(B)

nbb, bs, ops, am = B.nbb, B.bs, B.ops, B.am
CONFIG = "configs/P3.7-03.yaml"
BRIEF = REPO / "docs" / "prerun" / "P3.7-03.md"


def test_the_fit_config_runs_a_b_c_f_with_hs_scope_and_p3_7_01s_inputs():
    config = B.parsed(CONFIG)
    adapters = config.adapters
    # Revision 2: (a) and (b) for G4's second read, then (c) and (f), (c) before (f).
    assert [(a.name, a.source, a.steps) for a in adapters.arms] == [
        ("base", "base", None), ("h", "hyper", None),
        ("sgd", "sgd", None), ("h-fit", "hyper_fit", None),
    ]  # fmt: skip
    # The arms (a) to (c) are P3.7-01's, by name and source.
    dev_arms = [(a.name, a.source) for a in B.parsed(B.CONFIGS["P3.7-01"]).adapters.arms]
    assert [(a.name, a.source) for a in adapters.arms][:3] == dev_arms[:3]
    assert adapters.train_log is True and adapters.views is None and adapters.gen is None
    hypernet = adapters.hypernet_config
    assert (config.scope.lora_r, list(config.scope.lora_targets)) == (
        hypernet["rank"], hypernet["target_modules"],
    )  # fmt: skip
    assert hypernet["alpha"] == nbb.SOLVER_LORA_ALPHA
    dev = B.parsed(B.CONFIGS["P3.7-01"])
    assert config.subset == dev.subset and config.scope == dev.scope
    assert config.accelerator == "NvidiaL4"
    # The fixed inputs are P3.7-01's; H's checkpoint and logs are whatever the config names (the
    # three-key swap), each attached at a version.
    held = set(B.h_outputs(CONFIG))
    inputs = config.notebook_inputs
    assert {k: v for k, v in inputs.items() if k not in held} == B.FIXED_INPUTS
    assert held <= set(inputs) and all(inputs[k] >= 1 for k in held)
    assert adapters.load_check["max_abs"] == dev.adapters.load_check["max_abs"]
    assert adapters.load_check["max_mean"] == dev.adapters.load_check["max_mean"]
    assert config.kaggle_notebook == "alyahya737/arc26-h-val-fit"
    assert config.code_dataset == "alyahya737/arc26-code-p37f" != dev.code_dataset
    reserve = config.solver_end_s + adapters.finish_s + nbb.ADAPTER_SELECTION_RESERVE_S
    assert config.solver_kill_s == reserve
    settings = adapters.settings()
    assert settings["train_log"] is True and list(settings)[:-1] == list(nbb.ADAPTER_KEYS)
    filled = {
        **settings,
        "arms": [{k: v for k, v in a.items() if k != "reason"} for a in settings["arms"]],
        "outputs_root": "/o", "log_dir": "/l", "cache_dir": "/c",
    }  # fmt: skip
    filled["load_check"] = {**filled["load_check"], "dev4_challenges": "/d.json"}
    arms = am.Arms.from_mapping(filled)
    assert arms.train_log
    assert [a.source for a in arms.arms] == ["base", "hyper", "sgd", "hyper_fit"]


@pytest.mark.parametrize("logs", [B.MT_03_OUTPUT, "alyahya737/arc26-mt-train-3"])
def test_the_fit_config_takes_the_real_three_key_swap(logs):
    """MT-03's output holds best.pt; the logs from it or from a later session (MT-04's)."""
    raw = B.raw_config(CONFIG)
    adapters = raw["adapters"]
    load_check = {**adapters["load_check"], "run_logs": f"{{input:{logs}}}/run_logs"}
    inputs = {B.MT_03_OUTPUT: 1, logs: 1, **B.FIXED_INPUTS}
    text = B.config_text(
        {"notebook_inputs": inputs}, CONFIG,
        {"checkpoint": f"{{input:{B.MT_03_OUTPUT}}}/checkpoints/best.pt", "load_check": load_check},
    )  # fmt: skip
    config = nbb.resolve_adapters(nbb.parse_config(text, Path(CONFIG)), B.read)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    sources = bs.kernel_metadata(config, pulled)["kernel_sources"]
    assert f"{B.MT_03_OUTPUT}/1" in sources and f"{logs}/1" in sources
    for stale in set(B.h_outputs(CONFIG)) - {B.MT_03_OUTPUT, logs}:
        assert not any(source.startswith(f"{stale}/") for source in sources)
    settings = config.adapters.settings()
    assert settings["checkpoint"] == f"{{input:{B.MT_03_OUTPUT}}}/checkpoints/best.pt"
    assert settings["load_check"]["run_logs"] == f"{{input:{logs}}}/run_logs"
    assert settings["train_log"] is True and [a["source"] for a in settings["arms"]] == [
        "base", "hyper", "sgd", "hyper_fit",
    ]  # fmt: skip


@pytest.mark.parametrize("run", sorted(B.CONFIGS))
def test_train_log_is_optional_and_absent_from_the_earlier_builds(run):
    settings = B.parsed(B.CONFIGS[run]).adapters.settings()
    assert "train_log" not in settings and list(settings) == list(nbb.ADAPTER_KEYS)


@pytest.mark.parametrize(
    ("adapters", "match"),
    [
        ({"train_log": "yes"}, "train_log must be true or false"),
        ({"extra": 1}, "exactly the keys"),
        ({"arms": [{"name": "f", "source": "hyper_fit", "reason": "r", "steps": 128}]},
         "steps apply"),
    ],
)  # fmt: skip
def test_the_fit_config_refusals(adapters, match):
    B.refused(match, adapters=adapters, name=CONFIG)


def write_namespace(run, tmp_path, monkeypatch, competition=False):
    """The Config, Helpers, and validator cells, then the write cell's inputs set by hand.

    Returns the namespace and the code by cell name (the id less its ``arc26-NN-`` prefix).
    """
    config = B.parsed(run)
    info = nbb.build_info(
        config, tag="t", commit="0" * 40, config_file=run, config_sha256="2" * 64,
        wheel="arc26-0.0.1-py3-none-any.whl", wheel_sha256="1" * 64,
    )  # fmt: skip
    submit = (REPO / "arc26" / "submit.py").read_text(encoding="utf-8")
    notebook = nbb.build_notebook(config, info, submit)
    code = {
        c["id"].split("-", 2)[2]: "".join(c["source"])
        for c in notebook["cells"] if c["cell_type"] == "code"
    }  # fmt: skip
    monkeypatch.setenv("ARC26_KAGGLE_ROOT", str(tmp_path))
    if competition:
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    else:
        monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    namespace = {"__name__": "__main__"}
    for cell in ("config", "helpers", "validator"):
        exec(compile(code[cell], cell, "exec"), namespace)
    challenges = tmp_path / "challenges.json"
    challenges.write_text(json.dumps({"t1": {"train": [], "test": [{"input": [[1]]}]}}))
    namespace.update(CHALLENGES=challenges, ARM_RECORDS=[], STAGES={}, SUBMISSION_REPORT=None)
    namespace.setdefault("SERIES", None)
    return namespace, code


@pytest.mark.parametrize(
    ("run", "competition", "selects"),
    [
        (CONFIG, False, False),
        (CONFIG, True, True),
        ("configs/submission_P2.6-01.yaml", False, True),
    ],
)
def test_an_adapters_build_writes_without_a_top_level_selection(
    run, competition, selects, tmp_path, monkeypatch
):
    import arc26.runner as runner

    if not (REPO / run).is_file():
        pytest.skip(f"{run} is not in this tree")
    namespace, code = write_namespace(run, tmp_path, monkeypatch, competition)
    calls = []
    monkeypatch.setattr(runner, "make_submission", lambda cfg, path: calls.append(path))
    namespace["solver_config"] = lambda arm=None: {}
    exec(compile(code["write"], "write", "exec"), namespace)
    record = namespace["STAGES"]["write"]
    assert bool(calls) is selects
    if not selects:
        assert record["selection"] == "arms" and "make_submission_error" not in record
    # Either way submission.json is written, placeholders for the competition's tasks.
    written = json.loads((tmp_path / "working" / "submission.json").read_text())
    assert list(written) == ["t1"] and record["submission"] is not None


def test_only_adapter_builds_carry_the_write_edit():
    config = B.parsed(CONFIG)
    armed = B.cell_sources(nbb.build_notebook(config, B.BUILD_STUB, B.SUBMIT_STAND_IN))
    twin = config._replace(adapters=None, notebook_inputs={})
    plain = B.cell_sources(nbb.build_notebook(twin, B.BUILD_STUB, B.SUBMIT_STAND_IN))
    assert 'record["selection"] = "arms"' in armed["write"]
    assert 'record["selection"] = "arms"' not in plain["write"]
    assert "        if SERIES is None:\n            select_attempts(None" in plain["write"]


@pytest.fixture(scope="module")
def fit_builds(tmp_path_factory):
    """dist/P3.7-03/ twice from one throwaway repository."""
    if not B.BUILD_TESTS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = B.BUILD_TESTS.snapshot(tmp_path_factory.mktemp("repo"))
    first = B.BUILD_TESTS.read_files(bs.build("P3.7-03", repo / CONFIG, repo))
    again = bs.build("P3.7-03", repo / CONFIG, repo)
    return again, first


def test_the_fit_build_is_rebuilt_byte_for_byte_and_passes_the_gateway(fit_builds):
    folder, first = fit_builds
    assert B.BUILD_TESTS.read_files(folder) == first
    meta = ops.load_kernel_metadata(folder.resolve())
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder((folder / "arc26-code").resolve())
    assert meta["id"] == "alyahya737/arc26-h-val-fit"
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p37f"]
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | no, not for submission:" in manifest
    assert "`h-fit` (hyper_fit)" in manifest and "| adapters train_log" in manifest
    namespace = {}
    exec(B.BUILD_TESTS.code(folder)["arc26-02-config"], namespace)
    assert namespace["ADAPTERS"]["train_log"] is True


def test_the_fit_brief_header_matches_its_config():
    text = BRIEF.read_text(encoding="utf-8")
    assert "TODO" not in text
    header = ops.brief_header(text)
    config = B.parsed(CONFIG)
    assert header["Run-id"] == "P3.7-03" and header["Config"] == CONFIG
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Accelerator"] == config.accelerator and header["Internet"] == "off"
    assert int(header["Timeout"]) == config.solver_kill_s + 300 + 2700


def test_revision_2_reads_mt_03s_best_point():
    """The three-key swap of revision 2: best.pt and its logs from MT-03's output, version 1."""
    adapters = B.parsed(CONFIG).adapters
    assert B.h_outputs(CONFIG) == (B.MT_03_OUTPUT, B.MT_03_OUTPUT)
    assert adapters.checkpoint == f"{{input:{B.MT_03_OUTPUT}}}/checkpoints/best.pt"
    assert adapters.load_check["run_logs"] == f"{{input:{B.MT_03_OUTPUT}}}/run_logs"
    assert B.raw_config(CONFIG)["notebook_inputs"][B.MT_03_OUTPUT] == 1
