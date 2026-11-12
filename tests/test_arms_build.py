"""P3.7: the adapter arms in the notebook builder, and the builds of runs P3.7-01 and P3.7-02.

A submission config may set ``adapters`` (task P3.7, ADR-019): the single run's adapter arms,
development runs only, with ``notebook_inputs``, the other notebooks' outputs they read. Checked
here: the builder's rules equal ``arc26.arms``' (sources, views, keys); the refusals (the
submission notebook, no accelerator, the moved solver, a series, no scope, a subset with GEN, a
hard stop before the arms' selection, inputs that the paths do not name, malformed settings, a
scope that is not H's); every build without adapters keeps its cells (the golden builds of
``tests/test_scope_build.py`` stay as they were); an adapter build differs from its twin only in
the adapter lines; the notebook finds the attached outputs, and its Run cell passes the arms to
``arc26.runner.RunConfig`` with every path filled in a development run and none in a competition
rerun; H's checkpoint is a config-only swap (amendment 7: a later best point changes three
config lines, four when ``best.pt`` and the last logs sit in two outputs; a half swap is
refused); the two configs, their builds (rebuilt byte for byte, the gateway's checks, never for
submission), and their briefs' headers. Nothing reaches Kaggle.
"""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from arc26 import arms as am
from arc26.runner import ARMS_MODULE, RunConfig, solver_command

REPO = Path(__file__).resolve().parent.parent
CONFIGS = {"P3.7-01": "configs/P3.7-01.yaml", "P3.7-02": "configs/P3.7-02.yaml"}
BRIEFS = {run: REPO / "docs" / "prerun" / f"{run}.md" for run in CONFIGS}
# The inputs every P3.7 build pins besides H's: MT-01's cache, P3.4's build 2 and its check.
FIXED_INPUTS = {
    "alyahya737/arc26-mt-cache": 1,
    "alyahya737/arc26-p3-4-episodes": 2,
    "alyahya737/arc26-p3-4-leakage": 2,
}
PLACEHOLDER = "{input:"


def raw_config(name):
    return yaml.safe_load((REPO / name).read_text(encoding="utf-8"))


def h_outputs(name):
    """The outputs holding H's ``best.pt`` and its run logs, as the config alone names them.

    H's checkpoint is a config-only swap (task brief amendment 7): a later best point changes the
    ``checkpoint`` and ``load_check.run_logs`` lines and their ``notebook_inputs`` entries, and no
    test repeats the pin.
    """
    adapters = raw_config(name)["adapters"]
    paths = (adapters["checkpoint"], adapters["load_check"]["run_logs"])
    assert all(path.startswith(PLACEHOLDER) for path in paths), paths
    return tuple(path[len(PLACEHOLDER) : path.index("}")] for path in paths)


INPUTS = raw_config(CONFIGS["P3.7-01"])["notebook_inputs"]
H_OUTPUT = h_outputs(CONFIGS["P3.7-01"])[0]
BUILD_STUB = {
    "tag": "t", "commit": "0" * 40, "config": "configs/P3.7-01.yaml", "config_sha256": "2" * 64,
    "wheel": "arc26-0.0.1-py3-none-any.whl", "wheel_sha256": "1" * 64,
}  # fmt: skip
SUBMIT_STAND_IN = "# arc26/submit.py: a stand-in\n"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


BUILD_TESTS = load("_arms_build_submission", REPO / "tests" / "test_build_submission.py")
nbb, bs, ops = BUILD_TESTS.nbb, BUILD_TESTS.bs, BUILD_TESTS.ops
DROP = BUILD_TESTS.DROP


def read(name):
    return (REPO / name).read_bytes()


def parsed(name, resolve=True):
    config = nbb.parse_config(read(name), Path(name))
    return nbb.resolve_adapters(config, read) if resolve else config


def config_text(changes, name=CONFIGS["P3.7-01"], adapters=None):
    raw = yaml.safe_load((REPO / name).read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is DROP:
            raw.pop(key)
        else:
            raw[key] = value
    for key, value in (adapters or {}).items():
        if value is DROP:
            raw["adapters"].pop(key)
        else:
            raw["adapters"][key] = value
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def refused(match, changes=None, adapters=None, name=CONFIGS["P3.7-01"]):
    with pytest.raises(nbb.BuildError, match=match):
        nbb.parse_config(config_text(changes or {}, name, adapters), Path("c.yaml"))


def cell_sources(notebook):
    return {cell["id"].split("-", 2)[2]: "".join(cell["source"]) for cell in notebook["cells"]}


# --- the builder's rules are arc26.arms' ---------------------------------------------------------


def test_the_builders_adapter_constants_are_the_arms_modules():
    assert nbb.ADAPTER_SOURCES == am.SOURCES
    assert set(nbb.ADAPTER_H_SOURCES) == am.H_SOURCES
    assert set(nbb.ADAPTER_POLISH_SOURCES) == am.POLISH_SOURCES
    assert [set(g) for g in nbb.ADAPTER_VIEW_GROUPS] == [set(g) for g in am.LENGTH_GROUPS]
    assert nbb.ADAPTER_MAX_BATCH == am.MAX_BATCH and nbb.EPOCH_VIEWS == am.EPOCH_VIEWS
    # The config's keys are Arms' fields less the folders the notebook adds.
    fields = [f.name for f in dataclasses.fields(am.Arms)]
    keys = set(nbb.ADAPTER_KEYS) | set(nbb.ADAPTER_OPTIONAL_KEYS)
    assert set(fields) - keys == {"outputs_root", "log_dir", "cache_dir"}
    assert keys <= set(fields) and not set(nbb.ADAPTER_KEYS) & set(nbb.ADAPTER_OPTIONAL_KEYS)
    # The optional keys' defaults in arc26.arms are what a config without them gets.
    defaults = {f.name: f.default for f in dataclasses.fields(am.Arms)}
    assert {k: defaults[k] for k in nbb.ADAPTER_OPTIONAL_KEYS} == {"train_log": False}
    assert set(nbb.ADAPTER_H_SOURCES) - {"hyper_fit"} == set(am.H_SOURCES) - {am.HYPER_FIT}
    lc = {f.name for f in dataclasses.fields(am.LoadCheckSettings)}
    assert lc - set(nbb.LOAD_CHECK_KEYS) == {"dev4_challenges"}
    gen = {f.name for f in dataclasses.fields(am.GenSettings)}
    assert gen - set(nbb.GEN_KEYS) == {"out_dir"}
    hypernet = pytest.importorskip("arc26.hyper.hypernet")
    names = {f.name for f in dataclasses.fields(hypernet.HypernetConfig)}
    assert set(nbb.HYPERNET_KEYS) | set(nbb.HYPERNET_OPTIONAL_KEYS) == names
    assert not set(nbb.HYPERNET_KEYS) & set(nbb.HYPERNET_OPTIONAL_KEYS)
    # Task P3.3b's aggregator block: its fields and defaults, the ones a checkpoint leaves out.
    assert nbb.AGGREGATOR_DEFAULTS == dataclasses.asdict(hypernet.AggregatorConfig())


def test_the_solver_loads_the_alpha_the_builder_requires():
    import ast

    tree = ast.parse((REPO / "arc26" / "solver.py").read_text(encoding="utf-8"))
    alphas = [
        k.value.value for n in ast.walk(tree) if isinstance(n, ast.Call)
        for k in n.keywords if k.arg == "lora_alpha"
    ]  # fmt: skip
    assert alphas == [nbb.SOLVER_LORA_ALPHA]


# --- the two configs -----------------------------------------------------------------------------


@pytest.mark.parametrize("run", sorted(CONFIGS))
def test_each_config_parses_resolves_and_is_never_for_submission(run):
    config = parsed(CONFIGS[run])
    adapters = config.adapters
    assert [arm.source for arm in adapters.arms] == ["base", "hyper", "sgd", "hyper_polish"]
    assert [arm.steps for arm in adapters.arms] == [None, None, None, 13]
    assert config.notebook_inputs == INPUTS and config.scope.lora_r == 16
    expected = yaml.safe_load(read("configs/hypernet.yaml"))
    if expected.get("aggregator") == nbb.AGGREGATOR_DEFAULTS:
        del expected["aggregator"]  # task P3.3b: the mean H's block, which a checkpoint leaves out
    assert adapters.hypernet_config == expected
    assert adapters.manifest_sha256.startswith("b1f2cd04")
    assert config.accelerator == "NvidiaL4" and config.workers == 4 and config.solver == "split"
    blocker = nbb.submission_blocker(config)
    assert "adapter arm" in blocker and config.kaggle_notebook in blocker
    settings = adapters.settings()
    assert list(settings) == list(nbb.ADAPTER_KEYS)
    # The settings, with the notebook's folders, are valid arc26.arms.Arms.
    filled = {
        **settings,
        "arms": [{k: v for k, v in a.items() if k != "reason"} for a in settings["arms"]],
        "outputs_root": "/o", "log_dir": "/l", "cache_dir": "/c",
    }  # fmt: skip
    if filled["gen"] is not None:
        filled["gen"] = {**filled["gen"], "out_dir": "/g"}
    filled["load_check"] = {**filled["load_check"], "dev4_challenges": "/d.json"}
    assert am.Arms.from_mapping(filled).arms[3].steps == 13


def test_dev40_keeps_16_views_and_gen_runs_100_puzzles_at_8():
    dev, gen = parsed(CONFIGS["P3.7-01"]), parsed(CONFIGS["P3.7-02"])
    assert dev.adapters.views is None and dev.adapters.gen is None
    assert len(dev.subset) == 40 and set(dev.subset) == set(__import__("arc26.data").data.DEV40)
    assert gen.adapters.views == am.DIHEDRAL_ONCE and gen.subset is None
    assert gen.adapters.gen == {"barc": 50, "rearc": 50}
    for config in (dev, gen):
        reserve = config.solver_end_s + config.adapters.finish_s + nbb.ADAPTER_SELECTION_RESERVE_S
        assert config.solver_kill_s == reserve


def test_both_configs_read_one_checkpoint_and_pin_the_other_inputs():
    """One best point for both sets (the load check's verdict is shared), H's outputs attached at
    a version, and every other input fixed."""
    assert h_outputs(CONFIGS["P3.7-01"]) == h_outputs(CONFIGS["P3.7-02"])
    for name in CONFIGS.values():
        inputs = raw_config(name)["notebook_inputs"]
        assert inputs == INPUTS
        held = set(h_outputs(name))
        assert held <= set(inputs) and all(inputs[output] >= 1 for output in held)
        assert {k: v for k, v in inputs.items() if k not in held} == FIXED_INPUTS


# Slugs no run has, so the swap tests hold whatever the configs pin (a real swap to MT-03's or
# MT-04's output, arc26-mt-train-2 or -3, included).
SWAP_BEST, SWAP_LAST = "alyahya737/arc26-test-swap-best", "alyahya737/arc26-test-swap-last"
# MT-03's real output (its brief): the swap a later best point would make.
MT_03_OUTPUT = "alyahya737/arc26-mt-train-2"


def swapped(name, best=SWAP_BEST, inputs=None):
    """The config with H's checkpoint from ``best`` and its logs from SWAP_LAST: the session that
    wrote ``best.pt`` (``best``; H_OUTPUT when a later session found no new low) and the last one,
    whose ``status.json`` names it (``best.session_start``)."""
    adapters = raw_config(name)["adapters"]
    load_check = {**adapters["load_check"], "run_logs": f"{{input:{SWAP_LAST}}}/run_logs"}
    if inputs is None:
        inputs = {best: 1, SWAP_LAST: 1, **FIXED_INPUTS}
    return config_text(
        {"notebook_inputs": inputs}, name,
        {"checkpoint": f"{{input:{best}}}/checkpoints/best.pt", "load_check": load_check},
    )  # fmt: skip


@pytest.mark.parametrize("best", [SWAP_BEST, H_OUTPUT])
@pytest.mark.parametrize("run", sorted(CONFIGS))
def test_a_later_best_point_is_a_config_only_swap(run, best):
    """Two outputs: best.pt from a later session (SWAP_BEST) or still MT-02's (H_OUTPUT), the
    logs from the last session (SWAP_LAST)."""
    text = swapped(CONFIGS[run], best)
    config = nbb.resolve_adapters(nbb.parse_config(text, Path("c.yaml")), read)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    sources = bs.kernel_metadata(config, pulled)["kernel_sources"]
    assert f"{best}/1" in sources and f"{SWAP_LAST}/1" in sources
    for stale in {SWAP_BEST, *h_outputs(CONFIGS[run])} - {best, SWAP_LAST}:
        assert not any(source.startswith(f"{stale}/") for source in sources)
    adapters = config.adapters.settings()
    assert adapters["checkpoint"] == f"{{input:{best}}}/checkpoints/best.pt"
    assert adapters["load_check"]["run_logs"] == f"{{input:{SWAP_LAST}}}/run_logs"
    # Everything else in the settings is the committed config's.
    committed = parsed(CONFIGS[run]).adapters.settings()
    rest = {k: v for k, v in adapters.items() if k not in ("checkpoint", "load_check")}
    assert rest == {k: v for k, v in committed.items() if k not in ("checkpoint", "load_check")}
    # A half swap is refused: the old pin left in notebook_inputs, or the new one missing.
    for inputs in ({SWAP_BEST: 1, H_OUTPUT: 1, **FIXED_INPUTS}, {best: 1, **FIXED_INPUTS}):
        with pytest.raises(nbb.BuildError, match="does not attach|never reach the arms"):
            nbb.parse_config(swapped(CONFIGS[run], best, inputs), Path("c.yaml"))


@pytest.mark.parametrize("run", sorted(CONFIGS))
def test_a_real_swap_of_the_three_keys_builds(run, tmp_path):
    """MT-03's output holds the best point and its logs: checkpoint, load_check.run_logs, and the
    notebook_inputs entry change; the build attaches it, and its MANIFEST names it."""
    raw = raw_config(CONFIGS[run])
    adapters = raw["adapters"]
    load_check = {**adapters["load_check"], "run_logs": f"{{input:{MT_03_OUTPUT}}}/run_logs"}
    inputs = {MT_03_OUTPUT: 1, **FIXED_INPUTS}
    text = config_text(
        {"notebook_inputs": inputs}, CONFIGS[run],
        {"checkpoint": f"{{input:{MT_03_OUTPUT}}}/checkpoints/best.pt", "load_check": load_check},
    )  # fmt: skip
    new = yaml.safe_load(text)
    # Only the three keys may differ from the committed config.
    assert {k for k in raw if raw[k] != new[k]} <= {"adapters", "notebook_inputs"}
    assert {k for k in adapters if adapters[k] != new["adapters"][k]} <= {
        "checkpoint", "load_check",
    }  # fmt: skip
    assert {k for k in adapters["load_check"] if adapters["load_check"][k]
            != new["adapters"]["load_check"][k]} <= {"run_logs"}  # fmt: skip
    config = nbb.resolve_adapters(nbb.parse_config(text, Path(CONFIGS[run])), read)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)
    assert f"{MT_03_OUTPUT}/1" in meta["kernel_sources"]
    for stale in set(h_outputs(CONFIGS[run])) - {MT_03_OUTPUT}:
        assert not any(s.startswith(f"{stale}/") for s in meta["kernel_sources"])
    assert config.notebook_inputs == inputs


@pytest.mark.parametrize("run", sorted(CONFIGS))
def test_the_run_cell_reads_best_pt_and_the_logs_from_two_outputs(run, tmp_path, monkeypatch):
    """best.pt stays in MT-02's output, the logs come from a later session's: each path is filled
    from its own mount."""
    config = nbb.resolve_adapters(
        nbb.parse_config(swapped(CONFIGS[run], H_OUTPUT), Path(CONFIGS[run])), read
    )
    _, calls = run_cell(run, tmp_path, monkeypatch, config=config)
    ((cfg, _),) = calls
    notebooks = tmp_path / "input" / "notebooks"
    assert cfg.adapters.checkpoint == str(notebooks / H_OUTPUT / "checkpoints" / "best.pt")
    assert cfg.adapters.load_check.run_logs == str(notebooks / SWAP_LAST / "run_logs")


# --- refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"kaggle_notebook": nbb.SUBMISSION_NOTEBOOK}, "never submitted"),
        ({"accelerator": None}, "no accelerator"),
        ({"solver": "nvarc"}, "needs 'split'"),
        ({"scope": DROP}, "no scope"),
        ({"solver_kill_s": 21000}, "leaves the arms' selection no time"),
        ({"notebook_inputs": DROP}, "notebook_inputs"),
        ({"notebook_inputs": {**INPUTS, "alyahya737/other-output": 1}}, "never reach the arms"),
        ({"notebook_inputs": dict(list(INPUTS.items())[1:])}, "does not attach"),
    ],
)
def test_the_build_level_refusals(changes, match):
    refused(match, changes)


def test_an_adapter_build_has_no_agent_stage_and_no_dev240():
    """Task int-W43: the adapter arms (task P3.7) and the hybrid build's agent stage or DEV240
    (tasks P6.5, P6.5b) were never combined or tested together, so the builder refuses both."""
    shot = yaml.safe_load((REPO / "configs" / "submission_P6.5-02.yaml").read_text())["agent"]
    refused("it has an agent stage", {"agent": shot})
    refused("is the hybrid's timing check", {"subset": nbb.DEV240_SUBSET})
    # Each alone is accepted: the shot build with its agent, the adapter build without one.
    assert parsed(CONFIGS["P3.7-01"], resolve=False).adapters is not None
    assert parsed("configs/submission_P6.5-02.yaml", resolve=False).agent is not None


def test_gen_needs_no_subset_and_inputs_need_adapters():
    refused("subset must be null with gen", {"subset": ["136b0064"]}, name=CONFIGS["P3.7-02"])
    refused("no adapters", {"adapters": DROP})


@pytest.mark.parametrize(
    ("adapters", "match"),
    [
        ({"ce": DROP}, "exactly the keys"),
        ({"extra": 1}, "exactly the keys"),
        ({"arms": []}, "non-empty list"),
        ({"arms": [{"name": "a", "source": "sgd", "reason": "r", "steps": 2}]}, "steps apply"),
        ({"arms": [{"name": "a", "source": "polish", "reason": "r"}]}, "polish needs steps"),
        ({"arms": [{"name": "A", "source": "sgd", "reason": "r"}]}, "name must be"),
        ({"arms": [{"name": "a", "source": "lora", "reason": "r"}]}, "source must be"),
        ({"arms": [{"name": "a", "source": "sgd", "reason": "r"}] * 2}, "distinct"),
        ({"views": [[0, 2]]}, "views must be"),
        ({"views": [[0, 1, 4, 5, 10]]}, "views must be"),
        ({"views": [[0], [0]]}, "views must be"),
        ({"order_seed": "0"}, "order_seed"),
        ({"finish_s": -1}, "finish_s"),
        ({"hypernet": "hypernet.yaml"}, "under configs/"),
        ({"manifest_sha256": "abc"}, "manifest_sha256"),
        ({"checkpoint": None}, "needs checkpoint"),
        ({"load_check": {"run_logs": "x", "max_abs": 0, "max_mean": 1}}, "positive number"),
        ({"load_check": {"run_logs": "x"}}, "exactly the keys"),
        ({"gen": {"barc": 0, "rearc": 0}}, "one positive"),
        ({"episodes_root": None}, "need episodes_root"),
    ],
)
def test_the_adapter_refusals(adapters, match):
    refused(match, adapters=adapters)


def test_resolve_refuses_a_scope_that_is_not_hs():
    scope = {"lora_r": 8, "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    config = nbb.parse_config(config_text({"scope": scope}), Path("c.yaml"))
    with pytest.raises(nbb.BuildError, match="ADR-017"):
        nbb.resolve_adapters(config, read)
    plain = nbb.parse_config(read(CONFIGS["P3.7-01"]), Path("c.yaml"))
    alpha = yaml.safe_load(read("configs/hypernet.yaml")) | {"alpha": 16}

    def other(name):
        return yaml.safe_dump(alpha).encode("utf-8")

    with pytest.raises(nbb.BuildError, match="alpha 16"):
        nbb.resolve_adapters(plain, other)
    with pytest.raises(nbb.BuildError, match="exactly the keys"):
        nbb.resolve_adapters(plain, lambda name: b"rank: 16\n")
    with pytest.raises(nbb.BuildError, match="was not read"):
        plain.adapters.settings()


# --- the notebook --------------------------------------------------------------------------------


def test_an_adapter_build_differs_from_its_twin_only_in_the_adapter_lines():
    config = parsed(CONFIGS["P3.7-01"])
    armed = cell_sources(nbb.build_notebook(config, BUILD_STUB, SUBMIT_STAND_IN))
    twin = config._replace(adapters=None, notebook_inputs={})
    plain = cell_sources(nbb.build_notebook(twin, BUILD_STUB, SUBMIT_STAND_IN))
    changed = {name for name in plain if armed[name] != plain[name]}
    assert changed == {"config-note", "config", "helpers", *nbb.ADAPTER_EDITS}
    for name, edits in nbb.ADAPTER_EDITS.items():
        text = plain[name]
        for old, new in edits:
            assert text.count(old) == 1, (name, old)
            text = text.replace(old, new)
        assert text == armed[name], name
    assert "ADAPTER" not in "".join(plain.values()) and "find_input_dirs" not in plain["helpers"]
    assert armed["helpers"] == nbb.helpers_cell_source(adapters=True)
    assert "def fill_args" in armed["helpers"] and "ARM_OUTPUTS = WORKING" in armed["helpers"]
    assert "Adapter arms build (task P3.7" in armed["config-note"]


def notebook_namespace(run, tmp_path, monkeypatch, competition=False, config=None):
    """The Config, Helpers, and validator cells run in process on a simulated tree, with the
    environment check's results set by hand; returns the namespace and the code by cell."""
    if config is None:
        config = parsed(CONFIGS[run])
    info = nbb.build_info(
        config, tag=run, commit="0" * 40, config_file=CONFIGS[run], config_sha256="2" * 64,
        wheel="arc26-0.0.1-py3-none-any.whl", wheel_sha256="1" * 64,
    )  # fmt: skip
    notebook = nbb.build_notebook(config, info, SUBMIT_STAND_IN)
    code = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
    monkeypatch.setenv("ARC26_KAGGLE_ROOT", str(tmp_path))
    monkeypatch.setenv("PYTHONHASHSEED", "stale")
    if competition:
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    else:
        monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    namespace = {"__name__": "__main__"}
    for cell in ("arc26-02-config", "arc26-04-helpers"):
        exec(compile(code[cell], cell, "exec"), namespace)
    return namespace, code


def mounts(tmp_path, inputs=INPUTS):
    """The attached outputs as Kaggle mounts them (notebooks/<owner>/<slug>)."""
    found = {}
    for name in inputs:
        owner, slug = name.split("/")
        folder = tmp_path / "input" / "notebooks" / owner / slug
        (folder / "run_logs").mkdir(parents=True)
        (folder / "run_logs" / "status.json").write_text("{}")
        found[name] = folder
    return found


def test_the_environment_check_finds_the_attached_outputs(tmp_path, monkeypatch, capsys):
    namespace, code = notebook_namespace("P3.7-02", tmp_path, monkeypatch)
    folders = mounts(tmp_path)
    exec(compile(code["arc26-06-validator"], "validator", "exec"), namespace)
    with pytest.raises(RuntimeError):  # no GPU, no data on the simulated tree: the check stops
        exec(compile(code["arc26-08-environment"], "environment", "exec"), namespace)
    assert namespace["INPUT_DIRS"] == folders
    printed = capsys.readouterr().out
    assert "adapter arms (task P3.7, ADR-019): 4 per puzzle" in printed
    assert "h-polish: hyper_polish, 13 steps" in printed
    assert "[[0, 4, 10, 14], [2, 6, 8, 12]]" in printed


def run_cell(run, tmp_path, monkeypatch, competition=False, config=None):
    import arc26.runner as runner

    namespace, code = notebook_namespace(run, tmp_path, monkeypatch, competition, config)
    calls = []
    monkeypatch.setattr(
        runner, "run_solver",
        lambda cfg, end_time, python=None, timeout=None: calls.append((cfg, end_time)) or 0,
    )  # fmt: skip
    namespace.update(
        INSTALLED=True,
        PYTHON="/usr/local/bin/python",
        DATA_DIR=tmp_path / "data",
        MODEL_DIR=tmp_path / "model",
        CHALLENGES=tmp_path / "data" / "arc-agi_evaluation_challenges.json",
    )
    inputs = INPUTS if config is None else config.notebook_inputs
    namespace["INPUT_DIRS"].update(mounts(tmp_path, inputs))
    exec(compile(code["arc26-12-run"], "run", "exec"), namespace)
    return namespace, calls


@pytest.mark.parametrize("run", sorted(CONFIGS))
def test_the_run_cell_passes_the_arms_with_every_path_filled(run, tmp_path, monkeypatch):
    namespace, calls = run_cell(run, tmp_path, monkeypatch)
    ((cfg, _),) = calls
    assert isinstance(cfg, RunConfig) and isinstance(cfg.adapters, am.Arms)
    assert solver_command(cfg, 1.0, "python")[2] == ARMS_MODULE
    arms = cfg.adapters
    notebooks = tmp_path / "input" / "notebooks"
    best, logs = h_outputs(CONFIGS[run])
    assert arms.checkpoint == str(notebooks / best / "checkpoints" / "best.pt")
    assert arms.load_check.run_logs == str(notebooks / logs / "run_logs")
    assert arms.cache.endswith("arc26-mt-cache/embedding_cache")
    assert arms.episodes_root.endswith("arc26-p3-4-episodes")
    assert arms.load_check.dev4_challenges == str(namespace["CHALLENGES"])
    assert arms.outputs_root == str(tmp_path / "working" / "arm_outputs")
    assert arms.log_dir == str(tmp_path / "working" / "run_logs" / "arms")
    assert arms.cache_dir == str(tmp_path / "h_cache")
    assert (cfg.lora_r, cfg.lora_targets) == (16, ("q_proj", "k_proj", "v_proj", "o_proj"))
    if run == "P3.7-02":
        assert arms.gen.out_dir == str(tmp_path / "working" / "gen_data") and cfg.subset is None
        assert arms.views == am.DIHEDRAL_ONCE
    else:
        assert arms.gen is None and len(cfg.subset) == 40 and arms.views is None
    assert namespace["STAGES"]["run"]["adapters"] is True


def test_a_competition_rerun_ignores_the_arms_and_runs_the_scope(tmp_path, monkeypatch):
    namespace, calls = run_cell("P3.7-02", tmp_path, monkeypatch, competition=True)
    ((cfg, _),) = calls
    assert cfg.adapters is None and namespace["ADAPTER_ARMS"] is None
    assert solver_command(cfg, 1.0, "python")[2] == "arc26.solver" and cfg.lora_r == 16
    assert namespace["STAGES"]["run"]["adapters"] is False


def test_metadata_attach_the_inputs_and_the_manifest_lists_the_arms():
    config = parsed(CONFIGS["P3.7-02"])
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)  # runs the gateway's metadata checks
    assert meta["id"] == "alyahya737/arc26-h-val-gen" and meta["machine_shape"] == "NvidiaL4"
    assert meta["kernel_sources"] == ["sorokin/pip-install-unsloth-flash-patch/2"] + [
        f"{name}/{version}" for name, version in INPUTS.items()
    ]
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p37"]
    info = nbb.build_info(config, tag="P3.7-02", commit="0" * 40, config_file=CONFIGS["P3.7-02"],
                          config_sha256="2" * 64, wheel="arc26-0.0.1-py3-none-any.whl",
                          wheel_sha256="1" * 64)  # fmt: skip
    assert info["notebook_inputs"] == INPUTS
    rows = bs._adapters_summary(config)
    assert rows[0].startswith("| adapters (development runs only; task P3.7, ADR-019) | `base`")
    assert any("[[0, 4, 10, 14], [2, 6, 8, 12]]" in row for row in rows)
    assert any(config.adapters.hypernet_sha256 in row for row in rows)


# --- the builds ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """dist/P3.7-01/ twice and dist/P3.7-02/ once from one throwaway repository."""
    if not BUILD_TESTS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = BUILD_TESTS.snapshot(tmp_path_factory.mktemp("repo"))
    first = BUILD_TESTS.read_files(bs.build("P3.7-01", repo / CONFIGS["P3.7-01"], repo))
    again = bs.build("P3.7-01", repo / CONFIGS["P3.7-01"], repo)
    gen = bs.build("P3.7-02", repo / CONFIGS["P3.7-02"], repo)
    return repo, again, first, gen


def test_the_p3_7_builds_are_rebuilt_byte_for_byte_with_one_wheel(builds):
    _, folder, first, gen = builds
    files = BUILD_TESTS.read_files(folder)
    assert set(files) == BUILD_TESTS.FILES and files == first
    wheel = f"arc26-code/{BUILD_TESTS.WHEEL}"
    assert files[wheel] == BUILD_TESTS.read_files(gen)[wheel]


@pytest.mark.parametrize("which", [1, 3])
def test_the_p3_7_builds_pass_the_gateway_and_are_not_for_submission(builds, which):
    folder = builds[which]
    meta = ops.load_kernel_metadata(folder.resolve())
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder((folder / "arc26-code").resolve())
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p37"]
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | no, not for submission:" in manifest
    assert "| adapters (development runs only; task P3.7, ADR-019) |" in manifest
    assert f"passed to the adapter arms as `{{input:{H_OUTPUT}}}`" in manifest
    namespace = {}
    exec(BUILD_TESTS.code(folder)["arc26-02-config"], namespace)
    assert [a["name"] for a in namespace["ADAPTERS"]["arms"]] == ["base", "h", "sgd", "h-polish"]
    assert namespace["BUILD"]["notebook_inputs"] == INPUTS


# --- the briefs ----------------------------------------------------------------------------------


@pytest.mark.parametrize("run", sorted(BRIEFS))
def test_each_brief_header_parses_and_matches_its_config(run):
    text = BRIEFS[run].read_text(encoding="utf-8")
    assert "TODO" not in text
    header = ops.brief_header(text)
    config = parsed(CONFIGS[run])
    assert header["Run-id"] == run and header["Config"] == CONFIGS[run]
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Accelerator"] == config.accelerator
    # The hard stop plus 300 s for the logs plus 2,700 s, the pre-script allowance: Kaggle has
    # spent up to about 41 minutes between RUNNING and the first cell, from which the deadlines
    # count (MT-03 revision 3, fde6baa).
    assert int(header["Timeout"]) == config.solver_kill_s + 300 + 2700
    assert header["Internet"] == "off"


def test_the_h_config_is_embedded_in_the_form_a_checkpoint_stores():
    """Task P3.3b's aggregator block (int-W43): a block equal to the defaults is left out, as
    ``HypernetConfig.to_dict`` leaves it out of a checkpoint's config, so the load check and the
    workers compare equal mappings; a block that differs is embedded with every key."""
    text = (REPO / "configs" / "hypernet.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    assert raw["aggregator"] == nbb.AGGREGATOR_DEFAULTS
    parsed = nbb.parse_hypernet(text.encode("utf-8"), "configs/hypernet.yaml")
    assert list(parsed) == list(nbb.HYPERNET_KEYS)
    without = {key: value for key, value in raw.items() if key != "aggregator"}
    assert nbb.parse_hypernet(yaml.safe_dump(without).encode(), "h.yaml") == parsed
    changed = {**raw, "aggregator": {"kind": "set"}}
    got = nbb.parse_hypernet(yaml.safe_dump(changed).encode(), "h.yaml")
    assert got["aggregator"] == {**nbb.AGGREGATOR_DEFAULTS, "kind": "set"}
    for bad in ({"kind": "set", "depth": 2}, None, ["set"]):
        with pytest.raises(nbb.BuildError, match="aggregator must be a mapping"):
            nbb.parse_hypernet(yaml.safe_dump({**raw, "aggregator": bad}).encode(), "h.yaml")
    hypernet = pytest.importorskip("arc26.hyper.hypernet")
    config = hypernet.HypernetConfig.from_yaml(REPO / "configs" / "hypernet.yaml")
    assert config.to_dict() == parsed
    assert hypernet.HypernetConfig(**got).to_dict() == got


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({"kind": "attention"}, "aggregator kind must be one of mean, set"),
        ({"kind": "set", "width": 0}, "aggregator width must be a positive integer"),
        ({"kind": "set", "blocks": -1}, "aggregator blocks must be a positive integer"),
        ({"kind": "set", "heads": True}, "aggregator heads must be a positive integer"),
        ({"kind": "set", "width": 2.5}, "aggregator width must be a positive integer"),
        ({"kind": "set", "heads": 7}, "aggregator heads 7 must divide its width 512"),
    ],
)
def test_the_h_config_refuses_aggregator_values_h_refuses(block, match):
    """Task int-W43: ``parse_hypernet`` checks the aggregator's values by ``AggregatorConfig``'s
    rules, so a value H would refuse fails at the build, not in the GPU session."""
    raw = yaml.safe_load((REPO / "configs" / "hypernet.yaml").read_text(encoding="utf-8"))
    data = yaml.safe_dump({**raw, "aggregator": block}).encode()
    with pytest.raises(nbb.BuildError, match=match):
        nbb.parse_hypernet(data, "h.yaml")
    assert nbb.AGGREGATOR_KINDS == pytest.importorskip("arc26.hyper.hypernet").AGGREGATORS
    with pytest.raises(ValueError):
        pytest.importorskip("arc26.hyper.hypernet").AggregatorConfig.from_mapping(block)
