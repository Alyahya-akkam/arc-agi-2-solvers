"""The hybrid build in the notebook builder (task P6.5, ADR-020): NVARC, then the agent stage.

A submission config may set ``agent``: NVARC runs for its share of the session, then the agent
stage installs the engine offline and runs ``python -m arc26.agent.run``, a final check keeping
NVARC's attempts wherever the agent leaves none. Against the task brief's acceptance this file
checks:

- every committed build config without an agent writes, byte for byte, what the builders wrote
  before task P6.5 (``tests/testdata/p6_5_golden_builds.json``, recorded at ``8cac42f``), P2.6-01's
  (the pure NVARC pick) included;
- the hybrid build's cells and inputs: a hybrid build differs from its twin without the agent
  only in its first two cells, ``HYBRID_EDITS`` in two fixed cells, and four added cells; the
  agent's run config is embedded as built; the metadata attach both download outputs; the
  manifest states the plan; ``dist/P6.5-01/`` is fit to submit and rebuilds byte for byte;
- the time arithmetic: every plan the builder accepts ends before Kaggle's 12 h limit with the
  share, the install, the boot, the agent's deadline, SIGKILL, and the final write in order, and
  NVARC's deadlines only shrink;
- the config rules and refusals, the run config's checks, and the agent stage's runtime helpers.

The agent's run config is task P6.4's ``configs/agent_default.yaml``. The notebook's runs on the
simulated Kaggle tree are ``tests/test_hybrid_notebook.py``'s. Nothing reaches Kaggle.
"""

import bz2
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import itertools
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import arc26.submit as submit

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


SCOPE_TESTS = load("_hybrid_scope_build", REPO / "tests" / "test_scope_build.py")
BUILD_TESTS = SCOPE_TESTS.BUILD_TESTS
LOCAL = SCOPE_TESTS.LOCAL
nbb, bs, ops = BUILD_TESTS.nbb, BUILD_TESTS.bs, BUILD_TESTS.ops
DROP = BUILD_TESTS.DROP
GOLDEN = json.loads((REPO / "tests" / "testdata" / "p6_5_golden_builds.json").read_text())
HYBRID_CONFIG = "configs/submission_P6.5-hybrid.yaml"
DEV40_CONFIG = "configs/submission_P6.5-01.yaml"
AGENT_ONLY_CONFIG = "configs/submission_P6.5-agent.yaml"
CPU_CONFIG = "configs/submission_P6.5-cpu.yaml"
P26_CONFIG = "configs/submission_P2.6-01.yaml"
HYBRID_CONFIGS = (HYBRID_CONFIG, DEV40_CONFIG, AGENT_ONLY_CONFIG, CPU_CONFIG)
# Task P6.5b's and P6.9's shot builds, which have an agent too (tests/test_shot_build.py).
SHOT_CONFIGS = (
    "configs/submission_P6.5-02-cpu.yaml", "configs/submission_P6.5-02.yaml",
    "configs/submission_P6.5-03.yaml", "configs/submission_P6.9-02-cpu.yaml",
    "configs/submission_P6.9-02.yaml", "configs/submission_P6.9-03-cpu.yaml",
    "configs/submission_P6.9-03.yaml", "configs/submission_P6.9-04-cpu.yaml",
    "configs/submission_P6.9-04.yaml", "configs/submission_P6.9-05-cpu.yaml",
    "configs/submission_P6.9-05.yaml", "configs/submission_P4.7-02-cpu.yaml",
    "configs/submission_P4.7-02.yaml", "configs/submission_P5.1-01-cpu.yaml",
    "configs/submission_P5.1-01.yaml",
)  # fmt: skip
# Builds without an agent committed after the goldens were recorded (task P4.7); their own tests
# check their bytes (two builds identical) and their settings (tests/test_p4_7.py).
LATER_CONFIGS = ("configs/submission_P4.7-01.yaml", "configs/submission_P4.7-01-cpu.yaml")
AGENT_CONFIG = "configs/agent_default.yaml"
MODEL_NOTEBOOK = "alyahya737/arc26-dl-qwen38-int4"
WHEELS_NOTEBOOK = "alyahya737/arc26-dl-agent-wheels"
REVISION = "91bd022d5b49442a868bc35008f6c21e1860edfa"
BUILD_STUB = {
    "tag": "t", "commit": "0" * 40, "config": HYBRID_CONFIG, "config_sha256": "2" * 64,
    "wheel": SCOPE_TESTS.GOLDEN_WHEEL, "wheel_sha256": SCOPE_TESTS.GOLDEN_WHEEL_SHA256,
}  # fmt: skip


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def agent_config_bytes():
    return (REPO / AGENT_CONFIG).read_bytes()


def read_repo(name):
    """A repository file's bytes, as resolve_agent reads them."""
    return (REPO / name).read_bytes()


def parsed(name):
    path = REPO / name
    return nbb.parse_config(path.read_bytes(), path)


def resolved(name=HYBRID_CONFIG):
    return nbb.resolve_agent(nbb.resolve_rounds(parsed(name), read_repo), read_repo)


def config_text(changes, agent_changes=None, name=HYBRID_CONFIG):
    """A committed config with changes (DROP removes a key), as text for parse_config."""
    raw = yaml.safe_load((REPO / name).read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is DROP:
            raw.pop(key)
        else:
            raw[key] = value
    for key, value in (agent_changes or {}).items():
        if value is DROP:
            raw["agent"].pop(key)
        else:
            raw["agent"][key] = value
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def cell_sources(notebook):
    """Each cell's text by its name, the cell id without ``arc26-NN-``."""
    return {cell["id"].split("-", 2)[2]: "".join(cell["source"]) for cell in notebook["cells"]}


def committed_build_configs():
    return sorted(
        p.relative_to(REPO).as_posix()
        for pattern in ("submission_*.yaml", "script_*.yaml")
        for p in (REPO / "configs").glob(pattern)
    )


# Builds without an agent: byte for byte as before task P6.5.


def test_the_goldens_cover_every_committed_build_config_but_the_hybrids():
    assert GOLDEN["commit"].startswith("8cac42f")
    assert set(GOLDEN["configs"]) == set(GOLDEN["inputs"])
    agents = {*HYBRID_CONFIGS, *SHOT_CONFIGS}
    later = set(LATER_CONFIGS)
    assert later <= set(committed_build_configs()) and not later & set(GOLDEN["configs"])
    assert set(GOLDEN["configs"]) == set(committed_build_configs()) - agents - later
    assert P26_CONFIG in GOLDEN["configs"]


@pytest.mark.parametrize("name", sorted(GOLDEN["configs"]))
def test_every_committed_build_without_an_agent_writes_its_recorded_bytes(name):
    """The notebook, both metadata files, and MANIFEST.md of every committed build config equal
    what the builders wrote before task P6.5, while the build's inputs hash as recorded then (a
    config whose input a later task changes is skipped with the file named)."""
    recorded = GOLDEN["inputs"][name]
    changed = [
        file
        for file, digest in recorded.items()
        if not (REPO / file).is_file() or sha256((REPO / file).read_bytes()) != digest
    ]
    if changed:
        pytest.skip(f"{name}: changed since the goldens were recorded: {', '.join(changed)}")
    assert parsed(name).kind == "script" or parsed(name).agent is None
    assert SCOPE_TESTS.golden_inputs(bs, name) == recorded
    assert SCOPE_TESTS.golden_digests(bs, name) == GOLDEN["configs"][name]


# The P6.5 configs.


def test_the_hybrid_config_is_p2_6_01s_with_the_agent_on_its_own_code_dataset():
    hybrid, p26 = parsed(HYBRID_CONFIG), parsed(P26_CONFIG)
    assert hybrid.code_dataset == "alyahya737/arc26-code-p65" != p26.code_dataset
    assert hybrid._replace(code_dataset=p26.code_dataset, agent=None) == p26
    assert nbb.submission_blocker(hybrid) is None and nbb.submission_blocker(p26) is None
    agent = hybrid.agent
    assert agent.stage_settings() == {
        "enabled": True, "nvarc_share_s": 21600, "end_s": 42000, "term_s": 120,
        "min_stage_s": 1800, "install_timeout_s": 1500,
    }  # fmt: skip
    assert agent.model == nbb.AgentModel(MODEL_NOTEBOOK, 1, "qwen38-27b-int4", REVISION)
    assert agent.wheels == nbb.AgentWheels(WHEELS_NOTEBOOK, 1, "wheelhouse", "0.31.0")
    assert (agent.config, agent.base_python, agent.scratch_dir, agent.min_driver) == (
        AGENT_CONFIG, "/usr/bin/python3.11", "/tmp/arc26-agent", "580.65.06"
    )  # fmt: skip
    assert agent.inputs() == {MODEL_NOTEBOOK: 1, WHEELS_NOTEBOOK: 1}


def test_the_agent_only_config_is_the_hybrids_with_a_share_of_0():
    hybrid, only = parsed(HYBRID_CONFIG), parsed(AGENT_ONLY_CONFIG)
    assert only.agent.nvarc_share_s == 0
    assert only._replace(agent=only.agent._replace(nvarc_share_s=21600)) == hybrid
    assert nbb.submission_blocker(only) is None


def test_the_cpu_check_is_the_hybrids_notebook_on_cpu_with_the_cpu_checks_probe():
    """P6.5-cpu: the hybrid's agent inputs, engine, and scope, the CPU check's notebook, subset,
    worker, and deadlines; no accelerator, so the agent check needs no GPU and the build is not
    fit to submit."""
    cpu, hybrid = parsed(CPU_CONFIG), parsed(HYBRID_CONFIG)
    check = parsed("configs/submission_cpu_check.yaml")
    assert cpu.accelerator is None and cpu.kaggle_notebook == check.kaggle_notebook
    assert (cpu.subset, cpu.workers, cpu.solver_end_s, cpu.solver_kill_s) == (
        check.subset, check.workers, check.solver_end_s, check.solver_kill_s
    )  # fmt: skip
    assert (cpu.code_dataset, cpu.scope, cpu.solver) == (
        hybrid.code_dataset, hybrid.scope, hybrid.solver
    )  # fmt: skip
    same = ("enabled", "term_s", "install_timeout_s", "config", "model", "wheels", "base_python")
    same += ("scratch_dir", "min_driver")
    assert all(getattr(cpu.agent, key) == getattr(hybrid.agent, key) for key in same)
    assert (cpu.agent.nvarc_share_s, cpu.agent.end_s, cpu.agent.min_stage_s) == (600, 3600, 60)
    assert "no accelerator" in nbb.submission_blocker(cpu)
    agent = resolved(CPU_CONFIG).agent
    assert plan(agent.stage_settings(), agent.config_data)["problems"] == []
    assert cpu.kaggle_notebook not in ops.briefed_notebooks()


def test_p6_5_01_is_the_hybrid_on_dev40_at_a_sixth_of_its_times():
    """Run P6.5-01: the hybrid's build on DEV40 in development mode, NVARC's share and the agent's
    generation window a sixth (40 of 240 tasks) of the hybrid's, the fixed overheads (NVARC's
    600 s gap, its write, the install, the boot, the final margin) unscaled; on its own notebook
    and not fit to submit, since a rerun would run these times too."""
    from arc26.data import DEV40

    dev40, hybrid = parsed(DEV40_CONFIG), parsed(HYBRID_CONFIG)
    assert set(dev40.subset) == set(DEV40) and len(dev40.subset) == 40
    assert dev40.kaggle_notebook == "alyahya737/arc26-hybrid-dev" != nbb.SUBMISSION_NOTEBOOK
    assert dev40.kaggle_notebook not in {parsed(name).kaggle_notebook for name in GOLDEN["configs"]}
    same = ("enabled", "term_s", "min_stage_s", "install_timeout_s", "config", "model", "wheels")
    same += ("base_python", "scratch_dir", "min_driver")
    assert all(getattr(dev40.agent, key) == getattr(hybrid.agent, key) for key in same)
    other = ("kaggle_notebook", "subset", "agent")
    assert dev40._replace(**{key: getattr(hybrid, key) for key in other}) == hybrid
    assert dev40.agent.nvarc_share_s * 240 == hybrid.agent.nvarc_share_s * 40
    # The generation window: from the engine being up (NVARC's hard stop, its 60 s write, the
    # 300 s install and the 600 s boot, all unscaled) to the generation stop, a sixth of the
    # hybrid's, so each of the 40 tasks gets the rerun's time per task.
    final = resolved(DEV40_CONFIG).agent.config_data["deadline"]["final_margin_s"]

    def window(agent):
        up = agent.nvarc_share_s + nbb.NVARC_KILL_GAP_S + 60 + 300 + 600
        return agent.end_s - final - up

    assert window(dev40.agent) * 240 == window(hybrid.agent) * 40 == 2840 * 240
    blocker = nbb.submission_blocker(dev40)
    assert blocker.startswith("its notebook is alyahya737/arc26-hybrid-dev")
    assert "its agent stage ends 9800 s after the start, a development scale" in blocker
    agent = resolved(DEV40_CONFIG).agent
    assert plan(agent.stage_settings(), agent.config_data) == {
        "nvarc_end_s": 3600,
        "nvarc_kill_s": 4200,
        "agent_start_s": 4500,
        "install_end_s": 6000,
        "boot_end_s": 7800,
        "generation_stop_s": 8000,
        "agent_end_s": 9800,
        "agent_kill_s": 9920,
        "notebook_end_s": 10250,
        "problems": [],
    }


def test_a_scaled_agent_end_is_no_submission_even_on_the_submission_notebook():
    scaled = parsed(HYBRID_CONFIG)
    scaled = scaled._replace(agent=scaled.agent._replace(end_s=9800))
    assert nbb.submission_blocker(scaled) == (
        "its agent stage ends 9800 s after the start, a development scale: a competition rerun, "
        "which runs AGENT's times too, would end the agent hours before 12 h (a production agent "
        f"ends at {nbb.COMPETITION_SOLVER_END_S} s or later)"
    )
    for name in (HYBRID_CONFIG, AGENT_ONLY_CONFIG):
        assert nbb.submission_blocker(parsed(name)) is None


def test_the_hybrid_config_spells_out_the_spikes_inputs():
    """The download outputs and the engine are P6.3-02's: the same notebooks at version 1, the
    revision, the vLLM version, the interpreter (configs/P6.3-02*.yaml on task/P6.3b)."""
    text = (REPO / HYBRID_CONFIG).read_text(encoding="utf-8")
    for fact in (MODEL_NOTEBOOK, WHEELS_NOTEBOOK, REVISION, "0.31.0", "/usr/bin/python3.11"):
        assert fact in text


@pytest.mark.parametrize("name", HYBRID_CONFIGS)
def test_every_p6_5_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == 18 + 12 + 8, len(keys)  # top level, agent, model and wheels
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


# The config rules.


def model(**changes):
    raw = {"notebook": MODEL_NOTEBOOK, "version": 1, "folder": "w", "revision": REVISION}
    return {**raw, **changes}


def wheels(**changes):
    raw = {"notebook": WHEELS_NOTEBOOK, "version": 1, "folder": "w", "vllm": "0.31.0"}
    return {**raw, **changes}


@pytest.mark.parametrize(
    ("agent_changes", "message"),
    [
        ({"enabled": "yes"}, "agent: enabled must be true or false"),
        ({"nvarc_share_s": 599}, "nvarc_share_s must be 0 (agent only) or an integer from 600"),
        ({"nvarc_share_s": 41401}, "to 41400, the competition's soft deadline"),
        ({"nvarc_share_s": True}, "nvarc_share_s must be 0"),
        ({"nvarc_share_s": 21600.0}, "nvarc_share_s must be 0"),
        ({"term_s": 29}, "term_s must be an integer from 30 to 600"),
        ({"term_s": 601}, "term_s must be an integer from 30 to 600"),
        ({"end_s": 42751}, "end_s must be an integer from 600 to 42750: SIGKILL follows"),
        ({"end_s": 599}, "end_s must be an integer from 600 to"),
        ({"min_stage_s": 59}, "min_stage_s must be an integer from 60"),
        ({"install_timeout_s": 7201}, "install_timeout_s must be an integer from 60 to 7200"),
        ({"config": "agent_default.yaml"}, "config must be a .yaml file under configs/"),
        ({"config": "configs/../x.yaml"}, "config must be a .yaml file under configs/"),
        ({"model": {"notebook": MODEL_NOTEBOOK}}, "agent model must have exactly the keys"),
        ({"model": model(notebook="someone/x")}, "agent model: notebook must be <owner>/<slug>"),
        ({"model": model(version=0)}, "agent model: version must be an integer >= 1"),
        ({"model": model(folder="a/b")}, "agent model: folder must be one folder name"),
        ({"model": model(revision="91bd022d")}, "revision must be a full commit SHA"),
        ({"wheels": wheels(vllm=0.31)}, "vllm must be a version such as"),
        ({"wheels": wheels(notebook=MODEL_NOTEBOOK)}, "model and wheels must be two notebooks"),
        (
            {"wheels": wheels(notebook="alyahya737/arc26-submission")},
            "is the notebook itself; Kaggle refuses a notebook's own output",
        ),
        (
            {"wheels": wheels(notebook="sorokin/pip-install-unsloth-flash-patch")},
            "in both kernel_source_versions",
        ),
        ({"base_python": "python3.11"}, "base_python must be an absolute path"),
        ({"base_python": "/usr/bin/../python"}, "base_python must be an absolute path"),
        ({"scratch_dir": "/kaggle/working/agent"}, "scratch_dir must be an absolute path outside"),
        ({"scratch_dir": "/kaggle/input"}, "scratch_dir must be an absolute path outside"),
        ({"scratch_dir": "tmp"}, "scratch_dir must be an absolute path outside"),
        ({"min_driver": "580"}, "min_driver must be a driver version"),
        ({"min_driver": 580.65}, "min_driver must be a driver version"),
        ({"extra": 1}, "agent must be null or have exactly the keys enabled, nvarc_share_s"),
        ({"enabled": DROP}, "agent must be null or have exactly the keys"),
    ],
)  # fmt: skip
def test_a_malformed_agent_is_refused(agent_changes, message):
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(config_text({}, agent_changes), Path("c.yaml"))
    assert message in str(refused.value)
    assert str(refused.value).startswith("c.yaml: agent")


@pytest.mark.parametrize("agent", [[], "on", 1])
def test_an_agent_that_is_not_a_mapping_is_refused(agent):
    with pytest.raises(nbb.BuildError, match="agent must be null or have exactly the keys"):
        nbb.parse_config(config_text({"agent": agent}), Path("c.yaml"))


def test_an_agent_beside_arms_is_refused():
    arms = [BUILD_TESTS.REFERENCE_ARM]
    with pytest.raises(nbb.BuildError) as refused:
        nbb.parse_config(config_text({"arms": arms, "scope": DROP}), Path("c.yaml"))
    assert str(refused.value) == (
        "c.yaml: agent and arms exclude each other: a series compares NVARC's scopes in "
        "development runs, while the agent stage follows the single run in both modes"
    )


def test_a_script_config_cannot_set_an_agent():
    raw = yaml.safe_load((REPO / HYBRID_CONFIG).read_text(encoding="utf-8"))["agent"]
    with pytest.raises(nbb.BuildError, match=re.escape("unknown keys ['agent']")):
        nbb.parse_config(BUILD_TESTS.script_text({"agent": raw}), Path("c.yaml"))


def test_a_null_agent_is_no_agent():
    null = nbb.parse_config(config_text({"agent": None}), Path("c.yaml"))
    assert null.agent is None
    assert null == nbb.parse_config(config_text({"agent": DROP}), Path("c.yaml"))
    assert null == parsed(HYBRID_CONFIG)._replace(agent=None)


# The agent's run config, read and checked by resolve_agent.


def test_resolve_agent_embeds_the_run_config_and_its_sha256():
    config = resolved()
    assert config.agent.config_data == yaml.safe_load(agent_config_bytes())
    assert config.agent.config_sha256 == sha256(agent_config_bytes())
    unread = config.agent._replace(config_data=None, config_sha256=None)
    assert unread == parsed(HYBRID_CONFIG).agent
    build = config.agent.build_settings()
    assert build["config_sha256"] == sha256(agent_config_bytes())
    assert build["model"]["revision"] == REVISION and build["wheels"]["vllm"] == "0.31.0"
    # Any other config passes through unchanged.
    assert nbb.resolve_agent(parsed(P26_CONFIG), read_repo) == parsed(P26_CONFIG)


def test_the_run_configs_json_copy_reads_back_through_yaml():
    data = resolved().agent.config_data
    assert yaml.safe_load(json.dumps(data)) == data


def run_config_reader(change):
    """A reader serving the agent's run config with ``change`` applied to its parsed content."""
    data = yaml.safe_load(agent_config_bytes())
    change(data)
    text = yaml.safe_dump(data).encode("utf-8")
    return lambda name: text if name == AGENT_CONFIG else read_repo(name)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("deadline", None, None, "the agent stage needs deadline with final_margin_s, grace_s"),
        ("server", "stop_timeout_s", DROP, "the agent stage needs server with"),
        ("deadline", "final_margin_s", -1, "deadline final_margin_s must be a number >= 0"),
        ("deadline", "grace_s", "60", "deadline grace_s must be a number >= 0"),
        ("server", "boot_timeout_s", 0, "server boot_timeout_s must be a positive number"),
        ("server", "tensor_parallel_size", 0, "tensor_parallel_size must be an integer >= 1"),
        ("server", "tensor_parallel_size", 8, "needs 8 GPUs, but NvidiaL4 has 4"),
        ("server", "stop_timeout_s", 95, "agent term_s 120 must exceed the engine's"),
        ("deadline", "final_margin_s", 19000, "the agent stage could not start"),
        ("server", "extra", 0.00001, "its JSON copy reads back differently"),
    ],
)  # fmt: skip
def test_resolve_agent_refuses_a_run_config_the_plan_cannot_use(section, key, value, message):
    def change(data):
        if key is None:
            data.pop(section)
        elif value is DROP:
            data[section].pop(key)
        else:
            data[section][key] = value

    with pytest.raises(nbb.BuildError, match=re.escape(message)):
        nbb.resolve_agent(parsed(HYBRID_CONFIG), run_config_reader(change))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (b"\xff", "is not UTF-8 text"),
        (b"seed: 0  # TODO\n", "still has TODO"),
        (b"a: [\n", "is not valid YAML"),
        (b"- 1\n", "must hold a mapping"),
    ],
)
def test_resolve_agent_refuses_a_malformed_run_config_file(text, message):
    def read(name):
        return text if name == AGENT_CONFIG else read_repo(name)

    with pytest.raises(nbb.BuildError, match=re.escape(message)):
        nbb.resolve_agent(parsed(HYBRID_CONFIG), read)


def test_resolve_agent_refuses_a_run_config_missing_from_the_commit():
    def read(name):
        raise FileNotFoundError(name)

    with pytest.raises(nbb.BuildError, match="cannot read configs/agent_default.yaml"):
        nbb.resolve_agent(parsed(HYBRID_CONFIG), read)


def test_a_hybrid_build_needs_its_run_config_read():
    config = parsed(HYBRID_CONFIG)
    with pytest.raises(nbb.BuildError, match="was not read"):
        nbb.build_info(
            config, tag="t", commit="0" * 40, config_file=HYBRID_CONFIG, config_sha256="2" * 64,
            wheel=SCOPE_TESTS.GOLDEN_WHEEL, wheel_sha256="1" * 64,
        )  # fmt: skip
    with pytest.raises(nbb.BuildError, match="was not read"):
        nbb.build_notebook(config, BUILD_STUB, SCOPE_TESTS.SUBMIT_STAND_IN)


# The time arithmetic.


def plan(agent_settings, data):
    return nbb.agent_plan(
        agent_settings,
        data["deadline"]["final_margin_s"],
        data["server"]["boot_timeout_s"],
        nbb.NVARC_KILL_GAP_S,
        nbb.WRITE_RESERVE_S,
        nbb.AGENT_KILL_WAIT_S,
        nbb.SESSION_LIMIT_S,
    )


def test_the_hybrid_plan():
    agent = resolved().agent
    assert plan(agent.stage_settings(), agent.config_data) == {
        "nvarc_end_s": 21600,
        "nvarc_kill_s": 22200,
        "agent_start_s": 22500,
        "install_end_s": 24000,
        "boot_end_s": 25800,
        "generation_stop_s": 40200,
        "agent_end_s": 42000,
        "agent_kill_s": 42120,
        "notebook_end_s": 42450,
        "problems": [],
    }
    only = resolved(AGENT_ONLY_CONFIG).agent
    assert plan(only.stage_settings(), only.config_data)["agent_start_s"] == 300


def test_every_plan_the_builder_accepts_ends_before_kaggles_limit_in_order():
    """Over a grid of shares, ends, graces, stage minimums, install timeouts, and final margins:
    each agent the builder accepts (parse_agent, then resolve_agent) has a plan whose every time
    follows the one before and whose notebook ends by 12 h, NVARC's deadlines within the
    competition's; the rest break a rule of the plan or a bound of parse_agent."""
    base = parsed(HYBRID_CONFIG)
    raw = yaml.safe_load((REPO / HYBRID_CONFIG).read_text(encoding="utf-8"))["agent"]
    data = yaml.safe_load(agent_config_bytes())
    data["server"]["stop_timeout_s"] = 1
    texts = {}
    for final in (0, 1800, 9000):
        data["deadline"]["final_margin_s"] = final
        texts[final] = yaml.safe_dump(data).encode("utf-8")
    order = [
        "nvarc_end_s", "nvarc_kill_s", "agent_start_s", "install_end_s", "boot_end_s",
        "generation_stop_s", "agent_end_s", "agent_kill_s", "notebook_end_s",
    ]  # fmt: skip
    accepted = refused = 0
    grid = itertools.product(
        (0, 600, 21600, 41400),  # nvarc_share_s
        (3600, 42000, 42750),  # end_s
        (30, 120),  # term_s
        (60, 12000),  # min_stage_s
        (60, 7200),  # install_timeout_s
        (0, 1800, 9000),  # the run config's final_margin_s
    )
    for share, end, term, stage, install, final in grid:
        changes = {
            "nvarc_share_s": share, "end_s": end, "term_s": term, "min_stage_s": stage,
            "install_timeout_s": install,
        }  # fmt: skip

        def read(name, text=texts[final]):
            return text if name == AGENT_CONFIG else read_repo(name)

        try:
            agent = nbb.parse_agent({**raw, **changes}, base._asdict(), Path("c.yaml"))
            config = nbb.resolve_agent(base._replace(agent=agent), read)
        except nbb.BuildError:
            refused += 1
            continue
        accepted += 1
        times = plan(config.agent.stage_settings(), config.agent.config_data)
        values = [times[key] for key in order]
        assert times["problems"] == [] and values == sorted(values), times
        assert times["notebook_end_s"] <= nbb.SESSION_LIMIT_S
        assert times["generation_stop_s"] - times["agent_start_s"] >= stage
        end_s, kill_s = nbb.nvarc_deadlines(
            nbb.COMPETITION_SOLVER_END_S, nbb.COMPETITION_SOLVER_KILL_S,
            config.agent.stage_settings(), nbb.NVARC_KILL_GAP_S,
        )  # fmt: skip
        assert (end_s, kill_s) == (share, share + nbb.NVARC_KILL_GAP_S)
        assert end_s <= nbb.COMPETITION_SOLVER_END_S and kill_s <= nbb.COMPETITION_SOLVER_KILL_S
    assert accepted > 40 and refused > 40


def test_nvarc_deadlines_follow_the_share_only_when_the_agent_follows():
    agent = {"enabled": True, "nvarc_share_s": 21600}
    gap = nbb.NVARC_KILL_GAP_S
    assert nbb.nvarc_deadlines(41400, 42000, agent, gap) == (21600, 22200)
    # Only ever shorter: a development run's shorter deadlines stand.
    assert nbb.nvarc_deadlines(3600, 4500, agent, gap) == (3600, 4500)
    assert nbb.nvarc_deadlines(41400, 22000, agent, gap) == (21600, 22000)
    # No agent stage to follow: NVARC keeps its deadlines.
    assert nbb.nvarc_deadlines(41400, 42000, None, gap) == (41400, 42000)
    assert nbb.nvarc_deadlines(41400, 42000, {**agent, "enabled": False}, gap) == (41400, 42000)
    assert nbb.nvarc_deadlines(41400, 42000, {**agent, "nvarc_share_s": 0}, gap) == (0, 600)


def test_a_plan_problem_names_what_breaks():
    agent = {
        "nvarc_share_s": 41400, "end_s": 42000, "term_s": 120, "min_stage_s": 1800,
        "install_timeout_s": 1500,
    }  # fmt: skip
    problems = nbb.agent_plan(agent, 1800, 1800, 600, 300, 30, 43200)["problems"]
    assert problems == [
        "the agent stage could not start: after NVARC's hard stop (42000 s) and its write (300 "
        "s), -2100 s are left before the generation stop at 40200 s (end_s minus "
        "final_margin_s), less than min_stage_s (1800 s)"
    ]
    late = nbb.agent_plan({**agent, "nvarc_share_s": 0, "end_s": 42900}, 0, 1, 600, 300, 30, 43200)
    assert late["problems"] == [
        "the notebook could end at 43350 s, after Kaggle's 43200 s limit: SIGKILL at 43020 s, "
        "30 s for the process group, 300 s for the final check and the logs"
    ]


# The runtime helpers of the agent stage.


def gpu(driver):
    return {"name": "NVIDIA L4", "memory.total": "23034 MiB", "compute_cap": "8.9",
            "driver_version": driver}  # fmt: skip


def test_driver_refusal_reads_the_driver_of_every_gpu():
    assert nbb.driver_refusal([gpu("580.159.04")] * 4, "580.65.06") is None
    assert nbb.driver_refusal([gpu("580.65.06")], "580.65.06") is None
    assert nbb.driver_refusal([gpu("581.1")], "580.65.06") is None
    assert "535.104.05, older than 580.65.06" in nbb.driver_refusal(
        [gpu("580.159.04"), gpu("535.104.05")], "580.65.06"
    )
    assert "unknown, older" in nbb.driver_refusal([gpu("")], "580.65.06")
    assert "N/A, older" in nbb.driver_refusal([gpu("N/A")], "580.65.06")
    assert nbb.driver_refusal([], "580.65.06") == "nvidia-smi lists no GPU, so no driver version"


def write_model(root, manifest_changes=None, files=None):
    folder = root / "qwen38-27b-int4"
    folder.mkdir(parents=True)
    files = files if files is not None else {"config.json": b"{}", "model.safetensors": b"x" * 9}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    manifest = {
        "ok": True, "revision": REVISION,
        "files": [{"name": name, "size": len(data)} for name, data in files.items()],
    }  # fmt: skip
    (root / "MANIFEST.json").write_text(json.dumps({**manifest, **(manifest_changes or {})}))


def test_check_model_output_reads_the_manifest_and_the_sizes(tmp_path):
    write_model(tmp_path)
    folder, facts = nbb.check_model_output(str(tmp_path), "qwen38-27b-int4", REVISION)
    assert folder == str(tmp_path / "qwen38-27b-int4")
    assert facts == {"folder": folder, "files": 2, "bytes": 11}


@pytest.mark.parametrize(
    ("manifest", "files", "message"),
    [
        ({"ok": False}, None, "does not say ok"),
        ({"revision": "1" * 40}, None, "names revision '1111"),
        ({"files": []}, None, "no files listed or no config.json"),
        ({"files": [{"name": "config.json", "size": 2}, {"name": "w.bin", "size": 1}]}, None,
         "w.bin missing"),
        ({"files": [{"name": "config.json", "size": 5}]}, None, "config.json has 2 bytes, not 5"),
        (None, {"model.safetensors": b"x"}, "no files listed or no config.json"),
    ],
)  # fmt: skip
def test_check_model_output_refuses_a_bad_output(tmp_path, manifest, files, message):
    write_model(tmp_path, manifest, files)
    with pytest.raises(ValueError, match=re.escape(message)):
        nbb.check_model_output(str(tmp_path), "qwen38-27b-int4", REVISION)


def test_check_model_output_refuses_a_missing_manifest(tmp_path):
    with pytest.raises(ValueError, match="cannot read .*MANIFEST.json"):
        nbb.check_model_output(str(tmp_path), "qwen38-27b-int4", REVISION)


def write_wheels(root, manifest_changes=None, skip=()):
    house = root / "wheelhouse"
    house.mkdir(parents=True)
    for name in ("install_wheelhouse.py", "requirements.txt"):
        if name not in skip:
            (house / name).write_text("x")
    manifest = {
        "ok": True,
        "files": [{"name": "vllm", "version": "0.31.0"}, {"name": "torch", "version": "2.13.0"},
                  {"file": "requirements.txt"}],
        "install_check": {"seconds": 41.5},
    }  # fmt: skip
    (house / "MANIFEST.json").write_text(json.dumps({**manifest, **(manifest_changes or {})}))


def test_check_wheelhouse_reads_the_manifest_and_the_installer(tmp_path):
    write_wheels(tmp_path)
    folder, facts = nbb.check_wheelhouse(str(tmp_path), "wheelhouse", "0.31.0")
    assert folder == str(tmp_path / "wheelhouse")
    assert facts == {"folder": folder, "wheels": 2, "torch": "2.13.0", "install_check_s": 41.5}


@pytest.mark.parametrize(
    ("manifest", "skip", "message"),
    [
        ({"ok": False}, (), "does not say ok: the wheelhouse or its install check failed"),
        ({"files": [{"name": "vllm", "version": "0.30.0"}]}, (), "lists vllm '0.30.0', not 0.31.0"),
        ({"files": []}, (), "lists vllm None"),
        (None, ("install_wheelhouse.py",), "has no install_wheelhouse.py"),
        (None, ("requirements.txt",), "has no requirements.txt"),
    ],
)  # fmt: skip
def test_check_wheelhouse_refuses_a_bad_output(tmp_path, manifest, skip, message):
    write_wheels(tmp_path, manifest, skip)
    with pytest.raises(ValueError, match=re.escape(message)):
        nbb.check_wheelhouse(str(tmp_path), "wheelhouse", "0.31.0")


COUNTS = {"aaaaaaaa": 1, "bbbbbbbb": 2}
NVARC = {
    "aaaaaaaa": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
    "bbbbbbbb": [
        {"attempt_1": [[2]], "attempt_2": [[3]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}
    ],
}  # fmt: skip


def test_merge_submission_keeps_a_valid_file_as_it_is():
    agent = copy.deepcopy(NVARC)
    agent["bbbbbbbb"][1]["attempt_1"] = [[7, 7]]
    merged, sources = nbb.merge_submission(agent, NVARC, COUNTS, submit.grid_problem)
    assert merged == agent and list(merged) == list(COUNTS)
    assert sources == {"current": 6, "fallback": 0, "placeholder": 0, "changed": 1}


def test_merge_submission_fills_every_missing_or_invalid_slot_with_nvarcs_grids_in_order():
    agent = {
        "aaaaaaaa": [{"attempt_1": [[9]], "attempt_2": [[10]]}],  # attempt_2 invalid
        "bbbbbbbb": [{"attempt_1": [[5]]}],  # attempt_2 and the second entry missing
        "cccccccc": [{"attempt_1": [[1]], "attempt_2": [[1]]}],  # not a task of the challenges
    }
    merged, sources = nbb.merge_submission(agent, NVARC, COUNTS, submit.grid_problem)
    # An open slot takes NVARC's attempt 1 first (its top grid), then its attempt 2.
    assert merged == {
        "aaaaaaaa": [{"attempt_1": [[9]], "attempt_2": [[1]]}],
        "bbbbbbbb": [
            {"attempt_1": [[5]], "attempt_2": [[2]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}
        ],
    }  # fmt: skip
    assert sources == {"current": 2, "fallback": 4, "placeholder": 0, "changed": 2}
    assert submit.validate(merged, {t: {"test": [{}] * n} for t, n in COUNTS.items()}) == []


def test_merge_submission_never_keeps_a_placeholder_over_an_nvarc_grid():
    """The probe case: a run that could not read NVARC's file writes [[0]] in every slot it did
    not solve; each of those slots takes NVARC's grids, its attempt 1 first, never repeating a
    grid the entry holds."""
    agent = {
        "aaaaaaaa": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
        "bbbbbbbb": [
            {"attempt_1": [[7]], "attempt_2": [[0]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}
        ],
    }  # fmt: skip
    merged, sources = nbb.merge_submission(agent, NVARC, COUNTS, submit.grid_problem)
    assert merged == {
        "aaaaaaaa": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
        "bbbbbbbb": [
            {"attempt_1": [[7]], "attempt_2": [[2]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}
        ],
    }  # fmt: skip
    assert sources == {"current": 4, "fallback": 2, "placeholder": 0, "changed": 1}
    # NVARC's attempt 2 alone, or both of NVARC's grids, never repeated within an entry.
    nvarc = {"aaaaaaaa": [{"attempt_1": [[0]], "attempt_2": [[4]]}]}
    merged, _ = nbb.merge_submission(
        {"aaaaaaaa": [{"attempt_1": [[6]], "attempt_2": [[0]]}]}, nvarc, {"aaaaaaaa": 1},
        submit.grid_problem,
    )  # fmt: skip
    assert merged == {"aaaaaaaa": [{"attempt_1": [[6]], "attempt_2": [[4]]}]}
    nvarc = {"aaaaaaaa": [{"attempt_1": [[6]], "attempt_2": [[4]]}]}
    merged, _ = nbb.merge_submission(
        {"aaaaaaaa": [{"attempt_1": [[6]], "attempt_2": [[0]]}]}, nvarc, {"aaaaaaaa": 1},
        submit.grid_problem,
    )  # fmt: skip
    assert merged == {"aaaaaaaa": [{"attempt_1": [[6]], "attempt_2": [[4]]}]}


def test_merge_submission_with_nvarc_first_lets_the_agent_fill_nvarcs_placeholders():
    """The agent-only fallback: NVARC's selection wins, the agent's answers fill its [[0]]s."""
    agent = {
        "aaaaaaaa": [{"attempt_1": [[8]], "attempt_2": [[0]]}],
        "bbbbbbbb": [
            {"attempt_1": [[2]], "attempt_2": [[0]]}, {"attempt_1": [[9]], "attempt_2": [[0]]}
        ],
    }  # fmt: skip
    merged, sources = nbb.merge_submission(NVARC, agent, COUNTS, submit.grid_problem)
    assert merged == {
        "aaaaaaaa": [{"attempt_1": [[1]], "attempt_2": [[8]]}],
        "bbbbbbbb": [
            {"attempt_1": [[2]], "attempt_2": [[3]]}, {"attempt_1": [[9]], "attempt_2": [[0]]}
        ],
    }  # fmt: skip
    assert sources["fallback"] == 2


@pytest.mark.parametrize("current", [None, [], "garbage", {"aaaaaaaa": "x"}])
def test_merge_submission_restores_nvarcs_file_from_anything_unusable(current):
    merged, sources = nbb.merge_submission(current, NVARC, COUNTS, submit.grid_problem)
    assert merged == NVARC and sources["fallback"] == 6 and sources["changed"] == 0


def test_merge_submission_without_either_file_writes_placeholders():
    merged, sources = nbb.merge_submission(None, None, COUNTS, submit.grid_problem)
    assert merged == submit.repair(None, {t: {"test": [{}] * n} for t, n in COUNTS.items()})
    assert sources == {"current": 0, "fallback": 0, "placeholder": 6, "changed": 0}


def test_candidates_json_writes_the_candidates_as_dump_run_logs_does(tmp_path):
    np = pytest.importorskip("numpy")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    row = {"beam_score": np.float64(0.5), "score_aug": [1.0, 2.0], "solution": np.array([[1, 2]])}
    with bz2.BZ2File(outputs / "aaaaaaaa_0.rot90", "w") as handle:
        pickle.dump([row], handle)
    (outputs / "bbbbbbbb_1.transpose").write_bytes(pickle.dumps([{**row, "beam_score": 1}]))
    (outputs / "cccccccc_0.cut").write_bytes(bz2.compress(pickle.dumps([row]))[:20])
    (outputs / "dddddddd_0.big").write_bytes(b"x" * 10001)
    target = tmp_path / "scratch" / "nvarc_candidates.json"
    report = nbb.candidates_json(str(outputs), str(target), 10000)
    assert report["written"] is True and report["files"] == 2
    assert report["skipped"] == [str(outputs / "dddddddd_0.big")]
    assert len(report["errors"]) == 1 and report["errors"][0].startswith("cccccccc_0.cut")
    data = json.loads(target.read_text())
    assert data == {
        "aaaaaaaa_0.rot90": [{"beam_score": 0.5, "score_aug": [1.0, 2.0], "solution": [[1, 2]]}],
        "bbbbbbbb_1.transpose": [{"beam_score": 1, "score_aug": [1.0, 2.0], "solution": [[1, 2]]}],
    }
    assert report["bytes"] == target.stat().st_size


def test_candidates_json_without_candidates_writes_nothing_and_removes_a_stale_file(tmp_path):
    target = tmp_path / "nvarc_candidates.json"
    target.write_text("{}")
    report = nbb.candidates_json(str(tmp_path / "none"), str(target), 100)
    assert report == {"written": False, "files": 0, "skipped": [], "errors": [], "bytes": 0}
    assert not target.exists()


def test_clean_env_drops_the_images_python_path_and_pip_settings():
    env = nbb.clean_env(
        {"PATH": "/bin", "PYTHONPATH": "/kaggle/lib", "PIP_INDEX_URL": "x", "VIRTUAL_ENV": "/v",
         "PYTHONHOME": "/h", "PYTHONSTARTUP": "s", "PYTHONUSERBASE": "u", "KEEP": "1"}
    )  # fmt: skip
    assert env == {"PATH": "/bin", "KEEP": "1", "PYTHONNOUSERSITE": "1"}


def child(code):
    """A command that runs ``code`` in a fresh interpreter."""
    return [sys.executable, "-S", "-c", code]


def test_run_supervised_copies_the_output_and_returns_the_exit_code(capsys):
    code, ended = nbb.run_supervised(child("print('hello'); raise SystemExit(3)"), 30, 5)
    assert (code, ended) == (3, "exited")
    assert "hello" in capsys.readouterr().out


def test_run_supervised_sends_sigterm_at_the_timeout_and_a_command_may_clean_up(capsys):
    script = (
        "import signal, sys, time\n"
        "def stop(signum, frame):\n"
        "    print('cleaning up', flush=True)\n"
        "    sys.exit(143)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('running', flush=True)\n"
        "time.sleep(60)\n"
    )
    began = time.monotonic()
    code, ended = nbb.run_supervised(child(script), 1.0, 10)
    assert (code, ended) == (143, "terminated") and time.monotonic() - began < 8
    assert "cleaning up" in capsys.readouterr().out


def test_run_supervised_kills_a_command_that_ignores_sigterm():
    script = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n"
    began = time.monotonic()
    code, ended = nbb.run_supervised(child(script), 1.0, 1.0)
    assert (code, ended) == (-signal.SIGKILL, "killed") and time.monotonic() - began < 10


def test_run_supervised_does_not_wait_for_a_child_left_holding_the_output(tmp_path):
    """The command exits, a process it started keeps its output open: after a few seconds the
    group's leftovers are killed and the call returns."""
    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, '-S', '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
    )
    began = time.monotonic()
    code, ended = nbb.run_supervised(child(script), 60, 5)
    assert (code, ended) == (0, "exited") and time.monotonic() - began < 15
    assert LOCAL.gone(int(pid_file.read_text()))


def test_run_supervised_reports_a_command_that_cannot_start():
    assert nbb.run_supervised(["/nonexistent/program"], 10, 1) == (127, "not started")


def engine(tmp_path, argv=None):
    """A process in a session of its own with its pid file, as P6.4's server records vLLM."""
    argv = argv or [sys.executable, "-S", "-c", "import time; time.sleep(60)"]
    process = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL)
    pid_file = tmp_path / "engine.pid"
    pid_file.write_text(json.dumps({"pid": process.pid, "argv": argv}))
    return process, pid_file


def test_stop_engine_stops_the_group_its_pid_file_names(tmp_path):
    process, pid_file = engine(tmp_path)
    try:
        message = nbb.stop_engine(str(pid_file), 5)
        assert message == (
            f"stopped the engine's process group {process.pid}, which the agent left running"
        )
        assert process.wait(5) == -signal.SIGTERM
        assert not pid_file.exists()
    finally:
        process.kill()


def test_stop_engine_leaves_a_process_that_reused_the_pid_alone(tmp_path):
    process, pid_file = engine(tmp_path)
    try:
        record = json.loads(pid_file.read_text())
        pid_file.write_text(json.dumps({**record, "argv": ["/opt/vllm/bin/vllm", "serve"]}))
        assert nbb.stop_engine(str(pid_file), 5) == f"the engine (pid {process.pid}) no longer runs"
        assert process.poll() is None and not pid_file.exists()
    finally:
        process.kill()
        process.wait()


def test_stop_engine_without_a_pid_file_or_with_a_dead_engine(tmp_path):
    assert nbb.stop_engine(str(tmp_path / "engine.pid"), 1) is None
    process, pid_file = engine(tmp_path, [sys.executable, "-S", "-c", "pass"])
    process.wait()
    assert nbb.stop_engine(str(pid_file), 1) == f"the engine (pid {process.pid}) no longer runs"
    pid_file.write_text("{not json")
    assert nbb.stop_engine(str(pid_file), 1).startswith("unreadable pid file removed")
    assert not pid_file.exists()


def test_stop_engine_kills_what_the_group_holds_once_its_leader_is_gone(tmp_path):
    """vLLM's workers can outlive the API server they belong to; Linux keeps the group id unused
    while they run, so the group is still the engine's."""
    worker_pid = tmp_path / "worker.pid"
    leader = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-S', '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(worker_pid)!r}, 'w').write(str(p.pid))\n"
    )
    process, pid_file = engine(tmp_path, [sys.executable, "-S", "-c", leader])
    process.wait()
    worker = int(worker_pid.read_text())
    try:
        message = nbb.stop_engine(str(pid_file), 5)
        assert message.startswith("stopped the engine's process group") and LOCAL.gone(worker)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(worker, signal.SIGKILL)


def test_copy_logs_copies_what_the_pull_takes_and_keeps_a_large_files_end(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.json").write_text("{}")
    (logs / "engine.log").write_bytes(b"a" * 50 + b"END")
    (logs / "engine.pid").write_text("1")
    (logs / ".hidden.json").write_text("{}")
    (tmp_path / "install.json").write_text("{}")
    report = nbb.copy_logs(
        [str(logs), str(tmp_path / "install.json"), str(tmp_path / "missing.log")],
        str(tmp_path / "out"),
        10,
    )
    assert report == {
        "copied": {"engine.log": 10, "run.json": 2, "install.json": 2},
        "truncated": ["engine.log"],
        "errors": [],
    }
    assert (tmp_path / "out" / "engine.log").read_bytes() == b"aaaaaaaEND"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "engine.log", "install.json", "run.json"
    ]  # fmt: skip


def test_gpu_memory_reads_the_memory_in_use_per_gpu_and_skips_other_lines(monkeypatch):
    calls = []

    def run_logged(command, timeout=None):
        calls.append(command)
        stdout = "0, 3\n1, 15102\nWARNING: a driver message\n2, n/a\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(nbb, "run_logged", run_logged)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert nbb.gpu_memory() == [{"index": 0, "used_mib": 3}, {"index": 1, "used_mib": 15102}]
    assert calls == [
        ["/usr/bin/nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    ]
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert nbb.gpu_memory() is None


def test_wait_for_free_gpus_waits_for_listed_processes_and_for_memory_in_use(monkeypatch):
    """Free means no process listed and every GPU at or under the threshold; both are logged at
    the start and at the end."""
    held = [{"pid": "4242", "used_memory": "15102 MiB"}]
    apps = iter([held, [], []])
    memory = iter([[{"index": 0, "used_mib": 15102}], [{"index": 0, "used_mib": 2048}],
                   [{"index": 0, "used_mib": 1024}]])  # fmt: skip
    monkeypatch.setattr(nbb, "gpu_apps", lambda: next(apps))
    monkeypatch.setattr(nbb, "gpu_memory", lambda: next(memory))
    result = nbb.wait_for_free_gpus(60, 1024, interval_s=0)
    assert result["before"] == held and result["after"] == []
    assert result["memory_before"] == [{"index": 0, "used_mib": 15102}]
    assert result["memory_after"] == [{"index": 0, "used_mib": 1024}]
    assert result["used_max_mib"] == 1024
    # A CPU session (no nvidia-smi) has nothing to wait for; held GPUs stop at the cap.
    monkeypatch.setattr(nbb, "gpu_apps", lambda: None)
    monkeypatch.setattr(nbb, "gpu_memory", lambda: None)
    assert nbb.wait_for_free_gpus(60, 1024, interval_s=0)["after"] is None
    monkeypatch.setattr(nbb, "gpu_memory", lambda: [{"index": 0, "used_mib": 4000}])
    began = time.monotonic()
    result = nbb.wait_for_free_gpus(0.2, 1024, interval_s=0.05)
    assert result["memory_after"] == [{"index": 0, "used_mib": 4000}]
    assert time.monotonic() - began < 5


# The hybrid build's cells.


def hybrid_notebook(config=None):
    config = config or resolved()
    return nbb.build_notebook(config, BUILD_STUB, SCOPE_TESTS.SUBMIT_STAND_IN)


def test_a_hybrid_build_is_its_twin_plus_the_agent_cells_and_the_hybrid_edits():
    config = resolved()
    hybrid = cell_sources(hybrid_notebook(config))
    twin = cell_sources(hybrid_notebook(config._replace(agent=None)))
    added = ["agent-check-note", "agent-check", "agent-note", "agent"]
    assert [name for name in hybrid if name not in added] == list(twin)
    order = list(hybrid)
    assert order.index("agent-check") == order.index("environment") + 2
    assert order.index("agent") == order.index("write") + 2
    assert len(order) == 20 and order[-1] == "logs"
    changed = {name for name in twin if hybrid[name] != twin[name]}
    assert changed == {"config-note", "config", *nbb.HYBRID_EDITS}
    assert set(nbb.HYBRID_EDITS) == {"run-note", "run"}
    for name, edits in nbb.HYBRID_EDITS.items():
        text = twin[name]
        for old, new in edits:
            assert text.count(old) == 1, (name, old)
            text = text.replace(old, new)
        assert text == hybrid[name], name
    assert hybrid["agent-check-note"] == nbb.AGENT_CHECK_NOTE
    assert hybrid["agent-note"] == nbb.AGENT_NOTE and hybrid["agent"] == nbb.AGENT_CODE
    assert "AGENT" not in "".join(twin.values())


def test_the_config_cell_holds_agent_and_build_the_agents_provenance():
    config = resolved()
    namespace = {}
    exec(cell_sources(hybrid_notebook(config))["config"], namespace)
    assert namespace["AGENT"] == config.agent.stage_settings()
    info = nbb.build_info(
        config, tag="t", commit="0" * 40, config_file=HYBRID_CONFIG, config_sha256="2" * 64,
        wheel=SCOPE_TESTS.GOLDEN_WHEEL, wheel_sha256="1" * 64,
    )  # fmt: skip
    assert info["agent"] == config.agent.build_settings()
    assert list(info)[-1] == "agent"
    text = cell_sources(hybrid_notebook(config))["config"]
    assert text.index("AGENT = ") > text.index("ARMS = ") and text.index("AGENT = ") < text.index(
        "BUILD = "
    )
    for line in nbb.AGENT_COMMENT:
        assert len(line) <= 96 and line in text


def test_the_agent_check_cell_embeds_the_run_config_and_the_helpers_verbatim():
    config = resolved()
    cell = cell_sources(hybrid_notebook(config))["agent-check"]
    start = cell.index("AGENT_CONFIG = ")
    end = cell.index("\n\n\n", start)
    assert eval(cell[start + len("AGENT_CONFIG = ") : end]) == config.agent.config_data
    for helper in nbb.AGENT_RUNTIME_HELPERS:
        assert cell.count(inspect.getsource(helper).rstrip()) == 1, helper.__name__
    assert 'os.environ.get("ARC26_AGENT_SCRATCH")' in cell
    namespace = {"os": os, "Path": Path, "BUILD": {}}
    start = cell.index("AGENT_SCRATCH = Path(")
    exec(cell[start : cell.index("\n)", start) + 2], namespace)
    assert namespace["AGENT_SCRATCH"] == Path(nbb.AGENT_DEFAULT_SCRATCH) == Path("/tmp/arc26-agent")


def test_the_first_cell_states_the_plan_and_the_inputs():
    note = cell_sources(hybrid_notebook())["config-note"]
    assert (
        "Hybrid build (task P6.5): after NVARC, the agent solver "
        "(`docs/decisions/ADR-020-agent-solver-pivot.md`) runs in both modes, a competition "
        "rerun included: NVARC until 21600 s (its hard stop at 22200 s), then the agent until "
        "42000 s: generation stops at 40200 s, SIGTERM at 42000 s, SIGKILL at 42120 s."
    ) in note
    assert f"`{WHEELS_NOTEBOOK}` (version 1) and `{MODEL_NOTEBOOK}` (version 1)" in note
    assert "every slot the agent left invalid or a `[[0]]` placeholder" in note
    assert "`SOLVER`, `SCOPE`, and `AGENT` apply in both modes." in note
    assert "Not for submission" not in note
    agent_only = cell_sources(hybrid_notebook(resolved(AGENT_ONLY_CONFIG)))["config-note"]
    for text in (
        "Hybrid build (task P6.5): the agent solver (`docs/decisions/ADR-020-agent-solver-pivot"
        ".md`) runs in both modes, a competition rerun included: agent only, the agent until "
        "42000 s",
        "NVARC runs only as the fallback, when the agent stage ends at least 1800 s before "
        "NVARC's soft deadline.",
        "the engine is stopped and NVARC's stage runs with the deadlines it would have had; "
        "P6.2's selection of the agent's attempt log, with NVARC's new attempts as its NVARC "
        "input, then keeps every verified program in attempt 1 and lets NVARC fill the rest.",
        "An agent that runs until its generation stop leaves NVARC no time, so its file has no "
        "NVARC attempt.",
    ):
        assert text in agent_only, text
    assert "NVARC's `submission.json` is written first" not in agent_only
    assert (REPO / "docs" / "decisions" / nbb.AGENT_ADR).is_file()
    unscoped = cell_sources(hybrid_notebook(resolved()._replace(scope=None)))["config-note"]
    assert "`SOLVER` and `AGENT` apply in both modes." in unscoped


def test_a_hybrid_edit_that_no_longer_matches_its_cell_stops_the_build(monkeypatch):
    edits = {**nbb.HYBRID_EDITS, "run": (("    elif not INSTALLED:\n", "x"),) * 2}
    monkeypatch.setattr(nbb, "HYBRID_EDITS", edits)
    with pytest.raises(RuntimeError, match="the run cell changed: update HYBRID_EDITS"):
        hybrid_notebook()


def test_every_hybrid_code_cell_compiles_and_the_helpers_are_in_the_check_cell():
    notebook = hybrid_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    run = cell_sources(notebook)["run"]
    assert "nvarc_deadlines(" in run and 'AGENT["nvarc_share_s"] == 0' in run


def test_the_fallback_reads_the_attempt_log_p6_4_writes_and_never_its_exit_summary():
    """run.json exists only after the agent's exit; the attempt log survives a kill."""
    from arc26.agent import run as agent_run

    cell = cell_sources(hybrid_notebook())["agent"]
    fallback = cell[cell.index("def nvarc_fallback(") : cell.index("def final_check(")]
    assert f'AGENT_LOGS / "{agent_run.ATTEMPTS_FILE}"' in fallback
    assert "from arc26.agent.select import select_files" in fallback
    assert f'"{agent_run.SUMMARY_FILE}"' not in fallback and 'record.get("run")' not in fallback
    assert "wait_for_free_gpus(GPU_FREE_WAIT_S, GPU_USED_MAX_MIB)" in fallback


# The metadata and the manifest.


def test_the_hybrid_metadata_attach_both_download_outputs_and_pass_the_gateway():
    config = resolved()
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)  # runs the gateway's metadata checks
    assert meta["id"] == nbb.SUBMISSION_NOTEBOOK and meta["machine_shape"] == "NvidiaL4"
    assert meta["kernel_sources"] == [
        "sorokin/pip-install-unsloth-flash-patch/2", f"{MODEL_NOTEBOOK}/1", f"{WHEELS_NOTEBOOK}/1"
    ]  # fmt: skip
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p65"]
    plain = bs.kernel_metadata(config._replace(agent=None), pulled)
    assert plain["kernel_sources"] == ["sorokin/pip-install-unsloth-flash-patch/2"]


def hybrid_files(name=HYBRID_CONFIG):
    """What the builders write for a hybrid config from fixed provenance, as golden_files does."""
    data = (REPO / name).read_bytes()
    config = resolved(name)
    info = nbb.build_info(
        config, tag="golden", commit=SCOPE_TESTS.GOLDEN_COMMIT, config_file=name,
        config_sha256=sha256(data), wheel=SCOPE_TESTS.GOLDEN_WHEEL,
        wheel_sha256=SCOPE_TESTS.GOLDEN_WHEEL_SHA256,
    )  # fmt: skip
    notebook = nbb.build_notebook(config, info, SCOPE_TESTS.SUBMIT_STAND_IN)
    pulled = json.loads((REPO / config.reference_metadata).read_text(encoding="utf-8"))
    meta = bs.kernel_metadata(config, pulled)
    files = {
        nbb.NOTEBOOK_FILE: nbb.notebook_bytes(notebook), bs.METADATA_FILE: bs._json_bytes(meta)
    }  # fmt: skip
    return bs.render_manifest(
        tag="golden", build=info, commit_time=0, wheel_commit=SCOPE_TESTS.GOLDEN_COMMIT,
        wheel_epoch=0, config=config, builder="uv build --wheel --offline",
        generator="hatchling 1.27.0", metadata=meta, files=files,
    ).decode("utf-8")  # fmt: skip


def test_the_hybrid_manifest_states_the_agent_its_inputs_and_the_sessions_timeout():
    manifest = hybrid_files()
    sha = sha256(agent_config_bytes())
    for row in (
        "| fit to submit | yes |",
        "| agent (both modes) | `AGENT`, enabled: NVARC until 21600 s (its hard stop at 22200 s), "
        "then the agent until 42000 s: generation stops at 40200 s, SIGTERM at 42000 s, SIGKILL "
        "at 42120 s; the stage starts with at least 1800 s left before the generation stop; "
        "install timeout 1500 s; task P6.5, ADR-020 |",
        f"| agent run config | `{AGENT_CONFIG}`, sha256 `{sha}`, embedded in the agent check "
        "cell |",
        f"| agent inputs | weights `{MODEL_NOTEBOOK}` version 1 (`qwen38-27b-int4`, revision "
        f"`{REVISION}`); wheelhouse `{WHEELS_NOTEBOOK}` version 1 (`wheelhouse`, vLLM 0.31.0) |",
        "| agent engine | a venv made by `/usr/bin/python3.11` in `/tmp/arc26-agent`; NVIDIA "
        "driver 580.65.06 or later |",
        f"- notebook output `{MODEL_NOTEBOOK}/1`, the agent's weights (`qwen38-27b-int4`, "
        "revision `91bd022d5b49`; task P6.5);",
        f"- notebook output `{WHEELS_NOTEBOOK}/1`, the agent's engine wheelhouse (`wheelhouse`, "
        "vLLM 0.31.0; task P6.5);",
        "- install notebook `sorokin/pip-install-unsloth-flash-patch/2` (the package stack, "
        "ADR-010);",
        "`python scripts/kaggle_ops.py push dist/golden --run-id <run-id>`, with the brief's "
        "`Timeout` at least 42450 s",
    ):
        assert row in manifest, row
    assert (
        "| agent (both modes) | `AGENT`, enabled: agent only, the agent until 42000 s: generation "
        "stops at 40200 s, SIGTERM at 42000 s, SIGKILL at 42120 s; NVARC runs only as the "
        "fallback, when the agent stage ends at least 1800 s before NVARC's soft deadline; "
    ) in hybrid_files(AGENT_ONLY_CONFIG)
    cpu = hybrid_files(CPU_CONFIG)
    push = "`python scripts/kaggle_ops.py push dist/golden --timeout 4050` (CPU: no pre-run brief)"
    assert push in cpu
    assert "| fit to submit | no, not for submission: " in cpu


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """dist/P6.5-hybrid/ twice, dist/P6.5-01/, dist/P6.5-agent/, and dist/P2.6-01/, from one
    throwaway repository holding the working tree."""
    if not BUILD_TESTS.BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    repo = BUILD_TESTS.snapshot(tmp_path_factory.mktemp("repo"))
    first = BUILD_TESTS.read_files(bs.build("P6.5-hybrid", repo / HYBRID_CONFIG, repo))
    again = bs.build("P6.5-hybrid", repo / HYBRID_CONFIG, repo)
    only = bs.build("P6.5-agent", repo / AGENT_ONLY_CONFIG, repo)
    p26 = bs.build("P2.6-01", repo / P26_CONFIG, repo)
    dev40 = bs.build("P6.5-01", repo / DEV40_CONFIG, repo)
    return repo, again, first, only, p26, dev40


def test_the_hybrid_build_is_rebuilt_byte_for_byte_and_shares_the_commits_wheel(builds):
    _, folder, first, only, p26, dev40 = builds
    files = BUILD_TESTS.read_files(folder)
    assert set(files) == BUILD_TESTS.FILES and files == first
    wheel = f"arc26-code/{BUILD_TESTS.WHEEL}"
    for other in (p26, only, dev40):
        assert files[wheel] == BUILD_TESTS.read_files(other)[wheel]


def test_the_hybrid_build_passes_the_gateway_and_is_fit_to_submit(builds):
    repo, folder, _, _, _, _ = builds
    meta = ops.load_kernel_metadata(folder.resolve())
    assert meta["id"] == nbb.SUBMISSION_NOTEBOOK and meta["machine_shape"] == "NvidiaL4"
    ops.check_kernel_metadata(meta, True)
    ops.check_flags(meta, ops.accelerator_id(meta, None))
    ops.check_code_file(folder.resolve(), meta)
    ops.check_dataset_folder((folder / "arc26-code").resolve())
    manifest = (folder / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | yes |" in manifest and "| agent (both modes) | `AGENT`" in manifest
    namespace = {}
    exec(BUILD_TESTS.code(folder)["arc26-02-config"], namespace)
    build = namespace["BUILD"]
    assert build["commit"] == BUILD_TESTS.git(repo, "rev-parse", "HEAD").strip()
    assert build["agent"]["config_sha256"] == sha256((repo / AGENT_CONFIG).read_bytes())
    assert namespace["AGENT"]["nvarc_share_s"] == 21600
    ids = [cell["id"] for cell in BUILD_TESTS.cells(folder)]
    assert ids[9] == "arc26-10-agent-check" and ids[17] == "arc26-18-agent"


def test_the_p6_5_01_build_is_a_measurement_never_a_submission(builds):
    _, _, _, _, _, dev40 = builds
    meta = ops.load_kernel_metadata(dev40.resolve())
    assert meta["id"] == "alyahya737/arc26-hybrid-dev" and meta["machine_shape"] == "NvidiaL4"
    ops.check_kernel_metadata(meta, True)
    manifest = (dev40 / "MANIFEST.md").read_text(encoding="utf-8")
    assert "| fit to submit | no, not for submission: its notebook is" in manifest
    assert "a development scale" in manifest
    assert "with the brief's `Timeout` at least 10250 s" in manifest
    note = "".join(BUILD_TESTS.cells(dev40)[0]["source"])
    assert note.count("**Not for submission.**") == 1
    namespace = {}
    exec(BUILD_TESTS.code(dev40)["arc26-02-config"], namespace)
    assert (namespace["AGENT"]["nvarc_share_s"], namespace["AGENT"]["end_s"]) == (3600, 9800)
    assert len(namespace["SUBSET"]) == 40


# The draft pre-run brief.


def test_the_p6_5_01_brief_draft_parses_with_the_gateways_reader():
    text = (REPO / "docs" / "prerun" / "P6.5-01.md").read_text(encoding="utf-8")
    assert "TODO" not in text
    header = ops.brief_header(text)
    config = resolved(DEV40_CONFIG)
    assert header["Run-id"] == "P6.5-01" and header["Config"] == DEV40_CONFIG
    assert header["Kaggle notebook"] == config.kaggle_notebook == "alyahya737/arc26-hybrid-dev"
    assert header["Accelerator"] == config.accelerator == "NvidiaL4"
    bound = plan(config.agent.stage_settings(), config.agent.config_data)["notebook_end_s"]
    assert int(header["Timeout"]) == 10800 > bound == 10250
    assert header["Code-commit"].startswith("pending")
    assert config.kaggle_notebook in ops.briefed_notebooks()
