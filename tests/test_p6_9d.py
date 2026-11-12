"""Task P6.9d: MTP and the 69,632 cap on Submission 4's build (run ``P6.9-04``).

Against the task brief (``docs/tasks/P6.9d.md``) this file checks, on CPU:

- the shot's run config ``configs/agent_shot_P6.9d.yaml`` is Submission 4's
  (``configs/agent_shot_P6.9.yaml``) in every key but ``waves.max_cap`` (69,632) and
  ``server.extra_args`` (task/P6.10's two speculative arguments, verbatim), every value
  commented;
- its time model: the smallest wave, the shot's minimum stage, and so the notebook's
  ``min_stage_s``, the latest start, and the deadlines are P6.9's; every planned start is at
  most 69,632 and at least 65,536; 15 streams fit MTP's Kaggle pool estimate (13 at 94,208);
- the build configs: ``P6.9-04`` is ``P6.9-03`` with only ``agent.config`` changed, fit to
  submit; its CPU twin is ``P6.9-03-cpu`` with the same one change;
- vLLM's command line through the shot's own loader holds ``--speculative-config`` with the local
  copy and with the warm-up's fallback flag after it;
- two builds of each config are byte-identical, the notebooks differ from ``P6.9-03``'s only in
  the provenance and the embedded run config, the code dataset's files are ``P6.9-03``'s;
- end to end on the simulated Kaggle tree (the real ``arc26.agent.shot`` with P6.4's fake engine
  as the venv's ``vllm``), both modes: the engine command holds the speculative flag and the
  local copy, the shot merges over NVARC's placeholders.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


SB = load("_warmup_shot_build", REPO / "tests" / "test_shot_build.py")
WN = load("_p6_9d_warmup_notebook", REPO / "tests" / "test_warmup_notebook.py")
HYB, nbb = SB.HYB, SB.nbb
SHOTN, HYBN = WN.SHOTN, WN.HYBN
SHOT_D = "configs/agent_shot_P6.9d.yaml"
SHOT_4 = "configs/agent_shot_P6.9.yaml"
GPU, CPU = "configs/submission_P6.9-04.yaml", "configs/submission_P6.9-04-cpu.yaml"
GPU_3, CPU_3 = "configs/submission_P6.9-03.yaml", "configs/submission_P6.9-03-cpu.yaml"
SPECULATIVE = ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 1}']
CAP = 69632
# MTP's pool on Kaggle, estimated from P6.10-01's two Modal boots and P6.5-03's Kaggle boot:
# 1,507,328 x 1,699,566 / 1,746,170 (docs/experiments/P6.10-01.md, "KV pool").
MTP_POOL = round(1507328 * 1699566 / 1746170)
# MTP's per-request blocks beyond the tokens, P6.10-02's brief (kv_units): 6.53 of 1,588.3.
MTP_OVERHEAD = 6.53 * 1588.3


def raw(name):
    return yaml.safe_load((REPO / name).read_text(encoding="utf-8"))


def shot():
    return pytest.importorskip("arc26.agent.shot")


# The shot's run config.


def test_the_run_config_is_submission_4s_but_the_cap_and_the_speculative_flag():
    new, old = raw(SHOT_D), raw(SHOT_4)
    assert new["waves"]["max_cap"] == CAP and old["waves"]["max_cap"] == 94208
    assert new["server"]["extra_args"] == SPECULATIVE and old["server"]["extra_args"] == []
    assert new["waves"]["min_cap"] == old["waves"]["min_cap"] == 65536
    new["waves"]["max_cap"], new["server"]["extra_args"] = 94208, []
    assert new == old


def test_the_speculative_flag_is_task_p6_10s_verbatim():
    """The P6.10 config is on another branch: its line is pinned here as task/P6.10 holds it."""
    line = (
        """  extra_args: ["--speculative-config", """
        """'{"method": "mtp", "num_speculative_tokens": 1}']"""
    )
    assert line in (REPO / SHOT_D).read_text(encoding="utf-8").splitlines()
    assert json.loads(SPECULATIVE[1]) == {"method": "mtp", "num_speculative_tokens": 1}


def test_every_run_config_value_has_a_comment():
    lines = (REPO / SHOT_D).read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        assert lines[k - 1].strip().startswith("#") or stripped.endswith(":"), line
    assert "TODO" not in "\n".join(lines)


def test_the_time_model_keeps_p6_9s_smallest_wave_and_minimum_stage():
    module = shot()
    new, old = module.load_config(REPO / SHOT_D), module.load_config(REPO / SHOT_4)
    a, b = module.stage_arithmetic(new), module.stage_arithmetic(old)
    for key in ("setup_s", "smallest_wave_s", "min_shot_s", "final_margin_s"):
        assert a[key] == b[key], key
    assert (a["min_shot_s"], a["smallest_wave_s"]) == (6262.5, 5812.5)
    assert a["full_wave_s"] == 6130.2 and b["full_wave_s"] == 8036.5
    # The notebook's min_stage_s (6,500) still covers the install, the GPU wait, and min_shot_s.
    assert raw(GPU)["agent"]["min_stage_s"] == 6500 >= 138 + 60 + a["min_shot_s"]
    assert new.server.boot_timeout_s == 900 and new.waves.wave_tasks == 15


@pytest.mark.parametrize("left", [9485.0, 8036.5, 7000.0, 6130.2, 6000.0, 5812.5, 5812.0, 3000.0])
def test_every_start_is_p6_9s_cap_at_most_69632_and_none_under_65536(left):
    """The refill queue's planned start (plan_start) at 15 streams: the new cap is P6.9's capped
    at 69,632, and the last start comes at the same time left as P6.9's."""
    module = shot()
    new, old = module.load_config(REPO / SHOT_D), module.load_config(REPO / SHOT_4)
    item, free = ("t", 6771), 10**9
    a = module.plan_start(item, left, new.waves, free, 15)
    b = module.plan_start(item, left, old.waves, free, 15)
    assert a.cap == min(b.cap, CAP)
    assert bool(a.tasks) == bool(b.tasks)
    if a.tasks:
        assert 65536 <= a.cap <= CAP and a.planned_s <= left


def test_fifteen_streams_fit_mtps_pool_at_69632_and_thirteen_at_94208():
    module = shot()
    new, old = module.load_config(REPO / SHOT_D), module.load_config(REPO / SHOT_4)
    queue = [(f"t{i:02d}", 6771) for i in range(20)]
    wave = module.plan_wave(queue, 20000.0, new.waves, MTP_POOL)
    assert (len(wave.tasks), wave.cap) == (15, CAP) and wave.kv_need == 1238205
    assert 15 * (6771 + CAP + 6144 + MTP_OVERHEAD) <= MTP_POOL == 1467099
    wide = module.plan_wave(queue, 20000.0, old.waves, MTP_POOL)
    assert (len(wide.tasks), wide.cap) == (13, 94208)


# The build configs.


@pytest.mark.parametrize("name", [GPU, CPU])
def test_every_p6_9_04_build_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if HYB.re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == 18 + 13 + 8 + 10, len(keys)
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


@pytest.mark.parametrize(("name", "before"), [(GPU, GPU_3), (CPU, CPU_3)], ids=["gpu", "cpu"])
def test_p6_9_04_is_p6_9_03_with_only_the_shot_config_changed(name, before):
    new, old = HYB.parsed(name), HYB.parsed(before)
    assert new.agent.config == SHOT_D and old.agent.config == SHOT_4
    assert new._replace(agent=new.agent._replace(config=old.agent.config)) == old
    assert SB.resolved(name).agent.config_data == raw(SHOT_D)


def test_p6_9_04_is_fit_to_submit_and_its_twin_is_not():
    assert nbb.submission_blocker(HYB.parsed(GPU)) is None
    cpu = HYB.parsed(CPU)
    assert "no accelerator" in nbb.submission_blocker(cpu)
    assert cpu.kaggle_notebook not in HYB.ops.briefed_notebooks()
    assert HYB.parsed(GPU).kaggle_notebook == "alyahya737/arc26-submission"


@pytest.mark.parametrize("fallback", [False, True], ids=["local", "input"])
def test_the_engine_command_holds_the_speculative_flag_with_either_folder(tmp_path, fallback):
    """What the notebook writes for the shot (the run config, with the warm-up's fallback flag
    appended to server.extra_args) gives vLLM's command line through the shot's own loader."""
    module = shot()
    from arc26.agent.server import vllm_command

    data = copy.deepcopy(SB.resolved(GPU).agent.config_data)
    folder = "/kaggle/input/arc26-dl-qwen38-int4/qwen38-27b-int4" if fallback else "/tmp/w/q"
    if fallback:
        data["server"]["extra_args"] = [
            *(data["server"].get("extra_args") or []), "--safetensors-load-strategy", "prefetch"
        ]
    path = tmp_path / "agent_config.json"
    path.write_text(json.dumps(data))
    command = vllm_command("/tmp/v/bin/vllm", folder, module.load_config(path).server)
    assert command[:3] == ["/tmp/v/bin/vllm", "serve", folder]
    start = command.index("--speculative-config")
    tail = SPECULATIVE + (["--safetensors-load-strategy", "prefetch"] if fallback else [])
    assert command[start:] == tail


# The builds.


@pytest.mark.parametrize(("name", "before"), [(GPU, GPU_3), (CPU, CPU_3)], ids=["gpu", "cpu"])
def test_two_builds_are_byte_identical_and_differ_from_p6_9_03s_only_in_the_shot(name, before):
    first = SB.hybrid_golden_files(SB.bs, name)
    assert first == SB.hybrid_golden_files(SB.bs, name)
    old = SB.hybrid_golden_files(SB.bs, before)
    code = f"{SB.bs.CODE_DIR}/{SB.bs.DATASET_METADATA_FILE}"
    assert first[code] == old[code] and first[SB.bs.METADATA_FILE] == old[SB.bs.METADATA_FILE]
    new_cells = HYB.cell_sources(json.loads(first[nbb.NOTEBOOK_FILE]))
    old_cells = HYB.cell_sources(json.loads(old[nbb.NOTEBOOK_FILE]))
    assert set(new_cells) == set(old_cells)
    changed = {cell for cell in new_cells if new_cells[cell] != old_cells[cell]}
    assert changed == {"config-note", "config", "agent-check"}, changed
    embedded = "\n".join(new_cells[cell] for cell in changed)
    assert "--speculative-config" in embedded and "69632" in embedded
    assert "start_warmup(" in new_cells["agent-check"]
    assert "warmup_weights(" in new_cells["agent"]


# End to end on the simulated Kaggle tree.


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    pytest.importorskip("numpy")
    return HYBN.LOCAL.make_wheel(tmp_path_factory.mktemp("build"))


@pytest.fixture(scope="module")
def notebook(tmp_path_factory, wheel):
    """The real-shot build with P6.9-04's run config at the simulation's times (caps and rates
    scaled down as tests/test_warmup_notebook.py's) and the simulation's warm-up; the speculative
    flag stays (the fake engine ignores it)."""
    shot()
    server = pytest.importorskip("arc26.agent.server")
    run_config = raw(SHOT_D)
    run_config["deadline"].update(final_margin_s=1)
    run_config["waves"].update(
        max_cap=4096, min_cap=256, rate_tokens_per_s=1e6, prefill_tokens_per_s=1e9,
        margin_s=0, setup_s=0,
    )  # fmt: skip
    run_config["server"].update(
        port=server.free_port(), boot_timeout_s=30, stop_timeout_s=5, health_interval_s=0.05,
        monitor_interval_s=0.1, hang_s=10,
    )  # fmt: skip
    run_config["engine"].update(max_retries=0)
    run_config["engine_prep"].update(enabled=False, preflight=False)
    config = SHOTN.shot_config(config=SHOT_D, config_data=run_config, warmup=WN.SIM_WARMUP)
    return HYBN.build(tmp_path_factory.mktemp("mtp"), wheel, "mtp", config)


@pytest.mark.parametrize("competition", [False, True], ids=["development", "competition"])
def test_the_shot_boots_with_the_speculative_flag_from_the_local_copy(
    tmp_path, notebook, wheel, competition
):
    run = WN.run_notebook(tmp_path, notebook, wheel, competition)
    stage = WN.assert_shot_merged(run, competition)
    local = run.scratch / "weights" / WN.FOLDER
    assert (stage["weights"]["source"], stage["weights"]["model_dir"]) == ("local", str(local))
    command = WN.engine_command(run)
    assert command[1:3] == ["serve", str(local)]
    assert command[command.index("--speculative-config") :] == SPECULATIVE
    if competition:
        assert HYBN.working_files(run) == ["submission.json"]
