"""Run ``P4.7-01``: one rank-256 NVARC pass on 8 DEV120 tasks where rank 16 leaves outputs empty.

Task P4.7 (``docs/tasks/P4.7.md``, ``docs/prerun/P4.7-01.md``). This file checks, on CPU:

- the build config holds no scope, no agent, and no series, so the single run trains RunConfig's
  defaults, the reference's LoRA settings; the 8-task subset, 4 workers on ``NvidiaL4``, the
  fixed seed, and the split solver; every value commented;
- its build is a development build ("Not for submission") for its own notebook, never
  ``arc26-submission``, and its brief's header names that notebook, the accelerator, and a
  ``Timeout`` that covers the hard stop, the write reserve, and the pre-script allowance;
- the deadlines cover ``P1.1-01``'s puzzle seconds on these tasks in the sorted 4-worker queue
  and the per-puzzle cap's worst case for the last task;
- the CPU twin is the build with only the CPU checks' four values changed;
- the split solver's LoRA, training, and decode settings with RunConfig's defaults equal
  ``P1.1-01``'s (commit ``d94304d``, the moved reference's literals);
- two builds of each config are byte-identical, the notebook's cells are ``P1.1-01``'s and
  ``P2.0-01``'s but the first cell and the Config cell, and the Config cells differ from theirs
  only in the values named.
"""

import ast
import heapq
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


HYB = load("_p4_7_hybrid_build", REPO / "tests" / "test_hybrid_build.py")
LORA = load("_p4_7_lora_scope", REPO / "tests" / "test_lora_scope.py")
nbb, bs, ops = HYB.nbb, HYB.bs, HYB.ops
GPU, CPU = "configs/submission_P4.7-01.yaml", "configs/submission_P4.7-01-cpu.yaml"
P1_1, P2_0 = "configs/submission_P1.1-01.yaml", "configs/submission_P2.0-01.yaml"
BRIEF = REPO / "docs" / "prerun" / "P4.7-01.md"
NOTEBOOK = "alyahya737/arc26-p4-7-r256"
# P1.1-01's puzzle seconds on the 8 tasks (its session log's "[Rank n] finished <id> in <s>s").
P1_1_SECONDS = {
    "136b0064": 209.5, "291dc1e1": 214.5, "581f7754": 437.1, "65b59efc": 942.0,
    "6ffbe589": 254.7, "80a900e0": 601.5, "9385bd28": 559.6, "e12f9a14": 825.0,
}  # fmt: skip
# Seconds from the notebook's start to the last worker's first puzzle, the upper end measured:
# P2.0-01's solver stage (1,632 s) less its longest puzzle (1,231 s) is about 400 s.
START_UP_S = 400
# The decode stop and one DFS batch's budget (arc26/decode.py lines 369 and 129).
PUZZLE_CAP_S, DFS_BATCH_S = 1200, 540


def raw(name):
    return yaml.safe_load((REPO / name).read_text(encoding="utf-8"))


def queue(seconds, workers):
    """Each task's start and the makespan of the solver's queue: sorted ids, the next free worker
    takes the next id (arc26/solver.py's queue; the workers start together)."""
    free = [(0.0, rank) for rank in range(workers)]
    starts = {}
    for key in sorted(seconds):
        at, rank = heapq.heappop(free)
        starts[key] = at
        heapq.heappush(free, (at + seconds[key], rank))
    return starts, max(at for at, _ in free)


def assigned(source, function):
    """The ``name = dict(...)`` and constant assignments of ``function`` in ``source``, evaluated
    with ``cfg`` RunConfig's defaults as the solver receives them; other values are skipped."""
    tree = ast.parse(source)
    (node,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function]
    values = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = eval(
                        compile(ast.Expression(statement.value), "<p4.7>", "eval"),
                        {"cfg": LORA.solver_cfg(), "np": SimpleNamespace(log=math.log)},
                    )
                except (NameError, AttributeError, TypeError, KeyError):
                    continue
    return values


def config_values(cell):
    """The Config cell's top-level assignments, evaluated."""
    values = {}
    for statement in ast.parse(cell).body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            try:
                values[statement.targets[0].id] = ast.literal_eval(statement.value)
            except ValueError:
                continue
    return values


def cells(name):
    files = HYB.SCOPE_TESTS.golden_files(bs, name)
    return HYB.cell_sources(json.loads(files[nbb.NOTEBOOK_FILE])), files


# The build config.


@pytest.mark.parametrize("name", [GPU, CPU])
def test_every_p4_7_build_config_value_has_a_comment(name):
    lines = (REPO / name).read_text(encoding="utf-8").splitlines()
    keys = [i for i, line in enumerate(lines) if HYB.re.match(r"\s*[a-z_0-9]+:", line)]
    assert len(keys) == 13 + 1, len(keys)
    for index in keys:
        assert lines[index - 1].lstrip().startswith("#"), f"no comment above {lines[index]!r}"
    assert "TODO" not in "\n".join(lines)


def test_the_config_is_nvarc_alone_with_the_reference_lora_settings():
    data, config = raw(GPU), HYB.parsed(GPU)
    assert not {"scope", "agent", "arms"} & set(data)
    assert config.scope is None and config.agent is None and config.arms is None
    assert config.solver == "split" and not config.dev240


def test_the_config_runs_the_8_tasks_on_4_workers_on_4xl4():
    config = HYB.parsed(GPU)
    assert config.subset == tuple(sorted(P1_1_SECONDS))
    assert (config.workers, config.accelerator, config.hash_seed) == (4, "NvidiaL4", 0)
    assert config.run_solver and config.kaggle_notebook == NOTEBOOK
    assert config.code_dataset == "alyahya737/arc26-code-p69"
    # P1.1-01's and P6.9-05's inputs: the same competition, model, image, and install notebook.
    for other in (P1_1, "configs/submission_P6.9-05.yaml"):
        before = HYB.parsed(other)
        for key in ("competition", "reference_metadata", "kernel_source_versions", "model_slug"):
            assert getattr(config, key) == getattr(before, key), (other, key)


def test_the_build_is_a_development_build_on_its_own_notebook():
    config = HYB.parsed(GPU)
    blocker = nbb.submission_blocker(config)
    assert blocker is not None and blocker.startswith(f"its notebook is {NOTEBOOK}")
    assert config.kaggle_notebook != nbb.SUBMISSION_NOTEBOOK


def test_the_deadlines_cover_p1_1_01s_seconds_and_the_puzzle_cap():
    config = HYB.parsed(GPU)
    starts, makespan = queue(P1_1_SECONDS, config.workers)
    assert round(makespan, 1) == 1289.2 and round(starts["e12f9a14"], 1) == 464.2
    # 30% slower puzzles than P1.1-01's still end before the soft deadline.
    assert START_UP_S + 1.3 * makespan < config.solver_end_s
    # The last task's decode stops by the per-puzzle cap plus one DFS batch.
    assert START_UP_S + max(starts.values()) + PUZZLE_CAP_S + DFS_BATCH_S < config.solver_end_s
    assert config.solver_kill_s - config.solver_end_s == (
        nbb.COMPETITION_SOLVER_KILL_S - nbb.COMPETITION_SOLVER_END_S
    )


@pytest.mark.skipif(not BRIEF.is_file(), reason="the brief follows its Code-commit")
def test_the_brief_names_the_notebook_accelerator_and_a_timeout_past_the_hard_stop():
    header = ops.brief_header(BRIEF.read_text(encoding="utf-8"))
    config = HYB.parsed(GPU)
    assert header["Run-id"] == "P4.7-01"
    assert header["Kaggle notebook"] == config.kaggle_notebook
    assert header["Accelerator"] == config.accelerator
    floor = config.solver_kill_s + nbb.WRITE_RESERVE_S + nbb.PRE_SCRIPT_ALLOWANCE_S
    assert floor <= int(header["Timeout"]) <= floor + 600
    assert NOTEBOOK in ops.briefed_notebooks()


def test_the_cpu_twin_is_the_build_with_the_cpu_checks_four_values():
    cpu, gpu = HYB.parsed(CPU), HYB.parsed(GPU)
    check = HYB.parsed("configs/submission_cpu_check.yaml")
    assert (
        cpu._replace(
            kaggle_notebook=gpu.kaggle_notebook,
            accelerator=gpu.accelerator,
            workers=gpu.workers,
            solver_end_s=gpu.solver_end_s,
            solver_kill_s=gpu.solver_kill_s,
        )
        == gpu
    )
    assert (cpu.kaggle_notebook, cpu.accelerator, cpu.workers) == (
        check.kaggle_notebook,
        check.accelerator,
        check.workers,
    )
    assert (cpu.solver_end_s, cpu.solver_kill_s) == (check.solver_end_s, check.solver_kill_s)
    assert "no accelerator" in nbb.submission_blocker(cpu)
    assert cpu.kaggle_notebook not in ops.briefed_notebooks()


# The reference's LoRA settings under the current code.

# P1.1-01's values (commit d94304d, arc26/nvarc/arc_solver.py lines 233 to 271 and 300): the
# literals of the reference's cell 4 that the moved worker ran.
P1_1_PEFT = {
    "r": 256,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ],
    "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none", "use_gradient_checkpointing": False,
    "random_state": 42, "use_rslora": True, "loftq_config": None,
}  # fmt: skip
P1_1_TRAIN = {
    "per_device_eval_batch_size": 1, "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1, "num_train_epochs": 1, "warmup_steps": 0,
    "warmup_ratio": 0.1, "max_grad_norm": 1.0, "learning_rate": 5e-5, "optim": "adamw_torch",
    "weight_decay": 0.0, "lr_scheduler_type": "cosine", "seed": 42, "report_to": "none",
    "save_strategy": "no", "eval_strategy": "no", "logging_strategy": "no", "fp16": False,
    "bf16": True, "fsdp": "", "ddp_find_unused_parameters": False, "dataloader_num_workers": 0,
    "gradient_checkpointing": False,
}  # fmt: skip


def test_the_split_solvers_settings_with_the_defaults_are_p1_1_01s():
    values = assigned((REPO / "arc26" / "solver.py").read_text(encoding="utf-8"), "load")
    assert values["peft_params"] == P1_1_PEFT
    assert values["train_args"] == P1_1_TRAIN
    assert values["max_seq_length"] == 8192
    assert math.isclose(values["max_score"], -math.log(0.2))
    reference = assigned((REPO / "arc26" / "nvarc" / "arc_solver.py").read_text("utf-8"), "worker")
    for key in ("peft_params", "train_args", "max_seq_length", "max_score"):
        assert values[key] == reference[key], key


def test_the_views_and_the_decode_caps_are_p1_1_01s():
    ttt = (REPO / "arc26" / "ttt.py").read_text(encoding="utf-8")
    decode = (REPO / "arc26" / "decode.py").read_text(encoding="utf-8")
    assert "train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)" in ttt
    assert "puzzle_ds_multi.augment(n=2, seed=2 if view_seed is None else view_seed)" in decode
    assert "seed=hash(bk) % 1024**2" in decode
    assert "while time.time() - start_time < 540 and time.time() < end_time:" in decode
    assert "if spend_time > 1200 or time.time() > end_time:" in decode


# The builds.


@pytest.mark.parametrize("name", [GPU, CPU], ids=["gpu", "cpu"])
def test_two_builds_are_byte_identical_and_the_cells_are_p1_1_01s_but_two(name):
    first, files = cells(name)
    assert files == HYB.SCOPE_TESTS.golden_files(bs, name)
    for other in (P1_1, P2_0):
        before, _ = cells(other)
        assert set(first) == set(before)
        assert {cell for cell in first if first[cell] != before[cell]} == {"config-note", "config"}
    assert "**Not for submission.**" in first["config-note"]
    assert "    scope = {}\n" in first["run"] and "SCOPE" not in first["config"]
    assert "AGENT" not in first["config"]
    assert "| fit to submit | no, not for submission:" in files[bs.MANIFEST_FILE].decode("utf-8")


def test_the_config_cell_differs_from_p1_1_01s_and_p2_0_01s_only_in_the_values_named():
    new = config_values(cells(GPU)[0]["config"])
    p11, p20 = (config_values(cells(name)[0]["config"]) for name in (P1_1, P2_0))
    assert new["ARMS"] is None and new["SUBSET"] == tuple(sorted(P1_1_SECONDS))
    assert {k for k in new if new[k] != p11[k]} == {
        "SUBSET",
        "SOLVER",
        "SOLVER_END_S",
        "SOLVER_KILL_S",
        "BUILD",
    }
    assert {k for k in new if new[k] != p20[k]} == {
        "SUBSET",
        "SOLVER_END_S",
        "SOLVER_KILL_S",
        "BUILD",
    }
    assert (p11["SOLVER"], p20["SOLVER"], new["SOLVER"]) == ("nvarc", "split", "split")


def test_the_gpu_builds_metadata_attaches_p2_0_01s_inputs_but_the_p6_9_code_dataset():
    meta = json.loads(cells(GPU)[1][bs.METADATA_FILE])
    p20 = json.loads(HYB.SCOPE_TESTS.golden_files(bs, P2_0)[bs.METADATA_FILE])
    assert (meta["id"], meta["title"], meta["machine_shape"]) == (
        NOTEBOOK,
        "arc26 p4 7 r256",
        "NvidiaL4",
    )
    assert meta["dataset_sources"] == ["alyahya737/arc26-code-p69"]
    assert meta["is_private"] and not meta["enable_internet"]
    for key in ("docker_image", "competition_sources", "model_sources", "kernel_sources"):
        assert meta[key] == p20[key], key
