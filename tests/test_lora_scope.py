"""The LoRA scope as run settings (task P1.3, deliverable 1).

Two replacements in configs/extraction.yaml turn the reference's rank and target modules into
``cfg["lora_r"]`` and ``cfg["lora_targets"]``; ``arc26.runner.RunConfig`` defaults them to the
reference's values. With those defaults the solver must be unchanged. ``arc_solver.py`` imports
torch and unsloth, so its ``worker`` is never imported here: the ``peft_params`` assignment is
taken from the AST of the generated module and of the reference's cell 4 and evaluated alone,
with the cfg the solver receives (a ``RunConfig`` through the ``--config`` JSON, parsed back as
``starter.py`` parses it).
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from arc26 import runner
from arc26.runner import RunConfig, solver_command

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "reference" / "reference_notebook.ipynb"
SPEC = REPO / "configs" / "extraction.yaml"
SOLVER = REPO / "arc26" / "nvarc" / "arc_solver.py"
NINE = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
]  # fmt: skip
LORA_OLDS = (
    "r=256,",
    'target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", '
    '"down_proj", "embed_tokens", "lm_head"],',
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_cell():
    """Cell 4 as its %%writefile magic writes it: the cell after its first line."""
    source = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][4]["source"]
    text = source if isinstance(source, str) else "".join(source)
    first, _, code = text.partition("\n")
    assert first == "%%writefile arc_solver.py"
    return code


def peft_params(source, cfg=None):
    """Evaluate the ``peft_params = dict(...)`` statement of ``worker`` in ``source``."""
    functions = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)]
    (worker,) = [node for node in functions if node.name == "worker"]
    (assign,) = [
        node
        for node in worker.body
        if isinstance(node, ast.Assign)
        and [getattr(target, "id", None) for target in node.targets] == ["peft_params"]
    ]
    expression = ast.Expression(assign.value)
    return eval(compile(expression, "<peft_params>", "eval"), {"cfg": cfg})


def solver_cfg(**fields):
    """The cfg dict the moved worker receives for these RunConfig fields."""
    cfg = RunConfig(data_dir="/data", model_dir="/model", **fields)
    return json.loads(solver_command(cfg, 0.0, "python")[6])


def test_with_the_defaults_peft_params_equal_the_reference():
    reference = peft_params(reference_cell())
    assert reference["r"] == 256 and reference["target_modules"] == NINE
    assert reference["lora_alpha"] == 32 and reference["use_rslora"] is True
    moved = peft_params(SOLVER.read_text(encoding="utf-8"), solver_cfg())
    assert moved == reference
    assert type(moved["target_modules"]) is list


@pytest.mark.parametrize(
    ("rank", "targets"),
    [
        (16, ("q_proj", "k_proj", "v_proj", "o_proj")),
        (32, ("q_proj", "k_proj", "v_proj", "o_proj")),
        (16, tuple(NINE[:7])),
        (64, tuple(NINE[:7])),
        (256, tuple(NINE[:7])),
        (1, ("q_proj",)),
    ],
)
def test_an_arm_changes_only_the_rank_and_the_target_modules(rank, targets):
    reference = peft_params(reference_cell())
    moved = peft_params(
        SOLVER.read_text(encoding="utf-8"), solver_cfg(lora_r=rank, lora_targets=targets)
    )
    assert moved == {**reference, "r": rank, "target_modules": list(targets)}
    assert type(moved["target_modules"]) is list


def test_the_spec_turns_exactly_the_two_lora_literals_into_run_settings():
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    found = {
        entry["module"]: [
            (item["old"], item["new"], item["count"])
            for item in entry["replacements"]
            if "lora" in item["new"]
        ]
        for entry in spec["modules"]
    }
    assert found.pop("arc_solver.py") == [
        (LORA_OLDS[0], 'r=cfg["lora_r"],', 1),
        (LORA_OLDS[1], 'target_modules=list(cfg["lora_targets"]),', 1),
    ]
    assert all(items == [] for items in found.values())
    for old in LORA_OLDS:
        assert reference_cell().count(old) == 1
        assert old not in SOLVER.read_text(encoding="utf-8")


def test_the_reference_scope_constants_agree_with_the_notebook():
    builder = load("build_kaggle_notebook", REPO / "scripts" / "build_kaggle_notebook.py")
    reference = peft_params(reference_cell())
    assert reference["r"] == runner.LORA_R == builder.REFERENCE_LORA_R
    assert tuple(reference["target_modules"]) == runner.LORA_TARGETS
    assert runner.LORA_TARGETS == builder.REFERENCE_LORA_TARGETS
    assert runner.LORA_PROJECTIONS == builder.LORA_PROJECTIONS == tuple(NINE[:7])
