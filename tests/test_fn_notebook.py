"""Runs version 11's notebook (scripts/build_fn_notebooks.py, ``notebook``) on CPU.

The three code cells execute in order on a fake ``/kaggle`` tree under ``tmp_path`` (``FN_ROOT``):
a code dataset holding a wheel-shaped zip with ``arc26.submit``, ``fn_probe.py``, and the fake
SGLang server, checked by sha256 as on Kaggle; the competition's challenges; a model folder. The
config is the smoke's (no install, the fake server, the real tool loop), so the cells run the
real solver end to end in seconds. No network beyond localhost.
"""

import hashlib
import importlib.util
import json
import shutil
import socket
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

# The real package first, so cell 1's "from arc26.submit import ..." (with the code dataset's
# wheel first on sys.path) finds it cached and no later test imports arc26 from the fake wheel.
import arc26.submit  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"_tn_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load("build_fn_notebooks")
probe = load("fn_probe")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(tmp_path, mode):
    """Lay out the fake inputs and return the notebook's config."""
    config = probe.apply_smoke(yaml.safe_load((REPO / "configs" / "P5.2-01.yaml").read_text()))
    config["mode"] = mode
    config["engine"]["port"] = free_port()
    config["inputs"]["competition"]["path"] = (
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
    config["inputs"]["code"] = {"path": "/kaggle/input/datasets/alyahya737/arc26-code-fn",
                                "marker": "fn_code.json", "hint": "arc26-code-fn",
                                "required": True}
    root = tmp_path / "root"
    probe.build_smoke_tree(root, config)
    code = root / "kaggle/input/datasets/alyahya737/arc26-code-fn"
    code.mkdir(parents=True)
    wheel = code / "arc26-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ("arc26/__init__.py", "arc26/submit.py"):
            archive.write(REPO / name, name)
    for name in ("fn_probe.py", "fn_fake_sglang.py"):
        shutil.copyfile(SCRIPTS / name, code / name)
    files = {p.name: sha(p) for p in sorted(code.iterdir())}
    (code / "fn_code.json").write_text(json.dumps({"files": files}))
    config["build"] = {"commit": "abc123", "config": "configs/P5.2-01.yaml",
                       "wheel": wheel.name, "files": files}
    return root, config


def run_cells(root, config, monkeypatch):
    monkeypatch.setenv("FN_ROOT", str(root))
    # Cell 1 puts the code dataset's wheel first on sys.path; later tests must not see it.
    monkeypatch.setattr(sys, "path", list(sys.path))
    book = json.loads(builder.notebook(config))
    assert "kaggle" not in book["metadata"]
    namespace = {}
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "cell", "exec"), namespace)
    return namespace, root / "kaggle/working"


@pytest.mark.parametrize("mode", ["development", "competition"])
def test_the_notebook_writes_a_valid_submission(tmp_path, monkeypatch, mode):
    root, config = tree(tmp_path, mode)
    namespace, working = run_cells(root, config, monkeypatch)
    assert namespace["CODE_OK"] is True
    assert namespace["SOLVE"] == {"mode": mode, "end_s": 80.0, "exit": 0}
    final = json.loads((working / "run_logs/fn/final_check.json").read_text())
    assert final["valid"] is True and final["tasks"] == 6
    setup = json.loads((working / "run_logs/fn/notebook_setup.json").read_text())
    assert setup["placeholder"]["valid"] is True and setup["placeholder"]["zero_attempts"] == 12
    submission = json.loads((working / "submission.json").read_text())
    answered = sum(entry[0]["attempt_1"] != [[0]] for entry in submission.values())
    attempts = (working / "run_logs/fn/attempts.jsonl").read_text().splitlines()
    if mode == "competition":
        assert answered == 6 and len(attempts) == 6
    else:
        # Development mode answers the evaluation tasks in the logs only; the file keeps its
        # placeholders.
        assert answered == 0 and len(attempts) == 12


def test_a_failed_code_check_keeps_the_placeholders(tmp_path, monkeypatch):
    root, config = tree(tmp_path, "competition")
    config["build"]["files"]["fn_probe.py"] = "0" * 64
    namespace, working = run_cells(root, config, monkeypatch)
    assert namespace["CODE_OK"] is False
    assert namespace["SOLVE"]["exit"].startswith("code dataset check failed")
    final = json.loads((working / "run_logs/fn/final_check.json").read_text())
    assert final["valid"] is True and final["zero_attempts"] == 12
