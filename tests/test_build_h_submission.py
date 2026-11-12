"""Guard H attachment, preflight failure and preservation of the v1 notebook path."""

import contextlib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h_wrapper", ROOT / "scripts/build_h_submission.py")
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)
SOURCE = {"dataset": "alyahya737/arc26-p3-9-01", "version": 2, "filename": "best.pt",
          "sha256": hashlib.sha256(b"checkpoint").hexdigest()}


def notebook():
    return {"cells": [
        {"cell_type": "markdown", "source": ["Original production note\n"]},
        {"cell_type": "code", "source": ["INSTALLED = False\nINSTALLED = True\n"]},
        {"cell_type": "code", "source": ["def solver_config():\n    return RunConfig(\n"
                                         "        solver='split',\n    )\n"]},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


@pytest.mark.parametrize("field,value", [
    ("version", True), ("version", 0), ("version", "2"), ("dataset", "../owner/slug"),
    ("filename", "../best.pt"), ("filename", "/best.pt"), ("filename", "best.ckpt"),
    ("sha256", "a" * 63), ("sha256", "A" * 64), ("extra", 1),
])
def test_source_rejects_unpinned_or_unsafe_identity(field, value):
    with pytest.raises(BUILDER.BuildError):
        BUILDER.parse_source(yaml.safe_dump({**SOURCE, field: value}))


def test_notebook_injects_once_after_install_and_preserves_v1_arguments():
    original = notebook()
    wrapped = BUILDER.patch_notebook(original, SOURCE)
    codes = ["".join(c["source"]) for c in wrapped["cells"] if c["cell_type"] == "code"]
    assert "H_CHECK_PASSED" in codes[1]
    assert "INSTALLED = True" in codes[0]
    assert "hyper_checkpoint=H_CHECKPOINT" in codes[2]
    assert "hyper_sha256=H_SOURCE" in codes[2]
    assert 'hyper_log_dir=str(RUN_LOGS / "h-start")' in codes[2]
    assert "solver='split'" in codes[2]
    assert "lora_r=" not in codes[2] and "SCOPE" not in codes[2]
    assert "hyper_checkpoint" not in "".join(original["cells"][-1]["source"])
    assert "evaluation_solutions" not in "".join(codes)
    assert "production H initialization" in "".join(wrapped["cells"][0]["source"])


def test_changed_template_is_refused_instead_of_silent_plain_v1():
    wrong = notebook()
    wrong["cells"][-1]["source"] = ["def solver_config():\n    return None\n"]
    with pytest.raises(BUILDER.BuildError, match="exactly one RunConfig"):
        BUILDER.patch_notebook(wrong, SOURCE)
    duplicate = notebook()
    duplicate["cells"][-1]["source"] += ["\n" + BUILDER.RUN_ANCHOR + "    )\n"]
    with pytest.raises(BUILDER.BuildError, match="exactly one RunConfig anchor"):
        BUILDER.patch_notebook(duplicate, SOURCE)


def run_preflight(tmp_path, *, bad_cli=False, competition=False, duplicate=False, wrong_hash=False):
    inputs = tmp_path / "input"
    checkpoint = inputs / "datasets" / SOURCE["dataset"] / SOURCE["filename"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"wrong" if wrong_hash else b"checkpoint")
    if duplicate:
        other = inputs / SOURCE["dataset"].split("/")[1] / SOURCE["filename"]
        other.parent.mkdir(parents=True)
        other.write_bytes(b"checkpoint")
    logs = tmp_path / "logs"
    logs.mkdir()
    calls = []

    @contextlib.contextmanager
    def stage(_):
        try:
            yield {}
        except Exception:
            if not competition:
                raise

    def logged(args, timeout):
        calls.append((args, timeout))
        Path(args[args.index("--report") + 1]).write_text(json.dumps({"ok": not bad_cli}))
        return SimpleNamespace(returncode=1 if bad_cli else 0)

    env = {"INSTALLED": True, "PYTHON": "python", "INPUT": inputs, "RUN_LOGS": logs,
           "stage": stage, "run_logged": logged, "json": json,
           "file_sha256": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    exec(BUILDER.checkpoint_cell(SOURCE), env)
    return env, calls


@pytest.mark.parametrize("competition", [False, True])
def test_cpu_preflight_runs_the_real_cli_before_either_mode(tmp_path, competition):
    env, calls = run_preflight(tmp_path, competition=competition)
    assert env["H_CHECK_PASSED"] is True
    assert len(calls) == 1
    args, timeout = calls[0]
    assert args[:5] == ["python", "-m", "arc26.h_submission", "--check-only", "--checkpoint"]
    assert args[args.index("--sha256") + 1] == SOURCE["sha256"]
    assert timeout == 900


@pytest.mark.parametrize("failure", ["bad_cli", "duplicate", "wrong_hash"])
def test_competition_cannot_swallow_a_failed_h_preflight(tmp_path, failure):
    with pytest.raises(RuntimeError, match="H preflight failed"):
        run_preflight(tmp_path, competition=True, **{failure: True})


def test_wrapper_attaches_pinned_dataset_and_hashes_final_outputs(tmp_path, monkeypatch):
    out = tmp_path / "dist" / "H-submit-01"
    out.mkdir(parents=True)
    original = notebook()
    (out / "kaggle_submission.ipynb").write_text(json.dumps(original))
    meta = {"id": "alyahya737/arc26-submission", "code_file": "kaggle_submission.ipynb",
            "dataset_sources": ["alyahya737/arc26-code-h-submit"], "is_private": True}
    (out / "kernel-metadata.json").write_text(json.dumps(meta))
    (out / "wheel.whl").write_bytes(b"wheel")
    (out / "MANIFEST.md").write_text("old stale notebook hash")
    monkeypatch.setattr(BUILDER.BASE, "require_clean", lambda _: None)
    monkeypatch.setattr(BUILDER.BASE, "head_commit", lambda _: ("a" * 40, 0))
    monkeypatch.setattr(BUILDER.BASE, "_git", lambda *a: yaml.safe_dump(SOURCE))
    monkeypatch.setattr(BUILDER.BASE, "build", lambda *a: out)
    result = BUILDER.build("H-submit-01", tmp_path / "config.yaml", tmp_path)
    metadata = json.loads((result / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"][-1] == "alyahya737/arc26-p3-9-01/2"
    manifest = (result / "MANIFEST.md").read_text()
    assert "old stale notebook hash" not in manifest
    for path in result.rglob("*"):
        if path.is_file() and path.name != "MANIFEST.md":
            assert hashlib.sha256(path.read_bytes()).hexdigest() in manifest
    assert SOURCE["sha256"] in manifest
