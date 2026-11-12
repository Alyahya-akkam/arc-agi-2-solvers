"""Repository hygiene checks that enforce the BRIEF A0 hard rules on every CI run.

These guard rules a reviewer can miss in a large diff: no em dashes, no data, weights, or secrets
in git, and no Kaggle access from code outside the one gateway (ADR-008).
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)  # built at runtime so this file passes its own check
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".sh", ".ipynb", ".json",
}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt"}
FORBIDDEN_NAMES = {"kaggle.json", ".env"}
FORBIDDEN_TOP_DIRS = {"data", "checkpoints", "runs", "dist"}
CODE_DIRS = ("arc26/", "scripts/", "notebooks/")
# ADR-008: every Kaggle action goes through this gateway, which enforces the approval rules.
# ADR-022: the Modal job's fetch and publish stages read and write Kaggle from inside Modal; only
# scripts/modal_ops.py launches them, after the same approval.
KAGGLE_GATEWAYS = {"scripts/kaggle_ops.py", "scripts/modal_job.py"}
# The Kaggle API clients (kaggle, kagglehub), the CLI in a shell line, or the CLI via subprocess
# argument lists.
KAGGLE_ACCESS = re.compile(
    r"^\s*(import kaggle\b|from kaggle\b|import kagglehub\b|from kagglehub\b)"
    r"|\bkaggle (competitions|kernels|datasets|models) "
    r"|[\"']kaggle[\"']\s*,",
    re.MULTILINE,
)
# ADR-022: every Modal action goes through scripts/modal_ops.py; scripts/modal_app.py defines the
# app it launches and is imported only by it (and by Modal's containers).
MODAL_GATEWAYS = {"scripts/modal_ops.py", "scripts/modal_app.py"}
# The Modal client imported, the CLI in a shell line, or the CLI through python -m.
MODAL_ACCESS = re.compile(
    r"^\s*(import modal\b|from modal\b)"
    r"|import_module\([\"']modal[\"']\)"
    r"|\bmodal (run|deploy|serve|shell|volume|app|secret|token|container) "
    r"|[\"']-m[\"'],\s*[\"']modal[\"']",
    re.MULTILINE,
)


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.stdout.split("\0") if p]


def test_no_em_dashes_in_tracked_text() -> None:
    offenders = [
        p
        for p in tracked_files()
        if Path(p).suffix in TEXT_SUFFIXES
        and EM_DASH in (REPO / p).read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders, f"em dashes found in: {offenders}"


def test_no_data_weights_or_secrets_tracked() -> None:
    offenders = [
        p
        for p in tracked_files()
        if Path(p).suffix in FORBIDDEN_SUFFIXES
        or Path(p).name in FORBIDDEN_NAMES
        or p.split("/", 1)[0] in FORBIDDEN_TOP_DIRS
    ]
    assert not offenders, f"forbidden files tracked: {offenders}"


def code_using(pattern: re.Pattern[str], allowed: set[str]) -> list[str]:
    return [
        p
        for p in tracked_files()
        if p.startswith(CODE_DIRS)
        and p not in allowed
        and Path(p).suffix in {".py", ".sh", ".ipynb"}
        and pattern.search((REPO / p).read_text(encoding="utf-8", errors="ignore"))
    ]


def test_no_kaggle_access_in_code() -> None:
    offenders = code_using(KAGGLE_ACCESS, KAGGLE_GATEWAYS)
    assert not offenders, f"Kaggle API or CLI usage found in: {offenders}"


def test_no_modal_access_in_code() -> None:
    offenders = code_using(MODAL_ACCESS, MODAL_GATEWAYS)
    assert not offenders, f"Modal client or CLI usage found in: {offenders}"


@pytest.mark.parametrize(
    "line",
    [
        "import kagglehub",
        "from kagglehub import dataset_upload",
        "import modal",
        "from modal import App",
        'importlib.import_module("modal")',
        "modal volume put arc26-modal x /",
        '[sys.executable, "-m", "modal", "app", "stop"]',
    ],
)
def test_access_patterns_catch_the_clients(line: str) -> None:
    assert KAGGLE_ACCESS.search(line) or MODAL_ACCESS.search(line)


def test_access_patterns_leave_the_job_module_name_alone() -> None:
    assert not MODAL_ACCESS.search("import modal_job  # noqa: E402")
