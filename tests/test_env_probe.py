"""Checks the K0 environment probe and its push-folder builder. No network, no Kaggle.

The probe runs on a synthetic ``/kaggle`` tree under ``tmp_path`` through ``--root``, with
``pip list`` replaced and every ``open`` recorded, so a read of the solutions file cannot go
unnoticed. The builder's folders go through the gateway's own CPU-push checks, with
``scripts/kaggle_ops.py`` loaded as ``tests/test_kaggle_ops.py`` loads it, ``KAGGLE_OWNER`` set,
and the kaggle CLI replaced by a recorder.
"""

import ast
import builtins
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load("env_probe")
builder = load("build_env_probe")
ops = load("kaggle_ops")
REAL_PIP_LIST = probe.pip_list

OWNER = "tester"
SECRET = "kst-0d9f8e7c6b5a4938-never-in-an-output"
# One secret-looking variable per marker, and a name in lower case.
MARKED = {
    "KAGGLE_DATA_PROXY_TOKEN": "tok-7f3a9c1e5b2d-proxy",
    "KAGGLE_CLIENT_SECRET": "sec-4e8b2a6c0d1f-client",
    "KAGGLE_KEY": "key-9a1b3c5d7e2f-api",
    "KAGGLE_PASSWORD": "pwd-2c4e6a8b0d3f-pass",
    "KAGGLE_AUTH_HEADER": "Bearer-5d7f9b1a3c4e",
    "KAGGLE_CREDENTIALS_FILE": "/secret/cred-8b0d2f4a6c.json",
    "kaggle_lower_secret": "low-3e5a7c9b1d0f-case",
}
SHOWN = {
    "KAGGLE_KERNEL_RUN_TYPE": "Batch",
    "KAGGLE_IS_COMPETITION_RERUN": None,
    "KAGGLE_DOCKER_IMAGE": None,
    "KAGGLE_URL_BASE": "https://www.kaggle.com",
}
# Verbatim from the task brief.
BRIEF_DISTRIBUTIONS = (
    "torch", "transformers", "peft", "accelerate", "trl", "datasets", "unsloth", "unsloth_zoo",
    "xformers", "flash_attn", "flash-attn", "triton", "bitsandbytes", "vllm", "numpy",
)
BRIEF_MARKERS = {"TOKEN", "SECRET", "KEY", "PASS", "AUTH", "CREDENTIAL"}
COMPETITION_DIR = "kaggle/input/competitions/arc-prize-2026-arc-agi-2"
MODEL_DIR = "kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
OS_RELEASE = 'PRETTY_NAME="Ubuntu 22.04.4 LTS"\nNAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n'
MEMINFO = (
    "MemTotal:       32874380 kB\nMemFree:        30000000 kB\n"
    "MemAvailable:   31000000 kB\nSwapTotal:             0 kB\n"
)
PIP_LIST = [{"name": "numpy", "version": "2.2.6"}, {"name": "torch", "version": "2.8.0+cu128"}]


def grid(rows, cols, seed):
    """A grid whose rows are easy to spot if an output ever quoted one."""
    return [[(seed + 3 * r + 7 * c) % 10 for c in range(cols)] for r in range(rows)]


TASKS = {
    "0a1b2c3d": {
        "train": [
            {"input": grid(3, 3, 1), "output": grid(3, 3, 2)},
            {"input": grid(2, 4, 3), "output": grid(4, 2, 4)},
        ],
        "test": [{"input": grid(5, 5, 5)}],
    },
    "1b2c3d4e": {
        "train": [
            {"input": grid(1, 1, 6), "output": grid(1, 1, 7)},
            {"input": grid(2, 2, 8), "output": grid(2, 2, 9)},
            {"input": grid(3, 1, 0), "output": grid(1, 3, 1)},
        ],
        "test": [{"input": grid(2, 3, 2)}, {"input": grid(30, 30, 3)}],
    },
    "2c3d4e5f": {
        "train": [{"input": grid(4, 6, 4), "output": grid(6, 4, 5)}],
        # A test output is not among the counted grids, so this 12x12 must not show up.
        "test": [{"input": grid(7, 2, 6), "output": grid(12, 12, 7)}],
    },
}
EXPECTED_STATS = {
    "0a1b2c3d": {"train_pairs": 2, "test_inputs": 1, "max_side": 5, "cells": 9 + 9 + 8 + 8 + 25},
    "1b2c3d4e": {
        "train_pairs": 3,
        "test_inputs": 2,
        "max_side": 30,
        "cells": 1 + 1 + 4 + 4 + 3 + 3 + 6 + 900,
    },
    "2c3d4e5f": {"train_pairs": 1, "test_inputs": 1, "max_side": 7, "cells": 24 + 24 + 14},
}
SOLUTIONS = {task_id: [grid(6, 6, 9)] * len(task["test"]) for task_id, task in TASKS.items()}


def all_grids():
    for task in TASKS.values():
        for pair in task["train"] + task["test"]:
            yield from (pair[key] for key in ("input", "output") if key in pair)
    for outputs in SOLUTIONS.values():
        yield from outputs


def grid_leaks(output):
    """Grids, and rows of four or more cells, found in ``output`` whatever its whitespace.

    Compared without whitespace, so an indented JSON dump, which spreads a list over many lines,
    is caught as well as a one-line repr. No list of four or more integers is legitimate in a
    report (``version_info`` has three).
    """
    compact = "".join(output.split())
    markers = set()
    for item in all_grids():
        markers.add(json.dumps(item, separators=(",", ":")))
        markers.update(json.dumps(row, separators=(",", ":")) for row in item if len(row) >= 4)
    return sorted(marker for marker in markers if marker in compact)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        text = content if isinstance(content, str) else json.dumps(content)
        path.write_text(text, encoding="utf-8")
    return path


def is_solutions(path):
    """The probe's rule: the file name, not a folder (pytest's tmp folders repeat test names)."""
    return "solution" in Path(path).name.lower()


def record_opens(monkeypatch, opened):
    """Record every file opened through ``open``, ``io.open``, or ``os.open``; refuse solutions."""

    def recorder(real):
        def wrapper(file, *args, **kwargs):
            pathlike = isinstance(file, (str, bytes, os.PathLike))
            name = str(os.fspath(file)) if pathlike else str(file)
            opened.append(name)
            if is_solutions(name):
                raise AssertionError(f"opened {name}")
            return real(file, *args, **kwargs)

        return wrapper

    for module, attr in ((builtins, "open"), (io, "open"), (os, "open")):
        monkeypatch.setattr(module, attr, recorder(getattr(module, attr)))


def count_within(base, depth):
    """Entries under ``base`` down to ``depth``, counted independently of the probe."""
    total = 0
    for folder, dirnames, filenames in os.walk(base):
        total += len(dirnames) + len(filenames)
        if len(Path(folder).relative_to(base).parts) + 1 >= depth:
            dirnames[:] = []
    return total


class Session:
    """A synthetic Kaggle session: data, model, install notebook, working dir, system files."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "root"
        competition = self.root / COMPETITION_DIR
        self.challenges = write(competition / "arc-agi_evaluation_challenges.json", TASKS)
        self.solutions = write(competition / "arc-agi_evaluation_solutions.json", SOLUTIONS)
        subset = {"0a1b2c3d": TASKS["0a1b2c3d"]}
        self.test_challenges = write(competition / "arc-agi_test_challenges.json", subset)
        write(competition / "sample_submission.json", {})
        self.config = write(self.root / MODEL_DIR / "config.json", {"model_type": "qwen3"})
        write(self.root / MODEL_DIR / "model.safetensors", b"\0" * 64)
        # A config.json outside the model's folder, which must not count as the model's.
        notebook = self.root / "kaggle/input/notebooks/sorokin/pip-install-unsloth-flash-patch"
        write(notebook / "config.json", {})
        write(self.root / "kaggle/usr/lib/pip_install_unsloth_flash_patch/patch.py", "")
        (self.root / "kaggle/working").mkdir(parents=True)
        (self.root / "tmp").mkdir()
        write(self.root / "etc/os-release", OS_RELEASE)
        write(self.root / "etc/git_commit", "0a1b2c3d\n")
        write(self.root / "etc/build_date", "20260915-120000\nextra line\n")
        write(self.root / "proc/meminfo", MEMINFO)
        self.opened = []

    @property
    def logs(self):
        return self.root / "kaggle/working/run_logs"

    def run(self, *extra):
        """Run the probe; return the report, the JSON text, and the Markdown text."""
        assert probe.main(["--root", str(self.root), *extra]) == 0
        text = (self.logs / "env_probe.json").read_text(encoding="utf-8")
        return json.loads(text), text, (self.logs / "env_probe.md").read_text(encoding="utf-8")


@pytest.fixture
def clean_env(monkeypatch):
    """Only the test's KAGGLE_* and secret-named variables, and no subprocess for pip."""
    for name in list(os.environ):
        if name.upper().startswith("KAGGLE_") or probe.looks_secret(name):
            monkeypatch.delenv(name)
    monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", SECRET)
    for name, value in {**MARKED, **SHOWN}.items():
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(probe, "pip_list", lambda environ: PIP_LIST)


@pytest.fixture
def session(tmp_path, monkeypatch, clean_env):
    tree = Session(tmp_path)
    record_opens(monkeypatch, tree.opened)  # after the tree is written, before the probe runs
    return tree


# The probe.


def test_outputs_land_where_the_gateway_pull_keeps_them(session):
    session.run()
    for name in ("env_probe.json", "env_probe.md"):
        assert (session.logs / name).is_file()
        assert ops.pull_allowed(f"run_logs/{name}")


def test_session_fields(session):
    report, _, _ = session.run()
    assert report["errors"] == {}
    assert report["schema"] == 1
    assert report["python"]["version"] == sys.version
    assert report["python"]["version_info"] == list(sys.version_info[:3])
    assert report["python"]["executable"] == sys.executable
    assert report["os_release"] == {
        "PRETTY_NAME": "Ubuntu 22.04.4 LTS",
        "NAME": "Ubuntu",
        "VERSION_ID": "22.04",
        "ID": "ubuntu",
    }
    assert report["image"] == {"/etc/git_commit": "0a1b2c3d", "/etc/build_date": "20260915-120000"}
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    assert report["cpu"] == {"cpu_count": os.cpu_count(), "affinity": usable}
    assert report["memory"] == {
        "MemTotal": 32874380 * 1024,
        "MemAvailable": 31000000 * 1024,
        "SwapTotal": 0,
    }
    assert set(report["disk"]) == {"/kaggle/working", "/tmp"}
    for usage in report["disk"].values():
        assert usage["total"] >= usage["free"] >= 0 and usage["used"] >= 0
    assert report["sys_path"] == sys.path
    assert list(report["versions"]) == list(BRIEF_DISTRIBUTIONS)
    assert report["pip_list"] == PIP_LIST


def test_kaggle_variables_are_listed_by_name_with_only_four_values(session):
    report, _, md = session.run()
    env = report["kaggle_env"]
    assert env["names"] == sorted(["KAGGLE_USER_SECRETS_TOKEN", *MARKED, "KAGGLE_KERNEL_RUN_TYPE",
                                   "KAGGLE_URL_BASE"])
    assert env["values"] == SHOWN
    assert [name for name in env["names"] if name not in md] == []
    assert "| KAGGLE_KERNEL_RUN_TYPE | `Batch` |" in md
    assert "| KAGGLE_IS_COMPETITION_RERUN | (unset) |" in md
    assert "| KAGGLE_USER_SECRETS_TOKEN | (value not recorded) |" in md


def test_no_secret_value_or_grid_in_either_output(session, capsys):
    _, text, md = session.run()
    printed = capsys.readouterr().out
    assert "wrote /kaggle/working/run_logs/env_probe.json" in printed
    for output in (text, md, printed):
        assert [value for value in (SECRET, *MARKED.values()) if value in output] == []
        assert grid_leaks(output) == []


def test_a_secret_looking_name_is_withheld_even_if_listed_as_shown(session, monkeypatch):
    monkeypatch.setattr(probe, "SHOWN_VALUES", (*probe.SHOWN_VALUES, "KAGGLE_USER_SECRETS_TOKEN"))
    report, text, md = session.run()
    assert "KAGGLE_USER_SECRETS_TOKEN" not in report["kaggle_env"]["values"]
    assert SECRET not in text and SECRET not in md


def test_kaggle_env_alone_never_returns_a_secret_value(monkeypatch):
    listed = (*probe.SHOWN_VALUES, *MARKED, "KAGGLE_USER_SECRETS_TOKEN")
    monkeypatch.setattr(probe, "SHOWN_VALUES", listed)
    environ = {"KAGGLE_USER_SECRETS_TOKEN": SECRET, **MARKED, "PATH": "/usr/bin", "HOME": "/root"}
    env = probe.kaggle_env(environ)
    assert env["names"] == sorted(["KAGGLE_USER_SECRETS_TOKEN", *MARKED])
    assert [v for v in (SECRET, *MARKED.values()) if v in json.dumps(env)] == []


def test_secret_values_are_scrubbed_from_every_field(session, monkeypatch):
    monkeypatch.syspath_prepend(f"/opt/{SECRET}/lib")  # a leak path the allowlist cannot see
    report, text, md = session.run()
    assert SECRET not in text and SECRET not in md
    assert f"/opt/{probe.REDACTED}/lib" in report["sys_path"]


def test_a_secret_named_variable_outside_kaggle_is_scrubbed_and_kept_from_pip(
    session, monkeypatch, capsys
):
    token = "hf_9c8b7a6d5e4f3a2b1c0d-not-for-output"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("PROBE_PLAIN_SETTING", "kept-for-pip")
    monkeypatch.syspath_prepend(f"/opt/{token}/site-packages")
    calls = []

    def run(command, **kwargs):  # pip fails and echoes the token on stderr
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=f"ERROR: token {token}")

    monkeypatch.setattr(probe, "pip_list", REAL_PIP_LIST)
    monkeypatch.setattr(probe.subprocess, "run", run)
    report, text, md = session.run()
    printed = capsys.readouterr().out
    assert [output for output in (text, md, printed) if token in output] == []
    assert report["errors"]["pip_list"].endswith(f"ERROR: token {probe.REDACTED}")
    assert f"ERROR: token {probe.REDACTED}" in md and f"ERROR: token {probe.REDACTED}" in printed
    assert f"/opt/{probe.REDACTED}/site-packages" in report["sys_path"]
    (kwargs,) = calls
    assert [name for name in kwargs["env"] if probe.looks_secret(name)] == []
    assert kwargs["env"]["PROBE_PLAIN_SETTING"] == "kept-for-pip"


def test_long_values_of_every_secret_named_variable_are_scrubbed():
    environ = {"KAGGLE_A_TOKEN": "1", "KAGGLE_B_KEY": "b" * 8, "KAGGLE_C_KEY": "c" * 12,
               "OTHER_TOKEN": "d" * 20, "KAGGLE_URL_BASE": "e" * 20, "HOME": "f" * 20}
    assert probe.secret_values(environ) == ["d" * 20, "c" * 12, "b" * 8]
    scrubbed = probe.scrub({"k-" + "c" * 12: ["x-" + "c" * 12, 5, None]}, ["c" * 12])
    assert scrubbed == {"k-<redacted>": ["x-<redacted>", 5, None]}


def test_resolved_paths(session):
    report, _, md = session.run()
    competition = f"/{COMPETITION_DIR}"
    config = {"path": f"/{MODEL_DIR}/config.json", "size": session.config.stat().st_size}
    assert report["resolved"] == {
        "arc_agi_files": [
            {"path": f"{competition}/{path.name}", "size": path.stat().st_size}
            for path in (session.challenges, session.solutions, session.test_challenges)
        ],
        "model_configs": [config],
        "skipped_links": [],
        "errors": [],
    }
    assert f"`/{MODEL_DIR}/config.json`" in md


def test_evaluation_statistics(session):
    report, _, md = session.run()
    stats = report["evaluation_stats"]
    assert stats["source"] == f"/{COMPETITION_DIR}/arc-agi_evaluation_challenges.json"
    data = session.challenges.read_bytes()
    assert (stats["bytes"], stats["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())
    assert stats["per_task"] == EXPECTED_STATS
    assert stats["summary"] == {
        "tasks": 3,
        "malformed": 0,
        "train_pairs": 6,
        "test_inputs": 4,
        "max_side": 30,
        "cells": sum(task["cells"] for task in EXPECTED_STATS.values()),
    }
    assert "| 1b2c3d4e | 3 | 2 | 30 | 922 |" in md


def test_solutions_file_is_never_opened(session):
    report, _, _ = session.run()
    assert str(session.challenges) in session.opened  # the recorder sees the probe's reads
    assert [path for path in session.opened if is_solutions(path)] == []
    assert report["errors"] == {}  # no refused read either


@pytest.mark.parametrize(
    "name",
    ["arc-agi_evaluation_solutions.json", "SOLUTIONS.JSON", "arc-agi_training_solution.json"],
)
def test_read_bytes_refuses_solutions_without_opening_them(tmp_path, monkeypatch, name):
    path = write(tmp_path / name, "{}")
    opened = []
    record_opens(monkeypatch, opened)
    with pytest.raises(probe.ProbeError, match="never read"):
        probe.read_bytes(path)
    assert opened == []


def test_read_bytes_refuses_a_link_to_a_solutions_file(tmp_path, monkeypatch):
    target = write(tmp_path / "arc-agi_evaluation_solutions.json", "{}")
    link = tmp_path / "arc-agi_evaluation_challenges.json"
    link.symlink_to(target.name)
    opened = []
    record_opens(monkeypatch, opened)
    with pytest.raises(probe.ProbeError):
        probe.read_bytes(link)
    assert opened == []


def test_listing_goes_four_levels_deep(session):
    report, _, _ = session.run()
    listing = report["listings"]["/kaggle/input"]
    entries = {entry["path"]: entry for entry in listing["entries"]}
    model = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139"
    assert entries[f"{model}/transformers"] == {"path": f"{model}/transformers", "type": "dir"}
    assert f"{model}/transformers/bfloat16" not in entries  # depth 5
    challenges = f"/{COMPETITION_DIR}/arc-agi_evaluation_challenges.json"
    size = session.challenges.stat().st_size
    assert entries[challenges] == {"path": challenges, "type": "file", "size": size}
    assert (listing["depth"], listing["cap"], listing["truncated"]) == (4, 2000, False)
    within = count_within(session.root / "kaggle/input", 4)
    assert listing["total"] == len(listing["entries"]) == within
    usr_lib = report["listings"]["/kaggle/usr/lib"]
    assert usr_lib["exists"]
    assert "/kaggle/usr/lib/pip_install_unsloth_flash_patch/patch.py" in {
        entry["path"] for entry in usr_lib["entries"]
    }


def test_listing_cap_keeps_the_shallow_levels_and_is_recorded(session):
    bulk = session.root / "kaggle/input/notebooks/sorokin/big-tree"  # its files are at depth 4
    bulk.mkdir(parents=True)
    for index in range(probe.LIST_CAP + 10):
        (bulk / f"m{index:05d}.py").touch()
    report, text, md = session.run()
    listing = report["listings"]["/kaggle/input"]
    assert (listing["cap"], listing["truncated"], len(listing["entries"])) == (2000, True, 2000)
    assert listing["total"] == count_within(session.root / "kaggle/input", 4) > 2000
    kept = {entry["path"] for entry in listing["entries"]}
    assert f"/{COMPETITION_DIR}/arc-agi_evaluation_challenges.json" in kept  # depth 3 survives
    assert f"2000 of {listing['total']} entries within depth 4, truncated at the cap of 2000" in md


def test_links_into_walked_folders_are_skipped_and_links_out_are_followed(session):
    legacy = "/kaggle/input/arc-prize-2026-arc-agi-2"
    target = "competitions/arc-prize-2026-arc-agi-2"
    (session.root / legacy.lstrip("/")).symlink_to(target)
    (session.root / "kaggle/input/competitions/loop").symlink_to("..")
    extra = write(session.root / "mnt/extra/arc-agi_extra.json", {})  # outside /kaggle/input
    (session.root / "kaggle/input/extra").symlink_to(extra.parent)
    report, _, _ = session.run()
    entries = {entry["path"]: entry for entry in report["listings"]["/kaggle/input"]["entries"]}
    assert entries[legacy] == {"path": legacy, "type": "link", "target": target}
    assert entries["/kaggle/input/competitions/loop"]["target"] == ".."
    competition = f"/{COMPETITION_DIR}"
    assert [item["path"] for item in report["resolved"]["arc_agi_files"]] == [
        f"{competition}/arc-agi_evaluation_challenges.json",
        f"{competition}/arc-agi_evaluation_solutions.json",
        f"{competition}/arc-agi_test_challenges.json",
        "/kaggle/input/extra/arc-agi_extra.json",
    ]
    assert report["resolved"]["skipped_links"] == [
        {"path": legacy, "target": competition},
        {"path": "/kaggle/input/competitions/loop", "target": "/kaggle/input"},
    ]
    stats = report["evaluation_stats"]
    assert stats["source"] == f"{competition}/arc-agi_evaluation_challenges.json"
    assert stats["per_task"] == EXPECTED_STATS
    assert [path for path in session.opened if is_solutions(path)] == []


def test_no_folder_is_walked_twice_across_links(session):
    inner = write(session.root / "mnt/extra/inner/arc-agi_inner.json", {}).parent
    (session.root / "kaggle/input/aaa-inner").symlink_to(inner)  # followed first
    (session.root / "kaggle/input/extra").symlink_to(inner.parent)  # reaches inner again
    resolved = probe.find_inputs(session.root)
    found = [item["path"] for item in resolved["arc_agi_files"]]
    assert "/kaggle/input/aaa-inner/arc-agi_inner.json" in found
    assert "/kaggle/input/extra/inner/arc-agi_inner.json" not in found
    assert resolved["skipped_links"] == [
        {"path": "/kaggle/input/extra/inner", "target": "/mnt/extra/inner"}
    ]


def test_a_link_to_the_root_is_recorded_and_not_walked(session, monkeypatch):
    (session.root / "kaggle/input/rootlink").symlink_to(session.root)
    write(session.root / "tmp/arc-agi_outside.json", {})  # reachable only through the link
    scanned = []
    real_scandir = os.scandir

    def scandir(path="."):
        scanned.append(os.path.realpath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    resolved = probe.find_inputs(session.root)
    base = os.path.realpath(session.root / "kaggle/input")
    assert resolved["skipped_links"] == [{"path": "/kaggle/input/rootlink", "target": "/"}]
    assert [path for path in scanned if os.path.commonpath([path, base]) != base] == []
    assert len(scanned) == len(set(scanned))  # every folder once
    found = [item["path"] for item in resolved["arc_agi_files"]]
    assert [path for path in found if "outside" in path or "rootlink" in path] == []
    report, _, md = session.run()
    assert report["resolved"]["skipped_links"] == resolved["skipped_links"]
    assert "not entered: `/kaggle/input/rootlink` -> `/`" in md


def test_missing_mounts_are_reported_and_the_rest_still_written(tmp_path, clean_env):
    root = tmp_path / "bare"
    root.mkdir()
    assert probe.main(["--root", str(root)]) == 0
    report = json.loads((root / "kaggle/working/run_logs/env_probe.json").read_text("utf-8"))
    assert report["listings"]["/kaggle/input"]["exists"] is False
    assert report["listings"]["/kaggle/usr/lib"]["exists"] is False
    assert report["resolved"]["arc_agi_files"] == report["resolved"]["model_configs"] == []
    assert "FileNotFoundError" in report["resolved"]["errors"][0]
    assert report["evaluation_stats"] is None
    assert "no arc-agi_evaluation_challenges.json" in report["errors"]["evaluation_stats"]
    assert set(report["errors"]) == {"os_release", "memory", "evaluation_stats"}
    assert report["image"] == {"/etc/git_commit": None, "/etc/build_date": None}
    assert "error" in report["disk"]["/kaggle/working"]
    assert report["python"]["version"] == sys.version
    md = (root / "kaggle/working/run_logs/env_probe.md").read_text("utf-8")
    assert "## Listing /kaggle/input\n\nNot present." in md


def test_unreadable_challenges_file_is_reported(session, monkeypatch):
    real = probe.read_bytes

    def deny(path):
        if path.name == "arc-agi_evaluation_challenges.json":
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)

    monkeypatch.setattr(probe, "read_bytes", deny)
    report, _, md = session.run()
    assert report["evaluation_stats"] is None
    assert report["errors"]["evaluation_stats"].startswith("PermissionError")
    assert report["resolved"]["arc_agi_files"] and report["memory"]  # the rest still ran
    assert "Not available: PermissionError" in md


def test_malformed_challenges_file_is_reported(session):
    session.challenges.write_text("{not json", encoding="utf-8")
    report, _, _ = session.run()
    assert report["evaluation_stats"] is None
    assert report["errors"]["evaluation_stats"].startswith("JSONDecodeError")


def test_malformed_task_is_flagged_by_type_only():
    tasks = {"good": TASKS["0a1b2c3d"], "bad": {"train": [{"input": grid(4, 4, 1)}], "test": []}}
    stats = probe.challenge_stats(tasks)
    assert stats["per_task"] == {"bad": {"error": "KeyError"}, "good": EXPECTED_STATS["0a1b2c3d"]}
    assert (stats["summary"]["tasks"], stats["summary"]["malformed"]) == (2, 1)


def test_versions_come_from_metadata_for_every_listed_name(monkeypatch):
    installed = {"torch": "2.8.0+cu128", "flash_attn": "2.8.3", "flash-attn": "2.8.3"}

    def version(name):
        if name in installed:
            return installed[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    versions = probe.package_versions()
    assert list(versions) == list(BRIEF_DISTRIBUTIONS)
    assert versions == {name: installed.get(name) for name in BRIEF_DISTRIBUTIONS}


def test_pip_list_runs_this_interpreter_without_secrets_or_a_version_check(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(PIP_LIST), stderr="")

    monkeypatch.setattr(probe.subprocess, "run", run)
    secrets = {"HF_TOKEN": "h" * 20, "KAGGLE_USER_SECRETS_TOKEN": SECRET, "DB_PASSWORD": "p" * 9}
    assert REAL_PIP_LIST({"PATH": "/usr/bin", "PIP_NO_INPUT": "1", **secrets}) == PIP_LIST
    ((command, kwargs),) = calls
    assert command[:4] == [sys.executable, "-m", "pip", "list"]
    assert {"--format=json", "--disable-pip-version-check"} <= set(command)
    assert kwargs["timeout"] == probe.PIP_TIMEOUT_S
    assert kwargs["env"] == {"PATH": "/usr/bin", "PIP_NO_INPUT": "1"}


@pytest.mark.parametrize("outcome", ["exit", "timeout", "not json"])
def test_pip_list_failure_is_a_section_error(session, monkeypatch, outcome):
    def run(command, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "exit":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="No module named pip")
        return subprocess.CompletedProcess(command, 0, stdout="pip 25.2", stderr="")

    monkeypatch.setattr(probe, "pip_list", REAL_PIP_LIST)
    monkeypatch.setattr(probe.subprocess, "run", run)
    report, _, md = session.run()
    assert report["pip_list"] is None
    assert set(report["errors"]) == {"pip_list"}
    assert list(report["versions"]) == list(BRIEF_DISTRIBUTIONS)
    assert "`pip list` failed" in md


def test_unknown_arguments_are_ignored(session):
    session.run("-f", "/root/.local/share/jupyter/runtime/kernel-1.json")


def test_script_runs_standalone(session):
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "env_probe.py"), "--root", str(session.root)],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ),
    )
    assert done.returncode == 0, done.stderr
    assert "wrote /kaggle/working/run_logs/env_probe.json" in done.stdout
    for name in ("env_probe.json", "env_probe.md"):
        assert SECRET not in (session.logs / name).read_text(encoding="utf-8")


def test_constants_match_the_brief():
    assert probe.DISTRIBUTIONS == BRIEF_DISTRIBUTIONS
    assert set(probe.SHOWN_VALUES) == set(SHOWN)
    assert set(probe.SECRET_MARKERS) == BRIEF_MARKERS
    assert (probe.LIST_DEPTH, probe.LIST_CAP) == (4, 2000)


def test_probe_imports_only_the_standard_library():
    tree = ast.parse((SCRIPTS / "env_probe.py").read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.add(node.module.split(".")[0])
    assert modules - sys.stdlib_module_names - {"__future__"} == set()


# The builder.

IMAGE = "gcr.io/kaggle-gpu-images/python@sha256:" + "ab" * 32
SOURCE = "sorokin/pip-install-unsloth-flash-patch/7"
MODEL = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
VARIANT_ARGS = {"latest": [], "pinned": ["--docker-image", IMAGE]}


def slugify(title):
    """python-slugify's default for an ASCII title, which kaggle-cli compares with the slug."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build(tmp_path, variant, *extra):
    out = tmp_path / "dist" / "K0" / variant
    argv = ["--owner", OWNER, "--variant", variant, "--out", str(out), *VARIANT_ARGS[variant]]
    assert builder.main([*argv, *extra]) == 0
    return out


def read_meta(folder):
    return json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))


def refused(tmp_path, *argv):
    """Run the builder expecting a refusal; return its exit code and the unwritten folder."""
    out = tmp_path / "refused"
    try:
        code = builder.main(["--out", str(out), *argv])
    except SystemExit as exc:  # argparse refused the value
        code = exc.code
    return code, out


def no_subprocess(*args, **kwargs):
    raise AssertionError(f"a test tried to start a process: {args}")


@pytest.mark.parametrize("source", [None, SOURCE])
@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_metadata_of_each_variant(tmp_path, variant, source):
    extra = [] if source is None else ["--kernel-source", source]
    meta = read_meta(build(tmp_path, variant, *extra))
    expected = {
        "id": f"{OWNER}/arc26-k0-env-probe-{variant}",
        "code_file": "env_probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": False,
        "docker_image_pinning_type": {"latest": "latest", "pinned": "original"}[variant],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": [MODEL],
        "kernel_sources": [] if source is None else [source],
        "dataset_sources": [],
    }
    if variant == "pinned":
        expected["docker_image"] = IMAGE
    assert {key: value for key, value in meta.items() if key != "title"} == expected
    assert {"enable_gpu", "enable_tpu", "machine_shape"} & set(meta) == set()


@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_title_slugifies_to_the_slug(tmp_path, variant):
    meta = read_meta(build(tmp_path, variant))
    assert len(meta["title"]) >= 5  # kaggle-cli's minimum
    assert slugify(meta["title"]) == meta["id"].split("/")[1]


@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_folder_holds_a_copy_of_the_probe_and_its_metadata(tmp_path, variant):
    folder = build(tmp_path, variant)
    names = sorted(path.name for path in folder.iterdir())
    assert names == ["env_probe.py", "kernel-metadata.json"]
    assert (folder / "env_probe.py").read_bytes() == (SCRIPTS / "env_probe.py").read_bytes()


@pytest.mark.parametrize("source", [None, SOURCE])
@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_both_variants_pass_the_gateway_cpu_checks(tmp_path, monkeypatch, variant, source):
    monkeypatch.setattr(ops, "KAGGLE_OWNER", OWNER)
    extra = [] if source is None else ["--kernel-source", source]
    folder = build(tmp_path, variant, *extra).resolve()
    meta = ops.load_kernel_metadata(folder)
    assert ops.uses_gpu(meta, None) is False
    ops.check_kernel_metadata(meta, False)
    ops.check_code_file(folder, meta)
    assert ops.accelerator_id(meta, None) is None
    ops.check_flags(meta, None)


@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_gateway_sends_the_folder_as_a_plain_cpu_push(tmp_path, monkeypatch, variant):
    calls = []
    monkeypatch.setattr(ops, "KAGGLE_OWNER", OWNER)
    monkeypatch.setattr(ops, "_kaggle", lambda *args: calls.append(list(args)) or 0)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    folder = build(tmp_path, variant, "--kernel-source", SOURCE)
    # The real docs/prerun/: a brief naming a probe slug would make the gateway refuse here.
    assert ops.main(["push", str(folder), "--timeout", "1800"]) == 0
    assert calls == [["kernels", "push", "-p", str(folder.resolve()), "-t", "1800"]]


def test_variants_have_separate_slugs_that_no_brief_names(tmp_path):
    slugs = {read_meta(build(tmp_path, v))["id"].split("/")[1] for v in builder.VARIANTS}
    assert slugs == {"arc26-k0-env-probe-latest", "arc26-k0-env-probe-pinned"}
    briefed = {notebook.split("/")[-1].lower() for notebook in ops.briefed_notebooks()}
    assert briefed & slugs == set()


@pytest.mark.parametrize("image", [None, ""])
def test_pinned_variant_without_docker_image_is_refused(tmp_path, capsys, image):
    extra = [] if image is None else ["--docker-image", image]
    code, out = refused(tmp_path, "--owner", OWNER, "--variant", "pinned", *extra)
    assert code == 2
    assert not out.exists()
    assert "--docker-image" in capsys.readouterr().err
    with pytest.raises(builder.BuildError, match="docker-image"):
        builder.build(out, OWNER, "pinned", image)
    assert not out.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--owner", OWNER, "--variant", "latest", "--docker-image", IMAGE],
        ["--owner", OWNER, "--variant", "pinned", "--docker-image", "two words"],
        ["--owner", "a/b", "--variant", "latest"],
        ["--owner", "..", "--variant", "latest"],
        ["--owner", "", "--variant", "latest"],
        ["--owner", OWNER, "--variant", "gpu"],
        ["--owner", OWNER, "--variant", "latest", "--kernel-source", SOURCE.rsplit("/", 1)[0]],
        ["--owner", OWNER, "--variant", "latest", "--kernel-source", SOURCE[:-1] + "v7"],
        ["--owner", OWNER, "--variant", "latest", "--kernel-source", SOURCE[:-1] + "\u0667"],
        ["--owner", OWNER, "--variant", "latest", "--kernel-source", "sorokin/a/b/7"],
        ["--owner", OWNER, "--variant", "latest", "--kernel-source", "../pip-install-x/7"],
        ["--owner", OWNER, "--variant", "latest", "--kernel-source", ""],
    ],
)
def test_bad_arguments_are_refused_before_anything_is_written(tmp_path, argv):
    code, out = refused(tmp_path, *argv)
    assert code == 2
    assert not out.exists()
