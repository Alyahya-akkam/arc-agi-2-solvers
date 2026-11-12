"""Checks that the Kaggle gateway refuses what ADR-008 forbids. No network access.

Pushes, pulls, and dataset actions run through ``main`` with the module seams ``REPO``,
``PRERUN_DIR``, ``KAGGLE_OWNER``, ``_git``, and ``_kaggle`` replaced, so nothing reaches git, the
kaggle CLI, or ``~/.kaggle/``.
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATEWAY = REPO / "scripts" / "kaggle_ops.py"

_spec = importlib.util.spec_from_file_location("kaggle_ops", GATEWAY)
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)

OWNER = "tester"
RUN_ID = "P9.9-01"
COMMIT = "0123abcd4567"
SIGNED = "Approved-by: Akkam, 2026-09-29"
# The Approved-by value `prerun_sync.py record` writes.
STAMP = (
    "owner (approval page https://claude.ai/artifact/example, "
    "2026-09-29T10:00:00Z, revision {})"
)
APPROVED_BRIEF = (
    "```\nRun-id:           P9.9-01\nCode-commit:      0123abcd\n```\n\n"
    f"## 14. Approval\n\n```\n{SIGNED}\nNotes:\n```\n"
)
DROP = object()  # a metadata or brief field to leave out
KINDS = ["cpu", "accelerated"]
DATASET_COMMANDS = [("dataset-create",), ("dataset-version", "-m", "notes")]
# Verbatim from the task brief (ADR-008 hardening, change 2).
BRIEF_ACCELERATORS = {
    "NvidiaTeslaT4",
    "NvidiaL4",
    "NvidiaL4X1",
    "NvidiaTeslaA100",
    "NvidiaH100",
    "NvidiaRtxPro6000",
    "TpuV5E8",
    "TpuV6E8",
}
# The header fields both scripts read (fix round 1, item 1), as brief_text writes them.
FIELDS = {
    "Run-id": RUN_ID,
    "Revision": "1",
    "Code-commit": COMMIT,
    "Accelerator": "NvidiaL4",
    "Timeout": "3600",
    "Kaggle notebook": f"{OWNER}/probe",
}
OTHER = {
    "Run-id": "P0.0-99",
    "Revision": "2",
    "Code-commit": "fff0000",
    "Accelerator": "NvidiaL4X1",
    "Timeout": "7200",
    "Kaggle notebook": f"{OWNER}/other",
}
STRAYS = "\n".join(f"{key}: {value}" for key, value in {**OTHER, "Revision": "9"}.items())
# Top-level notebook metadata that Kaggle's editor saves.
KAGGLE_SAVED = {
    "accelerator": "nvidiaTeslaT4",
    "isGpuEnabled": True,
    "isInternetEnabled": True,
    "language": "python",
    "sourceType": "notebook",
}


def kernel_meta(**fields):
    """kernel-metadata.json of a CPU push that passes every check, updated with ``fields``.

    ``enable_gpu`` and ``enable_tpu`` are absent, which suits any accelerator ID.
    """
    meta = {
        "id": f"{OWNER}/probe",
        "title": "probe",
        "code_file": "probe.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "docker_image_pinning_type": "original",
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    meta.update(fields)
    return {key: value for key, value in meta.items() if value is not DROP}


def ipynb(**metadata):
    """A minimal nbformat 4 notebook whose top-level metadata is updated with ``metadata``."""
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    meta.update(metadata)
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["print('probe')"],
    }
    return {"cells": [cell], "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def brief_text(
    accelerator="NvidiaL4",
    timeout="3600",
    approved=True,
    revision="1",
    notebook=f"{OWNER}/probe",
    run_id=RUN_ID,
    extra_header=(),
    approval=None,
):
    """A pre-run brief with the header shape of docs/prerun/TEMPLATE.md.

    ``approved`` is True (signed), False (blank), or the text after ``Approved-by:``;
    ``approval`` replaces everything after the header block.
    """
    header = {
        "Run-id": run_id,
        "Revision": revision,
        "Task": "P9.9 gateway check",
        "Title": "accelerated push through the gateway",
        "Kind": "smoke",
        "Decides": "nothing, test fixture",
        "Code-commit": COMMIT,
        "Config": "configs/probe.yaml",
        "Kaggle notebook": notebook,
        "Accelerator": accelerator,
        "Timeout": timeout,
        "Internet": "off",
        "GPU hours": "1",
        "Quota left after": "20",
        "Previous run": "none",
    }
    kept = {key: value for key, value in header.items() if value is not DROP}
    lines = [f"{key + ':':<18}{value}".rstrip() for key, value in kept.items()]
    if approval is None:
        if approved is True:
            signature = SIGNED
        elif approved is False:
            signature = "Approved-by:"
        else:
            signature = f"Approved-by: {approved}"
        approval = f"## 14. Approval\n\n```\n{signature}\nNotes:\n```\n"
    head = "\n".join([f"# Pre-run brief: {run_id}", "", "```", *lines, *extra_header, "```", ""])
    return f"{head}\n{approval}"


def with_strays(text):
    """Put other values of the header fields before the header block and in a later block."""
    title, rest = text.split("\n", 1)
    return f"{title}\n\n{STRAYS}\n{rest}\n## 15. Appendix\n\n```\n{STRAYS}\n```\n\n{STRAYS}\n"


def no_subprocess(*args, **kwargs):
    raise AssertionError(f"a test tried to start a process: {args}")


class Gateway:
    """A temporary repository with git and the kaggle CLI replaced by recorders."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "repo"
        self.prerun = self.root / "docs" / "prerun"
        self.prerun.mkdir(parents=True)
        self.folder = tmp_path / "kernel"
        self.folder.mkdir()
        self.code(ipynb())
        self.data = tmp_path / "data"
        self.data.mkdir()
        self.calls = []
        self.outputs = []  # files the fake `kernels output` downloads
        self.status = ""  # `git status --porcelain`
        self.changed = ""  # `git diff --name-only <commit> HEAD`
        monkeypatch.setattr(ops, "REPO", self.root)
        monkeypatch.setattr(ops, "PRERUN_DIR", self.prerun)
        monkeypatch.setattr(ops, "KAGGLE_OWNER", OWNER)
        monkeypatch.setattr(ops, "_git", self.git)
        monkeypatch.setattr(ops, "_kaggle", self.kaggle)
        monkeypatch.setattr(subprocess, "Popen", no_subprocess)

    def git(self, *args):
        if args[:1] == ("status",):
            return self.status
        if args[:1] == ("diff",):
            # Only the approved commit is unchanged, so a misread Code-commit fails the push.
            return self.changed if any(COMMIT in arg for arg in args) else "arc26/other.py\n"
        return ""

    def kaggle(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        call = list(args)
        self.calls.append(call)
        if call[:2] == ["kernels", "output"]:
            dest = Path(call[call.index("-p" if "-p" in call else "--path") + 1])
            for rel in self.outputs:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                (dest / rel).write_text("x", encoding="utf-8")
        return 0

    def code(self, content, name="probe.ipynb"):
        text = content if isinstance(content, str) else json.dumps(content)
        (self.folder / name).write_text(text, encoding="utf-8")

    def meta(self, **fields):
        text = json.dumps(kernel_meta(**fields))
        (self.folder / "kernel-metadata.json").write_text(text, encoding="utf-8")

    def brief(self, name=f"{RUN_ID}.md", **fields):
        (self.prerun / name).write_text(brief_text(**fields), encoding="utf-8")

    def run(self, *argv):
        try:
            return ops.main(list(argv))
        except SystemExit as exc:  # argparse refused a malformed value
            return exc.code

    def push(self, *argv):
        return self.run("push", str(self.folder), *argv)

    def dataset(self, command, **fields):
        meta = {"id": f"{OWNER}/probe-data", "title": "probe data", "licenses": []}
        meta.update(fields)
        meta = {key: value for key, value in meta.items() if value is not DROP}
        (self.data / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        return self.run(command[0], str(self.data), *command[1:])

    def accelerated(self, accelerator="NvidiaL4", timeout="3600"):
        folder = str(self.folder.resolve())
        return ["kernels", "push", "-p", folder, "--accelerator", accelerator, "-t", timeout]

    def cpu(self, *extra):
        return ["kernels", "push", "-p", str(self.folder.resolve()), *extra]


@pytest.fixture
def gw(tmp_path, monkeypatch):
    return Gateway(tmp_path, monkeypatch)


def setup_push(gw, kind, **fields):
    """Write a push of ``kind`` that passes every check, apply ``fields``; return its argv."""
    if kind == "cpu":
        gw.meta(**fields)
        return []
    gw.meta(**{"enable_gpu": True, **fields})
    kernel_id = fields.get("id", f"{OWNER}/probe")
    gw.brief(notebook=kernel_id if isinstance(kernel_id, str) else f"{OWNER}/probe")
    return ["--run-id", RUN_ID, "--accelerator", "NvidiaL4"]


def request(gw, source, value, **fields):
    """Ask for accelerator ``value`` by flag or by ``machine_shape``; return the push argv."""
    if source == "flag":
        gw.meta(**fields)
        return ["--run-id", RUN_ID, "--accelerator", value]
    gw.meta(machine_shape=value, **fields)
    return ["--run-id", RUN_ID]


def assert_refused(gw, *argv):
    assert gw.push(*argv) == 2
    assert gw.calls == []


def test_approved_brief_returns_code_commit():
    assert ops.approved_code_commit(APPROVED_BRIEF) == "0123abcd"


@pytest.mark.parametrize(
    "approval",
    [
        "```\nApproved-by:\nNotes:\n```\n",  # template left blank
        "```\nApproved-by:   \nNotes:\n```\n",
        "```\nNotes:\n```\n",
    ],
)
def test_unapproved_brief_is_refused(approval):
    text = f"```\nRun-id:           P9.9-01\nCode-commit:      0123abcd\n```\n\n{approval}"
    with pytest.raises(ops.GatewayError, match="not approved|Approved-by"):
        ops.approved_code_commit(text)


def test_brief_without_code_commit_is_refused():
    text = f"```\nRun-id:           P9.9-01\n```\n\n```\n{SIGNED}\nNotes:\n```\n"
    with pytest.raises(ops.GatewayError, match="Code-commit"):
        ops.approved_code_commit(text)


def test_template_is_not_approved():
    text = (REPO / "docs" / "prerun" / "TEMPLATE.md").read_text(encoding="utf-8")
    with pytest.raises(ops.GatewayError):
        ops.approved_code_commit(text)


def test_missing_brief_is_refused():
    with pytest.raises(ops.GatewayError, match="no pre-run brief"):
        ops.check_brief("P9.9-99-does-not-exist")


def test_check_brief_reads_prerun_dir_at_call_time(gw):
    with pytest.raises(ops.GatewayError, match="no pre-run brief"):
        ops.check_brief(RUN_ID)


def test_gpu_push_without_run_id_is_refused(gw, capsys):
    gw.meta(enable_gpu=True)
    gw.brief()
    assert gw.push("--accelerator", "NvidiaL4") == 2
    assert "needs --run-id" in capsys.readouterr().err


@pytest.mark.parametrize("value", [DROP, False, "true"])
def test_non_private_notebook_is_refused(gw, value):
    gw.meta(is_private=value)
    with pytest.raises(ops.GatewayError, match="is_private"):
        ops.load_kernel_metadata(gw.folder)


def test_gpu_detection():
    assert ops.uses_gpu({"enable_gpu": True}, None)
    assert ops.uses_gpu({"enable_gpu": False}, "NvidiaL4")
    assert not ops.uses_gpu({"enable_gpu": False}, None)


@pytest.mark.parametrize(
    "meta, accelerator",
    [
        ({"enable_gpu": "TRUE"}, None),
        ({"enable_gpu": "tRuE"}, None),
        ({"enable_tpu": True}, None),
        ({"enable_tpu": "True"}, None),
        ({"machine_shape": "NvidiaL4"}, None),
        ({"enable_gpu": False, "machine_shape": "TpuV5E8"}, None),
        ({"machine_shape": "NvidiaTeslaP100"}, None),  # accelerated, then refused as retired
        ({"machine_shape": " "}, None),  # non-empty: the CLI would send it as is
        ({"enable_gpu": "false"}, "NvidiaL4X1"),
    ],
)
def test_accelerated_push_detection(meta, accelerator):
    assert ops.uses_gpu(meta, accelerator)


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"enable_gpu": False, "enable_tpu": False, "machine_shape": ""},
        {"enable_gpu": "false", "enable_tpu": "FALSE"},
        {"enable_gpu": "FaLsE", "machine_shape": None},
    ],
)
def test_cpu_push_detection(meta):
    assert not ops.uses_gpu(meta, None)


def test_string_false_flags_are_a_cpu_push(gw):
    gw.meta(enable_gpu="FALSE", enable_tpu="false")
    assert gw.push() == 0
    assert gw.calls == [gw.cpu()]


def test_accelerators_constant_matches_the_brief():
    assert ops.ACCELERATORS == BRIEF_ACCELERATORS


def test_machine_shape_without_enable_gpu_is_refused(gw, capsys):
    gw.meta(machine_shape="NvidiaL4")
    assert_refused(gw)
    assert "--run-id" in capsys.readouterr().err


def test_machine_shape_without_enable_gpu_needs_an_approved_brief(gw):
    gw.meta(machine_shape="NvidiaL4")
    gw.brief(approved=False)
    assert_refused(gw, "--run-id", RUN_ID)


@pytest.mark.parametrize("value", [True, "true", "TRUE"])
def test_enable_tpu_is_refused_without_brief(gw, value):
    gw.meta(enable_tpu=value)
    assert_refused(gw)


@pytest.mark.parametrize("value", ["TRUE", "True", "tRuE", "true"])
def test_enable_gpu_string_true_in_any_case_is_refused_without_brief(gw, value):
    gw.meta(enable_gpu=value)
    assert_refused(gw)


@pytest.mark.parametrize(
    "value", ["NvidiaTeslaP100", "Gpu", "4xL4", "nvidial4", "NvidiaL4x1", "TpuV38"]
)
@pytest.mark.parametrize("source", ["flag", "machine_shape"])
def test_retired_and_unknown_ids_are_refused(gw, source, value):
    gw.brief(accelerator=value)  # approved for exactly this ID, so only the allowlist can refuse
    assert_refused(gw, *request(gw, source, value))


@pytest.mark.parametrize("value", [" ", "NvidiaL4 ", " NvidiaL4"])
@pytest.mark.parametrize("source", ["flag", "machine_shape"])
def test_whitespace_in_accelerator_id_is_refused(gw, source, value):
    gw.brief(accelerator="NvidiaL4")
    assert_refused(gw, *request(gw, source, value))


@pytest.mark.parametrize("approved_id", ["NvidiaL4", "NvidiaL4X1"])
def test_accelerator_differing_from_machine_shape_is_refused(gw, approved_id):
    gw.meta(machine_shape="NvidiaL4X1")
    gw.brief(accelerator=approved_id)
    assert_refused(gw, "--run-id", RUN_ID, "--accelerator", "NvidiaL4")


@pytest.mark.parametrize(
    "source, value, approved_id",
    [
        ("flag", "NvidiaL4X1", "NvidiaL4"),
        ("machine_shape", "NvidiaL4X1", "NvidiaL4"),
        ("flag", "NvidiaL4", "NvidiaTeslaT4"),
        ("flag", "NvidiaL4", "4xL4"),  # the wording of the template before this change
        ("flag", "NvidiaL4", DROP),  # brief without an Accelerator line
    ],
)
def test_accelerator_differing_from_brief_is_refused(gw, source, value, approved_id):
    gw.brief(accelerator=approved_id)
    assert_refused(gw, *request(gw, source, value))


@pytest.mark.parametrize(
    "fields, flag, accelerator",
    [
        ({"enable_gpu": True}, True, "NvidiaL4"),
        ({"machine_shape": "NvidiaL4X1"}, False, "NvidiaL4X1"),
        ({"enable_gpu": "TRUE", "machine_shape": "NvidiaTeslaT4"}, True, "NvidiaTeslaT4"),
        ({"enable_tpu": True}, True, "TpuV5E8"),
    ],
)
def test_approved_clean_matching_accelerated_push_reaches_kaggle_with_exact_args(
    gw, fields, flag, accelerator
):
    gw.meta(**fields)
    gw.brief(accelerator=accelerator)
    argv = ["--accelerator", accelerator] if flag else []
    assert gw.push("--run-id", RUN_ID, *argv) == 0
    assert gw.calls == [gw.accelerated(accelerator)]


@pytest.mark.parametrize("accelerator", sorted(BRIEF_ACCELERATORS))
def test_every_allowed_accelerator_id_is_forwarded(gw, accelerator):
    gw.brief(accelerator=accelerator)
    assert gw.push(*request(gw, "flag", accelerator)) == 0
    assert gw.calls == [gw.accelerated(accelerator)]


def test_accelerated_push_always_forwards_an_accelerator(gw):
    gw.meta(enable_gpu=True)  # no machine_shape and no --accelerator
    gw.brief()
    code = gw.push("--run-id", RUN_ID)
    assert gw.calls in ([], [gw.accelerated()])  # refused, or the approved ID made explicit
    assert (code == 0) == bool(gw.calls)


@pytest.mark.parametrize("status, changed", [(" M arc26/ttt.py\n", ""), ("", "arc26/ttt.py\n")])
def test_unclean_or_changed_tree_refuses_accelerated_push(gw, status, changed):
    argv = setup_push(gw, "accelerated")
    gw.status, gw.changed = status, changed
    assert_refused(gw, *argv)


@pytest.mark.parametrize("kind", KINDS)
def test_missing_enable_internet_is_refused(gw, kind):
    assert_refused(gw, *setup_push(gw, kind, enable_internet=DROP))


@pytest.mark.parametrize("value", [None, 0, 1, "", "yes", "off"])
@pytest.mark.parametrize("kind", KINDS)
def test_non_boolean_enable_internet_is_refused(gw, kind, value):
    assert_refused(gw, *setup_push(gw, kind, enable_internet=value))


@pytest.mark.parametrize("value", [True, "true", "True", "TRUE"])
def test_accelerated_push_with_internet_on_is_refused(gw, value):
    assert_refused(gw, *setup_push(gw, "accelerated", enable_internet=value))


@pytest.mark.parametrize("value", [False, "false"])
def test_accelerated_push_with_internet_off_is_accepted(gw, value):
    assert gw.push(*setup_push(gw, "accelerated", enable_internet=value)) == 0
    assert gw.calls == [gw.accelerated()]


@pytest.mark.parametrize("value", [True, False, "true", "false"])
def test_cpu_push_may_use_internet_either_way(gw, value):
    assert gw.push(*setup_push(gw, "cpu", enable_internet=value)) == 0
    assert gw.calls == [gw.cpu()]


@pytest.mark.parametrize(
    "timeout", [DROP, "", "<hard stop in seconds, from the wall-time kill criterion>"]
)
def test_brief_without_timeout_is_refused(gw, timeout):
    argv = setup_push(gw, "accelerated")
    gw.brief(timeout=timeout)
    assert_refused(gw, *argv)


@pytest.mark.parametrize("timeout", ["59", "43201", "0", "-60", "86400"])
def test_brief_timeout_out_of_range_is_refused(gw, timeout):
    argv = setup_push(gw, "accelerated")
    gw.brief(timeout=timeout)
    assert_refused(gw, *argv)


@pytest.mark.parametrize("timeout", ["3600.5", "1h", "one hour", "1e4"])
def test_brief_timeout_not_an_integer_is_refused(gw, timeout):
    argv = setup_push(gw, "accelerated")
    gw.brief(timeout=timeout)
    assert_refused(gw, *argv)


@pytest.mark.parametrize("timeout", ["60", "43200"])
def test_brief_timeout_bounds_are_inclusive(gw, timeout):
    argv = setup_push(gw, "accelerated")
    gw.brief(timeout=timeout)
    assert gw.push(*argv) == 0
    assert gw.calls == [gw.accelerated(timeout=timeout)]


def test_command_line_cannot_override_the_brief_timeout(gw):
    code = gw.push(*setup_push(gw, "accelerated"), "--timeout", "7200")
    assert gw.calls in ([], [gw.accelerated()])  # refused, or the brief's 3600 wins
    assert (code == 0) == bool(gw.calls)


def test_command_line_cannot_supply_a_missing_brief_timeout(gw):
    argv = setup_push(gw, "accelerated")
    gw.brief(timeout=DROP)
    assert_refused(gw, *argv, "--timeout", "3600")


def test_cpu_push_reaches_kaggle_without_timeout(gw):
    assert gw.push(*setup_push(gw, "cpu")) == 0
    assert gw.calls == [gw.cpu()]


def test_cpu_push_forwards_timeout_when_given(gw):
    assert gw.push(*setup_push(gw, "cpu"), "--timeout", "600") == 0
    assert gw.calls == [gw.cpu("-t", "600")]


@pytest.mark.parametrize("value", [DROP, None, "", "pinned", "Original"])
@pytest.mark.parametrize("kind", KINDS)
def test_missing_or_invalid_pinning_type_is_refused(gw, kind, value):
    assert_refused(gw, *setup_push(gw, kind, docker_image_pinning_type=value))


@pytest.mark.parametrize("value", ["original", "latest"])
@pytest.mark.parametrize("kind", KINDS)
def test_original_and_latest_pinning_are_accepted(gw, kind, value):
    assert gw.push(*setup_push(gw, kind, docker_image_pinning_type=value)) == 0
    assert len(gw.calls) == 1


@pytest.mark.parametrize("value", [12345, "12345"])
@pytest.mark.parametrize("kind", KINDS)
def test_id_no_is_refused(gw, kind, value):
    assert_refused(gw, *setup_push(gw, kind, id_no=value))


@pytest.mark.parametrize("kernel_id", ["someone/probe", f"{OWNER}-evil/probe"])
@pytest.mark.parametrize("kind", KINDS)
def test_foreign_owner_is_refused(gw, kind, kernel_id):
    assert_refused(gw, *setup_push(gw, kind, id=kernel_id))


# Without a slash the CLI fills in the token's user, so the owner must be explicit.
@pytest.mark.parametrize("kernel_id", [DROP, "probe", "/probe", f"{OWNER}/", f"{OWNER}/a/b"])
@pytest.mark.parametrize("kind", KINDS)
def test_malformed_kernel_id_is_refused(gw, kind, kernel_id):
    assert_refused(gw, *setup_push(gw, kind, id=kernel_id))


@pytest.mark.parametrize("command", DATASET_COMMANDS)
def test_own_dataset_reaches_kaggle(gw, command):
    assert gw.dataset(command) == 0
    assert [call[:2] for call in gw.calls] == [["datasets", command[0].removeprefix("dataset-")]]


@pytest.mark.parametrize(
    "dataset_id", ["someone/probe-data", f"{OWNER}-evil/probe-data", "probe-data", DROP]
)
@pytest.mark.parametrize("command", DATASET_COMMANDS)
def test_foreign_dataset_owner_is_refused(gw, command, dataset_id):
    assert gw.dataset(command, id=dataset_id) == 2
    assert gw.calls == []


@pytest.mark.parametrize("kernel_id", [f"{OWNER}/probe", "/probe"])
@pytest.mark.parametrize("kind", KINDS)
def test_empty_kaggle_owner_refuses_push(gw, monkeypatch, capsys, kind, kernel_id):
    monkeypatch.setattr(ops, "KAGGLE_OWNER", "")
    assert_refused(gw, *setup_push(gw, kind, id=kernel_id))
    assert "KAGGLE_OWNER" in capsys.readouterr().err


@pytest.mark.parametrize("dataset_id", [f"{OWNER}/probe-data", "/probe-data"])
@pytest.mark.parametrize("command", DATASET_COMMANDS)
def test_empty_kaggle_owner_refuses_dataset_actions(gw, monkeypatch, capsys, command, dataset_id):
    monkeypatch.setattr(ops, "KAGGLE_OWNER", "")
    assert gw.dataset(command, id=dataset_id) == 2
    assert gw.calls == []
    assert "KAGGLE_OWNER" in capsys.readouterr().err


def test_upload_cap(gw, monkeypatch):
    gw.dataset(("dataset-create",))  # writes a valid dataset-metadata.json
    (gw.data / "blob").write_bytes(b"x" * 2048)
    monkeypatch.setattr(ops, "MAX_UPLOAD_BYTES", 1024)
    with pytest.raises(ops.GatewayError, match="upload cap"):
        ops.check_dataset_folder(gw.data)


PULL_ACCEPTS = ["submission.json", "run_logs/timing.jsonl", "run_logs/a/b.json"]
PULL_REJECTS = [
    "tokenizer.json",
    "model/config.json",
    "run_logs/../x.json",
    "run_logs/.h.json",
    "/etc/x.json",
    "x.safetensors",
    "run_logs/w.bin",
    "submission.json.bak",
    "run_logs/x.pt",
]


@pytest.mark.parametrize(
    "path",
    [
        *PULL_ACCEPTS,
        *(f"run_logs/x.{suffix}" for suffix in ("md", "csv", "txt", "log", "yaml", "png")),
        "run_logs/a/b/c/d.csv",
        "run_logs/_x.txt",
        "run_logs/-x.log",
        "run_logs/0/1.jsonl",
        "run_logs/submission.json",
    ],
)
def test_pull_allowed_accepts(path):
    assert ops.pull_allowed(path)
    assert re.search(ops.PULL_PATTERN, path)


@pytest.mark.parametrize(
    "path",
    [
        *PULL_REJECTS,
        # root-level files the old extension pattern let through
        "timing.jsonl",
        "summary.md",
        "loss.png",
        "config.json",
        # weights and other large artifacts, also under run_logs/
        "adapter.safetensors",
        "run_logs/model.bin",
        "run_logs/h.ckpt",
        "run_logs/episodes.parquet",
        "run_logs/x.npy",
        # suffixes outside the list, or none
        "run_logs/x.yml",
        "run_logs/x",
        "run_logs/x.jsonx",
        "run_logs/x.json.bak",
        "submission.jsonl",
        # escapes, empty and hidden segments
        "run_logs/./x.json",
        "run_logs/a/../../x.json",
        "run_logs//x.json",
        "run_logs/a/.b/c.json",
        "./submission.json",
        "../submission.json",
        "/submission.json",
        "/run_logs/x.json",
        # names that only contain or resemble the allowed ones
        "sub/submission.json",
        "xsubmission.json",
        "run_logsx/a.json",
        "out/run_logs/a.json",
        "run_logs/",
        "run_logs",
        "",
    ],
)
def test_pull_allowed_rejects(path):
    assert not ops.pull_allowed(path)
    assert not re.search(ops.PULL_PATTERN, path)


# The CLI filters with re.search, where `$` also matches before a final newline.
@pytest.mark.parametrize("path", ["submission.json\n", "run_logs/timing.jsonl\n"])
def test_pull_pattern_rejects_trailing_newline(path):
    assert not ops.pull_allowed(path)
    assert not re.search(ops.PULL_PATTERN, path)


def pull(gw, *outputs):
    gw.outputs = list(outputs)
    code = gw.run("pull", f"{OWNER}/probe", "--task-id", "P9.9")
    return code, gw.root / "runs" / "P9.9"


def seed(folder, *paths):
    for rel in paths:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_text("old", encoding="utf-8")


def test_pull_filters_with_the_anchored_pattern(gw):
    pull(gw)
    (call,) = gw.calls
    assert call[:2] == ["kernels", "output"] and isinstance(ops.PULL_PATTERN, str)
    assert call[call.index("--file-pattern") + 1] == ops.PULL_PATTERN


def test_pull_with_unexpected_file_is_cleaned_up_and_returns_nonzero(gw, capsys):
    kept = ["submission.json", "run_logs/timing.jsonl", "run_logs/a/b.json", "probe.log"]
    stray = ["tokenizer.json", "model/config.json", "x.safetensors"]
    code, dest = pull(gw, *kept, *stray)
    assert code != 0
    assert [p for p in kept if not (dest / p).is_file()] == []
    assert [p for p in stray if (dest / p).exists()] == []
    report = "".join(capsys.readouterr())
    assert [p for p in stray if Path(p).name not in report] == []


def test_pull_removes_a_log_that_is_not_the_session_log(gw):
    code, dest = pull(gw, "probe.log", "other.log", "nested/probe.log")
    assert code != 0
    assert (dest / "probe.log").is_file()
    assert not (dest / "other.log").exists() and not (dest / "nested" / "probe.log").exists()


def test_clean_pull_keeps_allowed_files_and_returns_zero(gw):
    files = ["submission.json", "run_logs/timing.jsonl", "run_logs/a/b.json", "probe.log"]
    code, dest = pull(gw, *files)
    assert code == 0
    assert [p for p in files if not (dest / p).is_file()] == []


def test_gateway_has_no_publish_or_delete_path():
    source = GATEWAY.read_text(encoding="utf-8")
    for token in ('"delete"', '"-d"', '"-u"', '"--public"'):
        assert token not in source, token


def test_head_has_no_code_change_since_itself():
    try:
        assert ops.files_changed_outside_docs("HEAD") == []
    except Exception as err:  # not a git checkout
        pytest.skip(str(err))


# Fix round 1, item 1: header fields come only from the first fenced block, each at most once,
# and Approved-by appears exactly once, inside the Approval block.


def test_brief_header_reads_only_the_first_fenced_block():
    header = ops.brief_header(with_strays(brief_text()))
    assert {key: header.get(key) for key in FIELDS} == FIELDS


def test_fields_outside_the_header_block_are_ignored(gw):
    argv = setup_push(gw, "accelerated")
    gw.brief(approved=STAMP.format("1"))
    brief = gw.prerun / f"{RUN_ID}.md"
    brief.write_text(with_strays(brief.read_text(encoding="utf-8")), encoding="utf-8")
    assert gw.push(*argv) == 0
    assert gw.calls == [gw.accelerated()]


def test_timeout_only_outside_the_header_block_counts_as_missing(gw):
    argv = setup_push(gw, "accelerated")
    text = with_strays(brief_text(timeout=DROP))
    (gw.prerun / f"{RUN_ID}.md").write_text(text, encoding="utf-8")
    assert ops.brief_header(text).get("Timeout") in (None, "")
    assert_refused(gw, *argv)


def test_code_commit_outside_the_header_block_is_ignored():
    header = "```\nRun-id:           P9.9-01\n```\n\n"
    text = f"Code-commit:      0123abcd\n\n{header}```\n{SIGNED}\nNotes:\n```\n"
    with pytest.raises(ops.GatewayError, match="Code-commit"):
        ops.approved_code_commit(text)


@pytest.mark.parametrize("same", [True, False])
@pytest.mark.parametrize("key", list(FIELDS))
def test_repeated_header_key_is_refused(gw, key, same):
    argv = setup_push(gw, "accelerated")
    text = brief_text(extra_header=[f"{key}: {(FIELDS if same else OTHER)[key]}"])
    (gw.prerun / f"{RUN_ID}.md").write_text(text, encoding="utf-8")
    with pytest.raises(ops.GatewayError):
        ops.brief_header(text)
    assert_refused(gw, *argv)


NO_APPROVED_BY = "## 14. Approval\n\n```\nNotes:\n```\n"
APPROVAL_SECTIONS = {
    "signed twice": f"{SIGNED}\n\n## 14. Approval\n\n```\n{SIGNED}\nNotes:\n```\n",
    "blank block, signed body": f"{SIGNED}\n\n## 14. Approval\n\n```\nApproved-by:\nNotes:\n```\n",
    "signed body only": f"## 14. Approval\n\n{SIGNED}\n\n```\nNotes:\n```\n",
    "block without Notes": f"## 14. Approval\n\n```\n{SIGNED}\n```\n",
    "not fenced": f"## 14. Approval\n\n{SIGNED}\nNotes:\n",
    "missing": NO_APPROVED_BY,
}


@pytest.mark.parametrize("case", [*APPROVAL_SECTIONS, "signed header"])
def test_approved_by_must_appear_once_inside_the_approval_block(gw, case):
    argv = setup_push(gw, "accelerated")
    if case == "signed header":
        text = brief_text(extra_header=[SIGNED], approval=NO_APPROVED_BY)
    else:
        text = brief_text(approval=APPROVAL_SECTIONS[case])
    (gw.prerun / f"{RUN_ID}.md").write_text(text, encoding="utf-8")
    assert_refused(gw, *argv)


# Fix round 1, item 2: notebooks saved by Kaggle carry their own accelerator and internet
# settings, which the push would send as is.


@pytest.mark.parametrize("kaggle", [KAGGLE_SAVED, {}])
@pytest.mark.parametrize("kind", KINDS)
def test_notebook_with_kaggle_metadata_is_refused(gw, kind, kaggle):
    argv = setup_push(gw, kind)
    gw.code(ipynb(kaggle=kaggle))
    assert_refused(gw, *argv)


@pytest.mark.parametrize(
    "kernel_type, code_file", [("notebook", "probe.py"), ("script", "probe.ipynb")]
)
@pytest.mark.parametrize("kind", KINDS)
def test_kaggle_metadata_is_refused_for_either_notebook_marker(gw, kind, kernel_type, code_file):
    argv = setup_push(gw, kind, kernel_type=kernel_type, code_file=code_file)
    gw.code(ipynb(kaggle=KAGGLE_SAVED), code_file)
    assert_refused(gw, *argv)


@pytest.mark.parametrize("case", ["missing file", "no code_file", "empty", "not json"])
@pytest.mark.parametrize("kind", KINDS)
def test_missing_or_unparseable_code_file_is_refused(gw, kind, case):
    fields = {"missing file": {"code_file": "absent.ipynb"}, "no code_file": {"code_file": DROP}}
    argv = setup_push(gw, kind, **fields.get(case, {}))
    if case in ("empty", "not json"):
        gw.code("" if case == "empty" else '{"cells": [')
    assert_refused(gw, *argv)


@pytest.mark.parametrize("kind", KINDS)
def test_script_kernel_is_not_parsed_as_a_notebook(gw, kind):
    argv = setup_push(gw, kind, kernel_type="script", code_file="probe.py")
    gw.code("print('probe')\n", "probe.py")
    assert gw.push(*argv) == 0
    assert len(gw.calls) == 1


# Fix round 1, item 3.


def test_pre_existing_other_log_and_model_config_survive_a_clean_pull(gw):
    dest = gw.root / "runs" / "P9.9"
    seed(dest, "other.log", "model/config.json")
    code, _ = pull(gw, "submission.json", "run_logs/timing.jsonl", "probe.log")
    assert code == 0
    assert [(dest / p).read_text() for p in ("other.log", "model/config.json")] == ["old", "old"]


def test_pull_removes_only_the_strays_it_downloaded(gw):
    dest = gw.root / "runs" / "P9.9"
    seed(dest, "other.log", "model/config.json")
    code, _ = pull(gw, "probe.log", "tokenizer.json")
    assert code != 0
    assert not (dest / "tokenizer.json").exists()
    assert (dest / "other.log").is_file() and (dest / "model" / "config.json").is_file()


@pytest.mark.parametrize("where", ["parent", "absolute"])
def test_run_id_outside_prerun_dir_is_refused(gw, tmp_path, where):
    run_id = "../x" if where == "parent" else str(tmp_path / "x")
    brief = gw.prerun.parent / "x.md" if where == "parent" else tmp_path / "x.md"
    brief.write_text(brief_text(run_id=run_id), encoding="utf-8")  # approved and matching
    gw.meta(enable_gpu=True)
    assert_refused(gw, "--run-id", run_id, "--accelerator", "NvidiaL4")


@pytest.mark.parametrize("task_id", ["..", "../x", ".", "", "a/../..", "ABSOLUTE"])
def test_task_id_outside_runs_is_refused(gw, tmp_path, task_id):
    outside = tmp_path / "elsewhere"
    task_id = str(outside) if task_id == "ABSOLUTE" else task_id
    sentinels = [
        gw.root / "README.md",
        gw.root / "x" / "notes.bin",
        gw.root / "runs" / "P1.1" / "run_logs" / "timing.jsonl",
        outside / "keep.bin",
    ]
    for path in sentinels:
        seed(path.parent, path.name)
    gw.outputs = ["probe.log"]
    assert gw.run("pull", f"{OWNER}/probe", "--task-id", task_id) == 2
    assert gw.calls == []
    assert [p for p in sentinels if not p.is_file()] == []


# Fix round 1, item 4: an approval covers one notebook, and CPU pushes never reuse its slug.


@pytest.mark.parametrize(
    "notebook", [f"{OWNER}/other", f"{OWNER}/probe, version to be created", DROP]
)
def test_accelerated_push_id_must_equal_its_brief_notebook(gw, notebook):
    argv = setup_push(gw, "accelerated")
    gw.brief(notebook=notebook)
    assert_refused(gw, *argv)


def test_other_briefs_for_the_same_notebook_do_not_block_an_accelerated_push(gw):
    argv = setup_push(gw, "accelerated")
    gw.brief(name="P9.9-02.md", run_id="P9.9-02", approved=False)
    assert gw.push(*argv) == 0
    assert gw.calls == [gw.accelerated()]


@pytest.mark.parametrize("run_id, approved", [(RUN_ID, True), ("P3.5-02", False)])
def test_cpu_push_to_a_briefed_notebook_is_refused(gw, run_id, approved):
    gw.brief(name=f"{run_id}.md", run_id=run_id, approved=approved)
    assert_refused(gw, *setup_push(gw, "cpu"))


def test_template_and_readme_do_not_block_a_cpu_push(gw):
    gw.brief(name="TEMPLATE.md")
    gw.brief(name="README.md")
    gw.brief(name="P3.5-02.md", run_id="P3.5-02", notebook=f"{OWNER}/gpu-probe")
    assert gw.push(*setup_push(gw, "cpu")) == 0
    assert gw.calls == [gw.cpu()]


def test_briefed_notebooks_lists_every_brief_except_template_and_readme(gw):
    gw.brief(name=f"{RUN_ID}.md")
    gw.brief(name="P3.5-02.md", run_id="P3.5-02", notebook=f"{OWNER}/other", approved=False)
    gw.brief(name="TEMPLATE.md", notebook=f"{OWNER}/template")
    gw.brief(name="README.md", notebook=f"{OWNER}/readme")
    assert ops.briefed_notebooks() == {f"{OWNER}/probe", f"{OWNER}/other"}


# Fix round 1, item 5: flags must be booleans and agree with the accelerator ID.


@pytest.mark.parametrize("value", [1, 0, None, "", "yes", "on"])
@pytest.mark.parametrize("flag", ["enable_gpu", "enable_tpu"])
@pytest.mark.parametrize("kind", KINDS)
def test_non_boolean_flag_is_refused(gw, kind, flag, value):
    assert_refused(gw, *setup_push(gw, kind, **{flag: value}))


@pytest.mark.parametrize(
    "accelerator, fields",
    [
        ("NvidiaL4", {"enable_gpu": False}),
        ("NvidiaL4", {"enable_gpu": "FALSE"}),
        ("NvidiaL4", {"enable_tpu": True}),
        ("NvidiaTeslaT4", {"enable_gpu": True, "enable_tpu": "true"}),
        ("TpuV5E8", {"enable_tpu": False}),
        ("TpuV5E8", {"enable_tpu": "false"}),
        ("TpuV5E8", {"enable_gpu": True}),
        ("TpuV6E8", {"enable_tpu": True, "enable_gpu": "TRUE"}),
    ],
)
@pytest.mark.parametrize("source", ["flag", "machine_shape"])
def test_flag_contradicting_the_id_is_refused(gw, source, accelerator, fields):
    gw.brief(accelerator=accelerator)
    assert_refused(gw, *request(gw, source, accelerator, **fields))


@pytest.mark.parametrize(
    "accelerator, fields",
    [
        ("NvidiaL4", {}),
        ("NvidiaL4", {"enable_gpu": True, "enable_tpu": False}),
        ("NvidiaTeslaT4", {"enable_gpu": "true", "enable_tpu": "false"}),
        ("TpuV5E8", {}),
        ("TpuV5E8", {"enable_tpu": True, "enable_gpu": False}),
    ],
)
@pytest.mark.parametrize("source", ["flag", "machine_shape"])
def test_absent_or_consistent_flags_are_accepted(gw, source, accelerator, fields):
    gw.brief(accelerator=accelerator)
    assert gw.push(*request(gw, source, accelerator, **fields)) == 0
    assert gw.calls == [gw.accelerated(accelerator)]


@pytest.mark.parametrize("fields", [{}, {"enable_gpu": False, "enable_tpu": False}])
def test_absent_or_false_flags_are_a_cpu_push(gw, fields):
    assert gw.push(*setup_push(gw, "cpu", **fields)) == 0
    assert gw.calls == [gw.cpu()]


# Fix round 1, item 6: a recorded approval names its revision, which must be the brief's.


@pytest.mark.parametrize("revision, stamped", [("3", "2"), ("1", "12"), ("2", "1")])
def test_approval_stamp_from_another_revision_is_refused(gw, revision, stamped):
    argv = setup_push(gw, "accelerated")
    gw.brief(revision=revision, approved=STAMP.format(stamped))
    assert_refused(gw, *argv)


@pytest.mark.parametrize(
    "revision, approved", [("2", STAMP.format("2")), ("12", STAMP.format("12")), ("3", True)]
)
def test_approval_stamp_for_this_revision_or_without_one_is_accepted(gw, revision, approved):
    argv = setup_push(gw, "accelerated")
    gw.brief(revision=revision, approved=approved)
    assert gw.push(*argv) == 0
    assert gw.calls == [gw.accelerated()]


# Fix round 1, item 7.


def test_pull_cleanup_removes_emptied_directories(gw):
    dest = gw.root / "runs" / "P9.9"
    seed(dest, "old/deep/x.bin")  # present before the pull, so not the cleanup's to remove
    kept = ["probe.log", "run_logs/a/b.json", "old/deep/x.bin"]
    code, _ = pull(gw, "probe.log", "run_logs/a/b.json", "run_logs/a/.h.json", "model/config.json",
                   "deep/er/x.safetensors")
    assert code != 0
    assert [d for d in ("model", "deep") if (dest / d).exists()] == []
    assert [p for p in kept if not (dest / p).is_file()] == []


@pytest.mark.parametrize("timeout", ["59", "43201", "0", "-60", "1.5", "one"])
def test_cpu_timeout_out_of_range_is_refused(gw, timeout):
    assert_refused(gw, *setup_push(gw, "cpu"), "--timeout", timeout)


@pytest.mark.parametrize("timeout", ["60", "43200"])
def test_cpu_timeout_bounds_are_inclusive(gw, timeout):
    assert gw.push(*setup_push(gw, "cpu"), "--timeout", timeout) == 0
    assert gw.calls == [gw.cpu("-t", timeout)]


# Review round 2: engineer additions beyond the fix-round spec.


def test_brief_run_id_must_equal_its_file_name(gw):
    argv = setup_push(gw, "accelerated")
    gw.brief(run_id="P9.9-02")  # approved and matching, but its header names another run
    assert_refused(gw, *argv)


@pytest.mark.parametrize("case", ["parent", "symlink"])
@pytest.mark.parametrize("kind", KINDS)
def test_code_file_resolving_outside_the_folder_is_refused(gw, kind, case):
    outside = gw.folder.parent / "x.ipynb"
    outside.write_text(json.dumps(ipynb()), encoding="utf-8")
    if case == "parent":
        code_file = "../x.ipynb"
    else:
        code_file = "link.ipynb"
        (gw.folder / code_file).symlink_to(outside)
    argv = setup_push(gw, kind, code_file=code_file)
    assert_refused(gw, *argv)


@pytest.mark.parametrize("run_id", ["../x", "..", "a/b"])
def test_cpu_push_with_a_bad_run_id_is_refused(gw, run_id):
    gw.meta()
    assert_refused(gw, "--run-id", run_id)
