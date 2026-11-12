"""Checks the P6.3 download notebooks: config, push folders, and the generated kernels' logic.

No network, no pip, no Kaggle. The generated kernel and installer sources are executed in a
namespace (their ``__main__`` block does not run) and their functions are called with fakes for
``snapshot_download``, the Hub listing, and the command runner, so the code under test is the
code that goes to Kaggle. The push folders go through the gateway's own checks with the kaggle CLI
replaced by a recorder.
"""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
CONFIG = REPO / "configs" / "P6.3-download.yaml"


def load(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build_download_notebooks")
ops = load("kaggle_ops")
BUILD = {"commit": "0" * 40, "config": "configs/P6.3-download.yaml", "config_sha256": "1" * 64}


def namespace(source):
    scope = {"__name__": "generated"}
    exec(compile(source, "generated.py", "exec"), scope)
    return scope


def kernel(kind):
    config = builder.load_config(CONFIG)
    return namespace(builder.kernel_source(kind, config[kind], BUILD)), config[kind]


def digest_row(name, data):
    return {
        "name": name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_sha1": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
    }


# --- config ---------------------------------------------------------------------------------------


def test_every_config_value_has_a_comment():
    previous = ""
    for line in CONFIG.read_text().splitlines():
        if re.match(r"\s*[a-z_]+:", line):
            assert "#" in line or previous.lstrip().startswith("#"), line
        if line.strip():
            previous = line


def test_config_pins_the_reviewed_revision_and_a_consistent_wheel_set():
    config = builder.load_config(CONFIG)
    model, wheels = config["model"], config["wheels"]
    assert model["repo_id"] == "RedHatAI/Qwen3.8-27B-INT4"
    assert model["revision"].startswith("91bd022d") and len(model["revision"]) == 40
    assert len(model["files"]) == 12
    assert sum(item["size"] for item in model["files"]) == 19_472_990_100
    lfs = {item["name"] for item in model["files"] if item["git_sha1"] is None}
    assert lfs == {"model.safetensors", "model_mtp.safetensors", "tokenizer.json"}
    # The measured /kaggle/working (P0.5d-cpu, the pinned image) holds the files with the margin.
    free_at_start = 20_940_402_688
    assert free_at_start - 19_472_990_100 > model["free_margin_bytes"]
    assert wheels["python"] == "3.11"
    pins = set(wheels["constraints"])
    assert {"vllm==0.31.0", "torch==2.13.0", "httpx==0.28.1", "pip==26.2.1"} <= pins
    assert set(wheels["requirements"]) <= pins
    assert len(pins) == len(wheels["constraints"]) == 199


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["model"].update(revision="91bd022d"), "full 40-character"),
        (lambda c: c["model"]["files"][0].update(sha256="x"), "64 hex"),
        (lambda c: c["model"].update(kaggle_notebook="someone/arc26-dl-x"), "alyahya737/"),
        (lambda c: c["wheels"].update(requirements=["vllm==0.31.0"]), "must pin pip"),
        (lambda c: c["wheels"].update(requirements=["vllm==0.30.0", "pip==26.2.1"]), "same pin"),
        (lambda c: c["wheels"].update(expected_versions={"torch": "2.12.0"}), "not pinned so"),
        (lambda c: c["wheels"].update(python=3.11), "quoted major.minor"),
        (lambda c: c["wheels"].update(timeout_s=10), "integer from"),
        (lambda c: c.update(extra_key=1), "exactly the keys"),
    ],
)
def test_config_rules_refuse_bad_values(tmp_path, edit, message):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(builder.BuildError, match=message):
        builder.load_config(path)


# --- push folders ---------------------------------------------------------------------------------


@pytest.fixture
def folders(tmp_path):
    return builder.build(CONFIG, tmp_path / "dist", "a" * 40)


def test_build_writes_two_script_kernels_with_cpu_internet_metadata(folders):
    assert set(folders) == {"model", "wheels"}
    for kind, folder in folders.items():
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta["id"].split("/")[1] == folder.name
        assert meta["title"].replace(" ", "-") == folder.name
        assert meta["kernel_type"] == "script" and meta["code_file"] == builder.CODE_FILES[kind]
        assert meta["is_private"] is True and meta["enable_internet"] is True
        assert meta["enable_gpu"] is False and "machine_shape" not in meta
        assert meta["docker_image"].endswith(
            "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
        )
        assert meta["docker_image_pinning_type"] == "original"
        assert not ops.uses_gpu(meta, None)
        source = (folder / meta["code_file"]).read_text()
        compile(source, meta["code_file"], "exec")
        assert "a" * 40 in source.splitlines()[0]
    assert (folders["model"].parent / "MANIFEST.md").is_file()


def test_gateway_sends_each_folder_as_a_plain_cpu_push(folders, monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_kaggle", lambda *args: calls.append(list(args)) or 0)
    for folder in folders.values():
        assert ops.main(["push", str(folder), "--timeout", "3600"]) == 0
    assert calls == [
        ["kernels", "push", "-p", str(folder.resolve()), "-t", "3600"]
        for folder in folders.values()
    ]
    # No pre-run brief may name a download slug: the gateway would refuse the CPU push.
    briefed = {notebook.lower() for notebook in ops.briefed_notebooks()}
    config = builder.load_config(CONFIG)
    assert {config[k]["kaggle_notebook"] for k in builder.KINDS} & briefed == set()


def test_generated_kernels_carry_the_config_and_the_installer():
    model, settings = kernel("model")
    assert model["SETTINGS"] == settings and model["KIND"] == "model"
    assert model["BUILD"] == BUILD
    wheels, _ = kernel("wheels")
    installer = wheels["INSTALLER_SOURCE"]
    assert installer == builder.installer_source(BUILD)
    scope = namespace(installer)
    assert callable(scope["install_wheelhouse"]) and callable(scope["installer_main"])
    assert "INSTALLER_SOURCE" not in model


# --- the model kernel -----------------------------------------------------------------------------


FILES = {
    "config.json": b'{"architectures": ["Qwen3_5ForConditionalGeneration"]}\n',
    "model.safetensors": os.urandom(3000),
    "tokenizer.json": b'{"version": "1.0"}\n',
}


def model_settings(margin=0):
    return {
        "repo_id": "org/model",
        "revision": "b" * 40,
        "folder": "model",
        "files": [digest_row(name, data) for name, data in FILES.items()],
        "free_margin_bytes": margin,
        "max_workers": 2,
        "attempts": 2,
    }


def hub_listing(settings, sha=None):
    rows = [
        {**row, "git_sha1": None}
        if row["name"].endswith(".safetensors")
        else {**row, "sha256": None}
        for row in settings["files"]
    ]
    return lambda repo, revision: {"sha": sha or settings["revision"], "files": rows}


def writer(contents=None, fail_first=False):
    calls = []

    def snapshot(repo, revision, local_dir, workers):
        calls.append((repo, revision, local_dir, workers))
        if fail_first and len(calls) == 1:
            raise OSError("connection reset")
        os.makedirs(os.path.join(local_dir, ".cache", "huggingface"), exist_ok=True)
        Path(local_dir, ".cache", "huggingface", "x.metadata").write_text("bookkeeping")
        for name, data in (contents or FILES).items():
            Path(local_dir, name).write_bytes(data)

    return snapshot, calls


def test_model_kernel_downloads_verifies_and_writes_the_manifest(tmp_path):
    scope, _ = kernel("model")
    settings = model_settings()
    snapshot, calls = writer()
    status = scope["download_model"](settings, str(tmp_path), snapshot, hub_listing(settings))
    assert status["ok"], status["problems"]
    assert calls == [("org/model", "b" * 40, str(tmp_path / "model"), 2)]
    assert sorted(os.listdir(tmp_path / "model")) == sorted(FILES)  # .cache removed
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["ok"] and manifest["revision"] == "b" * 40
    assert {row["name"]: row["sha256"] for row in manifest["files"]} == {
        row["name"]: row["sha256"] for row in settings["files"]
    }
    assert json.loads((tmp_path / "run_logs" / "MANIFEST.json").read_text()) == manifest
    assert json.loads((tmp_path / "run_logs" / "status.json").read_text())["ok"] is True
    assert "ok: True" in (tmp_path / "run_logs" / "summary.md").read_text()
    assert status["hub_listing"] == "agrees with the pinned files"


def test_model_kernel_deletes_the_folder_when_a_hash_differs(tmp_path):
    scope, _ = kernel("model")
    settings = model_settings()
    corrupted = {**FILES, "model.safetensors": os.urandom(3000)}
    snapshot, _ = writer(corrupted)
    status = scope["download_model"](settings, str(tmp_path), snapshot, hub_listing(settings))
    assert not status["ok"] and status["deleted_folder"]
    assert not (tmp_path / "model").exists()
    assert "model.safetensors: sha256" in status["problems"][0]
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["ok"] is False


def test_model_kernel_refuses_an_extra_or_missing_file(tmp_path):
    scope, _ = kernel("model")
    settings = model_settings()
    snapshot, _ = writer({**FILES, "stray.bin": b"x"})
    status = scope["download_model"](settings, str(tmp_path), snapshot, hub_listing(settings))
    assert not status["ok"] and "stray.bin: unexpected file" in status["problems"][0]
    snapshot, _ = writer({k: v for k, v in FILES.items() if k != "tokenizer.json"})
    status = scope["download_model"](settings, str(tmp_path), snapshot, hub_listing(settings))
    assert not status["ok"] and "tokenizer.json: missing" in status["problems"][0]


def test_model_kernel_stops_before_downloading_without_room(tmp_path):
    scope, _ = kernel("model")
    settings = model_settings(margin=10**18)
    snapshot, calls = writer()
    status = scope["download_model"](settings, str(tmp_path), snapshot, hub_listing(settings))
    assert not status["ok"] and calls == []
    assert "bytes free" in status["problems"][0]


def test_model_kernel_refuses_a_hub_listing_that_differs_from_the_pins(tmp_path):
    scope, _ = kernel("model")
    settings = model_settings()
    snapshot, calls = writer()
    hub = hub_listing(settings, sha="c" * 40)
    status = scope["download_model"](settings, str(tmp_path), snapshot, hub)
    assert not status["ok"] and calls == []
    assert "resolved the revision" in status["problems"][0]


def test_model_kernel_uses_the_pins_when_the_hub_listing_fails_and_retries_a_download(tmp_path):
    scope, _ = kernel("model")
    settings = model_settings()
    snapshot, calls = writer(fail_first=True)

    def offline(repo, revision):
        raise OSError("no route")

    status = scope["download_model"](settings, str(tmp_path), snapshot, offline)
    assert status["ok"], status["problems"]
    assert len(calls) == 2 and status["download_attempts"][0]["ok"] is False
    assert status["hub_listing"].startswith("not read (OSError")


# --- the wheels kernel ----------------------------------------------------------------------------


class FakePip:
    """Plays pip and the installer: records commands and writes what each would write."""

    def __init__(self, install_ok=True, versions=None, extra_ok=True):
        self.calls = []
        self.install_ok = install_ok
        self.extra_ok = extra_ok
        self.versions = versions or {"vllm": "0.31.0", "torch": "2.13.0"}

    def __call__(self, command, log_path, timeout, env=None):
        self.calls.append(command)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        dest = command[command.index("--dest") + 1] if "--dest" in command else None
        if dest:
            os.makedirs(dest, exist_ok=True)
        if "pip==26.2.1" in command and "--no-deps" in command:
            Path(dest, "pip-26.2.1-py3-none-any.whl").write_bytes(b"pip")
        elif "-r" in command and "download" in command:
            for name, version in {**self.versions, "httpx": "0.28.1", "pip": "26.2.1"}.items():
                Path(dest, f"{name}-{version}-py3-none-any.whl").write_bytes(name.encode())
        elif "--find-links" in command and "download" in command:
            if not self.extra_ok:
                return {
                    "command": command,
                    "exit": 1,
                    "seconds": 0.1,
                    "timed_out": False,
                    "tail": [],
                }
            Path(dest, "flashinfer_jit_cache_sm89-0.7.0.post1+cu130-cp39-abi3-x.whl").write_bytes(
                b"f"
            )
        elif "--report" in command:
            report = {
                "ok": self.install_ok,
                "seconds": 12.5,
                "imports": {"torch": {"ok": True, "version": "2.13.0"}},
                "steps": [{"step": "install", "exit": 0 if self.install_ok else 1, "seconds": 9.0}],
                "problems": [] if self.install_ok else ["install exited 1"],
            }
            Path(command[command.index("--report") + 1]).write_text(json.dumps(report))
            return {"command": command, "exit": 0 if self.install_ok else 1, "seconds": 1.0,
                    "timed_out": False, "tail": []}  # fmt: skip
        return {"command": command, "exit": 0, "seconds": 0.1, "timed_out": False, "tail": []}


def wheels_run(tmp_path, fake, **changes):
    scope, settings = kernel("wheels")
    settings = {**settings, **changes}
    working, scratch = tmp_path / "working", tmp_path / "scratch"
    status = scope["build_wheelhouse"](
        settings, scope["INSTALLER_SOURCE"], str(working), fake, str(scratch)
    )
    return status, working, scratch


def test_wheels_kernel_downloads_under_constraints_and_runs_the_shipped_installer(tmp_path):
    fake = FakePip()
    status, working, scratch = wheels_run(tmp_path, fake)
    assert status["ok"], status["problems"]
    house = working / "wheelhouse"
    download = next(c for c in fake.calls if "-r" in c)
    assert download[1].endswith(
        "pip-26.2.1-py3-none-any.whl/pip"
    )  # the pinned pip, run from its wheel
    assert "--isolated" in download and "--only-binary" in download
    assert download[download.index("-c") + 1] == str(house / "constraints.txt")
    assert (house / "requirements.txt").read_text().split() == [
        "vllm==0.31.0", "httpx==0.28.1", "pip==26.2.1",
    ]  # fmt: skip
    assert len((house / "constraints.txt").read_text().split()) == 199
    assert (house / "extras.txt").read_text() == "flashinfer-jit-cache-sm89==0.7.0.post1+cu130\n"
    _, _settings = kernel("wheels")
    assert (house / "install_wheelhouse.py").read_text() == builder.installer_source(BUILD)
    check = fake.calls[-1]
    assert check[1] == str(house / "install_wheelhouse.py")
    assert check[check.index("--venv") + 1] == str(scratch / "check-venv")
    assert check[check.index("--imports") + 1] == "torch,vllm,httpx"
    manifest = json.loads((house / "MANIFEST.json").read_text())
    assert manifest["ok"] and manifest["install_check"]["ok"]
    files = {row["file"]: row for row in manifest["files"]}
    assert files["vllm-0.31.0-py3-none-any.whl"]["version"] == "0.31.0"
    assert files["vllm-0.31.0-py3-none-any.whl"]["sha256"] == hashlib.sha256(b"vllm").hexdigest()
    assert "install_wheelhouse.py" in files and "MANIFEST.json" not in files
    assert json.loads((working / "run_logs" / "status.json").read_text())["ok"] is True


def test_wheels_kernel_records_a_failed_extra_without_failing(tmp_path):
    status, working, _ = wheels_run(tmp_path, FakePip(extra_ok=False))
    assert status["ok"]
    assert status["extras"] == [
        {"requirement": "flashinfer-jit-cache-sm89==0.7.0.post1+cu130", "ok": False}
    ]
    assert (working / "wheelhouse" / "extras.txt").read_text() == ""


def test_wheels_kernel_fails_when_the_install_check_fails(tmp_path):
    status, working, _ = wheels_run(tmp_path, FakePip(install_ok=False))
    assert not status["ok"] and "install check failed" in status["problems"][0]
    assert json.loads((working / "run_logs" / "MANIFEST.json").read_text())["ok"] is False


def test_wheels_kernel_fails_on_a_wrong_engine_version(tmp_path):
    fake = FakePip(versions={"vllm": "0.30.0", "torch": "2.13.0"})
    status, _, _ = wheels_run(tmp_path, fake)
    assert not status["ok"] and "vllm 0.30.0, not 0.31.0" in status["problems"][0]
    assert not any("--report" in call for call in fake.calls)


def test_wheels_kernel_refuses_another_interpreter(tmp_path):
    fake = FakePip()
    status, _, _ = wheels_run(tmp_path, fake, python="3.99")
    assert not status["ok"] and "Python 3.99" in status["problems"][0]
    assert fake.calls == []


# --- the installer --------------------------------------------------------------------------------


def wheelhouse(tmp_path, extras=""):
    house = tmp_path / "house"
    house.mkdir()
    (house / "requirements.txt").write_text("vllm==0.31.0\nhttpx==0.28.1\npip==26.2.1\n")
    (house / "constraints.txt").write_text("vllm==0.31.0\n")
    (house / "extras.txt").write_text(extras)
    (house / "pip-26.2.1-py3-none-any.whl").write_bytes(b"pip")
    return house


class FakeRun:
    def __init__(self, fail=()):
        self.calls, self.envs, self.fail = [], [], fail

    def __call__(self, command, log_path, timeout, env=None):
        self.calls.append(command)
        self.envs.append(env)
        name = command[-1]
        code = 1 if any(f in " ".join(command) for f in self.fail) else 0
        tail = [f'IMPORT_PROBE {{"version": "1.0", "name": "{name}"}}'] if "-c" in command else []
        return {"command": command, "exit": code, "seconds": 0.1, "timed_out": False, "tail": tail}


def installer_scope():
    return namespace(builder.installer_source(BUILD))


def test_installer_creates_a_clean_venv_bootstraps_pip_and_installs_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/kaggle/usr/lib/notebooks/sorokin/pip_install")
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.invalid/simple")
    house = wheelhouse(tmp_path, extras="flashinfer-jit-cache-sm89==0.7.0.post1+cu130\n")
    venv = tmp_path / "venv"
    run = FakeRun()
    scope = installer_scope()
    report = scope["install_wheelhouse"](
        str(house), str(venv), ["torch", "vllm"], ["flashinfer"], str(tmp_path / "i.log"), run,
        "/usr/bin/python3",
    )  # fmt: skip
    assert report["ok"], report["problems"]
    vpy = str(venv / "bin" / "python")
    assert run.calls[0] == ["/usr/bin/python3", "-m", "venv", "--without-pip", str(venv)]
    assert run.calls[1][:3] == [vpy, str(house / "pip-26.2.1-py3-none-any.whl" / "pip"), "install"]
    assert run.calls[1][-1] == "pip==26.2.1"
    install = run.calls[2]
    assert install[:4] == [vpy, "-m", "pip", "install"] and "--no-index" in install
    assert install[install.index("--find-links") + 1] == str(house)
    assert install[install.index("-r") + 1] == str(house / "requirements.txt")
    assert install[install.index("-c") + 1] == str(house / "constraints.txt")
    assert "--no-compile" in install
    assert run.calls[3][-1] == "flashinfer-jit-cache-sm89==0.7.0.post1+cu130"
    assert run.calls[4] == [vpy, "-m", "pip", "check"]
    assert [call[-1] for call in run.calls[5:8]] == ["torch", "vllm", "flashinfer"]
    assert report["imports"]["torch"]["ok"] and report["imports"]["flashinfer"]["required"] is False
    for env in run.envs:
        assert "PYTHONPATH" not in env and "PIP_INDEX_URL" not in env
        assert env["PYTHONNOUSERSITE"] == "1" and env["PATH"].startswith(str(venv / "bin"))


@pytest.mark.parametrize(
    "fail, ok, problem",
    [
        ("-m pip install --no-index", False, "install exited 1"),
        ("vllm", False, "import vllm exited 1"),
        ("flashinfer", True, None),
        ("pip check", True, None),
    ],
)
def test_installer_failures_are_fatal_only_for_the_required_steps(tmp_path, fail, ok, problem):
    house = wheelhouse(tmp_path)
    report = installer_scope()["install_wheelhouse"](
        str(house), str(tmp_path / "venv"), ["torch", "vllm"], ["flashinfer"],
        str(tmp_path / "i.log"), FakeRun(fail=(fail,)), "/usr/bin/python3",
    )  # fmt: skip
    assert report["ok"] is ok
    if problem:
        assert problem in report["problems"]


def test_installer_needs_the_pip_wheel(tmp_path):
    house = wheelhouse(tmp_path)
    (house / "pip-26.2.1-py3-none-any.whl").unlink()
    run = FakeRun()
    report = installer_scope()["install_wheelhouse"](
        str(house), str(tmp_path / "venv"), ["torch"], [], str(tmp_path / "i.log"), run, "python"
    )
    assert not report["ok"] and run.calls == []
    assert "one pip wheel" in report["problems"][0]


def test_installer_script_runs_standalone_and_reports_a_failed_bootstrap(tmp_path):
    # A real venv, then the fake pip wheel (not a zip) cannot bootstrap pip: exit 1 and a report.
    house = wheelhouse(tmp_path)
    script = tmp_path / "install_wheelhouse.py"
    script.write_text(builder.installer_source(BUILD))
    report_path = tmp_path / "logs" / "install.json"
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--wheelhouse", str(house), "--venv",
         str(tmp_path / "venv"), "--report", str(report_path), "--imports", "torch"],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["steps"][0]["step"] == "venv" and report["steps"][0]["exit"] == 0
    assert (tmp_path / "venv" / "bin" / "python").exists()
    assert "pip bootstrap failed" in " ".join(report["problems"])
    assert (tmp_path / "logs" / "install.log").is_file()


def test_run_logged_tails_output_and_kills_at_the_timeout(tmp_path):
    scope = installer_scope()
    log = tmp_path / "run.log"
    done = scope["run_logged"]([sys.executable, "-c", "print('a'); print('b')"], str(log), 60)
    assert done["exit"] == 0 and done["tail"] == ["a", "b"] and not done["timed_out"]
    slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
    killed = scope["run_logged"](slow, str(log), 1.0)
    assert killed["timed_out"] and killed["exit"] != 0 and killed["seconds"] < 20
    missing = scope["run_logged"](["/nonexistent/program"], str(log), 5)
    assert missing["exit"] == 127
    assert "$ " in log.read_text()
