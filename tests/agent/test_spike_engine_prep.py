"""Checks the engine preparation of the agent spike (P6.3-01, attempt 2's fault and its fix).

Attempt 2 installed FlashInfer's sm_89 jit-cache provider without the ``flashinfer-jit-cache``
shim; ``import flashinfer.comm`` then read ``flashinfer_jit_cache.__version__`` from a namespace
package and every tensor-parallel worker died. These tests rebuild that layout in throwaway venvs
with small stand-ins for the three packages involved (no vLLM, no FlashInfer, no network) and
check ``prepare_engine`` and its steps: the orphan provider and its empty namespace folders go,
a shim is left alone, folders with other files or outside the venv stop the run, the import
check stops at the first error with its text, and the image's failing ``sitecustomize`` is
replaced.
"""

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "agent_spike.py"
CONFIG = REPO / "configs" / "P6.3-01.yaml"
FAR = "vllm.distributed.device_communicators.flashinfer_all_reduce"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = load(SCRIPT, "_test_agent_spike_prep")

# FlashInfer 0.7.0.post1's import path, reduced to the lines that matter (flashinfer/jit/env.py:
# has_flashinfer_jit_cache, then _check_jit_cache_version(..., flashinfer_jit_cache.__version__)).
FLASHINFER_INIT = """\
import importlib.util
if importlib.util.find_spec("flashinfer_jit_cache") is not None:
    import flashinfer_jit_cache
    VERSION = flashinfer_jit_cache.__version__
"""
# vLLM 0.31.0's guard around it (vllm/distributed/device_communicators/flashinfer_all_reduce.py).
VLLM_FAR = """\
fi_ar_available = False
try:
    import flashinfer.comm as flashinfer_comm
    fi_ar_available = hasattr(flashinfer_comm, "allreduce_fusion")
except ImportError:
    pass
"""
PROVIDER_FILES = {
    "flashinfer_jit_cache/providers/sm89/__init__.py": "MANIFEST = {}\n",
    "flashinfer_jit_cache/providers/sm89/jit_cache/batch_decode.so": "\0" * 16,
}


def make_venv(path, with_pip=False):
    command = [sys.executable, "-m", "venv", str(path)]
    if not with_pip:
        command.insert(3, "--without-pip")
    subprocess.run(command, check=True, capture_output=True, timeout=300)
    python = path / "bin" / "python"
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    return python, Path(purelib)


def install(site, name, version, files):
    """Install ``files`` as distribution ``name`` the way pip leaves it: files plus a RECORD."""
    dist_info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    (dist_info / "METADATA").write_text(metadata)
    rows = []
    for relative, text in files.items():
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        rows.append(f"{relative},,")
    rows += [f"{dist_info.name}/METADATA,,", f"{dist_info.name}/RECORD,,"]
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n")


def attempt_2_venv(tmp_path):
    """A venv laid out as attempt 2's: the sm_89 provider, no shim, FlashInfer, vLLM's guard."""
    python, site = make_venv(tmp_path / "venv")
    install(site, "flashinfer-jit-cache-sm89", "0.7.0.post1+cu130", PROVIDER_FILES)
    install(site, "flashinfer-python", "0.7.0.post1", {
        "flashinfer/__init__.py": FLASHINFER_INIT,
        "flashinfer/comm.py": "allreduce_fusion = object()\n",
    })  # fmt: skip
    install(site, "vllm", "0.31.0", {
        "vllm/__init__.py": "",
        "vllm/distributed/__init__.py": "",
        "vllm/distributed/device_communicators/__init__.py": "",
        "vllm/distributed/device_communicators/flashinfer_all_reduce.py": VLLM_FAR,
    })  # fmt: skip
    return python, site


def pip_like_uninstall(site, names):
    """What ``pip uninstall`` does to these layouts: it removes each RECORD file, and the folders
    that directly held them, but never a parent folder that held only folders."""
    for name in names:
        dist_info = next(site.glob(f"{name.replace('-', '_')}-*.dist-info"))
        records = (dist_info / "RECORD").read_text().splitlines()
        parents = set()
        for row in records:
            path = site / row.split(",")[0]
            if path.is_file():
                path.unlink()
                parents.add(path.parent)
        for folder in sorted(parents, key=lambda p: len(p.parts), reverse=True):
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()


@pytest.fixture
def fake_pip(monkeypatch):
    """Route ``python -m pip uninstall`` to ``pip_like_uninstall`` (the venvs have no pip)."""
    real = spike.run_command
    calls = []

    def run(command, log, timeout_s, env):
        if command[1:4] == ["-m", "pip", "uninstall"]:
            calls.append(command)
            site = Path(subprocess.run(
                [command[0], "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                check=True, capture_output=True, text=True,
            ).stdout.strip())  # fmt: skip
            pip_like_uninstall(site, [c for c in command[4:] if not c.startswith("-")])
            return 0, ["Successfully uninstalled"]
        return real(command, log, timeout_s, env)

    monkeypatch.setattr(spike, "run_command", run)
    return calls


def engine_env():
    return spike.clean_env(os.environ)


# --- attempt 2's fault and the fix -------------------------------------------------------------


def test_attempt_2_layout_fails_the_import_check_like_the_workers(tmp_path):
    python, _ = attempt_2_venv(tmp_path)
    report = spike.check_engine_imports(python, engine_env(), [FAR], tmp_path / "log", 60)
    assert report["ok"] is False and report["failed"] == FAR
    assert report["error"] == (
        "AttributeError: module 'flashinfer_jit_cache' has no attribute '__version__'"
    )
    assert any("flashinfer/__init__.py" in line for line in report["traceback"])


def test_prepare_engine_drops_the_orphan_and_the_imports_then_pass(tmp_path, fake_pip):
    python, site = attempt_2_venv(tmp_path)
    settings = {"engine_imports": [FAR], "engine_check_timeout_s": 60}
    env, record = spike.prepare_engine(python, engine_env(), settings, tmp_path, tmp_path / "s")
    jit = record["jit_cache"]
    assert jit["state"] == "orphan" and jit["removed"] == ["flashinfer-jit-cache-sm89"]
    assert jit["before"]["shim"] is None and jit["before"]["has_version"] is False
    # pip left the namespace folders (they held only folders); the step removed them.
    assert jit["folders"] == [str((site / "flashinfer_jit_cache").resolve())]
    assert jit["after"]["found"] is False and not (site / "flashinfer_jit_cache").exists()
    assert len(fake_pip) == 1 and fake_pip[0][-1] == "flashinfer-jit-cache-sm89"
    imports = record["imports"]
    assert imports["ok"] is True and imports["flashinfer_all_reduce_available"] is True
    assert [name for name, _ in imports["modules"]] == [FAR]
    # FlashInfer itself stays: vLLM imports it in every worker.
    assert (site / "flashinfer" / "__init__.py").is_file()
    assert env["PYTHONPATH"] == str(tmp_path / "s" / "engine_site")
    written = json.loads((tmp_path / "engine_prep.json").read_text())
    assert written["jit_cache"]["state"] == "orphan" and written["imports"]["ok"] is True


def test_a_second_prepare_finds_nothing_to_drop(tmp_path, fake_pip):
    python, _ = attempt_2_venv(tmp_path)
    settings = {"engine_imports": [FAR], "engine_check_timeout_s": 60}
    spike.prepare_engine(python, engine_env(), settings, tmp_path, tmp_path / "s")
    _, record = spike.prepare_engine(python, engine_env(), settings, tmp_path, tmp_path / "s")
    assert record["jit_cache"]["state"] == "absent" and record["jit_cache"]["removed"] == []


def test_a_provider_with_its_shim_is_left_alone(tmp_path, fake_pip):
    python, site = make_venv(tmp_path / "venv")
    install(site, "flashinfer-jit-cache", "0.7.0.post1+cu130", {
        "flashinfer_jit_cache/__init__.py": "__version__ = '0.7.0.post1+cu130'\n",
    })  # fmt: skip
    install(site, "flashinfer-jit-cache-sm89", "0.7.0.post1+cu130", PROVIDER_FILES)
    record = spike.drop_orphan_jit_cache(python, engine_env(), tmp_path / "log")
    assert record["state"] == "ready" and record["removed"] == [] and record["folders"] == []
    assert record["before"]["shim"] == "0.7.0.post1+cu130"
    assert record["before"]["providers"] == ["flashinfer-jit-cache-sm89"]
    assert (site / "flashinfer_jit_cache/providers/sm89/__init__.py").is_file()
    assert fake_pip == []


def test_a_namespace_holding_other_files_stops_the_run(tmp_path, fake_pip):
    python, site = attempt_2_venv(tmp_path)
    (site / "flashinfer_jit_cache/providers/notes.txt").write_text("not the provider's")
    with pytest.raises(spike.Stop, match="still imports without __version__"):
        spike.drop_orphan_jit_cache(python, engine_env(), tmp_path / "log")
    assert (site / "flashinfer_jit_cache/providers/notes.txt").is_file()


def test_a_namespace_outside_the_venv_is_never_removed(tmp_path, fake_pip):
    python, _ = make_venv(tmp_path / "venv")
    outside = tmp_path / "elsewhere"
    (outside / "flashinfer_jit_cache" / "providers").mkdir(parents=True)
    env = {**engine_env(), "PYTHONPATH": str(outside)}
    with pytest.raises(spike.Stop, match="still imports"):
        spike.drop_orphan_jit_cache(python, env, tmp_path / "log")
    assert (outside / "flashinfer_jit_cache" / "providers").is_dir() and fake_pip == []


@pytest.mark.skipif(
    importlib.util.find_spec("ensurepip") is None, reason="needs ensurepip for a venv with pip"
)
def test_real_pip_leaves_the_namespace_and_the_step_removes_it(tmp_path):
    """The same fix with real pip: an offline install of a hand-built provider wheel."""
    python, site = make_venv(tmp_path / "venv", with_pip=True)
    wheel = tmp_path / "flashinfer_jit_cache_sm89-0.7.0.post1+cu130-py3-none-any.whl"
    dist_info = "flashinfer_jit_cache_sm89-0.7.0.post1+cu130.dist-info"
    files = {
        **PROVIDER_FILES,
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\nName: flashinfer-jit-cache-sm89\nVersion: 0.7.0.post1+cu130\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    rows = []
    for path, text in files.items():
        digest = hashlib.sha256(text.encode()).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        rows.append(f"{path},sha256={encoded},{len(text.encode())}")
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
        archive.writestr(f"{dist_info}/RECORD", "\n".join(rows + [f"{dist_info}/RECORD,,"]))
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps",
         "--disable-pip-version-check", str(wheel)],
        check=True, capture_output=True, timeout=300,
    )  # fmt: skip
    record = spike.drop_orphan_jit_cache(python, engine_env(), tmp_path / "log")
    assert record["state"] == "orphan" and record["removed"] == ["flashinfer-jit-cache-sm89"]
    assert record["folders"] == [str((site / "flashinfer_jit_cache").resolve())]
    assert record["after"]["found"] is False


# --- the import check and the boot's error text ------------------------------------------------


def test_the_import_check_stops_at_the_first_error_with_its_text(tmp_path):
    python, site = make_venv(tmp_path / "venv")
    (site / "broken.py").write_text("raise AttributeError('no __version__ here')\n")
    modules = ["json", "broken", "never_reached"]
    report = spike.check_engine_imports(python, engine_env(), modules, tmp_path / "log", 60)
    assert report["ok"] is False and report["failed"] == "broken" and report["exit"] == 1
    assert report["error"] == "AttributeError: no __version__ here"
    assert [name for name, _ in report["modules"]] == ["json"]
    with pytest.raises(spike.Stop, match="failed at broken .*AttributeError: no __version__"):
        spike.prepare_engine(
            python, engine_env(), {"engine_imports": modules, "engine_check_timeout_s": 60},
            tmp_path, tmp_path / "s",
        )  # fmt: skip
    written = json.loads((tmp_path / "engine_prep.json").read_text())
    assert written["imports"]["failed"] == "broken" and "AttributeError" in written["error"]


def test_exception_lines_name_attempt_2s_cause():
    log = "\n".join([
        "Error in sitecustomize; set PYTHONVERBOSE for traceback:",
        "(Worker pid=1018) ERROR 10-10 12:42:37 [multiproc_executor.py:948] Traceback (most "
        "recent call last):",
        "(Worker pid=1018) ERROR 10-10 12:42:37 [multiproc_executor.py:948] AttributeError: "
        "module 'flashinfer_jit_cache' has no attribute '__version__'",
        "(Worker pid=1016) ERROR 10-10 12:42:37 [multiproc_executor.py:948] AttributeError: "
        "module 'flashinfer_jit_cache' has no attribute '__version__'",
        "RuntimeError: Engine core initialization failed. See root cause above.",
    ])  # fmt: skip
    assert spike.exception_lines(log) == [
        "AttributeError: module 'flashinfer_jit_cache' has no attribute '__version__'",
        "RuntimeError: Engine core initialization failed. See root cause above.",
    ]


def test_a_failed_boot_names_the_logged_error(tmp_path):
    code = "import sys; print('AttributeError: boom', file=sys.stderr); sys.exit(1)"
    server = spike.Server(
        [sys.executable, "-c", code], engine_env(), tmp_path / "server.log",
        f"http://127.0.0.1:{spike.free_port()}",
    )  # fmt: skip
    server.start()
    try:
        with pytest.raises(spike.Stop, match=r"exited with code 1 .*AttributeError: boom"):
            server.wait_healthy(60)
    finally:
        server.stop(5)


# --- the image's sitecustomize -----------------------------------------------------------------


def test_the_empty_sitecustomize_replaces_a_failing_one(tmp_path):
    image = tmp_path / "image"
    image.mkdir()
    # Kaggle's /usr/lib/python3.11/sitecustomize.py starts this way (docker-python, 0eb38ee).
    (image / "sitecustomize.py").write_text("import os\n\nfrom log import Log\n")
    command = [sys.executable, "-c", "print('ran')"]
    noisy = subprocess.run(
        command, env={**engine_env(), "PYTHONPATH": str(image)}, capture_output=True, text=True
    )
    assert "Error in sitecustomize" in noisy.stderr and "No module named 'log'" in noisy.stderr
    env = spike.quiet_sitecustomize(tmp_path / "engine_site", engine_env())
    assert env["PYTHONPATH"] == str(tmp_path / "engine_site")
    # The image's file sits behind ours on the path (there: the standard library's folder).
    env["PYTHONPATH"] += os.pathsep + str(image)
    quiet = subprocess.run(command, env=env, capture_output=True, text=True)
    assert quiet.returncode == 0 and quiet.stdout == "ran\n" and quiet.stderr == ""
    # Nothing else is shadowed: the folder holds the module and its bytecode only.
    assert {p.name for p in (tmp_path / "engine_site").iterdir()} <= {
        "sitecustomize.py",
        "__pycache__",
    }


# --- the config --------------------------------------------------------------------------------


def test_the_config_imports_what_the_workers_import():
    modules = spike.load_config(CONFIG)["wheels"]["engine_imports"]
    assert modules[0] == "vllm.v1.worker.gpu_worker"
    assert modules.index("vllm.distributed.device_communicators.cuda_communicator") < modules.index(
        FAR
    )
    for name in (
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "vllm.v1.attention.backends.triton_attn",
    ):
        assert name in modules
    assert len(set(modules)) == len(modules)


@pytest.mark.parametrize("value", [[], ["vllm..x"], [""], "vllm", [3]])
def test_the_config_refuses_bad_engine_imports(value):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["wheels"]["engine_imports"] = value
    with pytest.raises(spike.ConfigError, match="engine_imports"):
        spike.check_config(raw)
