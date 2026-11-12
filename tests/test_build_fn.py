"""Checks the P5.2 push-folder builder (scripts/build_fn_notebooks.py) and the gateway on its
folders. No network, no Kaggle: the gateway's kaggle CLI call is replaced by a recorder.

Every build reads the committed tree (``git archive HEAD``), so these tests check what a push
of ``HEAD`` would send.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(f"_tb_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load("build_fn_notebooks")
ops = load("kaggle_ops")
HAS_WHEEL_BUILDER = shutil.which("uv") is not None or importlib.util.find_spec("pip") is not None


def digests(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def embedded(path):
    # No __pycache__ inside the push folder, which the identity checks hash.
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location(f"_emb_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module.EMBEDDED_CONFIG


def committed(rel):
    import subprocess

    return yaml.safe_load(subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, check=True,
                                         capture_output=True, text=True).stdout)


@pytest.fixture(scope="module")
def env_folder(tmp_path_factory):
    out = tmp_path_factory.mktemp("b") / "P5.2-00"
    builder.build(Path("configs/P5.2-00.yaml"), out, allow_dirty=True)
    return out


@pytest.fixture(scope="module")
def probe_folder(tmp_path_factory):
    if not HAS_WHEEL_BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    out = tmp_path_factory.mktemp("b") / "P5.2-01"
    builder.build(Path("configs/P5.2-01.yaml"), out, allow_dirty=True)
    return out


def test_the_env_build_is_byte_identical_when_rebuilt(env_folder):
    first = digests(env_folder)
    builder.build(Path("configs/P5.2-00.yaml"), env_folder, allow_dirty=True)
    assert digests(env_folder) == first
    assert set(first) == {"fn_env.py", "kernel-metadata.json", "MANIFEST.md"}


def test_the_env_kernel_is_a_private_cpu_script_with_internet(env_folder):
    meta = json.loads((env_folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "alyahya737/arc26-fn-env" and meta["kernel_type"] == "script"
    assert meta["is_private"] is True and meta["enable_internet"] is True
    assert meta["enable_gpu"] is False and "machine_shape" not in meta
    # Kaggle's latest image (Python 3.12; recipe R1), so no digest is pinned.
    assert meta["docker_image_pinning_type"] == "latest" and "docker_image" not in meta
    assert meta["dataset_sources"] == ["dfranzen/pennyroyal-v253"]
    assert len(meta["model_sources"]) == 2 and meta["kernel_sources"] == []


def test_the_embedded_config_is_the_committed_one_plus_the_probe_flags(env_folder):
    config = embedded(env_folder / "fn_env.py")
    expected = committed("configs/P5.2-00.yaml")
    probe = committed("configs/P5.2-01.yaml")
    assert config["build"]["config"] == "configs/P5.2-00.yaml"
    flags = config["server_help"].pop("flags")
    assert flags == [f for f, v in probe["engine"]["flags"].items() if v not in (None, False)]
    assert "--tensor-parallel-size" in flags and "--expert-parallel-size" in flags
    assert "--attention-backend" in flags
    config.pop("build")
    assert config == expected


def test_embed_refuses_a_script_without_its_line():
    with pytest.raises(builder.BuildError):
        builder.embed("print(1)\n", {})


def test_version_11_is_a_gpu_notebook_with_a_checked_code_dataset(probe_folder):
    meta = json.loads((probe_folder / "kernel-metadata.json").read_text())
    assert meta["id"] == "alyahya737/arc26-submission" and meta["kernel_type"] == "notebook"
    assert meta["code_file"] == "kaggle_submission.ipynb"
    assert meta["machine_shape"] == "NvidiaL4" and meta["enable_gpu"] is True
    assert meta["enable_internet"] is False and meta["docker_image_pinning_type"] == "latest"
    # Revision 17: the KV4 lineage's own code dataset; revision 22: arc26-code-fn-kv4c.
    assert meta["dataset_sources"] == ["dfranzen/pennyroyal-v253", "alyahya737/arc26-code-fn-kv4c"]
    assert meta["kernel_sources"] == ["alyahya737/arc26-fn-env/1"]
    assert meta["model_sources"] == [
        "dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/Transformers/default/1"]
    code = probe_folder / "arc26-code-fn"
    listing = json.loads((code / "fn_code.json").read_text())["files"]
    book = json.loads((probe_folder / "kaggle_submission.ipynb").read_text())
    assert "kaggle" not in book["metadata"]
    setup = "".join(book["cells"][1]["source"])
    config = json.loads(eval(setup.split("CONFIG = json.loads(", 1)[1].split(")\n", 1)[0]))
    assert listing == config["build"]["files"]
    # Revision 18: fn_kvq as one .pyz beside the scripts (kaggle datasets version skips folders,
    # and Kaggle extracts a .zip by its extension).
    assert set(listing) == {config["build"]["wheel"], "fn_probe.py", "fn_fake_sglang.py",
                            "fn_kvq.pyz"}
    extracts = (".zip", ".tar", ".gz", ".tgz", ".7z", ".bz2", ".xz")
    archives = [p.name for p in code.rglob("*") if p.name.lower().endswith(extracts)]
    assert archives == [], archives
    import io
    import zipfile

    members = zipfile.ZipFile(io.BytesIO((code / "fn_kvq.pyz").read_bytes())).namelist()
    assert "fn_kvq/sglang_kv4.py" in members and all(m.startswith("fn_kvq/") for m in members)
    # The built file imports through sys.path by zipimport, as the engine's PYTHONPATH does.
    pyz = code / "fn_kvq.pyz"
    probe_code = ("import sys; sys.path.insert(0, sys.argv[1]); import fn_kvq, fn_kvq.layout; "
                  "print(fn_kvq.__file__)")
    done = subprocess.run([sys.executable, "-S", "-c", probe_code, str(pyz)], capture_output=True,
                          text=True, env={}, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(pyz / "fn_kvq" / "__init__.py")
    for name, digest in listing.items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest
    # The dataset's scripts are the committed ones.
    for name in ("fn_probe.py", "fn_fake_sglang.py"):
        committed_bytes = subprocess.run(["git", "show", f"HEAD:scripts/{name}"], cwd=REPO,
                                         check=True, capture_output=True).stdout
        assert (code / name).read_bytes() == committed_bytes
    assert config["kernel"]["id"] == "alyahya737/arc26-submission"
    assert config["solve"]["arm"] == "tool"
    dataset = json.loads((code / "dataset-metadata.json").read_text())
    assert dataset["id"] == "alyahya737/arc26-code-fn-kv4c"


def test_the_twin_extends_version_11_on_cpu(tmp_path):
    if not HAS_WHEEL_BUILDER:
        pytest.skip("no wheel builder (uv or pip)")
    out = tmp_path / "P5.2-01-cpu"
    builder.build(Path("configs/P5.2-01-cpu.yaml"), out, allow_dirty=True)
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "alyahya737/arc26-v11-cpu-check" and meta["enable_gpu"] is False
    assert meta["enable_internet"] is False and "machine_shape" not in meta
    assert meta["kernel_type"] == "notebook"
    book = json.loads((out / "kaggle_submission.ipynb").read_text())
    setup = "".join(book["cells"][1]["source"])
    config = json.loads(eval(setup.split("CONFIG = json.loads(", 1)[1].split(")\n", 1)[0]))
    assert config["mode"] == "development" and config["engine"]["script"] == "fn_fake_sglang.py"
    assert config["engine"]["boot_retry_flags"] is None
    assert config["engine"]["flags"]["--tokenizer-dir"] == "{model}"
    # Merged from version 11: the install, the edit, the tool arm.
    assert config["install"]["enabled"] is True and config["patches"]["items"]
    assert config["solve"]["dev_arms"] == ["plain", "tool"]
    assert "extends" not in config
    # The same code dataset bytes as version 11's build of the same commit.
    gpu = tmp_path / "P5.2-01"
    builder.build(Path("configs/P5.2-01.yaml"), gpu, allow_dirty=True)
    assert digests(out / "arc26-code-fn") == digests(gpu / "arc26-code-fn")


def test_the_probe_build_is_byte_identical_when_rebuilt(probe_folder):
    first = digests(probe_folder)
    builder.build(Path("configs/P5.2-01.yaml"), probe_folder, allow_dirty=True)
    assert digests(probe_folder) == first


def stub_gateway(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_kaggle", lambda *args: calls.append(list(args)) or 0)
    return calls


def test_gateway_sends_the_env_folder_as_a_cpu_push(env_folder, monkeypatch):
    calls = stub_gateway(monkeypatch)
    assert ops.main(["push", str(env_folder), "--timeout", "3600"]) == 0
    assert calls == [["kernels", "push", "-p", str(env_folder.resolve()), "-t", "3600"]]


def test_gateway_creates_the_code_dataset(probe_folder, monkeypatch):
    calls = stub_gateway(monkeypatch)
    folder = probe_folder / "arc26-code-fn"
    assert ops.main(["dataset-create", str(folder)]) == 0
    assert calls == [["datasets", "create", "-p", str(folder.resolve())]]


def test_gateway_refuses_the_gpu_push_without_an_approved_brief(probe_folder, monkeypatch,
                                                                capsys, tmp_path):
    # P5.2-01's brief is approved (revision 6); the gateway reads a copy with the approval
    # blanked, so the refusal for a missing approval stays tested (task P5.2m, fix round 1).
    briefs = tmp_path / "prerun"
    briefs.mkdir()
    text = (REPO / "docs" / "prerun" / "P5.2-01.md").read_text(encoding="utf-8")
    lines = ["Approved-by:" if line.startswith("Approved-by:") else line
             for line in text.splitlines()]
    (briefs / "P5.2-01.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(ops, "PRERUN_DIR", briefs)
    calls = stub_gateway(monkeypatch)
    assert ops.main(["push", str(probe_folder), "--run-id", "P5.2-01"]) == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "not approved" in err


def test_the_fa2_folder_is_a_cpu_push_too(tmp_path, monkeypatch):
    out = tmp_path / "P5.2-00-fa2"
    builder.build(Path("configs/P5.2-00-fa2.yaml"), out, allow_dirty=True)
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "alyahya737/arc26-fn-fa2" and meta["enable_internet"] is True
    assert meta["model_sources"] == []
    calls = stub_gateway(monkeypatch)
    assert ops.main(["push", str(out), "--timeout", "42600"]) == 0
    assert calls[0][-2:] == ["-t", "42600"]


def test_a_digest_is_pinned_as_original():
    config = committed("configs/P5.2-00.yaml")
    config["kernel"]["docker_image"] = "gcr.io/kaggle-images/python@sha256:0123"
    meta = builder.kernel_metadata(config, [])
    assert meta["docker_image_pinning_type"] == "original"
    assert meta["docker_image"].endswith(":0123")
