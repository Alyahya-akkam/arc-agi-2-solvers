"""Checks the Flash-Next environment check (scripts/fn_env.py, run P5.2-00). No network.

The CUDA architecture reader runs on synthetic fat binaries and cubin headers; the whole check
runs as its CPU smoke on a fake ``/kaggle`` tree under ``tmp_path`` (the system uv builds a real
venv from a one-wheel wheelhouse); the Python 3.12 fetch runs with its commands stubbed.
"""

import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "P5.2-00.yaml"


def load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


env = load("fn_env")


def cubin_header(sm, abi):
    head = bytearray(64)
    head[:4] = b"\x7fELF"
    head[4], head[5], head[8] = 2, 1, abi
    struct.pack_into("<H", head, 18, env.EM_CUDA)
    # V1 keeps sm in bits 0 to 7 beside EF_CUDA_64BIT_ADDRESS (0x400); V2 in bits 8 to 15
    # beside its accelerator flag (0x8).
    flags = (sm << 8) | 0x8 if abi >= env.CUDA_ABI_V2 else sm | 0x400
    struct.pack_into("<I", head, 48, flags)
    return bytes(head)


def test_fat_binary_entries_are_read_by_kind_and_arch(tmp_path):
    blob = b"junk" * 10 + env.fake_fatbin([(2, 86), (1, 80), (2, 90)]) + b"tail"
    path = tmp_path / "lib.so"
    path.write_bytes(blob)
    scan = env.scan_device_code(path)
    assert scan == {"elf": [86, 90], "ptx": [80], "other": [], "fatbins": 1}
    assert env.runs_on(scan) == "native"


def test_two_fat_binaries_and_false_magic_are_handled(tmp_path):
    fake_magic = env.FATBIN_MAGIC + b"\x09\x00"  # version 9: not a fat binary
    blob = fake_magic + env.fake_fatbin([(1, 75)]) + b"\0" * 7 + env.fake_fatbin([(2, 100)])
    path = tmp_path / "two.so"
    path.write_bytes(blob)
    scan = env.scan_device_code(path)
    assert scan["fatbins"] == 2 and scan["elf"] == [100] and scan["ptx"] == [75]
    assert env.runs_on(scan) == "ptx"


@pytest.mark.parametrize(
    ("elf", "ptx", "expected"),
    [([89], [], "native"), ([80], [], "native"), ([87], [], "native"), ([90], [], None),
     ([120], [90], None), ([90], [89], "ptx"), ([75], [], None), ([], [], None)],
)
def test_runs_on_follows_binary_and_ptx_compatibility(elf, ptx, expected):
    assert env.runs_on({"elf": elf, "ptx": ptx}) == expected


@pytest.mark.parametrize("abi", [7, 8])
def test_a_standalone_cubin_names_its_sm_in_either_abi(tmp_path, abi):
    path = tmp_path / "k.cubin"
    path.write_bytes(cubin_header(89, abi) + b"\0" * 32)
    assert env.cubin_arch(path.read_bytes()[:64]) == 89
    assert env.scan_device_code(path)["elf"] == [89]


def test_a_host_elf_is_not_a_cubin():
    head = bytearray(cubin_header(89, 7))
    struct.pack_into("<H", head, 18, 62)  # x86-64
    assert env.cubin_arch(bytes(head)) is None


def test_cuobjdump_listing_is_parsed():
    text = "ELF file    1: a.sm_89.cubin\nELF file 2: a.sm_90a.cubin\nPTX file 1: a.sm_80.ptx\n"
    assert env.parse_cuobjdump(text) == [80, 89, 90]


def test_package_verdicts():
    def f(path, elf, ptx):
        return {"path": path, "elf": elf, "ptx": ptx,
                "runs_on": env.runs_on({"elf": elf, "ptx": ptx})}

    assert env.summarize_package([f("a", [89], [])])["verdict"] == "native"
    assert env.summarize_package([f("a", [89], []), f("b", [90], [80])])["verdict"] == "ptx"
    partial = env.summarize_package([f("a", [89], []), f("b", [90], [])])
    assert partial["verdict"] == "partial" and partial["sm89_missing"] == ["b"]
    assert env.summarize_package([f("a", [100], [])])["verdict"] == "none"
    assert env.summarize_package([f("a", [], [])])["verdict"] == "no device code"


def test_component_sizes_come_from_headers_alone(tmp_path):
    tensors = {"model.layers.0.mlp.experts.3.down_proj.qweight": ("I32", [4, 8]),
               "model.ngram_embedding.weight": ("F8_E4M3", [10, 4]),
               "model.layers.0.mlp.shared_expert.up_proj.weight": ("BF16", [2, 2]),
               "lm_head.weight": ("BF16", [3, 2])}
    env.write_safetensors(tmp_path / "m.safetensors", tensors)
    import yaml

    rules = [tuple(r) for r in yaml.safe_load(CONFIG.read_text())["model"]["components"]]
    sizes = env.model_sizes(tmp_path, rules, 10)
    assert sizes["bytes_by_component"] == {"embed_head": 12, "experts": 128, "ple": 40,
                                           "shared_experts": 8}
    assert sizes["errors"] == {}


def test_solutions_are_never_read(tmp_path):
    path = tmp_path / "arc-agi_evaluation_solutions.json"
    path.write_text("{}")
    with pytest.raises(PermissionError):
        env.read_bytes(path)
    with pytest.raises(PermissionError):
        env.sha256_file(path)


def test_uv_is_taken_from_its_wheel(tmp_path):
    import zipfile

    wheel = tmp_path / "uv-0.9.0-py3-none-manylinux_2_17_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("uv-0.9.0.data/scripts/uv", b"\x7fELF fake")
        archive.writestr("uv/__init__.py", b"")
    dest = env.extract_uv(wheel, tmp_path / "bin" / "uv")
    assert dest.read_bytes() == b"\x7fELF fake" and dest.stat().st_mode & 0o111


def test_a_non_312_image_fetches_python_with_uv(tmp_path):
    import yaml

    config = yaml.safe_load(CONFIG.read_text())
    config["python"]["want"] = "3.99"
    config["uv"]["fallback_path"] = str(tmp_path / "uv")
    (tmp_path / "uv").write_text("")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["python", "install"]:
            target = Path(command[command.index("--install-dir") + 1]) / "cpython-3.99" / "bin"
            target.mkdir(parents=True)
            (target / "python3.99").write_text("")
        return env.subprocess.CompletedProcess(command, 0, "3.99.0\n", "")

    report = {"steps": {}}
    ctx = env.Context(config=config, root=tmp_path, report=report, out=tmp_path / "out",
                      runner=runner)
    result = env.step_python(ctx)
    assert result["source"] == "standalone"
    assert ctx.python.endswith("cpython-3.99/bin/python3.99")
    assert calls[0][1:4] == ["python", "install", "3.12"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="the smoke builds a venv with uv")
def test_smoke_runs_every_step_and_records_failures(tmp_path):
    assert env.main(["--config", str(CONFIG), "--smoke", "--root", str(tmp_path)]) == 0
    out = tmp_path / "kaggle/working/run_logs/fn_env"
    report = json.loads((out / "report.json").read_text())
    status = {k: v["status"] for k, v in report["steps"].items()}
    assert status["listing"] == status["venv"] == status["arch"] == "ok"
    assert status["model"] == status["fetch"] == status["package"] == "ok"
    # No transformers in the smoke venv: the failure is recorded and the next steps still ran.
    assert status["template"] == "failed"
    steps = report["steps"]
    assert all(v["as_configured"] for v in steps["listing"]["inputs"].values())
    assert steps["imports"]["modules"]["fnfake"]["version"] == "0.1"
    assert not steps["imports"]["modules"]["sgl_kernel"]["ok"]
    # The guarded edit is checked, not applied, and its context is saved.
    assert status["patch"] == "ok"
    assert steps["patch"]["patches"][0]["status"] == "checked"
    assert "__version__" in (out / "patch_fake_edit_context.txt").read_text()
    assert steps["venv"]["pip_check"]["returncode"] == 0
    # http.server's --help stands in for SGLang's: the made-up flag is reported unknown.
    assert status["server_help"] == "failed"
    assert steps["server_help"]["unknown_flags"] == ["--not-a-flag"]
    assert "--directory" in (out / "server_help.txt").read_text()
    assert steps["arch"]["packages"]["fnfake"]["verdict"] == "native"
    assert steps["arch"]["packages"]["flashinfer"]["verdict"] == "not installed"
    assert steps["model"]["model"]["gib_by_component"].keys() >= {"experts", "ple"}
    assert steps["fetch"]["fetched"][0]["check"]["import"]["returncode"] == 0
    marker = json.loads((tmp_path / "kaggle/working/fn_env_output.json").read_text())
    assert list(marker["wheels"]) == ["fnextra-0.2-py3-none-any.whl"]
    assert (out / "summary.md").read_text().count("|") > 10
    assert (out / "model_chat_template.jinja.txt").is_file()
    # Every file written for the pull matches the gateway's allowlist.
    gateway = load("kaggle_ops")
    for path in out.rglob("*"):
        rel = path.relative_to(tmp_path / "kaggle/working").as_posix()
        assert gateway.pull_allowed(rel), rel


@pytest.mark.skipif(shutil.which("uv") is None, reason="the smoke builds a venv with uv")
def test_resume_keeps_the_steps_that_passed(tmp_path):
    args = ["--config", str(CONFIG), "--smoke", "--root", str(tmp_path)]
    assert env.main(args) == 0
    assert env.main([*args, "--resume"]) == 0
    report = json.loads((tmp_path / "kaggle/working/run_logs/fn_env/report.json").read_text())
    assert report["steps"]["venv"].get("resumed") is True
    assert report["steps"]["arch"].get("resumed") is True
    assert report["steps"]["template"]["status"] == "failed"


def test_the_guarded_edit_applies_once_and_refuses_another_file(tmp_path):
    target = tmp_path / "pkg" / "mod.py"
    target.parent.mkdir()
    source = "x = f(\n    scratch_capacity,\n    k_buffer.shape[1],\n    k_buffer.shape[2],\n" \
             "    k_buffer.dtype,\n    k_buffer.device,\n)\ny = k_buffer.dtype,\n"
    target.write_text(source)
    import hashlib

    import yaml

    item = dict(yaml.safe_load(CONFIG.read_text())["patches"]["items"][0])
    item["file"] = "pkg/mod.py"
    item["sha256_before"] = hashlib.sha256(source.encode()).hexdigest()
    checked = env.apply_patch(tmp_path, item, dry_run=True)
    assert checked["status"] == "checked" and target.read_text() == source
    applied = env.apply_patch(tmp_path, item)
    assert applied["status"] == "applied" and applied["matches"] == 1
    text = target.read_text()
    assert "    q.dtype,\n    k_buffer.device" in text and "y = k_buffer.dtype," in text
    assert env.apply_patch(tmp_path, item)["status"] == "sha256 differs"
    item["sha256_after"] = applied["sha256_after"]
    assert env.apply_patch(tmp_path, item)["status"] == "already applied"
