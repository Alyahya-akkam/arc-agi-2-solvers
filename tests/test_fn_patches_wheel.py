"""All fourteen hash-guarded edits of P5.2-01 (revision 20) applied in order to the wheel's files.

Version 15 (revision 19) failed at boot on Kaggle because kv4_pool_class_helper inserted a
function between ``@dataclass(slots=True, kw_only=True)`` and ``class KVCacheConfigurator``: the
decorator wrapped the helper and the module failed to import. These checks run on the real
wheel's files (``runs/wheel-peek``, read on a Kaggle CPU session): every edit applies, every
patched file compiles, and every decorated function or class keeps exactly its decorators and its
name, while the new helpers carry none. No GPU, no network.
"""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
PEEK = REPO / "runs" / "wheel-peek" / "run_logs" / "peek"
CONFIGS = ("configs/P5.2-01.yaml", "configs/P5.5-01.yaml")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("fn_probe_wheel_patches_under_test", REPO / "scripts" / "fn_probe.py")


def peek_site(tmp_path):
    if not (PEEK / "peek.json").is_file():
        pytest.skip("no wheel peek (runs/wheel-peek) in this clone")
    site = tmp_path / "site"
    for path in PEEK.glob("*.txt"):
        rel = path.name[: -len(".txt")].replace("__", "/")
        target = site / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return site


def decorated(tree):
    """Every decorated def or class, by dotted name, with its decorators' source dumps."""
    out = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                out[name] = [ast.dump(d) for d in child.decorator_list]
                walk(child, f"{name}.")

    walk(tree, "")
    return out


@pytest.mark.parametrize("config_path", CONFIGS)
def test_every_edit_applies_compiles_and_keeps_every_decorator(tmp_path, config_path):
    config = yaml.safe_load((REPO / config_path).read_text())
    items = config["patches"]["items"]
    assert len(items) == 14
    site = peek_site(tmp_path)
    originals = {}
    for item in items:
        path = site / item["file"]
        originals.setdefault(item["file"], path.read_text())
        result = probe.apply_patch(site, item)
        assert result["status"] == "applied", (item["name"], result["status"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256_after"] or (
            "sha256_after" not in item), item["name"]
    for rel, original in originals.items():
        patched = (site / rel).read_text()
        compile(patched, rel, "exec")
        before, after = decorated(ast.parse(original)), decorated(ast.parse(patched))
        for name, decorators in before.items():
            assert after.get(name) == decorators, (rel, name)
        for name in set(after) - set(before):
            assert after[name] == [], (rel, name, "a new helper carries a decorator")
    # The configurator's class keeps its dataclass decorator and is no helper's decorator.
    tree = ast.parse((site / "sglang/srt/mem_cache/kv_cache_configurator.py").read_text())
    helper = next(n for n in tree.body if getattr(n, "name", "") == "_arc26_kv4_pool_class")
    assert helper.decorator_list == []
    cls = next(n for n in tree.body if getattr(n, "name", "") == "KVCacheConfigurator")
    assert len(cls.decorator_list) == 1


def test_both_configs_carry_the_same_fourteen_edits():
    a, b = (yaml.safe_load((REPO / c).read_text())["patches"]["items"] for c in CONFIGS)
    strip = lambda items: [{k: v for k, v in i.items() if k != "fallback_env"} for i in items]  # noqa: E731
    assert strip(a) == strip(b)


def test_the_peek_is_the_wheels_files():
    if not (PEEK / "peek.json").is_file():
        pytest.skip("no wheel peek in this clone")
    hashes = json.loads((PEEK / "peek.json").read_text())["files"]
    config = yaml.safe_load((REPO / CONFIGS[0]).read_text())
    firsts = {}
    for item in config["patches"]["items"]:
        firsts.setdefault(item["file"], item["sha256_before"])
    for rel, digest in firsts.items():
        entry = hashes.get(rel)
        value = entry.get("sha256") if isinstance(entry, dict) else entry
        assert value == digest, rel
