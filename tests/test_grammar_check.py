"""Offline preflight exercises backend API and fails closed on matcher errors."""

import json
import re
import sys
from types import SimpleNamespace

import pytest

from arc26.agent import grammar_check


def _modules(monkeypatch, *, version="0.2.1", broken=False, cuda=False, tokenizer_eos=999):
    calls = []

    class Tokenizer:
        name_or_path = "local-tokenizer"
        eos_token_id = tokenizer_eos

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [ord(c) for c in text]

    class TokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer, *, vocab_size, stop_token_ids):
            calls.append((tokenizer.name_or_path, vocab_size, stop_token_ids))
            return "info"

    class Compiler:
        def __init__(self, *, tokenizer_info):
            assert tokenizer_info == "info"

        def compile_json_schema(self, *, schema, any_whitespace):
            assert any_whitespace is True
            return json.loads(schema)

    class Matcher:
        def __init__(self, schema):
            self.schema = schema
            self.text = ""
            self.terminated = False

        def accept_token(self, token):
            if token in (998, 999):
                if broken:
                    return False
                obj = json.loads(self.text)
                props = self.schema["properties"]["answers"]
                answers = obj.get("answers", {})
                valid = set(obj) == {"answers"} and set(answers) == set(props["required"])
                for key, grids in answers.items():
                    if key not in props["properties"]:
                        valid = False
                        continue
                    bound = props["properties"][key]["maxItems"]
                    valid &= 1 <= len(grids) <= bound
                    for rows in grids:
                        valid &= 1 <= len(rows) <= 30
                        valid &= all(re.fullmatch(r"[0-9]{1,30}", row) for row in rows)
                        valid &= len({len(row) for row in rows}) == 1
                self.terminated = bool(valid)
                return self.terminated
            self.text += chr(token)
            return True

        def is_terminated(self):
            return self.terminated

    tokenizer = Tokenizer()
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path, **kw: tokenizer
                                      if kw == {"local_files_only": True,
                                                "trust_remote_code": True} else None),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: cuda)))
    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace(
        TokenizerInfo=TokenizerInfo, GrammarCompiler=Compiler, GrammarMatcher=Matcher))
    monkeypatch.setattr(grammar_check.importlib.metadata, "version", lambda package: version)
    return calls


def test_preflight_exercises_all_schemas_and_ragged_postcheck(monkeypatch, tmp_path):
    calls = _modules(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 1000, "eos_token_id": 999}))
    result = grammar_check.check(str(tmp_path))
    assert result["ok"] and result["xgrammar_version"] == "0.2.1"
    assert calls == [("local-tokenizer", 1000, [999])]
    assert set(result["schemas"]) == {"1", "2", "3", "4"}
    assert all(seconds >= 0 for seconds in result["compile_seconds"].values())
    assert "ragged_grammar" in result["schemas"]["4"]
    assert all(f"width_{width}" in result["schemas"]["4"] for width in (1, 22, 30))
    assert "two_attempts" in result["schemas"]["2"]
    assert result["cuda_initialized"] is False


def test_cli_writes_failure_on_matcher_error(monkeypatch, tmp_path):
    _modules(monkeypatch, broken=True)
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 1000, "eos_token_id": 999}))
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["check", "--model", str(tmp_path), "--report", str(report)])
    assert grammar_check.main() == 1
    result = json.loads(report.read_text())
    assert result["ok"] is False and "case valid" in result["error"]


def test_cli_fails_closed_on_version_and_cuda(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["check", "--model", "/local/model", "--report", str(report)])
    for kwargs, expected in [({"version": "0.3.0"}, "0.3.0"), ({"cuda": True}, "CUDA")]:
        _modules(monkeypatch, **kwargs)
        assert grammar_check.main() == 1
        assert expected in json.loads(report.read_text())["error"]


def test_nested_vocab_and_model_generation_eos_union(monkeypatch, tmp_path):
    calls = _modules(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({
        "vocab_size": 5000, "text_config": {"vocab_size": 1000}, "eos_token_id": 998,
    }))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [999, 998]}))
    result = grammar_check.check(str(tmp_path))
    assert calls == [("local-tokenizer", 1000, [998, 999])]
    assert result["model_eos_token_ids"] == result["eos_token_ids"] == [998, 999]


def test_absent_model_eos_uses_tokenizer_for_probe(monkeypatch, tmp_path):
    calls = _modules(monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 1000}))
    result = grammar_check.check(str(tmp_path))
    assert calls == [("local-tokenizer", 1000, None)]
    assert result["model_eos_token_ids"] == [] and result["eos_token_ids"] == [999]
    _modules(monkeypatch, tokenizer_eos=None)
    with pytest.raises(ValueError, match="EOS"):
        grammar_check.check(str(tmp_path))
