"""Offline CPU preflight for the structured forcing grammar and the model tokenizer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from arc26.agent.toolshot import _answer_schema, _schema_answers


def _payload(count: int, rows: list[str] | None = None) -> dict[str, Any]:
    return {"answers": {str(i): [rows or ["012", "345"]] for i in range(count)}}


def _matches(xgrammar: Any, compiled: Any, tokenizer: Any, text: str, eos: int) -> bool:
    matcher = xgrammar.GrammarMatcher(compiled)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("tokenizer returned no tokens")
    for token in [*ids, eos]:
        if not matcher.accept_token(int(token)):
            return False
    return bool(matcher.is_terminated())


def check(model: str) -> dict[str, Any]:
    """Compile and exercise actual schemas using only locally available model configuration.

    Args:
        model: Local tokenizer and model configuration directory.

    Returns:
        A report with tokenizer metadata and passed cases.

    Raises:
        ValueError: When the backend, tokenizer, schema, or matcher fails a check.
    """
    import torch
    import xgrammar
    from transformers import AutoTokenizer

    version = importlib.metadata.version("xgrammar")
    if version != "0.2.1":
        raise ValueError(f"expected xgrammar 0.2.1, got {version}")
    if torch.cuda.is_initialized():
        raise ValueError("CUDA already initialized")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    model_path = Path(model)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config") or config
    vocab_size = int(text_config.get("vocab_size", config.get("vocab_size")))
    generation_path = model_path / "generation_config.json"
    generation = (json.loads(generation_path.read_text(encoding="utf-8"))
                  if generation_path.exists() else {})

    def eos_ids(value: Any) -> list[int]:
        if value is None:
            return []
        return [int(value)] if isinstance(value, int) else [int(v) for v in value]

    model_eos = sorted(set(eos_ids(config.get("eos_token_id")) +
                           eos_ids(generation.get("eos_token_id"))))
    eos = model_eos or eos_ids(getattr(tokenizer, "eos_token_id", None))
    if vocab_size < 1 or not eos or any(v < 0 or v >= vocab_size for v in eos):
        raise ValueError("invalid model vocabulary or EOS ids")
    info = xgrammar.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos or None
    )
    compiler = xgrammar.GrammarCompiler(tokenizer_info=info)
    passed: dict[str, list[str]] = {}
    compile_seconds: dict[str, float] = {}
    for count in (1, 2, 3, 4):
        started = time.perf_counter()
        compiled = compiler.compile_json_schema(
            schema=json.dumps(_answer_schema(count)), any_whitespace=True
        )
        compile_seconds[str(count)] = time.perf_counter() - started
        valid = _payload(count)
        cases: dict[str, tuple[dict[str, Any], bool]] = {"valid": (valid, True)}
        for width in (1, 22, 30):
            cases[f"width_{width}"] = (_payload(count, ["1" * width, "2" * width]), True)
        if count < 4:
            two = _payload(count)
            two["answers"]["0"].append(["7"])
            cases["two_attempts"] = (two, True)
        bad_color = _payload(count, ["1a"])
        long_row = _payload(count, ["1" * 31])
        many_rows = _payload(count, ["1"] * 31)
        missing = _payload(count)
        del missing["answers"]["0"]
        extra = _payload(count)
        extra["answers"][str(count)] = [["1"]]
        cases.update({name: (value, False) for name, value in {
            "bad_color": bad_color, "long_row": long_row, "many_rows": many_rows,
            "missing_index": missing, "additional_index": extra,
        }.items()})
        ragged = _payload(count, ["12", "3"])
        cases["ragged_grammar"] = (ragged, False)
        for name, (value, expected) in cases.items():
            text = json.dumps(value, separators=(",", ":"))
            for eos_id in eos:
                if _matches(xgrammar, compiled, tokenizer, text, eos_id) != expected:
                    raise ValueError(f"schema {count} case {name} failed for EOS {eos_id}")
            if name == "valid" or name == "two_attempts":
                _schema_answers(text, count, "stop", ())
            if name == "ragged_grammar":
                try:
                    _schema_answers(text, count, "stop", ())
                except ValueError:
                    pass
                else:
                    raise ValueError("ragged grid passed postparse validation")
        passed[str(count)] = list(cases)
    if torch.cuda.is_initialized():
        raise ValueError("CUDA initialized during grammar check")
    return {"ok": True, "xgrammar_version": version, "model": str(model),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name": str(getattr(tokenizer, "name_or_path", "")),
            "vocab_size": vocab_size, "model_eos_token_ids": model_eos,
            "eos_token_ids": eos, "schemas": passed,
            "compile_seconds": compile_seconds, "cuda_initialized": False}


def main() -> int:
    """Run the offline preflight and write a JSON report; return nonzero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        report = check(args.model)
    except Exception as err:
        report = {"ok": False, "model": args.model,
                  "error": f"{type(err).__name__}: {err}"}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
