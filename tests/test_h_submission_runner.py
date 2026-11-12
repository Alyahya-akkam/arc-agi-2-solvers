"""Check explicit H routing while preserving the original solver scope."""

import json

import pytest

from arc26.runner import LORA_TARGETS, RunConfig, solver_command


def test_h_is_explicit_in_competition_command_and_keeps_v1_scope():
    cfg = RunConfig(data_dir="/data", model_dir="/model", hyper_checkpoint="/input/best.pt",
                    hyper_sha256="a" * 64, hyper_log_dir="/logs/h")
    cmd = solver_command(cfg, 1234, "python")
    assert cmd[2] == "arc26.h_submission"
    payload = json.loads(cmd[6])
    assert payload["hyper_checkpoint"] == "/input/best.pt"
    assert payload["hyper_sha256"] == "a" * 64
    assert payload["hyper_log_dir"] == "/logs/h"
    assert payload["lora_r"] == 256
    assert payload["lora_targets"] == list(LORA_TARGETS)
    assert payload["subset"] is None


@pytest.mark.parametrize("change", [
    {"solver": "nvarc"}, {"hyper_sha256": "bad"}, {"hyper_log_dir": ""},
    {"hyper_log_dir": "/kaggle/inference_outputs/logs"},
])
def test_h_rejects_incompatible_or_unpinned_settings(change):
    options = dict(data_dir="/data", model_dir="/model", hyper_checkpoint="/input/best.pt",
                   hyper_sha256="a" * 64, hyper_log_dir="/logs/h")
    with pytest.raises(ValueError):
        RunConfig(**{**options, **change})


def test_h_fields_cannot_silently_be_ignored():
    with pytest.raises(ValueError):
        RunConfig(data_dir="/data", model_dir="/model", hyper_sha256="a" * 64)
