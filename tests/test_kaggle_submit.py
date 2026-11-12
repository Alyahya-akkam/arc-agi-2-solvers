"""Owner-authorized pinned code submissions through the gateway; no Kaggle calls are made."""

import argparse
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_ops.py"
SPEC = importlib.util.spec_from_file_location("kaggle_submit_gateway", SOURCE)
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)

COMPETITION = "arc-prize-2026-arc-agi-2"
KERNEL = "tester/arc26-h-submission"


@pytest.fixture
def gateway(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "KAGGLE_OWNER", "tester")
    monkeypatch.setattr(ops, "_kaggle", lambda *args: calls.append(list(args)) or 0)
    return calls


def submit_args(**changes):
    values = {
        "--competition": COMPETITION,
        "--kernel": KERNEL,
        "--version": "1",
        "--file": "submission.json",
        "--message": "version 1 plus H, exact reviewed build",
    }
    values.update(changes)
    return ["competition-submit", *(value for pair in values.items() for value in pair
                                    if values[pair[0]] is not None)]


def run(argv):
    try:
        return ops.main(argv)
    except SystemExit as error:
        return error.code


def test_submit_pins_notebook_version_and_output_in_code_competition_mode(gateway):
    assert run(submit_args()) == 0
    assert gateway == [[
        "competitions", "submit", COMPETITION, "-k", KERNEL, "-v", "1",
        "-f", "submission.json", "--message=version 1 plus H, exact reviewed build",
    ]]


@pytest.mark.parametrize("missing", ["--competition", "--kernel", "--version", "--file",
                                     "--message"])
def test_submit_has_no_implicit_competition_latest_version_or_local_file(gateway, missing):
    assert run(submit_args(**{missing: None})) == 2
    assert gateway == []


@pytest.mark.parametrize("competition", ["other-competition", "", COMPETITION + "/", "."])
def test_submit_refuses_other_competitions(gateway, competition):
    assert run(submit_args(**{"--competition": competition})) == 2
    assert gateway == []


@pytest.mark.parametrize("kernel", ["foreign/notebook", "tester", "tester/../x", "tester/..",
                                    "tester/notebook/1", "tester/", "12345"])
def test_submit_refuses_foreign_or_ambiguous_notebook(gateway, kernel):
    assert run(submit_args(**{"--kernel": kernel})) == 2
    assert gateway == []


def test_submit_refuses_empty_owner(gateway, monkeypatch):
    monkeypatch.setattr(ops, "KAGGLE_OWNER", "")
    assert run(submit_args()) == 2
    assert gateway == []


@pytest.mark.parametrize("version", ["0", "-1", "latest", "1.0", "True"])
def test_submit_refuses_nonpositive_or_unpinned_version(gateway, version):
    assert run(submit_args(**{"--version": version})) == 2
    assert gateway == []


@pytest.mark.parametrize("filename", ["/tmp/submission.json", "../submission.json", "weights.pt",
                                      "run_logs/submission.json"])
def test_submit_refuses_a_local_upload_or_another_output(gateway, filename):
    assert run(submit_args(**{"--file": filename})) == 2
    assert gateway == []


@pytest.mark.parametrize("message", ["", " \n ", "description\x00tail"])
def test_submit_refuses_empty_or_invalid_message(gateway, message):
    assert run(submit_args(**{"--message": message})) == 2
    assert gateway == []


def test_submit_sends_description_as_one_argument_even_if_it_looks_like_a_flag(gateway):
    argv = submit_args(**{"--message": None}) + ["--message=--sandbox"]
    assert run(argv) == 0
    assert gateway[0][-1] == "--message=--sandbox"
    assert "--sandbox" not in gateway[0]


def test_submit_propagates_cli_failure_without_retry(gateway, monkeypatch):
    def failed(*args):
        gateway.append(list(args))
        return 7

    monkeypatch.setattr(ops, "_kaggle", failed)
    assert run(submit_args()) == 7
    assert len(gateway) == 1


def test_direct_submit_entrypoint_also_checks_version_and_output(gateway):
    args = argparse.Namespace(competition=COMPETITION, kernel=KERNEL, version=True,
                              file_name="submission.json", message="exact build")
    with pytest.raises(ops.GatewayError, match="positive"):
        ops.cmd_competition_submit(args)
    args.version, args.file_name = 1, "/tmp/submission.json"
    with pytest.raises(ops.GatewayError, match="exactly submission.json"):
        ops.cmd_competition_submit(args)
    assert gateway == []


def test_submission_history_is_read_only_allowlisted_json_with_explicit_paging(gateway):
    assert run(["competition-submissions", "--competition", COMPETITION,
                "--page-size", "50", "--page-token", "next-page"]) == 0
    assert gateway == [["competitions", "submissions", COMPETITION, "--format", "json",
                        "--page-size", "50", "--page-token", "next-page"]]


def test_submission_history_does_not_use_cli_default_competition(gateway):
    assert run(["competition-submissions"]) == 2
    assert gateway == []


def test_submission_history_refuses_other_competition(gateway):
    assert run(["competition-submissions", "--competition", "other"]) == 2
    assert gateway == []


@pytest.mark.parametrize("size", ["0", "201"])
def test_submission_history_refuses_invalid_page_size(gateway, size):
    assert run(["competition-submissions", "--competition", COMPETITION,
                "--page-size", size]) == 2
    assert gateway == []
