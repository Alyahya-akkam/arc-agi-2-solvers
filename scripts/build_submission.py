r"""Build the upload-ready folder of one Kaggle submission notebook (BRIEF A5, B5 P0.5).

Purpose: turn one commit and one submission config into everything Kaggle needs, built locally so
each milestone can be rebuilt from its commit (A5) and pushed only through
``scripts/kaggle_ops.py`` (ADR-008). Task P0.5; packaging decision ADR-010; runbook
``docs/runbooks/build_submission.md``; steps for the human ``docs/SUBMITTING.md``.

Inputs: ``--tag`` (the name of the output folder, for example the run id) and ``--config`` (a
``configs/submission_*.yaml`` inside the repository), plus the repository at ``HEAD``, whose
working tree must be clean. Outputs, in ``dist/<tag>/``:

- ``kaggle_submission.ipynb``: the notebook, from ``scripts/build_kaggle_notebook.py``
  (``kaggle_script.ipynb`` for a script config, ``kind: script``, task P0.5d).
- ``kernel-metadata.json``: private, internet off, the reference's Python 3.11 image pinned
  (``original``), the install notebook at its pinned version, the competition, the base model,
  and the code dataset; ``enable_gpu`` and ``machine_shape`` from the config's accelerator; a
  hybrid build (task P6.5) also attaches the two download notebooks' outputs at their pinned
  versions (the agent's weights and engine wheelhouse).
- ``arc26-code/``: the ``arc26`` wheel and ``dataset-metadata.json``, the private code dataset;
  a script build adds its script and the script's config, byte for byte from the commit.
- ``MANIFEST.md``: the commit, every file with its sha256, what goes where on Kaggle, and a
  summary of the config.

Method: the tree must be clean, so the build is exactly ``HEAD``. The commit is exported with
``git archive`` into a temporary folder, so no uncommitted or ignored file can reach the wheel,
and the config, a round arm's governor config (task P2.3), a hybrid build's agent run config
(task P6.5), and the reference metadata are read from that export. The wheel is built from the
export with ``SOURCE_DATE_EPOCH`` set to the time of the last commit that changed one of its
inputs (``WHEEL_INPUTS``), so a rebuild gives the same bytes, and so does a build after a commit
that changed only docs, configs, or scripts: the code dataset then needs no new version. The
builders are ``uv build --wheel --offline`` first, then online, then ``pip wheel``. Every file
is produced in memory and checked (the gateway's own metadata checks included) before the first
write; the folder is then written beside ``dist/<tag>/`` and swapped in, so a refused build
writes nothing and no stale file survives a rebuild.

Usage::

    python scripts/build_submission.py --tag P0.5-01 --config configs/submission_P0.5-01.yaml
    python scripts/build_submission.py --tag P0.5-cpu --config configs/submission_cpu_check.yaml
    python scripts/build_submission.py --tag script-smoke --config configs/script_smoke.yaml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent


def _load(name: str) -> ModuleType:
    # Sibling scripts are loaded by path, as the tests load them: scripts/ is not a package.
    spec = importlib.util.spec_from_file_location(f"_arc26_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


notebook_builder = _load("build_kaggle_notebook")
reference_run = notebook_builder.reference_run
gateway = notebook_builder.gateway
BuildError = notebook_builder.BuildError

NOTEBOOK_FILE = notebook_builder.NOTEBOOK_FILE
METADATA_FILE = "kernel-metadata.json"
CODE_DIR = "arc26-code"
DATASET_METADATA_FILE = "dataset-metadata.json"
MANIFEST_FILE = "MANIFEST.md"
# kaggle-cli 2.2.4 needs exactly one license. The wheel mixes Apache-2.0 code moved from the
# reference (arc26/nvarc) with project code whose license G5 decides; "other" says so.
DATASET_LICENSE = "other"
# Hard stop suggested for a CPU push: the CPU check finishes in minutes.
CPU_PUSH_TIMEOUT_S = 1800
# A local wheel build takes seconds; this only guards against a hung builder.
WHEEL_BUILD_TIMEOUT_S = 600
# Every path whose content reaches the wheel: the package, its metadata, the readme that the
# wheel's METADATA embeds, and the ignore file hatchling applies when it selects files.
WHEEL_INPUTS = ("arc26", "pyproject.toml", "README.md", ".gitignore")
# MANIFEST.md's line for each solver code of notebook_builder.SOLVERS (task P2.0): the human who
# submits reads it beside "fit to submit"; submission_blocker does not read the solver, since
# ADR-011 lets either submit.
SOLVER_NOTES = {
    "nvarc": "`nvarc`: the moved reference, `python -m arc26.nvarc.starter` (the parity anchor)",
    "split": (
        "`split`: the reference's statements split into `arc26.ttt` and `arc26.decode`, "
        "`python -m arc26.solver` (ADR-011, Accepted: run `P2.0-01` passed; the default)"
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        detail = getattr(err, "stderr", "") or err
        raise BuildError(f"git {' '.join(args)} failed in {repo}: {detail}") from err


def require_clean(repo: Path) -> None:
    """Refuse to build from a working tree with changes or untracked files.

    Why: the build must be exactly one commit, so its ``MANIFEST.md`` names everything that went
    in and a rebuild from that commit reproduces it.

    Args:
        repo: The repository root.

    Raises:
        BuildError: If ``git status`` lists anything, or git fails.
    """
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise BuildError(
            "the working tree is not clean; commit or remove these first:\n" + status.rstrip()
        )


def head_commit(repo: Path) -> tuple[str, int]:
    """Return the full SHA of ``HEAD`` and its commit time.

    Args:
        repo: The repository root.

    Returns:
        The SHA and the commit time in seconds since the epoch.

    Raises:
        BuildError: If git fails.
    """
    commit = _git(repo, "rev-parse", "HEAD").strip()
    return commit, int(_git(repo, "log", "-1", "--format=%ct", commit).strip())


def wheel_input_commit(repo: Path, commit: str) -> tuple[str, int]:
    """Return the last commit up to ``commit`` that changed a ``WHEEL_INPUTS`` path, and its time.

    Why: the wheel's timestamps come from ``SOURCE_DATE_EPOCH``. Taking it from this commit
    instead of ``HEAD`` keeps the wheel's bytes, and so its sha256 and the code dataset version,
    unchanged across commits that touch nothing the wheel holds, such as a pre-run brief.

    Args:
        repo: The repository root.
        commit: The commit being built.

    Returns:
        That commit's full SHA and its commit time in seconds since the epoch.

    Raises:
        BuildError: If git fails or no commit touches the inputs.
    """
    found = _git(repo, "log", "-1", "--format=%H %ct", commit, "--", *WHEEL_INPUTS).split()
    if len(found) != 2:
        raise BuildError(f"no commit up to {commit[:12]} touches {', '.join(WHEEL_INPUTS)}")
    return found[0], int(found[1])


def export_tree(repo: Path, commit: str, dest: Path) -> None:
    """Extract ``commit`` with ``git archive`` into ``dest``: tracked files only, as committed.

    Args:
        repo: The repository root.
        commit: The commit to export.
        dest: Empty folder to extract into.

    Raises:
        BuildError: If git fails or the archive cannot be extracted.
    """
    try:
        data = subprocess.run(
            ["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(dest, filter="data")
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as err:
        raise BuildError(f"cannot export {commit} with git archive: {err}") from err


def build_wheel(source: Path, out: Path, epoch: int) -> tuple[Path, str]:
    """Build the ``arc26`` wheel from ``source`` into ``out``.

    Why: ``SOURCE_DATE_EPOCH`` fixes the timestamps in the wheel (hatchling honors it), so the
    same source gives the same bytes. ``uv build`` offline reuses the build backend from uv's
    cache; online and ``pip wheel`` (which downloads the backend) are fallbacks for a cold cache
    or a machine without uv, such as CI.

    Args:
        source: A folder holding ``pyproject.toml`` and ``arc26/``.
        out: Empty folder for the build output.
        epoch: Value of ``SOURCE_DATE_EPOCH``.

    Returns:
        The wheel's path and the command that built it.

    Raises:
        BuildError: If no builder is available or every attempt fails.
    """
    attempts = []
    uv = shutil.which("uv")
    if uv is not None:
        base = [uv, "build", "--wheel", "--out-dir", str(out), str(source)]
        attempts.append(("uv build --wheel --offline", [*base, "--offline"]))
        attempts.append(("uv build --wheel", base))
    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(out)]
        attempts.append(("pip wheel --no-deps", [*command, str(source)]))
    if not attempts:
        raise BuildError("no wheel builder: install uv, or run with a Python that has pip")
    env = {**os.environ, "SOURCE_DATE_EPOCH": str(epoch)}
    failures = []
    for label, command in attempts:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                timeout=WHEEL_BUILD_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as err:
            failures.append(f"{label}: {err}")
            continue
        wheels = sorted(out.glob("arc26-*.whl"))
        if result.returncode == 0 and len(wheels) == 1:
            return wheels[0], label
        failures.append(f"{label}: exit {result.returncode}: {result.stderr.strip()[-600:]}")
        for wheel in wheels:
            wheel.unlink()
    raise BuildError("the wheel build failed:\n" + "\n".join(failures))


def wheel_generator(wheel: Path) -> str:
    """Return the build backend named in the wheel's ``WHEEL`` file, such as ``hatchling 1.32.4``.

    Why: a rebuild is byte-identical only with the same build backend, so ``MANIFEST.md`` names it.

    Args:
        wheel: The built wheel.

    Returns:
        The generator, or ``unknown`` when the wheel does not name one.
    """
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith(".dist-info/WHEEL"):
                for line in archive.read(name).decode("utf-8").splitlines():
                    if line.startswith("Generator:"):
                        return line.partition(":")[2].strip()
    return "unknown"


def kernel_metadata(
    config: notebook_builder.BuildConfig, pulled: dict[str, Any]
) -> dict[str, Any]:
    """Write the notebook's ``kernel-metadata.json`` from the config and the reference metadata.

    Why: the stack must be run A's (ADR-010), so the pinned image, the model source, and the
    install notebook come through ``build_reference_run.kernel_metadata``, which passed the
    gateway for run A and refuses a pulled file without them. Identity, accelerator, and the code
    dataset come from the config; ``id_no`` and the reference's title are never copied. A script
    config's ``notebook_inputs`` follow the reference's kernel sources, each pinned to its version
    (task P3.5), as do an adapter build's (task P3.7), and so do a hybrid build's two download
    outputs (task P6.5, the agent's weights and wheelhouse); an adapter build's ``dataset_inputs``
    follow the code dataset, each as ``<owner>/<slug>/<version>`` (task P3.9b); any other
    submission build attaches exactly what it did before.

    Args:
        config: The checked ``SubmissionConfig`` or ``ScriptConfig``.
        pulled: The reference notebook's pulled ``kernel-metadata.json``.

    Returns:
        The metadata, in the key order it is written.

    Raises:
        BuildError: If the pulled metadata is refused, a notebook input is one of the reference's
            kernel sources, or the result fails the gateway's checks.
    """
    owner, slug = config.kaggle_notebook.split("/")
    try:
        base = reference_run.kernel_metadata(
            pulled, owner, config.competition, config.kernel_source_versions
        )
    except reference_run.BuildError as err:
        raise BuildError(f"{config.reference_metadata}: {err}") from err
    kernel_sources = list(base["kernel_sources"])
    inputs = config.notebook_inputs
    if config.kind == "submission" and config.agent is not None:
        inputs = config.agent.inputs()
    attached = sorted(set(inputs) & {source.rpartition("/")[0] for source in kernel_sources})
    if attached:
        raise BuildError(
            f"notebook_inputs {attached} are kernel sources of {config.reference_metadata}: pin "
            "those in kernel_source_versions"
        )
    kernel_sources += [f"{name}/{version}" for name, version in inputs.items()]
    datasets = [config.code_dataset]
    if config.kind == "submission":
        datasets += [f"{name}/{version}" for name, version in config.dataset_inputs.items()]
    meta: dict[str, Any] = {
        "id": config.kaggle_notebook,
        "title": slug.replace("-", " "),
        "code_file": notebook_builder.notebook_file(config),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": config.accelerator is not None,
        "enable_internet": False,
    }
    if config.accelerator is not None:
        meta["machine_shape"] = config.accelerator
    meta.update(
        {
            "docker_image_pinning_type": "original",
            "docker_image": base["docker_image"],
            "competition_sources": base["competition_sources"],
            "model_sources": base["model_sources"],
            "dataset_sources": datasets,
            "kernel_sources": kernel_sources,
        }
    )
    try:
        accelerated = gateway.uses_gpu(meta, None)
        gateway.check_kernel_metadata(meta, accelerated)
        gateway.check_flags(meta, gateway.accelerator_id(meta, None))
    except gateway.GatewayError as err:
        raise BuildError(f"kernel-metadata.json would be refused by the gateway: {err}") from err
    return meta


def dataset_metadata(code_dataset: str, script: str | None = None) -> dict[str, Any]:
    """Return ``dataset-metadata.json`` of the private code dataset.

    Why: kaggle-cli 2.2.4 needs a title of 6 to 50 characters and exactly one license; the
    dataset stays private because the gateway never passes ``--public``.

    Args:
        code_dataset: ``<owner>/<slug>`` from the config.
        script: A script build's script, as the config names it; None for a submission build,
            whose metadata is as before task P0.5d.

    Returns:
        The metadata.

    Raises:
        BuildError: If the gateway's owner check refuses it.
    """
    description = (
        "The arc26 package as one wheel, for the arc26 submission notebooks. Built by "
        "scripts/build_submission.py from a clean commit; the build's MANIFEST.md names the "
        "commit and the wheel's sha256, which the notebook checks before installing."
    )
    if script is not None:
        description = (
            f"The arc26 package as one wheel, with {script} and its config, for an arc26 script "
            "notebook. Built by scripts/build_submission.py from a clean commit; the build's "
            "MANIFEST.md names the commit and every file's sha256, which the notebook checks "
            "before use."
        )
    meta = {
        "title": code_dataset.split("/")[1].replace("-", " "),
        "id": code_dataset,
        "licenses": [{"name": DATASET_LICENSE}],
        "description": description,
    }
    try:
        gateway.check_owner(meta, DATASET_METADATA_FILE)
    except gateway.GatewayError as err:
        raise BuildError(str(err)) from err
    return meta


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def render_manifest(
    *,
    tag: str,
    build: dict[str, str],
    commit_time: int,
    wheel_commit: str,
    wheel_epoch: int,
    config: notebook_builder.BuildConfig,
    builder: str,
    generator: str,
    metadata: dict[str, Any],
    files: dict[str, bytes],
) -> bytes:
    """Render ``MANIFEST.md``: provenance, file hashes, where each file goes, config summary.

    Args:
        tag: The build tag.
        build: The notebook's provenance (``build_info``).
        commit_time: The commit time of the commit built.
        wheel_commit: The last commit that changed the wheel's inputs (``wheel_input_commit``).
        wheel_epoch: Its commit time, the wheel's ``SOURCE_DATE_EPOCH``.
        config: The checked ``SubmissionConfig`` or ``ScriptConfig``.
        builder: The command that built the wheel.
        generator: The wheel's build backend (``wheel_generator``).
        metadata: The notebook's ``kernel-metadata.json``.
        files: Every other file of the folder, by path relative to it.

    Returns:
        The file's bytes. Nothing depends on the time of the build, so rebuilds are identical.
    """
    commit = build["commit"]
    date = datetime.datetime.fromtimestamp(commit_time, tz=datetime.UTC)
    date = date.strftime("%Y-%m-%d %H:%M:%S")
    code = f"dist/{tag}/{CODE_DIR}"
    script = config.kind == "script"
    agent = None if script else config.agent
    # A script build's session must outlast its hard stop by the time the logs take; a hybrid
    # build's (task P6.5) its agent's SIGKILL, the kill wait, and the final check and logs.
    timeout = None
    if script:
        timeout = config.script_kill_s + notebook_builder.WRITE_RESERVE_S
    elif agent is not None:
        timeout = (
            agent.end_s
            + agent.term_s
            + notebook_builder.AGENT_KILL_WAIT_S
            + notebook_builder.WRITE_RESERVE_S
        )
    if config.accelerator is None:
        hardware = "CPU"
        push = (
            "`python scripts/kaggle_ops.py push dist/"
            f"{tag} --timeout {max(CPU_PUSH_TIMEOUT_S, timeout or 0)}` (CPU: no pre-run brief)"
        )
    else:
        hardware = f"`{config.accelerator}`"
        push = (
            "only after its pre-run brief is approved: "
            f"`python scripts/kaggle_ops.py push dist/{tag} --run-id <run-id>`"
        )
        if timeout is not None:
            push += f", with the brief's `Timeout` at least {timeout} s"
    notebook = notebook_builder.notebook_file(config)
    inputs = config.notebook_inputs
    reader = "the script" if script else "the adapter arms"
    if script:
        outputs = "`run_logs/`"
        dataset = "the wheel, the script, and its config above; the notebook checks every sha256"
        summary = _script_summary(config, hardware)
    else:
        outputs = "`submission.json`, `run_logs/`"
        dataset = "the wheel above; the notebook checks its sha256"
        summary = _submission_summary(config, hardware)
    lines = [
        f"# Build `{tag}`",
        "",
        f"Written by `scripts/build_submission.py` from commit `{commit}` ({date} UTC) with a "
        "clean working tree. Rebuilding that commit with the same wheel builder gives "
        "byte-identical files:",
        "",
        "```bash",
        f"python scripts/build_submission.py --tag {tag} --config {build['config']}",
        "```",
        "",
        "## Build",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Commit | `{commit}` |",
        f"| Config | `{build['config']}`, sha256 `{build['config_sha256']}` |",
        f"| Wheel | `{build['wheel']}` from `git archive`, built by `{builder}` with "
        f"`{generator}`, `SOURCE_DATE_EPOCH={wheel_epoch}`, the time of `{wheel_commit[:12]}`, "
        f"the last commit that changed {', '.join(f'`{p}`' for p in WHEEL_INPUTS)} |",
        f"| Notebook | `{notebook}`, generated by `scripts/build_kaggle_notebook.py` |",
        "",
        "## Files",
        "",
        "| File | Bytes | SHA-256 |",
        "|---|---|---|",
        *(
            f"| `{name}` | {len(data)} | `{_sha256(data)}` |"
            for name, data in sorted(files.items())
        ),
        "",
        f"`{MANIFEST_FILE}` itself is not listed.",
        "",
        "## What goes where on Kaggle",
        "",
        "| Local | On Kaggle | Command, from the repository root |",
        "|---|---|---|",
        f"| `{CODE_DIR}/` | private dataset `{config.code_dataset}` | first build: "
        f"`python scripts/kaggle_ops.py dataset-create {code}`; later builds: "
        f'`python scripts/kaggle_ops.py dataset-version {code} -m "{tag} {commit[:12]}"` |',
        f"| `{notebook}`, `{METADATA_FILE}` | private notebook `{config.kaggle_notebook}`, "
        f"{hardware}, internet off | {push} |",
        f"| outputs | {outputs} into `runs/<task-id>/` | "
        f"`python scripts/kaggle_ops.py pull {config.kaggle_notebook} --task-id <task-id>` |",
        "",
        f"`{METADATA_FILE}` attaches:",
        "",
        *(f"- competition `{name}` (the data);" for name in metadata["competition_sources"]),
        *(f"- model `{name}`;" for name in metadata["model_sources"]),
        *(
            _kernel_source_line(name, inputs, agent, reader)
            for name in metadata["kernel_sources"]
        ),
        *(
            f"- dataset `{name}` ({dataset});"
            if name == config.code_dataset
            else f"- dataset `{name}`, passed to {reader} as `{{input:{name.rpartition('/')[0]}}}`;"
            for name in metadata["dataset_sources"]
        ),
        f"- image `{metadata['docker_image']}`, pinned (`original`).",
        "",
        "## Config summary",
        "",
        "| Key | Value |",
        "|---|---|",
        *summary,
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _kernel_source_line(
    name: str,
    inputs: dict[str, int],
    agent: notebook_builder.AgentSettings | None,
    reader: str,
) -> str:
    # MANIFEST.md's line for one kernel source: a script build's or an adapter build's notebook
    # input (tasks P3.5, P3.7; reader names which), a hybrid build's download output (task P6.5),
    # or the reference's install notebook.
    notebook = name.rpartition("/")[0]
    if notebook in inputs:
        return f"- notebook output `{name}`, passed to {reader} as `{{input:{notebook}}}`;"
    if agent is not None and notebook == agent.model.notebook:
        return (
            f"- notebook output `{name}`, the agent's weights (`{agent.model.folder}`, revision "
            f"`{agent.model.revision[:12]}`; task P6.5);"
        )
    if agent is not None and notebook == agent.wheels.notebook:
        return (
            f"- notebook output `{name}`, the agent's engine wheelhouse (`{agent.wheels.folder}`, "
            f"vLLM {agent.wheels.vllm}; task P6.5);"
        )
    return f"- install notebook `{name}` (the package stack, ADR-010);"


def _submission_summary(config: notebook_builder.SubmissionConfig, hardware: str) -> list[str]:
    # MANIFEST.md's config rows of a submission build; a scoped build (task P2.6) adds its scope
    # row beside the solver's, both read by a competition rerun too, and a hybrid build (task
    # P6.5) its agent rows.
    subset = "every task" if config.subset is None else ", ".join(config.subset)
    if config.dev240:
        subset = (
            "DEV240: every evaluation task and a copy of each under the first 8 hex digits of the "
            "sha256 of its id, scored against the original's solutions (task P6.5b)"
        )
    blocker = notebook_builder.submission_blocker(config)
    verdict = "yes" if blocker is None else f"no, not for submission: {blocker}"
    arms = "none (one run, the reference's LoRA settings)"
    scope = []
    if config.scope is not None:
        arms = "none (one run, the scope above)"
        scope = [
            f"| scope (both modes) | `SCOPE`, {notebook_builder.scope_text(config.scope)}; task "
            "P2.6, ADR-018 |"
        ]
    if config.arms is not None:
        arms = "; ".join(
            f"`{arm.name}` (r {arm.lora_r}: {', '.join(arm.lora_targets)}"
            + (
                ""
                if arm.holdout is None
                else f"; {notebook_builder.holdout_text(arm.holdout._asdict())}"
            )
            + ("" if arm.rounds is None else f"; {_rounds_summary(arm.rounds)}")
            + ")"
            for arm in config.arms
        )
    agent = []
    if config.agent is not None:
        a = config.agent
        agent = [
            f"| agent (both modes) | `AGENT`, {'enabled' if a.enabled else 'disabled'}: "
            f"{notebook_builder.agent_text(a)}; the stage starts with at least {a.min_stage_s} s "
            f"left before the generation stop; install timeout {a.install_timeout_s} s; task "
            "P6.5, ADR-020 |",
            f"| agent run config | `{a.config}`, sha256 `{a.config_sha256}`, embedded in the "
            "agent check cell |",
            f"| agent inputs | weights `{a.model.notebook}` version {a.model.version} "
            f"(`{a.model.folder}`, revision `{a.model.revision}`); wheelhouse "
            f"`{a.wheels.notebook}` version {a.wheels.version} (`{a.wheels.folder}`, vLLM "
            f"{a.wheels.vllm}) |",
            f"| agent engine | a venv made by `{a.base_python}` in `{a.scratch_dir}`; NVIDIA "
            f"driver {a.min_driver} or later |",
        ]
        if a.shot and a.skip_nvarc:
            agent.append(
                f"| agent entry | `python -m {a.entry}`, the long shot (task P6.5b; the arguments "
                "and exit codes of `arc26.agent.run`) on every task: LLM first (task P5.1), "
                "NVARC not run when the agent check passes |"
            )
        elif a.shot:
            agent.append(
                f"| agent entry | `python -m {a.entry}`, the long shot (task P6.5b; the arguments "
                "and exit codes of `arc26.agent.run`): NVARC keeps its deadlines, the shot takes "
                "the time it leaves |"
            )
        if a.tail is not None:
            agent.append(
                f"| agent tail | `python -m {notebook_builder.TAIL_MODULE}` after the shot, the "
                "reference's LoRA settings on rank 16's confidence-0 tasks, cheapest first, "
                "filling only placeholders (task P4.7b): "
                f"{notebook_builder.tail_text(a.tail)} |"
            )
    return [
        f"| fit to submit | {verdict} |",
        f"| accelerator | {hardware} |",
        f"| run_solver | {config.run_solver} |",
        f"| solver (both modes) | {SOLVER_NOTES[config.solver]} |",
        *scope,
        *agent,
        f"| subset (outside a competition rerun) | {subset} |",
        f"| workers | {config.workers} |",
        f"| hash_seed | {config.hash_seed} |",
        f"| solver_end_s (soft deadline) | {config.solver_end_s} |",
        f"| solver_kill_s (hard stop) | {config.solver_kill_s} |",
        f"| arms (series, development runs only) | {arms} |",
        *_adapters_summary(config),
        f"| model_slug | `{config.model_slug}` |",
    ]


def _adapters_summary(config: notebook_builder.SubmissionConfig) -> list[str]:
    # MANIFEST.md's rows of an adapter build (task P3.7): the arms, their settings, the H config
    # the notebook embeds (path and sha256 in the built commit), and the attached outputs. Every
    # other build's rows are as before.
    adapters = config.adapters
    if adapters is None:
        return []
    arms = "; ".join(
        f"`{arm.name}` ({arm.source}" + (f", {arm.steps} steps" if arm.steps else "") + ")"
        for arm in adapters.arms
    )
    views = "the reference's 16" if adapters.views is None else str(
        [list(batch) for batch in adapters.views]
    )
    inputs = ", ".join(f"`{name}` version {v}" for name, v in config.notebook_inputs.items())
    datasets = ", ".join(f"`{name}` version {v}" for name, v in config.dataset_inputs.items())
    digest = adapters.hypernet_sha256 or "not read"
    return [
        f"| adapters (development runs only; task P3.7, ADR-019) | {arms} |",
        f"| adapters views per test input | {views} |",
        f"| adapters order_seed, ce, finish_s | {adapters.order_seed}, {adapters.ce}, "
        f"{adapters.finish_s} s |",
        f"| adapters gen | {adapters.gen} |",
        *(
            [f"| adapters train_log (per-step training loss) | {adapters.train_log} |"]
            if adapters.train_log is not None
            else []
        ),
        f"| adapters load_check | {adapters.load_check} |",
        f"| adapters checkpoint, cache | `{adapters.checkpoint}`, `{adapters.cache}` |",
        f"| adapters episodes (manifest) | `{adapters.episodes_root}`, `{adapters.check_root}` "
        f"(`{adapters.manifest_sha256}`) |",
        f"| adapters hypernet | `{adapters.hypernet}` (sha256 `{digest}`) |",
        f"| notebook_inputs (`{{input:<owner>/<slug>}}` is its folder) | {inputs} |",
        *(
            [f"| dataset_inputs (`{{input:<owner>/<slug>}}` is its folder) | {datasets} |"]
            if datasets
            else []
        ),
    ]


def _rounds_summary(rounds: notebook_builder.ArmRounds) -> str:
    # A round arm in MANIFEST.md's arms row (task P2.3): its settings and the governor config the
    # notebook embeds, by path and sha256 in the built commit.
    digest = rounds.governor_sha256 or "not read"
    return (
        f"{notebook_builder.rounds_text(rounds._asdict())}; governor `{rounds.governor}` "
        f"(sha256 `{digest}`)"
    )


def _script_summary(config: notebook_builder.ScriptConfig, hardware: str) -> list[str]:
    # MANIFEST.md's config rows of a script build (task P0.5d).
    script = Path(config.script).name
    script_config = Path(config.script_config).name
    args = " ".join(f"`{arg}`" for arg in config.args) or "none"
    inputs = (
        ", ".join(f"`{name}` version {version}" for name, version in config.notebook_inputs.items())
        or "none"
    )
    return [
        "| fit to submit | no, a script notebook is never submitted |",
        "| kind | `script` |",
        f"| accelerator | {hardware} |",
        f"| script | `{config.script}`, as `{CODE_DIR}/{script}` |",
        f"| script_config | `{config.script_config}`, as `{CODE_DIR}/{script_config}` |",
        f"| args (after `--config`; `{{commit}}` is the commit's SHA) | {args} |",
        f"| notebook_inputs (`{{input:<owner>/<slug>}}` in args is its folder) | {inputs} |",
        f"| script_kill_s (hard stop) | {config.script_kill_s} |",
        f"| model_slug | `{config.model_slug}` |",
    ]


def write_folder(target: Path, files: dict[str, bytes]) -> None:
    """Write ``files`` into a new folder beside ``target``, then swap it in for ``target``.

    Why: a rebuild must not mix with an older build; a stale wheel left in ``arc26-code/`` would
    be uploaded with the new one.

    Args:
        target: The build folder, ``dist/<tag>``; replaced if it exists.
        files: Contents by path relative to ``target``.

    Raises:
        BuildError: If a file cannot be written.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    except OSError as err:
        raise BuildError(f"cannot create a folder beside {target}: {err}") from err
    try:
        for name, data in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except OSError as err:
        shutil.rmtree(staging, ignore_errors=True)
        raise BuildError(f"cannot write {target}: {err}") from err


def _read_export(tree: Path, name: str, commit: str) -> bytes:
    # A file of the exported commit, by its path in the repository.
    try:
        return (tree / name).read_bytes()
    except OSError as err:
        raise BuildError(f"{name} is not in commit {commit[:12]}: {err}") from err


def build(tag: str, config_path: Path, repo: Path = REPO) -> Path:
    """Build ``dist/<tag>/`` from ``HEAD`` and one build config.

    Why: see the module docstring. Every check runs before the first write. A script config's
    script and config come from the same export as everything else, so they are the commit's.

    Args:
        tag: Name of the output folder, letters, digits, ``.``, ``_``, ``-``.
        config_path: The config, inside ``repo`` and committed; relative paths are taken from the
            current directory.
        repo: The repository root.

    Returns:
        The build folder.

    Raises:
        BuildError: If the tag is malformed, the tree is not clean, the config is outside the
            repository, not committed, or refused (``parse_config``), a round arm's governor
            config is not in the commit or refused (``resolve_rounds``), a hybrid build's agent
            run config is not in the commit or refused (``resolve_agent``), a script config's script
            or config is not in the commit or refused (``check_script_files``), the reference
            metadata is missing or refused, the wheel build fails, or a file cannot be written.
    """
    if not notebook_builder.safe_id(tag):
        raise BuildError(f"--tag must be letters, digits, '.', '_', '-'; got {tag!r}")
    repo = repo.resolve()
    config_file = config_path.resolve()
    if not config_file.is_relative_to(repo):
        raise BuildError(f"the config {config_path} is not inside the repository {repo}")
    relative = config_file.relative_to(repo)
    require_clean(repo)
    commit, commit_time = head_commit(repo)
    wheel_commit, wheel_epoch = wheel_input_commit(repo, commit)
    with tempfile.TemporaryDirectory(prefix="arc26-build-") as temporary:
        tree = Path(temporary) / "tree"
        export_tree(repo, commit, tree)
        config_bytes = _read_export(tree, relative.as_posix(), commit)

        def read(name: str) -> bytes:
            return _read_export(tree, name, commit)

        # A round arm's governor config (task P2.3), an adapter build's H config (task P3.7), and
        # a hybrid build's agent run config (task P6.5) are other files of the same commit.
        config = notebook_builder.parse_config(config_bytes, relative)
        config = notebook_builder.resolve_agent(
            notebook_builder.resolve_adapters(notebook_builder.resolve_rounds(config, read), read),
            read,
        )
        # A script build's script and config, by their file names in the code dataset folder.
        extras: dict[str, bytes] = {}
        hashes: dict[str, str] = {}
        if config.kind == "script":
            script = _read_export(tree, config.script, commit)
            script_config = _read_export(tree, config.script_config, commit)
            extras = notebook_builder.check_script_files(config, script, script_config)
            hashes = {
                "script_sha256": _sha256(script),
                "script_config_sha256": _sha256(script_config),
            }
        try:
            pulled = json.loads((tree / config.reference_metadata).read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise BuildError(f"cannot read {config.reference_metadata}: {err}") from err
        if not isinstance(pulled, dict):
            raise BuildError(f"{config.reference_metadata} must hold a JSON object")
        metadata = kernel_metadata(config, pulled)
        dataset = dataset_metadata(
            config.code_dataset, config.script if config.kind == "script" else None
        )
        wheel, builder = build_wheel(tree, Path(temporary) / "wheel", wheel_epoch)
        wheel_bytes = wheel.read_bytes()
        info = notebook_builder.build_info(
            config,
            tag=tag,
            commit=commit,
            config_file=relative.as_posix(),
            config_sha256=_sha256(config_bytes),
            wheel=wheel.name,
            wheel_sha256=_sha256(wheel_bytes),
            **hashes,
        )
        if config.kind == "script":
            notebook = notebook_builder.build_script_notebook(config, info)
        else:
            notebook = notebook_builder.build_notebook(
                config, info, notebook_builder.wheel_member(wheel)
            )
        files = {
            notebook_builder.notebook_file(config): notebook_builder.notebook_bytes(notebook),
            METADATA_FILE: _json_bytes(metadata),
            f"{CODE_DIR}/{wheel.name}": wheel_bytes,
            f"{CODE_DIR}/{DATASET_METADATA_FILE}": _json_bytes(dataset),
            **{f"{CODE_DIR}/{name}": data for name, data in extras.items()},
        }
        files[MANIFEST_FILE] = render_manifest(
            tag=tag,
            build=info,
            commit_time=commit_time,
            wheel_commit=wheel_commit,
            wheel_epoch=wheel_epoch,
            config=config,
            builder=builder,
            generator=wheel_generator(wheel),
            metadata=metadata,
            files=dict(files),
        )
    target = repo / "dist" / tag
    write_folder(target, files)
    return target


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tag", required=True, help="output folder name, dist/<tag>/")
    parser.add_argument(
        "--config", required=True, help="configs/submission_<name>.yaml, or a kind: script config"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build one submission or script folder.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 when the build is refused.
    """
    args = build_parser().parse_args(argv)
    try:
        target = build(args.tag, Path(args.config))
    except BuildError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print(f"wrote {target}; {MANIFEST_FILE} lists every file and the Kaggle commands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
