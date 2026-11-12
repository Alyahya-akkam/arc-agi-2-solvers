r"""Generate the Kaggle submission notebook from a submission config (BRIEF A5, B5 P0.5).

Purpose: the notebook that runs the arc26 pipeline on Kaggle is generated, never hand-edited, so
every build can be reproduced from a commit (A5). Task P0.5; packaging decision ADR-010; runbook
``docs/runbooks/build_submission.md``; steps for the human ``docs/SUBMITTING.md``.
``scripts/build_submission.py`` calls this module to write ``dist/<tag>/kaggle_submission.ipynb``;
the command line here writes the notebook alone, for review.

Inputs: a submission config (``configs/submission_*.yaml``, read with ``yaml.safe_load``; this
module parses and checks it for both scripts) and the built ``arc26`` wheel. The notebook checks
the wheel's file name and sha256 before installing it, and embeds its ``arc26/submit.py``.
Output: an nbformat 4.5 notebook. Every code cell follows a markdown cell that says what it does
and why, and holds plain Python: no ``!`` or ``%`` magics, every command through ``subprocess``
with its exit code checked.

1. Config, the only cell a human edits: subset, workers, hash seed, the solver's code
   (``SOLVER``), a scoped build's LoRA scope (``SCOPE``), ``RUN_SOLVER``, the solver's soft
   deadline and hard stop, the series arms (``ARMS``), and the build provenance. It takes the
   notebook's start time.
2. Helpers: the mode from ``KAGGLE_IS_COMPETITION_RERUN``, paths under ``ARC26_KAGGLE_ROOT``
   (default ``/kaggle``), the series in effect and each arm's folders, stage timing, and the
   runtime helpers of this module, verbatim.
3. Submission validator: ``arc26/submit.py`` from the wheel, verbatim, so a valid submission can
   be written before and without the install.
4. Environment check: both interpreters, GPUs through ``nvidia-smi`` (the kernel never imports
   torch), library versions, and the input files found by search; then a placeholder submission.
5. Install: uninstall tensorflow, check the wheel's sha256, ``pip install --no-index --no-deps``
   with the ``python`` on PATH, check that the kernel imports the installed copy.
6. Run: ``arc26.runner.run_solver`` with the hard stop as its timeout.
7. Write: ``make_submission`` into a staging file inside ``try``; ``finalize_submission`` writes
   and validates ``submission.json`` in ``finally``.
8. Development mode only: ``run_logs/`` as run A of P0.1-01 recorded it (``dump_run_logs`` from
   ``scripts/build_reference_run.py``, verbatim) plus ``stages.json``.

Development mode stops on a missing input or a failed install; a competition rerun logs each
problem and carries on, so it always ends with a valid ``submission.json``. Development mode also
records timing (task P0.3b): the solver's workers and ``make_submission`` append JSONL records to
``run_logs/timing/`` (``TIMING_DIR``), which the pull brings back with the other logs; a
competition rerun passes no timing folder, so its path is the one validated before timing
existed.

Series mode (task P1.3): a config may list named arms, each a LoRA rank and target modules with a
reason. A development run then launches the solver once per arm, in the listed order, each with
its own candidate, timing, and ``run_logs/<arm>/`` folders, and starts no further arm when the
time left before the soft deadline cannot cover one (``series_refusal``) or an arm failed.
``submission.json`` comes from the first arm, which must be the notebook configuration or, from
task P2.5, an arm that holds a pair out (a loop measurement, whose scope is a run setting); a
competition rerun ignores the series. ``submission_blocker`` marks a build not for submission
unless it is for the submission notebook (``SUBMISSION_NOTEBOOK``) with the production settings
(four workers on ``NvidiaL4``, no series).

Solver code (task P2.0, ADR-011): ``solver`` is ``split``, the default since the split solver
passed its GPU parity run (``docs/experiments/P2.0-01.md``), which runs the reference's statements
split into ``arc26.ttt`` and ``arc26.decode`` (``python -m arc26.solver``), or ``nvarc``, the
moved reference, which the configs of the runs before the flip state. It applies in both modes, a
competition rerun included; ``submission_blocker`` does not read it (ADR-011 lets either submit).

Hold-out arms (task P2.1): an arm of a series may hold one demonstration pair per puzzle out of
training and prompts and run the held-out gate (``arc26.loop``): its ``holdout`` sets the rule,
the ``k = 2`` flag, and whether the pair stays in the test inputs' prompts, and the notebook adds
the arm's gate folders (``gate_outputs/<name>/`` for the gate's candidates, beside
``inference_outputs/``; ``run_logs/<name>/gate/`` for its verdicts). It needs ``solver: split``.
A competition rerun ignores the series, and with it every hold-out arm.

Round arms (task P2.3, ADR-015): a hold-out arm may also run the refinement rounds after each
puzzle's gate: its ``rounds`` sets ``max_rounds``, ``tau``, ``continue_steps``,
``final_in_prompts`` (true is ``arc26.loop.Rounds``' default: ADR-013 as amended after
``P2.1-02`` keeps the held-out pair in the decode that is submitted), and ``governor``, a config
file of the repository (``configs/governor.yaml``) whose parsed content the builder embeds
(``resolve_rounds``: the wheel carries ``arc26/`` only);
the notebook adds the arm's round folder (``round_outputs/<name>/``) and passes the settings
inside the arm's hold-out settings. Their records go beside the gate's verdicts. Task P2.5
(ADR-016) adds the optional ``variants``: ``[gradient]`` (``Rounds``' default), ``[in_context]``
(the in-context round: no fit, the promoted pairs join the prompts), or both in the governor's
order, so one arm can force the variant a run measures.

Scope (task P2.6, ADR-018): a config without arms may set ``scope``, one LoRA scope
(``lora_r``, ``lora_targets``, checked as an arm's) for its single run. Unlike an arm's, it
applies in both modes: the Config cell holds it as ``SCOPE``, and the Run cell passes it to
``RunConfig``, so a competition rerun trains what a development run of the build trained. A scope
is no reason against submitting (``submission_blocker``), a scope with arms is refused, and the
first cell, the environment check, and ``MANIFEST.md`` state it. A build without one is as
before, byte for byte: the scoped build's lines in fixed cells come from ``SCOPE_EDITS``.

Hybrid build (task P6.5, ADR-020): a config without arms may set ``agent``, which runs the agent
solver after NVARC, in both modes. An agent check after the environment check finds the two
download notebooks' outputs the build attaches (the weights and the engine wheelhouse, task P6.3)
and checks their manifests, the GPUs, and the driver. When ``AGENT['enabled']`` and the check
passes, NVARC's soft deadline is its share of the session (``nvarc_share_s``, 0 for an agent-only
build) and its hard stop ``NVARC_KILL_GAP_S`` later; otherwise NVARC keeps its full deadlines.
After NVARC's ``submission.json`` is written, the agent stage installs the wheelhouse offline and
runs ``python -m arc26.agent.run`` until ``end_s`` (SIGTERM, then SIGKILL ``term_s`` later), and a
final check keeps every real grid on disk and NVARC's for every other slot. In an agent-only build
an agent stage that ends ``NVARC_FALLBACK_MIN_S`` or more before NVARC's soft deadline is followed
by NVARC's stage, and P6.2's selection merges the agent's attempt log with NVARC's new attempts,
verified programs first (``nvarc_fallback``). The GPUs are logged and waited for before each
boot (``wait_for_free_gpus``: no process listed, ``GPU_USED_MAX_MIB`` at most). The builder embeds
the agent's run config (``resolve_agent``) and refuses a plan that could not start the agent or
end before Kaggle's limit (``agent_plan``). A hybrid build is fit to submit; a build without the
key is as before, byte for byte: the hybrid's lines in fixed cells come from ``HYBRID_EDITS``, and
its four cells are added.

Shot build (task P6.5b): a hybrid config may set ``agent.entry`` to ``arc26.agent.shot``
(``SHOT_ENTRY``; the default, ``arc26.agent.run``, builds as before), the long shot of run
``P6.3-05`` on the tasks NVARC is least sure of, which takes exactly ``arc26.agent.run``'s
arguments and exit codes. NVARC keeps its deadlines (its share must be the competition's soft
deadline), so the shot runs only in the time NVARC leaves: the builder plans it with
``shot_plan`` (the latest stage start that still runs it, ``end_s`` minus the run config's final
margin minus ``min_stage_s``), and the agent stage passes the evaluation solutions in a
development run for the shot's own score (``SHOT_EDITS``). DEV240 (``subset: DEV240``, task
P6.5b): a development run's environment check writes every evaluation task and a copy of each
under the first 8 hex digits of the sha256 of its id (``dev240_files``) and points ``DATA_DIR``
there, so every later stage reads 240 tasks; a competition rerun is unchanged.

LLM-first build (task P5.1): a shot build may set ``agent.skip_nvarc: true``. When the agent check
passes, in both modes, the Run cell does not start NVARC's solver and the Write cell runs no
selection, so ``submission.json`` holds ``[[0]]`` placeholders until the shot rewrites it; the
shot's run config must queue every task by its opening (``selection.order: opening``), and the
build has no rank-256 tail. A failed agent check still runs NVARC with its full deadlines. A
build without the key keeps every cell, byte for byte: the lines come from ``SKIP_NVARC_EDITS``.

bf16 stop (task P0.5d): a build made for an accelerator stops in the environment check, before
the install, when a visible GPU has compute capability under 8.0 or none is visible
(``bf16_refusal``); a competition rerun logs it and carries on.

Script mode (task P0.5d): a config with ``kind: script`` builds ``kaggle_script.ipynb``, which runs
one script of ``scripts/`` (BRIEF A8) on the same stack instead of the solver: Config (``ARGS``,
``SCRIPT_KILL_S``, ``BUILD``), Helpers, Environment check (the script and its config found beside
the wheel and sha256-checked), Install, Script (``python <script> --config <config> <ARGS>`` in a
process group of its own, killed at the hard stop), Run logs. It writes no ``submission.json``,
stops at once in a competition rerun, and records a problem instead of raising, so ``run_logs/``
is written however the run ends. ``scripts/build_submission.py`` puts the script and its config,
byte for byte, beside the wheel in the code dataset folder. A script config may also attach the
outputs of other notebooks at pinned versions (``notebook_inputs``, task P3.5): they join
``kernel_sources``, the environment check finds each mount as the folder named by its slug, and
``{input:<owner>/<slug>}`` in the script's arguments stands for that folder.

Usage::

    python scripts/build_kaggle_notebook.py --config configs/submission_P0.5-01.yaml \
        --wheel <arc26 wheel> --commit <sha> --tag <tag> --out <notebook.ipynb>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import pprint
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, NamedTuple

import yaml

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent


def _load(name: str) -> ModuleType:
    # Sibling scripts are loaded by path, as the tests load them: scripts/ is not a package.
    spec = importlib.util.spec_from_file_location(f"_arc26_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# dump_run_logs and the reference build's config rules; ACCELERATORS and KAGGLE_OWNER.
reference_run = _load("build_reference_run")
gateway = _load("kaggle_ops")

NOTEBOOK_FILE = "kaggle_submission.ipynb"
# A script build's notebook (task P0.5d): its own name, since it is never submitted.
SCRIPT_NOTEBOOK_FILE = "kaggle_script.ipynb"
SUBMIT_MEMBER = "arc26/submit.py"
# Kaggle ends a GPU session, and a competition rerun, after 12 h (research note, section 2).
SESSION_LIMIT_S = 12 * 3600
# Time kept between the solver's hard stop and the session limit for make_submission, the
# validator, and the development logs. In run A the score was printed 40 s after the last worker
# finished (the kernel's transformers import included) and the logs took 3 s more, so 5 minutes
# is a wide margin.
WRITE_RESERVE_S = 300
# A soft deadline under 10 minutes is a mistake: the workers import unsloth one after another,
# and the fourth started 326 s into run A.
MIN_SOLVER_END_S = 600
# Deadlines of a competition rerun, in seconds after the notebook's start, used whatever the Config
# cell says, so every build can be submitted. Kaggle stops the rerun at 12 h and reports up to
# 10 minutes of runtime variance (research note, section 2). The workers' soft deadline is 12 h
# minus 30 min, A5's watchdog margin and 20 min before the reference's own (12 h minus 600 s).
COMPETITION_SOLVER_END_S = SESSION_LIMIT_S - 30 * 60
# The hard stop is 10 min after that: the DFS stops at the soft deadline, and the rescoring and
# pickle writes that follow it take far less. It leaves 20 min before the limit to write the
# submission and absorb Kaggle's variance (run A needed under a minute to write).
COMPETITION_SOLVER_KILL_S = SESSION_LIMIT_S - 20 * 60
# GPUs per accelerator where known (docs/research/kaggle_first_runs.md, section 3.2); the solver
# runs one worker per GPU.
GPU_COUNTS = {"NvidiaL4": 4, "NvidiaL4X1": 1}
# Accelerators the gateway accepts whose sessions can never pass the bf16 stop (bf16_refusal,
# task P0.5d): a build for one would stop every development run in its environment check, after
# the queue wait. 2xT4 also runs pushes whose accelerator Kaggle does not grant (P0.3-01).
NO_BF16_ACCELERATORS = {
    "NvidiaTeslaT4": "T4 has compute capability 7.5, without bf16",
    "TpuV5E8": "a TPU session has no NVIDIA GPU",
    "TpuV6E8": "a TPU session has no NVIDIA GPU",
}
MAX_WORKERS = 8
# The one notebook submissions come from (docs/SUBMITTING.md): the human submits a version of it,
# and Kaggle reruns that version. Every other notebook (the timing run's, the CPU check's, a
# series', a future script's) holds development builds by construction, whatever its settings.
SUBMISSION_NOTEBOOK = "alyahya737/arc26-submission"
# Production: the settings a competition rerun of a submitted build needs, as the reference ran
# and as the scored rerun's hardware gives (BRIEF B1 item 2, B3): one worker per L4 on 4xL4.
PRODUCTION_ACCELERATOR = "NvidiaL4"
PRODUCTION_WORKERS = 4
CONFIG_KEYS = (
    "kaggle_notebook",
    "code_dataset",
    "competition",
    "accelerator",
    "reference_metadata",
    "kernel_source_versions",
    "model_slug",
    "subset",
    "workers",
    "hash_seed",
    "run_solver",
    "solver_end_s",
    "solver_kill_s",
)
# Keys a config may leave out: `arms` (series mode, task P1.3); absent or null, the build runs
# once with the reference's LoRA settings, as before P1.3. `solver` (task P2.0); absent, the
# split solver runs (the default since 2 Oct, ADR-011); the runs before the flip state nvarc. `kind`
# (task P0.5d); absent, the config is a submission config, as every config before it. `scope`
# (task P2.6, ADR-018); absent or null, the single run trains the reference's LoRA settings.
# `adapters` (task P3.7, ADR-019): the single run's adapter arms, development runs only, with
# `notebook_inputs`, the outputs of other notebooks they read (H's checkpoint, the embedding
# cache, P3.4's episodes); absent or null, one adapter source, the reference's fit.
# `dataset_inputs` (task P3.9b): with `adapters` only, private Kaggle datasets the arms read at
# pinned versions, `<owner>/<slug>: <version>` (H's checkpoint published by a Modal run, which is
# a dataset, not a notebook output); absent or null, none, as before P3.9b.
# `agent` (task P6.5, ADR-020); absent or null, no agent stage: NVARC alone, as before.
OPTIONAL_KEYS = (
    "arms", "solver", "kind", "scope", "adapters", "notebook_inputs", "dataset_inputs", "agent",
)  # fmt: skip
# Config kinds (task P0.5d): "submission" builds the solver notebook; "script" builds a notebook
# that runs one script (BRIEF A8) on the same stack and is never submitted.
KINDS = ("submission", "script")
DEFAULT_KIND = "submission"
# A script config: its kind, the keys every config shares (identity, attached sources,
# accelerator), the script and its config, the script's extra arguments, and the hard stop.
SCRIPT_CONFIG_KEYS = (
    "kind",
    "kaggle_notebook",
    "code_dataset",
    "competition",
    "accelerator",
    "reference_metadata",
    "kernel_source_versions",
    "model_slug",
    "script",
    "script_config",
    "args",
    "script_kill_s",
)
# Keys a script config may leave out (task P3.5): `notebook_inputs`, the outputs of other Kaggle
# notebooks that the notebook attaches as inputs, `<owner>/<slug>: <version>`. Absent or null, it
# attaches none, as before P3.5. The reference's own kernel sources stay in kernel_source_versions.
SCRIPT_OPTIONAL_KEYS = ("notebook_inputs",)
# Notebook outputs one script build may attach: each is a mount and a search.
MAX_NOTEBOOK_INPUTS = 8
# "{input:<owner>/<slug>}" in an argument stands for the mount folder of that attached notebook
# output, which the notebook finds by search (find_input_dirs, fill_args; task P3.5).
INPUT_PLACEHOLDER = re.compile(r"\{input:([^{}]*)\}")
# A hard stop under 5 minutes leaves a script almost nothing: the environment check and the
# install take about a minute (run A's tensorflow uninstall alone took 21 s).
MIN_SCRIPT_KILL_S = 300
MAX_ARGS = 32
# The solver's code, arc26.runner.SOLVER_MODULES's keys (tests/test_build_submission.py checks
# them against it): "nvarc" runs the moved reference (arc26.nvarc.starter), "split" the same
# statements split into arc26.ttt and arc26.decode (arc26.solver, ADR-011). "split" is the
# default since run P2.0-01 passed (2 Oct); configs of earlier runs state "nvarc".
SOLVERS = ("nvarc", "split")
DEFAULT_SOLVER = "split"
# The first markdown cell's line for a build whose solver is not DEFAULT_SOLVER.
OTHER_SOLVER_NOTES = {
    "nvarc": (
        "Moved-reference build (task P2.0): `SOLVER = 'nvarc'` runs the moved reference "
        "(`python -m arc26.nvarc.starter`), the parity anchor, instead of the split solver, in "
        "both modes (`docs/decisions/ADR-011-solver-split.md`).\n\n"
    ),
    "split": (
        "Split-solver build (task P2.0): `SOLVER = 'split'` runs the reference's statements "
        "split into `arc26.ttt` and `arc26.decode` (`python -m arc26.solver`) instead of the "
        "moved reference, in both modes (`docs/decisions/ADR-011-solver-split.md`).\n\n"
    ),
}
ARM_KEYS = ("name", "lora_r", "lora_targets", "reason")
# The build's LoRA scope (task P2.6, ADR-018): exactly these keys, an arm's and RunConfig's names.
SCOPE_KEYS = ("lora_r", "lora_targets")
# The decision record of the build's scope, under docs/decisions/; a scoped build's first cell
# names it.
SCOPE_ADR = "ADR-018-submission-2-without-the-loop.md"
# A hybrid build's agent stage (task P6.5, ADR-020): exactly these keys, each spelled out, since
# each one changes what a competition rerun of the build does. The first six are the Config
# cell's AGENT, which a human may edit; the rest are the build's (BUILD["agent"]).
AGENT_STAGE_KEYS = (
    "enabled", "nvarc_share_s", "end_s", "term_s", "min_stage_s", "install_timeout_s",
)  # fmt: skip
AGENT_BUILD_KEYS = ("config", "model", "wheels", "base_python", "scratch_dir", "min_driver")
AGENT_KEYS = (*AGENT_STAGE_KEYS, *AGENT_BUILD_KEYS)
# The two download notebooks' outputs a hybrid build attaches (task P6.3): the weights pinned by
# the revision their MANIFEST.json must name, the wheelhouse by the vLLM version it must hold.
AGENT_MODEL_KEYS = ("notebook", "version", "folder", "revision")
AGENT_WHEELS_KEYS = ("notebook", "version", "folder", "vllm")
# The agent run config's values the notebook reads (task P6.4's configs/agent_default.yaml, which
# arc26.agent.run checks in full): generation stops final_margin_s before the session end the
# notebook passes; the engine needs tensor_parallel_size GPUs and stops in stop_timeout_s.
AGENT_CONFIG_DEADLINE_KEYS = ("final_margin_s", "grace_s")
AGENT_CONFIG_SERVER_KEYS = ("tensor_parallel_size", "boot_timeout_s", "stop_timeout_s")
# The decision record of the agent solver; a hybrid build's first cell names it.
AGENT_ADR = "ADR-020-agent-solver-pivot.md"
# The module the agent stage runs (task P6.5b), the optional key entry: P6.4's agent (the default,
# so a config without the key builds as before, byte for byte) or P6.3-05's long shot on NVARC's
# least sure tasks, which takes exactly arc26.agent.run's arguments and exit codes, so the
# notebook switches modules by this one key.
# skip_nvarc (task P5.1): an LLM-first shot build, NVARC not run when the agent check passes.
AGENT_OPTIONAL_KEYS = ("entry", "warmup", "tail", "skip_nvarc")
DEFAULT_AGENT_ENTRY = "arc26.agent.run"
SHOT_ENTRY = "arc26.agent.shot"
AGENT_ENTRIES = (DEFAULT_AGENT_ENTRY, SHOT_ENTRY)
# The weights' warm-up (task P6.9c), the optional key agent.warmup of a shot build: at the agent
# check, before NVARC, a background process copies the weights from their input mount to local
# scratch and verifies them, so the engine loads them from local disk (P6.9-02: the input on
# FUSE.GCSFUSE took 802.41 s to load, past the 900 s boot limit). Exactly these keys, each
# changing what a competition rerun does; a build without the key keeps every cell, byte for byte.
AGENT_WARMUP_KEYS = (
    "delay_s", "workers", "piece_mib", "nice", "progress_s", "reserve_gb", "wait_s", "stall_s",
    "fallback_load_strategy",
)  # fmt: skip
# vLLM 0.31.0's SafetensorsLoadStrategy (vllm/config/load.py at db9527a4: lazy, eager, prefetch,
# torchao): the fallback may only force the page-cache prefetch its loader names for a filesystem
# it does not prefetch on its own (vllm/model_executor/model_loader/weight_utils.py, line 1086).
AGENT_WARMUP_LOAD_STRATEGIES = ("prefetch",)
# Bounds of the warm-up's settings: (key, least, most).
AGENT_WARMUP_BOUNDS = (
    ("delay_s", 0, 3600), ("workers", 1, 64), ("piece_mib", 1, 4096), ("nice", 0, 19),
    ("progress_s", 1, 600), ("reserve_gb", 0, 1000), ("wait_s", 0, 3600), ("stall_s", 5, 3600),
)  # fmt: skip
# The warm-up's files in the agent's scratch (task P6.9c): the copy under WARMUP_DIR/<folder>,
# its status JSON and log (copied to run_logs/agent/ in a development run), its spec, its script.
WARMUP_DIR = "weights"
WARMUP_STATUS = "warmup.json"
WARMUP_LOG = "warmup.log"
WARMUP_SPEC = "warmup_spec.json"
WARMUP_SCRIPT_FILE = "warm_weights.py"
# Seconds between the agent stage's reads of the warm-up's status while it waits for the copy.
WARMUP_POLL_S = 5
# The rank-256 tail (task P4.7b), the optional key agent.tail of a shot build: after the shot
# stage, NVARC's solver runs again with the reference's LoRA settings (RunConfig's defaults) on
# rank 16's confidence-0 tasks, cheapest first by NVARC's puzzle seconds, and fills only the
# outputs that still hold NVARC's placeholder. Exactly these keys, each changing what a
# competition rerun does; a build without the key keeps every cell, byte for byte.
AGENT_TAIL_KEYS = ("write_s", "kill_gap_s", "min_s")
# Bounds of the tail's settings: (key, least, most).
AGENT_TAIL_BOUNDS = (("write_s", 60, 3600), ("kill_gap_s", 30, 1800), ("min_s", 60, 43200))
# The tail's entry (arc26/tail.py): arc26.solver's arguments and workers, its challenges file's
# tasks queued in the file's order.
TAIL_MODULE = "arc26.tail"
# Where NVARC's workers write their timing records in a tail build's competition rerun: scratch
# under the root, never /kaggle/working, so the tail can order its tasks by NVARC's seconds.
TAIL_NVARC_TIMING_DIR = "arc26_timing"
# Bytes per read and write of the copy and the hash: 8 MiB, a few hundred calls per piece.
WARMUP_BLOCK_BYTES = 8 << 20
# A shot build's first cell title (task P6.5b). NVARC keeps Submission 2's deadlines there (the
# brief's design 1: its marginal task is worth far more than the shot's), so its share must be
# the competition soft deadline, and the shot runs in the time NVARC leaves.
SHOT_TITLE = "Hybrid build: NVARC, then the long shot"
# An LLM-first build's first cell title (task P5.1): the shot on every task, no NVARC first.
SKIP_NVARC_TITLE = "LLM-first build: the long shot on every task, no NVARC"
# The shot's queue order an LLM-first build needs (arc26.agent.shot.OPENING_ORDER): every task,
# the smallest opening first, since without NVARC no task has a confidence.
SKIP_NVARC_ORDER = "opening"
# The Run cell's record of the skip (task P5.1). Not "agent only", so the agent-only fallback,
# which reads arc26.agent.run's attempt log and would replace the shot's answers, never runs.
SKIP_NVARC_REASON = "LLM first: BUILD['agent']['skip_nvarc'] and the agent check passed"
# DEV240 (task P6.5b), a development subset for the 12 h timing check: DEV120 and a second copy of
# each task under the first 8 hex digits of the sha256 of its id, scored against the original's
# solutions. Its files are written to this folder under the root, never to /kaggle/working, and
# only in development mode.
DEV240_SUBSET = "DEV240"
# A shot build's allowance for the time Kaggle may count before the notebook's first cell (task
# P6.5b, the coordinator's decision of 14 Oct): P6.5-02 showed RUNNING 36.7 min before its first
# cell (P6.3-02 revision 5 41 min, P6.3-04 about 10; it grows with the attached inputs). If the
# 12 h limit counts from RUNNING, a shot planned against 43,200 s of notebook time would be
# killed mid-wave, so the shot plans against SESSION_LIMIT_S minus this (45 min, above every
# measured case). NVARC's own deadlines stay the competition's (Submission 2's).
PRE_SCRIPT_ALLOWANCE_S = 45 * 60
SHOT_SESSION_S = SESSION_LIMIT_S - PRE_SCRIPT_ALLOWANCE_S
DEV240_DIR = "arc26_dev240"
# NVARC's hard stop in a hybrid build comes this long after its share, the competition
# deadlines' gap: the 10 minutes cover what a worker does after its DFS stops (rescoring and
# pickle writes).
NVARC_KILL_GAP_S = COMPETITION_SOLVER_KILL_S - COMPETITION_SOLVER_END_S
# Seconds a process group gets to die after SIGKILL (the agent's, then its engine's) before the
# notebook goes on; P6.4's server waits as long for its engine.
AGENT_KILL_WAIT_S = 30
# Seconds between SIGTERM and SIGKILL of the installer's process group at its timeout: pip holds
# nothing worth a longer wait.
INSTALL_TERM_S = 10
# Seconds the agent stage waits for the GPUs to be free, before the engine boots (NVARC's workers
# held them a moment before) and before the agent-only fallback runs NVARC: a killed process's
# memory returns to the driver within seconds.
GPU_FREE_WAIT_S = 60
# A GPU counts as free when nvidia-smi lists no process on it and reports at most this much memory
# in use (MiB): inside a container --query-compute-apps can list nothing while a killed worker's
# memory is still held, so memory.used is polled too. vLLM asks for 0.90 of an L4's 23,034 MiB
# (configs/agent_default.yaml) and does not boot with less free, which leaves about 2,300 MiB to
# others; what an idle Kaggle L4 reports is unmeasured, and the wait's log records it.
GPU_USED_MAX_MIB = 1024
# The agent-only fallback (task P6.5) runs NVARC only with at least this long left before NVARC's
# soft deadline: the model loads in about 5 minutes, and a puzzle takes about 8 minutes on four
# workers (P2.6-01: 499.5 s), so less than 30 minutes adds next to nothing.
NVARC_FALLBACK_MIN_S = 1800
# Bounds of term_s, the agent's grace between SIGTERM and SIGKILL: at least enough for the run to
# stop its engine (stop_timeout_s, checked by resolve_agent), at most 10 minutes of a session.
MIN_AGENT_TERM_S, MAX_AGENT_TERM_S = 30, 600
# Bounds of min_stage_s (the least time from the stage's start to the generation stop worth an
# install and a boot) and of install_timeout_s.
MIN_AGENT_STAGE_S = 60
MIN_INSTALL_TIMEOUT_S, MAX_INSTALL_TIMEOUT_S = 60, 7200
# A pinned weights revision is a full commit SHA; a vLLM version and a driver version are dotted
# numbers (vLLM may add a local part, as 0.31.0+cu130).
REVISION = re.compile(r"[0-9a-f]{40}")
DOTTED_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)+(?:\+[0-9A-Za-z.]+)?")
# The agent's scratch folder must hold a venv of about 10 GB and caches: never the notebook's
# output folder (Kaggle saves it, 20 GB) or the read-only inputs. The notebook falls back to the
# default if BUILD lost its value (/tmp had about 1.1 TB free in every probe, K0).
AGENT_SCRATCH_REFUSED = ("/kaggle/working", "/kaggle/input")
AGENT_DEFAULT_SCRATCH = "/tmp/arc26-agent"
# An arm may also hold a pair out (task P2.1); absent or null, it trains and decodes on every pair.
# A hold-out arm may also run the refinement rounds (task P2.3); absent or null, it runs none.
ARM_OPTIONAL_KEYS = ("holdout", "rounds")
# arc26.loop.HoldOut's settings that a config chooses; the notebook adds the arm's folders.
HOLDOUT_KEYS = ("rule", "k2", "in_test_prompts")
# arc26.loop.HOLDOUT_RULES's names (tests/test_build_submission.py checks them against it).
HOLDOUT_RULES = ("last", "first")
# The solver that can hold a pair out: the moved reference trains and decodes in one loop body.
HOLDOUT_SOLVER = "split"
# arc26.loop.Rounds' settings that a config chooses, each spelled out (task P2.3, ADR-015): every
# one changes what a round may do. governor names a config file of the repository; the builder
# embeds its parsed content (resolve_rounds), and the notebook adds the arm's round folder.
ROUNDS_KEYS = ("max_rounds", "tau", "continue_steps", "final_in_prompts", "governor")
# arc26.loop.Rounds.OPTIONAL (task P2.5): a round arm may name the variants it offers; absent, it
# offers Rounds' default, the gradient round, so the configs of task P2.3 build as before.
ROUNDS_OPTIONAL_KEYS = ("variants",)
# arc26.governor.LADDER (tests/test_in_context_build.py checks it): the variants, in the governor's
# order; an arm offers one of them or both in this order (arc26.loop.Rounds' rule).
VARIANTS = ("gradient", "in_context")
# arc26.loop.MAX_ROUNDS and EPOCH_VIEWS (tests/test_round_build.py checks them against it): B5
# P2.3's hard cap on rounds, and the views of one epoch, at most the continuation's steps.
MAX_ROUNDS = 2
EPOCH_VIEWS = 128
# Top-level keys of a governor config that the builder checks (configs/governor.yaml);
# arc26.loop.Rounds checks the rest with Governor.from_mapping when the notebook builds its
# RunConfig. The rounds plan after the baseline decode, so only the reference decode policy.
GOVERNOR_KEYS = ("ops", "margin", "candidates_per_batch", "deadline")
GOVERNOR_POLICY = "reference"
# The adapter arms (task P3.7, ADR-019): the keys of a config's `adapters`, every one spelled out
# (null where it does not apply), and of each arm. arc26.arms.SOURCES's names, its H and polish
# sources, its views' length groups and batch width (tests/test_arms_build.py checks them).
ADAPTER_KEYS = (
    "arms", "views", "order_seed", "ce", "finish_s", "checkpoint", "cache", "hypernet",
    "episodes_root", "check_root", "manifest_sha256", "load_check", "gen",
)  # fmt: skip
# Optional keys of `adapters`: absent, the arms entry's default applies and the Config cell omits
# them, so the builds of P3.7-01 and P3.7-02 keep their settings (train_log: false).
ADAPTER_OPTIONAL_KEYS = ("train_log",)
ADAPTER_ARM_KEYS = ("name", "source", "reason")
ADAPTER_ARM_OPTIONAL_KEYS = ("steps",)
ADAPTER_SOURCES = (
    "base", "hyper", "sgd", "hyper_polish", "polish", "hyper_fit", "hyper_merge_fit",
)  # fmt: skip
ADAPTER_H_SOURCES = ("hyper", "hyper_polish", "hyper_fit", "hyper_merge_fit")
ADAPTER_POLISH_SOURCES = ("hyper_polish", "polish")
ADAPTER_VIEW_GROUPS = ((0, 1, 4, 5, 10, 11, 14, 15), (2, 3, 6, 7, 8, 9, 12, 13))
ADAPTER_MAX_BATCH = 4
LOAD_CHECK_KEYS = ("run_logs", "max_abs", "max_mean")
GEN_KEYS = ("barc", "rearc")
# arc26.hyper.hypernet.HypernetConfig's fields, configs/hypernet.yaml's keys.
HYPERNET_KEYS = (
    "embed_dim", "latent", "head_in", "dropout", "rank", "alpha", "target_modules", "seed"
)  # fmt: skip
# HypernetConfig's optional field (task P3.3b): how H reads the demonstration set, a mapping of
# arc26.hyper.hypernet.AggregatorConfig's fields, whose defaults (the mean H) are these. A
# checkpoint's config leaves out a block equal to them (HypernetConfig.to_dict), so the build
# embeds the H config in that form too, and an H config without the block builds as before.
HYPERNET_OPTIONAL_KEYS = ("aggregator",)
AGGREGATOR_DEFAULTS = {"kind": "mean", "width": 512, "blocks": 2, "heads": 8}
# AggregatorConfig's kinds; its sizes are positive integers and heads divides width.
AGGREGATOR_KINDS = ("mean", "set")
# arc26.solver.load's lora_alpha: H's alpha must be it, or the live adapter's scale differs.
SOLVER_LORA_ALPHA = 32
ADAPTERS_ADR = "ADR-019-h-validation-paired-arms.md"
# Seconds between the arms' finish deadline (soft deadline plus finish_s) and the hard stop: the
# parent selects and scores every arm after its workers (about a minute per arm, estimate).
ADAPTER_SELECTION_RESERVE_S = 900
SHA256 = re.compile(r"[0-9a-f]{64}")
# The reference's LoRA settings (cell 4, `peft_params`), as arc26.runner.LORA_R and LORA_TARGETS
# (tests/test_lora_scope.py checks both against the notebook): the first arm of a series must use
# them, because submission.json comes from it. The first seven targets are the linear
# projections; Unsloth trains embed_tokens and lm_head as full copies.
REFERENCE_LORA_R = 256
REFERENCE_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
)  # fmt: skip
LORA_PROJECTIONS = REFERENCE_LORA_TARGETS[:7]
# An arm's name is a folder name (inference_outputs/<name>/, run_logs/<name>/) that the pull
# allowlist accepts: lowercase letters, digits, "-".
ARM_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
MAX_REASON = 200
# Series stop rule (series_refusal): the next arm starts only if the time left before the soft
# deadline covers this factor times the longest arm so far. A heuristic, not a bound: the
# estimate already takes the longest arm, the first arm (the notebook configuration) is the most
# expensive scope, and puzzle times of equal runs differed by at most 9% on DEV4 (P0.5-01
# against P1.1-01); 25% more leaves room for a longer queue tail (one puzzle is at most 1381 s,
# about 20% of a DEV40 arm on 4xL4) without refusing arms that would likely finish. An arm that
# still overruns is cut at the soft deadline and does not count.
SERIES_MARGIN = 1.25
# <owner>/<slug>. Slugs are lowercase words joined by "-", so a title made by replacing "-" with
# spaces slugifies back to the slug, as kaggle-cli checks.
OWNED_SLUG = re.compile(r"([A-Za-z0-9._-]+)/([a-z0-9]+(?:-[a-z0-9]+)*)")
TASK_ID = re.compile(r"[0-9a-f]{8}")
COMMIT = re.compile(r"[0-9a-f]{7,40}")
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


class BuildError(Exception):
    """The config or an input breaks a rule of the submission build; nothing was written."""


class ArmHoldOut(NamedTuple):
    """An arm's hold-out settings (task P2.1), the fields of ``arc26.loop.HoldOut`` it chooses."""

    rule: str
    k2: bool
    in_test_prompts: bool


class ArmRounds(NamedTuple):
    """An arm's round settings (task P2.3), the fields of ``arc26.loop.Rounds`` it chooses.

    ``governor`` is the governor config's path in the repository; ``resolve_rounds`` fills
    ``governor_config`` with its parsed content, which the notebook passes to the solver, and
    ``governor_sha256`` with the file's sha256, for ``MANIFEST.md``. ``variants`` (task P2.5) is
    None when the arm does not name them (``Rounds``' default, the gradient round).
    """

    max_rounds: int
    tau: float
    continue_steps: int
    final_in_prompts: bool
    governor: str
    governor_config: dict[str, Any] | None = None
    governor_sha256: str | None = None
    variants: tuple[str, ...] | None = None

    def settings(self) -> dict[str, Any]:
        """Return the settings as the Config cell holds them: ``Rounds``' fields, the folder aside.

        Returns:
            ``max_rounds``, ``tau``, ``continue_steps``, ``final_in_prompts``, and ``governor``,
            the parsed config; ``variants`` too when the arm names them (task P2.5).

        Raises:
            BuildError: If ``resolve_rounds`` has not read the governor config yet.
        """
        if self.governor_config is None:
            raise BuildError(f"the governor config {self.governor} was not read (resolve_rounds)")
        settings = {
            "max_rounds": self.max_rounds,
            "tau": self.tau,
            "continue_steps": self.continue_steps,
            "final_in_prompts": self.final_in_prompts,
            "governor": self.governor_config,
        }
        if self.variants is not None:
            settings["variants"] = list(self.variants)
        return settings


class Arm(NamedTuple):
    """One arm of a series: a LoRA scope the solver trains in its own launch (task P1.3).

    ``holdout`` (task P2.1) holds one demonstration pair per puzzle out and runs the held-out
    gate; None trains and decodes on every pair, as the reference does. ``rounds`` (task P2.3)
    runs the refinement rounds after the gate; None runs none.
    """

    name: str
    lora_r: int
    lora_targets: tuple[str, ...]
    reason: str
    holdout: ArmHoldOut | None = None
    rounds: ArmRounds | None = None


class LoraScope(NamedTuple):
    """The build's LoRA scope (task P2.6, ADR-018): ``arc26.runner.RunConfig``'s two fields.

    The single run of a build without a series trains it in both modes, a competition rerun
    included; an arm's scope applies to development runs only.
    """

    lora_r: int
    lora_targets: tuple[str, ...]

    def settings(self) -> dict[str, Any]:
        """Return the scope as the Config cell holds it, ``RunConfig``'s keyword arguments.

        Returns:
            ``lora_r`` and ``lora_targets``, the targets as a list.
        """
        return {"lora_r": self.lora_r, "lora_targets": list(self.lora_targets)}


class AdapterArm(NamedTuple):
    """One adapter arm (task P3.7): name, source (``ADAPTER_SOURCES``), reason, polish steps.

    ``steps`` is None for a source other than a polish.
    """

    name: str
    source: str
    reason: str
    steps: int | None = None


class AdapterArms(NamedTuple):
    """A build's adapter arms (task P3.7, ADR-019): ``arc26.arms.Arms``' settings a config sets.

    Paths may hold ``{input:<owner>/<slug>}``, which the notebook fills with that attached
    notebook output's folder; the notebook adds its own folders. ``hypernet`` is the expected H
    config's path in the repository; ``resolve_adapters`` fills ``hypernet_config`` with its
    parsed content and ``hypernet_sha256`` with the file's sha256.
    """

    arms: tuple[AdapterArm, ...]
    views: tuple[tuple[int, ...], ...] | None
    order_seed: int
    ce: bool
    finish_s: int
    checkpoint: str | None
    cache: str | None
    hypernet: str
    episodes_root: str | None
    check_root: str | None
    manifest_sha256: str | None
    load_check: dict[str, Any] | None
    gen: dict[str, Any] | None
    train_log: bool | None = None
    hypernet_config: dict[str, Any] | None = None
    hypernet_sha256: str | None = None

    def paths(self) -> list[str]:
        """Return the settings that are paths, the ones ``{input:...}`` may appear in."""
        found = [self.checkpoint, self.cache, self.episodes_root, self.check_root]
        if self.load_check is not None:
            found.append(self.load_check["run_logs"])
        return [value for value in found if value is not None]

    def settings(self) -> dict[str, Any]:
        """Return the settings as the Config cell holds them (``ADAPTERS``).

        Returns:
            Every key of ``ADAPTER_KEYS``, then ``train_log`` when the config sets it; ``arms``
            as mappings (``steps`` only for a polish), ``views`` as lists, ``hypernet`` the
            parsed H config.

        Raises:
            BuildError: If ``resolve_adapters`` has not read the H config yet.
        """
        if self.hypernet_config is None:
            raise BuildError(f"the H config {self.hypernet} was not read (resolve_adapters)")
        arms = []
        for arm in self.arms:
            entry: dict[str, Any] = {"name": arm.name, "source": arm.source}
            if arm.steps is not None:
                entry["steps"] = arm.steps
            entry["reason"] = arm.reason
            arms.append(entry)
        out = {
            "arms": arms,
            "views": None if self.views is None else [list(batch) for batch in self.views],
            "order_seed": self.order_seed,
            "ce": self.ce,
            "finish_s": self.finish_s,
            "checkpoint": self.checkpoint,
            "cache": self.cache,
            "hypernet": dict(self.hypernet_config),
            "episodes_root": self.episodes_root,
            "check_root": self.check_root,
            "manifest_sha256": self.manifest_sha256,
            "load_check": None if self.load_check is None else dict(self.load_check),
            "gen": None if self.gen is None else dict(self.gen),
        }
        if self.train_log is not None:
            out["train_log"] = self.train_log
        return out


class AgentModel(NamedTuple):
    """The weights a hybrid build attaches (task P6.5): a download notebook's output (P6.3).

    ``folder`` holds the weights inside the output; ``revision`` is the commit its
    ``MANIFEST.json`` must name.
    """

    notebook: str
    version: int
    folder: str
    revision: str


class AgentWheels(NamedTuple):
    """The engine wheelhouse a hybrid build attaches (task P6.5): a download notebook's output.

    ``folder`` holds the wheels, their ``MANIFEST.json``, and ``install_wheelhouse.py``;
    ``vllm`` is the version the manifest must list.
    """

    notebook: str
    version: int
    folder: str
    vllm: str


class AgentSettings(NamedTuple):
    """A hybrid build's agent stage (task P6.5, ADR-020), from the config's ``agent``.

    The first six fields are the Config cell's ``AGENT`` (seconds after the notebook's start);
    the rest go into ``BUILD["agent"]``. ``resolve_agent`` fills ``config_data`` with the agent
    run config's parsed content, which the notebook embeds, and ``config_sha256``. ``entry`` is
    the module the stage runs (task P6.5b): ``DEFAULT_AGENT_ENTRY`` or ``SHOT_ENTRY``.
    ``warmup`` (task P6.9c) is a shot build's weight warm-up (``AGENT_WARMUP_KEYS``) or None.
    ``tail`` (task P4.7b) is a shot build's rank-256 tail (``AGENT_TAIL_KEYS``) or None.
    ``skip_nvarc`` (task P5.1): an LLM-first shot build, NVARC not run when the check passes.
    """

    enabled: bool
    nvarc_share_s: int
    end_s: int
    term_s: int
    min_stage_s: int
    install_timeout_s: int
    config: str
    model: AgentModel
    wheels: AgentWheels
    base_python: str
    scratch_dir: str
    min_driver: str
    config_data: dict[str, Any] | None = None
    config_sha256: str | None = None
    entry: str = DEFAULT_AGENT_ENTRY
    warmup: dict[str, Any] | None = None
    tail: dict[str, int] | None = None
    skip_nvarc: bool = False

    @property
    def shot(self) -> bool:
        """Whether the stage runs the long shot (``entry`` is ``SHOT_ENTRY``, task P6.5b)."""
        return self.entry == SHOT_ENTRY

    def stage_settings(self) -> dict[str, Any]:
        """Return the Config cell's ``AGENT``: the switch and the stage's times.

        Returns:
            ``AGENT_STAGE_KEYS`` with their values.
        """
        return {key: getattr(self, key) for key in AGENT_STAGE_KEYS}

    def build_settings(self) -> dict[str, Any]:
        """Return ``BUILD["agent"]``: the run config and its sha256, the inputs, the engine.

        Returns:
            ``config``, ``config_sha256``, ``model``, ``wheels``, ``base_python``,
            ``scratch_dir``, and ``min_driver``; then ``entry`` when it is not the default (task
            P6.5b), ``warmup`` when set (task P6.9c), ``tail`` when set (task P4.7b), and
            ``skip_nvarc`` when true (task P5.1), so every earlier build keeps its bytes.

        Raises:
            BuildError: If ``resolve_agent`` has not read the run config yet.
        """
        if self.config_data is None or self.config_sha256 is None:
            raise BuildError(f"the agent config {self.config} was not read (resolve_agent)")
        settings = {
            "config": self.config,
            "config_sha256": self.config_sha256,
            "model": self.model._asdict(),
            "wheels": self.wheels._asdict(),
            "base_python": self.base_python,
            "scratch_dir": self.scratch_dir,
            "min_driver": self.min_driver,
        }
        if self.entry != DEFAULT_AGENT_ENTRY:
            settings["entry"] = self.entry
        if self.warmup is not None:
            settings["warmup"] = dict(self.warmup)
        if self.tail is not None:
            settings["tail"] = dict(self.tail)
        if self.skip_nvarc:
            settings["skip_nvarc"] = True
        return settings

    def inputs(self) -> dict[str, int]:
        """Return the notebook outputs the build attaches, ``<owner>/<slug>`` to its version.

        Returns:
            The weights' and the wheelhouse's notebooks with their versions.
        """
        return {self.model.notebook: self.model.version, self.wheels.notebook: self.wheels.version}


class SubmissionConfig(NamedTuple):
    """The settings of one submission build, from ``configs/submission_*.yaml``.

    ``arms`` is None for a single run, or the series' arms in launch order, the first one the
    notebook configuration. ``solver`` is the solver's code, one of ``SOLVERS``. ``scope`` (task
    P2.6) is the single run's LoRA scope in both modes, or None for the reference's settings.
    ``adapters`` (task P3.7) is the single run's adapter arms, development runs only, or None;
    ``notebook_inputs`` pins the notebook outputs they read, empty without them, and
    ``dataset_inputs`` (task P3.9b) the datasets they read, empty without them.
    ``agent`` (task P6.5) is the hybrid build's agent stage, in both modes, or None for NVARC
    alone. ``dev240`` (task P6.5b) is ``subset: DEV240``: a development run solves every
    evaluation task twice, each second copy under a derived id (``dev240_files``); ``subset`` is
    then None.
    """

    kaggle_notebook: str
    code_dataset: str
    competition: str
    accelerator: str | None
    reference_metadata: str
    kernel_source_versions: dict[str, int]
    model_slug: str
    subset: tuple[str, ...] | None
    workers: int
    hash_seed: int
    run_solver: bool
    solver_end_s: int
    solver_kill_s: int
    arms: tuple[Arm, ...] | None = None
    solver: str = DEFAULT_SOLVER
    scope: LoraScope | None = None
    adapters: AdapterArms | None = None
    notebook_inputs: dict[str, int] = {}  # never mutated; _parse_submission passes a fresh dict
    agent: AgentSettings | None = None
    dev240: bool = False
    dataset_inputs: dict[str, int] = {}  # never mutated; _parse_submission passes a fresh dict

    @property
    def kind(self) -> str:
        """The config's kind, ``"submission"``; callers dispatch on it, not on the class."""
        return "submission"


class ScriptConfig(NamedTuple):
    """The settings of one script notebook build, from a config with ``kind: script`` (P0.5d).

    ``script`` and ``script_config`` are paths in the repository, under ``scripts/`` and
    ``configs/``; ``args`` follow ``--config <script_config>`` on the script's command line, where
    ``{commit}`` stands for the built commit's SHA and ``{input:<owner>/<slug>}`` for the mount
    folder of an attached notebook output; ``script_kill_s`` is the hard stop in seconds after the
    notebook's start; ``notebook_inputs`` pins the attached notebook outputs, ``<owner>/<slug>`` to
    a version (task P3.5), empty when none is attached.
    """

    kaggle_notebook: str
    code_dataset: str
    competition: str
    accelerator: str | None
    reference_metadata: str
    kernel_source_versions: dict[str, int]
    model_slug: str
    script: str
    script_config: str
    args: tuple[str, ...]
    script_kill_s: int
    notebook_inputs: dict[str, int] = {}  # never mutated; _parse_script passes a fresh dict

    @property
    def kind(self) -> str:
        """The config's kind, ``"script"``; callers dispatch on it, not on the class.

        The tests load this module more than once, so a class check could fail across copies.
        """
        return "script"


BuildConfig = SubmissionConfig | ScriptConfig


def safe_id(value: object) -> bool:
    """Return whether ``value`` is a plain name: letters, digits, ``.``, ``_``, ``-``.

    Why: tags and slugs become folder names and path parts; the gateway applies the same rule to
    run and task ids, so none can leave its folder.

    Args:
        value: The candidate name.

    Returns:
        True for a non-empty string of those characters other than ``.`` and ``..``.
    """
    return isinstance(value, str) and bool(SAFE_ID.fullmatch(value)) and value not in {".", ".."}


def _integer(value: object) -> bool:
    return type(value) is int


def _has_todo(value: Any) -> bool:
    # A YAML escape such as "\x54ODO" spells TODO in a parsed value but not in the text.
    if isinstance(value, str):
        return "TODO" in value
    if isinstance(value, dict):
        return any(_has_todo(key) or _has_todo(item) for key, item in value.items())
    if isinstance(value, list):
        return any(map(_has_todo, value))
    return False


def _owned(raw: dict[str, Any], key: str, min_slug: int, path: Path) -> str:
    value = raw[key]
    match = OWNED_SLUG.fullmatch(value) if isinstance(value, str) else None
    if match is None or not min_slug <= len(match.group(2)) <= 50:
        raise BuildError(
            f"{path}: {key} must be <owner>/<slug>, the slug {min_slug} to 50 lowercase letters, "
            f"digits, and '-'; got {value!r}"
        )
    if match.group(1) != gateway.KAGGLE_OWNER:
        raise BuildError(
            f"{path}: {key} is owned by {match.group(1)!r}, but scripts/kaggle_ops.py acts only "
            f"for KAGGLE_OWNER {gateway.KAGGLE_OWNER!r}"
        )
    return value


def parse_holdout(raw: object, where: str) -> ArmHoldOut | None:
    """Parse and check an arm's ``holdout`` (task P2.1).

    Why: the settings reach the solver as ``arc26.loop.HoldOut`` in every worker, so they are
    checked before anything is built, and spelled out in full in the config: each one changes
    what the arm trains on or decodes.

    Args:
        raw: The parsed value of the arm's ``holdout``; None means no pair is held out.
        where: The config and arm, used in messages.

    Returns:
        The settings, or None.

    Raises:
        BuildError: If ``holdout`` is not null or a mapping with exactly the keys ``rule``,
            ``k2``, ``in_test_prompts``; ``rule`` is not one of ``HOLDOUT_RULES``; or a flag is
            not true or false.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(map(str, raw)) != set(HOLDOUT_KEYS):
        raise BuildError(
            f"{where}: holdout must be null or have exactly the keys {', '.join(HOLDOUT_KEYS)}"
        )
    if raw["rule"] not in HOLDOUT_RULES:
        raise BuildError(
            f"{where}: holdout rule must be one of {', '.join(HOLDOUT_RULES)}; got {raw['rule']!r}"
        )
    for flag in ("k2", "in_test_prompts"):
        if not isinstance(raw[flag], bool):
            raise BuildError(f"{where}: holdout {flag} must be true or false; got {raw[flag]!r}")
    return ArmHoldOut(raw["rule"], raw["k2"], raw["in_test_prompts"])


def parse_rounds(raw: object, where: str, holdout: ArmHoldOut | None) -> ArmRounds | None:
    """Parse and check an arm's ``rounds`` (task P2.3, ADR-015).

    Why: the settings reach every worker as ``arc26.loop.Rounds`` inside the hold-out settings,
    so they are checked before anything is built, with ``Rounds``' own rules, and spelled out in
    full: each one changes what a round may do. A round's gate checks the held-out pair, so the
    arm must hold one out, and with it out of the test inputs' prompts: round 0's test decode
    feeds the round's promotion (ADR-013).

    Args:
        raw: The parsed value of the arm's ``rounds``; None means no round.
        where: The config and arm, used in messages.
        holdout: The arm's hold-out settings.

    Returns:
        The settings, the governor config not yet read (``resolve_rounds``), or None.

    Raises:
        BuildError: If ``rounds`` is not null or a mapping with exactly the keys of
            ``ROUNDS_KEYS`` and optionally ``variants``; the arm holds no pair out or keeps it in
            the test prompts; ``max_rounds`` is not an integer from 1 to ``MAX_ROUNDS``; ``tau``
            is not a number from 0 to 1; ``continue_steps`` is not an integer from 1 to
            ``EPOCH_VIEWS``; ``final_in_prompts`` is not true or false; ``governor`` is not a
            ``.yaml`` path under ``configs/``; or ``variants`` is not a list of ``VARIANTS``,
            one or both, in that order (task P2.5).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(map(str, raw)) - set(ROUNDS_OPTIONAL_KEYS) != set(
        ROUNDS_KEYS
    ):
        raise BuildError(
            f"{where}: rounds must be null or have exactly the keys {', '.join(ROUNDS_KEYS)}, "
            f"and optionally {', '.join(ROUNDS_OPTIONAL_KEYS)}"
        )
    if holdout is None:
        raise BuildError(f"{where}: rounds need the arm's holdout: a round's gate checks that pair")
    if holdout.in_test_prompts:
        raise BuildError(
            f"{where}: rounds need holdout in_test_prompts false: round 0's test decode feeds "
            "their promotion, and no decode that feeds a promotion may hold the pair (ADR-013)"
        )
    max_rounds, tau, steps = raw["max_rounds"], raw["tau"], raw["continue_steps"]
    if not _integer(max_rounds) or not 1 <= max_rounds <= MAX_ROUNDS:
        raise BuildError(
            f"{where}: rounds max_rounds must be 1 to {MAX_ROUNDS}; got {max_rounds!r}"
        )
    if isinstance(tau, bool) or not isinstance(tau, int | float) or not 0 <= tau <= 1:
        raise BuildError(f"{where}: rounds tau must be a number from 0 to 1; got {tau!r}")
    if not _integer(steps) or not 1 <= steps <= EPOCH_VIEWS:
        raise BuildError(
            f"{where}: rounds continue_steps must be 1 to {EPOCH_VIEWS}; got {steps!r}"
        )
    if not isinstance(raw["final_in_prompts"], bool):
        raise BuildError(
            f"{where}: rounds final_in_prompts must be true or false; got "
            f"{raw['final_in_prompts']!r}"
        )
    if not _repo_file(raw["governor"], "configs", (".yaml",)):
        raise BuildError(
            f"{where}: rounds governor must be a .yaml file under configs/; got {raw['governor']!r}"
        )
    variants = raw.get("variants")
    if "variants" in raw:
        if (
            not isinstance(variants, list)
            or not variants
            or not all(isinstance(name, str) and name in VARIANTS for name in variants)
            or len(set(variants)) != len(variants)
            or variants != sorted(variants, key=VARIANTS.index)
        ):
            raise BuildError(
                f"{where}: rounds variants must be a list of {', '.join(VARIANTS)}, one or both "
                f"in that order (the governor's); got {variants!r}"
            )
        variants = tuple(variants)
    return ArmRounds(
        max_rounds,
        float(tau),
        steps,
        raw["final_in_prompts"],
        raw["governor"],
        variants=variants,
    )


def parse_governor(data: bytes, name: str) -> dict[str, Any]:
    """Parse and check a governor config for a round arm (task P2.3).

    Why: the solver reads the governor's config from its JSON, since the wheel carries
    ``arc26/`` only, so the builder embeds the config's content. It checks what a build can
    check without importing the package: a UTF-8 YAML mapping without ``TODO``, holding
    ``GOVERNOR_KEYS``, the ``reference`` decode policy, and JSON values only;
    ``arc26.loop.Rounds`` checks the rest when the notebook builds its ``RunConfig``.

    Args:
        data: The config file's bytes.
        name: Its path in the repository, used in messages.

    Returns:
        The parsed config.

    Raises:
        BuildError: If any check fails.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as err:
        raise BuildError(f"{name} is not UTF-8 text: {err}") from err
    if "TODO" in text:
        raise BuildError(f"{name} still has TODO")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise BuildError(f"{name} is not valid YAML: {err}") from err
    if not isinstance(value, dict):
        raise BuildError(f"{name} must hold a mapping")
    missing = [key for key in GOVERNOR_KEYS if key not in value]
    if missing:
        raise BuildError(f"{name}: a governor config needs {', '.join(missing)}")
    policy = value.get("decode_policy") or {}
    mode = policy.get("mode", GOVERNOR_POLICY) if isinstance(policy, dict) else None
    if mode != GOVERNOR_POLICY:
        raise BuildError(
            f"{name}: decode_policy mode must be {GOVERNOR_POLICY!r} for the rounds, which plan "
            f"after the baseline decode; got {mode!r}"
        )
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise BuildError(f"{name}: the solver JSON cannot carry it: {err}") from err
    return value


def resolve_rounds(config: BuildConfig, read: Callable[[str], bytes]) -> BuildConfig:
    """Read and embed the governor config of every round arm (task P2.3).

    Why: ``parse_config`` sees the build config's bytes only; the governor config is another
    file of the same commit, which ``scripts/build_submission.py`` reads from its export and
    the command line from the checkout.

    Args:
        config: The checked config.
        read: Returns a repository file's bytes by its path.

    Returns:
        The config, its round arms carrying the parsed governor config and its sha256; any other
        config unchanged.

    Raises:
        BuildError: If a governor config cannot be read or fails ``parse_governor``.
    """
    if config.kind != "submission" or not config.arms:
        return config
    arms = []
    for arm in config.arms:
        if arm.rounds is not None:
            try:
                data = read(arm.rounds.governor)
            except OSError as err:
                raise BuildError(f"cannot read {arm.rounds.governor}: {err}") from err
            rounds = arm.rounds._replace(
                governor_config=parse_governor(data, arm.rounds.governor),
                governor_sha256=hashlib.sha256(data).hexdigest(),
            )
            arm = arm._replace(rounds=rounds)
        arms.append(arm)
    return config._replace(arms=tuple(arms))


def parse_arms(raw: object, path: Path) -> tuple[Arm, ...] | None:
    """Parse and check the ``arms`` of a series config (task P1.3).

    Why: each arm becomes a solver launch, folder names, and the solver's LoRA settings, so the
    same rules as ``arc26.runner.RunConfig`` apply before anything is built, plus three that make
    a series readable and its submission meaningful: target names in the reference's order, so
    each scope has one spelling; distinct names; and the notebook configuration first, because
    ``submission.json`` comes from the first arm, unless the first arm holds a pair out (task
    P2.5): its attempts are a loop measurement's, never the reference settings', whatever its
    scope, and a series is never fit to submit (``submission_blocker``). An arm may also hold a
    pair out (task P2.1,
    ``parse_holdout``).

    Args:
        raw: The parsed value of ``arms``; None means no series.
        path: The config's path, used in messages.

    Returns:
        The arms in launch order, or None.

    Raises:
        BuildError: If ``arms`` is not null or a non-empty list of mappings with exactly the keys
            ``name``, ``lora_r``, ``lora_targets``, ``reason``, and optionally ``holdout`` and
            ``rounds`` (``parse_rounds``); a
            name is not 1 to 40 lowercase letters, digits, and ``-`` or repeats; a rank is not a
            positive integer; targets are not distinct names of ``REFERENCE_LORA_TARGETS`` in its
            order with at least one linear projection; a reason is not one line of 1 to
            ``MAX_REASON`` characters; ``holdout`` fails ``parse_holdout``; or the first arm is not
            ``REFERENCE_LORA_R`` on all of ``REFERENCE_LORA_TARGETS`` and holds no pair out.
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise BuildError(f"{path}: arms must be null or a non-empty list of arms")
    arms = []
    for index, item in enumerate(raw, start=1):
        where = f"{path}: arm {index}"
        keys = set(map(str, item)) if isinstance(item, dict) else set()
        if not isinstance(item, dict) or keys - set(ARM_OPTIONAL_KEYS) != set(ARM_KEYS):
            raise BuildError(
                f"{where} must have exactly the keys {', '.join(ARM_KEYS)}, and optionally "
                f"{', '.join(ARM_OPTIONAL_KEYS)}"
            )
        name, rank, targets, reason = (item[key] for key in ARM_KEYS)
        if not isinstance(name, str) or not ARM_NAME.fullmatch(name):
            raise BuildError(
                f"{where}: name must be 1 to 40 lowercase letters, digits, and '-', starting with "
                f"a letter or digit; got {name!r}"
            )
        rank, targets = _lora_scope(rank, targets, where)
        if not isinstance(reason, str) or not reason.strip() or "\n" in reason.strip():
            raise BuildError(f"{where}: reason must be one line of text; got {reason!r}")
        if len(reason.strip()) > MAX_REASON:
            raise BuildError(f"{where}: reason must be at most {MAX_REASON} characters")
        holdout = parse_holdout(item.get("holdout"), where)
        rounds = parse_rounds(item.get("rounds"), where, holdout)
        arms.append(Arm(name, rank, tuple(targets), reason.strip(), holdout, rounds))
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise BuildError(f"{path}: arm names must be distinct; got {names}")
    first = arms[0]
    if first.holdout is None and (first.lora_r, first.lora_targets) != (
        REFERENCE_LORA_R, REFERENCE_LORA_TARGETS
    ):  # fmt: skip
        raise BuildError(
            f"{path}: the first arm must be the notebook configuration, lora_r "
            f"{REFERENCE_LORA_R} on {', '.join(REFERENCE_LORA_TARGETS)}, or hold a pair out: "
            "submission.json comes from the first arm"
        )
    return tuple(arms)


def _lora_scope(rank: object, targets: object, where: str) -> tuple[int, tuple[str, ...]]:
    # An arm's or the build's LoRA scope (tasks P1.3, P2.6), checked with RunConfig's rules and
    # the reference's target order, so each scope has one spelling; returns the rank and the
    # targets as a tuple.
    if not _integer(rank) or rank < 1:
        raise BuildError(f"{where}: lora_r must be a positive integer; got {rank!r}")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(t, str) and t in REFERENCE_LORA_TARGETS for t in targets)
        or len(set(targets)) != len(targets)
        or targets != sorted(targets, key=REFERENCE_LORA_TARGETS.index)
    ):
        raise BuildError(
            f"{where}: lora_targets must be distinct names from "
            f"{', '.join(REFERENCE_LORA_TARGETS)}, in that order; got {targets!r}"
        )
    if not set(targets) & set(LORA_PROJECTIONS):
        raise BuildError(
            f"{where}: lora_targets needs at least one of {', '.join(LORA_PROJECTIONS)}; "
            "Unsloth trains embed_tokens and lm_head as full copies, not as LoRA factors"
        )
    return rank, tuple(targets)


def parse_scope(raw: object, path: Path) -> LoraScope | None:
    """Parse and check a submission config's ``scope`` (task P2.6, ADR-018).

    Why: the scope reaches ``arc26.runner.RunConfig`` for the single run in both modes, a
    competition rerun included, so an arm's rules apply to it before anything is built. The
    reference's own scope is refused: leaving the key out builds exactly that, so a build has one
    spelling and every build without the key stays as it was.

    Args:
        raw: The parsed value of ``scope``; None means the reference's LoRA settings.
        path: The config's path, used in messages.

    Returns:
        The scope, or None.

    Raises:
        BuildError: If ``scope`` is not null or a mapping with exactly the keys ``lora_r`` and
            ``lora_targets``; they break an arm's rules (``parse_arms``); or they are the
            reference's, ``REFERENCE_LORA_R`` on all of ``REFERENCE_LORA_TARGETS``.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(map(str, raw)) != set(SCOPE_KEYS):
        raise BuildError(
            f"{path}: scope must be null or have exactly the keys {', '.join(SCOPE_KEYS)}"
        )
    rank, targets = _lora_scope(raw["lora_r"], raw["lora_targets"], f"{path}: scope")
    if (rank, targets) == (REFERENCE_LORA_R, REFERENCE_LORA_TARGETS):
        raise BuildError(
            f"{path}: scope is the reference's, lora_r {REFERENCE_LORA_R} on all nine targets; "
            "leave it out, which builds exactly that"
        )
    return LoraScope(rank, targets)


def scope_text(scope: LoraScope) -> str:
    """Describe the build's LoRA scope in one line (task P2.6).

    Why: the first cell and ``MANIFEST.md`` state the scope a scoped build trains in both modes,
    in the same words.

    Args:
        scope: The checked scope.

    Returns:
        One line, for example "r = 16 on q_proj, k_proj, v_proj, o_proj (embed_tokens and
        lm_head frozen)".
    """
    copies = [name for name in REFERENCE_LORA_TARGETS if name not in LORA_PROJECTIONS]
    frozen = [name for name in copies if name not in scope.lora_targets]
    text = f"r = {scope.lora_r} on {', '.join(scope.lora_targets)}"
    return f"{text} ({' and '.join(frozen)} frozen)" if frozen else text


def _one_line(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not set(value) & {"\n", "\r", "\0"}


def _check_views(raw: object, where: str) -> tuple[tuple[int, ...], ...] | None:
    # arc26.arms.check_views' rules, checked before anything is built: null, or batches of at
    # most ADAPTER_MAX_BATCH distinct view indices 0 to 15, each batch inside one length group.
    if raw is None:
        return None
    message = (
        f"{where}: views must be null or a non-empty list of batches, each a list of at most "
        f"{ADAPTER_MAX_BATCH} distinct view indices 0 to 15 from one of "
        f"{[list(group) for group in ADAPTER_VIEW_GROUPS]}; got {raw!r}"
    )
    if not isinstance(raw, list) or not raw:
        raise BuildError(message)
    batches = []
    for batch in raw:
        if (
            not isinstance(batch, list)
            or not batch
            or len(batch) > ADAPTER_MAX_BATCH
            or not all(_integer(i) and 0 <= i < 16 for i in batch)
            or not any(set(batch) <= set(group) for group in ADAPTER_VIEW_GROUPS)
        ):
            raise BuildError(message)
        batches.append(tuple(batch))
    flat = [i for batch in batches for i in batch]
    if len(set(flat)) != len(flat):
        raise BuildError(message)
    return tuple(batches)


def parse_adapters(raw: object, path: Path) -> AdapterArms | None:
    """Parse and check a submission config's ``adapters`` (task P3.7, ADR-019).

    Why: the settings reach every worker as ``arc26.arms.Arms``, so ``Arms``' rules apply
    before anything is built, every key spelled out (null where it does not apply): each one
    changes what a puzzle runs or what the run checks first.

    Args:
        raw: The parsed value of ``adapters``; None means one adapter source.
        path: The config's path, used in messages.

    Returns:
        The settings, the H config not yet read (``resolve_adapters``), or None.

    Raises:
        BuildError: If ``adapters`` is not null or a mapping with exactly the keys of
            ``ADAPTER_KEYS``; ``arms`` is not a non-empty list of mappings with exactly the keys
            ``name``, ``source``, ``reason`` and, for a polish, ``steps`` (1 to
            ``EPOCH_VIEWS``), names distinct and folder-safe, reasons one line; ``views`` breaks
            ``_check_views``; ``order_seed`` is not an integer, ``ce`` not true or false,
            ``finish_s`` not an integer from 0; a path is not one line; ``hypernet`` is not a
            ``.yaml`` file under ``configs/``; ``manifest_sha256`` is not 64 hex characters;
            an H arm lacks ``checkpoint``; ``load_check`` is not null or a mapping of
            ``LOAD_CHECK_KEYS`` (positive tolerances) or lacks ``checkpoint``; ``gen`` is not
            null or a mapping of ``GEN_KEYS`` (integers from 0, at least one puzzle); or
            ``gen`` or ``load_check`` is set without ``episodes_root``, ``check_root``, and
            ``manifest_sha256``.
    """
    if raw is None:
        return None
    where = f"{path}: adapters"
    if not isinstance(raw, dict) or set(map(str, raw)) - set(ADAPTER_OPTIONAL_KEYS) != set(
        ADAPTER_KEYS
    ):
        raise BuildError(
            f"{where} must be null or have exactly the keys {', '.join(ADAPTER_KEYS)}, and "
            f"optionally {', '.join(ADAPTER_OPTIONAL_KEYS)}"
        )
    train_log = raw.get("train_log")
    if "train_log" in raw and not isinstance(train_log, bool):
        raise BuildError(f"{where}: train_log must be true or false")
    items = raw["arms"]
    if not isinstance(items, list) or not items:
        raise BuildError(f"{where}: arms must be a non-empty list")
    arms = []
    for index, item in enumerate(items, start=1):
        at = f"{where}: arm {index}"
        keys = set(map(str, item)) if isinstance(item, dict) else set()
        if not isinstance(item, dict) or keys - set(ADAPTER_ARM_OPTIONAL_KEYS) != set(
            ADAPTER_ARM_KEYS
        ):
            raise BuildError(
                f"{at} must have exactly the keys {', '.join(ADAPTER_ARM_KEYS)}, and steps for a "
                "polish"
            )
        name, source, reason = item["name"], item["source"], item["reason"]
        if not isinstance(name, str) or not ARM_NAME.fullmatch(name):
            raise BuildError(f"{at}: name must be 1 to 40 lowercase letters, digits, and '-'")
        if source not in ADAPTER_SOURCES:
            raise BuildError(f"{at}: source must be one of {', '.join(ADAPTER_SOURCES)}")
        steps = item.get("steps")
        if source in ADAPTER_POLISH_SOURCES:
            if not (_integer(steps) and 1 <= steps <= EPOCH_VIEWS):
                raise BuildError(f"{at}: a polish needs steps from 1 to {EPOCH_VIEWS}")
        elif "steps" in item:
            raise BuildError(f"{at}: steps apply to a polish ({', '.join(ADAPTER_POLISH_SOURCES)})")
        if not isinstance(reason, str) or not reason.strip() or "\n" in reason.strip():
            raise BuildError(f"{at}: reason must be one line of text")
        if len(reason.strip()) > MAX_REASON:
            raise BuildError(f"{at}: reason must be at most {MAX_REASON} characters")
        arms.append(AdapterArm(name, source, reason.strip(), steps))
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise BuildError(f"{where}: arm names must be distinct; got {names}")
    views = _check_views(raw["views"], where)
    if not _integer(raw["order_seed"]):
        raise BuildError(f"{where}: order_seed must be an integer")
    if not isinstance(raw["ce"], bool):
        raise BuildError(f"{where}: ce must be true or false")
    finish = raw["finish_s"]
    if not (_integer(finish) and finish >= 0):
        raise BuildError(f"{where}: finish_s must be an integer >= 0")
    for key in ("checkpoint", "cache", "episodes_root", "check_root"):
        if raw[key] is not None and not _one_line(raw[key]):
            raise BuildError(f"{where}: {key} must be null or a one-line path")
    if not _repo_file(raw["hypernet"], "configs", (".yaml",)):
        raise BuildError(f"{where}: hypernet must be a .yaml file under configs/")
    manifest = raw["manifest_sha256"]
    if manifest is not None and not (isinstance(manifest, str) and SHA256.fullmatch(manifest)):
        raise BuildError(f"{where}: manifest_sha256 must be null or 64 lowercase hex characters")
    if any(arm.source in ADAPTER_H_SOURCES for arm in arms) and raw["checkpoint"] is None:
        raise BuildError(f"{where}: an H arm needs checkpoint, H's best.pt")
    check = raw["load_check"]
    if check is not None:
        if not isinstance(check, dict) or set(map(str, check)) != set(LOAD_CHECK_KEYS):
            raise BuildError(
                f"{where}: load_check must be null or have exactly the keys "
                f"{', '.join(LOAD_CHECK_KEYS)}"
            )
        if not _one_line(check["run_logs"]):
            raise BuildError(f"{where}: load_check run_logs must be a one-line path")
        for key in ("max_abs", "max_mean"):
            value = check[key]
            if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
                raise BuildError(f"{where}: load_check {key} must be a positive number")
        if raw["checkpoint"] is None:
            raise BuildError(f"{where}: load_check needs checkpoint")
        check = dict(check)
    gen = raw["gen"]
    if gen is not None:
        if not isinstance(gen, dict) or set(map(str, gen)) != set(GEN_KEYS):
            raise BuildError(
                f"{where}: gen must be null or have exactly the keys {', '.join(GEN_KEYS)}"
            )
        if not all(_integer(gen[key]) and gen[key] >= 0 for key in GEN_KEYS) or not sum(
            gen[key] for key in GEN_KEYS
        ):
            raise BuildError(f"{where}: gen barc and rearc must be integers >= 0, one positive")
        gen = dict(gen)
    if (check is not None or gen is not None) and None in (
        raw["episodes_root"], raw["check_root"], manifest
    ):  # fmt: skip
        raise BuildError(
            f"{where}: gen and load_check need episodes_root, check_root, and manifest_sha256"
        )
    return AdapterArms(
        arms=tuple(arms),
        views=views,
        order_seed=raw["order_seed"],
        ce=raw["ce"],
        finish_s=finish,
        checkpoint=raw["checkpoint"],
        cache=raw["cache"],
        hypernet=raw["hypernet"],
        episodes_root=raw["episodes_root"],
        check_root=raw["check_root"],
        manifest_sha256=manifest,
        load_check=check,
        gen=gen,
        train_log=train_log,
    )


def parse_hypernet(data: bytes, name: str) -> dict[str, Any]:
    """Parse and check the expected H config of an adapter build (task P3.7).

    Why: the load check and every worker compare the checkpoint's H config with it, so the
    builder embeds the parsed file of the built commit (the wheel carries ``arc26/`` only).

    Args:
        data: The file's bytes (``configs/hypernet.yaml``).
        name: Its path in the repository, used in messages.

    Returns:
        The mapping of ``HYPERNET_KEYS``, ``target_modules`` a list, then ``aggregator`` with
        every key of ``AGGREGATOR_DEFAULTS`` when the file's block differs from them (task
        P3.3b; the form a checkpoint's config stores).

    Raises:
        BuildError: If it is not UTF-8 YAML without ``TODO`` holding exactly ``HYPERNET_KEYS``
            and optionally ``aggregator``, a mapping of some of ``AGGREGATOR_DEFAULTS``' keys
            whose values ``AggregatorConfig`` accepts (a kind of ``AGGREGATOR_KINDS``, positive
            integer sizes, ``heads`` dividing ``width``).
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as err:
        raise BuildError(f"{name} is not UTF-8 text: {err}") from err
    if "TODO" in text:
        raise BuildError(f"{name} still has TODO")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise BuildError(f"{name} is not valid YAML: {err}") from err
    if not isinstance(value, dict) or set(map(str, value)) - set(HYPERNET_OPTIONAL_KEYS) != set(
        HYPERNET_KEYS
    ):
        raise BuildError(
            f"{name} must hold exactly the keys {', '.join(HYPERNET_KEYS)}, and optionally "
            f"{', '.join(HYPERNET_OPTIONAL_KEYS)}"
        )
    aggregator = value.get("aggregator", {})
    if not isinstance(aggregator, dict) or not set(map(str, aggregator)) <= set(
        AGGREGATOR_DEFAULTS
    ):
        raise BuildError(
            f"{name}: aggregator must be a mapping of some of the keys "
            f"{', '.join(AGGREGATOR_DEFAULTS)}"
        )
    targets = value["target_modules"]
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise BuildError(f"{name}: target_modules must be a list of module names")
    out = {key: value[key] for key in HYPERNET_KEYS}
    aggregator = {**AGGREGATOR_DEFAULTS, **aggregator}
    # AggregatorConfig's own rules, so a value H would refuse fails here, not on a GPU.
    if aggregator["kind"] not in AGGREGATOR_KINDS:
        raise BuildError(
            f"{name}: aggregator kind must be one of {', '.join(AGGREGATOR_KINDS)}; got "
            f"{aggregator['kind']!r}"
        )
    for key in ("width", "blocks", "heads"):
        if not (_integer(aggregator[key]) and aggregator[key] > 0):
            raise BuildError(
                f"{name}: aggregator {key} must be a positive integer; got {aggregator[key]!r}"
            )
    if aggregator["width"] % aggregator["heads"]:
        raise BuildError(
            f"{name}: aggregator heads {aggregator['heads']} must divide its width "
            f"{aggregator['width']}"
        )
    if aggregator != AGGREGATOR_DEFAULTS:
        out["aggregator"] = aggregator
    return out


def resolve_adapters(config: BuildConfig, read: Callable[[str], bytes]) -> BuildConfig:
    """Read and embed an adapter build's expected H config, and check the scope against it.

    Why: H's adapter is swapped into the live adapter only at H's scope (ADR-017): the build's
    ``scope`` must be H's rank and targets, and ``arc26.solver.load``'s alpha H's.

    Args:
        config: The checked config.
        read: Returns a repository file's bytes by its path.

    Returns:
        The config, its adapters carrying the parsed H config and its sha256; any other config
        unchanged.

    Raises:
        BuildError: If the H config cannot be read, fails ``parse_hypernet``, or its scope is
            not the build's.
    """
    if config.kind != "submission" or config.adapters is None:
        return config
    adapters = config.adapters
    try:
        data = read(adapters.hypernet)
    except OSError as err:
        raise BuildError(f"cannot read {adapters.hypernet}: {err}") from err
    hypernet = parse_hypernet(data, adapters.hypernet)
    scope = config.scope
    if scope is None or (scope.lora_r, list(scope.lora_targets)) != (
        hypernet["rank"], list(hypernet["target_modules"])
    ) or hypernet["alpha"] != SOLVER_LORA_ALPHA:  # fmt: skip
        raise BuildError(
            f"{adapters.hypernet}: H emits rank {hypernet['rank']} on "
            f"{hypernet['target_modules']} with alpha {hypernet['alpha']}; the build's scope must "
            f"be the same, with arc26.solver.load's alpha {SOLVER_LORA_ALPHA} (ADR-017)"
        )
    return config._replace(
        adapters=adapters._replace(
            hypernet_config=hypernet, hypernet_sha256=hashlib.sha256(data).hexdigest()
        )
    )


def _agent_input(raw: object, keys: tuple[str, ...], where: str) -> dict[str, Any]:
    # One download notebook's output of a hybrid build (task P6.5): <owner>/<slug> at a pinned
    # version, the folder inside it, and its pin; the caller checks the pin.
    if not isinstance(raw, dict) or set(map(str, raw)) != set(keys):
        raise BuildError(f"{where} must have exactly the keys {', '.join(keys)}")
    notebook, version, folder = raw["notebook"], raw["version"], raw["folder"]
    match = OWNED_SLUG.fullmatch(notebook) if isinstance(notebook, str) else None
    if match is None or not 5 <= len(match.group(2)) <= 50:
        raise BuildError(
            f"{where}: notebook must be <owner>/<slug>, the slug 5 to 50 lowercase letters, "
            f"digits, and '-'; got {notebook!r}"
        )
    if not _integer(version) or version < 1:
        raise BuildError(f"{where}: version must be an integer >= 1; got {version!r}")
    if not safe_id(folder):
        raise BuildError(f"{where}: folder must be one folder name; got {folder!r}")
    return dict(raw)


def _absolute_path(value: object) -> bool:
    # An absolute POSIX path whose every part is a plain name (no "", ".", "..", spaces).
    return (
        isinstance(value, str)
        and value.startswith("/")
        and len(value) > 1
        and all(map(safe_id, value[1:].split("/")))
    )


def parse_agent(raw: object, stack: Mapping[str, Any], path: Path) -> AgentSettings | None:
    """Parse and check a submission config's ``agent`` (task P6.5, ADR-020).

    Why: the settings reach the competition rerun of the build, so each is spelled out and
    checked before anything is built: NVARC's share and the agent's end must leave the agent
    stage time and the notebook its final check before Kaggle's 12 h limit; the two download
    outputs must be other notebooks than the build's own, pinned at a version; the scratch folder
    must be outside the notebook's output and inputs. ``resolve_agent`` then reads the agent's
    run config and checks the plan against it.

    Args:
        raw: The parsed value of ``agent``; None means no agent stage.
        stack: The keys every config shares, checked (``_parse_stack``).
        path: The config's path, used in messages.

    Returns:
        The settings, the run config not yet read, or None.

    A shot build (``entry: arc26.agent.shot``, task P6.5b) keeps NVARC's deadlines: its share
    must be ``COMPETITION_SOLVER_END_S``, so ``nvarc_deadlines`` never shortens them, and the
    shot runs only in the time NVARC leaves (``shot_plan``).

    Raises:
        BuildError: If ``agent`` is not null or a mapping with exactly ``AGENT_KEYS`` (and
            optionally ``AGENT_OPTIONAL_KEYS``); or a value breaks its rule (see
            ``configs/submission_P6.5-01.yaml`` and ``configs/submission_P6.5-03.yaml``).
    """
    if raw is None:
        return None
    where = f"{path}: agent"
    keys = set(map(str, raw)) if isinstance(raw, dict) else set()
    allowed = {*AGENT_KEYS, *AGENT_OPTIONAL_KEYS}
    if not isinstance(raw, dict) or not set(AGENT_KEYS) <= keys <= allowed:
        raise BuildError(
            f"{where} must be null or have exactly the keys {', '.join(AGENT_KEYS)} (and "
            f"optionally {', '.join(AGENT_OPTIONAL_KEYS)})"
        )
    entry = raw.get("entry", DEFAULT_AGENT_ENTRY)
    if entry not in AGENT_ENTRIES:
        raise BuildError(
            f"{where}: entry must be one of {', '.join(AGENT_ENTRIES)} (the module the agent "
            f"stage runs); got {entry!r}"
        )
    if not isinstance(raw["enabled"], bool):
        raise BuildError(f"{where}: enabled must be true or false; got {raw['enabled']!r}")
    warmup = _agent_warmup(raw.get("warmup"), entry, where)
    share = raw["nvarc_share_s"]
    if not _integer(share) or not (
        share == 0 or MIN_SOLVER_END_S <= share <= COMPETITION_SOLVER_END_S
    ):
        raise BuildError(
            f"{where}: nvarc_share_s must be 0 (agent only) or an integer from "
            f"{MIN_SOLVER_END_S} to {COMPETITION_SOLVER_END_S}, the competition's soft deadline; "
            f"got {share!r}"
        )
    if entry == SHOT_ENTRY and share != COMPETITION_SOLVER_END_S:
        raise BuildError(
            f"{where}: a shot build ({SHOT_ENTRY}) keeps NVARC's deadlines, so nvarc_share_s "
            f"must be {COMPETITION_SOLVER_END_S}, the competition's soft deadline: the shot runs "
            f"only in the time NVARC leaves; got {share!r}"
        )
    term = raw["term_s"]
    if not _integer(term) or not MIN_AGENT_TERM_S <= term <= MAX_AGENT_TERM_S:
        raise BuildError(
            f"{where}: term_s must be an integer from {MIN_AGENT_TERM_S} to {MAX_AGENT_TERM_S}; "
            f"got {term!r}"
        )
    end = raw["end_s"]
    latest = SESSION_LIMIT_S - WRITE_RESERVE_S - AGENT_KILL_WAIT_S - term
    if not _integer(end) or not MIN_SOLVER_END_S <= end <= latest:
        raise BuildError(
            f"{where}: end_s must be an integer from {MIN_SOLVER_END_S} to {latest}: SIGKILL "
            f"follows term_s later, the process group gets {AGENT_KILL_WAIT_S} s to die, and "
            f"{WRITE_RESERVE_S} s stay for the final check and the logs before Kaggle's 12 h "
            f"limit; got {end!r}"
        )
    stage = raw["min_stage_s"]
    if not _integer(stage) or not MIN_AGENT_STAGE_S <= stage <= SESSION_LIMIT_S:
        raise BuildError(
            f"{where}: min_stage_s must be an integer from {MIN_AGENT_STAGE_S} to "
            f"{SESSION_LIMIT_S}; got {stage!r}"
        )
    install = raw["install_timeout_s"]
    if not _integer(install) or not MIN_INSTALL_TIMEOUT_S <= install <= MAX_INSTALL_TIMEOUT_S:
        raise BuildError(
            f"{where}: install_timeout_s must be an integer from {MIN_INSTALL_TIMEOUT_S} to "
            f"{MAX_INSTALL_TIMEOUT_S}; got {install!r}"
        )
    if not _repo_file(raw["config"], "configs", (".yaml",)):
        raise BuildError(
            f"{where}: config must be a .yaml file under configs/; got {raw['config']!r}"
        )
    model = _agent_input(raw["model"], AGENT_MODEL_KEYS, f"{where} model")
    if not isinstance(model["revision"], str) or not REVISION.fullmatch(model["revision"]):
        raise BuildError(
            f"{where} model: revision must be a full commit SHA, 40 lowercase hex digits; got "
            f"{model['revision']!r}"
        )
    wheels = _agent_input(raw["wheels"], AGENT_WHEELS_KEYS, f"{where} wheels")
    if not isinstance(wheels["vllm"], str) or not DOTTED_VERSION.fullmatch(wheels["vllm"]):
        raise BuildError(
            f"{where} wheels: vllm must be a version such as \"0.31.0\" (quoted when it has one "
            f"dot); got {wheels['vllm']!r}"
        )
    notebooks = [model["notebook"], wheels["notebook"]]
    if notebooks[0] == notebooks[1]:
        raise BuildError(
            f"{where}: model and wheels must be two notebooks; both are {notebooks[0]}"
        )
    if stack["kaggle_notebook"] in notebooks:
        raise BuildError(
            f"{where}: {stack['kaggle_notebook']} is the notebook itself; Kaggle refuses a "
            "notebook's own output (invalidKernelSources)"
        )
    pinned = sorted(set(notebooks) & set(stack["kernel_source_versions"]))
    if pinned:
        raise BuildError(
            f"{where}: {pinned} in both kernel_source_versions, the reference's own kernel "
            "sources, and the agent's inputs"
        )
    if not _absolute_path(raw["base_python"]):
        raise BuildError(
            f"{where}: base_python must be an absolute path; got {raw['base_python']!r}"
        )
    scratch = raw["scratch_dir"]
    if not _absolute_path(scratch) or any(
        PurePosixPath(scratch).is_relative_to(refused) for refused in AGENT_SCRATCH_REFUSED
    ):
        raise BuildError(
            f"{where}: scratch_dir must be an absolute path outside "
            f"{' and '.join(AGENT_SCRATCH_REFUSED)}; got {scratch!r}"
        )
    driver = raw["min_driver"]
    if not isinstance(driver, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver):
        raise BuildError(
            f"{where}: min_driver must be a driver version such as \"580.65.06\"; got {driver!r}"
        )
    tail = _agent_tail(raw.get("tail"), entry, end, where)
    skip = raw.get("skip_nvarc", False)
    if not isinstance(skip, bool):
        raise BuildError(f"{where}: skip_nvarc must be true or false; got {skip!r}")
    if skip and entry != SHOT_ENTRY:
        raise BuildError(f"{where}: skip_nvarc needs entry {SHOT_ENTRY} (a shot build)")
    if skip and tail is not None:
        raise BuildError(
            f"{where}: skip_nvarc and tail exclude each other: the tail runs NVARC's solver on "
            "the tasks rank 16 left at confidence 0, and without NVARC there is no rank 16"
        )
    return AgentSettings(
        enabled=raw["enabled"],
        nvarc_share_s=share,
        end_s=end,
        term_s=term,
        min_stage_s=stage,
        install_timeout_s=install,
        config=raw["config"],
        model=AgentModel(**model),
        wheels=AgentWheels(**wheels),
        base_python=raw["base_python"],
        scratch_dir=scratch,
        min_driver=driver,
        entry=entry,
        warmup=warmup,
        tail=tail,
        skip_nvarc=skip,
    )


def _agent_tail(raw: Any, entry: str, end_s: int, where: str) -> dict[str, int] | None:
    """Check a shot build's optional ``agent.tail`` (task P4.7b) and return it, or None.

    Why: the tail's solver must stop early enough for its selection, the merge, and the write
    before the agent stage's end ``end_s``, so the notebook's final check and logs keep the shot
    build's reserve after it (``tail_deadlines``).

    Args:
        raw: The parsed value of ``agent.tail``; absent or null means no tail.
        entry: The agent stage's module; the tail needs ``SHOT_ENTRY``.
        end_s: The agent stage's end, ``agent.end_s``.
        where: The config's ``agent`` section, used in messages.

    Returns:
        The settings with the keys in ``AGENT_TAIL_KEYS`` order, or None.

    Raises:
        BuildError: If the build is not a shot build, the keys are not exactly
            ``AGENT_TAIL_KEYS``, a number is out of ``AGENT_TAIL_BOUNDS``, or the tail's soft
            deadline would come before ``MIN_SOLVER_END_S``.
    """
    if raw is None:
        return None
    if entry != SHOT_ENTRY:
        raise BuildError(f"{where}: tail needs entry {SHOT_ENTRY} (a shot build)")
    if not isinstance(raw, dict) or set(map(str, raw)) != set(AGENT_TAIL_KEYS):
        raise BuildError(f"{where} tail must have exactly the keys {', '.join(AGENT_TAIL_KEYS)}")
    for key, least, most in AGENT_TAIL_BOUNDS:
        if not _integer(raw[key]) or not least <= raw[key] <= most:
            raise BuildError(
                f"{where} tail: {key} must be an integer from {least} to {most}; got {raw[key]!r}"
            )
    if end_s - raw["write_s"] - raw["kill_gap_s"] < MIN_SOLVER_END_S:
        raise BuildError(
            f"{where} tail: end_s less write_s and kill_gap_s, the tail's soft deadline, must be "
            f"at least {MIN_SOLVER_END_S} s"
        )
    return {key: raw[key] for key in AGENT_TAIL_KEYS}


def _agent_warmup(raw: Any, entry: str, where: str) -> dict[str, Any] | None:
    """Check a shot build's optional ``agent.warmup`` (task P6.9c) and return it, or None.

    Args:
        raw: The parsed value of ``agent.warmup``; absent or null means no warm-up.
        entry: The agent stage's module; the warm-up needs ``SHOT_ENTRY``.
        where: The config's ``agent`` section, used in messages.

    Returns:
        The settings with the keys in ``AGENT_WARMUP_KEYS`` order, or None.

    Raises:
        BuildError: If the build is not a shot build, the keys are not exactly
            ``AGENT_WARMUP_KEYS``, a number is out of ``AGENT_WARMUP_BOUNDS``, or the fallback is
            not one of ``AGENT_WARMUP_LOAD_STRATEGIES``.
    """
    if raw is None:
        return None
    if entry != SHOT_ENTRY:
        raise BuildError(f"{where}: warmup needs entry {SHOT_ENTRY} (a shot build)")
    if not isinstance(raw, dict) or set(map(str, raw)) != set(AGENT_WARMUP_KEYS):
        raise BuildError(
            f"{where} warmup must have exactly the keys {', '.join(AGENT_WARMUP_KEYS)}"
        )
    for key, least, most in AGENT_WARMUP_BOUNDS:
        if not _integer(raw[key]) or not least <= raw[key] <= most:
            raise BuildError(
                f"{where} warmup: {key} must be an integer from {least} to {most}; got "
                f"{raw[key]!r}"
            )
    if raw["fallback_load_strategy"] not in AGENT_WARMUP_LOAD_STRATEGIES:
        raise BuildError(
            f"{where} warmup: fallback_load_strategy must be one of "
            f"{', '.join(AGENT_WARMUP_LOAD_STRATEGIES)} (vLLM 0.31.0's "
            f"--safetensors-load-strategy); got {raw['fallback_load_strategy']!r}"
        )
    return {key: raw[key] for key in AGENT_WARMUP_KEYS}


def parse_agent_config(data: bytes, name: str) -> dict[str, Any]:
    """Parse and check the agent's run config for a hybrid build (task P6.5).

    Why: the notebook embeds it and writes it to scratch as JSON for ``python -m arc26.agent.run
    --config``, since the wheel carries ``arc26/`` only; ``arc26.agent.run`` checks every key
    when it starts. The builder checks what the notebook itself reads (the deadline margins and
    the engine's tensor parallelism, boot, and stop times) and that the JSON copy reads back
    through ``yaml.safe_load`` as the same mapping (YAML 1.1 reads ``1e-05`` as a string, for
    one).

    Args:
        data: The config file's bytes.
        name: Its path in the repository, used in messages.

    Returns:
        The parsed config.

    Raises:
        BuildError: If any check fails.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as err:
        raise BuildError(f"{name} is not UTF-8 text: {err}") from err
    if "TODO" in text:
        raise BuildError(f"{name} still has TODO")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise BuildError(f"{name} is not valid YAML: {err}") from err
    if not isinstance(value, dict):
        raise BuildError(f"{name} must hold a mapping")
    for section, keys in (
        ("deadline", AGENT_CONFIG_DEADLINE_KEYS), ("server", AGENT_CONFIG_SERVER_KEYS)
    ):  # fmt: skip
        part = value.get(section)
        if not isinstance(part, dict) or not all(key in part for key in keys):
            raise BuildError(f"{name}: the agent stage needs {section} with {', '.join(keys)}")

    def number(section: str, key: str, positive: bool) -> None:
        item = value[section][key]
        if isinstance(item, bool) or not isinstance(item, int | float) or not (
            item > 0 if positive else item >= 0
        ):
            sign = "a positive number" if positive else "a number >= 0"
            raise BuildError(f"{name}: {section} {key} must be {sign}; got {item!r}")

    for key in AGENT_CONFIG_DEADLINE_KEYS:
        number("deadline", key, False)
    for key in ("boot_timeout_s", "stop_timeout_s"):
        number("server", key, True)
    parallel = value["server"]["tensor_parallel_size"]
    if not _integer(parallel) or parallel < 1:
        raise BuildError(f"{name}: server tensor_parallel_size must be an integer >= 1")
    try:
        copy = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise BuildError(f"{name}: its JSON copy cannot be written: {err}") from err
    if yaml.safe_load(copy) != value:
        raise BuildError(
            f"{name}: its JSON copy reads back differently through yaml.safe_load; write the "
            "value that differs another way (a float with a decimal point, say)"
        )
    return value


def resolve_agent(config: BuildConfig, read: Callable[[str], bytes]) -> BuildConfig:
    """Read and embed a hybrid build's agent run config, and check its plan (task P6.5).

    Why: ``parse_config`` sees the build config's bytes only; the agent's run config is another
    file of the same commit, which ``scripts/build_submission.py`` reads from its export and the
    command line from the checkout, as ``resolve_rounds`` reads a governor config. Its values
    decide whether the plan works: the engine's tensor parallelism must fit the accelerator, the
    agent's SIGTERM grace must cover its engine's stop, and ``agent_plan`` must find no problem.

    Args:
        config: The checked config.
        read: Returns a repository file's bytes by its path.

    Returns:
        The config, its agent carrying the parsed run config and its sha256; any other config
        unchanged.

    Raises:
        BuildError: If the run config cannot be read, fails ``parse_agent_config``, or breaks
            the plan.
    """
    if config.kind != "submission" or config.agent is None:
        return config
    agent = config.agent
    try:
        data = read(agent.config)
    except OSError as err:
        raise BuildError(f"cannot read {agent.config}: {err}") from err
    value = parse_agent_config(data, agent.config)
    # An LLM-first build (task P5.1) queues every task by its opening: no NVARC, no confidence.
    order = (value.get("selection") or {}).get("order")
    if agent.skip_nvarc and order != SKIP_NVARC_ORDER:
        raise BuildError(
            f"{agent.config}: an LLM-first build (agent.skip_nvarc) needs selection.order "
            f"{SKIP_NVARC_ORDER}: without NVARC no task has a confidence; got {order!r}"
        )
    if not agent.skip_nvarc and order == SKIP_NVARC_ORDER:
        raise BuildError(
            f"{agent.config}: selection.order {SKIP_NVARC_ORDER} ignores NVARC's confidence, so "
            "it needs agent.skip_nvarc (an LLM-first build)"
        )
    server = value["server"]
    parallel, gpus = server["tensor_parallel_size"], GPU_COUNTS.get(config.accelerator)
    if gpus is not None and parallel > gpus:
        raise BuildError(
            f"{agent.config}: the engine's tensor_parallel_size {parallel} needs {parallel} GPUs, "
            f"but {config.accelerator} has {gpus}"
        )
    if agent.term_s <= server["stop_timeout_s"] + AGENT_KILL_WAIT_S:
        raise BuildError(
            f"agent term_s {agent.term_s} must exceed the engine's stop_timeout_s "
            f"({server['stop_timeout_s']} s, {agent.config}) plus {AGENT_KILL_WAIT_S} s, so the "
            "agent, given SIGTERM, can stop its engine and write before SIGKILL"
        )
    # A shot build (task P6.5b) runs in the time NVARC leaves: its plan has no worst-case start.
    plan = (shot_plan if agent.shot else agent_plan)(
        agent.stage_settings(),
        value["deadline"]["final_margin_s"],
        server["boot_timeout_s"],
        NVARC_KILL_GAP_S,
        WRITE_RESERVE_S,
        AGENT_KILL_WAIT_S,
        # A shot build plans against the session less the pre-script allowance (task P6.5b).
        SHOT_SESSION_S if agent.shot else SESSION_LIMIT_S,
    )
    if plan["problems"]:
        raise BuildError("agent: " + "; ".join(plan["problems"]))
    resolved = agent._replace(config_data=value, config_sha256=hashlib.sha256(data).hexdigest())
    return config._replace(agent=resolved)


def tail_text(tail: Mapping[str, int]) -> str:
    """Return a tail build's settings in words, for the first cell and ``MANIFEST.md`` (P4.7b).

    Args:
        tail: ``agent.tail``.

    Returns:
        One phrase.
    """
    return (
        f"the solver's soft deadline {tail['write_s']} + {tail['kill_gap_s']} s before the agent "
        f"stage's end, its hard stop {tail['kill_gap_s']} s after it, skipped with less than "
        f"{tail['min_s']} s before the soft deadline"
    )


def agent_text(agent: AgentSettings) -> str:
    """Describe a hybrid build's agent stage in one line (task P6.5).

    Why: the first cell and ``MANIFEST.md`` state the session's plan in the same words.

    Args:
        agent: The resolved settings (``resolve_agent``).

    Returns:
        One line, for example "NVARC until 21600 s (its hard stop at 22200 s), then the agent
        until 42000 s: generation stops at 40200 s, SIGTERM at 42000 s, SIGKILL at 42120 s"; an
        agent-only build (a share of 0) names NVARC's fallback instead, and a shot build (task
        P6.5b) the latest start that still runs the shot.

    Raises:
        BuildError: If the run config has not been read.
    """
    if agent.config_data is None:
        raise BuildError(f"the agent config {agent.config} was not read (resolve_agent)")
    final = agent.config_data["deadline"]["final_margin_s"]
    if agent.skip_nvarc:
        stop = agent.end_s - final
        return (
            f"no NVARC when the agent check passes (BUILD['agent']['skip_nvarc']), then the long "
            f"shot ({SHOT_ENTRY}) on every task, the smallest opening first, until "
            f"{agent.end_s} s: generation stops at {stop:g} s; SIGTERM at {agent.end_s} s, "
            f"SIGKILL at {agent.end_s + agent.term_s} s; a failed agent check runs NVARC alone "
            "with its full deadlines"
        )
    if agent.shot:
        stop = agent.end_s - final
        return (
            f"NVARC with its own deadlines (its soft deadline at most {agent.nvarc_share_s} s, "
            f"its hard stop at most {agent.nvarc_share_s + NVARC_KILL_GAP_S} s), then the long "
            f"shot ({SHOT_ENTRY}) in the time NVARC leaves, until {agent.end_s} s: it runs only "
            f"when its stage starts by {stop - agent.min_stage_s:g} s (min_stage_s "
            f"{agent.min_stage_s} s before the generation stop at {stop:g} s); SIGTERM at "
            f"{agent.end_s} s, SIGKILL at {agent.end_s + agent.term_s} s"
        )
    stage = (
        f"the agent until {agent.end_s} s: generation stops at {agent.end_s - final:g} s, "
        f"SIGTERM at {agent.end_s} s, SIGKILL at {agent.end_s + agent.term_s} s"
    )
    if agent.nvarc_share_s == 0:
        return (
            f"agent only, {stage}; NVARC runs only as the fallback, when the agent stage ends "
            f"at least {NVARC_FALLBACK_MIN_S} s before NVARC's soft deadline"
        )
    return (
        f"NVARC until {agent.nvarc_share_s} s (its hard stop at "
        f"{agent.nvarc_share_s + NVARC_KILL_GAP_S} s), then {stage}"
    )


def _repo_file(value: object, folder: str, suffixes: tuple[str, ...]) -> bool:
    # A relative POSIX path under folder/, every part a plain name (no "", ".", "..", "\\").
    if not isinstance(value, str) or not value.endswith(suffixes):
        return False
    parts = value.split("/")
    return len(parts) >= 2 and parts[0] == folder and all(map(safe_id, parts))


def _parse_script(raw: dict[str, Any], stack: dict[str, Any], path: Path) -> ScriptConfig:
    # The script keys of a kind: script config; parse_config has checked the keys and the stack.
    if stack["kaggle_notebook"] == SUBMISSION_NOTEBOOK:
        raise BuildError(
            f"{path}: a script notebook is never submitted, so it cannot be "
            f"{SUBMISSION_NOTEBOOK}, the notebook submissions come from"
        )
    script, script_config = raw["script"], raw["script_config"]
    if not _repo_file(script, "scripts", (".py",)):
        raise BuildError(
            f"{path}: script must be a relative path to a .py file under scripts/; got {script!r}"
        )
    if not _repo_file(script_config, "configs", (".yaml", ".yml")):
        raise BuildError(
            f"{path}: script_config must be a relative path to a .yaml file under configs/; got "
            f"{script_config!r}"
        )
    args = raw["args"]
    if (
        not isinstance(args, list)
        or len(args) > MAX_ARGS
        or not all(isinstance(a, str) and a and not set(a) & {"\n", "\r", "\0"} for a in args)
    ):
        raise BuildError(
            f"{path}: args must be a list of at most {MAX_ARGS} non-empty one-line strings; got "
            f"{args!r}"
        )
    override = config_override(args)
    if override is not None:
        raise BuildError(
            f"{path}: args must not set --config ({override!r}): the notebook passes "
            "script_config, the file whose sha256 it checks"
        )
    kill = raw["script_kill_s"]
    latest = SESSION_LIMIT_S - WRITE_RESERVE_S
    if not (_integer(kill) and MIN_SCRIPT_KILL_S <= kill <= latest):
        raise BuildError(
            f"{path}: script_kill_s must be an integer from {MIN_SCRIPT_KILL_S} to {latest} (12 h "
            f"minus {WRITE_RESERVE_S} s to write the logs); got {kill!r}"
        )
    return ScriptConfig(
        **stack,
        script=script,
        script_config=script_config,
        args=tuple(args),
        script_kill_s=kill,
        notebook_inputs=_parse_notebook_inputs(raw.get("notebook_inputs"), stack, args, path),
    )


def _pinned_inputs(raw: object, key: str, what: str, path: Path) -> dict[str, int]:
    # A mapping of pinned Kaggle sources, <owner>/<slug> to a version >= 1 (notebook_inputs,
    # task P3.5; dataset_inputs, task P3.9b); absent or null, none.
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or len(raw) > MAX_NOTEBOOK_INPUTS or not all(
        isinstance(name, str)
        and (match := OWNED_SLUG.fullmatch(name)) is not None
        and 5 <= len(match.group(2)) <= 50
        and _integer(version)
        and version >= 1
        for name, version in raw.items()
    ):
        raise BuildError(
            f"{path}: {key} must map at most {MAX_NOTEBOOK_INPUTS} {what} "
            "<owner>/<slug> (the slug 5 to 50 lowercase letters, digits, and '-') to a version "
            f"number >= 1; got {raw!r}"
        )
    return dict(raw)


def _parse_notebook_inputs(
    raw: object,
    stack: dict[str, Any],
    args: list[str],
    path: Path,
    reader: tuple[str, str] = ("args", "the script"),
    datasets: dict[str, int] | None = None,
) -> dict[str, int]:
    # notebook_inputs of a script config (task P3.5): pinned outputs of other notebooks, each
    # passed to the script as "{input:<owner>/<slug>}" in args. An adapter build (task P3.7)
    # passes its adapters' paths as args, and reader names them in messages; its datasets (task
    # P3.9b, already checked by _pinned_inputs) are named by the same placeholder.
    raw = _pinned_inputs(raw, "notebook_inputs", "notebooks", path)
    datasets = datasets or {}
    if len(raw) + len(datasets) > MAX_NOTEBOOK_INPUTS:
        raise BuildError(
            f"{path}: notebook_inputs and dataset_inputs attach {len(raw) + len(datasets)} "
            f"sources, at most {MAX_NOTEBOOK_INPUTS}"
        )
    both = sorted(set(raw) & set(datasets))
    if both:
        raise BuildError(
            f"{path}: {both} in both notebook_inputs and dataset_inputs: a slug is a notebook or a "
            "dataset, and {input:<owner>/<slug>} must name one mount"
        )
    if stack["code_dataset"] in datasets:
        raise BuildError(
            f"{path}: dataset_inputs attaches {stack['code_dataset']}, the code dataset, which "
            "the build attaches unversioned and checks by its wheel's sha256"
        )
    if stack["kaggle_notebook"] in raw:
        raise BuildError(
            f"{path}: notebook_inputs attaches {stack['kaggle_notebook']}, the notebook itself; "
            "Kaggle refuses a notebook's own output (invalidKernelSources), so resume from a "
            "notebook of another slug"
        )
    pinned = sorted(set(raw) & set(stack["kernel_source_versions"]))
    if pinned:
        raise BuildError(
            f"{path}: {pinned} in both kernel_source_versions, the reference's own kernel sources, "
            "and notebook_inputs"
        )
    named = {name for arg in args for name in INPUT_PLACEHOLDER.findall(arg)}
    unknown = sorted(named - set(raw) - set(datasets))
    if unknown:
        attaching = "notebook_inputs nor dataset_inputs" if datasets else "notebook_inputs"
        raise BuildError(
            f"{path}: {reader[0]} pass {{input:...}} for {unknown}, which {attaching} does "
            "not attach"
        )
    unused = sorted(set(raw) - named)
    if unused:
        raise BuildError(
            f"{path}: notebook_inputs {unused} never reach {reader[1]}: pass each as "
            f"{{input:<owner>/<slug>}} in {reader[0]}"
        )
    unused = sorted(set(datasets) - named)
    if unused:
        raise BuildError(
            f"{path}: dataset_inputs {unused} never reach {reader[1]}: pass each as "
            f"{{input:<owner>/<slug>}} in {reader[0]}"
        )
    return raw


def parse_config(data: bytes, path: Path) -> BuildConfig:
    """Parse and check a build config: a submission config, or a script config (task P0.5d).

    Why: every value that changes what runs on Kaggle comes from the config, so it is checked
    strictly before anything is built: exactly the known keys of its ``kind`` (a submission
    config may leave out ``kind``, ``arms``, ``solver``, ``scope``, ``adapters``,
    ``notebook_inputs``, and ``agent``), no ``TODO`` anywhere
    (comments included), owners equal to the gateway's ``KAGGLE_OWNER``, accelerator IDs the
    gateway accepts, no more workers than the accelerator has GPUs, deadlines that leave
    ``WRITE_RESERVE_S`` before Kaggle's 12 h limit, series arms that pass ``parse_arms``, a
    ``solver`` from ``SOLVERS`` (``DEFAULT_SOLVER`` when left out), ``HOLDOUT_SOLVER`` when an
    arm holds a pair out (task P2.1), a ``scope`` that passes ``parse_scope``, never beside
    ``arms`` (task P2.6), ``adapters`` that pass ``parse_adapters`` (task P3.7: a notebook
    other than ``SUBMISSION_NOTEBOOK``, an accelerator, the split solver, no series, a scope, no
    subset with GEN, a hard stop after the arms' selection), whose paths name exactly the
    ``notebook_inputs`` attached; ``notebook_inputs`` come only with them; and an ``agent`` that
    passes ``parse_agent``, never beside ``arms`` (task P6.5; ``resolve_agent`` reads its run
    config). A script config names a
    ``.py`` under ``scripts/`` and a ``.yaml`` under ``configs/``, extra arguments that never set
    ``--config``, a hard stop that leaves ``WRITE_RESERVE_S``, and a notebook other than
    ``SUBMISSION_NOTEBOOK``; its optional ``notebook_inputs`` pin outputs of other notebooks,
    disjoint from ``kernel_source_versions``, each passed in ``args`` as
    ``{input:<owner>/<slug>}`` and no other (task P3.5).

    Args:
        data: Raw bytes of the config.
        path: Its path, used in messages.

    Returns:
        The checked settings: a ``ScriptConfig`` for ``kind: script``, else a
        ``SubmissionConfig``.

    Raises:
        BuildError: If the text is not UTF-8 or holds ``TODO``; the YAML is invalid or not a
            mapping; ``kind`` is unknown; a key is missing or unknown; or a value breaks its rule
            (see the comments of ``configs/submission_P0.5-01.yaml``,
            ``configs/submission_P1.3-01.yaml``, ``configs/submission_P2.0-01.yaml``,
            ``configs/submission_P2.1-01.yaml``, ``configs/submission_P2.6-01.yaml``,
            ``configs/submission_P6.5-01.yaml``, and ``configs/script_smoke.yaml``).
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as err:
        raise BuildError(f"{path} is not UTF-8 text: {err}") from err
    marked = [str(n) for n, line in enumerate(text.splitlines(), start=1) if "TODO" in line]
    if marked:
        raise BuildError(f"{path} still has TODO on line(s) {', '.join(marked)}")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise BuildError(f"{path} is not valid YAML: {err}") from err
    if not isinstance(raw, dict):
        raise BuildError(f"{path} must hold a mapping")
    if _has_todo(raw):
        raise BuildError(f"{path} still has a TODO value")
    kind = raw.get("kind", DEFAULT_KIND)
    if not isinstance(kind, str) or kind not in KINDS:
        raise BuildError(f"{path}: kind must be one of {', '.join(KINDS)}; got {kind!r}")
    required, optional = CONFIG_KEYS, OPTIONAL_KEYS
    if kind == "script":
        required, optional = SCRIPT_CONFIG_KEYS, SCRIPT_OPTIONAL_KEYS
    keys = set(map(str, raw))
    missing = set(required) - keys
    unknown = keys - set(required) - set(optional)
    if missing or unknown:
        raise BuildError(
            f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )
    stack = _parse_stack(raw, path)
    if kind == "script":
        return _parse_script(raw, stack, path)
    return _parse_submission(raw, stack, path)


def _parse_stack(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    # The keys every config shares: identity, the attached sources, and the accelerator.
    notebook = _owned(raw, "kaggle_notebook", 5, path)
    dataset = _owned(raw, "code_dataset", 6, path)
    competition = raw["competition"]
    if not isinstance(competition, str) or not reference_run.COMPETITION_SLUG.fullmatch(
        competition
    ):
        raise BuildError(f"{path}: competition must be a Kaggle competition slug")
    accelerator = raw["accelerator"]
    if accelerator is not None and (
        not isinstance(accelerator, str) or accelerator not in gateway.ACCELERATORS
    ):
        raise BuildError(
            f"{path}: accelerator must be null (CPU) or one of "
            f"{', '.join(sorted(gateway.ACCELERATORS))}; got {accelerator!r}"
        )
    if accelerator in NO_BF16_ACCELERATORS:
        raise BuildError(
            f"{path}: accelerator {accelerator} can never pass the environment check's bf16 stop "
            f"({NO_BF16_ACCELERATORS[accelerator]}); request NvidiaL4"
        )
    metadata = raw["reference_metadata"]
    if (
        not isinstance(metadata, str)
        or not metadata.endswith(".json")
        or metadata.startswith("/")
        or ".." in metadata.split("/")
        or "\\" in metadata
    ):
        raise BuildError(f"{path}: reference_metadata must be a relative path to a .json file")
    pins = raw["kernel_source_versions"]
    if not isinstance(pins, dict) or not all(
        isinstance(name, str)
        and reference_run.UNVERSIONED_KERNEL.fullmatch(name)
        and _integer(version)
        and version >= 1
        for name, version in pins.items()
    ):
        raise BuildError(
            f"{path}: kernel_source_versions must map <owner>/<slug> to a version number >= 1"
        )
    model_slug = raw["model_slug"]
    if not safe_id(model_slug):
        raise BuildError(f"{path}: model_slug must be one folder name; got {model_slug!r}")
    return {
        "kaggle_notebook": notebook,
        "code_dataset": dataset,
        "competition": competition,
        "accelerator": accelerator,
        "reference_metadata": metadata,
        "kernel_source_versions": dict(pins),
        "model_slug": model_slug,
    }


def _parse_submission(raw: dict[str, Any], stack: dict[str, Any], path: Path) -> SubmissionConfig:
    # The solver keys of a submission config; parse_config has checked the keys and the stack.
    accelerator = stack["accelerator"]
    subset, dev240 = raw["subset"], raw["subset"] == DEV240_SUBSET
    if dev240:
        # DEV240 (task P6.5b): every evaluation task and its copy, built in the notebook.
        subset = None
    elif subset is not None:
        if (
            not isinstance(subset, list)
            or not subset
            or not all(isinstance(t, str) and TASK_ID.fullmatch(t) for t in subset)
            or len(set(subset)) != len(subset)
        ):
            raise BuildError(
                f"{path}: subset must be null or a non-empty list of distinct 8-digit hex task ids "
                f"(or {DEV240_SUBSET}, task P6.5b)"
            )
        subset = tuple(subset)
    workers = raw["workers"]
    if not _integer(workers) or not 1 <= workers <= MAX_WORKERS:
        raise BuildError(f"{path}: workers must be an integer from 1 to {MAX_WORKERS}")
    if accelerator in GPU_COUNTS and workers > GPU_COUNTS[accelerator]:
        raise BuildError(
            f"{path}: {workers} workers need {workers} GPUs, but {accelerator} has "
            f"{GPU_COUNTS[accelerator]} (one worker per GPU)"
        )
    hash_seed = raw["hash_seed"]
    if not _integer(hash_seed) or not 0 <= hash_seed <= reference_run.MAX_HASH_SEED:
        raise BuildError(
            f"{path}: hash_seed must be an integer from 0 to {reference_run.MAX_HASH_SEED}"
        )
    run_solver = raw["run_solver"]
    if not isinstance(run_solver, bool):
        raise BuildError(f"{path}: run_solver must be true or false")
    end, kill = raw["solver_end_s"], raw["solver_kill_s"]
    latest = SESSION_LIMIT_S - WRITE_RESERVE_S
    if not (_integer(end) and _integer(kill) and MIN_SOLVER_END_S <= end <= kill <= latest):
        raise BuildError(
            f"{path}: need integers {MIN_SOLVER_END_S} <= solver_end_s <= solver_kill_s <= "
            f"{latest} (12 h minus {WRITE_RESERVE_S} s to write the submission); got "
            f"{end!r} and {kill!r}"
        )
    solver = raw.get("solver", DEFAULT_SOLVER)
    if not isinstance(solver, str) or solver not in SOLVERS:
        raise BuildError(f"{path}: solver must be one of {', '.join(SOLVERS)}; got {solver!r}")
    arms = parse_arms(raw.get("arms"), path)
    held = [arm.name for arm in arms or () if arm.holdout is not None]
    if held and solver != HOLDOUT_SOLVER:
        raise BuildError(
            f"{path}: arm {held[0]} holds a pair out, which needs solver: {HOLDOUT_SOLVER} (the "
            "moved reference trains and decodes in one loop body, ADR-011); got solver "
            f"{solver!r}"
        )
    scope = parse_scope(raw.get("scope"), path)
    if scope is not None and arms is not None:
        raise BuildError(
            f"{path}: scope and arms exclude each other: a development run trains each arm's "
            "scope, while a competition rerun ignores the arms and would train the build's "
            "scope, which no development run of the build trained"
        )
    adapters = parse_adapters(raw.get("adapters"), path)
    notebook_inputs: dict[str, int] = {}
    dataset_inputs: dict[str, int] = {}
    if adapters is None:
        if raw.get("notebook_inputs") is not None:
            raise BuildError(
                f"{path}: notebook_inputs attach other notebooks' outputs for the adapter arms "
                "(task P3.7) only; this config has no adapters"
            )
        if raw.get("dataset_inputs") is not None:
            raise BuildError(
                f"{path}: dataset_inputs attach datasets for the adapter arms (task P3.9b) "
                "only; this config has no adapters"
            )
    else:
        reasons = []
        if stack["kaggle_notebook"] == SUBMISSION_NOTEBOOK:
            reasons.append(
                f"its notebook is {SUBMISSION_NOTEBOOK}, the one submissions come from, and an "
                "adapter arms build is never submitted"
            )
        if accelerator is None:
            reasons.append("it has no accelerator, which the arms' workers need")
        if solver != HOLDOUT_SOLVER:
            reasons.append(f"its solver is {solver!r}; arc26.arms needs {HOLDOUT_SOLVER!r}")
        if arms is not None:
            reasons.append("it has a series (arms): the adapter arms run in one launch")
        if scope is None:
            reasons.append("it has no scope: H's adapter needs the live adapter at H's scope")
        if adapters.gen is not None and subset is not None:
            reasons.append("GEN replaces the subset: subset must be null with gen")
        if dev240:
            reasons.append(f"subset {DEV240_SUBSET} is the hybrid's timing check (task P6.5b)")
        if raw.get("agent") is not None:
            reasons.append("it has an agent stage (task P6.5): the arms run alone")
        reserve = end + adapters.finish_s + ADAPTER_SELECTION_RESERVE_S
        if kill < reserve:
            reasons.append(
                f"solver_kill_s {kill} leaves the arms' selection no time: it must be at least "
                f"solver_end_s + finish_s + {ADAPTER_SELECTION_RESERVE_S} = {reserve}"
            )
        if reasons:
            raise BuildError(f"{path}: adapters refused: " + "; ".join(reasons))
        dataset_inputs = _pinned_inputs(
            raw.get("dataset_inputs"), "dataset_inputs", "datasets", path
        )
        notebook_inputs = _parse_notebook_inputs(
            raw.get("notebook_inputs"),
            stack,
            adapters.paths(),
            path,
            ("adapters", "the arms"),
            dataset_inputs,
        )
    agent = parse_agent(raw.get("agent"), stack, path)
    if agent is not None and arms is not None:
        raise BuildError(
            f"{path}: agent and arms exclude each other: a series compares NVARC's scopes in "
            "development runs, while the agent stage follows the single run in both modes"
        )
    return SubmissionConfig(
        **stack,
        subset=subset,
        workers=workers,
        hash_seed=hash_seed,
        run_solver=run_solver,
        solver_end_s=end,
        solver_kill_s=kill,
        arms=arms,
        solver=solver,
        scope=scope,
        adapters=adapters,
        notebook_inputs=notebook_inputs,
        agent=agent,
        dev240=dev240,
        dataset_inputs=dataset_inputs,
    )


def check_script_files(
    config: ScriptConfig, script: bytes, script_config: bytes
) -> dict[str, bytes]:
    """Check a script build's script and its config; return them by their dataset file names.

    Why: both go to Kaggle byte for byte, beside the wheel in the code dataset, and a mistake
    found there costs a session and its queue wait. The script must compile, and its config must
    be a YAML mapping without ``TODO``, as the build config must be.

    Args:
        config: The checked script config.
        script: The bytes of ``config.script``.
        script_config: The bytes of ``config.script_config``.

    Returns:
        Both files keyed by their base names, as the code dataset folder holds them (kaggle-cli
        drops subfolders from a dataset upload).

    Raises:
        BuildError: If the script is not UTF-8 Python that compiles, or its config is not UTF-8,
            holds ``TODO``, or is not a YAML mapping.
    """
    try:
        compile(script.decode("utf-8"), config.script, "exec")
    except (UnicodeDecodeError, SyntaxError, ValueError) as err:
        raise BuildError(f"{config.script} does not compile: {err}") from err
    try:
        text = script_config.decode("utf-8")
    except UnicodeDecodeError as err:
        raise BuildError(f"{config.script_config} is not UTF-8 text: {err}") from err
    if "TODO" in text:
        raise BuildError(f"{config.script_config} still has TODO")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise BuildError(f"{config.script_config} is not valid YAML: {err}") from err
    if not isinstance(value, dict):
        raise BuildError(f"{config.script_config} must hold a mapping")
    return {
        PurePosixPath(config.script).name: script,
        PurePosixPath(config.script_config).name: script_config,
    }


def notebook_file(config: BuildConfig) -> str:
    """Return the notebook's file name: ``SCRIPT_NOTEBOOK_FILE`` for a script build.

    Args:
        config: The checked config.

    Returns:
        The file name in ``dist/<tag>/`` and ``kernel-metadata.json``'s ``code_file``.
    """
    return SCRIPT_NOTEBOOK_FILE if config.kind == "script" else NOTEBOOK_FILE


def build_info(
    config: BuildConfig,
    *,
    tag: str,
    commit: str,
    config_file: str,
    config_sha256: str,
    wheel: str,
    wheel_sha256: str,
    script_sha256: str | None = None,
    script_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the provenance the notebook bakes into its Config cell as ``BUILD``.

    Why: the notebook checks the attached wheel against ``wheel`` and ``wheel_sha256``, finds its
    inputs by ``competition`` and ``model_slug``, requires GPUs only when built for an
    ``accelerator``, and records the whole mapping in ``env.json``, so every pulled log names the
    commit and config it came from. A script build adds ``kind``, the script and its config with
    their sha256, which the notebook checks before running the script, and the pinned notebook
    outputs it attaches, ``notebook_inputs``, when there are any (task P3.5). A hybrid build adds
    ``agent`` (task P6.5): its run config with its sha256, the two download outputs it attaches,
    and the engine's interpreter, scratch folder, and oldest driver.

    Args:
        config: The checked config.
        tag: Build tag, the name of the ``dist/`` folder.
        commit: Full SHA of the commit built.
        config_file: The config's path relative to the repository, with ``/`` separators.
        config_sha256: sha256 of the config's bytes.
        wheel: The wheel's file name.
        wheel_sha256: sha256 of the wheel.
        script_sha256: sha256 of a script build's script; None for a submission build.
        script_config_sha256: sha256 of a script build's script config; None for a submission
            build.

    Returns:
        The provenance mapping, in the order it is written.

    Raises:
        BuildError: If the tag, commit, or wheel name is malformed, a script build lacks a
            script hash, or a hybrid build's run config was not read (``resolve_agent``).
    """
    if not safe_id(tag):
        raise BuildError(f"tag must be letters, digits, '.', '_', '-'; got {tag!r}")
    if not COMMIT.fullmatch(commit):
        raise BuildError(f"commit must be a hex SHA; got {commit!r}")
    if not (safe_id(wheel) and wheel.startswith("arc26-") and wheel.endswith(".whl")):
        raise BuildError(f"not an arc26 wheel file name: {wheel!r}")
    info: dict[str, Any] = {
        "tag": tag,
        "commit": commit,
        "config": config_file,
        "config_sha256": config_sha256,
        "wheel": wheel,
        "wheel_sha256": wheel_sha256,
        "kaggle_notebook": config.kaggle_notebook,
        "code_dataset": config.code_dataset,
        "competition": config.competition,
        "model_slug": config.model_slug,
        "accelerator": config.accelerator,
    }
    if config.kind == "submission" and config.agent is not None:
        info["agent"] = config.agent.build_settings()
    if config.kind == "script":
        if script_sha256 is None or script_config_sha256 is None:
            raise BuildError("a script build needs the sha256 of its script and its config")
        info.update(
            kind="script",
            script=config.script,
            script_sha256=script_sha256,
            script_config=config.script_config,
            script_config_sha256=script_config_sha256,
        )
        if config.notebook_inputs:
            info["notebook_inputs"] = dict(config.notebook_inputs)
    else:
        # An adapter build's attached notebook outputs (task P3.7) and datasets (task P3.9b);
        # every other build's BUILD is as before.
        if config.notebook_inputs:
            info["notebook_inputs"] = dict(config.notebook_inputs)
        if config.dataset_inputs:
            info["dataset_inputs"] = dict(config.dataset_inputs)
    return info


# Runtime helpers. These functions run inside the Kaggle notebook, not in the builder: the Helpers
# cell holds their source verbatim (inspect.getsource). They import what they use, so they stand
# alone, and tests call them directly.


def find_inputs(input_root: str, names: list[str], max_depth: int) -> dict[str, list[str]]:
    """Find files by exact name under the input root, at most ``max_depth`` levels down.

    Why: Kaggle mounts competitions, models, datasets, and notebook outputs under the input root,
    and the layout changed in 2026 (datasets possibly under ``datasets/<owner>/<slug>/``), so the
    notebook finds its inputs by name instead of hardcoding paths. The walk is bounded and skips
    package trees, because a mounted notebook output can hold a whole site-packages folder.

    Args:
        input_root: Folder to search, ``/kaggle/input`` on Kaggle.
        names: File names to look for.
        max_depth: Deepest level searched; a file directly in ``input_root`` is at level 1.

    Returns:
        For every name, the paths found, sorted; an empty list when there is none.
    """
    import os

    skipped = {"site-packages", "dist-packages", "__pycache__", "node_modules", ".git"}
    found: dict[str, list[str]] = {name: [] for name in names}
    root = os.path.abspath(input_root)
    for directory, subdirectories, files in os.walk(root, followlinks=True):
        level = 0 if directory == root else os.path.relpath(directory, root).count(os.sep) + 1
        if level + 1 >= max_depth:
            subdirectories[:] = []
        else:
            subdirectories[:] = sorted(
                d
                for d in subdirectories
                if d not in skipped and not d.endswith((".dist-info", ".egg-info"))
            )
        for name in files:
            if name in found:
                found[name].append(os.path.join(directory, name))
    return {name: sorted(paths) for name, paths in found.items()}


def find_input_dirs(roots: list[str], names: list[str], max_depth: int) -> dict[str, list[str]]:
    """Find the mount folders of attached notebook outputs by their slugs (task P3.5).

    Why: Kaggle mounts another notebook's output under a folder named by its slug, and where is
    unverified for 2026: P3.4's checker found the builder's output by a search of
    ``/kaggle/input``, while the install notebook, also a kernel source, is mounted at
    ``/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch``, its slug with
    underscores (P0.5-01's session). So every root is searched for both spellings, a folder inside
    one named by the owner is preferred, and folders holding the same files (one output mounted
    twice) count once; the caller requires one folder and refuses the rest as ambiguous. The walks
    are bounded and skip package trees.

    Args:
        roots: Folders to search, ``/kaggle/input`` and ``/kaggle/usr/lib/notebooks`` on Kaggle;
            a missing one adds nothing.
        names: Notebook outputs as ``<owner>/<slug>``.
        max_depth: Deepest folder level searched; a folder directly in a root is at level 1.

    Returns:
        For every name, its distinct folders in search order (the roots in the order given): only
        those inside a folder named by its owner when there are any, and one folder per set of
        identical contents; an empty list when there is none.
    """
    import os

    skipped = {"site-packages", "dist-packages", "__pycache__", "node_modules", ".git"}

    def contents(folder: str) -> list[tuple[str, int]]:
        # Relative path and size of every file, each real folder walked once.
        listing, seen = [], set()
        for directory, subdirectories, files in os.walk(folder, followlinks=True):
            real = os.path.realpath(directory)
            if real in seen:
                subdirectories[:] = []
                continue
            seen.add(real)
            subdirectories.sort()
            for file in sorted(files):
                path = os.path.join(directory, file)
                listing.append((os.path.relpath(path, folder), os.path.getsize(path)))
        return listing

    spellings = {}
    for name in names:
        owner, slug = name.split("/", 1)
        spellings[name] = (owner, {slug, slug.replace("-", "_")})
    found: dict[str, list[str]] = {name: [] for name in names}
    for input_root in roots:
        root = os.path.abspath(input_root)
        for directory, subdirectories, _ in os.walk(root, followlinks=True):
            level = 0 if directory == root else os.path.relpath(directory, root).count(os.sep) + 1
            kept = sorted(
                d
                for d in subdirectories
                if d not in skipped and not d.endswith((".dist-info", ".egg-info"))
            )
            for folder in kept:
                for name, (_, forms) in spellings.items():
                    if folder in forms:
                        found[name].append(os.path.join(directory, folder))
            subdirectories[:] = kept if level + 1 < max_depth else []
    result = {}
    for name, paths in found.items():
        owner = spellings[name][0]
        owned = [path for path in paths if os.path.basename(os.path.dirname(path)) == owner]
        distinct: list[str] = []
        listings = []
        for path in owned or paths:
            if any(os.path.realpath(path) == os.path.realpath(d) for d in distinct):
                continue
            listing = contents(path)
            if listing not in listings:
                distinct.append(path)
                listings.append(listing)
        result[name] = distinct
    return result


def fill_args(args: list[str], commit: str, inputs: dict[str, str]) -> list[str]:
    """Fill a script's arguments: ``{commit}`` first, then ``{input:<owner>/<slug>}`` (task P3.5).

    Why: the script records the code it ran, and its copy in the code dataset has no git checkout,
    so ``{commit}`` stands for the built commit's SHA (task P0.5d); an attached notebook output's
    mount folder is known only once the notebook has found it, so ``{input:<owner>/<slug>}`` stands
    for that folder. The caller checks the filled arguments with ``config_override``.

    Args:
        args: The arguments after ``--config <config>``, as the Config cell's ``ARGS`` hold them.
        commit: The built commit's full SHA.
        inputs: Mount folder of every attached notebook output, by ``<owner>/<slug>``.

    Returns:
        The arguments with every placeholder replaced.

    Raises:
        ValueError: If an argument names a notebook output that ``inputs`` lacks.
    """
    import re

    def folder(match: re.Match) -> str:
        if match.group(1) not in inputs:
            raise ValueError(
                f"{match.group(0)} names a notebook output that this build does not attach"
            )
        return inputs[match.group(1)]

    return [re.sub(r"\{input:([^{}]*)\}", folder, arg.replace("{commit}", commit)) for arg in args]


def run_logged(command: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command, print it with its output and exit code, and return the result.

    Why: the notebook uses no shell magics such as ``!pip``, so every command runs through
    ``subprocess`` and the caller checks the exit code. The output is captured and printed from
    Python, so it lands in the notebook and in Kaggle's session log.

    Args:
        command: Program and arguments; no shell is involved.
        timeout: Seconds before the command is abandoned, or None for no limit.

    Returns:
        The completed process with text ``stdout`` and ``stderr``. A command that cannot start
        returns exit code 127 and one that overruns ``timeout`` 124, instead of raising, so the
        caller's check decides what happens.
    """
    import shlex
    import subprocess

    print("$ " + shlex.join(command), flush=True)
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        code = 124 if isinstance(err, subprocess.TimeoutExpired) else 127
        print(f"could not run: {err}", flush=True)
        return subprocess.CompletedProcess(command, code, "", str(err))
    for text in (result.stdout, result.stderr):
        if text:
            print(text.rstrip("\n"), flush=True)
    print(f"exit code {result.returncode}", flush=True)
    return result


def gpu_inventory() -> list[dict[str, str]]:
    """List the GPUs as ``nvidia-smi`` reports them, without importing torch.

    Why: torch touching CUDA in the notebook kernel would create a CUDA context on GPU 0 and take
    memory from solver worker 0, which the reference never did. ``compute_cap`` tells bf16 and
    FlashAttention 2 support (8.0 and up) without CUDA; a driver that does not know the field is
    asked again without it. Only lines with one value per queried field are GPUs: a warning that
    ``nvidia-smi`` prints on stdout (an infoROM message, say) would otherwise count as a GPU of
    unknown capability and stop a healthy run at the bf16 check.

    Returns:
        One mapping per GPU with ``name``, ``memory.total``, ``compute_cap`` (when known), and
        ``driver_version``; an empty list when ``nvidia-smi`` is absent or fails.
    """
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    for fields in (
        ("name", "memory.total", "compute_cap", "driver_version"),
        ("name", "memory.total", "driver_version"),
    ):
        try:
            result = subprocess.run(
                [smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            rows = [[part.strip() for part in line.split(",")] for line in lines]
            return [dict(zip(fields, row, strict=True)) for row in rows if len(row) == len(fields)]
    return []


def bf16_refusal(gpus: list[dict[str, str]], accelerator: str | None) -> str | None:
    """Return why a build made for an accelerator cannot run bf16 on these GPUs, or None.

    Why: the solver and the Phase 3 scripts train and infer in bf16, which needs compute
    capability 8.0 (Ampere) or higher. Kaggle runs a push whose accelerator it does not grant on
    its default 2xT4 (7.5) without a word: run ``P0.3-01`` asked for ``NvidiaL4X1``, installed,
    loaded the model, and failed at its first puzzle about 5 minutes in. Reading the capability
    from ``gpu_inventory`` keeps torch out of the kernel.

    Args:
        gpus: ``gpu_inventory()``; a GPU without a numeric ``compute_cap`` counts as below 8.0.
        accelerator: The build's accelerator ID; None for a CPU build.

    Returns:
        None for a CPU build, or when every GPU has compute capability 8.0 or higher; otherwise
        the reason, with each GPU below 8.0.
    """
    if accelerator is None:
        return None
    if not gpus:
        return f"the build asked for {accelerator}, but nvidia-smi lists no GPU"
    below = []
    for gpu in gpus:
        capability = gpu.get("compute_cap", "")
        if not (capability.replace(".", "", 1).isdigit() and float(capability) >= 8.0):
            below.append(f"{gpu.get('name', 'a GPU')} ({capability or 'capability unknown'})")
    if not below:
        return None
    return (
        f"the build asked for {accelerator}, but nvidia-smi lists GPUs under compute capability "
        f"8.0, without bf16: {', '.join(below)}. Kaggle runs an accelerator it does not grant on "
        "its default 2xT4; solver and bf16 builds request NvidiaL4"
    )


def config_override(args: list[str]) -> str | None:
    """Return the first argument that would set a script's ``--config``, or None.

    Why: a script notebook passes ``--config`` with the config whose sha256 it checked (task
    P0.5d). A later ``--config`` would replace it, and so would an abbreviation argparse accepts
    (``--conf``) or a short alias ``-c`` a script might define, in any of its forms (``-c x``,
    ``-cx``, ``-c=x``, or clustered with other short flags as in ``-vc x``). The builder checks a
    config's ``args``; the notebook checks the Config cell's ``ARGS`` again, since a human may
    edit them.

    Args:
        args: The arguments that follow ``--config <config>`` on the script's command line.

    Returns:
        The first such argument, or None when there is none.
    """
    for arg in args:
        name = arg.split("=", 1)[0]
        if name.startswith("--"):
            if len(name) > 2 and "--config".startswith(name):
                return arg
        elif arg[:1] == "-" and "c" in arg[1:].split("=", 1)[0]:
            return arg
    return None


def file_sha256(path: str) -> str:
    """Return the sha256 hex digest of a file, read in 1 MiB blocks.

    Args:
        path: The file to hash.

    Returns:
        The digest as 64 lowercase hex characters.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def series_refusal(
    durations: list[float], failed: str | None, now_s: float, end_s: float, margin: float
) -> str | None:
    r"""Return why the next arm of a series must not start, or None when it may (task P1.3).

    Why: the arms of a series solve the same tasks, so the arms already run predict the next
    one's wall time. An arm still running at the soft deadline stops taking tasks there, and its
    score would cover only part of the subset; an arm after a failed one would most likely fail
    the same way. Neither starts, so no quota is spent on an arm that cannot be compared. With
    :math:`d_j` the wall time of arm :math:`j`, the next arm starts only if
    :math:`t + m \max_j d_j \le s_\text{end}`.

    Args:
        durations: Wall seconds of every arm already run, from just before its launch to the
            solver's exit.
        failed: Name of an arm already run whose solver failed, or None.
        now_s: Seconds since the notebook's start, :math:`t`.
        end_s: The workers' soft deadline in seconds since the start, :math:`s_\text{end}`.
        margin: Factor :math:`m` on the longest arm so far.

    Returns:
        The reason, or None. The first arm (no durations, no failure) may always start.
    """
    if failed is not None:
        return f"arm {failed} failed, so no later arm starts"
    if not durations:
        return None
    longest = max(durations)
    estimate = margin * longest
    left = end_s - now_s
    if estimate > left:
        return (
            f"{left:.0f} s are left before the soft deadline, less than one more arm needs: "
            f"{estimate:.0f} s, {margin} times the longest arm so far ({longest:.0f} s)"
        )
    return None


def holdout_text(settings: dict[str, object]) -> str:
    """Describe an arm's hold-out settings in one line (task P2.1).

    Why: the environment check, the notebook's first cell, and ``MANIFEST.md`` state what an arm
    holds out beside its LoRA scope, in the same words.

    Args:
        settings: ``rule``, ``k2``, and ``in_test_prompts``, as a config's arm holds them.

    Returns:
        One line, for example "hold-out: the last real pair of each puzzle with 2 or more, kept
        out of training and of every prompt".
    """
    least = 2 if settings["k2"] else 3
    prompts = (
        "and of the held-out input's prompts (kept in the test inputs' prompts)"
        if settings["in_test_prompts"]
        else "and of every prompt"
    )
    return (
        f"hold-out: the {settings['rule']} real pair of each puzzle with {least} or more, kept "
        f"out of training {prompts}"
    )


def rounds_text(settings: dict[str, object]) -> str:
    """Describe an arm's round settings in one line (task P2.3).

    Why: the environment check, the notebook's first cell, and ``MANIFEST.md`` state what a
    round arm runs beside its hold-out, in the same words.

    Args:
        settings: ``max_rounds``, ``tau``, ``continue_steps``, ``final_in_prompts``, and the
            optional ``variants`` (task P2.5; absent or None, the gradient round), as a config's
            arm holds them; other keys are ignored.

    Returns:
        One line, for example "rounds: at most 1 per puzzle after the gate, promotion at
        agreement 0.5625, the adapter continued 32 steps and restored when the gate fails, the
        final decode with the held-out pair back in its prompts".
    """
    final = (
        "with the held-out pair back in its prompts"
        if settings["final_in_prompts"]
        else "without the held-out pair"
    )
    gradient = (
        f"the adapter continued {settings['continue_steps']} steps and restored when the gate "
        "fails"
    )
    in_context = (
        "in context: the promoted pairs join the prompts, no fit, and only the pairs are "
        "rolled back when the gate fails"
    )
    variants = list(settings.get("variants") or ["gradient"])
    if variants == ["in_context"]:
        how = in_context
    elif variants == ["gradient", "in_context"]:
        how = f"{gradient}, or, when that does not fit the time left, {in_context}"
    else:
        how = gradient
    return (
        f"rounds: at most {settings['max_rounds']} per puzzle after the gate, promotion at "
        f"agreement {settings['tau']:g}, {how}, the final decode {final}"
    )


def run_streamed(
    command: list[str],
    timeout: float,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, bool]:
    """Run a command in a session of its own, copy its output as it arrives, kill it at a timeout.

    Why: a script notebook's script (task P0.5d) can run for hours and start processes of its
    own. Its lines must reach the session log while it runs, so a killed session still shows
    them, and the hard stop must end every process it started, which ``subprocess.run``'s timeout
    does not: it kills the direct child only. It repeats ``arc26.runner.run_solver``'s pattern as
    a helper embedded verbatim, so script mode changes nothing in the ``arc26`` wheel.

    Args:
        command: Program and arguments; no shell is involved.
        timeout: Seconds before the command's process group is killed with ``SIGKILL``.
        cwd: Working directory of the command; None keeps the caller's.
        env: The command's whole environment; None inherits the caller's.

    Returns:
        The exit code (minus the signal number when a signal ended it, 127 when it could not
        start) and whether the timeout killed the command itself, not only processes it left.
    """
    import codecs
    import os
    import selectors
    import shlex
    import signal
    import subprocess
    import sys
    import time

    print("$ " + shlex.join(command), flush=True)
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as err:
        print(f"could not run: {err}", flush=True)
        return 127, False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def copy(until: float) -> bool:
        # Copy the output to sys.stdout until it ends (True) or the time `until` passes (False).
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                left = until - time.monotonic()
                if left <= 0:
                    return False
                if not selector.select(left):
                    continue
                chunk = os.read(process.stdout.fileno(), 1 << 16)
                if not chunk:
                    return True
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()

    def kill_group() -> None:
        # The group outlives its leader while a member runs, and Linux keeps its id unused.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    killed = False
    try:
        ended = copy(deadline)
        if ended:
            try:
                code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                ended = False
        if not ended:
            killed = process.poll() is None
            kill_group()
            copy(time.monotonic() + 5.0)
            code = process.wait()
    except BaseException:
        kill_group()
        process.wait()
        raise
    finally:
        sys.stdout.write(decoder.decode(b"", final=True))
        process.stdout.close()
    print(f"exit code {code}{', killed at the timeout' if killed else ''}", flush=True)
    return code, killed


def save_json(path: Path, value: object) -> None:
    """Write ``value`` as JSON through a temporary file and ``os.replace``.

    Why: a reader, or the pull of a session killed mid-write, sees the old file or the new one,
    never part of one. A script notebook has no ``arc26.submit.write_json_atomic``: its logs must
    be written even when the install failed.

    Args:
        path: The file to write; its parent folder is created if needed.
        value: A JSON value; other objects are written as their ``str``.
    """
    import json
    import os
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def driver_refusal(gpus: list[dict[str, str]], minimum: str) -> str | None:
    """Return why the NVIDIA driver cannot run the agent's engine, or None (task P6.5).

    Why: the engine wheelhouse holds torch 2.13 with the CUDA 13.0 runtime (vLLM 0.31.0, task
    P6.3), which needs driver 580.65.06 or later. An older driver would fail the engine's boot
    after NVARC's share, so the agent check reads the version from ``gpu_inventory`` first,
    without CUDA in the kernel.

    Args:
        gpus: ``gpu_inventory()``.
        minimum: The oldest driver that works, a dotted version.

    Returns:
        None when every GPU reports a driver of at least ``minimum``; otherwise the reason.
    """

    def version(text: str) -> tuple[int, ...]:
        parts = text.split(".")
        return tuple(map(int, parts)) if all(part.isdigit() for part in parts) else (-1,)

    if not gpus:
        return "nvidia-smi lists no GPU, so no driver version"
    found = sorted({gpu.get("driver_version", "") for gpu in gpus})
    old = [text or "unknown" for text in found if version(text) < version(minimum)]
    if old:
        return (
            f"the NVIDIA driver is {', '.join(old)}, older than {minimum}, the oldest that runs "
            "the engine's CUDA 13.0 runtime"
        )
    return None


def check_model_output(root: str, folder: str, revision: str) -> tuple[str, dict[str, object]]:
    """Check the weights notebook's output and return the model folder (task P6.5).

    Why: the weights notebook (task P6.3) writes ``MANIFEST.json`` with ``ok`` true and the
    pinned revision only after every file's sha256 matched the Hub's. A mount cut short, or
    another revision, would cost an engine boot after NVARC's share, so the agent check reads the
    manifest and each file's size before anything runs; 19.5 GB are not hashed again.

    Args:
        root: The output's mount folder.
        folder: The weights' folder inside it.
        revision: The revision the manifest must name.

    Returns:
        The model folder and what was checked: ``folder``, ``files``, ``bytes``.

    Raises:
        ValueError: If the manifest is missing, unreadable, not ``ok``, or of another revision,
            or a listed file is missing or of another size, or there is no ``config.json``.
    """
    import json
    import os

    path = os.path.join(root, "MANIFEST.json")
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as err:
        raise ValueError(f"cannot read {path}: {err}") from err
    if not isinstance(manifest, dict) or manifest.get("ok") is not True:
        raise ValueError(f"{path} does not say ok: the download did not pass")
    if manifest.get("revision") != revision:
        raise ValueError(f"{path} names revision {manifest.get('revision')!r}, not {revision}")
    model_dir = os.path.join(root, folder)
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    wrong, total = [], 0
    for row in files:
        name = row.get("name") if isinstance(row, dict) else None
        size = row.get("size") if isinstance(row, dict) else None
        file = os.path.join(model_dir, str(name))
        if not isinstance(name, str) or not os.path.isfile(file):
            wrong.append(f"{name} missing")
        elif os.path.getsize(file) != size:
            wrong.append(f"{name} has {os.path.getsize(file)} bytes, not {size}")
        else:
            total += size
    if wrong or not files or not os.path.isfile(os.path.join(model_dir, "config.json")):
        reason = "; ".join(wrong) or "no files listed or no config.json"
        raise ValueError(f"model folder {model_dir}: {reason}")
    return model_dir, {"folder": model_dir, "files": len(files), "bytes": total}


def check_wheelhouse(root: str, folder: str, vllm: str) -> tuple[str, dict[str, object]]:
    """Check the wheels notebook's output and return the wheelhouse folder (task P6.5).

    Why: the wheels notebook (task P6.3) writes ``MANIFEST.json`` beside the wheels, ``ok`` only
    after its own install check passed on the same image, and ships ``install_wheelhouse.py``,
    the installer that check ran. A missing installer or another vLLM would fail the agent
    stage after NVARC's share, so the agent check reads them first.

    Args:
        root: The output's mount folder.
        folder: The wheelhouse's folder inside it.
        vllm: The vLLM version the manifest must list.

    Returns:
        The wheelhouse folder and what was checked: ``folder``, ``wheels``, ``torch``, and the
        install check's seconds.

    Raises:
        ValueError: If the manifest is missing, unreadable, or not ``ok``, lists another vLLM,
            or the installer or ``requirements.txt`` is missing.
    """
    import json
    import os

    wheelhouse = os.path.join(root, folder)
    path = os.path.join(wheelhouse, "MANIFEST.json")
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as err:
        raise ValueError(f"cannot read {path}: {err}") from err
    if not isinstance(manifest, dict) or manifest.get("ok") is not True:
        raise ValueError(f"{path} does not say ok: the wheelhouse or its install check failed")
    rows = [row for row in manifest.get("files") or [] if isinstance(row, dict)]
    versions = {row.get("name"): row.get("version") for row in rows if row.get("name")}
    if versions.get("vllm") != vllm:
        raise ValueError(f"{path} lists vllm {versions.get('vllm')!r}, not {vllm}")
    for name in ("install_wheelhouse.py", "requirements.txt"):
        if not os.path.isfile(os.path.join(wheelhouse, name)):
            raise ValueError(f"{wheelhouse} has no {name}")
    check = manifest.get("install_check") if isinstance(manifest.get("install_check"), dict) else {}
    return wheelhouse, {
        "folder": wheelhouse,
        "wheels": len(versions),
        "torch": versions.get("torch"),
        "install_check_s": check.get("seconds"),
    }


def nvarc_deadlines(
    end_s: float, kill_s: float, agent: dict[str, object] | None, gap_s: float
) -> tuple[float, float]:
    """Return NVARC's soft deadline and hard stop in a hybrid build (task P6.5).

    Why: NVARC hands the session to the agent at its share, in both modes, but only when the
    agent stage will follow (``agent`` given: enabled, and the agent check passed); otherwise
    NVARC keeps the deadlines it had, the competition's in a rerun, so missing agent inputs never
    cost NVARC its time. The deadlines only shrink, so a development run's shorter ones stand.

    Args:
        end_s: NVARC's soft deadline without the agent, in seconds after the start.
        kill_s: Its hard stop without the agent.
        agent: The Config cell's ``AGENT`` when the agent stage will follow, else None.
        gap_s: Seconds from the share to the hard stop.

    Returns:
        The soft deadline and the hard stop: ``min(end_s, share)`` and
        ``min(kill_s, share + gap_s)`` with the agent, else both unchanged.
    """
    if not agent or not agent.get("enabled"):
        return end_s, kill_s
    share = agent["nvarc_share_s"]
    return min(end_s, share), min(kill_s, share + gap_s)


def agent_plan(
    agent: dict[str, object],
    final_margin_s: float,
    boot_timeout_s: float,
    gap_s: float,
    write_s: float,
    kill_wait_s: float,
    session_s: float,
) -> dict[str, object]:
    r"""Return a hybrid session's worst-case timeline and what is wrong with it (task P6.5).

    Why: the builder refuses a plan in which the agent stage could not start or the notebook
    could end after Kaggle's limit, and the agent check treats a problem as a failed check (the
    Config cell's ``AGENT`` may have been edited). With :math:`S` NVARC's share, :math:`g` the
    gap to its hard stop, :math:`w` the time kept for a write, :math:`I` the install's timeout,
    :math:`B` the engine's boot timeout, :math:`E` the agent's end, :math:`f` its run config's
    final margin, :math:`\tau` the SIGTERM grace, and :math:`k` the kill wait: the stage starts
    by :math:`a = S + g + w` (:math:`a = w` without NVARC), the install ends by
    :math:`\min(a + I, E - f)` (its timeout is cut at the generation stop), the engine is up by
    :math:`\min(a + I + B, E - f)`, generation stops at :math:`E - f`, SIGKILL comes at
    :math:`E + \tau`, and the notebook ends by :math:`E + \tau + k + w`. The plan needs
    :math:`E - f - a \ge m` (``min_stage_s``) and :math:`E + \tau + k + w \le T`.

    Args:
        agent: The Config cell's ``AGENT``.
        final_margin_s: The agent run config's ``deadline.final_margin_s``, :math:`f`.
        boot_timeout_s: Its ``server.boot_timeout_s``, :math:`B`.
        gap_s: :math:`g`, ``NVARC_KILL_GAP_S``.
        write_s: :math:`w`, ``WRITE_RESERVE_S``.
        kill_wait_s: :math:`k`, ``AGENT_KILL_WAIT_S``.
        session_s: :math:`T`, Kaggle's 12 h limit.

    Returns:
        ``nvarc_end_s``, ``nvarc_kill_s``, ``agent_start_s``, ``install_end_s``,
        ``boot_end_s``, ``generation_stop_s``, ``agent_end_s``, ``agent_kill_s``, and
        ``notebook_end_s``, in seconds after the start, and ``problems``, a list of reasons.
    """
    share = agent["nvarc_share_s"]
    nvarc_kill = share + gap_s if share else 0
    start = nvarc_kill + write_s
    stop = agent["end_s"] - final_margin_s
    install_end = min(start + agent["install_timeout_s"], stop)
    killed = agent["end_s"] + agent["term_s"]
    end = killed + kill_wait_s + write_s
    problems = []
    if stop - start < agent["min_stage_s"]:
        problems.append(
            f"the agent stage could not start: after NVARC's hard stop ({nvarc_kill} s) and its "
            f"write ({write_s} s), {stop - start:g} s are left before the generation stop at "
            f"{stop:g} s (end_s minus final_margin_s), less than min_stage_s "
            f"({agent['min_stage_s']} s)"
        )
    if end > session_s:
        problems.append(
            f"the notebook could end at {end:g} s, after Kaggle's {session_s} s limit: SIGKILL "
            f"at {killed} s, {kill_wait_s} s for the process group, {write_s} s for the final "
            "check and the logs"
        )
    return {
        "nvarc_end_s": share,
        "nvarc_kill_s": nvarc_kill,
        "agent_start_s": start,
        "install_end_s": install_end,
        "boot_end_s": min(install_end + boot_timeout_s, stop),
        "generation_stop_s": stop,
        "agent_end_s": agent["end_s"],
        "agent_kill_s": killed,
        "notebook_end_s": end,
        "problems": problems,
    }


def shot_plan(
    agent: dict[str, object],
    final_margin_s: float,
    boot_timeout_s: float,
    gap_s: float,
    write_s: float,
    kill_wait_s: float,
    session_s: float,
) -> dict[str, object]:
    r"""Return a shot build's timeline and what is wrong with it (task P6.5b).

    Why: a shot build keeps NVARC's deadlines, so NVARC may use the whole session and the shot
    runs only in the time it leaves: unlike ``agent_plan``, the stage has no worst-case start.
    The plan states the latest start that still runs the shot instead, :math:`L = E - f - m`
    (the agent stage skips the shot when it starts later, ``agent_stage``), with :math:`E` the
    shot's end, :math:`f` its run config's final margin, and :math:`m` ``min_stage_s`` (setup,
    the smallest wave, margins). The builder refuses a plan in which no NVARC end could leave
    the shot its :math:`m` seconds (:math:`L < w`: the stage starts after the Write cell's
    :math:`w`), one without NVARC (the shot picks its tasks from NVARC's candidates), and one
    whose notebook could end after the session's usable limit, :math:`E + \tau + k + w > T`,
    where :math:`T` is ``SHOT_SESSION_S``: Kaggle's 12 h less ``PRE_SCRIPT_ALLOWANCE_S``, the time
    Kaggle may count before the notebook's first cell. The agent check treats a problem as a
    failed check, as ``agent_plan``'s.

    Args:
        agent: The Config cell's ``AGENT``.
        final_margin_s: The shot's run config's ``deadline.final_margin_s``, :math:`f`.
        boot_timeout_s: Its ``server.boot_timeout_s``; the shot's own setup is in :math:`m`, so
            only ``agent_plan``'s signature, which this one shares, needs it.
        gap_s: :math:`g`, ``NVARC_KILL_GAP_S``.
        write_s: :math:`w`, ``WRITE_RESERVE_S``.
        kill_wait_s: :math:`k`, ``AGENT_KILL_WAIT_S``.
        session_s: :math:`T`, the session's usable limit in notebook time (``SHOT_SESSION_S``).

    Returns:
        ``nvarc_end_s`` and ``nvarc_kill_s`` (NVARC's soft deadline and hard stop with the
        share), ``latest_start_s`` (:math:`L`), ``generation_stop_s``, ``agent_end_s``,
        ``agent_kill_s``, and ``notebook_end_s``, in seconds after the start, and ``problems``,
        a list of reasons.
    """
    del boot_timeout_s
    share = agent["nvarc_share_s"]
    stop = agent["end_s"] - final_margin_s
    latest = stop - agent["min_stage_s"]
    killed = agent["end_s"] + agent["term_s"]
    end = killed + kill_wait_s + write_s
    problems = []
    if not share:
        problems.append(
            "the shot picks its tasks from NVARC's candidates and keeps NVARC's attempts, so it "
            "needs NVARC's stage: nvarc_share_s is 0"
        )
    if latest < write_s:
        problems.append(
            f"the shot could never run: it needs min_stage_s ({agent['min_stage_s']} s) before "
            f"the generation stop at {stop:g} s (end_s minus final_margin_s), so the stage must "
            f"start by {latest:g} s, but it follows NVARC's write ({write_s} s)"
        )
    if end > session_s:
        problems.append(
            f"the notebook could end at {end:g} s, after the session's usable {session_s} s (12 h "
            f"less the time before the first cell): SIGKILL at {killed} s, {kill_wait_s} s for "
            f"the process group, {write_s} s for the final check and the logs"
        )
    return {
        "nvarc_end_s": share,
        "nvarc_kill_s": share + gap_s if share else 0,
        "latest_start_s": latest,
        "generation_stop_s": stop,
        "agent_end_s": agent["end_s"],
        "agent_kill_s": killed,
        "notebook_end_s": end,
        "problems": problems,
    }


def session_clocks(notebook_start: float, working: str) -> dict[str, object]:
    """Read the clocks that may see the time before the notebook's first cell (task P6.5b).

    Why: Kaggle showed a notebook RUNNING up to 41 minutes before its first cell started
    (P6.5-02: 36.7 min). Whether the 12 h limit counts that time is unknown, so a shot build
    plans for it (``PRE_SCRIPT_ALLOWANCE_S``) and logs, from inside, each clock that could show
    it: the wall time, the machine's uptime, the ages of PID 1 and of this kernel (their start
    ticks in ``/proc/<pid>/stat`` against the uptime), and the times of the working folder.
    Logging only: nothing depends on the values, and a clock that cannot be read is left out.

    Args:
        notebook_start: ``NOTEBOOK_START``, the Config cell's ``time.time()``.
        working: The notebook's working folder (``/kaggle/working``).

    Returns:
        ``time`` and ``since_start_s``; when readable ``uptime_s``, ``pid1_age_s``,
        ``kernel_age_s``, ``working_ctime_age_s``, ``working_mtime_age_s``, and
        ``working_birth_age_s``; ``errors`` for the clocks that failed.
    """
    import os
    import time

    now = time.time()
    clocks: dict[str, object] = {"time": now, "since_start_s": round(now - notebook_start, 1)}
    errors = []
    uptime = None
    try:
        with open("/proc/uptime", encoding="ascii") as handle:
            uptime = float(handle.read().split()[0])
        clocks["uptime_s"] = round(uptime, 1)
    except (OSError, ValueError, IndexError) as err:
        errors.append(f"uptime: {err}")
    for name, pid in (("pid1_age_s", "1"), ("kernel_age_s", "self")):
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            # Field 22 of proc(5), the start time in clock ticks after boot; the split after the
            # command's closing parenthesis starts at field 3.
            started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
            if uptime is not None:
                clocks[name] = round(uptime - started, 1)
        except (OSError, ValueError, IndexError) as err:
            errors.append(f"{name}: {err}")
    try:
        info = os.stat(working)
        clocks["working_ctime_age_s"] = round(now - info.st_ctime, 1)
        clocks["working_mtime_age_s"] = round(now - info.st_mtime, 1)
        birth = getattr(info, "st_birthtime", None)
        if birth:
            clocks["working_birth_age_s"] = round(now - birth, 1)
    except OSError as err:
        errors.append(f"working: {err}")
    if errors:
        clocks["errors"] = errors
    return clocks


def dev240_files(data_dir: str, target_dir: str) -> dict[str, object]:
    """Write DEV240's challenges and solutions: DEV120 and a copy of each task (task P6.5b).

    Why: the 12 h timing check must run NVARC on as many tasks as a scored rerun of about 240,
    with development solutions for its score. Every evaluation task appears twice: under its own
    id and under the first 8 hex digits of the sha256 of that id, the 8-hex form of every ARC id,
    so every id parser takes it; a copy holds the original's demonstrations, test inputs, and
    solutions, and is scored apart, under its own id. A copy's id must be no id of any challenge
    or solution file in ``data_dir`` and no other copy's. The files keep the competition's names,
    so ``DATA_DIR`` can point at ``target_dir`` for NVARC, its selection and score, the agent,
    and the final scores.

    Args:
        data_dir: The competition data folder, with ``arc-agi_evaluation_challenges.json`` and
            ``arc-agi_evaluation_solutions.json``.
        target_dir: The folder to write, outside ``/kaggle/working``.

    Returns:
        ``challenges`` and ``solutions`` (the written paths), ``tasks``, ``test_inputs``,
        ``copies`` (each copy's id to its original's), and ``checked`` (the files whose ids the
        copies were checked against).

    Raises:
        ValueError: If a file does not load or is not a mapping, the solutions do not cover the
            challenges exactly, or a copy's id is taken.
    """
    import hashlib
    import json
    import os
    from pathlib import Path

    source, target = Path(data_dir), Path(target_dir)

    def load(name: str) -> dict[str, object]:
        try:
            value = json.loads((source / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise ValueError(f"{name} does not load: {err}") from None
        if not isinstance(value, dict):
            raise ValueError(f"{name} must hold a mapping of task ids")
        return value

    names = ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json")
    challenges, solutions = load(names[0]), load(names[1])
    if set(challenges) != set(solutions):
        raise ValueError("the evaluation solutions do not cover exactly the evaluation challenges")
    taken, checked = {}, []
    for path in sorted(source.glob("*.json")):
        if path.name.endswith(("_challenges.json", "_solutions.json")):
            for task in load(path.name):
                taken.setdefault(task, path.name)
            checked.append(path.name)
    copies = {}
    for task in challenges:
        copy = hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
        owner = taken.get(copy) or (f"the copy of {copies[copy]}" if copy in copies else None)
        if owner is not None:
            raise ValueError(f"the copy id {copy} of {task} is taken by {owner}")
        copies[copy] = task
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, value in zip(names, (challenges, solutions), strict=True):
        doubled = {**value, **{copy: value[task] for copy, task in copies.items()}}
        path = target / name
        temporary = path.with_name(f".{name}.tmp")
        temporary.write_text(json.dumps(doubled), encoding="utf-8")
        os.replace(temporary, path)
        written.append(str(path))
    inputs = sum(len(t.get("test", [])) for t in challenges.values() if isinstance(t, dict))
    return {
        "challenges": written[0],
        "solutions": written[1],
        "tasks": 2 * len(challenges),
        "test_inputs": 2 * inputs,
        "copies": copies,
        "checked": checked,
    }


def candidates_json(outputs_dir: str, path: str, max_bytes: int) -> dict[str, object]:
    """Write NVARC's candidates as one JSON file for the agent (task P6.5); never raises.

    Why: the agent orders its second phase by NVARC's confidence (ADR-014's agreement over the
    16 views), read from a ``candidates.json`` as ``dump_run_logs`` writes it: every candidate
    file of the solver, keyed by its path under the folder, as a list of ``{solution,
    beam_score, score_aug}``. A competition rerun writes no run logs, so the agent stage writes
    this copy to scratch itself. A file that does not load is skipped and recorded; without any
    candidate, no file is written and the agent goes without.

    Args:
        outputs_dir: The solver's candidate folder, ``/kaggle/inference_outputs``.
        path: The JSON file to write; a file of an earlier launch is removed first.
        max_bytes: Candidate files larger than this are skipped, as ``dump_run_logs`` skips them.

    Returns:
        ``written``, ``files`` (read), ``skipped`` (paths), ``errors``, and ``bytes`` (written).
    """
    import bz2
    import json
    import os
    import pickle
    from pathlib import Path

    def plain(value: object) -> object:
        # numpy arrays and scalars convert to Python values via tolist(), as dump_run_logs does.
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [plain(item) for item in value]
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return repr(value)

    report: dict[str, object] = {"written": False, "files": 0, "skipped": [], "errors": []}
    report["bytes"] = 0
    try:
        target = Path(path)
        target.unlink(missing_ok=True)
        root = Path(outputs_dir)
        candidates = {}
        for file in sorted(root.rglob("*")) if root.is_dir() else []:
            if not file.is_file():
                continue
            if file.stat().st_size > max_bytes:
                report["skipped"].append(str(file))
                continue
            try:
                data = file.read_bytes()
                if data[:3] == b"BZh":
                    data = bz2.decompress(data)
                candidates[file.relative_to(root).as_posix()] = plain(pickle.loads(data))
                report["files"] += 1
            except Exception as err:
                report["errors"].append(f"{file.name}: {type(err).__name__}: {err}")
        if candidates:
            data = json.dumps(candidates, separators=(",", ":")).encode("utf-8")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, target)
            report.update(written=True, bytes=len(data))
    except Exception as err:
        report["errors"].append(f"{type(err).__name__}: {err}")
    return report


def merge_submission(
    current: object, fallback: object, counts: dict[str, int], grid_problem: object
) -> tuple[dict[str, list[dict[str, object]]], dict[str, int]]:
    """Return a valid submission: ``current``'s real grids, ``fallback``'s in every other slot.

    Why: the final check after the agent stage (task P6.5). The agent rewrites the file
    atomically and validated, so it is normally valid; but a development run over a subset
    leaves the other tasks out of it, a failure could leave anything, and a ``[[0]]`` in it is a
    placeholder, never an answer (a run that could not read NVARC's file fills every slot it
    did not solve with one). So per test input: every attempt of ``current`` that is a valid grid
    other than ``[[0]]`` stays in its slot; each remaining slot takes the next of ``fallback``'s
    real grids, in its order (NVARC's attempt 1 first, as P6.2's selection ranks NVARC's
    attempts), skipping a grid the entry already holds; then ``current``'s own ``[[0]]``, else
    ``fallback``'s, else ``[[0]]``. A file without a ``[[0]]`` that a real fallback grid could
    replace comes out equal to itself. The agent-only fallback calls it the other way round,
    NVARC's selection first and the agent's file filling the rest.

    Args:
        current: The parsed submission whose real grids win, or None.
        fallback: The parsed submission that fills the other slots, or None.
        counts: Task id to its number of test inputs (``entry_counts`` of the challenges).
        grid_problem: The validator's ``grid_problem`` (``arc26/submit.py``).

    Returns:
        The submission, in the challenges' task order, and how many attempts came from
        ``current``, ``fallback``, and the ``placeholder``, and how many of ``current``'s real
        grids differ from ``fallback``'s in the same slot (``changed``; 0 without a fallback).
    """

    def attempt(source: object, task: str, index: int, name: str) -> object:
        entries = source.get(task) if isinstance(source, dict) else None
        if not isinstance(entries, list) or index >= len(entries):
            return None
        entry = entries[index]
        grid = entry.get(name) if isinstance(entry, dict) else None
        return grid if grid_problem(grid) is None else None

    def real(grid: object) -> bool:
        return grid is not None and grid != [[0]]

    names = ("attempt_1", "attempt_2")
    merged: dict[str, list[dict[str, object]]] = {}
    sources = {"current": 0, "fallback": 0, "placeholder": 0, "changed": 0}
    for task, count in counts.items():
        rows = []
        for index in range(count):
            mine = [attempt(current, task, index, name) for name in names]
            theirs = [attempt(fallback, task, index, name) for name in names]
            row = [None, None]
            for slot, grid in enumerate(mine):
                if real(grid):
                    row[slot] = grid
                    sources["current"] += 1
                    if isinstance(fallback, dict) and grid != theirs[slot]:
                        sources["changed"] += 1
            for slot in (0, 1):
                if row[slot] is not None:
                    continue
                spare = [grid for grid in theirs if real(grid) and grid not in row]
                if spare:
                    row[slot] = spare[0]
                    sources["fallback"] += 1
                elif mine[slot] is not None:
                    row[slot] = mine[slot]
                    sources["current"] += 1
                elif theirs[slot] is not None:
                    row[slot] = theirs[slot]
                    sources["fallback"] += 1
                else:
                    row[slot] = [[0]]
                    sources["placeholder"] += 1
            pairs = zip(names, row, strict=True)
            rows.append({name: [list(line) for line in grid] for name, grid in pairs})
        merged[task] = rows
    return merged, sources


def gpu_apps() -> list[dict[str, str]] | None:
    """List the processes that hold GPU memory, as ``nvidia-smi`` reports them (task P6.5).

    Why: the engine boots on the four L4s that NVARC's workers held a moment before; a killed
    worker may keep its memory until the driver reclaims it, and vLLM's boot fails on a GPU without
    the free memory it asks for. The agent stage logs ``nvidia-smi
    --query-compute-apps=pid,used_memory --format=csv`` before the engine boots and around the
    agent-only fallback, without CUDA in the kernel.

    Returns:
        One ``{"pid", "used_memory"}`` per process line, or None when ``nvidia-smi`` is missing
        or fails (a CPU session).
    """
    import shutil

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    result = run_logged([smi, "--query-compute-apps=pid,used_memory", "--format=csv"], 60)
    if result.returncode != 0:
        return None
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            rows.append({"pid": parts[0], "used_memory": parts[1]})
    return rows


def gpu_memory() -> list[dict[str, int]] | None:
    """Return the memory in use on each GPU, in MiB, as ``nvidia-smi`` reports it (task P6.5).

    Why: inside a container ``--query-compute-apps`` can list no process while a killed worker's
    memory is still held (its pid lives in another namespace); ``memory.used`` reports the memory
    itself. Logged with ``gpu_apps`` before the engine boots and around the agent-only fallback.

    Returns:
        One ``{"index", "used_mib"}`` per GPU line, or None when ``nvidia-smi`` is missing or
        fails (a CPU session).
    """
    import shutil

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    query = [smi, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    result = run_logged(query, 60)
    if result.returncode != 0:
        return None
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            rows.append({"index": int(parts[0]), "used_mib": int(parts[1])})
    return rows


def wait_for_free_gpus(
    timeout_s: float, used_max_mib: int, interval_s: float = 5.0
) -> dict[str, object]:
    """Wait until the GPUs are free, at most ``timeout_s`` seconds (task P6.5).

    Why: see ``gpu_apps`` and ``gpu_memory``; a short wait lets the driver reclaim a killed
    worker's memory before vLLM measures what is free. Free means no process listed and at most
    ``used_max_mib`` MiB in use on every GPU. A GPU still held after the wait is recorded, not
    fatal: the engine's boot then fails or succeeds on its own, and the file stands either way.

    Args:
        timeout_s: The longest wait.
        used_max_mib: The most memory in use, in MiB, that a free GPU may report.
        interval_s: Seconds between two queries.

    Returns:
        ``before`` and ``after`` (``gpu_apps()``), ``memory_before`` and ``memory_after``
        (``gpu_memory()``), each at the start and at the end and None on a CPU session;
        ``used_max_mib``; and ``seconds`` waited.
    """
    import time

    def held(apps: list | None, memory: list | None) -> bool:
        return bool(apps) or any(row["used_mib"] > used_max_mib for row in memory or ())

    began = time.monotonic()
    before = apps = gpu_apps()
    memory_before = memory = gpu_memory()
    while held(apps, memory) and time.monotonic() - began < timeout_s:
        time.sleep(interval_s)
        apps, memory = gpu_apps(), gpu_memory()
    return {
        "before": before,
        "after": apps,
        "memory_before": memory_before,
        "memory_after": memory,
        "used_max_mib": used_max_mib,
        "seconds": round(time.monotonic() - began, 1),
    }


def clean_env(environ: dict[str, str]) -> dict[str, str]:
    """Return an environment cut off from the image's Python packages and pip settings (P6.5).

    Why: the engine install makes a venv from the wheelhouse alone (task P6.3); Kaggle's
    ``PYTHONPATH`` carries the attached install notebook's torch 2.8, and a ``PIP_*`` setting
    could point pip elsewhere. The installer cleans its own commands' environment too.

    Args:
        environ: The current environment.

    Returns:
        A copy without ``PYTHONPATH``, ``PYTHONHOME``, ``PYTHONSTARTUP``, ``PYTHONUSERBASE``,
        ``VIRTUAL_ENV``, and any ``PIP_*``, with ``PYTHONNOUSERSITE=1``.
    """
    dropped = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"}
    env = {k: v for k, v in environ.items() if k not in dropped and not k.startswith("PIP_")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_supervised(
    command: list[str],
    timeout: float,
    term_s: float,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command in a session of its own, copy its output as it arrives, stop it at a timeout.

    Why: the agent stage's two processes (task P6.5), the engine install and ``python -m
    arc26.agent.run``, run for minutes and hours and start processes of their own. Their lines
    must reach the session log while they run, and the hard stop must reach every process they
    started, which ``subprocess.run``'s timeout does not. Unlike ``run_streamed``, the timeout
    sends SIGTERM to the group first, so the agent can stop its engine and write once more
    (``arc26.agent.run`` turns SIGTERM into its cleanup), and SIGKILL only ``term_s`` later.
    When the command itself has ended, a process it left holding the output is not waited for
    past a few seconds: whatever is left of the group is killed.

    Args:
        command: Program and arguments; no shell is involved.
        timeout: Seconds before SIGTERM reaches the command's process group.
        term_s: Seconds between SIGTERM and SIGKILL.
        cwd: Working directory; None keeps the caller's.
        env: The command's whole environment; None inherits the caller's.

    Returns:
        The exit code (minus the signal number when a signal ended it, 127 when it could not
        start) and how it ended: ``exited``, ``terminated`` (after SIGTERM, before SIGKILL),
        ``killed``, or ``not started``.
    """
    import codecs
    import os
    import selectors
    import shlex
    import signal
    import subprocess
    import sys
    import time

    print("$ " + shlex.join(command), flush=True)
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as err:
        print(f"could not run: {err}", flush=True)
        return 127, "not started"
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def copy(until: float) -> bool:
        # Copy the output until it ends, or 5 s after the command exits (True); False when the
        # time `until` passes while the command still runs.
        exited = None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                now = time.monotonic()
                if process.poll() is None:
                    if now >= until:
                        return False
                    wait = min(1.0, until - now)
                else:
                    exited = now if exited is None else exited
                    if now - exited >= 5.0:
                        return True
                    wait = 5.0 - (now - exited)
                if not selector.select(max(0.01, wait)):
                    continue
                chunk = os.read(process.stdout.fileno(), 1 << 16)
                if not chunk:
                    return True
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()

    def signal_group(sig: int) -> None:
        # The group outlives its leader while a member runs, and Linux keeps its id unused.
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    ended = "exited"
    try:
        if not copy(deadline):
            ended = "terminated"
            signal_group(signal.SIGTERM)
            if not copy(time.monotonic() + term_s):
                ended = "killed"
                signal_group(signal.SIGKILL)
                copy(time.monotonic() + 5.0)
        code = process.wait()
    except BaseException:
        signal_group(signal.SIGKILL)
        process.wait()
        raise
    finally:
        signal_group(signal.SIGKILL)
        sys.stdout.write(decoder.decode(b"", final=True))
        process.stdout.close()
    print(f"exit code {code} ({ended})", flush=True)
    return code, ended


def stop_engine(pid_path: str, wait_s: float) -> str | None:
    """Stop the engine an agent run's pid file names, if it still runs (task P6.5).

    Why: the agent starts vLLM in a session of its own (``arc26.agent.server``), so killing the
    agent's process group leaves the engine, which holds all four GPUs. Its server records the
    engine's pid and argv in ``engine.pid`` and removes the file when it stops the engine
    itself. The group is signalled when the pid's process still runs that argv (a console script
    gains its interpreter in front), or when that process is gone but members of its group live
    on (vLLM's workers: Linux keeps a group's id unused while it has members); a pid that another
    program now holds is never signalled.

    Args:
        pid_path: The pid file in the agent's log folder.
        wait_s: Seconds to wait for the group after SIGTERM, then after SIGKILL.

    Returns:
        What was done, or None when there is no pid file.
    """
    import json
    import os
    import signal
    import time
    from pathlib import Path

    path = Path(pid_path)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        pid, argv = int(record["pid"]), [str(part) for part in record["argv"]]
    except (OSError, ValueError, KeyError, TypeError) as err:
        path.unlink(missing_ok=True)
        return f"unreadable pid file removed ({err})"

    def members() -> bool:
        # Whether the group has a member that is not a zombie.
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                fields = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[-1].split()
            except OSError:
                continue
            if len(fields) > 2 and fields[2] == str(pid) and fields[0] != "Z":
                return True
        return False

    try:
        found = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        found = found[:-1] if found and found[-1] == "" else found
    except OSError:
        found = None
    rest = argv[1:]
    same = bool(found) and (
        found == argv or (bool(rest) and len(found) > len(rest) and found[-len(rest) :] == rest)
    )
    if pid <= 1 or not (same or (found is None and members())):
        path.unlink(missing_ok=True)
        return f"the engine (pid {pid}) no longer runs"
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        end = time.monotonic() + wait_s
        while members() and time.monotonic() < end:
            time.sleep(0.2)
        if not members():
            break
    path.unlink(missing_ok=True)
    return f"stopped the engine's process group {pid}, which the agent left running"


def copy_logs(sources: list[str], target: str, max_bytes: int) -> dict[str, object]:
    """Copy the agent stage's small logs into ``run_logs/agent/`` for the pull (P6.5); never raises.

    Why: a development run's analysis needs the agent's attempt log, run summary, engine log,
    and install report, which live in scratch; only files the pull's allowlist takes are copied,
    and a file over ``max_bytes`` keeps its last ``max_bytes`` (an engine log's end is the part
    that explains a failure).

    Args:
        sources: Files, or folders whose files (not their subfolders) are copied.
        target: The folder to copy into.
        max_bytes: The most bytes copied per file.

    Returns:
        ``copied`` (file name to bytes), ``truncated`` (names), and ``errors``.
    """
    import os
    from pathlib import Path

    suffixes = {".jsonl", ".json", ".log", ".txt", ".md", ".yaml", ".csv"}
    report: dict[str, object] = {"copied": {}, "truncated": [], "errors": []}
    files = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.is_file())
        elif path.is_file():
            files.append(path)
    for file in files:
        if file.suffix not in suffixes or file.name.startswith("."):
            continue
        try:
            size = file.stat().st_size
            with open(file, "rb") as handle:
                if size > max_bytes:
                    handle.seek(size - max_bytes)
                    report["truncated"].append(file.name)
                data = handle.read(max_bytes)
            destination = Path(target) / file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            report["copied"][file.name] = len(data)
        except OSError as err:
            report["errors"].append(f"{file.name}: {err}")
    return report


def copy_weights(spec_path: str) -> int:
    """Copy the engine's weights to local scratch and verify each file (task P6.9c).

    Why: run ``P6.9-02`` loaded the weights from their notebook-output mount, which Kaggle now
    serves through FUSE.GCSFUSE, in 802.41 s, past the shot's 900 s boot limit with
    ``torch.compile`` still to come; vLLM 0.31.0 prefetches only on NFS and Lustre, and its
    memory-mapped load reads the mount in small strided pieces. NVARC holds the GPUs for hours
    before the shot, with the CPUs and the local disk mostly idle, so this process copies the
    files meanwhile, at the lowest CPU priority and the lowest best-effort IO priority. Each file
    is cut into pieces copied in parallel, each piece read in order through a handle of its own,
    so the mount sees several sequential readers instead of one random one. Every file is then
    checked at its size and its sha256 against the weights' ``MANIFEST.json`` (19.5 GB hash in
    well under a minute per core from the page cache). The status JSON is rewritten atomically
    at every step and every ``progress_s``; the engine uses the copy only when it reads ``done``
    and ``ok``. Run as a script (the notebook writes this function's source to scratch with a
    ``__main__`` line), standard library only.

    Args:
        spec_path: The spec JSON: ``source`` and ``target`` (folders), ``files`` (``name``,
            ``size``, ``sha256``, the manifest's rows), ``status`` (the status file), ``delay_s``,
            ``workers``, ``piece_bytes``, ``block_bytes``, ``nice``, ``progress_s``.

    Returns:
        0 when every file was copied and verified, else 1 (the reasons in the status's
        ``errors``).
    """
    import concurrent.futures
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import threading
    import time

    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    files = spec["files"]
    block = int(spec["block_bytes"])
    lock = threading.Lock()
    status = {
        "state": "waiting",
        "pid": os.getpid(),
        "source": spec["source"],
        "target": spec["target"],
        "files": len(files),
        "bytes": sum(int(row["size"]) for row in files),
        "manifest": [[row["name"], int(row["size"])] for row in files],
        "workers": spec["workers"],
        "piece_bytes": spec["piece_bytes"],
        "delay_s": spec["delay_s"],
        "priority": {},
        "created": time.time(),
        "start": None,
        "end": None,
        "copy_s": None,
        "verify_s": None,
        "seconds": None,
        "rate_mb_s": None,
        "copied_bytes": 0,
        "hashed_bytes": 0,
        "verified": [],
        "size_only": [],
        "ok": False,
        "errors": [],
    }

    def save() -> None:
        with lock:
            text = json.dumps(status)
        temporary = spec["status"] + ".tmp"
        with open(temporary, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(temporary, spec["status"])

    def run(job: object, items: list[tuple]) -> None:
        # Every item in a pool of `workers` threads, the status saved every progress_s; the first
        # exception cancels what has not started and is recorded.
        with concurrent.futures.ThreadPoolExecutor(int(spec["workers"])) as pool:
            pending = {pool.submit(job, *item) for item in items}
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=spec["progress_s"], return_when="FIRST_EXCEPTION"
                )
                failed = [future.exception() for future in done if future.exception()]
                if failed:
                    status["errors"].append(f"{type(failed[0]).__name__}: {failed[0]}")
                    for future in pending:
                        future.cancel()
                    break
                save()

    def copy_piece(row: dict, offset: int, length: int) -> None:
        name = row["name"]
        buffer = bytearray(min(block, length))
        view = memoryview(buffer)
        source = os.path.join(spec["source"], name)
        with open(source, "rb", buffering=0) as src:
            with open(os.path.join(spec["target"], name), "r+b", buffering=0) as dst:
                src.seek(offset)
                dst.seek(offset)
                left = length
                while left:
                    count = src.readinto(view[: min(len(buffer), left)])
                    if not count:
                        raise OSError(f"{name} ended at byte {offset + length - left}")
                    written = 0
                    while written < count:
                        written += dst.write(view[written:count])
                    left -= count
                    with lock:
                        status["copied_bytes"] += count

    def verify(row: dict) -> None:
        name, size = row["name"], int(row["size"])
        path = os.path.join(spec["target"], name)
        found = os.path.getsize(path)
        if found != size:
            raise OSError(f"{name} has {found} bytes, not {size}")
        if not row.get("sha256"):
            with lock:
                status["size_only"].append(name)
            return
        digest = hashlib.sha256()
        buffer = bytearray(block)
        view = memoryview(buffer)
        with open(path, "rb", buffering=0) as handle:
            while count := handle.readinto(view):
                digest.update(view[:count])
                with lock:
                    status["hashed_bytes"] += count
        if digest.hexdigest() != row["sha256"]:
            raise OSError(f"{name}: sha256 {digest.hexdigest()} is not the manifest's")
        with lock:
            status["verified"].append(name)

    try:
        save()
        try:
            os.nice(int(spec["nice"]))
            status["priority"]["nice"] = os.nice(0)
        except OSError as err:
            status["priority"]["nice"] = f"not set: {err}"
        # Best-effort class at its lowest level, not the idle class: idle IO can starve while
        # another process keeps the disk busy; on a scheduler without priorities either is a no-op.
        ionice = shutil.which("ionice")
        if ionice is None:
            status["priority"]["ionice"] = "not set: no ionice on PATH"
        else:
            done = subprocess.run(
                [ionice, "-c", "2", "-n", "7", "-p", str(os.getpid())],
                capture_output=True, text=True, timeout=30, check=False,
            )  # fmt: skip
            status["priority"]["ionice"] = (
                "best-effort 7" if done.returncode == 0 else f"not set: {done.stderr.strip()}"
            )
        time.sleep(float(spec["delay_s"]))
        status.update(state="copying", start=time.time())
        save()
        piece = int(spec["piece_bytes"])
        pieces = []
        for row in files:
            path = os.path.join(spec["target"], row["name"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.truncate(int(row["size"]))
            size = int(row["size"])
            pieces += [(row, at, min(piece, size - at)) for at in range(0, size, piece)]
        run(copy_piece, pieces)
        status["copy_s"] = round(time.time() - status["start"], 1)
        if not status["errors"]:
            status["state"] = "verifying"
            save()
            began = time.time()
            run(verify, [(row,) for row in files])
            status["verify_s"] = round(time.time() - began, 1)
    except BaseException as err:  # a signal or a bug: the status says so, the copy is unused
        status["errors"].append(f"{type(err).__name__}: {err}")
    end = time.time()
    # The threads fill these lists in the order they finish: sorted, the status is deterministic.
    status.update(
        state="done",
        end=end,
        ok=not status["errors"],
        verified=sorted(status["verified"]),
        size_only=sorted(status["size_only"]),
    )
    if status["start"] is not None:
        status["seconds"] = round(end - status["start"], 1)
        if status["copy_s"]:
            status["rate_mb_s"] = round(status["copied_bytes"] / status["copy_s"] / 1e6, 1)
    save()
    return 0 if status["ok"] else 1


def start_warmup(
    manifest_path: str,
    model_dir: str,
    scratch: str,
    settings: dict[str, object],
    script: str,
    environ: dict[str, str],
) -> tuple[object, dict[str, object]]:
    """Start the weights' warm-up in the background, in a session of its own (task P6.9c).

    Why: the copy (``copy_weights``) must run while NVARC solves, so it starts at the agent check
    and nothing waits for it until the agent stage. It never fails the check: a warm-up that
    cannot start is recorded, and the agent stage then loads from the input mount with the
    fallback strategy. It starts only with the copy and ``reserve_gb`` free in scratch (the
    engine's venv, about 10 GB, and its caches share it).

    Args:
        manifest_path: The weights output's ``MANIFEST.json`` (its rows: name, size, sha256).
        model_dir: The weights' folder in the input mount.
        scratch: The agent's scratch folder; the copy goes to ``<scratch>/weights/<folder>``.
        settings: ``BUILD['agent']['warmup']`` (``AGENT_WARMUP_KEYS``).
        script: The copier's source, a script that takes the spec's path.
        environ: The copier's environment.

    Returns:
        The process (None when it did not start) and the record: ``started``, then ``pid``,
        ``target``, ``status``, ``log``, ``files``, ``bytes``, ``free_bytes``, or ``reason``.
    """
    import json
    import os
    import shutil
    import subprocess
    import sys

    folder = os.path.basename(os.path.normpath(model_dir))
    target = os.path.join(scratch, "weights", folder)
    paths = {
        name: os.path.join(scratch, name)
        for name in ("warmup.json", "warmup.log", "warmup_spec.json", "warm_weights.py")
    }
    record: dict[str, object] = {"started": False, "target": target}
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            rows = json.load(handle)["files"]
        files = [
            {"name": str(row["name"]), "size": int(row["size"]), "sha256": row.get("sha256")}
            for row in rows
        ]
        total = sum(row["size"] for row in files)
        os.makedirs(scratch, exist_ok=True)
        free = shutil.disk_usage(scratch).free
        record.update(files=len(files), bytes=total, free_bytes=free)
        need = total + int(settings["reserve_gb"]) * 10**9
        if free < need:
            record["reason"] = f"{free} bytes free in {scratch}, under the copy and the reserve"
            return None, record
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        spec = {
            "source": model_dir,
            "target": target,
            "files": files,
            "status": paths["warmup.json"],
            "delay_s": settings["delay_s"],
            "workers": settings["workers"],
            "piece_bytes": int(settings["piece_mib"]) << 20,
            "block_bytes": 8 << 20,
            "nice": settings["nice"],
            "progress_s": settings["progress_s"],
        }
        with open(paths["warmup_spec.json"], "w", encoding="utf-8") as handle:
            json.dump(spec, handle)
        with open(paths["warm_weights.py"], "w", encoding="utf-8") as handle:
            handle.write(script)
        with open(paths["warmup.log"], "ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-S", paths["warm_weights.py"], paths["warmup_spec.json"]],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=environ,
            )  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as err:
        record["reason"] = f"{type(err).__name__}: {err}"
        return None, record
    record.update(
        started=True, pid=process.pid, status=paths["warmup.json"], log=paths["warmup.log"]
    )
    return process, record


def stop_warmup(process: object, status_path: str | None, wait_s: float) -> dict[str, object]:
    """Stop the warm-up if it still runs and return its outcome (task P6.9c); never raises.

    Why: a copy still running when the engine falls back to the input mount competes with the
    engine's own reads, and one still running at the stage's end has no use left.

    Args:
        process: The copier's ``subprocess.Popen``, or None when it never started.
        status_path: Its status JSON, or None.
        wait_s: Seconds to wait for its process group after SIGKILL.

    Returns:
        ``exit`` (its code, minus the signal number when killed; None if it never started),
        ``stopped`` (whether this call killed it), and ``status``: the status JSON without its
        per-file lists, or None when unreadable.
    """
    import json
    import os
    import signal
    import subprocess

    stopped = False
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            stopped = True
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            pass
    status = None
    if status_path:
        try:
            with open(status_path, encoding="utf-8") as handle:
                status = json.load(handle)
        except (OSError, ValueError):
            status = None
    if isinstance(status, dict):
        status = {k: v for k, v in status.items() if k not in ("manifest", "verified")}
    code = None if process is None else process.poll()
    return {"exit": code, "stopped": stopped, "status": status}


def warmup_weights(
    process: object,
    record: dict[str, object] | None,
    input_dir: str,
    settings: dict[str, object],
    poll_s: float,
    kill_wait_s: float,
) -> dict[str, object]:
    """Choose the folder the engine loads its weights from (task P6.9c).

    Why: the local copy loads from local disk; the input mount (FUSE.GCSFUSE in ``P6.9-02``)
    needs vLLM's page-cache prefetch to read in sequence. A copy still making progress is waited
    for: the fallback reads the same bytes from the same mount through fewer readers, so it
    cannot end sooner. The wait ends at ``wait_s``, after ``stall_s`` without progress, or when
    the copier exits; then a copy that is not ``done``, ``ok``, and at every manifest size on
    disk is not used, the copier is stopped, and the engine loads from the input with
    ``fallback_load_strategy``.

    Args:
        process: The copier's ``subprocess.Popen``, or None when it never started.
        record: ``start_warmup``'s record, or None.
        input_dir: The weights' folder in the input mount.
        settings: ``BUILD['agent']['warmup']``.
        poll_s: Seconds between reads of the status.
        kill_wait_s: Seconds to wait for the copier's group after SIGKILL.

    Returns:
        ``model_dir`` (the folder for ``--model-dir``), ``source`` (``local`` or ``input``),
        ``load_strategy`` (None, or the fallback's ``--safetensors-load-strategy``),
        ``waited_s``, ``reason`` (why not local, else None), and ``warmup``
        (``stop_warmup``'s outcome).
    """
    import contextlib
    import json
    import os
    import subprocess
    import time

    status_path = (record or {}).get("status")

    def read() -> dict[str, object]:
        try:
            with open(str(status_path), encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    began = time.monotonic()
    reason = None
    status: dict[str, object] = {}
    if process is None or not status_path:
        reason = f"the warm-up did not start: {(record or {}).get('reason', 'no record')}"
    else:
        progress, changed = -1, began
        while True:
            exited = process.poll() is not None
            status = read()
            if status.get("state") == "done":
                # Its last write comes just before it exits: give it the moment to end.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=kill_wait_s)
                break
            if exited:
                reason = f"the warm-up exited {process.poll()} before it was done"
                break
            now = time.monotonic()
            moved = int(status.get("copied_bytes") or 0) + int(status.get("hashed_bytes") or 0)
            if moved != progress:
                progress, changed = moved, now
            if now - began >= float(settings["wait_s"]):
                reason = f"the warm-up was not done after {settings['wait_s']} s"
                break
            if now - changed >= float(settings["stall_s"]):
                reason = f"the warm-up made no progress for {settings['stall_s']} s"
                break
            time.sleep(poll_s)
        if reason is None and status.get("ok") is not True:
            reason = "the warm-up failed: " + "; ".join(map(str, status.get("errors") or []))
        target = str((record or {}).get("target"))
        if reason is None:
            rows = status.get("manifest") or []
            wrong = [
                str(name)
                for name, size in rows
                if not os.path.isfile(os.path.join(target, str(name)))
                or os.path.getsize(os.path.join(target, str(name))) != size
            ]
            if wrong or not rows:
                reason = f"the copy changed after its check: {', '.join(wrong) or 'no files'}"
    waited = round(time.monotonic() - began, 1)
    outcome = stop_warmup(process, status_path, kill_wait_s)
    if reason is None:
        return {
            "model_dir": str((record or {}).get("target")),
            "source": "local",
            "load_strategy": None,
            "waited_s": waited,
            "reason": None,
            "warmup": outcome,
        }
    return {
        "model_dir": input_dir,
        "source": "input",
        "load_strategy": settings["fallback_load_strategy"],
        "waited_s": waited,
        "reason": reason,
        "warmup": outcome,
    }


def nvarc_confidence(
    challenges: str,
    ids: list[str] | None,
    candidates: str | None,
    nvarc: object,
) -> tuple[dict[str, float | None], str]:
    """Return NVARC's confidence per task as the shot computes it, and its source (task P4.7b).

    Why: the agent stage's check of the shot's queue and the rank-256 tail's choice of tasks must
    read NVARC's confidence exactly as ``arc26.agent.shot.run`` does: the tasks of the challenges
    file (``ids`` in a development run with a subset), ``shot_confidence`` over NVARC's
    candidates file, and ``confidence_without_candidates`` when that file is missing, unreadable,
    or fails.

    Args:
        challenges: The mode's challenges file.
        ids: The task ids of a development run's subset, or None for every task.
        candidates: NVARC's candidates as ``candidates_json`` writes them, or None.
        nvarc: NVARC's parsed ``submission.json``, or None.

    Returns:
        Task id to a confidence in [0, 1] (None when unknown), and ``candidates`` or
        ``submission``, the source.
    """
    import json
    from collections.abc import Mapping

    from arc26.agent.select import nvarc_from_submission
    from arc26.agent.shot import confidence_without_candidates, shot_confidence
    from arc26.agent.tasks import count_test_inputs, load_tasks

    counts = count_test_inputs(load_tasks(challenges, ids))
    attempts = nvarc_from_submission(nvarc) if isinstance(nvarc, Mapping) else None
    parsed = None
    if candidates is not None:
        try:
            with open(candidates, encoding="utf-8") as handle:
                parsed = json.load(handle)
        except (OSError, ValueError):
            parsed = None
    if isinstance(parsed, Mapping):
        try:
            return shot_confidence(counts, parsed, attempts), "candidates"
        except Exception:  # an odd file: the shot's documented fallback
            pass
    return confidence_without_candidates(counts, attempts), "submission"


def shot_queue(confidence: dict[str, float | None], run_config: dict[str, object]) -> list[str]:
    """Return the task ids the shot would queue, in its order (task P4.7b).

    Why: ``arc26.agent.shot.run``'s own selection, with its run config's ``max_confidence``,
    ``order``, ``mid_bucket``, and ``zero_bucket``; an empty queue needs no engine.

    Args:
        confidence: NVARC's confidence per task (``nvarc_confidence``).
        run_config: The shot's run config as the notebook embeds it (``AGENT_CONFIG``).

    Returns:
        The queued task ids.

    Raises:
        ConfigError: If the shot refuses the run config (``config_from_mapping``).
    """
    from arc26.agent.shot import config_from_mapping, select_tasks

    config = config_from_mapping(run_config)
    chosen = select_tasks(
        confidence, config.max_confidence, (), config.order, config.mid_bucket,
        config.zero_bucket,
    )  # fmt: skip
    return [task for task, _ in chosen]


def shot_answered(path: str) -> list[str]:
    """Return the tasks the shot answered: those with a used row in its attempt log (P4.7b).

    Args:
        path: The shot's ``shot_attempts.jsonl``; missing means none.

    Returns:
        The task ids, sorted; a line that does not parse is skipped.
    """
    import json

    answered = set()
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("used") and isinstance(row.get("task_id"), str):
            answered.add(row["task_id"])
    return sorted(answered)


def puzzle_seconds(timing_dir: str | None) -> dict[str, float]:
    """Return each task's ``puzzle`` seconds from a solver's timing records (task P4.7b).

    Args:
        timing_dir: The folder of ``timing-*.jsonl`` files, or None.

    Returns:
        Task id to its seconds (summed over its records); empty without records.
    """
    from pathlib import Path

    from arc26.timing import FILE_PREFIX, FILE_SUFFIX, PUZZLE, read_records

    if timing_dir is None or not Path(timing_dir).is_dir():
        return {}
    paths = sorted(Path(timing_dir).glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"))
    try:
        records, _ = read_records(paths)
    except OSError:
        return {}
    seconds: dict[str, float] = {}
    for record in records:
        if record.op == PUZZLE and record.puzzle_id:
            seconds[record.puzzle_id] = seconds.get(record.puzzle_id, 0.0) + record.seconds
    return seconds


def tail_tasks(
    confidence: dict[str, float | None], answered: list[str], seconds: dict[str, float]
) -> tuple[list[str], str]:
    """Return the rank-256 tail's tasks in queue order, and what ordered them (task P4.7b).

    Why: the tail solves rank 16's confidence-0 tasks (NVARC's confidence exactly 0: an output
    without a candidate) that the shot did not answer, cheapest first, so the most of them end
    before its deadline: by ascending rank-16 ``puzzle`` seconds, ties and tasks without a record
    by id after the timed ones; by id (NVARC's own queue order) when no task has a record.

    Args:
        confidence: NVARC's confidence per task (``nvarc_confidence``).
        answered: The tasks the shot answered (``shot_answered``).
        seconds: NVARC's ``puzzle`` seconds per task (``puzzle_seconds``).

    Returns:
        The task ids, and ``rank-16 puzzle seconds`` or ``task id (no puzzle records)``.
    """
    skip = set(answered)
    zero = sorted(t for t, c in confidence.items() if c == 0 and t not in skip)
    timed = sorted((seconds[t], t) for t in zero if t in seconds)
    if not timed:
        return zero, "task id (no puzzle records)"
    untimed = [t for t in zero if t not in seconds]
    source = "rank-16 puzzle seconds"
    if untimed:
        source += f", then {len(untimed)} without a record by id"
    return [t for _, t in timed] + untimed, source


def tail_deadlines(now_s: float, end_s: float, settings: dict[str, int]) -> dict[str, object]:
    r"""Return the rank-256 tail's deadlines in notebook seconds, or why it is skipped (P4.7b).

    Why: the tail runs in the time the shot leaves before the agent stage's end :math:`E`
    (``AGENT['end_s']``). Its solver's soft deadline :math:`e = E - w - g` and hard stop
    :math:`k = e + g` leave the kill gap :math:`g` (``kill_gap_s``: the rescoring and pickle
    writes after the DFS stops) and then :math:`w` (``write_s``: the selection, the merge, and
    the write) before :math:`E`, so the final check and the logs keep the shot build's reserve
    after :math:`E`. With less than ``min_s`` before :math:`e` no worker would finish a puzzle
    after its model load, and the tail is skipped.

    Args:
        now_s: Seconds since the notebook's start.
        end_s: The agent stage's end :math:`E`.
        settings: ``BUILD['agent']['tail']`` (``write_s``, ``kill_gap_s``, ``min_s``).

    Returns:
        ``start_s``, ``solver_end_s``, ``solver_kill_s``, ``left_s`` (seconds to the soft
        deadline), and ``skipped`` (None, or the reason).
    """
    solver_end = end_s - settings["write_s"] - settings["kill_gap_s"]
    left = solver_end - now_s
    skipped = None
    if left < settings["min_s"]:
        skipped = (
            f"{left:.0f} s are left before the tail's soft deadline at {solver_end:g} s, less "
            f"than min_s ({settings['min_s']} s)"
        )
    return {
        "start_s": round(now_s, 1),
        "solver_end_s": solver_end,
        "solver_kill_s": solver_end + settings["kill_gap_s"],
        "left_s": round(left, 1),
        "skipped": skipped,
    }


def tail_merge(
    current: object, nvarc: object, tail: object, counts: dict[str, int], grid_problem: object
) -> tuple[dict[str, object], list[str]]:
    """Fill the outputs no one answered with the rank-256 tail's attempts (task P4.7b).

    Why: the tail runs on tasks where rank 16 left an output without a candidate; its attempts
    replace only placeholders. An output takes the tail's attempt 1 (and its attempt 2 where the
    file's attempt 2 is a placeholder too) when the file's attempt 1 is ``[[0]]``, NVARC's
    attempt 1 is ``[[0]]`` (no NVARC candidate; the shot put no answer there), and the tail's
    attempt 1 is a valid grid other than ``[[0]]``. Every other output stays as it is.

    Args:
        current: The parsed ``submission.json`` after the shot.
        nvarc: NVARC's parsed file as the agent stage copied it (or ``current`` without one).
        tail: The tail's selection, a parsed submission of its tasks.
        counts: Task id to its number of test inputs.
        grid_problem: The validator's ``grid_problem`` (``arc26/submit.py``).

    Returns:
        A new submission, and the filled outputs as ``<task>_<index>``.
    """
    import copy

    placeholder = [[0]]

    def entry(source: object, task: str, index: int) -> dict:
        rows = source.get(task) if isinstance(source, dict) else None
        if isinstance(rows, list) and index < len(rows) and isinstance(rows[index], dict):
            return rows[index]
        return {}

    def real(grid: object) -> bool:
        return grid is not None and grid != placeholder and grid_problem(grid) is None

    merged = copy.deepcopy(current) if isinstance(current, dict) else {}
    filled = []
    for task, count in counts.items():
        rows = merged.get(task)
        if not isinstance(rows, list):
            continue
        for index in range(min(count, len(rows))):
            here, theirs = rows[index], entry(tail, task, index)
            if not isinstance(here, dict) or here.get("attempt_1") != placeholder:
                continue
            if entry(nvarc, task, index).get("attempt_1", placeholder) != placeholder:
                continue
            if not real(theirs.get("attempt_1")):
                continue
            here["attempt_1"] = theirs["attempt_1"]
            if here.get("attempt_2") == placeholder and real(theirs.get("attempt_2")):
                here["attempt_2"] = theirs["attempt_2"]
            filled.append(f"{task}_{index}")
    return merged, filled


RUNTIME_HELPERS = (
    find_inputs,
    run_logged,
    gpu_inventory,
    bf16_refusal,
    file_sha256,
    series_refusal,
    holdout_text,
    rounds_text,
)
# A script notebook's helpers (task P0.5d): no series; the --config guard, the streaming runner,
# and the writer; the search for attached notebook outputs and the argument filler (task P3.5);
# the clocks the environment check logs (task MT, MT-03 revision 3).
SCRIPT_RUNTIME_HELPERS = (
    find_inputs,
    find_input_dirs,
    fill_args,
    run_logged,
    gpu_inventory,
    bf16_refusal,
    config_override,
    file_sha256,
    run_streamed,
    save_json,
    session_clocks,
)
# A hybrid build's agent check cell holds these (task P6.5): the search for the attached
# download outputs (a script notebook's), the checks of the outputs and the driver, NVARC's
# deadlines and the session's plan, and the agent stage's helpers.
AGENT_RUNTIME_HELPERS = (
    find_input_dirs,
    driver_refusal,
    check_model_output,
    check_wheelhouse,
    nvarc_deadlines,
    agent_plan,
    candidates_json,
    merge_submission,
    gpu_apps,
    gpu_memory,
    wait_for_free_gpus,
    clean_env,
    run_supervised,
    stop_engine,
    copy_logs,
)
# A shot build with the weights' warm-up (task P6.9c) adds these to its agent check cell; the
# copier itself (copy_weights) is embedded as the script WARMUP_SCRIPT, which runs in a process of
# its own.
WARMUP_RUNTIME_HELPERS = (start_warmup, stop_warmup, warmup_weights)
# A shot build with the rank-256 tail (task P4.7b) adds these to its agent check cell: NVARC's
# confidence and the shot's queue as the shot computes them, and the tail's choice of tasks, its
# deadlines, and its merge.
TAIL_RUNTIME_HELPERS = (
    nvarc_confidence,
    shot_queue,
    shot_answered,
    puzzle_seconds,
    tail_tasks,
    tail_deadlines,
    tail_merge,
)

# An adapter build's helpers (task P3.7): the search for attached notebook outputs and the
# argument filler, which fills the adapters' {input:<owner>/<slug>} paths.
ADAPTER_RUNTIME_HELPERS = (find_input_dirs, fill_args)

# Notebook cells. Code cells are plain text so the notebook reads top to bottom; the local test
# (tests/test_notebook_local.py) runs every one of them in both modes.

HELPERS_NOTE = """\
## 2. Helpers

The mode, paths, the competition rerun's fixed deadlines, the series in effect (`SERIES`: the \
Config cell's `ARMS` in a development run, None in a competition rerun) with each arm's folders, \
and stage bookkeeping, then eight helpers copied verbatim from \
`scripts/build_kaggle_notebook.py`: file search, command runner, GPU query, the bf16 check, file \
hash, the series stop rule, and the descriptions of a hold-out arm and of a round arm. A \
development run stops at the first problem; a competition rerun (Kaggle sets \
`KAGGLE_IS_COMPETITION_RERUN` in the rerun that scores a submission) logs it and carries on, \
because only a valid file scores. `ARC26_KAGGLE_ROOT` points the notebook at a simulated \
`/kaggle` tree, as the local test does."""

HELPERS_PATHS = '''\
# Mode, paths, deadlines, and stage bookkeeping, then the runtime helpers.
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Kaggle sets this variable in the rerun that scores a submission; every other run, the commit
# run included, is development mode.
COMPETITION = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
# /kaggle on Kaggle; the local test points it at a simulated tree.
ROOT = Path(os.environ.get("ARC26_KAGGLE_ROOT", "/kaggle"))
INPUT = ROOT / "input"
WORKING = ROOT / "working"
SUBMISSION = WORKING / "submission.json"
RUN_LOGS = WORKING / "run_logs"
# The solver writes candidate pickles here and worker markers into ROOT, outside
# /kaggle/working, as the reference does.
OUTPUTS_DIR = ROOT / "inference_outputs"
# make_submission writes here, outside /kaggle/working; only finalize_submission writes
# SUBMISSION, atomically.
STAGING = ROOT / "arc26_staging" / "submission.json"
# Timing records (arc26.timing): timing-rank<rank>.jsonl per solver worker and timing-main.jsonl
# for the selection in this kernel. A development run writes them inside RUN_LOGS, so the pull
# brings them back; a competition rerun records none, so its path stays the validated one.
TIMING_DIR = None if COMPETITION else RUN_LOGS / "timing"
# Deepest file level searched under INPUT; the model's config.json sits at level 7.
SEARCH_DEPTH = 8'''

# An adapter build's Helpers lines (task P3.7, ADR-019), after the series lines.
ADAPTERS_PATHS = '''\
# Adapter arms (task P3.7, ADR-019): the Config cell's ADAPTERS in a development run. A
# competition rerun ignores them and runs SCOPE once, as a single run.
ADAPTER_ARMS = ADAPTERS if ADAPTERS and not COMPETITION else None
# Kaggle also mounts attached notebooks here (P0.5-01's session), so outputs are searched here too.
NOTEBOOK_LIBS = ROOT / "usr" / "lib" / "notebooks"
# Mount folder of every attached notebook output, by <owner>/<slug>, from the environment check.
INPUT_DIRS = {}
# The arms' candidates, one folder per arm, in /kaggle/working, so the notebook's output keeps
# them (P4.6 trains on GEN's candidates); their logs, submissions, and scores in run_logs/arms/,
# which the pull brings back; each worker's embedding cache; GEN's files.
ARM_OUTPUTS = WORKING / "arm_outputs"
ARM_LOGS = RUN_LOGS / "arms"
H_CACHE = ROOT / "h_cache"
GEN_DIR = WORKING / "gen_data"'''

HELPERS_STAGES = '''\
STAGES = {}
PROBLEMS = []


@contextlib.contextmanager
def stage(name):
    """Time one stage; in a competition rerun, log an escaping exception and carry on."""
    start = time.time()
    record = STAGES[name] = {"start_s": round(start - NOTEBOOK_START, 1)}
    print(f"== {name}, {record['start_s']} s after the start", flush=True)
    try:
        yield record
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"
        if not COMPETITION:
            raise
        traceback.print_exc()
        print(f"competition rerun: {name} failed; carrying on to write submission.json")
    finally:
        record["seconds"] = round(time.time() - start, 1)


def problem(message):
    """Stop in development mode; in a competition rerun, log the problem and carry on."""
    PROBLEMS.append(message)
    if not COMPETITION:
        raise RuntimeError(message)
    print(f"PROBLEM, carrying on (competition rerun): {message}", flush=True)'''

VALIDATOR_NOTE = """\
## 3. Submission validator

`arc26/submit.py`, copied verbatim from the attached wheel when the notebook was built. It \
defines `validate` (every task id of the challenges file present, one entry per test input, \
`attempt_1` and `attempt_2` in each, every grid a rectangular list of integers 0 to 9 with sides \
1 to 30), `repair` (keeps every valid attempt, replaces the rest with `[[0]]`, the reference's \
placeholder), and `finalize_submission`, which writes the repaired file atomically. It is \
defined here, before anything can fail, so a valid `submission.json` can be written even if the \
install below fails."""

ENVIRONMENT_NOTE = """\
## 4. Environment check

Prints the mode, the build, both interpreters (the kernel's, and the `python` on PATH, which \
runs the installer and the solver as the reference's `!python starter.py` did), the GPUs as \
`nvidia-smi` reports them, and the versions of the packages the pinned image and the install \
notebook provide. It imports no torch: a CUDA context in this kernel would take memory on GPU 0 \
from solver worker 0. Inputs are found by name under the input root, never by a hardcoded path, \
because Kaggle's mount layout changed in 2026 and may change again: the challenges \
file (the one under the competition's folder, when there are several), the base model's \
`config.json`, and the wheel. A build made for an accelerator also needs one GPU per worker, and \
bf16 on every visible GPU (compute capability 8.0 or higher): Kaggle ran an `NvidiaL4X1` push on \
its default 2xT4 without a word, and that solver failed at its first puzzle after the install \
(run `P0.3-01`). A build without an accelerator runs the solver only as a launch probe. A \
series build lists its arms. Last, it writes a placeholder \
`submission.json`, every attempt `[[0]]`, so a valid file exists even if the session is killed \
later."""

ENVIRONMENT_CODE = """\
PYTHON = CHALLENGES = DATA_DIR = MODEL_DIR = WHEEL = None
GPUS = []
with stage("environment"):
    flag = os.environ.get("KAGGLE_IS_COMPETITION_RERUN")
    print(f"mode: {'competition rerun' if COMPETITION else 'development'} ({flag=})")
    for name in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE"):
        print(f"{name}={os.environ.get(name)!r}")
    print("build:", json.dumps(BUILD, indent=1))
    solves = RUN_SOLVER or COMPETITION
    end_s, kill_s = (
        (COMPETITION_SOLVER_END_S, COMPETITION_SOLVER_KILL_S)
        if COMPETITION
        else (SOLVER_END_S, SOLVER_KILL_S)
    )
    print(
        f"solver: {'runs' if solves else 'skipped (RUN_SOLVER False)'}; soft deadline {end_s} s, "
        f"hard stop {kill_s} s after the start"
    )
    print(f"solver code: {SOLVER}")
    if SERIES is not None:
        print(
            f"series: {len(SERIES)} arms, one solver launch each, in this order; the next one "
            f"starts while the time left covers {SERIES_MARGIN} times the longest arm so far"
        )
        for arm in SERIES:
            targets = ", ".join(arm["lora_targets"])
            print(f"  {arm['name']}: r = {arm['lora_r']} on {targets}; {arm['reason']}")
            if arm.get("holdout"):
                print(f"    {holdout_text(arm['holdout'])}; gate verdicts in {gate_dirs(arm)[1]}")
            if arm.get("rounds"):
                print(f"    {rounds_text(arm['rounds'])}; round files in {round_dir(arm)}")
        print(f"timing records: {RUN_LOGS}/<arm>/timing, one folder per arm")
    else:
        if ARMS:
            print("series: ignored in a competition rerun, which runs the reference settings once")
        print(f"timing records: {TIMING_DIR or 'none (competition rerun)'}")
    PYTHON = shutil.which("python")
    print(f"kernel interpreter: {sys.executable} -> {os.path.realpath(sys.executable)}")
    print(f"python on PATH: {PYTHON} -> {PYTHON and os.path.realpath(PYTHON)}")

    GPUS = gpu_inventory()
    print(f"GPUs (nvidia-smi): {len(GPUS)}")
    for gpu in GPUS:
        print("  " + ", ".join(f"{key} {value}" for key, value in gpu.items()))
    capabilities = [gpu.get("compute_cap", "") for gpu in GPUS]
    if capabilities:
        capable = all(c.replace(".", "", 1).isdigit() and float(c) >= 8.0 for c in capabilities)
        print(f"bf16 and FlashAttention 2 (compute capability 8.0 or higher): {capable}")
    bf16 = bf16_refusal(GPUS, BUILD["accelerator"])
    for name in (
        "torch", "transformers", "peft", "accelerate", "unsloth", "unsloth_zoo", "flash_attn",
        "xformers", "triton", "numpy", "tensorflow",
    ):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        print(f"{name}: {version}")

    challenges_name = (
        "arc-agi_test_challenges.json" if COMPETITION else "arc-agi_evaluation_challenges.json"
    )
    found = find_inputs(str(INPUT), [challenges_name, "config.json", BUILD["wheel"]], SEARCH_DEPTH)
    in_competition = [p for p in found[challenges_name] if BUILD["competition"] in Path(p).parts]
    candidates = {
        "challenges": in_competition or found[challenges_name],
        "model": [
            os.path.dirname(p) for p in found["config.json"] if BUILD["model_slug"] in Path(p).parts
        ],
        "wheel": found[BUILD["wheel"]],
    }
    missing = []
    for what, paths in candidates.items():
        print(f"found {what}: {paths}")
        if len(paths) != 1:
            missing.append(f"{len(paths)} {what} candidates under {INPUT}, expected 1")
    if candidates["challenges"]:
        CHALLENGES = Path(candidates["challenges"][0])
        DATA_DIR = CHALLENGES.parent
    if candidates["model"]:
        MODEL_DIR = Path(candidates["model"][0])
    if candidates["wheel"]:
        WHEEL = Path(candidates["wheel"][0])
    if PYTHON is None:
        missing.append("no python on PATH")
    if not COMPETITION and DATA_DIR is not None:
        solutions = DATA_DIR / "arc-agi_evaluation_solutions.json"
        if not solutions.is_file():
            missing.append(f"no {solutions}: make_submission scores a development run with it")
    # A build without an accelerator runs the solver only as a launch probe, which needs no GPU.
    if solves and BUILD["accelerator"] and len(GPUS) < WORKERS:
        missing.append(
            f"the solver needs {WORKERS} GPUs, one per worker ({BUILD['accelerator']}); "
            f"found {len(GPUS)}"
        )
    if bf16 is not None:
        missing.append(bf16)
    print(f"data: {DATA_DIR}")
    print(f"model: {MODEL_DIR}")
    print(f"wheel: {WHEEL}")
    if CHALLENGES is not None:
        placeholder = finalize_submission(SUBMISSION, CHALLENGES)
        print(
            f"placeholder {SUBMISSION}: {placeholder['tasks']} tasks, "
            f"{placeholder['test_inputs']} test inputs, every attempt [[0]]"
        )
    if missing:
        problem("environment check: " + "; ".join(missing))"""

INSTALL_NOTE = """\
## 5. Install

Uninstalls tensorflow, as the reference's second cell did, so that transformers in the solver \
does not load it. Checks the wheel's sha256 against the build's, then installs it with the \
`python` on PATH: `--no-index` (offline), `--no-deps` (every dependency comes from the pinned \
image and the install notebook, so nothing in the tested stack changes), `--force-reinstall` \
(an older copy is replaced). Last, the kernel imports `arc26` and checks that it sees the copy \
the `python` on PATH installed, because the kernel's interpreter has a different path (run A: \
`/usr/bin/python3` against `/usr/local/bin/python`)."""

INSTALL_CODE = """\
INSTALLED = False
with stage("install"):
    if PYTHON is None or WHEEL is None:
        problem("install skipped: no python on PATH or no wheel (see the environment check)")
    else:
        uninstall = run_logged([PYTHON, "-m", "pip", "uninstall", "-y", "tensorflow"], timeout=900)
        if uninstall.returncode:
            problem(f"pip uninstall tensorflow exited {uninstall.returncode}")
        digest = file_sha256(str(WHEEL))
        print(f"{WHEEL}: sha256 {digest}")
        if digest != BUILD["wheel_sha256"]:
            problem(
                f"{WHEEL.name} has sha256 {digest}, the build baked {BUILD['wheel_sha256']}: "
                "attach the code dataset version made from this build"
            )
        install = run_logged(
            [
                PYTHON, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall",
                "--disable-pip-version-check", str(WHEEL),
            ],
            timeout=900,
        )
        if install.returncode:
            problem(f"pip install exited {install.returncode}")
        else:
            probe = run_logged([PYTHON, "-c", "import arc26; print(arc26.__file__)"], timeout=300)
            lines = probe.stdout.strip().splitlines()
            installed = lines[-1] if probe.returncode == 0 and lines else None
            importlib.invalidate_caches()
            try:
                import arc26

                visible = arc26.__file__
            except ImportError as err:
                visible = None
                print(f"the kernel cannot import arc26: {err}")
            print(f"python on PATH imports {installed}; the kernel imports {visible}")
            if installed and visible and os.path.realpath(installed) == os.path.realpath(visible):
                INSTALLED = True
            else:
                problem("the kernel does not import the arc26 that the python on PATH installed")"""

RUN_NOTE = """\
## 6. Run

Starts the solver with `arc26.runner.run_solver`: with `SOLVER = 'split'`, the default, the \
reference's statements split into `arc26.ttt` and `arc26.decode` (`arc26.solver`, task P2.0), or \
with `SOLVER = 'nvarc'` the moved reference code (`arc26.nvarc`), in both modes; one worker per \
GPU, in a new process of the `python` on PATH with `PYTHONHASHSEED` set to `HASH_SEED`. The \
kernel sets `PYTHONHASHSEED` too, as run A's first cell did, and the run record in `stages.json` \
lists every variable the solver's environment adds (the reference's launch variables, \
`PYTHONHASHSEED`, `PYTHONUNBUFFERED`). A development run follows the Config cell: `RUN_SOLVER`, \
`SUBSET`, and the deadlines `SOLVER_END_S` and `SOLVER_KILL_S`. A competition rerun always runs \
it, on every test task, with the fixed deadlines of the Helpers cell (12 h minus 30 and minus 20 \
minutes), so development settings never cut a rerun short; whether a build may be submitted at \
all is stated in the first cell and in `MANIFEST.md`. After the soft deadline (`--end-time`) the \
workers start no new task and stop decoding, as in the reference; at the hard stop the solver is \
killed with its process group, and the rest of the session writes `submission.json` from the \
candidates already on disk. In a development run each worker appends timing records (model load, \
and per puzzle the training, each decode batch, each rescored candidate, and the whole puzzle) to \
`run_logs/timing/timing-rank<rank>.jsonl`; a competition rerun records none. Candidate, marker, \
and timing files left by an earlier launch in the same session are removed first. A solver \
failure or a non-zero exit is printed, not raised, so the next cells still write the submission \
and the logs.

**Series mode** (`ARMS` set, development runs only): the solver runs once per arm, in the \
listed order, each with the arm's LoRA rank and target modules, its own candidate folder \
(`inference_outputs/<name>/`), and its own timing records (`run_logs/<name>/timing/`), all \
within the same soft deadline and hard stop. The next arm starts only if the time left before \
the soft deadline covers `SERIES_MARGIN` times the longest arm so far, and never after an arm \
whose solver failed; an arm not started is recorded with the reason, so the cost of a series \
stays bounded and every arm that runs solves the whole subset. A competition rerun ignores \
`ARMS` and runs the reference's LoRA settings once.

**Hold-out arms** (task P2.1, `docs/modules/loop.md`): an arm with `holdout` runs the split \
solver with one demonstration pair per puzzle held out of training and prompts; after each \
puzzle's test decode, the held-out gate decodes the held-out input into \
`gate_outputs/<name>/<task>/` (never the arm's candidate folder) and appends its verdict to \
`run_logs/<name>/gate/gate-rank<rank>.jsonl`; the gate's timing records are `gate_decode`, \
`gate_selection`, and `gate`. Gate files left by an earlier launch are removed first.

**Round arms** (task P2.3, `docs/modules/loop.md`, ADR-015): a hold-out arm with `rounds` runs, \
after each puzzle's gate, at most `max_rounds` refinement rounds, each priced by the governor \
(the config embedded in its settings) before it runs: promote, continue the adapter from a \
snapshot, gate, keep or restore. A kept round decodes the test inputs again, and the inputs not \
promoted take those candidates. The round files go to `round_outputs/<name>/<task>/` (round 0's \
candidates in `round0/` once replaced), the plans, promotions, rounds, and decodes to \
`plan-`, `promote-`, `round-`, and `decode-rank<rank>.jsonl` beside the gate's verdicts. An arm's \
`variants` (task P2.5, ADR-016) may force the in-context round, which fits nothing: the promoted \
pairs join the prompts, and a failed gate rolls back the pairs alone. Every prompt that carries \
pseudo-pairs holds those that fit the model's limit; a decode that left one out writes a line to \
`context-rank<rank>.jsonl`. Round files and records left by an earlier launch are removed first."""

RUN_CODE = """\
def solver_config(arm=None):
    \"\"\"RunConfig for the solver: the settings above (SOLVER included) and the paths found by
    the search. A series arm adds its LoRA rank, target modules, and folders, a hold-out arm
    its hold-out settings with its gate folders (task P2.1), and a round arm its round settings
    with its round folder inside them (task P2.3); None keeps RunConfig's defaults, the
    reference's LoRA settings, and the single run's folders.\"\"\"
    from arc26.runner import RunConfig

    outputs_dir, _, timing_dir = arm_dirs(arm)
    scope = {}
    if arm is not None:
        scope = {"lora_r": arm["lora_r"], "lora_targets": arm["lora_targets"]}
        if arm.get("holdout"):
            gate_dir, gate_log = gate_dirs(arm)
            scope["holdout"] = {
                **arm["holdout"], "gate_dir": str(gate_dir), "log_dir": str(gate_log)
            }
            if arm.get("rounds"):
                # Task P2.3: the round settings ride inside the hold-out settings.
                scope["holdout"]["rounds"] = {**arm["rounds"], "round_dir": str(round_dir(arm))}
    return RunConfig(
        data_dir=str(DATA_DIR),
        model_dir=str(MODEL_DIR),
        outputs_dir=str(outputs_dir),
        marker_dir=str(ROOT),
        subset=SUBSET,
        workers=WORKERS,
        hash_seed=HASH_SEED,
        timing_dir=None if timing_dir is None else str(timing_dir),
        solver=SOLVER,
        **scope,
    )


def launch(arm, record, end_s, kill_s):
    \"\"\"Run the solver once, for a series arm or the single run (None), and return its exit
    code. What it was started with goes into record; a failure is printed and recorded there,
    not raised, and returns None.\"\"\"
    try:
        from arc26.runner import run_solver, solver_env

        cfg = solver_config(arm)
        # The variables run_solver adds to this kernel's environment for the solver: the
        # reference's launch variables, PYTHONHASHSEED, and PYTHONUNBUFFERED.
        record["launch_env"] = solver_env(cfg, {"PYTHONUNBUFFERED": "1"})
        print(f"solver environment adds {record['launch_env']}")
        # Candidates, worker markers, and timing records left by an earlier launch would mix
        # into this one (timing files are appended to), and a stale marker lets a worker start
        # before the previous one has imported unsloth (the reference's workaround). Every arm of
        # a series leaves its markers behind; a fresh Kaggle session has none.
        outputs_dir, _, timing_dir = arm_dirs(arm)
        stale = [p for p in ROOT.glob("worker*") if p.name[6:].isdigit() and p.is_file()]
        if outputs_dir.is_dir():
            stale += [p for p in outputs_dir.iterdir() if p.is_file()]
        if timing_dir is not None and timing_dir.is_dir():
            stale += [p for p in timing_dir.glob("timing-*.jsonl") if p.is_file()]
        if arm is not None and arm.get("holdout"):
            # The gate's candidates and verdicts (task P2.1), for the same reason.
            gate_dir, gate_log = gate_dirs(arm)
            if gate_dir.is_dir():
                stale += [p for p in gate_dir.rglob("*") if p.is_file()]
            if gate_log.is_dir():
                stale += [p for p in gate_log.glob("gate-*.jsonl") if p.is_file()]
        if arm is not None and arm.get("rounds"):
            # The rounds' files and records (task P2.3), for the same reason.
            if round_dir(arm).is_dir():
                stale += [p for p in round_dir(arm).rglob("*") if p.is_file()]
            gate_log = gate_dirs(arm)[1]
            # context-: the decodes the length guard trimmed (task P2.5).
            for prefix in ("plan", "promote", "round", "decode", "context"):
                if gate_log.is_dir():
                    stale += [p for p in gate_log.glob(f"{prefix}-*.jsonl") if p.is_file()]
        for path in stale:
            path.unlink()
        if stale:
            print(f"removed {len(stale)} candidate, marker, and timing files of an earlier launch")
        end_time = NOTEBOOK_START + end_s
        timeout = NOTEBOOK_START + kill_s - time.time()
        if timeout <= 0:
            raise RuntimeError(f"the hard stop, {kill_s} s after the start, has passed")
        print(f"solver: end time at {end_s} s, hard stop in {timeout:.0f} s")
        code = run_solver(cfg, end_time, python=PYTHON, timeout=timeout)
        print(f"solver exit code: {code}")
        if code != 0:
            print("the solver did not finish cleanly; the next cells still write the output")
        return code
    except Exception as err:
        traceback.print_exc()
        record["solver_error"] = f"{type(err).__name__}: {err}"
        print("the solver failed; submission.json is still written from the candidates on disk")
        return None


def run_series(end_s, kill_s):
    \"\"\"Launch the solver once per arm, in order, while series_refusal lets the next arm
    start; record each arm in ARM_RECORDS and return the first arm's exit code.\"\"\"
    first, durations, failed = None, [], None
    for index, (arm, entry) in enumerate(zip(SERIES, ARM_RECORDS)):
        now_s = time.time() - NOTEBOOK_START
        refusal = series_refusal(durations, failed, now_s, end_s, SERIES_MARGIN)
        if refusal is not None:
            entry.update(status="not started", why=refusal)
            print(f"arm {arm['name']} not started: {refusal}")
            continue
        print(
            f"== arm {index + 1} of {len(SERIES)}, {arm['name']}: r = {arm['lora_r']} on "
            f"{', '.join(arm['lora_targets'])}, {now_s:.0f} s after the start",
            flush=True,
        )
        entry.update(status="ran", start_s=round(now_s, 1))
        code = launch(arm, entry, end_s, kill_s)
        seconds = time.time() - NOTEBOOK_START - now_s
        entry.update(seconds=round(seconds, 1), solver_exit=code)
        durations.append(seconds)
        if index == 0:
            first = code
        if code != 0 and failed is None:
            failed = arm["name"]
    return first


SOLVER_EXIT = None
# Series mode: one record per arm, in launch order. "skipped" stays when RUN_SOLVER is False; the
# Write and Run logs cells then still select and log each arm from the files on disk.
ARM_RECORDS = [
    {key: arm[key] for key in ("name", "lora_r", "lora_targets", "reason", "holdout") if key in arm}
    | (
        {"rounds": {k: v for k, v in arm["rounds"].items() if k != "governor"}}
        if arm.get("rounds")
        else {}
    )
    | {"status": "skipped"}
    for arm in SERIES or []
]
with stage("run") as record:
    # Set in the kernel, as run A's first cell did, so every process started from here on
    # inherits it and env.json records it; run_solver also passes it to the solver explicitly.
    os.environ["PYTHONHASHSEED"] = str(HASH_SEED)
    # A competition rerun always runs the solver, with the deadlines of the Helpers cell; the
    # Config cell's RUN_SOLVER and deadlines apply to development runs only.
    if COMPETITION:
        end_s, kill_s = COMPETITION_SOLVER_END_S, COMPETITION_SOLVER_KILL_S
    else:
        end_s, kill_s = SOLVER_END_S, SOLVER_KILL_S
    record.update(solver_end_s=end_s, solver_kill_s=kill_s)
    if SERIES is not None:
        record.update(arms=ARM_RECORDS, series_margin=SERIES_MARGIN)
    if not (RUN_SOLVER or COMPETITION):
        print("RUN_SOLVER is False: the solver is skipped")
    elif not INSTALLED:
        problem("solver skipped: arc26 is not installed (see the install stage)")
    elif SERIES is None:
        SOLVER_EXIT = launch(None, record, end_s, kill_s)
    else:
        SOLVER_EXIT = run_series(end_s, kill_s)
    record["solver_exit"] = SOLVER_EXIT"""

WRITE_NOTE = """\
## 7. Write and validate `submission.json`

`make_submission` runs the reference's selection over the candidates on disk and writes the \
result to a staging file outside `/kaggle/working`. In a development run it then scores against \
the evaluation solutions, and raises when there are no candidates (for example with \
`RUN_SOLVER = False`); that traceback is expected. A development call that returns normally is \
recorded as one `main_selection` timing record in `run_logs/timing/timing-main.jsonl`. If \
selection fails before writing anything, \
as it does when a killed worker left a pickle cut mid-write, the unreadable candidate files are \
moved aside and selection runs once more. Whatever happens, `finalize_submission` \
(from `arc26/submit.py`) runs in `finally`: it keeps every valid attempt of the staging file, \
replaces anything missing or invalid with `[[0]]`, writes `/kaggle/working/submission.json` \
atomically, and validates what it wrote: every task of the challenges file, one entry per test \
input, `attempt_1` and `attempt_2`, rectangular grids of integers 0 to 9 with sides 1 to 30.

**Series mode:** every arm that was launched (or skipped with `RUN_SOLVER = False`) is selected \
the same way from its own candidate folder, after the last solver has finished, so the kernel \
imports no model code while a solver runs. What the selection prints, the development score \
included, is copied to `run_logs/<name>/selection.log`, its timing record goes to \
`run_logs/<name>/timing/`, and the arm's own valid submission to \
`run_logs/<name>/submission.json`. `/kaggle/working/submission.json` comes from the first arm, \
the notebook configuration, or a hold-out arm of another scope (task P2.5)."""

WRITE_CODE = """\
def quarantine_unreadable_candidates(outputs_dir, quarantine):
    \"\"\"Move candidate files that do not load out of outputs_dir and return their names.

    A solver killed at the hard stop can leave a pickle cut mid-write, and one unreadable file
    makes the reference's load_decoded_results fail for every task. Only files that fail as a
    truncated or corrupt bz2 pickle does are moved; any other error (a missing numpy, say)
    propagates, so a healthy folder is never emptied.
    \"\"\"
    import bz2
    import pickle

    moved = []
    for path in sorted(outputs_dir.iterdir()) if outputs_dir.is_dir() else []:
        try:
            with bz2.BZ2File(path) as handle:
                pickle.load(handle)
        except (EOFError, OSError, pickle.UnpicklingError):
            quarantine.mkdir(parents=True, exist_ok=True)
            os.replace(path, quarantine / path.name)
            moved.append(path.name)
    return moved


def select_attempts(arm, staging, record):
    \"\"\"make_submission for a series arm or the single run (None) into staging; a failure is
    printed and recorded in record, not raised.\"\"\"
    quarantine = ROOT / "arc26_quarantine"
    if arm is not None:
        quarantine = quarantine / arm["name"]
    try:
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.unlink(missing_ok=True)
        from arc26.runner import make_submission

        try:
            make_submission(solver_config(arm), str(staging))
        except Exception:
            # Salvage only when selection failed before writing anything; in a development run
            # the reference's scoring raises after the file is written, which is expected.
            if staging.exists():
                raise
            moved = quarantine_unreadable_candidates(arm_dirs(arm)[0], quarantine)
            if not moved:
                raise
            traceback.print_exc()
            record["quarantined"] = moved
            print(f"moved {len(moved)} unreadable candidate files aside; selecting again")
            make_submission(solver_config(arm), str(staging))
    except Exception as err:
        traceback.print_exc()
        record["make_submission_error"] = f"{type(err).__name__}: {err}"
        print("make_submission raised; finalize_submission keeps what it wrote, the rest is [[0]]")


@contextlib.contextmanager
def tee_output(path):
    \"\"\"Copy everything the block prints, on stdout and stderr, to path as well.\"\"\"

    class Tee:
        def __init__(self, stream, handle):
            self.stream, self.handle = stream, handle

        def write(self, text):
            self.handle.write(text)
            return self.stream.write(text)

        def flush(self):
            self.handle.flush()
            self.stream.flush()

        def __getattr__(self, name):
            return getattr(self.stream, name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)):
            with contextlib.redirect_stderr(Tee(sys.stderr, handle)):
                yield


def select_arm(arm, staging):
    \"\"\"Series mode: select one arm's attempts with what the selection prints (the development
    score) copied to run_logs/<name>/selection.log, then write the arm's own valid submission,
    run_logs/<name>/submission.json. Returns the arm's write record.\"\"\"
    logs = arm_dirs(arm)[1]
    entry = {}
    start = time.time()
    print(f"== selection of arm {arm['name']}", flush=True)
    with tee_output(logs / "selection.log"):
        select_attempts(arm, staging, entry)
    entry["seconds"] = round(time.time() - start, 1)
    if CHALLENGES is not None:
        entry["submission"] = finalize_submission(logs / "submission.json", CHALLENGES, staging)
    return entry


SUBMISSION_REPORT = None
with stage("write") as record:
    try:
        if SERIES is None:
            select_attempts(None, STAGING, record)
        else:
            # Every arm that ran, or was skipped with RUN_SOLVER False; submission.json comes
            # from the first arm, the notebook configuration or a hold-out arm (its selection
            # writes STAGING).
            record["arms"] = {}
            for arm, entry in zip(SERIES, ARM_RECORDS):
                if entry["status"] == "not started":
                    continue
                staging = STAGING
                if arm is not SERIES[0]:
                    staging = STAGING.parent / arm["name"] / STAGING.name
                record["arms"][arm["name"]] = select_arm(arm, staging)
    finally:
        if CHALLENGES is None:
            problem("no challenges file, so no submission.json (see the environment check)")
        else:
            SUBMISSION_REPORT = finalize_submission(SUBMISSION, CHALLENGES, STAGING)
            print(json.dumps(SUBMISSION_REPORT, indent=1))
    record["submission"] = SUBMISSION_REPORT"""

LOGS_NOTE = """\
## 8. Run logs (development mode only)

Writes `/kaggle/working/run_logs/`, which `scripts/kaggle_ops.py pull` downloads: \
`candidates.json` (every candidate the solver left in `inference_outputs/`: its grid, \
`beam_score`, and `score_aug`), `pip_freeze.txt`, and `env.json` (the software environment and \
this build), exactly as run A of P0.1-01 recorded them, with `dump_run_logs` copied verbatim \
from `scripts/build_reference_run.py`; and `stages.json` (each stage's start and wall time and \
error, the problems, the solver's exit code, the submission report, and the timing folder). The \
timing records are already in `run_logs/timing/`, written by the solver's workers and by \
`make_submission`; `dump_run_logs` reads only `inference_outputs/`, so it neither copies nor \
removes them. `scripts/timing_report.py` turns them into the per-puzzle table and the cost \
model. A competition rerun writes nothing here: only `submission.json` belongs in its output.

**Series mode:** each arm that was selected gets its own `run_logs/<name>/`: `candidates.json`, \
`env.json`, and `pip_freeze.txt` from `dump_run_logs` on its candidate folder, and its \
`stages.json` (the arm's settings and reason, launch, wall time, solver exit, and selection), \
beside the `selection.log`, `submission.json`, and `timing/` written before, and for a \
hold-out arm `gate/`, the verdicts its workers wrote. The top-level `stages.json` lists every \
arm, an arm not started with the reason."""

LOGS_TAIL = """\
with stage("run_logs"):
    if COMPETITION:
        print("competition rerun: no run_logs, only submission.json")
    elif SERIES is None:
        dump_run_logs(str(OUTPUTS_DIR), str(RUN_LOGS), MAX_LOG_FILE_BYTES, BUILD)
    else:
        selected = STAGES.get("write", {}).get("arms", {})
        for arm, entry in zip(SERIES, ARM_RECORDS):
            if entry["status"] == "not started":
                continue
            outputs_dir, logs, timing_dir = arm_dirs(arm)
            dump_run_logs(str(outputs_dir), str(logs), MAX_LOG_FILE_BYTES, BUILD)
            write_json_atomic(
                logs / "stages.json",
                {"arm": entry, "write": selected.get(arm["name"]), "timing_dir": str(timing_dir)},
            )

SUMMARY = {
    "mode": "competition rerun" if COMPETITION else "development",
    "build": BUILD,
    "settings": {
        "subset": SUBSET,
        "workers": WORKERS,
        "hash_seed": HASH_SEED,
        "solver": SOLVER,
        "run_solver": RUN_SOLVER,
        "solver_end_s": SOLVER_END_S,
        "solver_kill_s": SOLVER_KILL_S,
        "arms": ARMS,
    },
    "stages": STAGES,
    "problems": PROBLEMS,
    "solver_exit": SOLVER_EXIT,
    "submission": SUBMISSION_REPORT,
    "timing_dir": None if TIMING_DIR is None or SERIES is not None else str(TIMING_DIR),
    "total_s": round(time.time() - NOTEBOOK_START, 1),
}
if not COMPETITION:
    write_json_atomic(RUN_LOGS / "stages.json", SUMMARY)
print(
    f"done after {SUMMARY['total_s']} s: submission valid "
    f"{SUBMISSION_REPORT['valid'] if SUBMISSION_REPORT else 'unknown (no report)'}, "
    f"solver exit {SOLVER_EXIT}, problems {len(PROBLEMS)}"
)
if SERIES is not None:
    print("arms: " + "; ".join(f"{entry['name']} {entry['status']}" for entry in ARM_RECORDS))"""

# A scoped build's fixed cells (task P2.6, ADR-018): its single run trains the Config cell's
# SCOPE in both modes, so each line that gives or states the single run's LoRA settings names
# SCOPE instead, the environment check prints it, and the run stage records it. (old, new) per
# cell of build_notebook; each old text occurs exactly once in its cell (_edited). A build
# without a scope keeps every cell as it was, byte for byte.
SCOPE_EDITS = {
    "helpers": (
        (
            "runs the reference's LoRA settings once, as a single run.",
            "runs SCOPE once, as a single run.",
        ),
    ),
    "environment": (
        (
            '    print(f"solver code: {SOLVER}")\n',
            '    print(f"solver code: {SOLVER}")\n'
            '    print(f"LoRA scope, both modes: {SCOPE}")\n',
        ),
        ("which runs the reference settings once", "which runs SCOPE once"),
    ),
    "run-note": (
        (
            "\n\n**Series mode** (`ARMS` set, development runs only):",
            "\n\n**Scope** (task P2.6, ADR-018): the single run trains `SCOPE`, the Config cell's "
            "LoRA scope (rank `lora_r` on the modules `lora_targets`), instead of the reference's "
            "rank 256 on all nine target modules, in both modes: `solver_config` passes it to "
            "`RunConfig`, which writes it into the solver's `--config` JSON, so a competition "
            "rerun trains what a development run trains. The environment check prints it, and "
            "the run stage of `stages.json` records it as `scope`."
            "\n\n**Series mode** (`ARMS` set, development runs only):",
        ),
        ("runs the reference's LoRA settings once.", "runs `SCOPE` once."),
    ),
    "run": (
        (
            "None keeps RunConfig's defaults, the\n"
            "    reference's LoRA settings, and the single run's folders.",
            "None takes SCOPE, the build's LoRA scope\n"
            "    (task P2.6; RunConfig's defaults when it is None), and the single run's folders.",
        ),
        ("    scope = {}\n", "    scope = dict(SCOPE or {})\n"),
        (
            "    record.update(solver_end_s=end_s, solver_kill_s=kill_s)\n",
            "    record.update(solver_end_s=end_s, solver_kill_s=kill_s, scope=SCOPE)\n",
        ),
    ),
}


# An adapter build's fixed cells (task P3.7, ADR-019), applied after SCOPE_EDITS (an adapter build
# always has a scope): the environment check finds the attached notebook outputs and lists the
# arms, the Run cell passes the arms to RunConfig in a development run, the run stage and the
# summary record them, and the notes say so. A build without adapters keeps every cell.
ADAPTER_ENVIRONMENT = """\
    if ADAPTER_ARMS is not None:
        arms_count = len(ADAPTER_ARMS["arms"])
        print(
            f"adapter arms (task P3.7, ADR-019): {arms_count} per puzzle, every arm on a puzzle "
            "before the next, in this order (python -m arc26.arms):"
        )
        for arm in ADAPTER_ARMS["arms"]:
            steps = f", {arm['steps']} steps" if arm.get("steps") else ""
            print(f"  {arm['name']}: {arm['source']}{steps}; {arm['reason']}")
        print(f"decode views per test input: {ADAPTER_ARMS['views'] or 'the reference 16'}")
        print(f"GEN: {ADAPTER_ARMS['gen']}; load check: {ADAPTER_ARMS['load_check']}")
        print(f"arm candidates in {ARM_OUTPUTS}/<arm>, logs in {ARM_LOGS}/")
        inputs = BUILD.get("notebook_inputs") or {}
        # Pinned datasets (task P3.9b) mount as datasets/<owner>/<slug>: the same search.
        datasets = BUILD.get("dataset_inputs") or {}
        roots = [str(INPUT), str(NOTEBOOK_LIBS)]
        for name, paths in find_input_dirs(roots, [*inputs, *datasets], SEARCH_DEPTH).items():
            kind = "notebook output" if name in inputs else "dataset"
            version = inputs[name] if name in inputs else datasets[name]
            print(f"found {kind} {name} (version {version}): {paths}")
            if len(paths) == 1:
                INPUT_DIRS[name] = Path(paths[0])
            else:
                missing.append(
                    f"{len(paths)} distinct folders of the {kind} {name} under {INPUT} "
                    f"and {NOTEBOOK_LIBS}, expected 1: attach version {version}"
                )
    elif ADAPTERS:
        print("adapter arms: ignored in a competition rerun, which runs SCOPE once")
"""
ADAPTER_RUN_SETTINGS = '''\
def adapter_settings():
    """The adapter arms for RunConfig (task P3.7): ADAPTERS with each {input:<owner>/<slug>}
    filled with the folder the environment check found, this session's folders, the DEV4
    challenges of the load check, and GEN's folder; the arms' reasons are left out."""
    inputs = {name: str(folder) for name, folder in INPUT_DIRS.items()}

    def fill(value):
        return None if value is None else fill_args([value], BUILD["commit"], inputs)[0]

    settings = dict(ADAPTER_ARMS)
    settings["arms"] = [
        {key: value for key, value in arm.items() if key != "reason"} for arm in settings["arms"]
    ]
    for key in ("checkpoint", "cache", "episodes_root", "check_root"):
        settings[key] = fill(settings[key])
    settings.update(outputs_root=str(ARM_OUTPUTS), log_dir=str(ARM_LOGS), cache_dir=str(H_CACHE))
    if settings["load_check"] is not None:
        settings["load_check"] = {
            **settings["load_check"],
            "run_logs": fill(settings["load_check"]["run_logs"]),
            "dev4_challenges": str(CHALLENGES),
        }
    if settings["gen"] is not None:
        settings["gen"] = {**settings["gen"], "out_dir": str(GEN_DIR)}
    return settings


'''
ADAPTER_EDITS = {
    "helpers-note": (
        (
            "and the descriptions of a hold-out arm and of a round arm.",
            "and the descriptions of a hold-out arm and of a round arm; an adapter build (task "
            "P3.7) adds the arms in effect (`ADAPTER_ARMS`), their folders, the search for "
            "attached notebook outputs, and the argument filler.",
        ),
    ),
    "environment": (
        (
            '    print(f"data: {DATA_DIR}")\n',
            ADAPTER_ENVIRONMENT + '    print(f"data: {DATA_DIR}")\n',
        ),
    ),
    "run-note": (
        (
            "\n\n**Series mode** (`ARMS` set, development runs only):",
            "\n\n**Adapter arms** (task P3.7, ADR-019, development runs only): `ADAPTERS` runs "
            "the split solver's arms entry, `python -m arc26.arms`, in one launch: it builds GEN "
            "from P3.4's validation split when `gen` is set, runs H's load check on GPU 0 (a "
            "failed check stops the run before any decode, exit 5), then one worker per GPU takes "
            "the puzzles in a fixed shuffled order and runs every arm on a puzzle before the "
            "next, the adapter replaced per arm; no puzzle starts after the soft deadline, and a "
            "started puzzle's arms may run `finish_s` longer. Each arm's candidates go to "
            "`/kaggle/working/arm_outputs/<arm>/`, its timing records, selection log, "
            "submission, and scores to `run_logs/arms/<arm>/`; the workers' records and the "
            "load check's and GEN's reports to `run_logs/arms/`. The solver writes nothing into "
            "`inference_outputs/`, so `submission.json` holds placeholders: this build is never "
            "submitted. A competition rerun ignores `ADAPTERS` and runs `SCOPE` once."
            "\n\n**Series mode** (`ARMS` set, development runs only):",
        ),
    ),
    "run": (
        (
            "def solver_config(arm=None):\n",
            ADAPTER_RUN_SETTINGS + "def solver_config(arm=None):\n",
        ),
        (
            "    scope = dict(SCOPE or {})\n",
            "    scope = dict(SCOPE or {})\n"
            "    if arm is None and ADAPTER_ARMS is not None:\n"
            "        # Task P3.7: the adapter arms, development runs only.\n"
            '        scope["adapters"] = adapter_settings()\n',
        ),
        (
            "    record.update(solver_end_s=end_s, solver_kill_s=kill_s, scope=SCOPE)\n",
            "    record.update(solver_end_s=end_s, solver_kill_s=kill_s, scope=SCOPE)\n"
            '    record["adapters"] = ADAPTER_ARMS is not None\n',
        ),
    ),
    "logs": (
        ('        "arms": ARMS,\n', '        "arms": ARMS,\n        "adapters": ADAPTERS,\n'),
    ),
    # P3.7-01's record: the arms entry selects and scores every arm into run_logs/arms/<arm>/ and
    # writes nothing to inference_outputs/, so the single run's make_submission met an empty
    # folder and the reference's scoring raised on it (a zero-size array). An adapter run skips
    # that selection; submission.json stays placeholders, as the run note says.
    "write": (
        (
            "        if SERIES is None:\n            select_attempts(None, STAGING, record)\n",
            "        if SERIES is None and ADAPTER_ARMS is not None:\n"
            "            # Task P3.7: every arm was selected and scored by the arms entry\n"
            "            # (run_logs/arms/<arm>/); inference_outputs/ is empty by design.\n"
            '            record["selection"] = "arms"\n'
            "        elif SERIES is None:\n"
            "            select_attempts(None, STAGING, record)\n",
        ),
    ),
}


def _edited(
    text: str, edits: tuple[tuple[str, str], ...], cell: str, table: str = "SCOPE_EDITS"
) -> str:
    # A fixed cell with SCOPE_EDITS (or HYBRID_EDITS) applied. An old text that is not there
    # exactly once means the cell changed without its edit: a builder bug, never a config error.
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"the {cell} cell changed: update {table} for {old!r}")
        text = text.replace(old, new)
    return text


# A hybrid build's fixed cells (task P6.5, ADR-020): the Run cell cuts NVARC's deadlines at its
# share when the agent stage will follow, and skips NVARC in an agent-only build; the Run note
# says so. (old, new) per cell, applied after SCOPE_EDITS, each old text exactly once
# (_edited). A build without an agent keeps every cell as it was, byte for byte.
HYBRID_EDITS = {
    "run-note": (
        (
            "\n\n**Series mode** (`ARMS` set, development runs only):",
            "\n\n**Hybrid build** (task P6.5, ADR-020): when `AGENT['enabled']` and the agent "
            "check passed (`AGENT_READY`), NVARC's soft deadline is its share of the session, "
            "`AGENT['nvarc_share_s']`, and its hard stop `NVARC_KILL_GAP_S` (600 s) later, in "
            "both modes (`nvarc_deadlines`: either deadline only ever shrinks); with a share of 0 "
            "NVARC does not run (an agent-only build). Otherwise NVARC keeps the deadlines above, "
            "so a rerun whose agent inputs are missing gives NVARC the whole session. The agent "
            "stage follows the Write cell."
            "\n\n**Series mode** (`ARMS` set, development runs only):",
        ),
    ),
    "run": (
        (
            "        end_s, kill_s = SOLVER_END_S, SOLVER_KILL_S\n",
            "        end_s, kill_s = SOLVER_END_S, SOLVER_KILL_S\n"
            "    # Hybrid build (task P6.5): NVARC stops at its share when the agent follows.\n"
            "    end_s, kill_s = nvarc_deadlines(\n"
            "        end_s, kill_s, AGENT if AGENT_READY else None, NVARC_KILL_GAP_S\n"
            "    )\n",
        ),
        (
            "    elif not INSTALLED:\n",
            '    elif AGENT_READY and AGENT["nvarc_share_s"] == 0:\n'
            "        record[\"skipped\"] = \"agent only: AGENT['nvarc_share_s'] is 0\"\n"
            '        print("agent-only build: NVARC does not run; the agent stage takes the '
            'session")\n'
            "    elif not INSTALLED:\n",
        ),
    ),
}

AGENT_CHECK_NOTE = """\
## 4b. Agent check (hybrid build)

A hybrid build (task P6.5, ADR-020) runs the agent solver after NVARC. This cell checks, before \
anything is installed or solved, what the agent stage will need, so that a development run stops \
here rather than after NVARC's hours, and a competition rerun learns in time whether NVARC must \
keep the whole session. It finds the two download notebooks' outputs that `kernel-metadata.json` \
attaches (the weights and the engine wheelhouse, task P6.3) as folders named by their slugs, as a \
script notebook finds its notebook inputs, and reads each output's `MANIFEST.json`: it must say \
`ok`, the weights' the pinned revision with every listed file at its size, the wheelhouse's the \
pinned vLLM, beside its installer. A build made for an accelerator also needs the engine's \
tensor parallelism in GPUs and a driver that runs the engine's CUDA 13.0 runtime; a CPU build \
runs the agent stage as a launch probe. The session's plan is printed, and a plan an edited \
`AGENT` broke counts as a failed check. When every check passes, `AGENT_READY` is True and the \
Run cell stops NVARC at its share; otherwise a development run stops here, and a competition \
rerun logs the problem, skips the agent stage, and gives NVARC its full deadlines. With \
`AGENT['enabled']` False nothing is checked and NVARC runs alone. The agent's run config \
(`AGENT_CONFIG`) is embedded here as built, with the agent stage's helpers copied verbatim from \
`scripts/build_kaggle_notebook.py`."""

AGENT_CHECK_CODE = """\
# Everything that reads AGENT runs inside the stage: an AGENT edited badly in the Config cell then
# stops a development run, and a competition rerun logs it and gives NVARC its full deadlines.
AGENT_READY = False
AGENT_MODEL_DIR = AGENT_WHEELHOUSE = AGENT_BASE_PYTHON = AGENT_PLAN = None
with stage("agent_check") as record:
    record["settings"] = AGENT
    if not AGENT["enabled"]:
        record["skipped"] = "AGENT['enabled'] is False"
        print("agent stage: disabled (AGENT['enabled'] is False); NVARC runs alone")
    else:
        AGENT_PLAN = record["plan"] = agent_plan(
            AGENT,
            AGENT_CONFIG["deadline"]["final_margin_s"],
            AGENT_CONFIG["server"]["boot_timeout_s"],
            NVARC_KILL_GAP_S,
            WRITE_RESERVE_S,
            AGENT_KILL_WAIT_S,
            SESSION_LIMIT_S,
        )
        times = [f"{key} {value:g}" for key, value in AGENT_PLAN.items() if key != "problems"]
        print("plan, seconds after the start: " + ", ".join(times))
        missing = list(AGENT_PLAN["problems"])
        model, wheels = BUILD["agent"]["model"], BUILD["agent"]["wheels"]
        # Kaggle mounts a notebook's output under the input root or, as it mounts the install
        # notebook, under usr/lib/notebooks with the slug's hyphens as underscores.
        roots = [str(INPUT), str(ROOT / "usr" / "lib" / "notebooks")]
        found = find_input_dirs(roots, [model["notebook"], wheels["notebook"]], SEARCH_DEPTH)
        for what, pinned, check, pin in (
            ("weights", model, check_model_output, model["revision"]),
            ("wheelhouse", wheels, check_wheelhouse, wheels["vllm"]),
        ):
            folders = found[pinned["notebook"]]
            print(f"found {what} {pinned['notebook']} (version {pinned['version']}): {folders}")
            if len(folders) != 1:
                missing.append(
                    f"{len(folders)} distinct folders of the {what} output {pinned['notebook']}, "
                    f"expected 1: attach version {pinned['version']}"
                )
                continue
            try:
                folder, facts = check(folders[0], pinned["folder"], pin)
            except ValueError as err:
                missing.append(f"{what}: {err}")
                continue
            record[what] = facts
            print(f"{what}: {facts}")
            if what == "weights":
                AGENT_MODEL_DIR = Path(folder)
            else:
                AGENT_WHEELHOUSE = Path(folder)
        parallel = AGENT_CONFIG["server"]["tensor_parallel_size"]
        driver = driver_refusal(GPUS, BUILD["agent"]["min_driver"])
        record.update(gpus=len(GPUS), tensor_parallel_size=parallel, driver=driver)
        if BUILD["accelerator"]:
            if len(GPUS) < parallel:
                missing.append(
                    f"the engine needs {parallel} GPUs (its tensor parallelism); nvidia-smi lists "
                    f"{len(GPUS)}"
                )
            if driver is not None:
                missing.append(driver)
        else:
            print(f"CPU build: the agent stage runs as a launch probe (driver: {driver or 'ok'})")
        base = BUILD["agent"]["base_python"]
        AGENT_BASE_PYTHON = base if os.path.isfile(base) else sys.executable
        print(f"engine venv interpreter: {AGENT_BASE_PYTHON} (built for {base})")
        try:
            AGENT_SCRATCH.mkdir(parents=True, exist_ok=True)
            record["scratch_free_bytes"] = shutil.disk_usage(AGENT_SCRATCH).free
            free = record["scratch_free_bytes"] / 1e9
            print(f"agent scratch {AGENT_SCRATCH}: {free:.1f} GB free")
        except OSError as err:
            missing.append(f"the agent scratch folder {AGENT_SCRATCH} is not usable: {err}")
        if missing:
            problem("agent check: " + "; ".join(missing))
        else:
            AGENT_READY = True
    record["ready"] = AGENT_READY
    print(f"agent ready: {AGENT_READY}")"""

AGENT_NOTE = """\
## 7b. Agent stage (hybrid build)

Runs after NVARC's `submission.json` is written and valid, in both modes, when `AGENT['enabled']` \
is True, the agent check passed, `arc26` is installed, and at least `AGENT['min_stage_s']` \
seconds are left before the agent's generation stop (`AGENT['end_s']` minus the run config's \
`final_margin_s`); otherwise it records why it was skipped and NVARC's file stands. In scratch, \
never in `/kaggle/working`: a copy of NVARC's submission (the fallback); NVARC's candidates from \
`inference_outputs/` as one JSON file (phase 2's order by NVARC's confidence); the run config. \
Then the engine wheelhouse is installed offline into a venv by the installer it ships (task \
P6.3), in a process group of its own stopped at `AGENT['install_timeout_s']` or the generation \
stop, whichever comes first. Then `python -m arc26.agent.run` (task P6.4) runs with the \
challenges of the mode, `--out /kaggle/working/submission.json`, the venv and the weights (it \
starts vLLM itself and restarts it once), NVARC's two files, `--deadline` at `AGENT['end_s']`, \
and a scratch log folder; in a development run with a `SUBSET`, on those tasks only. Before it \
starts, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and `nvidia-smi \
--query-gpu=index,memory.used --format=csv,noheader,nounits` are logged and repeated every 5 s, \
for up to 60 s, until no process is listed and no GPU uses more than `GPU_USED_MAX_MIB` MiB: \
NVARC's workers held the GPUs a moment before, and inside a container the process list can be \
empty while their memory is still held. The agent never gets a solutions path, in either mode. It \
keeps the file valid at every moment: each finished task rewrites it atomically, NVARC's attempts \
as the fill. Its output reaches this cell as it arrives. If it still runs at `AGENT['end_s']`, \
its process group gets SIGTERM (it then stops its engine and writes once more) and SIGKILL \
`AGENT['term_s']` seconds later, and an engine it left running is stopped through its pid file. \
In an agent-only build (`AGENT['nvarc_share_s']` 0), where NVARC did not run, an agent stage that \
ends at least 30 minutes before NVARC's soft deadline (a failed install or boot, a crash or a \
kill, an engine gone for good) is followed by NVARC: the engine is stopped, the GPUs are logged \
and waited for as above, NVARC's stage runs with the deadlines it would have had, and P6.2's \
selection writes the file from the agent's attempt log (`attempts.jsonl`, written before each \
rewrite, so a kill loses nothing logged) with NVARC's new attempts as its NVARC input: a verified \
program keeps attempt 1, and NVARC fills the rest. Whatever happened, the final check then \
removes a cut temporary file, keeps every real grid of the file on disk, takes NVARC's grids for \
every slot that is missing, invalid, or a `[[0]]` placeholder, and writes and validates \
`submission.json` atomically. A development run then scores NVARC's file and the final one on the \
subset with `arc26.evaluate`, after the agent has ended, and copies the agent's small logs to \
`run_logs/agent/`; a competition rerun opens no solutions file and writes nothing but \
`submission.json` to `/kaggle/working`."""

AGENT_CODE = """\
AGENT_EXIT = None
# The agent stage's files in scratch (task P6.5); none of them is in /kaggle/working.
AGENT_LOGS = AGENT_SCRATCH / "logs"
AGENT_NVARC = AGENT_SCRATCH / "nvarc_submission.json"
AGENT_CANDIDATES = AGENT_SCRATCH / "nvarc_candidates.json"
AGENT_CONFIG_FILE = AGENT_SCRATCH / "agent_config.json"
AGENT_VENV = AGENT_SCRATCH / "venv"


def agent_stage(record):
    \"\"\"Install the engine and run the agent when the stage may start, else record why not. A
    step that fails records why and ends the stage; nothing here removes NVARC's file.\"\"\"
    global AGENT_EXIT
    now_s = time.time() - NOTEBOOK_START
    stop_s = AGENT["end_s"] - AGENT_CONFIG["deadline"]["final_margin_s"]
    record.update(start_s=round(now_s, 1), generation_stop_s=stop_s)
    reasons = []
    if not AGENT["enabled"]:
        reasons.append("AGENT['enabled'] is False")
    elif not AGENT_READY:
        reasons.append("the agent check did not pass (stages.agent_check)")
    if not INSTALLED:
        reasons.append("arc26 is not installed (see the install stage)")
    if CHALLENGES is None or PYTHON is None:
        reasons.append("no challenges file or no python on PATH (see the environment check)")
    if not reasons and stop_s - now_s < AGENT["min_stage_s"]:
        reasons.append(
            f"{stop_s - now_s:.0f} s are left before the generation stop at {stop_s:g} s, less "
            f"than AGENT['min_stage_s'] ({AGENT['min_stage_s']} s)"
        )
    if reasons:
        record["skipped"] = "; ".join(reasons)
        print(f"agent stage skipped: {record['skipped']}; NVARC's submission.json stands")
        return
    # NVARC's attempts: the agent's fill, and the final check's fallback.
    AGENT_SCRATCH.mkdir(parents=True, exist_ok=True)
    write_json_atomic(AGENT_NVARC, json.loads(SUBMISSION.read_text(encoding="utf-8")))
    stale = stop_engine(str(AGENT_LOGS / "engine.pid"), AGENT_KILL_WAIT_S)
    if stale:
        record["stale_engine"] = stale
    # A fresh log folder: arc26.agent.run resumes from an attempt log it finds there (a folder
    # that cannot be removed is kept: a resume beats no agent).
    shutil.rmtree(AGENT_LOGS, ignore_errors=True)
    AGENT_LOGS.mkdir(parents=True, exist_ok=True)
    record["candidates"] = candidates_json(
        str(OUTPUTS_DIR), str(AGENT_CANDIDATES), AGENT_LOG_MAX_BYTES
    )
    write_json_atomic(AGENT_CONFIG_FILE, AGENT_CONFIG)

    # The engine: the wheelhouse installed offline into a venv by the installer it ships.
    report_path = AGENT_SCRATCH / "install.json"
    report_path.unlink(missing_ok=True)
    timeout = min(AGENT["install_timeout_s"], NOTEBOOK_START + stop_s - time.time())
    command = [
        str(AGENT_BASE_PYTHON), str(AGENT_WHEELHOUSE / "install_wheelhouse.py"),
        "--wheelhouse", str(AGENT_WHEELHOUSE), "--venv", str(AGENT_VENV),
        "--report", str(report_path), "--log", str(AGENT_SCRATCH / "install.log"),
        "--imports", "torch,vllm",
    ]
    print(f"agent stage: the engine install, at most {timeout:.0f} s", flush=True)
    began = time.time()
    code, ended = run_supervised(
        command, timeout, INSTALL_TERM_S, str(AGENT_SCRATCH), clean_env(os.environ)
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    engine = AGENT_VENV / "bin" / "vllm"
    record["install"] = {
        "exit": code,
        "ended": ended,
        "seconds": round(time.time() - began, 1),
        "ok": report.get("ok") is True,
        "problems": report.get("problems"),
        "imports": report.get("imports"),
        "engine": engine.is_file(),
    }
    if code != 0 or report.get("ok") is not True or not engine.is_file():
        record["failure"] = "the engine install failed (see install), so the agent did not run"
        print(f"agent stage: {record['failure']}; NVARC's submission.json stands")
        return

    # The agent: it starts vLLM from the venv, solves, and rewrites submission.json atomically.
    command = [
        PYTHON, "-m", "arc26.agent.run",
        "--mode", "competition" if COMPETITION else "dev",
        "--tasks", str(CHALLENGES), "--out", str(SUBMISSION),
        "--config", str(AGENT_CONFIG_FILE),
        "--engine-venv", str(AGENT_VENV), "--model-dir", str(AGENT_MODEL_DIR),
        "--nvarc-submission", str(AGENT_NVARC),
        "--deadline", repr(NOTEBOOK_START + AGENT["end_s"]),
        "--log-dir", str(AGENT_LOGS),
    ]
    if record["candidates"]["written"]:
        command += ["--nvarc-candidates", str(AGENT_CANDIDATES)]
    if SUBSET and not COMPETITION:
        command += ["--task-ids", ",".join(SUBSET)]
    record["command"] = command
    # The GPUs NVARC's workers held a moment ago (killed at the hard stop, or just exited): logged,
    # and waited for until free, before vLLM measures their memory at its boot.
    record["gpus_before_engine"] = wait_for_free_gpus(GPU_FREE_WAIT_S, GPU_USED_MAX_MIB)
    timeout = NOTEBOOK_START + AGENT["end_s"] - time.time()
    if timeout <= 0:
        record["failure"] = f"AGENT['end_s'] ({AGENT['end_s']} s) passed during the install"
        print(f"agent stage: {record['failure']}; NVARC's submission.json stands")
        return
    print(
        f"agent stage: arc26.agent.run until {AGENT['end_s']} s after the start (generation "
        f"stops at {stop_s:g} s), then SIGTERM, and SIGKILL {AGENT['term_s']} s later",
        flush=True,
    )
    began = time.time()
    AGENT_EXIT, ended = run_supervised(
        command, timeout, AGENT["term_s"], str(AGENT_LOGS), {**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    record.update(agent_exit=AGENT_EXIT, ended=ended, run_s=round(time.time() - began, 1))
    stopped = stop_engine(str(AGENT_LOGS / "engine.pid"), AGENT_KILL_WAIT_S)
    if stopped:
        record["engine_stopped"] = stopped
    try:
        summary = json.loads((AGENT_LOGS / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = None
    if isinstance(summary, dict):
        keys = ("problems", "submission", "engine", "scheduler", "deadline_in_s", "wall_s")
        record["run"] = {key: summary.get(key) for key in keys}
    if AGENT_EXIT != 0:
        print(
            f"agent stage: the agent exited {AGENT_EXIT} ({ended}); the final check keeps every "
            "valid attempt and NVARC's for the rest"
        )


def nvarc_fallback(record):
    \"\"\"The agent-only build's fallback: NVARC did not run (AGENT['nvarc_share_s'] is 0), so
    when the agent stage has ended with NVARC_FALLBACK_MIN_S or more left before NVARC's soft
    deadline (the install or the boot failed, the agent crashed or was killed, its engine was
    gone for good, or it ran out of work), NVARC's stage runs now with the deadlines it would
    have had, on GPUs the agent no longer uses: the engine is stopped, the GPUs logged and waited
    for first. P6.2's selection then picks every slot from the agent's crash-safe attempt log
    (attempts.jsonl through read_records, a cut last line skipped; never run.json, which the
    agent writes only at its exit) with NVARC's new attempts as its NVARC input, so a verified
    program keeps attempt 1 and NVARC only fills; a slot both leave empty keeps the grid of the
    agent's file. The file on disk stays the agent's, valid, until that write.\"\"\"
    global SOLVER_EXIT
    if not str(STAGES.get("run", {}).get("skipped", "")).startswith("agent only"):
        return
    if COMPETITION:
        end_s, kill_s = COMPETITION_SOLVER_END_S, COMPETITION_SOLVER_KILL_S
    else:
        end_s, kill_s = SOLVER_END_S, SOLVER_KILL_S
    now_s = time.time() - NOTEBOOK_START
    fallback = record["nvarc_fallback"] = {
        "start_s": round(now_s, 1), "solver_end_s": end_s, "solver_kill_s": kill_s
    }
    if end_s - now_s < NVARC_FALLBACK_MIN_S:
        fallback["skipped"] = (
            f"{end_s - now_s:.0f} s are left before NVARC's soft deadline, less than "
            f"{NVARC_FALLBACK_MIN_S} s"
        )
        print(f"agent-only fallback skipped: {fallback['skipped']}")
        return
    if not INSTALLED or CHALLENGES is None:
        fallback["skipped"] = "arc26 is not installed or there is no challenges file"
        return
    print(
        "agent-only build: the agent stage has ended with time left, so NVARC's stage runs now, "
        f"until its soft deadline at {end_s} s after the start",
        flush=True,
    )
    fallback["gpus_before"] = gpu_apps()
    stopped = stop_engine(str(AGENT_LOGS / "engine.pid"), AGENT_KILL_WAIT_S)
    if stopped:
        fallback["engine_stopped"] = stopped
    fallback["gpus"] = wait_for_free_gpus(GPU_FREE_WAIT_S, GPU_USED_MAX_MIB)
    SOLVER_EXIT = fallback["solver_exit"] = launch(None, fallback, end_s, kill_s)
    staging = STAGING.with_name(f"fallback-{STAGING.name}")
    select_attempts(None, staging, fallback)
    if not staging.is_file():
        fallback["merge"] = "NVARC's selection wrote no file, so the agent's file stands"
        print(f"agent-only fallback: {fallback['merge']}")
        return
    from arc26.agent.select import select_files

    attempts = AGENT_LOGS / "attempts.jsonl"
    selected = AGENT_SCRATCH / "fallback_selection.json"
    fallback["selection"] = select_files(
        attempts if attempts.is_file() else None, CHALLENGES, selected, staging
    )

    def read(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    counts = entry_counts(json.loads(CHALLENGES.read_text(encoding="utf-8")))
    merged, fallback["sources"] = merge_submission(
        read(selected), read(SUBMISSION), counts, grid_problem
    )
    write_json_atomic(SUBMISSION, merged)
    print(
        "agent-only fallback: P6.2's selection of the agent's attempts and NVARC's written, "
        f"slots by source {fallback['selection']['slots']}",
        flush=True,
    )


def final_check(record):
    \"\"\"Keep submission.json valid after the stage: remove a cut temporary file, keep every
    valid attempt on disk, NVARC's for every other slot, [[0]] where neither has one; write
    atomically only when that changes the file, then validate. Returns the report.\"\"\"
    stray = SUBMISSION.with_name(f".{SUBMISSION.name}.tmp")
    if stray.exists():
        stray.unlink()
        record["removed"] = str(stray)
    if CHALLENGES is None:
        return None
    challenges = json.loads(CHALLENGES.read_text(encoding="utf-8"))
    counts = entry_counts(challenges)

    def read(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    current = read(SUBMISSION)
    merged, sources = merge_submission(current, read(AGENT_NVARC), counts, grid_problem)
    if merged != current:
        write_json_atomic(SUBMISSION, merged)
    written = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    problems = validate(written, challenges)
    zeros = sum(
        entry[name] == [[0]]
        for entries in written.values()
        for entry in entries
        for name in ("attempt_1", "attempt_2")
    )
    return {
        "out": str(SUBMISSION),
        "rewritten": merged != current,
        "sources": sources,
        "tasks": len(counts),
        "test_inputs": sum(counts.values()),
        "zero_attempts": zeros,
        "problem_count": len(problems),
        "problems": problems[:20],
        "valid": not problems,
    }


def dev_scores(record):
    \"\"\"Development mode, after the agent has ended: NVARC's file and the final one scored on
    the subset with arc26.evaluate, and the test outputs the agent gained or lost.\"\"\"
    from arc26.evaluate import load_predictions, load_solutions, score_submission

    answers = load_solutions(DATA_DIR / "arc-agi_evaluation_solutions.json")
    ids = list(SUBSET) if SUBSET else list(json.loads(CHALLENGES.read_text(encoding="utf-8")))
    scores = {}
    for label, path in (("nvarc", AGENT_NVARC), ("final", SUBMISSION)):
        if path.exists():
            scored = score_submission(load_predictions(path), answers, ids, label)
            print("\\n".join(scored.report()))
            scores[label] = {
                "score": scored.score,
                "tasks": scored.tasks,
                "solved_outputs": scored.solved_outputs,
                "test_outputs": scored.test_outputs,
                "solved": [
                    f"{task.task_id}_{index}"
                    for task in scored.per_task
                    for index, correct in enumerate(task.correct)
                    if correct
                ],
            }
    if "nvarc" in scores:
        before, after = set(scores["nvarc"]["solved"]), set(scores["final"]["solved"])
        scores.update(gained=sorted(after - before), lost=sorted(before - after))
    record["score"] = scores


with stage("agent") as record:
    record["settings"] = AGENT
    with contextlib.suppress(OSError):
        AGENT_NVARC.unlink(missing_ok=True)
    try:
        agent_stage(record)
    except Exception as err:
        traceback.print_exc()
        record["failure"] = f"{type(err).__name__}: {err}"
        print("the agent stage failed; the final check keeps NVARC's attempts where it left none")
    try:
        nvarc_fallback(record)
    except Exception as err:
        traceback.print_exc()
        record["fallback_error"] = f"{type(err).__name__}: {err}"
    try:
        final = final_check(record)
    except Exception as err:
        traceback.print_exc()
        final = None
        record["final_check_error"] = f"{type(err).__name__}: {err}"
        print("the final check failed; submission.json is the last file written atomically")
    if final is not None:
        record["final"] = SUBMISSION_REPORT = final
        print(json.dumps(final, indent=1))
    if not COMPETITION and INSTALLED and DATA_DIR is not None and CHALLENGES is not None:
        try:
            dev_scores(record)
        except Exception as err:
            traceback.print_exc()
            record["score_error"] = f"{type(err).__name__}: {err}"
    if not COMPETITION and ("install" in record or "command" in record):
        sources = [AGENT_LOGS, AGENT_SCRATCH / "install.json", AGENT_SCRATCH / "install.log"]
        record["logs"] = copy_logs(
            [str(path) for path in sources], str(RUN_LOGS / "agent"), AGENT_LOG_MAX_BYTES
        )"""


# The agent stage's lines that give a shot build's development run the evaluation solutions
# (task P6.5b), after the task ids.
SHOT_SOLUTIONS_CODE = """\
    # The shot scores itself in a development run (shot_metrics.json, after its answers); a
    # competition rerun never names a solutions file.
    if not COMPETITION and DATA_DIR is not None:
        command += ["--solutions", str(DATA_DIR / "arc-agi_evaluation_solutions.json")]
"""
# A shot build's cells (task P6.5b): the agent stage runs SHOT_ENTRY (AGENT_ENTRY, from the agent
# check's constants) with arc26.agent.run's arguments, plus the evaluation solutions in a
# development run for the shot's own score; the agent check plans with shot_plan, since NVARC
# keeps its deadlines and the shot runs in the time it leaves; the notes say so. (old, new) per
# cell, applied after HYBRID_EDITS, each old text exactly once (_edited). A hybrid build without
# entry keeps every cell as it was, byte for byte.
SHOT_EDITS = {
    "environment": (
        (
            'GPUS = []\nwith stage("environment"):\n',
            "# The clocks that may see the time before this first cell (task P6.5b), verbatim\n"
            "# from scripts/build_kaggle_notebook.py: logged in stages.json, nothing depends on\n"
            "# them.\n"
            f"{inspect.getsource(session_clocks).rstrip()}\n\n\n"
            'GPUS = []\nwith stage("environment"):\n'
            '    STAGES["environment"]["clocks"] = session_clocks(NOTEBOOK_START, str(WORKING))\n'
            "    print(f\"clocks: {STAGES['environment']['clocks']}\")\n",
        ),
    ),
    "run-note": (
        (
            " The agent stage follows the Write cell.",
            " The agent stage follows the Write cell. In a shot build (task P6.5b) the share is "
            "the competition's soft deadline, so NVARC keeps its deadlines and the long shot runs "
            "only in the time NVARC leaves.",
        ),
    ),
    "agent-check-note": (
        (
            "The session's plan is printed, and a plan an edited `AGENT` broke counts as a failed "
            "check.",
            "The session's plan is printed, and a plan an edited `AGENT` broke counts as a failed "
            "check. A shot build (task P6.5b) prints `shot_plan`'s: NVARC keeps its deadlines, so "
            "the plan names the latest start of the agent stage that still runs the shot "
            "(`latest_start_s`, `AGENT['min_stage_s']` before the generation stop) instead of a "
            "start.",
        ),
    ),
    "agent-check": (
        (
            'AGENT_PLAN = record["plan"] = agent_plan(\n',
            '# A shot build (task P6.5b): NVARC keeps its deadlines, the shot takes what is left.\n'
            '        AGENT_PLAN = record["plan"] = shot_plan(\n',
        ),
        ("            SESSION_LIMIT_S,\n", "            SHOT_SESSION_S,\n"),
    ),
    "agent-note": (
        (
            "The agent never gets a solutions path, in either mode.",
            "The agent never gets a solutions path in a competition rerun; a shot build passes "
            "the evaluation solutions in a development run (below).",
        ),
        (
            "writes nothing but `submission.json` to `/kaggle/working`.",
            "writes nothing but `submission.json` to `/kaggle/working`.\n\n**Shot build** (task "
            "P6.5b, `docs/modules/agent_shot.md`): the stage runs `python -m arc26.agent.shot`, "
            "`P6.3-05`'s long shot, with the arguments and exit codes of `arc26.agent.run`, and "
            "in a development run `--solutions` for its own score (`shot_metrics.json`, read "
            "after its answers). NVARC kept its deadlines, so the shot runs only in the time "
            "NVARC left, when at least `AGENT['min_stage_s']` remain before the generation stop "
            "(the install, the engine's preparation and boot, the smallest wave, and margins). It "
            "ranks the tasks by NVARC's confidence from its candidates (per test output the "
            "softmax of `score_kgmon` over the output's distinct grids, the top grid's share; 0 "
            "without a candidate; per task the minimum over its outputs), takes the least sure "
            "under its run config's `max_confidence` in waves, each wave's thinking cap the "
            "largest the time left and the KV pool allow, and merges per test output: NVARC's "
            "first attempt stays and the shot's first answer takes the second slot when it "
            "differs, its second answer when the first equals NVARC's and NVARC's second slot "
            "is empty (both of the shot's answers where NVARC had no candidate). Calls still "
            "running at its end are cancelled, and their tasks keep NVARC's attempts.",
        ),
    ),
    "agent": (
        (
            '        PYTHON, "-m", "arc26.agent.run",\n',
            '        PYTHON, "-m", AGENT_ENTRY,\n',
        ),
        (
            '        command += ["--task-ids", ",".join(SUBSET)]\n',
            '        command += ["--task-ids", ",".join(SUBSET)]\n' + SHOT_SOLUTIONS_CODE,
        ),
        ('f"agent stage: arc26.agent.run until ', 'f"agent stage: {AGENT_ENTRY} until '),
        # The shot's run.json (docs/modules/agent_shot.md): its selection and its waves' report
        # instead of P6.4's scheduler and deadline.
        (
            'keys = ("problems", "submission", "engine", "scheduler", "deadline_in_s", "wall_s")',
            'keys = ("problems", "tasks", "submission", "engine", "selection", "shot")',
        ),
    ),
}

# A shot build's cells with the weights' warm-up (task P6.9c, the optional key agent.warmup):
# the agent check starts the copy before NVARC, the agent stage gives the engine the verified
# local copy (or the input mount with the fallback load strategy), and stops the copy and keeps
# its status at the end; the notes say so. (old, new) per cell, applied after SHOT_EDITS, each
# old text exactly once (_edited). A shot build without warmup keeps every cell, byte for byte.
WARMUP_EDITS = {
    "agent-check-note": (
        (
            "with the agent stage's helpers copied verbatim from "
            "`scripts/build_kaggle_notebook.py`.",
            "with the agent stage's helpers copied verbatim from "
            "`scripts/build_kaggle_notebook.py`.\n\n**Weights warm-up** (task P6.9c): when the "
            "check passes, in both modes, a background process (`WARMUP_SCRIPT`, the builder's "
            "`copy_weights`) copies the weights' files from their input mount to "
            "`<scratch>/weights/<folder>` while NVARC solves: after `BUILD['agent']['warmup']"
            "['delay_s']` (NVARC's model loads first), at the lowest CPU priority and the lowest "
            "best-effort IO priority, each file in pieces copied in parallel, then every file "
            "checked at its size and its sha256 against the weights' `MANIFEST.json`. Its status "
            "is `<scratch>/warmup.json`. It never fails the check: a warm-up that cannot start "
            "(too little free scratch, an unreadable manifest) is recorded, and the engine then "
            "loads from the input mount.",
        ),
    ),
    "agent-check": (
        (
            '            if what == "weights":\n                AGENT_MODEL_DIR = Path(folder)\n',
            '            if what == "weights":\n                AGENT_MODEL_DIR = Path(folder)\n'
            '                AGENT_WEIGHTS_MANIFEST = Path(folders[0]) / "MANIFEST.json"\n',
        ),
        (
            '    record["ready"] = AGENT_READY\n    print(f"agent ready: {AGENT_READY}")',
            '    record["ready"] = AGENT_READY\n    print(f"agent ready: {AGENT_READY}")\n'
            "    # The weights' warm-up (task P6.9c), in both modes: the files copied to local\n"
            "    # scratch while NVARC solves, so the engine loads them from local disk. It never\n"
            "    # fails the check: the agent stage falls back to the input mount.\n"
            "    if AGENT_READY and AGENT_WEIGHTS_MANIFEST is not None:\n"
            "        try:\n"
            "            AGENT_WARMUP_PROCESS, AGENT_WARMUP = start_warmup(\n"
            "                str(AGENT_WEIGHTS_MANIFEST), str(AGENT_MODEL_DIR),\n"
            "                str(AGENT_SCRATCH),\n"
            '                BUILD["agent"]["warmup"], WARMUP_SCRIPT, clean_env(os.environ),\n'
            "            )\n"
            "        except Exception as err:\n"
            '            reason = f"{type(err).__name__}: {err}"\n'
            '            AGENT_WARMUP = {"started": False, "reason": reason}\n'
            '        record["warmup"] = AGENT_WARMUP\n'
            '        print(f"weights warm-up: {AGENT_WARMUP}")',
        ),
    ),
    "agent-note": (
        (
            "running at its end are cancelled, and their tasks keep NVARC's attempts.",
            "running at its end are cancelled, and their tasks keep NVARC's attempts.\n\n"
            "**Weights warm-up** (task P6.9c): after the install, the engine's weights folder is "
            "chosen (`warmup_weights`, in `stages.agent.weights`). The warm-up's local copy is "
            "used when its status reads `done` and `ok` and every file is at its manifest size; "
            "a copy still making progress is waited for, up to `BUILD['agent']['warmup']"
            "['wait_s']` and never past `stall_s` without progress (a read of the same mount "
            "through fewer readers cannot end sooner). Otherwise the copy is stopped and vLLM "
            "loads from the input mount with `--safetensors-load-strategy` set to "
            "`fallback_load_strategy` (vLLM 0.31.0's page-cache prefetch, which it turns on by "
            "itself only on NFS and Lustre), added to the run config's `server.extra_args` in "
            "the copy written to scratch. The boot limit stays the run config's. At the stage's "
            "end a copy still running is stopped (`stages.agent.warmup`), and a development run "
            "copies `warmup.json` and `warmup.log` to `run_logs/agent/`.",
        ),
    ),
    "agent": (
        (
            "    # The agent: it starts vLLM from the venv, solves, and rewrites submission.json "
            "atomically.\n",
            "    # The weights (task P6.9c): the warm-up's local copy when it is done and\n"
            "    # verified (waited for while it makes progress), else the input mount with the\n"
            "    # fallback load strategy in the engine's flags.\n"
            '    weights = record["weights"] = warmup_weights(\n'
            "        AGENT_WARMUP_PROCESS, AGENT_WARMUP, str(AGENT_MODEL_DIR),\n"
            '        BUILD["agent"]["warmup"], WARMUP_POLL_S, AGENT_KILL_WAIT_S,\n'
            "    )\n"
            "    print(\n"
            "        f\"agent stage: the engine's weights from {weights['model_dir']} \"\n"
            "        f\"({weights['source']}, after {weights['waited_s']} s; \"\n"
            "        f\"{weights['reason'] or 'verified'})\",\n"
            "        flush=True,\n"
            "    )\n"
            '    if weights["load_strategy"]:\n'
            "        served = json.loads(json.dumps(AGENT_CONFIG))\n"
            '        served["server"]["extra_args"] = [\n'
            '            *(served["server"].get("extra_args") or []),\n'
            '            "--safetensors-load-strategy", weights["load_strategy"],\n'
            "        ]\n"
            "        write_json_atomic(AGENT_CONFIG_FILE, served)\n\n"
            "    # The agent: it starts vLLM from the venv, solves, and rewrites submission.json "
            "atomically.\n",
        ),
        (
            '"--engine-venv", str(AGENT_VENV), "--model-dir", str(AGENT_MODEL_DIR),',
            '"--engine-venv", str(AGENT_VENV), "--model-dir", weights["model_dir"],',
        ),
        (
            '    if not COMPETITION and ("install" in record or "command" in record):\n',
            "    # The weights' warm-up (task P6.9c): stopped if it still runs (a stage that did\n"
            "    # not reach the engine); a development run keeps its status and log for the\n"
            "    # pull.\n"
            '    record["warmup"] = stop_warmup(\n'
            '        AGENT_WARMUP_PROCESS, (AGENT_WARMUP or {}).get("status"), AGENT_KILL_WAIT_S\n'
            "    )\n"
            "    if not COMPETITION and AGENT_WARMUP:\n"
            '        warmup_files = [AGENT_WARMUP.get(key) for key in ("status", "log")]\n'
            '        record["warmup_logs"] = copy_logs(\n'
            "            [str(path) for path in warmup_files if path], str(RUN_LOGS / \"agent\"),\n"
            "            AGENT_LOG_MAX_BYTES,\n"
            "        )\n"
            '    if not COMPETITION and ("install" in record or "command" in record):\n',
        ),
    ),
}

# A shot build's cells with the rank-256 tail (task P4.7b, the optional key agent.tail): NVARC's
# workers record their timing in a competition rerun too (in scratch), the agent stage checks the
# shot's queue first and runs no install and no engine for an empty one, and the tail cell
# follows the agent cell; the notes say so. (old, new) per cell, applied after WARMUP_EDITS, each
# old text exactly once (_edited). A shot build without tail keeps every cell, byte for byte.
TAIL_EDITS = {
    "helpers": (
        (
            'TIMING_DIR = None if COMPETITION else RUN_LOGS / "timing"',
            "# A tail build (task P4.7b): NVARC's workers record their timing in a competition\n"
            "# rerun too, in scratch outside /kaggle/working, so the rank-256 tail can take its\n"
            "# tasks cheapest first by NVARC's puzzle seconds.\n"
            f'TIMING_DIR = ROOT / "{TAIL_NVARC_TIMING_DIR}" if COMPETITION else '
            'RUN_LOGS / "timing"',
        ),
    ),
    "agent": (
        (
            "    write_json_atomic(AGENT_CONFIG_FILE, AGENT_CONFIG)\n",
            "    write_json_atomic(AGENT_CONFIG_FILE, AGENT_CONFIG)\n"
            "    # A tail build (task P4.7b): the shot's queue as the shot computes it, from\n"
            "    # NVARC's candidates and file. Empty, the stage installs nothing, waits for no\n"
            "    # copy, and boots no engine, so the tail starts at once; on any error it runs as\n"
            "    # before.\n"
            "    try:\n"
            "        ids = list(SUBSET) if SUBSET and not COMPETITION else None\n"
            '        written = record["candidates"]["written"]\n'
            "        candidates = str(AGENT_CANDIDATES) if written else None\n"
            '        nvarc = json.loads(AGENT_NVARC.read_text(encoding="utf-8"))\n'
            "        confidence, _ = nvarc_confidence(str(CHALLENGES), ids, candidates, nvarc)\n"
            '        record["shot_queue"] = shot_queue(confidence, AGENT_CONFIG)\n'
            "    except Exception as err:\n"
            '        record["shot_queue_error"] = f"{type(err).__name__}: {err}"\n'
            "    else:\n"
            '        if not record["shot_queue"]:\n'
            '            record["skipped"] = (\n'
            '                "the shot\'s queue is empty, so no engine is installed"\n'
            "            )\n"
            "            print(f\"agent stage skipped: {record['skipped']}; the tail follows\")\n"
            "            return\n",
        ),
    ),
    "agent-note": (
        (
            "writes nothing but `submission.json` to `/kaggle/working`.\n\n**Shot build**",
            "writes nothing but `submission.json` to `/kaggle/working`.\n\n**Tail build** (task "
            "P4.7b): before the install, the stage computes the shot's queue as the shot does "
            "(its run config's selection over NVARC's candidates and file, `stages.agent."
            "shot_queue`); an empty queue installs nothing and boots no engine, and the tail "
            "cell below starts at once. The shot itself returns, and stops its engine, as soon "
            "as no queued task can start and no call is running.\n\n**Shot build**",
        ),
    ),
}

# An LLM-first build's cells (task P5.1, the optional key agent.skip_nvarc): when the agent check
# passed, the Run cell starts no NVARC (in both modes) and the Write cell runs no selection, so
# submission.json holds [[0]] placeholders until the shot rewrites it; the notes say so. The skip
# is not recorded as "agent only", so the agent-only fallback never runs. (old, new) per cell,
# applied after WARMUP_EDITS, each old text exactly once (_edited). A shot build without the key
# keeps every cell, byte for byte.
SKIP_NVARC_EDITS = {
    "run-note": (
        (
            " the long shot runs only in the time NVARC leaves.",
            " the long shot runs only in the time NVARC leaves. In an LLM-first build (task P5.1, "
            "`BUILD['agent']['skip_nvarc']`) NVARC does not run at all once the agent check has "
            "passed, in both modes: the shot takes every task for the whole session. A failed "
            "agent check gives NVARC its full deadlines, as in every hybrid build.",
        ),
    ),
    "run": (
        (
            "    elif not INSTALLED:\n",
            '    elif AGENT_READY and BUILD["agent"].get("skip_nvarc"):\n'
            "        # LLM first (task P5.1): the shot takes every task, no NVARC.\n"
            f"        record[\"skipped\"] = {SKIP_NVARC_REASON!r}\n"
            '        print("LLM-first build: NVARC does not run; the shot takes every task")\n'
            "    elif not INSTALLED:\n",
        ),
    ),
    "write-note": (
        (
            "`/kaggle/working/submission.json` atomically, and validates what it wrote:",
            "`/kaggle/working/submission.json` atomically, and validates what it wrote (in an "
            "LLM-first build whose NVARC did not run, task P5.1, no selection runs and every slot "
            "is `[[0]]` until the shot rewrites the file):",
        ),
    ),
    "write": (
        (
            "        if SERIES is None:\n            select_attempts(None, STAGING, record)\n",
            f'        if STAGES.get("run", {{}}).get("skipped") == {SKIP_NVARC_REASON!r}:\n'
            "            # LLM first (task P5.1): NVARC did not run, so nothing to select.\n"
            "            STAGING.unlink(missing_ok=True)\n"
            '            record["skipped"] = "LLM first: NVARC did not run; every slot [[0]]"\n'
            "        elif SERIES is None:\n            select_attempts(None, STAGING, record)\n",
        ),
        (
            "            SUBMISSION_REPORT = finalize_submission(SUBMISSION, CHALLENGES, "
            "STAGING)\n",
            '            raw = None if record.get("skipped") else STAGING\n'
            "            SUBMISSION_REPORT = finalize_submission(SUBMISSION, CHALLENGES, raw)\n",
        ),
    ),
    "agent-check-note": (
        (
            "the plan names the latest start of the agent stage that still runs the shot "
            "(`latest_start_s`, `AGENT['min_stage_s']` before the generation stop) instead of a "
            "start.",
            "the plan names the latest start of the agent stage that still runs the shot "
            "(`latest_start_s`, `AGENT['min_stage_s']` before the generation stop) instead of a "
            "start. In an LLM-first build (task P5.1) a passed check means NVARC does not run, so "
            "the plan's NVARC times apply only when the check fails.",
        ),
    ),
    "agent-note": (
        (
            "running at its end are cancelled, and their tasks keep NVARC's attempts.",
            "running at its end are cancelled, and their tasks keep NVARC's attempts.\n\n"
            "**LLM-first build** (task P5.1, `BUILD['agent']['skip_nvarc']`): NVARC did not run, "
            "so the file holds `[[0]]` everywhere and no candidates are passed. The shot queues "
            "every task by ascending opening bound (its run config's `selection.order: opening`; "
            "NVARC's confidence is not computed), each answered output takes the shot's two "
            "answers $S_1$ and $S_2$, and a task the shot did not reach or answer keeps its "
            "placeholders. In a development run the run config's `waves.dev_cap` lowers the "
            "thinking cap, so the version's own check is short; a competition rerun keeps it.",
        ),
    ),
}


TAIL_NOTE = """\
## 7c. Rank-256 tail (tail build)

Runs after the agent stage, in both modes (task P4.7b, `docs/tasks/P4.7b.md`). In the time the \
shot leaves before `AGENT['end_s']`, NVARC's solver runs again with the reference's LoRA \
settings (`RunConfig`'s defaults: rank 256 on the nine target modules) on rank 16's \
confidence-0 tasks: NVARC's confidence from its candidates exactly as the shot computes it \
(`nvarc_confidence`), the tasks at exactly 0 the shot did not answer, cheapest first by NVARC's \
`puzzle` seconds (`tail_tasks`; by id without records). Their challenges go to a data folder of \
the tail's own under the root, in that order, as the file the solver reads in this mode \
(`arc-agi_test_challenges.json` in a competition rerun, else \
`arc-agi_evaluation_challenges.json`, with the subset's solutions beside it for the selection's \
development score); `python -m arc26.tail` queues them in the file's order and runs \
`arc26.solver`'s workers, with candidate, marker, and timing folders of its own. Its soft \
deadline leaves `kill_gap_s` and then `write_s` before `AGENT['end_s']` (`tail_deadlines`), and \
it is skipped with less than `min_s` left. Before the workers load, the shot's engine is stopped \
through its pid file and the GPUs are waited for, as before the engine's boot; GPUs still held \
skip the tail. NVARC's selection (`make_submission`) then picks the tail's two attempts per \
output, and only an output whose attempt 1 is a placeholder in the file and in NVARC's (no NVARC \
candidate, no shot answer) takes them (`tail_merge`); every other output stays. The merged file \
is validated before it is written atomically, and any failure of the tail leaves the shot's file \
as it was. The final check then runs again; a development run scores the shot's file and the \
final one on the subset and copies the tail's candidates and record to `run_logs/tail/`; a \
competition rerun keeps them in scratch."""

TAIL_CODE = """\
TAIL = BUILD["agent"]["tail"]
# The tail's own folders: candidates and worker markers beside NVARC's, the data folder of the
# challenges it solves, and its logs and timing records (run_logs in a development run, scratch
# in a competition rerun, which writes nothing but submission.json to /kaggle/working).
TAIL_OUTPUTS = ROOT / "tail_outputs"
TAIL_MARKERS = ROOT / "tail_markers"
TAIL_DATA = ROOT / "tail_data"
TAIL_LOGS = AGENT_SCRATCH / "tail" if COMPETITION else RUN_LOGS / "tail"
TAIL_TIMING = TAIL_LOGS / "timing"
TAIL_SELECTION = TAIL_DATA / "submission.json"
# The file as the shot left it: the fallback while the tail runs, and its development score.
TAIL_SHOT_FILE = AGENT_SCRATCH / "shot_submission.json"
TAIL_MODULE = "@TAIL_MODULE@"


def read_json_file(path):
    \"\"\"A parsed JSON file, or None when it is missing or does not parse.\"\"\"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail_select(cfg, record):
    \"\"\"NVARC's selection over the tail's candidates into TAIL_SELECTION; a failure is recorded,
    and unreadable candidate files a kill cut are moved aside before one more try.\"\"\"
    from arc26.runner import make_submission

    TAIL_SELECTION.unlink(missing_ok=True)
    for attempt in (1, 2):
        try:
            make_submission(cfg, str(TAIL_SELECTION))
            return
        except Exception as err:
            # A development run's score raises after the file is written: expected.
            record["error"] = f"{type(err).__name__}: {err}"
            if TAIL_SELECTION.exists() or attempt == 2:
                return
            quarantine = ROOT / "arc26_quarantine" / "tail"
            moved = quarantine_unreadable_candidates(TAIL_OUTPUTS, quarantine)
            if not moved:
                return
            record["quarantined"] = moved


def tail_stage(record):
    \"\"\"The rank-256 tail (task P4.7b): choose the tasks, run the solver until its deadlines,
    select, and fill the placeholders; a step that cannot run records why and ends the stage,
    and nothing here writes a file that does not validate.\"\"\"
    plan = tail_deadlines(time.time() - NOTEBOOK_START, AGENT["end_s"], TAIL)
    record.update(plan)
    reasons = []
    if not INSTALLED:
        reasons.append("arc26 is not installed (see the install stage)")
    if CHALLENGES is None or PYTHON is None or MODEL_DIR is None:
        reasons.append("no challenges file, python, or model (see the environment check)")
    if not (RUN_SOLVER or COMPETITION):
        reasons.append("RUN_SOLVER is False")
    if plan["skipped"]:
        reasons.append(plan["skipped"])
    if reasons:
        record["skipped"] = "; ".join(reasons)
        print(f"tail skipped: {record['skipped']}; the shot's submission.json stands")
        return
    current = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    AGENT_SCRATCH.mkdir(parents=True, exist_ok=True)
    write_json_atomic(TAIL_SHOT_FILE, current)
    # NVARC's file as the agent stage copied it; the file itself when the stage stopped first.
    nvarc = read_json_file(AGENT_NVARC)
    if nvarc is None:
        nvarc = current
    if not AGENT_CANDIDATES.is_file():
        record["candidates"] = candidates_json(
            str(OUTPUTS_DIR), str(AGENT_CANDIDATES), AGENT_LOG_MAX_BYTES
        )
    ids = list(SUBSET) if SUBSET and not COMPETITION else None
    candidates = str(AGENT_CANDIDATES) if AGENT_CANDIDATES.is_file() else None
    confidence, record["confidence_source"] = nvarc_confidence(
        str(CHALLENGES), ids, candidates, nvarc
    )
    answered = record["shot_answered"] = shot_answered(str(AGENT_LOGS / "shot_attempts.jsonl"))
    seconds = puzzle_seconds(None if TIMING_DIR is None else str(TIMING_DIR))
    order, record["order_source"] = tail_tasks(confidence, answered, seconds)
    record["queued"] = order
    record["rank16_seconds"] = {task: seconds.get(task) for task in order}
    if not order:
        record["skipped"] = "no task at NVARC's confidence 0 that the shot did not answer"
        print(f"tail skipped: {record['skipped']}")
        return
    # The GPUs: the shot's engine stopped through its pid file, then waited for, as before the
    # engine's boot; GPUs still held would fail the workers' model loads.
    stopped = stop_engine(str(AGENT_LOGS / "engine.pid"), AGENT_KILL_WAIT_S)
    if stopped:
        record["engine_stopped"] = stopped
@WARMUP_STOP@    gpus = record["gpus"] = wait_for_free_gpus(GPU_FREE_WAIT_S, GPU_USED_MAX_MIB)
    held = gpus["after"] or any(
        row["used_mib"] > GPU_USED_MAX_MIB for row in gpus["memory_after"] or ()
    )
    if held:
        record["skipped"] = "the GPUs are still held after the wait (see gpus)"
        print(f"tail skipped: {record['skipped']}")
        return
    # The tasks, as the solver reads them in this mode, in the tail's order; a development run
    # adds the subset's solutions for the selection's own score.
    challenges = json.loads(CHALLENGES.read_text(encoding="utf-8"))
    for folder in (TAIL_OUTPUTS, TAIL_MARKERS, TAIL_DATA, TAIL_TIMING):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
    name = "arc-agi_test_challenges.json" if COMPETITION else "arc-agi_evaluation_challenges.json"
    write_json_atomic(TAIL_DATA / name, {task: challenges[task] for task in order})
    if not COMPETITION and DATA_DIR is not None:
        solutions = read_json_file(DATA_DIR / "arc-agi_evaluation_solutions.json") or {}
        write_json_atomic(
            TAIL_DATA / "arc-agi_evaluation_solutions.json",
            {task: solutions[task] for task in order if task in solutions},
        )
    from arc26.runner import RunConfig, run_solver, solver_env

    # The reference's LoRA settings: RunConfig's defaults (no scope), as P1.1-01.
    cfg = RunConfig(
        data_dir=str(TAIL_DATA),
        model_dir=str(MODEL_DIR),
        outputs_dir=str(TAIL_OUTPUTS),
        marker_dir=str(TAIL_MARKERS),
        workers=min(WORKERS, len(order)),
        hash_seed=HASH_SEED,
        timing_dir=str(TAIL_TIMING),
        solver="split",
    )
    record.update(
        workers=cfg.workers,
        lora_r=cfg.lora_r,
        lora_targets=list(cfg.lora_targets),
        launch_env=solver_env(cfg, {"PYTHONUNBUFFERED": "1"}),
    )
    timeout = NOTEBOOK_START + plan["solver_kill_s"] - time.time()
    if timeout <= 0:
        record["skipped"] = f"the tail's hard stop at {plan['solver_kill_s']} s has passed"
        return
    print(
        f"tail: {len(order)} tasks on {cfg.workers} workers, rank {cfg.lora_r}, soft deadline "
        f"{plan['solver_end_s']} s, hard stop in {timeout:.0f} s ({record['order_source']})",
        flush=True,
    )
    began = time.time()
    record["solver_exit"] = run_solver(
        cfg, NOTEBOOK_START + plan["solver_end_s"], python=PYTHON, timeout=timeout,
        module=TAIL_MODULE,
    )
    record["solver_s"] = round(time.time() - began, 1)
    finished = puzzle_seconds(str(TAIL_TIMING))
    started = {
        path.name.split(".", 1)[0].rpartition("_")[0]
        for path in TAIL_OUTPUTS.iterdir()
        if path.is_file()
    }
    record["started"] = [task for task in order if task in started or task in finished]
    record["finished"] = [task for task in order if task in finished]
    record["tail_seconds"] = {task: round(finished[task], 1) for task in record["finished"]}
    selection = record["selection"] = {}
    tail_select(cfg, selection)
    selected = read_json_file(TAIL_SELECTION)
    if not isinstance(selected, dict):
        record["merge"] = "the tail's selection wrote no file; the shot's file stands"
        print(f"tail: {record['merge']}")
        return
    counts = entry_counts(challenges)
    merged, filled = tail_merge(
        json.loads(SUBMISSION.read_text(encoding="utf-8")), nvarc, selected, counts, grid_problem
    )
    record["filled"] = filled
    problems = validate(merged, challenges)
    if problems:
        record["merge"] = f"the merged file has {len(problems)} problems; the shot's file stands"
        record["merge_problems"] = problems[:20]
        print(f"tail: {record['merge']}")
        return
    if filled:
        write_json_atomic(SUBMISSION, merged)
    record["merge"] = f"{len(filled)} outputs filled"
    print(f"tail: {record['merge']}: {', '.join(filled) or 'none'}", flush=True)


def tail_scores(record):
    \"\"\"Development mode: the shot's file and the final one scored on the subset with
    arc26.evaluate, and the test outputs the tail gained or lost.\"\"\"
    from arc26.evaluate import load_predictions, load_solutions, score_submission

    answers = load_solutions(DATA_DIR / "arc-agi_evaluation_solutions.json")
    ids = list(SUBSET) if SUBSET else list(json.loads(CHALLENGES.read_text(encoding="utf-8")))
    scores = {}
    for label, path in (("shot", TAIL_SHOT_FILE), ("final", SUBMISSION)):
        if path.exists():
            scored = score_submission(load_predictions(path), answers, ids, label)
            print("\\n".join(scored.report()))
            scores[label] = {
                "score": scored.score,
                "tasks": scored.tasks,
                "solved_outputs": scored.solved_outputs,
                "test_outputs": scored.test_outputs,
                "solved": [
                    f"{task.task_id}_{index}"
                    for task in scored.per_task
                    for index, correct in enumerate(task.correct)
                    if correct
                ],
            }
    if "shot" in scores and "final" in scores:
        before, after = set(scores["shot"]["solved"]), set(scores["final"]["solved"])
        scores.update(gained=sorted(after - before), lost=sorted(before - after))
    record["score"] = scores


with stage("tail") as record:
    record["settings"] = TAIL
    try:
        tail_stage(record)
    except Exception as err:
        traceback.print_exc()
        record["failure"] = f"{type(err).__name__}: {err}"
        print("the tail failed; the shot's submission.json stands")
    try:
        final = final_check(record)
    except Exception as err:
        traceback.print_exc()
        final = None
        record["final_check_error"] = f"{type(err).__name__}: {err}"
    if final is not None:
        record["final"] = SUBMISSION_REPORT = final
        print(json.dumps(final, indent=1))
    if not COMPETITION and INSTALLED and DATA_DIR is not None and CHALLENGES is not None:
        try:
            tail_scores(record)
        except Exception as err:
            traceback.print_exc()
            record["score_error"] = f"{type(err).__name__}: {err}"
    if not COMPETITION and "solver_exit" in record:
        record["candidates_log"] = candidates_json(
            str(TAIL_OUTPUTS), str(TAIL_LOGS / "candidates.json"), AGENT_LOG_MAX_BYTES
        )
try:
    TAIL_LOGS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(TAIL_LOGS / "tail.json", STAGES["tail"])
except Exception as err:
    print(f"the tail's record was not written: {type(err).__name__}: {err}")"""

# The tail's warm-up line (a tail build with agent.warmup): a copy still running stops before the
# tail's workers load the base model from the same mounts.
TAIL_WARMUP_STOP = (
    "    # The weights' warm-up (task P6.9c): a copy still running stops before the workers load.\n"
    '    record["warmup"] = stop_warmup(\n'
    '        AGENT_WARMUP_PROCESS, (AGENT_WARMUP or {}).get("status"), AGENT_KILL_WAIT_S\n'
    "    )\n"
)


def tail_cell_source(agent: AgentSettings) -> str:
    """Return a tail build's tail cell (task P4.7b).

    Args:
        agent: The build's agent stage, its ``tail`` set.

    Returns:
        Python source: ``TAIL_CODE`` with the entry module, and the warm-up's stop when the build
        has one.
    """
    stop = TAIL_WARMUP_STOP if agent.warmup is not None else ""
    return TAIL_CODE.replace("@TAIL_MODULE@", TAIL_MODULE).replace("@WARMUP_STOP@", stop)


# The Config cell's lines on a shot build's agent stage (task P6.5b), in place of AGENT_COMMENT.
SHOT_AGENT_COMMENT = (
    "# The agent stage (task P6.5b, ADR-020), in both modes, a competition rerun included: the",
    "# long shot, python -m arc26.agent.shot. 'nvarc_share_s' is the competition's soft deadline,",
    "# so NVARC keeps its deadlines; with 'enabled' True and the agent check passed, the engine is",
    "# then installed offline (at most 'install_timeout_s') and the shot runs on NVARC's least",
    "# sure tasks until 'end_s', its generation stopping the run config's final_margin_s earlier;",
    "# SIGTERM at 'end_s', SIGKILL 'term_s' later. The stage starts only with 'min_stage_s' left",
    "# before the generation stop. Seconds after NOTEBOOK_START. 'enabled' False: NVARC alone.",
)


# The Config cell's lines on an LLM-first build's agent stage (task P5.1), in place of
# SHOT_AGENT_COMMENT.
SKIP_NVARC_AGENT_COMMENT = (
    "# The agent stage (task P5.1, LLM first), in both modes, a competition rerun included: the",
    "# long shot, python -m arc26.agent.shot, on every task. With 'enabled' True and the agent",
    "# check passed NVARC does not run (BUILD['agent']['skip_nvarc']); the engine is installed",
    "# offline (at most 'install_timeout_s') and the shot runs until 'end_s', its generation",
    "# stopping the run config's final_margin_s earlier; SIGTERM at 'end_s', SIGKILL 'term_s'",
    "# later. The stage starts only with 'min_stage_s' left before the generation stop.",
    "# 'nvarc_share_s' applies only when the check fails. Seconds after NOTEBOOK_START.",
    "# 'enabled' False: NVARC alone.",
)


# DEV240's lines in the environment check (task P6.5b), before this line of the cell.
DEV240_ANCHOR = (
    "    # A build without an accelerator runs the solver only as a launch probe, which needs no "
    "GPU.\n"
)
DEV240_ENVIRONMENT_CODE = f"""\
    # DEV240 (task P6.5b), development runs only: every evaluation task and a copy of each under
    # a derived id, written under the root; from here on NVARC, its selection and score, the
    # agent stage, and the final scores read DATA_DIR there.
    if DEV240 and not COMPETITION and DATA_DIR is not None:
        try:
            DEV240_REPORT = dev240_files(str(DATA_DIR), str(ROOT / "{DEV240_DIR}"))
        except (OSError, ValueError) as err:
            missing.append(f"DEV240 could not be written: {{err}}")
        else:
            CHALLENGES = Path(DEV240_REPORT["challenges"])
            DATA_DIR = CHALLENGES.parent
            print(
                f"DEV240: {{DEV240_REPORT['tasks']}} tasks, {{DEV240_REPORT['test_inputs']}} test "
                f"inputs in {{DATA_DIR}}; the copies' ids checked against "
                f"{{DEV240_REPORT['checked']}}"
            )
"""


def dev240_edits() -> dict[str, tuple[tuple[str, str], ...]]:
    """Return a DEV240 build's edits of two fixed cells (task P6.5b).

    Why: in a development run the environment check writes DEV240's files (``dev240_files``,
    its source verbatim at the cell's top) and points ``CHALLENGES`` and ``DATA_DIR`` at them,
    so NVARC, its selection and score, the agent stage, and the final scores all read the 240
    tasks; ``stages.json`` records what was written. A competition rerun is unchanged. Applied
    with ``_edited``, so every other build keeps its cells, byte for byte.

    Returns:
        (old, new) pairs per cell name.
    """
    helper = inspect.getsource(dev240_files).rstrip()
    return {
        "environment": (
            (
                "PYTHON = CHALLENGES = DATA_DIR = MODEL_DIR = WHEEL = None\n",
                "# DEV240's helper (task P6.5b), verbatim from scripts/build_kaggle_notebook.py.\n"
                f"{helper}\n\n\n"
                "DEV240_REPORT = None\n"
                "PYTHON = CHALLENGES = DATA_DIR = MODEL_DIR = WHEEL = None\n",
            ),
            (DEV240_ANCHOR, DEV240_ENVIRONMENT_CODE + DEV240_ANCHOR),
        ),
        "logs": (
            (
                '        "subset": SUBSET,\n',
                '        "subset": SUBSET,\n        "dev240": DEV240_REPORT,\n',
            ),
        ),
    }


# Script notebook cells (task P0.5d). The Install cell is the submission notebook's, run only
# while no problem stopped the run; the other cells are the script's own. The local test
# (tests/test_notebook_local.py) runs every one of them on the simulated tree.

SCRIPT_HELPERS_NOTE = """\
## 2. Helpers

Paths under `ARC26_KAGGLE_ROOT` (default `/kaggle`; the local test points it at a simulated tree) \
and stage bookkeeping, then ten helpers copied verbatim from \
`scripts/build_kaggle_notebook.py`: file search, the search for attached notebook outputs, the \
argument filler, command runner, GPU query, the bf16 check, the `--config` guard, file hash, the \
streaming runner with the hard stop, and a JSON writer. A problem stops the run \
without raising: every later stage is skipped, and the Run logs cell still writes `run_logs/`, \
so a failed run comes back with its logs. A competition rerun (Kaggle sets \
`KAGGLE_IS_COMPETITION_RERUN` in the rerun that scores a submission) stops in this cell, because \
a script notebook is never submitted."""

SCRIPT_HELPERS_PATHS = '''\
# Paths and stage bookkeeping, then the runtime helpers.
import contextlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Kaggle sets this variable in the rerun that scores a submission. A script notebook is never
# submitted, so a rerun stops here, before anything runs or is written.
if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    raise RuntimeError(
        "competition rerun of a script notebook: it is never submitted and writes no "
        "submission.json; submit a build of the submission notebook"
    )
# /kaggle on Kaggle; the local test points it at a simulated tree.
ROOT = Path(os.environ.get("ARC26_KAGGLE_ROOT", "/kaggle"))
INPUT = ROOT / "input"
WORKING = ROOT / "working"
# The script writes its own outputs here (BRIEF A8), and the notebook its logs.
RUN_LOGS = WORKING / "run_logs"
# Where a solver writes its candidates; dump_run_logs reads it, as in a submission build.
OUTPUTS_DIR = ROOT / "inference_outputs"
# The notebook's own files in RUN_LOGS; a script that writes one of them is reported.
NOTEBOOK_LOGS = ("candidates.json", "env.json", "pip_freeze.txt", "stages.json")
# Kaggle also mounts attached notebooks here: the install notebook as
# notebooks/<owner>/<slug with underscores> (P0.5-01's session), so outputs are searched here too.
NOTEBOOK_LIBS = ROOT / "usr" / "lib" / "notebooks"
# Deepest file level searched under INPUT; the model's config.json sits at level 7.
SEARCH_DEPTH = 8'''

SCRIPT_HELPERS_STAGES = '''\
STAGES = {}
PROBLEMS = []


class Stop(Exception):
    """A problem that stops the run, raised by problem() and recorded by stage()."""


@contextlib.contextmanager
def stage(name):
    """Time one stage and record what stopped it instead of raising, so that the Run logs cell
    still writes the logs; every later stage but that one is skipped."""
    start = time.time()
    record = STAGES[name] = {"start_s": round(start - NOTEBOOK_START, 1)}
    print(f"== {name}, {record['start_s']} s after the start", flush=True)
    try:
        yield record
    except Exception as err:
        if isinstance(err, Stop):
            reason = str(err)
        else:
            traceback.print_exc()
            reason = f"{name} failed: {type(err).__name__}: {err}"
        record["error"] = reason
        PROBLEMS.append(reason)
        print(f"STOPPED: {reason}", flush=True)
    finally:
        record["seconds"] = round(time.time() - start, 1)


def problem(message):
    """Stop the run: leave the stage with the problem recorded."""
    raise Stop(message)'''

SCRIPT_ENVIRONMENT_NOTE = """\
## 3. Environment check

Logs the clocks that may see the time Kaggle spends before the notebook's first cell \
(`/proc/uptime`, the ages of PID 1, of the kernel, and of the working folder; \
`stages.environment.clocks`, logging only). Prints the build, the script's command line, both \
interpreters (the kernel's, and the `python` on PATH, which installs the wheel and runs the \
script), the GPUs as `nvidia-smi` reports them, \
and the package versions of the pinned image and the install notebook. It imports no torch. \
Inputs are found by name under the input root, as in the submission notebook: the challenges \
file under the competition's folder, the base model's `config.json`, the wheel, and beside the \
wheel the script and its config, whose sha256 must equal `BUILD`'s. The outputs of other \
notebooks that the build attaches (`notebook_inputs` in `BUILD`) are found as folders named by \
their slugs, with hyphens or underscores, under `/kaggle/input` and \
`/kaggle/usr/lib/notebooks`: a folder inside one named by the owner is preferred, folders with \
the same files are one output mounted twice, and anything but one folder stops the run. A build \
made for an \
accelerator needs bf16 on every visible GPU (compute capability 8.0 or higher): Kaggle ran an \
`NvidiaL4X1` push on its default 2xT4 without a word (run `P0.3-01`). Any problem stops the run \
here, before the install."""

SCRIPT_ENVIRONMENT_CODE = """\
PYTHON = CHALLENGES = MODEL_DIR = WHEEL = SCRIPT_FILE = SCRIPT_CONFIG_FILE = None
GPUS = []
# Mount folder of every attached notebook output, by <owner>/<slug> (task P3.5).
INPUT_DIRS = {}
with stage("environment"):
    # The clocks that may see the time before the notebook's first cell (task MT): logged in
    # stages.json, nothing depends on them.
    STAGES["environment"]["clocks"] = session_clocks(NOTEBOOK_START, str(WORKING))
    print(f"clocks: {STAGES['environment']['clocks']}")
    print("mode: script notebook (development; never submitted)")
    for name in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE"):
        print(f"{name}={os.environ.get(name)!r}")
    print("build:", json.dumps(BUILD, indent=1))
    print(
        f"script: python {BUILD['script']} --config {BUILD['script_config']}"
        + "".join(f" {arg}" for arg in ARGS)
        + f"; hard stop {SCRIPT_KILL_S} s after the start"
    )
    PYTHON = shutil.which("python")
    print(f"kernel interpreter: {sys.executable} -> {os.path.realpath(sys.executable)}")
    print(f"python on PATH: {PYTHON} -> {PYTHON and os.path.realpath(PYTHON)}")

    GPUS = gpu_inventory()
    print(f"GPUs (nvidia-smi): {len(GPUS)}")
    for gpu in GPUS:
        print("  " + ", ".join(f"{key} {value}" for key, value in gpu.items()))
    bf16 = bf16_refusal(GPUS, BUILD["accelerator"])
    if BUILD["accelerator"]:
        print(f"bf16 on every GPU (compute capability 8.0 or higher): {bf16 is None}")
    for name in (
        "torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "unsloth",
        "flash_attn", "triton", "numpy", "pyyaml", "tensorflow",
    ):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        print(f"{name}: {version}")

    script_name = Path(BUILD["script"]).name
    config_name = Path(BUILD["script_config"]).name
    challenges_name = "arc-agi_evaluation_challenges.json"
    names = [challenges_name, "config.json", BUILD["wheel"], script_name, config_name]
    found = find_inputs(str(INPUT), names, SEARCH_DEPTH)
    in_competition = [p for p in found[challenges_name] if BUILD["competition"] in Path(p).parts]
    # The script and its config are the files beside the wheel, in the code dataset; a file of
    # the same name elsewhere under the input root is never run.
    dataset = {os.path.dirname(p) for p in found[BUILD["wheel"]]}
    candidates = {
        "challenges": in_competition or found[challenges_name],
        "model": [
            os.path.dirname(p) for p in found["config.json"] if BUILD["model_slug"] in Path(p).parts
        ],
        "wheel": found[BUILD["wheel"]],
        "script": [p for p in found[script_name] if os.path.dirname(p) in dataset],
        "script config": [p for p in found[config_name] if os.path.dirname(p) in dataset],
    }
    missing = []
    for what, paths in candidates.items():
        print(f"found {what}: {paths}")
        if len(paths) != 1:
            missing.append(f"{len(paths)} {what} candidates under {INPUT}, expected 1")
    if candidates["challenges"]:
        CHALLENGES = Path(candidates["challenges"][0])
    if candidates["model"]:
        MODEL_DIR = Path(candidates["model"][0])
    if candidates["wheel"]:
        WHEEL = Path(candidates["wheel"][0])
    if candidates["script"]:
        SCRIPT_FILE = Path(candidates["script"][0])
    if candidates["script config"]:
        SCRIPT_CONFIG_FILE = Path(candidates["script config"][0])
    # Outputs of other notebooks, pinned by version in the build; their mounts sit wherever Kaggle
    # puts them, so each is found as the one folder named by its slug (hyphens or underscores).
    inputs = BUILD.get("notebook_inputs") or {}
    roots = [str(INPUT), str(NOTEBOOK_LIBS)]
    for name, paths in find_input_dirs(roots, list(inputs), SEARCH_DEPTH).items():
        print(f"found notebook output {name} (version {inputs[name]}): {paths}")
        if len(paths) == 1:
            INPUT_DIRS[name] = Path(paths[0])
        else:
            missing.append(
                f"{len(paths)} distinct folders of the notebook output {name} under {INPUT} and "
                f"{NOTEBOOK_LIBS}, expected 1: attach version {inputs[name]}"
            )
    checks = ((SCRIPT_FILE, "script_sha256"), (SCRIPT_CONFIG_FILE, "script_config_sha256"))
    for path, key in checks:
        if path is not None:
            digest = file_sha256(str(path))
            print(f"{path}: sha256 {digest}")
            if digest != BUILD[key]:
                missing.append(
                    f"{path.name} has sha256 {digest}, the build baked {BUILD[key]}: attach the "
                    "code dataset version made from this build"
                )
    if PYTHON is None:
        missing.append("no python on PATH")
    if bf16 is not None:
        missing.append(bf16)
    print(f"data: {CHALLENGES}")
    print(f"model: {MODEL_DIR}")
    print(f"wheel: {WHEEL}")
    print(f"script: {SCRIPT_FILE}")
    print(f"script config: {SCRIPT_CONFIG_FILE}")
    for name, folder in INPUT_DIRS.items():
        print(f"notebook output {name}: {folder}")
    if missing:
        problem("environment check: " + "; ".join(missing))"""

SCRIPT_INSTALL_NOTE = """\
## 4. Install

As in the submission notebook: uninstalls tensorflow, so that transformers in the script does \
not load it; checks the wheel's sha256 against the build's; installs it with the `python` on \
PATH (`--no-index`, `--no-deps`, `--force-reinstall`); and checks that the kernel imports the \
copy that `python` installed. Skipped after a problem."""

# The submission notebook's Install cell with one guard in front: in a script notebook a problem
# leaves its stage without stopping the notebook, so the install must not start after one.
_INSTALL_FIRST_CHECK = "    if PYTHON is None or WHEEL is None:\n"
if INSTALL_CODE.count(_INSTALL_FIRST_CHECK) != 1:
    raise RuntimeError("INSTALL_CODE changed: update SCRIPT_INSTALL_CODE's guard")
SCRIPT_INSTALL_CODE = INSTALL_CODE.replace(
    _INSTALL_FIRST_CHECK,
    "    if PROBLEMS:\n"
    '        STAGES["install"]["skipped"] = "an earlier stage stopped the run"\n'
    "    elif PYTHON is None or WHEEL is None:\n",
)

SCRIPT_RUN_NOTE = """\
## 5. Script

Runs `python <script> --config <config> <ARGS>` with the `python` on PATH, in \
`/kaggle/working`, in a process group of its own, with `PYTHONUNBUFFERED=1`, so its lines reach \
the session log as they are printed. At the hard stop, `SCRIPT_KILL_S` seconds after the start, \
the script is killed with every process it started, which leaves time to write the logs before \
the session's Timeout. The script writes its own outputs into `/kaggle/working/run_logs/` (BRIEF \
A8) and keeps to its own budget; its exit code, and whether the hard stop killed it, are \
recorded, not raised. In `ARGS`, `{commit}` becomes the built commit's SHA and \
`{input:<owner>/<slug>}` the mount folder the environment check found for that notebook output; \
then `ARGS` that would set `--config` (`--config`, an abbreviation such as `--conf`, or `-c`) \
are refused, since the checked config must be the one the script reads. The \
notebook's file names in `run_logs/` (`candidates.json`, `env.json`, `pip_freeze.txt`, \
`stages.json`) are its own: a script that writes one is reported as a problem. Skipped after a \
problem."""

SCRIPT_RUN_CODE = """\
SCRIPT_EXIT = None
with stage("script") as record:
    if PROBLEMS:
        record["skipped"] = "an earlier stage stopped the run"
        print("script skipped: " + "; ".join(PROBLEMS))
    elif not INSTALLED:
        problem("script skipped: arc26 is not installed (see the install stage)")
    else:
        # "{commit}" in an argument stands for the built commit's full SHA: the script records
        # the code it ran, and its copy in the code dataset has no git checkout to read it from.
        # "{input:<owner>/<slug>}" stands for that attached notebook output's mount folder.
        try:
            args = fill_args(ARGS, BUILD["commit"], {k: str(v) for k, v in INPUT_DIRS.items()})
        except ValueError as err:
            problem(f"ARGS: {err}")
        # The builder checked the config's args; ARGS may have been edited since.
        override = config_override(args)
        if override is not None:
            problem(
                f"ARGS must not set --config ({override!r}): the script reads the config whose "
                "sha256 the environment check compared with BUILD"
            )
        command = [PYTHON, str(SCRIPT_FILE), "--config", str(SCRIPT_CONFIG_FILE), *args]
        # Lines reach the session log when printed and survive a kill; nothing computed changes.
        launch_env = {"PYTHONUNBUFFERED": "1"}
        record.update(command=command, launch_env=launch_env, script_kill_s=SCRIPT_KILL_S)
        timeout = NOTEBOOK_START + SCRIPT_KILL_S - time.time()
        if timeout <= 0:
            problem(f"the hard stop, {SCRIPT_KILL_S} s after the start, has passed")
        RUN_LOGS.mkdir(parents=True, exist_ok=True)
        print(f"script: hard stop in {timeout:.0f} s", flush=True)
        SCRIPT_EXIT, killed = run_streamed(
            command, timeout, str(WORKING), {**os.environ, **launch_env}
        )
        record.update(script_exit=SCRIPT_EXIT, killed=killed)
        if killed:
            print(f"the script was killed at the hard stop, {SCRIPT_KILL_S} s after the start")
        clashes = [name for name in NOTEBOOK_LOGS if (RUN_LOGS / name).exists()]
        if clashes:
            problem(
                f"the script wrote {', '.join(clashes)} into {RUN_LOGS}, where the notebook "
                "writes its own logs, which replace them"
            )"""

SCRIPT_LOGS_NOTE = """\
## 6. Run logs

Writes the notebook's files beside the script's in `/kaggle/working/run_logs/`, which \
`scripts/kaggle_ops.py pull` downloads: `env.json` and `pip_freeze.txt` (and `candidates.json`, \
`{}` unless the script left candidates in `/kaggle/inference_outputs/`) from `dump_run_logs`, \
copied verbatim from `scripts/build_reference_run.py`, then `stages.json`: each stage's start, \
wall time, and error, the problems, the script's command, exit code, and whether the hard stop \
killed it, and `ok`. It runs after any problem, and the notebook then ends without an error: \
whether Kaggle keeps the outputs of a failed session is unverified, so `ok` in `stages.json` \
says whether the run passed."""

SCRIPT_LOGS_TAIL = """\
with stage("run_logs"):
    # dump_run_logs reads candidates from the solver's folder, as in a submission build: empty
    # unless the script left candidates there, so candidates.json is {} for most scripts.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dump_run_logs(str(OUTPUTS_DIR), str(RUN_LOGS), MAX_LOG_FILE_BYTES, BUILD)

SUMMARY = {
    "mode": "script",
    "build": BUILD,
    "settings": {"args": ARGS, "script_kill_s": SCRIPT_KILL_S},
    "stages": STAGES,
    "problems": PROBLEMS,
    "script_exit": SCRIPT_EXIT,
    "ok": not PROBLEMS and SCRIPT_EXIT == 0,
    "total_s": round(time.time() - NOTEBOOK_START, 1),
}
save_json(RUN_LOGS / "stages.json", SUMMARY)
print(
    f"done after {SUMMARY['total_s']} s: script exit {SCRIPT_EXIT}, problems {len(PROBLEMS)}; "
    + ("ok" if SUMMARY["ok"] else "FAILED, see run_logs/stages.json")
)"""


def submission_blocker(config: SubmissionConfig) -> str | None:
    """Return why a build is not for submission, or None when it may be submitted.

    Why: Kaggle reruns the submitted version on the hidden test set. There the notebook always
    runs the solver on every test task with ``COMPETITION_SOLVER_END_S`` and
    ``COMPETITION_SOLVER_KILL_S``, records no timing, and ignores a series, so the development
    subset, ``RUN_SOLVER``, and the development deadlines cannot cut a rerun short. Submissions
    come only from ``SUBMISSION_NOTEBOOK``, so a build for any other notebook is a development
    build, however production-like its settings. What the rerun takes from the build must also be
    production: valid competition deadlines, an accelerator (on CPU the solver cannot start),
    ``PRODUCTION_ACCELERATOR`` with ``PRODUCTION_WORKERS`` workers, one per L4 as the reference
    ran (one worker would reach about 66 tasks by the deadline at P1.1-01's 622 s per puzzle),
    and no series: a series build would rerun the reference settings, but it was built to
    compare arms; nor adapter arms (task P3.7), for the same reason. A ``scope`` (task P2.6,
    ADR-018) is no reason: the rerun trains it as a development run of the same build does. Nor
    is an ``agent`` (task P6.5): the rerun runs the agent stage as a development run of the build
    does, and the final check keeps NVARC's attempts wherever the agent leaves none; but an agent
    stage that ends before ``COMPETITION_SOLVER_END_S`` is a development scale (run P6.5-01's
    sixth of the session) and is a reason, since a rerun of it would leave most of the 12 h
    unused.

    Args:
        config: The checked config.

    Returns:
        Every reason, the notebook's first, joined by ``"; "``, or None when the build is for
        ``SUBMISSION_NOTEBOOK`` and has the production settings.
    """
    reasons = []
    if config.kaggle_notebook != SUBMISSION_NOTEBOOK:
        reasons.append(
            f"its notebook is {config.kaggle_notebook}, not {SUBMISSION_NOTEBOOK}, the only "
            "notebook submissions come from"
        )
    latest = SESSION_LIMIT_S - WRITE_RESERVE_S
    if not MIN_SOLVER_END_S <= COMPETITION_SOLVER_END_S <= COMPETITION_SOLVER_KILL_S <= latest:
        reasons.append(
            f"the competition deadlines ({COMPETITION_SOLVER_END_S} s, "
            f"{COMPETITION_SOLVER_KILL_S} s) are outside {MIN_SOLVER_END_S} to {latest} s"
        )
    if config.accelerator is None:
        reasons.append(
            "it has no accelerator, so a competition rerun would run on CPU, where the solver "
            "cannot start"
        )
    elif config.accelerator != PRODUCTION_ACCELERATOR:
        reasons.append(
            f"its accelerator is {config.accelerator}, not {PRODUCTION_ACCELERATOR} (4xL4), "
            "the scored rerun's"
        )
    if config.workers != PRODUCTION_WORKERS:
        reasons.append(
            f"it runs {config.workers} worker{'' if config.workers == 1 else 's'}, not "
            f"{PRODUCTION_WORKERS} (one per L4), so a competition rerun would reach only part of "
            "the test tasks by the deadline"
        )
    if config.arms is not None:
        count = len(config.arms)
        built = "LoRA scopes"
        if any(arm.holdout is not None for arm in config.arms):
            built = "LoRA scopes and the held-out gate"
        if any(arm.rounds is not None for arm in config.arms):
            built = "LoRA scopes, the held-out gate, and the refinement rounds"
        reasons.append(
            f"it is a series of {count} arm{'' if count == 1 else 's'}, built to compare {built} "
            "(a competition rerun would ignore the series and run the reference settings once)"
        )
    if config.adapters is not None:
        count = len(config.adapters.arms)
        reasons.append(
            f"it runs {count} adapter arm{'' if count == 1 else 's'} (task P3.7), built to "
            "compare adapter sources (a competition rerun would ignore them and run SCOPE once)"
        )
    # A shot build's production end is earlier by the pre-script allowance (task P6.5b).
    production = COMPETITION_SOLVER_END_S
    if config.agent is not None and config.agent.shot:
        production -= PRE_SCRIPT_ALLOWANCE_S
    if config.agent is not None and config.agent.end_s < production:
        reasons.append(
            f"its agent stage ends {config.agent.end_s} s after the start, a development scale: "
            "a competition rerun, which runs AGENT's times too, would end the agent hours before "
            f"12 h (a production agent ends at {production} s or later)"
        )
    return "; ".join(reasons) or None


def config_note(config: SubmissionConfig, build: Mapping[str, str]) -> str:
    """Return the first markdown cell: what the notebook is, and how to edit its Config cell.

    Args:
        config: The checked config.
        build: The provenance from ``build_info``.

    Returns:
        Markdown text; a build that is not for submission says so first, with every reason.
    """
    blocker = submission_blocker(config)
    warning = (
        ""
        if blocker is None
        else f"**Not for submission.** A development build: {blocker}.\n\n"
    )
    split = "" if config.solver == DEFAULT_SOLVER else OTHER_SOLVER_NOTES[config.solver]
    # What a competition rerun trains, and the Config cell's settings for both modes.
    scope, rerun, names = "", "the reference's LoRA settings", ["`SOLVER`"]
    if config.scope is not None:
        scope = (
            "Scoped build (task P2.6): `SCOPE` sets the LoRA scope of every solver launch, "
            f"{scope_text(config.scope)}, instead of the reference's r = {REFERENCE_LORA_R} on "
            "all nine target modules, in both modes: a competition rerun trains what a "
            f"development run trains (`docs/decisions/{SCOPE_ADR}`).\n\n"
        )
        rerun = "`SCOPE`"
        names.append("`SCOPE`")
    # A hybrid build's paragraph (task P6.5), and its NVARC deadlines in the Config paragraph.
    hybrid, deadlines = "", "the fixed deadlines of the Helpers cell"
    if config.agent is not None:
        agent = config.agent
        if agent.nvarc_share_s == 0:
            order = "the agent solver"
            keeps = (
                "NVARC does not run first, so the Write cell's `submission.json` holds `[[0]]` "
                "placeholders until the agent rewrites it atomically after each task. When the "
                "agent stage ends with that time left (the install or the boot failed, the agent "
                "crashed or was killed, its engine was gone for good), the engine is stopped and "
                "NVARC's stage runs with the deadlines it would have had; P6.2's selection of the "
                "agent's attempt log, with NVARC's new attempts as its NVARC input, then keeps "
                "every verified program in attempt 1 and lets NVARC fill the rest. An agent that "
                "runs until its generation stop leaves NVARC no time, so its file has no NVARC "
                "attempt"
            )
        else:
            order = "after NVARC, the agent solver"
            keeps = (
                "NVARC's `submission.json` is written first; the agent rewrites it atomically "
                "with NVARC's attempts as the fill, and a final check keeps NVARC's attempt for "
                "every slot the agent left invalid or a `[[0]]` placeholder, so no failure of the "
                "agent stage (install, boot, crash, timeout) costs NVARC's attempts"
            )
        hybrid = (
            f"Hybrid build (task P6.5): {order} (`docs/decisions/{AGENT_ADR}`) runs in both "
            f"modes, a competition rerun included: {agent_text(agent)}. {keeps}. The engine and "
            f"the weights come from the attached outputs `{agent.wheels.notebook}` (version "
            f"{agent.wheels.version}) and `{agent.model.notebook}` (version "
            f"{agent.model.version}); when they fail the agent check, a competition rerun gives "
            "NVARC its full deadlines and skips the agent. `AGENT` in the next cell holds the "
            "switch and the times.\n\n"
        )
        if agent.shot:
            # A shot build (task P6.5b): NVARC keeps its deadlines; the shot takes what is left.
            hybrid = (
                f"{SHOT_TITLE} (task P6.5b, `docs/decisions/{AGENT_ADR}`), in both modes, a "
                f"competition rerun included: {agent_text(agent)}. NVARC's `submission.json` is "
                "written first, with its deadlines unchanged (NVARC's next task is worth more "
                "than the shot's); then the long shot (`P6.3-05`'s direct answers at xhigh) "
                "takes the tasks NVARC is least sure of in the time NVARC left, puts its first "
                "answer in the second slot where it differs from NVARC's first attempt (its "
                "second answer where the first equals it and NVARC's second slot is empty), and "
                "rewrites the file atomically after every task; a final check keeps NVARC's "
                "attempt for every slot the shot left invalid, so no failure of the shot (install, "
                "boot, engine, deadline) costs NVARC's attempts. The engine and the weights come "
                f"from the attached outputs `{agent.wheels.notebook}` (version "
                f"{agent.wheels.version}) and `{agent.model.notebook}` (version "
                f"{agent.model.version}); when they fail the agent check, a competition rerun "
                "runs NVARC alone. `AGENT` in the next cell holds the switch and the times.\n\n"
            )
            if agent.skip_nvarc:
                # LLM first (task P5.1): the shot on every task, no NVARC.
                hybrid = (
                    f"{SKIP_NVARC_TITLE} (task P5.1, `docs/decisions/{AGENT_ADR}`), in both "
                    f"modes, a competition rerun included: {agent_text(agent)}. The Write cell's "
                    "`submission.json` holds `[[0]]` placeholders; the long shot (`P6.3-05`'s "
                    "direct answers at xhigh) takes every task, the smallest opening first, puts "
                    "its two answers in the two slots of each output it answers, and rewrites the "
                    "file atomically after every task; a task it does not reach keeps its "
                    "placeholders. The engine and the weights come from the attached outputs "
                    f"`{agent.wheels.notebook}` (version {agent.wheels.version}) and "
                    f"`{agent.model.notebook}` (version {agent.model.version}); when they fail "
                    "the agent check, a competition rerun runs NVARC alone. `AGENT` in the next "
                    "cell holds the switch and the times.\n\n"
                )
            if agent.tail is not None:
                # The rank-256 tail (task P4.7b).
                hybrid += (
                    "Tail build (task P4.7b): the shot queues no task at NVARC's confidence 0; "
                    "after the shot stage, in the time left before `AGENT['end_s']`, NVARC's "
                    "solver runs again with the reference's LoRA settings (rank 256) on those "
                    "tasks, cheapest first by NVARC's puzzle seconds, and its attempts fill only "
                    "the outputs that still hold a placeholder, in both modes "
                    f"(`BUILD['agent']['tail']`: {tail_text(agent.tail)}).\n\n"
                )
        else:
            deadlines += " (NVARC's cut at its share when the agent stage follows)"
        names.append("`AGENT`")
    if len(names) == 1:
        both = f"{names[0]} applies in both modes."
    elif len(names) == 2:
        both = f"{names[0]} and {names[1]} apply in both modes."
    else:
        both = f"{', '.join(names[:-1])}, and {names[-1]} apply in both modes."
    dev240 = ""
    if config.dev240:
        dev240 = (
            "DEV240 (task P6.5b), development runs only: the environment check writes every "
            "evaluation task twice, the second copy under the first 8 hex digits of the sha256 "
            f"of its id, to `{DEV240_DIR}/` under the root, with the original's solutions; "
            "NVARC, its score, the agent stage, and the final scores read those 240 tasks, each "
            "copy scored apart. A competition rerun solves the test tasks as always.\n\n"
        )
    series = ""
    if config.arms is not None:
        names = ", ".join(f"`{arm.name}`" for arm in config.arms)
        first = config.arms[0]
        source = (
            "the notebook configuration"
            if (first.lora_r, first.lora_targets) == (REFERENCE_LORA_R, REFERENCE_LORA_TARGETS)
            else f"`{first.name}`, which holds a pair out (task P2.5)"
        )
        series = (
            f"Series build: a development run launches the solver once per arm of `ARMS`, in "
            f"this order: {names}. Each arm trains its own LoRA scope and writes its own "
            f"`run_logs/<name>/`; `submission.json` comes from the first, {source} "
            "(`docs/runbooks/build_submission.md`, series mode).\n\n"
        )
        for arm in config.arms:
            if arm.holdout is not None:
                series += (
                    f"Arm `{arm.name}` runs the held-out gate (task P2.1, `docs/modules/loop.md`, "
                    f"ADR-013). Its {holdout_text(arm.holdout._asdict())}.\n\n"
                )
            if arm.rounds is not None:
                tasks = (
                    "task P2.3, ADR-015"
                    if arm.rounds.variants is None
                    else "tasks P2.3 and P2.5, ADR-015 and ADR-016"
                )
                series += (
                    f"Arm `{arm.name}` runs the refinement rounds ({tasks}), its "
                    f"{rounds_text(arm.rounds._asdict())}; the governor's config is "
                    f"`{arm.rounds.governor}` of this commit, embedded in the Config cell.\n\n"
                )
    adapters = ""
    if config.adapters is not None:
        arms = ", ".join(f"`{arm.name}` ({arm.source})" for arm in config.adapters.arms)
        where = "the subset" if config.subset is not None else "every task"
        if config.adapters.gen is not None:
            gen = config.adapters.gen
            where = f"GEN ({gen['barc']} BARC and {gen['rearc']} ReARC held-out generated puzzles)"
        views = config.adapters.views
        count = "the reference's 16" if views is None else str(sum(map(len, views)))
        inputs = ", ".join(f"`{name}` version {v}" for name, v in config.notebook_inputs.items())
        if config.dataset_inputs:
            datasets = ", ".join(
                f"`{name}` version {v}" for name, v in config.dataset_inputs.items()
            )
            inputs += f"; and datasets, attached at pinned versions: {datasets}"
        adapters = (
            f"Adapter arms build (task P3.7, `docs/decisions/{ADAPTERS_ADR}`): a development run "
            f"runs every puzzle of {where} under each arm of `ADAPTERS`, in this order: {arms}, "
            f"paired by puzzle, {count} decode views per test input "
            "(`python -m arc26.arms`), after H's load check. Each arm's candidates go to "
            "`/kaggle/working/arm_outputs/<arm>/`, its logs, submission, and scores to "
            "`run_logs/arms/<arm>/`; `submission.json` holds placeholders. It also reads the "
            f"outputs of other notebooks, attached at pinned versions: {inputs}. A competition "
            "rerun ignores `ADAPTERS` and runs `SCOPE` once.\n\n"
        )
    return (
        "# arc26 submission notebook\n"
        "\n"
        f"{warning}"
        f"Build `{build['tag']}` of commit `{build['commit'][:12]}`, generated by "
        f"`scripts/build_kaggle_notebook.py` from `{build['config']}`. It runs offline: the code "
        f"comes from the private dataset `{config.code_dataset}` (one `arc26` wheel), the base "
        "model and the competition data from their attached sources, and every other package "
        "from the pinned image and the attached install notebook "
        "(`docs/decisions/ADR-010-packaging.md`). Rebuild with `scripts/build_submission.py` "
        "instead of editing any cell but the next one.\n"
        "\n"
        f"{split}"
        f"{scope}"
        f"{hybrid}"
        f"{dev240}"
        f"{series}"
        f"{adapters}"
        "## 1. Config\n"
        "\n"
        "The only cell edited by hand. It takes the start time first, as the reference's first "
        "cell does, because every deadline counts from it. `SUBSET`, `RUN_SOLVER`, `ARMS`, and "
        "the two deadlines apply to development runs; `RUN_SOLVER = False` checks the "
        "environment, installs, and writes a valid submission without solving. A competition "
        f"rerun always solves every test task once, with {rerun} and {deadlines}. {both} "
        "`BUILD` records what was built and is not for editing."
    )


# The Config cell's lines on an arm's holdout (task P2.1), only in a build that has one, so the
# Config cell of every other build stays as it was.
HOLDOUT_COMMENT = (
    "# An arm's holdout (task P2.1, docs/modules/loop.md): one real pair per puzzle is held out",
    "# of training and prompts by its rule ('last' or 'first'), at 2 real pairs too when k2 is",
    "# True, and stays in the test inputs' prompts only when in_test_prompts is True; then the",
    "# held-out gate checks it. It needs SOLVER = 'split'.",
)
# The Config cell's lines on an arm's rounds (task P2.3), only in a build that has one.
ROUNDS_COMMENT = (
    "# An arm's rounds (task P2.3, ADR-015): after each puzzle's gate, at most max_rounds (1 or",
    "# 2) refinement rounds, priced by the governor (its config is configs/governor.yaml as",
    "# built): promote at agreement tau, continue the adapter continue_steps steps, roll back",
    "# when the gate fails; final_in_prompts puts the held-out pair back in the final decode.",
)
# The Config cell's lines on the series' first arm, which gives submission.json: the notebook
# configuration, or (task P2.5) a hold-out arm of another scope.
SERIES_FIRST_COMMENT = (
    "# within the deadlines above; the first arm is the notebook configuration and gives",
    "# submission.json. A competition rerun ignores ARMS and runs the reference settings",
    "# once. None: one run with the reference settings.",
)
SERIES_HOLDOUT_FIRST_COMMENT = (
    "# within the deadlines above; the first arm, a hold-out arm of another scope (task",
    "# P2.5), gives submission.json. A competition rerun ignores ARMS and runs the reference",
    "# settings once. None: one run with the reference settings.",
)
# The Config cell's lines on the build's scope (task P2.6), only in a build that has one, and
# its lines on ARMS there: a series would replace SCOPE in development runs only.
SCOPE_COMMENT = (
    "# The LoRA scope of every launch (task P2.6, ADR-018), in both modes, a competition rerun",
    "# included: rank lora_r on the modules lora_targets, rsLoRA with alpha 32; embed_tokens and",
    "# lm_head train, as full copies, only when listed. None: the reference's r = 256 on all nine.",
)
SERIES_SCOPE_COMMENT = (
    "# within the deadlines above; the first arm is the notebook configuration and gives",
    "# submission.json. A competition rerun ignores ARMS and runs SCOPE once. None: one run",
    "# with SCOPE.",
)
# The Config cell's lines on the adapter arms (task P3.7), only in a build that has them.
ADAPTERS_COMMENT = (
    "# Adapter arms (task P3.7, ADR-019), development runs only: python -m arc26.arms runs every",
    "# puzzle under each arm, in this order, before the next puzzle (paired), after H's load",
    "# check; a competition rerun ignores ADAPTERS and runs SCOPE once. {input:<owner>/<slug>}",
    "# stands for the folder of that notebook output, attached at the version BUILD pins.",
)
# The Config cell's lines on a hybrid build's agent stage (task P6.5), only in a build that has
# one.
AGENT_COMMENT = (
    "# The agent stage (task P6.5, ADR-020), in both modes, a competition rerun included. With",
    "# 'enabled' True and the agent check passed, NVARC's soft deadline is 'nvarc_share_s' (0:",
    "# NVARC does not run) and its hard stop 600 s later; then the engine is installed offline (at",
    "# most 'install_timeout_s') and arc26.agent.run solves until 'end_s', its generation stopping",
    "# the run config's final_margin_s earlier; SIGTERM at 'end_s', SIGKILL 'term_s' later. The",
    "# stage starts only with 'min_stage_s' left before the generation stop. Seconds after",
    "# NOTEBOOK_START. 'enabled' False: NVARC alone, with its full deadlines, and no agent stage.",
)
# The Config cell's lines on an arm's variants (task P2.5), only in a build whose arm names them.
VARIANTS_COMMENT = (
    "# variants (task P2.5, ADR-016): 'gradient' (the above), 'in_context' (no fit: the promoted",
    "# pairs join the prompts, only the pairs roll back), or both in this order (the governor's",
    "# ladder: in context when the gradient round does not fit the time left).",
)


def config_cell_source(config: SubmissionConfig, build: Mapping[str, str]) -> str:
    """Return the Config cell: the settings a human may edit, then the baked provenance.

    Args:
        config: The checked config.
        build: The provenance from ``build_info``.

    Returns:
        Python source.
    """
    literal = pprint.pformat(dict(build), sort_dicts=False, width=96)
    arms = None
    if config.arms is not None:
        # An arm without a held-out pair keeps the four keys of task P1.3, so its cell is
        # unchanged; a hold-out arm adds its settings (task P2.1).
        arms = []
        for arm in config.arms:
            entry = {**arm._asdict(), "lora_targets": list(arm.lora_targets)}
            if arm.holdout is None:
                del entry["holdout"]
            else:
                entry["holdout"] = arm.holdout._asdict()
            if arm.rounds is None:
                del entry["rounds"]
            else:
                entry["rounds"] = arm.rounds.settings()
            arms.append(entry)
    series = SERIES_HOLDOUT_FIRST_COMMENT
    if config.scope is not None:
        series = SERIES_SCOPE_COMMENT
    elif not arms or (arms[0]["lora_r"], tuple(arms[0]["lora_targets"])) == (
        REFERENCE_LORA_R, REFERENCE_LORA_TARGETS
    ):  # fmt: skip
        series = SERIES_FIRST_COMMENT
    # A scoped build's SCOPE (task P2.6), only there, so every other Config cell stays as it was.
    scope = ()
    if config.scope is not None:
        scope = (
            *SCOPE_COMMENT,
            f"SCOPE = {pprint.pformat(config.scope.settings(), sort_dicts=False, width=96)}",
        )
    # An adapter build's ADAPTERS (task P3.7), only there.
    if config.adapters is not None:
        literal_adapters = pprint.pformat(config.adapters.settings(), sort_dicts=False, width=96)
        scope = (*scope, *ADAPTERS_COMMENT, f"ADAPTERS = {literal_adapters}")
    # DEV240 (task P6.5b), only in a build that sets it, for the same reason.
    dev240 = ()
    if config.dev240:
        dev240 = (
            "# DEV240 (task P6.5b), development runs only: every evaluation task twice, the second",
            "# copy under the first 8 hex digits of the sha256 of its id, scored against the",
            "# original's solutions (SUBSET stays None). False: the evaluation tasks as they are.",
            "DEV240 = True",
        )
    # A hybrid build's AGENT (task P6.5), only there, for the same reason.
    agent = ()
    if config.agent is not None:
        stage_settings = config.agent.stage_settings()
        comment = SHOT_AGENT_COMMENT if config.agent.shot else AGENT_COMMENT
        if config.agent.skip_nvarc:
            comment = SKIP_NVARC_AGENT_COMMENT
        agent = (
            *comment,
            f"AGENT = {pprint.pformat(stage_settings, sort_dicts=False, width=96)}",
        )
    return "\n".join(
        [
            "# The only cell edited by hand. scripts/build_kaggle_notebook.py generated every cell",
            f"# from {build['config']}; rebuild with scripts/build_submission.py for the others.",
            "import time",
            "",
            "# Taken first, as the reference's first cell takes its time budget: every deadline",
            "# below counts from here.",
            "NOTEBOOK_START = time.time()",
            "",
            "# Task ids solved in a development run; None solves every task. A competition rerun",
            "# always solves every test task, as in the reference.",
            f"SUBSET = {config.subset!r}",
            *dev240,
            "# Solver processes, one per GPU.",
            f"WORKERS = {config.workers!r}",
            "# PYTHONHASHSEED of the solver, which seeds the reference's rescoring augmentations.",
            f"HASH_SEED = {config.hash_seed!r}",
            "# The solver's code (task P2.0): 'nvarc' runs the moved reference (arc26.nvarc),",
            "# 'split' the same statements split into arc26.ttt and arc26.decode (arc26.solver).",
            "# Both modes use it, a competition rerun included.",
            f"SOLVER = {config.solver!r}",
            *scope,
            "# False skips the solver in a development run; the notebook still checks, installs,",
            "# and writes a valid submission. A competition rerun always runs the solver.",
            f"RUN_SOLVER = {config.run_solver!r}",
            "# Seconds after NOTEBOOK_START, development runs only (a competition rerun uses the",
            "# Helpers cell's COMPETITION_SOLVER_*): the solver's --end-time, its soft deadline.",
            f"SOLVER_END_S = {config.solver_end_s!r}",
            "# Seconds after NOTEBOOK_START, development runs only: the hard stop, when the solver",
            "# is killed so the rest of the session can write submission.json.",
            f"SOLVER_KILL_S = {config.solver_kill_s!r}",
            "# Series mode, development runs only (task P1.3): one solver launch per arm, in this",
            "# order, each with its LoRA rank and target modules and its own run_logs/<name>/,",
            *series,
            *(HOLDOUT_COMMENT if any(arm.get("holdout") for arm in arms or ()) else ()),
            *(ROUNDS_COMMENT if any(arm.get("rounds") for arm in arms or ()) else ()),
            *(
                VARIANTS_COMMENT
                if any("variants" in (arm.get("rounds") or {}) for arm in arms or ())
                else ()
            ),
            f"ARMS = {pprint.pformat(arms, sort_dicts=False, width=96)}",
            *agent,
            "",
            "# Build provenance, written by the builder; not for editing.",
            f"BUILD = {literal}",
        ]
    )


def agent_constants_source(entry: str = DEFAULT_AGENT_ENTRY) -> str:
    """Return the agent check cell's constants, with their reasons (task P6.5).

    Args:
        entry: The module the agent stage runs; a shot build's (task P6.5b) adds
            ``AGENT_ENTRY``, the default adds nothing, so earlier builds keep their bytes.

    Returns:
        Python source assigning the hybrid stage's constants from this module's, and
        ``AGENT_SCRATCH``.
    """
    shot = ""
    if entry != DEFAULT_AGENT_ENTRY:
        shot = (
            "\n# The module the agent stage runs (task P6.5b), BUILD['agent']['entry'] as built:\n"
            "# it takes arc26.agent.run's arguments and exit codes.\n"
            f'AGENT_ENTRY = "{entry}"\n'
            "# The session's usable limit in notebook time for the shot's plan: 12 h less the\n"
            "# time Kaggle may count before this notebook's first cell (45 min, above every\n"
            "# measured case).\n"
            f"SHOT_SESSION_S = {SHOT_SESSION_S}"
        )
    return (
        "# NVARC's hard stop comes this long after its share: the competition deadlines' gap.\n"
        f"NVARC_KILL_GAP_S = {NVARC_KILL_GAP_S}\n"
        "# Kaggle's 12 h limit, and the time kept for a write (NVARC's selection; the final check\n"
        "# and the logs).\n"
        f"SESSION_LIMIT_S = {SESSION_LIMIT_S}\n"
        f"WRITE_RESERVE_S = {WRITE_RESERVE_S}\n"
        "# Seconds a process group gets to die after SIGKILL (the agent's, then its engine's).\n"
        f"AGENT_KILL_WAIT_S = {AGENT_KILL_WAIT_S}\n"
        "# Seconds between SIGTERM and SIGKILL of the installer at its timeout.\n"
        f"INSTALL_TERM_S = {INSTALL_TERM_S}\n"
        "# Seconds to wait for the GPUs to be free before the engine boots, and before NVARC's\n"
        "# fallback runs: no process listed, and at most GPU_USED_MAX_MIB MiB in use on each.\n"
        f"GPU_FREE_WAIT_S = {GPU_FREE_WAIT_S}\n"
        f"GPU_USED_MAX_MIB = {GPU_USED_MAX_MIB}\n"
        "# An agent-only build's fallback runs NVARC only with this much left before NVARC's soft\n"
        "# deadline: its model loads in about 5 minutes and a puzzle takes about 8 on 4 workers.\n"
        f"NVARC_FALLBACK_MIN_S = {NVARC_FALLBACK_MIN_S}\n"
        "# Candidate files larger than this are skipped, as dump_run_logs skips them, and agent\n"
        "# logs are cut to their last this many bytes for the pull.\n"
        f"AGENT_LOG_MAX_BYTES = {reference_run.MAX_LOG_FILE_BYTES}\n"
        "# The agent's scratch: the venv, the engine's caches, NVARC's copies, the logs; never\n"
        "# /kaggle/working. ARC26_AGENT_SCRATCH points it elsewhere, as the local test does. Read\n"
        "# without a subscript: nothing outside a stage may raise, or a rerun would stop here.\n"
        "AGENT_SCRATCH = Path(\n"
        '    os.environ.get("ARC26_AGENT_SCRATCH")\n'
        '    or (BUILD.get("agent") or {}).get("scratch_dir")\n'
        f'    or "{AGENT_DEFAULT_SCRATCH}"\n'
        ")"
        f"{shot}"
    )


def agent_check_cell_source(config: SubmissionConfig) -> str:
    """Return a hybrid build's agent check cell (task P6.5).

    Args:
        config: The checked config, its agent resolved (``resolve_agent``).

    Returns:
        Python source: the constants, the agent's run config as built (``AGENT_CONFIG``), the
        agent stage's helpers (``inspect.getsource``), then the checks.

    Raises:
        BuildError: If the agent's run config has not been read.
    """
    agent = config.agent
    if agent is None or agent.config_data is None:
        raise BuildError("the agent config was not read (resolve_agent)")
    # A shot build (task P6.5b) adds its plan; every other build keeps its helpers.
    runtime = (*AGENT_RUNTIME_HELPERS, shot_plan) if agent.shot else AGENT_RUNTIME_HELPERS
    warmup = ""
    if agent.warmup is not None:
        # The weights' warm-up (task P6.9c): its helpers, and the copier as a script.
        runtime = (*runtime, *WARMUP_RUNTIME_HELPERS)
        warmup = f"{warmup_constants_source()}\n\n\n"
    if agent.tail is not None:
        # The rank-256 tail (task P4.7b): the shot's queue check and the tail's helpers.
        runtime = (*runtime, *TAIL_RUNTIME_HELPERS)
    helpers = "\n\n\n".join(inspect.getsource(helper).rstrip() for helper in runtime)
    literal = pprint.pformat(agent.config_data, sort_dicts=False, width=96)
    return (
        "# Agent check (task P6.5): the hybrid stage's constants, the agent's run config, and the\n"
        "# helpers of the agent stage, then the checks.\n"
        f"{agent_constants_source(agent.entry)}\n"
        f"# The agent's run config: {agent.config} of the built commit, its sha256 in\n"
        "# BUILD['agent']['config_sha256'], parsed. The agent stage writes it to scratch as JSON\n"
        "# for python -m arc26.agent.run --config (YAML reads JSON; the builder checked that this\n"
        "# copy reads back the same).\n"
        f"AGENT_CONFIG = {literal}\n\n\n"
        "# Runtime helpers of the agent stage, verbatim from scripts/build_kaggle_notebook.py.\n"
        f"{helpers}\n\n\n"
        f"{warmup}"
        f"{AGENT_CHECK_CODE}"
    )


def warmup_constants_source() -> str:
    """Return a warm-up build's constants for the agent check cell (task P6.9c).

    Returns:
        Python source assigning ``WARMUP_SCRIPT`` (``copy_weights``' source, verbatim, with a
        ``__main__`` line that runs it on the spec path it gets), ``WARMUP_POLL_S``, and the
        warm-up's state, empty until the agent check starts it.

    Raises:
        BuildError: If the copier's source holds a backslash or three single quotes, which the
            literal cannot carry verbatim.
    """
    source = inspect.getsource(copy_weights).rstrip()
    quotes = "'" * 3
    if "\\" in source or quotes in source:
        raise BuildError("copy_weights' source must hold no backslash and no three single quotes")
    script = (
        f"{source}\n\n\n"
        'if __name__ == "__main__":\n'
        "    import sys\n\n"
        "    sys.exit(copy_weights(sys.argv[1]))\n"
    )
    return (
        "# The weights' warm-up (task P6.9c): the copier, copy_weights verbatim from\n"
        "# scripts/build_kaggle_notebook.py, run as a script in a session of its own at the\n"
        "# lowest CPU and IO priority while NVARC solves.\n"
        f"WARMUP_SCRIPT = {quotes}\\\n{script}{quotes}\n"
        "# Seconds between the agent stage's reads of the warm-up's status while it waits.\n"
        f"WARMUP_POLL_S = {WARMUP_POLL_S}\n"
        "# The copier's process and start_warmup's record; None until the agent check starts it.\n"
        "AGENT_WARMUP = AGENT_WARMUP_PROCESS = AGENT_WEIGHTS_MANIFEST = None"
    )


def script_config_note(config: ScriptConfig, build: Mapping[str, str]) -> str:
    """Return a script notebook's first markdown cell: never submitted, what it runs, the Config.

    Args:
        config: The checked script config.
        build: The provenance from ``build_info``.

    Returns:
        Markdown text that starts with the warning that the notebook is never submitted.
    """
    command = " ".join(["python", config.script, "--config", config.script_config, *config.args])
    inputs = ""
    if config.notebook_inputs:
        attached = ", ".join(f"`{name}` version {v}" for name, v in config.notebook_inputs.items())
        inputs = (
            f" It also reads the outputs of other notebooks, attached at pinned versions: "
            f"{attached}; each is found by its folder name and passed where "
            "`{input:<owner>/<slug>}` stands in its arguments."
        )
    return (
        "# arc26 script notebook\n"
        "\n"
        "**Never submitted.** A script notebook runs one script for development (BRIEF A8) and "
        "writes no `submission.json`; it stops at once in a competition rerun. Submissions come "
        f"only from `{SUBMISSION_NOTEBOOK}`.\n"
        "\n"
        f"Build `{build['tag']}` of commit `{build['commit'][:12]}`, generated by "
        f"`scripts/build_kaggle_notebook.py` from `{build['config']}`. It runs `{command}` "
        "offline: the script and its config come byte for byte from that commit, beside the "
        f"`arc26` wheel in the private dataset `{config.code_dataset}`; the base model and the "
        "competition data come from their attached sources, and every other package from the "
        "pinned image and the attached install notebook "
        f"(`docs/decisions/ADR-010-packaging.md`).{inputs} Rebuild with "
        "`scripts/build_submission.py` instead of editing any cell but the next one.\n"
        "\n"
        "## 1. Config\n"
        "\n"
        "The only cell edited by hand. It takes the start time first, because the hard stop "
        "counts from it. `ARGS` follow `--config <config>` on the script's command line "
        "(`{commit}` in one stands for the built commit's SHA, `{input:<owner>/<slug>}` for an "
        "attached notebook output's folder), and `SCRIPT_KILL_S` is the hard "
        "stop. `BUILD` records what was built, the sha256 of the script and of its config "
        "included, and is not for editing."
    )


def script_config_cell_source(config: ScriptConfig, build: Mapping[str, str]) -> str:
    """Return a script notebook's Config cell: the editable settings, then the provenance.

    Args:
        config: The checked script config.
        build: The provenance from ``build_info``.

    Returns:
        Python source.
    """
    literal = pprint.pformat(dict(build), sort_dicts=False, width=96)
    return "\n".join(
        [
            "# The only cell edited by hand. scripts/build_kaggle_notebook.py generated every cell",
            f"# from {build['config']}; rebuild with scripts/build_submission.py for the others.",
            "import time",
            "",
            "# Taken first: the hard stop counts from here.",
            "NOTEBOOK_START = time.time()",
            "",
            '# Arguments after `--config <config>` on the script\'s command line; "{commit}" in',
            "# one stands for the full SHA of BUILD's commit, \"{input:<owner>/<slug>}\" for the",
            "# folder of that notebook output, attached at the version BUILD pins.",
            f"ARGS = {list(config.args)!r}",
            "# Seconds after NOTEBOOK_START: the hard stop, when the script is killed with every",
            "# process it started, so the rest of the session can write the logs.",
            f"SCRIPT_KILL_S = {config.script_kill_s!r}",
            "",
            "# Build provenance, written by the builder; not for editing. The script and its",
            "# config run only if their sha256 equal script_sha256 and script_config_sha256.",
            f"BUILD = {literal}",
        ]
    )


def competition_deadlines_source() -> str:
    """Return the lines that set the competition rerun's deadlines, with their reasons.

    Returns:
        Python source assigning ``COMPETITION_SOLVER_END_S`` and ``COMPETITION_SOLVER_KILL_S``
        from this module's constants.
    """
    return (
        "# Deadlines of a competition rerun, in seconds after NOTEBOOK_START, used whatever the\n"
        "# Config cell says, so any version can be submitted. Kaggle stops the rerun at 12 h and\n"
        "# reports up to 10 minutes of runtime variance. The workers' soft deadline is 12 h\n"
        "# minus 30 min (BRIEF A5's watchdog margin), so they normally stop before the hard stop\n"
        "# at 12 h minus 20 min, which still leaves 20 min to write submission.json.\n"
        f"COMPETITION_SOLVER_END_S = {COMPETITION_SOLVER_END_S}\n"
        f"COMPETITION_SOLVER_KILL_S = {COMPETITION_SOLVER_KILL_S}"
    )


def series_source() -> str:
    """Return the Helpers cell's series lines: the series in effect, the margin, arm folders.

    Returns:
        Python source assigning ``SERIES`` and ``SERIES_MARGIN`` (from this module's constant)
        and defining ``arm_dirs``, ``gate_dirs``, and ``round_dir``.
    """
    return (
        "# Series mode (task P1.3): the Config cell's arms in a development run. A competition\n"
        "# rerun ignores them and runs the reference's LoRA settings once, as a single run.\n"
        "SERIES = ARMS if ARMS and not COMPETITION else None\n"
        "# The next arm starts only while the time left before the soft deadline covers this\n"
        "# factor times the longest arm so far (series_refusal). A heuristic, not a bound: room\n"
        "# for run-to-run spread and a longer queue tail; an arm that still overruns is cut at\n"
        "# the soft deadline and does not count.\n"
        f"SERIES_MARGIN = {SERIES_MARGIN}\n"
        "\n"
        "\n"
        "def arm_dirs(arm):\n"
        '    """Candidate, run-log, and timing folders of a series arm (None: a single run)."""\n'
        "    if arm is None:\n"
        "        return OUTPUTS_DIR, RUN_LOGS, TIMING_DIR\n"
        '    name = arm["name"]\n'
        '    return OUTPUTS_DIR / name, RUN_LOGS / name, RUN_LOGS / name / "timing"\n'
        "\n"
        "\n"
        "def gate_dirs(arm):\n"
        '    """A hold-out arm\'s gate folders (task P2.1): the gate\'s candidate files, beside\n'
        "    inference_outputs and never in the arm's own candidate folder, and its\n"
        '    verdicts."""\n'
        '    name = arm["name"]\n'
        '    return ROOT / "gate_outputs" / name, RUN_LOGS / name / "gate"\n'
        "\n"
        "\n"
        "def round_dir(arm):\n"
        '    """A round arm\'s folder of the rounds\' candidate files (task P2.3), beside\n'
        '    inference_outputs and gate_outputs, one subfolder per puzzle."""\n'
        '    return ROOT / "round_outputs" / arm["name"]'
    )


def helpers_cell_source(adapters: bool = False) -> str:
    """Return the Helpers cell: mode, paths, deadlines, series, stage bookkeeping, and helpers.

    Args:
        adapters: Whether the build has adapter arms (task P3.7): then the cell adds their
            folders (``ADAPTERS_PATHS``) and ``ADAPTER_RUNTIME_HELPERS``; every other build's
            cell is as before.

    Returns:
        Python source; the helpers are embedded with ``inspect.getsource``.
    """
    chosen = RUNTIME_HELPERS + (ADAPTER_RUNTIME_HELPERS if adapters else ())
    helpers = "\n\n\n".join(inspect.getsource(helper).rstrip() for helper in chosen)
    return (
        f"{HELPERS_PATHS}\n"
        f"{competition_deadlines_source()}\n"
        f"{series_source()}\n\n\n"
        + (f"{ADAPTERS_PATHS}\n\n\n" if adapters else "")
        + f"{HELPERS_STAGES}\n\n\n"
        "# Runtime helpers, verbatim from scripts/build_kaggle_notebook.py.\n"
        f"{helpers}"
    )


def script_helpers_cell_source() -> str:
    """Return a script notebook's Helpers cell: paths, stage bookkeeping, and helpers.

    Returns:
        Python source; the helpers are embedded with ``inspect.getsource``.
    """
    helpers = "\n\n\n".join(
        inspect.getsource(helper).rstrip() for helper in SCRIPT_RUNTIME_HELPERS
    )
    return (
        f"{SCRIPT_HELPERS_PATHS}\n\n\n"
        f"{SCRIPT_HELPERS_STAGES}\n\n\n"
        "# Runtime helpers, verbatim from scripts/build_kaggle_notebook.py.\n"
        f"{helpers}"
    )


def validator_cell_source(build: Mapping[str, str], submit_source: str) -> str:
    """Return the Submission validator cell: ``arc26/submit.py`` as the wheel holds it.

    Args:
        build: The provenance from ``build_info``, which names the wheel.
        submit_source: ``arc26/submit.py`` read from that wheel (``wheel_member``).

    Returns:
        Python source.
    """
    return (
        f"# arc26/submit.py, verbatim from {build['wheel']}\n"
        f"# (sha256 {build['wheel_sha256']}).\n"
        f"{submit_source.rstrip()}"
    )


def logs_cell_source(tail: str = LOGS_TAIL) -> str:
    """Return a Run logs cell, which embeds ``dump_run_logs`` verbatim.

    Args:
        tail: The code after the function: ``LOGS_TAIL`` for the submission notebook,
            ``SCRIPT_LOGS_TAIL`` for a script notebook.

    Returns:
        Python source.
    """
    return (
        "# dump_run_logs, verbatim from scripts/build_reference_run.py, as run A of P0.1-01 ran "
        "it.\n"
        f"{inspect.getsource(reference_run.dump_run_logs).rstrip()}\n\n\n"
        "# Largest candidate pickle read and log file written, as in run A: pulls stay small.\n"
        f"MAX_LOG_FILE_BYTES = {reference_run.MAX_LOG_FILE_BYTES}\n\n"
        f"{tail}"
    )


def _cell(index: int, name: str, kind: str, text: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": kind, "id": f"arc26-{index:02d}-{name}", "metadata": {}}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cell["source"] = text.splitlines(keepends=True)
    return cell


def build_notebook(
    config: SubmissionConfig, build: Mapping[str, str], submit_source: str
) -> dict[str, Any]:
    """Build the submission notebook (see the module docstring for its cells).

    Why: only the Config cell depends on the config; every other cell is fixed text or source
    embedded verbatim, so two builds differ exactly where their configs and builds differ. A
    scoped build (task P2.6) also takes ``SCOPE_EDITS`` in four fixed cells, the lines on the
    single run's LoRA settings, and an adapter build (task P3.7) the adapter Helpers and
    ``ADAPTER_EDITS``. A hybrid build (task P6.5) takes ``HYBRID_EDITS`` in two fixed cells and
    adds four: the agent check after the environment check, the agent stage after the Write
    cell. Every other build's fixed cells are as before, byte for byte.

    Args:
        config: The checked config, a hybrid build's agent resolved (``resolve_agent``).
        build: The provenance from ``build_info``.
        submit_source: ``arc26/submit.py`` as the wheel holds it.

    Returns:
        The notebook as an nbformat 4.5 mapping, without Kaggle editor settings in its metadata.

    Raises:
        BuildError: If a generated code cell does not compile, or a hybrid build's agent run
            config was not read.
        RuntimeError: If a fixed cell no longer holds the text a scope or hybrid edit replaces.
    """
    cells = [
        ("config-note", "markdown", config_note(config, build)),
        ("config", "code", config_cell_source(config, build)),
        ("helpers-note", "markdown", HELPERS_NOTE),
        ("helpers", "code", helpers_cell_source()),
        ("validator-note", "markdown", VALIDATOR_NOTE),
        ("validator", "code", validator_cell_source(build, submit_source)),
        ("environment-note", "markdown", ENVIRONMENT_NOTE),
        ("environment", "code", ENVIRONMENT_CODE),
        ("install-note", "markdown", INSTALL_NOTE),
        ("install", "code", INSTALL_CODE),
        ("run-note", "markdown", RUN_NOTE),
        ("run", "code", RUN_CODE),
        ("write-note", "markdown", WRITE_NOTE),
        ("write", "code", WRITE_CODE),
        ("logs-note", "markdown", LOGS_NOTE),
        ("logs", "code", logs_cell_source()),
    ]
    if config.scope is not None:
        cells = [
            (name, kind, _edited(text, SCOPE_EDITS.get(name, ()), name))
            for name, kind, text in cells
        ]
    if config.adapters is not None:
        cells = [
            (
                name,
                kind,
                helpers_cell_source(adapters=True)
                if name == "helpers"
                else _edited(text, ADAPTER_EDITS.get(name, ()), name, "ADAPTER_EDITS"),
            )
            for name, kind, text in cells
        ]
    if config.dev240:
        # DEV240 (task P6.5b): the environment check writes the 240 tasks in development mode.
        edits = dev240_edits()
        cells = [
            (name, kind, _edited(text, edits.get(name, ()), name, "dev240_edits"))
            for name, kind, text in cells
        ]
    if config.agent is not None:
        cells = _hybrid_cells(config, cells)
    return _assemble(cells)


def _hybrid_cells(
    config: SubmissionConfig, cells: list[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    # A hybrid build's cells (task P6.5): HYBRID_EDITS in the Run note and the Run cell, the agent
    # check after the environment check (before anything is installed or solved), and the agent
    # stage after the Write cell (after NVARC's submission.json is written).
    added = {
        "environment": (
            ("agent-check-note", "markdown", AGENT_CHECK_NOTE),
            ("agent-check", "code", agent_check_cell_source(config)),
        ),
        "write": (("agent-note", "markdown", AGENT_NOTE), ("agent", "code", AGENT_CODE)),
    }
    out = []
    for name, kind, text in cells:
        out.append((name, kind, _edited(text, HYBRID_EDITS.get(name, ()), name, "HYBRID_EDITS")))
        out.extend(added.get(name, ()))
    if config.agent.shot:
        # A shot build (task P6.5b): SHOT_EDITS after HYBRID_EDITS, in the added cells too.
        out = [
            (name, kind, _edited(text, SHOT_EDITS.get(name, ()), name, "SHOT_EDITS"))
            for name, kind, text in out
        ]
    if config.agent.warmup is not None:
        # The weights' warm-up (task P6.9c): WARMUP_EDITS after SHOT_EDITS.
        out = [
            (name, kind, _edited(text, WARMUP_EDITS.get(name, ()), name, "WARMUP_EDITS"))
            for name, kind, text in out
        ]
    if config.agent.skip_nvarc:
        # LLM first (task P5.1): SKIP_NVARC_EDITS after WARMUP_EDITS (a build without a tail).
        out = [
            (name, kind, _edited(text, SKIP_NVARC_EDITS.get(name, ()), name, "SKIP_NVARC_EDITS"))
            for name, kind, text in out
        ]
    if config.agent.tail is not None:
        # The rank-256 tail (task P4.7b): TAIL_EDITS after WARMUP_EDITS, then its cells after
        # the agent stage's.
        out = [
            (name, kind, _edited(text, TAIL_EDITS.get(name, ()), name, "TAIL_EDITS"))
            for name, kind, text in out
        ]
        at = next(i for i, (name, _, _) in enumerate(out) if name == "agent") + 1
        out[at:at] = [
            ("tail-note", "markdown", TAIL_NOTE),
            ("tail", "code", tail_cell_source(config.agent)),
        ]
    return out


def build_script_notebook(config: ScriptConfig, build: Mapping[str, str]) -> dict[str, Any]:
    """Build a script notebook (task P0.5d; see the module docstring for its cells).

    Why: a script runs on the submission notebook's stack and plumbing (offline install, inputs
    found by search, sha256 checks, the bf16 stop, ``run_logs/``), but with no solver, no
    submission, and no competition rerun. Only the Config cell and the first markdown cell depend
    on the config.

    Args:
        config: The checked script config.
        build: The provenance from ``build_info``, with the script's and its config's sha256.

    Returns:
        The notebook as an nbformat 4.5 mapping, without Kaggle editor settings in its metadata.

    Raises:
        BuildError: If a generated code cell does not compile.
    """
    cells = [
        ("config-note", "markdown", script_config_note(config, build)),
        ("config", "code", script_config_cell_source(config, build)),
        ("helpers-note", "markdown", SCRIPT_HELPERS_NOTE),
        ("helpers", "code", script_helpers_cell_source()),
        ("environment-note", "markdown", SCRIPT_ENVIRONMENT_NOTE),
        ("environment", "code", SCRIPT_ENVIRONMENT_CODE),
        ("install-note", "markdown", SCRIPT_INSTALL_NOTE),
        ("install", "code", SCRIPT_INSTALL_CODE),
        ("script-note", "markdown", SCRIPT_RUN_NOTE),
        ("script", "code", SCRIPT_RUN_CODE),
        ("logs-note", "markdown", SCRIPT_LOGS_NOTE),
        ("logs", "code", logs_cell_source(SCRIPT_LOGS_TAIL)),
    ]
    return _assemble(cells)


def _assemble(cells: list[tuple[str, str, str]]) -> dict[str, Any]:
    # Compile every code cell, then lay the cells out as nbformat 4.5.
    for name, kind, text in cells:
        if kind == "code":
            try:
                compile(text, f"<{name} cell>", "exec")
            except SyntaxError as err:
                raise BuildError(f"the generated {name} cell does not compile: {err}") from err
    return {
        "cells": [_cell(i, *cell) for i, cell in enumerate(cells, start=1)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_bytes(notebook: Mapping[str, Any]) -> bytes:
    """Serialize a notebook as the builders write it: one-space indent, UTF-8, final newline.

    Args:
        notebook: The mapping from ``build_notebook``.

    Returns:
        The file's bytes; equal inputs give equal bytes.
    """
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def wheel_member(wheel: Path, member: str = SUBMIT_MEMBER) -> str:
    """Return a text file from inside a wheel.

    Why: the notebook embeds ``arc26/submit.py`` exactly as the wheel it installs holds it, so the
    validator that runs before the install is the one the package ships.

    Args:
        wheel: The wheel file.
        member: Path of the file inside the wheel.

    Returns:
        The file's text.

    Raises:
        BuildError: If the wheel cannot be read or does not hold ``member`` as UTF-8 text.
    """
    try:
        with zipfile.ZipFile(wheel) as archive:
            return archive.read(member).decode("utf-8")
    except (OSError, KeyError, zipfile.BadZipFile, UnicodeDecodeError) as err:
        raise BuildError(f"cannot read {member} from {wheel}: {err}") from err


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config", required=True, help="configs/submission_<name>.yaml, or a kind: script config"
    )
    parser.add_argument("--wheel", required=True, help="the arc26 wheel the notebook installs")
    parser.add_argument("--commit", required=True, help="full SHA of the commit the wheel is from")
    parser.add_argument("--tag", required=True, help="build tag recorded in the notebook")
    parser.add_argument("--out", required=True, help="notebook file to write")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the notebook alone, for review; ``scripts/build_submission.py`` builds the folder.

    A script config's script and config, a round arm's governor config, and a hybrid build's
    agent run config are read from this checkout, at their paths in the repository.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 when the build is refused.
    """
    args = build_parser().parse_args(argv)
    config_path, wheel, out = Path(args.config), Path(args.wheel), Path(args.out)
    try:
        try:
            config_bytes, wheel_bytes = config_path.read_bytes(), wheel.read_bytes()
        except OSError as err:
            raise BuildError(f"cannot read an input: {err}") from err

        def read(name: str) -> bytes:
            return (REPO / name).read_bytes()

        # A round arm's governor config, an adapter build's H config, and a hybrid build's agent
        # run config (tasks P2.3, P3.7, P6.5).
        config = parse_config(config_bytes, config_path)
        config = resolve_agent(resolve_adapters(resolve_rounds(config, read), read), read)
        resolved = config_path.resolve()
        shown = resolved.relative_to(REPO) if resolved.is_relative_to(REPO) else resolved
        hashes = {}
        if config.kind == "script":
            try:
                script = (REPO / config.script).read_bytes()
                script_config = (REPO / config.script_config).read_bytes()
            except OSError as err:
                raise BuildError(f"cannot read an input: {err}") from err
            check_script_files(config, script, script_config)
            hashes = {
                "script_sha256": hashlib.sha256(script).hexdigest(),
                "script_config_sha256": hashlib.sha256(script_config).hexdigest(),
            }
        build = build_info(
            config,
            tag=args.tag,
            commit=args.commit,
            config_file=shown.as_posix(),
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            wheel=wheel.name,
            wheel_sha256=hashlib.sha256(wheel_bytes).hexdigest(),
            **hashes,
        )
        if config.kind == "script":
            notebook = build_script_notebook(config, build)
        else:
            notebook = build_notebook(config, build, wheel_member(wheel))
        data = notebook_bytes(notebook)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        except OSError as err:
            raise BuildError(f"cannot write {out}: {err}") from err
    except BuildError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
