r"""GPU spike of the agent solver on 4x L4: vLLM boot, throughput, and a solve probe (task P6.3).

Purpose: ADR-020 turns the entry toward an agent that writes Python programs with Qwen3.8-27B.
Its gate needs two numbers nobody has measured on Kaggle's 4x L4: the engine's generated tokens
per second at the agent's concurrency and context, and whether the model solves ARC tasks
through programs at all. Run ``P6.3-01`` (``docs/prerun/P6.3-01.md``) measures both in one
session; run ``P6.3-02`` (the full harness on DEV40) reuses the engine code. Runbook:
``docs/runbooks/agent_spike.md``.

Inputs: the INT4 weights and the engine wheelhouse, the outputs of the two download notebooks
(``scripts/build_download_notebooks.py``), attached as notebook inputs and passed as
``--model-root`` and ``--wheelhouse-root``; the competition's ARC-AGI-2 training challenges and
solutions, found under ``--input-root``. Outputs, in ``out_dir``: ``run.json``, ``config.yaml``
(resolved), ``metrics.json``, ``summary.md``, ``timing.jsonl``, ``throughput.jsonl``,
``throughput_cells.json``, ``probe.jsonl``, ``boot.json``, ``install.json``, ``engine_prep.json``,
``preflight.json``, ``server_metrics.jsonl``, ``gpu_samples.jsonl``, and ``vllm_server.log``
(head and tail); ``harness.log`` (every log record of the arc26 package, tracebacks included);
and with the agent phase ``harness_preflight.json`` and the ``agent_*`` files.

Method: install the wheelhouse offline into a venv cut off from the image's packages, with the
installer the wheels notebook shipped and checked; prepare it (``prepare_engine``: drop the
FlashInfer jit-cache provider the wheelhouse holds without its shim, which killed attempt 2's
boot, and import what the engine's workers import); check that torch 2.13 sees the GPUs; start
``vllm serve`` with tensor parallelism 4, an FP8 KV cache, prefix caching, Triton attention, and
no speculative decoding; record the boot time and the KV pool and maximum concurrency vLLM
prints. Throughput: for each mean context (about 8k and 32k tokens: synthetic ARC-like prompts
of digit-row grids plus 2048 generated tokens with ``ignore_eos``) and each concurrency (1, 8,
16, 32, 64), send that many streaming requests at once with thinking on and measure the
aggregate output tokens per second over the wave, the steady decode rate while every request is
decoding, the time to first token, and degenerate outputs (64 identical tokens in a row, the
symptom of NaN logits), while a sampler records memory per GPU and the server's counters. Probe:
10 ARC-AGI-2 training tasks (a seeded sample of the small ones with one test input), thinking on
at low effort, one request for ``transform(grid)`` and one feedback round; a program is verified
when a subprocess checker reproduces every demonstration exactly with it, and correct when it
also reproduces the training solution of the test input.

``--smoke`` runs the same code on the CPU against a fake OpenAI-compatible server (this file
with ``--fake-server``) on three fixture tasks, with tiny cells; no install, no GPU.

Usage::

    python scripts/agent_spike.py --config configs/P6.3-01.yaml --commit <sha> \
        --model-root <dir> --wheelhouse-root <dir>
    python scripts/agent_spike.py --config configs/P6.3-01.yaml --smoke --out-dir <dir>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import hashlib
import http.client
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

# Exit codes: 0 every phase ran; 1 an input, install, boot, or kill criterion stopped the run;
# 2 the config or the arguments were refused; 3 the wall-time budget ended the run first.
EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_WALL = 0, 1, 2, 3
TOP_KEYS = (
    "run_id", "seed", "max_wall_s", "phases", "input_root", "out_dir", "scratch_dir", "model",
    "wheels", "server", "sanity", "degenerate", "kill",
)  # fmt: skip
# Sections required only when a phase needs them.
PHASE_SECTIONS = {
    "throughput": "throughput", "probe": "probe", "agent": "agent", "preserve": "preserve",
    "forcing": "forcing", "direct": "direct",
}  # fmt: skip
MODEL_KEYS = ("folder", "revision", "verify_hashes")
WHEELS_KEYS = (
    "folder", "vllm", "base_python", "venv", "timeout_s", "engine_imports",
    "engine_check_timeout_s",
)  # fmt: skip
SERVER_KEYS = (
    "host", "port", "served_model_name", "tensor_parallel_size", "kv_cache_dtype",
    "enable_prefix_caching", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "gpu_memory_utilization", "attention_backend", "language_model_only", "reasoning_parser",
    "generation_config", "extra_args", "env", "unset_env", "boot_timeout_s", "stop_timeout_s",
    "sample_interval_s",
)  # fmt: skip
SANITY_KEYS = ("max_tokens", "reasoning_effort")
THROUGHPUT_KEYS = (
    "output_tokens", "concurrency", "reasoning_effort", "contexts", "min_cell_s", "warmup_tokens",
)  # fmt: skip
CONTEXT_KEYS = ("prompt_tokens", "cell_timeout_s", "budget_s")
PROBE_KEYS = (
    "competition", "challenges", "solutions", "n_tasks", "max_task_chars", "rounds",
    "max_tokens", "reasoning_effort", "budget_s", "checker_timeout_s", "checker_memory_mb",
)  # fmt: skip
DEGENERATE_KEYS = ("max_run", "special_min_id", "eos_ids")
KILL_KEYS = ("degenerate_share", "error_share")
AGENT_KEYS = (
    "competition", "challenges", "solutions", "subset", "nvarc_misses", "concurrency",
    "context_tokens", "verifier_workers", "call_timeout_s", "memory_mb", "cell_timeout_s",
    "forcing_tokens", "min_tokens_per_s", "stall_s", "phases",
)  # fmt: skip
AGENT_PHASE_KEYS = (
    "max_tokens", "token_budget", "rounds", "reasoning_effort", "temperature", "budget_s",
    "preserve_thinking",
)  # fmt: skip
PRESERVE_KEYS = ("n_tasks", "turn1_max_tokens", "reasoning_effort", "tolerance_tokens", "budget_s")
# Run P6.3-04's direct-answer arm and the readings of its two routes.
DIRECT_KEYS = (
    "concurrency", "context_tokens", "max_tokens", "token_budget", "forcing_tokens",
    "reasoning_effort", "budget_s",
)  # fmt: skip
READING_KEYS = (
    "enter_misses", "affordable_tokens_per_task", "contamination_gap", "max_cut_share",
)  # fmt: skip
# Direct keys that a config may leave out (run P6.3-05 sets them), with their defaults: the tasks
# to run (None: the agent's subset), whether a view is the second sample, and how many answers a
# reply may give per test input (2: pass@2 from one reply).
DIRECT_OPTIONAL = {
    "task_ids": None, "with_view": True, "answers_per_test": 1, "view_ops": None,
    # Run P6.9-01 (task P6.9): the answer forced inside the thinking request through vLLM's
    # thinking budget, the phrase the server forces before </think>, and the lenient answer
    # parser with the implied-shape choice of the first attempt. Off: P6.3-05's request.
    "forcing_in_request": False, "thinking_end_note": "", "lenient_answers": False,
    # Run P6.9-01: the client-side cut of an in-request thinking request, in seconds (None: none).
    "turn_cut_s": None,
}  # fmt: skip
# The single-shot reading (run P6.3-05: one sample per task, no view).
SHOT_READING_KEYS = ("min_solves", "margin_over_nvarc", "nvarc_solved", "max_cut_share")
# Agent keys that a config may leave out, with their defaults.
AGENT_OPTIONAL = {"only_nvarc_misses": False}
FORCING_KEYS = (
    "n_tasks", "turn1_max_tokens", "reasoning_effort", "min_programs", "tolerance_tokens",
    "budget_s",
)  # fmt: skip
EFFORTS = (None, "low", "medium", "xhigh")
# Files the script notebook writes into run_logs/ itself; the script never writes these names.
NOTEBOOK_LOGS = ("candidates.json", "env.json", "pip_freeze.txt", "stages.json")
# Lines of the server log kept in out_dir: the boot (head) and the end (tail).
LOG_HEAD_LINES, LOG_TAIL_LINES = 400, 2000
# Deepest folder level searched for inputs below a root.
SEARCH_DEPTH = 6
FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL)


class ConfigError(Exception):
    """The config or the arguments break a rule; nothing ran."""


class Stop(Exception):
    """A problem that ends the run as an Error (exit 1): missing input, failed install or boot."""


class Kill(Stop):
    """A kill criterion fired (brief, section 7): degenerate or failed outputs, a dead server."""


class WallTime(Stop):
    """The wall-time budget ran out (exit 3), for example during the engine's boot."""


# --- config ------------------------------------------------------------------------------------


def _section(raw: object, keys: Sequence[str], where: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(map(str, raw)) != set(keys):
        found = sorted(map(str, raw)) if isinstance(raw, dict) else raw
        raise ConfigError(f"{where} must have exactly the keys {', '.join(keys)}; got {found}")
    return dict(raw)


def _int(value: object, where: str, low: int = 0) -> int:
    if type(value) is not int or value < low:
        raise ConfigError(f"{where} must be an integer of at least {low}; got {value!r}")
    return value


def _number(value: object, where: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not low <= value <= high:
        raise ConfigError(f"{where} must be a number from {low} to {high}; got {value!r}")
    return float(value)


def _effort(value: object, where: str) -> str | None:
    if value not in EFFORTS:
        raise ConfigError(f"{where} must be one of {EFFORTS}; got {value!r}")
    return value


def phase_kind(phase: str) -> str:
    """Return the kind of a phase name: ``throughput``, ``probe``, or ``agent``.

    Args:
        phase: A name from the config's ``phases``, for example ``throughput-8k``.

    Returns:
        The kind.

    Raises:
        ConfigError: If the name is not ``probe``, ``agent``, ``preserve``, ``forcing``, or
            ``throughput-<context>``.
    """
    if phase in ("probe", "agent", "preserve", "forcing", "direct"):
        return phase
    if phase.startswith("throughput-") and len(phase) > len("throughput-"):
        return "throughput"
    raise ConfigError(
        f"phase {phase!r} must be probe, agent, preserve, forcing, direct, or throughput-<context>"
    )


def check_config(raw: object) -> dict[str, Any]:
    """Check a parsed config against the rules its comments state.

    Args:
        raw: The parsed YAML.

    Returns:
        The config as plain dicts.

    Raises:
        ConfigError: If a key is missing or unknown, or a value breaks its rule.
    """
    if not isinstance(raw, dict):
        raise ConfigError("the config must be a mapping")
    phases = raw.get("phases")
    if not isinstance(phases, list) or not phases or len(set(phases)) != len(phases):
        raise ConfigError("phases must be a non-empty list of distinct names")
    kinds = {phase_kind(str(p)) for p in phases}
    # The agent section serves its own phase and the phases that take tasks, limits, or health
    # rules from it (run P6.3-05 has it without an agent phase).
    if "agent" not in raw:
        if "preserve" in kinds:
            raise ConfigError("the preserve phase takes its tasks from the agent section")
        if "forcing" in kinds:
            raise ConfigError(
                "the forcing check takes its tasks and limits from the agent section"
            )
        if "direct" in kinds:
            raise ConfigError(
                "the direct arm takes its tasks and health rules from the agent section"
            )
    if kinds & {"preserve", "forcing", "direct"}:
        kinds = kinds | {"agent"}
    keys = list(TOP_KEYS) + [PHASE_SECTIONS[k] for k in sorted(kinds)]
    if "direct" in kinds:
        keys.append("reading")
    config = _section(raw, keys, "the config")
    if not isinstance(config["run_id"], str) or not re.fullmatch(r"P[0-9.]+-\w+", config["run_id"]):
        raise ConfigError(f"run_id must look like P6.3-01; got {config['run_id']!r}")
    _int(config["seed"], "seed")
    _int(config["max_wall_s"], "max_wall_s", 60)
    for key in ("input_root", "out_dir", "scratch_dir"):
        if not isinstance(config[key], str) or not config[key]:
            raise ConfigError(f"{key} must be a path")
    model = config["model"] = _section(config["model"], MODEL_KEYS, "model")
    if not isinstance(model["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", model["revision"]
    ):
        raise ConfigError("model.revision must be a full commit")
    if not isinstance(model["verify_hashes"], bool):
        raise ConfigError("model.verify_hashes must be true or false")
    wheels = config["wheels"] = _section(config["wheels"], WHEELS_KEYS, "wheels")
    _int(wheels["timeout_s"], "wheels.timeout_s", 60)
    modules = wheels["engine_imports"]
    if (
        not isinstance(modules, list)
        or not modules
        or not all(
            isinstance(m, str) and re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", m)
            for m in modules
        )
    ):
        raise ConfigError("wheels.engine_imports must be a non-empty list of module names")
    _int(wheels["engine_check_timeout_s"], "wheels.engine_check_timeout_s", 30)
    server = config["server"] = _section(config["server"], SERVER_KEYS, "server")
    for key in ("port", "tensor_parallel_size", "max_model_len", "max_num_seqs"):
        _int(server[key], f"server.{key}", 1)
    _int(server["max_num_batched_tokens"], "server.max_num_batched_tokens", 1)
    _number(server["gpu_memory_utilization"], "server.gpu_memory_utilization", 0.1, 0.99)
    for key in ("enable_prefix_caching", "language_model_only"):
        if not isinstance(server[key], bool):
            raise ConfigError(f"server.{key} must be true or false")
    if not isinstance(server["extra_args"], list) or not all(
        isinstance(a, str) for a in server["extra_args"]
    ):
        raise ConfigError("server.extra_args must be a list of strings")
    if any(arg.startswith("--speculative") for arg in server["extra_args"]):
        raise ConfigError("server.extra_args: no speculative decoding (vLLM #55533, #55766)")
    env = server["env"]
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ConfigError("server.env must map names to string values (quote numbers)")
    if not isinstance(server["unset_env"], list):
        raise ConfigError("server.unset_env must be a list of names")
    for key in ("boot_timeout_s", "stop_timeout_s"):
        _int(server[key], f"server.{key}", 1)
    _number(server["sample_interval_s"], "server.sample_interval_s", 0.1, 600)
    sanity = config["sanity"] = _section(config["sanity"], SANITY_KEYS, "sanity")
    _int(sanity["max_tokens"], "sanity.max_tokens", 1)
    _effort(sanity["reasoning_effort"], "sanity.reasoning_effort")
    degenerate = config["degenerate"] = _section(
        config["degenerate"], DEGENERATE_KEYS, "degenerate"
    )
    _int(degenerate["max_run"], "degenerate.max_run", 2)
    _int(degenerate["special_min_id"], "degenerate.special_min_id", 1)
    eos = degenerate["eos_ids"]
    if not isinstance(eos, list) or not all(type(t) is int and t >= 0 for t in eos):
        raise ConfigError("degenerate.eos_ids must be a list of token ids")
    kill = config["kill"] = _section(config["kill"], KILL_KEYS, "kill")
    for key in KILL_KEYS:
        _number(kill[key], f"kill.{key}", 0.0, 1.0)
    if "throughput" in kinds:
        throughput = config["throughput"] = _section(
            config["throughput"], THROUGHPUT_KEYS, "throughput"
        )
        _int(throughput["output_tokens"], "throughput.output_tokens", 1)
        levels = throughput["concurrency"]
        if (
            not isinstance(levels, list)
            or not levels
            or not all(type(c) is int and 1 <= c <= server["max_num_seqs"] for c in levels)
        ):
            raise ConfigError("throughput.concurrency must be integers from 1 to max_num_seqs")
        _effort(throughput["reasoning_effort"], "throughput.reasoning_effort")
        _int(throughput["min_cell_s"], "throughput.min_cell_s", 1)
        _int(throughput["warmup_tokens"], "throughput.warmup_tokens", 0)
        contexts = throughput["contexts"]
        if not isinstance(contexts, dict) or not contexts:
            raise ConfigError("throughput.contexts must map names to settings")
        for name, settings in list(contexts.items()):
            settings = contexts[name] = _section(settings, CONTEXT_KEYS, f"context {name}")
            _int(settings["prompt_tokens"], f"context {name} prompt_tokens", 16)
            _int(settings["cell_timeout_s"], f"context {name} cell_timeout_s", 1)
            if settings["budget_s"] is not None:
                _int(settings["budget_s"], f"context {name} budget_s", 1)
            total = settings["prompt_tokens"] + throughput["output_tokens"]
            if total > server["max_model_len"]:
                raise ConfigError(f"context {name}: {total} tokens exceed server.max_model_len")
        for phase in phases:
            if phase_kind(phase) == "throughput" and phase[len("throughput-") :] not in contexts:
                raise ConfigError(f"phase {phase} names no context of throughput.contexts")
    if "agent" in kinds:
        check_agent(config, server)
    if "preserve" in kinds:
        preserve = config["preserve"] = _section(config["preserve"], PRESERVE_KEYS, "preserve")
        for key in ("n_tasks", "turn1_max_tokens", "tolerance_tokens", "budget_s"):
            _int(preserve[key], f"preserve.{key}", 1)
        if preserve["turn1_max_tokens"] >= server["max_model_len"]:
            raise ConfigError("preserve.turn1_max_tokens reaches server.max_model_len")
        _effort(preserve["reasoning_effort"], "preserve.reasoning_effort")
    if "forcing" in kinds:
        check_forcing(config, server)
    if "direct" in kinds:
        check_direct(config, server)
    if "probe" in kinds:
        probe = config["probe"] = _section(config["probe"], PROBE_KEYS, "probe")
        if "evaluation" in probe["challenges"] or "evaluation" in probe["solutions"]:
            raise ConfigError("the probe reads training tasks only (BRIEF A0)")
        for key in ("n_tasks", "max_task_chars", "rounds", "max_tokens", "budget_s"):
            _int(probe[key], f"probe.{key}", 1)
        _effort(probe["reasoning_effort"], "probe.reasoning_effort")
        _number(probe["checker_timeout_s"], "probe.checker_timeout_s", 0.1, 600)
        _int(probe["checker_memory_mb"], "probe.checker_memory_mb", 64)
    return config


def check_agent(config: dict[str, Any], server: Mapping[str, Any]) -> None:
    """Check the ``agent`` section of a stage-2 config (run P6.3-02) in place.

    Args:
        config: The config, its ``agent`` section still raw.
        server: The checked ``server`` section.

    Raises:
        ConfigError: If a key is missing or unknown, or a value breaks its rule.
    """
    raw_agent = dict(config["agent"]) if isinstance(config["agent"], dict) else config["agent"]
    optional = {}
    if isinstance(raw_agent, dict):
        optional = {k: raw_agent.pop(k) for k in AGENT_OPTIONAL if k in raw_agent}
    agent = config["agent"] = _section(raw_agent, AGENT_KEYS, "agent")
    for key, default in AGENT_OPTIONAL.items():
        agent[key] = optional.get(key, default)
        if not isinstance(agent[key], bool):
            raise ConfigError(f"agent.{key} must be true or false")
    if "evaluation" not in agent["challenges"] or "solutions" not in agent["solutions"]:
        raise ConfigError("agent: the evaluation challenges, with solutions for scoring only")
    if agent["subset"] not in ("DEV4", "DEV40", "DEV120"):
        raise ConfigError("agent.subset must be DEV4, DEV40, or DEV120 (arc26.data)")
    misses = agent["nvarc_misses"]
    if not isinstance(misses, list) or not all(
        isinstance(t, str) and re.fullmatch(r"[0-9a-z][0-9a-z-]*", t) for t in misses
    ):
        raise ConfigError("agent.nvarc_misses must be a list of task ids")
    concurrency = _int(agent["concurrency"], "agent.concurrency", 1)
    if concurrency > server["max_num_seqs"]:
        raise ConfigError("agent.concurrency exceeds server.max_num_seqs")
    context = _int(agent["context_tokens"], "agent.context_tokens", 1024)
    if context > server["max_model_len"]:
        raise ConfigError("agent.context_tokens exceeds server.max_model_len")
    _int(agent["verifier_workers"], "agent.verifier_workers", 1)
    _number(agent["call_timeout_s"], "agent.call_timeout_s", 0.1, 60)
    _int(agent["memory_mb"], "agent.memory_mb", 256)
    _number(agent["cell_timeout_s"], "agent.cell_timeout_s", 0.1, 600)
    _number(agent["min_tokens_per_s"], "agent.min_tokens_per_s", 0.0, 1000.0)
    _int(agent["stall_s"], "agent.stall_s", 60)
    forcing = agent["forcing_tokens"]
    if forcing is not None:
        # P6.2's Budget: a forcing reply is never requested with less than min_turn_tokens.
        _int(forcing, "agent.forcing_tokens", 256)
    phases = agent["phases"]
    if not isinstance(phases, list) or not phases:
        raise ConfigError("agent.phases must be a non-empty list")
    for index, phase in enumerate(phases):
        phase = phases[index] = _section(phase, AGENT_PHASE_KEYS, f"agent phase {index + 1}")
        for key in ("max_tokens", "token_budget", "rounds", "budget_s"):
            _int(phase[key], f"agent phase {index + 1} {key}", 1)
        if phase["max_tokens"] > phase["token_budget"]:
            raise ConfigError(f"agent phase {index + 1}: max_tokens exceeds token_budget")
        if phase["max_tokens"] + (forcing or 0) >= context:
            # A turn's cap and its forcing turn must leave room for the prompt in the context.
            raise ConfigError(
                f"agent phase {index + 1}: max_tokens and forcing_tokens reach "
                "agent.context_tokens"
            )
        if forcing is not None and phase["max_tokens"] + forcing > phase["token_budget"]:
            raise ConfigError(
                f"agent phase {index + 1}: max_tokens and forcing_tokens exceed token_budget"
            )
        _effort(phase["reasoning_effort"], f"agent phase {index + 1} reasoning_effort")
        if phase["temperature"] is not None:
            _number(phase["temperature"], f"agent phase {index + 1} temperature", 0.0, 2.0)
        if not isinstance(phase["preserve_thinking"], bool):
            raise ConfigError(f"agent phase {index + 1} preserve_thinking must be true or false")


def check_forcing(config: dict[str, Any], server: Mapping[str, Any]) -> None:
    """Check the ``forcing`` section (the forcing check before the agent's phases) in place.

    Args:
        config: The config, its ``agent`` section checked and its ``forcing`` section raw.
        server: The checked ``server`` section.

    Raises:
        ConfigError: If a key is missing or unknown, or a value breaks its rule.
    """
    forcing = config["forcing"] = _section(config["forcing"], FORCING_KEYS, "forcing")
    for key in ("n_tasks", "turn1_max_tokens", "min_programs", "tolerance_tokens", "budget_s"):
        _int(forcing[key], f"forcing.{key}", 1)
    if forcing["min_programs"] > forcing["n_tasks"]:
        raise ConfigError("forcing.min_programs exceeds forcing.n_tasks")
    if forcing["turn1_max_tokens"] + config["agent"]["forcing_tokens"] >= server["max_model_len"]:
        raise ConfigError("forcing: turn 1 and the forcing reply reach server.max_model_len")
    _effort(forcing["reasoning_effort"], "forcing.reasoning_effort")


def check_direct(config: dict[str, Any], server: Mapping[str, Any]) -> None:
    """Check the ``direct`` and ``reading`` sections (run P6.3-04) in place.

    Args:
        config: The config, its ``direct`` and ``reading`` sections raw.
        server: The checked ``server`` section.

    Raises:
        ConfigError: If a key is missing or unknown, or a value breaks its rule.
    """
    from arc26.agent.views import OPS

    raw = dict(config["direct"]) if isinstance(config["direct"], dict) else config["direct"]
    optional = {}
    if isinstance(raw, dict):
        optional = {k: raw.pop(k) for k in DIRECT_OPTIONAL if k in raw}
    direct = config["direct"] = _section(raw, DIRECT_KEYS, "direct")
    direct.update({k: optional.get(k, default) for k, default in DIRECT_OPTIONAL.items()})
    if not isinstance(direct["with_view"], bool):
        raise ConfigError("direct.with_view must be true or false")
    if direct["answers_per_test"] not in (1, 2):
        raise ConfigError("direct.answers_per_test must be 1 or 2")
    ids = direct["task_ids"]
    if ids is not None and (
        not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
        or not all(isinstance(t, str) and re.fullmatch(r"[0-9a-f]{8}", t) for t in ids)
    ):  # fmt: skip
        raise ConfigError("direct.task_ids must be a list of distinct task ids, or null")
    for key in ("concurrency", "max_tokens", "token_budget", "budget_s"):
        _int(direct[key], f"direct.{key}", 1)
    _int(direct["forcing_tokens"], "direct.forcing_tokens", 256)
    if direct["concurrency"] > server["max_num_seqs"]:
        raise ConfigError("direct.concurrency exceeds server.max_num_seqs")
    context = _int(direct["context_tokens"], "direct.context_tokens", 1024)
    if context > server["max_model_len"]:
        raise ConfigError("direct.context_tokens exceeds server.max_model_len")
    for key in ("forcing_in_request", "lenient_answers"):
        if not isinstance(direct[key], bool):
            raise ConfigError(f"direct.{key} must be true or false")
    # Under in-request forcing the thinking request holds the answer's forcing_tokens too, and
    # the attempt keeps a separate forcing turn's for a server that ignores the budget.
    forcing = direct["forcing_tokens"] * (2 if direct["forcing_in_request"] else 1)
    if direct["max_tokens"] + forcing >= context:
        raise ConfigError("direct: max_tokens and forcing_tokens reach direct.context_tokens")
    if direct["max_tokens"] + forcing > direct["token_budget"]:
        raise ConfigError("direct: max_tokens and forcing_tokens exceed token_budget")
    note = direct["thinking_end_note"]
    if not isinstance(note, str) or "think>" in note:
        raise ConfigError("direct.thinking_end_note must be a string without think tags")
    cut = direct["turn_cut_s"]
    if cut is not None and (
        not direct["forcing_in_request"] or isinstance(cut, bool)
        or not isinstance(cut, int | float) or not 0 < cut < direct["budget_s"]
    ):  # fmt: skip
        raise ConfigError("direct.turn_cut_s needs forcing_in_request and must be under budget_s")
    if direct["forcing_in_request"]:
        if not note.strip():
            raise ConfigError("direct.forcing_in_request needs a thinking_end_note")
        from arc26.agent.shot import THINK_END, reasoning_end_str

        try:
            end = reasoning_end_str([str(a) for a in server["extra_args"] or ()])
        except Exception as err:
            raise ConfigError(str(err)) from err
        if end != note + THINK_END:
            raise ConfigError(
                "direct.forcing_in_request needs server.extra_args to hold --reasoning-config "
                "with reasoning_end_str equal to direct.thinking_end_note followed by </think>"
            )
    _effort(direct["reasoning_effort"], "direct.reasoning_effort")
    ops = direct["view_ops"]
    if direct["with_view"] and (
        not isinstance(ops, list) or not ops or any(op not in OPS for op in ops)
    ):
        raise ConfigError(f"direct.view_ops must be a non-empty list of {OPS}")
    if not direct["with_view"]:
        # Run P6.3-05's single-shot reading: one sample per task against NVARC on the same tasks.
        reading = config["reading"] = _section(config["reading"], SHOT_READING_KEYS, "reading")
        _int(reading["min_solves"], "reading.min_solves", 1)
        _int(reading["margin_over_nvarc"], "reading.margin_over_nvarc", 0)
        _number(reading["max_cut_share"], "reading.max_cut_share", 0.0, 1.0)
        solved = reading["nvarc_solved"]
        if not isinstance(solved, list) or not all(isinstance(t, str) for t in solved):
            raise ConfigError("reading.nvarc_solved must be a list of task ids")
        if ids is not None and not set(solved) <= set(ids):
            raise ConfigError("reading.nvarc_solved must be among direct.task_ids")
        return
    reading = config["reading"] = _section(config["reading"], READING_KEYS, "reading")
    _int(reading["enter_misses"], "reading.enter_misses", 1)
    _int(reading["affordable_tokens_per_task"], "reading.affordable_tokens_per_task", 1)
    _number(reading["contamination_gap"], "reading.contamination_gap", 0.0, 1.0)
    _number(reading["max_cut_share"], "reading.max_cut_share", 0.0, 1.0)


def load_config(path: Path) -> dict[str, Any]:
    """Read and check a spike config.

    Args:
        path: The YAML file.

    Returns:
        The checked config.

    Raises:
        ConfigError: If the file cannot be read or ``check_config`` refuses it.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        raise ConfigError(f"{path}: cannot read: {err}") from err
    try:
        return check_config(raw)
    except ConfigError as err:
        raise ConfigError(f"{path}: {err}") from err


# --- small utilities -----------------------------------------------------------------------------


def write_json(path: Path, value: object) -> None:
    """Write JSON through a temporary file and ``os.replace``, so a reader never sees half a file.

    Args:
        path: The file; its folder is created when missing.
        value: A JSON value; other objects are written as their ``str``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class JsonlWriter:
    """Append-only JSONL files shared by threads; each record is flushed as written."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def append(self, path: Path, record: Mapping[str, Any]) -> None:
        """Append one record as one line.

        Args:
            path: The JSONL file.
            record: A JSON-serializable mapping.
        """
        line = json.dumps(record, default=str) + "\n"
        with self._lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(line)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Return the ``q`` percentile (0 to 100) by linear interpolation, numpy's default.

    Args:
        values: The sample.
        q: The percentile.

    Returns:
        The value, or None for an empty sample.
    """
    data = sorted(values)
    if not data:
        return None
    position = (len(data) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (position - low)


def find_files(root: Path, names: Sequence[str], max_depth: int = SEARCH_DEPTH) -> dict[str, list]:
    """Find files by exact name under a root, at most ``max_depth`` folder levels down.

    Args:
        root: The folder searched.
        names: File names.
        max_depth: Deepest level; a file in ``root`` is at level 1.

    Returns:
        For every name, the sorted paths found.
    """
    found: dict[str, list[Path]] = {name: [] for name in names}
    skipped = {"site-packages", "dist-packages", "__pycache__", ".git", "node_modules"}
    root = root.resolve()
    for directory, subdirectories, files in os.walk(root, followlinks=True):
        level = len(Path(directory).relative_to(root).parts)
        subdirectories[:] = (
            sorted(d for d in subdirectories if d not in skipped) if level + 1 < max_depth else []
        )
        for name in files:
            if name in found:
                found[name].append(Path(directory) / name)
    return {name: sorted(paths) for name, paths in found.items()}


def clean_env(environ: Mapping[str, str], unset: Sequence[str] = ()) -> dict[str, str]:
    """Return an environment cut off from the image's Python packages and pip settings.

    Why: Kaggle's ``PYTHONPATH`` carries the attached install notebook (torch 2.8 and its stack),
    which would shadow the venv's torch 2.13 in every engine process.

    Args:
        environ: The current environment.
        unset: More names to remove.

    Returns:
        A copy without ``PYTHONPATH``, ``PYTHONHOME``, ``PYTHONSTARTUP``, ``PYTHONUSERBASE``,
        ``VIRTUAL_ENV``, any ``PIP_*``, and ``unset``, with ``PYTHONNOUSERSITE=1``.
    """
    dropped = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"}
    dropped |= set(unset)
    env = {k: v for k, v in environ.items() if k not in dropped and not k.startswith("PIP_")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def free_port() -> int:
    """Return a free local TCP port (for the smoke test's fake server)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- inputs --------------------------------------------------------------------------------------


def read_manifest(folder: Path) -> dict[str, Any]:
    """Read a download notebook's ``MANIFEST.json`` and require ``ok``.

    Args:
        folder: A folder holding ``MANIFEST.json``.

    Returns:
        The manifest.

    Raises:
        Stop: If it is missing, unreadable, or not ``ok``.
    """
    path = folder / "MANIFEST.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise Stop(f"{path}: cannot read the download notebook's manifest: {err}") from err
    if manifest.get("ok") is not True:
        raise Stop(f"{path} says the download did not pass (ok {manifest.get('ok')!r})")
    return manifest


def resolve_model(root: Path, settings: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Find the verified weights in the model notebook's output and check every file's size.

    Args:
        root: The output's mount folder (``--model-root``).
        settings: The config's ``model`` section.

    Returns:
        The model folder and a record of the checks.

    Raises:
        Stop: If the manifest is not ok, names another revision, or a file is missing or its size
            or (with ``verify_hashes``) its sha256 differs.
    """
    manifest = read_manifest(root)
    if manifest.get("revision") != settings["revision"]:
        raise Stop(f"the model output is revision {manifest.get('revision')}, not the pinned one")
    folder = root / settings["folder"]
    problems, checked = [], 0
    for row in manifest.get("files", []):
        path = folder / row["name"]
        if not path.is_file():
            problems.append(f"{row['name']} missing")
            continue
        if path.stat().st_size != row["size"]:
            problems.append(f"{row['name']} has {path.stat().st_size} bytes, not {row['size']}")
            continue
        if settings["verify_hashes"]:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 24), b""):
                    digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                problems.append(f"{row['name']} sha256 differs from the manifest")
        checked += 1
    if problems or not checked or not (folder / "config.json").is_file():
        raise Stop(f"model folder {folder}: {problems or 'no files listed or no config.json'}")
    return folder, {
        "folder": str(folder),
        "files_checked": checked,
        "hashed": settings["verify_hashes"],
    }


def resolve_wheelhouse(root: Path, settings: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Find the checked wheelhouse in the wheels notebook's output.

    Args:
        root: The output's mount folder (``--wheelhouse-root``).
        settings: The config's ``wheels`` section.

    Returns:
        The wheelhouse folder and a record of the checks.

    Raises:
        Stop: If the manifest is not ok, lacks the pinned vLLM, or the installer is missing.
    """
    folder = root / settings["folder"]
    manifest = read_manifest(folder)
    versions = {row.get("name"): row.get("version") for row in manifest.get("files", [])}
    if versions.get("vllm") != settings["vllm"]:
        raise Stop(f"the wheelhouse holds vllm {versions.get('vllm')}, not {settings['vllm']}")
    if not (folder / "install_wheelhouse.py").is_file():
        raise Stop(f"{folder} has no install_wheelhouse.py")
    check = manifest.get("install_check") or {}
    return folder, {
        "folder": str(folder),
        "wheels": sum(1 for row in manifest.get("files", []) if row.get("name")),
        "torch": versions.get("torch"),
        "cpu_install_check_s": check.get("seconds"),
    }


def resolve_task_files(root: Path, settings: Mapping[str, Any]) -> tuple[Path, Path]:
    """Find the probe's challenges and solutions files: each the one under the competition folder.

    Args:
        root: The input root.
        settings: The config's ``probe`` section.

    Returns:
        The challenges and the solutions paths.

    Raises:
        Stop: If either is not found exactly once.
    """
    names = [settings["challenges"], settings["solutions"]]
    found = find_files(root, names)
    paths = []
    for name in names:
        inside = [p for p in found[name] if settings["competition"] in p.parts] or found[name]
        if len(inside) != 1:
            raise Stop(f"{len(inside)} files named {name} under {root}, expected 1: {inside}")
        paths.append(inside[0])
    return paths[0], paths[1]


# --- install and preflight ---------------------------------------------------------------


PREFLIGHT = """\
import json, sys, torch
want = int(sys.argv[1])
info = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "available": torch.cuda.is_available(), "count": torch.cuda.device_count()}
devices = []
for index in range(info["count"]):
    props = torch.cuda.get_device_properties(index)
    x = torch.randn(512, 512, device=f"cuda:{index}", dtype=torch.bfloat16)
    devices.append({"name": props.name, "capability": f"{props.major}.{props.minor}",
                    "memory_bytes": props.total_memory, "matmul": float((x @ x).float().sum())})
info["devices"] = devices
if info["available"]:
    info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    info["p2p"] = {f"{a}->{b}": torch.cuda.can_device_access_peer(a, b)
                   for a in range(info["count"]) for b in range(info["count"]) if a != b}
keys = ("libnccl", "libcudart", "libcublas", "libcudnn")
with open("/proc/self/maps") as maps:
    info["libraries"] = sorted({line.split()[-1] for line in maps
                                if "/" in line and any(k in line for k in keys)})
info["ok"] = info["available"] and info["count"] >= want
print("PREFLIGHT " + json.dumps(info))
"""


def run_command(
    command: list[str], log: Path, timeout_s: float, env: Mapping[str, str]
) -> tuple[int, list[str]]:
    """Run a command with its output appended to a log; return the exit code and the last lines.

    Args:
        command: Program and arguments; no shell.
        log: The log file.
        timeout_s: Seconds before the command is killed (exit 124).
        env: Its environment.

    Returns:
        The exit code (127 when it cannot start, 124 at the timeout) and its last 50 lines.
    """
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(env),
                text=True, errors="replace", timeout=timeout_s, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired as err:
            text = err.stdout if isinstance(err.stdout, str) else ""
            handle.write(text + f"\ntimeout after {timeout_s} s\n")
            return 124, text.splitlines()[-50:]
        except OSError as err:
            handle.write(f"could not run: {err}\n")
            return 127, [str(err)]
        handle.write(result.stdout)
    return result.returncode, result.stdout.splitlines()[-50:]


def install_engine(
    wheelhouse: Path, settings: Mapping[str, Any], out: Path, env: Mapping[str, str]
) -> tuple[Path, dict[str, Any]]:
    """Install the wheelhouse offline into a fresh venv with the installer it ships.

    Args:
        wheelhouse: The wheelhouse folder.
        settings: The config's ``wheels`` section.
        out: The output folder (``install.json``, ``install.log``).
        env: The base environment (``clean_env`` is applied by the installer too).

    Returns:
        The venv's python and the installer's report.

    Raises:
        Stop: If the installer fails or its report is not ok.
    """
    base = settings["base_python"] if Path(settings["base_python"]).exists() else sys.executable
    report_path = out / "install.json"
    command = [
        base, str(wheelhouse / "install_wheelhouse.py"), "--wheelhouse", str(wheelhouse),
        "--venv", settings["venv"], "--report", str(report_path), "--log", str(out / "install.log"),
        "--imports", "torch,vllm",
    ]  # fmt: skip
    code, tail = run_command(command, out / "install.log", settings["timeout_s"], env)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {"ok": False, "problems": [f"no report; exit {code}: {tail[-5:]}"]}
    report["base_python"] = base
    if code != 0 or not report.get("ok"):
        raise Stop(f"offline install failed (exit {code}): {report.get('problems')}")
    return Path(settings["venv"]) / "bin" / "python", report


def preflight(python: Path, tp: int, out: Path, env: Mapping[str, str]) -> dict[str, Any]:
    """Check from the venv that torch sees enough working GPUs; record NCCL, P2P, and libraries.

    Args:
        python: The venv's python.
        tp: GPUs the server needs.
        out: The output folder (``preflight.json``).
        env: The engine's environment.

    Returns:
        The probe's findings.

    Raises:
        Stop: If CUDA is unavailable, fewer than ``tp`` GPUs answer, or the probe fails.
    """
    code, tail = run_command(
        [str(python), "-c", PREFLIGHT, str(tp)], out / "preflight.log", 600, env
    )
    info: dict[str, Any] = {"exit": code}
    for line in tail:
        if line.startswith("PREFLIGHT "):
            info.update(json.loads(line[len("PREFLIGHT ") :]))
    write_json(out / "preflight.json", info)
    if code != 0 or not info.get("ok"):
        raise Stop(f"GPU preflight failed (exit {code}): {info or tail[-5:]}")
    return info


# --- engine preparation: between the install and the boot -------------------------------------

# Run by the venv's python: what FlashInfer's import will find as ``flashinfer_jit_cache``.
JIT_CACHE_PROBE = """\
import importlib, importlib.metadata, importlib.util, json, re, sys
shim, providers = None, []
for dist in importlib.metadata.distributions():
    name = re.sub(r"[-_.]+", "-", dist.metadata["Name"] or "").lower()
    if name == "flashinfer-jit-cache":
        shim = dist.version
    elif name.startswith("flashinfer-jit-cache-"):
        providers.append(name)
spec = importlib.util.find_spec("flashinfer_jit_cache")
info = {"prefix": sys.prefix, "shim": shim, "providers": sorted(set(providers)),
        "found": spec is not None, "has_version": False, "locations": []}
if spec is not None:
    info["locations"] = list(spec.submodule_search_locations or [])
    info["has_version"] = hasattr(importlib.import_module("flashinfer_jit_cache"), "__version__")
print("JIT_CACHE " + json.dumps(info))
"""

# Run by the venv's python with module names as arguments: import each, stop at the first error.
ENGINE_IMPORT_CHECK = """\
import importlib, json, sys, time, traceback
report = {"ok": True, "modules": [], "python": sys.executable}
for name in sys.argv[1:]:
    began = time.monotonic()
    try:
        importlib.import_module(name)
    except BaseException as err:
        report.update(ok=False, failed=name, error=f"{type(err).__name__}: {err}",
                      traceback=traceback.format_exc().splitlines()[-40:])
        break
    report["modules"].append([name, round(time.monotonic() - began, 2)])
far = sys.modules.get("vllm.distributed.device_communicators.flashinfer_all_reduce")
if far is not None:
    report["flashinfer_all_reduce_available"] = bool(getattr(far, "fi_ar_available", False))
print("ENGINE_IMPORTS " + json.dumps(report), flush=True)
sys.exit(0 if report["ok"] else 1)
"""

QUIET_SITECUSTOMIZE = '''\
"""Empty on purpose: written by scripts/agent_spike.py (quiet_sitecustomize).

Kaggle's image installs its sitecustomize as /usr/lib/python3.11/sitecustomize.py, on the path of
every Python 3.11, venvs included. Its second statement imports ``log`` from the image's
dist-packages, which a venv does not see, so in the engine's processes it stopped there and only
printed "Error in sitecustomize ... No module named 'log'". This module, first on PYTHONPATH,
takes its place.
"""
'''


def probe_line(tail: Sequence[str], tag: str) -> dict[str, Any] | None:
    """Return the JSON a probe printed after ``tag``, from the last such line, or None."""
    for line in reversed(tail):
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1 :])
    return None


def jit_cache_state(python: Path, env: Mapping[str, str], log: Path) -> dict[str, Any]:
    """Report what ``flashinfer_jit_cache`` is in a venv, as FlashInfer's import will see it.

    Args:
        python: The venv's python.
        env: The engine's environment.
        log: The log the probe appends to.

    Returns:
        ``prefix`` (the venv), ``shim`` (the ``flashinfer-jit-cache`` version or None),
        ``providers`` (installed ``flashinfer-jit-cache-*`` distributions), ``found`` (the name
        imports), ``has_version`` (it has ``__version__``), and its ``locations``.

    Raises:
        Stop: If the probe does not run.
    """
    code, tail = run_command([str(python), "-c", JIT_CACHE_PROBE], log, 300, env)
    state = probe_line(tail, "JIT_CACHE")
    if code != 0 or state is None:
        raise Stop(f"the jit-cache probe failed (exit {code}): {tail[-5:]}")
    return state


def drop_orphan_jit_cache(python: Path, env: Mapping[str, str], log: Path) -> dict[str, Any]:
    """Remove FlashInfer jit-cache providers installed without their shim; else do nothing.

    Why: P6.3-01's attempt 2 died at the boot in every tensor-parallel worker. vLLM 0.31.0's
    ``CudaCommunicator`` always imports its FlashInfer all-reduce module, which runs ``import
    flashinfer.comm`` inside ``except ImportError``; FlashInfer 0.7.0.post1 (``jit/env.py``) then
    finds a module ``flashinfer_jit_cache`` and reads its ``__version__``. The wheelhouse holds
    the sm_89 provider ``flashinfer-jit-cache-sm89`` (package
    ``flashinfer_jit_cache.providers.sm89``) but not the shim ``flashinfer-jit-cache`` that owns
    ``flashinfer_jit_cache/__init__.py``, its ``__version__``, and the provider registry. Alone,
    the provider leaves a namespace package, and the read raises an AttributeError that the
    guard does not catch. FlashInfer reaches a provider only through the shim, so this one was
    never usable: removing it changes nothing else. ``pip uninstall`` keeps the empty namespace
    folders, which still import, so empty ones inside the venv go too.

    Args:
        python: The venv's python (pip is installed in it).
        env: The engine's environment.
        log: The log every command appends to.

    Returns:
        The record: ``state`` ("absent", "ready" with the shim, or "orphan"), ``before``,
        ``removed`` (distributions), ``folders`` (removed), and ``after``.

    Raises:
        Stop: If pip fails, or ``flashinfer_jit_cache`` still imports without ``__version__``.
    """
    before = jit_cache_state(python, env, log)
    record: dict[str, Any] = {"before": before, "removed": [], "folders": []}
    if not before["found"] or before["has_version"]:
        record["state"] = "ready" if before["found"] else "absent"
        return record
    record["state"] = "orphan"
    if before["providers"]:
        command = [
            str(python), "-m", "pip", "uninstall", "--yes", "--disable-pip-version-check",
            *before["providers"],
        ]  # fmt: skip
        code, tail = run_command(command, log, 300, env)
        if code != 0:
            raise Stop(f"pip could not remove {before['providers']} (exit {code}): {tail[-5:]}")
        record["removed"] = before["providers"]
    prefix = Path(before["prefix"]).resolve()
    for location in before["locations"]:
        folder = Path(location).resolve()
        inside = prefix in folder.parents and folder.is_dir() and not folder.is_symlink()
        if inside and all(p.is_dir() and not p.is_symlink() for p in folder.rglob("*")):
            shutil.rmtree(folder)
            record["folders"].append(str(folder))
    after = record["after"] = jit_cache_state(python, env, log)
    if after["found"] and not after["has_version"]:
        raise Stop(
            f"flashinfer_jit_cache still imports without __version__ from {after['locations']} "
            f"after removing {record['removed']} and {record['folders']}; FlashInfer's import "
            "would fail in every worker"
        )
    return record


def quiet_sitecustomize(folder: Path, env: Mapping[str, str]) -> dict[str, str]:
    """Return the engine's environment with an empty ``sitecustomize`` first on its path.

    Why: the image's ``sitecustomize`` (see ``QUIET_SITECUSTOMIZE``) fails in the venv and prints
    two lines of noise into every engine process's log. Replacing it is safe because it already
    stopped at its second statement there, so the engine loses nothing, and the folder holds only
    this file. The alternatives do not work: ``PYTHONNOUSERSITE`` does not reach the standard
    library's folder, and ``-S`` would also drop the venv's site-packages.

    Args:
        folder: Where to write ``sitecustomize.py`` (scratch).
        env: The engine's environment, without ``PYTHONPATH`` (``clean_env``).

    Returns:
        A copy of ``env`` with ``PYTHONPATH`` set to ``folder``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sitecustomize.py").write_text(QUIET_SITECUSTOMIZE, encoding="utf-8")
    return {**env, "PYTHONPATH": str(folder)}


def check_engine_imports(
    python: Path, env: Mapping[str, str], modules: Sequence[str], log: Path, timeout_s: float
) -> dict[str, Any]:
    """Import, with the venv's python and the engine's environment, what its workers import.

    Why: in attempt 2 the install's probes (``import torch``, ``import vllm``) passed and the
    workers then failed on an import vLLM makes only inside them. One process importing the same
    modules stops a broken venv in under a minute, with the error, before a boot that loads
    19.5 GB.

    Args:
        python: The venv's python.
        env: The engine's environment.
        modules: Module names, imported in this order.
        log: The log the check appends to.
        timeout_s: Seconds allowed.

    Returns:
        The report: ``ok``, each module's import seconds, and on a failure the module, the error,
        and its traceback; ``flashinfer_all_reduce_available`` once vLLM's FlashInfer all-reduce
        module imported.
    """
    code, tail = run_command(
        [str(python), "-c", ENGINE_IMPORT_CHECK, *modules], log, timeout_s, env
    )
    report: dict[str, Any] = {"exit": code, **(probe_line(tail, "ENGINE_IMPORTS") or {})}
    if "ok" not in report:
        report.update(ok=False, error=f"no report from the check: {tail[-5:]}")
    report["ok"] = bool(report["ok"]) and code == 0
    return report


def prepare_engine(
    python: Path,
    env: Mapping[str, str],
    settings: Mapping[str, Any],
    out: Path,
    scratch: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Make an installed engine venv bootable and prove it: the step between install and boot.

    In order: ``drop_orphan_jit_cache`` (attempt 2's fault), ``quiet_sitecustomize`` (the
    image's failing ``sitecustomize``), and ``check_engine_imports`` (the workers' imports, in
    the environment the engine gets). Every engine launched from this wheelhouse needs it after
    the install.

    Args:
        python: The venv's python.
        env: The engine's environment so far (``clean_env`` plus ``server.env``).
        settings: The config's ``wheels`` section (``engine_imports``,
            ``engine_check_timeout_s``).
        out: The output folder (``engine_prep.json``, ``engine_prep.log``).
        scratch: The scratch folder (the empty ``sitecustomize``).

    Returns:
        The environment to start the engine with, and the record (also ``engine_prep.json``).

    Raises:
        Stop: If a step fails, with the error; the record is written first.
    """
    log = out / "engine_prep.log"
    record: dict[str, Any] = {"python": str(python)}
    began = time.monotonic()
    try:
        record["jit_cache"] = drop_orphan_jit_cache(python, env, log)
        env = quiet_sitecustomize(scratch / "engine_site", env)
        record["pythonpath"] = env["PYTHONPATH"]
        imports = record["imports"] = check_engine_imports(
            python, env, settings["engine_imports"], log, settings["engine_check_timeout_s"]
        )
        if not imports["ok"]:
            raise Stop(
                f"engine import check failed at {imports.get('failed')} (exit "
                f"{imports['exit']}): {imports.get('error')}"
            )
    except Stop as err:
        record["error"] = str(err)
        raise
    finally:
        record["seconds"] = round(time.monotonic() - began, 1)
        write_json(out / "engine_prep.json", record)
    return dict(env), record


# --- the server ----------------------------------------------------------------------------------


def server_command(engine: str | Path, model_dir: Path, settings: Mapping[str, Any]) -> list[str]:
    """Return the ``vllm serve`` command line for the config's server section.

    Args:
        engine: The ``vllm`` executable.
        model_dir: The weights.
        settings: The config's ``server`` section.

    Returns:
        Program and arguments.
    """
    command = [
        str(engine), "serve", str(model_dir),
        "--host", settings["host"],
        "--port", str(settings["port"]),
        "--served-model-name", settings["served_model_name"],
        "--tensor-parallel-size", str(settings["tensor_parallel_size"]),
        "--kv-cache-dtype", settings["kv_cache_dtype"],
        "--max-model-len", str(settings["max_model_len"]),
        "--max-num-seqs", str(settings["max_num_seqs"]),
        "--max-num-batched-tokens", str(settings["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(settings["gpu_memory_utilization"]),
        "--attention-backend", settings["attention_backend"],
        "--reasoning-parser", settings["reasoning_parser"],
        "--generation-config", settings["generation_config"],
        "--seed", "0",
        "--disable-uvicorn-access-log",
        "--enable-prefix-caching" if settings["enable_prefix_caching"]
        else "--no-enable-prefix-caching",
    ]  # fmt: skip
    if settings["language_model_only"]:
        command.append("--language-model-only")
    return command + list(settings["extra_args"])


def http_get(base_url: str, path: str, timeout_s: float = 10.0) -> tuple[int, str]:
    """GET a path from a local server.

    Args:
        base_url: For example ``http://127.0.0.1:8000``.
        path: For example ``/health``.
        timeout_s: Socket timeout.

    Returns:
        The status (0 when the connection failed) and the body as text.
    """
    parts = urllib.parse.urlsplit(base_url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException):
        return 0, ""
    finally:
        connection.close()


EXCEPTION_LINE = re.compile(r"(?:^|\] )((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception)\b:.*)$")


def exception_lines(text: str, limit: int = 4) -> list[str]:
    """Return the first distinct exception lines of a log (``Name: message``), at most ``limit``.

    Why: a failed boot's Stop then names its cause (attempt 2's said only "exited with code 1").

    Args:
        text: The log.
        limit: Lines to keep.

    Returns:
        Each distinct line once, in log order, cut to 300 characters.
    """
    found: list[str] = []
    for line in text.splitlines():
        match = EXCEPTION_LINE.search(line.strip())
        if match and match.group(1)[:300] not in found:
            found.append(match.group(1)[:300])
            if len(found) == limit:
                break
    return found


class Server:
    """An engine process in a session of its own, logging to a file."""

    def __init__(
        self, command: list[str], env: Mapping[str, str], log: Path, base_url: str
    ) -> None:
        self.command, self.env, self.log_path, self.base_url = command, dict(env), log, base_url
        # The engine runs in the log's folder (scratch): anything it writes to its working
        # directory stays out of /kaggle/working, which Kaggle saves as the notebook's output.
        self.cwd = log.parent
        self.process: subprocess.Popen | None = None
        self.started = 0.0
        self._log = None

    def start(self) -> None:
        """Start the process; its output goes to the log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")  # closed in stop()
        self.started = time.monotonic()
        self.process = subprocess.Popen(
            self.command, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=self.env, start_new_session=True, cwd=self.cwd,
        )  # fmt: skip

    def alive(self) -> bool:
        """Return whether the process is running."""
        return self.process is not None and self.process.poll() is None

    def wait_healthy(self, timeout_s: float) -> float:
        """Wait until ``/health`` answers 200.

        Args:
            timeout_s: Seconds allowed from the start.

        Returns:
            Seconds from the start to the first healthy answer.

        Raises:
            Stop: If the process exits or the time runs out first.
        """
        while time.monotonic() - self.started < timeout_s:
            if not self.alive():
                try:
                    text = self.log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""
                raise Stop(
                    f"the engine exited with code {self.process.poll()} during boot; errors in "
                    f"its log: {exception_lines(text) or 'none'}"
                )
            if http_get(self.base_url, "/health", 5)[0] == 200:
                return time.monotonic() - self.started
            time.sleep(2)
        raise Stop(f"the engine was not healthy after {timeout_s} s")

    def stop(self, timeout_s: float) -> int | None:
        """Stop the process group: SIGTERM, then SIGKILL after ``timeout_s``.

        Args:
            timeout_s: Seconds between the two signals.

        Returns:
            The exit code, or None if it never started.
        """
        if self.process is None:
            return None
        for sig, wait_s in ((signal.SIGTERM, timeout_s), (signal.SIGKILL, 30)):
            if self.process.poll() is not None:
                break
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.process.wait(wait_s)
            except subprocess.TimeoutExpired:
                continue
        if self._log is not None:
            self._log.close()
        return self.process.poll()


BOOT_PATTERNS = {
    "kv_cache_tokens": r"KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
    "weights_load_s": r"Loading weights took ([\d.]+) seconds",
    "model_load": r"Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds",
    "compile_s": r"torch\.compile takes ([\d.]+) s in total",
    "graph_capture": r"Graph capturing finished in ([\d.]+) secs?(?:, took ([\d.-]+) GiB)?",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
    "attention_backend": r"Using ([A-Z][A-Z_0-9]+) (?:attention )?backend",
    "nccl_version": r"NCCL version ([\d.]+[^\s]*)",
    "nccl_transport": r"via (P2P|SHM|NET|NVLS|CollNet)",
    "custom_allreduce": r"(?i)(custom all-?reduce[^\n]{0,160})",
    "marlin": r"(?i)(using [^\n]{0,40}marlin[^\n]{0,80})",
}


def parse_boot_log(text: str) -> dict[str, Any]:
    """Extract the engine's boot facts from its log.

    Args:
        text: The server log.

    Returns:
        For each of ``BOOT_PATTERNS``, up to 8 matches (a group tuple or a string), plus counts of
        tracebacks, ERROR lines, and lines mentioning NaN, and ``kv_cache_tokens_first`` and
        ``max_concurrency_first`` as numbers when found.
    """
    found: dict[str, Any] = {}
    for name, pattern in BOOT_PATTERNS.items():
        matches = re.findall(pattern, text)
        if name == "nccl_transport":
            found[name] = {kind: matches.count(kind) for kind in sorted(set(matches))}
        else:
            found[name] = matches[:8]
    if found["kv_cache_tokens"]:
        found["kv_cache_tokens_first"] = int(found["kv_cache_tokens"][0].replace(",", ""))
    if found["max_concurrency"]:
        length, factor = found["max_concurrency"][0]
        found["max_concurrency_first"] = {
            "tokens": int(length.replace(",", "")),
            "x": float(factor),
        }
    found["tracebacks"] = text.count("Traceback (most recent call last)")
    found["error_lines"] = len(re.findall(r"(?m)^.*\bERROR\b.*$", text))
    found["nan_lines"] = len(re.findall(r"(?mi)^.*\bnan\b.*$", text))
    return found


METRIC = re.compile(r"^(vllm:[a-zA-Z_:]+?)(?:_total)?(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)\s*$")
KEPT_METRICS = (
    "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions",
    "prompt_tokens", "generation_tokens", "prefix_cache_queries", "prefix_cache_hits",
    "request_success",
)  # fmt: skip


def parse_metrics(text: str) -> dict[str, float]:
    """Sum the engine's Prometheus samples of interest over their labels.

    Args:
        text: The body of ``/metrics``.

    Returns:
        ``KEPT_METRICS`` present in the text, by name without ``vllm:`` and ``_total``.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC.match(line.strip())
        if not match:
            continue
        name = match.group(1)[len("vllm:") :]
        if name in KEPT_METRICS:
            try:
                values[name] = values.get(name, 0.0) + float(match.group(3))
            except ValueError:
                continue
    return values


def cache_config_info(text: str) -> dict[str, str]:
    """Return the labels of ``vllm:cache_config_info`` (block size, GPU blocks), if present.

    Args:
        text: The body of ``/metrics``.

    Returns:
        Label names to values; empty when the sample is absent.
    """
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            return dict(re.findall(r'(\w+)="([^"]*)"', line))
    return {}


class Sampler:
    """Samples ``nvidia-smi`` and the engine's ``/metrics`` in a thread, into two JSONL files."""

    GPU_FIELDS = ("index", "memory.used", "memory.total", "utilization.gpu", "power.draw")

    def __init__(self, base_url: str, out: Path, interval_s: float, gpus: bool) -> None:
        self.base_url, self.out, self.interval_s, self.gpus = base_url, out, interval_s, gpus
        self.samples: list[dict[str, Any]] = []
        self.peak_memory_mib: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._writer = JsonlWriter()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start sampling."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the thread."""
        self._stop.set()
        self._thread.join(timeout=30)

    def metrics_now(self) -> dict[str, float]:
        """Return one fresh read of the engine's counters (empty if it does not answer)."""
        status, body = http_get(self.base_url, "/metrics", 10)
        return parse_metrics(body) if status == 200 else {}

    def window_max(self, start: float, end: float, name: str) -> float | None:
        """Return the largest sampled value of a metric between two monotonic times.

        Args:
            start: Window start (``time.monotonic``).
            end: Window end.
            name: A name of ``KEPT_METRICS``.

        Returns:
            The maximum, or None without samples in the window.
        """
        with self._lock:
            values = [
                s["metrics"][name]
                for s in self.samples
                if start <= s["t"] <= end and name in s.get("metrics", {})
            ]
        return max(values) if values else None

    def _gpu(self) -> list[dict[str, str]]:
        smi = shutil.which("nvidia-smi")
        if not self.gpus or smi is None:
            return []
        try:
            result = subprocess.run(
                [smi, f"--query-gpu={','.join(self.GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=False,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return []
        rows = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == len(self.GPU_FIELDS):
                rows.append(dict(zip(self.GPU_FIELDS, parts, strict=True)))
        return rows

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            sample: dict[str, Any] = {"t": now, "metrics": self.metrics_now()}
            gpus = self._gpu()
            for row in gpus:
                try:
                    used = float(row["memory.used"])
                except ValueError:
                    continue
                key = row["index"]
                self.peak_memory_mib[key] = max(self.peak_memory_mib.get(key, 0.0), used)
            with self._lock:
                self.samples.append(sample)
            self._writer.append(self.out / "server_metrics.jsonl", sample)
            if gpus:
                self._writer.append(self.out / "gpu_samples.jsonl", {"t": now, "gpus": gpus})
            self._stop.wait(self.interval_s)


# --- the streaming client ------------------------------------------------------------------------


class Inflight:
    """The open connections of one wave or phase, cut together when its time is up.

    Only the thread that reads a connection ever closes it. ``http.client`` objects are not
    thread-safe: in ``P6.3-01``'s attempt 3 a close from the cancelling thread, while a reader was
    inside ``readline``, left ``http.client`` calling ``close`` on its own ``fp`` after the other
    thread had set it to None (``AttributeError``), which ended the run. ``cancel`` therefore only
    shuts the sockets down: every blocked read then ends, and each reader closes its own
    connection.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: set[http.client.HTTPConnection] = set()
        self.cancelled = threading.Event()

    def add(self, connection: http.client.HTTPConnection) -> None:
        """Register a connected connection; one registered after ``cancel`` is shut down at once."""
        with self._lock:
            self._connections.add(connection)
        if self.cancelled.is_set():
            shutdown_socket(connection)

    def discard(self, connection: http.client.HTTPConnection) -> None:
        """Forget a finished connection."""
        with self._lock:
            self._connections.discard(connection)

    def cancel(self) -> None:
        """Shut every open socket down; vLLM aborts a request whose client disconnects."""
        self.cancelled.set()
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            shutdown_socket(connection)


def shutdown_socket(connection: http.client.HTTPConnection) -> None:
    """Shut a connection's socket down in both directions, from any thread, without closing it.

    A read blocked on the socket returns end of stream and a write fails, so the reading thread
    ends its request and closes the connection itself.

    Args:
        connection: The connection; nothing happens when it has no socket (yet, or any more).
    """
    sock = connection.sock
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # already closed by its reader, or never connected


def empty_record(t_send: float) -> dict[str, Any]:
    """Return a ``stream_chat`` record with nothing received yet.

    Args:
        t_send: The send time (``time.monotonic()``).

    Returns:
        The record, ``error`` None.
    """
    return {
        "t_send": t_send, "t_first": None, "t_last": None, "chunks": [], "token_ids": [],
        "usage": None, "finish_reason": None, "error": None, "token_ids_returned": False,
        "content": "", "reasoning": "",
    }  # fmt: skip


def error_text(err: BaseException, inflight: Inflight | None) -> str:
    """Name a request's failure: ``"cancelled"`` after its wave was cut, else the exception.

    Args:
        err: The exception.
        inflight: The registry the request belonged to, if any.

    Returns:
        The text recorded under ``error``.
    """
    if inflight is not None and inflight.cancelled.is_set():
        return "cancelled"
    return f"{type(err).__name__}: {err}"[:500]


def stream_chat(
    base_url: str, payload: Mapping[str, Any], timeout_s: float, inflight: Inflight | None = None
) -> dict[str, Any]:
    """Send one streaming chat completion and time every chunk; never raise.

    Every failure becomes the request's ``error``: an HTTP status, a refused or broken
    connection, a malformed event, ``http.client``'s own errors (attempt 3 of ``P6.3-01`` ended on
    an ``AttributeError`` from it), or a stream that ends without ``[DONE]``. After ``inflight``
    was cancelled, the error is ``"cancelled"``, which the kill criteria do not count.

    Args:
        base_url: The server.
        payload: The request body without ``stream`` and ``stream_options``.
        timeout_s: Socket timeout (a silent server for this long fails the request).
        inflight: Registry that can cancel the request.

    Returns:
        ``t_send``, ``t_first`` (first chunk with a generated token), ``t_last`` (monotonic),
        ``chunks`` as ``(time, tokens)``, ``token_ids``, ``content``, ``reasoning``, ``usage``,
        ``finish_reason``, ``token_ids_returned``, and ``error`` (None on success); what arrived
        before a failure is kept.
    """
    parts = urllib.parse.urlsplit(base_url)
    body = json.dumps(
        {**payload, "stream": True, "stream_options": {"include_usage": True}}
    ).encode("utf-8")
    record = empty_record(time.monotonic())
    content: list[str] = []
    reasoning: list[str] = []
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
    try:
        # Connected before it is registered, so a cancel always finds a socket to shut down.
        connection.connect()
        if inflight is not None:
            inflight.add(connection)
        connection.request(
            "POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(2000).decode("utf-8", errors="replace")
            record["error"] = f"HTTP {response.status}: {detail}"
            return record
        finished = False
        while True:
            # http.client ends a chunked stream cut mid-way as a clean end of file, hence the
            # check for [DONE] below.
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if data == b"[DONE]":
                finished = True
                break
            event = json.loads(data)
            now = time.monotonic()
            if event.get("usage"):
                record["usage"] = event["usage"]
            for choice in event.get("choices") or []:
                ids = choice.get("token_ids")
                delta = choice.get("delta") or {}
                text = delta.get("content")
                thought = delta.get("reasoning") or delta.get("reasoning_content")
                if ids is not None:
                    record["token_ids_returned"] = True
                    record["token_ids"].extend(ids)
                count = len(ids) if ids is not None else int(bool(text or thought))
                if text:
                    content.append(text)
                if thought:
                    reasoning.append(thought)
                if count:
                    if record["t_first"] is None:
                        record["t_first"] = now
                    record["chunks"].append((now, count))
                    record["t_last"] = now
                if choice.get("finish_reason"):
                    record["finish_reason"] = choice["finish_reason"]
        if not finished:
            record["error"] = error_text(EOFError("the stream ended before [DONE]"), inflight)
    except Exception as err:  # any failure, http.client's own included, is this request's error
        record["error"] = error_text(err, inflight)
    finally:
        try:
            connection.close()
        except Exception:  # a half-read chunked response can fail to close; nothing is lost
            pass
        if inflight is not None:
            inflight.discard(connection)
        record["content"] = "".join(content)
        record["reasoning"] = "".join(reasoning)
    return record


def token_stats(
    token_ids: Sequence[int],
    special_min_id: int,
    max_run: int,
    text: str = "",
    eos_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Measure repetition in one output: the longest run of one token, special tokens excluded.

    Why: NaN logits make a sampler emit one token over and over (often id 0); a run this long is
    the cheap signal of a broken engine, the kill criterion of the brief. Only the tokens before
    the first end-of-sequence id count: with ``ignore_eos`` a model that has finished keeps
    generating, and what follows its end may legitimately repeat.

    Args:
        token_ids: The generated tokens.
        special_min_id: Ids at or above this are special and break runs.
        max_run: Run length that marks the output degenerate.
        text: The generated text, used for a character run when no ids came back.
        eos_ids: End-of-sequence ids; the count stops at the first one.

    Returns:
        ``max_run``, ``max_run_token``, ``token0_count``, ``degenerate``, ``source``, and
        ``eos_at`` (the index of the first end-of-sequence id, or None).
    """
    stops = set(eos_ids)
    eos_at = next((i for i, token in enumerate(token_ids) if token in stops), None)
    if eos_at is not None:
        token_ids = token_ids[:eos_at]
    best, best_token, run, previous = 0, None, 0, None
    for token in token_ids:
        if token >= special_min_id:
            run, previous = 0, None
            continue
        run = run + 1 if token == previous else 1
        previous = token
        if run > best:
            best, best_token = run, token
    source = "token_ids"
    if not token_ids and text:
        source = "characters"
        for match in re.finditer(r"(.)\1*", text, re.DOTALL):
            length = len(match.group(0))
            if length > best:
                best, best_token = length, match.group(1)
    return {
        "max_run": best,
        "max_run_token": best_token,
        "token0_count": sum(1 for t in token_ids if t == 0),
        "degenerate": best >= max_run,
        "source": source,
        "eos_at": eos_at,
    }


def request_summary(record: Mapping[str, Any], degenerate: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one ``stream_chat`` record to the numbers written per request.

    Args:
        record: The record.
        degenerate: The config's ``degenerate`` section.

    Returns:
        Prompt and completion tokens, TTFT, end-to-end seconds, per-request decode rate,
        finish reason, repetition stats, and the error.
    """
    usage = record.get("usage") or {}
    counted = sum(n for _, n in record["chunks"])
    completion = usage.get("completion_tokens", counted)
    first, last, sent = record["t_first"], record["t_last"], record["t_send"]
    rate = None
    if first is not None and last is not None and last > first:
        rate = (counted - record["chunks"][0][1]) / (last - first)
    stats = token_stats(
        record["token_ids"], degenerate["special_min_id"], degenerate["max_run"],
        record.get("reasoning", "") + record.get("content", ""), degenerate["eos_ids"],
    )  # fmt: skip
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "ttft_s": None if first is None else round(first - sent, 3),
        "e2e_s": None if last is None else round(last - sent, 3),
        "decode_tps": None if rate is None else round(rate, 2),
        "finish_reason": record["finish_reason"],
        "error": record["error"],
        **stats,
    }


# --- synthetic ARC-like prompts ------------------------------------------------------------------


def render_grid(grid: Sequence[Sequence[int]]) -> str:
    """Render a grid as rows of digits, one row per line (one token per cell in Qwen's tokenizer).

    Args:
        grid: Rows of colors 0 to 9.

    Returns:
        The text, without a trailing newline.
    """
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def synthetic_grid(rng: random.Random, rows: int, cols: int) -> list[list[int]]:
    """Return an ARC-like grid: a background color with a few colored rectangles.

    Args:
        rng: The random source.
        rows: Height.
        cols: Width.

    Returns:
        The grid.
    """
    background = rng.choice([0, 0, 0, 5, 8])
    grid = [[background] * cols for _ in range(rows)]
    for _ in range(rng.randint(2, 6)):
        height, width = rng.randint(1, max(1, rows // 3)), rng.randint(1, max(1, cols // 3))
        top, left = rng.randint(0, rows - height), rng.randint(0, cols - width)
        color = rng.choice([c for c in range(10) if c != background])
        for r in range(top, top + height):
            for c in range(left, left + width):
                grid[r][c] = color
    return grid


SYNTHETIC_SYSTEM = (
    "You study ARC puzzles. Each puzzle shows example pairs of an input grid and an output grid, "
    "written as rows of digits (colors 0 to 9); one rule maps every input to its output."
)


def synthetic_messages(rng: random.Random, prompt_tokens: int) -> list[dict[str, str]]:
    """Build a chat of about ``prompt_tokens`` tokens of ARC-like example pairs and a test input.

    Why: digit rows tokenize at one token per cell plus one per row (Qwen's tokenizer splits
    digits), so the text's grid characters estimate its tokens; the server's ``usage`` reports
    the exact count, which the cell records.

    Args:
        rng: The random source (one per request, so prompts share no prefix past the system).
        prompt_tokens: The target.

    Returns:
        A system and a user message.
    """
    parts: list[str] = []
    budget = prompt_tokens - 60  # the system message and the chat template
    example = 0
    while budget > 0:
        rows, cols = rng.randint(6, 30), rng.randint(6, 30)
        grid = synthetic_grid(rng, rows, cols)
        output = [row[::-1] for row in grid]
        example += 1
        block = (
            f"Example {example}\nInput ({rows} rows, {cols} columns):\n{render_grid(grid)}\n"
            f"Output ({rows} rows, {cols} columns):\n{render_grid(output)}\n"
        )
        cost = 2 * rows * (cols + 1) + 24
        if cost > budget and example > 1:
            # Fill the rest with one smaller pair so the prompt lands near the target.
            side = max(2, int((budget / 2) ** 0.5) - 1)
            grid = synthetic_grid(rng, side, side)
            block = (
                f"Example {example}\nInput ({side} rows, {side} columns):\n{render_grid(grid)}\n"
                f"Output ({side} rows, {side} columns):\n{render_grid(grid)}\n"
            )
            cost = budget
        parts.append(block)
        budget -= cost
    test = synthetic_grid(rng, 12, 12)
    parts.append(f"Test input (12 rows, 12 columns):\n{render_grid(test)}\n")
    parts.append("Find the rule and write the output grid for the test input.")
    return [
        {"role": "system", "content": SYNTHETIC_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


# --- the throughput phase ------------------------------------------------------------------------


def steady_rate(records: Sequence[Mapping[str, Any]], min_window_s: float = 5.0) -> dict[str, Any]:
    """Return the aggregate decode rate while every request of a wave was decoding.

    With :math:`t_0` the last first token and :math:`t_1` the first last token of the wave, the
    rate is the tokens of all chunks in :math:`(t_0, t_1]` over :math:`t_1 - t_0`: no prefill and
    no tail with fewer sequences.

    Args:
        records: ``stream_chat`` records of one wave.
        min_window_s: Shorter windows give no rate.

    Returns:
        ``steady_tps`` (or None), ``window_s``, and ``window_tokens``.
    """
    started = [r["t_first"] for r in records]
    finished = [r["t_last"] for r in records]
    if not records or any(t is None for t in started + finished):
        return {"steady_tps": None, "window_s": None, "window_tokens": None}
    start, end = max(started), min(finished)
    if end - start < min_window_s:
        return {"steady_tps": None, "window_s": round(end - start, 3), "window_tokens": None}
    tokens = sum(n for r in records for t, n in r["chunks"] if start < t <= end)
    return {
        "steady_tps": round(tokens / (end - start), 2),
        "window_s": round(end - start, 3),
        "window_tokens": tokens,
    }


def run_wave(
    base_url: str, payloads: Sequence[Mapping[str, Any]], timeout_s: float
) -> tuple[list[dict[str, Any]], bool]:
    """Send every payload at once and wait for all, cancelling what is open at the timeout.

    Args:
        base_url: The server.
        payloads: Request bodies.
        timeout_s: Seconds for the whole wave.

    Returns:
        The records in payload order and whether the timeout cut the wave.
    """
    inflight = Inflight()
    began = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        futures = [pool.submit(stream_chat, base_url, p, timeout_s, inflight) for p in payloads]
        _, pending = concurrent.futures.wait(futures, timeout=timeout_s)
        cut = bool(pending)
        if pending:
            inflight.cancel()
        records = []
        for future in futures:
            try:
                records.append(future.result())
            except Exception as err:  # stream_chat records its own errors; this is a backstop
                records.append({**empty_record(began), "error": error_text(err, inflight)})
    return records, cut


def cell_summary(
    name: str,
    concurrency: int,
    records: Sequence[dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    cut: bool,
    output_tokens: int,
) -> dict[str, Any]:
    """Summarize one throughput cell.

    Args:
        name: The context's name.
        concurrency: Requests in the wave.
        records: Their ``stream_chat`` records.
        rows: Their ``request_summary`` rows.
        cut: Whether the cell's timeout ended it.
        output_tokens: Tokens each request was asked for.

    Returns:
        The cell's numbers: wall, aggregate and steady output tokens per second, TTFT
        percentiles, prompt and mean-context sizes, errors, degenerate count.
    """
    sent = min(r["t_send"] for r in records)
    ends = [r["t_last"] for r in records if r["t_last"] is not None]
    wall = (max(ends) - sent) if ends else None
    produced = sum(row["completion_tokens"] or 0 for row in rows)
    ttfts = [row["ttft_s"] for row in rows if row["ttft_s"] is not None]
    prompts = [row["prompt_tokens"] for row in rows if row["prompt_tokens"]]
    rates = [row["decode_tps"] for row in rows if row["decode_tps"] is not None]
    firsts = [r["t_first"] for r in records if r["t_first"] is not None]
    mean_prompt = sum(prompts) / len(prompts) if prompts else None
    return {
        "context": name,
        "concurrency": concurrency,
        "requests": len(rows),
        "complete": not cut and all(row["error"] is None for row in rows),
        "cut": cut,
        "errors": sum(1 for row in rows if row["error"] is not None),
        "degenerate": sum(1 for row in rows if row["degenerate"]),
        "wall_s": None if wall is None else round(wall, 2),
        "output_tokens": produced,
        "agg_tps": None if not wall else round(produced / wall, 2),
        **steady_rate(records),
        "ttft_p50_s": None if not ttfts else round(percentile(ttfts, 50), 3),
        "ttft_p90_s": None if not ttfts else round(percentile(ttfts, 90), 3),
        "ttft_max_s": max(ttfts) if ttfts else None,
        "prefill_span_s": None if not firsts else round(max(firsts) - sent, 2),
        "prompt_tokens_mean": None if mean_prompt is None else round(mean_prompt, 1),
        "prompt_tokens_total": sum(prompts),
        "mean_context": None if mean_prompt is None else round(mean_prompt + output_tokens / 2, 1),
        "per_request_tps_p50": None if not rates else round(percentile(rates, 50), 2),
        "finish_reasons": sorted({str(row["finish_reason"]) for row in rows}),
    }


# --- the probe -----------------------------------------------------------------------------------


PROBE_SYSTEM = """\
You solve ARC puzzles by writing Python.

A puzzle shows example pairs of an input grid and an output grid; one hidden rule turns every \
input into its output. Grids are written as rows of digits, each digit a color from 0 to 9.

Write a function transform(grid) that applies the rule: it takes one input grid as a list of \
lists of ints and returns the output grid as a list of lists of ints. It may use the Python \
standard library and numpy (import numpy as np). It must work for every example input and for \
the test input, so never copy example outputs into the code.

Answer with exactly one Python code block that defines transform."""


def task_text(task: Mapping[str, Any]) -> str:
    """Render a task for the model: every example pair, then the test inputs, as digit rows.

    Args:
        task: ``{"train": [{"input", "output"}], "test": [{"input"}]}``; test outputs, if
            present, are never read.

    Returns:
        The text.
    """

    def shape(grid: Sequence[Sequence[int]]) -> str:
        return f"{len(grid)} rows, {len(grid[0]) if grid else 0} columns"

    parts = []
    for index, pair in enumerate(task["train"], start=1):
        parts.append(
            f"Example {index}\nInput ({shape(pair['input'])}):\n{render_grid(pair['input'])}\n"
            f"Output ({shape(pair['output'])}):\n{render_grid(pair['output'])}\n"
        )
    for index, pair in enumerate(task["test"], start=1):
        label = "Test input" if len(task["test"]) == 1 else f"Test input {index}"
        parts.append(f"{label} ({shape(pair['input'])}):\n{render_grid(pair['input'])}\n")
    return "\n".join(parts)


def select_tasks(
    challenges: Mapping[str, Any], n_tasks: int, seed: int, max_chars: int
) -> tuple[list[str], int]:
    """Draw the probe's tasks: a seeded sample of the tasks with one test input and short text.

    Args:
        challenges: The training challenges.
        n_tasks: Tasks to draw.
        seed: The sample's seed.
        max_chars: Longest ``task_text`` allowed.

    Returns:
        The task ids, sorted, and the number of eligible tasks.

    Raises:
        Stop: If fewer than ``n_tasks`` tasks are eligible.
    """
    eligible = [
        task_id
        for task_id in sorted(challenges)
        if len(challenges[task_id]["test"]) == 1
        and len(task_text(challenges[task_id])) <= max_chars
    ]
    if len(eligible) < n_tasks:
        raise Stop(f"{len(eligible)} eligible probe tasks, fewer than {n_tasks}")
    return sorted(random.Random(seed).sample(eligible, n_tasks)), len(eligible)


def extract_program(text: str) -> str | None:
    """Return the last fenced code block that defines ``transform``, or None.

    Args:
        text: The answer (the visible content, not the reasoning).

    Returns:
        The program source.
    """
    blocks = [block for block in FENCE.findall(text or "") if "def transform" in block]
    return blocks[-1] if blocks else None


CHECKER = """\
import copy, json, resource, sys
limit = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
cpu = int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
payload = json.loads(sys.stdin.read())
scope = {"__name__": "candidate"}
try:
    exec(compile(payload["source"], "<transform>", "exec"), scope)
    transform = scope["transform"]
except BaseException as err:
    print("CHECK " + json.dumps({"load_error": (type(err).__name__ + ": " + str(err))[:400]}))
    sys.exit(0)

def valid(grid):
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not isinstance(grid, (list, tuple)) or not 1 <= len(grid) <= 30:
        return None
    rows = []
    for row in grid:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 30:
            return None
        cells = []
        for cell in row:
            if isinstance(cell, bool) or int(cell) != cell or not 0 <= int(cell) <= 9:
                return None
            cells.append(int(cell))
        rows.append(cells)
    if len({len(r) for r in rows}) != 1:
        return None
    return rows

results = []
for grid in payload["inputs"]:
    try:
        output = valid(transform(copy.deepcopy(grid)))
        results.append({"output": output} if output is not None else {"error": "not a grid"})
    except BaseException as err:
        results.append({"error": (type(err).__name__ + ": " + str(err))[:300]})
print("CHECK " + json.dumps({"results": results}))
"""


def check_program(
    source: str, inputs: Sequence[Any], python: str, timeout_s: float, memory_mb: int
) -> dict[str, Any]:
    """Run a program's ``transform`` on every input in a fresh, limited subprocess.

    Why: a minimal stand-in for P6.1's verifier: model-written code never runs in this process;
    the child has no task state, an address-space and CPU limit, and a wall timeout that kills its
    whole process group.

    Args:
        source: The program.
        inputs: The grids, demonstrations first.
        python: The interpreter (the image's, which has numpy).
        timeout_s: Wall seconds for all inputs.
        memory_mb: Address-space limit.

    Returns:
        ``results`` (one ``output`` or ``error`` per input) or ``load_error``, ``timeout``, and
        ``seconds``.
    """
    start = time.monotonic()
    command = [python, "-I", "-c", CHECKER, str(memory_mb * 1024 * 1024), str(int(timeout_s) + 1)]
    # One BLAS thread: on a 48-CPU host, per-thread buffers would exhaust the address-space limit
    # at the numpy import.
    env = clean_env(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, cwd=tempfile_dir(), env=env,
    )  # fmt: skip
    try:
        stdout, _ = process.communicate(json.dumps({"source": source, "inputs": inputs}), timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        return {"timeout": True, "seconds": round(time.monotonic() - start, 2)}
    result: dict[str, Any] = {"timeout": False, "seconds": round(time.monotonic() - start, 2)}
    for line in stdout.splitlines():
        if line.startswith("CHECK "):
            result.update(json.loads(line[len("CHECK ") :]))
    if "results" not in result and "load_error" not in result:
        result["load_error"] = f"the checker exited {process.returncode} without a result"
    return result


def tempfile_dir() -> str:
    """Return a scratch folder for checker processes (their working directory)."""
    import tempfile

    folder = os.path.join(tempfile.gettempdir(), "agent-spike-checker")
    os.makedirs(folder, exist_ok=True)
    return folder


def judge(
    check: Mapping[str, Any] | None, task: Mapping[str, Any], solution: Sequence[Any] | None
) -> dict[str, Any]:
    """Score one checked program: demonstrations reproduced, verified, and correct on test.

    Args:
        check: ``check_program``'s result, or None when no program was found.
        task: The task.
        solution: The test outputs (training solutions), or None.

    Returns:
        ``passed``, ``total``, ``verified``, ``test_correct``, and ``outputs``.
    """
    total = len(task["train"])
    if not check or "results" not in check:
        return {"passed": 0, "total": total, "verified": False, "test_correct": False}
    results = check["results"]
    train, test = results[:total], results[total:]
    passed = sum(1 for r, pair in zip(train, task["train"], strict=True)
                 if r.get("output") == pair["output"])  # fmt: skip
    verified = passed == total and all("output" in r for r in test)
    correct = bool(solution) and all(
        r.get("output") == expected for r, expected in zip(test, solution, strict=True)
    )
    return {"passed": passed, "total": total, "verified": verified, "test_correct": correct}


def feedback_text(
    check: Mapping[str, Any] | None, task: Mapping[str, Any], found: bool, limit: int = 6000
) -> str:
    """Write the checker's report for the feedback round.

    Args:
        check: ``check_program``'s result, or None.
        task: The task.
        found: Whether the answer had a program at all.
        limit: Longest report in characters.

    Returns:
        The user message of the feedback round.
    """
    closing = "Fix the rule and answer with exactly one Python code block that defines transform."
    if not found:
        return "Your answer had no Python code block defining transform. " + closing
    if check is None or check.get("timeout"):
        return "Your transform did not finish within the time limit on the examples. " + closing
    if "load_error" in check:
        return f"Your program failed before transform ran: {check['load_error']}. " + closing
    lines = ["Your transform was run on the examples:"]
    shown = False
    # The results hold the test inputs after the demonstrations; zip stops at the last pair.
    pairs = zip(check["results"], task["train"], strict=False)
    for index, (result, pair) in enumerate(pairs, start=1):
        expected = pair["output"]
        if "error" in result:
            lines.append(f"Example {index}: error: {result['error']}")
            continue
        output = result["output"]
        if output == expected:
            lines.append(f"Example {index}: correct.")
            continue
        size = f"{len(output)}x{len(output[0])}"
        if (len(output), len(output[0])) != (len(expected), len(expected[0])):
            lines.append(
                f"Example {index}: wrong: your output is {size}, expected "
                f"{len(expected)}x{len(expected[0])}."
            )
        else:
            cells = zip(output, expected, strict=True)
            right = sum(a == b for got, want in cells for a, b in zip(got, want, strict=True))
            lines.append(
                f"Example {index}: wrong: {right} of {len(output) * len(output[0])} cells."
            )
        if not shown:
            lines.append(f"Expected:\n{render_grid(expected)}\nYour output:\n{render_grid(output)}")
            shown = True
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit] + "\n(report truncated)"
    return text + "\n" + closing


def solve_task(
    task_id: str,
    task: Mapping[str, Any],
    solution: Sequence[Any] | None,
    settings: Mapping[str, Any],
    context: Mapping[str, Any],
    inflight: Inflight,
) -> dict[str, Any]:
    """Run one probe task: a program request, the check, and the feedback rounds.

    Args:
        task_id: The task id.
        task: The task.
        solution: Its test outputs.
        settings: The config's ``probe`` section.
        context: ``base_url``, ``model``, ``seed``, ``python``, ``degenerate``, ``timing``,
            ``writer``, ``timeout_s``.
        inflight: The phase's registry.

    Returns:
        The task's record: each round's tokens, timing, check, and verdict, then the totals.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PROBE_SYSTEM},
        {"role": "user", "content": task_text(task)},
    ]
    inputs = [pair["input"] for pair in task["train"]] + [pair["input"] for pair in task["test"]]
    rounds = []
    verdict: dict[str, Any] = {"verified": False, "test_correct": False}
    for number in range(1, settings["rounds"] + 1):
        payload = {
            "model": context["model"],
            "messages": messages,
            "max_tokens": settings["max_tokens"],
            "seed": context["seed"] + number,
            "return_token_ids": True,
        }
        if settings["reasoning_effort"]:
            payload["reasoning_effort"] = settings["reasoning_effort"]
        record = stream_chat(context["base_url"], payload, context["timeout_s"], inflight)
        row = request_summary(record, context["degenerate"])
        context["writer"].append(
            context["timing"],
            {"puzzle_id": task_id, "op": "engine", "seconds": row["e2e_s"],
             "tokens": row["completion_tokens"]},
        )  # fmt: skip
        program = extract_program(record["content"])
        check = None
        if program is not None and record["error"] is None:
            check = check_program(
                program, inputs, context["python"], settings["checker_timeout_s"],
                settings["checker_memory_mb"],
            )  # fmt: skip
        verdict = judge(check, task, solution)
        rounds.append(
            {
                "round": number, **row, "program_found": program is not None,
                "program": program, "reasoning_chars": len(record["reasoning"]),
                "content_chars": len(record["content"]),
                "check": None if check is None else {
                    k: check[k] for k in ("timeout", "seconds", "load_error") if k in check
                } | {"errors": sum(1 for r in check.get("results", []) if "error" in r)},
                **verdict,
            }
        )  # fmt: skip
        if verdict["verified"] or record["error"] is not None or number == settings["rounds"]:
            break
        messages = messages + [
            {"role": "assistant", "content": record["content"], "reasoning": record["reasoning"]},
            {"role": "user", "content": feedback_text(check, task, program is not None)},
        ]
    return {
        "task_id": task_id,
        "rounds": rounds,
        "rounds_used": len(rounds),
        "verified": verdict["verified"],
        "verified_round": next((r["round"] for r in rounds if r["verified"]), None),
        "test_correct": verdict["verified"] and verdict["test_correct"],
        "any_test_correct": any(r["test_correct"] for r in rounds),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rounds),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rounds),
        "seconds": round(sum(r["e2e_s"] or 0 for r in rounds), 2),
        "errors": sum(1 for r in rounds if r["error"] and r["error"] != "cancelled"),
        "cancelled": sum(1 for r in rounds if r["error"] == "cancelled"),
        "degenerate": sum(1 for r in rounds if r["degenerate"]),
        "length_stops": sum(1 for r in rounds if r["finish_reason"] == "length"),
    }


def task_outcome(future: concurrent.futures.Future, task_id: str) -> dict[str, Any]:
    """Return a probe task's record, or an errored one when its thread raised.

    Args:
        future: The task's future, finished.
        task_id: Its task id.

    Returns:
        ``solve_task``'s record; on an exception, the same keys with no rounds and one error.
    """
    try:
        return future.result()
    except Exception as err:  # a task's failure is recorded with it, never ends the phase
        return {
            "task_id": task_id, "rounds": [], "rounds_used": 0, "verified": False,
            "verified_round": None, "test_correct": False, "any_test_correct": False,
            "completion_tokens": 0, "prompt_tokens": 0, "seconds": 0.0, "errors": 1,
            "cancelled": 0, "degenerate": 0, "length_stops": 0,
            "error": f"{type(err).__name__}: {err}"[:500],
        }  # fmt: skip


# --- the run -------------------------------------------------------------------------------------


class Run:
    """State of one run: config, folders, deadline, the server, and everything measured."""

    def __init__(self, config: dict[str, Any], args: argparse.Namespace) -> None:
        self.config, self.args = config, args
        self.start = time.monotonic()
        self.deadline = self.start + config["max_wall_s"]
        self.out = Path(config["out_dir"])
        self.scratch = Path(config["scratch_dir"])
        self.writer = JsonlWriter()
        self.metrics: dict[str, Any] = {"run_id": config["run_id"], "smoke": args.smoke}
        self.stages: dict[str, Any] = {}
        self.problems: list[str] = []
        self.server: Server | None = None
        self.sampler: Sampler | None = None
        # P6.1's REPL factory and verifier pool, built by the harness preflight and reused by the
        # agent phase; finish() closes the pool whatever happened.
        self.sandbox: tuple[Any, Any] | None = None
        self.log_handlers: list[logging.Handler] = []
        self.exit_code = EXIT_OK
        self.base_url = f"http://{config['server']['host']}:{config['server']['port']}"
        self.done = load_resume(args.resume_from) if args.resume_from else {}

    def left(self) -> float:
        """Seconds left before the run's deadline."""
        return self.deadline - time.monotonic()

    def stage(self, name: str, seconds: float, **extra: Any) -> None:
        """Record a stage's wall time and facts."""
        self.stages[name] = {"seconds": round(seconds, 2), **extra}
        print(f"== {name}: {seconds:.1f} s {json.dumps(extra, default=str)[:400]}", flush=True)


def load_resume(folder: str) -> dict[str, Any]:
    """Read a previous run's completed cells and probe tasks, which this run then skips.

    Args:
        folder: A previous ``out_dir`` (for example an attached earlier output).

    Returns:
        ``cells`` (key ``<context>/<concurrency>`` to summary) and ``tasks`` (id to record).
    """
    path = Path(folder)
    done: dict[str, Any] = {"cells": {}, "tasks": {}}
    try:
        for cell in json.loads((path / "throughput_cells.json").read_text(encoding="utf-8")):
            if cell.get("complete"):
                done["cells"][f"{cell['context']}/{cell['concurrency']}"] = cell
    except (OSError, ValueError):
        pass
    try:
        for line in (path / "probe.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done["tasks"][record["task_id"]] = record
    except (OSError, ValueError, KeyError):
        pass
    return done


def kill_check(run: Run, where: str, rows: Sequence[Mapping[str, Any]], key: str) -> None:
    """Apply the kill criteria to a group of requests.

    Args:
        run: The run.
        where: The cell or phase, for the message.
        rows: Rows with ``error`` and ``degenerate`` (or counts under ``key``).
        key: ``request`` for request rows, ``task`` for probe records.

    Raises:
        Kill: If the share of failed or degenerate requests passes its limit, or the server died.
    """
    kill = run.config["kill"]
    if not rows:
        return
    if key == "task":
        failed = sum(1 for r in rows if r["errors"]) / len(rows)
        broken = sum(1 for r in rows if r["degenerate"]) / len(rows)
    else:
        failed = sum(1 for r in rows if r["error"] and r["error"] != "cancelled") / len(rows)
        broken = sum(1 for r in rows if r["degenerate"]) / len(rows)
    if run.server is not None and not run.server.alive():
        raise Kill(f"{where}: the engine process died (exit {run.server.process.poll()})")
    if failed > kill["error_share"]:
        raise Kill(f"{where}: {failed:.0%} of requests failed, over {kill['error_share']:.0%}")
    if broken > kill["degenerate_share"]:
        raise Kill(f"{where}: {broken:.0%} degenerate outputs, over {kill['degenerate_share']:.0%}")


def throughput_phase(run: Run, phase: str) -> None:
    """Run one context's cells, smallest concurrency first, within the phase's budget.

    Args:
        run: The run.
        phase: ``throughput-<context>``.
    """
    settings = run.config["throughput"]
    name = phase[len("throughput-") :]
    context = settings["contexts"][name]
    budget = context["budget_s"]
    phase_end = run.deadline if budget is None else min(run.deadline, time.monotonic() + budget)
    cells = run.metrics.setdefault("throughput", [])
    served = run.config["server"]["served_model_name"]
    todo = [c for c in settings["concurrency"] if f"{name}/{c}" not in run.done.get("cells", {})]
    if settings["warmup_tokens"] and todo:
        # One unmeasured request at this prompt length first: kernels tuned per shape on first
        # use (Triton) would otherwise land in the first cell. Its cold TTFT is recorded.
        payload = {
            "model": served,
            "messages": synthetic_messages(random.Random(-1 - len(name)), context["prompt_tokens"]),
            "max_tokens": settings["warmup_tokens"],
            "ignore_eos": True,
            "return_token_ids": True,
        }
        record = stream_chat(run.base_url, payload, context["cell_timeout_s"])
        row = request_summary(record, run.config["degenerate"])
        run.metrics.setdefault("warmup", {})[name] = row
        run.writer.append(
            run.out / "timing.jsonl",
            {"puzzle_id": f"warmup-{name}", "op": "engine", "seconds": row["e2e_s"],
             "tokens": row["completion_tokens"]},
        )  # fmt: skip
        print(f"warmup {name}: TTFT {row['ttft_s']} s, error {row['error']}", flush=True)
        kill_check(run, f"warmup {name}", [row], "request")
    for concurrency in settings["concurrency"]:
        key = f"{name}/{concurrency}"
        if key in run.done.get("cells", {}):
            cells.append({**run.done["cells"][key], "resumed": True})
            continue
        left = phase_end - time.monotonic()
        if left < settings["min_cell_s"]:
            cells.append({"context": name, "concurrency": concurrency, "skipped": "budget"})
            continue
        seed = run.config["seed"] * 1_000_003 + concurrency * 1009 + context["prompt_tokens"]
        payloads = []
        for index in range(concurrency):
            rng = random.Random(seed + index)
            payload = {
                "model": served,
                "messages": synthetic_messages(rng, context["prompt_tokens"]),
                "max_tokens": settings["output_tokens"],
                "ignore_eos": True,
                "seed": seed + index,
                "return_token_ids": True,
            }
            if settings["reasoning_effort"]:
                payload["reasoning_effort"] = settings["reasoning_effort"]
            payloads.append(payload)
        before = run.sampler.metrics_now() if run.sampler else {}
        timeout = min(context["cell_timeout_s"], left)
        print(f"cell {key}: {concurrency} requests, timeout {timeout:.0f} s", flush=True)
        records, cut = run_wave(run.base_url, payloads, timeout)
        after = run.sampler.metrics_now() if run.sampler else {}
        rows = [request_summary(r, run.config["degenerate"]) for r in records]
        for index, row in enumerate(rows):
            run.writer.append(run.out / "throughput.jsonl", {"cell": key, "index": index, **row})
            run.writer.append(
                run.out / "timing.jsonl",
                {"puzzle_id": f"synthetic-{name}-c{concurrency}-{index}", "op": "engine",
                 "seconds": row["e2e_s"], "tokens": row["completion_tokens"]},
            )  # fmt: skip
        summary = cell_summary(name, concurrency, records, rows, cut, settings["output_tokens"])
        start = min(r["t_send"] for r in records)
        end = max([r["t_last"] for r in records if r["t_last"]] or [time.monotonic()])
        summary["server"] = {
            key: after.get(key, 0.0) - before.get(key, 0.0)
            for key in ("generation_tokens", "prompt_tokens", "num_preemptions",
                        "prefix_cache_queries", "prefix_cache_hits")
        }  # fmt: skip
        if run.sampler:
            for metric in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
                summary["server"][f"{metric}_max"] = run.sampler.window_max(start, end, metric)
        cells.append(summary)
        write_json(run.out / "throughput_cells.json", cells)
        print(
            f"cell {key}: agg {summary['agg_tps']} tok/s, steady {summary['steady_tps']} tok/s, "
            f"TTFT p50 {summary['ttft_p50_s']} s, errors {summary['errors']}, "
            f"degenerate {summary['degenerate']}",
            flush=True,
        )
        kill_check(run, f"cell {key}", rows, "request")


def probe_phase(run: Run, challenges_path: Path, solutions_path: Path) -> None:
    """Run the solve probe: every task at once, each with its rounds, within the phase budget.

    Args:
        run: The run.
        challenges_path: The training challenges.
        solutions_path: The training solutions.
    """
    settings = run.config["probe"]
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    task_ids, eligible = select_tasks(
        challenges, settings["n_tasks"], run.config["seed"], settings["max_task_chars"]
    )
    budget = min(settings["budget_s"], max(run.left(), 1))
    context = {
        "base_url": run.base_url,
        "model": run.config["server"]["served_model_name"],
        "seed": run.config["seed"],
        "python": sys.executable,
        "degenerate": run.config["degenerate"],
        "timing": run.out / "timing.jsonl",
        "writer": run.writer,
        "timeout_s": budget,
    }
    records = []
    todo = [t for t in task_ids if t not in run.done.get("tasks", {})]
    for task_id in task_ids:
        if task_id in run.done.get("tasks", {}):
            records.append({**run.done["tasks"][task_id], "resumed": True})
    inflight = Inflight()
    began = time.monotonic()
    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as pool:
            futures = {
                pool.submit(
                    solve_task, task_id, challenges[task_id], solutions.get(task_id), settings,
                    context, inflight,
                ): task_id
                for task_id in todo
            }  # fmt: skip
            try:
                for future in concurrent.futures.as_completed(futures, timeout=budget):
                    record = task_outcome(future, futures[future])
                    records.append(record)
                    run.writer.append(run.out / "probe.jsonl", record)
            except concurrent.futures.TimeoutError:
                # Shutting the sockets down ends the open requests at once; each task then
                # returns what it has, marked as cut by the phase budget.
                inflight.cancel()
                seen = {record["task_id"] for record in records}
                for future, task_id in futures.items():
                    if task_id in seen:
                        continue
                    record = {**task_outcome(future, task_id), "cut": True}
                    records.append(record)
                    run.writer.append(run.out / "probe.jsonl", record)
    verified = [r for r in records if r["verified"]]
    correct = [r for r in records if r["test_correct"]]
    tokens = sum(r["completion_tokens"] for r in records)
    run.metrics["probe"] = {
        "tasks": sorted(r["task_id"] for r in records),
        "eligible": eligible,
        "verified": len(verified),
        "verified_round1": sum(1 for r in records if r["verified_round"] == 1),
        "verified_round2": sum(1 for r in records if r["verified_round"] == 2),
        "test_correct": len(correct),
        "any_test_correct": sum(1 for r in records if r["any_test_correct"]),
        "precision": None if not verified else round(len(correct) / len(verified), 3),
        "completion_tokens": tokens,
        "tokens_per_verified": None if not verified else round(tokens / len(verified)),
        "length_stops": sum(r["length_stops"] for r in records),
        "no_program_rounds": sum(1 for r in records for x in r["rounds"] if not x["program_found"]),
        "errors": sum(r["errors"] for r in records),
        "degenerate": sum(r["degenerate"] for r in records),
        "cut": sum(1 for r in records if r.get("cut")),
        "wall_s": round(time.monotonic() - began, 1),
    }
    print(f"probe: {json.dumps(run.metrics['probe'])}", flush=True)
    kill_check(run, "probe", records, "task")


def sanity_check(run: Run) -> None:
    """Send one short request and require a sane, non-degenerate answer before measuring.

    Args:
        run: The run.

    Raises:
        Kill: If the request fails, returns nothing, or is degenerate.
    """
    settings = run.config["sanity"]
    payload = {
        "model": run.config["server"]["served_model_name"],
        "messages": [{"role": "user", "content": "What is 17 + 25? Reply with the number only."}],
        "max_tokens": settings["max_tokens"],
        "seed": run.config["seed"],
        "return_token_ids": True,
    }
    if settings["reasoning_effort"]:
        payload["reasoning_effort"] = settings["reasoning_effort"]
    began = time.monotonic()
    record = stream_chat(run.base_url, payload, 600)
    row = request_summary(record, run.config["degenerate"])
    # A wrong answer is recorded, not a kill: throughput is still measured right, and the probe
    # then shows whether the outputs make sense.
    answer = record["content"][-200:]
    run.metrics["sanity"] = {**row, "answer": answer, "answer_ok": "42" in answer}
    run.stage(
        "sanity",
        time.monotonic() - began,
        answer_ok="42" in answer,
        tokens=row["completion_tokens"],
    )
    if record["error"] or row["degenerate"] or not (record["content"] or record["reasoning"]):
        raise Kill(f"sanity request failed: {row}")


def copy_server_log(source: Path, target: Path) -> int:
    """Copy the head and the tail of the server log into the outputs.

    Args:
        source: The full log on scratch.
        target: ``vllm_server.log`` in ``out_dir``.

    Returns:
        The full log's line count.
    """
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    if len(lines) <= LOG_HEAD_LINES + LOG_TAIL_LINES:
        kept = lines
    else:
        omitted = len(lines) - LOG_HEAD_LINES - LOG_TAIL_LINES
        kept = lines[:LOG_HEAD_LINES] + [f"... {omitted} lines omitted ..."]
        kept += lines[-LOG_TAIL_LINES:]
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(lines)


def summary_markdown(run: Run) -> str:
    """Write ``summary.md``: the verdict, the boot, the throughput table, and the probe.

    Args:
        run: The finished run.

    Returns:
        The Markdown text.
    """
    metrics = run.metrics
    lines = [
        f"# {run.config['run_id']}: agent spike{' (smoke)' if run.args.smoke else ''}",
        "",
        f"- commit: `{run.args.commit}`; exit code {run.exit_code}; wall {metrics.get('wall_s')} s",
    ]
    lines += [f"- PROBLEM: {problem}" for problem in run.problems]
    preflight, agent = metrics.get("harness_preflight"), metrics.get("agent")
    shot = metrics.get("shot")
    if preflight or agent or metrics.get("forcing") or metrics.get("direct") or shot:
        lines += ["", "## Harness", ""]
    if preflight:
        lines.append(
            f"- preflight: ok {preflight['ok']} on {preflight['task']['task_id']} in "
            f"{preflight['seconds']} s (`harness_preflight.json`)"
        )
    forcing = metrics.get("forcing")
    if forcing:
        lines.append(
            f"- forcing check: {forcing['verdict']}; {forcing['programs']} forced programs of "
            f"{len(forcing['tasks'])} tasks (at least {forcing['min_programs']}); "
            f"{forcing['reasons'] or 'no problem'} (`forcing_check.json`)"
        )
    if shot:
        lines.append(
            f"- single shot: score {shot['score']} of {shot['tasks']}; solves {shot['solves']} "
            f"{shot['solved']} against NVARC's {shot['nvarc_solves']}; {shot['tokens_per_task']} "
            f"tokens per task; forcing turns {shot['forcing_turns']}; cut "
            f"{shot['forcing_cut']['cut']}; reading {shot['reading']} (`shot_metrics.json`)"
        )
    direct = metrics.get("direct")
    if direct:
        lines.append(
            f"- arm D (direct answers, pass@2 of the task and its view): score {direct['score']} "
            f"of {direct['tasks']}; M_direct {direct['M_direct']} of {direct['of']} (by the view "
            f"{direct['M_view']}; counted {direct['M_counted']}, contaminated "
            f"{direct['contaminated']}); accuracy "
            f"original {direct['samples']['original']['accuracy']}, view "
            f"{direct['samples']['view']['accuracy']}; agreement {direct['agreement']}, precision "
            f"when agreed {direct['precision_when_agreed']}; {direct['tokens_per_task']} tokens "
            f"per task; route {direct['route']['reading']} (`direct_metrics.json`)"
        )
    if agent and "route" in agent:
        lines.append(f"- arm L route: {agent['route']['reading']}; {agent['route']}")
    if agent and "stopped_by" in agent:
        lines.append(f"- gate: {agent['gate']['row']}; the {agent['stopped_by']} stopped the run")
    elif agent:
        gate = agent["gate"]
        lines.append(
            f"- gate: {gate['row']}; M {gate['M']} of {gate['of']}; precision "
            f"{gate['precision']}; engine healthy {gate['healthy']}; harness healthy "
            f"{gate['harness']}; score {agent['score']} of {agent['tasks']} ({agent['subset']})"
        )
        for phase in agent.get("phases", []):
            lines.append(
                f"- phase {phase['phase']}: attempts {phase['attempts']}, finished "
                f"{phase.get('finished')}, raised {phase.get('raised')} {phase.get('errors', {})}"
                f", forcing turns {phase.get('forcing_turns')}, forced programs "
                f"{phase.get('forced_programs')}"
            )
    boot = metrics.get("boot", {})
    prep = metrics.get("engine_prep", {})
    jit, imports = prep.get("jit_cache", {}), prep.get("imports", {})
    lines += [
        "",
        "## Engine",
        "",
        f"- engine prep: jit cache {jit.get('state')}, removed {jit.get('removed')}; "
        f"{len(imports.get('modules', []))} modules imported; FlashInfer all-reduce importable "
        f"{imports.get('flashinfer_all_reduce_available')}; {prep.get('seconds')} s",
        f"- boot to healthy: {boot.get('healthy_s')} s; KV pool "
        f"{boot.get('kv_cache_tokens_first')} tokens; maximum concurrency "
        f"{boot.get('max_concurrency_first')}",
        f"- peak memory per GPU (MiB): {metrics.get('peak_memory_mib')}",
        f"- NCCL transports: {boot.get('nccl_transport')}; P2P: "
        f"{metrics.get('preflight', {}).get('p2p')}",
        "",
        "## Throughput",
        "",
        "| Context | Concurrency | Agg tok/s | Steady tok/s | TTFT p50 s | TTFT p90 s | "
        "Mean context | Errors | Degenerate | Preemptions |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in metrics.get("throughput", []):
        if "skipped" in cell:
            lines.append(f"| {cell['context']} | {cell['concurrency']} | skipped: "
                         f"{cell['skipped']} | | | | | | | |")  # fmt: skip
            continue
        lines.append(
            f"| {cell['context']} | {cell['concurrency']} | {cell['agg_tps']} | "
            f"{cell['steady_tps']} | {cell['ttft_p50_s']} | {cell['ttft_p90_s']} | "
            f"{cell['mean_context']} | {cell['errors']} | {cell['degenerate']} | "
            f"{cell.get('server', {}).get('num_preemptions')} |"
        )
    probe = metrics.get("probe")
    if probe:
        lines += [
            "",
            "## Probe",
            "",
            f"- tasks: {len(probe['tasks'])} ({probe['eligible']} eligible); verified "
            f"{probe['verified']} (round 1: {probe['verified_round1']}, round 2: "
            f"{probe['verified_round2']}); correct on test {probe['test_correct']}; precision "
            f"{probe['precision']}",
            f"- completion tokens {probe['completion_tokens']}; per verified task "
            f"{probe['tokens_per_verified']}; length stops {probe['length_stops']}; rounds "
            f"without a program {probe['no_program_rounds']}; wall {probe['wall_s']} s",
        ]
    return "\n".join(lines) + "\n"


def execute(run: Run) -> None:
    """Run every stage and phase; raises Stop or Kill on a problem.

    Args:
        run: The run.
    """
    config, args = run.config, run.args
    began = time.monotonic()
    challenges = solutions = None
    if any(phase_kind(p) == "probe" for p in config["phases"]):
        if args.smoke:
            challenges, solutions = Path(args.smoke_challenges), Path(args.smoke_solutions)
        else:
            challenges, solutions = resolve_task_files(Path(config["input_root"]), config["probe"])
    model_dir, venv_python = None, None
    facts: dict[str, Any] = {}
    if not args.smoke:
        model_dir, facts["model"] = resolve_model(Path(args.model_root), config["model"])
        if args.venv_python is None:
            wheelhouse, facts["wheels"] = resolve_wheelhouse(
                Path(args.wheelhouse_root), config["wheels"]
            )
    if not args.smoke and "direct" in config["phases"] and config["direct"]["task_ids"]:
        # Refuse a configured shot task the challenges file lacks before the install and boot.
        shot_challenges, _ = resolve_task_files(Path(config["input_root"]), config["agent"])
        _, facts["direct_tasks"] = selected_tasks(
            shot_challenges, config["direct"]["task_ids"], config["agent"]["subset"]
        )
    run.stage("inputs", time.monotonic() - began, challenges=challenges, **facts)
    if any(phase_kind(p) in ("agent", "forcing") for p in config["phases"]):
        began = time.monotonic()
        harness = harness_preflight(run)
        run.metrics["harness_preflight"] = {
            key: harness[key] for key in ("ok", "task", "cell", "verification", "seconds")
        }
        run.stage("harness", time.monotonic() - began, task=harness["task"]["task_id"])
    env = clean_env(os.environ, config["server"]["unset_env"])
    env.update(config["server"]["env"])
    if not args.smoke:
        began = time.monotonic()
        if args.venv_python is None:
            venv_python, report = install_engine(wheelhouse, config["wheels"], run.out, env)
            imports = report.get("imports", {})
        else:
            venv_python, imports = Path(args.venv_python), {"given": args.venv_python}
        run.stage("install", time.monotonic() - began, imports=imports)
        # The venv's tools (ninja, the console scripts) come first for anything the engine runs.
        env["PATH"] = str(venv_python.parent) + os.pathsep + env.get("PATH", "")
    else:
        # The smoke prepares its own interpreter; its config imports only harmless modules.
        venv_python = Path(sys.executable)
    began = time.monotonic()
    env, prep = prepare_engine(venv_python, env, config["wheels"], run.out, run.scratch)
    run.metrics["engine_prep"] = prep
    run.stage(
        "engine_prep", time.monotonic() - began, jit_cache=prep["jit_cache"]["state"],
        removed=prep["jit_cache"]["removed"], modules=len(prep["imports"]["modules"]),
    )  # fmt: skip
    if not args.smoke:
        began = time.monotonic()
        run.metrics["preflight"] = preflight(
            venv_python, config["server"]["tensor_parallel_size"], run.out, env
        )
        run.stage("preflight", time.monotonic() - began)
        command = server_command(venv_python.parent / "vllm", model_dir, config["server"])
    else:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--fake-server",
            "--port", str(config["server"]["port"]), "--fake-replies", args.fake_replies,
            "--fake-mode", args.fake_mode,
        ]  # fmt: skip
    log = run.scratch / "vllm_server.log"
    run.server = Server(command, env, log, run.base_url)
    run.metrics["server_command"] = command
    run.server.start()
    limit = min(config["server"]["boot_timeout_s"], max(run.left(), 1))
    try:
        healthy = run.server.wait_healthy(limit)
    except Stop as err:
        if limit < config["server"]["boot_timeout_s"] and run.server.alive():
            raise WallTime(f"wall-time budget ran out during the boot: {err}") from err
        raise
    boot = parse_boot_log(log.read_text(encoding="utf-8", errors="replace"))
    status, body = http_get(run.base_url, "/v1/models")
    status_metrics, metrics_body = http_get(run.base_url, "/metrics")
    boot.update(
        healthy_s=round(healthy, 1), models=body[:500] if status == 200 else None,
        cache_config=cache_config_info(metrics_body) if status_metrics == 200 else {},
    )  # fmt: skip
    run.metrics["boot"] = boot
    write_json(run.out / "boot.json", boot)
    run.writer.append(
        run.out / "timing.jsonl",
        {"puzzle_id": None, "op": "boot", "seconds": round(healthy, 1), "tokens": None},
    )
    run.stage("boot", healthy, kv_cache_tokens=boot.get("kv_cache_tokens_first"))
    run.sampler = Sampler(
        run.base_url, run.out, config["server"]["sample_interval_s"], not args.smoke
    )
    run.sampler.start()
    sanity_check(run)
    for phase in config["phases"]:
        if run.left() <= 0:
            run.exit_code = EXIT_WALL
            run.problems.append(f"wall-time budget ran out before phase {phase}")
            return
        began = time.monotonic()
        kind = phase_kind(phase)
        if kind == "throughput":
            throughput_phase(run, phase)
        elif kind == "probe":
            probe_phase(run, challenges, solutions)
        elif kind == "preserve":
            preserve_phase(run)
        elif kind == "forcing":
            forcing_phase(run)
        elif kind == "direct":
            direct_phase(run)
        else:
            agent_phase(run)
        run.stage(phase, time.monotonic() - began)
        if run.left() <= 0:
            run.exit_code = EXIT_WALL
            run.problems.append(f"wall-time budget ran out during or after phase {phase}")
            return


# --- logging and the harness preflight ----------------------------------------------------------

# The run's log file in out_dir (a .log under run_logs/, so the pull keeps it): every record of
# the arc26 package from INFO up, with its traceback.
HARNESS_LOG = "harness.log"
# The preflight's cell, run as the agent's cells are: numpy's version and the number of helpers
# in lib's prompt index (one line each after the header line).
PREFLIGHT_CELL = (
    "import numpy as np\n"
    "print(np.version.version)\n"
    "print(len(lib.index_text().splitlines()) - 1)\n"
)
# The preflight's probe of a child, the Python-level policy off (the limits and the audit hook
# stay on): its interpreter, its import path, and where it loaded the agent's packages from.
PREFLIGHT_PROBE = (
    "import json, sys\n"
    "found = {}\n"
    "for name in ('arc26', 'numpy', 'scipy'):\n"
    "    module = sys.modules.get(name)\n"
    "    found[name] = None if module is None else {\n"
    "        'file': module.__file__, 'version': getattr(module, '__version__', None)}\n"
    "print(json.dumps({'executable': sys.executable, 'path': sys.path,\n"
    "                  'no_site': sys.flags.no_site, 'modules': found}))\n"
)
# The verifier's check: a program that runs on every task.
PREFLIGHT_PROGRAM = "def transform(grid):\n    return grid\n"


def error_kind(exc: BaseException) -> str:
    """Name an exception's kind: its type and its message's last line, with digits as ``N``.

    Why: one fault repeats across attempts with different process ids; a sandbox process that
    failed to start carries its own traceback in the message, whose last line names the cause
    (in run ``P6.3-02``, an ``ImportError`` in every REPL).

    Args:
        exc: The exception.

    Returns:
        For example ``"SandboxError: ImportError: cannot import name 'Sequence' from ..."``.
    """
    try:
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    except Exception:
        lines = []
    last = re.sub(r"\d+", "N", lines[-1])[:200] if lines else ""
    return f"{type(exc).__name__}: {last}" if last else type(exc).__name__


class FirstOfKind(logging.Filter):
    """Pass every record without an exception, and of the others the first of each kind.

    Why: one fault in 40 attempts would print 40 copies of its traceback; the log file keeps them
    all, and the phase report counts them by kind.
    """

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether a record is printed.

        Args:
            record: The log record.

        Returns:
            True for a record without an exception and for the first of each ``error_kind``.
        """
        exc = record.exc_info[1] if record.exc_info else None
        if exc is None:
            return True
        kind = error_kind(exc)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return self.counts[kind] == 1


def configure_logging(out: Path) -> list[logging.Handler]:
    """Send the arc26 package's log records to standard output and to ``harness.log``.

    Why: run ``P6.3-02`` (12 Oct) lost the traceback of each of its 80 failed attempts. P6.2's
    runner logs a faulting attempt with ``logger.exception``, and nothing in the spike's process
    printed those records (the image's ``sitecustomize`` runs there and sets up logging). The
    ``arc26`` logger now has handlers of its own and does not propagate, so the root logger's
    setup no longer matters: the file keeps every record from INFO up with its traceback, and
    standard output shows warnings and errors, each exception kind's traceback once.

    Args:
        out: The output folder.

    Returns:
        The two handlers, for ``close_logging``.
    """
    logging.disable(logging.NOTSET)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    to_file = logging.FileHandler(out / HARNESS_LOG, encoding="utf-8")
    to_file.setLevel(logging.INFO)
    to_stdout = logging.StreamHandler(sys.stdout)
    to_stdout.setLevel(logging.WARNING)
    to_stdout.addFilter(FirstOfKind())
    logger = logging.getLogger("arc26")
    for handler in (to_file, to_stdout):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return [to_file, to_stdout]


def close_logging(handlers: Sequence[logging.Handler]) -> None:
    """Detach and close the handlers of ``configure_logging``; the logger propagates again.

    Args:
        handlers: Its handlers.
    """
    logger = logging.getLogger("arc26")
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True


def sandbox_parts(settings: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return P6.1's REPL factory and a verifier pool with the config's limits.

    Args:
        settings: The config's ``agent`` section.

    Returns:
        A callable that starts a ``SubprocessRepl`` for a task, and a ``PoolVerifier``.
    """
    import functools

    from arc26.agent.sandbox import SubprocessRepl
    from arc26.agent.verify import PoolVerifier

    memory = settings["memory_mb"] * 1024 * 1024
    verifier = PoolVerifier(
        settings["verifier_workers"], call_timeout_s=settings["call_timeout_s"], memory_bytes=memory
    )
    return functools.partial(SubprocessRepl, memory_bytes=memory), verifier


def run_sandbox(run: Run, settings: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return the run's REPL factory and verifier pool: the preflight's, else new ones.

    Args:
        run: The run (``run.sandbox`` is set here when empty; ``finish`` closes the pool).
        settings: The config's ``agent`` section.

    Returns:
        The factory and the pool.
    """
    if run.sandbox is None:
        run.sandbox = sandbox_parts(settings)
    return run.sandbox


def parent_facts() -> dict[str, Any]:
    """Return where this process imports the agent's packages from, and how it starts a child.

    Returns:
        The interpreter, ``sys.path``, the effective user id, the temporary folder, each package's
        file and version (scipy found, not imported), ``import_roots``, and the child's command.
    """
    import numpy

    import arc26
    from arc26.agent.sandbox import import_roots, spawn_command

    scipy = importlib.util.find_spec("scipy")
    return {
        "executable": sys.executable,
        "path": list(sys.path),
        "euid": os.geteuid(),
        "tmp": tempfile.gettempdir(),
        "modules": {
            "arc26": {"file": arc26.__file__, "version": arc26.__version__},
            "numpy": {"file": numpy.__file__, "version": numpy.__version__},
            "scipy": None if scipy is None else {"file": scipy.origin, "version": None},
        },
        "import_roots": import_roots(),
        "child_command": spawn_command("arc26.agent.sandbox"),
    }


def preflight_task(run: Run) -> Any:
    """Return the harness preflight's task: the first one the agent phase runs on the subset.

    The smallest (``task_cells``) of the config's subset. The smoke path takes it when the
    competition's files are under the input root (a Kaggle CPU session), else its first fixture.

    Args:
        run: The run.

    Returns:
        The ``AgentTask``.

    Raises:
        Stop: If a real run finds no challenges file.
    """
    from arc26.agent.tasks import load_tasks
    from arc26.data import subset_ids

    settings = run.config["agent"]
    try:
        challenges, _ = resolve_task_files(Path(run.config["input_root"]), settings)
        tasks = list(load_tasks(challenges, subset_ids(settings["subset"])))
    except (Stop, ValueError):
        if not run.args.smoke:
            raise
        tasks = list(load_tasks(run.args.smoke_challenges))
    return min(tasks, key=task_cells)


def harness_preflight(run: Run) -> dict[str, Any]:
    """Prove in seconds, before the install and the boot, that P6.1's sandbox works here.

    Why: in run ``P6.3-02`` (12 Oct) every sandbox process died on import, and the run found out
    after the install, the boot, and the preserve check, with no traceback. Each step uses the
    agent phase's own pieces (``sandbox_parts``, the config's limits): (1) the parent's facts;
    (2) a ``SubprocessRepl`` from the agent's REPL factory, for the agent's first task, runs a
    cell that imports numpy and counts lib's helpers, and both must equal the parent's; (3) a
    second REPL from the factory, its Python-level policy off, reports the child's interpreter,
    import path, and where arc26, numpy, and scipy came from, and arc26 and numpy must be the
    parent's files; (4) the agent's ``PoolVerifier`` runs the identity program on the task, and
    every demonstration and test input must come back a grid. The factory and the pool stay in
    ``run.sandbox`` for the agent phase.

    Args:
        run: The run.

    Returns:
        The record, also written to ``harness_preflight.json``.

    Raises:
        Stop: If a step fails; its traceback is printed and recorded first.
    """
    import numpy

    from arc26.agent import lib

    settings = run.config["agent"]
    timeout = settings["cell_timeout_s"]
    began = time.monotonic()
    record: dict[str, Any] = {"ok": False}
    step = "parent"
    try:
        record["parent"] = parent_facts()
        step = "task"
        task = preflight_task(run)
        record["task"] = {"task_id": task.task_id, "cells": task_cells(task)}
        step = "sandbox"
        factory, verifier = run_sandbox(run, settings)
        step = "repl"
        with factory(task) as repl:
            cell = repl.run_cell(PREFLIGHT_CELL, timeout)
        record["cell"] = {"stdout": cell.stdout, "error": cell.error, "seconds": cell.seconds}
        if cell.error is not None:
            raise Stop(f"the cell failed: {cell.error}\n{cell.stdout}")
        expected = [numpy.__version__, str(len(lib.index_text().splitlines()) - 1)]
        if cell.stdout.split() != expected:
            raise Stop(f"the cell printed {cell.stdout.split()}; the parent has {expected}")
        step = "probe"
        with factory(task, policy=False) as repl:
            probe = repl.run_cell(PREFLIGHT_PROBE, timeout)
        if probe.error is not None:
            raise Stop(f"the probe failed: {probe.error}\n{probe.stdout}")
        child = record["child"] = json.loads(probe.stdout)
        for name in ("arc26", "numpy"):
            mine = os.path.realpath(record["parent"]["modules"][name]["file"])
            theirs = os.path.realpath((child["modules"].get(name) or {}).get("file") or "")
            if mine != theirs:
                raise Stop(f"the child imports {name} from {theirs}, the parent from {mine}")
        step = "verifier"
        verification = verifier.verify(task, PREFLIGHT_PROGRAM)
        failures = [f"pair {p.index}: {p.error}" for p in verification.pairs if p.error]
        failures += [
            f"test input {k}: no grid"
            for k, grid in enumerate(verification.test_outputs)
            if grid is None
        ]
        record["verification"] = {
            "pairs": len(verification.pairs), "passed": verification.passed,
            "test_outputs": len(verification.test_outputs), "failures": failures,
            "seconds": verification.seconds,
        }  # fmt: skip
        if failures:
            raise Stop(f"the verifier failed on the identity program: {failures}")
        record["ok"] = True
    except Exception as err:
        text = traceback.format_exc()
        reason = str(err) if isinstance(err, Stop) else error_kind(err)
        record.update(failed_step=step, error=reason, traceback=text)
        print(f"harness preflight failed at {step}:\n{text}", flush=True)
        raise Stop(f"harness preflight failed at {step}: {reason}") from err
    finally:
        record["seconds"] = round(time.monotonic() - began, 2)
        write_json(run.out / "harness_preflight.json", record)
    child_numpy = record["child"]["modules"]["numpy"]
    print(
        f"harness preflight: ok in {record['seconds']} s on {task.task_id}; child numpy "
        f"{child_numpy['version']} from {child_numpy['file']}; child path "
        f"{record['child']['path']}",
        flush=True,
    )
    return record


def reasoning_location(message: Mapping[str, Any]) -> dict[str, Any]:
    """Return where one chat completion message carries its reasoning.

    Why: vLLM 0.31 names the field ``reasoning``, SGLang and older vLLM ``reasoning_content``,
    and a server without a reasoning parser leaves the thinking inline in ``content`` up to
    ``</think>``. P6.2's client reads all three; the spike records which one each request used.

    Args:
        message: ``choices[0].message`` of a non-streaming response.

    Returns:
        ``location``: the first of ``"reasoning"``, ``"reasoning_content"``, ``"inline"``,
        ``"none"`` that applies; ``fields``: the reasoning fields present and non-empty;
        ``think_close_in_content``: whether ``content`` holds ``</think>``.
    """
    fields = [name for name in ("reasoning", "reasoning_content") if message.get(name)]
    content = message.get("content")
    inline = isinstance(content, str) and "</think>" in content
    location = fields[0] if fields else ("inline" if inline else "none")
    return {"location": location, "fields": fields, "think_close_in_content": inline}


def make_engine(
    run: Run, max_connections: int, capture_prompts: bool = False, **overrides: Any
) -> Any:
    """Build P6.2's ``OpenAIEngine`` with a hook that records every call's reasoning location.

    Each call appends one row to ``engine_calls.jsonl``: the attempt (from the loop's context),
    ``reasoning_location`` of the reply, how many assistant turns the request sent back with
    reasoning and with which ``preserve_thinking``, the call's thinking switch (False on a forcing
    turn), the token counts, and the finish reason. The
    engine's own timing records go to ``timing/timing-engine.jsonl``.

    Args:
        run: The run (server address, model name, output folder, JSONL writer).
        max_connections: Connections in flight (the agent's concurrency).
        capture_prompts: Keep each answered call's prompt as the response states it (its
            ``prompt_token_ids`` and, requested only then, ``prompt_text``) in the engine's
            ``prompts`` list, for the forcing check's render check.
        **overrides: Further ``EngineConfig`` fields (the preserve check's variants).

    Returns:
        The engine.
    """
    from arc26.agent.engine import EngineConfig, OpenAIEngine
    from arc26.agent.loop import current_attempt
    from arc26.timing import TimingLogger

    calls = run.out / "engine_calls.jsonl"
    writer = run.writer
    degenerate = run.config["degenerate"]

    class RecordingEngine(OpenAIEngine):
        """``OpenAIEngine`` that logs each call: reasoning location, health, tokens, errors."""

        async def count_prompt_tokens(self, messages: Sequence[Any], **kwargs: Any) -> int:
            """Count the opening prompt and log it to ``prompt_counts.jsonl`` (revision 6)."""
            context = current_attempt()
            row: dict[str, Any] = {"attempt_id": None if context is None else context.attempt_id}
            try:
                count = await super().count_prompt_tokens(messages, **kwargs)
            except BaseException as err:
                row["error"] = f"{type(err).__name__}: {err}"[:300]
                writer.append(self.counts_path, row)
                raise
            row.update(count=count, error=None)
            writer.append(self.counts_path, row)
            return count

        async def _post(self, body: Mapping[str, Any]) -> Any:
            began = time.monotonic()
            context = current_attempt()
            field = self.config.reasoning_field
            sent = [m for m in body["messages"] if m.get("role") == "assistant" and m.get(field)]
            template = body.get("chat_template_kwargs") or {}
            row: dict[str, Any] = {
                "puzzle_id": None if context is None else context.task_id,
                "attempt_id": None if context is None else context.attempt_id,
                "reasoning_sent": len(sent),
                "preserve_thinking": template.get("preserve_thinking"),
                # False on a forcing turn (P6.2's loop, revision 5); absent means the default, on.
                "thinking": template.get(self.config.thinking_kwarg),
            }
            if "thinking_token_budget" in body:  # in-request forcing only (run P6.9-01)
                row["thinking_token_budget"] = body["thinking_token_budget"]
            try:
                data = await super()._post(body)
            except BaseException as err:
                row.update(error=f"{type(err).__name__}: {err}"[:500])
                row["seconds"] = round(time.monotonic() - began, 3)
                row["ended"] = round(time.time(), 3)
                writer.append(calls, row)
                raise
            choices = data.get("choices") if isinstance(data, dict) else None
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            usage = (data.get("usage") if isinstance(data, dict) else None) or {}
            text = (message.get("content") or "") if isinstance(message.get("content"), str) else ""
            thought = message.get("reasoning") or message.get("reasoning_content") or ""
            stats = call_health(
                choice.get("token_ids") or [], text, thought, body.get("max_tokens"), degenerate
            )
            if capture_prompts and isinstance(data, dict):
                self.prompts.append(
                    {"ids": data.get("prompt_token_ids"), "text": data.get("prompt_text")}
                )
            row.update(
                error=None,
                **reasoning_location(message),
                **stats,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=choice.get("finish_reason"),
                seconds=round(time.monotonic() - began, 3),
                ended=round(time.time(), 3),
            )
            writer.append(calls, row)
            return data

    # vLLM returns each reply's token ids on request, which the degenerate check reads (and the
    # prompt's, ChatCompletionResponse.prompt_token_ids); the forcing check also asks for the
    # templated prompt text (return_prompt_text, vLLM 0.31).
    extra = {"return_token_ids": True, **dict(overrides.pop("extra_body", {}))}
    if capture_prompts:
        extra["return_prompt_text"] = True
    counts = run.out / "prompt_counts.jsonl"
    config = EngineConfig(
        base_url=run.base_url + "/v1",
        # The opening prompt's count (revision 5): vLLM's /tokenize, outside /v1.
        tokenize_url=run.base_url + "/tokenize",
        model=run.config["server"]["served_model_name"],
        max_connections=max_connections,
        extra_body=extra,
        **overrides,
    )
    engine = RecordingEngine(config, timing=TimingLogger(run.out / "timing", "engine"))
    engine.prompts = []
    engine.counts_path = counts
    return engine


def call_health(
    token_ids: Sequence[int],
    text: str,
    thought: str,
    max_tokens: int | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Judge one engine reply for the health check: repetition, emptiness, NaN.

    A reply is degenerate when its longest run of one non-special token before the first
    end-of-sequence id reaches ``max_run`` (``token_stats``, as stage 1 counts it; a run of token
    0 is the usual sign of NaN logits). An empty reply (no answer and no reasoning) is counted
    apart as ``empty``, a model behavior reported, not a degenerate output: run ``P6.3-04``'s
    whole Fault came from one forced reply that stopped after one token, with no run and no
    token 0. A request capped under 16 tokens (the preserve check's one-token turns) may come
    back blank, so it is never counted empty.

    Args:
        token_ids: The reply's token ids (vLLM's ``return_token_ids``), empty when absent.
        text: The answer text.
        thought: The reasoning text.
        max_tokens: The request's cap on generated tokens.
        settings: The config's ``degenerate`` section.

    Returns:
        ``max_run``, ``token0_count``, ``empty``, ``degenerate``, and the run's ``source``.
    """
    stats = token_stats(
        token_ids, settings["special_min_id"], settings["max_run"], thought + text,
        settings["eos_ids"],
    )  # fmt: skip
    empty = (max_tokens or 0) >= 16 and not text.strip() and not thought.strip()
    return {
        "max_run": stats["max_run"],
        "token0_count": stats["token0_count"],
        "run_source": stats["source"],
        "empty": empty,
        "degenerate": bool(stats["degenerate"]),
    }


class AgentKit:
    """The harness of one run: an engine, a REPL factory, and a verifier pool.

    Real runs pair P6.2's engine with P6.1's sandboxed ``SubprocessRepl`` and ``PoolVerifier``;
    the smoke path swaps in P6.2's ``ScriptedEngine`` and keeps P6.1's real REPL and verifier,
    which makes it the integration test of P6.1 and P6.2.
    """

    def __init__(self, engine: Any, repl_factory: Any, verifier: Any) -> None:
        self.engine, self.repl_factory, self.verifier = engine, repl_factory, verifier

    async def aclose(self) -> None:
        """Close the engine's connection pool inside the event loop that used it."""
        closer = getattr(self.engine, "aclose", None)
        if closer is not None:
            await closer()

    def close(self) -> None:
        """Stop the verifier's worker processes."""
        self.verifier.close()


def real_kit(run: Run, settings: Mapping[str, Any]) -> AgentKit:
    """Build the GPU run's harness: P6.2's engine (recording) with P6.1's REPL and verifier.

    Args:
        run: The run.
        settings: The config's ``agent`` section.

    Returns:
        The harness.
    """
    factory, verifier = run_sandbox(run, settings)
    return AgentKit(make_engine(run, settings["concurrency"]), factory, verifier)


def smoke_scripts(tasks: Sequence[Any], solutions: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Script P6.2's ``ScriptedEngine`` for the smoke path's tasks.

    The fixtures get the replies that exercise each outcome: a reply cut at its cap while
    reasoning, then the right program in the forcing turn (``smoke-mirror``), a wrong program then
    a fix after the verifier's report (``smoke-transpose``), and for ``smoke-recolor`` no code,
    then a program that copies the demonstration outputs (P6.1's guards reject it), then the right
    grid as a direct answer. Real tasks (DEV4, when their files are given) get a cut reply, the
    identity program in its forcing turn, the guard-rejected copy, and the test input as a direct
    answer: the plumbing on real grid sizes, not solves.

    Args:
        tasks: The ``AgentTask`` objects.
        solutions: Test outputs by task id (the fixtures' answers).

    Returns:
        Replies per task id.
    """
    from arc26.agent.mock_engine import (
        answer_reply,
        code_reply,
        hostile_programs,
        no_code_reply,
        overlong_reply,
    )

    # Reasoning longer than any smoke turn's cap (smoke_overrides: 1,024 tokens, 4 characters a
    # token), so the reply is cut before its visible text.
    cut = overlong_reply(SMOKE_CUT_CHARS)
    scripts: dict[str, list[Any]] = {}
    for task in tasks:
        copy_outputs = code_reply(hostile_programs(task)["literal_lookup"], "A lookup table.")
        if task.task_id == "smoke-mirror":
            scripts[task.task_id] = [cut, code_reply(MIRROR, "The rows are mirrored.")]
        elif task.task_id == "smoke-transpose":
            scripts[task.task_id] = [code_reply(MIRROR, "Mirror?"), code_reply(TRANSPOSE)]
        elif task.task_id == "smoke-recolor":
            answer = solutions[task.task_id][0]
            scripts[task.task_id] = [no_code_reply(), copy_outputs, answer_reply(0, answer)]
        else:
            identity = code_reply("def transform(grid):\n    return grid\n", "Identity first.")
            scripts[task.task_id] = [
                cut, identity, copy_outputs, answer_reply(0, task.test_inputs[0]),
            ]  # fmt: skip
    return scripts


def smoke_kit(
    run: Run, settings: Mapping[str, Any], tasks: Sequence[Any], solutions: Mapping[str, Any]
) -> AgentKit:
    """Build the smoke path's harness: P6.2's ``ScriptedEngine`` with P6.1's REPL and verifier.

    Args:
        run: The run.
        settings: The config's ``agent`` section.
        tasks: The smoke's tasks.
        solutions: Their test outputs.

    Returns:
        The harness.
    """
    from arc26.agent.mock_engine import ScriptedEngine
    from arc26.timing import TimingLogger

    factory, verifier = run_sandbox(run, settings)
    engine = ScriptedEngine(
        smoke_scripts(tasks, solutions), timing=TimingLogger(run.out / "timing", "engine")
    )
    return AgentKit(engine, factory, verifier)


def task_cells(task: Any) -> int:
    """Return a task's cell count over all its grids: the scheduler's ease order (small first).

    Args:
        task: An ``AgentTask``.

    Returns:
        The number of cells.
    """
    grids = [g for p in task.train for g in (p.input, p.output)] + list(task.test_inputs)
    return sum(len(g) * len(g[0]) for g in grids)


async def run_cascade(
    tasks: Sequence[Any], kit: AgentKit, settings: Mapping[str, Any], run: Run
) -> dict[str, Any]:
    """Run the phases with P6.2's ``run_tasks``: each phase attempts the still unverified tasks.

    Phase 1 gives every task one short attempt, smallest first; phase 2 one long attempt to the
    tasks no verified program covers, with their failed programs shown (design note, 5.4).
    Each phase stops at its budget or the run's deadline: attempts not yet started are skipped,
    and running ones stop at their next engine call (P6.2's ``deadline``).

    Args:
        tasks: The ``AgentTask`` objects.
        kit: The harness.
        settings: The config's ``agent`` section.
        run: The run (seed, deadline, outputs).

    Returns:
        ``records`` (every ``AttemptRecord``), ``phases`` (per phase: attempts, finished,
        attempts that raised and their kinds, verified so far, stop reasons, forcing turns and
        forced programs, wall time), and
        ``errors`` (per ``error_kind``: the phase, the first attempt, and its traceback).
    """
    from arc26.agent.loop import Budget, run_tasks

    ordered = sorted(tasks, key=task_cells)
    records: list[Any] = []
    report: list[dict[str, Any]] = []
    solved: set[str] = set()
    prior: dict[str, list[Any]] = {}
    first_errors: dict[str, dict[str, Any]] = {}
    for number, phase in enumerate(settings["phases"], start=1):
        todo = [task for task in ordered if task.task_id not in solved]
        began, started = time.monotonic(), time.time()
        end = min(run.deadline, began + phase["budget_s"])
        if not todo or end <= began:
            report.append({"phase": number, "attempts": 0, "skipped": "none left" if not todo
                           else "deadline"})  # fmt: skip
            continue
        budget = Budget(
            max_turns=phase["rounds"],
            attempt_tokens=phase["token_budget"],
            turn_tokens=phase["max_tokens"],
            context_tokens=settings["context_tokens"],
            temperature=1.0 if phase["temperature"] is None else phase["temperature"],
            reasoning_effort=phase["reasoning_effort"],
            cell_timeout_s=settings["cell_timeout_s"],
            preserve_thinking=phase["preserve_thinking"],
            forcing_tokens=settings.get("forcing_tokens"),
        )
        errors: dict[str, int] = {}

        def on_error(
            attempt_id: str, exc: Exception, errors: dict[str, int] = errors, phase_no: int = number
        ) -> None:
            kind = error_kind(exc)
            errors[kind] = errors.get(kind, 0) + 1
            if kind not in first_errors:
                text = "".join(traceback.format_exception(exc))
                first_errors[kind] = {
                    "phase": phase_no, "attempt_id": attempt_id, "traceback": text[-8000:],
                }  # fmt: skip

        done = await run_tasks(
            todo, kit.engine, kit.repl_factory, kit.verifier, budget,
            concurrency=settings["concurrency"], phase=number, seed=run.config["seed"],
            deadline=end, log_path=run.out / "agent_attempts.jsonl",
            transcript_path=run.out / "agent_transcripts.jsonl", prior=prior or None,
            on_error=on_error,
        )  # fmt: skip
        for record in done:
            records.append(record)
            if any(v.verified for v in record.verifications):
                solved.add(record.task_id)
            else:
                prior.setdefault(record.task_id, []).extend(record.verifications)
        stops: dict[str, int] = {}
        for record in done:
            stops[record.stop_reason] = stops.get(record.stop_reason, 0) + 1
        report.append(
            {
                "phase": number,
                "attempts": len(todo),
                "finished": len(done),
                "raised": sum(errors.values()),
                "errors": errors,
                "verified_after": len(solved),
                "stop_reasons": stops,
                "completion_tokens": sum(r.completion_tokens for r in done),
                "forcing_turns": sum(r.forcing_turns for r in done),
                "forced_programs": sum(r.forced_programs for r in done),
                "wall_s": round(time.monotonic() - began, 1),
                "started": round(started, 3),
                "ended": round(time.time(), 3),
            }
        )
        print(f"agent phase {number}: {json.dumps(report[-1])}", flush=True)
    return {"records": records, "phases": report, "errors": first_errors}


SANDBOX_ERRORS = ("timeout", "memory", "crashed")


def gate_row(solved: int, precision: float | None, healthy: bool) -> str:
    """Return the row of ADR-020's amended solve gate (9 Oct) for stage 2's numbers.

    The rows are read in the order Good, Bad, Borderline: Good at 7 or more of the 29 misses
    solved by verified programs with a precision of at least 0.8; Bad at 4 or fewer, only with a
    healthy engine and harness; Borderline at 5 or 6, or at 7 or more with a precision from 0.6
    up to 0.8; Fault for everything else (an unhealthy engine or harness below 5, or 7 or more
    under 0.6). A phase whose attempts all raised (run ``P6.3-02``, 12 Oct) is thus a Fault,
    never Bad.

    Args:
        solved: :math:`M`, the misses solved by verified programs.
        precision: The share of verified programs right on every test output, or None.
        healthy: The engine (errors on under 10% of calls, no degenerate output) and the harness
            (``harness_health``) both healthy.

    Returns:
        ``"Good"``, ``"Bad"``, ``"Borderline"``, or ``"Fault"``.
    """
    if solved >= 7 and precision is not None and precision >= 0.8:
        return "Good"
    if solved <= 4 and healthy:
        return "Bad"
    if solved in (5, 6) or (solved >= 7 and precision is not None and 0.6 <= precision < 0.8):
        return "Borderline"
    return "Fault"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file's rows; an absent file or a cut line gives fewer rows, never an error.

    Args:
        path: The file.

    Returns:
        The parsed rows.
    """
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def phase_stalls(
    phases: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    min_tokens_per_s: float,
    stall_s: float,
) -> list[str]:
    """Find agent phases that ended on a hung engine, for the engine's health (revision 5).

    Why: P6.2's engine has no read timeout, so a hung engine's calls end as cancellations at the
    phase's deadline, which ``engine_call_summary`` does not count as failures, and the gate
    would read Bad on a fault. Only a phase whose deadline cancelled calls is judged; it is
    stalled when its finished attempts generated fewer than ``min_tokens_per_s`` tokens per
    second of its wall time, or when its last answered call ended more than ``stall_s`` before
    the phase did.

    Args:
        phases: ``run_cascade``'s phase report (with ``started`` and ``ended``).
        calls: ``engine_calls.jsonl``'s rows (with ``ended`` on every row).
        min_tokens_per_s: The floor of the phase's rate.
        stall_s: The longest wait for an answered call before the deadline.

    Returns:
        One line per stalled phase and rule; empty when none.
    """
    faults = []
    for row in phases:
        if not row.get("attempts") or "ended" not in row or "started" not in row:
            continue
        tag = f"/p{row['phase']}/"
        own = [c for c in calls if tag in str(c.get("attempt_id") or "")]
        cut = sum(1 for c in own if str(c.get("error") or "").startswith("CancelledError"))
        if not cut:
            continue
        rate = row.get("completion_tokens", 0) / max(float(row.get("wall_s") or 0), 1e-6)
        if rate < min_tokens_per_s:
            faults.append(
                f"phase {row['phase']}: {rate:.1f} tokens per second, under {min_tokens_per_s}, "
                f"with {cut} calls cut at its deadline"
            )
        answered = [float(c["ended"]) for c in own if not c.get("error") and "ended" in c]
        gap = float(row["ended"]) - max(answered, default=float(row["started"]))
        if gap > stall_s:
            faults.append(
                f"phase {row['phase']}: no call answered in its last {gap:.0f} s (over {stall_s}), "
                f"with {cut} calls cut at its deadline"
            )
    return faults


def count_check(
    counts: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare each attempt's opening count with its first call's real ``prompt_tokens``.

    Why: the opening bound is the server's count plus 64 (P6.2's ``COUNT_SLACK``); an
    undercount past that would let a request overrun the context (revision 6 logs both).

    Args:
        counts: ``prompt_counts.jsonl``'s rows.
        calls: ``engine_calls.jsonl``'s rows.

    Returns:
        ``attempts`` compared, ``failed`` counts, ``exact`` matches, the largest and smallest
        ``prompt_tokens - count``, and ``within_slack`` (every difference at most 64).
    """
    from arc26.agent.loop import COUNT_SLACK

    first: dict[str, int] = {}
    for row in calls:
        attempt = row.get("attempt_id")
        if attempt and not row.get("error") and attempt not in first:
            if isinstance(row.get("prompt_tokens"), int):
                first[attempt] = row["prompt_tokens"]
    diffs = [
        first[row["attempt_id"]] - int(row["count"])
        for row in counts
        if isinstance(row.get("count"), int) and row.get("attempt_id") in first
    ]
    return {
        "attempts": len(diffs),
        "failed": sum(1 for row in counts if row.get("error")),
        "exact": sum(1 for d in diffs if d == 0),
        "max_prompt_minus_count": max(diffs, default=None),
        "min_prompt_minus_count": min(diffs, default=None),
        "within_slack": all(d <= COUNT_SLACK for d in diffs),
    }


def harness_health(phases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Judge whether the harness ran the cascade's attempts, for the gate's Bad row.

    Why: run ``P6.3-02`` (12 Oct) read Bad with a healthy engine while none of its 80 attempts
    had started (every REPL failed): no evidence about the model. Healthy: every phase that
    started attempts finished at least one, and under 10% of its attempts raised (the engine's
    error share, ADR-020's amendment). A phase skipped at the deadline started none.

    Args:
        phases: ``run_cascade``'s phase report.

    Returns:
        ``healthy`` and ``faults`` (one line per phase that broke a rule).
    """
    faults = []
    for row in phases:
        attempts = row.get("attempts", 0)
        if not attempts:
            continue
        finished, raised = row.get("finished", 0), row.get("raised", 0)
        if not finished:
            faults.append(f"phase {row['phase']}: none of {attempts} attempts finished")
        elif raised / attempts >= 0.10:
            faults.append(f"phase {row['phase']}: {raised} of {attempts} attempts raised")
    return {"healthy": not faults, "faults": faults}


def agent_metrics(
    tasks: Sequence[Any],
    records: Sequence[Any],
    selection: Mapping[str, Any],
    submission: Mapping[str, Any],
    solutions: Mapping[str, Any],
    nvarc_misses: Sequence[str],
    label: str,
    health: Mapping[str, Any],
    harness: Mapping[str, Any],
) -> dict[str, Any]:
    """Score stage 2 in development mode and compute the gate and the design note's 6.2 measures.

    The score is ``arc26.evaluate``'s. A task counts as **solved by a verified program** only
    when, for every test output, one of the two chosen attempts is right and comes from a verified
    program (every demonstration reproduced, no fatal guard: P6.2's ``"program"`` candidates with
    at least one vote); a task scored by a direct answer, an unverified program, or only in part
    is reported separately and never counts toward :math:`M`. Precision is the share of distinct
    verified programs (by source hash, per task) right on every test output.

    Args:
        tasks: The ``AgentTask`` objects.
        records: Every ``AttemptRecord``.
        selection: P6.2's ``build_selection`` of the records (the candidates with their sources).
        submission: Its ``submission_from_selection``.
        solutions: Test outputs by task id (read only after the attempts, for this).
        nvarc_misses: The tasks NVARC's ``P2.6-01`` scored 0 on.
        label: The subset's name.
        health: ``engine_call_summary``'s result (``healthy`` for the Bad row).
        harness: ``harness_health``'s result (``healthy`` for the Bad row).

    Returns:
        The gate (``M``, precision, the engine's and the harness's health, row), the totals, and
        one row per task.
    """
    from arc26.evaluate import score_submission

    ids = [task.task_id for task in tasks]
    scored = score_submission(submission, solutions, ids, label)
    scores = {t.task_id: t for t in scored.per_task}
    by_task: dict[str, list[Any]] = {}
    for record in records:
        by_task.setdefault(record.task_id, []).append(record)
    per_task = {}
    programs = right_programs = 0
    for task in tasks:
        own = by_task.get(task.task_id, [])
        expected = list(solutions[task.task_id])
        checks = [v for r in own for v in r.verifications]
        verified = {v.source_sha256: v for v in checks if v.verified}
        right = sum(1 for v in verified.values() if list(v.test_outputs) == expected)
        programs += len(verified)
        right_programs += right
        chosen = selection.get(task.task_id, [])
        by_program = len(chosen) == len(expected) and all(
            any(c.grid == want and c.source == "program" and c.votes >= 1 for c in options)
            for options, want in zip(chosen, expected, strict=True)
        )
        score = scores[task.task_id].score
        if by_program:
            how = "verified program"
        elif score >= 1:
            how = "other answer"
        elif score > 0:
            how = "partial"
        else:
            how = "unsolved"
        first = [row["attempt_1"] for row in submission.get(task.task_id, [])]
        per_task[task.task_id] = {
            "score": round(score, 3),
            "solved_by": how,
            "verified": bool(verified),
            "verified_programs": len(verified),
            "right_programs": right,
            "top_correct": bool(verified) and first == expected,
            "attempts": len(own),
            "phases": sorted({r.phase for r in own}),
            "turns": sum(r.turns for r in own),
            "completion_tokens": sum(r.completion_tokens for r in own),
            "prompt_tokens": sum(r.prompt_tokens for r in own),
            "programs": len(checks),
            "direct_answers": sum(len(r.direct_answers) for r in own),
            "guard_rejections": sum(1 for v in checks if any(g.fatal for g in v.guards)),
            "sandbox_failures": sum(
                1
                for v in checks
                for p in v.pairs
                if p.error and str(p.error).startswith(SANDBOX_ERRORS)
            ),  # fmt: skip
            "stop_reasons": sorted({r.stop_reason for r in own}),
            "forcing_turns": sum(r.forcing_turns for r in own),
            "forced_programs": sum(r.forced_programs for r in own),
        }
    verified_tasks = [t for t, row in per_task.items() if row["verified"]]
    tokens = sum(row["completion_tokens"] for row in per_task.values())
    turns = [row["turns"] for row in per_task.values() if row["attempts"]]
    misses = [t for t in nvarc_misses if t in per_task]
    solved = sorted(t for t in misses if per_task[t]["solved_by"] == "verified program")
    precision = round(right_programs / programs, 3) if programs else None
    healthy = bool(health.get("healthy", False))
    harness_ok = bool(harness.get("healthy", False))
    return {
        "subset": label,
        "tasks": len(tasks),
        "gate": {
            "M": len(solved),
            "of": len(misses),
            "precision": precision,
            "healthy": healthy,
            "harness": harness_ok,
            "row": gate_row(len(solved), precision, healthy and harness_ok),
        },
        "nvarc_misses_solved_by_verified_programs": solved,
        "nvarc_misses_scored_otherwise": sorted(
            t for t in misses if per_task[t]["solved_by"] == "other answer"
        ),
        "nvarc_misses_partial": sorted(t for t in misses if per_task[t]["solved_by"] == "partial"),
        "score": round(scored.score, 3),
        "percent": round(scored.percent, 2),
        "test_outputs": scored.test_outputs,
        "solved_outputs": scored.solved_outputs,
        "verified": len(verified_tasks),
        "verified_rate": round(len(verified_tasks) / len(tasks), 3) if tasks else None,
        "verified_programs": programs,
        "right_programs": right_programs,
        "precision_tasks": round(
            sum(per_task[t]["top_correct"] for t in verified_tasks) / len(verified_tasks), 3
        ) if verified_tasks else None,
        "completion_tokens": tokens,
        "tokens_per_verified": round(tokens / len(verified_tasks)) if verified_tasks else None,
        "turns_mean": round(sum(turns) / len(turns), 2) if turns else None,
        "forcing_turns": sum(row["forcing_turns"] for row in per_task.values()),
        "forced_programs": sum(row["forced_programs"] for row in per_task.values()),
        "guard_rejections": sum(row["guard_rejections"] for row in per_task.values()),
        "sandbox_failures": sum(row["sandbox_failures"] for row in per_task.values()),
        "engine": dict(health),
        "harness": dict(harness),
        "per_task": per_task,
    }  # fmt: skip


def smoke_agent_inputs(run: Run) -> tuple[list[Any], dict[str, Any], str]:
    """Return the smoke path's tasks, their solutions, and a label.

    The fixtures always; DEV4's four tasks too when the competition's evaluation files are
    found under ``--input-root`` (the repository holds no evaluation data, so a local smoke runs
    on the fixtures alone).

    Args:
        run: The run.

    Returns:
        The ``AgentTask`` objects, solutions by task id, and the label.
    """
    from arc26.agent.tasks import load_tasks
    from arc26.data import DEV4

    tasks = list(load_tasks(run.args.smoke_challenges))
    solutions = json.loads(Path(run.args.smoke_solutions).read_text(encoding="utf-8"))
    label = "fixtures"
    settings = run.config["agent"]
    try:
        challenges, solutions_path = resolve_task_files(Path(run.config["input_root"]), settings)
    except Stop:
        return tasks, solutions, label
    tasks += list(load_tasks(challenges, DEV4))
    solutions.update(json.loads(solutions_path.read_text(encoding="utf-8")))
    return tasks, solutions, "fixtures+DEV4"


def route_reading(
    arm: str,
    solved: int,
    fault: bool | str,
    tokens: int,
    tasks: int,
    reading: Mapping[str, Any],
) -> dict[str, Any]:
    """Read one route of run P6.3-04 by its pre-registered rule.

    A route enters P6.5's hybrid design when it solves at least ``enter_misses`` of the NVARC
    misses at a token cost a 12 h run can afford (``affordable_tokens_per_task``: about 54k per
    task at about 300 tokens per second, ``docs/experiments/P6.3-02.md``).

    Args:
        arm: ``"D"`` (direct answers) or ``"L"`` (the long generation).
        solved: The misses the route solved (direct answers for D, verified programs for L).
        fault: False when the arm's health rules hold; True for ``"Fault"``, or the fault's own
            label (``"Fault (rate)"``: arm L's forcing turns cut by its deadline).
        tokens: The arm's generated tokens.
        tasks: The tasks it attempted.
        reading: The config's ``reading`` section.

    Returns:
        ``arm``, ``solved``, ``tokens_per_task``, ``affordable``, and ``reading``: the fault's
        label, ``"enters"``, or ``"does not enter"``.
    """
    per_task = round(tokens / tasks) if tasks else None
    affordable = per_task is not None and per_task <= reading["affordable_tokens_per_task"]
    if fault:
        verdict = fault if isinstance(fault, str) else "Fault"
    elif solved >= reading["enter_misses"] and affordable:
        verdict = "enters"
    else:
        verdict = "does not enter"
    return {"arm": arm, "solved": solved, "tokens_per_task": per_task, "affordable": affordable,
            "reading": verdict}  # fmt: skip


def forcing_cut(tasks: Sequence[Any], records: Sequence[Any]) -> dict[str, Any]:
    """Count arm L's attempts that the deadline left without their forcing turn.

    An attempt reached its end when it verified, its thinking turn stopped by itself, or it got
    its forcing turn; it was cut when it stopped at the deadline with no forcing turn, or never
    started (no record).

    Args:
        tasks: The arm's tasks.
        records: Its ``AttemptRecord`` objects.

    Returns:
        ``attempts`` (the tasks), ``cut``, and ``share``.
    """
    started = {record.task_id for record in records}
    cut = sum(1 for r in records if r.stop_reason == "deadline" and not r.forcing_turns)
    cut += sum(1 for task in tasks if task.task_id not in started)
    return {"attempts": len(tasks), "cut": cut,
            "share": round(cut / len(tasks), 3) if tasks else 0.0}  # fmt: skip


def arm_l_fault(
    gate: Mapping[str, Any], cut: Mapping[str, Any], reading: Mapping[str, Any]
) -> bool | str:
    """Return arm L's fault for its route reading (the brief's rule, not ADR-020's rows).

    Health first: an unhealthy engine (errors, degenerate output, a stall) or harness reads
    ``"Fault"``; then, when more than ``reading.max_cut_share`` of the attempts were cut by the
    deadline before their forcing turn, ``"Fault (rate)"``: the arm measured too little to read.
    Precision does not enter the route rule.

    Args:
        gate: ``agent_metrics``' gate (``healthy``, ``harness``).
        cut: ``forcing_cut``'s result.
        reading: The config's ``reading`` section.

    Returns:
        False, ``True`` (Fault), or ``"Fault (rate)"``.
    """
    if not (gate.get("healthy") and gate.get("harness")):
        return True
    if cut["share"] > reading["max_cut_share"]:
        return "Fault (rate)"
    return False


def long_context_rates(
    phases: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return arm L's rate and its calls' context lengths (64k was never measured before).

    Args:
        phases: ``run_cascade``'s phase report.
        calls: ``engine_calls.jsonl``'s rows.

    Returns:
        Per phase: generated tokens per second of its wall time, and its answered calls'
        context (prompt plus generated tokens) median and maximum.
    """
    rows = []
    for row in phases:
        if not row.get("attempts"):
            continue
        tag = f"/p{row['phase']}/"
        sizes = sorted(
            int(c.get("prompt_tokens") or 0) + int(c.get("completion_tokens") or 0)
            for c in calls
            if tag in str(c.get("attempt_id") or "") and not c.get("error")
        )
        wall = float(row.get("wall_s") or 0)
        rows.append({
            "phase": row["phase"],
            "tokens_per_s": round(row.get("completion_tokens", 0) / wall, 1) if wall else None,
            "context_median": sizes[len(sizes) // 2] if sizes else None,
            "context_max": sizes[-1] if sizes else None,
            "calls": len(sizes),
        })  # fmt: skip
    return {"phases": rows}


class NoRepl:
    """The direct-answer arm's REPL: none. A code block is answered with a note, never run."""

    def run_cell(self, code: str, timeout_s: float) -> Any:
        """Refuse a code block.

        Args:
            code: The block (not run).
            timeout_s: Unused.

        Returns:
            A ``CellResult`` with the note as its error and no ``transform``.
        """
        from arc26.agent.types import CellResult

        note = "no code interpreter in this mode: give the output grids as answer blocks"
        return CellResult("", note, False, 0.0, False)

    def close(self) -> None:
        """Nothing to stop."""


class NoVerifier:
    """The direct-answer arm verifies no program (``NoRepl`` never reports a ``transform``)."""

    replaced = 0

    def verify(self, task: Any, source: str) -> Any:
        """Refuse: the arm has no programs.

        Args:
            task: Unused.
            source: Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("the direct-answer arm verifies no program")

    def close(self) -> None:
        """Nothing to stop."""


VIEW_SUFFIX = "~v1"


def last_answers(record: Any) -> dict[int, Any]:
    """Return an attempt's direct answers by test index, the last answer of each index winning.

    Args:
        record: An ``AttemptRecord``.

    Returns:
        Test index to grid.
    """
    return {int(index): grid for index, grid in record.direct_answers}


def direct_metrics(
    tasks: Sequence[Any],
    records: Sequence[Any],
    view: Any,
    solutions: Mapping[str, Any],
    nvarc_misses: Sequence[str],
    label: str,
    wall_s: float,
    health: Mapping[str, Any],
    reading: Mapping[str, Any],
) -> dict[str, Any]:
    """Score arm D (direct answers, two samples per task) and its route reading.

    Sample 1 answered the task, sample 2 its view (``arc26.agent.views``), whose answers are
    mapped back by the view's inverse. The two are the two attempts per test output (pass@2,
    the competition's rule; ``arc26.evaluate``). A miss is solved when the task scores 1.

    Args:
        tasks: The ``AgentTask`` objects (originals).
        records: Every ``AttemptRecord`` of the arm (views under ``<id>~v1``).
        view: The ``View``.
        solutions: Test outputs by task id (read only after the attempts).
        nvarc_misses: The NVARC misses.
        label: The subset's name.
        wall_s: The arm's wall time.
        health: The engine's health (``engine_call_summary`` with any stalls).
        reading: The config's ``reading`` section.

    Returns:
        The score; ``M_direct``, the misses solved; ``M_view``, those the view alone solved, and
        ``M_counted``, the route's count (the view's when ``contaminated``: the original's
        accuracy at least ``reading.contamination_gap`` above the view's); per sample the share
        of test outputs answered and right and the tasks fully right (the memorization control:
        the original against the view); agreement and precision when the two agree; tokens per task;
        solves per hour of 4x L4; forcing turns; the route reading; one row per task.
    """
    from arc26.evaluate import score_submission

    by_id = {record.task_id: record for record in records}
    submission, per_task = {}, {}
    totals = {"outputs": 0, "answered": [0, 0], "right": [0, 0], "tasks_right": [0, 0],
              "both": 0, "agree": 0, "agree_right": 0}  # fmt: skip
    for task in tasks:
        expected = list(solutions[task.task_id])
        original = by_id.get(task.task_id)
        seen = by_id.get(task.task_id + VIEW_SUFFIX)
        samples = [
            last_answers(original) if original else {},
            {j: view.inverse(g) for j, g in last_answers(seen).items()} if seen else {},
        ]
        rows = []
        right = [0, 0]
        for j, want in enumerate(expected):
            got = [samples[0].get(j), samples[1].get(j)]
            rows.append({"attempt_1": got[0] or [[0]], "attempt_2": got[1] or [[0]]})
            for k in (0, 1):
                totals["answered"][k] += got[k] is not None
                right[k] += got[k] == want
            if got[0] is not None and got[1] is not None:
                totals["both"] += 1
                if got[0] == got[1]:
                    totals["agree"] += 1
                    totals["agree_right"] += got[0] == want
        totals["outputs"] += len(expected)
        for k in (0, 1):
            totals["right"][k] += right[k]
            totals["tasks_right"][k] += right[k] == len(expected)
        submission[task.task_id] = rows
        tokens = sum(r.completion_tokens for r in (original, seen) if r is not None)
        per_task[task.task_id] = {
            "right_original": right[0], "right_view": right[1], "outputs": len(expected),
            "completion_tokens": tokens,
            "forcing_turns": sum(r.forcing_turns for r in (original, seen) if r is not None),
            "stop_reasons": [r.stop_reason for r in (original, seen) if r is not None],
        }  # fmt: skip
    ids = [task.task_id for task in tasks]
    scored = score_submission(submission, solutions, ids, label)
    scores = {t.task_id: t.score for t in scored.per_task}
    for task_id, row in per_task.items():
        row["score"] = round(scores[task_id], 3)
    misses = [t for t in nvarc_misses if t in per_task]
    solved = sorted(t for t in misses if per_task[t]["score"] >= 1)
    by_view = sorted(t for t in misses if per_task[t]["right_view"] == per_task[t]["outputs"])
    outputs = totals["outputs"] or 1
    tokens = sum(row["completion_tokens"] for row in per_task.values())
    accuracy = [round(totals["right"][k] / outputs, 3) for k in (0, 1)]
    fault = not health.get("healthy", False)
    # The brief's pre-registered rule (docs/prerun/P6.3-04.md, section 1): a gap of
    # reading.contamination_gap or more between the samples reads the arm from the view alone.
    gap = totals["right"][0] / outputs - totals["right"][1] / outputs
    contaminated = gap >= reading["contamination_gap"] - 1e-9
    counted = by_view if contaminated else solved
    return {
        "subset": label, "tasks": len(tasks), "view": view.key,
        "score": round(scored.score, 3), "percent": round(scored.percent, 2),
        "test_outputs": scored.test_outputs, "solved_outputs": scored.solved_outputs,
        "M_direct": len(solved), "of": len(misses), "nvarc_misses_solved": solved,
        "M_view": len(by_view), "nvarc_misses_solved_by_view": by_view,
        "score_view": round(
            sum(row["right_view"] / row["outputs"] for row in per_task.values()), 3
        ),
        "contaminated": contaminated,
        "M_counted": len(counted),
        "samples": {
            "original": {"answered": totals["answered"][0], "right": totals["right"][0],
                         "accuracy": accuracy[0], "tasks_right": totals["tasks_right"][0]},
            "view": {"answered": totals["answered"][1], "right": totals["right"][1],
                     "accuracy": accuracy[1], "tasks_right": totals["tasks_right"][1]},
            # The memorization control: a model that recalls published outputs scores far
            # higher on the original than on the view, which no published copy shows.
            "original_minus_view": round(accuracy[0] - accuracy[1], 3),
        },
        "agreement": round(totals["agree"] / totals["both"], 3) if totals["both"] else None,
        "agreed": totals["agree"],
        "precision_when_agreed": (
            round(totals["agree_right"] / totals["agree"], 3) if totals["agree"] else None
        ),
        "completion_tokens": tokens,
        "tokens_per_task": round(tokens / len(tasks)) if tasks else None,
        "wall_s": round(wall_s, 1),
        "solves_per_hour": round(scored.score / (wall_s / 3600), 2) if wall_s > 0 else None,
        "forcing_turns": sum(row["forcing_turns"] for row in per_task.values()),
        "engine": dict(health),
        "route": route_reading("D", len(counted), fault, tokens, len(tasks), reading),
        "per_task": per_task,
    }  # fmt: skip


def _strict_only(record: Any) -> Any:
    """The record with only the answers the strict parser read (P6.9's strict replay)."""
    import dataclasses

    flags = list(record.answer_lenient) or [False] * len(record.direct_answers)
    kept = tuple(a for a, lenient in zip(record.direct_answers, flags, strict=True) if not lenient)
    return dataclasses.replace(record, direct_answers=kept, answer_lenient=())


def shot_attempts(
    record: Any, count: int, shapes: Sequence[Any] | None
) -> tuple[list[dict[str, Any]], list[int]]:
    """Return a shot record's two attempts per test output, as the shot's merge picks them.

    Without ``shapes``: the reply's first two different answers, in order (P6.3-05). With the
    implied shapes (run P6.9-01, the lenient answers): ``arc26.agent.shot.shot_answers``'s
    choice, the first answer of the implied shape first.

    Args:
        record: The ``AttemptRecord``, or None.
        count: The task's test outputs.
        shapes: ``arc26.agent.shot.implied_shapes`` of the task, or None.

    Returns:
        One ``{"attempt_1", "attempt_2"}`` per output (``[[0]]`` where no answer exists), and
        per output the number of answers given (0 to 2).
    """
    from arc26.agent.shot import shot_answers

    rows, given = [], []
    for j in range(count):
        implied = shapes[j] if shapes is not None else None
        found = shot_answers([record] if record else [], j, implied)
        padded = found + [[[0]], [[0]]]
        rows.append({"attempt_1": padded[0], "attempt_2": padded[1]})
        given.append(len(found))
    return rows, given


def shot_metrics(
    tasks: Sequence[Any],
    records: Sequence[Any],
    solutions: Mapping[str, Any],
    label: str,
    wall_s: float,
    health: Mapping[str, Any],
    reading: Mapping[str, Any],
    lenient: bool = False,
) -> dict[str, Any]:
    """Score run P6.3-05's single shot (one sample per task, up to two answers per test input).

    A test output's two attempts are the reply's first two different answers for it, in order
    (the prompt asks for the best first). A task is solved when it scores 1 (``arc26.evaluate``).
    Reading: Fault when the engine's health fails; Fault (rate) when more than
    ``max_cut_share`` of the tasks were cut by the deadline before an answer-forcing turn or never
    started; "enters" when at least ``min_solves`` tasks are solved and at least
    ``margin_over_nvarc`` more than NVARC solved on the same tasks; else "does not enter".

    Run P6.9-01 adds, without changing the above for P6.3-05's configs: with ``lenient`` the
    attempts are :func:`shot_attempts` with the implied shapes, and ``strict`` scores the same
    records with the strict parser's answers and P6.3-05's choice; per output whether each
    attempt is right (``right``) and the implied shape (``implied``), which the merge replay
    (``scripts/merge_replay.py``) reads; per task the turns the server ended at the thinking
    budget (``budget_forced``); and ``forced_answers``, the outputs of the tasks whose thinking
    was cut (by the budget or by the cap, then a forcing turn) and how many of them got an
    answer, the brief's forcing row.

    Args:
        tasks: The tasks.
        records: Their ``AttemptRecord`` objects.
        solutions: Test outputs by task id (read only after the attempts).
        label: The subset's name.
        wall_s: The arm's wall time.
        health: The engine's health (``engine_call_summary`` with any stalls).
        reading: The config's single-shot ``reading`` section.
        lenient: The lenient answers' choice of attempts (run P6.9-01).

    Returns:
        The score, the tasks solved (and by the second attempt alone), NVARC's solves on the same
        tasks, answered outputs, tokens per task, forcing turns, the cut count, the reading, and
        one row per task.
    """
    from arc26.agent.shot import implied_shapes
    from arc26.evaluate import score_submission

    by_id = {record.task_id: record for record in records}
    submission, strict, per_task = {}, {}, {}
    answered = outputs = 0
    cut = {"tasks": 0, "outputs": 0, "answered": 0, "budget_forced": 0, "forcing_turns": 0}
    for task in tasks:
        record = by_id.get(task.task_id)
        expected = list(solutions[task.task_id])
        shapes = implied_shapes(task) if lenient else None
        rows, given = shot_attempts(record, len(expected), shapes)
        answered += sum(bool(g) for g in given)
        outputs += len(expected)
        submission[task.task_id] = rows
        strict[task.task_id] = shot_attempts(
            _strict_only(record) if record else None, len(expected), None
        )[0]
        forced = record.budget_forced if record else 0
        turns = record.forcing_turns if record else 0
        if forced or turns:
            cut["tasks"] += 1
            cut["outputs"] += len(expected)
            cut["answered"] += sum(bool(g) for g in given)
            cut["budget_forced"] += forced
            cut["forcing_turns"] += turns
        per_task[task.task_id] = {
            "first_right": sum(r["attempt_1"] == y for r, y in zip(rows, expected, strict=True)),
            "outputs": len(expected),
            "completion_tokens": record.completion_tokens if record else 0,
            "forcing_turns": turns,
            "stop_reason": record.stop_reason if record else "not started",
        }  # fmt: skip
        if lenient or forced:
            per_task[task.task_id].update(
                budget_forced=forced,
                lenient_answers=sum(record.answer_lenient) if record else 0,
                right=[[r["attempt_1"] == y, r["attempt_2"] == y]
                       for r, y in zip(rows, expected, strict=True)],
                implied=[list(x) if x else None for x in (shapes or [None] * len(expected))],
            )  # fmt: skip
    ids = [task.task_id for task in tasks]
    scored = score_submission(submission, solutions, ids, label)
    for row in scored.per_task:
        per_task[row.task_id]["score"] = round(row.score, 3)
    solved = sorted(t for t, row in per_task.items() if row["score"] >= 1)
    second_only = sorted(t for t in solved if per_task[t]["first_right"] < per_task[t]["outputs"])
    nvarc = sorted(t for t in reading["nvarc_solved"] if t in per_task)
    deadline_cut = forcing_cut(tasks, records)
    tokens = sum(row["completion_tokens"] for row in per_task.values())
    if not health.get("healthy", False):
        verdict = "Fault"
    elif deadline_cut["share"] > reading["max_cut_share"]:
        verdict = "Fault (rate)"
    elif len(solved) >= reading["min_solves"] and len(solved) >= len(nvarc) + reading[
        "margin_over_nvarc"
    ]:
        verdict = "enters"
    else:
        verdict = "does not enter"
    metrics = {
        "subset": label, "tasks": len(tasks), "score": round(scored.score, 3),
        "test_outputs": scored.test_outputs, "solved_outputs": scored.solved_outputs,
        "solved": solved, "solves": len(solved), "solved_by_second_attempt_only": second_only,
        "nvarc_solved": nvarc, "nvarc_solves": len(nvarc), "answered_outputs": answered,
        "completion_tokens": tokens,
        "tokens_per_task": round(tokens / len(tasks)) if tasks else None,
        "wall_s": round(wall_s, 1),
        "forcing_turns": sum(row["forcing_turns"] for row in per_task.values()),
        "forcing_cut": deadline_cut, "engine": dict(health), "reading": verdict,
        "per_task": per_task,
    }  # fmt: skip
    if lenient or cut["budget_forced"]:
        strict_score = score_submission(strict, solutions, ids, f"{label}, strict")
        metrics["strict"] = {
            "score": round(strict_score.score, 3),
            "solved": sorted(t.task_id for t in strict_score.per_task if t.score >= 1),
            "solved_outputs": strict_score.solved_outputs,
        }
        rate = cut["answered"] / cut["outputs"] if cut["outputs"] else None
        metrics["forced_answers"] = {**cut, "rate": None if rate is None else round(rate, 3)}
    return metrics


def direct_scripts(
    tasks: Sequence[Any],
    view: Any,
    solutions: Mapping[str, Any],
    two: bool = False,
    in_request: bool = False,
    lenient: bool = False,
) -> dict[str, list[Any]]:
    """Script P6.2's ``ScriptedEngine`` for the smoke's direct arm.

    ``smoke-mirror``: a reply cut while reasoning, then the right grid in the answer-forcing turn,
    and the right grid on the view (the two agree). ``smoke-transpose``: right on the original,
    the test input on the view (they disagree). ``smoke-recolor``: prose without an answer, then
    right on the view. Other tasks (DEV4): the test input on both (they agree, wrongly).

    With ``two`` (run ``P6.3-05``: no view, two answers per test input), ``smoke-transpose``
    answers the test input first and the right grid second, so it is solved by its second
    attempt. With ``in_request`` (run ``P6.9-01``), ``smoke-mirror``'s one reply holds the
    overlong reasoning and the right answer after it: the engine cuts the reasoning at the
    request's thinking budget and the answer arrives in the same reply. With ``lenient``,
    ``smoke-recolor`` writes the right grid with its rows as digit strings, which only the
    lenient parser reads.

    Args:
        tasks: The original tasks.
        view: The arm's ``View``, or None without a view.
        solutions: The fixtures' test outputs.
        two: Whether a reply may give two answers per test input.
        in_request: Whether the answer is forced inside the thinking request.
        lenient: Whether the lenient answer parser is on.

    Returns:
        Replies per task id (views under ``<id>~v1``).
    """
    from arc26.agent.mock_engine import ScriptedReply, answer_reply, no_code_reply, overlong_reply

    if view is None:
        from arc26.agent.views import View

        view = View((), tuple(range(10)))

    scripts: dict[str, list[Any]] = {}
    for task in tasks:
        own, seen = task.task_id, task.task_id + VIEW_SUFFIX
        guess = task.test_inputs[0]
        right = solutions[own][0] if own in SMOKE_TASKS else guess
        if own == "smoke-mirror" and in_request:
            long = overlong_reply(SMOKE_CUT_CHARS).reasoning
            scripts[own] = [ScriptedReply(text=answer_reply(0, right).text, reasoning=long)]
        elif own == "smoke-mirror":
            scripts[own] = [overlong_reply(SMOKE_CUT_CHARS), answer_reply(0, right)]
            scripts[seen] = [answer_reply(0, view.forward(right))]
        elif own == "smoke-recolor" and lenient:
            rows = ", ".join(json.dumps("".join(str(c) for c in row)) for row in right)
            body = f'{{"test_index": 0, "grid": [{rows}]}}'
            text = f"The ones become twos.\n\n```answer\n{body}\n```\n"
            scripts[own] = [ScriptedReply(text=text)]
        elif own == "smoke-transpose" and two:
            both = answer_reply(0, guess).text + answer_reply(0, right).text
            scripts[own] = [ScriptedReply(text=both)]
        elif own == "smoke-transpose":
            scripts[own] = [answer_reply(0, right)]
            scripts[seen] = [answer_reply(0, view.forward(guess))]
        elif own == "smoke-recolor":
            scripts[own] = [no_code_reply("The rule recolors the ones, I think.")]
            scripts[seen] = [answer_reply(0, view.forward(right))]
        else:
            scripts[own] = [answer_reply(0, guess)]
            scripts[seen] = [answer_reply(0, view.forward(guess))]
    return scripts


def selected_tasks(challenges: Path, task_ids: Sequence[str], subset: str) -> tuple[list[Any], str]:
    """Load exactly the configured tasks of the shot, in the configured order, and their label.

    Runs P6.3-05 and P6.3-06: the tasks come from the whole challenges file through P6.2's
    loader (demonstrations and test inputs only), so they may lie outside ``agent.subset``
    (P6.3-06's 15 are DEV120 tasks outside DEV40), while the forcing check and the harness
    preflight keep the subset's smallest tasks.

    Args:
        challenges: The evaluation challenges file.
        task_ids: ``direct.task_ids``.
        subset: ``agent.subset``: the label when it holds every id, else ``DEV120``.

    Returns:
        The ``AgentTask`` objects and the label, for example ``DEV120, 15 selected``.

    Raises:
        Stop: If an id is not in the challenges file, before any engine call.
    """
    from arc26.agent.tasks import load_tasks
    from arc26.data import subset_ids

    try:
        loaded = load_tasks(challenges, task_ids)
    except ValueError as err:
        raise Stop(f"direct.task_ids: {err} ({challenges.name})") from err
    by_id = {task.task_id: task for task in loaded}
    name = subset if set(task_ids) <= set(subset_ids(subset)) else "DEV120"
    return [by_id[t] for t in task_ids], f"{name}, {len(task_ids)} selected"


async def budget_check(engine: Any, settings: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Check once, before the shot, that the server ends the thinking at a request's budget.

    ``arc26.agent.shot.budget_probe`` (the sanity question at low effort, a budget of 8 tokens),
    whose reasoning must end with ``direct.thinking_end_note`` and be followed by an answer.
    On a failure (an error such as vLLM's HTTP 400 without a reasoning config, or no forced end)
    the direct phase turns ``forcing_in_request`` off for the whole run: ``P6.3-05``'s request
    with the separate forcing turn.

    Args:
        engine: The direct arm's engine.
        settings: The config's ``direct`` section.
        seed: The run's seed.

    Returns:
        The probe's record (``ok``, ``forced``, the reply's and the reasoning's tails, tokens,
        finish, seconds, ``error``).
    """
    from arc26.agent.shot import budget_probe

    return await budget_probe(engine, settings["thinking_end_note"], seed, 600.0)


def direct_phase(run: Run) -> None:
    """Arm D of run P6.3-04: direct answers at low effort, two samples per task (pass@2).

    Every task of the subset (the fixtures, and DEV4 when given, on the smoke path) is asked
    twice through P6.2's loop with the direct-answer system prompt and no REPL: once as it is,
    once through one fixed view (a dihedral transform and a color permutation, the run's seed).
    One thinking turn; a turn cut at its cap gets one answer-forcing turn with thinking off. The
    two samples of a task run next to each other, smallest tasks first, so a deadline cuts whole
    tasks last. The solutions are read after every attempt has ended (BRIEF A0).

    Args:
        run: The run.
    """
    import asyncio

    from arc26.agent.loop import Budget, run_tasks
    from arc26.agent.prompts import DIRECT_SYSTEM_PROMPT, DIRECT_TWO_SYSTEM_PROMPT
    from arc26.agent.tasks import load_tasks
    from arc26.agent.views import fixed_view
    from arc26.data import subset_ids

    settings, agent = run.config["direct"], run.config["agent"]
    view = fixed_view(settings["view_ops"], run.config["seed"]) if settings["with_view"] else None
    two = settings["answers_per_test"] == 2
    solutions_path = None
    if run.args.smoke:
        tasks, solutions, label = smoke_agent_inputs(run)
    else:
        challenges, solutions_path = resolve_task_files(Path(run.config["input_root"]), agent)
        if settings["task_ids"] is not None:
            tasks, label = selected_tasks(challenges, settings["task_ids"], agent["subset"])
        else:
            label = agent["subset"]
            tasks = list(load_tasks(challenges, subset_ids(label)))
        solutions = None
    ordered = []
    for task in sorted(tasks, key=task_cells) if view is not None else tasks:
        ordered += [task, view.task(task, VIEW_SUFFIX)] if view is not None else [task]
    budget = Budget(
        max_turns=1, attempt_tokens=settings["token_budget"], turn_tokens=settings["max_tokens"],
        context_tokens=settings["context_tokens"], reasoning_effort=settings["reasoning_effort"],
        cell_timeout_s=1.0, preserve_thinking=True, forcing_tokens=settings["forcing_tokens"],
        forcing_kind="answer2" if two else "answer",
        forcing_in_request=settings["forcing_in_request"],
        thinking_end_note=settings["thinking_end_note"],
        lenient_answers=settings["lenient_answers"], turn_cut_s=settings["turn_cut_s"],
    )  # fmt: skip
    if run.args.smoke:
        from arc26.agent.mock_engine import ScriptedEngine, ScriptedReply
        from arc26.timing import TimingLogger

        scripts = direct_scripts(tasks, view, solutions, two, settings["forcing_in_request"],
                                 settings["lenient_answers"])  # fmt: skip
        if settings["forcing_in_request"]:
            # The budget check's reply: reasoning past the budget, then the answer.
            reasoning = "Seventeen plus twenty-five is forty-two. " * 8
            scripts["budget-check"] = [ScriptedReply(text="42", reasoning=reasoning)]
        engine: Any = ScriptedEngine(
            scripts,
            timing=TimingLogger(run.out / "timing", "engine"),
            budget_note=settings["thinking_end_note"],
        )  # fmt: skip
    else:
        engine = make_engine(run, settings["concurrency"])
    began, started = time.monotonic(), time.time()
    deadline = min(run.deadline, began + settings["budget_s"])

    async def arm() -> list[Any]:
        nonlocal budget
        try:
            if settings["forcing_in_request"]:
                check = await budget_check(engine, settings, run.config["seed"])
                check["in_request"] = check["ok"]
                if not check["ok"]:
                    # The whole run falls back to P6.3-05's request and its forcing turn.
                    budget = dataclasses.replace(budget, forcing_in_request=False,
                                                 turn_cut_s=None)  # fmt: skip
                    run.problems.append(f"budget check failed, in-request forcing off: {check}")
                run.metrics["budget_check"] = check
                write_json(run.out / "budget_check.json", check)
            return await run_tasks(
                ordered, engine, lambda task: NoRepl(), NoVerifier(), budget,
                concurrency=settings["concurrency"], phase=0, seed=run.config["seed"],
                deadline=deadline, log_path=run.out / "direct_attempts.jsonl",
                transcript_path=run.out / "direct_transcripts.jsonl",
                system=DIRECT_TWO_SYSTEM_PROMPT if two else DIRECT_SYSTEM_PROMPT,
            )  # fmt: skip
        finally:
            closer = getattr(engine, "aclose", None)
            if closer is not None:
                await closer()

    records = asyncio.run(arm())
    wall = time.monotonic() - began
    if solutions is None:
        solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    phase = {"phase": 0, "attempts": len(ordered), "finished": len(records),
             "completion_tokens": sum(r.completion_tokens for r in records),
             "wall_s": round(wall, 1), "started": round(started, 3),
             "ended": round(time.time(), 3)}  # fmt: skip
    health = engine_call_summary(run.out / "engine_calls.jsonl")
    stalls = phase_stalls(
        [phase], read_jsonl(run.out / "engine_calls.jsonl"), agent["min_tokens_per_s"],
        agent["stall_s"],
    )  # fmt: skip
    if stalls:
        health = {**health, "healthy": False, "stalls": stalls}
        run.problems.extend(f"engine: {stall}" for stall in stalls)
    if not health:
        health = {"healthy": True, "note": "no recorded engine calls (the smoke's scripted engine)"}
    if view is None:
        metrics = shot_metrics(
            tasks, records, solutions, label, wall, health, run.config["reading"],
            lenient=settings["lenient_answers"],
        )  # fmt: skip
        metrics["phase"] = phase
        run.metrics["shot"] = metrics
        write_json(run.out / "shot_metrics.json", metrics)
        print(f"shot: score {metrics['score']} of {metrics['tasks']}, solves {metrics['solves']} "
              f"(NVARC {metrics['nvarc_solves']}), reading {metrics['reading']}",
              flush=True)  # fmt: skip
        return
    metrics = direct_metrics(
        tasks, records, view, solutions, agent["nvarc_misses"], label, wall, health,
        run.config["reading"],
    )  # fmt: skip
    metrics["phase"] = phase
    run.metrics["direct"] = metrics
    write_json(run.out / "direct_metrics.json", metrics)
    print(f"direct: score {metrics['score']} of {metrics['tasks']}, M_direct "
          f"{metrics['M_direct']} of {metrics['of']}, original {metrics['samples']['original']}"
          f", view {metrics['samples']['view']}, route {metrics['route']}", flush=True)  # fmt: skip


def agent_phase(run: Run) -> None:
    """Stage 2 (run P6.3-02): the P6.1 and P6.2 harness on a DEV subset, scored in dev mode.

    The evaluation challenges give the tasks (demonstrations and test inputs only, through
    P6.2's loader, which refuses solution files); the solutions are read after every attempt
    has finished, for the score alone (BRIEF A0).

    Args:
        run: The run.
    """
    import asyncio

    from arc26.agent.select import build_selection, submission_from_selection
    from arc26.agent.tasks import count_test_inputs, load_tasks
    from arc26.data import subset_ids

    settings = run.config["agent"]
    if run.args.smoke:
        tasks, solutions, label = smoke_agent_inputs(run)
        kit = smoke_kit(run, settings, tasks, solutions)
    else:
        challenges, solutions_path = resolve_task_files(Path(run.config["input_root"]), settings)
        label = settings["subset"]
        tasks = list(load_tasks(challenges, subset_ids(label)))
        kit = real_kit(run, settings)
        solutions = None
    if settings["only_nvarc_misses"]:
        # Run P6.3-04's arm L: the long generation on the NVARC misses alone.
        misses = set(settings["nvarc_misses"])
        tasks = [task for task in tasks if task.task_id in misses]
        label = f"{label}, NVARC misses"

    async def cascade() -> dict[str, Any]:
        try:
            return await run_cascade(tasks, kit, settings, run)
        finally:
            await kit.aclose()

    try:
        result = asyncio.run(cascade())
    finally:
        replaced = kit.verifier.replaced
        kit.close()
    selection = build_selection(count_test_inputs(tasks), result["records"])
    submission = submission_from_selection(selection)
    write_json(run.out / "agent_submission.json", submission)
    if solutions is None:
        solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    health = engine_call_summary(run.out / "engine_calls.jsonl")
    stalls = phase_stalls(
        result["phases"], read_jsonl(run.out / "engine_calls.jsonl"),
        settings["min_tokens_per_s"], settings["stall_s"],
    )  # fmt: skip
    if stalls:
        health = {**health, "healthy": False, "stalls": stalls}
        run.problems.extend(f"engine: {stall}" for stall in stalls)
    harness = harness_health(result["phases"])
    metrics = agent_metrics(
        tasks, result["records"], selection, submission, solutions, settings["nvarc_misses"],
        label, health, harness,
    )  # fmt: skip
    metrics["phases"] = result["phases"]
    metrics["attempt_errors"] = result["errors"]
    metrics["verifier_workers_replaced"] = replaced
    if "reading" in run.config:
        cut = forcing_cut(tasks, result["records"])
        metrics["forcing_cut"] = cut
        metrics["route"] = route_reading(
            "L", metrics["gate"]["M"],
            arm_l_fault(metrics["gate"], cut, run.config["reading"]),
            metrics["completion_tokens"], len(tasks), run.config["reading"],
        )  # fmt: skip
        metrics["long_context"] = long_context_rates(
            result["phases"], read_jsonl(run.out / "engine_calls.jsonl")
        )
    for fault in harness["faults"]:
        run.problems.append(f"harness: {fault}")
    if harness["faults"] and run.exit_code == EXIT_OK:
        run.exit_code = EXIT_ERROR
    run.metrics["agent"] = metrics
    write_json(run.out / "agent_metrics.json", metrics)
    print(f"agent: score {metrics['score']} of {metrics['tasks']}, gate {metrics['gate']}",
          flush=True)  # fmt: skip


PRESERVE_FOLLOWUP = "Check your rule against every example once more before you answer."


def http_post_json(base_url: str, path: str, body: Mapping[str, Any], timeout_s: float = 60) -> Any:
    """POST a JSON body to a local server and return the parsed JSON reply.

    Args:
        base_url: For example ``http://127.0.0.1:8000``.
        path: For example ``/tokenize``.
        body: The request.
        timeout_s: Socket timeout.

    Returns:
        The parsed reply.

    Raises:
        Stop: On a connection error, a status other than 200, or a reply that is not JSON.
    """
    parts = urllib.parse.urlsplit(base_url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_s)
    try:
        connection.request(
            "POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        text = response.read().decode("utf-8", errors="replace")
        if response.status != 200:
            raise Stop(f"POST {path}: HTTP {response.status}: {text[:300]}")
        return json.loads(text)
    except (OSError, http.client.HTTPException, ValueError) as err:
        raise Stop(f"POST {path} failed: {err}") from err
    finally:
        connection.close()


async def preserve_task(
    run: Run, task: Any, settings: Mapping[str, Any], deadline: float
) -> dict[str, Any]:
    r"""Run the preserve check on one task, through P6.2's engine (one fresh client per variant).

    Turn 1 is the agent's own opening (P6.2's system prompt and task message) with thinking on.
    Turn 2 resends it with the reply and a follow-up, generating one token, in four variants:
    ``off`` (the reply without its reasoning, as P6.2's loop sends it by default), ``on``
    (``Message.reasoning`` sent as ``reasoning_content`` with ``preserve_thinking`` true, P6.2's
    engine), ``on_reasoning_field`` (the same in the ``reasoning`` field), and ``sent_not_kept``
    (reasoning sent, ``preserve_thinking`` forced false). With :math:`R` the reasoning's own
    token count (``/tokenize``) and :math:`P_v` turn 2's ``prompt_tokens``, preserved thinking
    works when :math:`|P_\text{on} - P_\text{off} - R|` and :math:`|P_\text{sent\_not\_kept} -
    P_\text{off}|` are within ``tolerance_tokens``.

    Args:
        run: The run.
        task: An ``AgentTask``.
        settings: The config's ``preserve`` section.
        deadline: A ``time.monotonic()`` value: P6.2's engine has no read timeout, so every call
            is cut there and the task comes back inconclusive.

    Returns:
        The numbers and the verdict (``pass``, ``fail``, or ``inconclusive``).
    """
    import asyncio

    async def bounded(call: Any) -> Any:
        return await asyncio.wait_for(call, timeout=max(0.1, deadline - time.monotonic()))

    from arc26.agent.loop import AttemptContext, attempt_context
    from arc26.agent.prompts import initial_messages
    from arc26.agent.types import Message

    first = initial_messages(task)
    engine = make_engine(run, 4)
    try:
        with attempt_context(AttemptContext(task.task_id, f"{task.task_id}/preserve/0", 0)):
            reply = await bounded(engine.complete(
                first, max_tokens=settings["turn1_max_tokens"], temperature=1.0,
                reasoning_effort=settings["reasoning_effort"], seed=run.config["seed"],
            ))  # fmt: skip
    except TimeoutError:
        return {"task_id": task.task_id, "verdict": "inconclusive: the check's deadline passed"}
    finally:
        await engine.aclose()
    row: dict[str, Any] = {
        "task_id": task.task_id,
        "turn1_completion_tokens": reply.completion_tokens,
        "turn1_finish_reason": reply.finish_reason,
    }
    if not reply.reasoning:
        row["verdict"] = "inconclusive: turn 1 returned no reasoning"
        return row
    counted = http_post_json(
        run.base_url, "/tokenize", {"model": run.config["server"]["served_model_name"],
                                    "prompt": reply.reasoning, "add_special_tokens": False},
    )  # fmt: skip
    reasoning_tokens = int(counted["count"])
    row["reasoning_tokens"] = reasoning_tokens
    if reasoning_tokens <= 2 * settings["tolerance_tokens"]:
        # Growth within the tolerance of such a short reasoning could mean either answer.
        row["verdict"] = "inconclusive: the reasoning is too short to tell kept from dropped"
        return row
    followup = Message("user", PRESERVE_FOLLOWUP)
    variants = {
        "off": ({}, Message("assistant", reply.text)),
        "on": ({}, Message("assistant", reply.text, reply.reasoning)),
        "on_reasoning_field": (
            {"reasoning_field": "reasoning"},
            Message("assistant", reply.text, reply.reasoning),
        ),
        "sent_not_kept": (
            {"template_kwargs": {"preserve_thinking": False}},
            Message("assistant", reply.text, reply.reasoning),
        ),
    }
    prompts = {}
    for name, (overrides, assistant) in variants.items():
        engine = make_engine(run, 4, **overrides)
        try:
            with attempt_context(
                AttemptContext(task.task_id, f"{task.task_id}/preserve/{name}", 0)
            ):
                answer = await bounded(
                    engine.complete([*first, assistant, followup], max_tokens=1, temperature=1.0)
                )
        except TimeoutError:
            row["verdict"] = "inconclusive: the check's deadline passed"
            return row
        finally:
            await engine.aclose()
        prompts[name] = answer.prompt_tokens
    tolerance = settings["tolerance_tokens"]
    growth = {name: prompts[name] - prompts["off"] for name in prompts if name != "off"}
    row.update(
        reasoning_tokens=reasoning_tokens,
        prompt_tokens=prompts,
        growth=growth,
        on_keeps_reasoning=abs(growth["on"] - reasoning_tokens) <= tolerance,
        field_keeps_reasoning=abs(growth["on_reasoning_field"] - reasoning_tokens) <= tolerance,
        off_drops_reasoning=abs(growth["sent_not_kept"]) <= tolerance,
    )
    row["verdict"] = "pass" if row["on_keeps_reasoning"] and row["off_drops_reasoning"] else "fail"
    return row


def preserve_phase(run: Run) -> None:
    """Run the preserve check on the configured number of tasks; a failure is recorded, not fatal.

    The tasks are the subset's smallest (the agent's own order), or the smoke's fixtures. A task
    whose check raises (a dropped connection, an engine or ``/tokenize`` error) is recorded as
    inconclusive with its error and the next task runs; a failure outside the tasks (finding
    them) is recorded the same way. Either way the run goes on to the agent's phases.

    Args:
        run: The run.
    """
    import asyncio

    from arc26.agent.tasks import load_tasks
    from arc26.data import subset_ids

    settings = run.config["preserve"]
    deadline = min(run.deadline, time.monotonic() + settings["budget_s"])
    rows: list[dict[str, Any]] = []

    async def check_all(chosen: Sequence[Any]) -> None:
        for task in chosen:
            if time.monotonic() >= deadline:
                rows.append({"task_id": task.task_id, "verdict": "inconclusive: not started"})
                continue
            try:
                rows.append(await preserve_task(run, task, settings, deadline))
            except Exception as err:  # one task's failure is that task's result
                error = f"{type(err).__name__}: {err}"[:500]
                rows.append(
                    {"task_id": task.task_id, "verdict": f"inconclusive: {error}", "error": error}
                )

    error = None
    try:
        if run.args.smoke:
            tasks = list(load_tasks(run.args.smoke_challenges))
        else:
            agent = run.config["agent"]
            challenges, _ = resolve_task_files(Path(run.config["input_root"]), agent)
            tasks = list(load_tasks(challenges, subset_ids(agent["subset"])))
        asyncio.run(check_all(sorted(tasks, key=task_cells)[: settings["n_tasks"]]))
    except Exception as err:  # recorded; the agent's phases still run
        error = f"{type(err).__name__}: {err}"[:500]
        traceback.print_exc()
    verdicts = [row["verdict"] for row in rows]
    judged = [v for v in verdicts if v in ("pass", "fail")]
    verdict = (
        "inconclusive" if not judged else ("pass" if all(v == "pass" for v in judged) else "fail")
    )
    result: dict[str, Any] = {"verdict": verdict, "tasks": rows}
    if error is not None:
        result["error"] = error
    run.metrics["preserve"] = result
    write_json(run.out / "preserve_check.json", result)
    print(f"preserve check: {verdict} {[row.get('growth') for row in rows]}", flush=True)


class CallLog:
    """An engine wrapper that keeps every call's messages, arguments, and reply.

    The forcing check reads the two calls of its attempt from it: the cut thinking turn and the
    forcing turn.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []

    async def complete(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Forward one call and keep it.

        Args:
            messages: The conversation.
            **kwargs: The call's arguments.

        Returns:
            The engine's reply.
        """
        reply = await self.engine.complete(messages, **kwargs)
        self.calls.append((tuple(messages), dict(kwargs), reply))
        return reply

    async def count_prompt_tokens(self, messages: Sequence[Any], **kwargs: Any) -> int:
        """Forward the opening prompt's count (not kept: it is no engine call).

        Args:
            messages: The conversation.
            **kwargs: The count's arguments.

        Returns:
            The engine's count.
        """
        return await self.engine.count_prompt_tokens(messages, **kwargs)


def defines_transform(code: str) -> bool:
    """Return whether a code block defines ``transform`` at its top level.

    Args:
        code: The block's source.

    Returns:
        True for a top-level ``def transform``; False otherwise or when it does not parse.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "transform"
        for node in tree.body
    )


THINK_EMPTY = "<think>\n\n</think>\n\n"
# Characters of the kept reasoning's start and end the rendered forcing prompt must hold.
RENDER_PROBE_CHARS = 200


def served_prompt(run: Run, prompt: Mapping[str, Any]) -> tuple[str, str]:
    """Return the prompt one chat call fed the model, from that call's own response.

    Why: run ``P6.3-02`` revision 5 (13 Oct) re-rendered the forcing request through vLLM's
    ``/tokenize``, whose path dropped the assistant turn's reasoning that the chat call itself
    had rendered (the call's ``prompt_tokens`` grew by the reasoning's count): the check judged a
    different rendering from the one the model saw. vLLM 0.31's chat response carries the
    prompt's token ids under ``return_token_ids`` (``ChatCompletionResponse.prompt_token_ids``)
    and the templated text under ``return_prompt_text`` (``prompt_text``).

    Args:
        run: The run.
        prompt: ``{"ids": ..., "text": ...}`` as the recording engine kept them for the call.

    Returns:
        The text and its source: the ids decoded by ``/detokenize`` (a tokenizer call, no
        template), else the response's ``prompt_text``.

    Raises:
        ValueError: If the response carried neither, or the ids could not be decoded and no text
            came with them.
    """
    ids, text = prompt.get("ids"), prompt.get("text")
    error = "the response carried no prompt_token_ids"
    if isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids):
        try:
            reply = http_post_json(run.base_url, "/detokenize", {
                "model": run.config["server"]["served_model_name"], "tokens": ids,
            })  # fmt: skip
            decoded = reply["prompt"]
            if isinstance(decoded, str):
                return decoded, "prompt_token_ids"
            error = "/detokenize replied without a prompt string"
        except Exception as err:  # a failed or malformed /detokenize reply
            error = f"/detokenize failed: {type(err).__name__}: {err}"
    if isinstance(text, str) and text:
        return text, "prompt_text"
    raise ValueError(f"{error}, and no prompt_text")


def render_check(run: Run, prompts: Sequence[Mapping[str, Any]], reasoning: str) -> dict:
    """Check the forcing prompt the model actually saw (revision 6).

    It must hold the kept reasoning, start and end, closed by the template's ``</think>`` (the
    cut reasoning has none of its own), and end with the empty think block that turns thinking
    off. Reported, not judged: whether it starts with the thinking turn's whole prompt (the
    prefix the engine's cache can serve), by token ids when both calls carried them.

    Args:
        run: The run.
        prompts: The recording engine's prompts of the thinking turn and the forcing turn.
        reasoning: The cut reasoning the forcing request sent back.

    Returns:
        ``status`` (``ok``, ``failed: <what>``, or ``unavailable: <why>`` when the responses
        carried no prompt or it could not be read: reported, never judged), ``source``,
        ``prefix_kept``, and the prompts' lengths in characters.
    """
    try:
        prompt1, _ = served_prompt(run, prompts[0])
        prompt2, source = served_prompt(run, prompts[1])
    except Exception as err:
        return {"status": f"unavailable: {err}"[:300], "source": None, "prefix_kept": None}
    reasoning = reasoning.strip()
    head, tail = reasoning[:RENDER_PROBE_CHARS], reasoning[-RENDER_PROBE_CHARS:]
    missing = [
        name for name, ok in (
            ("the reasoning's start", "<think>\n" + head in prompt2),
            ("the reasoning's end closed by </think>", tail + "\n</think>\n\n" in prompt2),
            ("the empty think block", prompt2.endswith(THINK_EMPTY)),
        ) if not ok
    ]  # fmt: skip
    ids1, ids2 = prompts[0].get("ids"), prompts[1].get("ids")
    if isinstance(ids1, list) and isinstance(ids2, list) and ids1:
        prefix = ids2[: len(ids1)] == ids1
    else:
        prefix = prompt2.startswith(prompt1)
    status = "ok" if not missing else "failed: no " + ", no ".join(missing)
    return {"status": status, "source": source, "prefix_kept": prefix,
            "chars": [len(prompt1), len(prompt2)]}  # fmt: skip


async def forcing_task(
    run: Run, task: Any, settings: Mapping[str, Any], deadline: float
) -> dict[str, Any]:
    r"""Run the forcing check on one task: a thinking turn cut short, then its forcing turn.

    One P6.2 attempt with the agent's own loop and engine: one round of at most
    ``turn1_max_tokens`` at the check's effort, reasoning kept, and the agent's
    ``forcing_tokens`` for the forcing turn, in a P6.1 REPL with the run's verifier. With
    :math:`P_1` and :math:`P_2` the two calls' ``prompt_tokens``, :math:`R` the cut reasoning's
    own token count and :math:`T` the count of turn 1's text and the forcing message
    (``/tokenize``), the reasoning is kept when :math:`|P_2 - P_1 - R - T|` is within
    ``tolerance_tokens`` (the template's tags, and the effort note the thinking turn's system
    message carries); with the reasoning dropped the growth would miss :math:`R`.

    Args:
        run: The run.
        task: An ``AgentTask``.
        settings: The config's ``forcing`` section.
        deadline: A ``time.monotonic()`` value at which the attempt stops.

    Returns:
        The task's row: turn 1's finish and tokens, the forcing call's switch, finish, tokens,
        and whether its reply holds a complete python block defining ``transform``; the
        attempt's stop reason and verification; the growth check; the render check.
    """
    import asyncio

    from arc26.agent.loop import CODE_TAGS, Budget, fenced_blocks, run_attempt

    agent = run.config["agent"]
    turn1 = settings["turn1_max_tokens"]
    budget = Budget(
        max_turns=1, attempt_tokens=turn1 + agent["forcing_tokens"], turn_tokens=turn1,
        min_turn_tokens=min(256, turn1), context_tokens=agent["context_tokens"],
        reasoning_effort=settings["reasoning_effort"], cell_timeout_s=agent["cell_timeout_s"],
        preserve_thinking=True, forcing_tokens=agent["forcing_tokens"],
    )  # fmt: skip
    factory, verifier = run_sandbox(run, agent)
    loop = asyncio.get_running_loop()
    engine = make_engine(run, 4, capture_prompts=True)
    log = CallLog(engine)
    repl = await loop.run_in_executor(None, factory, task)
    try:
        record = await run_attempt(
            task, log, repl, verifier, budget, attempt_id=f"{task.task_id}/forcing/0", phase=0,
            seed=run.config["seed"], deadline=deadline,
        )  # fmt: skip
    finally:
        await engine.aclose()
        await loop.run_in_executor(None, repl.close)
    row: dict[str, Any] = {
        "task_id": task.task_id, "stop_reason": record.stop_reason, "turns": record.turns,
        "verified": any(v.verified for v in record.verifications),
    }  # fmt: skip
    if not log.calls:
        row["verdict"] = "inconclusive: no engine call answered"
        return row
    first = log.calls[0][2]
    row["turn1"] = {"finish_reason": first.finish_reason, "prompt_tokens": first.prompt_tokens,
                    "completion_tokens": first.completion_tokens}  # fmt: skip
    if len(log.calls) < 2 or log.calls[1][1].get("thinking") is not False:
        row["verdict"] = (
            "inconclusive: turn 1 was not cut" if first.finish_reason != "length"
            else "fail: no forcing turn followed the cut turn"
        )
        return row
    messages, kwargs, forced = log.calls[1]
    blocks = [body for tag, body in fenced_blocks(forced.text) if tag in CODE_TAGS]
    row["forcing"] = {
        "thinking": kwargs.get("thinking"), "max_tokens": kwargs.get("max_tokens"),
        "finish_reason": forced.finish_reason, "prompt_tokens": forced.prompt_tokens,
        "completion_tokens": forced.completion_tokens,
        "program": any(defines_transform(code) for code in blocks),
        "text_head": forced.text[:300],
    }  # fmt: skip
    model = run.config["server"]["served_model_name"]

    def count(text: str) -> int:
        reply = http_post_json(run.base_url, "/tokenize",
                               {"model": model, "prompt": text, "add_special_tokens": False})
        return int(reply["count"])  # fmt: skip

    reasoning = messages[2].reasoning or ""
    tolerance = settings["tolerance_tokens"]
    growth = forced.prompt_tokens - first.prompt_tokens
    try:
        r_tokens = count(reasoning) if reasoning else 0
        t_tokens = count(messages[2].content + messages[3].content)
    except Exception as err:  # a failed or odd /tokenize: judge by the generated count instead
        # The cut reply's generated count bounds its reasoning; a kept reasoning makes the
        # prompt grow by at least that, a dropped one by about the forcing message alone.
        r_tokens = first.completion_tokens
        kept = None if r_tokens <= 2 * tolerance else growth >= r_tokens - tolerance
        row["growth"] = {"prompt_growth": growth, "reasoning_tokens": r_tokens,
                         "other_tokens": None, "kept": kept,
                         "counts": f"generated count; /tokenize failed: {err!r}"[:300]}  # fmt: skip
    else:
        kept = None if r_tokens <= 2 * tolerance else abs(growth - r_tokens - t_tokens) <= tolerance
        row["growth"] = {"prompt_growth": growth, "reasoning_tokens": r_tokens,
                         "other_tokens": t_tokens, "kept": kept, "counts": "tokenize"}  # fmt: skip
    row["render"] = render_check(run, engine.prompts[:2], reasoning)
    if kept is False:
        row["verdict"] = "fail: the reasoning did not reach the forcing prompt"
    elif row["render"]["status"].startswith("failed"):
        row["verdict"] = f"fail: render {row['render']['status']}"
    elif not row["forcing"]["program"]:
        row["verdict"] = "no program: the forced reply has no python block defining transform"
    else:
        row["verdict"] = "pass"
    return row


def forcing_phase(run: Run) -> None:
    """Run the forcing check; on a failure record a Fault row and stop before the agent phases.

    The tasks are the subset's smallest (the agent's own order), or the smoke's fixtures, all at
    once. The check passes when at least ``min_programs`` forced replies hold a complete python
    block defining ``transform``, at least one task shows its reasoning kept, no task shows it
    dropped, and no render check failed (``unavailable`` is reported, not judged). A task whose
    check raises is recorded with its error.

    Args:
        run: The run.

    Raises:
        Stop: If the check fails: the agent's phases would measure a broken turn structure, so
            the run ends here as a Fault (exit 1) and spends no more quota.
    """
    import asyncio

    from arc26.agent.tasks import load_tasks
    from arc26.data import subset_ids

    settings = run.config["forcing"]
    deadline = min(run.deadline, time.monotonic() + settings["budget_s"])
    agent = run.config["agent"]
    if run.args.smoke:
        tasks = list(load_tasks(run.args.smoke_challenges))
    else:
        challenges, _ = resolve_task_files(Path(run.config["input_root"]), agent)
        tasks = list(load_tasks(challenges, subset_ids(agent["subset"])))
    chosen = sorted(tasks, key=task_cells)[: settings["n_tasks"]]

    async def one(task: Any) -> dict[str, Any]:
        try:
            return await forcing_task(run, task, settings, deadline)
        except Exception as err:  # one task's failure is that task's result
            error = f"{type(err).__name__}: {err}"[:500]
            return {"task_id": task.task_id, "verdict": f"inconclusive: {error}", "error": error}

    async def check_all() -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(one(task) for task in chosen)))

    rows = asyncio.run(check_all())
    programs = sum(1 for row in rows if row.get("forcing", {}).get("program"))
    kept = [row["growth"]["kept"] for row in rows if "growth" in row]
    reasons = []
    if programs < settings["min_programs"]:
        reasons.append(f"{programs} forced programs, fewer than {settings['min_programs']}")
    if False in kept:
        reasons.append("a forcing prompt lacked the kept reasoning")
    if True not in kept:
        reasons.append("no task showed its reasoning kept")
    if any(row.get("render", {}).get("status", "").startswith("failed") for row in rows):
        reasons.append("a rendered forcing prompt failed its check")
    result: dict[str, Any] = {
        "verdict": "fail" if reasons else "pass", "reasons": reasons, "programs": programs,
        "min_programs": settings["min_programs"], "tasks": rows,
    }  # fmt: skip
    run.metrics["forcing"] = result
    write_json(run.out / "forcing_check.json", result)
    print(f"forcing check: {result['verdict']}; {programs} programs; {reasons}", flush=True)
    if reasons:
        gate = {"M": None, "of": len(agent["nvarc_misses"]), "precision": None, "healthy": None,
                "harness": None, "row": "Fault"}  # fmt: skip
        run.metrics["agent"] = {"gate": gate, "stopped_by": "forcing check", "reasons": reasons}
        write_json(run.out / "agent_metrics.json", run.metrics["agent"])
        raise Stop(f"forcing check failed ({'; '.join(reasons)}): Fault, no agent phase ran")


def engine_call_summary(path: Path) -> dict[str, Any]:
    """Count the engine calls by where their reasoning arrived (``engine_calls.jsonl``).

    Args:
        path: The JSONL file.

    Returns:
        Calls in all and per location, calls the run's deadline cancelled (``CancelledError``,
        not counted as errors), errors, degenerate and empty replies, the finish reasons
        of all answered calls and of the forcing turns' (thinking off), and the health verdict;
        empty when the file is absent.
    """
    if not path.is_file():
        return {}
    counts: dict[str, int] = {}
    finishes: dict[str, int] = {}
    forcing: dict[str, int] = {}
    total = errors = cancelled = degenerate = empty = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if str(row.get("error") or "").startswith("CancelledError"):
            # The loop's own deadline cut the call (P6.2's wait_for): not an engine failure.
            # Revision 4's 5 failed calls were all this, in flight at phase 2's end.
            cancelled += 1
            continue
        if row.get("error"):
            errors += 1
            continue
        location = row.get("location", "none")
        counts[location] = counts.get(location, 0) + 1
        finish = str(row.get("finish_reason"))
        finishes[finish] = finishes.get(finish, 0) + 1
        if row.get("thinking") is False:
            forcing[finish] = forcing.get(finish, 0) + 1
        degenerate += bool(row.get("degenerate"))
        empty += bool(row.get("empty"))
    answered = total - cancelled
    share = errors / answered if answered else 0.0
    return {
        "calls": total,
        "by_location": counts,
        "cancelled": cancelled,
        "errors": errors,
        "error_share": round(share, 4),
        "degenerate": degenerate,
        "empty": empty,
        "finish": finishes,
        "forcing_finish": forcing,
        # ADR-020 (amended 9 Oct): engine errors on under 10% of calls (those the deadline did
        # not cancel), no degenerate outputs.
        "healthy": share < 0.10 and degenerate == 0,
    }


def finish(run: Run) -> None:
    """Stop the sampler and the server, and write every summary file, whatever happened.

    Args:
        run: The run.
    """
    if run.sandbox is not None:
        run.sandbox[1].close()
    if run.sampler is not None:
        run.sampler.stop()
        run.metrics["peak_memory_mib"] = run.sampler.peak_memory_mib
    if run.server is not None:
        run.metrics["server_exit"] = run.server.stop(run.config["server"]["stop_timeout_s"])
        lines = copy_server_log(run.server.log_path, run.out / "vllm_server.log")
        run.metrics["server_log_lines"] = lines
        if lines and "boot" in run.metrics:
            late = parse_boot_log(run.server.log_path.read_text(encoding="utf-8", errors="replace"))
            run.metrics["log_counts"] = {
                key: late[key] for key in ("tracebacks", "error_lines", "nan_lines")
            }
    calls = engine_call_summary(run.out / "engine_calls.jsonl")
    if calls:
        run.metrics["engine_calls"] = calls
    if (run.out / "prompt_counts.jsonl").is_file():
        run.metrics["prompt_counts"] = count_check(
            read_jsonl(run.out / "prompt_counts.jsonl"), read_jsonl(run.out / "engine_calls.jsonl")
        )
    run.metrics["wall_s"] = round(time.monotonic() - run.start, 1)
    run.metrics["stages"] = run.stages
    run.metrics["problems"] = run.problems
    run.metrics["exit_code"] = run.exit_code
    write_json(run.out / "metrics.json", run.metrics)
    write_json(
        run.out / "run.json",
        {"run_id": run.config["run_id"], "commit": run.args.commit, "smoke": run.args.smoke,
         "exit_code": run.exit_code, "wall_s": run.metrics["wall_s"], "problems": run.problems,
         "phases": run.config["phases"], "stages": run.stages},
    )  # fmt: skip
    (run.out / "summary.md").write_text(summary_markdown(run), encoding="utf-8")
    close_logging(run.log_handlers)


def run_spike(config: dict[str, Any], args: argparse.Namespace) -> int:
    """Run the spike end to end and always write the outputs.

    Args:
        config: The checked config (smoke overrides applied).
        args: The parsed arguments.

    Returns:
        The exit code.
    """
    run = Run(config, args)
    run.out.mkdir(parents=True, exist_ok=True)
    run.scratch.mkdir(parents=True, exist_ok=True)
    (run.out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(run.out / "run.json", {"run_id": config["run_id"], "commit": args.commit,
                                      "started": time.time(), "smoke": args.smoke})  # fmt: skip
    run.log_handlers = configure_logging(run.out)
    try:
        execute(run)
    except Stop as err:
        run.exit_code = EXIT_WALL if isinstance(err, WallTime) else EXIT_ERROR
        run.problems.append(f"{type(err).__name__}: {err}")
    except Exception as err:  # an unexpected failure still writes the logs
        run.exit_code = EXIT_ERROR
        run.problems.append(f"{type(err).__name__}: {err}")
        traceback.print_exc()
    finally:
        finish(run)
    print(f"done: exit {run.exit_code}; {run.problems or 'no problems'}", flush=True)
    return run.exit_code


# --- the smoke path: fixture tasks and a fake OpenAI-compatible server --------------------------


SMOKE_TASKS = {
    "smoke-mirror": {
        "train": [
            {"input": [[1, 2, 0], [0, 3, 4]], "output": [[0, 2, 1], [4, 3, 0]]},
            {"input": [[5, 0], [6, 7]], "output": [[0, 5], [7, 6]]},
        ],
        "test": [{"input": [[8, 9, 1], [2, 0, 3]]}],
        "solution": [[[1, 9, 8], [3, 0, 2]]],
    },
    "smoke-transpose": {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 3], [2, 4]]},
            {"input": [[5, 6, 7], [8, 9, 0]], "output": [[5, 8], [6, 9], [7, 0]]},
        ],
        "test": [{"input": [[2, 4, 6], [1, 3, 5]]}],
        "solution": [[[2, 1], [4, 3], [6, 5]]],
    },
    "smoke-recolor": {
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
            {"input": [[1, 1], [3, 1]], "output": [[2, 2], [3, 2]]},
        ],
        "test": [{"input": [[0, 1], [1, 4]]}],
        "solution": [[[0, 2], [2, 4]]],
    },
}
MIRROR = "def transform(grid):\n    return [list(reversed(row)) for row in grid]\n"
# Characters of the scripted cut reply's reasoning: 2,500 simulated tokens, over the smoke's cap.
SMOKE_CUT_CHARS = 10_000
TRANSPOSE = "def transform(grid):\n    return [list(column) for column in zip(*grid)]\n"
# Per fixture task, the fake server's answer in each round (matched on the first example).
SMOKE_REPLIES = {
    "smoke-mirror": [f"The rows are mirrored.\n\n```python\n{MIRROR}```\n"],
    "smoke-transpose": [
        f"Mirror the rows?\n\n```python\n{MIRROR}```\n",
        f"The grid is transposed.\n\n```python\n{TRANSPOSE}```\n",
    ],
    "smoke-recolor": ["I am not sure of the rule.", f"```python\n{MIRROR}```"],
}


def smoke_overrides(config: dict[str, Any], out: Path, scratch: Path, port: int) -> dict[str, Any]:
    """Return the config of a ``--smoke`` run: tiny cells, three fixture tasks, a fake server.

    Args:
        config: The checked config of the real run.
        out: The smoke's output folder.
        scratch: Its scratch folder.
        port: The fake server's port.

    Returns:
        A checked config with the same phases and kill rules.
    """
    smoke = copy.deepcopy(config)
    smoke.update(max_wall_s=300, out_dir=str(out), scratch_dir=str(scratch))
    smoke["server"].update(port=port, boot_timeout_s=60, stop_timeout_s=5, sample_interval_s=0.5)
    # The smoke's interpreter has no vLLM: the engine-import check imports what it has.
    smoke["wheels"].update(engine_imports=["json", "yaml"], engine_check_timeout_s=60)
    smoke["sanity"].update(max_tokens=16)
    # Smoke outputs are 16 to 24 tokens, so a shorter run marks one degenerate.
    smoke["degenerate"].update(max_run=8)
    if "throughput" in smoke:
        smoke["throughput"].update(
            output_tokens=24, concurrency=[1, 4], min_cell_s=1, warmup_tokens=4
        )
        for index, settings in enumerate(smoke["throughput"]["contexts"].values()):
            settings.update(prompt_tokens=300 * (index + 1), cell_timeout_s=60, budget_s=120)
    if "agent" in smoke:
        smoke["agent"].update(concurrency=4, verifier_workers=2)
        if smoke["agent"]["forcing_tokens"] is not None:
            smoke["agent"].update(forcing_tokens=256)
        for phase in smoke["agent"]["phases"]:
            phase.update(max_tokens=1024, token_budget=4096, budget_s=120)
    if "preserve" in smoke:
        smoke["preserve"].update(n_tasks=2, turn1_max_tokens=64, budget_s=60)
    if "direct" in smoke:
        smoke["direct"].update(concurrency=4, max_tokens=1024, token_budget=4096,
                               forcing_tokens=256, budget_s=120)  # fmt: skip
        if smoke["direct"].get("turn_cut_s") is not None:
            smoke["direct"]["turn_cut_s"] = 60
    if "forcing" in smoke:
        # The fake server's reasoning is 56 pieces: 40 cut it, and its 40 pieces are about 54
        # fake tokens, over twice the tolerance, so a dropped reasoning shows.
        smoke["forcing"].update(n_tasks=len(SMOKE_TASKS), turn1_max_tokens=40,
                                tolerance_tokens=16, budget_s=60)  # fmt: skip
    if "agent" in smoke:
        # Every fixture counts as an NVARC miss here, so the smoke shows which solves reach M.
        smoke["agent"].update(nvarc_misses=sorted(SMOKE_TASKS))
    if "probe" in smoke:
        smoke["probe"].update(n_tasks=len(SMOKE_TASKS), max_task_chars=10_000, max_tokens=512,
                              budget_s=120, checker_timeout_s=10)  # fmt: skip
    return check_config(smoke)


def write_smoke_inputs(folder: Path) -> tuple[Path, Path, Path]:
    """Write the fixture tasks, their solutions, and the fake server's replies.

    Args:
        folder: Where to write them.

    Returns:
        The challenges, solutions, and replies files.
    """
    folder.mkdir(parents=True, exist_ok=True)
    challenges = {k: {"train": v["train"], "test": v["test"]} for k, v in SMOKE_TASKS.items()}
    solutions = {k: v["solution"] for k, v in SMOKE_TASKS.items()}
    replies = [
        {"match": render_grid(SMOKE_TASKS[k]["train"][0]["input"]), "replies": v}
        for k, v in SMOKE_REPLIES.items()
    ]
    paths = (folder / "challenges.json", folder / "solutions.json", folder / "replies.json")
    for path, value in zip(paths, (challenges, solutions, replies), strict=True):
        write_json(path, value)
    return paths


def fake_tokens(text: str) -> int:
    """Return the fake engine's token count of a text: one token per 4 characters, rounded up.

    Args:
        text: The text.

    Returns:
        The count.
    """
    return (len(text) + 3) // 4


def fake_effort_note(body: Mapping[str, Any]) -> str | None:
    """Return the effort note Qwen3.8's template would put at the head of the system message.

    Args:
        body: A chat completion or ``/tokenize`` request.

    Returns:
        P6.2's ``QWEN38_EFFORT_NOTES`` entry of the request's effort (top level or template
        variable) while thinking is on, else None.
    """
    from arc26.agent.engine import QWEN38_EFFORT_NOTES

    template = body.get("chat_template_kwargs") or {}
    if template.get("enable_thinking") is False:
        return None
    effort = template.get("reasoning_effort", body.get("reasoning_effort"))
    return QWEN38_EFFORT_NOTES.get(effort) if isinstance(effort, str) else None


def fake_prompt_tokens(body: Mapping[str, Any], honor_preserve: bool = True) -> int:
    """Count a request's prompt tokens as the chat template would render it, in fake tokens.

    Each message costs its content plus 4; an assistant message's reasoning (``reasoning`` or
    ``reasoning_content``) is rendered, plus 4 for the think tags, only when
    ``chat_template_kwargs.preserve_thinking`` is true, as Qwen3.8's template does for turns
    before the last user message; with thinking on, a system message also carries the effort's
    note (``fake_effort_note``).

    Args:
        body: The chat completion request.
        honor_preserve: False renders no reasoning at all, as a server path that loses it would
            (the fake's ``drop`` mode, which the preserve and forcing checks must catch).

    Returns:
        The count.
    """
    keep = honor_preserve and bool(
        (body.get("chat_template_kwargs") or {}).get("preserve_thinking")
    )
    note = fake_effort_note(body)
    total = 0
    for k, message in enumerate(body["messages"]):
        total += fake_tokens(message.get("content") or "") + 4
        if k == 0 and message.get("role") == "system" and note:
            total += fake_tokens(note + "\n\n")
        thought = message.get("reasoning") or message.get("reasoning_content")
        if keep and message.get("role") == "assistant" and thought:
            total += fake_tokens(thought) + 4
    return total


def fake_render(body: Mapping[str, Any], honor_preserve: bool = True) -> str:
    """Render a request's prompt as Qwen3.8's chat template does, for the fake ``/tokenize``.

    Args:
        body: A ``/tokenize`` request with ``messages``.
        honor_preserve: False drops every kept reasoning (the fake's ``drop`` mode).

    Returns:
        The text: each message in ``<|im_start|>role ... <|im_end|>``, an assistant message's
        reasoning in a think block while ``preserve_thinking`` is true, the effort's note at the
        head of the system message while thinking is on, and the generation prompt (an open
        think block, or the empty one when thinking is off).
    """
    template = body.get("chat_template_kwargs") or {}
    keep = honor_preserve and bool(template.get("preserve_thinking"))
    note = fake_effort_note(body)
    parts = []
    for k, message in enumerate(body["messages"]):
        role, content = message["role"], (message.get("content") or "").strip()
        if k == 0 and role == "system" and note:
            content = f"{note}\n\n{content}"
        if role == "assistant" and keep:
            thought = (message.get("reasoning") or message.get("reasoning_content") or "").strip()
            content = f"<think>\n{thought}\n</think>\n\n{content}"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if body.get("add_generation_prompt", True):
        off = template.get("enable_thinking") is False
        parts.append("<|im_start|>assistant\n" + (THINK_EMPTY if off else "<think>\n"))
    return "".join(parts)


def fake_server_main(
    port: int, replies_path: str, mode: str = "normal", model: str = "fake"
) -> int:
    """Serve a fake OpenAI-compatible engine for the smoke test (stdlib only).

    It answers ``/health``, ``/v1/models``, ``/metrics``, ``/tokenize`` (a text's count, or a
    conversation rendered by ``fake_render`` as one token per character), ``/detokenize``, and
    ``/v1/chat/completions``, streaming or not: a prompt whose first user message contains a
    fixture's first example input gets that fixture's scripted reply for the round (one round
    per user message), after 56 pieces of reasoning unless thinking is off, cut at
    ``max_tokens`` pieces with finish ``length``; any other prompt gets ``max_tokens`` synthetic
    tokens, half as reasoning. Streamed chunks carry ``token_ids`` and the last one ``usage``; a
    non-streamed reply carries its reasoning in ``reasoning``, as vLLM 0.31's do, and its prompt
    tokens follow ``fake_prompt_tokens``.

    Args:
        port: The port on 127.0.0.1.
        replies_path: JSON list of ``{"match", "replies"}``.
        mode: ``normal``; ``degenerate`` to emit token 0 over and over (NaN's symptom), which
            the tests use to fire the kill criterion; ``inline`` to put a non-streamed reply's
            reasoning in ``content`` before ``</think>`` (a server without a reasoning parser);
            ``drop`` to render no reasoning sent back (the preserve and forcing checks must
            fail); ``prose`` to answer a call with thinking off in prose, without code (the
            forcing check must fail); or ``badtokenize`` to answer ``/tokenize`` and
            ``/detokenize`` without their fields (counts and renders fall back; no Fault); or
            ``noprompt`` to send chat replies without ``prompt_token_ids`` and ``prompt_text``
            (the render check is unavailable; no Fault).
        model: The served model name.

    Returns:
        0 after a SIGTERM.
    """
    replies = json.loads(Path(replies_path).read_text(encoding="utf-8"))
    counters = {"prompt_tokens": 0, "generation_tokens": 0, "requests": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
            print("fake: " + format % args, flush=True)

        def _send(self, status: int, body: str, kind: str = "application/json") -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802, D102
            if self.path == "/health":
                self._send(200, "")
            elif self.path == "/v1/models":
                self._send(200, json.dumps({"data": [{"id": model}]}))
            elif self.path == "/metrics":
                with lock:
                    text = (
                        f"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                        f"vllm:kv_cache_usage_perc 0.1\n"
                        f"vllm:prompt_tokens_total {counters['prompt_tokens']}\n"
                        f"vllm:generation_tokens_total {counters['generation_tokens']}\n"
                        'vllm:cache_config_info{block_size="16",num_gpu_blocks="100"} 1.0\n'
                    )
                self._send(200, text, "text/plain")
            else:
                self._send(404, "{}")

        def do_POST(self) -> None:  # noqa: N802, D102
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path in ("/tokenize", "/detokenize") and mode == "badtokenize":
                # A reply without its field: the counts and the render must fall back.
                self._send(200, json.dumps({"detail": "no such field"}))
                return
            if self.path == "/tokenize":
                if "messages" in body:
                    # One token per character: /detokenize maps them back exactly.
                    tokens = [ord(c) for c in fake_render(body, honor_preserve=mode != "drop")]
                    self._send(200, json.dumps({"count": len(tokens), "tokens": tokens}))
                    return
                count = fake_tokens(body.get("prompt", ""))
                self._send(200, json.dumps({"count": count, "max_model_len": 65536, "tokens": []}))
                return
            if self.path == "/detokenize":
                self._send(200, json.dumps({"prompt": "".join(map(chr, body["tokens"]))}))
                return
            users = [m["content"] for m in body["messages"] if m["role"] == "user"]
            prompt = fake_prompt_tokens(body, honor_preserve=mode != "drop")
            text = None
            for rule in replies:
                if users and rule["match"] in users[0]:
                    text = rule["replies"][min(len(users), len(rule["replies"])) - 1]
            if text is None:
                pieces = [("reasoning" if i < body["max_tokens"] // 2 else "content", f"t{i % 7} ")
                          for i in range(body["max_tokens"])]  # fmt: skip
                finish = "length"
            else:
                words = re.findall(r"\S+\s*|\s+", text)
                # A reasoning of about 75 fake tokens in 56 pieces: long enough for the preserve
                # and forcing checks, and none with thinking off.
                thought = re.findall(r"\S+\s*", "Let me compare each input with its output. " * 7)
                off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
                if off and mode == "prose":
                    words = re.findall(r"\S+\s*", "The rule mirrors each row, I believe.")
                pieces = [] if off else [("reasoning", w) for w in thought]
                pieces += [("content", w) for w in words]
                finish = "stop"
                if len(pieces) > body["max_tokens"]:
                    pieces, finish = pieces[: body["max_tokens"]], "length"
            with lock:
                counters["prompt_tokens"] += prompt
                counters["generation_tokens"] += len(pieces)
                counters["requests"] += 1
            if not body.get("stream"):
                thought = "".join(piece for field, piece in pieces if field == "reasoning")
                answer = "".join(piece for field, piece in pieces if field == "content")
                message: dict[str, Any] = {"role": "assistant", "content": answer}
                if mode == "inline":
                    message["content"] = f"<think>\n{thought}\n</think>\n\n{answer}"
                else:
                    message["reasoning"] = thought
                choice: dict[str, Any] = {"index": 0, "message": message, "finish_reason": finish}
                if body.get("return_token_ids"):
                    choice["token_ids"] = [
                        0 if mode == "degenerate" else 1000 + i % 50 for i in range(len(pieces))
                    ]
                reply = {
                    "choices": [choice],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": len(pieces)},
                }
                if mode != "noprompt":
                    # The prompt the model saw, as vLLM 0.31 returns it: token ids (one per
                    # character here) and, on request, the templated text.
                    seen = fake_render(body, honor_preserve=mode != "drop")
                    if body.get("return_token_ids"):
                        reply["prompt_token_ids"] = [ord(c) for c in seen]
                    if body.get("return_prompt_text"):
                        reply["prompt_text"] = seen
                self._send(200, json.dumps(reply))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def emit(event: dict[str, Any]) -> None:
                self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
                self.wfile.flush()

            emit({"choices": [{"index": 0, "delta": {"role": "assistant"}, "token_ids": []}]})
            for index, (field, piece) in enumerate(pieces):
                token = 0 if mode == "degenerate" else 1000 + index % 50
                emit({"choices": [{"index": 0, "delta": {field: piece}, "token_ids": [token]}]})
                time.sleep(0.002)
            emit({"choices": [{"index": 0, "delta": {}, "finish_reason": finish, "token_ids": []}]})
            usage = {"prompt_tokens": prompt, "completion_tokens": len(pieces)}
            emit({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(f"fake engine on port {port}", flush=True)
    server.serve_forever()
    return 0


# --- command line --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", help="configs/P6.3-01.yaml")
    parser.add_argument("--commit", default="unknown", help="the code's commit, for run.json")
    parser.add_argument("--model-root", help="mount folder of the model notebook's output")
    parser.add_argument("--wheelhouse-root", help="mount folder of the wheels notebook's output")
    parser.add_argument("--input-root", help="overrides input_root")
    parser.add_argument("--out-dir", help="overrides out_dir")
    parser.add_argument("--scratch-dir", help="overrides scratch_dir")
    parser.add_argument(
        "--venv-python",
        help="an existing python with vLLM installed (a plain GPU machine): skips the install",
    )
    parser.add_argument("--resume-from", help="a previous out_dir: skip its finished work")
    parser.add_argument("--smoke", action="store_true", help="CPU run against a fake server")
    parser.add_argument("--fake-server", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--fake-replies", help=argparse.SUPPRESS)
    parser.add_argument("--fake-mode", default="normal", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-wall-s", type=int, help="overrides max_wall_s (after the smoke settings)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the spike, its smoke path, or the fake server.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``. Unknown arguments are
            ignored (a notebook kernel's argv can carry an IPython connection file).

    Returns:
        The exit code (see ``EXIT_*``).
    """
    args, _ = build_parser().parse_known_args(argv)
    if args.fake_server:
        return fake_server_main(args.port, args.fake_replies, args.fake_mode)
    try:
        if not args.config:
            raise ConfigError("--config is required")
        config = load_config(Path(args.config))
        for key in ("input_root", "out_dir", "scratch_dir"):
            value = getattr(args, key)
            if value:
                config[key] = value
        # Absolute paths everywhere: the engine runs in the scratch folder, so a relative path
        # on its command line would point elsewhere.
        for key in ("input_root", "out_dir", "scratch_dir"):
            config[key] = os.path.abspath(config[key])
        for key in (
            "model_root", "wheelhouse_root", "venv_python", "resume_from", "out_dir",
            "scratch_dir", "input_root",
        ):  # fmt: skip
            value = getattr(args, key)
            if value:
                setattr(args, key, os.path.abspath(value))
        args.smoke_challenges = args.smoke_solutions = None
        if args.smoke:
            if not args.out_dir:
                raise ConfigError("--smoke needs --out-dir (it never writes to /kaggle)")
            out = Path(args.out_dir)
            scratch = Path(args.scratch_dir or out / "scratch")
            challenges, solutions, replies = write_smoke_inputs(scratch / "inputs")
            args.smoke_challenges, args.smoke_solutions = str(challenges), str(solutions)
            args.fake_replies = str(replies)
            config = smoke_overrides(config, out, scratch, free_port())
        elif not args.model_root or (not args.wheelhouse_root and not args.venv_python):
            raise ConfigError(
                "a GPU run needs --model-root and --wheelhouse-root (or --venv-python)"
            )
        if args.max_wall_s is not None:
            config["max_wall_s"] = args.max_wall_s
    except ConfigError as err:
        print(f"refused: {err}", file=sys.stderr)
        return EXIT_CONFIG
    return run_spike(config, args)


if __name__ == "__main__":
    sys.exit(main())
