r"""Command line of the agent solver for development and competition runs (P6.4).

Why: the scored rerun must end with a valid ``submission.json`` whatever happens on the way: an
engine that never boots, a crash mid-run, a kill. This entry point orders the work so that a
valid file exists from the first seconds on, and every later step only improves it.

Usage::

    python -m arc26.agent.run --mode competition --tasks <challenges.json> \
        --out /kaggle/working/submission.json --config configs/agent_default.yaml \
        --engine-venv <venv> --model-dir <weights> --session-start <epoch seconds> \
        [--nvarc-submission <file>] [--nvarc-candidates <candidates.json>] [--log-dir <dir>]
    python -m arc26.agent.run --mode dev ... --engine-url http://127.0.0.1:8000 \
        --solutions <solutions.json>
    python -m arc26.agent.run --mode dev ... --fake-engine <replies.json>     # the CPU smoke

The engine is either started here from a venv (``--engine-venv`` and ``--model-dir``, the
config's ``server`` flags), taken from a running server (``--engine-url``, its root), or the fake
engine of :mod:`arc26.agent.server` (``--fake-engine``, CPU runs and tests).

Order of work:

1. Read the config and the arguments; a refusal exits 2 and writes nothing.
2. Read the tasks; NVARC's submission (from the run's own copy in the log folder, made at the
   first start and never overwritten, so a resume never reads this run's output as NVARC's) and
   candidates when given (an unreadable one is reported and skipped); the attempt log of an
   earlier run in the same log folder (resume).
3. Write ``submission.json`` at once: NVARC's attempts (or ``[[0]]``) with the resumed attempts
   on top. From here on the file is valid at every moment.
4. Start the engine (one more start if it crashes during its boot) or check the URL; start the
   verifier pool; run :class:`~arc26.agent.scheduler.Scheduler` to the deadline, rewriting the
   file after every attempt.
5. Stop the engine and the sandboxes and write the final file; in development mode, score it.

Deadline: the session ends at ``--deadline`` (Unix time), or ``session_s`` after
``--session-start`` (Unix time, the notebook's start). Competition mode requires one of the two,
since this process starts after NVARC's stage and cannot know how much of the session is gone.
Development mode may give ``--budget-s`` (the end, seconds from now) or nothing: the end is then
``session_s - startup_margin_s`` from this process's start. Generation stops ``final_margin_s``
before the end.

Exit codes: 0 when a valid ``submission.json`` is in place at the end, whatever failed on the way
(each problem is in ``run.json`` and the log); 1 when no valid file could be written; 2 when the
arguments or the config are refused.

Competition mode never opens a solutions file: ``--solutions`` is refused there, and the one
function that reads solutions (:func:`score_dev`) refuses any mode but development.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from arc26.agent.scheduler import (
    GovernorSettings,
    PhaseSettings,
    Scheduler,
    SchedulerSettings,
    SubmissionWriter,
    nvarc_confidence,
)
from arc26.agent.server import (
    THINK_END,
    EngineBootTimeout,
    EngineServer,
    EngineStartError,
    EngineSupervisor,
    ServerConfig,
    engine_env,
    fake_command,
    http_get,
    reasoning_end_str,
    vllm_command,
)

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_NO_SUBMISSION, EXIT_REFUSED = 0, 1, 2
DEV, COMPETITION = "dev", "competition"
ATTEMPTS_FILE = "attempts.jsonl"
TRANSCRIPTS_FILE = "transcripts.jsonl"
ENGINE_LOG = "engine.log"
ENGINE_PID = "engine.pid"
NVARC_COPY = "nvarc_submission.json"
SUMMARY_FILE = "run.json"
MIB = 1024 * 1024

TOP_KEYS = (
    "seed", "deadline", "concurrency", "sandbox", "governor", "scheduler", "phases", "engine",
    "server", "logs",
)  # fmt: skip
DEADLINE_KEYS = ("session_s", "startup_margin_s", "final_margin_s", "grace_s")
SANDBOX_KEYS = (
    "verifier_workers", "call_timeout_s", "memory_mb", "cell_timeout_s", "max_cells",
    "feedback_chars",
)  # fmt: skip
GOVERNOR_KEYS = ("prior_rate", "window_s", "min_calls", "rate_safety")
SCHEDULER_KEYS = (
    "engine_error_retries", "cycle_error_rounds", "min_pairs_passed", "min_turn_tokens",
    "nvarc_views", "prior_programs", "prior_chars",
)  # fmt: skip
PHASE_KEYS = (
    "attempts", "min_attempt_tokens", "max_attempt_tokens", "turn_tokens", "max_turns",
    "reasoning_effort", "temperature", "share", "preserve_thinking", "prior",
)  # fmt: skip
ENGINE_KEYS = (
    "effort_transport", "thinking", "connect_timeout_s", "max_retries", "backoff_s",
    "backoff_max_s",
)  # fmt: skip
SERVER_KEYS = (
    "host", "port", "served_model_name", "tensor_parallel_size", "kv_cache_dtype",
    "enable_prefix_caching", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "gpu_memory_utilization", "attention_backend", "language_model_only", "reasoning_parser",
    "generation_config", "extra_args", "env", "unset_env", "boot_timeout_s", "stop_timeout_s",
    "health_interval_s", "monitor_interval_s", "hang_s", "restarts",
)  # fmt: skip
LOGS_KEYS = ("dev_transcripts",)


class ConfigError(ValueError):
    """The config or the arguments are refused; nothing ran and nothing was written."""


@dataclass(frozen=True)
class RunConfig:
    """The run config (``configs/agent_default.yaml``), checked.

    Attributes:
        seed: The run's seed.
        session_s: The session's length in seconds (Kaggle: 12 h).
        startup_margin_s: Seconds of the session spent before this process starts.
        final_margin_s: Seconds kept after generation stops, to stop and write.
        concurrency: Attempts in flight.
        verifier_workers: Verifier processes.
        call_timeout_s: Seconds per ``transform`` call in the verifier.
        memory_bytes: Address-space limit of every REPL and program process.
        scheduler: The scheduler's settings (phases, governor, limits).
        nvarc_views: Views NVARC decodes per test input (ADR-014's :math:`n`).
        engine: Keyword arguments of :class:`~arc26.agent.engine.EngineConfig`.
        server: The engine server's settings.
        dev_transcripts: Write every conversation to a JSONL in development mode.
    """

    seed: int
    session_s: float
    startup_margin_s: float
    final_margin_s: float
    concurrency: int
    verifier_workers: int
    call_timeout_s: float
    memory_bytes: int
    scheduler: SchedulerSettings
    nvarc_views: int
    engine: Mapping[str, Any]
    server: ServerConfig
    dev_transcripts: bool


def _section(raw: object, keys: Sequence[str], where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    found = {str(k) for k in raw}
    missing, unknown = sorted(set(keys) - found), sorted(found - set(keys))
    if missing or unknown:
        raise ConfigError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return {str(k): v for k, v in raw.items()}


def config_from_mapping(raw: object) -> RunConfig:
    """Check a parsed run config and build its settings.

    Every key of every section is required and no other key is accepted, so a typo fails at
    once instead of running with a default.

    Args:
        raw: The parsed YAML.

    Returns:
        The config.

    Raises:
        ConfigError: If a section or a key is missing or unknown, or a value is out of range.
    """
    top = _section(raw, TOP_KEYS, "config")
    try:
        deadline = _section(top["deadline"], DEADLINE_KEYS, "deadline")
        sandbox = _section(top["sandbox"], SANDBOX_KEYS, "sandbox")
        governor = GovernorSettings(**_section(top["governor"], GOVERNOR_KEYS, "governor"))
        scheduler = _section(top["scheduler"], SCHEDULER_KEYS, "scheduler")
        if not isinstance(top["phases"], list) or len(top["phases"]) != 4:
            raise ConfigError("phases must be a list of 4 phases")
        phases = tuple(
            PhaseSettings(**_section(p, PHASE_KEYS, f"phases[{i}]"))
            for i, p in enumerate(top["phases"])
        )
        server = _section(top["server"], SERVER_KEYS, "server")
        server["extra_args"] = tuple(str(a) for a in server["extra_args"] or ())
        server["unset_env"] = tuple(str(a) for a in server["unset_env"] or ())
        server["env"] = {str(k): str(v) for k, v in (server["env"] or {}).items()}
        server_config = ServerConfig(**server)
        settings = SchedulerSettings(
            phases=phases,
            governor=governor,
            concurrency=int(top["concurrency"]),
            seed=int(top["seed"]),
            context_tokens=server_config.max_model_len,
            cell_timeout_s=float(sandbox["cell_timeout_s"]),
            max_cells=int(sandbox["max_cells"]),
            feedback_chars=int(sandbox["feedback_chars"]),
            min_turn_tokens=int(scheduler["min_turn_tokens"]),
            engine_error_retries=int(scheduler["engine_error_retries"]),
            cycle_error_rounds=int(scheduler["cycle_error_rounds"]),
            min_pairs_passed=int(scheduler["min_pairs_passed"]),
            prior_programs=int(scheduler["prior_programs"]),
            prior_chars=int(scheduler["prior_chars"]),
            grace_s=float(deadline["grace_s"]),
        )
        engine = _section(top["engine"], ENGINE_KEYS, "engine")
        logs = _section(top["logs"], LOGS_KEYS, "logs")
        config = RunConfig(
            seed=int(top["seed"]),
            session_s=float(deadline["session_s"]),
            startup_margin_s=float(deadline["startup_margin_s"]),
            final_margin_s=float(deadline["final_margin_s"]),
            concurrency=settings.concurrency,
            verifier_workers=int(sandbox["verifier_workers"]),
            call_timeout_s=float(sandbox["call_timeout_s"]),
            memory_bytes=int(sandbox["memory_mb"]) * MIB,
            scheduler=settings,
            nvarc_views=int(scheduler["nvarc_views"]),
            engine=engine,
            server=server_config,
            dev_transcripts=bool(logs["dev_transcripts"]),
        )
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError) as err:
        raise ConfigError(f"config: {err}") from err
    if config.session_s <= config.startup_margin_s + config.final_margin_s:
        raise ConfigError("deadline: session_s must exceed startup_margin_s + final_margin_s")
    if config.verifier_workers < 1 or config.nvarc_views < 1 or config.memory_bytes < MIB:
        raise ConfigError("verifier_workers, nvarc_views, and memory_mb must be positive")
    if config.engine["effort_transport"] not in ("request", "template"):
        raise ConfigError("engine.effort_transport must be request or template")
    return config


def load_config(path: str | os.PathLike[str]) -> RunConfig:
    """Read and check a run config.

    Args:
        path: The YAML file.

    Returns:
        The config.

    Raises:
        ConfigError: If the file cannot be read or parsed, or :func:`config_from_mapping`
            refuses it.
    """
    import yaml

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"cannot read the config {path}: {err}") from err
    return config_from_mapping(raw)


@dataclass(frozen=True)
class RunOptions:
    """One run's resolved arguments and config.

    Attributes:
        mode: ``dev`` or ``competition``.
        tasks: The challenges file.
        out: The submission file.
        config: The checked config.
        log_dir: Where the attempt log, the engine log, and ``run.json`` go.
        engine_kind: ``vllm``, ``url``, or ``fake``.
        engine_url: The server root for ``url``.
        engine_venv: The venv holding ``bin/vllm`` for ``vllm``.
        model_dir: The weights for ``vllm``.
        fake_replies: The fake engine's replies for ``fake``.
        fake_options: The fake engine's test options (delay, crash, failed boot).
        nvarc_submission: NVARC's ``submission.json``, or None.
        nvarc_candidates: NVARC's ``candidates.json``, or None.
        solutions: The solutions file (development mode only), or None.
        task_ids: Task ids to run, or None for all.
        session_end: The session's end, Unix time.
        deadline_source: Where the end comes from: ``deadline``, ``session_start``,
            ``budget_s``, or ``default`` (development mode: from this process's start).
    """

    mode: str
    tasks: Path
    out: Path
    config: RunConfig
    log_dir: Path
    engine_kind: str
    engine_url: str | None
    engine_venv: Path | None
    model_dir: Path | None
    fake_replies: Path | None
    fake_options: Mapping[str, Any] = field(default_factory=dict)
    nvarc_submission: Path | None = None
    nvarc_candidates: Path | None = None
    solutions: Path | None = None
    task_ids: tuple[str, ...] | None = None
    session_end: float = 0.0
    deadline_source: str = "default"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m arc26.agent.run", description=__doc__.split("\n", 1)[0]
    )
    parser.add_argument("--mode", choices=(DEV, COMPETITION), required=True)
    parser.add_argument("--tasks", required=True, help="the challenges file")
    parser.add_argument("--out", required=True, help="the submission.json to keep valid")
    parser.add_argument("--config", required=True, help="configs/agent_default.yaml")
    parser.add_argument("--log-dir", help="attempt log, engine log, run.json (default: by --out)")
    engine = parser.add_mutually_exclusive_group(required=True)
    engine.add_argument("--engine-url", help="root URL of a running server")
    engine.add_argument("--engine-venv", help="venv with vLLM: start bin/vllm from it")
    engine.add_argument("--fake-engine", help="replies JSON: start the fake engine (CPU)")
    parser.add_argument("--model-dir", help="the weights folder, with --engine-venv")
    parser.add_argument("--port", type=int, help="override the config's server port")
    parser.add_argument("--nvarc-submission", help="NVARC's submission.json: the fill")
    parser.add_argument("--nvarc-candidates", help="NVARC's candidates.json: phase 2's order")
    parser.add_argument("--solutions", help="development mode only: score the result")
    parser.add_argument("--task-ids", help="comma-separated task ids to run (default: all)")
    deadline = parser.add_mutually_exclusive_group()
    deadline.add_argument("--deadline", type=float, help="the session's end, Unix time")
    deadline.add_argument(
        "--session-start", type=float, help="the session's start, Unix time (end: + session_s)"
    )
    deadline.add_argument(
        "--budget-s", type=float, help="development mode: the session's end, seconds from now"
    )
    parser.add_argument("--fake-delay-s", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--fake-crash-after", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--fake-crash-marker", help=argparse.SUPPRESS)
    parser.add_argument("--fake-fail-boot", action="store_true", help=argparse.SUPPRESS)
    return parser


def resolve_options(
    args: argparse.Namespace,
    now: float | None = None,
    load: Callable[[str], Any] = load_config,
) -> RunOptions:
    """Check the arguments and read the config.

    Args:
        args: The parsed arguments.
        now: The current Unix time (tests pass one).
        load: Reads and checks the config file; :func:`load_config` here, the shot's own
            (:mod:`arc26.agent.shot`) there. Its config needs ``server``, ``session_s``, and
            ``startup_margin_s``.

    Returns:
        The options.

    Raises:
        ConfigError: If competition mode is given a solutions file, lacks ``--deadline`` and
            ``--session-start``, or is given ``--budget-s``; the vLLM start lacks its model
            folder; or the config is refused.
    """
    if args.mode == COMPETITION and args.solutions:
        raise ConfigError("competition mode never opens a solutions file; drop --solutions")
    if args.mode == COMPETITION and args.deadline is None and args.session_start is None:
        raise ConfigError(
            "competition mode needs --deadline or --session-start: this process starts after "
            "NVARC's stage and cannot tell how much of the session is gone"
        )
    if args.engine_venv and not args.model_dir:
        raise ConfigError("--engine-venv needs --model-dir")
    config = load(args.config)
    if args.port is not None:
        try:
            server = dataclasses.replace(config.server, port=args.port)
        except ValueError as err:
            raise ConfigError(str(err)) from err
        config = dataclasses.replace(config, server=server)
    out = Path(args.out).resolve()
    now = time.time() if now is None else now
    if args.deadline is not None:
        end, source = args.deadline, "deadline"
    elif args.session_start is not None:
        end, source = args.session_start + config.session_s, "session_start"
    elif args.budget_s is not None:
        end, source = now + args.budget_s, "budget_s"
    else:
        end, source = now + config.session_s - config.startup_margin_s, "default"
    kind = "url" if args.engine_url else "vllm" if args.engine_venv else "fake"
    url = None if args.engine_url is None else args.engine_url.rstrip("/").removesuffix("/v1")
    ids = None
    if args.task_ids:
        ids = tuple(t.strip() for t in args.task_ids.split(",") if t.strip())
    return RunOptions(
        mode=args.mode,
        tasks=Path(args.tasks).resolve(),
        out=out,
        config=config,
        log_dir=Path(args.log_dir).resolve() if args.log_dir else out.parent / "agent_logs",
        engine_kind=kind,
        engine_url=url,
        engine_venv=None if args.engine_venv is None else Path(args.engine_venv).resolve(),
        model_dir=None if args.model_dir is None else Path(args.model_dir).resolve(),
        fake_replies=None if args.fake_engine is None else Path(args.fake_engine).resolve(),
        fake_options={
            "delay_s": args.fake_delay_s,
            "crash_after": args.fake_crash_after,
            "crash_marker": args.fake_crash_marker,
            "fail_boot": args.fake_fail_boot,
        },
        nvarc_submission=_path(args.nvarc_submission),
        nvarc_candidates=_path(args.nvarc_candidates),
        solutions=_path(args.solutions),
        task_ids=ids,
        session_end=end,
        deadline_source=source,
    )


def _path(value: str | None) -> Path | None:
    return None if value is None else Path(value).resolve()


def score_dev(
    mode: str, submission: Path, solutions: Path, task_ids: Sequence[str]
) -> dict[str, Any]:
    """Score a submission against a solutions file, in development mode only.

    Why: the only place a solutions file is opened; competition mode must never reach it
    (BRIEF A0), so it refuses any other mode before touching the file.

    Args:
        mode: The run's mode; anything but ``dev`` is refused.
        submission: The submission file.
        solutions: The solutions file.
        task_ids: The tasks of the run.

    Returns:
        ``score`` and ``tasks`` (the per-task metric with its task count), ``solved_outputs``,
        ``test_outputs``, and ``report`` (the evaluator's lines).

    Raises:
        PermissionError: If ``mode`` is not ``dev``.
    """
    if mode != DEV:
        raise PermissionError("competition mode never opens a solutions file")
    from arc26.evaluate import load_predictions, load_solutions, score_submission

    answers = load_solutions(solutions)
    scored = score_submission(
        load_predictions(submission), answers, [t for t in task_ids if t in answers], "run"
    )
    return {
        "score": scored.score,
        "tasks": scored.tasks,
        "solved_outputs": scored.solved_outputs,
        "test_outputs": scored.test_outputs,
        "report": scored.report(per_task=True),
    }


def _read_json(path: Path | None, what: str, problems: list[str]) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        problems.append(f"{what} {path} skipped: {err}")
        logger.error("%s %s is unreadable; going on without it: %s", what, path, err)
        return None


def nvarc_copy(source: Path | None, log_dir: Path, problems: list[str]) -> tuple[Path | None, Any]:
    """Read NVARC's submission from the run's own copy, made at the first start.

    Why: the notebook may name the same file for NVARC's submission and for this run's output,
    which this run rewrites; a resumed run must still see NVARC's attempts, not its own. So the
    first start copies NVARC's file into the log folder (before the first write of the output),
    and every start reads the copy, which is never overwritten. A source that cannot be read at
    the first start leaves a copy holding ``null``: later starts then run without NVARC rather
    than read a file that may by now be this run's output.

    Args:
        source: NVARC's ``submission.json`` as given, or None.
        log_dir: The run's log folder.
        problems: Where problems are reported.

    Returns:
        The copy's path (None when no NVARC file was ever given or it could not be copied) and
        its parsed content (None when there is none).
    """
    from arc26.submit import write_json_atomic

    copy = log_dir / NVARC_COPY
    if not copy.exists():
        if source is None:
            return None, None
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            problems.append(f"NVARC's submission {source} skipped: {err}")
            logger.error("NVARC's submission %s is unreadable; going on without it", source)
            data = None
        try:
            write_json_atomic(copy, data)
        except (OSError, TypeError, ValueError) as err:
            problems.append(f"NVARC's submission could not be copied to {copy}: {err}")
            return None, data
        return copy, data
    data = _read_json(copy, "NVARC's copy", problems)
    if data is None and copy.exists():
        problems.append("NVARC's copy holds no submission (unreadable at the first start)")
    return copy, data


def _valid_on_disk(path: Path, counts: Mapping[str, int]) -> bool:
    from arc26.submit import validate

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return not validate(data, {task_id: {"test": [{}] * n} for task_id, n in counts.items()})


def _engine_server(options: RunOptions) -> EngineServer | None:
    config = options.config.server
    if options.engine_kind == "url":
        return None
    if options.engine_kind == "vllm":
        bin_dir = options.engine_venv / "bin"
        command = vllm_command(bin_dir / "vllm", str(options.model_dir), config)
        env = engine_env(os.environ, config, str(bin_dir))
    else:
        fake = dict(options.fake_options)
        end = reasoning_end_str(config.extra_args)
        command = fake_command(
            config.port,
            options.fake_replies,
            model=config.served_model_name,
            boot_delay_s=0.0,
            delay_s=fake.get("delay_s") or 0.0,
            crash_after=fake.get("crash_after"),
            crash_marker=fake.get("crash_marker"),
            fail_boot=bool(fake.get("fail_boot")),
            budget_note=end.removesuffix(THINK_END) if end else None,
        )
        env = engine_env(os.environ, config)
    return EngineServer(
        command,
        env,
        options.log_dir / ENGINE_LOG,
        config.root_url,
        boot_timeout_s=config.boot_timeout_s,
        stop_timeout_s=config.stop_timeout_s,
        health_interval_s=config.health_interval_s,
        pid_path=options.log_dir / ENGINE_PID,
    )


async def _boot(
    server: EngineServer, deadline: float, restarts: int, summary: dict[str, Any]
) -> int | None:
    """Start the engine; one more start if it crashed during its boot. Return restarts left."""
    for attempt in range(1 + min(restarts, 1)):
        left = deadline - time.monotonic()
        if left <= 0:
            summary["problems"].append("the deadline passed before the engine was up")
            return None
        try:
            seconds = await asyncio.to_thread(server.start, left)
        except EngineBootTimeout as err:
            summary["problems"].append(f"engine start {attempt + 1}: {err}")
            return None
        except EngineStartError as err:
            summary["problems"].append(f"engine start {attempt + 1}: {err}")
            continue
        summary["engine"]["boot_s"] = round(seconds, 1)
        return restarts - attempt
    return None


async def _url_ready(root: str, timeout_s: float, interval_s: float) -> bool:
    """Wait until a server given by URL answers ``/health`` with 200."""
    end = time.monotonic() + max(0.0, timeout_s)
    while True:
        status, _ = await asyncio.to_thread(http_get, root, "/health", 10.0)
        if status == 200:
            return True
        if time.monotonic() >= end:
            return False
        await asyncio.sleep(min(interval_s, max(0.0, end - time.monotonic())))


async def _solve(
    options: RunOptions,
    server: EngineServer | None,
    tasks: Sequence[Any],
    writer: SubmissionWriter,
    resumed: Sequence[Any],
    confidence: Mapping[str, float],
    deadline: float,
    summary: dict[str, Any],
) -> None:
    """Start the engine and the sandboxes, run the scheduler, and stop everything again."""
    from arc26.agent.engine import EngineConfig, OpenAIEngine
    from arc26.agent.sandbox import SubprocessRepl
    from arc26.agent.verify import PoolVerifier
    from arc26.timing import TimingLogger

    config = options.config
    problems: list[str] = summary["problems"]
    root = config.server.root_url if server is not None else str(options.engine_url)
    supervisor: EngineSupervisor | None = None
    engine: OpenAIEngine | None = None
    verifier: PoolVerifier | None = None
    try:
        if server is not None:
            restarts = await _boot(server, deadline, config.server.restarts, summary)
            if restarts is None:
                return
        else:
            limit = min(config.server.boot_timeout_s, deadline - time.monotonic())
            if not await _url_ready(root, limit, config.server.health_interval_s):
                problems.append(f"the engine at {root} did not answer healthy")
                return
            restarts = 0
        supervisor = EngineSupervisor(
            root,
            server,
            interval_s=config.server.monitor_interval_s,
            hang_s=config.server.hang_s,
            restarts=restarts,
            deadline=deadline,
        )
        supervisor.start()
        verifier = await asyncio.to_thread(
            PoolVerifier,
            config.verifier_workers,
            call_timeout_s=config.call_timeout_s,
            memory_bytes=config.memory_bytes,
        )
        engine_settings = dict(config.engine)
        engine = OpenAIEngine(
            EngineConfig(
                base_url=f"{root}/v1",
                model=config.server.served_model_name,
                max_connections=max(config.concurrency, 1),
                **engine_settings,
            ),
            timing=TimingLogger(options.log_dir, "engine"),
        )
        transcripts = None
        if options.mode == DEV and config.dev_transcripts:
            transcripts = options.log_dir / TRANSCRIPTS_FILE
        scheduler = Scheduler(
            tasks,
            engine,
            functools.partial(SubprocessRepl, memory_bytes=config.memory_bytes),
            verifier,
            config.scheduler,
            deadline=deadline,
            records=resumed,
            confidence=confidence,
            log_path=options.log_dir / ATTEMPTS_FILE,
            transcript_path=transcripts,
            on_record=writer.add,
            gate=supervisor,
        )
        try:
            await scheduler.run()
        finally:
            summary["scheduler"] = scheduler.report.as_dict()
            if scheduler.report.failed:
                problems.append(
                    f"{scheduler.report.failed} attempts raised (a REPL or verifier fault) and "
                    "left no record; see the log"
                )
    finally:
        if supervisor is not None:
            await supervisor.close()
            summary["engine"]["events"] = supervisor.events
            if supervisor.reason:
                problems.append(f"engine: {supervisor.reason}")
        if engine is not None:
            with contextlib.suppress(Exception):
                await engine.aclose()
        if verifier is not None:
            await asyncio.to_thread(verifier.close)
        if server is not None:
            summary["engine"]["exit_code"] = await asyncio.to_thread(server.stop)


def run(options: RunOptions) -> int:
    """Run the agent solver with resolved options; always leave a valid submission if possible.

    Args:
        options: The options.

    Returns:
        The exit code: 0 with a valid submission at the end, 1 without one.
    """
    started = time.monotonic()
    deadline = started + (options.session_end - time.time()) - options.config.final_margin_s
    summary: dict[str, Any] = {
        "mode": options.mode,
        "tasks_file": str(options.tasks),
        "out": str(options.out),
        "engine": {"kind": options.engine_kind},
        "deadline_in_s": round(deadline - started, 1),
        "deadline_source": options.deadline_source,
        "problems": [],
    }
    problems: list[str] = summary["problems"]
    try:
        options.log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:  # the logs are not worth the submission: run with temporary ones
        fallback = _fallback_log_dir(options.out)
        problems.append(f"log folder {options.log_dir} unusable ({err}); logs in {fallback}")
        options = dataclasses.replace(options, log_dir=fallback)
    from arc26.agent.loop import read_records
    from arc26.agent.select import nvarc_from_submission
    from arc26.agent.tasks import count_test_inputs, load_tasks

    try:
        tasks = load_tasks(options.tasks, options.task_ids)
    except (OSError, ValueError) as err:
        problems.append(f"tasks: {err}")
        logger.error("cannot read the tasks: %s", err)
        _write_summary(options, summary, started)
        return EXIT_NO_SUBMISSION
    counts = count_test_inputs(tasks)
    summary["tasks"] = len(counts)
    nvarc = None
    copy, raw = nvarc_copy(options.nvarc_submission, options.log_dir, problems)
    if raw is not None:
        try:
            nvarc = nvarc_from_submission(raw)
        except ValueError as err:
            problems.append(f"NVARC's submission skipped: {err}")
    summary["nvarc"] = None if copy is None else {"copy": str(copy), "tasks": len(nvarc or {})}
    candidates = _read_json(options.nvarc_candidates, "NVARC's candidates", problems)
    if candidates is not None and not isinstance(candidates, Mapping):
        problems.append("NVARC's candidates skipped: not a JSON object")
        candidates = None
    try:
        confidence = nvarc_confidence(counts, candidates, nvarc, options.config.nvarc_views)
    except Exception as err:  # the order is a refinement; a bad file must not stop the run
        problems.append(f"NVARC's confidence skipped: {err}")
        confidence = {}
    resumed: list[Any] = []
    log = options.log_dir / ATTEMPTS_FILE
    if log.exists():
        try:
            resumed, skipped = read_records(log)
            summary["resume"] = {"records": len(resumed), "skipped_lines": skipped}
        except OSError as err:
            problems.append(f"the attempt log could not be read: {err}")
    writer = SubmissionWriter(options.out, counts, nvarc, resumed)
    if not writer.write():
        problems.append(f"the first write of {options.out} failed: {writer.last_error}")
    server = _engine_server(options)
    try:
        asyncio.run(
            _solve(options, server, tasks, writer, resumed, confidence, deadline, summary)
        )
    except BaseException as err:  # a bug, a signal, an interrupt: the final write still happens
        problems.append(f"{type(err).__name__}: {err}")
        logger.error("the run stopped early:\n%s", traceback.format_exc())
    finally:
        if server is not None:
            with contextlib.suppress(Exception):
                server.stop()
        writer.write()
    valid = _valid_on_disk(options.out, counts)
    summary["submission"] = {
        "valid": valid,
        "writes": writer.writes,
        "failed_writes": writer.failures,
        "slots": writer.slots(),
    }
    if options.mode == DEV and options.solutions is not None:
        try:
            summary["score"] = score_dev(options.mode, options.out, options.solutions, list(counts))
            logger.info("score: %s", summary["score"]["report"][:3])
        except (OSError, ValueError) as err:
            problems.append(f"scoring failed: {err}")
    _write_summary(options, summary, started)
    if not valid:
        logger.error("no valid submission at %s", options.out)
        return EXIT_NO_SUBMISSION
    return EXIT_OK


def _fallback_log_dir(out: Path) -> Path:
    """A log folder under the temporary directory, the same for every start with this output.

    Why: a resume must find the attempt log and NVARC's copy of the first start again.
    """
    digest = hashlib.sha256(str(out).encode("utf-8")).hexdigest()[:12]
    folder = Path(tempfile.gettempdir()) / f"arc26-agent-logs-{digest}"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        folder = Path(tempfile.mkdtemp(prefix="arc26-agent-logs-"))
    return folder


def _write_summary(options: RunOptions, summary: dict[str, Any], started: float) -> None:
    from arc26.submit import write_json_atomic

    summary["wall_s"] = round(time.monotonic() - started, 1)
    try:
        write_json_atomic(options.log_dir / SUMMARY_FILE, summary)
    except (OSError, TypeError, ValueError) as err:
        logger.error("cannot write %s: %s", SUMMARY_FILE, err)


def _raise_exit(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the agent solver from the command line.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        The exit code (``EXIT_*``).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        options = resolve_options(args)
    except SystemExit as err:  # argparse's own refusal
        return EXIT_REFUSED if err.code else EXIT_OK
    except ConfigError as err:
        print(f"refused: {err}", file=sys.stderr)
        return EXIT_REFUSED
    # SIGTERM (a notebook stopping the run) runs the cleanup and the final write, as Ctrl-C does.
    previous = signal.signal(signal.SIGTERM, _raise_exit)
    try:
        return run(options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
