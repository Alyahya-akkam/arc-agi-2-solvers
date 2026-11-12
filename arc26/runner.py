"""Launch the solver and write its submission (tasks P0.1 and P2.0, ADR-009 and ADR-011).

The reference notebook launches ``starter.py`` from a shell line (cell 6) and then selects the
attempts inline (cell 7). This module is that launch and that selection as functions for the P0.5
notebook: ``RunConfig`` holds the values the notebook hard-codes (paths, DEV4 filter, worker
count, LoRA rank and target modules), ``run_solver`` runs the solver's module (``RunConfig.solver``)
with the reference's launch environment, and ``make_submission`` runs the moved selection cell. With
``timing_dir`` set, the solver's workers and ``make_submission`` append JSONL timing records there
(task P0.3b, ``arc26.timing``). The LoRA settings default to the reference's (task P1.3).
``RunConfig.solver`` picks the solver's code (task P2.0, ADR-011): ``"split"``, the default since
run ``P2.0-01`` passed, the reference's statements split into ``arc26.ttt`` and ``arc26.decode``,
started as ``python -m arc26.solver``; ``"nvarc"``, the moved reference, the parity anchor.
``RunConfig.holdout`` (task P2.1) holds one demonstration pair per puzzle out of training and
prompts and runs the held-out gate (``arc26.loop``), on the split solver only; its ``rounds``
(task P2.3) add the refinement rounds after the gate, gradient or in context (task P2.5).
``RunConfig.adapters`` (task P3.7, ADR-019) runs the split solver's arms entry, ``arc26.arms``,
instead of ``arc26.solver``: every puzzle under several adapter sources, paired. See
``docs/modules/runner.md``. The opt-in ``hyper_checkpoint``, ``hyper_sha256`` and
``hyper_log_dir`` fields route the original solver scope through ``arc26.h_submission``:
a pinned H adapter is merged before each puzzle's ordinary training, then restored exactly.
"""

import codecs
import json
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from arc26.arms import Arms
from arc26.loop import HoldOut
from arc26.timing import MAIN_SELECTION, TimingLogger

# Cell 6 of the reference launches the solver as
# `!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12
# python starter.py --end-time {global_end_time}`.
LAUNCH_ENV = {
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "OMP_NUM_THREADS": "12",
}
# The solver's process entry per RunConfig.solver (task P2.0, ADR-011). Both take the same
# arguments and run the same statements; "split" runs them as arc26.ttt.fit and
# arc26.decode.decode_puzzle. "split" is the default since the split solver passed the GPU
# parity run P2.0-01 (docs/experiments/P2.0-01.md) and the human approved the flip (2 Oct).
SOLVER_MODULES = {"nvarc": "arc26.nvarc.starter", "split": "arc26.solver"}
# The split solver's arms entry (task P3.7, ADR-019): the module RunConfig.adapters starts.
ARMS_MODULE = "arc26.arms"
TASK_ID = re.compile(r"[0-9a-f]{8}")
# Python accepts PYTHONHASHSEED values from 0 to 2**32 - 1 and refuses to start otherwise.
MAX_HASH_SEED = 2**32 - 1
CHUNK_BYTES = 64 * 1024
# After a timeout kill, output already in the pipe is still copied, for at most this long.
DRAIN_SECONDS = 5.0
# Worker label of the process that runs make_submission: its records go to timing-main.jsonl,
# beside the solver workers' timing-rank<rank>.jsonl.
MAIN_WORKER = "main"
# The reference's LoRA settings (cell 4, `worker`, `peft_params`): rank 256 on these nine target
# modules, in this order. The seven linear projections of every layer get LoRA factors; Unsloth
# trains embed_tokens and lm_head as full copies instead (PEFT modules_to_save,
# docs/experiments/P1.1-01.md).
LORA_R = 256
LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
)  # fmt: skip
LORA_PROJECTIONS = LORA_TARGETS[:7]


@dataclass(frozen=True)
class RunConfig:
    """Settings of one solver run: the values the reference notebook hard-codes.

    Attributes:
        data_dir: Competition data directory, holding ``arc-agi_*_challenges.json``.
        model_dir: Base model directory, holding ``config.json``.
        outputs_dir: Folder where the workers write one bz2 pickle of candidates per augmented
            test input; the reference's ``/kaggle/inference_outputs`` by default.
        marker_dir: Folder where worker ``rank`` writes the file ``worker{rank}`` once it has
            imported unsloth, so that worker ``rank + 1`` may start (the reference's workaround
            for unsloth patching); the reference's ``/kaggle`` by default.
        subset: Task ids solved outside a competition rerun; None solves every task. A list is
            stored as a tuple. A competition rerun always solves every task, as in the reference.
        workers: Worker processes, one per GPU (worker ``rank`` sees GPU ``rank`` only).
        hash_seed: ``PYTHONHASHSEED`` of the solver process and its workers; it fixes the
            rescoring augmentation seeds ``hash(bk) % 1024**2`` (flagged conflict 5).
        timing_dir: Folder of the JSONL timing records (``arc26.timing``): each solver worker
            writes ``timing-rank<rank>.jsonl`` and ``make_submission`` writes
            ``timing-main.jsonl``. None, the default, records nothing, as the reference.
        lora_r: Rank of the per-task LoRA adapter; the reference's 256 by default. The scale
            stays rsLoRA's ``lora_alpha / sqrt(lora_r)`` with the reference's alpha of 32.
        lora_targets: Module names the adapter trains, taken from ``LORA_TARGETS``, the
            reference's nine, which are the default; a list is stored as a tuple. At least one of
            the seven linear projections is required, because Unsloth trains ``embed_tokens``
            and ``lm_head`` as full copies, not as LoRA factors.
        solver: The solver's code, a key of ``SOLVER_MODULES``: ``"split"``, the default, runs
            the reference's statements split into ``arc26.ttt`` and ``arc26.decode``
            (``arc26.solver``, task P2.0, ADR-011); ``"nvarc"`` runs the moved reference
            (``arc26.nvarc.starter``). It picks the module only: the solver's ``--config`` JSON is
            the same for both.
        holdout: The hold-out settings (task P2.1, ``arc26.loop.HoldOut``), or None, the
            default, to train and decode on every pair as the reference does. A mapping of its
            fields is stored as a ``HoldOut``. Set, it reaches the solver as the JSON's
            ``holdout`` and needs ``solver`` ``"split"``: the moved reference cannot train and
            decode apart (ADR-011). Its optional ``rounds`` (task P2.3, ``arc26.loop.Rounds``)
            runs the refinement rounds after the gate: ``tau``, ``max_rounds``,
            ``continue_steps``, the governor's config, whether a final decode with the
            held-out pair back in its prompts follows them (default on, ADR-013 as amended after
            ``P2.1-02``), and the round variants offered (task P2.5, ``variants``: the gradient
            round by default, the in-context round forced, or both); absent, no round runs and
            the JSON is task P2.1's.
        adapters: The adapter arms (task P3.7, ``arc26.arms.Arms``), or None, the default, to
            run one adapter source, the reference's fit. A mapping of its fields is stored as
            ``Arms``. Set, it reaches the solver as the JSON's ``adapters``, and
            ``solver_command`` starts ``arc26.arms`` (``ARMS_MODULE``) instead of
            ``arc26.solver``; it needs ``solver`` ``"split"`` and no ``holdout`` (no loop,
            ADR-018), and its folders must lie outside ``outputs_dir``.

    Raises:
        TypeError: If a directory is not a string (``timing_dir``: None or a string), ``subset``
            is not None, a list, or a tuple of strings, ``workers``, ``hash_seed``, or
            ``lora_r`` is not an integer, ``lora_targets`` is not a list or tuple of strings,
            ``solver`` is not a string, ``holdout`` is not None, a ``HoldOut``, or a mapping
            of its fields (or a field has the wrong type), or ``adapters`` is not None, an
            ``Arms``, or a mapping of its fields.
        ValueError: If a directory is empty, ``subset`` is empty or holds an id that is not
            eight lowercase hex digits, ``workers < 1``, ``hash_seed`` is outside
            ``[0, 2**32 - 1]``, ``lora_r < 1``, ``lora_targets`` holds a name outside
            ``LORA_TARGETS``, repeats a name, or has no linear projection, ``solver`` is not
            a key of ``SOLVER_MODULES``, ``holdout`` is set with the ``"nvarc"`` solver, a
            ``holdout`` field is invalid, or its gate, log, or round folder lies inside
            ``outputs_dir``, or ``adapters`` is set with the ``"nvarc"`` solver or with
            ``holdout``, is invalid, or has a folder inside ``outputs_dir``.
    """

    data_dir: str
    model_dir: str
    outputs_dir: str = "/kaggle/inference_outputs"
    marker_dir: str = "/kaggle"
    subset: tuple[str, ...] | None = None
    workers: int = 4
    hash_seed: int = 0
    timing_dir: str | None = None
    lora_r: int = LORA_R
    lora_targets: tuple[str, ...] = LORA_TARGETS
    solver: str = "split"
    holdout: HoldOut | None = None
    adapters: Arms | None = None
    hyper_checkpoint: str | None = None
    hyper_sha256: str | None = None
    hyper_log_dir: str | None = None

    def __post_init__(self) -> None:
        if self.hyper_checkpoint is not None:
            if self.solver != "split" or self.holdout is not None or self.adapters is not None:
                raise ValueError("H submission requires split solver without holdout or arms")
            for name in ("hyper_checkpoint", "hyper_log_dir"):
                if not isinstance(getattr(self, name), str) or not getattr(self, name):
                    raise ValueError(f"{name} must be a nonempty path")
            if not isinstance(self.hyper_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", self.hyper_sha256
            ):
                raise ValueError("hyper_sha256 must pin the checkpoint with 64 hex digits")
            if os.path.commonpath([
                os.path.abspath(self.hyper_log_dir), os.path.abspath(self.outputs_dir)
            ]) == os.path.abspath(self.outputs_dir):
                raise ValueError("H logs must be outside candidate outputs_dir")
        elif self.hyper_sha256 is not None or self.hyper_log_dir is not None:
            raise ValueError("H hash and logs require hyper_checkpoint")
        for name in ("data_dir", "model_dir", "outputs_dir", "marker_dir", "timing_dir"):
            value = getattr(self, name)
            if name == "timing_dir" and value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string path; got {value!r}")
            if not value:
                raise ValueError(f"{name} must not be empty")
        if self.subset is not None:
            if not isinstance(self.subset, list | tuple):
                raise TypeError(f"subset must be None or a tuple of task ids; got {self.subset!r}")
            ids = tuple(self.subset)
            if not ids:
                raise ValueError("subset must not be empty; use None to solve every task")
            if not all(isinstance(task, str) for task in ids):
                raise TypeError(f"subset must hold task id strings; got {ids!r}")
            bad = [task for task in ids if not TASK_ID.fullmatch(task)]
            if bad:
                raise ValueError(f"subset ids must be eight lowercase hex digits; got {bad}")
            object.__setattr__(self, "subset", ids)
        for name, low, high in (
            ("workers", 1, None),
            ("hash_seed", 0, MAX_HASH_SEED),
            ("lora_r", 1, None),
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer; got {value!r}")
            if value < low or (high is not None and value > high):
                bound = f"from {low} to {high}" if high is not None else f">= {low}"
                raise ValueError(f"{name} must be {bound}; got {value}")
        targets = self.lora_targets
        if not isinstance(targets, list | tuple) or not all(isinstance(n, str) for n in targets):
            raise TypeError(f"lora_targets must be a tuple of module names; got {targets!r}")
        targets = tuple(targets)
        unknown = [name for name in targets if name not in LORA_TARGETS]
        if unknown:
            raise ValueError(f"lora_targets must be names from {LORA_TARGETS}; got {unknown}")
        if len(set(targets)) != len(targets):
            raise ValueError(f"lora_targets repeats a name: {targets}")
        if not set(targets) & set(LORA_PROJECTIONS):
            # PEFT refuses a LoRA config that matches no module of the model.
            raise ValueError(
                f"lora_targets needs at least one of {LORA_PROJECTIONS}; got {targets}"
            )
        object.__setattr__(self, "lora_targets", targets)
        if not isinstance(self.solver, str):
            raise TypeError(f"solver must be a string; got {self.solver!r}")
        if self.solver not in SOLVER_MODULES:
            raise ValueError(f"solver must be one of {sorted(SOLVER_MODULES)}; got {self.solver!r}")
        if isinstance(self.holdout, Mapping):
            object.__setattr__(self, "holdout", HoldOut.from_mapping(self.holdout))
        if self.holdout is not None:
            if not isinstance(self.holdout, HoldOut):
                raise TypeError(f"holdout must be None or HoldOut settings; got {self.holdout!r}")
            if self.solver != "split":
                raise ValueError(
                    "holdout needs solver 'split': the moved reference trains and decodes in one "
                    "loop body, so it cannot leave a pair out of one and not the other (ADR-011)"
                )
            outputs = os.path.abspath(self.outputs_dir)
            folders = [(name, getattr(self.holdout, name)) for name in ("gate_dir", "log_dir")]
            if self.holdout.rounds is not None:
                # Task P2.3: the rounds' candidate files, decoded before any is installed.
                folders.append(("rounds round_dir", self.holdout.rounds.round_dir))
            for name, folder in folders:
                if folder is not None and os.path.commonpath(
                    [os.path.abspath(folder), outputs]
                ) == outputs:
                    # The selection reads every entry of outputs_dir as a candidate file.
                    raise ValueError(
                        f"holdout {name} {folder!r} must lie outside outputs_dir "
                        f"{self.outputs_dir!r}, whose every file the selection reads"
                    )
        if isinstance(self.adapters, Mapping):
            object.__setattr__(self, "adapters", Arms.from_mapping(self.adapters))
        if self.adapters is not None:
            if not isinstance(self.adapters, Arms):
                raise TypeError(f"adapters must be None or Arms settings; got {self.adapters!r}")
            if self.solver != "split":
                raise ValueError(
                    "adapters need solver 'split': arc26.arms calls arc26.ttt.fit and "
                    "arc26.decode.decode_puzzle apart (ADR-011)"
                )
            if self.holdout is not None:
                raise ValueError(
                    "adapters exclude holdout: every arm trains and decodes on every pair, with "
                    "no loop (ADR-018)"
                )
            outputs = os.path.abspath(self.outputs_dir)
            for name, folder in self.adapters.folders():
                if os.path.commonpath([os.path.abspath(folder), outputs]) == outputs:
                    raise ValueError(
                        f"{name} {folder!r} must lie outside outputs_dir {self.outputs_dir!r}, "
                        "whose every file the selection reads"
                    )


def _solver_config(cfg: RunConfig) -> dict[str, Any]:
    # Exactly the keys the moved starter.py and arc_solver.py read from their cfg dict
    # (configs/extraction.yaml), which arc26.solver reads too; hash_seed reaches the solver
    # through its environment instead, and solver picks the module. holdout (task P2.1) is added
    # only when set, so a run without it passes the JSON the moved code reads, unchanged; its
    # rounds (task P2.3) ride inside it, only when set.
    config: dict[str, Any] = {
        "data_dir": cfg.data_dir,
        "model_dir": cfg.model_dir,
        "outputs_dir": cfg.outputs_dir,
        "marker_dir": cfg.marker_dir,
        "subset": None if cfg.subset is None else list(cfg.subset),
        "workers": cfg.workers,
        "timing_dir": cfg.timing_dir,
        "lora_r": cfg.lora_r,
        "lora_targets": list(cfg.lora_targets),
    }
    if cfg.holdout is not None:
        config["holdout"] = cfg.holdout.to_json()
    if cfg.adapters is not None:
        # Task P3.7: only when set, so every other run passes the JSON it passed before.
        config["adapters"] = cfg.adapters.to_json()
    if cfg.hyper_checkpoint is not None:
        config.update(hyper_checkpoint=cfg.hyper_checkpoint, hyper_sha256=cfg.hyper_sha256,
                      hyper_log_dir=cfg.hyper_log_dir)
    return config


def solver_env(cfg: RunConfig, base: Mapping[str, str]) -> dict[str, str]:
    """Return the solver's environment: ``base`` plus the reference's launch variables.

    Why: the reference's launch line (cell 6) sets ``UNSLOTH_DISABLE_STATISTICS``,
    ``TRITON_PTXAS_PATH``, and ``OMP_NUM_THREADS`` for the solver only, and run A fixed
    ``PYTHONHASHSEED`` so that the rescoring seeds repeat across runs.

    Args:
        cfg: The run settings; ``hash_seed`` becomes ``PYTHONHASHSEED``.
        base: The environment to extend, usually ``os.environ``; not modified.

    Returns:
        A new mapping: ``base``, then the three launch variables and ``PYTHONHASHSEED``, which
        override any value ``base`` holds for them.
    """
    env = dict(base)
    env.update(LAUNCH_ENV)
    env["PYTHONHASHSEED"] = str(cfg.hash_seed)
    return env


def solver_command(
    cfg: RunConfig, end_time: float, python: str, module: str | None = None
) -> list[str]:
    """Return the argument list that starts the solver.

    Why: the reference runs ``python starter.py --end-time {global_end_time}``; the moved
    ``starter.py`` is a package module and takes its settings as one JSON argument, and so does
    the split solver's entry, ``arc26.solver``.

    Args:
        cfg: The run settings: ``solver`` picks the module (``SOLVER_MODULES``), and the rest is
            passed as ``--config`` JSON with the keys ``data_dir``, ``model_dir``,
            ``outputs_dir``, ``marker_dir``, ``subset`` (a list of task ids or null),
            ``workers``, ``timing_dir`` (a folder or null), ``lora_r``, and ``lora_targets`` (a
            list of module names), plus ``holdout`` (``HoldOut.to_json()``) and ``adapters``
            (``Arms.to_json()``) only when set.
        end_time: Absolute deadline in seconds since the epoch (``time.time()`` scale); workers
            stop taking puzzles after it, as in the reference.
        python: The interpreter to run.
        module: The entry module; None takes ``SOLVER_MODULES[cfg.solver]``. The notebook's
            rank-256 tail (task P4.7b) passes ``"arc26.tail"``, which takes the same arguments.

    Returns:
        ``[python, "-m", module, "--end-time", repr(end_time), "--config", json]``, ``module``
        ``"arc26.nvarc.starter"`` or ``"arc26.solver"``, or ``ARMS_MODULE`` with ``adapters``
        (``"arc26.h_submission"`` with ``hyper_checkpoint``), unless given.

    Raises:
        ValueError: If ``end_time`` is not a finite number.
    """
    end = float(end_time)
    if not math.isfinite(end):
        raise ValueError(f"end_time must be finite; got {end_time!r}")
    config = json.dumps(_solver_config(cfg))
    if module is None:
        module = SOLVER_MODULES[cfg.solver] if cfg.adapters is None else ARMS_MODULE
        if cfg.hyper_checkpoint is not None:
            module = "arc26.h_submission"
    return [python, "-m", module, "--end-time", repr(end), "--config", config]


def _write(text: str) -> None:
    if text:
        sys.stdout.write(text)
        sys.stdout.flush()


def _copy_output(fd: int, decoder: codecs.IncrementalDecoder, deadline: float | None) -> bool:
    # Copy the child's output to sys.stdout as it arrives (the notebook cell shows sys.stdout).
    # Returns True at end of output, False when the deadline passes first.
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            if not selector.select(remaining):
                continue
            chunk = os.read(fd, CHUNK_BYTES)
            if not chunk:
                _write(decoder.decode(b"", final=True))
                return True
            _write(decoder.decode(chunk))


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    # Only called before the process is reaped, so its process group id cannot be reused yet.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_solver(
    cfg: RunConfig,
    end_time: float,
    python: str | None = None,
    timeout: float | None = None,
    module: str | None = None,
) -> int:
    """Run the solver to completion, streaming its output; kill it after ``timeout``.

    Why: the reference runs the solver as a separate process (``!python starter.py``), which
    spawns one worker per GPU. The solver runs in its own session, so a timeout, or an exception
    such as ``KeyboardInterrupt``, kills it with every worker it spawned. Its stdout and stderr
    are copied to ``sys.stdout`` as they arrive. ``PYTHONUNBUFFERED=1`` is added to the base
    environment, so lines arrive when printed and survive a kill (IPython's ``!`` had the same
    effect through a terminal); it changes no computation.

    Args:
        cfg: The run settings.
        end_time: Absolute deadline passed to the workers (``solver_command``).
        python: The interpreter; defaults to ``shutil.which("python")``, the interpreter the
            reference's ``!python starter.py`` used.
        timeout: Seconds before the solver and its workers are killed; None waits forever.
        module: The entry module (``solver_command``); None for the config's solver.

    Returns:
        The solver's exit code, or ``-signal.SIGKILL`` (-9) when it was killed at the timeout.

    Raises:
        FileNotFoundError: If ``python`` is None and no ``python`` is on ``PATH``.
        ValueError: If ``end_time`` is not finite.
    """
    if python is None:
        python = shutil.which("python")
        if python is None:
            raise FileNotFoundError("no python on PATH; pass the interpreter as python=")
    command = solver_command(cfg, end_time, python, module)
    env = solver_env(cfg, {**os.environ, "PYTHONUNBUFFERED": "1"})
    deadline = None if timeout is None else time.monotonic() + timeout
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    process = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    fd = process.stdout.fileno()
    try:
        if _copy_output(fd, decoder, deadline):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                return process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        _kill_group(process)
        _copy_output(fd, decoder, time.monotonic() + DRAIN_SECONDS)
        return process.wait()
    except BaseException:
        _kill_group(process)
        process.wait()
        raise
    finally:
        process.stdout.close()


def make_submission(cfg: RunConfig, submission_path: str) -> None:
    """Select two attempts per test output from the candidates and write the submission file.

    Why: this is the reference's cell 7 (moved as ``arc26.nvarc.select.write_submission``). It
    reads ``KAGGLE_IS_COMPETITION_RERUN`` itself: in a rerun it reads only the test challenges;
    otherwise it also reads the evaluation solutions and prints the dev-mode score. The
    candidate folder is created first, so a solver that wrote nothing still yields a
    submission of the reference's default ``[[0]]`` attempts. With ``cfg.timing_dir`` set, the
    call is recorded as op ``main_selection`` with a null ``puzzle_id`` and null tokens in
    ``<timing_dir>/timing-main.jsonl`` (B4); the import before it is not part of the op. A
    call that raises writes no record.

    Args:
        cfg: The run settings; ``data_dir``, ``outputs_dir``, and ``timing_dir`` are read.
        submission_path: Where to write the submission JSON.

    Raises:
        ImportError: If ``transformers`` is not installed (``arc_loader`` imports it).
        ValueError: In dev mode, from the reference's ``benchmark_selection_algos`` when no
            candidate is correct; the submission file is already written by then.
        OSError: If the timing record cannot be written.
    """
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    # Imported here: arc_loader imports transformers, which only the Kaggle image provides.
    from arc26.nvarc.select import write_submission

    if cfg.timing_dir is None:
        write_submission(cfg.data_dir, cfg.outputs_dir, submission_path)
        return
    # A logger of its own rather than arc26.timing.configure, so the calling process (the
    # notebook kernel) keeps no process-wide logger.
    with TimingLogger(cfg.timing_dir, MAIN_WORKER).timed(None, MAIN_SELECTION):
        write_submission(cfg.data_dir, cfg.outputs_dir, submission_path)
