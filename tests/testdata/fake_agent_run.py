"""A stand-in for ``python -m arc26.agent.run`` in the hybrid notebook's CPU simulation (P6.5).

``tests/test_hybrid_notebook.py`` runs the generated notebook on a simulated Kaggle tree; its
``python`` shim runs this file instead of the module when ``ARC26_FAKE_AGENT`` names it. It takes
task P6.4's command line (``docs/modules/agent_run.md``, ``arc26/agent/run.py`` on ``task/P6.4``),
records the arguments and what it read in ``<log-dir>/fake_agent.json``, and keeps the output
valid as the real run does: NVARC's attempts at once, then atomic rewrites. As the real run after
P6.4's fix round, it refuses (exit 2) a competition run without ``--deadline`` or with
``--solutions``. Then it behaves as ``ARC26_FAKE_AGENT_MODE`` says:

- ``solve``: sets attempt 1 of the first test output NVARC left at ``[[0]]`` to the stand-in
  solution, from a program verified on every demonstration whose record it logs first in
  ``attempts.jsonl`` as P6.4's run does (P6.2's ``record_to_json`` format), writes ``run.json``,
  exits 0;
- ``boot-fail``: the engine never comes up: NVARC's attempts stay, ``run.json`` lists the
  problem, exits 0 (the real run's rule: a valid file at the end is exit 0);
- ``refused``: exits 2 at once, writing nothing (a refused config);
- ``crash``: writes its improvement (its record logged first, then a record cut mid-line, as a
  kill during a write leaves it), leaves an engine running in a session of its own (its pid in
  ``engine.pid``) and a cut temporary file beside the output, then dies by SIGKILL without
  ``run.json``;
- ``hang-term``: writes its improvement, starts the engine, and waits; on SIGTERM it improves
  one more output and exits 143, as the real run's cleanup would;
- ``hang-kill``: as ``hang-term``, but ignores SIGTERM, so only SIGKILL ends it;
- ``corrupt``: writes a truncated, invalid file over the output directly and exits 1;
- ``invalid``: writes a JSON file that leaves a task out and holds an invalid grid, exits 0.

The engine it starts is ``python -c 'import time; time.sleep(600)'`` in a session of its own,
its output to ``engine.log`` and its pid in ``engine.pid``, as P6.4's server starts and records
vLLM, so the notebook must stop it through that file.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SOLUTION = [[1, 2], [3, 4]]
SECOND = [[4, 3], [2, 1]]


def write_atomic(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def record_line(task, count, grid):
    """One attempt record as P6.4's run logs it (P6.2's ``record_to_json``): a program verified
    on the task's one demonstration whose output on every test input is ``grid``."""
    source = f"def transform(grid):\n    return {grid!r}\n"
    verification = {
        "source": source,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "pairs": [
            {"index": 0, "exact": True, "output": SOLUTION, "error": None, "shape_match": True,
             "cell_accuracy": 1.0}
        ],
        "test_outputs": [grid] * count,
        "guards": [],
        "seconds": 0.5,
    }  # fmt: skip
    record = {
        "task_id": task,
        "attempt_id": f"{task}/p1/0",
        "phase": 1,
        "verifications": [verification],
        "direct_answers": [],
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "turns": 1,
        "seconds": 1.0,
        "stop_reason": "verified",
    }
    return json.dumps(record, separators=(",", ":")) + "\n"


def start_engine(log_dir):
    # As P6.4's server starts vLLM: a session of its own, its output to engine.log.
    argv = [sys.executable, "-S", "-c", "import time; time.sleep(600)"]
    with open(log_dir / "engine.log", "ab") as log:
        engine = subprocess.Popen(
            argv, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log
        )
    (log_dir / "engine.pid").write_text(json.dumps({"pid": engine.pid, "argv": argv}))
    return engine.pid


def main():
    parser = argparse.ArgumentParser(prog="python -m arc26.agent.run")
    parser.add_argument("--mode", choices=("dev", "competition"), required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-dir")
    engine = parser.add_mutually_exclusive_group(required=True)
    engine.add_argument("--engine-url")
    engine.add_argument("--engine-venv")
    engine.add_argument("--fake-engine")
    parser.add_argument("--model-dir")
    parser.add_argument("--port", type=int)
    parser.add_argument("--nvarc-submission")
    parser.add_argument("--nvarc-candidates")
    parser.add_argument("--solutions")
    parser.add_argument("--task-ids")
    deadline = parser.add_mutually_exclusive_group()
    deadline.add_argument("--deadline", type=float)
    deadline.add_argument("--budget-s", type=float)
    args = parser.parse_args()
    mode = os.environ.get("ARC26_FAKE_AGENT_MODE", "solve")
    if mode == "refused" or (args.mode == "competition" and args.solutions):
        print("refused: competition mode never opens a solutions file", file=sys.stderr)
        return 2
    if args.mode == "competition" and args.deadline is None:
        # P6.4's fix round: competition mode needs an explicit deadline, never a default.
        print("refused: competition mode needs --deadline", file=sys.stderr)
        return 2
    if "solution" in Path(args.tasks).name.lower():
        print("refusing to read a solutions file", file=sys.stderr)
        return 2
    log_dir = Path(args.log_dir)
    out = Path(args.out)
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    ids = args.task_ids.split(",") if args.task_ids else list(tasks)
    nvarc = json.loads(Path(args.nvarc_submission).read_text(encoding="utf-8"))
    candidates = None
    if args.nvarc_candidates:
        candidates = json.loads(Path(args.nvarc_candidates).read_text(encoding="utf-8"))
    record = {
        "argv": sys.argv[1:],
        "mode": args.mode,
        "tasks": args.tasks,
        "task_ids": ids,
        "config": json.loads(Path(args.config).read_text(encoding="utf-8")),
        "nvarc": nvarc,
        "candidates": sorted(candidates) if candidates is not None else None,
        "deadline": args.deadline,
        "engine_venv": args.engine_venv,
        "model_dir": args.model_dir,
        "cwd": os.getcwd(),
        "fake_mode": mode,
    }
    (log_dir / "fake_agent.json").write_text(json.dumps(record))
    print(f"fake agent: {mode} on {len(ids)} tasks until {args.deadline}", flush=True)
    submission = {task: nvarc[task] for task in ids}
    write_atomic(out, submission)
    if mode == "boot-fail":
        summary = {"problems": ["engine start 1: the engine exited with code 1 during its boot"]}
        (log_dir / "run.json").write_text(json.dumps(summary))
        return 0
    if mode == "corrupt":
        out.write_text('{"truncated": [', encoding="utf-8")
        return 1
    if mode == "invalid":
        broken = dict(submission)
        first = sorted(broken)[0]
        broken[first] = [dict(entry, attempt_2=[[10]]) for entry in broken[first]]
        broken.pop(sorted(broken)[-1])
        write_atomic(out, broken)
        return 0
    open_slots = [
        (task, index)
        for task in sorted(submission)
        for index, entry in enumerate(submission[task])
        if entry["attempt_1"] == [[0]]
    ]
    attempts = log_dir / "attempts.jsonl"
    if open_slots:
        # As P6.4's scheduler: the record is logged first, then the file rewritten.
        task, index = open_slots[0]
        with open(attempts, "a", encoding="utf-8") as handle:
            handle.write(record_line(task, len(submission[task]), SOLUTION))
        submission[task][index]["attempt_1"] = SOLUTION
        write_atomic(out, submission)
    print(f"fake agent: solved {open_slots[:1]}", flush=True)
    if mode == "solve":
        summary = {"problems": [], "submission": {"valid": True, "slots": {"program": 1}}}
        (log_dir / "run.json").write_text(json.dumps(summary))
        return 0
    engine_pid = start_engine(log_dir)
    print(f"fake agent: engine {engine_pid} started", flush=True)
    if mode == "crash":
        with open(attempts, "a", encoding="utf-8") as handle:
            handle.write('{"task_id":"cut","attempt_id":"cut/p1/1","pha')
        out.with_name(f".{out.name}.tmp").write_text('{"cut', encoding="utf-8")
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "hang-term":

        def on_term(signum, frame):
            if len(open_slots) > 1:
                task, index = open_slots[1]
                submission[task][index]["attempt_2"] = SECOND
                write_atomic(out, submission)
            print("fake agent: SIGTERM, wrote once more", flush=True)
            sys.exit(143)

        signal.signal(signal.SIGTERM, on_term)
    elif mode == "hang-kill":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
