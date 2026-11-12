r"""Score a pulled ``P5.2-01`` run per arm and apply its rule locally (task P5.2; fn_probe.md).

Purpose: the notebook never sees the right grids, so its attempts are scored here, after
``kaggle_ops.py pull``, against the evaluation solutions (dev-mode scoring only; BRIEF A0): per
arm (A plain, B tool), on the tasks both arms finished, beside ``P6.9-01``'s Qwen3.8-27B on the
same tasks; then the harness note's pre-registered rule picks version 11's arm.

Inputs: the pulled ``run_logs/fn`` folder (``attempts.jsonl``, ``boot.json``, ``state.json``),
the run's config (``report``: the 27B's per-task scores and cap, the rule's thresholds), and the
solutions: the Kaggle file ``arc-agi_evaluation_solutions.json`` or a folder of ARC-AGI-2 task
files (``<id>.json`` with ``test[i].output``). Outputs: ``<out>/fn_report.md`` and
``fn_report.json``.

Method: each arm's attempts become a submission and are scored by
:func:`arc26.evaluate.score_submission` (a task scores the share of its test outputs that either
attempt matches). With :math:`P` the tasks both arms finished,

.. math:: S_X = \sum_{t \in P} s^X_t, \qquad
   c_B = \frac{\sum_{\text{tool turns}} \text{cached}}{\sum_{\text{tool turns}} \text{prompt}},

and the rule (:func:`decide`): B when :math:`S_B \ge S_A`, B's median wall per task is at most
1.3 times A's, and :math:`c_B \ge 0.9`; A when :math:`S_A - S_B \ge 2` or B's median wall is
above 1.5 times A's; a Fault means A; otherwise the human's call. With 15 tasks or fewer, a
difference under 2 tasks is noise (CLAUDE.md's DEV40 rule, which a smaller set only loosens).
Decode rates come from the solve's own request records (no sweep).

Usage::

    python scripts/fn_report.py --run-logs runs/P5.2-01/run_logs/fn --config configs/P5.2-01.yaml
        --solutions <arc-agi_evaluation_solutions.json or a folder> --out reports/P5.2-01
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EMPTY = [[0]]
NOISE_TASKS = 2.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping a torn line; a missing file gives no rows."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def load_solutions(path: Path, task_ids: Sequence[str]) -> dict[str, list[Any]]:
    """Return the solution grids of the given tasks from a Kaggle file or an ARC task folder.

    Args:
        path: ``arc-agi_evaluation_solutions.json`` (id to a list of grids), or a folder of
            ``<id>.json`` files whose ``test`` entries hold ``output``.
        task_ids: The tasks to read.

    Returns:
        Task id to its output grids, for the ids found.
    """
    if path.is_dir():
        out = {}
        for task_id in task_ids:
            file = path / f"{task_id}.json"
            if file.is_file():
                data = json.loads(file.read_text(encoding="utf-8"))
                out[task_id] = [entry["output"] for entry in data["test"]]
        return out
    from arc26.evaluate import load_solutions as load

    data = load(path)
    return {t: data[t] for t in task_ids if t in data}


def predictions(attempts: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Turn the run's attempt records into submission-format predictions.

    Args:
        attempts: ``attempts.jsonl`` rows: ``attempts`` holds, per test output, up to two grids.

    Returns:
        Task id to one ``{"attempt_1", "attempt_2"}`` per test output (``[[0]]`` where missing).
    """
    out = {}
    for row in attempts:
        if "attempts" not in row:
            continue
        entries = []
        for picks in row["attempts"]:
            first = picks[0] if picks else EMPTY
            second = picks[1] if len(picks) > 1 else EMPTY
            entries.append({"attempt_1": first, "attempt_2": second})
        out[row["task_id"]] = entries
    return out


def _median(values: Sequence[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def arm_readings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return one arm's readings from its attempt records.

    Args:
        rows: The arm's ``attempts.jsonl`` rows.

    Returns:
        Tasks, forced share, median wall, median per-stream decode rate of the thinking turns,
        calls and their failures, the cached share of the continuation turns, and the answer
        blocks per task.
    """
    turns = [t for r in rows for t in r.get("turns", [])]
    tool_turns = [t for t in turns if t.get("kind") == "tool"]
    prompt = sum(t.get("prompt_tokens") or 0 for t in tool_turns)
    cached = sum(t.get("cached_tokens") or 0 for t in tool_turns)
    calls = [c for r in rows for c in r.get("calls") or []]
    return {
        "tasks": len(rows),
        "forced_share": round(sum(1 for r in rows if r.get("forced")) / len(rows), 3)
        if rows else None,
        "median_wall_s": _median([r.get("wall_s") for r in rows]),
        "median_decode_tok_s": _median([t.get("decode_tok_s") for t in turns
                                        if t.get("kind") in ("think", "tool")]),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
        "calls": len(calls),
        "call_failures": sum(1 for c in calls if c.get("status") != "ok"),
        "programs": sum(1 for r in rows if r.get("program")),
        "continuation_cached_share": round(cached / prompt, 3) if prompt else None,
        "blocks": {r["task_id"]: r.get("blocks") for r in rows},
        "errors": sum(1 for r in rows if "error" in r),
    }


def decide(values: Mapping[str, Any], rule: Mapping[str, Any]) -> tuple[str, str]:
    r"""Apply the harness note's pre-registered rule to arms A (plain) and B (tool).

    Fault (the boot, the first generation, the stop rules, the tool self-test, or no paired
    task) means A. Else **B** when :math:`S_B \ge S_A`, B's median wall is at most 1.3 times
    A's, and B's continuations are at least 90% cached; **A** when :math:`S_A - S_B \ge 2` or
    B's median wall is above 1.5 times A's; otherwise the human's call.

    Args:
        values: :func:`report`'s values (``fault``, ``paired``, ``arms``).
        rule: ``report.rule`` of the config.

    Returns:
        The arm (``A``, ``B``, or ``human``) and the reason.
    """
    if values.get("fault"):
        return "A", f"Fault: {values['fault']}"
    arms = values["arms"]
    if "plain" not in arms or "tool" not in arms or not values["paired"]:
        return "A", "Fault: no paired task"
    s_a, s_b = arms["plain"]["paired_score"], arms["tool"]["paired_score"]
    wall_a, wall_b = arms["plain"]["median_wall_s"], arms["tool"]["median_wall_s"]
    ratio = wall_b / wall_a if wall_a and wall_b else None
    cached = arms["tool"]["continuation_cached_share"]
    if s_a - s_b >= float(rule["margin_tasks"]) or (ratio is not None and ratio > float(
            rule["a_wall_ratio"])):
        return "A", f"S_A {s_a} against S_B {s_b}, wall ratio {ratio}"
    if s_b >= s_a and ratio is not None and ratio <= float(rule["b_wall_ratio"]) and (
            cached is not None and cached >= float(rule["cached_share"])):
        confirmed = " (confirmed)" if s_b - s_a >= float(rule["margin_tasks"]) else ""
        return "B", f"S_B {s_b} against S_A {s_a}, wall ratio {ratio}, cached {cached}{confirmed}"
    return "human", f"S_A {s_a}, S_B {s_b}, wall ratio {ratio}, cached {cached}"


def report(run_logs: Path, config: Mapping[str, Any], solutions_path: Path) -> dict[str, Any]:
    """Score each arm and gather the readings the brief names.

    Args:
        run_logs: The pulled ``run_logs/fn`` folder.
        config: The run's config.
        solutions_path: The solutions file or folder.

    Returns:
        The report's values (also rendered by :func:`render`).
    """
    from arc26.evaluate import score_submission

    spec = config["report"]
    attempts = [r for r in read_jsonl(run_logs / "attempts.jsonl")]
    order = config["solve"]["task_ids"]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        by_arm.setdefault(row.get("arm") or "plain", []).append(row)
    ok = {arm: {r["task_id"] for r in rows if "error" not in r} for arm, rows in by_arm.items()}
    paired = [t for t in order if ok and all(t in tasks for tasks in ok.values())]
    solutions = load_solutions(solutions_path, [t for t in order if any(
        t in tasks for tasks in ok.values())])
    reference = spec["reference_scores"]
    arms: dict[str, Any] = {}
    rows_out = []
    for arm, rows in sorted(by_arm.items()):
        preds = predictions(rows)
        scored = [t for t in order if t in preds and t in solutions]
        score = score_submission(preds, solutions, scored, label=f"P5.2-01 {arm}")
        per_task = {s.task_id: round(s.score, 3) for s in score.per_task}
        readings = arm_readings(rows)
        paired_rows = [r for r in rows if r["task_id"] in paired]
        readings.update(
            score=round(sum(per_task.values()), 3), scored=len(scored),
            paired_score=round(sum(per_task.get(t, 0.0) for t in paired), 3),
            paired_median_wall_s=_median([r.get("wall_s") for r in paired_rows]))
        readings["median_wall_s"] = readings["paired_median_wall_s"] or readings["median_wall_s"]
        arms[arm] = readings
        for row in rows:
            rows_out.append({
                "task_id": row["task_id"], "arm": arm, "score": per_task.get(row["task_id"]),
                "reference": reference.get(row["task_id"]),
                "ended_by": row.get("ended_by", "error"),
                "completion_tokens": row.get("completion_tokens"),
                "calls": len(row.get("calls") or []), "blocks": row.get("blocks"),
                "wall_s": row.get("wall_s"), "error": row.get("error"),
            })
    state = {}
    if (run_logs / "state.json").is_file():
        state = json.loads((run_logs / "state.json").read_text(encoding="utf-8"))
    fault = state.get("killed") or (state.get("solve") or {}).get("killed")
    # A run cut by the Timeout or killed mid-solve leaves no finished solve stage: its rows are
    # partial and decide nothing.
    solve_status = (state.get("solve") or {}).get("status")
    if solve_status != "ok":
        fault = fault or f"solve {solve_status or 'missing'} (cut before the solve finished)"
    if (state.get("outer") or {}).get("exit_inner") == 124:
        fault = fault or "the inner process hit its time limit"
    for stage in ("boot", "tool_self_test"):
        if (state.get(stage) or {}).get("status") not in (None, "ok"):
            fault = fault or f"{stage} {state[stage]['status']}"
    boot = {}
    if (run_logs / "boot.json").is_file():
        boot = json.loads((run_logs / "boot.json").read_text(encoding="utf-8"))
    values: dict[str, Any] = {
        "paired": paired, "arms": arms, "rows": rows_out, "fault": fault,
        "noise_tasks": NOISE_TASKS, "cap": config["solve"]["cap"],
        "reference_run": spec["reference_run"], "reference_cap": spec["reference_cap"],
        "reference_paired_score": round(sum(float(reference.get(t, 0.0)) for t in paired), 3),
        "reference_aggregate_tok_s": spec["reference_aggregate_tok_s"],
        "solve": {k: (state.get("solve") or {}).get(k) for k in (
            "streams", "per_arm", "pool_tokens", "max_running_requests", "restarts")},
        "boot": {k: boot.get(k) for k in ("ready_s", "pool_tokens", "max_running_requests",
                                          "gpu_peak_used_mib", "ram_available_gib",
                                          "flag_overrides")},
    }
    values["decision"], values["decision_reason"] = decide(values, spec["rule"])
    return values


def render(values: Mapping[str, Any]) -> str:
    """Render the report as Markdown.

    Args:
        values: :func:`report`'s result.

    Returns:
        The Markdown text.
    """
    boot, solve = values["boot"], values["solve"]
    lines = ["# P5.2-01: version 11's development check (scored locally)", "",
             f"- **Decision: {values['decision']}** ({values['decision_reason']}).",
             f"- Boot to healthy {boot.get('ready_s')} s; KV pool {boot.get('pool_tokens')} "
             f"tokens; request limit {boot.get('max_running_requests')}; streams "
             f"{solve.get('streams')} ({solve.get('per_arm')} per arm); restarts "
             f"{solve.get('restarts')}; GPU peak MiB {boot.get('gpu_peak_used_mib')}.",
             f"- Paired tasks: {len(values['paired'])}; {values['reference_run']}'s 27B on them "
             f"{values['reference_paired_score']} (its cap {values['reference_cap']:,}; this run "
             f"{values['cap']:,}). A difference under {values['noise_tasks']:.0f} tasks is "
             "noise.", "",
             "| Arm | Paired score | Score (scored tasks) | Median wall s | Median decode tok/s "
             "| Forced share | Calls (failed) | Cached share | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, a in values["arms"].items():
        lines.append(f"| {arm} | {a['paired_score']} | {a['score']} of {a['scored']} | "
                     f"{a['median_wall_s']} | {a['median_decode_tok_s']} | {a['forced_share']} | "
                     f"{a['calls']} ({a['call_failures']}) | {a['continuation_cached_share']} | "
                     f"{a['errors']} |")
    lines += ["", "| Task | Arm | Score | 27B | Ended by | Tokens | Calls | Blocks | Wall s |",
              "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in values["rows"]:
        lines.append(f"| `{row['task_id']}` | {row['arm']} | {row['score']} | "
                     f"{row['reference']} | {row['ended_by']} | {row['completion_tokens']} | "
                     f"{row['calls']} | {row['blocks']} | {row['wall_s']} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-logs", type=Path, required=True, help="pulled run_logs/fn")
    parser.add_argument("--config", type=Path, required=True, help="configs/P5.2-01.yaml")
    parser.add_argument("--solutions", type=Path, required=True,
                        help="arc-agi_evaluation_solutions.json or a folder of task files")
    parser.add_argument("--out", type=Path, required=True, help="folder for fn_report.md/json")
    parser.add_argument("--smoke", action="store_true", help="use the config's smoke section")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the report.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0.
    """
    import yaml

    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        for key, value in config.get("smoke", {}).items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    values = report(args.run_logs, config, args.solutions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fn_report.json").write_text(json.dumps(values, indent=1) + "\n", "utf-8")
    (args.out / "fn_report.md").write_text(render(values), encoding="utf-8")
    print(render(values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
