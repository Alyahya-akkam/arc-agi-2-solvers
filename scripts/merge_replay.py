r"""Replay the shot's merge on NVARC's stored attempts and read run P6.9-01's pre-registered rows.

Why: run ``P6.9-01`` (``docs/prerun/P6.9-01.md``) shoots 15 DEV120 tasks with the new request
but without NVARC beside it; what the shot is worth is what Submission 4's merge
(:func:`arc26.agent.shot.merge_output`) would have made of its answers next to NVARC's attempts
of ``P2.6-01``, as ``reports/P6.3-05/merge.md`` (task P6.3) did for ``P6.3-05``. CPU only; no
solutions file is read: an output's attempts are right where a log already scored them, NVARC's
in P2.8's per-candidate rows (``correct`` of the ``attempt`` 1 and 2 rows) and the shot's in the
run's ``shot_metrics.json`` (``per_task.<id>.right``, computed on Kaggle in development mode).

Per test output :math:`j` of task :math:`t`: :math:`N_1, N_2` from ``P2.6-01``'s
``submission.json`` (a slot without a candidate row is NVARC's placeholder), :math:`S_1, S_2`
from the run's ``direct_attempts.jsonl`` chosen as the run chose them
(:func:`arc26.agent.shot.shot_answers` with the implied shapes it logged), the merged pair
:math:`M = \operatorname{merge}(N_1, N_2, S_1, S_2)`, and the output right under a rule when one
of its two slots holds a right grid. A task scores the mean over its :math:`n_t` outputs,
:math:`s_t = \frac{1}{n_t}\sum_j \mathbb{1}[\text{a slot of } j \text{ is right}]`, the
competition's rule (``arc26.evaluate``).

Reading (``docs/prerun/P6.9-01.md``, section 1, in order): a Fault of the run stops the reading;
a harness check with 0 of its 5 tasks solved by the shot opens an investigation; the queue extends
to :math:`[0.5, 0.7)` (before confidence 0) when that bucket's merged net (merged less NVARC, task
units) less the expected loss of a replaced attempt 2, :math:`\Delta - n \ell` with
:math:`\ell` from P2.8's rows of two runs (``--mid-loss``), is at least ``--mid-net`` (1 of 6);
the in-request forcing stays when the outputs of the tasks whose thinking was cut got an answer
at least as often as with the separate forcing turn (``--forced-rate``, ``P6.3-05``'s 13 of 15).

Usage::

    python scripts/merge_replay.py --run-logs runs/P6.9-01/run_logs \
        --nvarc-submission ~/wt-P2.6/runs/P2.6-01/submission.json \
        --nvarc-csv reports/P2.8/P2.6-01.csv --out reports/P6.9-01

Outputs (``--out``): ``merge.csv`` (one row per test output), ``merge.md`` (the buckets and the
reading), ``reading.json`` (the same as data). Exit 0 when written, 2 on a missing or
inconsistent input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without the package installed
    sys.path.insert(0, str(REPO))

from arc26.agent.loop import record_from_json  # noqa: E402
from arc26.agent.shot import BUCKET_EDGE, MID_EDGE, merge_output, shot_answers  # noqa: E402

# The harness bucket's lower edge (option 2c: 5 random tasks NVARC solved at confidence 0.85 or
# more).
HIGH_EDGE = 0.85
# Pre-registered thresholds (docs/prerun/P6.9-01.md, section 1). The middle bucket's merged net in
# task units: 1 of 6, above 6 g_0 = 0.4 tasks, what the same 6 attempts would earn at confidence 0
# (g_0 = 3.5 / 53, docs/research/beyond_h_options.md section 1.5).
MID_NET = 1.0
# The expected task loss per shot task in [0.5, 0.7) when the shot's S1 replaces NVARC's attempt
# 2: the score its attempt-2-only outputs carry, 2.0 of 23 tasks in P2.6-01 and 3.5 of 23 in
# P1.1-01 (reports/P2.8, 3 and 5 tasks with such an output), pooled. An upper bound (it assumes
# the shot wrong there); taken from the CSVs because a 6-task sample may hold no such output.
MID_LOSS = (2.0 + 3.5) / (23 + 23)
# The answered share of the cut tasks' outputs with the separate forcing turn: P6.3-05's 13 of the
# 15 outputs of its 8 forced tasks.
FORCED_RATE = 13 / 15
# The harness check fails at this many shot solves of its 5 tasks.
HARNESS_FAIL = 0
BUCKETS = ("(0, 0.5)", "0", "[0.5, 0.7)", "[0.7, 0.85)", "[0.85, 1]")


def bucket(confidence: float) -> str:
    """Return a confidence's bucket label (``docs/research/beyond_h_options.md``, 1.5).

    Args:
        confidence: NVARC's task confidence.

    Returns:
        One of :data:`BUCKETS`.
    """
    if confidence == 0:
        return "0"
    if confidence < BUCKET_EDGE:
        return "(0, 0.5)"
    if confidence < MID_EDGE:
        return "[0.5, 0.7)"
    if confidence < HIGH_EDGE:
        return "[0.7, 0.85)"
    return "[0.85, 1]"


@dataclass
class NvarcRows:
    """What P2.8's rows say of NVARC's attempts on one run.

    Attributes:
        outputs: Task id to its number of test outputs.
        confidence: Task id to the shot's confidence (``softmax_p`` of each output's
            ``rank_kgmon`` 1 row, 0 without one; the minimum over outputs).
        real: ``(task, output, attempt)`` of every slot with a candidate.
        right: ``(task, output, attempt)`` of every slot whose candidate was right.
    """

    outputs: dict[str, int]
    confidence: dict[str, float]
    real: set[tuple[str, int, int]]
    right: set[tuple[str, int, int]]


def read_nvarc_rows(path: Path) -> NvarcRows:
    """Read P2.8's per-candidate CSV of one NVARC run.

    Args:
        path: ``reports/P2.8/<run>.csv``.

    Returns:
        The rows' facts.
    """
    outputs: dict[str, int] = {}
    top: dict[tuple[str, int], float] = {}
    real, right = set(), set()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task, output = row["task_id"], int(row["output"])
            outputs[task] = round(1 / float(row["weight"]))
            if row["rank_kgmon"] == "1":
                top[(task, output)] = float(row["softmax_p"])
            if row["attempt"] in ("1", "2"):
                real.add((task, output, int(row["attempt"])))
                if row["correct"] == "1":
                    right.add((task, output, int(row["attempt"])))
    confidence = {t: min(top.get((t, j), 0.0) for j in range(n)) for t, n in outputs.items()}
    return NvarcRows(outputs, confidence, real, right)


def task_score(rights: Sequence[bool]) -> float:
    """Return a task's score: the share of its outputs with a right slot.

    Args:
        rights: Per output, whether a slot is right.

    Returns:
        The mean, 0 for no outputs.
    """
    return sum(rights) / len(rights) if rights else 0.0


def replay(
    metrics: Mapping[str, Any],
    records: Sequence[Any],
    nvarc: Mapping[str, Sequence[Mapping[str, Any]]],
    rows: NvarcRows,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Replay the merge on every shot task, per output and per task.

    Args:
        metrics: The run's ``shot_metrics.json``.
        records: Its attempt records (``direct_attempts.jsonl``).
        nvarc: NVARC's ``submission.json``.
        rows: NVARC's P2.8 rows.

    Returns:
        One row per test output, and per task its bucket, confidence, and the NVARC, shot, and
        merged scores.

    Raises:
        ValueError: If a shot task is unknown to the NVARC files, or the run's metrics lack the
            per-output flags, or the replayed answers disagree with the metrics' count.
    """
    by_task: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)
    out_rows, tasks = [], {}
    for task_id, row in metrics["per_task"].items():
        if task_id not in nvarc:
            raise ValueError(f"{task_id}: not in NVARC's submission")
        if "right" not in row:
            raise ValueError(f"{task_id}: shot_metrics.json has no per-output flags")
        count = row["outputs"]
        # A task without any candidate has no P2.8 row: its outputs are the submission's, its
        # confidence 0 (P6.3-05's rule).
        rows.outputs.setdefault(task_id, len(nvarc[task_id]))
        rows.confidence.setdefault(task_id, 0.0)
        if count != rows.outputs[task_id] or count != len(nvarc[task_id]):
            raise ValueError(f"{task_id}: {count} outputs, NVARC's files disagree")
        implied = row.get("implied") or [None] * count
        merged, alone, base = [], [], []
        for j in range(count):
            shape = tuple(implied[j]) if implied[j] else None
            answers = shot_answers(by_task.get(task_id, []), j, shape) + [None, None]
            s1, s2 = answers[0], answers[1]
            entry = nvarc[task_id][j]
            n1 = entry["attempt_1"] if (task_id, j, 1) in rows.real else None
            n2 = entry["attempt_2"] if (task_id, j, 2) in rows.real else None
            right = {
                id(n1): (task_id, j, 1) in rows.right,
                id(n2): (task_id, j, 2) in rows.right,
            }
            s_right = row["right"][j]
            if s1 is not None:
                right[id(s1)] = bool(s_right[0])
            if s2 is not None:
                right[id(s2)] = bool(s_right[1])
            slots = merge_output(n1, n2, s1, s2)
            ok = any(grid is not None and right.get(id(grid), False) for grid, _ in slots)
            nv = (task_id, j, 1) in rows.right or (task_id, j, 2) in rows.right
            shot_ok = (s1 is not None and bool(s_right[0])) or (s2 is not None and bool(s_right[1]))
            merged.append(ok)
            base.append(nv)
            alone.append(shot_ok)
            out_rows.append({
                "task_id": task_id, "output": j, "bucket": bucket(rows.confidence[task_id]),
                "confidence": round(rows.confidence[task_id], 6),
                "N1_real": int(n1 is not None), "N2_real": int(n2 is not None),
                "N1_right": int((task_id, j, 1) in rows.right),
                "N2_right": int((task_id, j, 2) in rows.right),
                "S_answers": sum(a is not None for a in (s1, s2)),
                "S1_right": int(s1 is not None and bool(s_right[0])),
                "S2_right": int(s2 is not None and bool(s_right[1])),
                "S1_equals_N1": int(s1 is not None and s1 == n1),
                "merged_pair": ",".join(_slot_name(g, n1, n2, s1, s2) for g, _ in slots),
                "merged_right": int(ok),
            })  # fmt: skip
        tasks[task_id] = {
            "bucket": bucket(rows.confidence[task_id]),
            "confidence": rows.confidence[task_id],
            "nvarc": task_score(base),
            "shot": task_score(alone),
            "merged": task_score(merged),
        }
    return out_rows, tasks


def _slot_name(grid: Any, n1: Any, n2: Any, s1: Any, s2: Any) -> str:
    for name, value in (("N1", n1), ("N2", n2), ("S1", s1), ("S2", s2)):
        if grid is not None and grid is value:
            return name
    return "empty"


def bucket_table(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Sum the per-task scores by bucket.

    Args:
        tasks: :func:`replay`'s per-task results.

    Returns:
        Per bucket present: the task count, NVARC's, the shot's, and the merged scores, the net
        (merged less NVARC), and the tasks gained and lost.
    """
    table: dict[str, dict[str, Any]] = {}
    for name in BUCKETS:
        chosen = {t: v for t, v in tasks.items() if v["bucket"] == name}
        if not chosen:
            continue
        nvarc = sum(v["nvarc"] for v in chosen.values())
        merged = sum(v["merged"] for v in chosen.values())
        table[name] = {
            "tasks": len(chosen),
            "nvarc": round(nvarc, 3),
            "shot": round(sum(v["shot"] for v in chosen.values()), 3),
            "merged": round(merged, 3),
            "net": round(merged - nvarc, 3),
            "gained": sorted(t for t, v in chosen.items() if v["merged"] > v["nvarc"]),
            "lost": sorted(t for t, v in chosen.items() if v["merged"] < v["nvarc"]),
            "shot_solved": sorted(t for t, v in chosen.items() if v["shot"] >= 1),
        }
    return table


def read_rows(
    metrics: Mapping[str, Any],
    table: Mapping[str, Mapping[str, Any]],
    budget_check: Mapping[str, Any] | None,
    mid_net: float = MID_NET,
    forced_rate: float = FORCED_RATE,
    mid_loss: float = MID_LOSS,
) -> dict[str, Any]:
    """Apply the brief's pre-registered rows, in order.

    Args:
        metrics: The run's ``shot_metrics.json``.
        table: :func:`bucket_table`'s buckets.
        budget_check: The run's ``budget_check.json``, or None.
        mid_net: The middle bucket's adjusted net needed to extend the queue.
        forced_rate: The cut tasks' answered share needed to keep the in-request forcing.
        mid_loss: The expected loss per middle-bucket task (:data:`MID_LOSS`).

    Returns:
        ``fault`` (None or the script's Fault reading); ``harness`` (shot solves of the
        :math:`[0.85, 1]` tasks and whether an investigation opens); ``extend`` (the middle
        bucket's net and the decision); ``forcing`` (the answered share and the decision);
        ``lenient`` (the score with and without the lenient parser, reported only).
    """
    reading: dict[str, Any] = {"fault": None}
    if str(metrics.get("reading", "")).startswith("Fault"):
        reading["fault"] = metrics["reading"]
        return reading
    high = table.get("[0.85, 1]", {})
    solved = len(high.get("shot_solved", []))
    reading["harness"] = {
        "tasks": high.get("tasks", 0), "shot_solved": solved,
        "investigate": solved <= HARNESS_FAIL,
    }  # fmt: skip
    mid = table.get("[0.5, 0.7)")
    if mid is None:
        reading["extend"] = {"net": None, "decision": "not measured: no task in [0.5, 0.7)"}
    else:
        adjusted = round(mid["net"] - mid["tasks"] * mid_loss, 3)
        extend = adjusted >= mid_net - 1e-9
        reading["extend"] = {
            "net": mid["net"], "of": mid["tasks"],
            "expected_loss": round(mid["tasks"] * mid_loss, 3), "adjusted": adjusted,
            "threshold": mid_net,
            "decision": "extend the queue to [0.5, 0.7) before confidence 0 (mid_bucket true)"
            if extend else "keep (0, 0.5) then 0 (mid_bucket false)",
        }  # fmt: skip
    forced = metrics.get("forced_answers") or {}
    rate = forced.get("rate")
    honored = bool(forced.get("budget_forced")) and (budget_check or {}).get("ok", True)
    if rate is None:
        decision = "not measured (no cut turn): keep the separate forcing turn"
    elif not honored:
        decision = "the server did not end the thinking at the budget: keep the separate turn"
    elif rate >= forced_rate - 1e-9:
        decision = "keep the in-request forcing"
    else:
        decision = "revert to the separate forcing turn"
    reading["forcing"] = {
        "answered": forced.get("answered"), "outputs": forced.get("outputs"), "rate": rate,
        "threshold": round(forced_rate, 3), "budget_forced": forced.get("budget_forced"),
        "separate_turns": forced.get("forcing_turns"), "decision": decision,
    }  # fmt: skip
    strict = metrics.get("strict") or {}
    reading["lenient"] = {"score": metrics.get("score"), "strict_score": strict.get("score")}
    return reading


def markdown(
    table: Mapping[str, Mapping[str, Any]], reading: Mapping[str, Any], sources: Mapping[str, str]
) -> str:
    """Return ``merge.md``.

    Args:
        table: The buckets.
        reading: :func:`read_rows`'s result.
        sources: Input name to path, for the header.

    Returns:
        The Markdown text.
    """
    lines = [
        "# P6.9-01: the shot's merge replayed on P2.6-01's attempts",
        "",
        "Generated by `scripts/merge_replay.py`; no solutions file read (method in its module",
        "docstring). Inputs: " + "; ".join(f"{k} `{v}`" for k, v in sources.items()) + ".",
        "",
        "| Bucket | Tasks | NVARC | Shot alone | Merged | Net | Gained | Lost |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, row in table.items():
        lines.append(
            f"| {name} | {row['tasks']} | {row['nvarc']} | {row['shot']} | {row['merged']} | "
            f"{row['net']:+} | {', '.join(row['gained']) or 'none'} | "
            f"{', '.join(row['lost']) or 'none'} |"
        )
    lines += ["", "## Reading (docs/prerun/P6.9-01.md, section 1)", ""]
    if reading.get("fault"):
        lines.append(f"- **{reading['fault']}**: the decision rows are not read.")
        return "\n".join(lines) + "\n"
    harness = reading["harness"]
    lines.append(
        f"- Harness: the shot solved {harness['shot_solved']} of {harness['tasks']} NVARC solves; "
        + ("**investigate the harness first**." if harness["investigate"] else "passes.")
    )
    extend = reading["extend"]
    lines.append(
        f"- Queue: [0.5, 0.7) merged net {extend['net']} of {extend.get('of')} tasks, less the "
        f"expected attempt-2 loss {extend.get('expected_loss')}: {extend.get('adjusted')} "
        f"(threshold {extend.get('threshold')}): **{extend['decision']}**."
    )
    forcing = reading["forcing"]
    lines.append(
        f"- Forcing: {forcing['answered']} of {forcing['outputs']} outputs of the cut tasks "
        f"answered (rate {forcing['rate']}, threshold {forcing['threshold']}; "
        f"{forcing['budget_forced']} turns ended at the budget, {forcing['separate_turns']} "
        f"separate forcing turns): **{forcing['decision']}**."
    )
    lenient = reading["lenient"]
    lines.append(
        f"- Lenient parser (reported only): score {lenient['score']} against "
        f"{lenient['strict_score']} with the strict parser alone."
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Replay the merge and write the reports.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        0 when written, 2 on a missing or inconsistent input.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-logs", type=Path, required=True,
                        help="the pulled run_logs/ of the wave")  # fmt: skip
    parser.add_argument("--nvarc-submission", type=Path, required=True,
                        help="P2.6-01's submission.json")  # fmt: skip
    parser.add_argument("--nvarc-csv", type=Path, default=REPO / "reports/P2.8/P2.6-01.csv",
                        help="P2.8's rows of the same NVARC run")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True, help="the report folder")
    parser.add_argument("--mid-net", type=float, default=MID_NET)
    parser.add_argument("--mid-loss", type=float, default=MID_LOSS)
    parser.add_argument("--forced-rate", type=float, default=FORCED_RATE)
    args = parser.parse_args(argv)
    try:
        metrics = json.loads((args.run_logs / "shot_metrics.json").read_text(encoding="utf-8"))
        records = [
            record_from_json(json.loads(line))
            for line in (args.run_logs / "direct_attempts.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        ]
        check_path = args.run_logs / "budget_check.json"
        check = json.loads(check_path.read_text("utf-8")) if check_path.exists() else None
        nvarc = json.loads(args.nvarc_submission.read_text(encoding="utf-8"))
        rows = read_nvarc_rows(args.nvarc_csv)
        out_rows, tasks = replay(metrics, records, nvarc, rows)
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"refused: {err}", file=sys.stderr)
        return 2
    table = bucket_table(tasks)
    reading = read_rows(metrics, table, check, args.mid_net, args.forced_rate, args.mid_loss)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "merge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]) if out_rows else ["task_id"])
        writer.writeheader()
        writer.writerows(out_rows)
    sources = {"run": str(args.run_logs), "NVARC": str(args.nvarc_submission),
               "rows": str(args.nvarc_csv)}  # fmt: skip
    (args.out / "merge.md").write_text(markdown(table, reading, sources), encoding="utf-8")
    payload = {"buckets": table, "tasks": tasks, "reading": reading}
    (args.out / "reading.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(reading))
    return 0


if __name__ == "__main__":
    sys.exit(main())
