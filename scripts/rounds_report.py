r"""Round records of a hold-out arm with refinement rounds as Markdown tables (task P2.3).

Why: run ``P2.3-01``'s decision rule (``docs/prerun/P2.3-01.md``) counts records, and its report
gives each round's outcome, the promotion reasons, the fingerprints, the round's measured seconds
against the governor's estimate, and each final decode's seconds and the inputs it gave no
candidate. This script reads only the pulled files of one arm:

- ``run_logs/stages.json``: the arm's settings (the subset, the soft deadline, the governor config
  embedded in ``rounds.governor``), its exit, and the notebook's submission check;
- ``run_logs/<arm>/gate/``: the round-0 gate verdicts and the ``plan-``, ``promote-``, ``round-``,
  and ``decode-rank<r>.jsonl`` records (``docs/modules/loop.md``, Rounds);
- ``run_logs/<arm>/timing/``: the timing records, in file order per worker;
- the session log (the run folder's ``*.log``): ``[Rank r] stop!`` and ``timeout after`` lines;
- ``run_logs/<arm>/gate/context-rank<r>.jsonl``: the length guard's records (task P2.5), one per
  decode whose prompts left out a pseudo-pair.

With ``--baseline`` runs (their ``candidates.json``, selection output, ``submission.json``, and
optionally timing records), it adds a table per test output: where this run's submitted
candidates come from (round 0's, kept by a promotion or left by a final decode without a
candidate, or the final decode's), and in this run and each baseline the candidates, the correct
ones, and the correct grid's rank under ``score_kgmon``. A correct grid is a candidate that some
run's notebook marked ``ALL_CORRECT``; no solution file is read and no grid is written.

The governor's estimate of a planned round is recomputed from the embedded config with
``arc26.governor.CostModel`` and the plan's ``shape``, and split into its terms: the candidate fit
(``train`` at ``train_tokens`` in ``train_steps``), the re-decodes, the gate, and the test decodes
the plan reserved (the final decode), each ``decode_cost`` at the plan's ``observed_batch``, and
the overhead. A puzzle's records are split by file order: round 0 up to its first ``gate``
record, round ``r`` up to its ``r``-th ``round`` record, then the decodes after the rounds.

Usage::

    python scripts/rounds_report.py runs/P2.3-01 --arm r256-proj-emb-rounds \
        --out reports/P2.3-01/rounds.md [--baseline name=<label> arm=<arm folder> ...]

A baseline spec is ``key=value`` words: ``name`` (required), ``arm`` (the default folder of the
others), ``candidates``, ``selection`` (``selection.log``, or a session log holding the
selection's lines), ``submission``, and ``timing`` (a folder, or ``none``).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc26.evaluate import SubsetScore, TaskScore
from arc26.governor import CostModel
from arc26.timing import TimingRecord, read_records

# Ops a puzzle's records need per phase (docs/prerun/P2.3-01.md, section 1, timing complete).
ROUND0_OPS = ("train", "decode", "selection", "gate_decode", "gate")
FIT_OPS = ("fingerprint", "snapshot", "candidate_train", "gate_decode", "gate", "round")
# The in-context round (ADR-016) fits nothing: no snapshot, fit, or restore.
CONTEXT_OPS = ("fingerprint", "gate_decode", "gate", "round")
AFTER_OPS = ("round_decode", "round_selection", "final_decode", "final_selection", "puzzle")
RAISING = {"redecode_raised", "fit_raised", "gate_raised"}
RAN = {"kept", "rolled_back", "not_promoted"}
NOTHING = {"nothing_to_promote", "no_held_out"}
ALL_CORRECT = re.compile(r"^ALL_CORRECT:\s*\S+\s+-\s+\S+\s+\d+x\d+\s+\[(\S+)\]$")
KGMON = "('score_kgmon')"
BASELINE_KEYS = frozenset(("name", "arm", "candidates", "selection", "submission", "timing"))


def read_jsonl(folder: Path, prefix: str) -> list[dict[str, Any]]:
    """Return the records of every ``<prefix>-rank<r>.jsonl`` in a folder, in file order.

    Args:
        folder: The arm's ``gate/`` folder.
        prefix: ``gate``, ``plan``, ``promote``, ``round``, or ``decode``.

    Returns:
        The parsed lines, each with its worker label under ``worker``.
    """
    records = []
    for path in sorted(folder.glob(f"{prefix}-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append({**json.loads(line), "worker": path.stem.split("-", 1)[1]})
    return records


def phases(records: Sequence[TimingRecord]) -> dict[str, list[tuple[str, TimingRecord]]]:
    """Label each puzzle's timing records by phase, in file order.

    Why: round 0's gate and a round's candidate gate both write ``gate`` and ``gate_decode``;
    only the order tells them apart. Round 0 ends with the first ``gate`` record, round ``r`` with
    the ``r``-th ``round`` record, and what follows is ``after``.

    Args:
        records: Timing records in file order.

    Returns:
        Per puzzle, ``(phase, record)`` pairs: phase ``round0``, ``round<r>``, or ``after``.
    """
    labelled: dict[str, list[tuple[str, TimingRecord]]] = defaultdict(list)
    state: dict[str, tuple[str, int]] = {}
    for record in records:
        if record.puzzle_id is None:
            continue
        phase, rounds = state.get(record.puzzle_id, ("round0", 0))
        if phase != "round0" and record.op in AFTER_OPS:
            phase = "after"
        labelled[record.puzzle_id].append((phase, record))
        if phase == "round0" and record.op == "gate":
            phase = "round1"
        elif record.op == "round":
            rounds += 1
            phase = f"round{rounds + 1}"
        state[record.puzzle_id] = (phase, rounds)
    return labelled


def _sum(pairs: Sequence[tuple[str, TimingRecord]], op: str, phase: str | None = None) -> float:
    return sum(r.seconds for p, r in pairs if r.op == op and (phase is None or p == phase))


def _ops(pairs: Sequence[tuple[str, TimingRecord]], op: str, phase: str) -> list[TimingRecord]:
    return [r for p, r in pairs if r.op == op and p == phase]


def _f(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(str(c) for c in row) + " |" for row in rows),
    ]


def plan_variant(plan: Mapping[str, Any]) -> str:
    """Return the round variant a plan priced: its decision, or the one estimate it holds.

    Args:
        plan: A plan record.

    Returns:
        ``gradient`` or ``in_context``.
    """
    if plan["decision"] in ("gradient", "in_context"):
        return plan["decision"]
    return next(iter(plan["estimates"]))


def estimate_terms(cost: CostModel, plan: Mapping[str, Any]) -> dict[str, float]:
    r"""Split a round's estimate into the governor's terms (``CostModel.round_costs``).

    Why: the gradient round prices the candidate fit, the re-decodes, the gate, and the test
    decodes that may follow it at the slowest round-0 batch; the in-context round (ADR-016)
    has no fit and prices the gate and the test decodes at the prompts' length ratio $\rho$.

    Args:
        cost: The run's cost model.
        plan: The plan record.

    Returns:
        Seconds of ``fit``, ``redecode``, ``gate``, ``test``, ``overhead``, and ``total``, margin
        included; ``total`` equals the plan's estimate of its variant.
    """
    shape = plan["shape"]
    seen = shape["observed_batch"]
    if plan_variant(plan) == "in_context":
        context = shape["in_context"]
        ratio = context["length_ratio"]
        terms = {
            "fit": 0.0,
            "redecode": cost.decode_cost(context["redecode_batches"], 1.0, seen),
            "gate": cost.decode_cost(shape["gate_batches"], ratio, seen),
            "test": cost.decode_cost(shape["test_batches"], ratio, seen),
        }
    else:
        terms = {
            "fit": cost.estimate("train", shape["train_tokens"], steps=shape["train_steps"]),
            "redecode": cost.decode_cost(shape["redecode_batches"], 1.0, seen),
            "gate": cost.decode_cost(shape["gate_batches"], 1.0, seen),
            "test": cost.decode_cost(shape["test_batches"], 1.0, seen),
        }
    terms["overhead"] = (1.0 + cost.margin) * cost.overhead
    terms["total"] = sum(terms.values())
    return terms


def missing_ops(
    pairs: Sequence[tuple[str, TimingRecord]],
    rounds: Sequence[Mapping[str, Any]],
    final_ran: bool,
    rescored: bool = True,
) -> list[str]:
    """Return the timing records a puzzle lacks, by phase.

    Args:
        pairs: The puzzle's ``(phase, record)`` pairs from :func:`phases`.
        rounds: The puzzle's round records.
        final_ran: Whether its final decode ran (not skipped).
        rescored: Whether round 0's decode produced a candidate to rescore; without one it
            writes no ``selection`` record.

    Returns:
        ``<phase> <op>`` for each record the decision rule needs and the files lack: round 0's
        ops; per round that ran, a ``redecode`` and its ``redecode_decode`` per screened input,
        and with a fit two ``fingerprint`` records, ``snapshot``, ``candidate_train``, the
        candidate gate's ``gate_decode`` and ``gate``, ``round``, and ``restore`` after a
        rollback (an in-context round: the fingerprints, its gate, and ``round``);
        ``final_decode`` when the final decode ran; ``puzzle``.
    """
    have: dict[tuple[str, str], int] = defaultdict(int)
    for phase, record in pairs:
        have[(phase, record.op)] += 1
    need: dict[tuple[str, str], int] = {
        ("round0", op): 1 for op in ROUND0_OPS if rescored or op != "selection"
    }
    need[("after", "puzzle")] = 1
    for record in rounds:
        phase = f"round{record['round']}"
        if record["outcome"] in RAN:
            need[(phase, "redecode")] = len(record["screened"])
            need[(phase, "redecode_decode")] = len(record["screened"])
        if record["outcome"] in ("kept", "rolled_back") and record["decision"] == "in_context":
            need.update({(phase, op): 1 for op in CONTEXT_OPS})
            need[(phase, "fingerprint")] = 2
        elif record["outcome"] in ("kept", "rolled_back"):
            need.update({(phase, op): 1 for op in FIT_OPS})
            need[(phase, "fingerprint")] = 2
            if record["outcome"] == "rolled_back":
                need[(phase, "restore")] = 1
    if final_ran:
        need[("after", "final_decode")] = 1
    return [f"{p} {op}" for (p, op), n in sorted(need.items()) if have[(p, op)] < n]


def read_texts(path: Path) -> list[str]:
    """Return a text file's lines, or the lines of a Kaggle session log (a JSON array).

    Args:
        path: ``selection.log`` or a session log that holds the selection's lines.

    Returns:
        The lines, without the empty ones.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines() if not text.startswith("[") else [
        line for entry in json.loads(text) for line in str(entry.get("data", "")).splitlines()
    ]
    return [line for line in lines if line.strip()]


def kgmon_set(texts: Sequence[str]) -> set[str]:
    """Return the ``score_kgmon`` correct set: the line printed before its ``acc:`` line.

    Args:
        texts: The selection output's non-empty lines.

    Returns:
        The output keys whose top two ``score_kgmon`` grids hold the solution.

    Raises:
        ValueError: If the selection output has no ``score_kgmon`` line.
    """
    for previous, text in zip(texts, texts[1:], strict=False):
        if text.rstrip().endswith(KGMON):
            return set() if previous.strip() == "set()" else set(ast.literal_eval(previous))
    raise ValueError("no score_kgmon line in the selection output")


@dataclass(frozen=True)
class Outputs:
    """One run's candidates and selection, for the per-output comparison.

    Attributes:
        name: Label in the tables.
        candidates: Per test output (``<task>_<i>``), candidate key (``<view>.out<j>``) to
            ``{"beam_score", "score_aug", "solution"}``.
        correct_keys: Candidate keys of the ``ALL_CORRECT`` lines.
        solved: The ``score_kgmon`` correct set.
        outputs: Test outputs per task, from ``submission.json``.
        prompts: Per task, the prompt tokens of its ``decode`` records, in order; empty without
            timing records.
    """

    name: str
    candidates: dict[str, dict[str, dict]]
    correct_keys: list[str]
    solved: set[str]
    outputs: dict[str, int]
    prompts: dict[str, list[int | None]]


def load_outputs(spec: Mapping[str, str]) -> Outputs:
    """Read a run's candidates, selection output, submission, and decode prompt tokens.

    Args:
        spec: ``name``, ``candidates``, ``selection``, ``submission``, and ``timing`` (a folder
            of ``timing-*.jsonl``, or ``none``).

    Returns:
        The run's outputs.
    """
    raw = json.loads(Path(spec["candidates"]).read_text(encoding="utf-8"))
    candidates: dict[str, dict[str, dict]] = defaultdict(dict)
    for view in sorted(raw):
        for index, sample in enumerate(raw[view]):
            candidates[view.split(".", 1)[0]][f"{view}.out{index}"] = sample
    texts = read_texts(Path(spec["selection"]))
    submission = json.loads(Path(spec["submission"]).read_text(encoding="utf-8"))
    prompts: dict[str, list[int | None]] = defaultdict(list)
    timing = Path(spec.get("timing", "none"))
    if spec.get("timing", "none") != "none" and timing.is_dir():
        records, _ = read_records(sorted(timing.glob("timing-*.jsonl")))
        for record in records:
            if record.op == "decode" and record.puzzle_id is not None:
                prompts[record.puzzle_id].append(record.tokens)
    return Outputs(
        spec["name"],
        dict(candidates),
        [m.group(1) for text in texts if (m := ALL_CORRECT.match(text))],
        kgmon_set(texts),
        {t: len(v) for t, v in submission.items() if isinstance(v, list)},
        dict(prompts),
    )


def parse_baseline(words: Sequence[str]) -> dict[str, str]:
    """Read a baseline spec of ``key=value`` words and fill in the arm folder's default files.

    Args:
        words: The words after ``--baseline``.

    Returns:
        The spec with ``candidates``, ``selection``, ``submission``, and ``timing`` set.

    Raises:
        ValueError: On a word without ``=``, an unknown key, or no ``name``.
    """
    spec: dict[str, str] = {}
    for word in words:
        key, sep, value = word.partition("=")
        if not sep or key not in BASELINE_KEYS:
            raise ValueError(f"bad baseline spec word {word!r}")
        spec[key] = value
    if "name" not in spec:
        raise ValueError(f"baseline spec without name: {' '.join(words)}")
    arm = Path(spec.get("arm", "."))
    for key, default in (("candidates", "candidates.json"), ("selection", "selection.log"),
                         ("submission", "submission.json"), ("timing", "timing")):
        spec.setdefault(key, str(arm / default))
    return spec


def _rank(run: Outputs, base: str, grid: object) -> tuple[int, int, int | None]:
    """A run's candidates for an output, those equal to ``grid``, and its ``score_kgmon`` rank."""
    from arc26.nvarc.arc_decoder import score_kgmon  # the notebook's selection; needs numpy

    guesses = run.candidates.get(base, {})
    ordered = [[list(row) for row in g] for g in score_kgmon(guesses)] if guesses else []
    correct = sum(sample["solution"] == grid for sample in guesses.values())
    rank = next((i + 1 for i, g in enumerate(ordered) if g == grid), None)
    return len(guesses), correct, rank


def render_baselines(
    subject: Outputs,
    baselines: Sequence[Outputs],
    subset: Sequence[str],
    rounds: Sequence[Mapping[str, Any]],
    decodes: Sequence[Mapping[str, Any]],
    labelled: Mapping[str, Sequence[tuple[str, TimingRecord]]],
) -> list[str]:
    """Compare each test output of the subject run with the baselines.

    Args:
        subject: This run's outputs.
        baselines: The baselines' outputs, in column order.
        subset: Task ids.
        rounds: This run's round records.
        decodes: This run's decode records.
        labelled: This run's timing records by puzzle and phase (:func:`phases`).

    Returns:
        The section's Markdown lines.

    Raises:
        ValueError: If two runs mark different correct grids for an output.
    """
    grids: dict[str, object] = {}
    for run in (subject, *baselines):
        for key in run.correct_keys:
            base = key.split(".", 1)[0]
            grid = run.candidates[base][key]["solution"]
            if grids.setdefault(base, grid) != grid:
                raise ValueError(f"{run.name}: {base} has two correct grids")
    promoted = {(r["puzzle_id"], i) for r in rounds for i in r["promoted"]}
    installed = {(d["puzzle_id"], i) for d in decodes for i in d["installed"]}
    left = {(d["puzzle_id"], i) for d in decodes for i in d["inputs"]} - installed

    def source(task: str, index: int) -> str:
        if (task, index) in promoted:
            return "round 0, promoted"
        if (task, index) in installed:
            return "final decode"
        if (task, index) in left:
            return "round 0, the final decode gave none"
        return "round 0"

    runs = (subject, *baselines)
    names = ", ".join(r.name for r in runs)
    scores = [
        SubsetScore("subset", tuple(
            TaskScore(t, tuple(f"{t}_{i}" in r.solved for i in range(subject.outputs[t])))
            for t in subset
        ))
        for r in runs
    ]
    rows, totals = [], [0] * len(runs)
    for t in subset:
        for i in range(subject.outputs[t]):
            key = f"{t}_{i}"
            if key not in grids:
                continue
            stats = [_rank(r, key, grids[key]) for r in runs]
            for j, (_, correct, _) in enumerate(stats):
                totals[j] += correct
            rows.append([
                f"`{key}`", source(t, i),
                " / ".join("yes" if key in r.solved else "no" for r in runs),
                " / ".join(f"{c} of {n}" for n, c, _ in stats),
                " / ".join(str(k) if k else "-" for _, _, k in stats),
            ])
    counts = [
        sum(len(v) for b, v in r.candidates.items() if b.rsplit("_", 1)[0] in subset)
        for r in runs
    ]
    first = baselines[0]
    final_tasks = [t for t in subset if _ops(labelled.get(t, []), "final_decode", "after")]
    same = [
        t for t in final_tasks
        if [b.tokens for b in _ops(labelled[t], "final_decode", "after")]
        == first.prompts.get(t, [None])
    ]
    lost = sorted(k for k in first.solved - subject.solved if k.rsplit("_", 1)[0] in subset)
    gained = sorted(k for k in subject.solved - first.solved if k.rsplit("_", 1)[0] in subset)
    lines = [
        "",
        f"## Per test output: {names}",
        "",
        "Scores (`arc26.evaluate` over each run's `score_kgmon` set): "
        + "; ".join(
            f"{r.name} {s.score:g} of {s.tasks} ({s.solved_outputs} of {s.test_outputs} outputs)"
            for r, s in zip(runs, scores, strict=True)
        )
        + ".",
        "",
        f"Against {first.name}: lost "
        + (", ".join(f"`{k}` ({source(*_split(k))})" for k in lost) or "none")
        + "; gained "
        + (", ".join(f"`{k}` ({source(*_split(k))})" for k in gained) or "none")
        + f". Final decodes whose prompt tokens equal {first.name}'s test decode, batch for "
        f"batch: {len(same)} of {len(final_tasks)}"
        + ("" if first.prompts else f" (no {first.name} token counts)")
        + ".",
        "",
        "Rows: the outputs with a correct candidate in some run. Source: where this run's "
        "submitted candidates come from. Columns with ` / ` give the runs in the order above; "
        "rank: the correct grid's place under `score_kgmon` (1 and 2 are submitted), `-` when no "
        "candidate equals it.",
        "",
        *_table(["Output", "Source", "Solved", "Correct candidates", "Rank"], rows),
        "",
        "Correct candidates in all: "
        + "; ".join(
            f"{r.name} {n} of {count}"
            for r, n, count in zip(runs, totals, counts, strict=True)
        )
        + ".",
    ]
    return lines


def _split(key: str) -> tuple[str, int]:
    task, index = key.rsplit("_", 1)
    return task, int(index)


def render(
    run: Path, arm: str, baselines: Sequence[Outputs] = (), command: str | None = None
) -> list[str]:
    """Read one arm's records and build the report.

    Args:
        run: The pulled run folder, ``runs/<task-id>``.
        arm: The arm's name, its folder under ``run_logs/``.
        baselines: Runs to compare each test output with, in column order.
        command: The command line to quote; this run's own when None.

    Returns:
        The Markdown lines.

    Raises:
        ValueError: If the arm is not in ``stages.json`` or has no governor config.
    """
    stages = json.loads((run / "run_logs" / "stages.json").read_text(encoding="utf-8"))
    settings = next((a for a in stages["settings"]["arms"] if a["name"] == arm), None)
    ran = next((a for a in stages["stages"]["run"]["arms"] if a["name"] == arm), None)
    if settings is None or ran is None or not settings.get("rounds"):
        raise ValueError(f"{arm}: not a round arm of {run}")
    cost = CostModel.from_mapping(settings["rounds"]["governor"])
    subset = stages["settings"]["subset"]
    gate_dir = run / "run_logs" / arm / "gate"
    verdicts = {r["puzzle_id"]: r for r in read_jsonl(gate_dir, "gate")}
    plans = {(r["puzzle_id"], r["round"]): r for r in read_jsonl(gate_dir, "plan")}
    promotions = read_jsonl(gate_dir, "promote")
    rounds = read_jsonl(gate_dir, "round")
    decodes = read_jsonl(gate_dir, "decode")
    contexts = read_jsonl(gate_dir, "context")
    records, skipped = read_records(sorted((run / "run_logs" / arm / "timing").glob("*.jsonl")))
    labelled = phases(records)
    logs = sorted(run.glob("*.log"))
    texts = [] if not logs else [
        line for e in json.loads(logs[0].read_text(encoding="utf-8")) for line in
        str(e.get("data", "")).splitlines()
    ]
    stops = [t for t in texts if t.startswith("[Rank") and t.rstrip().endswith("stop!")]
    caps = [t.strip() for t in texts if "timeout after" in t]
    submission = stages["stages"]["write"]["arms"][arm]["submission"]
    end = ran["start_s"] + ran["seconds"]

    puzzle_ids = [t for t in subset if any(r.op == "puzzle" for _, r in labelled.get(t, []))]
    by_round = {(r["puzzle_id"], r["round"]): r for r in rounds}
    raised = [r for r in rounds if r["outcome"] == "raised" or r["error"]]
    raising = [
        p for p in promotions
        if p["reason"] in RAISING and "DecodeCut" not in str(p.get("error") or p.get("gate"))
    ] + [d for d in decodes if d["error"]]
    cuts = [p for p in promotions if p["reason"] == "redecode_cut"] + [
        p for p in promotions
        if p["reason"] == "gate_raised" and "DecodeCut" in json.dumps(p.get("gate"))
    ] + [d for d in decodes if d["cut"]]
    classified = 0
    for r in rounds:
        plan = plans.get((r["puzzle_id"], r["round"]))
        decision = plan["decision"] if plan else None
        classified += (
            (r["outcome"] in RAN and decision in ("gradient", "in_context"))
            or (r["outcome"] == "declined" and decision == "none")
            or r["outcome"] in NOTHING
        )
    rolled = [r for r in rounds if r["outcome"] == "rolled_back"]
    restored = [
        r for r in rolled
        if r["restored"] is True
        and r["weights_before"]
        and r["weights_before"] == r["weights_after"]
    ]
    fitted = [
        r for r in rounds
        if r["outcome"] in ("kept", "rolled_back") and r["decision"] == "gradient"
    ]
    printed = [r for r in fitted if r["weights_before"] and r["weights_after"]]
    contexts_ran = [
        r for r in rounds
        if r["outcome"] in ("kept", "rolled_back") and r["decision"] == "in_context"
    ]
    unchanged = [
        r for r in contexts_ran
        if r["weights_before"] and r["weights_before"] == r["weights_after"]
    ]
    held = [t for t in puzzle_ids if verdicts.get(t, {}).get("applied")]
    decode_ok = {
        d["puzzle_id"] for d in decodes
        if d["puzzle_id"] in held and d["name"] == "final" and (
            (not d["error"] and not d["cut"] and d["skipped"] is None)
            or d["skipped"] == "all_promoted"
        )
    }
    missing: list[str] = []
    inside: list[str] = []
    for t in puzzle_ids:
        pairs = labelled[t]
        final_ran = any(d["puzzle_id"] == t and d["skipped"] is None for d in decodes)
        own = [r for r in rounds if r["puzzle_id"] == t]
        tallies = [p["candidates"] for p in promotions if p["puzzle_id"] == t and p["round"] == 1]
        rescored = not tallies or sum(tallies) > 0
        missing += [f"{t} {op}" for op in missing_ops(pairs, own, final_ran, rescored)]
        last = pairs[-1][1]
        top = sum(
            r.seconds for p, r in pairs
            if (p == "round0" and r.op in ("train", "decode", "selection", "gate"))
            or r.op in ("round", "final_decode", "final_selection", "round_decode",
                        "round_selection")
        )
        nested = all(
            sum(r.seconds for p, r in pairs if p == f"round{k}" and r.op in (
                "redecode", "fingerprint", "snapshot", "candidate_train", "gate", "restore"
            )) <= next((r.seconds for p, r in pairs if p == f"round{k}" and r.op == "round"), 1e18)
            for k in (1, 2)
        )
        if last.op == "puzzle" and top <= last.seconds and nested:
            inside.append(t)

    lines = [
        f"# Rounds of `{arm}` (`{run.name}`)",
        "",
        "Generated by `scripts/rounds_report.py` from the pulled records only. Command, from the "
        "repository root:",
        "",
        "```bash",
        command or f"python scripts/rounds_report.py {run} --arm {arm} --out <this file>",
        "```",
        "",
        "## The decision rule's checks",
        "",
        *_table(["Check", "Measured", "Source"], [
            ["the arm ran, `solver_exit`", f"`{ran['status']}`, {ran['solver_exit']}",
             "`run_logs/stages.json`"],
            ["`puzzle` timing records", f"{len(puzzle_ids)} of {len(subset)} tasks; "
             f"{len(records)} records, {skipped} skipped lines", "`timing/`"],
            ["`[Rank r] stop!` lines; `timeout after` lines", f"{len(stops)}; {len(caps)}",
             "the session log"],
            ["valid `submission.json` (the notebook's check)",
             f"{submission['valid']}: {submission['tasks']} tasks, "
             f"{submission['test_inputs']} test inputs, {submission['problem_count']} problems",
             "`stages.json`"],
            ["the arm's end against the soft deadline",
             f"{_f(end)} s against {stages['settings']['solver_end_s']:,} s", "`stages.json`"],
            ["round records", f"{len(rounds)} for {len({r['puzzle_id'] for r in rounds})} "
             "puzzles", "`gate/round-*.jsonl`"],
            ["rounds that ran (a plan `gradient` or `in_context`), were declined (`none`), or "
             "had nothing to promote", f"{classified} of {len(rounds)}",
             "`round-`, `plan-*.jsonl`"],
            ["`raised` rounds; raising ops", f"{len(raised)}; {len(raising)}",
             "`round-`, `promote-`, `decode-*.jsonl`"],
            ["cut ops", str(len(cuts)), "`promote-`, `decode-*.jsonl`"],
            ["`rolled_back` rounds that restored the weights", f"{len(restored)} of {len(rolled)}",
             "`round-*.jsonl`"],
            ["rounds whose fit ran (kept or rolled back); with both fingerprints",
             f"{len(fitted)}; {len(printed)}", "`round-*.jsonl`"],
            ["in-context rounds that reached their gate; with the weights unchanged "
             "(fingerprints before = after)", f"{len(contexts_ran)}; {len(unchanged)}",
             "`round-*.jsonl`"],
            ["final decodes installed or skipped `all_promoted`, of the puzzles with a held-out "
             "pair", f"{len(decode_ok)} of {len(held)}", "`decode-*.jsonl`, `gate-*.jsonl`"],
            ["timing records missing; puzzles whose ops fit inside their span",
             f"{len(missing)}{' (' + ', '.join(missing) + ')' if missing else ''}; "
             f"{len(inside)} of {len(puzzle_ids)}", "`timing/`"],
            ["context records (a prompt that left out a pseudo-pair, the length guard)",
             str(len(contexts)), "`gate/context-*.jsonl`"],
        ]),
        "",
        "An op fits inside its puzzle's span when its record precedes the puzzle's `puzzle` "
        "record in its worker's file, the puzzle's top-level ops (round 0's train, decode, "
        "selection, and gate; each `round`; the decodes after it) sum to at most the `puzzle` "
        "seconds, and each round's own ops (re-decode, fingerprints, snapshot, fit, gate, "
        "restore) to at most its `round` seconds.",
        "",
        "## Each puzzle's round",
        "",
    ]
    rows = []
    for t in puzzle_ids:
        v = verdicts.get(t, {})
        for r in [r for r in rounds if r["puzzle_id"] == t]:
            plan = plans.get((t, r["round"]))
            rows.append([
                f"`{t}`", v.get("real_pairs", "n/a"),
                f"{'passed' if v.get('passed') else 'failed'} (rank {v.get('expected_rank')}, "
                f"{_f(v.get('seconds'))} s)",
                r["round"], f"`{r['outcome']}`", r["screened"], r["promoted"],
                plan["decision"] if plan else "none made",
                _f(r["estimate"]), _f(r["remaining"]), _f(r["seconds"]),
                f"`{r['weights_before']}`" if r["weights_before"] else "n/a",
                f"`{r['weights_after']}`" if r["weights_after"] else "n/a",
                r["restored"] if r["restored"] is not None else "n/a",
            ])
    lines += _table(
        ["Puzzle", "$k$", "Round-0 gate", "Round", "Outcome", "Screened", "Promoted", "Plan",
         "Estimate s", "Left s", "Round s", "Weights before", "Weights after", "Restored"],
        rows,
    )
    lines += [
        "",
        "Weights before: the adapter's blake2b digest before the candidate fit; after: at the "
        "round's end (the kept adapter, or the restored one). Round s: the round record's span.",
        "",
        "## Promotion records",
        "",
    ]
    rows = []
    for p in promotions:
        red, gate = p.get("redecode"), p.get("gate")
        rows.append([
            f"`{p['puzzle_id']}`", p["round"], p["test_index"], p["candidates"],
            p["votes"][:3], _f(p["agreement"], 4), f"`{p['reason']}`",
            "n/a" if not red else f"agreed {red['agreed']}, votes {red['votes'][:3]} of "
            f"{red['candidates']}, cut {red['cut']}, {_f(red['seconds'])} s",
            "n/a" if not gate else f"passed {gate['passed']}, rank {gate['expected_rank']}, "
            f"votes {gate['votes'][:3]} of {gate['candidates']}, {_f(gate['seconds'])} s",
        ])
    lines += _table(
        ["Puzzle", "Round", "Input", "Candidates", "Top votes", "Agreement", "Reason",
         "Re-decode", "Candidate gate"],
        rows,
    )
    lines += ["", "## The governor's estimate against the measured seconds", ""]
    rows = []
    for (t, k), plan in sorted(plans.items()):
        shape = plan["shape"]
        variant = plan_variant(plan)
        terms = estimate_terms(cost, plan)
        redecode_batches = (
            shape["in_context"]["redecode_batches"] if variant == "in_context"
            else shape["redecode_batches"]
        )
        ratio = f" at length ratio {shape['in_context']['length_ratio']:.3f}" if (
            variant == "in_context"
        ) else ""
        pairs = labelled[t]
        phase = f"round{k}"
        record = by_round.get((t, k), {})
        last_round = max(r for (u, r) in plans if u == t)
        final = next((d for d in decodes if d["puzzle_id"] == t and d["name"] == "final"), None)
        final = final if k == last_round else None
        final_s = final["seconds"] if final else 0.0
        measured = {
            "fit": _sum(pairs, "candidate_train", phase),
            "redecode": _sum(pairs, "redecode", phase),
            "gate": _sum(pairs, "gate", phase),
            "test": final_s,
            "copies": sum(_sum(pairs, op, phase) for op in ("snapshot", "fingerprint", "restore")),
        }
        fit = _ops(pairs, "candidate_train", phase)
        gate_batches = _ops(pairs, "gate_decode", phase)
        rows += [
            [f"`{t}`", "candidate fit",
             "none (in-context)" if variant == "in_context"
             else f"{shape['train_tokens']:,} tokens, {shape['train_steps']} steps",
             _f(terms["fit"]), _f(measured["fit"]),
             f"{measured['fit'] / terms['fit']:.2f}" if fit else "n/a"],
            [f"`{t}`", "re-decodes", f"{redecode_batches} batches",
             _f(terms["redecode"]), _f(measured["redecode"]),
             f"{measured['redecode'] / terms['redecode']:.2f}" if terms["redecode"] else "n/a"],
            [f"`{t}`", "gate",
             f"{shape['gate_batches']} batches{ratio} ({len(gate_batches)} ran)",
             _f(terms["gate"]), _f(measured["gate"]),
             f"{measured['gate'] / terms['gate']:.2f}" if terms["gate"] else "n/a"],
            [f"`{t}`", "test decodes reserved (the final decode)",
             f"{shape['test_batches']} batches{ratio} ({final['batches'] or 0} ran"
             f"{', skipped ' + final['skipped'] if final and final['skipped'] else ''})"
             if final else f"{shape['test_batches']} batches",
             _f(terms["test"]), _f(measured["test"]),
             f"{measured['test'] / terms['test']:.2f}" if terms["test"] else "n/a"],
            [f"`{t}`", "overhead; snapshot, fingerprints, restore", "", _f(terms["overhead"]),
             _f(measured["copies"]), ""],
            [f"`{t}`", "**round and final decode**",
             f"{variant}, plan `{plan['decision']}`; slowest round-0 batch "
             f"{_f(shape['observed_batch'])} s; estimate {_f(plan['estimates'][variant])} s",
             f"**{_f(terms['total'])}**", f"**{_f(record.get('seconds', 0.0) + final_s)}**",
             f"**{(record.get('seconds', 0.0) + final_s) / terms['total']:.2f}**"],
        ]
    lines += _table(["Puzzle", "Term", "Priced", "Estimate s", "Measured s", "Ratio"], rows)
    lines += [
        "",
        "Estimate: the governor's terms with the margin, `CostModel` from the config embedded "
        "in the build (`stages.json`), each decode term at the plan's slowest round-0 batch "
        "(an in-context round's gate and test decodes also at its length ratio); their sum "
        "equals the plan's estimate of its variant. Measured: the `candidate_train`, "
        "`redecode`, and candidate `gate` records of the round, the final decode record's "
        "seconds, and the `snapshot`, `fingerprint`, and `restore` records; the last row the "
        "`round` record plus the final decode.",
        "",
        "## Final decodes",
        "",
    ]
    rows = []
    for d in decodes:
        pairs = labelled.get(d["puzzle_id"], [])
        batches = _ops(pairs, "final_decode", "after")
        tokens = [b.tokens for b in batches if b.tokens is not None]
        rows.append([
            f"`{d['puzzle_id']}`", d["name"], d["prompts"], d["inputs"], d["installed"],
            sorted(set(d["inputs"]) - set(d["installed"])) or "none",
            f"{d['batches'] if d['batches'] is not None else 0} of {d['expected']}",
            d["skipped"] or "no", _f(d["estimate"]), _f(d["remaining"]), _f(d["seconds"]),
            ", ".join(_f(b.seconds) for b in batches) or "n/a",
            f"{sum(tokens) // len(tokens):,}" if tokens else "n/a",
        ])
    lines += _table(
        ["Puzzle", "Name", "Prompts", "Inputs", "Installed", "No candidate", "Batches",
         "Skipped", "Estimate s", "Left s", "Seconds", "Batch seconds", "Prompt tokens"],
        rows,
    )
    lines += [
        "",
        "No candidate: inputs the decode gave none, which keep round 0's candidates. Estimate: "
        "the final decode's own check, $D(4 n_\\text{open}, 1, s)$ plus the overhead.",
        "",
        "## The length guard",
        "",
    ]
    if contexts:
        lines += _table(
            ["Puzzle", "Round", "Decode", "Kept", "Dropped", "Lengths", "Limit"],
            [[f"`{c.get('puzzle_id')}`", c.get("round_added", c.get("round")), c.get("name"),
              c.get("kept"), c.get("dropped"), c.get("lengths"), c.get("limit")]
             for c in contexts],
        )
    else:
        lines.append("No context record: no decode's prompts left out a pseudo-pair.")
    if baselines:
        subject = load_outputs({
            "name": run.name,
            "candidates": str(run / "run_logs" / arm / "candidates.json"),
            "selection": str(run / "run_logs" / arm / "selection.log"),
            "submission": str(run / "run_logs" / arm / "submission.json"),
            "timing": "none",
        })
        lines += render_baselines(subject, baselines, subset, rounds, decodes, labelled)
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Write the round report of one arm.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when None.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("run", type=Path, help="the pulled run folder, runs/<task-id>")
    parser.add_argument("--arm", required=True, help="the arm's folder under run_logs/")
    parser.add_argument("--out", type=Path, required=True, help="the Markdown file to write")
    parser.add_argument("--baseline", nargs="+", action="append", default=[],
                        metavar="KEY=VALUE", help="a run to compare each test output with")
    args = parser.parse_args(argv)
    baselines = [load_outputs(parse_baseline(words)) for words in args.baseline]
    words = sys.argv[1:] if argv is None else list(argv)
    command = "python scripts/rounds_report.py " + " ".join(words).replace(str(Path.home()), "~")
    lines = render(args.run, args.arm, baselines, command if args.baseline else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).replace("<this file>", str(args.out)) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
