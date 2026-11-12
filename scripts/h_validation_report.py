r"""H's validation table and gate G4's line from the arms runs (task P3.7; B5 P3.7, B7 G4, A8).

Purpose: gate G4 reads the fraction of the gap between the zero adapter (a) and the SGD adapter (c)
that H (b) closes, on GEN (the held-out generated puzzles) and on DEV40. This script turns the
pulled logs of runs ``P3.7-01`` (DEV40) and ``P3.7-02`` (GEN) into that table: per arm and set the
task accuracy with voting (the competition metric, two attempts), the attempt-1 exact match (on
GEN, B5's held-out pair exact match), the teacher-forced CE of the answer, and the seconds per
puzzle by op; the fractions $f_b$ and $f_d$ with paired bootstrap intervals; per-source rows on
GEN; H's load check; and the G4 line. Runbook ``docs/runbooks/h_validation.md``; decision record
ADR-019.

Inputs: one or more ``--run LABEL=<run_logs folder>`` (``kaggle_ops.py pull`` writes
``runs/<task-id>/run_logs``; the arms' files are under its ``arms/``). Outputs: ``--out`` (a
markdown report) and ``--json`` (the same numbers). Nothing on Kaggle is touched.

Method. The paired set $P$ of a run is its puzzles on which every arm decoded to the end (status
``done``) and was scored; a soft-deadline cut leaves out the same puzzles in every arm (the queue's
shuffled order, ``arc26.arms``). For a metric $m$ (the sum of task scores, the share of outputs
whose attempt 1 is exact, the mean CE),

$$f_x = \frac{m_x - m_a}{m_c - m_a}, \qquad x \in \{b, d\},$$

the share of the (a) to (c) gap the arm closes (CE's gap is negative, the ratio the same). The
interval is the 5th and 95th percentile of $f_x$ over 10,000 paired bootstrap resamples of $P$
(one fixed seed); a resample with $m_c = m_a$ is undefined and counted. On task accuracy $f_x$ is
undefined unless the SGD arm beats the zero adapter by at least 2 tasks, $m_c - m_a \ge 2$
(DEV40's run-to-run spread is 3 tasks, ``P2.1-03``; the same 2 puzzles on GEN).

G4's rows (the briefs ``docs/prerun/P3.7-01.md`` and ``P3.7-02.md``, section 1), read in this
order, each set in exactly one: **No gap** first, when the SGD arm does not beat the zero adapter
on either metric, $m^{\text{acc}}_c - m^{\text{acc}}_a < 2$ and $m^{\text{ce}}_c \ge
m^{\text{ce}}_a$; else $f_b$ on task accuracy (on CE when accuracy's is undefined): **Good** at
least 0.5, **Borderline** 0.25 to 0.5, **Bad** below 0.25. The consequence depends on whether H's
meta-training has ended (``--training-ended``: the chain stopped for good, ``MT-04`` at step 5,600
or a Kill or failure row with no resume planned; by default it has not, as while ``MT-03`` runs).
Not ended: Bad means H's training continues and P3.7 repeats on the chain's final best point;
Phase 4 does not stop. Ended: Bad means Submission 2's system is final and Phase 4 stops for the
deadline. No gap on one set: G4 reads the other set alone; on both: not ended, training continues
and P3.7 repeats; ended, H does not enter Phase 4 for the deadline.

``--smoke`` runs the whole path on the CPU first (BRIEF A8): a tiny H from ``train.py``'s smoke
mode (the toy config's smoke section), GEN from the episode fixture, the load check, every arm
(``arc26.arms.run`` with one in-process worker on the tiny random Qwen3, the GPU stack's unsloth
and datasets replaced by stand-ins, as the CPU tests replace them), the selection and scores, then
this report, all under ``runs/P3.7-smoke/``. The smoke's arms are (a) to (g), with ``train_log``
on, so the start comparison of run ``P3.7-03`` and the merge comparisons of run ``P3.9-02`` run
too. ``--smoke --layout CONFIG`` (task P3.9b)
trains its tiny H as ``P3.9-01`` trained the real one, the MT smoke and then the P3.9 smoke
warm-started from its step 8 (``objective: fomaml``, the inner-rate probe), and lays ``best.pt``
and the run logs out at the paths the build config's ``adapters`` read (``checkpoint``,
``load_check.run_logs``) under a folder named by their input, so the load check and every arm
read a FOMAML ``best.pt`` where the notebook finds the real one; outputs under
``runs/P3.9-02-smoke/``.

Usage::

    python scripts/h_validation_report.py --run DEV40=runs/P3.7-01/run_logs \
        --run GEN=runs/P3.7-02/run_logs --out reports/P3.7/report.md --json reports/P3.7/report.json
    python scripts/h_validation_report.py ... --training-ended   # after the chain's final point
    python scripts/h_validation_report.py --smoke
    python scripts/h_validation_report.py --smoke --layout configs/P3.9-02.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.machinery
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARMS_DIR = "arms"
# The roles of the arms, by source: (a) the zero adapter, (b) H, (c) SGD, (d) H plus the polish,
# (e) the polish alone.
# (f) is sgd's training started from H's adapter (amendment 8, run P3.7-03); (g) is sgd's training
# of a fresh LoRA on the base with H's adapter merged in (task P3.9b amendment 1, run P3.9-02).
ROLE = {
    "base": "a", "hyper": "b", "sgd": "c", "hyper_polish": "d", "polish": "e", "hyper_fit": "f",
    "hyper_merge_fit": "g",
}  # fmt: skip
METRICS = ("accuracy", "attempt_1", "ce")
METRIC_TITLES = {
    "accuracy": "task accuracy (2 attempts)",
    "attempt_1": "attempt-1 exact match",
    "ce": "answer CE",
}
BOOTSTRAP = 10_000
SEED = 0
# DEV40's noise floor on task accuracy: differences under 2 tasks are noise (CLAUDE.md; three
# runs of one configuration spanned 3 tasks, P2.1-03). GEN takes the same count of puzzles.
FLOOR_TASKS = 2.0
# Run P3.7-03's reading of (f) against (c) (its brief, section 1): the paired difference in tasks
# against the 2-task noise floor, and how soon (f)'s training loss reaches (c)'s final one: the
# trailing mean over LOSS_WINDOW steps (one view per step at batch 1, so single steps are noisy),
# "sooner" when the median puzzle gets there within FASTER_SHARE of (c)'s steps.
START_BETTER, START_SAME, START_WORSE = "Start helps", "Same score", "Start hurts"
# Run P3.9-02's rows for (g) against (c) (its brief, section 1), and its line against (f), on the
# same 2-task floor.
MERGE_ROWS = ("Good", "Neutral", "Bad")
MERGE_FIT_ROWS = ("Merge ahead", "Same score", "Merge behind")
LOSS_WINDOW = 8
FASTER_SHARE = 0.5
LOSS_STEPS = (1, 8, 16, 32, 64, 96, 128)
# G4's branches on f_b (BRIEF B7).
G4_PLANNED, G4_HEAVY = 0.5, 0.25
# Reference rows on DEV40: the SGD arm's build at rank 16 on q, k, v, o in two earlier runs.
DEV40_REFERENCES = (
    ("P2.6-01, its DEV40 subset (reports/P2.6-01/tasks.md)", 10.5),
    ("P1.3-01 arm 2, r16-attn (docs/experiments/P1.3-01.md)", 10.5),
)
# Timing ops summed per arm and puzzle; the span is the arm's "puzzle" record.
OPS = (
    ("train", ("train", "candidate_train")),
    ("encode", ("encode",)),
    ("h_forward", ("h_forward",)),
    ("decode", ("decode",)),
    ("rescoring", ("selection",)),
    ("ce", ("ce",)),
)


# --- reading a run -------------------------------------------------------------------------------


def _jsonl(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut by a hard kill
    return rows


def read_run(run_logs: str | Path) -> dict[str, Any]:
    """Read one arms run: its summary, records, scores, timing, load check, and GEN index.

    Args:
        run_logs: The pulled ``run_logs`` folder (or its ``arms/`` folder).

    Returns:
        ``arms`` (name to source, in order), ``order``, ``records`` (per puzzle and arm),
        ``scores`` (per arm and puzzle), ``timing`` (per arm, puzzle, and op: seconds),
        ``load_check``, ``sources`` (GEN's source per puzzle, or empty), and ``summary``.

    Raises:
        FileNotFoundError: If the folder has no ``arms.json``.
    """
    root = Path(run_logs)
    if (root / ARMS_DIR).is_dir():
        root = root / ARMS_DIR
    summary = json.loads((root / "arms.json").read_text(encoding="utf-8"))
    arms = {arm["name"]: arm["source"] for arm in summary["settings"]["arms"]}
    records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _jsonl(sorted(glob.glob(str(root / "records-*.jsonl")))):
        records[row["puzzle_id"]][row["arm"]] = row
    scores: dict[str, dict[str, Any]] = {}
    timing: dict[str, dict[str, dict[str, float]]] = {}
    for name in arms:
        path = root / name / "scores.json"
        if path.is_file():
            scores[name] = json.loads(path.read_text(encoding="utf-8"))["puzzles"]
        per: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in _jsonl(sorted(glob.glob(str(root / name / "timing" / "timing-*.jsonl")))):
            if row.get("puzzle_id") is not None:
                per[row["puzzle_id"]][row["op"]] += float(row["seconds"])
        timing[name] = per
    check = None
    if (root / "load_check.json").is_file():
        check = json.loads((root / "load_check.json").read_text(encoding="utf-8"))
    sources = {}
    gen = summary.get("gen")
    if gen and gen.get("index"):
        sources = {entry["key"]: entry["source"] for entry in gen["index"]}
    return {
        "arms": arms,
        "order": summary.get("order") or [],
        "records": records,
        "scores": scores,
        "timing": timing,
        "load_check": check,
        "sources": sources,
        "summary": summary,
    }


def paired(run: Mapping[str, Any]) -> list[str]:
    """Return the run's paired set: puzzles every arm decoded to the end and scored, in order.

    Args:
        run: ``read_run``'s result.

    Returns:
        The puzzle ids, in the queue's order.
    """
    out = []
    for key in run["order"]:
        rows = run["records"].get(key, {})
        if all(
            rows.get(name, {}).get("status") == "done" and key in run["scores"].get(name, {})
            for name in run["arms"]
        ):
            out.append(key)
    return out


# --- the arithmetic ------------------------------------------------------------------------------


def puzzle_values(run: Mapping[str, Any], name: str, key: str) -> dict[str, float]:
    """Return one arm's per-puzzle values: task score, attempt-1 matches and outputs, mean CE.

    Args:
        run: ``read_run``'s result.
        name: The arm.
        key: The puzzle.

    Returns:
        ``accuracy`` (the task score), ``a1`` and ``outputs`` (attempt-1 exact matches and the
        test outputs), and ``ce`` (the mean CE over the outputs, NaN without CE records).
    """
    score = run["scores"][name][key]
    ces = run["records"][key][name].get("ce") or []
    return {
        "accuracy": float(score["score"]),
        "a1": float(sum(score["attempt_1"])),
        "outputs": float(len(score["attempt_1"])),
        "ce": sum(item["ce"] for item in ces) / len(ces) if ces else math.nan,
    }


def aggregate(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Return a set's metrics from per-puzzle values.

    Args:
        values: ``puzzle_values`` of one arm over the puzzles of the set.

    Returns:
        ``accuracy`` (the sum of task scores, in tasks), ``attempt_1`` (attempt-1 exact matches
        over the outputs), and ``ce`` (the mean of the per-puzzle CEs).
    """
    outputs = sum(v["outputs"] for v in values)
    ces = [v["ce"] for v in values]
    return {
        "accuracy": sum(v["accuracy"] for v in values),
        "attempt_1": sum(v["a1"] for v in values) / outputs if outputs else math.nan,
        "ce": sum(ces) / len(ces) if ces else math.nan,
    }


def gap_closed(m_x: float, m_a: float, m_c: float) -> float | None:
    """Return $f_x = (m_x - m_a) / (m_c - m_a)$, or None when $m_c = m_a$ or a value is NaN.

    Args:
        m_x: The arm's metric.
        m_a: The zero adapter's.
        m_c: The SGD adapter's.

    Returns:
        The fraction, or None.
    """
    if any(math.isnan(v) for v in (m_x, m_a, m_c)) or m_c == m_a:
        return None
    return (m_x - m_a) / (m_c - m_a)


def bootstrap(
    values: Mapping[str, Sequence[Mapping[str, float]]],
    arm: str,
    metric: str,
    resamples: int = BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    """Return the paired bootstrap interval of $f$ for one arm and metric.

    Args:
        values: Per role (``a``, ``c``, and ``arm``) the per-puzzle values over the paired set, in
            one puzzle order.
        arm: The role whose fraction is computed (``b`` or ``d``).
        metric: One of ``METRICS``.
        resamples: Bootstrap resamples.
        seed: The resampling seed (one fixed seed, CLAUDE.md).

    Returns:
        ``low`` and ``high`` (the 5th and 95th percentiles over the defined resamples, None when
        none is defined) and ``undefined`` (resamples with $m_c = m_a$).
    """
    n = len(values["a"])
    rng = random.Random(seed)
    found = []
    undefined = 0
    for _ in range(resamples):
        index = [rng.randrange(n) for _ in range(n)]
        m = {
            role: aggregate([values[role][i] for i in index])[metric] for role in ("a", "c", arm)
        }
        f = gap_closed(m[arm], m["a"], m["c"])
        if f is None:
            undefined += 1
        else:
            found.append(f)
    found.sort()
    if not found:
        return {"low": None, "high": None, "undefined": undefined}
    pick = lambda q: found[min(len(found) - 1, int(q * (len(found) - 1) + 0.5))]  # noqa: E731
    return {"low": pick(0.05), "high": pick(0.95), "undefined": undefined}


NO_GAP, GOOD, BORDERLINE, BAD, UNDEFINED = "No gap", "Good", "Borderline", "Bad", "Undefined"
# The sets G4 combines; per-source tables (GEN barc, GEN rearc) get lines but no vote.
MAIN_SETS = ("DEV40", "GEN")
# What each row means, by whether H's meta-training has ended (the briefs, section 1).
CONSEQUENCES = {
    (GOOD, False): "Phase 4 as planned",
    (GOOD, True): "Phase 4 as planned",
    (BORDERLINE, False): "Phase 4 with the heaviest polish that fits",
    (BORDERLINE, True): "Phase 4 with the heaviest polish that fits",
    (BAD, False): (
        "H's training continues and P3.7 repeats on the chain's final best point; "
        "Phase 4 does not stop"
    ),
    (BAD, True): "Submission 2's system is final and Phase 4 stops for the deadline",
    (NO_GAP, False): (
        "this set measures no closure: G4 reads the other set alone; on both sets, H's training "
        "continues and P3.7 repeats on the chain's final best point"
    ),
    (NO_GAP, True): (
        "this set measures no closure: G4 reads the other set alone; on both sets, H does not "
        "enter Phase 4 for the deadline"
    ),
    (UNDEFINED, False): "no reading (a metric is missing); the human decides at G4",
    (UNDEFINED, True): "no reading (a metric is missing); the human decides at G4",
}


def g4_row(f: float | None) -> str:
    """Return G4's row for $f_b$ (BRIEF B7), once No gap is ruled out.

    Args:
        f: $f_b$, or None.

    Returns:
        ``GOOD``, ``BORDERLINE``, ``BAD``, or ``UNDEFINED``.
    """
    if f is None:
        return UNDEFINED
    if f >= G4_PLANNED:
        return GOOD
    if f >= G4_HEAVY:
        return BORDERLINE
    return BAD


def g4_branch(f: float | None, training_ended: bool = False) -> str:
    """Return G4's row and its consequence for $f_b$, once No gap is ruled out.

    Args:
        f: $f_b$, or None.
        training_ended: Whether H's meta-training chain has stopped for good.

    Returns:
        The row and its consequence, in words.
    """
    row = g4_row(f)
    return f"{row}: {CONSEQUENCES[(row, training_ended)]}"


def no_gap(table: Mapping[str, Any]) -> bool | None:
    r"""Return whether the SGD arm beats the zero adapter on neither metric (G4's No gap row).

    Signed: $m^{\text{acc}}_c - m^{\text{acc}}_a < 2$ (tasks, or puzzles on GEN; higher is
    better) and $m^{\text{ce}}_c \ge m^{\text{ce}}_a$ (lower is better). Read before the other
    rows, which it overrides: then $f_b$ on CE may be defined but measures no closure.

    Args:
        table: ``analyse``'s result.

    Returns:
        True or False, or None when an arm (a) or (c) or a metric is missing.
    """
    roles = {arm["role"]: arm["metrics"] for arm in table["arms"].values()}
    m_a, m_c = roles.get("a"), roles.get("c")
    if not m_a or not m_c or any(
        math.isnan(m[k]) for m in (m_a, m_c) for k in ("accuracy", "ce")
    ):
        return None
    return m_c["accuracy"] - m_a["accuracy"] < FLOOR_TASKS and m_c["ce"] >= m_a["ce"]


def op_seconds(run: Mapping[str, Any], name: str, keys: Sequence[str]) -> dict[str, float]:
    """Return one arm's mean seconds per puzzle, its span and each op of ``OPS``.

    Args:
        run: ``read_run``'s result.
        name: The arm.
        keys: The puzzles averaged over.

    Returns:
        ``span`` (the arm's ``puzzle`` record) and each op's mean seconds, over ``keys``.
    """
    per = run["timing"].get(name, {})
    out = {"span": 0.0, **{op: 0.0 for op, _ in OPS}}
    if not keys:
        return out
    for key in keys:
        ops = per.get(key, {})
        out["span"] += ops.get("puzzle", 0.0)
        for op, names in OPS:
            out[op] += sum(ops.get(n, 0.0) for n in names)
    return {op: value / len(keys) for op, value in out.items()}


def analyse(
    run: Mapping[str, Any], label: str, keys: Sequence[str] | None = None
) -> dict[str, Any]:
    """Compute one set's table: metrics per arm, $f_b$ and $f_d$ with intervals, seconds.

    Args:
        run: ``read_run``'s result.
        label: The set's name (``DEV40``, ``GEN``, ``GEN BARC``).
        keys: Puzzles to use, a subset of the paired set; None uses all of it.

    Returns:
        ``label``, ``puzzles``, ``arms`` (per arm: role, source, metrics, seconds), and
        ``fractions`` (per role b and d and metric: ``f``, the interval, ``floor`` when the
        accuracy gap is under the noise floor).
    """
    keys = paired(run) if keys is None else list(keys)
    by_role = {ROLE[source]: name for name, source in run["arms"].items()}
    values = {
        name: [puzzle_values(run, name, key) for key in keys] for name in run["arms"]
    }
    arms = {}
    for name, source in run["arms"].items():
        arms[name] = {
            "role": ROLE[source],
            "source": source,
            "metrics": aggregate(values[name]) if keys else None,
            "seconds": op_seconds(run, name, keys),
        }
    fractions: dict[str, dict[str, Any]] = {}
    if keys and "a" in by_role and "c" in by_role:
        m_a = arms[by_role["a"]]["metrics"]
        m_c = arms[by_role["c"]]["metrics"]
        for role in ("b", "d"):
            if role not in by_role:
                continue
            m_x = arms[by_role[role]]["metrics"]
            fractions[role] = {}
            for metric in METRICS:
                f = gap_closed(m_x[metric], m_a[metric], m_c[metric])
                # Signed: accuracy's f counts only when SGD beats the zero adapter by the floor.
                floor = metric == "accuracy" and m_c[metric] - m_a[metric] < FLOOR_TASKS
                sample = {
                    "a": values[by_role["a"]],
                    "c": values[by_role["c"]],
                    role: values[by_role[role]],
                }
                interval = bootstrap(sample, role, metric)
                fractions[role][metric] = {
                    "f": None if floor else f,
                    "raw": f,
                    "floor": floor,
                    **interval,
                }
    start = merge = merge_fit = None
    if keys and "c" in by_role and "f" in by_role:
        start = start_comparison(run, keys, by_role["c"], by_role["f"], values)
    if keys and "c" in by_role and "g" in by_role:
        merge = start_comparison(
            run, keys, by_role["c"], by_role["g"], values, roles=("c", "g"), rows=MERGE_ROWS
        )
    if keys and "f" in by_role and "g" in by_role:
        merge_fit = start_comparison(
            run, keys, by_role["f"], by_role["g"], values, roles=("f", "g"), rows=MERGE_FIT_ROWS
        )
    return {
        "label": label, "puzzles": len(keys), "arms": arms, "fractions": fractions, "start": start,
        "merge": merge, "merge_fit": merge_fit,
    }  # fmt: skip


def trailing(losses: Sequence[float], window: int = LOSS_WINDOW) -> list[float]:
    """Return the trailing mean of a per-step loss, over at most ``window`` steps.

    Args:
        losses: The loss of steps 1, 2, ... in order.
        window: Steps averaged.

    Returns:
        One value per step: the mean of that step's loss and the ``window - 1`` before it.
    """
    out = []
    for i in range(len(losses)):
        part = losses[max(0, i + 1 - window) : i + 1]
        out.append(sum(part) / len(part))
    return out


def reach_step(start: Sequence[float], reference: Sequence[float]) -> int | None:
    """Return the first step at which ``start``'s trailing loss is at most ``reference``'s final.

    Args:
        start: (f)'s per-step loss.
        reference: (c)'s per-step loss on the same puzzle.

    Returns:
        The step (1-based), or None when it never gets there or a curve is empty.
    """
    if not start or not reference:
        return None
    target = trailing(reference)[-1]
    for step, value in enumerate(trailing(start), start=1):
        if value <= target:
            return step
    return None


def _losses(run: Mapping[str, Any], name: str, key: str) -> list[float]:
    log = run["records"][key][name].get("train_log") or []
    return [entry["loss"] for entry in sorted(log, key=lambda entry: entry["step"])]


def start_comparison(
    run: Mapping[str, Any],
    keys: Sequence[str],
    c_name: str,
    f_name: str,
    values: Mapping[str, Sequence[Mapping[str, float]]],
    resamples: int = BOOTSTRAP,
    seed: int = SEED,
    roles: tuple[str, str] = ("c", "f"),
    rows: tuple[str, str, str] = (START_BETTER, START_SAME, START_WORSE),
) -> dict[str, Any]:
    r"""Compare (f), sgd's training from H's adapter, with (c), from the initial weights.

    The paired difference $\Delta = m^{\text{acc}}_f - m^{\text{acc}}_c$ in tasks with a 90%
    paired bootstrap interval; the row (``START_BETTER`` at $\Delta \ge 2$, ``START_WORSE`` at
    $\Delta \le -2$, else ``START_SAME``); the mean trailing training loss at ``LOSS_STEPS``;
    and per puzzle the step at which (f)'s trailing loss reaches (c)'s final one. The same
    reading compares (g) with (c) and with (f) (run ``P3.9-02``): ``c_name`` is then the
    reference arm and ``f_name`` the arm read against it, named by ``roles`` and ``rows``.

    Args:
        run: ``read_run``'s result.
        keys: The paired puzzles.
        c_name: The (c) arm, the reference.
        f_name: The (f) arm, read against it.
        values: ``puzzle_values`` per arm over ``keys``.
        resamples: Bootstrap resamples.
        seed: The resampling seed.
        roles: The two arms' roles, reference first: the keys of ``loss``.
        rows: The row names at $\Delta \ge 2$, between, and at $\Delta \le -2$.

    Returns:
        ``roles``, ``delta``, ``low``, ``high``, ``row``, ``straddles``, ``loss`` (per role,
        step to mean trailing loss), ``reach`` (per puzzle, None when never),
        ``reach_median``, ``reached`` (puzzles that got there), ``logged`` (puzzles with both
        curves), ``steps`` (median steps of (c)'s curve), and ``sooner`` (whether the median
        reach is within ``FASTER_SHARE`` of them).
    """
    c, f = values[c_name], values[f_name]
    delta = sum(v["accuracy"] for v in f) - sum(v["accuracy"] for v in c)
    rng = random.Random(seed)
    n = len(keys)
    found = []
    for _ in range(resamples):
        index = [rng.randrange(n) for _ in range(n)]
        found.append(sum(f[i]["accuracy"] - c[i]["accuracy"] for i in index))
    found.sort()
    pick = lambda q: found[min(len(found) - 1, int(q * (len(found) - 1) + 0.5))]  # noqa: E731
    low, high = pick(0.05), pick(0.95)
    better, same, worse = rows
    row = (
        better if delta >= FLOOR_TASKS
        else worse if delta <= -FLOOR_TASKS
        else same
    )  # fmt: skip
    straddles = any(low < edge <= high for edge in (-FLOOR_TASKS, FLOOR_TASKS))
    ref, arm = roles
    curves: dict[str, dict[str, list[float]]] = {ref: {}, arm: {}}
    reach: dict[str, int | None] = {}
    lengths = []
    for key in keys:
        lc, lf = _losses(run, c_name, key), _losses(run, f_name, key)
        if lc:
            curves[ref][key] = trailing(lc)
            lengths.append(len(lc))
        if lf:
            curves[arm][key] = trailing(lf)
        if lc and lf:
            reach[key] = reach_step(lf, lc)
    loss = {}
    for role, per in curves.items():
        loss[role] = {}
        for step in LOSS_STEPS:
            at = [curve[step - 1] for curve in per.values() if len(curve) >= step]
            loss[role][step] = sum(at) / len(at) if at else None
    reached = sorted(v for v in reach.values() if v is not None)
    # The median over every logged puzzle, a puzzle that never gets there counting as last.
    ranked = sorted(v if v is not None else math.inf for v in reach.values())
    median = ranked[(len(ranked) - 1) // 2] if ranked else None
    steps = sorted(lengths)[(len(lengths) - 1) // 2] if lengths else None
    sooner = None
    if median is not None and steps:
        sooner = median <= FASTER_SHARE * steps
    return {
        "roles": [ref, arm],
        "delta": delta, "low": low, "high": high, "row": row, "straddles": straddles,
        "loss": loss, "reach": reach, "reach_median": None if median == math.inf else median,
        "reached": len(reached), "logged": len(reach), "steps": steps, "sooner": sooner,
    }  # fmt: skip


def g4_line(table: Mapping[str, Any], training_ended: bool = False) -> dict[str, Any]:
    """Return G4's reading of one set: No gap first, else $f_b$ on accuracy (else CE) and its row.

    Args:
        table: ``analyse``'s result.
        training_ended: Whether H's meta-training chain has stopped for good.

    Returns:
        ``metric``, ``f``, ``low``, ``high``, ``row``, ``branch`` (the row and its consequence),
        ``no_gap``, and ``straddles`` (whether the interval crosses a branch boundary).
    """
    b = table["fractions"].get("b") or {}
    entry = b.get("accuracy") or {}
    metric = "accuracy"
    if entry.get("f") is None:
        metric, entry = "ce", b.get("ce") or {}
    f, low, high = entry.get("f"), entry.get("low"), entry.get("high")
    straddles = None
    if low is not None and high is not None:
        straddles = any(low < edge <= high for edge in (G4_HEAVY, G4_PLANNED))
    gap = no_gap(table)
    row = NO_GAP if gap else g4_row(f)
    return {
        "metric": metric,
        "f": f,
        "low": low,
        "high": high,
        "row": row,
        "branch": f"{row}: {CONSEQUENCES[(row, training_ended)]}",
        "no_gap": gap,
        "straddles": None if row == NO_GAP else straddles,
    }


def g4_decision(lines: Mapping[str, Mapping[str, Any]], training_ended: bool = False) -> str:
    """Return G4's combined reading of DEV40 and GEN (the briefs, section 1).

    Args:
        lines: ``g4_line`` results by set label (the main sets only).
        training_ended: Whether H's meta-training chain has stopped for good.

    Returns:
        The decision in words.
    """
    measured = {k: v for k, v in lines.items() if v["row"] != NO_GAP}
    if lines and not measured:
        if training_ended:
            return (
                "No gap on every set: G4 cannot measure closure; H does not enter Phase 4 for "
                "the deadline."
            )
        return (
            "No gap on every set: G4 cannot measure closure; H's training continues and P3.7 "
            "repeats on the chain's final best point."
        )
    if not measured:
        return "No set to read."
    alone = len(measured) < len(lines)
    rows = {v["row"] for v in measured.values()}
    straddle = any(v["straddles"] for v in measured.values())
    prefix = (
        f"No gap on {', '.join(k for k in lines if k not in measured)}: G4 reads "
        f"{', '.join(measured)} alone. " if alone else ""
    )
    if len(rows) > 1 or straddle:
        detail = "; ".join(f"{k}: {v['row']}" for k, v in measured.items())
        return prefix + (
            f"The sets fall in different G4 branches or an interval crosses a boundary ({detail}): "
            "the human decides at G4."
        )
    (row,) = rows
    return prefix + f"{row}: {CONSEQUENCES[(row, training_ended)]}."


# --- the report ----------------------------------------------------------------------------------


def _num(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def render(
    tables: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Any]],
    training_ended: bool = False,
) -> str:
    """Render the markdown report: the load checks, every set's table, and the G4 lines.

    Args:
        tables: ``analyse``'s results, in order.
        runs: ``read_run``'s results by label (for the load checks and completion counts).
        training_ended: Whether H's meta-training chain has stopped for good.

    Returns:
        The report text.
    """
    lines = ["# P3.7: H's validation (gate G4)", ""]
    lines += [
        "Generated by `scripts/h_validation_report.py`. Paired sets: puzzles every arm decoded to "
        "the end and scored. $f_x = (m_x - m_a)/(m_c - m_a)$, 90% paired bootstrap interval "
        f"({BOOTSTRAP:,} resamples, seed {SEED}).",
        "",
        "## Load check",
        "",
        "| Run | Verdict | Current H: max abs, mean diff | Zero: max abs, mean diff "
        "| bf16 minus fp32 (SE, z) | Flagged | H gain fp32 minus zero (SE) "
        "| DEV4 digests equal |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, run in runs.items():
        check = run["load_check"]
        if not check or "current" not in check:
            verdict = (check or {}).get("verdict", "none")
            lines.append(f"| {label} | {verdict} | | | | | | |")
            continue
        cur, zero, b16, gain = (
            check["current"], check["zero"], check["bf16_minus_fp32"], check["fp32_minus_zero"]
        )
        dev4 = check.get("dev4") or {}
        lines.append(
            f"| {label} | {check['verdict']} | {cur['max_abs']:.2e}, {cur['mean_diff']:.2e} | "
            f"{zero['max_abs']:.2e}, {zero['mean_diff']:.2e} | {b16['mean']:.2e} "
            f"({b16['se']:.2e}, {_num(b16.get('z'), 2)}) | {b16['flagged']} | "
            f"{gain['mean']:.2e} ({gain['se']:.2e}) | {dev4.get('equal', 'n/a')} |"
        )
    lines.append("")
    for table in tables:
        label = table["label"]
        lines += [f"## {label}: {table['puzzles']} paired puzzles", ""]
        lines += [
            "| Arm | Role | Task accuracy (tasks) | Attempt-1 exact | Answer CE | s per puzzle "
            "| train | encode | h_forward | decode | rescoring |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, arm in table["arms"].items():
            m = arm["metrics"] or {}
            s = arm["seconds"]
            lines.append(
                f"| `{name}` | ({arm['role']}) {arm['source']} | "
                f"{_num(m.get('accuracy'), 2)} of {table['puzzles']} | "
                f"{_pct(m.get('attempt_1'))} | {_num(m.get('ce'), 4)} | {s['span']:.1f} | "
                f"{s['train']:.1f} | {s['encode']:.1f} | "
                f"{s['h_forward']:.2f} | {s['decode']:.1f} | {s['rescoring']:.1f} |"
            )
        if label == "DEV40":
            for name, value in DEV40_REFERENCES:
                lines.append(f"| reference: {name} | (c) | {value} of 40 | | | | | | | | |")
        lines += ["", "| Fraction | Metric | f | 90% interval | Note |", "|---|---|---:|---|---|"]
        for role, entries in table["fractions"].items():
            for metric in METRICS:
                entry = entries[metric]
                note = "gap under the noise floor (2)" if entry["floor"] else ""
                if entry["undefined"]:
                    note += f"{'; ' if note else ''}{entry['undefined']} resamples undefined"
                lines.append(
                    f"| $f_{role}$ | {METRIC_TITLES[metric]} | {_pct(entry['f'])} | "
                    f"{_pct(entry['low'])} to {_pct(entry['high'])} | {note} |"
                )
        if table.get("start") is not None:
            lines += render_start(table["start"], label)
        for name in ("merge", "merge_fit"):
            if table.get(name) is not None:
                lines += render_start(table[name], label)
        if not has_gap_arms(table):
            continue
        line = g4_line(table, training_ended)
        lines += [
            "",
            f"**G4 line ({label}):** $f_b$ on {METRIC_TITLES[line['metric']]} = {_pct(line['f'])} "
            f"({_pct(line['low'])} to {_pct(line['high'])}): {line['branch']}"
            + ("; the interval crosses a branch boundary." if line["straddles"] else "."),
            "",
        ]
    main = {
        t["label"]: g4_line(t, training_ended)
        for t in tables if t["label"] in MAIN_SETS and has_gap_arms(t)
    }  # fmt: skip
    if main:
        ended = "ended" if training_ended else "not ended (the consequences of an early read)"
        lines += [f"**G4 (H's meta-training {ended}):** " + g4_decision(main, training_ended), ""]
    return "\n".join(lines)


def has_gap_arms(table: Mapping[str, Any]) -> bool:
    """Return whether a table holds G4's arms: (a), (b), and (c), so $f_b$ exists."""
    roles = {arm["role"] for arm in table["arms"].values()}
    return {"a", "b", "c"} <= roles


def render_start(start: Mapping[str, Any], label: str) -> list[str]:
    """Render run P3.7-03's reading of (f) against (c): the row, the loss curves, the reach.

    The same for run P3.9-02's (g) against (c) and against (f), by ``start["roles"]``.

    Args:
        start: ``start_comparison``'s result.
        label: The set's name.

    Returns:
        Markdown lines.
    """
    ref, arm = start.get("roles", ("c", "f"))
    lines = [
        "",
        f"**({arm}) against ({ref}) ({label}):** $\\Delta = m_{arm} - m_{ref}$ = "
        f"{start['delta']:+.2f} tasks "
        f"(90% interval {start['low']:+.2f} to {start['high']:+.2f}): {start['row']}"
        + ("; the interval crosses a floor of 2 tasks." if start["straddles"] else "."),
        "",
        f"| Training loss (trailing mean of {LOSS_WINDOW} steps) | "
        + " | ".join(f"step {step}" for step in LOSS_STEPS)
        + " |",
        "|---|" + "---:|" * len(LOSS_STEPS),
    ]
    for role in (ref, arm):
        values = start["loss"][role]
        lines.append(
            f"| ({role}) | " + " | ".join(_num(values[step], 4) for step in LOSS_STEPS) + " |"
        )
    sooner = {True: "yes", False: "no", None: "n/a"}[start["sooner"]]
    lines += [
        "",
        f"({arm}) reaches ({ref})'s final training loss on {start['reached']} of {start['logged']} "
        f"logged puzzles; median step {_num(start['reach_median'], 0)} of {start['steps']} "
        f"(sooner, within {FASTER_SHARE:.0%} of the steps: {sooner}).",
        "",
    ]
    return lines


def report(
    runs: Mapping[str, str | Path],
    out: str | Path | None,
    json_out: str | Path | None,
    training_ended: bool = False,
) -> dict[str, Any]:
    """Read the runs, compute every table, and write the report.

    Args:
        runs: Run log folders by label (``DEV40``, ``GEN``).
        out: The markdown file, or None.
        json_out: The JSON file, or None.
        training_ended: Whether H's meta-training chain has stopped for good (the G4
            consequences of a Bad or No gap row depend on it).

    Returns:
        The tables and G4 lines.
    """
    read = {label: read_run(path) for label, path in runs.items()}
    tables = []
    for label, run in read.items():
        tables.append(analyse(run, label))
        if run["sources"]:
            keys = paired(run)
            for source in sorted(set(run["sources"].values())):
                subset = [k for k in keys if run["sources"].get(k) == source]
                tables.append(analyse(run, f"{label} {source}", subset))
    g4 = {t["label"]: g4_line(t, training_ended) for t in tables}
    main = {k: v for k, v in g4.items() if k in MAIN_SETS}
    result = {
        "tables": tables,
        "g4": g4,
        "training_ended": training_ended,
        "decision": g4_decision(main, training_ended),
    }
    text = render(tables, read, training_ended)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(text)
    return result


# --- the CPU smoke path (BRIEF A8) ---------------------------------------------------------------


class StandinFastLanguageModel:
    """unsloth's mode switches, as the CPU tests stub them: training and evaluation mode."""

    @staticmethod
    def for_training(model: Any) -> Any:
        """Return the model in training mode."""
        return model.train()

    @staticmethod
    def for_inference(model: Any) -> Any:
        """Return the model in evaluation mode."""
        return model.eval()


class _Prepared:
    # The trainer's prepared training set: the texts and their tokens, as trl prepares it.
    def __init__(self, rows: Sequence[Mapping[str, Any]], tokenizer: Any, field: str) -> None:
        self.columns = {field: [row[field] for row in rows]}
        self.columns["input_ids"] = [tokenizer.encode(row[field]) for row in rows]

    def __getitem__(self, name: str) -> Any:
        return self.columns[name]


class StandinTrainer:
    """unsloth's trainer on the CPU: AdamW over the adapter, one view per step.

    The first ``max_steps`` views when set; the moved collator and
    ``UnslothFixedTrainer.compute_loss`` run as on the GPU (the subclass overrides nothing else).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        data_collator: Any,
        train_dataset: Any,
        dataset_text_field: str,
        max_seq_length: int,
        args: Mapping[str, Any],
    ) -> None:
        """Keep what ``train`` needs."""
        self.model, self.data_collator, self.args = model, data_collator, args
        self.train_dataset = _Prepared(train_dataset.rows, tokenizer, dataset_text_field)
        self.state = types.SimpleNamespace(global_step=0, log_history=[])
        self.label_smoother = None
        self.accelerator = types.SimpleNamespace(
            unwrap_model=lambda model, keep_fp32_wrapper: model, num_processes=1
        )

    def train(self) -> Any:
        """Run AdamW over the views; return the step count."""
        import torch

        steps = self.args.get("max_steps", -1)
        views = self.train_dataset["input_ids"]
        rows = views if steps < 0 else views[:steps]
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.args["learning_rate"])
        logged = self.args.get("logging_strategy") == "steps"
        every = self.args.get("logging_steps") if logged else 0
        for ids in rows:
            loss = self.compute_loss(self.model, self.data_collator([{"input_ids": ids}]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.state.global_step += 1
            if every and self.state.global_step % every == 0:
                lr = optimizer.param_groups[0]["lr"]
                self.log({"loss": float(loss.detach()), "learning_rate": lr, "grad_norm": None})
        return types.SimpleNamespace(global_step=len(rows))

    def log(self, logs: Mapping[str, Any], *args: Any) -> None:
        """Keep a log entry, as transformers' ``Trainer.log`` appends to the state's history."""
        self.state.log_history.append({**logs, "step": self.state.global_step})


class _Dataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]]) -> _Dataset:
        return cls(rows)


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    # transformers probes importlib.util.find_spec on its optional packages, which needs a spec.
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


@contextlib.contextmanager
def standins() -> Iterator[None]:
    """Replace the GPU stack's unsloth and datasets, and torch.cuda's memory calls, for the CPU.

    Why: ``arc26.ttt`` and ``arc26.decode`` import unsloth, and ``fit`` resets CUDA's peak
    memory, which raises without a GPU; everything else (the moved fit, collator, decode, the
    providers, H, PEFT, the tiny model) runs as it is. Restores everything on exit.
    """
    import torch

    names = ("arc26.ttt", "arc26.decode")
    saved = {name: sys.modules.get(name) for name in (*names, "unsloth", "datasets")}
    patched = ("reset_peak_memory_stats", "max_memory_allocated", "empty_cache")
    cuda = {name: getattr(torch.cuda, name) for name in patched}
    import logging

    level = logging.root.manager.disable
    sys.modules["unsloth"] = _module(
        "unsloth",
        FastLanguageModel=StandinFastLanguageModel,
        UnslothTrainer=StandinTrainer,
        UnslothTrainingArguments=dict,
    )
    sys.modules["datasets"] = _module("datasets", Dataset=_Dataset)
    torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
    torch.cuda.max_memory_allocated = lambda *args, **kwargs: 0
    torch.cuda.empty_cache = lambda: None
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        logging.disable(level)  # importing arc26.ttt disables logging at WARNING
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, function in cuda.items():
            setattr(torch.cuda, name, function)


def cpu_loaded(
    data_dir: str | Path,
    rank: int,
    hypernet: Mapping[str, Any],
    timing_dir: Path,
    base_seed: int = 0,
) -> Any:
    """Return ``arc26.solver.Loaded`` for the tiny random Qwen3, wrapped as ``load`` wraps it.

    Args:
        data_dir: The run's data folder (``arc-agi_evaluation_challenges.json``).
        rank: Worker rank, for the timing file.
        hypernet: H's config: the adapter's rank, alpha, and targets.
        timing_dir: The worker's timing folder (``load`` and ``puzzle`` records).
        base_seed: Seed of the tiny base, the smoke training's (its config's ``seed``).

    Returns:
        The ``Loaded`` tuple: rsLoRA at H's scope, dropout 0, no bias; fp32 on the CPU (the smoke
        checks logic, not precision, as ``train.py``'s smoke does).
    """
    import numpy as np
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    import arc26.hyper.train as tr
    from arc26 import timing
    from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter
    from arc26.solver import Loaded
    from arc26.ttt import QwenDataCollatorForCompletionOnlyLM

    config = LoraConfig(
        r=hypernet["rank"],
        lora_alpha=hypernet["alpha"],
        target_modules=list(hypernet["target_modules"]),
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(tr.tiny_base(base_seed), config).eval()
    tokenizer = tr.standin_tokenizer()
    weights = get_peft_model_state_dict(model, adapter_name="default")
    formatter = QwenFormatter(tokenizer=tokenizer)
    return Loaded(
        train_args={"learning_rate": 5e-5, "num_train_epochs": 1, "seed": 42},
        max_seq_length=8192,
        timing_logger=timing.TimingLogger(timing_dir, f"rank{rank}"),
        model=model,
        tokenizer=tokenizer,
        default_weights={k: v.clone().detach() for k, v in weights.items()},
        collator=QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False),
        formatter=formatter,
        max_new_tokens=formatter.max_new_tokens(),
        max_score=-np.log(0.2),
        arc_test_set=ArcDataset.from_file(
            str(Path(data_dir) / "arc-agi_evaluation_challenges.json")
        ),
        dir_outputs=str(timing_dir),
    )


def _trainer() -> types.ModuleType:
    # scripts/train_hypernet.py, for load_config (the smoke sections' overrides).
    spec = importlib.util.spec_from_file_location(
        "_p37_train", REPO / "scripts" / "train_hypernet.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def _smoke_config(trainer: types.ModuleType, name: str, folder: Path) -> Any:
    # A training config's smoke section, with its logs and checkpoints under folder.
    return trainer.load_config(
        REPO / "configs" / name,
        True,
        {"out_dir": str(folder / "run_logs"), "ckpt_dir": str(folder / "checkpoints")},
    )


def layout_paths(config: str | Path) -> tuple[str, str, str]:
    """Return the input and the relative paths of ``best.pt`` and the logs a build config reads.

    Args:
        config: A build config with ``adapters`` (``configs/P3.9-02.yaml``).

    Returns:
        ``<owner>/<slug>``, the checkpoint's path inside it, and the run logs' path inside it
        (``""`` when the logs are the input's root, as in ``P3.9-01``'s flat version 2, task
        P3.9r).

    Raises:
        ValueError: If the checkpoint is not ``{input:<owner>/<slug>}/<path>``, the logs neither
            that nor ``{input:<owner>/<slug>}``, or the two are not in one input that the config
            attaches.
    """
    import yaml

    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    adapters = raw["adapters"]
    attached = {**(raw.get("notebook_inputs") or {}), **(raw.get("dataset_inputs") or {})}
    found = []
    checkpoint, logs = adapters["checkpoint"], adapters["load_check"]["run_logs"]
    for value, root_ok in ((checkpoint, False), (logs, True)):
        body = value.removeprefix("{input:")
        name, sep, rest = body.partition("}/")
        if root_ok and body.endswith("}") and "}" not in body[:-1]:
            name, sep, rest = body[:-1], "}", ""
        if not value.startswith("{input:") or not sep or not (rest or root_ok):
            raise ValueError(f"{config}: {value!r} is not {{input:<owner>/<slug>}}/<path>")
        found.append((name, rest))
    names = {name for name, _ in found}
    if len(names) != 1 or not names <= set(attached):
        raise ValueError(f"{config}: best.pt and its logs must sit in one attached input: {names}")
    return found[0][0], found[0][1], found[1][1]


def fomaml_h(out: Path, config: str | Path) -> tuple[Any, Path, Path]:
    """Train the smoke's FOMAML H as ``P3.9-01`` trained H, laid out where ``config`` reads it.

    Why: ``P3.9-01``'s ``best.pt`` (``objective: fomaml``, a warm start, the probe) is read by the
    arms' loader for the first time in ``P3.9-02``; the smoke reads the same kind of file at the
    same relative paths, so a config or format mismatch fails on the CPU, not on Kaggle.

    Args:
        out: The smoke's folder.
        config: The build config whose ``adapters`` paths give the layout.

    Returns:
        The P3.9 smoke's training config, the laid-out ``best.pt``, and the laid-out run logs.

    Raises:
        RuntimeError: If a training phase exits other than 0.
    """
    import shutil

    from arc26.hyper.train import BEST_FILE, train

    trainer = _trainer()
    mt = _smoke_config(trainer, "train_hypernet_MT.yaml", out / "mt")
    if train(mt) != 0:
        raise RuntimeError("the MT smoke exited other than 0")
    cfg = _smoke_config(trainer, "train_hypernet_P3.9-01.yaml", out / "p3.9")
    if train(cfg, init_from=str(out / "mt" / "checkpoints" / "ckpt-000008.pt")) != 0:
        raise RuntimeError("the P3.9 smoke exited other than 0")
    name, checkpoint, logs = layout_paths(config)
    root = out / "inputs" / name
    (root / checkpoint).parent.mkdir(parents=True)
    shutil.copy2(out / "p3.9" / "checkpoints" / BEST_FILE, root / checkpoint)
    # dirs_exist_ok: the logs may share the input's root with best.pt (task P3.9r).
    shutil.copytree(out / "p3.9" / "run_logs", root / logs, dirs_exist_ok=True)
    return cfg, root / checkpoint, root / logs


def smoke(out: Path, layout: str | Path | None = None) -> dict[str, Any]:
    """Run the P3.7 path end to end on the CPU and report on it (BRIEF A8).

    Steps: ``train.py``'s smoke mode writes a tiny H (``best.pt``, ``status.json``,
    ``metrics.jsonl``) on the episode fixture (with ``layout``, a FOMAML H, ``fomaml_h``);
    ``arc26.arms.run`` then builds GEN from the fixture's validation split (2 BARC, 1 ReARC), runs
    the load check on the CPU against the smoke run's logged CE (and the DEV4 fingerprints on
    stand-in tasks), and runs every arm, (a) to (g), on one in-process worker with the tiny model;
    the selection and scores follow; last, the report.

    Args:
        out: The smoke's folder (``runs/P3.7-smoke``), replaced.
        layout: None for the toy H; a build config (``configs/P3.9-02.yaml``) for the FOMAML H at
            that config's paths.

    Returns:
        The report's tables and G4 lines.
    """
    import shutil

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from arc26.hyper.train import train

    if layout is None:
        cfg = _smoke_config(_trainer(), "train_hypernet_toy.yaml", out / "h")
        code = train(cfg)
        if code != 0:
            raise RuntimeError(f"the tiny H's training exited {code}")
        checkpoint, h_logs = out / "h" / "checkpoints" / "best.pt", out / "h" / "run_logs"
    else:
        cfg, checkpoint, h_logs = fomaml_h(out, layout)
    hypernet = cfg.to_dict()["hypernet"]
    # The smoke section's relative paths are resolved from the repository root by load_config.
    fixture = Path(cfg.episodes_root)
    check_root = Path(cfg.check_root)
    # Stand-in DEV4 tasks for the fingerprint stage: the fixture's toy episodes as puzzles.
    from arc26.data import DEV4
    from arc26.hyper.train import load_split, open_episode_set

    episodes = open_episode_set(fixture, cfg.manifest_sha256, check_root, cfg.max_tokens)
    toy = load_split(episodes, "toy", cfg.max_tokens)
    dev4 = {}
    for key, record in zip(DEV4, toy, strict=False):
        dev4[key] = {
            "train": [{"input": x.tolist(), "output": y.tolist()} for x, y in record.demos],
            "test": [{"input": x.tolist()} for x, _ in record.queries],
        }
    (out / "dev4.json").write_text(json.dumps(dev4) + "\n", encoding="utf-8")
    run_logs = out / "run_logs"
    settings = {
        "arms": [
            {"name": "base", "source": "base"},
            {"name": "h", "source": "hyper"},
            {"name": "sgd", "source": "sgd"},
            {"name": "h-polish", "source": "hyper_polish", "steps": 2},
            {"name": "polish", "source": "polish", "steps": 2},
            {"name": "h-fit", "source": "hyper_fit"},
            {"name": "h-merge-fit", "source": "hyper_merge_fit"},
        ],
        "train_log": True,
        "outputs_root": str(out / "arm_outputs"),
        "log_dir": str(run_logs / ARMS_DIR),
        "cache_dir": str(out / "h_cache"),
        "checkpoint": str(checkpoint),
        "cache": None,
        "hypernet": hypernet,
        "views": [[0, 4, 10, 14], [2, 6, 8, 12]],
        "order_seed": SEED,
        "ce": True,
        "finish_s": 600,
        "episodes_root": str(fixture),
        "check_root": str(check_root),
        "manifest_sha256": cfg.manifest_sha256,
        "gen": {"out_dir": str(out / "gen_data"), "barc": 2, "rearc": 1},
        "load_check": {
            "run_logs": str(h_logs),
            "max_abs": 1e-3,
            "max_mean": 1e-4,
            "dev4_challenges": str(out / "dev4.json"),
        },
    }
    solver_cfg = {
        "data_dir": str(out / "unused"),
        "model_dir": None,
        "outputs_dir": str(out / "inference_outputs"),
        "marker_dir": str(out),
        "subset": None,
        "workers": 1,
        "timing_dir": str(run_logs / "timing"),
        "lora_r": hypernet["rank"],
        "lora_targets": list(hypernet["target_modules"]),
        "adapters": settings,
    }
    with standins():
        import queue as queues

        from arc26 import arms
        from arc26.hyper.validation import EXIT_FAIL, EXIT_PASS, load_check

        def check(arm_settings: Any, cfg_: Mapping[str, Any], end_time: float) -> int:
            lc = arm_settings.load_check
            result = load_check(
                {
                    "checkpoint": arm_settings.checkpoint,
                    "episodes_root": arm_settings.episodes_root,
                    "check_root": arm_settings.check_root,
                    "manifest_sha256": arm_settings.manifest_sha256,
                    "run_logs": lc.run_logs,
                    "hypernet": arm_settings.hypernet,
                    "model_dir": None,
                    "cache": None,
                    "device": "cpu",
                    "max_abs": lc.max_abs,
                    "max_mean": lc.max_mean,
                    "dev4_challenges": lc.dev4_challenges,
                    "timing_dir": cfg_["timing_dir"],
                }
            )
            Path(arm_settings.log_dir, arms.LOAD_CHECK_FILE).write_text(
                json.dumps(result, indent=1) + "\n", encoding="utf-8"
            )
            print(f"load check: {result['verdict']} {result['reasons']}")
            return EXIT_PASS if result["verdict"] == "pass" else EXIT_FAIL

        def launch(order: Sequence[str], end_time: float, worker_cfg: Mapping[str, Any]) -> None:
            queue: queues.Queue = queues.Queue()
            for key in order:
                queue.put(key)
            queue.put(None)
            loaded = cpu_loaded(
                worker_cfg["data_dir"], 0, hypernet, Path(worker_cfg["timing_dir"]), cfg.seed
            )
            arms.run_worker(0, queue, end_time, worker_cfg, loaded)

        started = time.time()
        code = arms.run(solver_cfg, started + 3600, launch=launch, check=check)
    if code != 0:
        raise RuntimeError(f"the smoke run exited {code}")
    return report(
        {"GEN": run_logs}, out / "report.md", out / "report.json"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run", action="append", default=[], metavar="LABEL=PATH",
        help="a pulled run_logs folder, labelled DEV40 or GEN; repeat for both",
    )  # fmt: skip
    parser.add_argument("--out", type=Path, help="the markdown report")
    parser.add_argument("--json", type=Path, help="the numbers as JSON")
    parser.add_argument(
        "--smoke", action="store_true", help="run the CPU path first, into runs/P3.7-smoke/"
    )
    parser.add_argument(
        "--layout", type=Path, metavar="CONFIG",
        help="with --smoke: a FOMAML H laid out at this build config's adapter paths, as "
        "P3.9-01 trained H (into runs/P3.9-02-smoke/)",
    )  # fmt: skip
    parser.add_argument(
        "--training-ended", action="store_true",
        help="H's meta-training chain has stopped for good (MT-04 at step 5,600, or a Kill or "
        "failure row with no resume): a Bad row then stops Phase 4; by default it has not",
    )  # fmt: skip
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report (or the smoke path, then the report).

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        0, or 2 on a bad command line.
    """
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.layout is not None and not args.smoke:
        print("--layout needs --smoke", file=sys.stderr)
        return 2
    if args.smoke:
        if args.layout is None:
            smoke(REPO / "runs" / "P3.7-smoke")
        else:
            smoke(REPO / "runs" / "P3.9-02-smoke", args.layout)
        return 0
    runs = {}
    for item in args.run:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            print(f"--run takes LABEL=PATH; got {item!r}", file=sys.stderr)
            return 2
        runs[label] = path
    if not runs:
        print("give at least one --run, or --smoke", file=sys.stderr)
        return 2
    report(runs, args.out, args.json, args.training_ended)
    return 0


if __name__ == "__main__":
    sys.exit(main())
