r"""Meta-train the hypernetwork H on the Q6 task loss (task P3.5; BRIEF B2 Q6, B5 P3.5, A8).

Purpose: H must learn, from many puzzles, to emit the LoRA adapter that test-time training finds
by SGD (Q3, Q6). This script trains H end to end on the task loss through the frozen base and
reports whether it learns. Run ``P3.5-01`` (``configs/train_hypernet_toy.yaml``) is B5's
acceptance: on P3.4's 200-puzzle toy split, the training loss decreases and the validation
episode CE beats the zero adapter within 2k steps. ``configs/train_hypernet.yaml`` holds the
meta-training defaults (accumulation 8), not run in P3.5. Pre-run brief ``docs/prerun/P3.5-01.md``,
runbook ``docs/runbooks/train_hypernet.md``, library ``arc26/hyper/train.py``.

Inputs: the YAML config; P3.4's episode build (``MANIFEST.json`` and JSONL shards) and its passing
leakage check report (``leakage.json``), both found by search under ``episodes_root`` and
``check_root`` and bound to the config's manifest sha256; the base model folder; optionally a
P3.2 embedding cache. Outputs in ``out_dir`` (on Kaggle ``/kaggle/working/run_logs``, pulled):
``config.yaml`` (resolved), ``run.json`` (commit, versions, devices, model, data, and H facts),
``status.json``, ``timing-rank<k>.jsonl`` (ops ``encode``, ``h_forward``, ``train_step``,
``validate``, each ``{puzzle_id, op, seconds, tokens}``), ``metrics.jsonl`` (one line per step
and per validation), ``summary.md``, ``loss_curve.csv``, ``loss_curve.png`` (when matplotlib is
installed, as on Kaggle), and ``error.log`` on a failure. Checkpoints go to ``ckpt_dir`` and stay
on Kaggle.

Method: an episode is $(D, Q, a)$, demonstrations, one query, and an augmentation. H maps
$\bar z(D)$, the mean of P3.2's pair embeddings of the un-augmented demonstrations, to rank-16
factors for q, k, v, o of every layer; P3.1's ``inject`` adds them to the frozen bf16 base, which
reads $a(D)$ then $a(Q)$ in the notebook's format; the loss is the mean cross-entropy of the
query's output-grid tokens, teacher forced, and its gradient flows through the base into H only.
AdamW with a warmup-cosine schedule, clipping, and DistributedDataParallel over ``gpus``
processes (one episode per process per micro step, ``accum`` micro steps per update). Validation
at step 0 and every ``eval_every`` steps compares the zero adapter, the current H, and H fed the
$\bar z$ of another family's episode. The script enforces its own kill criteria (non-finite loss
or gradient, step time, projected wall time: exit 4) and a session budget (checkpoint, exit 3);
``--resume`` continues from a checkpoint, bit-identically on CPU. ``--max-wall-s`` sets one
session's budget without touching the config: a session shorter than the config's (run ``MT-02``
revision 2, capped by the week's quota) and the sessions that resume it share one config file,
and the budget is not a resume setting. ``--init-from`` warm-starts a ``set`` H
(``hypernet.aggregator``, P3.3b) from a full checkpoint of a ``mean`` H: weights by name, the new
aggregator at its zero-output initialization, the checkpoint's step, data cursor, optimizer, and
cosine kept, the new parameters in groups of their own (``new_params_lr``,
``new_params_warmup_steps``). It excludes ``--resume``; a stopped warm-started run resumes with
``--resume`` from its own checkpoints. ``--init-from`` also starts a ``fomaml`` run (P3.9,
``configs/train_hypernet_P3.9-01.yaml``) from a ``zero_step`` checkpoint of the same H: the query
CE after ``inner.steps`` inner steps on the demonstrations trains H first order, and validation
adds decode-aligned metrics and the post-step scores (timing ops ``inner_step``, ``h_backward``,
``validate_inner_step``, ``validate_adapted``). With ``fresh_schedule: true`` the warm start is
a new training phase (fresh AdamW moments, its own warmup and cosine), refused without
``--init-from``; with ``inner_probe`` the start step's validation scores H after its inner steps
at each probed inner rate and the run trains at the rate the pre-registered rule picks
(``summary.md`` shows the probe). Run ``P3.9-01`` goes through ``scripts/modal_ops.py``
(``configs/modal_P3.9-01.yaml``); its CPU smoke needs the MT smoke's checkpoint first::

    python scripts/train_hypernet.py --config configs/train_hypernet_MT.yaml --smoke
    python scripts/train_hypernet.py --config configs/train_hypernet_P3.9-01.yaml --smoke \
        --init-from runs/MT-smoke/checkpoints/ckpt-000008.pt

Exit codes: 0 complete, 1 failed, 2 invalid config or command line, 3 stopped at the session
budget (resume to continue), 4 killed by a kill criterion.

Usage::

    python scripts/train_hypernet.py --config configs/train_hypernet_toy.yaml --commit SHA
    python scripts/train_hypernet.py --config configs/train_hypernet_toy.yaml \
        --resume /kaggle/input/<previous output>/checkpoints --commit SHA
    python scripts/train_hypernet.py --config configs/train_hypernet_MT.yaml --max-wall-s 12600 \
        --commit SHA
    python scripts/train_hypernet.py --config configs/train_hypernet_<set run>.yaml \
        --init-from /kaggle/input/<mean run output>/checkpoints/ckpt-001028.pt --commit SHA
    python scripts/train_hypernet.py --config configs/train_hypernet_toy.yaml --smoke
"""

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml

from arc26.hyper.train import (
    EXIT_CONFIG,
    EXIT_FAILED,
    METRICS_FILE,
    REFERENCES,
    STATUS_FILE,
    TrainConfig,
    atomic_write_bytes,
    check_start,
    read_status,
    train,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nothing is downloaded; Kaggle runs offline

REPO = Path(__file__).resolve().parent.parent
SECTIONS = ("encoder", "hypernet", "kill", "inner", "inner_probe")
# Paths a smoke run reads relative to the repository root, so it runs from any folder.
PATH_KEYS = ("episodes_root", "check_root", "ckpt_dir", "out_dir", "embedding_cache", "model_dir")
# Chart tokens from the dataviz reference palette (validated: blue and orange pass every
# categorical check on the light surface); the zero adapter is a reference, drawn in muted ink.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)  # fmt: skip
SERIES = {"current": "#2a78d6", "mismatched": "#eb6834", "zero": MUTED}
RAW_TRAIN = "#b7d3f6"  # a light step of the blue ramp: per-step values recede behind the mean
LABELS = {
    "zero": "zero adapter (the base)",
    "current": "current H",
    "mismatched": "H with another family's z",
}


class ConfigError(ValueError):
    """The config file or a command-line override is invalid."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Apply the smoke section: top-level keys replace, the nested sections merge by key."""
    merged = dict(base)
    for key, value in override.items():
        if key not in base:
            raise ConfigError(f"smoke.{key} is not a config key")
        if key in SECTIONS:
            if not isinstance(value, dict) or not set(value) <= set(base[key]):
                raise ConfigError(f"smoke.{key} must map keys of {key}")
            merged[key] = {**base[key], **value}
        else:
            merged[key] = value
    return merged


def load_config(path: Path, smoke: bool, overrides: dict[str, Any]) -> TrainConfig:
    """Read the YAML config, apply the smoke section and command-line overrides, and validate.

    The file's own values and its smoke values are both validated in either mode, so a smoke run
    also checks the GPU run's settings. In smoke mode relative paths are taken from the
    repository root.

    Args:
        path: The YAML file.
        smoke: Whether to apply the ``smoke`` section.
        overrides: Config keys set on the command line; None means not given. They win over the
            smoke section too.

    Returns:
        The resolved config.

    Raises:
        ConfigError: If the file is unreadable, a key is missing or unknown, or a value is
            invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"cannot read {path}: {err}") from err
    if not isinstance(raw, dict) or not isinstance(raw.get("smoke"), dict):
        raise ConfigError(f"{path} must hold a mapping with a smoke section")
    section = raw.pop("smoke")
    smoked = _merge(raw, section)
    if smoke:
        for key in PATH_KEYS:
            value = smoked.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                smoked[key] = str(REPO / value)
    try:
        base = TrainConfig.from_dict(raw)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"{path}: {err}") from err
    try:
        smoke_cfg = TrainConfig.from_dict(smoked)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"{path}, smoke section: {err}") from err
    values = (smoke_cfg if smoke else base).to_dict()
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    try:
        return TrainConfig.from_dict(values)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"command line: {err}") from err


def git_commit(repo: Path) -> tuple[str | None, bool | None]:
    """Return the checkout's HEAD and whether its tree has changes; ``(None, None)`` outside git.

    Args:
        repo: A folder inside the checkout.

    Returns:
        ``(sha, dirty)``.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return sha, bool(status.strip())


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Read ``metrics.jsonl``; a line cut short by a killed session is skipped.

    Args:
        path: The file; a missing file is empty.

    Returns:
        The records in file order.
    """
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _moving_mean(values: Sequence[float], width: int) -> list[float]:
    """Trailing mean over the last ``width`` values (fewer at the start)."""
    out, total = [], 0.0
    for i, value in enumerate(values):
        total += value
        if i >= width:
            total -= values[i - width]
        out.append(total / min(i + 1, width))
    return out


def write_curves(records: Sequence[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Write ``loss_curve.csv`` and, when matplotlib is available, ``loss_curve.png``.

    The CSV is the table view of the chart: one row per step, the training columns on training
    steps and the validation columns on validation steps.

    Args:
        records: ``metrics.jsonl`` records.
        out_dir: The run-log folder.

    Returns:
        ``csv``, ``png`` (None when not drawn), ``note`` (why not), and ``smoothing`` (steps).
    """
    train = [r for r in records if r["kind"] == "train"]
    val = {r["step"]: r for r in records if r["kind"] == "val"}
    width = max(1, min(50, len(train) // 20))
    smooth = dict(
        zip(
            (r["step"] for r in train),
            _moving_mean([r["loss"] for r in train], width),
            strict=True,
        )
    )
    by_step = {r["step"]: r for r in train}
    columns = ["step", "lr", "train_ce", f"train_ce_mean{width}", "grad_norm", "step_s"]
    columns += [f"val_{ref}" for ref in REFERENCES]
    csv_path = out_dir / "loss_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for step in sorted(set(by_step) | set(val)):
            row: list[Any] = [step]
            t = by_step.get(step)
            row += (
                [t["lr"], t["loss"], smooth[step], t["grad_norm"], t["step_s"]] if t else [""] * 5
            )
            v = val.get(step)
            row += [v["ce"][ref] for ref in REFERENCES] if v else [""] * len(REFERENCES)
            writer.writerow(row)
    result: dict[str, Any] = {"csv": str(csv_path), "png": None, "note": "", "smoothing": width}
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        result["note"] = "matplotlib is not installed here; loss_curve.csv holds the data"
        return result
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor=SURFACE)
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelcolor=INK_2)
    legend = {
        "frameon": False,
        "labelcolor": INK_2,
        "loc": "upper left",
        "bbox_to_anchor": (1.01, 1),
    }
    if train:
        steps = [r["step"] for r in train]
        raw = {"color": RAW_TRAIN, "linewidth": 1} if width > 1 else {"color": SERIES["current"]}
        top.plot(steps, [r["loss"] for r in train], label="per step", **raw)
        if width > 1:
            top.plot(
                steps,
                [smooth[s] for s in steps],
                color=SERIES["current"],
                linewidth=2,
                solid_capstyle="round",
                label=f"mean of the last {width} steps",
            )
        top.legend(**legend)
    else:
        top.text(
            0.5, 0.5, "no training step yet", transform=top.transAxes, color=MUTED, ha="center"
        )
    top.set_title("Training: query CE per step", color=INK, loc="left")
    top.set_ylabel("cross-entropy (nats per token)", color=INK_2)
    if val:
        steps = sorted(val)
        for ref in REFERENCES:
            ys = [val[s]["ce"][ref] for s in steps]
            bottom.plot(
                steps,
                ys,
                color=SERIES[ref],
                linewidth=2,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                solid_capstyle="round",
                label=LABELS[ref],
            )
        # No end labels: current and mismatched converge when H ignores z, and labels nudged
        # apart would leave their lines; summary.md and the CSV carry the numbers.
        bottom.legend(**legend)
        n = val[steps[-1]]["n"]
        bottom.set_title(f"Validation: mean query CE over {n} episodes", color=INK, loc="left")
    else:
        bottom.text(0.5, 0.5, "no validation yet", transform=bottom.transAxes, color=MUTED)
        bottom.set_title("Validation", color=INK, loc="left")
    bottom.set_xlabel("optimizer step", color=INK_2)
    bottom.set_ylabel("cross-entropy (nats per token)", color=INK_2)
    fig.tight_layout()
    png = out_dir / "loss_curve.png"
    fig.savefig(png, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    result["png"] = str(png)
    return result


def failure_reason(err: BaseException) -> str:
    """Return one line naming a failure: the exception's type and its last non-empty line.

    Why: with ``gpus`` above 1 a worker's exception reaches the script as
    ``torch.multiprocessing.ProcessRaisedException``, whose message starts with a blank line and
    ends with the worker's own exception (``arc26.hyper.train.EpisodeSetError: ...``), so the
    first line says nothing and the last says what failed. A one-line message is its own last
    line.

    Args:
        err: The exception the run ended with.

    Returns:
        ``<type>: <last non-empty line of its message>``, or the type alone without a message.
    """
    lines = [line.strip() for line in str(err).splitlines() if line.strip()]
    return f"{type(err).__name__}: {lines[-1]}" if lines else type(err).__name__


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def training_windows(train: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Mean training CE over the first and the last 5% of the steps taken.

    The fallback comparison for runs with fewer than two complete passes (``training_passes``).

    Args:
        train: The ``train`` records, in step order.

    Returns:
        ``width``, ``first``, ``last`` (means), ``first_steps``, ``last_steps``, and ``change``
        (last minus first); None without records.
    """
    if not train:
        return None
    width = max(1, len(train) // 20)
    first, last = train[:width], train[-width:]
    mean_first = sum(r["loss"] for r in first) / width
    mean_last = sum(r["loss"] for r in last) / width
    return {
        "width": width,
        "first": mean_first,
        "last": mean_last,
        "first_steps": (first[0]["step"], first[-1]["step"]),
        "last_steps": (last[0]["step"], last[-1]["step"]),
        "change": mean_last - mean_first,
    }


def training_passes(train: Sequence[dict[str, Any]], pool: int) -> dict[str, Any] | None:
    r"""Training CE per pass over the pool, and the paired change from the first to the last pass.

    Draw $j$ of step $s$ (slot order) is slot $g = (s - 1) P + j$ with $P$ draws per step; pass
    $\lfloor g / N \rfloor$ shows each of the $N$ pool episodes exactly once. The first and the
    last complete pass are compared episode by episode: $d_i = \mathrm{CE}_\text{last}(i) -
    \mathrm{CE}_\text{first}(i)$, mean and standard error over the $N$ episodes (the decision
    rule of P3.5-01, criterion A).

    Args:
        train: The ``train`` records, in step order, with ``episodes`` and ``episode_ce``.
        pool: $N$, the training split's size.

    Returns:
        ``passes`` (index, first and last step, episodes, mean CE) and, with two complete passes,
        ``paired`` (``first``, ``last`` pass indices, ``mean``, ``se``, ``relative``); None when
        the records lack per-episode CEs or no pass is complete.
    """
    if not train or pool < 1 or any("episode_ce" not in r for r in train):
        return None
    passes: dict[int, dict[str, Any]] = {}
    for record in train:
        per_step = len(record["episodes"])
        for j, (episode, ce) in enumerate(
            zip(record["episodes"], record["episode_ce"], strict=True)
        ):
            index = ((record["step"] - 1) * per_step + j) // pool
            entry = passes.setdefault(index, {"ce": {}, "steps": [record["step"], record["step"]]})
            entry["ce"][episode] = ce
            entry["steps"][1] = record["step"]
    complete = sorted(i for i, entry in passes.items() if len(entry["ce"]) == pool)
    if not complete:
        return None
    rows = [
        {
            "pass": i + 1,
            "steps": tuple(passes[i]["steps"]),
            "episodes": len(passes[i]["ce"]),
            "mean": sum(passes[i]["ce"].values()) / len(passes[i]["ce"]),
        }
        for i in complete
    ]
    result: dict[str, Any] = {"passes": rows}
    if len(complete) >= 2:
        first, last = passes[complete[0]]["ce"], passes[complete[-1]]["ce"]
        diffs = np.array([last[e] - first[e] for e in first])
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        mean_first = sum(first.values()) / len(first)
        result["paired"] = {
            "first": complete[0] + 1,
            "last": complete[-1] + 1,
            "mean": float(diffs.mean()),
            "se": se,
            "relative": float(diffs.mean()) / mean_first if mean_first else float("nan"),
        }
    return result


def render_summary(
    cfg: TrainConfig,
    status: dict[str, Any],
    records: Sequence[dict[str, Any]],
    info: dict[str, Any],
    curves: dict[str, Any],
) -> str:
    """Write the short human-readable report of a run.

    Args:
        cfg: The run config.
        status: ``status.json`` (or the failure the script recorded).
        records: ``metrics.jsonl``.
        info: ``run.json``.
        curves: ``write_curves``'s result.

    Returns:
        Markdown text.
    """
    train = [r for r in records if r["kind"] == "train"]
    val = [r for r in records if r["kind"] == "val"]
    episodes = info.get("episodes", {})
    embeddings = info.get("embeddings", {})
    hypernet = info.get("hypernet", {})
    lines = [
        f"# {cfg.run_id}: meta-training of the hypernetwork H",
        "",
        f"- Status: **{status.get('status')}** (exit {status.get('exit_code')})"
        + (f": {status['reason']}" if status.get("reason") not in (None, "", "complete") else "")
        + f"; step {_fmt(status.get('step'))} of {cfg.steps}; wall "
        f"{_fmt(status.get('elapsed_s'), 0)} s over all sessions.",
        f"- Commit: {info.get('commit')} (dirty: {info.get('dirty')}); smoke: "
        f"{info.get('smoke')}; resumed from: {info.get('resumed_from')}.",
        f"- Data: manifest `{cfg.manifest_sha256}`; check report "
        f"`{episodes.get('check_report')}`; train split `{cfg.train_split}` "
        f"({episodes.get('train')} episodes), validation `{cfg.val_split}` "
        f"({episodes.get('val')} episodes).",
        f"- Base: {cfg.model_id}, {cfg.dtype}, attention {info.get('attn_implementation')}, "
        f"{info.get('device_name')}; {cfg.gpus} processes x accumulation {cfg.accum} = "
        f"{cfg.gpus * cfg.accum} episodes per step.",
        f"- H: {hypernet.get('parameters')} parameters, {hypernet.get('emitted_per_adapter')} "
        f"values per adapter (rank {cfg.hypernet.rank}, {', '.join(cfg.hypernet.target_modules)});"
        f" lr {cfg.lr:g} to {cfg.min_lr:g}, warmup {cfg.warmup_steps}, clip {cfg.clip:g}.",
        f"- Embeddings: {embeddings.get('pairs')} distinct demonstration pairs, "
        f"{embeddings.get('cache_hits')} from the cache, {embeddings.get('encoded')} encoded in "
        f"{_fmt(embeddings.get('seconds'), 1)} s ({embeddings.get('views')} views, layer "
        f"{cfg.encoder.layer}).",
        f"- Checkpoint: `{status.get('checkpoint')}`.",
        "",
        "## Training query CE",
        "",
    ]
    passes = training_passes(train, int(episodes.get("train") or 0))
    if passes:
        lines += ["| Pass | Steps | Episodes | Mean CE |", "|---|---|---|---|"]
        for row in passes["passes"]:
            lines.append(
                f"| {row['pass']} | {row['steps'][0]} to {row['steps'][1]} | {row['episodes']} | "
                f"{_fmt(row['mean'])} |"
            )
        paired = passes.get("paired")
        if paired:
            lines += [
                "",
                f"Pass {paired['last']} minus pass {paired['first']}, paired over the same "
                f"episodes: {paired['mean']:+.4f} (SE {paired['se']:.4f}; "
                f"{100 * paired['relative']:+.1f}%).",
            ]
        lines.append("")
    windows = None if passes and "paired" in passes else training_windows(train)
    if windows:
        change = windows["change"]
        relative = change / windows["first"] if windows["first"] else float("nan")
        lines += [
            "| Window | Steps | Mean CE |",
            "|---|---|---|",
            f"| first 5% | {windows['first_steps'][0]} to {windows['first_steps'][1]} | "
            f"{_fmt(windows['first'])} |",
            f"| last 5% | {windows['last_steps'][0]} to {windows['last_steps'][1]} | "
            f"{_fmt(windows['last'])} |",
            "",
            f"Change, last minus first: {change:+.4f} ({100 * relative:+.1f}%; fewer than two "
            "complete passes, so not paired).",
        ]
    if train:
        lines.append(
            f"Last step: grad norm {_fmt(train[-1]['grad_norm'], 3)}, lr {train[-1]['lr']:.3g}, "
            f"{_fmt(train[-1]['step_s'], 2)} s. Mean step time "
            f"{_fmt(sum(r['step_s'] for r in train) / len(train), 2)} s."
        )
    else:
        lines.append("No training step was taken.")
    lines += ["", "## Validation query CE", ""]
    if val:
        lines += [
            "| Step | Zero adapter | Current H | Mismatched z | Current minus zero (SE) | "
            "Mismatched minus current (SE) |",
            "|---|---|---|---|---|---|",
        ]
        for r in val:
            cz, mc = r["current_minus_zero"], r["mismatched_minus_current"]
            lines.append(
                f"| {r['step']} | {_fmt(r['ce']['zero'])} | {_fmt(r['ce']['current'])} | "
                f"{_fmt(r['ce']['mismatched'])} | {cz['mean']:+.4f} ({cz['se']:.4f}) | "
                f"{mc['mean']:+.4f} ({mc['se']:.4f}) |"
            )
        final = val[-1]
        lines += ["", f"Per source at step {final['step']}:", ""]
        lines += ["| Source | Episodes | Zero | Current | Mismatched |", "|---|---|---|---|---|"]
        for source, row in final["by_source"].items():
            lines.append(
                f"| {source} | {row['n']} | {_fmt(row['zero'])} | {_fmt(row['current'])} | "
                f"{_fmt(row['mismatched'])} |"
            )
        lines += decode_lines(val)
        lines += probe_lines(val)
        first = val[0]
        if first["step"] == 0:
            gap = max(
                abs(c - z)
                for c, z in zip(
                    first["per_episode"]["current"], first["per_episode"]["zero"], strict=True
                )
            )
            lines += [
                "",
                f"Step-0 check: max over episodes of |current - zero| = {gap:.3g} "
                "(H's zero B head reproduces the base, so 0 is expected).",
            ]
    else:
        lines.append("No validation was run.")
    if info.get("smoke"):
        reading = (
            "Smoke run: a tiny random base and fixture episodes on the CPU check the code path; "
            "the numbers say nothing about the real base."
        )
    else:
        reading = f"Decision rule: docs/prerun/{cfg.run_id}.md, section 1."
    curve = curves.get("png") or curves.get("note")
    lines += [
        "",
        "## Reading",
        "",
        reading,
        f"Curves: {curve}; data: loss_curve.csv (train mean over {curves.get('smoothing')} steps).",
        "Logs: metrics.jsonl, timing-rank<k>.jsonl, run.json, status.json.",
        "",
    ]
    return "\n".join(lines)


def decode_lines(val: Sequence[dict[str, Any]]) -> list[str]:
    """The summary's decode-aligned table (P3.9), from the records that carry it.

    Args:
        val: The validation records.

    Returns:
        Markdown lines: the current H at 0 steps and, under ``fomaml``, after its inner steps;
        none for records written before P3.9.
    """
    rows = [r for r in val if "decode" in r]
    if not rows:
        return []
    adapted = any("adapted" in r for r in rows)
    head = "| Step | Zero exact | Current exact | Current under ln 5 | Current geomean CE |"
    rule = "|---|---|---|---|---|"
    if adapted:
        head += " After k: CE | exact | under ln 5 | mean S | displacement (mean) |"
        rule += "---|---|---|---|---|"
    lines = ["", "Decode-aligned (shares of episodes; S the summed target NLL):", "", head, rule]
    for r in rows:
        zero, cur = r["decode"]["zero"], r["decode"]["current"]
        line = (
            f"| {r['step']} | {_fmt(zero['exact_match'], 3)} | {_fmt(cur['exact_match'], 3)} | "
            f"{_fmt(cur['under_bound'], 3)} | {_fmt(cur['geomean_ce'], 6)} |"
        )
        if adapted:
            a = r.get("adapted")
            if a is None:
                line += " | | | | |"
            else:
                moved = a["displacement"]
                line += (
                    f" {_fmt(a['ce'])} | {_fmt(a['decode']['exact_match'], 3)} | "
                    f"{_fmt(a['decode']['under_bound'], 3)} | {_fmt(a['decode']['nll_sum'], 4)} | "
                    f"{_fmt(sum(moved.values()) / len(moved), 4)} |"
                )
        lines.append(line)
    return lines


def probe_lines(val: Sequence[dict[str, Any]]) -> list[str]:
    r"""The summary's inner-rate probe table (P3.9), from the start validation that ran it.

    Args:
        val: The validation records.

    Returns:
        Markdown lines: per probed rate the post-$k$ exact matches, mean $S$, share under
        $\ln 5$, CE, and mean displacement, and the rate the rule picked; none without a probe.
    """
    probed = [r for r in val if "probe" in r]
    if not probed:
        return []
    record = probed[0]
    probe = record["probe"]
    lines = [
        "",
        f"Inner-rate probe at step {record['step']} (rule: the most post-k exact matches within "
        f"{probe['margin']} episodes, then the lowest mean S; {_fmt(probe['seconds'], 0)} s "
        "beyond the picked rate's validation):",
        "",
        "| Inner lr | Exact (episodes) | Mean S | Under ln 5 | CE | Displacement (mean) | Picked |",
        "|---|---|---|---|---|---|---|",
    ]
    for score, block in zip(probe["scores"], probe["results"], strict=True):
        moved = block["displacement"]
        lines.append(
            f"| {score['lr']:g} | {score['exact']} | {_fmt(score['nll_sum'], 4)} | "
            f"{_fmt(block['decode']['under_bound'], 3)} | {_fmt(score['ce'])} | "
            f"{_fmt(sum(moved.values()) / len(moved), 4)} | "
            f"{'yes' if score['lr'] == probe['chosen'] else ''} |"
        )
    return lines


def wall_seconds(text: str) -> float:
    """Parse ``--max-wall-s``: a finite number of seconds above 0.

    Why: ``float`` alone accepts ``inf`` and ``nan``; an infinite budget would leave the stop to
    the notebook's hard stop, which kills the script without its last checkpoint and logs.

    Args:
        text: The argument as given.

    Returns:
        The seconds.

    Raises:
        argparse.ArgumentTypeError: If the text is not a finite number above 0.
    """
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"seconds must be finite and above 0, not {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; no option other than ``--config`` names a config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, type=Path, help="configs/train_hypernet*.yaml")
    parser.add_argument(
        "--smoke", action="store_true", help="CPU run: tiny random Qwen3, fixture episodes, gloo"
    )
    parser.add_argument("--resume", help="checkpoint file, or folder with latest.json")
    parser.add_argument(
        "--init-from",
        help="full checkpoint of a mean H to warm-start a set H from (excludes --resume)",
    )
    parser.add_argument("--commit", help="code commit to record when not in a git checkout")
    parser.add_argument("--out-dir", help="run-log folder (config: out_dir)")
    parser.add_argument("--ckpt-dir", help="checkpoint folder (config: ckpt_dir)")
    parser.add_argument("--episodes-root", help="folder holding the P3.4 build")
    parser.add_argument("--check-root", help="folder holding its leakage check report")
    parser.add_argument("--model-dir", help="Hugging Face model folder (config: model_dir)")
    parser.add_argument("--embedding-cache", help="P3.2 embedding cache root")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="config: device")
    parser.add_argument(
        "--max-wall-s",
        type=wall_seconds,
        help="this session's wall budget in seconds (config: max_wall_s; not a resume setting)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        The exit code: 0 complete, 1 failed (``error.log``), 2 invalid config or command line,
        3 stopped at the session budget, 4 killed by a kill criterion.
    """
    started = time.time()
    args = build_parser().parse_args(argv)
    if args.smoke and args.model_dir:
        print("--smoke builds its own tiny model: drop --model-dir", file=sys.stderr)
        return EXIT_CONFIG
    if args.resume and args.init_from:
        print("--resume and --init-from exclude each other", file=sys.stderr)
        return EXIT_CONFIG
    overrides = {
        "out_dir": args.out_dir,
        "ckpt_dir": args.ckpt_dir,
        "episodes_root": args.episodes_root,
        "check_root": args.check_root,
        "model_dir": args.model_dir,
        "embedding_cache": args.embedding_cache,
        "device": args.device,
        "max_wall_s": args.max_wall_s,
    }
    try:
        cfg = load_config(args.config, args.smoke, overrides)
        check_start(cfg, args.resume, args.init_from)
    except (ConfigError, ValueError) as err:
        print(f"config error: {err}", file=sys.stderr)
        return EXIT_CONFIG
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error.log").unlink(missing_ok=True)
    resolved = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    atomic_write_bytes(out_dir / "config.yaml", resolved.encode("utf-8"))
    commit, dirty = (args.commit, None) if args.commit else git_commit(REPO)
    provenance = {
        "run_id": cfg.run_id,
        "smoke": args.smoke,
        "argv": list(sys.argv[1:] if argv is None else argv),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "resume": args.resume,
        **({"init_from": args.init_from} if args.init_from else {}),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
    }
    run_json = out_dir / "run.json"
    atomic_write_bytes(run_json, (json.dumps(provenance, indent=2) + "\n").encode("utf-8"))
    code, failure = EXIT_FAILED, None
    try:
        code = train(cfg, resume=args.resume, started=started, init_from=args.init_from)
    except KeyboardInterrupt:
        failure = "interrupted"
    except Exception as err:
        failure = failure_reason(err)
        (out_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    status = read_status(out_dir) or {}
    if failure is not None:
        status = {"status": "failed", "exit_code": EXIT_FAILED, "reason": failure, **{
            key: status.get(key) for key in ("step", "checkpoint", "elapsed_s")
        }}  # fmt: skip
        atomic_write_bytes(
            out_dir / STATUS_FILE, (json.dumps(status, indent=2) + "\n").encode("utf-8")
        )
    records = read_metrics(out_dir / METRICS_FILE)
    try:
        curves = write_curves(records, out_dir)
    except Exception as err:  # a plotting problem must not hide the run's result
        curves = {"png": None, "note": f"curves failed: {type(err).__name__}: {err}"}
    info = json.loads(run_json.read_text(encoding="utf-8"))
    summary = render_summary(cfg, status, records, info, curves)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
