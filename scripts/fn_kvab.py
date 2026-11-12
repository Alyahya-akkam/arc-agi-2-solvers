r"""KV-cache A/B probe of Flash-Next: a kernel check, then an FP8 boot and 4-bit KV boots.

Purpose: run ``P5.5-01`` (task P5.5, brief ``docs/prerun/P5.5-01.md``) on Modal's 2 x L40S
(sm89, the L4's architecture). Phase K checks the 4-bit KV kernels of ``scripts/fn_kvq/`` on one
GPU (their benchmark and their ``gpu`` tests); a failure ends the run before any model boots.
Phase M boots the full model in sequence: A, the FP8 baseline (P5.2-07's flags); B, the same
with ``ARC26_KV4=1`` and the Hadamard rotation (the hash-guarded edits of ``configs/P5.5-01.yaml``,
applied at the install and inert while the variable is unset); C, chosen by B's reading (the
vision-tower skip when B passes on quality, the V-only fallback ``ARC26_KV4=v`` when it fails on
quality). Each boot records its pool and memory, a teacher-forced pass over one long text, decode
rates at 1 to 3 streams, and DEV tasks at the real sampling; then it is stopped. Runbook
``docs/runbooks/fn_kvab.md``.

It runs where ``fn_probe.py`` runs (the notebook's cell 2 names it by the run config's
``entry``) and reuses ``fn_probe.py`` unchanged for everything before the server (inputs, local
copy, Python 3.12, the venv with the config's hash-guarded edits, the CUDA links) and
``fn_bisect.py`` unchanged for a boot's mechanics (variables, health, logs, the first error).

The teacher-forced pass. For the text's ids $x_0, \dots, x_{L-1}$ the server returns, at every
position $t \ge 1$, $\ell_t = \log p(x_t \mid x_{<t})$ and the top $k$ log-probabilities
$\lambda_{t,1} \ge \dots \ge \lambda_{t,k}$ with their tokens; the top-1 token is $a_t$ and the
truncated entropy is

.. math:: \hat H_t = -\sum_{i=1}^{k} e^{\lambda_{t,i}} \lambda_{t,i}.

Against the baseline A, a boot $V$ reports, over the $n = L - 1$ positions,

.. math::

    \text{agree} = \frac{1}{n} \sum_t [a^V_t = a^A_t], \quad
    \bar\Delta = \frac{1}{n} \sum_t \lvert \ell^V_t - \ell^A_t \rvert, \quad
    \frac{\bar H^V}{\bar H^A},

and the 99th percentile of $\lvert \ell^V_t - \ell^A_t \rvert$ (nearest rank).

Decode. With $n$ concurrent requests of short prompts and 512 new tokens each (greedy, ignoring
EOS), the per-stream rate is the median of the server's ``gen throughput`` lines logged with
$n$ running requests, divided by $n$ (the wall rate per request is kept beside it).

Time. With $t_0$ the notebook's first cell, $E$ the development ``end_s`` and $K$ the kill
reserve, a variant starting at $t$ runs only if $t_0 + E - K - t \ge w$ (its ``min_window_s``),
and its deadline is $d = \min(t + B, t_0 + E - K)$ for its ``budget_s`` $B$; every measurement
starts only with its own floor left before $d$, so the run ends by $t_0 + E$.

Outputs under ``/kaggle/working/run_logs/fn/`` (pulled): ``kvab.json`` (rewritten before each
boot and after it), ``kvq_<command>.log`` (phase K), per boot ``kvab_<n>_<name>_tf.json``,
``kvab_<n>_<name>_dev.jsonl``, ``kvab_<n>_<name>_transcripts.jsonl``, ``kvab_<n>_<name>_tail.log``
and a bounded ``kvab_<n>_<name>_engine.log``, plus ``fn_probe.py``'s outer records and
``summary.md``.

Usage::

    python fn_kvab.py --config fn_config.json --root / --session-start <t0>   # cell 2
    python scripts/fn_kvab.py --config fn_config.json --reread runs/.../run_logs/fn
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    # A sibling script from the same folder (the code dataset, or scripts/ in tests), unchanged.
    spec = importlib.util.spec_from_file_location(f"_{name}_for_kvab", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fp = _load("fn_probe")
fb = _load("fn_bisect")

SCHEMA = 1
KVAB_FILE = "kvab.json"
# What ``run_if`` may name: always, or a reading of an earlier variant.
RUN_IF = ("always", "healthy", "quality_pass", "quality_fail")
ROLES = ("baseline", "candidate")
OUTCOMES = ("measured", "exited_at_boot", "boot_timeout", "edit_failed", "patch_missing",
            "died", "error", "not_run")
VARIANT_KEYS = {"name", "why", "role", "flags", "env", "edits", "require_patches", "run_if",
                "min_window_s", "pool_against", "pool_ratio_min", "logs"}
APPLIED = ("applied", "already applied")


# ---- the arithmetic (pure: tested on the CPU, used again by --reread) ---------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    r"""Return the nearest-rank percentile: the $\lceil q n \rceil$-th smallest of $n$ values.

    Args:
        values: The sample.
        q: The quantile in $(0, 1]$.

    Returns:
        The value, or None for an empty sample.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def median(values: Sequence[float]) -> float | None:
    """Return the median (the mean of the middle two for an even count), or None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def tf_positions(meta: Mapping[str, Any], length: int) -> dict[str, Any]:
    """Return a teacher-forced reply's per-position values for positions 1 to ``length - 1``.

    Why: SGLang returns ``input_token_logprobs`` as ``(logprob, token_id, text)`` per prompt
    position (the first one's logprob is None) and ``input_top_logprobs`` as a list of such
    tuples per position (None at the first); a boot is compared to the baseline position by
    position, so the arrays must be complete and aligned.

    Args:
        meta: The reply's ``meta_info``.
        length: The number of ids sent.

    Returns:
        ``lp`` (the sent token's log-probability), ``top1`` (the argmax token), ``entropy``
        (the truncated entropy over the top k), ``mass`` (the top k's probability), ``ids_back``
        (the token ids the server read, for the alignment check).

    Raises:
        ValueError: If the reply's arrays are missing or not ``length`` long.
    """
    tokens = meta.get("input_token_logprobs")
    tops = meta.get("input_top_logprobs")
    if not isinstance(tokens, list) or not isinstance(tops, list):
        raise ValueError("the reply has no input_token_logprobs or input_top_logprobs")
    if len(tokens) != length or len(tops) != length:
        raise ValueError(f"the reply covers {len(tokens)} and {len(tops)} positions, not {length}")
    out: dict[str, list[Any]] = {"lp": [], "top1": [], "entropy": [], "mass": [], "ids_back": []}
    for t in range(1, length):
        logprob, token_id = tokens[t][0], tokens[t][1]
        top = tops[t] or []
        if logprob is None or not top:
            raise ValueError(f"position {t} has no log-probability or no top list")
        probs = [(math.exp(float(e[0])), float(e[0])) for e in top]
        out["lp"].append(float(logprob))
        out["top1"].append(int(max(top, key=lambda e: float(e[0]))[1]))
        out["entropy"].append(-sum(p * lam for p, lam in probs))
        out["mass"].append(sum(p for p, _ in probs))
        out["ids_back"].append(int(token_id))
    return out


def compare_tf(base: Mapping[str, Sequence[float]], other: Mapping[str, Sequence[float]]
               ) -> dict[str, Any]:
    r"""Return a boot's teacher-forced agreement with the baseline (the module's formulas).

    Args:
        base: The baseline's ``lp``, ``top1``, ``entropy``.
        other: The boot's, aligned with the baseline's.

    Returns:
        ``positions``, ``top1_agreement``, ``mean_abs_lp_diff``, ``p99_abs_lp_diff``,
        ``max_abs_lp_diff``, ``mean_entropy`` and ``base_mean_entropy``, ``entropy_ratio``,
        ``mean_nll`` and ``base_mean_nll`` (the text's mean negative log-likelihood).

    Raises:
        ValueError: If the two do not cover the same number of positions.
    """
    n = len(base["lp"])
    if n == 0 or len(other["lp"]) != n or len(other["top1"]) != n:
        raise ValueError(f"positions differ: {n} and {len(other['lp'])}")
    diffs = [abs(a - b) for a, b in zip(base["lp"], other["lp"], strict=True)]
    agree = sum(a == b for a, b in zip(base["top1"], other["top1"], strict=True)) / n
    h_base = sum(base["entropy"]) / n
    h_other = sum(other["entropy"]) / n
    return {
        "positions": n,
        "top1_agreement": round(agree, 6),
        "mean_abs_lp_diff": round(sum(diffs) / n, 6),
        "p99_abs_lp_diff": round(percentile(diffs, 0.99), 6),
        "max_abs_lp_diff": round(max(diffs), 6),
        "mean_entropy": round(h_other, 6),
        "base_mean_entropy": round(h_base, 6),
        "entropy_ratio": round(h_other / h_base, 6) if h_base > 0 else None,
        "mean_nll": round(-sum(other["lp"]) / n, 6),
        "base_mean_nll": round(-sum(base["lp"]) / n, 6),
    }


def decode_rate_lines(text: str, streams: int) -> dict[str, Any]:
    """Return the per-stream decode rate from the server's lines logged with ``streams`` running.

    Args:
        text: The engine log written while the requests ran.
        streams: The number of concurrent requests.

    Returns:
        ``lines`` (how many matched) and ``per_stream_tok_s`` (the median line over
        ``streams``), None when no line matched.
    """
    rates = [float(m.group(2)) for m in fp.DECODE_LINE.finditer(text)
             if int(m.group(1)) == streams and float(m.group(2)) > 0]
    value = median(rates)
    return {"lines": len(rates),
            "per_stream_tok_s": None if value is None else round(value / streams, 3)}


def dev_lengths(records: Sequence[Mapping[str, Any]], cap: int) -> dict[str, Any]:
    """Return the thinking turn's length per DEV task: the natural end, or the cap if hit.

    Why: a 4-bit cache can lengthen the thinking (SGLang PR #38724: MXFP4 KV on Qwen3.8-27B,
    43% to 61% longer generations), and longer thinking costs tasks. The first turn ends by
    itself (``stop``), at its cap (``length``), or at the variant's deadline (``cancelled``); a
    deadline cut is censored below the cap, so the length reading is judged only without one.

    Args:
        records: ``fn_probe.attempt_task`` records.
        cap: The thinking turn's cap (the task cap less the forcing tokens).

    Returns:
        ``lengths`` (task to tokens), ``ended`` (task to how), ``median``, ``capped``, ``cut``.
    """
    lengths, ended = {}, {}
    for record in records:
        turns = [t for t in record.get("turns") or [] if t.get("kind") == "think"]
        if "error" in record or not turns:
            ended[record["task_id"]] = "error"
            continue
        first = turns[0]
        how = {"stop": "natural", "length": "cap", "cancelled": "cut"}.get(first.get("finish"),
                                                                            "error")
        ended[record["task_id"]] = how
        if how == "cap":
            lengths[record["task_id"]] = cap
        elif how in ("natural", "cut"):
            lengths[record["task_id"]] = int(first.get("completion_tokens") or 0)
    return {"lengths": lengths, "ended": ended, "median": median(list(lengths.values())),
            "capped": sum(v == "cap" for v in ended.values()),
            "cut": sum(v in ("cut", "error") for v in ended.values())}


def speed_ratios(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Return each stream count's per-stream decode rate ratio to the baseline, and the least.

    The log's rate is used where both have one, else both wall rates.
    """
    ratios = {}
    for key, row in (other or {}).items():
        ref = (base or {}).get(key) or {}
        mine, theirs = (row.get("log") or {}).get("per_stream_tok_s"), (
            ref.get("log") or {}).get("per_stream_tok_s")
        if not (mine and theirs):
            mine, theirs = row.get("wall_per_stream_tok_s"), ref.get("wall_per_stream_tok_s")
        if mine and theirs:
            ratios[key] = round(mine / theirs, 4)
    return {"by_streams": ratios, "min": min(ratios.values()) if ratios else None}


def judge(row: Mapping[str, Any], base: Mapping[str, Any] | None,
          rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a variant's reading by the brief's pre-registered rules (``kvab.readings``).

    The baseline reads ``ok`` or ``fault``. A candidate reads ``fault`` when it did not serve
    (boot, edits, a fatal log line, a dead server, a failed teacher-forced request) or the
    baseline did not; else ``good`` when every criterion holds and ``bad`` otherwise, with the
    misses named. ``quality_pass`` covers the quality criteria alone (top-1 agreement, the mean
    absolute log-probability difference, the entropy ratio, and the thinking length when it is
    judged), so that ``run_if`` can choose the next variant.

    Args:
        row: The variant's row.
        base: The baseline's row, or None.
        rows: Every row so far (``pool_against`` names one of them).
        spec: ``kvab.readings``.

    Returns:
        ``reading``, ``quality_pass``, ``misses``, ``values``.
    """
    measured = row.get("outcome") == "measured"
    fatal = bool(row.get("first_error")) or row.get("alive_after") is False
    tf_ok = (row.get("tf") or {}).get("status") == "ok"
    if row.get("role") == "baseline":
        ok = measured and not fatal and tf_ok
        return {"reading": "ok" if ok else "fault", "quality_pass": None,
                "misses": [] if ok else ["the baseline did not serve or its pass failed"],
                "values": {}}
    if base is None or judge(base, None, rows, spec)["reading"] != "ok":
        return {"reading": "fault", "quality_pass": None,
                "misses": ["no baseline to compare with"], "values": {}}
    if not measured or fatal or not tf_ok or "compare" not in row["tf"]:
        why = row.get("outcome") if not measured else (
            "a fatal log line or a dead server" if fatal else "the teacher-forced pass failed")
        return {"reading": "fault", "quality_pass": None, "misses": [str(why)], "values": {}}
    values: dict[str, Any] = {}
    misses: list[str] = []
    quality: list[bool] = []
    tf = row["tf"]["compare"]
    values.update(top1_agreement=tf["top1_agreement"], mean_abs_lp_diff=tf["mean_abs_lp_diff"],
                  p99_abs_lp_diff=tf["p99_abs_lp_diff"], entropy_ratio=tf["entropy_ratio"])
    checks = [
        ("top1_agreement", tf["top1_agreement"] >= float(spec["top1_min"]),
         f"top-1 agreement {tf['top1_agreement']} < {spec['top1_min']}"),
        ("mean_abs_lp_diff", tf["mean_abs_lp_diff"] <= float(spec["mean_abs_lp_max"]),
         f"mean |dlogp| {tf['mean_abs_lp_diff']} > {spec['mean_abs_lp_max']}"),
        ("entropy_ratio", tf["entropy_ratio"] is not None
         and tf["entropy_ratio"] <= float(spec["entropy_ratio_max"]),
         f"entropy ratio {tf['entropy_ratio']} > {spec['entropy_ratio_max']}"),
    ]
    for _, ok, text in checks:
        quality.append(ok)
        if not ok:
            misses.append(text)
    mine, theirs = row.get("dev_lengths") or {}, base.get("dev_lengths") or {}
    judged = (mine.get("median") and theirs.get("median") and not mine.get("cut")
              and not theirs.get("cut")
              and len(mine.get("lengths", {})) >= int(spec["length_min_tasks"]))
    if judged:
        ratio = round(mine["median"] / theirs["median"], 4)
        values["length_ratio"] = ratio
        ok = ratio <= float(spec["length_ratio_max"])
        quality.append(ok)
        if not ok:
            misses.append(f"median thinking length ratio {ratio} > {spec['length_ratio_max']}")
    else:
        values["length_ratio"] = None
        values["length_not_judged"] = "a DEV task cut by the deadline or failed, or too few"
    against = next((r for r in rows if r.get("name") == row.get("pool_against")), base)
    pool, ref = (row.get("pool") or {}).get("pool_tokens"), (against.get("pool") or {}).get(
        "pool_tokens")
    values["pool_tokens"], values["pool_against"] = pool, against.get("name")
    if pool and ref:
        values["pool_ratio"] = round(pool / ref, 4)
        if values["pool_ratio"] < float(row.get("pool_ratio_min", spec["pool_ratio_min"])):
            misses.append(f"pool ratio {values['pool_ratio']} against {against.get('name')} < "
                          f"{row.get('pool_ratio_min', spec['pool_ratio_min'])}")
    else:
        values["pool_ratio"] = None
        misses.append("no pool reading")
    speed = speed_ratios(base.get("decode") or {}, row.get("decode") or {})
    values["decode_ratio_min"] = speed["min"]
    values["decode_ratios"] = speed["by_streams"]
    if speed["min"] is None or speed["min"] < float(spec["decode_ratio_min"]):
        misses.append(f"decode ratio {speed['min']} < {spec['decode_ratio_min']}")
    errors = row.get("request_errors") or []
    if errors:
        misses.append(f"{len(errors)} request errors")
    return {"reading": "good" if not misses else "bad", "quality_pass": all(quality),
            "misses": misses, "values": values}


def should_run(variant: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Return why a variant does not run (its ``run_if`` against earlier readings), or None.

    ``run_if`` is ``always`` or ``<reading>:<variant>`` with ``healthy`` (the variant served and
    its reading is not ``fault``), ``quality_pass`` or ``quality_fail`` (its quality criteria
    held or missed, without a fault).
    """
    rule = str(variant.get("run_if", "always"))
    if rule == "always":
        return None
    kind, _, name = rule.partition(":")
    earlier = next((r for r in rows if r.get("name") == name), None)
    if earlier is None or "judged" not in earlier:
        return f"run_if {rule}: {name} has no reading"
    judged = earlier["judged"]
    if judged["reading"] == "fault":
        return f"run_if {rule}: {name} read fault"
    if kind == "healthy":
        return None
    passed = judged.get("quality_pass")
    if (kind == "quality_pass" and passed) or (kind == "quality_fail" and passed is False):
        return None
    return f"run_if {rule}: {name}'s quality_pass is {passed}"


def check_variants(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate ``kvab.variants`` and return them as plain values.

    Raises:
        ValueError: On an unsafe or repeated name, an unknown field, a role or ``run_if`` out of
            the allowed values, a ``run_if`` naming a later or unknown variant, an edit without
            its hash guard, or not exactly one baseline, first.
    """
    variants = json.loads(json.dumps(list(spec["variants"])))
    seen: list[str] = []
    for variant in variants:
        name = str(variant.get("name", ""))
        if not fb.NAME.fullmatch(name) or name in seen:
            raise ValueError(f"variant name {name!r} is unsafe or repeated")
        if set(variant) - VARIANT_KEYS:
            raise ValueError(f"variant {name}: unknown {sorted(set(variant) - VARIANT_KEYS)}")
        if variant.get("role") not in ROLES:
            raise ValueError(f"variant {name}: role must be one of {ROLES}")
        rule = str(variant.get("run_if", "always"))
        kind, _, other = rule.partition(":")
        if kind not in RUN_IF or (kind != "always" and other not in seen):
            raise ValueError(f"variant {name}: run_if {rule!r} must be always or "
                             f"<{'|'.join(RUN_IF[1:])}>:<an earlier variant>")
        for item in variant.get("edits") or []:
            if not {"name", "file", "sha256_before", "pattern", "replacement"} <= set(item):
                raise ValueError(f"variant {name}: an edit needs its hash guard")
        seen.append(name)
    roles = [v["role"] for v in variants]
    if not variants or roles[0] != "baseline" or roles.count("baseline") != 1:
        raise ValueError("kvab.variants: exactly one baseline, first")
    return variants


def tf_text_ids(tok: Any, source: Mapping[str, Any], template: Mapping[str, Any],
                paths: Mapping[str, Any], max_tokens: int) -> tuple[list[int], dict[str, Any]]:
    """Return the teacher-forced ids and where they came from.

    ``transcript``: one record of a ``direct_transcripts.jsonl`` (P6.9-01's), its system and user
    messages rendered with this model's chat template, then the assistant's reasoning, encoded
    once and cut to ``max_tokens``. ``synthetic`` (the smoke): a fixed text repeated to length.

    Args:
        tok: The tokenizer (``fn_probe.HFTokenizer`` on the GPU, the fake in the smoke).
        source: ``kvab.tf.source``.
        template: ``template.kwargs``.
        paths: ``fn_probe``'s found inputs (``source.input`` names one).
        max_tokens: The cut.

    Returns:
        The ids and a record: the source, the file's sha256, ``tokens``, ``ids_sha256``.

    Raises:
        ValueError: If the file's sha256 is not the config's or the record is missing.
    """
    if source["kind"] == "synthetic":
        text = str(source["text"]) * int(source["repeat"])
        ids = tok.encode(text)[:max_tokens]
        record = {"kind": "synthetic"}
    else:
        folder = paths.get(source["input"])
        if not folder:
            raise ValueError(f"input {source['input']} was not found")
        path = Path(folder) / source["file"]
        digest = fp.sha256_file(path)
        if digest != source["sha256"]:
            raise ValueError(f"{source['file']}: sha256 {digest}, not {source['sha256']}")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        row = next((r for r in records if r.get("attempt_id") == source["attempt_id"]), None)
        if row is None:
            raise ValueError(f"no record {source['attempt_id']} in {source['file']}")
        messages = [m for m in row["messages"] if m["role"] in ("system", "user")]
        reasoning = next(m.get("reasoning") or "" for m in row["messages"]
                         if m["role"] == "assistant")
        text = tok.render(messages, **template) + reasoning
        ids = tok.encode(text)[:max_tokens]
        record = {"kind": "transcript", "file": source["file"], "sha256": digest,
                  "attempt_id": source["attempt_id"], "prompt_messages": len(messages)}
    record.update(tokens=len(ids),
                  ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    return ids, record


# ---- phase K -------------------------------------------------------------------------------------


def fill(value: str, values: Mapping[str, str]) -> str:
    """Replace ``{name}`` placeholders in one argument."""
    for key, text in values.items():
        value = value.replace("{" + key + "}", text)
    return value


def timed(inner: Any, op: str, seconds: float | None, tokens: int | None) -> None:
    """Append the GPU op's timing record (``arc26.timing``), when a logger is configured."""
    from arc26 import timing

    logger = timing.get_logger()
    if logger is not None and seconds is not None:
        logger.record(None, op, float(seconds), int(tokens or 0))


def extract_bundle(run: Any) -> Path:
    """Extract the kernel bundle into the scratch folder once and return the tree.

    Why: phase K runs the bundle's commands from the tree, and the patched engine of B and C
    imports ``fn_kvq`` from its ``scripts/`` folder (their ``PYTHONPATH`` is ``{kvq}``): the
    arc26 wheel does not carry ``fn_kvq`` (the competition path's wheel stays as it is).

    Raises:
        RuntimeError: If the bundle is not among the code files.
    """
    spec = run.config["kvab"]["kernel_check"]
    tree = run.path(run.config["paths"]["scratch"]) / "kvq"
    if (tree / ".extracted").is_file():
        return tree
    code = Path(run.state.get("paths", {}).get("code") or HERE)
    bundle = code / spec["bundle"]
    if not bundle.is_file():
        raise RuntimeError(f"no kernel bundle {bundle.name} in the code folder")
    tree.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle) as archive:
        archive.extractall(tree, filter="data")
    (tree / ".extracted").write_text(bundle.name + "\n", encoding="utf-8")
    return tree


def stage_kernel_check(inner: Any) -> dict[str, Any]:
    """Phase K: extract the kernel bundle and run its commands on one GPU, in order.

    Why: a kernel that fails its own tests or benchmark on sm89 would only waste the boots; the
    commands run before any model, each with its timeout cut to the phase's budget, on the GPU
    ``kernel_check.gpu`` alone (``CUDA_VISIBLE_DEVICES``). The bundle's tree is on
    ``PYTHONPATH`` (its root and ``scripts/``, so the tests import ``fn_kvq`` as in the
    repository), with the probe image's pytest (``modal_job.PROBE_TEST_TARGET``) after it.

    Returns:
        ``commands`` (name, return code, seconds, the log's name) and the GPUs' use after.

    Raises:
        StageSkipped: If the check is off.
        RuntimeError: If the bundle is missing or a command fails or times out.
    """
    run = inner.run
    spec = run.config["kvab"]["kernel_check"]
    if not spec.get("enabled", True):
        raise fp.StageSkipped("kvab.kernel_check.enabled is off")
    tree = extract_bundle(run)
    extra = [str(tree), str(tree / "scripts")]
    if spec.get("pytest_path") and Path(spec["pytest_path"]).is_dir():
        extra.append(str(spec["pytest_path"]))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(extra + [p for p in [env.get("PYTHONPATH")] if p])
    env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    values = {"python": sys.executable, "out": str(run.out), "tree": str(tree)}
    began = fp._now()
    done = []
    for command in spec["commands"]:
        left = float(spec["budget_s"]) - (fp._now() - began)
        if left <= 5:
            raise RuntimeError(f"the phase's {spec['budget_s']} s ran out before {command['name']}")
        argv = [fill(str(a), values) for a in command["argv"]]
        log = run.out / f"kvq_{command['name']}.log"
        start = fp._now()
        try:
            proc = subprocess.run(argv, cwd=tree, env=env, capture_output=True,
                                  timeout=min(float(command["timeout_s"]), left), check=False)
            code_out, output = proc.returncode, proc.stdout + b"\n--- stderr ---\n" + proc.stderr
        except subprocess.TimeoutExpired as err:
            code_out = 124
            output = (err.stdout or b"") + b"\n--- stderr ---\n" + (err.stderr or b"")
        log.write_bytes(fb.bounded(output, int(spec["max_log_bytes"])))
        row = {"name": command["name"], "argv": argv, "returncode": code_out,
               "seconds": round(fp._now() - start, 1), "log": log.name}
        done.append(row)
        print(f"   kernel check {command['name']}: exit {code_out} in {row['seconds']} s",
              flush=True)
        if code_out != 0:
            raise RuntimeError(f"{command['name']} exited {code_out} (see {log.name})")
    return {"commands": done, "gpu_used_mib": fb.gpu_used()}


# ---- phase M: one boot ---------------------------------------------------------------------------


def stop_boot(inner: Any, tag: str, spec: Mapping[str, Any],
              before: list[float] | str | None) -> dict[str, Any]:
    """Stop the boot's server, every process it tagged, and wait for its port and its GPUs.

    Why: the next boot binds the same port and needs the same GPU memory and the same pinned
    host table; a rank that outlived its group would make the next boot fail, or read a smaller
    pool, for a reason that is not its own (``fn_bisect.stop_variant``'s rule, with this
    config's margin).

    Returns:
        ``killed`` (tagged pids still alive after the group's stop), ``port_free``,
        ``gpu_used_mib``, ``seconds``.
    """
    began = fp._now()
    fp.kill_server(inner)
    left = fb.tagged_pids(tag)
    for pid in left:
        try:
            os.kill(pid, 9)
        except OSError:
            continue
    port = int(inner.run.config["engine"]["port"])
    margin = float(spec["gpu_free_mib"])

    def held(used: list[float] | str) -> bool:
        if not isinstance(used, list) or not isinstance(before, list) or len(used) != len(before):
            return False
        return any(u > b + margin for u, b in zip(used, before, strict=True))

    used = fb.gpu_used()
    while fp._now() - began < float(spec["stop_settle_s"]) and (
            fb.tagged_pids(tag) or not fb.port_free(port) or held(used)):
        time.sleep(0.5)
        used = fb.gpu_used()
    return {"killed": left, "port_free": fb.port_free(port), "gpu_used_mib": used,
            "seconds": round(fp._now() - began, 1)}


def generate_once(inner: Any, body: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    """Send one non-streaming ``/generate``; return its status, seconds, and reply."""
    began = fp._now()
    try:
        status, reply = fp.http_json(inner.base, "POST", "/generate", dict(body), timeout=timeout)
    except OSError as err:
        return {"status": None, "error": fp._describe(err), "seconds": round(fp._now() - began, 1)}
    out = {"status": status, "seconds": round(fp._now() - began, 3), "reply": reply}
    if status != 200:
        out["error"] = fp._tail(json.dumps(reply) if not isinstance(reply, str) else reply, 1000)
    return out


def measure_tf(inner: Any, ids: Sequence[int], spec: Mapping[str, Any], timeout: float,
               path: Path) -> dict[str, Any]:
    """Run the teacher-forced pass and write its per-position values to ``path``.

    One greedy request of one new token with ``return_logprob``, ``logprob_start_len`` 0 (so
    no prefix comes from the radix cache: SGLang caps the prefix match at the start) and
    ``top_logprobs_num`` k. The pulled file holds numbers only, never the ids or the text.

    Returns:
        ``status``, ``seconds``, ``positions``, ``ids_match`` (the server read the ids sent),
        and the means (``mean_entropy``, ``mean_nll``, ``mean_top_mass``).
    """
    body = {"input_ids": list(ids), "return_logprob": True, "logprob_start_len": 0,
            "top_logprobs_num": int(spec["top_k"]), "return_text_in_logprobs": False,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
    sent = generate_once(inner, body, timeout)
    reply = sent.pop("reply", None)
    if sent["status"] != 200:
        return {**sent, "http_status": sent["status"], "status": "failed"}
    try:
        values = tf_positions((reply or {}).get("meta_info") or {}, len(ids))
    except ValueError as err:
        return {**sent, "status": "failed", "error": str(err)}
    match = values.pop("ids_back") == list(ids[1:])
    n = len(values["lp"])
    sent.pop("status")
    fp.write_json(path, {"positions": n, "top_k": int(spec["top_k"]),
                         "lp": [round(v, 6) for v in values["lp"]], "top1": values["top1"],
                         "entropy": [round(v, 6) for v in values["entropy"]],
                         "mass": [round(v, 6) for v in values["mass"]]})
    return {**sent, "status": "ok" if match else "failed", "positions": n, "ids_match": match,
            "file": path.name, "mean_entropy": round(sum(values["entropy"]) / n, 6),
            "mean_nll": round(-sum(values["lp"]) / n, 6),
            "mean_top_mass": round(sum(values["mass"]) / n, 6),
            **({} if match else {"error": "the server's prompt ids differ from those sent"})}


def measure_decode(inner: Any, spec: Mapping[str, Any], streams: int, log_path: Path,
                   timeout: float) -> dict[str, Any]:
    """Run ``streams`` concurrent greedy requests of ``max_tokens`` new tokens each.

    The prompts are the config's short chat prompts (one per stream, rendered with the chat
    template), so every boot decodes the same tokens from the same prefixes.

    Returns:
        ``requests`` (status, tokens, seconds each), ``wall_per_stream_tok_s`` (their mean),
        ``log`` (``decode_rate_lines`` over the log written meanwhile), ``errors``.
    """
    config = inner.run.config
    prompts = [inner.tok.encode(inner.tok.render([{"role": "user", "content": p}],
                                                 **config["template"]["kwargs"]))
               for p in spec["prompts"][:streams]]
    body = {"sampling_params": {"max_new_tokens": int(spec["max_tokens"]), "temperature": 0.0,
                                "ignore_eos": True}}
    offset = log_path.stat().st_size if log_path.is_file() else 0
    with ThreadPoolExecutor(max_workers=streams) as pool:
        sent = list(pool.map(lambda ids: generate_once(inner, {**body, "input_ids": ids},
                                                       timeout), prompts))
    rows, rates, errors = [], [], []
    for result in sent:
        reply = result.pop("reply", None)
        made = ((reply or {}).get("meta_info") or {}).get("completion_tokens") if isinstance(
            reply, dict) else None
        rows.append({**result, "completion_tokens": made})
        if result["status"] == 200 and isinstance(made, int) and made > 0 and result["seconds"]:
            rates.append(made / result["seconds"])
        else:
            errors.append(result.get("error") or f"status {result['status']}")
    with open(log_path, "rb") as handle:
        handle.seek(offset)
        text = handle.read().decode("utf-8", errors="replace")
    return {"streams": streams, "requests": rows, "errors": errors,
            "wall_per_stream_tok_s": round(sum(rates) / len(rates), 3) if rates else None,
            "log": decode_rate_lines(text, streams)}


def measure_dev(inner: Any, spec: Mapping[str, Any], deadline: float, index: int,
                name: str) -> dict[str, Any]:
    """Run the DEV tasks as parallel streams with the real sampling, through ``fn_probe``.

    Each task is ``fn_probe.attempt_task`` on the plain arm: the thinking turn to the task cap
    less the forcing tokens, then the forcing turn, so the answers are the solve's; the thinking
    turn's end gives the length reading (``dev_lengths``). Requests still running at the
    variant's deadline are cut (``fn_probe.Clock``).

    Returns:
        ``tasks``, ``file`` (the records), ``lengths`` (``dev_lengths``), ``errors``.
    """
    run = inner.run
    solve = run.config["solve"]
    tasks = list(inner.tasks)
    clock = fp.Clock(force_at=deadline - float(spec["force_reserve_s"]), end=deadline)
    with ThreadPoolExecutor(max_workers=max(len(tasks), 1)) as pool:
        records = list(pool.map(lambda pair: fp.attempt_task(inner, pair[1], clock, pair[0],
                                                             spec["arm"]), enumerate(tasks)))
    stem = f"kvab_{index}_{name}"
    path = run.out / f"{stem}_dev.jsonl"
    for record in records:
        record["variant"] = name
        fp.append_jsonl(path, record)
    transcripts = run.out / "transcripts.jsonl"
    if transcripts.is_file():
        transcripts.replace(run.out / f"{stem}_transcripts.jsonl")
    cap = int(solve["cap"]) - int(solve["forcing_tokens"])
    lengths = dev_lengths(records, cap)
    errors = [f"{r['task_id']}: {r['error']}" for r in records if "error" in r]
    return {"tasks": [t.task_id for t in tasks], "file": path.name, "lengths": lengths,
            "think_cap": cap, "errors": errors}


def boot_variant(inner: Any, variant: Mapping[str, Any], index: int, site: Path | None,
                 deadline: float, ids: Sequence[int], rows: Sequence[Mapping[str, Any]]
                 ) -> dict[str, Any]:
    """Boot one variant, take its measurements in order, stop it, and read its log.

    Steps after the boot, each only with its floor left before ``deadline``: the pool and the
    GPUs' memory; the teacher-forced pass; decode at each stream count; the DEV tasks.

    Args:
        inner: ``fn_probe.Inner`` (the run, the base URL, the tokenizer, the tasks).
        variant: The variant (``check_variants``).
        index: Its position (file names).
        site: The venv's site-packages for its own edits, or None.
        deadline: ``fn_probe._now()`` value by which its measurements end.
        ids: The teacher-forced ids.
        rows: The rows so far (the baseline's teacher-forced values are compared here).

    Returns:
        The row of ``kvab.json``.
    """
    run = inner.run
    config = run.config
    spec = config["kvab"]
    name = variant["name"]
    began = fp._now()
    row: dict[str, Any] = {
        "index": index, "name": name, "role": variant["role"], "why": variant.get("why", ""),
        "flags": variant.get("flags") or {}, "env": variant.get("env") or {},
        "pool_against": variant.get("pool_against"),
        "started_s": round(began - run.start, 1), "deadline_s": round(deadline - run.start, 1),
        "request_errors": [],
    }
    if variant.get("pool_ratio_min") is not None:
        row["pool_ratio_min"] = variant["pool_ratio_min"]
    patches = {p["name"]: p["status"] for p in run.state.get("install", {}).get("patches") or []}
    missing = [p for p in variant.get("require_patches") or [] if patches.get(p) not in APPLIED]
    if missing:
        row.update(outcome="patch_missing", missing=missing, seconds=0.0,
                   patch_status={p: patches.get(p) for p in missing})
        return row
    restore: list[tuple[Path, bytes]] = []
    tag = f"{name}-{uuid.uuid4().hex[:8]}"
    folder = run.path(config["paths"]["scratch"]) / "kvab" / f"{index}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "engine.log"
    values = {"out": str(run.out), "scratch": str(folder), "variant": name,
              "kvq": str(run.path(config["paths"]["scratch"]) / "kvq" / "scripts")}
    boots = spec["boots"]
    measure = spec["measure"]
    try:
        row["edits"], ready = fb.apply_edits(site, variant.get("edits") or [], restore)
        if not ready:
            row["outcome"] = "edit_failed"
            return row
        overrides = {**run.state.get("flag_overrides", {}), **(spec.get("base_flags") or {}),
                     **(variant.get("flags") or {})}
        argv = fp.server_argv(config, sys.executable, fp.server_values(run), overrides)
        env = fb.variant_env(os.environ, {**(spec.get("base_env") or {}),
                                          **(variant.get("env") or {})}, tag, values)
        row["argv"] = argv
        fp.append_jsonl(run.out / "engine_commands.jsonl",
                        {"t": round(fp._now() - run.start, 1), "variant": name, "argv": argv})
        row["gpu_before_mib"] = fb.gpu_used()
        with open(log_path, "wb", buffering=0) as handle:
            inner.proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True)
        row["health"] = {}
        boot = fb.wait_healthy(inner, deadline - float(boots["measure_floor_s"]),
                               float(boots["poll_s"]), row["health"])
        row["boot"] = boot
        row["ready_s"] = round(fp._now() - began, 1) if boot == "healthy" else None
        timed(inner, f"kvab_boot_{name}", fp._now() - began, 0)
        if boot != "healthy":
            row["outcome"] = "exited_at_boot" if boot == "exited" else "boot_timeout"
            return row
        row["pool"] = fb.pool_reading(inner)
        row["gpu_after_boot"] = fp.gpu_memory()
        tf_spec = measure["tf"]
        if deadline - fp._now() >= float(tf_spec["min_s"]):
            row["tf"] = measure_tf(inner, ids, tf_spec, min(float(tf_spec["timeout_s"]),
                                                            deadline - fp._now()),
                                   run.out / f"kvab_{index}_{name}_tf.json")
            base = next((r for r in rows if r.get("role") == "baseline"), None)
            if row["tf"]["status"] == "ok" and base is not None and (
                    (base.get("tf") or {}).get("status") == "ok"):
                mine = json.loads((run.out / row["tf"]["file"]).read_text(encoding="utf-8"))
                theirs = json.loads((run.out / base["tf"]["file"]).read_text(encoding="utf-8"))
                try:
                    row["tf"]["compare"] = compare_tf(theirs, mine)
                except ValueError as err:
                    row["tf"].update(status="failed", error=str(err))
            if row["tf"]["status"] != "ok":
                row["request_errors"].append(f"tf: {row['tf'].get('error')}")
            timed(inner, f"kvab_tf_{name}", row["tf"].get("seconds"), len(ids))
        else:
            row["tf"] = {"status": "skipped", "reason": "under tf.min_s before the deadline"}
        row["gpu_after_tf"] = fp.gpu_memory()
        dec_spec = measure["decode"]
        row["decode"] = {}
        for streams in dec_spec["streams"]:
            if deadline - fp._now() < float(dec_spec["min_s"]) or not fb.server_alive(inner):
                row["decode"][str(streams)] = {"skipped": "under decode.min_s or no server"}
                continue
            result = measure_decode(inner, dec_spec, int(streams), log_path,
                                    min(float(dec_spec["timeout_s"]), deadline - fp._now()))
            row["decode"][str(streams)] = result
            row["request_errors"].extend(f"decode {streams}: {e}" for e in result["errors"])
            for request in result["requests"]:
                timed(inner, f"kvab_decode{streams}_{name}", request.get("seconds"),
                      request.get("completion_tokens"))
        dev_spec = measure["dev"]
        if deadline - fp._now() >= float(dev_spec["min_s"]) and fb.server_alive(inner):
            row["dev"] = measure_dev(inner, dev_spec, deadline, index, name)
            row["dev_lengths"] = row["dev"]["lengths"]
            row["request_errors"].extend(f"dev {e}" for e in row["dev"]["errors"])
        else:
            row["dev"] = {"skipped": "under dev.min_s before the deadline, or no server"}
        row["gpu_after_dev"] = fp.gpu_memory()
        time.sleep(float(boots["settle_s"]))
        row["alive_after"] = fb.server_alive(inner)
        row["outcome"] = "measured" if row["alive_after"] else "died"
        return row
    except OSError as err:
        row.setdefault("outcome", "died")
        row["error"] = fp._describe(err)
        return row
    except Exception as err:  # a measurement's own bug: recorded, the server still stopped
        row["outcome"] = "error"
        row["error"] = fp._describe(err)
        row["traceback"] = fp._tail(traceback.format_exc(), 3000)
        return row
    finally:
        if inner.proc is not None:
            row["exit_code"] = inner.proc.poll()
            row["stop"] = stop_boot(inner, tag, boots, row.get("gpu_before_mib"))
            inner.proc = None
        row["restored"] = fb.restore_edits(restore)
        row["seconds"] = round(fp._now() - began, 1)
        if log_path.is_file():
            data = log_path.read_bytes()
            text = data.decode("utf-8", errors="replace")
            row["first_error"] = fb.first_error(text, spec["error_patterns"],
                                                spec.get("ignore_patterns") or [],
                                                spec["kernel_frame_skip"])
            row["marks"] = fb.log_marks(text, spec.get("log_marks") or {})
            stem = f"kvab_{index}_{name}"
            lines = text.splitlines()
            (run.out / f"{stem}_tail.log").write_text(
                "\n".join(lines[-int(spec["tail_lines"]):]) + "\n", encoding="utf-8")
            (run.out / f"{stem}_engine.log").write_bytes(fb.bounded(data,
                                                                    int(spec["max_log_bytes"])))
        row["logs"] = collect_logs(run, variant, index, folder)


def collect_logs(run: Any, variant: Mapping[str, Any], index: int, folder: Path) -> list[str]:
    """Copy a bounded copy of each file the variant's ``logs`` globs name into the pulled folder."""
    spec = run.config["kvab"]
    values = {"out": str(run.out), "scratch": str(folder), "variant": variant["name"]}
    names = []
    for pattern in variant.get("logs") or []:
        for path in sorted(Path("/").glob(fill(pattern, values).lstrip("/")))[:4]:
            if path.is_file():
                safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in path.stem)
                name = f"kvab_{index}_{variant['name']}_{safe}.log"
                (run.out / name).write_bytes(fb.bounded(path.read_bytes(),
                                                        int(spec["max_log_bytes"])))
                names.append(name)
    return names


# ---- phase M: the sequence -----------------------------------------------------------------------


def write_kvab(run: Any, data: Mapping[str, Any]) -> None:
    """Write ``kvab.json``."""
    fp.write_json(run.out / KVAB_FILE, {"schema": SCHEMA, "run_id": run.config["run_id"], **data})


def unapplied_edits(run: Any, variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return every edit a variant requires that the install did not apply, with its status.

    Why: P5.5-01 revision 1's KV4 edits on ``kv_cache_configurator.py`` were anchored on the
    clone's file, not the wheel's (the Kaggle CPU twin of P5.2-01 revision 18 refused them), so
    its candidates would have booted FP8 against the FP8 baseline. Revision 2 stops before any
    boot when one is missing.

    Args:
        run: The run (``state.install.patches``, ``patches.json``'s results).
        variants: The checked variants (their ``require_patches``).

    Returns:
        Edit name to its status (``None`` when the install recorded none).
    """
    patches = {p["name"]: p["status"] for p in run.state.get("install", {}).get("patches") or []}
    required = [name for v in variants for name in v.get("require_patches") or []]
    return {name: patches.get(name) for name in dict.fromkeys(required)
            if patches.get(name) not in APPLIED}


def stage_ab(inner: Any) -> dict[str, Any]:
    """Phase M: boot every variant whose ``run_if`` holds and whose window fits, in order.

    Returns:
        ``booted``, each variant's outcome and reading, and the decision line.

    Raises:
        StageSkipped: If ``kvab.enabled`` is off or phase K did not pass.
    """
    run = inner.run
    spec = run.config["kvab"]
    if not spec.get("enabled"):
        raise fp.StageSkipped("kvab.enabled is off")
    check = run.state.get("kernel_check", {}).get("status")
    if check not in ("ok", "skipped"):
        raise fp.StageSkipped(f"phase K read {check}: the boots do not start")
    variants = check_variants(spec)
    # P5.5-01 revision 2's stop rule: every KV4 edit a candidate needs must be applied, or no
    # boot runs at all (the baseline alone would compare FP8 with nothing, revision 1's flaw).
    missing = unapplied_edits(run, variants) if spec.get("stop_if_unapplied") else {}
    if missing:
        write_kvab(run, {"rows": [], "running": None, "unapplied": missing,
                         "decision": "stopped before phase M: KV4 edits not applied"})
        raise fp.StageSkipped(f"KV4 edits not applied, so no boot: {sorted(missing)}")
    ids, text_record = tf_text_ids(inner.tok, spec["measure"]["tf"]["source"],
                                   run.config["template"]["kwargs"],
                                   run.state.get("paths", {}),
                                   int(spec["measure"]["tf"]["max_tokens"]))
    site = fb.edit_site(run)
    extract_bundle(run)
    end = run.start + float(run.config["time_plan"][run.state.get("mode",
                                                                   "development")]["end_s"])
    stop_at = end - float(spec["boots"]["kill_reserve_s"])
    rows: list[dict[str, Any]] = []
    data: dict[str, Any] = {"tf_text": text_record, "rows": rows, "running": None}
    base: dict[str, Any] | None = None
    for index, variant in enumerate(variants):
        now = fp._now()
        why_not = should_run(variant, rows)
        window = stop_at - now
        if why_not is None and window < float(variant.get("min_window_s",
                                                          spec["boots"]["min_window_s"])):
            why_not = f"a window of {round(window)} s, under min_window_s"
        if why_not is not None:
            rows.append({"index": index, "name": variant["name"], "role": variant["role"],
                         "why": variant.get("why", ""), "outcome": "not_run",
                         "reason": why_not})
            continue
        deadline = min(now + float(spec["boots"]["budget_s"]), stop_at)
        write_kvab(run, {**data, "running": variant["name"]})
        print(f"== boot {index} {variant['name']} (t={round(now - run.start)} s, deadline "
              f"{round(deadline - run.start)} s)", flush=True)
        row = boot_variant(inner, variant, index, site, deadline, ids, rows)
        if variant["role"] == "baseline":
            base = row
        row["judged"] = judge(row, base, rows, spec["readings"])
        rows.append(row)
        print(f"   {variant['name']}: {row['outcome']}, {row['judged']['reading']} in "
              f"{row['seconds']} s {row['judged']['misses'][:3]}", flush=True)
        write_kvab(run, data)
    data["decision"] = decision(rows)
    write_kvab(run, data)
    return {"booted": sum(r["outcome"] != "not_run" for r in rows),
            "outcomes": {r["name"]: r["outcome"] for r in rows},
            "readings": {r["name"]: (r.get("judged") or {}).get("reading") for r in rows},
            "decision": data["decision"]}


def decision(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the pre-registered outcome of the run: the first candidate's reading.

    ``good``, ``bad``, or ``fault`` for the first candidate (B); ``fault`` too when the baseline
    did not serve. A later candidate (C) is reported beside it, never instead.
    """
    base = next((r for r in rows if r.get("role") == "baseline"), None)
    if base is None or (base.get("judged") or {}).get("reading") != "ok":
        return "fault"
    first = next((r for r in rows if r.get("role") == "candidate"), None)
    if first is None or first.get("outcome") == "not_run":
        return "fault"
    return (first.get("judged") or {}).get("reading", "fault")


def summary_markdown(run: Any) -> str:
    """Return ``summary.md``: the stages, phase K, and one line per boot."""
    lines = [f"# {run.config['run_id']}: KV-cache A/B", "", "| stage | status | seconds |",
             "|---|---|---|"]
    for name, record in run.state.items():
        if isinstance(record, dict) and "status" in record:
            lines.append(f"| {name} | {record['status']} | {record.get('seconds')} |")
    data = {}
    if (run.out / KVAB_FILE).is_file():
        data = json.loads((run.out / KVAB_FILE).read_text(encoding="utf-8"))
    lines += ["", f"Decision (the first candidate's reading): **{data.get('decision')}**", "",
              "| n | boot | outcome | reading | ready s | pool | top-1 | mean dlogp | p99 dlogp | "
              "H ratio | len ratio | decode ratio | misses |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in data.get("rows", []):
        judged = row.get("judged") or {}
        values = judged.get("values") or {}
        misses = "; ".join(judged.get("misses") or [row.get("reason", "")]).replace("|", "/")
        lines.append(f"| {row['index']} | {row['name']} | {row['outcome']} | "
                     f"{judged.get('reading')} | {row.get('ready_s')} | "
                     f"{(row.get('pool') or {}).get('pool_tokens')} | "
                     f"{values.get('top1_agreement')} | {values.get('mean_abs_lp_diff')} | "
                     f"{values.get('p99_abs_lp_diff')} | {values.get('entropy_ratio')} | "
                     f"{values.get('length_ratio')} | {values.get('decode_ratio_min')} | "
                     f"{misses[:200]} |")
    return "\n".join(lines) + "\n"


# ---- the processes -------------------------------------------------------------------------------


def inner_main(state_path: Path) -> int:
    """The inner process (the venv's Python): the tasks, phase K, phase M.

    Returns:
        0 when phase M ran (its rows hold the readings), else 1.
    """
    from arc26 import timing

    out = state_path.parent
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    run = fp.Run(config=meta["config"], root=Path(meta["root"]), out=out, start=meta["start"],
                 state=json.loads(state_path.read_text(encoding="utf-8")))
    paths = run.state.get("paths", {})
    inner = fp.Inner(run=run, base=f"http://127.0.0.1:{run.config['engine']['port']}",
                     tok=fp.make_tokenizer(run.config["tokenizer"], paths.get("model")),
                     mode=run.state.get("mode", fp.DEVELOPMENT))
    inner.context = int(run.config["solve"]["context_tokens"])
    timing.configure(out, worker="inner")
    try:
        run.stage("tasks", lambda: fp.stage_tasks(inner))
        kernel = run.stage("kernel_check", lambda: stage_kernel_check(inner))
        if kernel["status"] == "failed":
            run.state["killed"] = "phase K failed: no boot started"
        record = run.stage("ab", lambda: stage_ab(inner))
    finally:
        fp.kill_server(inner)
        run.save()
    return 0 if record["status"] == "ok" else 1


def outer_main(config: dict[str, Any], root: Path, commit: str,
               session_start: float | None = None) -> int:
    """The outer process: ``fn_probe.py``'s stages up to the server, then the inner process.

    Returns:
        The inner process's exit code, or 1 when a stage before it failed.
    """
    out = root / config["paths"]["out"].lstrip("/")
    out.mkdir(parents=True, exist_ok=True)
    for name in (KVAB_FILE, "engine_commands.jsonl"):
        (out / name).unlink(missing_ok=True)
    mode = fp.run_mode(config, os.environ)
    run = fp.Run(config=config, root=root, out=out, start=session_start or fp._now(),
                 state={"mode": mode})
    fp.write_json(out / "run.json", {"schema": fp.SCHEMA, "run_id": config["run_id"],
                                     "entry": "fn_kvab.py", "commit": commit,
                                     "start": run.start, "mode": mode, "root": str(root),
                                     "config": config, "python": sys.version,
                                     "outer_started_s": round(fp._now() - run.start, 1)})
    inputs = run.stage("inputs", lambda: fp.stage_inputs(run))
    run.state["scratch"] = str(run.path(config["paths"]["scratch"]))
    if inputs["status"] == "ok":
        run.stage("local_inputs", lambda: fp.stage_local_inputs(run))
        run.stage("python", lambda: fp.stage_python(run))
        run.stage("install", lambda: fp.stage_install(run))
        run.stage("cuda", lambda: fp.stage_cuda(run))
    venv_python = run.state.get("venv_python")
    code = 1
    if venv_python:
        paths = run.state.get("paths", {})
        script = Path(paths["code"]) / "fn_kvab.py" if paths.get("code") else Path(__file__)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(run.state.get("env_overrides", {}))
        if not config["install"].get("enabled", True):
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        run.save()
        timeout = max(run.left(), 60)
        print(f"== inner process ({round(timeout)} s left)", flush=True)
        try:
            done = subprocess.run([venv_python, str(script), "--inner", str(out / "state.json")],
                                  env=env, timeout=timeout, check=False)
            code = done.returncode
        except subprocess.TimeoutExpired:
            code = 124
        run.state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    run.state["outer"] = {"exit_inner": code, "wall_s": round(fp._now() - run.start, 1)}
    run.save()
    (out / "summary.md").write_text(summary_markdown(run), encoding="utf-8")
    return code


def reread(logs: Path, config: Mapping[str, Any], solutions: Path | None = None
           ) -> dict[str, Any]:
    """Recompute every comparison and reading of a pulled run with ``config``'s rules.

    Why: a rule or a parse fixed after a run must not need another GPU run; the pulled
    ``kvab_<n>_<name>_tf.json`` and ``_dev.jsonl`` files hold every number the readings use. With
    ``solutions`` (development scoring only, never in training) each boot's DEV answers are
    scored by ``arc26.evaluate`` through ``fn_report.py``'s helpers.

    Returns:
        ``kvab.json`` with each row's comparison, lengths, and reading recomputed, the
        decision, ``scores`` when asked, and ``reread`` true.
    """
    spec = config["kvab"]
    data = json.loads((logs / KVAB_FILE).read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    base = next((r for r in rows if r.get("role") == "baseline"), None)
    cap = int(config["solve"]["cap"]) - int(config["solve"]["forcing_tokens"])
    report = _load("fn_report") if solutions is not None else None
    scores: dict[str, Any] = {}
    done: list[dict[str, Any]] = []
    for row in rows:
        stem = f"kvab_{row['index']}_{row['name']}"
        tf_file = logs / f"{stem}_tf.json"
        base_file = logs / str((base or {}).get("tf", {}).get("file"))
        if base is not None and row is not base and tf_file.is_file() and base_file.is_file():
            theirs = json.loads(base_file.read_text(encoding="utf-8"))
            row["tf"]["compare"] = compare_tf(theirs, json.loads(
                tf_file.read_text(encoding="utf-8")))
        dev_file = logs / f"{stem}_dev.jsonl"
        if dev_file.is_file():
            records = fp.read_jsonl(dev_file)
            row["dev_lengths"] = dev_lengths(records, cap)
            if report is not None:
                preds = report.predictions(records)
                sols = report.load_solutions(solutions, sorted(preds))
                from arc26.evaluate import score_submission

                scored = [t for t in sorted(preds) if t in sols]
                result = score_submission(preds, sols, scored, label=f"{row['name']}")
                scores[row["name"]] = {"tasks": len(scored), "score": round(
                    sum(s.score for s in result.per_task), 3)}
        if row.get("outcome") != "not_run":
            row["judged"] = judge(row, base, done, spec["readings"])
        done.append(row)
    data["decision"] = decision(rows)
    if solutions is not None:
        data["scores"] = scores
    data["reread"] = True
    return data


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (``fn_probe.py``'s, which the notebook's cell 2 calls)."""
    parser = fp.build_parser()
    parser.add_argument("--reread", type=Path, default=None,
                        help="a pulled run_logs/fn folder: recompute the readings with "
                        "--config's rules into kvab_reread.json")
    parser.add_argument("--solutions", type=Path, default=None,
                        help="with --reread: score each boot's DEV answers (development only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe (outer process), its inner process with ``--inner``, or ``--reread``.

    Returns:
        The exit code: 0 when phase M ran, 1 when a stage before it failed.
    """
    args = build_parser().parse_args(argv)
    if args.inner is not None:
        return inner_main(args.inner)
    config = fp.load_config(args.config)
    if args.reread is not None:
        data = reread(args.reread, config, args.solutions)
        fp.write_json(args.reread / "kvab_reread.json", data)
        print(json.dumps({"decision": data["decision"], **{
            r["name"]: (r.get("judged") or {}).get("reading") for r in data["rows"]}}))
        return 0
    root = args.root.resolve()
    if args.smoke:
        if root == Path("/"):
            raise SystemExit("--smoke needs --root, a scratch folder for the fake inputs")
        config = fp.apply_smoke(config)
        fp.build_smoke_tree(root, config)
    commit = args.commit or config.get("build", {}).get("commit") or "unknown"
    code = outer_main(config, root, commit, args.session_start)
    print(f"exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
