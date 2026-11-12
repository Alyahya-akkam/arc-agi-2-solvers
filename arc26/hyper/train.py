r"""Meta-training loop for the hypernetwork H (task P3.5; BRIEF B2 Q1, Q2, Q3, Q6, B5 P3.5).

H learns from P3.4's episodes to emit the LoRA adapter that makes the frozen base predict a query's
output grid from its demonstrations. An episode $(D, Q, a)$ is a set of demonstration pairs $D$,
one query pair $Q = (x_Q, y_Q)$, and an augmentation id $a$ (``episodes.PermutationTable``). The
base reads $a(D)$ then $a(Q)$ in the notebook's exact token format (``QwenFormatter.fmt_train``,
the formatter P3.2's ``render_pair`` checks), and the loss is the teacher-forced cross-entropy of
the query's output-grid tokens $O_Q$ (Q6, ADR-006), here with the ``<|im_end|>`` that closes the
grid (``loss_includes_eos``, on by default; see "Reading of Q6" below):

$$\mathcal{L}(D, Q, a) = -\frac{1}{|O_Q|} \sum_{t \in O_Q}
\log p\big(y_t \mid y_{<t};\ \theta_\text{base} + \Delta\theta_{H(\bar z(D))}\big),$$

normalized per episode (D2L's per-context mean) and averaged over the episodes of an optimizer
step. $\bar z(D) = \frac{1}{|D|} \sum_{i \in D} \bar e_i$, where $\bar e_i$ is P3.2's pair
embedding (the mean over the encoder's views of the pooled layer-$L$ states), computed from the
**un-augmented** demonstrations: one adapter must serve every decode view at test time, so H's
input must not depend on $a$. Gradients flow through the frozen bf16 base into the injected
factors and from there into H (P3.1's ``inject``, backward inside the block). Math, data flow,
and limits: ``docs/modules/train.md``; runbook ``docs/runbooks/train_hypernet.md``.

Pieces, in data-flow order:

- Episode set: ``open_episode_set`` finds P3.4's build by its ``MANIFEST.json``, refuses any
  manifest whose sha256 is not the config's and any leakage check report (``leakage.json``) for it
  that did not pass; ``load_split`` verifies each shard's size and sha256 and validates every
  record (``episodes.read_shards``), so an over-length episode stops the run instead of being cut.
- Sequences: ``build_sequence`` renders $a(D), a(Q)$ pair by pair with ``render_pair``, checks
  that the whole text tokenizes to the concatenation, and locates $O_Q$; ``query_ce`` is the loss.
- Embeddings: ``PairEmbeddings`` holds $\bar e_i$ by pair hash for the run, read from an
  ``EmbeddingCache`` or encoded with ``PairEncoder`` on the same base instance.
- Training: ``train`` runs ``run_worker`` in ``gpus`` processes (``torch.multiprocessing.spawn``,
  NCCL on GPU, gloo on CPU) with DistributedDataParallel on H only; the base is frozen and
  replicated. Each optimizer step takes ``gpus * accum`` episodes from a seeded order
  (``EpisodeOrder``, slot $g = (s \cdot A + m) \cdot W + k$ for step $s$, micro step $m$, rank $k$):
  every episode once per pass, or with ``source_weights`` each source at its configured share of
  the draws (blocks of ``SOURCE_BLOCK`` slots), each episode once per pass over its source.
- Validation at step 0 and every ``eval_every`` steps: mean query CE on the validation split for
  the zero adapter, the current H, and a mismatched-$\bar z$ control (``mismatch_permutation``),
  with both paired differences in standard errors.
- Checkpoints (``save_checkpoint``) hold H, the optimizer, the scheduler, the step, the data
  cursor, every rank's RNG states, and the metrics history; ``--resume`` continues bit-identically
  on CPU. ``save_best`` rewrites ``best.pt``, H's weights alone, at every new best validation
  point over all sessions (``is_better``): under ``zero_step`` a new low of the current H's
  validation CE, under ``fomaml`` a higher post-$k$ greedy exact match, ties by a lower mean $S$.
- Warm start (``--init-from``, P3.3b): a ``set`` H (``hypernet.aggregator``) starts from a full
  checkpoint of a ``mean`` H. Its weights load by name, the new aggregator keeps its
  initialization ($W_o = 0$, so the loaded H's adapters are reproduced exactly), and the run
  continues at the checkpoint's step and data cursor with its optimizer state and cosine; the new
  parameters get fresh AdamW moments and parameter groups of their own (``new_params_lr``, a
  warmup of ``new_params_warmup_steps`` from the start step, then a cosine to ``min_lr`` at
  ``steps``; ``NewParamsSchedule``).
- Objective (P3.9, ADR-006 amendment 2): ``zero_step`` (the default) is the loss above;
  ``fomaml`` takes ``inner.steps`` inner steps on the demonstrations from a detached copy of H's
  factors (``adapt``) and trains H on the query CE at the adapted factors, its gradient applied at
  H's output (first-order MAML); each inner gradient is clipped as the notebook's fit clips its
  own (``clip_inner_grads``). Validation adds decode-aligned metrics for every reference
  (``decode_metrics``) and, under ``fomaml``, the current H after its inner steps (``adapted``);
  ``--init-from`` may change the objective of a full checkpoint without adding parameters, and
  with ``fresh_schedule`` start a new phase, also from a phase's checkpoint. ``inner_probe``
  scores the warm-started H after its inner steps at several inner rates before training and
  trains at the one ``choose_inner_lr`` picks. ``best.pt`` carries ``h_config`` (no objective
  keys), so a loader on the code before P3.9 parses it.
- Kill criteria checked by the loop itself: a non-finite loss or gradient norm, a step over
  ``kill.max_step_s``, a projected run wall time over ``kill.max_projected_s`` (exit 4); the
  session budget ``max_wall_s`` checkpoints and stops (exit 3, resumable).

Reading of Q6. Q6 scores "output-grid tokens". With ``loss_includes_eos`` (the default) $O_Q$
also holds the ``<|im_end|>`` that ends the query's output, as the notebook's completion-only
training does (``QwenDataCollatorForCompletionOnlyLM``: each reply's grid tokens and its end
token): H's adapter serves the notebook's decode, where that token ends a grid, so it carries the
grid's last row and with it its shape. Off, $O_Q$ is the grid's digits and newlines alone.

Deviations from the task brief (docs/tasks/P3.5.md), each the smallest sound change:

1. Embeddings are encoded in one sharded pass before step 0, not lazily per step. Chunk $j$ of the
   sorted missing pair hashes goes to rank $j \bmod W$ and the results are all-gathered, so each
   distinct pair is encoded once per run instead of once per rank, step times hold no encoder
   spikes (the step-time kill criterion stays meaningful), and a pair's batch, hence its
   embedding, does not depend on $W$ or on a resume (for the same cache contents: the chunks are
   cut from the pairs the cache lacks). Values are the brief's: ``PairEncoder`` on the same base
   under ``torch.no_grad``, memoized by pair hash.
2. The zero-adapter CE is computed once, at the first validation: the base is frozen, so it is a
   constant, which the checkpoint carries.
3. Beside the brief's kill criteria (exit 4), a session wall budget stops the run with a
   checkpoint (exit 3) so a long run can span Kaggle sessions; a kill also writes a checkpoint
   (the state before the step that tripped it), so an amended run need not start over.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import re
import socket
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from arc26.hyper.encoder import (
    POOLS,
    ArcPair,
    EmbeddingCache,
    PairEncoder,
    apply_view,
    pair_hash,
    render_pair,
    view_descriptors,
)
from arc26.hyper.episodes import (
    SOURCES,
    SPLITS,
    EpisodeRecord,
    PermutationTable,
    derive_seed,
    keyed_order,
    read_shards,
    rng_for,
)
from arc26.hyper.hypernet import BaseShapes, Hypernet, HypernetConfig
from arc26.hyper.inject import LoRAFactors, inject, wrap_targets
from arc26.nvarc.arc_loader import QwenFormatter
from arc26.timing import TimingLogger

EXIT_OK, EXIT_FAILED, EXIT_CONFIG, EXIT_STOPPED, EXIT_KILLED = 0, 1, 2, 3, 4
EPISODES_TASK = "P3.4"  # the task field of P3.4's MANIFEST.json
MANIFEST = "MANIFEST.json"
CHECK_REPORT = "leakage.json"
LOG_DIR = "run_logs"  # P3.4 copies its manifest into run_logs/, which is never the build
STATUS_FILE = "status.json"
METRICS_FILE = "metrics.jsonl"
LATEST_FILE = "latest.json"
# H's weights at the best validation point so far (is_better: the lowest current-H CE under
# zero_step, the most post-k exact matches under fomaml), beside the full checkpoints.
BEST_FILE = "best.pt"
CHECKPOINT_SCHEMA = 1
# Slots per block of a source-weighted episode order: every aligned block of SOURCE_BLOCK slots
# holds each source's share exactly, rounded to whole draws. A window that straddles two blocks
# can differ by a few points of share (up to 3.9 at 992 slots on the MT pool, 1.0 at 3,200).
SOURCE_BLOCK = 1000
REFERENCES = ("zero", "current", "mismatched")
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
ATTENTION = ("auto", "flash_attention_2", "sdpa", "eager")
PENALTIES = ("none", "l1", "l2")
DEVICES = ("cuda", "cpu")
# Settings a resumed run must share with its checkpoint: they define the data order, the model,
# and the optimization. Paths, limits, and the evaluation and checkpoint cadence may change.
RESUME_KEYS = (
    "manifest_sha256",
    "train_split",
    "val_split",
    "max_tokens",
    "loss_includes_eos",
    "model_id",
    "dtype",
    "encoder",
    "hypernet",
    "gpus",
    "accum",
    "steps",
    "lr",
    "min_lr",
    "warmup_steps",
    "weight_decay",
    "betas",
    "eps",
    "clip",
    "penalty",
    "penalty_weight",
    "kl_weight",
    "seed",
    "source_weights",
    "objective",
    "inner",
    "fresh_schedule",
    "inner_probe",
)
# The P3.9 settings among RESUME_KEYS (ADR-006 amendment 2): absent from every config and
# checkpoint written before P3.9, so an absent key reads as its default (MT-02's, MT-03's, and
# X1-01's checkpoints resume as before), and to_dict leaves a default out. --init-from may change
# them: a new objective or a new schedule starts a new training phase.
OBJECTIVE_KEYS = ("objective", "inner", "fresh_schedule", "inner_probe")
OBJECTIVE_DEFAULTS: dict[str, Any] = {
    "objective": "zero_step",
    "inner": None,
    "fresh_schedule": False,
    "inner_probe": None,
}
# The schedule settings a fresh-schedule warm start sets anew (the phase's own warmup and cosine).
SCHEDULE_KEYS = ("steps", "lr", "min_lr", "warmup_steps")
# zero_step: the query CE at H's adapter itself (ADR-006). fomaml: the query CE after k inner
# steps on the demonstrations, its gradient applied at H's adapter (first-order MAML).
OBJECTIVES = ("zero_step", "fomaml")
INNER_OPTIMIZERS = ("adam", "sgd")
# episode: every inner step reads a(D), the episode's own view; drawn: inner step j reads
# b_j(D), b_j drawn per episode from the build's augmentations (the fit's one view per step).
INNER_VIEWS = ("episode", "drawn")
# The decode's per-view bound: a continuation survives while its summed NLL stays under -ln 0.2.
DECODE_BOUND = math.log(5)
# Floor of the geometric-mean CE: the logs' 6-decimal rounding (survey section 2.3).
GEOMEAN_FLOOR = 1e-6
# The new parameter groups' settings of a warm start (--init-from). Not in RESUME_KEYS, so a
# resume of a mean run compares what it compared before P3.3b; a resume of a warm-started run
# checks them against its checkpoint's init_from record instead.
NEW_PARAMS_KEYS = ("new_params_lr", "new_params_warmup_steps")
# The notebook's 16 token ids (ARC_VOCAB and the ids in arc26/nvarc/arc_solver.py): digits 0 to 9,
# "Ċ" (a byte-level newline) 10, user 11, assistant 12, padding 13, <|im_start|> 14 (the one id
# left, inferred), <|im_end|> 15. The same table as scripts/cache_embeddings.py's stand-in.
STANDIN_VOCAB = {
    **{str(d): d for d in range(10)},
    "Ċ": 10,
    "user": 11,
    "assistant": 12,
    "<|endoftext|>": 13,
    "<|im_start|>": 14,
    "<|im_end|>": 15,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class EpisodeSetError(ValueError):
    """The episode build, its manifest, or its leakage check report is refused."""


# --- configuration -------------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EncoderSettings:
    r"""P3.2's encoder settings for $\bar z$; they must equal the test-time encoder's.

    Attributes:
        layer: $L$, decoder layers run before pooling.
        pool: ``output`` (Q1) or ``input``.
        n_perm: Color permutations per dihedral transform; 2 gives the 16 decode views.
        view_seed: Seed of the permutation draws; 2 is the decode path's.
        batch_tokens: Padded tokens per prefill batch.
        chunk_pairs: Pairs per encode call in the sharded pass (deviation 1 of the module doc).

    Raises:
        ValueError: If a value has the wrong type or range.
    """

    layer: int
    pool: str
    n_perm: int
    view_seed: int
    batch_tokens: int
    chunk_pairs: int

    def __post_init__(self) -> None:
        for name in ("layer", "n_perm", "view_seed"):
            value = getattr(self, name)
            _require(_is_int(value) and value >= 0, f"encoder.{name} must be an integer >= 0")
        for name in ("batch_tokens", "chunk_pairs"):
            value = getattr(self, name)
            _require(_is_int(value) and value >= 1, f"encoder.{name} must be an integer >= 1")
        _require(self.pool in POOLS, f"encoder.pool must be one of {POOLS}")


@dataclass(frozen=True)
class KillCriteria:
    """Limits the loop enforces itself; a breach writes a checkpoint and exits 4.

    Attributes:
        max_step_s: Longest allowed optimizer step, seconds.
        max_projected_s: Longest allowed projected wall time of the whole run, seconds, over all
            sessions: elapsed plus the remaining steps, evaluations, and checkpoints at this
            session's mean costs.
        projection_after_steps: Steps of this session before the projection is checked.

    Raises:
        ValueError: If a value has the wrong type or range.
    """

    max_step_s: float
    max_projected_s: float
    projection_after_steps: int

    def __post_init__(self) -> None:
        for name in ("max_step_s", "max_projected_s"):
            value = getattr(self, name)
            _require(_is_number(value) and value > 0, f"kill.{name} must be a positive number")
        value = self.projection_after_steps
        _require(_is_int(value) and value >= 1, "kill.projection_after_steps must be >= 1")


@dataclass(frozen=True)
class InnerLoop:
    r"""The inner loop of the ``fomaml`` objective: $k$ steps on the demonstrations per episode.

    Step $j$ moves the episode's own copy of the factors,
    $\theta_{j+1} = \theta_j - \alpha\,u_j$, with $u_j = g_j$ (``sgd``) or Adam's
    bias-corrected normalized step from moments that start at zero for every episode (``adam``):
    $m_j = \beta_1 m_{j-1} + (1 - \beta_1) g_j$, $v_j = \beta_2 v_{j-1} + (1 - \beta_2) g_j^2$,
    $u_j = \hat m_j / (\sqrt{\hat v_j} + \epsilon)$, $\hat m_j = m_j / (1 - \beta_1^{j+1})$,
    $\hat v_j = v_j / (1 - \beta_2^{j+1})$; $g_j$ is the gradient of the demonstrations' CE at
    $\theta_j$, clipped as the notebook's fit clips it (``max_grad_norm`` 1.0 in
    ``arc26/solver.py``, torch's ``clip_grad_norm_``): $g_j \leftarrow c_j\,g_j$ with
    $c_j = \min(1, \tau / (\lVert g_j\rVert_2 + 10^{-6}))$, the norm over every factor of the
    episode (``clip_inner_grads``). No weight decay, as the fit. Adam's first step is
    $\mathrm{sign}(g_0)$ whatever $c_0$ is (up to $\epsilon$), but from the second step on the
    clip changes the update: $\hat m_1 \propto \beta_1(1-\beta_1)\,c_0 g_0 + (1-\beta_1)\,c_1 g_1$
    weighs the two gradients by their clip factors, and ``sgd`` moves by $c_j g_j$.

    Attributes:
        steps: $k$; 0 makes ``fomaml`` the ``zero_step`` objective, bit for bit.
        lr: $\alpha$.
        optimizer: ``adam`` or ``sgd``.
        betas: $(\beta_1, \beta_2)$ of ``adam``.
        eps: $\epsilon$ of ``adam``.
        views: ``episode`` or ``drawn`` (``INNER_VIEWS``).
        clip: $\tau$, the global gradient-norm clip of each inner step; None for no clip.

    Raises:
        ValueError: If a value has the wrong type or range.
    """

    steps: int
    lr: float
    optimizer: str
    betas: tuple[float, float]
    eps: float
    views: str
    clip: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        _require(_is_int(self.steps) and self.steps >= 0, "inner.steps must be an integer >= 0")
        for name in ("lr", "eps"):
            value = getattr(self, name)
            _require(_is_number(value) and value > 0, f"inner.{name} must be a positive number")
        _require(self.optimizer in INNER_OPTIMIZERS, f"inner.optimizer in {INNER_OPTIMIZERS}")
        _require(
            len(self.betas) == 2 and all(_is_number(b) and 0 <= b < 1 for b in self.betas),
            "inner.betas must be two numbers in [0, 1)",
        )
        _require(self.views in INNER_VIEWS, f"inner.views must be one of {INNER_VIEWS}")
        _require(
            self.clip is None or (_is_number(self.clip) and self.clip > 0),
            "inner.clip must be null or a positive number",
        )


@dataclass(frozen=True)
class InnerProbe:
    r"""The inner learning rates a warm start scores before it trains, and the rule that picks one.

    At the start step of a ``--init-from`` run, validation scores the loaded H after its $k$
    inner steps at every rate of ``lrs`` (``adapted`` per rate), then trains at the rate
    ``choose_inner_lr`` picks: with $E(\alpha)$ the post-$k$ greedy exact matches (episodes) and
    $\bar S(\alpha)$ the mean summed NLL, among the rates with finite scores,

    $$\alpha^* = \operatorname*{arg\,min}_{\alpha \in C} \bar S(\alpha), \qquad
    C = \{\alpha : E(\alpha) \ge \max_\beta E(\beta) - m\},$$

    ties to the smaller $\alpha$. $m$ = ``margin`` episodes count as a tie on the exact match.

    Attributes:
        lrs: The candidate rates, ascending and distinct; ``inner.lr`` must be one of them.
        margin: $m$, in episodes.

    Raises:
        ValueError: If a value has the wrong type or range.
    """

    lrs: tuple[float, ...]
    margin: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "lrs", tuple(self.lrs))
        lrs = self.lrs
        _require(
            len(lrs) >= 2 and all(_is_number(v) and v > 0 for v in lrs),
            "inner_probe.lrs must hold at least two positive numbers",
        )
        _require(
            all(a < b for a, b in zip(lrs, lrs[1:], strict=False)),
            "inner_probe.lrs must be ascending and distinct",
        )
        _require(
            _is_int(self.margin) and self.margin >= 0, "inner_probe.margin must be an integer >= 0"
        )


@dataclass(frozen=True)
class TrainConfig:
    r"""One training run; ``configs/train_hypernet*.yaml`` give every value with its reason.

    Attributes:
        run_id: Pre-run brief id.
        episodes_root: Folder searched for P3.4's build (its ``MANIFEST.json``).
        manifest_sha256: sha256 of the build's ``MANIFEST.json`` bytes; any other is refused.
        check_root: Folder searched for the leakage check report of that manifest.
        train_split: Split trained on (``toy`` or ``train``).
        val_split: Split validated on (``val``).
        max_tokens: Longest sequence allowed (the notebook's 8192); longer episodes are refused.
        loss_includes_eos: Whether $O_Q$ holds the ``<|im_end|>`` after the query's grid (default
            true), as the notebook's completion-only training does.
        model_id: Weights and precision of the base, part of the embedding-cache key.
        model_dir: Hugging Face model folder; None selects the tiny random stand-in (smoke).
        device: ``cuda`` (rank $k$ uses ``cuda:k``) or ``cpu``.
        dtype: Base precision, a key of ``DTYPES``.
        attn_implementation: One of ``ATTENTION``; ``auto`` is FlashAttention 2 when CUDA, bf16,
            and ``flash_attn`` are available, else SDPA.
        gradient_checkpointing: Non-reentrant activation checkpointing of the decoder layers.
        encoder: P3.2's settings for $\bar z$.
        embedding_cache: ``EmbeddingCache`` root to read pair embeddings from; None encodes all.
        hypernet: H's architecture and scope (``configs/hypernet.yaml``).
        gpus: Processes, one GPU each (or CPU processes on gloo).
        accum: Micro steps (episodes) per process per optimizer step.
        steps: Optimizer steps.
        lr: Peak learning rate.
        min_lr: Learning rate at the end of the cosine.
        warmup_steps: Steps of linear warmup.
        weight_decay: AdamW decay of 2-D weights (not norms, biases, layer and module codes).
        betas: AdamW betas.
        eps: AdamW epsilon.
        clip: Gradient-norm clip.
        penalty: ``none``, ``l1``, or ``l2`` on the emitted factors.
        penalty_weight: Weight of the penalty; positive exactly when ``penalty`` is not ``none``.
        kl_weight: $\lambda$ of Q6's optional KL term; must be 0 (not implemented, A7 cut 4).
        seed: Seed of the data order, the mismatch control, and the per-rank RNGs.
        eval_every: Steps between validations (also at step 0 and at the end).
        ckpt_every: Steps between checkpoints (also at the end and at every stop).
        keep_checkpoints: Newest checkpoints kept in ``ckpt_dir``.
        ckpt_dir: Checkpoint folder (stays on Kaggle).
        out_dir: Run-log folder (pulled).
        max_wall_s: Session wall budget, seconds from the script's start; checkpoint and exit 3.
        kill: Kill criteria.
        loss_includes_eos: Whether $O_Q$ holds the closing ``<|im_end|>`` (default true).
        source_weights: Share of the training draws per source (keys from
            ``episodes.SOURCES``, any positive scale; ``EpisodeOrder`` normalizes them and
            requires exactly the training split's sources). None (the default) draws every
            episode once per pass, so each source in proportion to its episodes.
        new_params_lr: Peak learning rate of the parameters ``--init-from`` leaves at their
            initialization (the set aggregator); None (the default) when the run adds none.
        new_params_warmup_steps: Their linear warmup, in steps from the start step; None (the
            default) when the run adds none.
        objective: ``zero_step`` (the default, ADR-006) or ``fomaml`` (P3.9: the query CE after
            ``inner.steps`` inner steps, first order).
        inner: The inner loop; set exactly when ``objective`` is ``fomaml``.
        fresh_schedule: With ``--init-from``, start a new training phase (P3.9): H's weights from
            the checkpoint, fresh AdamW moments for every parameter, and a new warmup of
            ``warmup_steps`` and cosine from ``lr`` to ``min_lr`` over the phase, which runs from
            the checkpoint's step to ``steps`` (the logged step continues the checkpoint's).
            False (the default) continues the checkpoint's optimizer and cosine. Setup refuses
            it without ``--init-from`` (or a ``--resume`` of such a phase).
        inner_probe: The inner rates a warm start scores at its start step before training at
            the one ``choose_inner_lr`` picks (``InnerProbe``); None (the default) trains at
            ``inner.lr``. Needs ``fomaml`` with $k \ge 1$, and ``inner.lr`` among the rates.

    Raises:
        ValueError: If a value has the wrong type or range, ``kl_weight`` is not 0,
            ``inner`` is set for ``zero_step`` or missing for ``fomaml``, or ``inner_probe`` is
            set without an inner step or without ``inner.lr`` among its rates.
    """

    run_id: str
    episodes_root: str
    manifest_sha256: str
    check_root: str
    train_split: str
    val_split: str
    max_tokens: int
    model_id: str
    model_dir: str | None
    device: str
    dtype: str
    attn_implementation: str
    gradient_checkpointing: bool
    encoder: EncoderSettings
    embedding_cache: str | None
    hypernet: HypernetConfig
    gpus: int
    accum: int
    steps: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    clip: float
    penalty: str
    penalty_weight: float
    kl_weight: float
    seed: int
    eval_every: int
    ckpt_every: int
    keep_checkpoints: int
    ckpt_dir: str
    out_dir: str
    max_wall_s: float
    kill: KillCriteria
    loss_includes_eos: bool = True
    source_weights: dict[str, float] | None = None
    new_params_lr: float | None = None
    new_params_warmup_steps: int | None = None
    objective: str = "zero_step"
    inner: InnerLoop | None = None
    fresh_schedule: bool = False
    inner_probe: InnerProbe | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        _require(self.objective in OBJECTIVES, f"objective must be one of {OBJECTIVES}")
        _require(
            (self.inner is not None) == (self.objective == "fomaml"),
            "inner must be set exactly when objective is fomaml",
        )
        _require(
            self.inner is None or isinstance(self.inner, InnerLoop), "inner must be an InnerLoop"
        )
        _require(isinstance(self.fresh_schedule, bool), "fresh_schedule must be true or false")
        probe = self.inner_probe
        if probe is not None:
            _require(isinstance(probe, InnerProbe), "inner_probe must be an InnerProbe")
            _require(
                self.inner is not None and self.inner.steps >= 1,
                "inner_probe needs objective fomaml with inner.steps >= 1",
            )
            _require(self.inner.lr in probe.lrs, "inner.lr must be one of inner_probe.lrs")
        weights = self.source_weights
        if weights is not None:
            _require(
                isinstance(weights, Mapping) and len(weights) > 0,
                "source_weights must be null or a mapping of source to weight",
            )
            unknown = sorted(str(name) for name in weights if name not in SOURCES)
            _require(not unknown, f"source_weights: unknown sources {unknown}, known {SOURCES}")
            _require(
                all(_is_number(value) and value >= 0 for value in weights.values()),
                "source_weights: every weight must be a number >= 0",
            )
            _require(sum(weights.values()) > 0, "source_weights: one weight must be positive")
            # Sorted plain floats, so to_dict and the resume comparison see one form.
            object.__setattr__(
                self, "source_weights", {name: float(weights[name]) for name in sorted(weights)}
            )
        for name in ("run_id", "episodes_root", "check_root", "model_id", "ckpt_dir", "out_dir"):
            value = getattr(self, name)
            _require(isinstance(value, str) and value != "", f"{name} must be a non-empty string")
        for name in ("model_dir", "embedding_cache"):
            value = getattr(self, name)
            _require(
                value is None or (isinstance(value, str) and value != ""), f"{name}: null or a path"
            )
        _require(
            isinstance(self.manifest_sha256, str) and bool(_SHA256.fullmatch(self.manifest_sha256)),
            "manifest_sha256 must be 64 lowercase hex characters",
        )
        _require(self.train_split in SPLITS and self.val_split in SPLITS, f"splits from {SPLITS}")
        _require(self.train_split != self.val_split, "train_split and val_split must differ")
        _require(self.device in DEVICES, f"device must be one of {DEVICES}")
        _require(self.dtype in DTYPES, f"dtype must be one of {sorted(DTYPES)}")
        _require(self.attn_implementation in ATTENTION, f"attn_implementation in {ATTENTION}")
        _require(isinstance(self.gradient_checkpointing, bool), "gradient_checkpointing: bool")
        _require(
            isinstance(self.loss_includes_eos, bool), "loss_includes_eos must be true or false"
        )
        _require(isinstance(self.encoder, EncoderSettings), "encoder must be EncoderSettings")
        _require(isinstance(self.hypernet, HypernetConfig), "hypernet must be a HypernetConfig")
        _require(isinstance(self.kill, KillCriteria), "kill must be KillCriteria")
        for name in ("max_tokens", "gpus", "accum", "steps", "eval_every", "ckpt_every"):
            value = getattr(self, name)
            _require(_is_int(value) and value >= 1, f"{name} must be an integer >= 1")
        _require(_is_int(self.keep_checkpoints) and self.keep_checkpoints >= 1, "keep_checkpoints")
        _require(_is_int(self.seed), "seed must be an integer")
        warm = self.warmup_steps
        _require(_is_int(warm) and 0 <= warm < self.steps, "warmup_steps must be in [0, steps)")
        for name in ("lr", "eps", "clip", "max_wall_s"):
            value = getattr(self, name)
            _require(_is_number(value) and value > 0, f"{name} must be a positive number")
        for name in ("min_lr", "weight_decay", "penalty_weight", "kl_weight"):
            value = getattr(self, name)
            _require(_is_number(value) and value >= 0, f"{name} must be a number >= 0")
        _require(self.min_lr <= self.lr, "min_lr must not exceed lr")
        _require(
            len(self.betas) == 2 and all(_is_number(b) and 0 <= b < 1 for b in self.betas),
            "betas must be two numbers in [0, 1)",
        )
        _require(self.penalty in PENALTIES, f"penalty must be one of {PENALTIES}")
        _require(
            (self.penalty == "none") == (self.penalty_weight == 0),
            "penalty_weight must be positive exactly when penalty is not none",
        )
        _require(
            self.kl_weight == 0,
            "kl_weight must be 0: Q6's KL term is an interface only (BRIEF A7 cut order, item 4)",
        )
        lr_new, warm_new = self.new_params_lr, self.new_params_warmup_steps
        _require(
            lr_new is None or (_is_number(lr_new) and lr_new >= self.min_lr and lr_new > 0),
            "new_params_lr must be null or a positive number >= min_lr",
        )
        _require(
            warm_new is None or (_is_int(warm_new) and warm_new >= 0),
            "new_params_warmup_steps must be null or an integer >= 0",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        """Build a config from plain values, such as a parsed YAML file.

        Args:
            data: Every field without a default (``loss_includes_eos``, ``source_weights``, and
                the ``new_params_*`` settings have one); ``encoder``, ``hypernet``, and ``kill``
                as mappings (``hypernet.aggregator`` may be left out: ``mean``).

        Returns:
            The validated config.

        Raises:
            ValueError: If a key is missing or unknown, or a value is invalid.
        """
        names = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING}
        missing, unknown = sorted(required - data.keys()), sorted(data.keys() - names)
        _require(not missing and not unknown, f"missing keys {missing}, unknown keys {unknown}")
        nested = {"encoder": EncoderSettings, "hypernet": HypernetConfig, "kill": KillCriteria}
        values = dict(data)
        for key, kind in nested.items():
            section = values[key]
            _require(isinstance(section, Mapping), f"{key} must be a mapping")
            expected = {f.name for f in fields(kind)}
            needed = {f.name for f in fields(kind) if f.default is MISSING}
            missing, unknown = sorted(needed - section.keys()), sorted(section.keys() - expected)
            _require(not missing and not unknown, f"{key}: missing {missing}, unknown {unknown}")
            values[key] = kind(**section)
        inner = values.get("inner")
        if inner is not None:
            _require(isinstance(inner, Mapping), "inner must be a mapping")
            expected = {f.name for f in fields(InnerLoop)}
            missing, unknown = sorted(expected - inner.keys()), sorted(inner.keys() - expected)
            _require(not missing and not unknown, f"inner: missing {missing}, unknown {unknown}")
            values["inner"] = InnerLoop(**inner)
        probe = values.get("inner_probe")
        if probe is not None:
            _require(isinstance(probe, Mapping), "inner_probe must be a mapping")
            expected = {f.name for f in fields(InnerProbe)}
            missing, unknown = sorted(expected - probe.keys()), sorted(probe.keys() - expected)
            _require(
                not missing and not unknown, f"inner_probe: missing {missing}, unknown {unknown}"
            )
            values["inner_probe"] = InnerProbe(**probe)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain JSON and YAML values (tuples become lists).

        ``hypernet`` is ``HypernetConfig.to_dict`` (no ``aggregator`` block for the default
        ``mean``), and unset ``new_params_*`` settings are left out, so a run without the P3.3b
        options writes the config of the code before them; so are the ``OBJECTIVE_KEYS`` at
        their defaults, the config of the code before P3.9.
        """
        data = _plain(asdict(self))
        data["hypernet"] = _plain(self.hypernet.to_dict())
        for name in NEW_PARAMS_KEYS:
            if data[name] is None:
                del data[name]
        for name in OBJECTIVE_KEYS:
            if data[name] == OBJECTIVE_DEFAULTS[name]:
                del data[name]
        return data


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --- the episode set ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeSet:
    """A verified P3.4 build.

    Attributes:
        folder: The folder holding ``MANIFEST.json``.
        manifest: The parsed manifest.
        manifest_sha256: sha256 of the manifest's bytes, equal to the config's.
        check_report: The leakage check report that passed for this manifest.
        table: The build's permutation table, which decodes the augmentation ids.
    """

    folder: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    check_report: Path
    table: PermutationTable


def _walk_files(root: Path) -> Iterator[Path]:
    """Every file under ``root`` in sorted order, following links into each real folder once."""
    seen: set[str] = set()
    for folder, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(folder)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        for name in sorted(files):
            yield Path(folder) / name


def find_episode_build(root: Path, manifest_sha256: str) -> Path:
    """Find the P3.4 build whose ``MANIFEST.json`` has the given sha256.

    Why: Kaggle mounts moved within 2026 (docs/runbooks/build_submission.md), so the build is
    found by its manifest, and only the manifest the leakage check passed may be trained on.

    Args:
        root: The build folder or a folder above it, such as ``/kaggle/input``.
        manifest_sha256: Expected sha256 of the manifest's bytes.

    Returns:
        The folder holding the manifest.

    Raises:
        EpisodeSetError: If no P3.4 manifest is found, or none has that sha256.
    """
    found: dict[Path, str] = {}
    for path in _walk_files(Path(root)):
        if path.name != MANIFEST or path.parent.name == LOG_DIR:
            continue
        try:
            data = path.read_bytes()
            task = json.loads(data).get("task")
        except (OSError, ValueError, AttributeError):
            continue
        if task == EPISODES_TASK:
            found[path.parent] = hashlib.sha256(data).hexdigest()
    if not found:
        raise EpisodeSetError(f"no {EPISODES_TASK} {MANIFEST} under {root}")
    matching = [folder for folder, digest in found.items() if digest == manifest_sha256]
    if not matching:
        seen = sorted({digest[:12] for digest in found.values()})
        raise EpisodeSetError(
            f"no {EPISODES_TASK} build under {root} has manifest sha256 {manifest_sha256}; "
            f"found {seen}"
        )
    return matching[0]


def find_check_report(root: Path, manifest_sha256: str) -> Path:
    """Find the passing leakage check report (P3.4's checker) of a manifest.

    Every ``leakage.json`` under ``root`` that names the manifest must pass: a failed check of the
    same build refuses the build even when another report passed.

    Args:
        root: Folder searched, such as ``/kaggle/input``.
        manifest_sha256: The manifest the report must name (``manifest_sha256``).

    Returns:
        The first passing report in path order.

    Raises:
        EpisodeSetError: If no report names the manifest, or one that does has ``pass`` other
            than true.
    """
    reports: list[tuple[Path, Any]] = []
    for path in _walk_files(Path(root)):
        if path.name != CHECK_REPORT:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(report, dict) and report.get("manifest_sha256") == manifest_sha256:
            reports.append((path, report))
    if not reports:
        raise EpisodeSetError(f"no {CHECK_REPORT} for manifest {manifest_sha256} under {root}")
    failed = [str(path) for path, report in reports if report.get("pass") is not True]
    if failed:
        raise EpisodeSetError(f"the leakage check of manifest {manifest_sha256} failed: {failed}")
    return reports[0][0]


def open_episode_set(
    episodes_root: Path, manifest_sha256: str, check_root: Path, max_tokens: int
) -> EpisodeSet:
    """Locate and verify the build and its leakage check (task brief, deliverable 1).

    Args:
        episodes_root: Folder searched for the build.
        manifest_sha256: The configured manifest sha256.
        check_root: Folder searched for the check report.
        max_tokens: The run's sequence cap; the build's own cap may not exceed it.

    Returns:
        The episode set.

    Raises:
        EpisodeSetError: If the build or the report is refused, or the build allows longer
            episodes than ``max_tokens``.
    """
    folder = find_episode_build(Path(episodes_root), manifest_sha256)
    data = (folder / MANIFEST).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest_sha256:
        raise EpisodeSetError(f"{folder / MANIFEST} changed while it was read")
    manifest = json.loads(data)
    report = find_check_report(Path(check_root), manifest_sha256)
    if manifest.get("max_tokens", math.inf) > max_tokens:
        raise EpisodeSetError(
            f"the build keeps episodes up to {manifest.get('max_tokens')} tokens, above the "
            f"run's max_tokens {max_tokens}"
        )
    table = PermutationTable.from_strings(manifest["augmentation"]["permutations"])
    return EpisodeSet(folder, manifest, manifest_sha256, report, table)


def load_split(episode_set: EpisodeSet, split: str, max_tokens: int) -> list[EpisodeRecord]:
    """Read one split's records after checking its shards against the manifest.

    Args:
        episode_set: The verified build.
        split: ``train``, ``val``, or ``toy``.
        max_tokens: Longest episode allowed; a longer one raises (the notebook's cap, asserted,
            never cut).

    Returns:
        The records in shard order (the build's pool order).

    Raises:
        EpisodeSetError: If the split has no shards, a shard differs from the manifest, or the
            record count differs from it.
        RecordError: If a record breaks the schema or exceeds ``max_tokens``.
    """
    manifest = episode_set.manifest
    entries = [
        entry
        for entry in manifest["files"]
        if entry.get("split") == split and entry["path"].startswith("episodes/")
    ]
    if not entries:
        raise EpisodeSetError(f"the build has no {split} shards")
    paths = []
    for entry in entries:
        path = episode_set.folder / entry["path"]
        if not path.is_file():
            raise EpisodeSetError(f"{entry['path']} is listed in the manifest but missing")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise EpisodeSetError(f"{entry['path']} differs from the manifest")
        paths.append(path)
    records = list(
        read_shards(paths, n_augmentations=episode_set.table.n_augmentations, max_tokens=max_tokens)
    )
    expected = manifest.get("counts", {}).get(split, sum(e["records"] for e in entries))
    if len(records) != expected or any(record.split != split for record in records):
        raise EpisodeSetError(
            f"{split}: {len(records)} records in the shards, the manifest says {expected}"
        )
    return records


# --- sequences and the loss ----------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeSequence:
    """One episode as the base reads it.

    Attributes:
        episode_id: The record's id.
        source: ``arc2train``, ``rearc``, or ``barc``.
        family: The record's split unit (validation reports and the mismatch control use it).
        input_ids: Tokens of $a(D)$ then $a(Q)$, int64 on the CPU, ``[seq]``.
        target: $O_Q$, the positions of the query's output-grid tokens (and of the closing
            ``<|im_end|>`` with ``loss_includes_eos``).
        demo_hashes: ``pair_hash`` of each un-augmented demonstration, in record order.
    """

    episode_id: str
    source: str
    family: str
    input_ids: Tensor  # [seq]
    target: range
    demo_hashes: tuple[str, ...]


def augmented_pairs(record: EpisodeRecord, table: PermutationTable) -> list[dict[str, np.ndarray]]:
    """Apply the record's augmentation to every pair with the notebook's ``forward_mod``.

    Args:
        record: The episode.
        table: The build's permutation table.

    Returns:
        $a(D)$ then $a(Q)$, each ``{"input", "output"}`` as arrays.
    """
    descriptor = table.descriptor(record.augmentation_id)
    return [apply_view({"input": x, "output": y}, descriptor) for x, y in record.pairs]


def demo_hashes(record: EpisodeRecord) -> tuple[str, ...]:
    """Return the content addresses of the un-augmented demonstrations (the cache keys)."""
    return tuple(pair_hash({"input": x, "output": y}) for x, y in record.demos)


def build_sequence(
    record: EpisodeRecord,
    table: PermutationTable,
    formatter: QwenFormatter,
    max_tokens: int,
    include_eos: bool = True,
) -> EpisodeSequence:
    """Tokenize an episode in the notebook's format and locate the query's output tokens.

    Each augmented pair goes through P3.2's ``render_pair``, which checks the prefix property and
    that its output span decodes back to exactly its grid's text; the concatenation must equal the
    tokens of ``QwenFormatter.fmt_train`` over all pairs, the text the notebook trains on.

    Args:
        record: The episode.
        table: The build's permutation table.
        formatter: The notebook formatter around the base's tokenizer.
        max_tokens: The notebook's cap; P3.4 dropped longer episodes, so a longer one raises.
        include_eos: Whether $O_Q$ also holds the ``<|im_end|>`` that closes the query's grid
            (``TrainConfig.loss_includes_eos``).

    Returns:
        The sequence and $O_Q$: the grid's tokens, plus the closing token with ``include_eos``.

    Raises:
        ValueError: If a pair fails ``render_pair``, the tokenizer merges across pairs, the
            sequence exceeds ``max_tokens``, or with ``include_eos`` the grid is not followed by
            exactly one closing token.
    """
    pairs = augmented_pairs(record, table)
    rendered = [render_pair(pair, formatter) for pair in pairs]
    ids = [token for pair in rendered for token in pair.token_ids]
    if list(formatter.tokenizer.encode(formatter.fmt_train(pairs))) != ids:
        raise ValueError(
            f"{record.episode_id}: the tokens of the whole episode differ from its pairs'"
        )
    if len(ids) > max_tokens:
        raise ValueError(f"{record.episode_id}: {len(ids)} tokens, above max_tokens {max_tokens}")
    offset = len(ids) - len(rendered[-1].token_ids)
    span = rendered[-1].output_positions
    stop = span.stop
    if include_eos:
        # The pair ends "{grid}<|im_end|>": exactly one token after the grid, the last one.
        if span.stop != len(rendered[-1].token_ids) - 1:
            raise ValueError(f"{record.episode_id}: the query's grid is not closed by one token")
        stop += 1
    return EpisodeSequence(
        episode_id=record.episode_id,
        source=record.source,
        family=record.family,
        input_ids=torch.tensor(ids, dtype=torch.long),
        target=range(offset + span.start, offset + stop),
        demo_hashes=demo_hashes(record),
    )


def query_ce(logits: Tensor, input_ids: Tensor, target: range) -> Tensor:
    r"""Mean cross-entropy of the query's output-grid tokens, teacher forced, in fp32.

    The logit at position $t - 1$ predicts token $t$, so $O_Q$'s predictions are the rows
    ``target.start - 1`` to ``target.stop - 2``.

    Args:
        logits: Next-token logits of one sequence, ``[seq, vocab]``.
        input_ids: Its tokens, ``[seq]``.
        target: $O_Q$; ``target.start`` is at least 1.

    Returns:
        A scalar, $-\frac{1}{|O_Q|} \sum_{t \in O_Q} \log p(y_t \mid y_{<t})$.
    """
    predicted = logits[target.start - 1 : target.stop - 1].float()  # [|O_Q|, vocab]
    labels = input_ids[target.start : target.stop].to(predicted.device)  # [|O_Q|]
    return F.cross_entropy(predicted, labels)


def factor_penalty(factors: LoRAFactors, kind: str) -> Tensor:
    """Mean absolute value (``l1``) or mean square (``l2``) over every emitted factor entry.

    D2L penalizes the generated factors with L1, T2L with L2 (research note, sections 1.4 and 2);
    the mean keeps the weight independent of the scope's size.

    Args:
        factors: One adapter, ``(A [r, d_in], B [d_out, r])`` per key.
        kind: ``l1`` or ``l2``.

    Returns:
        A scalar in fp32.

    Raises:
        ValueError: If ``kind`` is neither.
    """
    if kind not in ("l1", "l2"):
        raise ValueError(f"penalty kind must be l1 or l2, got {kind!r}")
    total, count = None, 0
    for pair in factors.values():
        for tensor in pair:
            value = tensor.float().abs() if kind == "l1" else tensor.float().square()
            total = value.sum() if total is None else total + value.sum()
            count += tensor.numel()
    assert total is not None
    return total / count


def query_scores(logits: Tensor, input_ids: Tensor, target: range) -> tuple[Tensor, bool]:
    r"""The query CE (``query_ce``) and the teacher-forced greedy exact match, from one forward.

    The exact match holds when every argmax over $O_Q$'s prediction rows is the target token,
    which is when greedy decoding reproduces the target (ties aside: ``argmax`` takes the first).

    Args:
        logits: Next-token logits of one sequence, ``[seq, vocab]``.
        input_ids: Its tokens, ``[seq]``.
        target: $O_Q$; ``target.start`` is at least 1.

    Returns:
        The CE, a scalar exactly as ``query_ce`` computes it, and the match.
    """
    ce = query_ce(logits, input_ids, target)
    rows = logits[target.start - 1 : target.stop - 1]  # [|O_Q|, vocab]
    labels = input_ids[target.start : target.stop].to(rows.device)  # [|O_Q|]
    return ce, bool((rows.argmax(dim=-1) == labels).all())


@dataclass(frozen=True)
class DemoSequence:
    r"""The demonstrations of an episode under one view, $b(D)$, as an inner step reads them.

    Attributes:
        episode_id: The record's id.
        augmentation_id: $b$, an id of the build's ``PermutationTable``.
        input_ids: Tokens of $b(D)$, int64 on the CPU, ``[seq]``.
        targets: Each demonstration's output-grid positions (with its closing ``<|im_end|>``
            under ``loss_includes_eos``), the fit's completion-only labels.
    """

    episode_id: str
    augmentation_id: int
    input_ids: Tensor  # [seq]
    targets: tuple[range, ...]


def build_demo_sequence(
    record: EpisodeRecord,
    table: PermutationTable,
    formatter: QwenFormatter,
    augmentation_id: int,
    max_tokens: int,
    include_eos: bool = True,
) -> DemoSequence:
    """Tokenize the demonstrations under one augmentation, as ``build_sequence`` does a(D), a(Q).

    Under the record's own augmentation the tokens are the prefix of ``build_sequence``'s, up to
    the query.

    Args:
        record: The episode.
        table: The build's permutation table.
        formatter: The notebook formatter around the base's tokenizer.
        augmentation_id: The view $b$.
        max_tokens: The notebook's cap.
        include_eos: Whether each target holds the ``<|im_end|>`` closing its grid.

    Returns:
        The sequence and every demonstration's output positions.

    Raises:
        ValueError: If a pair fails ``render_pair``, the tokenizer merges across pairs, the
            sequence exceeds ``max_tokens``, or a grid is not closed by one token.
    """
    descriptor = table.descriptor(augmentation_id)
    pairs = [apply_view({"input": x, "output": y}, descriptor) for x, y in record.demos]
    rendered = [render_pair(pair, formatter) for pair in pairs]
    ids = [token for pair in rendered for token in pair.token_ids]
    if list(formatter.tokenizer.encode(formatter.fmt_train(pairs))) != ids:
        raise ValueError(f"{record.episode_id}: the demonstrations' tokens differ from the pairs'")
    if len(ids) > max_tokens:
        raise ValueError(f"{record.episode_id}: {len(ids)} demonstration tokens, over {max_tokens}")
    targets, offset = [], 0
    for pair in rendered:
        span, stop = pair.output_positions, pair.output_positions.stop
        if include_eos:
            if span.stop != len(pair.token_ids) - 1:
                raise ValueError(f"{record.episode_id}: a demonstration's grid is not closed")
            stop += 1
        targets.append(range(offset + span.start, offset + stop))
        offset += len(pair.token_ids)
    return DemoSequence(
        record.episode_id, augmentation_id, torch.tensor(ids, dtype=torch.long), tuple(targets)
    )


def demo_ce(logits: Tensor, input_ids: Tensor, targets: Sequence[range]) -> Tensor:
    r"""Mean CE over every demonstration's output tokens, teacher forced, in fp32.

    The notebook's fit loss on one view: its completion-only collator labels each reply's grid
    and end token, and the mean runs over all labeled tokens of the sequence.

    Args:
        logits: Next-token logits of $b(D)$, ``[seq, vocab]``.
        input_ids: Its tokens, ``[seq]``.
        targets: ``DemoSequence.targets``.

    Returns:
        A scalar, $-\frac{1}{\sum_i |O_i|} \sum_i \sum_{t \in O_i} \log p(y_t \mid y_{<t})$.
    """
    positions = torch.tensor([t for target in targets for t in target], dtype=torch.long)
    predicted = logits[(positions - 1).to(logits.device)].float()  # [n, vocab]
    labels = input_ids[positions].to(predicted.device)  # [n]
    return F.cross_entropy(predicted, labels)


def inner_view_ids(
    seed: int, episode_id: str, own: int, steps: int, n_augmentations: int, views: str
) -> list[int]:
    """The augmentation of each inner step: the episode's own, or a seeded draw per episode.

    ``drawn`` shuffles the build's ids with ``rng_for(seed, "inner-view", episode_id)`` and takes
    them in order (cycling past ``n_augmentations``), so step $j$'s view is a pure function of the
    seed and the episode, the same in training, validation, a resume, and any world size.

    Args:
        seed: The run seed.
        episode_id: The record's id.
        own: The record's augmentation id, $a$.
        steps: $k$.
        n_augmentations: Ids of the build's ``PermutationTable``.
        views: ``episode`` or ``drawn``.

    Returns:
        $k$ augmentation ids.
    """
    if views == "episode":
        return [own] * steps
    order = list(range(n_augmentations))
    rng_for(seed, "inner-view", episode_id).shuffle(order)
    return [order[j % n_augmentations] for j in range(steps)]


class InnerOptimizer:
    r"""The inner loop's update, in place on the episode's copy of the factors (``InnerLoop``).

    ``adam``'s moments start at zero for every episode and are allocated only when a later step
    reads them ($k \ge 2$): the last step's moments are temporaries, so at $k = 1$ the loop holds
    the factor copy alone.

    Args:
        settings: The inner loop.
    """

    def __init__(self, settings: InnerLoop) -> None:
        self.settings = settings
        self.state: list[tuple[Tensor, Tensor]] | None = None

    @torch.no_grad()
    def step(self, params: Sequence[Tensor], grads: Sequence[Tensor], j: int) -> None:
        r"""Apply $\theta \leftarrow \theta - \alpha\,u_j$ to every tensor, in place.

        Args:
            params: The factor copies (leaves), each ``[r, d_in]`` or ``[d_out, r]``.
            grads: Their gradients $g_j$, same shapes.
            j: The step's index, from 0.
        """
        cfg = self.settings
        if cfg.optimizer == "sgd":
            for param, grad in zip(params, grads, strict=True):
                param.sub_(grad, alpha=cfg.lr)
            return
        beta1, beta2 = cfg.betas
        keep = j < cfg.steps - 1
        if self.state is None and keep:
            self.state = [(torch.zeros_like(p), torch.zeros_like(p)) for p in params]
        c1, c2 = 1 - beta1 ** (j + 1), 1 - beta2 ** (j + 1)
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if self.state is None:  # the first step is the last: m = (1 - b1) g, v = (1 - b2) g^2
                m, v = grad * (1 - beta1), grad.square() * (1 - beta2)
            else:
                m, v = self.state[index]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (v / c2).sqrt_().add_(cfg.eps)
            param.addcdiv_(m, denom, value=-cfg.lr / c1)


def flat_factors(factors: LoRAFactors) -> list[Tensor]:
    """The factors as one list, ``A`` then ``B`` of each key in key order."""
    return [tensor for pair in factors.values() for tensor in pair]


@torch.no_grad()
def clip_inner_grads(grads: Sequence[Tensor], max_norm: float | None) -> float:
    r"""Clip one inner step's gradients in place by their global norm, as the fit clips its own.

    torch's ``clip_grad_norm_`` formula, which the notebook's Trainer applies with
    ``max_grad_norm`` 1.0: $g \leftarrow \min(1, \tau / (\lVert g\rVert_2 + 10^{-6}))\,g$, the
    norm over every tensor at once.

    Args:
        grads: The step's gradients, each ``[r, d_in]`` or ``[d_out, r]``.
        max_norm: $\tau$; None leaves them unclipped.

    Returns:
        $\lVert g\rVert_2$ before clipping.
    """
    norms = torch.stack([torch.linalg.vector_norm(g) for g in grads])  # [n_tensors]
    total = torch.linalg.vector_norm(norms)
    if max_norm is not None:
        factor = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        for grad in grads:
            grad.mul_(factor)
    return float(total)


def adapt(
    model: nn.Module,
    factors: LoRAFactors,
    demos: Sequence[DemoSequence],
    settings: InnerLoop,
    scaling: float,
    device: torch.device,
    timing: TimingLogger | None = None,
    op: str = "inner_step",
) -> tuple[LoRAFactors, list[float], list[float]]:
    r"""Run the inner loop from $\theta_0$ = ``factors`` and return $\theta_k$, a fresh copy.

    $\theta_0$ is detached and copied (the one extra factor copy), so the loop never reaches H's
    graph, and the base is frozen, so its gradients are the copy's alone
    (``torch.autograd.grad``, nothing accumulates in any ``.grad``). Forward and backward of each
    step run inside one ``inject`` block, so activation checkpointing recomputes under the
    factors the forward used (P3.1, risk 2); the clip (``clip_inner_grads``) and the update run
    after the block.

    Args:
        model: The base prepared by ``wrap_targets``, in training mode when checkpointing.
        factors: $\theta_0$, ``(A [r, d_in], B [d_out, r])`` per key.
        demos: One demonstration sequence per inner step, $k$ = ``len(demos)``.
        settings: The inner loop.
        scaling: The LoRA scale $s$.
        device: The model's device.
        timing: Receives one record per step, ``{puzzle_id, op, seconds, tokens}``.
        op: The records' op: ``inner_step`` in training, ``validate_inner_step`` in validation.

    Returns:
        $\theta_k$ (leaf tensors that require grad, fp32 as H emits them), the demonstrations'
        CE at each $\theta_j$, $j < k$, and each step's gradient norm before the clip.
    """
    leaves = {
        key: (a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True))
        for key, (a, b) in factors.items()
    }
    params = flat_factors(leaves)
    optimizer = InnerOptimizer(settings)
    losses, norms = [], []
    for j, demo in enumerate(demos):
        ids = demo.input_ids.to(device)  # [seq]
        timed = (
            timing.timed(demo.episode_id, op, int(ids.numel()))
            if timing is not None
            else nullcontext()
        )
        with torch.enable_grad(), timed, inject(model, leaves, scaling):
            logits = model(input_ids=ids[None], use_cache=False).logits[0]
            loss = demo_ce(logits, ids, demo.targets)
            grads = torch.autograd.grad(loss, params)
            losses.append(float(loss.item()))
        norms.append(clip_inner_grads(grads, settings.clip))
        optimizer.step(params, grads, j)
        del grads
    return leaves, losses, norms


def choose_inner_lr(candidates: Sequence[Mapping[str, float]], margin: int) -> int:
    r"""The pre-registered rule of ``InnerProbe``: the rate to train at, from its post-$k$ scores.

    Among the candidates with a finite mean CE and mean $S$, those within ``margin`` episodes of
    the most post-$k$ greedy exact matches tie on the match, and the lowest mean $S$ among them
    wins; an exact tie keeps the earlier (smaller) rate.

    Args:
        candidates: In ascending rate order, each with ``exact`` (matching episodes),
            ``nll_sum`` (mean $S$), and ``ce`` (mean CE).
        margin: $m$, episodes.

    Returns:
        The winner's index.

    Raises:
        ValueError: If no candidate has finite scores.
    """
    finite = [
        i
        for i, c in enumerate(candidates)
        if math.isfinite(c["nll_sum"]) and math.isfinite(c["ce"])
    ]
    if not finite:
        raise ValueError("inner_probe: no inner rate gives finite post-k scores")
    top = max(candidates[i]["exact"] for i in finite)
    tied = [i for i in finite if candidates[i]["exact"] >= top - margin]
    return min(tied, key=lambda i: (candidates[i]["nll_sum"], i))


def h_config(cfg: TrainConfig) -> dict[str, Any]:
    """The config ``best.pt`` carries: ``to_dict`` without the ``OBJECTIVE_KEYS``.

    Why: they describe how H was trained, not how to build it, its encoder, or its validation;
    without them a ``fomaml`` H's ``best.pt`` holds the config keys of the code before P3.9, so a
    loader on that code (P3.7's load check parses the config strictly with
    ``TrainConfig.from_dict``) reads it as it reads ``MT-03``'s. ``best.pt`` keeps them in
    ``objective``.

    Args:
        cfg: The run's config.

    Returns:
        Plain values.
    """
    return {k: v for k, v in cfg.to_dict().items() if k not in OBJECTIVE_KEYS}


def _gram_norm(b: Tensor, a: Tensor) -> Tensor:
    r"""$\lVert BA\rVert_F = \sqrt{\mathrm{tr}((B^\top B)(AA^\top))}$ from Gram matrices, fp64."""
    b, a = b.detach().double(), a.detach().double()
    return ((b.T @ b) * (a @ a.T)).sum().clamp_min(0).sqrt()


def displacement(start: LoRAFactors, end: LoRAFactors, scaling: float) -> dict[str, float]:
    r"""Per projection, $\lVert s(B_kA_k - B_0A_0)\rVert_F$, the inner loop's move of the delta.

    $B_kA_k - B_0A_0 = [B_k, -B_0]\,[A_k; A_0]$, so the norm is ``_gram_norm`` of the stacked
    factors: Gram matrices of side $2r$, never the $d_\text{out} \times d_\text{in}$ delta.

    Args:
        start: $\theta_0$, ``(A [r, d_in], B [d_out, r])`` per key.
        end: $\theta_k$, same keys.
        scaling: $s$.

    Returns:
        Key to the norm.
    """
    out = {}
    for key, (a0, b0) in start.items():
        ak, bk = end[key]
        stacked_b = torch.cat((bk.detach(), -b0.detach()), dim=1)  # [d_out, 2r]
        stacked_a = torch.cat((ak.detach(), a0.detach()), dim=0)  # [2r, d_in]
        out[key] = scaling * float(_gram_norm(stacked_b, stacked_a))
    return out


def delta_norms(factors: LoRAFactors, scaling: float) -> dict[str, float]:
    r"""Per projection, $\lVert sBA\rVert_F$, the size of an adapter's delta (survey 9.3).

    Args:
        factors: ``(A [r, d_in], B [d_out, r])`` per key.
        scaling: $s$.

    Returns:
        Key to the norm.
    """
    return {key: scaling * float(_gram_norm(b, a)) for key, (a, b) in factors.items()}


def by_module(values: Mapping[str, float]) -> dict[str, float]:
    """Mean over layers of per-projection values, by module type (``q_proj`` and so on)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for key, value in values.items():
        groups[key.rpartition(".")[2]].append(value)
    return {name: sum(v) / len(v) for name, v in sorted(groups.items())}


def decode_metrics(ces: Tensor, lengths: Tensor, exact: Tensor) -> dict[str, float]:
    r"""Decode-aligned summaries of one reference over the validation episodes (survey 11a, 2).

    $S = |O_Q|\,\bar\ell$ is the summed NLL of the target under teacher forcing; the decode keeps
    a candidate under a view only while $S < \ln 5$.

    Args:
        ces: Per-episode mean CE $\bar\ell$, ``[n]``.
        lengths: $|O_Q|$ per episode, ``[n]``.
        exact: Greedy exact match per episode, 0 or 1, ``[n]``.

    Returns:
        ``tokens`` (mean $|O_Q|$), ``nll_sum`` (mean $S$), ``under_bound`` (share with
        $S < \ln 5$), ``exact_match`` (share), ``geomean_ce``
        ($\exp \overline{\ln \max(\bar\ell, 10^{-6})}$).
    """
    ces, lengths, exact = ces.double(), lengths.double(), exact.double()
    sums = ces * lengths
    return {
        "tokens": float(lengths.mean()),
        "nll_sum": float(sums.mean()),
        "under_bound": float((sums < DECODE_BOUND).double().mean()),
        "exact_match": float(exact.mean()),
        "geomean_ce": float(ces.clamp_min(GEOMEAN_FLOOR).log().mean().exp()),
    }


# --- embeddings ----------------------------------------------------------------------------------


def demo_pairs(records: Iterable[EpisodeRecord]) -> dict[str, ArcPair]:
    """Collect every distinct un-augmented demonstration pair, keyed by ``pair_hash``.

    Args:
        records: Episodes.

    Returns:
        Hash to ``{"input", "output"}``, first occurrence kept.
    """
    pairs: dict[str, ArcPair] = {}
    for record in records:
        for x, y in record.demos:
            pair = {"input": x, "output": y}
            pairs.setdefault(pair_hash(pair), pair)
    return pairs


class _Comm:
    """The collectives the run uses; identity operations in a single process."""

    def __init__(self, world: int, device: torch.device) -> None:
        self.world = world
        # NCCL moves CUDA tensors only; gloo moves CPU tensors.
        self.device = device if device.type == "cuda" else torch.device("cpu")

    def all_gather(self, tensor: Tensor) -> list[Tensor]:
        """Gather an equally shaped tensor from every rank, as CPU tensors in rank order."""
        if self.world == 1:
            return [tensor.cpu()]
        local = tensor.to(self.device).contiguous()
        parts = [torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(parts, local)
        return [part.cpu() for part in parts]

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        """Sum a tensor over the ranks; returned on the CPU."""
        if self.world == 1:
            return tensor.cpu()
        local = tensor.to(self.device).contiguous()
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local.cpu()

    def broadcast_int(self, value: int) -> int:
        """Return rank 0's ``value`` on every rank."""
        if self.world == 1:
            return value
        box = torch.tensor([value], dtype=torch.int64, device=self.device)
        dist.broadcast(box, src=0)
        return int(box.item())

    def all_gather_object(self, value: Any) -> list[Any]:
        """Gather a picklable object from every rank, in rank order."""
        if self.world == 1:
            return [value]
        out: list[Any] = [None] * self.world
        dist.all_gather_object(out, value)
        return out


class PairEmbeddings:
    r"""P3.2's pair embeddings $\bar e_i$ by pair hash, for the $\bar z$ lookups of a run.

    Attributes:
        cache_hits: Pairs read from the ``EmbeddingCache``.
        encoded: Pairs encoded in this process's run (all ranks' shares).
        encode_tokens: Real tokens this rank prefilled.
    """

    def __init__(self) -> None:
        self._memo: dict[str, Tensor] = {}
        self.cache_hits = 0
        self.encoded = 0
        self.encode_tokens = 0

    def __len__(self) -> int:
        return len(self._memo)

    def __contains__(self, key: object) -> bool:
        return key in self._memo

    def fill(
        self,
        pairs: Mapping[str, ArcPair],
        encoder: PairEncoder,
        cache: EmbeddingCache | None,
        comm: _Comm,
        rank: int,
        chunk_pairs: int,
        timing: TimingLogger | None = None,
    ) -> None:
        r"""Make every pair available: cache hits first, then one sharded encode pass.

        Chunk $j$ of the sorted missing hashes is encoded by rank $j \bmod W$ and all-gathered
        each round, so every rank ends with the same values and each pair's prefill batch is fixed
        by its chunk alone. Every rank must call this with the same pairs.

        Args:
            pairs: Hash to un-augmented pair.
            encoder: P3.2's encoder on the run's base (eval mode, nothing injected).
            cache: Cache of the encoder's namespace, or None.
            comm: The run's collectives.
            rank: This process's rank.
            chunk_pairs: Pairs per encode call.
            timing: Receives one ``encode`` record per chunk this rank encodes.
        """
        todo = sorted(key for key in pairs if key not in self._memo)
        if cache is not None:
            hits = [key for key in todo if key in cache]
            if hits:
                for key, row in zip(hits, cache.get(hits), strict=True):
                    self._memo[key] = row
                self.cache_hits += len(hits)
            todo = [key for key in todo if key not in self._memo]
        chunks = [todo[i : i + chunk_pairs] for i in range(0, len(todo), chunk_pairs)]
        width = encoder.hidden_size
        for start in range(0, len(chunks), comm.world):
            mine = start + rank
            block = torch.zeros(chunk_pairs, width, dtype=torch.float32)
            if mine < len(chunks):
                tokens: list[int] = []

                def count(record: dict[str, Any], tokens: list[int] = tokens) -> None:
                    if record["op"] == "encode_prefill":
                        tokens.append(int(record["tokens"]))

                encoder.timing = count
                began = time.perf_counter()
                rows = encoder.encode_pairs([pairs[key] for key in chunks[mine]])  # [n, hidden]
                seconds = time.perf_counter() - began
                encoder.timing = None
                block[: len(rows)] = rows
                self.encode_tokens += sum(tokens)
                if timing is not None:
                    timing.record(None, "encode", seconds, sum(tokens))
            for k, part in enumerate(comm.all_gather(block)):
                if start + k < len(chunks):
                    chunk = chunks[start + k]
                    for key, row in zip(chunk, part[: len(chunk)], strict=True):
                        self._memo[key] = row.clone()
                    self.encoded += len(chunk)

    def z_bar(self, hashes: Sequence[str]) -> Tensor:
        r"""Return $\bar z$, the mean of the pair embeddings, in fp32 on the CPU, ``[hidden]``.

        Args:
            hashes: ``pair_hash`` of each demonstration.

        Raises:
            KeyError: If a pair was never filled.
            ValueError: If ``hashes`` is empty.
        """
        if not hashes:
            raise ValueError("z_bar needs at least one demonstration")
        return torch.stack([self._memo[key] for key in hashes]).mean(dim=0)

    def pair_set(self, hashes: Sequence[str]) -> Tensor:
        r"""Return the pair embeddings $\bar e_i$ themselves, fp32 on the CPU, ``[n, hidden]``.

        The input of a ``set`` H beside $\bar z$ (its mean): rows in ``hashes`` order, repeats
        kept, as ``z_bar`` counts them.

        Args:
            hashes: ``pair_hash`` of each demonstration.

        Raises:
            KeyError: If a pair was never filled.
            ValueError: If ``hashes`` is empty.
        """
        if not hashes:
            raise ValueError("pair_set needs at least one demonstration")
        return torch.stack([self._memo[key] for key in hashes])


# --- data order, controls, optimizer -------------------------------------------------------------


def train_slot(step: int, micro: int, rank: int, accum: int, world: int) -> int:
    r"""Return the global slot of an episode draw: $g = (s \cdot A + m) \cdot W + k$.

    The slots of one optimizer step are contiguous, so ``world`` processes with accumulation
    ``accum`` see the same episodes as any other split of ``world * accum``.

    Args:
        step: Optimizer steps completed before this one ($s$).
        micro: Micro step within the optimizer step ($m$).
        rank: Process rank ($k$).
        accum: Micro steps per optimizer step ($A$).
        world: Processes ($W$).

    Returns:
        The slot.
    """
    return (step * accum + micro) * world + rank


def block_counts(shares: Sequence[float], block: int) -> list[int]:
    """Split ``block`` draws by ``shares`` with the largest-remainder rule (ties to the first).

    Args:
        shares: Non-negative shares that sum to 1.
        block: Draws to split.

    Returns:
        Whole counts that sum to ``block``, each within 1 of its share times ``block``.
    """
    raw = [share * block for share in shares]
    counts = [math.floor(value) for value in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: counts[i] - raw[i])
    for i in by_remainder[: block - sum(counts)]:
        counts[i] += 1
    return counts


class EpisodeOrder:
    r"""The global episode order, a pure function of the seed and the slot.

    Without source weights, pass $e$ is a seeded permutation of the pool: slot $g$ falls in pass
    $e = \lfloor g / N \rfloor$ at position $g \bmod N$, the permutation drawn from
    ``rng_for(seed, "train-order", e)``.

    With source weights $w_s$, slots form blocks of $B$ = ``SOURCE_BLOCK``. Block $b$ holds $c_s$
    slots of source $s$, $c_s$ the largest-remainder rounding of $B w_s / \sum_t w_t$
    (``block_counts``), in the order of a shuffle drawn from ``rng_for(seed, "train-source",
    b)``. The slot that is the $k$-th draw of source $s$, $k = b c_s + $ its rank among block
    $b$'s slots of $s$, takes position $k \bmod N_s$ of pass $\lfloor k / N_s \rfloor$ over that
    source's $N_s$ episodes, a permutation drawn from ``rng_for(seed, "train-order", s, pass)``.
    Every block then has exactly the rounded shares, each source shows every episode of its own
    once per pass over that source, and the episode of any slot is again a pure function, so a
    resumed run sees the same sequence.

    Args:
        size: Episodes in the pool, $N$.
        seed: The run seed.
        sources: The source of each pool episode, in pool order; required with ``weights``.
        weights: Draw share of each source (any positive scale), naming exactly the pool's
            sources; None draws every episode once per pass.

    Raises:
        ValueError: If the pool is empty, or with ``weights``, if ``sources`` does not cover the
            pool or ``weights`` does not name exactly its sources.
    """

    def __init__(
        self,
        size: int,
        seed: int,
        sources: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("the training split is empty")
        self.size = size
        self.seed = seed
        self.weights = None if weights is None else dict(weights)
        self._epoch = -1
        self._perm: list[int] = []
        if weights is None:
            return
        if sources is None or len(sources) != size:
            raise ValueError("a source-weighted order needs the source of every pool episode")
        present = sorted(set(sources))
        if sorted(weights) != present:
            raise ValueError(
                f"source_weights names {sorted(weights)}, the training split holds {present}"
            )
        total = sum(weights.values())
        self.names = present
        shares = [weights[s] / total for s in present]
        self.counts = dict(zip(present, block_counts(shares, SOURCE_BLOCK), strict=True))
        self.pools = {s: [i for i, source in enumerate(sources) if source == s] for s in present}
        self._block = -1
        self._block_names: list[str] = []
        self._block_ranks: list[int] = []
        self._source_perm: dict[str, tuple[int, list[int]]] = {}

    def __call__(self, slot: int) -> int:
        """Return the pool index of slot ``slot``."""
        if self.weights is None:
            epoch, position = divmod(slot, self.size)
            if epoch != self._epoch:
                perm = list(range(self.size))
                rng_for(self.seed, "train-order", epoch).shuffle(perm)
                self._epoch, self._perm = epoch, perm
            return self._perm[position]
        block, position = divmod(slot, SOURCE_BLOCK)
        if block != self._block:
            names = [name for name in self.names for _ in range(self.counts[name])]
            rng_for(self.seed, "train-source", block).shuffle(names)
            seen = dict.fromkeys(self.names, 0)
            ranks = []
            for name in names:
                ranks.append(seen[name])
                seen[name] += 1
            self._block, self._block_names, self._block_ranks = block, names, ranks
        name = self._block_names[position]
        draw = block * self.counts[name] + self._block_ranks[position]
        pool = self.pools[name]
        epoch, index = divmod(draw, len(pool))
        cached = self._source_perm.get(name)
        if cached is None or cached[0] != epoch:
            perm = list(range(len(pool)))
            rng_for(self.seed, "train-order", name, epoch).shuffle(perm)
            cached = (epoch, perm)
            self._source_perm[name] = cached
        return pool[cached[1][index]]

    def source_draws(self, slots: int) -> dict[str, int] | None:
        """Draws of each source among slots ``0`` to ``slots - 1`` of a weighted order.

        Args:
            slots: Slots counted from the start.

        Returns:
            Draws by source, exact; None for an unweighted order (each episode once per pass).
        """
        if self.weights is None:
            return None
        full, rest = divmod(slots, SOURCE_BLOCK)
        draws = {name: full * count for name, count in self.counts.items()}
        if rest:
            self(full * SOURCE_BLOCK)  # builds the partial block's order
            for name in self._block_names[:rest]:
                draws[name] += 1
        return draws


def mismatch_permutation(families: Sequence[str], seed: int) -> list[int]:
    r"""Pair every validation episode with one of another family, as a fixed permutation.

    Episodes are listed family by family (families in seeded order) and episode $p$ of the list
    takes the $\bar z$ of episode $p + M$ (mod $n$), where $M$ is the largest family: two
    positions $M$ apart never share a family block when $n \ge 2M$. Another family, not only
    another puzzle, because a ReARC family's puzzles share one transformation.

    Args:
        families: Family of each validation episode.
        seed: The run seed.

    Returns:
        ``perm`` with ``families[perm[i]] != families[i]`` for every ``i``; a bijection.

    Raises:
        ValueError: If one family holds more than half the episodes.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        groups[family].append(index)
    order = [i for family in keyed_order(groups, seed, "mismatch") for i in groups[family]]
    n, largest = len(order), max((len(g) for g in groups.values()), default=0)
    if n == 0 or n < 2 * largest:
        raise ValueError(
            f"the mismatched control needs every family below half the episodes; "
            f"{n} episodes, the largest family {largest}"
        )
    perm = [0] * n
    for position, index in enumerate(order):
        perm[index] = order[(position + largest) % n]
    return perm


def param_groups(
    hypernet: nn.Module,
    weight_decay: float,
    new: Sequence[str] = (),
    new_lr: float | None = None,
) -> list[dict[str, Any]]:
    """Split H's parameters for AdamW: decay on 2-D weights only.

    No decay on norms, biases (the head biases hold $A_0$, bias-HyperInit), the layer and
    module codes (D2L, research note 1.4), and the set aggregator's queries (codes as well).
    A warm start (``--init-from``) puts the parameters it adds in two groups of their own, after
    the two of the checkpoint, so these keep their parameter order and AdamW state.

    Args:
        hypernet: H.
        weight_decay: Decay of the decayed groups.
        new: Names of the parameters a warm start adds; empty for every other run.
        new_lr: Their peak learning rate, required when ``new`` is not empty.

    Returns:
        Two parameter groups, or four with ``new`` (the new groups carry ``lr``).

    Raises:
        ValueError: If ``new`` names a parameter H lacks, or ``new_lr`` is missing.
    """
    added = set(new)
    unknown = added - {name for name, _ in hypernet.named_parameters()}
    if unknown:
        raise ValueError(f"param_groups: H has no parameters {sorted(unknown)}")
    if added and new_lr is None:
        raise ValueError("param_groups: new parameters need new_lr")
    decay, keep, new_decay, new_keep = [], [], [], []
    for name, param in hypernet.named_parameters():
        codes = name.startswith(("layer_embedding.", "module_embedding.", "aggregator.queries"))
        decayed = param.ndim >= 2 and not codes
        if name in added:
            (new_decay if decayed else new_keep).append(param)
        else:
            (decay if decayed else keep).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": keep, "weight_decay": 0.0},
    ]
    if added:
        groups += [
            {"params": new_decay, "weight_decay": weight_decay, "lr": new_lr},
            {"params": new_keep, "weight_decay": 0.0, "lr": new_lr},
        ]
    return groups


class WarmupCosine:
    r"""Learning-rate multiplier: linear warmup, then a cosine to ``min_ratio``.

    For optimizer step $n = s + 1$ (``LambdaLR`` passes $s$, the steps taken):
    $m(n) = n / W$ for $n \le W$, else
    $m(n) = \rho + (1 - \rho) \cdot \tfrac12 (1 + \cos(\pi (n - W) / (T - W)))$, so the first
    step already learns and the last uses ``min_lr``. A class, not a closure, so ``LambdaLR``'s
    state dict holds its settings.

    Args:
        warmup: $W$, warmup steps.
        total: $T$, optimizer steps.
        min_ratio: $\rho$, ``min_lr / lr``.
    """

    def __init__(self, warmup: int, total: int, min_ratio: float) -> None:
        self.warmup = warmup
        self.total = total
        self.min_ratio = min_ratio

    def __call__(self, steps_taken: int) -> float:
        """Return the multiplier of the next step."""
        n = steps_taken + 1
        if n <= self.warmup:
            return n / self.warmup
        progress = min(1.0, (n - self.warmup) / max(1, self.total - self.warmup))
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class NewParamsSchedule:
    r"""Learning-rate multiplier of the parameters a warm start adds, from its start step $s_0$.

    For optimizer step $n = s + 1 - s_0$ of the warm-started run ($s$ the steps taken, as
    ``LambdaLR`` counts them over the whole run): $m(n) = 0$ for $n \le 0$, $n / W$ for
    $n \le W$, else a cosine from 1 to $\rho$ over the steps left to $T$. The loaded groups keep
    ``WarmupCosine``: they continue the checkpoint's schedule, which by the warm start's step may
    be too low to move parameters that start at zero (scoping note, section 1, risks).

    Args:
        start: $s_0$, the checkpoint's step.
        warmup: $W$, warmup steps.
        total: $T$, the run's optimizer steps.
        min_ratio: $\rho$, ``min_lr / new_params_lr``.
    """

    def __init__(self, start: int, warmup: int, total: int, min_ratio: float) -> None:
        self.start = start
        self.warmup = warmup
        self.total = total
        self.min_ratio = min_ratio

    def __call__(self, steps_taken: int) -> float:
        """Return the multiplier of the next step."""
        n = steps_taken + 1 - self.start
        if n <= 0:
            return 0.0
        if n <= self.warmup:
            return n / self.warmup
        span = max(1, self.total - self.start - self.warmup)
        progress = min(1.0, (n - self.warmup) / span)
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# --- the base model ------------------------------------------------------------------------------


def standin_tokenizer() -> PreTrainedTokenizerFast:
    """Build the stand-in tokenizer of the smoke path and the tests; nothing is downloaded.

    A word-level vocabulary with the notebook's 16 ids (``STANDIN_VOCAB``): the chat markers are
    special tokens, the rest splits into single digits, newlines (byte-level ``Ċ``), and the role
    words, as ``scripts/cache_embeddings.py``'s stand-in does.

    Returns:
        A fast tokenizer with ``<|endoftext|>`` as padding and ``<|im_end|>`` as end of sequence.
    """
    tok = Tokenizer(models.WordLevel(vocab=STANDIN_VOCAB, unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\d|\n|[a-z]+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )


def tiny_base(seed: int = 0, attn_implementation: str = "sdpa") -> Qwen3ForCausalLM:
    """Build the tiny random Qwen3 of the smoke path and the tests, frozen, in eval mode.

    Hidden size 64, 4 layers, 4 query heads, 2 key-value heads of width 16, vocabulary 16: the
    shape relations of Qwen3-4B (GQA, explicit ``head_dim``). ``initializer_range`` 0.2 gives
    logits of order 1, so an adapter's effect on the loss shows within a few steps. The global
    torch generator is left as it was.

    Args:
        seed: Seed of the random weights.
        attn_implementation: Attention kernel name.

    Returns:
        The model, fp32 on the CPU, gradients off.
    """
    config = Qwen3Config(
        vocab_size=len(STANDIN_VOCAB),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=8192,
        initializer_range=0.2,
        attn_implementation=attn_implementation,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config)
    return model.eval().requires_grad_(False)


def resolve_attention(requested: str, device: torch.device) -> str:
    """Choose the attention kernel: FlashAttention 2 when available on a bf16 GPU, else SDPA.

    Args:
        requested: A value of ``ATTENTION``.
        device: The process's device.

    Returns:
        ``requested`` unless it is ``auto``.
    """
    if requested != "auto":
        return requested
    if (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
        and importlib.util.find_spec("flash_attn") is not None
    ):
        return "flash_attention_2"
    return "sdpa"


def load_base(
    cfg: TrainConfig, device: torch.device
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Load the frozen base and its tokenizer on ``device``.

    Args:
        cfg: The run config; ``model_dir`` None selects ``tiny_base`` and ``standin_tokenizer``.
        device: Target device.

    Returns:
        The model (frozen, eval mode, ``use_cache`` off), the tokenizer, and the attention
        kernel in use.

    Raises:
        ValueError: If the model has attention dropout: training mode, which transformers'
            checkpointing needs, must compute what evaluation mode computes.
    """
    attn = resolve_attention(cfg.attn_implementation, device)
    dtype = DTYPES[cfg.dtype]
    if cfg.model_dir is None:
        model: PreTrainedModel = tiny_base(cfg.seed, attn).to(device=device, dtype=dtype)
        tokenizer: PreTrainedTokenizerBase = standin_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, local_files_only=True)
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": True}
        if device.type == "cuda":
            kwargs["device_map"] = {"": device}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_dir, attn_implementation=attn, **kwargs
            )
        except (ImportError, ValueError):
            if cfg.attn_implementation != "auto" or attn == "sdpa":
                raise
            attn = "sdpa"  # flash_attn present but unusable with this build: fall back, recorded
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_dir, attn_implementation=attn, **kwargs
            )
    # Device only: a dtype cast would also round the rotary inv_freq buffer, which transformers
    # keeps in fp32 under torch_dtype, as the notebook's loader does.
    model = model.to(device).eval().requires_grad_(False)
    model.config.use_cache = False
    if getattr(model.config, "attention_dropout", 0.0):
        raise ValueError("the base has attention dropout; training mode would differ from eval")
    return model, tokenizer, attn


# --- checkpoints ---------------------------------------------------------------------------------


def checkpoint_name(step: int) -> str:
    """File name of the checkpoint after ``step`` optimizer steps, ``ckpt-000200.pt``."""
    return f"ckpt-{step:06d}.pt"


def _fsync_dir(path: Path) -> None:
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` so that readers see the old file or the new one, never part of it.

    Args:
        path: Destination; its folder must exist.
        data: Content.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)


def save_checkpoint(ckpt_dir: Path, payload: dict[str, Any], keep: int) -> Path:
    """Write a checkpoint atomically, point ``latest.json`` at it, and prune old ones.

    The file is written under a temporary name and renamed; ``latest.json`` is replaced after it,
    so a killed session leaves the previous checkpoint as the latest.

    Args:
        ckpt_dir: Checkpoint folder; created if missing.
        payload: The checkpoint, with ``step``.
        keep: Newest ``ckpt-*.pt`` files kept.

    Returns:
        The checkpoint's path.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / checkpoint_name(payload["step"])
    _atomic_torch_save(path, payload)
    pointer = {"file": path.name, "step": payload["step"]}
    atomic_write_bytes(ckpt_dir / LATEST_FILE, (json.dumps(pointer) + "\n").encode("utf-8"))
    for old in sorted(ckpt_dir.glob("ckpt-*.pt"))[:-keep]:
        if old != path:
            old.unlink(missing_ok=True)
    return path


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """``torch.save`` to a temporary name, flushed to disk, then renamed over ``path``."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_best(ckpt_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the weights-only checkpoint of the best validation point, atomically.

    Why: H is taken from its best validation point (``P3.5-02``), which the two newest full
    checkpoints kept for resuming have usually passed; a full checkpoint holds AdamW's two
    moments beside H's weights (three times H's size), so keeping every validation point's would
    outgrow Kaggle's working output. ``best.pt`` holds H's weights, the config, and the
    validation record, which is what P3.7 and P4.1 load (``load_checkpoint`` reads it); the name
    never matches ``ckpt-*.pt``, so pruning and ``resolve_checkpoint`` on a folder ignore it.

    Args:
        ckpt_dir: Checkpoint folder; created if missing.
        payload: ``schema``, ``kind`` ``best``, ``step``, ``config``, ``hypernet``, and the
            validation record.

    Returns:
        The file's path.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / BEST_FILE
    _atomic_torch_save(path, payload)
    return path


def resolve_checkpoint(path: Path) -> Path:
    """Return the checkpoint file a ``--resume`` argument names.

    Args:
        path: A checkpoint file, or a folder with ``latest.json`` (or ``ckpt-*.pt`` files).

    Returns:
        The file.

    Raises:
        FileNotFoundError: If there is no checkpoint.
    """
    path = Path(path)
    if path.is_file():
        return path
    pointer = path / LATEST_FILE
    if pointer.is_file():
        target = path / json.loads(pointer.read_text(encoding="utf-8"))["file"]
        if target.is_file():
            return target
    found = sorted(path.glob("ckpt-*.pt"))
    if not found:
        raise FileNotFoundError(f"no checkpoint at {path}")
    return found[-1]


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a checkpoint written by ``save_checkpoint`` or ``save_best`` (our own files).

    A ``save_best`` file has ``kind`` ``best`` and holds no optimizer state, so it can be
    evaluated or deployed but not resumed.

    Args:
        path: The file.

    Returns:
        The payload, tensors on the CPU.

    Raises:
        ValueError: If the schema is not ``CHECKPOINT_SCHEMA``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path}: checkpoint schema {payload.get('schema')}, expected 1")
    return payload


def objective_conflicts(saved: Mapping[str, Any], cfg: TrainConfig) -> list[str]:
    """Return the ``OBJECTIVE_KEYS`` that differ; a missing key reads as its default.

    Args:
        saved: A checkpoint's config.
        cfg: The run's config.

    Returns:
        The differing keys, in ``OBJECTIVE_KEYS`` order.
    """
    now = cfg.to_dict()
    return [
        key
        for key in OBJECTIVE_KEYS
        if saved.get(key, OBJECTIVE_DEFAULTS[key]) != now.get(key, OBJECTIVE_DEFAULTS[key])
    ]


def resume_conflicts(saved: Mapping[str, Any], cfg: TrainConfig) -> list[str]:
    """Return the ``RESUME_KEYS`` whose values differ between a checkpoint and a config.

    An ``OBJECTIVE_KEYS`` key absent from either side reads as its default (``zero_step``, no
    inner loop, no fresh schedule), so checkpoints written before P3.9 resume as before and a
    ``fomaml`` checkpoint never resumes under ``zero_step`` or another inner loop.
    """
    now = cfg.to_dict()
    objective = set(objective_conflicts(saved, cfg))
    return [
        key
        for key in RESUME_KEYS
        if (key in objective if key in OBJECTIVE_KEYS else saved.get(key) != now[key])
    ]


def init_conflicts(saved: Mapping[str, Any], cfg: TrainConfig) -> list[str]:
    """Return the settings a warm start (``--init-from``) may not change, where they differ.

    ``RESUME_KEYS`` as for a resume, except that ``hypernet`` may differ in its ``aggregator``
    block only: the warm start continues the checkpoint's data order and optimization. The
    ``OBJECTIVE_KEYS`` may differ (P3.9 starts ``fomaml`` from a ``zero_step`` H), and with
    ``fresh_schedule`` the ``SCHEDULE_KEYS`` too (the new phase's own schedule).
    """
    now = cfg.to_dict()
    free = {"hypernet", *OBJECTIVE_KEYS, *(SCHEDULE_KEYS if cfg.fresh_schedule else ())}
    conflicts = [key for key in RESUME_KEYS if key not in free and saved.get(key) != now[key]]

    def trunk(hypernet: Mapping[str, Any] | None) -> dict[str, Any]:
        return {k: v for k, v in (hypernet or {}).items() if k != "aggregator"}

    if trunk(saved.get("hypernet")) != trunk(now["hypernet"]):
        conflicts.append("hypernet")
    return conflicts


def rng_state(device: torch.device) -> dict[str, Any]:
    """Capture every random generator this process uses (Python, numpy, torch, CUDA)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def set_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    """Restore what ``rng_state`` captured."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


# --- metrics -------------------------------------------------------------------------------------


class MetricsLog:
    """``metrics.jsonl``: one JSON line per step and per validation, written by rank 0.

    The history travels in each checkpoint, so a resumed run rewrites the file up to its
    checkpoint and appends from there: steps after the checkpoint never appear twice.

    Args:
        path: The file.
        history: Records restored from a checkpoint.
    """

    def __init__(self, path: Path, history: Sequence[dict[str, Any]] = ()) -> None:
        self.path = path
        self.history = list(history)
        text = "".join(json.dumps(record, allow_nan=True) + "\n" for record in self.history)
        atomic_write_bytes(path, text.encode("utf-8"))

    def add(self, record: dict[str, Any]) -> None:
        """Append one record to the file and the history."""
        self.history.append(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=True) + "\n")


def _mean_se(values: Tensor) -> dict[str, float]:
    """Mean and standard error of a 1-D sample (SE 0 for one value)."""
    n = values.numel()
    mean = float(values.mean())
    se = float(values.std(unbiased=True) / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "se": se}


def _z(stat: Mapping[str, float]) -> float | None:
    """A paired difference in standard errors, ``mean / se``; None when the SE is 0."""
    return stat["mean"] / stat["se"] if stat["se"] > 0 else None


def selection_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    """The decode metrics a ``fomaml`` run selects on: after its inner steps, else at 0 steps."""
    if "adapted" in record:
        return record["adapted"]["decode"]
    return record["decode"]["current"]


def is_better(
    record: Mapping[str, Any], best: Mapping[str, Any] | None, objective: str = "zero_step"
) -> bool:
    """Whether a validation record is a new best point.

    ``zero_step``: a lower current-H CE. ``fomaml`` (P3.9): a higher greedy exact match after the
    inner steps, ties by a lower mean summed NLL $S$; an exact tie keeps the earlier point.

    Args:
        record: The new record.
        best: The best so far, or None.
        objective: The run's objective.

    Returns:
        True for a new best point.
    """
    if best is None:
        return True
    if objective == "fomaml":
        new, old = selection_metrics(record), selection_metrics(best)
        if new["exact_match"] != old["exact_match"]:
            return new["exact_match"] > old["exact_match"]
        return new["nll_sum"] < old["nll_sum"]
    return record["ce"]["current"] < best["ce"]["current"]


def best_validation(
    history: Sequence[Mapping[str, Any]], objective: str = "zero_step"
) -> dict[str, Any] | None:
    """Return the best validation record (``is_better``; the earliest on a tie).

    The best point of a run that spans sessions: a resumed session reads it from the
    checkpoint's history, so it rewrites ``best.pt`` only for a new best.

    Args:
        history: ``metrics.jsonl`` records in order.
        objective: The run's objective; ``zero_step`` selects on the current H's CE.

    Returns:
        The record, or None before the first validation.
    """
    best = None
    for record in history:
        if record.get("kind") == "val" and is_better(record, best, objective):
            best = dict(record)
    return best


def validation_record(
    step: int,
    ces: Mapping[str, Tensor],
    sequences: Sequence[EpisodeSequence],
    seconds: float,
    exact: Mapping[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Summarize one validation: means per reference, per source, and paired differences.

    Args:
        step: Optimizer steps taken.
        ces: Per-episode CE for each of ``REFERENCES``, ``[n_val]`` each.
        sequences: The validation episodes, in the same order.
        seconds: Wall time of the validation.
        exact: Per-episode greedy exact match (0 or 1) for each of ``REFERENCES``, ``[n_val]``
            each; adds ``decode`` (``decode_metrics`` per reference) and ``per_episode_exact``.

    Returns:
        The ``metrics.jsonl`` record (``kind`` ``val``); ``z`` holds both paired differences in
        standard errors (criterion B reads ``current_minus_zero``, check C
        ``mismatched_minus_current``).
    """
    sources = sorted({seq.source for seq in sequences})
    by_source = {}
    for source in sources:
        index = torch.tensor([i for i, seq in enumerate(sequences) if seq.source == source])
        by_source[source] = {"n": int(index.numel())} | {
            ref: float(ces[ref][index].mean()) for ref in REFERENCES
        }
    current_minus_zero = _mean_se(ces["current"] - ces["zero"])
    mismatched_minus_current = _mean_se(ces["mismatched"] - ces["current"])
    return {
        "kind": "val",
        "step": step,
        "n": len(sequences),
        "ce": {ref: float(ces[ref].mean()) for ref in REFERENCES},
        "by_source": by_source,
        "current_minus_zero": current_minus_zero,
        "mismatched_minus_current": mismatched_minus_current,
        "z": {
            "current_minus_zero": _z(current_minus_zero),
            "mismatched_minus_current": _z(mismatched_minus_current),
        },
        "seconds": seconds,
        "per_episode": {ref: [round(float(v), 6) for v in ces[ref]] for ref in REFERENCES},
    } | ({} if exact is None else _decode_block(ces, exact, sequences))


def _decode_block(
    ces: Mapping[str, Tensor], exact: Mapping[str, Tensor], sequences: Sequence[EpisodeSequence]
) -> dict[str, Any]:
    """``decode`` and ``per_episode_exact`` of a validation record, for every reference."""
    lengths = torch.tensor([len(seq.target) for seq in sequences], dtype=torch.float64)
    return {
        "decode": {ref: decode_metrics(ces[ref], lengths, exact[ref]) for ref in REFERENCES},
        "per_episode_exact": {ref: [int(v) for v in exact[ref]] for ref in REFERENCES},
    }


def _without_lists(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A record block without its per-episode and per-projection detail (for status.json)."""
    if block is None:
        return None
    return {
        key: value
        for key, value in block.items()
        if not key.startswith("per_episode") and key != "per_projection"
    }


# --- the per-process run -------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    """Wait for the device, so a timing record covers the GPU work it names."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device(cfg: TrainConfig, rank: int) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() < cfg.gpus:
        raise RuntimeError(
            f"{cfg.gpus} CUDA devices requested, {torch.cuda.device_count()} available"
        )
    device = torch.device("cuda", rank)
    if DTYPES[cfg.dtype] is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(device)} has no bf16 (T4 is not usable)")
    return device


@dataclass
class _Clock:
    """Session times the budget and the projection read."""

    started: float  # time.time() at the script's start
    elapsed_before: float = 0.0  # wall seconds of earlier sessions, from the checkpoint
    steps: list[float] = field(default_factory=list)
    evals: list[float] = field(default_factory=list)
    checkpoints: list[float] = field(default_factory=list)

    def session(self) -> float:
        return time.time() - self.started

    @staticmethod
    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0


def check_start(cfg: TrainConfig, resume: str | None, init_from: str | None) -> None:
    """Refuse a start the config cannot have: called before any data or model is read.

    Why: ``fresh_schedule`` and ``inner_probe`` act at a warm start's start step; without
    ``--init-from`` (or a ``--resume`` of such a run, whose checkpoint records the phase and the
    probe) both would be ignored and the run would train from step 0 under a config that says
    otherwise.

    Args:
        cfg: The run's config.
        resume: The ``--resume`` argument, or None.
        init_from: The ``--init-from`` argument, or None.

    Raises:
        ValueError: If both arguments are given, or neither while the config sets
            ``fresh_schedule`` or ``inner_probe``.
    """
    if resume is not None and init_from is not None:
        raise ValueError("--resume and --init-from exclude each other")
    if resume is None and init_from is None and (cfg.fresh_schedule or cfg.inner_probe):
        raise ValueError(
            "fresh_schedule and inner_probe act at a warm start's start step: pass "
            "--init-from a full checkpoint (or --resume such a run's own checkpoints)"
        )


class _Worker:
    """One process of a training run: setup, the loop, validation, checkpoints, status."""

    def __init__(
        self,
        rank: int,
        world: int,
        cfg: TrainConfig,
        resume: str | None,
        started: float,
        init_from: str | None = None,
    ) -> None:
        check_start(cfg, resume, init_from)
        self.rank, self.world, self.cfg, self.resume = rank, world, cfg, resume
        self.init_from = init_from
        # The set aggregator's input: the demonstrations' pair embeddings beside z_bar.
        self.uses_set = cfg.hypernet.aggregator.kind == "set"
        # P3.9: the query CE after k inner steps; k = 0 runs zero_step's micro step itself.
        self.fomaml = cfg.objective == "fomaml"
        self.inner_k = cfg.inner.steps if cfg.inner is not None else 0
        # A warm start's record (checkpoint, start step, new parameters, their settings), kept
        # in every checkpoint of the run so a resume rebuilds the same parameter groups.
        self.init_info: dict[str, Any] | None = None
        # True from a warm start's setup with inner_probe until its start validation picked the
        # training rate (_run_probe).
        self.probe_pending = False
        self.device = _device(cfg, rank)
        self.comm = _Comm(world, self.device)
        self.main = rank == 0
        self.out_dir = Path(cfg.out_dir)
        self.timing = TimingLogger(self.out_dir, worker=f"rank{rank}")
        self.clock = _Clock(started)
        self.step = 0
        self.zero_ce: Tensor | None = None
        # The zero adapter's greedy exact matches, constant as its CE (absent from checkpoints
        # written before P3.9: then the next validation scores the zero adapter once more).
        self.zero_exact: Tensor | None = None
        self.last_match: bool | None = None  # the exact match of episode_ce's last forward
        self.last_checkpoint: str | None = None
        self.checkpoint_step: int | None = None
        self.resumed_from: str | None = None
        self.history: list[dict[str, Any]] = []
        self.metrics: MetricsLog | None = None
        # The best validation point over all sessions (rank 0), the step this session started
        # at, and the best.pt this session wrote, if any.
        self.best: dict[str, Any] | None = None
        self.session_start = 0
        self.best_file: str | None = None

    # -- setup -------------------------------------------------------------------------------

    def setup(self) -> None:
        """Verify and load the episodes, the base, H, the optimizer, and the embeddings."""
        cfg = self.cfg
        torch.manual_seed(derive_seed(cfg.seed, "torch", self.rank) % 2**63)
        random.seed(derive_seed(cfg.seed, "python", self.rank))
        np.random.seed(derive_seed(cfg.seed, "numpy", self.rank) % 2**32)
        episodes = open_episode_set(
            Path(cfg.episodes_root), cfg.manifest_sha256, Path(cfg.check_root), cfg.max_tokens
        )
        self.episodes = episodes
        self.table = episodes.table
        self.train_records = load_split(episodes, cfg.train_split, cfg.max_tokens)
        self.val_records = load_split(episodes, cfg.val_split, cfg.max_tokens)
        self.order = EpisodeOrder(
            len(self.train_records),
            cfg.seed,
            sources=[record.source for record in self.train_records],
            weights=cfg.source_weights,
        )

        load_started = time.perf_counter()
        model, tokenizer, self.attn = load_base(cfg, self.device)
        self.load_seconds = time.perf_counter() - load_started
        self.keys = wrap_targets(model, cfg.hypernet.target_modules)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model = model
        self.formatter = QwenFormatter(tokenizer)
        self.val = [
            build_sequence(r, self.table, self.formatter, cfg.max_tokens, cfg.loss_includes_eos)
            for r in self.val_records
        ]
        self.mismatch = mismatch_permutation([seq.family for seq in self.val], cfg.seed)

        if cfg.hypernet.embed_dim != model.config.hidden_size:
            raise ValueError(
                f"hypernet.embed_dim {cfg.hypernet.embed_dim} differs from the base's hidden "
                f"size {model.config.hidden_size}, the width of z_bar"
            )
        hypernet = Hypernet(cfg.hypernet, BaseShapes.from_config(model.config))
        payload = None
        if self.resume is not None:
            path = resolve_checkpoint(Path(self.resume))
            payload = load_checkpoint(path)
            if payload.get("kind") == "best":
                raise ValueError(
                    f"{path} holds H's weights at the best validation point only, no optimizer "
                    "state; resume from a full checkpoint (its folder's latest.json)"
                )
            self.init_info = payload.get("init_from")
            cfg = self._resumed_config(path)
            conflicts = resume_conflicts(payload["config"], cfg)
            if conflicts:
                raise ValueError(f"{path} was written with other settings: {conflicts}")
            if payload["world_size"] != self.world:
                raise ValueError(f"{path} was written by {payload['world_size']} processes")
            hypernet.load_state_dict(payload["hypernet"])
            self.resumed_from = str(path)
            self.last_checkpoint, self.checkpoint_step = str(path), payload["step"]
            if self.init_info is not None:
                settings = {key: getattr(cfg, key) for key in NEW_PARAMS_KEYS}
                if settings != self.init_info["settings"]:
                    raise ValueError(
                        f"{path} was warm-started with {self.init_info['settings']}, not {settings}"
                    )
        elif self.init_from is not None:
            payload = self._warm_start(hypernet)
            self.probe_pending = cfg.inner_probe is not None
        self.hypernet = hypernet.to(self.device)
        self.h_train: nn.Module = (
            DistributedDataParallel(
                self.hypernet,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
            if self.world > 1
            else self.hypernet
        )
        self.optimizer = torch.optim.AdamW(
            param_groups(self.hypernet, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps
        )
        self.scheduler = LambdaLR(
            self.optimizer, WarmupCosine(cfg.warmup_steps, cfg.steps, cfg.min_lr / cfg.lr)
        )
        if self.init_info is not None and "fresh_schedule" in self.init_info:
            self._phase_optimizer()
        elif self.init_info is not None and self.init_info["parameters"]:
            self._warm_optimizer()

        views = view_descriptors(cfg.encoder.n_perm, cfg.encoder.view_seed)
        self.encoder = PairEncoder(
            model,
            tokenizer,
            layer=cfg.encoder.layer,
            pool=cfg.encoder.pool,
            views=views,
            batch_tokens=cfg.encoder.batch_tokens,
        )
        cache = None
        if cfg.embedding_cache is not None:
            cache = EmbeddingCache(
                cfg.embedding_cache,
                model_id=cfg.model_id,
                layer=cfg.encoder.layer,
                pool=cfg.encoder.pool,
                view_set_id=self.encoder.view_set_id,
            )
        self.embeddings = PairEmbeddings()
        model.eval()
        encode_started = time.perf_counter()
        self.embeddings.fill(
            demo_pairs([*self.train_records, *self.val_records]),
            self.encoder,
            cache,
            self.comm,
            self.rank,
            cfg.encoder.chunk_pairs,
            self.timing,
        )
        self.encode_seconds = time.perf_counter() - encode_started
        self.val_z = [self.embeddings.z_bar(seq.demo_hashes) for seq in self.val]
        self.val_sets = (
            [self.embeddings.pair_set(seq.demo_hashes) for seq in self.val] if self.uses_set else []
        )

        if payload is not None:
            if self.init_from is not None and "fresh_schedule" in self.init_info:
                pass  # a new phase: fresh moments, the phase schedule at its start
            elif self.init_from is not None and self.init_info["parameters"]:
                self._continue_optimizer(payload)
            else:
                self.optimizer.load_state_dict(payload["optimizer"])
                self.scheduler.load_state_dict(payload["scheduler"])
            self.step = payload["step"]
            if payload["cursor"] != train_slot(self.step, 0, 0, cfg.accum, self.world):
                raise ValueError("the checkpoint's data cursor does not match its step")
            self.zero_ce = payload["zero_ce"]
            self.zero_exact = payload.get("zero_exact")
            # A warm start begins its own history and clock: its validations and best.pt are
            # the new H's, its projected wall time this run's.
            if self.init_from is None:
                self.history = list(payload["history"])
                self.clock.elapsed_before = payload["elapsed_s"]
            set_rng_state(payload["rng"][self.rank], self.device)
        self.session_start = self.step
        self.best = best_validation(self.history, cfg.objective)
        if self.main:
            self.metrics = MetricsLog(self.out_dir / METRICS_FILE, self.history)
            self._write_run_info()

    def _resumed_config(self, path: Path) -> TrainConfig:
        """The config a resume runs under: the probe's pick as ``inner.lr``, when there was one.

        A probed warm start's checkpoints hold its config with ``inner.lr`` the rate the probe
        picked (``init_from.inner_probe.chosen``); the run's YAML holds the configured rate, so
        the pick is applied before ``resume_conflicts`` compares them.

        Args:
            path: The checkpoint, for messages.

        Returns:
            The config, also set as ``self.cfg``.

        Raises:
            ValueError: If the config sets ``fresh_schedule`` or ``inner_probe`` and the
                checkpoint records no such warm start.
        """
        cfg, info = self.cfg, self.init_info or {}
        if cfg.fresh_schedule and "fresh_schedule" not in info:
            raise ValueError(f"{path} records no fresh-schedule phase; the config sets one")
        if cfg.inner_probe is not None:
            if "inner_probe" not in info:
                raise ValueError(f"{path} records no inner-rate probe; the config sets one")
            chosen = info["inner_probe"]["chosen"]
            cfg = replace(cfg, inner=replace(cfg.inner, lr=chosen))
        self.cfg = cfg
        return cfg

    def _warm_start(self, hypernet: Hypernet) -> dict[str, Any]:
        """Load a full checkpoint into H by name (``--init-from``): new parameters or objective.

        A ``mean`` H's checkpoint into a ``set`` H (P3.3b) adds the aggregator; a ``zero_step``
        checkpoint under a ``fomaml`` config (P3.9) changes the objective and may add nothing,
        and then the optimizer and the scheduler load as a resume loads them.

        Returns:
            The checkpoint's payload; ``init_info`` records the warm start.

        Raises:
            ValueError: If the checkpoint holds no optimizer state, was written with other
                settings or another world size, is itself warm-started while ``fresh_schedule``
                is off, has keys the new H lacks, neither adds a parameter nor changes the
                objective nor starts a phase (then ``--resume`` is the tool), or adds parameters
                while the ``new_params_*`` settings are unset.
        """
        cfg = self.cfg
        path = resolve_checkpoint(Path(self.init_from))
        payload = load_checkpoint(path)
        if payload.get("kind") == "best":
            raise ValueError(
                f"{path} holds H's weights only; a warm start continues the step, the data "
                "cursor, and the optimizer, so it needs a full checkpoint"
            )
        conflicts = init_conflicts(payload["config"], cfg)
        if conflicts:
            raise ValueError(f"{path} was written with other settings: {conflicts}")
        if payload["world_size"] != self.world:
            raise ValueError(f"{path} was written by {payload['world_size']} processes")
        # A new phase may start from a phase (a longer run from P3.9-01's last checkpoint): its
        # optimizer and schedule are fresh, so the earlier warm start's groups do not carry over.
        if "init_from" in payload and not cfg.fresh_schedule:
            raise ValueError(
                f"{path} is itself warm-started: resume it with --resume, or start a new phase "
                "(fresh_schedule: true)"
            )
        missing, unexpected = hypernet.load_state_dict(payload["hypernet"], strict=False)
        if unexpected:
            raise ValueError(f"{path} holds parameters this H lacks: {unexpected[:4]}")
        objective = [
            key for key in objective_conflicts(payload["config"], cfg) if key != "fresh_schedule"
        ]
        if not missing and not objective and not cfg.fresh_schedule:
            raise ValueError(f"{path} holds every parameter of this H: use --resume")
        unset = [key for key in NEW_PARAMS_KEYS if getattr(cfg, key) is None]
        if cfg.fresh_schedule:
            if len(unset) != len(NEW_PARAMS_KEYS):
                raise ValueError(
                    "fresh_schedule puts every parameter on one schedule: unset new_params_*"
                )
            if cfg.warmup_steps >= cfg.steps - payload["step"]:
                raise ValueError(
                    f"the phase runs steps {payload['step'] + 1} to {cfg.steps}: warmup_steps "
                    f"{cfg.warmup_steps} must be shorter"
                )
        elif missing and unset:
            raise ValueError(f"a warm start that adds parameters needs {unset} in the config")
        self.init_info = {
            "checkpoint": str(path),
            "step": payload["step"],
            "parameters": list(missing),
            "settings": {key: getattr(cfg, key) for key in NEW_PARAMS_KEYS},
        }
        if "init_from" in payload:
            self.init_info["previous"] = payload["init_from"]
        if objective:
            saved = payload["config"].get("objective", OBJECTIVE_DEFAULTS["objective"])
            self.init_info["objective"] = {"from": saved, "to": cfg.objective}
        if cfg.fresh_schedule:
            # The phase starts at the checkpoint's step; its schedule counts from there.
            self.init_info["fresh_schedule"] = {
                "start": payload["step"],
                "phase_steps": cfg.steps - payload["step"],
                "warmup_steps": cfg.warmup_steps,
                "lr": cfg.lr,
                "min_lr": cfg.min_lr,
            }
        self.resumed_from = None
        return payload

    def _phase_optimizer(self) -> None:
        r"""A new training phase (``fresh_schedule``): one AdamW over every parameter, fresh.

        ``WarmupCosine`` over the phase's own steps: ``LambdaLR`` counts the steps taken since
        it was built, so phase step $n$ (global step $s_0 + n$) has multiplier $n / W$ for
        $n \le W$ and the cosine to ``min_lr`` at the phase's last step. A resume rebuilds the
        same pair from ``init_info`` and loads their state over it.
        """
        cfg, phase = self.cfg, self.init_info["fresh_schedule"]
        self.optimizer = torch.optim.AdamW(
            param_groups(self.hypernet, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            WarmupCosine(phase["warmup_steps"], phase["phase_steps"], cfg.min_lr / cfg.lr),
        )

    def _warm_optimizer(self) -> None:
        """Replace the optimizer and scheduler with a warm start's four groups (no state yet).

        The loaded parameters keep the checkpoint's two groups and ``WarmupCosine``; the new
        ones get two groups of their own at ``new_params_lr`` under ``NewParamsSchedule``. Called
        by ``setup`` after it built the default pair, so ``setup``'s own calls stay those of a
        run without a warm start.
        """
        cfg, info = self.cfg, self.init_info
        new, new_lr = info["parameters"], cfg.new_params_lr
        groups = param_groups(self.hypernet, cfg.weight_decay, new, new_lr)
        self.optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
        loaded = WarmupCosine(cfg.warmup_steps, cfg.steps, cfg.min_lr / cfg.lr)
        fresh = NewParamsSchedule(
            info["step"], cfg.new_params_warmup_steps, cfg.steps, cfg.min_lr / new_lr
        )
        self.scheduler = LambdaLR(self.optimizer, [loaded, loaded, fresh, fresh])

    def _continue_optimizer(self, payload: Mapping[str, Any]) -> None:
        """Load the checkpoint's AdamW state into the loaded groups; the new ones start fresh.

        The first two groups are the checkpoint's (same parameters in the same order, so its
        state indices apply): their moments, step counts, and current learning rates load
        unchanged. The two new groups keep empty state (fresh moments) and take
        ``NewParamsSchedule``'s rate for the start step. The scheduler is set to the start step,
        so the loaded groups continue ``WarmupCosine`` exactly as a resume would.
        """
        start = payload["step"]
        saved, fresh = payload["optimizer"], self.optimizer.state_dict()
        loaded = [group["params"] for group in saved["param_groups"]]
        if loaded != [group["params"] for group in fresh["param_groups"][:2]]:
            raise ValueError("the checkpoint's optimizer groups are not this H's loaded groups")
        new_groups = []
        for group, factor in zip(
            fresh["param_groups"][2:], self.scheduler.lr_lambdas[2:], strict=True
        ):
            new_groups.append({**group, "lr": group["initial_lr"] * factor(start)})
        self.optimizer.load_state_dict(
            {"state": saved["state"], "param_groups": [*saved["param_groups"], *new_groups]}
        )
        state = self.scheduler.state_dict()
        state.update(
            last_epoch=start,
            _step_count=start + 1,
            _last_lr=[group["lr"] for group in self.optimizer.param_groups],
        )
        self.scheduler.load_state_dict(state)

    def _write_run_info(self) -> None:
        """Merge the model, data, and H facts into ``run.json`` (the script wrote the rest)."""
        path = self.out_dir / "run.json"
        info = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        config = self.model.config
        n_params = sum(p.numel() for p in self.hypernet.parameters())
        rank = self.cfg.hypernet.rank
        emitted = self.hypernet.base.n_layers * sum(
            rank * (d_in + d_out) for d_in, d_out in self.hypernet.shapes.values()
        )
        info.update(
            {
                "world_size": self.world,
                "device_name": (
                    torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
                ),
                "attn_implementation": self.attn,
                "model": {
                    "load_seconds": self.load_seconds,
                    "num_hidden_layers": config.num_hidden_layers,
                    "hidden_size": config.hidden_size,
                    "vocab_size": config.vocab_size,
                    "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
                },
                "episodes": {
                    "folder": str(self.episodes.folder),
                    "manifest_sha256": self.episodes.manifest_sha256,
                    "check_report": str(self.episodes.check_report),
                    "train": len(self.train_records),
                    "val": len(self.val_records),
                    "val_episodes": [
                        {
                            "episode_id": seq.episode_id,
                            "source": seq.source,
                            "family": seq.family,
                            "tokens": int(seq.input_ids.numel()),
                            "target_tokens": len(seq.target),
                            "mismatched_with": self.val[self.mismatch[i]].episode_id,
                        }
                        for i, seq in enumerate(self.val)
                    ],
                },
                "order": self._order_plan(),
                "hypernet": {"parameters": n_params, "emitted_per_adapter": emitted},
                "embeddings": {
                    "pairs": len(self.embeddings),
                    "cache_hits": self.embeddings.cache_hits,
                    "encoded": self.embeddings.encoded,
                    "rank0_encode_tokens": self.embeddings.encode_tokens,
                    "seconds": self.encode_seconds,
                    "views": len(self.encoder.views),
                    "view_set_id": self.encoder.view_set_id,
                },
                "resumed_from": self.resumed_from,
                "start_step": self.step,
            }
        )
        if self.init_info is not None:
            info["init_from"] = self.init_info
        atomic_write_bytes(path, (json.dumps(info, indent=2) + "\n").encode("utf-8"))

    def _order_plan(self) -> dict[str, Any]:
        """The training draws planned per source: episodes, draws over all steps, and passes."""
        cfg = self.cfg
        pool = defaultdict(int)
        for record in self.train_records:
            pool[record.source] += 1
        slots = cfg.steps * cfg.accum * self.world
        draws = self.order.source_draws(slots)
        if draws is None:  # each episode once per pass: draws in proportion to the pool
            draws = {s: slots * n / len(self.train_records) for s, n in pool.items()}
        return {
            "source_weights": cfg.source_weights,
            "block": SOURCE_BLOCK if cfg.source_weights is not None else None,
            "per_block": getattr(self.order, "counts", None),
            "episodes": dict(sorted(pool.items())),
            "draws": {s: draws[s] for s in sorted(pool)},
            "passes": {s: draws[s] / pool[s] for s in sorted(pool)},
        }

    def _best_summary(self) -> dict[str, Any] | None:
        """``status.json``'s ``best``: the best point over all sessions (``is_better``)."""
        best = self.best
        if best is None:
            return None
        return {
            "step": best["step"],
            "ce": best["ce"],
            "current_minus_zero": best["current_minus_zero"],
            "mismatched_minus_current": best["mismatched_minus_current"],
            "z": best.get("z"),
            "by_source": best.get("by_source"),
            "decode": best.get("decode"),
            "adapted": _without_lists(best.get("adapted")),
            "session_start": best.get("session_start"),
            # best.pt lives in this session's ckpt_dir only if this session wrote it; otherwise
            # in the output of the session that started at session_start.
            "file": self.best_file,
        }

    # -- the loop ----------------------------------------------------------------------------

    def run(self) -> tuple[int, str]:
        """Set up, validate at step 0, train to ``steps``, and report.

        Returns:
            The exit code (``EXIT_OK``, ``EXIT_STOPPED``, or ``EXIT_KILLED``) and its reason.
        """
        self.setup()
        if self.step == 0 and not any(r["kind"] == "val" and r["step"] == 0 for r in self.history):
            self.evaluate()
        elif self.init_from is not None:
            # The warm-started H at its start step: equal to the loaded H (W_o = 0).
            self.evaluate()
            if self.cfg.inner_probe is not None:
                # The probe's pick goes into a checkpoint of the start step, so a restart
                # resumes at the picked rate instead of probing again.
                self.save()
        code, reason = self.loop()
        if self.main:
            status = {
                "status": {EXIT_OK: "complete", EXIT_STOPPED: "stopped", EXIT_KILLED: "killed"}[
                    code
                ],
                "exit_code": code,
                "reason": reason,
                "step": self.step,
                "steps": self.cfg.steps,
                "checkpoint": self.last_checkpoint,
                "best": self._best_summary(),
                "elapsed_s": self.clock.elapsed_before + self.clock.session(),
            }
            atomic_write_bytes(
                self.out_dir / STATUS_FILE, (json.dumps(status, indent=2) + "\n").encode("utf-8")
            )
        return code, reason

    def loop(self) -> tuple[int, str]:
        """Train until ``steps``, the session budget, or a kill criterion.

        Returns:
            The exit code and its reason.
        """
        cfg = self.cfg
        while self.step < cfg.steps:
            if self.comm.broadcast_int(self._over_budget() if self.main else 0):
                if self.checkpoint_step != self.step:
                    self.save()
                return EXIT_STOPPED, f"session budget: max_wall_s {cfg.max_wall_s:g} s reached"
            code, reason = self.train_step()
            if code:
                if self.checkpoint_step != self.step:
                    self.save()
                return code, reason
            if self.step % cfg.eval_every == 0 or self.step == cfg.steps:
                self.evaluate()
            if self.step % cfg.ckpt_every == 0 or self.step == cfg.steps:
                self.save()
            reason = self._kill_reason() if self.main else ""
            if self.comm.broadcast_int(EXIT_KILLED if reason else 0):
                if self.checkpoint_step != self.step:
                    self.save()
                return EXIT_KILLED, reason
        if self.checkpoint_step != self.step:
            self.save()
        return EXIT_OK, "complete"

    def _remaining(self, every: int) -> int:
        """Events at multiples of ``every`` or at the end in the steps still to run."""
        steps, now = self.cfg.steps, self.step
        if now >= steps:
            return 0
        return steps // every - now // every + (steps % every != 0)

    def _over_budget(self) -> int:
        """``EXIT_STOPPED`` when the next step might cross ``max_wall_s``, else 0."""
        clock, cfg = self.clock, self.cfg
        need = clock.mean(clock.steps) + clock.mean(clock.checkpoints)
        upcoming = self.step + 1
        if upcoming % cfg.eval_every == 0 or upcoming == cfg.steps:
            need += clock.mean(clock.evals)
        return EXIT_STOPPED if clock.session() + need > cfg.max_wall_s else 0

    def projected_wall(self) -> float:
        """Projected wall seconds of the whole run at this session's mean costs."""
        clock, cfg = self.clock, self.cfg
        return (
            clock.elapsed_before
            + clock.session()
            + (cfg.steps - self.step) * clock.mean(clock.steps)
            + self._remaining(cfg.eval_every) * clock.mean(clock.evals)
            + self._remaining(cfg.ckpt_every) * clock.mean(clock.checkpoints)
        )

    def _kill_reason(self) -> str:
        """The first kill criterion the last step breached, or empty; none after the last step."""
        kill = self.cfg.kill
        if self.step >= self.cfg.steps:
            return ""
        last = self.clock.steps[-1]
        if last > kill.max_step_s:
            return f"step {self.step} took {last:.1f} s, over kill.max_step_s {kill.max_step_s:g}"
        if len(self.clock.steps) >= kill.projection_after_steps:
            projected = self.projected_wall()
            if projected > kill.max_projected_s:
                return (
                    f"projected wall time {projected:.0f} s after step {self.step}, over "
                    f"kill.max_projected_s {kill.max_projected_s:g}"
                )
        return ""

    # -- one optimizer step ------------------------------------------------------------------

    def _train_modes(self) -> None:
        # transformers checkpoints only in training mode; Qwen3's attention dropout is 0
        # (load_base checks), so training mode computes what evaluation mode computes.
        self.model.train(self.cfg.gradient_checkpointing)
        self.hypernet.train()

    def micro_step(self, seq: EpisodeSequence, micro: int) -> tuple[float, float]:
        r"""Forward and backward of one episode; gradients accumulate in H.

        H runs on $\bar z$ (and a ``set`` H on the pair embeddings too), then forward, loss, and
        ``backward`` all run inside ``inject``:
        activation checkpointing recomputes layer forwards during backward, which must see the
        same factors (P3.1, risk 2). DDP synchronizes on the last micro step only.

        Args:
            seq: The episode.
            micro: Micro step index within the optimizer step.

        Returns:
            The episode's query CE and its factor penalty (0 when off), as floats.
        """
        cfg, device = self.cfg, self.device
        z_bar = self.embeddings.z_bar(seq.demo_hashes).to(device)[None]  # [1, hidden]
        inputs = (z_bar,)
        if self.uses_set:
            pairs = self.embeddings.pair_set(seq.demo_hashes).to(device)[None]  # [1, n, hidden]
            inputs += (pairs,)
        ids = seq.input_ids.to(device)  # [seq]
        last = micro == cfg.accum - 1
        sync = nullcontext() if self.world == 1 or last else self.h_train.no_sync()
        with sync:
            with self.timing.timed(seq.episode_id, "h_forward"):
                factors = self.h_train(*inputs)[0]
                _sync(device)
            with (
                inject(self.model, factors, self.cfg.hypernet.scaling),
                self.timing.timed(seq.episode_id, "train_step", int(ids.numel())),
            ):
                logits = self.model(input_ids=ids[None], use_cache=False).logits[0]
                ce = query_ce(logits, ids, seq.target)
                loss = ce
                penalty = torch.zeros((), device=device)
                if cfg.penalty != "none":
                    penalty = factor_penalty(factors, cfg.penalty)
                    loss = loss + cfg.penalty_weight * penalty
                (loss / cfg.accum).backward()
                ce_value, penalty_value = float(ce.item()), float(penalty.item())
        return ce_value, penalty_value

    def inner_demos(self, record: EpisodeRecord) -> list[DemoSequence]:
        """The demonstration sequence of each inner step of an episode (``inner_view_ids``)."""
        cfg = self.cfg
        ids = inner_view_ids(
            cfg.seed,
            record.episode_id,
            record.augmentation_id,
            self.inner_k,
            self.table.n_augmentations,
            cfg.inner.views,
        )
        built: dict[int, DemoSequence] = {}
        for aug in ids:
            if aug not in built:
                built[aug] = build_demo_sequence(
                    record, self.table, self.formatter, aug, cfg.max_tokens, cfg.loss_includes_eos
                )
        return [built[aug] for aug in ids]

    def fomaml_micro_step(
        self, record: EpisodeRecord, seq: EpisodeSequence, micro: int
    ) -> tuple[float, float, list[float]]:
        r"""Forward and backward of one episode under ``fomaml``; gradients accumulate in H.

        $\theta_0 = H(\bar z)$; ``adapt`` takes $k$ inner steps on a detached copy; the query
        CE at $\theta_k$ gives $g = \nabla_\theta \mathcal L_Q(\theta_k)$ (forward and
        ``torch.autograd.grad`` inside one ``inject`` block); then
        ``torch.autograd.backward(theta_0, g)`` applies $g$ at $\theta_0$, so H receives
        $J_\phi^\top g$, the first-order MAML gradient (no second derivatives: the inner steps
        are outside H's graph). A factor penalty acts on $\theta_0$ as under ``zero_step``. With
        $k = 0$ this is ``micro_step`` itself.

        Args:
            record: The episode's record (its demonstrations feed the inner steps).
            seq: Its sequence.
            micro: Micro step index within the optimizer step.

        Returns:
            The query CE at $\theta_k$, the penalty, and the inner statistics: the
            demonstrations' CE at $\theta_0, \ldots, \theta_{k-1}$, each step's gradient norm
            before the clip, the tokens the $k$ inner steps read, then per module type (in
            sorted order) $\lVert s(B_kA_k - B_0A_0)\rVert_F$ and $\lVert sB_0A_0\rVert_F$,
            each the mean over layers; empty for $k = 0$.
        """
        if self.inner_k == 0:
            return (*self.micro_step(seq, micro), [])
        cfg, device = self.cfg, self.device
        scaling = cfg.hypernet.scaling
        z_bar = self.embeddings.z_bar(seq.demo_hashes).to(device)[None]  # [1, hidden]
        inputs = (z_bar,)
        if self.uses_set:
            pairs = self.embeddings.pair_set(seq.demo_hashes).to(device)[None]  # [1, n, hidden]
            inputs += (pairs,)
        ids = seq.input_ids.to(device)  # [seq]
        demos = self.inner_demos(record)
        last = micro == cfg.accum - 1
        sync = nullcontext() if self.world == 1 or last else self.h_train.no_sync()
        with sync:
            with self.timing.timed(seq.episode_id, "h_forward"):
                factors = self.h_train(*inputs)[0]
                _sync(device)
            adapted, inner_ces, inner_norms = adapt(
                self.model, factors, demos, cfg.inner, scaling, device, self.timing
            )
            params = flat_factors(adapted)
            with (
                inject(self.model, adapted, scaling),
                self.timing.timed(seq.episode_id, "train_step", int(ids.numel())),
            ):
                logits = self.model(input_ids=ids[None], use_cache=False).logits[0]
                ce = query_ce(logits, ids, seq.target)
                grads = list(torch.autograd.grad(ce / cfg.accum, params))
                ce_value = float(ce.item())
            tensors = flat_factors(factors)
            penalty = torch.zeros((), device=device)
            if cfg.penalty != "none":
                penalty = factor_penalty(factors, cfg.penalty)
                tensors.append(cfg.penalty_weight * penalty / cfg.accum)
                grads.append(torch.ones_like(tensors[-1]))
            with self.timing.timed(seq.episode_id, "h_backward"):
                torch.autograd.backward(tensors, grads)
                _sync(device)
            penalty_value = float(penalty.item())
        moved = by_module(displacement(factors, adapted, scaling))
        start = by_module(delta_norms(factors, scaling))
        tokens = float(sum(demo.input_ids.numel() for demo in demos))
        stats = [*inner_ces, *inner_norms, tokens, *moved.values(), *start.values()]
        return ce_value, penalty_value, stats

    def train_step(self) -> tuple[int, str]:
        """One optimizer step over ``gpus * accum`` episodes.

        Returns:
            ``(EXIT_KILLED, reason)`` on a non-finite loss or gradient norm (nothing is applied),
            else ``(0, "")``.
        """
        cfg = self.cfg
        began = time.perf_counter()
        self._train_modes()
        rows, inner_rows = [], []
        for micro in range(cfg.accum):
            index = self.order(train_slot(self.step, micro, self.rank, cfg.accum, self.world))
            seq = build_sequence(
                self.train_records[index],
                self.table,
                self.formatter,
                cfg.max_tokens,
                cfg.loss_includes_eos,
            )
            if self.fomaml:
                ce, penalty, stats = self.fomaml_micro_step(self.train_records[index], seq, micro)
                inner_rows.append(stats)
            else:
                ce, penalty = self.micro_step(seq, micro)
            rows.append([ce, penalty, float(seq.input_ids.numel())])
        table = torch.stack(self.comm.all_gather(torch.tensor(rows, dtype=torch.float64)))
        inner = None
        if self.inner_k > 0:
            inner = torch.stack(self.comm.all_gather(torch.tensor(inner_rows, dtype=torch.float64)))
        grad_norm = float(nn.utils.clip_grad_norm_(self.hypernet.parameters(), cfg.clip))
        finite = bool(torch.isfinite(table).all()) and math.isfinite(grad_norm)
        if inner is not None:
            finite = finite and bool(torch.isfinite(inner).all())
        if self.comm.broadcast_int(0 if finite else EXIT_KILLED):
            self.optimizer.zero_grad(set_to_none=True)
            return EXIT_KILLED, f"non-finite loss or gradient norm in step {self.step + 1}"
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        seconds = time.perf_counter() - began
        self.clock.steps.append(seconds)
        if self.metrics is not None:
            self._log_step(table, lr, grad_norm, seconds, inner)
        return 0, ""

    def _log_step(
        self,
        table: Tensor,
        lr: float,
        grad_norm: float,
        seconds: float,
        inner: Tensor | None = None,
    ) -> None:
        r"""Rank 0: append the step's record; ``table`` is ``[world, accum, 3]``.

        ``inner`` (``fomaml`` with $k \ge 1$) is ``[world, accum, 2k + 1 + 2m]`` for $m$ module
        types (``fomaml_micro_step``'s statistics); its means over the step's episodes (its
        tokens summed) become the record's ``inner`` block. ``tokens`` stays the query
        sequences' tokens, comparable with a ``zero_step`` run's.
        """
        cfg = self.cfg
        episodes, sources = [], []
        for micro in range(cfg.accum):
            for rank in range(self.world):
                slot = train_slot(self.step - 1, micro, rank, cfg.accum, self.world)
                record = self.train_records[self.order(slot)]
                episodes.append(record.episode_id)
                sources.append(record.source)
        flat = table.permute(1, 0, 2).reshape(-1, 3)  # slot order: micro-major, then rank
        by_source: dict[str, list[float]] = defaultdict(list)
        for source, value in zip(sources, flat[:, 0].tolist(), strict=True):
            by_source[source].append(value)
        self.metrics.add(
            {
                "kind": "train",
                "step": self.step,
                "lr": lr,
                "loss": float(flat[:, 0].mean()),
                "loss_by_source": {s: sum(v) / len(v) for s, v in sorted(by_source.items())},
                "penalty": float(flat[:, 1].mean()),
                "grad_norm": grad_norm,
                "tokens": int(flat[:, 2].sum()),
                "step_s": seconds,
                "episodes": episodes,
                "episode_ce": [round(value, 6) for value in flat[:, 0].tolist()],
            }
            | ({} if inner is None else {"inner": self._inner_summary(inner)})
        )

    def _inner_summary(self, inner: Tensor) -> dict[str, Any]:
        """The ``inner`` block of a training record from ``[world, accum, 2k + 1 + 2m]`` rows."""
        k = self.inner_k
        modules = sorted(set(self.cfg.hypernet.target_modules))
        rows = inner.reshape(-1, inner.shape[-1])
        means = rows.mean(dim=0).tolist()
        m, t = len(modules), 2 * k  # t: the tokens column
        return {
            "steps": k,
            "lr": self.cfg.inner.lr,
            "demo_ce": means[:k],
            "grad_norm": means[k:t],
            "tokens": int(rows[:, t].sum()),
            "displacement": dict(zip(modules, means[t + 1 : t + 1 + m], strict=True)),
            "start_norm": dict(zip(modules, means[t + 1 + m : t + 1 + 2 * m], strict=True)),
        }

    # -- validation --------------------------------------------------------------------------

    def episode_ce(
        self, seq: EpisodeSequence, z_bar: Tensor | None, pairs: Tensor | None = None
    ) -> float:
        r"""Query CE of one validation episode, without an adapter or with H's for ``z_bar``.

        Args:
            seq: The episode.
            z_bar: $\bar z$, ``[hidden]``; None scores the zero adapter (the base itself).
            pairs: The pair embeddings behind ``z_bar``, ``[n, hidden]``, for a ``set`` H;
                None for a ``mean`` H.

        Returns:
            The CE; ``last_match`` holds the greedy exact match of the same forward.
        """
        ce, self.last_match = self.episode_scores(seq, z_bar, pairs)
        return ce

    def _h_inputs(self, z_bar: Tensor, pairs: Tensor | None) -> tuple[Tensor, ...]:
        inputs = (z_bar.to(self.device)[None],)  # [1, hidden]
        if pairs is not None:
            inputs += (pairs.to(self.device)[None],)  # [1, n, hidden]
        return inputs

    def episode_scores(
        self, seq: EpisodeSequence, z_bar: Tensor | None, pairs: Tensor | None = None
    ) -> tuple[float, bool]:
        r"""``episode_ce`` and the greedy exact match of the same forward (``query_scores``).

        Args:
            seq: The episode.
            z_bar: $\bar z$, ``[hidden]``; None scores the zero adapter.
            pairs: The pair embeddings behind ``z_bar``, ``[n, hidden]``, for a ``set`` H.

        Returns:
            The CE and the match.
        """
        ids = seq.input_ids.to(self.device)
        with torch.no_grad():
            if z_bar is None:
                logits = self.model(input_ids=ids[None], use_cache=False).logits[0]
            else:
                factors = self.hypernet(*self._h_inputs(z_bar, pairs))[0]
                with inject(self.model, factors, self.cfg.hypernet.scaling):
                    logits = self.model(input_ids=ids[None], use_cache=False).logits[0]
            ce, exact = query_scores(logits, ids, seq.target)
            return float(ce.item()), exact

    def adapted_scores(
        self,
        index: int,
        z_bar: Tensor,
        pairs: Tensor | None = None,
        settings: InnerLoop | None = None,
    ) -> tuple[float, bool, dict[str, float], dict[str, float], list[float]]:
        r"""Score one validation episode after $k$ inner steps from the current H's adapter.

        H runs in evaluation mode under ``torch.no_grad``; the inner steps run with the base in
        training mode when checkpointing (as in training, and for the same memory), then the
        query is scored under ``torch.no_grad``. Timing ops ``validate_inner_step`` and
        ``validate_adapted``, apart from training's ``inner_step``.

        Args:
            index: The episode's position in the validation split.
            z_bar: Its $\bar z$, ``[hidden]``.
            pairs: Its pair embeddings, ``[n, hidden]``, for a ``set`` H.
            settings: The inner loop; None is the config's (``InnerProbe`` passes each rate).

        Returns:
            The query CE and greedy exact match at $\theta_k$, per projection the inner
            displacement and $\lVert sB_0A_0\rVert_F$, and each inner step's gradient norm.
        """
        cfg, seq = self.cfg, self.val[index]
        settings = cfg.inner if settings is None else settings
        scaling = cfg.hypernet.scaling
        ids = seq.input_ids.to(self.device)
        with torch.no_grad():
            factors = self.hypernet(*self._h_inputs(z_bar, pairs))[0]
        self.model.train(cfg.gradient_checkpointing)
        try:
            demos = self.inner_demos(self.val_records[index])
            adapted, _, norms = adapt(
                self.model,
                factors,
                demos,
                settings,
                scaling,
                self.device,
                self.timing,
                op="validate_inner_step",
            )
        finally:
            self.model.eval()
        with (
            torch.no_grad(),
            inject(self.model, adapted, scaling),
            self.timing.timed(seq.episode_id, "validate_adapted", int(ids.numel())),
        ):
            logits = self.model(input_ids=ids[None], use_cache=False).logits[0]
            ce, exact = query_scores(logits, ids, seq.target)
            ce_value = float(ce.item())
        moved = displacement(factors, adapted, scaling)
        return ce_value, exact, moved, delta_norms(factors, scaling), norms

    def evaluate(self) -> None:
        r"""Validate: each rank scores episodes ``rank::world``; rank 0 logs the record.

        Every reference is scored at 0 steps (CE and greedy exact match); under ``fomaml`` with
        $k \ge 1$ the current H is also scored after its $k$ inner steps (``adapted``). At a warm
        start's start step with ``inner_probe`` (``probe_pending``), ``adapted`` is scored at every
        probed rate, the rule picks the training rate (``_run_probe``), and the record's
        ``probe`` holds every rate's block; the other rates' time is left out of the clock's
        validation mean, which the budget and the projection read.
        """
        began = time.perf_counter()
        cached = self.zero_ce is not None and self.zero_exact is not None
        refs = ("current", "mismatched") if cached else REFERENCES
        self.model.eval()
        self.hypernet.eval()
        n = len(self.val)
        local = torch.zeros(len(refs), n, dtype=torch.float64)
        hits = torch.zeros(len(refs), n, dtype=torch.float64)
        for i in range(self.rank, n, self.world):
            seq = self.val[i]
            for row, ref in enumerate(refs):
                # The mismatched control feeds the other family's set along with its z_bar.
                source = {"zero": None, "current": i, "mismatched": self.mismatch[i]}[ref]
                z_bar = None if source is None else self.val_z[source]
                pairs = self.val_sets[source] if self.uses_set and source is not None else None
                with self.timing.timed(seq.episode_id, "validate", int(seq.input_ids.numel())):
                    extra = () if pairs is None else (pairs,)
                    # episode_ce is the validation's one scoring call (tests script it); its
                    # forward's exact match comes back in last_match, 0 when it is not set.
                    self.last_match = None
                    local[row, i] = self.episode_ce(seq, z_bar, *extra)
                    hits[row, i] = float(bool(self.last_match))
        values = dict(zip(refs, self.comm.all_reduce_sum(local), strict=True))
        matches = dict(zip(refs, self.comm.all_reduce_sum(hits), strict=True))
        if self.zero_ce is None:
            self.zero_ce = values["zero"]
        if self.zero_exact is None:
            self.zero_exact = matches["zero"]
        values["zero"], matches["zero"] = self.zero_ce, self.zero_exact
        probe, probe_s = None, 0.0
        if self.probe_pending:
            adapted, probe, probe_s = self._run_probe()
        else:
            adapted = self._evaluate_adapted() if self.inner_k > 0 else None
        seconds = time.perf_counter() - began
        if self.metrics is not None:
            record = validation_record(self.step, values, self.val, seconds, matches)
            if adapted is not None:
                record["adapted"] = self._adapted_block(values, *adapted)
            if probe is not None:
                record["probe"] = {
                    **probe["summary"],
                    "seconds": probe_s,
                    "results": [self._adapted_block(values, *found) for found in probe["found"]],
                }
            record["session_start"] = self.session_start
            # A new best point over all sessions (the first validation is one).
            record["best"] = is_better(record, self.best, self.cfg.objective)
            if record["best"]:
                written = time.perf_counter()
                path = save_best(Path(self.cfg.ckpt_dir), self._best_payload(record))
                self.best_file = str(path)
                record["best_write_s"] = time.perf_counter() - written
                self.best = record
            self.metrics.add(record)
        # Rank 0's clock feeds the budget and the projection, so it includes best.pt's write.
        self.clock.evals.append(time.perf_counter() - began - probe_s)

    def _run_probe(self) -> tuple[tuple[Tensor, ...], dict[str, Any], float]:
        r"""Score ``adapted`` at every rate of ``inner_probe`` and train at the rule's pick.

        Every rank computes the pick from the same all-reduced scores; rank 0's index is
        broadcast all the same, so the ranks cannot train at different rates. ``self.cfg``
        then holds ``inner.lr`` = the pick (the checkpoints' config, which ``--resume`` rebuilds
        from ``init_from.inner_probe``), and ``init_info`` records the probe.

        Returns:
            The pick's ``_evaluate_adapted`` result, the probe (``found``: every rate's result
            in rate order; ``summary``: the rates, the margin, the pick, and each rate's scores),
            and the seconds of the rates other than the pick.
        """
        cfg, probe = self.cfg, self.cfg.inner_probe
        lengths = torch.tensor([len(seq.target) for seq in self.val], dtype=torch.float64)
        found, scores, seconds = [], [], []
        for lr in probe.lrs:
            began = time.perf_counter()
            result = self._evaluate_adapted(replace(cfg.inner, lr=lr))
            seconds.append(time.perf_counter() - began)
            ces, hits = result[0], result[1]
            found.append(result)
            scores.append(
                {
                    "lr": lr,
                    "exact": int(hits.sum()),
                    "nll_sum": float((ces * lengths).mean()),
                    "ce": float(ces.mean()),
                }
            )
        pick = self.comm.broadcast_int(choose_inner_lr(scores, probe.margin) if self.main else 0)
        chosen = probe.lrs[pick]
        self.cfg = replace(cfg, inner=replace(cfg.inner, lr=chosen))
        summary = {
            "lrs": list(probe.lrs),
            "margin": probe.margin,
            "configured_lr": cfg.inner.lr,
            "chosen": chosen,
            "scores": scores,
        }
        self.init_info["inner_probe"] = summary
        self.probe_pending = False
        if self.main:
            self._write_run_info()
        return found[pick], {"found": found, "summary": summary}, sum(seconds) - seconds[pick]

    def _evaluate_adapted(self, settings: InnerLoop | None = None) -> tuple[Tensor, ...]:
        r"""Every rank's share of the post-$k$ scores, summed over the ranks.

        Args:
            settings: The inner loop; None is the config's.

        Returns:
            Per episode the CE and exact match at $\theta_k$, ``[n_val]`` each; per projection
            (``self.keys`` order) the sums over episodes of the displacement and of
            $\lVert sB_0A_0\rVert_F$, ``[n_keys]`` each; and per inner step the sum over episodes
            of the gradient norm before the clip, ``[k]``.
        """
        n, n_keys = len(self.val), len(self.keys)
        ces = torch.zeros(n, dtype=torch.float64)
        hits = torch.zeros(n, dtype=torch.float64)
        moved = torch.zeros(n_keys, dtype=torch.float64)
        start = torch.zeros(n_keys, dtype=torch.float64)
        norms = torch.zeros(self.inner_k, dtype=torch.float64)
        for i in range(self.rank, n, self.world):
            pairs = self.val_sets[i] if self.uses_set else None
            ce, exact, disp, norm, grads = self.adapted_scores(i, self.val_z[i], pairs, settings)
            ces[i], hits[i] = ce, exact
            moved += torch.tensor([disp[key] for key in self.keys], dtype=torch.float64)
            start += torch.tensor([norm[key] for key in self.keys], dtype=torch.float64)
            norms += torch.tensor(grads, dtype=torch.float64)
        summed = self.comm.all_reduce_sum(torch.stack((ces, hits)))
        moved_start = self.comm.all_reduce_sum(torch.stack((moved, start)))
        return (
            summed[0],
            summed[1],
            moved_start[0],
            moved_start[1],
            self.comm.all_reduce_sum(norms),
        )

    def _adapted_block(
        self,
        values: Mapping[str, Tensor],
        ces: Tensor,
        hits: Tensor,
        moved: Tensor,
        start: Tensor,
        norms: Tensor,
    ) -> dict[str, Any]:
        """The ``adapted`` block of a validation record: the current H after $k$ inner steps."""
        n = len(self.val)
        lengths = torch.tensor([len(seq.target) for seq in self.val], dtype=torch.float64)
        sources = sorted({seq.source for seq in self.val})
        by_source = {}
        for source in sources:
            index = torch.tensor([i for i, seq in enumerate(self.val) if seq.source == source])
            by_source[source] = {"n": int(index.numel()), "ce": float(ces[index].mean())}
        moved_mean = dict(zip(self.keys, (moved / n).tolist(), strict=True))
        start_mean = dict(zip(self.keys, (start / n).tolist(), strict=True))
        minus_current = _mean_se(ces - values["current"])
        minus_zero = _mean_se(ces - values["zero"])
        return {
            "steps": self.inner_k,
            "ce": float(ces.mean()),
            "by_source": by_source,
            "decode": decode_metrics(ces, lengths, hits),
            "minus_current": minus_current,
            "minus_zero": minus_zero,
            "z": {"minus_current": _z(minus_current), "minus_zero": _z(minus_zero)},
            "displacement": by_module(moved_mean),
            "start_norm": by_module(start_mean),
            "grad_norm": (norms / n).tolist(),
            "per_projection": {"displacement": moved_mean, "start_norm": start_mean},
            "per_episode": [round(float(v), 6) for v in ces],
            "per_episode_exact": [int(v) for v in hits],
        }

    def _best_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        """``best.pt``: H's weights, its config, and the validation record, no optimizer state.

        ``config`` is ``h_config`` (no ``OBJECTIVE_KEYS``, so a loader on the code before P3.9
        parses it); ``objective`` holds those keys as the run trained under them (``inner.lr``
        the probe's pick), present only when one differs from its default.
        """
        keys = ("step", "n", "ce", "by_source", "current_minus_zero", "mismatched_minus_current")
        extra = {key: _without_lists(record[key]) for key in ("adapted",) if key in record}
        if "decode" in record:
            extra["decode"] = record["decode"]
        full = self.cfg.to_dict()
        objective = {key: full[key] for key in OBJECTIVE_KEYS if key in full}
        return {
            "schema": CHECKPOINT_SCHEMA,
            "kind": "best",
            "step": self.step,
            "world_size": self.world,
            "config": h_config(self.cfg),
            **({"objective": objective} if objective else {}),
            "hypernet": self.hypernet.state_dict(),
            "validation": {key: record[key] for key in keys} | {"z": record["z"]} | extra,
            "session_start": self.session_start,
            "elapsed_s": self.clock.elapsed_before + self.clock.session(),
        }

    # -- checkpoints -------------------------------------------------------------------------

    def save(self) -> None:
        """Write a checkpoint of this step; every rank contributes its RNG states."""
        began = time.perf_counter()
        rng = self.comm.all_gather_object(rng_state(self.device))
        if self.main:
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "step": self.step,
                "cursor": train_slot(self.step, 0, 0, self.cfg.accum, self.world),
                "world_size": self.world,
                "config": self.cfg.to_dict(),
                "hypernet": self.hypernet.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "rng": rng,
                "zero_ce": self.zero_ce,
                "zero_exact": self.zero_exact,
                "history": self.metrics.history if self.metrics is not None else [],
                "elapsed_s": self.clock.elapsed_before + self.clock.session(),
            }
            if self.init_info is not None:
                payload["init_from"] = self.init_info
            path = save_checkpoint(Path(self.cfg.ckpt_dir), payload, self.cfg.keep_checkpoints)
            self.last_checkpoint = str(path)
        self.checkpoint_step = self.step
        self.clock.checkpoints.append(time.perf_counter() - began)


def run_worker(
    rank: int,
    world: int,
    cfg: TrainConfig,
    resume: str | None,
    started: float,
    port: int,
    init_from: str | None = None,
) -> None:
    """Entry point of one training process (``torch.multiprocessing.spawn`` calls it).

    Args:
        rank: This process's rank.
        world: Number of processes.
        cfg: The run config.
        resume: Checkpoint file or folder to continue from, or None.
        started: ``time.time()`` at the script's start, for the session budget.
        port: TCP port of the process group's rendezvous on 127.0.0.1 (unused when ``world``
            is 1).
        init_from: Full checkpoint (file or folder) to warm-start a ``set`` H from, or None.
    """
    if world > 1:
        if cfg.device == "cuda":
            torch.cuda.set_device(rank)
        else:
            # One process per core group: oversubscribed threads slow every process down.
            torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))
        dist.init_process_group(
            backend="nccl" if cfg.device == "cuda" else "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
        )
    try:
        _Worker(rank, world, cfg, resume, started, init_from).run()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def free_port() -> int:
    """Return a TCP port on 127.0.0.1 that was free a moment ago (for the rendezvous)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_status(out_dir: Path) -> dict[str, Any] | None:
    """Return ``status.json`` of a finished run, or None if no process wrote one."""
    path = Path(out_dir) / STATUS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def train(
    cfg: TrainConfig,
    resume: str | None = None,
    started: float | None = None,
    init_from: str | None = None,
) -> int:
    """Run the meta-training: ``cfg.gpus`` processes, or this process when ``gpus`` is 1.

    The status of the run is ``status.json`` in ``out_dir``, written by rank 0; an exception in
    any process propagates (``torch.multiprocessing`` raises it in the parent).

    Args:
        cfg: The run config.
        resume: Checkpoint file or folder to continue from.
        started: ``time.time()`` at the caller's start; now when None.
        init_from: Full checkpoint of a ``mean`` H to warm-start a ``set`` H from (P3.3b); it
            excludes ``resume``.

    Returns:
        The exit code rank 0 recorded: 0 complete, 3 stopped (resumable), 4 killed.

    Raises:
        RuntimeError: If no process recorded a status.
    """
    check_start(cfg, resume, init_from)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / STATUS_FILE).unlink(missing_ok=True)
    started = time.time() if started is None else started
    if cfg.gpus == 1:
        run_worker(0, 1, cfg, resume, started, 0, init_from)
    else:
        mp.spawn(
            run_worker,
            args=(cfg.gpus, cfg, resume, started, free_port(), init_from),
            nprocs=cfg.gpus,
        )
    status = read_status(out_dir)
    if status is None:
        raise RuntimeError("the run ended without a status")
    return int(status["exit_code"])
