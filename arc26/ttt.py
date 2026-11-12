"""Per-puzzle test-time training of the split solver (tasks P2.0, P2.3): ``fit``, ``SGDProvider``.

The reference's test-time training, moved out of ``arc_solver.worker`` so that it can be called on
its own (ADR-011): ``fit`` resets the LoRA adapter to its default weights and trains it for one
epoch on one puzzle's demonstration pairs under 128 augmented views, then returns the model in
inference mode for ``arc26.decode.decode_puzzle``. ``arc26.solver.worker`` calls it once per
puzzle. ``UnslothFixedTrainer`` and ``QwenDataCollatorForCompletionOnlyLM`` move with it. The
statements are the reference's, unchanged, with their timing record (``train``; P0.3b). Importing
this module imports unsloth first and disables every logging call at ``WARNING`` and below for
the process, as importing ``arc_solver`` did; torch and unsloth exist only on the Kaggle image.
Task P2.3 adds ``SGDProvider`` beside ``fit``, our code, not moved: the refinement round's
adapter provider (B4), ``sgd_continue`` as ``fit`` from a snapshot of the current adapter with
``max_steps`` set, and the rollback that loads the snapshot back (ADR-015). See
``docs/modules/ttt.md``.
"""

# Moved from reference/reference_notebook.ipynb, cell 4 (%%writefile arc_solver.py), through the
# generated arc26/nvarc/arc_solver.py (P0.1; P0.3b timing records; P1.3 LoRA settings):
#   logging.disable: line 36; UnslothFixedTrainer: lines 60-85;
#   QwenDataCollatorForCompletionOnlyLM: lines 88-105;
#   fit: lines 331-382, the test-time training block of the worker's per-puzzle loop;
#   imports: those of lines 10-34 that this code reads, unsloth first.
# Moved, not rewritten: tests/test_split_solver.py checks every statement against
# arc_solver.py on the AST and lists the intended differences (signatures, docstrings, returns).
# Only the layout changed: long lines wrapped, trailing whitespace removed.
# Origin: the NVARC team's ARC Prize 2025 (ARC-AGI-2) first-place notebook,
#   Kaggle sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue
# License: Apache License 2.0; attribution in reference/README.md

from __future__ import annotations

# First, as in arc_solver.py: unsloth patches transformers, peft, and trl, so it must be imported
# before them.
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

# isort: split
import gc
import io
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
from datasets import Dataset
from peft import set_peft_model_state_dict
from transformers import DataCollatorForLanguageModeling

from arc26 import timing
from arc26.decode import ASSISTANT_TOKEN_ID, EOS_ID, USER_TOKEN_ID

if TYPE_CHECKING:
    from arc26.loop import DemoSet
    from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter

logging.disable(logging.WARNING)


class UnslothFixedTrainer(UnslothTrainer):
    """``UnslothTrainer`` with the reference's ``compute_loss`` fix (unsloth issue 2435)."""

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(
        self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
    ) -> Any:
        """Fixed compute_loss that handles Unsloth's view tensor issue.

        Why: Unsloth's loss can be a view tensor, which the in-place DDP scaling breaks; the
        loss is cloned first.

        Args:
            model: The model being trained.
            inputs: The collated batch; ``labels`` is popped when a label smoother is set.
            return_outputs: Return the model outputs with the loss.
            **kwargs: Accepted for the trainer's signature, unused.

        Returns:
            The scalar loss tensor, or ``(loss, outputs)`` when ``return_outputs`` is true.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if (
                hasattr(unwrapped_model, "_get_name")
                and "unsloth" in unwrapped_model._get_name().lower()
            ):
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Causal-LM collator that trains on the assistant replies only.

    Labels are -100 everywhere except, in each user and assistant turn pair, the assistant's
    reply tokens from its third token through its ``EOS_ID``: the output grids and their ends.
    """

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:  # noqa: UP007
        """Collate as the parent class does, then mask every label outside the replies.

        Args:
            examples: Tokenized training samples.

        Returns:
            The parent's batch, its ``labels`` [batch, seq] set to -100 outside the replies.
        """
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):  # noqa: B905
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


def fit(
    model: Any,
    default_weights: dict[str, torch.Tensor],
    arc_test_set: ArcDataset,
    key: str,
    formatter: QwenFormatter,
    max_seq_length: int,
    tokenizer: Any,
    collator: QwenDataCollatorForCompletionOnlyLM,
    train_args: dict[str, Any],
    timing_logger: timing.TimingLogger | None,
    rank: int,
) -> tuple[Any, ArcDataset]:
    """Train the LoRA adapter from its default weights on one puzzle and return it for decoding.

    Why: the test-time training block of the reference worker, callable on its own (ADR-011).
    The adapter is reset to ``default_weights``, so no puzzle starts from another's training.
    The training set is the puzzle's demonstration pairs under 128 views (16 color permutations
    of each of the 8 dihedral transforms, ``augment(n=16, shfl_keys=True, seed=1)``; numpy's
    global seed is reset), cut to ``max_seq_length`` tokens; one epoch of
    ``UnslothFixedTrainer`` runs with the trainer's output captured. The peak training memory
    and the trainer's statistics are printed, and the peak is reset for the decode. With a
    logger, ``trainer.train()`` is recorded as ``train`` with the prepared dataset's token count
    (null when the trainer keeps no ``input_ids``).

    Args:
        model: The PEFT-wrapped model from ``arc26.solver.load``, or as the previous call
            returned it.
        default_weights: The adapter's weights right after PEFT wrapping, keyed by PEFT name.
        arc_test_set: The challenges of the run; ``key`` selects the puzzle.
        key: The puzzle's task id.
        formatter: Prompt and reply formatter of the model.
        max_seq_length: Token limit of a training sample (8192).
        tokenizer: The model's tokenizer.
        collator: The completion-only collator.
        train_args: Keyword arguments of ``UnslothTrainingArguments``.
        timing_logger: The worker's timing logger, or None to record nothing.
        rank: Worker rank, used in the printed lines.

    Returns:
        ``(model, puzzle_ds)``: the trained model in inference mode, to be used from here on
        (``for_training``, the trainer's unwrap, and ``for_inference`` may each return a new
        object), and the puzzle as a one-key ``ArcDataset`` for ``decode_puzzle``.
    """
    torch.cuda.reset_peak_memory_stats()

    load_result = set_peft_model_state_dict(  # noqa: F841
        model,
        default_weights.copy(),
        adapter_name="default",
    )

    model = FastLanguageModel.for_training(model)

    puzzle_ds = arc_test_set.change_keys([key])

    train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
    train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):

        trainer = UnslothFixedTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=UnslothTrainingArguments(**train_args),
        )

        train_start = time.perf_counter()
        stats = trainer.train()
        if timing_logger is not None:
            train_seconds = time.perf_counter() - train_start
            try:
                train_tokens = sum(map(len, trainer.train_dataset["input_ids"]))
            except Exception:
                train_tokens = None
            timing_logger.record(key, timing.TRAIN, train_seconds, train_tokens)

        model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

        del trainer

    model = FastLanguageModel.for_inference(model)

    gc.collect()
    torch.cuda.empty_cache()

    memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
    print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

    torch.cuda.reset_peak_memory_stats()

    print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

    return model, puzzle_ds


class SGDProvider:
    r"""B4's ``AdapterProvider`` in Phase 2: ``sgd_continue`` from a snapshot (task P2.3, ADR-015).

    Why: ADR-004's round adapter is :math:`\theta_t = \mathrm{sgd\_continue}(\theta_{t-1}, D_t,
    s')`, a fraction of an epoch on the enlarged set. ``fit`` resets the adapter to the weights
    it is given and then trains; given a copy of the current adapter as ``default_weights`` and
    ``train_args`` with ``max_steps`` :math:`s'`, it continues from :math:`\theta_{t-1}` for
    :math:`s'` of its 128 training views, and no moved statement changes (ADR-011). The model is
    trained in place, so the copy, kept on the CPU (about 1.1 GB in bf16 at the notebook's rank
    256), is also the rollback: ``rollback`` loads it back when the gate fails, and
    ``fingerprint`` hashes the weights so the round's record shows the restore was bit for bit.
    With a logger, ``adapt`` records the copy as ``snapshot`` and the fit's trainer as
    ``candidate_train`` (P2.2's name), ``rollback`` its load as ``restore``, and ``fingerprint``
    its read as ``fingerprint``.

    Attributes:
        key: The puzzle's task id.
        formatter: Prompt and reply formatter of the model.
        max_seq_length: Token limit of a training sample (8192).
        tokenizer: The model's tokenizer.
        collator: The completion-only collator.
        train_args: Keyword arguments of ``UnslothTrainingArguments``; ``max_steps`` is added.
        timing_logger: The worker's timing logger, or None.
        rank: Worker rank, used in the printed lines.
        steps: :math:`s'`, the optimizer steps of a continuation.
    """

    def __init__(
        self,
        *,
        key: str,
        formatter: QwenFormatter,
        max_seq_length: int,
        tokenizer: Any,
        collator: QwenDataCollatorForCompletionOnlyLM,
        train_args: dict[str, Any],
        timing_logger: timing.TimingLogger | None,
        rank: int,
        steps: int,
    ) -> None:
        """Build the provider of one puzzle from the worker's locals.

        Args:
            key: The puzzle's task id.
            formatter: Prompt and reply formatter of the model.
            max_seq_length: Token limit of a training sample.
            tokenizer: The model's tokenizer.
            collator: The completion-only collator.
            train_args: Keyword arguments of ``UnslothTrainingArguments``.
            timing_logger: The worker's timing logger, or None.
            rank: Worker rank.
            steps: :math:`s'`, at least 1.

        Raises:
            TypeError: If ``steps`` is not an integer.
            ValueError: If ``steps`` is under 1.
        """
        if type(steps) is not int:
            raise TypeError(f"steps must be an integer; got {steps!r}")
        if steps < 1:
            raise ValueError(f"steps must be at least 1; got {steps}")
        self.key = key
        self.formatter = formatter
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.collator = collator
        self.train_args = dict(train_args)
        self.timing_logger = timing_logger
        self.rank = rank
        self.steps = steps
        # The adapter's weights before the last adapt, on the CPU; and the model object that the
        # last fit returned, which holds the live weights.
        self._snapshot: dict[str, torch.Tensor] | None = None
        self._live: Any = None

    def _record(self, op: str, start: float) -> None:
        if self.timing_logger is not None:
            self.timing_logger.record(self.key, op, time.perf_counter() - start)

    def adapt(self, demos: DemoSet, init: Any, budget_s: float) -> Any:
        """Continue the adapter on ``demos`` for :math:`s'` steps, from a snapshot of ``init``.

        Args:
            demos: The enlarged state without its held-out pair; ``fit`` trains on
                ``demos.task()``, its pairs under the 128 views.
            init: The model holding the current adapter, as the last fit or rollback left it.
            budget_s: Seconds left to the puzzle's deadline; the governor priced the
                :math:`s'` steps within it, so it is printed, not enforced.

        Returns:
            The model in inference mode with the candidate adapter, as ``fit`` returns it.
        """
        from peft import get_peft_model_state_dict

        from arc26.loop import CANDIDATE_OPS, SNAPSHOT, RenamedOps
        from arc26.nvarc.arc_loader import ArcDataset

        # An earlier round's copy must never be restored onto this round's adapter: a copy that
        # raises leaves no snapshot, and the rollback then has nothing to load.
        self._snapshot = None
        self._live = init
        start = time.perf_counter()
        weights = get_peft_model_state_dict(init, adapter_name="default")
        self._snapshot = {
            name: tensor.detach().to("cpu", copy=True) for name, tensor in weights.items()
        }
        self._record(SNAPSHOT, start)
        print(
            f"[Rank {self.rank}] continuing the adapter of {self.key} for {self.steps} steps on "
            f"{len(demos.pairs)} pairs, {budget_s:.0f} s left"
        )
        model, _ = fit(
            model=init,
            default_weights=self._snapshot,
            arc_test_set=ArcDataset(queries={self.key: demos.task()}, replies={}),
            key=self.key,
            formatter=self.formatter,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            collator=self.collator,
            train_args={**self.train_args, "max_steps": self.steps},
            timing_logger=(
                None
                if self.timing_logger is None
                else RenamedOps(self.timing_logger, CANDIDATE_OPS)
            ),
            rank=self.rank,
        )
        self._live = model
        return model

    def rollback(self, previous: Any) -> Any:
        """Load the snapshot back into the model: the adapter before the last ``adapt``.

        Args:
            previous: The model ``adapt`` started from, returned unchanged when the last
                ``adapt`` took no snapshot (its copy raised, or none ran).

        Returns:
            The model in inference mode with the previous adapter's weights.
        """
        from arc26.loop import RESTORE

        if self._snapshot is None:
            return previous
        start = time.perf_counter()
        model = self._live if self._live is not None else previous
        set_peft_model_state_dict(model, dict(self._snapshot), adapter_name="default")
        # A fit that raised may leave the model in training mode and its trainer's memory held.
        model = FastLanguageModel.for_inference(model)
        gc.collect()
        torch.cuda.empty_cache()
        self._live = model
        self._record(RESTORE, start)
        return model

    def fingerprint(self, adapter: Any) -> str:
        """Hash the adapter's weights, so a restore can be shown bit for bit.

        Args:
            adapter: The model holding the adapter.

        Returns:
            The blake2b digest (16 bytes, hex) of every adapter tensor in name order: its name,
            dtype, shape, and raw bytes, read one tensor at a time on the CPU.
        """
        import hashlib

        from peft import get_peft_model_state_dict

        from arc26.loop import FINGERPRINT

        start = time.perf_counter()
        weights = get_peft_model_state_dict(adapter, adapter_name="default")
        digest = hashlib.blake2b(digest_size=16)
        for name in sorted(weights):
            tensor = weights[name].detach().to("cpu").contiguous()
            digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
            digest.update(tensor.view(torch.uint8).numpy())
        self._record(FINGERPRINT, start)
        return digest.hexdigest()

    def fit_shape(self, demos: DemoSet) -> tuple[int, int]:
        """Count what ``adapt`` would train on ``demos``, for the governor's price.

        Why: the governor's train model is per token of the trained samples and per optimizer
        step (``CostModel``, fitted on 128-step fits). The 128 views are built as ``fit`` builds
        them, and tokenized; :math:`s'` steps at batch 1 train on :math:`s'` of them, so the
        fit's tokens are their total times :math:`s'/128`. numpy's random state is kept.

        Args:
            demos: The state ``adapt`` would train on, without its held-out pair.

        Returns:
            ``(tokens, steps)``: the tokens of :math:`s'` average views, rounded up, and
            :math:`s'` (at most the views).
        """
        import math

        from arc26.nvarc.arc_loader import ArcDataset

        state = np.random.get_state()
        try:
            dataset = ArcDataset(queries={self.key: demos.task()}, replies={})
            views = dataset.change_keys([self.key]).augment(n=16, shfl_keys=True, seed=1)
            views = views.cut_to_len(
                formatter=self.formatter, name="text", max_len=self.max_seq_length
            )
            texts = [row["text"] for row in views.as_list(self.formatter)]
        finally:
            np.random.set_state(state)
        if not texts:
            return 0, self.steps
        tokens = sum(len(self.tokenizer.encode(text)) for text in texts)
        steps = min(self.steps, len(texts))
        return math.ceil(tokens * steps / len(texts)), steps
