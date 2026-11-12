"""Model load, per-puzzle worker, and process entry of the split solver (task P2.0).

``python -m arc26.solver --end-time T --config JSON`` runs the reference's solver with its worker
split into ``load``, ``arc26.ttt.fit``, and ``arc26.decode.decode_puzzle`` (ADR-011), so that the
refinement loop can call training and decoding separately. It is equivalent to
``python -m arc26.nvarc.starter``: the same arguments, task queue, worker markers, and
``torch.multiprocessing.spawn`` with one worker per GPU; ``arc26.runner`` starts it when
``RunConfig.solver`` is ``"split"``. The statements are the reference's, unchanged, with their
timing records (``load``, ``puzzle``; P0.3b). Importing this module imports torch only: a worker
process imports unsloth and the rest of the GPU stack in ``local_worker``, after it is pinned to
its GPU and the previous rank has finished the same import, as in the reference. With the
config's optional ``holdout`` (task P2.1), each puzzle runs through ``arc26.loop.solve_puzzle``:
the same ``fit`` and ``decode_puzzle`` with one demonstration pair held out, then the held-out
gate, and with ``holdout.rounds`` (task P2.3) the refinement rounds. See
``docs/modules/solver.md``.
"""

# Moved from reference/reference_notebook.ipynb through the generated arc26/nvarc/ modules
# (P0.1; P0.3b timing records; P1.3 LoRA settings):
#   load: cell 4 (%%writefile arc_solver.py), arc26/nvarc/arc_solver.py lines 233-316, the
#     start of worker, after its own imports: those of lines 10-34 that it reads, unsloth first;
#   worker: cell 4, arc_solver.py lines 318-329 and 497-503, the per-puzzle loop around
#     arc26.ttt.fit (lines 331-382) and arc26.decode.decode_puzzle (lines 384-495);
#   local_worker and the __main__ block: cell 5 (%%writefile starter.py),
#     arc26/nvarc/starter.py lines 20-42 and 45-73; the module's imports, starter.py lines 10-17.
# Moved, not rewritten: tests/test_split_solver.py checks every statement against arc_solver.py
# and starter.py on the AST and lists the intended differences (signatures, docstrings, imports,
# returns, and the calls of fit and decode_puzzle). Task P2.1 adds to worker the hold-out
# settings and a branch around the two calls: without settings they run as moved, with them
# arc26.loop.solve_puzzle runs instead. Task P2.3 passes it the queue size and the worker count,
# the governor's inputs for the refinement rounds.
# Only the layout changed: trailing whitespace removed.
# Origin: the NVARC team's ARC Prize 2025 (ARC-AGI-2) first-place notebook,
#   Kaggle sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue
# License: Apache License 2.0; attribution in reference/README.md

from __future__ import annotations

import argparse
import json
import os
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.multiprocessing as mp

from arc26 import timing

if TYPE_CHECKING:
    from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter
    from arc26.ttt import QwenDataCollatorForCompletionOnlyLM


class Loaded(NamedTuple):
    """What ``load`` returns: the model and everything a worker's puzzles share.

    Attributes:
        train_args: Keyword arguments of ``UnslothTrainingArguments``: one epoch, batch 1,
            learning rate 5e-5 on a cosine schedule with 10% warmup, bf16, seed 42.
        max_seq_length: Token limit of a sample, 8192.
        timing_logger: The process's timing logger (``arc26.timing.get_logger``), or None.
        model: The base model with its LoRA adapter (PEFT), float32 parameters cast to bf16.
        tokenizer: The model's tokenizer.
        default_weights: Copies of the adapter's weights right after wrapping, keyed by PEFT
            name; ``arc26.ttt.fit`` resets the adapter to them before every puzzle.
        collator: The completion-only collator of the training samples.
        formatter: Prompt and reply formatter of the model.
        max_new_tokens: Longest reply in tokens: a 30 by 30 grid and its end.
        max_score: NLL bound of the DFS, -ln(0.2).
        arc_test_set: The challenges: the test set in a competition rerun, the evaluation set
            otherwise.
        dir_outputs: Folder of the candidate files, created by ``load``.
    """

    train_args: dict[str, Any]
    max_seq_length: int
    timing_logger: timing.TimingLogger | None
    model: Any
    tokenizer: Any
    default_weights: dict[str, torch.Tensor]
    collator: QwenDataCollatorForCompletionOnlyLM
    formatter: QwenFormatter
    max_new_tokens: int
    max_score: float
    arc_test_set: ArcDataset
    dir_outputs: str


def load(cfg: dict[str, Any]) -> Loaded:
    """Load the model with its LoRA adapter and build what every puzzle of a worker shares.

    Why: the start of the reference worker, before its per-puzzle loop. It loads the base model
    from ``cfg["model_dir"]`` (bf16, no 4-bit, local files only), wraps it with a rank
    ``cfg["lora_r"]`` rsLoRA adapter (alpha 32) on ``cfg["lora_targets"]``, casts float32
    parameters to bf16, and keeps a copy of the adapter's weights; with a logger, these steps
    are recorded as ``load``. Then it builds the collator and formatter, reads the challenges
    (the test set when ``KAGGLE_IS_COMPETITION_RERUN`` is set), and creates the candidate folder.

    Args:
        cfg: The solver config, the ``--config`` JSON of ``arc26.runner.solver_command``; this
            reads ``model_dir``, ``lora_r``, ``lora_targets``, ``data_dir``, and ``outputs_dir``.

    Returns:
        The model, its settings, and the challenges, as ``Loaded``.
    """
    # The GPU stack, which local_worker has already imported when a worker runs this; unsloth
    # first all the same, before peft and transformers, as in arc_solver.py, for any other caller.
    from unsloth import FastLanguageModel

    # isort: split
    import numpy as np
    from peft import get_peft_model_state_dict

    from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter
    from arc26.ttt import QwenDataCollatorForCompletionOnlyLM

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    peft_params = dict(
        r=cfg["lora_r"],
        target_modules=list(cfg["lora_targets"]),
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    timing_logger = timing.get_logger()
    load_start = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_dir"],
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():  # noqa: B007
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}
    if timing_logger is not None:
        timing_logger.record(None, "load", time.perf_counter() - load_start)

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    max_score = -np.log(0.2)

    if rerun_mode:
        test_path = os.path.join(cfg["data_dir"], "arc-agi_test_challenges.json")
    else:
        test_path = os.path.join(cfg["data_dir"], "arc-agi_evaluation_challenges.json")

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = cfg["outputs_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    return Loaded(
        train_args=train_args,
        max_seq_length=max_seq_length,
        timing_logger=timing_logger,
        model=model,
        tokenizer=tokenizer,
        default_weights=default_weights,
        collator=collator,
        formatter=formatter,
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        arc_test_set=arc_test_set,
        dir_outputs=dir_outputs,
    )


def worker(rank: int, queue: Any, end_time: float, cfg: dict[str, Any]) -> None:
    """Solve puzzles from the queue: ``fit``, then ``decode_puzzle``, one puzzle at a time.

    Why: the reference worker, with its per-puzzle loop body split into ``arc26.ttt.fit`` and
    ``arc26.decode.decode_puzzle``, called in the reference's order with the reference's
    locals. It stops when the queue is empty, when it takes a ``None``, or when ``end_time`` has
    passed before a puzzle starts. After each puzzle it prints the peak inference memory and the
    puzzle's wall time and, with a logger, records the puzzle's span as ``puzzle``. When the
    config holds ``holdout`` (task P2.1), each puzzle runs through ``arc26.loop.solve_puzzle``
    instead: the same two calls with one demonstration pair held out, then the held-out gate,
    and with ``holdout.rounds`` the refinement rounds (task P2.3), for which it also gets the
    queue's size after the puzzle was taken and the worker count (``Governor.puzzle_budget``).

    Args:
        rank: Worker rank, used in the printed lines.
        queue: The shared task queue: task ids, then one ``None`` per worker.
        end_time: Absolute deadline, ``time.time()`` scale: no puzzle starts after it, and
            decoding stops at it.
        cfg: The solver config (see ``load``); its optional ``holdout`` holds the settings of
            ``arc26.loop.HoldOut``.
    """
    from arc26.decode import decode_puzzle
    from arc26.loop import HoldOut, solve_puzzle
    from arc26.ttt import fit

    (
        train_args,
        max_seq_length,
        timing_logger,
        model,
        tokenizer,
        default_weights,
        collator,
        formatter,
        max_new_tokens,
        max_score,
        arc_test_set,
        dir_outputs,
    ) = load(cfg)

    # Task P2.1: None runs the reference's per-puzzle body; settings run solve_puzzle.
    holdout = HoldOut.from_cfg(cfg)

    while not queue.empty():

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()
        puzzle_start = time.perf_counter()

        if holdout is None:
            model, puzzle_ds = fit(
                model=model,
                default_weights=default_weights,
                arc_test_set=arc_test_set,
                key=key,
                formatter=formatter,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                collator=collator,
                train_args=train_args,
                timing_logger=timing_logger,
                rank=rank,
            )

            decode_puzzle(
                puzzle_ds=puzzle_ds,
                formatter=formatter,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                start_time=start_time,
                end_time=end_time,
                rank=rank,
                key=key,
                tokenizer=tokenizer,
                model=model,
                max_score=max_score,
                timing_logger=timing_logger,
                dir_outputs=dir_outputs,
            )
        else:
            model = solve_puzzle(
                holdout=holdout,
                model=model,
                default_weights=default_weights,
                arc_test_set=arc_test_set,
                key=key,
                formatter=formatter,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                collator=collator,
                train_args=train_args,
                timing_logger=timing_logger,
                rank=rank,
                max_new_tokens=max_new_tokens,
                start_time=start_time,
                end_time=end_time,
                max_score=max_score,
                dir_outputs=dir_outputs,
                queue_size=queue.qsize(),
                workers=cfg["workers"],
            )

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")

        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
        if timing_logger is not None:
            timing_logger.record(key, timing.PUZZLE, time.perf_counter() - puzzle_start)


def local_worker(rank: int, queue: Any, end_time: float, cfg: dict[str, Any]) -> None:
    """Run one spawned worker process: pin it to GPU ``rank``, import the GPU stack, solve.

    Why: the reference's process body (``starter.local_worker``). The process sees only GPU
    ``rank``. Rank ``r > 0`` waits for the marker file ``worker{r-1}`` before importing the GPU
    stack (the reference's workaround for unsloth's patching), and writes its own marker once
    the import is done, so the imports run one rank after another. With ``cfg["timing_dir"]``
    set, the process's timing logger writes ``timing-rank<rank>.jsonl`` there.

    Args:
        rank: Worker rank, which is also the GPU index.
        queue: The shared task queue.
        end_time: Absolute deadline, ``time.time()`` scale.
        cfg: The solver config; this reads ``marker_dir`` and ``timing_dir``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(os.path.join(cfg["marker_dir"], f"worker{rank-1}")):
            time.sleep(5)

    from arc26 import decode, ttt  # noqa: F401

    with open(os.path.join(cfg["marker_dir"], f"worker{rank}"), "w") as f:
        f.write("Ok")

    if cfg["timing_dir"] is not None:
        timing.configure(cfg["timing_dir"], f"rank{rank}")
    print(f"[Rank {rank}] start!")

    worker(rank, queue, end_time, cfg)

    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config)

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = os.path.join(cfg["data_dir"], "arc-agi_test_challenges.json")
    else:
        test_path = os.path.join(cfg["data_dir"], "arc-agi_evaluation_challenges.json")

    with open(test_path, "r") as f:  # noqa: UP015
        data = json.load(f)

    queue = mp.Manager().Queue()

    for key in sorted(data.keys()):
        if not rerun_mode:
            if cfg["subset"] is not None and key not in cfg["subset"]:
                continue
        queue.put(key)
    for _ in range(cfg["workers"]):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, args.end_time, cfg), nprocs=cfg["workers"])
