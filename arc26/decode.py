"""DFS decoding, augmented rescoring, and candidate files of the split solver (task P2.0).

The reference's per-puzzle decode, moved out of ``arc_solver.worker`` so that it can be called on
its own (ADR-011): ``decode_puzzle`` decodes every test input of a puzzle under 16 augmented views
(2 color permutations of each of the 8 dihedral transforms), four views per batch, with
``inference_turbo_dfs``; it rescores each new candidate grid under 8 augmented views with
``calc_scores`` and writes one bz2 pickle of candidates per view. ``arc26.solver.worker`` calls it
after ``arc26.ttt.fit``. The statements are the reference's, unchanged, with their timing records
(``decode``, ``selection``; P0.3b). This module imports torch and unsloth, which only the Kaggle
image provides. See ``docs/modules/decode.md``.
"""

# Moved from reference/reference_notebook.ipynb, cell 4 (%%writefile arc_solver.py), through the
# generated arc26/nvarc/arc_solver.py (P0.1; P0.3b timing records; P1.3 LoRA settings):
#   ARC_VOCAB to EOS_ID: lines 38-57;
#   turbo_dfs: lines 108-175; inference_turbo_dfs: lines 178-197; calc_scores: lines 200-228;
#   decode_puzzle: lines 384-495, the decode block of the worker's per-puzzle loop;
#   imports: those of lines 10-34 that this code reads, unsloth first.
# Moved, not rewritten: tests/test_split_solver.py checks every statement against
# arc_solver.py on the AST and lists the intended differences (signatures, docstrings, returns).
# Only the layout changed: long lines wrapped, trailing whitespace removed.
# Task P2.2 adds to decode_puzzle one keyword-only parameter, view_seed (default None), for the
# promotion's re-decode under fresh views (ADR-014): the seeds of its two augment calls read
# "<the reference's seed> if view_seed is None else <a seed from view_seed>", so the default runs
# the reference's statements.
# Task P3.7 adds a second keyword-only parameter, views (default None), for decoding a subset of
# the 16 views (ADR-019): the batch loop reads "batches if views is None else <the views' batches
# per test input>", so the default runs the reference's statements too.
# Origin: the NVARC team's ARC Prize 2025 (ARC-AGI-2) first-place notebook,
#   Kaggle sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue
# License: Apache License 2.0; attribution in reference/README.md

from __future__ import annotations

# First, as in arc_solver.py: unsloth patches transformers, which arc_loader imports, so it must
# be imported before it, whichever of arc26.decode and arc26.ttt a process imports first.
import unsloth  # noqa: F401

# isort: split
import bz2
import os
import pickle
import time
from collections import defaultdict
from typing import Any

import torch

from arc26 import timing
from arc26.nvarc.arc_loader import ArcDataset, QwenFormatter

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


def turbo_dfs(
    model: Any,
    logits: torch.Tensor,  # [n, vocab]
    max_new_tokens: int,
    max_score: float,
    scores: list[float],
    pos: int,
    cache: Any,
    start_time: float,
    end_time: float,
) -> dict[int, list[tuple[float, list[int]]]]:
    """Extend ``n`` decoding rows depth first, keeping every continuation under ``max_score``.

    Why: the reference's decoder. A row's candidates are the tokens of ``ARC_TOKENS`` whose
    cumulative negative log-likelihood (NLL) stays below ``max_score``; ``EOS_ID`` ends a
    suffix. Each step feeds the best untried candidate of every row (``PAD_ID`` with NLL 1000 for
    a row with none left) as one batch through the cache, recurses one position deeper, and
    repeats until no row has a candidate, 540 s have passed since ``start_time``, or
    ``end_time`` is reached.

    Args:
        model: The PEFT-wrapped Unsloth model in inference mode.
        logits: Next-token logits at the last position of each row.  # [n, vocab]
        max_new_tokens: Tokens the suffix may still take, ``EOS_ID`` included.
        max_score: NLL bound; the solver passes -ln(0.2), about 1.61 (probability 0.2).
        scores: Cumulative NLL of each row so far.  # [n]
        pos: Position id of the next token.
        cache: The model's ``past_key_values`` for the rows.
        start_time: ``time.time()`` when the search started.
        end_time: Absolute deadline, ``time.time()`` scale.

    Returns:
        Per row index, the finished suffixes as ``(nll, tokens)`` pairs, ``tokens`` ending with
        ``EOS_ID``.
    """
    n = logits.size(0)

    nll = (
        torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    )

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]

    while time.time() - start_time < 540 and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(
    model: Any,
    prefix_tokens: list[list[int]],  # [n, prompt_len]
    max_new_tokens: int,
    max_score: float,
    end_time: float,
) -> list[tuple[int, list[tuple[float, list[int]]]]]:
    """Decode ``n`` prompts of equal length with ``turbo_dfs``; sort each row's beams by NLL.

    Why: the reference's batched decode call. One prefill pass over the prompts gives the first
    logits and the cache, then ``turbo_dfs`` searches from them with a fresh 540 s budget.

    Args:
        model: The PEFT-wrapped Unsloth model in inference mode.
        prefix_tokens: Token ids of the ``n`` prompts, all of one length.  # [n, prompt_len]
        max_new_tokens: Longest suffix, ``EOS_ID`` included (``formatter.max_new_tokens()``).
        max_score: NLL bound of ``turbo_dfs``.
        end_time: Absolute deadline, ``time.time()`` scale.

    Returns:
        ``(row, beams)`` per row that finished a suffix, ``beams`` the ``(nll, tokens)`` pairs in
        ascending NLL.
    """
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries: list[str], answers: list[str], tokenizer: Any, model: Any) -> list[float]:
    """Return the NLL of each answer given its query, in one batched forward pass.

    Why: the reference rescores a decoded grid under augmented views of its puzzle; the
    selection (``arc_decoder``) combines these scores. Rows are right-padded with ``PAD_ID``,
    which the causal model cannot see from the answer positions.

    Args:
        queries: Formatted prompts, one per row.
        answers: Formatted replies, one per row.
        tokenizer: The model's tokenizer.
        model: The PEFT-wrapped Unsloth model.

    Returns:
        Per row, the answer's NLL: minus the sum over its tokens of the log-probability of each
        token given the query and the answer tokens before it.
    """
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):  # noqa: B905
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(  # noqa: B905
        batch_logits, batch_query_tokens, batch_answer_tokens
    ):
        query_length = len(query_tokens)
        answer_logits = logits[query_length-1:query_length-1+len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


def decode_puzzle(
    puzzle_ds: ArcDataset,
    formatter: QwenFormatter,
    max_seq_length: int,
    max_new_tokens: int,
    start_time: float,
    end_time: float,
    rank: int,
    key: str,
    tokenizer: Any,
    model: Any,
    max_score: float,
    timing_logger: timing.TimingLogger | None,
    dir_outputs: str,
    *,
    view_seed: int | None = None,
    views: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    """Decode every test input of one puzzle and write each view's rescored candidates.

    Why: the decode block of the reference worker, callable on its own (ADR-011). Each test input
    becomes 16 augmented views (``augment(n=2, seed=2)``), cut to fit ``max_seq_length`` with
    room for the reply. The views are decoded four at a time, in the reference's order: per test
    input the identity and 180 degree views, then the 90 and 270 degree ones; then the same for
    the transposed views (each batch holds prompts of one length). A parsed grid is mapped back
    by the view's inverse transform; a new grid is rescored with ``calc_scores`` under the 8
    views of ``augment(seed=hash(bk) % 1024**2)`` (``PYTHONHASHSEED`` fixes the seed), and a
    repeated one reuses its scores. Decoding stops at 1200 s after ``start_time`` or at
    ``end_time``. With a logger, each DFS call is recorded as ``decode`` (prompt tokens) and each
    rescoring as ``selection`` (query and answer tokens), as in P0.3b. ``view_seed`` (task P2.2,
    ADR-014) picks other views for the promotion's re-decode: ``augment(n=2, seed=view_seed)``
    and rescoring under ``augment(seed=hash((bk, view_seed)) % 1024**2)``, with the same batch
    order; ``arc26.loop.view_keys`` computes them, and ``arc26.loop.promote`` checks that they
    share no view with the decode that produced its candidates. ``views`` (task P3.7, ADR-019)
    decodes a subset of each test input's 16 views: its batches, each a tuple of indices into the
    input's view keys in sorted order (``2 d + c`` for dihedral map ``d`` of
    ``arc26.arms.DIHEDRAL_ORDER`` and color permutation ``c``), all of one prompt length
    (``arc26.arms.check_views``), decoded test input by test input; rescoring is unchanged.

    Args:
        puzzle_ds: The puzzle as ``fit`` returns it: one key, unaugmented.
        formatter: Prompt and reply formatter of the model.
        max_seq_length: Token limit of a prompt plus its reply (8192).
        max_new_tokens: Longest reply in tokens, ``EOS_ID`` included.
        start_time: ``time.time()`` when the puzzle started.
        end_time: Absolute deadline of the run, ``time.time()`` scale.
        rank: Worker rank, used in the printed lines.
        key: The puzzle's task id.
        tokenizer: The model's tokenizer.
        model: The model after ``fit``, in inference mode.
        max_score: NLL bound of the DFS, -ln(0.2).
        timing_logger: The worker's timing logger, or None to record nothing.
        dir_outputs: Folder of the candidate files: one bz2 pickle per view that yielded a grid,
            named by the view's key, holding a list of ``{"beam_score", "score_aug",
            "solution"}``.
        view_seed: None (the default) decodes and rescores under the reference's views; an
            integer under the views it seeds, as above.
        views: None (the default) decodes every view in the reference's batches; else the view
            indices of each batch, per test input, as above.
    """
    puzzle_ds_multi = puzzle_ds.split_multi_replies()

    eval_ds = puzzle_ds_multi.augment(n=2, seed=2 if view_seed is None else view_seed)
    eval_ds = eval_ds.cut_to_len(
        formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens
    )

    test_id_to_subkeys = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        test_id_to_subkeys[test_id].append(subkey)

    batches = []
    for test_id, subkeys in test_id_to_subkeys.items():  # noqa: B007
        # 0: permute x 2
        # 4: rot90.rot90.permute x 2
        batch = []
        for offset in [0, 4]:
            batch.extend(subkeys[offset:offset+2])
        batches.append(batch)
        # 2: permute.rot90 x 2
        # 6: rot90.rot90.rot90.permute x 2
        batch = []
        for offset in [2, 6]:
            batch.extend(subkeys[offset:offset+2])
        batches.append(batch)
    for test_id, subkeys in test_id_to_subkeys.items():  # noqa: B007
        # 8: transpose.permute x 2
        # 12: transpose.rot90.rot90.permute x 2
        batch = []
        for offset in [8, 12]:
            batch.extend(subkeys[offset:offset+2])
        batches.append(batch)
        # 10: transpose.rot90.permute x 2
        # 14: transpose.rot90.rot90.rot90.permute x 2
        batch = []
        for offset in [10, 14]:
            batch.extend(subkeys[offset:offset+2])
        batches.append(batch)

    with torch.inference_mode():

        known_scores = {}

        for subkeys in (
            batches
            if views is None
            else [
                [keys[i] for i in batch]
                for keys in test_id_to_subkeys.values()
                for batch in views
            ]
        ):

            spend_time = time.time() - start_time
            if spend_time > 1200 or time.time() > end_time:
                print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                break

            print(f"[Rank {rank}] decoding {subkeys}")

            tokens = []
            for subkey in subkeys:
                data = eval_ds.get(subkey, formatter)
                tokens.append(tokenizer.encode(data["input"]))

            decode_start = time.perf_counter()
            dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)
            if timing_logger is not None:
                timing_logger.record(
                    key, timing.DECODE, time.perf_counter() - decode_start, sum(map(len, tokens))
                )

            for subkey_id, scored_beams in dfs_result:

                subkey = subkeys[subkey_id]
                bk = subkey.split(".")[0]
                decoded_result = []

                for beam_score, tokens in scored_beams:

                    array = formatter.convert_tokens_to_array(tokens)
                    if array is None:
                        continue

                    solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                    grid_id = (bk, tuple(map(tuple, solution)))

                    if grid_id in known_scores:
                        augmented_scores = known_scores[grid_id]
                    else:
                        print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                        aug_dataset = ArcDataset(
                            keys=[bk],
                            queries={bk: puzzle_ds_multi.queries.get(bk)},
                            replies={bk: [solution.tolist()]},
                        )
                        aug_dataset = aug_dataset.augment(
                            seed=hash(bk) % 1024**2
                            if view_seed is None
                            else hash((bk, view_seed)) % 1024**2
                        )
                        aug_dataset = aug_dataset.cut_to_len(
                            formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens
                        )
                        aug_queries = []
                        aug_answers = []
                        for augmented_sample in aug_dataset.as_list(formatter):
                            aug_queries.append(augmented_sample["input"])
                            aug_answers.append(augmented_sample["reply"])
                        selection_start = time.perf_counter()
                        augmented_scores1 = calc_scores(
                            aug_queries[:4], aug_answers[:4], tokenizer, model
                        )
                        augmented_scores2 = calc_scores(
                            aug_queries[4:], aug_answers[4:], tokenizer, model
                        )
                        if timing_logger is not None:
                            selection_seconds = time.perf_counter() - selection_start
                            selection_tokens = sum(
                                len(tokenizer.encode(text)) for text in aug_queries + aug_answers
                            )
                            timing_logger.record(
                                key, timing.SELECTION, selection_seconds, selection_tokens
                            )
                        augmented_scores = augmented_scores1 + augmented_scores2
                        known_scores[grid_id] = augmented_scores

                    decoded_result.append({
                        "beam_score": beam_score,
                        "score_aug": augmented_scores,
                        "solution": solution,
                    })

                if len(decoded_result):
                    with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                        pickle.dump(decoded_result, f)
